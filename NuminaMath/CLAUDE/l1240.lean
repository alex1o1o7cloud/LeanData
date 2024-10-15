import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l1240_124015

theorem equation_solutions :
  (∃ x : ℚ, (1/2) * x - 3 = 2 * x + 1/2 ∧ x = -7/3) ∧
  (∃ x : ℚ, (x-3)/2 - (2*x+1)/3 = 1 ∧ x = -17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1240_124015


namespace NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l1240_124044

/-- Given a right triangle with legs of lengths 6 and 8, 
    the length of the median on the hypotenuse is 5. -/
theorem right_triangle_median_on_hypotenuse : 
  ∀ (a b c m : ℝ), 
    a = 6 → 
    b = 8 → 
    c^2 = a^2 + b^2 → 
    m = c / 2 → 
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l1240_124044


namespace NUMINAMATH_CALUDE_complex_number_properties_l1240_124059

theorem complex_number_properties (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 1 + 2*I) (hz₂ : z₂ = 3 - 4*I) : 
  (Complex.im (z₁ * z₂) = 2) ∧ 
  (Complex.re (z₁ * z₂) > 0 ∧ Complex.im (z₁ * z₂) > 0) ∧
  (Complex.re z₁ > 0 ∧ Complex.im z₁ > 0) := by
  sorry


end NUMINAMATH_CALUDE_complex_number_properties_l1240_124059


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1240_124024

theorem quadratic_root_problem (p : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + p * x - 8 = 0 ∧ x = 2 + Complex.I) → p = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1240_124024


namespace NUMINAMATH_CALUDE_daisy_percentage_is_62_l1240_124070

/-- Represents the composition of flowers in a garden -/
structure Garden where
  total : ℝ
  yellow_ratio : ℝ
  yellow_tulip_ratio : ℝ
  red_daisy_ratio : ℝ

/-- The percentage of daisies in the garden -/
def daisy_percentage (g : Garden) : ℝ :=
  ((g.yellow_ratio - g.yellow_ratio * g.yellow_tulip_ratio) + 
   ((1 - g.yellow_ratio) * g.red_daisy_ratio)) * 100

/-- Theorem stating that the percentage of daisies in the garden is 62% -/
theorem daisy_percentage_is_62 (g : Garden) 
  (h1 : g.yellow_tulip_ratio = 1/5)
  (h2 : g.red_daisy_ratio = 1/2)
  (h3 : g.yellow_ratio = 4/10) :
  daisy_percentage g = 62 := by
  sorry

#eval daisy_percentage { total := 100, yellow_ratio := 0.4, yellow_tulip_ratio := 0.2, red_daisy_ratio := 0.5 }

end NUMINAMATH_CALUDE_daisy_percentage_is_62_l1240_124070


namespace NUMINAMATH_CALUDE_orthogonal_projection_area_range_l1240_124057

/-- Regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  base_side : ℝ
  lateral_edge : ℝ

/-- Orthogonal projection area of a regular quadrangular pyramid -/
def orthogonal_projection_area (p : RegularQuadrangularPyramid) (angle : ℝ) : ℝ :=
  sorry

/-- Theorem: Range of orthogonal projection area -/
theorem orthogonal_projection_area_range 
  (p : RegularQuadrangularPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_edge = Real.sqrt 6) : 
  ∀ angle, 2 ≤ orthogonal_projection_area p angle ∧ 
           orthogonal_projection_area p angle ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_orthogonal_projection_area_range_l1240_124057


namespace NUMINAMATH_CALUDE_interest_calculation_l1240_124005

/-- Problem Statement: A sum is divided into two parts with specific interest conditions. -/
theorem interest_calculation (total_sum second_sum : ℕ) 
  (first_rate second_rate : ℚ) (first_years : ℕ) : 
  total_sum = 2795 →
  second_sum = 1720 →
  first_rate = 3/100 →
  second_rate = 5/100 →
  first_years = 8 →
  ∃ (second_years : ℕ),
    (total_sum - second_sum) * first_rate * first_years = 
    second_sum * second_rate * second_years ∧
    second_years = 3 := by
  sorry


end NUMINAMATH_CALUDE_interest_calculation_l1240_124005


namespace NUMINAMATH_CALUDE_sequence_properties_l1240_124091

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem statement
theorem sequence_properties :
  (a 1 = -28) ∧
  (∀ n : ℕ, a n = S n - S (n-1)) ∧
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 4) :=
by sorry

-- The fact that the sequence is arithmetic follows from the third conjunct
-- of the theorem above, as the difference between consecutive terms is constant.

end NUMINAMATH_CALUDE_sequence_properties_l1240_124091


namespace NUMINAMATH_CALUDE_sixth_episode_length_is_115_l1240_124055

/-- The length of the sixth episode in a series of six episodes -/
def sixth_episode_length (ep1 ep2 ep3 ep4 ep5 total : ℕ) : ℕ :=
  total - (ep1 + ep2 + ep3 + ep4 + ep5)

/-- Theorem stating the length of the sixth episode -/
theorem sixth_episode_length_is_115 :
  sixth_episode_length 58 62 65 71 79 450 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sixth_episode_length_is_115_l1240_124055


namespace NUMINAMATH_CALUDE_subset_property_j_bound_l1240_124036

variable (m n : ℕ+)
variable (A : Finset ℕ)
variable (B : Finset ℕ)
variable (S : Finset (ℕ × ℕ))

def setA : Finset ℕ := Finset.range n
def setB : Finset ℕ := Finset.range m

def property_j (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ (a b x y : ℕ), (a, b) ∈ S → (x, y) ∈ S → (a - x) * (b - y) ≤ 0

theorem subset_property_j_bound :
  A = setA m → B = setB n → S ⊆ A ×ˢ B → property_j S → S.card ≤ m + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_property_j_bound_l1240_124036


namespace NUMINAMATH_CALUDE_water_dispenser_capacity_l1240_124075

/-- A cylindrical water dispenser with capacity x liters -/
structure WaterDispenser where
  capacity : ℝ
  cylindrical : Bool

/-- The water dispenser contains 60 liters when it is 25% full -/
def quarter_full (d : WaterDispenser) : Prop :=
  0.25 * d.capacity = 60

/-- Theorem: A cylindrical water dispenser that contains 60 liters when 25% full has a total capacity of 240 liters -/
theorem water_dispenser_capacity (d : WaterDispenser) 
  (h1 : d.cylindrical = true) 
  (h2 : quarter_full d) : 
  d.capacity = 240 := by
  sorry

end NUMINAMATH_CALUDE_water_dispenser_capacity_l1240_124075


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_l1240_124051

/-- The count of three-digit numbers. -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical non-adjacent digits. -/
def excluded_numbers : ℕ := 81

/-- The count of valid three-digit numbers after exclusion. -/
def valid_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers :
  valid_numbers = 819 :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_l1240_124051


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l1240_124099

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem ninth_term_of_sequence (a d : ℝ) :
  arithmetic_sequence a d 3 = 20 →
  arithmetic_sequence a d 6 = 26 →
  arithmetic_sequence a d 9 = 32 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l1240_124099


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1240_124041

theorem complex_equation_solution : 
  ∃ z : ℂ, z * (1 - Complex.I) = 2 * Complex.I ∧ z = 1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1240_124041


namespace NUMINAMATH_CALUDE_min_value_theorem_l1240_124004

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 36/x^2 ≥ 12 * (4^(1/4)) ∧
  (x^2 + 6*x + 36/x^2 = 12 * (4^(1/4)) ↔ x = 36^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1240_124004


namespace NUMINAMATH_CALUDE_largest_prime_divisor_13_plus_14_factorial_l1240_124038

theorem largest_prime_divisor_13_plus_14_factorial (p : ℕ) :
  (p.Prime ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14)) →
  p ≤ 13 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_13_plus_14_factorial_l1240_124038


namespace NUMINAMATH_CALUDE_total_red_balloons_l1240_124007

/-- The total number of red balloons Fred, Sam, and Dan have is 72. -/
theorem total_red_balloons : 
  let fred_balloons : ℕ := 10
  let sam_balloons : ℕ := 46
  let dan_balloons : ℕ := 16
  fred_balloons + sam_balloons + dan_balloons = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_red_balloons_l1240_124007


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1240_124098

theorem triangle_angle_c (A B C : Real) (h1 : 2 * Real.sin A + 5 * Real.cos B = 5) 
  (h2 : 5 * Real.sin B + 2 * Real.cos A = 2) 
  (h3 : A + B + C = Real.pi) : 
  C = Real.arcsin (1/5) ∨ C = Real.pi - Real.arcsin (1/5) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1240_124098


namespace NUMINAMATH_CALUDE_equation_solution_l1240_124006

theorem equation_solution (x : ℝ) (h : x ≠ 3) :
  (x + 6) / (x - 3) = 5 / 2 ↔ x = 9 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1240_124006


namespace NUMINAMATH_CALUDE_root_in_interval_l1240_124045

def f (x : ℝ) := x^2 - 1

theorem root_in_interval : ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1240_124045


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1240_124095

/-- Given a line segment CD where C(5,4) is one endpoint and M(4,8) is the midpoint,
    the product of the coordinates of the other endpoint D is 36. -/
theorem midpoint_coordinate_product (C D M : ℝ × ℝ) : 
  C = (5, 4) →
  M = (4, 8) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 * D.2 = 36 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1240_124095


namespace NUMINAMATH_CALUDE_sports_competition_theorem_l1240_124086

-- Part a
def highest_average_rank (num_athletes : ℕ) (num_judges : ℕ) (max_rank_diff : ℕ) : ℚ :=
  8/3

-- Part b
def highest_winner_rank (num_players : ℕ) (max_rank_diff : ℕ) : ℕ :=
  21

theorem sports_competition_theorem :
  (highest_average_rank 20 9 3 = 8/3) ∧
  (highest_winner_rank 1024 2 = 21) :=
by sorry

end NUMINAMATH_CALUDE_sports_competition_theorem_l1240_124086


namespace NUMINAMATH_CALUDE_distance_moonbase_to_skyhaven_l1240_124008

theorem distance_moonbase_to_skyhaven :
  let moonbase : ℂ := 0
  let skyhaven : ℂ := 900 + 1200 * I
  Complex.abs (skyhaven - moonbase) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_distance_moonbase_to_skyhaven_l1240_124008


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l1240_124081

theorem log_2_base_10_bounds :
  (10^2 = 100) →
  (10^3 = 1000) →
  (2^7 = 128) →
  (2^10 = 1024) →
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (3 / 10 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l1240_124081


namespace NUMINAMATH_CALUDE_exists_motion_with_one_stationary_point_l1240_124016

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rigid body in 3D space -/
structure RigidBody where
  points : Set Point3D

/-- A motion of a rigid body -/
def Motion := RigidBody → ℝ → RigidBody

/-- A point is stationary under a motion if its position doesn't change over time -/
def IsStationary (p : Point3D) (m : Motion) (b : RigidBody) : Prop :=
  ∀ t : ℝ, p ∈ (m b t).points → p ∈ b.points

/-- A motion has exactly one stationary point -/
def HasExactlyOneStationaryPoint (m : Motion) (b : RigidBody) : Prop :=
  ∃! p : Point3D, IsStationary p m b ∧ p ∈ b.points

/-- Theorem: There exists a motion for a rigid body where exactly one point remains stationary -/
theorem exists_motion_with_one_stationary_point :
  ∃ (b : RigidBody) (m : Motion), HasExactlyOneStationaryPoint m b :=
sorry

end NUMINAMATH_CALUDE_exists_motion_with_one_stationary_point_l1240_124016


namespace NUMINAMATH_CALUDE_geometric_mean_inequality_l1240_124085

theorem geometric_mean_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let g := Real.sqrt (x * y)
  (g ≥ 3 → 1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) ∧
  (g ≤ 2 → 1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_inequality_l1240_124085


namespace NUMINAMATH_CALUDE_grandfather_wins_l1240_124073

/-- The number of games played -/
def total_games : ℕ := 12

/-- Points scored by grandfather for each win -/
def grandfather_points : ℕ := 1

/-- Points scored by grandson for each win -/
def grandson_points : ℕ := 3

/-- Theorem stating the number of games won by the grandfather -/
theorem grandfather_wins (x : ℕ) 
  (h1 : x ≤ total_games)
  (h2 : x * grandfather_points = (total_games - x) * grandson_points) : 
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_wins_l1240_124073


namespace NUMINAMATH_CALUDE_trombone_players_l1240_124072

/-- Represents the number of players for each instrument in an orchestra -/
structure Orchestra where
  total : Nat
  drummer : Nat
  trumpet : Nat
  frenchHorn : Nat
  violin : Nat
  cello : Nat
  contrabass : Nat
  clarinet : Nat
  flute : Nat
  maestro : Nat

/-- Theorem stating the number of trombone players in the orchestra -/
theorem trombone_players (o : Orchestra)
  (h1 : o.total = 21)
  (h2 : o.drummer = 1)
  (h3 : o.trumpet = 2)
  (h4 : o.frenchHorn = 1)
  (h5 : o.violin = 3)
  (h6 : o.cello = 1)
  (h7 : o.contrabass = 1)
  (h8 : o.clarinet = 3)
  (h9 : o.flute = 4)
  (h10 : o.maestro = 1) :
  o.total - (o.drummer + o.trumpet + o.frenchHorn + o.violin + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro) = 4 := by
  sorry


end NUMINAMATH_CALUDE_trombone_players_l1240_124072


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1240_124083

/-- A geometric sequence with first term a₁ and common ratio q -/
def GeometricSequence (a₁ q : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ * q^(n - 1)

theorem geometric_sequence_increasing_condition (a₁ q : ℝ) :
  (a₁ < 0 ∧ 0 < q ∧ q < 1 →
    ∀ n : ℕ, n > 0 → GeometricSequence a₁ q (n + 1) > GeometricSequence a₁ q n) ∧
  (∃ a₁' q' : ℝ, (∀ n : ℕ, n > 0 → GeometricSequence a₁' q' (n + 1) > GeometricSequence a₁' q' n) ∧
    ¬(a₁' < 0 ∧ 0 < q' ∧ q' < 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1240_124083


namespace NUMINAMATH_CALUDE_correct_product_l1240_124062

theorem correct_product (x y : ℚ) (z : ℕ) (h1 : x = 63 / 10000) (h2 : y = 385 / 100) (h3 : z = 24255) :
  x * y = 24255 / 1000000 :=
sorry

end NUMINAMATH_CALUDE_correct_product_l1240_124062


namespace NUMINAMATH_CALUDE_midnight_temperature_l1240_124011

/-- 
Given an initial temperature, a temperature rise, and a temperature drop,
calculate the final temperature.
-/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/--
Theorem: Given the specific temperature changes in the problem,
the final temperature is -4°C.
-/
theorem midnight_temperature : final_temperature (-3) 6 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l1240_124011


namespace NUMINAMATH_CALUDE_root_implies_a_values_l1240_124093

theorem root_implies_a_values (a : ℝ) :
  ((-1)^2 * a^2 + 2011 * (-1) * a - 2012 = 0) →
  (a = 2012 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_values_l1240_124093


namespace NUMINAMATH_CALUDE_part1_part2_l1240_124060

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part1 (t : Triangle) (h : (t.a^2 + t.c^2 - t.b^2) / 4 = 1/2 * t.a * t.c * Real.sin t.B) :
  t.B = π/4 := by
  sorry

-- Part 2
theorem part2 (t : Triangle) 
  (h1 : t.a * t.c = Real.sqrt 3)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.B)
  (h3 : t.C = π/6) :
  t.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l1240_124060


namespace NUMINAMATH_CALUDE_abc_inequality_l1240_124022

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (hsum : a^2 + b^2 + c^2 = 9) : a * b * c + 1 > 3 * a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1240_124022


namespace NUMINAMATH_CALUDE_acute_angle_range_l1240_124056

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

def acute_angle_obtainable (x : ℝ) : Prop :=
  let dot_product := (a x).1 * (b x).1 + (a x).2 * (b x).2
  dot_product > 0 ∧ dot_product < (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt ((b x).1^2 + (b x).2^2))

theorem acute_angle_range :
  ∀ x : ℝ, acute_angle_obtainable x ↔ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_range_l1240_124056


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l1240_124089

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

theorem probability_two_white_balls :
  (white_balls : ℚ) / total_balls * (white_balls - 1) / (total_balls - 1) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l1240_124089


namespace NUMINAMATH_CALUDE_fifteenth_triangular_number_l1240_124079

/-- The nth triangular number -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular_number : triangularNumber 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_number_l1240_124079


namespace NUMINAMATH_CALUDE_strip_overlap_area_l1240_124094

theorem strip_overlap_area (β : Real) : 
  let strip1_width : Real := 1
  let strip2_width : Real := 2
  let circle_radius : Real := 1
  let rhombus_area : Real := (1/2) * strip1_width * strip2_width * Real.sin β
  let circle_area : Real := Real.pi * circle_radius^2
  rhombus_area - circle_area = Real.sin β - Real.pi := by sorry

end NUMINAMATH_CALUDE_strip_overlap_area_l1240_124094


namespace NUMINAMATH_CALUDE_farm_problem_solution_l1240_124001

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_daily_rate : ℕ  -- Planned hectares per day
  actual_daily_rate : ℕ   -- Actual hectares per day
  extra_days : ℕ          -- Additional days worked
  remaining_area : ℕ      -- Hectares left to plough

/-- Calculates the total area and initially planned days for a given farm field problem -/
def solve_farm_problem (field : FarmField) : ℕ × ℕ :=
  sorry

/-- Theorem stating the solution to the specific farm field problem -/
theorem farm_problem_solution :
  let field := FarmField.mk 90 85 2 40
  solve_farm_problem field = (3780, 42) :=
sorry

end NUMINAMATH_CALUDE_farm_problem_solution_l1240_124001


namespace NUMINAMATH_CALUDE_caffeine_in_coffee_l1240_124082

/-- The amount of caffeine in a cup of coffee -/
def caffeine_per_cup : ℝ := 80

/-- Lisa's daily caffeine limit in milligrams -/
def daily_limit : ℝ := 200

/-- The number of cups Lisa drinks -/
def cups_drunk : ℝ := 3

/-- The amount Lisa exceeds her limit by in milligrams -/
def excess_amount : ℝ := 40

theorem caffeine_in_coffee :
  caffeine_per_cup * cups_drunk = daily_limit + excess_amount :=
by sorry

end NUMINAMATH_CALUDE_caffeine_in_coffee_l1240_124082


namespace NUMINAMATH_CALUDE_cards_arrangement_unique_l1240_124087

-- Define the suits and ranks
inductive Suit : Type
| Hearts | Diamonds | Clubs

inductive Rank : Type
| Four | Five | Eight

-- Define a card as a pair of rank and suit
def Card : Type := Rank × Suit

-- Define the arrangement of cards
def Arrangement : Type := List Card

-- Define the conditions
def club_right_of_heart_and_diamond (arr : Arrangement) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ 
    (arr.get i).2 = Suit.Hearts ∧ 
    (arr.get j).2 = Suit.Diamonds ∧ 
    (arr.get k).2 = Suit.Clubs

def five_left_of_heart (arr : Arrangement) : Prop :=
  ∃ i j, i < j ∧ 
    (arr.get i).1 = Rank.Five ∧ 
    (arr.get j).2 = Suit.Hearts

def eight_right_of_four (arr : Arrangement) : Prop :=
  ∃ i j, i < j ∧ 
    (arr.get i).1 = Rank.Four ∧ 
    (arr.get j).1 = Rank.Eight

-- Define the correct arrangement
def correct_arrangement : Arrangement :=
  [(Rank.Five, Suit.Diamonds), (Rank.Four, Suit.Hearts), (Rank.Eight, Suit.Clubs)]

-- Theorem statement
theorem cards_arrangement_unique :
  ∀ (arr : Arrangement),
    arr.length = 3 ∧
    club_right_of_heart_and_diamond arr ∧
    five_left_of_heart arr ∧
    eight_right_of_four arr →
    arr = correct_arrangement :=
sorry

end NUMINAMATH_CALUDE_cards_arrangement_unique_l1240_124087


namespace NUMINAMATH_CALUDE_value_congr_digitSum_mod_nine_divisible_by_nine_iff_digitSum_divisible_by_nine_l1240_124021

/-- Represents a non-negative integer as a list of its digits in reverse order -/
def Digits := List Nat

/-- Computes the value of a number from its digits -/
def value (d : Digits) : Nat :=
  d.enum.foldl (fun acc (i, digit) => acc + digit * 10^i) 0

/-- Computes the sum of digits -/
def digitSum (d : Digits) : Nat :=
  d.sum

/-- States that for any number, its value is congruent to its digit sum modulo 9 -/
theorem value_congr_digitSum_mod_nine (d : Digits) :
  value d ≡ digitSum d [MOD 9] := by
  sorry

/-- The main theorem: a number is divisible by 9 iff its digit sum is divisible by 9 -/
theorem divisible_by_nine_iff_digitSum_divisible_by_nine (d : Digits) :
  9 ∣ value d ↔ 9 ∣ digitSum d := by
  sorry

end NUMINAMATH_CALUDE_value_congr_digitSum_mod_nine_divisible_by_nine_iff_digitSum_divisible_by_nine_l1240_124021


namespace NUMINAMATH_CALUDE_hairdresser_initial_amount_l1240_124071

def hairdresser_savings (initial_amount : ℕ) : Prop :=
  let first_year_spent := initial_amount / 2
  let second_year_spent := initial_amount / 3
  let third_year_spent := 200
  let remaining := initial_amount - first_year_spent - second_year_spent - third_year_spent
  (remaining = 50) ∧ 
  (first_year_spent = initial_amount / 2) ∧
  (second_year_spent = initial_amount / 3) ∧
  (third_year_spent = 200)

theorem hairdresser_initial_amount : 
  ∃ (initial_amount : ℕ), hairdresser_savings initial_amount ∧ initial_amount = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_hairdresser_initial_amount_l1240_124071


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1240_124088

theorem rectangle_circle_area_ratio :
  ∀ (w r : ℝ),
  w > 0 → r > 0 →
  6 * w = 2 * Real.pi * r →
  (2 * w * w) / (Real.pi * r * r) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1240_124088


namespace NUMINAMATH_CALUDE_min_cost_at_zero_min_cost_value_l1240_124097

/-- Represents a transportation plan for machines between two locations --/
structure TransportPlan where
  x : ℕ  -- Number of machines transported from B to A
  h : x ≤ 6  -- Constraint on x

/-- Calculates the total cost of a transport plan --/
def totalCost (plan : TransportPlan) : ℕ :=
  200 * plan.x + 8600

/-- Theorem: The minimum cost occurs when no machines are moved from B to A --/
theorem min_cost_at_zero :
  ∀ plan : TransportPlan, totalCost plan ≥ 8600 := by
  sorry

/-- Theorem: The minimum cost is 8600 yuan --/
theorem min_cost_value :
  (∃ plan : TransportPlan, totalCost plan = 8600) ∧
  (∀ plan : TransportPlan, totalCost plan ≥ 8600) := by
  sorry

end NUMINAMATH_CALUDE_min_cost_at_zero_min_cost_value_l1240_124097


namespace NUMINAMATH_CALUDE_power_relation_l1240_124048

theorem power_relation (a : ℝ) (m n : ℤ) (hm : a ^ m = 4) (hn : a ^ n = 2) :
  a ^ (m - 2 * n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l1240_124048


namespace NUMINAMATH_CALUDE_second_derivative_at_pi_over_six_l1240_124049

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem second_derivative_at_pi_over_six :
  (deriv^[2] f) (π / 6) = -(1 - Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_second_derivative_at_pi_over_six_l1240_124049


namespace NUMINAMATH_CALUDE_circular_road_circumference_sum_l1240_124053

theorem circular_road_circumference_sum (R : ℝ) (h1 : R > 0) : 
  let r := R / 3
  let road_width := R - r
  road_width = 7 →
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circular_road_circumference_sum_l1240_124053


namespace NUMINAMATH_CALUDE_customers_without_tip_l1240_124064

theorem customers_without_tip (total_customers : ℕ) (total_tips : ℕ) (tip_per_customer : ℕ) :
  total_customers = 7 →
  total_tips = 6 →
  tip_per_customer = 3 →
  total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_customers_without_tip_l1240_124064


namespace NUMINAMATH_CALUDE_equation_rewrite_l1240_124078

theorem equation_rewrite :
  ∃ (m n : ℝ), (∀ x, x^2 - 12*x + 33 = 0 ↔ (x + m)^2 = n) ∧ m = -6 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l1240_124078


namespace NUMINAMATH_CALUDE_find_number_l1240_124018

theorem find_number (x : ℝ) : x^2 * 15^2 / 356 = 51.193820224719104 → x = 9 ∨ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1240_124018


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1240_124043

theorem no_real_solution_for_log_equation :
  ¬ ∃ (x : ℝ), (Real.log (x + 4) + Real.log (x - 2) = Real.log (x^2 - 6*x - 5)) ∧
               (x + 4 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 6*x - 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1240_124043


namespace NUMINAMATH_CALUDE_b_share_is_360_l1240_124039

/-- Represents the rental information for a person --/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total horse-months for a rental --/
def horsemonths (r : RentalInfo) : ℕ := r.horses * r.months

theorem b_share_is_360 (total_rent : ℕ) (a b c : RentalInfo) 
  (h1 : total_rent = 870)
  (h2 : a = ⟨12, 8⟩)
  (h3 : b = ⟨16, 9⟩)
  (h4 : c = ⟨18, 6⟩) :
  (horsemonths b * total_rent) / (horsemonths a + horsemonths b + horsemonths c) = 360 := by
  sorry

#eval (16 * 9 * 870) / (12 * 8 + 16 * 9 + 18 * 6)

end NUMINAMATH_CALUDE_b_share_is_360_l1240_124039


namespace NUMINAMATH_CALUDE_transformation_confluence_l1240_124029

/-- Represents a word in the alphabet {a, b} --/
inductive Word
| empty : Word
| cons_a : ℕ → Word → Word
| cons_b : ℕ → Word → Word

/-- Represents a transformation rule --/
structure TransformRule where
  k : ℕ
  l : ℕ
  k' : ℕ
  l' : ℕ
  h_k : k ≥ 1
  h_l : l ≥ 1
  h_k' : k' ≥ 1
  h_l' : l' ≥ 1

/-- Applies a transformation rule to a word --/
def applyRule (rule : TransformRule) (w : Word) : Option Word :=
  sorry

/-- Checks if a word is terminal with respect to a rule --/
def isTerminal (rule : TransformRule) (w : Word) : Prop :=
  applyRule rule w = none

/-- Represents a sequence of transformations --/
def TransformSequence := List (TransformRule × Word)

/-- Applies a sequence of transformations to a word --/
def applySequence (seq : TransformSequence) (w : Word) : Word :=
  sorry

theorem transformation_confluence (rule : TransformRule) (w : Word) :
  ∀ (seq1 seq2 : TransformSequence),
    isTerminal rule (applySequence seq1 w) →
    isTerminal rule (applySequence seq2 w) →
    applySequence seq1 w = applySequence seq2 w :=
  sorry

end NUMINAMATH_CALUDE_transformation_confluence_l1240_124029


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1240_124066

/-- Given two points A and B in a 2D plane, this theorem proves that
    if the vector from A to B is (3, 4) and A has coordinates (-2, -1),
    then B has coordinates (1, 3). -/
theorem point_B_coordinates
  (A B : ℝ × ℝ)
  (h1 : A = (-2, -1))
  (h2 : B.1 - A.1 = 3 ∧ B.2 - A.2 = 4) :
  B = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l1240_124066


namespace NUMINAMATH_CALUDE_circle_center_trajectory_equation_l1240_124017

/-- The trajectory of the center of a circle passing through (4,0) and 
    intersecting the y-axis with a chord of length 8 -/
def circle_center_trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1 - 16}

/-- The fixed point through which the circle passes -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- The length of the chord cut by the circle on the y-axis -/
def chord_length : ℝ := 8

/-- Theorem stating that the trajectory of the circle's center satisfies the given equation -/
theorem circle_center_trajectory_equation :
  ∀ (x y : ℝ), (x, y) ∈ circle_center_trajectory ↔ y^2 = 8*x - 16 :=
sorry

end NUMINAMATH_CALUDE_circle_center_trajectory_equation_l1240_124017


namespace NUMINAMATH_CALUDE_three_digit_sum_l1240_124030

theorem three_digit_sum (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  a < 10 ∧ b < 10 ∧ c < 10 →
  21 * (a + b + c) = 231 →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) ∨ 
  (a = 6 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 6 ∧ b = 3 ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_three_digit_sum_l1240_124030


namespace NUMINAMATH_CALUDE_incorrect_statement_l1240_124026

theorem incorrect_statement : ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (p ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1240_124026


namespace NUMINAMATH_CALUDE_f_min_at_negative_seven_l1240_124003

/-- The quadratic function f(x) = x^2 + 14x - 12 -/
def f (x : ℝ) : ℝ := x^2 + 14*x - 12

/-- The point where f attains its minimum -/
def min_point : ℝ := -7

theorem f_min_at_negative_seven :
  ∀ x : ℝ, f x ≥ f min_point := by
sorry

end NUMINAMATH_CALUDE_f_min_at_negative_seven_l1240_124003


namespace NUMINAMATH_CALUDE_cubic_factorization_l1240_124074

theorem cubic_factorization (y : ℝ) : y^3 - 16*y = y*(y+4)*(y-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1240_124074


namespace NUMINAMATH_CALUDE_decimal_representation_digits_l1240_124061

theorem decimal_representation_digits (n : ℕ) (d : ℕ) (h : n / d = 7^3 / (14^2 * 125)) : 
  (∃ k : ℕ, n / d = k / 1000 ∧ k < 1000 ∧ k ≥ 100) := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_digits_l1240_124061


namespace NUMINAMATH_CALUDE_comparison_problems_l1240_124035

theorem comparison_problems :
  (-0.1 < -0.01) ∧
  (-(-1) = abs (-1)) ∧
  (-abs (-7/8) < -(5/6)) := by sorry

end NUMINAMATH_CALUDE_comparison_problems_l1240_124035


namespace NUMINAMATH_CALUDE_tea_cheese_ratio_l1240_124019

/-- Represents the prices of items in Ursula's purchase -/
structure PurchasePrices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ

/-- The conditions of Ursula's purchase -/
def purchase_conditions (p : PurchasePrices) : Prop :=
  p.butter + p.bread + p.cheese + p.tea = 21 ∧
  p.bread = p.butter / 2 ∧
  p.butter = 0.8 * p.cheese ∧
  p.tea = 10

/-- The theorem stating the ratio of tea price to cheese price -/
theorem tea_cheese_ratio (p : PurchasePrices) :
  purchase_conditions p → p.tea / p.cheese = 2 := by
  sorry

end NUMINAMATH_CALUDE_tea_cheese_ratio_l1240_124019


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l1240_124025

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_five : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l1240_124025


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l1240_124067

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat

/-- The theorem to be proved -/
theorem twelfth_day_is_monday (m : Month) : 
  m.fridayCount = 5 ∧ 
  m.firstDay ≠ DayOfWeek.Friday ∧ 
  m.lastDay ≠ DayOfWeek.Friday ∧
  m.dayCount ≥ 28 ∧ m.dayCount ≤ 31 →
  (DayOfWeek.Monday : DayOfWeek) = 
    match (m.firstDay, 11) with
    | (DayOfWeek.Sunday, n) => DayOfWeek.Wednesday
    | (DayOfWeek.Monday, n) => DayOfWeek.Thursday
    | (DayOfWeek.Tuesday, n) => DayOfWeek.Friday
    | (DayOfWeek.Wednesday, n) => DayOfWeek.Saturday
    | (DayOfWeek.Thursday, n) => DayOfWeek.Sunday
    | (DayOfWeek.Friday, n) => DayOfWeek.Monday
    | (DayOfWeek.Saturday, n) => DayOfWeek.Tuesday
  := by sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l1240_124067


namespace NUMINAMATH_CALUDE_reciprocal_sum_is_one_l1240_124063

theorem reciprocal_sum_is_one :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_is_one_l1240_124063


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1240_124027

theorem sum_of_a_and_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) :
  a + b = 176 / 57 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1240_124027


namespace NUMINAMATH_CALUDE_ball_bird_intersection_time_l1240_124012

/-- The time at which a ball thrown off a cliff and a bird flying upwards from the base of the cliff are at the same height -/
theorem ball_bird_intersection_time : 
  ∃ t : ℝ, t > 0 ∧ (60 - 9*t - 8*t^2 = 3*t^2 + 4*t) ∧ t = 20/11 := by
  sorry

#check ball_bird_intersection_time

end NUMINAMATH_CALUDE_ball_bird_intersection_time_l1240_124012


namespace NUMINAMATH_CALUDE_product_has_34_digits_l1240_124009

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The product of two large numbers -/
def n : ℕ := 3659893456789325678 * 342973489379256

/-- Theorem stating that the product has 34 digits -/
theorem product_has_34_digits : num_digits n = 34 := by sorry

end NUMINAMATH_CALUDE_product_has_34_digits_l1240_124009


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1240_124058

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

def is_basis (v w : V) : Prop :=
  LinearIndependent ℝ ![v, w] ∧ Submodule.span ℝ {v, w} = ⊤

theorem vectors_form_basis (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ ∃ (k : ℝ), a = k • b) :
  is_basis V (a + b) (a - b) := by
  sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1240_124058


namespace NUMINAMATH_CALUDE_charlie_bobby_age_difference_l1240_124010

theorem charlie_bobby_age_difference :
  ∀ (jenny charlie bobby : ℕ),
  jenny = charlie + 5 →
  ∃ (x : ℕ), charlie + x = 11 ∧ jenny + x = 2 * (bobby + x) →
  charlie = bobby + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_charlie_bobby_age_difference_l1240_124010


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l1240_124096

/-- Given a triangle PQR with side lengths, prove that the area of triangle MOI is 11/4 -/
theorem area_of_triangle_MOI (P Q R O I M : ℝ × ℝ) : 
  let pq : ℝ := 10
  let pr : ℝ := 8
  let qr : ℝ := 6
  -- O is the circumcenter
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = (O.1 - Q.1)^2 + (O.2 - Q.2)^2 ∧
  (O.1 - Q.1)^2 + (O.2 - Q.2)^2 = (O.1 - R.1)^2 + (O.2 - R.2)^2 →
  -- I is the incenter
  (I.1 - P.1) / pq + (I.1 - Q.1) / qr + (I.1 - R.1) / pr = 0 ∧
  (I.2 - P.2) / pq + (I.2 - Q.2) / qr + (I.2 - R.2) / pr = 0 →
  -- M is the center of a circle tangent to PR, QR, and the circumcircle
  ∃ (r : ℝ), 
    r = (M.1 - P.1)^2 + (M.2 - P.2)^2 ∧
    r = (M.1 - R.1)^2 + (M.2 - R.2)^2 ∧
    r + ((O.1 - M.1)^2 + (O.2 - M.2)^2).sqrt = (O.1 - P.1)^2 + (O.2 - P.2)^2 →
  -- Area of triangle MOI is 11/4
  abs ((O.1 * (I.2 - M.2) + I.1 * (M.2 - O.2) + M.1 * (O.2 - I.2)) / 2) = 11/4 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l1240_124096


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1240_124080

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ),
    a > 2 ∧ b > 2 ∧
    n = 2 * a + 1 ∧
    n = 1 * b + 2 ∧
    (∀ (m : ℕ) (c d : ℕ),
      c > 2 → d > 2 →
      m = 2 * c + 1 →
      m = 1 * d + 2 →
      n ≤ m) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1240_124080


namespace NUMINAMATH_CALUDE_phase_shift_of_sine_l1240_124090

theorem phase_shift_of_sine (φ : Real) : 
  (0 ≤ φ ∧ φ ≤ 2 * Real.pi) →
  (∀ x, Real.sin (x + φ) = Real.sin (x - Real.pi / 6)) →
  φ = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_phase_shift_of_sine_l1240_124090


namespace NUMINAMATH_CALUDE_opposite_expression_implies_ab_zero_l1240_124014

/-- Given that for all x, ax + bx^2 = -(a(-x) + b(-x)^2), prove that ab = 0 -/
theorem opposite_expression_implies_ab_zero (a b : ℝ) 
  (h : ∀ x : ℝ, a * x + b * x^2 = -(a * (-x) + b * (-x)^2)) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expression_implies_ab_zero_l1240_124014


namespace NUMINAMATH_CALUDE_average_difference_l1240_124084

theorem average_difference (x : ℝ) : (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ↔ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1240_124084


namespace NUMINAMATH_CALUDE_solution_when_a_is_3_root_of_multiplicity_l1240_124076

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (a * x + 1) / (x - 1) - 2 / (1 - x) = 1

-- Part 1: Prove that when a = 3, the solution is x = -2
theorem solution_when_a_is_3 :
  ∃ x : ℝ, x ≠ 1 ∧ equation 3 x ∧ x = -2 :=
sorry

-- Part 2: Prove that the equation has a root of multiplicity when a = -3
theorem root_of_multiplicity :
  ∃ x : ℝ, x = 1 ∧ equation (-3) x :=
sorry

end NUMINAMATH_CALUDE_solution_when_a_is_3_root_of_multiplicity_l1240_124076


namespace NUMINAMATH_CALUDE_f_has_minimum_value_neg_twelve_l1240_124065

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

theorem f_has_minimum_value_neg_twelve :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_value_neg_twelve_l1240_124065


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1240_124013

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1240_124013


namespace NUMINAMATH_CALUDE_science_club_enrollment_l1240_124000

theorem science_club_enrollment (total : ℕ) (math physics chem : ℕ) 
  (math_physics math_chem physics_chem : ℕ) (all_three : ℕ) 
  (h_total : total = 150)
  (h_math : math = 90)
  (h_physics : physics = 70)
  (h_chem : chem = 40)
  (h_math_physics : math_physics = 20)
  (h_math_chem : math_chem = 15)
  (h_physics_chem : physics_chem = 10)
  (h_all_three : all_three = 5) :
  total - (math + physics + chem - math_physics - math_chem - physics_chem + all_three) = 5 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l1240_124000


namespace NUMINAMATH_CALUDE_modular_inverse_27_mod_28_l1240_124032

theorem modular_inverse_27_mod_28 : ∃ a : ℕ, 0 ≤ a ∧ a ≤ 27 ∧ (27 * a) % 28 = 1 :=
by
  use 27
  sorry

end NUMINAMATH_CALUDE_modular_inverse_27_mod_28_l1240_124032


namespace NUMINAMATH_CALUDE_average_pages_is_23_l1240_124037

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 161

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Calculates the average number of pages read per day -/
def average_pages_per_day : ℚ := total_pages / days_in_week

/-- Theorem stating that the average number of pages read per day is 23 -/
theorem average_pages_is_23 : average_pages_per_day = 23 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_is_23_l1240_124037


namespace NUMINAMATH_CALUDE_total_commute_time_is_19_point_1_l1240_124020

/-- Represents the commute schedule for a week --/
structure CommuteSchedule where
  normalWalkTime : ℝ
  normalBikeTime : ℝ
  wednesdayExtraTime : ℝ
  fridayExtraTime : ℝ
  rainIncreaseFactor : ℝ
  mondayIsWalking : Bool
  tuesdayIsBiking : Bool
  wednesdayIsWalking : Bool
  thursdayIsWalking : Bool
  fridayIsBiking : Bool
  mondayIsRainy : Bool
  thursdayIsRainy : Bool

/-- Calculates the total commute time for a week given a schedule --/
def totalCommuteTime (schedule : CommuteSchedule) : ℝ :=
  let mondayTime := if schedule.mondayIsWalking then
    (if schedule.mondayIsRainy then schedule.normalWalkTime * (1 + schedule.rainIncreaseFactor) else schedule.normalWalkTime) * 2
  else schedule.normalBikeTime * 2

  let tuesdayTime := if schedule.tuesdayIsBiking then schedule.normalBikeTime * 2
  else schedule.normalWalkTime * 2

  let wednesdayTime := if schedule.wednesdayIsWalking then (schedule.normalWalkTime + schedule.wednesdayExtraTime) * 2
  else schedule.normalBikeTime * 2

  let thursdayTime := if schedule.thursdayIsWalking then
    (if schedule.thursdayIsRainy then schedule.normalWalkTime * (1 + schedule.rainIncreaseFactor) else schedule.normalWalkTime) * 2
  else schedule.normalBikeTime * 2

  let fridayTime := if schedule.fridayIsBiking then (schedule.normalBikeTime + schedule.fridayExtraTime) * 2
  else schedule.normalWalkTime * 2

  mondayTime + tuesdayTime + wednesdayTime + thursdayTime + fridayTime

/-- The main theorem stating that given the specific schedule, the total commute time is 19.1 hours --/
theorem total_commute_time_is_19_point_1 :
  let schedule : CommuteSchedule := {
    normalWalkTime := 2
    normalBikeTime := 1
    wednesdayExtraTime := 0.5
    fridayExtraTime := 0.25
    rainIncreaseFactor := 0.2
    mondayIsWalking := true
    tuesdayIsBiking := true
    wednesdayIsWalking := true
    thursdayIsWalking := true
    fridayIsBiking := true
    mondayIsRainy := true
    thursdayIsRainy := true
  }
  totalCommuteTime schedule = 19.1 := by sorry

end NUMINAMATH_CALUDE_total_commute_time_is_19_point_1_l1240_124020


namespace NUMINAMATH_CALUDE_sum_of_two_angles_in_plane_l1240_124052

/-- 
Given three angles meeting at a point in a plane, where one angle is 130°, 
prove that the sum of the other two angles is 230°.
-/
theorem sum_of_two_angles_in_plane (x y : ℝ) : 
  x + y + 130 = 360 → x + y = 230 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_angles_in_plane_l1240_124052


namespace NUMINAMATH_CALUDE_factoring_expression_l1240_124023

theorem factoring_expression (x : ℝ) : 5*x*(x-2) + 9*(x-2) = (x-2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1240_124023


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1240_124068

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 150 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1240_124068


namespace NUMINAMATH_CALUDE_initialMenCountIs8_l1240_124092

/-- The initial number of men in a group where:
  - The average age increases by 2 years when two women replace two men
  - The two men being replaced are aged 20 and 24 years
  - The average age of the women is 30 years
-/
def initialMenCount : ℕ := by
  -- Define the increase in average age
  let averageAgeIncrease : ℕ := 2
  -- Define the ages of the men being replaced
  let replacedManAge1 : ℕ := 20
  let replacedManAge2 : ℕ := 24
  -- Define the average age of the women
  let womenAverageAge : ℕ := 30
  
  -- The proof goes here
  sorry

/-- Theorem stating that the initial number of men is 8 -/
theorem initialMenCountIs8 : initialMenCount = 8 := by sorry

end NUMINAMATH_CALUDE_initialMenCountIs8_l1240_124092


namespace NUMINAMATH_CALUDE_cos_decreasing_interval_l1240_124033

theorem cos_decreasing_interval (k : ℤ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 3)
  let a := k * π + π / 6
  let b := k * π + 2 * π / 3
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_cos_decreasing_interval_l1240_124033


namespace NUMINAMATH_CALUDE_lcm_18_24_l1240_124034

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1240_124034


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l1240_124046

/-- A triangle represented by three points on a 2D plane. -/
structure Triangle :=
  (a b c : ℕ × ℕ)

/-- Checks if a triangle is isosceles. -/
def isIsosceles (t : Triangle) : Bool :=
  let d1 := ((t.a.1 - t.b.1)^2 + (t.a.2 - t.b.2)^2 : ℕ)
  let d2 := ((t.b.1 - t.c.1)^2 + (t.b.2 - t.c.2)^2 : ℕ)
  let d3 := ((t.c.1 - t.a.1)^2 + (t.c.2 - t.a.2)^2 : ℕ)
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

def triangles : List Triangle := [
  ⟨(0, 6), (2, 6), (1, 4)⟩,
  ⟨(3, 4), (3, 6), (5, 4)⟩,
  ⟨(0, 1), (3, 2), (6, 1)⟩,
  ⟨(7, 4), (6, 6), (9, 4)⟩,
  ⟨(8, 1), (9, 3), (10, 0)⟩
]

theorem four_isosceles_triangles :
  (triangles.filter isIsosceles).length = 4 := by
  sorry


end NUMINAMATH_CALUDE_four_isosceles_triangles_l1240_124046


namespace NUMINAMATH_CALUDE_stratified_sampling_primary_schools_l1240_124031

theorem stratified_sampling_primary_schools 
  (total_schools : ℕ) 
  (primary_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = 250) 
  (h2 : primary_schools = 150) 
  (h3 : sample_size = 30) :
  (primary_schools : ℚ) / total_schools * sample_size = 18 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_primary_schools_l1240_124031


namespace NUMINAMATH_CALUDE_complement_of_union_is_empty_l1240_124040

def U : Finset Nat := {1, 2, 3, 4}
def A : Finset Nat := {1, 3, 4}
def B : Finset Nat := {2, 3, 4}

theorem complement_of_union_is_empty :
  (U \ (A ∪ B) : Finset Nat) = ∅ := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_empty_l1240_124040


namespace NUMINAMATH_CALUDE_alpha_value_l1240_124002

theorem alpha_value (α : Real) 
  (h1 : -π/2 < α ∧ α < π/2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 2 / 2) : 
  α = -π/12 := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l1240_124002


namespace NUMINAMATH_CALUDE_unique_solution_l1240_124050

theorem unique_solution : ∃! x : ℝ, (x = (1/x)*(-x) - 5) ∧ (x^2 - 3*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1240_124050


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1240_124042

/-- Proves that the initial amount of money is $160 given the conditions of the problem -/
theorem initial_money_calculation (your_weekly_savings : ℕ) (friend_initial_money : ℕ) 
  (friend_weekly_savings : ℕ) (weeks : ℕ) (h1 : your_weekly_savings = 7) 
  (h2 : friend_initial_money = 210) (h3 : friend_weekly_savings = 5) (h4 : weeks = 25) :
  ∃ (your_initial_money : ℕ), 
    your_initial_money + your_weekly_savings * weeks = 
    friend_initial_money + friend_weekly_savings * weeks ∧ 
    your_initial_money = 160 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1240_124042


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_l1240_124047

/-- Given two points P and Q in a Cartesian coordinate system,
    where P has coordinates (m, 3) and Q has coordinates (2-2m, m-3),
    and PQ is parallel to the y-axis, prove that m = 2/3. -/
theorem parallel_to_y_axis (m : ℚ) : 
  let P : ℚ × ℚ := (m, 3)
  let Q : ℚ × ℚ := (2 - 2*m, m - 3)
  (P.1 = Q.1) → m = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_l1240_124047


namespace NUMINAMATH_CALUDE_train_length_l1240_124054

/-- Calculates the length of a train given the time it takes to cross a bridge and pass a lamp post. -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ) :
  bridge_length = 150 →
  bridge_time = 7.5 →
  lamp_time = 2.5 →
  ∃ (train_length : ℝ), train_length = 75 ∧ 
    (train_length / lamp_time = (train_length + bridge_length) / bridge_time) :=
by sorry


end NUMINAMATH_CALUDE_train_length_l1240_124054


namespace NUMINAMATH_CALUDE_parents_without_jobs_l1240_124077

/-- The percentage of parents without full-time jobs -/
def percentage_without_jobs (mother_job_rate : ℝ) (father_job_rate : ℝ) (mother_percentage : ℝ) : ℝ :=
  100 - (mother_job_rate * mother_percentage + father_job_rate * (100 - mother_percentage))

theorem parents_without_jobs :
  percentage_without_jobs 90 75 40 = 19 := by
  sorry

end NUMINAMATH_CALUDE_parents_without_jobs_l1240_124077


namespace NUMINAMATH_CALUDE_inequality_multiplication_l1240_124028

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l1240_124028


namespace NUMINAMATH_CALUDE_f_properties_and_value_l1240_124069

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem stating the properties of f and its value at -1 -/
theorem f_properties_and_value :
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ 
  (∀ x, f x = 3 * f⁻¹ x + 5) ∧
  (f 0 = 3) →
  f (-1) = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_f_properties_and_value_l1240_124069
