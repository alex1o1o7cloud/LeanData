import Mathlib

namespace NUMINAMATH_CALUDE_interest_difference_implies_sum_l1000_100025

/-- Proves that if the difference between compound interest and simple interest
    on a sum at 5% per annum for 2 years is Rs. 60, then the sum is Rs. 24,000. -/
theorem interest_difference_implies_sum (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) - P * (0.05 * 2) = 60 → P = 24000 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_sum_l1000_100025


namespace NUMINAMATH_CALUDE_min_value_sin_cos_cubic_min_value_achievable_l1000_100016

theorem min_value_sin_cos_cubic (x : ℝ) : 
  Real.sin x ^ 3 + 2 * Real.cos x ^ 3 ≥ -4 * Real.sqrt 2 / 3 :=
sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sin x ^ 3 + 2 * Real.cos x ^ 3 = -4 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_cubic_min_value_achievable_l1000_100016


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l1000_100077

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 6| - |3*x - 9|

-- Define the domain of x
def domain (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 12

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max min : ℝ), 
    (∀ x, domain x → g x ≤ max) ∧
    (∃ x, domain x ∧ g x = max) ∧
    (∀ x, domain x → min ≤ g x) ∧
    (∃ x, domain x ∧ g x = min) ∧
    max + min = -6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l1000_100077


namespace NUMINAMATH_CALUDE_proper_divisor_cube_difference_l1000_100063

theorem proper_divisor_cube_difference (n : ℕ) : 
  (∃ (x y : ℕ), 
    x > 1 ∧ y > 1 ∧
    x ∣ n ∧ y ∣ n ∧
    n ≠ x ∧ n ≠ y ∧
    (∀ z : ℕ, z > 1 ∧ z ∣ n ∧ n ≠ z → z ≥ x) ∧
    (∀ z : ℕ, z > 1 ∧ z ∣ n ∧ n ≠ z → z ≤ y) ∧
    (y = x^3 + 3 ∨ y = x^3 - 3)) ↔
  (n = 10 ∨ n = 22) :=
sorry

end NUMINAMATH_CALUDE_proper_divisor_cube_difference_l1000_100063


namespace NUMINAMATH_CALUDE_water_storage_solution_l1000_100078

/-- Represents the water storage problem with barrels and casks. -/
def WaterStorage (cask_capacity : ℕ) (barrel_count : ℕ) : Prop :=
  let barrel_capacity := 2 * cask_capacity + 3
  barrel_count * barrel_capacity = 172

/-- Theorem stating that given the problem conditions, the total water storage is 172 gallons. -/
theorem water_storage_solution :
  WaterStorage 20 4 := by
  sorry

end NUMINAMATH_CALUDE_water_storage_solution_l1000_100078


namespace NUMINAMATH_CALUDE_simplify_fraction_l1000_100049

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1000_100049


namespace NUMINAMATH_CALUDE_earl_money_proof_l1000_100053

def earl_initial_money (e f g : ℕ) : Prop :=
  f = 48 ∧ 
  g = 36 ∧ 
  e - 28 + 40 + (g + 32 - 40) = 130 ∧
  e = 90

theorem earl_money_proof :
  ∀ e f g : ℕ, earl_initial_money e f g :=
by
  sorry

end NUMINAMATH_CALUDE_earl_money_proof_l1000_100053


namespace NUMINAMATH_CALUDE_power_function_coefficient_l1000_100015

/-- A function f is a power function if it has the form f(x) = ax^n, where a and n are constants and n ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), n ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- If f(x) = (2m-1)x^3 is a power function, then m = 1 -/
theorem power_function_coefficient (m : ℝ) :
  IsPowerFunction (fun x => (2 * m - 1) * x ^ 3) → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_coefficient_l1000_100015


namespace NUMINAMATH_CALUDE_simplify_fraction_l1000_100044

theorem simplify_fraction : (96 : ℚ) / 160 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1000_100044


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1000_100083

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 500| ≤ 5} = {x : ℝ | 495 ≤ x ∧ x ≤ 505} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1000_100083


namespace NUMINAMATH_CALUDE_right_triangle_area_l1000_100032

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) : 
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1000_100032


namespace NUMINAMATH_CALUDE_three_parallel_lines_theorem_l1000_100085

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Checks if three lines are coplanar -/
def are_coplanar (l1 l2 l3 : Line3D) : Prop := sorry

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- The number of planes determined by three lines -/
def planes_from_lines (l1 l2 l3 : Line3D) : ℕ := sorry

/-- The number of parts the space is divided into by these planes -/
def space_divisions (planes : ℕ) : ℕ := sorry

theorem three_parallel_lines_theorem (a b c : Line3D) 
  (h_parallel_ab : are_parallel a b)
  (h_parallel_bc : are_parallel b c)
  (h_parallel_ac : are_parallel a c)
  (h_not_coplanar : ¬ are_coplanar a b c) :
  planes_from_lines a b c = 3 ∧ space_divisions (planes_from_lines a b c) = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_parallel_lines_theorem_l1000_100085


namespace NUMINAMATH_CALUDE_three_people_selection_l1000_100081

-- Define the number of people in the group
def n : ℕ := 30

-- Define the number of enemies each person has
def enemies_per_person : ℕ := 6

-- Define the function to calculate the number of ways to select 3 people
-- such that any two of them are either friends or enemies
def select_three_people (n : ℕ) (enemies_per_person : ℕ) : ℕ :=
  -- The actual calculation is not implemented, as per instructions
  sorry

-- The theorem to prove
theorem three_people_selection :
  select_three_people n enemies_per_person = 1990 := by
  sorry

end NUMINAMATH_CALUDE_three_people_selection_l1000_100081


namespace NUMINAMATH_CALUDE_decimal_77_to_octal_l1000_100061

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem decimal_77_to_octal :
  decimal_to_octal 77 = [5, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_77_to_octal_l1000_100061


namespace NUMINAMATH_CALUDE_parabola_points_l1000_100030

/-- A point on a parabola with equation y² = 4x that is 3 units away from its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_from_focus : (x - 1)^2 + y^2 = 3^2

/-- The theorem stating that (2, 2√2) and (2, -2√2) are the points on the parabola y² = 4x
    that are 3 units away from its focus -/
theorem parabola_points : 
  (∃ (p : ParabolaPoint), p.x = 2 ∧ p.y = 2 * Real.sqrt 2) ∧
  (∃ (p : ParabolaPoint), p.x = 2 ∧ p.y = -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_points_l1000_100030


namespace NUMINAMATH_CALUDE_system_solution_l1000_100041

theorem system_solution (a b c d : ℚ) 
  (eq1 : 4 * a + 2 * b + 6 * c + 8 * d = 48)
  (eq2 : 4 * d + 2 * c = 2 * b)
  (eq3 : 4 * b + 2 * c = 2 * a)
  (eq4 : c + 2 = d) :
  a * b * c * d = -11033 / 1296 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1000_100041


namespace NUMINAMATH_CALUDE_min_points_eleventh_game_l1000_100070

/-- Represents the scores of a basketball player -/
structure BasketballScores where
  scores_7_to_10 : Fin 4 → ℕ
  total_after_6 : ℕ
  total_after_10 : ℕ
  total_after_11 : ℕ

/-- The minimum number of points required in the 11th game -/
def min_points_11th_game (bs : BasketballScores) : ℕ := bs.total_after_11 - bs.total_after_10

/-- Theorem stating the minimum points required in the 11th game -/
theorem min_points_eleventh_game 
  (bs : BasketballScores)
  (h1 : bs.scores_7_to_10 = ![21, 15, 12, 19])
  (h2 : (bs.total_after_10 : ℚ) / 10 > (bs.total_after_6 : ℚ) / 6)
  (h3 : (bs.total_after_11 : ℚ) / 11 > 20)
  (h4 : bs.total_after_10 = bs.total_after_6 + (bs.scores_7_to_10 0) + (bs.scores_7_to_10 1) + 
                            (bs.scores_7_to_10 2) + (bs.scores_7_to_10 3))
  : min_points_11th_game bs = 58 := by
  sorry


end NUMINAMATH_CALUDE_min_points_eleventh_game_l1000_100070


namespace NUMINAMATH_CALUDE_blue_candies_count_l1000_100065

/-- The number of blue candies in a bag, given the following conditions:
    - There are 5 green candies and 4 red candies
    - The probability of picking a blue candy is 25% -/
def num_blue_candies : ℕ :=
  let green_candies : ℕ := 5
  let red_candies : ℕ := 4
  let prob_blue : ℚ := 1/4
  3

theorem blue_candies_count :
  let green_candies : ℕ := 5
  let red_candies : ℕ := 4
  let prob_blue : ℚ := 1/4
  let total_candies : ℕ := green_candies + red_candies + num_blue_candies
  (num_blue_candies : ℚ) / total_candies = prob_blue :=
by sorry

end NUMINAMATH_CALUDE_blue_candies_count_l1000_100065


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l1000_100064

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 886 / 1000)
  : (total_bananas - (good_fruits_percentage * (total_oranges + total_bananas) - (1 - rotten_oranges_percentage) * total_oranges)) / total_bananas = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l1000_100064


namespace NUMINAMATH_CALUDE_jemma_grasshopper_count_l1000_100071

/-- The number of grasshoppers Jemma found on the African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found on the grass -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshopper_count_l1000_100071


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1000_100094

theorem probability_of_red_ball (white_balls red_balls : ℕ) : 
  white_balls = 3 → red_balls = 7 → (red_balls : ℚ) / (white_balls + red_balls) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1000_100094


namespace NUMINAMATH_CALUDE_square_root_problem_l1000_100004

theorem square_root_problem (h1 : Real.sqrt 99225 = 315) (h2 : Real.sqrt x = 3.15) : x = 9.9225 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1000_100004


namespace NUMINAMATH_CALUDE_inequality_proof_l1000_100034

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1000_100034


namespace NUMINAMATH_CALUDE_correct_transformation_l1000_100054

theorem correct_transformation (y : ℝ) : y + 2 = -3 → y = -5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1000_100054


namespace NUMINAMATH_CALUDE_exponent_division_l1000_100059

theorem exponent_division (a : ℝ) : 2 * a^3 / a^2 = 2 * a := by sorry

end NUMINAMATH_CALUDE_exponent_division_l1000_100059


namespace NUMINAMATH_CALUDE_valid_pairings_count_l1000_100037

def number_of_bowls : ℕ := 5
def number_of_glasses : ℕ := 5
def number_of_colors : ℕ := 5

def total_pairings : ℕ := number_of_bowls * number_of_glasses

def invalid_pairings : ℕ := 1

theorem valid_pairings_count : 
  total_pairings - invalid_pairings = 24 :=
sorry

end NUMINAMATH_CALUDE_valid_pairings_count_l1000_100037


namespace NUMINAMATH_CALUDE_win_sector_area_l1000_100076

/-- Given a circular spinner with radius 8 cm and a probability of winning 3/8,
    the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l1000_100076


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1000_100027

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1000_100027


namespace NUMINAMATH_CALUDE_farm_feet_count_l1000_100066

/-- Given a farm with hens and cows, calculate the total number of feet -/
theorem farm_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 46 → hen_count = 22 → (hen_count * 2 + (total_heads - hen_count) * 4 = 140) :=
by
  sorry

#check farm_feet_count

end NUMINAMATH_CALUDE_farm_feet_count_l1000_100066


namespace NUMINAMATH_CALUDE_organize_60_toys_in_15_minutes_l1000_100072

/-- Represents the toy organizing scenario with Mia and her dad -/
structure ToyOrganizing where
  totalToys : ℕ
  cycleTime : ℕ
  dadPlaces : ℕ
  miaTakesOut : ℕ

/-- Calculates the time in minutes to organize all toys -/
def timeToOrganize (scenario : ToyOrganizing) : ℚ :=
  sorry

/-- Theorem stating that the time to organize 60 toys is 15 minutes -/
theorem organize_60_toys_in_15_minutes :
  let scenario : ToyOrganizing := {
    totalToys := 60,
    cycleTime := 30,  -- in seconds
    dadPlaces := 6,
    miaTakesOut := 4
  }
  timeToOrganize scenario = 15 := by sorry

end NUMINAMATH_CALUDE_organize_60_toys_in_15_minutes_l1000_100072


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1000_100074

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the ratio between two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

def Arun : Age := ⟨20⟩
def Deepak : Age := ⟨30⟩

def currentRatio : AgeRatio := ⟨2, 3⟩

theorem age_ratio_proof :
  (Arun.years + 5 = 25) ∧
  (Deepak.years = 30) →
  (currentRatio.numerator * Deepak.years = currentRatio.denominator * Arun.years) :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1000_100074


namespace NUMINAMATH_CALUDE_registration_scientific_correct_l1000_100011

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of people registered for the national college entrance examination in 2023 -/
def registration_number : ℕ := 12910000

/-- The scientific notation representation of the registration number -/
def registration_scientific : ScientificNotation :=
  { coefficient := 1.291,
    exponent := 7,
    is_valid := by sorry }

/-- Theorem stating that the registration number is correctly represented in scientific notation -/
theorem registration_scientific_correct :
  (registration_scientific.coefficient * (10 : ℝ) ^ registration_scientific.exponent) = registration_number := by
  sorry

end NUMINAMATH_CALUDE_registration_scientific_correct_l1000_100011


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l1000_100056

/-- The curve on which point P moves -/
def curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The first line l₁ -/
def line1 (y : ℝ) : Prop := y = 2

/-- The second line l₂ -/
def line2 (x : ℝ) : Prop := x = -1

/-- The distance from a point (x, y) to line1 -/
def dist_to_line1 (y : ℝ) : ℝ := |y - 2|

/-- The distance from a point (x, y) to line2 -/
def dist_to_line2 (x : ℝ) : ℝ := |x + 1|

/-- The sum of distances from a point (x, y) to both lines -/
def sum_of_distances (x y : ℝ) : ℝ := dist_to_line1 y + dist_to_line2 x

/-- The theorem stating the minimum value of the sum of distances -/
theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 4 - Real.sqrt 2 ∧
  ∀ (x y : ℝ), curve x y → sum_of_distances x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l1000_100056


namespace NUMINAMATH_CALUDE_set_equality_implies_m_values_l1000_100097

def A : Set ℝ := {x | x^2 - 3*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

theorem set_equality_implies_m_values (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1/2 ∨ m = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_values_l1000_100097


namespace NUMINAMATH_CALUDE_parallelogram_reflection_l1000_100001

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the parallelogram
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the perpendicular line
def perpendicular_line (p : Parallelogram) (t : Line) : Prop :=
  -- Assuming some condition for perpendicularity
  sorry

-- Define the intersection points
def intersection_point (l1 l2 : Line) : Point :=
  -- Assuming some method to find intersection
  sorry

-- Define the reflection operation
def reflect_point (p : Point) (t : Line) : Point :=
  -- Assuming some method to reflect a point over a line
  sorry

-- The main theorem
theorem parallelogram_reflection 
  (p : Parallelogram) 
  (t : Line) 
  (h_perp : perpendicular_line p t) :
  ∃ (p' : Parallelogram),
    let K := intersection_point (Line.mk 0 1 0) t  -- Assuming AB is on y-axis for simplicity
    let L := intersection_point (Line.mk 0 1 0) t  -- Assuming CD is parallel to AB
    p'.A = reflect_point p.A t ∧
    p'.B = reflect_point p.B t ∧
    p'.C = reflect_point p.C t ∧
    p'.D = reflect_point p.D t ∧
    p'.A = Point.mk (2 * K.x - p.A.x) (2 * K.y - p.A.y) ∧
    p'.B = Point.mk (2 * K.x - p.B.x) (2 * K.y - p.B.y) ∧
    p'.C = Point.mk (2 * L.x - p.C.x) (2 * L.y - p.C.y) ∧
    p'.D = Point.mk (2 * L.x - p.D.x) (2 * L.y - p.D.y) :=
  by sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_l1000_100001


namespace NUMINAMATH_CALUDE_smallest_positive_root_l1000_100017

noncomputable def α : Real := Real.arctan (2 / 9)
noncomputable def β : Real := Real.arctan (6 / 7)

def equation (x : Real) : Prop :=
  2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x)

theorem smallest_positive_root :
  ∃ (x : Real), x > 0 ∧ equation x ∧ ∀ (y : Real), y > 0 ∧ equation y → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_l1000_100017


namespace NUMINAMATH_CALUDE_distance_is_134_l1000_100052

/-- The distance between two girls walking in opposite directions after 12 hours -/
def distance_between_girls : ℝ :=
  let girl1_speed1 : ℝ := 7
  let girl1_time1 : ℝ := 6
  let girl1_speed2 : ℝ := 10
  let girl1_time2 : ℝ := 6
  let girl2_speed1 : ℝ := 3
  let girl2_time1 : ℝ := 8
  let girl2_speed2 : ℝ := 2
  let girl2_time2 : ℝ := 4
  let girl1_distance : ℝ := girl1_speed1 * girl1_time1 + girl1_speed2 * girl1_time2
  let girl2_distance : ℝ := girl2_speed1 * girl2_time1 + girl2_speed2 * girl2_time2
  girl1_distance + girl2_distance

/-- Theorem stating that the distance between the girls after 12 hours is 134 km -/
theorem distance_is_134 : distance_between_girls = 134 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_134_l1000_100052


namespace NUMINAMATH_CALUDE_product_of_sums_inequality_l1000_100010

theorem product_of_sums_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
sorry

end NUMINAMATH_CALUDE_product_of_sums_inequality_l1000_100010


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1000_100043

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (product_eq : x * y = 200) : 
  |x - y| = 10 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1000_100043


namespace NUMINAMATH_CALUDE_perpendicular_vector_equation_l1000_100031

/-- Given two vectors a and b in ℝ², find the value of t such that a is perpendicular to (t * a + b) -/
theorem perpendicular_vector_equation (a b : ℝ × ℝ) (h : a = (1, 2) ∧ b = (4, 3)) :
  ∃ t : ℝ, a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ∧ t = -2 := by
  sorry

#check perpendicular_vector_equation

end NUMINAMATH_CALUDE_perpendicular_vector_equation_l1000_100031


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1000_100047

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1000_100047


namespace NUMINAMATH_CALUDE_beadshop_profit_l1000_100058

theorem beadshop_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ)
  (h_total : total_profit = 1200)
  (h_monday : monday_fraction = 1/3)
  (h_tuesday : tuesday_fraction = 1/4) :
  total_profit - (monday_fraction * total_profit + tuesday_fraction * total_profit) = 500 := by
  sorry

end NUMINAMATH_CALUDE_beadshop_profit_l1000_100058


namespace NUMINAMATH_CALUDE_valid_numbers_l1000_100020

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  (n / 100 = n % 10) ∧  -- hundreds and units digits are the same
  n % 15 = 0            -- divisible by 15

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {525, 555, 585} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1000_100020


namespace NUMINAMATH_CALUDE_t_level_quasi_increasing_range_l1000_100003

/-- Definition of t-level quasi-increasing function -/
def is_t_level_quasi_increasing (f : ℝ → ℝ) (t : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, (x + t) ∈ M ∧ f (x + t) ≥ f x

/-- The function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The interval we're considering -/
def M : Set ℝ := {x | x ≥ 1}

/-- The main theorem -/
theorem t_level_quasi_increasing_range :
  {t : ℝ | is_t_level_quasi_increasing f t M} = {t : ℝ | t ≥ 1} := by sorry

end NUMINAMATH_CALUDE_t_level_quasi_increasing_range_l1000_100003


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1000_100079

/-- 
Given a boat that travels at different speeds with and against a stream,
this theorem proves that its speed in still water is 6 km/hr.
-/
theorem boat_speed_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 7) 
  (h2 : speed_against_stream = 5) : 
  (speed_with_stream + speed_against_stream) / 2 = 6 := by
sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1000_100079


namespace NUMINAMATH_CALUDE_lines_perpendicular_l1000_100073

-- Define the lines l₁ and l
def l₁ (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y - 1 = 0
def l (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the theorem
theorem lines_perpendicular :
  ∃ a : ℝ, 
    (l₁ a 1 1) ∧ 
    (∀ x y : ℝ, l₁ a x y → l x y → (2 : ℝ) * (-1/2 : ℝ) = -1) :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l1000_100073


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l1000_100040

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (increased_speed : ℝ)
  (target_average_speed : ℝ)
  (h1 : initial_distance = 15)
  (h2 : initial_speed = 30)
  (h3 : increased_speed = 55)
  (h4 : target_average_speed = 50) :
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / increased_speed)) = target_average_speed ∧
    additional_distance = 110 :=
by sorry

end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l1000_100040


namespace NUMINAMATH_CALUDE_linear_function_intersection_l1000_100012

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the theorem
theorem linear_function_intersection (k : ℝ) :
  (∃ t : ℝ, t > 0 ∧ f k t = 0) →  -- x-axis intersection exists and is positive
  (f k 0 = 3) →  -- y-axis intersection is (0, 3)
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f k x₁ > f k x₂) →  -- y decreases as x increases
  (∃ t : ℝ, t > 0 ∧ f k t = 0 ∧ t^2 + 3^2 = 5^2) →  -- distance between intersections is 5
  k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l1000_100012


namespace NUMINAMATH_CALUDE_b_minus_c_equals_one_l1000_100090

theorem b_minus_c_equals_one (A B C : ℤ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h2 : A = 9 - 4)
  (h3 : B = A + 5)
  (h4 : C - 8 = 1) : 
  B - C = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_c_equals_one_l1000_100090


namespace NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l1000_100075

theorem definite_integral_exp_plus_2x : 
  ∫ x in (-1)..1, (Real.exp x + 2 * x) = Real.exp 1 - Real.exp (-1) := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l1000_100075


namespace NUMINAMATH_CALUDE_sum_not_odd_l1000_100057

theorem sum_not_odd (n m : ℤ) 
  (h1 : Even (n^3 + m^3))
  (h2 : (n^3 + m^3) % 4 = 0) : 
  ¬(Odd (n + m)) := by
sorry

end NUMINAMATH_CALUDE_sum_not_odd_l1000_100057


namespace NUMINAMATH_CALUDE_henrys_initial_income_l1000_100099

theorem henrys_initial_income (initial_income : ℝ) : 
  initial_income * 1.5 = 180 → initial_income = 120 := by
  sorry

end NUMINAMATH_CALUDE_henrys_initial_income_l1000_100099


namespace NUMINAMATH_CALUDE_no_linear_term_implies_n_eq_neg_two_l1000_100002

theorem no_linear_term_implies_n_eq_neg_two (n : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + n) * (x + 2) = a * x^2 + b) → n = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_n_eq_neg_two_l1000_100002


namespace NUMINAMATH_CALUDE_blake_receives_four_dollars_change_l1000_100033

/-- The amount of change Blake receives after purchasing lollipops and chocolate. -/
def blakes_change (lollipop_count : ℕ) (chocolate_pack_count : ℕ) (lollipop_price : ℕ) (bill_count : ℕ) (bill_value : ℕ) : ℕ :=
  let chocolate_pack_price := 4 * lollipop_price
  let total_cost := lollipop_count * lollipop_price + chocolate_pack_count * chocolate_pack_price
  let payment := bill_count * bill_value
  payment - total_cost

/-- Theorem stating that Blake's change is $4 given the problem conditions. -/
theorem blake_receives_four_dollars_change :
  blakes_change 4 6 2 6 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_receives_four_dollars_change_l1000_100033


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1000_100088

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12) ∧ (a + b = 50) ∧ 
  (∀ (c d : ℕ+), c ≠ d → ((1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12) → (c + d ≥ 50)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1000_100088


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1000_100092

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + b + 5*I = 9 + a*I → b = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1000_100092


namespace NUMINAMATH_CALUDE_parallelogram_area_l1000_100042

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area : 
  ∀ (base height : ℝ), 
  base = 20 → 
  height = 4 → 
  base * height = 80 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1000_100042


namespace NUMINAMATH_CALUDE_max_polyline_length_6x10_l1000_100091

/-- Represents a checkered field with rows and columns -/
structure CheckeredField where
  rows : Nat
  columns : Nat

/-- Represents a polyline on a checkered field -/
structure Polyline where
  field : CheckeredField
  length : Nat
  closed : Bool
  nonSelfIntersecting : Bool

/-- The maximum length of a closed, non-self-intersecting polyline on a given field -/
def maxPolylineLength (field : CheckeredField) : Nat :=
  sorry

/-- Theorem: The maximum length of a closed, non-self-intersecting polyline
    on a 6 × 10 checkered field is 76 -/
theorem max_polyline_length_6x10 :
  let field := CheckeredField.mk 6 10
  maxPolylineLength field = 76 := by
  sorry

end NUMINAMATH_CALUDE_max_polyline_length_6x10_l1000_100091


namespace NUMINAMATH_CALUDE_expansion_coefficient_x_squared_l1000_100026

/-- The coefficient of x^2 in the expansion of (1 + x + x^(1/2018))^10 -/
def coefficient_x_squared : ℕ :=
  Nat.choose 10 2

theorem expansion_coefficient_x_squared :
  coefficient_x_squared = 45 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_x_squared_l1000_100026


namespace NUMINAMATH_CALUDE_pascal_and_coin_toss_l1000_100009

/-- Pascal's Triangle row sum -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Probability of k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

theorem pascal_and_coin_toss :
  pascal_row_sum 10 = 1024 ∧
  binomial_probability 10 5 (1/2) = 63/256 := by sorry

end NUMINAMATH_CALUDE_pascal_and_coin_toss_l1000_100009


namespace NUMINAMATH_CALUDE_complement_union_M_N_l1000_100006

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_M_N :
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l1000_100006


namespace NUMINAMATH_CALUDE_bus_seat_difference_l1000_100060

/-- Represents the seating configuration of a bus --/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  regular_seat_capacity : ℕ
  total_capacity : ℕ

/-- Theorem about the difference in seats between left and right sides of the bus --/
theorem bus_seat_difference (bus : BusSeating) : 
  bus.left_seats = 15 →
  bus.regular_seat_capacity = 3 →
  bus.back_seat_capacity = 8 →
  bus.total_capacity = 89 →
  bus.left_seats > bus.right_seats →
  bus.left_seats - bus.right_seats = 3 := by
  sorry

#check bus_seat_difference

end NUMINAMATH_CALUDE_bus_seat_difference_l1000_100060


namespace NUMINAMATH_CALUDE_cubic_difference_l1000_100084

theorem cubic_difference (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) :
  a^3 - b^3 = 342 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l1000_100084


namespace NUMINAMATH_CALUDE_domino_pigeonhole_l1000_100005

/-- Represents a domino with two halves -/
structure Domino :=
  (half1 : Fin 7)
  (half2 : Fin 7)

/-- Represents the state of dominoes after cutting -/
structure DominoState :=
  (row : List Domino)
  (cut_halves : List (Fin 7))

/-- The theorem statement -/
theorem domino_pigeonhole 
  (dominoes : List Domino)
  (h1 : dominoes.length = 28)
  (h2 : ∀ i : Fin 7, (dominoes.map Domino.half1 ++ dominoes.map Domino.half2).count i = 7)
  (state : DominoState)
  (h3 : state.row.length = 26)
  (h4 : state.cut_halves.length = 4)
  (h5 : ∀ d ∈ dominoes, d ∈ state.row ∨ (d.half1 ∈ state.cut_halves ∧ d.half2 ∈ state.cut_halves)) :
  ∃ i j : Fin 4, i ≠ j ∧ state.cut_halves[i] = state.cut_halves[j] :=
sorry

end NUMINAMATH_CALUDE_domino_pigeonhole_l1000_100005


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1000_100024

/-- Given a rectangle with perimeter 160 feet and length twice its width,
    the maximum area that can be enclosed is 12800/9 square feet. -/
theorem max_rectangle_area (w : ℝ) (l : ℝ) (h1 : w > 0) (h2 : l > 0) 
    (h3 : 2 * w + 2 * l = 160) (h4 : l = 2 * w) : w * l ≤ 12800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1000_100024


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l1000_100023

theorem lcm_gcd_relation (n : ℕ+) : 
  Nat.lcm n.val 180 = Nat.gcd n.val 180 + 360 → n.val = 450 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l1000_100023


namespace NUMINAMATH_CALUDE_base_7_even_digits_403_l1000_100051

/-- Counts the number of even digits in a base-7 number -/
def countEvenDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 -/
def toBase7 (n : ℕ) : List ℕ := sorry

theorem base_7_even_digits_403 :
  let base7Repr := toBase7 403
  countEvenDigitsBase7 403 = 1 := by sorry

end NUMINAMATH_CALUDE_base_7_even_digits_403_l1000_100051


namespace NUMINAMATH_CALUDE_emily_chocolate_sales_l1000_100045

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (bars_left : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - bars_left) * price_per_bar

/-- Proves that Emily makes $20 from selling chocolate bars -/
theorem emily_chocolate_sales : money_made 8 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_chocolate_sales_l1000_100045


namespace NUMINAMATH_CALUDE_count_less_equal_04_l1000_100014

def count_less_equal (threshold : ℚ) (numbers : List ℚ) : ℕ :=
  (numbers.filter (λ x => x ≤ threshold)).length

theorem count_less_equal_04 : count_less_equal (4/10) [8/10, 1/2, 3/10] = 1 := by
  sorry

end NUMINAMATH_CALUDE_count_less_equal_04_l1000_100014


namespace NUMINAMATH_CALUDE_distance_AC_proof_l1000_100021

/-- The distance between two cities A and C, given specific travel conditions. -/
def distance_AC : ℝ := 17.5

/-- The speed of the truck in km/h. -/
def truck_speed : ℝ := 50

/-- The distance traveled by delivery person A before meeting the truck, in km. -/
def distance_A_meeting : ℝ := 3

/-- The time between the meeting point and arrival at C, in hours. -/
def time_after_meeting : ℝ := 0.2  -- 12 minutes = 0.2 hours

/-- Theorem stating the distance between cities A and C under given conditions. -/
theorem distance_AC_proof :
  ∃ (speed_delivery : ℝ),
    speed_delivery > 0 ∧
    distance_AC = truck_speed * (time_after_meeting + distance_A_meeting / truck_speed) :=
by sorry


end NUMINAMATH_CALUDE_distance_AC_proof_l1000_100021


namespace NUMINAMATH_CALUDE_bellas_bistro_purchase_l1000_100028

/-- The cost of a sandwich at Bella's Bistro -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Bella's Bistro -/
def soda_cost : ℕ := 1

/-- The number of sandwiches to be purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas to be purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase at Bella's Bistro -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem bellas_bistro_purchase :
  total_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_bellas_bistro_purchase_l1000_100028


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1000_100036

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 3 = 0, 
    its center is at (1, -2) and its radius is √2 -/
theorem circle_center_and_radius :
  let f : ℝ × ℝ → ℝ := λ (x, y) => x^2 + y^2 - 2*x + 4*y + 3
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), f p = 0 ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1000_100036


namespace NUMINAMATH_CALUDE_inconsistent_game_statistics_l1000_100098

theorem inconsistent_game_statistics :
  ∀ (total_games : ℕ) (first_part_games : ℕ) (win_percentage : ℚ),
  total_games = 75 →
  first_part_games = 100 →
  (0 : ℚ) ≤ win_percentage ∧ win_percentage ≤ 1 →
  ¬(∃ (first_part_win_percentage : ℚ) (remaining_win_percentage : ℚ),
    first_part_win_percentage * (first_part_games : ℚ) / (total_games : ℚ) +
    remaining_win_percentage * ((total_games - first_part_games) : ℚ) / (total_games : ℚ) = win_percentage ∧
    remaining_win_percentage = 1/2 ∧
    (0 : ℚ) ≤ first_part_win_percentage ∧ first_part_win_percentage ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_inconsistent_game_statistics_l1000_100098


namespace NUMINAMATH_CALUDE_triangle_side_length_l1000_100087

theorem triangle_side_length (A B C : ℝ) (angleA angleB : ℝ) (sideAC : ℝ) :
  angleA = π / 4 →
  angleB = 5 * π / 12 →
  sideAC = 6 →
  ∃ (sideBC : ℝ), sideBC = 6 * (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1000_100087


namespace NUMINAMATH_CALUDE_least_xy_value_l1000_100048

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 96 ∧ ∃ (a b : ℕ+), (a : ℚ) / b + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 ∧ (a * b : ℕ) = 96 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l1000_100048


namespace NUMINAMATH_CALUDE_factor_x12_minus_729_l1000_100013

theorem factor_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) :=
by
  have h : 729 = 3^6 := by norm_num
  sorry

end NUMINAMATH_CALUDE_factor_x12_minus_729_l1000_100013


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l1000_100038

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10

def is_eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ n₀ p, p > 0 ∧ ∀ k ≥ n₀, a (k + p) = a k

theorem sequence_eventually_periodic (a : ℕ → ℕ) (h : is_valid_sequence a) :
  is_eventually_periodic a := by
  sorry

#check sequence_eventually_periodic

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l1000_100038


namespace NUMINAMATH_CALUDE_correct_statements_l1000_100035

theorem correct_statements :
  (abs (-5) = 5) ∧ (-(- 3) = 3) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l1000_100035


namespace NUMINAMATH_CALUDE_circle_center_l1000_100089

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The original equation of the circle -/
def OriginalEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 4

theorem circle_center :
  ∃ (r : ℝ), ∀ (x y : ℝ), OriginalEquation x y ↔ CircleEquation (-4) 2 r x y :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1000_100089


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1000_100019

/-- Given an ellipse with equation 4(x-2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x - 2)^2 + 16 * y^2 = 64 → 
      ((x = C.1 ∧ y = C.2) ∨ (x = D.1 ∧ y = D.2))) →
    (C.1 - 2)^2 / 16 + C.2^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 + D.2^2 / 4 = 1 →
    C.1 ≠ D.1 →
    C.2 ≠ D.2 →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1000_100019


namespace NUMINAMATH_CALUDE_oil_price_reduction_reduced_price_is_30_l1000_100008

/-- Represents the price reduction of oil -/
def price_reduction : ℝ := 0.2

/-- Represents the additional amount of oil obtained after price reduction -/
def additional_oil : ℝ := 4

/-- Represents the total cost -/
def total_cost : ℝ := 600

/-- Calculates the reduced price per kg given the original price -/
def reduced_price (original_price : ℝ) : ℝ :=
  original_price * (1 - price_reduction)

/-- Represents the relationship between original and reduced prices -/
theorem oil_price_reduction (original_amount original_price : ℝ) :
  original_amount * original_price = 
  (original_amount + additional_oil) * (reduced_price original_price) →
  reduced_price original_price = 30 := by
  sorry

/-- Main theorem: Proves that the reduced price per kg is Rs. 30 -/
theorem reduced_price_is_30 :
  ∃ (original_amount original_price : ℝ),
    original_amount * original_price = total_cost ∧
    (original_amount + additional_oil) * (reduced_price original_price) = total_cost ∧
    reduced_price original_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_reduced_price_is_30_l1000_100008


namespace NUMINAMATH_CALUDE_smallest_divisor_power_l1000_100000

def Q (z : ℂ) : ℂ := z^10 + z^9 + z^6 + z^5 + z^4 + z + 1

theorem smallest_divisor_power : 
  ∃! k : ℕ, k > 0 ∧ 
  (∀ z : ℂ, Q z = 0 → z^k = 1) ∧
  (∀ m : ℕ, m > 0 → m < k → ∃ z : ℂ, Q z = 0 ∧ z^m ≠ 1) ∧
  k = 84 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_power_l1000_100000


namespace NUMINAMATH_CALUDE_triangle_product_theorem_l1000_100039

/-- Represent a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represent a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- ABC is an acute triangle -/
def is_acute (t : Triangle) : Prop := sorry

/-- P is the foot of the perpendicular from C to AB -/
def foot_of_perpendicular_C_to_AB (t : Triangle) (P : Point) : Prop := sorry

/-- Q is the foot of the perpendicular from B to AC -/
def foot_of_perpendicular_B_to_AC (t : Triangle) (Q : Point) : Prop := sorry

/-- Line PQ intersects the circumcircle of triangle ABC at points X and Y -/
def line_intersects_circumcircle (t : Triangle) (P Q X Y : Point) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculate the product of lengths AB and AC -/
def product_AB_AC (t : Triangle) : ℝ := sorry

theorem triangle_product_theorem (t : Triangle) (P Q X Y : Point) :
  is_acute t →
  foot_of_perpendicular_C_to_AB t P →
  foot_of_perpendicular_B_to_AC t Q →
  line_intersects_circumcircle t P Q X Y →
  distance X P = 12 →
  distance P Q = 20 →
  distance Q Y = 18 →
  product_AB_AC t = 360 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_product_theorem_l1000_100039


namespace NUMINAMATH_CALUDE_min_value_sum_l1000_100096

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧ 
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1000_100096


namespace NUMINAMATH_CALUDE_system_solutions_solutions_satisfy_system_l1000_100095

/-- The system of equations has only two solutions: (0,0,0) and (1,1,1) -/
theorem system_solutions (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^2 / (1 + x₁^2) = x₂) ∧ 
  (2 * x₂^2 / (1 + x₂^2) = x₃) ∧ 
  (2 * x₃^2 / (1 + x₃^2) = x₁) →
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1)) := by
  sorry

/-- The solutions (0,0,0) and (1,1,1) satisfy the system of equations -/
theorem solutions_satisfy_system : 
  (2 * 0^2 / (1 + 0^2) = 0 ∧ 2 * 0^2 / (1 + 0^2) = 0 ∧ 2 * 0^2 / (1 + 0^2) = 0) ∧
  (2 * 1^2 / (1 + 1^2) = 1 ∧ 2 * 1^2 / (1 + 1^2) = 1 ∧ 2 * 1^2 / (1 + 1^2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_solutions_satisfy_system_l1000_100095


namespace NUMINAMATH_CALUDE_rent_increase_is_thirty_percent_l1000_100068

/-- Calculates the percentage increase in rent given last year's expenses and this year's total increase --/
def rent_increase_percentage (last_year_rent : ℕ) (last_year_food : ℕ) (last_year_insurance : ℕ) (food_increase_percent : ℕ) (insurance_multiplier : ℕ) (total_yearly_increase : ℕ) : ℕ :=
  let last_year_monthly_total := last_year_rent + last_year_food + last_year_insurance
  let this_year_food := last_year_food + (last_year_food * food_increase_percent) / 100
  let this_year_insurance := last_year_insurance * insurance_multiplier
  let monthly_increase_without_rent := (this_year_food + this_year_insurance) - (last_year_food + last_year_insurance)
  let yearly_increase_without_rent := monthly_increase_without_rent * 12
  let rent_increase := total_yearly_increase - yearly_increase_without_rent
  (rent_increase * 100) / (last_year_rent * 12)

theorem rent_increase_is_thirty_percent :
  rent_increase_percentage 1000 200 100 50 3 7200 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_is_thirty_percent_l1000_100068


namespace NUMINAMATH_CALUDE_solve_equation_l1000_100082

theorem solve_equation (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 1 / y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1000_100082


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_l1000_100069

theorem sqrt_product_quotient : (Real.sqrt 3 * Real.sqrt 15) / Real.sqrt 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_l1000_100069


namespace NUMINAMATH_CALUDE_rogers_final_money_rogers_final_money_proof_l1000_100093

/-- Calculates Roger's final amount of money after various transactions -/
theorem rogers_final_money (initial_amount : ℝ) (birthday_money : ℝ) (found_money : ℝ) 
  (game_cost : ℝ) (gift_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_amount + birthday_money + found_money
  let after_game_purchase := total_before_spending - game_cost
  let gift_cost := gift_percentage * after_game_purchase
  let final_amount := after_game_purchase - gift_cost
  final_amount

/-- Proves that Roger's final amount of money is $106.25 -/
theorem rogers_final_money_proof :
  rogers_final_money 84 56 20 35 0.15 = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_rogers_final_money_rogers_final_money_proof_l1000_100093


namespace NUMINAMATH_CALUDE_max_ab_value_l1000_100046

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * log x + (a + 1) * x - (1/2) * x^2

theorem max_ab_value (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, f a x ≥ -(1/2) * x^2 + a * x + b) →
  (∃ c : ℝ, c = Real.exp 1 / 2 ∧ ∀ b : ℝ, a * b ≤ c) :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1000_100046


namespace NUMINAMATH_CALUDE_average_weight_increase_l1000_100062

theorem average_weight_increase (initial_count : ℕ) (initial_weight : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 89 →
  (initial_count * initial_weight - old_weight + new_weight) / initial_count - initial_weight = 3 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1000_100062


namespace NUMINAMATH_CALUDE_distinct_remainders_l1000_100080

theorem distinct_remainders (n : ℕ+) :
  ∀ (i j : Fin n), i ≠ j →
    (2 * i.val + 1) ^ (2 * i.val + 1) % (2 ^ n.val) ≠
    (2 * j.val + 1) ^ (2 * j.val + 1) % (2 ^ n.val) := by
  sorry

end NUMINAMATH_CALUDE_distinct_remainders_l1000_100080


namespace NUMINAMATH_CALUDE_exists_m_divides_polynomial_l1000_100007

theorem exists_m_divides_polynomial (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ (m^3 + m^2 - 2*m - 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divides_polynomial_l1000_100007


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l1000_100018

theorem fixed_point_parabola (k : ℝ) : 
  225 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l1000_100018


namespace NUMINAMATH_CALUDE_hidden_dots_count_l1000_100067

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of dots on all faces of a standard die -/
def sumOfDots : ℕ := 21

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 3

/-- The visible faces on the stack of dice -/
def visibleFaces : List ℕ := [1, 3, 4, 5, 6]

/-- The total number of faces on the stack of dice -/
def totalFaces : ℕ := 18

/-- The number of hidden faces on the stack of dice -/
def hiddenFaces : ℕ := 13

/-- Theorem stating that the total number of hidden dots is 44 -/
theorem hidden_dots_count :
  (numberOfDice * sumOfDots) - (visibleFaces.sum) = 44 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l1000_100067


namespace NUMINAMATH_CALUDE_resort_group_combinations_l1000_100022

theorem resort_group_combinations : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_resort_group_combinations_l1000_100022


namespace NUMINAMATH_CALUDE_probability_three_black_face_cards_l1000_100086

theorem probability_three_black_face_cards (total_cards : ℕ) (drawn_cards : ℕ) 
  (black_face_cards : ℕ) (non_black_face_cards : ℕ) :
  total_cards = 36 →
  drawn_cards = 6 →
  black_face_cards = 8 →
  non_black_face_cards = 28 →
  (Nat.choose black_face_cards 3 * Nat.choose non_black_face_cards 3) / 
  Nat.choose total_cards drawn_cards = 11466 / 121737 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_black_face_cards_l1000_100086


namespace NUMINAMATH_CALUDE_third_face_area_is_60_l1000_100055

/-- Represents a cuboidal box with given dimensions -/
structure CuboidalBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The area of the first adjacent face -/
def first_face_area (box : CuboidalBox) : ℝ := box.length * box.width

/-- The area of the second adjacent face -/
def second_face_area (box : CuboidalBox) : ℝ := box.width * box.height

/-- The area of the third adjacent face -/
def third_face_area (box : CuboidalBox) : ℝ := box.length * box.height

/-- The volume of the box -/
def volume (box : CuboidalBox) : ℝ := box.length * box.width * box.height

/-- Theorem stating the area of the third face given the conditions -/
theorem third_face_area_is_60 (box : CuboidalBox) 
  (h1 : first_face_area box = 120)
  (h2 : second_face_area box = 72)
  (h3 : volume box = 720) :
  third_face_area box = 60 := by
  sorry


end NUMINAMATH_CALUDE_third_face_area_is_60_l1000_100055


namespace NUMINAMATH_CALUDE_rogers_expenses_l1000_100029

theorem rogers_expenses (A : ℝ) (m s p : ℝ) : 
  (m = 0.25 * (A - s - p)) →
  (s = 0.1 * (A - m - p)) →
  (p = 0.05 * (A - m - s)) →
  (A > 0) →
  (m > 0) →
  (s > 0) →
  (p > 0) →
  (abs ((m + s + p) / A - 0.32) < 0.005) := by
sorry

end NUMINAMATH_CALUDE_rogers_expenses_l1000_100029


namespace NUMINAMATH_CALUDE_mode_best_for_market_share_l1000_100050

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Mean
  | Median
  | Mode
  | Variance

/-- Represents a shoe factory -/
structure ShoeFactory where
  survey_data : List Nat  -- List of shoe sizes from the survey

/-- Determines the most appropriate statistical measure for increasing market share -/
def best_measure_for_market_share (factory : ShoeFactory) : StatisticalMeasure :=
  StatisticalMeasure.Mode

/-- Theorem stating that the mode is the most appropriate measure for increasing market share -/
theorem mode_best_for_market_share (factory : ShoeFactory) :
  best_measure_for_market_share factory = StatisticalMeasure.Mode := by
  sorry


end NUMINAMATH_CALUDE_mode_best_for_market_share_l1000_100050
