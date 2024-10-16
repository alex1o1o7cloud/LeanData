import Mathlib

namespace NUMINAMATH_CALUDE_mechanic_worked_five_and_half_hours_l1809_180911

/-- Calculates the number of hours a mechanic worked given the total cost, part costs, labor rate, and break time. -/
def mechanic_work_hours (total_cost parts_cost labor_rate_per_minute break_minutes : ℚ) : ℚ :=
  ((total_cost - parts_cost) / labor_rate_per_minute - break_minutes) / 60

/-- Proves that the mechanic worked 5.5 hours given the problem conditions. -/
theorem mechanic_worked_five_and_half_hours :
  let total_cost : ℚ := 220
  let parts_cost : ℚ := 2 * 20
  let labor_rate_per_minute : ℚ := 0.5
  let break_minutes : ℚ := 30
  mechanic_work_hours total_cost parts_cost labor_rate_per_minute break_minutes = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_mechanic_worked_five_and_half_hours_l1809_180911


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1809_180900

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1809_180900


namespace NUMINAMATH_CALUDE_fraction_problem_l1809_180981

theorem fraction_problem (x : ℝ) : 
  (0.3 * x = 63.0000000000001) → 
  (∃ f : ℝ, f = 0.4 * x + 12 ∧ f = 96) :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1809_180981


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1809_180916

theorem arithmetic_calculation : 5 + 15 / 3 - 2^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1809_180916


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1809_180940

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ x^2 + x - 2 < 0}

theorem complement_of_A_in_U :
  {x | x ∈ U ∧ x ∉ A} = {-2, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1809_180940


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1809_180948

/-- Given a reflection of point (-2, 3) across line y = mx + b to point (4, -5), prove m + b = -1 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 4 ∧ y = -5 ∧ 
    (x - (-2))^2 + (y - 3)^2 = (x - (-2))^2 + (m * (x - (-2)) + b - 3)^2 ∧
    y = m * x + b) →
  m + b = -1 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1809_180948


namespace NUMINAMATH_CALUDE_platform_length_l1809_180999

/-- Given a train of length 300 m that crosses a platform in 39 seconds
    and a signal pole in 36 seconds, the length of the platform is 25 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 36) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 25 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1809_180999


namespace NUMINAMATH_CALUDE_parabola_circle_area_ratio_l1809_180978

/-- The ratio of areas S1 to S2 for a parabola and tangent circle -/
theorem parabola_circle_area_ratio 
  (d : ℝ) 
  (hd : d > 0) : 
  let K : ℝ → ℝ := fun x ↦ (1/d) * x^2
  let P : ℝ × ℝ := (d, d)
  let Q : ℝ × ℝ := (0, d)
  let S1 : ℝ := ∫ x in (0)..(d), (d - K x)
  let S2 : ℝ := ∫ x in (0)..(d), (d - K x)
  S1 / S2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_area_ratio_l1809_180978


namespace NUMINAMATH_CALUDE_solve_problem_l1809_180986

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^(digits.length - 1 - i)) 0

def problem : Prop :=
  let base_6_num := [5, 4, 3, 2, 1, 0]
  let base_7_num := [4, 3, 2, 1, 0]
  (base_to_decimal base_6_num 6) - (base_to_decimal base_7_num 7) = 34052

theorem solve_problem : problem := by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l1809_180986


namespace NUMINAMATH_CALUDE_cube_of_negative_product_l1809_180906

theorem cube_of_negative_product (a b : ℝ) : (-2 * a * b) ^ 3 = -8 * a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_product_l1809_180906


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1809_180912

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1809_180912


namespace NUMINAMATH_CALUDE_odd_digits_base4_345_l1809_180934

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 345₁₀ is 3 -/
theorem odd_digits_base4_345 : countOddDigits (toBase4 345) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_345_l1809_180934


namespace NUMINAMATH_CALUDE_range_of_m_l1809_180968

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (m ≥ 4 ∨ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1809_180968


namespace NUMINAMATH_CALUDE_thirteen_step_staircase_l1809_180942

/-- 
Represents a staircase where each step is made of toothpicks following an arithmetic sequence.
The first step uses 3 toothpicks, and each subsequent step uses 2 more toothpicks than the previous one.
-/
def Staircase (n : ℕ) : ℕ := n * (n + 2)

/-- A staircase with 5 steps uses 55 toothpicks -/
axiom five_step_staircase : Staircase 5 = 55

/-- Theorem: A staircase with 13 steps uses 210 toothpicks -/
theorem thirteen_step_staircase : Staircase 13 = 210 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_step_staircase_l1809_180942


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1809_180958

def is_divisible_by_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

def unit_digit (n : ℕ) : ℕ := n % 10

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

def uniquely_determined_by_divisors (n : ℕ) : Prop :=
  ∀ m : ℕ, m < 60 → unit_digit m = unit_digit n → num_divisors m = num_divisors n → m = n

theorem unique_n_satisfying_conditions :
  ∃! n : ℕ, n < 60 ∧
    is_divisible_by_two_primes n ∧
    uniquely_determined_by_divisors n ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1809_180958


namespace NUMINAMATH_CALUDE_rainfall_difference_calculation_l1809_180936

/-- Represents the rainfall data for the first three days of May --/
structure RainfallData where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  average : ℝ

/-- Calculates the difference between the average rainfall and the actual rainfall --/
def rainfallDifference (data : RainfallData) : ℝ :=
  data.average - (data.day1 + data.day2 + data.day3)

/-- Theorem stating the rainfall difference for the given data --/
theorem rainfall_difference_calculation (data : RainfallData) 
  (h1 : data.day1 = 26)
  (h2 : data.day2 = 34)
  (h3 : data.day3 = data.day2 - 12)
  (h4 : data.average = 140) :
  rainfallDifference data = 58 := by
  sorry

#eval rainfallDifference { day1 := 26, day2 := 34, day3 := 22, average := 140 }

end NUMINAMATH_CALUDE_rainfall_difference_calculation_l1809_180936


namespace NUMINAMATH_CALUDE_orange_ribbons_l1809_180963

theorem orange_ribbons (total : ℚ) (black : ℕ) : 
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + black = total →
  black = 40 →
  (1/6 : ℚ) * total = 80/3 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l1809_180963


namespace NUMINAMATH_CALUDE_houses_with_neither_l1809_180935

theorem houses_with_neither (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 65 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 10 :=
by sorry

end NUMINAMATH_CALUDE_houses_with_neither_l1809_180935


namespace NUMINAMATH_CALUDE_kyle_total_laps_l1809_180959

-- Define the number of laps jogged in P.E. class
def pe_laps : ℝ := 1.12

-- Define the number of laps jogged during track practice
def track_laps : ℝ := 2.12

-- Define the total number of laps
def total_laps : ℝ := pe_laps + track_laps

-- Theorem statement
theorem kyle_total_laps : total_laps = 3.24 := by
  sorry

end NUMINAMATH_CALUDE_kyle_total_laps_l1809_180959


namespace NUMINAMATH_CALUDE_initial_quarters_l1809_180998

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- The total value of coins in cents -/
def total_value (dimes nickels quarters : ℕ) : ℕ :=
  dimes * coin_value "dime" + nickels * coin_value "nickel" + quarters * coin_value "quarter"

theorem initial_quarters (initial_dimes initial_nickels mom_quarters : ℕ) 
  (total_cents : ℕ) (h1 : initial_dimes = 4) (h2 : initial_nickels = 7) 
  (h3 : mom_quarters = 5) (h4 : total_cents = 300) :
  ∃ initial_quarters : ℕ, 
    total_value initial_dimes initial_nickels (initial_quarters + mom_quarters) = total_cents ∧ 
    initial_quarters = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_quarters_l1809_180998


namespace NUMINAMATH_CALUDE_problem_solution_l1809_180930

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 5

-- Theorem statement
theorem problem_solution :
  -- Condition 1: y-1 is directly proportional to x+2
  (∃ k : ℝ, ∀ x y : ℝ, y = f x → y - 1 = k * (x + 2)) ∧
  -- Condition 2: When x=1, y=7
  (f 1 = 7) ∧
  -- Solution 1: The function f satisfies the conditions
  (∀ x : ℝ, f x = 2 * x + 5) ∧
  -- Solution 2: The point (-7/2, -2) lies on the graph of f
  (f (-7/2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1809_180930


namespace NUMINAMATH_CALUDE_contrapositive_correct_l1809_180913

-- Define the proposition p
def p (passing_score : ℝ) (A_passes B_passes C_passes : Prop) : Prop :=
  (passing_score < 70) → (¬A_passes ∧ ¬B_passes ∧ ¬C_passes)

-- Define the contrapositive of p
def contrapositive_p (passing_score : ℝ) (A_passes B_passes C_passes : Prop) : Prop :=
  (A_passes ∨ B_passes ∨ C_passes) → (passing_score ≥ 70)

-- Theorem stating that contrapositive_p is indeed the contrapositive of p
theorem contrapositive_correct (passing_score : ℝ) (A_passes B_passes C_passes : Prop) :
  contrapositive_p passing_score A_passes B_passes C_passes ↔
  (¬p passing_score A_passes B_passes C_passes → False) → False :=
sorry

end NUMINAMATH_CALUDE_contrapositive_correct_l1809_180913


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l1809_180918

theorem complex_magnitude_one (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w^2 + 1/w^2 = s) : 
  Complex.abs w = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l1809_180918


namespace NUMINAMATH_CALUDE_trig_identity_l1809_180945

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1809_180945


namespace NUMINAMATH_CALUDE_constant_speed_travel_time_l1809_180903

/-- Given a constant speed, if a 120-mile trip takes 3 hours, then a 200-mile trip takes 5 hours. -/
theorem constant_speed_travel_time 
  (speed : ℝ) 
  (h₁ : speed > 0) 
  (h₂ : 120 / speed = 3) : 
  200 / speed = 5 := by
sorry

end NUMINAMATH_CALUDE_constant_speed_travel_time_l1809_180903


namespace NUMINAMATH_CALUDE_infinite_solutions_l1809_180964

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The first equation: 3x - 4y = 10 -/
def equation1 (p : Point) : Prop := 3 * p.x - 4 * p.y = 10

/-- The second equation: 9x - 12y = 30 -/
def equation2 (p : Point) : Prop := 9 * p.x - 12 * p.y = 30

/-- A solution satisfies both equations -/
def is_solution (p : Point) : Prop := equation1 p ∧ equation2 p

/-- The set of all solutions -/
def solution_set : Set Point := {p | is_solution p}

/-- The theorem stating that there are infinitely many solutions -/
theorem infinite_solutions : Set.Infinite solution_set := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1809_180964


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1809_180960

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Represents a point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem: Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (P Q : HyperbolaPoint h) (F₁ : ℝ × ℝ) :
  (∃ (line : ℝ → ℝ × ℝ), 
    line 0 = right_focus h ∧ 
    (∃ t₁ t₂, line t₁ = (P.x, P.y) ∧ line t₂ = (Q.x, Q.y)) ∧
    ((P.x - Q.x) * (P.x - F₁.1) + (P.y - Q.y) * (P.y - F₁.2) = 0) ∧
    ((P.x - Q.x)^2 + (P.y - Q.y)^2 = (P.x - F₁.1)^2 + (P.y - F₁.2)^2)) →
  eccentricity h = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1809_180960


namespace NUMINAMATH_CALUDE_circle_equation_from_line_intersection_l1809_180907

/-- Given a line in polar coordinates that intersects the polar axis, 
    this theorem proves the equation of a circle centered at the intersection point. -/
theorem circle_equation_from_line_intersection (ρ θ : ℝ) :
  (ρ * Real.cos (θ + π/4) = Real.sqrt 2) →
  ∃ C : ℝ × ℝ,
    (C.1 = 2 ∧ C.2 = 0) ∧
    (∀ (ρ' θ' : ℝ), (ρ' * Real.cos θ' - C.1)^2 + (ρ' * Real.sin θ' - C.2)^2 = 1 ↔
                     ρ'^2 - 4*ρ'*Real.cos θ' + 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_from_line_intersection_l1809_180907


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l1809_180972

theorem contrapositive_real_roots (m : ℝ) :
  (¬(∃ x : ℝ, x^2 = m) → m < 0) ↔
  (m ≥ 0 → ∃ x : ℝ, x^2 = m) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l1809_180972


namespace NUMINAMATH_CALUDE_pizza_slices_per_friend_l1809_180975

theorem pizza_slices_per_friend (num_friends : ℕ) (total_slices : ℕ) (h1 : num_friends = 4) (h2 : total_slices = 16) :
  ∃ (slices_per_friend : ℕ),
    slices_per_friend * num_friends = total_slices ∧
    slices_per_friend = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_friend_l1809_180975


namespace NUMINAMATH_CALUDE_acid_dilution_l1809_180956

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.4 →
  final_concentration = 0.25 →
  water_added = 30 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l1809_180956


namespace NUMINAMATH_CALUDE_flower_bed_weeds_count_l1809_180967

/-- The number of weeds in the flower bed -/
def flower_bed_weeds : ℕ := 11

/-- The number of weeds in the vegetable patch -/
def vegetable_patch_weeds : ℕ := 14

/-- The number of weeds in the grass around the fruit trees -/
def grass_weeds : ℕ := 32

/-- The amount Lucille earns per weed in cents -/
def cents_per_weed : ℕ := 6

/-- The cost of the soda in cents -/
def soda_cost : ℕ := 99

/-- The amount of money Lucille has left in cents -/
def money_left : ℕ := 147

theorem flower_bed_weeds_count : 
  flower_bed_weeds = 11 :=
by sorry

end NUMINAMATH_CALUDE_flower_bed_weeds_count_l1809_180967


namespace NUMINAMATH_CALUDE_prob_second_white_given_first_white_l1809_180961

/-- Represents the total number of balls -/
def total_balls : ℕ := 9

/-- Represents the number of white balls -/
def white_balls : ℕ := 5

/-- Represents the number of black balls -/
def black_balls : ℕ := 4

/-- Represents the probability of drawing a white ball first -/
def prob_first_white : ℚ := white_balls / total_balls

/-- Represents the probability of drawing two white balls consecutively -/
def prob_both_white : ℚ := (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))

/-- Theorem stating the probability of drawing a white ball second, given the first was white -/
theorem prob_second_white_given_first_white :
  prob_both_white / prob_first_white = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_second_white_given_first_white_l1809_180961


namespace NUMINAMATH_CALUDE_scientific_notation_110000_l1809_180939

theorem scientific_notation_110000 : 
  110000 = 1.1 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_110000_l1809_180939


namespace NUMINAMATH_CALUDE_bombardment_percentage_approx_10_percent_l1809_180991

def initial_population : ℕ := 8515
def final_population : ℕ := 6514
def departure_rate : ℚ := 15 / 100

def bombardment_percentage : ℚ :=
  (initial_population - final_population) / initial_population * 100

theorem bombardment_percentage_approx_10_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  abs (bombardment_percentage - 10) < ε ∧
  final_population = 
    initial_population * (1 - bombardment_percentage / 100) * (1 - departure_rate) :=
by sorry

end NUMINAMATH_CALUDE_bombardment_percentage_approx_10_percent_l1809_180991


namespace NUMINAMATH_CALUDE_tangent_line_k_values_l1809_180996

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 - x^2 + x

-- Define the tangent line function
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

-- Theorem statement
theorem tangent_line_k_values :
  ∀ k : ℝ, (∃ x : ℝ, f x = tangent_line k x ∧ 
    (∀ y : ℝ, y ≠ x → f y ≠ tangent_line k y)) →
  k = 1 ∨ k = 3/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_k_values_l1809_180996


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1809_180988

/-- An arithmetic sequence is a sequence where the difference between 
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h1 : a 1 + a 4 + a 7 = 45) 
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1809_180988


namespace NUMINAMATH_CALUDE_selections_with_paperback_count_l1809_180953

/-- The number of books on the shelf -/
def total_books : ℕ := 7

/-- The number of paperback books -/
def paperbacks : ℕ := 2

/-- The number of hardback books -/
def hardbacks : ℕ := 5

/-- The number of possible selections that include at least one paperback -/
def selections_with_paperback : ℕ := 96

/-- Theorem stating that the number of selections with at least one paperback
    is equal to the total number of possible selections minus the number of
    selections with no paperbacks -/
theorem selections_with_paperback_count :
  selections_with_paperback = 2^total_books - 2^hardbacks :=
by sorry

end NUMINAMATH_CALUDE_selections_with_paperback_count_l1809_180953


namespace NUMINAMATH_CALUDE_soccer_games_total_l1809_180933

theorem soccer_games_total (win_percentage : ℝ) (games_won : ℕ) (h1 : win_percentage = 0.65) (h2 : games_won = 182) :
  (games_won : ℝ) / win_percentage = 280 := by
  sorry

end NUMINAMATH_CALUDE_soccer_games_total_l1809_180933


namespace NUMINAMATH_CALUDE_no_four_digit_square_palindromes_l1809_180929

/-- A function that checks if a natural number is a 4-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that checks if a natural number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

/-- Theorem stating that there are no 4-digit square numbers that are palindromes -/
theorem no_four_digit_square_palindromes : 
  ¬∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_four_digit_square_palindromes_l1809_180929


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l1809_180950

theorem trigonometric_expression_value (m x : ℝ) (h : m * Real.tan x = 2) :
  (6 * m * Real.sin (2 * x) + 2 * m * Real.cos (2 * x)) /
  (m * Real.cos (2 * x) - 3 * m * Real.sin (2 * x)) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l1809_180950


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1809_180927

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1809_180927


namespace NUMINAMATH_CALUDE_sum_base4_equals_l1809_180954

/-- Convert a base 4 number to its decimal representation -/
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Convert a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Addition of two base 4 numbers -/
def addBase4 (a b : List Nat) : List Nat :=
  decimalToBase4 (base4ToDecimal a + base4ToDecimal b)

theorem sum_base4_equals : 
  addBase4 (addBase4 [3, 0, 2] [2, 1, 1]) [0, 3, 3] = [0, 1, 1, 3, 1] := by
  sorry


end NUMINAMATH_CALUDE_sum_base4_equals_l1809_180954


namespace NUMINAMATH_CALUDE_vector_CQ_equals_2p_l1809_180995

-- Define the space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points
variable (A B C P Q : V)

-- Define vector p
variable (p : V)

-- Conditions
variable (h1 : P ∈ interior (triangle A B C))
variable (h2 : A - P + 2 • (B - P) + 3 • (C - P) = 0)
variable (h3 : ∃ t : ℝ, Q = C + t • (P - C) ∧ Q ∈ line_through A B)
variable (h4 : C - P = p)

-- Theorem to prove
theorem vector_CQ_equals_2p : C - Q = 2 • p := by sorry

end NUMINAMATH_CALUDE_vector_CQ_equals_2p_l1809_180995


namespace NUMINAMATH_CALUDE_neil_cookies_l1809_180931

theorem neil_cookies (total : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  remaining = 12 ∧ given_away = (2 : ℕ) * total / 5 ∧ remaining = (3 : ℕ) * total / 5 → total = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_neil_cookies_l1809_180931


namespace NUMINAMATH_CALUDE_junk_mail_delivery_l1809_180983

/-- Calculates the total pieces of junk mail delivered given the number of houses with white and red mailboxes -/
def total_junk_mail (total_houses : ℕ) (white_mailboxes : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ) : ℕ :=
  (white_mailboxes + red_mailboxes) * mail_per_house

/-- Proves that the total junk mail delivered is 30 pieces given the specified conditions -/
theorem junk_mail_delivery :
  total_junk_mail 8 2 3 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_delivery_l1809_180983


namespace NUMINAMATH_CALUDE_total_carrots_is_40_l1809_180977

/-- The number of carrots grown by Joan -/
def joan_carrots : ℕ := 29

/-- The number of carrots grown by Jessica -/
def jessica_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := joan_carrots + jessica_carrots

/-- Theorem stating that the total number of carrots grown is 40 -/
theorem total_carrots_is_40 : total_carrots = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_is_40_l1809_180977


namespace NUMINAMATH_CALUDE_total_payment_l1809_180973

def payment_structure (year1 year2 year3 year4 : ℕ) : Prop :=
  year1 = 20 ∧ 
  year2 = year1 + 2 ∧ 
  year3 = year2 + 3 ∧ 
  year4 = year3 + 4

theorem total_payment (year1 year2 year3 year4 : ℕ) :
  payment_structure year1 year2 year3 year4 →
  year1 + year2 + year3 + year4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_l1809_180973


namespace NUMINAMATH_CALUDE_square_area_difference_l1809_180952

theorem square_area_difference (area_A : ℝ) (side_diff : ℝ) : 
  area_A = 25 → side_diff = 4 → 
  let side_A := Real.sqrt area_A
  let side_B := side_A + side_diff
  side_B ^ 2 = 81 := by
sorry

end NUMINAMATH_CALUDE_square_area_difference_l1809_180952


namespace NUMINAMATH_CALUDE_range_of_f_when_a_neg_four_range_of_a_when_two_roots_l1809_180925

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4^x + a * 2^x + 3

theorem range_of_f_when_a_neg_four :
  ∀ x ∈ Set.Icc 0 2, ∃ y ∈ Set.Icc (-1) 3, f (-4) x = y :=
sorry

theorem range_of_a_when_two_roots :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  a ∈ Set.Ioo (-4) (-2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_when_a_neg_four_range_of_a_when_two_roots_l1809_180925


namespace NUMINAMATH_CALUDE_linear_transformation_uniqueness_l1809_180962

theorem linear_transformation_uniqueness (z₁ z₂ w₁ w₂ : ℂ) 
  (h₁ : z₁ ≠ z₂) (h₂ : w₁ ≠ w₂) :
  ∃! (a b : ℂ), (a * z₁ + b = w₁) ∧ (a * z₂ + b = w₂) := by
  sorry

end NUMINAMATH_CALUDE_linear_transformation_uniqueness_l1809_180962


namespace NUMINAMATH_CALUDE_inequality_proof_l1809_180938

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x^3 + x*y^2 + 2*x*y ≤ 2*x^2*y + x^2 + x + y :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1809_180938


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_4_l1809_180990

theorem greatest_four_digit_divisible_by_3_and_4 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 3 ∣ n ∧ 4 ∣ n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_4_l1809_180990


namespace NUMINAMATH_CALUDE_remainder_sum_mod_11_l1809_180932

theorem remainder_sum_mod_11 (a b c : ℕ) : 
  1 ≤ a ∧ a ≤ 10 →
  1 ≤ b ∧ b ≤ 10 →
  1 ≤ c ∧ c ≤ 10 →
  (a * b * c) % 11 = 2 →
  (7 * c) % 11 = 3 →
  (8 * b) % 11 = (4 + b) % 11 →
  (a + b + c) % 11 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_11_l1809_180932


namespace NUMINAMATH_CALUDE_amy_pencil_count_l1809_180989

/-- The number of pencils Amy has after buying and giving away some pencils -/
def final_pencil_count (initial : ℕ) (bought_monday : ℕ) (bought_tuesday : ℕ) (given_away : ℕ) : ℕ :=
  initial + bought_monday + bought_tuesday - given_away

/-- Theorem stating that Amy has 12 pencils at the end -/
theorem amy_pencil_count : final_pencil_count 3 7 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_amy_pencil_count_l1809_180989


namespace NUMINAMATH_CALUDE_orange_juice_cartons_bought_l1809_180904

def prove_orange_juice_cartons : Nat :=
  let initial_money : Nat := 86
  let bread_loaves : Nat := 3
  let bread_cost : Nat := 3
  let juice_cost : Nat := 6
  let remaining_money : Nat := 59
  let spent_money : Nat := initial_money - remaining_money
  let bread_total_cost : Nat := bread_loaves * bread_cost
  let juice_total_cost : Nat := spent_money - bread_total_cost
  juice_total_cost / juice_cost

theorem orange_juice_cartons_bought :
  prove_orange_juice_cartons = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_cartons_bought_l1809_180904


namespace NUMINAMATH_CALUDE_circle_condition_l1809_180974

theorem circle_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0) ↔ (k > 4 ∨ k < -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l1809_180974


namespace NUMINAMATH_CALUDE_lucy_lovely_age_difference_l1809_180997

/-- Represents the ages of Lucy and Lovely at different points in time -/
structure Ages where
  lucy_current : ℕ
  lovely_current : ℕ
  years_ago : ℕ

/-- Conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  a.lucy_current = 50 ∧
  a.lucy_current - a.years_ago = 3 * (a.lovely_current - a.years_ago) ∧
  a.lucy_current + 10 = 2 * (a.lovely_current + 10)

/-- Theorem stating the solution to the problem -/
theorem lucy_lovely_age_difference :
  ∃ (a : Ages), problem_conditions a ∧ a.years_ago = 5 :=
sorry

end NUMINAMATH_CALUDE_lucy_lovely_age_difference_l1809_180997


namespace NUMINAMATH_CALUDE_soccer_team_strikers_l1809_180970

theorem soccer_team_strikers (goalies defenders midfielders strikers total : ℕ) : 
  goalies = 3 →
  defenders = 10 →
  midfielders = 2 * defenders →
  total = 40 →
  strikers = total - (goalies + defenders + midfielders) →
  strikers = 7 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_strikers_l1809_180970


namespace NUMINAMATH_CALUDE_quadrupled_base_exponent_l1809_180928

theorem quadrupled_base_exponent (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  (4 * c)^(4 * d) = (c^d * y^d)^2 → y = 16 * c := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_exponent_l1809_180928


namespace NUMINAMATH_CALUDE_triangle_rigidity_connected_beams_rigidity_l1809_180965

-- Define a structure for a triangle with three sides
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

-- Define a property for a triangle to be rigid
def is_rigid (t : Triangle) : Prop :=
  ∀ (t' : Triangle), t.side1 = t'.side1 ∧ t.side2 = t'.side2 ∧ t.side3 = t'.side3 →
    t = t'

-- Theorem stating that a triangle with fixed side lengths is rigid
theorem triangle_rigidity (t : Triangle) :
  is_rigid t :=
sorry

-- Define a beam as a line segment with fixed length
def Beam := ℝ

-- Define a structure for the connected beams
structure ConnectedBeams :=
  (beam1 : Beam)
  (beam2 : Beam)
  (beam3 : Beam)

-- Function to convert connected beams to a triangle
def beams_to_triangle (b : ConnectedBeams) : Triangle :=
  { side1 := b.beam1,
    side2 := b.beam2,
    side3 := b.beam3 }

-- Theorem stating that connected beams with fixed lengths form a rigid structure
theorem connected_beams_rigidity (b : ConnectedBeams) :
  is_rigid (beams_to_triangle b) :=
sorry

end NUMINAMATH_CALUDE_triangle_rigidity_connected_beams_rigidity_l1809_180965


namespace NUMINAMATH_CALUDE_sum_of_squares_l1809_180992

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 28) : x^2 + y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1809_180992


namespace NUMINAMATH_CALUDE_rectangle_strip_problem_l1809_180914

theorem rectangle_strip_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43) :
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_strip_problem_l1809_180914


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_and_b_l1809_180937

/-- A function f(x) = ax³ + bx has an extreme value of -2 at x = 1 -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => a * x^3 + b * x
  f 1 = -2 ∧ (deriv f) 1 = 0

/-- Theorem: If f(x) = ax³ + bx has an extreme value of -2 at x = 1, then a = 1 and b = -3 -/
theorem extreme_value_implies_a_and_b :
  ∀ a b : ℝ, has_extreme_value a b → a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_and_b_l1809_180937


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1809_180947

/-- Proves that the equation 3(x+1)² = 2(x+1) is equivalent to a quadratic equation in the standard form ax² + bx + c = 0, where a ≠ 0 -/
theorem equation_is_quadratic (x : ℝ) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (3 * (x + 1)^2 = 2 * (x + 1)) ↔ (a * x^2 + b * x + c = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l1809_180947


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_96_l1809_180908

/-- The square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  let (x1, y1) := c1_center
  let (x2, y2) := c2_center
  -- Definition of the function, to be implemented
  0

/-- The theorem stating the square of the distance between intersection points -/
theorem intersection_distance_squared_is_96 :
  intersection_distance_squared (3, 2) (3, -4) 5 7 = 96 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_is_96_l1809_180908


namespace NUMINAMATH_CALUDE_range_of_m_l1809_180993

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∀ y ∈ Set.Icc (-10) (-6), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1809_180993


namespace NUMINAMATH_CALUDE_annual_piano_clarinet_cost_difference_l1809_180979

/-- Calculates the difference in annual cost between piano and clarinet lessons --/
def annual_lesson_cost_difference (clarinet_hourly_rate piano_hourly_rate : ℕ) 
  (clarinet_weekly_hours piano_weekly_hours : ℕ) (weeks_per_year : ℕ) : ℕ :=
  ((piano_hourly_rate * piano_weekly_hours) - (clarinet_hourly_rate * clarinet_weekly_hours)) * weeks_per_year

/-- Proves that the difference in annual cost between piano and clarinet lessons is $1040 --/
theorem annual_piano_clarinet_cost_difference : 
  annual_lesson_cost_difference 40 28 3 5 52 = 1040 := by
  sorry

end NUMINAMATH_CALUDE_annual_piano_clarinet_cost_difference_l1809_180979


namespace NUMINAMATH_CALUDE_quadratic_value_bound_l1809_180923

theorem quadratic_value_bound (a b : ℝ) : ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ |x^2 + a*x + b| ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_bound_l1809_180923


namespace NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_parabola_properties_l1809_180926

/-- Ellipse properties -/
theorem ellipse_properties (x y : ℝ) :
  x^2 / 4 + y^2 = 1 →
  ∃ (a b : ℝ), a = 2 * b ∧ a > 0 ∧ b > 0 ∧
    x^2 / a^2 + y^2 / b^2 = 1 ∧
    (2 : ℝ)^2 / a^2 + 0^2 / b^2 = 1 :=
sorry

/-- Hyperbola properties -/
theorem hyperbola_properties (x y : ℝ) :
  y^2 / 20 - x^2 / 16 = 1 →
  ∃ (a b : ℝ), a = 2 * Real.sqrt 5 ∧ a > 0 ∧ b > 0 ∧
    y^2 / a^2 - x^2 / b^2 = 1 ∧
    5^2 / a^2 - 2^2 / b^2 = 1 :=
sorry

/-- Parabola properties -/
theorem parabola_properties (x y : ℝ) :
  y^2 = 4 * x →
  ∃ (p : ℝ), p > 0 ∧
    y^2 = 4 * p * x ∧
    (-2)^2 = 4 * p * 1 ∧
    (∀ (x₀ y₀ : ℝ), y₀^2 = 4 * p * x₀ → x₀ = 0 → y₀ = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_parabola_properties_l1809_180926


namespace NUMINAMATH_CALUDE_geralds_bag_contains_40_apples_l1809_180902

/-- The number of bags Pam has -/
def pams_bags : ℕ := 10

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- The number of apples in each of Gerald's bags -/
def geralds_bag_apples : ℕ := pams_total_apples / (3 * pams_bags)

/-- Theorem stating that each of Gerald's bags contains 40 apples -/
theorem geralds_bag_contains_40_apples : geralds_bag_apples = 40 := by
  sorry

end NUMINAMATH_CALUDE_geralds_bag_contains_40_apples_l1809_180902


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1809_180917

/-- Proves that the length of a rectangular plot is 62 meters given the specified conditions -/
theorem rectangular_plot_length : ∀ (breadth length perimeter : ℝ),
  length = breadth + 24 →
  perimeter = 2 * (length + breadth) →
  perimeter * 26.5 = 5300 →
  length = 62 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1809_180917


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l1809_180944

theorem smallest_n_with_conditions : ∃ (m a : ℕ),
  145^2 = m^3 - (m-1)^3 + 5 ∧
  2*145 + 117 = a^2 ∧
  ∀ (n : ℕ), n > 0 → n < 145 →
    (∀ (m' a' : ℕ), n^2 ≠ m'^3 - (m'-1)^3 + 5 ∨ 2*n + 117 ≠ a'^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l1809_180944


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l1809_180920

/-- The cost of Amanda's kitchen upgrade --/
def kitchen_upgrade_cost (cabinet_knobs_count : ℕ) (cabinet_knob_price : ℚ) 
                         (drawer_pulls_count : ℕ) (drawer_pull_price : ℚ) : ℚ :=
  (cabinet_knobs_count : ℚ) * cabinet_knob_price + (drawer_pulls_count : ℚ) * drawer_pull_price

/-- Theorem stating that the cost of Amanda's kitchen upgrade is $77.00 --/
theorem amanda_kitchen_upgrade_cost : 
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 :=
by sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l1809_180920


namespace NUMINAMATH_CALUDE_triangle_height_calculation_l1809_180949

/-- Given a triangle with area 615 m² and one side of 123 meters, 
    the length of the perpendicular dropped on this side from the opposite vertex is 10 meters. -/
theorem triangle_height_calculation (A : ℝ) (b h : ℝ) 
    (h_area : A = 615)
    (h_base : b = 123)
    (h_triangle_area : A = (b * h) / 2) : h = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_calculation_l1809_180949


namespace NUMINAMATH_CALUDE_water_drinking_ratio_l1809_180943

/-- Proof of the water drinking ratio problem -/
theorem water_drinking_ratio :
  let morning_water : ℝ := 1.5
  let total_water : ℝ := 6
  let afternoon_water : ℝ := total_water - morning_water
  afternoon_water / morning_water = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_drinking_ratio_l1809_180943


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l1809_180919

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (home_to_grocery : ℝ) 
  (grocery_to_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : home_to_grocery = 100) 
  (h2 : grocery_to_gym = 180) 
  (h3 : time_difference = 40) :
  let v := home_to_grocery / ((grocery_to_gym / 2) / time_difference + home_to_grocery)
  2 * v = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_angelina_walking_speed_l1809_180919


namespace NUMINAMATH_CALUDE_zhuge_liang_army_count_l1809_180955

theorem zhuge_liang_army_count : 
  let n := 8
  let sum := n + n^2 + n^3 + n^4 + n^5 + n^6
  sum = (1 / 7) * (n^7 - n) := by
  sorry

end NUMINAMATH_CALUDE_zhuge_liang_army_count_l1809_180955


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1809_180921

theorem no_positive_integer_solutions :
  ¬∃ (x y z : ℕ+), x^3 + 2*y^3 = 4*z^3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1809_180921


namespace NUMINAMATH_CALUDE_calculate_expression_l1809_180909

theorem calculate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^3) = (3 / 4) * y^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1809_180909


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1809_180969

theorem fraction_sum_equality (x y z : ℝ) 
  (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1809_180969


namespace NUMINAMATH_CALUDE_end_at_multiple_of_4_probability_l1809_180966

/-- Represents the possible moves on the spinner -/
inductive SpinnerMove
| Left2 : SpinnerMove
| Right2 : SpinnerMove
| Right1 : SpinnerMove

/-- The probability of a specific move on the spinner -/
def spinnerProbability (move : SpinnerMove) : ℚ :=
  match move with
  | SpinnerMove.Left2 => 1/4
  | SpinnerMove.Right2 => 1/2
  | SpinnerMove.Right1 => 1/4

/-- The set of cards Jeff can pick from -/
def cardSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

/-- Whether a number is a multiple of 4 -/
def isMultipleOf4 (n : ℕ) : Prop := ∃ k, n = 4 * k

/-- The probability of ending at a multiple of 4 -/
def probEndAtMultipleOf4 : ℚ := 1/32

theorem end_at_multiple_of_4_probability : 
  probEndAtMultipleOf4 = 1/32 :=
sorry

end NUMINAMATH_CALUDE_end_at_multiple_of_4_probability_l1809_180966


namespace NUMINAMATH_CALUDE_triangle_problem_l1809_180941

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (a - c) * (a + c) * Real.sin C = c * (b - c) * Real.sin B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  Real.sin B * Real.sin C = 1/4 →
  A = π/3 ∧ a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1809_180941


namespace NUMINAMATH_CALUDE_point_D_value_l1809_180946

/-- The number corresponding to point D on a number line, given that:
    - A corresponds to 5
    - B corresponds to 8
    - C corresponds to -10
    - The sum of the four numbers remains unchanged when the direction of the number line is reversed
-/
def point_D : ℝ := -3

/-- The sum of the numbers corresponding to points A, B, C, and D -/
def sum_forward (d : ℝ) : ℝ := 5 + 8 + (-10) + d

/-- The sum of the numbers corresponding to points A, B, C, and D when the direction is reversed -/
def sum_reversed (d : ℝ) : ℝ := (-5) + (-8) + 10 + (-d)

/-- Theorem stating that point D corresponds to -3 -/
theorem point_D_value : 
  sum_forward point_D = sum_reversed point_D :=
by sorry

end NUMINAMATH_CALUDE_point_D_value_l1809_180946


namespace NUMINAMATH_CALUDE_min_sum_floor_l1809_180980

theorem min_sum_floor (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a+b+c)/d⌋ + ⌊(a+b+d)/c⌋ + ⌊(a+c+d)/b⌋ + ⌊(b+c+d)/a⌋ ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_floor_l1809_180980


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1809_180951

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers,
    with at least one object in each container. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 6 ways to distribute 5 scoops into 3 flavors
    with at least one scoop of each flavor. -/
theorem ice_cream_flavors :
  distribute_with_minimum 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1809_180951


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1809_180987

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
    sheep * 7 = horses * 5 →
    horses * 230 = 12880 →
    sheep = 40 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1809_180987


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1809_180901

theorem quadratic_minimum (c : ℝ) : 
  (1/3 : ℝ) * c^2 + 6*c + 4 ≥ (1/3 : ℝ) * (-9)^2 + 6*(-9) + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1809_180901


namespace NUMINAMATH_CALUDE_michaels_class_size_l1809_180994

theorem michaels_class_size (b : ℕ) : 
  (100 < b ∧ b < 200) ∧ 
  (∃ k : ℕ, b = 4 * k - 2) ∧ 
  (∃ l : ℕ, b = 5 * l - 3) ∧ 
  (∃ m : ℕ, b = 6 * m - 4) →
  (b = 122 ∨ b = 182) := by
sorry

end NUMINAMATH_CALUDE_michaels_class_size_l1809_180994


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1809_180957

theorem polynomial_simplification (x : ℝ) :
  (15 * x^13 + 10 * x^12 + 7 * x^11) + (3 * x^15 + 2 * x^13 + x^11 + 4 * x^9 + 2 * x^5 + 6) =
  3 * x^15 + 17 * x^13 + 10 * x^12 + 8 * x^11 + 4 * x^9 + 2 * x^5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1809_180957


namespace NUMINAMATH_CALUDE_f_at_two_equals_one_fourth_l1809_180905

/-- Given a function f(x) = 2^x + 2^(-x) - 4, prove that f(2) = 1/4 -/
theorem f_at_two_equals_one_fourth :
  let f : ℝ → ℝ := λ x ↦ 2^x + 2^(-x) - 4
  f 2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_equals_one_fourth_l1809_180905


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1809_180984

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 6) : 
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1809_180984


namespace NUMINAMATH_CALUDE_janet_lives_l1809_180910

theorem janet_lives (initial_lives lost_lives gained_lives : ℕ) :
  initial_lives ≥ lost_lives →
  initial_lives - lost_lives + gained_lives =
    initial_lives + gained_lives - lost_lives :=
by
  sorry

#check janet_lives 38 16 32

end NUMINAMATH_CALUDE_janet_lives_l1809_180910


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1809_180971

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x ≤ 6 →
  x + 4*x > 20 →
  4*x + 20 > x →
  x + 20 > 4*x →
  (∀ y : ℕ, y > 0 → y ≤ 6 → y + 4*y > 20 → 4*y + 20 > y → y + 20 > 4*y → x + 4*x + 20 ≥ y + 4*y + 20) →
  x + 4*x + 20 = 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1809_180971


namespace NUMINAMATH_CALUDE_friend_team_assignment_count_l1809_180976

theorem friend_team_assignment_count : 
  let n_friends : ℕ := 8
  let n_teams : ℕ := 4
  n_teams ^ n_friends = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignment_count_l1809_180976


namespace NUMINAMATH_CALUDE_daps_to_dips_l1809_180985

/-- The number of daps equivalent to one dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dops equivalent to one dip -/
def dops_per_dip : ℚ := 3 / 11

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 66

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_to_dips : daps_per_dop * dops_per_dip⁻¹ * target_dips = 45 / 2 :=
by sorry

end NUMINAMATH_CALUDE_daps_to_dips_l1809_180985


namespace NUMINAMATH_CALUDE_intersection_complement_l1809_180922

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement : S ∩ (U \ T) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_l1809_180922


namespace NUMINAMATH_CALUDE_third_generation_tail_length_l1809_180915

/-- The tail length growth factor between generations -/
def growth_factor : ℝ := 1.25

/-- The initial tail length of the first generation in centimeters -/
def initial_length : ℝ := 16

/-- The tail length of the nth generation -/
def tail_length (n : ℕ) : ℝ := initial_length * growth_factor ^ n

theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end NUMINAMATH_CALUDE_third_generation_tail_length_l1809_180915


namespace NUMINAMATH_CALUDE_move_down_two_units_l1809_180924

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moving a point down in a Cartesian coordinate system -/
def moveDown (p : Point) (distance : ℝ) : Point :=
  ⟨p.x, p.y - distance⟩

/-- Theorem: Moving a point (a,b) down 2 units results in (a,b-2) -/
theorem move_down_two_units (a b : ℝ) :
  moveDown ⟨a, b⟩ 2 = ⟨a, b - 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_move_down_two_units_l1809_180924


namespace NUMINAMATH_CALUDE_exponent_division_l1809_180982

theorem exponent_division (a : ℝ) : 2 * a^3 / a = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1809_180982
