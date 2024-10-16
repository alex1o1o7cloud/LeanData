import Mathlib

namespace NUMINAMATH_CALUDE_fraction_evaluation_l1453_145364

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1453_145364


namespace NUMINAMATH_CALUDE_friday_pushups_equal_total_l1453_145378

def monday_pushups : ℕ := 5
def tuesday_pushups : ℕ := 7
def wednesday_pushups : ℕ := 2 * tuesday_pushups
def thursday_pushups : ℕ := (monday_pushups + tuesday_pushups + wednesday_pushups) / 2
def total_monday_to_thursday : ℕ := monday_pushups + tuesday_pushups + wednesday_pushups + thursday_pushups

theorem friday_pushups_equal_total : total_monday_to_thursday = 39 := by
  sorry

end NUMINAMATH_CALUDE_friday_pushups_equal_total_l1453_145378


namespace NUMINAMATH_CALUDE_intersection_point_polar_coordinates_l1453_145326

/-- The intersection point of ρ = 2sinθ and ρ = 2cosθ in polar coordinates -/
theorem intersection_point_polar_coordinates :
  ∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < π/2 ∧ 
  ρ = 2 * Real.sin θ ∧ ρ = 2 * Real.cos θ ∧
  ρ = Real.sqrt 2 ∧ θ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_polar_coordinates_l1453_145326


namespace NUMINAMATH_CALUDE_books_written_proof_l1453_145302

/-- The number of books written by Zig -/
def zig_books : ℕ := 60

/-- The number of books written by Flo -/
def flo_books : ℕ := zig_books / 4

/-- The number of books written by Tim -/
def tim_books : ℕ := flo_books / 2

/-- The total number of books written by Zig, Flo, and Tim -/
def total_books : ℕ := zig_books + flo_books + tim_books

theorem books_written_proof : total_books = 82 := by
  sorry

end NUMINAMATH_CALUDE_books_written_proof_l1453_145302


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1453_145346

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1453_145346


namespace NUMINAMATH_CALUDE_optimal_plan_is_best_l1453_145355

/-- Represents a bus purchasing plan -/
structure BusPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a bus plan is valid according to the given constraints -/
def isValidPlan (plan : BusPlan) : Prop :=
  plan.typeA + plan.typeB = 10 ∧
  100 * plan.typeA + 150 * plan.typeB ≤ 1200 ∧
  60 * plan.typeA + 100 * plan.typeB ≥ 680

/-- Calculates the total cost of a bus plan in million RMB -/
def totalCost (plan : BusPlan) : ℕ :=
  100 * plan.typeA + 150 * plan.typeB

/-- The optimal bus purchasing plan -/
def optimalPlan : BusPlan :=
  { typeA := 8, typeB := 2 }

/-- Theorem stating that the optimal plan is valid and minimizes the total cost -/
theorem optimal_plan_is_best :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
sorry

#check optimal_plan_is_best

end NUMINAMATH_CALUDE_optimal_plan_is_best_l1453_145355


namespace NUMINAMATH_CALUDE_sector_properties_l1453_145329

/-- Proves that a circular sector with perimeter 4 and area 1 has radius 1 and central angle 2 -/
theorem sector_properties :
  ∀ r θ : ℝ,
  r > 0 →
  θ > 0 →
  2 * r + θ * r = 4 →
  1 / 2 * θ * r^2 = 1 →
  r = 1 ∧ θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_properties_l1453_145329


namespace NUMINAMATH_CALUDE_no_water_overflow_l1453_145367

/-- Represents the dimensions and properties of a cylindrical container and an iron block. -/
structure ContainerProblem where
  container_depth : ℝ
  container_outer_diameter : ℝ
  container_wall_thickness : ℝ
  water_depth : ℝ
  block_diameter : ℝ
  block_height : ℝ

/-- Calculates the volume of water that will overflow when an iron block is placed in a cylindrical container. -/
noncomputable def water_overflow (p : ContainerProblem) : ℝ :=
  let container_inner_radius := (p.container_outer_diameter - 2 * p.container_wall_thickness) / 2
  let initial_water_volume := Real.pi * container_inner_radius ^ 2 * p.water_depth
  let container_max_volume := Real.pi * container_inner_radius ^ 2 * p.container_depth
  let block_volume := Real.pi * (p.block_diameter / 2) ^ 2 * p.block_height
  let new_total_volume := container_max_volume - block_volume
  max (initial_water_volume - new_total_volume) 0

/-- Theorem stating that no water will overflow in the given problem. -/
theorem no_water_overflow : 
  let problem : ContainerProblem := {
    container_depth := 30,
    container_outer_diameter := 22,
    container_wall_thickness := 1,
    water_depth := 27.5,
    block_diameter := 10,
    block_height := 30
  }
  water_overflow problem = 0 := by sorry

end NUMINAMATH_CALUDE_no_water_overflow_l1453_145367


namespace NUMINAMATH_CALUDE_prob_red_third_eq_147_1000_l1453_145359

/-- A fair 10-sided die with exactly 3 red sides -/
structure RedDie :=
  (sides : Nat)
  (red_sides : Nat)
  (h_sides : sides = 10)
  (h_red : red_sides = 3)

/-- The probability of rolling a non-red side -/
def prob_non_red (d : RedDie) : ℚ :=
  (d.sides - d.red_sides : ℚ) / d.sides

/-- The probability of rolling a red side -/
def prob_red (d : RedDie) : ℚ :=
  d.red_sides / d.sides

/-- The probability of rolling a red side for the first time on the third roll -/
def prob_red_third (d : RedDie) : ℚ :=
  (prob_non_red d) * (prob_non_red d) * (prob_red d)

theorem prob_red_third_eq_147_1000 (d : RedDie) : 
  prob_red_third d = 147 / 1000 := by sorry

end NUMINAMATH_CALUDE_prob_red_third_eq_147_1000_l1453_145359


namespace NUMINAMATH_CALUDE_cookies_per_batch_l1453_145392

theorem cookies_per_batch (family_members : ℕ) (batches : ℕ) (chips_per_cookie : ℕ) (chips_per_member : ℕ)
  (h1 : family_members = 4)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 2)
  (h4 : chips_per_member = 18) :
  (chips_per_member * family_members) / (chips_per_cookie * batches) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_batch_l1453_145392


namespace NUMINAMATH_CALUDE_sunday_calorie_intake_l1453_145393

-- Define the calorie content for base meals
def breakfast_calories : ℝ := 500
def lunch_calories : ℝ := breakfast_calories * 1.25
def dinner_calories : ℝ := lunch_calories * 2
def snack_calories : ℝ := lunch_calories * 0.7
def morning_snack_calories : ℝ := breakfast_calories + 200
def afternoon_snack_calories : ℝ := lunch_calories * 0.8
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Define the total calories for each day
def monday_calories : ℝ := breakfast_calories + lunch_calories + dinner_calories + snack_calories
def tuesday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories
def wednesday_calories : ℝ := breakfast_calories + lunch_calories + (dinner_calories * 0.85) + dessert_calories
def thursday_calories : ℝ := tuesday_calories
def friday_calories : ℝ := wednesday_calories + (2 * energy_drink_calories)
def weekend_calories : ℝ := tuesday_calories

-- Theorem to prove
theorem sunday_calorie_intake : weekend_calories = 3575 := by
  sorry

end NUMINAMATH_CALUDE_sunday_calorie_intake_l1453_145393


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l1453_145381

theorem integer_triple_divisibility :
  ∀ a b c : ℤ,
  (1 < a ∧ a < b ∧ b < c) →
  ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l1453_145381


namespace NUMINAMATH_CALUDE_constant_sequence_l1453_145340

theorem constant_sequence (a : Fin 2016 → ℕ) 
  (h1 : ∀ i, a i ≤ 2016)
  (h2 : ∀ i j : Fin 2016, (i.val + j.val) ∣ (i.val * a i + j.val * a j)) :
  ∀ i j : Fin 2016, a i = a j :=
by sorry

end NUMINAMATH_CALUDE_constant_sequence_l1453_145340


namespace NUMINAMATH_CALUDE_keith_grew_six_turnips_l1453_145330

/-- The number of turnips Alyssa grew -/
def alyssas_turnips : ℕ := 9

/-- The total number of turnips Keith and Alyssa grew together -/
def total_turnips : ℕ := 15

/-- The number of turnips Keith grew -/
def keiths_turnips : ℕ := total_turnips - alyssas_turnips

theorem keith_grew_six_turnips : keiths_turnips = 6 := by
  sorry

end NUMINAMATH_CALUDE_keith_grew_six_turnips_l1453_145330


namespace NUMINAMATH_CALUDE_monic_quartic_specific_values_l1453_145388

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f (-3) = -9)
  (h4 : f 5 = -25) :
  f 2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_specific_values_l1453_145388


namespace NUMINAMATH_CALUDE_function_value_at_cos_15_deg_l1453_145334

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 1

theorem function_value_at_cos_15_deg :
  f (Real.cos (15 * π / 180)) = -(Real.sqrt 3 / 2) - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_value_at_cos_15_deg_l1453_145334


namespace NUMINAMATH_CALUDE_shelby_rain_time_l1453_145312

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  speed_sunny : ℝ  -- Speed when not raining (miles per hour)
  speed_rainy : ℝ  -- Speed when raining (miles per hour)
  total_time : ℝ   -- Total journey time (minutes)
  stop_time : ℝ    -- Total stop time (minutes)
  total_distance : ℝ -- Total distance covered (miles)

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating that given Shelby's driving conditions, she drove 48 minutes in the rain -/
theorem shelby_rain_time (scenario : DrivingScenario) 
  (h1 : scenario.speed_sunny = 40)
  (h2 : scenario.speed_rainy = 25)
  (h3 : scenario.total_time = 75)
  (h4 : scenario.stop_time = 15)
  (h5 : scenario.total_distance = 28) :
  time_in_rain scenario = 48 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rain_time_l1453_145312


namespace NUMINAMATH_CALUDE_f_has_three_roots_l1453_145327

def f (x : ℝ) := x^3 - 64*x

theorem f_has_three_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry

end NUMINAMATH_CALUDE_f_has_three_roots_l1453_145327


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l1453_145394

/-- The area of the circle inscribed in a right triangle with perimeter 2p and hypotenuse c is π(p - c)². -/
theorem inscribed_circle_area (p c : ℝ) (h1 : 0 < p) (h2 : 0 < c) (h3 : c < 2 * p) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y + c = 2 * p ∧
  ∃ (r : ℝ), r > 0 ∧ r = p - c ∧
  ∃ (S : ℝ), S = π * r^2 ∧ S = π * (p - c)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_area_l1453_145394


namespace NUMINAMATH_CALUDE_election_votes_proof_l1453_145336

theorem election_votes_proof (V : ℕ) (W L : ℕ) : 
  (W > L) →  -- Winner has more votes than loser
  (W - L = (V : ℚ) * (1 / 5)) →  -- Winner's margin is 20% of total votes
  ((L + 1000) - (W - 1000) = (V : ℚ) * (1 / 5)) →  -- Loser would win by 20% if 1000 votes change
  V = 5000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_proof_l1453_145336


namespace NUMINAMATH_CALUDE_monic_quartic_problem_l1453_145345

-- Define a monic quartic polynomial
def monicQuartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_problem (p : ℝ → ℝ) 
  (h_monic : monicQuartic p)
  (h1 : p 1 = 3)
  (h2 : p 2 = 7)
  (h3 : p 3 = 13)
  (h4 : p 4 = 21) :
  p 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_problem_l1453_145345


namespace NUMINAMATH_CALUDE_value_of_x_l1453_145354

theorem value_of_x : ∀ (x y z w v : ℕ),
  x = y + 7 →
  y = z + 12 →
  z = w + 25 →
  w = v + 5 →
  v = 90 →
  x = 139 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1453_145354


namespace NUMINAMATH_CALUDE_prob_aces_or_kings_correct_l1453_145375

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The number of aces in the deck -/
def num_aces : ℕ := 5

/-- The number of kings in the deck -/
def num_kings : ℕ := 4

/-- The probability of drawing either two aces or at least one king -/
def prob_aces_or_kings : ℚ := 104 / 663

theorem prob_aces_or_kings_correct :
  let prob_two_aces := (num_aces * (num_aces - 1)) / (deck_size * (deck_size - 1))
  let prob_one_king := 2 * (num_kings * (deck_size - num_kings)) / (deck_size * (deck_size - 1))
  let prob_two_kings := (num_kings * (num_kings - 1)) / (deck_size * (deck_size - 1))
  prob_two_aces + prob_one_king + prob_two_kings = prob_aces_or_kings := by
  sorry

end NUMINAMATH_CALUDE_prob_aces_or_kings_correct_l1453_145375


namespace NUMINAMATH_CALUDE_average_of_numbers_l1453_145399

def numbers : List ℕ := [1, 2, 4, 5, 6, 9, 9, 10, 12, 12]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 7 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1453_145399


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1453_145324

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 - 3 * Complex.I) = 31/13 + 29/13 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1453_145324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1453_145377

def arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : arithmetic_sequence a q)
  (h2 : q > 1)
  (h3 : a 3 * a 7 = 72)
  (h4 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1453_145377


namespace NUMINAMATH_CALUDE_abc_sum_l1453_145350

theorem abc_sum (a b c : ℕ+) 
  (h1 : a * b + c = 57)
  (h2 : b * c + a = 57)
  (h3 : a * c + b = 57) : 
  a + b + c = 9 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l1453_145350


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l1453_145342

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem tomorrow_is_saturday 
  (h : advanceDay (advanceDay DayOfWeek.Wednesday 2) 5 = DayOfWeek.Monday) :
  nextDay DayOfWeek.Friday = DayOfWeek.Saturday :=
by
  sorry


end NUMINAMATH_CALUDE_tomorrow_is_saturday_l1453_145342


namespace NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l1453_145396

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of the decimal representation of a rational number -/
def decimal_period (q : ℚ) : ℕ := sorry

/-- The count of a specific digit in one period of the decimal representation -/
def digit_count_in_period (q : ℚ) (d : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a specific digit from the decimal representation -/
def digit_probability (q : ℚ) (d : ℕ) : ℚ :=
  (digit_count_in_period q d : ℚ) / (decimal_period q : ℚ)

theorem probability_of_two_in_three_elevenths :
  digit_probability (3/11) 2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l1453_145396


namespace NUMINAMATH_CALUDE_solve_for_a_l1453_145331

def U (a : ℝ) : Set ℝ := {2, 4, 1-a}
def A (a : ℝ) : Set ℝ := {2, a^2-a+2}

theorem solve_for_a (a : ℝ) : 
  (U a \ A a = {-1}) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1453_145331


namespace NUMINAMATH_CALUDE_circle_area_from_PQ_l1453_145308

-- Define the points P and Q
def P : ℝ × ℝ := (1, 3)
def Q : ℝ × ℝ := (5, 8)

-- Define the circle based on the diameter endpoints
def circle_from_diameter (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x : ℝ × ℝ | ∃ (c : ℝ × ℝ), (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p.1 - q.1)^2 + (p.2 - q.2)^2) / 4}

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_area_from_PQ : 
  circle_area (circle_from_diameter P Q) = 41 * π / 4 := by sorry

end NUMINAMATH_CALUDE_circle_area_from_PQ_l1453_145308


namespace NUMINAMATH_CALUDE_inequality_proof_l1453_145352

theorem inequality_proof (x : ℝ) : 2/3 < x ∧ x < 5/4 → (4*x - 5)^2 + (3*x - 2)^2 < (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1453_145352


namespace NUMINAMATH_CALUDE_simplify_expression_l1453_145348

theorem simplify_expression : (5^8 + 3^7)*(0^5 - (-1)^5)^10 = 392812 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1453_145348


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l1453_145386

theorem interest_difference_theorem (P : ℝ) : 
  P * ((1 + 0.1)^2 - 1) - P * 0.1 * 2 = 36 → P = 3600 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l1453_145386


namespace NUMINAMATH_CALUDE_first_month_sale_l1453_145387

/-- Proves that the sale in the first month was 6235, given the sales for months 2-6
    and the desired average sale for 6 months. -/
theorem first_month_sale
  (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ) (average : ℕ)
  (h1 : sale_2 = 6927)
  (h2 : sale_3 = 6855)
  (h3 : sale_4 = 7230)
  (h4 : sale_5 = 6562)
  (h5 : sale_6 = 5191)
  (h6 : average = 6500) :
  6235 = 6 * average - (sale_2 + sale_3 + sale_4 + sale_5 + sale_6) :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l1453_145387


namespace NUMINAMATH_CALUDE_least_integer_abs_inequality_l1453_145316

theorem least_integer_abs_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y + 4| ≤ 25 → x ≤ y) ∧ |3*x + 4| ≤ 25 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_abs_inequality_l1453_145316


namespace NUMINAMATH_CALUDE_exists_fib_divisible_by_10_8_l1453_145309

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem exists_fib_divisible_by_10_8 :
  ∃ k : ℕ, k ≤ 10000000000000002 ∧ fib k % (10^8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_by_10_8_l1453_145309


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l1453_145306

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- A point bisects a chord of the ellipse -/
def bisects_chord (x y : ℝ) (px py : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_ellipse x1 y1 ∧
    is_on_ellipse x2 y2 ∧
    px = (x1 + x2) / 2 ∧
    py = (y1 + y2) / 2

/-- The equation of a line -/
def on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem ellipse_chord_theorem :
  ∀ (x y : ℝ),
    is_on_ellipse x y →
    bisects_chord x y 4 2 →
    on_line x y 1 2 (-8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l1453_145306


namespace NUMINAMATH_CALUDE_range_of_t_l1453_145374

-- Define the solution set of (a-1)^x > 1
def solution_set (a : ℝ) : Set ℝ := {x | x < 0}

-- Define the inequality q
def q (a t : ℝ) : Prop := a^2 - 2*t*a + t^2 - 1 < 0

-- Define the negation of p
def not_p (a : ℝ) : Prop := a ≤ 1 ∨ a ≥ 2

-- Define the negation of q
def not_q (a t : ℝ) : Prop := ¬(q a t)

-- Statement of the theorem
theorem range_of_t :
  (∀ a, solution_set a = {x | x < 0}) →
  (∀ a t, not_p a → not_q a t) →
  (∃ a t, not_p a ∧ q a t) →
  ∀ t, (∀ a, not_p a → not_q a t) → 1 ≤ t ∧ t ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l1453_145374


namespace NUMINAMATH_CALUDE_rachel_chocolate_sales_l1453_145337

/-- The amount of money Rachel made by selling chocolate bars -/
def rachel_money_made (total_bars : ℕ) (price_per_bar : ℚ) (unsold_bars : ℕ) : ℚ :=
  (total_bars - unsold_bars : ℚ) * price_per_bar

/-- Theorem stating that Rachel made $58.50 from selling chocolate bars -/
theorem rachel_chocolate_sales : rachel_money_made 25 3.25 7 = 58.50 := by
  sorry

end NUMINAMATH_CALUDE_rachel_chocolate_sales_l1453_145337


namespace NUMINAMATH_CALUDE_sum_of_two_angles_in_triangle_l1453_145304

/-- Theorem: In a triangle where one angle is 72°, the sum of the other two angles is 108° -/
theorem sum_of_two_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : B = 72) : 
  A + C = 108 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_angles_in_triangle_l1453_145304


namespace NUMINAMATH_CALUDE_log_inequality_l1453_145390

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1453_145390


namespace NUMINAMATH_CALUDE_equivalent_angle_sets_l1453_145344

def angle_set (base : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + base}

theorem equivalent_angle_sets :
  angle_set (-437) = angle_set 283 :=
sorry

end NUMINAMATH_CALUDE_equivalent_angle_sets_l1453_145344


namespace NUMINAMATH_CALUDE_michael_has_two_cats_l1453_145320

/-- The number of dogs Michael has -/
def num_dogs : ℕ := 3

/-- The cost per night per animal for pet-sitting -/
def cost_per_animal : ℕ := 13

/-- The total cost for pet-sitting -/
def total_cost : ℕ := 65

/-- The number of cats Michael has -/
def num_cats : ℕ := (total_cost - num_dogs * cost_per_animal) / cost_per_animal

theorem michael_has_two_cats : num_cats = 2 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_two_cats_l1453_145320


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1453_145341

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Theorem statement
theorem complement_of_A_in_U : Set.compl A = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1453_145341


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l1453_145318

theorem largest_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  (∃ (k : ℕ), (n + 1) * (n + 3) * (n + 7) * (n + 9) * (n + 11) = 15 * k) ∧
  (∀ (m : ℕ), m > 15 → ∃ (n : ℕ), Even n ∧ 0 < n ∧
    ¬(∃ (k : ℕ), (n + 1) * (n + 3) * (n + 7) * (n + 9) * (n + 11) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l1453_145318


namespace NUMINAMATH_CALUDE_count_distinct_digits_eq_2688_l1453_145371

/-- The number of integers between 1000 and 9999 with four distinct digits, none of which is '5' -/
def count_distinct_digits : ℕ :=
  let first_digit := 8  -- 9 digits excluding 5
  let second_digit := 8 -- 9 digits excluding 5 and the first digit
  let third_digit := 7  -- 8 digits excluding 5 and the first two digits
  let fourth_digit := 6 -- 7 digits excluding 5 and the first three digits
  first_digit * second_digit * third_digit * fourth_digit

theorem count_distinct_digits_eq_2688 : count_distinct_digits = 2688 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_digits_eq_2688_l1453_145371


namespace NUMINAMATH_CALUDE_magic_square_solution_l1453_145323

/-- Represents a 3x3 magic square -/
def MagicSquare (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧
  a + b + c = g + h + i ∧
  a + b + c = a + d + g ∧
  a + b + c = b + e + h ∧
  a + b + c = c + f + i ∧
  a + b + c = a + e + i ∧
  a + b + c = c + e + g

theorem magic_square_solution :
  ∃ x : ℤ, MagicSquare 4017 2012 (4017 + 2012 - 4017 - 2012) 4015 (x - 2003) 11 2014 9 x ∧ x = 4003 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l1453_145323


namespace NUMINAMATH_CALUDE_lloyds_hourly_rate_l1453_145356

-- Define Lloyd's regular work hours
def regular_hours : ℝ := 7.5

-- Define Lloyd's overtime multiplier
def overtime_multiplier : ℝ := 1.5

-- Define Lloyd's actual work hours on the given day
def actual_hours : ℝ := 10.5

-- Define Lloyd's total earnings for the day
def total_earnings : ℝ := 42

-- Theorem to prove Lloyd's hourly rate
theorem lloyds_hourly_rate :
  ∃ (rate : ℝ),
    rate * regular_hours + 
    (actual_hours - regular_hours) * rate * overtime_multiplier = total_earnings ∧
    rate = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_lloyds_hourly_rate_l1453_145356


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l1453_145303

theorem restaurant_bill_theorem :
  let num_people : ℕ := 7
  let regular_spend : ℕ := 11
  let num_regular : ℕ := 6
  let extra_spend : ℕ := 6
  let total_spend : ℕ := regular_spend * num_regular + 
    (regular_spend * num_regular + (total_spend / num_people + extra_spend))
  total_spend = 84 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l1453_145303


namespace NUMINAMATH_CALUDE_car_speed_problem_l1453_145372

/-- Two cars start from the same point and travel in opposite directions. -/
structure TwoCars where
  car1_speed : ℝ
  car2_speed : ℝ
  travel_time : ℝ
  total_distance : ℝ

/-- The theorem states that given the conditions of the problem, 
    the speed of the second car is 50 mph. -/
theorem car_speed_problem (cars : TwoCars) 
  (h1 : cars.car1_speed = 40)
  (h2 : cars.travel_time = 5)
  (h3 : cars.total_distance = 450) :
  cars.car2_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1453_145372


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1453_145369

/-- Proves that DeShawn made 12 free-throws given the conditions of the basketball practice problem. -/
theorem basketball_free_throws 
  (deshawn : ℕ) -- DeShawn's free-throws
  (kayla : ℕ) -- Kayla's free-throws
  (annieka : ℕ) -- Annieka's free-throws
  (h1 : kayla = deshawn + deshawn / 2) -- Kayla made 50% more than DeShawn
  (h2 : annieka = kayla - 4) -- Annieka made 4 fewer than Kayla
  (h3 : annieka = 14) -- Annieka made 14 free-throws
  : deshawn = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1453_145369


namespace NUMINAMATH_CALUDE_vegetarians_count_l1453_145343

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def total_vegetarians (f : FamilyDiet) : ℕ :=
  f.only_veg + f.both

/-- Theorem stating that the number of vegetarians in the given family is 31 -/
theorem vegetarians_count (f : FamilyDiet) 
  (h1 : f.only_veg = 19)
  (h2 : f.only_non_veg = 9)
  (h3 : f.both = 12) :
  total_vegetarians f = 31 := by
  sorry

#eval total_vegetarians ⟨19, 9, 12⟩

end NUMINAMATH_CALUDE_vegetarians_count_l1453_145343


namespace NUMINAMATH_CALUDE_set_M_value_l1453_145358

def M : Set ℤ := {a | ∃ (n : ℕ+), 6 / (5 - a) = n}

theorem set_M_value : M = {-1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_M_value_l1453_145358


namespace NUMINAMATH_CALUDE_fraction_order_l1453_145382

theorem fraction_order : 
  let f1 : ℚ := -16/12
  let f2 : ℚ := -18/14
  let f3 : ℚ := -20/15
  f3 = f1 ∧ f1 < f2 := by sorry

end NUMINAMATH_CALUDE_fraction_order_l1453_145382


namespace NUMINAMATH_CALUDE_complex_modulus_two_thirds_plus_three_i_l1453_145311

theorem complex_modulus_two_thirds_plus_three_i :
  Complex.abs (Complex.ofReal (2/3) + Complex.I * 3) = Real.sqrt 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_two_thirds_plus_three_i_l1453_145311


namespace NUMINAMATH_CALUDE_base_k_fraction_l1453_145398

/-- If k is a positive integer and 7/51 equals 0.23̅ₖ in base k, then k equals 16. -/
theorem base_k_fraction (k : ℕ) (h1 : k > 0) 
  (h2 : (7 : ℚ) / 51 = (2 * k + 3 : ℚ) / (k^2 - 1)) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_base_k_fraction_l1453_145398


namespace NUMINAMATH_CALUDE_fraction_reduction_l1453_145370

theorem fraction_reduction (a b c : ℝ) 
  (h : (3*a^2 + 6*a*c - 3*c^2 - 6*a*b) ≠ 0) : 
  (4*a^2 + 2*c^2 - 4*b^2 - 8*b*c) / (3*a^2 + 6*a*c - 3*c^2 - 6*a*b) = 
  (4/3) * ((a-2*b+c)*(a-c)) / ((a-b+c)*(a-b-c)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1453_145370


namespace NUMINAMATH_CALUDE_sum_of_products_is_zero_l1453_145314

theorem sum_of_products_is_zero (a b c : ℝ) :
  (b - c) * (b * c - a^2) + (c - a) * (c * a - b^2) + (a - b) * (a * b - c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_is_zero_l1453_145314


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1453_145368

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ Real.sin (Real.arccos (Real.tanh (Real.arcsin x))) = x ∧ x = Real.sqrt (1/2) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1453_145368


namespace NUMINAMATH_CALUDE_class_size_l1453_145380

theorem class_size (S : ℕ) 
  (h1 : S / 3 + S * 2 / 5 + 12 = S) : S = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1453_145380


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1453_145300

theorem max_value_trig_expression (a b c : ℝ) :
  (⨆ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ)) = Real.sqrt (a^2 + b^2 + 4 * c^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1453_145300


namespace NUMINAMATH_CALUDE_y_derivative_l1453_145310

noncomputable def y (x : ℝ) : ℝ := (1 - x^2) / Real.exp x

theorem y_derivative (x : ℝ) : 
  deriv y x = (x^2 - 2*x - 1) / Real.exp x :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1453_145310


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1453_145365

theorem inequality_equivalence :
  {x : ℝ | |(6 - 2*x + 5) / 4| < 3} = {x : ℝ | -1/2 < x ∧ x < 23/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1453_145365


namespace NUMINAMATH_CALUDE_relationship_equation_l1453_145373

theorem relationship_equation (x : ℝ) : 
  (2023 : ℝ) = (1/4 : ℝ) * x + 1 ↔ 
    (∃ A B : ℝ, A = 2023 ∧ B = x ∧ A = (1/4 : ℝ) * B + 1) :=
by sorry

end NUMINAMATH_CALUDE_relationship_equation_l1453_145373


namespace NUMINAMATH_CALUDE_optimal_start_time_maximizes_minimum_attention_l1453_145349

/-- Represents the attention index of students during a class -/
noncomputable def attentionIndex (x : ℝ) : ℝ :=
  if x ≤ 8 then 2 * x + 68
  else -1/8 * x^2 + 4 * x + 60

/-- The duration of the class in minutes -/
def classDuration : ℝ := 45

/-- The duration of the key explanation in minutes -/
def keyExplanationDuration : ℝ := 24

/-- The optimal start time for the key explanation -/
def optimalStartTime : ℝ := 4

theorem optimal_start_time_maximizes_minimum_attention :
  ∀ t : ℝ, 0 ≤ t ∧ t + keyExplanationDuration ≤ classDuration →
    (∀ x : ℝ, t ≤ x ∧ x ≤ t + keyExplanationDuration →
      attentionIndex x ≥ min (attentionIndex t) (attentionIndex (t + keyExplanationDuration))) →
    t = optimalStartTime := by sorry


end NUMINAMATH_CALUDE_optimal_start_time_maximizes_minimum_attention_l1453_145349


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_bound_l1453_145383

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  -- We don't need to specify the exact structure, just that it exists
  dummy : Unit

/-- The radius of the inscribed sphere of a regular quadrilateral pyramid -/
def inscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  sorry

/-- The radius of the circumscribed sphere of a regular quadrilateral pyramid -/
def circumscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  sorry

/-- The theorem stating that the ratio of the circumscribed sphere radius to the inscribed sphere radius
    is greater than or equal to 1 + √2 for any regular quadrilateral pyramid -/
theorem sphere_radius_ratio_bound (p : RegularQuadrilateralPyramid) :
  circumscribed_sphere_radius p / inscribed_sphere_radius p ≥ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_bound_l1453_145383


namespace NUMINAMATH_CALUDE_puppies_per_cage_l1453_145361

/-- Given a pet store scenario with initial puppies, bought puppies, and cages used,
    calculate the number of puppies per cage. -/
theorem puppies_per_cage
  (initial_puppies : ℝ)
  (bought_puppies : ℝ)
  (cages_used : ℝ)
  (h1 : initial_puppies = 18.0)
  (h2 : bought_puppies = 3.0)
  (h3 : cages_used = 4.2) :
  (initial_puppies + bought_puppies) / cages_used = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l1453_145361


namespace NUMINAMATH_CALUDE_a_nonzero_sufficient_not_necessary_l1453_145351

/-- A cubic polynomial function -/
def cubic_polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The property that a cubic polynomial has a root -/
def has_root (a b c d : ℝ) : Prop := ∃ x : ℝ, cubic_polynomial a b c d x = 0

/-- The statement that "a≠0" is sufficient but not necessary for a cubic polynomial to have a root -/
theorem a_nonzero_sufficient_not_necessary :
  (∀ a b c d : ℝ, a ≠ 0 → has_root a b c d) ∧
  ¬(∀ a b c d : ℝ, has_root a b c d → a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_a_nonzero_sufficient_not_necessary_l1453_145351


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_m_and_n_l1453_145335

theorem sqrt_equality_implies_m_and_n (m n : ℝ) :
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt m - Real.sqrt n →
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_m_and_n_l1453_145335


namespace NUMINAMATH_CALUDE_quadratic_polynomial_k_value_l1453_145313

/-- A polynomial is quadratic if its degree is exactly 2 -/
def IsQuadratic (p : Polynomial ℝ) : Prop :=
  p.degree = 2

theorem quadratic_polynomial_k_value :
  ∀ k : ℝ,
    IsQuadratic (Polynomial.monomial 3 (k - 2) + Polynomial.monomial 2 k + Polynomial.monomial 1 (-2) + Polynomial.monomial 0 (-6))
    → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_k_value_l1453_145313


namespace NUMINAMATH_CALUDE_energy_drink_consumption_l1453_145397

/-- Represents the relationship between relaxation hours and energy drink consumption --/
def inverse_proportional (h g : ℝ) (k : ℝ) : Prop := h * g = k

theorem energy_drink_consumption 
  (h₁ h₂ g₁ g₂ : ℝ) 
  (h₁_pos : h₁ > 0) 
  (h₂_pos : h₂ > 0) 
  (g₁_pos : g₁ > 0) 
  (h₁_val : h₁ = 4) 
  (h₂_val : h₂ = 2) 
  (g₁_val : g₁ = 5) 
  (prop_const : ℝ) 
  (inv_prop₁ : inverse_proportional h₁ g₁ prop_const) 
  (inv_prop₂ : inverse_proportional h₂ g₂ prop_const) : 
  g₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_energy_drink_consumption_l1453_145397


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_162_l1453_145353

theorem percentage_of_360_equals_162 : 
  (162 / 360) * 100 = 45 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_162_l1453_145353


namespace NUMINAMATH_CALUDE_fun_run_signups_l1453_145319

/-- The number of people who signed up for the Fun Run last year -/
def signups_last_year : ℕ := sorry

/-- The number of people who did not show up to run last year -/
def no_shows : ℕ := 40

/-- The number of people running this year -/
def runners_this_year : ℕ := 320

theorem fun_run_signups :
  signups_last_year = 200 ∧
  runners_this_year = 2 * (signups_last_year - no_shows) :=
sorry

end NUMINAMATH_CALUDE_fun_run_signups_l1453_145319


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l1453_145357

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks ∧
    chemistry_marks = 67 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l1453_145357


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1453_145332

-- Define the function g
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (1 / (x + 3))
       else if x < -3 then Int.floor (1 / (x + 3))
       else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1453_145332


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l1453_145384

/-- Given two vectors in a 2D plane with specific magnitudes and angle between them,
    prove that the shorter diagonal of the parallelogram formed by these vectors has length √3. -/
theorem shorter_diagonal_length (a b : ℝ × ℝ) : 
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a • b = 1 →  -- This represents cos(π/3) = 1/2, as |a||b|cos(π/3) = 1
  min (‖a + b‖) (‖a - b‖) = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_shorter_diagonal_length_l1453_145384


namespace NUMINAMATH_CALUDE_train_probability_l1453_145315

/-- John's arrival time in minutes past noon -/
def john_arrival : ℝ := sorry

/-- Train's arrival time in minutes past noon -/
def train_arrival : ℝ := sorry

/-- The train is present at the station for a given time -/
def train_present (t : ℝ) : Prop :=
  train_arrival ≤ t ∧ t ≤ train_arrival + 30

/-- The probability space for John and train arrivals -/
def arrival_space : Set (ℝ × ℝ) :=
  {p | 120 ≤ p.1 ∧ p.1 ≤ 240 ∧ 120 ≤ p.2 ∧ p.2 ≤ 240}

/-- The event where John finds the train at the station -/
def john_meets_train : Set (ℝ × ℝ) :=
  {p ∈ arrival_space | train_present p.1}

/-- The measure of the arrival space -/
noncomputable def arrival_space_measure : ℝ := sorry

/-- The measure of the event where John meets the train -/
noncomputable def john_meets_train_measure : ℝ := sorry

theorem train_probability :
  john_meets_train_measure / arrival_space_measure = 7 / 32 := by sorry

end NUMINAMATH_CALUDE_train_probability_l1453_145315


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l1453_145385

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

/-- Theorem: The given equation is a quadratic equation -/
theorem given_equation_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_given_equation_is_quadratic_l1453_145385


namespace NUMINAMATH_CALUDE_slope_of_line_l1453_145328

theorem slope_of_line (x y : ℝ) :
  4 * x - 7 * y = 14 → (y - (-2)) / (x - 0) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1453_145328


namespace NUMINAMATH_CALUDE_bulls_ploughing_problem_l1453_145347

/-- Represents the number of fields ploughed by a group of bulls -/
def fields_ploughed (bulls : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℚ :=
  (bulls * days * hours_per_day : ℚ) / 15

/-- The problem statement -/
theorem bulls_ploughing_problem :
  let group1_fields := fields_ploughed 10 3 10
  let group2_fields := fields_ploughed 30 2 8
  group2_fields = 32 →
  group1_fields = 20 := by
sorry


end NUMINAMATH_CALUDE_bulls_ploughing_problem_l1453_145347


namespace NUMINAMATH_CALUDE_abs_negative_2023_l1453_145307

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l1453_145307


namespace NUMINAMATH_CALUDE_inverse_sum_equals_golden_ratio_minus_one_l1453_145339

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2 - x else x^3 - 2*x^2 + x

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≤ 0 then (1 + Real.sqrt 5) / 2
  else if y = 1 then 1
  else -2

-- Theorem statement
theorem inverse_sum_equals_golden_ratio_minus_one :
  f_inv (-1) + f_inv 1 + f_inv 4 = (Real.sqrt 5 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_golden_ratio_minus_one_l1453_145339


namespace NUMINAMATH_CALUDE_tank_capacity_l1453_145305

/-- Represents the capacity of a tank and its inlet/outlet pipes. -/
structure TankSystem where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- Theorem stating the capacity of the tank given the conditions. -/
theorem tank_capacity (t : TankSystem)
  (h1 : t.outlet_time = 5)
  (h2 : t.inlet_rate = 4 * 60)  -- 4 litres/min converted to litres/hour
  (h3 : t.combined_time = 8)
  : t.capacity = 3200 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1453_145305


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_equivalence_l1453_145366

theorem polynomial_irreducibility_equivalence 
  (f : Polynomial ℤ) : 
  Irreducible f ↔ Irreducible (f.map (algebraMap ℤ ℚ)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_equivalence_l1453_145366


namespace NUMINAMATH_CALUDE_right_angle_times_l1453_145395

/-- Represents a time on a 12-hour analog clock -/
structure ClockTime where
  hour : Fin 12
  minute : Fin 60

/-- Calculates the angle between hour and minute hands at a given time -/
def angleBetweenHands (time : ClockTime) : ℝ :=
  sorry

/-- Checks if the angle between hands is a right angle (90 degrees) -/
def isRightAngle (time : ClockTime) : Prop :=
  angleBetweenHands time = 90

/-- The theorem stating that when the hands form a right angle, the time is either 3:00 or 9:00 -/
theorem right_angle_times :
  ∀ (time : ClockTime), isRightAngle time →
    (time.hour = 3 ∧ time.minute = 0) ∨ (time.hour = 9 ∧ time.minute = 0) :=
  sorry

end NUMINAMATH_CALUDE_right_angle_times_l1453_145395


namespace NUMINAMATH_CALUDE_impossible_mixture_l1453_145301

/-- Represents the properties of an ingredient -/
structure Ingredient :=
  (volume : ℝ)
  (water_content : ℝ)

/-- Proves that it's impossible to create a mixture with exactly 20% water content
    using the given volumes of tomato juice, tomato paste, and secret sauce -/
theorem impossible_mixture
  (tomato_juice : Ingredient)
  (tomato_paste : Ingredient)
  (secret_sauce : Ingredient)
  (h1 : tomato_juice.volume = 40)
  (h2 : tomato_juice.water_content = 0.9)
  (h3 : tomato_paste.volume = 20)
  (h4 : tomato_paste.water_content = 0.45)
  (h5 : secret_sauce.volume = 10)
  (h6 : secret_sauce.water_content = 0.7)
  : ¬ ∃ (x y z : ℝ),
    0 ≤ x ∧ x ≤ tomato_juice.volume ∧
    0 ≤ y ∧ y ≤ tomato_paste.volume ∧
    0 ≤ z ∧ z ≤ secret_sauce.volume ∧
    (x * tomato_juice.water_content + y * tomato_paste.water_content + z * secret_sauce.water_content) / (x + y + z) = 0.2 :=
sorry


end NUMINAMATH_CALUDE_impossible_mixture_l1453_145301


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l1453_145362

/-- Given (√x + 2/√x)^n, where the binomial coefficients of the second, third, and fourth terms 
    in its expansion form an arithmetic sequence, prove that n = 7 and the expansion does not 
    contain a constant term. -/
theorem binomial_expansion_property (x : ℝ) (n : ℕ) 
  (h : (Nat.choose n 2) * 2 = (Nat.choose n 1) + (Nat.choose n 3)) : 
  (n = 7) ∧ (∀ k : ℕ, 2 * k ≠ n) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l1453_145362


namespace NUMINAMATH_CALUDE_complex_power_six_l1453_145360

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l1453_145360


namespace NUMINAMATH_CALUDE_consecutive_integers_reciprocal_sum_l1453_145338

/-- The sum of reciprocals of all pairs of three consecutive integers is an integer -/
def is_sum_reciprocals_integer (x : ℤ) : Prop :=
  ∃ (n : ℤ), (x / (x + 1) : ℚ) + (x / (x + 2) : ℚ) + ((x + 1) / x : ℚ) + 
             ((x + 1) / (x + 2) : ℚ) + ((x + 2) / x : ℚ) + ((x + 2) / (x + 1) : ℚ) = n

/-- The only sets of three consecutive integers satisfying the condition are {1, 2, 3} and {-3, -2, -1} -/
theorem consecutive_integers_reciprocal_sum :
  ∀ x : ℤ, is_sum_reciprocals_integer x ↔ (x = 1 ∨ x = -3) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_reciprocal_sum_l1453_145338


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l1453_145376

/-- Given a purchase with total cost, sales tax, and tax rate, calculate the cost of tax-free items -/
theorem tax_free_items_cost
  (total_cost : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_cost = 40)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.06)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = total_cost - sales_tax / tax_rate :=
by
  sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l1453_145376


namespace NUMINAMATH_CALUDE_steiner_inellipse_center_distance_l1453_145389

/-- Triangle with vertices (0, 0), (3, 0), and (0, 3/2) -/
def T : Set (ℝ × ℝ) := {(0, 0), (3, 0), (0, 3/2)}

/-- The Steiner inellipse of triangle T -/
def E : Set (ℝ × ℝ) := sorry

/-- The center of the Steiner inellipse E -/
def center_E : ℝ × ℝ := sorry

/-- The distance from the center of E to (0, 0) -/
def distance_to_origin : ℝ := sorry

theorem steiner_inellipse_center_distance :
  distance_to_origin = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_steiner_inellipse_center_distance_l1453_145389


namespace NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l1453_145317

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) ∧
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof is omitted
theorem bob_pennies_proof : bob_pennies 9 31 := by sorry

end NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l1453_145317


namespace NUMINAMATH_CALUDE_product_bound_l1453_145333

theorem product_bound (m : ℕ) (a : ℕ → ℕ) (h1 : ∀ i, i ∈ Finset.range m → a i > 0)
  (h2 : ∀ i, i ∈ Finset.range m → a i ≠ 10)
  (h3 : (Finset.range m).sum a = 10 * m) :
  ((Finset.range m).prod a) ^ (1 / m : ℝ) ≤ 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_product_bound_l1453_145333


namespace NUMINAMATH_CALUDE_annual_savings_l1453_145325

/-- Represents the parking garage rental rates and conditions -/
structure ParkingGarage where
  regular_peak_weekly : ℕ
  regular_nonpeak_weekly : ℕ
  regular_peak_monthly : ℕ
  regular_nonpeak_monthly : ℕ
  large_peak_weekly : ℕ
  large_nonpeak_weekly : ℕ
  large_peak_monthly : ℕ
  large_nonpeak_monthly : ℕ
  holiday_surcharge : ℕ
  nonpeak_weeks : ℕ
  peak_holiday_weeks : ℕ
  total_weeks : ℕ

/-- Calculates the annual cost of renting a large space weekly -/
def weekly_cost (pg : ParkingGarage) : ℕ :=
  pg.large_nonpeak_weekly * pg.nonpeak_weeks +
  pg.large_peak_weekly * (pg.total_weeks - pg.nonpeak_weeks - pg.peak_holiday_weeks) +
  (pg.large_peak_weekly + pg.holiday_surcharge) * pg.peak_holiday_weeks

/-- Calculates the annual cost of renting a large space monthly -/
def monthly_cost (pg : ParkingGarage) : ℕ :=
  pg.large_nonpeak_monthly * (pg.nonpeak_weeks / 4) +
  pg.large_peak_monthly * ((pg.total_weeks - pg.nonpeak_weeks) / 4)

/-- Theorem: The annual savings from renting monthly instead of weekly is $124 -/
theorem annual_savings (pg : ParkingGarage) : weekly_cost pg - monthly_cost pg = 124 :=
  by
    have h1 : pg.regular_peak_weekly = 10 := by sorry
    have h2 : pg.regular_nonpeak_weekly = 8 := by sorry
    have h3 : pg.regular_peak_monthly = 40 := by sorry
    have h4 : pg.regular_nonpeak_monthly = 35 := by sorry
    have h5 : pg.large_peak_weekly = 12 := by sorry
    have h6 : pg.large_nonpeak_weekly = 10 := by sorry
    have h7 : pg.large_peak_monthly = 48 := by sorry
    have h8 : pg.large_nonpeak_monthly = 42 := by sorry
    have h9 : pg.holiday_surcharge = 2 := by sorry
    have h10 : pg.nonpeak_weeks = 16 := by sorry
    have h11 : pg.peak_holiday_weeks = 6 := by sorry
    have h12 : pg.total_weeks = 52 := by sorry
    sorry

end NUMINAMATH_CALUDE_annual_savings_l1453_145325


namespace NUMINAMATH_CALUDE_project_completion_theorem_l1453_145363

/-- The number of days it takes to complete the project -/
def project_completion_time (a_time b_time : ℝ) (a_quit_before : ℝ) : ℝ :=
  let a_rate := 1 / a_time
  let b_rate := 1 / b_time
  let combined_rate := a_rate + b_rate
  15

/-- Theorem stating that the project will be completed in 15 days -/
theorem project_completion_theorem :
  project_completion_time 10 30 10 = 15 := by
  sorry

#eval project_completion_time 10 30 10

end NUMINAMATH_CALUDE_project_completion_theorem_l1453_145363


namespace NUMINAMATH_CALUDE_fourth_tree_growth_difference_l1453_145322

/-- Represents the daily growth rates of four trees and their total growth over a period -/
structure TreeGrowth where
  first_tree : ℝ
  second_tree : ℝ
  third_tree : ℝ
  fourth_tree : ℝ
  total_days : ℕ
  total_growth : ℝ

/-- Theorem stating the difference in daily growth between the fourth and third tree -/
theorem fourth_tree_growth_difference (g : TreeGrowth)
  (h1 : g.first_tree = 1)
  (h2 : g.second_tree = 2 * g.first_tree)
  (h3 : g.third_tree = 2)
  (h4 : g.total_days = 4)
  (h5 : g.total_growth = 32)
  (h6 : g.first_tree * g.total_days + g.second_tree * g.total_days + 
        g.third_tree * g.total_days + g.fourth_tree * g.total_days = g.total_growth) :
  g.fourth_tree - g.third_tree = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_tree_growth_difference_l1453_145322


namespace NUMINAMATH_CALUDE_track_width_l1453_145321

theorem track_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) 
  (h₂ : 2 * π * r₁ - 2 * π * r₂ = 20 * π) 
  (h₃ : r₁ - r₂ = 2 * (r₁ - r₂) / 2) : 
  r₁ - r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l1453_145321


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1453_145379

/-- For a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ
  h₁ : a₁ > 0

/-- The third term of a geometric sequence -/
def GeometricSequence.a₃ (g : GeometricSequence) : ℝ := g.a₁ * g.q^2

theorem geometric_sequence_condition (g : GeometricSequence) :
  (g.q > 1 → g.a₁ < g.a₃) ∧ 
  ¬(g.a₁ < g.a₃ → g.q > 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1453_145379


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l1453_145391

/-- A type representing a circular arrangement of 9 digits -/
def CircularArrangement := Fin 9 → Fin 9

/-- Checks if a number is composite -/
def is_composite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Checks if two adjacent digits in the arrangement form a composite number -/
def adjacent_composite (arr : CircularArrangement) (i : Fin 9) : Prop :=
  let n := (arr i).val * 10 + (arr ((i.val + 1) % 9)).val
  is_composite n

/-- The main theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (arr : CircularArrangement), 
  (∀ i : Fin 9, 1 ≤ (arr i).val ∧ (arr i).val ≤ 9) ∧ 
  (∀ i j : Fin 9, i ≠ j → arr i ≠ arr j) ∧
  (∀ i : Fin 9, adjacent_composite arr i) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l1453_145391
