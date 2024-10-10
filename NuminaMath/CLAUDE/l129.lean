import Mathlib

namespace sugar_amount_is_one_cup_l129_12979

/-- Represents the ratio of ingredients in a recipe --/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def originalRatio : Ratio :=
  { flour := 7, water := 2, sugar := 1 }

/-- The new recipe ratio --/
def newRatio : Ratio :=
  { flour := originalRatio.flour * 2,
    water := originalRatio.water,
    sugar := originalRatio.sugar * 2 }

/-- The amount of water in the new recipe (in cups) --/
def newWaterAmount : ℚ := 2

/-- Calculates the amount of sugar needed in the new recipe --/
def sugarNeeded (r : Ratio) (waterAmount : ℚ) : ℚ :=
  (waterAmount * r.sugar) / r.water

/-- Theorem stating that the amount of sugar needed in the new recipe is 1 cup --/
theorem sugar_amount_is_one_cup :
  sugarNeeded newRatio newWaterAmount = 1 := by
  sorry


end sugar_amount_is_one_cup_l129_12979


namespace inequality_and_equality_conditions_l129_12908

theorem inequality_and_equality_conditions (a b : ℝ) :
  (a^2 + b^2 - a - b - a*b + 0.25 ≥ 0) ∧
  (a^2 + b^2 - a - b - a*b + 0.25 = 0 ↔ (a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0)) := by
  sorry

end inequality_and_equality_conditions_l129_12908


namespace box_areas_product_l129_12907

/-- For a rectangular box with dimensions a, b, and c, and a constant k,
    where the areas of the bottom, side, and front are kab, kbc, and kca respectively,
    the product of these areas is equal to k^3 × (abc)^2. -/
theorem box_areas_product (a b c k : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (a * b * c)^2 := by
  sorry

end box_areas_product_l129_12907


namespace cube_painting_theorem_l129_12977

theorem cube_painting_theorem (n : ℕ) (h : n > 4) :
  (6 * (n - 4)^2 : ℕ) = (n - 4)^3 ↔ n = 10 := by sorry

end cube_painting_theorem_l129_12977


namespace min_value_trig_function_l129_12972

open Real

theorem min_value_trig_function (θ : Real) (h₁ : θ > 0) (h₂ : θ < π / 2) :
  ∀ y : Real, y = 1 / (sin θ)^2 + 9 / (cos θ)^2 → y ≥ 16 := by
  sorry

end min_value_trig_function_l129_12972


namespace equation_solution_l129_12987

theorem equation_solution : ∃ x : ℝ, 6 * x + 12 * x = 558 - 9 * (x - 4) ∧ x = 22 := by
  sorry

end equation_solution_l129_12987


namespace digit_2009_is_zero_l129_12952

/-- Represents the sequence of consecutive natural numbers starting from 1 -/
def consecutiveNaturals : ℕ → ℕ
  | 0 => 1
  | n + 1 => consecutiveNaturals n + 1

/-- Returns the nth digit in the sequence of consecutive natural numbers -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 2009th digit in the sequence is 0 -/
theorem digit_2009_is_zero : nthDigit 2009 = 0 := by sorry

end digit_2009_is_zero_l129_12952


namespace infinite_series_sum_l129_12913

theorem infinite_series_sum : 
  (∑' n : ℕ+, (1 : ℝ) / (n * (n + 3))) = 11 / 18 := by
  sorry

end infinite_series_sum_l129_12913


namespace thirtieth_digit_of_sum_l129_12911

def fraction_sum (a b c : ℚ) : ℚ := a + b + c

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem thirtieth_digit_of_sum :
  nth_digit_after_decimal (fraction_sum (1/7) (1/3) (1/11)) 30 = 9 :=
sorry

end thirtieth_digit_of_sum_l129_12911


namespace trip_charges_eq_14_l129_12906

/-- Represents the daily mileage and charging capacity for a 7-day trip. -/
structure TripData where
  daily_mileage : Fin 7 → ℕ
  initial_charging_capacity : ℕ
  daily_capacity_increment : ℕ
  weather_reduction_days : Finset (Fin 7)
  weather_reduction_percent : ℚ
  stop_interval : ℕ
  stop_days : Finset (Fin 7)

/-- Calculates the number of charges needed for a given day. -/
def charges_needed (data : TripData) (day : Fin 7) : ℕ :=
  sorry

/-- Calculates the total number of charges needed for the entire trip. -/
def total_charges (data : TripData) : ℕ :=
  sorry

/-- The main theorem stating that the total number of charges for the given trip data is 14. -/
theorem trip_charges_eq_14 : ∃ (data : TripData),
  data.daily_mileage = ![135, 259, 159, 189, 210, 156, 240] ∧
  data.initial_charging_capacity = 106 ∧
  data.daily_capacity_increment = 15 ∧
  data.weather_reduction_days = {3, 6} ∧
  data.weather_reduction_percent = 5 / 100 ∧
  data.stop_interval = 55 ∧
  data.stop_days = {1, 5} ∧
  total_charges data = 14 :=
  sorry

end trip_charges_eq_14_l129_12906


namespace picture_placement_l129_12918

theorem picture_placement (wall_width picture_width : ℝ) 
  (hw : wall_width = 22)
  (hp : picture_width = 4) : 
  (wall_width - picture_width) / 2 = 9 := by
  sorry

end picture_placement_l129_12918


namespace smallest_non_prime_sums_l129_12980

theorem smallest_non_prime_sums : ∃ (n : ℕ), n = 7 ∧
  (∀ m : ℕ, m < n →
    (Prime (m + 1 + m + 2 + m + 3) ∨
     Prime (m + m + 2 + m + 3) ∨
     Prime (m + m + 1 + m + 3) ∨
     Prime (m + m + 1 + m + 2))) ∧
  (¬ Prime (n + 1 + n + 2 + n + 3) ∧
   ¬ Prime (n + n + 2 + n + 3) ∧
   ¬ Prime (n + n + 1 + n + 3) ∧
   ¬ Prime (n + n + 1 + n + 2)) :=
by sorry

end smallest_non_prime_sums_l129_12980


namespace complement_of_55_degrees_l129_12995

def angle_A : ℝ := 55

def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_55_degrees :
  complement angle_A = 35 := by
  sorry

end complement_of_55_degrees_l129_12995


namespace modulus_of_z_l129_12920

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = Complex.abs (2 + Complex.I) + 2 * Complex.I) : 
  Complex.abs z = 3 := by
sorry

end modulus_of_z_l129_12920


namespace intersection_complement_eq_set_l129_12946

def R : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x^2 - 2*x > 3}

theorem intersection_complement_eq_set : M ∩ (R \ N) = {-1, 1, 2} := by
  sorry

end intersection_complement_eq_set_l129_12946


namespace function_shift_l129_12992

theorem function_shift (f : ℝ → ℝ) : 
  (∀ x, f (x - 1) = x^2) → (∀ x, f x = (x + 1)^2) := by
sorry

end function_shift_l129_12992


namespace max_portions_is_two_l129_12905

/-- Represents the number of bags for each ingredient -/
structure Ingredients :=
  (nuts : ℕ)
  (dried_fruit : ℕ)
  (chocolate : ℕ)
  (coconut : ℕ)

/-- Represents the ratio of ingredients in each portion -/
structure Ratio :=
  (nuts : ℕ)
  (dried_fruit : ℕ)
  (chocolate : ℕ)
  (coconut : ℕ)

/-- Calculates the maximum number of portions that can be made -/
def max_portions (ingredients : Ingredients) (ratio : Ratio) : ℕ :=
  min (ingredients.nuts / ratio.nuts)
      (min (ingredients.dried_fruit / ratio.dried_fruit)
           (min (ingredients.chocolate / ratio.chocolate)
                (ingredients.coconut / ratio.coconut)))

/-- Proves that the maximum number of portions is 2 -/
theorem max_portions_is_two :
  let ingredients := Ingredients.mk 16 6 8 4
  let ratio := Ratio.mk 4 3 2 1
  max_portions ingredients ratio = 2 :=
by
  sorry

#eval max_portions (Ingredients.mk 16 6 8 4) (Ratio.mk 4 3 2 1)

end max_portions_is_two_l129_12905


namespace inverse_matrices_l129_12978

/-- Two 2x2 matrices are inverses if their product is the identity matrix -/
def are_inverses (A B : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * B = !![1, 0; 0, 1]

/-- Matrix A definition -/
def A (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![x, 3; 1, 5]

/-- Matrix B definition -/
def B (y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![-5/31, 1/31; y, 3/31]

/-- The theorem to be proved -/
theorem inverse_matrices :
  are_inverses (A (-9)) (B (1/31)) := by sorry

end inverse_matrices_l129_12978


namespace challenge_points_40_l129_12957

/-- Calculates the number of activities required for a given number of challenge points. -/
def activities_required (n : ℕ) : ℕ :=
  let segment := (n - 1) / 10
  (n.min 10) * 1 + 
  ((n - 10).max 0).min 10 * 2 + 
  ((n - 20).max 0).min 10 * 3 + 
  ((n - 30).max 0).min 10 * 4 +
  ((n - 40).max 0) * (segment + 1)

/-- Proves that 40 challenge points require 100 activities. -/
theorem challenge_points_40 : activities_required 40 = 100 := by
  sorry

#eval activities_required 40

end challenge_points_40_l129_12957


namespace george_total_earnings_l129_12997

/-- The total amount earned by George from selling toys -/
def george_earnings (num_cars : ℕ) (price_per_car : ℕ) (lego_price : ℕ) : ℕ :=
  num_cars * price_per_car + lego_price

/-- Theorem: George earned $45 from selling 3 cars at $5 each and a set of Legos for $30 -/
theorem george_total_earnings : george_earnings 3 5 30 = 45 := by
  sorry

end george_total_earnings_l129_12997


namespace no_integer_solution_l129_12927

theorem no_integer_solution : ∀ x y : ℤ, x^2 - 37*y ≠ 2 := by
  sorry

end no_integer_solution_l129_12927


namespace arithmetic_sequence_property_l129_12993

/-- An arithmetic sequence. -/
structure ArithmeticSequence (α : Type*) [Add α] [SMul ℕ α] :=
  (a : ℕ → α)
  (d : α)
  (h : ∀ n, a (n + 1) = a n + d)

/-- Theorem: In an arithmetic sequence where a₂ + a₈ = 12, a₅ = 6. -/
theorem arithmetic_sequence_property 
  (a : ArithmeticSequence ℝ) 
  (h : a.a 2 + a.a 8 = 12) : 
  a.a 5 = 6 := by sorry

end arithmetic_sequence_property_l129_12993


namespace fish_population_estimation_l129_12909

/-- Calculates the number of fish in a lake on May 1st given certain conditions --/
theorem fish_population_estimation (marked_may : ℕ) (caught_sept : ℕ) (marked_sept : ℕ)
  (death_rate : ℚ) (new_fish_rate : ℚ) :
  marked_may = 60 →
  caught_sept = 70 →
  marked_sept = 3 →
  death_rate = 1/4 →
  new_fish_rate = 2/5 →
  ∃ (fish_may : ℕ), fish_may = 840 ∧ 
    (fish_may : ℚ) * (1 - death_rate) * (marked_sept : ℚ) / (caught_sept : ℚ) = 
    (marked_may : ℚ) * (1 - death_rate) ∧
    (fish_may : ℚ) * (1 - death_rate) / (1 - new_fish_rate) = 
    (fish_may : ℚ) * (1 - death_rate) * (marked_sept : ℚ) / (caught_sept : ℚ) / (1 - new_fish_rate) :=
by
  sorry


end fish_population_estimation_l129_12909


namespace stack_weight_error_l129_12944

/-- The weight of a disc with exactly 1 meter diameter in kg -/
def standard_weight : ℝ := 100

/-- The nominal radius of a disc in meters -/
def nominal_radius : ℝ := 0.5

/-- The standard deviation of the radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The number of discs in the stack -/
def num_discs : ℕ := 100

/-- The expected weight of a single disc given the manufacturing variation -/
def expected_single_disc_weight : ℝ := sorry

/-- The expected weight of the stack of discs -/
def expected_stack_weight : ℝ := sorry

/-- Engineer Sidorov's estimate of the stack weight -/
def sidorov_estimate : ℝ := 10000

theorem stack_weight_error :
  expected_stack_weight - sidorov_estimate = 4 := by sorry

end stack_weight_error_l129_12944


namespace largest_circle_area_l129_12961

/-- The area of the largest circle formed from a string that fits exactly around a rectangle -/
theorem largest_circle_area (string_length : ℝ) (rectangle_area : ℝ) : 
  string_length = 60 →
  rectangle_area = 200 →
  (∃ (x y : ℝ), x * y = rectangle_area ∧ 2 * (x + y) = string_length) →
  (π * (string_length / (2 * π))^2 : ℝ) = 900 / π :=
by sorry

end largest_circle_area_l129_12961


namespace jerry_earnings_duration_l129_12958

def jerry_earnings : ℕ := 14 + 31 + 20

def jerry_weekly_expenses : ℕ := 5 + 10 + 8

theorem jerry_earnings_duration : 
  ⌊(jerry_earnings : ℚ) / jerry_weekly_expenses⌋ = 2 := by sorry

end jerry_earnings_duration_l129_12958


namespace sqrt_200_range_l129_12904

theorem sqrt_200_range : 14 < Real.sqrt 200 ∧ Real.sqrt 200 < 15 := by
  sorry

end sqrt_200_range_l129_12904


namespace total_interest_after_ten_years_l129_12954

/-- Calculate the total interest after 10 years given:
  * The simple interest on the initial principal for 10 years is 1400
  * The principal is trebled after 5 years
-/
theorem total_interest_after_ten_years (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 1400) : 
  (P * R * 5 / 100) + (3 * P * R * 5 / 100) = 280 := by
  sorry

end total_interest_after_ten_years_l129_12954


namespace max_green_socks_l129_12951

/-- Represents the number of socks in a drawer -/
structure SockDrawer where
  green : ℕ
  yellow : ℕ
  total_bound : green + yellow ≤ 2500

/-- The probability of choosing two socks of the same color -/
def same_color_probability (d : SockDrawer) : ℚ :=
  let t := d.green + d.yellow
  (d.green * (d.green - 1) + d.yellow * (d.yellow - 1)) / (t * (t - 1))

/-- The theorem stating the maximum number of green socks possible -/
theorem max_green_socks (d : SockDrawer) 
  (h : same_color_probability d = 2/3) : 
  d.green ≤ 1275 ∧ ∃ d' : SockDrawer, d'.green = 1275 ∧ same_color_probability d' = 2/3 := by
  sorry


end max_green_socks_l129_12951


namespace parallelogram_area_l129_12921

/-- The area of a parallelogram with base 36 cm and height 24 cm is 864 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 36
  let height : ℝ := 24
  let area := base * height
  area = 864 := by sorry

end parallelogram_area_l129_12921


namespace problem_solution_l129_12915

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 20 →
  x^2 + Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 10201 / 200 := by
  sorry

end problem_solution_l129_12915


namespace first_statue_weight_l129_12968

/-- Given the weights of a marble block and its carved statues, prove the weight of the first statue -/
theorem first_statue_weight
  (total_weight : ℝ)
  (second_statue : ℝ)
  (third_statue : ℝ)
  (fourth_statue : ℝ)
  (discarded : ℝ)
  (h1 : total_weight = 80)
  (h2 : second_statue = 18)
  (h3 : third_statue = 15)
  (h4 : fourth_statue = 15)
  (h5 : discarded = 22)
  : ∃ (first_statue : ℝ),
    first_statue + second_statue + third_statue + fourth_statue + discarded = total_weight ∧
    first_statue = 10 := by
  sorry

end first_statue_weight_l129_12968


namespace winning_condition_l129_12973

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- Represents the result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The game played on a chessboard -/
def game (n : ℕ) (board : Chessboard n) : GameResult := sorry

/-- Theorem stating the winning condition based on the parity of n -/
theorem winning_condition (n : ℕ) (board : Chessboard n) :
  game n board = GameResult.FirstPlayerWins ↔ Even n := by sorry

end winning_condition_l129_12973


namespace cosine_product_eleven_l129_12916

theorem cosine_product_eleven : 
  Real.cos (π / 11) * Real.cos (2 * π / 11) * Real.cos (3 * π / 11) * 
  Real.cos (4 * π / 11) * Real.cos (5 * π / 11) = 1 / 32 :=
by sorry

end cosine_product_eleven_l129_12916


namespace g_1994_of_4_l129_12986

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => λ x => g (g_n n x)

theorem g_1994_of_4 : g_n 1994 4 = 87 / 50 := by
  sorry

end g_1994_of_4_l129_12986


namespace triangle_ratio_l129_12936

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b*sin(2A) = 3a*sin(B) and c = 2b, then a/b = √2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end triangle_ratio_l129_12936


namespace triangle_angle_inequalities_l129_12984

-- Define a structure for a triangle's angles
structure TriangleAngles where
  a : Real
  b : Real
  c : Real
  sum_is_pi : a + b + c = Real.pi

-- Theorem statement
theorem triangle_angle_inequalities (t : TriangleAngles) :
  (Real.sin t.a + Real.sin t.b + Real.sin t.c ≤ 3 * Real.sqrt 3 / 2) ∧
  (Real.cos (t.a / 2) + Real.cos (t.b / 2) + Real.cos (t.c / 2) ≤ 3 * Real.sqrt 3 / 2) ∧
  (Real.cos t.a * Real.cos t.b * Real.cos t.c ≤ 1 / 8) ∧
  (Real.sin (2 * t.a) + Real.sin (2 * t.b) + Real.sin (2 * t.c) ≤ Real.sin t.a + Real.sin t.b + Real.sin t.c) :=
by
  sorry


end triangle_angle_inequalities_l129_12984


namespace complex_rectangle_perimeter_l129_12971

/-- A structure representing a rectangle with an internal complex shape. -/
structure ComplexRectangle where
  width : ℝ
  height : ℝ
  enclosed_area : ℝ

/-- The perimeter of a ComplexRectangle is equal to 2 * (width + height) -/
def perimeter (r : ComplexRectangle) : ℝ := 2 * (r.width + r.height)

theorem complex_rectangle_perimeter :
  ∀ (r : ComplexRectangle),
    r.width = 15 ∧ r.height = 10 ∧ r.enclosed_area = 108 →
    perimeter r = 50 := by
  sorry

end complex_rectangle_perimeter_l129_12971


namespace liliane_alice_relationship_l129_12928

/-- Represents the amount of soda each person has -/
structure SodaAmounts where
  jacqueline : ℝ
  liliane : ℝ
  alice : ℝ
  bruno : ℝ

/-- The conditions of the soda problem -/
def SodaProblem (amounts : SodaAmounts) : Prop :=
  amounts.liliane = amounts.jacqueline * 1.6 ∧
  amounts.alice = amounts.jacqueline * 1.4 ∧
  amounts.bruno = amounts.jacqueline * 0.8

/-- The theorem stating the relationship between Liliane's and Alice's soda amounts -/
theorem liliane_alice_relationship (amounts : SodaAmounts) 
  (h : SodaProblem amounts) : 
  ∃ ε > 0, ε < 0.005 ∧ amounts.liliane = amounts.alice * (1 + 0.15 + ε) :=
sorry

end liliane_alice_relationship_l129_12928


namespace positive_y_intercept_l129_12932

/-- A line that intersects the y-axis in the positive half-plane -/
structure PositiveYInterceptLine where
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line equation is y = 2x + b -/
  equation : ∀ (x y : ℝ), y = 2 * x + b
  /-- The line intersects the y-axis in the positive half-plane -/
  positive_intercept : ∃ (y : ℝ), y > 0 ∧ y = b

/-- The y-intercept of a line that intersects the y-axis in the positive half-plane is positive -/
theorem positive_y_intercept (l : PositiveYInterceptLine) : l.b > 0 := by
  sorry

end positive_y_intercept_l129_12932


namespace derek_rides_more_than_carla_l129_12960

-- Define the speeds and times
def carla_speed : ℝ := 12
def derek_speed : ℝ := 15
def derek_time : ℝ := 3
def time_difference : ℝ := 0.5

-- Theorem statement
theorem derek_rides_more_than_carla :
  derek_speed * derek_time - carla_speed * (derek_time + time_difference) = 3 := by
  sorry

end derek_rides_more_than_carla_l129_12960


namespace tan_theta_minus_pi_fourth_l129_12991

theorem tan_theta_minus_pi_fourth (θ : ℝ) (z : ℂ) : 
  z = (Real.cos θ - 4/5) + (Real.sin θ - 3/5) * Complex.I ∧ 
  z.re = 0 ∧ z.im ≠ 0 → 
  Real.tan (θ - π/4) = -7 :=
by sorry

end tan_theta_minus_pi_fourth_l129_12991


namespace fixed_point_of_exponential_function_l129_12902

/-- Given a > 0 and a ≠ 1, prove that the function f(x) = 2 - a^(x+1) always passes through the point (-1, 1) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ 2 - a^(x + 1)
  f (-1) = 1 := by
  sorry

end fixed_point_of_exponential_function_l129_12902


namespace dog_count_l129_12923

theorem dog_count (total : ℕ) (cats : ℕ) (h1 : total = 17) (h2 : cats = 8) :
  total - cats = 9 := by
  sorry

end dog_count_l129_12923


namespace quadratic_inequality_condition_l129_12996

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - x + 1 > 0) ↔ a > (1/4 : ℝ) := by sorry

end quadratic_inequality_condition_l129_12996


namespace cricket_players_l129_12975

theorem cricket_players (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 460)
  (h2 : football = 325)
  (h3 : neither = 50)
  (h4 : both = 90) :
  ∃ cricket : ℕ, cricket = 175 ∧ 
  cricket = total - neither - (football - both) := by
  sorry

end cricket_players_l129_12975


namespace math_grade_calculation_l129_12917

theorem math_grade_calculation (history_grade third_subject_grade : ℝ) 
  (h1 : history_grade = 84)
  (h2 : third_subject_grade = 67)
  (h3 : (math_grade + history_grade + third_subject_grade) / 3 = 75) :
  math_grade = 74 := by
sorry

end math_grade_calculation_l129_12917


namespace negation_of_all_squared_nonnegative_l129_12967

theorem negation_of_all_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_all_squared_nonnegative_l129_12967


namespace x_equals_n_l129_12965

def x : ℕ → ℚ
  | 0 => 0
  | n + 1 => ((n^2 + n + 1) * x n + 1) / (n^2 + n + 1 - x n)

theorem x_equals_n (n : ℕ) : x n = n := by
  sorry

end x_equals_n_l129_12965


namespace largest_inscribed_triangle_area_l129_12938

-- Define the circle
def circle_radius : ℝ := 8

-- Define the diameter
def diameter : ℝ := 2 * circle_radius

-- Define the height of the triangle (equal to the radius)
def triangle_height : ℝ := circle_radius

-- Theorem statement
theorem largest_inscribed_triangle_area :
  let triangle_area := (1 / 2) * diameter * triangle_height
  triangle_area = 64 := by sorry

end largest_inscribed_triangle_area_l129_12938


namespace circle_reflection_translation_l129_12945

/-- Given a point (3, -4), prove that reflecting it across the x-axis
    and then translating it 5 units to the left results in the point (-2, 4) -/
theorem circle_reflection_translation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point : ℝ × ℝ := (initial_point.1, -initial_point.2)
  let final_point : ℝ × ℝ := (reflected_point.1 - 5, reflected_point.2)
  final_point = (-2, 4) := by
sorry

end circle_reflection_translation_l129_12945


namespace ze_and_triplet_ages_l129_12983

/-- Represents the ages of Zé Roberto and his children -/
structure FamilyAges where
  ze : ℕ
  twin : ℕ
  triplet : ℕ

/-- Conditions for Zé Roberto's family ages -/
def valid_family_ages (f : FamilyAges) : Prop :=
  -- Zé's current age equals the sum of his children's ages
  f.ze = 2 * f.twin + 3 * f.triplet ∧
  -- In 15 years, the sum of the children's ages will be twice Zé's age
  2 * (f.ze + 15) = 2 * (f.twin + 15) + 3 * (f.triplet + 15) ∧
  -- In 15 years, the sum of the twins' ages will equal the sum of the triplets' ages
  2 * (f.twin + 15) = 3 * (f.triplet + 15)

/-- Theorem stating Zé's current age and the age of each triplet -/
theorem ze_and_triplet_ages (f : FamilyAges) (h : valid_family_ages f) :
  f.ze = 45 ∧ f.triplet = 5 := by
  sorry


end ze_and_triplet_ages_l129_12983


namespace largest_equal_cost_number_l129_12953

/-- Calculates the sum of digits of a number in base 10 -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of binary digits of a number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is less than 500 -/
def lessThan500 (n : ℕ) : Prop := n < 500

/-- Checks if the cost is the same for both options -/
def equalCost (n : ℕ) : Prop := sumDigits n = sumBinaryDigits n

theorem largest_equal_cost_number :
  ∀ n : ℕ, lessThan500 n → equalCost n → n ≤ 247 :=
sorry

end largest_equal_cost_number_l129_12953


namespace nabla_square_l129_12933

theorem nabla_square (odot nabla : ℕ) : 
  odot ≠ nabla → 
  0 < odot → odot < 20 → 
  0 < nabla → nabla < 20 → 
  nabla * nabla * nabla = nabla →
  nabla * nabla = 64 := by
sorry

end nabla_square_l129_12933


namespace overhead_percentage_l129_12942

theorem overhead_percentage (purchase_price markup net_profit : ℝ) :
  purchase_price = 48 →
  markup = 45 →
  net_profit = 12 →
  (((purchase_price + markup - net_profit) - purchase_price) / purchase_price) * 100 = 68.75 := by
  sorry

end overhead_percentage_l129_12942


namespace monotone_increasing_quadratic_l129_12941

/-- A function f(x) = 4x^2 - kx - 8 is monotonically increasing on [5, +∞) if and only if k ≤ 40 -/
theorem monotone_increasing_quadratic (k : ℝ) :
  (∀ x ≥ 5, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ k ≤ 40 := by
  sorry

end monotone_increasing_quadratic_l129_12941


namespace work_completion_theorem_l129_12943

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 25

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 20

/-- The number of men in the first group -/
def men_first_group : ℕ := men_second_group * days_second_group / days_first_group

theorem work_completion_theorem :
  men_first_group * days_first_group = men_second_group * days_second_group ∧
  men_first_group = 16 := by
  sorry

end work_completion_theorem_l129_12943


namespace jackson_pbj_sandwiches_l129_12930

/-- The number of weeks in the school year -/
def school_weeks : ℕ := 36

/-- The number of days per week Jackson could eat peanut butter and jelly sandwiches -/
def pbj_days_per_week : ℕ := 2

/-- The number of Wednesdays Jackson missed -/
def missed_wednesdays : ℕ := 1

/-- The number of Fridays Jackson missed -/
def missed_fridays : ℕ := 2

/-- The total number of peanut butter and jelly sandwiches Jackson ate -/
def total_pbj_sandwiches : ℕ := school_weeks * pbj_days_per_week - (missed_wednesdays + missed_fridays)

theorem jackson_pbj_sandwiches :
  total_pbj_sandwiches = 69 := by
  sorry

end jackson_pbj_sandwiches_l129_12930


namespace valid_seven_digit_integers_l129_12929

-- Define the recurrence relation
def a : ℕ → ℕ
  | 0 => 0  -- Base case (not used)
  | 1 => 4  -- a₁ = 4
  | 2 => 17 -- a₂ = 17
  | n + 3 => 4 * a (n + 2) + 2 * a (n + 1)

-- Theorem statement
theorem valid_seven_digit_integers : a 7 = 29776 := by
  sorry

end valid_seven_digit_integers_l129_12929


namespace number_puzzle_l129_12956

theorem number_puzzle (x : ℤ) : x + (x - 1) = 33 → 6 * x - 2 = 100 := by
  sorry

end number_puzzle_l129_12956


namespace inequality_solution_l129_12912

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 3) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end inequality_solution_l129_12912


namespace betty_afternoon_flies_l129_12994

/-- The number of flies Betty caught in the afternoon before one escaped -/
def afternoon_flies : ℕ := 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies the frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of flies Betty caught in the morning -/
def morning_flies : ℕ := 5

/-- The number of additional flies Betty needs for the whole week -/
def additional_flies_needed : ℕ := 4

/-- The number of flies that escaped when Betty removed the lid -/
def escaped_flies : ℕ := 1

theorem betty_afternoon_flies :
  afternoon_flies = 
    days_in_week * flies_per_day - morning_flies - additional_flies_needed + escaped_flies :=
by sorry

end betty_afternoon_flies_l129_12994


namespace calculate_mixed_number_l129_12937

theorem calculate_mixed_number : 7 * (9 + 2/5) - 3 = 62 + 4/5 := by
  sorry

end calculate_mixed_number_l129_12937


namespace monic_quadratic_polynomial_l129_12989

theorem monic_quadratic_polynomial (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 5*x + 6) → 
  f 0 = 6 ∧ f 1 = 12 := by
sorry

end monic_quadratic_polynomial_l129_12989


namespace tanks_needed_l129_12901

def existing_tanks : ℕ := 3
def existing_capacity : ℕ := 15
def new_capacity : ℕ := 10
def total_fish : ℕ := 75

theorem tanks_needed : 
  (total_fish - existing_tanks * existing_capacity) / new_capacity = 3 := by
  sorry

end tanks_needed_l129_12901


namespace soccer_team_lineups_l129_12988

/-- The number of ways to select a starting lineup from a soccer team -/
def numStartingLineups (totalPlayers : ℕ) (regularPlayers : ℕ) : ℕ :=
  totalPlayers * Nat.choose (totalPlayers - 1) regularPlayers

/-- Theorem stating that the number of starting lineups for a team of 16 players,
    with 1 goalie and 10 regular players, is 48,048 -/
theorem soccer_team_lineups :
  numStartingLineups 16 10 = 48048 := by
  sorry

#eval numStartingLineups 16 10

end soccer_team_lineups_l129_12988


namespace tangent_line_at_origin_l129_12982

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a - 3)x and its derivative f'(x) is an even function,
    then the equation of the tangent line to the curve y = f(x) at the origin is 2x + y = 0. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a - 3)*x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + (a - 3)
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (λ x y ↦ 2*x + y = 0) = (λ x y ↦ y = f' 0 * x) := by
  sorry

end tangent_line_at_origin_l129_12982


namespace inverse_relation_l129_12935

theorem inverse_relation (x : ℝ) (h : 1 / x = 40) : x = 1 / 40 := by
  sorry

end inverse_relation_l129_12935


namespace smallest_k_with_remainder_l129_12966

theorem smallest_k_with_remainder (k : ℕ) : k = 135 ↔ 
  (k > 1) ∧ 
  (∃ a : ℕ, k = 11 * a + 3) ∧ 
  (∃ b : ℕ, k = 4 * b + 3) ∧ 
  (∃ c : ℕ, k = 3 * c + 3) ∧ 
  (∀ m : ℕ, m > 1 → 
    ((∃ x : ℕ, m = 11 * x + 3) ∧ 
     (∃ y : ℕ, m = 4 * y + 3) ∧ 
     (∃ z : ℕ, m = 3 * z + 3)) → 
    m ≥ k) :=
by sorry

end smallest_k_with_remainder_l129_12966


namespace product_of_sums_equals_3280_l129_12947

theorem product_of_sums_equals_3280 :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_of_sums_equals_3280_l129_12947


namespace solution_to_equation_l129_12934

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^5 = (14 * x)^4 ↔ x = 16/7 := by
  sorry

end solution_to_equation_l129_12934


namespace remaining_pie_portion_l129_12924

theorem remaining_pie_portion (carlos_share : Real) (maria_fraction : Real) : 
  carlos_share = 0.8 →
  maria_fraction = 0.25 →
  (1 - carlos_share) * (1 - maria_fraction) = 0.15 :=
by
  sorry

end remaining_pie_portion_l129_12924


namespace ian_lottery_winnings_l129_12900

theorem ian_lottery_winnings :
  ∀ (lottery_winnings : ℕ) (colin_payment helen_payment benedict_payment remaining : ℕ),
  colin_payment = 20 →
  helen_payment = 2 * colin_payment →
  benedict_payment = helen_payment / 2 →
  remaining = 20 →
  lottery_winnings = colin_payment + helen_payment + benedict_payment + remaining →
  lottery_winnings = 100 :=
by
  sorry

end ian_lottery_winnings_l129_12900


namespace original_price_after_discounts_l129_12998

theorem original_price_after_discounts (final_price : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) (original_price : ℝ) : 
  final_price = 144 ∧ 
  discount1 = 0.1 ∧ 
  discount2 = 0.2 ∧ 
  final_price = original_price * (1 - discount1) * (1 - discount2) → 
  original_price = 200 := by sorry

end original_price_after_discounts_l129_12998


namespace gcd_product_is_square_l129_12919

theorem gcd_product_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x y).gcd z * x * y * z = k ^ 2 := by
sorry

end gcd_product_is_square_l129_12919


namespace polynomial_divisibility_l129_12903

def polynomial (x m : ℝ) : ℝ := 3 * x^2 - 9 * x + m

theorem polynomial_divisibility (m : ℝ) : 
  (∃ q : ℝ → ℝ, ∀ x, polynomial x m = (x - 2) * q x) ↔ m = 6 := by
  sorry

end polynomial_divisibility_l129_12903


namespace triangle_inequality_l129_12925

theorem triangle_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end triangle_inequality_l129_12925


namespace iris_shopping_l129_12963

theorem iris_shopping (jacket_price shorts_price pants_price total_spent : ℕ)
  (jacket_quantity pants_quantity : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  pants_price = 12 →
  jacket_quantity = 3 →
  pants_quantity = 4 →
  total_spent = 90 →
  ∃ shorts_quantity : ℕ, 
    total_spent = jacket_price * jacket_quantity + 
                  shorts_price * shorts_quantity + 
                  pants_price * pants_quantity ∧
    shorts_quantity = 2 :=
by sorry

end iris_shopping_l129_12963


namespace intersection_singleton_k_value_l129_12999

theorem intersection_singleton_k_value (k : ℝ) : 
  let A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2 * (p.1 + p.2)}
  let B : Set (ℝ × ℝ) := {p | k * p.1 - p.2 + k + 3 ≥ 0}
  (Set.Subsingleton (A ∩ B)) → k = -2 - Real.sqrt 3 :=
by sorry

#check intersection_singleton_k_value

end intersection_singleton_k_value_l129_12999


namespace no_two_digit_even_square_palindromes_l129_12922

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem no_two_digit_even_square_palindromes :
  ¬ ∃ n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ 
    (∃ m : ℕ, n = m * m ∧ is_even m) ∧ is_palindrome n :=
sorry

end no_two_digit_even_square_palindromes_l129_12922


namespace cabinet_can_pass_through_door_l129_12990

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a room with given dimensions -/
def Room := Dimensions

/-- Represents a cabinet with given dimensions -/
def Cabinet := Dimensions

/-- Represents a door with given dimensions -/
structure Door where
  width : ℝ
  height : ℝ

/-- Checks if a cabinet can pass through a door -/
def can_pass_through (c : Cabinet) (d : Door) : Prop :=
  (c.width ≤ d.width ∧ c.height ≤ d.height) ∨
  (c.width ≤ d.height ∧ c.height ≤ d.width) ∨
  (c.length ≤ d.width ∧ c.height ≤ d.height) ∨
  (c.length ≤ d.height ∧ c.height ≤ d.width)

theorem cabinet_can_pass_through_door 
  (room : Room)
  (cabinet : Cabinet)
  (door : Door)
  (h_room : room = ⟨4, 2.5, 2.3⟩)
  (h_cabinet : cabinet = ⟨1.8, 0.6, 2.1⟩)
  (h_door : door = ⟨0.8, 1.9⟩) :
  can_pass_through cabinet door :=
sorry

end cabinet_can_pass_through_door_l129_12990


namespace friends_journey_time_l129_12931

/-- Represents the journey of three friends with a bicycle --/
theorem friends_journey_time :
  -- Define the walking speed of the friends
  ∀ (walking_speed : ℝ),
  -- Define the bicycle speed
  ∀ (bicycle_speed : ℝ),
  -- Conditions
  (walking_speed > 0) →
  (bicycle_speed > 0) →
  -- Second friend walks 6 km in the first hour
  (walking_speed * 1 = 6) →
  -- Third friend rides 12 km in 2/3 hour
  (bicycle_speed * (2/3) = 12) →
  -- Total journey time
  ∃ (total_time : ℝ),
  total_time = 2 + 2/3 :=
by sorry

end friends_journey_time_l129_12931


namespace distinct_prime_factors_of_84_l129_12969

theorem distinct_prime_factors_of_84 : ∃ (p q r : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  84 = p * q * r := by
  sorry

end distinct_prime_factors_of_84_l129_12969


namespace average_equals_50y_implies_y_value_l129_12981

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_equals_50y_implies_y_value :
  let n := 99
  let sum_1_to_99 := sum_to_n n
  ∀ y : ℚ, (sum_1_to_99 + y) / (n + 1 : ℚ) = 50 * y → y = 4950 / 4999 := by
sorry

end average_equals_50y_implies_y_value_l129_12981


namespace min_value_sum_of_reciprocals_min_value_is_nine_l129_12976

theorem min_value_sum_of_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  ∀ x y, x > 0 → y > 0 → 2/a + 1/b ≤ 2/x + 1/y :=
by
  sorry

theorem min_value_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  ∃ x y, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 9 :=
by
  sorry

end min_value_sum_of_reciprocals_min_value_is_nine_l129_12976


namespace dvd_packs_calculation_l129_12962

theorem dvd_packs_calculation (total_money : ℕ) (pack_cost : ℕ) (h1 : total_money = 104) (h2 : pack_cost = 26) :
  total_money / pack_cost = 4 := by
  sorry

end dvd_packs_calculation_l129_12962


namespace smallest_prime_divisor_of_sum_l129_12914

theorem smallest_prime_divisor_of_sum (n : ℕ) (h : n = 3^15 + 5^21) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q :=
by
  sorry

end smallest_prime_divisor_of_sum_l129_12914


namespace factorial_ratio_equality_l129_12948

theorem factorial_ratio_equality : (Nat.factorial 9)^2 / (Nat.factorial 4 * Nat.factorial 5) = 45760000 := by
  sorry

end factorial_ratio_equality_l129_12948


namespace ticket_draw_theorem_l129_12970

theorem ticket_draw_theorem (total : ℕ) (blue green red yellow orange : ℕ) : 
  total = 400 ∧ 
  blue + green + red + yellow + orange = total ∧
  blue * 2 = green ∧ 
  green * 2 = red ∧
  green * 3 = yellow ∧
  yellow * 2 = orange →
  (∃ n : ℕ, n ≤ 196 ∧ 
    (∀ m : ℕ, m < n → 
      (m ≤ blue ∨ m ≤ green ∨ m ≤ red ∨ m ≤ yellow ∨ m ≤ orange) ∧ 
      m < 50)) ∧
  (∃ color : ℕ, color ≥ 50 ∧ 
    (color = blue ∨ color = green ∨ color = red ∨ color = yellow ∨ color = orange) ∧
    color ≤ 196) := by
  sorry

end ticket_draw_theorem_l129_12970


namespace son_age_problem_l129_12964

theorem son_age_problem (son_age father_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end son_age_problem_l129_12964


namespace center_is_nine_l129_12950

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Check if a grid satisfies the consecutive number condition --/
def consecutiveAdjacent (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, (g i j).succ = g k l → adjacent (i, j) (k, l)

/-- Sum of corner numbers in the grid --/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The center number in the grid --/
def centerNumber (g : Grid) : Nat := g 1 1

/-- All numbers from 1 to 9 are used in the grid --/
def usesAllNumbers (g : Grid) : Prop :=
  ∀ n : Nat, n ≥ 1 → n ≤ 9 → ∃ i j : Fin 3, g i j = n

theorem center_is_nine (g : Grid) 
    (h1 : usesAllNumbers g)
    (h2 : consecutiveAdjacent g)
    (h3 : cornerSum g = 20) :
  centerNumber g = 9 := by
  sorry

end center_is_nine_l129_12950


namespace batsman_average_increase_l129_12926

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  total_runs : Nat
  average : Rat

/-- Calculates the increase in average after a new inning -/
def average_increase (b : Batsman) (new_runs : Nat) : Rat :=
  let new_total := b.total_runs + new_runs
  let new_average : Rat := new_total / (b.innings + 1)
  new_average - b.average

/-- Theorem: The batsman's average increases by 3 -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 16 →
    (b.total_runs + 86) / 17 = 38 →
    average_increase b 86 = 3 := by
  sorry

end batsman_average_increase_l129_12926


namespace parabola_point_x_coordinate_l129_12940

/-- Given a parabola y² = 4x and a point M on the parabola whose distance to the focus is 3,
    prove that the x-coordinate of M is 2. -/
theorem parabola_point_x_coordinate (x y : ℝ) : 
  y^2 = 4*x →  -- M is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 3^2 →  -- Distance from M to focus (1, 0) is 3
  x = 2 := by sorry

end parabola_point_x_coordinate_l129_12940


namespace square_minus_eight_equals_power_of_three_l129_12949

theorem square_minus_eight_equals_power_of_three (b n : ℕ) :
  b^2 - 8 = 3^n ↔ b = 3 ∧ n = 0 := by sorry

end square_minus_eight_equals_power_of_three_l129_12949


namespace arithmetic_sequence_properties_l129_12985

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_3_eq_9 : a 3 = 9
  S_3_eq_33 : (a 1 + a 2 + a 3) = 33

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  ∃ (d : ℝ),
    (∀ n, seq.a n = seq.a 1 + (n - 1) * d) ∧
    d = -2 ∧
    (∀ n, seq.a n = 15 - 2 * n) ∧
    (∃ n_max : ℕ, ∀ n : ℕ, 
      (n * (seq.a 1 + seq.a n) / 2) ≤ (n_max * (seq.a 1 + seq.a n_max) / 2) ∧
      n_max * (seq.a 1 + seq.a n_max) / 2 = 49) :=
by sorry


end arithmetic_sequence_properties_l129_12985


namespace combined_mean_score_l129_12910

/-- Given two classes with different average scores and a ratio of students, 
    calculate the combined mean score. -/
theorem combined_mean_score (avg1 avg2 : ℝ) (ratio1 ratio2 : ℕ) : 
  avg1 = 90 →
  avg2 = 75 →
  ratio1 = 2 →
  ratio2 = 3 →
  (avg1 * ratio1 + avg2 * ratio2) / (ratio1 + ratio2) = 81 := by
sorry

end combined_mean_score_l129_12910


namespace distribution_methods_count_l129_12959

/-- The number of ways to distribute 4 out of 7 different books to 4 students -/
def distribute_books (total_books : ℕ) (books_to_distribute : ℕ) (students : ℕ) 
  (restricted_books : ℕ) (restricted_student : ℕ) : ℕ :=
  (total_books - restricted_books) * 
  (Nat.factorial (total_books - 1) / (Nat.factorial (total_books - books_to_distribute) * 
   Nat.factorial (books_to_distribute - 1)))

/-- Theorem stating the number of distribution methods -/
theorem distribution_methods_count : 
  distribute_books 7 4 4 2 1 = 600 := by
  sorry

end distribution_methods_count_l129_12959


namespace joan_took_25_marbles_l129_12939

-- Define the initial number of yellow marbles
def initial_yellow_marbles : ℕ := 86

-- Define the remaining number of yellow marbles
def remaining_yellow_marbles : ℕ := 61

-- Define the number of yellow marbles Joan took
def marbles_taken : ℕ := initial_yellow_marbles - remaining_yellow_marbles

-- Theorem to prove
theorem joan_took_25_marbles : marbles_taken = 25 := by
  sorry

end joan_took_25_marbles_l129_12939


namespace droid_weekly_usage_l129_12974

/-- Represents the daily coffee bean usage in Droid's coffee shop -/
structure DailyUsage where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Calculates the total daily usage -/
def totalDailyUsage (usage : DailyUsage) : ℕ :=
  usage.morning + usage.afternoon + usage.evening

/-- Represents the weekly coffee bean usage in Droid's coffee shop -/
structure WeeklyUsage where
  weekday : DailyUsage
  saturday : DailyUsage
  sunday : DailyUsage

/-- Calculates the total weekly usage -/
def totalWeeklyUsage (usage : WeeklyUsage) : ℕ :=
  5 * totalDailyUsage usage.weekday + totalDailyUsage usage.saturday + totalDailyUsage usage.sunday

/-- The coffee bean usage pattern for Droid's coffee shop -/
def droidUsage : WeeklyUsage where
  weekday := { morning := 3, afternoon := 9, evening := 6 }
  saturday := { morning := 4, afternoon := 8, evening := 6 }
  sunday := { morning := 2, afternoon := 2, evening := 2 }

theorem droid_weekly_usage : totalWeeklyUsage droidUsage = 114 := by
  sorry

end droid_weekly_usage_l129_12974


namespace polynomial_division_remainder_l129_12955

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^5 + 2 = (x^2 - 2*x + 3) * q + (-4*x^2 - 3*x + 2) := by sorry

end polynomial_division_remainder_l129_12955
