import Mathlib

namespace exam_mean_score_l1702_170242

/-- Given an exam where a score of 58 is 2 standard deviations below the mean
    and a score of 98 is 3 standard deviations above the mean,
    prove that the mean score is 74. -/
theorem exam_mean_score (mean sd : ℝ) 
    (h1 : 58 = mean - 2 * sd)
    (h2 : 98 = mean + 3 * sd) : 
  mean = 74 := by
  sorry

end exam_mean_score_l1702_170242


namespace piper_gym_schedule_theorem_l1702_170275

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns true if Piper goes to gym on the given day -/
def goes_to_gym (day : DayOfWeek) : Bool :=
  match day with
  | DayOfWeek.Sunday => false
  | DayOfWeek.Monday => true
  | DayOfWeek.Tuesday => false
  | DayOfWeek.Wednesday => true
  | DayOfWeek.Thursday => false
  | DayOfWeek.Friday => true
  | DayOfWeek.Saturday => true

/-- Returns the next day of the week -/
def next_day (day : DayOfWeek) : DayOfWeek :=
  match day with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Returns the day after n days from the given start day -/
def day_after (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => day_after (next_day start) n

/-- Counts the number of gym sessions from the start day up to n days -/
def count_sessions (start : DayOfWeek) (n : Nat) : Nat :=
  match n with
  | 0 => if goes_to_gym start then 1 else 0
  | n + 1 => count_sessions start n + if goes_to_gym (day_after start n) then 1 else 0

theorem piper_gym_schedule_theorem :
  ∃ (n : Nat), count_sessions DayOfWeek.Monday n = 35 ∧ 
               day_after DayOfWeek.Monday n = DayOfWeek.Monday :=
  sorry


end piper_gym_schedule_theorem_l1702_170275


namespace sqrt_difference_is_two_l1702_170260

theorem sqrt_difference_is_two (x : ℝ) : 
  Real.sqrt (x + 2 + 2 * Real.sqrt (x + 1)) - Real.sqrt (x + 2 - 2 * Real.sqrt (x + 1)) = 2 :=
by sorry

end sqrt_difference_is_two_l1702_170260


namespace two_buckets_water_amount_l1702_170204

/-- A container for water -/
structure Container where
  capacity : ℕ

/-- A jug is a container with a capacity of 5 liters -/
def Jug : Container :=
  { capacity := 5 }

/-- A bucket is a container that can hold 4 jugs worth of water -/
def Bucket : Container :=
  { capacity := 4 * Jug.capacity }

/-- The amount of water in multiple containers -/
def water_amount (n : ℕ) (c : Container) : ℕ :=
  n * c.capacity

theorem two_buckets_water_amount :
  water_amount 2 Bucket = 40 :=
by sorry

end two_buckets_water_amount_l1702_170204


namespace element_in_M_l1702_170299

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}

theorem element_in_M : 2.5 ∈ M := by
  sorry

end element_in_M_l1702_170299


namespace largest_five_digit_congruent_16_mod_25_l1702_170297

theorem largest_five_digit_congruent_16_mod_25 : ∃ n : ℕ,
  n = 99991 ∧
  10000 ≤ n ∧ n < 100000 ∧
  n % 25 = 16 ∧
  ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 25 = 16 → m ≤ n :=
by sorry

end largest_five_digit_congruent_16_mod_25_l1702_170297


namespace inequality_solution_set_l1702_170240

theorem inequality_solution_set 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) 
  (h2 : a < 0) : 
  ∀ x, a*(x^2 + 1) + b*(x - 1) + c > 2*a*x ↔ 0 < x ∧ x < 3 := by
sorry

end inequality_solution_set_l1702_170240


namespace square_difference_equality_l1702_170279

theorem square_difference_equality : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end square_difference_equality_l1702_170279


namespace total_days_on_island_l1702_170227

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The duration of the first expedition in weeks -/
def first_expedition : ℕ := 3

/-- Calculates the duration of the second expedition in weeks -/
def second_expedition : ℕ := first_expedition + 2

/-- Calculates the duration of the last expedition in weeks -/
def last_expedition : ℕ := 2 * second_expedition

/-- Calculates the total number of weeks spent on all expeditions -/
def total_weeks : ℕ := first_expedition + second_expedition + last_expedition

/-- Theorem stating the total number of days spent on the island -/
theorem total_days_on_island : total_weeks * days_per_week = 126 := by
  sorry

end total_days_on_island_l1702_170227


namespace inverse_function_value_l1702_170245

theorem inverse_function_value (a : ℝ) (h : a > 1) :
  let f (x : ℝ) := a^(x + 1) - 2
  let f_inv := Function.invFun f
  f_inv (-1) = -1 := by
  sorry

end inverse_function_value_l1702_170245


namespace cards_in_unfilled_box_l1702_170270

theorem cards_in_unfilled_box (total_cards : Nat) (cards_per_box : Nat) (h1 : total_cards = 94) (h2 : cards_per_box = 8) :
  total_cards % cards_per_box = 6 := by
sorry

end cards_in_unfilled_box_l1702_170270


namespace polar_to_cartesian_circle_l1702_170264

theorem polar_to_cartesian_circle (ρ θ x y : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end polar_to_cartesian_circle_l1702_170264


namespace max_value_theorem_l1702_170248

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 + y^2 = 6 * x) :
  ∃ (max_val : ℝ), max_val = 15 ∧ ∀ (a b : ℝ), 2 * a^2 + b^2 = 6 * a → a^2 + b^2 + 2 * a ≤ max_val :=
by sorry

end max_value_theorem_l1702_170248


namespace magpies_in_park_l1702_170222

/-- The number of magpies in a park with blackbirds and magpies -/
theorem magpies_in_park (trees : ℕ) (blackbirds_per_tree : ℕ) (total_birds : ℕ) 
  (h1 : trees = 7)
  (h2 : blackbirds_per_tree = 3)
  (h3 : total_birds = 34) :
  total_birds - trees * blackbirds_per_tree = 13 := by
  sorry

end magpies_in_park_l1702_170222


namespace number_ordering_l1702_170292

theorem number_ordering : (2 : ℝ)^27 < 10^9 ∧ 10^9 < 5^13 := by sorry

end number_ordering_l1702_170292


namespace triangle_angle_not_all_greater_than_60_l1702_170274

theorem triangle_angle_not_all_greater_than_60 :
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Angles are positive
  (a + b + c = 180) →        -- Sum of angles in a triangle is 180 degrees
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end triangle_angle_not_all_greater_than_60_l1702_170274


namespace gcd_squared_plus_one_l1702_170239

theorem gcd_squared_plus_one (n : ℕ+) : 
  (Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 1) ∨ 
  (Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 5) := by
sorry

end gcd_squared_plus_one_l1702_170239


namespace correct_dye_jobs_scheduled_l1702_170272

def haircut_price : ℕ := 30
def perm_price : ℕ := 40
def dye_job_price : ℕ := 60
def dye_job_cost : ℕ := 10
def haircuts_scheduled : ℕ := 4
def perms_scheduled : ℕ := 1
def tips : ℕ := 50
def total_revenue : ℕ := 310

def dye_jobs_scheduled : ℕ := 
  (total_revenue - (haircut_price * haircuts_scheduled + perm_price * perms_scheduled + tips)) / 
  (dye_job_price - dye_job_cost)

theorem correct_dye_jobs_scheduled : dye_jobs_scheduled = 2 := by
  sorry

end correct_dye_jobs_scheduled_l1702_170272


namespace not_divisible_by_101_l1702_170263

theorem not_divisible_by_101 (k : ℤ) : ¬(101 ∣ k^2 + k + 1) := by
  sorry

end not_divisible_by_101_l1702_170263


namespace indoor_tables_count_l1702_170285

/-- The number of indoor tables in a bakery. -/
def num_indoor_tables : ℕ := sorry

/-- The number of outdoor tables in a bakery. -/
def num_outdoor_tables : ℕ := 12

/-- The number of chairs per indoor table. -/
def chairs_per_indoor_table : ℕ := 3

/-- The number of chairs per outdoor table. -/
def chairs_per_outdoor_table : ℕ := 3

/-- The total number of chairs in the bakery. -/
def total_chairs : ℕ := 60

/-- Theorem stating that the number of indoor tables is 8. -/
theorem indoor_tables_count : num_indoor_tables = 8 := by
  sorry

end indoor_tables_count_l1702_170285


namespace two_numbers_difference_l1702_170267

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 17402)
  (b_div_10 : 10 ∣ b)
  (a_eq_b_div_10 : a = b / 10) :
  b - a = 14238 := by
  sorry

end two_numbers_difference_l1702_170267


namespace impossible_all_win_bets_l1702_170216

/-- Represents the outcome of a girl's jump -/
inductive JumpOutcome
  | Success
  | Failure

/-- Represents the three girls -/
inductive Girl
  | First
  | Second
  | Third

/-- The bet condition: one girl's success is equivalent to another girl's failure -/
def betCondition (g1 g2 : Girl) (outcomes : Girl → JumpOutcome) : Prop :=
  outcomes g1 = JumpOutcome.Success ↔ outcomes g2 = JumpOutcome.Failure

/-- The theorem stating it's impossible for all girls to win their bets -/
theorem impossible_all_win_bets :
  ¬∃ (outcomes : Girl → JumpOutcome),
    (betCondition Girl.First Girl.Second outcomes) ∧
    (betCondition Girl.Second Girl.Third outcomes) ∧
    (betCondition Girl.Third Girl.First outcomes) :=
by sorry

end impossible_all_win_bets_l1702_170216


namespace shaded_area_of_concentric_circles_l1702_170247

theorem shaded_area_of_concentric_circles :
  ∀ (r R : ℝ),
  R > 0 →
  r = R / 2 →
  π * R^2 = 100 * π →
  (π * R^2) / 2 + (π * r^2) / 2 = 62.5 * π :=
by
  sorry

end shaded_area_of_concentric_circles_l1702_170247


namespace power_of_product_equals_product_of_powers_l1702_170246

theorem power_of_product_equals_product_of_powers (a : ℝ) : 
  (-2 * a^3)^4 = 16 * a^12 := by
  sorry

end power_of_product_equals_product_of_powers_l1702_170246


namespace election_votes_calculation_l1702_170209

theorem election_votes_calculation (V : ℝ) 
  (h1 : 0.30 * V + 0.25 * V + 0.20 * V + 0.25 * V = V)  -- Condition 2
  (h2 : (0.30 * V + 0.0225 * V) - (0.25 * V + 0.0225 * V) = 1350)  -- Conditions 3, 4, and 5
  : V = 27000 := by
  sorry

end election_votes_calculation_l1702_170209


namespace pure_imaginary_complex_number_l1702_170231

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = -2 := by
  sorry

end pure_imaginary_complex_number_l1702_170231


namespace perpendicular_line_l1702_170289

/-- Given a line l: mx - m²y - 1 = 0, prove that the line perpendicular to l 
    passing through the point (2, 1) has the equation x + y - 3 = 0 -/
theorem perpendicular_line (m : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y => m * x - m^2 * y - 1 = 0
  let p : ℝ × ℝ := (2, 1)
  let perpendicular : ℝ → ℝ → Prop := λ x y => x + y - 3 = 0
  (∀ x y, perpendicular x y ↔ 
    (l x y → False) ∧ 
    (x - p.1) * m + (y - p.2) * (-m^2) = 0 ∧
    perpendicular p.1 p.2) :=
by sorry

end perpendicular_line_l1702_170289


namespace womans_speed_in_still_water_l1702_170294

/-- The speed of a woman rowing a boat in still water, given her downstream performance. -/
theorem womans_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 60) 
  (h2 : distance = 500 / 1000) 
  (h3 : time = 9.99920006399488 / 3600) : 
  ∃ (speed : ℝ), abs (speed - 120.01800180018) < 0.00000000001 := by
  sorry

end womans_speed_in_still_water_l1702_170294


namespace expand_product_l1702_170217

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 2*x + 4) = x^3 + 5*x^2 + 10*x + 12 := by
  sorry

end expand_product_l1702_170217


namespace robe_savings_l1702_170293

def repair_cost : ℕ := 10
def initial_savings : ℕ := 630

def corner_light_cost (repair : ℕ) : ℕ := 2 * repair
def brake_disk_cost (light : ℕ) : ℕ := 3 * light

def total_expenses (repair : ℕ) : ℕ :=
  repair + corner_light_cost repair + 2 * brake_disk_cost (corner_light_cost repair)

theorem robe_savings : 
  initial_savings + total_expenses repair_cost = 780 :=
sorry

end robe_savings_l1702_170293


namespace money_division_l1702_170243

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 406)
  (h_sum : a + b + c = total)
  (h_a_half_b : a = (1/2) * b)
  (h_b_half_c : b = (1/2) * c) :
  c = 232 := by
sorry

end money_division_l1702_170243


namespace joan_seashells_l1702_170250

def seashells_problem (initial_shells : ℕ) (given_away : ℕ) : ℕ :=
  initial_shells - given_away

theorem joan_seashells : seashells_problem 79 63 = 16 := by
  sorry

end joan_seashells_l1702_170250


namespace system_solution_range_l1702_170241

theorem system_solution_range (x y z : ℝ) (a : ℝ) :
  (3 * x^2 + 2 * y^2 + 2 * z^2 = a ∧ 
   4 * x^2 + 4 * y^2 + 5 * z^2 = 1 - a) →
  (2/7 : ℝ) ≤ a ∧ a ≤ (3/7 : ℝ) := by
sorry

end system_solution_range_l1702_170241


namespace morning_sales_is_eight_l1702_170296

/-- Represents the sale of souvenirs at the London Olympics --/
structure SouvenirSale where
  total_souvenirs : Nat
  morning_price : Nat
  afternoon_price : Nat
  morning_sales : Nat
  afternoon_sales : Nat

/-- Checks if the given SouvenirSale satisfies all conditions --/
def is_valid_sale (sale : SouvenirSale) : Prop :=
  sale.total_souvenirs = 24 ∧
  sale.morning_price = 7 ∧
  sale.morning_sales < sale.total_souvenirs / 2 ∧
  sale.morning_sales + sale.afternoon_sales = sale.total_souvenirs ∧
  sale.morning_sales * sale.morning_price + sale.afternoon_sales * sale.afternoon_price = 120

/-- Theorem: The number of souvenirs sold in the morning is 8 --/
theorem morning_sales_is_eight :
  ∃ (sale : SouvenirSale), is_valid_sale sale ∧ sale.morning_sales = 8 :=
sorry

end morning_sales_is_eight_l1702_170296


namespace tooth_arrangements_l1702_170200

def word_length : ℕ := 5
def t_count : ℕ := 2
def o_count : ℕ := 2

theorem tooth_arrangements : 
  (word_length.factorial) / (t_count.factorial * o_count.factorial) = 30 := by
  sorry

end tooth_arrangements_l1702_170200


namespace min_draw_theorem_l1702_170206

/-- A box containing black and white balls -/
structure BallBox where
  black : ℕ
  white : ℕ

/-- The minimum number of balls to draw to ensure at least 2 of the same color -/
def minDrawSameColor (box : BallBox) : ℕ := 3

/-- The minimum number of balls to draw to ensure at least 2 white balls -/
def minDrawTwoWhite (box : BallBox) : ℕ := box.black + 2

/-- Theorem stating the minimum number of balls to draw for both conditions -/
theorem min_draw_theorem (box : BallBox) 
  (h1 : box.black = 100) (h2 : box.white = 100) : 
  minDrawSameColor box = 3 ∧ minDrawTwoWhite box = 102 := by
  sorry

end min_draw_theorem_l1702_170206


namespace three_digit_number_reversal_subtraction_l1702_170273

theorem three_digit_number_reversal_subtraction (c : ℕ) 
  (h1 : c < 10) 
  (h2 : c + 3 < 10) 
  (h3 : 2 * c < 10) : 
  (121 * c + 300 - (121 * c + 3)) % 10 = 7 := by
sorry

end three_digit_number_reversal_subtraction_l1702_170273


namespace pells_equation_unique_solution_l1702_170288

-- Define the fundamental solution
def fundamental_solution (x₀ y₀ : ℤ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1 ∧ x₀ > 0 ∧ y₀ > 0

-- Define the property that all prime factors of x divide x₀
def all_prime_factors_divide (x x₀ : ℤ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p : ℤ) ∣ x → (p : ℤ) ∣ x₀

-- Main theorem
theorem pells_equation_unique_solution
  (x₀ y₀ x y : ℤ)
  (h_fund : fundamental_solution x₀ y₀)
  (h_sol : x^2 - 2003 * y^2 = 1)
  (h_pos : x > 0 ∧ y > 0)
  (h_divide : all_prime_factors_divide x x₀) :
  x = x₀ ∧ y = y₀ :=
sorry

end pells_equation_unique_solution_l1702_170288


namespace correct_average_l1702_170254

theorem correct_average (n : ℕ) (initial_avg : ℚ) (correction1 : ℚ) (wrong2 : ℚ) (correct2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  correction1 = 19 →
  wrong2 = 13 →
  correct2 = 31 →
  let initial_sum := n * initial_avg
  let corrected_sum := initial_sum - correction1 - wrong2 + correct2
  let corrected_avg := corrected_sum / n
  corrected_avg = 40.1 := by
  sorry

end correct_average_l1702_170254


namespace circles_intersection_l1702_170266

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 10*y + 36 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 5)

-- Theorem statement
theorem circles_intersection :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y ↔ (x, y) = intersection_point) ∧
  (intersection_point.1 * intersection_point.2 = 15) :=
sorry

end circles_intersection_l1702_170266


namespace employment_percentage_l1702_170255

theorem employment_percentage (total_population : ℝ) (employed_population : ℝ) 
  (h1 : employed_population > 0) 
  (h2 : employed_population ≤ total_population)
  (h3 : employed_population * 0.7 + employed_population * 0.3 = employed_population)
  (h4 : employed_population * 0.3 = total_population * 0.21) : 
  employed_population / total_population = 0.7 := by
sorry

end employment_percentage_l1702_170255


namespace number_puzzle_l1702_170252

theorem number_puzzle (x y : ℕ) : x = 20 → 3 * (2 * x + y) = 135 → y = 5 := by
  sorry

end number_puzzle_l1702_170252


namespace total_serving_time_is_44_minutes_l1702_170259

/-- Represents the properties of a soup pot -/
structure SoupPot where
  gallons : Float
  servingRate : Float  -- bowls per minute
  bowlSize : Float     -- ounces per bowl

/-- Calculates the time to serve a pot of soup -/
def timeToServe (pot : SoupPot) : Float :=
  let ouncesInPot := pot.gallons * 128
  let bowls := (ouncesInPot / pot.bowlSize).floor
  bowls / pot.servingRate

/-- Proves that the total serving time for all soups is 44 minutes when rounded -/
theorem total_serving_time_is_44_minutes (pot1 pot2 pot3 : SoupPot)
  (h1 : pot1 = { gallons := 8, servingRate := 5, bowlSize := 10 })
  (h2 : pot2 = { gallons := 5.5, servingRate := 4, bowlSize := 12 })
  (h3 : pot3 = { gallons := 3.25, servingRate := 6, bowlSize := 8 }) :
  (timeToServe pot1 + timeToServe pot2 + timeToServe pot3).round = 44 := by
  sorry

#eval (timeToServe { gallons := 8, servingRate := 5, bowlSize := 10 } +
       timeToServe { gallons := 5.5, servingRate := 4, bowlSize := 12 } +
       timeToServe { gallons := 3.25, servingRate := 6, bowlSize := 8 }).round

end total_serving_time_is_44_minutes_l1702_170259


namespace area_equality_l1702_170235

/-- Represents a regular hexagon with side length 1 -/
def RegularHexagon : Set (ℝ × ℝ) := sorry

/-- Represents an equilateral triangle with side length 1 -/
def EquilateralTriangle : Set (ℝ × ℝ) := sorry

/-- Represents the region R, which is the union of the hexagon and 18 triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- Represents the smallest convex polygon S that contains R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of a set in the plane -/
def area : Set (ℝ × ℝ) → ℝ := sorry

theorem area_equality : area S = area R := by sorry

end area_equality_l1702_170235


namespace video_votes_l1702_170205

theorem video_votes (total_votes : ℕ) (score : ℤ) (like_percentage : ℚ) : 
  score = 0 + (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) →
  like_percentage = 3/4 →
  score = 120 →
  total_votes = 240 := by
sorry

end video_votes_l1702_170205


namespace dividend_calculation_l1702_170271

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 149 := by
sorry

end dividend_calculation_l1702_170271


namespace sum_of_integers_l1702_170238

theorem sum_of_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120 →
  p + q + r + s + t = 32 := by
sorry

end sum_of_integers_l1702_170238


namespace at_least_one_calculation_incorrect_l1702_170281

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem at_least_one_calculation_incorrect
  (first_num second_num : ℕ)
  (sum_digits_first sum_digits_second : ℕ)
  (h1 : first_num * sum_digits_second = 201320132013)
  (h2 : second_num * sum_digits_first = 201420142014)
  (h3 : is_divisible_by_9 201320132013)
  (h4 : ¬ is_divisible_by_9 201420142014) :
  ¬(∀ (x y : ℕ), x * y = 201320132013 → is_divisible_by_9 (x * y)) ∨
  ¬(∀ (x y : ℕ), x * y = 201420142014 → ¬ is_divisible_by_9 (x * y)) :=
by sorry

end at_least_one_calculation_incorrect_l1702_170281


namespace divisible_by_seventeen_l1702_170282

theorem divisible_by_seventeen (n : ℕ) : 17 ∣ (2^(5*n+3) + 5^n * 3^(n+2)) := by
  sorry

end divisible_by_seventeen_l1702_170282


namespace product_remainder_mod_10_l1702_170228

theorem product_remainder_mod_10 : (2457 * 7963 * 92324) % 10 = 4 := by
  sorry

end product_remainder_mod_10_l1702_170228


namespace conic_is_ellipse_with_major_axis_8_l1702_170226

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A conic section passing through five points -/
structure ConicSection where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  p5 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The conic section defined by the given five points -/
def givenConic : ConicSection :=
  { p1 := { x := -2, y := 0 }
  , p2 := { x := 0,  y := 1 }
  , p3 := { x := 0,  y := 3 }
  , p4 := { x := 4,  y := 1 }
  , p5 := { x := 4,  y := 3 }
  }

/-- Definition of an ellipse -/
def isEllipse (c : ConicSection) : Prop :=
  ∃ (center : Point) (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    ∀ (p : Point),
      ((p.x - center.x)^2 / a^2 + (p.y - center.y)^2 / b^2 = 1) ↔
      (p = c.p1 ∨ p = c.p2 ∨ p = c.p3 ∨ p = c.p4 ∨ p = c.p5)

/-- The main theorem -/
theorem conic_is_ellipse_with_major_axis_8 :
  (¬ collinear givenConic.p1 givenConic.p2 givenConic.p3) ∧
  (¬ collinear givenConic.p1 givenConic.p2 givenConic.p4) ∧
  (¬ collinear givenConic.p1 givenConic.p2 givenConic.p5) ∧
  (¬ collinear givenConic.p1 givenConic.p3 givenConic.p4) ∧
  (¬ collinear givenConic.p1 givenConic.p3 givenConic.p5) ∧
  (¬ collinear givenConic.p1 givenConic.p4 givenConic.p5) ∧
  (¬ collinear givenConic.p2 givenConic.p3 givenConic.p4) ∧
  (¬ collinear givenConic.p2 givenConic.p3 givenConic.p5) ∧
  (¬ collinear givenConic.p2 givenConic.p4 givenConic.p5) ∧
  (¬ collinear givenConic.p3 givenConic.p4 givenConic.p5) →
  isEllipse givenConic ∧ ∃ (a : ℝ), a = 8 :=
by sorry

end conic_is_ellipse_with_major_axis_8_l1702_170226


namespace triangle_properties_l1702_170287

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: If b - (1/2)c = a cos C, 4(b + c) = 3bc, and a = 2√3 in a triangle ABC,
    then angle A = 60° and the area of the triangle is 2√3 --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b - (1/2) * t.c = t.a * Real.cos t.C)
  (h2 : 4 * (t.b + t.c) = 3 * t.b * t.c)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end triangle_properties_l1702_170287


namespace total_age_in_10_years_l1702_170253

def jackson_age : ℕ := 20
def mandy_age : ℕ := jackson_age + 10
def adele_age : ℕ := (3 * jackson_age) / 4

theorem total_age_in_10_years : 
  (jackson_age + 10) + (mandy_age + 10) + (adele_age + 10) = 95 := by
  sorry

end total_age_in_10_years_l1702_170253


namespace inequality_proof_l1702_170268

theorem inequality_proof (x y z t : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (ht : 0 < t ∧ t < 1) : 
  Real.sqrt (x^2 + (1-t)^2) + Real.sqrt (y^2 + (1-x)^2) + 
  Real.sqrt (z^2 + (1-y)^2) + Real.sqrt (t^2 + (1-z)^2) < 4 := by
  sorry

end inequality_proof_l1702_170268


namespace tangent_line_at_2_l1702_170262

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_2 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = 13 * x - 32 :=
by sorry

end tangent_line_at_2_l1702_170262


namespace point_on_number_line_l1702_170218

theorem point_on_number_line (x : ℝ) : 
  abs x = 5.5 → x = 5.5 ∨ x = -5.5 := by
  sorry

end point_on_number_line_l1702_170218


namespace cube_lateral_surface_area_l1702_170201

/-- The lateral surface area of a cube with side length 12 meters is 576 square meters. -/
theorem cube_lateral_surface_area :
  let side_length : ℝ := 12
  let lateral_surface_area : ℝ := 4 * side_length * side_length
  lateral_surface_area = 576 := by sorry

end cube_lateral_surface_area_l1702_170201


namespace interest_rate_difference_l1702_170208

/-- Proves that the difference between two interest rates is 1/3% when one rate produces $81 more interest than the other over 3 years for a $900 principal using simple interest. -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (rate1 : ℝ) (rate2 : ℝ) : 
  principal = 900 → 
  time = 3 → 
  principal * rate2 * time - principal * rate1 * time = 81 → 
  rate2 - rate1 = 1/3 * (1/100) := by sorry

end interest_rate_difference_l1702_170208


namespace fraction_problem_l1702_170258

theorem fraction_problem (x : ℚ) : 
  (9 - x = 4.5 * (1/2)) → x = 27/4 := by
  sorry

end fraction_problem_l1702_170258


namespace degree_three_polynomial_l1702_170210

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 2 - 3*x - 7*x^3 + 12*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- Theorem: The value of c that makes h(x) a polynomial of degree 3 is -5/12 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -5/12 ∧ 
  (∀ (x : ℝ), h c x = 1 + (-12 - 3*c)*x + (3 - 0*c)*x^2 + (-4 - 7*c)*x^3) :=
sorry

end degree_three_polynomial_l1702_170210


namespace book_arrangement_theorem_l1702_170237

theorem book_arrangement_theorem :
  let total_books : ℕ := 7
  let identical_books : ℕ := 3
  let different_books : ℕ := 4
  (total_books = identical_books + different_books) →
  (Nat.factorial total_books / Nat.factorial identical_books = 840) := by
  sorry

end book_arrangement_theorem_l1702_170237


namespace smallest_rectangles_covering_square_l1702_170207

theorem smallest_rectangles_covering_square :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → 
    (∃ (s : ℕ), s > 0 ∧ 
      s * s = m * 3 * 4 ∧ 
      s % 3 = 0 ∧ 
      s % 4 = 0) → 
    m ≥ n) ∧
  (∃ (s : ℕ), s > 0 ∧ 
    s * s = n * 3 * 4 ∧ 
    s % 3 = 0 ∧ 
    s % 4 = 0) ∧
  n = 12 :=
by sorry

end smallest_rectangles_covering_square_l1702_170207


namespace mrs_hilt_pies_l1702_170234

def pecan_pies : ℝ := 16.0
def apple_pies : ℝ := 14.0
def increase_factor : ℝ := 5.0

theorem mrs_hilt_pies : 
  (pecan_pies + apple_pies) * increase_factor = 150.0 := by
  sorry

end mrs_hilt_pies_l1702_170234


namespace max_value_expression_l1702_170244

theorem max_value_expression (a b c d : ℝ) 
  (ha : -13.5 ≤ a ∧ a ≤ 13.5)
  (hb : -13.5 ≤ b ∧ b ≤ 13.5)
  (hc : -13.5 ≤ c ∧ c ≤ 13.5)
  (hd : -13.5 ≤ d ∧ d ≤ 13.5) :
  (∀ x y z w, -13.5 ≤ x ∧ x ≤ 13.5 → 
              -13.5 ≤ y ∧ y ≤ 13.5 → 
              -13.5 ≤ z ∧ z ≤ 13.5 → 
              -13.5 ≤ w ∧ w ≤ 13.5 → 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 756) ∧ 
  (∃ x y z w, -13.5 ≤ x ∧ x ≤ 13.5 ∧ 
              -13.5 ≤ y ∧ y ≤ 13.5 ∧ 
              -13.5 ≤ z ∧ z ≤ 13.5 ∧ 
              -13.5 ≤ w ∧ w ≤ 13.5 ∧ 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 756) :=
by
  sorry

end max_value_expression_l1702_170244


namespace sams_money_l1702_170230

-- Define the value of a penny and a quarter in cents
def penny_value : ℚ := 1
def quarter_value : ℚ := 25

-- Define the number of pennies and quarters Sam has
def num_pennies : ℕ := 9
def num_quarters : ℕ := 7

-- Calculate the total value in cents
def total_value : ℚ := num_pennies * penny_value + num_quarters * quarter_value

-- Theorem to prove
theorem sams_money : total_value = 184 := by
  sorry

end sams_money_l1702_170230


namespace age_difference_proof_l1702_170283

theorem age_difference_proof : ∃! n : ℝ, n > 0 ∧ 
  ∃ A C : ℝ, A > 0 ∧ C > 0 ∧ 
    A = C + n ∧ 
    A - 2 = 4 * (C - 2) ∧ 
    A = C^3 ∧ 
    n = 1.875 := by
  sorry

end age_difference_proof_l1702_170283


namespace sammy_math_problems_l1702_170280

theorem sammy_math_problems (total : ℕ) (left : ℕ) (finished : ℕ) : 
  total = 9 → left = 7 → finished = total - left → finished = 2 := by
sorry

end sammy_math_problems_l1702_170280


namespace systematic_sampling_l1702_170278

theorem systematic_sampling (total_bags : Nat) (num_groups : Nat) (fourth_group_sample : Nat) (first_group_sample : Nat) :
  total_bags = 50 →
  num_groups = 5 →
  fourth_group_sample = 36 →
  first_group_sample = 6 →
  (total_bags / num_groups) * 3 + first_group_sample = fourth_group_sample :=
by sorry

end systematic_sampling_l1702_170278


namespace triathlon_bicycle_speed_l1702_170291

/-- Triathlon problem -/
theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance : ℝ) :
  total_time = 2 →
  swim_distance = 0.5 →
  swim_speed = 3 →
  run_distance = 5 →
  run_speed = 10 →
  bike_distance = 20 →
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 15 := by
  sorry


end triathlon_bicycle_speed_l1702_170291


namespace inlet_pipe_rate_l1702_170236

/-- Given a tank with the following properties:
    - Capacity of 4320 litres
    - Empties through a leak in 6 hours when full
    - Empties in 12 hours when both the leak and an inlet pipe are open
    This theorem proves that the rate at which the inlet pipe fills water is 6 litres per minute. -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ)
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : combined_empty_time = 12) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / combined_empty_time
  let inlet_rate := leak_rate - net_empty_rate
  inlet_rate / 60 = 6 := by sorry

end inlet_pipe_rate_l1702_170236


namespace complementary_angles_difference_l1702_170249

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 4 / 5 →  -- ratio of angles is 4:5
  |a - b| = 10 := by  -- positive difference is 10°
sorry

end complementary_angles_difference_l1702_170249


namespace square_1849_product_l1702_170223

theorem square_1849_product (y : ℤ) (h : y^2 = 1849) : (y+2)*(y-2) = 1845 := by
  sorry

end square_1849_product_l1702_170223


namespace middle_circle_radius_l1702_170212

/-- A configuration of five consecutively tangent circles between two parallel lines -/
structure CircleConfiguration where
  /-- The radii of the five circles, from smallest to largest -/
  radii : Fin 5 → ℝ
  /-- The radii are positive -/
  radii_pos : ∀ i, 0 < radii i
  /-- The radii are in ascending order -/
  radii_ascending : ∀ i j, i < j → radii i ≤ radii j
  /-- The circles are tangent to each other -/
  tangent_circles : ∀ i : Fin 4, radii i + radii (i + 1) = radii (i + 1) - radii i

/-- The theorem stating that the middle circle's radius is 10 cm -/
theorem middle_circle_radius
  (config : CircleConfiguration)
  (h_smallest : config.radii 0 = 5)
  (h_largest : config.radii 4 = 15) :
  config.radii 2 = 10 :=
sorry

end middle_circle_radius_l1702_170212


namespace larger_integer_proof_l1702_170295

theorem larger_integer_proof (x y : ℕ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 272 → x = 24 := by
  sorry

end larger_integer_proof_l1702_170295


namespace parallel_lines_perpendicular_lines_l1702_170276

/-- Two lines in a plane -/
structure TwoLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y => (a - 1) * x + 2 * y + 1 = 0
  l₂ : ℝ → ℝ → Prop := λ x y => x + a * y + 3 = 0

/-- Parallel lines theorem -/
theorem parallel_lines (lines : TwoLines) :
  (∀ x y, lines.l₁ x y ↔ ∃ k, lines.l₂ (x + k) (y + k)) →
  lines.a = 2 ∨ lines.a = -1 := by sorry

/-- Perpendicular lines theorem -/
theorem perpendicular_lines (lines : TwoLines) :
  (∀ x₁ y₁ x₂ y₂, lines.l₁ x₁ y₁ → lines.l₂ x₂ y₂ → 
    ((x₂ - x₁) * (lines.a - 1) + (y₂ - y₁) * 2 = 0) ∧
    ((x₂ - x₁) * 1 + (y₂ - y₁) * lines.a = 0)) →
  (lines.a - 1) + 2 * lines.a = 0 := by sorry

end parallel_lines_perpendicular_lines_l1702_170276


namespace kernels_in_first_bag_l1702_170215

/-- The number of kernels in the first bag -/
def first_bag : ℕ := 74

/-- The number of popped kernels in the first bag -/
def popped_first : ℕ := 60

/-- The number of kernels in the second bag -/
def second_bag : ℕ := 50

/-- The number of popped kernels in the second bag -/
def popped_second : ℕ := 42

/-- The number of kernels in the third bag -/
def third_bag : ℕ := 100

/-- The number of popped kernels in the third bag -/
def popped_third : ℕ := 82

/-- The average percentage of popped kernels -/
def avg_percentage : ℚ := 82/100

theorem kernels_in_first_bag :
  (popped_first + popped_second + popped_third : ℚ) / 
  (first_bag + second_bag + third_bag : ℚ) = avg_percentage :=
by sorry

end kernels_in_first_bag_l1702_170215


namespace jenny_hotel_cost_l1702_170219

/-- The total cost of a hotel stay for a group of people -/
def total_cost (cost_per_night_per_person : ℕ) (num_people : ℕ) (num_nights : ℕ) : ℕ :=
  cost_per_night_per_person * num_people * num_nights

/-- Theorem stating the total cost for Jenny and her friends' hotel stay -/
theorem jenny_hotel_cost :
  total_cost 40 3 3 = 360 := by
  sorry

end jenny_hotel_cost_l1702_170219


namespace doughnuts_left_l1702_170203

def doughnuts_problem (total_doughnuts : ℕ) (total_staff : ℕ) 
  (staff_3 : ℕ) (staff_2 : ℕ) (doughnuts_3 : ℕ) (doughnuts_2 : ℕ) (doughnuts_4 : ℕ) : Prop :=
  total_doughnuts = 120 ∧
  total_staff = 35 ∧
  staff_3 = 15 ∧
  staff_2 = 10 ∧
  doughnuts_3 = 3 ∧
  doughnuts_2 = 2 ∧
  doughnuts_4 = 4 ∧
  total_staff = staff_3 + staff_2 + (total_staff - staff_3 - staff_2)

theorem doughnuts_left (total_doughnuts : ℕ) (total_staff : ℕ) 
  (staff_3 : ℕ) (staff_2 : ℕ) (doughnuts_3 : ℕ) (doughnuts_2 : ℕ) (doughnuts_4 : ℕ) :
  doughnuts_problem total_doughnuts total_staff staff_3 staff_2 doughnuts_3 doughnuts_2 doughnuts_4 →
  total_doughnuts - (staff_3 * doughnuts_3 + staff_2 * doughnuts_2 + (total_staff - staff_3 - staff_2) * doughnuts_4) = 15 :=
by
  sorry

end doughnuts_left_l1702_170203


namespace salary_change_l1702_170251

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let after_first_decrease := initial_salary * 0.5
  let after_increase := after_first_decrease * 1.3
  let final_salary := after_increase * 0.8
  (final_salary - initial_salary) / initial_salary = -0.48 :=
by sorry

end salary_change_l1702_170251


namespace perimeter_of_parallelogram_l1702_170257

/-- Triangle PQR with properties -/
structure Triangle (P Q R : ℝ × ℝ) :=
  (pq_eq_pr : dist P Q = dist P R)
  (pq_length : dist P Q = 30)
  (qr_length : dist Q R = 28)

/-- Point on a line segment -/
def PointOnSegment (A B M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B

/-- Parallel lines -/
def Parallel (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)

/-- Perimeter of a quadrilateral -/
def Perimeter (A B C D : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D A

theorem perimeter_of_parallelogram
  (P Q R M N O : ℝ × ℝ)
  (tri : Triangle P Q R)
  (m_on_pq : PointOnSegment P Q M)
  (n_on_qr : PointOnSegment Q R N)
  (o_on_pr : PointOnSegment P R O)
  (mn_parallel_pr : Parallel M N P R)
  (no_parallel_pq : Parallel N O P Q) :
  Perimeter P M N O = 60 := by
  sorry

end perimeter_of_parallelogram_l1702_170257


namespace count_positive_area_triangles_l1702_170214

/-- The total number of points in the grid -/
def total_points : ℕ := 7

/-- The number of sets of three collinear points -/
def collinear_sets : ℕ := 5

/-- The number of triangles with positive area -/
def positive_area_triangles : ℕ := 30

/-- Theorem stating the number of triangles with positive area -/
theorem count_positive_area_triangles :
  (Nat.choose total_points 3) - collinear_sets = positive_area_triangles :=
by sorry

end count_positive_area_triangles_l1702_170214


namespace fraction_A_proof_l1702_170284

/-- The fraction that A gets compared to what B and C together get -/
def fraction_A (total amount_A amount_B amount_C : ℚ) : ℚ :=
  amount_A / (amount_B + amount_C)

theorem fraction_A_proof 
  (total : ℚ) 
  (amount_A amount_B amount_C : ℚ) 
  (h1 : total = 1260)
  (h2 : ∃ x : ℚ, amount_A = x * (amount_B + amount_C))
  (h3 : amount_B = 2/7 * (amount_A + amount_C))
  (h4 : amount_A = amount_B + 35)
  (h5 : total = amount_A + amount_B + amount_C) :
  fraction_A total amount_A amount_B amount_C = 63/119 := by
  sorry

#eval fraction_A 1260 315 280 665

end fraction_A_proof_l1702_170284


namespace unique_prime_mersenne_sequence_l1702_170265

theorem unique_prime_mersenne_sequence (n : ℕ+) : 
  (Nat.Prime (2^n.val - 1) ∧ 
   Nat.Prime (2^(n.val + 2) - 1) ∧ 
   ¬(7 ∣ (2^(n.val + 1) - 1))) ↔ 
  n = 3 := by
sorry

end unique_prime_mersenne_sequence_l1702_170265


namespace well_depth_is_896_l1702_170213

/-- The depth of the well in feet -/
def well_depth : ℝ := 896

/-- The initial velocity of the stone in feet per second (downward) -/
def initial_velocity : ℝ := 32

/-- The total time until the sound is heard in seconds -/
def total_time : ℝ := 8.5

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1120

/-- The stone's displacement function in feet, given time t in seconds -/
def stone_displacement (t : ℝ) : ℝ := 32 * t + 16 * t^2

/-- Theorem stating that the well depth is 896 feet given the conditions -/
theorem well_depth_is_896 :
  ∃ (t₁ : ℝ), 
    stone_displacement t₁ = well_depth ∧
    t₁ + well_depth / sound_velocity = total_time :=
by sorry

end well_depth_is_896_l1702_170213


namespace cost_price_calculation_l1702_170286

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 288 →
  profit_percentage = 0.20 →
  selling_price = (1 + profit_percentage) * (selling_price / (1 + profit_percentage)) →
  selling_price / (1 + profit_percentage) = 240 := by
sorry

end cost_price_calculation_l1702_170286


namespace river_road_ratio_l1702_170290

def river_road_vehicles (cars : ℕ) (bus_difference : ℕ) : ℕ × ℕ :=
  (cars - bus_difference, cars)

def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem river_road_ratio :
  let (buses, cars) := river_road_vehicles 100 90
  simplify_ratio buses cars = (1, 10) := by
sorry

end river_road_ratio_l1702_170290


namespace travel_theorem_l1702_170224

/-- Represents the scenario of two people traveling with a bicycle --/
structure TravelScenario where
  distance : ℝ
  walkSpeed : ℝ
  bikeSpeed : ℝ
  cLeaveTime : ℝ

/-- Checks if the travel scenario results in simultaneous arrival --/
def simultaneousArrival (scenario : TravelScenario) : Prop :=
  let t := scenario.distance / scenario.walkSpeed
  let meetTime := (scenario.distance * scenario.walkSpeed) / (scenario.bikeSpeed + scenario.walkSpeed)
  let cTravelTime := scenario.distance / scenario.walkSpeed - scenario.cLeaveTime
  t = meetTime + (scenario.distance - meetTime * scenario.bikeSpeed) / scenario.walkSpeed

/-- The main theorem to prove --/
theorem travel_theorem (scenario : TravelScenario) 
  (h1 : scenario.distance = 15)
  (h2 : scenario.walkSpeed = 6)
  (h3 : scenario.bikeSpeed = 15)
  (h4 : scenario.cLeaveTime = 3/11) :
  simultaneousArrival scenario := by
  sorry


end travel_theorem_l1702_170224


namespace proportional_sum_l1702_170298

theorem proportional_sum (M N : ℚ) : 
  (3 / 5 : ℚ) = M / 45 ∧ (3 / 5 : ℚ) = 60 / N → M + N = 127 := by
  sorry

end proportional_sum_l1702_170298


namespace sum_coordinates_of_D_l1702_170225

/-- Given a line segment CD where C(11, 4) and P(5, 10) is the midpoint,
    prove that the sum of the coordinates of point D is 15. -/
theorem sum_coordinates_of_D (C D P : ℝ × ℝ) : 
  C = (11, 4) →
  P = (5, 10) →
  P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 15 := by
  sorry

end sum_coordinates_of_D_l1702_170225


namespace radio_selling_price_l1702_170261

theorem radio_selling_price 
  (purchase_price : ℝ) 
  (overhead_expenses : ℝ) 
  (profit_percent : ℝ) 
  (h1 : purchase_price = 225)
  (h2 : overhead_expenses = 20)
  (h3 : profit_percent = 22.448979591836732) : 
  ∃ (selling_price : ℝ), selling_price = 300 ∧ 
  selling_price = purchase_price + overhead_expenses + 
  (profit_percent / 100) * (purchase_price + overhead_expenses) :=
sorry

end radio_selling_price_l1702_170261


namespace degree_of_x2y3_is_five_l1702_170232

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

/-- Theorem: The degree of the monomial x^2 * y^3 is 5 -/
theorem degree_of_x2y3_is_five : degree_of_monomial 2 3 = 5 := by
  sorry

end degree_of_x2y3_is_five_l1702_170232


namespace ant_walk_length_l1702_170211

theorem ant_walk_length (r₁ r₂ : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 15) : 
  let quarter_large := (1/4) * 2 * Real.pi * r₂
  let half_small := (1/2) * 2 * Real.pi * r₁
  let radial := r₂ - r₁
  quarter_large + half_small + 2 * radial = 12.5 * Real.pi + 20 := by
sorry

end ant_walk_length_l1702_170211


namespace married_couples_with_more_than_three_children_l1702_170277

theorem married_couples_with_more_than_three_children 
  (total_couples : ℝ) 
  (couples_more_than_one_child : ℝ) 
  (couples_two_or_three_children : ℝ) 
  (h1 : couples_more_than_one_child = (3 / 5) * total_couples)
  (h2 : couples_two_or_three_children = 0.2 * total_couples) :
  (couples_more_than_one_child - couples_two_or_three_children) / total_couples = 2 / 5 := by
  sorry

end married_couples_with_more_than_three_children_l1702_170277


namespace quadratic_roots_and_inequality_l1702_170229

theorem quadratic_roots_and_inequality (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 4 * (a + 1) * x + 3 * a + 3
  (∃ x y : ℝ, x < 2 ∧ y < 2 ∧ x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∀ x : ℝ, (a + 1) * x^2 - a * x + a - 1 < 0) ↔ a < -2 * Real.sqrt 3 / 3 :=
by sorry

end quadratic_roots_and_inequality_l1702_170229


namespace rectangle_rotation_path_length_l1702_170221

/-- The length of the path traveled by point A of a rectangle ABCD after two 90° rotations -/
theorem rectangle_rotation_path_length (A B C D : ℝ × ℝ) : 
  let AB := 3
  let BC := 8
  let first_rotation_radius := Real.sqrt (AB^2 + BC^2)
  let second_rotation_radius := BC
  let first_arc_length := (π / 2) * first_rotation_radius
  let second_arc_length := (π / 2) * second_rotation_radius
  let total_path_length := first_arc_length + second_arc_length
  total_path_length = (4 + Real.sqrt 73 / 2) * π := by
sorry


end rectangle_rotation_path_length_l1702_170221


namespace watch_profit_percentage_l1702_170233

/-- Calculate the percentage of profit given the cost price and selling price -/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The percentage of profit for a watch with cost price 90 and selling price 144 is 60% -/
theorem watch_profit_percentage :
  percentage_profit 90 144 = 60 := by
  sorry

end watch_profit_percentage_l1702_170233


namespace percentage_calculation_l1702_170202

theorem percentage_calculation (x : ℝ) : 
  (0.20 * x = 80) → (0.40 * x = 160) := by
  sorry

end percentage_calculation_l1702_170202


namespace lily_lemur_hops_l1702_170256

theorem lily_lemur_hops : 
  let hop_fraction : ℚ := 1/4
  let num_hops : ℕ := 4
  let total_distance : ℚ := (1 - (1 - hop_fraction)^num_hops) / hop_fraction
  total_distance = 175/256 := by sorry

end lily_lemur_hops_l1702_170256


namespace watch_cost_price_l1702_170220

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  cp > 0 ∧
  cp * 1.04 - cp * 0.84 = 140 ∧
  cp = 700 := by
  sorry

end watch_cost_price_l1702_170220


namespace smallest_a_equals_36_l1702_170269

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f (2 * x) = 2 * f x) ∧
  (∀ x, 1 < x ∧ x ≤ 2 → f x = 2 - x)

/-- The theorem statement -/
theorem smallest_a_equals_36 (f : ℝ → ℝ) (hf : special_function f) :
  (∃ a : ℝ, a > 0 ∧ f a = f 2020 ∧ ∀ b, b > 0 ∧ f b = f 2020 → a ≤ b) →
  (∃ a : ℝ, a > 0 ∧ f a = f 2020 ∧ ∀ b, b > 0 ∧ f b = f 2020 → a ≤ b) ∧ a = 36 :=
sorry

end smallest_a_equals_36_l1702_170269
