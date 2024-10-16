import Mathlib

namespace NUMINAMATH_CALUDE_orange_probability_l1995_199594

/-- Given a box of fruit with apples and oranges, calculate the probability of selecting an orange -/
theorem orange_probability (apples oranges : ℕ) (h1 : apples = 20) (h2 : oranges = 10) :
  (oranges : ℚ) / ((apples : ℚ) + (oranges : ℚ)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_probability_l1995_199594


namespace NUMINAMATH_CALUDE_last_element_value_l1995_199589

/-- Represents a triangular number table -/
def TriangularTable (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the table contains the first n positive integers -/
def FirstRowCorrect (t : TriangularTable 100) : Prop :=
  ∀ i : Fin 100, t 0 i = i.val + 1

/-- Each element (except in the first row) is the sum of two elements above it -/
def ElementSum (t : TriangularTable 100) : Prop :=
  ∀ (i : Fin 99) (j : Fin (99 - i.val)), 
    t (i + 1) j = t i j + t i (j + 1)

/-- The last row contains only one element -/
def LastRowSingleton (t : TriangularTable 100) : Prop :=
  ∀ j : Fin 100, j.val > 0 → t 99 j = 0

/-- The main theorem: given the conditions, the last element is 101 * 2^98 -/
theorem last_element_value (t : TriangularTable 100) 
  (h1 : FirstRowCorrect t) 
  (h2 : ElementSum t)
  (h3 : LastRowSingleton t) : 
  t 99 0 = 101 * 2^98 := by
  sorry

end NUMINAMATH_CALUDE_last_element_value_l1995_199589


namespace NUMINAMATH_CALUDE_family_probability_l1995_199504

theorem family_probability : 
  let p_boy := (1 : ℚ) / 2
  let p_girl := (1 : ℚ) / 2
  let p_all_boys := p_boy ^ 4
  let p_all_girls := p_girl ^ 4
  let p_at_least_one_of_each := 1 - (p_all_boys + p_all_girls)
  p_at_least_one_of_each = (7 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_family_probability_l1995_199504


namespace NUMINAMATH_CALUDE_jerry_max_showers_l1995_199507

/-- Represents the water usage scenario for Jerry in July --/
structure WaterUsage where
  total_allowance : ℕ
  drinking_cooking : ℕ
  shower_usage : ℕ
  pool_length : ℕ
  pool_width : ℕ
  pool_height : ℕ
  gallon_per_cubic_foot : ℕ
  leakage_rate : ℕ
  days_in_july : ℕ

/-- Calculates the maximum number of showers Jerry can take in July --/
def max_showers (w : WaterUsage) : ℕ :=
  let pool_volume := w.pool_length * w.pool_width * w.pool_height
  let pool_water := pool_volume * w.gallon_per_cubic_foot
  let total_leakage := w.leakage_rate * w.days_in_july
  let water_for_showers := w.total_allowance - w.drinking_cooking - pool_water - total_leakage
  water_for_showers / w.shower_usage

/-- Theorem stating that Jerry can take at most 7 showers in July --/
theorem jerry_max_showers :
  let w : WaterUsage := {
    total_allowance := 1000,
    drinking_cooking := 100,
    shower_usage := 20,
    pool_length := 10,
    pool_width := 10,
    pool_height := 6,
    gallon_per_cubic_foot := 1,
    leakage_rate := 5,
    days_in_july := 31
  }
  max_showers w = 7 := by
  sorry


end NUMINAMATH_CALUDE_jerry_max_showers_l1995_199507


namespace NUMINAMATH_CALUDE_no_snow_probability_l1995_199524

theorem no_snow_probability (p : ℚ) : 
  p = 2/3 → (1 - p)^4 = 1/81 := by sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1995_199524


namespace NUMINAMATH_CALUDE_f_max_value_l1995_199531

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℚ) * (S (n + 1) : ℚ))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1/50) ∧ (∃ n : ℕ, f n = 1/50) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1995_199531


namespace NUMINAMATH_CALUDE_energetic_cycling_hours_l1995_199525

theorem energetic_cycling_hours 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (energetic_speed : ℝ) 
  (fatigued_speed : ℝ) 
  (h1 : total_distance = 150) 
  (h2 : total_time = 12) 
  (h3 : energetic_speed = 15) 
  (h4 : fatigued_speed = 10) : 
  ∃ (energetic_hours : ℝ), 
    energetic_hours * energetic_speed + (total_time - energetic_hours) * fatigued_speed = total_distance ∧ 
    energetic_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_energetic_cycling_hours_l1995_199525


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1995_199572

def to_base_7_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem balls_in_boxes (n : ℕ) (h : n = 3010) : 
  (to_base_7_digits n).sum = 16 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1995_199572


namespace NUMINAMATH_CALUDE_min_addition_to_prime_l1995_199581

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  n = 10 * a + b ∧ 2 * a * b = n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_addition_to_prime :
  ∃ n : ℕ, is_valid_number n ∧
  (∀ k : ℕ, k < 1 → ¬(is_prime (n + k))) ∧
  is_prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_min_addition_to_prime_l1995_199581


namespace NUMINAMATH_CALUDE_jacket_purchase_price_l1995_199515

/-- The purchase price of a jacket given selling price and profit conditions -/
theorem jacket_purchase_price (S P : ℝ) (h1 : S = P + 0.25 * S) 
  (h2 : ∃ D : ℝ, D = 0.8 * S ∧ D - P = 4) : P = 60 := by
  sorry

end NUMINAMATH_CALUDE_jacket_purchase_price_l1995_199515


namespace NUMINAMATH_CALUDE_parallelogram_area_l1995_199539

/-- The area of a parallelogram, given the area of a triangle formed by its diagonal -/
theorem parallelogram_area (triangle_area : ℝ) (h : triangle_area = 64) : 
  2 * triangle_area = 128 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1995_199539


namespace NUMINAMATH_CALUDE_chair_production_theorem_l1995_199533

/-- Represents the chair production scenario in a furniture factory -/
structure ChairProduction where
  workers : ℕ
  individual_rate : ℕ
  total_time : ℕ
  total_chairs : ℕ

/-- Calculates the frequency of producing an additional chair as a group -/
def group_chair_frequency (cp : ChairProduction) : ℚ :=
  cp.total_time / (cp.total_chairs - cp.workers * cp.individual_rate * cp.total_time)

/-- Theorem stating the group chair frequency for the given scenario -/
theorem chair_production_theorem (cp : ChairProduction) 
  (h1 : cp.workers = 3)
  (h2 : cp.individual_rate = 4)
  (h3 : cp.total_time = 6)
  (h4 : cp.total_chairs = 73) :
  group_chair_frequency cp = 6 := by
  sorry

#eval group_chair_frequency ⟨3, 4, 6, 73⟩

end NUMINAMATH_CALUDE_chair_production_theorem_l1995_199533


namespace NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l1995_199534

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a 30-day month -/
def Month := Fin 30

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in the month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem stating that there are no starting days that result in equal Tuesdays and Fridays -/
theorem no_equal_tuesdays_fridays :
  ∀ startDay : DayOfWeek,
    countDayOccurrences startDay DayOfWeek.Tuesday ≠
    countDayOccurrences startDay DayOfWeek.Friday :=
  sorry

end NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l1995_199534


namespace NUMINAMATH_CALUDE_sqrt_640000_equals_800_l1995_199570

theorem sqrt_640000_equals_800 : Real.sqrt 640000 = 800 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_640000_equals_800_l1995_199570


namespace NUMINAMATH_CALUDE_weeks_to_save_is_36_l1995_199554

/-- The number of weeks Nina needs to save to buy all items -/
def weeks_to_save : ℕ :=
let video_game_cost : ℚ := 50
let headset_cost : ℚ := 70
let gift_cost : ℚ := 30
let sales_tax_rate : ℚ := 12 / 100
let weekly_allowance : ℚ := 10
let initial_savings_rate : ℚ := 33 / 100
let later_savings_rate : ℚ := 50 / 100
let initial_savings_weeks : ℕ := 6

let total_cost_before_tax : ℚ := video_game_cost + headset_cost + gift_cost
let total_cost_with_tax : ℚ := total_cost_before_tax * (1 + sales_tax_rate)
let gift_cost_with_tax : ℚ := gift_cost * (1 + sales_tax_rate)

let initial_savings : ℚ := weekly_allowance * initial_savings_rate * initial_savings_weeks
let remaining_gift_cost : ℚ := gift_cost_with_tax - initial_savings
let weeks_for_gift : ℕ := (remaining_gift_cost / (weekly_allowance * later_savings_rate)).ceil.toNat

let remaining_cost : ℚ := total_cost_with_tax - gift_cost_with_tax
let weeks_for_remaining : ℕ := (remaining_cost / (weekly_allowance * later_savings_rate)).ceil.toNat

initial_savings_weeks + weeks_for_gift + weeks_for_remaining

theorem weeks_to_save_is_36 : weeks_to_save = 36 := by sorry

end NUMINAMATH_CALUDE_weeks_to_save_is_36_l1995_199554


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1995_199510

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a5 : a 5 = 15) : 
  a 2 + a 4 + a 6 + a 8 = 60 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1995_199510


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1995_199543

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_third_term : a 3 = 12 / 5)
  (h_seventh_term : a 7 = 48) :
  a 5 = 12 / 5 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1995_199543


namespace NUMINAMATH_CALUDE_flag_paint_cost_l1995_199549

-- Define constants
def flag_width : Real := 3.5
def flag_height : Real := 2.5
def paint_cost_per_quart : Real := 4
def paint_coverage_per_quart : Real := 4
def sq_ft_per_sq_m : Real := 10.7639

-- Define the theorem
theorem flag_paint_cost : 
  let flag_area := flag_width * flag_height
  let total_area := 2 * flag_area
  let total_area_sq_ft := total_area * sq_ft_per_sq_m
  let quarts_needed := ⌈total_area_sq_ft / paint_coverage_per_quart⌉
  let total_cost := quarts_needed * paint_cost_per_quart
  total_cost = 192 := by
  sorry


end NUMINAMATH_CALUDE_flag_paint_cost_l1995_199549


namespace NUMINAMATH_CALUDE_first_to_receive_target_candies_l1995_199505

/-- Represents the candy distribution pattern for 8 children in a circle -/
def candy_distribution : List Nat := [1, 3, 6, 8, 3, 5, 8, 2, 5, 7, 2, 4, 7, 1, 4, 6]

/-- The number of children in the circle -/
def num_children : Nat := 8

/-- The number of candies required to win -/
def target_candies : Nat := 10

/-- The length of one complete distribution cycle -/
def cycle_length : Nat := candy_distribution.length

theorem first_to_receive_target_candies :
  ∃ (n : Nat), n ≤ num_children ∧
    (∀ m : Nat, m ≤ num_children → 
      (candy_distribution.filter (· = n)).length * (target_candies / 2) ≤ cycle_length →
      (candy_distribution.filter (· = m)).length * (target_candies / 2) ≤ 
      (candy_distribution.filter (· = n)).length * (target_candies / 2)) ∧
    n = 3 := by sorry

end NUMINAMATH_CALUDE_first_to_receive_target_candies_l1995_199505


namespace NUMINAMATH_CALUDE_fruit_salad_price_l1995_199509

/-- Represents the cost of the picnic basket items -/
structure PicnicBasket where
  numPeople : Nat
  sandwichPrice : Nat
  sodaPrice : Nat
  snackPrice : Nat
  numSnacks : Nat
  totalCost : Nat

/-- Calculates the cost of fruit salads given the picnic basket information -/
def fruitSaladCost (basket : PicnicBasket) : Nat :=
  basket.totalCost - 
  (basket.numPeople * basket.sandwichPrice + 
   2 * basket.numPeople * basket.sodaPrice + 
   basket.numSnacks * basket.snackPrice)

/-- Theorem stating that the cost of each fruit salad is $3 -/
theorem fruit_salad_price (basket : PicnicBasket) 
  (h1 : basket.numPeople = 4)
  (h2 : basket.sandwichPrice = 5)
  (h3 : basket.sodaPrice = 2)
  (h4 : basket.snackPrice = 4)
  (h5 : basket.numSnacks = 3)
  (h6 : basket.totalCost = 60) :
  fruitSaladCost basket / basket.numPeople = 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_price_l1995_199509


namespace NUMINAMATH_CALUDE_larger_number_proof_l1995_199526

theorem larger_number_proof (x y : ℤ) : 
  y = 2 * x + 3 → 
  x + y = 27 → 
  max x y = 19 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1995_199526


namespace NUMINAMATH_CALUDE_units_digit_of_product_l1995_199590

theorem units_digit_of_product (n : ℕ) : (3^1001 * 7^1002 * 13^1003) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l1995_199590


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2021_l1995_199508

theorem reciprocal_of_negative_2021 :
  let reciprocal (x : ℚ) := 1 / x
  reciprocal (-2021) = -1 / 2021 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2021_l1995_199508


namespace NUMINAMATH_CALUDE_unique_divisible_by_six_l1995_199580

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem unique_divisible_by_six : 
  ∀ B : ℕ, B < 10 → 
    (is_divisible_by (7520 + B) 6 ↔ B = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_six_l1995_199580


namespace NUMINAMATH_CALUDE_initial_orchids_l1995_199514

theorem initial_orchids (initial_roses : ℕ) (final_roses : ℕ) (final_orchids : ℕ) 
  (orchid_rose_difference : ℕ) : 
  initial_roses = 7 → 
  final_roses = 11 → 
  final_orchids = 20 → 
  orchid_rose_difference = 9 → 
  final_orchids = final_roses + orchid_rose_difference →
  initial_roses + orchid_rose_difference = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_orchids_l1995_199514


namespace NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_l1995_199563

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- The diagonals of a rhombus. -/
def Rhombus.diagonals (r : Rhombus) : Fin 2 → ℝ × ℝ := sorry

/-- Two lines are perpendicular if their dot product is zero. -/
def perpendicular (l1 l2 : ℝ × ℝ) : Prop :=
  l1.1 * l2.1 + l1.2 * l2.2 = 0

/-- The diagonals of a rhombus are always perpendicular to each other. -/
theorem rhombus_diagonals_perpendicular (r : Rhombus) :
  perpendicular (r.diagonals 0) (r.diagonals 1) := by sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_l1995_199563


namespace NUMINAMATH_CALUDE_team_selection_with_girls_l1995_199584

theorem team_selection_with_girls (boys girls team_size min_girls : ℕ) 
  (h_boys : boys = 10)
  (h_girls : girls = 12)
  (h_team_size : team_size = 6)
  (h_min_girls : min_girls = 2) : 
  (Finset.range (team_size - min_girls + 1)).sum (λ i => 
    Nat.choose girls (i + min_girls) * Nat.choose boys (team_size - (i + min_girls))) = 71379 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_with_girls_l1995_199584


namespace NUMINAMATH_CALUDE_total_rainfall_calculation_l1995_199512

theorem total_rainfall_calculation (storm1_rate : ℝ) (storm2_rate : ℝ) 
  (total_duration : ℝ) (storm1_duration : ℝ) :
  storm1_rate = 30 →
  storm2_rate = 15 →
  total_duration = 45 →
  storm1_duration = 20 →
  storm1_rate * storm1_duration + storm2_rate * (total_duration - storm1_duration) = 975 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_calculation_l1995_199512


namespace NUMINAMATH_CALUDE_comic_book_ratio_l1995_199586

/-- Given the initial number of comic books, the number bought, and the final number,
    prove that the ratio of comic books sold to initial comic books is 1/2. -/
theorem comic_book_ratio 
  (initial : ℕ) (bought : ℕ) (final : ℕ) 
  (h1 : initial = 14) 
  (h2 : bought = 6) 
  (h3 : final = 13) : 
  (initial - (final - bought)) / initial = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_comic_book_ratio_l1995_199586


namespace NUMINAMATH_CALUDE_power_multiplication_division_equality_l1995_199511

theorem power_multiplication_division_equality : (12 : ℚ)^2 * 6^3 / 432 = 72 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_division_equality_l1995_199511


namespace NUMINAMATH_CALUDE_books_bought_at_yard_sale_l1995_199546

-- Define the initial number of books
def initial_books : ℕ := 35

-- Define the final number of books
def final_books : ℕ := 56

-- Theorem: The number of books bought at the yard sale is 21
theorem books_bought_at_yard_sale :
  final_books - initial_books = 21 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_at_yard_sale_l1995_199546


namespace NUMINAMATH_CALUDE_arrange_seven_white_five_black_l1995_199521

/-- The number of ways to arrange white and black balls with constraints -/
def arrangeBalls (white black : ℕ) : ℕ :=
  Nat.choose (white + 1) black

/-- Theorem stating the number of ways to arrange 7 white balls and 5 black balls -/
theorem arrange_seven_white_five_black :
  arrangeBalls 7 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arrange_seven_white_five_black_l1995_199521


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1995_199575

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (-2, -2)

-- Define a tangent point
def TangentPoint (x y : ℝ) : Prop :=
  Circle x y ∧ ((x + 2) * x + (y + 2) * y = 0)

-- Theorem statement
theorem tangent_line_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    TangentPoint x₁ y₁ → TangentPoint x₂ y₂ →
    (2 * x₁ + 2 * y₁ + 1 = 0) ∧ (2 * x₂ + 2 * y₂ + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1995_199575


namespace NUMINAMATH_CALUDE_seventy_five_days_after_wednesday_is_monday_l1995_199567

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def days_after (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | m + 1 => next_day (days_after start m)

theorem seventy_five_days_after_wednesday_is_monday :
  days_after DayOfWeek.Wednesday 75 = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_seventy_five_days_after_wednesday_is_monday_l1995_199567


namespace NUMINAMATH_CALUDE_soccer_team_wins_l1995_199596

theorem soccer_team_wins (total_matches : ℕ) (total_points : ℕ) (lost_matches : ℕ) 
  (h1 : total_matches = 10)
  (h2 : total_points = 17)
  (h3 : lost_matches = 3) :
  ∃ (won_matches : ℕ) (drawn_matches : ℕ),
    won_matches = 5 ∧
    drawn_matches = total_matches - won_matches - lost_matches ∧
    3 * won_matches + drawn_matches = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l1995_199596


namespace NUMINAMATH_CALUDE_sunset_colors_l1995_199583

/-- The number of colors the sky turns during a sunset -/
def sky_colors (sunset_duration : ℕ) (color_change_interval : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  (sunset_duration * minutes_per_hour) / color_change_interval

/-- Theorem: During a 2-hour sunset, with the sky changing color every 10 minutes,
    and each hour being 60 minutes long, the sky turns 12 different colors. -/
theorem sunset_colors :
  sky_colors 2 10 60 = 12 := by
  sorry

#eval sky_colors 2 10 60

end NUMINAMATH_CALUDE_sunset_colors_l1995_199583


namespace NUMINAMATH_CALUDE_factor_representation_1000000_l1995_199535

/-- The number of ways to represent 1,000,000 as the product of three factors -/
def factor_representation (n : ℕ) (distinct_order : Bool) : ℕ :=
  if distinct_order then 784 else 139

/-- Theorem stating the number of ways to represent 1,000,000 as the product of three factors -/
theorem factor_representation_1000000 :
  (factor_representation 1000000 true = 784) ∧
  (factor_representation 1000000 false = 139) := by
  sorry

end NUMINAMATH_CALUDE_factor_representation_1000000_l1995_199535


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1995_199540

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_side_ratio (t : Triangle) : 
  (t.A : Real) / (t.B : Real) = 1 / 2 ∧ 
  (t.B : Real) / (t.C : Real) = 2 / 3 → 
  t.a / t.b = 1 / Real.sqrt 3 ∧ 
  t.b / t.c = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1995_199540


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1995_199595

theorem students_not_in_biology (total_students : ℕ) (enrolled_percentage : ℚ) : 
  total_students = 880 → 
  enrolled_percentage = 30 / 100 → 
  (total_students : ℚ) * (1 - enrolled_percentage) = 616 :=
by sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1995_199595


namespace NUMINAMATH_CALUDE_unique_solution_for_quadratic_difference_l1995_199565

theorem unique_solution_for_quadratic_difference (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b : ℝ), ∀ x : ℝ, (x + m)^2 - (x + n)^2 = (m - n)^2 → x = a * m + b * n ∧ a = 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_quadratic_difference_l1995_199565


namespace NUMINAMATH_CALUDE_theater_popcorn_packages_l1995_199588

/-- The number of popcorn buckets needed by the theater -/
def total_buckets : ℕ := 426

/-- The number of buckets in each package -/
def buckets_per_package : ℕ := 8

/-- The minimum number of packages required -/
def min_packages : ℕ := 54

theorem theater_popcorn_packages :
  min_packages = (total_buckets + buckets_per_package - 1) / buckets_per_package :=
by sorry

end NUMINAMATH_CALUDE_theater_popcorn_packages_l1995_199588


namespace NUMINAMATH_CALUDE_selling_price_articles_l1995_199571

/-- Proves that if the cost price of 50 articles equals the selling price of N articles,
    and the gain percent is 25%, then N = 40. -/
theorem selling_price_articles (C : ℝ) (N : ℕ) (h1 : N * (C + 0.25 * C) = 50 * C) : N = 40 := by
  sorry

#check selling_price_articles

end NUMINAMATH_CALUDE_selling_price_articles_l1995_199571


namespace NUMINAMATH_CALUDE_simplify_fraction_and_multiply_l1995_199552

theorem simplify_fraction_and_multiply :
  (144 : ℚ) / 1296 * 36 = 4 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_and_multiply_l1995_199552


namespace NUMINAMATH_CALUDE_w_magnitude_bounds_l1995_199502

theorem w_magnitude_bounds (z : ℂ) (h : Complex.abs z = 1) : 
  let w : ℂ := z^4 - z^3 - 3 * z^2 * Complex.I - z + 1
  3 ≤ Complex.abs w ∧ Complex.abs w ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_w_magnitude_bounds_l1995_199502


namespace NUMINAMATH_CALUDE_m_is_positive_l1995_199529

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

-- State the theorem
theorem m_is_positive (m : ℝ) (h : (M m) ∩ N ≠ ∅) : m > 0 := by
  sorry

end NUMINAMATH_CALUDE_m_is_positive_l1995_199529


namespace NUMINAMATH_CALUDE_solution_value_l1995_199551

theorem solution_value (x m : ℝ) : x = 3 ∧ (11 - 2*x = m*x - 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1995_199551


namespace NUMINAMATH_CALUDE_student_calculation_l1995_199566

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 155 → 
  chosen_number * 2 - 200 = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l1995_199566


namespace NUMINAMATH_CALUDE_max_intersections_circle_square_l1995_199541

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A square in a plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- The number of intersection points between a circle and a square -/
def intersection_points (c : Circle) (s : Square) : ℕ :=
  sorry

/-- The maximum number of intersection points between any circle and any square -/
def max_intersection_points : ℕ := sorry

/-- Theorem: The maximum number of intersection points between a circle and a square is 8 -/
theorem max_intersections_circle_square : max_intersection_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_square_l1995_199541


namespace NUMINAMATH_CALUDE_k_gonal_number_formula_l1995_199519

/-- The n-th k-gonal number -/
def N (n k : ℕ) : ℚ :=
  ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n

/-- Theorem: The formula for the n-th k-gonal number -/
theorem k_gonal_number_formula (n k : ℕ) (h1 : n ≥ 1) (h2 : k ≥ 3) :
  N n k = ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n :=
by sorry

end NUMINAMATH_CALUDE_k_gonal_number_formula_l1995_199519


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l1995_199544

theorem multiplication_addition_equality : 12 * 24 + 36 * 12 = 720 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l1995_199544


namespace NUMINAMATH_CALUDE_price_adjustment_theorem_l1995_199518

theorem price_adjustment_theorem (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let first_increase := 1.20
  let second_increase := 1.10
  let third_increase := 1.15
  let discount := 0.95
  let tax := 1.07
  let final_price := original_price * first_increase * second_increase * third_increase * discount * tax
  let required_decrease := 0.351852
  final_price * (1 - required_decrease) = original_price := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_theorem_l1995_199518


namespace NUMINAMATH_CALUDE_constant_ratio_solution_l1995_199573

/-- The constant ratio of (3x - 4) to (y + 15) -/
def k (x y : ℚ) : ℚ := (3 * x - 4) / (y + 15)

theorem constant_ratio_solution (x₀ y₀ x₁ y₁ : ℚ) 
  (h₀ : y₀ = 4)
  (h₁ : x₀ = 5)
  (h₂ : y₁ = 15)
  (h₃ : k x₀ y₀ = k x₁ y₁) :
  x₁ = 406 / 57 := by
  sorry

end NUMINAMATH_CALUDE_constant_ratio_solution_l1995_199573


namespace NUMINAMATH_CALUDE_weekly_fat_intake_l1995_199555

def morning_rice : ℕ := 3
def afternoon_rice : ℕ := 2
def evening_rice : ℕ := 5
def fat_per_cup : ℕ := 10
def days_in_week : ℕ := 7

theorem weekly_fat_intake : 
  (morning_rice + afternoon_rice + evening_rice) * fat_per_cup * days_in_week = 700 := by
  sorry

end NUMINAMATH_CALUDE_weekly_fat_intake_l1995_199555


namespace NUMINAMATH_CALUDE_campsite_return_strategy_l1995_199568

structure CampsiteScenario where
  num_students : ℕ
  time_remaining : ℕ
  num_roads : ℕ
  time_per_road : ℕ
  num_liars : ℕ

def has_reliable_strategy (scenario : CampsiteScenario) : Prop :=
  ∃ (strategy : CampsiteScenario → Bool),
    strategy scenario = true

theorem campsite_return_strategy 
  (scenario1 : CampsiteScenario)
  (scenario2 : CampsiteScenario)
  (h1 : scenario1.num_students = 8)
  (h2 : scenario1.time_remaining = 60)
  (h3 : scenario2.num_students = 4)
  (h4 : scenario2.time_remaining = 100)
  (h5 : scenario1.num_roads = 4)
  (h6 : scenario2.num_roads = 4)
  (h7 : scenario1.time_per_road = 20)
  (h8 : scenario2.time_per_road = 20)
  (h9 : scenario1.num_liars = 2)
  (h10 : scenario2.num_liars = 2) :
  has_reliable_strategy scenario1 ∧ has_reliable_strategy scenario2 :=
sorry

end NUMINAMATH_CALUDE_campsite_return_strategy_l1995_199568


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1995_199536

theorem quadratic_expression_value :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 + z^2 + 2*x*z = 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1995_199536


namespace NUMINAMATH_CALUDE_inequality_solution_l1995_199527

theorem inequality_solution (x : ℝ) : 
  (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 2 ↔ 
  (x > -5 ∧ x < -20/3) ∨ (x > 2/3 ∧ x < 4/3) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1995_199527


namespace NUMINAMATH_CALUDE_candy_distribution_contradiction_l1995_199560

theorem candy_distribution_contradiction (N : ℕ) : 
  (∃ (x : ℕ), N = 2 * x) →
  (∃ (y : ℕ), N = 3 * y) →
  (∃ (z : ℕ), N / 3 = 2 * z + 3) →
  False :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_contradiction_l1995_199560


namespace NUMINAMATH_CALUDE_farmer_livestock_purchase_l1995_199597

theorem farmer_livestock_purchase :
  ∀ (num_cows num_sheep num_rabbits : ℕ)
    (cost_cow cost_sheep cost_rabbit : ℚ),
  num_cows + num_sheep + num_rabbits = 100 →
  cost_cow = 50 →
  cost_sheep = 10 →
  cost_rabbit = 1/2 →
  cost_cow * num_cows + cost_sheep * num_sheep + cost_rabbit * num_rabbits = 1000 →
  num_cows = 19 ∧ num_sheep = 1 ∧ num_rabbits = 80 ∧
  cost_cow * num_cows = 950 ∧ cost_sheep * num_sheep = 10 ∧ cost_rabbit * num_rabbits = 40 :=
by sorry

end NUMINAMATH_CALUDE_farmer_livestock_purchase_l1995_199597


namespace NUMINAMATH_CALUDE_second_player_wins_123_l1995_199585

/-- A game where players color points on a circle. -/
structure ColorGame where
  num_points : ℕ
  first_player : Bool
  
/-- The result of the game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- Determine the winner of the color game. -/
def winner (game : ColorGame) : GameResult :=
  if game.num_points % 2 = 1 then GameResult.SecondPlayerWins
  else GameResult.FirstPlayerWins

/-- The main theorem stating that the second player wins in a game with 123 points. -/
theorem second_player_wins_123 :
  ∀ (game : ColorGame), game.num_points = 123 → winner game = GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_123_l1995_199585


namespace NUMINAMATH_CALUDE_factorial16_trailingZeroes_base8_l1995_199503

/-- The number of trailing zeroes in the base 8 representation of 16! -/
def trailingZeroesBase8Factorial16 : ℕ := 5

/-- Theorem stating that the number of trailing zeroes in the base 8 representation of 16! is 5 -/
theorem factorial16_trailingZeroes_base8 :
  trailingZeroesBase8Factorial16 = 5 := by sorry

end NUMINAMATH_CALUDE_factorial16_trailingZeroes_base8_l1995_199503


namespace NUMINAMATH_CALUDE_last_two_digits_of_nine_to_2008_l1995_199577

theorem last_two_digits_of_nine_to_2008 : 9^2008 % 100 = 21 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_nine_to_2008_l1995_199577


namespace NUMINAMATH_CALUDE_alcohol_percentage_problem_l1995_199578

/-- Proves that the initial alcohol percentage is 30% given the conditions of the problem -/
theorem alcohol_percentage_problem (initial_volume : ℝ) (added_alcohol : ℝ) (final_percentage : ℝ) :
  initial_volume = 6 →
  added_alcohol = 2.4 →
  final_percentage = 50 →
  (∃ initial_percentage : ℝ,
    initial_percentage * initial_volume / 100 + added_alcohol = 
    final_percentage * (initial_volume + added_alcohol) / 100 ∧
    initial_percentage = 30) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_problem_l1995_199578


namespace NUMINAMATH_CALUDE_consecutive_sum_at_least_17_l1995_199592

theorem consecutive_sum_at_least_17 (a : Fin 10 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 11 \ {0}) : 
  ∃ i : Fin 10, a i + a (i + 1) + a (i + 2) ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_at_least_17_l1995_199592


namespace NUMINAMATH_CALUDE_max_band_members_l1995_199582

theorem max_band_members : ∃ (m : ℕ), m = 234 ∧
  (∃ (k : ℕ), m = k^2 + 9) ∧
  (∃ (n : ℕ), m = n * (n + 5)) ∧
  (∀ (m' : ℕ), m' > m →
    (∃ (k : ℕ), m' = k^2 + 9) →
    (∃ (n : ℕ), m' = n * (n + 5)) →
    False) :=
by sorry

end NUMINAMATH_CALUDE_max_band_members_l1995_199582


namespace NUMINAMATH_CALUDE_xyz_value_l1995_199522

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 20 * Real.rpow 2 (1/3))
  (hxz : x * z = 35 * Real.rpow 2 (1/3))
  (hyz : y * z = 14 * Real.rpow 2 (1/3)) :
  x * y * z = 140 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1995_199522


namespace NUMINAMATH_CALUDE_f_properties_l1995_199569

noncomputable section

open Real

/-- The function f(x) = ae^(2x) - ae^x - xe^x --/
def f (a : ℝ) (x : ℝ) : ℝ := a * exp (2 * x) - a * exp x - x * exp x

/-- The theorem stating the properties of f --/
theorem f_properties :
  ∀ a : ℝ, a ≥ 0 → (∀ x : ℝ, f a x ≥ 0) →
  ∃ x₀ : ℝ,
    a = 1 ∧
    (∀ x : ℝ, f 1 x ≤ f 1 x₀) ∧
    (∀ x : ℝ, x ≠ x₀ → f 1 x < f 1 x₀) ∧
    (log 2 / (2 * exp 1) + 1 / (4 * exp 1 ^ 2) ≤ f 1 x₀) ∧
    (f 1 x₀ < 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1995_199569


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l1995_199537

/-- The ellipse C with semi-major axis a and semi-minor axis b -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The line tangent to the circle -/
def tangent_line (x y : ℝ) : Prop :=
  Real.sqrt 7 * x - Real.sqrt 5 * y + 12 = 0

/-- The point A -/
def A : ℝ × ℝ := (-4, 0)

/-- The point R -/
def R : ℝ × ℝ := (3, 0)

/-- The vertical line that M and N lie on -/
def vertical_line (x : ℝ) : Prop :=
  x = 16/3

theorem ellipse_slope_product (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 1/4) 
  (h4 : ∃ (x y : ℝ), ellipse b b x y ∧ tangent_line x y) :
  ∃ (P Q M N : ℝ × ℝ) (k1 k2 : ℝ),
    ellipse a b P.1 P.2 ∧
    ellipse a b Q.1 Q.2 ∧
    vertical_line M.1 ∧
    vertical_line N.1 ∧
    k1 * k2 = -12/7 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_l1995_199537


namespace NUMINAMATH_CALUDE_factorization_of_8a_squared_minus_2_l1995_199545

theorem factorization_of_8a_squared_minus_2 (a : ℝ) : 8 * a^2 - 2 = 2 * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_8a_squared_minus_2_l1995_199545


namespace NUMINAMATH_CALUDE_marble_problem_l1995_199591

def initial_red_marbles : ℕ := 33
def initial_green_marbles : ℕ := 22

theorem marble_problem :
  (initial_red_marbles : ℚ) / initial_green_marbles = 3 / 2 ∧
  (initial_red_marbles - 18 : ℚ) / (initial_green_marbles + 15) = 2 / 5 :=
by
  sorry

#check marble_problem

end NUMINAMATH_CALUDE_marble_problem_l1995_199591


namespace NUMINAMATH_CALUDE_square_root_two_minus_one_squared_plus_two_times_plus_three_l1995_199559

theorem square_root_two_minus_one_squared_plus_two_times_plus_three (x : ℝ) :
  x = Real.sqrt 2 - 1 → x^2 + 2*x + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_two_minus_one_squared_plus_two_times_plus_three_l1995_199559


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1995_199599

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 11 = 12 / 77 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1995_199599


namespace NUMINAMATH_CALUDE_broken_stick_pairing_probability_l1995_199587

/-- The number of sticks --/
def n : ℕ := 5

/-- The probability of pairing each long part with a short part when rearranging broken sticks --/
theorem broken_stick_pairing_probability :
  (2^n : ℚ) / (Nat.choose (2*n) n : ℚ) = 8/63 := by sorry

end NUMINAMATH_CALUDE_broken_stick_pairing_probability_l1995_199587


namespace NUMINAMATH_CALUDE_valid_placements_count_l1995_199564

/-- Represents a ball -/
inductive Ball : Type
| A : Ball
| B : Ball
| C : Ball
| D : Ball

/-- Represents a box -/
inductive Box : Type
| one : Box
| two : Box
| three : Box

/-- A placement of balls into boxes -/
def Placement := Ball → Box

/-- Checks if a placement is valid -/
def isValidPlacement (p : Placement) : Prop :=
  (∀ b : Box, ∃ ball : Ball, p ball = b) ∧ 
  (p Ball.A ≠ p Ball.B)

/-- The number of valid placements -/
def numValidPlacements : ℕ := sorry

theorem valid_placements_count : numValidPlacements = 30 := by sorry

end NUMINAMATH_CALUDE_valid_placements_count_l1995_199564


namespace NUMINAMATH_CALUDE_height_of_C_l1995_199561

/-- Given three people A, B, and C with heights hA, hB, and hC respectively (in cm),
    prove that C's height is 143 cm under the following conditions:
    1. The average height of A, B, and C is 143 cm.
    2. A's height increased by 4.5 cm becomes the average height of B and C.
    3. B is 3 cm taller than C. -/
theorem height_of_C (hA hB hC : ℝ) : 
  (hA + hB + hC) / 3 = 143 →
  hA + 4.5 = (hB + hC) / 2 →
  hB = hC + 3 →
  hC = 143 := by sorry

end NUMINAMATH_CALUDE_height_of_C_l1995_199561


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1995_199501

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 5) ↔ x ≥ 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1995_199501


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1995_199550

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 + a*x + b < 0}) :
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1995_199550


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1995_199556

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 7 * π / 4
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π) ∧
  r = 2 * Real.sqrt 2 ∧
  θ = 7 * π / 4 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1995_199556


namespace NUMINAMATH_CALUDE_sector_central_angle_l1995_199598

theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 3 * π / 8) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1995_199598


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l1995_199579

theorem lcm_gcf_ratio_240_360 : 
  (Nat.lcm 240 360) / (Nat.gcd 240 360) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l1995_199579


namespace NUMINAMATH_CALUDE_negation_of_existence_l1995_199593

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1995_199593


namespace NUMINAMATH_CALUDE_henry_pill_cost_l1995_199538

/-- Calculates the total cost of pills for Henry over 21 days -/
def totalPillCost (daysTotal : ℕ) (pillsPerDay : ℕ) (pillType1Count : ℕ) (pillType2Count : ℕ)
  (pillType1Cost : ℚ) (pillType2Cost : ℚ) (pillType3ExtraCost : ℚ) 
  (discountRate : ℚ) (priceIncrease : ℚ) : ℚ :=
  let pillType3Count := pillsPerDay - (pillType1Count + pillType2Count)
  let pillType3Cost := pillType2Cost + pillType3ExtraCost
  let regularDayCost := pillType1Count * pillType1Cost + pillType2Count * pillType2Cost + 
                        pillType3Count * pillType3Cost
  let discountDays := daysTotal / 3
  let regularDays := daysTotal - discountDays
  let discountDayCost := (1 - discountRate) * (pillType1Count * pillType1Cost + pillType2Count * pillType2Cost) +
                         pillType3Count * (pillType3Cost + priceIncrease)
  regularDays * regularDayCost + discountDays * discountDayCost

/-- The total cost of Henry's pills over 21 days is $1485.10 -/
theorem henry_pill_cost : 
  totalPillCost 21 12 4 5 (3/2) 7 3 (1/5) (5/2) = 1485.1 := by
  sorry

end NUMINAMATH_CALUDE_henry_pill_cost_l1995_199538


namespace NUMINAMATH_CALUDE_computer_sales_ratio_l1995_199553

theorem computer_sales_ratio (total : ℕ) (laptops : ℕ) (desktops : ℕ) (netbooks : ℕ) :
  total = 72 →
  laptops = total / 2 →
  desktops = 12 →
  netbooks = total - laptops - desktops →
  (netbooks : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_computer_sales_ratio_l1995_199553


namespace NUMINAMATH_CALUDE_flower_bed_side_length_l1995_199557

/-- Given a rectangular flower bed with area 6a^2 - 4ab + 2a and one side of length 2a,
    the length of the other side is 3a - 2b + 1 -/
theorem flower_bed_side_length (a b : ℝ) :
  let area := 6 * a^2 - 4 * a * b + 2 * a
  let side1 := 2 * a
  area / side1 = 3 * a - 2 * b + 1 := by
sorry

end NUMINAMATH_CALUDE_flower_bed_side_length_l1995_199557


namespace NUMINAMATH_CALUDE_min_value_expression_l1995_199500

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 4) * (b^2 + 4*b + 4) * (c^2 + 4*c + 4) / (a * b * c) ≥ 729 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1995_199500


namespace NUMINAMATH_CALUDE_average_age_decrease_l1995_199548

theorem average_age_decrease (initial_size : ℕ) (replaced_age new_age : ℕ) : 
  initial_size = 10 → replaced_age = 42 → new_age = 12 → 
  (replaced_age - new_age) / initial_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1995_199548


namespace NUMINAMATH_CALUDE_binary_subtraction_theorem_l1995_199520

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 111001₂ -/
def binary_111001 : List Bool := [true, false, false, true, true, true]

theorem binary_subtraction_theorem :
  binary_to_decimal binary_111001 - 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_theorem_l1995_199520


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1995_199547

theorem max_value_on_circle :
  let circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 + 4*p.1 - 6*p.2 + 4) = 0}
  ∃ (max : ℝ), max = -13 ∧ 
    (∀ p ∈ circle, 3*p.1 - 4*p.2 ≤ max) ∧
    (∃ p ∈ circle, 3*p.1 - 4*p.2 = max) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1995_199547


namespace NUMINAMATH_CALUDE_divisibility_criteria_l1995_199562

theorem divisibility_criteria (n : ℕ) : 
  let q : ℕ := n / 10
  let r : ℕ := n % 10
  (7 ∣ n ↔ 7 ∣ (q - 2*r)) ∧ 
  (7 ∣ 2023) ∧
  (13 ∣ n ↔ 13 ∣ (q + 4*r)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_criteria_l1995_199562


namespace NUMINAMATH_CALUDE_loan_interest_calculation_l1995_199532

/-- Calculates simple interest given principal, rate, and time --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: The simple interest on a loan of $1200 at 3% for 3 years is $108 --/
theorem loan_interest_calculation :
  let principal : ℝ := 1200
  let rate : ℝ := 0.03
  let time : ℝ := 3
  simple_interest principal rate time = 108 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_calculation_l1995_199532


namespace NUMINAMATH_CALUDE_largest_square_area_l1995_199530

/-- Represents a right triangle with squares on each side -/
structure RightTriangleWithSquares where
  -- Side lengths
  xz : ℝ
  yz : ℝ
  xy : ℝ
  -- Right angle condition
  right_angle : xy^2 = xz^2 + yz^2
  -- Non-negativity of side lengths
  xz_nonneg : xz ≥ 0
  yz_nonneg : yz ≥ 0
  xy_nonneg : xy ≥ 0

/-- The theorem to be proved -/
theorem largest_square_area
  (t : RightTriangleWithSquares)
  (sum_area : t.xz^2 + t.yz^2 + t.xy^2 = 450) :
  t.xy^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l1995_199530


namespace NUMINAMATH_CALUDE_caiden_roofing_cost_l1995_199574

-- Define the parameters
def total_feet : ℕ := 300
def cost_per_foot : ℚ := 8
def discount_rate : ℚ := 0.1
def shipping_fee : ℚ := 150
def sales_tax_rate : ℚ := 0.05
def free_feet : ℕ := 250

-- Define the calculation steps
def paid_feet : ℕ := total_feet - free_feet
def base_cost : ℚ := paid_feet * cost_per_foot
def discounted_cost : ℚ := base_cost * (1 - discount_rate)
def cost_with_shipping : ℚ := discounted_cost + shipping_fee
def total_cost : ℚ := cost_with_shipping * (1 + sales_tax_rate)

-- Theorem to prove
theorem caiden_roofing_cost :
  total_cost = 535.5 := by sorry

end NUMINAMATH_CALUDE_caiden_roofing_cost_l1995_199574


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1995_199523

/-- Given a group of 8 persons, if replacing one person with a new person weighing 94 kg
    increases the average weight by 3 kg, then the weight of the replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (original_count : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : original_count = 8)
  (h2 : weight_increase = 3)
  (h3 : new_person_weight = 94)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1995_199523


namespace NUMINAMATH_CALUDE_circle_radius_nine_iff_k_94_l1995_199528

/-- The equation of a circle in general form --/
def circle_equation (x y k : ℝ) : Prop :=
  2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 0

/-- The equation of a circle in standard form with center (h, k) and radius r --/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the given equation represents a circle with radius 9 iff k = 94 --/
theorem circle_radius_nine_iff_k_94 :
  (∃ h k : ℝ, ∀ x y : ℝ, circle_equation x y 94 ↔ standard_circle_equation x y h k 9) ↔
  (∀ k : ℝ, (∃ h k : ℝ, ∀ x y : ℝ, circle_equation x y k ↔ standard_circle_equation x y h k 9) → k = 94) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_nine_iff_k_94_l1995_199528


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1995_199558

/-- Given a rectangle with dimensions (x - 3) by (2x + 3) and area 4x - 9, prove that x = 7/2 -/
theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (2 * x + 3) = 4 * x - 9 → x = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1995_199558


namespace NUMINAMATH_CALUDE_equation_result_l1995_199542

theorem equation_result : (88320 : ℤ) + 1315 + 9211 - 1569 = 97277 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l1995_199542


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l1995_199576

/-- The number of ways to allocate volunteers to projects -/
def allocate_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * (k.factorial)

/-- Theorem stating that allocating 5 volunteers to 4 projects results in 240 schemes -/
theorem volunteer_allocation_schemes :
  allocate_volunteers 5 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l1995_199576


namespace NUMINAMATH_CALUDE_right_triangle_existence_l1995_199506

theorem right_triangle_existence (a : ℤ) (h : a ≥ 5) :
  ∃ b c : ℤ, c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l1995_199506


namespace NUMINAMATH_CALUDE_cost_price_is_1200_l1995_199513

/-- Calculates the cost price of a toy given the total selling price, number of toys sold, and the gain condition. -/
def cost_price_of_toy (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) : ℕ :=
  let selling_price_per_toy := total_selling_price / num_toys_sold
  let cost_price := selling_price_per_toy * num_toys_sold / (num_toys_sold + num_toys_gain)
  cost_price

/-- Theorem stating that under the given conditions, the cost price of a toy is 1200. -/
theorem cost_price_is_1200 :
  cost_price_of_toy 50400 36 6 = 1200 := by
  sorry

#eval cost_price_of_toy 50400 36 6

end NUMINAMATH_CALUDE_cost_price_is_1200_l1995_199513


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1995_199516

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Median AM in triangle ABC --/
def median_AM (t : Triangle) : ℝ := sorry

theorem triangle_ABC_properties (t : Triangle) 
  (h1 : t.a^2 - (t.b - t.c)^2 = (2 - Real.sqrt 3) * t.b * t.c)
  (h2 : Real.sin t.A * Real.sin t.B = (Real.cos (t.C / 2))^2)
  (h3 : median_AM t = Real.sqrt 7) :
  t.A = π / 6 ∧ t.B = π / 6 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ABC_properties_l1995_199516


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l1995_199517

/-- The center of a circle tangent to two parallel lines and lying on a third line -/
theorem circle_center_coordinates (x y : ℚ) : 
  (∃ (r : ℚ), r > 0 ∧ 
    (∀ (x' y' : ℚ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (3*x' + 4*y' = 24 ∨ 3*x' + 4*y' = -16))) → 
  x - 3*y = 0 → 
  (x, y) = (12/13, 4/13) := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l1995_199517
