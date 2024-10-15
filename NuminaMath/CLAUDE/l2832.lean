import Mathlib

namespace NUMINAMATH_CALUDE_one_carbon_per_sheet_l2832_283253

/-- Represents the number of carbon copies produced when sheets are folded and typed on -/
def carbon_copies_when_folded : ℕ := 2

/-- Represents the total number of sheets -/
def total_sheets : ℕ := 3

/-- Represents the number of carbons in each sheet -/
def carbons_per_sheet : ℕ := 1

/-- Theorem stating that there is 1 carbon in each sheet -/
theorem one_carbon_per_sheet :
  (carbons_per_sheet = 1) ∧ 
  (carbon_copies_when_folded = 2) ∧
  (total_sheets = 3) := by
  sorry

end NUMINAMATH_CALUDE_one_carbon_per_sheet_l2832_283253


namespace NUMINAMATH_CALUDE_special_line_equation_l2832_283232

/-- A line passing through the point (3, -4) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  equation : ℝ → ℝ → ℝ → ℝ
  /-- The line passes through the point (3, -4) -/
  passes_through_point : equation 3 (-4) 0 = 0
  /-- The x-intercept and y-intercept are opposite numbers -/
  opposite_intercepts : ∃ (a : ℝ), equation a 0 0 = 0 ∧ equation 0 (-a) 0 = 0

/-- The equation of the special line is either 4x + 3y = 0 or x - y - 7 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, l.equation x y 0 = 4*x + 3*y) ∨
  (∀ x y, l.equation x y 0 = x - y - 7) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l2832_283232


namespace NUMINAMATH_CALUDE_equal_numbers_theorem_l2832_283295

theorem equal_numbers_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) :
  a = b ∨ a = c ∨ b = c :=
sorry

end NUMINAMATH_CALUDE_equal_numbers_theorem_l2832_283295


namespace NUMINAMATH_CALUDE_nickel_chocolates_l2832_283239

theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 10) 
  (h2 : robert = nickel + 5) : 
  nickel = 5 := by
  sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l2832_283239


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2832_283209

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 16) : 
  min x y = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2832_283209


namespace NUMINAMATH_CALUDE_initial_money_theorem_l2832_283273

def meat_cost : ℕ := 17
def chicken_cost : ℕ := 22
def veggies_cost : ℕ := 43
def eggs_cost : ℕ := 5
def dog_food_cost : ℕ := 45
def cat_food_cost : ℕ := 18
def money_left : ℕ := 35

def total_spent : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + cat_food_cost

theorem initial_money_theorem : 
  meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + cat_food_cost + money_left = 185 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_theorem_l2832_283273


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l2832_283204

/-- Represents the number of meters of cloth sold -/
def meters_sold : ℕ := 30

/-- The total selling price in Rupees -/
def total_selling_price : ℕ := 4500

/-- The profit per meter in Rupees -/
def profit_per_meter : ℕ := 10

/-- The cost price per meter in Rupees -/
def cost_price_per_meter : ℕ := 140

/-- Theorem stating that the number of meters sold is correct given the conditions -/
theorem cloth_sale_calculation :
  meters_sold * (cost_price_per_meter + profit_per_meter) = total_selling_price :=
by sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l2832_283204


namespace NUMINAMATH_CALUDE_root_in_interval_l2832_283286

-- Define the function
def f (x : ℝ) := x^3 - 2*x - 5

-- Theorem statement
theorem root_in_interval :
  (∃ x ∈ Set.Icc 2 3, f x = 0) →  -- root exists in [2,3]
  f 2.5 > 0 →                    -- f(2.5) > 0
  (∃ x ∈ Set.Ioo 2 2.5, f x = 0) -- root exists in (2,2.5)
  := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2832_283286


namespace NUMINAMATH_CALUDE_milk_water_mixture_volume_l2832_283226

theorem milk_water_mixture_volume 
  (initial_milk_percentage : Real)
  (final_milk_percentage : Real)
  (added_water : Real)
  (h1 : initial_milk_percentage = 0.84)
  (h2 : final_milk_percentage = 0.60)
  (h3 : added_water = 24)
  : ∃ initial_volume : Real,
    initial_volume * initial_milk_percentage = 
    (initial_volume + added_water) * final_milk_percentage ∧
    initial_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_mixture_volume_l2832_283226


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2832_283213

theorem opposite_of_negative_two : (-(- 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2832_283213


namespace NUMINAMATH_CALUDE_max_discount_rate_l2832_283201

/-- The maximum discount rate that can be offered on an item while maintaining a minimum profit margin -/
theorem max_discount_rate (cost_price selling_price min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1)
  (h4 : cost_price > 0)
  (h5 : selling_price > cost_price) :
  ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      0 ≤ discount → discount ≤ max_discount → 
      (selling_price * (1 - discount / 100) - cost_price) / cost_price ≥ min_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l2832_283201


namespace NUMINAMATH_CALUDE_f_composition_of_three_l2832_283256

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_three : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l2832_283256


namespace NUMINAMATH_CALUDE_fraction_equality_l2832_283206

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2832_283206


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l2832_283288

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Payment per task in dollars -/
def payment_per_task : ℚ := 6/5

/-- Days worked per week -/
def days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * payment_per_task * days_per_week

theorem tim_weekly_earnings :
  weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l2832_283288


namespace NUMINAMATH_CALUDE_three_step_to_one_eleven_step_to_one_l2832_283254

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

def reaches_one_in (n : ℕ) (steps : ℕ) : Prop :=
  ∃ (sequence : ℕ → ℕ), 
    sequence 0 = n ∧
    sequence steps = 1 ∧
    ∀ i < steps, sequence (i + 1) = operation (sequence i)

theorem three_step_to_one :
  ∃! (s : Finset ℕ), 
    s.card = 3 ∧ 
    ∀ n, n ∈ s ↔ reaches_one_in n 3 :=
sorry

theorem eleven_step_to_one :
  ∃! (s : Finset ℕ), 
    s.card = 3 ∧ 
    ∀ n, n ∈ s ↔ reaches_one_in n 11 :=
sorry

end NUMINAMATH_CALUDE_three_step_to_one_eleven_step_to_one_l2832_283254


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l2832_283297

theorem chess_game_draw_probability (P_A_not_losing P_B_not_losing : ℝ) 
  (h1 : P_A_not_losing = 0.8)
  (h2 : P_B_not_losing = 0.7)
  (h3 : ∀ P_A_win P_B_win P_draw : ℝ, 
    P_A_win + P_draw = P_A_not_losing → 
    P_B_win + P_draw = P_B_not_losing → 
    P_A_win + P_B_win + P_draw = 1 → 
    P_draw = 0.5) :
  ∃ P_draw : ℝ, P_draw = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l2832_283297


namespace NUMINAMATH_CALUDE_complementary_event_of_A_l2832_283262

-- Define the sample space for a fair cubic die
def DieOutcome := Fin 6

-- Define event A
def EventA (outcome : DieOutcome) : Prop := outcome.val % 2 = 1

-- Define the complementary event of A
def ComplementA (outcome : DieOutcome) : Prop := outcome.val % 2 = 0

-- Theorem statement
theorem complementary_event_of_A :
  ∀ (outcome : DieOutcome), ¬(EventA outcome) ↔ ComplementA outcome :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_of_A_l2832_283262


namespace NUMINAMATH_CALUDE_polynomial_derivative_symmetry_l2832_283260

/-- Given a polynomial function f(x) = ax^4 + bx^2 + c, 
    if f'(1) = 2, then f'(-1) = -2 -/
theorem polynomial_derivative_symmetry 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h_f'_1 : (deriv f) 1 = 2) : 
  (deriv f) (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_symmetry_l2832_283260


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_100_l2832_283242

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_100 :
  units_digit (factorial_sum 100) = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_100_l2832_283242


namespace NUMINAMATH_CALUDE_stock_value_decrease_l2832_283205

theorem stock_value_decrease (F : ℝ) (h1 : F > 0) : 
  let J := 0.9 * F
  let M := J / 1.2
  (F - M) / F * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_stock_value_decrease_l2832_283205


namespace NUMINAMATH_CALUDE_neg_p_true_when_k_3_k_range_when_p_or_q_false_l2832_283264

-- Define propositions p and q
def p (k : ℝ) : Prop := ∃ x : ℝ, k * x^2 + 1 ≤ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * k * x + 1 > 0

-- Theorem 1: When k = 3, ¬p is true
theorem neg_p_true_when_k_3 : ∀ x : ℝ, 3 * x^2 + 1 > 0 := by sorry

-- Theorem 2: The set of k for which both p and q are false
theorem k_range_when_p_or_q_false : 
  {k : ℝ | ¬(p k) ∧ ¬(q k)} = {k : ℝ | k ≤ -1 ∨ k ≥ 1} := by sorry

end NUMINAMATH_CALUDE_neg_p_true_when_k_3_k_range_when_p_or_q_false_l2832_283264


namespace NUMINAMATH_CALUDE_sarah_trucks_l2832_283210

theorem sarah_trucks (trucks_to_jeff trucks_to_ashley trucks_remaining : ℕ) 
  (h1 : trucks_to_jeff = 13)
  (h2 : trucks_to_ashley = 21)
  (h3 : trucks_remaining = 38) :
  trucks_to_jeff + trucks_to_ashley + trucks_remaining = 72 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_l2832_283210


namespace NUMINAMATH_CALUDE_square_angle_problem_l2832_283202

/-- In a square ABCD with a segment CE, if two angles formed are 7α and 8α, then α = 9°. -/
theorem square_angle_problem (α : ℝ) : 
  (7 * α + 8 * α + 45 = 180) → α = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_angle_problem_l2832_283202


namespace NUMINAMATH_CALUDE_coat_cost_price_l2832_283290

theorem coat_cost_price (markup_percentage : ℝ) (final_price : ℝ) (cost_price : ℝ) : 
  markup_percentage = 0.25 →
  final_price = 275 →
  final_price = cost_price * (1 + markup_percentage) →
  cost_price = 220 := by
  sorry

end NUMINAMATH_CALUDE_coat_cost_price_l2832_283290


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2832_283251

theorem circle_area_theorem (z₁ z₂ : ℂ) 
  (h₁ : z₁^2 - 4*z₁*z₂ + 4*z₂^2 = 0) 
  (h₂ : Complex.abs z₂ = 2) : 
  Real.pi * (Complex.abs z₁ / 2)^2 = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2832_283251


namespace NUMINAMATH_CALUDE_median_is_106_l2832_283287

-- Define the list
def list_size (n : ℕ) : ℕ := if n ≤ 150 then n else 0

-- Define the sum of the list sizes
def total_elements : ℕ := (Finset.range 151).sum list_size

-- Define the median position
def median_position : ℕ := (total_elements + 1) / 2

-- Theorem statement
theorem median_is_106 : 
  ∃ (cumsum : ℕ → ℕ), 
    (∀ n, cumsum n = (Finset.range (n + 1)).sum list_size) ∧
    (cumsum 105 < median_position) ∧
    (median_position ≤ cumsum 106) :=
sorry

end NUMINAMATH_CALUDE_median_is_106_l2832_283287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2832_283270

theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 = 2 →                                            -- given: a_1 = 2
  a 3 + a 5 = 8 →                                      -- given: a_3 + a_5 = 8
  a 7 = 6 :=                                           -- to prove: a_7 = 6
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2832_283270


namespace NUMINAMATH_CALUDE_blue_shirts_count_l2832_283258

theorem blue_shirts_count (total_shirts green_shirts : ℕ) 
  (h1 : total_shirts = 23)
  (h2 : green_shirts = 17)
  (h3 : total_shirts = green_shirts + blue_shirts) :
  blue_shirts = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_shirts_count_l2832_283258


namespace NUMINAMATH_CALUDE_difference_in_rubber_bands_l2832_283203

-- Define the number of rubber bands Harper has
def harper_bands : ℕ := 15

-- Define the total number of rubber bands they have together
def total_bands : ℕ := 24

-- Define the number of rubber bands Harper's brother has
def brother_bands : ℕ := total_bands - harper_bands

-- Theorem to prove
theorem difference_in_rubber_bands :
  harper_bands - brother_bands = 6 ∧ brother_bands < harper_bands :=
by sorry

end NUMINAMATH_CALUDE_difference_in_rubber_bands_l2832_283203


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2832_283219

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, 2^n + 65 = m^2) ↔ (n = 10 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2832_283219


namespace NUMINAMATH_CALUDE_total_birds_l2832_283274

/-- The number of birds on an oak tree --/
def bird_count (bluebirds cardinals goldfinches sparrows robins swallows : ℕ) : Prop :=
  -- There are twice as many cardinals as bluebirds
  cardinals = 2 * bluebirds ∧
  -- The number of goldfinches is equal to the product of bluebirds and swallows
  goldfinches = bluebirds * swallows ∧
  -- The number of sparrows is half the sum of cardinals and goldfinches
  2 * sparrows = cardinals + goldfinches ∧
  -- The number of robins is 2 less than the quotient of bluebirds divided by swallows
  robins + 2 = bluebirds / swallows ∧
  -- There are 12 swallows
  swallows = 12 ∧
  -- The number of swallows is half as many as the number of bluebirds
  2 * swallows = bluebirds

theorem total_birds (bluebirds cardinals goldfinches sparrows robins swallows : ℕ) :
  bird_count bluebirds cardinals goldfinches sparrows robins swallows →
  bluebirds + cardinals + goldfinches + sparrows + robins + swallows = 540 :=
by sorry

end NUMINAMATH_CALUDE_total_birds_l2832_283274


namespace NUMINAMATH_CALUDE_perpendicular_line_properties_l2832_283284

/-- Given a line l₁ and a point A, this theorem proves properties of the perpendicular line l₂ passing through A -/
theorem perpendicular_line_properties (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 1 = 0
  let A : ℝ × ℝ := (3, 0)
  let l₂ : ℝ → ℝ → Prop := λ x y => 4 * x - 3 * y - 12 = 0
  -- l₂ passes through A
  (l₂ A.1 A.2) →
  -- l₂ is perpendicular to l₁
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₁ x₂ y₂ → l₂ x₁ y₁ → l₂ x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (3) + (y₂ - y₁) * (4)) * ((x₂ - x₁) * (4) + (y₂ - y₁) * (-3)) = 0) →
  -- The equation of l₂ is correct
  (∀ x y, l₂ x y ↔ 4 * x - 3 * y - 12 = 0) ∧
  -- The area of the triangle is 6
  (let x_intercept := 3
   let y_intercept := 4
   (1 / 2 : ℝ) * x_intercept * y_intercept = 6) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_properties_l2832_283284


namespace NUMINAMATH_CALUDE_complement_of_A_is_zero_l2832_283281

def A : Set ℤ := {x | |x| ≥ 1}

theorem complement_of_A_is_zero : 
  (Set.univ : Set ℤ) \ A = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_is_zero_l2832_283281


namespace NUMINAMATH_CALUDE_city_rentals_per_mile_rate_l2832_283255

/-- Represents the daily rental rate in dollars -/
def daily_rate_sunshine : ℝ := 17.99

/-- Represents the per-mile rate for Sunshine Car Rentals in dollars -/
def per_mile_rate_sunshine : ℝ := 0.18

/-- Represents the daily rental rate for City Rentals in dollars -/
def daily_rate_city : ℝ := 18.95

/-- Represents the number of miles driven -/
def miles_driven : ℝ := 48

/-- Represents the unknown per-mile rate for City Rentals -/
def per_mile_rate_city : ℝ := 0.16

theorem city_rentals_per_mile_rate :
  daily_rate_sunshine + per_mile_rate_sunshine * miles_driven =
  daily_rate_city + per_mile_rate_city * miles_driven :=
by sorry

#check city_rentals_per_mile_rate

end NUMINAMATH_CALUDE_city_rentals_per_mile_rate_l2832_283255


namespace NUMINAMATH_CALUDE_convex_ngon_angle_theorem_l2832_283217

theorem convex_ngon_angle_theorem (n : ℕ) : 
  n ≥ 3 →  -- n-gon must have at least 3 sides
  (∃ (x : ℝ), x > 0 ∧ x < 150 ∧ 150 * (n - 1) + x = 180 * (n - 2)) →
  (n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11) :=
by sorry

end NUMINAMATH_CALUDE_convex_ngon_angle_theorem_l2832_283217


namespace NUMINAMATH_CALUDE_total_time_is_twelve_years_l2832_283289

def years_to_get_in_shape : ℕ := 2
def years_to_learn_climbing (y : ℕ) : ℕ := 2 * y
def number_of_mountains : ℕ := 7
def months_per_mountain : ℕ := 5
def months_to_learn_diving : ℕ := 13
def years_of_diving : ℕ := 2

def total_time : ℕ :=
  years_to_get_in_shape +
  years_to_learn_climbing years_to_get_in_shape +
  (number_of_mountains * months_per_mountain + months_to_learn_diving) / 12 +
  years_of_diving

theorem total_time_is_twelve_years :
  total_time = 12 := by sorry

end NUMINAMATH_CALUDE_total_time_is_twelve_years_l2832_283289


namespace NUMINAMATH_CALUDE_ediths_books_l2832_283220

/-- The total number of books Edith has, given the number of novels and their relation to writing books -/
theorem ediths_books (novels : ℕ) (writing_books : ℕ) 
  (h1 : novels = 80) 
  (h2 : novels = writing_books / 2) : 
  novels + writing_books = 240 := by
  sorry

end NUMINAMATH_CALUDE_ediths_books_l2832_283220


namespace NUMINAMATH_CALUDE_parallelogram_circles_theorem_l2832_283257

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Checks if four points form a parallelogram -/
def is_parallelogram (p : Parallelogram) : Prop :=
  -- Add parallelogram conditions here
  sorry

/-- Checks if a circle passes through four points -/
def circle_passes_through (c : Circle) (p1 p2 p3 p4 : Point) : Prop :=
  -- Add circle condition here
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Add distance calculation here
  sorry

theorem parallelogram_circles_theorem (ABCD : Parallelogram) (E F : Point) (ω1 ω2 : Circle) :
  is_parallelogram ABCD →
  distance ABCD.A ABCD.B > distance ABCD.B ABCD.C →
  circle_passes_through ω1 ABCD.A ABCD.D E F →
  circle_passes_through ω2 ABCD.B ABCD.C E F →
  ∃ (X Y : Point),
    distance ABCD.B X = 200 ∧
    distance X Y = 9 ∧
    distance Y ABCD.D = 80 →
    distance ABCD.B ABCD.C = 51 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_circles_theorem_l2832_283257


namespace NUMINAMATH_CALUDE_find_number_l2832_283212

theorem find_number (A B : ℕ+) (h1 : B = 286) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 26) : A = 210 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2832_283212


namespace NUMINAMATH_CALUDE_not_in_A_iff_less_than_neg_three_l2832_283272

-- Define the set A
def A : Set ℝ := {x | x + 3 ≥ 0}

-- State the theorem
theorem not_in_A_iff_less_than_neg_three (a : ℝ) : a ∉ A ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_not_in_A_iff_less_than_neg_three_l2832_283272


namespace NUMINAMATH_CALUDE_x_cube_plus_four_x_equals_eight_l2832_283240

theorem x_cube_plus_four_x_equals_eight (x : ℝ) (h : x^3 + 4*x = 8) :
  x^7 + 64*x^2 = 128 := by
sorry

end NUMINAMATH_CALUDE_x_cube_plus_four_x_equals_eight_l2832_283240


namespace NUMINAMATH_CALUDE_at_least_one_genuine_product_l2832_283285

theorem at_least_one_genuine_product (total : Nat) (genuine : Nat) (defective : Nat) (selected : Nat) :
  total = genuine + defective →
  total = 12 →
  genuine = 10 →
  defective = 2 →
  selected = 3 →
  ∀ (selection : Finset (Fin total)),
    selection.card = selected →
    ∃ (i : Fin total), i ∈ selection ∧ i.val < genuine :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_genuine_product_l2832_283285


namespace NUMINAMATH_CALUDE_max_value_fraction_l2832_283271

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -3 ≤ x' ∧ x' ≤ -1 → 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2832_283271


namespace NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l2832_283269

/-- Given a sequence {a_n} where (n, a_n) lies on the line y = 2x,
    prove that it is an arithmetic sequence with common difference 2 -/
theorem sequence_on_line_is_arithmetic (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = 2 * n) →
  (∀ n : ℕ, a (n + 1) - a n = 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l2832_283269


namespace NUMINAMATH_CALUDE_odd_function_theorem_l2832_283245

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 4

-- State the theorem
theorem odd_function_theorem (h1 : is_odd f) (h2 : satisfies_equation f) : f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_theorem_l2832_283245


namespace NUMINAMATH_CALUDE_distinct_integers_with_swapped_digits_l2832_283292

def has_2n_digits (x : ℕ) (n : ℕ) : Prop :=
  10^(2*n - 1) ≤ x ∧ x < 10^(2*n)

def first_n_digits (x : ℕ) (n : ℕ) : ℕ :=
  x / 10^n

def last_n_digits (x : ℕ) (n : ℕ) : ℕ :=
  x % 10^n

theorem distinct_integers_with_swapped_digits (n : ℕ) (a b : ℕ) :
  n > 0 →
  a ≠ b →
  a > 0 ∧ b > 0 →
  has_2n_digits a n →
  has_2n_digits b n →
  a ∣ b →
  first_n_digits a n = last_n_digits b n →
  last_n_digits a n = first_n_digits b n →
  ((a = 2442 ∧ b = 4224) ∨ (a = 3993 ∧ b = 9339)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_integers_with_swapped_digits_l2832_283292


namespace NUMINAMATH_CALUDE_complex_number_equality_l2832_283227

theorem complex_number_equality (z : ℂ) :
  Complex.abs (z - 2) = 5 ∧ 
  Complex.abs (z + 4) = 5 ∧ 
  Complex.abs (z - 2*I) = 5 → 
  z = -1 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2832_283227


namespace NUMINAMATH_CALUDE_factorization_equality_l2832_283282

theorem factorization_equality (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2832_283282


namespace NUMINAMATH_CALUDE_smallest_k_for_15_digit_period_l2832_283231

/-- Represents a positive rational number with a decimal representation having a period of 30 digits -/
def RationalWith30DigitPeriod : Type := { q : ℚ // q > 0 ∧ ∃ m : ℕ, q = m / (10^30 - 1) }

/-- The theorem statement -/
theorem smallest_k_for_15_digit_period 
  (a b : RationalWith30DigitPeriod)
  (h_diff : ∃ p : ℤ, (a.val - b.val : ℚ) = p / (10^15 - 1)) :
  (∃ k : ℕ, k > 0 ∧ ∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)) ∧
  (∀ k : ℕ, k > 0 → k < 6 → ¬∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_15_digit_period_l2832_283231


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l2832_283248

/-- Given a 2x2 matrix B with its inverse, prove that the inverse of B^3 is as stated. -/
theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, 4], ![-2, -2]]) : 
  (B^3)⁻¹ = ![![3, 4], ![-6, -28]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l2832_283248


namespace NUMINAMATH_CALUDE_defective_product_selection_l2832_283279

/-- The set of possible numbers of defective products when selecting from a pool --/
def PossibleDefectives (total : ℕ) (defective : ℕ) (selected : ℕ) : Set ℕ :=
  {n : ℕ | n ≤ min defective selected ∧ n ≤ selected ∧ defective - n ≤ total - selected}

/-- Theorem stating the possible values for the number of defective products selected --/
theorem defective_product_selection (total : ℕ) (defective : ℕ) (selected : ℕ)
  (h_total : total = 8)
  (h_defective : defective = 2)
  (h_selected : selected = 3) :
  PossibleDefectives total defective selected = {0, 1, 2} :=
by sorry

end NUMINAMATH_CALUDE_defective_product_selection_l2832_283279


namespace NUMINAMATH_CALUDE_ball_returns_in_five_throws_l2832_283268

/-- The number of elements in the circular arrangement -/
def n : ℕ := 13

/-- The number of elements skipped in each throw -/
def skip : ℕ := 4

/-- The number of throws needed to return to the starting element -/
def throws : ℕ := 5

/-- Function to calculate the next position after a throw -/
def nextPosition (current : ℕ) : ℕ :=
  (current + skip + 1) % n

/-- Theorem stating that it takes 5 throws to return to the starting position -/
theorem ball_returns_in_five_throws :
  (throws.iterate nextPosition 0) % n = 0 := by sorry

end NUMINAMATH_CALUDE_ball_returns_in_five_throws_l2832_283268


namespace NUMINAMATH_CALUDE_select_four_from_eighteen_l2832_283261

theorem select_four_from_eighteen (n m : ℕ) : n = 18 ∧ m = 4 → Nat.choose n m = 3060 := by
  sorry

end NUMINAMATH_CALUDE_select_four_from_eighteen_l2832_283261


namespace NUMINAMATH_CALUDE_mary_final_card_count_l2832_283207

/-- The number of baseball cards Mary has after repairing some torn cards, receiving gifts, and buying new ones. -/
def final_card_count (initial_cards torn_cards repaired_percentage gift_cards bought_cards : ℕ) : ℕ :=
  let repaired_cards := (torn_cards * repaired_percentage) / 100
  let cards_after_repair := initial_cards - torn_cards + repaired_cards
  cards_after_repair + gift_cards + bought_cards

/-- Theorem stating that Mary ends up with 82 baseball cards given the initial conditions. -/
theorem mary_final_card_count :
  final_card_count 18 8 75 26 40 = 82 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_card_count_l2832_283207


namespace NUMINAMATH_CALUDE_sarahs_mean_score_l2832_283233

def scores : List ℕ := [78, 80, 85, 87, 90, 95, 100]

theorem sarahs_mean_score 
  (john_score_count : ℕ) 
  (sarah_score_count : ℕ)
  (total_score_count : ℕ)
  (john_mean : ℚ)
  (h1 : john_score_count = 4)
  (h2 : sarah_score_count = 3)
  (h3 : total_score_count = john_score_count + sarah_score_count)
  (h4 : john_mean = 86)
  (h5 : (scores.sum : ℚ) = john_mean * john_score_count + sarah_score_count * sarah_mean) :
  sarah_mean = 90 + 1/3 := by
    sorry

#check sarahs_mean_score

end NUMINAMATH_CALUDE_sarahs_mean_score_l2832_283233


namespace NUMINAMATH_CALUDE_max_sphere_area_from_cube_l2832_283247

/-- The maximum surface area of a sphere carved from a cube -/
theorem max_sphere_area_from_cube (cube_side : ℝ) (sphere_radius : ℝ) : 
  cube_side = 2 →
  sphere_radius ≤ 1 →
  sphere_radius > 0 →
  (4 : ℝ) * Real.pi * sphere_radius^2 ≤ 4 * Real.pi :=
by
  sorry

#check max_sphere_area_from_cube

end NUMINAMATH_CALUDE_max_sphere_area_from_cube_l2832_283247


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2832_283266

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ 
  (m * 0^2 + 5 * 0 + m^2 - 2 * m = 0) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2832_283266


namespace NUMINAMATH_CALUDE_population_growth_percentage_l2832_283238

theorem population_growth_percentage (initial_population final_population : ℕ) 
  (h1 : initial_population = 684)
  (h2 : final_population = 513) :
  ∃ (P : ℝ), 
    (P > 0) ∧ 
    (initial_population : ℝ) * (1 + P / 100) * (1 - 40 / 100) = final_population ∧
    P = 25 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_percentage_l2832_283238


namespace NUMINAMATH_CALUDE_square_sum_and_reciprocal_l2832_283243

theorem square_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_reciprocal_l2832_283243


namespace NUMINAMATH_CALUDE_division_remainder_l2832_283224

theorem division_remainder : ∀ (dividend divisor quotient : ℕ),
  dividend = 166 →
  divisor = 18 →
  quotient = 9 →
  dividend = divisor * quotient + 4 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l2832_283224


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2832_283293

theorem sphere_surface_area (volume : ℝ) (h : volume = 72 * Real.pi) :
  let r := (3 * volume / (4 * Real.pi)) ^ (1/3)
  4 * Real.pi * r^2 = 36 * 2^(2/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2832_283293


namespace NUMINAMATH_CALUDE_expression_value_at_three_l2832_283235

theorem expression_value_at_three :
  let x : ℝ := 3
  (x^3 - 2*x^2 - 21*x + 36) / (x - 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l2832_283235


namespace NUMINAMATH_CALUDE_greatest_integer_for_fraction_l2832_283215

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem greatest_integer_for_fraction : 
  (∀ x : ℤ, x > 14 → ¬is_integer ((x^2 - 5*x + 14) / (x - 4))) ∧
  is_integer ((14^2 - 5*14 + 14) / (14 - 4)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_fraction_l2832_283215


namespace NUMINAMATH_CALUDE_trig_equation_solution_l2832_283283

theorem trig_equation_solution (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 2 ∨ 
            x = k * Real.pi / 2 + Real.pi / 4 ∨ 
            x = k * Real.pi / 3 + Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l2832_283283


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2832_283214

theorem sum_mod_nine : (8150 + 8151 + 8152 + 8153 + 8154 + 8155) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2832_283214


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l2832_283275

/-- Proves that given a total sum of 2678, if the interest on the first part for 8 years at 3% per annum
    is equal to the interest on the second part for 3 years at 5% per annum, then the second part is 1648. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first : ℝ) (second : ℝ) :
  total = 2678 →
  first + second = total →
  (first * (3/100) * 8) = (second * (5/100) * 3) →
  second = 1648 := by
sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l2832_283275


namespace NUMINAMATH_CALUDE_female_democrats_count_l2832_283249

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 720 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 120 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2832_283249


namespace NUMINAMATH_CALUDE_probability_one_success_out_of_three_l2832_283294

/-- The probability of passing a single computer test -/
def p : ℚ := 1 / 3

/-- The number of tests taken -/
def n : ℕ := 3

/-- The number of successful attempts -/
def k : ℕ := 1

/-- Binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℚ := (n.choose k : ℚ)

/-- The probability of passing exactly k tests out of n attempts -/
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

theorem probability_one_success_out_of_three :
  probability_k_successes n k p = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_one_success_out_of_three_l2832_283294


namespace NUMINAMATH_CALUDE_decimal_sum_as_fraction_l2832_283218

theorem decimal_sum_as_fraction : 
  (0.2 + 0.03 + 0.004 + 0.0006 + 0.00007 + 0.000008 + 0.0000009 : ℚ) = 2340087/10000000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_as_fraction_l2832_283218


namespace NUMINAMATH_CALUDE_system_solution_l2832_283241

theorem system_solution : ∃! (x y : ℚ), 3 * x + 4 * y = 20 ∧ 9 * x - 8 * y = 36 ∧ x = 76 / 15 ∧ y = 18 / 15 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2832_283241


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l2832_283237

-- Define variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 (x y : ℝ) : Prop := 2 * x - 3 * y < x - 1
def condition2 (x y : ℝ) : Prop := 3 * x + 4 * y > 2 * y + 5

-- State the theorem
theorem relationship_between_x_and_y 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  x < 3 * y - 1 ∧ y > (5 - 3 * x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l2832_283237


namespace NUMINAMATH_CALUDE_polygon_25_sides_l2832_283267

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2

/-- Number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of triangles formed by choosing any three vertices of a polygon with n sides -/
def numTriangles (n : ℕ) : ℕ := n.choose 3

theorem polygon_25_sides (P : ConvexPolygon 25) : 
  numDiagonals 25 = 275 ∧ numTriangles 25 = 2300 := by
  sorry


end NUMINAMATH_CALUDE_polygon_25_sides_l2832_283267


namespace NUMINAMATH_CALUDE_xy_max_and_x2_y2_min_l2832_283276

theorem xy_max_and_x2_y2_min (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (∃ (x0 y0 : ℝ), x0 > 0 ∧ y0 > 0 ∧ x0 + 2*y0 = 1 ∧ x0*y0 = 1/8 ∧ ∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 2*y' = 1 → x'*y' ≤ 1/8) ∧
  (x^2 + y^2 ≥ 1/5 ∧ ∃ (x1 y1 : ℝ), x1 > 0 ∧ y1 > 0 ∧ x1 + 2*y1 = 1 ∧ x1^2 + y1^2 = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_xy_max_and_x2_y2_min_l2832_283276


namespace NUMINAMATH_CALUDE_smallest_cube_ending_2016_l2832_283234

theorem smallest_cube_ending_2016 :
  ∃ (n : ℕ), n^3 % 10000 = 2016 ∧ ∀ (m : ℕ), m < n → m^3 % 10000 ≠ 2016 :=
by
  use 856
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_2016_l2832_283234


namespace NUMINAMATH_CALUDE_power_division_rule_l2832_283229

theorem power_division_rule (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2832_283229


namespace NUMINAMATH_CALUDE_number_puzzle_l2832_283208

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 9) = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2832_283208


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l2832_283280

-- Define an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = m

-- Define an equal difference of squares sequence
def is_equal_diff_squares_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1)^2 - a n^2 = d

theorem m_zero_sufficient_not_necessary :
  (∀ b : ℕ → ℝ, is_arithmetic_seq b 0 → is_equal_diff_squares_seq b) ∧
  (∃ b : ℕ → ℝ, ∃ m : ℝ, m ≠ 0 ∧ is_arithmetic_seq b m ∧ is_equal_diff_squares_seq b) :=
by sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l2832_283280


namespace NUMINAMATH_CALUDE_shopkeeper_total_cards_l2832_283211

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : ℕ := 3

-- Define the number of additional cards
def additional_cards : ℕ := 4

-- Theorem to prove
theorem shopkeeper_total_cards : 
  complete_decks * standard_deck_size + additional_cards = 160 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_cards_l2832_283211


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2832_283200

theorem isosceles_triangle_angle_measure :
  ∀ (D E F : ℝ),
  -- Triangle DEF is isosceles with angle D congruent to angle F
  D = F →
  -- The measure of angle F is three times the measure of angle E
  F = 3 * E →
  -- The sum of angles in a triangle is 180 degrees
  D + E + F = 180 →
  -- The measure of angle D is 540/7 degrees
  D = 540 / 7 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2832_283200


namespace NUMINAMATH_CALUDE_supermarket_growth_l2832_283244

/-- Represents the growth of a supermarket's turnover from January to March -/
theorem supermarket_growth (x : ℝ) : 
  (36 : ℝ) * (1 + x)^2 = 48 ↔ 
  (∃ (jan mar : ℝ), 
    jan = 36 ∧ 
    mar = 48 ∧ 
    mar = jan * (1 + x)^2 ∧ 
    x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_supermarket_growth_l2832_283244


namespace NUMINAMATH_CALUDE_chess_piece_arrangements_l2832_283265

theorem chess_piece_arrangements (n m : ℕ) (hn : n = 9) (hm : m = 6) :
  (Finset.card (Finset.univ : Finset (Fin n → Fin m))) = (14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6) := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_arrangements_l2832_283265


namespace NUMINAMATH_CALUDE_cube_cutting_l2832_283222

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l2832_283222


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2832_283221

theorem algebraic_expression_value (a b : ℝ) (h : 3 * a * b - 3 * b^2 - 2 = 0) :
  (1 - (2 * a * b - b^2) / a^2) / ((a - b) / (a^2 * b)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2832_283221


namespace NUMINAMATH_CALUDE_mode_is_97_l2832_283277

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The list of all scores from the stem-and-leaf plot -/
def scores : List Score := [
  ⟨6, 5⟩, ⟨6, 5⟩,
  ⟨7, 1⟩, ⟨7, 3⟩, ⟨7, 3⟩, ⟨7, 6⟩,
  ⟨8, 0⟩, ⟨8, 0⟩, ⟨8, 4⟩, ⟨8, 4⟩, ⟨8, 8⟩, ⟨8, 8⟩, ⟨8, 8⟩,
  ⟨9, 2⟩, ⟨9, 2⟩, ⟨9, 5⟩, ⟨9, 7⟩, ⟨9, 7⟩, ⟨9, 7⟩, ⟨9, 7⟩,
  ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 4⟩, ⟨10, 6⟩,
  ⟨11, 0⟩, ⟨11, 0⟩, ⟨11, 0⟩
]

/-- Convert a Score to its numerical value -/
def scoreValue (s : Score) : Nat :=
  s.stem * 10 + s.leaf

/-- Count the occurrences of a value in the list of scores -/
def countOccurrences (value : Nat) : Nat :=
  (scores.filter (fun s => scoreValue s = value)).length

/-- The mode is the most frequent score -/
def isMode (value : Nat) : Prop :=
  ∀ (other : Nat), countOccurrences value ≥ countOccurrences other

/-- Theorem: The mode of the scores is 97 -/
theorem mode_is_97 : isMode 97 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_97_l2832_283277


namespace NUMINAMATH_CALUDE_manufacturer_buyers_count_l2832_283236

theorem manufacturer_buyers_count :
  ∀ (N : ℕ) 
    (cake_buyers muffin_buyers both_buyers : ℕ)
    (prob_neither : ℚ),
  cake_buyers = 50 →
  muffin_buyers = 40 →
  both_buyers = 15 →
  prob_neither = 1/4 →
  (N : ℚ) * prob_neither = N - (cake_buyers + muffin_buyers - both_buyers) →
  N = 100 := by
sorry

end NUMINAMATH_CALUDE_manufacturer_buyers_count_l2832_283236


namespace NUMINAMATH_CALUDE_lucy_shells_found_l2832_283263

theorem lucy_shells_found (initial_shells final_shells : ℝ) 
  (h1 : initial_shells = 68.3)
  (h2 : final_shells = 89.5) :
  final_shells - initial_shells = 21.2 := by
  sorry

end NUMINAMATH_CALUDE_lucy_shells_found_l2832_283263


namespace NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l2832_283228

/-- Represents a gathering of people with specific knowledge relationships. -/
structure Gathering where
  total : Nat
  know_each_other : Nat
  know_no_one : Nat

/-- Calculates the number of handshakes in a gathering. -/
def count_handshakes (g : Gathering) : Nat :=
  g.know_no_one * (g.total - 1)

/-- Theorem stating that in a specific gathering, 217 handshakes occur. -/
theorem handshakes_in_specific_gathering :
  ∃ (g : Gathering),
    g.total = 30 ∧
    g.know_each_other = 15 ∧
    g.know_no_one = 15 ∧
    count_handshakes g = 217 := by
  sorry

#check handshakes_in_specific_gathering

end NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l2832_283228


namespace NUMINAMATH_CALUDE_problem_solution_l2832_283259

theorem problem_solution (a b c : ℝ) : 
  (-(a) = -2) → 
  (1 / b = -3/2) → 
  (abs c = 2) → 
  a + b + c^2 = 16/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2832_283259


namespace NUMINAMATH_CALUDE_dot_product_a_b_l2832_283246

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![3, -2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Theorem statement
theorem dot_product_a_b : dot_product a b = 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_a_b_l2832_283246


namespace NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l2832_283291

/-- Given a circle and a line, proves that the radius of the circle is 11 
    when the chord formed by their intersection has length 6 -/
theorem circle_line_intersection_chord_length (a : ℝ) : 
  (∃ x y : ℝ, (x + 2)^2 + (y - 2)^2 = a ∧ x + y + 2 = 0) →  -- Circle intersects line
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + 2)^2 + (y₁ - 2)^2 = a ∧ 
    (x₂ + 2)^2 + (y₂ - 2)^2 = a ∧ 
    x₁ + y₁ + 2 = 0 ∧ 
    x₂ + y₂ + 2 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →  -- Chord length is 6
  a = 11 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l2832_283291


namespace NUMINAMATH_CALUDE_selection_schemes_six_four_two_l2832_283296

/-- The number of ways to select 4 students from 6 to visit 4 cities,
    where 2 specific students cannot visit one particular city. -/
def selection_schemes (n : ℕ) (k : ℕ) (restricted : ℕ) : ℕ :=
  (n - restricted) * (n - restricted - 1) * (n - 2) * (n - 3)

theorem selection_schemes_six_four_two :
  selection_schemes 6 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_six_four_two_l2832_283296


namespace NUMINAMATH_CALUDE_combined_average_score_l2832_283250

/-- Given two math modeling clubs with their respective member counts and average scores,
    calculate the combined average score of both clubs. -/
theorem combined_average_score
  (club_a_members : ℕ)
  (club_b_members : ℕ)
  (club_a_average : ℝ)
  (club_b_average : ℝ)
  (h1 : club_a_members = 40)
  (h2 : club_b_members = 50)
  (h3 : club_a_average = 90)
  (h4 : club_b_average = 81) :
  (club_a_members * club_a_average + club_b_members * club_b_average) /
  (club_a_members + club_b_members : ℝ) = 85 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l2832_283250


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2832_283230

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 480 → majority = 192 → 
  ∃ (winner_percentage : ℚ), 
    winner_percentage = 70 / 100 ∧ 
    (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = majority :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2832_283230


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2832_283299

theorem cubic_expression_evaluation (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 7 = 219 / 7 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2832_283299


namespace NUMINAMATH_CALUDE_range_of_a_l2832_283252

def A (a : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l2832_283252


namespace NUMINAMATH_CALUDE_speed_decrease_percentage_l2832_283278

theorem speed_decrease_percentage (distance : ℝ) (fast_speed slow_speed : ℝ) 
  (h_distance_positive : distance > 0)
  (h_fast_speed_positive : fast_speed > 0)
  (h_slow_speed_positive : slow_speed > 0)
  (h_fast_time : distance / fast_speed = 40)
  (h_slow_time : distance / slow_speed = 50) :
  (fast_speed - slow_speed) / fast_speed = 1/5 := by
sorry

end NUMINAMATH_CALUDE_speed_decrease_percentage_l2832_283278


namespace NUMINAMATH_CALUDE_product_is_real_product_is_imaginary_l2832_283225

/-- The product of two complex numbers is real if and only if ad + bc = 0 -/
theorem product_is_real (a b c d : ℝ) :
  (Complex.I * b + a) * (Complex.I * d + c) ∈ Set.range (Complex.ofReal) ↔ a * d + b * c = 0 := by
  sorry

/-- The product of two complex numbers is purely imaginary if and only if ac - bd = 0 -/
theorem product_is_imaginary (a b c d : ℝ) :
  ∃ (k : ℝ), (Complex.I * b + a) * (Complex.I * d + c) = Complex.I * k ↔ a * c - b * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_is_real_product_is_imaginary_l2832_283225


namespace NUMINAMATH_CALUDE_nine_integer_lengths_l2832_283298

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex E to the hypotenuse DF in a right triangle DEF -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

theorem nine_integer_lengths (t : RightTriangle) 
  (h1 : t.de = 24) (h2 : t.ef = 25) : 
  countIntegerLengths t = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_lengths_l2832_283298


namespace NUMINAMATH_CALUDE_vector_collinearity_l2832_283216

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (3, 2)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, u.1 * v.2 = t * u.2 * v.1

theorem vector_collinearity (k : ℝ) :
  collinear c ((k * a.1 + b.1, k * a.2 + b.2)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2832_283216


namespace NUMINAMATH_CALUDE_some_multiplier_value_l2832_283223

theorem some_multiplier_value : ∃ m : ℕ, (422 + 404)^2 - (m * 422 * 404) = 324 ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_multiplier_value_l2832_283223
