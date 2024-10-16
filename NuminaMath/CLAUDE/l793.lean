import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_at_one_fifth_l793_79326

theorem expression_value_at_one_fifth :
  let x : ℚ := 1/5
  (x^2 - 4) / (x^2 - 2*x) = 11 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_one_fifth_l793_79326


namespace NUMINAMATH_CALUDE_fifth_score_for_average_85_l793_79369

/-- Given four test scores and a desired average, calculate the required fifth score -/
def calculate_fifth_score (score1 score2 score3 score4 : ℕ) (desired_average : ℚ) : ℚ :=
  5 * desired_average - (score1 + score2 + score3 + score4)

/-- Theorem: The fifth score needed to achieve an average of 85 given the first four scores -/
theorem fifth_score_for_average_85 :
  calculate_fifth_score 85 79 92 84 85 = 85 := by sorry

end NUMINAMATH_CALUDE_fifth_score_for_average_85_l793_79369


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_base6_l793_79399

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- The sum of an arithmetic series -/
def arithmeticSeriesSum (a : ℕ) (l : ℕ) (n : ℕ) : ℕ :=
  n * (a + l) / 2

theorem arithmetic_series_sum_base6 :
  let first := 1
  let last := base6ToBase10 55
  let terms := base6ToBase10 55
  let sum := arithmeticSeriesSum first last terms
  (sum = 630) ∧ (base10ToBase6 sum = 2530) := by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_base6_l793_79399


namespace NUMINAMATH_CALUDE_jogger_multiple_l793_79304

/-- The number of joggers bought by each person -/
structure JoggerPurchase where
  tyson : ℕ
  alexander : ℕ
  christopher : ℕ

/-- The conditions of the jogger purchase problem -/
def JoggerProblem (jp : JoggerPurchase) : Prop :=
  jp.alexander = jp.tyson + 22 ∧
  jp.christopher = 80 ∧
  jp.christopher = jp.alexander + 54 ∧
  ∃ m : ℕ, jp.christopher = m * jp.tyson

theorem jogger_multiple (jp : JoggerPurchase) (h : JoggerProblem jp) :
  ∃ m : ℕ, jp.christopher = m * jp.tyson ∧ m = 20 := by
  sorry

#check jogger_multiple

end NUMINAMATH_CALUDE_jogger_multiple_l793_79304


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l793_79317

/-- Nell's initial number of cards -/
def nell_initial : ℕ := 528

/-- Nell's remaining number of cards -/
def nell_remaining : ℕ := 252

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := nell_initial - nell_remaining

theorem cards_given_to_jeff : cards_given = 276 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l793_79317


namespace NUMINAMATH_CALUDE_gcd_of_168_and_294_l793_79363

theorem gcd_of_168_and_294 : Nat.gcd 168 294 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_168_and_294_l793_79363


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3_l793_79388

theorem unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3 :
  ∃! n : ℕ+, 20 ∣ n ∧ (8.2 : ℝ) < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 8.3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3_l793_79388


namespace NUMINAMATH_CALUDE_triangle_properties_l793_79313

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c →
  Real.sin A * Real.sin B = (Real.cos (C / 2))^2 →
  ((a^2 + b^2 - c^2) / 4 + (c^2 * (Real.cos (C / 2))^2)) = 7 →
  A = Real.pi / 6 ∧ B = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l793_79313


namespace NUMINAMATH_CALUDE_cookie_pans_problem_l793_79386

/-- Given a number of cookies per pan and a total number of cookies,
    calculate the number of pans needed. -/
def calculate_pans (cookies_per_pan : ℕ) (total_cookies : ℕ) : ℕ :=
  total_cookies / cookies_per_pan

theorem cookie_pans_problem :
  let cookies_per_pan : ℕ := 8
  let total_cookies : ℕ := 40
  calculate_pans cookies_per_pan total_cookies = 5 := by
  sorry

#eval calculate_pans 8 40

end NUMINAMATH_CALUDE_cookie_pans_problem_l793_79386


namespace NUMINAMATH_CALUDE_symmetric_polynomial_square_factor_l793_79351

/-- A polynomial in two variables that is symmetric in its arguments -/
def SymmetricPolynomial (R : Type) [CommRing R] :=
  {p : R → R → R // ∀ x y, p x y = p y x}

theorem symmetric_polynomial_square_factor
  {R : Type} [CommRing R] (p : SymmetricPolynomial R)
  (h : ∃ q : R → R → R, ∀ x y, p.val x y = (x - y) * q x y) :
  ∃ r : R → R → R, ∀ x y, p.val x y = (x - y)^2 * r x y := by
  sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_square_factor_l793_79351


namespace NUMINAMATH_CALUDE_parabola_comparison_l793_79305

theorem parabola_comparison :
  ∀ x : ℝ, x^2 - 3/4*x + 3 ≥ x^2 + 1/4*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_comparison_l793_79305


namespace NUMINAMATH_CALUDE_sophie_savings_l793_79373

/-- Represents the number of loads of laundry Sophie does per week -/
def loads_per_week : ℕ := 4

/-- Represents the number of dryer sheets Sophie uses per load -/
def sheets_per_load : ℕ := 1

/-- Represents the cost of a box of dryer sheets in dollars -/
def cost_per_box : ℚ := 5.5

/-- Represents the number of dryer sheets in a box -/
def sheets_per_box : ℕ := 104

/-- Represents the number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Theorem stating the amount of money Sophie saves in a year by not buying dryer sheets -/
theorem sophie_savings : 
  (loads_per_week * sheets_per_load * weeks_per_year / sheets_per_box : ℚ) * cost_per_box = 11 := by
  sorry

end NUMINAMATH_CALUDE_sophie_savings_l793_79373


namespace NUMINAMATH_CALUDE_divisibility_by_17_and_32_l793_79382

theorem divisibility_by_17_and_32 (n : ℕ) (hn : n > 0) :
  (∃ k : ℤ, 5 * 3^(4*n + 1) + 2^(6*n + 1) = 17 * k) ∧
  (∃ m : ℤ, 5^2 * 7^(2*n + 1) + 3^(4*n) = 32 * m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_17_and_32_l793_79382


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l793_79384

theorem decimal_to_fraction (x : ℚ) : x = 3.675 → x = 147 / 40 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l793_79384


namespace NUMINAMATH_CALUDE_equal_slopes_iff_parallel_l793_79347

-- Define the concept of a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define what it means for two lines to be distinct
def are_distinct (l1 l2 : Line) : Prop :=
  l1 ≠ l2

-- Theorem statement
theorem equal_slopes_iff_parallel (l1 l2 : Line) :
  are_distinct l1 l2 → (l1.slope = l2.slope ↔ are_parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_equal_slopes_iff_parallel_l793_79347


namespace NUMINAMATH_CALUDE_orchestra_members_count_l793_79377

theorem orchestra_members_count :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 300 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 1 ∧ 
  n % 7 = 5 ∧
  n = 651 := by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l793_79377


namespace NUMINAMATH_CALUDE_power_of_three_expression_l793_79354

theorem power_of_three_expression : ∀ (a b c d e f g h : ℕ), 
  a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 4 ∧ e = 8 ∧ f = 16 ∧ g = 32 ∧ h = 64 →
  3^a * 3^b / 3^c / 3^d / 3^e * 3^f * 3^g * 3^h = 3^99 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_l793_79354


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l793_79303

/- Define the quadratic function f(x) -/
def f (x : ℝ) := 2 * x^2 - 10 * x

/- Theorem stating the properties of f(x) and the solution sets -/
theorem quadratic_function_properties :
  (∀ x, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (∀ x ∈ Set.Icc (-1) 4, f x ≤ 12) ∧
  (∃ x ∈ Set.Icc (-1) 4, f x = 12) ∧
  (∀ a < 0,
    (∀ x, (2 * x^2 + (a - 10) * x + 5) / f x > 1 ↔
      ((-1 < a ∧ a < 0 ∧ (x < 0 ∨ (5 < x ∧ x < -5/a))) ∨
       (a = -1 ∧ x < 0) ∨
       (a < -1 ∧ (x < 0 ∨ (-5/a < x ∧ x < 5))))))
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l793_79303


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l793_79392

-- Define the universal set I as ℝ
def I : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^(Real.sqrt (3 + 2*x - x^2))}

-- Define set N
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 2)}

-- Theorem statement
theorem intersection_complement_theorem : M ∩ (I \ N) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l793_79392


namespace NUMINAMATH_CALUDE_conjunction_false_implies_one_false_l793_79311

theorem conjunction_false_implies_one_false (p q : Prop) :
  (p ∧ q) = False → (p = False ∨ q = False) :=
by sorry

end NUMINAMATH_CALUDE_conjunction_false_implies_one_false_l793_79311


namespace NUMINAMATH_CALUDE_mans_to_sons_age_ratio_l793_79390

/-- Given a man who is 28 years older than his son, and the son's present age is 26,
    prove that the ratio of the man's age to his son's age in two years is 2:1. -/
theorem mans_to_sons_age_ratio (son_age : ℕ) (man_age : ℕ) : 
  son_age = 26 → 
  man_age = son_age + 28 → 
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_to_sons_age_ratio_l793_79390


namespace NUMINAMATH_CALUDE_cookie_sugar_measurement_l793_79389

def sugar_needed : ℚ := 15/4  -- 3¾ cups of sugar
def cup_capacity : ℚ := 1/3   -- ⅓ cup measuring cup

theorem cookie_sugar_measurement : ∃ n : ℕ, n * cup_capacity ≥ sugar_needed ∧ 
  ∀ m : ℕ, m * cup_capacity ≥ sugar_needed → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_sugar_measurement_l793_79389


namespace NUMINAMATH_CALUDE_copper_in_mixture_l793_79300

theorem copper_in_mixture (lead_percentage : Real) (copper_percentage : Real) (lead_mass : Real) (copper_mass : Real) : 
  lead_percentage = 0.25 →
  copper_percentage = 0.60 →
  lead_mass = 5 →
  copper_mass = 12 →
  copper_mass = (copper_percentage / lead_percentage) * lead_mass :=
by
  sorry

#check copper_in_mixture

end NUMINAMATH_CALUDE_copper_in_mixture_l793_79300


namespace NUMINAMATH_CALUDE_base7_to_base49_conversion_l793_79342

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a list of digits in base 49 to a natural number -/
def fromBase49 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 49 * acc + d) 0

/-- The theorem stating the equality of the base 7 and base 49 representations -/
theorem base7_to_base49_conversion :
  fromBase7 [6, 2, 6] = fromBase49 [0, 6, 0, 2, 0, 6] := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base49_conversion_l793_79342


namespace NUMINAMATH_CALUDE_alternating_color_probability_l793_79387

def box := {white : ℕ // white = 5} × {black : ℕ // black = 5}

def total_arrangements (b : box) : ℕ := Nat.choose (b.1 + b.2) b.1

def alternating_arrangements : ℕ := 2

theorem alternating_color_probability (b : box) :
  (alternating_arrangements : ℚ) / (total_arrangements b : ℚ) = 1 / 126 :=
sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l793_79387


namespace NUMINAMATH_CALUDE_electricity_billing_theorem_l793_79312

/-- Represents the tariff rates for different zones --/
structure TariffRates where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Represents the meter readings --/
structure MeterReadings where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Calculates the maximum possible additional payment --/
def max_additional_payment (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) : ℝ :=
  sorry

/-- Calculates the expected difference between company's calculation and customer's payment --/
def expected_difference (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) : ℝ :=
  sorry

/-- Main theorem stating the correct results for the given problem --/
theorem electricity_billing_theorem (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) :
  rates.peak = 4.03 ∧ rates.night = 1.01 ∧ rates.half_peak = 3.39 ∧
  readings.a = 1214 ∧ readings.b = 1270 ∧ readings.c = 1298 ∧
  readings.d = 1337 ∧ readings.e = 1347 ∧ readings.f = 1402 ∧
  paid_amount = 660.72 →
  max_additional_payment rates readings paid_amount = 397.34 ∧
  expected_difference rates readings paid_amount = 19.30 :=
by sorry

end NUMINAMATH_CALUDE_electricity_billing_theorem_l793_79312


namespace NUMINAMATH_CALUDE_bake_sale_donation_percentage_l793_79379

/-- Proves that the percentage of bake sale proceeds donated to the shelter is 75% --/
theorem bake_sale_donation_percentage :
  ∀ (carwash_earnings bake_sale_earnings lawn_mowing_earnings total_donation : ℚ),
  carwash_earnings = 100 →
  bake_sale_earnings = 80 →
  lawn_mowing_earnings = 50 →
  total_donation = 200 →
  0.9 * carwash_earnings + 1 * lawn_mowing_earnings + 
    (total_donation - (0.9 * carwash_earnings + 1 * lawn_mowing_earnings)) = total_donation →
  (total_donation - (0.9 * carwash_earnings + 1 * lawn_mowing_earnings)) / bake_sale_earnings = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_bake_sale_donation_percentage_l793_79379


namespace NUMINAMATH_CALUDE_max_value_of_expression_l793_79376

theorem max_value_of_expression (x y : ℝ) 
  (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : 
  ∃ (a b : ℝ), |a - 2*b + 1| = 6 ∧ |x - 2*y + 1| ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l793_79376


namespace NUMINAMATH_CALUDE_tan_ratio_inequality_l793_79340

theorem tan_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2) : 
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_inequality_l793_79340


namespace NUMINAMATH_CALUDE_arrangements_without_A_at_head_l793_79368

def total_people : Nat := 5
def people_to_select : Nat := 3

def total_arrangements : Nat := total_people * (total_people - 1) * (total_people - 2)
def arrangements_with_A_at_head : Nat := (total_people - 1) * (total_people - 2)

theorem arrangements_without_A_at_head :
  total_arrangements - arrangements_with_A_at_head = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_without_A_at_head_l793_79368


namespace NUMINAMATH_CALUDE_tims_books_l793_79348

theorem tims_books (sandy_books : ℕ) (benny_books : ℕ) (total_books : ℕ) :
  sandy_books = 10 →
  benny_books = 24 →
  total_books = 67 →
  ∃ tim_books : ℕ, tim_books = total_books - (sandy_books + benny_books) ∧ tim_books = 33 :=
by sorry

end NUMINAMATH_CALUDE_tims_books_l793_79348


namespace NUMINAMATH_CALUDE_chess_draw_probability_l793_79372

theorem chess_draw_probability 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l793_79372


namespace NUMINAMATH_CALUDE_emma_in_middle_l793_79398

-- Define the friends
inductive Friend
| Allen
| Brian
| Chris
| Diana
| Emma

-- Define the car positions
inductive Position
| First
| Second
| Third
| Fourth
| Fifth

-- Define the seating arrangement
def Arrangement := Friend → Position

-- Define the conditions
def validArrangement (a : Arrangement) : Prop :=
  a Friend.Allen = Position.Second ∧
  a Friend.Diana = Position.First ∧
  (a Friend.Brian = Position.Fourth ∧ a Friend.Chris = Position.Fifth) ∨
  (a Friend.Brian = Position.Third ∧ a Friend.Chris = Position.Fourth) ∧
  (a Friend.Emma = Position.Third ∨ a Friend.Emma = Position.Fifth)

-- Theorem to prove
theorem emma_in_middle (a : Arrangement) :
  validArrangement a → a Friend.Emma = Position.Third :=
sorry

end NUMINAMATH_CALUDE_emma_in_middle_l793_79398


namespace NUMINAMATH_CALUDE_skill_player_water_consumption_l793_79319

/-- Proves that skill position players drink 6 ounces each given the conditions of the football team's water consumption problem. -/
theorem skill_player_water_consumption
  (total_water : ℕ)
  (num_linemen : ℕ)
  (num_skill_players : ℕ)
  (lineman_consumption : ℕ)
  (num_skill_players_before_refill : ℕ)
  (h1 : total_water = 126)
  (h2 : num_linemen = 12)
  (h3 : num_skill_players = 10)
  (h4 : lineman_consumption = 8)
  (h5 : num_skill_players_before_refill = 5)
  : (total_water - num_linemen * lineman_consumption) / num_skill_players_before_refill = 6 := by
  sorry

#check skill_player_water_consumption

end NUMINAMATH_CALUDE_skill_player_water_consumption_l793_79319


namespace NUMINAMATH_CALUDE_ball_cost_price_l793_79375

/-- The cost price of a single ball -/
def cost_price : ℕ := sorry

/-- The selling price of 20 balls -/
def selling_price : ℕ := 720

/-- The number of balls sold -/
def balls_sold : ℕ := 20

/-- The number of balls whose cost equals the loss -/
def balls_loss : ℕ := 5

theorem ball_cost_price : 
  cost_price = 48 ∧ 
  selling_price = balls_sold * cost_price - balls_loss * cost_price :=
sorry

end NUMINAMATH_CALUDE_ball_cost_price_l793_79375


namespace NUMINAMATH_CALUDE_flowers_in_vase_l793_79366

theorem flowers_in_vase (roses : ℕ) (carnations : ℕ) : 
  roses = 5 → carnations = 5 → roses + carnations = 10 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_vase_l793_79366


namespace NUMINAMATH_CALUDE_pyramid_volume_l793_79337

/-- Volume of a pyramid with a right triangular base -/
theorem pyramid_volume (c α β : ℝ) (hc : c > 0) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  let volume := c^3 * Real.sin (2*α) * Real.tan β / 24
  ∃ (V : ℝ), V = volume ∧ V > 0 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l793_79337


namespace NUMINAMATH_CALUDE_complex_square_on_positive_imaginary_axis_l793_79364

theorem complex_square_on_positive_imaginary_axis (a : ℝ) :
  let z : ℂ := a + 2 * Complex.I
  (∃ (y : ℝ), y > 0 ∧ z^2 = Complex.I * y) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_positive_imaginary_axis_l793_79364


namespace NUMINAMATH_CALUDE_circle_area_ratio_l793_79357

theorem circle_area_ratio (R S : ℝ) (hR : R > 0) (hS : S > 0) (h : R = 0.2 * S) :
  (π * (R / 2)^2) / (π * (S / 2)^2) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l793_79357


namespace NUMINAMATH_CALUDE_initial_value_proof_l793_79374

-- Define the property tax rate
def tax_rate : ℝ := 0.10

-- Define the new assessed value
def new_value : ℝ := 28000

-- Define the property tax increase
def tax_increase : ℝ := 800

-- Theorem statement
theorem initial_value_proof :
  ∃ (initial_value : ℝ),
    initial_value * tax_rate + tax_increase = new_value * tax_rate ∧
    initial_value = 20000 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l793_79374


namespace NUMINAMATH_CALUDE_bucket_capacity_l793_79310

/-- The capacity of a bucket in litres, given that when it is 2/3 full, it contains 9 litres of maple syrup. -/
theorem bucket_capacity : ℝ := by
  -- Define the capacity of the bucket
  let C : ℝ := 13.5

  -- Define the fraction of the bucket that is full
  let fraction_full : ℝ := 2/3

  -- Define the current volume of maple syrup
  let current_volume : ℝ := 9

  -- State that the current volume is equal to the fraction of the capacity
  have h1 : fraction_full * C = current_volume := by sorry

  -- Prove that the capacity is indeed 13.5 litres
  have h2 : C = 13.5 := by sorry

  -- Return the capacity
  exact C

end NUMINAMATH_CALUDE_bucket_capacity_l793_79310


namespace NUMINAMATH_CALUDE_sqrt_4_minus_x_real_range_l793_79308

theorem sqrt_4_minus_x_real_range : 
  {x : ℝ | ∃ y : ℝ, y ^ 2 = 4 - x} = {x : ℝ | x ≤ 4} := by
sorry

end NUMINAMATH_CALUDE_sqrt_4_minus_x_real_range_l793_79308


namespace NUMINAMATH_CALUDE_bee_closest_point_to_flower_l793_79332

/-- The point where the bee starts moving away from the flower -/
def closest_point : ℝ × ℝ := (4.6, 13.8)

/-- The location of the flower -/
def flower_location : ℝ × ℝ := (10, 12)

/-- The path of the bee -/
def bee_path (x : ℝ) : ℝ := 3 * x

theorem bee_closest_point_to_flower :
  let (c, d) := closest_point
  -- The point is on the bee's path
  (d = bee_path c) ∧
  -- This point is the closest to the flower
  (∀ x y, y = bee_path x → (x - 10)^2 + (y - 12)^2 ≥ (c - 10)^2 + (d - 12)^2) ∧
  -- The sum of coordinates is 18.4
  (c + d = 18.4) := by sorry

end NUMINAMATH_CALUDE_bee_closest_point_to_flower_l793_79332


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_parallel_lines_l793_79330

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_parallel_lines
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : contained_in m β)
  (h3 : parallel_lines l m) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_parallel_lines_l793_79330


namespace NUMINAMATH_CALUDE_complement_A_in_U_l793_79353

def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l793_79353


namespace NUMINAMATH_CALUDE_david_crunches_l793_79343

theorem david_crunches (zachary_crunches : ℕ) (david_less : ℕ) 
  (h1 : zachary_crunches = 17)
  (h2 : david_less = 13) : 
  zachary_crunches - david_less = 4 := by
  sorry

end NUMINAMATH_CALUDE_david_crunches_l793_79343


namespace NUMINAMATH_CALUDE_union_equals_reals_subset_condition_l793_79352

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < 3 + a}
def B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 1}

-- Theorem 1: A ∪ B = ℝ iff -2 ≤ a ≤ -1
theorem union_equals_reals (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 := by
  sorry

-- Theorem 2: A ⊆ B iff a ≤ -4 or a ≥ 1
theorem subset_condition (a : ℝ) : A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_subset_condition_l793_79352


namespace NUMINAMATH_CALUDE_spectators_count_l793_79328

/-- The number of wristbands distributed -/
def total_wristbands : ℕ := 250

/-- The number of wristbands each person received -/
def wristbands_per_person : ℕ := 2

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 125 := by
  sorry

end NUMINAMATH_CALUDE_spectators_count_l793_79328


namespace NUMINAMATH_CALUDE_race_has_six_laps_l793_79360

/-- Represents a cyclist in the race -/
structure Cyclist where
  name : String
  lap_time : ℕ

/-- Represents the race setup -/
structure Race where
  total_laps : ℕ
  vasya : Cyclist
  petya : Cyclist
  kolya : Cyclist

/-- The race conditions are satisfied -/
def race_conditions (r : Race) : Prop :=
  r.vasya.lap_time + 2 = r.petya.lap_time ∧
  r.petya.lap_time + 3 = r.kolya.lap_time ∧
  r.vasya.lap_time * r.total_laps = r.petya.lap_time * (r.total_laps - 1) ∧
  r.vasya.lap_time * r.total_laps = r.kolya.lap_time * (r.total_laps - 2)

/-- The theorem stating that the race has 6 laps -/
theorem race_has_six_laps :
  ∃ (r : Race), race_conditions r ∧ r.total_laps = 6 := by sorry

end NUMINAMATH_CALUDE_race_has_six_laps_l793_79360


namespace NUMINAMATH_CALUDE_count_multiples_eq_42_l793_79361

/-- The number of positive integers less than 201 that are multiples of either 6 or 8, but not both -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => n % 6 = 0 ∨ n % 8 = 0) (Finset.range 201)).card -
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 8 = 0) (Finset.range 201)).card

theorem count_multiples_eq_42 : count_multiples = 42 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_eq_42_l793_79361


namespace NUMINAMATH_CALUDE_computer_cost_l793_79325

theorem computer_cost (C : ℝ) : 
  C + (1/5) * C + 300 = 2100 → C = 1500 := by
  sorry

end NUMINAMATH_CALUDE_computer_cost_l793_79325


namespace NUMINAMATH_CALUDE_product_of_functions_l793_79350

theorem product_of_functions (x : ℝ) (h : x > 0) :
  Real.sqrt (x * (x + 1)) * (1 / Real.sqrt x) = Real.sqrt (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_product_of_functions_l793_79350


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l793_79371

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 4| ≤ 6) ↔ (-2 ≤ x ∧ x ≤ 10) := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l793_79371


namespace NUMINAMATH_CALUDE_perfect_squares_from_equation_l793_79302

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ (a b c : ℕ), (x - y = a^2) ∧ (2 * x + 2 * y + 1 = b^2) ∧ (3 * x + 3 * y + 1 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_from_equation_l793_79302


namespace NUMINAMATH_CALUDE_cone_volume_l793_79324

/-- Given a cone with base area 2π and lateral area 4π, its volume is (2√6/3)π -/
theorem cone_volume (r l h : ℝ) (h_base_area : π * r^2 = 2) (h_lateral_area : π * r * l = 4) 
  (h_height : h^2 = l^2 - r^2) : 
  (1/3) * π * r^2 * h = (2 * Real.sqrt 6 / 3) * π := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_l793_79324


namespace NUMINAMATH_CALUDE_science_competition_accuracy_l793_79338

theorem science_competition_accuracy (correct : ℕ) (wrong : ℕ) (target_accuracy : ℚ) (additional : ℕ) : 
  correct = 30 →
  wrong = 6 →
  target_accuracy = 85/100 →
  (correct + additional) / (correct + wrong + additional) = target_accuracy →
  additional = 4 := by
sorry

end NUMINAMATH_CALUDE_science_competition_accuracy_l793_79338


namespace NUMINAMATH_CALUDE_xyz_value_l793_79378

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30 * Real.rpow 4 (1/3))
  (hxz : x * z = 45 * Real.rpow 4 (1/3))
  (hyz : y * z = 18 * Real.rpow 4 (1/3)) :
  x * y * z = 540 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l793_79378


namespace NUMINAMATH_CALUDE_intersection_of_circles_l793_79320

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- Define the theorem
theorem intersection_of_circles (r : ℝ) (hr : r > 0) :
  (∃! p, p ∈ A ∩ B r) → r = 3 ∨ r = 7 := by
  sorry


end NUMINAMATH_CALUDE_intersection_of_circles_l793_79320


namespace NUMINAMATH_CALUDE_farmer_goats_problem_l793_79383

/-- Represents the number of additional goats needed to make half of the animals goats -/
def additional_goats (cows sheep initial_goats : ℕ) : ℕ :=
  let total := cows + sheep + initial_goats
  (total - 2 * initial_goats)

theorem farmer_goats_problem (cows sheep initial_goats : ℕ) 
  (h_cows : cows = 7)
  (h_sheep : sheep = 8)
  (h_initial_goats : initial_goats = 6) :
  additional_goats cows sheep initial_goats = 9 := by
  sorry

#eval additional_goats 7 8 6

end NUMINAMATH_CALUDE_farmer_goats_problem_l793_79383


namespace NUMINAMATH_CALUDE_smallest_number_l793_79334

-- Define the numbers in their respective bases
def num_base3 : ℕ := 1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0
def num_base6 : ℕ := 2 * 6^2 + 1 * 6^1 + 0 * 6^0
def num_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0
def num_base2 : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Theorem statement
theorem smallest_number :
  num_base2 < num_base3 ∧ num_base2 < num_base6 ∧ num_base2 < num_base4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l793_79334


namespace NUMINAMATH_CALUDE_remainder_theorem_l793_79309

theorem remainder_theorem (N : ℤ) (h : N % 100 = 11) : N % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l793_79309


namespace NUMINAMATH_CALUDE_supremum_of_expression_l793_79318

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ s : ℝ, s = -9/2 ∧ (- 1/(2*a) - 2/b ≤ s) ∧ 
  ∀ t : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → - 1/(2*x) - 2/y ≤ t) → s ≤ t :=
by sorry

end NUMINAMATH_CALUDE_supremum_of_expression_l793_79318


namespace NUMINAMATH_CALUDE_ball_in_ice_l793_79322

theorem ball_in_ice (r : ℝ) (h : r = 16.25) :
  let d := 30  -- diameter of the hole
  let depth := 10  -- depth of the hole
  let x := r - depth  -- distance from center of ball to surface
  d^2 / 4 + x^2 = r^2 := by sorry

end NUMINAMATH_CALUDE_ball_in_ice_l793_79322


namespace NUMINAMATH_CALUDE_mary_saw_90_snakes_l793_79380

/-- The number of breeding balls -/
def num_breeding_balls : ℕ := 5

/-- The number of snakes in each breeding ball -/
def snakes_per_ball : ℕ := 12

/-- The number of additional pairs of snakes -/
def num_additional_pairs : ℕ := 15

/-- The total number of snakes Mary saw -/
def total_snakes : ℕ := num_breeding_balls * snakes_per_ball + 2 * num_additional_pairs

theorem mary_saw_90_snakes : total_snakes = 90 := by
  sorry

end NUMINAMATH_CALUDE_mary_saw_90_snakes_l793_79380


namespace NUMINAMATH_CALUDE_alpha_value_l793_79306

theorem alpha_value (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1))
  (h4 : Real.cos (Complex.arg α) = 1/2) :
  α = (-1 + Real.sqrt 33) / 4 + Complex.I * Real.sqrt (3 * ((-1 + Real.sqrt 33) / 4)^2) ∨
  α = (-1 - Real.sqrt 33) / 4 + Complex.I * Real.sqrt (3 * ((-1 - Real.sqrt 33) / 4)^2) :=
by sorry

end NUMINAMATH_CALUDE_alpha_value_l793_79306


namespace NUMINAMATH_CALUDE_special_sum_of_squares_l793_79355

theorem special_sum_of_squares (n : ℕ) (a b : ℕ) : 
  n ≥ 2 →
  n = a^2 + b^2 →
  (∀ d : ℕ, d > 1 ∧ d ∣ n → a ≤ d) →
  a ∣ n →
  b ∣ n →
  n = 8 ∨ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_special_sum_of_squares_l793_79355


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l793_79370

theorem simplify_trig_expression :
  7 * 8 * (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) /
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) =
  Real.tan (15 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l793_79370


namespace NUMINAMATH_CALUDE_log_sum_equals_six_l793_79385

theorem log_sum_equals_six :
  2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) + 8^(2/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_six_l793_79385


namespace NUMINAMATH_CALUDE_angle_sum_equals_pi_l793_79394

theorem angle_sum_equals_pi (x y : Real) : 
  x > 0 → x < π / 2 → y > 0 → y < π / 2 →
  4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 2 →
  4 * Real.cos (2 * x) + 3 * Real.cos (2 * y) = 1 →
  2 * x + y = π := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equals_pi_l793_79394


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l793_79396

theorem quadratic_equation_equivalence : ∃ (x : ℝ), 16 * x^2 - 32 * x - 512 = 0 ↔ ∃ (x : ℝ), (x - 1)^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l793_79396


namespace NUMINAMATH_CALUDE_inverse_and_determinant_properties_l793_79314

/-- Given a 2x2 matrix A with its inverse, prove properties about A^2 and (A^(-1))^2 -/
theorem inverse_and_determinant_properties (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, 4], ![-2, -2]]) : 
  (A^2)⁻¹ = ![![1, 4], ![-2, 0]] ∧ 
  Matrix.det ((A⁻¹)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_inverse_and_determinant_properties_l793_79314


namespace NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_929_div_30_l793_79345

theorem least_addend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! x : ℕ, x < d ∧ (n + x) % d = 0 :=
by sorry

theorem least_addend_for_929_div_30 :
  (∃! x : ℕ, x < 30 ∧ (929 + x) % 30 = 0) ∧
  (∀ y : ℕ, y < 30 ∧ (929 + y) % 30 = 0 → y = 1) :=
by sorry

end NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_929_div_30_l793_79345


namespace NUMINAMATH_CALUDE_existence_of_special_functions_l793_79316

theorem existence_of_special_functions :
  ∃ (f g : ℝ → ℝ), 
    (∀ x y : ℝ, x < y → (f (g y)) < (f (g x))) ∧
    (∀ x y : ℝ, x < y → (g (f x)) < (g (f y))) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_functions_l793_79316


namespace NUMINAMATH_CALUDE_frisbee_cost_l793_79391

/-- The cost of a frisbee given initial money, kite cost, and remaining money --/
theorem frisbee_cost (initial_money kite_cost remaining_money : ℕ) : 
  initial_money = 78 → 
  kite_cost = 8 → 
  remaining_money = 61 → 
  initial_money - kite_cost - remaining_money = 9 := by
sorry

end NUMINAMATH_CALUDE_frisbee_cost_l793_79391


namespace NUMINAMATH_CALUDE_calculate_product_l793_79346

theorem calculate_product : 
  (0.125 : ℝ)^3 * (-8 : ℝ)^3 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l793_79346


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_for_ax0_eq_b_l793_79339

theorem necessary_sufficient_condition_for_ax0_eq_b 
  (a b x₀ : ℝ) (h : a < 0) :
  (a * x₀ = b) ↔ 
  (∀ x : ℝ, (1/2) * a * x^2 - b * x ≤ (1/2) * a * x₀^2 - b * x₀) :=
by sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_for_ax0_eq_b_l793_79339


namespace NUMINAMATH_CALUDE_complex_second_quadrant_l793_79358

theorem complex_second_quadrant (a : ℝ) : 
  let z : ℂ := a^2 * (1 + Complex.I) - a * (4 + Complex.I) - 6 * Complex.I
  (z.re < 0 ∧ z.im > 0) → (3 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_l793_79358


namespace NUMINAMATH_CALUDE_apartment_count_l793_79321

theorem apartment_count (total_keys : ℕ) (keys_per_apartment : ℕ) (num_complexes : ℕ) :
  total_keys = 72 →
  keys_per_apartment = 3 →
  num_complexes = 2 →
  ∃ (apartments_per_complex : ℕ), 
    apartments_per_complex * keys_per_apartment * num_complexes = total_keys ∧
    apartments_per_complex = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_apartment_count_l793_79321


namespace NUMINAMATH_CALUDE_petyas_calculation_error_l793_79333

theorem petyas_calculation_error (a : ℕ) (h1 : a > 2) : 
  ¬ (∃ (n : ℕ), 
    (a - 2) * (a + 3) - a = n ∧ 
    (∃ (k : ℕ), n.digits 10 = List.replicate 2023 8 ++ List.replicate 2023 3 ++ List.replicate k 0)) :=
by sorry

end NUMINAMATH_CALUDE_petyas_calculation_error_l793_79333


namespace NUMINAMATH_CALUDE_daily_wage_calculation_l793_79367

def days_in_week : ℕ := 7
def weeks : ℕ := 6
def total_earnings : ℕ := 2646

theorem daily_wage_calculation (days_worked : ℕ) (daily_wage : ℚ) 
  (h1 : days_worked = days_in_week * weeks)
  (h2 : daily_wage * days_worked = total_earnings) :
  daily_wage = 63 := by sorry

end NUMINAMATH_CALUDE_daily_wage_calculation_l793_79367


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l793_79344

theorem vector_magnitude_problem (m : ℝ) (a : ℝ × ℝ) :
  m > 0 → a = (m, 4) → ‖a‖ = 5 → m = 3 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l793_79344


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l793_79327

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ≥ 4 * Real.sqrt 3 :=
by sorry

theorem min_value_attainable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l793_79327


namespace NUMINAMATH_CALUDE_smallest_n_correct_l793_79393

/-- The smallest positive integer n for which (x^3 - 1/x^2)^n contains a non-zero constant term -/
def smallest_n : ℕ := 5

/-- Predicate to check if (x^3 - 1/x^2)^n has a non-zero constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (k : ℕ), k ≠ 0 ∧ (3 * n = 5 * k)

theorem smallest_n_correct :
  (has_constant_term smallest_n) ∧
  (∀ m : ℕ, m < smallest_n → ¬(has_constant_term m)) :=
by sorry

#check smallest_n_correct

end NUMINAMATH_CALUDE_smallest_n_correct_l793_79393


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l793_79397

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (2 / 5 : ℂ) + (4 / 9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (2 / 5 : ℂ) - (4 / 9 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l793_79397


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l793_79335

/-- Given a line segment from (3, 7) to (-9, y) with length 15 and y > 0, prove y = 16 -/
theorem line_segment_endpoint (y : ℝ) : 
  (((3 : ℝ) - (-9))^2 + (y - 7)^2 = 15^2) → y > 0 → y = 16 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l793_79335


namespace NUMINAMATH_CALUDE_inequality_proof_l793_79365

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l793_79365


namespace NUMINAMATH_CALUDE_divisibility_problem_l793_79356

theorem divisibility_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h_div : (5 * m + n) ∣ (5 * n + m)) : m ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l793_79356


namespace NUMINAMATH_CALUDE_correct_num_dancers_l793_79362

/-- The number of dancers on the dance team -/
def num_dancers : ℕ := 8

/-- The number of braids per dancer -/
def braids_per_dancer : ℕ := 5

/-- The time in seconds to make one braid -/
def seconds_per_braid : ℕ := 30

/-- The total time in minutes to braid all dancers' hair -/
def total_time_minutes : ℕ := 20

/-- Theorem stating that the number of dancers is correct given the conditions -/
theorem correct_num_dancers :
  num_dancers * braids_per_dancer * seconds_per_braid = total_time_minutes * 60 :=
by sorry

end NUMINAMATH_CALUDE_correct_num_dancers_l793_79362


namespace NUMINAMATH_CALUDE_log_half_decreasing_l793_79323

-- Define the function f(x) = log_(1/2)(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem log_half_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_log_half_decreasing_l793_79323


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l793_79336

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.73264264264

/-- The denominator of the fraction we're looking for -/
def denominator : ℕ := 999900

/-- The numerator of the fraction we're looking for -/
def numerator : ℕ := 732635316

/-- Theorem stating that our decimal is equal to the fraction numerator/denominator -/
theorem decimal_equals_fraction : decimal = (numerator : ℚ) / denominator := by
  sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l793_79336


namespace NUMINAMATH_CALUDE_author_writing_speed_l793_79395

/-- Given an author who writes 25,000 words in 50 hours, prove that their average writing speed is 500 words per hour. -/
theorem author_writing_speed :
  let total_words : ℕ := 25000
  let total_hours : ℕ := 50
  let average_speed : ℕ := total_words / total_hours
  average_speed = 500 :=
by sorry

end NUMINAMATH_CALUDE_author_writing_speed_l793_79395


namespace NUMINAMATH_CALUDE_acute_iff_three_equal_projections_l793_79341

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A direction in a 2D plane, represented by an angle from the positive x-axis --/
def Direction := ℝ

/-- The length of the projection of a triangle onto a given direction --/
def projectionLength (t : Triangle) (d : Direction) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled --/
def isAcute (t : Triangle) : Prop := sorry

/-- Theorem stating that a triangle is acute if and only if it has three equal projections in distinct directions --/
theorem acute_iff_three_equal_projections (t : Triangle) :
  isAcute t ↔
  ∃ (d₁ d₂ d₃ : Direction),
    d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃ ∧
    ∃ (l : ℝ),
      projectionLength t d₁ = l ∧
      projectionLength t d₂ = l ∧
      projectionLength t d₃ = l :=
sorry

end NUMINAMATH_CALUDE_acute_iff_three_equal_projections_l793_79341


namespace NUMINAMATH_CALUDE_inequality_proof_l793_79331

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 1 ≥ x*y + x + y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l793_79331


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l793_79359

theorem binomial_coefficient_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l793_79359


namespace NUMINAMATH_CALUDE_fourth_month_sale_l793_79315

/-- Given the sales of five months and the required average, 
    prove that the sale in the fourth month is 7230 --/
theorem fourth_month_sale 
  (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 6435)
  (h2 : sales 1 = 6927)
  (h3 : sales 2 = 6855)
  (h4 : sales 4 = 6562)
  (h5 : sales 5 = 4991)
  (h_avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 6500) :
  sales 3 = 7230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l793_79315


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_15_eq_600_l793_79381

/-- The count of positive four-digit integers divisible by 15 -/
def count_four_digit_divisible_by_15 : ℕ :=
  (Finset.filter (fun n => n % 15 = 0) (Finset.range 9000)).card

/-- Theorem stating that the count of positive four-digit integers divisible by 15 is 600 -/
theorem count_four_digit_divisible_by_15_eq_600 :
  count_four_digit_divisible_by_15 = 600 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_15_eq_600_l793_79381


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l793_79301

theorem multiplication_subtraction_difference (x : ℝ) (h : x = 10) : 3 * x - (20 - x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l793_79301


namespace NUMINAMATH_CALUDE_eugene_apples_proof_l793_79349

def apples_from_eugene (initial_apples final_apples : ℝ) : ℝ :=
  final_apples - initial_apples

theorem eugene_apples_proof (initial_apples final_apples : ℝ) :
  apples_from_eugene initial_apples final_apples =
  final_apples - initial_apples :=
by
  sorry

#eval apples_from_eugene 20.0 27.0

end NUMINAMATH_CALUDE_eugene_apples_proof_l793_79349


namespace NUMINAMATH_CALUDE_cubic_polynomial_remainder_l793_79307

/-- A cubic polynomial of the form ax³ - 6x² + bx - 5 -/
def f (a b x : ℝ) : ℝ := a * x^3 - 6 * x^2 + b * x - 5

theorem cubic_polynomial_remainder (a b : ℝ) :
  (f a b 1 = -5) ∧ (f a b (-2) = -53) → a = 7 ∧ b = -7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_remainder_l793_79307


namespace NUMINAMATH_CALUDE_contrapositive_square_equality_l793_79329

theorem contrapositive_square_equality (a b : ℝ) : a^2 ≠ b^2 → a ≠ b := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_square_equality_l793_79329
