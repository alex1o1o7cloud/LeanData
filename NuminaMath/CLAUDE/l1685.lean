import Mathlib

namespace base3_addition_theorem_l1685_168534

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else convert (m / 3) ((m % 3) :: acc)
    convert n []

theorem base3_addition_theorem :
  let a := base3ToDecimal [2]
  let b := base3ToDecimal [0, 2, 1]
  let c := base3ToDecimal [1, 2, 2]
  let d := base3ToDecimal [2, 1, 1, 1]
  let e := base3ToDecimal [2, 2, 0, 1]
  let sum := a + b + c + d + e
  decimalToBase3 sum = [1, 0, 2, 1, 2] := by sorry

end base3_addition_theorem_l1685_168534


namespace sibling_age_difference_l1685_168575

/-- Given three siblings whose ages are in the ratio 3:2:1 and whose total combined age is 90 years,
    the difference between the eldest sibling's age and the youngest sibling's age is 30 years. -/
theorem sibling_age_difference (x : ℝ) (h1 : 3*x + 2*x + x = 90) : 3*x - x = 30 := by
  sorry

end sibling_age_difference_l1685_168575


namespace skips_per_meter_l1685_168527

theorem skips_per_meter
  (p q r s t u : ℝ)
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (skip_jump : p * q⁻¹ = 1)
  (jump_foot : r * s⁻¹ = 1)
  (foot_meter : t * u⁻¹ = 1) :
  (t * r * p) * (u * s * q)⁻¹ = 1 := by
sorry

end skips_per_meter_l1685_168527


namespace festival_sunny_days_l1685_168500

def probability_exactly_two_sunny (n : ℕ) (p : ℝ) : ℝ :=
  (n.choose 2 : ℝ) * (1 - p)^2 * p^(n - 2)

theorem festival_sunny_days :
  probability_exactly_two_sunny 5 0.6 = 216 / 625 := by
  sorry

end festival_sunny_days_l1685_168500


namespace rectangle_area_rectangle_area_proof_l1685_168598

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 6
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1296 10 = 60 := by
  sorry

end rectangle_area_rectangle_area_proof_l1685_168598


namespace tracy_candies_l1685_168515

theorem tracy_candies : ∃ (initial : ℕ) (brother_took : ℕ), 
  initial % 20 = 0 ∧ 
  1 ≤ brother_took ∧ 
  brother_took ≤ 6 ∧
  (3 * initial) / 5 - 40 - brother_took = 4 ∧
  initial = 80 := by sorry

end tracy_candies_l1685_168515


namespace contractor_daily_wage_l1685_168576

/-- Represents the contractor's payment scenario --/
structure ContractorPayment where
  totalDays : ℕ
  absentDays : ℕ
  finePerDay : ℚ
  totalPayment : ℚ

/-- Calculates the daily wage given the contractor's payment scenario --/
def calculateDailyWage (c : ContractorPayment) : ℚ :=
  (c.totalPayment + c.finePerDay * c.absentDays) / (c.totalDays - c.absentDays)

/-- Theorem stating that the daily wage is 25 rupees given the problem conditions --/
theorem contractor_daily_wage :
  let c : ContractorPayment := {
    totalDays := 30,
    absentDays := 10,
    finePerDay := 15/2,
    totalPayment := 425
  }
  calculateDailyWage c = 25 := by sorry

end contractor_daily_wage_l1685_168576


namespace line_chart_most_appropriate_for_temperature_over_time_l1685_168501

-- Define the types of charts
inductive ChartType
| PieChart
| LineChart
| BarChart

-- Define the properties of the data
structure DataProperties where
  isTemperature : Bool
  isOverTime : Bool
  needsChangeObservation : Bool

-- Define the function to determine the most appropriate chart type
def mostAppropriateChart (props : DataProperties) : ChartType :=
  if props.isTemperature ∧ props.isOverTime ∧ props.needsChangeObservation then
    ChartType.LineChart
  else
    ChartType.BarChart  -- Default to BarChart for other cases

-- Theorem statement
theorem line_chart_most_appropriate_for_temperature_over_time 
  (props : DataProperties) 
  (h1 : props.isTemperature = true) 
  (h2 : props.isOverTime = true) 
  (h3 : props.needsChangeObservation = true) : 
  mostAppropriateChart props = ChartType.LineChart := by
  sorry


end line_chart_most_appropriate_for_temperature_over_time_l1685_168501


namespace quinary1234_equals_octal302_l1685_168562

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinaryToDecimal (q : ℕ) : ℕ := sorry

/-- Converts a decimal (base-10) number to octal (base-8) --/
def decimalToOctal (d : ℕ) : ℕ := sorry

/-- The quinary representation of 1234 --/
def quinary1234 : ℕ := 1234

/-- The octal representation of 302 --/
def octal302 : ℕ := 302

theorem quinary1234_equals_octal302 : 
  decimalToOctal (quinaryToDecimal quinary1234) = octal302 := by sorry

end quinary1234_equals_octal302_l1685_168562


namespace final_value_one_fourth_l1685_168532

theorem final_value_one_fourth (x : ℝ) : 
  (1 / 4) * ((5 * x + 3) - 1) = (5 * x) / 4 + 1 / 2 := by
  sorry

end final_value_one_fourth_l1685_168532


namespace lcm_of_5_6_8_9_l1685_168523

theorem lcm_of_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end lcm_of_5_6_8_9_l1685_168523


namespace largest_four_digit_odd_sum_19_l1685_168592

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d ≤ 9

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def all_odd_digits (n : ℕ) : Prop :=
  is_odd_digit (n / 1000) ∧
  is_odd_digit ((n / 100) % 10) ∧
  is_odd_digit ((n / 10) % 10) ∧
  is_odd_digit (n % 10)

theorem largest_four_digit_odd_sum_19 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ all_odd_digits n ∧ digit_sum n = 19 →
  n ≤ 9711 :=
sorry

end largest_four_digit_odd_sum_19_l1685_168592


namespace even_increasing_inequality_l1685_168545

-- Define an even function that is increasing on [0,+∞)
def is_even_and_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y)

-- State the theorem
theorem even_increasing_inequality (f : ℝ → ℝ) 
  (h : is_even_and_increasing_on_nonneg f) : 
  f π > f (-3) ∧ f (-3) > f (-2) := by
  sorry

end even_increasing_inequality_l1685_168545


namespace zeros_product_greater_than_e_squared_l1685_168564

/-- Given a function f(x) = ax² - bx + ln x where a and b are real numbers,
    and g(x) = f(x) - ax² = -bx + ln x has two distinct zeros x₁ and x₂,
    prove that x₁x₂ > e² -/
theorem zeros_product_greater_than_e_squared (a b x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : -b * x₁ + Real.log x₁ = 0) 
  (h₃ : -b * x₂ + Real.log x₂ = 0) : 
  x₁ * x₂ > Real.exp 2 := by
  sorry

end zeros_product_greater_than_e_squared_l1685_168564


namespace problem_solution_l1685_168502

theorem problem_solution (m : ℤ) (a : ℝ) : 
  ((-2 : ℝ)^(2*m) = a^(3-m)) → (m = 1) → (a = 2) := by sorry

end problem_solution_l1685_168502


namespace quadratic_equation_solution_l1685_168548

theorem quadratic_equation_solution (h : 108 * (3/4)^2 + 61 = 145 * (3/4) - 7) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 108 * x^2 + 61 = 145 * x - 7 ∧ x = 68/81 := by
  sorry

end quadratic_equation_solution_l1685_168548


namespace only_B_is_difference_of_squares_l1685_168546

-- Define the difference of squares formula
def difference_of_squares (a b : ℝ) : ℝ := a^2 - b^2

-- Define the expressions
def expr_A (x : ℝ) : ℝ := (x - 2) * (x + 1)
def expr_B (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)
def expr_C (x y : ℝ) : ℝ := (x + y) * (-x - y)
def expr_D (x : ℝ) : ℝ := (-x + 1) * (x - 1)

-- Theorem stating that only expr_B fits the difference of squares formula
theorem only_B_is_difference_of_squares :
  (∃ (a b : ℝ), expr_B x y = difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_A x ≠ difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_C x y ≠ difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_D x ≠ difference_of_squares a b) :=
by sorry

end only_B_is_difference_of_squares_l1685_168546


namespace solve_for_y_l1685_168522

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 4*x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  sorry

end solve_for_y_l1685_168522


namespace correct_categorization_l1685_168568

-- Define the given numbers
def numbers : List ℚ := [-2/9, -9, -301, -314/100, 2004, 0, 22/7]

-- Define the sets
def fractions : Set ℚ := {x | x ∈ numbers ∧ x ≠ 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def negative_fractions : Set ℚ := {x | x ∈ fractions ∧ x < 0}
def integers : Set ℚ := {x | x ∈ numbers ∧ ∃ (n : ℤ), x = n}
def positive_integers : Set ℚ := {x | x ∈ integers ∧ x > 0}
def positive_rationals : Set ℚ := {x | x ∈ numbers ∧ x > 0 ∧ ∃ (a b : ℤ), b > 0 ∧ x = a / b}

-- State the theorem
theorem correct_categorization :
  fractions = {-2/9, 22/7} ∧
  negative_fractions = {-2/9} ∧
  integers = {-9, -301, 2004, 0} ∧
  positive_integers = {2004} ∧
  positive_rationals = {2004, 22/7} :=
sorry

end correct_categorization_l1685_168568


namespace gcd_lcm_consecutive_naturals_l1685_168526

theorem gcd_lcm_consecutive_naturals (m : ℕ) (h : m > 0) :
  let n := m + 1
  (Nat.gcd m n = 1) ∧ (Nat.lcm m n = m * n) := by
  sorry

end gcd_lcm_consecutive_naturals_l1685_168526


namespace extra_parts_calculation_l1685_168542

/-- The number of extra parts produced compared to the original plan -/
def extra_parts (initial_rate : ℕ) (initial_days : ℕ) (rate_increase : ℕ) (total_parts : ℕ) : ℕ :=
  let total_days := (total_parts - initial_rate * initial_days) / (initial_rate + rate_increase) + initial_days
  let actual_production := initial_rate * initial_days + (initial_rate + rate_increase) * (total_days - initial_days)
  let planned_production := initial_rate * total_days
  actual_production - planned_production

theorem extra_parts_calculation :
  extra_parts 25 3 5 675 = 100 := by
  sorry

end extra_parts_calculation_l1685_168542


namespace cookie_box_duration_l1685_168524

/-- Given a box of cookies and daily consumption, calculate how many days the box will last -/
def cookiesDuration (totalCookies : ℕ) (oldestSonCookies : ℕ) (youngestSonCookies : ℕ) : ℕ :=
  totalCookies / (oldestSonCookies + youngestSonCookies)

/-- Prove that a box of 54 cookies lasts 9 days when 4 cookies are given to the oldest son
    and 2 cookies are given to the youngest son daily -/
theorem cookie_box_duration :
  cookiesDuration 54 4 2 = 9 := by
  sorry

#eval cookiesDuration 54 4 2

end cookie_box_duration_l1685_168524


namespace fib_formula_l1685_168517

/-- The golden ratio φ, defined as the positive solution of x² = x + 1 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The negative solution φ' of x² = x + 1 -/
noncomputable def φ' : ℝ := (1 - Real.sqrt 5) / 2

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The nth Fibonacci number is given by (φⁿ - φ'ⁿ) / √5 -/
theorem fib_formula (n : ℕ) : (fib n : ℝ) = (φ ^ n - φ' ^ n) / Real.sqrt 5 := by
  sorry

end fib_formula_l1685_168517


namespace combined_efficiency_l1685_168507

-- Define the variables
def ray_efficiency : ℚ := 50
def tom_efficiency : ℚ := 10
def ray_distance : ℚ := 50
def tom_distance : ℚ := 100

-- Define the theorem
theorem combined_efficiency :
  let total_distance := ray_distance + tom_distance
  let ray_fuel := ray_distance / ray_efficiency
  let tom_fuel := tom_distance / tom_efficiency
  let total_fuel := ray_fuel + tom_fuel
  total_distance / total_fuel = 150 / 11 :=
by sorry

end combined_efficiency_l1685_168507


namespace probability_in_standard_deck_l1685_168544

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)
  (hearts : Nat)
  (face_hearts : Nat)

/-- Calculates the probability of drawing a face card, then any heart, then a face card -/
def probability_face_heart_face (d : Deck) : Rat :=
  let first_draw := d.face_cards / d.total_cards
  let second_draw := (d.hearts - d.face_hearts) / (d.total_cards - 1)
  let third_draw := (d.face_cards - 1) / (d.total_cards - 2)
  first_draw * second_draw * third_draw

/-- Standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , face_cards := 12
  , hearts := 13
  , face_hearts := 3 }

theorem probability_in_standard_deck :
  probability_face_heart_face standard_deck = 1320 / 132600 := by
  sorry

end probability_in_standard_deck_l1685_168544


namespace fraction_decimal_difference_l1685_168538

theorem fraction_decimal_difference : 
  2/3 - 0.66666667 = 1/(3 * 10^8) := by sorry

end fraction_decimal_difference_l1685_168538


namespace inequality_proof_l1685_168597

theorem inequality_proof (r p q : ℝ) (hr : r > 0) (hp : p > 0) (hq : q > 0) (h : p^2 * r > q^2 * r) :
  1 > -q/p := by
sorry

end inequality_proof_l1685_168597


namespace jar_weight_theorem_l1685_168581

theorem jar_weight_theorem (jar_weight : ℝ) (full_weight : ℝ) 
  (h1 : jar_weight = 0.1 * full_weight)
  (h2 : 0 < full_weight) :
  let remaining_fraction : ℝ := 0.5555555555555556
  let remaining_weight := jar_weight + remaining_fraction * (full_weight - jar_weight)
  remaining_weight / full_weight = 0.6 := by sorry

end jar_weight_theorem_l1685_168581


namespace g_g_eq_5_has_two_solutions_l1685_168528

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 6

theorem g_g_eq_5_has_two_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g (g x₁) = 5 ∧ g (g x₂) = 5 ∧
  ∀ (x : ℝ), g (g x) = 5 → x = x₁ ∨ x = x₂ :=
sorry

end g_g_eq_5_has_two_solutions_l1685_168528


namespace sequence_A_l1685_168585

theorem sequence_A (a : ℕ → ℕ) : 
  (a 1 = 2) → 
  (∀ n : ℕ, a (n + 1) = a n + n + 1) → 
  (a 20 = 211) :=
by sorry


end sequence_A_l1685_168585


namespace smaller_number_proof_l1685_168530

theorem smaller_number_proof (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) :
  min x y = 18.5 := by
  sorry

end smaller_number_proof_l1685_168530


namespace sum_real_imag_parts_complex_fraction_l1685_168567

theorem sum_real_imag_parts_complex_fraction : ∃ (z : ℂ), 
  z = (3 - 3 * Complex.I) / (1 - Complex.I) ∧ 
  z.re + z.im = 3 :=
sorry

end sum_real_imag_parts_complex_fraction_l1685_168567


namespace children_playing_both_sports_l1685_168510

theorem children_playing_both_sports 
  (total : ℕ) 
  (tennis : ℕ) 
  (squash : ℕ) 
  (neither : ℕ) 
  (h1 : total = 38) 
  (h2 : tennis = 19) 
  (h3 : squash = 21) 
  (h4 : neither = 10) : 
  tennis + squash - (total - neither) = 12 := by
sorry

end children_playing_both_sports_l1685_168510


namespace cost_of_four_birdhouses_l1685_168569

/-- The cost to build a given number of birdhouses -/
def cost_of_birdhouses (num_birdhouses : ℕ) : ℚ :=
  let planks_per_house : ℕ := 7
  let nails_per_house : ℕ := 20
  let cost_per_plank : ℚ := 3
  let cost_per_nail : ℚ := 1/20
  num_birdhouses * (planks_per_house * cost_per_plank + nails_per_house * cost_per_nail)

/-- Theorem stating that the cost to build 4 birdhouses is $88 -/
theorem cost_of_four_birdhouses :
  cost_of_birdhouses 4 = 88 := by
  sorry

end cost_of_four_birdhouses_l1685_168569


namespace quadratic_roots_ratio_l1685_168554

theorem quadratic_roots_ratio (p q α β : ℝ) (h1 : α + β = p) (h2 : α * β = 6) 
  (h3 : x^2 - p*x + q = 0 → x = α ∨ x = β) (h4 : p^2 ≠ 12) : 
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) := by
sorry

end quadratic_roots_ratio_l1685_168554


namespace geometric_sequence_sum_l1685_168572

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * q) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l1685_168572


namespace investment_strategy_optimal_l1685_168558

/-- Represents the maximum interest earned from a two-rate investment strategy --/
def max_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (max_at_rate1 : ℝ) : ℝ :=
  rate1 * max_at_rate1 + rate2 * (total_investment - max_at_rate1)

/-- Theorem stating the maximum interest earned under given conditions --/
theorem investment_strategy_optimal (total_investment : ℝ) (rate1 rate2 : ℝ) (max_at_rate1 : ℝ)
    (h1 : total_investment = 25000)
    (h2 : rate1 = 0.07)
    (h3 : rate2 = 0.12)
    (h4 : max_at_rate1 = 11000) :
    max_interest total_investment rate1 rate2 max_at_rate1 = 2450 := by
  sorry

#eval max_interest 25000 0.07 0.12 11000

end investment_strategy_optimal_l1685_168558


namespace socorro_multiplication_time_l1685_168583

/-- The time spent on multiplication problems each day, given the total training time,
    number of training days, and daily time spent on division problems. -/
def time_on_multiplication (total_hours : ℕ) (days : ℕ) (division_minutes : ℕ) : ℕ :=
  ((total_hours * 60) - (days * division_minutes)) / days

/-- Theorem stating that Socorro spends 10 minutes each day on multiplication problems. -/
theorem socorro_multiplication_time :
  time_on_multiplication 5 10 20 = 10 := by
  sorry

end socorro_multiplication_time_l1685_168583


namespace fraction_simplification_l1685_168503

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hyd : y - 2/x ≠ 0) :
  (2*x - 3/y) / (3*y - 2/x) = (2*x*y - 3) / (3*x*y - 2) := by
  sorry

end fraction_simplification_l1685_168503


namespace weights_division_condition_l1685_168589

/-- A function that checks if a set of weights from 1 to n grams can be divided into three equal mass piles -/
def canDivideWeights (n : ℕ) : Prop :=
  ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧
                         a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
                         (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)

/-- The theorem stating the condition for when weights can be divided into three equal mass piles -/
theorem weights_division_condition (n : ℕ) (h : n > 3) :
  canDivideWeights n ↔ n % 3 = 0 ∨ n % 3 = 2 :=
sorry

end weights_division_condition_l1685_168589


namespace probability_two_even_toys_l1685_168533

def number_of_toys : ℕ := 21
def number_of_even_toys : ℕ := 10

theorem probability_two_even_toys :
  let p := (number_of_even_toys / number_of_toys) * ((number_of_even_toys - 1) / (number_of_toys - 1))
  p = 3 / 14 := by
  sorry

end probability_two_even_toys_l1685_168533


namespace equation_solution_l1685_168590

theorem equation_solution : ∃! x : ℚ, (5 * x / (x + 3) - 3 / (x + 3) = 1 / (x + 3)) ∧ x = 4 / 5 := by
  sorry

end equation_solution_l1685_168590


namespace helen_raisin_cookies_l1685_168509

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := 300

/-- The number of raisin cookies Helen baked the day before yesterday -/
def raisin_cookies_day_before : ℕ := 280

/-- The difference in raisin cookies between yesterday and the day before -/
def raisin_cookie_difference : ℕ := raisin_cookies_yesterday - raisin_cookies_day_before

theorem helen_raisin_cookies : raisin_cookie_difference = 20 := by
  sorry

end helen_raisin_cookies_l1685_168509


namespace D_72_eq_27_l1685_168563

/-- 
D(n) represents the number of ways to write a positive integer n as a product of 
integers greater than 1, where the order matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- 
factorizations(n) represents the list of all valid factorizations of n,
where each factorization is a list of integers greater than 1.
-/
def factorizations (n : ℕ+) : List (List ℕ+) := sorry

/-- 
is_valid_factorization(n, factors) checks if the given list of factors
is a valid factorization of n according to the problem's conditions.
-/
def is_valid_factorization (n : ℕ+) (factors : List ℕ+) : Prop :=
  factors.all (· > 1) ∧ factors.prod = n

theorem D_72_eq_27 : D 72 = 27 := by sorry

end D_72_eq_27_l1685_168563


namespace ratio_difference_l1685_168561

/-- Given two positive numbers in a 7:11 ratio where the smaller is 28, 
    prove that the larger exceeds the smaller by 16. -/
theorem ratio_difference (small large : ℝ) : 
  small > 0 ∧ large > 0 ∧ 
  large / small = 11 / 7 ∧ 
  small = 28 → 
  large - small = 16 := by
sorry

end ratio_difference_l1685_168561


namespace exam_failure_count_l1685_168541

theorem exam_failure_count (total_students : ℕ) (pass_percentage : ℚ) 
  (h1 : total_students = 840)
  (h2 : pass_percentage = 35 / 100) :
  (total_students : ℚ) * (1 - pass_percentage) = 546 := by
  sorry

end exam_failure_count_l1685_168541


namespace cone_volume_increase_l1685_168512

/-- The volume of a cone increases by a factor of 8 when its height and radius are doubled -/
theorem cone_volume_increase (r h V : ℝ) (r' h' V' : ℝ) : 
  V = (1/3) * π * r^2 * h →  -- Original volume
  r' = 2*r →                 -- New radius is doubled
  h' = 2*h →                 -- New height is doubled
  V' = (1/3) * π * r'^2 * h' →  -- New volume
  V' = 8*V := by
sorry


end cone_volume_increase_l1685_168512


namespace bridge_length_l1685_168514

/-- The length of a bridge given train parameters -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 215 :=
by sorry

end bridge_length_l1685_168514


namespace partner_C_profit_share_l1685_168519

-- Define the investment ratios and time periods
def investment_ratio_A : ℚ := 4
def investment_ratio_B : ℚ := 1
def investment_ratio_C : ℚ := 16/3
def investment_ratio_D : ℚ := 2
def investment_ratio_E : ℚ := 2/3

def time_period_A : ℚ := 6
def time_period_B : ℚ := 9
def time_period_C : ℚ := 12
def time_period_D : ℚ := 8
def time_period_E : ℚ := 10

def total_profit : ℚ := 220000

-- Define the theorem
theorem partner_C_profit_share :
  let total_share := investment_ratio_A * time_period_A +
                     investment_ratio_B * time_period_B +
                     investment_ratio_C * time_period_C +
                     investment_ratio_D * time_period_D +
                     investment_ratio_E * time_period_E
  let C_share := (investment_ratio_C * time_period_C / total_share) * total_profit
  ∃ ε > 0, |C_share - 49116.32| < ε :=
by sorry

end partner_C_profit_share_l1685_168519


namespace exam_scores_l1685_168593

theorem exam_scores (total_items : Nat) (marion_score : Nat) (marion_ella_relation : Nat) :
  total_items = 40 →
  marion_score = 24 →
  marion_score = marion_ella_relation + 6 →
  ∃ (ella_score : Nat),
    marion_score = ella_score / 2 + 6 ∧
    ella_score = total_items - 4 :=
by sorry

end exam_scores_l1685_168593


namespace total_tickets_sold_l1685_168586

/-- Proves the total number of tickets sold given ticket prices, total receipts, and number of senior citizen tickets --/
theorem total_tickets_sold (adult_price senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  adult_price = 25 →
  senior_price = 15 →
  total_receipts = 9745 →
  senior_tickets = 348 →
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    adult_tickets + senior_tickets = 529 :=
by sorry

end total_tickets_sold_l1685_168586


namespace quadratic_equation_solution_l1685_168518

theorem quadratic_equation_solution (x : ℝ) : 9 * x^2 - 4 = 0 ↔ x = 2/3 ∨ x = -2/3 := by
  sorry

end quadratic_equation_solution_l1685_168518


namespace leadership_arrangements_l1685_168511

/-- Represents the number of teachers -/
def num_teachers : ℕ := 5

/-- Represents the number of extracurricular groups -/
def num_groups : ℕ := 3

/-- Represents the maximum number of leaders per group -/
def max_leaders_per_group : ℕ := 2

/-- Represents that teachers A and B cannot lead alone -/
def ab_cannot_lead_alone : Prop := True

/-- The number of different leadership arrangements -/
def num_arrangements : ℕ := 54

/-- Theorem stating that the number of different leadership arrangements
    for the given conditions is equal to 54 -/
theorem leadership_arrangements :
  num_teachers = 5 ∧
  num_groups = 3 ∧
  max_leaders_per_group = 2 ∧
  ab_cannot_lead_alone →
  num_arrangements = 54 := by
  sorry

end leadership_arrangements_l1685_168511


namespace ferris_wheel_rides_count_l1685_168547

/-- Represents the number of ferris wheel rides -/
def ferris_wheel_rides : ℕ := sorry

/-- Represents the number of bumper car rides -/
def bumper_car_rides : ℕ := 4

/-- Represents the cost of each ride in tickets -/
def cost_per_ride : ℕ := 7

/-- Represents the total number of tickets used -/
def total_tickets : ℕ := 63

/-- Theorem stating that the number of ferris wheel rides is 5 -/
theorem ferris_wheel_rides_count : ferris_wheel_rides = 5 := by
  sorry

end ferris_wheel_rides_count_l1685_168547


namespace smallest_c_is_22_l1685_168578

/-- A polynomial with three positive integer roots -/
structure PolynomialWithThreeRoots where
  c : ℤ
  d : ℤ
  root1 : ℤ
  root2 : ℤ
  root3 : ℤ
  root1_pos : root1 > 0
  root2_pos : root2 > 0
  root3_pos : root3 > 0
  is_root1 : root1^3 - c*root1^2 + d*root1 - 2310 = 0
  is_root2 : root2^3 - c*root2^2 + d*root2 - 2310 = 0
  is_root3 : root3^3 - c*root3^2 + d*root3 - 2310 = 0

/-- The smallest possible value of c for a polynomial with three positive integer roots -/
def smallest_c : ℤ := 22

/-- Theorem stating that 22 is the smallest possible value of c -/
theorem smallest_c_is_22 (p : PolynomialWithThreeRoots) : p.c ≥ smallest_c := by
  sorry

end smallest_c_is_22_l1685_168578


namespace union_of_A_and_B_l1685_168560

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end union_of_A_and_B_l1685_168560


namespace job_completion_time_l1685_168521

/-- Given workers p and q, where p can complete a job in 4 days and q's daily work rate
    is one-third of p's, prove that p and q working together can complete the job in 3 days. -/
theorem job_completion_time (p q : ℝ) 
    (hp : p = 1 / 4)  -- p's daily work rate
    (hq : q = 1 / 3 * p) : -- q's daily work rate relative to p
    1 / (p + q) = 3 := by
  sorry


end job_completion_time_l1685_168521


namespace computer_price_increase_l1685_168551

theorem computer_price_increase (d : ℝ) : 
  (d * 1.3 = 377) → (2 * d = 580) := by
  sorry

end computer_price_increase_l1685_168551


namespace inequality_solution_l1685_168579

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) ↔ 
  (x < 0 ∨ (1 < x ∧ x < 2) ∨ 2 < x) :=
by sorry

end inequality_solution_l1685_168579


namespace roses_difference_l1685_168536

theorem roses_difference (initial_roses : ℕ) (thrown_away : ℕ) (final_roses : ℕ)
  (h1 : initial_roses = 21)
  (h2 : thrown_away = 34)
  (h3 : final_roses = 15) :
  thrown_away - (thrown_away + final_roses - initial_roses) = 6 := by
  sorry

end roses_difference_l1685_168536


namespace f_quadrants_l1685_168595

/-- A linear function in the Cartesian coordinate system -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Quadrants in the Cartesian coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The set of quadrants a linear function passes through -/
def quadrants_passed (f : LinearFunction) : Set Quadrant :=
  sorry

/-- The specific linear function y = -x - 2 -/
def f : LinearFunction :=
  { slope := -1, intercept := -2 }

/-- Theorem stating which quadrants the function y = -x - 2 passes through -/
theorem f_quadrants :
  quadrants_passed f = {Quadrant.II, Quadrant.III, Quadrant.IV} :=
sorry

end f_quadrants_l1685_168595


namespace yield_prediction_80kg_l1685_168537

/-- Predicts the rice yield based on the amount of fertilizer applied. -/
def predict_yield (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem stating that when 80 kg of fertilizer is applied, the predicted yield is 650 kg. -/
theorem yield_prediction_80kg : predict_yield 80 = 650 := by
  sorry

end yield_prediction_80kg_l1685_168537


namespace sandys_initial_fish_count_l1685_168587

theorem sandys_initial_fish_count (initial_fish current_fish bought_fish : ℕ) : 
  current_fish = initial_fish + bought_fish →
  current_fish = 32 →
  bought_fish = 6 →
  initial_fish = 26 := by
sorry

end sandys_initial_fish_count_l1685_168587


namespace quadratic_two_distinct_roots_l1685_168520

theorem quadratic_two_distinct_roots : ∃ x y : ℝ, x ≠ y ∧ 
  x^2 + 2*x - 5 = 0 ∧ y^2 + 2*y - 5 = 0 := by
  sorry

end quadratic_two_distinct_roots_l1685_168520


namespace log_y_equals_negative_two_l1685_168549

theorem log_y_equals_negative_two (y : ℝ) : 
  y = (Real.log 3 / Real.log 27) ^ (Real.log 81 / Real.log 3) → 
  Real.log y / Real.log 9 = -2 := by
  sorry

end log_y_equals_negative_two_l1685_168549


namespace committee_formation_count_l1685_168505

/-- The number of ways to choose a committee under given conditions -/
def committee_formations (total_boys : ℕ) (total_girls : ℕ) (committee_size : ℕ) 
  (boys_with_event_planning : ℕ) (girls_with_leadership : ℕ) : ℕ :=
  let boys_to_choose := committee_size / 2
  let girls_to_choose := committee_size / 2
  let remaining_boys := total_boys - boys_with_event_planning
  let remaining_girls := total_girls - girls_with_leadership
  (Nat.choose remaining_boys (boys_to_choose - 1)) * 
  (Nat.choose remaining_girls (girls_to_choose - 1))

/-- Theorem stating the number of ways to form the committee -/
theorem committee_formation_count :
  committee_formations 8 6 8 1 1 = 350 :=
by sorry

end committee_formation_count_l1685_168505


namespace fifteenth_even_multiple_of_5_l1685_168582

/-- A function that returns the nth positive integer that is both even and a multiple of 5 -/
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

/-- The 15th positive integer that is both even and a multiple of 5 is 150 -/
theorem fifteenth_even_multiple_of_5 : evenMultipleOf5 15 = 150 := by
  sorry

end fifteenth_even_multiple_of_5_l1685_168582


namespace total_time_circling_island_l1685_168506

/-- The time in minutes to navigate around the island once. -/
def time_per_round : ℕ := 30

/-- The number of rounds completed on Saturday. -/
def saturday_rounds : ℕ := 11

/-- The number of rounds completed on Sunday. -/
def sunday_rounds : ℕ := 15

/-- The total time spent circling the island over the weekend. -/
theorem total_time_circling_island : 
  (saturday_rounds + sunday_rounds) * time_per_round = 780 := by sorry

end total_time_circling_island_l1685_168506


namespace polyhedron_ball_covering_inequality_l1685_168553

/-- A non-degenerate polyhedron -/
structure Polyhedron where
  nondegenerate : Bool

/-- A collection of balls covering a polyhedron -/
structure BallCovering (P : Polyhedron) where
  n : ℕ
  V : ℝ
  covers_surface : Bool

/-- Theorem: For any non-degenerate polyhedron, there exists a positive constant
    such that any ball covering satisfies the given inequality -/
theorem polyhedron_ball_covering_inequality (P : Polyhedron) 
    (h : P.nondegenerate = true) :
    ∃ c : ℝ, c > 0 ∧ 
    ∀ (B : BallCovering P), B.covers_surface → B.n > c / (B.V ^ 2) := by
  sorry

end polyhedron_ball_covering_inequality_l1685_168553


namespace sqrt_product_equality_l1685_168557

theorem sqrt_product_equality : 
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1685_168557


namespace prob_even_sum_is_half_l1685_168513

/-- Represents a wheel with a given number of sections and even sections -/
structure Wheel where
  total_sections : ℕ
  even_sections : ℕ

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def prob_even_sum (wheel1 wheel2 : Wheel) : ℚ :=
  let p1_even := wheel1.even_sections / wheel1.total_sections
  let p2_even := wheel2.even_sections / wheel2.total_sections
  p1_even * p2_even + (1 - p1_even) * (1 - p2_even)

/-- The main theorem stating that the probability of getting an even sum
    when spinning the two given wheels is 1/2 -/
theorem prob_even_sum_is_half :
  let wheel1 : Wheel := { total_sections := 5, even_sections := 2 }
  let wheel2 : Wheel := { total_sections := 4, even_sections := 2 }
  prob_even_sum wheel1 wheel2 = 1/2 := by
  sorry


end prob_even_sum_is_half_l1685_168513


namespace opposite_of_negative_seven_l1685_168559

theorem opposite_of_negative_seven : 
  (-(- 7 : ℤ)) = (7 : ℤ) := by sorry

end opposite_of_negative_seven_l1685_168559


namespace unique_solution_quadratic_l1685_168525

theorem unique_solution_quadratic (k : ℝ) :
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ (k = 0 ∨ k = 1) := by
  sorry

end unique_solution_quadratic_l1685_168525


namespace parallel_vectors_m_value_l1685_168599

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2*m + 1, -1/2)
  let b : ℝ × ℝ := (2*m, 1)
  are_parallel a b → m = -1/3 := by
sorry

end parallel_vectors_m_value_l1685_168599


namespace parabola_c_value_l1685_168580

/-- A parabola with equation y = ax^2 + bx + c, vertex (-1, -2), and passing through (-2, -1) has c = -1 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a*x^2 + b*x + c) →  -- Equation of the parabola
  (-2 = a*(-1)^2 + b*(-1) + c) →      -- Vertex condition
  (-1 = a*(-2)^2 + b*(-2) + c) →      -- Point condition
  c = -1 := by
sorry


end parabola_c_value_l1685_168580


namespace perimeter_equality_l1685_168540

/-- The perimeter of a rectangle -/
def rectangle_perimeter (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height)

/-- The perimeter of a figure composed of two rectangles sharing one edge -/
def composite_perimeter (width1 : ℕ) (height1 : ℕ) (width2 : ℕ) (height2 : ℕ) (shared_edge : ℕ) : ℕ :=
  rectangle_perimeter width1 height1 + rectangle_perimeter width2 height2 - 2 * shared_edge

theorem perimeter_equality :
  rectangle_perimeter 4 3 = composite_perimeter 2 3 3 2 3 := by
  sorry

#eval rectangle_perimeter 4 3
#eval composite_perimeter 2 3 3 2 3

end perimeter_equality_l1685_168540


namespace percentage_grade_c_l1685_168565

def scores : List Nat := [49, 58, 65, 77, 84, 70, 88, 94, 55, 82, 60, 86, 68, 74, 99, 81, 73, 79, 53, 91]

def is_grade_c (score : Nat) : Bool :=
  78 ≤ score ∧ score ≤ 86

def count_grade_c (scores : List Nat) : Nat :=
  scores.filter is_grade_c |>.length

theorem percentage_grade_c : 
  (count_grade_c scores : Rat) / (scores.length : Rat) * 100 = 25 := by
  sorry

end percentage_grade_c_l1685_168565


namespace colorings_count_l1685_168535

/-- Represents the number of colors available --/
def num_colors : ℕ := 4

/-- Represents the number of triangles in the configuration --/
def num_triangles : ℕ := 4

/-- Represents the number of ways to color the first triangle --/
def first_triangle_colorings : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Represents the number of ways to color each subsequent triangle --/
def subsequent_triangle_colorings : ℕ := (num_colors - 1) * (num_colors - 2)

/-- The total number of possible colorings for the entire configuration --/
def total_colorings : ℕ := first_triangle_colorings * subsequent_triangle_colorings^(num_triangles - 1)

theorem colorings_count :
  total_colorings = 5184 :=
sorry

end colorings_count_l1685_168535


namespace derivative_of_linear_function_l1685_168566

theorem derivative_of_linear_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => 2 * x
  (deriv y) x = 2 := by
  sorry

end derivative_of_linear_function_l1685_168566


namespace tank_volume_ratio_l1685_168596

theorem tank_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end tank_volume_ratio_l1685_168596


namespace parallel_vectors_cos_identity_l1685_168591

/-- Given two vectors a and b, where a is parallel to b, prove that cos(π/2 + α) = -1/3 -/
theorem parallel_vectors_cos_identity (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α))
  (hb : b = (Real.cos α, 1))
  (hparallel : ∃ (k : ℝ), a = k • b) :
  Real.cos (π/2 + α) = -1/3 := by
  sorry

end parallel_vectors_cos_identity_l1685_168591


namespace no_base_for_630_four_digits_odd_final_l1685_168516

theorem no_base_for_630_four_digits_odd_final : ¬ ∃ b : ℕ, 
  2 ≤ b ∧ 
  b^3 ≤ 630 ∧ 
  630 < b^4 ∧ 
  (630 % b) % 2 = 1 :=
by sorry

end no_base_for_630_four_digits_odd_final_l1685_168516


namespace cost_per_person_l1685_168552

def total_cost : ℚ := 13500
def num_friends : ℕ := 15

theorem cost_per_person :
  total_cost / num_friends = 900 :=
sorry

end cost_per_person_l1685_168552


namespace ratio_of_repeating_decimals_l1685_168543

/-- Represents a repeating decimal where the decimal part repeats infinitely -/
def RepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / (999 : ℚ)

/-- The fraction 0.888... -/
def a : ℚ := RepeatingDecimal 0 888

/-- The fraction 1.222... -/
def b : ℚ := RepeatingDecimal 1 222

/-- Theorem stating that the ratio of 0.888... to 1.222... is equal to 8/11 -/
theorem ratio_of_repeating_decimals : a / b = 8 / 11 := by
  sorry

end ratio_of_repeating_decimals_l1685_168543


namespace alissa_presents_l1685_168573

/-- Given that Ethan has 31 presents and Alissa has 22 more presents than Ethan,
    prove that Alissa has 53 presents. -/
theorem alissa_presents (ethan_presents : ℕ) (alissa_extra : ℕ) :
  ethan_presents = 31 → alissa_extra = 22 → ethan_presents + alissa_extra = 53 :=
by sorry

end alissa_presents_l1685_168573


namespace partner_a_capital_l1685_168570

/-- Represents the partnership structure and profit distribution --/
structure Partnership where
  total_profit : ℝ
  a_share : ℝ
  b_share : ℝ
  c_share : ℝ
  a_share_def : a_share = (2/3) * total_profit
  bc_share_def : b_share = c_share
  bc_share_sum : b_share + c_share = (1/3) * total_profit

/-- Represents the change in profit rate and its effect on partner a's income --/
structure ProfitChange where
  initial_rate : ℝ
  final_rate : ℝ
  a_income_increase : ℝ
  rate_def : final_rate - initial_rate = 0.02
  initial_rate_def : initial_rate = 0.05
  income_increase_def : a_income_increase = 200

/-- The main theorem stating the capital of partner a --/
theorem partner_a_capital 
  (p : Partnership) 
  (pc : ProfitChange) : 
  ∃ (capital_a : ℝ), capital_a = 300000 := by
  sorry

end partner_a_capital_l1685_168570


namespace chef_potatoes_per_week_l1685_168584

/-- Calculates the total number of potatoes used by a chef in one week -/
def total_potatoes_per_week (lunch_potatoes : ℕ) (work_days : ℕ) : ℕ :=
  let dinner_potatoes := 2 * lunch_potatoes
  let lunch_total := lunch_potatoes * work_days
  let dinner_total := dinner_potatoes * work_days
  lunch_total + dinner_total

/-- Proves that the chef uses 90 potatoes in one week -/
theorem chef_potatoes_per_week :
  total_potatoes_per_week 5 6 = 90 :=
by
  sorry

#eval total_potatoes_per_week 5 6

end chef_potatoes_per_week_l1685_168584


namespace shortest_distance_on_specific_cone_l1685_168577

/-- Represents a right circular cone -/
structure RightCircularCone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ  -- Angle from a fixed reference line on the cone surface

/-- Calculates the shortest distance between two points on the surface of a cone -/
def shortestDistanceOnCone (cone : RightCircularCone) (p1 p2 : ConePoint) : ℝ := sorry

/-- Theorem stating the shortest distance between two specific points on a cone -/
theorem shortest_distance_on_specific_cone :
  let cone : RightCircularCone := ⟨450, 300 * Real.sqrt 3⟩
  let p1 : ConePoint := ⟨200, 0⟩
  let p2 : ConePoint := ⟨300 * Real.sqrt 3, π⟩
  shortestDistanceOnCone cone p1 p2 = 200 + 300 * Real.sqrt 3 := by sorry

end shortest_distance_on_specific_cone_l1685_168577


namespace semicircle_radius_l1685_168556

theorem semicircle_radius (width length : ℝ) (h1 : width = 3) (h2 : length = 8) :
  let rectangle_area := width * length
  let semicircle_radius := Real.sqrt (2 * rectangle_area / Real.pi)
  semicircle_radius = Real.sqrt (48 / Real.pi) := by
  sorry

end semicircle_radius_l1685_168556


namespace max_profit_at_100_l1685_168529

-- Define the cost function
def C (x : ℕ) : ℚ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

-- Define the profit function
def L (x : ℕ) : ℚ :=
  if x < 80 then -(1/3) * x^2 + 40 * x - 250
  else 1200 - (x + 10000 / x)

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x : ℕ, x > 0 → L x ≤ 1000 ∧ L 100 = 1000 :=
sorry

end max_profit_at_100_l1685_168529


namespace wall_height_breadth_ratio_l1685_168594

/-- Proves that the ratio of height to breadth of a wall with given dimensions is 5:1 -/
theorem wall_height_breadth_ratio :
  ∀ (h b l : ℝ),
    b = 0.4 →
    l = 8 * h →
    ∃ (n : ℝ), h = n * b →
    l * b * h = 12.8 →
    n = 5 := by
  sorry

end wall_height_breadth_ratio_l1685_168594


namespace symmetry_of_shifted_even_function_l1685_168539

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a vertical line
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  IsEven (fun x ↦ f (x + 1)) → SymmetricAboutLine f 1 := by
  sorry


end symmetry_of_shifted_even_function_l1685_168539


namespace inequality_solution_l1685_168508

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) ≥ 2 * x ↔ x ≤ 0 ∨ (1 < x ∧ x ≤ 3/2) :=
by sorry

end inequality_solution_l1685_168508


namespace average_of_multiples_of_four_l1685_168555

theorem average_of_multiples_of_four : 
  let numbers := (Finset.range 33).filter (fun n => (n + 8) % 4 = 0)
  let sum := numbers.sum (fun n => n + 8)
  let count := numbers.card
  sum / count = 22 := by sorry

end average_of_multiples_of_four_l1685_168555


namespace furniture_fraction_l1685_168550

theorem furniture_fraction (original_savings tv_cost : ℚ) 
  (h1 : original_savings = 500)
  (h2 : tv_cost = 100) : 
  (original_savings - tv_cost) / original_savings = 4 / 5 := by
  sorry

end furniture_fraction_l1685_168550


namespace isosceles_triangle_perimeter_l1685_168504

/-- The roots of the quadratic equation x^2 - 7x + 12 = 0 -/
def roots : Set ℝ := {x : ℝ | x^2 - 7*x + 12 = 0}

/-- An isosceles triangle with two sides from the roots set -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 ∈ roots) ∨ (side1 = side3 ∧ side2 ∈ roots) ∨ (side2 = side3 ∧ side1 ∈ roots)
  sides_from_roots : {side1, side2, side3} ∩ roots = {side1, side2} ∨ {side1, side2, side3} ∩ roots = {side1, side3} ∨ {side1, side2, side3} ∩ roots = {side2, side3}

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem: The perimeter of the isosceles triangle is either 10 or 11 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end isosceles_triangle_perimeter_l1685_168504


namespace bill_drew_140_lines_l1685_168588

/-- The number of lines drawn for a given shape --/
def lines_for_shape (num_shapes : ℕ) (sides_per_shape : ℕ) : ℕ :=
  num_shapes * sides_per_shape

/-- The total number of lines drawn by Bill --/
def total_lines : ℕ :=
  let triangles := lines_for_shape 12 3
  let squares := lines_for_shape 8 4
  let pentagons := lines_for_shape 4 5
  let hexagons := lines_for_shape 6 6
  let octagons := lines_for_shape 2 8
  triangles + squares + pentagons + hexagons + octagons

theorem bill_drew_140_lines : total_lines = 140 := by
  sorry

end bill_drew_140_lines_l1685_168588


namespace largest_quantity_l1685_168571

theorem largest_quantity (a b c d : ℝ) 
  (h : a + 1 = b - 2 ∧ a + 1 = c + 3 ∧ a + 1 = d - 4) : 
  d = max a (max b c) ∧ d ≥ a ∧ d ≥ b ∧ d ≥ c := by
  sorry

end largest_quantity_l1685_168571


namespace two_workers_better_l1685_168574

/-- Represents the number of production lines -/
def num_lines : ℕ := 3

/-- Represents the probability of failure for each production line -/
def failure_prob : ℚ := 1/3

/-- Represents the monthly salary of each maintenance worker -/
def worker_salary : ℕ := 10000

/-- Represents the monthly profit of a production line with no failure -/
def profit_no_failure : ℕ := 120000

/-- Represents the monthly profit of a production line with failure and repair -/
def profit_with_repair : ℕ := 80000

/-- Represents the monthly profit of a production line with failure and no repair -/
def profit_no_repair : ℕ := 0

/-- Calculates the expected profit with a given number of maintenance workers -/
def expected_profit (num_workers : ℕ) : ℚ :=
  sorry

/-- Theorem stating that the expected profit with 2 workers is greater than with 1 worker -/
theorem two_workers_better :
  expected_profit 2 > expected_profit 1 := by
  sorry

end two_workers_better_l1685_168574


namespace plane_equation_proof_l1685_168531

def plane_equation (w : ℝ × ℝ × ℝ) (s t : ℝ) : Prop :=
  w = (2 + 2*s - 3*t, 4 - 2*s, 1 - s + 3*t)

theorem plane_equation_proof :
  ∃ (A B C D : ℤ),
    (∀ x y z : ℝ, (∃ s t : ℝ, plane_equation (x, y, z) s t) ↔ A * x + B * y + C * z + D = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 2 ∧ B = -1 ∧ C = 2 ∧ D = -2 :=
by sorry

end plane_equation_proof_l1685_168531
