import Mathlib

namespace one_four_one_not_reappear_l1925_192540

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

def next_numbers (n : ℕ) : Set ℕ :=
  {n + digit_product n, n - digit_product n}

def reachable_numbers (start : ℕ) : Set ℕ :=
  {n | ∃ (seq : ℕ → ℕ), seq 0 = start ∧ ∀ i, seq (i + 1) ∈ next_numbers (seq i)}

theorem one_four_one_not_reappear : 141 ∉ reachable_numbers 141 \ {141} := by
  sorry

end one_four_one_not_reappear_l1925_192540


namespace negation_of_existence_l1925_192594

theorem negation_of_existence (p : Prop) :
  (¬∃ x₀ : ℝ, x₀ ≥ 1 ∧ x₀^2 - x₀ < 0) ↔ (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) := by
  sorry

end negation_of_existence_l1925_192594


namespace building_occupancy_l1925_192597

/-- Given a building with a certain number of stories, apartments per floor, and people per apartment,
    calculate the total number of people housed in the building. -/
def total_people (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem stating that a 25-story building with 4 apartments per floor and 2 people per apartment
    houses 200 people in total. -/
theorem building_occupancy :
  total_people 25 4 2 = 200 := by
  sorry

end building_occupancy_l1925_192597


namespace add_decimals_l1925_192520

theorem add_decimals : (124.75 : ℝ) + 0.35 = 125.10 := by sorry

end add_decimals_l1925_192520


namespace daily_earnings_l1925_192503

/-- Calculates the daily earnings of a person who works every day, given their earnings over a 4-week period. -/
theorem daily_earnings (total_earnings : ℚ) (h : total_earnings = 1960) : 
  total_earnings / (4 * 7) = 70 := by
  sorry

end daily_earnings_l1925_192503


namespace r_value_when_n_is_3_l1925_192515

/-- Given n = 3, prove that r = 177136, where r = 3^s - s and s = 2^n + n -/
theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + n
  let r : ℕ := 3^s - s
  r = 177136 := by
sorry

end r_value_when_n_is_3_l1925_192515


namespace adult_tickets_sold_l1925_192549

/-- Theorem: Number of adult tickets sold in a movie theater --/
theorem adult_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) : 
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧ 
    adult_tickets = 500 := by
  sorry

end adult_tickets_sold_l1925_192549


namespace playground_children_count_l1925_192560

theorem playground_children_count :
  let boys : ℕ := 27
  let girls : ℕ := 35
  boys + girls = 62 := by sorry

end playground_children_count_l1925_192560


namespace cannot_row_against_stream_l1925_192575

theorem cannot_row_against_stream (rate_still : ℝ) (speed_with_stream : ℝ) :
  rate_still = 1 →
  speed_with_stream = 6 →
  let stream_speed := speed_with_stream - rate_still
  stream_speed > rate_still →
  ¬∃ (speed_against_stream : ℝ), speed_against_stream > 0 ∧ speed_against_stream = rate_still - stream_speed :=
by
  sorry

end cannot_row_against_stream_l1925_192575


namespace polynomial_divisibility_l1925_192544

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ n : ℤ, (d * n^3 + c * n^2 + b * n + a) % 5 = 0 :=
by sorry

end polynomial_divisibility_l1925_192544


namespace bus_children_difference_l1925_192580

theorem bus_children_difference (initial : ℕ) (got_off : ℕ) (final : ℕ) : 
  initial = 5 → got_off = 63 → final = 14 → 
  ∃ (got_on : ℕ), got_on - got_off = 9 ∧ initial - got_off + got_on = final :=
sorry

end bus_children_difference_l1925_192580


namespace sin_alpha_for_point_l1925_192527

/-- If the terminal side of angle α passes through the point (-4, 3), then sin α = 3/5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) → 
  Real.sin α = 3/5 := by
sorry

end sin_alpha_for_point_l1925_192527


namespace distance_between_cities_l1925_192501

def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem distance_between_cities (D : ℝ) : D = 330 :=
  let train1_speed : ℝ := 60
  let train1_time : ℝ := 3
  let train2_speed : ℝ := 75
  let train2_time : ℝ := 2
  let train1_distance := train_distance train1_speed train1_time
  let train2_distance := train_distance train2_speed train2_time
  have h1 : D = train1_distance + train2_distance := by sorry
  have h2 : train1_distance = 180 := by sorry
  have h3 : train2_distance = 150 := by sorry
  sorry

end distance_between_cities_l1925_192501


namespace polynomial_division_theorem_l1925_192539

theorem polynomial_division_theorem (x : ℝ) :
  let dividend := x^5 - 20*x^3 + 15*x^2 - 18*x + 12
  let divisor := x - 2
  let quotient := x^4 + 2*x^3 - 16*x^2 - 17*x - 52
  let remainder := -92
  dividend = divisor * quotient + remainder := by sorry

end polynomial_division_theorem_l1925_192539


namespace units_digit_of_power_sum_divided_l1925_192552

/-- The units digit of (4^503 + 6^503) / 10 is 1 -/
theorem units_digit_of_power_sum_divided : ∃ n : ℕ, (4^503 + 6^503) / 10 = 10 * n + 1 := by
  sorry

end units_digit_of_power_sum_divided_l1925_192552


namespace negation_of_universal_proposition_l1925_192538

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > x) ↔ (∃ x : ℝ, x^2 ≤ x) := by
  sorry

end negation_of_universal_proposition_l1925_192538


namespace card_collection_difference_l1925_192599

/-- Represents the number of cards each person has -/
structure CardCollection where
  heike : ℕ
  anton : ℕ
  ann : ℕ
  bertrand : ℕ
  carla : ℕ
  desmond : ℕ

/-- The conditions of the card collection problem -/
def card_collection_conditions (c : CardCollection) : Prop :=
  c.anton = 3 * c.heike ∧
  c.ann = 6 * c.heike ∧
  c.bertrand = 2 * c.heike ∧
  c.carla = 4 * c.heike ∧
  c.desmond = 8 * c.heike ∧
  c.ann = 60

/-- The theorem stating the difference between the highest and lowest number of cards -/
theorem card_collection_difference (c : CardCollection) 
  (h : card_collection_conditions c) : 
  max c.anton (max c.ann (max c.bertrand (max c.carla c.desmond))) - 
  min c.heike (min c.anton (min c.ann (min c.bertrand (min c.carla c.desmond)))) = 70 := by
  sorry

end card_collection_difference_l1925_192599


namespace ram_original_price_l1925_192505

/-- Represents the price change of RAM due to market conditions --/
def ram_price_change (original_price : ℝ) : Prop :=
  let increased_price := original_price * 1.3
  let final_price := increased_price * 0.8
  final_price = 52

/-- Theorem stating that the original price of RAM was $50 --/
theorem ram_original_price : ∃ (price : ℝ), ram_price_change price ∧ price = 50 := by
  sorry

end ram_original_price_l1925_192505


namespace train_speed_conversion_l1925_192561

theorem train_speed_conversion (speed_kmph : ℝ) (speed_ms : ℝ) : 
  speed_kmph = 216 → speed_ms = 60 → speed_kmph * 1000 / 3600 = speed_ms := by
  sorry

end train_speed_conversion_l1925_192561


namespace rachel_milk_consumption_l1925_192581

theorem rachel_milk_consumption (don_milk : ℚ) (rachel_fraction : ℚ) :
  don_milk = 3 / 7 →
  rachel_fraction = 4 / 5 →
  rachel_fraction * don_milk = 12 / 35 :=
by sorry

end rachel_milk_consumption_l1925_192581


namespace angle_sum_bounds_l1925_192548

theorem angle_sum_bounds (x y z : Real) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hz : 0 < z ∧ z < π/2) 
  (h : Real.cos x ^ 2 + Real.cos y ^ 2 + Real.cos z ^ 2 = 1) : 
  3 * π / 4 < x + y + z ∧ x + y + z < π := by
sorry

end angle_sum_bounds_l1925_192548


namespace escalator_theorem_l1925_192551

def escalator_problem (stationary_time walking_time : ℝ) : Prop :=
  let s := 1 / stationary_time -- Clea's walking speed
  let d := 1 -- normalized distance of the escalator
  let v := d / walking_time - s -- speed of the escalator
  (d / v) = 50

theorem escalator_theorem : 
  escalator_problem 75 30 := by sorry

end escalator_theorem_l1925_192551


namespace expected_total_rainfall_l1925_192562

/-- Represents the weather conditions for each day -/
structure DailyWeather where
  sun_prob : Real
  rain_3in_prob : Real
  rain_8in_prob : Real

/-- Calculates the expected rainfall for a single day -/
def expected_daily_rainfall (w : DailyWeather) : Real :=
  w.sun_prob * 0 + w.rain_3in_prob * 3 + w.rain_8in_prob * 8

/-- The weather forecast for the week -/
def weather_forecast : DailyWeather :=
  { sun_prob := 0.3
  , rain_3in_prob := 0.4
  , rain_8in_prob := 0.3 }

/-- The number of days in the forecast -/
def num_days : Nat := 5

/-- Theorem: The expected total rainfall for the week is 18 inches -/
theorem expected_total_rainfall :
  (expected_daily_rainfall weather_forecast) * num_days = 18 := by
  sorry

end expected_total_rainfall_l1925_192562


namespace shorts_folded_l1925_192582

/-- Given the following:
  * There are 20 shirts and 8 pairs of shorts in total
  * 12 shirts are folded
  * 11 pieces of clothing remain to be folded
  Prove that 5 pairs of shorts were folded -/
theorem shorts_folded (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (remaining_to_fold : ℕ) : ℕ :=
  by
  have h1 : total_shirts = 20 := by sorry
  have h2 : total_shorts = 8 := by sorry
  have h3 : folded_shirts = 12 := by sorry
  have h4 : remaining_to_fold = 11 := by sorry
  exact 5

end shorts_folded_l1925_192582


namespace houses_not_yellow_l1925_192593

theorem houses_not_yellow (green yellow red : ℕ) : 
  green = 3 * yellow →
  yellow + 40 = red →
  green = 90 →
  green + red = 160 :=
by sorry

end houses_not_yellow_l1925_192593


namespace ratio_calculation_l1925_192592

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 := by
  sorry

end ratio_calculation_l1925_192592


namespace abs_sum_fraction_inequality_l1925_192530

theorem abs_sum_fraction_inequality (a b : ℝ) :
  |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
  sorry

end abs_sum_fraction_inequality_l1925_192530


namespace complex_fraction_equals_neg_one_minus_i_l1925_192528

theorem complex_fraction_equals_neg_one_minus_i :
  let i : ℂ := Complex.I
  (1 + i)^3 / (1 - i)^2 = -1 - i :=
by sorry

end complex_fraction_equals_neg_one_minus_i_l1925_192528


namespace min_sum_with_reciprocal_constraint_l1925_192578

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end min_sum_with_reciprocal_constraint_l1925_192578


namespace stating_sheets_taken_exists_l1925_192535

/-- Represents the total number of pages in Hiram's algebra notes -/
def total_pages : ℕ := 60

/-- Represents the total number of sheets in Hiram's algebra notes -/
def total_sheets : ℕ := 30

/-- Represents the average of the remaining page numbers -/
def target_average : ℕ := 21

/-- 
Theorem stating that there exists a number of consecutive sheets taken 
such that the average of the remaining page numbers is the target average
-/
theorem sheets_taken_exists : 
  ∃ c : ℕ, c > 0 ∧ c < total_sheets ∧
  ∃ b : ℕ, b ≥ 0 ∧ b + c ≤ total_sheets ∧
  (b * (2 * b + 1) + 
   ((2 * (b + c) + 1 + total_pages) * (total_pages - 2 * c - 2 * b)) / 2) / 
   (total_pages - 2 * c) = target_average :=
sorry

end stating_sheets_taken_exists_l1925_192535


namespace tennis_ball_order_l1925_192585

theorem tennis_ball_order (white yellow : ℕ) : 
  white = yellow →                        -- Initially equal number of white and yellow balls
  (white : ℚ) / (yellow + 70 : ℚ) = 8/13 →  -- Ratio after error
  white + yellow = 224 :=                 -- Total number of balls ordered
by sorry

end tennis_ball_order_l1925_192585


namespace x_plus_p_equals_2p_plus_3_l1925_192517

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2 * p + 3 := by
sorry

end x_plus_p_equals_2p_plus_3_l1925_192517


namespace quadratic_real_roots_l1925_192512

/-- The quadratic equation (k-1)x^2 + 3x - 1 = 0 has real roots if and only if k ≥ -5/4 and k ≠ 1 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -5/4 ∧ k ≠ 1) :=
sorry

end quadratic_real_roots_l1925_192512


namespace wire_cut_square_circle_ratio_l1925_192590

theorem wire_cut_square_circle_ratio (x y : ℝ) (h : x > 0) (k : y > 0) : 
  (x^2 / 16 = y^2 / (4 * Real.pi)) → x / y = 2 / Real.sqrt Real.pi := by
  sorry

end wire_cut_square_circle_ratio_l1925_192590


namespace triangle_properties_l1925_192547

-- Define the triangle ABC
def Triangle (A B C : Real) (a b c : Real) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Given conditions
axiom triangle_condition {A B C a b c : Real} (h : Triangle A B C a b c) :
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

axiom c_value {A B C a b c : Real} (h : Triangle A B C a b c) :
  c = Real.sqrt 7

axiom area_value {A B C a b c : Real} (h : Triangle A B C a b c) :
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2

-- Theorem to prove
theorem triangle_properties {A B C a b c : Real} (h : Triangle A B C a b c) :
  C = Real.pi / 3 ∧ a + b + c = 5 + Real.sqrt 7 := by
  sorry

end triangle_properties_l1925_192547


namespace gcf_60_90_l1925_192533

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l1925_192533


namespace binary_to_quaternary_conversion_l1925_192595

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1011011110₂ -/
def binary_input : List Bool := [true, false, true, true, false, true, true, true, true, false]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_input) = [2, 3, 1, 3, 2] :=
sorry

end binary_to_quaternary_conversion_l1925_192595


namespace max_naive_number_with_divisible_ratio_l1925_192542

def is_naive_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 % 10 = n % 10 + 6) ∧
  (n / 100 % 10 = n / 10 % 10 + 2)

def P (n : ℕ) : ℕ :=
  3 * (n / 1000 % 10 + n / 100 % 10) + n / 10 % 10 + n % 10

def Q (n : ℕ) : ℕ :=
  n / 1000 % 10 - 5

theorem max_naive_number_with_divisible_ratio :
  ∃ (m : ℕ), is_naive_number m ∧ 
             (∀ n, is_naive_number n → P n / Q n % 10 = 0 → n ≤ m) ∧
             (P m / Q m % 10 = 0) ∧
             m = 9313 :=
sorry

end max_naive_number_with_divisible_ratio_l1925_192542


namespace problem_solution_l1925_192541

theorem problem_solution (x y : ℝ) :
  y = (Real.sqrt (x^2 - 4) + Real.sqrt (4 - x^2) + 1) / (x - 2) →
  3 * x + 4 * y = -7 := by
sorry

end problem_solution_l1925_192541


namespace scale_length_difference_l1925_192521

/-- Proves that a 7 ft scale divided into 4 equal parts of 24 inches each has 12 additional inches -/
theorem scale_length_difference : 
  let scale_length_ft : ℕ := 7
  let num_parts : ℕ := 4
  let part_length_inches : ℕ := 24
  let inches_per_foot : ℕ := 12
  
  (num_parts * part_length_inches) - (scale_length_ft * inches_per_foot) = 12 := by
  sorry

end scale_length_difference_l1925_192521


namespace calculate_expression_l1925_192559

theorem calculate_expression : (28 * (9 + 2 - 5)) * 3 = 504 := by
  sorry

end calculate_expression_l1925_192559


namespace prime_factors_of_30_factorial_l1925_192584

theorem prime_factors_of_30_factorial (n : ℕ) : n = 30 →
  (Finset.filter (Nat.Prime) (Finset.range (n + 1))).card = 10 := by
  sorry

end prime_factors_of_30_factorial_l1925_192584


namespace geometric_sum_five_quarters_l1925_192532

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_five_quarters :
  geometric_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end geometric_sum_five_quarters_l1925_192532


namespace inequality_proof_l1925_192570

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end inequality_proof_l1925_192570


namespace chocolate_milk_syrup_amount_l1925_192524

/-- Proves that the amount of chocolate syrup in each glass is 1.5 ounces -/
theorem chocolate_milk_syrup_amount :
  let glass_size : ℝ := 8
  let milk_per_glass : ℝ := 6.5
  let total_milk : ℝ := 130
  let total_syrup : ℝ := 60
  let total_mixture : ℝ := 160
  ∃ (num_glasses : ℕ) (syrup_per_glass : ℝ),
    (↑num_glasses : ℝ) * glass_size = total_mixture ∧
    (↑num_glasses : ℝ) * milk_per_glass = total_milk ∧
    (↑num_glasses : ℝ) * syrup_per_glass ≤ total_syrup ∧
    glass_size = milk_per_glass + syrup_per_glass ∧
    syrup_per_glass = 1.5 :=
by
  sorry

end chocolate_milk_syrup_amount_l1925_192524


namespace or_necessary_not_sufficient_for_and_l1925_192534

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (∀ (p q : Prop), (p ∧ q) → (p ∨ q)) ∧
  (∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q)) :=
by sorry

end or_necessary_not_sufficient_for_and_l1925_192534


namespace expected_heads_theorem_l1925_192589

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The number of maximum flips -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads on a single flip -/
def prob_heads : ℚ := 1/2

/-- The probability of a coin showing heads after all flips -/
def prob_heads_after_flips : ℚ := 15/16

/-- The expected number of coins showing heads after all flips -/
def expected_heads : ℚ := num_coins * prob_heads_after_flips

theorem expected_heads_theorem :
  ⌊expected_heads⌋ = 94 :=
sorry

end expected_heads_theorem_l1925_192589


namespace unique_modular_solution_l1925_192563

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [ZMOD 9] ∧ n = 2 := by
  sorry

end unique_modular_solution_l1925_192563


namespace square_of_negative_product_l1925_192531

theorem square_of_negative_product (m n : ℝ) : (-2 * m * n)^2 = 4 * m^2 * n^2 := by
  sorry

end square_of_negative_product_l1925_192531


namespace drill_bits_purchase_l1925_192596

theorem drill_bits_purchase (cost_per_set : ℝ) (tax_rate : ℝ) (total_paid : ℝ) 
  (h1 : cost_per_set = 6)
  (h2 : tax_rate = 0.1)
  (h3 : total_paid = 33) :
  ∃ (num_sets : ℕ), (cost_per_set * (num_sets : ℝ)) * (1 + tax_rate) = total_paid ∧ num_sets = 5 := by
  sorry

end drill_bits_purchase_l1925_192596


namespace a_less_than_sqrt3b_l1925_192509

theorem a_less_than_sqrt3b (a b : ℤ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : (a + b) ∣ (a * b + 1)) 
  (h4 : (a - b) ∣ (a * b - 1)) : 
  a < Real.sqrt 3 * b := by
sorry

end a_less_than_sqrt3b_l1925_192509


namespace remainder_theorem_l1925_192591

def dividend (k : ℤ) (x : ℝ) : ℝ := 3 * x^3 + k * x^2 + 8 * x - 24

def divisor (x : ℝ) : ℝ := 3 * x + 4

theorem remainder_theorem (k : ℤ) :
  (∃ q : ℝ → ℝ, ∀ x, dividend k x = (divisor x) * (q x) + 5) ↔ k = 29 := by
  sorry

end remainder_theorem_l1925_192591


namespace equation_solution_l1925_192543

theorem equation_solution : ∃ (x₁ x₂ : ℚ),
  x₁ = -1/3 ∧ x₂ = -2 ∧
  (∀ x : ℚ, x ≠ 3 → x ≠ 1/2 → 
    ((2*x + 4) / (x - 3) = (x + 2) / (2*x - 1) ↔ x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solution_l1925_192543


namespace less_than_implies_less_than_minus_one_l1925_192529

theorem less_than_implies_less_than_minus_one {a b : ℝ} (h : a < b) : a - 1 < b - 1 := by
  sorry

end less_than_implies_less_than_minus_one_l1925_192529


namespace min_difference_triangle_sides_l1925_192579

theorem min_difference_triangle_sides (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2010 →
  PQ < QR →
  QR < PR →
  (∀ PQ' QR' PR' : ℕ, 
    PQ' + QR' + PR' = 2010 →
    PQ' < QR' →
    QR' < PR' →
    QR - PQ ≤ QR' - PQ') →
  QR - PQ = 1 := by
sorry

end min_difference_triangle_sides_l1925_192579


namespace largest_number_l1925_192583

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -1 → b = 0 → c = 2 → d = Real.sqrt 3 →
  a < b ∧ b < d ∧ d < c :=
fun a b c d ha hb hc hd => by
  sorry

end largest_number_l1925_192583


namespace rationalize_and_product_l1925_192518

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (2 : ℝ) - Real.sqrt 5 / (3 + Real.sqrt 5) = A + B * Real.sqrt C ∧
  A * B * C = -50 := by
  sorry

end rationalize_and_product_l1925_192518


namespace polynomial_divisibility_l1925_192565

theorem polynomial_divisibility (a : ℤ) : ∃ k : ℤ, (3*a + 5)^2 - 4 = k * (a + 1) := by
  sorry

end polynomial_divisibility_l1925_192565


namespace negation_of_p_l1925_192555

def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
  sorry

end negation_of_p_l1925_192555


namespace order_of_logarithmic_expressions_l1925_192572

open Real

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := log x / log 10

-- State the theorem
theorem order_of_logarithmic_expressions (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  sqrt (lg a * lg b) < (lg a + lg b) / 2 ∧ (lg a + lg b) / 2 < lg ((a + b) / 2) :=
by sorry

end order_of_logarithmic_expressions_l1925_192572


namespace solution_of_system_l1925_192507

/-- Given a system of equations with four distinct real numbers a₁, a₂, a₃, a₄,
    prove that the solution is x₁ = 1 / (a₄ - a₁), x₂ = 0, x₃ = 0, x₄ = 1 / (a₄ - a₁) -/
theorem solution_of_system (a₁ a₂ a₃ a₄ : ℝ) 
    (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ),
    (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
    (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
    (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
    (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1) ∧
    x₁ = 1 / (a₄ - a₁) ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / (a₄ - a₁) := by
  sorry


end solution_of_system_l1925_192507


namespace quadratic_vertex_and_extremum_l1925_192546

/-- Given a quadratic equation y = -x^2 + cx + d with roots -5 and 3,
    prove that its vertex is (4, 1) and it represents a maximum point. -/
theorem quadratic_vertex_and_extremum (c d : ℝ) :
  (∀ x, -x^2 + c*x + d = 0 ↔ x = -5 ∨ x = 3) →
  (∃! p : ℝ × ℝ, p.1 = 4 ∧ p.2 = 1 ∧ 
    (∀ x, -x^2 + c*x + d ≤ -p.1^2 + c*p.1 + d)) :=
by sorry

end quadratic_vertex_and_extremum_l1925_192546


namespace bob_cereal_difference_l1925_192566

/-- Represents the number of sides on Bob's die -/
def dieSides : ℕ := 8

/-- Represents the threshold for eating organic cereal -/
def organicThreshold : ℕ := 5

/-- Represents the number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Probability of eating organic cereal -/
def probOrganic : ℚ := 4 / 7

/-- Probability of eating gluten-free cereal -/
def probGlutenFree : ℚ := 3 / 7

/-- Expected difference in days between eating organic and gluten-free cereal -/
def expectedDifference : ℚ := daysInYear * (probOrganic - probGlutenFree)

theorem bob_cereal_difference :
  expectedDifference = 365 * (4/7 - 3/7) :=
sorry

end bob_cereal_difference_l1925_192566


namespace same_color_pair_count_l1925_192569

/-- The number of ways to choose a pair of socks of the same color -/
def choose_same_color_pair (green red purple : ℕ) : ℕ :=
  Nat.choose green 2 + Nat.choose red 2 + Nat.choose purple 2

/-- Theorem stating that choosing a pair of socks of the same color from 
    5 green, 6 red, and 4 purple socks results in 31 possibilities -/
theorem same_color_pair_count : choose_same_color_pair 5 6 4 = 31 := by
  sorry

end same_color_pair_count_l1925_192569


namespace inequality_proof_l1925_192526

theorem inequality_proof (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ a*b + 3*b + 2*c := by
  sorry

end inequality_proof_l1925_192526


namespace find_divisor_l1925_192536

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 56 → quotient = 4 → divisor * quotient = dividend → divisor = 14 := by
sorry

end find_divisor_l1925_192536


namespace binomial_product_l1925_192558

variable (x : ℝ)

theorem binomial_product :
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end binomial_product_l1925_192558


namespace combined_polyhedron_faces_l1925_192500

/-- A regular tetrahedron -/
structure Tetrahedron :=
  (edge_length : ℝ)

/-- A regular octahedron -/
structure Octahedron :=
  (edge_length : ℝ)

/-- A polyhedron formed by combining a tetrahedron and an octahedron -/
structure CombinedPolyhedron :=
  (tetra : Tetrahedron)
  (octa : Octahedron)
  (combined : tetra.edge_length = octa.edge_length)

/-- The number of faces in the combined polyhedron -/
def num_faces (p : CombinedPolyhedron) : ℕ := 7

theorem combined_polyhedron_faces (p : CombinedPolyhedron) : 
  num_faces p = 7 := by sorry

end combined_polyhedron_faces_l1925_192500


namespace no_valid_covering_for_6_and_7_l1925_192573

/-- Represents the L-shaped or T-shaped 4-cell figure -/
inductive TetrominoShape
| L
| T

/-- Represents a position on the n×n square -/
structure Position (n : ℕ) where
  x : Fin n
  y : Fin n

/-- Represents a tetromino (4-cell figure) placement on the square -/
structure TetrominoPlacement (n : ℕ) where
  shape : TetrominoShape
  position : Position n
  rotation : Fin 4  -- 0, 90, 180, or 270 degrees

/-- Checks if a tetromino placement is valid within the n×n square -/
def is_valid_placement (n : ℕ) (placement : TetrominoPlacement n) : Prop := sorry

/-- Checks if a set of tetromino placements covers the entire n×n square exactly once -/
def covers_square_once (n : ℕ) (placements : List (TetrominoPlacement n)) : Prop := sorry

theorem no_valid_covering_for_6_and_7 :
  ¬ (∃ (placements : List (TetrominoPlacement 6)), covers_square_once 6 placements) ∧
  ¬ (∃ (placements : List (TetrominoPlacement 7)), covers_square_once 7 placements) := by
  sorry

end no_valid_covering_for_6_and_7_l1925_192573


namespace fraction_sum_l1925_192564

theorem fraction_sum : (1 : ℚ) / 6 + (1 : ℚ) / 3 + (5 : ℚ) / 9 = (19 : ℚ) / 18 := by
  sorry

end fraction_sum_l1925_192564


namespace not_arithmetic_sequence_x_i_l1925_192522

/-- Given real constants a and b, a geometric sequence {c_i} with common ratio ≠ 1,
    and the line ax + by + c_i = 0 intersecting the parabola y^2 = 2px (p > 0)
    forming chords with midpoints M_i(x_i, y_i), prove that {x_i} cannot be an arithmetic sequence. -/
theorem not_arithmetic_sequence_x_i 
  (a b : ℝ) 
  (c : ℕ+ → ℝ) 
  (p : ℝ) 
  (hp : p > 0)
  (hc : ∃ (r : ℝ), r ≠ 1 ∧ ∀ (i : ℕ+), c (i + 1) = r * c i)
  (x y : ℕ+ → ℝ)
  (h_intersect : ∀ (i : ℕ+), ∃ (t : ℝ), a * t + b * y i + c i = 0 ∧ (y i)^2 = 2 * p * t)
  (h_midpoint : ∀ (i : ℕ+), ∃ (t₁ t₂ : ℝ), 
    a * t₁ + b * (y i) + c i = 0 ∧ (y i)^2 = 2 * p * t₁ ∧
    a * t₂ + b * (y i) + c i = 0 ∧ (y i)^2 = 2 * p * t₂ ∧
    x i = (t₁ + t₂) / 2 ∧ y i = (y i + y i) / 2) :
  ¬ (∃ (d : ℝ), ∀ (i : ℕ+), x (i + 1) - x i = d) :=
sorry

end not_arithmetic_sequence_x_i_l1925_192522


namespace unique_function_solution_l1925_192587

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1) ↔ 
  (∀ x : ℝ, f x = 1 - x^2 / 2) :=
by sorry

end unique_function_solution_l1925_192587


namespace anna_candy_purchase_l1925_192537

def candy_problem (initial_money : ℚ) (gum_price : ℚ) (gum_quantity : ℕ) 
  (chocolate_price : ℚ) (cane_price : ℚ) (cane_quantity : ℕ) (money_left : ℚ) : Prop :=
  ∃ (chocolate_quantity : ℕ),
    initial_money - 
    (gum_price * gum_quantity + 
     chocolate_price * chocolate_quantity + 
     cane_price * cane_quantity) = money_left ∧
    chocolate_quantity = 5

theorem anna_candy_purchase : 
  candy_problem 10 1 3 1 0.5 2 1 := by sorry

end anna_candy_purchase_l1925_192537


namespace runner_a_race_time_l1925_192525

/-- Runner A in a race scenario --/
structure RunnerA where
  race_distance : ℝ
  head_start_distance : ℝ
  head_start_time : ℝ

/-- Theorem: Runner A completes the race in 200 seconds --/
theorem runner_a_race_time (a : RunnerA) 
  (h1 : a.race_distance = 1000)
  (h2 : a.head_start_distance = 50)
  (h3 : a.head_start_time = 10) : 
  a.race_distance / (a.head_start_distance / a.head_start_time) = 200 := by
  sorry

end runner_a_race_time_l1925_192525


namespace root_equation_implies_value_l1925_192516

theorem root_equation_implies_value (m : ℝ) : 
  m^2 - 2*m - 2019 = 0 → 2*m^2 - 4*m = 4038 := by
  sorry

end root_equation_implies_value_l1925_192516


namespace product_of_difference_and_sum_squares_l1925_192554

theorem product_of_difference_and_sum_squares (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a^2 + b^2 = 48) : 
  a * b = 6 := by
sorry

end product_of_difference_and_sum_squares_l1925_192554


namespace circle_equation_min_distance_l1925_192504

theorem circle_equation_min_distance (x y : ℝ) :
  (x^2 + y^2 - 64 = 0) → (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 64 :=
by sorry

end circle_equation_min_distance_l1925_192504


namespace money_left_after_trip_l1925_192571

def initial_savings : ℕ := 6000
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000

theorem money_left_after_trip : 
  initial_savings - (flight_cost + hotel_cost + food_cost) = 1000 := by
  sorry

end money_left_after_trip_l1925_192571


namespace min_value_of_f_l1925_192506

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / (x - 1)

theorem min_value_of_f :
  ∃ (x_min : ℝ), x_min > 1 ∧
  (∀ (x : ℝ), x > 1 → f x ≥ f x_min) ∧
  f x_min = 2 * Real.sqrt 3 + 2 :=
sorry

end min_value_of_f_l1925_192506


namespace total_road_cost_l1925_192588

/-- Represents the dimensions of a rectangular lawn -/
structure LawnDimensions where
  length : ℕ
  width : ℕ

/-- Represents a road segment with its length and cost per square meter -/
structure RoadSegment where
  length : ℕ
  cost_per_sqm : ℕ

/-- Calculates the total cost of a road given its segments and width -/
def road_cost (segments : List RoadSegment) (width : ℕ) : ℕ :=
  segments.foldl (fun acc segment => acc + segment.length * segment.cost_per_sqm * width) 0

/-- The main theorem stating the total cost of traveling the two roads -/
theorem total_road_cost (lawn : LawnDimensions)
  (length_road : List RoadSegment) (breadth_road : List RoadSegment) (road_width : ℕ) :
  lawn.length = 100 ∧ lawn.width = 60 ∧
  road_width = 10 ∧
  length_road = [⟨30, 4⟩, ⟨40, 5⟩, ⟨30, 6⟩] ∧
  breadth_road = [⟨20, 3⟩, ⟨40, 2⟩] →
  road_cost length_road road_width + road_cost breadth_road road_width = 6400 := by
  sorry

end total_road_cost_l1925_192588


namespace ellipse_hyperbola_same_foci_l1925_192577

/-- Given an ellipse and a hyperbola with equations as specified,
    if they have the same foci, then m = ±1 -/
theorem ellipse_hyperbola_same_foci (m : ℝ) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → x^2 / m^2 - y^2 / 2 = 1 → 
    (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m^2 + 2)) →
  m = 1 ∨ m = -1 := by
  sorry

end ellipse_hyperbola_same_foci_l1925_192577


namespace fish_count_l1925_192557

theorem fish_count (bass trout bluegill : ℕ) : 
  bass = 32 →
  trout = bass / 4 →
  bluegill = 2 * bass →
  bass + trout + bluegill = 104 := by
sorry

end fish_count_l1925_192557


namespace hyunseung_outfit_combinations_l1925_192553

/-- The number of types of tops in Hyunseung's closet -/
def num_tops : ℕ := 3

/-- The number of types of bottoms in Hyunseung's closet -/
def num_bottoms : ℕ := 2

/-- The number of types of shoes in Hyunseung's closet -/
def num_shoes : ℕ := 5

/-- The total number of combinations of tops, bottoms, and shoes Hyunseung can wear -/
def total_combinations : ℕ := num_tops * num_bottoms * num_shoes

theorem hyunseung_outfit_combinations : total_combinations = 30 := by
  sorry

end hyunseung_outfit_combinations_l1925_192553


namespace finance_marketing_specialization_contradiction_l1925_192508

theorem finance_marketing_specialization_contradiction 
  (finance_percent1 : ℝ) 
  (finance_percent2 : ℝ) 
  (marketing_percent : ℝ) 
  (h1 : finance_percent1 = 88) 
  (h2 : marketing_percent = 76) 
  (h3 : finance_percent2 = 90) 
  (h4 : 0 ≤ finance_percent1 ∧ finance_percent1 ≤ 100) 
  (h5 : 0 ≤ finance_percent2 ∧ finance_percent2 ≤ 100) 
  (h6 : 0 ≤ marketing_percent ∧ marketing_percent ≤ 100) :
  finance_percent1 ≠ finance_percent2 := by
  sorry

end finance_marketing_specialization_contradiction_l1925_192508


namespace certain_number_problem_l1925_192586

theorem certain_number_problem (x : ℝ) : ((7 * (x + 10)) / 5) - 5 = 44 → x = 25 := by
  sorry

end certain_number_problem_l1925_192586


namespace largest_prime_to_test_primality_l1925_192574

theorem largest_prime_to_test_primality (n : ℕ) (h : 1100 ≤ n ∧ n ≤ 1150) :
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q, Nat.Prime q → q^2 ≤ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q, Nat.Prime q → q^2 ≤ n → q ≤ p ∧ p = 31) :=
by sorry

end largest_prime_to_test_primality_l1925_192574


namespace lucy_moved_fish_l1925_192511

/-- The number of fish moved to a different tank -/
def fish_moved (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

/-- Proof that Lucy moved 68 fish to a different tank -/
theorem lucy_moved_fish : fish_moved 212 144 = 68 := by
  sorry

end lucy_moved_fish_l1925_192511


namespace area_of_inscribed_rectangle_l1925_192514

/-- Given a square with side length s ≥ 4 containing a 2x2 square, 
    a 2x4 rectangle, and a non-overlapping rectangle R, 
    the area of R is exactly 4. -/
theorem area_of_inscribed_rectangle (s : ℝ) (h_s : s ≥ 4) : 
  s^2 - (2 * 2 + 2 * 4) = 4 := by sorry

end area_of_inscribed_rectangle_l1925_192514


namespace subset_sum_divisible_by_2n_l1925_192550

theorem subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) 
  (S : Finset ℕ) (hS : S.card = n) (hS_subset : ∀ x ∈ S, x ∈ Finset.range (2*n)) :
  ∃ T : Finset ℕ, T ⊆ S ∧ (2*n) ∣ (T.sum id) :=
sorry

end subset_sum_divisible_by_2n_l1925_192550


namespace min_bing_toys_l1925_192556

/-- Represents the cost and pricing of Olympic mascot toys --/
structure OlympicToys where
  bing_cost : ℕ  -- Cost of Bing Dwen Dwen
  shuey_cost : ℕ  -- Cost of Shuey Rongrong
  bing_price : ℕ  -- Selling price of Bing Dwen Dwen
  shuey_price : ℕ  -- Selling price of Shuey Rongrong

/-- Theorem about the minimum number of Bing Dwen Dwen toys to purchase --/
theorem min_bing_toys (t : OlympicToys) 
  (h1 : 4 * t.bing_cost + 5 * t.shuey_cost = 1000)
  (h2 : 5 * t.bing_cost + 10 * t.shuey_cost = 1550)
  (h3 : t.bing_price = 180)
  (h4 : t.shuey_price = 100)
  (h5 : ∀ x : ℕ, x + (180 - x) = 180)
  (h6 : ∀ x : ℕ, x * (t.bing_price - t.bing_cost) + (180 - x) * (t.shuey_price - t.shuey_cost) ≥ 4600) :
  ∃ (min_bing : ℕ), min_bing = 100 ∧ 
    ∀ (x : ℕ), x ≥ min_bing → 
      x * (t.bing_price - t.bing_cost) + (180 - x) * (t.shuey_price - t.shuey_cost) ≥ 4600 :=
sorry

end min_bing_toys_l1925_192556


namespace perpendicular_lines_parallel_lines_l1925_192502

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := a * 1 + 2 * (a - 1) = 0

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := a / 1 = 2 / (a - 1) ∧ a / 1 ≠ 6 / (a^2 - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perpendicular a → a = 2/3 :=
sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 :=
sorry

end perpendicular_lines_parallel_lines_l1925_192502


namespace taehyung_average_problems_l1925_192513

/-- The average number of problems solved per day -/
def average_problems_per_day (total_problems : ℕ) (num_days : ℕ) : ℚ :=
  (total_problems : ℚ) / (num_days : ℚ)

/-- Theorem stating that the average number of problems solved per day is 23 -/
theorem taehyung_average_problems :
  average_problems_per_day 161 7 = 23 := by
  sorry

end taehyung_average_problems_l1925_192513


namespace sqrt_1575n_integer_exists_l1925_192519

theorem sqrt_1575n_integer_exists : ∃ n : ℕ+, ∃ k : ℕ, (k : ℝ) ^ 2 = 1575 * n := by
  sorry

end sqrt_1575n_integer_exists_l1925_192519


namespace investment_worth_l1925_192510

def investment_problem (initial_investment : ℚ) (months : ℕ) (monthly_earnings : ℚ) : Prop :=
  let total_earnings := monthly_earnings * months
  let current_worth := initial_investment + total_earnings
  (months = 5) ∧
  (monthly_earnings = 12) ∧
  (total_earnings = 2 * initial_investment) ∧
  (current_worth = 90)

theorem investment_worth :
  ∃ (initial_investment : ℚ), investment_problem initial_investment 5 12 :=
by sorry

end investment_worth_l1925_192510


namespace three_number_set_range_l1925_192568

theorem three_number_set_range (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ascending order
  a = 2 ∧  -- Smallest number is 2
  b = 5 ∧  -- Median is 5
  (a + b + c) / 3 = 5 →  -- Mean is 5
  c - a = 6 :=  -- Range is 6
by sorry

end three_number_set_range_l1925_192568


namespace perpendicular_vectors_l1925_192523

/-- Given vectors a and b in ℝ², prove that if (a + x*b) is perpendicular to (a - b), then x = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 4))
  (h2 : b = (2, 1))
  (h3 : (a + x • b) • (a - b) = 0) :
  x = -3 := by sorry

end perpendicular_vectors_l1925_192523


namespace angle_between_polar_lines_theorem_l1925_192567

/-- The angle between two lines in polar coordinates -/
def angle_between_polar_lines (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- First line in polar coordinates: ρ(2cosθ + sinθ) = 2 -/
def line1 (ρ θ : ℝ) : Prop :=
  ρ * (2 * Real.cos θ + Real.sin θ) = 2

/-- Second line in polar coordinates: ρcosθ = 1 -/
def line2 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 1

/-- Theorem stating the angle between the two lines -/
theorem angle_between_polar_lines_theorem :
  angle_between_polar_lines line1 line2 = Real.arctan (1 / 2) :=
sorry

end angle_between_polar_lines_theorem_l1925_192567


namespace parabola_single_intersection_parabola_y_decreases_l1925_192576

def parabola (x m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_single_intersection (m : ℝ) :
  (∃! x, parabola x m = 0) ↔ m = -2 := sorry

theorem parabola_y_decreases (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  parabola x₁ m = y₁ →
  parabola x₂ m = y₂ →
  x₁ > x₂ →
  x₂ > 2 →
  y₁ < y₂ := sorry

end parabola_single_intersection_parabola_y_decreases_l1925_192576


namespace complex_simplification_and_multiplication_l1925_192545

theorem complex_simplification_and_multiplication :
  ((4 - 3 * Complex.I) - (7 - 5 * Complex.I)) * (1 + 2 * Complex.I) = -7 - 4 * Complex.I :=
by sorry

end complex_simplification_and_multiplication_l1925_192545


namespace five_numbers_satisfy_conditions_l1925_192598

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Calculates the sum of digits of a two-digit number -/
def sumOfDigits (n : TwoDigitNumber) : ℕ :=
  (n.val / 10) + (n.val % 10)

/-- Performs the operation described in the problem -/
def operation (n : TwoDigitNumber) : ℕ :=
  n.val - sumOfDigits n

/-- Checks if the units digit of a number is 4 -/
def hasUnitsDigit4 (n : ℕ) : Prop :=
  n % 10 = 4

/-- The main theorem stating that exactly 5 two-digit numbers satisfy the conditions -/
theorem five_numbers_satisfy_conditions :
  ∃! (s : Finset TwoDigitNumber),
    (∀ n ∈ s, isEven n.val ∧ hasUnitsDigit4 (operation n)) ∧
    s.card = 5 :=
sorry

end five_numbers_satisfy_conditions_l1925_192598
