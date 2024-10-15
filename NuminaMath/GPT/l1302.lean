import Mathlib

namespace NUMINAMATH_GPT_area_quadrilateral_EFGH_l1302_130260

-- Define the rectangles ABCD and XYZR
def area_rectangle_ABCD : ℝ := 60 
def area_rectangle_XYZR : ℝ := 4

-- Define what needs to be proven: the area of quadrilateral EFGH
theorem area_quadrilateral_EFGH (a b c d : ℝ) :
  (area_rectangle_ABCD = area_rectangle_XYZR + 2 * (a + b + c + d)) →
  (a + b + c + d = 28) →
  (area_rectangle_XYZR = 4) →
  (area_rectangle_ABCD = 60) →
  (a + b + c + d + area_rectangle_XYZR = 32) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_area_quadrilateral_EFGH_l1302_130260


namespace NUMINAMATH_GPT_tony_graduate_degree_years_l1302_130226

-- Define the years spent for each degree and the total time
def D1 := 4 -- years for the first degree in science
def D2 := 4 -- years for each of the two additional degrees
def T := 14 -- total years spent in school
def G := 2 -- years spent for the graduate degree in physics

-- Theorem: Given the conditions, prove that Tony spent 2 years on his graduate degree in physics
theorem tony_graduate_degree_years : 
  D1 + 2 * D2 + G = T :=
by
  sorry

end NUMINAMATH_GPT_tony_graduate_degree_years_l1302_130226


namespace NUMINAMATH_GPT_ordered_triples_count_l1302_130290

def similar_prisms_count (b : ℕ) (c : ℕ) (a : ℕ) := 
  (a ≤ c ∧ c ≤ b ∧ 
   ∃ (x y z : ℕ), x ≤ z ∧ z ≤ y ∧ y = b ∧ 
   x < a ∧ y < b ∧ z < c ∧ 
   ((x : ℚ) / a = (y : ℚ) / b ∧ (y : ℚ) / b = (z : ℚ) / c))

theorem ordered_triples_count : 
  ∃ (n : ℕ), n = 24 ∧ ∀ a c, similar_prisms_count 2000 c a → a < c :=
sorry

end NUMINAMATH_GPT_ordered_triples_count_l1302_130290


namespace NUMINAMATH_GPT_P_is_in_third_quadrant_l1302_130275

noncomputable def point : Type := (ℝ × ℝ)

def P : point := (-3, -4)

def is_in_third_quadrant (p : point) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem P_is_in_third_quadrant : is_in_third_quadrant P :=
by {
  -- Prove that P is in the third quadrant
  sorry
}

end NUMINAMATH_GPT_P_is_in_third_quadrant_l1302_130275


namespace NUMINAMATH_GPT_gcd_squares_example_l1302_130206

noncomputable def gcd_of_squares : ℕ :=
  Nat.gcd (101 ^ 2 + 202 ^ 2 + 303 ^ 2) (100 ^ 2 + 201 ^ 2 + 304 ^ 2)

theorem gcd_squares_example : gcd_of_squares = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_squares_example_l1302_130206


namespace NUMINAMATH_GPT_seating_arrangements_l1302_130256

def total_seats_front := 11
def total_seats_back := 12
def middle_seats_front := 3

def number_of_arrangements := 334

theorem seating_arrangements: 
  (total_seats_front - middle_seats_front) * (total_seats_front - middle_seats_front - 1) / 2 +
  (total_seats_back * (total_seats_back - 1)) / 2 +
  (total_seats_front - middle_seats_front) * total_seats_back +
  total_seats_back * (total_seats_front - middle_seats_front) = number_of_arrangements := 
sorry

end NUMINAMATH_GPT_seating_arrangements_l1302_130256


namespace NUMINAMATH_GPT_cos2alpha_plus_sin2alpha_l1302_130249

theorem cos2alpha_plus_sin2alpha (α : Real) (h : Real.tan (Real.pi + α) = 2) : 
  Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_cos2alpha_plus_sin2alpha_l1302_130249


namespace NUMINAMATH_GPT_completing_square_16x2_32x_512_eq_33_l1302_130217

theorem completing_square_16x2_32x_512_eq_33:
  (∃ p q : ℝ, (16 * x ^ 2 + 32 * x - 512 = 0) → (x + p) ^ 2 = q ∧ q = 33) :=
by
  sorry

end NUMINAMATH_GPT_completing_square_16x2_32x_512_eq_33_l1302_130217


namespace NUMINAMATH_GPT_sufficient_condition_B_is_proper_subset_of_A_l1302_130293

def A : Set ℝ := {x | x^2 + x = 6}
def B (m : ℝ) : Set ℝ := {-1 / m}

theorem sufficient_condition_B_is_proper_subset_of_A (m : ℝ) : 
  m = -1/2 → B m ⊆ A ∧ B m ≠ A :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_B_is_proper_subset_of_A_l1302_130293


namespace NUMINAMATH_GPT_power_of_two_representation_l1302_130235

/-- Prove that any number 2^n, where n = 3,4,5,..., can be represented 
as 7x^2 + y^2 where x and y are odd numbers. -/
theorem power_of_two_representation (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, (2*x ≠ 0 ∧ 2*y ≠ 0) ∧ 2^n = 7 * x^2 + y^2 :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_representation_l1302_130235


namespace NUMINAMATH_GPT_line_through_points_l1302_130202

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : x1 ≠ x2) (hx1 : x1 = -3) (hy1 : y1 = 1) (hx2 : x2 = 1) (hy2 : y2 = 5) :
  ∃ (m b : ℝ), (m + b = 5) ∧ (y1 = m * x1 + b) ∧ (y2 = m * x2 + b) :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_l1302_130202


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1302_130232

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) -- condition for arithmetic sequence
  (h_condition : a 3 + a 5 + a 7 + a 9 + a 11 = 100) : 
  3 * a 9 - a 13 = 40 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1302_130232


namespace NUMINAMATH_GPT_student_monthly_earnings_l1302_130223

theorem student_monthly_earnings :
  let daily_rate := 1250
  let days_per_week := 4
  let weeks_per_month := 4
  let income_tax_rate := 0.13
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_before_tax := weekly_earnings * weeks_per_month
  let income_tax_amount := monthly_earnings_before_tax * income_tax_rate
  let monthly_earnings_after_tax := monthly_earnings_before_tax - income_tax_amount
  monthly_earnings_after_tax = 17400 := by
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_student_monthly_earnings_l1302_130223


namespace NUMINAMATH_GPT_distribute_books_into_bags_l1302_130292

def number_of_ways_to_distribute_books (books : Finset ℕ) (bags : ℕ) : ℕ :=
  if (books.card = 5) ∧ (bags = 3) then 51 else 0

theorem distribute_books_into_bags :
  number_of_ways_to_distribute_books (Finset.range 5) 3 = 51 := by
  sorry

end NUMINAMATH_GPT_distribute_books_into_bags_l1302_130292


namespace NUMINAMATH_GPT_min_positive_period_cos_2x_l1302_130255

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem min_positive_period_cos_2x :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T := 
sorry

end NUMINAMATH_GPT_min_positive_period_cos_2x_l1302_130255


namespace NUMINAMATH_GPT_smallest_M_exists_l1302_130218

theorem smallest_M_exists :
  ∃ M : ℕ, M = 249 ∧
  (∃ k1 : ℕ, (M + k1 = 8 * k1 ∨ M + k1 + 1 = 8 * k1 ∨ M + k1 + 2 = 8 * k1)) ∧
  (∃ k2 : ℕ, (M + k2 = 27 * k2 ∨ M + k2 + 1 = 27 * k2 ∨ M + k2 + 2 = 27 * k2)) ∧
  (∃ k3 : ℕ, (M + k3 = 125 * k3 ∨ M + k3 + 1 = 125 * k3 ∨ M + k3 + 2 = 125 * k3)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_M_exists_l1302_130218


namespace NUMINAMATH_GPT_initial_money_is_correct_l1302_130262

-- Given conditions
def spend_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12
def money_left_after_year : ℕ := 104

-- Define the initial amount of money
def initial_amount_of_money (spend_per_trip trips_per_month months_per_year money_left_after_year : ℕ) : ℕ :=
  money_left_after_year + (spend_per_trip * trips_per_month * months_per_year)

-- Theorem stating that under the given conditions, the initial amount of money is 200
theorem initial_money_is_correct :
  initial_amount_of_money spend_per_trip trips_per_month months_per_year money_left_after_year = 200 :=
  sorry

end NUMINAMATH_GPT_initial_money_is_correct_l1302_130262


namespace NUMINAMATH_GPT_car_travel_inequality_l1302_130298

variable (x : ℕ)

theorem car_travel_inequality (hx : 8 * (x + 19) > 2200) : 8 * (x + 19) > 2200 :=
by
  sorry

end NUMINAMATH_GPT_car_travel_inequality_l1302_130298


namespace NUMINAMATH_GPT_total_rainfall_November_l1302_130204

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_November_l1302_130204


namespace NUMINAMATH_GPT_find_p_l1302_130208

theorem find_p
  (p : ℝ)
  (h1 : ∃ (x y : ℝ), p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1)
  (h2 : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
         x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
         p * (x₁^2 - y₁^2) = (p^2 - 1) * x₁ * y₁ ∧ |x₁ - 1| + |y₁| = 1 ∧
         p * (x₂^2 - y₂^2) = (p^2 - 1) * x₂ * y₂ ∧ |x₂ - 1| + |y₂| = 1 ∧
         p * (x₃^2 - y₃^2) = (p^2 - 1) * x₃ * y₃ ∧ |x₃ - 1| + |y₃| = 1) :
  p = 1 ∨ p = -1 :=
by sorry

end NUMINAMATH_GPT_find_p_l1302_130208


namespace NUMINAMATH_GPT_number_of_pairs_x_y_l1302_130240

theorem number_of_pairs_x_y (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 85) : 
    (1 : ℕ) + (1 : ℕ) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_pairs_x_y_l1302_130240


namespace NUMINAMATH_GPT_average_sales_is_167_5_l1302_130251

def sales_january : ℝ := 150
def sales_february : ℝ := 90
def sales_march : ℝ := 1.5 * sales_february
def sales_april : ℝ := 180
def sales_may : ℝ := 210
def sales_june : ℝ := 240
def total_sales : ℝ := sales_january + sales_february + sales_march + sales_april + sales_may + sales_june
def number_of_months : ℝ := 6

theorem average_sales_is_167_5 :
  total_sales / number_of_months = 167.5 :=
sorry

end NUMINAMATH_GPT_average_sales_is_167_5_l1302_130251


namespace NUMINAMATH_GPT_work_days_l1302_130250

theorem work_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
    1 / (A + B) = 6 :=
by
  sorry

end NUMINAMATH_GPT_work_days_l1302_130250


namespace NUMINAMATH_GPT_breakfast_cost_l1302_130269

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_breakfast_cost_l1302_130269


namespace NUMINAMATH_GPT_wayne_took_cards_l1302_130276

-- Let's define the problem context
variable (initial_cards : ℕ := 76)
variable (remaining_cards : ℕ := 17)

-- We need to show that Wayne took away 59 cards
theorem wayne_took_cards (x : ℕ) (h : x = initial_cards - remaining_cards) : x = 59 :=
by
  sorry

end NUMINAMATH_GPT_wayne_took_cards_l1302_130276


namespace NUMINAMATH_GPT_no_n_exists_11_div_mod_l1302_130296

theorem no_n_exists_11_div_mod (n : ℕ) (h1 : n > 0) (h2 : 3^5 ≡ 1 [MOD 11]) (h3 : 4^5 ≡ 1 [MOD 11]) : ¬ (11 ∣ (3^n + 4^n)) := 
sorry

end NUMINAMATH_GPT_no_n_exists_11_div_mod_l1302_130296


namespace NUMINAMATH_GPT_fox_jeans_price_l1302_130225

theorem fox_jeans_price (pony_price : ℝ)
                        (total_savings : ℝ)
                        (total_discount_rate : ℝ)
                        (pony_discount_rate : ℝ)
                        (fox_discount_rate : ℝ)
                        (fox_price : ℝ) :
    pony_price = 18 ∧
    total_savings = 8.91 ∧
    total_discount_rate = 0.22 ∧
    pony_discount_rate = 0.1099999999999996 ∧
    fox_discount_rate = 0.11 →
    (3 * fox_discount_rate * fox_price + 2 * pony_discount_rate * pony_price = total_savings) →
    fox_price = 15 :=
by
  intros h h_eq
  rcases h with ⟨h_pony, h_savings, h_total_rate, h_pony_rate, h_fox_rate⟩
  sorry

end NUMINAMATH_GPT_fox_jeans_price_l1302_130225


namespace NUMINAMATH_GPT_age_ratio_in_ten_years_l1302_130247

-- Definitions of given conditions
variable (A : ℕ) (B : ℕ)
axiom age_condition : A = 20
axiom sum_of_ages : A + 10 + (B + 10) = 45

-- Theorem and proof skeleton for the ratio of ages in ten years.
theorem age_ratio_in_ten_years (A B : ℕ) (hA : A = 20) (hSum : A + 10 + (B + 10) = 45) :
  (A + 10) / (B + 10) = 2 := by
  sorry

end NUMINAMATH_GPT_age_ratio_in_ten_years_l1302_130247


namespace NUMINAMATH_GPT_circle_equation_l1302_130219

theorem circle_equation (x y : ℝ) :
  (∀ (C P : ℝ × ℝ), C = (8, -3) ∧ P = (5, 1) →
    ∃ R : ℝ, (x - 8)^2 + (y + 3)^2 = R^2 ∧ R^2 = 25) :=
sorry

end NUMINAMATH_GPT_circle_equation_l1302_130219


namespace NUMINAMATH_GPT_count_integers_with_factors_12_and_7_l1302_130215

theorem count_integers_with_factors_12_and_7 :
  ∃ k : ℕ, k = 4 ∧
    (∀ (n : ℕ), 500 ≤ n ∧ n ≤ 800 ∧ 12 ∣ n ∧ 7 ∣ n ↔ (84 ∣ n ∧
      n = 504 ∨ n = 588 ∨ n = 672 ∨ n = 756)) :=
sorry

end NUMINAMATH_GPT_count_integers_with_factors_12_and_7_l1302_130215


namespace NUMINAMATH_GPT_least_value_expression_l1302_130234

-- Definition of the expression
def expression (x y : ℝ) := (x * y - 2) ^ 2 + (x - 1 + y) ^ 2

-- Statement to prove the least possible value of the expression
theorem least_value_expression : ∃ x y : ℝ, expression x y = 2 := 
sorry

end NUMINAMATH_GPT_least_value_expression_l1302_130234


namespace NUMINAMATH_GPT_third_divisor_is_11_l1302_130280

theorem third_divisor_is_11 (n : ℕ) (x : ℕ) : 
  n = 200 ∧ (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % x = 0 ∧ (n - 20) % 60 = 0 → 
  x = 11 :=
by
  sorry

end NUMINAMATH_GPT_third_divisor_is_11_l1302_130280


namespace NUMINAMATH_GPT_prove_sequences_and_sum_l1302_130220

theorem prove_sequences_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (a 1 = 5) →
  (a 2 = 2) →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  (∀ n, ∃ r1, (a (n + 1) - 2 * a n) = (a 2 - 2 * a 1) * r1 ^ n) ∧
  (∀ n, ∃ r2, (a (n + 1) - (1 / 2) * a n) = (a 2 - (1 / 2) * a 1) * r2 ^ n) ∧
  (∀ n, S n = (4 * n) / 3 + (4 ^ n) / 36 - 1 / 36) :=
by
  sorry

end NUMINAMATH_GPT_prove_sequences_and_sum_l1302_130220


namespace NUMINAMATH_GPT_two_cards_totaling_15_probability_l1302_130268

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end NUMINAMATH_GPT_two_cards_totaling_15_probability_l1302_130268


namespace NUMINAMATH_GPT_truth_values_of_p_and_q_l1302_130266

variable (p q : Prop)

theorem truth_values_of_p_and_q
  (h1 : ¬ (p ∧ q))
  (h2 : (¬ p ∨ q)) :
  ¬ p ∧ (q ∨ ¬ q) :=
by {
  sorry
}

end NUMINAMATH_GPT_truth_values_of_p_and_q_l1302_130266


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1302_130214

theorem solution_set_of_inequality :
  {x : ℝ | (x-2)*(3-x) > 0} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1302_130214


namespace NUMINAMATH_GPT_units_digit_n_squared_plus_two_pow_n_l1302_130244

theorem units_digit_n_squared_plus_two_pow_n
  (n : ℕ)
  (h : n = 2018^2 + 2^2018) : 
  (n^2 + 2^n) % 10 = 5 := by
  sorry

end NUMINAMATH_GPT_units_digit_n_squared_plus_two_pow_n_l1302_130244


namespace NUMINAMATH_GPT_quadratic_part_of_equation_l1302_130254

theorem quadratic_part_of_equation (x: ℝ) :
  (x^2 - 8*x + 21 = |x - 5| + 4) → (x^2 - 8*x + 21) = x^2 - 8*x + 21 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_quadratic_part_of_equation_l1302_130254


namespace NUMINAMATH_GPT_sum_six_seven_l1302_130265

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom arithmetic_sequence : ∀ (n : ℕ), a (n + 1) = a n + d
axiom sum_condition : a 2 + a 5 + a 8 + a 11 = 48

theorem sum_six_seven : a 6 + a 7 = 24 :=
by
  -- Using given axioms and properties of arithmetic sequence
  sorry

end NUMINAMATH_GPT_sum_six_seven_l1302_130265


namespace NUMINAMATH_GPT_arithmetic_sequence_b1_l1302_130257

theorem arithmetic_sequence_b1 
  (b : ℕ → ℝ) 
  (U : ℕ → ℝ)
  (U2023 : ℝ) 
  (b2023 : ℝ)
  (hb2023 : b 2023 = b 1 + 2022 * (b 2 - b 1))
  (hU2023 : U 2023 = 2023 * (b 1 + 1011 * (b 2 - b 1))) 
  (hUn : ∀ n, U n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1)) / 2)) :
  b 1 = (U 2023 - 2023 * b 2023) / 2023 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_b1_l1302_130257


namespace NUMINAMATH_GPT_combined_time_l1302_130264

def time_pulsar : ℕ := 10
def time_polly : ℕ := 3 * time_pulsar
def time_petra : ℕ := time_polly / 6

theorem combined_time : time_pulsar + time_polly + time_petra = 45 := 
by 
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_combined_time_l1302_130264


namespace NUMINAMATH_GPT_percent_profit_l1302_130207

-- Definitions based on given conditions
variables (P : ℝ) -- original price of the car

def discounted_price := 0.90 * P
def first_year_value := 0.945 * P
def second_year_value := 0.9828 * P
def third_year_value := 1.012284 * P
def selling_price := 1.62 * P

-- Theorem statement
theorem percent_profit : (selling_price P - P) / P * 100 = 62 := by
  sorry

end NUMINAMATH_GPT_percent_profit_l1302_130207


namespace NUMINAMATH_GPT_number_of_sons_l1302_130279

noncomputable def land_area_hectares : ℕ := 3
noncomputable def hectare_to_m2 : ℕ := 10000
noncomputable def profit_per_section_per_3months : ℕ := 500
noncomputable def section_area_m2 : ℕ := 750
noncomputable def profit_per_son_per_year : ℕ := 10000
noncomputable def months_in_year : ℕ := 12
noncomputable def months_per_season : ℕ := 3

theorem number_of_sons :
  let total_land_area_m2 := land_area_hectares * hectare_to_m2
  let yearly_profit_per_section := profit_per_section_per_3months * (months_in_year / months_per_season)
  let number_of_sections := total_land_area_m2 / section_area_m2
  let total_yearly_profit := number_of_sections * yearly_profit_per_section
  let n := total_yearly_profit / profit_per_son_per_year
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sons_l1302_130279


namespace NUMINAMATH_GPT_GP_length_l1302_130233

theorem GP_length (X Y Z G P Q : Type) 
  (XY XZ YZ : ℝ) 
  (hXY : XY = 12) 
  (hXZ : XZ = 9) 
  (hYZ : YZ = 15) 
  (hG_centroid : true)  -- Medians intersect at G (Centroid property)
  (hQ_altitude : true)  -- Q is the foot of the altitude from X to YZ
  (hP_below_G : true)  -- P is the point on YZ directly below G
  : GP = 2.4 := 
sorry

end NUMINAMATH_GPT_GP_length_l1302_130233


namespace NUMINAMATH_GPT_tank_volume_ratio_l1302_130278

variable {V1 V2 : ℝ}

theorem tank_volume_ratio
  (h1 : 3 / 4 * V1 = 5 / 8 * V2) :
  V1 / V2 = 5 / 6 :=
sorry

end NUMINAMATH_GPT_tank_volume_ratio_l1302_130278


namespace NUMINAMATH_GPT_factorize_expression_l1302_130230

-- Define the variables a and b
variables (a b : ℝ)

-- State the theorem
theorem factorize_expression : 5*a^2*b - 20*b^3 = 5*b*(a + 2*b)*(a - 2*b) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l1302_130230


namespace NUMINAMATH_GPT_last_digit_of_2_pow_2004_l1302_130274

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_last_digit_of_2_pow_2004_l1302_130274


namespace NUMINAMATH_GPT_thalassa_population_2050_l1302_130212

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end NUMINAMATH_GPT_thalassa_population_2050_l1302_130212


namespace NUMINAMATH_GPT_cistern_fill_time_l1302_130243

theorem cistern_fill_time (A B : ℝ) (hA : A = 1/60) (hB : B = 1/45) : (|A - B|)⁻¹ = 180 := by
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l1302_130243


namespace NUMINAMATH_GPT_find_coordinates_of_P_l1302_130201

-- Definitions based on the conditions:
-- Point P has coordinates (a, 2a-1) and lies on the line y = x.

def lies_on_bisector (a : ℝ) : Prop :=
  (2 * a - 1) = a -- This is derived from the line y = x for the given point coordinates.

-- The final statement to prove:
theorem find_coordinates_of_P (a : ℝ) (P : ℝ × ℝ) (h1 : P = (a, 2 * a - 1)) (h2 : lies_on_bisector a) :
  P = (1, 1) :=
by
  -- Proof steps are omitted and replaced with sorry.
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l1302_130201


namespace NUMINAMATH_GPT_cos_alpha_value_cos_2alpha_value_l1302_130248

noncomputable def x : ℤ := -3
noncomputable def y : ℤ := 4
noncomputable def r : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def cos_alpha : ℝ := x / r
noncomputable def cos_2alpha : ℝ := 2 * cos_alpha^2 - 1

theorem cos_alpha_value : cos_alpha = -3 / 5 := by
  sorry

theorem cos_2alpha_value : cos_2alpha = -7 / 25 := by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_cos_2alpha_value_l1302_130248


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1302_130229

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : I * z = 1 + I) : z.im = -1 := 
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1302_130229


namespace NUMINAMATH_GPT_number_of_trailing_zeros_l1302_130241

def trailing_zeros (n : Nat) : Nat :=
  let powers_of_two := 2 * 52^5
  let powers_of_five := 2 * 25^2
  min powers_of_two powers_of_five

theorem number_of_trailing_zeros : trailing_zeros (525^(25^2) * 252^(52^5)) = 1250 := 
by sorry

end NUMINAMATH_GPT_number_of_trailing_zeros_l1302_130241


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1302_130286

theorem sum_of_reciprocals
  (m n p : ℕ)
  (HCF_mnp : Nat.gcd (Nat.gcd m n) p = 26)
  (LCM_mnp : Nat.lcm (Nat.lcm m n) p = 6930)
  (sum_mnp : m + n + p = 150) :
  (1 / (m : ℚ) + 1 / (n : ℚ) + 1 / (p : ℚ) = 1 / 320166) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1302_130286


namespace NUMINAMATH_GPT_cards_given_to_Jeff_l1302_130283

-- Definitions according to the conditions
def initial_cards : Nat := 304
def remaining_cards : Nat := 276

-- The proof problem
theorem cards_given_to_Jeff : initial_cards - remaining_cards = 28 :=
by
  sorry

end NUMINAMATH_GPT_cards_given_to_Jeff_l1302_130283


namespace NUMINAMATH_GPT_price_difference_proof_l1302_130236

theorem price_difference_proof (y : ℝ) (n : ℕ) :
  ∃ n : ℕ, (4.20 + 0.45 * n) = (6.30 + 0.01 * y * n + 0.65) → 
  n = (275 / (45 - y)) :=
by
  sorry

end NUMINAMATH_GPT_price_difference_proof_l1302_130236


namespace NUMINAMATH_GPT_triangle_area_is_integer_l1302_130205

theorem triangle_area_is_integer (x1 x2 x3 y1 y2 y3 : ℤ) 
  (hx_even : (x1 + x2 + x3) % 2 = 0) 
  (hy_even : (y1 + y2 + y3) % 2 = 0) : 
  ∃ k : ℤ, 
    abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 2 * k := 
sorry

end NUMINAMATH_GPT_triangle_area_is_integer_l1302_130205


namespace NUMINAMATH_GPT_tomato_puree_water_percentage_l1302_130282

theorem tomato_puree_water_percentage :
  (∀ (juice_purity water_percentage : ℝ), 
    (juice_purity = 0.90) → 
    (20 * juice_purity = 18) →
    (2.5 - 2) = 0.5 →
    (2.5 * water_percentage - 0.5) = 0 →
    water_percentage = 0.20) :=
by
  intros juice_purity water_percentage h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tomato_puree_water_percentage_l1302_130282


namespace NUMINAMATH_GPT_decreasing_interval_l1302_130281

noncomputable def f (x : ℝ) := Real.exp (abs (x - 1))

theorem decreasing_interval : ∀ x y : ℝ, x ≤ y → y ≤ 1 → f y ≤ f x :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_l1302_130281


namespace NUMINAMATH_GPT_hollow_circles_in_2001_pattern_l1302_130289

theorem hollow_circles_in_2001_pattern :
  let pattern_length := 9
  let hollow_in_pattern := 3
  let total_circles := 2001
  let complete_patterns := total_circles / pattern_length
  let remaining_circles := total_circles % pattern_length
  let hollow_in_remaining := if remaining_circles >= 3 then 1 else 0
  let total_hollow := complete_patterns * hollow_in_pattern + hollow_in_remaining
  total_hollow = 667 :=
by
  sorry

end NUMINAMATH_GPT_hollow_circles_in_2001_pattern_l1302_130289


namespace NUMINAMATH_GPT_probability_of_all_heads_or_tails_l1302_130213

theorem probability_of_all_heads_or_tails :
  let possible_outcomes := 256
  let favorable_outcomes := 2
  favorable_outcomes / possible_outcomes = 1 / 128 := by
  sorry

end NUMINAMATH_GPT_probability_of_all_heads_or_tails_l1302_130213


namespace NUMINAMATH_GPT_find_number_l1302_130227

theorem find_number :
  ∃ x : ℕ, (8 * x + 5400) / 12 = 530 ∧ x = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1302_130227


namespace NUMINAMATH_GPT_domain_ln_x_minus_x_sq_l1302_130259

noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

theorem domain_ln_x_minus_x_sq : { x : ℝ | x - x^2 > 0 } = { x : ℝ | 0 < x ∧ x < 1 } :=
by {
  -- These are placeholders for conditions needed in the proof
  sorry
}

end NUMINAMATH_GPT_domain_ln_x_minus_x_sq_l1302_130259


namespace NUMINAMATH_GPT_decorative_plate_painted_fraction_l1302_130291

noncomputable def fraction_painted_area (total_area painted_area : ℕ) : ℚ :=
  painted_area / total_area

theorem decorative_plate_painted_fraction :
  let side_length := 4
  let total_area := side_length * side_length
  let painted_smaller_squares := 6
  fraction_painted_area total_area painted_smaller_squares = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_decorative_plate_painted_fraction_l1302_130291


namespace NUMINAMATH_GPT_problem_statement_l1302_130263

theorem problem_statement (m n : ℝ) :
  (m^2 - 1840 * m + 2009 = 0) → (n^2 - 1840 * n + 2009 = 0) → 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_problem_statement_l1302_130263


namespace NUMINAMATH_GPT_real_solution_exists_l1302_130203

theorem real_solution_exists : ∃ x : ℝ, x^3 + (x+1)^4 + (x+2)^3 = (x+3)^4 :=
sorry

end NUMINAMATH_GPT_real_solution_exists_l1302_130203


namespace NUMINAMATH_GPT_equilateral_triangles_circle_l1302_130216

-- Definitions and conditions
structure Triangle :=
  (A B C : ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 12)

structure Circle :=
  (S : ℝ)

def PointOnArc (P1 P2 P : ℝ) : Prop :=
  -- Definition to describe P lies on the arc P1P2
  sorry

-- Theorem stating the proof problem
theorem equilateral_triangles_circle
  (S : Circle)
  (T1 T2 : Triangle)
  (H1 : T1.side_length = 12)
  (H2 : T2.side_length = 12)
  (HAonArc : PointOnArc T2.B T2.C T1.A)
  (HBonArc : PointOnArc T2.A T2.B T1.B) :
  (T1.A - T2.A) ^ 2 + (T1.B - T2.B) ^ 2 + (T1.C - T2.C) ^ 2 = 288 :=
sorry

end NUMINAMATH_GPT_equilateral_triangles_circle_l1302_130216


namespace NUMINAMATH_GPT_range_of_a_l1302_130209

theorem range_of_a (x y a : ℝ) :
  (2 * x + y ≥ 4) → 
  (x - y ≥ 1) → 
  (x - 2 * y ≤ 2) → 
  (x = 2) → 
  (y = 0) → 
  (z = a * x + y) → 
  (Ax = 2) → 
  (Ay = 0) → 
  (-1/2 < a ∧ a < 2) := sorry

end NUMINAMATH_GPT_range_of_a_l1302_130209


namespace NUMINAMATH_GPT_simplify_fraction_l1302_130221

variable {x y : ℝ}

theorem simplify_fraction (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1302_130221


namespace NUMINAMATH_GPT_vasya_number_l1302_130200

theorem vasya_number (a b c : ℕ) (h1 : 100 ≤ 100*a + 10*b + c) (h2 : 100*a + 10*b + c < 1000) 
  (h3 : a + c = 1) (h4 : a * b = 4) (h5 : a ≠ 0) : 100*a + 10*b + c = 140 :=
by
  sorry

end NUMINAMATH_GPT_vasya_number_l1302_130200


namespace NUMINAMATH_GPT_ratio_of_3_numbers_l1302_130252

variable (A B C : ℕ)
variable (k : ℕ)

theorem ratio_of_3_numbers (h₁ : A = 5 * k) (h₂ : B = k) (h₃ : C = 4 * k) (h_sum : A + B + C = 1000) : C = 400 :=
  sorry

end NUMINAMATH_GPT_ratio_of_3_numbers_l1302_130252


namespace NUMINAMATH_GPT_geometric_sequence_sum_range_l1302_130297

theorem geometric_sequence_sum_range (a b c : ℝ) 
  (h1 : ∃ q : ℝ, q ≠ 0 ∧ a = b * q ∧ c = b / q) 
  (h2 : a + b + c = 1) : 
  a + c ∈ (Set.Icc (2 / 3 : ℝ) 1 \ Set.Iio 1) ∪ (Set.Ioo 1 2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_range_l1302_130297


namespace NUMINAMATH_GPT_box_volume_l1302_130261

structure Box where
  L : ℝ  -- Length
  W : ℝ  -- Width
  H : ℝ  -- Height

def front_face_area (box : Box) : ℝ := box.L * box.H
def top_face_area (box : Box) : ℝ := box.L * box.W
def side_face_area (box : Box) : ℝ := box.H * box.W

noncomputable def volume (box : Box) : ℝ := box.L * box.W * box.H

theorem box_volume (box : Box)
  (h1 : front_face_area box = 0.5 * top_face_area box)
  (h2 : top_face_area box = 1.5 * side_face_area box)
  (h3 : side_face_area box = 72) :
  volume box = 648 := by
  sorry

end NUMINAMATH_GPT_box_volume_l1302_130261


namespace NUMINAMATH_GPT_y_intercept_of_line_l1302_130299

theorem y_intercept_of_line :
  ∃ y, (∀ x : ℝ, 2 * x - 3 * y = 6) ∧ (y = -2) :=
sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1302_130299


namespace NUMINAMATH_GPT_Luca_milk_water_needed_l1302_130228

def LucaMilk (flour : ℕ) : ℕ := (flour / 250) * 50
def LucaWater (flour : ℕ) : ℕ := (flour / 250) * 30

theorem Luca_milk_water_needed (flour : ℕ) (h : flour = 1250) : LucaMilk flour = 250 ∧ LucaWater flour = 150 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_Luca_milk_water_needed_l1302_130228


namespace NUMINAMATH_GPT_fraction_of_shoppers_avoiding_checkout_l1302_130211

theorem fraction_of_shoppers_avoiding_checkout 
  (total_shoppers : ℕ) 
  (shoppers_at_checkout : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : shoppers_at_checkout = 180) : 
  (total_shoppers - shoppers_at_checkout) / total_shoppers = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_shoppers_avoiding_checkout_l1302_130211


namespace NUMINAMATH_GPT_range_of_a_l1302_130270

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 > 0) ↔ (0 ≤ a ∧ a < 12) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1302_130270


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l1302_130253

theorem cylinder_volume_ratio (h1 h2 r1 r2 V1 V2 : ℝ)
  (h1_eq : h1 = 9)
  (h2_eq : h2 = 6)
  (circumference1_eq : 2 * π * r1 = 6)
  (circumference2_eq : 2 * π * r2 = 9)
  (V1_eq : V1 = π * r1^2 * h1)
  (V2_eq : V2 = π * r2^2 * h2)
  (V1_calculated : V1 = 81 / π)
  (V2_calculated : V2 = 243 / (4 * π)) :
  (max V1 V2) / (min V1 V2) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l1302_130253


namespace NUMINAMATH_GPT_shopkeeper_profit_percentage_l1302_130224

theorem shopkeeper_profit_percentage (C : ℝ) (hC : C > 0) :
  let selling_price := 12 * C
  let cost_price := 10 * C
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percentage_l1302_130224


namespace NUMINAMATH_GPT_count_4_digit_numbers_divisible_by_13_l1302_130273

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end NUMINAMATH_GPT_count_4_digit_numbers_divisible_by_13_l1302_130273


namespace NUMINAMATH_GPT_number_is_seven_l1302_130231

theorem number_is_seven (x : ℝ) (h : x^2 + 120 = (x - 20)^2) : x = 7 := 
by
  sorry

end NUMINAMATH_GPT_number_is_seven_l1302_130231


namespace NUMINAMATH_GPT_sandwiches_bought_is_2_l1302_130222

-- The given costs and totals
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87
def total_cost : ℝ := 10.46
def sodas_bought : ℕ := 4

-- We need to prove that the number of sandwiches bought, S, is 2
theorem sandwiches_bought_is_2 (S : ℕ) :
  sandwich_cost * S + soda_cost * sodas_bought = total_cost → S = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sandwiches_bought_is_2_l1302_130222


namespace NUMINAMATH_GPT_exist_positive_integers_for_perfect_squares_l1302_130242

theorem exist_positive_integers_for_perfect_squares :
  ∃ (x y : ℕ), (0 < x ∧ 0 < y) ∧ (∃ a b c : ℕ, x + y = a^2 ∧ x^2 + y^2 = b^2 ∧ x^3 + y^3 = c^2) :=
by
  sorry

end NUMINAMATH_GPT_exist_positive_integers_for_perfect_squares_l1302_130242


namespace NUMINAMATH_GPT_no_integer_roots_l1302_130285

theorem no_integer_roots (n : ℕ) (p : Fin (2*n + 1) → ℤ)
  (non_zero : ∀ i, p i ≠ 0)
  (sum_non_zero : (Finset.univ.sum (λ i => p i)) ≠ 0) :
  ∃ P : ℤ → ℤ, ∀ x : ℤ, P x ≠ 0 → x > 1 ∨ x < -1 := sorry

end NUMINAMATH_GPT_no_integer_roots_l1302_130285


namespace NUMINAMATH_GPT_math_proof_l1302_130258

-- Definitions
def U := Set ℝ
def A : Set ℝ := {x | x ≥ 3}
def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem
theorem math_proof (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | x ≥ 1}) ∧
  (C a ∪ A = A → a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_math_proof_l1302_130258


namespace NUMINAMATH_GPT_probability_red_jelly_bean_l1302_130284

variable (r b g : Nat) (eaten_green eaten_blue : Nat)

theorem probability_red_jelly_bean
    (h_r : r = 15)
    (h_b : b = 20)
    (h_g : g = 16)
    (h_eaten_green : eaten_green = 1)
    (h_eaten_blue : eaten_blue = 1)
    (h_total : r + b + g = 51)
    (h_remaining_total : r + (b - eaten_blue) + (g - eaten_green) = 49) :
    (r : ℚ) / 49 = 15 / 49 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_jelly_bean_l1302_130284


namespace NUMINAMATH_GPT_no_positive_integer_makes_expression_integer_l1302_130246

theorem no_positive_integer_makes_expression_integer : 
  ∀ n : ℕ, n > 0 → ¬ ∃ k : ℤ, (n^(3 * n - 2) - 3 * n + 1) = k * (3 * n - 2) := 
by 
  intro n hn
  sorry

end NUMINAMATH_GPT_no_positive_integer_makes_expression_integer_l1302_130246


namespace NUMINAMATH_GPT_cos_sin_15_deg_l1302_130237

theorem cos_sin_15_deg :
  400 * (Real.cos (15 * Real.pi / 180))^5 +  (Real.sin (15 * Real.pi / 180))^5 / (Real.cos (15 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 100 := 
sorry

end NUMINAMATH_GPT_cos_sin_15_deg_l1302_130237


namespace NUMINAMATH_GPT_smallest_number_divisible_by_6_in_permutations_list_l1302_130288

def is_divisible_by_6 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 6 * k)

noncomputable def permutations_5_digits := 
  [1, 2, 3, 4, 5].permutations.map (λ l => l.foldl (λ acc x => 10 * acc + x) 0)

theorem smallest_number_divisible_by_6_in_permutations_list :
  ∃ n ∈ permutations_5_digits, is_divisible_by_6 n ∧ (∀ m ∈ permutations_5_digits, is_divisible_by_6 m → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_6_in_permutations_list_l1302_130288


namespace NUMINAMATH_GPT_inheritance_calculation_l1302_130239

theorem inheritance_calculation
  (x : ℝ)
  (h1 : 0.25 * x + 0.15 * (0.75 * x) = 14000) :
  x = 38600 := by
  sorry

end NUMINAMATH_GPT_inheritance_calculation_l1302_130239


namespace NUMINAMATH_GPT_height_average_inequality_l1302_130238

theorem height_average_inequality 
    (a b c d : ℝ)
    (h1 : 3 * a + 2 * b = 2 * c + 3 * d)
    (h2 : a > d) : 
    (|c + d| / 2 > |a + b| / 2) :=
sorry

end NUMINAMATH_GPT_height_average_inequality_l1302_130238


namespace NUMINAMATH_GPT_intersection_eq_l1302_130287

open Set

def setA : Set ℤ := {x | x ≥ -4}
def setB : Set ℤ := {x | x ≤ 3}

theorem intersection_eq : (setA ∩ setB) = {x | -4 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1302_130287


namespace NUMINAMATH_GPT_height_difference_l1302_130294

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_height_difference_l1302_130294


namespace NUMINAMATH_GPT_max_n_value_l1302_130272

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 1/(a - b) + 1/(b - c) ≥ n / (a - c)) :
  n ≤ 4 := 
sorry

end NUMINAMATH_GPT_max_n_value_l1302_130272


namespace NUMINAMATH_GPT_scalene_triangle_area_l1302_130295

theorem scalene_triangle_area (outer_triangle_area : ℝ) (hexagon_area : ℝ) (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25) (h2 : hexagon_area = 4) (h3 : num_scalene_triangles = 6) : 
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_scalene_triangle_area_l1302_130295


namespace NUMINAMATH_GPT_John_took_more_chickens_than_Ray_l1302_130267

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end NUMINAMATH_GPT_John_took_more_chickens_than_Ray_l1302_130267


namespace NUMINAMATH_GPT_probability_of_selection_l1302_130245

noncomputable def probability_selected (total_students : ℕ) (excluded_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / (total_students - excluded_students)

theorem probability_of_selection :
  probability_selected 2008 8 50 = 25 / 1004 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selection_l1302_130245


namespace NUMINAMATH_GPT_diana_owes_l1302_130210

-- Define the conditions
def initial_charge : ℝ := 60
def annual_interest_rate : ℝ := 0.06
def time_in_years : ℝ := 1

-- Define the simple interest calculation
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

-- Define the total amount owed calculation
def total_amount_owed (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

-- State the theorem: Diana will owe $63.60 after one year
theorem diana_owes : total_amount_owed initial_charge (simple_interest initial_charge annual_interest_rate time_in_years) = 63.60 :=
by sorry

end NUMINAMATH_GPT_diana_owes_l1302_130210


namespace NUMINAMATH_GPT_f_at_7_l1302_130271

-- Define the function f and its properties
axiom f : ℝ → ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom values_f : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- Prove that f(7) = -2
theorem f_at_7 : f 7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_at_7_l1302_130271


namespace NUMINAMATH_GPT_inequality_proof_l1302_130277

theorem inequality_proof (a b c d : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a + b = 2 → c + d = 2 → 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := 
by 
  intros ha hb hc hd hab hcd
  sorry

end NUMINAMATH_GPT_inequality_proof_l1302_130277
