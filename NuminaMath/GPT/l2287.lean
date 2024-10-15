import Mathlib

namespace NUMINAMATH_GPT_find_a_plus_b_l2287_228748

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0) (h : x = a + Real.sqrt b) 
  (hx : x^2 + 3 * x + ↑(3) / x + 1 / x^2 = 30) : 
  a + b = 5 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l2287_228748


namespace NUMINAMATH_GPT_g_neg_one_l2287_228723

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h₀ : ∀ x : ℝ, f (-x) + x^2 = -(f x + x^2))
variables (h₁ : f 1 = 1)
variables (h₂ : ∀ x : ℝ, g x = f x + 2)

theorem g_neg_one : g (-1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_g_neg_one_l2287_228723


namespace NUMINAMATH_GPT_solve_inequalities_l2287_228763

theorem solve_inequalities :
  {x : ℤ | (x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - 5 < -3 * x} = {-1, 0} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l2287_228763


namespace NUMINAMATH_GPT_total_sum_spent_l2287_228736

theorem total_sum_spent (b gift : ℝ) (friends tanya : ℕ) (extra_payment : ℝ)
  (h1 : friends = 10)
  (h2 : tanya = 1)
  (h3 : extra_payment = 3)
  (h4 : gift = 15)
  (h5 : b = 270)
  : (b + gift) = 285 :=
by {
  -- Given:
  -- friends = 10 (number of dinner friends),
  -- tanya = 1 (Tanya who forgot to pay),
  -- extra_payment = 3 (extra payment by each of the remaining 9 friends),
  -- gift = 15 (cost of the gift),
  -- b = 270 (total bill for the dinner excluding the gift),

  -- We need to prove:
  -- total sum spent by the group is $285, i.e., (b + gift) = 285

  sorry 
}

end NUMINAMATH_GPT_total_sum_spent_l2287_228736


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l2287_228790

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 3 * b + 3 * c = a ^ 2) (h2 : a + 3 * b - 3 * c = -4) 
  (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : a + b > c) (h7 : a + c > b) (h8 : b + c > a) : 
  ∃ C : ℝ, C = 120 ∧ (by exact sorry) := sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l2287_228790


namespace NUMINAMATH_GPT_cubic_roots_expression_l2287_228727

noncomputable def polynomial : Polynomial ℂ :=
  Polynomial.X^3 - 3 * Polynomial.X - 2

theorem cubic_roots_expression (α β γ : ℂ)
  (h1 : (Polynomial.X - Polynomial.C α) * 
        (Polynomial.X - Polynomial.C β) * 
        (Polynomial.X - Polynomial.C γ) = polynomial) :
  α * (β - γ)^2 + β * (γ - α)^2 + γ * (α - β)^2 = -18 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_expression_l2287_228727


namespace NUMINAMATH_GPT_pairs_of_integers_l2287_228795

-- The main theorem to prove:
theorem pairs_of_integers (x y : ℤ) :
  y ^ 2 = x ^ 3 + 16 ↔ (x = 0 ∧ (y = 4 ∨ y = -4)) :=
by sorry

end NUMINAMATH_GPT_pairs_of_integers_l2287_228795


namespace NUMINAMATH_GPT_bread_rolls_count_l2287_228798

theorem bread_rolls_count (total_items croissants bagels : Nat) 
  (h1 : total_items = 90) 
  (h2 : croissants = 19) 
  (h3 : bagels = 22) : 
  total_items - croissants - bagels = 49 := 
by
  sorry

end NUMINAMATH_GPT_bread_rolls_count_l2287_228798


namespace NUMINAMATH_GPT_university_diploma_percentage_l2287_228792

variables (population : ℝ)
          (U : ℝ) -- percentage of people with a university diploma
          (J : ℝ := 0.40) -- percentage of people with the job of their choice
          (S : ℝ := 0.10) -- percentage of people with a secondary school diploma pursuing further education

-- Condition 1: 18% of the people do not have a university diploma but have the job of their choice.
-- Condition 2: 25% of the people who do not have the job of their choice have a university diploma.
-- Condition 3: 10% of the people have a secondary school diploma and are pursuing further education.
-- Condition 4: 60% of the people with secondary school diploma have the job of their choice.
-- Condition 5: 30% of the people in further education have a job of their choice as well.
-- Condition 6: 40% of the people have the job of their choice.

axiom condition_1 : 0.18 * population = (0.18 * (1 - U)) * (population)
axiom condition_2 : 0.25 * (100 - J * 100) = 0.25 * (population - J * population)
axiom condition_3 : S * population = 0.10 * population
axiom condition_4 : 0.60 * S * population = (0.60 * S) * population
axiom condition_5 : 0.30 * S * population = (0.30 * S) * population
axiom condition_6 : J * population = 0.40 * population

theorem university_diploma_percentage : U * 100 = 37 :=
by sorry

end NUMINAMATH_GPT_university_diploma_percentage_l2287_228792


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l2287_228775

theorem sum_of_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 930) : x + (x + 1) = 61 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l2287_228775


namespace NUMINAMATH_GPT_calculate_product_l2287_228701

theorem calculate_product (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3*x1*y1^2 = 2030)
  (h2 : y1^3 - 3*x1^2*y1 = 2029)
  (h3 : x2^3 - 3*x2*y2^2 = 2030)
  (h4 : y2^3 - 3*x2^2*y2 = 2029)
  (h5 : x3^3 - 3*x3*y3^2 = 2030)
  (h6 : y3^3 - 3*x3^2*y3 = 2029) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 / 1015 :=
sorry

end NUMINAMATH_GPT_calculate_product_l2287_228701


namespace NUMINAMATH_GPT_range_of_a_l2287_228765

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, y = (a * (Real.cos x)^2 - 3) * (Real.sin x) ∧ y ≥ -3) 
  → a ∈ Set.Icc (-3/2 : ℝ) 12 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2287_228765


namespace NUMINAMATH_GPT_number_half_reduction_l2287_228766

/-- Define the conditions -/
def percentage_more (percent : Float) (amount : Float) : Float := amount + (percent / 100) * amount

theorem number_half_reduction (x : Float) : percentage_more 30 75 = 97.5 → (x / 2) = 97.5 → x = 195 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_half_reduction_l2287_228766


namespace NUMINAMATH_GPT_cleaning_time_together_l2287_228759

theorem cleaning_time_together (lisa_time kay_time ben_time sarah_time : ℕ)
  (h_lisa : lisa_time = 8) (h_kay : kay_time = 12) 
  (h_ben : ben_time = 16) (h_sarah : sarah_time = 24) :
  1 / ((1 / (lisa_time:ℚ)) + (1 / (kay_time:ℚ)) + (1 / (ben_time:ℚ)) + (1 / (sarah_time:ℚ))) = (16 / 5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_cleaning_time_together_l2287_228759


namespace NUMINAMATH_GPT_sin_pi_minus_alpha_l2287_228773

theorem sin_pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = 4 / 5) :
  Real.sin (Real.pi - α) = 3 / 5 := 
sorry

end NUMINAMATH_GPT_sin_pi_minus_alpha_l2287_228773


namespace NUMINAMATH_GPT_total_selling_price_16800_l2287_228712

noncomputable def total_selling_price (CP_per_toy : ℕ) : ℕ :=
  let CP_18 := 18 * CP_per_toy
  let Gain := 3 * CP_per_toy
  CP_18 + Gain

theorem total_selling_price_16800 :
  total_selling_price 800 = 16800 :=
by
  sorry

end NUMINAMATH_GPT_total_selling_price_16800_l2287_228712


namespace NUMINAMATH_GPT_gcd_condition_implies_equality_l2287_228776

theorem gcd_condition_implies_equality (a b : ℤ) (h : ∀ n : ℤ, n ≥ 1 → Int.gcd (a + n) (b + n) > 1) : a = b :=
sorry

end NUMINAMATH_GPT_gcd_condition_implies_equality_l2287_228776


namespace NUMINAMATH_GPT_induction_example_l2287_228721

theorem induction_example (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end NUMINAMATH_GPT_induction_example_l2287_228721


namespace NUMINAMATH_GPT_find_a_l2287_228779

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
    (h3 : 13 ∣ 53^2016 + a) : a = 12 := 
by 
  -- proof would be written here
  sorry

end NUMINAMATH_GPT_find_a_l2287_228779


namespace NUMINAMATH_GPT_jack_change_l2287_228786

theorem jack_change :
  let discountedCost1 := 4.50
  let discountedCost2 := 4.50
  let discountedCost3 := 5.10
  let cost4 := 7.00
  let totalDiscountedCost := discountedCost1 + discountedCost2 + discountedCost3 + cost4
  let tax := totalDiscountedCost * 0.05
  let taxRounded := 1.06 -- Tax rounded to nearest cent
  let totalCostWithTax := totalDiscountedCost + taxRounded
  let totalCostWithServiceFee := totalCostWithTax + 2.00
  let totalPayment := 20 + 10 + 4 * 1
  let change := totalPayment - totalCostWithServiceFee
  change = 9.84 :=
by
  sorry

end NUMINAMATH_GPT_jack_change_l2287_228786


namespace NUMINAMATH_GPT_donny_money_left_l2287_228732

-- Definitions based on Conditions
def initial_amount : ℝ := 78
def cost_kite : ℝ := 8
def cost_frisbee : ℝ := 9

-- Discounted cost of roller skates
def original_cost_roller_skates : ℝ := 15
def discount_rate_roller_skates : ℝ := 0.10
def discounted_cost_roller_skates : ℝ :=
  original_cost_roller_skates * (1 - discount_rate_roller_skates)

-- Cost of LEGO set with coupon
def original_cost_lego_set : ℝ := 25
def coupon_lego_set : ℝ := 5
def discounted_cost_lego_set : ℝ :=
  original_cost_lego_set - coupon_lego_set

-- Cost of puzzle with tax
def original_cost_puzzle : ℝ := 12
def tax_rate_puzzle : ℝ := 0.05
def taxed_cost_puzzle : ℝ :=
  original_cost_puzzle * (1 + tax_rate_puzzle)

-- Total cost calculated from item costs
def total_cost : ℝ :=
  cost_kite + cost_frisbee + discounted_cost_roller_skates + discounted_cost_lego_set + taxed_cost_puzzle

def money_left_after_shopping : ℝ :=
  initial_amount - total_cost

-- Prove the main statement
theorem donny_money_left : money_left_after_shopping = 14.90 := by
  sorry

end NUMINAMATH_GPT_donny_money_left_l2287_228732


namespace NUMINAMATH_GPT_ratio_expression_value_l2287_228740

theorem ratio_expression_value (a b : ℝ) (h : a / b = 4 / 1) : 
  (a - 3 * b) / (2 * a - b) = 1 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_expression_value_l2287_228740


namespace NUMINAMATH_GPT_how_many_buns_each_student_gets_l2287_228785

theorem how_many_buns_each_student_gets 
  (packages : ℕ) 
  (buns_per_package : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages = 30)
  (h3 : classes = 4)
  (h4 : students_per_class = 30) :
  (packages * buns_per_package) / (classes * students_per_class) = 2 :=
by sorry

end NUMINAMATH_GPT_how_many_buns_each_student_gets_l2287_228785


namespace NUMINAMATH_GPT_least_number_of_cars_per_work_day_l2287_228783

-- Define the conditions as constants in Lean
def paul_work_hours_per_day := 8
def jack_work_hours_per_day := 8
def paul_cars_per_hour := 2
def jack_cars_per_hour := 3

-- Define the total number of cars Paul and Jack can change in a workday
def total_cars_per_day := (paul_cars_per_hour + jack_cars_per_hour) * paul_work_hours_per_day

-- State the theorem to be proved
theorem least_number_of_cars_per_work_day : total_cars_per_day = 40 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_least_number_of_cars_per_work_day_l2287_228783


namespace NUMINAMATH_GPT_arcsin_sqrt3_div_2_eq_pi_div_3_l2287_228762

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : 
  Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 := 
by 
    sorry

end NUMINAMATH_GPT_arcsin_sqrt3_div_2_eq_pi_div_3_l2287_228762


namespace NUMINAMATH_GPT_vasim_share_l2287_228702

theorem vasim_share (x : ℝ)
  (h_ratio : ∀ (f v r : ℝ), f = 3 * x ∧ v = 5 * x ∧ r = 6 * x)
  (h_diff : 6 * x - 3 * x = 900) :
  5 * x = 1500 :=
by
  try sorry

end NUMINAMATH_GPT_vasim_share_l2287_228702


namespace NUMINAMATH_GPT_ratio_a_to_b_l2287_228791

variable (a x c d b : ℝ)
variable (h1 : d = 3 * x + c)
variable (h2 : b = 4 * x)

theorem ratio_a_to_b : a / b = -1 / 4 := by 
  sorry

end NUMINAMATH_GPT_ratio_a_to_b_l2287_228791


namespace NUMINAMATH_GPT_intersection_of_sets_l2287_228741

def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_sets : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2287_228741


namespace NUMINAMATH_GPT_commission_rate_correct_l2287_228729

variables (weekly_earnings : ℕ) (commission : ℕ) (total_earnings : ℕ) (sales : ℕ) (commission_rate : ℕ)

-- Base earnings per week without commission
def base_earnings : ℕ := 190

-- Total earnings target
def earnings_goal : ℕ := 500

-- Minimum sales required to meet the earnings goal
def sales_needed : ℕ := 7750

-- Definition of the commission as needed to meet the goal
def needed_commission : ℕ := earnings_goal - base_earnings

-- Definition of the actual commission rate
def commission_rate_per_sale : ℕ := (needed_commission * 100) / sales_needed

-- Proof goal: Show that commission_rate_per_sale is 4
theorem commission_rate_correct : commission_rate_per_sale = 4 :=
by
  sorry

end NUMINAMATH_GPT_commission_rate_correct_l2287_228729


namespace NUMINAMATH_GPT_total_sample_needed_l2287_228745

-- Given constants
def elementary_students : ℕ := 270
def junior_high_students : ℕ := 360
def senior_high_students : ℕ := 300
def junior_high_sample : ℕ := 12

-- Calculate the total number of students in the school
def total_students : ℕ := elementary_students + junior_high_students + senior_high_students

-- Define the sampling ratio based on junior high section
def sampling_ratio : ℚ := junior_high_sample / junior_high_students

-- Apply the sampling ratio to the total number of students to get the total sample size
def total_sample : ℚ := sampling_ratio * total_students

-- Prove that the total number of students that need to be sampled is 31
theorem total_sample_needed : total_sample = 31 := sorry

end NUMINAMATH_GPT_total_sample_needed_l2287_228745


namespace NUMINAMATH_GPT_quadratic_completion_l2287_228710

theorem quadratic_completion (x : ℝ) :
  2 * x^2 + 3 * x + 1 = 0 ↔ 2 * (x + 3 / 4)^2 - 1 / 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_completion_l2287_228710


namespace NUMINAMATH_GPT_remainder_q_x_plus_2_l2287_228728

def q (x : ℝ) (D E F : ℝ) : ℝ := D * x ^ 6 + E * x ^ 4 + F * x ^ 2 + 5

theorem remainder_q_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 13) : q (-2) D E F = 13 :=
by
  sorry

end NUMINAMATH_GPT_remainder_q_x_plus_2_l2287_228728


namespace NUMINAMATH_GPT_probability_of_odd_divisor_l2287_228703

noncomputable def factorial_prime_factors : ℕ → List (ℕ × ℕ)
| 21 => [(2, 18), (3, 9), (5, 4), (7, 3), (11, 1), (13, 1), (17, 1), (19, 1)]
| _ => []

def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

def number_of_odd_factors (factors : List (ℕ × ℕ)) : ℕ :=
  number_of_factors (factors.filter (λ ⟨p, _⟩ => p != 2))

theorem probability_of_odd_divisor : (number_of_odd_factors (factorial_prime_factors 21)) /
(number_of_factors (factorial_prime_factors 21)) = 1 / 19 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_odd_divisor_l2287_228703


namespace NUMINAMATH_GPT_willie_gave_emily_7_stickers_l2287_228706

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end NUMINAMATH_GPT_willie_gave_emily_7_stickers_l2287_228706


namespace NUMINAMATH_GPT_find_number_l2287_228794

theorem find_number (n : ℕ) (h : 2 * 2 + n = 6) : n = 2 := by
  sorry

end NUMINAMATH_GPT_find_number_l2287_228794


namespace NUMINAMATH_GPT_stream_speed_is_one_l2287_228726

noncomputable def speed_of_stream (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_is_one : speed_of_stream 10 8 = 1 := by
  sorry

end NUMINAMATH_GPT_stream_speed_is_one_l2287_228726


namespace NUMINAMATH_GPT_can_form_triangle_l2287_228772

theorem can_form_triangle : Prop :=
  ∃ (a b c : ℝ), 
    (a = 8 ∧ b = 6 ∧ c = 4) ∧
    (a + b > c ∧ a + c > b ∧ b + c > a)

#check can_form_triangle

end NUMINAMATH_GPT_can_form_triangle_l2287_228772


namespace NUMINAMATH_GPT_find_x_l2287_228758

theorem find_x (t : ℤ) : 
∃ x : ℤ, (x % 7 = 3) ∧ (x^2 % 49 = 44) ∧ (x^3 % 343 = 111) ∧ (x = 343 * t + 17) :=
sorry

end NUMINAMATH_GPT_find_x_l2287_228758


namespace NUMINAMATH_GPT_solve_for_x_l2287_228718

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_solve_for_x_l2287_228718


namespace NUMINAMATH_GPT_least_possible_square_area_l2287_228737

theorem least_possible_square_area (measured_length : ℝ) (h : measured_length = 7) : 
  ∃ (actual_length : ℝ), 6.5 ≤ actual_length ∧ actual_length < 7.5 ∧ 
  (∀ (side : ℝ), 6.5 ≤ side ∧ side < 7.5 → side * side ≥ actual_length * actual_length) ∧ 
  actual_length * actual_length = 42.25 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_square_area_l2287_228737


namespace NUMINAMATH_GPT_digits_sum_is_31_l2287_228725

noncomputable def digits_sum_proof (A B C D E F G : ℕ) : Prop :=
  (1000 * A + 100 * B + 10 * C + D + 100 * E + 10 * F + G = 2020) ∧ 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧
  (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧
  (E ≠ F) ∧ (E ≠ G) ∧
  (F ≠ G)

theorem digits_sum_is_31 (A B C D E F G : ℕ) (h : digits_sum_proof A B C D E F G) : 
  A + B + C + D + E + F + G = 31 :=
sorry

end NUMINAMATH_GPT_digits_sum_is_31_l2287_228725


namespace NUMINAMATH_GPT_simplify_complex_fraction_l2287_228731

theorem simplify_complex_fraction :
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  numerator / denominator = (31 / 13 : ℂ) - (1 / 13) * I :=
by
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l2287_228731


namespace NUMINAMATH_GPT_maximum_value_of_transformed_function_l2287_228708

theorem maximum_value_of_transformed_function (a b : ℝ) (h_max : ∀ x : ℝ, a * (Real.cos x) + b ≤ 1)
  (h_min : ∀ x : ℝ, a * (Real.cos x) + b ≥ -7) : 
  ∃ ab : ℝ, (ab = 3 + a * b * (Real.sin x)) ∧ (∀ x : ℝ, ab ≤ 15) :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_transformed_function_l2287_228708


namespace NUMINAMATH_GPT_odd_function_inequality_solution_l2287_228761

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x - 2 else -(x - 2)

theorem odd_function_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by
  -- A placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_odd_function_inequality_solution_l2287_228761


namespace NUMINAMATH_GPT_seq_properties_l2287_228764

-- Conditions for the sequence a_n
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * a n + 1

-- The statements to prove given the sequence definition
theorem seq_properties (a : ℕ → ℝ) (h : seq a) :
  (∀ n, a (n + 1) ≥ 2 * a n) ∧
  (∀ n, a (n + 1) / a n ≥ a n) ∧
  (∀ n, a n ≥ n * n - 2 * n + 2) :=
by
  sorry

end NUMINAMATH_GPT_seq_properties_l2287_228764


namespace NUMINAMATH_GPT_electric_blankets_sold_l2287_228719

theorem electric_blankets_sold (T H E : ℕ)
  (h1 : 2 * T + 6 * H + 10 * E = 1800)
  (h2 : T = 7 * H)
  (h3 : H = 2 * E) : 
  E = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_electric_blankets_sold_l2287_228719


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l2287_228782

variable (S T I : ℝ)

theorem fraction_of_income_from_tips (h1 : T = (5 / 2) * S) (h2 : I = S + T) : 
  T / I = 5 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l2287_228782


namespace NUMINAMATH_GPT_range_of_m_l2287_228738

theorem range_of_m (m x : ℝ) (h₁ : (x / (x - 3) - 2 = m / (x - 3))) (h₂ : x ≠ 3) : x > 0 ↔ m < 6 ∧ m ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2287_228738


namespace NUMINAMATH_GPT_count_valid_words_l2287_228707

def total_words (n : ℕ) : ℕ := 25 ^ n

def words_with_no_A (n : ℕ) : ℕ := 24 ^ n

def words_with_one_A (n : ℕ) : ℕ := n * 24 ^ (n - 1)

def words_with_less_than_two_As : ℕ :=
  (words_with_no_A 2) + (2 * 24) +
  (words_with_no_A 3) + (3 * 24 ^ 2) +
  (words_with_no_A 4) + (4 * 24 ^ 3) +
  (words_with_no_A 5) + (5 * 24 ^ 4)

def valid_words : ℕ :=
  (total_words 1 + total_words 2 + total_words 3 + total_words 4 + total_words 5) -
  words_with_less_than_two_As

theorem count_valid_words : valid_words = sorry :=
by sorry

end NUMINAMATH_GPT_count_valid_words_l2287_228707


namespace NUMINAMATH_GPT_icosahedron_probability_div_by_three_at_least_one_fourth_l2287_228747
open ProbabilityTheory

theorem icosahedron_probability_div_by_three_at_least_one_fourth (a b c : ℕ) (h : a + b + c = 20) :
  (a^3 + b^3 + c^3 + 6 * a * b * c : ℚ) / (a + b + c)^3 ≥ 1 / 4 :=
sorry

end NUMINAMATH_GPT_icosahedron_probability_div_by_three_at_least_one_fourth_l2287_228747


namespace NUMINAMATH_GPT_find_number_l2287_228722

def exceeding_condition (x : ℝ) : Prop :=
  x = 0.16 * x + 84

theorem find_number : ∃ x : ℝ, exceeding_condition x ∧ x = 100 :=
by
  -- Proof goes here, currently omitted.
  sorry

end NUMINAMATH_GPT_find_number_l2287_228722


namespace NUMINAMATH_GPT_inconsistent_fractions_l2287_228754

theorem inconsistent_fractions : (3 / 5 : ℚ) + (17 / 20 : ℚ) > 1 := by
  sorry

end NUMINAMATH_GPT_inconsistent_fractions_l2287_228754


namespace NUMINAMATH_GPT_min_rectangle_perimeter_l2287_228757

theorem min_rectangle_perimeter (x y : ℤ) (h1 : x * y = 50) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y, x * y = 50 → 2 * (x + y) ≥ 30) ∧ 
  ∃ x y, x * y = 50 ∧ 2 * (x + y) = 30 := 
by sorry

end NUMINAMATH_GPT_min_rectangle_perimeter_l2287_228757


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2287_228734

theorem minimum_value_of_expression (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ (x y z : ℝ), (x^2 + (y - 1)^2 + z^2) = 18 / 7 ∧ y = -2 / 7 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2287_228734


namespace NUMINAMATH_GPT_bryden_receives_10_dollars_l2287_228743

theorem bryden_receives_10_dollars 
  (collector_rate : ℝ := 5)
  (num_quarters : ℝ := 4)
  (face_value_per_quarter : ℝ := 0.50) :
  collector_rate * num_quarters * face_value_per_quarter = 10 :=
by
  sorry

end NUMINAMATH_GPT_bryden_receives_10_dollars_l2287_228743


namespace NUMINAMATH_GPT_range_of_varphi_l2287_228767

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ) + 1

theorem range_of_varphi (ω ϕ : ℝ) (h_ω_pos : ω > 0) (h_ϕ_bound : |ϕ| ≤ (Real.pi) / 2)
  (h_intersection : (∀ x, f x ω ϕ = -1 → (∃ k : ℤ, x = (k * Real.pi) / ω)))
  (h_f_gt_1 : (∀ x, -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ω ϕ > 1)) :
  ω = 2 → (Real.pi / 6 ≤ ϕ) ∧ (ϕ ≤ Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_varphi_l2287_228767


namespace NUMINAMATH_GPT_modified_cube_cubies_l2287_228797

structure RubiksCube :=
  (original_cubies : ℕ := 27)
  (removed_corners : ℕ := 8)
  (total_layers : ℕ := 3)
  (edges_per_layer : ℕ := 4)
  (faces_center_cubies : ℕ := 6)
  (center_cubie : ℕ := 1)

noncomputable def cubies_with_n_faces (n : ℕ) : ℕ :=
  if n = 4 then 12
  else if n = 1 then 6
  else if n = 0 then 1
  else 0

theorem modified_cube_cubies :
  (cubies_with_n_faces 4 = 12) ∧ (cubies_with_n_faces 1 = 6) ∧ (cubies_with_n_faces 0 = 1) := by
  sorry

end NUMINAMATH_GPT_modified_cube_cubies_l2287_228797


namespace NUMINAMATH_GPT_solve_for_a_l2287_228713

def g (x : ℝ) : ℝ := 5 * x - 6

theorem solve_for_a (a : ℝ) : g a = 4 → a = 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2287_228713


namespace NUMINAMATH_GPT_find_f_of_neg2_l2287_228793

theorem find_f_of_neg2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x + 1) = 9 * x ^ 2 - 6 * x + 5) : f (-2) = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_neg2_l2287_228793


namespace NUMINAMATH_GPT_infinitely_many_not_representable_l2287_228709

def can_be_represented_as_p_n_2k (c : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ c = p + n^(2 * k)

theorem infinitely_many_not_representable :
  ∃ᶠ m in at_top, ¬ can_be_represented_as_p_n_2k (2^m + 1) := 
sorry

end NUMINAMATH_GPT_infinitely_many_not_representable_l2287_228709


namespace NUMINAMATH_GPT_right_triangle_side_length_l2287_228781

theorem right_triangle_side_length (hypotenuse : ℝ) (θ : ℝ) (sin_30 : Real.sin 30 = 1 / 2) (h : θ = 30) 
  (hyp_len : hypotenuse = 10) : 
  let opposite_side := hypotenuse * Real.sin θ
  opposite_side = 5 := by
  sorry

end NUMINAMATH_GPT_right_triangle_side_length_l2287_228781


namespace NUMINAMATH_GPT_filling_time_with_ab_l2287_228751

theorem filling_time_with_ab (a b c l : ℝ) (h1 : a + b + c - l = 5 / 6) (h2 : a + c - l = 1 / 2) (h3 : b + c - l = 1 / 3) : 
  1 / (a + b) = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_filling_time_with_ab_l2287_228751


namespace NUMINAMATH_GPT_trains_cross_time_l2287_228760

noncomputable def time_to_cross : ℝ := 
  let length_train1 := 110 -- length of the first train in meters
  let length_train2 := 150 -- length of the second train in meters
  let speed_train1 := 60 * 1000 / 3600 -- speed of the first train in meters per second
  let speed_train2 := 45 * 1000 / 3600 -- speed of the second train in meters per second
  let bridge_length := 340 -- length of the bridge in meters
  let total_distance := length_train1 + length_train2 + bridge_length -- total distance to be covered
  let relative_speed := speed_train1 + speed_train2 -- relative speed in meters per second
  total_distance / relative_speed

theorem trains_cross_time :
  abs (time_to_cross - 20.57) < 0.01 :=
sorry

end NUMINAMATH_GPT_trains_cross_time_l2287_228760


namespace NUMINAMATH_GPT_find_smaller_number_l2287_228796

theorem find_smaller_number (x : ℕ) (h1 : ∃ y, y = 3 * x) (h2 : x + 3 * x = 124) : x = 31 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_find_smaller_number_l2287_228796


namespace NUMINAMATH_GPT_num_customers_did_not_tip_l2287_228733

def total_customers : Nat := 9
def total_earnings : Nat := 32
def tip_per_customer : Nat := 8
def customers_who_tipped := total_earnings / tip_per_customer
def customers_who_did_not_tip := total_customers - customers_who_tipped

theorem num_customers_did_not_tip : customers_who_did_not_tip = 5 := 
by
  -- We use the definitions provided.
  have eq1 : customers_who_tipped = 4 := by
    sorry
  have eq2 : customers_who_did_not_tip = total_customers - customers_who_tipped := by
    sorry
  have eq3 : customers_who_did_not_tip = 9 - 4 := by
    sorry
  exact eq3

end NUMINAMATH_GPT_num_customers_did_not_tip_l2287_228733


namespace NUMINAMATH_GPT_julia_total_watches_l2287_228755

namespace JuliaWatches

-- Given conditions
def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def platinum_watches : ℕ := 2 * bronze_watches
def gold_watches : ℕ := (20 * (silver_watches + platinum_watches)) / 100  -- 20 is 20% and division by 100 to get the percentage

-- Proving the total watches Julia owns after the purchase
theorem julia_total_watches : silver_watches + bronze_watches + platinum_watches + gold_watches = 228 := by
  sorry

end JuliaWatches

end NUMINAMATH_GPT_julia_total_watches_l2287_228755


namespace NUMINAMATH_GPT_part1_part2_l2287_228714

open Set

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1 / a) * x + 1

theorem part1 (x : ℝ) : f 2 (2^x) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

theorem part2 (a x : ℝ) (h : a > 2) : f a x ≥ 0 ↔ x ∈ (Iic (1/a) ∪ Ici a) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2287_228714


namespace NUMINAMATH_GPT_sum_of_digits_l2287_228788

theorem sum_of_digits (x y z w : ℕ) 
  (hxz : z + x = 10) 
  (hyz : y + z = 9) 
  (hxw : x + w = 9) 
  (hx_ne_hy : x ≠ y)
  (hx_ne_hz : x ≠ z)
  (hx_ne_hw : x ≠ w)
  (hy_ne_hz : y ≠ z)
  (hy_ne_hw : y ≠ w)
  (hz_ne_hw : z ≠ w) :
  x + y + z + w = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l2287_228788


namespace NUMINAMATH_GPT_find_speed_in_second_hour_l2287_228778

-- Define the given conditions as hypotheses
def speed_in_first_hour : ℝ := 50
def average_speed : ℝ := 55
def total_time : ℝ := 2

-- Define a function that represents the speed in the second hour
def speed_second_hour (s2 : ℝ) := 
  (speed_in_first_hour + s2) / total_time = average_speed

-- The statement to prove: the speed in the second hour is 60 km/h
theorem find_speed_in_second_hour : speed_second_hour 60 :=
by sorry

end NUMINAMATH_GPT_find_speed_in_second_hour_l2287_228778


namespace NUMINAMATH_GPT_find_m_plus_n_l2287_228716

theorem find_m_plus_n
  (m n : ℝ)
  (l1 : ∀ x y : ℝ, 2 * x + m * y + 2 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + y - 1 = 0)
  (l3 : ∀ x y : ℝ, x + n * y + 1 = 0)
  (parallel_l1_l2 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) → (2 * x + y - 1 = 0))
  (perpendicular_l1_l3 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) ∧ (x + n * y + 1 = 0) → true) :
  m + n = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l2287_228716


namespace NUMINAMATH_GPT_fuel_needed_to_empty_l2287_228777

theorem fuel_needed_to_empty (x : ℝ) 
  (h1 : (3/4) * x - (1/3) * x = 15) :
  (1/3) * x = 12 :=
by 
-- Proving the result
sorry

end NUMINAMATH_GPT_fuel_needed_to_empty_l2287_228777


namespace NUMINAMATH_GPT_seating_arrangements_exactly_two_adjacent_empty_l2287_228730

theorem seating_arrangements_exactly_two_adjacent_empty :
  let seats := 6
  let people := 3
  let arrangements := (seats.factorial / (seats - people).factorial)
  let non_adj_non_empty := ((seats - people).choose people * people.factorial)
  let all_adj_empty := ((seats - (people + 1)).choose 1 * people.factorial)
  arrangements - non_adj_non_empty - all_adj_empty = 72 := by
  sorry

end NUMINAMATH_GPT_seating_arrangements_exactly_two_adjacent_empty_l2287_228730


namespace NUMINAMATH_GPT_tickets_sold_total_l2287_228739

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end NUMINAMATH_GPT_tickets_sold_total_l2287_228739


namespace NUMINAMATH_GPT_find_value_of_reciprocal_cubic_sum_l2287_228746

theorem find_value_of_reciprocal_cubic_sum
  (a b c r s : ℝ)
  (h₁ : a + b + c = 0)
  (h₂ : a ≠ 0)
  (h₃ : b^2 - 4 * a * c ≥ 0)
  (h₄ : r ≠ 0)
  (h₅ : s ≠ 0)
  (h₆ : a * r^2 + b * r + c = 0)
  (h₇ : a * s^2 + b * s + c = 0)
  (h₈ : r + s = -b / a)
  (h₉ : r * s = -c / a) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3 * a^2 + 3 * a * b) / (a + b)^3 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_reciprocal_cubic_sum_l2287_228746


namespace NUMINAMATH_GPT_tan_alpha_eq_2_l2287_228770

theorem tan_alpha_eq_2 (α : ℝ) (h : Real.tan α = 2) : (Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 7 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_2_l2287_228770


namespace NUMINAMATH_GPT_sqrt_sum_eq_five_sqrt_three_l2287_228717

theorem sqrt_sum_eq_five_sqrt_three : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_five_sqrt_three_l2287_228717


namespace NUMINAMATH_GPT_best_coupon1_price_l2287_228749

theorem best_coupon1_price (x : ℝ) 
    (h1 : 60 ≤ x ∨ x = 60)
    (h2_1 : 25 < 0.12 * x) 
    (h2_2 : 0.12 * x > 0.2 * x - 30) :
    x = 209.95 ∨ x = 229.95 ∨ x = 249.95 :=
by sorry

end NUMINAMATH_GPT_best_coupon1_price_l2287_228749


namespace NUMINAMATH_GPT_bathroom_length_l2287_228724

theorem bathroom_length (A L W : ℝ) (h₁ : A = 8) (h₂ : W = 2) (h₃ : A = L * W) : L = 4 :=
by
  -- Skip the proof with sorry
  sorry

end NUMINAMATH_GPT_bathroom_length_l2287_228724


namespace NUMINAMATH_GPT_top_leftmost_rectangle_is_B_l2287_228752

-- Define the sides of the rectangles
structure Rectangle :=
  (w : ℕ)
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

-- Define the specific rectangles with their side values
noncomputable def rectA : Rectangle := ⟨2, 7, 4, 7⟩
noncomputable def rectB : Rectangle := ⟨0, 6, 8, 5⟩
noncomputable def rectC : Rectangle := ⟨6, 3, 1, 1⟩
noncomputable def rectD : Rectangle := ⟨8, 4, 0, 2⟩
noncomputable def rectE : Rectangle := ⟨5, 9, 3, 6⟩
noncomputable def rectF : Rectangle := ⟨7, 5, 9, 0⟩

-- Prove that Rectangle B is the top leftmost rectangle
theorem top_leftmost_rectangle_is_B :
  (rectB.w = 0 ∧ rectB.x = 6 ∧ rectB.y = 8 ∧ rectB.z = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_top_leftmost_rectangle_is_B_l2287_228752


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_exists_l2287_228784

theorem sum_of_consecutive_integers_exists : 
  ∃ k : ℕ, 150 * k + 11325 = 5827604250 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_exists_l2287_228784


namespace NUMINAMATH_GPT_value_of_expression_l2287_228769

variables {x y z w : ℝ}

theorem value_of_expression (h1 : 4 * x * z + y * w = 4) (h2 : x * w + y * z = 8) :
  (2 * x + y) * (2 * z + w) = 20 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2287_228769


namespace NUMINAMATH_GPT_dormitory_problem_l2287_228756

theorem dormitory_problem (x : ℕ) :
  9 < x ∧ x < 12
  → (x = 10 ∧ 4 * x + 18 = 58)
  ∨ (x = 11 ∧ 4 * x + 18 = 62) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_dormitory_problem_l2287_228756


namespace NUMINAMATH_GPT_f_3_minus_f_4_l2287_228774

noncomputable def f : ℝ → ℝ := sorry
axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom initial_condition : f 1 = 1

theorem f_3_minus_f_4 : f 3 - f 4 = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_3_minus_f_4_l2287_228774


namespace NUMINAMATH_GPT_fifth_inequality_l2287_228780

theorem fifth_inequality :
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < 11 / 6 :=
sorry

end NUMINAMATH_GPT_fifth_inequality_l2287_228780


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2287_228789

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2287_228789


namespace NUMINAMATH_GPT_compute_fraction_power_l2287_228704

theorem compute_fraction_power : (45000 ^ 3 / 15000 ^ 3) = 27 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_power_l2287_228704


namespace NUMINAMATH_GPT_defective_chip_ratio_l2287_228799

theorem defective_chip_ratio (defective_chips total_chips : ℕ)
  (h1 : defective_chips = 15)
  (h2 : total_chips = 60000) :
  defective_chips / total_chips = 1 / 4000 :=
by
  sorry

end NUMINAMATH_GPT_defective_chip_ratio_l2287_228799


namespace NUMINAMATH_GPT_original_number_is_10_l2287_228771

theorem original_number_is_10 (x : ℤ) (h : 2 * x + 3 = 23) : x = 10 :=
sorry

end NUMINAMATH_GPT_original_number_is_10_l2287_228771


namespace NUMINAMATH_GPT_inequality_range_l2287_228735

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 :=
  sorry

end NUMINAMATH_GPT_inequality_range_l2287_228735


namespace NUMINAMATH_GPT_square_perimeter_eq_16_l2287_228711

theorem square_perimeter_eq_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_perimeter_eq_16_l2287_228711


namespace NUMINAMATH_GPT_proof_problem_l2287_228750

def g : ℕ → ℕ := sorry
def g_inv : ℕ → ℕ := sorry

axiom g_inv_is_inverse : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y
axiom g_4_eq_6 : g 4 = 6
axiom g_6_eq_2 : g 6 = 2
axiom g_3_eq_7 : g 3 = 7

theorem proof_problem :
  g_inv (g_inv 7 + g_inv 6) = 3 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2287_228750


namespace NUMINAMATH_GPT_area_OBEC_is_19_5_l2287_228787

-- Definitions for the points and lines from the conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨5, 0⟩
def B : Point := ⟨0, 15⟩
def C : Point := ⟨6, 0⟩
def E : Point := ⟨3, 6⟩

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * |(P1.x * P2.y + P2.x * P3.y + P3.x * P1.y) - (P1.y * P2.x + P2.y * P3.x + P3.y * P1.x)|

-- Definitions of the vertices of the quadrilateral
def O : Point := ⟨0, 0⟩

-- Calculating the area of triangles OCE and OBE
def OCE_area : ℝ := triangle_area O C E
def OBE_area : ℝ := triangle_area O B E

-- Total area of quadrilateral OBEC
def OBEC_area : ℝ := OCE_area + OBE_area

-- Proof statement: The area of quadrilateral OBEC is 19.5
theorem area_OBEC_is_19_5 : OBEC_area = 19.5 := sorry

end NUMINAMATH_GPT_area_OBEC_is_19_5_l2287_228787


namespace NUMINAMATH_GPT_length_of_AB_l2287_228744

/-- A triangle ABC lies between two parallel lines where AC = 5 cm. Prove that AB = 10 cm. -/
noncomputable def triangle_is_between_two_parallel_lines : Prop := sorry

noncomputable def segmentAC : ℝ := 5

theorem length_of_AB :
  ∃ (AB : ℝ), triangle_is_between_two_parallel_lines ∧ segmentAC = 5 ∧ AB = 10 :=
sorry

end NUMINAMATH_GPT_length_of_AB_l2287_228744


namespace NUMINAMATH_GPT_angle_B_in_parallelogram_l2287_228715

variable (A B : ℝ)

theorem angle_B_in_parallelogram (h_parallelogram : ∀ {A B C D : ℝ}, A + B = 180 ↔ A = B) 
  (h_A : A = 50) : B = 130 := by
  sorry

end NUMINAMATH_GPT_angle_B_in_parallelogram_l2287_228715


namespace NUMINAMATH_GPT_inequality_transform_l2287_228705

theorem inequality_transform (x y : ℝ) (h : y > x) : 2 * y > 2 * x := 
  sorry

end NUMINAMATH_GPT_inequality_transform_l2287_228705


namespace NUMINAMATH_GPT_max_sum_x_y_under_condition_l2287_228768

-- Define the conditions
variables (x y : ℝ)

-- State the problem and what needs to be proven
theorem max_sum_x_y_under_condition : 
  (3 * (x^2 + y^2) = x - y) → (x + y) ≤ (1 / Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_max_sum_x_y_under_condition_l2287_228768


namespace NUMINAMATH_GPT_solve_for_x_l2287_228700

theorem solve_for_x (x : ℝ) (h : 4 / (1 + 3 / x) = 1) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2287_228700


namespace NUMINAMATH_GPT_gym_membership_total_cost_l2287_228742

-- Definitions for the conditions stated in the problem
def first_gym_monthly_fee : ℕ := 10
def first_gym_signup_fee : ℕ := 50
def first_gym_discount_rate : ℕ := 10
def first_gym_personal_training_cost : ℕ := 25
def first_gym_sessions_per_year : ℕ := 52

def second_gym_multiplier : ℕ := 3
def second_gym_monthly_fee : ℕ := 3 * first_gym_monthly_fee
def second_gym_signup_fee_multiplier : ℕ := 4
def second_gym_discount_rate : ℕ := 10
def second_gym_personal_training_cost : ℕ := 45
def second_gym_sessions_per_year : ℕ := 52

-- Proof of the total amount John paid in the first year
theorem gym_membership_total_cost:
  let first_gym_annual_cost := (first_gym_monthly_fee * 12) +
                                (first_gym_signup_fee * (100 - first_gym_discount_rate) / 100) +
                                (first_gym_personal_training_cost * first_gym_sessions_per_year)
  let second_gym_annual_cost := (second_gym_monthly_fee * 12) +
                                (second_gym_monthly_fee * second_gym_signup_fee_multiplier * (100 - second_gym_discount_rate) / 100) +
                                (second_gym_personal_training_cost * second_gym_sessions_per_year)
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  total_annual_cost = 4273 := by
  -- Declaration of the variables used in the problem
  let first_gym_annual_cost := 1465
  let second_gym_annual_cost := 2808
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  -- Simplify and verify the total cost
  sorry

end NUMINAMATH_GPT_gym_membership_total_cost_l2287_228742


namespace NUMINAMATH_GPT_player_A_always_wins_l2287_228720

theorem player_A_always_wins (a b c : ℤ) :
  ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (x - x1) * (x - x2) * (x - x3) = x^3 + a*x^2 + b*x + c :=
sorry

end NUMINAMATH_GPT_player_A_always_wins_l2287_228720


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l2287_228753

variable (x : ℝ)

theorem solve_eq1 : (2 * x - 3 * (2 * x - 3) = x + 4) → (x = 1) :=
by
  intro h
  sorry

theorem solve_eq2 : ((3 / 4 * x - 1 / 4) - 1 = (5 / 6 * x - 7 / 6)) → (x = -1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l2287_228753
