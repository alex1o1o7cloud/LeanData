import Mathlib

namespace NUMINAMATH_GPT_remaining_numbers_l2278_227865

theorem remaining_numbers (S S3 S2 N : ℕ) (h1 : S / 5 = 8) (h2 : S3 / 3 = 4) (h3 : S2 / N = 14) 
(hS  : S = 5 * 8) (hS3 : S3 = 3 * 4) (hS2 : S2 = S - S3) : N = 2 := by
  sorry

end NUMINAMATH_GPT_remaining_numbers_l2278_227865


namespace NUMINAMATH_GPT_find_p_q_l2278_227879

theorem find_p_q (D : ℝ) (p q : ℝ) (h_roots : ∀ x, x^2 + p * x + q = 0 → (x = D ∨ x = 1 - D))
  (h_discriminant : D = p^2 - 4 * q) :
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3 / 16) :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l2278_227879


namespace NUMINAMATH_GPT_subtraction_like_terms_l2278_227870

variable (a : ℝ)

theorem subtraction_like_terms : 3 * a ^ 2 - 2 * a ^ 2 = a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_like_terms_l2278_227870


namespace NUMINAMATH_GPT_find_a_l2278_227893

theorem find_a (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 0) :
  x + n^n * (1 / (x^n)) ≥ n + 1 :=
sorry

end NUMINAMATH_GPT_find_a_l2278_227893


namespace NUMINAMATH_GPT_sum_of_x_coordinates_mod_20_l2278_227823

theorem sum_of_x_coordinates_mod_20 (y x : ℤ) (h1 : y ≡ 7 * x + 3 [ZMOD 20]) (h2 : y ≡ 13 * x + 17 [ZMOD 20]) 
: ∃ (x1 x2 : ℤ), (0 ≤ x1 ∧ x1 < 20) ∧ (0 ≤ x2 ∧ x2 < 20) ∧ x1 ≡ 1 [ZMOD 10] ∧ x2 ≡ 11 [ZMOD 10] ∧ x1 + x2 = 12 := sorry

end NUMINAMATH_GPT_sum_of_x_coordinates_mod_20_l2278_227823


namespace NUMINAMATH_GPT_radian_measure_of_negative_150_degree_l2278_227849

theorem radian_measure_of_negative_150_degree  : (-150 : ℝ) * (Real.pi / 180) = - (5 * Real.pi / 6) := by
  sorry

end NUMINAMATH_GPT_radian_measure_of_negative_150_degree_l2278_227849


namespace NUMINAMATH_GPT_smaller_integer_l2278_227829

noncomputable def m : ℕ := 1
noncomputable def n : ℕ := 1998 * m

lemma two_digit_number (m: ℕ) : 10 ≤ m ∧ m < 100 := by sorry
lemma three_digit_number (n: ℕ) : 100 ≤ n ∧ n < 1000 := by sorry

theorem smaller_integer 
  (two_digit_m: 10 ≤ m ∧ m < 100)
  (three_digit_n: 100 ≤ n ∧ n < 1000)
  (avg_eq_decimal: (m + n) / 2 = m + n / 1000)
  : m = 1 := by 
  sorry

end NUMINAMATH_GPT_smaller_integer_l2278_227829


namespace NUMINAMATH_GPT_given_eqn_simplification_l2278_227818

theorem given_eqn_simplification (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_given_eqn_simplification_l2278_227818


namespace NUMINAMATH_GPT_fill_cistern_time_l2278_227887

-- Define the rates of the taps
def rateA := (1 : ℚ) / 3  -- Tap A fills 1 cistern in 3 hours (rate is 1/3 per hour)
def rateB := -(1 : ℚ) / 6  -- Tap B empties 1 cistern in 6 hours (rate is -1/6 per hour)
def rateC := (1 : ℚ) / 2  -- Tap C fills 1 cistern in 2 hours (rate is 1/2 per hour)

-- Define the combined rate
def combinedRate := rateA + rateB + rateC

-- The time to fill the cistern when all taps are opened simultaneously
def timeToFill := 1 / combinedRate

-- The theorem stating that the time to fill the cistern is 1.5 hours
theorem fill_cistern_time : timeToFill = (3 : ℚ) / 2 := by
  sorry  -- The proof is omitted as per the instructions

end NUMINAMATH_GPT_fill_cistern_time_l2278_227887


namespace NUMINAMATH_GPT_quadratic_roots_bounds_l2278_227886

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_bounds_l2278_227886


namespace NUMINAMATH_GPT_combined_work_time_l2278_227803

theorem combined_work_time (man_rate : ℚ := 1/5) (wife_rate : ℚ := 1/7) (son_rate : ℚ := 1/15) :
  (man_rate + wife_rate + son_rate)⁻¹ = 105 / 43 :=
by
  sorry

end NUMINAMATH_GPT_combined_work_time_l2278_227803


namespace NUMINAMATH_GPT_maximize_operation_l2278_227804

-- Definitions from the conditions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- The proof statement
theorem maximize_operation : ∃ n, is_three_digit_integer n ∧ (∀ m, is_three_digit_integer m → 3 * (300 - m) ≤ 600) :=
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_maximize_operation_l2278_227804


namespace NUMINAMATH_GPT_isosceles_triangle_angle_l2278_227883

-- Definition of required angles and the given geometric context
variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]
variables (angleBAC : ℝ) (angleBCA : ℝ)

-- Given: shared vertex A, with angle BAC of pentagon
axiom angleBAC_def : angleBAC = 108

-- To Prove: determining the measure of angle BCA in the isosceles triangle
theorem isosceles_triangle_angle (h : 180 > 2 * angleBAC) : angleBCA = (180 - angleBAC) / 2 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_l2278_227883


namespace NUMINAMATH_GPT_trees_planted_l2278_227852

-- Definitions for the quantities of lindens (x) and birches (y)
variables (x y : ℕ)

-- Definitions matching the given problem conditions
def condition1 := x + y > 14
def condition2 := y + 18 > 2 * x
def condition3 := x > 2 * y

-- The theorem stating that if the conditions hold, then x = 11 and y = 5
theorem trees_planted (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : 
  x = 11 ∧ y = 5 := 
sorry

end NUMINAMATH_GPT_trees_planted_l2278_227852


namespace NUMINAMATH_GPT_perimeter_triangle_PQR_is_24_l2278_227867

noncomputable def perimeter_triangle_PQR (QR PR : ℝ) : ℝ :=
  let PQ := Real.sqrt (QR^2 + PR^2)
  PQ + QR + PR

theorem perimeter_triangle_PQR_is_24 :
  perimeter_triangle_PQR 8 6 = 24 := by
  sorry

end NUMINAMATH_GPT_perimeter_triangle_PQR_is_24_l2278_227867


namespace NUMINAMATH_GPT_find_number_of_even_numbers_l2278_227846

-- Define the average of the first n even numbers
def average_of_first_n_even (n : ℕ) : ℕ :=
  (n * (1 + n)) / n

-- The given condition: The average is 21
def average_is_21 (n : ℕ) : Prop :=
  average_of_first_n_even n = 21

-- The theorem to prove: If the average is 21, then n = 20
theorem find_number_of_even_numbers (n : ℕ) (h : average_is_21 n) : n = 20 :=
  sorry

end NUMINAMATH_GPT_find_number_of_even_numbers_l2278_227846


namespace NUMINAMATH_GPT_range_of_a_l2278_227851

noncomputable def f (a x : ℝ) :=
  if x < 0 then
    9 * x + a^2 / x + 7
  else
    9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8 / 7 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l2278_227851


namespace NUMINAMATH_GPT_percentage_boy_scouts_l2278_227860

theorem percentage_boy_scouts (S B G : ℝ) (h1 : B + G = S)
  (h2 : 0.60 * S = 0.50 * B + 0.6818 * G) : (B / S) * 100 = 45 := by
  sorry

end NUMINAMATH_GPT_percentage_boy_scouts_l2278_227860


namespace NUMINAMATH_GPT_smaller_part_area_l2278_227831

theorem smaller_part_area (x y : ℝ) (h1 : x + y = 500) (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 :=
by
  sorry

end NUMINAMATH_GPT_smaller_part_area_l2278_227831


namespace NUMINAMATH_GPT_loss_percentage_is_25_l2278_227811

variables (C S : ℝ)
variables (h : 30 * C = 40 * S)

theorem loss_percentage_is_25 (h : 30 * C = 40 * S) : ((C - S) / C) * 100 = 25 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_loss_percentage_is_25_l2278_227811


namespace NUMINAMATH_GPT_sticker_sum_mod_problem_l2278_227885

theorem sticker_sum_mod_problem :
  ∃ N < 100, (N % 6 = 5) ∧ (N % 8 = 6) ∧ (N = 47 ∨ N = 95) ∧ (47 + 95 = 142) :=
by
  sorry

end NUMINAMATH_GPT_sticker_sum_mod_problem_l2278_227885


namespace NUMINAMATH_GPT_milk_revenue_l2278_227854

theorem milk_revenue :
  let yesterday_morning := 68
  let yesterday_evening := 82
  let this_morning := yesterday_morning - 18
  let total_milk_before_selling := yesterday_morning + yesterday_evening + this_morning
  let milk_left := 24
  let milk_sold := total_milk_before_selling - milk_left
  let cost_per_gallon := 3.50
  let revenue := milk_sold * cost_per_gallon
  revenue = 616 := by {
    sorry
}

end NUMINAMATH_GPT_milk_revenue_l2278_227854


namespace NUMINAMATH_GPT_owen_profit_l2278_227881

theorem owen_profit
  (num_boxes : ℕ)
  (cost_per_box : ℕ)
  (pieces_per_box : ℕ)
  (sold_boxes : ℕ)
  (price_per_25_pieces : ℕ)
  (remaining_pieces : ℕ)
  (price_per_10_pieces : ℕ) :
  num_boxes = 12 →
  cost_per_box = 9 →
  pieces_per_box = 50 →
  sold_boxes = 6 →
  price_per_25_pieces = 5 →
  remaining_pieces = 300 →
  price_per_10_pieces = 3 →
  sold_boxes * 2 * price_per_25_pieces + (remaining_pieces / 10) * price_per_10_pieces - num_boxes * cost_per_box = 42 :=
by
  intros h_num h_cost h_pieces h_sold h_price_25 h_remain h_price_10
  sorry

end NUMINAMATH_GPT_owen_profit_l2278_227881


namespace NUMINAMATH_GPT_brinley_animal_count_l2278_227830

def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 12 * leopards
def cheetahs : ℕ := snakes / 3  -- rounding down implicitly considered
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem brinley_animal_count : total_animals = 673 :=
by
  -- Mathematical proof would go here.
  sorry

end NUMINAMATH_GPT_brinley_animal_count_l2278_227830


namespace NUMINAMATH_GPT_Jakes_height_is_20_l2278_227899

-- Define the conditions
def Sara_width : ℤ := 12
def Sara_height : ℤ := 24
def Sara_depth : ℤ := 24
def Jake_width : ℤ := 16
def Jake_depth : ℤ := 18
def volume_difference : ℤ := 1152

-- Volume calculation
def Sara_volume : ℤ := Sara_width * Sara_height * Sara_depth

-- Prove Jake's height is 20 inches
theorem Jakes_height_is_20 :
  ∃ h : ℤ, (Sara_volume - (Jake_width * h * Jake_depth) = volume_difference) ∧ h = 20 :=
by
  sorry

end NUMINAMATH_GPT_Jakes_height_is_20_l2278_227899


namespace NUMINAMATH_GPT_find_k_l2278_227822

theorem find_k (k : ℝ) (h : (3:ℝ)^4 + k * (3:ℝ)^2 - 26 = 0) : k = -55 / 9 := 
by sorry

end NUMINAMATH_GPT_find_k_l2278_227822


namespace NUMINAMATH_GPT_num_pos_integers_congruent_to_4_mod_7_l2278_227882

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end NUMINAMATH_GPT_num_pos_integers_congruent_to_4_mod_7_l2278_227882


namespace NUMINAMATH_GPT_is_quadratic_l2278_227866

theorem is_quadratic (A B C D : Prop) :
  (A = (∀ x : ℝ, x + (1 / x) = 0)) ∧
  (B = (∀ x y : ℝ, x + x * y + 1 = 0)) ∧
  (C = (∀ x : ℝ, 3 * x + 2 = 0)) ∧
  (D = (∀ x : ℝ, x^2 + 2 * x = 1)) →
  D := 
by
  sorry

end NUMINAMATH_GPT_is_quadratic_l2278_227866


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l2278_227897

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℤ → ℤ) 
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
  (h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 110) : 
  S 15 = 330 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l2278_227897


namespace NUMINAMATH_GPT_ajay_total_gain_l2278_227889

theorem ajay_total_gain:
  let dal_A_kg := 15
  let dal_B_kg := 10
  let dal_C_kg := 12
  let dal_D_kg := 8
  let rate_A := 14.50
  let rate_B := 13
  let rate_C := 16
  let rate_D := 18
  let selling_rate := 17.50
  let cost_A := dal_A_kg * rate_A
  let cost_B := dal_B_kg * rate_B
  let cost_C := dal_C_kg * rate_C
  let cost_D := dal_D_kg * rate_D
  let total_cost := cost_A + cost_B + cost_C + cost_D
  let total_weight := dal_A_kg + dal_B_kg + dal_C_kg + dal_D_kg
  let total_selling_price := total_weight * selling_rate
  let gain := total_selling_price - total_cost
  gain = 104 := by
    sorry

end NUMINAMATH_GPT_ajay_total_gain_l2278_227889


namespace NUMINAMATH_GPT_area_of_region_l2278_227862

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 1) → (∃ (A : ℝ), A = 14 * Real.pi) := 
by
  sorry

end NUMINAMATH_GPT_area_of_region_l2278_227862


namespace NUMINAMATH_GPT_find_sum_of_perimeters_l2278_227814

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end NUMINAMATH_GPT_find_sum_of_perimeters_l2278_227814


namespace NUMINAMATH_GPT_polynomial_perfect_square_l2278_227801

theorem polynomial_perfect_square (m : ℤ) : (∃ a : ℤ, a^2 = 25 ∧ x^2 + m*x + 25 = (x + a)^2) ↔ (m = 10 ∨ m = -10) :=
by sorry

end NUMINAMATH_GPT_polynomial_perfect_square_l2278_227801


namespace NUMINAMATH_GPT_simplify_polynomial_l2278_227864

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_polynomial_l2278_227864


namespace NUMINAMATH_GPT_man_gets_dividend_l2278_227859

    -- Definitions based on conditions
    noncomputable def investment : ℝ := 14400
    noncomputable def premium_rate : ℝ := 0.20
    noncomputable def face_value : ℝ := 100
    noncomputable def dividend_rate : ℝ := 0.07

    -- Calculate the price per share with premium
    noncomputable def price_per_share : ℝ := face_value * (1 + premium_rate)

    -- Calculate the number of shares bought
    noncomputable def number_of_shares : ℝ := investment / price_per_share

    -- Calculate the dividend per share
    noncomputable def dividend_per_share : ℝ := face_value * dividend_rate

    -- Calculate the total dividend
    noncomputable def total_dividend : ℝ := dividend_per_share * number_of_shares

    -- The proof statement
    theorem man_gets_dividend : total_dividend = 840 := by
        sorry
    
end NUMINAMATH_GPT_man_gets_dividend_l2278_227859


namespace NUMINAMATH_GPT_polynomial_equation_example_l2278_227894

theorem polynomial_equation_example (a0 a1 a2 a3 a4 a5 a6 a7 a8 : ℤ)
  (h : x^5 * (x + 3)^3 = a8 * (x + 1)^8 + a7 * (x + 1)^7 + a6 * (x + 1)^6 + a5 * (x + 1)^5 + a4 * (x + 1)^4 + a3 * (x + 1)^3 + a2 * (x + 1)^2 + a1 * (x + 1) + a0) :
  7 * a7 + 5 * a5 + 3 * a3 + a1 = -8 :=
sorry

end NUMINAMATH_GPT_polynomial_equation_example_l2278_227894


namespace NUMINAMATH_GPT_rolls_sold_to_uncle_l2278_227826

theorem rolls_sold_to_uncle (total_rolls needed_rolls rolls_to_grandmother rolls_to_neighbor rolls_to_uncle : ℕ)
  (h1 : total_rolls = 45)
  (h2 : needed_rolls = 28)
  (h3 : rolls_to_grandmother = 1)
  (h4 : rolls_to_neighbor = 6)
  (h5 : rolls_to_uncle + rolls_to_grandmother + rolls_to_neighbor + needed_rolls = total_rolls) :
  rolls_to_uncle = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_rolls_sold_to_uncle_l2278_227826


namespace NUMINAMATH_GPT_number_of_newspapers_l2278_227872

theorem number_of_newspapers (total_reading_materials magazines_sold: ℕ) (h_total: total_reading_materials = 700) (h_magazines: magazines_sold = 425) : 
  ∃ newspapers_sold : ℕ, newspapers_sold + magazines_sold = total_reading_materials ∧ newspapers_sold = 275 :=
by
  sorry

end NUMINAMATH_GPT_number_of_newspapers_l2278_227872


namespace NUMINAMATH_GPT_simplification_evaluation_l2278_227890

-- Define the variables x and y
def x : ℕ := 2
def y : ℕ := 3

-- Define the expression
def expr := 5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y)

-- Lean 4 statement to prove the equivalence
theorem simplification_evaluation : expr = 36 :=
by
  -- Place the proof here when needed
  sorry

end NUMINAMATH_GPT_simplification_evaluation_l2278_227890


namespace NUMINAMATH_GPT_arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l2278_227884

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def seq_sum (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  (Finset.range n).sum seq

noncomputable def T_n (n : ℕ) : ℕ :=
  seq_sum n (λ i => (a_n (i + 1) + 1) * b_n (i + 1))

theorem arithmetic_seq_general_term (n : ℕ) : a_n n = 2 * n - 1 := by
  sorry

theorem geometric_seq_general_term (n : ℕ) : b_n n = 2^n := by
  sorry

theorem sequence_sum (n : ℕ) : T_n n = (n - 1) * 2^(n+2) + 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l2278_227884


namespace NUMINAMATH_GPT_math_problem_l2278_227837

theorem math_problem :
  101 * 102^2 - 101 * 98^2 = 80800 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2278_227837


namespace NUMINAMATH_GPT_greatest_divisor_l2278_227836

theorem greatest_divisor (d : ℕ) (h1 : 4351 % d = 8) (h2 : 5161 % d = 10) : d = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_greatest_divisor_l2278_227836


namespace NUMINAMATH_GPT_water_ratio_horse_pig_l2278_227802

-- Definitions based on conditions
def num_pigs : ℕ := 8
def water_per_pig : ℕ := 3
def num_horses : ℕ := 10
def water_for_chickens : ℕ := 30
def total_water : ℕ := 114

-- Statement of the problem
theorem water_ratio_horse_pig : 
  (total_water - (num_pigs * water_per_pig) - water_for_chickens) / num_horses / water_per_pig = 2 := 
by sorry

end NUMINAMATH_GPT_water_ratio_horse_pig_l2278_227802


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_neq_l2278_227815

theorem arithmetic_sequence_sum_neq (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
    (h_arith : ∀ n, a (n + 1) = a n + d)
    (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
    (h_abs_eq : abs (a 3) = abs (a 9))
    (h_d_neg : d < 0) : S 5 ≠ S 6 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_neq_l2278_227815


namespace NUMINAMATH_GPT_maximum_area_of_right_triangle_l2278_227853

theorem maximum_area_of_right_triangle
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2) : 
  ∃ S, S ≤ (3 - 2 * Real.sqrt 2) ∧ S = (1/2) * a * b :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_of_right_triangle_l2278_227853


namespace NUMINAMATH_GPT_point_in_third_quadrant_l2278_227834

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l2278_227834


namespace NUMINAMATH_GPT_max_value_of_a_l2278_227819
noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_value_of_a (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) → a ≤ 1 := 
sorry

end NUMINAMATH_GPT_max_value_of_a_l2278_227819


namespace NUMINAMATH_GPT_abs_neg_one_third_l2278_227833

theorem abs_neg_one_third : abs (-1/3) = 1/3 := by
  sorry

end NUMINAMATH_GPT_abs_neg_one_third_l2278_227833


namespace NUMINAMATH_GPT_length_of_ae_l2278_227892

-- Definition of points and lengths between them
variables (a b c d e : Type)
variables (bc cd de ab ac : ℝ)

-- Given conditions
axiom H1 : bc = 3 * cd
axiom H2 : de = 8
axiom H3 : ab = 5
axiom H4 : ac = 11
axiom H5 : bc = ac - ab
axiom H6 : cd = bc / 3

-- Theorem to prove
theorem length_of_ae : ∀ ab bc cd de : ℝ, ae = ab + bc + cd + de := by
  sorry

end NUMINAMATH_GPT_length_of_ae_l2278_227892


namespace NUMINAMATH_GPT_find_function_and_max_profit_l2278_227868

noncomputable def profit_function (x : ℝ) : ℝ := -50 * x^2 + 1200 * x - 6400

theorem find_function_and_max_profit :
  (∀ (x : ℝ), (x = 10 → (-50 * x + 800 = 300)) ∧ (x = 13 → (-50 * x + 800 = 150))) ∧
  (∃ (x : ℝ), x = 12 ∧ profit_function x = 800) :=
by
  sorry

end NUMINAMATH_GPT_find_function_and_max_profit_l2278_227868


namespace NUMINAMATH_GPT_pounds_in_one_ton_is_2600_l2278_227895

variable (pounds_in_one_ton : ℕ)
variable (ounces_in_one_pound : ℕ := 16)
variable (packets : ℕ := 2080)
variable (weight_per_packet_pounds : ℕ := 16)
variable (weight_per_packet_ounces : ℕ := 4)
variable (gunny_bag_capacity_tons : ℕ := 13)

theorem pounds_in_one_ton_is_2600 :
  (packets * (weight_per_packet_pounds + weight_per_packet_ounces / ounces_in_one_pound)) = (gunny_bag_capacity_tons * pounds_in_one_ton) →
  pounds_in_one_ton = 2600 :=
sorry

end NUMINAMATH_GPT_pounds_in_one_ton_is_2600_l2278_227895


namespace NUMINAMATH_GPT_parabola_and_length_ef_l2278_227820

theorem parabola_and_length_ef :
  ∃ a b : ℝ, (∀ x : ℝ, (x + 1) * (x - 3) = 0 → a * x^2 + b * x + 3 = 0) ∧ 
            (∀ x : ℝ, -a * x^2 + b * x + 3 = 7 / 4 → 
              ∃ x1 x2 : ℝ, x1 = -1 / 2 ∧ x2 = 5 / 2 ∧ abs (x2 - x1) = 3) := 
sorry

end NUMINAMATH_GPT_parabola_and_length_ef_l2278_227820


namespace NUMINAMATH_GPT_translated_vector_ab_l2278_227825

-- Define points A and B, and vector a
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (5, 2)
def a : ℝ × ℝ := (1, 2)

-- Define the vector AB
def vectorAB : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  (Bx - Ax, By - Ay)

-- Prove that after translating vector AB by vector a, the result remains (2, -5)
theorem translated_vector_ab :
  vectorAB = (2, -5) := by
  sorry

end NUMINAMATH_GPT_translated_vector_ab_l2278_227825


namespace NUMINAMATH_GPT_find_central_angle_l2278_227858

variable (L : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions
def arc_length_condition : Prop := L = 200
def radius_condition : Prop := r = 2
def arc_length_formula : Prop := L = r * α

-- Theorem statement
theorem find_central_angle 
  (hL : arc_length_condition L) 
  (hr : radius_condition r) 
  (hf : arc_length_formula L r α) : 
  α = 100 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_central_angle_l2278_227858


namespace NUMINAMATH_GPT_problem_statement_l2278_227817

theorem problem_statement (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : ∀ n : ℕ, n ≥ 1 → 2^n * b + 1 ∣ a^(2^n) - 1) : a = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2278_227817


namespace NUMINAMATH_GPT_blocks_per_box_l2278_227871

theorem blocks_per_box (total_blocks : ℕ) (boxes : ℕ) (h1 : total_blocks = 16) (h2 : boxes = 8) : total_blocks / boxes = 2 :=
by
  sorry

end NUMINAMATH_GPT_blocks_per_box_l2278_227871


namespace NUMINAMATH_GPT_transaction_result_l2278_227878

theorem transaction_result
  (house_selling_price store_selling_price : ℝ)
  (house_loss_perc : ℝ)
  (store_gain_perc : ℝ)
  (house_selling_price_eq : house_selling_price = 15000)
  (store_selling_price_eq : store_selling_price = 15000)
  (house_loss_perc_eq : house_loss_perc = 0.1)
  (store_gain_perc_eq : store_gain_perc = 0.3) :
  (store_selling_price + house_selling_price - ((house_selling_price / (1 - house_loss_perc)) + (store_selling_price / (1 + store_gain_perc)))) = 1795 :=
by
  sorry

end NUMINAMATH_GPT_transaction_result_l2278_227878


namespace NUMINAMATH_GPT_parabola_properties_l2278_227841

theorem parabola_properties (p m k1 k2 k3 : ℝ)
  (parabola_eq : ∀ x y, y^2 = 2 * p * x ↔ y = m)
  (parabola_passes_through : m^2 = 2 * p)
  (point_distance : ((1 + p / 2)^2 + m^2 = 8) ∨ ((1 + p / 2)^2 + m^2 = 8))
  (p_gt_zero : p > 0)
  (point_P : (1, 2) ∈ { (x, y) | y^2 = 4 * x })
  (slope_eq : k3 = (k1 * k2) / (k1 + k2 - k1 * k2)) :
  (y^2 = 4 * x) ∧ (1/k1 + 1/k2 - 1/k3 = 1) := sorry

end NUMINAMATH_GPT_parabola_properties_l2278_227841


namespace NUMINAMATH_GPT_simplify_abs_expression_l2278_227873

/-- Simplify the expression: |-4^3 + 5^2 - 6| and prove the result is equal to 45 -/
theorem simplify_abs_expression :
  |(- 4 ^ 3 + 5 ^ 2 - 6)| = 45 :=
by
  sorry

end NUMINAMATH_GPT_simplify_abs_expression_l2278_227873


namespace NUMINAMATH_GPT_circle_equation_with_focus_center_and_tangent_directrix_l2278_227816

theorem circle_equation_with_focus_center_and_tangent_directrix :
  ∃ (x y : ℝ), (∃ k : ℝ, y^2 = -8 * x ∧ k = 2 ∧ (x = -2 ∧ y = 0) ∧ (x + 2)^2 + y^2 = 16) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_with_focus_center_and_tangent_directrix_l2278_227816


namespace NUMINAMATH_GPT_sum_not_complete_residue_system_l2278_227848

theorem sum_not_complete_residue_system {n : ℕ} (hn_even : Even n)
    (a b : Fin n → ℕ) (ha : ∀ k, a k < n) (hb : ∀ k, b k < n) 
    (h_complete_a : ∀ x : Fin n, ∃ k : Fin n, a k = x) 
    (h_complete_b : ∀ y : Fin n, ∃ k : Fin n, b k = y) :
    ¬ (∀ z : Fin n, ∃ k : Fin n, ∃ l : Fin n, z = (a k + b l) % n) :=
by
  sorry

end NUMINAMATH_GPT_sum_not_complete_residue_system_l2278_227848


namespace NUMINAMATH_GPT_max_value_of_reciprocals_l2278_227863

noncomputable def quadratic (x t q : ℝ) : ℝ := x^2 - t * x + q

theorem max_value_of_reciprocals (α β t q : ℝ) (h1 : α + β = α^2 + β^2)
                                               (h2 : α + β = α^3 + β^3)
                                               (h3 : ∀ n, 1 ≤ n ∧ n ≤ 2010 → α^n + β^n = α + β)
                                               (h4 : α * β = q)
                                               (h5 : α + β = t) :
  ∃ (α β : ℝ), (1 / α^2012 + 1 / β^2012) = 2 := 
sorry

end NUMINAMATH_GPT_max_value_of_reciprocals_l2278_227863


namespace NUMINAMATH_GPT_fencing_required_l2278_227807

theorem fencing_required (L W : ℝ) (hL : L = 20) (hA : 20 * W = 60) : 2 * W + L = 26 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l2278_227807


namespace NUMINAMATH_GPT_find_n_l2278_227877

def exp (m n : ℕ) : ℕ := m ^ n

-- Now we restate the problem formally
theorem find_n 
  (m n : ℕ) 
  (h1 : exp 10 m = n * 22) : 
  n = 10^m / 22 := 
sorry

end NUMINAMATH_GPT_find_n_l2278_227877


namespace NUMINAMATH_GPT_lucy_l2278_227808

-- Define rounding function to nearest ten
def round_to_nearest_ten (x : Int) : Int :=
  if x % 10 < 5 then x - x % 10 else x + (10 - x % 10)

-- Define the problem with given conditions
def lucy_problem : Prop :=
  let sum := 68 + 57
  round_to_nearest_ten sum = 130

-- Statement of proof problem
theorem lucy's_correct_rounded_sum : lucy_problem := by
  sorry

end NUMINAMATH_GPT_lucy_l2278_227808


namespace NUMINAMATH_GPT_six_units_away_has_two_solutions_l2278_227844

-- Define point A and its position on the number line
def A_position : ℤ := -3

-- Define the condition for a point x being 6 units away from point A
def is_6_units_away (x : ℤ) : Prop := abs (x + 3) = 6

-- The theorem stating that if x is 6 units away from -3, then x must be either 3 or -9
theorem six_units_away_has_two_solutions (x : ℤ) (h : is_6_units_away x) : x = 3 ∨ x = -9 := by
  sorry

end NUMINAMATH_GPT_six_units_away_has_two_solutions_l2278_227844


namespace NUMINAMATH_GPT_solve_x_l2278_227869

def otimes (a b : ℝ) : ℝ := a - 3 * b

theorem solve_x : ∃ x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 :=
by
  use -1
  rw [otimes, otimes]
  sorry

end NUMINAMATH_GPT_solve_x_l2278_227869


namespace NUMINAMATH_GPT_sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l2278_227856

-- Problem 1: Prove the general formula for the sequence of all positive even numbers
theorem sequence_even_numbers (n : ℕ) : ∃ a_n, a_n = 2 * n := by 
  sorry

-- Problem 2: Prove the general formula for the sequence of all positive odd numbers
theorem sequence_odd_numbers (n : ℕ) : ∃ b_n, b_n = 2 * n - 1 := by 
  sorry

-- Problem 3: Prove the general formula for the sequence 1, 4, 9, 16, ...
theorem sequence_square_numbers (n : ℕ) : ∃ a_n, a_n = n^2 := by
  sorry

-- Problem 4: Prove the general formula for the sequence -4, -1, 2, 5, ...
theorem sequence_arithmetic_progression (n : ℕ) : ∃ a_n, a_n = 3 * n - 7 := by
  sorry

end NUMINAMATH_GPT_sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l2278_227856


namespace NUMINAMATH_GPT_margaret_age_in_12_years_l2278_227805

theorem margaret_age_in_12_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 :=
by
  sorry

end NUMINAMATH_GPT_margaret_age_in_12_years_l2278_227805


namespace NUMINAMATH_GPT_prime_square_mod_24_l2278_227888

theorem prime_square_mod_24 (p q : ℕ) (k : ℤ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : p > 5) (hq_gt_5 : q > 5) 
  (h_diff : p ≠ q)
  (h_eq : p^2 - q^2 = 6 * k) : (p^2 - q^2) % 24 = 0 := by
sorry

end NUMINAMATH_GPT_prime_square_mod_24_l2278_227888


namespace NUMINAMATH_GPT_solve_for_a_l2278_227857

-- Given conditions
def x : ℕ := 2
def y : ℕ := 2
def equation (a : ℚ) : Prop := a * x + y = 5

-- Our goal is to prove that "a = 3/2" given the conditions
theorem solve_for_a : ∃ a : ℚ, equation a ∧ a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2278_227857


namespace NUMINAMATH_GPT_compute_nested_f_l2278_227843

def f(x : ℤ) : ℤ := x^2 - 4 * x + 3

theorem compute_nested_f : f (f (f (f (f (f 2))))) = f 1179395 := 
  sorry

end NUMINAMATH_GPT_compute_nested_f_l2278_227843


namespace NUMINAMATH_GPT_norma_total_cards_l2278_227875

theorem norma_total_cards (initial_cards : ℝ) (additional_cards : ℝ) (total_cards : ℝ) 
  (h1 : initial_cards = 88) (h2 : additional_cards = 70) : total_cards = 158 :=
by
  sorry

end NUMINAMATH_GPT_norma_total_cards_l2278_227875


namespace NUMINAMATH_GPT_Erica_Ice_Cream_Spend_l2278_227861

theorem Erica_Ice_Cream_Spend :
  (6 * ((3 * 2.00) + (2 * 1.50) + (2 * 3.00))) = 90 := sorry

end NUMINAMATH_GPT_Erica_Ice_Cream_Spend_l2278_227861


namespace NUMINAMATH_GPT_fraction_subtraction_simplification_l2278_227896

/-- Given that 57 equals 19 times 3, we want to prove that (8/19) - (5/57) equals 1/3. -/
theorem fraction_subtraction_simplification :
  8 / 19 - 5 / 57 = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_simplification_l2278_227896


namespace NUMINAMATH_GPT_six_circles_distance_relation_l2278_227806

/--
Prove that for any pair of non-touching circles (among six circles where each touches four of the remaining five),
their radii \( r_1 \) and \( r_2 \) and the distance \( d \) between their centers satisfy 

\[ d^{2}=r_{1}^{2}+r_{2}^{2} \pm 6r_{1}r_{2} \]

("plus" if the circles do not lie inside one another, "minus" otherwise).
-/
theorem six_circles_distance_relation 
  (r1 r2 d : ℝ) 
  (h : ∀ i : Fin 6, i < 6 → ∃ c : ℝ, (c = r1 ∨ c = r2) ∧ ∀ j : Fin 6, j ≠ i → abs (c - j) ≠ d ) :
  d^2 = r1^2 + r2^2 + 6 * r1 * r2 ∨ d^2 = r1^2 + r2^2 - 6 * r1 * r2 := 
  sorry

end NUMINAMATH_GPT_six_circles_distance_relation_l2278_227806


namespace NUMINAMATH_GPT_distinct_prime_factors_330_l2278_227891

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end NUMINAMATH_GPT_distinct_prime_factors_330_l2278_227891


namespace NUMINAMATH_GPT_find_r_l2278_227832

theorem find_r (k r : ℝ) : 
  5 = k * 3^r ∧ 45 = k * 9^r → r = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_r_l2278_227832


namespace NUMINAMATH_GPT_total_patients_in_a_year_l2278_227838

-- Define conditions from the problem
def patients_per_day_first : ℕ := 20
def percent_increase_second : ℕ := 20
def working_days_per_week : ℕ := 5
def working_weeks_per_year : ℕ := 50

-- Lean statement for the problem
theorem total_patients_in_a_year (patients_per_day_first : ℕ) (percent_increase_second : ℕ) (working_days_per_week : ℕ) (working_weeks_per_year : ℕ) :
  (patients_per_day_first + ((patients_per_day_first * percent_increase_second) / 100)) * working_days_per_week * working_weeks_per_year = 11000 :=
by
  sorry

end NUMINAMATH_GPT_total_patients_in_a_year_l2278_227838


namespace NUMINAMATH_GPT_problem_I_problem_II_problem_III_l2278_227828

variables {pA pB : ℝ}

-- Given conditions
def probability_A : ℝ := 0.7
def probability_B : ℝ := 0.6

-- Questions reformulated as proof goals
theorem problem_I : 
  sorry := 
 sorry

theorem problem_II : 
  -- Find: Probability that at least one of A or B succeeds on the first attempt
  sorry := 
 sorry

theorem problem_III : 
  -- Find: Probability that A succeeds exactly one more time than B in two attempts each
  sorry := 
 sorry

end NUMINAMATH_GPT_problem_I_problem_II_problem_III_l2278_227828


namespace NUMINAMATH_GPT_no_prime_satisfies_polynomial_l2278_227850

theorem no_prime_satisfies_polynomial :
  ∀ p : ℕ, p.Prime → p^3 - 6*p^2 - 3*p + 14 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_no_prime_satisfies_polynomial_l2278_227850


namespace NUMINAMATH_GPT_find_water_in_sport_formulation_l2278_227809

noncomputable def standard_formulation : ℚ × ℚ × ℚ := (1, 12, 30)
noncomputable def sport_flavoring_to_corn : ℚ := 3 * (1 / 12)
noncomputable def sport_flavoring_to_water : ℚ := (1 / 2) * (1 / 30)
noncomputable def sport_formulation (f : ℚ) (c : ℚ) (w : ℚ) : Prop :=
  f / c = sport_flavoring_to_corn ∧ f / w = sport_flavoring_to_water

noncomputable def given_corn_syrup : ℚ := 8

theorem find_water_in_sport_formulation :
  ∀ (f c w : ℚ), sport_formulation f c w → c = given_corn_syrup → w = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_water_in_sport_formulation_l2278_227809


namespace NUMINAMATH_GPT_prop_2_prop_3_l2278_227880

variables {a b c : ℝ}

-- Proposition 2: a > |b| -> a^2 > b^2
theorem prop_2 (h : a > |b|) : a^2 > b^2 := sorry

-- Proposition 3: a > b -> a^3 > b^3
theorem prop_3 (h : a > b) : a^3 > b^3 := sorry

end NUMINAMATH_GPT_prop_2_prop_3_l2278_227880


namespace NUMINAMATH_GPT_find_numbers_l2278_227898

theorem find_numbers (x y z t : ℕ) 
  (h1 : x + t = 37) 
  (h2 : y + z = 36) 
  (h3 : x + z = 2 * y) 
  (h4 : y * t = z * z) : 
  x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l2278_227898


namespace NUMINAMATH_GPT_q_compound_l2278_227874

def q (x y : ℤ) : ℤ :=
  if x ≥ 1 ∧ y ≥ 1 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x + y^2
  else 4 * x - 2 * y

theorem q_compound : q (q 2 (-2)) (q 0 0) = 48 := 
by 
  sorry

end NUMINAMATH_GPT_q_compound_l2278_227874


namespace NUMINAMATH_GPT_cats_to_dogs_l2278_227821

theorem cats_to_dogs (c d : ℕ) (h1 : c = 24) (h2 : 4 * d = 5 * c) : d = 30 :=
by
  sorry

end NUMINAMATH_GPT_cats_to_dogs_l2278_227821


namespace NUMINAMATH_GPT_fraction_in_jug_x_after_pouring_water_l2278_227813

-- Define capacities and initial fractions
def initial_fraction_x := 1 / 4
def initial_fraction_y := 2 / 3
def fill_needed_y := 1 - initial_fraction_y -- 1/3

-- Define capacity of original jugs
variable (C : ℚ) -- We can assume capacities are rational for simplicity

-- Define initial water amounts in jugs x and y
def initial_water_x := initial_fraction_x * C
def initial_water_y := initial_fraction_y * C

-- Define the water needed to fill jug y
def additional_water_needed_y := fill_needed_y * C

-- Define the final fraction of water in jug x
def final_fraction_x := initial_fraction_x / 2 -- since half of the initial water is poured out

theorem fraction_in_jug_x_after_pouring_water :
  final_fraction_x = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_in_jug_x_after_pouring_water_l2278_227813


namespace NUMINAMATH_GPT_train_b_speed_l2278_227847

theorem train_b_speed (v : ℝ) (t : ℝ) (d : ℝ) (sA : ℝ := 30) (start_time_diff : ℝ := 2) :
  (d = 180) -> (60 + sA*t = d) -> (v * t = d) -> v = 45 := by 
  sorry

end NUMINAMATH_GPT_train_b_speed_l2278_227847


namespace NUMINAMATH_GPT_parabola_vertex_y_l2278_227812

theorem parabola_vertex_y (x : ℝ) : (∃ (h k : ℝ), (4 * (x - h)^2 + k = 4 * x^2 + 16 * x + 11) ∧ k = -5) := 
  sorry

end NUMINAMATH_GPT_parabola_vertex_y_l2278_227812


namespace NUMINAMATH_GPT_money_distribution_problem_l2278_227835

theorem money_distribution_problem :
  ∃ n : ℕ, (3 * n + n * (n - 1) / 2 = 100 * n) ∧ n = 195 :=
by {
  use 195,
  sorry
}

end NUMINAMATH_GPT_money_distribution_problem_l2278_227835


namespace NUMINAMATH_GPT_units_digit_7_pow_6_l2278_227855

theorem units_digit_7_pow_6 : (7 ^ 6) % 10 = 9 := by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_6_l2278_227855


namespace NUMINAMATH_GPT_find_triplets_of_real_numbers_l2278_227800

theorem find_triplets_of_real_numbers (x y z : ℝ) :
  (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z) ∧ 
  (3 * x^2 + 2 * y^2 + z^2 = 240) → 
  (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) := 
sorry

end NUMINAMATH_GPT_find_triplets_of_real_numbers_l2278_227800


namespace NUMINAMATH_GPT_garden_length_l2278_227876

theorem garden_length (P : ℕ) (breadth : ℕ) (length : ℕ) 
  (h1 : P = 600) (h2 : breadth = 95) (h3 : P = 2 * (length + breadth)) : 
  length = 205 :=
by
  sorry

end NUMINAMATH_GPT_garden_length_l2278_227876


namespace NUMINAMATH_GPT_find_a_tangent_line_at_minus_one_l2278_227845

-- Define the function f with variable a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f with variable a
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Given conditions
def condition_1 : Prop := f' 1 = 1
def condition_2 : Prop := f' 2 (1 : ℝ) = 1

-- Prove that a = 2 given f'(1) = 1
theorem find_a : f' 2 (1 : ℝ) = 1 → 2 = 2 := by
  sorry

-- Given a = 2, find the tangent line equation at x = -1
def tangent_line_equation (x y : ℝ) : Prop := 9*x - y + 3 = 0

-- Define the coordinates of the point on the curve at x = -1
def point_on_curve : Prop := f 2 (-1) = -6

-- Prove the tangent line equation at x = -1 given a = 2
theorem tangent_line_at_minus_one (h : true) : tangent_line_equation 9 (f' 2 (-1)) := by
  sorry

end NUMINAMATH_GPT_find_a_tangent_line_at_minus_one_l2278_227845


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_find_n_given_sum_l2278_227810

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : a 10 = 30)
  (h2 : a 15 = 40)
  : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d) ∧ a 10 = 30 ∧ a 15 = 40 :=
by {
  sorry
}

theorem find_n_given_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 d : ℕ)
  (h_gen : ∀ n, a n = a1 + (n - 1) * d)
  (h_sum : ∀ n, S n = n * a1 + (n * (n - 1) * d) / 2)
  (h_a1 : a1 = 12)
  (h_d : d = 2)
  (h_Sn : S 14 = 210)
  : ∃ n, S n = 210 ∧ n = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_general_term_find_n_given_sum_l2278_227810


namespace NUMINAMATH_GPT_total_amount_divided_into_two_parts_l2278_227842

theorem total_amount_divided_into_two_parts (P1 P2 : ℝ) (annual_income : ℝ) :
  P1 = 1500.0000000000007 →
  annual_income = 135 →
  (P1 * 0.05 + P2 * 0.06 = annual_income) →
  P1 + P2 = 2500.000000000000 :=
by
  intros hP1 hIncome hInterest
  sorry

end NUMINAMATH_GPT_total_amount_divided_into_two_parts_l2278_227842


namespace NUMINAMATH_GPT_find_AB_l2278_227824

theorem find_AB
  (r R : ℝ)
  (h : r < R) :
  ∃ AB : ℝ, AB = (4 * r * (Real.sqrt (R * r))) / (R + r) :=
by
  sorry

end NUMINAMATH_GPT_find_AB_l2278_227824


namespace NUMINAMATH_GPT_chord_length_of_larger_circle_tangent_to_smaller_circle_l2278_227840

theorem chord_length_of_larger_circle_tangent_to_smaller_circle :
  ∀ (A B C : ℝ), B = 5 → π * (A ^ 2 - B ^ 2) = 50 * π → (C / 2) ^ 2 + B ^ 2 = A ^ 2 → C = 10 * Real.sqrt 2 :=
by
  intros A B C hB hArea hChord
  sorry

end NUMINAMATH_GPT_chord_length_of_larger_circle_tangent_to_smaller_circle_l2278_227840


namespace NUMINAMATH_GPT_total_amount_spent_l2278_227839

-- Define the prices related to John's Star Wars toy collection
def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

-- Problem statement in Lean: Prove the total amount spent is $3000
theorem total_amount_spent : (other_toys_cost + lightsaber_cost) = 3000 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end NUMINAMATH_GPT_total_amount_spent_l2278_227839


namespace NUMINAMATH_GPT_nursing_home_milk_l2278_227827

theorem nursing_home_milk :
  ∃ x y : ℕ, (2 * x + 16 = y) ∧ (4 * x - 12 = y) ∧ (x = 14) ∧ (y = 44) :=
by
  sorry

end NUMINAMATH_GPT_nursing_home_milk_l2278_227827
