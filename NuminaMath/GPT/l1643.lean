import Mathlib

namespace NUMINAMATH_GPT_man_age_twice_son_age_l1643_164342

theorem man_age_twice_son_age (S M : ℕ) (h1 : M = S + 24) (h2 : S = 22) : 
  ∃ Y : ℕ, M + Y = 2 * (S + Y) ∧ Y = 2 :=
by 
  sorry

end NUMINAMATH_GPT_man_age_twice_son_age_l1643_164342


namespace NUMINAMATH_GPT_pencils_added_by_sara_l1643_164339

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end NUMINAMATH_GPT_pencils_added_by_sara_l1643_164339


namespace NUMINAMATH_GPT_triangle_inequality_l1643_164311

variable {a b c : ℝ}

theorem triangle_inequality (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) : 
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1643_164311


namespace NUMINAMATH_GPT_opposite_number_l1643_164340

variable (a : ℝ)

theorem opposite_number (a : ℝ) : -(3 * a - 2) = -3 * a + 2 := by
  sorry

end NUMINAMATH_GPT_opposite_number_l1643_164340


namespace NUMINAMATH_GPT_f_periodic_function_l1643_164368

noncomputable def f : ℝ → ℝ := sorry

theorem f_periodic_function (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x : ℝ, f (x + 4) = f x + f 2)
    (h3 : f 1 = 2) : 
    f 2013 = 2 := sorry

end NUMINAMATH_GPT_f_periodic_function_l1643_164368


namespace NUMINAMATH_GPT_angle_B_is_30_degrees_l1643_164305

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Assuming the conditions given in the problem
variables (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) 
          (h2 : a > b)

-- The proof to establish the measure of angle B as 30 degrees
theorem angle_B_is_30_degrees (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) (h2 : a > b) : B = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_angle_B_is_30_degrees_l1643_164305


namespace NUMINAMATH_GPT_remainder_of_2365487_div_3_l1643_164385

theorem remainder_of_2365487_div_3 : (2365487 % 3) = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2365487_div_3_l1643_164385


namespace NUMINAMATH_GPT_find_ratio_EG_ES_l1643_164373

variables (EF GH EH EG ES QR : ℝ) -- lengths of the segments
variables (x y : ℝ) -- unknowns for parts of the segments
variables (Q R S : Point) -- points

-- Define conditions based on the problem
def parallelogram_EFGH (EF GH EH EG : ℝ) : Prop :=
  ∀ (x y : ℝ), EF = 8 * x ∧ EH = 9 * y

def point_on_segment_Q (Q : Point) (EF EQ : ℝ) : Prop :=
  ∃ x : ℝ, EQ = (1 / 8) * EF

def point_on_segment_R (R : Point) (EH ER : ℝ) : Prop :=
  ∃ y : ℝ, ER = (1 / 9) * EH

def intersection_at_S (EG QR ES : ℝ) : Prop :=
  ∃ x y : ℝ, ES = (1 / 8) * EG + (1 / 9) * EG

theorem find_ratio_EG_ES :
  parallelogram_EFGH EF GH EH EG →
  point_on_segment_Q Q EF (1/8 * EF) →
  point_on_segment_R R EH (1/9 * EH) →
  intersection_at_S EG QR ES →
  EG / ES = 72 / 17 :=
by
  intros h_parallelogram h_pointQ h_pointR h_intersection
  sorry

end NUMINAMATH_GPT_find_ratio_EG_ES_l1643_164373


namespace NUMINAMATH_GPT_find_m_l1643_164398

theorem find_m (m : ℝ) : 
  (∃ α β : ℝ, (α + β = 2 * (m + 1)) ∧ (α * β = m + 4) ∧ ((1 / α) + (1 / β) = 1)) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1643_164398


namespace NUMINAMATH_GPT_simplify_expression_l1643_164314

theorem simplify_expression :
  (8 : ℝ)^(1/3) - (343 : ℝ)^(1/3) = -5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1643_164314


namespace NUMINAMATH_GPT_biking_time_l1643_164312

noncomputable def east_bound_speed : ℝ := 22
noncomputable def west_bound_speed : ℝ := east_bound_speed + 4
noncomputable def total_distance : ℝ := 200

theorem biking_time :
  (east_bound_speed + west_bound_speed) * (t : ℝ) = total_distance → t = 25 / 6 :=
by
  -- The proof is omitted and replaced with sorry.
  sorry

end NUMINAMATH_GPT_biking_time_l1643_164312


namespace NUMINAMATH_GPT_max_original_chess_pieces_l1643_164318

theorem max_original_chess_pieces (m n M N : ℕ) (h1 : m ≤ 19) (h2 : n ≤ 19) (h3 : M ≤ 19) (h4 : N ≤ 19) (h5 : M * N = m * n + 45) (h6 : M = m ∨ N = n) : m * n ≤ 285 :=
by
  sorry

end NUMINAMATH_GPT_max_original_chess_pieces_l1643_164318


namespace NUMINAMATH_GPT_floodDamageInUSD_l1643_164300

def floodDamageAUD : ℝ := 45000000
def exchangeRateAUDtoUSD : ℝ := 1.2

theorem floodDamageInUSD : floodDamageAUD * (1 / exchangeRateAUDtoUSD) = 37500000 := 
by 
  sorry

end NUMINAMATH_GPT_floodDamageInUSD_l1643_164300


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1643_164321

theorem solution_set_of_inequality (x : ℝ) : 
  (|x+1| - |x-4| > 3) ↔ x > 3 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1643_164321


namespace NUMINAMATH_GPT_sum_of_bases_is_16_l1643_164358

/-
  Given the fractions G_1 and G_2 in two different bases S_1 and S_2, we need to show 
  that the sum of these bases S_1 and S_2 in base ten is 16.
-/
theorem sum_of_bases_is_16 (S_1 S_2 G_1 G_2 : ℕ) :
  (G_1 = (4 * S_1 + 5) / (S_1^2 - 1)) →
  (G_2 = (5 * S_1 + 4) / (S_1^2 - 1)) →
  (G_1 = (S_2 + 4) / (S_2^2 - 1)) →
  (G_2 = (4 * S_2 + 1) / (S_2^2 - 1)) →
  S_1 + S_2 = 16 :=
by
  intros hG1_S1 hG2_S1 hG1_S2 hG2_S2
  sorry

end NUMINAMATH_GPT_sum_of_bases_is_16_l1643_164358


namespace NUMINAMATH_GPT_find_a_value_l1643_164306

noncomputable def collinear (points : List (ℚ × ℚ)) := 
  ∃ a b c, ∀ (x y : ℚ), (x, y) ∈ points → a * x + b * y + c = 0

theorem find_a_value (a : ℚ) :
  collinear [(3, -5), (-a + 2, 3), (2*a + 3, 2)] → a = -7 / 23 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l1643_164306


namespace NUMINAMATH_GPT_number_of_diagonals_25_sides_l1643_164394

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_25_sides_l1643_164394


namespace NUMINAMATH_GPT_cells_surpass_10_pow_10_in_46_hours_l1643_164326

noncomputable def cells_exceed_threshold_hours : ℕ := 46

theorem cells_surpass_10_pow_10_in_46_hours : 
  ∀ (n : ℕ), (100 * ((3 / 2 : ℝ) ^ n) > 10 ^ 10) ↔ n ≥ cells_exceed_threshold_hours := 
by
  sorry

end NUMINAMATH_GPT_cells_surpass_10_pow_10_in_46_hours_l1643_164326


namespace NUMINAMATH_GPT_room_tiling_problem_correct_l1643_164372

noncomputable def room_tiling_problem : Prop :=
  let room_length := 6.72
  let room_width := 4.32
  let tile_size := 0.3
  let room_area := room_length * room_width
  let tile_area := tile_size * tile_size
  let num_tiles := (room_area / tile_area).ceil
  num_tiles = 323

theorem room_tiling_problem_correct : room_tiling_problem := 
  sorry

end NUMINAMATH_GPT_room_tiling_problem_correct_l1643_164372


namespace NUMINAMATH_GPT_find_a_b_find_A_l1643_164377

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * (Real.log x / Real.log 2) ^ 2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b

theorem find_a_b : (∀ x : ℝ, 0 < x → f x a b = 2 * (Real.log x / Real.log 2)^2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b) 
                     → f (1/2) a b = -8 
                     ∧ ∀ x : ℝ, 0 < x → x ≠ 1/2 → f x a b ≥ f (1 / 2) a b
                     → a = -2 ∧ b = -6 := 
sorry

theorem find_A (a b : ℝ) (h₁ : a = -2) (h₂ : b = -6) : 
  { x : ℝ | 0 < x ∧ f x a b > 0 } = {x | 0 < x ∧ (x < 1/8 ∨ x > 2)} :=
sorry

end NUMINAMATH_GPT_find_a_b_find_A_l1643_164377


namespace NUMINAMATH_GPT_output_value_of_y_l1643_164364

/-- Define the initial conditions -/
def l : ℕ := 2
def m : ℕ := 3
def n : ℕ := 5

/-- Define the function that executes the flowchart operations -/
noncomputable def flowchart_operation (l m n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem output_value_of_y : flowchart_operation l m n = 68 := sorry

end NUMINAMATH_GPT_output_value_of_y_l1643_164364


namespace NUMINAMATH_GPT_num_valid_triples_l1643_164344

theorem num_valid_triples : ∃! (count : ℕ), count = 22 ∧
  ∀ k m n : ℕ, (0 ≤ k) ∧ (k ≤ 100) ∧ (0 ≤ m) ∧ (m ≤ 100) ∧ (0 ≤ n) ∧ (n ≤ 100) → 
  (2^m * n - 2^n * m = 2^k) → count = 22 :=
sorry

end NUMINAMATH_GPT_num_valid_triples_l1643_164344


namespace NUMINAMATH_GPT_y_in_interval_l1643_164389

theorem y_in_interval :
  ∃ (y : ℝ), y = 5 + (1/y) * -y ∧ 2 < y ∧ y ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_y_in_interval_l1643_164389


namespace NUMINAMATH_GPT_cost_of_each_lunch_packet_l1643_164322

-- Definitions of the variables
def num_students := 50
def total_cost := 3087

-- Variables representing the unknowns
variable (s c n : ℕ)

-- Conditions
def more_than_half_students_bought : Prop := s > num_students / 2
def apples_less_than_cost_per_packet : Prop := n < c
def total_cost_condition : Prop := s * c = total_cost

-- The statement to prove
theorem cost_of_each_lunch_packet :
  (s : ℕ) * c = total_cost ∧
  (s > num_students / 2) ∧
  (n < c)
  -> c = 9 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_lunch_packet_l1643_164322


namespace NUMINAMATH_GPT_max_min_ab_bc_ca_l1643_164384

theorem max_min_ab_bc_ca (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 12) (h_ab_bc_ca : a * b + b * c + c * a = 30) :
  max (min (a * b) (min (b * c) (c * a))) = 9 :=
sorry

end NUMINAMATH_GPT_max_min_ab_bc_ca_l1643_164384


namespace NUMINAMATH_GPT_consecutive_odd_integers_l1643_164395

theorem consecutive_odd_integers (x : ℤ) (h : x + 4 = 15) : 3 * x - 2 * (x + 4) = 3 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_integers_l1643_164395


namespace NUMINAMATH_GPT_pure_imaginary_number_l1643_164359

theorem pure_imaginary_number (m : ℝ) (h_real : m^2 - 5 * m + 6 = 0) (h_imag : m^2 - 3 * m ≠ 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_number_l1643_164359


namespace NUMINAMATH_GPT_sum_of_powers_l1643_164386

theorem sum_of_powers (m n : ℤ)
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) :
  m^9 + n^9 = 76 :=
sorry

end NUMINAMATH_GPT_sum_of_powers_l1643_164386


namespace NUMINAMATH_GPT_average_cost_across_all_products_sold_is_670_l1643_164315

-- Definitions based on conditions
def iphones_sold : ℕ := 100
def ipad_sold : ℕ := 20
def appletv_sold : ℕ := 80

def cost_iphone : ℕ := 1000
def cost_ipad : ℕ := 900
def cost_appletv : ℕ := 200

-- Calculations based on conditions
def revenue_iphone : ℕ := iphones_sold * cost_iphone
def revenue_ipad : ℕ := ipad_sold * cost_ipad
def revenue_appletv : ℕ := appletv_sold * cost_appletv

def total_revenue : ℕ := revenue_iphone + revenue_ipad + revenue_appletv
def total_products_sold : ℕ := iphones_sold + ipad_sold + appletv_sold

def average_cost := total_revenue / total_products_sold

-- Theorem to be proved
theorem average_cost_across_all_products_sold_is_670 :
  average_cost = 670 :=
by
  sorry

end NUMINAMATH_GPT_average_cost_across_all_products_sold_is_670_l1643_164315


namespace NUMINAMATH_GPT_central_angle_l1643_164341

theorem central_angle (r l θ : ℝ) (condition1: 2 * r + l = 8) (condition2: (1 / 2) * l * r = 4) (theta_def : θ = l / r) : |θ| = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_l1643_164341


namespace NUMINAMATH_GPT_basketball_game_l1643_164388

theorem basketball_game (E H : ℕ) (h1 : E = H + 18) (h2 : E + H = 50) : H = 16 :=
by
  sorry

end NUMINAMATH_GPT_basketball_game_l1643_164388


namespace NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l1643_164353

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def expansion_coefficient_x3 : ℤ :=
  let term1 := (-1 : ℤ) ^ 3 * binomial_coefficient 6 3
  let term2 := (1 : ℤ) * binomial_coefficient 6 2
  term1 + term2

theorem coefficient_of_x3_in_expansion :
  expansion_coefficient_x3 = -5 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l1643_164353


namespace NUMINAMATH_GPT_percentage_of_millet_in_Brand_A_l1643_164329

variable (A B : ℝ)
variable (B_percent : B = 0.65)
variable (mix_millet_percent : 0.60 * A + 0.40 * B = 0.50)

theorem percentage_of_millet_in_Brand_A :
  A = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_millet_in_Brand_A_l1643_164329


namespace NUMINAMATH_GPT_john_total_amount_l1643_164349

/-- Define the amounts of money John has and needs additionally -/
def johnHas : ℝ := 0.75
def needsMore : ℝ := 1.75

/-- Prove the total amount of money John needs given the conditions -/
theorem john_total_amount : johnHas + needsMore = 2.50 := by
  sorry

end NUMINAMATH_GPT_john_total_amount_l1643_164349


namespace NUMINAMATH_GPT_ratio_Pat_Mark_l1643_164381

-- Definitions inferred from the conditions
def total_hours : ℕ := 135
def Kate_hours (K : ℕ) : ℕ := K
def Pat_hours (K : ℕ) : ℕ := 2 * K
def Mark_hours (K : ℕ) : ℕ := K + 75

-- The main statement
theorem ratio_Pat_Mark (K : ℕ) (h : Kate_hours K + Pat_hours K + Mark_hours K = total_hours) :
  (Pat_hours K) / (Mark_hours K) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_Pat_Mark_l1643_164381


namespace NUMINAMATH_GPT_primes_sum_solutions_l1643_164325

theorem primes_sum_solutions :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧
  p + q^2 + r^3 = 200 ∧ 
  ((p = 167 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 11 ∧ r = 2) ∨ 
   (p = 23 ∧ q = 13 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 2 ∧ r = 5)) :=
sorry

end NUMINAMATH_GPT_primes_sum_solutions_l1643_164325


namespace NUMINAMATH_GPT_least_total_cost_is_172_l1643_164327

noncomputable def least_total_cost : ℕ :=
  let lcm := Nat.lcm (Nat.lcm 6 5) 8
  let strawberry_packs := lcm / 6
  let blueberry_packs := lcm / 5
  let cherry_packs := lcm / 8
  let strawberry_cost := strawberry_packs * 2
  let blueberry_cost := blueberry_packs * 3
  let cherry_cost := cherry_packs * 4
  strawberry_cost + blueberry_cost + cherry_cost

theorem least_total_cost_is_172 : least_total_cost = 172 := 
by
  sorry

end NUMINAMATH_GPT_least_total_cost_is_172_l1643_164327


namespace NUMINAMATH_GPT_michael_earnings_l1643_164356

theorem michael_earnings :
  let price_extra_large := 150
  let price_large := 100
  let price_medium := 80
  let price_small := 60
  let qty_extra_large := 3
  let qty_large := 5
  let qty_medium := 8
  let qty_small := 10
  let discount_large := 0.10
  let tax := 0.05
  let cost_materials := 300
  let commission_fee := 0.10

  let total_initial_sales := (qty_extra_large * price_extra_large) + 
                             (qty_large * price_large) + 
                             (qty_medium * price_medium) + 
                             (qty_small * price_small)

  let discount_on_large := discount_large * (qty_large * price_large)
  let sales_after_discount := total_initial_sales - discount_on_large

  let sales_tax := tax * sales_after_discount
  let total_collected := sales_after_discount + sales_tax

  let commission := commission_fee * sales_after_discount
  let total_deductions := cost_materials + commission
  let earnings := total_collected - total_deductions

  earnings = 1733 :=
by
  sorry

end NUMINAMATH_GPT_michael_earnings_l1643_164356


namespace NUMINAMATH_GPT_geo_series_sum_l1643_164369

theorem geo_series_sum (a r : ℚ) (n: ℕ) (ha : a = 1/3) (hr : r = 1/2) (hn : n = 8) : 
    (a * (1 - r^n) / (1 - r)) = 85 / 128 := 
by
  sorry

end NUMINAMATH_GPT_geo_series_sum_l1643_164369


namespace NUMINAMATH_GPT_stratified_sample_selection_l1643_164380

def TotalStudents : ℕ := 900
def FirstYearStudents : ℕ := 300
def SecondYearStudents : ℕ := 200
def ThirdYearStudents : ℕ := 400
def SampleSize : ℕ := 45
def SamplingRatio : ℚ := 1 / 20

theorem stratified_sample_selection :
  (FirstYearStudents * SamplingRatio = 15) ∧
  (SecondYearStudents * SamplingRatio = 10) ∧
  (ThirdYearStudents * SamplingRatio = 20) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sample_selection_l1643_164380


namespace NUMINAMATH_GPT_simple_interest_l1643_164387

theorem simple_interest (TD : ℝ) (Sum : ℝ) (SI : ℝ) 
  (h1 : TD = 78) 
  (h2 : Sum = 947.1428571428571) 
  (h3 : SI = Sum - (Sum - TD)) : 
  SI = 78 := 
by 
  sorry

end NUMINAMATH_GPT_simple_interest_l1643_164387


namespace NUMINAMATH_GPT_inscribed_circle_ratio_l1643_164309

theorem inscribed_circle_ratio (a b h r : ℝ) (h_triangle : h = Real.sqrt (a^2 + b^2))
  (A : ℝ) (H1 : A = (1/2) * a * b) (s : ℝ) (H2 : s = (a + b + h) / 2) 
  (H3 : A = r * s) : (π * r / A) = (π * r) / (h + r) :=
sorry

end NUMINAMATH_GPT_inscribed_circle_ratio_l1643_164309


namespace NUMINAMATH_GPT_bahs_equal_to_yahs_l1643_164303

theorem bahs_equal_to_yahs (bahs rahs yahs : ℝ) 
  (h1 : 18 * bahs = 30 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) : 
  1200 * yahs = 432 * bahs := 
by
  sorry

end NUMINAMATH_GPT_bahs_equal_to_yahs_l1643_164303


namespace NUMINAMATH_GPT_min_p_value_l1643_164361

variable (p q r s : ℝ)

theorem min_p_value (h1 : p + q + r + s = 10)
                    (h2 : pq + pr + ps + qr + qs + rs = 20)
                    (h3 : p^2 * q^2 * r^2 * s^2 = 16) :
  p ≥ 2 ∧ ∃ q r s, q + r + s = 10 - p ∧ pq + pr + ps + qr + qs + rs = 20 ∧ (p^2 * q^2 * r^2 * s^2 = 16) :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_min_p_value_l1643_164361


namespace NUMINAMATH_GPT_total_pieces_correct_l1643_164324

-- Definition of the pieces of chicken required per type of order
def chicken_pieces_per_chicken_pasta : ℕ := 2
def chicken_pieces_per_barbecue_chicken : ℕ := 3
def chicken_pieces_per_fried_chicken_dinner : ℕ := 8

-- Definition of the number of each type of order tonight
def num_fried_chicken_dinner_orders : ℕ := 2
def num_chicken_pasta_orders : ℕ := 6
def num_barbecue_chicken_orders : ℕ := 3

-- Calculate the total number of pieces of chicken needed
def total_chicken_pieces_needed : ℕ :=
  (num_fried_chicken_dinner_orders * chicken_pieces_per_fried_chicken_dinner) +
  (num_chicken_pasta_orders * chicken_pieces_per_chicken_pasta) +
  (num_barbecue_chicken_orders * chicken_pieces_per_barbecue_chicken)

-- The proof statement
theorem total_pieces_correct : total_chicken_pieces_needed = 37 :=
by
  -- Our exact computation here
  sorry

end NUMINAMATH_GPT_total_pieces_correct_l1643_164324


namespace NUMINAMATH_GPT_lisa_takes_72_more_minutes_than_ken_l1643_164334

theorem lisa_takes_72_more_minutes_than_ken
  (ken_speed : ℕ) (lisa_speed : ℕ) (book_pages : ℕ)
  (h_ken_speed: ken_speed = 75)
  (h_lisa_speed: lisa_speed = 60)
  (h_book_pages: book_pages = 360) :
  ((book_pages / lisa_speed:ℚ) - (book_pages / ken_speed:ℚ)) * 60 = 72 :=
by
  sorry

end NUMINAMATH_GPT_lisa_takes_72_more_minutes_than_ken_l1643_164334


namespace NUMINAMATH_GPT_molecular_weight_NaClO_is_74_44_l1643_164323

-- Define the atomic weights
def atomic_weight_Na : Real := 22.99
def atomic_weight_Cl : Real := 35.45
def atomic_weight_O : Real := 16.00

-- Define the calculation of molecular weight
def molecular_weight_NaClO : Real :=
  atomic_weight_Na + atomic_weight_Cl + atomic_weight_O

-- Define the theorem statement
theorem molecular_weight_NaClO_is_74_44 :
  molecular_weight_NaClO = 74.44 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_molecular_weight_NaClO_is_74_44_l1643_164323


namespace NUMINAMATH_GPT_point_on_line_l1643_164392

theorem point_on_line (x : ℝ) : 
    (∃ k : ℝ, (-4) = k * (-4) + 8) → 
    (-4 = 2 * x + 8) → 
    x = -6 := 
sorry

end NUMINAMATH_GPT_point_on_line_l1643_164392


namespace NUMINAMATH_GPT_qin_jiushao_algorithm_v2_l1643_164352

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x to evaluate the polynomial at
def x0 : ℝ := -1

-- Define the intermediate value v2 according to Horner's rule
def v1 : ℝ := 2 * x0^4 - 3 * x0^3 + x0^2
def v2 : ℝ := v1 * x0 + 2

theorem qin_jiushao_algorithm_v2 : v2 = -4 := 
by 
  -- The proof will be here, for now we place sorry.
  sorry

end NUMINAMATH_GPT_qin_jiushao_algorithm_v2_l1643_164352


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1643_164378

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 36) 
  (h2 : b + c = 55) 
  (h3 : c + a = 60) : 
  a + b + c = 75.5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1643_164378


namespace NUMINAMATH_GPT_remaining_amount_correct_l1643_164367

def initial_amount : ℝ := 70
def coffee_cost_per_pound : ℝ := 8.58
def coffee_pounds : ℝ := 4.0
def total_cost : ℝ := coffee_pounds * coffee_cost_per_pound
def remaining_amount : ℝ := initial_amount - total_cost

theorem remaining_amount_correct : remaining_amount = 35.68 :=
by
  -- Skip the proof; this is a placeholder.
  sorry

end NUMINAMATH_GPT_remaining_amount_correct_l1643_164367


namespace NUMINAMATH_GPT_smallest_AAB_l1643_164374

theorem smallest_AAB : ∃ (A B : ℕ), (1 <= A ∧ A <= 9) ∧ (1 <= B ∧ B <= 9) ∧ (AB = 10 * A + B) ∧ (AAB = 100 * A + 10 * A + B) ∧ (110 * A + B = 8 * (10 * A + B)) ∧ (AAB = 221) :=
by
  sorry

end NUMINAMATH_GPT_smallest_AAB_l1643_164374


namespace NUMINAMATH_GPT_batsman_average_after_12th_l1643_164355

theorem batsman_average_after_12th (runs_12th : ℕ) (average_increase : ℕ) (initial_innings : ℕ)
   (initial_average : ℝ) (runs_before_12th : ℕ → ℕ) 
   (h1 : runs_12th = 48)
   (h2 : average_increase = 2)
   (h3 : initial_innings = 11)
   (h4 : initial_average = 24)
   (h5 : ∀ i, i < initial_innings → runs_before_12th i ≥ 20)
   (h6 : ∃ i, runs_before_12th i = 25 ∧ runs_before_12th (i + 1) = 25) :
   (11 * initial_average + runs_12th) / 12 = 26 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_l1643_164355


namespace NUMINAMATH_GPT_vinegar_mixture_concentration_l1643_164337

theorem vinegar_mixture_concentration :
  let c1 := 5 / 100
  let c2 := 10 / 100
  let v1 := 10
  let v2 := 10
  (v1 * c1 + v2 * c2) / (v1 + v2) = 7.5 / 100 :=
by
  sorry

end NUMINAMATH_GPT_vinegar_mixture_concentration_l1643_164337


namespace NUMINAMATH_GPT_negation_proposition_l1643_164317

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ ∃ x0 : ℝ, x0^2 - 2*x0 + 4 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1643_164317


namespace NUMINAMATH_GPT_profit_distribution_l1643_164379

noncomputable def profit_sharing (investment_a investment_d profit: ℝ) : ℝ × ℝ :=
  let total_investment := investment_a + investment_d
  let share_a := investment_a / total_investment
  let share_d := investment_d / total_investment
  (share_a * profit, share_d * profit)

theorem profit_distribution :
  let investment_a := 22500
  let investment_d := 35000
  let first_period_profit := 9600
  let second_period_profit := 12800
  let third_period_profit := 18000
  profit_sharing investment_a investment_d first_period_profit = (3600, 6000) ∧
  profit_sharing investment_a investment_d second_period_profit = (5040, 7760) ∧
  profit_sharing investment_a investment_d third_period_profit = (7040, 10960) :=
sorry

end NUMINAMATH_GPT_profit_distribution_l1643_164379


namespace NUMINAMATH_GPT_sharon_distance_l1643_164371

noncomputable def usual_speed (x : ℝ) := x / 180
noncomputable def reduced_speed (x : ℝ) := (x / 180) - 0.5

theorem sharon_distance (x : ℝ) (h : 300 = (x / 2) / usual_speed x + (x / 2) / reduced_speed x) : x = 157.5 :=
by sorry

end NUMINAMATH_GPT_sharon_distance_l1643_164371


namespace NUMINAMATH_GPT_square_perimeter_from_area_l1643_164363

def square_area (s : ℝ) : ℝ := s * s -- Definition of the area of a square based on its side length.
def square_perimeter (s : ℝ) : ℝ := 4 * s -- Definition of the perimeter of a square based on its side length.

theorem square_perimeter_from_area (s : ℝ) (h : square_area s = 900) : square_perimeter s = 120 :=
by {
  sorry -- Placeholder for the proof.
}

end NUMINAMATH_GPT_square_perimeter_from_area_l1643_164363


namespace NUMINAMATH_GPT_score_sd_above_mean_l1643_164345

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end NUMINAMATH_GPT_score_sd_above_mean_l1643_164345


namespace NUMINAMATH_GPT_find_a₉_l1643_164383

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom S_6_eq : S 6 = 3
axiom S_11_eq : S 11 = 18

noncomputable def a₉ : ℝ := sorry -- Define a₉ here, proof skipped by "sorry"

theorem find_a₉ (a : ℕ → ℝ) (S : ℕ → ℝ) :
  S 6 = 3 →
  S 11 = 18 →
  a₉ = 3 :=
by
  intros S_6_eq S_11_eq
  sorry -- Proof goes here

end NUMINAMATH_GPT_find_a₉_l1643_164383


namespace NUMINAMATH_GPT_find_integers_l1643_164393

theorem find_integers (x y : ℕ) (h : 2 * x * y = 21 + 2 * x + y) : (x = 1 ∧ y = 23) ∨ (x = 6 ∧ y = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_integers_l1643_164393


namespace NUMINAMATH_GPT_Dean_handled_100_transactions_l1643_164320

-- Definitions for the given conditions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := (9 * Mabel_transactions) / 10 + Mabel_transactions
def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3
def Jade_transactions : ℕ := Cal_transactions + 14
def Dean_transactions : ℕ := (Jade_transactions * 25) / 100 + Jade_transactions

-- Define the theorem we need to prove
theorem Dean_handled_100_transactions : Dean_transactions = 100 :=
by
  -- Statement to skip the actual proof
  sorry

end NUMINAMATH_GPT_Dean_handled_100_transactions_l1643_164320


namespace NUMINAMATH_GPT_cricket_team_average_age_difference_l1643_164390

theorem cricket_team_average_age_difference :
  let team_size := 11
  let captain_age := 26
  let keeper_age := captain_age + 3
  let avg_whole_team := 23
  let total_team_age := avg_whole_team * team_size
  let combined_age := captain_age + keeper_age
  let remaining_players := team_size - 2
  let total_remaining_age := total_team_age - combined_age
  let avg_remaining_players := total_remaining_age / remaining_players
  avg_whole_team - avg_remaining_players = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cricket_team_average_age_difference_l1643_164390


namespace NUMINAMATH_GPT_percentage_of_number_is_40_l1643_164351

theorem percentage_of_number_is_40 (N : ℝ) (P : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 35) 
  (h2 : (P/100) * N = 420) : 
  P = 40 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_number_is_40_l1643_164351


namespace NUMINAMATH_GPT_length_of_segment_CD_l1643_164328

theorem length_of_segment_CD (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
  (h_ratio1 : x = (3 / 5) * (3 + y))
  (h_ratio2 : (x + 3) / y = 4 / 7)
  (h_RS : 3 = 3) :
  x + 3 + y = 273.6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_CD_l1643_164328


namespace NUMINAMATH_GPT_find_ck_l1643_164382

theorem find_ck (d r k : ℕ) (a_n b_n c_n : ℕ → ℕ) 
  (h_an : ∀ n, a_n n = 1 + (n - 1) * d)
  (h_bn : ∀ n, b_n n = r ^ (n - 1))
  (h_cn : ∀ n, c_n n = a_n n + b_n n)
  (h_ckm1 : c_n (k - 1) = 30)
  (h_ckp1 : c_n (k + 1) = 300) :
  c_n k = 83 := 
sorry

end NUMINAMATH_GPT_find_ck_l1643_164382


namespace NUMINAMATH_GPT_alley_width_theorem_l1643_164301

noncomputable def width_of_alley (a k h : ℝ) (h₁ : k = a / 2) (h₂ : h = a * (Real.sqrt 2) / 2) : ℝ :=
  Real.sqrt ((a * (Real.sqrt 2) / 2)^2 + (a / 2)^2)

theorem alley_width_theorem (a k h w : ℝ)
  (h₁ : k = a / 2)
  (h₂ : h = a * (Real.sqrt 2) / 2)
  (h₃ : w = width_of_alley a k h h₁ h₂) :
  w = (Real.sqrt 3) * a / 2 :=
by
  sorry

end NUMINAMATH_GPT_alley_width_theorem_l1643_164301


namespace NUMINAMATH_GPT_max_distance_traveled_l1643_164307

def distance_traveled (t : ℝ) : ℝ := 15 * t - 6 * t^2

theorem max_distance_traveled : ∃ t : ℝ, distance_traveled t = 75 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_traveled_l1643_164307


namespace NUMINAMATH_GPT_overall_average_output_l1643_164376

theorem overall_average_output 
  (initial_cogs : ℕ := 60) 
  (rate_1 : ℕ := 36) 
  (rate_2 : ℕ := 60) 
  (second_batch_cogs : ℕ := 60) :
  (initial_cogs + second_batch_cogs) / ((initial_cogs / rate_1) + (second_batch_cogs / rate_2)) = 45 := 
  sorry

end NUMINAMATH_GPT_overall_average_output_l1643_164376


namespace NUMINAMATH_GPT_transformed_passes_through_l1643_164399

def original_parabola (x : ℝ) : ℝ :=
  -x^2 - 2*x + 3

def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 1)^2 + 2

theorem transformed_passes_through : transformed_parabola (-1) = 1 :=
  by sorry

end NUMINAMATH_GPT_transformed_passes_through_l1643_164399


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l1643_164397

theorem base_eight_to_base_ten : (4 * 8^1 + 5 * 8^0 = 37) := by
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_l1643_164397


namespace NUMINAMATH_GPT_cyclist_C_speed_l1643_164360

theorem cyclist_C_speed 
  (dist_XY : ℝ)
  (speed_diff : ℝ)
  (meet_point : ℝ)
  (c d : ℝ)
  (h1 : dist_XY = 90)
  (h2 : speed_diff = 5)
  (h3 : meet_point = 15)
  (h4 : d = c + speed_diff)
  (h5 : 75 = dist_XY - meet_point)
  (h6 : 105 = dist_XY + meet_point)
  (h7 : 75 / c = 105 / d) :
  c = 12.5 :=
sorry

end NUMINAMATH_GPT_cyclist_C_speed_l1643_164360


namespace NUMINAMATH_GPT_projectiles_meet_in_84_minutes_l1643_164336

theorem projectiles_meet_in_84_minutes :
  ∀ (d v₁ v₂ : ℝ), d = 1386 → v₁ = 445 → v₂ = 545 → (20 : ℝ) = 20 → 
  ((1386 / (445 + 545) / 60) * 60 * 60 = 84) :=
by
  intros d v₁ v₂ h_d h_v₁ h_v₂ h_wind
  sorry

end NUMINAMATH_GPT_projectiles_meet_in_84_minutes_l1643_164336


namespace NUMINAMATH_GPT_length_of_each_part_l1643_164343

theorem length_of_each_part (ft : ℕ) (inch : ℕ) (parts : ℕ) (total_length : ℕ) (part_length : ℕ) :
  ft = 6 → inch = 8 → parts = 5 → total_length = 12 * ft + inch → part_length = total_length / parts → part_length = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_length_of_each_part_l1643_164343


namespace NUMINAMATH_GPT_num_diagonals_octagon_l1643_164357

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_octagon : num_diagonals 8 = 20 :=
by
  sorry

end NUMINAMATH_GPT_num_diagonals_octagon_l1643_164357


namespace NUMINAMATH_GPT_total_camels_l1643_164313

theorem total_camels (x y : ℕ) (humps_eq : x + 2 * y = 23) (legs_eq : 4 * (x + y) = 60) : x + y = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_camels_l1643_164313


namespace NUMINAMATH_GPT_cricket_team_members_l1643_164354

theorem cricket_team_members (n : ℕ) 
  (avg_age_team : ℕ) 
  (age_captain : ℕ) 
  (age_wkeeper : ℕ) 
  (avg_age_remaining : ℕ) 
  (total_age_team : ℕ) 
  (total_age_excl_cw : ℕ) 
  (total_age_remaining : ℕ) :
  avg_age_team = 23 →
  age_captain = 26 →
  age_wkeeper = 29 →
  avg_age_remaining = 22 →
  total_age_team = avg_age_team * n →
  total_age_excl_cw = total_age_team - (age_captain + age_wkeeper) →
  total_age_remaining = avg_age_remaining * (n - 2) →
  total_age_excl_cw = total_age_remaining →
  n = 11 :=
by
  sorry

end NUMINAMATH_GPT_cricket_team_members_l1643_164354


namespace NUMINAMATH_GPT_minimum_moves_to_find_coin_l1643_164310

/--
Consider a circle of 100 thimbles with a coin hidden under one of them. 
You can check four thimbles per move. After each move, the coin moves to a neighboring thimble.
Prove that the minimum number of moves needed to guarantee finding the coin is 33.
-/
theorem minimum_moves_to_find_coin 
  (N : ℕ) (hN : N = 100) (M : ℕ) (hM : M = 4) :
  ∃! k : ℕ, k = 33 :=
by sorry

end NUMINAMATH_GPT_minimum_moves_to_find_coin_l1643_164310


namespace NUMINAMATH_GPT_perfect_square_difference_l1643_164302

def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem perfect_square_difference :
  ∃ a b : ℕ, ∃ x y : ℕ,
    a = x^2 ∧ b = y^2 ∧
    lastDigit a = 6 ∧
    lastDigit b = 4 ∧
    lastDigit (a - b) = 2 ∧
    lastDigit a > lastDigit b :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_difference_l1643_164302


namespace NUMINAMATH_GPT_triangle_area_l1643_164348

noncomputable def area_triangle (A B C : ℝ) (b c : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem triangle_area
  (A B C : ℝ) (b : ℝ) 
  (hA : A = π / 4)
  (h0 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  ∃ c : ℝ, area_triangle A B C b c = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1643_164348


namespace NUMINAMATH_GPT_combined_surface_area_of_cube_and_sphere_l1643_164319

theorem combined_surface_area_of_cube_and_sphere (V_cube : ℝ) :
  V_cube = 729 →
  ∃ (A_combined : ℝ), A_combined = 486 + 81 * Real.pi :=
by
  intro V_cube
  sorry

end NUMINAMATH_GPT_combined_surface_area_of_cube_and_sphere_l1643_164319


namespace NUMINAMATH_GPT_production_period_l1643_164366

-- Define the conditions as constants
def daily_production : ℕ := 1500
def price_per_computer : ℕ := 150
def total_earnings : ℕ := 1575000

-- Define the computation to find the period and state what we need to prove
theorem production_period : (total_earnings / price_per_computer) / daily_production = 7 :=
by
  -- you can provide the steps, but it's optional since the proof is omitted
  sorry

end NUMINAMATH_GPT_production_period_l1643_164366


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l1643_164375

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (h_product : a * b * c = 1) :
  (a^6 / ((a - b) * (a - c)) + b^6 / ((b - c) * (b - a)) + c^6 / ((c - a) * (c - b)) > 15) := 
by sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l1643_164375


namespace NUMINAMATH_GPT_binary_to_octal_conversion_l1643_164335

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end NUMINAMATH_GPT_binary_to_octal_conversion_l1643_164335


namespace NUMINAMATH_GPT_square_area_inside_ellipse_l1643_164396

theorem square_area_inside_ellipse :
  (∃ s : ℝ, 
    ∀ (x y : ℝ), 
      (x = s ∧ y = s) → 
      (x^2 / 4 + y^2 / 8 = 1) ∧ 
      (4 * (s^2 / 3) = 1) ∧ 
      (area = 4 * (8 / 3))) →
    ∃ area : ℝ, 
      area = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_square_area_inside_ellipse_l1643_164396


namespace NUMINAMATH_GPT_trains_clear_in_approx_6_85_seconds_l1643_164365

noncomputable def length_first_train : ℝ := 111
noncomputable def length_second_train : ℝ := 165
noncomputable def speed_first_train : ℝ := 80 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def speed_second_train : ℝ := 65 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_in_approx_6_85_seconds : abs (time_to_clear - 6.85) < 0.01 := sorry

end NUMINAMATH_GPT_trains_clear_in_approx_6_85_seconds_l1643_164365


namespace NUMINAMATH_GPT_arthur_bought_hamburgers_on_first_day_l1643_164370

-- Define the constants and parameters
def D : ℕ := 1
def H : ℕ := 2
def total_cost_day1 : ℕ := 10
def total_cost_day2 : ℕ := 7

-- Define the equation representing the transactions
def equation_day1 (h : ℕ) := H * h + 4 * D = total_cost_day1
def equation_day2 := 2 * H + 3 * D = total_cost_day2

-- The theorem we need to prove: the number of hamburgers h bought on the first day is 3
theorem arthur_bought_hamburgers_on_first_day (h : ℕ) (hd1 : equation_day1 h) (hd2 : equation_day2) : h = 3 := 
by 
  sorry

end NUMINAMATH_GPT_arthur_bought_hamburgers_on_first_day_l1643_164370


namespace NUMINAMATH_GPT_yellow_bags_count_l1643_164331

theorem yellow_bags_count (R B Y : ℕ) 
  (h1 : R + B + Y = 12) 
  (h2 : 10 * R + 50 * B + 100 * Y = 500) 
  (h3 : R = B) : 
  Y = 2 := 
by 
  sorry

end NUMINAMATH_GPT_yellow_bags_count_l1643_164331


namespace NUMINAMATH_GPT_sufficient_condition_l1643_164316

variable (a b c d : ℝ)

-- Condition p: a and b are the roots of the equation.
def condition_p : Prop := a * a + b * b + c * (a + b) + d = 0

-- Condition q: a + b + c = 0
def condition_q : Prop := a + b + c = 0

theorem sufficient_condition : condition_p a b c d → condition_q a b c := by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l1643_164316


namespace NUMINAMATH_GPT_third_term_binomial_expansion_l1643_164346

-- Let a, x be real numbers
variables (a x : ℝ)

-- Binomial theorem term for k = 2
def binomial_term (n k : ℕ) (x y : ℝ) : ℝ :=
  (Nat.choose n k) * x^(n-k) * y^k

theorem third_term_binomial_expansion :
  binomial_term 6 2 (a / Real.sqrt x) (-Real.sqrt x / a^2) = 15 / x :=
by
  sorry

end NUMINAMATH_GPT_third_term_binomial_expansion_l1643_164346


namespace NUMINAMATH_GPT_value_of_nested_fraction_l1643_164338

def nested_fraction : ℚ :=
  2 - (1 / (2 - (1 / (2 - 1 / 2))))

theorem value_of_nested_fraction : nested_fraction = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_nested_fraction_l1643_164338


namespace NUMINAMATH_GPT_cylinder_cone_volume_l1643_164333

theorem cylinder_cone_volume (V_total : ℝ) (Vc Vcone : ℝ)
  (h1 : V_total = 48)
  (h2 : V_total = Vc + Vcone)
  (h3 : Vc = 3 * Vcone) :
  Vc = 36 ∧ Vcone = 12 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_cone_volume_l1643_164333


namespace NUMINAMATH_GPT_matthew_egg_rolls_l1643_164330

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end NUMINAMATH_GPT_matthew_egg_rolls_l1643_164330


namespace NUMINAMATH_GPT_book_pages_read_l1643_164391

theorem book_pages_read (pages_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) :
  (pages_per_day = 100) →
  (days_per_week = 3) →
  (weeks = 7) →
  total_pages = pages_per_day * days_per_week * weeks →
  total_pages = 2100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_book_pages_read_l1643_164391


namespace NUMINAMATH_GPT_distinct_sequences_count_l1643_164347

noncomputable def number_of_distinct_sequences (n : ℕ) : ℕ :=
  if n = 6 then 12 else sorry

theorem distinct_sequences_count : number_of_distinct_sequences 6 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_sequences_count_l1643_164347


namespace NUMINAMATH_GPT_avg_age_of_five_students_l1643_164332

-- step a: Define the conditions
def avg_age_seventeen_students : ℕ := 17
def total_seventeen_students : ℕ := 17 * avg_age_seventeen_students

def num_students_with_unknown_avg : ℕ := 5

def avg_age_nine_students : ℕ := 16
def num_students_with_known_avg : ℕ := 9
def total_age_nine_students : ℕ := num_students_with_known_avg * avg_age_nine_students

def age_seventeenth_student : ℕ := 75

-- step c: Compute the average age of the 5 students
noncomputable def total_age_five_students : ℕ :=
  total_seventeen_students - total_age_nine_students - age_seventeenth_student

def correct_avg_age_five_students : ℕ := 14

theorem avg_age_of_five_students :
  total_age_five_students / num_students_with_unknown_avg = correct_avg_age_five_students :=
sorry

end NUMINAMATH_GPT_avg_age_of_five_students_l1643_164332


namespace NUMINAMATH_GPT_arithmetic_seq_8th_term_l1643_164308

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 22) 
  (h6 : a + 5 * d = 46) : 
  a + 7 * d = 70 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_8th_term_l1643_164308


namespace NUMINAMATH_GPT_sector_central_angle_l1643_164304

theorem sector_central_angle (r α: ℝ) (hC: 4 * r = 2 * r + α * r): α = 2 :=
by
  -- Proof is to be filled in
  sorry

end NUMINAMATH_GPT_sector_central_angle_l1643_164304


namespace NUMINAMATH_GPT_marble_choice_l1643_164350

def numDifferentGroupsOfTwoMarbles (red green blue : ℕ) (yellow : ℕ) (orange : ℕ) : ℕ :=
  if (red = 1 ∧ green = 1 ∧ blue = 1 ∧ yellow = 2 ∧ orange = 2) then 12 else 0

theorem marble_choice:
  let red := 1
  let green := 1
  let blue := 1
  let yellow := 2
  let orange := 2
  numDifferentGroupsOfTwoMarbles red green blue yellow orange = 12 :=
by
  dsimp[numDifferentGroupsOfTwoMarbles]
  split_ifs
  · rfl
  · sorry

-- Ensure the theorem type matches the expected Lean 4 structure.
#print marble_choice

end NUMINAMATH_GPT_marble_choice_l1643_164350


namespace NUMINAMATH_GPT_intersections_correct_l1643_164362

-- Define the distances (in meters)
def gretzky_street_length : ℕ := 5600
def segment_a_distance : ℕ := 350
def segment_b_distance : ℕ := 400
def segment_c_distance : ℕ := 450

-- Definitions based on conditions
def segment_a_intersections : ℕ :=
  gretzky_street_length / segment_a_distance - 2 -- subtract Orr Street and Howe Street

def segment_b_intersections : ℕ :=
  gretzky_street_length / segment_b_distance

def segment_c_intersections : ℕ :=
  gretzky_street_length / segment_c_distance

-- Sum of all intersections
def total_intersections : ℕ :=
  segment_a_intersections + segment_b_intersections + segment_c_intersections

theorem intersections_correct :
  total_intersections = 40 :=
by
  sorry

end NUMINAMATH_GPT_intersections_correct_l1643_164362
