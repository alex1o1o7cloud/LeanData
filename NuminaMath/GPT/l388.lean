import Mathlib

namespace NUMINAMATH_GPT_ellipse_equation_minimum_distance_l388_38841

-- Define the conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2) / (a^2) + (y^2) / (b^2) = 1)

def eccentricity (a c : ℝ) : Prop :=
  c = a / 2

def focal_distance (c : ℝ) : Prop :=
  2 * c = 4

def foci_parallel (F1 A B C D : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := F1;
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (yA - y1) / (xA - x1) = (yC - y1) / (xC - x1) ∧ 
  (yB - y1) / (xB - x1) = (yD - y1) / (xD - x1)

def orthogonal_vectors (A C B D : ℝ × ℝ) : Prop :=
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (xC - xA) * (xD - xB) + (yC - yA) * (yD - yB) = 0

-- Prove equation of ellipse E
theorem ellipse_equation (a b : ℝ) (x y : ℝ) (c : ℝ)
  (h1 : ellipse a b x y)
  (h2 : eccentricity a c)
  (h3 : focal_distance c) :
  (a = 4) ∧ (b^2 = 12) ∧ (x^2 / 16 + y^2 / 12 = 1) :=
sorry

-- Prove minimum value of |AC| + |BD|
theorem minimum_distance (A B C D : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (h1 : foci_parallel F1 A B C D)
  (h2 : orthogonal_vectors A C B D) :
  |(AC : ℝ)| + |(BD : ℝ)| = 96 / 7 :=
sorry

end NUMINAMATH_GPT_ellipse_equation_minimum_distance_l388_38841


namespace NUMINAMATH_GPT_line_equation_and_inclination_l388_38819

variable (t : ℝ)
variable (x y : ℝ)
variable (α : ℝ)
variable (l : x = -3 + t ∧ y = 1 + sqrt 3 * t)

theorem line_equation_and_inclination 
  (H : l) : 
  (∃ a b c : ℝ, a = sqrt 3 ∧ b = -1 ∧ c = 3 * sqrt 3 + 1 ∧ a * x + b * y + c = 0) ∧
  α = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_and_inclination_l388_38819


namespace NUMINAMATH_GPT_roots_of_polynomial_l388_38879

theorem roots_of_polynomial :
  ∀ x : ℝ, x * (x + 2)^2 * (3 - x) * (5 + x) = 0 ↔ (x = 0 ∨ x = -2 ∨ x = 3 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l388_38879


namespace NUMINAMATH_GPT_sale_in_fifth_month_l388_38800

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 avg_sale num_months total_sales known_sales_five_months sale5: ℕ) :
  sale1 = 6400 →
  sale2 = 7000 →
  sale3 = 6800 →
  sale4 = 7200 →
  sale6 = 5100 →
  avg_sale = 6500 →
  num_months = 6 →
  total_sales = avg_sale * num_months →
  known_sales_five_months = sale1 + sale2 + sale3 + sale4 + sale6 →
  sale5 = total_sales - known_sales_five_months →
  sale5 = 6500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l388_38800


namespace NUMINAMATH_GPT_man_l388_38845

theorem man's_rate_in_still_water 
  (V_s V_m : ℝ)
  (with_stream : V_m + V_s = 24)  -- Condition 1
  (against_stream : V_m - V_s = 10) -- Condition 2
  : V_m = 17 := 
by
  sorry

end NUMINAMATH_GPT_man_l388_38845


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l388_38883

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ)     -- arithmetic sequence
  (d : ℝ)         -- common difference
  (h: ∀ n, a (n + 1) = a n + d)     -- definition of arithmetic sequence
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l388_38883


namespace NUMINAMATH_GPT_marks_age_more_than_thrice_aarons_l388_38834

theorem marks_age_more_than_thrice_aarons :
  ∃ (A : ℕ)(X : ℕ), 28 = A + 17 ∧ 25 = 3 * (A - 3) + X ∧ 32 = 2 * (A + 4) + 2 ∧ X = 1 :=
by
  sorry

end NUMINAMATH_GPT_marks_age_more_than_thrice_aarons_l388_38834


namespace NUMINAMATH_GPT_pond_field_area_ratio_l388_38872

theorem pond_field_area_ratio (w l s A_field A_pond : ℕ) (h1 : l = 2 * w) (h2 : l = 96) (h3 : s = 8) (h4 : A_field = l * w) (h5 : A_pond = s * s) :
  A_pond.toFloat / A_field.toFloat = 1 / 72 := 
by
  sorry

end NUMINAMATH_GPT_pond_field_area_ratio_l388_38872


namespace NUMINAMATH_GPT_find_angle_A_l388_38887

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hB : B = Real.pi / 3) : 
  A = Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_find_angle_A_l388_38887


namespace NUMINAMATH_GPT_quadratic_sum_l388_38874

theorem quadratic_sum (x : ℝ) :
  ∃ a h k : ℝ, (5*x^2 - 10*x - 3 = a*(x - h)^2 + k) ∧ (a + h + k = -2) :=
sorry

end NUMINAMATH_GPT_quadratic_sum_l388_38874


namespace NUMINAMATH_GPT_Maddie_spent_on_tshirts_l388_38898

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end NUMINAMATH_GPT_Maddie_spent_on_tshirts_l388_38898


namespace NUMINAMATH_GPT_game_A_probability_greater_than_B_l388_38859

-- Defining the probabilities of heads and tails for the biased coin
def prob_heads : ℚ := 2 / 3
def prob_tails : ℚ := 1 / 3

-- Defining the winning probabilities for Game A
def prob_winning_A : ℚ := (prob_heads^4) + (prob_tails^4)

-- Defining the winning probabilities for Game B
def prob_winning_B : ℚ := (prob_heads^3 * prob_tails) + (prob_tails^3 * prob_heads)

-- The statement we want to prove
theorem game_A_probability_greater_than_B : prob_winning_A - prob_winning_B = 7 / 81 := by
  sorry

end NUMINAMATH_GPT_game_A_probability_greater_than_B_l388_38859


namespace NUMINAMATH_GPT_combined_total_pets_l388_38873

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end NUMINAMATH_GPT_combined_total_pets_l388_38873


namespace NUMINAMATH_GPT_parallelogram_area_l388_38836

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (hb : b = 15) (hh : h = 2 * b) (hA : A = b * h) : A = 450 := 
by
  rw [hb, hh] at hA
  rw [hA]
  sorry

end NUMINAMATH_GPT_parallelogram_area_l388_38836


namespace NUMINAMATH_GPT_coefficient_x2y2_l388_38832

theorem coefficient_x2y2 : 
  let expr1 := (1 + x) ^ 3
  let expr2 := (1 + y) ^ 4
  let C3_2 := Nat.choose 3 2
  let C4_2 := Nat.choose 4 2
  (C3_2 * C4_2 = 18) := by
    sorry

end NUMINAMATH_GPT_coefficient_x2y2_l388_38832


namespace NUMINAMATH_GPT_children_too_heavy_l388_38850

def Kelly_weight : ℝ := 34
def Sam_weight : ℝ := 40
def Daisy_weight : ℝ := 28
def Megan_weight := 1.1 * Kelly_weight
def Mike_weight := Megan_weight + 5

def Total_weight := Kelly_weight + Sam_weight + Daisy_weight + Megan_weight + Mike_weight
def Bridge_limit : ℝ := 130

theorem children_too_heavy :
  Total_weight - Bridge_limit = 51.8 :=
by
  sorry

end NUMINAMATH_GPT_children_too_heavy_l388_38850


namespace NUMINAMATH_GPT_relationship_l388_38866

-- Define sequences
variable (a b : ℕ → ℝ)

-- Define conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → a m = a 1 + (m - 1) * (a n - a 1) / (n - 1)

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → b m = b 1 * (b n / b 1)^(m - 1) / (n - 1)

noncomputable def sequences_conditions : Prop :=
  a 1 = b 1 ∧ a 1 > 0 ∧ ∀ n, a n = b n ∧ b n > 0

-- The main theorem
theorem relationship (h: sequences_conditions a b) : ∀ m n : ℕ, 1 < m → m < n → a m ≥ b m := 
by
  sorry

end NUMINAMATH_GPT_relationship_l388_38866


namespace NUMINAMATH_GPT_square_side_increase_factor_l388_38806

theorem square_side_increase_factor (s k : ℕ) (x new_x : ℕ) (h1 : x = 4 * s) (h2 : new_x = 4 * x) (h3 : new_x = 4 * (k * s)) : k = 4 :=
by
  sorry

end NUMINAMATH_GPT_square_side_increase_factor_l388_38806


namespace NUMINAMATH_GPT_consecutive_odd_numbers_sum_power_fourth_l388_38809

theorem consecutive_odd_numbers_sum_power_fourth :
  ∃ x1 x2 x3 : ℕ, 
  x1 % 2 = 1 ∧ x2 % 2 = 1 ∧ x3 % 2 = 1 ∧ 
  x1 + 2 = x2 ∧ x2 + 2 = x3 ∧ 
  (∃ n : ℕ, n < 10 ∧ (x1 + x2 + x3 = n^4)) :=
sorry

end NUMINAMATH_GPT_consecutive_odd_numbers_sum_power_fourth_l388_38809


namespace NUMINAMATH_GPT_max_value_of_f_in_interval_l388_38870

noncomputable def f (x m : ℝ) : ℝ := -x^3 + 3 * x^2 + m

theorem max_value_of_f_in_interval (m : ℝ) (h₁ : ∀ x ∈ [-2, 2], - x^3 + 3 * x^2 + m ≥ 1) : 
  ∃ x ∈ [-2, 2], f x m = 21 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_in_interval_l388_38870


namespace NUMINAMATH_GPT_abs_add_lt_abs_sub_l388_38848

-- Define the conditions
variables {a b : ℝ} (h1 : a * b < 0)

-- Prove the statement
theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| := sorry

end NUMINAMATH_GPT_abs_add_lt_abs_sub_l388_38848


namespace NUMINAMATH_GPT_bill_difference_zero_l388_38869

theorem bill_difference_zero (l m : ℝ) 
  (hL : (25 / 100) * l = 5) 
  (hM : (15 / 100) * m = 3) : 
  l - m = 0 := 
sorry

end NUMINAMATH_GPT_bill_difference_zero_l388_38869


namespace NUMINAMATH_GPT_average_of_values_l388_38864

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end NUMINAMATH_GPT_average_of_values_l388_38864


namespace NUMINAMATH_GPT_elizabeth_net_profit_l388_38896

-- Define the conditions
def cost_per_bag : ℝ := 3.00
def bags_produced : ℕ := 20
def selling_price_per_bag : ℝ := 6.00
def bags_sold_full_price : ℕ := 15
def discount_percentage : ℝ := 0.25

-- Define the net profit computation
def net_profit : ℝ :=
  let revenue_full_price := bags_sold_full_price * selling_price_per_bag
  let remaining_bags := bags_produced - bags_sold_full_price
  let discounted_price_per_bag := selling_price_per_bag * (1 - discount_percentage)
  let revenue_discounted := remaining_bags * discounted_price_per_bag
  let total_revenue := revenue_full_price + revenue_discounted
  let total_cost := bags_produced * cost_per_bag
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.50 := by
  sorry

end NUMINAMATH_GPT_elizabeth_net_profit_l388_38896


namespace NUMINAMATH_GPT_parabola_distance_focus_l388_38863

theorem parabola_distance_focus (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 16) : x = 3 := by
  sorry

end NUMINAMATH_GPT_parabola_distance_focus_l388_38863


namespace NUMINAMATH_GPT_divisor_count_l388_38846

theorem divisor_count (m : ℕ) (h : m = 2^15 * 5^12) :
  let m_squared := m * m
  let num_divisors_m := (15 + 1) * (12 + 1)
  let num_divisors_m_squared := (30 + 1) * (24 + 1)
  let divisors_of_m_squared_less_than_m := (num_divisors_m_squared - 1) / 2
  num_divisors_m_squared - num_divisors_m = 179 :=
by
  subst h
  sorry

end NUMINAMATH_GPT_divisor_count_l388_38846


namespace NUMINAMATH_GPT_second_quadrant_point_l388_38849

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_second_quadrant_point_l388_38849


namespace NUMINAMATH_GPT_perimeter_ACFHK_is_correct_l388_38817

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the points of the pentagon within the dodecagon
def ACFHK_points : ℕ := 5

-- Define the perimeter of the pentagon ACFHK in the dodecagon
noncomputable def perimeter_of_ACFHK : ℝ :=
  let triangle_side := radius
  let isosceles_right_triangle_side := radius * Real.sqrt 2
  3 * triangle_side + 2 * isosceles_right_triangle_side

-- Verify that the calculated perimeter matches the expected value
theorem perimeter_ACFHK_is_correct : perimeter_of_ACFHK = 18 + 12 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_perimeter_ACFHK_is_correct_l388_38817


namespace NUMINAMATH_GPT_game_result_l388_38825

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_sequence : List ℕ := [2, 5, 6, 4, 3]
def jill_sequence : List ℕ := [1, 6, 3, 2, 5]

def calculate_score (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc + g x) 0

theorem game_result : calculate_score jack_sequence * calculate_score jill_sequence = 420 :=
by
  sorry

end NUMINAMATH_GPT_game_result_l388_38825


namespace NUMINAMATH_GPT_min_segments_for_octagon_perimeter_l388_38871

/-- Given an octagon formed by cutting a smaller rectangle from a larger rectangle,
the minimum number of distinct line segment lengths needed to calculate the perimeter 
of this octagon is 3. --/
theorem min_segments_for_octagon_perimeter (a b c d e f g h : ℝ)
  (cond : a = c ∧ b = d ∧ e = g ∧ f = h) :
  ∃ (u v w : ℝ), u ≠ v ∧ v ≠ w ∧ u ≠ w :=
by
  sorry

end NUMINAMATH_GPT_min_segments_for_octagon_perimeter_l388_38871


namespace NUMINAMATH_GPT_percentage_cities_in_range_l388_38860

-- Definitions of percentages as given conditions
def percentage_cities_between_50k_200k : ℕ := 40
def percentage_cities_below_50k : ℕ := 35
def percentage_cities_above_200k : ℕ := 25

-- Statement of the problem
theorem percentage_cities_in_range :
  percentage_cities_between_50k_200k = 40 := 
by
  sorry

end NUMINAMATH_GPT_percentage_cities_in_range_l388_38860


namespace NUMINAMATH_GPT_free_time_left_after_cleaning_l388_38884

-- Define the time it takes for each task
def vacuuming_time : ℤ := 45
def dusting_time : ℤ := 60
def mopping_time : ℤ := 30
def brushing_time_per_cat : ℤ := 5
def number_of_cats : ℤ := 3
def total_free_time_in_minutes : ℤ := 3 * 60 -- 3 hours converted to minutes

-- Define the total cleaning time
def total_cleaning_time : ℤ := vacuuming_time + dusting_time + mopping_time + (brushing_time_per_cat * number_of_cats)

-- Prove that the free time left after cleaning is 30 minutes
theorem free_time_left_after_cleaning : (total_free_time_in_minutes - total_cleaning_time) = 30 :=
by
  sorry

end NUMINAMATH_GPT_free_time_left_after_cleaning_l388_38884


namespace NUMINAMATH_GPT_no_solutions_for_a_gt_1_l388_38814

theorem no_solutions_for_a_gt_1 (a b : ℝ) (h_a_gt_1 : 1 < a) :
  ¬∃ x : ℝ, a^(2-2*x^2) + (b+4) * a^(1-x^2) + 3*b + 4 = 0 ↔ 0 < b ∧ b < 4 :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_for_a_gt_1_l388_38814


namespace NUMINAMATH_GPT_base_256_6_digits_l388_38877

theorem base_256_6_digits (b : ℕ) (h1 : b ^ 5 ≤ 256) (h2 : 256 < b ^ 6) : b = 3 := 
sorry

end NUMINAMATH_GPT_base_256_6_digits_l388_38877


namespace NUMINAMATH_GPT_triangle_side_b_l388_38888

open Real

variable {a b c : ℝ} (A B C : ℝ)

theorem triangle_side_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin B = 6 * cos A * sin C) : b = 3 :=
sorry

end NUMINAMATH_GPT_triangle_side_b_l388_38888


namespace NUMINAMATH_GPT_factor_polynomials_l388_38838

theorem factor_polynomials :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomials_l388_38838


namespace NUMINAMATH_GPT_anne_bob_total_difference_l388_38810

-- Define specific values as constants
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.08

-- Define the calculations according to Anne's method
def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)

-- Define the calculations according to Bob's method
def bob_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

-- State the theorem that the difference between Anne's and Bob's totals is zero
theorem anne_bob_total_difference : anne_total - bob_total = 0 :=
by sorry  -- Proof not required

end NUMINAMATH_GPT_anne_bob_total_difference_l388_38810


namespace NUMINAMATH_GPT_last_digit_p_adic_l388_38842

theorem last_digit_p_adic (a : ℤ) (p : ℕ) (hp : Nat.Prime p) (h_last_digit_nonzero : a % p ≠ 0) : (a ^ (p - 1) - 1) % p = 0 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_p_adic_l388_38842


namespace NUMINAMATH_GPT_find_k_and_b_l388_38882

theorem find_k_and_b (k b : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
  ((P.1 - 1)^2 + P.2^2 = 1) ∧ 
  ((Q.1 - 1)^2 + Q.2^2 = 1) ∧ 
  (P.2 = k * P.1) ∧ 
  (Q.2 = k * Q.1) ∧ 
  (P.1 - P.2 + b = 0) ∧ 
  (Q.1 - Q.2 + b = 0) ∧ 
  ((P.1 + Q.1) / 2 = (P.2 + Q.2) / 2)) →
  k = -1 ∧ b = -1 :=
sorry

end NUMINAMATH_GPT_find_k_and_b_l388_38882


namespace NUMINAMATH_GPT_fourth_function_form_l388_38839

variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)
variable (hf : Function.LeftInverse f_inv f)
variable (hf_inv : Function.RightInverse f_inv f)

theorem fourth_function_form :
  (∀ x, y = (-(f (-x - 1)) + 2) ↔ y = f_inv (x + 2) + 1 ↔ -(x + y) = 0) :=
  sorry

end NUMINAMATH_GPT_fourth_function_form_l388_38839


namespace NUMINAMATH_GPT_axis_of_parabola_l388_38894

-- Define the given equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8 * y

-- Define the standard form of a vertical parabola and the value we need to prove (axis of the parabola)
def standard_form (p y : ℝ) : Prop := y = 2

-- The proof problem: Given the equation of the parabola, prove the equation of its axis.
theorem axis_of_parabola : 
  ∀ x y : ℝ, (parabola x y) → (standard_form y 2) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_axis_of_parabola_l388_38894


namespace NUMINAMATH_GPT_shaded_fraction_is_one_eighth_l388_38818

noncomputable def total_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

noncomputable def half_area (length : ℕ) (width : ℕ) : ℚ :=
  total_area length width / 2

noncomputable def shaded_area (length : ℕ) (width : ℕ) : ℚ :=
  half_area length width / 4

theorem shaded_fraction_is_one_eighth : 
  ∀ (length width : ℕ), length = 15 → width = 21 → shaded_area length width / total_area length width = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_is_one_eighth_l388_38818


namespace NUMINAMATH_GPT_prove_a5_l388_38867

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end NUMINAMATH_GPT_prove_a5_l388_38867


namespace NUMINAMATH_GPT_max_regular_hours_l388_38829

/-- A man's regular pay is $3 per hour up to a certain number of hours, and his overtime pay rate
    is twice the regular pay rate. The man was paid $180 and worked 10 hours overtime.
    Prove that the maximum number of hours he can work at his regular pay rate is 40 hours.
-/
theorem max_regular_hours (P R OT : ℕ) (hP : P = 180) (hOT : OT = 10) (reg_rate overtime_rate : ℕ)
  (hreg_rate : reg_rate = 3) (hovertime_rate : overtime_rate = 2 * reg_rate) :
  P = reg_rate * R + overtime_rate * OT → R = 40 :=
by
  sorry

end NUMINAMATH_GPT_max_regular_hours_l388_38829


namespace NUMINAMATH_GPT_root_conditions_imply_sum_l388_38815

-- Define the variables a and b in the context that their values fit the given conditions.
def a : ℝ := 5
def b : ℝ := -6

-- Define the quadratic equation and conditions on roots.
def quadratic_eq (x : ℝ) := x^2 - a * x - b

-- Given that 2 and 3 are the roots of the quadratic equation.
def roots_condition := (quadratic_eq 2 = 0) ∧ (quadratic_eq 3 = 0)

-- The theorem to prove.
theorem root_conditions_imply_sum :
  roots_condition → a + b = -1 :=
by
sorry

end NUMINAMATH_GPT_root_conditions_imply_sum_l388_38815


namespace NUMINAMATH_GPT_Larry_spends_108_minutes_l388_38876

-- Define conditions
def half_hour_twice_daily := 30 * 2
def fifth_of_an_hour_daily := 60 / 5
def quarter_hour_twice_daily := 15 * 2
def tenth_of_an_hour_daily := 60 / 10

-- Define total times spent on each pet
def total_time_dog := half_hour_twice_daily + fifth_of_an_hour_daily
def total_time_cat := quarter_hour_twice_daily + tenth_of_an_hour_daily

-- Define the total time spent on pets
def total_time_pets := total_time_dog + total_time_cat

-- Lean theorem statement
theorem Larry_spends_108_minutes : total_time_pets = 108 := 
  by 
    sorry

end NUMINAMATH_GPT_Larry_spends_108_minutes_l388_38876


namespace NUMINAMATH_GPT_no_four_distinct_nat_dividing_pairs_l388_38811

theorem no_four_distinct_nat_dividing_pairs (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∣ (b - c)) (h8 : a ∣ (b - d))
  (h9 : a ∣ (c - d)) (h10 : b ∣ (a - c)) (h11 : b ∣ (a - d)) (h12 : b ∣ (c - d))
  (h13 : c ∣ (a - b)) (h14 : c ∣ (a - d)) (h15 : c ∣ (b - d)) (h16 : d ∣ (a - b))
  (h17 : d ∣ (a - c)) (h18 : d ∣ (b - c)) : False := 
sorry

end NUMINAMATH_GPT_no_four_distinct_nat_dividing_pairs_l388_38811


namespace NUMINAMATH_GPT_friday_lending_tuesday_vs_thursday_total_lending_l388_38844

def standard_lending_rate : ℕ := 50
def monday_excess : ℤ := 0
def tuesday_excess : ℤ := 8
def wednesday_excess : ℤ := 6
def thursday_shortfall : ℤ := -3
def friday_shortfall : ℤ := -7

theorem friday_lending : (standard_lending_rate + friday_shortfall) = 43 := by
  sorry

theorem tuesday_vs_thursday : (tuesday_excess - thursday_shortfall) = 11 := by
  sorry

theorem total_lending : 
  (5 * standard_lending_rate + (monday_excess + tuesday_excess + wednesday_excess + thursday_shortfall + friday_shortfall)) = 254 := by
  sorry

end NUMINAMATH_GPT_friday_lending_tuesday_vs_thursday_total_lending_l388_38844


namespace NUMINAMATH_GPT_square_of_larger_number_is_1156_l388_38861

theorem square_of_larger_number_is_1156
  (x y : ℕ)
  (h1 : x + y = 60)
  (h2 : x - y = 8) :
  x^2 = 1156 := by
  sorry

end NUMINAMATH_GPT_square_of_larger_number_is_1156_l388_38861


namespace NUMINAMATH_GPT_customers_left_l388_38875

-- Definitions based on problem conditions
def initial_customers : ℕ := 14
def remaining_customers : ℕ := 3

-- Theorem statement based on the question and the correct answer
theorem customers_left : initial_customers - remaining_customers = 11 := by
  sorry

end NUMINAMATH_GPT_customers_left_l388_38875


namespace NUMINAMATH_GPT_train_crossing_time_l388_38802

def train_length : ℕ := 320
def train_speed_kmh : ℕ := 72
def kmh_to_ms (v : ℕ) : ℕ := v * 1000 / 3600
def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh
def crossing_time (length : ℕ) (speed : ℕ) : ℕ := length / speed

theorem train_crossing_time : crossing_time train_length train_speed_ms = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_train_crossing_time_l388_38802


namespace NUMINAMATH_GPT_greatest_int_less_than_150_with_gcd_30_eq_5_l388_38804

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end NUMINAMATH_GPT_greatest_int_less_than_150_with_gcd_30_eq_5_l388_38804


namespace NUMINAMATH_GPT_exists_three_with_gcd_d_l388_38854

theorem exists_three_with_gcd_d (n : ℕ) (nums : Fin n.succ → ℕ) (d : ℕ)
  (h1 : n ≥ 2)  -- because n+1 (number of elements nums : Fin n.succ) ≥ 3 given that n ≥ 2
  (h2 : ∀ i, nums i > 0) 
  (h3 : ∀ i, nums i ≤ 100) 
  (h4 : Nat.gcd (nums 0) (Nat.gcd (nums 1) (nums 2)) = d) : 
  ∃ i j k : Fin n.succ, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ Nat.gcd (nums i) (Nat.gcd (nums j) (nums k)) = d :=
by
  sorry

end NUMINAMATH_GPT_exists_three_with_gcd_d_l388_38854


namespace NUMINAMATH_GPT_remainder_of_sum_div_10_l388_38855

theorem remainder_of_sum_div_10 : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_div_10_l388_38855


namespace NUMINAMATH_GPT_cooking_time_l388_38895

theorem cooking_time (total_potatoes cooked_potatoes potato_time : ℕ) 
    (h1 : total_potatoes = 15) 
    (h2 : cooked_potatoes = 6) 
    (h3 : potato_time = 8) : 
    total_potatoes - cooked_potatoes * potato_time = 72 :=
by
    sorry

end NUMINAMATH_GPT_cooking_time_l388_38895


namespace NUMINAMATH_GPT_balloon_permutations_count_l388_38843

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end NUMINAMATH_GPT_balloon_permutations_count_l388_38843


namespace NUMINAMATH_GPT_division_pairs_l388_38856

theorem division_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (ab^2 + b + 7) ∣ (a^2 * b + a + b) →
  (∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k) ∨ (a, b) = (11, 1) ∨ (a, b) = (49, 1) :=
sorry

end NUMINAMATH_GPT_division_pairs_l388_38856


namespace NUMINAMATH_GPT_saturday_price_is_correct_l388_38852

-- Define Thursday's price
def thursday_price : ℝ := 50

-- Define the price increase rate on Friday
def friday_increase_rate : ℝ := 0.2

-- Define the discount rate on Saturday
def saturday_discount_rate : ℝ := 0.15

-- Calculate the price on Friday
def friday_price : ℝ := thursday_price * (1 + friday_increase_rate)

-- Calculate the discount amount on Saturday
def saturday_discount : ℝ := friday_price * saturday_discount_rate

-- Calculate the price on Saturday
def saturday_price : ℝ := friday_price - saturday_discount

-- Theorem stating the price on Saturday
theorem saturday_price_is_correct : saturday_price = 51 := by
  -- Definitions are already embedded into the conditions
  -- so here we only state the property to be proved.
  sorry

end NUMINAMATH_GPT_saturday_price_is_correct_l388_38852


namespace NUMINAMATH_GPT_jason_total_expenditure_l388_38816

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end NUMINAMATH_GPT_jason_total_expenditure_l388_38816


namespace NUMINAMATH_GPT_range_of_M_l388_38881

theorem range_of_M (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by
  -- We would start the proof here by using the given constraints
  sorry

end NUMINAMATH_GPT_range_of_M_l388_38881


namespace NUMINAMATH_GPT_correct_calculation_l388_38847

theorem correct_calculation :
  ∃ (a : ℤ), (a^2 + a^2 = 2 * a^2) ∧ 
  (¬(3*a + 4*(a : ℤ) = 12*a*(a : ℤ))) ∧ 
  (¬((a*(a : ℤ)^2)^3 = a*(a : ℤ)^6)) ∧ 
  (¬((a + 3)^2 = a^2 + 9)) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l388_38847


namespace NUMINAMATH_GPT_average_hamburgers_per_day_l388_38862

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end NUMINAMATH_GPT_average_hamburgers_per_day_l388_38862


namespace NUMINAMATH_GPT_find_a_if_lines_perpendicular_l388_38891

-- Define the lines and the statement about their perpendicularity
theorem find_a_if_lines_perpendicular 
    (a : ℝ)
    (h_perpendicular : (2 * a) / (3 * (a - 1)) = 1) :
    a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_lines_perpendicular_l388_38891


namespace NUMINAMATH_GPT_log_three_div_square_l388_38835

theorem log_three_div_square (x y : ℝ) (h₁ : x ≠ 1) (h₂ : y ≠ 1) (h₃ : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h₄ : x * y = 243) :
  (Real.log (x / y) / Real.log 3) ^ 2 = 9 := 
sorry

end NUMINAMATH_GPT_log_three_div_square_l388_38835


namespace NUMINAMATH_GPT_exists_prime_q_l388_38803

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) (h2 : 2 < p) : 
  ∃ q : ℕ, Nat.Prime q ∧ q < p ∧ ¬ (p ^ 2 ∣ q ^ (p - 1) - 1) := 
sorry

end NUMINAMATH_GPT_exists_prime_q_l388_38803


namespace NUMINAMATH_GPT_solve_frac_eqn_l388_38868

theorem solve_frac_eqn (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) +
   1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8) ↔ 
  (x = 13 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_solve_frac_eqn_l388_38868


namespace NUMINAMATH_GPT_compute_expression_l388_38823

theorem compute_expression :
  20 * ((144 / 3) + (36 / 6) + (16 / 32) + 2) = 1130 := sorry

end NUMINAMATH_GPT_compute_expression_l388_38823


namespace NUMINAMATH_GPT_gcd_8251_6105_l388_38878

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end NUMINAMATH_GPT_gcd_8251_6105_l388_38878


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l388_38889

-- 1. Prove that (3ab³)² = 9a²b⁶
theorem problem1 (a b : ℝ) : (3 * a * b^3)^2 = 9 * a^2 * b^6 :=
by sorry

-- 2. Prove that x ⋅ x³ + x² ⋅ x² = 2x⁴
theorem problem2 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 :=
by sorry

-- 3. Prove that (12x⁴ - 6x³) ÷ 3x² = 4x² - 2x
theorem problem3 (x : ℝ) : (12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l388_38889


namespace NUMINAMATH_GPT_op_add_mul_example_l388_38893

def op_add (a b : ℤ) : ℤ := a + b - 1
def op_mul (a b : ℤ) : ℤ := a * b - 1

theorem op_add_mul_example : op_mul (op_add 6 8) (op_add 3 5) = 90 :=
by
  -- Rewriting it briefly without proof steps
  sorry

end NUMINAMATH_GPT_op_add_mul_example_l388_38893


namespace NUMINAMATH_GPT_min_final_exam_score_l388_38833

theorem min_final_exam_score (q1 q2 q3 q4 final_exam : ℤ)
    (H1 : q1 = 90) (H2 : q2 = 85) (H3 : q3 = 77) (H4 : q4 = 96) :
    (1/2) * (q1 + q2 + q3 + q4) / 4 + (1/2) * final_exam ≥ 90 ↔ final_exam ≥ 93 :=
by
    sorry

end NUMINAMATH_GPT_min_final_exam_score_l388_38833


namespace NUMINAMATH_GPT_tan_half_angle_product_zero_l388_38820

theorem tan_half_angle_product_zero (a b : ℝ) 
  (h: 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) 
  : Real.tan (a / 2) * Real.tan (b / 2) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_tan_half_angle_product_zero_l388_38820


namespace NUMINAMATH_GPT_simplify_fraction_l388_38840

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l388_38840


namespace NUMINAMATH_GPT_candy_not_chocolate_l388_38892

theorem candy_not_chocolate (candy_total : ℕ) (bags : ℕ) (choc_heart_bags : ℕ) (choc_kiss_bags : ℕ) : 
  candy_total = 63 ∧ bags = 9 ∧ choc_heart_bags = 2 ∧ choc_kiss_bags = 3 → 
  (candy_total - (choc_heart_bags * (candy_total / bags) + choc_kiss_bags * (candy_total / bags))) = 28 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_candy_not_chocolate_l388_38892


namespace NUMINAMATH_GPT_rectangle_sides_l388_38807

theorem rectangle_sides (k : ℝ) (μ : ℝ) (a b : ℝ) 
  (h₀ : k = 8) 
  (h₁ : μ = 3/10) 
  (h₂ : 2 * (a + b) = k) 
  (h₃ : a * b = μ * (a^2 + b^2)) : 
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) :=
sorry

end NUMINAMATH_GPT_rectangle_sides_l388_38807


namespace NUMINAMATH_GPT_op_neg2_3_l388_38827

def op (a b : ℤ) : ℤ := a^2 + 2 * a * b

theorem op_neg2_3 : op (-2) 3 = -8 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_op_neg2_3_l388_38827


namespace NUMINAMATH_GPT_find_p_root_relation_l388_38885

theorem find_p_root_relation (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 = 3 * x1 ∧ x1^2 + p * x1 + 2 * p = 0 ∧ x2^2 + p * x2 + 2 * p = 0) ↔ (p = 0 ∨ p = 32 / 3) :=
by sorry

end NUMINAMATH_GPT_find_p_root_relation_l388_38885


namespace NUMINAMATH_GPT_cos_double_angle_l388_38853

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 5) = Real.sqrt 3 / 3) :
  Real.cos (2 * α + 2 * Real.pi / 5) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l388_38853


namespace NUMINAMATH_GPT_Iain_pennies_left_l388_38824

theorem Iain_pennies_left (initial_pennies older_pennies : ℕ) (percentage : ℝ)
  (h_initial : initial_pennies = 200)
  (h_older : older_pennies = 30)
  (h_percentage : percentage = 0.20) :
  initial_pennies - older_pennies - (percentage * (initial_pennies - older_pennies)) = 136 :=
by
  sorry

end NUMINAMATH_GPT_Iain_pennies_left_l388_38824


namespace NUMINAMATH_GPT_smallest_top_block_number_l388_38851

-- Define the pyramid structure and number assignment problem
def block_pyramid : Type := sorry

-- Given conditions:
-- 4 layers, specific numberings, and block support structure.
structure Pyramid :=
  (Layer1 : Fin 16 → ℕ)
  (Layer2 : Fin 9 → ℕ)
  (Layer3 : Fin 4 → ℕ)
  (TopBlock : ℕ)

-- Constraints on block numbers
def is_valid (P : Pyramid) : Prop :=
  -- base layer numbers are from 1 to 16
  (∀ i, 1 ≤ P.Layer1 i ∧ P.Layer1 i ≤ 16) ∧
  -- each above block is the sum of directly underlying neighboring blocks
  (∀ i, P.Layer2 i = P.Layer1 (i * 3) + P.Layer1 (i * 3 + 1) + P.Layer1 (i * 3 + 2)) ∧
  (∀ i, P.Layer3 i = P.Layer2 (i * 3) + P.Layer2 (i * 3 + 1) + P.Layer2 (i * 3 + 2)) ∧
  P.TopBlock = P.Layer3 0 + P.Layer3 1 + P.Layer3 2 + P.Layer3 3

-- Statement of the theorem
theorem smallest_top_block_number : ∃ P : Pyramid, is_valid P ∧ P.TopBlock = ComputedValue := sorry

end NUMINAMATH_GPT_smallest_top_block_number_l388_38851


namespace NUMINAMATH_GPT_milkshakes_per_hour_l388_38886

variable (L : ℕ) -- number of milkshakes Luna can make per hour

theorem milkshakes_per_hour
  (h1 : ∀ (A : ℕ), A = 3) -- Augustus makes 3 milkshakes per hour
  (h2 : ∀ (H : ℕ), H = 8) -- they have been making milkshakes for 8 hours
  (h3 : ∀ (Total : ℕ), Total = 80) -- together they made 80 milkshakes
  (h4 : ∀ (Augustus_milkshakes : ℕ), Augustus_milkshakes = 3 * 8) -- Augustus made 24 milkshakes in 8 hours
 : L = 7 := sorry

end NUMINAMATH_GPT_milkshakes_per_hour_l388_38886


namespace NUMINAMATH_GPT_kangaroo_can_jump_1000_units_l388_38858

noncomputable def distance (x y : ℕ) : ℕ := x + y

def valid_small_jump (x y : ℕ) : Prop :=
  x + 1 ≥ 0 ∧ y - 1 ≥ 0

def valid_big_jump (x y : ℕ) : Prop :=
  x - 5 ≥ 0 ∧ y + 7 ≥ 0

theorem kangaroo_can_jump_1000_units (x y : ℕ) (h : x + y > 6) :
  distance x y ≥ 1000 :=
sorry

end NUMINAMATH_GPT_kangaroo_can_jump_1000_units_l388_38858


namespace NUMINAMATH_GPT_ratio_both_to_onlyB_is_2_l388_38897

variables (num_A num_B both: ℕ)

-- Given conditions
axiom A_eq_2B : num_A = 2 * num_B
axiom both_eq_500 : both = 500
axiom both_multiple_of_only_B : ∃ k : ℕ, both = k * (num_B - both)
axiom only_A_eq_1000 : (num_A - both) = 1000

-- Define the Lean theorem statement
theorem ratio_both_to_onlyB_is_2 : (both : ℝ) / (num_B - both : ℝ) = 2 := 
sorry

end NUMINAMATH_GPT_ratio_both_to_onlyB_is_2_l388_38897


namespace NUMINAMATH_GPT_geometric_progression_a5_value_l388_38880

theorem geometric_progression_a5_value
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n, a (n + 1) = a n * r)
  (h_roots : ∃ x y, x^2 - 5*x + 4 = 0 ∧ y^2 - 5*y + 4 = 0 ∧ x = a 3 ∧ y = a 7) :
  a 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_a5_value_l388_38880


namespace NUMINAMATH_GPT_JakeMowingEarnings_l388_38805

theorem JakeMowingEarnings :
  (∀ rate hours_mowing hours_planting (total_charge : ℝ),
      rate = 20 →
      hours_mowing = 1 →
      hours_planting = 2 →
      total_charge = 45 →
      (total_charge = hours_planting * rate + 5) →
      hours_mowing * rate = 20) :=
by
  intros rate hours_mowing hours_planting total_charge
  sorry

end NUMINAMATH_GPT_JakeMowingEarnings_l388_38805


namespace NUMINAMATH_GPT_distinct_integers_sum_of_three_elems_l388_38801

-- Define the set S and the property of its elements
def S : Set ℕ := {1, 4, 7, 10, 13, 16, 19}

-- Define the property that each element in S is of the form 3k + 1
def is_form_3k_plus_1 (x : ℕ) : Prop := ∃ k : ℤ, x = 3 * k + 1

theorem distinct_integers_sum_of_three_elems (h₁ : ∀ x ∈ S, is_form_3k_plus_1 x) :
  (∃! n, n = 13) :=
by
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_of_three_elems_l388_38801


namespace NUMINAMATH_GPT_gcd_linear_combination_l388_38857

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := by
  sorry

end NUMINAMATH_GPT_gcd_linear_combination_l388_38857


namespace NUMINAMATH_GPT_deductible_increase_l388_38826

theorem deductible_increase (current_deductible : ℝ) (increase_fraction : ℝ) (next_year_deductible : ℝ) : 
  current_deductible = 3000 ∧ increase_fraction = 2 / 3 ∧ next_year_deductible = (1 + increase_fraction) * current_deductible →
  next_year_deductible - current_deductible = 2000 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_deductible_increase_l388_38826


namespace NUMINAMATH_GPT_initial_interest_rate_l388_38837

variable (P r : ℕ)

theorem initial_interest_rate (h1 : 405 = (P * r) / 100) (h2 : 450 = (P * (r + 5)) / 100) : r = 45 :=
sorry

end NUMINAMATH_GPT_initial_interest_rate_l388_38837


namespace NUMINAMATH_GPT_inequality_system_solution_l388_38890

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l388_38890


namespace NUMINAMATH_GPT_distinct_roots_difference_l388_38830

theorem distinct_roots_difference (r s : ℝ) (h₀ : r ≠ s) (h₁ : r > s) (h₂ : ∀ x, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) :
  r - s = Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_GPT_distinct_roots_difference_l388_38830


namespace NUMINAMATH_GPT_value_of_a_l388_38821

-- Define the variables and conditions as lean definitions/constants
variable (a b c : ℝ)
variable (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variable (h2 : a * 15 * 11 = 1)

-- Statement to prove
theorem value_of_a : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l388_38821


namespace NUMINAMATH_GPT_gcd_solution_l388_38808

noncomputable def gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : ℤ :=
  Int.gcd (4 * b^2 + 35 * b + 72) (3 * b + 8)

theorem gcd_solution (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : gcd_problem b h = 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_solution_l388_38808


namespace NUMINAMATH_GPT_guilt_proof_l388_38822

variables (E F G : Prop)

theorem guilt_proof
  (h1 : ¬G → F)
  (h2 : ¬E → G)
  (h3 : G → E)
  (h4 : E → ¬F)
  : E ∧ G :=
by
  sorry

end NUMINAMATH_GPT_guilt_proof_l388_38822


namespace NUMINAMATH_GPT_simplify_f_l388_38813

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) (h : Real.sin (α - 3 / 2 * Real.pi) = 1 / 5) : f α = -1 / 5 := by
  sorry

end NUMINAMATH_GPT_simplify_f_l388_38813


namespace NUMINAMATH_GPT_store_money_left_l388_38899

variable (total_items : Nat) (original_price : ℝ) (discount_percent : ℝ)
variable (percent_sold : ℝ) (amount_owed : ℝ)

theorem store_money_left
  (h_total_items : total_items = 2000)
  (h_original_price : original_price = 50)
  (h_discount_percent : discount_percent = 0.80)
  (h_percent_sold : percent_sold = 0.90)
  (h_amount_owed : amount_owed = 15000)
  : (total_items * original_price * (1 - discount_percent) * percent_sold - amount_owed) = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_store_money_left_l388_38899


namespace NUMINAMATH_GPT_tom_travel_time_to_virgo_island_l388_38831

-- Definitions based on conditions
def boat_trip_time : ℕ := 2
def plane_trip_time : ℕ := 4 * boat_trip_time
def total_trip_time : ℕ := plane_trip_time + boat_trip_time

-- Theorem we need to prove
theorem tom_travel_time_to_virgo_island : total_trip_time = 10 := by
  sorry

end NUMINAMATH_GPT_tom_travel_time_to_virgo_island_l388_38831


namespace NUMINAMATH_GPT_number_of_solutions_proof_l388_38865

noncomputable def number_of_real_solutions (x y z w : ℝ) : ℝ :=
  if (x = z + w + 2 * z * w * x) ∧ (y = w + x + 2 * w * x * y) ∧ (z = x + y + 2 * x * y * z) ∧ (w = y + z + 2 * y * z * w) then
    5
  else
    0

theorem number_of_solutions_proof :
  ∃ x y z w : ℝ, x = z + w + 2 * z * w * x ∧ y = w + x + 2 * w * x * y ∧ z = x + y + 2 * x * y * z ∧ w = y + z + 2 * y * z * w → number_of_real_solutions x y z w = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_proof_l388_38865


namespace NUMINAMATH_GPT_symmetric_point_origin_l388_38828

-- Define the original point P with given coordinates
def P : ℝ × ℝ := (-2, 3)

-- Define the symmetric point P' with respect to the origin
def P'_symmetric (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The theorem states that the symmetric point of P is (2, -3)
theorem symmetric_point_origin : P'_symmetric P = (2, -3) := 
by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l388_38828


namespace NUMINAMATH_GPT_shopping_people_count_l388_38812

theorem shopping_people_count :
  ∃ P : ℕ, P = 10 ∧
  ∃ (stores : ℕ) (total_visits : ℕ) (two_store_visitors : ℕ) 
    (at_least_one_store_visitors : ℕ) (max_stores_visited : ℕ),
    stores = 8 ∧
    total_visits = 22 ∧
    two_store_visitors = 8 ∧
    at_least_one_store_visitors = P ∧
    max_stores_visited = 3 ∧
    total_visits = (two_store_visitors * 2) + 6 ∧
    P = two_store_visitors + 2 :=
by {
    sorry
}

end NUMINAMATH_GPT_shopping_people_count_l388_38812
