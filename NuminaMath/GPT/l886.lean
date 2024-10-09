import Mathlib

namespace girls_together_count_l886_88630

-- Define the problem conditions
def boys : ℕ := 4
def girls : ℕ := 2
def total_entities : ℕ := boys + (girls - 1) -- One entity for the two girls together

-- Calculate the factorial
noncomputable def factorial (n: ℕ) : ℕ :=
  if n = 0 then 1 else (List.range (n+1)).foldl (λx y => x * y) 1

-- Define the total number of ways girls can be together
noncomputable def ways_girls_together : ℕ :=
  factorial total_entities * factorial girls

-- State the theorem that needs to be proved
theorem girls_together_count : ways_girls_together = 240 := by
  sorry

end girls_together_count_l886_88630


namespace Jasper_height_in_10_minutes_l886_88683

noncomputable def OmarRate : ℕ := 240 / 12
noncomputable def JasperRate : ℕ := 3 * OmarRate
noncomputable def JasperHeight (time: ℕ) : ℕ := JasperRate * time

theorem Jasper_height_in_10_minutes :
  JasperHeight 10 = 600 :=
by
  sorry

end Jasper_height_in_10_minutes_l886_88683


namespace timber_volume_after_two_years_correct_l886_88666

-- Definitions based on the conditions in the problem
variables (a p b : ℝ) -- Assume a, p, and b are real numbers

-- Timber volume after one year
def timber_volume_one_year (a p b : ℝ) : ℝ := a * (1 + p) - b

-- Timber volume after two years
def timber_volume_two_years (a p b : ℝ) : ℝ := (timber_volume_one_year a p b) * (1 + p) - b

-- Prove that the timber volume after two years is equal to the given expression
theorem timber_volume_after_two_years_correct (a p b : ℝ) :
  timber_volume_two_years a p b = a * (1 + p)^2 - (2 + p) * b := sorry

end timber_volume_after_two_years_correct_l886_88666


namespace maximize_root_product_l886_88633

theorem maximize_root_product :
  (∃ k : ℝ, ∀ x : ℝ, 6 * x^2 - 5 * x + k = 0 ∧ (25 - 24 * k ≥ 0)) →
  ∃ k : ℝ, k = 25 / 24 :=
by
  sorry

end maximize_root_product_l886_88633


namespace red_toys_removed_l886_88644

theorem red_toys_removed (R W : ℕ) (h1 : R + W = 134) (h2 : 2 * W = 88) (h3 : R - 2 * W / 2 = 88) : R - 88 = 2 :=
by {
  sorry
}

end red_toys_removed_l886_88644


namespace units_digit_k_squared_plus_pow2_k_l886_88656

def n : ℕ := 4016
def k : ℕ := n^2 + 2^n

theorem units_digit_k_squared_plus_pow2_k :
  (k^2 + 2^k) % 10 = 7 := sorry

end units_digit_k_squared_plus_pow2_k_l886_88656


namespace reflection_matrix_values_l886_88697

theorem reflection_matrix_values (a b : ℝ) (I : Matrix (Fin 2) (Fin 2) ℝ) :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 9/26], ![b, 17/26]]
  (R * R = I) → a = -17/26 ∧ b = 0 :=
by
  sorry

end reflection_matrix_values_l886_88697


namespace initial_balls_in_bag_l886_88637

theorem initial_balls_in_bag (n : ℕ) 
  (h_add_white : ∀ x : ℕ, x = n + 1)
  (h_probability : (5 / 8) = 0.625):
  n = 7 :=
sorry

end initial_balls_in_bag_l886_88637


namespace total_black_dots_l886_88695

def num_butterflies : ℕ := 397
def black_dots_per_butterfly : ℕ := 12

theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end total_black_dots_l886_88695


namespace ratio_of_rises_l886_88615

noncomputable def radius_narrower_cone : ℝ := 4
noncomputable def radius_wider_cone : ℝ := 8
noncomputable def sphere_radius : ℝ := 2

noncomputable def height_ratio (h1 h2 : ℝ) : Prop := h1 = 4 * h2

noncomputable def volume_displacement := (4 / 3) * Real.pi * (sphere_radius^3)

noncomputable def new_height_narrower (h1 : ℝ) : ℝ := h1 + (volume_displacement / ((Real.pi * (radius_narrower_cone^2))))

noncomputable def new_height_wider (h2 : ℝ) : ℝ := h2 + (volume_displacement / ((Real.pi * (radius_wider_cone^2))))

theorem ratio_of_rises (h1 h2 : ℝ) (hr : height_ratio h1 h2) :
  (new_height_narrower h1 - h1) / (new_height_wider h2 - h2) = 4 :=
sorry

end ratio_of_rises_l886_88615


namespace meaningful_expression_range_l886_88668

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l886_88668


namespace expansion_correct_l886_88692

variable (x y : ℝ)

theorem expansion_correct : 
  (3 * x - 15) * (4 * y + 20) = 12 * x * y + 60 * x - 60 * y - 300 :=
by
  sorry

end expansion_correct_l886_88692


namespace num_mappings_from_A_to_A_is_4_l886_88613

-- Define the number of elements in set A
def set_A_card := 2

-- Define the proof problem
theorem num_mappings_from_A_to_A_is_4 (h : set_A_card = 2) : (set_A_card ^ set_A_card) = 4 :=
by
  sorry

end num_mappings_from_A_to_A_is_4_l886_88613


namespace GuntherFreeTime_l886_88661

def GuntherCleaning : Nat := 45 + 60 + 30 + 15

def TotalFreeTime : Nat := 180

theorem GuntherFreeTime : TotalFreeTime - GuntherCleaning = 30 := by
  sorry

end GuntherFreeTime_l886_88661


namespace right_triangle_area_perimeter_ratio_l886_88698

theorem right_triangle_area_perimeter_ratio :
  let a := 4
  let b := 8
  let area := (1/2) * a * b
  let c := Real.sqrt (a^2 + b^2)
  let perimeter := a + b + c
  let ratio := area / perimeter
  ratio = 3 - Real.sqrt 5 :=
by
  sorry

end right_triangle_area_perimeter_ratio_l886_88698


namespace Tom_total_yearly_intake_l886_88611

def soda_weekday := 5 * 12
def water_weekday := 64
def juice_weekday := 3 * 8
def sports_drink_weekday := 2 * 16

def total_weekday_intake := soda_weekday + water_weekday + juice_weekday + sports_drink_weekday

def soda_weekend_holiday := 5 * 12
def water_weekend_holiday := 64
def juice_weekend_holiday := 3 * 8
def sports_drink_weekend_holiday := 1 * 16
def fruit_smoothie_weekend_holiday := 32

def total_weekend_holiday_intake := soda_weekend_holiday + water_weekend_holiday + juice_weekend_holiday + sports_drink_weekend_holiday + fruit_smoothie_weekend_holiday

def weekdays := 260
def weekend_days := 104
def holidays := 1

def total_yearly_intake := (weekdays * total_weekday_intake) + (weekend_days * total_weekend_holiday_intake) + (holidays * total_weekend_holiday_intake)

theorem Tom_total_yearly_intake :
  total_yearly_intake = 67380 := by
  sorry

end Tom_total_yearly_intake_l886_88611


namespace find_n_l886_88696

theorem find_n (n : ℕ) : (Nat.lcm n 10 = 36) ∧ (Nat.gcd n 10 = 5) → n = 18 :=
by
  -- The proof will be provided here
  sorry

end find_n_l886_88696


namespace probability_A_B_l886_88673

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l886_88673


namespace second_eq_value_l886_88684

variable (x y z w : ℝ)

theorem second_eq_value (h1 : 4 * x * z + y * w = 3) (h2 : (2 * x + y) * (2 * z + w) = 15) : 
  x * w + y * z = 6 :=
by
  sorry

end second_eq_value_l886_88684


namespace fraction_to_decimal_l886_88650

theorem fraction_to_decimal :
  (3 / 8 : ℝ) = 0.375 :=
sorry

end fraction_to_decimal_l886_88650


namespace overall_winning_percentage_is_fifty_l886_88678

def winning_percentage_of_first_games := (40 / 100) * 30
def total_games_played := 40
def remaining_games := total_games_played - 30
def winning_percentage_of_remaining_games := (80 / 100) * remaining_games
def total_games_won := winning_percentage_of_first_games + winning_percentage_of_remaining_games

theorem overall_winning_percentage_is_fifty : 
  (total_games_won / total_games_played) * 100 = 50 := 
by
  sorry

end overall_winning_percentage_is_fifty_l886_88678


namespace second_alloy_amount_l886_88690

theorem second_alloy_amount (x : ℝ) : 
  (0.10 * 15 + 0.08 * x = 0.086 * (15 + x)) → 
  x = 35 := by 
sorry

end second_alloy_amount_l886_88690


namespace students_in_class_l886_88601

theorem students_in_class (n S : ℕ) 
    (h1 : S = 15 * n)
    (h2 : (S + 56) / (n + 1) = 16) : n = 40 :=
by
  sorry

end students_in_class_l886_88601


namespace simplified_value_of_f_l886_88672

variable (x : ℝ)

noncomputable def f : ℝ := 3 * x + 5 - 4 * x^2 + 2 * x - 7 + x^2 - 3 * x + 8

theorem simplified_value_of_f : f x = -3 * x^2 + 2 * x + 6 := by
  unfold f
  sorry

end simplified_value_of_f_l886_88672


namespace sum_mod_9_is_6_l886_88676

noncomputable def sum_modulo_9 : ℤ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

theorem sum_mod_9_is_6 : sum_modulo_9 % 9 = 6 := 
  by
    sorry

end sum_mod_9_is_6_l886_88676


namespace vertical_asymptotes_polynomial_l886_88602

theorem vertical_asymptotes_polynomial (a b : ℝ) (h₁ : -3 * 2 = b) (h₂ : -3 + 2 = a) : a + b = -5 := by
  sorry

end vertical_asymptotes_polynomial_l886_88602


namespace min_y_ellipse_l886_88639

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 49) + ((y - 3)^2 / 25) = 1

-- Problem statement: Prove that the smallest y-coordinate is -2
theorem min_y_ellipse : 
  ∀ x y, ellipse x y → y ≥ -2 :=
sorry

end min_y_ellipse_l886_88639


namespace systematic_sample_seat_number_l886_88687

theorem systematic_sample_seat_number (total_students sample_size interval : ℕ) (seat1 seat2 seat3 : ℕ) 
  (H_total_students : total_students = 56)
  (H_sample_size : sample_size = 4)
  (H_interval : interval = total_students / sample_size)
  (H_seat1 : seat1 = 3)
  (H_seat2 : seat2 = 31)
  (H_seat3 : seat3 = 45) :
  ∃ seat4 : ℕ, seat4 = 17 :=
by 
  sorry

end systematic_sample_seat_number_l886_88687


namespace tickets_sold_in_total_l886_88627

def total_tickets
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ) : ℕ :=
  adult_tickets + student_tickets

theorem tickets_sold_in_total 
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ)
    (h1 : adult_price = 6)
    (h2 : student_price = 3)
    (h3 : total_revenue = 3846)
    (h4 : adult_tickets = 410)
    (h5 : student_tickets = 436) :
  total_tickets adult_price student_price total_revenue adult_tickets student_tickets = 846 :=
by
  sorry

end tickets_sold_in_total_l886_88627


namespace prob_diff_fruit_correct_l886_88688

noncomputable def prob_same_all_apple : ℝ := (0.4)^3
noncomputable def prob_same_all_orange : ℝ := (0.3)^3
noncomputable def prob_same_all_banana : ℝ := (0.2)^3
noncomputable def prob_same_all_grape : ℝ := (0.1)^3

noncomputable def prob_same_fruit_all_day : ℝ := 
  prob_same_all_apple + prob_same_all_orange + prob_same_all_banana + prob_same_all_grape

noncomputable def prob_diff_fruit (prob_same : ℝ) : ℝ := 1 - prob_same

theorem prob_diff_fruit_correct :
  prob_diff_fruit prob_same_fruit_all_day = 0.9 :=
by
  sorry

end prob_diff_fruit_correct_l886_88688


namespace no_three_real_numbers_satisfy_inequalities_l886_88652

theorem no_three_real_numbers_satisfy_inequalities (a b c : ℝ) :
  ¬ (|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b| ) :=
by
  sorry

end no_three_real_numbers_satisfy_inequalities_l886_88652


namespace find_ff_of_five_half_l886_88617

noncomputable def f (x : ℝ) : ℝ :=
if x <= 1 then 2^x - 2 else Real.log x / Real.log 2

theorem find_ff_of_five_half : f (f (5/2)) = -1/2 := by
  sorry

end find_ff_of_five_half_l886_88617


namespace range_of_x_minus_2y_l886_88655

theorem range_of_x_minus_2y (x y : ℝ) (h₁ : -1 ≤ x) (h₂ : x < 2) (h₃ : 0 < y) (h₄ : y ≤ 1) :
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 :=
sorry

end range_of_x_minus_2y_l886_88655


namespace camel_water_ratio_l886_88663

theorem camel_water_ratio (gallons_water : ℕ) (ounces_per_gallon : ℕ) (traveler_ounces : ℕ)
  (total_ounces : ℕ) (camel_ounces : ℕ) (ratio : ℕ) 
  (h1 : gallons_water = 2) 
  (h2 : ounces_per_gallon = 128) 
  (h3 : traveler_ounces = 32) 
  (h4 : total_ounces = gallons_water * ounces_per_gallon) 
  (h5 : camel_ounces = total_ounces - traveler_ounces)
  (h6 : ratio = camel_ounces / traveler_ounces) : 
  ratio = 7 := 
by
  sorry

end camel_water_ratio_l886_88663


namespace find_slope_of_line_l886_88622

-- Define the parabola, point M, and the conditions leading to the slope k.
theorem find_slope_of_line (k : ℝ) :
  let C := {p : ℝ × ℝ | p.2^2 = 4 * p.1}
  let focus : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (-1, 1)
  let line (k : ℝ) (x : ℝ) := k * (x - 1)
  ∃ A B : (ℝ × ℝ), 
    A ∈ C ∧ B ∈ C ∧
    A ≠ B ∧
    A.1 + 1 = B.1 + 1 ∧ 
    A.2 - 1 = B.2 - 1 ∧
    ((A.1 + 1) * (B.1 + 1) + (A.2 - 1) * (B.2 - 1) = 0) -> k = 2 := 
by
  sorry

end find_slope_of_line_l886_88622


namespace largest_t_value_maximum_t_value_l886_88675

noncomputable def largest_t : ℚ :=
  (5 : ℚ) / 2

theorem largest_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ (5 : ℚ) / 2 :=
sorry

theorem maximum_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  (5 : ℚ) / 2 = largest_t :=
sorry

end largest_t_value_maximum_t_value_l886_88675


namespace actual_number_of_sides_l886_88691

theorem actual_number_of_sides (apparent_angle : ℝ) (distortion_factor : ℝ)
  (sum_exterior_angles : ℝ) (actual_sides : ℕ) :
  apparent_angle = 18 ∧ distortion_factor = 1.5 ∧ sum_exterior_angles = 360 ∧ 
  apparent_angle / distortion_factor = sum_exterior_angles / actual_sides →
  actual_sides = 30 :=
by
  sorry

end actual_number_of_sides_l886_88691


namespace new_total_weight_correct_l886_88641

-- Definitions based on the problem statement
variables (R S k : ℝ)
def ram_original_weight : ℝ := 2 * k
def shyam_original_weight : ℝ := 5 * k
def ram_new_weight : ℝ := 1.10 * (ram_original_weight k)
def shyam_new_weight : ℝ := 1.17 * (shyam_original_weight k)

-- Definition for total original weight and increased weight
def total_original_weight : ℝ := ram_original_weight k + shyam_original_weight k
def total_weight_increased : ℝ := 1.15 * total_original_weight k
def new_total_weight : ℝ := ram_new_weight k + shyam_new_weight k

-- The proof statement
theorem new_total_weight_correct :
  new_total_weight k = total_weight_increased k :=
by
  sorry

end new_total_weight_correct_l886_88641


namespace fifth_largest_divisor_of_1209600000_is_75600000_l886_88628

theorem fifth_largest_divisor_of_1209600000_is_75600000 :
  let n : ℤ := 1209600000
  let fifth_largest_divisor : ℤ := 75600000
  n = 2^10 * 5^5 * 3 * 503 →
  fifth_largest_divisor = n / 2^5 :=
by
  sorry

end fifth_largest_divisor_of_1209600000_is_75600000_l886_88628


namespace find_difference_l886_88657

variable (k1 k2 t1 t2 : ℝ)

theorem find_difference (h1 : t1 = 5 / 9 * (k1 - 32))
                        (h2 : t2 = 5 / 9 * (k2 - 32))
                        (h3 : t1 = 105)
                        (h4 : t2 = 80) :
  k1 - k2 = 45 :=
by
  sorry

end find_difference_l886_88657


namespace impossible_distinct_values_l886_88664

theorem impossible_distinct_values :
  ∀ a b c : ℝ, 
  (a * (a - 4) = 12) → 
  (b * (b - 4) = 12) → 
  (c * (c - 4) = 12) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  false := 
sorry

end impossible_distinct_values_l886_88664


namespace negation_of_proposition_l886_88680

theorem negation_of_proposition
  (h : ∀ x : ℝ, x^2 - 2 * x + 2 > 0) :
  ∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0 :=
sorry

end negation_of_proposition_l886_88680


namespace small_glass_cost_l886_88689

theorem small_glass_cost 
  (S : ℝ)
  (small_glass_cost : ℝ)
  (large_glass_cost : ℝ := 5)
  (initial_money : ℝ := 50)
  (num_small : ℝ := 8)
  (change : ℝ := 1)
  (num_large : ℝ := 5)
  (spent_money : ℝ := initial_money - change)
  (total_large_cost : ℝ := num_large * large_glass_cost)
  (total_cost : ℝ := num_small * S + total_large_cost)
  (total_cost_eq : total_cost = spent_money) :
  S = 3 :=
by
  sorry

end small_glass_cost_l886_88689


namespace jessica_balloons_l886_88681

-- Given conditions
def joan_balloons : Nat := 9
def sally_balloons : Nat := 5
def total_balloons : Nat := 16

-- The theorem to prove the number of balloons Jessica has
theorem jessica_balloons : (total_balloons - (joan_balloons + sally_balloons) = 2) :=
by
  -- Proof goes here
  sorry

end jessica_balloons_l886_88681


namespace fixed_point_exists_l886_88648

noncomputable def fixed_point : Prop := ∀ d : ℝ, ∃ (p q : ℝ), (p = -3) ∧ (q = 45) ∧ (q = 5 * p^2 + d * p + 3 * d)

theorem fixed_point_exists : fixed_point :=
by
  sorry

end fixed_point_exists_l886_88648


namespace min_sum_intercepts_of_line_l886_88654

theorem min_sum_intercepts_of_line (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : a + b = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_intercepts_of_line_l886_88654


namespace radar_arrangements_l886_88604

-- Define the number of letters in the word RADAR
def total_letters : Nat := 5

-- Define the number of times each letter is repeated
def repetition_R : Nat := 2
def repetition_A : Nat := 2

-- The expected number of unique arrangements
def expected_unique_arrangements : Nat := 30

theorem radar_arrangements :
  (Nat.factorial total_letters) / (Nat.factorial repetition_R * Nat.factorial repetition_A) = expected_unique_arrangements := by
  sorry

end radar_arrangements_l886_88604


namespace solution_set_quadratic_inequality_l886_88659

def quadraticInequalitySolutionSet 
  (x : ℝ) : Prop := 
  3 + 5 * x - 2 * x^2 > 0

theorem solution_set_quadratic_inequality :
  { x : ℝ | quadraticInequalitySolutionSet x } = 
  { x : ℝ | - (1:ℝ) / 2 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_quadratic_inequality_l886_88659


namespace fraction_to_decimal_l886_88653

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l886_88653


namespace congruent_semicircles_span_diameter_l886_88636

theorem congruent_semicircles_span_diameter (N : ℕ) (r : ℝ) 
  (h1 : 2 * N * r = 2 * (N * r)) 
  (h2 : (N * (π * r^2 / 2)) / ((N^2 * (π * r^2 / 2)) - (N * (π * r^2 / 2))) = 1/4) 
  : N = 5 :=
by
  sorry

end congruent_semicircles_span_diameter_l886_88636


namespace seven_n_form_l886_88632

theorem seven_n_form (n : ℤ) (a b : ℤ) (h : 7 * n = a^2 + 3 * b^2) : 
  ∃ c d : ℤ, n = c^2 + 3 * d^2 :=
by {
  sorry
}

end seven_n_form_l886_88632


namespace expansion_number_of_terms_l886_88674

theorem expansion_number_of_terms (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 5) : (A.card * B.card = 20) :=
by 
  sorry

end expansion_number_of_terms_l886_88674


namespace value_of_k_l886_88693

theorem value_of_k (k : ℝ) (h1 : k ≠ 0) (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k * x₁ - 100) < (k * x₂ - 100)) : k = 1 :=
by
  have h3 : k > 0 :=
    sorry -- We know that if y increases as x increases, then k > 0
  have h4 : k = 1 :=
    sorry -- For this specific problem, we can take k = 1 which satisfies the conditions
  exact h4

end value_of_k_l886_88693


namespace line_tangent_to_ellipse_l886_88623

theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x : ℝ, x^2 + 4 * (m * x + 1)^2 = 1) → m^2 = 3 / 4 :=
by
  sorry

end line_tangent_to_ellipse_l886_88623


namespace total_age_proof_l886_88647

variable (K : ℕ) -- Kaydence's age
variable (T : ℕ) -- Total age of people in the gathering

def Kaydence_father_age : ℕ := 60
def Kaydence_mother_age : ℕ := Kaydence_father_age - 2
def Kaydence_brother_age : ℕ := Kaydence_father_age / 2
def Kaydence_sister_age : ℕ := 40
def elder_cousin_age : ℕ := Kaydence_brother_age + 2 * Kaydence_sister_age
def younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
def grandmother_age : ℕ := 3 * Kaydence_mother_age - 5

theorem total_age_proof (K : ℕ) : T = 525 + K :=
by 
  sorry

end total_age_proof_l886_88647


namespace largest_non_sum_of_multiple_of_30_and_composite_l886_88618

theorem largest_non_sum_of_multiple_of_30_and_composite :
  ∃ (n : ℕ), n = 211 ∧ ∀ a b : ℕ, (a > 0) → (b > 0) → (b < 30) → 
  n ≠ 30 * a + b ∧ ¬ ∃ k : ℕ, k > 1 ∧ k < b ∧ b % k = 0 :=
sorry

end largest_non_sum_of_multiple_of_30_and_composite_l886_88618


namespace susie_remaining_money_l886_88645

noncomputable def calculate_remaining_money : Float :=
  let weekday_hours := 4.0
  let weekday_rate := 12.0
  let weekdays := 5.0
  let weekend_hours := 2.5
  let weekend_rate := 15.0
  let weekends := 2.0
  let total_weekday_earnings := weekday_hours * weekday_rate * weekdays
  let total_weekend_earnings := weekend_hours * weekend_rate * weekends
  let total_earnings := total_weekday_earnings + total_weekend_earnings
  let spent_makeup := 3 / 8 * total_earnings
  let remaining_after_makeup := total_earnings - spent_makeup
  let spent_skincare := 2 / 5 * remaining_after_makeup
  let remaining_after_skincare := remaining_after_makeup - spent_skincare
  let spent_cellphone := 1 / 6 * remaining_after_skincare
  let final_remaining := remaining_after_skincare - spent_cellphone
  final_remaining

theorem susie_remaining_money : calculate_remaining_money = 98.4375 := by
  sorry

end susie_remaining_money_l886_88645


namespace first_pump_half_time_l886_88651

theorem first_pump_half_time (t : ℝ) : 
  (∃ (t : ℝ), (1/(2*t) + 1/1.1111111111111112) * (1/2) = 1/2) -> 
  t = 5 :=
by
  sorry

end first_pump_half_time_l886_88651


namespace find_primes_l886_88699

theorem find_primes (p : ℕ) (x y : ℕ) (hx : x > 0) (hy : y > 0) (hp : Nat.Prime p) : 
  (x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) := sorry

end find_primes_l886_88699


namespace intersection_M_N_l886_88670

noncomputable def M : Set ℕ := { x | 0 < x ∧ x < 8 }
def N : Set ℕ := { x | ∃ n : ℕ, x = 2 * n + 1 }
def K : Set ℕ := { 1, 3, 5, 7 }

theorem intersection_M_N : M ∩ N = K :=
by sorry

end intersection_M_N_l886_88670


namespace derivative_at_one_l886_88629

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : deriv f 1 = 2 * Real.exp 1 := by
sorry

end derivative_at_one_l886_88629


namespace max_bananas_l886_88694

theorem max_bananas (a o b : ℕ) (h_a : a ≥ 1) (h_o : o ≥ 1) (h_b : b ≥ 1) (h_eq : 3 * a + 5 * o + 8 * b = 100) : b ≤ 11 :=
by {
  sorry
}

end max_bananas_l886_88694


namespace minimum_value_f_range_of_m_l886_88665

noncomputable def f (x m : ℝ) := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f (m : ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : 
  if m ≤ 2 then f x m = 2 - m 
  else if m ≥ Real.exp 1 + 1 then f x m = Real.exp 1 - m - (m - 1) / Real.exp 1 
  else f x m = m - 2 - m * Real.log (m - 1) :=
sorry

theorem range_of_m (m : ℝ) :
  (m ≤ 2 ∧ ∀ x2 ∈ [-2, 0], ∃ x1 ∈ [Real.exp 1, Real.exp 2], f x1 m ≤ g x2) ↔
  (m ∈ [ (Real.exp 2 - Real.exp 1 + 1) / (Real.exp 1 + 1), 2 ]) :=
sorry

end minimum_value_f_range_of_m_l886_88665


namespace shopkeeper_net_loss_percent_l886_88658

theorem shopkeeper_net_loss_percent (cp : ℝ)
  (sp1 sp2 sp3 sp4 : ℝ)
  (h_cp : cp = 1000)
  (h_sp1 : sp1 = cp * 1.1)
  (h_sp2 : sp2 = cp * 0.9)
  (h_sp3 : sp3 = cp * 1.2)
  (h_sp4 : sp4 = cp * 0.75) :
  ((cp + cp + cp + cp) - (sp1 + sp2 + sp3 + sp4)) / (cp + cp + cp + cp) * 100 = 1.25 :=
by sorry

end shopkeeper_net_loss_percent_l886_88658


namespace given_problem_l886_88616

theorem given_problem (x y : ℝ) (hx : x ≠ 0) (hx4 : x ≠ 4) (hy : y ≠ 0) (hy6 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
sorry

end given_problem_l886_88616


namespace angela_january_additional_sleep_l886_88643

-- Definitions corresponding to conditions in part a)
def december_sleep_hours : ℝ := 6.5
def january_sleep_hours : ℝ := 8.5
def days_in_january : ℕ := 31

-- The proof statement, proving the January's additional sleep hours
theorem angela_january_additional_sleep :
  (january_sleep_hours - december_sleep_hours) * days_in_january = 62 :=
by
  -- Since the focus is only on the statement, we skip the actual proof.
  sorry

end angela_january_additional_sleep_l886_88643


namespace g_45_l886_88667

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y
axiom g_30 : g 30 = 30

theorem g_45 : g 45 = 20 := by
  -- proof to be completed
  sorry

end g_45_l886_88667


namespace sin_thirty_degree_l886_88621

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l886_88621


namespace maximum_n_for_sequence_l886_88612

theorem maximum_n_for_sequence :
  ∃ (n : ℕ), 
  (∀ a S : ℕ → ℝ, 
    a 1 = 1 → 
    (∀ n : ℕ, n > 0 → 2 * a (n + 1) + S n = 2) → 
    (1001 / 1000 < S (2 * n) / S n ∧ S (2 * n) / S n < 11 / 10)) →
  n = 9 :=
sorry

end maximum_n_for_sequence_l886_88612


namespace system1_solution_system2_solution_l886_88669

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end system1_solution_system2_solution_l886_88669


namespace population_decrease_rate_l886_88686

theorem population_decrease_rate (r : ℕ) (h₀ : 6000 > 0) (h₁ : 4860 = 6000 * (1 - r / 100)^2) : r = 10 :=
by sorry

end population_decrease_rate_l886_88686


namespace find_a_with_constraints_l886_88606

theorem find_a_with_constraints (x y a : ℝ) 
  (h1 : 2 * x - y + 2 ≥ 0) 
  (h2 : x - 3 * y + 1 ≤ 0)
  (h3 : x + y - 2 ≤ 0)
  (h4 : a > 0)
  (h5 : ∃ (x1 x2 x3 y1 y2 y3 : ℝ), 
    ((x1, y1) = (1, 1) ∨ (x1, y1) = (5 / 3, 1 / 3) ∨ (x1, y1) = (2, 0)) ∧ 
    ((x2, y2) = (1, 1) ∨ (x2, y2) = (5 / 3, 1 / 3) ∨ (x2, y2) = (2, 0)) ∧ 
    ((x3, y3) = (1, 1) ∨ (x3, y3) = (5 / 3, 1 / 3) ∨ (x3, y3) = (2, 0)) ∧ 
    (ax1 - y1 = ax2 - y2) ∧ (ax2 - y2 = ax3 - y3)) :
  a = 1 / 3 :=
sorry

end find_a_with_constraints_l886_88606


namespace solution_set_inequalities_l886_88607

theorem solution_set_inequalities (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 2 * x) / 3 > x - 1) → (x ≤ 1) :=
by
  intros h
  sorry

end solution_set_inequalities_l886_88607


namespace gray_region_area_l886_88642

-- Definitions based on given conditions
def radius_inner (r : ℝ) := r
def radius_outer (r : ℝ) := r + 3

-- Statement to prove: the area of the gray region
theorem gray_region_area (r : ℝ) : 
  (π * (radius_outer r)^2 - π * (radius_inner r)^2) = 6 * π * r + 9 * π := by
  sorry

end gray_region_area_l886_88642


namespace B_and_C_mutually_exclusive_l886_88608

-- Defining events in terms of products being defective or not
def all_not_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, ¬x

def all_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, x

def not_all_defective (products : List Bool) : Prop := 
  ∃ x ∈ products, ¬x

-- Given a batch of three products, define events A, B, and C
def A (products : List Bool) : Prop := all_not_defective products
def B (products : List Bool) : Prop := all_defective products
def C (products : List Bool) : Prop := not_all_defective products

-- The theorem to prove that B and C are mutually exclusive
theorem B_and_C_mutually_exclusive (products : List Bool) (h : products.length = 3) : 
  ¬ (B products ∧ C products) :=
by
  sorry

end B_and_C_mutually_exclusive_l886_88608


namespace prime_divisors_of_1320_l886_88679

theorem prime_divisors_of_1320 : 
  ∃ (S : Finset ℕ), (S = {2, 3, 5, 11}) ∧ S.card = 4 := 
by
  sorry

end prime_divisors_of_1320_l886_88679


namespace area_of_square_l886_88609

-- Definitions
def radius_ratio (r R : ℝ) : Prop := R = 7 / 3 * r
def small_circle_circumference (r : ℝ) : Prop := 2 * Real.pi * r = 8
def square_side_length (R side : ℝ) : Prop := side = 2 * R
def square_area (side area : ℝ) : Prop := area = side * side

-- Problem statement
theorem area_of_square (r R side area : ℝ) 
    (h1 : radius_ratio r R)
    (h2 : small_circle_circumference r)
    (h3 : square_side_length R side)
    (h4 : square_area side area) :
    area = 3136 / (9 * Real.pi^2) := 
  by sorry

end area_of_square_l886_88609


namespace ten_digit_number_l886_88685

open Nat

theorem ten_digit_number (a : Fin 10 → ℕ) (h1 : a 4 = 2)
  (h2 : a 8 = 3)
  (h3 : ∀ i, i < 8 → a i * a (i + 1) * a (i + 2) = 24) :
  a = ![4, 2, 3, 4, 2, 3, 4, 2, 3, 4] :=
sorry

end ten_digit_number_l886_88685


namespace jordan_width_45_l886_88662

noncomputable def carolRectangleLength : ℕ := 15
noncomputable def carolRectangleWidth : ℕ := 24
noncomputable def jordanRectangleLength : ℕ := 8
noncomputable def carolRectangleArea : ℕ := carolRectangleLength * carolRectangleWidth
noncomputable def jordanRectangleWidth (area : ℕ) : ℕ := area / jordanRectangleLength

theorem jordan_width_45 : jordanRectangleWidth carolRectangleArea = 45 :=
by sorry

end jordan_width_45_l886_88662


namespace tiffany_found_bags_l886_88682

theorem tiffany_found_bags (initial_bags : ℕ) (total_bags : ℕ) (found_bags : ℕ) :
  initial_bags = 4 ∧ total_bags = 12 ∧ total_bags = initial_bags + found_bags → found_bags = 8 :=
by
  sorry

end tiffany_found_bags_l886_88682


namespace parallel_condition_l886_88625

theorem parallel_condition (a : ℝ) : (a = -1) ↔ (¬ (a = -1 ∧ a ≠ 1)) ∧ (¬ (a ≠ -1 ∧ a = 1)) :=
by
  sorry

end parallel_condition_l886_88625


namespace disproving_iff_l886_88626

theorem disproving_iff (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : (a^2 > b^2) ∧ ¬(a > b) :=
by
  sorry

end disproving_iff_l886_88626


namespace train_length_is_300_l886_88610

theorem train_length_is_300 (L V : ℝ)
    (h1 : L = V * 20)
    (h2 : L + 285 = V * 39) :
    L = 300 := by
  sorry

end train_length_is_300_l886_88610


namespace problem1_sol_l886_88603

noncomputable def problem1 :=
  let total_people := 200
  let avg_feelings_total := 70
  let female_total := 100
  let a := 30 -- derived from 2a + (70 - a) = 100
  let chi_square := 200 * (70 * 40 - 30 * 60) ^ 2 / (130 * 70 * 100 * 100)
  let k_95 := 3.841 -- critical value for 95% confidence
  let p_xi_2 := (1 / 3)
  let p_xi_3 := (1 / 2)
  let p_xi_4 := (1 / 6)
  let exi := (2 * (1 / 3)) + (3 * (1 / 2)) + (4 * (1 / 6))
  chi_square < k_95 ∧ exi = 17 / 6

theorem problem1_sol : problem1 :=
  by {
    sorry
  }

end problem1_sol_l886_88603


namespace part1_monotonicity_part2_inequality_l886_88614

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem part1_monotonicity (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x : ℝ, (x < Real.log (1 / a) → f a x > f a (x + 1)) ∧
  (x > Real.log (1 / a) → f a x < f a (x + 1))) := sorry

theorem part2_inequality (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + (3 / 2) := sorry

end part1_monotonicity_part2_inequality_l886_88614


namespace hypotenuse_is_correct_l886_88635

noncomputable def hypotenuse_of_right_triangle (a b : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_is_correct :
  hypotenuse_of_right_triangle 140 210 = 70 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_is_correct_l886_88635


namespace buns_cost_eq_1_50_l886_88620

noncomputable def meat_cost : ℝ := 2 * 3.50
noncomputable def tomato_cost : ℝ := 1.5 * 2.00
noncomputable def pickles_cost : ℝ := 2.50 - 1.00
noncomputable def lettuce_cost : ℝ := 1.00
noncomputable def total_other_items_cost : ℝ := meat_cost + tomato_cost + pickles_cost + lettuce_cost
noncomputable def total_amount_spent : ℝ := 20.00 - 6.00
noncomputable def buns_cost : ℝ := total_amount_spent - total_other_items_cost

theorem buns_cost_eq_1_50 : buns_cost = 1.50 := by
  sorry

end buns_cost_eq_1_50_l886_88620


namespace solve_absolute_value_equation_l886_88646

theorem solve_absolute_value_equation (x : ℝ) :
  |2 * x - 3| = x + 1 → (x = 4 ∨ x = 2 / 3) := by
  sorry

end solve_absolute_value_equation_l886_88646


namespace number_of_small_cubes_l886_88619

-- Definition of the conditions from the problem
def painted_cube (n : ℕ) :=
  6 * (n - 2) * (n - 2) = 54

-- The theorem we need to prove
theorem number_of_small_cubes (n : ℕ) (h : painted_cube n) : n^3 = 125 :=
by
  have h1 : 6 * (n - 2) * (n - 2) = 54 := h
  sorry

end number_of_small_cubes_l886_88619


namespace solve_for_x_l886_88649

theorem solve_for_x : 
  (35 / (6 - (2 / 5)) = 25 / 4) := 
by
  sorry 

end solve_for_x_l886_88649


namespace g_five_l886_88631

def g (x : ℝ) : ℝ := 4 * x + 2

theorem g_five : g 5 = 22 := by
  sorry

end g_five_l886_88631


namespace quadratic_inequality_solution_l886_88671

theorem quadratic_inequality_solution :
  {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x * (x + 2) < 3} :=
by
  sorry

end quadratic_inequality_solution_l886_88671


namespace inequality_solution_l886_88660

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l886_88660


namespace least_wins_to_40_points_l886_88624

theorem least_wins_to_40_points 
  (points_per_victory : ℕ)
  (points_per_draw : ℕ)
  (points_per_defeat : ℕ)
  (total_matches : ℕ)
  (initial_points : ℕ)
  (matches_played : ℕ)
  (target_points : ℕ) :
  points_per_victory = 3 →
  points_per_draw = 1 →
  points_per_defeat = 0 →
  total_matches = 20 →
  initial_points = 12 →
  matches_played = 5 →
  target_points = 40 →
  ∃ wins_needed : ℕ, wins_needed = 10 :=
by
  sorry

end least_wins_to_40_points_l886_88624


namespace bags_of_hammers_to_load_l886_88600

noncomputable def total_crate_capacity := 15 * 20
noncomputable def weight_of_nails := 4 * 5
noncomputable def weight_of_planks := 10 * 30
noncomputable def weight_to_be_left_out := 80
noncomputable def effective_capacity := total_crate_capacity - weight_to_be_left_out
noncomputable def weight_of_loaded_planks := 220

theorem bags_of_hammers_to_load : (effective_capacity - weight_of_nails - weight_of_loaded_planks = 0) :=
by
  sorry

end bags_of_hammers_to_load_l886_88600


namespace no_green_ball_in_bag_l886_88638

theorem no_green_ball_in_bag (bag : Set String) (h : bag = {"red", "yellow", "white"}): ¬ ("green" ∈ bag) :=
by
  sorry

end no_green_ball_in_bag_l886_88638


namespace leftover_coverage_l886_88640

variable (bagCoverage lawnLength lawnWidth bagsPurchased : ℕ)

def area_of_lawn (length width : ℕ) : ℕ :=
  length * width

def total_coverage (bagCoverage bags : ℕ) : ℕ :=
  bags * bagCoverage

theorem leftover_coverage :
  let lawnLength := 22
  let lawnWidth := 36
  let bagCoverage := 250
  let bagsPurchased := 4
  let lawnArea := area_of_lawn lawnLength lawnWidth
  let totalSeedCoverage := total_coverage bagCoverage bagsPurchased
  totalSeedCoverage - lawnArea = 208 := by
  sorry

end leftover_coverage_l886_88640


namespace fraction_operation_correct_l886_88634

theorem fraction_operation_correct 
  (a b : ℝ) : 
  (0.2 * (3 * a + 10 * b) = 6 * a + 20 * b) → 
  (0.1 * (2 * a + 5 * b) = 2 * a + 5 * b) →
  (∀ c : ℝ, c ≠ 0 → (a / b = (a * c) / (b * c))) ∨
  (∀ x y : ℝ, ((x - y) / (x + y) ≠ (y - x) / (x - y))) ∨
  (∀ x : ℝ, (x + x * x * x + x * y ≠ 1 / x * x)) →
  ((0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b)) :=
sorry

end fraction_operation_correct_l886_88634


namespace chess_tournament_game_count_l886_88605

theorem chess_tournament_game_count (n : ℕ) (h1 : ∃ n, ∀ i j, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ i ≠ j → ∃ games_between, games_between = n ∧ games_between * (Nat.choose 6 2) = 30) : n = 2 :=
by
  sorry

end chess_tournament_game_count_l886_88605


namespace running_speed_l886_88677

theorem running_speed
  (walking_speed : Float)
  (walking_time : Float)
  (running_time : Float)
  (distance : Float) :
  walking_speed = 8 → walking_time = 3 → running_time = 1.5 → distance = walking_speed * walking_time → 
  (distance / running_time) = 16 :=
by
  intros h_walking_speed h_walking_time h_running_time h_distance
  sorry

end running_speed_l886_88677
