import Mathlib

namespace NUMINAMATH_GPT_average_cost_price_per_meter_l2276_227679

noncomputable def average_cost_per_meter (total_cost total_meters : ℝ) : ℝ :=
  total_cost / total_meters

theorem average_cost_price_per_meter :
  let silk_cost := 416.25
  let silk_meters := 9.25
  let cotton_cost := 337.50
  let cotton_meters := 7.5
  let wool_cost := 378.0
  let wool_meters := 6.0
  let total_cost := silk_cost + cotton_cost + wool_cost
  let total_meters := silk_meters + cotton_meters + wool_meters
  average_cost_per_meter total_cost total_meters = 49.75 := by
  sorry

end NUMINAMATH_GPT_average_cost_price_per_meter_l2276_227679


namespace NUMINAMATH_GPT_volume_of_given_wedge_l2276_227695

noncomputable def volume_of_wedge (d : ℝ) (angle : ℝ) : ℝ := 
  let r := d / 2
  let height := d
  let cos_angle := Real.cos angle
  (r^2 * height * Real.pi / 2) * cos_angle

theorem volume_of_given_wedge :
  volume_of_wedge 20 (Real.pi / 6) = 1732 * Real.pi :=
by {
  -- The proof logic will go here.
  sorry
}

end NUMINAMATH_GPT_volume_of_given_wedge_l2276_227695


namespace NUMINAMATH_GPT_cone_slant_height_l2276_227632

theorem cone_slant_height (r l : ℝ) (h1 : r = 1)
  (h2 : 2 * r * Real.pi = (1 / 2) * 2 * l * Real.pi) :
  l = 2 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_cone_slant_height_l2276_227632


namespace NUMINAMATH_GPT_smaller_angle_at_8_15_l2276_227656

noncomputable def hour_hand_position (h m : ℕ) : ℝ := (↑h % 12) * 30 + (↑m / 60) * 30

noncomputable def minute_hand_position (m : ℕ) : ℝ := ↑m / 60 * 360

noncomputable def angle_between_hands (h m : ℕ) : ℝ :=
  let θ := |hour_hand_position h m - minute_hand_position m|
  min θ (360 - θ)

theorem smaller_angle_at_8_15 : angle_between_hands 8 15 = 157.5 := by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_8_15_l2276_227656


namespace NUMINAMATH_GPT_original_pencils_l2276_227651

-- Definition of the conditions
def pencils_initial := 115
def pencils_added := 100
def pencils_total := 215

-- Theorem stating the problem to be proved
theorem original_pencils :
  pencils_initial + pencils_added = pencils_total :=
by
  sorry

end NUMINAMATH_GPT_original_pencils_l2276_227651


namespace NUMINAMATH_GPT_weekly_sales_correct_l2276_227663

open Real

noncomputable def cost_left_handed_mouse (cost_normal_mouse : ℝ) : ℝ :=
  cost_normal_mouse * 1.3

noncomputable def cost_left_handed_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  cost_normal_keyboard * 1.2

noncomputable def cost_left_handed_scissors (cost_normal_scissors : ℝ) : ℝ :=
  cost_normal_scissors * 1.5

noncomputable def daily_sales_mouse (cost_normal_mouse : ℝ) : ℝ :=
  25 * cost_left_handed_mouse cost_normal_mouse

noncomputable def daily_sales_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  10 * cost_left_handed_keyboard cost_normal_keyboard

noncomputable def daily_sales_scissors (cost_normal_scissors : ℝ) : ℝ :=
  15 * cost_left_handed_scissors cost_normal_scissors

noncomputable def bundle_price (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  (cost_left_handed_mouse cost_normal_mouse + cost_left_handed_keyboard cost_normal_keyboard + cost_left_handed_scissors cost_normal_scissors) * 0.9

noncomputable def daily_sales_bundle (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  5 * bundle_price cost_normal_mouse cost_normal_keyboard cost_normal_scissors

noncomputable def weekly_sales (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  3 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors) +
  1.5 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors)

theorem weekly_sales_correct :
  weekly_sales 120 80 30 = 29922.25 := sorry

end NUMINAMATH_GPT_weekly_sales_correct_l2276_227663


namespace NUMINAMATH_GPT_percentage_increase_l2276_227648

theorem percentage_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) : ((M - N) / N) * 100 = P :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l2276_227648


namespace NUMINAMATH_GPT_ab_cd_l2276_227671

theorem ab_cd {a b c d : ℕ} {w x y z : ℕ}
  (hw : Prime w) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (horder : w < x ∧ x < y ∧ y < z)
  (hprod : w^a * x^b * y^c * z^d = 660) :
  (a + b) - (c + d) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ab_cd_l2276_227671


namespace NUMINAMATH_GPT_range_of_a_for_local_maximum_l2276_227697

noncomputable def f' (a x : ℝ) := a * (x + 1) * (x - a)

theorem range_of_a_for_local_maximum {a : ℝ} (hf_max : ∀ x : ℝ, f' a x = 0 → ∀ y : ℝ, y ≠ x → f' a y ≤ f' a x) :
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_local_maximum_l2276_227697


namespace NUMINAMATH_GPT_sqrt_14_range_l2276_227617

theorem sqrt_14_range : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 :=
by
  -- We know that 9 < 14 < 16, so we can take the square root of all parts to get 3 < sqrt(14) < 4.
  sorry

end NUMINAMATH_GPT_sqrt_14_range_l2276_227617


namespace NUMINAMATH_GPT_ethanol_in_full_tank_l2276_227678

theorem ethanol_in_full_tank:
  ∀ (capacity : ℕ) (vol_A : ℕ) (vol_B : ℕ) (eth_A_perc : ℝ) (eth_B_perc : ℝ) (eth_A : ℝ) (eth_B : ℝ),
  capacity = 208 →
  vol_A = 82 →
  vol_B = (capacity - vol_A) →
  eth_A_perc = 0.12 →
  eth_B_perc = 0.16 →
  eth_A = vol_A * eth_A_perc →
  eth_B = vol_B * eth_B_perc →
  eth_A + eth_B = 30 :=
by
  intros capacity vol_A vol_B eth_A_perc eth_B_perc eth_A eth_B h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_ethanol_in_full_tank_l2276_227678


namespace NUMINAMATH_GPT_calculate_total_selling_price_l2276_227673

noncomputable def total_selling_price (cost_price1 cost_price2 cost_price3 profit_percent1 profit_percent2 profit_percent3 : ℝ) : ℝ :=
  let sp1 := cost_price1 + (profit_percent1 / 100 * cost_price1)
  let sp2 := cost_price2 + (profit_percent2 / 100 * cost_price2)
  let sp3 := cost_price3 + (profit_percent3 / 100 * cost_price3)
  sp1 + sp2 + sp3

theorem calculate_total_selling_price :
  total_selling_price 550 750 1000 30 25 20 = 2852.5 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_calculate_total_selling_price_l2276_227673


namespace NUMINAMATH_GPT_line_tangent_to_parabola_j_eq_98_l2276_227680

theorem line_tangent_to_parabola_j_eq_98 (j : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 7 * y + j = 0 → x ≠ 0) →
  j = 98 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_j_eq_98_l2276_227680


namespace NUMINAMATH_GPT_k_lt_zero_l2276_227626

noncomputable def k_negative (k : ℝ) : Prop :=
  (∃ x : ℝ, x < 0 ∧ k * x > 0) ∧ (∃ x : ℝ, x > 0 ∧ k * x < 0)

theorem k_lt_zero (k : ℝ) : k_negative k → k < 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_k_lt_zero_l2276_227626


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2276_227677

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2276_227677


namespace NUMINAMATH_GPT_sin_double_angle_l2276_227662

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l2276_227662


namespace NUMINAMATH_GPT_find_number_l2276_227687

theorem find_number (N : ℕ) (h1 : ∃ k : ℤ, N = 13 * k + 11) (h2 : ∃ m : ℤ, N = 17 * m + 9) : N = 89 := 
sorry

end NUMINAMATH_GPT_find_number_l2276_227687


namespace NUMINAMATH_GPT_relationship_among_abc_l2276_227667

noncomputable def a : ℝ := 4^(1/3 : ℝ)
noncomputable def b : ℝ := Real.log 1/7 / Real.log 3
noncomputable def c : ℝ := (1/3 : ℝ)^(1/5 : ℝ)

theorem relationship_among_abc : a > c ∧ c > b := 
by 
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l2276_227667


namespace NUMINAMATH_GPT_cost_prices_sum_l2276_227693

theorem cost_prices_sum
  (W B : ℝ)
  (h1 : 0.9 * W + 196 = 1.04 * W)
  (h2 : 1.08 * B - 150 = 1.02 * B) :
  W + B = 3900 := 
sorry

end NUMINAMATH_GPT_cost_prices_sum_l2276_227693


namespace NUMINAMATH_GPT_find_number_l2276_227659

theorem find_number :
  ∃ x : ℚ, x * (-1/2) = 1 ↔ x = -2 := 
sorry

end NUMINAMATH_GPT_find_number_l2276_227659


namespace NUMINAMATH_GPT_points_per_vegetable_correct_l2276_227620

-- Given conditions
def total_points_needed : ℕ := 200
def number_of_students : ℕ := 25
def number_of_weeks : ℕ := 2
def veggies_per_student_per_week : ℕ := 2

-- Derived values
def total_veggies_eaten_by_class : ℕ :=
  number_of_students * number_of_weeks * veggies_per_student_per_week

def points_per_vegetable : ℕ :=
  total_points_needed / total_veggies_eaten_by_class

-- Theorem to be proven
theorem points_per_vegetable_correct :
  points_per_vegetable = 2 := by
sorry

end NUMINAMATH_GPT_points_per_vegetable_correct_l2276_227620


namespace NUMINAMATH_GPT_square_area_l2276_227650

theorem square_area (p : ℝ) (h : p = 20) : (p / 4) ^ 2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l2276_227650


namespace NUMINAMATH_GPT_miles_per_gallon_city_l2276_227621

theorem miles_per_gallon_city
  (T : ℝ) -- tank size
  (h c : ℝ) -- miles per gallon on highway 'h' and in the city 'c'
  (h_eq : h = (462 / T))
  (c_eq : c = (336 / T))
  (relation : c = h - 9)
  (solution : c = 24) : c = 24 := 
sorry

end NUMINAMATH_GPT_miles_per_gallon_city_l2276_227621


namespace NUMINAMATH_GPT_tan_cos_solution_count_l2276_227658

theorem tan_cos_solution_count : 
  ∃ (n : ℕ), n = 5 ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.tan (2 * x) = Real.cos (x / 2) → x ∈ Set.Icc 0 (2 * Real.pi) :=
sorry

end NUMINAMATH_GPT_tan_cos_solution_count_l2276_227658


namespace NUMINAMATH_GPT_zeros_indeterminate_in_interval_l2276_227627

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ) (ha : a < b) (hf : f a * f b < 0)

-- The theorem statement
theorem zeros_indeterminate_in_interval :
  (∀ (f : ℝ → ℝ), f a * f b < 0 → (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∨ (∀ (x : ℝ), a < x ∧ x < b → f x ≠ 0) ∨ (∃ (x1 x2 : ℝ), a < x1 ∧ x1 < x2 ∧ x2 < b ∧ f x1 = 0 ∧ f x2 = 0)) :=
by sorry

end NUMINAMATH_GPT_zeros_indeterminate_in_interval_l2276_227627


namespace NUMINAMATH_GPT_ratio_of_trees_l2276_227618

theorem ratio_of_trees (plums pears apricots : ℕ) (h_plums : plums = 3) (h_pears : pears = 3) (h_apricots : apricots = 3) :
  plums = pears ∧ pears = apricots :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_trees_l2276_227618


namespace NUMINAMATH_GPT_emery_family_first_hour_distance_l2276_227655

noncomputable def total_time : ℝ := 4
noncomputable def remaining_distance : ℝ := 300
noncomputable def first_hour_distance : ℝ := 100

theorem emery_family_first_hour_distance :
  (remaining_distance / (total_time - 1)) = first_hour_distance :=
sorry

end NUMINAMATH_GPT_emery_family_first_hour_distance_l2276_227655


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_6_mod_17_l2276_227645

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_6_mod_17_l2276_227645


namespace NUMINAMATH_GPT_intersection_A_B_union_A_compB_l2276_227664

-- Define the sets A and B
def A : Set ℝ := { x | x^2 + 3 * x - 10 < 0 }
def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define the complement of B in the universal set
def comp_B : Set ℝ := { x | ¬ B x }

-- 1. Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_A_B :
  A ∩ B = { x | -5 < x ∧ x ≤ -1 } :=
by 
  sorry

-- 2. Prove that A ∪ (complement of B) = {x | -5 < x ∧ x < 3}
theorem union_A_compB :
  A ∪ comp_B = { x | -5 < x ∧ x < 3 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_union_A_compB_l2276_227664


namespace NUMINAMATH_GPT_number_of_triangles_with_perimeter_27_l2276_227647

theorem number_of_triangles_with_perimeter_27 : 
  ∃ (n : ℕ), (∀ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a + b + c = 27 → a + b > c ∧ a + c > b ∧ b + c > a → 
  n = 19 ) :=
  sorry

end NUMINAMATH_GPT_number_of_triangles_with_perimeter_27_l2276_227647


namespace NUMINAMATH_GPT_period_length_divisor_l2276_227600

theorem period_length_divisor (p d : ℕ) (hp_prime : Nat.Prime p) (hd_period : ∀ n : ℕ, n ≥ 1 → 10^n % p = 1 ↔ n = d) :
  d ∣ (p - 1) :=
sorry

end NUMINAMATH_GPT_period_length_divisor_l2276_227600


namespace NUMINAMATH_GPT_distance_between_Petrovo_and_Nikolaevo_l2276_227630

theorem distance_between_Petrovo_and_Nikolaevo :
  ∃ S : ℝ, (10 + (S - 10) / 4) + (20 + (S - 20) / 3) = S ∧ S = 50 := by
    sorry

end NUMINAMATH_GPT_distance_between_Petrovo_and_Nikolaevo_l2276_227630


namespace NUMINAMATH_GPT_circle_through_origin_and_point_l2276_227665

theorem circle_through_origin_and_point (a r : ℝ) :
  (∃ a r : ℝ, (a^2 + (5 - 3 * a)^2 = r^2) ∧ ((a - 3)^2 + (3 * a - 6)^2 = r^2)) →
  a = 5/3 ∧ r^2 = 25/9 :=
sorry

end NUMINAMATH_GPT_circle_through_origin_and_point_l2276_227665


namespace NUMINAMATH_GPT_bricks_required_l2276_227622

-- Courtyard dimensions in meters
def length_courtyard_m := 23
def width_courtyard_m := 15

-- Brick dimensions in centimeters
def length_brick_cm := 17
def width_brick_cm := 9

-- Conversion from meters to centimeters
def meter_to_cm (m : Int) : Int :=
  m * 100

-- Area of courtyard in square centimeters
def area_courtyard_cm2 : Int :=
  meter_to_cm length_courtyard_m * meter_to_cm width_courtyard_m

-- Area of a single brick in square centimeters
def area_brick_cm2 : Int :=
  length_brick_cm * width_brick_cm

-- Calculate the number of bricks needed, ensuring we round up to the nearest whole number
def total_bricks_needed : Int :=
  (area_courtyard_cm2 + area_brick_cm2 - 1) / area_brick_cm2

-- The theorem stating the total number of bricks needed
theorem bricks_required :
  total_bricks_needed = 22550 := by
  sorry

end NUMINAMATH_GPT_bricks_required_l2276_227622


namespace NUMINAMATH_GPT_B_more_than_C_l2276_227646

variables (A B C : ℕ)
noncomputable def total_subscription : ℕ := 50000
noncomputable def total_profit : ℕ := 35000
noncomputable def A_profit : ℕ := 14700
noncomputable def A_subscr : ℕ := B + 4000

theorem B_more_than_C (B_subscr C_subscr : ℕ) (h1 : A_subscr + B_subscr + C_subscr = total_subscription)
    (h2 : 14700 * 50000 = 35000 * A_subscr) :
    B_subscr - C_subscr = 5000 :=
sorry

end NUMINAMATH_GPT_B_more_than_C_l2276_227646


namespace NUMINAMATH_GPT_amount_after_two_years_l2276_227668

/-- Defining given conditions. -/
def initial_value : ℤ := 65000
def first_year_increase : ℚ := 12 / 100
def second_year_increase : ℚ := 8 / 100

/-- The main statement that needs to be proved. -/
theorem amount_after_two_years : 
  let first_year_amount := initial_value + (initial_value * first_year_increase)
  let second_year_amount := first_year_amount + (first_year_amount * second_year_increase)
  second_year_amount = 78624 := 
by 
  sorry

end NUMINAMATH_GPT_amount_after_two_years_l2276_227668


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l2276_227609

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (5 * Real.pi / 6)
noncomputable def c : ℝ := Real.tan (7 * Real.pi / 5)

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l2276_227609


namespace NUMINAMATH_GPT_amount_paid_out_l2276_227635

theorem amount_paid_out 
  (amount : ℕ) 
  (h1 : amount % 50 = 0) 
  (h2 : ∃ (n : ℕ), n ≥ 15 ∧ amount = n * 5000 ∨ amount = n * 1000)
  (h3 : ∃ (n : ℕ), n ≥ 35 ∧ amount = n * 1000) : 
  amount = 29950 :=
by 
  sorry

end NUMINAMATH_GPT_amount_paid_out_l2276_227635


namespace NUMINAMATH_GPT_num_lighting_methods_l2276_227649

-- Definitions of the problem's conditions
def total_lights : ℕ := 15
def lights_off : ℕ := 6
def lights_on : ℕ := total_lights - lights_off
def available_spaces : ℕ := lights_on - 1

-- Statement of the mathematically equivalent proof problem
theorem num_lighting_methods : Nat.choose available_spaces lights_off = 28 := by
  sorry

end NUMINAMATH_GPT_num_lighting_methods_l2276_227649


namespace NUMINAMATH_GPT_todd_ate_cupcakes_l2276_227657

theorem todd_ate_cupcakes :
  let C := 38   -- Total cupcakes baked by Sarah
  let P := 3    -- Number of packages made
  let c := 8    -- Number of cupcakes per package
  let L := P * c  -- Total cupcakes left after packaging
  C - L = 14 :=  -- Cupcakes Todd ate is 14
by
  sorry

end NUMINAMATH_GPT_todd_ate_cupcakes_l2276_227657


namespace NUMINAMATH_GPT_nested_fraction_expression_l2276_227661

theorem nested_fraction_expression : 
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := 
by sorry

end NUMINAMATH_GPT_nested_fraction_expression_l2276_227661


namespace NUMINAMATH_GPT_cube_faces_sum_39_l2276_227699

theorem cube_faces_sum_39 (a b c d e f g h : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0)
    (vertex_sum : (a*e*b*h + a*e*c*h + a*f*b*h + a*f*c*h + d*e*b*h + d*e*c*h + d*f*b*h + d*f*c*h) = 2002) :
    (a + b + c + d + e + f + g + h) = 39 := 
sorry

end NUMINAMATH_GPT_cube_faces_sum_39_l2276_227699


namespace NUMINAMATH_GPT_arithmetic_sequence_20th_term_l2276_227615

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 3
  let n := 20
  (a + (n - 1) * d) = 59 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_20th_term_l2276_227615


namespace NUMINAMATH_GPT_remainder_sum_div_7_l2276_227602

theorem remainder_sum_div_7 :
  (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_div_7_l2276_227602


namespace NUMINAMATH_GPT_artworks_per_student_in_first_half_l2276_227654

theorem artworks_per_student_in_first_half (x : ℕ) (h1 : 10 = 10) (h2 : 20 = 20) (h3 : 5 * x + 5 * 4 = 35) : x = 3 := by
  sorry

end NUMINAMATH_GPT_artworks_per_student_in_first_half_l2276_227654


namespace NUMINAMATH_GPT_smaller_of_two_numbers_l2276_227603

theorem smaller_of_two_numbers (a b : ℕ) (h1 : a * b = 4761) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 53 :=
by {
  sorry -- proof skips as directed
}

end NUMINAMATH_GPT_smaller_of_two_numbers_l2276_227603


namespace NUMINAMATH_GPT_number_of_erasers_l2276_227613

theorem number_of_erasers (P E : ℕ) (h1 : P + E = 240) (h2 : P = E - 2) : E = 121 := by
  sorry

end NUMINAMATH_GPT_number_of_erasers_l2276_227613


namespace NUMINAMATH_GPT_man_can_lift_one_box_each_hand_l2276_227684

theorem man_can_lift_one_box_each_hand : 
  ∀ (people boxes : ℕ), people = 7 → boxes = 14 → (boxes / people) / 2 = 1 :=
by
  intros people boxes h_people h_boxes
  sorry

end NUMINAMATH_GPT_man_can_lift_one_box_each_hand_l2276_227684


namespace NUMINAMATH_GPT_units_digit_x_pow_75_plus_6_eq_9_l2276_227660

theorem units_digit_x_pow_75_plus_6_eq_9 (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9)
  (h3 : (x ^ 75 + 6) % 10 = 9) : x = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_x_pow_75_plus_6_eq_9_l2276_227660


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2276_227666

-- First proof statement
theorem problem_1 : 2017^2 - 2016 * 2018 = 1 :=
by
  sorry

-- Definitions for the second problem
variables {a b : ℤ}

-- Second proof statement
theorem problem_2 (h1 : a + b = 7) (h2 : a * b = -1) : (a + b)^2 = 49 :=
by
  sorry

-- Third proof statement (part of the second problem)
theorem problem_3 (h1 : a + b = 7) (h2 : a * b = -1) : a^2 - 3 * a * b + b^2 = 54 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2276_227666


namespace NUMINAMATH_GPT_total_fish_correct_l2276_227628

-- Define the number of pufferfish
def num_pufferfish : ℕ := 15

-- Define the number of swordfish as 5 times the number of pufferfish
def num_swordfish : ℕ := 5 * num_pufferfish

-- Define the total number of fish as the sum of pufferfish and swordfish
def total_num_fish : ℕ := num_pufferfish + num_swordfish

-- Theorem stating the total number of fish
theorem total_fish_correct : total_num_fish = 90 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_fish_correct_l2276_227628


namespace NUMINAMATH_GPT_find_y_l2276_227636

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end NUMINAMATH_GPT_find_y_l2276_227636


namespace NUMINAMATH_GPT_original_price_correct_percentage_growth_rate_l2276_227653

-- Definitions and conditions
def original_price := 45
def sale_discount := 15
def price_after_discount := original_price - sale_discount

def initial_cost_before_event := 90
def final_cost_during_event := 120
def ratio_of_chickens := 2

def initial_buyers := 50
def increase_percentage := 20
def total_sales := 5460
def time_slots := 2  -- 1 hour = 2 slots of 30 minutes each

-- The problem: Prove the original price and growth rate
theorem original_price_correct (x : ℕ) : (120 / (x - 15) = 2 * (90 / x) → x = original_price) :=
by
  sorry

theorem percentage_growth_rate (m : ℕ) :
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = total_sales / (original_price - sale_discount) →
  m = increase_percentage) :=
by
  sorry

end NUMINAMATH_GPT_original_price_correct_percentage_growth_rate_l2276_227653


namespace NUMINAMATH_GPT_new_triangle_area_l2276_227638

theorem new_triangle_area (a b : ℝ) (x y : ℝ) (hypotenuse : x = a ∧ y = b ∧ x^2 + y^2 = (a + b)^2) : 
    (3  * (1 / 2) * a * b) = (3 / 2) * a * b :=
by
  sorry

end NUMINAMATH_GPT_new_triangle_area_l2276_227638


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l2276_227624

theorem right_triangle_hypotenuse 
  (shorter_leg longer_leg hypotenuse : ℝ)
  (h1 : longer_leg = 2 * shorter_leg - 1)
  (h2 : 1 / 2 * shorter_leg * longer_leg = 60) :
  hypotenuse = 17 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l2276_227624


namespace NUMINAMATH_GPT_area_of_shaded_region_l2276_227623

noncomputable def area_shaded (side : ℝ) : ℝ :=
  let area_square := side * side
  let radius := side / 2
  let area_circle := Real.pi * radius * radius
  area_square - area_circle

theorem area_of_shaded_region :
  let perimeter := 28
  let side := perimeter / 4
  area_shaded side = 49 - π * 12.25 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l2276_227623


namespace NUMINAMATH_GPT_initial_oranges_l2276_227694

variable (x : ℕ)
variable (total_oranges : ℕ := 8)
variable (oranges_from_joyce : ℕ := 3)

theorem initial_oranges (h : total_oranges = x + oranges_from_joyce) : x = 5 := by
  sorry

end NUMINAMATH_GPT_initial_oranges_l2276_227694


namespace NUMINAMATH_GPT_value_of_a2012_l2276_227675

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) - a n = 2 * n

theorem value_of_a2012 (a : ℕ → ℤ) (h : seq a) : a 2012 = 2012 * 2011 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_a2012_l2276_227675


namespace NUMINAMATH_GPT_polynomial_evaluation_l2276_227688

theorem polynomial_evaluation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l2276_227688


namespace NUMINAMATH_GPT_number_of_people_l2276_227696

def totalCups : ℕ := 10
def cupsPerPerson : ℕ := 2

theorem number_of_people {n : ℕ} (h : n = totalCups / cupsPerPerson) : n = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_people_l2276_227696


namespace NUMINAMATH_GPT_tax_percentage_l2276_227672

theorem tax_percentage (total_pay take_home_pay: ℕ) (h1 : total_pay = 650) (h2 : take_home_pay = 585) :
  ((total_pay - take_home_pay) * 100 / total_pay) = 10 :=
by
  -- Assumptions
  have hp1 : total_pay = 650 := h1
  have hp2 : take_home_pay = 585 := h2
  -- Calculate tax paid
  let tax_paid := total_pay - take_home_pay
  -- Calculate tax percentage
  let tax_percentage := (tax_paid * 100) / total_pay
  -- Prove the tax percentage is 10%
  sorry

end NUMINAMATH_GPT_tax_percentage_l2276_227672


namespace NUMINAMATH_GPT_distance_from_point_to_x_axis_l2276_227670

theorem distance_from_point_to_x_axis (x y : ℤ) (h : (x, y) = (5, -12)) : |y| = 12 :=
by
  -- sorry serves as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_distance_from_point_to_x_axis_l2276_227670


namespace NUMINAMATH_GPT_expected_turns_formula_l2276_227629

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1))))

theorem expected_turns_formula (n : ℕ) (h : n > 1) :
  expected_turns n = n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1)))) :=
by
  unfold expected_turns
  sorry

end NUMINAMATH_GPT_expected_turns_formula_l2276_227629


namespace NUMINAMATH_GPT_TruckloadsOfSand_l2276_227683

theorem TruckloadsOfSand (S : ℝ) (totalMat dirt cement : ℝ) 
  (h1 : totalMat = 0.67) 
  (h2 : dirt = 0.33) 
  (h3 : cement = 0.17) 
  (h4 : totalMat = S + dirt + cement) : 
  S = 0.17 := 
  by 
    sorry

end NUMINAMATH_GPT_TruckloadsOfSand_l2276_227683


namespace NUMINAMATH_GPT_unique_number_encoding_l2276_227681

-- Defining participants' score ranges 
def score_range := {x : ℕ // x ≤ 5}

-- Defining total score
def total_score (s1 s2 s3 s4 s5 s6 : score_range) : ℕ := 
  s1.val + s2.val + s3.val + s4.val + s5.val + s6.val

-- Main statement to encode participant's scores into a unique number
theorem unique_number_encoding (s1 s2 s3 s4 s5 s6 : score_range) :
  ∃ n : ℕ, ∃ s : ℕ, 
    s = total_score s1 s2 s3 s4 s5 s6 ∧ 
    n = s * 10^6 + s1.val * 10^5 + s2.val * 10^4 + s3.val * 10^3 + s4.val * 10^2 + s5.val * 10 + s6.val := 
sorry

end NUMINAMATH_GPT_unique_number_encoding_l2276_227681


namespace NUMINAMATH_GPT_employee_payment_proof_l2276_227643

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price as 20 percent above the wholesale cost
def retail_price (C_w : ℝ) : ℝ := C_w + 0.2 * C_w

-- Define the employee discount on the retail price
def employee_discount (C_r : ℝ) : ℝ := 0.15 * C_r

-- Define the amount paid by the employee
def amount_paid_by_employee (C_w : ℝ) : ℝ :=
  let C_r := retail_price C_w
  let D_e := employee_discount C_r
  C_r - D_e

-- Main theorem to prove the employee paid $204
theorem employee_payment_proof : amount_paid_by_employee wholesale_cost = 204 :=
by
  sorry

end NUMINAMATH_GPT_employee_payment_proof_l2276_227643


namespace NUMINAMATH_GPT_solution_set_l2276_227606

def op (a b : ℝ) : ℝ := -2 * a + b

theorem solution_set (x : ℝ) : (op x 4 > 0) ↔ (x < 2) :=
by {
  -- proof required here
  sorry
}

end NUMINAMATH_GPT_solution_set_l2276_227606


namespace NUMINAMATH_GPT_union_of_sets_l2276_227682

open Set

theorem union_of_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l2276_227682


namespace NUMINAMATH_GPT_contrapositive_even_sum_l2276_227610

theorem contrapositive_even_sum (a b : ℕ) :
  (¬(a % 2 = 0 ∧ b % 2 = 0) → ¬(a + b) % 2 = 0) ↔ (¬((a + b) % 2 = 0) → ¬(a % 2 = 0 ∧ b % 2 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_even_sum_l2276_227610


namespace NUMINAMATH_GPT_avg_age_of_new_persons_l2276_227607

-- We define the given conditions
def initial_persons : ℕ := 12
def initial_avg_age : ℝ := 16
def new_persons : ℕ := 12
def new_avg_age : ℝ := 15.5

-- Define the total initial age
def total_initial_age : ℝ := initial_persons * initial_avg_age

-- Define the total number of persons after new persons join
def total_persons_after_join : ℕ := initial_persons + new_persons

-- Define the total age after new persons join
def total_age_after_join : ℝ := total_persons_after_join * new_avg_age

-- We wish to prove that the average age of the new persons who joined is 15
theorem avg_age_of_new_persons : 
  (total_initial_age + new_persons * 15) = total_age_after_join :=
sorry

end NUMINAMATH_GPT_avg_age_of_new_persons_l2276_227607


namespace NUMINAMATH_GPT_sum_of_digits_1197_l2276_227634

theorem sum_of_digits_1197 : (1 + 1 + 9 + 7 = 18) := by sorry

end NUMINAMATH_GPT_sum_of_digits_1197_l2276_227634


namespace NUMINAMATH_GPT_correct_multiplication_result_l2276_227637

theorem correct_multiplication_result :
  0.08 * 3.25 = 0.26 :=
by
  -- This is to ensure that the theorem is well-formed and logically connected
  sorry

end NUMINAMATH_GPT_correct_multiplication_result_l2276_227637


namespace NUMINAMATH_GPT_range_of_a_l2276_227639

noncomputable def set_A : Set ℝ := { x | x^2 - 3 * x - 10 < 0 }
noncomputable def set_B : Set ℝ := { x | x^2 + 2 * x - 8 > 0 }
def set_C (a : ℝ) : Set ℝ := { x | 2 * a < x ∧ x < a + 3 }

theorem range_of_a (a : ℝ) :
  (A ∩ B) ∩ set_C a = set_C a → 1 ≤ a := 
sorry

end NUMINAMATH_GPT_range_of_a_l2276_227639


namespace NUMINAMATH_GPT_Nick_riding_speed_l2276_227689

theorem Nick_riding_speed (Alan_speed Maria_ratio Nick_ratio : ℝ) 
(h1 : Alan_speed = 6) (h2 : Maria_ratio = 3/4) (h3 : Nick_ratio = 4/3) : 
Nick_ratio * (Maria_ratio * Alan_speed) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_Nick_riding_speed_l2276_227689


namespace NUMINAMATH_GPT_compute_expression_l2276_227644

theorem compute_expression : 6^2 + 2 * 5 - 4^2 = 30 :=
by sorry

end NUMINAMATH_GPT_compute_expression_l2276_227644


namespace NUMINAMATH_GPT_exist_x_y_satisfy_condition_l2276_227616

theorem exist_x_y_satisfy_condition (f g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 0) (h2 : ∀ y, 0 ≤ y ∧ y ≤ 1 → g y ≥ 0) :
  ∃ (x : ℝ), ∃ (y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |f x + g y - x * y| ≥ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_exist_x_y_satisfy_condition_l2276_227616


namespace NUMINAMATH_GPT_parts_per_hour_l2276_227641

variables {x y : ℕ}

-- Condition 1: The time it takes for A to make 90 parts is the same as the time it takes for B to make 120 parts.
def time_ratio (x y : ℕ) := (x:ℚ) / y = 90 / 120

-- Condition 2: A and B together make 35 parts per hour.
def total_parts_per_hour (x y : ℕ) := x + y = 35

-- Given the conditions, prove the number of parts A and B each make per hour.
theorem parts_per_hour (x y : ℕ) (h1 : time_ratio x y) (h2 : total_parts_per_hour x y) : x = 15 ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_parts_per_hour_l2276_227641


namespace NUMINAMATH_GPT_sounds_meet_at_x_l2276_227652

theorem sounds_meet_at_x (d c s : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : 0 < s) :
  ∃ x : ℝ, x = d / 2 * (1 + s / c) ∧ x <= d ∧ x > 0 :=
by
  sorry

end NUMINAMATH_GPT_sounds_meet_at_x_l2276_227652


namespace NUMINAMATH_GPT_find_a_l2276_227608

noncomputable def polynomial1 (x : ℝ) : ℝ := x^3 + 3 * x^2 - x - 3
noncomputable def polynomial2 (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem find_a (a : ℝ) (x : ℝ) (hx1 : polynomial1 x > 0)
  (hx2 : polynomial2 x a ≤ 0) (ha : a > 0) : 
  3 / 4 ≤ a ∧ a < 4 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_l2276_227608


namespace NUMINAMATH_GPT_number_of_square_tiles_l2276_227604

theorem number_of_square_tiles (a b : ℕ) (h1 : a + b = 32) (h2 : 3 * a + 4 * b = 110) : b = 14 :=
by
  -- the proof steps are skipped
  sorry

end NUMINAMATH_GPT_number_of_square_tiles_l2276_227604


namespace NUMINAMATH_GPT_binary_111_is_7_l2276_227611

def binary_to_decimal (b0 b1 b2 : ℕ) : ℕ :=
  b0 * (2^0) + b1 * (2^1) + b2 * (2^2)

theorem binary_111_is_7 : binary_to_decimal 1 1 1 = 7 :=
by
  -- We will provide the proof here.
  sorry

end NUMINAMATH_GPT_binary_111_is_7_l2276_227611


namespace NUMINAMATH_GPT_compare_neg_fractions_l2276_227692

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end NUMINAMATH_GPT_compare_neg_fractions_l2276_227692


namespace NUMINAMATH_GPT_frank_candy_bags_l2276_227691

theorem frank_candy_bags (total_candies : ℕ) (candies_per_bag : ℕ) (bags : ℕ) 
  (h1 : total_candies = 22) (h2 : candies_per_bag = 11) : bags = 2 :=
by
  sorry

end NUMINAMATH_GPT_frank_candy_bags_l2276_227691


namespace NUMINAMATH_GPT_amount_subtracted_is_15_l2276_227633

theorem amount_subtracted_is_15 (n x : ℕ) (h1 : 7 * n - x = 2 * n + 10) (h2 : n = 5) : x = 15 :=
by 
  sorry

end NUMINAMATH_GPT_amount_subtracted_is_15_l2276_227633


namespace NUMINAMATH_GPT_find_m_l2276_227640

noncomputable def a_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1 : ℝ) * d)

theorem find_m (a d : ℝ) (m : ℕ) 
  (h1 : a_seq a d (m-1) + a_seq a d (m+1) - a = 0)
  (h2 : S_n a d (2*m - 1) = 38) : 
  m = 10 := 
sorry

end NUMINAMATH_GPT_find_m_l2276_227640


namespace NUMINAMATH_GPT_sin_theta_value_l2276_227690

theorem sin_theta_value (θ : ℝ) (h₁ : 8 * (Real.tan θ) = 3 * (Real.cos θ)) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_sin_theta_value_l2276_227690


namespace NUMINAMATH_GPT_numbers_in_ratio_l2276_227676

theorem numbers_in_ratio (a b c : ℤ) :
  (∃ x : ℤ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x) ∧ (a * a + b * b + c * c = 725) →
  (a = 10 ∧ b = 15 ∧ c = 20 ∨ a = -10 ∧ b = -15 ∧ c = -20) :=
by
  sorry

end NUMINAMATH_GPT_numbers_in_ratio_l2276_227676


namespace NUMINAMATH_GPT_tiffany_total_lives_l2276_227631

-- Define the conditions
def initial_lives : Float := 43.0
def hard_part_won : Float := 14.0
def next_level_won : Float := 27.0

-- State the theorem
theorem tiffany_total_lives : 
  initial_lives + hard_part_won + next_level_won = 84.0 :=
by 
  sorry

end NUMINAMATH_GPT_tiffany_total_lives_l2276_227631


namespace NUMINAMATH_GPT_train_crossing_time_l2276_227625

theorem train_crossing_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) :
  length_of_train = 1500 → speed_of_train = 95 → speed_of_man = 5 → 
  (length_of_train / ((speed_of_train - speed_of_man) * (1000 / 3600))) = 60 :=
by
  intros h1 h2 h3
  have h_rel_speed : ((speed_of_train - speed_of_man) * (1000 / 3600)) = 25 := by
    rw [h2, h3]
    norm_num
  rw [h1, h_rel_speed]
  norm_num

end NUMINAMATH_GPT_train_crossing_time_l2276_227625


namespace NUMINAMATH_GPT_brokerage_percentage_calculation_l2276_227669

theorem brokerage_percentage_calculation
  (face_value : ℝ)
  (discount_percentage : ℝ)
  (cost_price : ℝ)
  (h_face_value : face_value = 100)
  (h_discount_percentage : discount_percentage = 6)
  (h_cost_price : cost_price = 94.2) :
  ((cost_price - (face_value - (discount_percentage / 100 * face_value))) / cost_price * 100) = 0.2124 := 
by
  sorry

end NUMINAMATH_GPT_brokerage_percentage_calculation_l2276_227669


namespace NUMINAMATH_GPT_john_spent_on_sweets_l2276_227614

def initial_amount := 7.10
def amount_given_per_friend := 1.00
def amount_left := 4.05
def amount_spent_on_friends := 2 * amount_given_per_friend
def amount_remaining_after_friends := initial_amount - amount_spent_on_friends
def amount_spent_on_sweets := amount_remaining_after_friends - amount_left

theorem john_spent_on_sweets : amount_spent_on_sweets = 1.05 := 
by
  sorry

end NUMINAMATH_GPT_john_spent_on_sweets_l2276_227614


namespace NUMINAMATH_GPT_derivative_of_f_tangent_line_at_pi_l2276_227612

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : deriv f x = (x * Real.cos x - Real.sin x) / (x ^ 2) :=
  sorry

theorem tangent_line_at_pi : 
  let M := (Real.pi, 0)
  let slope := -1 / Real.pi
  let tangent_line (x : ℝ) : ℝ := -x / Real.pi + 1
  ∀ (x y : ℝ), (x, y) = M → y = tangent_line x :=
  sorry

end NUMINAMATH_GPT_derivative_of_f_tangent_line_at_pi_l2276_227612


namespace NUMINAMATH_GPT_new_sum_after_decrease_l2276_227674

theorem new_sum_after_decrease (a b : ℕ) (h₁ : a + b = 100) (h₂ : a' = a - 48) : a' + b = 52 := by
  sorry

end NUMINAMATH_GPT_new_sum_after_decrease_l2276_227674


namespace NUMINAMATH_GPT_solve_for_x_l2276_227698

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 14.7 -> x = 105 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2276_227698


namespace NUMINAMATH_GPT_find_a_m_18_l2276_227619

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (a1 : ℝ)
variable (m : ℕ)

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) :=
  ∀ n : ℕ, a n = a1 * r^n

def problem_conditions (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :=
  (geometric_sequence a a1 r) ∧
  a m = 3 ∧
  a (m + 6) = 24

theorem find_a_m_18 (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :
  problem_conditions a r a1 m → a (m + 18) = 1536 :=
by
  sorry

end NUMINAMATH_GPT_find_a_m_18_l2276_227619


namespace NUMINAMATH_GPT_midpoint_range_l2276_227605

variable {x0 y0 : ℝ}

-- Conditions
def point_on_line1 (P : ℝ × ℝ) := P.1 + 2 * P.2 - 1 = 0
def point_on_line2 (Q : ℝ × ℝ) := Q.1 + 2 * Q.2 + 3 = 0
def is_midpoint (P Q M : ℝ × ℝ) := P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2
def midpoint_condition (M : ℝ × ℝ) := M.2 > M.1 + 2

-- Theorem
theorem midpoint_range
  (P Q M : ℝ × ℝ)
  (hP : point_on_line1 P)
  (hQ : point_on_line2 Q)
  (hM : is_midpoint P Q M)
  (h_cond : midpoint_condition M)
  (hx0 : x0 = M.1)
  (hy0 : y0 = M.2)
  : - (1 / 2) < y0 / x0 ∧ y0 / x0 < - (1 / 5) :=
sorry

end NUMINAMATH_GPT_midpoint_range_l2276_227605


namespace NUMINAMATH_GPT_no_conditions_satisfy_l2276_227601

-- Define the conditions
def condition1 (a b c : ℤ) : Prop := a = 1 ∧ b = 1 ∧ c = 1
def condition2 (a b c : ℤ) : Prop := a = b - 1 ∧ b = c - 1
def condition3 (a b c : ℤ) : Prop := a = b ∧ b = c
def condition4 (a b c : ℤ) : Prop := a > c ∧ c = b - 1 

-- Define the equations
def equation1 (a b c : ℤ) : ℤ := a * (a - b)^3 + b * (b - c)^3 + c * (c - a)^3
def equation2 (a b c : ℤ) : Prop := a + b + c = 3

-- Proof statement for the original problem
theorem no_conditions_satisfy (a b c : ℤ) :
  ¬ (condition1 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition2 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition3 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition4 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) :=
sorry

end NUMINAMATH_GPT_no_conditions_satisfy_l2276_227601


namespace NUMINAMATH_GPT_telepathic_connection_correct_l2276_227685

def telepathic_connection_probability : ℚ := sorry

theorem telepathic_connection_correct :
  telepathic_connection_probability = 7 / 25 := sorry

end NUMINAMATH_GPT_telepathic_connection_correct_l2276_227685


namespace NUMINAMATH_GPT_grapes_total_sum_l2276_227686

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end NUMINAMATH_GPT_grapes_total_sum_l2276_227686


namespace NUMINAMATH_GPT_determine_y_l2276_227642

theorem determine_y (y : ℝ) (h1 : 0 < y) (h2 : y * (⌊y⌋ : ℝ) = 90) : y = 10 :=
sorry

end NUMINAMATH_GPT_determine_y_l2276_227642
