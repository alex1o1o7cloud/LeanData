import Mathlib

namespace NUMINAMATH_GPT_find_ordered_pair_l1110_111096

theorem find_ordered_pair :
  ∃ (x y : ℚ), 7 * x - 3 * y = 6 ∧ 4 * x + 5 * y = 23 ∧ 
               x = 99 / 47 ∧ y = 137 / 47 :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l1110_111096


namespace NUMINAMATH_GPT_total_value_of_goods_l1110_111072

theorem total_value_of_goods (V : ℝ) (tax_paid : ℝ) (tax_exemption : ℝ) (tax_rate : ℝ) :
  tax_exemption = 600 → tax_rate = 0.11 → tax_paid = 123.2 → 0.11 * (V - 600) = tax_paid → V = 1720 :=
by
  sorry

end NUMINAMATH_GPT_total_value_of_goods_l1110_111072


namespace NUMINAMATH_GPT_claudia_fills_4ounce_glasses_l1110_111077

theorem claudia_fills_4ounce_glasses :
  ∀ (total_water : ℕ) (five_ounce_glasses : ℕ) (eight_ounce_glasses : ℕ) 
    (four_ounce_glass_volume : ℕ),
  total_water = 122 →
  five_ounce_glasses = 6 →
  eight_ounce_glasses = 4 →
  four_ounce_glass_volume = 4 →
  (total_water - (five_ounce_glasses * 5 + eight_ounce_glasses * 8)) / four_ounce_glass_volume = 15 :=
by
  intros _ _ _ _ _ _ _ _ 
  sorry

end NUMINAMATH_GPT_claudia_fills_4ounce_glasses_l1110_111077


namespace NUMINAMATH_GPT_cube_volume_l1110_111033

theorem cube_volume (d : ℝ) (h : d = 6 * Real.sqrt 2) : 
  ∃ v : ℝ, v = 48 * Real.sqrt 6 := by
  let s := d / Real.sqrt 3
  let volume := s ^ 3
  use volume
  /- Proof of the volume calculation is omitted. -/
  sorry

end NUMINAMATH_GPT_cube_volume_l1110_111033


namespace NUMINAMATH_GPT_james_sells_boxes_l1110_111050

theorem james_sells_boxes (profit_per_candy_bar : ℝ) (total_profit : ℝ) 
                          (candy_bars_per_box : ℕ) (x : ℕ)
                          (h1 : profit_per_candy_bar = 1.5 - 1)
                          (h2 : total_profit = 25)
                          (h3 : candy_bars_per_box = 10) 
                          (h4 : total_profit = (x * candy_bars_per_box) * profit_per_candy_bar) :
                          x = 5 :=
by
  sorry

end NUMINAMATH_GPT_james_sells_boxes_l1110_111050


namespace NUMINAMATH_GPT_ball_height_intersect_l1110_111001

noncomputable def ball_height (h : ℝ) (t₁ t₂ : ℝ) (h₁ h₂ : ℝ → ℝ) : Prop :=
  (∀ t, h₁ t = h₂ (t - 1) ↔ t = t₁) ∧
  (h₁ t₁ = h ∧ h₂ t₁ = h) ∧ 
  (∀ t, h₂ (t - 1) = h₁ t) ∧ 
  (h₁ (1.1) = h ∧ h₂ (1.1) = h)

theorem ball_height_intersect (h : ℝ)
  (h₁ h₂ : ℝ → ℝ)
  (h_max : ∀ t₁ t₂, ball_height h t₁ t₂ h₁ h₂) :
  (∃ t₁, t₁ = 1.6) :=
sorry

end NUMINAMATH_GPT_ball_height_intersect_l1110_111001


namespace NUMINAMATH_GPT_min_distance_between_curves_l1110_111024

noncomputable def distance_between_intersections : ℝ :=
  let f (x : ℝ) := (2 * x + 1) - (x + Real.log x)
  let f' (x : ℝ) := 1 - 1 / x
  let minimum_distance :=
    if hs : 1 < 1 then 2 else
    if hs : 1 > 1 then 2 else
    2
  minimum_distance

theorem min_distance_between_curves : distance_between_intersections = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_between_curves_l1110_111024


namespace NUMINAMATH_GPT_melanie_batches_l1110_111047

theorem melanie_batches (total_brownies_given: ℕ)
                        (brownies_per_batch: ℕ)
                        (fraction_bake_sale: ℚ)
                        (fraction_container: ℚ)
                        (remaining_brownies_given: ℕ) :
                        brownies_per_batch = 20 →
                        fraction_bake_sale = 3/4 →
                        fraction_container = 3/5 →
                        total_brownies_given = 20 →
                        (remaining_brownies_given / (brownies_per_batch * (1 - fraction_bake_sale) * (1 - fraction_container))) = 10 :=
by
  sorry

end NUMINAMATH_GPT_melanie_batches_l1110_111047


namespace NUMINAMATH_GPT_cos_300_eq_half_l1110_111095

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_cos_300_eq_half_l1110_111095


namespace NUMINAMATH_GPT_radiator_water_fraction_l1110_111048

theorem radiator_water_fraction :
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  fraction_remaining_per_replacement^4 = 81 / 256 := by
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  sorry

end NUMINAMATH_GPT_radiator_water_fraction_l1110_111048


namespace NUMINAMATH_GPT_total_ticket_cost_is_correct_l1110_111005

-- Definitions based on the conditions provided
def child_ticket_cost : ℝ := 4.25
def adult_ticket_cost : ℝ := child_ticket_cost + 3.50
def senior_ticket_cost : ℝ := adult_ticket_cost - 1.75

def number_adult_tickets : ℕ := 2
def number_child_tickets : ℕ := 4
def number_senior_tickets : ℕ := 1

def total_ticket_cost_before_discount : ℝ := 
  number_adult_tickets * adult_ticket_cost + 
  number_child_tickets * child_ticket_cost + 
  number_senior_tickets * senior_ticket_cost

def total_tickets : ℕ := number_adult_tickets + number_child_tickets + number_senior_tickets
def discount : ℝ := if total_tickets >= 5 then 3.0 else 0.0

def total_ticket_cost_after_discount : ℝ := total_ticket_cost_before_discount - discount

-- The proof statement: proving the total ticket cost after the discount is $35.50
theorem total_ticket_cost_is_correct : total_ticket_cost_after_discount = 35.50 := by
  -- Note: The exact solution is omitted and replaced with sorry to denote where the proof would be.
  sorry

end NUMINAMATH_GPT_total_ticket_cost_is_correct_l1110_111005


namespace NUMINAMATH_GPT_expand_product_l1110_111093

theorem expand_product (x : ℝ) : (x + 2) * (x^2 + 3 * x + 4) = x^3 + 5 * x^2 + 10 * x + 8 := 
by
  sorry

end NUMINAMATH_GPT_expand_product_l1110_111093


namespace NUMINAMATH_GPT_speed_conversion_l1110_111099

noncomputable def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_conversion (h : 1 = 3.6) : mps_to_kmph 12.7788 = 45.96 :=
  by
    sorry

end NUMINAMATH_GPT_speed_conversion_l1110_111099


namespace NUMINAMATH_GPT_find_roots_of_g_l1110_111060

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - a*x - b
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := b*x^2 - a*x - 1

theorem find_roots_of_g :
  (∀ a b : ℝ, f 2 a b = 0 ∧ f 3 a b = 0 → ∃ (x1 x2 : ℝ), g x1 a b = 0 ∧ g x2 a b = 0 ∧
    (x1 = -1/2 ∨ x1 = -1/3) ∧ (x2 = -1/2 ∨ x2 = -1/3) ∧ x1 ≠ x2) :=
by
  sorry

end NUMINAMATH_GPT_find_roots_of_g_l1110_111060


namespace NUMINAMATH_GPT_arithmetic_and_geometric_mean_l1110_111051

theorem arithmetic_and_geometric_mean (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 100) : a^2 + b^2 = 1400 := by
  sorry

end NUMINAMATH_GPT_arithmetic_and_geometric_mean_l1110_111051


namespace NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1110_111041

variable (n : ℕ)
variable (h1 : 100 ≤ n ∧ n < 1000)
variable (h2 : (n + 7) % 6 = 0)
variable (h3 : (n - 5) % 9 = 0)

theorem smallest_n_satisfying_conditions : n = 113 := by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1110_111041


namespace NUMINAMATH_GPT_find_multiple_of_q_l1110_111000

theorem find_multiple_of_q
  (q : ℕ)
  (x : ℕ := 55 + 2 * q)
  (y : ℕ)
  (m : ℕ)
  (h1 : y = m * q + 41)
  (h2 : x = y)
  (h3 : q = 7) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_q_l1110_111000


namespace NUMINAMATH_GPT_price_per_kilo_of_bananas_l1110_111085

def initial_money : ℕ := 500
def potatoes_cost : ℕ := 6 * 2
def tomatoes_cost : ℕ := 9 * 3
def cucumbers_cost : ℕ := 5 * 4
def bananas_weight : ℕ := 3
def remaining_money : ℕ := 426

-- Defining total cost of all items
def total_item_cost : ℕ := initial_money - remaining_money

-- Defining the total cost of bananas
def cost_bananas : ℕ := total_item_cost - (potatoes_cost + tomatoes_cost + cucumbers_cost)

-- Final question: Prove that the price per kilo of bananas is $5
theorem price_per_kilo_of_bananas : cost_bananas / bananas_weight = 5 :=
by
  sorry

end NUMINAMATH_GPT_price_per_kilo_of_bananas_l1110_111085


namespace NUMINAMATH_GPT_gcd_840_1764_l1110_111058

def a : ℕ := 840
def b : ℕ := 1764

theorem gcd_840_1764 : Nat.gcd a b = 84 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1110_111058


namespace NUMINAMATH_GPT_stream_speed_l1110_111090

theorem stream_speed (v : ℝ) : (24 + v) = 168 / 6 → v = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_stream_speed_l1110_111090


namespace NUMINAMATH_GPT_find_N_value_l1110_111021

variable (a b N : ℚ)
variable (h1 : a + 2 * b = N)
variable (h2 : a * b = 4)
variable (h3 : 2 / a + 1 / b = 1.5)

theorem find_N_value : N = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_N_value_l1110_111021


namespace NUMINAMATH_GPT_relationship_of_ys_l1110_111070

variable (f : ℝ → ℝ)

def inverse_proportion := ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x

theorem relationship_of_ys
  (h_inv_prop : inverse_proportion f)
  (h_pts1 : f (-2) = 3)
  (h_pts2 : f (-3) = y₁)
  (h_pts3 : f 1 = y₂)
  (h_pts4 : f 2 = y₃) :
  y₂ < y₃ ∧ y₃ < y₁ :=
sorry

end NUMINAMATH_GPT_relationship_of_ys_l1110_111070


namespace NUMINAMATH_GPT_globe_division_l1110_111079

theorem globe_division (parallels meridians : ℕ)
  (h_parallels : parallels = 17)
  (h_meridians : meridians = 24) :
  let slices_per_sector := parallels + 1
  let sectors := meridians
  let total_parts := slices_per_sector * sectors
  total_parts = 432 := by
  sorry

end NUMINAMATH_GPT_globe_division_l1110_111079


namespace NUMINAMATH_GPT_shaded_area_l1110_111026

-- Defining the conditions
def total_area_of_grid : ℕ := 38
def base_of_triangle : ℕ := 12
def height_of_triangle : ℕ := 4

-- Using the formula for the area of a right triangle
def area_of_unshaded_triangle : ℕ := (base_of_triangle * height_of_triangle) / 2

-- The goal: Prove the area of the shaded region
theorem shaded_area : total_area_of_grid - area_of_unshaded_triangle = 14 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l1110_111026


namespace NUMINAMATH_GPT_smallest_five_consecutive_even_sum_320_l1110_111008

theorem smallest_five_consecutive_even_sum_320 : ∃ (a b c d e : ℤ), a + b + c + d + e = 320 ∧ (∀ i j : ℤ, (i = a ∨ i = b ∨ i = c ∨ i = d ∨ i = e) → (j = a ∨ j = b ∨ j = c ∨ j = d ∨ j = e) → (i = j + 2 ∨ i = j - 2 ∨ i = j)) ∧ (a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∧ a = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_consecutive_even_sum_320_l1110_111008


namespace NUMINAMATH_GPT_total_pies_eaten_l1110_111062

variable (Adam Bill Sierra : ℕ)

axiom condition1 : Adam = Bill + 3
axiom condition2 : Sierra = 2 * Bill
axiom condition3 : Sierra = 12

theorem total_pies_eaten : Adam + Bill + Sierra = 27 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_total_pies_eaten_l1110_111062


namespace NUMINAMATH_GPT_problem1_problem2_l1110_111002

-- Definitions of the conditions
def periodic_func (f: ℝ → ℝ) (a: ℝ) (x: ℝ) : Prop :=
(∀ x, f (x + 3) = f x) ∧ 
(∀ x, -2 ≤ x ∧ x < 0 → f x = x + a) ∧ 
(∀ x, 0 ≤ x ∧ x < 1 → f x = (1/2)^x)

-- 1. Prove f(13/2) = sqrt(2)/2
theorem problem1 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) : f (13/2) = (Real.sqrt 2) / 2 := 
sorry

-- 2. Prove that if f(x) has a minimum value but no maximum value, then 1 < a ≤ 5/2
theorem problem2 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) (hmin: ∃ m, ∀ x, f x ≥ m) (hmax: ¬∃ M, ∀ x, f x ≤ M) : 1 < a ∧ a ≤ 5/2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1110_111002


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l1110_111075

theorem quadratic_distinct_roots (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l1110_111075


namespace NUMINAMATH_GPT_multiply_24_99_l1110_111022

theorem multiply_24_99 : 24 * 99 = 2376 :=
by
  sorry

end NUMINAMATH_GPT_multiply_24_99_l1110_111022


namespace NUMINAMATH_GPT_more_trees_in_ahmeds_orchard_l1110_111061

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end NUMINAMATH_GPT_more_trees_in_ahmeds_orchard_l1110_111061


namespace NUMINAMATH_GPT_express_308_million_in_scientific_notation_l1110_111067

theorem express_308_million_in_scientific_notation :
    (308000000 : ℝ) = 3.08 * (10 ^ 8) :=
by
  sorry

end NUMINAMATH_GPT_express_308_million_in_scientific_notation_l1110_111067


namespace NUMINAMATH_GPT_sqrt_sum_ineq_l1110_111045

open Real

theorem sqrt_sum_ineq (a b c d : ℝ) (h : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0)
  (h4 : a + b + c + d = 4) : 
  sqrt (a + b + c) + sqrt (b + c + d) + sqrt (c + d + a) + sqrt (d + a + b) ≥ 6 :=
sorry

end NUMINAMATH_GPT_sqrt_sum_ineq_l1110_111045


namespace NUMINAMATH_GPT_total_savings_in_2_months_l1110_111063

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_savings_in_2_months_l1110_111063


namespace NUMINAMATH_GPT_Jirina_number_l1110_111087

theorem Jirina_number (a b c d : ℕ) (h_abcd : 1000 * a + 100 * b + 10 * c + d = 1468) :
  (1000 * a + 100 * b + 10 * c + d) +
  (1000 * a + 100 * d + 10 * c + b) = 3332 ∧ 
  (1000 * a + 100 * b + 10 * c + d)+
  (1000 * c + 100 * b + 10 * a + d) = 7886 :=
by
  sorry

end NUMINAMATH_GPT_Jirina_number_l1110_111087


namespace NUMINAMATH_GPT_abe_age_sum_is_31_l1110_111040

-- Define the present age of Abe
def abe_present_age : ℕ := 19

-- Define Abe's age 7 years ago
def abe_age_7_years_ago : ℕ := abe_present_age - 7

-- Define the sum of Abe's present age and his age 7 years ago
def abe_age_sum : ℕ := abe_present_age + abe_age_7_years_ago

-- Prove that the sum is 31
theorem abe_age_sum_is_31 : abe_age_sum = 31 := 
by 
  sorry

end NUMINAMATH_GPT_abe_age_sum_is_31_l1110_111040


namespace NUMINAMATH_GPT_new_planet_volume_eq_l1110_111088

noncomputable def volume_of_new_planet (V_earth : ℝ) (scaling_factor : ℝ) : ℝ :=
  V_earth * (scaling_factor^3)

theorem new_planet_volume_eq 
  (V_earth : ℝ)
  (scaling_factor : ℝ)
  (hV_earth : V_earth = 1.08 * 10^12)
  (h_scaling_factor : scaling_factor = 10^4) :
  volume_of_new_planet V_earth scaling_factor = 1.08 * 10^24 :=
by
  sorry

end NUMINAMATH_GPT_new_planet_volume_eq_l1110_111088


namespace NUMINAMATH_GPT_vanessa_recycled_correct_l1110_111065

-- Define conditions as separate hypotheses
variable (weight_per_point : ℕ := 9)
variable (points_earned : ℕ := 4)
variable (friends_recycled : ℕ := 16)

-- Define the total weight recycled as points earned times the weight per point
def total_weight_recycled (points_earned weight_per_point : ℕ) : ℕ := points_earned * weight_per_point

-- Define the weight recycled by Vanessa
def vanessa_recycled (total_recycled friends_recycled : ℕ) : ℕ := total_recycled - friends_recycled

-- Main theorem statement
theorem vanessa_recycled_correct (weight_per_point points_earned friends_recycled : ℕ) 
    (hw : weight_per_point = 9) (hp : points_earned = 4) (hf : friends_recycled = 16) : 
    vanessa_recycled (total_weight_recycled points_earned weight_per_point) friends_recycled = 20 := 
by 
  sorry

end NUMINAMATH_GPT_vanessa_recycled_correct_l1110_111065


namespace NUMINAMATH_GPT_ball_cost_l1110_111017

theorem ball_cost (C x y : ℝ)
  (H1 :  x = 1/3 * (C/2 + y + 5) )
  (H2 :  y = 1/4 * (C/2 + x + 5) )
  (H3 :  C/2 + x + y + 5 = C ) : C = 20 := 
by
  sorry

end NUMINAMATH_GPT_ball_cost_l1110_111017


namespace NUMINAMATH_GPT_final_salt_concentration_is_25_l1110_111053

-- Define the initial conditions
def original_solution_weight : ℝ := 100
def original_salt_concentration : ℝ := 0.10
def added_salt_weight : ℝ := 20

-- Define the amount of salt in the original solution
def original_salt_weight := original_solution_weight * original_salt_concentration

-- Define the total amount of salt after adding pure salt
def total_salt_weight := original_salt_weight + added_salt_weight

-- Define the total weight of the new solution
def new_solution_weight := original_solution_weight + added_salt_weight

-- Define the final salt concentration
noncomputable def final_salt_concentration := (total_salt_weight / new_solution_weight) * 100

-- Prove the final salt concentration equals 25%
theorem final_salt_concentration_is_25 : final_salt_concentration = 25 :=
by
  sorry

end NUMINAMATH_GPT_final_salt_concentration_is_25_l1110_111053


namespace NUMINAMATH_GPT_original_bill_l1110_111069

theorem original_bill (n : ℕ) (d : ℝ) (p : ℝ) (B : ℝ) (h1 : n = 5) (h2 : d = 0.06) (h3 : p = 18.8)
  (h4 : 0.94 * B = n * p) :
  B = 100 :=
sorry

end NUMINAMATH_GPT_original_bill_l1110_111069


namespace NUMINAMATH_GPT_right_triangle_exists_l1110_111049

theorem right_triangle_exists (a b c d : ℕ) (h1 : ab = cd) (h2 : a + b = c - d) : 
  ∃ (x y z : ℕ), x * y / 2 = ab ∧ x^2 + y^2 = z^2 :=
sorry

end NUMINAMATH_GPT_right_triangle_exists_l1110_111049


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1110_111086

def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def notU (s : Set ℝ) : Set ℝ := { x | x ∉ s ∧ x ∈ U }

theorem problem_1 : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
sorry

theorem problem_2 : notU A ∪ B = { x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7) } :=
sorry

theorem problem_3 : A ∩ notU B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1110_111086


namespace NUMINAMATH_GPT_ratio_a3_a2_l1110_111036

theorem ratio_a3_a2 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_3 / a_2 = -2 :=
sorry

end NUMINAMATH_GPT_ratio_a3_a2_l1110_111036


namespace NUMINAMATH_GPT_vector_orthogonality_l1110_111074

variables (x : ℝ)

def vec_a := (x - 1, 2)
def vec_b := (1, x)

theorem vector_orthogonality :
  (vec_a x).fst * (vec_b x).fst + (vec_a x).snd * (vec_b x).snd = 0 ↔ x = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_vector_orthogonality_l1110_111074


namespace NUMINAMATH_GPT_base_b_sum_correct_l1110_111011

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end NUMINAMATH_GPT_base_b_sum_correct_l1110_111011


namespace NUMINAMATH_GPT_rank_from_last_l1110_111035

theorem rank_from_last (total_students : ℕ) (rank_from_top : ℕ) (rank_from_last : ℕ) : 
  total_students = 35 → 
  rank_from_top = 14 → 
  rank_from_last = (total_students - rank_from_top + 1) → 
  rank_from_last = 22 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_rank_from_last_l1110_111035


namespace NUMINAMATH_GPT_larger_number_of_ratio_and_lcm_l1110_111073

theorem larger_number_of_ratio_and_lcm (x : ℕ) (h1 : (2 * x) % (5 * x) = 160) : (5 * x) = 160 := by
  sorry

end NUMINAMATH_GPT_larger_number_of_ratio_and_lcm_l1110_111073


namespace NUMINAMATH_GPT_sequence_a4_value_l1110_111089

theorem sequence_a4_value :
  ∀ {a : ℕ → ℚ}, (a 1 = 3) → ((∀ n, a (n + 1) = 3 * a n / (a n + 3))) → (a 4 = 3 / 4) :=
by
  intros a h1 hRec
  sorry

end NUMINAMATH_GPT_sequence_a4_value_l1110_111089


namespace NUMINAMATH_GPT_new_average_increased_by_40_percent_l1110_111068

theorem new_average_increased_by_40_percent 
  (n : ℕ) (initial_avg : ℝ) (initial_marks : ℝ) (new_marks : ℝ) (new_avg : ℝ)
  (h1 : n = 37)
  (h2 : initial_avg = 73)
  (h3 : initial_marks = (initial_avg * n))
  (h4 : new_marks = (initial_marks * 1.40))
  (h5 : new_avg = (new_marks / n)) :
  new_avg = 102.2 :=
sorry

end NUMINAMATH_GPT_new_average_increased_by_40_percent_l1110_111068


namespace NUMINAMATH_GPT_total_steps_eliana_walked_l1110_111081

-- Define the conditions of the problem.
def first_day_exercise_steps : Nat := 200
def first_day_additional_steps : Nat := 300
def second_day_multiplier : Nat := 2
def third_day_additional_steps : Nat := 100

-- Define the steps calculation for each day.
def first_day_total_steps : Nat := first_day_exercise_steps + first_day_additional_steps
def second_day_total_steps : Nat := second_day_multiplier * first_day_total_steps
def third_day_total_steps : Nat := second_day_total_steps + third_day_additional_steps

-- Prove that the total number of steps Eliana walked during these three days is 1600.
theorem total_steps_eliana_walked :
  first_day_total_steps + second_day_total_steps + third_day_additional_steps = 1600 :=
by
  -- Conditional values are constants. We can use Lean's deterministic evaluator here.
  -- Hence, there's no need to write out full proof for now. Using sorry to bypass actual proof.
  sorry

end NUMINAMATH_GPT_total_steps_eliana_walked_l1110_111081


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1110_111010

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1110_111010


namespace NUMINAMATH_GPT_triangle_area_is_12_5_l1110_111004

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨5, 0⟩
def N : Point := ⟨0, 5⟩
noncomputable def P (x y : ℝ) (h : x + y = 8) : Point := ⟨x, y⟩

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem triangle_area_is_12_5 (x y : ℝ) (h : x + y = 8) :
  area_triangle M N (P x y h) = 12.5 :=
sorry

end NUMINAMATH_GPT_triangle_area_is_12_5_l1110_111004


namespace NUMINAMATH_GPT_find_principal_amount_l1110_111034

theorem find_principal_amount (P r : ℝ) (h1 : 720 = P * (1 + 2 * r)) (h2 : 1020 = P * (1 + 7 * r)) : P = 600 :=
by sorry

end NUMINAMATH_GPT_find_principal_amount_l1110_111034


namespace NUMINAMATH_GPT_P_necessary_but_not_sufficient_for_q_l1110_111076

def M : Set ℝ := {x : ℝ | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x : ℝ | x^2 + x < 0}

theorem P_necessary_but_not_sufficient_for_q :
  (∀ x, x ∈ N → x ∈ M) ∧ (∃ x, x ∈ M ∧ x ∉ N) :=
by
  sorry

end NUMINAMATH_GPT_P_necessary_but_not_sufficient_for_q_l1110_111076


namespace NUMINAMATH_GPT_amber_worked_hours_l1110_111098

-- Define the variables and conditions
variables (A : ℝ) (Armand_hours : ℝ) (Ella_hours : ℝ)
variables (h1 : Armand_hours = A / 3) (h2 : Ella_hours = 2 * A)
variables (h3 : A + Armand_hours + Ella_hours = 40)

-- Prove the statement
theorem amber_worked_hours : A = 12 :=
by
  sorry

end NUMINAMATH_GPT_amber_worked_hours_l1110_111098


namespace NUMINAMATH_GPT_sum_to_12_of_7_chosen_l1110_111054

theorem sum_to_12_of_7_chosen (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (T : Finset ℕ) (hT1 : T ⊆ S) (hT2 : T.card = 7) :
  ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_to_12_of_7_chosen_l1110_111054


namespace NUMINAMATH_GPT_div_relation_l1110_111019

variables (a b c : ℚ)

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 1 / 2) : c / a = 2 / 3 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_div_relation_l1110_111019


namespace NUMINAMATH_GPT_inequality_for_natural_n_l1110_111043

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_natural_n_l1110_111043


namespace NUMINAMATH_GPT_correct_proposition_l1110_111029

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Conditions
axiom perpendicular (m : Line) (α : Plane) : Prop
axiom parallel (n : Line) (α : Plane) : Prop

-- Specific conditions given
axiom m_perp_α : perpendicular m α
axiom n_par_α : parallel n α

-- Statement to prove
theorem correct_proposition : perpendicular m n := sorry

end NUMINAMATH_GPT_correct_proposition_l1110_111029


namespace NUMINAMATH_GPT_weight_of_grapes_l1110_111052

theorem weight_of_grapes :
  ∀ (weight_of_fruits weight_of_apples weight_of_oranges weight_of_strawberries weight_of_grapes : ℕ),
  weight_of_fruits = 10 →
  weight_of_apples = 3 →
  weight_of_oranges = 1 →
  weight_of_strawberries = 3 →
  weight_of_fruits = weight_of_apples + weight_of_oranges + weight_of_strawberries + weight_of_grapes →
  weight_of_grapes = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_weight_of_grapes_l1110_111052


namespace NUMINAMATH_GPT_correct_percentage_is_500_over_7_l1110_111025

-- Given conditions
variable (x : ℕ)
def total_questions : ℕ := 7 * x
def missed_questions : ℕ := 2 * x

-- Definition of the fraction and percentage calculation
def correct_fraction : ℚ := (total_questions x - missed_questions x : ℕ) / total_questions x
def correct_percentage : ℚ := correct_fraction x * 100

-- The theorem to prove
theorem correct_percentage_is_500_over_7 : correct_percentage x = 500 / 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_correct_percentage_is_500_over_7_l1110_111025


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1110_111037

variables (a b c d x : ℕ)

theorem value_of_a_plus_b : (b + c = 9) → (c + d = 3) → (a + d = 8) → (a + b = x) → x = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1110_111037


namespace NUMINAMATH_GPT_quadratic_always_positive_l1110_111097

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - k + 4 > 0) ↔ -2 * Real.sqrt 3 < k ∧ k < 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_quadratic_always_positive_l1110_111097


namespace NUMINAMATH_GPT_deliveries_conditions_l1110_111066

variables (M P D : ℕ)
variables (MeMa MeBr MeQu MeBx: ℕ)

def distribution := (MeMa = 3 * MeBr) ∧ (MeBr = MeBr) ∧ (MeQu = MeBr) ∧ (MeBx = MeBr)

theorem deliveries_conditions 
  (h1 : P = 8 * M) 
  (h2 : D = 4 * M) 
  (h3 : M + P + D = 75) 
  (h4 : MeMa + MeBr + MeQu + MeBx = M)
  (h5 : distribution MeMa MeBr MeQu MeBx) :
  M = 5 ∧ MeMa = 2 ∧ MeBr = 1 ∧ MeQu = 1 ∧ MeBx = 1 :=
    sorry 

end NUMINAMATH_GPT_deliveries_conditions_l1110_111066


namespace NUMINAMATH_GPT_root_in_interval_iff_a_outside_range_l1110_111016

theorem root_in_interval_iff_a_outside_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ a * x + 1 = 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_iff_a_outside_range_l1110_111016


namespace NUMINAMATH_GPT_problem1_problem2_l1110_111092

-- Problem1
theorem problem1 (a : ℤ) (h : a = -2) :
    ( (a^2 + a) / (a^2 - 3 * a) / (a^2 - 1) / (a - 3) - 1 / (a + 1) = 2 / 3) :=
by 
  sorry

-- Problem2
theorem problem2 (x : ℤ) :
    ( (x^2 - 1) / (x - 4) / (x + 1) / (4 - x) = 1 - x) :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1110_111092


namespace NUMINAMATH_GPT_f_2_value_l1110_111023

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_2_value :
  (f a b (-2)) = 2 → (f a b 2) = -10 :=
by
  intro h
  -- Provide the solution steps here, starting with simplifying the equation. Sorry for now
  sorry

end NUMINAMATH_GPT_f_2_value_l1110_111023


namespace NUMINAMATH_GPT_additional_coins_needed_l1110_111031

def num_friends : Nat := 15
def current_coins : Nat := 105

def total_coins_needed (n : Nat) : Nat :=
  n * (n + 1) / 2
  
theorem additional_coins_needed :
  let coins_needed := total_coins_needed num_friends
  let additional_coins := coins_needed - current_coins
  additional_coins = 15 :=
by
  sorry

end NUMINAMATH_GPT_additional_coins_needed_l1110_111031


namespace NUMINAMATH_GPT_area_of_curves_l1110_111007

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.sqrt x - x^2)

theorem area_of_curves :
  enclosed_area = 1 / 3 :=
sorry

end NUMINAMATH_GPT_area_of_curves_l1110_111007


namespace NUMINAMATH_GPT_BobsFruitDrinkCost_l1110_111027

theorem BobsFruitDrinkCost 
  (AndySpent : ℕ)
  (BobSpent : ℕ)
  (AndySodaCost : ℕ)
  (AndyHamburgerCost : ℕ)
  (BobSandwichCost : ℕ)
  (FruitDrinkCost : ℕ) :
  AndySpent = 5 ∧ AndySodaCost = 1 ∧ AndyHamburgerCost = 2 ∧ 
  AndySpent = BobSpent ∧ 
  BobSandwichCost = 3 ∧ 
  FruitDrinkCost = BobSpent - BobSandwichCost →
  FruitDrinkCost = 2 := by
  sorry

end NUMINAMATH_GPT_BobsFruitDrinkCost_l1110_111027


namespace NUMINAMATH_GPT_weight_of_A_l1110_111032

theorem weight_of_A (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : (B + C + D + E) / 4 = 79) 
  (h4 : E = D + 7): 
  A = 79 := by
  have h5 : A + B + C = 252 := by
    linarith [h1]
  have h6 : A + B + C + D = 320 := by
    linarith [h2]
  have h7 : B + C + D + E = 316 := by
    linarith [h3]
  have hD : D = 68 := by
    linarith [h5, h6]
  have hE : E = 75 := by
    linarith [hD, h4]
  have hBC : B + C = 252 - A := by
    linarith [h5]
  have : 252 - A + 68 + 75 = 316 := by
    linarith [h7, hBC, hD, hE]
  linarith

end NUMINAMATH_GPT_weight_of_A_l1110_111032


namespace NUMINAMATH_GPT_place_mat_length_l1110_111038

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) (inner_touch : Bool)
  (h1 : r = 4)
  (h2 : n = 6)
  (h3 : w = 1)
  (h4 : inner_touch = true)
  : x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_place_mat_length_l1110_111038


namespace NUMINAMATH_GPT_product_of_coordinates_of_D_l1110_111084

theorem product_of_coordinates_of_D 
  (x y : ℝ)
  (midpoint_x : (5 + x) / 2 = 4)
  (midpoint_y : (3 + y) / 2 = 7) : 
  x * y = 33 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_coordinates_of_D_l1110_111084


namespace NUMINAMATH_GPT_length_of_each_train_l1110_111039

noncomputable def length_of_train : ℝ := 
  let speed_fast := 46 -- in km/hr
  let speed_slow := 36 -- in km/hr
  let relative_speed := speed_fast - speed_slow -- 10 km/hr
  let relative_speed_km_per_sec := relative_speed / 3600.0 -- converting to km/sec
  let time_sec := 18.0 -- time in seconds
  let distance_km := relative_speed_km_per_sec * time_sec -- calculates distance in km
  distance_km * 1000.0 -- converts to meters

theorem length_of_each_train : length_of_train = 50 :=
  by
    sorry

end NUMINAMATH_GPT_length_of_each_train_l1110_111039


namespace NUMINAMATH_GPT_time_to_pass_l1110_111015
-- Import the Mathlib library

-- Define the lengths of the trains
def length_train1 := 150 -- meters
def length_train2 := 150 -- meters

-- Define the speeds of the trains in km/h
def speed_train1_kmh := 95 -- km/h
def speed_train2_kmh := 85 -- km/h

-- Convert speeds to m/s
def speed_train1_ms := (speed_train1_kmh * 1000) / 3600 -- meters per second
def speed_train2_ms := (speed_train2_kmh * 1000) / 3600 -- meters per second

-- Calculate the relative speed in m/s (since they move in opposite directions, the relative speed is additive)
def relative_speed_ms := speed_train1_ms + speed_train2_ms -- meters per second

-- Calculate the total distance to be covered (sum of the lengths of the trains)
def total_length := length_train1 + length_train2 -- meters

-- State the theorem: the time taken for the trains to pass each other
theorem time_to_pass :
  total_length / relative_speed_ms = 6 := by
  sorry

end NUMINAMATH_GPT_time_to_pass_l1110_111015


namespace NUMINAMATH_GPT_annual_income_of_A_l1110_111083

theorem annual_income_of_A 
  (ratio_AB : ℕ → ℕ → Prop)
  (income_C : ℕ)
  (income_B_more_C : ℕ → ℕ → Prop)
  (income_B_from_ratio : ℕ → ℕ → Prop)
  (income_C_value : income_C = 16000)
  (income_B_condition : ∀ c, income_B_more_C 17920 c)
  (income_A_condition : ∀ b, ratio_AB 5 (b/2))
  : ∃ a, a = 537600 :=
by
  sorry

end NUMINAMATH_GPT_annual_income_of_A_l1110_111083


namespace NUMINAMATH_GPT_file_organization_ratio_l1110_111078

variable (X : ℕ) -- The number of files organized in the morning
variable (total_files morning_files afternoon_files missing_files : ℕ)

-- Conditions
def condition1 : total_files = 60 := by sorry
def condition2 : afternoon_files = 15 := by sorry
def condition3 : missing_files = 15 := by sorry
def condition4 : morning_files = X := by sorry
def condition5 : morning_files + afternoon_files + missing_files = total_files := by sorry

-- Question
def ratio_morning_to_total : Prop :=
  let organized_files := total_files - afternoon_files - missing_files
  (organized_files / total_files : ℚ) = 1 / 2

-- Proof statement
theorem file_organization_ratio : 
  ∀ (X total_files morning_files afternoon_files missing_files : ℕ), 
    total_files = 60 → 
    afternoon_files = 15 → 
    missing_files = 15 → 
    morning_files = X → 
    morning_files + afternoon_files + missing_files = total_files → 
    (X / 60 : ℚ) = 1 / 2 := by 
  sorry

end NUMINAMATH_GPT_file_organization_ratio_l1110_111078


namespace NUMINAMATH_GPT_unequal_numbers_l1110_111059

theorem unequal_numbers {k : ℚ} (h : 3 * (1 : ℚ) + 7 * (1 : ℚ) + 2 * k = 0) (d : (7^2 : ℚ) - 4 * 3 * 2 * k = 0) : 
    (3 : ℚ) ≠ (7 : ℚ) ∧ (3 : ℚ) ≠ k ∧ (7 : ℚ) ≠ k :=
by
  -- adding sorry for skipping proof
  sorry

end NUMINAMATH_GPT_unequal_numbers_l1110_111059


namespace NUMINAMATH_GPT_max_prime_difference_l1110_111046

theorem max_prime_difference (a b c d : ℕ) 
  (p1 : Prime a) (p2 : Prime b) (p3 : Prime c) (p4 : Prime d)
  (p5 : Prime (a + b + c + 18 + d)) (p6 : Prime (a + b + c + 18 - d))
  (p7 : Prime (b + c)) (p8 : Prime (c + d))
  (h1 : a + b + c = 2010) (h2 : a ≠ 3) (h3 : b ≠ 3) (h4 : c ≠ 3) (h5 : d ≠ 3) (h6 : d ≤ 50)
  (distinct_primes : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ (a + b + c + 18 + d)
                    ∧ a ≠ (a + b + c + 18 - d) ∧ a ≠ (b + c) ∧ a ≠ (c + d)
                    ∧ b ≠ c ∧ b ≠ d ∧ b ≠ (a + b + c + 18 + d)
                    ∧ b ≠ (a + b + c + 18 - d) ∧ b ≠ (b + c) ∧ b ≠ (c + d)
                    ∧ c ≠ d ∧ c ≠ (a + b + c + 18 + d)
                    ∧ c ≠ (a + b + c + 18 - d) ∧ c ≠ (b + c) ∧ c ≠ (c + d)
                    ∧ d ≠ (a + b + c + 18 + d) ∧ d ≠ (a + b + c + 18 - d)
                    ∧ d ≠ (b + c) ∧ d ≠ (c + d)
                    ∧ (a + b + c + 18 + d) ≠ (a + b + c + 18 - d)
                    ∧ (a + b + c + 18 + d) ≠ (b + c) ∧ (a + b + c + 18 + d) ≠ (c + d)
                    ∧ (a + b + c + 18 - d) ≠ (b + c) ∧ (a + b + c + 18 - d) ≠ (c + d)
                    ∧ (b + c) ≠ (c + d)) :
  ∃ max_diff : ℕ, max_diff = 2067 := sorry

end NUMINAMATH_GPT_max_prime_difference_l1110_111046


namespace NUMINAMATH_GPT_max_tickets_l1110_111082

theorem max_tickets (n : ℕ) (H : 15 * n ≤ 120) : n ≤ 8 :=
by sorry

end NUMINAMATH_GPT_max_tickets_l1110_111082


namespace NUMINAMATH_GPT_find_c_l1110_111042

theorem find_c (x : ℝ) (c : ℝ) (h1: 3 * x + 6 = 0) (h2: c * x + 15 = 3) : c = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_c_l1110_111042


namespace NUMINAMATH_GPT_price_of_72_cans_l1110_111044

def regular_price_per_can : ℝ := 0.60
def discount_percentage : ℝ := 0.20
def total_price : ℝ := 34.56

theorem price_of_72_cans (discounted_price_per_can : ℝ) (number_of_cans : ℕ)
  (H1 : discounted_price_per_can = regular_price_per_can - (discount_percentage * regular_price_per_can))
  (H2 : number_of_cans = total_price / discounted_price_per_can) :
  total_price = number_of_cans * discounted_price_per_can := by
  sorry

end NUMINAMATH_GPT_price_of_72_cans_l1110_111044


namespace NUMINAMATH_GPT_wage_difference_l1110_111064

variable (P Q : ℝ)
variable (h : ℝ)
axiom wage_relation : P = 1.5 * Q
axiom time_relation : 360 = P * h
axiom time_relation_q : 360 = Q * (h + 10)

theorem wage_difference : P - Q = 6 :=
  by
  sorry

end NUMINAMATH_GPT_wage_difference_l1110_111064


namespace NUMINAMATH_GPT_stickers_total_l1110_111028

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end NUMINAMATH_GPT_stickers_total_l1110_111028


namespace NUMINAMATH_GPT_sum_series_75_to_99_l1110_111057

theorem sum_series_75_to_99 : 
  let a := 75
  let l := 99
  let n := l - a + 1
  let s := n * (a + l) / 2
  s = 2175 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_75_to_99_l1110_111057


namespace NUMINAMATH_GPT_sculpture_cost_in_CNY_l1110_111014

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sculpture_cost_in_CNY_l1110_111014


namespace NUMINAMATH_GPT_log_equation_solution_l1110_111013

theorem log_equation_solution (x : ℝ) (h : Real.log x + Real.log (x + 4) = Real.log (2 * x + 8)) : x = 2 :=
sorry

end NUMINAMATH_GPT_log_equation_solution_l1110_111013


namespace NUMINAMATH_GPT_corrections_needed_l1110_111009

-- Define the corrected statements
def corrected_statements : List String :=
  ["A = 50", "B = A", "x = 1", "y = 2", "z = 3", "INPUT“How old are you?”;x",
   "INPUT x", "PRINT“A+B=”;C", "PRINT“Good-bye!”"]

-- Define the function to check if the statement is correctly formatted
def is_corrected (statement : String) : Prop :=
  statement ∈ corrected_statements

-- Lean theorem statement to prove each original incorrect statement should be correctly formatted
theorem corrections_needed (s : String) (incorrect : s ∈ ["A = B = 50", "x = 1, y = 2, z = 3", 
  "INPUT“How old are you”x", "INPUT, x", "PRINT A+B=;C", "PRINT Good-bye!"]) :
  ∃ t : String, is_corrected t :=
by 
  sorry

end NUMINAMATH_GPT_corrections_needed_l1110_111009


namespace NUMINAMATH_GPT_distinct_pairwise_products_l1110_111020

theorem distinct_pairwise_products
  (n a b c d : ℕ) (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_bounds: n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2) :
  (a * b ≠ a * c ∧ a * b ≠ a * d ∧ a * b ≠ b * c ∧ a * b ≠ b * d ∧ a * b ≠ c * d) ∧
  (a * c ≠ a * d ∧ a * c ≠ b * c ∧ a * c ≠ b * d ∧ a * c ≠ c * d) ∧
  (a * d ≠ b * c ∧ a * d ≠ b * d ∧ a * d ≠ c * d) ∧
  (b * c ≠ b * d ∧ b * c ≠ c * d) ∧
  (b * d ≠ c * d) :=
sorry

end NUMINAMATH_GPT_distinct_pairwise_products_l1110_111020


namespace NUMINAMATH_GPT_vector_computation_equiv_l1110_111030

variables (u v w : ℤ × ℤ)

def vector_expr (u v w : ℤ × ℤ) :=
  2 • u + 4 • v - 3 • w

theorem vector_computation_equiv :
  u = (3, -5) →
  v = (-1, 6) →
  w = (2, -4) →
  vector_expr u v w = (-4, 26) :=
by
  intros hu hv hw
  rw [hu, hv, hw]
  dsimp [vector_expr]
  -- The actual proof goes here, but we use 'sorry' to skip it.
  sorry

end NUMINAMATH_GPT_vector_computation_equiv_l1110_111030


namespace NUMINAMATH_GPT_felix_trees_per_sharpening_l1110_111071

theorem felix_trees_per_sharpening (dollars_spent : ℕ) (cost_per_sharpen : ℕ) (trees_chopped : ℕ) 
  (h1 : dollars_spent = 35) (h2 : cost_per_sharpen = 5) (h3 : trees_chopped ≥ 91) :
  (91 / (35 / 5)) = 13 := 
by 
  sorry

end NUMINAMATH_GPT_felix_trees_per_sharpening_l1110_111071


namespace NUMINAMATH_GPT_total_time_taken_l1110_111094

theorem total_time_taken (b km : ℝ) : 
  (b / 50 + km / 80) = (8 * b + 5 * km) / 400 := 
sorry

end NUMINAMATH_GPT_total_time_taken_l1110_111094


namespace NUMINAMATH_GPT_find_constant_term_of_polynomial_with_negative_integer_roots_l1110_111080

theorem find_constant_term_of_polynomial_with_negative_integer_roots
  (p q r s : ℝ) (t1 t2 t3 t4 : ℝ)
  (h_roots : ∀ {x : ℝ}, x^4 + p*x^3 + q*x^2 + r*x + s = (x + t1)*(x + t2)*(x + t3)*(x + t4))
  (h_neg_int_roots : ∀ {i : ℕ}, i < 4 → t1 = i ∨ t2 = i ∨ t3 = i ∨ t4 = i)
  (h_sum_coeffs : p + q + r + s = 168) :
  s = 144 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_term_of_polynomial_with_negative_integer_roots_l1110_111080


namespace NUMINAMATH_GPT_sin_2alpha_pos_if_tan_alpha_pos_l1110_111091

theorem sin_2alpha_pos_if_tan_alpha_pos (α : ℝ) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end NUMINAMATH_GPT_sin_2alpha_pos_if_tan_alpha_pos_l1110_111091


namespace NUMINAMATH_GPT_oranges_in_bin_after_changes_l1110_111012

def initial_oranges := 31
def thrown_away_oranges := 9
def new_oranges := 38

theorem oranges_in_bin_after_changes : 
  initial_oranges - thrown_away_oranges + new_oranges = 60 := by
  sorry

end NUMINAMATH_GPT_oranges_in_bin_after_changes_l1110_111012


namespace NUMINAMATH_GPT_determinant_condition_l1110_111018

variable (p q r s : ℝ)

theorem determinant_condition (h: p * s - q * r = 5) :
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 20 :=
by
  sorry

end NUMINAMATH_GPT_determinant_condition_l1110_111018


namespace NUMINAMATH_GPT_james_tylenol_intake_per_day_l1110_111055

variable (hours_in_day : ℕ := 24) 
variable (tablets_per_dose : ℕ := 2) 
variable (mg_per_tablet : ℕ := 375)
variable (hours_per_dose : ℕ := 6)

theorem james_tylenol_intake_per_day :
  (tablets_per_dose * mg_per_tablet) * (hours_in_day / hours_per_dose) = 3000 := by
  sorry

end NUMINAMATH_GPT_james_tylenol_intake_per_day_l1110_111055


namespace NUMINAMATH_GPT_QT_value_l1110_111006

noncomputable def find_QT (PQ RS PT : ℝ) : ℝ :=
  let tan_gamma := (RS / PQ)
  let QT := (RS / tan_gamma) - PT
  QT

theorem QT_value :
  let PQ := 45
  let RS := 75
  let PT := 15
  find_QT PQ RS PT = 210 := by
  sorry

end NUMINAMATH_GPT_QT_value_l1110_111006


namespace NUMINAMATH_GPT_convert_degrees_to_radians_l1110_111056

theorem convert_degrees_to_radians (θ : ℝ) (h : θ = -630) : θ * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_convert_degrees_to_radians_l1110_111056


namespace NUMINAMATH_GPT_largest_circle_radius_l1110_111003

noncomputable def largest_inscribed_circle_radius (AB BC CD DA : ℝ) : ℝ :=
  let s := (AB + BC + CD + DA) / 2
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA))
  A / s

theorem largest_circle_radius {AB BC CD DA : ℝ} (hAB : AB = 10) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 13)
  : largest_inscribed_circle_radius AB BC CD DA = 3 * Real.sqrt 245 / 10 :=
by
  simp [largest_inscribed_circle_radius, hAB, hBC, hCD, hDA]
  sorry

end NUMINAMATH_GPT_largest_circle_radius_l1110_111003
