import Mathlib

namespace NUMINAMATH_GPT_pinky_pig_apples_l853_85310

variable (P : ℕ)

theorem pinky_pig_apples (h : P + 73 = 109) : P = 36 := sorry

end NUMINAMATH_GPT_pinky_pig_apples_l853_85310


namespace NUMINAMATH_GPT_gcd_lcm_252_l853_85369

theorem gcd_lcm_252 {a b : ℕ} (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃ S : Finset ℕ, S.card = 8 ∧ ∀ d ∈ S, d = Nat.gcd a b :=
by sorry

end NUMINAMATH_GPT_gcd_lcm_252_l853_85369


namespace NUMINAMATH_GPT_trains_cross_in_12_seconds_l853_85302

noncomputable def length := 120 -- Length of each train in meters
noncomputable def time_train1 := 10 -- Time taken by the first train to cross the post in seconds
noncomputable def time_train2 := 15 -- Time taken by the second train to cross the post in seconds

noncomputable def speed_train1 := length / time_train1 -- Speed of the first train in m/s
noncomputable def speed_train2 := length / time_train2 -- Speed of the second train in m/s

noncomputable def relative_speed := speed_train1 + speed_train2 -- Relative speed when traveling in opposite directions in m/s
noncomputable def total_length := 2 * length -- Total distance covered when crossing each other

noncomputable def crossing_time := total_length / relative_speed -- Time to cross each other in seconds

theorem trains_cross_in_12_seconds : crossing_time = 12 := by
  sorry

end NUMINAMATH_GPT_trains_cross_in_12_seconds_l853_85302


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_is_171_l853_85385

theorem arithmetic_sequence_15th_term_is_171 :
  ∀ (a d : ℕ), a = 3 → d = 15 - a → a + 14 * d = 171 :=
by
  intros a d h_a h_d
  rw [h_a, h_d]
  -- The proof would follow with the arithmetic calculation to determine the 15th term
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_is_171_l853_85385


namespace NUMINAMATH_GPT_john_not_stronger_than_ivan_l853_85337

-- Define strength relations
axiom stronger (a b : Type) : Prop

variable (whiskey liqueur vodka beer : Type)

axiom whiskey_stronger_than_vodka : stronger whiskey vodka
axiom liqueur_stronger_than_beer : stronger liqueur beer

-- Define types for cocktails and their strengths
variable (John_cocktail Ivan_cocktail : Type)

axiom John_mixed_whiskey_liqueur : John_cocktail
axiom Ivan_mixed_vodka_beer : Ivan_cocktail

-- Prove that it can't be asserted that John's cocktail is stronger
theorem john_not_stronger_than_ivan :
  ¬ (stronger John_cocktail Ivan_cocktail) :=
sorry

end NUMINAMATH_GPT_john_not_stronger_than_ivan_l853_85337


namespace NUMINAMATH_GPT_seated_people_count_l853_85343

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end NUMINAMATH_GPT_seated_people_count_l853_85343


namespace NUMINAMATH_GPT_problem1_problem2_l853_85390

variables {a b c : ℝ}

-- (1) Prove that a + b + c = 4 given the conditions
theorem problem1 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_min : ∀ x, abs (x + a) + abs (x - b) + c ≥ 4) : a + b + c = 4 := 
sorry

-- (2) Prove that the minimum value of (1/4)a^2 + (1/9)b^2 + c^2 is 8/7 given the conditions and that a + b + c = 4
theorem problem2 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8 / 7 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l853_85390


namespace NUMINAMATH_GPT_mary_flour_total_l853_85375

-- Definitions for conditions
def initial_flour : ℝ := 7.0
def extra_flour : ℝ := 2.0
def total_flour (x y : ℝ) : ℝ := x + y

-- The statement we want to prove
theorem mary_flour_total : total_flour initial_flour extra_flour = 9.0 := 
by sorry

end NUMINAMATH_GPT_mary_flour_total_l853_85375


namespace NUMINAMATH_GPT_gigi_has_15_jellybeans_l853_85349

variable (G : ℕ) -- G is the number of jellybeans Gigi has
variable (R : ℕ) -- R is the number of jellybeans Rory has
variable (L : ℕ) -- L is the number of jellybeans Lorelai has eaten

-- Conditions
def condition1 := R = G + 30
def condition2 := L = 3 * (G + R)
def condition3 := L = 180

-- Proof statement
theorem gigi_has_15_jellybeans (G R L : ℕ) (h1 : condition1 G R) (h2 : condition2 G R L) (h3 : condition3 L) : G = 15 := by
  sorry

end NUMINAMATH_GPT_gigi_has_15_jellybeans_l853_85349


namespace NUMINAMATH_GPT_positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l853_85332

theorem positive_integers_ab_divides_asq_bsq_implies_a_eq_b
  (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : a * b ∣ a^2 + b^2) : a = b := by
  sorry

end NUMINAMATH_GPT_positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l853_85332


namespace NUMINAMATH_GPT_sum_consecutive_triangular_sum_triangular_2020_l853_85309

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to be proved
theorem sum_consecutive_triangular (n : ℕ) : triangular n + triangular (n + 1) = (n + 1)^2 :=
by 
  sorry

-- Applying the theorem for the specific case of n = 2020
theorem sum_triangular_2020 : triangular 2020 + triangular 2021 = 2021^2 :=
by 
  exact sum_consecutive_triangular 2020

end NUMINAMATH_GPT_sum_consecutive_triangular_sum_triangular_2020_l853_85309


namespace NUMINAMATH_GPT_friendP_walks_23_km_l853_85397

noncomputable def friendP_distance (v : ℝ) : ℝ :=
  let trail_length := 43
  let speedP := 1.15 * v
  let speedQ := v
  let dQ := trail_length - 23
  let timeP := 23 / speedP
  let timeQ := dQ / speedQ
  if timeP = timeQ then 23 else 0  -- Ensuring that both reach at the same time.

theorem friendP_walks_23_km (v : ℝ) : 
  friendP_distance v = 23 :=
by
  sorry

end NUMINAMATH_GPT_friendP_walks_23_km_l853_85397


namespace NUMINAMATH_GPT_min_xy_eq_nine_l853_85329

theorem min_xy_eq_nine (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + y + 3) : x * y = 9 :=
sorry

end NUMINAMATH_GPT_min_xy_eq_nine_l853_85329


namespace NUMINAMATH_GPT_amare_additional_fabric_needed_l853_85316

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end NUMINAMATH_GPT_amare_additional_fabric_needed_l853_85316


namespace NUMINAMATH_GPT_words_per_page_large_font_l853_85393

theorem words_per_page_large_font
    (total_words : ℕ)
    (large_font_pages : ℕ)
    (small_font_pages : ℕ)
    (small_font_words_per_page : ℕ)
    (total_pages : ℕ)
    (words_in_large_font : ℕ) :
    total_words = 48000 →
    total_pages = 21 →
    large_font_pages = 4 →
    small_font_words_per_page = 2400 →
    words_in_large_font = total_words - (small_font_pages * small_font_words_per_page) →
    small_font_pages = total_pages - large_font_pages →
    (words_in_large_font = large_font_pages * 1800) :=
by 
    sorry

end NUMINAMATH_GPT_words_per_page_large_font_l853_85393


namespace NUMINAMATH_GPT_sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l853_85355

noncomputable def f (x b : ℝ) : ℝ := x^2 + b*x

theorem sufficient_condition_for_min_value (b : ℝ) : b < 0 → ∀ x, min (f (f x b) b) = min (f x b) :=
sorry

theorem not_necessary_condition_for_min_value (b : ℝ) : (b < 0) ∧ (∀ x, min (f (f x b) b) = min (f x b)) → b ≤ 0 ∨ b ≥ 2 := 
sorry

end NUMINAMATH_GPT_sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l853_85355


namespace NUMINAMATH_GPT_raven_current_age_l853_85395

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end NUMINAMATH_GPT_raven_current_age_l853_85395


namespace NUMINAMATH_GPT_recipe_required_ingredients_l853_85352

-- Define the number of cups required for each ingredient in the recipe
def sugar_cups : Nat := 11
def flour_cups : Nat := 8
def cocoa_cups : Nat := 5

-- Define the cups of flour and cocoa already added
def flour_already_added : Nat := 3
def cocoa_already_added : Nat := 2

-- Define the cups of flour and cocoa that still need to be added
def flour_needed_to_add : Nat := 6
def cocoa_needed_to_add : Nat := 3

-- Sum the total amount of flour and cocoa powder based on already added and still needed amounts
def total_flour: Nat := flour_already_added + flour_needed_to_add
def total_cocoa: Nat := cocoa_already_added + cocoa_needed_to_add

-- Total ingredients calculation according to the problem's conditions
def total_ingredients : Nat := sugar_cups + total_flour + total_cocoa

-- The theorem to be proved
theorem recipe_required_ingredients : total_ingredients = 24 := by
  sorry

end NUMINAMATH_GPT_recipe_required_ingredients_l853_85352


namespace NUMINAMATH_GPT_find_b_l853_85305

-- Define the constants and assumptions
variables {a b c d : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d)

-- The function completes 5 periods between 0 and 2π
def completes_5_periods (b : ℝ) : Prop :=
  (2 * Real.pi) / b = (2 * Real.pi) / 5

theorem find_b (h : completes_5_periods b) : b = 5 :=
sorry

end NUMINAMATH_GPT_find_b_l853_85305


namespace NUMINAMATH_GPT_battery_life_remaining_l853_85360

variables (full_battery_life : ℕ) (used_fraction : ℚ) (exam_duration : ℕ) (remaining_battery : ℕ)

def brody_calculator_conditions :=
  full_battery_life = 60 ∧
  used_fraction = 3 / 4 ∧
  exam_duration = 2

theorem battery_life_remaining
  (h : brody_calculator_conditions full_battery_life used_fraction exam_duration) :
  remaining_battery = 13 :=
by 
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_battery_life_remaining_l853_85360


namespace NUMINAMATH_GPT_largest_divisor_of_n4_sub_4n2_is_4_l853_85347

theorem largest_divisor_of_n4_sub_4n2_is_4 (n : ℤ) : 4 ∣ (n^4 - 4 * n^2) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n4_sub_4n2_is_4_l853_85347


namespace NUMINAMATH_GPT_factorize_x9_minus_512_l853_85368

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end NUMINAMATH_GPT_factorize_x9_minus_512_l853_85368


namespace NUMINAMATH_GPT_melissa_bananas_l853_85322

theorem melissa_bananas (a b : ℕ) (h1 : a = 88) (h2 : b = 4) : a - b = 84 :=
by
  sorry

end NUMINAMATH_GPT_melissa_bananas_l853_85322


namespace NUMINAMATH_GPT_preimage_of_3_2_eq_l853_85394

noncomputable def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

theorem preimage_of_3_2_eq (x y : ℝ) :
  f (x, y) = (-3, 2) ↔ (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 3) :=
by
  sorry

end NUMINAMATH_GPT_preimage_of_3_2_eq_l853_85394


namespace NUMINAMATH_GPT_domain_of_f_l853_85384

def function_domain (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x, x ∈ domain ↔ ∃ y, f y = x

noncomputable def f (x : ℝ) : ℝ :=
  (x + 6) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f :
  function_domain f ((Set.Iio 2) ∪ (Set.Ioi 3)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l853_85384


namespace NUMINAMATH_GPT_jerome_contact_list_l853_85371

def classmates := 20
def out_of_school_friends := classmates / 2
def family_members := 3
def total_contacts := classmates + out_of_school_friends + family_members

theorem jerome_contact_list : total_contacts = 33 := by
  sorry

end NUMINAMATH_GPT_jerome_contact_list_l853_85371


namespace NUMINAMATH_GPT_divides_expression_l853_85324

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end NUMINAMATH_GPT_divides_expression_l853_85324


namespace NUMINAMATH_GPT_group_A_can_form_triangle_l853_85321

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem group_A_can_form_triangle : can_form_triangle 9 6 13 :=
by
  sorry

end NUMINAMATH_GPT_group_A_can_form_triangle_l853_85321


namespace NUMINAMATH_GPT_correct_calculation_l853_85330

theorem correct_calculation (x : ℕ) (h1 : 21 * x = 63) : x + 40 = 43 :=
by
  -- proof steps would go here, but we skip them with 'sorry'
  sorry

end NUMINAMATH_GPT_correct_calculation_l853_85330


namespace NUMINAMATH_GPT_triangle_perimeter_l853_85304

-- Definitions of the geometric problem conditions
def inscribed_circle_tangent (A B C P : Type) : Prop := sorry
def radius_of_inscribed_circle (r : ℕ) : Prop := r = 24
def segment_lengths (AP PB : ℕ) : Prop := AP = 25 ∧ PB = 29

-- Main theorem to prove the perimeter of the triangle ABC
theorem triangle_perimeter (A B C P : Type) (r AP PB : ℕ)
  (H1 : inscribed_circle_tangent A B C P)
  (H2 : radius_of_inscribed_circle r)
  (H3 : segment_lengths AP PB) :
  2 * (54 + 208.72) = 525.44 :=
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l853_85304


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l853_85358

-- Definitions of the sets
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l853_85358


namespace NUMINAMATH_GPT_problem1_l853_85362

theorem problem1 {a b c : ℝ} (h : a + b + c = 2) : a^2 + b^2 + c^2 + 2 * a * b * c < 2 :=
sorry

end NUMINAMATH_GPT_problem1_l853_85362


namespace NUMINAMATH_GPT_main_theorem_l853_85317

noncomputable def circle_center : Prop :=
  ∃ x y : ℝ, 2*x - y - 7 = 0 ∧ y = -3 ∧ x = 2

noncomputable def circle_equation : Prop :=
  (∀ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5)

noncomputable def tangent_condition (k : ℝ) : Prop :=
  (3 + 3*k)^2 / (1 + k^2) = 5

noncomputable def symmetric_circle_center : Prop :=
  ∃ x y : ℝ, x = -22/5 ∧ y = 1/5

noncomputable def symmetric_circle_equation : Prop :=
  (∀ (x y : ℝ), (x + 22/5)^2 + (y - 1/5)^2 = 5)

theorem main_theorem : circle_center → circle_equation ∧ (∃ k : ℝ, tangent_condition k) ∧ symmetric_circle_center → symmetric_circle_equation :=
  by sorry

end NUMINAMATH_GPT_main_theorem_l853_85317


namespace NUMINAMATH_GPT_remainder_2503_div_28_l853_85308

theorem remainder_2503_div_28 : 2503 % 28 = 11 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_remainder_2503_div_28_l853_85308


namespace NUMINAMATH_GPT_half_ears_kernels_l853_85364

theorem half_ears_kernels (stalks ears_per_stalk total_kernels : ℕ) (X : ℕ)
  (half_ears : ℕ := stalks * ears_per_stalk / 2)
  (total_ears : ℕ := stalks * ears_per_stalk)
  (condition_e1 : stalks = 108)
  (condition_e2 : ears_per_stalk = 4)
  (condition_e3 : total_kernels = 237600)
  (condition_kernel_sum : total_kernels = 216 * X + 216 * (X + 100)) :
  X = 500 := by
  have condition_eq : 432 * X + 21600 = 237600 := by sorry
  have X_value : X = 216000 / 432 := by sorry
  have X_result : X = 500 := by sorry
  exact X_result

end NUMINAMATH_GPT_half_ears_kernels_l853_85364


namespace NUMINAMATH_GPT_optionA_optionC_optionD_l853_85380

noncomputable def f (x : ℝ) := (3 : ℝ) ^ x / (1 + (3 : ℝ) ^ x)

theorem optionA : ∀ x : ℝ, f (-x) + f x = 1 := by
  sorry

theorem optionC : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (y > 0 ∧ y < 1) := by
  sorry

theorem optionD : ∀ x : ℝ, f (2 * x - 3) + f (x - 3) > 1 ↔ x > 2 := by
  sorry

end NUMINAMATH_GPT_optionA_optionC_optionD_l853_85380


namespace NUMINAMATH_GPT_daily_harvest_l853_85301

theorem daily_harvest (sacks_per_section : ℕ) (num_sections : ℕ) 
  (h1 : sacks_per_section = 45) (h2 : num_sections = 8) : 
  sacks_per_section * num_sections = 360 :=
by
  sorry

end NUMINAMATH_GPT_daily_harvest_l853_85301


namespace NUMINAMATH_GPT_factorize_x_squared_minus_25_l853_85318

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_25_l853_85318


namespace NUMINAMATH_GPT_range_of_a_l853_85366

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≥ f x₂)
                    (h₂ : -2 ≤ a + 1 ∧ a + 1 ≤ 4)
                    (h₃ : -2 ≤ 2 * a ∧ 2 * a ≤ 4)
                    (h₄ : f (a + 1) > f (2 * a)) : 1 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l853_85366


namespace NUMINAMATH_GPT_jean_spots_l853_85372

theorem jean_spots (total_spots upper_torso_spots back_hindspots sides_spots : ℕ)
  (h1 : upper_torso_spots = 30)
  (h2 : total_spots = 2 * upper_torso_spots)
  (h3 : back_hindspots = total_spots / 3)
  (h4 : sides_spots = total_spots - upper_torso_spots - back_hindspots) :
  sides_spots = 10 :=
by
  sorry

end NUMINAMATH_GPT_jean_spots_l853_85372


namespace NUMINAMATH_GPT_product_of_sums_of_four_squares_is_sum_of_four_squares_l853_85386

theorem product_of_sums_of_four_squares_is_sum_of_four_squares (x1 x2 x3 x4 y1 y2 y3 y4 : ℤ) :
  let a := x1^2 + x2^2 + x3^2 + x4^2
  let b := y1^2 + y2^2 + y3^2 + y4^2
  let z1 := x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4
  let z2 := x1 * y2 - x2 * y1 + x3 * y4 - x4 * y3
  let z3 := x1 * y3 - x3 * y1 + x4 * y2 - x2 * y4
  let z4 := x1 * y4 - x4 * y1 + x2 * y3 - x3 * y2
  a * b = z1^2 + z2^2 + z3^2 + z4^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_sums_of_four_squares_is_sum_of_four_squares_l853_85386


namespace NUMINAMATH_GPT_w_z_ratio_l853_85377

theorem w_z_ratio (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2023) : (w + z) / (w - z) = -2023 :=
by sorry

end NUMINAMATH_GPT_w_z_ratio_l853_85377


namespace NUMINAMATH_GPT_linlin_speed_l853_85398

theorem linlin_speed (distance time : ℕ) (q_speed linlin_speed : ℕ)
  (h1 : distance = 3290)
  (h2 : time = 7)
  (h3 : q_speed = 70)
  (h4 : distance = (q_speed + linlin_speed) * time) : linlin_speed = 400 :=
by sorry

end NUMINAMATH_GPT_linlin_speed_l853_85398


namespace NUMINAMATH_GPT_solve_equation_l853_85326

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ x = -4 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l853_85326


namespace NUMINAMATH_GPT_candy_distribution_l853_85328

theorem candy_distribution (A B C : ℕ) (x y : ℕ)
  (h1 : A > 2 * B)
  (h2 : B > 3 * C)
  (h3 : A + B + C = 200) :
  (A = 121) ∧ (C = 19) :=
  sorry

end NUMINAMATH_GPT_candy_distribution_l853_85328


namespace NUMINAMATH_GPT_erased_number_is_one_or_twenty_l853_85334

theorem erased_number_is_one_or_twenty (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 20)
  (h₂ : (210 - x) % 19 = 0) : x = 1 ∨ x = 20 :=
  by sorry

end NUMINAMATH_GPT_erased_number_is_one_or_twenty_l853_85334


namespace NUMINAMATH_GPT_longer_part_length_l853_85354

-- Conditions
def total_length : ℕ := 180
def diff_length : ℕ := 32

-- Hypothesis for the shorter part of the wire
def shorter_part (x : ℕ) : Prop :=
  x + (x + diff_length) = total_length

-- The goal is to find the longer part's length
theorem longer_part_length (x : ℕ) (h : shorter_part x) : x + diff_length = 106 := by
  sorry

end NUMINAMATH_GPT_longer_part_length_l853_85354


namespace NUMINAMATH_GPT_eesha_late_by_15_minutes_l853_85323

theorem eesha_late_by_15_minutes 
  (T usual_time : ℕ) (delay : ℕ) (slower_factor : ℚ) (T' : ℕ) 
  (usual_time_eq : usual_time = 60)
  (delay_eq : delay = 30)
  (slower_factor_eq : slower_factor = 0.75)
  (new_time_eq : T' = unusual_time * slower_factor) 
  (T'' : ℕ) (total_time_eq: T'' = T' + delay)
  (time_taken : ℕ) (time_diff_eq : time_taken = T'' - usual_time) :
  time_taken = 15 :=
by
  -- Proof construction
  sorry

end NUMINAMATH_GPT_eesha_late_by_15_minutes_l853_85323


namespace NUMINAMATH_GPT_steven_has_15_more_peaches_than_jill_l853_85336

-- Definitions based on conditions
def peaches_jill : ℕ := 12
def peaches_jake : ℕ := peaches_jill - 1
def peaches_steven : ℕ := peaches_jake + 16

-- The proof problem
theorem steven_has_15_more_peaches_than_jill : peaches_steven - peaches_jill = 15 := by
  sorry

end NUMINAMATH_GPT_steven_has_15_more_peaches_than_jill_l853_85336


namespace NUMINAMATH_GPT_projectile_reaches_50_first_at_0point5_l853_85314

noncomputable def height_at_time (t : ℝ) : ℝ := -16 * t^2 + 100 * t

theorem projectile_reaches_50_first_at_0point5 :
  ∃ t : ℝ, (height_at_time t = 50) ∧ (t = 0.5) :=
sorry

end NUMINAMATH_GPT_projectile_reaches_50_first_at_0point5_l853_85314


namespace NUMINAMATH_GPT_equivalent_problem_l853_85344

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_problem_l853_85344


namespace NUMINAMATH_GPT_part1_part2_l853_85342

-- Definition of the operation '※'
def operation (a b : ℝ) : ℝ := a^2 - b^2

-- Part 1: Proving 2※(-4) = -12
theorem part1 : operation 2 (-4) = -12 := 
by
  sorry

-- Part 2: Proving the solutions to the equation (x + 5)※3 = 0 are x = -8 and x = -2
theorem part2 : (∃ x : ℝ, operation (x + 5) 3 = 0) ↔ (x = -8 ∨ x = -2) := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l853_85342


namespace NUMINAMATH_GPT_fabric_amount_for_each_dress_l853_85379

def number_of_dresses (total_hours : ℕ) (hours_per_dress : ℕ) : ℕ :=
  total_hours / hours_per_dress 

def fabric_per_dress (total_fabric : ℕ) (number_of_dresses : ℕ) : ℕ :=
  total_fabric / number_of_dresses

theorem fabric_amount_for_each_dress (total_fabric : ℕ) (hours_per_dress : ℕ) (total_hours : ℕ) :
  total_fabric = 56 ∧ hours_per_dress = 3 ∧ total_hours = 42 →
  fabric_per_dress total_fabric (number_of_dresses total_hours hours_per_dress) = 4 :=
by
  sorry

end NUMINAMATH_GPT_fabric_amount_for_each_dress_l853_85379


namespace NUMINAMATH_GPT_domain_g_l853_85341

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_g_l853_85341


namespace NUMINAMATH_GPT_derivative_f_eq_l853_85357

noncomputable def f (x : ℝ) : ℝ :=
  (7^x * (3 * Real.sin (3 * x) + Real.cos (3 * x) * Real.log 7)) / (9 + Real.log 7 ^ 2)

theorem derivative_f_eq :
  ∀ x : ℝ, deriv f x = 7^x * Real.cos (3 * x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_derivative_f_eq_l853_85357


namespace NUMINAMATH_GPT_rachel_remaining_pictures_l853_85333

theorem rachel_remaining_pictures 
  (p1 p2 p_colored : ℕ)
  (h1 : p1 = 23)
  (h2 : p2 = 32)
  (h3 : p_colored = 44) :
  (p1 + p2 - p_colored = 11) :=
by
  sorry

end NUMINAMATH_GPT_rachel_remaining_pictures_l853_85333


namespace NUMINAMATH_GPT_defect_free_product_probability_is_correct_l853_85338

noncomputable def defect_free_probability : ℝ :=
  let p1 := 0.2
  let p2 := 0.3
  let p3 := 0.5
  let d1 := 0.95
  let d2 := 0.90
  let d3 := 0.80
  p1 * d1 + p2 * d2 + p3 * d3

theorem defect_free_product_probability_is_correct :
  defect_free_probability = 0.86 :=
by
  sorry

end NUMINAMATH_GPT_defect_free_product_probability_is_correct_l853_85338


namespace NUMINAMATH_GPT_pants_and_coat_cost_l853_85312

noncomputable def pants_shirt_costs : ℕ := 100
noncomputable def coat_cost_times_shirt : ℕ := 5
noncomputable def coat_cost : ℕ := 180

theorem pants_and_coat_cost (p s c : ℕ) 
  (h1 : p + s = pants_shirt_costs)
  (h2 : c = coat_cost_times_shirt * s)
  (h3 : c = coat_cost) :
  p + c = 244 :=
by
  sorry

end NUMINAMATH_GPT_pants_and_coat_cost_l853_85312


namespace NUMINAMATH_GPT_glycerin_percentage_proof_l853_85363

-- Conditions given in problem
def original_percentage : ℝ := 0.90
def original_volume : ℝ := 4
def added_volume : ℝ := 0.8

-- Total glycerin in original solution
def glycerin_amount : ℝ := original_percentage * original_volume

-- Total volume after adding water
def new_volume : ℝ := original_volume + added_volume

-- Desired percentage proof statement
theorem glycerin_percentage_proof : 
  (glycerin_amount / new_volume) * 100 = 75 := 
by
  sorry

end NUMINAMATH_GPT_glycerin_percentage_proof_l853_85363


namespace NUMINAMATH_GPT_football_game_spectators_l853_85346

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ)
  (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : 
  total_wristbands / wristbands_per_person = 125 :=
by
  sorry

end NUMINAMATH_GPT_football_game_spectators_l853_85346


namespace NUMINAMATH_GPT_Giovanni_burgers_l853_85387

theorem Giovanni_burgers : 
  let toppings := 10
  let patty_choices := 4
  let topping_combinations := 2 ^ toppings
  let total_combinations := patty_choices * topping_combinations
  total_combinations = 4096 :=
by
  sorry

end NUMINAMATH_GPT_Giovanni_burgers_l853_85387


namespace NUMINAMATH_GPT_children_neither_happy_nor_sad_l853_85378

theorem children_neither_happy_nor_sad (total_children happy_children sad_children : ℕ)
  (total_boys total_girls happy_boys sad_girls boys_neither_happy_nor_sad : ℕ)
  (h₀ : total_children = 60)
  (h₁ : happy_children = 30)
  (h₂ : sad_children = 10)
  (h₃ : total_boys = 19)
  (h₄ : total_girls = 41)
  (h₅ : happy_boys = 6)
  (h₆ : sad_girls = 4)
  (h₇ : boys_neither_happy_nor_sad = 7) :
  total_children - happy_children - sad_children = 20 :=
by
  sorry

end NUMINAMATH_GPT_children_neither_happy_nor_sad_l853_85378


namespace NUMINAMATH_GPT_problem_statement_l853_85361

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 := a 1 = 1
def condition2 := (a 3 + a 4) / (a 1 + a 2) = 4
def increasing := q > 0

-- Definition of S_n
def sum_geom (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)

theorem problem_statement (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_geom : increasing_geometric_sequence a q) 
  (h_condition1 : condition1 a) 
  (h_condition2 : condition2 a) 
  (h_increasing : increasing q)
  (h_sum_geom : sum_geom a q S) : 
  S 5 = 31 :=
sorry

end NUMINAMATH_GPT_problem_statement_l853_85361


namespace NUMINAMATH_GPT_range_2a_minus_b_and_a_div_b_range_3x_minus_y_l853_85327

-- Proof for finding the range of 2a - b and a / b
theorem range_2a_minus_b_and_a_div_b (a b : ℝ) (h_a : 12 < a ∧ a < 60) (h_b : 15 < b ∧ b < 36) : 
  -12 < 2 * a - b ∧ 2 * a - b < 105 ∧ 1 / 3 < a / b ∧ a / b < 4 :=
by
  sorry

-- Proof for finding the range of 3x - y
theorem range_3x_minus_y (x y : ℝ) (h_xy_diff : -1 / 2 < x - y ∧ x - y < 1 / 2) (h_xy_sum : 0 < x + y ∧ x + y < 1) : 
  -1 < 3 * x - y ∧ 3 * x - y < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_2a_minus_b_and_a_div_b_range_3x_minus_y_l853_85327


namespace NUMINAMATH_GPT_gcd_of_gy_and_y_l853_85351

theorem gcd_of_gy_and_y (y : ℕ) (h : ∃ k : ℕ, y = k * 3456) :
  gcd ((5 * y + 4) * (9 * y + 1) * (12 * y + 6) * (3 * y + 9)) y = 216 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_of_gy_and_y_l853_85351


namespace NUMINAMATH_GPT_find_possible_values_of_y_l853_85300

noncomputable def solve_y (x : ℝ) : ℝ :=
  ((x - 3) ^ 2 * (x + 4)) / (2 * x - 4)

theorem find_possible_values_of_y (x : ℝ) 
  (h : x ^ 2 + 9 * (x / (x - 3)) ^ 2 = 90) : 
  solve_y x = 0 ∨ solve_y x = 105.23 := 
sorry

end NUMINAMATH_GPT_find_possible_values_of_y_l853_85300


namespace NUMINAMATH_GPT_original_price_l853_85350

theorem original_price (P : ℝ) (h₁ : P - 0.30 * P = 0.70 * P) (h₂ : P - 0.20 * P = 0.80 * P) (h₃ : 0.70 * P + 0.80 * P = 50) :
  P = 100 / 3 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_original_price_l853_85350


namespace NUMINAMATH_GPT_find_y_l853_85339

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := 
by sorry

end NUMINAMATH_GPT_find_y_l853_85339


namespace NUMINAMATH_GPT_probability_top_card_king_l853_85365

theorem probability_top_card_king :
  let total_cards := 52
  let total_kings := 4
  let probability := total_kings / total_cards
  probability = 1 / 13 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_probability_top_card_king_l853_85365


namespace NUMINAMATH_GPT_rain_probability_in_two_locations_l853_85359

noncomputable def probability_no_rain_A : ℝ := 0.3
noncomputable def probability_no_rain_B : ℝ := 0.4

-- The probability of raining at a location is 1 - the probability of no rain at that location
noncomputable def probability_rain_A : ℝ := 1 - probability_no_rain_A
noncomputable def probability_rain_B : ℝ := 1 - probability_no_rain_B

-- The rain status in location A and location B are independent
theorem rain_probability_in_two_locations :
  probability_rain_A * probability_rain_B = 0.42 := by
  sorry

end NUMINAMATH_GPT_rain_probability_in_two_locations_l853_85359


namespace NUMINAMATH_GPT_percentage_exceed_l853_85313

theorem percentage_exceed (x y : ℝ) (h : y = x + (0.25 * x)) : (y - x) / x * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_exceed_l853_85313


namespace NUMINAMATH_GPT_lengths_C_can_form_triangle_l853_85370

-- Definition of sets of lengths
def lengths_A := (3, 6, 9)
def lengths_B := (3, 5, 9)
def lengths_C := (4, 6, 9)
def lengths_D := (2, 6, 4)

-- Triangle condition for a given set of lengths
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Proof problem statement 
theorem lengths_C_can_form_triangle : can_form_triangle 4 6 9 :=
by
  sorry

end NUMINAMATH_GPT_lengths_C_can_form_triangle_l853_85370


namespace NUMINAMATH_GPT_average_age_increase_l853_85356

theorem average_age_increase (n : ℕ) (m : ℕ) (a b : ℝ) (h1 : n = 19) (h2 : m = 20) (h3 : a = 20) (h4 : b = 40) :
  ((n * a + b) / (n + 1)) - a = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_average_age_increase_l853_85356


namespace NUMINAMATH_GPT_total_sum_lent_l853_85331

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ) 
  (h1 : second_part = 1640) 
  (h2 : (x * 8 * 0.03) = (second_part * 3 * 0.05)) :
  total_sum = x + second_part → total_sum = 2665 := by
  sorry

end NUMINAMATH_GPT_total_sum_lent_l853_85331


namespace NUMINAMATH_GPT_total_time_to_fill_tank_l853_85373

-- Definitions as per conditions
def tank_fill_time_for_one_tap (total_time : ℕ) : Prop :=
  total_time = 16

def number_of_taps_for_second_half (num_taps : ℕ) : Prop :=
  num_taps = 4

-- Theorem statement to prove the total time taken to fill the tank
theorem total_time_to_fill_tank : ∀ (time_one_tap time_total : ℕ),
  tank_fill_time_for_one_tap time_one_tap →
  number_of_taps_for_second_half 4 →
  time_total = 10 :=
by
  intros time_one_tap time_total h1 h2
  -- Proof needed here
  sorry

end NUMINAMATH_GPT_total_time_to_fill_tank_l853_85373


namespace NUMINAMATH_GPT_no_five_coprime_two_digit_composites_l853_85383

/-- 
  Prove that there do not exist five two-digit composite 
  numbers such that each pair of them is coprime, under 
  the conditions that each composite number must be made 
  up of the primes 2, 3, 5, and 7.
-/
theorem no_five_coprime_two_digit_composites :
  ¬∃ (a b c d e : ℕ),
    10 ≤ a ∧ a < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ a → p ∣ a) ∧
    10 ≤ b ∧ b < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ b → p ∣ b) ∧
    10 ≤ c ∧ c < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ c → p ∣ c) ∧
    10 ≤ d ∧ d < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ d → p ∣ d) ∧
    10 ≤ e ∧ e < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ e → p ∣ e) ∧
    ∀ (x y : ℕ), (x ∈ [a, b, c, d, e] ∧ y ∈ [a, b, c, d, e] ∧ x ≠ y) → Nat.gcd x y = 1 :=
by
  sorry

end NUMINAMATH_GPT_no_five_coprime_two_digit_composites_l853_85383


namespace NUMINAMATH_GPT_least_number_subtraction_l853_85335

theorem least_number_subtraction (n : ℕ) (h₀ : n = 3830) (k : ℕ) (h₁ : k = 5) : (n - k) % 15 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_number_subtraction_l853_85335


namespace NUMINAMATH_GPT_product_not_perfect_square_l853_85367

theorem product_not_perfect_square :
  ¬ ∃ n : ℕ, n^2 = (2021^1004) * (6^3) :=
by
  sorry

end NUMINAMATH_GPT_product_not_perfect_square_l853_85367


namespace NUMINAMATH_GPT_find_three_leaf_clovers_l853_85307

-- Define the conditions
def total_leaves : Nat := 1000

-- Define the statement
theorem find_three_leaf_clovers (n : Nat) (h : 3 * n + 4 = total_leaves) : n = 332 :=
  sorry

end NUMINAMATH_GPT_find_three_leaf_clovers_l853_85307


namespace NUMINAMATH_GPT_lower_limit_brother_opinion_l853_85376

variables (w B : ℝ)

-- Conditions
-- Arun's weight is between 61 and 72 kg
def arun_cond := 61 < w ∧ w < 72
-- Arun's brother's opinion: greater than B, less than 70
def brother_cond := B < w ∧ w < 70
-- Arun's mother's view: not greater than 64
def mother_cond :=  w ≤ 64

-- Given the average
def avg_weight := 63

theorem lower_limit_brother_opinion (h_arun : arun_cond w) (h_brother: brother_cond w B) (h_mother: mother_cond w) (h_avg: avg_weight = (B + 64)/2) : 
  B = 62 :=
sorry

end NUMINAMATH_GPT_lower_limit_brother_opinion_l853_85376


namespace NUMINAMATH_GPT_extra_apples_correct_l853_85399

def num_red_apples : ℕ := 6
def num_green_apples : ℕ := 15
def num_students : ℕ := 5
def num_apples_ordered : ℕ := num_red_apples + num_green_apples
def num_apples_taken : ℕ := num_students
def num_extra_apples : ℕ := num_apples_ordered - num_apples_taken

theorem extra_apples_correct : num_extra_apples = 16 := by
  sorry

end NUMINAMATH_GPT_extra_apples_correct_l853_85399


namespace NUMINAMATH_GPT_evaluate_polynomial_at_3_l853_85391

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem evaluate_polynomial_at_3 : f 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_3_l853_85391


namespace NUMINAMATH_GPT_sum_of_ages_l853_85389

theorem sum_of_ages 
  (a1 a2 a3 : ℕ) 
  (h1 : a1 ≠ a2) 
  (h2 : a1 ≠ a3) 
  (h3 : a2 ≠ a3) 
  (h4 : 1 ≤ a1 ∧ a1 ≤ 9) 
  (h5 : 1 ≤ a2 ∧ a2 ≤ 9) 
  (h6 : 1 ≤ a3 ∧ a3 ≤ 9) 
  (h7 : a1 * a2 = 18) 
  (h8 : a3 * min a1 a2 = 28) : 
  a1 + a2 + a3 = 18 := 
sorry

end NUMINAMATH_GPT_sum_of_ages_l853_85389


namespace NUMINAMATH_GPT_isabella_euros_l853_85319

theorem isabella_euros (d : ℝ) : 
  (5 / 8) * d - 80 = 2 * d → d = 58 :=
by
  sorry

end NUMINAMATH_GPT_isabella_euros_l853_85319


namespace NUMINAMATH_GPT_evaluate_product_roots_of_unity_l853_85311

theorem evaluate_product_roots_of_unity :
  let w := Complex.exp (2 * Real.pi * Complex.I / 13)
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) =
  (3^12 + 3^11 + 3^10 + 3^9 + 3^8 + 3^7 + 3^6 + 3^5 + 3^4 + 3^3 + 3^2 + 3 + 1) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_product_roots_of_unity_l853_85311


namespace NUMINAMATH_GPT_calculate_f_at_2_l853_85306

def f (x : ℝ) : ℝ := 15 * x ^ 5 - 24 * x ^ 4 + 33 * x ^ 3 - 42 * x ^ 2 + 51 * x

theorem calculate_f_at_2 : f 2 = 294 := by
  sorry

end NUMINAMATH_GPT_calculate_f_at_2_l853_85306


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l853_85320

theorem isosceles_triangle_perimeter (a b : ℕ) (h_eq : a = 5 ∨ a = 9) (h_side : b = 9 ∨ b = 5) (h_neq : a ≠ b) : 
  (a + a + b = 19 ∨ a + a + b = 23) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l853_85320


namespace NUMINAMATH_GPT_percent_students_two_novels_l853_85353

theorem percent_students_two_novels :
  let total_students := 240
  let students_three_or_more := (1/6 : ℚ) * total_students
  let students_one := (5/12 : ℚ) * total_students
  let students_none := 16
  let students_two := total_students - students_three_or_more - students_one - students_none
  (students_two / total_students) * 100 = 35 := 
by
  sorry

end NUMINAMATH_GPT_percent_students_two_novels_l853_85353


namespace NUMINAMATH_GPT_kilos_of_bananas_l853_85381

-- Define the conditions
def initial_money := 500
def remaining_money := 426
def cost_per_kilo_potato := 2
def cost_per_kilo_tomato := 3
def cost_per_kilo_cucumber := 4
def cost_per_kilo_banana := 5
def kilos_potato := 6
def kilos_tomato := 9
def kilos_cucumber := 5

-- Total cost of potatoes, tomatoes, and cucumbers
def total_cost_vegetables : ℕ := 
  (kilos_potato * cost_per_kilo_potato) +
  (kilos_tomato * cost_per_kilo_tomato) +
  (kilos_cucumber * cost_per_kilo_cucumber)

-- Money spent on bananas
def money_spent_on_bananas : ℕ := initial_money - remaining_money - total_cost_vegetables

-- The proof problem statement
theorem kilos_of_bananas : money_spent_on_bananas / cost_per_kilo_banana = 14 :=
by
  -- The sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_kilos_of_bananas_l853_85381


namespace NUMINAMATH_GPT_salted_duck_eggs_min_cost_l853_85382

-- Define the system of equations and their solutions
def salted_duck_eggs_pricing (a b : ℕ) : Prop :=
  (9 * a + 6 * b = 390) ∧ (5 * a + 8 * b = 310)

-- Total number of boxes and constraints
def total_boxes_conditions (x y : ℕ) : Prop :=
  (x + y = 30) ∧ (x ≥ y + 5) ∧ (x ≤ 2 * y)

-- Minimize cost function given prices and constraints
def minimum_cost (x y a b : ℕ) : Prop :=
  (salted_duck_eggs_pricing a b) ∧
  (total_boxes_conditions x y) ∧
  (a = 30) ∧ (b = 20) ∧
  (10 * x + 600 = 780)

-- Statement to prove
theorem salted_duck_eggs_min_cost : ∃ x y : ℕ, minimum_cost x y 30 20 :=
by
  sorry

end NUMINAMATH_GPT_salted_duck_eggs_min_cost_l853_85382


namespace NUMINAMATH_GPT_train_travel_distance_l853_85348

theorem train_travel_distance (speed time: ℕ) (h1: speed = 85) (h2: time = 4) : speed * time = 340 :=
by
-- Given: speed = 85 km/hr and time = 4 hr
-- To prove: speed * time = 340
-- Since speed = 85 and time = 4, then 85 * 4 = 340
sorry

end NUMINAMATH_GPT_train_travel_distance_l853_85348


namespace NUMINAMATH_GPT_intermission_length_l853_85396

def concert_duration : ℕ := 80
def song_duration_total : ℕ := 70

theorem intermission_length : 
  concert_duration - song_duration_total = 10 :=
by
  -- conditions are already defined above
  sorry

end NUMINAMATH_GPT_intermission_length_l853_85396


namespace NUMINAMATH_GPT_clips_and_earnings_l853_85388

variable (x y z : ℝ)
variable (h_y : y = x / 2)
variable (totalClips : ℝ := 48 * x + y)
variable (avgEarning : ℝ := z / totalClips)

theorem clips_and_earnings :
  totalClips = 97 * x / 2 ∧ avgEarning = 2 * z / (97 * x) :=
by
  sorry

end NUMINAMATH_GPT_clips_and_earnings_l853_85388


namespace NUMINAMATH_GPT_tree_age_difference_l853_85315

theorem tree_age_difference
  (groups_rings : ℕ)
  (rings_per_group : ℕ)
  (first_tree_groups : ℕ)
  (second_tree_groups : ℕ)
  (rings_per_year : ℕ)
  (h_rg : rings_per_group = 6)
  (h_ftg : first_tree_groups = 70)
  (h_stg : second_tree_groups = 40)
  (h_rpy : rings_per_year = 1) :
  ((first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group)) = 180 := 
by
  sorry

end NUMINAMATH_GPT_tree_age_difference_l853_85315


namespace NUMINAMATH_GPT_yunkyung_work_per_day_l853_85374

theorem yunkyung_work_per_day (T : ℝ) (h : T > 0) (H : T / 3 = 1) : T / 3 = 1/3 := 
by sorry

end NUMINAMATH_GPT_yunkyung_work_per_day_l853_85374


namespace NUMINAMATH_GPT_four_digit_integers_with_repeated_digits_l853_85325

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end NUMINAMATH_GPT_four_digit_integers_with_repeated_digits_l853_85325


namespace NUMINAMATH_GPT_igor_reach_top_time_l853_85345

-- Define the conditions
def cabins_numbered_consecutively := (1, 99)
def igor_initial_cabin := 42
def first_aligned_cabin := 13
def second_aligned_cabin := 12
def alignment_time := 15
def total_cabins := 99
def expected_time := 17 * 60 + 15

-- State the problem as a theorem
theorem igor_reach_top_time :
  ∃ t, t = expected_time ∧
  -- Assume the cabins are numbered consecutively
  cabins_numbered_consecutively = (1, total_cabins) ∧
  -- Igor starts in cabin #42
  igor_initial_cabin = 42 ∧
  -- Cabin #42 first aligns with cabin #13, then aligns with cabin #12, 15 seconds later
  first_aligned_cabin = 13 ∧
  second_aligned_cabin = 12 ∧
  alignment_time = 15 :=
sorry

end NUMINAMATH_GPT_igor_reach_top_time_l853_85345


namespace NUMINAMATH_GPT_min_prime_factors_of_expression_l853_85303

theorem min_prime_factors_of_expression (m n : ℕ) : 
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ p1 ≠ p2 ∧ p1 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧ p2 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) := 
sorry

end NUMINAMATH_GPT_min_prime_factors_of_expression_l853_85303


namespace NUMINAMATH_GPT_sum_largest_and_second_smallest_l853_85392

-- Define the list of numbers
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Define a predicate to get the largest number
def is_largest (n : ℕ) : Prop := ∀ x ∈ numbers, x ≤ n

-- Define a predicate to get the second smallest number
def is_second_smallest (n : ℕ) : Prop :=
  ∃ a b, (a ∈ numbers ∧ b ∈ numbers ∧ a < b ∧ b < n ∧ ∀ x ∈ numbers, (x < a ∨ x > b))

-- The main goal: To prove that the sum of the largest number and the second smallest number is 25
theorem sum_largest_and_second_smallest : 
  ∃ l s, is_largest l ∧ is_second_smallest s ∧ l + s = 25 := 
sorry

end NUMINAMATH_GPT_sum_largest_and_second_smallest_l853_85392


namespace NUMINAMATH_GPT_number_of_candidates_l853_85340

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 132) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_candidates_l853_85340
