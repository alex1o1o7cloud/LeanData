import Mathlib

namespace NUMINAMATH_GPT_total_price_of_shoes_l847_84765

theorem total_price_of_shoes
  (S J : ℝ) 
  (h1 : 6 * S + 4 * J = 560) 
  (h2 : J = S / 4) :
  6 * S = 480 :=
by 
  -- Begin the proof environment
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_total_price_of_shoes_l847_84765


namespace NUMINAMATH_GPT_length_of_parallelepiped_l847_84791

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_parallelepiped_l847_84791


namespace NUMINAMATH_GPT_milk_exchange_l847_84738

theorem milk_exchange (initial_empty_bottles : ℕ) (exchange_rate : ℕ) (start_full_bottles : ℕ) : initial_empty_bottles = 43 → exchange_rate = 4 → start_full_bottles = 0 → ∃ liters_of_milk : ℕ, liters_of_milk = 14 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_milk_exchange_l847_84738


namespace NUMINAMATH_GPT_percent_children_with_both_colors_l847_84706

theorem percent_children_with_both_colors
  (F : ℕ) (C : ℕ) 
  (even_F : F % 2 = 0)
  (children_pick_two_flags : C = F / 2)
  (sixty_percent_blue : 6 * C / 10 = 6 * C / 10)
  (fifty_percent_red : 5 * C / 10 = 5 * C / 10)
  : (6 * C / 10) + (5 * C / 10) - C = C / 10 :=
by
  sorry

end NUMINAMATH_GPT_percent_children_with_both_colors_l847_84706


namespace NUMINAMATH_GPT_probability_red_then_green_l847_84792

-- Total number of balls and their representation
def total_balls : ℕ := 3
def red_balls : ℕ := 2
def green_balls : ℕ := 1

-- The total number of outcomes when drawing two balls with replacement
def total_outcomes : ℕ := total_balls * total_balls

-- The desired outcomes: drawing a red ball first and a green ball second
def desired_outcomes : ℕ := 2 -- (1,3) and (2,3)

-- Calculating the probability of drawing a red ball first and a green ball second
def probability_drawing_red_then_green : ℚ := desired_outcomes / total_outcomes

-- The theorem we need to prove
theorem probability_red_then_green :
  probability_drawing_red_then_green = 2 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_probability_red_then_green_l847_84792


namespace NUMINAMATH_GPT_range_of_x_for_a_range_of_a_l847_84733

-- Define propositions p and q
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (I)
theorem range_of_x_for_a (a x : ℝ) (ha : a = 1) (hpq : prop_p a x ∧ prop_q x) : 2 < x ∧ x < 3 :=
by
  sorry

-- Part (II)
theorem range_of_a (p q : ℝ → Prop) (hpq : ∀ x : ℝ, ¬p x → ¬q x) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_a_range_of_a_l847_84733


namespace NUMINAMATH_GPT_number_of_people_in_room_l847_84781

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end NUMINAMATH_GPT_number_of_people_in_room_l847_84781


namespace NUMINAMATH_GPT_negation_of_proposition_l847_84799

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x < 1) ↔ ∀ x : ℝ, x ≥ 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l847_84799


namespace NUMINAMATH_GPT_john_remaining_amount_l847_84701

theorem john_remaining_amount (initial_amount games: ℕ) (food souvenirs: ℕ) :
  initial_amount = 100 →
  games = 20 →
  food = 3 * games →
  souvenirs = (1 / 2 : ℚ) * games →
  initial_amount - (games + food + souvenirs) = 10 :=
by
  sorry

end NUMINAMATH_GPT_john_remaining_amount_l847_84701


namespace NUMINAMATH_GPT_find_fraction_value_l847_84736

variable (a b : ℝ)

theorem find_fraction_value (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 := 
  sorry

end NUMINAMATH_GPT_find_fraction_value_l847_84736


namespace NUMINAMATH_GPT_arithmetic_seq_product_of_first_two_terms_l847_84796

theorem arithmetic_seq_product_of_first_two_terms
    (a d : ℤ)
    (h1 : a + 4 * d = 17)
    (h2 : d = 2) :
    (a * (a + d) = 99) := 
by
    -- Proof to be done
    sorry

end NUMINAMATH_GPT_arithmetic_seq_product_of_first_two_terms_l847_84796


namespace NUMINAMATH_GPT_range_of_a_for_f_ge_a_l847_84725

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a_for_f_ge_a :
  (∀ x : ℝ, (-1 ≤ x → f x a ≥ a)) ↔ (-3 ≤ a ∧ a ≤ 1) :=
  sorry

end NUMINAMATH_GPT_range_of_a_for_f_ge_a_l847_84725


namespace NUMINAMATH_GPT_tangent_line_at_P_eq_2x_l847_84772

noncomputable def tangentLineEq (f : ℝ → ℝ) (P : ℝ × ℝ) : ℝ → ℝ :=
  let slope := deriv f P.1
  fun x => slope * (x - P.1) + P.2

theorem tangent_line_at_P_eq_2x : 
  ∀ (f : ℝ → ℝ) (x y : ℝ),
    f x = x^2 + 1 → 
    (x = 1) → (y = 2) →
    tangentLineEq f (x, y) x = 2 * x :=
by
  intros f x y f_eq hx hy
  sorry

end NUMINAMATH_GPT_tangent_line_at_P_eq_2x_l847_84772


namespace NUMINAMATH_GPT_arithmetic_mean_is_five_sixths_l847_84782

theorem arithmetic_mean_is_five_sixths :
  let a := 3 / 4
  let b := 5 / 6
  let c := 7 / 8
  (a + c) / 2 = b := sorry

end NUMINAMATH_GPT_arithmetic_mean_is_five_sixths_l847_84782


namespace NUMINAMATH_GPT_theta_in_fourth_quadrant_l847_84751

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan (θ + Real.pi / 4) = 1 / 3) : 
  (θ > 3 * Real.pi / 2) ∧ (θ < 2 * Real.pi) :=
sorry

end NUMINAMATH_GPT_theta_in_fourth_quadrant_l847_84751


namespace NUMINAMATH_GPT_mean_is_six_greater_than_median_l847_84760

theorem mean_is_six_greater_than_median (x a : ℕ) 
  (h1 : (x + a) + (x + 4) + (x + 7) + (x + 37) + x == 5 * (x + 10)) :
  a = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_mean_is_six_greater_than_median_l847_84760


namespace NUMINAMATH_GPT_sum_of_products_two_at_a_time_l847_84712

theorem sum_of_products_two_at_a_time
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 222)
  (h2 : a + b + c = 22) :
  a * b + b * c + c * a = 131 := 
sorry

end NUMINAMATH_GPT_sum_of_products_two_at_a_time_l847_84712


namespace NUMINAMATH_GPT_find_pink_highlighters_l847_84742

def yellow_highlighters : ℕ := 7
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 15

theorem find_pink_highlighters : (total_highlighters - (yellow_highlighters + blue_highlighters)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_pink_highlighters_l847_84742


namespace NUMINAMATH_GPT_fraction_subtraction_l847_84763

theorem fraction_subtraction :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l847_84763


namespace NUMINAMATH_GPT_carrot_price_l847_84715

variables (total_tomatoes : ℕ) (total_carrots : ℕ) (price_per_tomato : ℝ) (total_revenue : ℝ)

theorem carrot_price :
  total_tomatoes = 200 →
  total_carrots = 350 →
  price_per_tomato = 1 →
  total_revenue = 725 →
  (total_revenue - total_tomatoes * price_per_tomato) / total_carrots = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_carrot_price_l847_84715


namespace NUMINAMATH_GPT_exponent_multiplication_l847_84752

theorem exponent_multiplication :
  (-1 / 2 : ℝ) ^ 2022 * (2 : ℝ) ^ 2023 = 2 :=
by sorry

end NUMINAMATH_GPT_exponent_multiplication_l847_84752


namespace NUMINAMATH_GPT_jogging_track_circumference_l847_84773

theorem jogging_track_circumference 
  (deepak_speed : ℝ)
  (wife_speed : ℝ)
  (meeting_time : ℝ)
  (circumference : ℝ)
  (H1 : deepak_speed = 4.5)
  (H2 : wife_speed = 3.75)
  (H3 : meeting_time = 4.08) :
  circumference = 33.66 := sorry

end NUMINAMATH_GPT_jogging_track_circumference_l847_84773


namespace NUMINAMATH_GPT_total_increase_percentage_l847_84777

-- Define the conditions: original speed S, first increase by 30%, then another increase by 10%
def original_speed (S : ℝ) := S
def first_increase (S : ℝ) := S * 1.30
def second_increase (S : ℝ) := (S * 1.30) * 1.10

-- Prove that the total increase in speed is 43% of the original speed
theorem total_increase_percentage (S : ℝ) :
  (second_increase S - original_speed S) / original_speed S * 100 = 43 :=
by
  sorry

end NUMINAMATH_GPT_total_increase_percentage_l847_84777


namespace NUMINAMATH_GPT_sin_double_angle_of_tan_l847_84716

theorem sin_double_angle_of_tan (α : ℝ) (hα1 : Real.tan α = 2) (hα2 : 0 < α ∧ α < Real.pi / 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_of_tan_l847_84716


namespace NUMINAMATH_GPT_right_triangle_acute_angles_l847_84757

theorem right_triangle_acute_angles (a b : ℝ)
  (h_right_triangle : a + b = 90)
  (h_ratio : a / b = 3 / 2) :
  (a = 54) ∧ (b = 36) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angles_l847_84757


namespace NUMINAMATH_GPT_ratio_bee_eaters_leopards_l847_84721

variables (s f l c a t e r : ℕ)

-- Define the conditions from the problem.
def conditions : Prop :=
  s = 100 ∧
  f = 80 ∧
  l = 20 ∧
  c = s / 2 ∧
  a = 2 * (f + l) ∧
  t = 670 ∧
  e = t - (s + f + l + c + a)

-- The theorem statement proving the ratio.
theorem ratio_bee_eaters_leopards (h : conditions s f l c a t e) : r = (e / l) := by
  sorry

end NUMINAMATH_GPT_ratio_bee_eaters_leopards_l847_84721


namespace NUMINAMATH_GPT_sample_capacity_l847_84780

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ)
  (h1 : frequency = 30)
  (h2 : frequency_rate = 25 / 100) :
  n = 120 :=
by
  sorry

end NUMINAMATH_GPT_sample_capacity_l847_84780


namespace NUMINAMATH_GPT_tricycles_in_garage_l847_84798

theorem tricycles_in_garage 
    (T : ℕ) 
    (total_bicycles : ℕ := 3) 
    (total_unicycles : ℕ := 7) 
    (bicycle_wheels : ℕ := 2) 
    (tricycle_wheels : ℕ := 3) 
    (unicycle_wheels : ℕ := 1) 
    (total_wheels : ℕ := 25) 
    (eq_wheels : total_bicycles * bicycle_wheels + total_unicycles * unicycle_wheels + T * tricycle_wheels = total_wheels) :
    T = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_tricycles_in_garage_l847_84798


namespace NUMINAMATH_GPT_circumference_in_scientific_notation_l847_84771

noncomputable def circumference_m : ℝ := 4010000

noncomputable def scientific_notation (m: ℝ) : Prop :=
  m = 4.01 * 10^6

theorem circumference_in_scientific_notation : scientific_notation circumference_m :=
by
  sorry

end NUMINAMATH_GPT_circumference_in_scientific_notation_l847_84771


namespace NUMINAMATH_GPT_Susan_initial_amount_l847_84753

def initial_amount (S : ℝ) : Prop :=
  let Spent_in_September := (1/6) * S
  let Spent_in_October := (1/8) * S
  let Spent_in_November := 0.3 * S
  let Spent_in_December := 100
  let Remaining := 480
  S - (Spent_in_September + Spent_in_October + Spent_in_November + Spent_in_December) = Remaining

theorem Susan_initial_amount : ∃ S : ℝ, initial_amount S ∧ S = 1420 :=
by
  sorry

end NUMINAMATH_GPT_Susan_initial_amount_l847_84753


namespace NUMINAMATH_GPT_max_chords_through_line_l847_84710

noncomputable def maxChords (n : ℕ) : ℕ :=
  let k := n / 2
  k * k + n

theorem max_chords_through_line (points : ℕ) (h : points = 2017) : maxChords 2016 = 1018080 :=
by
  have h1 : (2016 / 2) * (2016 / 2) + 2016 = 1018080 := by norm_num
  rw [← h1]; sorry

end NUMINAMATH_GPT_max_chords_through_line_l847_84710


namespace NUMINAMATH_GPT_adjusted_retail_price_l847_84748

variable {a : ℝ} {m n : ℝ}

theorem adjusted_retail_price (h : 0 ≤ m ∧ 0 ≤ n) : (a * (1 + m / 100) * (n / 100)) = a * (1 + m / 100) * (n / 100) :=
by
  sorry

end NUMINAMATH_GPT_adjusted_retail_price_l847_84748


namespace NUMINAMATH_GPT_hyperbola_equation_l847_84776

theorem hyperbola_equation (c a b : ℝ) (ecc : ℝ) (h_c : c = 3) (h_ecc : ecc = 3 / 2) (h_a : a = 2) (h_b : b^2 = c^2 - a^2) :
    (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 4 - y^2 / 5 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l847_84776


namespace NUMINAMATH_GPT_percentage_in_excess_l847_84707

theorem percentage_in_excess 
  (A B : ℝ) (x : ℝ)
  (h1 : ∀ A',  A' = A * (1 + x / 100))
  (h2 : ∀ B',  B' = 0.94 * B)
  (h3 : ∀ A' B', A' * B' = A * B * (1 + 0.0058)) :
  x = 7 :=
by
  sorry

end NUMINAMATH_GPT_percentage_in_excess_l847_84707


namespace NUMINAMATH_GPT_part1_min_value_part2_find_b_part3_range_b_div_a_l847_84767

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - abs (a*x - b)

-- Part (1)
theorem part1_min_value : f 1 1 1 = -5/4 :=
by 
  sorry

-- Part (2)
theorem part2_find_b (b : ℝ) (h : b ≥ 2) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b) (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) : 
  b = 2 :=
by 
  sorry

-- Part (3)
theorem part3_range_b_div_a (a b : ℝ) (h_distinct : (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x a b = 1 ∧ ∀ y : ℝ, 0 < y ∧ y < 2 ∧ f y a b = 1 ∧ x ≠ y)) : 
  1 < b / a ∧ b / a < 2 :=
by 
  sorry

end NUMINAMATH_GPT_part1_min_value_part2_find_b_part3_range_b_div_a_l847_84767


namespace NUMINAMATH_GPT_exists_2016_integers_with_product_9_and_sum_0_l847_84724

theorem exists_2016_integers_with_product_9_and_sum_0 :
  ∃ (L : List ℤ), L.length = 2016 ∧ L.prod = 9 ∧ L.sum = 0 := by
  sorry

end NUMINAMATH_GPT_exists_2016_integers_with_product_9_and_sum_0_l847_84724


namespace NUMINAMATH_GPT_arithmetic_seq_a5_value_l847_84727

theorem arithmetic_seq_a5_value (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_a5_value_l847_84727


namespace NUMINAMATH_GPT_ratio_of_efficacy_l847_84754

-- Define original conditions
def original_sprigs_of_mint := 3
def green_tea_leaves_per_sprig := 2

-- Define new condition
def new_green_tea_leaves := 12

-- Calculate the number of sprigs of mint corresponding to the new green tea leaves in the new mud
def new_sprigs_of_mint := new_green_tea_leaves / green_tea_leaves_per_sprig

-- Statement of the theorem: ratio of the efficacy of new mud to original mud is 1:2
theorem ratio_of_efficacy : new_sprigs_of_mint = 2 * original_sprigs_of_mint :=
by
    sorry

end NUMINAMATH_GPT_ratio_of_efficacy_l847_84754


namespace NUMINAMATH_GPT_greatest_value_of_a_l847_84779

theorem greatest_value_of_a (a : ℝ) : a^2 - 12 * a + 32 ≤ 0 → a ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_a_l847_84779


namespace NUMINAMATH_GPT_find_red_peaches_l847_84730

def num_red_peaches (red yellow green : ℕ) : Prop :=
  (green = red + 1) ∧ yellow = 71 ∧ green = 8

theorem find_red_peaches (red : ℕ) :
  num_red_peaches red 71 8 → red = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_red_peaches_l847_84730


namespace NUMINAMATH_GPT_cat_collars_needed_l847_84790

-- Define the given constants
def nylon_per_dog_collar : ℕ := 18
def nylon_per_cat_collar : ℕ := 10
def total_nylon : ℕ := 192
def dog_collars : ℕ := 9

-- Compute the number of cat collars needed
theorem cat_collars_needed : (total_nylon - (dog_collars * nylon_per_dog_collar)) / nylon_per_cat_collar = 3 :=
by
  sorry

end NUMINAMATH_GPT_cat_collars_needed_l847_84790


namespace NUMINAMATH_GPT_quadratic_passes_through_point_l847_84728

theorem quadratic_passes_through_point (a b : ℝ) (h : a ≠ 0) (h₁ : ∃ y : ℝ, y = a * 1^2 + b * 1 - 1 ∧ y = 1) : a + b + 1 = 3 :=
by
  obtain ⟨y, hy1, hy2⟩ := h₁
  sorry

end NUMINAMATH_GPT_quadratic_passes_through_point_l847_84728


namespace NUMINAMATH_GPT_pears_weight_l847_84700

theorem pears_weight (x : ℕ) (h : 2 * x + 50 = 250) : x = 100 :=
sorry

end NUMINAMATH_GPT_pears_weight_l847_84700


namespace NUMINAMATH_GPT_Hulk_jump_more_than_500_l847_84768

theorem Hulk_jump_more_than_500 :
  ∀ n : ℕ, 2 * 3^(n - 1) > 500 → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_Hulk_jump_more_than_500_l847_84768


namespace NUMINAMATH_GPT_max_m_value_real_roots_interval_l847_84783

theorem max_m_value_real_roots_interval :
  (∃ x ∈ (Set.Icc 0 1), x^3 - 3 * x - m = 0) → m ≤ 0 :=
by
  sorry 

end NUMINAMATH_GPT_max_m_value_real_roots_interval_l847_84783


namespace NUMINAMATH_GPT_largest_common_value_lt_1000_l847_84718

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end NUMINAMATH_GPT_largest_common_value_lt_1000_l847_84718


namespace NUMINAMATH_GPT_annual_income_before_tax_l847_84702

theorem annual_income_before_tax (I : ℝ) (h1 : 0.42 * I - 0.28 * I = 4830) : I = 34500 :=
sorry

end NUMINAMATH_GPT_annual_income_before_tax_l847_84702


namespace NUMINAMATH_GPT_option_D_is_correct_l847_84770

variable (a b : ℝ)

theorem option_D_is_correct :
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^2 + 3 * a ≠ 4 * a^2) ∧
  ((a + 2) * (a - 2) ≠ a^2 - 2) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end NUMINAMATH_GPT_option_D_is_correct_l847_84770


namespace NUMINAMATH_GPT_solve_for_x_l847_84793

theorem solve_for_x (x : ℝ) (h : 3375 = (1 / 4) * x + 144) : x = 12924 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l847_84793


namespace NUMINAMATH_GPT_closest_integer_to_cbrt_250_l847_84762

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end NUMINAMATH_GPT_closest_integer_to_cbrt_250_l847_84762


namespace NUMINAMATH_GPT_relationship_between_x_y_l847_84749

theorem relationship_between_x_y (x y : ℝ) (h1 : x^2 - y^2 > 2 * x) (h2 : x * y < y) : x < y ∧ y < 0 := 
sorry

end NUMINAMATH_GPT_relationship_between_x_y_l847_84749


namespace NUMINAMATH_GPT_intersection_point_of_lines_l847_84745

theorem intersection_point_of_lines :
  let line1 (x : ℝ) := 3 * x - 4
  let line2 (x : ℝ) := - (1 / 3) * x + 5
  (∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ x = 2.7 ∧ y = 4.1) :=
by {
    sorry
}

end NUMINAMATH_GPT_intersection_point_of_lines_l847_84745


namespace NUMINAMATH_GPT_dwarf_diamond_distribution_l847_84711

-- Definitions for conditions
def dwarves : Type := Fin 8
structure State :=
  (diamonds : dwarves → ℕ)

-- Initial condition: Each dwarf has 3 diamonds
def initial_state : State := 
  { diamonds := fun _ => 3 }

-- Transition function: Each dwarf divides diamonds into two piles and passes them to neighbors
noncomputable def transition (s : State) : State := sorry

-- Proof goal: At a certain point in time, 3 specific dwarves have 24 diamonds in total,
-- with one dwarf having 7 diamonds, then prove the other two dwarves have 12 and 5 diamonds.
theorem dwarf_diamond_distribution (s : State)
  (h1 : ∃ t, s = (transition^[t]) initial_state ∧ ∃ i j k : dwarves, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    s.diamonds i + s.diamonds j + s.diamonds k = 24 ∧
    s.diamonds i = 7)
  : ∃ a b : dwarves, a ≠ b ∧ s.diamonds a = 12 ∧ s.diamonds b = 5 := sorry

end NUMINAMATH_GPT_dwarf_diamond_distribution_l847_84711


namespace NUMINAMATH_GPT_min_a2_plus_b2_l847_84758

theorem min_a2_plus_b2 (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_a2_plus_b2_l847_84758


namespace NUMINAMATH_GPT_max_min_values_of_f_l847_84778

-- Define the function f(x) and the conditions about its coefficients
def f (x : ℝ) (p q : ℝ) : ℝ := x^3 - p * x^2 - q * x

def intersects_x_axis_at_1 (p q : ℝ) : Prop :=
  f 1 p q = 0

-- Define the maximum and minimum values on the interval [-1, 1]
theorem max_min_values_of_f (p q : ℝ) 
  (h1 : f 1 p q = 0) :
  (p = 2) ∧ (q = -1) ∧ (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x 2 (-1) ≤ f (1/3) 2 (-1)) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-1) 2 (-1) ≤ f x 2 (-1)) :=
sorry

end NUMINAMATH_GPT_max_min_values_of_f_l847_84778


namespace NUMINAMATH_GPT_car_speeds_l847_84759

theorem car_speeds (d x : ℝ) (small_car_speed large_car_speed : ℝ) 
  (h1 : d = 135) 
  (h2 : small_car_speed = 5 * x) 
  (h3 : large_car_speed = 2 * x) 
  (h4 : 135 / small_car_speed + (4 + 0.5) = 135 / large_car_speed)
  : small_car_speed = 45 ∧ large_car_speed = 18 := by
  sorry

end NUMINAMATH_GPT_car_speeds_l847_84759


namespace NUMINAMATH_GPT_value_of_f_at_2_l847_84737

def f (x : ℤ) : ℤ := x^3 - x

theorem value_of_f_at_2 : f 2 = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l847_84737


namespace NUMINAMATH_GPT_union_of_A_and_B_l847_84713

-- Define set A
def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define set B
def B := {x : ℝ | x < 1}

-- The proof problem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} :=
by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l847_84713


namespace NUMINAMATH_GPT_sugar_percentage_l847_84714

theorem sugar_percentage 
  (initial_volume : ℝ) (initial_water_perc : ℝ) (initial_kola_perc: ℝ) (added_sugar : ℝ) (added_water : ℝ) (added_kola : ℝ)
  (initial_solution: initial_volume = 340) 
  (perc_water : initial_water_perc = 0.75) 
  (perc_kola: initial_kola_perc = 0.05)
  (added_sugar_amt : added_sugar = 3.2) 
  (added_water_amt : added_water = 12) 
  (added_kola_amt : added_kola = 6.8) : 
  (71.2 / 362) * 100 = 19.67 := 
by 
  sorry

end NUMINAMATH_GPT_sugar_percentage_l847_84714


namespace NUMINAMATH_GPT_parabola_focus_equals_hyperbola_focus_l847_84708

noncomputable def hyperbola_right_focus : (Float × Float) := (2, 0)

noncomputable def parabola_focus (p : Float) : (Float × Float) := (p / 2, 0)

theorem parabola_focus_equals_hyperbola_focus (p : Float) :
  parabola_focus p = hyperbola_right_focus → p = 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_focus_equals_hyperbola_focus_l847_84708


namespace NUMINAMATH_GPT_percentage_failed_hindi_l847_84756

theorem percentage_failed_hindi 
  (F_E F_B P_BE : ℕ) 
  (h₁ : F_E = 42) 
  (h₂ : F_B = 28) 
  (h₃ : P_BE = 56) :
  ∃ F_H, F_H = 30 := 
by
  sorry

end NUMINAMATH_GPT_percentage_failed_hindi_l847_84756


namespace NUMINAMATH_GPT_cubic_difference_l847_84720

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_difference_l847_84720


namespace NUMINAMATH_GPT_train_length_correct_l847_84794

noncomputable def length_bridge : ℝ := 300
noncomputable def time_to_cross : ℝ := 45
noncomputable def speed_train_kmh : ℝ := 44

-- Conversion from km/h to m/s
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

-- Total distance covered
noncomputable def total_distance_covered : ℝ := speed_train_ms * time_to_cross

-- Length of the train
noncomputable def length_train : ℝ := total_distance_covered - length_bridge

theorem train_length_correct : abs (length_train - 249.9) < 0.1 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l847_84794


namespace NUMINAMATH_GPT_problem1_problem2_l847_84769

/-- Proof statement for the first mathematical problem -/
theorem problem1 (x : ℝ) (h : (x - 2) ^ 2 = 9) : x = 5 ∨ x = -1 :=
by {
  -- Proof goes here
  sorry
}

/-- Proof statement for the second mathematical problem -/
theorem problem2 (x : ℝ) (h : 27 * (x + 1) ^ 3 + 8 = 0) : x = -5 / 3 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_l847_84769


namespace NUMINAMATH_GPT_find_pairs_l847_84785

theorem find_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔ ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) :=
by sorry

end NUMINAMATH_GPT_find_pairs_l847_84785


namespace NUMINAMATH_GPT_inverse_of_B_squared_l847_84764

theorem inverse_of_B_squared (B_inv : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B_inv = ![![3, -2], ![0, 5]]) : 
  (B_inv * B_inv) = ![![9, -16], ![0, 25]] :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_B_squared_l847_84764


namespace NUMINAMATH_GPT_tan_sub_pi_over_4_l847_84750

-- Define the conditions and the problem statement
variable (α : ℝ) (h : Real.tan α = 2)

-- State the problem as a theorem
theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_sub_pi_over_4_l847_84750


namespace NUMINAMATH_GPT_sum_of_youngest_and_oldest_nephews_l847_84775

theorem sum_of_youngest_and_oldest_nephews 
    (n1 n2 n3 n4 n5 n6 : ℕ) 
    (mean_eq : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = 10) 
    (median_eq : (n3 + n4) / 2 = 12) : 
    n1 + n6 = 12 := 
by 
    sorry

end NUMINAMATH_GPT_sum_of_youngest_and_oldest_nephews_l847_84775


namespace NUMINAMATH_GPT_dilation_transformation_result_l847_84747

theorem dilation_transformation_result
  (x y x' y' : ℝ)
  (h₀ : x'^2 / 4 + y'^2 / 9 = 1) 
  (h₁ : x' = 2 * x)
  (h₂ : y' = 3 * y)
  (h₃ : x^2 + y^2 = 1)
  : x'^2 / 4 + y'^2 / 9 = 1 := 
by
  sorry

end NUMINAMATH_GPT_dilation_transformation_result_l847_84747


namespace NUMINAMATH_GPT_cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l847_84734

-- Problem Part 1
theorem cos_alpha_implies_sin_alpha (alpha : ℝ) (h1 : Real.cos alpha = -4/5) (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin alpha = -3/5 := sorry

-- Problem Part 2
theorem tan_theta_implies_expr (theta : ℝ) (h1 : Real.tan theta = 3) : 
  (Real.sin theta + Real.cos theta) / (2 * Real.sin theta + Real.cos theta) = 4 / 7 := sorry

end NUMINAMATH_GPT_cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l847_84734


namespace NUMINAMATH_GPT_sum_lent_is_300_l847_84786

-- Define the conditions
def interest_rate : ℕ := 4
def time_period : ℕ := 8
def interest_amounted_less : ℕ := 204

-- Prove that the sum lent P is 300 given the conditions
theorem sum_lent_is_300 (P : ℕ) : 
  (P * interest_rate * time_period / 100 = P - interest_amounted_less) -> P = 300 := by
  sorry

end NUMINAMATH_GPT_sum_lent_is_300_l847_84786


namespace NUMINAMATH_GPT_Henry_age_l847_84735

-- Define the main proof statement
theorem Henry_age (h s : ℕ) 
(h1 : h + 8 = 3 * (s - 1))
(h2 : (h - 25) + (s - 25) = 83) : h = 97 :=
by
  sorry

end NUMINAMATH_GPT_Henry_age_l847_84735


namespace NUMINAMATH_GPT_first_divisor_is_six_l847_84746

theorem first_divisor_is_six {d : ℕ} 
  (h1: (1394 - 14) % d = 0)
  (h2: (2535 - 1929) % d = 0)
  (h3: (40 - 34) % d = 0)
  : d = 6 :=
sorry

end NUMINAMATH_GPT_first_divisor_is_six_l847_84746


namespace NUMINAMATH_GPT_sin_minus_cos_eq_l847_84719

variable {α : ℝ} (h₁ : 0 < α ∧ α < π) (h₂ : Real.sin α + Real.cos α = 1/3)

theorem sin_minus_cos_eq : Real.sin α - Real.cos α = Real.sqrt 17 / 3 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_l847_84719


namespace NUMINAMATH_GPT_sufficient_condition_l847_84744

theorem sufficient_condition (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l847_84744


namespace NUMINAMATH_GPT_discount_percentage_l847_84704

variable (P : ℝ)  -- Original price of the car
variable (D : ℝ)  -- Discount percentage in decimal form
variable (S : ℝ)  -- Selling price of the car

theorem discount_percentage
  (h1 : S = P * (1 - D) * 1.70)
  (h2 : S = P * 1.1899999999999999) :
  D = 0.3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_discount_percentage_l847_84704


namespace NUMINAMATH_GPT_ratio_difference_l847_84787

theorem ratio_difference (x : ℕ) (h_largest : 7 * x = 70) : 70 - 3 * x = 40 := by
  sorry

end NUMINAMATH_GPT_ratio_difference_l847_84787


namespace NUMINAMATH_GPT_product_of_total_points_l847_84740

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def Allie_rolls : List ℕ := [3, 5, 6, 2, 4]
def Betty_rolls : List ℕ := [3, 2, 1, 6, 4]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem product_of_total_points :
  total_points Allie_rolls * total_points Betty_rolls = 256 :=
by
  sorry

end NUMINAMATH_GPT_product_of_total_points_l847_84740


namespace NUMINAMATH_GPT_regular_polygon_sides_l847_84755

theorem regular_polygon_sides (n : ℕ) (h : (180 * (n - 2) = 135 * n)) : n = 8 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l847_84755


namespace NUMINAMATH_GPT_find_11th_place_l847_84732

def placement_problem (Amara Bindu Carlos Devi Eshan Farel: ℕ): Prop :=
  (Carlos + 5 = Amara) ∧
  (Bindu = Eshan + 3) ∧
  (Carlos = Devi + 2) ∧
  (Devi = 6) ∧
  (Eshan + 1 = Farel) ∧
  (Bindu + 4 = Amara) ∧
  (Farel = 9)

theorem find_11th_place (Amara Bindu Carlos Devi Eshan Farel: ℕ) 
  (h : placement_problem Amara Bindu Carlos Devi Eshan Farel) : 
  Eshan = 11 := 
sorry

end NUMINAMATH_GPT_find_11th_place_l847_84732


namespace NUMINAMATH_GPT_volume_increase_l847_84703

theorem volume_increase (l w h: ℕ) 
(h1: l * w * h = 4320) 
(h2: l * w + w * h + h * l = 852) 
(h3: l + w + h = 52) : 
(l + 1) * (w + 1) * (h + 1) = 5225 := 
by 
  sorry

end NUMINAMATH_GPT_volume_increase_l847_84703


namespace NUMINAMATH_GPT_roots_product_l847_84766

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end NUMINAMATH_GPT_roots_product_l847_84766


namespace NUMINAMATH_GPT_binomial_sum_eval_l847_84731

theorem binomial_sum_eval :
  (Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 5)) +
  (Nat.factorial 6 / (Nat.factorial 4 * Nat.factorial 2)) = 36 := by
sorry

end NUMINAMATH_GPT_binomial_sum_eval_l847_84731


namespace NUMINAMATH_GPT_mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l847_84705

def molar_mass_carbon : Float := 12.01
def molar_mass_hydrogen : Float := 1.008
def moles_of_nonane : Float := 23.0
def num_carbons_in_nonane : Float := 9.0
def num_hydrogens_in_nonane : Float := 20.0

theorem mass_of_23_moles_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  mass_23_moles = 2950.75 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  have molar_mass_C9H20_val : molar_mass_C9H20 = 128.25 := sorry
  have mass_23_moles_val : mass_23_moles = 2950.75 := sorry
  exact mass_23_moles_val

theorem percentage_composition_C_H_O_in_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  percentage_carbon = 84.27 ∧ percentage_hydrogen = 15.73 ∧ percentage_oxygen = 0 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  have percentage_carbon_val : percentage_carbon = 84.27 := sorry
  have percentage_hydrogen_val : percentage_hydrogen = 15.73 := sorry
  have percentage_oxygen_val : percentage_oxygen = 0 := by rfl
  exact ⟨percentage_carbon_val, percentage_hydrogen_val, percentage_oxygen_val⟩

end NUMINAMATH_GPT_mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l847_84705


namespace NUMINAMATH_GPT_evaluated_result_l847_84761

noncomputable def evaluate_expression (y : ℝ) (hy : y ≠ 0) : ℝ :=
  (18 * y^3) * (4 * y^2) * (1 / (2 * y)^3)

theorem evaluated_result (y : ℝ) (hy : y ≠ 0) : evaluate_expression y hy = 9 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_evaluated_result_l847_84761


namespace NUMINAMATH_GPT_distinct_arrangements_l847_84784

-- Define the conditions: 7 books, 3 are identical
def total_books : ℕ := 7
def identical_books : ℕ := 3

-- Statement that the number of distinct arrangements is 840
theorem distinct_arrangements : (Nat.factorial total_books) / (Nat.factorial identical_books) = 840 := 
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_l847_84784


namespace NUMINAMATH_GPT_total_interest_l847_84722

def P : ℝ := 1000
def r : ℝ := 0.1
def n : ℕ := 3

theorem total_interest : (P * (1 + r)^n) - P = 331 := by
  sorry

end NUMINAMATH_GPT_total_interest_l847_84722


namespace NUMINAMATH_GPT_maria_should_buy_more_l847_84743

-- Define the conditions as assumptions.
variables (needs total_cartons : ℕ) (strawberries blueberries : ℕ)

-- Specify the given conditions.
def maria_conditions (needs total_cartons strawberries blueberries : ℕ) : Prop :=
  needs = 21 ∧ strawberries = 4 ∧ blueberries = 8 ∧ total_cartons = strawberries + blueberries

-- State the theorem to be proven.
theorem maria_should_buy_more
  (needs total_cartons : ℕ) (strawberries blueberries : ℕ)
  (h : maria_conditions needs total_cartons strawberries blueberries) :
  needs - total_cartons = 9 :=
sorry

end NUMINAMATH_GPT_maria_should_buy_more_l847_84743


namespace NUMINAMATH_GPT_angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l847_84789

-- Problem part (a)
theorem angles_in_arithmetic_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (arithmetic_progression : ∃ (d : ℝ) (α : ℝ), β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0):
  (∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0) :=
sorry

-- Problem part (b)
theorem angles_not_in_geometric_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (geometric_progression : ∃ (r : ℝ) (α : ℝ), β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1 ∧ r > 0):
  ¬(∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1) :=
sorry

end NUMINAMATH_GPT_angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l847_84789


namespace NUMINAMATH_GPT_proof_a_l847_84795

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (y - 3) / (x - 2) = 3}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ a * x + 2 * y + a = 0}

-- Given conditions that M ∩ N = ∅, prove that a = -6 or a = -2
theorem proof_a (h : ∃ a : ℝ, (N a ∩ M = ∅)) : ∃ a : ℝ, a = -6 ∨ a = -2 :=
  sorry

end NUMINAMATH_GPT_proof_a_l847_84795


namespace NUMINAMATH_GPT_cheryl_material_used_l847_84709

noncomputable def total_material_needed : ℚ :=
  (5 / 11) + (2 / 3)

noncomputable def material_left : ℚ :=
  25 / 55

noncomputable def material_used : ℚ :=
  total_material_needed - material_left

theorem cheryl_material_used :
  material_used = 22 / 33 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_material_used_l847_84709


namespace NUMINAMATH_GPT_negative_expression_P_minus_Q_l847_84741

theorem negative_expression_P_minus_Q :
  ∀ (P Q R S T : ℝ), 
    P = -4.0 → 
    Q = -2.0 → 
    R = 0.2 → 
    S = 1.1 → 
    T = 1.7 → 
    P - Q < 0 := 
by 
  intros P Q R S T hP hQ hR hS hT
  rw [hP, hQ]
  sorry

end NUMINAMATH_GPT_negative_expression_P_minus_Q_l847_84741


namespace NUMINAMATH_GPT_remainder_of_55_pow_55_plus_15_mod_8_l847_84723

theorem remainder_of_55_pow_55_plus_15_mod_8 :
  (55^55 + 15) % 8 = 6 := by
  -- This statement does not include any solution steps.
  sorry

end NUMINAMATH_GPT_remainder_of_55_pow_55_plus_15_mod_8_l847_84723


namespace NUMINAMATH_GPT_rolls_for_mode_of_two_l847_84717

theorem rolls_for_mode_of_two (n : ℕ) (p : ℚ := 1/6) (m0 : ℕ := 32) : 
  (n : ℚ) * p - (1 - p) ≤ m0 ∧ m0 ≤ (n : ℚ) * p + p ↔ 191 ≤ n ∧ n ≤ 197 := 
by
  sorry

end NUMINAMATH_GPT_rolls_for_mode_of_two_l847_84717


namespace NUMINAMATH_GPT_connie_total_markers_l847_84774

theorem connie_total_markers :
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  red_markers + blue_markers + green_markers + purple_markers = 15225 :=
by
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  -- Proof would go here, but we use sorry to skip it for now
  sorry

end NUMINAMATH_GPT_connie_total_markers_l847_84774


namespace NUMINAMATH_GPT_second_number_is_12_l847_84797

noncomputable def expression := (26.3 * 12 * 20) / 3 + 125

theorem second_number_is_12 :
  expression = 2229 → 12 = 12 :=
by sorry

end NUMINAMATH_GPT_second_number_is_12_l847_84797


namespace NUMINAMATH_GPT_original_number_is_fraction_l847_84726

theorem original_number_is_fraction (x : ℚ) (h : 1 + 1/x = 7/3) : x = 3/4 :=
sorry

end NUMINAMATH_GPT_original_number_is_fraction_l847_84726


namespace NUMINAMATH_GPT_inequality_one_solution_inequality_two_solution_cases_l847_84739

-- Setting up the problem for the first inequality
theorem inequality_one_solution :
  {x : ℝ | -1 ≤ x ∧ x ≤ 4} = {x : ℝ |  -x ^ 2 + 3 * x + 4 ≥ 0} :=
sorry

-- Setting up the problem for the second inequality with different cases of 'a'
theorem inequality_two_solution_cases (a : ℝ) :
  (a = 0 ∧ {x : ℝ | true} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a > 0 ∧ {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a < 0 ∧ {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0}) :=
sorry

end NUMINAMATH_GPT_inequality_one_solution_inequality_two_solution_cases_l847_84739


namespace NUMINAMATH_GPT_proof_eq1_proof_eq2_l847_84788

variable (x : ℝ)

-- Proof problem for Equation (1)
theorem proof_eq1 (h : (1 - x) / 3 - 2 = x / 6) : x = -10 / 3 := sorry

-- Proof problem for Equation (2)
theorem proof_eq2 (h : (x + 1) / 0.25 - (x - 2) / 0.5 = 5) : x = -3 / 2 := sorry

end NUMINAMATH_GPT_proof_eq1_proof_eq2_l847_84788


namespace NUMINAMATH_GPT_ratio_of_cows_sold_l847_84729

-- Condition 1: The farmer originally has 51 cows.
def original_cows : ℕ := 51

-- Condition 2: The farmer adds 5 new cows to the herd.
def new_cows : ℕ := 5

-- Condition 3: The farmer has 42 cows left after selling a portion of the herd.
def remaining_cows : ℕ := 42

-- Defining total cows after adding new cows
def total_cows_after_addition : ℕ := original_cows + new_cows

-- Defining cows sold
def cows_sold : ℕ := total_cows_after_addition - remaining_cows

-- The theorem states the ratio of 'cows sold' to 'total cows after addition' is 1 : 4
theorem ratio_of_cows_sold : (cows_sold : ℚ) / (total_cows_after_addition : ℚ) = 1 / 4 := by
  -- Proof would go here
  sorry


end NUMINAMATH_GPT_ratio_of_cows_sold_l847_84729
