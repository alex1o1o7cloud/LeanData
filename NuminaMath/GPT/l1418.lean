import Mathlib

namespace NUMINAMATH_GPT_minimum_degree_q_l1418_141888

variable (p q r : Polynomial ℝ)

theorem minimum_degree_q (h1 : 2 * p + 5 * q = r)
                        (hp : p.degree = 7)
                        (hr : r.degree = 10) :
  q.degree = 10 :=
sorry

end NUMINAMATH_GPT_minimum_degree_q_l1418_141888


namespace NUMINAMATH_GPT_piecewise_function_not_composed_of_multiple_functions_l1418_141811

theorem piecewise_function_not_composed_of_multiple_functions :
  ∀ (f : ℝ → ℝ), (∃ (I : ℝ → Prop) (f₁ f₂ : ℝ → ℝ),
    (∀ x, I x → f x = f₁ x) ∧ (∀ x, ¬I x → f x = f₂ x)) →
    ¬(∃ (g₁ g₂ : ℝ → ℝ), (∀ x, f x = g₁ x ∨ f x = g₂ x)) :=
by
  sorry

end NUMINAMATH_GPT_piecewise_function_not_composed_of_multiple_functions_l1418_141811


namespace NUMINAMATH_GPT_total_questions_attempted_l1418_141822

theorem total_questions_attempted (C W T : ℕ) 
    (hC : C = 36) 
    (hScore : 120 = (4 * C) - W) 
    (hT : T = C + W) : 
    T = 60 := 
by 
  sorry

end NUMINAMATH_GPT_total_questions_attempted_l1418_141822


namespace NUMINAMATH_GPT_percentage_of_315_out_of_900_is_35_l1418_141882

theorem percentage_of_315_out_of_900_is_35 :
  (315 : ℝ) / 900 * 100 = 35 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_315_out_of_900_is_35_l1418_141882


namespace NUMINAMATH_GPT_prime_factorization_675_l1418_141802

theorem prime_factorization_675 :
  ∃ (n h : ℕ), n > 1 ∧ n = 3 ∧ h = 225 ∧ 675 = (3^3) * (5^2) :=
by
  sorry

end NUMINAMATH_GPT_prime_factorization_675_l1418_141802


namespace NUMINAMATH_GPT_calculate_taxi_fare_l1418_141804

theorem calculate_taxi_fare :
  ∀ (f_80 f_120: ℝ), f_80 = 160 ∧ f_80 = 20 + (80 * (140/80)) →
                      f_120 = 20 + (120 * (140/80)) →
                      f_120 = 230 :=
by
  intro f_80 f_120
  rintro ⟨h80, h_proportional⟩ h_120
  sorry

end NUMINAMATH_GPT_calculate_taxi_fare_l1418_141804


namespace NUMINAMATH_GPT_maria_fraction_of_remaining_distance_l1418_141853

theorem maria_fraction_of_remaining_distance (total_distance remaining_distance distance_travelled : ℕ) 
(h_total : total_distance = 480) 
(h_first_stop : distance_travelled = total_distance / 2) 
(h_remaining : remaining_distance = total_distance - distance_travelled)
(h_final_leg : remaining_distance - distance_travelled = 180) : 
(distance_travelled / remaining_distance) = (1 / 4) := 
by
  sorry

end NUMINAMATH_GPT_maria_fraction_of_remaining_distance_l1418_141853


namespace NUMINAMATH_GPT_mooney_ate_correct_l1418_141814

-- Define initial conditions
def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mother_added : ℕ := 24
def final_brownies : ℕ := 36

-- Define Mooney ate some brownies
variable (mooney_ate : ℕ)

-- Prove that Mooney ate 4 brownies
theorem mooney_ate_correct :
  (initial_brownies - father_ate) - mooney_ate + mother_added = final_brownies →
  mooney_ate = 4 :=
by
  sorry

end NUMINAMATH_GPT_mooney_ate_correct_l1418_141814


namespace NUMINAMATH_GPT_exists_xyz_prime_expression_l1418_141854

theorem exists_xyz_prime_expression (a b c p : ℤ) (h_prime : Prime p)
    (h_div : p ∣ (a^2 + b^2 + c^2 - ab - bc - ca))
    (h_gcd : Int.gcd p ((a^2 + b^2 + c^2 - ab - bc - ca) / p) = 1) :
    ∃ x y z : ℤ, p = x^2 + y^2 + z^2 - xy - yz - zx := by
  sorry

end NUMINAMATH_GPT_exists_xyz_prime_expression_l1418_141854


namespace NUMINAMATH_GPT_original_price_of_tennis_racket_l1418_141825

theorem original_price_of_tennis_racket
  (sneaker_cost : ℝ) (outfit_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (price_of_tennis_racket : ℝ) :
  sneaker_cost = 200 → 
  outfit_cost = 250 → 
  discount_rate = 0.20 → 
  total_spent = 750 → 
  price_of_tennis_racket = 289.77 :=
by
  intros hs ho hd ht
  have ht := ht.symm   -- To rearrange the equation
  sorry

end NUMINAMATH_GPT_original_price_of_tennis_racket_l1418_141825


namespace NUMINAMATH_GPT_OM_geq_ON_l1418_141891

variables {A B C D E F G H P Q M N O : Type*}

-- Definitions for geometrical concepts
def is_intersection_of_diagonals (M : Type*) (A B C D : Type*) : Prop :=
-- M is the intersection of the diagonals AC and BD
sorry

def is_intersection_of_midlines (N : Type*) (A B C D : Type*) : Prop :=
-- N is the intersection of the midlines connecting the midpoints of opposite sides
sorry

def is_center_of_circumscribed_circle (O : Type*) (A B C D : Type*) : Prop :=
-- O is the center of the circumscribed circle around quadrilateral ABCD
sorry

-- Proof problem
theorem OM_geq_ON (A B C D M N O : Type*) 
  (hm : is_intersection_of_diagonals M A B C D)
  (hn : is_intersection_of_midlines N A B C D)
  (ho : is_center_of_circumscribed_circle O A B C D) : 
  ∃ (OM ON : ℝ), OM ≥ ON :=
sorry

end NUMINAMATH_GPT_OM_geq_ON_l1418_141891


namespace NUMINAMATH_GPT_directrix_of_parabola_l1418_141829

theorem directrix_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 4 * x^2 - 6) : 
    ∃ d, (∀ x, y x = 4 * x^2 - 6) ∧ d = -97/16 ↔ (y (-6 - d)) = -10 := 
    sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1418_141829


namespace NUMINAMATH_GPT_jacket_cost_is_30_l1418_141808

-- Let's define the given conditions
def num_dresses := 5
def cost_per_dress := 20 -- dollars
def num_pants := 3
def cost_per_pant := 12 -- dollars
def num_jackets := 4
def transport_cost := 5 -- dollars
def initial_amount := 400 -- dollars
def remaining_amount := 139 -- dollars

-- Define the cost per jacket
def cost_per_jacket := 30 -- dollars

-- Final theorem statement to be proved
theorem jacket_cost_is_30:
  num_dresses * cost_per_dress + num_pants * cost_per_pant + num_jackets * cost_per_jacket + transport_cost = initial_amount - remaining_amount :=
sorry

end NUMINAMATH_GPT_jacket_cost_is_30_l1418_141808


namespace NUMINAMATH_GPT_smallest_sum_l1418_141839

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end NUMINAMATH_GPT_smallest_sum_l1418_141839


namespace NUMINAMATH_GPT_binomial_sum_eq_728_l1418_141883

theorem binomial_sum_eq_728 :
  (Nat.choose 6 1) * 2^1 +
  (Nat.choose 6 2) * 2^2 +
  (Nat.choose 6 3) * 2^3 +
  (Nat.choose 6 4) * 2^4 +
  (Nat.choose 6 5) * 2^5 +
  (Nat.choose 6 6) * 2^6 = 728 :=
by
  sorry

end NUMINAMATH_GPT_binomial_sum_eq_728_l1418_141883


namespace NUMINAMATH_GPT_find_b_l1418_141859

-- Given conditions
def varies_inversely (a b : ℝ) := ∃ K : ℝ, K = a * b
def constant_a (a : ℝ) := a = 1500
def constant_b (b : ℝ) := b = 0.25

-- The theorem to prove
theorem find_b (a : ℝ) (b : ℝ) (h_inv: varies_inversely a b)
  (h_a: constant_a a) (h_b: constant_b b): b = 0.125 := 
sorry

end NUMINAMATH_GPT_find_b_l1418_141859


namespace NUMINAMATH_GPT_toys_of_Jason_l1418_141862

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end NUMINAMATH_GPT_toys_of_Jason_l1418_141862


namespace NUMINAMATH_GPT_sean_needs_six_packs_l1418_141821

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end NUMINAMATH_GPT_sean_needs_six_packs_l1418_141821


namespace NUMINAMATH_GPT_sum_first_five_terms_eq_15_l1418_141875

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d 

variable (a : ℕ → ℝ) (h_arith_seq : is_arithmetic_sequence a) (h_a3 : a 3 = 3)

theorem sum_first_five_terms_eq_15 : (a 1 + a 2 + a 3 + a 4 + a 5 = 15) :=
sorry

end NUMINAMATH_GPT_sum_first_five_terms_eq_15_l1418_141875


namespace NUMINAMATH_GPT_range_of_m_l1418_141887

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = 3) (h3 : x + y > 0) : m > -4 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1418_141887


namespace NUMINAMATH_GPT_juanita_sunscreen_cost_l1418_141871

theorem juanita_sunscreen_cost:
  let bottles_per_month := 1
  let months_in_year := 12
  let cost_per_bottle := 30.0
  let discount_rate := 0.30
  let total_bottles := bottles_per_month * months_in_year
  let total_cost_before_discount := total_bottles * cost_per_bottle
  let discount_amount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  total_cost_after_discount = 252.00 := 
by
  sorry

end NUMINAMATH_GPT_juanita_sunscreen_cost_l1418_141871


namespace NUMINAMATH_GPT_middle_integer_is_six_l1418_141827

def valid_even_integer (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = x ∧ x = n - 2 ∧ y = n ∧ z = n + 2 ∧ x < y ∧ y < z ∧
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9

theorem middle_integer_is_six (n : ℕ) (h : valid_even_integer n) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_middle_integer_is_six_l1418_141827


namespace NUMINAMATH_GPT_three_digit_numbers_subtract_297_l1418_141849

theorem three_digit_numbers_subtract_297:
  (∃ (p q r : ℕ), 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ 0 ≤ r ∧ r ≤ 9 ∧ (100 * p + 10 * q + r - 297 = 100 * r + 10 * q + p)) →
  (num_valid_three_digit_numbers = 60) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_subtract_297_l1418_141849


namespace NUMINAMATH_GPT_riddles_ratio_l1418_141860

theorem riddles_ratio (Josh_riddles : ℕ) (Ivory_riddles : ℕ) (Taso_riddles : ℕ) 
  (h1 : Josh_riddles = 8) 
  (h2 : Ivory_riddles = Josh_riddles + 4) 
  (h3 : Taso_riddles = 24) : 
  Taso_riddles / Ivory_riddles = 2 := 
by sorry

end NUMINAMATH_GPT_riddles_ratio_l1418_141860


namespace NUMINAMATH_GPT_students_who_like_both_apple_pie_and_chocolate_cake_l1418_141840

def total_students := 50
def students_who_like_apple_pie := 22
def students_who_like_chocolate_cake := 20
def students_who_like_neither := 10
def students_who_like_only_cookies := 5

theorem students_who_like_both_apple_pie_and_chocolate_cake :
  (students_who_like_apple_pie + students_who_like_chocolate_cake - (total_students - students_who_like_neither - students_who_like_only_cookies)) = 7 := 
by
  sorry

end NUMINAMATH_GPT_students_who_like_both_apple_pie_and_chocolate_cake_l1418_141840


namespace NUMINAMATH_GPT_simplify_expression_correct_l1418_141865

noncomputable def simplify_expression : Prop :=
  (1 / (Real.log 3 / Real.log 6 + 1) + 1 / (Real.log 7 / Real.log 15 + 1) + 1 / (Real.log 4 / Real.log 12 + 1)) = -Real.log 84 / Real.log 10

theorem simplify_expression_correct : simplify_expression :=
  by
    sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1418_141865


namespace NUMINAMATH_GPT_units_digit_of_45_pow_125_plus_7_pow_87_l1418_141818

theorem units_digit_of_45_pow_125_plus_7_pow_87 :
  (45 ^ 125 + 7 ^ 87) % 10 = 8 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_units_digit_of_45_pow_125_plus_7_pow_87_l1418_141818


namespace NUMINAMATH_GPT_car_speed_l1418_141864

theorem car_speed (rev_per_min : ℕ) (circ : ℝ) (h_rev : rev_per_min = 400) (h_circ : circ = 5) : 
  (rev_per_min * circ) * 60 / 1000 = 120 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l1418_141864


namespace NUMINAMATH_GPT_value_added_to_each_number_is_11_l1418_141832

-- Given definitions and conditions
def initial_average : ℝ := 40
def number_count : ℕ := 15
def new_average : ℝ := 51

-- Mathematically equivalent proof statement
theorem value_added_to_each_number_is_11 (x : ℝ) 
  (h1 : number_count * initial_average = 600)
  (h2 : (600 + number_count * x) / number_count = new_average) : 
  x = 11 := 
by 
  sorry

end NUMINAMATH_GPT_value_added_to_each_number_is_11_l1418_141832


namespace NUMINAMATH_GPT_p_2015_coordinates_l1418_141873

namespace AaronWalk

def position (n : ℕ) : ℤ × ℤ :=
sorry

theorem p_2015_coordinates : position 2015 = (22, 57) := 
sorry

end AaronWalk

end NUMINAMATH_GPT_p_2015_coordinates_l1418_141873


namespace NUMINAMATH_GPT_sum_a_b_l1418_141823

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 5 * b = 47) (h2 : 4 * a + 2 * b = 38) : a + b = 85 / 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_l1418_141823


namespace NUMINAMATH_GPT_andrew_kept_stickers_l1418_141889

theorem andrew_kept_stickers :
  ∃ (b d f e g h : ℕ), b = 2000 ∧ d = (5 * b) / 100 ∧ f = d + 120 ∧ e = (d + f) / 2 ∧ g = 80 ∧ h = (e + g) / 5 ∧ (b - (d + f + e + g + h) = 1392) :=
sorry

end NUMINAMATH_GPT_andrew_kept_stickers_l1418_141889


namespace NUMINAMATH_GPT_balance_weights_l1418_141834

def pair_sum {α : Type*} (l : List α) [Add α] : List (α × α) :=
  l.zip l.tail

theorem balance_weights (w : Fin 100 → ℝ) (h : ∀ i j, |w i - w j| ≤ 20) :
  ∃ (l r : Finset (Fin 100)), l.card = 50 ∧ r.card = 50 ∧
  |(l.sum w - r.sum w)| ≤ 20 :=
sorry

end NUMINAMATH_GPT_balance_weights_l1418_141834


namespace NUMINAMATH_GPT_probability_of_shaded_triangle_l1418_141893

theorem probability_of_shaded_triangle 
  (triangles : Finset ℝ) 
  (shaded_triangles : Finset ℝ)
  (h1 : triangles = {1, 2, 3, 4, 5})
  (h2 : shaded_triangles = {1, 4})
  : (shaded_triangles.card / triangles.card) = 2 / 5 := 
  by
  sorry

end NUMINAMATH_GPT_probability_of_shaded_triangle_l1418_141893


namespace NUMINAMATH_GPT_greatest_sum_l1418_141880

theorem greatest_sum {x y : ℤ} (h₁ : x^2 + y^2 = 49) : x + y ≤ 9 :=
sorry

end NUMINAMATH_GPT_greatest_sum_l1418_141880


namespace NUMINAMATH_GPT_steve_annual_salary_l1418_141847

variable (S : ℝ)

theorem steve_annual_salary :
  (0.70 * S - 800 = 27200) → (S = 40000) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_steve_annual_salary_l1418_141847


namespace NUMINAMATH_GPT_find_k_l1418_141801

-- Definitions of conditions
def equation1 (x k : ℝ) : Prop := x^2 + k*x + 10 = 0
def equation2 (x k : ℝ) : Prop := x^2 - k*x + 10 = 0
def roots_relation (a b k : ℝ) : Prop :=
  equation1 a k ∧ 
  equation1 b k ∧ 
  equation2 (a + 3) k ∧
  equation2 (b + 3) k

-- Statement to be proven
theorem find_k (a b k : ℝ) (h : roots_relation a b k) : k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l1418_141801


namespace NUMINAMATH_GPT_ratio_proof_l1418_141890

variables {d l e : ℕ} -- Define variables representing the number of doctors, lawyers, and engineers
variables (hd : ℕ → ℕ) (hl : ℕ → ℕ) (he : ℕ → ℕ) (ho : ℕ → ℕ)

-- Condition: Average ages
def avg_age_doctors := 40 * d
def avg_age_lawyers := 55 * l
def avg_age_engineers := 35 * e

-- Condition: Overall average age is 45 years
def overall_avg_age := (40 * d + 55 * l + 35 * e) / (d + l + e)

theorem ratio_proof (h1 : 40 * d + 55 * l + 35 * e = 45 * (d + l + e)) : 
  d = l ∧ e = 2 * l :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l1418_141890


namespace NUMINAMATH_GPT_scientific_notation_of_10760000_l1418_141845

theorem scientific_notation_of_10760000 : 
  (10760000 : ℝ) = 1.076 * 10^7 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_10760000_l1418_141845


namespace NUMINAMATH_GPT_circle_area_and_circumference_changes_l1418_141803

noncomputable section

structure Circle :=
  (r : ℝ)

def area (c : Circle) : ℝ := Real.pi * c.r^2

def circumference (c : Circle) : ℝ := 2 * Real.pi * c.r

def percentage_change (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem circle_area_and_circumference_changes
  (r1 r2 : ℝ) (c1 : Circle := {r := r1}) (c2 : Circle := {r := r2})
  (h1 : r1 = 5) (h2 : r2 = 4) :
  let original_area := area c1
  let new_area := area c2
  let original_circumference := circumference c1
  let new_circumference := circumference c2
  percentage_change original_area new_area = 36 ∧
  new_circumference = 8 * Real.pi ∧
  percentage_change original_circumference new_circumference = 20 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_and_circumference_changes_l1418_141803


namespace NUMINAMATH_GPT_length_sum_l1418_141876

theorem length_sum : 
  let m := 1 -- Meter as base unit
  let cm := 0.01 -- 1 cm in meters
  let mm := 0.001 -- 1 mm in meters
  2 * m + 3 * cm + 5 * mm = 2.035 * m :=
by sorry

end NUMINAMATH_GPT_length_sum_l1418_141876


namespace NUMINAMATH_GPT_delaney_travel_time_l1418_141835

def bus_leaves_at := 8 * 60
def delaney_left_at := 7 * 60 + 50
def missed_by := 20

theorem delaney_travel_time
  (bus_leaves_at : ℕ) (delaney_left_at : ℕ) (missed_by : ℕ) :
  delaney_left_at + (bus_leaves_at + missed_by - bus_leaves_at) - delaney_left_at = 30 :=
by
  exact sorry

end NUMINAMATH_GPT_delaney_travel_time_l1418_141835


namespace NUMINAMATH_GPT_students_play_both_football_and_tennis_l1418_141897

theorem students_play_both_football_and_tennis 
  (T : ℕ) (F : ℕ) (L : ℕ) (N : ℕ) (B : ℕ)
  (hT : T = 38) (hF : F = 26) (hL : L = 20) (hN : N = 9) :
  B = F + L - (T - N) → B = 17 :=
by 
  intros h
  rw [hT, hF, hL, hN] at h
  exact h

end NUMINAMATH_GPT_students_play_both_football_and_tennis_l1418_141897


namespace NUMINAMATH_GPT_remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l1418_141892

theorem remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14 
  (a b c d e f g h : ℤ) 
  (h1 : a = 11085)
  (h2 : b = 11087)
  (h3 : c = 11089)
  (h4 : d = 11091)
  (h5 : e = 11093)
  (h6 : f = 11095)
  (h7 : g = 11097)
  (h8 : h = 11099) :
  (2 * (a + b + c + d + e + f + g + h)) % 14 = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l1418_141892


namespace NUMINAMATH_GPT_cost_of_coffee_A_per_kg_l1418_141838

theorem cost_of_coffee_A_per_kg (x : ℝ) :
  (240 * x + 240 * 12 = 480 * 11) → x = 10 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cost_of_coffee_A_per_kg_l1418_141838


namespace NUMINAMATH_GPT_evaluate_exponentiation_l1418_141843

theorem evaluate_exponentiation : (3 ^ 3) ^ 4 = 531441 := by
  sorry

end NUMINAMATH_GPT_evaluate_exponentiation_l1418_141843


namespace NUMINAMATH_GPT_infinite_triples_exists_l1418_141844

/-- There are infinitely many ordered triples (a, b, c) of positive integers such that 
the greatest common divisor of a, b, and c is 1, and the sum a^2b^2 + b^2c^2 + c^2a^2 
is the square of an integer. -/
theorem infinite_triples_exists : ∃ (a b c : ℕ), (∀ p q : ℕ, p ≠ q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ 2 < p ∧ 2 < q →
  let a := p * q
  let b := 2 * p^2
  let c := q^2
  gcd (gcd a b) c = 1 ∧
  ∃ k : ℕ, a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = k^2) :=
sorry

end NUMINAMATH_GPT_infinite_triples_exists_l1418_141844


namespace NUMINAMATH_GPT_each_friend_pays_20_l1418_141851

def rent_cottage_cost_per_hour : ℕ := 5
def rent_cottage_hours : ℕ := 8
def total_rent_cost := rent_cottage_cost_per_hour * rent_cottage_hours
def number_of_friends : ℕ := 2
def each_friend_pays := total_rent_cost / number_of_friends

theorem each_friend_pays_20 :
  each_friend_pays = 20 := by
  sorry

end NUMINAMATH_GPT_each_friend_pays_20_l1418_141851


namespace NUMINAMATH_GPT_balls_balance_l1418_141879

theorem balls_balance (G Y W B : ℕ) (h1 : G = 2 * B) (h2 : Y = 5 * B / 2) (h3 : W = 3 * B / 2) :
  5 * G + 3 * Y + 3 * W = 22 * B :=
by
  sorry

end NUMINAMATH_GPT_balls_balance_l1418_141879


namespace NUMINAMATH_GPT_total_donations_correct_l1418_141852

def num_basketball_hoops : Nat := 60

def num_hoops_with_balls : Nat := num_basketball_hoops / 2

def num_pool_floats : Nat := 120
def num_damaged_floats : Nat := num_pool_floats / 4
def num_remaining_floats : Nat := num_pool_floats - num_damaged_floats

def num_footballs : Nat := 50
def num_tennis_balls : Nat := 40

def num_hoops_without_balls : Nat := num_basketball_hoops - num_hoops_with_balls

def total_donations : Nat := 
  num_hoops_without_balls + num_hoops_with_balls + num_remaining_floats + num_footballs + num_tennis_balls

theorem total_donations_correct : total_donations = 240 := by
  sorry

end NUMINAMATH_GPT_total_donations_correct_l1418_141852


namespace NUMINAMATH_GPT_number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l1418_141836

theorem number_of_pentagonal_faces_is_12_more_than_heptagonal_faces
  (convex : Prop)
  (trihedral : Prop)
  (faces_have_5_6_or_7_sides : Prop)
  (V E F : ℕ)
  (a b c : ℕ)
  (euler : V - E + F = 2)
  (edges_def : E = (5 * a + 6 * b + 7 * c) / 2)
  (vertices_def : V = (5 * a + 6 * b + 7 * c) / 3) :
  a = c + 12 :=
  sorry

end NUMINAMATH_GPT_number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l1418_141836


namespace NUMINAMATH_GPT_perimeter_greater_than_diagonals_l1418_141896

namespace InscribedQuadrilateral

def is_convex_quadrilateral (AB BC CD DA AC BD: ℝ) : Prop :=
  -- Conditions for a convex quadrilateral (simple check)
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ AC > 0 ∧ BD > 0

def is_inscribed_in_circle (AB BC CD DA AC BD: ℝ) (r: ℝ) : Prop :=
  -- Check if quadrilateral is inscribed in a circle of radius 1
  r = 1

theorem perimeter_greater_than_diagonals 
  (AB BC CD DA AC BD: ℝ) 
  (r: ℝ)
  (h1 : is_convex_quadrilateral AB BC CD DA AC BD) 
  (h2 : is_inscribed_in_circle AB BC CD DA AC BD r) :
  0 < (AB + BC + CD + DA) - (AC + BD) ∧ (AB + BC + CD + DA) - (AC + BD) < 2 :=
by
  sorry 

end InscribedQuadrilateral

end NUMINAMATH_GPT_perimeter_greater_than_diagonals_l1418_141896


namespace NUMINAMATH_GPT_distance_between_cars_l1418_141848

-- Definitions representing the initial conditions and distances traveled by the cars
def initial_distance : ℕ := 113
def first_car_distance_on_road : ℕ := 50
def second_car_distance_on_road : ℕ := 35

-- Statement of the theorem to be proved
theorem distance_between_cars : initial_distance - (first_car_distance_on_road + second_car_distance_on_road) = 28 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_cars_l1418_141848


namespace NUMINAMATH_GPT_area_of_sine_triangle_l1418_141800

-- We define the problem conditions and the statement we want to prove
theorem area_of_sine_triangle (A B C : Real) (area_ABC : ℝ) (unit_circle : ℝ) :
  unit_circle = 1 → area_ABC = 1 / 2 →
  let a := 2 * Real.sin A
  let b := 2 * Real.sin B
  let c := 2 * Real.sin C
  let s := (a + b + c) / 2
  let area_sine_triangle := 
    (s * (s - a) * (s - b) * (s - c)).sqrt / 4 
  area_sine_triangle = 1 / 8 :=
by
  intros
  sorry -- Proof is left as an exercise

end NUMINAMATH_GPT_area_of_sine_triangle_l1418_141800


namespace NUMINAMATH_GPT_magazines_cover_area_l1418_141885

theorem magazines_cover_area (S : ℝ) (n : ℕ) (h_n_15 : n = 15) (h_cover : ∀ m ≤ n, ∃(Sm:ℝ), (Sm ≥ (m : ℝ) / n * S) ) :
  ∃ k : ℕ, k = n - 7 ∧ ∃ (Sk : ℝ), (Sk ≥ 8/15 * S) := 
by
  sorry

end NUMINAMATH_GPT_magazines_cover_area_l1418_141885


namespace NUMINAMATH_GPT_product_evaluation_l1418_141898

theorem product_evaluation : 
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end NUMINAMATH_GPT_product_evaluation_l1418_141898


namespace NUMINAMATH_GPT_probability_of_success_l1418_141809

theorem probability_of_success 
  (pA : ℚ) (pB : ℚ) 
  (hA : pA = 2 / 3) 
  (hB : pB = 3 / 5) :
  1 - ((1 - pA) * (1 - pB)) = 13 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_success_l1418_141809


namespace NUMINAMATH_GPT_units_digit_of_24_pow_4_add_42_pow_4_l1418_141826

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end NUMINAMATH_GPT_units_digit_of_24_pow_4_add_42_pow_4_l1418_141826


namespace NUMINAMATH_GPT_inhabitants_reach_ball_on_time_l1418_141833

theorem inhabitants_reach_ball_on_time
  (kingdom_side_length : ℝ)
  (messenger_sent_at : ℕ)
  (ball_begins_at : ℕ)
  (inhabitant_speed : ℝ)
  (time_available : ℝ)
  (max_distance_within_square : ℝ)
  (H_side_length : kingdom_side_length = 2)
  (H_messenger_time : messenger_sent_at = 12)
  (H_ball_time : ball_begins_at = 19)
  (H_speed : inhabitant_speed = 3)
  (H_time_avail : time_available = 7)
  (H_max_distance : max_distance_within_square = 2 * Real.sqrt 2) :
  ∃ t : ℝ, t ≤ time_available ∧ max_distance_within_square / inhabitant_speed ≤ t :=
by
  -- You would write the proof here.
  sorry

end NUMINAMATH_GPT_inhabitants_reach_ball_on_time_l1418_141833


namespace NUMINAMATH_GPT_average_age_before_new_students_joined_l1418_141884

theorem average_age_before_new_students_joined 
  (A : ℝ) 
  (N : ℕ) 
  (new_students_average_age : ℝ) 
  (average_age_drop : ℝ) 
  (original_class_strength : ℕ)
  (hN : N = 17) 
  (h_new_students : new_students_average_age = 32)
  (h_age_drop : average_age_drop = 4)
  (h_strength : original_class_strength = 17)
  (h_equation : 17 * A + 17 * new_students_average_age = (2 * original_class_strength) * (A - average_age_drop)) :
  A = 40 :=
by sorry

end NUMINAMATH_GPT_average_age_before_new_students_joined_l1418_141884


namespace NUMINAMATH_GPT_error_percentage_in_area_l1418_141861

theorem error_percentage_in_area
  (L W : ℝ)          -- Actual length and width of the rectangle
  (hL' : ℝ)          -- Measured length with 8% excess
  (hW' : ℝ)          -- Measured width with 5% deficit
  (hL'_def : hL' = 1.08 * L)  -- Condition for length excess
  (hW'_def : hW' = 0.95 * W)  -- Condition for width deficit
  :
  ((hL' * hW' - L * W) / (L * W) * 100 = 2.6) := sorry

end NUMINAMATH_GPT_error_percentage_in_area_l1418_141861


namespace NUMINAMATH_GPT_number_of_terms_is_10_l1418_141813

noncomputable def arith_seq_number_of_terms (a : ℕ) (n : ℕ) (d : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((n-1)*d = 16) ∧ (n * (2*a + (n-2)*d) = 56) ∧ (n * (2*a + n*d) = 76)

theorem number_of_terms_is_10 (a d n : ℕ) (h : arith_seq_number_of_terms a n d) : n = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_terms_is_10_l1418_141813


namespace NUMINAMATH_GPT_abigail_spent_in_store_l1418_141828

theorem abigail_spent_in_store (initial_amount : ℕ) (amount_left : ℕ) (amount_lost : ℕ) (spent : ℕ) 
  (h1 : initial_amount = 11) 
  (h2 : amount_left = 3)
  (h3 : amount_lost = 6) :
  spent = initial_amount - (amount_left + amount_lost) :=
by
  sorry

end NUMINAMATH_GPT_abigail_spent_in_store_l1418_141828


namespace NUMINAMATH_GPT_find_f1_find_f3_range_of_x_l1418_141867

-- Define f as described
axiom f : ℝ → ℝ
axiom f_domain : ∀ (x : ℝ), x > 0 → ∃ (y : ℝ), f y = f x

-- Given conditions
axiom condition1 : ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0
axiom condition2 : ∀ (x y : ℝ), 0 < x ∧ 0 < y → f (x * y) = f x + f y
axiom condition3 : f (1 / 3) = 1

-- Prove f(1) = 0
theorem find_f1 : f 1 = 0 := by sorry

-- Prove f(3) = -1
theorem find_f3 : f 3 = -1 := by sorry

-- Given inequality condition
axiom condition4 : ∀ x : ℝ, 0 < x → f x < 2 + f (2 - x)

-- Prove range of x for given inequality
theorem range_of_x : ∀ x, x > 1 / 5 ∧ x < 2 ↔ f x < 2 + f (2 - x) := by sorry

end NUMINAMATH_GPT_find_f1_find_f3_range_of_x_l1418_141867


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1418_141895

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def is_geometric_sequence (x y z : ℝ) (q : ℝ) : Prop :=
  y^2 = x * z

theorem common_ratio_of_geometric_sequence 
    (a_n : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a_n) 
    (a1 a3 a5 : ℝ)
    (h1 : a1 = a_n 1 + 1) 
    (h3 : a3 = a_n 3 + 3) 
    (h5 : a5 = a_n 5 + 5) 
    (h_geom : is_geometric_sequence a1 a3 a5 1) : 
  1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1418_141895


namespace NUMINAMATH_GPT_expression_evaluation_l1418_141805

theorem expression_evaluation : 
  3 / 5 * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1418_141805


namespace NUMINAMATH_GPT_cost_to_replace_and_install_l1418_141878

theorem cost_to_replace_and_install (s l : ℕ) 
  (h1 : l = 3 * s) (h2 : 2 * s + 2 * l = 640) 
  (cost_per_foot : ℕ) (cost_per_gate : ℕ) (installation_cost_per_gate : ℕ) 
  (h3 : cost_per_foot = 5) (h4 : cost_per_gate = 150) (h5 : installation_cost_per_gate = 75) : 
  (s * cost_per_foot + 2 * (cost_per_gate + installation_cost_per_gate)) = 850 := 
by 
  sorry

end NUMINAMATH_GPT_cost_to_replace_and_install_l1418_141878


namespace NUMINAMATH_GPT_maximize_A_plus_C_l1418_141863

theorem maximize_A_plus_C (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
 (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (hB : B = 2) (h7 : (A + C) % (B + D) = 0) 
 (h8 : A < 10) (h9 : B < 10) (h10 : C < 10) (h11 : D < 10) : 
 A + C ≤ 15 :=
sorry

end NUMINAMATH_GPT_maximize_A_plus_C_l1418_141863


namespace NUMINAMATH_GPT_painter_completion_time_l1418_141816

def hours_elapsed (start_time end_time : String) : ℕ :=
  match (start_time, end_time) with
  | ("9:00 AM", "12:00 PM") => 3
  | _ => 0

-- The initial conditions, the start time is 9:00 AM, and 3 hours later 1/4th is done
def start_time := "9:00 AM"
def partial_completion_time := "12:00 PM"
def partial_completion_fraction := 1 / 4
def partial_time_hours := hours_elapsed start_time partial_completion_time

-- The painter works consistently, so it would take 4 times the partial time to complete the job
def total_time_hours := 4 * partial_time_hours

-- Calculate the completion time by adding total_time_hours to the start_time
def completion_time : String :=
  match start_time with
  | "9:00 AM" => "9:00 PM"
  | _         => "unknown"

theorem painter_completion_time :
  completion_time = "9:00 PM" :=
by
  -- Definitions and calculations already included in the setup
  sorry

end NUMINAMATH_GPT_painter_completion_time_l1418_141816


namespace NUMINAMATH_GPT_problem_R_l1418_141820

noncomputable def R (g S h : ℝ) : ℝ := g * S + h

theorem problem_R {g h : ℝ} (h_h : h = 6 - 4 * g) :
  R g 14 h = 56 :=
by
  sorry

end NUMINAMATH_GPT_problem_R_l1418_141820


namespace NUMINAMATH_GPT_find_c_l1418_141815

open Real

noncomputable def triangle_side_c (a b c : ℝ) (A B C : ℝ) :=
  A = (π / 4) ∧
  2 * b * sin B - c * sin C = 2 * a * sin A ∧
  (1/2) * b * c * (sqrt 2)/2 = 3 →
  c = 2 * sqrt 2
  
theorem find_c {a b c A B C : ℝ} (h : triangle_side_c a b c A B C) : c = 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_c_l1418_141815


namespace NUMINAMATH_GPT_minimum_value_expression_l1418_141881

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 4)

theorem minimum_value_expression : (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) = 192 := by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l1418_141881


namespace NUMINAMATH_GPT_opposite_of_neg_one_third_l1418_141869

theorem opposite_of_neg_one_third : -(-1/3) = 1/3 := 
sorry

end NUMINAMATH_GPT_opposite_of_neg_one_third_l1418_141869


namespace NUMINAMATH_GPT_roberta_started_with_8_records_l1418_141858

variable (R : ℕ)

def received_records := 12
def bought_records := 30
def total_received_and_bought := received_records + bought_records

theorem roberta_started_with_8_records (h : R + total_received_and_bought = 50) : R = 8 :=
by
  sorry

end NUMINAMATH_GPT_roberta_started_with_8_records_l1418_141858


namespace NUMINAMATH_GPT_no_such_natural_numbers_l1418_141837

theorem no_such_natural_numbers :
  ¬ ∃ (x y : ℕ), (∃ (a b : ℕ), x^2 + y = a^2 ∧ x - y = b^2) := 
sorry

end NUMINAMATH_GPT_no_such_natural_numbers_l1418_141837


namespace NUMINAMATH_GPT_area_of_smallest_square_l1418_141855

theorem area_of_smallest_square (radius : ℝ) (h : radius = 6) : 
    ∃ s : ℝ, s = 2 * radius ∧ s^2 = 144 :=
by
  sorry

end NUMINAMATH_GPT_area_of_smallest_square_l1418_141855


namespace NUMINAMATH_GPT_green_beads_in_each_necklace_l1418_141868

theorem green_beads_in_each_necklace (G : ℕ) :
  (∀ n, (n = 5) → (6 * n ≤ 45) ∧ (3 * n ≤ 45) ∧ (G * n = 45)) → G = 9 :=
by
  intros h
  have hn : 5 = 5 := rfl
  cases h 5 hn
  sorry

end NUMINAMATH_GPT_green_beads_in_each_necklace_l1418_141868


namespace NUMINAMATH_GPT_particle_speed_correct_l1418_141894

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 9)

noncomputable def particle_speed : ℝ :=
  Real.sqrt (3 ^ 2 + 5 ^ 2)

theorem particle_speed_correct : particle_speed = Real.sqrt 34 := by
  sorry

end NUMINAMATH_GPT_particle_speed_correct_l1418_141894


namespace NUMINAMATH_GPT_solve_for_a_l1418_141846

theorem solve_for_a : ∀ (a : ℝ), (2 * a - 16 = 9) → (a = 12.5) :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_solve_for_a_l1418_141846


namespace NUMINAMATH_GPT_most_lines_of_symmetry_circle_l1418_141817

-- Define the figures and their lines of symmetry
def regular_pentagon_lines_of_symmetry : ℕ := 5
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def circle_lines_of_symmetry : ℕ := 0  -- Representing infinite lines of symmetry in Lean is unconventional; we'll use a special case.
def regular_hexagon_lines_of_symmetry : ℕ := 6
def ellipse_lines_of_symmetry : ℕ := 2

-- Define a predicate to check if one figure has more lines of symmetry than all others
def most_lines_of_symmetry {α : Type} [LinearOrder α] (f : α) (others : List α) : Prop :=
  ∀ x ∈ others, f ≥ x

-- Define the problem statement in Lean
theorem most_lines_of_symmetry_circle :
  most_lines_of_symmetry circle_lines_of_symmetry [
    regular_pentagon_lines_of_symmetry,
    isosceles_triangle_lines_of_symmetry,
    regular_hexagon_lines_of_symmetry,
    ellipse_lines_of_symmetry ] :=
by {
  -- To represent infinite lines, we consider 0 as a larger "dummy" number in this context,
  -- since in Lean we don't have a built-in representation for infinity in finite ordering.
  -- Replace with a suitable model if necessary.
  sorry
}

end NUMINAMATH_GPT_most_lines_of_symmetry_circle_l1418_141817


namespace NUMINAMATH_GPT_delphine_chocolates_l1418_141877

theorem delphine_chocolates (x : ℕ) 
  (h1 : ∃ n, n = (2 * x - 3)) 
  (h2 : ∃ m, m = (x - 2))
  (h3 : ∃ p, p = (x - 3))
  (total_eq : x + (2 * x - 3) + (x - 2) + (x - 3) + 12 = 24) : 
  x = 4 := 
sorry

end NUMINAMATH_GPT_delphine_chocolates_l1418_141877


namespace NUMINAMATH_GPT_difference_in_probabilities_is_twenty_percent_l1418_141819

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end NUMINAMATH_GPT_difference_in_probabilities_is_twenty_percent_l1418_141819


namespace NUMINAMATH_GPT_sequence_convergence_l1418_141886

noncomputable def alpha : ℝ := sorry
def bounded (a : ℕ → ℝ) : Prop := ∃ M > 0, ∀ n, ‖a n‖ ≤ M

-- Translation of the math problem
theorem sequence_convergence (a : ℕ → ℝ) (ha : bounded a) (hα : 0 < alpha ∧ alpha ≤ 1) 
  (ineq : ∀ n ≥ 2, a (n+1) ≤ alpha * a n + (1 - alpha) * a (n-1)) : 
  ∃ l, ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖a n - l‖ < ε := 
sorry

end NUMINAMATH_GPT_sequence_convergence_l1418_141886


namespace NUMINAMATH_GPT_minimum_value_x2_y2_l1418_141810

variable {x y : ℝ}

theorem minimum_value_x2_y2 (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x * y = 1) : x^2 + y^2 = 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_x2_y2_l1418_141810


namespace NUMINAMATH_GPT_xiao_li_place_l1418_141806

def guess_A (place : String) : Prop :=
  place ≠ "first" ∧ place ≠ "second"

def guess_B (place : String) : Prop :=
  place ≠ "first" ∧ place = "third"

def guess_C (place : String) : Prop :=
  place ≠ "third" ∧ place = "first"

def correct_guesses (guess : String → Prop) (place : String) : Prop :=
  guess place

def half_correct_guesses (guess : String → Prop) (place : String) : Prop :=
  (guess "first" = (place = "first")) ∨
  (guess "second" = (place = "second")) ∨
  (guess "third" = (place = "third"))

theorem xiao_li_place :
  ∃ (place : String),
  (correct_guesses guess_A place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_B place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_B place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_A place) :=
sorry

end NUMINAMATH_GPT_xiao_li_place_l1418_141806


namespace NUMINAMATH_GPT_group_total_people_l1418_141830

theorem group_total_people (k : ℕ) (h1 : k = 7) (h2 : ((n - k) / n : ℝ) - (k / n : ℝ) = 0.30000000000000004) : n = 20 :=
  sorry

end NUMINAMATH_GPT_group_total_people_l1418_141830


namespace NUMINAMATH_GPT_fraction_spent_at_arcade_l1418_141856

theorem fraction_spent_at_arcade :
  ∃ f : ℝ, 
    (2.25 - (2.25 * f) - ((2.25 - (2.25 * f)) / 3) = 0.60) → 
    f = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_at_arcade_l1418_141856


namespace NUMINAMATH_GPT_solve_problem_l1418_141824

noncomputable def problem_statement : Prop :=
  let a := Real.arcsin (4/5)
  let b := Real.arccos (1/2)
  Real.sin (a + b) = (4 + 3 * Real.sqrt 3) / 10

theorem solve_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1418_141824


namespace NUMINAMATH_GPT_pizzas_ordered_l1418_141866

variable (m : ℕ) (x : ℕ)

theorem pizzas_ordered (h1 : m * 2 * x = 14) (h2 : x = 1 / 2 * m) (h3 : m > 13) : 
  14 + 13 * x = 15 := 
sorry

end NUMINAMATH_GPT_pizzas_ordered_l1418_141866


namespace NUMINAMATH_GPT_plan_b_rate_l1418_141870

noncomputable def cost_plan_a (duration : ℕ) : ℝ :=
  if duration ≤ 4 then 0.60
  else 0.60 + 0.06 * (duration - 4)

def cost_plan_b (duration : ℕ) (rate : ℝ) : ℝ :=
  rate * duration

theorem plan_b_rate (rate : ℝ) : 
  cost_plan_a 18 = cost_plan_b 18 rate → rate = 0.08 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_plan_b_rate_l1418_141870


namespace NUMINAMATH_GPT_Isabel_afternoon_runs_l1418_141850

theorem Isabel_afternoon_runs (circuit_length morning_runs weekly_distance afternoon_runs : ℕ)
  (h_circuit_length : circuit_length = 365)
  (h_morning_runs : morning_runs = 7)
  (h_weekly_distance : weekly_distance = 25550)
  (h_afternoon_runs : weekly_distance = morning_runs * circuit_length * 7 + afternoon_runs * circuit_length) :
  afternoon_runs = 21 :=
by
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_Isabel_afternoon_runs_l1418_141850


namespace NUMINAMATH_GPT_parallelogram_point_D_l1418_141812

/-- Given points A, B, and C, the coordinates of point D in parallelogram ABCD -/
theorem parallelogram_point_D (A B C D : (ℝ × ℝ))
  (hA : A = (1, 1))
  (hB : B = (3, 2))
  (hC : C = (6, 3))
  (hMid : (2 * (A.1 + C.1), 2 * (A.2 + C.2)) = (2 * (B.1 + D.1), 2 * (B.2 + D.2))) :
  D = (4, 2) :=
sorry

end NUMINAMATH_GPT_parallelogram_point_D_l1418_141812


namespace NUMINAMATH_GPT_investment_amount_l1418_141874

noncomputable def PV (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

theorem investment_amount (FV : ℝ) (r : ℝ) (n : ℕ) (PV : ℝ) : FV = 1000000 ∧ r = 0.08 ∧ n = 20 → PV = 1000000 / (1 + 0.08)^20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_investment_amount_l1418_141874


namespace NUMINAMATH_GPT_length_of_walls_l1418_141842

-- Definitions of the given conditions.
def wall_height : ℝ := 12
def third_wall_length : ℝ := 20
def third_wall_height : ℝ := 12
def total_area : ℝ := 960

-- The area of two walls with length L each and height 12 feet.
def two_walls_area (L : ℝ) : ℝ := 2 * L * wall_height

-- The area of the third wall.
def third_wall_area : ℝ := third_wall_length * third_wall_height

-- The proof statement
theorem length_of_walls (L : ℝ) (h1 : two_walls_area L + third_wall_area = total_area) : L = 30 :=
by
  sorry

end NUMINAMATH_GPT_length_of_walls_l1418_141842


namespace NUMINAMATH_GPT_total_students_1150_l1418_141857

theorem total_students_1150 (T G : ℝ) (h1 : 92 + G = T) (h2 : G = 0.92 * T) : T = 1150 := 
by
  sorry

end NUMINAMATH_GPT_total_students_1150_l1418_141857


namespace NUMINAMATH_GPT_toms_total_miles_l1418_141899

-- Define the conditions as facts
def days_in_year : ℕ := 365
def first_part_days : ℕ := 183
def second_part_days : ℕ := days_in_year - first_part_days
def miles_per_day_first_part : ℕ := 30
def miles_per_day_second_part : ℕ := 35

-- State the final theorem
theorem toms_total_miles : 
  (first_part_days * miles_per_day_first_part) + (second_part_days * miles_per_day_second_part) = 11860 := by 
  sorry

end NUMINAMATH_GPT_toms_total_miles_l1418_141899


namespace NUMINAMATH_GPT_triangle_area_is_2_l1418_141831

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_2_l1418_141831


namespace NUMINAMATH_GPT_burger_cost_l1418_141872

theorem burger_cost 
    (b s : ℕ) 
    (h1 : 5 * b + 3 * s = 500) 
    (h2 : 3 * b + 2 * s = 310) :
    b = 70 := by
  sorry

end NUMINAMATH_GPT_burger_cost_l1418_141872


namespace NUMINAMATH_GPT_marbles_lost_l1418_141807

theorem marbles_lost (initial_marbs remaining_marbs marbles_lost : ℕ)
  (h1 : initial_marbs = 38)
  (h2 : remaining_marbs = 23)
  : marbles_lost = initial_marbs - remaining_marbs :=
by
  sorry

end NUMINAMATH_GPT_marbles_lost_l1418_141807


namespace NUMINAMATH_GPT_team_arrangement_count_l1418_141841

-- Definitions of the problem
def veteran_players := 2
def new_players := 3
def total_players := veteran_players + new_players
def team_size := 3

-- Conditions
def condition_veteran : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → Finset.card (team ∩ (Finset.range veteran_players)) ≥ 1

def condition_new_player : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → 
    ∃ (p1 p2 : ℕ), p1 ∈ team ∧ p2 ∈ team ∧ 
    p1 ≠ p2 ∧ p1 < team_size ∧ p2 < team_size ∧
    (p1 ∈ (Finset.Ico veteran_players total_players) ∨ p2 ∈ (Finset.Ico veteran_players total_players))

-- Goal
def number_of_arrangements := 48

-- The statement to prove
theorem team_arrangement_count : condition_veteran → condition_new_player → 
  (∃ (arrangements : ℕ), arrangements = number_of_arrangements) :=
by
  sorry

end NUMINAMATH_GPT_team_arrangement_count_l1418_141841
