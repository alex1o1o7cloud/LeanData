import Mathlib

namespace NUMINAMATH_GPT_Tim_eats_91_pickle_slices_l292_29269

theorem Tim_eats_91_pickle_slices :
  let Sammy := 25
  let Tammy := 3 * Sammy
  let Ron := Tammy - 0.15 * Tammy
  let Amy := Sammy + 0.50 * Sammy
  let CombinedTotal := Ron + Amy
  let Tim := CombinedTotal - 0.10 * CombinedTotal
  Tim = 91 :=
by
  admit

end NUMINAMATH_GPT_Tim_eats_91_pickle_slices_l292_29269


namespace NUMINAMATH_GPT_product_is_cube_l292_29252

/-
  Given conditions:
    - a, b, and c are distinct composite natural numbers.
    - None of a, b, and c are divisible by any of the integers from 2 to 100 inclusive.
    - a, b, and c are the smallest possible numbers satisfying the above conditions.

  We need to prove that their product a * b * c is a cube of a natural number.
-/

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q

theorem product_is_cube (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : is_composite a) (h5 : is_composite b) (h6 : is_composite c)
  (h7 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ a))
  (h8 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ b))
  (h9 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ c))
  (h10 : ∀ (d e f : ℕ), is_composite d → is_composite e → is_composite f → d ≠ e → e ≠ f → d ≠ f → 
         (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ d)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ e)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ f)) →
         (d * e * f ≥ a * b * c)) :
  ∃ (n : ℕ), a * b * c = n ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_product_is_cube_l292_29252


namespace NUMINAMATH_GPT_dinosaur_book_cost_l292_29246

theorem dinosaur_book_cost (D : ℕ) : 
  (11 + D + 7 = 37) → (D = 19) := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_dinosaur_book_cost_l292_29246


namespace NUMINAMATH_GPT_sara_ticket_cost_l292_29291

noncomputable def calc_ticket_price : ℝ :=
  let rented_movie_cost := 1.59
  let bought_movie_cost := 13.95
  let total_cost := 36.78
  let total_tickets := 2
  let spent_on_tickets := total_cost - (rented_movie_cost + bought_movie_cost)
  spent_on_tickets / total_tickets

theorem sara_ticket_cost : calc_ticket_price = 10.62 := by
  sorry

end NUMINAMATH_GPT_sara_ticket_cost_l292_29291


namespace NUMINAMATH_GPT_sum_of_cubes_l292_29253

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 20) : x^3 + y^3 = 87.5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l292_29253


namespace NUMINAMATH_GPT_divisor_count_of_45_l292_29219

theorem divisor_count_of_45 : 
  ∃ (n : ℤ), n = 12 ∧ ∀ d : ℤ, d ∣ 45 → (d > 0 ∨ d < 0) := sorry

end NUMINAMATH_GPT_divisor_count_of_45_l292_29219


namespace NUMINAMATH_GPT_Xiaoming_age_l292_29214

theorem Xiaoming_age (x : ℕ) (h1 : x = x) (h2 : x + 18 = 2 * (x + 6)) : x = 6 :=
sorry

end NUMINAMATH_GPT_Xiaoming_age_l292_29214


namespace NUMINAMATH_GPT_oldest_child_age_l292_29287

def avg (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem oldest_child_age (a b : ℕ) (h1 : avg a b x = 10) (h2 : a = 8) (h3 : b = 11) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_oldest_child_age_l292_29287


namespace NUMINAMATH_GPT_arthur_num_hamburgers_on_first_day_l292_29208

theorem arthur_num_hamburgers_on_first_day (H D : ℕ) (hamburgers_1 hamburgers_2 : ℕ) (hotdogs_1 hotdogs_2 : ℕ)
  (h1 : hamburgers_1 * H + hotdogs_1 * D = 10)
  (h2 : hamburgers_2 * H + hotdogs_2 * D = 7)
  (hprice : D = 1)
  (h1_hotdogs : hotdogs_1 = 4)
  (h2_hotdogs : hotdogs_2 = 3) : 
  hamburgers_1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_arthur_num_hamburgers_on_first_day_l292_29208


namespace NUMINAMATH_GPT_combined_weight_of_daughter_and_child_l292_29242

theorem combined_weight_of_daughter_and_child 
  (G D C : ℝ)
  (h1 : G + D + C = 110)
  (h2 : C = 1/5 * G)
  (h3 : D = 50) :
  D + C = 60 :=
sorry

end NUMINAMATH_GPT_combined_weight_of_daughter_and_child_l292_29242


namespace NUMINAMATH_GPT_explicit_expression_solve_inequality_l292_29249

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n+1)

theorem explicit_expression (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x)) :
  (∀ n x, f n x = x^3) :=
by
  sorry

theorem solve_inequality (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x))
  (f_eq : ∀ n x, f n x = x^3) :
  ∀ x, (x + 1)^3 + (3 - 2*x)^3 > 0 → x < 4 :=
by
  sorry

end NUMINAMATH_GPT_explicit_expression_solve_inequality_l292_29249


namespace NUMINAMATH_GPT_work_hours_l292_29272

namespace JohnnyWork

variable (dollarsPerHour : ℝ) (totalDollars : ℝ)

theorem work_hours 
  (h_wage : dollarsPerHour = 3.25)
  (h_earned : totalDollars = 26) 
  : (totalDollars / dollarsPerHour) = 8 := 
by
  rw [h_wage, h_earned]
  -- proof goes here
  sorry

end JohnnyWork

end NUMINAMATH_GPT_work_hours_l292_29272


namespace NUMINAMATH_GPT_original_number_of_men_l292_29241

theorem original_number_of_men (M : ℤ) (h1 : 8 * M = 5 * (M + 10)) : M = 17 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_original_number_of_men_l292_29241


namespace NUMINAMATH_GPT_product_of_values_l292_29250

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := |2 * x| + 4 = 38

-- State the theorem
theorem product_of_values : ∃ x1 x2 : ℝ, satisfies_eq x1 ∧ satisfies_eq x2 ∧ x1 * x2 = -289 := 
by
  sorry

end NUMINAMATH_GPT_product_of_values_l292_29250


namespace NUMINAMATH_GPT_arithmetic_seq_general_term_geometric_seq_general_term_l292_29285

theorem arithmetic_seq_general_term (a : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2) :
  ∀ n, a n = 2 * n + 2 :=
by sorry

theorem geometric_seq_general_term (a b : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2)
  (h3 : b 2 = a 3) (h4 : b 3 = a 7) :
  ∀ n, b n = 2 ^ (n + 1) :=
by sorry

end NUMINAMATH_GPT_arithmetic_seq_general_term_geometric_seq_general_term_l292_29285


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l292_29201
open Real

theorem sufficient_not_necessary_condition (m : ℝ) :
  ((m = 0) → ∃ x y : ℝ, (m + 1) * x + (1 - m) * y - 1 = 0 ∧ (m - 1) * x + (2 * m + 1) * y + 4 = 0 ∧ 
  ((m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0 ∨ (m = 1 ∨ m = 0))) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l292_29201


namespace NUMINAMATH_GPT_original_amount_l292_29243

theorem original_amount (X : ℝ) (h : 0.05 * X = 25) : X = 500 :=
sorry

end NUMINAMATH_GPT_original_amount_l292_29243


namespace NUMINAMATH_GPT_gunther_typing_l292_29231

theorem gunther_typing :
  ∀ (wpm : ℚ), (wpm = 160 / 3) → 480 * wpm = 25598 :=
by
  intros wpm h
  sorry

end NUMINAMATH_GPT_gunther_typing_l292_29231


namespace NUMINAMATH_GPT_find_values_of_a_b_solve_inequality_l292_29221

variable (a b : ℝ)
variable (h1 : ∀ x : ℝ, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 2)

theorem find_values_of_a_b (h2 : a = -2) (h3 : b = 3) : 
  a = -2 ∧ b = 3 :=
by
  constructor
  exact h2
  exact h3


theorem solve_inequality 
  (h2 : a = -2) (h3 : b = 3) :
  ∀ x : ℝ, (a * x^2 + b * x - 1 > 0) ↔ (1/2 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_a_b_solve_inequality_l292_29221


namespace NUMINAMATH_GPT_mike_age_l292_29222

theorem mike_age : ∀ (m M : ℕ), m = M - 18 ∧ m + M = 54 → m = 18 :=
by
  intros m M
  intro h
  sorry

end NUMINAMATH_GPT_mike_age_l292_29222


namespace NUMINAMATH_GPT_factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l292_29293

theorem factorize_polynomial_1 (x y : ℝ) : 
  12 * x ^ 3 * y - 3 * x * y ^ 2 = 3 * x * y * (4 * x ^ 2 - y) := 
by sorry

theorem factorize_polynomial_2 (x : ℝ) : 
  x - 9 * x ^ 3 = x * (1 + 3 * x) * (1 - 3 * x) :=
by sorry

theorem factorize_polynomial_3 (a b : ℝ) : 
  3 * a ^ 2 - 12 * a * b * (a - b) = 3 * (a - 2 * b) ^ 2 := 
by sorry

end NUMINAMATH_GPT_factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l292_29293


namespace NUMINAMATH_GPT_sequence_product_l292_29259

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def a4_value (a : ℕ → ℕ) : Prop :=
a 4 = 2

-- The statement to be proven
theorem sequence_product (a : ℕ → ℕ) (q : ℕ) (h_geo_seq : geometric_sequence a q) (h_a4 : a4_value a) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end NUMINAMATH_GPT_sequence_product_l292_29259


namespace NUMINAMATH_GPT_geometric_sequence_k_eq_6_l292_29225

theorem geometric_sequence_k_eq_6 
  (a : ℕ → ℝ) (q : ℝ) (k : ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n = a 1 * q ^ (n - 1))
  (h3 : q ≠ 1)
  (h4 : q ≠ -1)
  (h5 : a k = a 2 * a 5) :
  k = 6 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_k_eq_6_l292_29225


namespace NUMINAMATH_GPT_wire_length_ratio_l292_29238

open Real

noncomputable def bonnie_wire_length : ℝ := 12 * 8
noncomputable def bonnie_cube_volume : ℝ := 8^3
noncomputable def roark_unit_cube_volume : ℝ := 2^3
noncomputable def roark_number_of_cubes : ℝ := bonnie_cube_volume / roark_unit_cube_volume
noncomputable def roark_wire_length_per_cube : ℝ := 12 * 2
noncomputable def roark_total_wire_length : ℝ := roark_number_of_cubes * roark_wire_length_per_cube
noncomputable def bonnie_to_roark_wire_ratio := bonnie_wire_length / roark_total_wire_length

theorem wire_length_ratio : bonnie_to_roark_wire_ratio = (1 : ℝ) / 16 :=
by
  sorry

end NUMINAMATH_GPT_wire_length_ratio_l292_29238


namespace NUMINAMATH_GPT_domain_log_function_l292_29247

/-- The quadratic expression x^2 - 2x + 3 is always positive. -/
lemma quadratic_positive (x : ℝ) : x^2 - 2*x + 3 > 0 :=
by
  sorry

/-- The domain of the function y = log(x^2 - 2x + 3) is all real numbers. -/
theorem domain_log_function : ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*x + 3) :=
by
  have h := quadratic_positive
  sorry

end NUMINAMATH_GPT_domain_log_function_l292_29247


namespace NUMINAMATH_GPT_words_written_first_two_hours_l292_29217

def essay_total_words : ℕ := 1200
def words_per_hour_first_two_hours (W : ℕ) : ℕ := 2 * W
def words_per_hour_next_two_hours : ℕ := 2 * 200

theorem words_written_first_two_hours (W : ℕ) (h : words_per_hour_first_two_hours W + words_per_hour_next_two_hours = essay_total_words) : W = 400 := 
by 
  sorry

end NUMINAMATH_GPT_words_written_first_two_hours_l292_29217


namespace NUMINAMATH_GPT_paco_initial_cookies_l292_29271

-- Define the given conditions
def cookies_given : ℕ := 14
def cookies_eaten : ℕ := 10
def cookies_left : ℕ := 12

-- Proposition to prove: Paco initially had 36 cookies
theorem paco_initial_cookies : (cookies_given + cookies_eaten + cookies_left = 36) :=
by
  sorry

end NUMINAMATH_GPT_paco_initial_cookies_l292_29271


namespace NUMINAMATH_GPT_fifth_score_l292_29278

theorem fifth_score (r : ℕ) 
  (h1 : r % 5 = 0)
  (h2 : (60 + 75 + 85 + 95 + r) / 5 = 80) : 
  r = 85 := by 
  sorry

end NUMINAMATH_GPT_fifth_score_l292_29278


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l292_29257

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 5/4) 
  (h_sequence : ∀ n, a n = a 1 * q ^ (n - 1)) : 
  q = 1/2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l292_29257


namespace NUMINAMATH_GPT_box_volume_l292_29264

theorem box_volume (a b c : ℝ) (H1 : a * b = 15) (H2 : b * c = 10) (H3 : c * a = 6) : a * b * c = 30 := 
sorry

end NUMINAMATH_GPT_box_volume_l292_29264


namespace NUMINAMATH_GPT_joan_video_game_spending_l292_29254

theorem joan_video_game_spending:
  let basketball_game := 5.20
  let racing_game := 4.23
  basketball_game + racing_game = 9.43 := 
by
  sorry

end NUMINAMATH_GPT_joan_video_game_spending_l292_29254


namespace NUMINAMATH_GPT_discount_percentage_of_sale_l292_29266

theorem discount_percentage_of_sale (initial_price sale_coupon saved_amount final_price : ℝ)
    (h1 : initial_price = 125)
    (h2 : sale_coupon = 10)
    (h3 : saved_amount = 44)
    (h4 : final_price = 81) :
    ∃ x : ℝ, x = 0.20 ∧ 
             (initial_price - initial_price * x - sale_coupon) - 
             0.10 * (initial_price - initial_price * x - sale_coupon) = final_price :=
by
  -- Proof should be constructed here
  sorry

end NUMINAMATH_GPT_discount_percentage_of_sale_l292_29266


namespace NUMINAMATH_GPT_speed_of_current_is_2_l292_29251

noncomputable def speed_current : ℝ :=
  let still_water_speed := 14  -- kmph
  let distance_m := 40         -- meters
  let time_s := 8.9992800576   -- seconds
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let downstream_speed := distance_km / time_h
  downstream_speed - still_water_speed

theorem speed_of_current_is_2 :
  speed_current = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_is_2_l292_29251


namespace NUMINAMATH_GPT_charge_per_trousers_l292_29218

-- Definitions
def pairs_of_trousers : ℕ := 10
def shirts : ℕ := 10
def bill : ℕ := 140
def charge_per_shirt : ℕ := 5

-- Theorem statement
theorem charge_per_trousers :
  ∃ (T : ℕ), (pairs_of_trousers * T + shirts * charge_per_shirt = bill) ∧ (T = 9) :=
by 
  sorry

end NUMINAMATH_GPT_charge_per_trousers_l292_29218


namespace NUMINAMATH_GPT_range_of_a_l292_29230

noncomputable def A (x : ℝ) : Prop := x < -2 ∨ x ≥ 1
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) : (∀ x, A x ∨ B x a) ↔ a ≤ -2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l292_29230


namespace NUMINAMATH_GPT_length_of_DC_l292_29265

noncomputable def AB : ℝ := 30
noncomputable def sine_A : ℝ := 4 / 5
noncomputable def sine_C : ℝ := 1 / 4
noncomputable def angle_ADB : ℝ := Real.pi / 2

theorem length_of_DC (h_AB : AB = 30) (h_sine_A : sine_A = 4 / 5) (h_sine_C : sine_C = 1 / 4) (h_angle_ADB : angle_ADB = Real.pi / 2) :
  ∃ DC : ℝ, DC = 24 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_GPT_length_of_DC_l292_29265


namespace NUMINAMATH_GPT_olivia_spent_89_l292_29245

-- Define initial and subsequent amounts
def initial_amount : ℕ := 100
def atm_amount : ℕ := 148
def after_supermarket : ℕ := 159

-- Total amount before supermarket
def total_before_supermarket : ℕ := initial_amount + atm_amount

-- Amount spent
def amount_spent : ℕ := total_before_supermarket - after_supermarket

-- Proof that Olivia spent 89 dollars
theorem olivia_spent_89 : amount_spent = 89 := sorry

end NUMINAMATH_GPT_olivia_spent_89_l292_29245


namespace NUMINAMATH_GPT_total_fish_caught_l292_29256

-- Definitions based on conditions
def sums : List ℕ := [7, 9, 14, 14, 19, 21]

-- Statement of the proof problem
theorem total_fish_caught : 
  (∃ (a b c d : ℕ), [a+b, a+c, a+d, b+c, b+d, c+d] = sums) → 
  ∃ (a b c d : ℕ), a + b + c + d = 28 :=
by 
  sorry

end NUMINAMATH_GPT_total_fish_caught_l292_29256


namespace NUMINAMATH_GPT_h_j_h_of_3_l292_29294

def h (x : ℤ) : ℤ := 5 * x + 2
def j (x : ℤ) : ℤ := 3 * x + 4

theorem h_j_h_of_3 : h (j (h 3)) = 277 := by
  sorry

end NUMINAMATH_GPT_h_j_h_of_3_l292_29294


namespace NUMINAMATH_GPT_explicit_x_n_formula_l292_29212

theorem explicit_x_n_formula (x y : ℕ → ℕ) (n : ℕ) :
  x 0 = 2 ∧ y 0 = 1 ∧
  (∀ n, x (n + 1) = x n ^ 2 + y n ^ 2) ∧
  (∀ n, y (n + 1) = 2 * x n * y n) →
  x n = (3 ^ (2 ^ n) + 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_explicit_x_n_formula_l292_29212


namespace NUMINAMATH_GPT_fraction_zero_implies_x_is_neg_2_l292_29220

theorem fraction_zero_implies_x_is_neg_2 {x : ℝ} 
  (h₁ : x^2 - 4 = 0)
  (h₂ : x^2 - 4 * x + 4 ≠ 0) 
  : x = -2 := 
by
  sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_is_neg_2_l292_29220


namespace NUMINAMATH_GPT_maximum_fraction_sum_l292_29211

noncomputable def max_fraction_sum (n : ℕ) (a b c d : ℕ) : ℝ :=
  1 - (1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1)))

theorem maximum_fraction_sum (n a b c d : ℕ) (h₀ : n > 1) (h₁ : a + c ≤ n) (h₂ : (a : ℚ) / b + (c : ℚ) / d < 1) :
  ∃ m : ℝ, m = max_fraction_sum n a b c d := by
  sorry

end NUMINAMATH_GPT_maximum_fraction_sum_l292_29211


namespace NUMINAMATH_GPT_inverse_implies_negation_l292_29227

-- Let's define p as a proposition
variable (p : Prop)

-- The inverse of a proposition p, typically the implication of not p implies not q
def inverse (p q : Prop) := ¬p → ¬q

-- The negation of a proposition p is just ¬p
def negation (p : Prop) := ¬p

-- The math problem statement. Prove that if the inverse of p is true, the negation of p is true.
theorem inverse_implies_negation (q : Prop) (h : inverse p q) : negation q := by
  sorry

end NUMINAMATH_GPT_inverse_implies_negation_l292_29227


namespace NUMINAMATH_GPT_average_speed_calculation_l292_29274

-- Define constants and conditions
def speed_swimming : ℝ := 1
def speed_running : ℝ := 6
def distance : ℝ := 1  -- We use a generic distance d = 1 (assuming normalized unit distance)

-- Proof statement
theorem average_speed_calculation :
  (2 * distance) / ((distance / speed_swimming) + (distance / speed_running)) = 12 / 7 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_calculation_l292_29274


namespace NUMINAMATH_GPT_exists_contiguous_figure_l292_29288

-- Definition of the type for different types of rhombuses
inductive RhombusType
| wide
| narrow

-- Definition of a figure composed of rhombuses
structure Figure where
  count_wide : ℕ
  count_narrow : ℕ
  connected : Prop

-- Statement of the proof problem
theorem exists_contiguous_figure : ∃ (f : Figure), f.count_wide = 3 ∧ f.count_narrow = 8 ∧ f.connected :=
sorry

end NUMINAMATH_GPT_exists_contiguous_figure_l292_29288


namespace NUMINAMATH_GPT_average_waiting_time_l292_29209

/-- 
A traffic light at a pedestrian crossing allows pedestrians to cross the street 
for one minute and prohibits crossing for two minutes. Prove that the average 
waiting time for a pedestrian who arrives at the intersection is 40 seconds.
-/ 
theorem average_waiting_time (pG : ℝ) (pR : ℝ) (eTG : ℝ) (eTR : ℝ) (cycle : ℝ) :
  pG = 1 / 3 ∧ pR = 2 / 3 ∧ eTG = 0 ∧ eTR = 1 ∧ cycle = 3 → 
  (eTG * pG + eTR * pR) * (60 / cycle) = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_waiting_time_l292_29209


namespace NUMINAMATH_GPT_wolf_and_nobel_prize_laureates_l292_29202

-- Definitions from the conditions
def num_total_scientists : ℕ := 50
def num_wolf_prize_laureates : ℕ := 31
def num_nobel_prize_laureates : ℕ := 29
def num_no_wolf_prize_and_yes_nobel := 3 -- N_W = N_W'
def num_without_wolf_or_nobel : ℕ := num_total_scientists - num_wolf_prize_laureates - 11 -- Derived from N_W' 

-- The statement to be proved
theorem wolf_and_nobel_prize_laureates :
  ∃ W_N, W_N = num_nobel_prize_laureates - (19 - 3) ∧ W_N = 18 :=
  by
    sorry

end NUMINAMATH_GPT_wolf_and_nobel_prize_laureates_l292_29202


namespace NUMINAMATH_GPT_imaginary_part_of_z_l292_29298

open Complex

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : 
  im ((x + I) / (y - I)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l292_29298


namespace NUMINAMATH_GPT_math_proof_l292_29270

noncomputable def problem (a b : ℝ) : Prop :=
  a - b = 2 ∧ a^2 + b^2 = 25 → a * b = 10.5

-- We state the problem as a theorem:
theorem math_proof (a b : ℝ) (h1: a - b = 2) (h2: a^2 + b^2 = 25) : a * b = 10.5 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_math_proof_l292_29270


namespace NUMINAMATH_GPT_sin_A_and_height_on_AB_l292_29239

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end NUMINAMATH_GPT_sin_A_and_height_on_AB_l292_29239


namespace NUMINAMATH_GPT_pie_eating_contest_l292_29296

theorem pie_eating_contest :
  let first_student_round1 := (5 : ℚ) / 6
  let first_student_round2 := (1 : ℚ) / 6
  let second_student_total := (2 : ℚ) / 3
  let first_student_total := first_student_round1 + first_student_round2
  first_student_total - second_student_total = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_pie_eating_contest_l292_29296


namespace NUMINAMATH_GPT_betty_age_l292_29210

def ages (A M B : ℕ) : Prop :=
  A = 2 * M ∧ A = 4 * B ∧ M = A - 22

theorem betty_age (A M B : ℕ) : ages A M B → B = 11 :=
by
  sorry

end NUMINAMATH_GPT_betty_age_l292_29210


namespace NUMINAMATH_GPT_balls_in_rightmost_box_l292_29290

theorem balls_in_rightmost_box (a : ℕ → ℕ)
  (h₀ : a 1 = 7)
  (h₁ : ∀ i, 1 ≤ i ∧ i ≤ 1990 → a i + a (i + 1) + a (i + 2) + a (i + 3) = 30) :
  a 1993 = 7 :=
sorry

end NUMINAMATH_GPT_balls_in_rightmost_box_l292_29290


namespace NUMINAMATH_GPT_find_k_l292_29233

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (0, 1)

-- Define the vector c involving variable k
variables (k : ℝ)
def vec_c : ℝ × ℝ := (k, -2)

-- Define the combined vector (a + 2b)
def combined_vec : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem to prove
theorem find_k (h : dot_product combined_vec (vec_c k) = 0) : k = 8 :=
by sorry

end NUMINAMATH_GPT_find_k_l292_29233


namespace NUMINAMATH_GPT_train_distance_proof_l292_29263

theorem train_distance_proof (c₁ c₂ c₃ : ℝ) : 
  (5 / c₁ + 5 / c₂ = 15) →
  (5 / c₂ + 5 / c₃ = 11) →
  ∀ (x : ℝ), (x / c₁ = 10 / c₂ + (10 + x) / c₃) →
  x = 27.5 := 
by
  sorry

end NUMINAMATH_GPT_train_distance_proof_l292_29263


namespace NUMINAMATH_GPT_circumradius_relationship_l292_29215

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end NUMINAMATH_GPT_circumradius_relationship_l292_29215


namespace NUMINAMATH_GPT_find_a_l292_29258

open Set

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) (h : (A ∪ B a) ⊆ (A ∩ B a)) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l292_29258


namespace NUMINAMATH_GPT_total_shingles_needed_l292_29200

-- Defining the dimensions of the house and the porch
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_length : ℝ := 6
def porch_width : ℝ := 4.5

-- The goal is to prove that the total area of the shingles needed is 232 square feet
theorem total_shingles_needed :
  (house_length * house_width) + (porch_length * porch_width) = 232 := by
  sorry

end NUMINAMATH_GPT_total_shingles_needed_l292_29200


namespace NUMINAMATH_GPT_second_candidate_extra_marks_l292_29279

theorem second_candidate_extra_marks (T : ℝ) (marks_40_percent : ℝ) (marks_passing : ℝ) (marks_60_percent : ℝ) 
  (h1 : marks_40_percent = 0.40 * T)
  (h2 : marks_passing = 160)
  (h3 : marks_60_percent = 0.60 * T)
  (h4 : marks_passing = marks_40_percent + 40) :
  (marks_60_percent - marks_passing) = 20 :=
by
  sorry

end NUMINAMATH_GPT_second_candidate_extra_marks_l292_29279


namespace NUMINAMATH_GPT_pies_sold_in_week_l292_29237

def daily_pies : ℕ := 8
def days_in_week : ℕ := 7

theorem pies_sold_in_week : daily_pies * days_in_week = 56 := by
  sorry

end NUMINAMATH_GPT_pies_sold_in_week_l292_29237


namespace NUMINAMATH_GPT_expectedValueProof_l292_29276

-- Definition of the problem conditions
def veryNormalCoin {n : ℕ} : Prop :=
  ∀ t : ℕ, (5 < t → (t - 5) = n → (t+1 = t + 1)) ∧ (t ≤ 5 ∨ n = t)

-- Definition of the expected value calculation
def expectedValue (n : ℕ) : ℚ :=
  if n > 0 then (1/2)^n else 0

-- Expected value for the given problem
def expectedValueProblem : ℚ := 
  let a1 := -2/683
  let expectedFirstFlip := 1/2 - 1/(2 * 683)
  100 * 341 + 683

-- Main statement to prove
theorem expectedValueProof : expectedValueProblem = 34783 := 
  sorry -- Proof omitted

end NUMINAMATH_GPT_expectedValueProof_l292_29276


namespace NUMINAMATH_GPT_solveTheaterProblem_l292_29223

open Nat

def theaterProblem : Prop :=
  ∃ (A C : ℕ), (A + C = 80) ∧ (12 * A + 5 * C = 519) ∧ (C = 63)

theorem solveTheaterProblem : theaterProblem :=
  by
  sorry

end NUMINAMATH_GPT_solveTheaterProblem_l292_29223


namespace NUMINAMATH_GPT_incorrect_method_D_l292_29277

-- Conditions definitions
def conditionA (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus ↔ cond p)

def conditionB (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

def conditionC (locus : Set α) (cond : α → Prop) :=
  ∀ p, (¬ (p ∈ locus) ↔ ¬ (cond p))

def conditionD (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus → cond p) ∧ (∃ p, cond p ∧ ¬ (p ∈ locus))

def conditionE (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

-- Main theorem
theorem incorrect_method_D {α : Type} (locus : Set α) (cond : α → Prop) :
  conditionD locus cond →
  ¬ (conditionA locus cond) ∧
  ¬ (conditionB locus cond) ∧
  ¬ (conditionC locus cond) ∧
  ¬ (conditionE locus cond) :=
  sorry

end NUMINAMATH_GPT_incorrect_method_D_l292_29277


namespace NUMINAMATH_GPT_simplify_polynomial_l292_29229

theorem simplify_polynomial :
  (6 * p ^ 4 + 2 * p ^ 3 - 8 * p + 9) + (-3 * p ^ 3 + 7 * p ^ 2 - 5 * p - 1) = 
  6 * p ^ 4 - p ^ 3 + 7 * p ^ 2 - 13 * p + 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l292_29229


namespace NUMINAMATH_GPT_power_of_xy_l292_29240

-- Problem statement: Given a condition on x and y, find x^y.
theorem power_of_xy (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 13 = 0) : x^y = -8 :=
by {
  -- Proof will be added here
  sorry
}

end NUMINAMATH_GPT_power_of_xy_l292_29240


namespace NUMINAMATH_GPT_soda_cost_l292_29261

variable (b s : ℕ)

theorem soda_cost (h1 : 2 * b + s = 210) (h2 : b + 2 * s = 240) : s = 90 := by
  sorry

end NUMINAMATH_GPT_soda_cost_l292_29261


namespace NUMINAMATH_GPT_nick_total_quarters_l292_29205

theorem nick_total_quarters (Q : ℕ)
  (h1 : 2 / 5 * Q = state_quarters)
  (h2 : 1 / 2 * state_quarters = PA_quarters)
  (h3 : PA_quarters = 7) :
  Q = 35 := by
  sorry

end NUMINAMATH_GPT_nick_total_quarters_l292_29205


namespace NUMINAMATH_GPT_inexperienced_sailors_count_l292_29283

theorem inexperienced_sailors_count
  (I E : ℕ)
  (h1 : I + E = 17)
  (h2 : ∀ (rate_inexperienced hourly_rate experienced_rate : ℕ), hourly_rate = 10 → experienced_rate = 12 → rate_inexperienced = 2400)
  (h3 : ∀ (total_income experienced_salary : ℕ), total_income = 34560 → experienced_salary = 2880)
  (h4 : ∀ (monthly_income : ℕ), monthly_income = 34560)
  : I = 5 := sorry

end NUMINAMATH_GPT_inexperienced_sailors_count_l292_29283


namespace NUMINAMATH_GPT_multiples_of_4_between_88_and_104_l292_29275

theorem multiples_of_4_between_88_and_104 : 
  ∃ n, (104 - 4 * 23 = n) ∧ n = 88 ∧ ( ∀ x, (x ≥ 88 ∧ x ≤ 104 ∧ x % 4 = 0) → ( x - 88) / 4 < 24) :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_4_between_88_and_104_l292_29275


namespace NUMINAMATH_GPT_tripling_base_exponent_l292_29260

variables (a b x : ℝ)

theorem tripling_base_exponent (b_ne_zero : b ≠ 0) (r_def : (3 * a)^(3 * b) = a^b * x^b) : x = 27 * a^2 :=
by
  -- Proof omitted as requested
  sorry

end NUMINAMATH_GPT_tripling_base_exponent_l292_29260


namespace NUMINAMATH_GPT_inequality_proof_l292_29248

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a + 1 / b + 9 / c + 25 / d) ≥ (100 / (a + b + c + d)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l292_29248


namespace NUMINAMATH_GPT_kitchen_width_l292_29281

theorem kitchen_width (length : ℕ) (height : ℕ) (rate : ℕ) (hours : ℕ) (coats : ℕ) 
  (total_painted : ℕ) (half_walls_area : ℕ) (total_walls_area : ℕ)
  (width : ℕ) : 
  length = 12 ∧ height = 10 ∧ rate = 40 ∧ hours = 42 ∧ coats = 3 ∧ 
  total_painted = rate * hours ∧ total_painted = coats * total_walls_area ∧
  half_walls_area = 2 * length * height ∧ total_walls_area = half_walls_area + 2 * width * height ∧
  2 * (total_walls_area - half_walls_area / 2) = 2 * width * height →
  width = 16 := 
by
  sorry

end NUMINAMATH_GPT_kitchen_width_l292_29281


namespace NUMINAMATH_GPT_min_value_inequality_l292_29268

open Real

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 3 * a + 2 * b + c ≥ 18 := 
sorry

end NUMINAMATH_GPT_min_value_inequality_l292_29268


namespace NUMINAMATH_GPT_calculate_fraction_l292_29267

theorem calculate_fraction :
  (-1 / 42) / (1 / 6 - 3 / 14 + 2 / 3 - 2 / 7) = -1 / 14 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l292_29267


namespace NUMINAMATH_GPT_find_y_value_l292_29282

theorem find_y_value (x y : ℝ) (k : ℝ) 
  (h1 : 5 * y = k / x^2)
  (h2 : y = 4)
  (h3 : x = 2)
  (h4 : k = 80) :
  ( ∃ y : ℝ, 5 * y = k / 4^2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l292_29282


namespace NUMINAMATH_GPT_cannot_form_62_cents_with_six_coins_l292_29228

-- Define the coin denominations and their values
structure Coin :=
  (value : ℕ)
  (count : ℕ)

def penny : Coin := ⟨1, 6⟩
def nickel : Coin := ⟨5, 6⟩
def dime : Coin := ⟨10, 6⟩
def quarter : Coin := ⟨25, 6⟩
def halfDollar : Coin := ⟨50, 6⟩

-- Define the main theorem statement
theorem cannot_form_62_cents_with_six_coins :
  ¬ (∃ (p n d q h : ℕ),
      p + n + d + q + h = 6 ∧
      1 * p + 5 * n + 10 * d + 25 * q + 50 * h = 62) :=
sorry

end NUMINAMATH_GPT_cannot_form_62_cents_with_six_coins_l292_29228


namespace NUMINAMATH_GPT_Maria_green_towels_l292_29216

-- Definitions
variable (G : ℕ) -- number of green towels

-- Conditions
def initial_towels := G + 21
def final_towels := initial_towels - 34

-- Theorem statement
theorem Maria_green_towels : final_towels = 22 → G = 35 :=
by
  sorry

end NUMINAMATH_GPT_Maria_green_towels_l292_29216


namespace NUMINAMATH_GPT_max_value_ineq_l292_29244

theorem max_value_ineq (x y : ℝ) (h : x^2 + y^2 = 20) : xy + 8*x + y ≤ 42 := by
  sorry

end NUMINAMATH_GPT_max_value_ineq_l292_29244


namespace NUMINAMATH_GPT_prove_a_is_perfect_square_l292_29284

-- Definition of a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Main theorem statement
theorem prove_a_is_perfect_square 
  (a b : ℕ) 
  (hb_odd : b % 2 = 1) 
  (h_integer : ∃ k : ℕ, ((a + b) * (a + b) + 4 * a) = k * a * b) :
  is_perfect_square a :=
sorry

end NUMINAMATH_GPT_prove_a_is_perfect_square_l292_29284


namespace NUMINAMATH_GPT_solve_quadratic_l292_29226

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) := by
  intro x
  sorry

end NUMINAMATH_GPT_solve_quadratic_l292_29226


namespace NUMINAMATH_GPT_find_N_sum_e_l292_29299

theorem find_N_sum_e (N : ℝ) (e1 e2 : ℝ) :
  (2 * abs (2 - e1) = N) ∧
  (2 * abs (2 - e2) = N) ∧
  (e1 ≠ e2) ∧
  (e1 + e2 = 4) →
  N = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_N_sum_e_l292_29299


namespace NUMINAMATH_GPT_seedlings_total_l292_29213

theorem seedlings_total (seeds_per_packet : ℕ) (packets : ℕ) (total_seedlings : ℕ) 
  (h1 : seeds_per_packet = 7) (h2 : packets = 60) : total_seedlings = 420 :=
by {
  sorry
}

end NUMINAMATH_GPT_seedlings_total_l292_29213


namespace NUMINAMATH_GPT_find_n_l292_29273

def valid_n (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [MOD 15]

theorem find_n : ∃ n, valid_n n ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l292_29273


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l292_29207
-- Import the necessary libraries

-- Define the real number condition
def real_number (a : ℝ) : Prop := true

-- Define line l1
def line_l1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define line l2
def line_l2 (a y x: ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel condition
def parallel_lines (a : ℝ) : Prop :=
  (a = 2 ∨ a = -2) ∧ 
  ∀ x y : ℝ, line_l1 a x y ∧ line_l2 a x y → a * x + 4 * x + 6 = 3

-- State the main theorem to prove
theorem necessary_not_sufficient_condition (a : ℝ) : 
  real_number a → (a = 2 ∨ a = -2) ↔ (parallel_lines a) := 
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l292_29207


namespace NUMINAMATH_GPT_trig_identity_l292_29204

variable (α : ℝ)
variable (h : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (h₁ : Real.sin α = 4 / 5)

theorem trig_identity : Real.sin (α + Real.pi / 4) + Real.cos (α + Real.pi / 4) = -3 * Real.sqrt 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l292_29204


namespace NUMINAMATH_GPT_sum_of_cubes_of_roots_l292_29289

theorem sum_of_cubes_of_roots (r1 r2 r3 : ℂ) (h1 : r1 + r2 + r3 = 3) (h2 : r1 * r2 + r1 * r3 + r2 * r3 = 0) (h3 : r1 * r2 * r3 = -1) : 
  r1^3 + r2^3 + r3^3 = 24 :=
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_roots_l292_29289


namespace NUMINAMATH_GPT_third_person_fraction_removed_l292_29206

-- Define the number of teeth for each person and the fractions that are removed
def total_teeth := 32
def total_removed := 40

def first_person_removed := (1 / 4) * total_teeth
def second_person_removed := (3 / 8) * total_teeth
def fourth_person_removed := 4

-- Define the total teeth removed by the first, second, and fourth persons
def known_removed := first_person_removed + second_person_removed + fourth_person_removed

-- Define the total teeth removed by the third person
def third_person_removed := total_removed - known_removed

-- Prove that the third person had 1/2 of his teeth removed
theorem third_person_fraction_removed :
  third_person_removed / total_teeth = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_third_person_fraction_removed_l292_29206


namespace NUMINAMATH_GPT_find_principal_l292_29295

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h₁ : SI = 8625) (h₂ : R = 50 / 3) (h₃ : T = 3 / 4) :
  SI = (P * R * T) / 100 → P = 69000 := sorry

end NUMINAMATH_GPT_find_principal_l292_29295


namespace NUMINAMATH_GPT_find_a6_l292_29236

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end NUMINAMATH_GPT_find_a6_l292_29236


namespace NUMINAMATH_GPT_temperature_reading_l292_29203

theorem temperature_reading (scale_min scale_max : ℝ) (arrow : ℝ) (h1 : scale_min = -6.0) (h2 : scale_max = -5.5) (h3 : scale_min < arrow) (h4 : arrow < scale_max) : arrow = -5.7 :=
sorry

end NUMINAMATH_GPT_temperature_reading_l292_29203


namespace NUMINAMATH_GPT_simplify_expression_l292_29297

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l292_29297


namespace NUMINAMATH_GPT_slope_line_point_l292_29232

theorem slope_line_point (m b : ℝ) (h_slope : m = 3) (h_point : 2 = m * 5 + b) : m + b = -10 :=
by
  sorry

end NUMINAMATH_GPT_slope_line_point_l292_29232


namespace NUMINAMATH_GPT_coat_price_reduction_l292_29286

theorem coat_price_reduction 
    (original_price : ℝ) 
    (reduction_amount : ℝ) 
    (h1 : original_price = 500) 
    (h2 : reduction_amount = 300) : 
    (reduction_amount / original_price) * 100 = 60 := 
by 
  sorry

end NUMINAMATH_GPT_coat_price_reduction_l292_29286


namespace NUMINAMATH_GPT_range_of_m_no_zeros_inequality_when_m_zero_l292_29234

-- Statement for Problem 1
theorem range_of_m_no_zeros (m : ℝ) (h : ∀ x : ℝ, (x^2 + m * x + m) * Real.exp x ≠ 0) : 0 < m ∧ m < 4 :=
sorry

-- Statement for Problem 2
theorem inequality_when_m_zero (x : ℝ) : 
  (x^2) * (Real.exp x) ≥ x^2 + x^3 :=
sorry

end NUMINAMATH_GPT_range_of_m_no_zeros_inequality_when_m_zero_l292_29234


namespace NUMINAMATH_GPT_scientific_notation_of_19672_l292_29292

theorem scientific_notation_of_19672 :
  ∃ a b, 19672 = a * 10^b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.9672 ∧ b = 4 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_19672_l292_29292


namespace NUMINAMATH_GPT_probability_of_matching_pair_l292_29262

theorem probability_of_matching_pair (blackSocks blueSocks : ℕ) (h_black : blackSocks = 12) (h_blue : blueSocks = 10) : 
  let totalSocks := blackSocks + blueSocks
  let totalWays := Nat.choose totalSocks 2
  let blackPairWays := Nat.choose blackSocks 2
  let bluePairWays := Nat.choose blueSocks 2
  let matchingPairWays := blackPairWays + bluePairWays
  totalWays = 231 ∧ matchingPairWays = 111 → (matchingPairWays : ℚ) / totalWays = 111 / 231 := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_of_matching_pair_l292_29262


namespace NUMINAMATH_GPT_smallest_base_conversion_l292_29224

theorem smallest_base_conversion :
  let n1 := 8 * 9 + 5 -- 85 in base 9
  let n2 := 2 * 6^2 + 1 * 6 -- 210 in base 6
  let n3 := 1 * 4^3 -- 1000 in base 4
  let n4 := 1 * 2^7 - 1 -- 1111111 in base 2
  n3 < n1 ∧ n3 < n2 ∧ n3 < n4 :=
by
  let n1 := 8 * 9 + 5
  let n2 := 2 * 6^2 + 1 * 6
  let n3 := 1 * 4^3
  let n4 := 1 * 2^7 - 1
  sorry

end NUMINAMATH_GPT_smallest_base_conversion_l292_29224


namespace NUMINAMATH_GPT_part1_part2_l292_29235

-- Part 1
theorem part1 (n : ℕ) (hn : n ≠ 0) (d : ℕ) (hd : d ∣ 2 * n^2) : 
  ∀ m : ℕ, ¬ (m ≠ 0 ∧ m^2 = n^2 + d) :=
by
  sorry 

-- Part 2
theorem part2 (n : ℕ) (hn : n ≠ 0) : 
  ∀ d : ℕ, (d ∣ 3 * n^2 ∧ ∃ m : ℕ, m ≠ 0 ∧ m^2 = n^2 + d) → d = 3 * n^2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l292_29235


namespace NUMINAMATH_GPT_quadratic_other_root_l292_29255

theorem quadratic_other_root (a : ℝ) (h1 : ∃ (x : ℝ), x^2 - 2 * x + a = 0 ∧ x = -1) :
  ∃ (x2 : ℝ), x2^2 - 2 * x2 + a = 0 ∧ x2 = 3 :=
sorry

end NUMINAMATH_GPT_quadratic_other_root_l292_29255


namespace NUMINAMATH_GPT_scott_invests_l292_29280

theorem scott_invests (x r : ℝ) (h1 : 2520 = x + 1260) (h2 : 2520 * 0.08 = x * r) : r = 0.16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_scott_invests_l292_29280
