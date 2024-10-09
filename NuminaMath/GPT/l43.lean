import Mathlib

namespace book_arrangements_l43_4377

theorem book_arrangements :
  let math_books := 4
  let english_books := 4
  let groups := 2
  (groups.factorial) * (math_books.factorial) * (english_books.factorial) = 1152 :=
by
  sorry

end book_arrangements_l43_4377


namespace factorize_expression_l43_4359

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l43_4359


namespace slices_served_during_dinner_l43_4393

theorem slices_served_during_dinner (slices_lunch slices_total slices_dinner : ℕ)
  (h1 : slices_lunch = 7)
  (h2 : slices_total = 12)
  (h3 : slices_dinner = slices_total - slices_lunch) :
  slices_dinner = 5 := 
by 
  sorry

end slices_served_during_dinner_l43_4393


namespace pizzas_difference_l43_4328

def pizzas (craig_first_day craig_second_day heather_first_day heather_second_day total_pizzas: ℕ) :=
  heather_first_day = 4 * craig_first_day ∧
  heather_second_day = craig_second_day - 20 ∧
  craig_first_day = 40 ∧
  craig_first_day + heather_first_day + craig_second_day + heather_second_day = total_pizzas

theorem pizzas_difference :
  ∀ (craig_first_day craig_second_day heather_first_day heather_second_day : ℕ),
  pizzas craig_first_day craig_second_day heather_first_day heather_second_day 380 →
  craig_second_day - craig_first_day = 60 :=
by
  intros craig_first_day craig_second_day heather_first_day heather_second_day h
  sorry

end pizzas_difference_l43_4328


namespace n_divisible_by_6_l43_4371

open Int -- Open integer namespace for convenience

theorem n_divisible_by_6 (m n : ℤ)
    (h1 : ∃ (a b : ℤ), a + b = -m ∧ a * b = -n)
    (h2 : ∃ (c d : ℤ), c + d = m ∧ c * d = n) :
    6 ∣ n := 
sorry

end n_divisible_by_6_l43_4371


namespace total_money_spent_l43_4349

def candy_bar_cost : ℕ := 14
def cookie_box_cost : ℕ := 39
def total_spent : ℕ := candy_bar_cost + cookie_box_cost

theorem total_money_spent : total_spent = 53 := by
  sorry

end total_money_spent_l43_4349


namespace find_value_l43_4307

-- Define the mean, standard deviation, and the number of standard deviations
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5
def num_std_dev : ℝ := 2.7

-- The theorem to prove that the value is exactly 10.75
theorem find_value : mean - (num_std_dev * std_dev) = 10.75 := 
by
  sorry

end find_value_l43_4307


namespace commercials_count_l43_4300

-- Given conditions as definitions
def total_airing_time : ℤ := 90         -- 1.5 hours in minutes
def commercial_time : ℤ := 10           -- each commercial lasts 10 minutes
def show_time : ℤ := 60                 -- TV show (without commercials) lasts 60 minutes

-- Statement: Prove that the number of commercials is 3
theorem commercials_count :
  (total_airing_time - show_time) / commercial_time = 3 :=
sorry

end commercials_count_l43_4300


namespace geometric_to_arithmetic_l43_4373

theorem geometric_to_arithmetic {a1 a2 a3 a4 q : ℝ}
  (hq : q ≠ 1)
  (geom_seq : a2 = a1 * q ∧ a3 = a1 * q^2 ∧ a4 = a1 * q^3)
  (arith_seq : (2 * a3 = a1 + a4 ∨ 2 * a2 = a1 + a4)) :
  q = (1 + Real.sqrt 5) / 2 ∨ q = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_to_arithmetic_l43_4373


namespace quotient_of_sum_of_remainders_div_16_eq_0_l43_4341

-- Define the set of distinct remainders of squares modulo 16 for n in 1 to 15
def distinct_remainders_mod_16 : Finset ℕ :=
  {1, 4, 9, 0}

-- Define the sum of the distinct remainders
def sum_of_remainders : ℕ :=
  distinct_remainders_mod_16.sum id

-- Proposition to prove the quotient when sum_of_remainders is divided by 16 is 0
theorem quotient_of_sum_of_remainders_div_16_eq_0 :
  (sum_of_remainders / 16) = 0 :=
by
  sorry

end quotient_of_sum_of_remainders_div_16_eq_0_l43_4341


namespace cone_height_ratio_l43_4308

theorem cone_height_ratio (circumference : ℝ) (orig_height : ℝ) (short_volume : ℝ)
  (h_circumference : circumference = 20 * Real.pi)
  (h_orig_height : orig_height = 40)
  (h_short_volume : short_volume = 400 * Real.pi) :
  let r := circumference / (2 * Real.pi)
  let h_short := (3 * short_volume) / (Real.pi * r^2)
  (h_short / orig_height) = 3 / 10 :=
by {
  sorry
}

end cone_height_ratio_l43_4308


namespace ratio_cubed_eq_27_l43_4368

theorem ratio_cubed_eq_27 : (81000^3) / (27000^3) = 27 := 
by
  sorry

end ratio_cubed_eq_27_l43_4368


namespace sum_of_ages_l43_4362

theorem sum_of_ages (a b c d e : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 9) 
  (h2 : 1 ≤ b ∧ b ≤ 9) 
  (h3 : 1 ≤ c ∧ c ≤ 9) 
  (h4 : 1 ≤ d ∧ d ≤ 9) 
  (h5 : 1 ≤ e ∧ e ≤ 9) 
  (h6 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h7 : a * b = 28 ∨ a * c = 28 ∨ a * d = 28 ∨ a * e = 28 ∨ b * c = 28 ∨ b * d = 28 ∨ b * e = 28 ∨ c * d = 28 ∨ c * e = 28 ∨ d * e = 28)
  (h8 : a * b = 20 ∨ a * c = 20 ∨ a * d = 20 ∨ a * e = 20 ∨ b * c = 20 ∨ b * d = 20 ∨ b * e = 20 ∨ c * d = 20 ∨ c * e = 20 ∨ d * e = 20)
  (h9 : a + b = 14 ∨ a + c = 14 ∨ a + d = 14 ∨ a + e = 14 ∨ b + c = 14 ∨ b + d = 14 ∨ b + e = 14 ∨ c + d = 14 ∨ c + e = 14 ∨ d + e = 14) 
  : a + b + c + d + e = 25 :=
by
  sorry

end sum_of_ages_l43_4362


namespace distance_to_nearest_edge_l43_4310

theorem distance_to_nearest_edge (wall_width picture_width : ℕ) (h1 : wall_width = 19) (h2 : picture_width = 3) (h3 : 2 * x + picture_width = wall_width) :
  x = 8 :=
by
  sorry

end distance_to_nearest_edge_l43_4310


namespace shift_graph_to_right_l43_4379

theorem shift_graph_to_right (x : ℝ) : 
  4 * Real.cos (2 * x + π / 4) = 4 * Real.cos (2 * (x - π / 8) + π / 4) :=
by 
  -- sketch of the intended proof without actual steps for clarity
  sorry

end shift_graph_to_right_l43_4379


namespace value_of_b_l43_4338

theorem value_of_b (a c : ℝ) (b : ℝ) (h1 : a = 105) (h2 : c = 70) (h3 : a^4 = 21 * 25 * 15 * b * c^3) : b = 0.045 :=
by
  sorry

end value_of_b_l43_4338


namespace popsicle_total_l43_4365

def popsicle_count (g c b : Nat) : Nat :=
  g + c + b

theorem popsicle_total : 
  let g := 2
  let c := 13
  let b := 2
  popsicle_count g c b = 17 := by
  sorry

end popsicle_total_l43_4365


namespace range_of_x_l43_4372

theorem range_of_x (x : ℝ) :
  (∀ y : ℝ, 0 < y → y^2 + (2*x - 5)*y - x^2 * (Real.log x - Real.log y) ≤ 0) ↔ x = 5 / 2 :=
by 
  sorry

end range_of_x_l43_4372


namespace sequence_term_a_1000_eq_2340_l43_4325

theorem sequence_term_a_1000_eq_2340
  (a : ℕ → ℤ)
  (h1 : a 1 = 2007)
  (h2 : a 2 = 2008)
  (h_rec : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = n) :
  a 1000 = 2340 :=
sorry

end sequence_term_a_1000_eq_2340_l43_4325


namespace cos_double_angle_from_sin_shift_l43_4319

theorem cos_double_angle_from_sin_shift (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := 
by 
  sorry

end cos_double_angle_from_sin_shift_l43_4319


namespace career_preference_angles_l43_4355

theorem career_preference_angles (m f : ℕ) (total_degrees : ℕ) (one_fourth_males one_half_females : ℚ) (male_ratio female_ratio : ℚ) :
  total_degrees = 360 → male_ratio = 2/3 → female_ratio = 3/3 →
  m = 2 * f / 3 → one_fourth_males = 1/4 * m → one_half_females = 1/2 * f →
  (one_fourth_males + one_half_females) / (m + f) * total_degrees = 144 :=
by
  sorry

end career_preference_angles_l43_4355


namespace P_of_7_l43_4303

noncomputable def P (x : ℝ) : ℝ := 12 * (x - 1) * (x - 2) * (x - 3) * (x - 4)^2 * (x - 5)^2 * (x - 6)

theorem P_of_7 : P 7 = 51840 :=
by
  sorry

end P_of_7_l43_4303


namespace cost_of_each_green_hat_l43_4337

theorem cost_of_each_green_hat
  (total_hats : ℕ) (cost_blue_hat : ℕ) (total_price : ℕ) (green_hats : ℕ) (blue_hats : ℕ) (cost_green_hat : ℕ)
  (h1 : total_hats = 85) 
  (h2 : cost_blue_hat = 6) 
  (h3 : total_price = 550) 
  (h4 : green_hats = 40) 
  (h5 : blue_hats = 45) 
  (h6 : green_hats + blue_hats = total_hats) 
  (h7 : total_price = green_hats * cost_green_hat + blue_hats * cost_blue_hat) :
  cost_green_hat = 7 := 
sorry

end cost_of_each_green_hat_l43_4337


namespace selection_of_projects_l43_4323

-- Mathematical definitions
def numberOfWaysToSelect2ProjectsFrom4KeyAnd6General (key: Finset ℕ) (general: Finset ℕ) : ℕ :=
  (key.card.choose 2) * (general.card.choose 2)

def numberOfWaysToSelectAtLeastOneProjectAorB (key: Finset ℕ) (general: Finset ℕ) (A B: ℕ) : ℕ :=
  let total_ways := (key.card.choose 2) * (general.card.choose 2)
  let ways_without_A := ((key.erase A).card.choose 2) * (general.card.choose 2)
  let ways_without_B := (key.card.choose 2) * ((general.erase B).card.choose 2)
  let ways_without_A_and_B := ((key.erase A).card.choose 2) * ((general.erase B).card.choose 2)
  total_ways - ways_without_A_and_B

-- Theorem we need to prove
theorem selection_of_projects (key general: Finset ℕ) (A B: ℕ) (hA: A ∈ key) (hB: B ∈ general) (h_key_card: key.card = 4) (h_general_card: general.card = 6) :
  numberOfWaysToSelectAtLeastOneProjectAorB key general A B = 60 := 
sorry

end selection_of_projects_l43_4323


namespace tiles_needed_l43_4327

-- Definitions of the given conditions
def side_length_smaller_tile : ℝ := 0.3
def number_smaller_tiles : ℕ := 500
def side_length_larger_tile : ℝ := 0.5

-- Statement to prove the required number of larger tiles
theorem tiles_needed (x : ℕ) :
  side_length_larger_tile * side_length_larger_tile * x =
  side_length_smaller_tile * side_length_smaller_tile * number_smaller_tiles →
  x = 180 :=
by
  sorry

end tiles_needed_l43_4327


namespace equation1_solution_equation2_solution_l43_4332

theorem equation1_solution (x : ℝ) (h : 3 * x - 1 = x + 7) : x = 4 := by
  sorry

theorem equation2_solution (x : ℝ) (h : (x + 1) / 2 - 1 = (1 - 2 * x) / 3) : x = 5 / 7 := by
  sorry

end equation1_solution_equation2_solution_l43_4332


namespace total_jelly_beans_l43_4339

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l43_4339


namespace convert_cylindrical_to_rectangular_l43_4344

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 7 (Real.pi / 4) 8 = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2, 8) :=
by
  sorry

end convert_cylindrical_to_rectangular_l43_4344


namespace quadratic_two_distinct_real_roots_l43_4364

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ - k^2 = 0) ∧ (x₂^2 - x₂ - k^2 = 0) :=
by
  -- The proof is omitted as requested.
  sorry

end quadratic_two_distinct_real_roots_l43_4364


namespace number_of_ways_to_choose_officers_l43_4342

-- Define the number of boys and girls.
def num_boys : ℕ := 12
def num_girls : ℕ := 13

-- Define the total number of boys and girls.
def num_members : ℕ := num_boys + num_girls

-- Calculate the number of ways to choose the president, vice-president, and secretary with given conditions.
theorem number_of_ways_to_choose_officers : 
  (num_boys * num_girls * (num_boys - 1)) + (num_girls * num_boys * (num_girls - 1)) = 3588 :=
by
  -- The first part calculates the ways when the president is a boy.
  -- The second part calculates the ways when the president is a girl.
  sorry

end number_of_ways_to_choose_officers_l43_4342


namespace birds_in_trees_l43_4392

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_l43_4392


namespace root_quad_eqn_l43_4320

theorem root_quad_eqn (a : ℝ) (h : a^2 - a - 50 = 0) : a^3 - 51 * a = 50 :=
sorry

end root_quad_eqn_l43_4320


namespace total_bill_l43_4330

theorem total_bill (n : ℝ) (h : 9 * (n / 10 + 3) = n) : n = 270 := 
sorry

end total_bill_l43_4330


namespace jazmin_dolls_correct_l43_4382

-- Define the number of dolls Geraldine has.
def geraldine_dolls : ℕ := 2186

-- Define the number of extra dolls Geraldine has compared to Jazmin.
def extra_dolls : ℕ := 977

-- Define the calculation of the number of dolls Jazmin has.
def jazmin_dolls : ℕ := geraldine_dolls - extra_dolls

-- Prove that the number of dolls Jazmin has is 1209.
theorem jazmin_dolls_correct : jazmin_dolls = 1209 := by
  -- Include the required steps in the future proof here.
  sorry

end jazmin_dolls_correct_l43_4382


namespace trapezoid_median_l43_4347

theorem trapezoid_median
  (h : ℝ)
  (area_triangle : ℝ)
  (area_trapezoid : ℝ)
  (bt : ℝ)
  (bt_sum : ℝ)
  (ht_positive : h ≠ 0)
  (triangle_area : area_triangle = (1/2) * bt * h)
  (trapezoid_area : area_trapezoid = area_triangle)
  (trapezoid_bt_sum : bt_sum = 40)
  (triangle_bt : bt = 24)
  : (bt_sum / 2) = 20 :=
by
  sorry

end trapezoid_median_l43_4347


namespace circle_tangent_x_axis_at_origin_l43_4318

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → (∃ r : ℝ, r^2 = x^2 + y^2) ∧ y = 0) →
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 := 
sorry

end circle_tangent_x_axis_at_origin_l43_4318


namespace total_expenditure_of_Louis_l43_4394

def fabric_cost (yards price_per_yard : ℕ) : ℕ :=
  yards * price_per_yard

def thread_cost (spools price_per_spool : ℕ) : ℕ :=
  spools * price_per_spool

def total_cost (yards price_per_yard pattern_cost spools price_per_spool : ℕ) : ℕ :=
  fabric_cost yards price_per_yard + pattern_cost + thread_cost spools price_per_spool

theorem total_expenditure_of_Louis :
  total_cost 5 24 15 2 3 = 141 :=
by
  sorry

end total_expenditure_of_Louis_l43_4394


namespace michael_wants_to_buy_more_packs_l43_4395

theorem michael_wants_to_buy_more_packs
  (initial_packs : ℕ)
  (cost_per_pack : ℝ)
  (total_value_after_purchase : ℝ)
  (value_of_current_packs : ℝ := initial_packs * cost_per_pack)
  (additional_value_needed : ℝ := total_value_after_purchase - value_of_current_packs)
  (packs_to_buy : ℝ := additional_value_needed / cost_per_pack)
  (answer : ℕ := 2) :
  initial_packs = 4 → cost_per_pack = 2.5 → total_value_after_purchase = 15 → packs_to_buy = answer :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end michael_wants_to_buy_more_packs_l43_4395


namespace product_of_symmetric_complex_numbers_l43_4363

def z1 : ℂ := 1 + 2 * Complex.I

def z2 : ℂ := -1 + 2 * Complex.I

theorem product_of_symmetric_complex_numbers :
  z1 * z2 = -5 :=
by 
  sorry

end product_of_symmetric_complex_numbers_l43_4363


namespace find_number_l43_4352

theorem find_number (x : ℝ) (h : (2 / 5) * x = 10) : x = 25 :=
sorry

end find_number_l43_4352


namespace inequality_correct_transformation_l43_4316

-- Definitions of the conditions
variables (a b : ℝ)

-- The equivalent proof problem
theorem inequality_correct_transformation (h : a > b) : -a < -b :=
by sorry

end inequality_correct_transformation_l43_4316


namespace min_lit_bulbs_l43_4397

theorem min_lit_bulbs (n : ℕ) (h : n ≥ 1) : 
  ∃ rows cols, (rows ⊆ Finset.range n) ∧ (cols ⊆ Finset.range n) ∧ 
  (∀ i j, (i ∈ rows ∧ j ∈ cols) ↔ (i + j) % 2 = 1) ∧ 
  rows.card * (n - cols.card) + cols.card * (n - rows.card) = 2 * n - 2 :=
by sorry

end min_lit_bulbs_l43_4397


namespace solve_equation_l43_4389

theorem solve_equation (x : ℂ) (h : (x^2 + 3*x + 4) / (x + 3) = x + 6) : x = -7 / 3 := sorry

end solve_equation_l43_4389


namespace charles_paints_l43_4375

-- Define the ratio and total work conditions
def ratio_a_to_c (a c : ℕ) := a * 6 = c * 2

def total_work (total : ℕ) := total = 320

-- Define the question, i.e., the amount of work Charles does
theorem charles_paints (a c total : ℕ) (h_ratio : ratio_a_to_c a c) (h_total : total_work total) : 
  (total / (a + c)) * c = 240 :=
by 
  -- We include sorry to indicate the need for proof here
  sorry

end charles_paints_l43_4375


namespace sequence_inequality_l43_4326

theorem sequence_inequality (a : ℕ → ℕ)
  (h1 : a 0 > 0) -- Ensure all entries are positive integers.
  (h2 : ∀ k l m n : ℕ, k * l = m * n → a k + a l = a m + a n)
  {p q : ℕ} (hpq : p ∣ q) :
  a p ≤ a q :=
sorry

end sequence_inequality_l43_4326


namespace point_on_line_and_in_first_quadrant_l43_4301

theorem point_on_line_and_in_first_quadrant (x y : ℝ) (hline : y = -2 * x + 3) (hfirst_quadrant : x > 0 ∧ y > 0) :
    (x, y) = (1, 1) :=
by
  sorry

end point_on_line_and_in_first_quadrant_l43_4301


namespace quadratic_equality_l43_4384

theorem quadratic_equality (a_2 : ℝ) (a_1 : ℝ) (a_0 : ℝ) (r : ℝ) (s : ℝ) (x : ℝ)
  (h₁ : a_2 ≠ 0)
  (h₂ : a_0 ≠ 0)
  (h₃ : a_2 * r^2 + a_1 * r + a_0 = 0)
  (h₄ : a_2 * s^2 + a_1 * s + a_0 = 0) :
  a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s) :=
by
  sorry

end quadratic_equality_l43_4384


namespace problem_solution_l43_4343

theorem problem_solution (s t : ℕ) (hpos_s : 0 < s) (hpos_t : 0 < t) (h_eq : s * (s - t) = 29) : s + t = 57 :=
by
  sorry

end problem_solution_l43_4343


namespace solution_set_l43_4311

noncomputable def f : ℝ → ℝ := sorry

axiom deriv_f_pos (x : ℝ) : deriv f x > 1 - f x
axiom f_at_zero : f 0 = 3

theorem solution_set (x : ℝ) : e^x * f x > e^x + 2 ↔ x > 0 :=
by sorry

end solution_set_l43_4311


namespace coefficient_x3_in_product_l43_4399

-- Definitions for the polynomials
def P(x : ℕ → ℕ) : ℕ → ℤ
| 4 => 3
| 3 => 4
| 2 => -2
| 1 => 8
| 0 => -5
| _ => 0

def Q(x : ℕ → ℕ) : ℕ → ℤ
| 3 => 2
| 2 => -7
| 1 => 5
| 0 => -3
| _ => 0

-- Statement of the problem
theorem coefficient_x3_in_product :
  (P 3 * Q 0 + P 2 * Q 1 + P 1 * Q 2) = -78 :=
by
  sorry

end coefficient_x3_in_product_l43_4399


namespace sqrt_expression_value_l43_4390

theorem sqrt_expression_value :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 :=
by
  sorry

end sqrt_expression_value_l43_4390


namespace no_solution_for_n_eq_neg2_l43_4398

theorem no_solution_for_n_eq_neg2 : ∀ (x y : ℝ), ¬ (2 * x = 1 + -2 * y ∧ -2 * x = 1 + 2 * y) :=
by sorry

end no_solution_for_n_eq_neg2_l43_4398


namespace smallest_three_digit_multiple_of_eleven_l43_4335

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end smallest_three_digit_multiple_of_eleven_l43_4335


namespace girls_in_school_l43_4391

theorem girls_in_school (boys girls : ℕ) (ratio : ℕ → ℕ → Prop) (h1 : ratio 5 4) (h2 : boys = 1500) :
    girls = 1200 :=
by
  sorry

end girls_in_school_l43_4391


namespace trigonometric_inequality_l43_4374

noncomputable def a : Real := (1/2) * Real.cos (8 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (8 * Real.pi / 180)
noncomputable def b : Real := (2 * Real.tan (14 * Real.pi / 180)) / (1 - (Real.tan (14 * Real.pi / 180))^2)
noncomputable def c : Real := Real.sqrt ((1 - Real.cos (48 * Real.pi / 180)) / 2)

theorem trigonometric_inequality :
  a < c ∧ c < b := by
  sorry

end trigonometric_inequality_l43_4374


namespace triangle_sides_angles_l43_4380

open Real

variables {a b c : ℝ} {α β γ : ℝ}

theorem triangle_sides_angles
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (angles_sum : α + β + γ = π)
  (condition : 3 * α + 2 * β = π) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_sides_angles_l43_4380


namespace shells_total_l43_4386

theorem shells_total (a s v : ℕ) 
  (h1 : s = v + 16) 
  (h2 : v = a - 5) 
  (h3 : a = 20) : 
  s + v + a = 66 := 
by
  sorry

end shells_total_l43_4386


namespace negation_statement_l43_4313

variable {α : Type} 
variable (student prepared : α → Prop)

theorem negation_statement :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by 
  -- proof will be provided here
  sorry

end negation_statement_l43_4313


namespace grocery_delivery_amount_l43_4361

theorem grocery_delivery_amount (initial_savings final_price trips : ℕ) 
(fixed_charge : ℝ) (percent_charge : ℝ) (total_saved : ℝ) : 
  initial_savings = 14500 →
  final_price = 14600 →
  trips = 40 →
  fixed_charge = 1.5 →
  percent_charge = 0.05 →
  total_saved = final_price - initial_savings →
  60 + percent_charge * G = total_saved →
  G = 800 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end grocery_delivery_amount_l43_4361


namespace range_of_a_l43_4317

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x ^ 2 + (a - 1) * x + 1 / 2 ≤ 0) → (-1 < a ∧ a < 3) :=
by 
  sorry

end range_of_a_l43_4317


namespace elena_bouquet_petals_l43_4360

def num_petals (count : ℕ) (petals_per_flower : ℕ) : ℕ :=
  count * petals_per_flower

theorem elena_bouquet_petals :
  let num_lilies := 4
  let lilies_petal_count := num_petals num_lilies 6
  
  let num_tulips := 2
  let tulips_petal_count := num_petals num_tulips 3

  let num_roses := 2
  let roses_petal_count := num_petals num_roses 5
  
  let num_daisies := 1
  let daisies_petal_count := num_petals num_daisies 12
  
  lilies_petal_count + tulips_petal_count + roses_petal_count + daisies_petal_count = 52 := by
  sorry

end elena_bouquet_petals_l43_4360


namespace real_solution_unique_l43_4381

variable (x : ℝ)

theorem real_solution_unique :
  (x ≠ 2 ∧ (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = 3) ↔ x = 1 := 
by 
  sorry

end real_solution_unique_l43_4381


namespace evaluate_expression_l43_4378

noncomputable def a := Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def b := -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def d := -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 3 / 50 :=
by
  sorry

end evaluate_expression_l43_4378


namespace sufficient_but_not_necessary_necessary_but_not_sufficient_l43_4354

def M (x : ℝ) : Prop := (x + 3) * (x - 5) > 0
def P (x : ℝ) (a : ℝ) : Prop := x^2 + (a - 8)*x - 8*a ≤ 0
def I : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x, M x ∧ P x a ↔ x ∈ I) → a = 0 :=
sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, (M x ∧ P x a → x ∈ I) ∧ (∀ x, x ∈ I → M x ∧ P x a)) → a ≤ 3 :=
sorry

end sufficient_but_not_necessary_necessary_but_not_sufficient_l43_4354


namespace find_unknown_number_l43_4353

def unknown_number (x : ℝ) : Prop :=
  (0.5^3) - (0.1^3 / 0.5^2) + x + (0.1^2) = 0.4

theorem find_unknown_number : ∃ (x : ℝ), unknown_number x ∧ x = 0.269 :=
by
  sorry

end find_unknown_number_l43_4353


namespace find_coordinates_of_C_l43_4306

structure Point where
  x : Int
  y : Int

def isSymmetricalAboutXAxis (A B : Point) : Prop :=
  A.x = B.x ∧ A.y = -B.y

def isSymmetricalAboutOrigin (B C : Point) : Prop :=
  C.x = -B.x ∧ C.y = -B.y

theorem find_coordinates_of_C :
  ∃ C : Point, let A := Point.mk 2 (-3)
               let B := Point.mk 2 3
               isSymmetricalAboutXAxis A B →
               isSymmetricalAboutOrigin B C →
               C = Point.mk (-2) (-3) :=
by
  sorry

end find_coordinates_of_C_l43_4306


namespace solve_for_y_l43_4309

theorem solve_for_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by sorry

end solve_for_y_l43_4309


namespace geometric_series_first_term_l43_4350

theorem geometric_series_first_term (a : ℝ) (r : ℝ) (s : ℝ) 
  (h1 : r = -1/3) (h2 : s = 12) (h3 : s = a / (1 - r)) : a = 16 :=
by
  -- Placeholder for the proof
  sorry

end geometric_series_first_term_l43_4350


namespace smallest_prime_with_digit_sum_18_l43_4345

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_18 : ∃ p : ℕ, Prime p ∧ 18 = sum_of_digits p ∧ (∀ q : ℕ, (Prime q ∧ 18 = sum_of_digits q) → p ≤ q) :=
by
  sorry

end smallest_prime_with_digit_sum_18_l43_4345


namespace sale_in_third_month_l43_4351

def sales_in_months (m1 m2 m3 m4 m5 m6 : Int) : Prop :=
  m1 = 5124 ∧
  m2 = 5366 ∧
  m4 = 6124 ∧
  m6 = 4579 ∧
  (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 5400

theorem sale_in_third_month (m5 : Int) :
  (∃ m3 : Int, sales_in_months 5124 5366 m3 6124 m5 4579 → m3 = 11207) :=
sorry

end sale_in_third_month_l43_4351


namespace douglas_votes_in_county_y_l43_4322

variable (V : ℝ) -- Number of voters in County Y
variable (A B : ℝ) -- Votes won by Douglas in County X and County Y respectively

-- Conditions
axiom h1 : A = 0.74 * 2 * V
axiom h2 : A + B = 0.66 * 3 * V
axiom ratio : (2 * V) / V = 2

-- Proof Statement
theorem douglas_votes_in_county_y :
  (B / V) * 100 = 50 := by
sorry

end douglas_votes_in_county_y_l43_4322


namespace eight_percent_is_64_l43_4369

-- Definition of the condition
variable (x : ℝ)

-- The theorem that states the problem to be proven
theorem eight_percent_is_64 (h : (8 / 100) * x = 64) : x = 800 :=
sorry

end eight_percent_is_64_l43_4369


namespace min_xy_l43_4324

variable {x y : ℝ}

theorem min_xy (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : x * y ≥ 180 := 
sorry

end min_xy_l43_4324


namespace roots_are_positive_integers_implies_r_values_l43_4366

theorem roots_are_positive_integers_implies_r_values (r x : ℕ) (h : (r * x^2 - (2 * r + 7) * x + (r + 7) = 0) ∧ (x > 0)) :
  r = 7 ∨ r = 0 ∨ r = 1 :=
by
  sorry

end roots_are_positive_integers_implies_r_values_l43_4366


namespace crayons_ratio_l43_4334

theorem crayons_ratio (K B G J : ℕ) 
  (h1 : K = 2 * B)
  (h2 : B = 2 * G)
  (h3 : G = J)
  (h4 : K = 128)
  (h5 : J = 8) : 
  G / J = 4 :=
by
  sorry

end crayons_ratio_l43_4334


namespace range_of_x_satisfies_conditions_l43_4321

theorem range_of_x_satisfies_conditions (x : ℝ) (h : x^2 - 4 < 0 ∨ |x| = 2) : -2 ≤ x ∧ x ≤ 2 := 
by
  sorry

end range_of_x_satisfies_conditions_l43_4321


namespace sequence_formula_l43_4385

theorem sequence_formula (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h : ∀ n : ℕ, S n = 3 * a n + (-1)^n) :
  ∀ n : ℕ, a n = (1/10) * (3/2)^(n-1) - (2/5) * (-1)^n :=
by sorry

end sequence_formula_l43_4385


namespace impossible_network_of_triangles_l43_4314

-- Define the conditions of the problem, here we could define vertices and properties of the network
structure Vertex :=
(triangles_meeting : Nat)

def five_triangles_meeting (v : Vertex) : Prop :=
v.triangles_meeting = 5

-- The main theorem statement - it's impossible to cover the entire plane with such a network
theorem impossible_network_of_triangles :
  ¬ (∀ v : Vertex, five_triangles_meeting v) :=
sorry

end impossible_network_of_triangles_l43_4314


namespace remainder_of_b2_minus_3a_div_6_l43_4396

theorem remainder_of_b2_minus_3a_div_6 (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 5) : 
  (b^2 - 3 * a) % 6 = 1 := 
sorry

end remainder_of_b2_minus_3a_div_6_l43_4396


namespace total_people_at_evening_l43_4387

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l43_4387


namespace polynomial_not_factorable_l43_4329

theorem polynomial_not_factorable (b c d : Int) (h₁ : (b * d + c * d) % 2 = 1) : 
  ¬ ∃ p q r : Int, (x + p) * (x^2 + q * x + r) = x^3 + b * x^2 + c * x + d :=
by 
  sorry

end polynomial_not_factorable_l43_4329


namespace value_of_expression_when_x_is_2_l43_4336

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l43_4336


namespace problem1_problem2_l43_4340

def f (x a : ℝ) : ℝ := abs (1 - x - a) + abs (2 * a - x)

theorem problem1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
  sorry

theorem problem2 (a x : ℝ) (h : a ≥ 2/3) : f x a ≥ 1 :=
  sorry

end problem1_problem2_l43_4340


namespace intersection_eq_l43_4370

def setA : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def expectedIntersection : Set ℝ := { y | 0 < y }

theorem intersection_eq :
  setA ∩ setB = expectedIntersection :=
sorry

end intersection_eq_l43_4370


namespace inequality_must_hold_l43_4304

theorem inequality_must_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by {
  sorry
}

end inequality_must_hold_l43_4304


namespace min_distance_curveC1_curveC2_l43_4302

-- Definitions of the conditions
def curveC1 (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 3 + Real.cos θ ∧ P.2 = 4 + Real.sin θ

def curveC2 (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

-- Proof statement
theorem min_distance_curveC1_curveC2 :
  (∀ A B : ℝ × ℝ,
    curveC1 A →
    curveC2 B →
    ∃ m : ℝ, m = 3 ∧ ∀ d : ℝ, (d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) → d ≥ m) := 
  sorry

end min_distance_curveC1_curveC2_l43_4302


namespace sand_needed_for_sandbox_l43_4383

def length1 : ℕ := 50
def width1 : ℕ := 30
def length2 : ℕ := 20
def width2 : ℕ := 15
def area_per_bag : ℕ := 80
def weight_per_bag : ℕ := 30

theorem sand_needed_for_sandbox :
  (length1 * width1 + length2 * width2 + area_per_bag - 1) / area_per_bag * weight_per_bag = 690 :=
by sorry

end sand_needed_for_sandbox_l43_4383


namespace gamma_max_success_ratio_l43_4333

theorem gamma_max_success_ratio :
  ∀ (x y z w : ℕ),
    x > 0 → z > 0 →
    (5 * x < 3 * y) →
    (5 * z < 3 * w) →
    (y + w = 600) →
    (x + z ≤ 359) :=
by
  intros x y z w hx hz hxy hzw hyw
  sorry

end gamma_max_success_ratio_l43_4333


namespace leos_current_weight_l43_4315

theorem leos_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 180) : L = 104 := 
by 
  sorry

end leos_current_weight_l43_4315


namespace false_inverse_proposition_l43_4388

theorem false_inverse_proposition (a b : ℝ) : (a^2 = b^2) → (a = b ∨ a = -b) := sorry

end false_inverse_proposition_l43_4388


namespace sandra_beignets_l43_4356

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l43_4356


namespace speed_difference_l43_4358

noncomputable def park_distance : ℝ := 10
noncomputable def kevin_time_hours : ℝ := 1 / 4
noncomputable def joel_time_hours : ℝ := 2

theorem speed_difference : (10 / kevin_time_hours) - (10 / joel_time_hours) = 35 := by
  sorry

end speed_difference_l43_4358


namespace completing_the_square_l43_4305

theorem completing_the_square (x : ℝ) : x^2 + 2 * x - 5 = 0 → (x + 1)^2 = 6 := by
  intro h
  -- Starting from h and following the steps outlined to complete the square.
  sorry

end completing_the_square_l43_4305


namespace second_and_third_shooters_cannot_win_or_lose_simultaneously_l43_4346

-- Define the conditions C1, C2, and C3
variables (C1 C2 C3 : Prop)

-- The first shooter bets that at least one of the second or third shooters will miss
def first_shooter_bet : Prop := ¬ (C2 ∧ C3)

-- The second shooter bets that if the first shooter hits, then at least one of the remaining shooters will miss
def second_shooter_bet : Prop := C1 → ¬ (C2 ∧ C3)

-- The third shooter bets that all three will hit the target on the first attempt
def third_shooter_bet : Prop := C1 ∧ C2 ∧ C3

-- Prove that it is impossible for both the second and third shooters to either win or lose their bets concurrently
theorem second_and_third_shooters_cannot_win_or_lose_simultaneously :
  ¬ ((second_shooter_bet C1 C2 C3 ∧ third_shooter_bet C1 C2 C3) ∨ (¬ second_shooter_bet C1 C2 C3 ∧ ¬ third_shooter_bet C1 C2 C3)) :=
by
  sorry

end second_and_third_shooters_cannot_win_or_lose_simultaneously_l43_4346


namespace union_set_A_set_B_l43_4348

def set_A : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }
def set_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def set_union (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∨ x ∈ B }

theorem union_set_A_set_B : set_union set_A set_B = { x | -3 < x ∧ x < 6 } := 
by sorry

end union_set_A_set_B_l43_4348


namespace decode_plaintext_l43_4312

theorem decode_plaintext (a x y : ℕ) (h1 : y = a^x - 2) (h2 : 6 = a^3 - 2) (h3 : y = 14) : x = 4 := by
  sorry

end decode_plaintext_l43_4312


namespace fabric_per_pair_of_pants_l43_4376

theorem fabric_per_pair_of_pants 
  (jenson_shirts_per_day : ℕ)
  (kingsley_pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric_needed : ℕ)
  (days : ℕ)
  (fabric_per_pant : ℕ) :
  jenson_shirts_per_day = 3 →
  kingsley_pants_per_day = 5 →
  fabric_per_shirt = 2 →
  total_fabric_needed = 93 →
  days = 3 →
  fabric_per_pant = 5 :=
by sorry

end fabric_per_pair_of_pants_l43_4376


namespace sqrt_of_1_5625_eq_1_25_l43_4331

theorem sqrt_of_1_5625_eq_1_25 : Real.sqrt 1.5625 = 1.25 :=
  sorry

end sqrt_of_1_5625_eq_1_25_l43_4331


namespace primes_with_large_gap_exists_l43_4357

noncomputable def exists_primes_with_large_gap_and_composites_between : Prop :=
  ∃ p q : ℕ, p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p > 2015 ∧ (∀ n : ℕ, p < n ∧ n < q → ¬Nat.Prime n)

theorem primes_with_large_gap_exists : exists_primes_with_large_gap_and_composites_between := sorry

end primes_with_large_gap_exists_l43_4357


namespace gcd_840_1764_gcd_459_357_l43_4367

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := sorry

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := sorry

end gcd_840_1764_gcd_459_357_l43_4367
