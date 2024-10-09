import Mathlib

namespace class_strength_l1113_111395

/-- The average age of an adult class is 40 years.
    12 new students with an average age of 32 years join the class,
    therefore decreasing the average by 4 years.
    What was the original strength of the class? -/
theorem class_strength (x : ℕ) (h1 : ∃ (x : ℕ), ∀ (y : ℕ), y ≠ x → y = 40) 
                       (h2 : 12 ≥ 0) (h3 : 32 ≥ 0) (h4 : (x + 12) * 36 = 40 * x + 12 * 32) : 
  x = 12 := 
sorry

end class_strength_l1113_111395


namespace sequence_sum_eq_ten_implies_n_eq_120_l1113_111331

theorem sequence_sum_eq_ten_implies_n_eq_120 :
  (∀ (a : ℕ → ℝ), (∀ n, a n = 1 / (Real.sqrt n + Real.sqrt (n + 1))) →
    (∃ n, (Finset.sum (Finset.range n) a) = 10 → n = 120)) :=
by
  intro a h
  use 120
  intro h_sum
  sorry

end sequence_sum_eq_ten_implies_n_eq_120_l1113_111331


namespace trigonometric_identity_l1113_111357

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31 / 5 := by
  sorry

end trigonometric_identity_l1113_111357


namespace base_eight_to_base_ten_l1113_111364

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l1113_111364


namespace negation_of_forall_ge_2_l1113_111353

theorem negation_of_forall_ge_2 :
  (¬ ∀ x : ℝ, x ≥ 2) = (∃ x₀ : ℝ, x₀ < 2) :=
sorry

end negation_of_forall_ge_2_l1113_111353


namespace solve_quartic_eq_l1113_111301

theorem solve_quartic_eq {x : ℝ} : (x - 4)^4 + (x - 6)^4 = 16 → (x = 4 ∨ x = 6) :=
by
  sorry

end solve_quartic_eq_l1113_111301


namespace batsman_average_after_17th_inning_l1113_111363

theorem batsman_average_after_17th_inning 
  (score_17 : ℕ)
  (delta_avg : ℤ)
  (n_before : ℕ)
  (initial_avg : ℤ)
  (h1 : score_17 = 74)
  (h2 : delta_avg = 3)
  (h3 : n_before = 16)
  (h4 : initial_avg = 23) :
  (initial_avg + delta_avg) = 26 := 
by
  sorry

end batsman_average_after_17th_inning_l1113_111363


namespace total_amount_paid_l1113_111317

-- Definitions
def original_aquarium_price : ℝ := 120
def aquarium_discount : ℝ := 0.5
def aquarium_coupon : ℝ := 0.1
def aquarium_sales_tax : ℝ := 0.05

def plants_decorations_price_before_discount : ℝ := 75
def plants_decorations_discount : ℝ := 0.15
def plants_decorations_sales_tax : ℝ := 0.08

def fish_food_price : ℝ := 25
def fish_food_sales_tax : ℝ := 0.06

-- Final result to be proved
theorem total_amount_paid : 
  let discounted_aquarium_price := original_aquarium_price * (1 - aquarium_discount)
  let coupon_aquarium_price := discounted_aquarium_price * (1 - aquarium_coupon)
  let total_aquarium_price := coupon_aquarium_price * (1 + aquarium_sales_tax)
  let discounted_plants_decorations_price := plants_decorations_price_before_discount * (1 - plants_decorations_discount)
  let total_plants_decorations_price := discounted_plants_decorations_price * (1 + plants_decorations_sales_tax)
  let total_fish_food_price := fish_food_price * (1 + fish_food_sales_tax)
  total_aquarium_price + total_plants_decorations_price + total_fish_food_price = 152.05 :=
by 
  sorry

end total_amount_paid_l1113_111317


namespace accessory_factory_growth_l1113_111349

theorem accessory_factory_growth (x : ℝ) :
  600 + 600 * (1 + x) + 600 * (1 + x) ^ 2 = 2180 :=
sorry

end accessory_factory_growth_l1113_111349


namespace parallel_lines_condition_l1113_111320

theorem parallel_lines_condition (a : ℝ) : 
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x y : ℝ, l1 x + a * y + 6 = 0) ∧ 
    (∀ x y : ℝ, (a - 2) * x + 3 * y + 2 * a = 0) ∧
    l1 = l2 ↔ a = 3) :=
sorry

end parallel_lines_condition_l1113_111320


namespace max_min_condition_monotonic_condition_l1113_111316

-- (1) Proving necessary and sufficient condition for f(x) to have both a maximum and minimum value
theorem max_min_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ -2*x₁ + a - (1/x₁) = 0 ∧ -2*x₂ + a - (1/x₂) = 0) ↔ a > Real.sqrt 8 :=
sorry

-- (2) Proving the range of values for a such that f(x) is monotonic on [1, 2]
theorem monotonic_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≥ 0) ∨
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≤ 0) ↔ a ≤ 3 ∨ a ≥ 4.5 :=
sorry

end max_min_condition_monotonic_condition_l1113_111316


namespace find_missing_digit_divisibility_by_4_l1113_111370

theorem find_missing_digit_divisibility_by_4 (x : ℕ) (h : x < 10) :
  (3280 + x) % 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 :=
by
  sorry

end find_missing_digit_divisibility_by_4_l1113_111370


namespace avg_growth_rate_l1113_111365

theorem avg_growth_rate {a p q x : ℝ} (h_eq : (1 + p) * (1 + q) = (1 + x) ^ 2) : 
  x ≤ (p + q) / 2 := 
by
  sorry

end avg_growth_rate_l1113_111365


namespace complex_square_l1113_111346

theorem complex_square (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_square_l1113_111346


namespace smallest_n_l1113_111382

theorem smallest_n (n : ℕ) (hn : n > 0) (h : 623 * n % 32 = 1319 * n % 32) : n = 4 :=
sorry

end smallest_n_l1113_111382


namespace compute_vector_expression_l1113_111373

theorem compute_vector_expression :
  4 • (⟨3, -5⟩ : ℝ × ℝ) - 3 • (⟨2, -6⟩ : ℝ × ℝ) + 2 • (⟨0, 3⟩ : ℝ × ℝ) = (⟨6, 4⟩ : ℝ × ℝ) := 
sorry

end compute_vector_expression_l1113_111373


namespace pen_sales_average_l1113_111358

theorem pen_sales_average (d : ℕ) (h1 : 96 + 44 * d > 0) (h2 : (96 + 44 * d) / (d + 1) = 48) : d = 12 :=
by
  sorry

end pen_sales_average_l1113_111358


namespace kay_weight_training_time_l1113_111379

variables (total_minutes : ℕ) (aerobic_ratio weight_ratio : ℕ)
-- Conditions
def kay_exercise := total_minutes = 250
def ratio_cond := aerobic_ratio = 3 ∧ weight_ratio = 2
def total_ratio_parts := aerobic_ratio + weight_ratio

-- Question and proof goal
theorem kay_weight_training_time (h1 : kay_exercise total_minutes) (h2 : ratio_cond aerobic_ratio weight_ratio) :
  (total_minutes / total_ratio_parts * weight_ratio) = 100 :=
by
  sorry

end kay_weight_training_time_l1113_111379


namespace min_max_of_quadratic_l1113_111384

theorem min_max_of_quadratic 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 - 6 * x + 1)
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 1) : 
  (∃ xmin, ∃ xmax, f xmin = -3 ∧ f xmax = 9 ∧ -1 ≤ xmin ∧ xmin ≤ 1 ∧ -1 ≤ xmax ∧ xmax ≤ 1) :=
sorry

end min_max_of_quadratic_l1113_111384


namespace find_possible_values_l1113_111314
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a bc de fg : ℕ) : Prop :=
  (a % 2 = 0) ∧ (is_prime bc) ∧ (de % 5 = 0) ∧ (fg % 3 = 0) ∧
  (fg - de = de - bc) ∧ (de - bc = bc - a)

theorem find_possible_values :
  ∃ (debc1 debc2 : ℕ),
    (satisfies_conditions 6 (debc1 % 100) ((debc1 / 100) % 100) ((debc1 / 10000) % 100)) ∧
    (satisfies_conditions 6 (debc2 % 100) ((debc2 / 100) % 100) ((debc2 / 10000) % 100)) ∧
    (debc1 = 2013 ∨ debc1 = 4023) ∧
    (debc2 = 2013 ∨ debc2 = 4023) :=
  sorry

end find_possible_values_l1113_111314


namespace min_value_l1113_111322

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end min_value_l1113_111322


namespace Maggie_bought_one_fish_book_l1113_111335

-- Defining the variables and constants
def books_about_plants := 9
def science_magazines := 10
def price_book := 15
def price_magazine := 2
def total_amount_spent := 170
def cost_books_about_plants := books_about_plants * price_book
def cost_science_magazines := science_magazines * price_magazine
def cost_books_about_fish := total_amount_spent - (cost_books_about_plants + cost_science_magazines)
def books_about_fish := cost_books_about_fish / price_book

-- Theorem statement
theorem Maggie_bought_one_fish_book : books_about_fish = 1 := by
  -- Proof goes here
  sorry

end Maggie_bought_one_fish_book_l1113_111335


namespace tan_domain_l1113_111366

open Real

theorem tan_domain (k : ℤ) (x : ℝ) :
  (∀ k : ℤ, x ≠ (k * π / 2) + (3 * π / 8)) ↔ 
  (∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2) := sorry

end tan_domain_l1113_111366


namespace derivative_at_one_l1113_111333

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end derivative_at_one_l1113_111333


namespace simplify_neg_neg_l1113_111338

theorem simplify_neg_neg (a b : ℝ) : -(-a - b) = a + b :=
sorry

end simplify_neg_neg_l1113_111338


namespace No_of_boxes_in_case_l1113_111375

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end No_of_boxes_in_case_l1113_111375


namespace rent_for_each_room_l1113_111377

theorem rent_for_each_room (x : ℝ) (ha : 4800 / x = 4200 / (x - 30)) (hx : x = 240) :
  x = 240 ∧ (x - 30) = 210 :=
by
  sorry

end rent_for_each_room_l1113_111377


namespace aku_mother_packages_l1113_111376

theorem aku_mother_packages
  (friends : Nat)
  (cookies_per_package : Nat)
  (cookies_per_child : Nat)
  (total_children : Nat)
  (birthday : Nat)
  (H_friends : friends = 4)
  (H_cookies_per_package : cookies_per_package = 25)
  (H_cookies_per_child : cookies_per_child = 15)
  (H_total_children : total_children = friends + 1)
  (H_birthday : birthday = 10) :
  (total_children * cookies_per_child) / cookies_per_package = 3 :=
by
  sorry

end aku_mother_packages_l1113_111376


namespace chord_length_intercepted_by_line_on_curve_l1113_111345

-- Define the curve and line from the problem
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Prove the length of the chord intercepted by the line on the curve is 4
theorem chord_length_intercepted_by_line_on_curve : 
  ∀ (x y : ℝ), curve x y → line x y → False := sorry

end chord_length_intercepted_by_line_on_curve_l1113_111345


namespace arithmetic_sequence_fifth_term_l1113_111313

theorem arithmetic_sequence_fifth_term :
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  a5 = 19 :=
by
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  show a5 = 19
  sorry

end arithmetic_sequence_fifth_term_l1113_111313


namespace total_goals_other_members_l1113_111312

theorem total_goals_other_members (x y : ℕ) (h1 : y = (7 * x) / 15 - 18)
  (h2 : 1 / 3 * x + 1 / 5 * x + 18 + y = x)
  (h3 : ∀ n, 0 ≤ n ∧ n ≤ 3 → ¬(n * 8 > y))
  : y = 24 :=
by
  sorry

end total_goals_other_members_l1113_111312


namespace consecutive_sum_ways_l1113_111327

theorem consecutive_sum_ways (S : ℕ) (hS : S = 385) :
  ∃! n : ℕ, ∃! k : ℕ, n ≥ 2 ∧ S = n * (2 * k + n - 1) / 2 :=
sorry

end consecutive_sum_ways_l1113_111327


namespace walls_per_person_l1113_111329

theorem walls_per_person (people : ℕ) (rooms : ℕ) (r4_walls r5_walls : ℕ) (total_walls : ℕ) (walls_each_person : ℕ)
  (h1 : people = 5)
  (h2 : rooms = 9)
  (h3 : r4_walls = 5 * 4)
  (h4 : r5_walls = 4 * 5)
  (h5 : total_walls = r4_walls + r5_walls)
  (h6 : walls_each_person = total_walls / people) :
  walls_each_person = 8 := by
  sorry

end walls_per_person_l1113_111329


namespace measureable_weights_count_l1113_111305

theorem measureable_weights_count (a b c : ℕ) (ha : a = 1) (hb : b = 3) (hc : c = 9) :
  ∃ s : Finset ℕ, s.card = 13 ∧ ∀ x ∈ s, x ≥ 1 ∧ x ≤ 13 := 
sorry

end measureable_weights_count_l1113_111305


namespace fraction_div_addition_l1113_111348

noncomputable def fraction_5_6 : ℚ := 5 / 6
noncomputable def fraction_9_10 : ℚ := 9 / 10
noncomputable def fraction_1_15 : ℚ := 1 / 15
noncomputable def fraction_402_405 : ℚ := 402 / 405

theorem fraction_div_addition :
  (fraction_5_6 / fraction_9_10) + fraction_1_15 = fraction_402_405 :=
by
  sorry

end fraction_div_addition_l1113_111348


namespace distribution_of_balls_l1113_111321

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l1113_111321


namespace ellipse_equation_l1113_111360

theorem ellipse_equation (c a b : ℝ)
  (foci1 foci2 : ℝ × ℝ) 
  (h_foci1 : foci1 = (-1, 0)) 
  (h_foci2 : foci2 = (1, 0)) 
  (h_c : c = 1) 
  (h_major_axis : 2 * a = 10) 
  (h_b_sq : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1)) :=
by
  sorry

end ellipse_equation_l1113_111360


namespace max_omega_l1113_111380

open Real

-- Define the function f(x) = sin(ωx + φ)
noncomputable def f (ω φ x : ℝ) := sin (ω * x + φ)

-- ω > 0 and |φ| ≤ π / 2
def condition_omega_pos (ω : ℝ) := ω > 0
def condition_phi_bound (φ : ℝ) := abs φ ≤ π / 2

-- x = -π/4 is a zero of f(x)
def condition_zero (ω φ : ℝ) := f ω φ (-π/4) = 0

-- x = π/4 is the axis of symmetry for the graph of y = f(x)
def condition_symmetry (ω φ : ℝ) := 
  ∀ x : ℝ, f ω φ (π/4 - x) = f ω φ (π/4 + x)

-- f(x) is monotonic in the interval (π/18, 5π/36)
def condition_monotonic (ω φ : ℝ) := 
  ∀ x₁ x₂ : ℝ, π/18 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * π / 36 
  → f ω φ x₁ ≤ f ω φ x₂

-- Prove that the maximum value of ω satisfying all the conditions is 9
theorem max_omega (ω : ℝ) (φ : ℝ)
  (h1 : condition_omega_pos ω)
  (h2 : condition_phi_bound φ)
  (h3 : condition_zero ω φ)
  (h4 : condition_symmetry ω φ)
  (h5 : condition_monotonic ω φ) :
  ω ≤ 9 :=
sorry

end max_omega_l1113_111380


namespace num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l1113_111332

theorem num_of_tenths_in_1_9 : (1.9 / 0.1) = 19 :=
by sorry

theorem num_of_hundredths_in_0_8 : (0.8 / 0.01) = 80 :=
by sorry

end num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l1113_111332


namespace napkin_ratio_l1113_111391

theorem napkin_ratio (initial_napkins : ℕ) (napkins_after : ℕ) (olivia_napkins : ℕ) (amelia_napkins : ℕ)
  (h1 : initial_napkins = 15) (h2 : napkins_after = 45) (h3 : olivia_napkins = 10)
  (h4 : initial_napkins + olivia_napkins + amelia_napkins = napkins_after) :
  amelia_napkins / olivia_napkins = 2 := by
  sorry

end napkin_ratio_l1113_111391


namespace at_least_two_inequalities_hold_l1113_111337

variable {a b c : ℝ}

theorem at_least_two_inequalities_hold (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) :=
  sorry

end at_least_two_inequalities_hold_l1113_111337


namespace total_math_and_biology_homework_l1113_111315

-- Definitions
def math_homework_pages : ℕ := 8
def biology_homework_pages : ℕ := 3

-- Theorem stating the problem to prove
theorem total_math_and_biology_homework :
  math_homework_pages + biology_homework_pages = 11 :=
by
  sorry

end total_math_and_biology_homework_l1113_111315


namespace solve_fraction_eq_l1113_111309

theorem solve_fraction_eq (x : ℝ) 
  (h₁ : x ≠ -9) 
  (h₂ : x ≠ -7) 
  (h₃ : x ≠ -10) 
  (h₄ : x ≠ -6) 
  (h₅ : 1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) : 
  x = -8 := 
sorry

end solve_fraction_eq_l1113_111309


namespace polynomial_sum_of_squares_l1113_111372

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 < P.eval x) :
  ∃ (U V : Polynomial ℝ), P = U^2 + V^2 := 
by
  sorry

end polynomial_sum_of_squares_l1113_111372


namespace max_sum_arithmetic_sequence_l1113_111392

theorem max_sum_arithmetic_sequence (n : ℕ) (M : ℝ) (hM : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ k, a (k + 1) - a k = a 1 - a 0) 
  (h_constraint : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S, S = (n + 1) * (Real.sqrt (10 * M)) / 2 :=
sorry

end max_sum_arithmetic_sequence_l1113_111392


namespace johns_speed_l1113_111304

theorem johns_speed (J : ℝ)
  (lewis_speed : ℝ := 60)
  (distance_AB : ℝ := 240)
  (meet_distance_A : ℝ := 160)
  (time_lewis_to_B : ℝ := distance_AB / lewis_speed)
  (time_lewis_back_80 : ℝ := 80 / lewis_speed)
  (total_time_meet : ℝ := time_lewis_to_B + time_lewis_back_80)
  (total_distance_john_meet : ℝ := J * total_time_meet) :
  total_distance_john_meet = meet_distance_A → J = 30 := 
by
  sorry

end johns_speed_l1113_111304


namespace skateboard_total_distance_is_3720_l1113_111371

noncomputable def skateboard_distance : ℕ :=
  let a1 := 10
  let d := 9
  let n := 20
  let flat_time := 10
  let a_n := a1 + (n - 1) * d
  let ramp_distance := n * (a1 + a_n) / 2
  let flat_distance := a_n * flat_time
  ramp_distance + flat_distance

theorem skateboard_total_distance_is_3720 : skateboard_distance = 3720 := 
by
  sorry

end skateboard_total_distance_is_3720_l1113_111371


namespace axis_of_symmetry_l1113_111323

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + (Real.pi / 2))) * (Real.cos (x + (Real.pi / 4)))

theorem axis_of_symmetry : 
  ∃ (a : ℝ), a = 5 * Real.pi / 8 ∧ ∀ x : ℝ, f (2 * a - x) = f x := 
by
  sorry

end axis_of_symmetry_l1113_111323


namespace largest_whole_number_l1113_111362

theorem largest_whole_number (x : ℤ) : 9 * x < 200 → x ≤ 22 := by
  sorry

end largest_whole_number_l1113_111362


namespace maximum_b_value_l1113_111326

noncomputable def f (a x : ℝ) := (1 / 2) * x ^ 2 + a * x
noncomputable def g (a b x : ℝ) := 2 * a ^ 2 * Real.log x + b

theorem maximum_b_value (a b : ℝ) (h_a : 0 < a) :
  (∃ x : ℝ, f a x = g a b x ∧ (deriv (f a) x = deriv (g a b) x))
  → b ≤ Real.exp (1 / 2) := 
sorry

end maximum_b_value_l1113_111326


namespace midpoint_of_complex_numbers_l1113_111336

theorem midpoint_of_complex_numbers :
  let A := (1 - 1*I) / (1 + 1)
  let B := (1 + 1*I) / (1 + 1)
  (A + B) / 2 = 1 / 2 := by
sorry

end midpoint_of_complex_numbers_l1113_111336


namespace multiplicative_inverse_sum_is_zero_l1113_111302

theorem multiplicative_inverse_sum_is_zero (a b : ℝ) (h : a * b = 1) :
  a^(2015) * b^(2016) + a^(2016) * b^(2017) + a^(2017) * b^(2016) + a^(2016) * b^(2015) = 0 :=
sorry

end multiplicative_inverse_sum_is_zero_l1113_111302


namespace union_complement_l1113_111311

open Set

variable (U A B : Set ℕ)
variable (u_spec : U = {1, 2, 3, 4, 5})
variable (a_spec : A = {1, 2, 3})
variable (b_spec : B = {2, 4})

theorem union_complement (U A B : Set ℕ)
  (u_spec : U = {1, 2, 3, 4, 5})
  (a_spec : A = {1, 2, 3})
  (b_spec : B = {2, 4}) :
  A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end union_complement_l1113_111311


namespace hawks_score_l1113_111318

theorem hawks_score (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 22) : H = 30 :=
by
  sorry

end hawks_score_l1113_111318


namespace yoongi_rank_l1113_111350

def namjoon_rank : ℕ := 2
def yoongi_offset : ℕ := 10

theorem yoongi_rank : namjoon_rank + yoongi_offset = 12 := 
by
  sorry

end yoongi_rank_l1113_111350


namespace max_value_quadratic_function_l1113_111352

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 8

theorem max_value_quadratic_function : ∃(x : ℝ), quadratic_function x = 8 :=
by
  sorry

end max_value_quadratic_function_l1113_111352


namespace discount_problem_l1113_111324

variable (x : ℝ)

theorem discount_problem :
  (400 * (1 - x)^2 = 225) :=
sorry

end discount_problem_l1113_111324


namespace inequality_l1113_111393

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c) ≤ 1 / (a * b * c) :=
sorry

end inequality_l1113_111393


namespace sin_double_angle_neg_l1113_111300

variable {α : ℝ} {k : ℤ}

-- Condition: α in the fourth quadrant.
def in_fourth_quadrant (α : ℝ) (k : ℤ) : Prop :=
  - (Real.pi / 2) + 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi

-- Goal: Prove sin 2α < 0 given that α is in the fourth quadrant.
theorem sin_double_angle_neg (α : ℝ) (k : ℤ) (h : in_fourth_quadrant α k) : Real.sin (2 * α) < 0 := by
  sorry

end sin_double_angle_neg_l1113_111300


namespace eq_has_exactly_one_real_root_l1113_111386

theorem eq_has_exactly_one_real_root : ∀ x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 ↔ x = 0 :=
by
sorry

end eq_has_exactly_one_real_root_l1113_111386


namespace cricket_average_increase_l1113_111388

-- Define the conditions as variables
variables (innings_initial : ℕ) (average_initial : ℕ) (runs_next_innings : ℕ)
variables (runs_increase : ℕ)

-- Given conditions
def conditions := (innings_initial = 13) ∧ (average_initial = 22) ∧ (runs_next_innings = 92)

-- Target: Calculate the desired increase in average (runs_increase)
theorem cricket_average_increase (h : conditions innings_initial average_initial runs_next_innings) :
  runs_increase = 5 :=
  sorry

end cricket_average_increase_l1113_111388


namespace problem_1_problem_2_l1113_111354

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem problem_1 (x : ℝ) : (f x 2) ≥ (7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) := 
by
  sorry

theorem problem_2 (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h : (f (1/m) 1) + (f (1/(2*n)) 1) = 1) : m + 4 * n ≥ 2 * Real.sqrt 2 + 3 := 
by
  sorry

end problem_1_problem_2_l1113_111354


namespace A_share_of_gain_l1113_111341

-- Definitions of conditions
variables 
  (x : ℕ) -- Initial investment by A
  (annual_gain : ℕ := 24000) -- Total annual gain
  (A_investment_period : ℕ := 12) -- Months A invested
  (B_investment_period : ℕ := 6) -- Months B invested after 6 months
  (C_investment_period : ℕ := 4) -- Months C invested after 8 months

-- Investment ratios
def A_ratio := x * A_investment_period
def B_ratio := (2 * x) * B_investment_period
def C_ratio := (3 * x) * C_investment_period

-- Proof statement
theorem A_share_of_gain : 
  A_ratio = 12 * x ∧ B_ratio = 12 * x ∧ C_ratio = 12 * x ∧ annual_gain = 24000 →
  annual_gain / 3 = 8000 :=
by
  sorry

end A_share_of_gain_l1113_111341


namespace number_of_special_three_digit_numbers_l1113_111374

theorem number_of_special_three_digit_numbers : ∃ (n : ℕ), n = 3 ∧
  (∀ (A B C : ℕ), 
    (100 * A + 10 * B + C < 1000 ∧ 100 * A + 10 * B + C ≥ 100) ∧
    B = 2 * C ∧
    B = (A + C) / 2 → 
    (A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 312 ∨ 
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 642 ∨
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 963))
:= 
sorry

end number_of_special_three_digit_numbers_l1113_111374


namespace first_position_remainder_one_l1113_111399

theorem first_position_remainder_one (a : ℕ) (h1 : 1 ≤ a ∧ a ≤ 2023)
(h2 : ∀ b c d : ℕ, b = a ∧ c = a + 2 ∧ d = a + 4 → 
  b % 3 ≠ c % 3 ∧ c % 3 ≠ d % 3 ∧ d % 3 ≠ b % 3):
  a % 3 = 1 :=
sorry

end first_position_remainder_one_l1113_111399


namespace part_a_part_b_part_c_l1113_111387

-- Defining a structure for the problem
structure Rectangle :=
(area : ℝ)

structure Figure :=
(area : ℝ)

-- Defining the conditions
variables (R : Rectangle) 
  (F1 F2 F3 F4 F5 : Figure)
  (overlap_area_pair : Figure → Figure → ℝ)
  (overlap_area_triple : Figure → Figure → Figure → ℝ)

-- Given conditions
axiom R_area : R.area = 1
axiom F1_area : F1.area = 0.5
axiom F2_area : F2.area = 0.5
axiom F3_area : F3.area = 0.5
axiom F4_area : F4.area = 0.5
axiom F5_area : F5.area = 0.5

-- Statements to prove
theorem part_a : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 3 / 20 := sorry
theorem part_b : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 1 / 5 := sorry
theorem part_c : ∃ (F1 F2 F3 : Figure), overlap_area_triple F1 F2 F3 ≥ 1 / 20 := sorry

end part_a_part_b_part_c_l1113_111387


namespace find_flat_fee_l1113_111385

def flat_fee_exists (f n : ℝ) : Prop :=
  f + n = 120 ∧ f + 4 * n = 255

theorem find_flat_fee : ∃ f n, flat_fee_exists f n ∧ f = 75 := by
  sorry

end find_flat_fee_l1113_111385


namespace factorization_of_expression_l1113_111397

noncomputable def factorized_form (x : ℝ) : ℝ :=
  (x + 5 / 2 + Real.sqrt 13 / 2) * (x + 5 / 2 - Real.sqrt 13 / 2)

theorem factorization_of_expression (x : ℝ) :
  x^2 - 5 * x + 3 = factorized_form x :=
by
  sorry

end factorization_of_expression_l1113_111397


namespace simplify_polynomial_l1113_111394

variable (x : ℝ)

theorem simplify_polynomial : (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 :=
by
  sorry

end simplify_polynomial_l1113_111394


namespace positional_relationship_perpendicular_l1113_111351

theorem positional_relationship_perpendicular 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h : b * Real.sin A - a * Real.sin B = 0) :
  (∀ x y : ℝ, (x * Real.sin A + a * y + c = 0) ↔ (b * x - y * Real.sin B + Real.sin C = 0)) :=
sorry

end positional_relationship_perpendicular_l1113_111351


namespace smallest_debt_exists_l1113_111340

theorem smallest_debt_exists :
  ∃ (p g : ℤ), 50 = 200 * p + 150 * g := by
  sorry

end smallest_debt_exists_l1113_111340


namespace quadratic_inequality_solution_range_l1113_111330

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end quadratic_inequality_solution_range_l1113_111330


namespace tan_pi_over_12_eq_l1113_111389

theorem tan_pi_over_12_eq : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_pi_over_12_eq_l1113_111389


namespace PU_squared_fraction_l1113_111368

noncomputable def compute_PU_squared : ℚ :=
  sorry -- Proof of the distance computation PU^2.

theorem PU_squared_fraction :
  ∃ (a b : ℕ), (gcd a b = 1) ∧ (compute_PU_squared = a / b) :=
  sorry -- Proof that the resulting fraction a/b is in its simplest form.

end PU_squared_fraction_l1113_111368


namespace parabola_equation_l1113_111396

theorem parabola_equation (A B : ℝ × ℝ) (x₁ x₂ y₁ y₂ p : ℝ) :
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  x₁ + x₂ = (p + 8) / 2 →
  x₁ * x₂ = 4 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 45 →
  (y₁ = 2 * x₁ - 4) →
  (y₂ = 2 * x₂ - 4) →
  ((y₁^2 = 2 * p * x₁) ∧ (y₂^2 = 2 * p * x₂)) →
  (y₁^2 = 4 * x₁ ∨ y₂^2 = -36 * x₂) := 
by {
  sorry
}

end parabola_equation_l1113_111396


namespace find_f_29_l1113_111367

theorem find_f_29 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 3) = (x - 3) * (x + 4)) : f 29 = 170 := 
by
  sorry

end find_f_29_l1113_111367


namespace range_of_k_l1113_111347

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k^2 - 1) * x^2 - (k + 1) * x + 1 > 0) ↔ (1 ≤ k ∧ k ≤ 5 / 3) := 
sorry

end range_of_k_l1113_111347


namespace sheila_hourly_wage_l1113_111308

def weekly_working_hours : Nat :=
  (8 * 3) + (6 * 2)

def weekly_earnings : Nat :=
  468

def hourly_wage : Nat :=
  weekly_earnings / weekly_working_hours

theorem sheila_hourly_wage : hourly_wage = 13 :=
by
  sorry

end sheila_hourly_wage_l1113_111308


namespace twelfth_term_of_geometric_sequence_l1113_111334

theorem twelfth_term_of_geometric_sequence (a : ℕ) (r : ℕ) (h1 : a * r ^ 4 = 8) (h2 : a * r ^ 8 = 128) : 
  a * r ^ 11 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l1113_111334


namespace sufficient_condition_increasing_l1113_111339

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem sufficient_condition_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 < x → x < y → (f x a ≤ f y a)) → a = -1 := sorry

end sufficient_condition_increasing_l1113_111339


namespace label_sum_l1113_111369

theorem label_sum (n : ℕ) : 
  (∃ S : ℕ → ℕ, S 1 = 2 ∧ (∀ k, k > 1 → (S (k + 1) = 2 * S k)) ∧ S n = 2 * 3 ^ (n - 1)) := 
sorry

end label_sum_l1113_111369


namespace parabola_properties_l1113_111378

-- Define the parabola function as y = x^2 + px + q
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p * x + q

-- Prove the properties of parabolas for varying p and q.
theorem parabola_properties (p q p' q' : ℝ) :
  (∀ x : ℝ, parabola p q x = x^2 + p * x + q) ∧
  (∀ x : ℝ, parabola p' q' x = x^2 + p' * x + q') →
  (∀ x : ℝ, ( ∃ k h : ℝ, parabola p q x = (x + h)^2 + k ) ∧ 
               ( ∃ k' h' : ℝ, parabola p' q' x = (x + h')^2 + k' ) ) ∧
  (∀ x : ℝ, h = -p / 2 ∧ k = q - p^2 / 4 ) ∧
  (∀ x : ℝ, h' = -p' / 2 ∧ k' = q' - p'^2 / 4 ) ∧
  (∀ x : ℝ, (h, k) ≠ (h', k') → parabola p q x ≠ parabola p' q' x) ∧
  (∀ x : ℝ, h = h' ∧ k = k' → parabola p q x = parabola p' q' x) :=
by
  sorry

end parabola_properties_l1113_111378


namespace relationship_between_A_and_B_l1113_111310

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2 * x = 0}

theorem relationship_between_A_and_B : B ⊆ A :=
sorry

end relationship_between_A_and_B_l1113_111310


namespace nth_group_sum_correct_l1113_111381

-- Define the function that computes the sum of the numbers in the nth group
def nth_group_sum (n : ℕ) : ℕ :=
  n * (n^2 + 1) / 2

-- The theorem statement
theorem nth_group_sum_correct (n : ℕ) : 
  nth_group_sum n = n * (n^2 + 1) / 2 := by
  sorry

end nth_group_sum_correct_l1113_111381


namespace remainder_when_divided_by_14_l1113_111383

theorem remainder_when_divided_by_14 (A : ℕ) (h1 : A % 1981 = 35) (h2 : A % 1982 = 35) : A % 14 = 7 :=
sorry

end remainder_when_divided_by_14_l1113_111383


namespace third_group_members_l1113_111344

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end third_group_members_l1113_111344


namespace melted_mixture_weight_l1113_111390

theorem melted_mixture_weight (Z C : ℝ) (h_ratio : Z / C = 9 / 11) (h_zinc : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l1113_111390


namespace infinite_solutions_x2_y2_z2_x3_y3_z3_l1113_111303

-- Define the parametric forms
def param_x (k : ℤ) := k * (2 * k^2 + 1)
def param_y (k : ℤ) := 2 * k^2 + 1
def param_z (k : ℤ) := -k * (2 * k^2 + 1)

-- Prove the equation
theorem infinite_solutions_x2_y2_z2_x3_y3_z3 :
  ∀ k : ℤ, param_x k ^ 2 + param_y k ^ 2 + param_z k ^ 2 = param_x k ^ 3 + param_y k ^ 3 + param_z k ^ 3 :=
by
  intros k
  -- Calculation needs to be proved here, we place a placeholder for now
  sorry

end infinite_solutions_x2_y2_z2_x3_y3_z3_l1113_111303


namespace scarves_per_box_l1113_111356

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ)
  (mittens_per_box : ℕ)
  (total_clothes : ℕ)
  (h1 : boxes = 4)
  (h2 : mittens_per_box = 6)
  (h3 : total_clothes = 32)
  (total_mittens := boxes * mittens_per_box)
  (total_scarves := total_clothes - total_mittens) :
  total_scarves / boxes = 2 :=
by
  sorry

end scarves_per_box_l1113_111356


namespace person_age_l1113_111342

-- Define the conditions
def current_age : ℕ := 18

-- Define the equation based on the person's statement
def age_equation (A Y : ℕ) : Prop := 3 * (A + 3) - 3 * (A - Y) = A

-- Statement to be proven
theorem person_age (Y : ℕ) : 
  age_equation current_age Y → Y = 3 := 
by 
  sorry

end person_age_l1113_111342


namespace blue_paper_side_length_l1113_111398

theorem blue_paper_side_length (side_red : ℝ) (side_blue : ℝ) (same_area : side_red^2 = side_blue * x) (side_red_val : side_red = 5) (side_blue_val : side_blue = 4) : x = 6.25 :=
by
  sorry

end blue_paper_side_length_l1113_111398


namespace find_a_plus_2b_l1113_111325

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end find_a_plus_2b_l1113_111325


namespace find_range_a_l1113_111307

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1) * x + (a - 2)

theorem find_range_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1 ) :
  -2 < a ∧ a < 1 := sorry

end find_range_a_l1113_111307


namespace hockey_cards_count_l1113_111361

-- Define integer variables for the number of hockey, football and baseball cards
variables (H F B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := F = 4 * H
def condition2 : Prop := B = F - 50
def condition3 : Prop := H > 0
def condition4 : Prop := H + F + B = 1750

-- The theorem to prove
theorem hockey_cards_count 
  (h1 : condition1 H F)
  (h2 : condition2 F B)
  (h3 : condition3 H)
  (h4 : condition4 H F B) : 
  H = 200 := by
sorry

end hockey_cards_count_l1113_111361


namespace distance_between_trees_l1113_111343

-- The conditions given
def trees_on_yard := 26
def yard_length := 500
def trees_at_ends := true

-- Theorem stating the proof
theorem distance_between_trees (h1 : trees_on_yard = 26) 
                               (h2 : yard_length = 500) 
                               (h3 : trees_at_ends = true) : 
  500 / (26 - 1) = 20 :=
by
  sorry

end distance_between_trees_l1113_111343


namespace fraction_equality_l1113_111355

-- Defining the main problem statement
theorem fraction_equality (x y z : ℚ) (k : ℚ) 
  (h1 : x = 3 * k) (h2 : y = 5 * k) (h3 : z = 7 * k) :
  (y + z) / (3 * x - y) = 3 :=
by
  sorry

end fraction_equality_l1113_111355


namespace find_k_eq_neg2_l1113_111319

theorem find_k_eq_neg2 (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by sorry

end find_k_eq_neg2_l1113_111319


namespace find_point_on_parabola_l1113_111306

open Real

theorem find_point_on_parabola :
  ∃ (x y : ℝ), 
  (0 ≤ x ∧ 0 ≤ y) ∧
  (x^2 = 8 * y) ∧
  sqrt (x^2 + (y - 2)^2) = 120 ∧
  (x = 2 * sqrt 236 ∧ y = 118) :=
by
  sorry

end find_point_on_parabola_l1113_111306


namespace factor_expression_l1113_111328

theorem factor_expression (x y z : ℝ) :
  ((x^3 - y^3)^3 + (y^3 - z^3)^3 + (z^3 - x^3)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  ((x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2)) :=
by {
  sorry  -- The proof goes here
}

end factor_expression_l1113_111328


namespace semicircle_triangle_l1113_111359

variable (a b r : ℝ)

-- Conditions: 
-- (1) Semicircle of radius r inside a right-angled triangle
-- (2) Shorter edges of the triangle (tangents to the semicircle) have lengths a and b
-- (3) Diameter of the semicircle lies on the hypotenuse of the triangle

theorem semicircle_triangle (h1 : a > 0) (h2 : b > 0) (h3 : r > 0)
  (tangent_property : true) -- Assumed relevant tangent properties are true
  (angle_property : true) -- Assumed relevant angle properties are true
  (geom_configuration : true) -- Assumed specific geometric configuration is correct
  : 1 / r = 1 / a + 1 / b := 
  sorry

end semicircle_triangle_l1113_111359
