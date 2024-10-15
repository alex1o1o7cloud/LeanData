import Mathlib

namespace NUMINAMATH_GPT_shaded_region_area_l1280_128042

open Real

noncomputable def area_of_shaded_region (side : ℝ) (radius : ℝ) : ℝ :=
  let area_square := side ^ 2
  let area_sector := π * radius ^ 2 / 4
  let area_triangle := (1 / 2) * (side / 2) * sqrt ((side / 2) ^ 2 - radius ^ 2)
  area_square - 8 * area_triangle - 4 * area_sector

theorem shaded_region_area (h_side : ℝ) (h_radius : ℝ)
  (h1 : h_side = 8) (h2 : h_radius = 3) :
  area_of_shaded_region h_side h_radius = 64 - 16 * sqrt 7 - 3 * π :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1280_128042


namespace NUMINAMATH_GPT_probability_not_all_same_l1280_128029

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end NUMINAMATH_GPT_probability_not_all_same_l1280_128029


namespace NUMINAMATH_GPT_price_of_basic_computer_l1280_128087

-- Conditions
variables (C P : ℝ)
axiom cond1 : C + P = 2500
axiom cond2 : 3 * P = C + 500

-- Prove that the price of the basic computer is $1750
theorem price_of_basic_computer : C = 1750 :=
by 
  sorry

end NUMINAMATH_GPT_price_of_basic_computer_l1280_128087


namespace NUMINAMATH_GPT_swimming_club_cars_l1280_128026

theorem swimming_club_cars (c : ℕ) :
  let vans := 3
  let people_per_car := 5
  let people_per_van := 3
  let max_people_per_car := 6
  let max_people_per_van := 8
  let extra_people := 17
  let total_people := 5 * c + (people_per_van * vans)
  let max_capacity := max_people_per_car * c + (max_people_per_van * vans)
  (total_people + extra_people = max_capacity) → c = 2 := by
  sorry

end NUMINAMATH_GPT_swimming_club_cars_l1280_128026


namespace NUMINAMATH_GPT_sum_weights_greater_than_2p_l1280_128015

variables (p x y l l' : ℝ)

-- Conditions
axiom balance1 : x * l = p * l'
axiom balance2 : y * l' = p * l

-- The statement to prove
theorem sum_weights_greater_than_2p : x + y > 2 * p :=
by
  sorry

end NUMINAMATH_GPT_sum_weights_greater_than_2p_l1280_128015


namespace NUMINAMATH_GPT_tan_alpha_result_l1280_128013

theorem tan_alpha_result (α : ℝ) (h : Real.tan (α - Real.pi / 4) = 1 / 6) : Real.tan α = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_result_l1280_128013


namespace NUMINAMATH_GPT_proof_A_union_B_eq_R_l1280_128035

def A : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - 5) < a }

theorem proof_A_union_B_eq_R (a : ℝ) (h : a > 6) : 
  A ∪ B a = Set.univ :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_A_union_B_eq_R_l1280_128035


namespace NUMINAMATH_GPT_evaluate_polynomial_l1280_128054

theorem evaluate_polynomial : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_l1280_128054


namespace NUMINAMATH_GPT_range_of_a_l1280_128052

open Real

-- The quadratic expression
def quadratic (a x : ℝ) : ℝ := a*x^2 + 2*x + a

-- The condition of the problem
def quadratic_nonnegative_for_all (a : ℝ) := ∀ x : ℝ, quadratic a x ≥ 0

-- The theorem to be proven
theorem range_of_a (a : ℝ) (h : quadratic_nonnegative_for_all a) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1280_128052


namespace NUMINAMATH_GPT_sum_xyz_eq_10_l1280_128088

theorem sum_xyz_eq_10 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * x * y + 3 * x * y * z = 115) : 
  x + y + z = 10 :=
sorry

end NUMINAMATH_GPT_sum_xyz_eq_10_l1280_128088


namespace NUMINAMATH_GPT_find_y_for_orthogonality_l1280_128022

theorem find_y_for_orthogonality (y : ℝ) : (3 * y + 7 * (-4) = 0) → y = 28 / 3 := by
  sorry

end NUMINAMATH_GPT_find_y_for_orthogonality_l1280_128022


namespace NUMINAMATH_GPT_defeat_giant_enemy_crab_l1280_128027

-- Definitions for the conditions of cutting legs and claws
def claws : ℕ := 2
def legs : ℕ := 6
def totalCuts : ℕ := claws + legs
def valid_sequences : ℕ :=
  (Nat.factorial legs) * (Nat.factorial claws) * Nat.choose (totalCuts - claws - 1) claws

-- Statement to prove the number of valid sequences of cuts given the conditions
theorem defeat_giant_enemy_crab : valid_sequences = 14400 := by
  sorry

end NUMINAMATH_GPT_defeat_giant_enemy_crab_l1280_128027


namespace NUMINAMATH_GPT_admission_fee_for_adults_l1280_128070

theorem admission_fee_for_adults (C : ℝ) (N M N_c N_a : ℕ) (A : ℝ) 
  (h1 : C = 1.50) 
  (h2 : N = 2200) 
  (h3 : M = 5050) 
  (h4 : N_c = 700) 
  (h5 : N_a = 1500) :
  A = 2.67 := 
by
  sorry

end NUMINAMATH_GPT_admission_fee_for_adults_l1280_128070


namespace NUMINAMATH_GPT_intersection_M_N_l1280_128074

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ -3}

-- Prove the intersection of M and N is [1, 2)
theorem intersection_M_N : (M ∩ N) = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1280_128074


namespace NUMINAMATH_GPT_heavier_boxes_weight_l1280_128046

theorem heavier_boxes_weight
  (x y : ℤ)
  (h1 : x ≥ 0)
  (h2 : x ≤ 30)
  (h3 : 10 * x + (30 - x) * y = 540)
  (h4 : 10 * x + (15 - x) * y = 240) :
  y = 20 :=
by
  sorry

end NUMINAMATH_GPT_heavier_boxes_weight_l1280_128046


namespace NUMINAMATH_GPT_part1_max_price_part2_min_sales_volume_l1280_128001

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def original_revenue : ℝ := original_price * original_sales_volume
noncomputable def max_new_price (t : ℝ) : Prop := t * (130000 - 2000 * t) ≥ original_revenue

theorem part1_max_price (t : ℝ) (ht : max_new_price t) : t ≤ 40 :=
sorry

noncomputable def investment (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600) + 50 + (x / 5)
noncomputable def min_sales_volume (x : ℝ) (a : ℝ) : Prop := a * x ≥ original_revenue + investment x

theorem part2_min_sales_volume (a : ℝ) : min_sales_volume 30 a → a ≥ 10.2 :=
sorry

end NUMINAMATH_GPT_part1_max_price_part2_min_sales_volume_l1280_128001


namespace NUMINAMATH_GPT_dodecagon_diagonals_l1280_128084

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem dodecagon_diagonals : num_diagonals 12 = 54 :=
by
  -- by sorry means we skip the actual proof
  sorry

end NUMINAMATH_GPT_dodecagon_diagonals_l1280_128084


namespace NUMINAMATH_GPT_prime_condition_composite_condition_l1280_128047

theorem prime_condition (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a)
  (h_prime : Prime (2 * n - 1)) :
  ∃ i j : Fin n, i ≠ j ∧ ((a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) := 
sorry

theorem composite_condition (n : ℕ) (h_composite : ¬ Prime (2 * n - 1)) :
  ∃ a : Fin n → ℕ, Function.Injective a ∧ (∀ i j : Fin n, i ≠ j → ((a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1)) := 
sorry

end NUMINAMATH_GPT_prime_condition_composite_condition_l1280_128047


namespace NUMINAMATH_GPT_ratio_of_girls_participated_to_total_l1280_128096

noncomputable def ratio_participating_girls {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ℚ :=
    girls_participated / (girls_participated + boys_participated)

theorem ratio_of_girls_participated_to_total {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ratio_participating_girls h1 equal_boys_girls girls_participated boys_participated = 9 / 17 :=
by
    sorry

end NUMINAMATH_GPT_ratio_of_girls_participated_to_total_l1280_128096


namespace NUMINAMATH_GPT_minimum_seats_occupied_l1280_128048

-- Define the conditions
def initial_seat_count : Nat := 150
def people_initially_leaving_up_to_two_empty_seats := true
def eventually_rule_changes_to_one_empty_seat := true

-- Define the function which checks the minimum number of occupied seats needed
def fewest_occupied_seats (total_seats : Nat) (initial_rule : Bool) (final_rule : Bool) : Nat :=
  if initial_rule && final_rule && total_seats = 150 then 57 else 0

-- The main theorem we need to prove
theorem minimum_seats_occupied {total_seats : Nat} : 
  total_seats = initial_seat_count → 
  people_initially_leaving_up_to_two_empty_seats → 
  eventually_rule_changes_to_one_empty_seat → 
  fewest_occupied_seats total_seats people_initially_leaving_up_to_two_empty_seats eventually_rule_changes_to_one_empty_seat = 57 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_minimum_seats_occupied_l1280_128048


namespace NUMINAMATH_GPT_train_pass_bridge_in_50_seconds_l1280_128058

def length_of_train : ℕ := 360
def length_of_bridge : ℕ := 140
def speed_of_train_kmh : ℕ := 36
def total_distance : ℕ := length_of_train + length_of_bridge
def speed_of_train_ms : ℚ := (speed_of_train_kmh * 1000 : ℚ) / 3600 -- we use ℚ to avoid integer division issues
def time_to_pass_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge = 50 := by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_in_50_seconds_l1280_128058


namespace NUMINAMATH_GPT_solve_for_x_l1280_128092

theorem solve_for_x (x : ℕ) : 100^3 = 10^x → x = 6 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1280_128092


namespace NUMINAMATH_GPT_sum_of_common_ratios_l1280_128077

theorem sum_of_common_ratios (k p r a2 a3 b2 b3 : ℝ)
  (h1 : a3 = k * p^2) (h2 : a2 = k * p) 
  (h3 : b3 = k * r^2) (h4 : b2 = k * r)
  (h5 : p ≠ r)
  (h6 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2)) :
  p + r = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_common_ratios_l1280_128077


namespace NUMINAMATH_GPT_kaleb_books_l1280_128080

-- Define the initial number of books
def initial_books : ℕ := 34

-- Define the number of books sold
def books_sold : ℕ := 17

-- Define the number of books bought
def books_bought : ℕ := 7

-- Prove that the final number of books is 24
theorem kaleb_books (h : initial_books - books_sold + books_bought = 24) : initial_books - books_sold + books_bought = 24 :=
by
  exact h

end NUMINAMATH_GPT_kaleb_books_l1280_128080


namespace NUMINAMATH_GPT_course_selection_l1280_128066

noncomputable def number_of_ways (nA nB : ℕ) : ℕ :=
  (Nat.choose nA 2) * (Nat.choose nB 1) + (Nat.choose nA 1) * (Nat.choose nB 2)

theorem course_selection :
  (number_of_ways 3 4) = 30 :=
by
  sorry

end NUMINAMATH_GPT_course_selection_l1280_128066


namespace NUMINAMATH_GPT_rhombus_area_l1280_128050

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) : 
  1 / 2 * d1 * d2 = 15 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_rhombus_area_l1280_128050


namespace NUMINAMATH_GPT_find_angle_D_l1280_128044

theorem find_angle_D 
  (A B C D : ℝ) 
  (h1 : A + B = 180) 
  (h2 : C = D + 10) 
  (h3 : A = 50)
  : D = 20 := by
  sorry

end NUMINAMATH_GPT_find_angle_D_l1280_128044


namespace NUMINAMATH_GPT_natural_number_square_l1280_128004

theorem natural_number_square (n : ℕ) : 
  (∃ x : ℕ, n^4 + 4 * n^3 + 5 * n^2 + 6 * n = x^2) ↔ n = 1 := 
by 
  sorry

end NUMINAMATH_GPT_natural_number_square_l1280_128004


namespace NUMINAMATH_GPT_fraction_of_a_eq_1_fifth_of_b_l1280_128083

theorem fraction_of_a_eq_1_fifth_of_b (a b : ℝ) (x : ℝ) 
  (h1 : a + b = 100) 
  (h2 : (1/5) * b = 12)
  (h3 : b = 60) : x = 3/10 := by
  sorry

end NUMINAMATH_GPT_fraction_of_a_eq_1_fifth_of_b_l1280_128083


namespace NUMINAMATH_GPT_rounding_and_scientific_notation_l1280_128025

-- Define the original number
def original_number : ℕ := 1694000

-- Define the function to round to the nearest hundred thousand
def round_to_nearest_hundred_thousand (n : ℕ) : ℕ :=
  ((n + 50000) / 100000) * 100000

-- Define the function to convert to scientific notation
def to_scientific_notation (n : ℕ) : String :=
  let base := n / 1000000
  let exponent := 6
  s!"{base}.0 × 10^{exponent}"

-- Assert the equivalence
theorem rounding_and_scientific_notation :
  to_scientific_notation (round_to_nearest_hundred_thousand original_number) = "1.7 × 10^{6}" :=
by
  sorry

end NUMINAMATH_GPT_rounding_and_scientific_notation_l1280_128025


namespace NUMINAMATH_GPT_factor_polynomials_l1280_128028

theorem factor_polynomials (x : ℝ) :
  (x^2 + 4 * x + 3) * (x^2 + 9 * x + 20) + (x^2 + 6 * x - 9) = 
  (x^2 + 6 * x + 6) * (x^2 + 6 * x + 3) :=
sorry

end NUMINAMATH_GPT_factor_polynomials_l1280_128028


namespace NUMINAMATH_GPT_most_followers_is_sarah_l1280_128067

def initial_followers_susy : ℕ := 100
def initial_followers_sarah : ℕ := 50

def susy_week1_new : ℕ := 40
def susy_week2_new := susy_week1_new / 2
def susy_week3_new := susy_week2_new / 2
def susy_total_new := susy_week1_new + susy_week2_new + susy_week3_new
def susy_final_followers := initial_followers_susy + susy_total_new

def sarah_week1_new : ℕ := 90
def sarah_week2_new := sarah_week1_new / 3
def sarah_week3_new := sarah_week2_new / 3
def sarah_total_new := sarah_week1_new + sarah_week2_new + sarah_week3_new
def sarah_final_followers := initial_followers_sarah + sarah_total_new

theorem most_followers_is_sarah : 
    sarah_final_followers ≥ susy_final_followers := by
  sorry

end NUMINAMATH_GPT_most_followers_is_sarah_l1280_128067


namespace NUMINAMATH_GPT_unique_function_l1280_128090

def satisfies_inequality (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

theorem unique_function (k : ℤ) (h : k > 0) :
  ∃! f : ℝ → ℝ, satisfies_inequality f k :=
by
  sorry

end NUMINAMATH_GPT_unique_function_l1280_128090


namespace NUMINAMATH_GPT_first_quarter_spending_l1280_128089

variables (spent_february_start spent_march_end spent_april_end : ℝ)

-- Given conditions
def begin_february_spent : Prop := spent_february_start = 0.5
def end_march_spent : Prop := spent_march_end = 1.5
def end_april_spent : Prop := spent_april_end = 2.0

-- Proof statement
theorem first_quarter_spending (h1 : begin_february_spent spent_february_start) 
                               (h2 : end_march_spent spent_march_end) 
                               (h3 : end_april_spent spent_april_end) : 
                                spent_march_end - spent_february_start = 1.5 :=
by sorry

end NUMINAMATH_GPT_first_quarter_spending_l1280_128089


namespace NUMINAMATH_GPT_pow_mod_1110_l1280_128034

theorem pow_mod_1110 (n : ℕ) (h₀ : 0 ≤ n ∧ n < 1111)
    (h₁ : 2^1110 % 11 = 1) (h₂ : 2^1110 % 101 = 14) : 
    n = 1024 := 
sorry

end NUMINAMATH_GPT_pow_mod_1110_l1280_128034


namespace NUMINAMATH_GPT_product_of_sequence_l1280_128014

theorem product_of_sequence : 
  (1 / 2) * (4 / 1) * (1 / 8) * (16 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) = 64 := 
by
  sorry

end NUMINAMATH_GPT_product_of_sequence_l1280_128014


namespace NUMINAMATH_GPT_square_pyramid_intersection_area_l1280_128056

theorem square_pyramid_intersection_area (a b c d e : ℝ) (h_midpoints : a = 2 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4) : 
  ∃ p : ℝ, (p = 80) :=
by
  sorry

end NUMINAMATH_GPT_square_pyramid_intersection_area_l1280_128056


namespace NUMINAMATH_GPT_sum_of_digits_of_m_l1280_128081

theorem sum_of_digits_of_m (k m : ℕ) : 
  1 ≤ k ∧ k ≤ 3 ∧ 10000 ≤ 11131 * k + 1203 ∧ 11131 * k + 1203 < 100000 ∧ 
  11131 * k + 1203 = m * m ∧ 3 * k < 10 → 
  (m.digits 10).sum = 15 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_m_l1280_128081


namespace NUMINAMATH_GPT_fraction_of_shaded_hexagons_l1280_128009

-- Definitions
def total_hexagons : ℕ := 9
def shaded_hexagons : ℕ := 5

-- Theorem statement
theorem fraction_of_shaded_hexagons : 
  (shaded_hexagons: ℚ) / (total_hexagons : ℚ) = 5 / 9 := by
sorry

end NUMINAMATH_GPT_fraction_of_shaded_hexagons_l1280_128009


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1280_128091

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (h1 : a 1 + a 1 * q ^ 2 = 10) (h2 : a 1 * q + a 1 * q ^ 3 = 5) (h3 : geometric_sequence a q) :
  a 8 = 1 / 16 := sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1280_128091


namespace NUMINAMATH_GPT_cost_price_per_meter_l1280_128068

-- Define the given conditions
def selling_price : ℕ := 8925
def meters : ℕ := 85
def profit_per_meter : ℕ := 35

-- Define the statement to be proved
theorem cost_price_per_meter :
  (selling_price - profit_per_meter * meters) / meters = 70 := 
by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l1280_128068


namespace NUMINAMATH_GPT_ellipse_standard_eq_l1280_128095

theorem ellipse_standard_eq
  (e : ℝ) (a b : ℝ) (h1 : e = 1 / 2) (h2 : 2 * a = 4) (h3 : b^2 = a^2 - (a * e)^2)
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) ↔
    ( ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_eq_l1280_128095


namespace NUMINAMATH_GPT_find_n_cosine_l1280_128055

theorem find_n_cosine (n : ℤ) (h1 : 100 ≤ n ∧ n ≤ 300) (h2 : Real.cos (n : ℝ) = Real.cos 140) : n = 220 :=
by
  sorry

end NUMINAMATH_GPT_find_n_cosine_l1280_128055


namespace NUMINAMATH_GPT_chandler_needs_to_sell_more_rolls_l1280_128078

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end NUMINAMATH_GPT_chandler_needs_to_sell_more_rolls_l1280_128078


namespace NUMINAMATH_GPT_product_of_consecutive_integers_l1280_128039

theorem product_of_consecutive_integers
  (a b : ℕ) (n : ℕ)
  (h1 : a = 12)
  (h2 : b = 22)
  (mean_five_numbers : (a + b + n + (n + 1) + (n + 2)) / 5 = 17) :
  (n * (n + 1) * (n + 2)) = 4896 := by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_l1280_128039


namespace NUMINAMATH_GPT_orange_slices_l1280_128053

theorem orange_slices (x : ℕ) (hx1 : 5 * x = x + 8) : x + 2 * x + 5 * x = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_orange_slices_l1280_128053


namespace NUMINAMATH_GPT_find_y_for_orthogonal_vectors_l1280_128069

theorem find_y_for_orthogonal_vectors : 
  (∀ y, ((3:ℝ) * y + (-4:ℝ) * 9 = 0) → y = 12) :=
by
  sorry

end NUMINAMATH_GPT_find_y_for_orthogonal_vectors_l1280_128069


namespace NUMINAMATH_GPT_new_pressure_of_helium_l1280_128049

noncomputable def helium_pressure (p V p' V' : ℝ) (k : ℝ) : Prop :=
  p * V = k ∧ p' * V' = k

theorem new_pressure_of_helium :
  ∀ (p V p' V' k : ℝ), 
  p = 8 ∧ V = 3.5 ∧ V' = 7 ∧ k = 28 →
  helium_pressure p V p' V' k →
  p' = 4 :=
by
  intros p V p' V' k h1 h2
  sorry

end NUMINAMATH_GPT_new_pressure_of_helium_l1280_128049


namespace NUMINAMATH_GPT_min_homework_assignments_l1280_128000

variable (p1 p2 p3 : Nat)

-- Define the points and assignments
def points_first_10 : Nat := 10
def assignments_first_10 : Nat := 10 * 1

def points_second_10 : Nat := 10
def assignments_second_10 : Nat := 10 * 2

def points_third_10 : Nat := 10
def assignments_third_10 : Nat := 10 * 3

def total_points : Nat := points_first_10 + points_second_10 + points_third_10
def total_assignments : Nat := assignments_first_10 + assignments_second_10 + assignments_third_10

theorem min_homework_assignments (hp1 : points_first_10 = 10) (ha1 : assignments_first_10 = 10) 
  (hp2 : points_second_10 = 10) (ha2 : assignments_second_10 = 20)
  (hp3 : points_third_10 = 10) (ha3 : assignments_third_10 = 30)
  (tp : total_points = 30) : 
  total_assignments = 60 := 
by sorry

end NUMINAMATH_GPT_min_homework_assignments_l1280_128000


namespace NUMINAMATH_GPT_workers_contribution_eq_l1280_128051

variable (W C : ℕ)

theorem workers_contribution_eq :
  W * C = 300000 → W * (C + 50) = 320000 → W = 400 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_workers_contribution_eq_l1280_128051


namespace NUMINAMATH_GPT_rent_percentage_l1280_128072

variable (E : ℝ)

def rent_last_year (E : ℝ) : ℝ := 0.20 * E 
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 0.25 * (earnings_this_year E)

-- Prove that the rent this year is 143.75% of the rent last year
theorem rent_percentage : (rent_this_year E) = 1.4375 * (rent_last_year E) :=
by
  sorry

end NUMINAMATH_GPT_rent_percentage_l1280_128072


namespace NUMINAMATH_GPT_problem_statement_l1280_128076

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x + 2) * (x - 1) > 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 0}
def C_U (B : Set ℝ) : Set ℝ := {x | x ∉ B}

theorem problem_statement : A ∪ C_U B = {x | x < -2 ∨ x ≥ 0} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1280_128076


namespace NUMINAMATH_GPT_B_cycling_speed_l1280_128017

/--
A walks at 10 kmph. 10 hours after A starts, B cycles after him at a certain speed.
B catches up with A at a distance of 200 km from the start. Prove that B's cycling speed is 20 kmph.
-/
theorem B_cycling_speed (speed_A : ℝ) (time_A_to_start_B : ℝ) 
  (distance_at_catch : ℝ) (B_speed : ℝ)
  (h1 : speed_A = 10) 
  (h2 : time_A_to_start_B = 10)
  (h3 : distance_at_catch = 200)
  (h4 : distance_at_catch = speed_A * time_A_to_start_B + speed_A * (distance_at_catch / speed_B)) :
    B_speed = 20 := by
  sorry

end NUMINAMATH_GPT_B_cycling_speed_l1280_128017


namespace NUMINAMATH_GPT_circle_radius_eq_two_l1280_128063

theorem circle_radius_eq_two (x y : ℝ) : (x^2 + y^2 + 1 = 2 * x + 4 * y) → (∃ c : ℝ × ℝ, ∃ r : ℝ, ((x - c.1)^2 + (y - c.2)^2 = r^2) ∧ r = 2) := by
  sorry

end NUMINAMATH_GPT_circle_radius_eq_two_l1280_128063


namespace NUMINAMATH_GPT_tan_frac_eq_l1280_128024

theorem tan_frac_eq (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
  sorry

end NUMINAMATH_GPT_tan_frac_eq_l1280_128024


namespace NUMINAMATH_GPT_m_value_if_Q_subset_P_l1280_128061

noncomputable def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}
def m_values (m : ℝ) : Prop := Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1

theorem m_value_if_Q_subset_P (m : ℝ) : m_values m :=
sorry

end NUMINAMATH_GPT_m_value_if_Q_subset_P_l1280_128061


namespace NUMINAMATH_GPT_square_root_calc_l1280_128018

theorem square_root_calc (x : ℤ) (hx : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end NUMINAMATH_GPT_square_root_calc_l1280_128018


namespace NUMINAMATH_GPT_smallest_n_value_l1280_128085

-- Define the given expression
def exp := (2^5) * (6^2) * (7^3) * (13^4)

-- Define the conditions
def condition_5_2 (n : ℕ) := ∃ k, n * exp = k * 5^2
def condition_3_3 (n : ℕ) := ∃ k, n * exp = k * 3^3
def condition_11_2 (n : ℕ) := ∃ k, n * exp = k * 11^2

-- Define the smallest possible value of n
def smallest_n (n : ℕ) : Prop :=
  condition_5_2 n ∧ condition_3_3 n ∧ condition_11_2 n ∧ ∀ m, (condition_5_2 m ∧ condition_3_3 m ∧ condition_11_2 m) → m ≥ n

-- The theorem statement
theorem smallest_n_value : smallest_n 9075 :=
  by
    sorry

end NUMINAMATH_GPT_smallest_n_value_l1280_128085


namespace NUMINAMATH_GPT_planA_text_message_cost_l1280_128008

def planA_cost (x : ℝ) : ℝ := 60 * x + 9
def planB_cost : ℝ := 60 * 0.40

theorem planA_text_message_cost (x : ℝ) (h : planA_cost x = planB_cost) : x = 0.25 :=
by
  -- h represents the condition that the costs are equal
  -- The proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_planA_text_message_cost_l1280_128008


namespace NUMINAMATH_GPT_determine_k_l1280_128040

theorem determine_k (S : ℕ → ℝ) (k : ℝ)
  (hSn : ∀ n, S n = k + 2 * (1 / 3)^n)
  (a1 : ℝ := S 1)
  (a2 : ℝ := S 2 - S 1)
  (a3 : ℝ := S 3 - S 2)
  (geom_property : a2^2 = a1 * a3) :
  k = -2 := 
by
  sorry

end NUMINAMATH_GPT_determine_k_l1280_128040


namespace NUMINAMATH_GPT_dave_more_than_derek_l1280_128012

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_dave_more_than_derek_l1280_128012


namespace NUMINAMATH_GPT_g_of_five_eq_one_l1280_128065

variable (g : ℝ → ℝ)

theorem g_of_five_eq_one (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
    (h2 : ∀ x : ℝ, g x ≠ 0) : g 5 = 1 :=
sorry

end NUMINAMATH_GPT_g_of_five_eq_one_l1280_128065


namespace NUMINAMATH_GPT_area_of_octagon_l1280_128057

-- Define the basic geometric elements and properties
variables {A B C D E F G H : Type}
variables (isRectangle : BDEF A B C D E F G H)
variables (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2)
variables (isRightIsosceles : ABC A B C D E F G H)

-- Assumptions and known facts
def BDEF_is_rectangle : Prop := isRectangle
def AB_eq_2 : AB = 2 := hAB
def BC_eq_2 : BC = 2 := hBC
def ABC_is_right_isosceles : Prop := isRightIsosceles

-- Statement of the problem to be proved
theorem area_of_octagon : (exists (area : ℝ), area = 8 * Real.sqrt 2) :=
by {
  -- The proof details will go here, which we skip for now
  sorry
}

end NUMINAMATH_GPT_area_of_octagon_l1280_128057


namespace NUMINAMATH_GPT_g_evaluation_l1280_128075

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then (2 * a * b - a + 3) / (3 * a)
  else (a * b - b - 1) / (-3 * b)

theorem g_evaluation : g 2 1 + g 2 4 = 7 / 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_g_evaluation_l1280_128075


namespace NUMINAMATH_GPT_person_last_name_length_l1280_128007

theorem person_last_name_length (samantha_lastname: ℕ) (bobbie_lastname: ℕ) (person_lastname: ℕ) 
  (h1: samantha_lastname + 3 = bobbie_lastname)
  (h2: bobbie_lastname - 2 = 2 * person_lastname)
  (h3: samantha_lastname = 7) :
  person_lastname = 4 :=
by 
  sorry

end NUMINAMATH_GPT_person_last_name_length_l1280_128007


namespace NUMINAMATH_GPT_probability_positive_ball_drawn_is_half_l1280_128082

-- Definition of the problem elements
def balls : List Int := [-1, 0, 2, 3]

-- Definition for the event of drawing a positive number
def is_positive (x : Int) : Bool := x > 0

-- The proof statement
theorem probability_positive_ball_drawn_is_half : 
  (List.filter is_positive balls).length / balls.length = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_positive_ball_drawn_is_half_l1280_128082


namespace NUMINAMATH_GPT_combined_girls_avg_l1280_128006

noncomputable def centralHS_boys_avg := 68
noncomputable def deltaHS_boys_avg := 78
noncomputable def combined_boys_avg := 74
noncomputable def centralHS_girls_avg := 72
noncomputable def deltaHS_girls_avg := 85
noncomputable def centralHS_combined_avg := 70
noncomputable def deltaHS_combined_avg := 80

theorem combined_girls_avg (C c D d : ℝ) 
  (h1 : (68 * C + 72 * c) / (C + c) = 70)
  (h2 : (78 * D + 85 * d) / (D + d) = 80)
  (h3 : (68 * C + 78 * D) / (C + D) = 74) :
  (3/7 * 72 + 4/7 * 85) = 79 := 
by 
  sorry

end NUMINAMATH_GPT_combined_girls_avg_l1280_128006


namespace NUMINAMATH_GPT_intersection_A1_B1_complement_A1_B1_union_A2_B2_l1280_128010

-- Problem 1: Intersection and Complement
def setA1 : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def setB1 : Set ℕ := {1, 2, 3}

theorem intersection_A1_B1 : (setA1 ∩ setB1) = {1, 2, 3} := by
  sorry

theorem complement_A1_B1 : {x : ℕ | x ∈ setA1 ∧ x ∉ setB1} = {4, 5, 6, 7, 8} := by
  sorry

-- Problem 2: Union
def setA2 : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def setB2 : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_A2_B2 : (setA2 ∪ setB2) = {x : ℝ | (-3 < x ∧ x < 1) ∨ (2 < x ∧ x < 10)} := by
  sorry

end NUMINAMATH_GPT_intersection_A1_B1_complement_A1_B1_union_A2_B2_l1280_128010


namespace NUMINAMATH_GPT_seq_inequality_l1280_128073

noncomputable def sequence_of_nonneg_reals (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, a (n + m) ≤ a n + a m

theorem seq_inequality
  (a : ℕ → ℝ)
  (h : sequence_of_nonneg_reals a)
  (h_nonneg : ∀ n, 0 ≤ a n) :
  ∀ n m : ℕ, m > 0 → n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := 
by
  sorry

end NUMINAMATH_GPT_seq_inequality_l1280_128073


namespace NUMINAMATH_GPT_average_growth_rate_income_prediction_l1280_128071

-- Define the given conditions
def income2018 : ℝ := 20000
def income2020 : ℝ := 24200
def growth_rate : ℝ := 0.1
def predicted_income2021 : ℝ := 26620

-- Lean 4 statement for the first part of the problem
theorem average_growth_rate :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  growth_rate = 0.1 :=
by
  intros h
  sorry

-- Lean 4 statement for the second part of the problem
theorem income_prediction :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  (growth_rate = 0.1) →
  (income2018 * (1 + growth_rate)^3 = predicted_income2021) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_average_growth_rate_income_prediction_l1280_128071


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1280_128060

theorem ellipse_eccentricity (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 16 = 1) ∧ (∃ e : ℝ, e = 3 / 4) ∧ (∀ c : ℝ, c = 3 / 4)
   → a = 7 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1280_128060


namespace NUMINAMATH_GPT_David_pushups_l1280_128059

-- Definitions and setup conditions
def Zachary_pushups : ℕ := 7
def additional_pushups : ℕ := 30

-- Theorem statement to be proved
theorem David_pushups 
  (zachary_pushups : ℕ) 
  (additional_pushups : ℕ) 
  (Zachary_pushups_val : zachary_pushups = Zachary_pushups) 
  (additional_pushups_val : additional_pushups = additional_pushups) :
  zachary_pushups + additional_pushups = 37 :=
sorry

end NUMINAMATH_GPT_David_pushups_l1280_128059


namespace NUMINAMATH_GPT_plain_pancakes_l1280_128019

/-- Define the given conditions -/
def total_pancakes : ℕ := 67
def blueberry_pancakes : ℕ := 20
def banana_pancakes : ℕ := 24

/-- Define a theorem stating the number of plain pancakes given the conditions -/
theorem plain_pancakes : total_pancakes - (blueberry_pancakes + banana_pancakes) = 23 := by
  -- Here we will provide a proof
  sorry

end NUMINAMATH_GPT_plain_pancakes_l1280_128019


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l1280_128032

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ x y : ℝ, ax + y + 1 = 0) ∧ (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ x y : ℝ, (y = -ax)) → a = -1 := by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l1280_128032


namespace NUMINAMATH_GPT_quadratic_has_unique_solution_l1280_128097

theorem quadratic_has_unique_solution (k : ℝ) :
  (∀ x : ℝ, (x + 6) * (x + 3) = k + 3 * x) → k = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_has_unique_solution_l1280_128097


namespace NUMINAMATH_GPT_domain_of_f_l1280_128099

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ x ∈ {x : ℝ | f x = f x} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1280_128099


namespace NUMINAMATH_GPT_sum_series_l1280_128021

theorem sum_series : (List.sum [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56, -59]) = -30 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_l1280_128021


namespace NUMINAMATH_GPT_directrix_of_parabola_l1280_128011

theorem directrix_of_parabola (y x : ℝ) : 
  (∃ a h k : ℝ, y = a * (x - h)^2 + k ∧ a = 1/8 ∧ h = 4 ∧ k = 0) → 
  y = -1/2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1280_128011


namespace NUMINAMATH_GPT_mans_rate_in_still_water_l1280_128031

/-- The man's rowing speed in still water given his rowing speeds with and against the stream. -/
theorem mans_rate_in_still_water (v_with_stream v_against_stream : ℝ) (h1 : v_with_stream = 6) (h2 : v_against_stream = 2) : (v_with_stream + v_against_stream) / 2 = 4 := by
  sorry

end NUMINAMATH_GPT_mans_rate_in_still_water_l1280_128031


namespace NUMINAMATH_GPT_solution_strategy_l1280_128016

-- Defining the total counts for the groups
def total_elderly : ℕ := 28
def total_middle_aged : ℕ := 54
def total_young : ℕ := 81

-- The sample size we need
def sample_size : ℕ := 36

-- Proposing the strategy
def appropriate_sampling_method : Prop := 
  (total_elderly - 1) % sample_size.gcd (total_middle_aged.gcd total_young) = 0

theorem solution_strategy :
  appropriate_sampling_method :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_strategy_l1280_128016


namespace NUMINAMATH_GPT_exists_digit_sum_divisible_by_11_l1280_128020

-- Define a function to compute the sum of the digits of a natural number
def digit_sum (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

-- The main theorem to be proven
theorem exists_digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k) % 11 = 0) := 
sorry

end NUMINAMATH_GPT_exists_digit_sum_divisible_by_11_l1280_128020


namespace NUMINAMATH_GPT_apps_addition_vs_deletion_l1280_128033

-- Defining the initial conditions
def initial_apps : ℕ := 21
def added_apps : ℕ := 89
def remaining_apps : ℕ := 24

-- The proof problem statement
theorem apps_addition_vs_deletion :
  added_apps - (initial_apps + added_apps - remaining_apps) = 3 :=
by
  sorry

end NUMINAMATH_GPT_apps_addition_vs_deletion_l1280_128033


namespace NUMINAMATH_GPT_range_of_m_l1280_128041

theorem range_of_m (a b m : ℝ) (h₀ : a > 0) (h₁ : b > 1) (h₂ : a + b = 2) (h₃ : ∀ m, (4/a + 1/(b-1)) > m^2 + 8*m) : -9 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1280_128041


namespace NUMINAMATH_GPT_find_T_l1280_128023

theorem find_T (T : ℝ) (h : (3/4) * (1/8) * T = (1/2) * (1/6) * 72) : T = 64 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_find_T_l1280_128023


namespace NUMINAMATH_GPT_original_price_of_article_l1280_128045

theorem original_price_of_article (SP : ℝ) (profit_rate : ℝ) (P : ℝ) (h1 : SP = 550) (h2 : profit_rate = 0.10) (h3 : SP = P * (1 + profit_rate)) : P = 500 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_article_l1280_128045


namespace NUMINAMATH_GPT_find_k_for_circle_l1280_128037

theorem find_k_for_circle (k : ℝ) : (∃ x y : ℝ, (x^2 + 8*x + y^2 + 4*y - k = 0) ∧ (x + 4)^2 + (y + 2)^2 = 25) → k = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_k_for_circle_l1280_128037


namespace NUMINAMATH_GPT_find_k_l1280_128094

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end NUMINAMATH_GPT_find_k_l1280_128094


namespace NUMINAMATH_GPT_parallel_lines_slope_equal_l1280_128005

theorem parallel_lines_slope_equal (k : ℝ) : (∀ x : ℝ, 2 * x = k * x + 3) → k = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_equal_l1280_128005


namespace NUMINAMATH_GPT_tan_shifted_value_l1280_128003

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end NUMINAMATH_GPT_tan_shifted_value_l1280_128003


namespace NUMINAMATH_GPT_budget_percentage_for_genetically_modified_organisms_l1280_128038

theorem budget_percentage_for_genetically_modified_organisms
  (microphotonics : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (industrial_lubricants : ℝ)
  (astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 15 →
  industrial_lubricants = 8 →
  astrophysics_degrees = 72 →
  (72 / 360) * 100 = 20 →
  100 - (14 + 24 + 15 + 8 + 20) = 19 :=
  sorry

end NUMINAMATH_GPT_budget_percentage_for_genetically_modified_organisms_l1280_128038


namespace NUMINAMATH_GPT_roger_has_more_candies_l1280_128098

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end NUMINAMATH_GPT_roger_has_more_candies_l1280_128098


namespace NUMINAMATH_GPT_total_kayaks_built_l1280_128086

/-- Geometric sequence sum definition -/
def geom_sum (a r : ℕ) (n : ℕ) : ℕ :=
  if r = 1 then n * a
  else a * (r ^ n - 1) / (r - 1)

/-- Problem statement: Prove that the total number of kayaks built by the end of June is 726 -/
theorem total_kayaks_built : geom_sum 6 3 5 = 726 :=
  sorry

end NUMINAMATH_GPT_total_kayaks_built_l1280_128086


namespace NUMINAMATH_GPT_work_completion_days_l1280_128079

variable (Paul_days Rose_days Sam_days : ℕ)

def Paul_rate := 1 / 80
def Rose_rate := 1 / 120
def Sam_rate := 1 / 150

def combined_rate := Paul_rate + Rose_rate + Sam_rate

noncomputable def days_to_complete_work := 1 / combined_rate

theorem work_completion_days :
  Paul_days = 80 →
  Rose_days = 120 →
  Sam_days = 150 →
  days_to_complete_work = 37 := 
by
  intros
  simp only [Paul_rate, Rose_rate, Sam_rate, combined_rate, days_to_complete_work]
  sorry

end NUMINAMATH_GPT_work_completion_days_l1280_128079


namespace NUMINAMATH_GPT_problem1_problem2_l1280_128002

-- Definitions of M and N
def setM : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def setN (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

-- Problem 1: Prove that if M ∩ N has only one element, then k = -1
theorem problem1 (h : ∀ x, x ∈ setM ∩ setN k → x = -1) : k = -1 := by 
  sorry

-- Problem 2: Given k = 2, prove the sets M ∩ N and M ∪ N
theorem problem2 (hk : k = 2) : (setM ∩ setN k = {x | -1 ≤ x ∧ x ≤ 2}) ∧ (setM ∪ setN k = {x | x ≤ 5}) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1280_128002


namespace NUMINAMATH_GPT_students_to_add_l1280_128043

theorem students_to_add (students := 1049) (teachers := 9) : ∃ n, students + n ≡ 0 [MOD teachers] ∧ n = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_students_to_add_l1280_128043


namespace NUMINAMATH_GPT_classroom_problem_l1280_128036

noncomputable def classroom_problem_statement : Prop :=
  ∀ (B G : ℕ) (b g : ℝ),
    b > 0 →
    g > 0 →
    B > 0 →
    G > 0 →
    ¬ ((B * g + G * b) / (B + G) = b + g ∧ b > 0 ∧ g > 0)

theorem classroom_problem : classroom_problem_statement :=
  by
    intros B G b g hb_gt0 hg_gt0 hB_gt0 hG_gt0
    sorry

end NUMINAMATH_GPT_classroom_problem_l1280_128036


namespace NUMINAMATH_GPT_ben_minimum_test_score_l1280_128030

theorem ben_minimum_test_score 
  (scores : List ℕ) 
  (current_avg : ℕ) 
  (desired_increase : ℕ) 
  (lowest_score : ℕ) 
  (required_score : ℕ) 
  (h_scores : scores = [95, 85, 75, 65, 90]) 
  (h_current_avg : current_avg = 82) 
  (h_desired_increase : desired_increase = 5) 
  (h_lowest_score : lowest_score = 65) 
  (h_required_score : required_score = 112) :
  (current_avg + desired_increase) = 87 ∧ 
  (6 * (current_avg + desired_increase)) = 522 ∧ 
  required_score = (522 - (95 + 85 + 75 + 65 + 90)) ∧ 
  (522 - (95 + 85 + 75 + 65 + 90)) > lowest_score :=
by 
  sorry

end NUMINAMATH_GPT_ben_minimum_test_score_l1280_128030


namespace NUMINAMATH_GPT_corrected_multiplication_result_l1280_128062

theorem corrected_multiplication_result :
  ∃ n : ℕ, 987 * n = 559989 ∧ 987 * n ≠ 559981 ∧ 559981 % 100 = 98 :=
by
  sorry

end NUMINAMATH_GPT_corrected_multiplication_result_l1280_128062


namespace NUMINAMATH_GPT_jack_bought_apples_l1280_128093

theorem jack_bought_apples :
  ∃ n : ℕ, 
    (∃ k : ℕ, k = 10 ∧ ∃ m : ℕ, m = 5 * 9 ∧ n = k + m) ∧ n = 55 :=
by
  sorry

end NUMINAMATH_GPT_jack_bought_apples_l1280_128093


namespace NUMINAMATH_GPT_lara_flowers_l1280_128064

theorem lara_flowers (M : ℕ) : 52 - M - (M + 6) - 16 = 0 → M = 15 :=
by
  sorry

end NUMINAMATH_GPT_lara_flowers_l1280_128064
