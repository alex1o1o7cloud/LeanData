import Mathlib

namespace average_death_rate_l1215_121544

-- Definitions of the given conditions
def birth_rate_two_seconds := 10
def net_increase_one_day := 345600
def seconds_per_day := 24 * 60 * 60 

-- Define the theorem to be proven
theorem average_death_rate :
  (birth_rate_two_seconds / 2) - (net_increase_one_day / seconds_per_day) = 1 :=
by 
  sorry

end average_death_rate_l1215_121544


namespace deepak_age_l1215_121557

-- Defining the problem with the given conditions in Lean:
theorem deepak_age (x : ℕ) (rahul_current : ℕ := 4 * x) (deepak_current : ℕ := 3 * x) :
  (rahul_current + 6 = 38) → (deepak_current = 24) :=
by
  sorry

end deepak_age_l1215_121557


namespace positive_difference_two_solutions_abs_eq_15_l1215_121510

theorem positive_difference_two_solutions_abs_eq_15 :
  ∀ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 > x2) → (x1 - x2 = 30) :=
by
  intros x1 x2 h
  sorry

end positive_difference_two_solutions_abs_eq_15_l1215_121510


namespace pizza_toppings_combination_l1215_121592

def num_combinations {α : Type} (s : Finset α) (k : ℕ) : ℕ :=
  (s.card.choose k)

theorem pizza_toppings_combination (s : Finset ℕ) (h : s.card = 7) : num_combinations s 3 = 35 :=
by
  sorry

end pizza_toppings_combination_l1215_121592


namespace cars_on_river_road_l1215_121549

-- Define the given conditions
variables (B C : ℕ)
axiom ratio_condition : B = C / 13
axiom difference_condition : B = C - 60 

-- State the theorem to be proved
theorem cars_on_river_road : C = 65 :=
by
  -- proof would go here 
  sorry

end cars_on_river_road_l1215_121549


namespace tan_angle_sum_l1215_121573

variable (α β : ℝ)

theorem tan_angle_sum (h1 : Real.tan (α - Real.pi / 6) = 3 / 7)
                      (h2 : Real.tan (Real.pi / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
by
  sorry

end tan_angle_sum_l1215_121573


namespace sum_of_squares_of_coefficients_l1215_121570

theorem sum_of_squares_of_coefficients :
  ∃ a b c d e f : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 :=
by
  sorry

end sum_of_squares_of_coefficients_l1215_121570


namespace rectangular_prism_dimensions_l1215_121518

theorem rectangular_prism_dimensions (b l h : ℕ) 
  (h1 : l = 3 * b) 
  (h2 : l = 2 * h) 
  (h3 : l * b * h = 12168) :
  b = 14 ∧ l = 42 ∧ h = 21 :=
by
  -- The proof will go here
  sorry

end rectangular_prism_dimensions_l1215_121518


namespace wrapping_paper_fraction_l1215_121547

theorem wrapping_paper_fraction (s l : ℚ) (h1 : 4 * s + 2 * l = 5 / 12) (h2 : l = 2 * s) :
  s = 5 / 96 ∧ l = 5 / 48 :=
by
  sorry

end wrapping_paper_fraction_l1215_121547


namespace a_gt_one_l1215_121566

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 - x - 1

theorem a_gt_one (a : ℝ) :
  (∃! x, 0 < x ∧ x < 1 ∧ f a x = 0) → 1 < a :=
by
  sorry

end a_gt_one_l1215_121566


namespace candy_pieces_per_package_l1215_121533

theorem candy_pieces_per_package (packages_gum : ℕ) (packages_candy : ℕ) (total_candies : ℕ) :
  packages_gum = 21 →
  packages_candy = 45 →
  total_candies = 405 →
  total_candies / packages_candy = 9 := by
  intros h1 h2 h3
  sorry

end candy_pieces_per_package_l1215_121533


namespace money_left_l1215_121563

def initial_money : ℝ := 18
def spent_on_video_games : ℝ := 6
def spent_on_snack : ℝ := 3
def toy_original_cost : ℝ := 4
def toy_discount : ℝ := 0.25

theorem money_left (initial_money spent_on_video_games spent_on_snack toy_original_cost toy_discount : ℝ) :
  initial_money = 18 →
  spent_on_video_games = 6 →
  spent_on_snack = 3 →
  toy_original_cost = 4 →
  toy_discount = 0.25 →
  (initial_money - (spent_on_video_games + spent_on_snack + (toy_original_cost * (1 - toy_discount)))) = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end money_left_l1215_121563


namespace sum_of_areas_of_rectangles_l1215_121580

theorem sum_of_areas_of_rectangles :
  let width := 2
  let lengths := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => l * width)
  let total_area := areas.sum
  total_area = 182 := by
  sorry

end sum_of_areas_of_rectangles_l1215_121580


namespace price_Ramesh_paid_l1215_121521

-- Define the conditions
def labelled_price_sold (P : ℝ) := 1.10 * P
def discount_price_paid (P : ℝ) := 0.80 * P
def additional_costs := 125 + 250
def total_cost (P : ℝ) := discount_price_paid P + additional_costs

-- The main theorem stating that given the conditions,
-- the price Ramesh paid for the refrigerator is Rs. 13175.
theorem price_Ramesh_paid (P : ℝ) (H : labelled_price_sold P = 17600) :
  total_cost P = 13175 :=
by
  -- Providing a placeholder, as we do not need to provide the proof steps in the problem formulation
  sorry

end price_Ramesh_paid_l1215_121521


namespace express_c_in_terms_of_a_b_l1215_121539

-- Defining the vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)

-- Defining the given vectors
def a := vec 1 1
def b := vec 1 (-1)
def c := vec (-1) 2

-- The statement
theorem express_c_in_terms_of_a_b :
  c = (1/2) • a + (-3/2) • b :=
sorry

end express_c_in_terms_of_a_b_l1215_121539


namespace algebra_expr_value_l1215_121552

theorem algebra_expr_value (x y : ℝ) (h : x - 2 * y = 3) : 4 * y + 1 - 2 * x = -5 :=
sorry

end algebra_expr_value_l1215_121552


namespace find_triples_l1215_121574

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x^2 + y^2 = 3 * 2016^z + 77) :
  (x = 4 ∧ y = 8 ∧ z = 0) ∨ (x = 8 ∧ y = 4 ∧ z = 0) ∨
  (x = 14 ∧ y = 77 ∧ z = 1) ∨ (x = 77 ∧ y = 14 ∧ z = 1) ∨
  (x = 35 ∧ y = 70 ∧ z = 1) ∨ (x = 70 ∧ y = 35 ∧ z = 1) :=
sorry

end find_triples_l1215_121574


namespace sequence_general_formula_l1215_121545

open Nat

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), n > 0 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a (n + 1)) * (a n) = 0

theorem sequence_general_formula :
  ∃ (a : ℕ → ℝ), seq a ∧ (a 1 = 1) ∧ (∀ (n : ℕ), n > 0 → a n = 1 / n) :=
by
  sorry

end sequence_general_formula_l1215_121545


namespace ultramen_defeat_monster_in_5_minutes_l1215_121541

theorem ultramen_defeat_monster_in_5_minutes :
  ∀ (attacksRequired : ℕ) (attackRate1 attackRate2 : ℕ),
    (attacksRequired = 100) →
    (attackRate1 = 12) →
    (attackRate2 = 8) →
    (attacksRequired / (attackRate1 + attackRate2) = 5) :=
by
  intros
  sorry

end ultramen_defeat_monster_in_5_minutes_l1215_121541


namespace original_number_one_more_reciprocal_is_11_over_5_l1215_121577

theorem original_number_one_more_reciprocal_is_11_over_5 (x : ℚ) (h : 1 + 1/x = 11/5) : x = 5/6 :=
by
  sorry

end original_number_one_more_reciprocal_is_11_over_5_l1215_121577


namespace largest_integer_less_than_100_div_8_rem_5_l1215_121560

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l1215_121560


namespace positive_number_square_roots_l1215_121538

theorem positive_number_square_roots (a : ℝ) (x : ℝ) (h1 : x = (a - 7)^2)
  (h2 : x = (2 * a + 1)^2) : x = 25 := by
sorry

end positive_number_square_roots_l1215_121538


namespace alex_needs_additional_coins_l1215_121515

theorem alex_needs_additional_coins :
  let n := 15
  let current_coins := 63
  let target_sum := (n * (n + 1)) / 2
  let additional_coins := target_sum - current_coins
  additional_coins = 57 :=
by
  sorry

end alex_needs_additional_coins_l1215_121515


namespace compute_expression_l1215_121531

theorem compute_expression : 11 * (1 / 17) * 34 = 22 := 
sorry

end compute_expression_l1215_121531


namespace season_duration_l1215_121511

theorem season_duration (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 323) (h2 : games_per_month = 19) :
  (total_games / games_per_month) = 17 :=
by
  sorry

end season_duration_l1215_121511


namespace number_of_bricks_l1215_121530

noncomputable def bricklayer_one_hours : ℝ := 8
noncomputable def bricklayer_two_hours : ℝ := 12
noncomputable def reduction_rate : ℝ := 12
noncomputable def combined_hours : ℝ := 6

theorem number_of_bricks (y : ℝ) :
  ((combined_hours * ((y / bricklayer_one_hours) + (y / bricklayer_two_hours) - reduction_rate)) = y) →
  y = 288 :=
by sorry

end number_of_bricks_l1215_121530


namespace sequence_decreasing_l1215_121575

theorem sequence_decreasing : 
  ∀ (n : ℕ), n ≥ 1 → (1 / 2^(n - 1)) > (1 / 2^n) := 
by {
  sorry
}

end sequence_decreasing_l1215_121575


namespace solution_existence_l1215_121536

def problem_statement : Prop :=
  ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160

theorem solution_existence : problem_statement :=
  sorry

end solution_existence_l1215_121536


namespace hex_B2F_to_base10_l1215_121542

theorem hex_B2F_to_base10 :
  let b := 11
  let two := 2
  let f := 15
  let base := 16
  (b * base^2 + two * base^1 + f * base^0) = 2863 :=
by
  sorry

end hex_B2F_to_base10_l1215_121542


namespace least_value_a_l1215_121548

theorem least_value_a (a : ℤ) :
  (∃ a : ℤ, a ≥ 0 ∧ (a ^ 6) % 1920 = 0) → a = 8 ∧ (a ^ 6) % 1920 = 0 :=
by
  sorry

end least_value_a_l1215_121548


namespace min_value_of_a3_l1215_121562

open Real

theorem min_value_of_a3 (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n) (hgeo : ∀ n, a (n + 1) / a n = a 1 / a 0)
    (h : a 1 * a 2 * a 3 = a 1 + a 2 + a 3) : a 2 ≥ sqrt 3 := by {
  sorry
}

end min_value_of_a3_l1215_121562


namespace division_problem_l1215_121588

theorem division_problem : 0.05 / 0.0025 = 20 := 
sorry

end division_problem_l1215_121588


namespace find_sum_of_numbers_l1215_121504

theorem find_sum_of_numbers (x A B C : ℝ) (h1 : x > 0) (h2 : A = x) (h3 : B = 2 * x) (h4 : C = 3 * x) (h5 : A^2 + B^2 + C^2 = 2016) : A + B + C = 72 :=
sorry

end find_sum_of_numbers_l1215_121504


namespace right_triangle_leg_squared_l1215_121508

variable (a b c : ℝ)

theorem right_triangle_leg_squared (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_leg_squared_l1215_121508


namespace complete_square_solution_l1215_121501

-- Define the initial equation 
def equation_to_solve (x : ℝ) : Prop := x^2 - 4 * x = 6

-- Define the transformed equation after completing the square
def transformed_equation (x : ℝ) : Prop := (x - 2)^2 = 10

-- Prove that solving the initial equation using completing the square results in the transformed equation
theorem complete_square_solution : 
  ∀ x : ℝ, equation_to_solve x → transformed_equation x := 
by
  -- Proof will be provided here
  sorry

end complete_square_solution_l1215_121501


namespace remainder_of_M_div_by_51_is_zero_l1215_121512

open Nat

noncomputable def M := 1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950

theorem remainder_of_M_div_by_51_is_zero :
  M % 51 = 0 :=
sorry

end remainder_of_M_div_by_51_is_zero_l1215_121512


namespace other_root_l1215_121529

theorem other_root (m : ℝ) (h : 1^2 + m*1 + 3 = 0) : 
  ∃ α : ℝ, (1 + α = -m ∧ 1 * α = 3) ∧ α = 3 := 
by 
  sorry

end other_root_l1215_121529


namespace tangent_ellipse_hyperbola_l1215_121584

-- Definitions of the curves
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y+3)^2 = 1

-- Condition for tangency: the curves must meet and the discriminant must be zero
noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Prove the given curves are tangent at some x and y for m = 8/9
theorem tangent_ellipse_hyperbola : 
    (∃ x y : ℝ, ellipse x y ∧ hyperbola x y (8 / 9)) ∧ 
    quadratic_discriminant ((8 / 9) + 9) (6 * (8 / 9)) ((-8/9) * (8 * (8/9)) - 8) = 0 :=
sorry

end tangent_ellipse_hyperbola_l1215_121584


namespace domain_of_f_l1215_121532

def condition1 (x : ℝ) : Prop := 4 - |x| ≥ 0
def condition2 (x : ℝ) : Prop := (x^2 - 5 * x + 6) / (x - 3) > 0

theorem domain_of_f (x : ℝ) :
  (condition1 x) ∧ (condition2 x) ↔ ((2 < x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_l1215_121532


namespace egg_roll_ratio_l1215_121553

-- Define the conditions as hypotheses 
variables (Matthew_eats Patrick_eats Alvin_eats : ℕ)

-- Define the specific conditions
def conditions : Prop :=
  (Matthew_eats = 6) ∧
  (Patrick_eats = Alvin_eats / 2) ∧
  (Alvin_eats = 4)

-- Define the ratio of Matthew's egg rolls to Patrick's egg rolls
def ratio (a b : ℕ) := a / b

-- State the theorem with the corresponding proof problem
theorem egg_roll_ratio : conditions Matthew_eats Patrick_eats Alvin_eats → ratio Matthew_eats Patrick_eats = 3 :=
by
  -- Proof is not required as mentioned. Adding sorry to skip the proof.
  sorry

end egg_roll_ratio_l1215_121553


namespace minimum_triangle_area_l1215_121526

theorem minimum_triangle_area (r a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a = b) : 
  ∀ T, (T = (a + b) * r / 2) → T = 2 * r * r :=
by 
  sorry

end minimum_triangle_area_l1215_121526


namespace sin_cos_power_four_l1215_121561

theorem sin_cos_power_four (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := 
sorry

end sin_cos_power_four_l1215_121561


namespace inequality_solution_l1215_121595

theorem inequality_solution (x : ℝ) : 
  (x < -4 ∨ x > 2) ↔ (x^2 + 3 * x - 4) / (x^2 - x - 2) > 0 :=
sorry

end inequality_solution_l1215_121595


namespace Sergey_full_years_l1215_121555

def full_years (years months weeks days hours : ℕ) : ℕ :=
  years + months / 12 + (weeks * 7 + days) / 365

theorem Sergey_full_years 
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  years = 36 →
  months = 36 →
  weeks = 36 →
  days = 36 →
  hours = 36 →
  full_years years months weeks days hours = 39 :=
by
  intros
  sorry

end Sergey_full_years_l1215_121555


namespace appropriate_speech_length_l1215_121502

-- Condition 1: Speech duration in minutes
def speech_duration_min : ℝ := 30
def speech_duration_max : ℝ := 45

-- Condition 2: Ideal rate of speech in words per minute
def ideal_rate : ℝ := 150

-- Question translated into Lean proof statement
theorem appropriate_speech_length (n : ℝ) (h : n = 5650) :
  speech_duration_min * ideal_rate ≤ n ∧ n ≤ speech_duration_max * ideal_rate :=
by
  sorry

end appropriate_speech_length_l1215_121502


namespace solve_pounds_l1215_121589

def price_per_pound_corn : ℝ := 1.20
def price_per_pound_beans : ℝ := 0.60
def price_per_pound_rice : ℝ := 0.80
def total_weight : ℕ := 30
def total_cost : ℝ := 24.00
def equal_beans_rice (b r : ℕ) : Prop := b = r

theorem solve_pounds (c b r : ℕ) (h1 : price_per_pound_corn * ↑c + price_per_pound_beans * ↑b + price_per_pound_rice * ↑r = total_cost)
    (h2 : c + b + r = total_weight) (h3 : equal_beans_rice b r) : c = 6 ∧ b = 12 ∧ r = 12 := by
  sorry

end solve_pounds_l1215_121589


namespace find_a_minus_b_l1215_121581

theorem find_a_minus_b (a b : ℝ) (h1 : a + b = 12) (h2 : a^2 - b^2 = 48) : a - b = 4 :=
by
  sorry

end find_a_minus_b_l1215_121581


namespace range_of_a_l1215_121509

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2 * a * x - 2 else x + (36 / x) - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, piecewise_f a x ≥ piecewise_f a 2) ↔ (2 ≤ a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l1215_121509


namespace Jeff_pays_when_picking_up_l1215_121586

-- Definition of the conditions
def deposit_rate : ℝ := 0.10
def increase_rate : ℝ := 0.40
def last_year_cost : ℝ := 250
def this_year_cost : ℝ := last_year_cost * (1 + increase_rate)
def deposit : ℝ := this_year_cost * deposit_rate

-- Lean statement of the proof
theorem Jeff_pays_when_picking_up : this_year_cost - deposit = 315 := by
  sorry

end Jeff_pays_when_picking_up_l1215_121586


namespace min_sum_x8y4z_l1215_121519

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end min_sum_x8y4z_l1215_121519


namespace infinite_product_eq_four_four_thirds_l1215_121558

theorem infinite_product_eq_four_four_thirds :
  ∏' n : ℕ, (4^(n+1)^(1/(2^(n+1)))) = 4^(4/3) :=
sorry

end infinite_product_eq_four_four_thirds_l1215_121558


namespace total_volume_calculation_l1215_121517

noncomputable def total_volume_of_four_cubes (edge_length_in_feet : ℝ) (conversion_factor : ℝ) : ℝ :=
  let edge_length_in_meters := edge_length_in_feet * conversion_factor
  let volume_of_one_cube := edge_length_in_meters^3
  4 * volume_of_one_cube

theorem total_volume_calculation :
  total_volume_of_four_cubes 5 0.3048 = 14.144 :=
by
  -- Proof needs to be filled in.
  sorry

end total_volume_calculation_l1215_121517


namespace quadratic_transformation_l1215_121587

theorem quadratic_transformation (a b c : ℝ) (h : a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) : 
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end quadratic_transformation_l1215_121587


namespace general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l1215_121578
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 3*n - 1
noncomputable def c_n (n : ℕ) : ℚ := (3*n - 1) / 2^(n-1)

-- 1. Prove that the sequence {a_n} is given by a_n = 2^(n-1) and {b_n} is given by b_n = 3n - 1
theorem general_formulas :
  (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) ∧
  (∀ n : ℕ, n > 0 → b_n n = 3*n - 1) :=
sorry

-- 2. Prove that the values of n for which c_n > 1 are n = 1, 2, 3, 4
theorem values_of_n_for_c_n_gt_one :
  { n : ℕ | n > 0 ∧ c_n n > 1 } = {1, 2, 3, 4} :=
sorry

-- 3. Prove that no three terms from {a_n} can form an arithmetic sequence
theorem no_three_terms_arithmetic_seq :
  ∀ p q r : ℕ, p < q ∧ q < r ∧ p > 0 ∧ q > 0 ∧ r > 0 →
  ¬ (2 * a_n q = a_n p + a_n r) :=
sorry

end general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l1215_121578


namespace area_of_triangle_BQW_l1215_121590

theorem area_of_triangle_BQW (AZ WC AB : ℝ) (h_trap_area : ℝ) (h_eq : AZ = WC) (AZ_val : AZ = 8) (AB_val : AB = 16) (trap_area_val : h_trap_area = 160) : 
  ∃ (BQW_area: ℝ), BQW_area = 48 :=
by
  let h_2 := 2 * h_trap_area / (AZ + AB)
  let h := AZ + h_2
  let BZW_area := h_trap_area - (1 / 2) * AZ * AB
  let BQW_area := 1 / 2 * BZW_area
  have AZ_eq : AZ = 8 := AZ_val
  have AB_eq : AB = 16 := AB_val
  have trap_area_eq : h_trap_area = 160 := trap_area_val
  let h_2_val : ℝ := 10 -- Calculated from h_2 = 2 * 160 / 32
  let h_val : ℝ := AZ + h_2_val -- full height
  let BZW_area_val : ℝ := 96 -- BZW area from 160 - 64
  let BQW_area_val : ℝ := 48 -- Half of BZW
  exact ⟨48, by sorry⟩ -- To complete the theorem

end area_of_triangle_BQW_l1215_121590


namespace geometric_sequence_product_l1215_121535

theorem geometric_sequence_product
    (a : ℕ → ℝ)
    (r : ℝ)
    (h₀ : a 1 = 1 / 9)
    (h₃ : a 4 = 3)
    (h_geom : ∀ n, a (n + 1) = a n * r) :
    (a 1) * (a 2) * (a 3) * (a 4) * (a 5) = 1 :=
sorry

end geometric_sequence_product_l1215_121535


namespace lemon_juice_fraction_l1215_121513

theorem lemon_juice_fraction :
  ∃ L : ℚ, 30 - 30 * L - (1 / 3) * (30 - 30 * L) = 6 ∧ L = 7 / 10 :=
sorry

end lemon_juice_fraction_l1215_121513


namespace random_event_is_B_l1215_121520

variable (isCertain : Event → Prop)
variable (isImpossible : Event → Prop)
variable (isRandom : Event → Prop)

variable (A : Event)
variable (B : Event)
variable (C : Event)
variable (D : Event)

-- Here we set the conditions as definitions in Lean 4:
def condition_A : isCertain A := sorry
def condition_B : isRandom B := sorry
def condition_C : isCertain C := sorry
def condition_D : isImpossible D := sorry

-- The theorem we need to prove:
theorem random_event_is_B : isRandom B := 
by
-- adding sorry to skip the proof
sorry

end random_event_is_B_l1215_121520


namespace solution_l1215_121507

-- Define the discount conditions
def discount (price : ℕ) : ℕ :=
  if price > 22 then price * 7 / 10 else
  if price < 20 then price * 8 / 10 else
  price

-- Define the given book prices
def book_prices : List ℕ := [25, 18, 21, 35, 12, 10]

-- Calculate total cost using the discount function
def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (λ acc price => acc + discount price) 0

def problem_statement : Prop :=
  total_cost book_prices = 95

theorem solution : problem_statement :=
  by
  unfold problem_statement
  unfold total_cost
  simp [book_prices, discount]
  sorry

end solution_l1215_121507


namespace bronze_medals_l1215_121598

theorem bronze_medals (G S B : ℕ) 
  (h1 : G + S + B = 89) 
  (h2 : G + S = 4 * B - 6) :
  B = 19 :=
sorry

end bronze_medals_l1215_121598


namespace solve_system_l1215_121582

def inequality1 (x : ℝ) : Prop := 5 / (x + 3) ≥ 1

def inequality2 (x : ℝ) : Prop := x^2 + x - 2 ≥ 0

def solution (x : ℝ) : Prop := (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)

theorem solve_system (x : ℝ) : inequality1 x ∧ inequality2 x → solution x := by
  sorry

end solve_system_l1215_121582


namespace Auston_height_in_cm_l1215_121516

theorem Auston_height_in_cm : 
  (60 : ℝ) * 2.54 = 152.4 :=
by sorry

end Auston_height_in_cm_l1215_121516


namespace speed_ratio_l1215_121540

theorem speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (dist_before dist_after : ℝ) (total_dist : ℝ)
  (h1 : dist_before + dist_after = total_dist)
  (h2 : dist_before = 20)
  (h3 : dist_after = 20)
  (h4 : t2 = t1 + 11)
  (h5 : t2 = 22)
  (h6 : t1 = dist_before / v1)
  (h7 : t2 = dist_after / v2) :
  v1 / v2 = 2 := 
sorry

end speed_ratio_l1215_121540


namespace triangle_perimeter_l1215_121576

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l1215_121576


namespace chocolate_bar_cost_l1215_121593

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end chocolate_bar_cost_l1215_121593


namespace find_a5_l1215_121559

variable (a_n : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 
  (h1 : is_arithmetic_sequence a_n d)
  (h2 : a_n 3 + a_n 8 = 22)
  (h3 : a_n 6 = 7) :
  a_n 5 = 15 :=
sorry

end find_a5_l1215_121559


namespace simple_interest_rate_l1215_121569

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 800) (hSI : SI = 128) (hT : T = 4) : 
  (SI = P * (R : ℝ) * T / 100) → R = 4 := 
by {
  -- Proof goes here.
  sorry
}

end simple_interest_rate_l1215_121569


namespace tan_double_angle_l1215_121550

theorem tan_double_angle (θ : ℝ) 
  (h1 : Real.sin θ = 4 / 5) 
  (h2 : Real.sin θ - Real.cos θ > 1) : 
  Real.tan (2 * θ) = 24 / 7 := 
sorry

end tan_double_angle_l1215_121550


namespace john_toy_store_fraction_l1215_121579

theorem john_toy_store_fraction
  (allowance : ℝ)
  (spent_at_arcade_fraction : ℝ)
  (remaining_allowance : ℝ)
  (spent_at_candy_store : ℝ)
  (spent_at_toy_store : ℝ)
  (john_allowance : allowance = 3.60)
  (arcade_fraction : spent_at_arcade_fraction = 3 / 5)
  (arcade_amount : remaining_allowance = allowance - (spent_at_arcade_fraction * allowance))
  (candy_store_amount : spent_at_candy_store = 0.96)
  (remaining_after_candy_store : spent_at_toy_store = remaining_allowance - spent_at_candy_store)
  : spent_at_toy_store / remaining_allowance = 1 / 3 :=
by
  sorry

end john_toy_store_fraction_l1215_121579


namespace sum_of_seven_consecutive_integers_l1215_121506

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
  sorry

end sum_of_seven_consecutive_integers_l1215_121506


namespace goat_cow_difference_l1215_121594

-- Given the number of pigs (P), cows (C), and goats (G) on a farm
variables (P C G : ℕ)

-- Conditions:
def pig_count := P = 10
def cow_count_relationship := C = 2 * P - 3
def total_animals := P + C + G = 50

-- Theorem: The difference between the number of goats and cows
theorem goat_cow_difference (h1 : pig_count P)
                           (h2 : cow_count_relationship P C)
                           (h3 : total_animals P C G) :
  G - C = 6 := 
  sorry

end goat_cow_difference_l1215_121594


namespace number_of_three_leaf_clovers_l1215_121523

theorem number_of_three_leaf_clovers (total_leaves : ℕ) (three_leaf_clover : ℕ) (four_leaf_clover : ℕ) (n : ℕ)
  (h1 : total_leaves = 40) (h2 : three_leaf_clover = 3) (h3 : four_leaf_clover = 4) (h4: total_leaves = 3 * n + 4) :
  n = 12 :=
by
  sorry

end number_of_three_leaf_clovers_l1215_121523


namespace jonah_profit_is_correct_l1215_121565

noncomputable def jonah_profit : Real :=
  let pineapples := 6
  let pricePerPineapple := 3
  let pineappleCostWithoutDiscount := pineapples * pricePerPineapple
  let discount := if pineapples > 4 then 0.20 * pineappleCostWithoutDiscount else 0
  let totalCostAfterDiscount := pineappleCostWithoutDiscount - discount
  let ringsPerPineapple := 10
  let totalRings := pineapples * ringsPerPineapple
  let ringsSoldIndividually := 2
  let pricePerIndividualRing := 5
  let revenueFromIndividualRings := ringsSoldIndividually * pricePerIndividualRing
  let ringsLeft := totalRings - ringsSoldIndividually
  let ringsPerSet := 4
  let setsSold := ringsLeft / ringsPerSet -- This should be interpreted as an integer division
  let pricePerSet := 16
  let revenueFromSets := setsSold * pricePerSet
  let totalRevenue := revenueFromIndividualRings + revenueFromSets
  let profit := totalRevenue - totalCostAfterDiscount
  profit
  
theorem jonah_profit_is_correct :
  jonah_profit = 219.60 := by
  sorry

end jonah_profit_is_correct_l1215_121565


namespace smallest_value_m_plus_n_l1215_121525

theorem smallest_value_m_plus_n (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : m + n = 60 :=
sorry

end smallest_value_m_plus_n_l1215_121525


namespace initial_percentage_of_grape_juice_l1215_121556

theorem initial_percentage_of_grape_juice (P : ℝ) 
  (h₀ : 10 + 30 = 40)
  (h₁ : 40 * 0.325 = 13)
  (h₂ : 30 * P + 10 = 13) : 
  P = 0.1 :=
  by 
    sorry

end initial_percentage_of_grape_juice_l1215_121556


namespace find_asymptotes_l1215_121599

def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

def shifted_hyperbola_asymptotes (x y : ℝ) : Prop :=
  y = 4 / 3 * x + 5 ∨ y = -4 / 3 * x + 5

theorem find_asymptotes (x y : ℝ) :
  (∃ y', y = y' + 5 ∧ hyperbola_eq x y')
  ↔ shifted_hyperbola_asymptotes x y :=
by
  sorry

end find_asymptotes_l1215_121599


namespace part1_solution_part2_solution_l1215_121534

noncomputable def f (x a : ℝ) := |x + a| + |x - a|

theorem part1_solution : (∀ x : ℝ, f x 1 ≥ 4 ↔ x ∈ Set.Iic (-2) ∨ x ∈ Set.Ici 2) := by
  sorry

theorem part2_solution : (∀ x : ℝ, f x a ≥ 6 → a ∈ Set.Iic (-3) ∨ a ∈ Set.Ici 3) := by
  sorry

end part1_solution_part2_solution_l1215_121534


namespace quadratic_to_standard_form_l1215_121572

theorem quadratic_to_standard_form (a b c : ℝ) (x : ℝ) :
  (20 * x^2 + 240 * x + 3200 = a * (x + b)^2 + c) → (a + b + c = 2506) :=
  sorry

end quadratic_to_standard_form_l1215_121572


namespace remaining_water_l1215_121597

def initial_water : ℚ := 3
def water_used : ℚ := 4 / 3

theorem remaining_water : initial_water - water_used = 5 / 3 := 
by sorry -- skipping the proof for now

end remaining_water_l1215_121597


namespace rectangular_park_area_l1215_121524

/-- Define the conditions for the rectangular park -/
def rectangular_park (w l : ℕ) : Prop :=
  l = 3 * w ∧ 2 * (w + l) = 72

/-- Prove that the area of the rectangular park is 243 square meters -/
theorem rectangular_park_area (w l : ℕ) (h : rectangular_park w l) : w * l = 243 := by
  sorry

end rectangular_park_area_l1215_121524


namespace number_of_types_of_sliced_meat_l1215_121500

-- Define the constants and conditions
def varietyPackCostWithoutRush := 40.00
def rushDeliveryPercentage := 0.30
def costPerTypeWithRush := 13.00
def totalCostWithRush := varietyPackCostWithoutRush + (rushDeliveryPercentage * varietyPackCostWithoutRush)

-- Define the statement that needs to be proven
theorem number_of_types_of_sliced_meat :
  (totalCostWithRush / costPerTypeWithRush) = 4 := by
  sorry

end number_of_types_of_sliced_meat_l1215_121500


namespace ratio_of_diagonals_to_sides_l1215_121514

-- Define the given parameters and formula
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem
theorem ratio_of_diagonals_to_sides (n : ℕ) (h : n = 5) : 
  (num_diagonals n) / n = 1 :=
by
  -- Proof skipped
  sorry

end ratio_of_diagonals_to_sides_l1215_121514


namespace parallelogram_side_length_l1215_121583

theorem parallelogram_side_length (s : ℝ) (h : 3 * s * s * (1 / 2) = 27 * Real.sqrt 3) : 
  s = 3 * Real.sqrt (2 * Real.sqrt 3) :=
sorry

end parallelogram_side_length_l1215_121583


namespace sum_of_three_numbers_eq_zero_l1215_121527

theorem sum_of_three_numbers_eq_zero 
  (a b c : ℝ) 
  (h_sorted : a ≤ b ∧ b ≤ c) 
  (h_median : b = 10) 
  (h_mean_least : (a + b + c) / 3 = a + 20) 
  (h_mean_greatest : (a + b + c) / 3 = c - 10) 
  : a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l1215_121527


namespace monotonic_intervals_l1215_121522

noncomputable def y : ℝ → ℝ := λ x => x * Real.log x

theorem monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < (1 / Real.exp 1) → y x < -1) ∧ 
  (∀ x : ℝ, (1 / Real.exp 1) < x → x < 5 → y x > 1) := 
by
  sorry -- Proof goes here.

end monotonic_intervals_l1215_121522


namespace day_53_days_from_thursday_is_monday_l1215_121551

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end day_53_days_from_thursday_is_monday_l1215_121551


namespace ratio_shirt_to_coat_l1215_121554

-- Define the given conditions
def total_cost := 600
def shirt_cost := 150

-- Define the coat cost based on the given conditions
def coat_cost := total_cost - shirt_cost

-- State the theorem to prove the ratio of shirt cost to coat cost is 1:3
theorem ratio_shirt_to_coat : (shirt_cost : ℚ) / (coat_cost : ℚ) = 1 / 3 :=
by
  -- The proof would go here
  sorry

end ratio_shirt_to_coat_l1215_121554


namespace find_h3_l1215_121528

noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^9 + 1) - 1) / (x^(3^3 - 1) - 1)

theorem find_h3 : h 3 = 3 := by
  sorry

end find_h3_l1215_121528


namespace leak_out_time_l1215_121546

theorem leak_out_time (T_A T_full : ℝ) (h1 : T_A = 16) (h2 : T_full = 80) :
  ∃ T_B : ℝ, (1 / T_A - 1 / T_B = 1 / T_full) ∧ T_B = 80 :=
by {
  sorry
}

end leak_out_time_l1215_121546


namespace arithmetic_sequence_sum_l1215_121568

variable (a : ℕ → ℝ) (d : ℝ)
-- Conditions
def is_arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def condition : Prop := a 4 + a 8 = 8

-- Question
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a d →
  condition a →
  (11 / 2) * (a 1 + a 11) = 44 :=
by
  sorry

end arithmetic_sequence_sum_l1215_121568


namespace exists_rationals_leq_l1215_121596

theorem exists_rationals_leq (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f (a + b) / 2 :=
by
  sorry

end exists_rationals_leq_l1215_121596


namespace remaining_adults_fed_l1215_121567

theorem remaining_adults_fed 
  (cans : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (remaining_cans : ℕ)
  (remaining_adults : ℕ) :
  (adults_per_can = 4) →
  (children_per_can = 6) →
  (initial_cans = 7) →
  (children_fed = 18) →
  (remaining_cans = initial_cans - children_fed / children_per_can) →
  (remaining_adults = remaining_cans * adults_per_can) →
  remaining_adults = 16 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end remaining_adults_fed_l1215_121567


namespace problem1_problem2_part1_problem2_part2_l1215_121564

-- Problem 1
theorem problem1 (x : ℚ) (h : x = 11 / 12) : 
  (2 * x - 5) * (2 * x + 5) - (2 * x - 3) ^ 2 = -23 := 
by sorry

-- Problem 2
theorem problem2_part1 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  a^2 + b^2 = 22 := 
by sorry

theorem problem2_part2 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  (a - b)^2 = 8 := 
by sorry

end problem1_problem2_part1_problem2_part2_l1215_121564


namespace smallest_a_l1215_121537

theorem smallest_a (a b c : ℚ)
  (h1 : a > 0)
  (h2 : b = -2 * a / 3)
  (h3 : c = a / 9 - 5 / 9)
  (h4 : (a + b + c).den = 1) : a = 5 / 4 :=
by
  sorry

end smallest_a_l1215_121537


namespace digit_encoding_problem_l1215_121591

theorem digit_encoding_problem :
  ∃ (A B : ℕ), 0 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 21 * A + B = 111 * B ∧ A = 5 ∧ B = 5 :=
by
  sorry

end digit_encoding_problem_l1215_121591


namespace smallest_positive_integer_with_20_divisors_is_432_l1215_121543

-- Define the condition that a number n has exactly 20 positive divisors
def has_exactly_20_divisors (n : ℕ) : Prop :=
  ∃ (a₁ a₂ : ℕ), a₁ + 1 = 5 ∧ a₂ + 1 = 4 ∧
                n = 2^a₁ * 3^a₂

-- The main statement to prove
theorem smallest_positive_integer_with_20_divisors_is_432 :
  ∀ n : ℕ, has_exactly_20_divisors n → n = 432 :=
sorry

end smallest_positive_integer_with_20_divisors_is_432_l1215_121543


namespace car_mpg_city_l1215_121585

theorem car_mpg_city (h c t : ℕ) (H1 : 560 = h * t) (H2 : 336 = c * t) (H3 : c = h - 6) : c = 9 :=
by
  sorry

end car_mpg_city_l1215_121585


namespace compare_abc_l1215_121503

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 5^(2/3)

theorem compare_abc : c > a ∧ a > b := 
by
  sorry

end compare_abc_l1215_121503


namespace f_even_f_increasing_f_range_l1215_121571

variables {R : Type*} [OrderedRing R] (f : R → R)

-- Conditions
axiom f_mul : ∀ x y : R, f (x * y) = f x * f y
axiom f_neg1 : f (-1) = 1
axiom f_27 : f 27 = 9
axiom f_lt_1 : ∀ x : R, 0 ≤ x → x < 1 → 0 ≤ f x ∧ f x < 1

-- Questions
theorem f_even (x : R) : f x = f (-x) :=
by sorry

theorem f_increasing (x1 x2 : R) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 < x2) : f x1 < f x2 :=
by sorry

theorem f_range (a : R) (h1 : 0 ≤ a) (h2 : f (a + 1) ≤ 39) : 0 ≤ a ∧ a ≤ 2 :=
by sorry

end f_even_f_increasing_f_range_l1215_121571


namespace cycling_problem_l1215_121505

theorem cycling_problem (x : ℚ) (h1 : 25 * x + 15 * (7 - x) = 140) : x = 7 / 2 := 
sorry

end cycling_problem_l1215_121505
