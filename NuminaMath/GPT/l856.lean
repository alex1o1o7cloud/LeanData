import Mathlib

namespace polynomial_transformation_l856_85642

theorem polynomial_transformation :
  ∀ (a h k : ℝ), (8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → a + h + k = -23.5 :=
by
  intros a h k h_eq
  sorry

end polynomial_transformation_l856_85642


namespace intersection_distance_l856_85670

theorem intersection_distance (p q : ℕ) (h1 : p = 65) (h2 : q = 2) :
  p - q = 63 := 
by
  sorry

end intersection_distance_l856_85670


namespace money_left_after_distributions_and_donations_l856_85615

theorem money_left_after_distributions_and_donations 
  (total_income : ℕ)
  (percent_to_children : ℕ)
  (percent_to_each_child : ℕ)
  (number_of_children : ℕ)
  (percent_to_wife : ℕ)
  (percent_to_orphan_house : ℕ)
  (remaining_income_percentage : ℕ)
  (children_distribution : ℕ → ℕ → ℕ)
  (wife_distribution : ℕ → ℕ)
  (calculate_remaining : ℕ → ℕ → ℕ)
  (calculate_donation : ℕ → ℕ → ℕ)
  (calculate_money_left : ℕ → ℕ → ℕ)
  (income : ℕ := 400000)
  (result : ℕ := 57000) :
  children_distribution percent_to_each_child number_of_children = 60 →
  percent_to_wife = 25 →
  remaining_income_percentage = 15 →
  percent_to_orphan_house = 5 →
  wife_distribution percent_to_wife = 100000 →
  calculate_remaining 100 85 = 15 →
  calculate_donation percent_to_orphan_house (calculate_remaining 100 85 * total_income) = 3000 →
  calculate_money_left (calculate_remaining 100 85 * total_income) 3000 = result →
  total_income = income →
  income - (60 * income / 100 + 25 * income / 100 + 5 * (15 * income / 100) / 100) = result
  :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end money_left_after_distributions_and_donations_l856_85615


namespace solution_set_of_quadratic_inequality_l856_85666

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 4 * x - 5 > 0} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l856_85666


namespace value_of_f_at_13_over_2_l856_85658

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_at_13_over_2
  (h1 : ∀ x : ℝ , f (-x) = -f (x))
  (h2 : ∀ x : ℝ , f (x - 2) = f (x + 2))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = -x^2) :
  f (13 / 2) = 9 / 4 :=
sorry

end value_of_f_at_13_over_2_l856_85658


namespace min_xy_positive_real_l856_85696

theorem min_xy_positive_real (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 / (2 + x) + 3 / (2 + y) = 1) :
  ∃ m : ℝ, m = 16 ∧ ∀ xy : ℝ, (xy = x * y) → xy ≥ m :=
by
  sorry

end min_xy_positive_real_l856_85696


namespace sum_of_coordinates_of_other_endpoint_l856_85628

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (midpoint_cond : (x + 1) / 2 = 3)
  (midpoint_cond2 : (y - 3) / 2 = 5) :
  x + y = 18 :=
sorry

end sum_of_coordinates_of_other_endpoint_l856_85628


namespace max_profit_at_nine_l856_85625

noncomputable def profit (x : ℝ) : ℝ := - (1 / 3) * x^3 + 81 * x - 23

theorem max_profit_at_nine :
  ∃ x : ℝ, x = 9 ∧ ∀ (ε : ℝ), ε > 0 → 
  (profit (9 - ε) < profit 9 ∧ profit (9 + ε) < profit 9) := 
by
  sorry

end max_profit_at_nine_l856_85625


namespace find_y_l856_85610

theorem find_y (y : ℝ) (h : (3 * y) / 7 = 12) : y = 28 :=
by
  -- The proof would go here
  sorry

end find_y_l856_85610


namespace x_is_integer_l856_85648

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ a : ℤ, a = x^1960 - x^1919)
  (h2 : ∃ b : ℤ, b = x^2001 - x^1960) :
  ∃ k : ℤ, x = k :=
sorry

end x_is_integer_l856_85648


namespace prob_diff_colors_correct_l856_85602

noncomputable def total_outcomes : ℕ :=
  let balls_pocket1 := 2 + 3 + 5
  let balls_pocket2 := 2 + 4 + 4
  balls_pocket1 * balls_pocket2

noncomputable def favorable_outcomes_same_color : ℕ :=
  let white_balls := 2 * 2
  let red_balls := 3 * 4
  let yellow_balls := 5 * 4
  white_balls + red_balls + yellow_balls

noncomputable def prob_same_color : ℚ :=
  favorable_outcomes_same_color / total_outcomes

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_diff_colors_correct :
  prob_different_color = 16 / 25 :=
by sorry

end prob_diff_colors_correct_l856_85602


namespace number_of_weeks_in_a_single_harvest_season_l856_85689

-- Define constants based on conditions
def weeklyEarnings : ℕ := 1357
def totalHarvestSeasons : ℕ := 73
def totalEarnings : ℕ := 22090603

-- Prove the number of weeks in a single harvest season
theorem number_of_weeks_in_a_single_harvest_season :
  (totalEarnings / weeklyEarnings) / totalHarvestSeasons = 223 := 
  by
    sorry

end number_of_weeks_in_a_single_harvest_season_l856_85689


namespace symmetry_condition_l856_85695

-- Define grid and initial conditions
def grid : Type := ℕ × ℕ
def is_colored (pos : grid) : Prop := 
  pos = (1,4) ∨ pos = (2,1) ∨ pos = (4,2)

-- Conditions for symmetry: horizontal and vertical line symmetry and 180-degree rotational symmetry
def is_symmetric_line (grid_size : grid) (pos : grid) : Prop :=
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨ 
  (pos.1 > grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2) ∨
  (pos.1 > grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2)

def grid_size : grid := (4, 5)
def add_squares_needed (num : ℕ) : Prop :=
  ∀ (pos : grid), is_symmetric_line grid_size pos → is_colored pos

theorem symmetry_condition : 
  ∃ n, add_squares_needed n ∧ n = 9
  := sorry

end symmetry_condition_l856_85695


namespace total_flowers_sold_l856_85605

def flowers_sold_on_monday : ℕ := 4
def flowers_sold_on_tuesday : ℕ := 8
def flowers_sold_on_friday : ℕ := 2 * flowers_sold_on_monday

theorem total_flowers_sold : flowers_sold_on_monday + flowers_sold_on_tuesday + flowers_sold_on_friday = 20 := by
  sorry

end total_flowers_sold_l856_85605


namespace max_profit_achieved_when_x_is_1_l856_85699

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2
noncomputable def fixed_costs : ℝ := 40
noncomputable def material_cost (x : ℕ) : ℝ := 5 * x
noncomputable def profit (x : ℕ) : ℝ := revenue x - (fixed_costs + material_cost x)
noncomputable def marginal_profit (x : ℕ) : ℝ := profit (x + 1) - profit x

theorem max_profit_achieved_when_x_is_1 :
  marginal_profit 1 = 24.40 :=
by
  -- Skip the proof
  sorry

end max_profit_achieved_when_x_is_1_l856_85699


namespace cost_formula_l856_85608

-- Given Conditions
def flat_fee := 5  -- flat service fee in cents
def first_kg_cost := 12  -- cost for the first kilogram in cents
def additional_kg_cost := 5  -- cost for each additional kilogram in cents

-- Integer weight in kilograms
variable (P : ℕ)

-- Total cost calculation proof problem
theorem cost_formula : ∃ C, C = flat_fee + first_kg_cost + additional_kg_cost * (P - 1) → C = 5 * P + 12 :=
by
  sorry

end cost_formula_l856_85608


namespace geom_series_sum_l856_85640

noncomputable def geom_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1) 

theorem geom_series_sum (S : ℕ) (a r n : ℕ) (eq1 : a = 1) (eq2 : r = 3)
  (eq3 : 19683 = a * r^(n-1)) (S_eq : S = geom_sum a r n) : 
  S = 29524 :=
by
  sorry

end geom_series_sum_l856_85640


namespace radius_of_larger_circle_l856_85690

theorem radius_of_larger_circle
  (r : ℝ) -- radius of the smaller circle
  (R : ℝ) -- radius of the larger circle
  (ratio : R = 4 * r) -- radii ratio 1:4
  (AC : ℝ) -- diameter of the larger circle
  (BC : ℝ) -- chord of the larger circle
  (AB : ℝ := 16) -- given condition AB = 16
  (diameter_AC : AC = 2 * R) -- AC is diameter of the larger circle
  (tangent : BC^2 = AB^2 + (2 * R)^2) -- Pythagorean theorem for the right triangle ABC
  :
  R = 32 := 
sorry

end radius_of_larger_circle_l856_85690


namespace notebook_problem_l856_85646

/-- Conditions:
1. If each notebook costs 3 yuan, 6 more notebooks can be bought.
2. If each notebook costs 5 yuan, there is a 30-yuan shortfall.

We need to show:
1. The total number of notebooks \( x \).
2. The number of 3-yuan notebooks \( n_3 \). -/
theorem notebook_problem (x y n3 : ℕ) (h1 : y = 3 * x + 18) (h2 : y = 5 * x - 30) (h3 : 3 * n3 + 5 * (x - n3) = y) :
  x = 24 ∧ n3 = 15 :=
by
  -- proof to be provided
  sorry

end notebook_problem_l856_85646


namespace print_rolls_sold_l856_85633

-- Defining the variables and conditions
def num_sold := 480
def total_amount := 2340
def solid_price := 4
def print_price := 6

-- Proposed theorem statement
theorem print_rolls_sold (S P : ℕ) (h1 : S + P = num_sold) (h2 : solid_price * S + print_price * P = total_amount) : P = 210 := sorry

end print_rolls_sold_l856_85633


namespace find_percentage_l856_85641

theorem find_percentage (P : ℝ) (h1 : (3 / 5) * 150 = 90) (h2 : (P / 100) * 90 = 36) : P = 40 :=
by
  sorry

end find_percentage_l856_85641


namespace polynomial_root_s_eq_pm1_l856_85680

theorem polynomial_root_s_eq_pm1
  (b_3 b_2 b_1 : ℤ)
  (s : ℤ)
  (h1 : s^3 ∣ 50)
  (h2 : (s^4 + b_3 * s^3 + b_2 * s^2 + b_1 * s + 50) = 0) :
  s = 1 ∨ s = -1 :=
sorry

end polynomial_root_s_eq_pm1_l856_85680


namespace initial_apples_l856_85693

theorem initial_apples (X : ℕ) (h : X - 2 + 3 = 5) : X = 4 :=
sorry

end initial_apples_l856_85693


namespace percentage_third_day_l856_85632

def initial_pieces : ℕ := 1000
def percentage_first_day : ℝ := 0.10
def percentage_second_day : ℝ := 0.20
def pieces_left_after_third_day : ℕ := 504

theorem percentage_third_day :
  let pieces_first_day := initial_pieces * percentage_first_day
  let remaining_after_first_day := initial_pieces - pieces_first_day
  let pieces_second_day := remaining_after_first_day * percentage_second_day
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day
  let pieces_third_day := remaining_after_second_day - pieces_left_after_third_day
  (pieces_third_day / remaining_after_second_day * 100 = 30) :=
by
  sorry

end percentage_third_day_l856_85632


namespace factorize_cubic_l856_85617

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l856_85617


namespace greatest_integer_less_than_M_over_100_l856_85619

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end greatest_integer_less_than_M_over_100_l856_85619


namespace num_invalid_d_l856_85627

noncomputable def square_and_triangle_problem (d : ℕ) : Prop :=
  ∃ a b : ℕ, 3 * a - 4 * b = 1989 ∧ a - b = d ∧ b > 0

theorem num_invalid_d : ∀ (d : ℕ), (d ≤ 663) → ¬ square_and_triangle_problem d :=
by {
  sorry
}

end num_invalid_d_l856_85627


namespace solution_statement_l856_85697

-- Define the set of courses
inductive Course
| Physics | Chemistry | Literature | History | Philosophy | Psychology

open Course

-- Define the condition that a valid program must include Physics and at least one of Chemistry or Literature
def valid_program (program : Finset Course) : Prop :=
  Course.Physics ∈ program ∧
  (Course.Chemistry ∈ program ∨ Course.Literature ∈ program)

-- Define the problem statement
def problem_statement : Prop :=
  ∃ programs : Finset (Finset Course),
    programs.card = 9 ∧ ∀ program ∈ programs, program.card = 5 ∧ valid_program program

theorem solution_statement : problem_statement := sorry

end solution_statement_l856_85697


namespace part1_part2_l856_85624

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l856_85624


namespace chocolate_flavored_cups_sold_l856_85684

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end chocolate_flavored_cups_sold_l856_85684


namespace quadratic_roots_r_l856_85660

theorem quadratic_roots_r (a b m p r : ℚ) :
  (∀ x : ℚ, x^2 - m * x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x : ℚ, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a + 1)) →
  r = 19 / 3 :=
by
  sorry

end quadratic_roots_r_l856_85660


namespace pizza_slices_with_both_l856_85694

theorem pizza_slices_with_both (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24) (h_pepperoni : pepperoni_slices = 15) (h_mushrooms : mushroom_slices = 14) :
  ∃ n, n = 5 ∧ total_slices = pepperoni_slices + mushroom_slices - n := 
by
  use 5
  sorry

end pizza_slices_with_both_l856_85694


namespace value_of_b_l856_85631

theorem value_of_b : (15^2 * 9^2 * 356 = 6489300) :=
by 
  sorry

end value_of_b_l856_85631


namespace complement_of_A_in_U_l856_85672

open Set

-- Define the sets U and A with their respective elements in the real numbers
def U : Set ℝ := Icc 0 1
def A : Set ℝ := Ico 0 1

-- State the theorem
theorem complement_of_A_in_U : (U \ A) = {1} := by
  sorry

end complement_of_A_in_U_l856_85672


namespace value_of_a_100_l856_85675

open Nat

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (succ k) => sequence k + 4

theorem value_of_a_100 : sequence 99 = 397 := by
  sorry

end value_of_a_100_l856_85675


namespace union_of_sets_l856_85636

theorem union_of_sets (A B : Set ℤ) (hA : A = {-1, 3}) (hB : B = {2, 3}) : A ∪ B = {-1, 2, 3} := 
by
  sorry

end union_of_sets_l856_85636


namespace smaller_than_neg3_l856_85663

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end smaller_than_neg3_l856_85663


namespace find_initial_shells_l856_85607

theorem find_initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end find_initial_shells_l856_85607


namespace area_of_plot_l856_85611

def cm_to_miles (a : ℕ) : ℕ := a * 9

def miles_to_acres (b : ℕ) : ℕ := b * 640

theorem area_of_plot :
  let bottom := 12
  let top := 18
  let height := 10
  let area_cm2 := ((bottom + top) * height) / 2
  let area_miles2 := cm_to_miles area_cm2
  let area_acres := miles_to_acres area_miles2
  area_acres = 864000 :=
by
  sorry

end area_of_plot_l856_85611


namespace lcm_of_numbers_is_91_l856_85620

def ratio (a b : ℕ) (p q : ℕ) : Prop := p * b = q * a

theorem lcm_of_numbers_is_91 (a b : ℕ) (h_ratio : ratio a b 7 13) (h_gcd : Nat.gcd a b = 15) :
  Nat.lcm a b = 91 := 
by sorry

end lcm_of_numbers_is_91_l856_85620


namespace range_of_a_l856_85652

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ (Set.Iio (-3) ∪ Set.Ioi 1) :=
by sorry

end range_of_a_l856_85652


namespace area_hexagon_DEFD_EFE_l856_85662

variable (D E F D' E' F' : Type)
variable (perimeter_DEF : ℝ) (radius_circumcircle : ℝ)
variable (area_hexagon : ℝ)

theorem area_hexagon_DEFD_EFE' (h1 : perimeter_DEF = 42)
    (h2 : radius_circumcircle = 7)
    (h_def : area_hexagon = 147) :
  area_hexagon = 147 := 
sorry

end area_hexagon_DEFD_EFE_l856_85662


namespace least_total_bananas_is_1128_l856_85669

noncomputable def least_total_bananas : ℕ :=
  let b₁ := 252
  let b₂ := 252
  let b₃ := 336
  let b₄ := 288
  b₁ + b₂ + b₃ + b₄

theorem least_total_bananas_is_1128 :
  least_total_bananas = 1128 :=
by
  sorry

end least_total_bananas_is_1128_l856_85669


namespace purely_imaginary_satisfies_condition_l856_85692

theorem purely_imaginary_satisfies_condition (m : ℝ) (h1 : m^2 + 3 * m - 4 = 0) (h2 : m + 4 ≠ 0) : m = 1 :=
by
  sorry

end purely_imaginary_satisfies_condition_l856_85692


namespace constant_term_in_expansion_l856_85691

-- Define the binomial expansion general term
def binomial_general_term (x : ℤ) (r : ℕ) : ℤ :=
  (-2)^r * 3^(5 - r) * (Nat.choose 5 r) * x^(10 - 5 * r)

-- Define the condition for the specific r that makes the exponent of x zero
def condition (r : ℕ) : Prop :=
  10 - 5 * r = 0

-- Define the constant term calculation
def const_term : ℤ :=
  4 * 27 * (Nat.choose 5 2)

-- Theorem statement
theorem constant_term_in_expansion : const_term = 1080 :=
by 
  -- The proof is omitted
  sorry

end constant_term_in_expansion_l856_85691


namespace kyungsoo_came_second_l856_85606

theorem kyungsoo_came_second
  (kyungsoo_jump : ℝ) (younghee_jump : ℝ) (jinju_jump : ℝ) (chanho_jump : ℝ)
  (h_kyungsoo : kyungsoo_jump = 2.3)
  (h_younghee : younghee_jump = 0.9)
  (h_jinju : jinju_jump = 1.8)
  (h_chanho : chanho_jump = 2.5) :
  kyungsoo_jump = 2.3 := 
by
  sorry

end kyungsoo_came_second_l856_85606


namespace kids_go_to_camp_l856_85673

variable (total_kids staying_home going_to_camp : ℕ)

theorem kids_go_to_camp (h1 : total_kids = 313473) (h2 : staying_home = 274865) (h3 : going_to_camp = total_kids - staying_home) :
  going_to_camp = 38608 :=
by
  sorry

end kids_go_to_camp_l856_85673


namespace mandy_cinnamon_amount_correct_l856_85674

def mandy_cinnamon_amount (nutmeg : ℝ) (cinnamon : ℝ) : Prop :=
  cinnamon = nutmeg + 0.17

theorem mandy_cinnamon_amount_correct :
  mandy_cinnamon_amount 0.5 0.67 :=
by
  sorry

end mandy_cinnamon_amount_correct_l856_85674


namespace region_in_quadrants_l856_85678

theorem region_in_quadrants (x y : ℝ) :
  (y > 3 * x) → (y > 5 - 2 * x) → (x > 0 ∧ y > 0) :=
by
  intros h₁ h₂
  sorry

end region_in_quadrants_l856_85678


namespace jade_transactions_l856_85677

-- Definitions for each condition
def transactions_mabel : ℕ := 90
def transactions_anthony : ℕ := transactions_mabel + (transactions_mabel / 10)
def transactions_cal : ℕ := 2 * transactions_anthony / 3
def transactions_jade : ℕ := transactions_cal + 17

-- The theorem stating that Jade handled 83 transactions
theorem jade_transactions : transactions_jade = 83 := by
  sorry

end jade_transactions_l856_85677


namespace dad_strawberries_final_weight_l856_85650

variable {M D : ℕ}

theorem dad_strawberries_final_weight :
  M + D = 22 →
  36 - M + 30 + D = D' →
  D' = 46 :=
by
  intros h h1
  sorry

end dad_strawberries_final_weight_l856_85650


namespace palindromes_between_300_800_l856_85698

def palindrome_count (l u : ℕ) : ℕ :=
  (u / 100 - l / 100 + 1) * 10

theorem palindromes_between_300_800 : palindrome_count 300 800 = 50 :=
by
  sorry

end palindromes_between_300_800_l856_85698


namespace alice_lost_second_game_l856_85612

/-- Alice, Belle, and Cathy had an arm-wrestling contest. In each game, two girls wrestled, while the third rested.
After each game, the winner played the next game against the girl who had rested.
Given that Alice played 10 times, Belle played 15 times, and Cathy played 17 times; prove Alice lost the second game. --/

theorem alice_lost_second_game (alice_plays : ℕ) (belle_plays : ℕ) (cathy_plays : ℕ) :
  alice_plays = 10 → belle_plays = 15 → cathy_plays = 17 → 
  ∃ (lost_second_game : String), lost_second_game = "Alice" := by
  intros hA hB hC
  sorry

end alice_lost_second_game_l856_85612


namespace solve_quadratic_eq_solve_cubic_eq_l856_85671

-- Problem 1: Solve (x-1)^2 = 9
theorem solve_quadratic_eq (x : ℝ) (h : (x - 1) ^ 2 = 9) : x = 4 ∨ x = -2 := 
by 
  sorry

-- Problem 2: Solve (x+3)^3 / 3 - 9 = 0
theorem solve_cubic_eq (x : ℝ) (h : (x + 3) ^ 3 / 3 - 9 = 0) : x = 0 := 
by 
  sorry

end solve_quadratic_eq_solve_cubic_eq_l856_85671


namespace no_fib_right_triangle_l856_85679

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem no_fib_right_triangle (n : ℕ) : 
  ¬ (fibonacci n)^2 + (fibonacci (n+1))^2 = (fibonacci (n+2))^2 := 
by 
  sorry

end no_fib_right_triangle_l856_85679


namespace arithmetic_sequence_common_difference_l856_85622

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 1)
  (h3 : a 3 = 11)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = d) : d = 5 :=
sorry

end arithmetic_sequence_common_difference_l856_85622


namespace max_m_value_l856_85637

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem max_m_value 
  (t : ℝ) 
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) <= x) : m ≤ 4 :=
sorry

end max_m_value_l856_85637


namespace arithmetic_geometric_sequence_l856_85609

theorem arithmetic_geometric_sequence (d : ℤ) (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = a (n - 1) * a (n + 2)) :
  a 2017 = 1 :=
sorry

end arithmetic_geometric_sequence_l856_85609


namespace convert_to_base8_l856_85656

theorem convert_to_base8 (n : ℕ) (h : n = 1024) : 
  (∃ (d3 d2 d1 d0 : ℕ), n = d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0 ∧ d3 = 2 ∧ d2 = 0 ∧ d1 = 0 ∧ d0 = 0) :=
by
  sorry

end convert_to_base8_l856_85656


namespace find_counterfeit_two_weighings_l856_85616

-- defining the variables and conditions
variable (coins : Fin 7 → ℝ)
variable (real_weight : ℝ)
variable (fake_weight : ℝ)
variable (is_counterfeit : Fin 7 → Prop)

-- conditions
axiom counterfeit_weight_diff : ∀ i, is_counterfeit i ↔ (coins i = fake_weight)
axiom consecutive_counterfeits : ∃ (start : Fin 7), ∀ i, (start ≤ i ∧ i < start + 4) → is_counterfeit (i % 7)
axiom weight_diff : fake_weight < real_weight

-- Theorem statement
theorem find_counterfeit_two_weighings : 
  (coins (1 : Fin 7) + coins (2 : Fin 7) = coins (4 : Fin 7) + coins (5 : Fin 7)) →
  is_counterfeit (6 : Fin 7) ∧ is_counterfeit (7 : Fin 7) := 
sorry

end find_counterfeit_two_weighings_l856_85616


namespace sum_of_x_and_y_l856_85688

theorem sum_of_x_and_y 
  (x y : ℤ)
  (h1 : x - y = 36) 
  (h2 : x = 28) : 
  x + y = 20 :=
by 
  sorry

end sum_of_x_and_y_l856_85688


namespace complement_set_l856_85661

def U := {x : ℝ | x > 0}
def A := {x : ℝ | x > 2}
def complement_U_A := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem complement_set :
  {x : ℝ | x ∈ U ∧ x ∉ A} = complement_U_A :=
sorry

end complement_set_l856_85661


namespace factorization_example_l856_85682

theorem factorization_example : 
  ∀ (a : ℝ), a^2 - 6 * a + 9 = (a - 3)^2 :=
by
  intro a
  sorry

end factorization_example_l856_85682


namespace digit_difference_l856_85653

theorem digit_difference (X Y : ℕ) (h1 : 0 ≤ X ∧ X ≤ 9) (h2 : 0 ≤ Y ∧ Y ≤ 9) (h3 : (10 * X + Y) - (10 * Y + X) = 54) : X - Y = 6 :=
sorry

end digit_difference_l856_85653


namespace domain_of_composite_function_l856_85603

theorem domain_of_composite_function :
  ∀ (f : ℝ → ℝ), (∀ x, -1 ≤ x ∧ x ≤ 3 → ∃ y, f x = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f (2*x - 1) = y) :=
by
  intros f domain_f x hx
  sorry

end domain_of_composite_function_l856_85603


namespace derivative_at_0_l856_85618

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else x^2 * Real.exp (|x|) * Real.sin (1 / x^2)

theorem derivative_at_0 : deriv f 0 = 0 := by
  sorry

end derivative_at_0_l856_85618


namespace below_sea_level_representation_l856_85630

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l856_85630


namespace derivative_f_at_1_l856_85654

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end derivative_f_at_1_l856_85654


namespace stratified_sampling_total_students_sampled_l856_85621

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end stratified_sampling_total_students_sampled_l856_85621


namespace infinite_arith_prog_contains_infinite_nth_powers_l856_85601

theorem infinite_arith_prog_contains_infinite_nth_powers
  (a d : ℕ) (n : ℕ) 
  (h_pos: 0 < d) 
  (h_power: ∃ k : ℕ, ∃ m : ℕ, a + k * d = m^n) :
  ∃ infinitely_many k : ℕ, ∃ m : ℕ, a + k * d = m^n :=
sorry

end infinite_arith_prog_contains_infinite_nth_powers_l856_85601


namespace maximize_q_l856_85687

noncomputable def maximum_q (X Y Z : ℕ) : ℕ :=
X * Y * Z + X * Y + Y * Z + Z * X

theorem maximize_q : ∃ (X Y Z : ℕ), X + Y + Z = 15 ∧ (∀ (A B C : ℕ), A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) ∧ maximum_q X Y Z = 200 :=
by
  sorry

end maximize_q_l856_85687


namespace arithmetic_sequence_sum_a3_a4_a5_l856_85647

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_a3_a4_a5
  (ha : is_arithmetic_sequence a d)
  (h : a 2 + a 3 + a 4 = 12) : 
  (7 * (a 0 + a 6)) / 2 = 28 := 
sorry

end arithmetic_sequence_sum_a3_a4_a5_l856_85647


namespace garbage_bill_problem_l856_85683

theorem garbage_bill_problem
  (R : ℝ)
  (trash_bins : ℝ := 2)
  (recycling_bins : ℝ := 1)
  (weekly_trash_cost_per_bin : ℝ := 10)
  (weeks_per_month : ℝ := 4)
  (discount_rate : ℝ := 0.18)
  (fine : ℝ := 20)
  (final_bill : ℝ := 102) :
  (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  - discount_rate * (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  + fine = final_bill →
  R = 5 := 
by
  sorry

end garbage_bill_problem_l856_85683


namespace z_in_fourth_quadrant_l856_85638

-- Given complex numbers z1 and z2
def z1 : ℂ := 3 - 2 * Complex.I
def z2 : ℂ := 1 + Complex.I

-- Define the multiplication of z1 and z2
def z : ℂ := z1 * z2

-- Prove that z is located in the fourth quadrant
theorem z_in_fourth_quadrant : z.re > 0 ∧ z.im < 0 :=
by
  -- Construction and calculations skipped for the math proof,
  -- the result should satisfy the conditions for being in the fourth quadrant
  sorry

end z_in_fourth_quadrant_l856_85638


namespace total_cost_is_58_l856_85639

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end total_cost_is_58_l856_85639


namespace negation_of_proposition_l856_85685

theorem negation_of_proposition (a : ℝ) :
  (¬ (∀ x : ℝ, (x - a) ^ 2 + 2 > 0)) ↔ (∃ x : ℝ, (x - a) ^ 2 + 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l856_85685


namespace jim_taxi_distance_l856_85664

theorem jim_taxi_distance (initial_fee charge_per_segment total_charge : ℝ) (segment_len_miles : ℝ)
(init_fee_eq : initial_fee = 2.5)
(charge_per_seg_eq : charge_per_segment = 0.35)
(total_charge_eq : total_charge = 5.65)
(segment_length_eq : segment_len_miles = 2/5):
  let charge_for_distance := total_charge - initial_fee
  let num_segments := charge_for_distance / charge_per_segment
  let total_miles := num_segments * segment_len_miles
  total_miles = 3.6 :=
by
  intros
  sorry

end jim_taxi_distance_l856_85664


namespace sum_tens_ones_digits_3_plus_4_power_17_l856_85665

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l856_85665


namespace find_k_inv_h_8_l856_85667

variable (h k : ℝ → ℝ)

-- Conditions
axiom h_inv_k_x (x : ℝ) : h⁻¹ (k x) = 3 * x - 4
axiom h_3x_minus_4 (x : ℝ) : k x = h (3 * x - 4)

-- The statement we want to prove
theorem find_k_inv_h_8 : k⁻¹ (h 8) = 8 := 
  sorry

end find_k_inv_h_8_l856_85667


namespace find_constants_l856_85614

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![0, 4]
]

def I : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![1, 0],
  ![0, 1]
]

theorem find_constants (c d : ℚ) : 
  (N⁻¹ = c • N + d • I) ↔ (c = -1/12 ∧ d = 7/12) :=
by
  sorry

end find_constants_l856_85614


namespace mia_weight_l856_85613

theorem mia_weight (a m : ℝ) (h1 : a + m = 220) (h2 : m - a = 2 * a) : m = 165 :=
sorry

end mia_weight_l856_85613


namespace total_distance_correct_l856_85686

def d1 : ℕ := 350
def d2 : ℕ := 375
def d3 : ℕ := 275
def total_distance : ℕ := 1000

theorem total_distance_correct : d1 + d2 + d3 = total_distance := by
  sorry

end total_distance_correct_l856_85686


namespace no_equal_partition_product_l856_85655

theorem no_equal_partition_product (n : ℕ) (h : n > 1) : 
  ¬ ∃ A B : Finset ℕ, 
    (A ∪ B = (Finset.range n).erase 0 ∧ A ∩ B = ∅ ∧ (A ≠ ∅) ∧ (B ≠ ∅) 
    ∧ A.prod id = B.prod id) := 
sorry

end no_equal_partition_product_l856_85655


namespace parallelogram_s_value_l856_85668

noncomputable def parallelogram_area (s : ℝ) : ℝ :=
  s * 2 * (s / Real.sqrt 2)

theorem parallelogram_s_value (s : ℝ) (h₀ : parallelogram_area s = 8 * Real.sqrt 2) : 
  s = 2 * Real.sqrt 2 :=
by
  sorry

end parallelogram_s_value_l856_85668


namespace quadrant_angle_l856_85629

theorem quadrant_angle (θ : ℝ) (k : ℤ) (h_theta : 0 < θ ∧ θ < 90) : 
  ((180 * k + θ) % 360 < 90) ∨ (180 * k + θ) % 360 ≥ 180 ∧ (180 * k + θ) % 360 < 270 :=
sorry

end quadrant_angle_l856_85629


namespace Willy_Lucy_more_crayons_l856_85623

def Willy_initial : ℕ := 1400
def Lucy_initial : ℕ := 290
def Max_crayons : ℕ := 650
def Willy_giveaway_percent : ℚ := 25 / 100
def Lucy_giveaway_percent : ℚ := 10 / 100

theorem Willy_Lucy_more_crayons :
  let Willy_remaining := Willy_initial - Willy_initial * Willy_giveaway_percent
  let Lucy_remaining := Lucy_initial - Lucy_initial * Lucy_giveaway_percent
  Willy_remaining + Lucy_remaining - Max_crayons = 661 := by
  sorry

end Willy_Lucy_more_crayons_l856_85623


namespace derek_age_l856_85645

theorem derek_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end derek_age_l856_85645


namespace volume_of_rectangular_prism_l856_85657

-- Defining the conditions as assumptions
variables (l w h : ℝ) 
variable (lw_eq : l * w = 10)
variable (wh_eq : w * h = 14)
variable (lh_eq : l * h = 35)

-- Stating the theorem to prove
theorem volume_of_rectangular_prism : l * w * h = 70 :=
by
  have lw := lw_eq
  have wh := wh_eq
  have lh := lh_eq
  sorry

end volume_of_rectangular_prism_l856_85657


namespace min_value_reciprocal_sum_l856_85643

theorem min_value_reciprocal_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 2) : 
  (∃ c : ℝ, c = (1/x) + (1/y) + (1/z) ∧ c ≥ 9/2) :=
by
  -- proof would go here
  sorry

end min_value_reciprocal_sum_l856_85643


namespace incorrect_statement_D_l856_85604

noncomputable def f : ℝ → ℝ := sorry

axiom A1 : ∃ x : ℝ, f x ≠ 0
axiom A2 : ∀ x : ℝ, f (x + 1) = -f (2 - x)
axiom A3 : ∀ x : ℝ, f (x + 3) = f (x - 3)

theorem incorrect_statement_D :
  ¬ (∀ x : ℝ, f (3 + x) + f (3 - x) = 0) :=
sorry

end incorrect_statement_D_l856_85604


namespace all_real_possible_values_l856_85626

theorem all_real_possible_values 
  (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 1) : 
  ∃ r : ℝ, r = (a^4 + b^4 + c^4) / (ab + bc + ca) :=
sorry

end all_real_possible_values_l856_85626


namespace evaluate_expression_l856_85659

theorem evaluate_expression : 
  (2 ^ 2015 + 2 ^ 2013 + 2 ^ 2011) / (2 ^ 2015 - 2 ^ 2013 + 2 ^ 2011) = 21 / 13 := 
by 
 sorry

end evaluate_expression_l856_85659


namespace rectangle_area_invariant_l856_85635

theorem rectangle_area_invariant (l w : ℝ) (A : ℝ) 
  (h0 : A = l * w)
  (h1 : A = (l + 3) * (w - 1))
  (h2 : A = (l - 1.5) * (w + 2)) :
  A = 13.5 :=
by
  sorry

end rectangle_area_invariant_l856_85635


namespace average_speed_is_69_l856_85676

-- Definitions for the conditions
def distance_hr1 : ℕ := 90
def distance_hr2 : ℕ := 30
def distance_hr3 : ℕ := 60
def distance_hr4 : ℕ := 120
def distance_hr5 : ℕ := 45
def total_distance : ℕ := distance_hr1 + distance_hr2 + distance_hr3 + distance_hr4 + distance_hr5
def total_time : ℕ := 5

-- The theorem to be proven
theorem average_speed_is_69 :
  (total_distance / total_time) = 69 :=
by
  sorry

end average_speed_is_69_l856_85676


namespace determinant_value_l856_85681

-- Define the determinant calculation for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the initial conditions
variables {x : ℝ}
axiom h : x^2 - 3*x + 1 = 0

-- State the theorem to be proved
theorem determinant_value : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = 1 :=
by
  sorry

end determinant_value_l856_85681


namespace solve_system_of_equations_l856_85644

def sys_eq1 (x y : ℝ) : Prop := 6 * (1 - x) ^ 2 = 1 / y
def sys_eq2 (x y : ℝ) : Prop := 6 * (1 - y) ^ 2 = 1 / x

theorem solve_system_of_equations (x y : ℝ) :
  sys_eq1 x y ∧ sys_eq2 x y ↔
  ((x = 3 / 2 ∧ y = 2 / 3) ∨
   (x = 2 / 3 ∧ y = 3 / 2) ∨
   (x = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)) ∧ y = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)))) :=
sorry

end solve_system_of_equations_l856_85644


namespace range_of_m_l856_85600

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + |x - 1| > m) → m < 1 :=
by
  sorry

end range_of_m_l856_85600


namespace proposition_B_correct_l856_85634

theorem proposition_B_correct : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x :=
by
  sorry

end proposition_B_correct_l856_85634


namespace daily_wage_of_man_l856_85651

-- Define the wages for men and women
variables (M W : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 24 * M + 16 * W = 11600
def condition2 : Prop := 12 * M + 37 * W = 11600

-- The theorem we want to prove
theorem daily_wage_of_man (h1 : condition1 M W) (h2 : condition2 M W) : M = 350 :=
by
  sorry

end daily_wage_of_man_l856_85651


namespace large_envelopes_count_l856_85649

theorem large_envelopes_count
  (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ)
  (H1 : total_letters = 80)
  (H2 : small_envelope_letters = 20)
  (H3 : letters_per_large_envelope = 2) :
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 :=
sorry

end large_envelopes_count_l856_85649
