import Mathlib

namespace total_test_subjects_l154_154116

-- Defining the conditions as mathematical entities
def number_of_colors : ℕ := 5
def unique_two_color_codes : ℕ := number_of_colors * number_of_colors
def excess_subjects : ℕ := 6

-- Theorem stating the question and correct answer
theorem total_test_subjects :
  unique_two_color_codes + excess_subjects = 31 :=
by
  -- Leaving the proof as sorry, since the task only requires statement creation
  sorry

end total_test_subjects_l154_154116


namespace g_is_zero_l154_154181

noncomputable def g (x : Real) : Real := 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)

theorem g_is_zero : ∀ x : Real, g x = 0 := by
  sorry

end g_is_zero_l154_154181


namespace find_x_between_0_and_180_l154_154827

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l154_154827


namespace impossible_gather_all_coins_in_one_sector_l154_154405

-- Definition of the initial condition with sectors and coins
def initial_coins_in_sectors := [1, 1, 1, 1, 1, 1] -- Each sector has one coin, represented by a list

-- Function to check if all coins are in one sector
def all_coins_in_one_sector (coins : List ℕ) := coins.count 6 == 1

-- Function to make a move (this is a helper; its implementation isn't necessary here but illustrates the idea)
def make_move (coins : List ℕ) (src dst : ℕ) : List ℕ := sorry

-- Proving that after 20 moves, coins cannot be gathered in one sector due to parity constraints
theorem impossible_gather_all_coins_in_one_sector : 
  ¬ ∃ (moves : List (ℕ × ℕ)), moves.length = 20 ∧ all_coins_in_one_sector (List.foldl (λ coins move => make_move coins move.1 move.2) initial_coins_in_sectors moves) :=
sorry

end impossible_gather_all_coins_in_one_sector_l154_154405


namespace tina_final_balance_l154_154304

noncomputable def monthlyIncome : ℝ := 1000
noncomputable def juneBonusRate : ℝ := 0.1
noncomputable def investmentReturnRate : ℝ := 0.05
noncomputable def taxRate : ℝ := 0.1

-- Savings rates
noncomputable def juneSavingsRate : ℝ := 0.25
noncomputable def julySavingsRate : ℝ := 0.20
noncomputable def augustSavingsRate : ℝ := 0.30

-- Expenses
noncomputable def juneRent : ℝ := 200
noncomputable def juneGroceries : ℝ := 100
noncomputable def juneBookRate : ℝ := 0.05

noncomputable def julyRent : ℝ := 250
noncomputable def julyGroceries : ℝ := 150
noncomputable def julyShoesRate : ℝ := 0.15

noncomputable def augustRent : ℝ := 300
noncomputable def augustGroceries : ℝ := 175
noncomputable def augustMiscellaneousRate : ℝ := 0.1

theorem tina_final_balance :
  let juneIncome := monthlyIncome * (1 + juneBonusRate)
  let juneSavings := juneIncome * juneSavingsRate
  let juneExpenses := juneRent + juneGroceries + juneIncome * juneBookRate
  let juneRemaining := juneIncome - juneSavings - juneExpenses

  let julyIncome := monthlyIncome
  let julyInvestmentReturn := juneSavings * investmentReturnRate
  let julyTotalIncome := julyIncome + julyInvestmentReturn
  let julySavings := julyTotalIncome * julySavingsRate
  let julyExpenses := julyRent + julyGroceries + julyIncome * julyShoesRate
  let julyRemaining := julyTotalIncome - julySavings - julyExpenses

  let augustIncome := monthlyIncome
  let augustInvestmentReturn := julySavings * investmentReturnRate
  let augustTotalIncome := augustIncome + augustInvestmentReturn
  let augustSavings := augustTotalIncome * augustSavingsRate
  let augustExpenses := augustRent + augustGroceries + augustIncome * augustMiscellaneousRate
  let augustRemaining := augustTotalIncome - augustSavings - augustExpenses

  let totalInvestmentReturn := julyInvestmentReturn + augustInvestmentReturn
  let totalTaxOnInvestment := totalInvestmentReturn * taxRate

  let finalBalance := juneRemaining + julyRemaining + augustRemaining - totalTaxOnInvestment

  finalBalance = 860.7075 := by
  sorry

end tina_final_balance_l154_154304


namespace minimum_value_of_f_l154_154028

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y, y = f x ∧ y = 1 :=
by
  sorry

end minimum_value_of_f_l154_154028


namespace sum_two_numbers_l154_154413

theorem sum_two_numbers :
  let X := (2 * 10) + 6
  let Y := (4 * 10) + 1
  X + Y = 67 :=
by
  sorry

end sum_two_numbers_l154_154413


namespace two_trains_distance_before_meeting_l154_154392

noncomputable def distance_one_hour_before_meeting (speed_A speed_B : ℕ) : ℕ :=
  speed_A + speed_B

theorem two_trains_distance_before_meeting (speed_A speed_B total_distance : ℕ) (h_speed_A : speed_A = 60) (h_speed_B : speed_B = 40) (h_total_distance : total_distance ≤ 250) :
  distance_one_hour_before_meeting speed_A speed_B = 100 :=
by
  sorry

end two_trains_distance_before_meeting_l154_154392


namespace right_triangle_least_side_l154_154979

theorem right_triangle_least_side (a b : ℕ) (h₁ : a = 8) (h₂ : b = 15) :
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2) ∧ c = Real.sqrt 161 := 
sorry

end right_triangle_least_side_l154_154979


namespace second_term_geometric_series_l154_154470

theorem second_term_geometric_series (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 :=
by
  -- Sorry is used to finalize the theorem without providing a proof here
  sorry

end second_term_geometric_series_l154_154470


namespace remainder_is_15x_minus_14_l154_154153

noncomputable def remainder_polynomial_division : Polynomial ℝ :=
  (Polynomial.X ^ 4) % (Polynomial.X ^ 2 - 3 * Polynomial.X + 2)

theorem remainder_is_15x_minus_14 :
  remainder_polynomial_division = 15 * Polynomial.X - 14 :=
by
  sorry

end remainder_is_15x_minus_14_l154_154153


namespace correct_tile_for_b_l154_154604

structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def TileI : Tile := {top := 5, right := 3, bottom := 1, left := 6}
def TileII : Tile := {top := 2, right := 6, bottom := 3, left := 5}
def TileIII : Tile := {top := 6, right := 1, bottom := 4, left := 2}
def TileIV : Tile := {top := 4, right := 5, bottom := 2, left := 1}

def RectangleBTile := TileIII

theorem correct_tile_for_b : RectangleBTile = TileIII :=
  sorry

end correct_tile_for_b_l154_154604


namespace x_coordinate_of_second_point_l154_154549

variable (m n : ℝ)

theorem x_coordinate_of_second_point
  (h1 : m = 2 * n + 5)
  (h2 : (m + 5) = 2 * (n + 2.5) + 5) :
  (m + 5) = m + 5 :=
by
  sorry

end x_coordinate_of_second_point_l154_154549


namespace sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l154_154319

theorem sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5 :
  let smallest := 125
  let largest := 521
  smallest + largest = 646 := by
  sorry

end sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l154_154319


namespace solve_m_l154_154881

theorem solve_m (m : ℝ) : (m + 1) / 6 = m / 1 → m = 1 / 5 :=
by
  intro h
  sorry

end solve_m_l154_154881


namespace find_phi_l154_154252

theorem find_phi (ϕ : ℝ) (h0 : 0 ≤ ϕ) (h1 : ϕ < π)
    (H : 2 * Real.cos (π / 3) = 2 * Real.sin (2 * (π / 3) + ϕ)) : ϕ = π / 6 :=
by
  sorry

end find_phi_l154_154252


namespace abs_a_gt_abs_c_sub_abs_b_l154_154747

theorem abs_a_gt_abs_c_sub_abs_b (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| :=
sorry

end abs_a_gt_abs_c_sub_abs_b_l154_154747


namespace sin_alpha_is_neg_5_over_13_l154_154975

-- Definition of the problem conditions
variables (α : Real) (h1 : 0 < α) (h2 : α < 2 * Real.pi)
variable (quad4 : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)
variable (h3 : Real.tan α = -5 / 12)

-- Proof statement
theorem sin_alpha_is_neg_5_over_13:
  Real.sin α = -5 / 13 :=
sorry

end sin_alpha_is_neg_5_over_13_l154_154975


namespace range_of_a_l154_154085

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- State the main theorem
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime a x1 = 0 ∧ f_prime a x2 = 0) →
  a < 0 :=
by
  sorry

end range_of_a_l154_154085


namespace incorrect_number_read_as_l154_154893

theorem incorrect_number_read_as (n a_incorrect a_correct correct_number incorrect_number : ℕ) 
(hn : n = 10) (h_inc_avg : a_incorrect = 18) (h_cor_avg : a_correct = 22) (h_cor_num : correct_number = 66) :
incorrect_number = 26 := by
  sorry

end incorrect_number_read_as_l154_154893


namespace degree_g_of_degree_f_and_h_l154_154532

noncomputable def degree (p : ℕ) := p -- definition to represent degree of polynomials

theorem degree_g_of_degree_f_and_h (f g : ℕ → ℕ) (h : ℕ → ℕ) 
  (deg_h : ℕ) (deg_f : ℕ) (deg_10 : deg_h = 10) (deg_3 : deg_f = 3) 
  (h_eq : ∀ x, degree (h x) = degree (f (g x)) + degree x ^ 5) :
  degree (g 0) = 4 :=
by
  sorry

end degree_g_of_degree_f_and_h_l154_154532


namespace sum_of_possible_coefficient_values_l154_154391

theorem sum_of_possible_coefficient_values :
  let pairs := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]
  let values := pairs.map (fun (r, s) => r + s)
  values.sum = 124 :=
by
  sorry

end sum_of_possible_coefficient_values_l154_154391


namespace infinite_sum_equals_one_fourth_l154_154295

theorem infinite_sum_equals_one_fourth :
  ∑' n : ℕ, (3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
sorry

end infinite_sum_equals_one_fourth_l154_154295


namespace find_a2_b2_geom_sequences_unique_c_l154_154705

-- Define the sequences as per the problem statement
def seqs (a b : ℕ → ℝ) :=
  a 1 = 0 ∧ b 1 = 2013 ∧
  ∀ n : ℕ, (1 ≤ n → (2 * a (n+1) = a n + b n)) ∧ (1 ≤ n → (4 * b (n+1) = a n + 3 * b n))

-- (1) Find values of a_2 and b_2
theorem find_a2_b2 {a b : ℕ → ℝ} (h : seqs a b) :
  a 2 = 1006.5 ∧ b 2 = 1509.75 :=
sorry

-- (2) Prove that {a_n - b_n} and {a_n + 2b_n} are geometric sequences
theorem geom_sequences {a b : ℕ → ℝ} (h : seqs a b) :
  ∃ r s : ℝ, (∃ c : ℝ, ∀ n : ℕ, a n - b n = c * r^n) ∧
             (∃ d : ℝ, ∀ n : ℕ, a n + 2 * b n = d * s^n) :=
sorry

-- (3) Prove there is a unique positive integer c such that a_n < c < b_n always holds
theorem unique_c {a b : ℕ → ℝ} (h : seqs a b) :
  ∃! c : ℝ, (0 < c) ∧ (∀ n : ℕ, 1 ≤ n → a n < c ∧ c < b n) :=
sorry

end find_a2_b2_geom_sequences_unique_c_l154_154705


namespace total_pink_crayons_l154_154187

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end total_pink_crayons_l154_154187


namespace cos_540_eq_neg_one_l154_154880

theorem cos_540_eq_neg_one : Real.cos (540 : ℝ) = -1 := by
  sorry

end cos_540_eq_neg_one_l154_154880


namespace total_solutions_l154_154889

-- Definitions and conditions
def tetrahedron_solutions := 1
def cube_solutions := 1
def octahedron_solutions := 3
def dodecahedron_solutions := 2
def icosahedron_solutions := 3

-- Main theorem statement
theorem total_solutions : 
  tetrahedron_solutions + cube_solutions + octahedron_solutions + dodecahedron_solutions + icosahedron_solutions = 10 := by
  sorry

end total_solutions_l154_154889


namespace laura_total_owed_l154_154013

-- Define the principal amounts charged each month
def january_charge : ℝ := 35
def february_charge : ℝ := 45
def march_charge : ℝ := 55
def april_charge : ℝ := 25

-- Define the respective interest rates for each month, as decimals
def january_interest_rate : ℝ := 0.05
def february_interest_rate : ℝ := 0.07
def march_interest_rate : ℝ := 0.04
def april_interest_rate : ℝ := 0.06

-- Define the interests accrued for each month's charges
def january_interest : ℝ := january_charge * january_interest_rate
def february_interest : ℝ := february_charge * february_interest_rate
def march_interest : ℝ := march_charge * march_interest_rate
def april_interest : ℝ := april_charge * april_interest_rate

-- Define the totals including original charges and their respective interests
def january_total : ℝ := january_charge + january_interest
def february_total : ℝ := february_charge + february_interest
def march_total : ℝ := march_charge + march_interest
def april_total : ℝ := april_charge + april_interest

-- Define the total amount owed a year later
def total_owed : ℝ := january_total + february_total + march_total + april_total

-- Prove that the total amount owed a year later is $168.60
theorem laura_total_owed :
  total_owed = 168.60 := by
  sorry

end laura_total_owed_l154_154013


namespace markup_percentage_l154_154919

-- Define the wholesale cost
def wholesale_cost : ℝ := sorry

-- Define the retail cost
def retail_cost : ℝ := sorry

-- Condition given in the problem: selling at 60% discount nets a 20% profit
def discount_condition (W R : ℝ) : Prop :=
  0.40 * R = 1.20 * W

-- We need to prove the markup percentage is 200%
theorem markup_percentage (W R : ℝ) (h : discount_condition W R) : 
  ((R - W) / W) * 100 = 200 :=
by sorry

end markup_percentage_l154_154919


namespace distinct_nonzero_real_product_l154_154201

noncomputable section
open Real

theorem distinct_nonzero_real_product
  (a b c d : ℝ)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hcd : c ≠ d)
  (hda : d ≠ a)
  (ha_ne_0 : a ≠ 0)
  (hb_ne_0 : b ≠ 0)
  (hc_ne_0 : c ≠ 0)
  (hd_ne_0 : d ≠ 0)
  (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 :=
sorry

end distinct_nonzero_real_product_l154_154201


namespace annual_rate_of_decrease_l154_154454

variable (r : ℝ) (initial_population population_after_2_years : ℝ)

-- Conditions
def initial_population_eq : initial_population = 30000 := sorry
def population_after_2_years_eq : population_after_2_years = 19200 := sorry
def population_formula : population_after_2_years = initial_population * (1 - r)^2 := sorry

-- Goal: Prove that the annual rate of decrease r is 0.2
theorem annual_rate_of_decrease :
  r = 0.2 := sorry

end annual_rate_of_decrease_l154_154454


namespace iris_to_tulip_ratio_l154_154876

theorem iris_to_tulip_ratio (earnings_per_bulb : ℚ)
  (tulip_bulbs daffodil_bulbs crocus_ratio total_earnings : ℕ)
  (iris_bulbs : ℕ) (h0 : earnings_per_bulb = 0.50)
  (h1 : tulip_bulbs = 20) (h2 : daffodil_bulbs = 30)
  (h3 : crocus_ratio = 3) (h4 : total_earnings = 75)
  (h5 : total_earnings = earnings_per_bulb * (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_ratio * daffodil_bulbs))
  : iris_bulbs = 10 → tulip_bulbs = 20 → (iris_bulbs : ℚ) / (tulip_bulbs : ℚ) = 1 / 2 :=
by {
  intros; sorry
}

end iris_to_tulip_ratio_l154_154876


namespace number_of_ways_to_fill_l154_154821

-- Definitions and conditions
def triangular_array (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the triangular array structure
  sorry 

def sum_based (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the sum-based condition
  sorry 

def valid_filling (x : Fin 13 → ℕ) :=
  (∀ i, x i = 0 ∨ x i = 1) ∧
  (x 0 + x 12) % 5 = 0

theorem number_of_ways_to_fill (x : Fin 13 → ℕ) :
  triangular_array 13 1 → sum_based 13 1 →
  valid_filling x → 
  (∃ (count : ℕ), count = 4096) :=
sorry

end number_of_ways_to_fill_l154_154821


namespace susan_min_packages_l154_154469

theorem susan_min_packages (n : ℕ) (cost_per_package : ℕ := 5) (earnings_per_package : ℕ := 15) (initial_cost : ℕ := 1200) :
  15 * n - 5 * n ≥ 1200 → n ≥ 120 :=
by {
  sorry -- Proof goes here
}

end susan_min_packages_l154_154469


namespace problem_l154_154710

theorem problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 7 - x = k^2) : x = 3 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end problem_l154_154710


namespace average_score_for_girls_l154_154562

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end average_score_for_girls_l154_154562


namespace rabbit_stashed_nuts_l154_154926

theorem rabbit_stashed_nuts :
  ∃ r: ℕ, 
  ∃ f: ℕ, 
  4 * r = 6 * f ∧ f = r - 5 ∧ 4 * r = 60 :=
by {
  sorry
}

end rabbit_stashed_nuts_l154_154926


namespace arrangements_15_cents_l154_154348

def numArrangements (n : ℕ) : ℕ :=
  sorry  -- Function definition which outputs the number of arrangements for sum n

theorem arrangements_15_cents : numArrangements 15 = X :=
  sorry  -- Replace X with the correct calculated number

end arrangements_15_cents_l154_154348


namespace juniors_score_l154_154342

theorem juniors_score (n : ℕ) (j s : ℕ) (avg_score students_avg seniors_avg : ℕ)
  (h1 : 0 < n)
  (h2 : j = n / 5)
  (h3 : s = 4 * n / 5)
  (h4 : avg_score = 80)
  (h5 : seniors_avg = 78)
  (h6 : students_avg = avg_score)
  (h7 : n * students_avg = n * avg_score)
  (h8 : s * seniors_avg = 78 * s) :
  (800 - 624) / j = 88 := by
  sorry

end juniors_score_l154_154342


namespace arc_length_of_sector_l154_154778

theorem arc_length_of_sector 
  (R : ℝ) (θ : ℝ) (hR : R = Real.pi) (hθ : θ = 2 * Real.pi / 3) : 
  (R * θ = 2 * Real.pi^2 / 3) := 
by
  rw [hR, hθ]
  sorry

end arc_length_of_sector_l154_154778


namespace number_of_spotted_blue_fish_l154_154898

def total_fish := 60
def blue_fish := total_fish / 3
def spotted_blue_fish := blue_fish / 2

theorem number_of_spotted_blue_fish : spotted_blue_fish = 10 :=
by
  -- Proof is omitted
  sorry

end number_of_spotted_blue_fish_l154_154898


namespace right_triangle_properties_l154_154665

theorem right_triangle_properties (a b c : ℝ) (h1 : c = 13) (h2 : a = 5)
  (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 ∧ a + b + c = 30 := by
  sorry

end right_triangle_properties_l154_154665


namespace solve_equation1_solve_equation2_l154_154567

-- Lean 4 statements for the given problems:
theorem solve_equation1 (x : ℝ) (h : x ≠ 0) : (2 / x = 3 / (x + 2)) ↔ (x = 4) := by
  sorry

theorem solve_equation2 (x : ℝ) (h : x ≠ 2) : ¬(5 / (x - 2) + 1 = (x - 7) / (2 - x)) := by
  sorry

end solve_equation1_solve_equation2_l154_154567


namespace perimeter_of_square_l154_154001

theorem perimeter_of_square (area : ℝ) (h : area = 392) : 
  ∃ (s : ℝ), 4 * s = 56 * Real.sqrt 2 :=
by 
  use (Real.sqrt 392)
  sorry

end perimeter_of_square_l154_154001


namespace water_distribution_scheme_l154_154708

theorem water_distribution_scheme (a b c : ℚ) : 
  a + b + c = 1 ∧ 
  (∀ x : ℂ, ∃ n : ℕ, x^n = 1 → x = 1) ∧
  (∀ (x : ℂ), (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12 + x^13 + x^14 + x^15 + x^16 + x^17 + x^18 + x^19 + x^20 + x^21 + x^22 = 0) → false) → 
  a = 0 ∧ b = 0 ∧ c = 1 :=
by
  sorry

end water_distribution_scheme_l154_154708


namespace millet_more_than_half_l154_154908

def daily_millet (n : ℕ) : ℝ :=
  1 - (0.7)^n

theorem millet_more_than_half (n : ℕ) : daily_millet 2 > 0.5 :=
by {
  sorry
}

end millet_more_than_half_l154_154908


namespace pool_filling_time_l154_154941

theorem pool_filling_time :
  (∀ t : ℕ, t >= 6 → ∃ v : ℝ, v = (2^(t-6)) * 0.25) →
  ∃ t : ℕ, t = 8 :=
by
  intros h
  existsi 8
  sorry

end pool_filling_time_l154_154941


namespace problem_1_problem_2_l154_154344

open Real

-- Part 1
theorem problem_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) :=
sorry

-- Part 2
theorem problem_2 : 
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 4 ∧ (4 / (a * b) + a / b = (1 + sqrt 5) / 2) :=
sorry

end problem_1_problem_2_l154_154344


namespace radius_of_tangent_circle_l154_154896

-- Define the conditions
def is_45_45_90_triangle (A B C : ℝ × ℝ) (AB BC AC : ℝ) : Prop :=
  (AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2) ∧
  (A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2))

def is_tangent_to_axes (O : ℝ × ℝ) (r : ℝ) : Prop :=
  O = (r, r)

def is_tangent_to_hypotenuse (O : ℝ × ℝ) (r : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - O.1) = Real.sqrt 2 * r ∧ (C.2 - O.2) = Real.sqrt 2 * r

-- Main theorem
theorem radius_of_tangent_circle :
  ∃ r : ℝ, ∀ (A B C O : ℝ × ℝ),
    is_45_45_90_triangle A B C (2) (2) (2 * Real.sqrt 2) →
    is_tangent_to_axes O r →
    is_tangent_to_hypotenuse O r C →
    r = Real.sqrt 2 :=
by
  sorry

end radius_of_tangent_circle_l154_154896


namespace min_a_plus_5b_l154_154381

theorem min_a_plus_5b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a * b + b^2 = b + 1) : 
  a + 5 * b ≥ 7 / 2 :=
by
  sorry

end min_a_plus_5b_l154_154381


namespace find_marks_in_english_l154_154305

theorem find_marks_in_english 
    (avg : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (chemistry_marks : ℕ) (biology_marks : ℕ) (total_subjects : ℕ)
    (avg_eq : avg = 78) 
    (math_eq : math_marks = 65) 
    (physics_eq : physics_marks = 82) 
    (chemistry_eq : chemistry_marks = 67) 
    (biology_eq : biology_marks = 85) 
    (subjects_eq : total_subjects = 5) : 
    math_marks + physics_marks + chemistry_marks + biology_marks + E = 78 * 5 → 
    E = 91 :=
by sorry

end find_marks_in_english_l154_154305


namespace age_of_youngest_child_l154_154956

/-- Given that the sum of ages of 5 children born at 3-year intervals is 70, prove the age of the youngest child is 8. -/
theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 := 
  sorry

end age_of_youngest_child_l154_154956


namespace remaining_macaroons_weight_is_103_l154_154763

-- Definitions based on the conditions
def coconutMacaroonsInitialCount := 12
def coconutMacaroonWeight := 5
def coconutMacaroonsBags := 4

def almondMacaroonsInitialCount := 8
def almondMacaroonWeight := 8
def almondMacaroonsBags := 2

def whiteChocolateMacaroonsInitialCount := 2
def whiteChocolateMacaroonWeight := 10

def steveAteCoconutMacaroons := coconutMacaroonsInitialCount / coconutMacaroonsBags
def steveAteAlmondMacaroons := (almondMacaroonsInitialCount / almondMacaroonsBags) / 2
def steveAteWhiteChocolateMacaroons := 1

-- Calculation of remaining macaroons weights
def remainingCoconutMacaroonsCount := coconutMacaroonsInitialCount - steveAteCoconutMacaroons
def remainingAlmondMacaroonsCount := almondMacaroonsInitialCount - steveAteAlmondMacaroons
def remainingWhiteChocolateMacaroonsCount := whiteChocolateMacaroonsInitialCount - steveAteWhiteChocolateMacaroons

-- Calculation of total remaining weight
def remainingCoconutMacaroonsWeight := remainingCoconutMacaroonsCount * coconutMacaroonWeight
def remainingAlmondMacaroonsWeight := remainingAlmondMacaroonsCount * almondMacaroonWeight
def remainingWhiteChocolateMacaroonsWeight := remainingWhiteChocolateMacaroonsCount * whiteChocolateMacaroonWeight

def totalRemainingWeight := remainingCoconutMacaroonsWeight + remainingAlmondMacaroonsWeight + remainingWhiteChocolateMacaroonsWeight

-- Statement to be proved
theorem remaining_macaroons_weight_is_103 :
  totalRemainingWeight = 103 := by
  sorry

end remaining_macaroons_weight_is_103_l154_154763


namespace tank_capacity_is_24_l154_154531

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end tank_capacity_is_24_l154_154531


namespace alayas_fruit_salads_l154_154796

theorem alayas_fruit_salads (A : ℕ) (H1 : 2 * A + A = 600) : A = 200 := 
by
  sorry

end alayas_fruit_salads_l154_154796


namespace problem_equiv_l154_154130

-- Definitions to match the conditions
def is_monomial (v : List ℤ) : Prop :=
  ∀ i ∈ v, True  -- Simplified; typically this would involve more specific definitions

def degree (e : String) : ℕ :=
  if e = "xy" then 2 else 0

noncomputable def coefficient (v : String) : ℤ :=
  if v = "m" then 1 else 0

-- Main fact to be proven
theorem problem_equiv :
  is_monomial [-3, 1, 5] :=
sorry

end problem_equiv_l154_154130


namespace sum_of_integers_greater_than_2_and_less_than_15_l154_154014

-- Define the set of integers greater than 2 and less than 15
def integersInRange : List ℕ := [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define the sum of these integers
def sumIntegersInRange : ℕ := integersInRange.sum

-- The main theorem to prove the sum
theorem sum_of_integers_greater_than_2_and_less_than_15 : sumIntegersInRange = 102 := by
  -- The proof part is omitted as per instructions
  sorry

end sum_of_integers_greater_than_2_and_less_than_15_l154_154014


namespace students_answered_both_correctly_l154_154067

theorem students_answered_both_correctly 
(total_students : ℕ) 
(did_not_answer_A_correctly : ℕ) 
(answered_A_correctly_but_not_B : ℕ) 
(h1 : total_students = 50) 
(h2 : did_not_answer_A_correctly = 12) 
(h3 : answered_A_correctly_but_not_B = 30) : 
    (total_students - did_not_answer_A_correctly - answered_A_correctly_but_not_B) = 8 :=
by
    sorry

end students_answered_both_correctly_l154_154067


namespace find_a_from_roots_l154_154859

theorem find_a_from_roots (a : ℝ) :
  let A := {x | (x = a) ∨ (x = a - 1)}
  2 ∈ A → a = 2 ∨ a = 3 :=
by
  intros A h
  sorry

end find_a_from_roots_l154_154859


namespace distinct_nonzero_reals_satisfy_equation_l154_154156

open Real

theorem distinct_nonzero_reals_satisfy_equation
  (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a) (h₄ : a ≠ 0) (h₅ : b ≠ 0) (h₆ : c ≠ 0)
  (h₇ : a + 2 / b = b + 2 / c) (h₈ : b + 2 / c = c + 2 / a) :
  (a + 2 / b) ^ 2 + (b + 2 / c) ^ 2 + (c + 2 / a) ^ 2 = 6 :=
sorry

end distinct_nonzero_reals_satisfy_equation_l154_154156


namespace point_not_in_fourth_quadrant_l154_154117

theorem point_not_in_fourth_quadrant (m : ℝ) : ¬(m-2 > 0 ∧ m+1 < 0) := 
by
  -- Since (m+1) - (m-2) = 3, which is positive,
  -- m+1 > m-2, thus the statement ¬(m-2 > 0 ∧ m+1 < 0) holds.
  sorry

end point_not_in_fourth_quadrant_l154_154117


namespace point_relationship_on_parabola_neg_x_plus_1_sq_5_l154_154137

theorem point_relationship_on_parabola_neg_x_plus_1_sq_5
  (y_1 y_2 y_3 : ℝ) :
  (A : ℝ × ℝ) = (-2, y_1) →
  (B : ℝ × ℝ) = (1, y_2) →
  (C : ℝ × ℝ) = (2, y_3) →
  (A.2 = -(A.1 + 1)^2 + 5) →
  (B.2 = -(B.1 + 1)^2 + 5) →
  (C.2 = -(C.1 + 1)^2 + 5) →
  y_1 > y_2 ∧ y_2 > y_3 :=
by
  sorry

end point_relationship_on_parabola_neg_x_plus_1_sq_5_l154_154137


namespace polynomial_identity_l154_154755

variable (x y : ℝ)

theorem polynomial_identity :
    (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 :=
sorry

end polynomial_identity_l154_154755


namespace division_631938_by_625_l154_154536

theorem division_631938_by_625 :
  (631938 : ℚ) / 625 = 1011.1008 :=
by
  -- Add a placeholder proof. We do not provide the solution steps.
  sorry

end division_631938_by_625_l154_154536


namespace reciprocals_not_arithmetic_sequence_l154_154174

theorem reciprocals_not_arithmetic_sequence 
  (a b c : ℝ) (h : 2 * b = a + c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_neq : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ¬ (1 / a + 1 / c = 2 / b) :=
by
  sorry

end reciprocals_not_arithmetic_sequence_l154_154174


namespace fraction_zero_iff_x_one_l154_154996

theorem fraction_zero_iff_x_one (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x - 5 ≠ 0) : x = 1 := by
  sorry

end fraction_zero_iff_x_one_l154_154996


namespace yard_area_l154_154313

theorem yard_area (posts : Nat) (spacing : Real) (longer_factor : Nat) (shorter_side_posts longer_side_posts : Nat)
  (h1 : posts = 24)
  (h2 : spacing = 3)
  (h3 : longer_factor = 3)
  (h4 : 2 * (shorter_side_posts + longer_side_posts) = posts - 4)
  (h5 : longer_side_posts = 3 * shorter_side_posts + 2) :
  (spacing * (shorter_side_posts - 1)) * (spacing * (longer_side_posts - 1)) = 144 :=
by
  sorry

end yard_area_l154_154313


namespace farmer_revenue_correct_l154_154175

-- Define the conditions
def average_bacon : ℕ := 20
def price_per_pound : ℕ := 6
def size_factor : ℕ := 1 / 2

-- Calculate the bacon from the runt pig
def bacon_from_runt := average_bacon * size_factor

-- Calculate the revenue from selling the bacon
def revenue := bacon_from_runt * price_per_pound

-- Lean 4 Statement to prove
theorem farmer_revenue_correct :
  revenue = 60 :=
sorry

end farmer_revenue_correct_l154_154175


namespace product_of_divisor_and_dividend_l154_154030

theorem product_of_divisor_and_dividend (d D : ℕ) (q : ℕ := 6) (r : ℕ := 3) 
  (h₁ : D = d + 78) 
  (h₂ : D = d * q + r) : 
  D * d = 1395 :=
by 
  sorry

end product_of_divisor_and_dividend_l154_154030


namespace max_n_for_regular_polygons_l154_154799

theorem max_n_for_regular_polygons (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 3)
  (h3 : (7 * (m - 2) * n) = (8 * (n - 2) * m)) : 
  n ≤ 112 ∧ (∃ m, (14 * n = (n - 16) * m)) :=
by
  sorry

end max_n_for_regular_polygons_l154_154799


namespace largest_integer_with_square_three_digits_base_7_l154_154727

theorem largest_integer_with_square_three_digits_base_7 : 
  ∃ M : ℕ, (7^2 ≤ M^2 ∧ M^2 < 7^3) ∧ ∀ n : ℕ, (7^2 ≤ n^2 ∧ n^2 < 7^3) → n ≤ M := 
sorry

end largest_integer_with_square_three_digits_base_7_l154_154727


namespace find_m_l154_154945

theorem find_m 
  (h : ( (1 ^ m) / (5 ^ m) ) * ( (1 ^ 16) / (4 ^ 16) ) = 1 / (2 * 10 ^ 31)) :
  m = 31 :=
by
  sorry

end find_m_l154_154945


namespace sin_double_angle_identity_l154_154814

open Real 

theorem sin_double_angle_identity 
  (A : ℝ) 
  (h1 : 0 < A) 
  (h2 : A < π / 2) 
  (h3 : cos A = 3 / 5) : 
  sin (2 * A) = 24 / 25 :=
by 
  sorry

end sin_double_angle_identity_l154_154814


namespace can_place_more_domino_domino_placement_possible_l154_154653

theorem can_place_more_domino (total_squares : ℕ := 36) (uncovered_squares : ℕ := 14) : Prop :=
∃ (n : ℕ), (n * 2 + uncovered_squares ≤ total_squares) ∧ (n ≥ 1)

/-- Proof that on a 6x6 chessboard with some 1x2 dominoes placed, if there are 14 uncovered
squares, then at least one more domino can be placed on the board. -/
theorem domino_placement_possible :
  can_place_more_domino := by
  sorry

end can_place_more_domino_domino_placement_possible_l154_154653


namespace A_form_k_l154_154590

theorem A_form_k (m n : ℕ) (h_m : 2 ≤ m) (h_n : 2 ≤ n) :
  ∃ k : ℕ, (A : ℝ) = (n + Real.sqrt (n^2 - 4)) / 2 ^ m → A = (k + Real.sqrt (k^2 - 4)) / 2 :=
by
  sorry

end A_form_k_l154_154590


namespace min_value_abs_plus_2023_proof_l154_154809

noncomputable def min_value_abs_plus_2023 (a : ℚ) : Prop :=
  |a| + 2023 ≥ 2023

theorem min_value_abs_plus_2023_proof (a : ℚ) : min_value_abs_plus_2023 a :=
  by
  sorry

end min_value_abs_plus_2023_proof_l154_154809


namespace least_positive_integer_l154_154998

open Nat

theorem least_positive_integer (n : ℕ) (h1 : n ≡ 2 [MOD 5]) (h2 : n ≡ 2 [MOD 4]) (h3 : n ≡ 0 [MOD 3]) : n = 42 :=
sorry

end least_positive_integer_l154_154998


namespace major_axis_length_l154_154378

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by sorry

end major_axis_length_l154_154378


namespace sqrt_50_product_consecutive_integers_l154_154211

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l154_154211


namespace sales_tax_difference_l154_154856

theorem sales_tax_difference :
  let price_before_tax := 40
  let tax_rate_8_percent := 0.08
  let tax_rate_7_percent := 0.07
  let sales_tax_8_percent := price_before_tax * tax_rate_8_percent
  let sales_tax_7_percent := price_before_tax * tax_rate_7_percent
  sales_tax_8_percent - sales_tax_7_percent = 0.4 := 
by
  sorry

end sales_tax_difference_l154_154856


namespace compute_b_l154_154672

noncomputable def rational_coefficients (a b : ℚ) :=
∃ x : ℚ, (x^3 + a * x^2 + b * x + 15 = 0)

theorem compute_b (a b : ℚ) (h1 : (3 + Real.sqrt 5)∈{root : ℝ | root^3 + a * root^2 + b * root + 15 = 0}) 
(h2 : rational_coefficients a b) : b = -18.5 :=
by
  sorry

end compute_b_l154_154672


namespace investment_ratio_l154_154504

-- Define the investments
def A_investment (x : ℝ) : ℝ := 3 * x
def B_investment (x : ℝ) : ℝ := x
def C_investment (y : ℝ) : ℝ := y

-- Define the total profit and B's share of the profit
def total_profit : ℝ := 4400
def B_share : ℝ := 800

-- Define the ratio condition B's share based on investments
def B_share_cond (x y : ℝ) : Prop := (B_investment x / (A_investment x + B_investment x + C_investment y)) * total_profit = B_share

-- Define what we need to prove
theorem investment_ratio (x y : ℝ) (h : B_share_cond x y) : x / y = 2 / 3 :=
by 
  sorry

end investment_ratio_l154_154504


namespace rhombus_area_correct_l154_154206

def rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_correct
  (d1 d2 : ℕ)
  (h1 : d1 = 70)
  (h2 : d2 = 160) :
  rhombus_area d1 d2 = 5600 := 
by
  sorry

end rhombus_area_correct_l154_154206


namespace proof_problem_l154_154226

def is_solution (x : ℝ) : Prop :=
  4 * Real.cos x * Real.cos (2 * x) * Real.cos (3 * x) = Real.cos (6 * x)

noncomputable def solution (l n : ℤ) : ℝ :=
  max (Real.pi / 3 * (3 * l + 1)) (Real.pi / 4 * (2 * n + 1))

theorem proof_problem (x : ℝ) (l n : ℤ) : is_solution x → x = solution l n :=
sorry

end proof_problem_l154_154226


namespace ratio_of_work_capacity_l154_154192

theorem ratio_of_work_capacity (work_rate_A work_rate_B : ℝ)
  (hA : work_rate_A = 1 / 45)
  (hAB : work_rate_A + work_rate_B = 1 / 18) :
  work_rate_A⁻¹ / work_rate_B⁻¹ = 3 / 2 :=
by
  sorry

end ratio_of_work_capacity_l154_154192


namespace possible_third_side_of_triangle_l154_154425

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end possible_third_side_of_triangle_l154_154425


namespace karl_total_miles_l154_154655

def car_mileage_per_gallon : ℕ := 30
def full_tank_gallons : ℕ := 14
def initial_drive_miles : ℕ := 300
def gas_bought_gallons : ℕ := 10
def final_tank_fraction : ℚ := 1 / 3

theorem karl_total_miles (initial_fuel : ℕ) :
  initial_fuel = full_tank_gallons →
  (initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons) = initial_fuel - (initial_fuel * final_tank_fraction) / car_mileage_per_gallon + (580 - initial_drive_miles) / car_mileage_per_gallon →
  initial_drive_miles + (initial_fuel - initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons - initial_fuel * final_tank_fraction / car_mileage_per_gallon) * car_mileage_per_gallon = 580 := 
sorry

end karl_total_miles_l154_154655


namespace no_lonely_points_eventually_l154_154191

structure Graph (α : Type) :=
(vertices : Finset α)
(edges : α → Finset α)

namespace Graph

def is_lonely {α : Type} (G : Graph α) (coloring : α → Bool) (v : α) : Prop :=
  let neighbors := G.edges v
  let different_color_neighbors := neighbors.filter (λ w => coloring w ≠ coloring v)
  2 * different_color_neighbors.card > neighbors.card

end Graph

theorem no_lonely_points_eventually
  {α : Type}
  (G : Graph α)
  (initial_coloring : α → Bool) :
  ∃ (steps : Nat),
  ∀ (coloring : α → Bool),
  (∃ (t : Nat), t ≤ steps ∧ 
    (∀ v, ¬ Graph.is_lonely G coloring v)) :=
sorry

end no_lonely_points_eventually_l154_154191


namespace rationalize_denominator_sum_l154_154074

noncomputable def rationalize_denominator (x y z : ℤ) :=
  x = 4 ∧ y = 49 ∧ z = 35 ∧ y ∣ 343 ∧ z > 0 

theorem rationalize_denominator_sum : 
  ∃ A B C : ℤ, rationalize_denominator A B C ∧ A + B + C = 88 :=
by
  sorry

end rationalize_denominator_sum_l154_154074


namespace no_primes_sum_to_53_l154_154289

open Nat

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem no_primes_sum_to_53 :
  ¬ ∃ (p q : Nat), p + q = 53 ∧ isPrime p ∧ isPrime q ∧ (p < 30 ∨ q < 30) :=
by
  sorry

end no_primes_sum_to_53_l154_154289


namespace tan_alpha_eq_2_l154_154167

theorem tan_alpha_eq_2 (α : Real) (h : Real.tan α = 2) : 
  1 / (Real.sin (2 * α) + Real.cos (α) ^ 2) = 1 := 
by 
  sorry

end tan_alpha_eq_2_l154_154167


namespace episode_length_l154_154370

/-- Subject to the conditions provided, we prove the length of each episode watched by Maddie. -/
theorem episode_length
  (total_episodes : ℕ)
  (monday_minutes : ℕ)
  (thursday_minutes : ℕ)
  (weekend_minutes : ℕ)
  (episodes_length : ℕ)
  (monday_watch : monday_minutes = 138)
  (thursday_watch : thursday_minutes = 21)
  (weekend_watch : weekend_minutes = 105)
  (total_episodes_watch : total_episodes = 8)
  (total_minutes : monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episodes_length) :
  episodes_length = 33 := 
by 
  sorry

end episode_length_l154_154370


namespace grandpa_movie_time_l154_154660

theorem grandpa_movie_time
  (each_movie_time : ℕ := 90)
  (max_movies_2_days : ℕ := 9)
  (x_movies_tuesday : ℕ)
  (movies_wednesday := 2 * x_movies_tuesday)
  (total_movies := x_movies_tuesday + movies_wednesday)
  (h : total_movies = max_movies_2_days) :
  90 * x_movies_tuesday = 270 :=
by
  sorry

end grandpa_movie_time_l154_154660


namespace range_of_a_l154_154736

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a (a : ℝ) :
  (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a p - f a q) / (p - q) > 1)
  ↔ 3 ≤ a :=
by
  sorry

end range_of_a_l154_154736


namespace ab_c_sum_geq_expr_ab_c_sum_eq_iff_l154_154995

theorem ab_c_sum_geq_expr (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) :=
sorry

theorem ab_c_sum_eq_iff (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c :=
sorry

end ab_c_sum_geq_expr_ab_c_sum_eq_iff_l154_154995


namespace balloon_highest_elevation_l154_154906

theorem balloon_highest_elevation
  (time_rise1 time_rise2 time_descent : ℕ)
  (rate_rise rate_descent : ℕ)
  (t1 : time_rise1 = 15)
  (t2 : time_rise2 = 15)
  (t3 : time_descent = 10)
  (rr : rate_rise = 50)
  (rd : rate_descent = 10)
  : (time_rise1 * rate_rise - time_descent * rate_descent + time_rise2 * rate_rise) = 1400 := 
by
  sorry

end balloon_highest_elevation_l154_154906


namespace calculate_total_cost_l154_154099

theorem calculate_total_cost : 
  let piano_cost := 500
  let lesson_cost_per_lesson := 40
  let number_of_lessons := 20
  let discount_rate := 0.25
  let missed_lessons := 3
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_lesson_cost := number_of_lessons * lesson_cost_per_lesson
  let discount := total_lesson_cost * discount_rate
  let discounted_lesson_cost := total_lesson_cost - discount
  let cost_of_missed_lessons := missed_lessons * lesson_cost_per_lesson
  let effective_lesson_cost := discounted_lesson_cost + cost_of_missed_lessons
  let total_cost := piano_cost + effective_lesson_cost + sheet_music_cost + maintenance_fees
  total_cost = 1395 :=
by
  sorry

end calculate_total_cost_l154_154099


namespace always_positive_expression_l154_154466

variable (x a b : ℝ)

theorem always_positive_expression (h : ∀ x, (x - a)^2 + b > 0) : b > 0 :=
sorry

end always_positive_expression_l154_154466


namespace men_took_dip_l154_154652

theorem men_took_dip 
  (tank_length : ℝ) (tank_breadth : ℝ) (water_rise_cm : ℝ) (man_displacement : ℝ)
  (H1 : tank_length = 40) (H2 : tank_breadth = 20) (H3 : water_rise_cm = 25) (H4 : man_displacement = 4) :
  let water_rise_m := water_rise_cm / 100
  let total_volume_displaced := tank_length * tank_breadth * water_rise_m
  let number_of_men := total_volume_displaced / man_displacement
  number_of_men = 50 :=
by
  sorry

end men_took_dip_l154_154652


namespace number_of_chickens_l154_154383

-- Definitions based on conditions
def totalAnimals := 100
def legDifference := 26

-- The problem statement to be proved
theorem number_of_chickens (x : Nat) (r : Nat) (legs_chickens : Nat) (legs_rabbits : Nat) (total : Nat := totalAnimals) (diff : Nat := legDifference) :
  x + r = total ∧ 2 * x + 4 * r - 4 * r = 2 * x + diff → x = 71 :=
by
  intro h
  sorry

end number_of_chickens_l154_154383


namespace find_number_of_math_problems_l154_154764

-- Define the number of social studies problems
def social_studies_problems : ℕ := 6

-- Define the number of science problems
def science_problems : ℕ := 10

-- Define the time to solve each type of problem in minutes
def time_per_math_problem : ℝ := 2
def time_per_social_studies_problem : ℝ := 0.5
def time_per_science_problem : ℝ := 1.5

-- Define the total time to solve all problems in minutes
def total_time : ℝ := 48

-- Define the theorem to find the number of math problems
theorem find_number_of_math_problems (M : ℕ) :
  time_per_math_problem * M + time_per_social_studies_problem * social_studies_problems + time_per_science_problem * science_problems = total_time → 
  M = 15 :=
by {
  -- proof is not required to be written, hence expressing the unresolved part
  sorry
}

end find_number_of_math_problems_l154_154764


namespace eccentricity_of_ellipse_l154_154568

theorem eccentricity_of_ellipse : 
  ∀ (a b c e : ℝ), a^2 = 16 → b^2 = 8 → c^2 = a^2 - b^2 → e = c / a → e = (Real.sqrt 2) / 2 := 
by 
  intros a b c e ha hb hc he
  sorry

end eccentricity_of_ellipse_l154_154568


namespace removed_number_is_34_l154_154699
open Real

theorem removed_number_is_34 (n : ℕ) (x : ℕ) (h₁ : 946 = (43 * (43 + 1)) / 2) (h₂ : 912 = 43 * (152 / 7)) : x = 34 :=
by
  sorry

end removed_number_is_34_l154_154699


namespace property_damage_worth_40000_l154_154723

-- Definitions based on conditions in a)
def medical_bills : ℝ := 70000
def insurance_rate : ℝ := 0.80
def carl_payment : ℝ := 22000
def carl_rate : ℝ := 0.20

theorem property_damage_worth_40000 :
  ∃ P : ℝ, P = 40000 ∧ 
    (carl_payment = carl_rate * (P + medical_bills)) :=
by
  sorry

end property_damage_worth_40000_l154_154723


namespace part1_part2_l154_154586

-- Definitions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b + a + 3

-- First proof: When a = -1 and b = 10, prove 4A - (3A - 2B) = -45
theorem part1 : 4 * A (-1) 10 - (3 * A (-1) 10 - 2 * B (-1) 10) = -45 := by
  sorry

-- Second proof: If a and b are reciprocal, prove 4A - (3A - 2B) = 10
theorem part2 (a b : ℝ) (hab : a * b = 1) : 4 * A a b - (3 * A a b - 2 * B a b) = 10 := by
  sorry

end part1_part2_l154_154586


namespace compute_fraction_sum_l154_154107

theorem compute_fraction_sum
  (a b c : ℝ)
  (h : a^3 - 6 * a^2 + 11 * a = 12)
  (h : b^3 - 6 * b^2 + 11 * b = 12)
  (h : c^3 - 6 * c^2 + 11 * c = 12) :
  (ab : ℝ) / c + (bc : ℝ) / a + (ca : ℝ) / b = -23 / 12 := by
  sorry

end compute_fraction_sum_l154_154107


namespace train_passes_man_in_approximately_24_seconds_l154_154437

noncomputable def train_length : ℝ := 880 -- length of the train in meters
noncomputable def train_speed_kmph : ℝ := 120 -- speed of the train in km/h
noncomputable def man_speed_kmph : ℝ := 12 -- speed of the man in km/h

noncomputable def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph
noncomputable def relative_speed : ℝ := train_speed_mps + man_speed_mps

noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem train_passes_man_in_approximately_24_seconds :
  abs (time_to_pass - 24) < 1 :=
sorry

end train_passes_man_in_approximately_24_seconds_l154_154437


namespace extra_pieces_correct_l154_154639

def pieces_per_package : ℕ := 7
def number_of_packages : ℕ := 5
def total_pieces : ℕ := 41

theorem extra_pieces_correct : total_pieces - (number_of_packages * pieces_per_package) = 6 :=
by
  sorry

end extra_pieces_correct_l154_154639


namespace range_of_x_squared_f_x_lt_x_squared_minus_f_1_l154_154355

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def satisfies_inequality (f f' : ℝ → ℝ) : Prop :=
∀ x : ℝ, 2 * f x + x * f' x < 2

theorem range_of_x_squared_f_x_lt_x_squared_minus_f_1 (f f' : ℝ → ℝ)
  (h_even : even_function f)
  (h_ineq : satisfies_inequality f f')
  : {x : ℝ | x^2 * f x - f 1 < x^2 - 1} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
sorry

end range_of_x_squared_f_x_lt_x_squared_minus_f_1_l154_154355


namespace scientific_notation_of_8_5_million_l154_154069

theorem scientific_notation_of_8_5_million :
  (8.5 * 10^6) = 8500000 :=
by sorry

end scientific_notation_of_8_5_million_l154_154069


namespace gcd_n_four_plus_sixteen_and_n_plus_three_l154_154632

theorem gcd_n_four_plus_sixteen_and_n_plus_three (n : ℕ) (hn1 : n > 9) (hn2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 :=
by
  sorry

end gcd_n_four_plus_sixteen_and_n_plus_three_l154_154632


namespace sum_of_special_multiples_l154_154607

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end sum_of_special_multiples_l154_154607


namespace expression_value_l154_154521

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end expression_value_l154_154521


namespace negation_of_universal_proposition_l154_154765

open Classical

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x : ℕ, x^2 ≤ x) :=
by
  sorry

end negation_of_universal_proposition_l154_154765


namespace average_weight_of_students_l154_154048

theorem average_weight_of_students (b_avg_weight g_avg_weight : ℝ) (num_boys num_girls : ℕ)
  (hb : b_avg_weight = 155) (hg : g_avg_weight = 125) (hb_num : num_boys = 8) (hg_num : num_girls = 5) :
  (num_boys * b_avg_weight + num_girls * g_avg_weight) / (num_boys + num_girls) = 143 :=
by sorry

end average_weight_of_students_l154_154048


namespace locus_of_point_T_l154_154233

theorem locus_of_point_T (r : ℝ) (a b : ℝ) (x y x1 y1 x2 y2 : ℝ)
  (hM_inside : a^2 + b^2 < r^2)
  (hK_on_circle : x1^2 + y1^2 = r^2)
  (hP_on_circle : x2^2 + y2^2 = r^2)
  (h_midpoints_eq : (x + a) / 2 = (x1 + x2) / 2 ∧ (y + b) / 2 = (y1 + y2) / 2)
  (h_diagonal_eq : (x - a)^2 + (y - b)^2 = (x1 - x2)^2 + (y1 - y2)^2) :
  x^2 + y^2 = 2 * r^2 - (a^2 + b^2) :=
  sorry

end locus_of_point_T_l154_154233


namespace asha_remaining_money_l154_154810

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l154_154810


namespace find_constant_l154_154592

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant : ∃ c : ℝ, (∀ x : ℝ, x = 0.4 → (3 * f (x - c)) / f 0 + 4 = f (2 * x + 1)) ∧ c = 2 :=
by
  sorry

end find_constant_l154_154592


namespace adoption_time_l154_154534

theorem adoption_time
  (p0 : ℕ) (p1 : ℕ) (rate : ℕ)
  (p0_eq : p0 = 10) (p1_eq : p1 = 15) (rate_eq : rate = 7) :
  Nat.ceil ((p0 + p1) / rate) = 4 := by
  sorry

end adoption_time_l154_154534


namespace factorization_of_polynomial_l154_154450

theorem factorization_of_polynomial (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := 
sorry

end factorization_of_polynomial_l154_154450


namespace movie_of_the_year_condition_l154_154932

noncomputable def smallest_needed_lists : Nat :=
  let total_lists := 765
  let required_fraction := 1 / 4
  Nat.ceil (total_lists * required_fraction)

theorem movie_of_the_year_condition :
  smallest_needed_lists = 192 := by
  sorry

end movie_of_the_year_condition_l154_154932


namespace remainder_when_a6_divided_by_n_l154_154411

theorem remainder_when_a6_divided_by_n (n : ℕ) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := 
sorry

end remainder_when_a6_divided_by_n_l154_154411


namespace fred_has_9_dimes_l154_154239

-- Fred has 90 cents in his bank.
def freds_cents : ℕ := 90

-- A dime is worth 10 cents.
def value_of_dime : ℕ := 10

-- Prove that the number of dimes Fred has is 9.
theorem fred_has_9_dimes : (freds_cents / value_of_dime) = 9 := by
  sorry

end fred_has_9_dimes_l154_154239


namespace grazing_months_of_A_l154_154988

-- Definitions of conditions
def oxen_months_A (x : ℕ) := 10 * x
def oxen_months_B := 12 * 5
def oxen_months_C := 15 * 3
def total_rent := 140
def rent_C := 36

-- Assuming a is the number of months a put his oxen for grazing, we need to prove that a = 7
theorem grazing_months_of_A (a : ℕ) :
  (45 * 140 = 36 * (10 * a + 60 + 45)) → a = 7 := 
by
  intro h
  sorry

end grazing_months_of_A_l154_154988


namespace find_a_value_l154_154238

theorem find_a_value 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x : ℝ, f x = x^3 + a*x^2 + 3*x - 9)
  (extreme_at_minus_3 : ∀ f' : ℝ → ℝ, (∀ x, f' x = 3*x^2 + 2*a*x + 3) → f' (-3) = 0) :
  a = 5 := 
sorry

end find_a_value_l154_154238


namespace impossible_circular_arrangement_1_to_60_l154_154243

theorem impossible_circular_arrangement_1_to_60 :
  (∀ (f : ℕ → ℕ), 
      (∀ n, 1 ≤ f n ∧ f n ≤ 60) ∧ 
      (∀ n, f (n + 2) + f n ≡ 0 [MOD 2]) ∧ 
      (∀ n, f (n + 3) + f n ≡ 0 [MOD 3]) ∧ 
      (∀ n, f (n + 7) + f n ≡ 0 [MOD 7]) 
      → false) := 
  sorry

end impossible_circular_arrangement_1_to_60_l154_154243


namespace find_initial_speed_l154_154716

-- Definitions for the conditions
def total_distance : ℕ := 800
def time_at_initial_speed : ℕ := 6
def time_at_60_mph : ℕ := 4
def time_at_40_mph : ℕ := 2
def speed_at_60_mph : ℕ := 60
def speed_at_40_mph : ℕ := 40

-- Setting up the equation: total distance covered
def distance_covered (v : ℕ) : ℕ :=
  time_at_initial_speed * v + time_at_60_mph * speed_at_60_mph + time_at_40_mph * speed_at_40_mph

-- Proof problem statement
theorem find_initial_speed : ∃ v : ℕ, distance_covered v = total_distance ∧ v = 80 := by
  existsi 80
  simp [distance_covered, total_distance, time_at_initial_speed, speed_at_60_mph, time_at_40_mph]
  norm_num
  sorry

end find_initial_speed_l154_154716


namespace removed_cubes_total_l154_154606

-- Define the large cube composed of 125 smaller cubes (5x5x5 cube)
def large_cube := 5 * 5 * 5

-- Number of smaller cubes removed from each face to opposite face
def removed_faces := (5 * 5 + 5 * 5 + 5 * 3)

-- Overlapping cubes deducted
def overlapping_cubes := (3 + 1)

-- Final number of removed smaller cubes
def removed_total := removed_faces - overlapping_cubes

-- Lean theorem statement
theorem removed_cubes_total : removed_total = 49 :=
by
  -- Definitions provided above imply the theorem
  sorry

end removed_cubes_total_l154_154606


namespace non_congruent_right_triangles_unique_l154_154839

theorem non_congruent_right_triangles_unique :
  ∃! (a: ℝ) (b: ℝ) (c: ℝ), a > 0 ∧ b = 2 * a ∧ c = a * Real.sqrt 5 ∧
  (3 * a + a * Real.sqrt 5 - a^2 = a * Real.sqrt 5) :=
by
  sorry

end non_congruent_right_triangles_unique_l154_154839


namespace abs_neg_2023_l154_154985

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l154_154985


namespace mn_value_l154_154486

theorem mn_value (m n : ℤ) (h1 : m + n = 1) (h2 : m - n + 2 = 1) : m * n = 0 := 
by 
  sorry

end mn_value_l154_154486


namespace max_constant_inequality_l154_154184

theorem max_constant_inequality (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha1 : a ≤ 1)
    (hb : 0 ≤ b) (hb1 : b ≤ 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (hd : 0 ≤ d) (hd1 : d ≤ 1) 
    : a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3) :=
sorry

end max_constant_inequality_l154_154184


namespace total_wall_area_l154_154905

variable (L W : ℝ) -- Length and width of the regular tile
variable (R : ℕ) -- Number of regular tiles

-- Conditions:
-- 1. The area covered by regular tiles is 70 square feet.
axiom regular_tiles_cover_area : R * (L * W) = 70

-- 2. Jumbo tiles make up 1/3 of the total tiles, and each jumbo tile has an area three times that of a regular tile.
axiom length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105

theorem total_wall_area (L W : ℝ) (R : ℕ) 
  (regular_tiles_cover_area : R * (L * W) = 70) 
  (length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105) : 
  (R * (L * W)) + (3 * (R / 2) * (L * W)) = 175 :=
by
  sorry

end total_wall_area_l154_154905


namespace rounding_no_order_l154_154685

theorem rounding_no_order (x : ℝ) (hx : x > 0) :
  let a := round (x * 100) / 100
  let b := round (x * 1000) / 1000
  let c := round (x * 10000) / 10000
  (¬((a ≥ b ∧ b ≥ c) ∨ (a ≤ b ∧ b ≤ c))) :=
sorry

end rounding_no_order_l154_154685


namespace infinite_primes_of_form_4n_plus_3_l154_154180

theorem infinite_primes_of_form_4n_plus_3 :
  ∀ (S : Finset ℕ), (∀ p ∈ S, Prime p ∧ p % 4 = 3) →
  ∃ q, Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
by 
  sorry

end infinite_primes_of_form_4n_plus_3_l154_154180


namespace music_track_duration_l154_154121

theorem music_track_duration (minutes : ℝ) (seconds_per_minute : ℝ) (duration_in_minutes : minutes = 12.5) (seconds_per_minute_is_60 : seconds_per_minute = 60) : minutes * seconds_per_minute = 750 := by
  sorry

end music_track_duration_l154_154121


namespace houses_with_both_l154_154388

theorem houses_with_both (G P N Total B : ℕ) 
  (hG : G = 50) 
  (hP : P = 40) 
  (hN : N = 10) 
  (hTotal : Total = 65)
  (hEquation : G + P - B = Total - N) 
  : B = 35 := 
by 
  sorry

end houses_with_both_l154_154388


namespace clothing_loss_l154_154292

theorem clothing_loss
  (a : ℝ)
  (h1 : ∃ x y : ℝ, x * 1.25 = a ∧ y * 0.75 = a ∧ x + y - 2 * a = -8) :
  a = 60 :=
sorry

end clothing_loss_l154_154292


namespace problem_l154_154141

theorem problem (n : ℝ) (h : n + 1 / n = 10) : n ^ 2 + 1 / n ^ 2 + 5 = 103 :=
by sorry

end problem_l154_154141


namespace f_no_zeros_in_interval_f_zeros_in_interval_l154_154618

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

theorem f_no_zeros_in_interval (x : ℝ) (hx1 : x > 1 / Real.exp 1) (hx2 : x < 1) :
  f x ≠ 0 := sorry

theorem f_zeros_in_interval (h1 : 1 < e) (x_exists : ∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :
  true := sorry

end f_no_zeros_in_interval_f_zeros_in_interval_l154_154618


namespace angle_measure_l154_154114

-- Define the problem conditions
def angle (x : ℝ) : Prop :=
  let complement := 3 * x + 6
  x + complement = 90

-- The theorem to prove
theorem angle_measure : ∃ x : ℝ, angle x ∧ x = 21 := 
sorry

end angle_measure_l154_154114


namespace initial_birds_on_fence_l154_154260

theorem initial_birds_on_fence (B S : ℕ) (S_val : S = 2) (total : B + 5 + S = 10) : B = 3 :=
by
  sorry

end initial_birds_on_fence_l154_154260


namespace least_multiple_of_24_gt_500_l154_154640

theorem least_multiple_of_24_gt_500 : ∃ x : ℕ, (x % 24 = 0) ∧ (x > 500) ∧ (∀ y : ℕ, (y % 24 = 0) ∧ (y > 500) → y ≥ x) ∧ (x = 504) := by
  sorry

end least_multiple_of_24_gt_500_l154_154640


namespace average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l154_154622

open Real

noncomputable def f (x : ℝ) := (2/3) * x ^ 3 + x ^ 2 + 2 * x

-- (1) Prove that the average velocity of the particle during the first second is 3 m/s
theorem average_velocity_first_second : (f 1 - f 0) / (1 - 0) = 3 := by
  sorry

-- (2) Prove that the instantaneous velocity at the end of the first second is 6 m/s
theorem instantaneous_velocity_end_first_second : deriv f 1 = 6 := by
  sorry

-- (3) Prove that the velocity of the particle reaches 14 m/s after 2 seconds
theorem velocity_reaches_14_after_2_seconds :
  ∃ x : ℝ, deriv f x = 14 ∧ x = 2 := by
  sorry

end average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l154_154622


namespace cement_percentage_first_concrete_correct_l154_154506

open Real

noncomputable def cement_percentage_of_first_concrete := 
  let total_weight := 4500 
  let cement_percentage := 10.8 / 100
  let weight_each_type := 1125
  let total_cement_weight := cement_percentage * total_weight
  let x := 2.0 / 100
  let y := 21.6 / 100 - x
  (weight_each_type * x + weight_each_type * y = total_cement_weight) →
  (x = 2.0 / 100)

theorem cement_percentage_first_concrete_correct :
  cement_percentage_of_first_concrete := sorry

end cement_percentage_first_concrete_correct_l154_154506


namespace sum_of_numbers_l154_154857

theorem sum_of_numbers (a b c d : ℕ) (h1 : a > d) (h2 : a * b = c * d) (h3 : a + b + c + d = a * c) (h4 : ∀ x y z w: ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ) : a + b + c + d = 12 :=
sorry

end sum_of_numbers_l154_154857


namespace ratio_of_x_and_y_l154_154849

theorem ratio_of_x_and_y (x y : ℝ) (h : (x - y) / (x + y) = 4) : x / y = -5 / 3 :=
by sorry

end ratio_of_x_and_y_l154_154849


namespace calculate_result_l154_154306

theorem calculate_result :
  (-24) * ((5 / 6 : ℚ) - (4 / 3) + (5 / 8)) = -3 := 
by
  sorry

end calculate_result_l154_154306


namespace math_books_count_l154_154593

theorem math_books_count (total_books : ℕ) (history_books : ℕ) (geography_books : ℕ) (math_books : ℕ) 
  (h1 : total_books = 100) 
  (h2 : history_books = 32) 
  (h3 : geography_books = 25) 
  (h4 : math_books = total_books - history_books - geography_books) 
  : math_books = 43 := 
by 
  rw [h1, h2, h3] at h4;
  exact h4;
-- use 'sorry' to skip the proof if needed
-- sorry

end math_books_count_l154_154593


namespace union_A_B_inter_complement_A_B_range_a_l154_154365

-- Define the sets A, B, and C
def A : Set ℝ := { x | 2 < x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | 5 - a < x ∧ x < a }

-- Part (I)
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x < 10 } := sorry

theorem inter_complement_A_B :
  (Set.univ \ A) ∩ B = { x | 7 ≤ x ∧ x < 10 } := sorry

-- Part (II)
theorem range_a (a : ℝ) (h : C a ⊆ B) : a ≤ 3 := sorry

end union_A_B_inter_complement_A_B_range_a_l154_154365


namespace find_f_two_l154_154836

-- Define the function f with the given properties
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1

-- Given conditions
variable (a b : ℝ)
axiom f_neg_two_zero : f (-2) a b = 0

-- Statement to be proven
theorem find_f_two : f 2 a b = 2 := 
by {
  sorry
}

end find_f_two_l154_154836


namespace range_of_abscissa_of_P_l154_154438

noncomputable def point_lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 + 1 = 0

noncomputable def point_lies_on_circle_c (M N : ℝ × ℝ) : Prop :=
  (M.1 - 2)^2 + (M.2 - 1)^2 = 1 ∧ (N.1 - 2)^2 + (N.2 - 1)^2 = 1

noncomputable def angle_mpn_eq_60 (P M N : ℝ × ℝ) : Prop :=
  true -- This is a placeholder because we have to define the geometrical angle condition which is complex.

theorem range_of_abscissa_of_P :
  ∀ (P M N : ℝ × ℝ),
  point_lies_on_line P →
  point_lies_on_circle_c M N →
  angle_mpn_eq_60 P M N →
  0 ≤ P.1 ∧ P.1 ≤ 2 := sorry

end range_of_abscissa_of_P_l154_154438


namespace basketball_team_first_competition_games_l154_154331

-- Definitions given the conditions
def first_competition_games (x : ℕ) := x
def second_competition_games (x : ℕ) := (5 * x) / 8
def third_competition_games (x : ℕ) := x + (5 * x) / 8
def total_games (x : ℕ) := x + (5 * x) / 8 + (x + (5 * x) / 8)

-- Lean 4 statement to prove the correct answer
theorem basketball_team_first_competition_games : 
  ∃ x : ℕ, total_games x = 130 ∧ first_competition_games x = 40 :=
by
  sorry

end basketball_team_first_competition_games_l154_154331


namespace andrew_eggs_bought_l154_154431

-- Define initial conditions
def initial_eggs : ℕ := 8
def final_eggs : ℕ := 70

-- Define the function to determine the number of eggs bought
def eggs_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

-- State the theorem we want to prove
theorem andrew_eggs_bought : eggs_bought initial_eggs final_eggs = 62 :=
by {
  -- Proof goes here
  sorry
}

end andrew_eggs_bought_l154_154431


namespace tiling_scheme_3_3_3_3_6_l154_154377

-- Definitions based on the conditions.
def angle_equilateral_triangle := 60
def angle_regular_hexagon := 120

-- The theorem states that using four equilateral triangles and one hexagon around a point forms a valid tiling.
theorem tiling_scheme_3_3_3_3_6 : 
  4 * angle_equilateral_triangle + angle_regular_hexagon = 360 := 
by
  -- Skip the proof with sorry
  sorry

end tiling_scheme_3_3_3_3_6_l154_154377


namespace derivative_at_2_l154_154844

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_at_2 : (deriv f 2) = 1 / 4 :=
by 
  sorry

end derivative_at_2_l154_154844


namespace eldest_child_age_l154_154384

variables (y m e : ℕ)

theorem eldest_child_age (h1 : m = y + 3)
                        (h2 : e = 3 * y)
                        (h3 : e = y + m + 2) : e = 15 :=
by
  sorry

end eldest_child_age_l154_154384


namespace scientific_notation_of_population_l154_154800

theorem scientific_notation_of_population : (85000000 : ℝ) = 8.5 * 10^7 := 
by
  sorry

end scientific_notation_of_population_l154_154800


namespace find_sixth_term_l154_154874

noncomputable def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def sum_first_n_terms (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem find_sixth_term :
  ∀ (a1 S3 : ℕ),
  a1 = 2 →
  S3 = 12 →
  ∃ d : ℕ, sum_first_n_terms a1 d 3 = S3 ∧ arithmetic_sequence a1 d 6 = 12 :=
by
  sorry

end find_sixth_term_l154_154874


namespace multiples_of_10_5_l154_154240

theorem multiples_of_10_5 (n : ℤ) (h1 : ∀ k : ℤ, k % 10 = 0 → k % 5 = 0) (h2 : n % 10 = 0) : n % 5 = 0 := 
by
  sorry

end multiples_of_10_5_l154_154240


namespace max_profit_at_l154_154924

variables (k x : ℝ) (hk : k > 0)

-- Define the quantities based on problem conditions
def profit (k x : ℝ) : ℝ :=
  0.072 * k * x ^ 2 - k * x ^ 3

-- State the theorem
theorem max_profit_at (k : ℝ) (hk : k > 0) : 
  ∃ x, profit k x = 0.072 * k * x ^ 2 - k * x ^ 3 ∧ x = 0.048 :=
sorry

end max_profit_at_l154_154924


namespace initial_potatoes_count_l154_154455

theorem initial_potatoes_count (initial_tomatoes picked_tomatoes total_remaining : ℕ) 
    (h_initial_tomatoes : initial_tomatoes = 177)
    (h_picked_tomatoes : picked_tomatoes = 53)
    (h_total_remaining : total_remaining = 136) :
  (initial_tomatoes - picked_tomatoes + x = total_remaining) → 
  x = 12 :=
by 
  sorry

end initial_potatoes_count_l154_154455


namespace weight_of_smallest_box_l154_154488

variables (M S L : ℕ)

theorem weight_of_smallest_box
  (h1 : M + S = 83)
  (h2 : L + S = 85)
  (h3 : L + M = 86) :
  S = 41 :=
sorry

end weight_of_smallest_box_l154_154488


namespace integer_solutions_count_l154_154050

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l154_154050


namespace percentage_increase_l154_154728

variable (A B y : ℝ)

theorem percentage_increase (h1 : B > A) (h2 : A > 0) :
  B = A + y / 100 * A ↔ y = 100 * (B - A) / A :=
by
  sorry

end percentage_increase_l154_154728


namespace average_speed_correct_l154_154189

-- Define the conditions
def distance_first_hour := 90 -- in km
def distance_second_hour := 30 -- in km
def time_first_hour := 1 -- in hours
def time_second_hour := 1 -- in hours

-- Define the total distance and total time
def total_distance := distance_first_hour + distance_second_hour
def total_time := time_first_hour + time_second_hour

-- Define the average speed
def avg_speed := total_distance / total_time

-- State the theorem to prove the average speed is 60
theorem average_speed_correct :
  avg_speed = 60 := 
by 
  -- Placeholder for the actual proof
  sorry

end average_speed_correct_l154_154189


namespace museum_discount_l154_154955

theorem museum_discount
  (Dorothy_age : ℕ)
  (total_family_members : ℕ)
  (regular_ticket_cost : ℕ)
  (discountapplies_age : ℕ)
  (before_trip : ℕ)
  (after_trip : ℕ)
  (spend : ℕ := before_trip - after_trip)
  (adults_tickets : ℕ := total_family_members - 2)
  (youth_tickets : ℕ := 2)
  (total_cost := adults_tickets * regular_ticket_cost + youth_tickets * (regular_ticket_cost - regular_ticket_cost * discount))
  (discount : ℚ)
  (expected_spend : ℕ := 44) :
  total_cost = spend :=
by
  sorry

end museum_discount_l154_154955


namespace unique_arrangements_of_BANANA_l154_154895

-- Define the conditions as separate definitions in Lean 4
def word := "BANANA"
def total_letters := 6
def count_A := 3
def count_N := 2
def count_B := 1

-- State the theorem to be proven
theorem unique_arrangements_of_BANANA : 
  (total_letters.factorial) / (count_A.factorial * count_N.factorial * count_B.factorial) = 60 := 
by
  sorry

end unique_arrangements_of_BANANA_l154_154895


namespace truncated_cone_volume_correct_larger_cone_volume_correct_l154_154467

def larger_base_radius : ℝ := 10 -- R
def smaller_base_radius : ℝ := 5  -- r
def height_truncated_cone : ℝ := 8 -- h
def height_small_cone : ℝ := 8 -- x

noncomputable def volume_truncated_cone : ℝ :=
  (1/3) * Real.pi * height_truncated_cone * 
  (larger_base_radius^2 + larger_base_radius * smaller_base_radius + smaller_base_radius^2)

theorem truncated_cone_volume_correct :
  volume_truncated_cone = 466 + 2/3 * Real.pi := sorry

noncomputable def total_height_larger_cone : ℝ :=
  height_small_cone + height_truncated_cone

noncomputable def volume_larger_cone : ℝ :=
  (1/3) * Real.pi * (larger_base_radius^2) * total_height_larger_cone

theorem larger_cone_volume_correct :
  volume_larger_cone = 533 + 1/3 * Real.pi := sorry

end truncated_cone_volume_correct_larger_cone_volume_correct_l154_154467


namespace probability_three_specific_cards_l154_154894

theorem probability_three_specific_cards :
  let total_deck := 52
  let total_spades := 13
  let total_tens := 4
  let total_queens := 4
  let p_case1 := ((12:ℚ) / total_deck) * (total_tens / (total_deck - 1)) * (total_queens / (total_deck - 2))
  let p_case2 := ((1:ℚ) / total_deck) * ((total_tens - 1) / (total_deck - 1)) * (total_queens / (total_deck - 2))
  p_case1 + p_case2 = (17:ℚ) / 11050 :=
by
  sorry

end probability_three_specific_cards_l154_154894


namespace coeff_exists_l154_154749

theorem coeff_exists :
  ∃ (A B C : ℕ), 
    ¬(8 ∣ A) ∧ ¬(8 ∣ B) ∧ ¬(8 ∣ C) ∧ 
    (∀ (n : ℕ), 8 ∣ (A * 5^n + B * 3^(n-1) + C))
    :=
sorry

end coeff_exists_l154_154749


namespace sphere_radius_l154_154032

theorem sphere_radius (A : ℝ) (k1 k2 k3 : ℝ) (h : A = 64 * Real.pi) : ∃ r : ℝ, r = 4 := 
by 
  sorry

end sphere_radius_l154_154032


namespace rogers_coaches_l154_154697

-- Define the structure for the problem conditions
structure snacks_problem :=
  (team_members : ℕ)
  (helpers : ℕ)
  (packs_purchased : ℕ)
  (pouches_per_pack : ℕ)

-- Create an instance of the problem with given conditions
def rogers_problem : snacks_problem :=
  { team_members := 13,
    helpers := 2,
    packs_purchased := 3,
    pouches_per_pack := 6 }

-- Define the theorem to state that given the conditions, the number of coaches is 3
theorem rogers_coaches (p : snacks_problem) : p.packs_purchased * p.pouches_per_pack - p.team_members - p.helpers = 3 :=
by
  sorry

end rogers_coaches_l154_154697


namespace augmented_matrix_solution_l154_154196

theorem augmented_matrix_solution (a b : ℝ) 
    (h1 : (∀ (x y : ℝ), (a * x = 2 ∧ y = b ↔ x = 2 ∧ y = 1))) : 
    a + b = 2 :=
by
  sorry

end augmented_matrix_solution_l154_154196


namespace cost_of_show_dogs_l154_154533

noncomputable def cost_per_dog : ℕ → ℕ → ℕ → ℕ
| total_revenue, total_profit, number_of_dogs => (total_revenue - total_profit) / number_of_dogs

theorem cost_of_show_dogs {revenue_per_puppy number_of_puppies profit number_of_dogs : ℕ}
  (h_puppies: number_of_puppies = 6)
  (h_revenue_per_puppy : revenue_per_puppy = 350)
  (h_profit : profit = 1600)
  (h_number_of_dogs : number_of_dogs = 2)
:
  cost_per_dog (number_of_puppies * revenue_per_puppy) profit number_of_dogs = 250 :=
by
  sorry

end cost_of_show_dogs_l154_154533


namespace number_of_points_in_star_polygon_l154_154499

theorem number_of_points_in_star_polygon :
  ∀ (n : ℕ) (D C : ℕ),
    (∀ i : ℕ, i < n → C = D - 15) →
    n * (D - (D - 15)) = 360 → n = 24 :=
by
  intros n D C h1 h2
  sorry

end number_of_points_in_star_polygon_l154_154499


namespace sequence_general_term_l154_154600

-- Define the sequence using a recurrence relation for clarity in formal proof
def a (n : ℕ) : ℕ :=
  if h : n > 0 then 2^n + 1 else 3

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → a n = 2^n + 1 := 
by 
  sorry

end sequence_general_term_l154_154600


namespace geometric_sequence_a2_l154_154245

noncomputable def geometric_sequence_sum (n : ℕ) (a : ℝ) : ℝ :=
  a * (3^n) - 2

theorem geometric_sequence_a2 (a : ℝ) : (∃ a1 a2 a3 : ℝ, 
  a1 = geometric_sequence_sum 1 a ∧ 
  a1 + a2 = geometric_sequence_sum 2 a ∧ 
  a1 + a2 + a3 = geometric_sequence_sum 3 a ∧ 
  a2 = 6 * a ∧ 
  a3 = 18 * a ∧ 
  (6 * a)^2 = (a1) * (a3) ∧ 
  a = 2) →
  a2 = 12 :=
by
  intros h
  sorry

end geometric_sequence_a2_l154_154245


namespace valid_N_values_l154_154515

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l154_154515


namespace greater_solution_of_quadratic_l154_154879

theorem greater_solution_of_quadratic :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 5 * x₁ - 84 = 0) ∧ (x₂^2 - 5 * x₂ - 84 = 0) ∧ (max x₁ x₂ = 12) :=
by
  sorry

end greater_solution_of_quadratic_l154_154879


namespace sequence_positions_l154_154324

noncomputable def position_of_a4k1 (x : ℕ) : ℕ := 4 * x + 1
noncomputable def position_of_a4k2 (x : ℕ) : ℕ := 4 * x + 2
noncomputable def position_of_a4k3 (x : ℕ) : ℕ := 4 * x + 3
noncomputable def position_of_a4k (x : ℕ) : ℕ := 4 * x

theorem sequence_positions (k : ℕ) :
  (6 + 1964 = 1970 ∧ position_of_a4k1 1964 = 7857) ∧
  (6 + 1965 = 1971 ∧ position_of_a4k1 1965 = 7861) ∧
  (8 + 1962 = 1970 ∧ position_of_a4k2 1962 = 7850) ∧
  (8 + 1963 = 1971 ∧ position_of_a4k2 1963 = 7854) ∧
  (16 + 2 * 977 = 1970 ∧ position_of_a4k3 977 = 3911) ∧
  (14 + 2 * (979 - 1) = 1970 ∧ position_of_a4k 979 = 3916) :=
by sorry

end sequence_positions_l154_154324


namespace find_first_number_l154_154623

theorem find_first_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 60 + 35) / 3 + 5 → 
  x = 10 := 
by 
  sorry

end find_first_number_l154_154623


namespace parabola_y_range_l154_154673

theorem parabola_y_range
  (x y : ℝ)
  (M_on_C : x^2 = 8 * y)
  (F : ℝ × ℝ)
  (F_focus : F = (0, 2))
  (circle_intersects_directrix : F.2 + y > 4) :
  y > 2 :=
by
  sorry

end parabola_y_range_l154_154673


namespace least_number_to_add_l154_154188

theorem least_number_to_add (x : ℕ) (h : 53 ∣ x ∧ 71 ∣ x) : 
  ∃ n : ℕ, x = 1357 + n ∧ n = 2406 :=
by sorry

end least_number_to_add_l154_154188


namespace ab_value_l154_154025

theorem ab_value (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ∧ (∀ y : ℝ, (x = 0 ∧ (y = 5 ∨ y = -5)))))
  (h2 : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ∧ (∀ x : ℝ, (y = 0 ∧ (x = 8 ∨ x = -8))))) :
  |a * b| = Real.sqrt 867.75 :=
by
  sorry

end ab_value_l154_154025


namespace range_of_a_l154_154194

theorem range_of_a (a : ℝ) :
  (∀ (x y z: ℝ), x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) ↔ (a ≤ -2 ∨ a ≥ 4) :=
by
sorry

end range_of_a_l154_154194


namespace part1_inequality_solution_l154_154054

def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 3|

theorem part1_inequality_solution :
  ∀ x : ℝ, f x ≤ 6 ↔ -4 / 3 ≤ x ∧ x ≤ 8 / 3 :=
by sorry

end part1_inequality_solution_l154_154054


namespace initial_walnut_trees_l154_154274

/-- 
  Given there are 29 walnut trees in the park after cutting down 13 walnut trees, 
  prove that initially there were 42 walnut trees in the park.
-/
theorem initial_walnut_trees (cut_walnut_trees remaining_walnut_trees initial_walnut_trees : ℕ) 
  (h₁ : cut_walnut_trees = 13)
  (h₂ : remaining_walnut_trees = 29)
  (h₃ : initial_walnut_trees = cut_walnut_trees + remaining_walnut_trees) :
  initial_walnut_trees = 42 := 
sorry

end initial_walnut_trees_l154_154274


namespace mushroom_pickers_l154_154386

theorem mushroom_pickers (n : ℕ) (hn : n = 18) (total_mushrooms : ℕ) (h_total : total_mushrooms = 162) (h_each : ∀ i : ℕ, i < n → 0 < 1) : 
  ∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (total_mushrooms / n = (total_mushrooms / n)) :=
sorry

end mushroom_pickers_l154_154386


namespace num_valid_constants_m_l154_154698

theorem num_valid_constants_m : 
  ∃ (m1 m2 : ℝ), 
  m1 ≠ m2 ∧ 
  (∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∧ 8 = m1 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∧ (1 / 2) = m2 ∨ 2 * c / d = 2)) ∧
  (∀ (m : ℝ), 
    (m = m1 ∨ m = m2) →
    ∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∨ 2 * c / d = 2)) :=
sorry

end num_valid_constants_m_l154_154698


namespace unique_fraction_satisfying_condition_l154_154661

theorem unique_fraction_satisfying_condition : ∃! (x y : ℕ), Nat.gcd x y = 1 ∧ y ≠ 0 ∧ (x + 1) * 5 * y = (y + 1) * 6 * x :=
by
  sorry

end unique_fraction_satisfying_condition_l154_154661


namespace factor_expression_l154_154694

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 2 * x * (x + 3) + (x + 3) = (x + 1)^2 * (x + 3) := by
  sorry

end factor_expression_l154_154694


namespace angle_coterminal_l154_154812

theorem angle_coterminal (k : ℤ) : 
  ∃ α : ℝ, α = 30 + k * 360 :=
sorry

end angle_coterminal_l154_154812


namespace total_notes_count_l154_154548

theorem total_notes_count :
  ∀ (rows : ℕ) (notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ),
  rows = 5 →
  notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  (rows * notes_per_row + (rows * notes_per_row * blue_notes_per_red + additional_blue_notes)) = 100 := by
  intros rows notes_per_row blue_notes_per_red additional_blue_notes
  sorry

end total_notes_count_l154_154548


namespace leap_year_hours_l154_154974

theorem leap_year_hours (days_in_regular_year : ℕ) (hours_in_day : ℕ) (is_leap_year : Bool) : 
  is_leap_year = true ∧ days_in_regular_year = 365 ∧ hours_in_day = 24 → 
  366 * hours_in_day = 8784 :=
by
  intros
  sorry

end leap_year_hours_l154_154974


namespace find_n_l154_154550

theorem find_n (n : ℕ) (h : (1 + n + (n * (n - 1)) / 2) / 2^n = 7 / 32) : n = 6 :=
sorry

end find_n_l154_154550


namespace rocket_altitude_time_l154_154936

theorem rocket_altitude_time (a₁ d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 2)
  (h₃ : n * a₁ + (n * (n - 1) * d) / 2 = 240) : n = 15 :=
by
  -- The proof is ignored as per instruction.
  sorry

end rocket_altitude_time_l154_154936


namespace power_function_m_eq_4_l154_154983

theorem power_function_m_eq_4 (m : ℝ) :
  (m^2 - 3*m - 3 = 1) → m = 4 :=
by
  sorry

end power_function_m_eq_4_l154_154983


namespace basketball_free_throws_l154_154724

theorem basketball_free_throws (total_players : ℕ) (number_captains : ℕ) (players_not_including_one : ℕ) 
  (free_throws_per_captain : ℕ) (total_free_throws : ℕ) 
  (h1 : total_players = 15)
  (h2 : number_captains = 2)
  (h3 : players_not_including_one = total_players - 1)
  (h4 : free_throws_per_captain = players_not_including_one * number_captains)
  (h5 : total_free_throws = free_throws_per_captain)
  : total_free_throws = 28 :=
by
  -- Proof is not required, so we provide sorry to skip it.
  sorry

end basketball_free_throws_l154_154724


namespace John_can_lift_now_l154_154654

def originalWeight : ℕ := 135
def trainingIncrease : ℕ := 265
def bracerIncreaseFactor : ℕ := 6

def newWeight : ℕ := originalWeight + trainingIncrease
def bracerIncrease : ℕ := newWeight * bracerIncreaseFactor
def totalWeight : ℕ := newWeight + bracerIncrease

theorem John_can_lift_now :
  totalWeight = 2800 :=
by
  -- proof steps go here
  sorry

end John_can_lift_now_l154_154654


namespace units_digit_product_l154_154835

theorem units_digit_product (a b : ℕ) (h1 : (a % 10 ≠ 0) ∧ (b % 10 ≠ 0)) : (a * b % 10 = 0) ∨ (a * b % 10 ≠ 0) :=
by
  sorry

end units_digit_product_l154_154835


namespace vertex_below_x_axis_l154_154382

theorem vertex_below_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + a < 0) → a < 1 :=
by 
  sorry

end vertex_below_x_axis_l154_154382


namespace sum_of_six_consecutive_integers_l154_154456

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l154_154456


namespace red_sequence_2018th_num_l154_154396

/-- Define the sequence of red-colored numbers based on the given conditions. -/
def red_sequenced_num (n : Nat) : Nat :=
  let k := Nat.sqrt (2 * n - 1) -- estimate block number
  let block_start := if k % 2 == 0 then (k - 1)*(k - 1) else k * (k - 1) + 1
  let position_in_block := n - (k * (k - 1) / 2) - 1
  if k % 2 == 0 then block_start + 2 * position_in_block else block_start + 2 * position_in_block

/-- Statement to assert the 2018th number is 3972 -/
theorem red_sequence_2018th_num : red_sequenced_num 2018 = 3972 := by
  sorry

end red_sequence_2018th_num_l154_154396


namespace find_FC_l154_154828

variable (DC : ℝ) (CB : ℝ) (AB AD ED : ℝ)
variable (FC : ℝ)
variable (h1 : DC = 9)
variable (h2 : CB = 6)
variable (h3 : AB = (1/3) * AD)
variable (h4 : ED = (2/3) * AD)

theorem find_FC : FC = 9 :=
by sorry

end find_FC_l154_154828


namespace initial_dogs_l154_154711

theorem initial_dogs (D : ℕ) (h : D + 5 + 3 = 10) : D = 2 :=
by sorry

end initial_dogs_l154_154711


namespace measure_of_angle_F_l154_154965

-- Definitions for the angles in triangle DEF
variables (D E F : ℝ)

-- Given conditions
def is_right_triangle (D : ℝ) : Prop := D = 90
def angle_relation (E F : ℝ) : Prop := E = 4 * F - 10
def angle_sum (D E F : ℝ) : Prop := D + E + F = 180

-- The proof problem statement
theorem measure_of_angle_F (h1 : is_right_triangle D) (h2 : angle_relation E F) (h3 : angle_sum D E F) : F = 20 :=
sorry

end measure_of_angle_F_l154_154965


namespace express_as_sum_of_cubes_l154_154477

variables {a b : ℝ}

theorem express_as_sum_of_cubes (a b : ℝ) : 
  2 * a * (a^2 + 3 * b^2) = (a + b)^3 + (a - b)^3 :=
by sorry

end express_as_sum_of_cubes_l154_154477


namespace problem1_problem2_l154_154444

open Real -- Open the Real namespace to use real number trigonometric functions

-- Problem 1
theorem problem1 (α : ℝ) (hα : tan α = 3) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5/7 :=
sorry

-- Problem 2
theorem problem2 (θ : ℝ) (hθ : tan θ = -3/4) : 
  2 + sin θ * cos θ - cos θ ^ 2 = 22 / 25 :=
sorry

end problem1_problem2_l154_154444


namespace roots_greater_than_one_l154_154387

def quadratic_roots_greater_than_one (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1

theorem roots_greater_than_one (a : ℝ) :
  -16/7 < a ∧ a < -1 → quadratic_roots_greater_than_one a :=
sorry

end roots_greater_than_one_l154_154387


namespace base_addition_is_10_l154_154522

-- The problem states that adding two numbers in a particular base results in a third number in the same base.
def valid_base_10_addition (n m k b : ℕ) : Prop :=
  let n_b := n / b^2 * b^2 + (n / b % b) * b + n % b
  let m_b := m / b^2 * b^2 + (m / b % b) * b + m % b
  let k_b := k / b^2 * b^2 + (k / b % b) * b + k % b
  n_b + m_b = k_b

theorem base_addition_is_10 : valid_base_10_addition 172 156 340 10 :=
  sorry

end base_addition_is_10_l154_154522


namespace faye_gave_away_books_l154_154668

theorem faye_gave_away_books (x : ℕ) (H1 : 34 - x + 48 = 79) : x = 3 :=
by {
  sorry
}

end faye_gave_away_books_l154_154668


namespace number_of_friends_is_five_l154_154581

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end number_of_friends_is_five_l154_154581


namespace proof_2_abs_a_plus_b_less_abs_4_plus_ab_l154_154848

theorem proof_2_abs_a_plus_b_less_abs_4_plus_ab (a b : ℝ) (h1 : abs a < 2) (h2 : abs b < 2) :
    2 * abs (a + b) < abs (4 + a * b) := 
by
  sorry

end proof_2_abs_a_plus_b_less_abs_4_plus_ab_l154_154848


namespace notebooks_last_days_l154_154284

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end notebooks_last_days_l154_154284


namespace incorrect_description_l154_154059

-- Conditions
def population_size : ℕ := 2000
def sample_size : ℕ := 150

-- Main Statement
theorem incorrect_description : ¬ (sample_size = 150) := 
by sorry

end incorrect_description_l154_154059


namespace select_4_people_arrangement_3_day_new_year_l154_154773

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end select_4_people_arrangement_3_day_new_year_l154_154773


namespace driving_distance_l154_154676

theorem driving_distance:
  ∀ a b: ℕ, (a + b = 500 ∧ a ≥ 150 ∧ b ≥ 150) → 
  (⌊Real.sqrt (a^2 + b^2)⌋ = 380) :=
by
  intro a b
  intro h
  sorry

end driving_distance_l154_154676


namespace bridge_length_l154_154072

theorem bridge_length (lorry_length : ℝ) (lorry_speed_kmph : ℝ) (cross_time_seconds : ℝ) : 
  lorry_length = 200 ∧ lorry_speed_kmph = 80 ∧ cross_time_seconds = 17.998560115190784 →
  lorry_length + lorry_speed_kmph * (1000 / 3600) * cross_time_seconds = 400 → 
  400 - lorry_length = 200 :=
by
  intro h₁ h₂
  cases h₁
  sorry

end bridge_length_l154_154072


namespace part1_part2_l154_154158

noncomputable def f (m x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem part1 (t : ℝ) :
  (1 / 2 < t ∧ t < 1) →
  (∃! t : ℝ, f 1 t = 0) := sorry

theorem part2 :
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) := sorry

end part1_part2_l154_154158


namespace problem_proof_l154_154969

theorem problem_proof :
  1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 :=
by
  sorry

end problem_proof_l154_154969


namespace only_integer_solution_is_zero_l154_154762

theorem only_integer_solution_is_zero (x y : ℤ) (h : x^4 + y^4 = 3 * x^3 * y) : x = 0 ∧ y = 0 :=
by {
  -- Here we would provide the proof steps.
  sorry
}

end only_integer_solution_is_zero_l154_154762


namespace binomial_12_3_eq_220_l154_154399

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l154_154399


namespace common_chord_equation_l154_154578

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_equation :
  ∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧
                     ∀ (x y : ℝ), (x - 2*y + 4 = 0) ↔ ((x, y) = A ∨ (x, y) = B) :=
by
  sorry

end common_chord_equation_l154_154578


namespace simplify_fraction_l154_154235

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 :=
by
  sorry

end simplify_fraction_l154_154235


namespace find_n_l154_154613

theorem find_n (n : ℕ) (b : Fin (n + 1) → ℝ) (h0 : b 0 = 45) (h1 : b 1 = 81) (hn : b n = 0) (rec : ∀ (k : ℕ), 1 ≤ k → k < n → b (k+1) = b (k-1) - 5 / b k) : 
  n = 730 :=
sorry

end find_n_l154_154613


namespace distance_DE_l154_154970

noncomputable def point := (ℝ × ℝ)

variables (A B C P D E : point)
variables (AB BC AC PC : ℝ)
variables (on_line : point → point → point → Prop)
variables (is_parallel : point → point → point → point → Prop)

axiom AB_length : AB = 13
axiom BC_length : BC = 14
axiom AC_length : AC = 15
axiom PC_length : PC = 10

axiom P_on_AC : on_line A C P
axiom D_on_BP : on_line B P D
axiom E_on_BP : on_line B P E

axiom AD_parallel_BC : is_parallel A D B C
axiom AB_parallel_CE : is_parallel A B C E

theorem distance_DE : ∀ (D E : point), 
  on_line B P D → on_line B P E → 
  is_parallel A D B C → is_parallel A B C E → 
  ∃ dist : ℝ, dist = 12 * Real.sqrt 2 :=
by
  sorry

end distance_DE_l154_154970


namespace problem_l154_154209

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end problem_l154_154209


namespace smallest_k_l154_154052

-- Define the set S
def S (m : ℕ) : Finset ℕ :=
  (Finset.range (30 * m)).filter (λ n => n % 2 = 1 ∧ n % 5 ≠ 0)

-- Theorem statement
theorem smallest_k (m : ℕ) (k : ℕ) : 
  (∀ (A : Finset ℕ), A ⊆ S m → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x ∣ y ∨ y ∣ x)) ↔ k ≥ 8 * m + 1 :=
sorry

end smallest_k_l154_154052


namespace find_coordinates_of_B_l154_154084

theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (h1 : ∃ (C1 C2 : ℝ × ℝ), C1.2 = 0 ∧ C2.2 = 0 ∧ (dist C1 A = dist C1 B) ∧ (dist C2 A = dist C2 B) ∧ (A ≠ B))
  (h2 : A = (-3, 2)) :
  B = (-3, -2) :=
sorry

end find_coordinates_of_B_l154_154084


namespace find_a_range_l154_154256

noncomputable
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x ^ 2 - 2 * x

theorem find_a_range (a : ℝ) : (∀ x : ℝ, -1 / Real.exp 1 ≤ f x a) → a ∈ Set.Ici (Real.exp 1) :=
  sorry

end find_a_range_l154_154256


namespace sum_of_net_gains_is_correct_l154_154426

namespace DepartmentRevenue

def revenueIncreaseA : ℝ := 0.1326
def revenueIncreaseB : ℝ := 0.0943
def revenueIncreaseC : ℝ := 0.7731
def taxRate : ℝ := 0.235
def initialRevenue : ℝ := 4.7 -- in millions

def netGain (revenueIncrease : ℝ) (taxRate : ℝ) (initialRevenue : ℝ) : ℝ :=
  (initialRevenue * (1 + revenueIncrease)) * (1 - taxRate)

def netGainA : ℝ := netGain revenueIncreaseA taxRate initialRevenue
def netGainB : ℝ := netGain revenueIncreaseB taxRate initialRevenue
def netGainC : ℝ := netGain revenueIncreaseC taxRate initialRevenue

def netGainSum : ℝ := netGainA + netGainB + netGainC

theorem sum_of_net_gains_is_correct :
  netGainSum = 14.38214 := by
    sorry

end DepartmentRevenue

end sum_of_net_gains_is_correct_l154_154426


namespace minimal_pyramid_height_l154_154524

theorem minimal_pyramid_height (r x a : ℝ) (h₁ : 0 < r) (h₂ : a = 2 * r * x / (x - r)) (h₃ : x > 4 * r) :
  x = (6 + 2 * Real.sqrt 3) * r :=
by
  -- Proof steps would go here
  sorry

end minimal_pyramid_height_l154_154524


namespace container_capacity_l154_154538

theorem container_capacity (C : ℝ) (h₁ : C > 15) (h₂ : 0 < (81 : ℝ)) (h₃ : (337 : ℝ) > 0) :
  ((C - 15) / C) ^ 4 = 81 / 337 :=
sorry

end container_capacity_l154_154538


namespace prime_divisor_property_l154_154301

open Classical

theorem prime_divisor_property (p n q : ℕ) (hp : Nat.Prime p) (hn : 0 < n) (hq : q ∣ (n + 1)^p - n^p) : p ∣ q - 1 :=
by
  sorry

end prime_divisor_property_l154_154301


namespace sum_in_range_l154_154310

def a : ℚ := 4 + 1/4
def b : ℚ := 2 + 3/4
def c : ℚ := 7 + 1/8

theorem sum_in_range : 14 < a + b + c ∧ a + b + c < 15 := by
  sorry

end sum_in_range_l154_154310


namespace product_of_two_numbers_is_21_l154_154972

noncomputable def product_of_two_numbers (x y : ℝ) : ℝ :=
  x * y

theorem product_of_two_numbers_is_21 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^2 + y^2 = 58) :
  product_of_two_numbers x y = 21 :=
by sorry

end product_of_two_numbers_is_21_l154_154972


namespace polynomial_decomposition_l154_154108

noncomputable def s (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1
noncomputable def t (x : ℝ) : ℝ := x + 18

def g (x : ℝ) : ℝ := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 6
def e (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem polynomial_decomposition : s 1 + t (-1) = 27 :=
by
  sorry

end polynomial_decomposition_l154_154108


namespace symmetric_sum_l154_154872

theorem symmetric_sum (m n : ℤ) (hA : n = 3) (hB : m = -2) : m + n = 1 :=
by
  rw [hA, hB]
  exact rfl

end symmetric_sum_l154_154872


namespace math_class_problem_l154_154963

theorem math_class_problem
  (x a : ℝ)
  (h_mistaken : (2 * (2 * 4 - 1) + 1 = 5 * (4 + a)))
  (h_original : (2 * x - 1) / 5 + 1 = (x + a) / 2)
  : a = -1 ∧ x = 13 := by
  sorry

end math_class_problem_l154_154963


namespace leak_time_to_empty_tank_l154_154968

-- Define variables for the rates
variable (A L : ℝ)

-- Given conditions
def rate_pipe_A : Prop := A = 1 / 4
def combined_rate : Prop := A - L = 1 / 6

-- Theorem statement: The time it takes for the leak to empty the tank
theorem leak_time_to_empty_tank (A L : ℝ) (h1 : rate_pipe_A A) (h2 : combined_rate A L) : 1 / L = 12 :=
by 
  sorry

end leak_time_to_empty_tank_l154_154968


namespace quadratic_equation_with_product_of_roots_20_l154_154520

theorem quadratic_equation_with_product_of_roots_20
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c / a = 20) :
  ∃ b : ℝ, ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  use 1
  use 20
  sorry

end quadratic_equation_with_product_of_roots_20_l154_154520


namespace sequence_expression_l154_154179

theorem sequence_expression (s a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → s n = (3 / 2 * (a n - 1))) :
  ∀ n : ℕ, 1 ≤ n → a n = 3^n :=
by
  sorry

end sequence_expression_l154_154179


namespace final_lights_on_l154_154160

def lights_on_by_children : ℕ :=
  let total_lights := 200
  let flips_x := total_lights / 7
  let flips_y := total_lights / 11
  let lcm_xy := 77  -- since lcm(7, 11) = 7 * 11 = 77
  let flips_both := total_lights / lcm_xy
  flips_x + flips_y - flips_both

theorem final_lights_on : lights_on_by_children = 44 :=
by
  sorry

end final_lights_on_l154_154160


namespace jaewoong_ran_the_most_l154_154436

def distance_jaewoong : ℕ := 20000 -- Jaewoong's distance in meters
def distance_seongmin : ℕ := 2600  -- Seongmin's distance in meters
def distance_eunseong : ℕ := 5000  -- Eunseong's distance in meters

theorem jaewoong_ran_the_most : distance_jaewoong > distance_seongmin ∧ distance_jaewoong > distance_eunseong := by
  sorry

end jaewoong_ran_the_most_l154_154436


namespace vertical_shift_d_l154_154227

variable (a b c d : ℝ)

theorem vertical_shift_d (h1: d + a = 5) (h2: d - a = 1) : d = 3 := 
by
  sorry

end vertical_shift_d_l154_154227


namespace bus_stop_time_per_hour_l154_154843

theorem bus_stop_time_per_hour
  (speed_no_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_no_stops = 50)
  (h2 : speed_with_stops = 35) : 
  18 = (60 * (1 - speed_with_stops / speed_no_stops)) :=
by
  sorry

end bus_stop_time_per_hour_l154_154843


namespace people_on_williams_bus_l154_154362

theorem people_on_williams_bus
  (P : ℕ)
  (dutch_people : ℕ)
  (dutch_americans : ℕ)
  (window_seats : ℕ)
  (h1 : dutch_people = (3 * P) / 5)
  (h2 : dutch_americans = dutch_people / 2)
  (h3 : window_seats = dutch_americans / 3)
  (h4 : window_seats = 9) : 
  P = 90 :=
sorry

end people_on_williams_bus_l154_154362


namespace boys_camp_percentage_l154_154701

theorem boys_camp_percentage (x : ℕ) (total_boys : ℕ) (percent_science : ℕ) (not_science_boys : ℕ) 
    (percent_not_science : ℕ) (h1 : not_science_boys = percent_not_science * (x / 100) * total_boys) 
    (h2 : percent_not_science = 100 - percent_science) (h3 : percent_science = 30) 
    (h4 : not_science_boys = 21) (h5 : total_boys = 150) : x = 20 :=
by 
  sorry

end boys_camp_percentage_l154_154701


namespace greatest_power_of_two_l154_154221

theorem greatest_power_of_two (n : ℕ) (h1 : n = 1004) (h2 : 10^n - 4^(n / 2) = k) : ∃ m : ℕ, 2 ∣ k ∧ m = 1007 :=
by
  sorry

end greatest_power_of_two_l154_154221


namespace tan_x_neg7_l154_154837

theorem tan_x_neg7 (x : ℝ) (h1 : Real.sin (x + π / 4) = 3 / 5) (h2 : Real.sin (x - π / 4) = 4 / 5) : 
  Real.tan x = -7 :=
sorry

end tan_x_neg7_l154_154837


namespace probability_diff_color_balls_l154_154752

theorem probability_diff_color_balls 
  (Box_A_red : ℕ) (Box_A_black : ℕ) (Box_A_white : ℕ) 
  (Box_B_yellow : ℕ) (Box_B_black : ℕ) (Box_B_white : ℕ) 
  (hA : Box_A_red = 3 ∧ Box_A_black = 3 ∧ Box_A_white = 3)
  (hB : Box_B_yellow = 2 ∧ Box_B_black = 2 ∧ Box_B_white = 2) :
  ((Box_A_red * (Box_B_black + Box_B_white + Box_B_yellow))
  + (Box_A_black * (Box_B_yellow + Box_B_white))
  + (Box_A_white * (Box_B_black + Box_B_yellow))) / 
  ((Box_A_red + Box_A_black + Box_A_white) * 
  (Box_B_yellow + Box_B_black + Box_B_white)) = 7 / 9 := 
by
  sorry

end probability_diff_color_balls_l154_154752


namespace inequality_for_large_exponent_l154_154706

theorem inequality_for_large_exponent (u : ℕ → ℕ) (x : ℕ) (k : ℕ) (hk : k = 100) (hu : u x = 2^x) : 
  2^(2^(x : ℕ)) > 2^(k * x) :=
by 
  sorry

end inequality_for_large_exponent_l154_154706


namespace direct_proportion_conditions_l154_154385

theorem direct_proportion_conditions (k b : ℝ) : 
  (y = (k - 4) * x + b → (k ≠ 4 ∧ b = 0)) ∧ ¬ (b ≠ 0 ∨ k ≠ 4) :=
sorry

end direct_proportion_conditions_l154_154385


namespace tea_blend_ratio_l154_154115

theorem tea_blend_ratio (x y : ℝ)
  (h1 : 18 * x + 20 * y = (21 * (x + y)) / 1.12)
  (h2 : x + y ≠ 0) :
  x / y = 5 / 3 :=
by
  -- proof will go here
  sorry

end tea_blend_ratio_l154_154115


namespace negation_of_exists_x_lt_0_l154_154680

theorem negation_of_exists_x_lt_0 :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) :=
by {
  sorry
}

end negation_of_exists_x_lt_0_l154_154680


namespace critical_force_rod_truncated_cone_l154_154356

-- Define the given conditions
variable (r0 : ℝ) (q : ℝ) (E : ℝ) (l : ℝ) (π : ℝ)

-- Assumptions
axiom q_positive : q > 0

-- Definition for the new radius based on q
def r1 : ℝ := r0 * (1 + q)

-- Proof problem statement
theorem critical_force_rod_truncated_cone (h : q > 0) : 
  ∃ Pkp : ℝ, Pkp = (E * π * r0^4 * 4.743 / l^2) * (1 + 2 * q) :=
sorry

end critical_force_rod_truncated_cone_l154_154356


namespace add_to_any_integer_l154_154086

theorem add_to_any_integer (y : ℤ) : (∀ x : ℤ, y + x = x) → y = 0 :=
  by
  sorry

end add_to_any_integer_l154_154086


namespace math_problem_l154_154246

theorem math_problem : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end math_problem_l154_154246


namespace inequality_x_alpha_y_beta_l154_154070

theorem inequality_x_alpha_y_beta (x y α β : ℝ) (hx : 0 < x) (hy : 0 < y) 
(hα : 0 < α) (hβ : 0 < β) (hαβ : α + β = 1) : x^α * y^β ≤ α * x + β * y := 
sorry

end inequality_x_alpha_y_beta_l154_154070


namespace intersection_M_N_l154_154022

open Set Real

def M : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def N : Set ℝ := {x | log x / log 2 ≤ 1}

theorem intersection_M_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_M_N_l154_154022


namespace cost_price_of_article_l154_154851

-- Define the conditions
variable (C : ℝ) -- Cost price of the article
variable (SP : ℝ) -- Selling price of the article

-- Conditions according to the problem
def condition1 : Prop := SP = 0.75 * C
def condition2 : Prop := SP + 500 = 1.15 * C

-- The theorem to prove the cost price
theorem cost_price_of_article (h₁ : condition1 C SP) (h₂ : condition2 C SP) : C = 1250 :=
by
  sorry

end cost_price_of_article_l154_154851


namespace zeros_not_adjacent_probability_l154_154688

-- Definitions based on the conditions
def total_arrangements : ℕ := Nat.choose 6 2
def non_adjacent_zero_arrangements : ℕ := Nat.choose 5 2

-- The probability that the 2 zeros are not adjacent
def probability_non_adjacent_zero : ℚ :=
  (non_adjacent_zero_arrangements : ℚ) / (total_arrangements : ℚ)

-- The theorem statement
theorem zeros_not_adjacent_probability :
  probability_non_adjacent_zero = 2 / 3 :=
by
  -- The proof would go here
  sorry

end zeros_not_adjacent_probability_l154_154688


namespace total_rent_of_field_is_correct_l154_154935

namespace PastureRental

def cowMonths (cows : ℕ) (months : ℕ) : ℕ := cows * months

def aCowMonths : ℕ := cowMonths 24 3
def bCowMonths : ℕ := cowMonths 10 5
def cCowMonths : ℕ := cowMonths 35 4
def dCowMonths : ℕ := cowMonths 21 3

def totalCowMonths : ℕ := aCowMonths + bCowMonths + cCowMonths + dCowMonths

def rentPerCowMonth : ℕ := 1440 / aCowMonths

def totalRent : ℕ := rentPerCowMonth * totalCowMonths

theorem total_rent_of_field_is_correct :
  totalRent = 6500 :=
by
  sorry

end PastureRental

end total_rent_of_field_is_correct_l154_154935


namespace find_P_nplus1_l154_154497

-- Conditions
def P (n : ℕ) (k : ℕ) : ℚ :=
  1 / Nat.choose n k

-- Lean 4 statement for the proof
theorem find_P_nplus1 (n : ℕ) : (if Even n then P n (n+1) = 1 else P n (n+1) = 0) := by
  sorry

end find_P_nplus1_l154_154497


namespace kitchen_upgrade_cost_l154_154944

def total_kitchen_upgrade_cost (num_knobs : ℕ) (cost_per_knob : ℝ) (num_pulls : ℕ) (cost_per_pull : ℝ) : ℝ :=
  (num_knobs * cost_per_knob) + (num_pulls * cost_per_pull)

theorem kitchen_upgrade_cost : total_kitchen_upgrade_cost 18 2.50 8 4.00 = 77.00 :=
  by
    sorry

end kitchen_upgrade_cost_l154_154944


namespace combined_weight_of_three_boxes_l154_154537

theorem combined_weight_of_three_boxes (a b c d : ℕ) (h₁ : a + b = 132) (h₂ : a + c = 136) (h₃ : b + c = 138) (h₄ : d = 60) : 
  a + b + c = 203 :=
sorry

end combined_weight_of_three_boxes_l154_154537


namespace larger_factor_of_lcm_l154_154200

theorem larger_factor_of_lcm (A B : ℕ) (hcf lcm X Y : ℕ) 
  (h_hcf: hcf = 63)
  (h_A: A = 1071)
  (h_lcm: lcm = hcf * X * Y)
  (h_X: X = 11)
  (h_factors: ∃ k: ℕ, A = hcf * k ∧ lcm = A * (B / k)):
  Y = 17 := 
by sorry

end larger_factor_of_lcm_l154_154200


namespace solution_set_of_inequality_l154_154914

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h1 : ∀ x : ℝ, deriv f x = 2 * f x)
                                    (h2 : f 0 = 1) :
  { x : ℝ | f (Real.log (x^2 - x)) < 4 } = { x | -1 < x ∧ x < 0 ∨ 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_of_inequality_l154_154914


namespace bacon_calories_percentage_l154_154402

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end bacon_calories_percentage_l154_154402


namespace find_c_l154_154131

noncomputable def f (x a b c : ℤ) := x^3 + a * x^2 + b * x + c

theorem find_c (a b c : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : f a a b c = a^3) (h₄ : f b a b c = b^3) : c = 16 :=
sorry

end find_c_l154_154131


namespace number_of_machines_in_first_group_l154_154123

-- Define the initial conditions
def first_group_production_rate (x : ℕ) : ℚ :=
  20 / (x * 10)

def second_group_production_rate : ℚ :=
  180 / (20 * 22.5)

-- The theorem we aim to prove
theorem number_of_machines_in_first_group (x : ℕ) (h1 : first_group_production_rate x = second_group_production_rate) :
  x = 5 :=
by
  -- Placeholder for the proof steps
  sorry

end number_of_machines_in_first_group_l154_154123


namespace square_area_ratio_l154_154928

theorem square_area_ratio (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a^2) = 16 * (b^2) := 
by sorry

end square_area_ratio_l154_154928


namespace probability_of_selecting_one_painted_face_and_one_unpainted_face_l154_154577

noncomputable def probability_of_specific_selection :
  ℕ → ℕ → ℕ → ℚ
| total_cubes, painted_face_cubes, unpainted_face_cubes =>
  let total_pairs := (total_cubes * (total_cubes - 1)) / 2
  let success_pairs := painted_face_cubes * unpainted_face_cubes
  success_pairs / total_pairs

theorem probability_of_selecting_one_painted_face_and_one_unpainted_face :
  probability_of_specific_selection 36 13 17 = 221 / 630 :=
by
  sorry

end probability_of_selecting_one_painted_face_and_one_unpainted_face_l154_154577


namespace intersection_point_l154_154602

theorem intersection_point (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : x + y - 3 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l154_154602


namespace part1_part2_l154_154294

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Proof statements to be demonstrated
theorem part1 (a : ℝ) : a = 1 := sorry

theorem part2 (f_inv : ℝ → ℝ) : 
  (∀ x : ℝ, x > -1 ∧ x < 1 → f (f_inv x) = x ∧ f_inv (f x) = x) :=
sorry

end part1_part2_l154_154294


namespace zhang_hua_repayment_l154_154118

noncomputable def principal_amount : ℕ := 480000
noncomputable def repayment_period : ℕ := 240
noncomputable def monthly_interest_rate : ℝ := 0.004
noncomputable def principal_payment : ℝ := principal_amount / repayment_period -- 2000, but keeping general form

noncomputable def interest (month : ℕ) : ℝ :=
  (principal_amount - (month - 1) * principal_payment) * monthly_interest_rate

noncomputable def monthly_repayment (month : ℕ) : ℝ :=
  principal_payment + interest month

theorem zhang_hua_repayment (n : ℕ) (h : 1 ≤ n ∧ n ≤ repayment_period) :
  monthly_repayment n = 3928 - 8 * n := 
by
  -- proof would be placed here
  sorry

end zhang_hua_repayment_l154_154118


namespace common_point_of_function_and_inverse_l154_154337

-- Define the points P, Q, M, and N
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (1, 2)
def M : ℝ × ℝ := (2, 3)
def N : ℝ × ℝ := (0.5, 0.25)

-- Define a predicate to check if a point lies on the line y = x
def lies_on_y_eq_x (point : ℝ × ℝ) : Prop := point.1 = point.2

-- The main theorem statement
theorem common_point_of_function_and_inverse (a : ℝ) : 
  lies_on_y_eq_x P ∧ ¬ lies_on_y_eq_x Q ∧ ¬ lies_on_y_eq_x M ∧ ¬ lies_on_y_eq_x N :=
by
  -- We write 'sorry' here to skip the proof
  sorry

end common_point_of_function_and_inverse_l154_154337


namespace equation_of_line_l154_154311

theorem equation_of_line (θ : ℝ) (b : ℝ) (k : ℝ) (y x : ℝ) :
  θ = Real.pi / 4 ∧ b = 2 ∧ k = Real.tan θ ∧ k = 1 ∧ y = k * x + b ↔ y = x + 2 :=
by
  intros
  sorry

end equation_of_line_l154_154311


namespace rocking_chair_legs_l154_154526

theorem rocking_chair_legs :
  let tables_4legs := 4 * 4
  let sofa_4legs := 1 * 4
  let chairs_4legs := 2 * 4
  let tables_3legs := 3 * 3
  let table_1leg := 1 * 1
  let total_legs := 40
  let accounted_legs := tables_4legs + sofa_4legs + chairs_4legs + tables_3legs + table_1leg
  ∃ rocking_chair_legs : Nat, total_legs = accounted_legs + rocking_chair_legs ∧ rocking_chair_legs = 2 :=
sorry

end rocking_chair_legs_l154_154526


namespace geom_seq_div_a5_a7_l154_154980

variable {a : ℕ → ℝ}

-- Given sequence is geometric and positive
def is_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Positive geometric sequence with decreasing terms
def is_positive_decreasing_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  is_geom_sequence a r ∧ ∀ n, a (n + 1) < a n ∧ a n > 0

-- Conditions
variables (r : ℝ) (hp : is_positive_decreasing_geom_sequence a r)
           (h2 : a 2 * a 8 = 6) (h3 : a 4 + a 6 = 5)

-- Goal
theorem geom_seq_div_a5_a7 : a 5 / a 7 = 3 / 2 :=
by
  sorry

end geom_seq_div_a5_a7_l154_154980


namespace deposit_percentage_correct_l154_154951

-- Define the conditions
def deposit_amount : ℕ := 50
def remaining_amount : ℕ := 950
def total_cost : ℕ := deposit_amount + remaining_amount

-- Define the proof problem statement
theorem deposit_percentage_correct :
  (deposit_amount / total_cost : ℚ) * 100 = 5 := 
by
  -- sorry is used to skip the proof
  sorry

end deposit_percentage_correct_l154_154951


namespace quadratic_decreasing_then_increasing_l154_154416

-- Define the given quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 6 * x + 10

-- Define the interval of interest
def interval (x : ℝ) : Prop := 2 < x ∧ x < 4

-- The main theorem to prove: the function is first decreasing on (2, 3] and then increasing on [3, 4)
theorem quadratic_decreasing_then_increasing :
  (∀ (x : ℝ), 2 < x ∧ x ≤ 3 → quadratic_function x > quadratic_function (x + ε) ∧ ε > 0) ∧
  (∀ (x : ℝ), 3 ≤ x ∧ x < 4 → quadratic_function x < quadratic_function (x + ε) ∧ ε > 0) :=
sorry

end quadratic_decreasing_then_increasing_l154_154416


namespace bread_consumption_snacks_per_day_l154_154692

theorem bread_consumption_snacks_per_day (members : ℕ) (breakfast_slices_per_member : ℕ) (slices_per_loaf : ℕ) (loaves : ℕ) (days : ℕ) (total_slices_breakfast : ℕ) (total_slices_all : ℕ) (snack_slices_per_member_per_day : ℕ) :
  members = 4 →
  breakfast_slices_per_member = 3 →
  slices_per_loaf = 12 →
  loaves = 5 →
  days = 3 →
  total_slices_breakfast = members * breakfast_slices_per_member * days →
  total_slices_all = slices_per_loaf * loaves →
  snack_slices_per_member_per_day = ((total_slices_all - total_slices_breakfast) / members / days) →
  snack_slices_per_member_per_day = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- We can insert the proof outline here based on the calculations from the solution steps
  sorry

end bread_consumption_snacks_per_day_l154_154692


namespace average_visitors_in_month_of_30_days_starting_with_sunday_l154_154825

def average_visitors_per_day (sundays_visitors : ℕ) (other_days_visitors : ℕ) (num_sundays : ℕ) (num_other_days : ℕ) : ℕ :=
  (sundays_visitors * num_sundays + other_days_visitors * num_other_days) / (num_sundays + num_other_days)

theorem average_visitors_in_month_of_30_days_starting_with_sunday :
  average_visitors_per_day 1000 700 5 25 = 750 := sorry

end average_visitors_in_month_of_30_days_starting_with_sunday_l154_154825


namespace cos_330_eq_sqrt_3_div_2_l154_154806

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l154_154806


namespace power_of_point_l154_154658

namespace ChordsIntersect

variables (A B C D P : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]

def AP := 4
def CP := 9

theorem power_of_point (BP DP : ℕ) :
  AP * BP = CP * DP -> (BP / DP) = 9 / 4 :=
by
  sorry

end ChordsIntersect

end power_of_point_l154_154658


namespace ramesh_paid_price_l154_154024

variables 
  (P : Real) -- Labelled price of the refrigerator
  (paid_price : Real := 0.80 * P + 125 + 250) -- Price paid after discount and additional costs
  (sell_price : Real := 1.16 * P) -- Price to sell for 16% profit
  (sell_at : Real := 18560) -- Target selling price for given profit

theorem ramesh_paid_price : 
  1.16 * P = 18560 → paid_price = 13175 :=
by
  sorry

end ramesh_paid_price_l154_154024


namespace combined_distance_proof_l154_154570

/-- Define the distances walked by Lionel, Esther, and Niklaus in their respective units -/
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

/-- Define the conversion factors -/
def miles_to_feet : ℕ := 5280
def yards_to_feet : ℕ := 3

/-- The total combined distance in feet -/
def total_distance_feet : ℕ :=
  (lionel_miles * miles_to_feet) + (esther_yards * yards_to_feet) + niklaus_feet

theorem combined_distance_proof : total_distance_feet = 24332 := by
  -- expand definitions and calculations here...
  -- lionel = 4 * 5280 = 21120
  -- esther = 975 * 3 = 2925
  -- niklaus = 1287
  -- sum = 21120 + 2925 + 1287 = 24332
  sorry

end combined_distance_proof_l154_154570


namespace twelve_hens_lay_48_eggs_in_twelve_days_l154_154774

theorem twelve_hens_lay_48_eggs_in_twelve_days :
  (∀ (hens eggs days : ℕ), hens = 3 → eggs = 3 → days = 3 → eggs / (hens * days) = 1/3) → 
  ∀ (hens days : ℕ), hens = 12 → days = 12 → hens * days * (1/3) = 48 :=
by
  sorry

end twelve_hens_lay_48_eggs_in_twelve_days_l154_154774


namespace arithmetic_seq_a8_l154_154791

theorem arithmetic_seq_a8 : ∀ (a : ℕ → ℤ), 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → 
  (a 5 + a 6 = 22) → 
  (a 3 = 7) → 
  a 8 = 15 :=
by
  intros a ha_arithmetic hsum h3
  sorry

end arithmetic_seq_a8_l154_154791


namespace prime_m_l154_154248

theorem prime_m (m : ℕ) (hm : m ≥ 2) :
  (∀ n : ℕ, (m / 3 ≤ n) → (n ≤ m / 2) → (n ∣ Nat.choose n (m - 2 * n))) → Nat.Prime m :=
by
  intro h
  sorry

end prime_m_l154_154248


namespace alice_age_30_l154_154326

variable (A T : ℕ)

def tom_younger_alice (A T : ℕ) := T = A - 15
def ten_years_ago (A T : ℕ) := A - 10 = 4 * (T - 10)

theorem alice_age_30 (A T : ℕ) (h1 : tom_younger_alice A T) (h2 : ten_years_ago A T) : A = 30 := 
by sorry

end alice_age_30_l154_154326


namespace quadratic_rewrite_l154_154081

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 6
noncomputable def c : ℕ := 284
noncomputable def quadratic_coeffs_sum : ℕ := a + b + c

theorem quadratic_rewrite :
  (∃ a b c : ℕ, 6 * (x : ℕ) ^ 2 + 72 * x + 500 = a * (x + b) ^ 2 + c) →
  quadratic_coeffs_sum = 296 := by sorry

end quadratic_rewrite_l154_154081


namespace sandy_spent_correct_amount_l154_154732

-- Definitions
def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def shoes_price : ℝ := 8.50
def accessories_price : ℝ := 10.75
def discount_rate : ℝ := 0.10
def coupon_amount : ℝ := 5.00
def tax_rate : ℝ := 0.075

-- Sum of all items before discounts and coupons
def total_before_discount : ℝ :=
  shorts_price + shirt_price + jacket_price + shoes_price + accessories_price

-- Total after applying the discount
def total_after_discount : ℝ :=
  total_before_discount * (1 - discount_rate)

-- Total after applying the coupon
def total_after_coupon : ℝ :=
  total_after_discount - coupon_amount

-- Total after applying the tax
def total_after_tax : ℝ :=
  total_after_coupon * (1 + tax_rate)

-- Theorem assertion that total amount spent is equal to $45.72
theorem sandy_spent_correct_amount : total_after_tax = 45.72 := by
  sorry

end sandy_spent_correct_amount_l154_154732


namespace alex_and_zhu_probability_l154_154395

theorem alex_and_zhu_probability :
  let num_students := 100
  let num_selected := 60
  let num_sections := 3
  let section_size := 20
  let P_alex_selected := 3 / 5
  let P_zhu_selected_given_alex_selected := 59 / 99
  let P_same_section_given_both_selected := 19 / 59
  (P_alex_selected * P_zhu_selected_given_alex_selected * P_same_section_given_both_selected) = 19 / 165 := 
by {
  sorry
}

end alex_and_zhu_probability_l154_154395


namespace hypotenuse_of_454590_triangle_l154_154775

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l154_154775


namespace total_amount_is_70000_l154_154133

-- Definitions based on the given conditions
def total_amount_divided (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  amount_10 + amount_20

def interest_earned (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  (amount_10 * 10 / 100) + (amount_20 * 20 / 100)

-- Statement to be proved
theorem total_amount_is_70000 (amount_10: ℕ) (amount_20: ℕ) (total_interest: ℕ) :
  amount_10 = 60000 →
  total_interest = 8000 →
  interest_earned amount_10 amount_20 = total_interest →
  total_amount_divided amount_10 amount_20 = 70000 :=
by
  intros h1 h2 h3
  sorry

end total_amount_is_70000_l154_154133


namespace right_triangle_incircle_excircle_condition_l154_154018

theorem right_triangle_incircle_excircle_condition
  (r R : ℝ) 
  (hr_pos : 0 < r) 
  (hR_pos : 0 < R) :
  R ≥ r * (3 + 2 * Real.sqrt 2) := sorry

end right_triangle_incircle_excircle_condition_l154_154018


namespace cars_selected_l154_154768

theorem cars_selected (num_cars num_clients selections_made total_selections : ℕ)
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_made = 2)
  (h4 : total_selections = num_clients * selections_made) :
  num_cars * (total_selections / num_cars) = 48 :=
by
  sorry

end cars_selected_l154_154768


namespace total_nuggets_ordered_l154_154986

noncomputable def Alyssa_nuggets : ℕ := 20
noncomputable def Keely_nuggets : ℕ := 2 * Alyssa_nuggets
noncomputable def Kendall_nuggets : ℕ := 2 * Alyssa_nuggets

theorem total_nuggets_ordered : Alyssa_nuggets + Keely_nuggets + Kendall_nuggets = 100 := by
  sorry -- Proof is intentionally omitted

end total_nuggets_ordered_l154_154986


namespace find_d_l154_154502

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l154_154502


namespace trees_variance_l154_154978

theorem trees_variance :
  let groups := [3, 4, 3]
  let trees := [5, 6, 7]
  let n := 10
  let mean := (5 * 3 + 6 * 4 + 7 * 3) / n
  let variance := (3 * (5 - mean)^2 + 4 * (6 - mean)^2 + 3 * (7 - mean)^2) / n
  variance = 0.6 := 
by
  sorry

end trees_variance_l154_154978


namespace value_of_fraction_l154_154634

theorem value_of_fraction : (20 + 15) / (30 - 25) = 7 := by
  sorry

end value_of_fraction_l154_154634


namespace abs_condition_implies_l154_154971

theorem abs_condition_implies (x : ℝ) 
  (h : |x - 1| < 2) : x < 3 := by
  sorry

end abs_condition_implies_l154_154971


namespace polynomial_remainder_l154_154769

theorem polynomial_remainder :
  ∀ (x : ℝ), (x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 5) % (x^2 - 3 * x + 2) = (24 * x - 25) :=
by
  sorry

end polynomial_remainder_l154_154769


namespace expected_number_of_digits_on_fair_icosahedral_die_l154_154793

noncomputable def expected_digits_fair_icosahedral_die : ℚ :=
  let prob_one_digit := (9 : ℚ) / 20
  let prob_two_digits := (11 : ℚ) / 20
  (prob_one_digit * 1) + (prob_two_digits * 2)

theorem expected_number_of_digits_on_fair_icosahedral_die : expected_digits_fair_icosahedral_die = 1.55 := by
  sorry

end expected_number_of_digits_on_fair_icosahedral_die_l154_154793


namespace sufficient_condition_l154_154312

theorem sufficient_condition (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r :=
by
  intro hp
  apply hqr
  apply hpq
  exact hp

end sufficient_condition_l154_154312


namespace correct_total_cost_l154_154927

-- Number of sandwiches and their cost
def num_sandwiches : ℕ := 7
def sandwich_cost : ℕ := 4

-- Number of sodas and their cost
def num_sodas : ℕ := 9
def soda_cost : ℕ := 3

-- Total cost calculation
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem correct_total_cost : total_cost = 55 := by
  -- skip the proof details
  sorry

end correct_total_cost_l154_154927


namespace parabola_vertex_correct_l154_154990

noncomputable def parabola_vertex (p q : ℝ) : ℝ × ℝ :=
  let a := -1
  let b := p
  let c := q
  let x_vertex := -b / (2 * a)
  let y_vertex := a * x_vertex^2 + b * x_vertex + c
  (x_vertex, y_vertex)

theorem parabola_vertex_correct (p q : ℝ) :
  (parabola_vertex 2 24 = (1, 25)) :=
  sorry

end parabola_vertex_correct_l154_154990


namespace electricity_cost_one_kilometer_minimum_electricity_kilometers_l154_154166

-- Part 1: Cost of traveling one kilometer using electricity only
theorem electricity_cost_one_kilometer (x : ℝ) (fuel_cost : ℝ) (electricity_cost : ℝ) 
  (total_fuel_cost : ℝ) (total_electricity_cost : ℝ) 
  (fuel_per_km_more_than_electricity : ℝ) (distance_fuel : ℝ) (distance_electricity : ℝ)
  (h1 : total_fuel_cost = distance_fuel * fuel_cost)
  (h2 : total_electricity_cost = distance_electricity * electricity_cost)
  (h3 : fuel_per_km_more_than_electricity = 0.5)
  (h4 : fuel_cost = electricity_cost + fuel_per_km_more_than_electricity)
  (h5 : distance_fuel = 76 / (electricity_cost + 0.5))
  (h6 : distance_electricity = 26 / electricity_cost) : 
  x = 0.26 :=
sorry

-- Part 2: Minimum kilometers traveled using electricity
theorem minimum_electricity_kilometers (total_trip_cost : ℝ) (electricity_per_km : ℝ) 
  (hybrid_total_km : ℝ) (max_total_cost : ℝ) (fuel_per_km : ℝ) (y : ℝ)
  (h1 : electricity_per_km = 0.26)
  (h2 : fuel_per_km = 0.26 + 0.5)
  (h3 : hybrid_total_km = 100)
  (h4 : max_total_cost = 39)
  (h5 : total_trip_cost = electricity_per_km * y + (hybrid_total_km - y) * fuel_per_km)
  (h6 : total_trip_cost ≤ max_total_cost) :
  y ≥ 74 :=
sorry

end electricity_cost_one_kilometer_minimum_electricity_kilometers_l154_154166


namespace problem_statement_l154_154333

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Definition of the geometric sequence {b_n}
def b (n : ℕ) : ℕ := 2^n

-- Definition of the sequence {c_n}
def c (n : ℕ) : ℕ := a n + b n

-- Sum of first n terms of the sequence {c_n}
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 2^(n + 1) - 2

-- Prove the problem statement
theorem problem_statement :
  (a 1 + a 2 = 3) ∧
  (a 4 - a 3 = 1) ∧
  (b 2 = a 4) ∧
  (b 3 = a 8) ∧
  (∀ n : ℕ, c n = a n + b n) ∧
  (∀ n : ℕ, S n = (n * (n + 1)) / 2 + 2^(n + 1) - 2) :=
by {
  sorry -- Proof goes here
}

end problem_statement_l154_154333


namespace jordan_length_eq_six_l154_154389

def carol_length := 12
def carol_width := 15
def jordan_width := 30

theorem jordan_length_eq_six
  (h1 : carol_length * carol_width = jordan_width * jordan_length) : 
  jordan_length = 6 := by
  sorry

end jordan_length_eq_six_l154_154389


namespace solve_equation_3x6_eq_3mx_div_xm1_l154_154176

theorem solve_equation_3x6_eq_3mx_div_xm1 (x : ℝ) 
  (h1 : x ≠ 1)
  (h2 : x^2 + 5*x - 6 ≠ 0) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ (x = 3 ∨ x = -6) :=
by 
  sorry

end solve_equation_3x6_eq_3mx_div_xm1_l154_154176


namespace julie_aaron_age_l154_154124

variables {J A m : ℕ}

theorem julie_aaron_age : (J = 4 * A) → (J + 10 = m * (A + 10)) → (m = 4) :=
by
  intros h1 h2
  sorry

end julie_aaron_age_l154_154124


namespace age_difference_64_l154_154561

variables (Patrick Michael Monica : ℕ)
axiom age_ratio_1 : ∃ (x : ℕ), Patrick = 3 * x ∧ Michael = 5 * x
axiom age_ratio_2 : ∃ (y : ℕ), Michael = 3 * y ∧ Monica = 5 * y
axiom age_sum : Patrick + Michael + Monica = 196

theorem age_difference_64 : Monica - Patrick = 64 :=
by {
  sorry
}

end age_difference_64_l154_154561


namespace find_part_of_number_l154_154264

theorem find_part_of_number (x y : ℕ) (h₁ : x = 1925) (h₂ : x / 7 = y + 100) : y = 175 :=
sorry

end find_part_of_number_l154_154264


namespace solve_system_1_solve_system_2_solve_system_3_solve_system_4_l154_154146

-- System 1
theorem solve_system_1 (x y : ℝ) (h1 : x = y + 1) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
  sorry

-- System 2
theorem solve_system_2 (x y : ℝ) (h1 : 3 * x + y = 8) (h2 : x - y = 4) : x = 3 ∧ y = -1 :=
by
  sorry

-- System 3
theorem solve_system_3 (x y : ℝ) (h1 : 5 * x + 3 * y = 2) (h2 : 3 * x + 2 * y = 1) : x = 1 ∧ y = -1 :=
by
  sorry

-- System 4
theorem solve_system_4 (x y z : ℝ) (h1 : x + y = 3) (h2 : y + z = -2) (h3 : z + x = 9) : x = 7 ∧ y = -4 ∧ z = 2 :=
by
  sorry

end solve_system_1_solve_system_2_solve_system_3_solve_system_4_l154_154146


namespace minimum_cost_of_candies_l154_154887

variable (Orange Apple Grape Strawberry : ℕ)

-- Conditions
def CandyRelation1 := Apple = 2 * Orange
def CandyRelation2 := Strawberry = 2 * Grape
def CandyRelation3 := Apple = 2 * Strawberry
def TotalCandies := Orange + Apple + Grape + Strawberry = 90
def CandyCost := 0.1

-- Question
theorem minimum_cost_of_candies :
  CandyRelation1 Orange Apple → 
  CandyRelation2 Grape Strawberry → 
  CandyRelation3 Apple Strawberry → 
  TotalCandies Orange Apple Grape Strawberry → 
  Orange ≥ 3 ∧ Apple ≥ 3 ∧ Grape ≥ 3 ∧ Strawberry ≥ 3 →
  (5 * CandyCost + 3 * CandyCost + 3 * CandyCost + 3 * CandyCost = 1.4) :=
sorry

end minimum_cost_of_candies_l154_154887


namespace stadium_surface_area_correct_l154_154347

noncomputable def stadium_length_yards : ℝ := 62
noncomputable def stadium_width_yards : ℝ := 48
noncomputable def stadium_height_yards : ℝ := 30

noncomputable def stadium_length_feet : ℝ := stadium_length_yards * 3
noncomputable def stadium_width_feet : ℝ := stadium_width_yards * 3
noncomputable def stadium_height_feet : ℝ := stadium_height_yards * 3

def total_surface_area_stadium (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

theorem stadium_surface_area_correct :
  total_surface_area_stadium stadium_length_feet stadium_width_feet stadium_height_feet = 110968 := by
  sorry

end stadium_surface_area_correct_l154_154347


namespace arithmetic_problem_l154_154651

theorem arithmetic_problem : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end arithmetic_problem_l154_154651


namespace pentagon_area_inequality_l154_154092

-- Definitions for the problem
structure Point :=
(x y : ℝ)

structure Triangle :=
(A B C : Point)

noncomputable def area (T : Triangle) : ℝ :=
  1 / 2 * abs ((T.B.x - T.A.x) * (T.C.y - T.A.y) - (T.C.x - T.A.x) * (T.B.y - T.A.y))

structure Pentagon :=
(A B C D E : Point)

noncomputable def pentagon_area (P : Pentagon) : ℝ :=
  area ⟨P.A, P.B, P.C⟩ + area ⟨P.A, P.C, P.D⟩ + area ⟨P.A, P.D, P.E⟩ -
  area ⟨P.E, P.B, P.C⟩

-- Given conditions
variables (A B C D E F : Point)
variables (P : Pentagon) 
-- P is a convex pentagon with points A, B, C, D, E in order 

-- Intersection point of AD and EC is F 
axiom intersect_diagonals (AD EC : Triangle) : AD.C = F ∧ EC.B = F

-- Theorem statement
theorem pentagon_area_inequality :
  let AED := Triangle.mk A E D
  let EDC := Triangle.mk E D C
  let EAB := Triangle.mk E A B
  let DCB := Triangle.mk D C B
  area AED + area EDC + area EAB + area DCB > pentagon_area P :=
  sorry

end pentagon_area_inequality_l154_154092


namespace gcd_2183_1947_l154_154361

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := 
by 
  sorry

end gcd_2183_1947_l154_154361


namespace total_cost_is_eight_x_l154_154177

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l154_154177


namespace how_many_years_later_will_tom_be_twice_tim_l154_154512

-- Conditions
def toms_age := 15
def total_age := 21
def tims_age := total_age - toms_age

-- Define the problem statement
theorem how_many_years_later_will_tom_be_twice_tim (x : ℕ) 
  (h1 : toms_age + tims_age = total_age) 
  (h2 : toms_age = 15) 
  (h3 : ∀ y : ℕ, toms_age + y = 2 * (tims_age + y) ↔ y = x) : 
  x = 3 
:= sorry

end how_many_years_later_will_tom_be_twice_tim_l154_154512


namespace division_remainder_l154_154323

theorem division_remainder (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (hrem : x % y = 3) (hdiv : (x : ℚ) / y = 96.15) : y = 20 :=
sorry

end division_remainder_l154_154323


namespace gcd_of_three_numbers_l154_154584

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end gcd_of_three_numbers_l154_154584


namespace ellipse_standard_equation_l154_154474

theorem ellipse_standard_equation
  (a b c : ℝ)
  (h1 : (3 * a) / (-a) + 16 / b = 1)
  (h2 : (3 * a) / c + 16 / (-b) = 1)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : a > b)
  (h6 : a^2 = b^2 + c^2) : 
  (a = 5 ∧ b = 4 ∧ c = 3) ∧ (∀ x y, x^2 / 25 + y^2 / 16 = 1 ↔ (a = 5 ∧ b = 4)) := 
sorry

end ellipse_standard_equation_l154_154474


namespace viable_combinations_l154_154273

-- Given conditions
def totalHerbs : Nat := 4
def totalCrystals : Nat := 6
def incompatibleComb1 : Nat := 2
def incompatibleComb2 : Nat := 1

-- Theorem statement proving the number of viable combinations
theorem viable_combinations : totalHerbs * totalCrystals - (incompatibleComb1 + incompatibleComb2) = 21 := by
  sorry

end viable_combinations_l154_154273


namespace find_number_of_shorts_l154_154447

def price_of_shorts : ℕ := 7
def price_of_shoes : ℕ := 20
def total_spent : ℕ := 75

-- We represent the price of 4 tops as a variable
variable (T : ℕ)

theorem find_number_of_shorts (S : ℕ) (h : 7 * S + 4 * T + 20 = 75) : S = 7 :=
by
  sorry

end find_number_of_shorts_l154_154447


namespace circle_length_l154_154102

theorem circle_length (n : ℕ) (arm_span : ℝ) (overlap : ℝ) (contribution : ℝ) (total_length : ℝ) :
  n = 16 ->
  arm_span = 10.4 ->
  overlap = 3.5 ->
  contribution = arm_span - overlap ->
  total_length = n * contribution ->
  total_length = 110.4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end circle_length_l154_154102


namespace trig_identity_l154_154105

variable (α : Real)
variable (h : Real.tan α = 2)

theorem trig_identity :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := by
  sorry

end trig_identity_l154_154105


namespace calculate_total_difference_in_miles_l154_154659

def miles_bus_a : ℝ := 1.25
def miles_walk_1 : ℝ := 0.35
def miles_bus_b : ℝ := 2.68
def miles_walk_2 : ℝ := 0.47
def miles_bus_c : ℝ := 3.27
def miles_walk_3 : ℝ := 0.21

def total_miles_on_buses : ℝ := miles_bus_a + miles_bus_b + miles_bus_c
def total_miles_walked : ℝ := miles_walk_1 + miles_walk_2 + miles_walk_3
def total_difference_in_miles : ℝ := total_miles_on_buses - total_miles_walked

theorem calculate_total_difference_in_miles :
  total_difference_in_miles = 6.17 := by
  sorry

end calculate_total_difference_in_miles_l154_154659


namespace solution_set_of_inequality_l154_154552

variable (f : ℝ → ℝ)

def g (x : ℝ) : ℝ := f x - x - 1

theorem solution_set_of_inequality (h₁ : f 1 = 2) (h₂ : ∀ x, (deriv f x) < 1) :
  { x : ℝ | f x < x + 1 } = { x | 1 < x } :=
by
  sorry

end solution_set_of_inequality_l154_154552


namespace tan_arithmetic_sequence_l154_154236

theorem tan_arithmetic_sequence {a : ℕ → ℝ}
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + n * d)
  (h_sum : a 1 + a 7 + a 13 = Real.pi) :
  Real.tan (a 2 + a 12) = - Real.sqrt 3 :=
sorry

end tan_arithmetic_sequence_l154_154236


namespace greatest_three_digit_multiple_of_17_l154_154899

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l154_154899


namespace delivery_in_april_l154_154008

theorem delivery_in_april (n_jan n_mar : ℕ) (growth_rate : ℝ) :
  n_jan = 100000 → n_mar = 121000 → (1 + growth_rate) ^ 2 = n_mar / n_jan →
  (n_mar * (1 + growth_rate) = 133100) :=
by
  intros n_jan_eq n_mar_eq growth_eq
  sorry

end delivery_in_april_l154_154008


namespace find_prime_pairs_l154_154782

open Nat

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def valid_prime_pairs (p q : ℕ): Prop :=
  is_prime p ∧ is_prime q ∧ divides p (30 * q - 1) ∧ divides q (30 * p - 1)

theorem find_prime_pairs :
  { (p, q) | valid_prime_pairs p q } = { (7, 11), (11, 7), (59, 61), (61, 59) } :=
sorry

end find_prime_pairs_l154_154782


namespace sam_earnings_difference_l154_154770

def hours_per_dollar := 1 / 10  -- Sam earns $10 per hour, so it takes 1/10 hour per dollar earned.

theorem sam_earnings_difference
  (hours_per_dollar : ℝ := 1 / 10)
  (E1 : ℝ := 200)  -- Earnings in the first month are $200.
  (total_hours : ℝ := 55)  -- Total hours he worked over two months.
  (total_hourly_earning : ℝ := total_hours / hours_per_dollar)  -- Total earnings over two months.
  (E2 : ℝ := total_hourly_earning - E1) :  -- Earnings in the second month.

  E2 - E1 = 150 :=  -- The difference in earnings between the second month and the first month is $150.
sorry

end sam_earnings_difference_l154_154770


namespace unique_shape_determination_l154_154957

theorem unique_shape_determination (ratio_sides_median : Prop) (ratios_three_sides : Prop) 
                                   (ratio_circumradius_side : Prop) (ratio_two_angles : Prop) 
                                   (length_one_side_heights : Prop) :
  ¬(ratio_circumradius_side → (ratio_sides_median ∧ ratios_three_sides ∧ ratio_two_angles ∧ length_one_side_heights)) := 
sorry

end unique_shape_determination_l154_154957


namespace p_n_div_5_iff_not_mod_4_zero_l154_154720

theorem p_n_div_5_iff_not_mod_4_zero (n : ℕ) (h : 0 < n) : 
  (1 + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
by {
  sorry
}

end p_n_div_5_iff_not_mod_4_zero_l154_154720


namespace exists_sequences_satisfying_conditions_l154_154094

noncomputable def satisfies_conditions (n : ℕ) (hn : Odd n) 
  (a : Fin n → ℕ) (b : Fin n → ℕ) : Prop :=
  ∀ (k : Fin n), 0 < k.val → k.val < n →
    ∀ (i : Fin n),
      let in3n := 3 * n;
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n ≠
      (a i + b i) % in3n ∧
      (a i + b i) % in3n ≠
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ∧
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ≠
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n

theorem exists_sequences_satisfying_conditions :
  ∀ n : ℕ, Odd n → ∃ (a : Fin n → ℕ) (b : Fin n → ℕ),
    satisfies_conditions n sorry a b :=
sorry

end exists_sequences_satisfying_conditions_l154_154094


namespace question_1_question_2_question_3_l154_154501

def deck_size : Nat := 32

theorem question_1 :
  let hands_when_order_matters := deck_size * (deck_size - 1)
  hands_when_order_matters = 992 :=
by
  let hands_when_order_matters := deck_size * (deck_size - 1)
  sorry

theorem question_2 :
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  hands_when_order_does_not_matter = 496 :=
by
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  sorry

theorem question_3 :
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  hands_3_cards_order_does_not_matter = 4960 :=
by
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  sorry

end question_1_question_2_question_3_l154_154501


namespace problem_solution_includes_024_l154_154214

theorem problem_solution_includes_024 (x : ℝ) :
  (2 * 88 * (abs (abs (abs (abs (x - 1) - 1) - 1) - 1)) = 0) →
  x = 0 ∨ x = 2 ∨ x = 4 :=
by
  sorry

end problem_solution_includes_024_l154_154214


namespace fraction_of_jumbo_tiles_l154_154448

-- Definitions for conditions
variables (L W : ℝ) -- Length and width of regular tiles
variables (n : ℕ) -- Number of regular tiles
variables (m : ℕ) -- Number of jumbo tiles

-- Conditions
def condition1 : Prop := (n : ℝ) * (L * W) = 40 -- Regular tiles cover 40 square feet
def condition2 : Prop := (n : ℝ) * (L * W) + (m : ℝ) * (3 * L * W) = 220 -- Entire wall is 220 square feet
def condition3 : Prop := ∃ (k : ℝ), (m : ℝ) = k * (n : ℝ) ∧ k = 1.5 -- Relationship ratio between jumbo and regular tiles

-- Theorem to be proved
theorem fraction_of_jumbo_tiles (L W : ℝ) (n m : ℕ)
  (h1 : condition1 L W n)
  (h2 : condition2 L W n m)
  (h3 : condition3 n m) :
  (m : ℝ) / ((n : ℝ) + (m : ℝ)) = 3 / 5 :=
sorry

end fraction_of_jumbo_tiles_l154_154448


namespace min_value_of_quadratic_l154_154144

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end min_value_of_quadratic_l154_154144


namespace problem_1956_Tokyo_Tech_l154_154263

theorem problem_1956_Tokyo_Tech (a b c : ℝ) (ha : 0 < a) (ha_lt_one : a < 1) (hb : 0 < b) 
(hb_lt_one : b < 1) (hc : 0 < c) (hc_lt_one : c < 1) : a + b + c - a * b * c < 2 := 
sorry

end problem_1956_Tokyo_Tech_l154_154263


namespace rational_solution_counts_l154_154779

theorem rational_solution_counts :
  (∃ (x y : ℚ), x^2 + y^2 = 2) ∧ 
  (¬ ∃ (x y : ℚ), x^2 + y^2 = 3) := 
by 
  sorry

end rational_solution_counts_l154_154779


namespace cos_7theta_l154_154457

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -45682/8192 :=
by
  sorry

end cos_7theta_l154_154457


namespace different_meal_combinations_l154_154787

-- Defining the conditions explicitly
def items_on_menu : ℕ := 12

-- A function representing possible combinations of choices for Yann and Camille
def meal_combinations (menu_items : ℕ) : ℕ :=
  menu_items * (menu_items - 1)

-- Theorem stating that given 12 items on the menu, the different combinations of meals is 132
theorem different_meal_combinations : meal_combinations items_on_menu = 132 :=
by
  sorry

end different_meal_combinations_l154_154787


namespace compare_xyz_l154_154811

theorem compare_xyz
  (a b c d : ℝ) (h : a < b ∧ b < c ∧ c < d)
  (x : ℝ) (hx : x = (a + b) * (c + d))
  (y : ℝ) (hy : y = (a + c) * (b + d))
  (z : ℝ) (hz : z = (a + d) * (b + c)) :
  x < y ∧ y < z :=
by sorry

end compare_xyz_l154_154811


namespace reciprocal_opposites_l154_154551

theorem reciprocal_opposites (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end reciprocal_opposites_l154_154551


namespace sum_prime_factors_77_l154_154042

theorem sum_prime_factors_77 : ∀ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 → p1 + p2 = 18 :=
by
  intros p1 p2 h
  sorry

end sum_prime_factors_77_l154_154042


namespace ratio_of_boys_to_girls_l154_154636

theorem ratio_of_boys_to_girls (G B : ℕ) (hg : G = 30) (hb : B = G + 18) : B / G = 8 / 5 :=
by
  sorry

end ratio_of_boys_to_girls_l154_154636


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l154_154554

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l154_154554


namespace exists_k_divides_poly_l154_154097

theorem exists_k_divides_poly (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : a 2 = 1) 
  (h₂ : ∀ k : ℕ, a (k + 2) = a (k + 1) + a k) :
  ∀ (m : ℕ), m > 0 → ∃ k : ℕ, m ∣ (a k ^ 4 - a k - 2) :=
by
  sorry

end exists_k_divides_poly_l154_154097


namespace repeating_decimals_subtraction_l154_154475

/--
Calculate the value of 0.\overline{234} - 0.\overline{567} - 0.\overline{891}.
Express your answer as a fraction in its simplest form.

Shown that:
Let x = 0.\overline{234}, y = 0.\overline{567}, z = 0.\overline{891},
Then 0.\overline{234} - 0.\overline{567} - 0.\overline{891} = -1224/999
-/
theorem repeating_decimals_subtraction : 
  let x : ℚ := 234 / 999
  let y : ℚ := 567 / 999
  let z : ℚ := 891 / 999
  x - y - z = -1224 / 999 := 
by
  sorry

end repeating_decimals_subtraction_l154_154475


namespace original_ratio_of_boarders_to_day_students_l154_154027

theorem original_ratio_of_boarders_to_day_students
    (original_boarders : ℕ)
    (new_boarders : ℕ)
    (new_ratio_b_d : ℕ → ℕ)
    (no_switch : Prop)
    (no_leave : Prop)
  : (original_boarders = 220) ∧ (new_boarders = 44) ∧ (new_ratio_b_d 1 = 2) ∧ no_switch ∧ no_leave →
  ∃ (original_day_students : ℕ), original_day_students = 528 ∧ (220 / 44 = 5) ∧ (528 / 44 = 12)
  := by
    sorry

end original_ratio_of_boarders_to_day_students_l154_154027


namespace problem_statement_l154_154845

theorem problem_statement (x y : ℕ) (hx : x = 7) (hy : y = 3) : (x - y)^2 * (x + y)^2 = 1600 :=
by
  rw [hx, hy]
  sorry

end problem_statement_l154_154845


namespace members_playing_both_l154_154138

variable (N B T Neither BT : ℕ)

theorem members_playing_both (hN : N = 30) (hB : B = 17) (hT : T = 17) (hNeither : Neither = 2) 
  (hBT : BT = B + T - (N - Neither)) : BT = 6 := 
by 
  rw [hN, hB, hT, hNeither] at hBT
  exact hBT

end members_playing_both_l154_154138


namespace sum_reciprocals_l154_154003

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end sum_reciprocals_l154_154003


namespace simplify_expr_l154_154807

theorem simplify_expr : 2 - 2 / (1 + Real.sqrt 2) - 2 / (1 - Real.sqrt 2) = -2 := by
  sorry

end simplify_expr_l154_154807


namespace sum_consecutive_evens_l154_154374

theorem sum_consecutive_evens (n k : ℕ) (hn : 2 < n) (hk : 2 < k) : 
  ∃ (m : ℕ), n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) :=
by
  sorry

end sum_consecutive_evens_l154_154374


namespace range_of_c_l154_154285

theorem range_of_c :
  (∃ (c : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 4) → ((12 * x - 5 * y + c) / 13 = 1))
  → (c > -13 ∧ c < 13) := 
sorry

end range_of_c_l154_154285


namespace hall_width_l154_154349

theorem hall_width 
  (L H cost total_expenditure : ℕ)
  (W : ℕ)
  (h1 : L = 20)
  (h2 : H = 5)
  (h3 : cost = 20)
  (h4 : total_expenditure = 19000)
  (h5 : total_expenditure = (L * W + 2 * (H * L) + 2 * (H * W)) * cost) :
  W = 25 := 
sorry

end hall_width_l154_154349


namespace ellipse_foci_coordinates_l154_154734

theorem ellipse_foci_coordinates (x y : ℝ) :
  2 * x^2 + 3 * y^2 = 1 →
  (∃ c : ℝ, (c = (Real.sqrt 6) / 6) ∧ ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
by
  sorry

end ellipse_foci_coordinates_l154_154734


namespace part_1_part_2_l154_154707

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_1 (x : ℝ) : f x ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

theorem part_2 (b : ℝ) (h₁ : b ≠ 0) (x : ℝ) (h₂ : f x ≥ (|2 * b + 1| + |1 - b|) / |b|) : x ≤ -1.5 :=
by sorry

end part_1_part_2_l154_154707


namespace solve_inequality_l154_154066

theorem solve_inequality (x : ℝ) : 
  (3 * x - 6 > 12 - 2 * x + x^2) ↔ (-1 < x ∧ x < 6) :=
sorry

end solve_inequality_l154_154066


namespace count_valid_c_l154_154480

theorem count_valid_c : ∃ (count : ℕ), count = 670 ∧ 
  ∀ (c : ℤ), (-2007 ≤ c ∧ c ≤ 2007) → 
    (∃ (x : ℤ), (x^2 + c) % (2^2007) = 0) ↔ count = 670 :=
sorry

end count_valid_c_l154_154480


namespace find_initial_population_l154_154408

noncomputable def population_first_year (P : ℝ) : ℝ :=
  let P1 := 0.90 * P    -- population after 1st year
  let P2 := 0.99 * P    -- population after 2nd year
  let P3 := 0.891 * P   -- population after 3rd year
  P3

theorem find_initial_population (h : population_first_year P = 4455) : P = 4455 / 0.891 :=
by
  sorry

end find_initial_population_l154_154408


namespace isosceles_triangle_vertex_angle_l154_154216

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (hABC : A + B + C = 180) (h_iso : A = B ∨ B = C ∨ A = C) (h_angle : A = 50 ∨ B = 50 ∨ C = 50) : (A = 50 ∨ A = 80) ∨ (B = 50 ∨ B = 80) ∨ (C = 50 ∨ C = 80) :=
by sorry

end isosceles_triangle_vertex_angle_l154_154216


namespace santana_brothers_l154_154992

theorem santana_brothers (b : ℕ) (x : ℕ) (h1 : x + b = 7) (h2 : 3 + 8 = x + 1 + 2 + 7) : x = 1 :=
by
  -- Providing the necessary definitions and conditions
  let brothers := 7 -- Santana has 7 brothers
  let march_birthday := 3 -- 3 brothers have birthdays in March
  let november_birthday := 1 -- 1 brother has a birthday in November
  let december_birthday := 2 -- 2 brothers have birthdays in December
  let total_presents_first_half := 3 -- Total presents in the first half of the year is 3 (March)
  let x := x -- Number of brothers with birthdays in October to be proved
  let total_presents_second_half := x + 1 + 2 + 7 -- Total presents in the second half of the year
  have h3 : total_presents_first_half + 8 = total_presents_second_half := h2 -- Condition equation
  
  -- Start solving the proof
  sorry

end santana_brothers_l154_154992


namespace gardening_project_cost_l154_154241

noncomputable def totalCost : Nat :=
  let roseBushes := 20
  let costPerRoseBush := 150
  let gardenerHourlyRate := 30
  let gardenerHoursPerDay := 5
  let gardenerDays := 4
  let soilCubicFeet := 100
  let soilCostPerCubicFoot := 5

  let costOfRoseBushes := costPerRoseBush * roseBushes
  let gardenerTotalHours := gardenerDays * gardenerHoursPerDay
  let costOfGardener := gardenerHourlyRate * gardenerTotalHours
  let costOfSoil := soilCostPerCubicFoot * soilCubicFeet

  costOfRoseBushes + costOfGardener + costOfSoil

theorem gardening_project_cost : totalCost = 4100 := by
  sorry

end gardening_project_cost_l154_154241


namespace find_value_l154_154767

variable {a b : ℝ}

theorem find_value (h : 2 * a + b + 1 = 0) : 1 + 4 * a + 2 * b = -1 := 
by
  sorry

end find_value_l154_154767


namespace saved_per_bagel_l154_154712

-- Definitions of the conditions
def bagel_cost_each : ℝ := 3.50
def dozen_cost : ℝ := 38
def bakers_dozen : ℕ := 13
def discount : ℝ := 0.05

-- The conjecture we need to prove
theorem saved_per_bagel : 
  let total_cost_without_discount := dozen_cost + bagel_cost_each
  let discount_amount := discount * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount_amount
  let cost_per_bagel_without_discount := dozen_cost / 12
  let cost_per_bagel_with_discount := total_cost_with_discount / bakers_dozen
  let savings_per_bagel := cost_per_bagel_without_discount - cost_per_bagel_with_discount
  let savings_in_cents := savings_per_bagel * 100
  savings_in_cents = 13.36 :=
by
  -- Placeholder for the actual proof
  sorry

end saved_per_bagel_l154_154712


namespace pow_mod_remainder_l154_154120

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end pow_mod_remainder_l154_154120


namespace number_of_tricycles_l154_154359

def num_bicycles : Nat := 24
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3
def total_wheels : Nat := 90

theorem number_of_tricycles : ∃ T : Nat, (wheels_per_bicycle * num_bicycles) + (wheels_per_tricycle * T) = total_wheels ∧ T = 14 := by
  sorry

end number_of_tricycles_l154_154359


namespace find_functions_l154_154129

theorem find_functions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2) :
  (∀ x, f x = (x^2) / 2002) ∨ (∀ x, f x = (x^2) / 2002 + 2 * x + 2002) :=
sorry

end find_functions_l154_154129


namespace part1_q1_l154_154960

open Set Real

def A (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def U : Set ℝ := univ

theorem part1_q1 (m : ℝ) (h : m = -1) : 
  A m ∪ B = {x | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

end part1_q1_l154_154960


namespace math_problem_l154_154220

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def a_n (n : ℕ) : ℕ := 3 * n - 5

theorem math_problem (C5_4 : ℕ) (C6_4 : ℕ) (C7_4 : ℕ) :
  C5_4 = binomial 5 4 →
  C6_4 = binomial 6 4 →
  C7_4 = binomial 7 4 →
  C5_4 + C6_4 + C7_4 = 55 →
  ∃ n : ℕ, a_n n = 55 ∧ n = 20 :=
by
  sorry

end math_problem_l154_154220


namespace lcm_even_numbers_between_14_and_21_l154_154279

-- Define the even numbers between 14 and 21
def evenNumbers := [14, 16, 18, 20]

-- Define a function to compute the LCM of a list of integers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Theorem statement: the LCM of the even numbers between 14 and 21 equals 5040
theorem lcm_even_numbers_between_14_and_21 :
  lcm_list evenNumbers = 5040 :=
by
  sorry

end lcm_even_numbers_between_14_and_21_l154_154279


namespace arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l154_154797

-- Arithmetic Progression
theorem arithmetic_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 - x2 = x2 - x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = (2 * a^3 + 27 * c) / (9 * a)) :=
sorry

-- Geometric Progression
theorem geometric_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x2 / x1 = x3 / x2 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = a * c^(1/3)) :=
sorry

-- Harmonic Sequence
theorem harmonic_sequence_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x1 - x2) / (x2 - x3) = x1 / x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (a = (2 * b^3 + 27 * c) / (9 * b^2)) :=
sorry

end arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l154_154797


namespace train_length_is_correct_l154_154336

noncomputable def speed_of_train_kmph : ℝ := 77.993280537557

noncomputable def speed_of_man_kmph : ℝ := 6

noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * conversion_factor

noncomputable def speed_of_man_mps : ℝ := speed_of_man_kmph * conversion_factor

noncomputable def relative_speed : ℝ := speed_of_train_mps + speed_of_man_mps

noncomputable def time_to_pass_man : ℝ := 6

noncomputable def length_of_train : ℝ := relative_speed * time_to_pass_man

theorem train_length_is_correct : length_of_train = 139.99 := by
  sorry

end train_length_is_correct_l154_154336


namespace combined_area_ratio_l154_154611

theorem combined_area_ratio (s : ℝ) (h₁ : s > 0) : 
  let r := s / 2
  let area_semicircle := (1/2) * π * r^2
  let area_quarter_circle := (1/4) * π * r^2
  let area_square := s^2
  let combined_area := area_semicircle + area_quarter_circle
  let ratio := combined_area / area_square
  ratio = 3 * π / 16 :=
by
  sorry

end combined_area_ratio_l154_154611


namespace absolute_value_inequality_range_of_xyz_l154_154650

-- Question 1 restated
theorem absolute_value_inequality (x : ℝ) :
  (|x + 2| + |x + 3| ≤ 2) ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

-- Question 2 restated
theorem range_of_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  -1/2 ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ 1 :=
sorry

end absolute_value_inequality_range_of_xyz_l154_154650


namespace cost_of_gravelling_the_path_l154_154372

-- Define the problem conditions
def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.70

-- Define the dimensions of the grassy area without the path
def grassy_length : ℝ := plot_length - 2 * path_width
def grassy_width : ℝ := plot_width - 2 * path_width

-- Define the area of the entire plot and the grassy area without the path
def area_entire_plot : ℝ := plot_length * plot_width
def area_grassy_area : ℝ := grassy_length * grassy_width

-- Define the area of the path
def area_path : ℝ := area_entire_plot - area_grassy_area

-- Define the cost of gravelling the path
def cost_gravelling_path : ℝ := area_path * cost_per_sq_meter

-- State the theorem
theorem cost_of_gravelling_the_path : cost_gravelling_path = 595 := 
by
  -- The proof is omitted
  sorry

end cost_of_gravelling_the_path_l154_154372


namespace find_a_l154_154005

noncomputable def f (a x : ℝ) := 3*x^3 - 9*x + a
noncomputable def f' (x : ℝ) : ℝ := 9*x^2 - 9

theorem find_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :
  a = 6 ∨ a = -6 :=
by sorry

end find_a_l154_154005


namespace g_at_pi_over_4_l154_154471

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem g_at_pi_over_4 : g (Real.pi / 4) = 3 / 2 :=
by 
  sorry

end g_at_pi_over_4_l154_154471


namespace difference_of_extremes_l154_154631

def digits : List ℕ := [2, 0, 1, 3]

def largest_integer : ℕ := 3210
def smallest_integer_greater_than_1000 : ℕ := 1023
def expected_difference : ℕ := 2187

theorem difference_of_extremes :
  largest_integer - smallest_integer_greater_than_1000 = expected_difference := by
  sorry

end difference_of_extremes_l154_154631


namespace range_of_m_l154_154585

namespace ProofProblem

-- Define propositions P and Q in Lean
def P (m : ℝ) : Prop := 2 * m > 1
def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Assumptions
variables (m : ℝ)
axiom hP_or_Q : P m ∨ Q m
axiom hP_and_Q_false : ¬(P m ∧ Q m)

-- We need to prove the range of m
theorem range_of_m : m ∈ (Set.Icc (-2 : ℝ) (1 / 2 : ℝ) ∪ Set.Ioi (2 : ℝ)) :=
sorry

end ProofProblem

end range_of_m_l154_154585


namespace exponent_rule_l154_154463

variable (a : ℝ) (m n : ℕ)

theorem exponent_rule (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_rule_l154_154463


namespace compute_volume_of_cube_l154_154075

-- Define the conditions and required properties
variable (s V : ℝ)

-- Given condition: the surface area of the cube is 384 sq cm
def surface_area (s : ℝ) : Prop := 6 * s^2 = 384

-- Define the volume of the cube
def volume (s : ℝ) (V : ℝ) : Prop := V = s^3

-- Theorem statement to prove the volume is correctly computed
theorem compute_volume_of_cube (h₁ : surface_area s) : volume s 512 :=
  sorry

end compute_volume_of_cube_l154_154075


namespace ratio_expression_l154_154047

theorem ratio_expression (p q s u : ℚ) (h1 : p / q = 3 / 5) (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 :=
by
  sorry

end ratio_expression_l154_154047


namespace distance_of_point_P_to_origin_l154_154496

noncomputable def dist_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

theorem distance_of_point_P_to_origin :
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let y_P := 1 / 2
  ∃ x_P : ℝ, (x_P, y_P) = P ∧
    (dist_to_origin P = Real.sqrt 6 / 2) :=
by
  sorry

end distance_of_point_P_to_origin_l154_154496


namespace base6_divisibility_13_l154_154182

theorem base6_divisibility_13 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 5) : (435 + 42 * d) % 13 = 0 ↔ d = 5 :=
by sorry

end base6_divisibility_13_l154_154182


namespace cannot_obtain_fraction_3_5_l154_154168

theorem cannot_obtain_fraction_3_5 (n k : ℕ) :
  ¬ ∃ (a b : ℕ), (a = 5 + k ∧ b = 8 + k ∨ (∃ m : ℕ, a = m * 5 ∧ b = m * 8)) ∧ (a = 3 ∧ b = 5) :=
by
  sorry

end cannot_obtain_fraction_3_5_l154_154168


namespace range_of_a_l154_154148

theorem range_of_a (x : ℝ) (a : ℝ) (h₀ : x ∈ Set.Icc (-2 : ℝ) 3)
(h₁ : 2 * x - x ^ 2 ≥ a) : a ≤ 1 :=
sorry

end range_of_a_l154_154148


namespace inequality_proof_l154_154981

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l154_154981


namespace find_a_circle_line_intersection_l154_154298

theorem find_a_circle_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0)
  (h2 : ∀ x y : ℝ, x + 2 * y + 1 = 0) :
  a = 3 := 
sorry

end find_a_circle_line_intersection_l154_154298


namespace inequality_proof_l154_154178

open Real

theorem inequality_proof (x y : ℝ) (hx : x > 1/2) (hy : y > 1) : 
  (4 * x^2) / (y - 1) + (y^2) / (2 * x - 1) ≥ 8 := 
by
  sorry

end inequality_proof_l154_154178


namespace squirrel_population_difference_l154_154493

theorem squirrel_population_difference :
  ∀ (total_population scotland_population rest_uk_population : ℕ), 
  scotland_population = 120000 →
  120000 = 75 * total_population / 100 →
  rest_uk_population = total_population - scotland_population →
  scotland_population - rest_uk_population = 80000 :=
by
  intros total_population scotland_population rest_uk_population h1 h2 h3
  sorry

end squirrel_population_difference_l154_154493


namespace find_other_integer_l154_154203

theorem find_other_integer (x y : ℤ) (h_sum : 3 * x + 2 * y = 115) (h_one_is_25 : x = 25 ∨ y = 25) : (x = 25 → y = 20) ∧ (y = 25 → x = 20) :=
by
  sorry

end find_other_integer_l154_154203


namespace average_weight_of_boys_l154_154215

theorem average_weight_of_boys 
  (n1 n2 : ℕ) 
  (w1 w2 : ℝ) 
  (h1 : n1 = 22) 
  (h2 : n2 = 8) 
  (h3 : w1 = 50.25) 
  (h4 : w2 = 45.15) : 
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 :=
by
  sorry

end average_weight_of_boys_l154_154215


namespace algebra_expression_correct_l154_154566

theorem algebra_expression_correct {x y : ℤ} (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
  sorry

end algebra_expression_correct_l154_154566


namespace trig_identity_l154_154230

-- Given conditions
variables (α : ℝ) (h_tan : Real.tan (Real.pi - α) = -2)

-- The goal is to prove the desired equality.
theorem trig_identity :
  1 / (Real.cos (2 * α) + Real.cos α * Real.cos α) = -5 / 2 :=
by
  sorry

end trig_identity_l154_154230


namespace sanity_indeterminable_transylvanian_is_upyr_l154_154435

noncomputable def transylvanianClaim := "I have lost my mind."

/-- Proving whether the sanity of the Transylvanian can be determined from the statement -/
theorem sanity_indeterminable (claim : String) : 
  claim = transylvanianClaim → 
  ¬ (∀ (sane : Prop), sane ∨ ¬ sane) := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

/-- Proving the nature of whether the Transylvanian is an upyr or human from the statement -/
theorem transylvanian_is_upyr (claim : String) : 
  claim = transylvanianClaim → 
  ∀ (human upyr : Prop), ¬ human ∧ upyr := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

end sanity_indeterminable_transylvanian_is_upyr_l154_154435


namespace no_prime_satisfies_condition_l154_154419

theorem no_prime_satisfies_condition (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, (Real.sqrt (p + n) + Real.sqrt n) = k :=
by
  sorry

end no_prime_satisfies_condition_l154_154419


namespace max_sum_a_b_c_d_e_f_g_l154_154193

theorem max_sum_a_b_c_d_e_f_g (a b c d e f g : ℕ)
  (h1 : a + b + c = 2)
  (h2 : b + c + d = 2)
  (h3 : c + d + e = 2)
  (h4 : d + e + f = 2)
  (h5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 := 
sorry

end max_sum_a_b_c_d_e_f_g_l154_154193


namespace solve_ineq_for_a_eq_0_values_of_a_l154_154871

theorem solve_ineq_for_a_eq_0 :
  ∀ x : ℝ, (|x + 2| - 3 * |x|) ≥ 0 ↔ (-1/2 <= x ∧ x <= 1) := 
by
  sorry

theorem values_of_a :
  ∀ x a : ℝ, (|x + 2| - 3 * |x|) ≥ a → (a ≤ 2) := 
by
  sorry

end solve_ineq_for_a_eq_0_values_of_a_l154_154871


namespace albert_needs_more_money_l154_154119

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end albert_needs_more_money_l154_154119


namespace sharon_trip_distance_l154_154127

noncomputable def usual_speed (x : ℝ) : ℝ := x / 200

noncomputable def reduced_speed (x : ℝ) : ℝ := x / 200 - 30 / 60

theorem sharon_trip_distance (x : ℝ) (h1 : (x / 3) / usual_speed x + (2 * x / 3) / reduced_speed x = 310) : 
x = 220 :=
by
  sorry

end sharon_trip_distance_l154_154127


namespace B_pow_101_eq_B_pow_5_l154_154039

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 0]]

theorem B_pow_101_eq_B_pow_5 : B^101 = B := 
by sorry

end B_pow_101_eq_B_pow_5_l154_154039


namespace no_intersection_points_l154_154452

theorem no_intersection_points : ¬ ∃ x y : ℝ, y = x ∧ y = x - 2 := by
  sorry

end no_intersection_points_l154_154452


namespace left_handed_women_percentage_l154_154482

noncomputable section

variables (x y : ℕ) (percentage : ℝ)

-- Conditions
def right_handed_ratio := 3
def left_handed_ratio := 1
def men_ratio := 3
def women_ratio := 2

def total_population_by_hand := right_handed_ratio * x + left_handed_ratio * x -- i.e., 4x
def total_population_by_gender := men_ratio * y + women_ratio * y -- i.e., 5y

-- Main Statement
theorem left_handed_women_percentage (h1 : total_population_by_hand = total_population_by_gender) :
    percentage = 25 :=
by
  sorry

end left_handed_women_percentage_l154_154482


namespace inscribed_sphere_radius_l154_154513

theorem inscribed_sphere_radius 
  (a : ℝ) 
  (h_angle : ∀ (lateral_face : ℝ), lateral_face = 60) : 
  ∃ (r : ℝ), r = a * (Real.sqrt 3) / 6 :=
by
  sorry

end inscribed_sphere_radius_l154_154513


namespace g_is_even_and_symmetric_l154_154364

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem g_is_even_and_symmetric :
  (∀ x : ℝ, g x = g (-x)) ∧ (∀ k : ℤ, g ((2 * k - 1) * π / 8) = 0) :=
by
  sorry

end g_is_even_and_symmetric_l154_154364


namespace percent_error_l154_154314

theorem percent_error (x : ℝ) (h : x > 0) :
  (abs ((12 * x) - (x / 3)) / (x / 3)) * 100 = 3500 :=
by
  sorry

end percent_error_l154_154314


namespace number_is_7625_l154_154940

-- We define x as a real number
variable (x : ℝ)

-- The condition given in the problem
def condition : Prop := x^2 + 95 = (x - 20)^2

-- The theorem we need to prove
theorem number_is_7625 (h : condition x) : x = 7.625 :=
by
  sorry

end number_is_7625_l154_154940


namespace fan_airflow_in_one_week_l154_154253

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end fan_airflow_in_one_week_l154_154253


namespace overall_gain_percentage_l154_154946

theorem overall_gain_percentage :
  let SP1 := 100
  let SP2 := 150
  let SP3 := 200
  let CP1 := SP1 / (1 + 0.20)
  let CP2 := SP2 / (1 + 0.15)
  let CP3 := SP3 / (1 - 0.05)
  let TCP := CP1 + CP2 + CP3
  let TSP := SP1 + SP2 + SP3
  let G := TSP - TCP
  let GP := (G / TCP) * 100
  GP = 6.06 := 
by {
  sorry
}

end overall_gain_percentage_l154_154946


namespace probability_three_primes_out_of_five_l154_154012

def probability_of_prime (p : ℚ) : Prop := ∃ k, k = 4 ∧ p = 4/10

def probability_of_not_prime (p : ℚ) : Prop := ∃ k, k = 6 ∧ p = 6/10

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_three_primes_out_of_five :
  ∀ p_prime p_not_prime : ℚ, 
  probability_of_prime p_prime →
  probability_of_not_prime p_not_prime →
  (combinations 5 3 * (p_prime^3 * p_not_prime^2) = 720/3125) :=
by
  intros p_prime p_not_prime h_prime h_not_prime
  sorry

end probability_three_primes_out_of_five_l154_154012


namespace find_difference_l154_154453

variable (f : ℝ → ℝ)

-- Conditions
axiom linear_f : ∀ x y a b, f (a * x + b * y) = a * f x + b * f y
axiom f_difference : f 6 - f 2 = 12

theorem find_difference : f 12 - f 2 = 30 :=
by
  sorry

end find_difference_l154_154453


namespace gcd_fact8_fact7_l154_154275

noncomputable def fact8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
noncomputable def fact7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem gcd_fact8_fact7 : Nat.gcd fact8 fact7 = fact7 := by
  unfold fact8 fact7
  exact sorry

end gcd_fact8_fact7_l154_154275


namespace find_blue_highlighters_l154_154862

theorem find_blue_highlighters
(h_pink : P = 9)
(h_yellow : Y = 8)
(h_total : T = 22)
(h_sum : P + Y + B = T) :
  B = 5 :=
by
  -- Proof would go here
  sorry

end find_blue_highlighters_l154_154862


namespace nonagon_perimeter_l154_154011

theorem nonagon_perimeter (n : ℕ) (side_length : ℝ) (P : ℝ) :
  n = 9 → side_length = 3 → P = n * side_length → P = 27 :=
by sorry

end nonagon_perimeter_l154_154011


namespace team_winning_percentage_l154_154931

theorem team_winning_percentage :
  let first_games := 100
  let remaining_games := 125 - first_games
  let won_first_games := 75
  let percentage_won := 50
  let won_remaining_games := Nat.ceil ((percentage_won : ℝ) / 100 * remaining_games)
  let total_won_games := won_first_games + won_remaining_games
  let total_games := 125
  let winning_percentage := (total_won_games : ℝ) / total_games * 100
  winning_percentage = 70.4 :=
by sorry

end team_winning_percentage_l154_154931


namespace E_eq_F_l154_154394

noncomputable def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

noncomputable def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_eq_F : E = F := 
sorry

end E_eq_F_l154_154394


namespace largest_prime_m_satisfying_quadratic_inequality_l154_154460

theorem largest_prime_m_satisfying_quadratic_inequality :
  ∃ (m : ℕ), m = 5 ∧ m^2 - 11 * m + 28 < 0 ∧ Prime m :=
by sorry

end largest_prime_m_satisfying_quadratic_inequality_l154_154460


namespace diameter_of_circle_A_l154_154354

theorem diameter_of_circle_A (r_B r_C : ℝ) (h1 : r_B = 12) (h2 : r_C = 3)
  (area_relation : ∀ (r_A : ℝ), π * (r_B^2 - r_A^2) = 4 * (π * r_C^2)) :
  ∃ r_A : ℝ, 2 * r_A = 12 * Real.sqrt 3 := by
  -- We will club the given conditions and logical sequence here
  sorry

end diameter_of_circle_A_l154_154354


namespace milkman_A_rent_share_l154_154646

theorem milkman_A_rent_share : 
  let A_cows := 24
  let A_months := 3
  let B_cows := 10
  let B_months := 5
  let C_cows := 35
  let C_months := 4
  let D_cows := 21
  let D_months := 3
  let total_rent := 3250
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months
  let fraction_A := A_cow_months / total_cow_months
  let A_rent_share := total_rent * fraction_A
  A_rent_share = 720 := 
by
  sorry

end milkman_A_rent_share_l154_154646


namespace smallest_d_for_divisibility_by_9_l154_154414

theorem smallest_d_for_divisibility_by_9 : ∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (437003 + d * 100) % 9 = 0 ∧ ∀ d', 0 ≤ d' ∧ d' < d → ((437003 + d' * 100) % 9 ≠ 0) :=
by
  sorry

end smallest_d_for_divisibility_by_9_l154_154414


namespace Meadow_sells_each_diaper_for_5_l154_154527

-- Define the conditions as constants
def boxes_per_week := 30
def packs_per_box := 40
def diapers_per_pack := 160
def total_revenue := 960000

-- Calculate total packs and total diapers
def total_packs := boxes_per_week * packs_per_box
def total_diapers := total_packs * diapers_per_pack

-- The target price per diaper
def price_per_diaper := total_revenue / total_diapers

-- Statement of the proof theorem
theorem Meadow_sells_each_diaper_for_5 : price_per_diaper = 5 := by
  sorry

end Meadow_sells_each_diaper_for_5_l154_154527


namespace average_wx_l154_154300

theorem average_wx (w x a b : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h1 : 6 / w + 6 / x = 6 / (a + b * i))
  (h2 : w * x = a + b * i) :
  (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_wx_l154_154300


namespace factorize_x4_minus_4x2_l154_154340

theorem factorize_x4_minus_4x2 (x : ℝ) : 
  x^4 - 4 * x^2 = x^2 * (x - 2) * (x + 2) :=
by
  sorry

end factorize_x4_minus_4x2_l154_154340


namespace find_m_n_l154_154173

theorem find_m_n (m n : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 1 → x^2 - m * x + n ≤ 0) → m = -4 ∧ n = -5 :=
by
  sorry

end find_m_n_l154_154173


namespace box_and_apples_weight_l154_154587

theorem box_and_apples_weight
  (total_weight : ℝ)
  (weight_after_half : ℝ)
  (h1 : total_weight = 62.8)
  (h2 : weight_after_half = 31.8) :
  ∃ (box_weight apple_weight : ℝ), box_weight = 0.8 ∧ apple_weight = 62 :=
by
  sorry

end box_and_apples_weight_l154_154587


namespace bead_necklaces_count_l154_154929

-- Define the conditions
def cost_per_necklace : ℕ := 9
def gemstone_necklaces_sold : ℕ := 3
def total_earnings : ℕ := 90

-- Define the total earnings from gemstone necklaces
def earnings_from_gemstone_necklaces : ℕ := gemstone_necklaces_sold * cost_per_necklace

-- Define the total earnings from bead necklaces
def earnings_from_bead_necklaces : ℕ := total_earnings - earnings_from_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_from_bead_necklaces / cost_per_necklace

-- The statement to be proved
theorem bead_necklaces_count : bead_necklaces_sold = 7 := by
  sorry

end bead_necklaces_count_l154_154929


namespace total_amount_contribution_l154_154942

theorem total_amount_contribution : 
  let r := 285
  let s := 35
  let a := 30
  let d := a / 2
  let c := 35
  r + s + a + d + c = 400 :=
by
  sorry

end total_amount_contribution_l154_154942


namespace geometric_sequence_S5_equals_l154_154420

theorem geometric_sequence_S5_equals :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    a 1 = 1 → 
    (a 3 + a 4) / (a 1 + a 2) = 4 → 
    ((S5 : ℤ) = 31 ∨ (S5 : ℤ) = 11) :=
by
  sorry

end geometric_sequence_S5_equals_l154_154420


namespace large_pyramid_tiers_l154_154542

def surface_area_pyramid (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

theorem large_pyramid_tiers :
  (∃ n : ℕ, surface_area_pyramid n = 42) →
  (∃ n : ℕ, surface_area_pyramid n = 2352) →
  ∃ n : ℕ, surface_area_pyramid n = 2352 ∧ n = 24 :=
by
  sorry

end large_pyramid_tiers_l154_154542


namespace point_above_line_l154_154071

/-- Given the point (-2, t) lies above the line x - 2y + 4 = 0,
    we want to prove t ∈ (1, +∞) -/
theorem point_above_line (t : ℝ) : (-2 - 2 * t + 4 > 0) → t > 1 :=
sorry

end point_above_line_l154_154071


namespace range_of_a_l154_154100

theorem range_of_a (x a : ℝ) :
  (∀ x, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l154_154100


namespace not_equivalent_to_0_0000375_l154_154943

theorem not_equivalent_to_0_0000375 : 
    ¬ (3 / 8000000 = 3.75 * 10 ^ (-5)) :=
by sorry

end not_equivalent_to_0_0000375_l154_154943


namespace slope_of_line_l154_154459

theorem slope_of_line {x1 x2 y1 y2 : ℝ} 
  (h1 : (1 / x1 + 2 / y1 = 0)) 
  (h2 : (1 / x2 + 2 / y2 = 0)) 
  (h_neq : x1 ≠ x2) : 
  (y2 - y1) / (x2 - x1) = -2 := 
sorry

end slope_of_line_l154_154459


namespace find_k_l154_154999

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem find_k :
  ∃ k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 35 ∧ k = 29 := 
sorry

end find_k_l154_154999


namespace age_difference_l154_154790

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end age_difference_l154_154790


namespace theater_total_bills_l154_154026

theorem theater_total_bills (tickets : ℕ) (price : ℕ) (x : ℕ) (number_of_5_bills : ℕ) (number_of_10_bills : ℕ) (number_of_20_bills : ℕ) :
  tickets = 300 →
  price = 40 →
  number_of_20_bills = x →
  number_of_10_bills = 2 * x →
  number_of_5_bills = 2 * x + 20 →
  20 * x + 10 * (2 * x) + 5 * (2 * x + 20) = tickets * price →
  number_of_5_bills + number_of_10_bills + number_of_20_bills = 1210 := by
    intro h_tickets h_price h_20_bills h_10_bills h_5_bills h_total
    sorry

end theater_total_bills_l154_154026


namespace add_neg_two_l154_154149

theorem add_neg_two : 1 + (-2 : ℚ) = -1 := by
  sorry

end add_neg_two_l154_154149


namespace quadratic_root_condition_l154_154270

theorem quadratic_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
by 
  sorry

end quadratic_root_condition_l154_154270


namespace geometric_progression_common_ratio_l154_154443

theorem geometric_progression_common_ratio (x y z w r : ℂ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h_geom : x * (y - w) = a ∧ y * (z - x) = a * r ∧ z * (w - y) = a * r^2 ∧ w * (x - z) = a * r^3) :
  1 + r + r^2 + r^3 = 0 :=
sorry

end geometric_progression_common_ratio_l154_154443


namespace proof_mn_eq_9_l154_154485

theorem proof_mn_eq_9 (m n : ℕ) (h1 : 2 * m + n = 8) (h2 : m - n = 1) : m^n = 9 :=
by {
  sorry 
}

end proof_mn_eq_9_l154_154485


namespace insurance_slogan_equivalence_l154_154352

variables (H I : Prop)

theorem insurance_slogan_equivalence :
  (∀ x, x → H → I) ↔ (∀ y, y → ¬I → ¬H) :=
sorry

end insurance_slogan_equivalence_l154_154352


namespace sqrt_neg_squared_eq_two_l154_154662

theorem sqrt_neg_squared_eq_two : (-Real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_neg_squared_eq_two_l154_154662


namespace aubrey_travel_time_l154_154078

def aubrey_time_to_school (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem aubrey_travel_time :
  aubrey_time_to_school 88 22 = 4 := by
  sorry

end aubrey_travel_time_l154_154078


namespace sin_690_l154_154596

-- Defining the known conditions as hypotheses:
axiom sin_periodic (x : ℝ) : Real.sin (x + 360) = Real.sin x
axiom sin_odd (x : ℝ) : Real.sin (-x) = - Real.sin x
axiom sin_thirty : Real.sin 30 = 1 / 2

theorem sin_690 : Real.sin 690 = -1 / 2 :=
by
  -- Proof would go here, but it is skipped with sorry.
  sorry

end sin_690_l154_154596


namespace binary_operations_unique_l154_154886

def binary_operation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (f a (f b c) = (f a b) * c)
  ∧ ∀ a : ℝ, a > 0 → a ≥ 1 → f a a ≥ 1

theorem binary_operations_unique (f : ℝ → ℝ → ℝ) (h : binary_operation f) :
  (∀ a b, f a b = a * b) ∨ (∀ a b, f a b = a / b) :=
sorry

end binary_operations_unique_l154_154886


namespace john_mary_game_l154_154091

theorem john_mary_game (n : ℕ) (h : n ≥ 3) :
  ∃ S : ℕ, S = n * (n + 1) :=
by
  sorry

end john_mary_game_l154_154091


namespace find_product_of_M1_M2_l154_154111

theorem find_product_of_M1_M2 (x M1 M2 : ℝ) 
  (h : (27 * x - 19) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) : 
  M1 * M2 = -2170 := 
sorry

end find_product_of_M1_M2_l154_154111


namespace find_fraction_squares_l154_154368

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end find_fraction_squares_l154_154368


namespace marbles_remaining_l154_154269

def original_marbles : Nat := 64
def given_marbles : Nat := 14
def remaining_marbles : Nat := original_marbles - given_marbles

theorem marbles_remaining : remaining_marbles = 50 :=
  by
    sorry

end marbles_remaining_l154_154269


namespace probability_of_at_least_one_vowel_is_799_over_1024_l154_154441

def Set1 : Set Char := {'a', 'e', 'i', 'b', 'c', 'd', 'f', 'g'}
def Set2 : Set Char := {'u', 'o', 'y', 'k', 'l', 'm', 'n', 'p'}
def Set3 : Set Char := {'e', 'u', 'v', 'r', 's', 't', 'w', 'x'}
def Set4 : Set Char := {'a', 'i', 'o', 'z', 'h', 'j', 'q', 'r'}

noncomputable def probability_of_at_least_one_vowel : ℚ :=
  1 - (5/8 : ℚ) * (3/4 : ℚ) * (3/4 : ℚ) * (5/8 : ℚ)

theorem probability_of_at_least_one_vowel_is_799_over_1024 :
  probability_of_at_least_one_vowel = 799 / 1024 :=
by
  sorry

end probability_of_at_least_one_vowel_is_799_over_1024_l154_154441


namespace Karls_Total_Travel_Distance_l154_154432

theorem Karls_Total_Travel_Distance :
  let consumption_rate := 35
  let full_tank_gallons := 14
  let initial_miles := 350
  let added_gallons := 8
  let remaining_gallons := 7
  let net_gallons_consumed := (full_tank_gallons + added_gallons - remaining_gallons)
  let total_distance := net_gallons_consumed * consumption_rate
  total_distance = 525 := 
by 
  sorry

end Karls_Total_Travel_Distance_l154_154432


namespace jan_drives_more_miles_than_ian_l154_154492

-- Definitions of conditions
variables (s t d m: ℝ)

-- Ian's travel equation
def ian_distance := d = s * t

-- Han's travel equation
def han_distance := (d + 115) = (s + 8) * (t + 2)

-- Jan's travel equation
def jan_distance := m = (s + 12) * (t + 3)

-- The proof statement we want to prove
theorem jan_drives_more_miles_than_ian :
    (∀ (s t d m : ℝ),
    d = s * t →
    (d + 115) = (s + 8) * (t + 2) →
    m = (s + 12) * (t + 3) →
    (m - d) = 184.5) :=
    sorry

end jan_drives_more_miles_than_ian_l154_154492


namespace quadratic_to_standard_form_div_l154_154325

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end quadratic_to_standard_form_div_l154_154325


namespace SharonOranges_l154_154139

-- Define the given conditions
def JanetOranges : Nat := 9
def TotalOranges : Nat := 16

-- Define the statement that needs to be proven
theorem SharonOranges (J : Nat) (T : Nat) (S : Nat) (hJ : J = 9) (hT : T = 16) (hS : S = T - J) : S = 7 := by
  -- (proof to be filled in later)
  sorry

end SharonOranges_l154_154139


namespace number_divided_by_three_l154_154397

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l154_154397


namespace least_palindrome_divisible_by_25_l154_154877

theorem least_palindrome_divisible_by_25 : ∃ (n : ℕ), 
  (10^4 ≤ n ∧ n < 10^5) ∧
  (∀ (a b c : ℕ), n = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a) ∧
  n % 25 = 0 ∧
  n = 10201 :=
by
  sorry

end least_palindrome_divisible_by_25_l154_154877


namespace compute_binom_value_l154_154412

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else x * binom (x - 1) (k - 1) / k

theorem compute_binom_value : 
  (binom (1/2) 2014 * 4^2014 / binom 4028 2014) = -1/4027 :=
by 
  sorry

end compute_binom_value_l154_154412


namespace consumer_installment_credit_l154_154783

theorem consumer_installment_credit (A C : ℝ) (h1 : A = 0.36 * C) (h2 : 35 = (1 / 3) * A) :
  C = 291.67 :=
by 
  -- The proof should go here
  sorry

end consumer_installment_credit_l154_154783


namespace translate_parabola_l154_154993

theorem translate_parabola :
  ∀ (x y : ℝ), y = -5*x^2 + 1 → y = -5*(x + 1)^2 - 1 := by
  sorry

end translate_parabola_l154_154993


namespace total_rainfall_in_Springdale_l154_154170

theorem total_rainfall_in_Springdale
    (rainfall_first_week rainfall_second_week : ℝ)
    (h1 : rainfall_second_week = 1.5 * rainfall_first_week)
    (h2 : rainfall_second_week = 12) :
    (rainfall_first_week + rainfall_second_week = 20) :=
by
  sorry

end total_rainfall_in_Springdale_l154_154170


namespace total_movies_shown_l154_154930

theorem total_movies_shown (screen1_movies : ℕ) (screen2_movies : ℕ) (screen3_movies : ℕ)
                          (screen4_movies : ℕ) (screen5_movies : ℕ) (screen6_movies : ℕ)
                          (h1 : screen1_movies = 3) (h2 : screen2_movies = 4) 
                          (h3 : screen3_movies = 2) (h4 : screen4_movies = 3) 
                          (h5 : screen5_movies = 5) (h6 : screen6_movies = 2) :
  screen1_movies + screen2_movies + screen3_movies + screen4_movies + screen5_movies + screen6_movies = 19 := 
by
  sorry

end total_movies_shown_l154_154930


namespace math_problem_l154_154204

theorem math_problem
    (p q s : ℕ)
    (prime_p : Nat.Prime p)
    (prime_q : Nat.Prime q)
    (prime_s : Nat.Prime s)
    (h1 : p * q = s + 6)
    (h2 : 3 < p)
    (h3 : p < q) :
    p = 5 :=
    sorry

end math_problem_l154_154204


namespace mountain_height_is_1700m_l154_154398

noncomputable def height_of_mountain (temp_base : ℝ) (temp_summit : ℝ) (rate_decrease : ℝ) : ℝ :=
  ((temp_base - temp_summit) / rate_decrease) * 100

theorem mountain_height_is_1700m :
  height_of_mountain 26 14.1 0.7 = 1700 :=
by
  sorry

end mountain_height_is_1700m_l154_154398


namespace siding_cost_l154_154947

noncomputable def front_wall_width : ℝ := 10
noncomputable def front_wall_height : ℝ := 8
noncomputable def triangle_base : ℝ := 10
noncomputable def triangle_height : ℝ := 4
noncomputable def panel_area : ℝ := 100
noncomputable def panel_cost : ℝ := 30

theorem siding_cost :
  let front_wall_area := front_wall_width * front_wall_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_area := front_wall_area + triangle_area
  let panels_needed := total_area / panel_area
  let total_cost := panels_needed * panel_cost
  total_cost = 30 := sorry

end siding_cost_l154_154947


namespace find_nat_nums_satisfying_eq_l154_154445

theorem find_nat_nums_satisfying_eq (m n : ℕ) (h_m : m = 3) (h_n : n = 3) : 2 ^ n + 1 = m ^ 2 :=
by
  rw [h_m, h_n]
  sorry

end find_nat_nums_satisfying_eq_l154_154445


namespace odd_function_expression_l154_154278

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 3 * x - 4 else - (x^2 - 3 * x - 4)

theorem odd_function_expression (x : ℝ) (h : x < 0) : 
  f x = -x^2 + 3 * x + 4 :=
by
  sorry

end odd_function_expression_l154_154278


namespace librarian_took_books_l154_154318

-- Define variables and conditions
def total_books : ℕ := 46
def books_per_shelf : ℕ := 4
def shelves_needed : ℕ := 9

-- Define the number of books Oliver has left to put away
def books_left : ℕ := shelves_needed * books_per_shelf

-- Define the number of books the librarian took
def books_taken : ℕ := total_books - books_left

-- State the theorem
theorem librarian_took_books : books_taken = 10 := by
  sorry

end librarian_took_books_l154_154318


namespace perfect_squares_of_nat_l154_154212

theorem perfect_squares_of_nat (a b c : ℕ) (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ m n p q : ℕ, ab = m^2 ∧ bc = n^2 ∧ ca = p^2 ∧ ab + bc + ca = q^2 :=
by sorry

end perfect_squares_of_nat_l154_154212


namespace relay_race_arrangements_l154_154545

noncomputable def number_of_arrangements (athletes : Finset ℕ) (a b : ℕ) : ℕ :=
  (athletes.erase a).card.factorial * ((athletes.erase b).card.factorial - 2) * (athletes.card.factorial / ((athletes.card - 4).factorial)) / 4

theorem relay_race_arrangements :
  let athletes := {0, 1, 2, 3, 4, 5}
  number_of_arrangements athletes 0 1 = 252 := 
by
  sorry

end relay_race_arrangements_l154_154545


namespace maintenance_cost_relation_maximize_average_profit_l154_154095

def maintenance_cost (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1400 * n - 1000

theorem maintenance_cost_relation :
  maintenance_cost 2 = 1800 ∧ maintenance_cost 5 = 6000 ∧
  (∀ n, n ≥ 2 → maintenance_cost n = 1400 * n - 1000) :=
by
  sorry

noncomputable def average_profit (n : ℕ) : ℝ :=
  if n < 2 then 0 else 60000 - (1 / n) * (137600 + 1400 * ((n - 1) * (n + 2) / 2) - 1000 * (n - 1))

theorem maximize_average_profit (n : ℕ) :
  n = 14 ↔ (average_profit n = 40700) :=
by
  sorry

end maintenance_cost_relation_maximize_average_profit_l154_154095


namespace line_points_k_l154_154984

noncomputable def k : ℝ := 8

theorem line_points_k (k : ℝ) : 
  (∀ k : ℝ, ∃ b : ℝ, b = (10 - k) / (5 - 5) ∧
  ∀ b, b = (-k) / (20 - 5) → k = 8) :=
  by
  sorry

end line_points_k_l154_154984


namespace new_person_weight_l154_154847

noncomputable def weight_of_new_person (weight_of_replaced : ℕ) (number_of_persons : ℕ) (increase_in_average : ℕ) := 
  weight_of_replaced + number_of_persons * increase_in_average

theorem new_person_weight:
  weight_of_new_person 70 8 3 = 94 :=
  by
  -- Proof omitted
  sorry

end new_person_weight_l154_154847


namespace sum_of_perimeters_l154_154869

theorem sum_of_perimeters (a : ℕ → ℝ) (h₁ : a 0 = 180) (h₂ : ∀ n, a (n + 1) = 1 / 2 * a n) :
  (∑' n, a n) = 360 :=
by
  sorry

end sum_of_perimeters_l154_154869


namespace sin_double_angle_value_l154_154035

theorem sin_double_angle_value (α : ℝ) (h₁ : Real.sin (π / 4 - α) = 3 / 5) (h₂ : 0 < α ∧ α < π / 4) : 
  Real.sin (2 * α) = 7 / 25 := 
sorry

end sin_double_angle_value_l154_154035


namespace percentage_increase_l154_154595

theorem percentage_increase (P Q R : ℝ) (x y : ℝ) 
  (h1 : P > 0) (h2 : Q > 0) (h3 : R > 0)
  (h4 : P = (1 + x / 100) * Q)
  (h5 : Q = (1 + y / 100) * R)
  (h6 : P = 2.4 * R) :
  x + y = 140 :=
sorry

end percentage_increase_l154_154595


namespace range_of_a_l154_154820

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x - 1) / x > 2 * a) ↔ a ∈ (Set.Ici (-1/2) : Set ℝ) :=
by
  sorry

end range_of_a_l154_154820


namespace problem_statement_l154_154805

theorem problem_statement (x y : ℝ) (h : -x + 2 * y = 5) :
  5 * (x - 2 * y) ^ 2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  sorry

end problem_statement_l154_154805


namespace initial_salary_increase_l154_154815

theorem initial_salary_increase :
  ∃ x : ℝ, 5000 * (1 + x/100) * 0.95 = 5225 := by
  sorry

end initial_salary_increase_l154_154815


namespace triangle_II_area_l154_154096

noncomputable def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem triangle_II_area (a b : ℝ) :
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  II_area = (a + b) ^ 2 :=
by
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  sorry

end triangle_II_area_l154_154096


namespace assignment_statement_correct_l154_154409

-- Definitions for the conditions:
def cond_A : Prop := ∀ M : ℕ, (M = M + 3)
def cond_B : Prop := ∀ M : ℕ, (M = M + (3 - M))
def cond_C : Prop := ∀ M : ℕ, (M = M + 3)
def cond_D : Prop := true ∧ cond_A ∧ cond_B ∧ cond_C

-- Theorem statement proving the correct interpretation of the assignment is condition B
theorem assignment_statement_correct : cond_B :=
by
  sorry

end assignment_statement_correct_l154_154409


namespace min_distance_parabola_midpoint_l154_154276

theorem min_distance_parabola_midpoint 
  (a : ℝ) (m : ℝ) (h_pos_a : a > 0) :
  (m ≥ 1 / a → ∃ M_y : ℝ, M_y = (2 * m * a - 1) / (4 * a)) ∧ 
  (m < 1 / a → ∃ M_y : ℝ, M_y = a * m^2 / 4) := 
by 
  sorry

end min_distance_parabola_midpoint_l154_154276


namespace atomic_weight_Ca_l154_154223

def molecular_weight_CaH2 : ℝ := 42
def atomic_weight_H : ℝ := 1.008

theorem atomic_weight_Ca : atomic_weight_H * 2 < molecular_weight_CaH2 :=
by sorry

end atomic_weight_Ca_l154_154223


namespace find_a_l154_154830

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l154_154830


namespace toy_store_fraction_l154_154628

theorem toy_store_fraction
  (allowance : ℝ) (arcade_fraction : ℝ) (candy_store_amount : ℝ)
  (h1 : allowance = 1.50)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : candy_store_amount = 0.40) :
  (0.60 - candy_store_amount) / (allowance - arcade_fraction * allowance) = 1 / 3 :=
by
  -- We're skipping the actual proof steps
  sorry

end toy_store_fraction_l154_154628


namespace oscar_leap_vs_elmer_stride_l154_154508

/--
Given:
1. The 51st telephone pole is exactly 6600 feet from the first pole.
2. Elmer the emu takes 50 equal strides to walk between consecutive telephone poles.
3. Oscar the ostrich can cover the same distance in 15 equal leaps.
4. There are 50 gaps between the 51 poles.

Prove:
Oscar's leap is 6 feet longer than Elmer's stride.
-/
theorem oscar_leap_vs_elmer_stride : 
  let total_distance := 6600 
  let elmer_strides_per_gap := 50
  let oscar_leaps_per_gap := 15
  let num_gaps := 50
  let elmer_total_strides := elmer_strides_per_gap * num_gaps
  let oscar_total_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 6 := 
by {
  -- The proof would go here.
  sorry
}

end oscar_leap_vs_elmer_stride_l154_154508


namespace quad_roots_sum_l154_154517

theorem quad_roots_sum {x₁ x₂ : ℝ} (h1 : x₁ + x₂ = 5) (h2 : x₁ * x₂ = -6) :
  1 / x₁ + 1 / x₂ = -5 / 6 :=
by
  sorry

end quad_roots_sum_l154_154517


namespace bryce_received_15_raisins_l154_154202

theorem bryce_received_15_raisins (x : ℕ) (c : ℕ) (h1 : c = x - 10) (h2 : c = x / 3) : x = 15 :=
by
  sorry

end bryce_received_15_raisins_l154_154202


namespace simplify_expression_l154_154375

theorem simplify_expression (a b : ℕ) (h : a / b = 1 / 3) : 
    1 - (a - b) / (a - 2 * b) / ((a ^ 2 - b ^ 2) / (a ^ 2 - 4 * a * b + 4 * b ^ 2)) = 3 / 4 := 
by sorry

end simplify_expression_l154_154375


namespace purchasing_plans_and_optimal_plan_l154_154309

def company_time := 10
def model_A_cost := 60000
def model_B_cost := 40000
def model_A_production := 15
def model_B_production := 10
def budget := 440000
def production_capacity := 102

theorem purchasing_plans_and_optimal_plan (x y : ℕ) (h1 : x + y = company_time) (h2 : model_A_cost * x + model_B_cost * y ≤ budget) :
  (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∧ (x = 1 ∧ y = 9) :=
by 
  sorry

end purchasing_plans_and_optimal_plan_l154_154309


namespace find_point_P_l154_154327

noncomputable def tangent_at (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem find_point_P :
  ∃ (x₀ y₀ : ℝ), (y₀ = (1 / x₀)) 
  ∧ (0 < x₀)
  ∧ (tangent_at (fun x => x^2) 2 = 4)
  ∧ (tangent_at (fun x => (1 / x)) x₀ = -1 / 4) 
  ∧ (x₀ = 2)
  ∧ (y₀ = 1 / 2) :=
sorry

end find_point_P_l154_154327


namespace solution_l154_154573

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem solution (a b : ℝ) (H : a = 5 * Real.pi / 8 ∧ b = 7 * Real.pi / 8) :
  is_monotonically_increasing g a b :=
sorry

end solution_l154_154573


namespace maximum_xy_l154_154464

theorem maximum_xy (x y : ℕ) (h1 : 7 * x + 2 * y = 110) : ∃ x y, (7 * x + 2 * y = 110) ∧ (x > 0) ∧ (y > 0) ∧ (x * y = 216) :=
by
  sorry

end maximum_xy_l154_154464


namespace trig_identity_l154_154068

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
by 
  sorry

end trig_identity_l154_154068


namespace pablo_puzzle_pieces_per_hour_l154_154317

theorem pablo_puzzle_pieces_per_hour
  (num_300_puzzles : ℕ)
  (num_500_puzzles : ℕ)
  (pieces_per_300_puzzle : ℕ)
  (pieces_per_500_puzzle : ℕ)
  (max_hours_per_day : ℕ)
  (total_days : ℕ)
  (total_pieces_completed : ℕ)
  (total_hours_spent : ℕ)
  (P : ℕ)
  (h1 : num_300_puzzles = 8)
  (h2 : num_500_puzzles = 5)
  (h3 : pieces_per_300_puzzle = 300)
  (h4 : pieces_per_500_puzzle = 500)
  (h5 : max_hours_per_day = 7)
  (h6 : total_days = 7)
  (h7 : total_pieces_completed = (num_300_puzzles * pieces_per_300_puzzle + num_500_puzzles * pieces_per_500_puzzle))
  (h8 : total_hours_spent = max_hours_per_day * total_days)
  (h9 : P = total_pieces_completed / total_hours_spent) :
  P = 100 :=
sorry

end pablo_puzzle_pieces_per_hour_l154_154317


namespace Bob_wins_game_l154_154576

theorem Bob_wins_game :
  ∀ (initial_set : Set ℕ),
    47 ∈ initial_set →
    2016 ∈ initial_set →
    (∀ (a b : ℕ), a ∈ initial_set → b ∈ initial_set → a > b → (a - b) ∉ initial_set → (a - b) ∈ initial_set) →
    (∀ (S : Set ℕ), S ⊆ initial_set → ∃ (n : ℕ), ∀ m ∈ S, m > n) → false :=
by
  sorry

end Bob_wins_game_l154_154576


namespace total_triangles_correct_l154_154777

-- Define the rectangle and additional constructions
structure Rectangle :=
  (A B C D : Type)
  (midpoint_AB midpoint_BC midpoint_CD midpoint_DA : Type)
  (AC BD diagonals : Type)

-- Hypothesize the structure
variables (rect : Rectangle)

-- Define the number of triangles
def number_of_triangles (r : Rectangle) : Nat := 16

-- The theorem statement
theorem total_triangles_correct : number_of_triangles rect = 16 :=
by
  sorry

end total_triangles_correct_l154_154777


namespace length_proof_l154_154257

noncomputable def length_of_plot 
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ) -- cost of fencing per meter on flat ground
  (height_rise : ℝ) -- total height rise in meters
  (total_cost: ℝ) -- total cost of fencing
  (length_increase : ℝ) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ) -- scaling factor for cost increase on breadth
  (increased_breadth_cost_rate : ℝ) -- actual increased cost rate per meter for breadth
: ℝ :=
2 * (b + length_increase) * fence_cost_flat + 
2 * b * (fence_cost_flat + fence_cost_flat * (height_rise * cost_increase_rate))

theorem length_proof
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ := 26.50) -- cost of fencing per meter on flat ground
  (height_rise : ℝ := 5) -- total height rise in meters
  (total_cost: ℝ := 5300) -- total cost of fencing
  (length_increase : ℝ := 20) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ := 0.10) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ := fence_cost_flat * 0.5) -- increased cost factor
  (increased_breadth_cost_rate : ℝ := 39.75) -- recalculated cost rate per meter for breadth
  (length: ℝ := b + length_increase)
  (proof_step : total_cost = length_of_plot b fence_cost_flat height_rise total_cost length_increase cost_increase_rate breadth_cost_increase_factor increased_breadth_cost_rate)
: length = 52 :=
by
  sorry -- Proof omitted

end length_proof_l154_154257


namespace average_speed_of_trip_l154_154904

theorem average_speed_of_trip :
  let speed1 := 30
  let time1 := 5
  let speed2 := 42
  let time2 := 10
  let total_time := 15
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let average_speed := total_distance / total_time
  average_speed = 38 := 
by 
  sorry

end average_speed_of_trip_l154_154904


namespace feb_03_2013_nine_day_l154_154147

-- Definitions of the main dates involved
def dec_21_2012 : Nat := 0  -- Assuming day 0 is Dec 21, 2012
def feb_03_2013 : Nat := 45  -- 45 days after Dec 21, 2012

-- Definition to determine the Nine-day period
def nine_day_period (x : Nat) : (Nat × Nat) :=
  let q := x / 9
  let r := x % 9
  (q + 1, r + 1)

-- Theorem we want to prove
theorem feb_03_2013_nine_day : nine_day_period feb_03_2013 = (5, 9) :=
by
  sorry

end feb_03_2013_nine_day_l154_154147


namespace factor_tree_value_l154_154080

theorem factor_tree_value :
  let F := 7 * (2 * 2)
  let H := 11 * 2
  let G := 11 * H
  let X := F * G
  X = 6776 :=
by
  sorry

end factor_tree_value_l154_154080


namespace compare_neg_fractions_l154_154667

theorem compare_neg_fractions : (-5/4 : ℚ) > (-4/3 : ℚ) := 
sorry

end compare_neg_fractions_l154_154667


namespace most_reasonable_sampling_method_l154_154620

-- Definitions for the conditions
def significant_difference_by_stage : Prop := 
  -- There is a significant difference in vision condition at different educational stages
  sorry

def no_significant_difference_by_gender : Prop :=
  -- There is no significant difference in vision condition between male and female students
  sorry

-- Theorem statement
theorem most_reasonable_sampling_method 
  (h1 : significant_difference_by_stage) 
  (h2 : no_significant_difference_by_gender) : 
  -- The most reasonable sampling method is stratified sampling by educational stage
  sorry :=
by
  -- Proof skipped
  sorry

end most_reasonable_sampling_method_l154_154620


namespace chemistry_problem_l154_154854

theorem chemistry_problem 
(C : ℝ)  -- concentration of the original salt solution
(h_mix : 1 * C / 100 = 15 * 2 / 100) : 
  C = 30 := 
sorry

end chemistry_problem_l154_154854


namespace part_one_part_two_l154_154913

-- 1. Prove that 1 + 2x^4 >= 2x^3 + x^2 for all real numbers x
theorem part_one (x : ℝ) : 1 + 2 * x^4 ≥ 2 * x^3 + x^2 := sorry

-- 2. Given x + 2y + 3z = 6, prove that x^2 + y^2 + z^2 ≥ 18 / 7
theorem part_two (x y z : ℝ) (h : x + 2 * y + 3 * z = 6) : x^2 + y^2 + z^2 ≥ 18 / 7 := sorry

end part_one_part_two_l154_154913


namespace original_cost_of_car_l154_154451

theorem original_cost_of_car (C : ℝ) 
  (repair_cost : ℝ := 15000)
  (selling_price : ℝ := 64900)
  (profit_percent : ℝ := 13.859649122807017) :
  C = 43837.21 :=
by
  have h1 : C + repair_cost = selling_price - (selling_price - (C + repair_cost)) := by sorry
  have h2 : profit_percent / 100 = (selling_price - (C + repair_cost)) / C := by sorry
  have h3 : C = 43837.21 := by sorry
  exact h3

end original_cost_of_car_l154_154451


namespace luncheon_cost_l154_154288

section LuncheonCosts

variables (s c p : ℝ)

/- Conditions -/
def eq1 : Prop := 2 * s + 5 * c + 2 * p = 6.25
def eq2 : Prop := 5 * s + 8 * c + 3 * p = 12.10

/- Goal -/
theorem luncheon_cost : eq1 s c p → eq2 s c p → s + c + p = 1.55 :=
by
  intro h1 h2
  sorry

end LuncheonCosts

end luncheon_cost_l154_154288


namespace lottery_probability_exactly_one_common_l154_154197

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l154_154197


namespace opposite_of_negative_six_is_six_l154_154571

theorem opposite_of_negative_six_is_six : ∀ (x : ℤ), (-6 + x = 0) → x = 6 :=
by
  intro x hx
  sorry

end opposite_of_negative_six_is_six_l154_154571


namespace range_of_m_l154_154669

noncomputable def f (x : ℝ) : ℝ := |x - 3| - 2
noncomputable def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem range_of_m :
  (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by sorry

end range_of_m_l154_154669


namespace personal_trainer_cost_proof_l154_154077

-- Define the conditions
def hourly_wage_before_raise : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 5
def old_bills_per_week : ℝ := 600
def leftover_money : ℝ := 980

-- Define the question
def new_hourly_wage : ℝ := hourly_wage_before_raise * (1 + raise_percentage)
def weekly_hours : ℕ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def total_weekly_expenses : ℝ := weekly_earnings - leftover_money
def personal_trainer_cost_per_week : ℝ := total_weekly_expenses - old_bills_per_week

-- Theorem statement
theorem personal_trainer_cost_proof : personal_trainer_cost_per_week = 100 := 
by
  -- Proof to be filled
  sorry

end personal_trainer_cost_proof_l154_154077


namespace diagonal_perimeter_ratio_l154_154328

theorem diagonal_perimeter_ratio
    (b : ℝ)
    (h : b ≠ 0) -- To ensure the garden has non-zero side lengths
    (a : ℝ) (h1: a = 3 * b) 
    (d : ℝ) (h2: d = (Real.sqrt (b^2 + a^2)))
    (P : ℝ) (h3: P = 2 * a + 2 * b)
    (h4 : d = b * (Real.sqrt 10)) :
  d / P = (Real.sqrt 10) / 8 := by
    sorry

end diagonal_perimeter_ratio_l154_154328


namespace johns_website_visits_l154_154401

theorem johns_website_visits (c: ℝ) (d: ℝ) (days: ℕ) (h1: c = 0.01) (h2: d = 10) (h3: days = 30) :
  d / c * days = 30000 :=
by
  sorry

end johns_website_visits_l154_154401


namespace repeating_decimal_product_l154_154888

theorem repeating_decimal_product :
  (8 / 99) * (36 / 99) = 288 / 9801 :=
by
  sorry

end repeating_decimal_product_l154_154888


namespace parabola_focus_directrix_distance_l154_154786

theorem parabola_focus_directrix_distance {a : ℝ} (h₀ : a > 0):
  (∃ (b : ℝ), ∃ (x1 x2 : ℝ), (x1 + x2 = 1 / a) ∧ (1 / (2 * a) = 1)) → 
  (1 / (2 * a) / 2 = 1 / 4) :=
by
  sorry

end parabola_focus_directrix_distance_l154_154786


namespace eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l154_154145

theorem eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five:
  (0.85 * 40) - (4 / 5 * 25) = 14 :=
by
  sorry

end eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l154_154145


namespace simplify_expression_l154_154938

theorem simplify_expression (x y : ℝ) :
  ((x + y)^2 - y * (2 * x + y) - 6 * x) / (2 * x) = (1 / 2) * x - 3 :=
by
  sorry

end simplify_expression_l154_154938


namespace pound_of_rice_cost_l154_154266

theorem pound_of_rice_cost 
(E R K : ℕ) (h1: E = R) (h2: K = 4 * (E / 12)) (h3: K = 11) : R = 33 := by
  sorry

end pound_of_rice_cost_l154_154266


namespace train_length_l154_154195

theorem train_length (T : ℕ) (S : ℕ) (conversion_factor : ℚ) (L : ℕ) 
  (hT : T = 16)
  (hS : S = 108)
  (hconv : conversion_factor = 5 / 18)
  (hL : L = 480) :
  L = ((S * conversion_factor : ℚ) * T : ℚ) :=
sorry

end train_length_l154_154195


namespace sector_area_l154_154547

noncomputable def l : ℝ := 4
noncomputable def θ : ℝ := 2
noncomputable def r : ℝ := l / θ

theorem sector_area :
  (1 / 2) * l * r = 4 :=
by
  -- Proof goes here
  sorry

end sector_area_l154_154547


namespace find_y_l154_154603

-- Definitions of vectors and parallel relationship
def vector_a : ℝ × ℝ := (4, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (6, y)
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- The theorem we want to prove
theorem find_y (y : ℝ) (h : parallel vector_a (vector_b y)) : y = 3 :=
sorry

end find_y_l154_154603


namespace total_lunch_cost_l154_154675

theorem total_lunch_cost
  (children chaperones herself additional_lunches cost_per_lunch : ℕ)
  (h1 : children = 35)
  (h2 : chaperones = 5)
  (h3 : herself = 1)
  (h4 : additional_lunches = 3)
  (h5 : cost_per_lunch = 7) :
  (children + chaperones + herself + additional_lunches) * cost_per_lunch = 308 :=
by
  sorry

end total_lunch_cost_l154_154675


namespace PB_distance_eq_l154_154516

theorem PB_distance_eq {
  A B C D P : Type
} (PA PD PC : ℝ) (hPA: PA = 6) (hPD: PD = 8) (hPC: PC = 10)
  (h_equidistant: ∃ y : ℝ, PA^2 + y^2 = PB^2 ∧ PD^2 + y^2 = PC^2) :
  ∃ PB : ℝ, PB = 6 * Real.sqrt 2 := 
by
  sorry

end PB_distance_eq_l154_154516


namespace lines_parallel_if_perpendicular_to_plane_l154_154481

variables {α β γ : Plane} {m n : Line}

-- Define the properties of perpendicular lines to planes and parallel lines
def perpendicular_to (l : Line) (p : Plane) : Prop := 
sorry -- definition skipped

def parallel_to (l1 l2 : Line) : Prop := 
sorry -- definition skipped

-- Theorem Statement (equivalent translation of the given question and its correct answer)
theorem lines_parallel_if_perpendicular_to_plane 
  (h1 : perpendicular_to m α) 
  (h2 : perpendicular_to n α) : parallel_to m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l154_154481


namespace remainder_for_second_number_l154_154921

theorem remainder_for_second_number (G R1 : ℕ) (first_number second_number : ℕ)
  (hG : G = 144) (hR1 : R1 = 23) (hFirst : first_number = 6215) (hSecond : second_number = 7373) :
  ∃ q2 R2, second_number = G * q2 + R2 ∧ R2 = 29 := 
by {
  -- Ensure definitions are in scope
  exact sorry
}

end remainder_for_second_number_l154_154921


namespace problem_statement_l154_154989

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end problem_statement_l154_154989


namespace shawn_divided_into_groups_l154_154056

theorem shawn_divided_into_groups :
  ∀ (total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups : ℕ),
  total_pebbles = 40 →
  red_pebbles = 9 →
  blue_pebbles = 13 →
  remaining_pebbles = total_pebbles - red_pebbles - blue_pebbles →
  remaining_pebbles % 3 = 0 →
  yellow_pebbles = blue_pebbles - 7 →
  remaining_pebbles = groups * yellow_pebbles →
  groups = 3 :=
by
  intros total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups
  intros h_total h_red h_blue h_remaining h_divisible h_yellow h_group
  sorry

end shawn_divided_into_groups_l154_154056


namespace rowing_speed_in_still_water_l154_154478

theorem rowing_speed_in_still_water (v c : ℝ) (t : ℝ) (h1 : c = 1.1) (h2 : (v + c) * t = (v - c) * 2 * t) : v = 3.3 :=
sorry

end rowing_speed_in_still_water_l154_154478


namespace smallest_angle_of_triangle_l154_154841

theorem smallest_angle_of_triangle (k : ℕ) (h : 4 * k + 5 * k + 9 * k = 180) : 4 * k = 40 :=
by {
  sorry
}

end smallest_angle_of_triangle_l154_154841


namespace total_spent_is_49_l154_154808

-- Define the prices of items
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define Paula's purchases
def paula_bracelets := 3
def paula_keychains := 2
def paula_coloring_book := 1
def paula_stickers := 4

-- Define Olive's purchases
def olive_bracelets := 2
def olive_coloring_book := 1
def olive_toy_car := 1
def olive_stickers := 3

-- Calculate total expenses
def paula_total := paula_bracelets * price_bracelet + paula_keychains * price_keychain + paula_coloring_book * price_coloring_book + paula_stickers * price_sticker
def olive_total := olive_coloring_book * price_coloring_book + olive_bracelets * price_bracelet + olive_toy_car * price_toy_car + olive_stickers * price_sticker
def total_expense := paula_total + olive_total

-- Prove the total expenses amount to $49
theorem total_spent_is_49 : total_expense = 49 :=
by
  have : paula_total = (3 * 4) + (2 * 5) + (1 * 3) + (4 * 1) := rfl
  have : olive_total = (1 * 3) + (2 * 4) + (1 *6) + (3 * 1) := rfl
  have : paula_total = 29 := rfl
  have : olive_total = 20 := rfl
  have : total_expense = 29 + 20 := rfl
  exact rfl

end total_spent_is_49_l154_154808


namespace unique_x_floor_eq_20_7_l154_154234

theorem unique_x_floor_eq_20_7 : ∀ x : ℝ, (⌊x⌋ + x + 1/2 = 20.7) → x = 10.2 :=
by
  sorry

end unique_x_floor_eq_20_7_l154_154234


namespace slope_of_tangent_line_at_x_2_l154_154679

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3*x

theorem slope_of_tangent_line_at_x_2 : (deriv curve 2) = 7 := by
  sorry

end slope_of_tangent_line_at_x_2_l154_154679


namespace quadratic_specific_a_l154_154272

noncomputable def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, (a + 2) * x^2 + 2 * a * x + 1 = 0

theorem quadratic_specific_a (a : ℝ) (h : quadratic_root_condition a) :
  a = 2 ∨ a = -1 :=
sorry

end quadratic_specific_a_l154_154272


namespace interior_sum_nine_l154_154307

-- Defining the function for the sum of the interior numbers in the nth row of Pascal's Triangle
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

-- Given conditions
axiom interior_sum_4 : interior_sum 4 = 6
axiom interior_sum_5 : interior_sum 5 = 14

-- Goal to prove
theorem interior_sum_nine : interior_sum 9 = 254 := by
  sorry

end interior_sum_nine_l154_154307


namespace tan_30_eq_sqrt3_div3_l154_154860

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l154_154860


namespace find_middle_number_l154_154258

theorem find_middle_number (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 12.5)
  (h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h3 : (a + b + c) / 3 = 11.6)
  (h4 : (c + d + e) / 3 = 13.5) : c = 12.8 :=
sorry

end find_middle_number_l154_154258


namespace coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l154_154058

/- The problem's conditions and questions rephrased for Lean:
  1. Prove: if \( n \) is divisible by 3, then a valid coloring is possible.
  2. Prove: if a valid coloring is possible, then \( n \) is divisible by 3.
-/

def is_colorable (n : ℕ) : Prop :=
  ∃ (colors : Fin 3 → Fin n → Fin 3),
    ∀ (i j : Fin n), i ≠ j → (colors 0 i ≠ colors 0 j ∧ colors 1 i ≠ colors 1 j ∧ colors 2 i ≠ colors 2 j)

theorem coloring_possible_if_divisible_by_three (n : ℕ) (h : n % 3 = 0) : is_colorable n :=
  sorry

theorem divisible_by_three_if_coloring_possible (n : ℕ) (h : is_colorable n) : n % 3 = 0 :=
  sorry

end coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l154_154058


namespace distance_between_foci_of_hyperbola_is_correct_l154_154738

noncomputable def distance_between_foci_of_hyperbola : ℝ := 
  let a_sq := 50
  let b_sq := 8
  let c_sq := a_sq + b_sq
  let c := Real.sqrt c_sq
  2 * c

theorem distance_between_foci_of_hyperbola_is_correct :
  distance_between_foci_of_hyperbola = 2 * Real.sqrt 58 :=
by
  sorry

end distance_between_foci_of_hyperbola_is_correct_l154_154738


namespace circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l154_154415

-- Define radii of the circles
def r1 : ℝ := 3
def r2 : ℝ := 5

-- Statement for first scenario (distance = 9)
theorem circles_do_not_intersect_first_scenario (d : ℝ) (h : d = 9) : ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

-- Statement for second scenario (distance = 1)
theorem circles_do_not_intersect_second_scenario (d : ℝ) (h : d = 1) : d < |r1 - r2| ∨ ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

end circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l154_154415


namespace find_r_condition_l154_154801

variable {x y z w r : ℝ}

axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : z ≠ 0
axiom h4 : w ≠ 0
axiom h5 : (x ≠ y) ∧ (x ≠ z) ∧ (x ≠ w) ∧ (y ≠ z) ∧ (y ≠ w) ∧ (z ≠ w)

noncomputable def is_geometric_progression (a b c d : ℝ) (r : ℝ) : Prop :=
  b = a * r ∧ c = a * r^2 ∧ d = a * r^3

theorem find_r_condition :
  is_geometric_progression (x * (y - z)) (y * (z - x)) (z * (x - y)) (w * (y - x)) r →
  r^3 + r^2 + r + 1 = 0 :=
by
  intros
  sorry

end find_r_condition_l154_154801


namespace total_garbage_collected_l154_154952

def Daliah := 17.5
def Dewei := Daliah - 2
def Zane := 4 * Dewei
def Bela := Zane + 3.75

theorem total_garbage_collected :
  Daliah + Dewei + Zane + Bela = 160.75 :=
by
  sorry

end total_garbage_collected_l154_154952


namespace find_B_values_l154_154164

theorem find_B_values (A B : ℤ) (h1 : 800 < A) (h2 : A < 1300) (h3 : B > 1) (h4 : A = B ^ 4) : B = 5 ∨ B = 6 := 
sorry

end find_B_values_l154_154164


namespace least_z_minus_x_l154_154208

theorem least_z_minus_x (x y z : ℤ) (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) : z - x = 7 :=
sorry

end least_z_minus_x_l154_154208


namespace circumference_to_diameter_ratio_l154_154472

-- Definitions from the conditions
def r : ℝ := 15
def C : ℝ := 90
def D : ℝ := 2 * r

-- The proof goal
theorem circumference_to_diameter_ratio : C / D = 3 := 
by sorry

end circumference_to_diameter_ratio_l154_154472


namespace inequality_proof_l154_154907

open Real

theorem inequality_proof (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l154_154907


namespace construct_rectangle_l154_154341

-- Define the essential properties of the rectangles
structure Rectangle where
  length : ℕ
  width : ℕ 

-- Define the given rectangles
def r1 : Rectangle := ⟨7, 1⟩
def r2 : Rectangle := ⟨6, 1⟩
def r3 : Rectangle := ⟨5, 1⟩
def r4 : Rectangle := ⟨4, 1⟩
def r5 : Rectangle := ⟨3, 1⟩
def r6 : Rectangle := ⟨2, 1⟩
def s  : Rectangle := ⟨1, 1⟩

-- Hypothesis for condition that length of each side of resulting rectangle should be > 1
def validSide (rect : Rectangle) : Prop :=
  rect.length > 1 ∧ rect.width > 1

-- The proof statement
theorem construct_rectangle : 
  (∃ rect1 rect2 rect3 rect4 : Rectangle, 
      rect1 = ⟨7, 1⟩ ∧ rect2 = ⟨6, 1⟩ ∧ rect3 = ⟨5, 1⟩ ∧ rect4 = ⟨4, 1⟩) →
  (∃ rect5 rect6 : Rectangle, 
      rect5 = ⟨3, 1⟩ ∧ rect6 = ⟨2, 1⟩) →
  (∃ square : Rectangle, 
      square = ⟨1, 1⟩) →
  (∃ compositeRect : Rectangle, 
      compositeRect.length = 7 ∧ 
      compositeRect.width = 4 ∧ 
      validSide compositeRect) :=
sorry

end construct_rectangle_l154_154341


namespace sqrt_expression_equal_cos_half_theta_l154_154006

noncomputable def sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta (θ : Real) : Real :=
  Real.sqrt (1 / 2 + 1 / 2 * Real.sqrt (1 / 2 + 1 / 2 * Real.cos (2 * θ))) - Real.sqrt (1 - Real.sin θ)

theorem sqrt_expression_equal_cos_half_theta (θ : Real) (h : π < θ) (h2 : θ < 3 * π / 2)
  (h3 : Real.cos θ < 0) (h4 : 0 < Real.sin (θ / 2)) (h5 : Real.cos (θ / 2) < 0) :
  sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta θ = Real.cos (θ / 2) :=
by
  sorry

end sqrt_expression_equal_cos_half_theta_l154_154006


namespace quadratic_roots_eq_l154_154846

theorem quadratic_roots_eq (a : ℝ) (b : ℝ) :
  (∀ x, (2 * x^2 - 3 * x - 8 = 0) → 
         ((x + 3)^2 + a * (x + 3) + b = 0)) → 
  b = 9.5 :=
by
  sorry

end quadratic_roots_eq_l154_154846


namespace train_length_is_correct_l154_154110

noncomputable def lengthOfTrain (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_s

theorem train_length_is_correct : lengthOfTrain 60 15 = 250.05 :=
by
  sorry

end train_length_is_correct_l154_154110


namespace balls_left_correct_l154_154670

def initial_balls : ℕ := 10
def balls_removed : ℕ := 3
def balls_left : ℕ := initial_balls - balls_removed

theorem balls_left_correct : balls_left = 7 := 
by
  -- Proof omitted
  sorry

end balls_left_correct_l154_154670


namespace ramu_selling_price_l154_154645

theorem ramu_selling_price (P R : ℝ) (profit_percent : ℝ) 
  (P_def : P = 42000)
  (R_def : R = 13000)
  (profit_percent_def : profit_percent = 17.272727272727273) :
  let total_cost := P + R
  let selling_price := total_cost * (1 + (profit_percent / 100))
  selling_price = 64500 := 
by
  sorry

end ramu_selling_price_l154_154645


namespace greatest_integer_solution_l154_154819

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 36 ≤ 0) : n ≤ 9 :=
by
  sorry

end greatest_integer_solution_l154_154819


namespace f_2011_l154_154507

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_defined_segment : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_2011 : f 2011 = -2 := by
  sorry

end f_2011_l154_154507


namespace hike_duration_l154_154490

def initial_water := 11
def final_water := 2
def leak_rate := 1
def water_drunk := 6

theorem hike_duration (time_hours : ℕ) :
  initial_water - final_water = water_drunk + time_hours * leak_rate →
  time_hours = 3 :=
by intro h; sorry

end hike_duration_l154_154490


namespace shopkeeper_loss_percent_l154_154088

theorem shopkeeper_loss_percent 
  (C : ℝ) (P : ℝ) (L : ℝ) 
  (hC : C = 100) 
  (hP : P = 10) 
  (hL : L = 50) : 
  ((C - (((C * (1 - L / 100)) * (1 + P / 100))) / C) * 100) = 45 :=
by
  sorry

end shopkeeper_loss_percent_l154_154088


namespace expression_max_value_l154_154642

open Real

theorem expression_max_value (x : ℝ) : ∃ M, M = 1/7 ∧ (∀ y : ℝ, y = x -> (y^3) / (y^6 + y^4 + y^3 - 3*y^2 + 9) ≤ M) :=
sorry

end expression_max_value_l154_154642


namespace exists_n_sum_digits_n3_eq_million_l154_154730

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem exists_n_sum_digits_n3_eq_million :
  ∃ n : ℕ, sum_digits n = 100 ∧ sum_digits (n ^ 3) = 1000000 := sorry

end exists_n_sum_digits_n3_eq_million_l154_154730


namespace plants_per_row_l154_154473

-- Define the conditions from the problem
def rows : ℕ := 7
def extra_plants : ℕ := 15
def total_plants : ℕ := 141

-- Define the problem statement to prove
theorem plants_per_row :
  ∃ x : ℕ, rows * x + extra_plants = total_plants ∧ x = 18 :=
by
  sorry

end plants_per_row_l154_154473


namespace nancy_coffee_expense_l154_154714

-- Definitions corresponding to the conditions
def cost_double_espresso : ℝ := 3.00
def cost_iced_coffee : ℝ := 2.50
def days : ℕ := 20

-- The statement of the problem
theorem nancy_coffee_expense :
  (days * (cost_double_espresso + cost_iced_coffee)) = 110.00 := by
  sorry

end nancy_coffee_expense_l154_154714


namespace proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l154_154703

structure CubeSymmetry where
  planes : Nat
  axes : Nat
  has_center : Bool

def general_cube_symmetry : CubeSymmetry :=
  { planes := 9, axes := 9, has_center := true }

def case_a : CubeSymmetry :=
  { planes := 4, axes := 1, has_center := false }

def case_b1 : CubeSymmetry :=
  { planes := 5, axes := 3, has_center := true }

def case_b2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

def case_c1 : CubeSymmetry :=
  { planes := 3, axes := 0, has_center := false }

def case_c2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

theorem proof_case_a : case_a = { planes := 4, axes := 1, has_center := false } := by
  sorry

theorem proof_case_b1 : case_b1 = { planes := 5, axes := 3, has_center := true } := by
  sorry

theorem proof_case_b2 : case_b2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

theorem proof_case_c1 : case_c1 = { planes := 3, axes := 0, has_center := false } := by
  sorry

theorem proof_case_c2 : case_c2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

end proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l154_154703


namespace rectangular_field_diagonal_length_l154_154267

noncomputable def diagonal_length_of_rectangular_field (a : ℝ) (A : ℝ) : ℝ :=
  let b := A / a
  let d := Real.sqrt (a^2 + b^2)
  d

theorem rectangular_field_diagonal_length :
  let a : ℝ := 14
  let A : ℝ := 135.01111065390137
  abs (diagonal_length_of_rectangular_field a A - 17.002) < 0.001 := by
    sorry

end rectangular_field_diagonal_length_l154_154267


namespace volume_of_prism_l154_154007

theorem volume_of_prism (a b c : ℝ)
  (h_ab : a * b = 36)
  (h_ac : a * c = 54)
  (h_bc : b * c = 72) :
  a * b * c = 648 :=
by
  sorry

end volume_of_prism_l154_154007


namespace volume_of_cube_is_correct_surface_area_of_cube_is_correct_l154_154785

-- Define the conditions: total edge length of the cube frame
def total_edge_length : ℕ := 60
def number_of_edges : ℕ := 12

-- Define the edge length of the cube
def edge_length (total_edge_length number_of_edges : ℕ) : ℕ := total_edge_length / number_of_edges

-- Define the volume of the cube
def cube_volume (a : ℕ) : ℕ := a ^ 3

-- Define the surface area of the cube
def cube_surface_area (a : ℕ) : ℕ := 6 * (a ^ 2)

-- Volume Proof Statement
theorem volume_of_cube_is_correct : cube_volume (edge_length total_edge_length number_of_edges) = 125 :=
by
  sorry

-- Surface Area Proof Statement
theorem surface_area_of_cube_is_correct : cube_surface_area (edge_length total_edge_length number_of_edges) = 150 :=
by
  sorry

end volume_of_cube_is_correct_surface_area_of_cube_is_correct_l154_154785


namespace zach_needs_more_tickets_l154_154379

theorem zach_needs_more_tickets {ferris_wheel_tickets roller_coaster_tickets log_ride_tickets zach_tickets : ℕ} :
  ferris_wheel_tickets = 2 ∧
  roller_coaster_tickets = 7 ∧
  log_ride_tickets = 1 ∧
  zach_tickets = 1 →
  (ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - zach_tickets = 9) :=
by
  intro h
  sorry

end zach_needs_more_tickets_l154_154379


namespace percentage_less_than_a_plus_d_l154_154010

-- Define the mean, standard deviation, and given conditions
variables (a d : ℝ)
axiom symmetric_distribution : ∀ x, x = 2 * a - x 

-- Main theorem
theorem percentage_less_than_a_plus_d :
  (∃ (P_less_than : ℝ → ℝ), P_less_than (a + d) = 0.84) :=
sorry

end percentage_less_than_a_plus_d_l154_154010


namespace min_value_expression_l154_154771

theorem min_value_expression : ∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 = 2 := 
sorry

end min_value_expression_l154_154771


namespace solve_equation1_solve_equation2_l154_154222

theorem solve_equation1 (x : ℝ) (h1 : 5 * x - 2 * (x - 1) = 3) : x = 1 / 3 := 
sorry

theorem solve_equation2 (x : ℝ) (h2 : (x + 3) / 2 - 1 = (2 * x - 1) / 3) : x = 5 :=
sorry

end solve_equation1_solve_equation2_l154_154222


namespace shopkeepers_total_profit_percentage_l154_154795

noncomputable def calculateProfitPercentage : ℝ :=
  let oranges := 1000
  let bananas := 800
  let apples := 750
  let rotten_oranges_percentage := 0.12
  let rotten_bananas_percentage := 0.05
  let rotten_apples_percentage := 0.10
  let profit_oranges_percentage := 0.20
  let profit_bananas_percentage := 0.25
  let profit_apples_percentage := 0.15
  let cost_per_orange := 2.5
  let cost_per_banana := 1.5
  let cost_per_apple := 2.0

  let rotten_oranges := rotten_oranges_percentage * oranges
  let rotten_bananas := rotten_bananas_percentage * bananas
  let rotten_apples := rotten_apples_percentage * apples

  let good_oranges := oranges - rotten_oranges
  let good_bananas := bananas - rotten_bananas
  let good_apples := apples - rotten_apples

  let cost_oranges := cost_per_orange * oranges
  let cost_bananas := cost_per_banana * bananas
  let cost_apples := cost_per_apple * apples

  let total_cost := cost_oranges + cost_bananas + cost_apples

  let selling_price_oranges := cost_per_orange * (1 + profit_oranges_percentage) * good_oranges
  let selling_price_bananas := cost_per_banana * (1 + profit_bananas_percentage) * good_bananas
  let selling_price_apples := cost_per_apple * (1 + profit_apples_percentage) * good_apples

  let total_selling_price := selling_price_oranges + selling_price_bananas + selling_price_apples

  let total_profit := total_selling_price - total_cost

  (total_profit / total_cost) * 100

theorem shopkeepers_total_profit_percentage :
  calculateProfitPercentage = 8.03 := sorry

end shopkeepers_total_profit_percentage_l154_154795


namespace expression_evaluates_to_one_l154_154735

theorem expression_evaluates_to_one :
  (1 / 3)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) + (Real.pi - 2016)^0 - (8:ℝ)^(1/3) = 1 :=
by
  -- step-by-step simplification skipped, as per requirements
  sorry

end expression_evaluates_to_one_l154_154735


namespace arrangement_count_BANANA_l154_154588

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l154_154588


namespace tree_height_equation_l154_154528

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end tree_height_equation_l154_154528


namespace product_of_a_values_l154_154977

/--
Let a be a real number and consider the points P = (3 * a, a - 5) and Q = (5, -2).
Given that the distance between P and Q is 3 * sqrt 10, prove that the product
of all possible values of a is -28 / 5.
-/
theorem product_of_a_values :
  ∀ (a : ℝ),
  (dist (3 * a, a - 5) (5, -2) = 3 * Real.sqrt 10) →
  ∃ (a₁ a₂ : ℝ), (5 * a₁ * a₁ - 18 * a₁ - 28 = 0) ∧ 
                 (5 * a₂ * a₂ - 18 * a₂ - 28 = 0) ∧ 
                 (a₁ * a₂ = -28 / 5) := 
by
  sorry

end product_of_a_values_l154_154977


namespace max_set_size_divisible_diff_l154_154259

theorem max_set_size_divisible_diff (S : Finset ℕ) (h1 : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → (5 ∣ (x - y) ∨ 25 ∣ (x - y))) : S.card ≤ 25 :=
sorry

end max_set_size_divisible_diff_l154_154259


namespace goldfish_equal_months_l154_154629

theorem goldfish_equal_months :
  ∃ (n : ℕ), 
    let B_n := 3 * 3^n 
    let G_n := 125 * 5^n 
    B_n = G_n ∧ n = 5 :=
by
  sorry

end goldfish_equal_months_l154_154629


namespace ant_minimum_distance_l154_154249

section
variables (x y z w u : ℝ)

-- Given conditions
axiom h1 : x + y + z = 22
axiom h2 : w + y + z = 29
axiom h3 : x + y + u = 30

-- Prove the ant crawls at least 47 cm to cover all paths
theorem ant_minimum_distance : x + y + z + w ≥ 47 :=
sorry
end

end ant_minimum_distance_l154_154249


namespace number_of_measures_of_C_l154_154609

theorem number_of_measures_of_C (C D : ℕ) (h1 : C + D = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ C = k * D) : 
  ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_measures_of_C_l154_154609


namespace pirate_coins_total_l154_154303

theorem pirate_coins_total (x : ℕ) (hx : x ≠ 0) (h_paul : ∃ k : ℕ, k = x / 2) (h_pete : ∃ m : ℕ, m = 5 * (x / 2)) 
  (h_ratio : (m : ℝ) = (k : ℝ) * 5) : (x = 4) → 
  ∃ total : ℕ, total = k + m ∧ total = 12 :=
by {
  sorry
}

end pirate_coins_total_l154_154303


namespace sequence_a3_equals_1_over_3_l154_154539

theorem sequence_a3_equals_1_over_3 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = 1 - 1 / (a (n - 1) + 1)) : 
  a 3 = 1 / 3 :=
sorry

end sequence_a3_equals_1_over_3_l154_154539


namespace sin_cos_theta_l154_154780

open Real

theorem sin_cos_theta (θ : ℝ) (H1 : θ > π / 2 ∧ θ < π) (H2 : tan (θ + π / 4) = 1 / 2) :
  sin θ + cos θ = -sqrt 10 / 5 :=
by
  sorry

end sin_cos_theta_l154_154780


namespace equivalent_octal_to_decimal_l154_154439

def octal_to_decimal (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n+1 => (n % 10) + 8 * octal_to_decimal (n / 10)

theorem equivalent_octal_to_decimal : octal_to_decimal 753 = 491 :=
by
  sorry

end equivalent_octal_to_decimal_l154_154439


namespace permutations_of_BANANA_l154_154818

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l154_154818


namespace a_2013_is_4_l154_154776

theorem a_2013_is_4
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n : ℕ, a (n+2) = (a n * a (n+1)) % 10) :
  a 2013 = 4 :=
sorry

end a_2013_is_4_l154_154776


namespace correlational_relationships_l154_154709

-- Definitions of relationships
def learning_attitude_and_academic_performance := "The relationship between a student's learning attitude and their academic performance"
def teacher_quality_and_student_performance := "The relationship between a teacher's teaching quality and students' academic performance"
def student_height_and_academic_performance := "The relationship between a student's height and their academic performance"
def family_economic_conditions_and_performance := "The relationship between family economic conditions and students' academic performance"

-- Definition of a correlational relationship
def correlational_relationship (relation : String) : Prop :=
  relation = learning_attitude_and_academic_performance ∨
  relation = teacher_quality_and_student_performance

-- Problem statement to prove
theorem correlational_relationships :
  correlational_relationship learning_attitude_and_academic_performance ∧ 
  correlational_relationship teacher_quality_and_student_performance :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end correlational_relationships_l154_154709


namespace ellipse_condition_l154_154061

theorem ellipse_condition (k : ℝ) : 
  (k > 1 ↔ 
  (k - 1 > 0 ∧ k + 1 > 0 ∧ k - 1 ≠ k + 1)) :=
by sorry

end ellipse_condition_l154_154061


namespace negation_of_proposition_l154_154165

theorem negation_of_proposition :
  (¬ (∃ x₀ : ℝ, x₀ > 2 ∧ x₀^3 - 2 * x₀^2 < 0)) ↔ (∀ x : ℝ, x > 2 → x^3 - 2 * x^2 ≥ 0) := by
  sorry

end negation_of_proposition_l154_154165


namespace loop_condition_l154_154920

theorem loop_condition (b : ℕ) : (b = 10 ∧ ∀ n, b = 10 + 3 * n ∧ b < 16 → n + 1 = 16) → ∀ (condition : ℕ → Prop), condition b → b = 16 :=
by sorry

end loop_condition_l154_154920


namespace petya_friends_l154_154657

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l154_154657


namespace trig_expression_l154_154418

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end trig_expression_l154_154418


namespace max_correct_questions_l154_154964

theorem max_correct_questions (a b c : ℕ) (h1 : a + b + c = 60) (h2 : 3 * a - 2 * c = 126) : a ≤ 49 :=
sorry

end max_correct_questions_l154_154964


namespace average_speed_of_train_l154_154950

-- Definitions based on the conditions
def distance1 : ℝ := 325
def distance2 : ℝ := 470
def time1 : ℝ := 3.5
def time2 : ℝ := 4

-- Proof statement
theorem average_speed_of_train :
  (distance1 + distance2) / (time1 + time2) = 106 := 
by 
  sorry

end average_speed_of_train_l154_154950


namespace polynomial_strictly_monotonic_l154_154954

variable {P : ℝ → ℝ}

/-- The polynomial P(x) is such that the polynomials P(P(x)) and P(P(P(x))) are strictly monotonic 
on the entire real axis. Prove that P(x) is also strictly monotonic on the entire real axis. -/
theorem polynomial_strictly_monotonic
  (h1 : StrictMono (P ∘ P))
  (h2 : StrictMono (P ∘ P ∘ P)) :
  StrictMono P :=
sorry

end polynomial_strictly_monotonic_l154_154954


namespace arithmetic_sequence_angles_sum_l154_154345

theorem arithmetic_sequence_angles_sum (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : 2 * B = A + C) :
  A + C = 120 :=
by
  sorry

end arithmetic_sequence_angles_sum_l154_154345


namespace find_y_l154_154487

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y (y : ℝ) (h : star 7 y = 47) : y = 5.4 := 
by 
  sorry

end find_y_l154_154487


namespace cost_per_day_additional_weeks_l154_154198

theorem cost_per_day_additional_weeks :
  let first_week_days := 7
  let first_week_cost_per_day := 18.00
  let first_week_cost := first_week_days * first_week_cost_per_day
  let total_days := 23
  let total_cost := 302.00
  let additional_days := total_days - first_week_days
  let additional_cost := total_cost - first_week_cost
  let cost_per_day_additional := additional_cost / additional_days
  cost_per_day_additional = 11.00 :=
by
  sorry

end cost_per_day_additional_weeks_l154_154198


namespace proposition_induction_l154_154427

variable (P : ℕ → Prop)
variable (k : ℕ)

theorem proposition_induction (h : ∀ k : ℕ, P k → P (k + 1))
    (h9 : ¬ P 9) : ¬ P 8 :=
by
  sorry

end proposition_induction_l154_154427


namespace sequence_inequality_l154_154713

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 
  if n = 0 then 1/2
  else a (n - 1) + (1 / (n:ℚ)^2) * (a (n - 1))^2

theorem sequence_inequality (n : ℕ) : 
  1 - 1 / 2 ^ (n + 1) ≤ a n ∧ a n < 7 / 5 := 
sorry

end sequence_inequality_l154_154713


namespace solve_xyz_l154_154366

theorem solve_xyz (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) :
  (x / 21) * (y / 189) + z = 1 ↔ x = 21 ∧ y = 567 ∧ z = 0 :=
sorry

end solve_xyz_l154_154366


namespace probability_rain_once_l154_154172

theorem probability_rain_once (p : ℚ) 
  (h₁ : p = 1 / 2) 
  (h₂ : 1 - p = 1 / 2) 
  (h₃ : (1 - p) ^ 4 = 1 / 16) 
  : 1 - (1 - p) ^ 4 = 15 / 16 :=
by
  sorry

end probability_rain_once_l154_154172


namespace certain_events_l154_154101

-- Define the idioms and their classifications
inductive Event
| impossible
| certain
| unlikely

-- Definitions based on the given conditions
def scooping_moon := Event.impossible
def rising_tide := Event.certain
def waiting_by_stump := Event.unlikely
def catching_turtles := Event.certain
def pulling_seeds := Event.impossible

-- The theorem statement
theorem certain_events :
  (rising_tide = Event.certain) ∧ (catching_turtles = Event.certain) := by
  -- Proof is omitted
  sorry

end certain_events_l154_154101


namespace three_digit_numbers_sorted_desc_l154_154610

theorem three_digit_numbers_sorted_desc :
  ∃ n, n = 84 ∧
    ∀ (h t u : ℕ), 100 <= 100 * h + 10 * t + u ∧ 100 * h + 10 * t + u <= 999 →
    1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u → 
    n = 84 := 
by
  sorry

end three_digit_numbers_sorted_desc_l154_154610


namespace documentaries_count_l154_154308

def number_of_documents
  (novels comics albums crates capacity : ℕ)
  (total_items := crates * capacity)
  (known_items := novels + comics + albums)
  (documentaries := total_items - known_items) : ℕ :=
  documentaries

theorem documentaries_count
  : number_of_documents 145 271 209 116 9 = 419 :=
by
  sorry

end documentaries_count_l154_154308


namespace mod_squares_eq_one_l154_154750

theorem mod_squares_eq_one
  (n : ℕ)
  (h : n = 5)
  (a : ℤ)
  (ha : ∃ b : ℕ, ↑b = a ∧ b * b ≡ 1 [MOD 5]) :
  (a * a) % n = 1 :=
by
  sorry

end mod_squares_eq_one_l154_154750


namespace max_prime_p_l154_154850

-- Define the variables and conditions
variable (a b : ℕ)
variable (p : ℝ)

-- Define the prime condition
def is_prime (n : ℝ) : Prop := sorry -- Placeholder for the prime definition

-- Define the equation condition
def p_eq (p : ℝ) (a b : ℕ) : Prop := 
  p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))

-- The theorem to prove
theorem max_prime_p (a b : ℕ) (p_max : ℝ) :
  (∃ p, is_prime p ∧ p_eq p a b) → p_max = 5 := 
sorry

end max_prime_p_l154_154850


namespace n_not_composite_l154_154753

theorem n_not_composite
  (n : ℕ) (h1 : n > 1)
  (a : ℕ) (q : ℕ) (hq_prime : Nat.Prime q)
  (hq1 : q ∣ (n - 1))
  (hq2 : q > Nat.sqrt n - 1)
  (hn_div : n ∣ (a^(n-1) - 1))
  (hgcd : Nat.gcd (a^(n-1)/q - 1) n = 1) :
  ¬ Nat.Prime n :=
sorry

end n_not_composite_l154_154753


namespace margaret_time_l154_154210

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def total_permutations (n : Nat) : Nat :=
  factorial n

def total_time_in_minutes (total_permutations : Nat) (rate : Nat) : Nat :=
  total_permutations / rate

def time_in_hours_and_minutes (total_minutes : Nat) : Nat × Nat :=
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

theorem margaret_time :
  let n := 8
  let r := 15
  let permutations := total_permutations n
  let total_minutes := total_time_in_minutes permutations r
  time_in_hours_and_minutes total_minutes = (44, 48) := by
  sorry

end margaret_time_l154_154210


namespace parallel_lines_perpendicular_lines_l154_154892

section LineEquation

variables (a : ℝ) (x y : ℝ)

def l1 := (a-2) * x + 3 * y + a = 0
def l2 := a * x + (a-2) * y - 1 = 0

theorem parallel_lines (a : ℝ) :
  ((a-2)/a = 3/(a-2)) ↔ (a = (7 + Real.sqrt 33) / 2 ∨ a = (7 - Real.sqrt 33) / 2) := sorry

theorem perpendicular_lines (a : ℝ) :
  (a = 2 ∨ ((2-a)/3 * (a/(2-a)) = -1)) ↔ (a = 2 ∨ a = -3) := sorry

end LineEquation

end parallel_lines_perpendicular_lines_l154_154892


namespace reciprocal_sum_neg_l154_154784

theorem reciprocal_sum_neg (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 8) : (1/a) + (1/b) + (1/c) < 0 := 
sorry

end reciprocal_sum_neg_l154_154784


namespace minimum_combined_horses_ponies_l154_154495

noncomputable def ranch_min_total (P H : ℕ) : ℕ :=
  P + H

theorem minimum_combined_horses_ponies (P H : ℕ) 
  (h1 : ∃ k : ℕ, P = 16 * k ∧ k ≥ 1)
  (h2 : H = P + 3) 
  (h3 : P = 80) 
  (h4 : H = 83) :
  ranch_min_total P H = 163 :=
by
  sorry

end minimum_combined_horses_ponies_l154_154495


namespace compute_expression_l154_154407

noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

theorem compute_expression : 5^(c / d) + 2^(d / c) = 2 * Real.sqrt 2 + 5^(2 / 3) :=
by
  sorry

end compute_expression_l154_154407


namespace number_of_solutions_l154_154953

theorem number_of_solutions :
  ∃ S : Finset (ℤ × ℤ), 
  (∀ (m n : ℤ), (m, n) ∈ S ↔ m^4 + 8 * n^2 + 425 = n^4 + 42 * m^2) ∧ 
  S.card = 16 :=
by { sorry }

end number_of_solutions_l154_154953


namespace integer_roots_of_polynomial_l154_154518

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end integer_roots_of_polynomial_l154_154518


namespace minimum_days_l154_154357

theorem minimum_days (n : ℕ) (rain_afternoon : ℕ) (sunny_afternoon : ℕ) (sunny_morning : ℕ) :
  rain_afternoon + sunny_afternoon = 7 ∧
  sunny_afternoon <= 5 ∧
  sunny_morning <= 6 ∧
  sunny_morning + rain_afternoon = 7 ∧
  n = 11 :=
by
  sorry

end minimum_days_l154_154357


namespace solve_inequalities_l154_154867

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l154_154867


namespace smallest_n_l154_154626

-- Definitions for arithmetic sequences with given conditions
def arithmetic_sequence_a (n : ℕ) (x : ℕ) : ℕ := 1 + (n-1) * x
def arithmetic_sequence_b (n : ℕ) (y : ℕ) : ℕ := 1 + (n-1) * y

-- Main theorem statement
theorem smallest_n (x y n : ℕ) (hxy : x < y) (ha1 : arithmetic_sequence_a 1 x = 1) (hb1 : arithmetic_sequence_b 1 y = 1) 
  (h_sum : arithmetic_sequence_a n x + arithmetic_sequence_b n y = 2556) : n = 3 :=
sorry

end smallest_n_l154_154626


namespace convex_polygon_interior_angle_l154_154421

theorem convex_polygon_interior_angle (n : ℕ) (h1 : 3 ≤ n)
  (h2 : (n - 2) * 180 = 2570 + x) : x = 130 :=
sorry

end convex_polygon_interior_angle_l154_154421


namespace triangular_pyramid_volume_l154_154541

theorem triangular_pyramid_volume
  (b : ℝ) (h : ℝ) (H : ℝ)
  (b_pos : b = 4.5) (h_pos : h = 6) (H_pos : H = 8) :
  let base_area := (b * h) / 2
  let volume := (base_area * H) / 3
  volume = 36 := by
  sorry

end triangular_pyramid_volume_l154_154541


namespace divisibility_condition_l154_154231

theorem divisibility_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ab ∣ (a^2 + b^2 - a - b + 1) → (a = 1 ∧ b = 1) :=
by sorry

end divisibility_condition_l154_154231


namespace min_cut_length_l154_154046

theorem min_cut_length (x : ℝ) (h_longer : 23 - x ≥ 0) (h_shorter : 15 - x ≥ 0) :
  23 - x ≥ 2 * (15 - x) → x ≥ 7 :=
by
  sorry

end min_cut_length_l154_154046


namespace negation_of_implication_l154_154343

theorem negation_of_implication {r p q : Prop} :
  ¬ (r → (p ∨ q)) ↔ (¬ r → (¬ p ∧ ¬ q)) :=
by sorry

end negation_of_implication_l154_154343


namespace proof_emails_in_morning_l154_154949

def emailsInAfternoon : ℕ := 2

def emailsMoreInMorning : ℕ := 4

def emailsInMorning : ℕ := 6

theorem proof_emails_in_morning
  (a : ℕ) (h1 : a = emailsInAfternoon)
  (m : ℕ) (h2 : m = emailsMoreInMorning)
  : emailsInMorning = a + m := by
  sorry

end proof_emails_in_morning_l154_154949


namespace increasing_function_l154_154423

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) (h : a ≥ 1) : 
  ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
by 
  sorry

end increasing_function_l154_154423


namespace solve_system_l154_154826

-- The system of equations as conditions in Lean
def system1 (x y : ℤ) : Prop := 5 * x + 2 * y = 25
def system2 (x y : ℤ) : Prop := 3 * x + 4 * y = 15

-- The statement that asserts the solution is (x = 5, y = 0)
theorem solve_system : ∃ x y : ℤ, system1 x y ∧ system2 x y ∧ x = 5 ∧ y = 0 :=
by
  sorry

end solve_system_l154_154826


namespace inequality_solution_l154_154760

theorem inequality_solution (x : ℝ) : (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 :=
by
  sorry

end inequality_solution_l154_154760


namespace sequence_correctness_l154_154510

def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -2
  else -(2^(n - 1))

def partial_sum_S (n : ℕ) : ℤ := -2^n

theorem sequence_correctness (n : ℕ) (h : n ≥ 1) :
  (sequence_a 1 = -2) ∧ (∀ n ≥ 2, sequence_a (n + 1) = partial_sum_S n) ∧
  (sequence_a n = -(2^(n - 1))) ∧ (partial_sum_S n = -2^n) :=
by
  sorry

end sequence_correctness_l154_154510


namespace total_guests_l154_154183

-- Define the conditions.
def number_of_tables := 252.0
def guests_per_table := 4.0

-- Define the statement to prove.
theorem total_guests : number_of_tables * guests_per_table = 1008.0 := by
  sorry

end total_guests_l154_154183


namespace ladder_cost_l154_154291

theorem ladder_cost (ladders1 ladders2 rung_count1 rung_count2 cost_per_rung : ℕ)
  (h1 : ladders1 = 10) (h2 : ladders2 = 20) (h3 : rung_count1 = 50) (h4 : rung_count2 = 60) (h5 : cost_per_rung = 2) :
  (ladders1 * rung_count1 + ladders2 * rung_count2) * cost_per_rung = 3400 :=
by 
  sorry

end ladder_cost_l154_154291


namespace lattice_points_distance_5_l154_154083

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem lattice_points_distance_5 : 
  ∃ S : Finset (ℤ × ℤ × ℤ), 
    (∀ p ∈ S, is_lattice_point p.1 p.2.1 p.2.2) ∧
    S.card = 78 :=
by
  sorry

end lattice_points_distance_5_l154_154083


namespace range_of_a_intersection_l154_154817

theorem range_of_a_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x - 2 * k + 2 ∧ y = a * x^2 - 2 * a * x - 3 * a) ↔ (a ≤ -2/3 ∨ a > 0) := by
  sorry

end range_of_a_intersection_l154_154817


namespace ordered_pair_unique_l154_154519

theorem ordered_pair_unique (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x, y) = (1, 14) :=
by
  sorry

end ordered_pair_unique_l154_154519


namespace row_column_crossout_l154_154424

theorem row_column_crossout (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ rows : Finset (Fin 1000), rows.card = 990 ∧ ∀ j : Fin 1000, ∃ i ∈ rowsᶜ, M i j = 1) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 990 ∧ ∀ i : Fin 1000, ∃ j ∈ colsᶜ, M i j = 0) :=
by {
  sorry
}

end row_column_crossout_l154_154424


namespace average_marks_of_all_students_l154_154855

/-
Consider two classes:
- The first class has 12 students with an average mark of 40.
- The second class has 28 students with an average mark of 60.

We are to prove that the average marks of all students from both classes combined is 54.
-/

theorem average_marks_of_all_students (s1 s2 : ℕ) (m1 m2 : ℤ)
  (h1 : s1 = 12) (h2 : m1 = 40) (h3 : s2 = 28) (h4 : m2 = 60) :
  (s1 * m1 + s2 * m2) / (s1 + s2) = 54 :=
by
  rw [h1, h2, h3, h4]
  sorry

end average_marks_of_all_students_l154_154855


namespace value_of_f_at_4_l154_154674

noncomputable def f (x : ℝ) (c : ℝ) (d : ℝ) : ℝ :=
  c * x ^ 2 + d * x + 3

theorem value_of_f_at_4 :
  (∃ c d : ℝ, f 1 c d = 3 ∧ f 2 c d = 5) → f 4 1 (-1) = 15 :=
by
  sorry

end value_of_f_at_4_l154_154674


namespace increasing_function_on_interval_l154_154511

section
  variable (a b : ℝ)
  def f (x : ℝ) : ℝ := |x^2 - 2*a*x + b|

  theorem increasing_function_on_interval (h : a^2 - b ≤ 0) :
    ∀ x y : ℝ, a ≤ x → x ≤ y → f x ≤ f y := 
  sorry
end

end increasing_function_on_interval_l154_154511


namespace geometric_sequence_condition_l154_154540

-- Given the sum of the first n terms of the sequence {a_n} is S_n = 2^n + c,
-- we need to prove that the sequence {a_n} is a geometric sequence if and only if c = -1.
theorem geometric_sequence_condition (c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n, S n = 2^n + c) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (∃ q, ∀ n ≥ 1, a n = a 1 * q ^ (n - 1)) ↔ (c = -1) :=
by
  -- Proof skipped
  sorry

end geometric_sequence_condition_l154_154540


namespace chloe_apples_l154_154128

theorem chloe_apples :
  ∃ x : ℕ, (∃ y : ℕ, x = y + 8 ∧ y = x / 3) ∧ x = 12 := 
by
  sorry

end chloe_apples_l154_154128


namespace mushrooms_collected_l154_154390

variable (P V : ℕ)

theorem mushrooms_collected (h1 : P = (V * 100) / (P + V)) (h2 : V % 2 = 1) :
  P + V = 25 ∨ P + V = 300 ∨ P + V = 525 ∨ P + V = 1900 ∨ P + V = 9900 := by
  sorry

end mushrooms_collected_l154_154390


namespace steve_keeps_total_money_excluding_advance_l154_154529

-- Definitions of the conditions
def totalCopies : ℕ := 1000000
def advanceCopies : ℕ := 100000
def pricePerCopy : ℕ := 2
def agentCommissionRate : ℚ := 0.1

-- Question and final proof
theorem steve_keeps_total_money_excluding_advance :
  let totalEarnings := totalCopies * pricePerCopy
  let agentCommission := agentCommissionRate * totalEarnings
  let moneyKept := totalEarnings - agentCommission
  moneyKept = 1800000 := by
  -- Proof goes here, but we skip it for now
  sorry

end steve_keeps_total_money_excluding_advance_l154_154529


namespace m_range_positive_solution_l154_154316

theorem m_range_positive_solution (m : ℝ) : (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) := by
  sorry

end m_range_positive_solution_l154_154316


namespace smallest_sum_B_c_l154_154656

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end smallest_sum_B_c_l154_154656


namespace shoe_price_l154_154034

theorem shoe_price :
  ∀ (P : ℝ),
    (6 * P + 18 * 2 = 27 * 2) → P = 3 :=
by
  intro P H
  sorry

end shoe_price_l154_154034


namespace find_BC_line_eq_l154_154293

def line1_altitude : Prop := ∃ x y : ℝ, 2*x - 3*y + 1 = 0
def line2_altitude : Prop := ∃ x y : ℝ, x + y = 0
def vertex_A : Prop := ∃ a1 a2 : ℝ, a1 = 1 ∧ a2 = 2
def side_BC_equation : Prop := ∃ b c d : ℝ, b = 2 ∧ c = 3 ∧ d = 7

theorem find_BC_line_eq (H1 : line1_altitude) (H2 : line2_altitude) (H3 : vertex_A) : side_BC_equation :=
sorry

end find_BC_line_eq_l154_154293


namespace problem_solution_l154_154290

variables {R : Type} [LinearOrder R]

def M (x y : R) : R := max x y
def m (x y : R) : R := min x y

theorem problem_solution (p q r s t : R) (h : p < q) (h1 : q < r) (h2 : r < s) (h3 : s < t) :
  M (M p (m q r)) (m s (M p t)) = q :=
by
  sorry

end problem_solution_l154_154290


namespace sum_first_49_odd_numbers_l154_154064

theorem sum_first_49_odd_numbers : (49^2 = 2401) :=
by
  sorry

end sum_first_49_odd_numbers_l154_154064


namespace Kyle_older_than_Julian_l154_154530

variable (Tyson_age : ℕ)
variable (Frederick_age Julian_age Kyle_age : ℕ)

-- Conditions
def condition1 := Tyson_age = 20
def condition2 := Frederick_age = 2 * Tyson_age
def condition3 := Julian_age = Frederick_age - 20
def condition4 := Kyle_age = 25

-- The proof problem (statement only)
theorem Kyle_older_than_Julian :
  Tyson_age = 20 ∧
  Frederick_age = 2 * Tyson_age ∧
  Julian_age = Frederick_age - 20 ∧
  Kyle_age = 25 →
  Kyle_age - Julian_age = 5 := by
  intro h
  sorry

end Kyle_older_than_Julian_l154_154530


namespace pipe_c_empty_time_l154_154429

theorem pipe_c_empty_time (x : ℝ) :
  (4/20 + 4/30 + 4/x) * 3 = 1 → x = 6 :=
by
  sorry

end pipe_c_empty_time_l154_154429


namespace mark_sideline_time_l154_154350

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end mark_sideline_time_l154_154350


namespace schedule_problem_l154_154637

def num_schedule_ways : Nat :=
  -- total ways to pick 3 out of 6 periods and arrange 3 courses
  let total_ways := Nat.choose 6 3 * Nat.factorial 3
  -- at least two consecutive courses (using Principle of Inclusion and Exclusion)
  let two_consecutive := 5 * 6 * 4
  let three_consecutive := 4 * 6
  let invalid_ways := two_consecutive + three_consecutive
  total_ways - invalid_ways

theorem schedule_problem (h : num_schedule_ways = 24) : num_schedule_ways = 24 := by {
  exact h
}

end schedule_problem_l154_154637


namespace line_passing_through_M_l154_154339

-- Define the point M
def M : ℝ × ℝ := (-3, 4)

-- Define the predicate for a line equation having equal intercepts and passing through point M
def line_eq (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ c : ℝ, ((a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) 

theorem line_passing_through_M (x y : ℝ) (a b : ℝ) (h₀ : (-3, 4) = M) (h₁ : ∃ c : ℝ, (a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) :
  (4 * x + 3 * y = 0) ∨ (x + y = 1) :=
by
  -- We add 'sorry' to skip the proof
  sorry

end line_passing_through_M_l154_154339


namespace tan_15_eq_sqrt3_l154_154973

theorem tan_15_eq_sqrt3 :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
sorry

end tan_15_eq_sqrt3_l154_154973


namespace initial_blocks_l154_154112

-- Definitions of the given conditions
def blocks_eaten : ℕ := 29
def blocks_remaining : ℕ := 26

-- The statement we need to prove
theorem initial_blocks : blocks_eaten + blocks_remaining = 55 :=
by
  -- Proof is not required as per instructions
  sorry

end initial_blocks_l154_154112


namespace smallest_n_logarithm_l154_154824

theorem smallest_n_logarithm :
  ∃ n : ℕ, 0 < n ∧ 
  (Real.log (Real.log n / Real.log 3) / Real.log 3^2 =
  Real.log (Real.log n / Real.log 2) / Real.log 2^3) ∧ 
  n = 9 :=
by
  sorry

end smallest_n_logarithm_l154_154824


namespace percent_employed_in_town_l154_154933

theorem percent_employed_in_town (E : ℝ) : 
  (0.14 * E) + 55 = E → E = 64 :=
by
  intro h
  have h1: 0.14 * E + 55 = E := h
  -- Proof step here, but we put sorry to skip the proof
  sorry

end percent_employed_in_town_l154_154933


namespace math_proof_problem_l154_154353

noncomputable def a : ℝ := Real.sqrt 18
noncomputable def b : ℝ := (-1 / 3) ^ (-2 : ℤ)
noncomputable def c : ℝ := abs (-3 * Real.sqrt 2)
noncomputable def d : ℝ := (1 - Real.sqrt 2) ^ 0

theorem math_proof_problem : a - b - c - d = -10 := by
  -- Sorry is used to skip the proof, as the proof steps are not required for this problem.
  sorry

end math_proof_problem_l154_154353


namespace solve_system_eq_l154_154832

theorem solve_system_eq (x y : ℝ) (h1 : x - y = 1) (h2 : 2 * x + 3 * y = 7) :
  x = 2 ∧ y = 1 := by
  sorry

end solve_system_eq_l154_154832


namespace graveyard_bones_count_l154_154757

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end graveyard_bones_count_l154_154757


namespace all_tell_truth_at_same_time_l154_154287

-- Define the probabilities of each person telling the truth.
def prob_Alice := 0.7
def prob_Bob := 0.6
def prob_Carol := 0.8
def prob_David := 0.5

-- Prove that the probability that all four tell the truth at the same time is 0.168.
theorem all_tell_truth_at_same_time :
  prob_Alice * prob_Bob * prob_Carol * prob_David = 0.168 :=
by
  sorry

end all_tell_truth_at_same_time_l154_154287


namespace find_m_l154_154901

theorem find_m (m : ℤ) (h1 : m + 1 ≠ 0) (h2 : m^2 + 3 * m + 1 = -1) : m = -2 := 
by 
  sorry

end find_m_l154_154901


namespace fraction_sum_eq_neg_one_l154_154959

theorem fraction_sum_eq_neg_one (p q : ℝ) (hpq : (1 / p) + (1 / q) = (1 / (p + q))) :
  (p / q) + (q / p) = -1 :=
by
  sorry

end fraction_sum_eq_neg_one_l154_154959


namespace speed_of_second_train_l154_154320

theorem speed_of_second_train
  (t₁ : ℕ := 2)  -- Time the first train sets off (2:00 pm in hours)
  (s₁ : ℝ := 70) -- Speed of the first train in km/h
  (t₂ : ℕ := 3)  -- Time the second train sets off (3:00 pm in hours)
  (t₃ : ℕ := 10) -- Time when the second train catches the first train (10:00 pm in hours)
  : ∃ S : ℝ, S = 80 := sorry

end speed_of_second_train_l154_154320


namespace inversely_proportional_x_y_l154_154430

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end inversely_proportional_x_y_l154_154430


namespace functions_are_equal_l154_154302

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- Statement to be proven
theorem functions_are_equal : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_equal_l154_154302


namespace fraction_of_time_l154_154740

-- Define the time John takes to clean the entire house
def John_time : ℝ := 6

-- Define the combined time it takes Nick and John to clean the entire house
def combined_time : ℝ := 3.6

-- Given this configuration, we need to prove the fraction result.
theorem fraction_of_time (N : ℝ) (H1 : John_time = 6) (H2 : ∀ N, (1/John_time) + (1/N) = 1/combined_time) :
  (John_time / 2) / N = 1 / 3 := 
by sorry

end fraction_of_time_l154_154740


namespace find_sum_3xyz_l154_154589

variables (x y z : ℚ)

def equation1 : Prop := y + z = 18 - 4 * x
def equation2 : Prop := x + z = 16 - 4 * y
def equation3 : Prop := x + y = 9 - 4 * z

theorem find_sum_3xyz (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : 
  3 * x + 3 * y + 3 * z = 43 / 2 := 
sorry

end find_sum_3xyz_l154_154589


namespace number_of_perfect_square_factors_450_l154_154601

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def prime_factorization_450 := (2, 1) :: (3, 2) :: (5, 2) :: []

def perfect_square_factors (n : ℕ) : ℕ :=
  if n = 450 then 4 else 0

theorem number_of_perfect_square_factors_450 : perfect_square_factors 450 = 4 :=
by
  sorry

end number_of_perfect_square_factors_450_l154_154601


namespace gcd_45_75_eq_15_l154_154594

theorem gcd_45_75_eq_15 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_eq_15_l154_154594


namespace handshake_count_l154_154624

theorem handshake_count
  (total_people : ℕ := 40)
  (groupA_size : ℕ := 30)
  (groupB_size : ℕ := 10)
  (groupB_knowsA_5 : ℕ := 3)
  (groupB_knowsA_0 : ℕ := 7)
  (handshakes_between_A_and_B5 : ℕ := groupB_knowsA_5 * (groupA_size - 5))
  (handshakes_between_A_and_B0 : ℕ := groupB_knowsA_0 * groupA_size)
  (handshakes_within_B : ℕ := groupB_size * (groupB_size - 1) / 2) :
  handshakes_between_A_and_B5 + handshakes_between_A_and_B0 + handshakes_within_B = 330 :=
sorry

end handshake_count_l154_154624


namespace complement_of_union_in_U_l154_154161

-- Define the universal set U
def U : Set ℕ := {x | x < 6 ∧ x > 0}

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of A ∪ B in U
def complement_U_union_A_B : Set ℕ := {x | x ∈ U ∧ x ∉ (A ∪ B)}

theorem complement_of_union_in_U : complement_U_union_A_B = {2, 4} :=
by {
  -- Placeholder for the proof
  sorry
}

end complement_of_union_in_U_l154_154161


namespace mul_example_l154_154695

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end mul_example_l154_154695


namespace a_41_eq_6585451_l154_154958

noncomputable def a : ℕ → ℕ
| 0     => 0 /- Not used practically since n >= 1 -/
| 1     => 1
| 2     => 1
| 3     => 2
| (n+4) => a n + a (n+2) + 1

theorem a_41_eq_6585451 : a 41 = 6585451 := by
  sorry

end a_41_eq_6585451_l154_154958


namespace solve_equation_l154_154635

theorem solve_equation (x : ℚ) :
  (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by sorry

end solve_equation_l154_154635


namespace problem1_problem2_l154_154476

-- Problem 1
theorem problem1 : ((- (1/2) - (1/3) + (3/4)) * -60) = 5 :=
by
  -- The proof steps would go here
  sorry

-- Problem 2
theorem problem2 : ((-1)^4 - (1/6) * (3 - (-3)^2)) = 2 :=
by
  -- The proof steps would go here
  sorry

end problem1_problem2_l154_154476


namespace solve_simultaneous_equations_l154_154251

theorem solve_simultaneous_equations (a b : ℚ) : 
  (a + b) * (a^2 - b^2) = 4 ∧ (a - b) * (a^2 + b^2) = 5 / 2 → 
  (a = 3 / 2 ∧ b = 1 / 2) ∨ (a = -1 / 2 ∧ b = -3 / 2) :=
by
  sorry

end solve_simultaneous_equations_l154_154251


namespace frigate_catches_smuggler_at_five_l154_154185

noncomputable def time_to_catch : ℝ :=
  2 + (12 / 4) -- Initial leading distance / Relative speed before storm
  
theorem frigate_catches_smuggler_at_five 
  (initial_distance : ℝ)
  (frigate_speed_before_storm : ℝ)
  (smuggler_speed_before_storm : ℝ)
  (time_before_storm : ℝ)
  (frigate_speed_after_storm : ℝ)
  (smuggler_speed_after_storm : ℝ) :
  initial_distance = 12 →
  frigate_speed_before_storm = 14 →
  smuggler_speed_before_storm = 10 →
  time_before_storm = 3 →
  frigate_speed_after_storm = 12 →
  smuggler_speed_after_storm = 9 →
  time_to_catch = 5 :=
by
{
  sorry
}

end frigate_catches_smuggler_at_five_l154_154185


namespace maria_total_eggs_l154_154816

def total_eggs (boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  boxes * eggs_per_box

theorem maria_total_eggs :
  total_eggs 3 7 = 21 :=
by
  -- Here, you would normally show the steps of computation
  -- which we can skip with sorry
  sorry

end maria_total_eggs_l154_154816


namespace tom_distance_before_karen_wins_l154_154742

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end tom_distance_before_karen_wins_l154_154742


namespace diana_shopping_for_newborns_l154_154282

-- Define the conditions
def num_toddlers : ℕ := 6
def num_teenagers : ℕ := 5 * num_toddlers
def total_children : ℕ := 40

-- Define the problem statement
theorem diana_shopping_for_newborns : (total_children - (num_toddlers + num_teenagers)) = 4 := by
  sorry

end diana_shopping_for_newborns_l154_154282


namespace sam_collected_42_cans_l154_154321

noncomputable def total_cans_collected (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  bags_saturday + bags_sunday * cans_per_bag

theorem sam_collected_42_cans :
  total_cans_collected 4 3 6 = 42 :=
by
  sorry

end sam_collected_42_cans_l154_154321


namespace sandy_marks_per_correct_sum_l154_154619

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ)
  (total_marks : ℤ)
  (correct_sums : ℕ)
  (marks_per_incorrect_sum : ℤ)
  (marks_obtained : ℤ) 
  (marks_per_correct_sum : ℕ) :
  total_sums = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  marks_per_incorrect_sum = 2 →
  marks_obtained = total_marks →
  marks_obtained = marks_per_correct_sum * correct_sums - marks_per_incorrect_sum * (total_sums - correct_sums) → 
  marks_per_correct_sum = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sandy_marks_per_correct_sum_l154_154619


namespace total_earnings_correct_l154_154134

noncomputable def total_earnings (a_days b_days c_days b_share : ℝ) : ℝ :=
  let a_work_per_day := 1 / a_days
  let b_work_per_day := 1 / b_days
  let c_work_per_day := 1 / c_days
  let combined_work_per_day := a_work_per_day + b_work_per_day + c_work_per_day
  let b_fraction_of_total_work := b_work_per_day / combined_work_per_day
  let total_earnings := b_share / b_fraction_of_total_work
  total_earnings

theorem total_earnings_correct :
  total_earnings 6 8 12 780.0000000000001 = 2340 :=
by
  sorry

end total_earnings_correct_l154_154134


namespace sum_infinite_series_l154_154376

noncomputable def series_term (n : ℕ) : ℚ := 
  (2 * n + 3) / (n * (n + 1) * (n + 2))

noncomputable def partial_fractions (n : ℕ) : ℚ := 
  (3 / 2) / n - 1 / (n + 1) - (1 / 2) / (n + 2)

theorem sum_infinite_series : 
  (∑' n : ℕ, series_term (n + 1)) = 5 / 4 := 
by
  sorry

end sum_infinite_series_l154_154376


namespace find_expression_for_f_x_neg_l154_154991

theorem find_expression_for_f_x_neg (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x, 0 < x → f x = x - Real.log (abs x)) :
  ∀ x, x < 0 → f x = x + Real.log (abs x) :=
by
  sorry

end find_expression_for_f_x_neg_l154_154991


namespace woman_work_completion_woman_days_to_complete_l154_154244

theorem woman_work_completion (M W B : ℝ) (h1 : M + W + B = 1/4) (h2 : M = 1/6) (h3 : B = 1/18) : W = 1/36 :=
by
  -- Substitute h2 and h3 into h1 and solve for W
  sorry

theorem woman_days_to_complete (W : ℝ) (h : W = 1/36) : 1 / W = 36 :=
by
  -- Calculate the reciprocal of h
  sorry

end woman_work_completion_woman_days_to_complete_l154_154244


namespace angle_relationship_l154_154617

theorem angle_relationship (u x y z w : ℝ)
    (H1 : ∀ (D E : ℝ), x + y + (360 - u - z) = 360)
    (H2 : ∀ (D E : ℝ), z + w + (360 - w - x) = 360) :
    x = (u + 2*z - y - w) / 2 := by
  sorry

end angle_relationship_l154_154617


namespace bottles_left_l154_154599

theorem bottles_left (total_bottles : ℕ) (bottles_per_day : ℕ) (days : ℕ)
  (h_total : total_bottles = 264)
  (h_bottles_per_day : bottles_per_day = 15)
  (h_days : days = 11) :
  total_bottles - bottles_per_day * days = 99 :=
by
  sorry

end bottles_left_l154_154599


namespace mod_equiv_example_l154_154967

theorem mod_equiv_example : (185 * 944) % 60 = 40 := by
  sorry

end mod_equiv_example_l154_154967


namespace find_percentage_l154_154103

theorem find_percentage (x : ℝ) (h1 : x = 780) (h2 : ∀ P : ℝ, P / 100 * x = 225 - 30) : P = 25 :=
by
  -- Definitions and conditions here
  -- Recall: x = 780 and P / 100 * x = 195
  sorry

end find_percentage_l154_154103


namespace number_of_rows_l154_154157

theorem number_of_rows (total_chairs : ℕ) (chairs_per_row : ℕ) (r : ℕ) 
  (h1 : total_chairs = 432) (h2 : chairs_per_row = 16) (h3 : total_chairs = chairs_per_row * r) : r = 27 :=
sorry

end number_of_rows_l154_154157


namespace fred_dark_blue_marbles_count_l154_154232

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end fred_dark_blue_marbles_count_l154_154232


namespace matilda_initial_bars_l154_154648

theorem matilda_initial_bars (M : ℕ) 
  (shared_evenly : 5 * M = 20 * 2 / 5)
  (half_given_to_father : M / 2 * 5 = 10)
  (father_bars : 5 + 3 + 2 = 10) :
  M = 4 := 
by
  sorry

end matilda_initial_bars_l154_154648


namespace length_of_tracks_l154_154136

theorem length_of_tracks (x y : ℕ) 
  (h1 : 6 * (x + 2 * y) = 5000)
  (h2 : 7 * (x + y) = 5000) : x = 5 * y :=
  sorry

end length_of_tracks_l154_154136


namespace vegetation_coverage_relationship_l154_154917

noncomputable def conditions :=
  let n := 20
  let sum_x := 60
  let sum_y := 1200
  let sum_xx := 80
  let sum_xy := 640
  (n, sum_x, sum_y, sum_xx, sum_xy)

theorem vegetation_coverage_relationship
  (n sum_x sum_y sum_xx sum_xy : ℕ)
  (h1 : n = 20)
  (h2 : sum_x = 60)
  (h3 : sum_y = 1200)
  (h4 : sum_xx = 80)
  (h5 : sum_xy = 640) :
  let b1 := sum_xy / sum_xx
  let mean_x := sum_x / n
  let mean_y := sum_y / n
  (b1 = 8) ∧ (b1 * (sum_xx / sum_xy) ≤ 1) ∧ ((3, 60) = (mean_x, mean_y)) :=
by
  sorry

end vegetation_coverage_relationship_l154_154917


namespace percent_decrease_is_20_l154_154717

/-- Define the original price and sale price as constants. -/
def P_original : ℕ := 100
def P_sale : ℕ := 80

/-- Define the formula for percent decrease. -/
def percent_decrease (P_original P_sale : ℕ) : ℕ :=
  ((P_original - P_sale) * 100) / P_original

/-- Prove that the percent decrease is 20%. -/
theorem percent_decrease_is_20 : percent_decrease P_original P_sale = 20 :=
by
  sorry

end percent_decrease_is_20_l154_154717


namespace find_angle_A_find_perimeter_l154_154280

noncomputable def cos_rule (b c a : ℝ) (h : b^2 + c^2 - a^2 = b * c) : ℝ :=
(b^2 + c^2 - a^2) / (2 * b * c)

theorem find_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : b^2 + c^2 - a^2 = b * c) (hA : cos_rule b c a h1 = 1 / 2) :
  A = Real.arccos (1 / 2) :=
by sorry

theorem find_perimeter (a b c : ℝ)
  (h_a : a = Real.sqrt 2) (hA : Real.sin (Real.arccos (1 / 2))^2 = (Real.sqrt 3 / 2)^2)
  (hBC : Real.sin (Real.arccos (1 / 2))^2 = Real.sin (Real.arccos (1 / 2)) * Real.sin (Real.arccos (1 / 2)))
  (h_bc : b * c = 2)
  (h_bc_eq : b^2 + c^2 - a^2 = b * c) :
  a + b + c = 3 * Real.sqrt 2 :=
by sorry

end find_angle_A_find_perimeter_l154_154280


namespace total_area_correct_l154_154897

noncomputable def total_area (r p q : ℝ) : ℝ :=
  r^2 + 4*p^2 + 12*q

theorem total_area_correct
  (r p q : ℝ)
  (h : 12 * q = r^2 + 4 * p^2 + 45)
  (r_val : r = 6)
  (p_val : p = 1.5)
  (q_val : q = 7.5) :
  total_area r p q = 135 := by
  sorry

end total_area_correct_l154_154897


namespace deposit_increases_l154_154915

theorem deposit_increases (X r s : ℝ) (hX : 0 < X) (hr : 0 ≤ r) (hs : s < 20) : 
  r > 100 * s / (100 - s) :=
by sorry

end deposit_increases_l154_154915


namespace hexagonal_prism_cross_section_l154_154838

theorem hexagonal_prism_cross_section (n : ℕ) (h₁: n ≥ 3) (h₂: n ≤ 8) : ¬ (n = 9):=
sorry

end hexagonal_prism_cross_section_l154_154838


namespace B_days_finish_work_l154_154338

theorem B_days_finish_work :
  ∀ (W : ℝ) (A_work B_work B_days : ℝ),
  (A_work = W / 9) → 
  (B_work = W / B_days) →
  (3 * (W / 9) + 10 * (W / B_days) = W) →
  B_days = 15 :=
by
  intros W A_work B_work B_days hA_work hB_work hTotal
  sorry

end B_days_finish_work_l154_154338


namespace proof_f_f_2008_eq_2008_l154_154638

-- Define the function f
axiom f : ℝ → ℝ

-- The conditions given in the problem
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodic_f : ∀ x, f (x + 6) = f x
axiom f_at_4 : f 4 = -2008

-- The goal to prove
theorem proof_f_f_2008_eq_2008 : f (f 2008) = 2008 :=
by
  sorry

end proof_f_f_2008_eq_2008_l154_154638


namespace vasya_lowest_position_l154_154358

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l154_154358


namespace intersection_A_B_l154_154406

def A : Set Int := {-1, 0, 1, 5, 8}
def B : Set Int := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} :=
by
  sorry

end intersection_A_B_l154_154406


namespace triangle_sine_inequality_l154_154684

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤
  1 + (1 / 2) * Real.cos ((A - B) / 4) ^ 2 :=
by
  sorry

end triangle_sine_inequality_l154_154684


namespace sym_sum_ineq_l154_154483

theorem sym_sum_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z = 1 / x + 1 / y + 1 / z) : x * y + y * z + z * x ≥ 3 :=
by
  sorry

end sym_sum_ineq_l154_154483


namespace sum_of_first_n_primes_eq_41_l154_154902

theorem sum_of_first_n_primes_eq_41 : 
  ∃ (n : ℕ) (primes : List ℕ), 
    primes = [2, 3, 5, 7, 11, 13] ∧ primes.sum = 41 ∧ primes.length = n := 
by 
  sorry

end sum_of_first_n_primes_eq_41_l154_154902


namespace product_of_sum_and_reciprocal_nonneg_l154_154939

theorem product_of_sum_and_reciprocal_nonneg (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
by
  sorry

end product_of_sum_and_reciprocal_nonneg_l154_154939


namespace inequality_S_l154_154523

def S (n m : ℕ) : ℕ := sorry

theorem inequality_S (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n :=
sorry

end inequality_S_l154_154523


namespace average_customers_per_table_l154_154982

-- Definitions for conditions
def tables : ℝ := 9.0
def women : ℝ := 7.0
def men : ℝ := 3.0

-- Proof problem statement
theorem average_customers_per_table : (women + men) / tables = 10.0 / 9.0 :=
by
  sorry

end average_customers_per_table_l154_154982


namespace neither_necessary_nor_sufficient_l154_154891

-- defining polynomial inequalities
def inequality_1 (a1 b1 c1 x : ℝ) : Prop := a1 * x^2 + b1 * x + c1 > 0
def inequality_2 (a2 b2 c2 x : ℝ) : Prop := a2 * x^2 + b2 * x + c2 > 0

-- defining proposition P and proposition Q
def P (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := ∀ x : ℝ, inequality_1 a1 b1 c1 x ↔ inequality_2 a2 b2 c2 x
def Q (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

-- prove that Q is neither a necessary nor sufficient condition for P
theorem neither_necessary_nor_sufficient {a1 b1 c1 a2 b2 c2 : ℝ} : ¬(Q a1 b1 c1 a2 b2 c2 ↔ P a1 b1 c1 a2 b2 c2) := 
sorry

end neither_necessary_nor_sufficient_l154_154891


namespace sandy_bought_6_books_l154_154696

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_bought_6_books_l154_154696


namespace cubic_eq_roots_l154_154277

theorem cubic_eq_roots (x1 x2 x3 : ℕ) (P : ℕ) 
  (h1 : x1 + x2 + x3 = 10) 
  (h2 : x1 * x2 * x3 = 30) 
  (h3 : x1 * x2 + x2 * x3 + x3 * x1 = P) : 
  P = 31 := by
  sorry

end cubic_eq_roots_l154_154277


namespace quadratic_even_coeff_l154_154255

theorem quadratic_even_coeff (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∃ r s : ℚ, r * s + b * r + c = 0) : (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0) := by
  sorry

end quadratic_even_coeff_l154_154255


namespace weight_of_B_l154_154330

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

theorem weight_of_B :
  (A + B + C) / 3 = 45 → 
  (A + B) / 2 = 40 → 
  (B + C) / 2 = 43 → 
  B = 31 :=
by
  intros h1 h2 h3
  -- detailed proof steps omitted
  sorry

end weight_of_B_l154_154330


namespace equal_elements_l154_154207

theorem equal_elements (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h_perm : ∃ (σ : Equiv.Perm (Fin 2011)), ∀ i, x' i = x (σ i))
  (h_eq : ∀ i : Fin 2011, x i + x ((i + 1) % 2011) = 2 * x' i) :
  ∀ i j : Fin 2011, x i = x j :=
by
  sorry

end equal_elements_l154_154207


namespace point_M_coordinates_l154_154579

theorem point_M_coordinates :
  ∃ M : ℝ × ℝ × ℝ, 
    M.1 = 0 ∧ M.2.1 = 0 ∧  
    (dist (1, 0, 2) (M.1, M.2.1, M.2.2) = dist (1, -3, 1) (M.1, M.2.1, M.2.2)) ∧ 
    M = (0, 0, -3) :=
by
  sorry

end point_M_coordinates_l154_154579


namespace average_lifespan_is_28_l154_154322

-- Define the given data
def batteryLifespans : List ℕ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

-- Define a function to calculate the average of a list of natural numbers
def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

-- State the theorem to be proved
theorem average_lifespan_is_28 :
  average batteryLifespans = 28 := by
  sorry

end average_lifespan_is_28_l154_154322


namespace Ms_Rush_Speed_to_be_on_time_l154_154751

noncomputable def required_speed (d t r : ℝ) :=
  d = 50 * (t + 1/12) ∧ 
  d = 70 * (t - 1/9) →
  r = d / t →
  r = 74

theorem Ms_Rush_Speed_to_be_on_time 
  (d t r : ℝ) 
  (h1 : d = 50 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/9)) 
  (h3 : r = d / t) : 
  r = 74 :=
sorry

end Ms_Rush_Speed_to_be_on_time_l154_154751


namespace find_value_of_z_l154_154616

theorem find_value_of_z (z : ℂ) (h1 : ∀ a : ℝ, z = a * I) (h2 : ((z + 2) / (1 - I)).im = 0) : z = -2 * I :=
sorry

end find_value_of_z_l154_154616


namespace find_positive_number_l154_154834

noncomputable def solve_number (x : ℝ) : Prop :=
  (2/3 * x = 64/216 * (1/x)) ∧ (x > 0)

theorem find_positive_number (x : ℝ) : solve_number x → x = (2/9) * Real.sqrt 3 :=
  by
  sorry

end find_positive_number_l154_154834


namespace cover_square_floor_l154_154346

theorem cover_square_floor (x : ℕ) (h : 2 * x - 1 = 37) : x^2 = 361 :=
by
  sorry

end cover_square_floor_l154_154346


namespace unique_root_value_l154_154373

theorem unique_root_value {x n : ℝ} (h : (15 - n) = 15 - (35 / 4)) :
  (x + 5) * (x + 3) = n + 3 * x → n = 35 / 4 :=
sorry

end unique_root_value_l154_154373


namespace watch_sticker_price_l154_154023

theorem watch_sticker_price (x : ℝ)
  (hx_X : 0.80 * x - 50 = y)
  (hx_Y : 0.90 * x = z)
  (savings : z - y = 25) : 
  x = 250 := by
  sorry

end watch_sticker_price_l154_154023


namespace proof_correct_props_l154_154315

variable (p1 : Prop) (p2 : Prop) (p3 : Prop) (p4 : Prop)

def prop1 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ (1 / 2) * x₀ < (1 / 3) * x₀
def prop2 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < 1 ∧ Real.log x₀ / Real.log (1 / 2) > Real.log x₀ / Real.log (1 / 3)
def prop3 : Prop := ∀ (x : ℝ), 0 < x ∧ (1 / 2) ^ x > Real.log x / Real.log (1 / 2)
def prop4 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 1 / 3 ∧ (1 / 2) ^ x < Real.log x / Real.log (1 / 3)

theorem proof_correct_props : prop2 ∧ prop4 :=
by
  sorry -- Proof goes here

end proof_correct_props_l154_154315


namespace willam_tax_paid_l154_154852

-- Define our conditions
variables (T : ℝ) (tax_collected : ℝ) (willam_percent : ℝ)

-- Initialize the conditions according to the problem statement
def is_tax_collected (tax_collected : ℝ) : Prop := tax_collected = 3840
def is_farm_tax_levied_on_cultivated_land : Prop := true -- Essentially means we acknowledge it is 50%
def is_willam_taxable_land_percentage (willam_percent : ℝ) : Prop := willam_percent = 0.25

-- The final theorem that states Mr. Willam's tax payment is $960 given the conditions
theorem willam_tax_paid  : 
  ∀ (T : ℝ),
  is_tax_collected 3840 → 
  is_farm_tax_levied_on_cultivated_land →
  is_willam_taxable_land_percentage 0.25 →
  0.25 * 3840 = 960 :=
sorry

end willam_tax_paid_l154_154852


namespace franks_daily_reading_l154_154369

-- Define the conditions
def total_pages : ℕ := 612
def days_to_finish : ℕ := 6

-- State the theorem we want to prove
theorem franks_daily_reading : (total_pages / days_to_finish) = 102 :=
by
  sorry

end franks_daily_reading_l154_154369


namespace sequence_a_1000_l154_154693

theorem sequence_a_1000 (a : ℕ → ℕ)
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n, a (n + 1) = 3 * a n - 2 * a (n - 1)) : 
  a 1000 = 2^1000 - 1 := 
sorry

end sequence_a_1000_l154_154693


namespace Ria_original_savings_l154_154019

variables {R F : ℕ}

def initial_ratio (R F : ℕ) : Prop :=
  R * 3 = F * 5

def withdrawn_amount (R : ℕ) : ℕ :=
  R - 160

def new_ratio (R' F : ℕ) : Prop :=
  R' * 5 = F * 3

theorem Ria_original_savings (initial_ratio: initial_ratio R F)
  (new_ratio: new_ratio (withdrawn_amount R) F) : 
  R = 250 :=
by
  sorry

end Ria_original_savings_l154_154019


namespace log_expression_equality_l154_154053

noncomputable def evaluate_log_expression : Real :=
  let log4_8 := (Real.log 8) / (Real.log 4)
  let log5_10 := (Real.log 10) / (Real.log 5)
  Real.sqrt (log4_8 + log5_10)

theorem log_expression_equality : 
  evaluate_log_expression = Real.sqrt ((5 / 2) + (Real.log 2 / Real.log 5)) :=
by
  sorry

end log_expression_equality_l154_154053


namespace evaluate_expression_l154_154442

theorem evaluate_expression (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  rw [hx, hy]
  sorry

end evaluate_expression_l154_154442


namespace total_kids_in_Lawrence_l154_154911

theorem total_kids_in_Lawrence (stay_home kids_camp total_kids : ℕ) (h1 : stay_home = 907611) (h2 : kids_camp = 455682) (h3 : total_kids = stay_home + kids_camp) : total_kids = 1363293 :=
by
  sorry

end total_kids_in_Lawrence_l154_154911


namespace players_per_group_l154_154262

-- Definitions for given conditions
def num_new_players : Nat := 48
def num_returning_players : Nat := 6
def num_groups : Nat := 9

-- Proof that the number of players in each group is 6
theorem players_per_group :
  let total_players := num_new_players + num_returning_players
  total_players / num_groups = 6 := by
  sorry

end players_per_group_l154_154262


namespace circle_equation_l154_154678

theorem circle_equation {a b c : ℝ} (hc : c ≠ 0) :
  ∃ D E F : ℝ, 
    (D = -(a + b)) ∧
    (E = - (c + ab / c)) ∧ 
    (F = ab) ∧
    ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 :=
sorry

end circle_equation_l154_154678


namespace intersection_M_N_l154_154150

def M : Set ℝ := {x : ℝ | |x| < 1}
def N : Set ℝ := {x : ℝ | x^2 - x < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_M_N_l154_154150


namespace smaller_rectangle_perimeter_l154_154169

def perimeter_original_rectangle (a b : ℝ) : Prop := 2 * (a + b) = 100
def number_of_cuts (vertical_cuts horizontal_cuts : ℕ) : Prop := vertical_cuts = 7 ∧ horizontal_cuts = 10
def total_length_of_cuts (a b : ℝ) : Prop := 7 * b + 10 * a = 434

theorem smaller_rectangle_perimeter (a b : ℝ) (vertical_cuts horizontal_cuts : ℕ) (m n : ℕ) :
  perimeter_original_rectangle a b →
  number_of_cuts vertical_cuts horizontal_cuts →
  total_length_of_cuts a b →
  (m = 8) →
  (n = 11) →
  (a / 8 + b / 11) * 2 = 11 :=
by
  sorry

end smaller_rectangle_perimeter_l154_154169


namespace angle_bisectors_and_median_inequality_l154_154829

open Real

variables (A B C : Point)
variables (a b c : ℝ) -- sides of the triangle
variables (p : ℝ) -- semi-perimeter of the triangle
variables (la lb mc : ℝ) -- angle bisectors and median lengths

-- Assume the given conditions
axiom angle_bisector_la (A B C : Point) : ℝ -- lengths of the angle bisector of ∠BAC
axiom angle_bisector_lb (A B C : Point) : ℝ -- lengths of the angle bisector of ∠ABC
axiom median_mc (A B C : Point) : ℝ -- length of the median from vertex C
axiom semi_perimeter (a b c : ℝ) : ℝ -- semi-perimeter of the triangle

-- The statement of the theorem
theorem angle_bisectors_and_median_inequality (la lb mc p : ℝ) :
  la + lb + mc ≤ sqrt 3 * p :=
sorry

end angle_bisectors_and_median_inequality_l154_154829


namespace min_value_expression_l154_154842

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  ∃ c, c = (1/(a+1) + 4/(b+1)) ∧ c ≥ 9/4 :=
by
  sorry

end min_value_expression_l154_154842


namespace intersection_A_B_l154_154228

open Set Real -- Opens necessary namespaces for sets and real numbers

-- Definitions for the sets A and B
def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {x | x > -1}

-- The proof statement for the intersection of sets A and B
theorem intersection_A_B : A ∩ B = (Ioo (-1 : ℝ) 0) ∪ (Ioi 1) :=
by
  sorry -- Proof not included

end intersection_A_B_l154_154228


namespace true_proposition_is_A_l154_154748

-- Define the propositions
def l1 := ∀ (x y : ℝ), x - 2 * y + 3 = 0
def l2 := ∀ (x y : ℝ), 2 * x + y + 3 = 0
def p : Prop := ¬(l1 ∧ l2 ∧ ¬(∃ (x y : ℝ), x - 2 * y + 3 = 0 ∧ 2 * x + y + 3 = 0 ∧ (1 * 2 + (-2) * 1 ≠ 0)))
def q : Prop := ∃ x₀ : ℝ, (0 < x₀) ∧ (x₀ + 2 > Real.exp x₀)

-- The proof problem statement
theorem true_proposition_is_A : (¬p) ∧ q :=
by
  sorry

end true_proposition_is_A_l154_154748


namespace tv_interest_rate_zero_l154_154756

theorem tv_interest_rate_zero (price_installment first_installment last_installment : ℕ) 
  (installment_count : ℕ) (total_price : ℕ) : 
  total_price = 60000 ∧  
  price_installment = 1000 ∧ 
  first_installment = price_installment ∧ 
  last_installment = 59000 ∧ 
  installment_count = 20 ∧  
  (20 * price_installment = 20000) ∧
  (total_price - first_installment = 59000) →
  0 = 0 :=
by 
  sorry

end tv_interest_rate_zero_l154_154756


namespace find_m_l154_154922

-- Definition and conditions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def vertex_property (a b c : ℝ) : Prop := 
  (∀ x, quadratic a b c x ≤ quadratic a b c 2) ∧ quadratic a b c 2 = 4

noncomputable def passes_through_origin (a b c : ℝ) : Prop :=
  quadratic a b c 0 = -7

-- Main theorem statement
theorem find_m (a b c m : ℝ) 
  (h1 : vertex_property a b c) 
  (h2 : passes_through_origin a b c) 
  (h3 : quadratic a b c 5 = m) :
  m = -83/4 :=
sorry

end find_m_l154_154922


namespace remainder_sum_products_l154_154163

theorem remainder_sum_products (a b c d : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) 
  (hd : d % 7 = 6) : 
  ((a * b + c * d) % 7) = 1 :=
by sorry

end remainder_sum_products_l154_154163


namespace x_intercept_of_line_l154_154997

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) :
  ∃ x0 : ℝ, (∀ y : ℝ, y = 0 → (∃ m : ℝ, y = m * (x0 - x1) + y1)) ∧ x0 = 4 :=
by
  sorry

end x_intercept_of_line_l154_154997


namespace mixed_bag_cost_l154_154190

def cost_per_pound_colombian : ℝ := 5.5
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def weight_colombian : ℝ := 28.8

noncomputable def cost_per_pound_mixed_bag : ℝ :=
  (weight_colombian * cost_per_pound_colombian + (total_weight - weight_colombian) * cost_per_pound_peruvian) / total_weight

theorem mixed_bag_cost :
  cost_per_pound_mixed_bag = 5.15 :=
  sorry

end mixed_bag_cost_l154_154190


namespace comb_comb_l154_154715

theorem comb_comb (n1 k1 n2 k2 : ℕ) (h1 : n1 = 10) (h2 : k1 = 3) (h3 : n2 = 8) (h4 : k2 = 4) :
  (Nat.choose n1 k1) * (Nat.choose n2 k2) = 8400 := by
  rw [h1, h2, h3, h4]
  change Nat.choose 10 3 * Nat.choose 8 4 = 8400
  -- Adding the proof steps is not necessary as per instructions
  sorry

end comb_comb_l154_154715


namespace count_multiples_of_12_l154_154643

theorem count_multiples_of_12 (a b : ℤ) (h1 : a = 5) (h2 : b = 145) :
  ∃ n : ℕ, (12 * n + 12 ≤ b) ∧ (12 * n + 12 > a) ∧ n = 12 :=
by
  sorry

end count_multiples_of_12_l154_154643


namespace linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l154_154744

variable {α : Type*} [Ring α]

def linear_function (a b x : α) : α :=
  a * x + b

def special_case_linear_function (a x : α) : α :=
  a * x

def quadratic_function (a b c x : α) : α :=
  a * x^2 + b * x + c

def special_case_quadratic_function1 (a c x : α) : α :=
  a * x^2 + c

def special_case_quadratic_function2 (a x : α) : α :=
  a * x^2

theorem linear_function_general_form (a b x : α) :
  ∃ y, y = linear_function a b x := by
  sorry

theorem special_case_linear_function_proof (a x : α) :
  ∃ y, y = special_case_linear_function a x := by
  sorry

theorem quadratic_function_general_form (a b c x : α) :
  a ≠ 0 → ∃ y, y = quadratic_function a b c x := by
  sorry

theorem special_case_quadratic_function1_proof (a b c x : α) :
  a ≠ 0 → b = 0 → ∃ y, y = special_case_quadratic_function1 a c x := by
  sorry

theorem special_case_quadratic_function2_proof (a b c x : α) :
  a ≠ 0 → b = 0 → c = 0 → ∃ y, y = special_case_quadratic_function2 a x := by
  sorry

end linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l154_154744


namespace francis_had_2_muffins_l154_154798

noncomputable def cost_of_francis_breakfast (m : ℕ) : ℕ := 2 * m + 6
noncomputable def cost_of_kiera_breakfast : ℕ := 4 + 3
noncomputable def total_cost (m : ℕ) : ℕ := cost_of_francis_breakfast m + cost_of_kiera_breakfast

theorem francis_had_2_muffins (m : ℕ) : total_cost m = 17 → m = 2 :=
by
  -- Sorry is used here to leave the proof steps blank.
  sorry

end francis_had_2_muffins_l154_154798


namespace min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l154_154036

theorem min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared 
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  (ae^2 : ℝ) + (bf^2 : ℝ) + (cg^2 : ℝ) + (dh^2 : ℝ) ≥ 32 := 
sorry

end min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l154_154036


namespace hyuksu_total_meat_l154_154093

/-- 
Given that Hyuksu ate 2.6 kilograms (kg) of meat yesterday and 5.98 kilograms (kg) of meat today,
prove that the total kilograms (kg) of meat he ate in two days is 8.58 kg.
-/
theorem hyuksu_total_meat (yesterday today : ℝ) (hy1 : yesterday = 2.6) (hy2 : today = 5.98) :
  yesterday + today = 8.58 := 
by
  rw [hy1, hy2]
  norm_num

end hyuksu_total_meat_l154_154093


namespace percentage_range_l154_154903

noncomputable def minimum_maximum_percentage (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : ℝ × ℝ := sorry

theorem percentage_range (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : 
    minimum_maximum_percentage x y z n m hx1 hx2 hx3 hx4 hx5 h1 h2 h3 h4 = (12.5, 15) :=
sorry

end percentage_range_l154_154903


namespace least_common_multiple_of_first_10_integers_l154_154557

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l154_154557


namespace emilia_donut_holes_count_l154_154863

noncomputable def surface_area (r : ℕ) : ℕ := 4 * r^2

def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

def donut_holes := 5103

theorem emilia_donut_holes_count :
  ∀ (S1 S2 S3 : ℕ), 
  S1 = surface_area 5 → 
  S2 = surface_area 7 → 
  S3 = surface_area 9 → 
  donut_holes = lcm S1 S2 S3 / S1 :=
by
  intros S1 S2 S3 hS1 hS2 hS3
  sorry

end emilia_donut_holes_count_l154_154863


namespace sum_reciprocals_factors_12_l154_154076

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l154_154076


namespace pathway_area_ratio_l154_154565

theorem pathway_area_ratio (AB AD: ℝ) (r: ℝ) (A_rectangle A_circles: ℝ):
  AB = 24 → (AD / AB) = (4 / 3) → r = AB / 2 → 
  A_rectangle = AD * AB → A_circles = π * r^2 →
  (A_rectangle / A_circles) = 16 / (3 * π) :=
by
  sorry

end pathway_area_ratio_l154_154565


namespace happy_numbers_l154_154865

theorem happy_numbers (n : ℕ) (h1 : n < 1000) 
(h2 : 7 ∣ n^2) (h3 : 8 ∣ n^2) (h4 : 9 ∣ n^2) (h5 : 10 ∣ n^2) : 
n = 420 ∨ n = 840 :=
sorry

end happy_numbers_l154_154865


namespace find_linear_combination_l154_154065

variable (a b c : ℝ)

theorem find_linear_combination (h1 : a + 2 * b - 3 * c = 4)
                               (h2 : 5 * a - 6 * b + 7 * c = 8) :
  9 * a + 2 * b - 5 * c = 24 :=
sorry

end find_linear_combination_l154_154065


namespace spending_total_march_to_july_l154_154535

/-- Given the conditions:
  1. Total amount spent by the beginning of March is 1.2 million,
  2. Total amount spent by the end of July is 5.4 million,
  Prove that the total amount spent during March, April, May, June, and July is 4.2 million. -/
theorem spending_total_march_to_july
  (spent_by_end_of_feb : ℝ)
  (spent_by_end_of_july : ℝ)
  (h1 : spent_by_end_of_feb = 1.2)
  (h2 : spent_by_end_of_july = 5.4) :
  spent_by_end_of_july - spent_by_end_of_feb = 4.2 :=
by
  sorry

end spending_total_march_to_july_l154_154535


namespace apples_prepared_l154_154002

variables (n_x n_l : ℕ)

theorem apples_prepared (hx : 3 * n_x = 5 * n_l - 12) (hs : 6 * n_l = 72) : n_x = 12 := 
by sorry

end apples_prepared_l154_154002


namespace primary_schools_to_be_selected_l154_154608

noncomputable def total_schools : ℕ := 150 + 75 + 25
noncomputable def proportion_primary : ℚ := 150 / total_schools
noncomputable def selected_primary : ℚ := proportion_primary * 30

theorem primary_schools_to_be_selected : selected_primary = 18 :=
by sorry

end primary_schools_to_be_selected_l154_154608


namespace cost_of_four_stamps_l154_154882

theorem cost_of_four_stamps (cost_one_stamp : ℝ) (h : cost_one_stamp = 0.34) : 4 * cost_one_stamp = 1.36 := 
by
  rw [h]
  norm_num

end cost_of_four_stamps_l154_154882


namespace area_ratio_of_squares_l154_154261

-- Definition of squares, and their perimeters' relationship
def perimeter (side_length : ℝ) := 4 * side_length

theorem area_ratio_of_squares (a b : ℝ) (h : perimeter a = 4 * perimeter b) : (a * a) = 16 * (b * b) :=
by
  -- We assume the given condition
  have ha : a = 4 * b := sorry
  -- We then prove the area ratio
  sorry

end area_ratio_of_squares_l154_154261


namespace find_dividend_and_divisor_l154_154559

theorem find_dividend_and_divisor (quotient : ℕ) (remainder : ℕ) (total : ℕ) (dividend divisor : ℕ) :
  quotient = 13 ∧ remainder = 6 ∧ total = 137 ∧ (dividend + divisor + quotient + remainder = total)
  ∧ dividend = 13 * divisor + remainder → 
  dividend = 110 ∧ divisor = 8 :=
by
  intro h
  sorry

end find_dividend_and_divisor_l154_154559


namespace situps_combined_l154_154479

theorem situps_combined (peter_situps : ℝ) (greg_per_set : ℝ) (susan_per_set : ℝ) 
                        (peter_per_set : ℝ) (sets : ℝ) 
                        (peter_situps_performed : peter_situps = sets * peter_per_set) 
                        (greg_situps_performed : sets * greg_per_set = 4.5 * 6)
                        (susan_situps_performed : sets * susan_per_set = 3.75 * 6) :
    peter_situps = 37.5 ∧ greg_per_set = 4.5 ∧ susan_per_set = 3.75 ∧ peter_per_set = 6.25 → 
    4.5 * 6 + 3.75 * 6 = 49.5 :=
by
  sorry

end situps_combined_l154_154479


namespace cost_per_pancake_correct_l154_154433

-- Define the daily rent expense
def daily_rent := 30

-- Define the daily supplies expense
def daily_supplies := 12

-- Define the number of pancakes needed to cover expenses
def number_of_pancakes := 21

-- Define the total daily expenses
def total_daily_expenses := daily_rent + daily_supplies

-- Define the cost per pancake calculation
def cost_per_pancake := total_daily_expenses / number_of_pancakes

-- The theorem to prove the cost per pancake
theorem cost_per_pancake_correct :
  cost_per_pancake = 2 := 
by
  sorry

end cost_per_pancake_correct_l154_154433


namespace gcd_of_g_and_y_l154_154686

-- Define the function g(y)
def g (y : ℕ) := (3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 14)

-- Define that y is a multiple of 45678
def isMultipleOf (y divisor : ℕ) : Prop := ∃ k, y = k * divisor

-- Define the proof problem
theorem gcd_of_g_and_y (y : ℕ) (h : isMultipleOf y 45678) : Nat.gcd (g y) y = 1512 :=
by
  sorry

end gcd_of_g_and_y_l154_154686


namespace weeks_to_save_remaining_l154_154615

-- Assuming the conditions
def cost_of_shirt : ℝ := 3
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

-- The proof goal
theorem weeks_to_save_remaining (cost_of_shirt amount_saved saving_per_week : ℝ) :
  cost_of_shirt = 3 ∧ amount_saved = 1.5 ∧ saving_per_week = 0.5 →
  ((cost_of_shirt - amount_saved) / saving_per_week) = 3 := by
  sorry

end weeks_to_save_remaining_l154_154615


namespace complement_A_B_eq_singleton_three_l154_154021

open Set

variable (A : Set ℕ) (B : Set ℕ) (a : ℕ)

theorem complement_A_B_eq_singleton_three (hA : A = {2, 3, 4})
    (hB : B = {a + 2, a}) (h_inter : A ∩ B = B) : A \ B = {3} :=
  sorry

end complement_A_B_eq_singleton_three_l154_154021


namespace distance_between_A_and_B_l154_154237

theorem distance_between_A_and_B (v_A v_B d d' : ℝ)
  (h1 : v_B = 50)
  (h2 : (v_A - v_B) * 30 = d')
  (h3 : (v_A + v_B) * 6 = d) :
  d = 750 :=
sorry

end distance_between_A_and_B_l154_154237


namespace unique_solution_to_equation_l154_154090

theorem unique_solution_to_equation (a : ℝ) (h : ∀ x : ℝ, a * x^2 + Real.sin x ^ 2 = a^2 - a) : a = 1 :=
sorry

end unique_solution_to_equation_l154_154090


namespace bruce_purchased_mangoes_l154_154912

-- Condition definitions
def cost_of_grapes (k_gra kg_cost_gra : ℕ) : ℕ := k_gra * kg_cost_gra
def amount_spent_on_mangoes (total_paid cost_gra : ℕ) : ℕ := total_paid - cost_gra
def quantity_of_mangoes (total_amt_mangoes rate_per_kg_mangoes : ℕ) : ℕ := total_amt_mangoes / rate_per_kg_mangoes

-- Parameters
variable (k_gra rate_per_kg_gra rate_per_kg_mangoes total_paid : ℕ)
variable (kg_gra_total_amt spent_amt_mangoes_qty : ℕ)

-- Given values
axiom A1 : k_gra = 7
axiom A2 : rate_per_kg_gra = 70
axiom A3 : rate_per_kg_mangoes = 55
axiom A4 : total_paid = 985

-- Calculations based on conditions
axiom H1 : cost_of_grapes k_gra rate_per_kg_gra = kg_gra_total_amt
axiom H2 : amount_spent_on_mangoes total_paid kg_gra_total_amt = spent_amt_mangoes_qty
axiom H3 : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9

-- Proof statement to be proven
theorem bruce_purchased_mangoes : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9 := sorry

end bruce_purchased_mangoes_l154_154912


namespace white_balls_in_bag_l154_154572

theorem white_balls_in_bag:
  ∀ (total balls green yellow red purple : Nat),
  total = 60 →
  green = 18 →
  yellow = 8 →
  red = 5 →
  purple = 7 →
  (1 - 0.8) = (red + purple : ℚ) / total →
  (W + green + yellow = total - (red + purple : ℚ)) →
  W = 22 :=
by
  intros total balls green yellow red purple ht hg hy hr hp hprob heqn
  sorry

end white_balls_in_bag_l154_154572


namespace sum_of_pairwise_relatively_prime_numbers_l154_154154

theorem sum_of_pairwise_relatively_prime_numbers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
    (h4 : a * b * c = 302400) (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
    a + b + c = 320 :=
sorry

end sum_of_pairwise_relatively_prime_numbers_l154_154154


namespace probability_heads_tails_4_tosses_l154_154743

-- Define the probabilities of heads and tails
variables (p q : ℝ)

-- Define the conditions
def unfair_coin (p q : ℝ) : Prop :=
  p ≠ q ∧ p + q = 1 ∧ 2 * p * q = 1/2

-- Define the theorem to prove the probability of two heads and two tails
theorem probability_heads_tails_4_tosses 
  (h_unfair : unfair_coin p q) 
  : 6 * (p * q)^2 = 3 / 8 :=
by sorry

end probability_heads_tails_4_tosses_l154_154743


namespace boat_speed_still_water_l154_154605

-- Define the conditions
def speed_of_stream : ℝ := 4
def distance_downstream : ℕ := 68
def time_downstream : ℕ := 4

-- State the theorem
theorem boat_speed_still_water : 
  ∃V_b : ℝ, distance_downstream = (V_b + speed_of_stream) * time_downstream ∧ V_b = 13 :=
by 
  sorry

end boat_speed_still_water_l154_154605


namespace find_values_l154_154151

variable (circle triangle : ℕ)

axiom condition1 : triangle = circle + circle + circle
axiom condition2 : triangle + circle = 40

theorem find_values : circle = 10 ∧ triangle = 30 :=
by
  sorry

end find_values_l154_154151


namespace chess_team_boys_l154_154976

-- Definitions based on the conditions
def members : ℕ := 30
def attendees : ℕ := 20

-- Variables representing boys (B) and girls (G)
variables (B G : ℕ)

-- Defining the conditions
def condition1 : Prop := B + G = members
def condition2 : Prop := (2 * G) / 3 + B = attendees

-- The problem statement: proving that B = 0
theorem chess_team_boys (h1 : condition1 B G) (h2 : condition2 B G) : B = 0 :=
  sorry

end chess_team_boys_l154_154976


namespace find_m_value_l154_154229

def magic_box (a b : ℝ) : ℝ := a^2 + 2 * b - 3

theorem find_m_value (m : ℝ) :
  magic_box m (-3 * m) = 4 ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_value_l154_154229


namespace wilson_theorem_application_l154_154428

theorem wilson_theorem_application (h_prime : Nat.Prime 101) : 
  Nat.factorial 100 % 101 = 100 :=
by
  -- By Wilson's theorem, (p - 1)! ≡ -1 (mod p) for a prime p.
  -- Here p = 101, so (101 - 1)! ≡ -1 (mod 101).
  -- Therefore, 100! ≡ -1 (mod 101).
  -- Knowing that -1 ≡ 100 (mod 101), we can conclude that
  -- 100! ≡ 100 (mod 101).
  sorry

end wilson_theorem_application_l154_154428


namespace regular_polygon_sides_l154_154722

theorem regular_polygon_sides (C : ℕ) (h : (C - 2) * 180 / C = 144) : C = 10 := 
sorry

end regular_polygon_sides_l154_154722


namespace cone_to_cylinder_ratio_l154_154745

theorem cone_to_cylinder_ratio (r : ℝ) (h_cyl : ℝ) (h_cone : ℝ) 
  (V_cyl : ℝ) (V_cone : ℝ) 
  (h_cyl_eq : h_cyl = 18)
  (r_eq : r = 5)
  (h_cone_eq : h_cone = h_cyl / 3)
  (volume_cyl_eq : V_cyl = π * r^2 * h_cyl)
  (volume_cone_eq : V_cone = 1/3 * π * r^2 * h_cone) :
  V_cone / V_cyl = 1 / 9 := by
  sorry

end cone_to_cylinder_ratio_l154_154745


namespace original_volume_of_ice_cube_l154_154802

theorem original_volume_of_ice_cube
  (V : ℝ)
  (h1 : V * (1/2) * (2/3) * (3/4) * (4/5) = 30)
  : V = 150 :=
sorry

end original_volume_of_ice_cube_l154_154802


namespace building_shadow_length_l154_154299

theorem building_shadow_length
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18) (s_flagpole : flagpole_shadow = 45) 
  (h_building : building_height = 26) :
  ∃ (building_shadow : ℝ), (building_height / building_shadow = flagpole_height / flagpole_shadow) ∧ building_shadow = 65 :=
by
  use 65
  sorry

end building_shadow_length_l154_154299


namespace bruce_will_be_3_times_as_old_in_6_years_l154_154033

variables (x : ℕ)

-- Definitions from conditions
def bruce_age_now := 36
def son_age_now := 8

-- Equivalent Lean 4 statement
theorem bruce_will_be_3_times_as_old_in_6_years :
  (bruce_age_now + x = 3 * (son_age_now + x)) → x = 6 :=
sorry

end bruce_will_be_3_times_as_old_in_6_years_l154_154033


namespace problem_statement_l154_154247

noncomputable def smallest_x : ℝ :=
  -8 - (Real.sqrt 292 / 2)

theorem problem_statement (x : ℝ) :
  (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 4 * x = 8 * x - 3 ↔ x = smallest_x :=
by
  sorry

end problem_statement_l154_154247


namespace gcd_lcm_condition_implies_divisibility_l154_154737

theorem gcd_lcm_condition_implies_divisibility
  (a b : ℤ) (h : Int.gcd a b + Int.lcm a b = a + b) : a ∣ b ∨ b ∣ a := 
sorry

end gcd_lcm_condition_implies_divisibility_l154_154737


namespace real_numbers_correspond_to_number_line_l154_154484

noncomputable def number_line := ℝ

def real_numbers := ℝ

theorem real_numbers_correspond_to_number_line :
  ∀ (p : ℝ), ∃ (r : real_numbers), r = p ∧ ∀ (r : real_numbers), ∃ (p : ℝ), p = r :=
by
  sorry

end real_numbers_correspond_to_number_line_l154_154484


namespace inequality_solution_l154_154017

-- Declare the constants m and n
variables (m n : ℝ)

-- State the conditions
def condition1 (x : ℝ) := m < 0
def condition2 := n = -m / 2

-- State the theorem
theorem inequality_solution (x : ℝ) (h1 : condition1 m n) (h2 : condition2 m n) : 
  nx - m < 0 ↔ x < -2 :=
sorry

end inequality_solution_l154_154017


namespace divisibility_by_11_l154_154878

theorem divisibility_by_11
  (n : ℕ) (hn : n ≥ 2)
  (h : (n^2 + (4^n) + (7^n)) % n = 0) :
  (n^2 + 4^n + 7^n) % 11 = 0 := 
by
  sorry

end divisibility_by_11_l154_154878


namespace find_x_l154_154489

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * f x - 19 = f (x - 4)) : x = 4 := 
by 
  sorry

end find_x_l154_154489


namespace inequality_solution_real_roots_range_l154_154649

noncomputable def f (x : ℝ) : ℝ :=
|2 * x - 4| - |x - 3|

theorem inequality_solution :
  ∀ x, f x ≤ 2 → x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

theorem real_roots_range (k : ℝ) :
  (∃ x, f x = 0) → k ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end inequality_solution_real_roots_range_l154_154649


namespace inequality_solver_l154_154546

variable {m n x : ℝ}

-- Main theorem statement validating the instances described above.
theorem inequality_solver (h : 2 * m * x + 3 < 3 * x + n) :
  (2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨ 
  (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨ 
  (m = 3 / 2 ∧ n > 3 ∧ ∀ x : ℝ, true) ∨ 
  (m = 3 / 2 ∧ n ≤ 3 ∧ ∀ x : ℝ, false) :=
sorry

end inequality_solver_l154_154546


namespace probability_at_least_one_girl_l154_154671

theorem probability_at_least_one_girl (total_students boys girls k : ℕ) (h_total: total_students = 5) (h_boys: boys = 3) (h_girls: girls = 2) (h_k: k = 3) : 
  (1 - ((Nat.choose boys k) / (Nat.choose total_students k))) = 9 / 10 :=
by
  sorry

end probability_at_least_one_girl_l154_154671


namespace determine_ABCC_l154_154962

theorem determine_ABCC :
  ∃ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ 
    C ≠ D ∧ C ≠ E ∧ 
    D ≠ E ∧ 
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    1000 * A + 100 * B + 11 * C = (11 * D - E) * 100 + 11 * D * E ∧ 
    1000 * A + 100 * B + 11 * C = 1966 :=
sorry

end determine_ABCC_l154_154962


namespace Jack_emails_evening_l154_154171

theorem Jack_emails_evening : 
  ∀ (morning_emails evening_emails : ℕ), 
  (morning_emails = 9) ∧ 
  (evening_emails = morning_emails - 2) → 
  evening_emails = 7 := 
by
  intros morning_emails evening_emails
  sorry

end Jack_emails_evening_l154_154171


namespace prob_two_girls_is_one_fourth_l154_154823

-- Define the probability of giving birth to a girl
def prob_girl : ℚ := 1 / 2

-- Define the probability of having two girls
def prob_two_girls : ℚ := prob_girl * prob_girl

-- Theorem statement: The probability of having two girls is 1/4
theorem prob_two_girls_is_one_fourth : prob_two_girls = 1 / 4 :=
by sorry

end prob_two_girls_is_one_fourth_l154_154823


namespace abs_diff_squares_110_108_l154_154677

theorem abs_diff_squares_110_108 : abs ((110 : ℤ)^2 - (108 : ℤ)^2) = 436 := by
  sorry

end abs_diff_squares_110_108_l154_154677


namespace gcd_proof_l154_154217

theorem gcd_proof :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 33 ∧ Nat.lcm a b = 90 ∧ Nat.gcd a b = 3 :=
sorry

end gcd_proof_l154_154217


namespace calculate_cherry_pies_l154_154866

-- Definitions for the conditions
def total_pies : ℕ := 40
def ratio_parts_apple : ℕ := 2
def ratio_parts_blueberry : ℕ := 5
def ratio_parts_cherry : ℕ := 3
def total_ratio_parts := ratio_parts_apple + ratio_parts_blueberry + ratio_parts_cherry

-- Calculating the number of pies per part and then the number of cherry pies
def pies_per_part : ℕ := total_pies / total_ratio_parts
def cherry_pies : ℕ := ratio_parts_cherry * pies_per_part

-- Proof statement
theorem calculate_cherry_pies : cherry_pies = 12 :=
by
  -- Lean proof goes here
  sorry

end calculate_cherry_pies_l154_154866


namespace sequence_sum_difference_l154_154224

def sum_odd (n : ℕ) : ℕ := n * n
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sequence_sum_difference :
  sum_even 1500 - sum_odd 1500 + sum_triangular 1500 = 563628000 :=
by
  sorry

end sequence_sum_difference_l154_154224


namespace total_weight_of_pumpkins_l154_154410

def first_pumpkin_weight : ℝ := 12.6
def second_pumpkin_weight : ℝ := 23.4
def total_weight : ℝ := 36

theorem total_weight_of_pumpkins :
  first_pumpkin_weight + second_pumpkin_weight = total_weight :=
by
  sorry

end total_weight_of_pumpkins_l154_154410


namespace inequality_inequality_hold_l154_154883

theorem inequality_inequality_hold (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a^2 + b^2 = 1/2) :
  (1 / (1 - a)) + (1 / (1 - b)) ≥ 4 :=
by
  sorry

end inequality_inequality_hold_l154_154883


namespace ensure_two_of_each_kind_l154_154417

def tablets_A := 10
def tablets_B := 14
def least_number_of_tablets_to_ensure_two_of_each := 12

theorem ensure_two_of_each_kind 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extracted : ℕ) 
  (hA : total_A = tablets_A) 
  (hB : total_B = tablets_B)
  (hExtract : extracted = least_number_of_tablets_to_ensure_two_of_each) : 
  ∃ (extracted : ℕ), extracted = least_number_of_tablets_to_ensure_two_of_each ∧ extracted ≥ tablets_A + 2 := 
sorry

end ensure_two_of_each_kind_l154_154417


namespace tammy_speed_on_second_day_l154_154143

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l154_154143


namespace arithmetic_sequence_100th_term_l154_154597

theorem arithmetic_sequence_100th_term (a b : ℤ)
  (h1 : 2 * a - a = a) -- definition of common difference d where d = a
  (h2 : b - 2 * a = a) -- b = 3a
  (h3 : a - 6 - b = -2 * a - 6) -- consistency of fourth term
  (h4 : 6 * a = -6) -- equation to solve for a
  : (a + 99 * (2 * a - a)) = -100 := 
sorry

end arithmetic_sequence_100th_term_l154_154597


namespace triangle_inequality_for_roots_l154_154045

theorem triangle_inequality_for_roots (p q r : ℝ) (hroots_pos : ∀ (u v w : ℝ), (u > 0) ∧ (v > 0) ∧ (w > 0) ∧ (u * v * w = -r) ∧ (u + v + w = -p) ∧ (u * v + u * w + v * w = q)) :
  p^3 - 4 * p * q + 8 * r > 0 :=
sorry

end triangle_inequality_for_roots_l154_154045


namespace initial_red_balloons_l154_154627

variable (initial_red : ℕ)
variable (given_away : ℕ := 24)
variable (left_with : ℕ := 7)

theorem initial_red_balloons : initial_red = given_away + left_with :=
by sorry

end initial_red_balloons_l154_154627


namespace floor_neg_seven_thirds_l154_154037

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end floor_neg_seven_thirds_l154_154037


namespace max_min_x2_min_xy_plus_y2_l154_154598

theorem max_min_x2_min_xy_plus_y2 (x y : ℝ) (h : x^2 + x * y + y^2 = 3) :
  1 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 9 :=
by sorry

end max_min_x2_min_xy_plus_y2_l154_154598


namespace smallest_num_rectangles_to_cover_square_l154_154833

-- Define essential conditions
def area_3by4_rectangle : ℕ := 3 * 4
def area_square (side_length : ℕ) : ℕ := side_length * side_length
def can_be_tiled_with_3by4 (side_length : ℕ) : Prop := (area_square side_length) % area_3by4_rectangle = 0

-- Define the main theorem
theorem smallest_num_rectangles_to_cover_square :
  can_be_tiled_with_3by4 12 → ∃ n : ℕ, n = (area_square 12) / area_3by4_rectangle ∧ n = 12 :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l154_154833


namespace intersection_of_sets_l154_154961

def setA (x : ℝ) : Prop := 2 * x + 1 > 0
def setB (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by 
  sorry  -- Placeholder for the proof

end intersection_of_sets_l154_154961


namespace dollar_expansion_l154_154020

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2 + a * b

theorem dollar_expansion : dollar ((x - y) ^ 3) ((y - x) ^ 3) = -((x - y) ^ 6) := by
  sorry

end dollar_expansion_l154_154020


namespace lock_probability_l154_154073

/-- The probability of correctly guessing the last digit of a three-digit combination lock,
given that the first two digits are correctly set and each digit ranges from 0 to 9. -/
theorem lock_probability : 
  ∀ (d1 d2 : ℕ), 
  (0 ≤ d1 ∧ d1 < 10) ∧ (0 ≤ d2 ∧ d2 < 10) →
  (0 ≤ d3 ∧ d3 < 10) → 
  (1/10 : ℝ) = (1 : ℝ) / (10 : ℝ) :=
by
  sorry

end lock_probability_l154_154073


namespace mod_2_200_sub_3_l154_154351

theorem mod_2_200_sub_3 (h1 : 2^1 % 7 = 2) (h2 : 2^2 % 7 = 4) (h3 : 2^3 % 7 = 1) : (2^200 - 3) % 7 = 1 := 
by
  sorry

end mod_2_200_sub_3_l154_154351


namespace probability_not_same_level_is_four_fifths_l154_154792

-- Definitions of the conditions
def nobility_levels := 5
def total_outcomes := nobility_levels * nobility_levels
def same_level_outcomes := nobility_levels

-- Definition of the probability
def probability_not_same_level := 1 - (same_level_outcomes / total_outcomes : ℚ)

-- The theorem statement
theorem probability_not_same_level_is_four_fifths :
  probability_not_same_level = 4 / 5 := 
  by sorry

end probability_not_same_level_is_four_fifths_l154_154792


namespace ball_drawing_ways_l154_154861

theorem ball_drawing_ways :
    ∃ (r w y : ℕ), 
      0 ≤ r ∧ r ≤ 2 ∧
      0 ≤ w ∧ w ≤ 3 ∧
      0 ≤ y ∧ y ≤ 5 ∧
      r + w + y = 5 ∧
      10 ≤ 5 * r + 2 * w + y ∧ 
      5 * r + 2 * w + y ≤ 15 := 
sorry

end ball_drawing_ways_l154_154861


namespace sample_size_is_13_l154_154329

noncomputable def stratified_sample_size : ℕ :=
  let A := 120
  let B := 80
  let C := 60
  let total_units := A + B + C
  let sampled_C_units := 3
  let sampling_fraction := sampled_C_units / C
  let n := sampling_fraction * total_units
  n

theorem sample_size_is_13 :
  stratified_sample_size = 13 := by
  sorry

end sample_size_is_13_l154_154329


namespace proof_custom_operations_l154_154462

def customOp1 (a b : ℕ) : ℕ := a * b / (a + b)
def customOp2 (a b : ℕ) : ℕ := a * a + b * b

theorem proof_custom_operations :
  customOp2 (customOp1 7 14) 2 = 200 := 
by 
  sorry

end proof_custom_operations_l154_154462


namespace cost_per_dvd_l154_154726

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) :
  total_cost = 4.80 ∧ num_dvds = 4 → cost_per_dvd = 1.20 :=
by
  intro h
  sorry

end cost_per_dvd_l154_154726


namespace organization_members_count_l154_154663

theorem organization_members_count (num_committees : ℕ) (pair_membership : ℕ → ℕ → ℕ) :
  num_committees = 5 →
  (∀ i j k l : ℕ, i ≠ j → k ≠ l → pair_membership i j = pair_membership k l → i = k ∧ j = l ∨ i = l ∧ j = k) →
  ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end organization_members_count_l154_154663


namespace minimum_value_ineq_l154_154614

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ (3 / 4) := sorry

end minimum_value_ineq_l154_154614


namespace amina_wins_is_21_over_32_l154_154543

/--
Amina and Bert alternate turns tossing a fair coin. Amina goes first and each player takes three turns.
The first player to toss a tail wins. If neither Amina nor Bert tosses a tail, then neither wins.
Prove that the probability that Amina wins is \( \frac{21}{32} \).
-/
def amina_wins_probability : ℚ :=
  let p_first_turn := 1 / 2
  let p_second_turn := (1 / 2) ^ 3
  let p_third_turn := (1 / 2) ^ 5
  p_first_turn + p_second_turn + p_third_turn

theorem amina_wins_is_21_over_32 :
  amina_wins_probability = 21 / 32 :=
sorry

end amina_wins_is_21_over_32_l154_154543


namespace exists_real_polynomial_l154_154062

noncomputable def has_negative_coeff (p : Polynomial ℝ) : Prop :=
  ∃ i, (p.coeff i) < 0

noncomputable def all_positive_coeff (n : ℕ) (p : Polynomial ℝ) : Prop :=
  ∀ i, (Polynomial.derivative^[n] p).coeff i > 0

theorem exists_real_polynomial :
  ∃ p : Polynomial ℝ, has_negative_coeff p ∧ (∀ n > 1, all_positive_coeff n p) :=
sorry

end exists_real_polynomial_l154_154062


namespace value_of_40th_expression_l154_154225

-- Define the sequence
def minuend (n : ℕ) : ℕ := 100 - (n - 1)
def subtrahend (n : ℕ) : ℕ := n
def expression_value (n : ℕ) : ℕ := minuend n - subtrahend n

-- Theorem: The value of the 40th expression in the sequence is 21
theorem value_of_40th_expression : expression_value 40 = 21 := by
  show 100 - (40 - 1) - 40 = 21
  sorry

end value_of_40th_expression_l154_154225


namespace lateral_surface_area_base_area_ratio_correct_l154_154057

noncomputable def lateral_surface_area_to_base_area_ratio
  (S P Q R : Type)
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12)
  : ℝ :=
  π * (4 * Real.sqrt 3 - 3) / 13

theorem lateral_surface_area_base_area_ratio_correct
  {S P Q R : Type}
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12) :
  lateral_surface_area_to_base_area_ratio S P Q R angle_PSR angle_SQR angle_PSQ
    h_PSR h_SQR h_PSQ = π * (4 * Real.sqrt 3 - 3) / 13 :=
  by sorry

end lateral_surface_area_base_area_ratio_correct_l154_154057


namespace sum_le_30_l154_154505

variable (a b x y : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx_pos : 0 < x) (hy_pos : 0 < y)
variable (h1 : a * x ≤ 5) (h2 : a * y ≤ 10) (h3 : b * x ≤ 10) (h4 : b * y ≤ 10)

theorem sum_le_30 : a * x + a * y + b * x + b * y ≤ 30 := sorry

end sum_le_30_l154_154505


namespace slope_tangent_at_point_l154_154574

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem slope_tangent_at_point : (deriv f 1) = 1 := 
by
  sorry

end slope_tangent_at_point_l154_154574


namespace maximize_area_l154_154104

theorem maximize_area (P L W : ℝ) (h1 : P = 2 * L + 2 * W) (h2 : 0 < P) : 
  (L = P / 4) ∧ (W = P / 4) :=
by
  sorry

end maximize_area_l154_154104


namespace range_of_a_l154_154560

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 16) / x

theorem range_of_a (a : ℝ) (h1 : 2 ≤ a) (h2 : (∀ x, 2 ≤ x ∧ x ≤ a → 9 ≤ f x ∧ f x ≤ 11)) : 4 ≤ a ∧ a ≤ 8 := by
  sorry

end range_of_a_l154_154560


namespace square_side_length_l154_154503

variable (x : ℝ) (π : ℝ) (hπ: π = Real.pi)

theorem square_side_length (h1: 4 * x = 10 * π) : 
  x = (5 * π) / 2 := 
by
  sorry

end square_side_length_l154_154503


namespace digit_D_is_five_l154_154283

variable (A B C D : Nat)
variable (h1 : (B * A) % 10 = A % 10)
variable (h2 : ∀ (C : Nat), B - A = B % 10 ∧ C ≤ A)

theorem digit_D_is_five : D = 5 :=
by
  sorry

end digit_D_is_five_l154_154283


namespace value_of_expression_l154_154403

theorem value_of_expression (x Q : ℝ) (π : Real) (h : 5 * (3 * x - 4 * π) = Q) : 10 * (6 * x - 8 * π) = 4 * Q :=
by 
  sorry

end value_of_expression_l154_154403


namespace mary_remaining_cards_l154_154840

variable (initial_cards : ℝ) (bought_cards : ℝ) (promised_cards : ℝ)

def remaining_cards (initial : ℝ) (bought : ℝ) (promised : ℝ) : ℝ :=
  initial + bought - promised

theorem mary_remaining_cards :
  initial_cards = 18.0 →
  bought_cards = 40.0 →
  promised_cards = 26.0 →
  remaining_cards initial_cards bought_cards promised_cards = 32.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end mary_remaining_cards_l154_154840


namespace curve_transformation_l154_154089

theorem curve_transformation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →
  (x' = 4 * x) →
  (y' = 2 * y) →
  (x'^2 / 16 + y'^2 / 4 = 1) :=
by
  sorry

end curve_transformation_l154_154089


namespace equation_solutions_l154_154122

theorem equation_solutions (x : ℝ) : x * (2 * x + 1) = 2 * x + 1 ↔ x = -1 / 2 ∨ x = 1 :=
by
  sorry

end equation_solutions_l154_154122


namespace find_length_AB_l154_154218

theorem find_length_AB 
(distance_between_parallels : ℚ)
(radius_of_incircle : ℚ)
(is_isosceles : Prop)
(h_parallel : distance_between_parallels = 18 / 25)
(h_radius : radius_of_incircle = 8 / 3)
(h_isosceles : is_isosceles) :
  ∃ AB : ℚ, AB = 20 := 
sorry

end find_length_AB_l154_154218


namespace cell_count_at_end_of_days_l154_154367

-- Defining the conditions
def initial_cells : ℕ := 2
def split_ratio : ℕ := 3
def days : ℕ := 9
def cycle_days : ℕ := 3

-- The main statement to be proved
theorem cell_count_at_end_of_days :
  (initial_cells * split_ratio^((days / cycle_days) - 1)) = 18 :=
by
  sorry

end cell_count_at_end_of_days_l154_154367


namespace values_of_x_l154_154371

theorem values_of_x (x : ℝ) : (-2 < x ∧ x < 2) ↔ (x^2 < |x| + 2) := by
  sorry

end values_of_x_l154_154371


namespace find_duplicated_page_number_l154_154491

noncomputable def duplicated_page_number (n : ℕ) (incorrect_sum : ℕ) : ℕ :=
  incorrect_sum - n * (n + 1) / 2

theorem find_duplicated_page_number :
  ∃ n k, (1 <= k ∧ k <= n) ∧ ( ∃ n, (1 <= n) ∧ ( n * (n + 1) / 2 + k = 2550) )
  ∧ duplicated_page_number 70 2550 = 65 :=
by
  sorry

end find_duplicated_page_number_l154_154491


namespace sum_of_perpendiculars_eq_altitude_l154_154297

variables {A B C P A' B' C' : Type*}
variables (AB AC BC PA' PB' PC' h : ℝ)

-- Conditions
def is_isosceles_triangle (AB AC BC : ℝ) : Prop :=
  AB = AC

def point_inside_triangle (P A B C : Type*) : Prop :=
  true -- Assume point P is inside the triangle

def is_perpendiculars_dropped (PA' PB' PC' : ℝ) : Prop :=
  true -- Assume PA', PB', PC' are the lengths of the perpendiculars from P to the sides BC, CA, AB

def base_of_triangle (BC : ℝ) : Prop :=
  true -- Assume BC is the base of triangle

-- Theorem statement
theorem sum_of_perpendiculars_eq_altitude
  (h : ℝ) (AB AC BC PA' PB' PC' : ℝ)
  (isosceles : is_isosceles_triangle AB AC BC)
  (point_inside_triangle' : point_inside_triangle P A B C)
  (perpendiculars_dropped : is_perpendiculars_dropped PA' PB' PC')
  (base_of_triangle' : base_of_triangle BC) : 
  PA' + PB' + PC' = h := 
sorry

end sum_of_perpendiculars_eq_altitude_l154_154297


namespace inequality_for_positive_reals_l154_154509

theorem inequality_for_positive_reals
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := 
sorry

end inequality_for_positive_reals_l154_154509


namespace bob_repayment_days_l154_154500

theorem bob_repayment_days :
  ∃ (x : ℕ), (15 + 3 * x ≥ 45) ∧ (∀ y : ℕ, (15 + 3 * y ≥ 45) → x ≤ y) ∧ x = 10 := 
by
  sorry

end bob_repayment_days_l154_154500


namespace opposite_of_neg_2023_l154_154296

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l154_154296


namespace A_B_together_l154_154125

/-- This represents the problem of finding out the number of days A and B together 
can finish a piece of work given the conditions. -/
theorem A_B_together (A_rate B_rate: ℝ) (A_days B_days: ℝ) (work: ℝ) :
  A_rate = 1 / 8 →
  A_days = 4 →
  B_rate = 1 / 12 →
  B_days = 6 →
  work = 1 →
  (A_days * A_rate + B_days * B_rate = work / 2) →
  (24 / (A_rate + B_rate) = 4.8) :=
by
  intros hA_rate hA_days hB_rate hB_days hwork hwork_done
  sorry

end A_B_together_l154_154125


namespace correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l154_154858

theorem correct_operation_B (a : ℝ) : a^3 / a = a^2 := 
by sorry

theorem incorrect_operation_A (a : ℝ) : a^2 + a^5 ≠ a^7 := 
by sorry

theorem incorrect_operation_C (a : ℝ) : (3 * a^2)^2 ≠ 6 * a^4 := 
by sorry

theorem incorrect_operation_D (a b : ℝ) : (a - b)^2 ≠ a^2 - b^2 := 
by sorry

end correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l154_154858


namespace equation_of_line_l154_154465

theorem equation_of_line (a b : ℝ) (h1 : a = -2) (h2 : b = 2) :
  (∀ x y : ℝ, (x / a + y / b = 1) → x - y + 2 = 0) :=
by
  sorry

end equation_of_line_l154_154465


namespace min_value_48_l154_154766

noncomputable def min_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : ℝ :=
  1 / a + 27 / b

theorem min_value_48 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : 
  min_value ha hb h = 48 := 
sorry

end min_value_48_l154_154766


namespace average_marks_l154_154641

-- Given conditions
variables (M P C : ℝ)
variables (h1 : M + P = 32) (h2 : C = P + 20)

-- Statement to be proved
theorem average_marks : (M + C) / 2 = 26 :=
by
  -- The proof will be inserted here
  sorry

end average_marks_l154_154641


namespace binom_20_17_l154_154029

theorem binom_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end binom_20_17_l154_154029


namespace grocery_store_more_expensive_per_can_l154_154647

theorem grocery_store_more_expensive_per_can :
  ∀ (bulk_case_price : ℝ) (bulk_cans_per_case : ℕ)
    (grocery_case_price : ℝ) (grocery_cans_per_case : ℕ),
  bulk_case_price = 12.00 →
  bulk_cans_per_case = 48 →
  grocery_case_price = 6.00 →
  grocery_cans_per_case = 12 →
  (grocery_case_price / grocery_cans_per_case - bulk_case_price / bulk_cans_per_case) * 100 = 25 :=
by
  intros _ _ _ _ h1 h2 h3 h4
  sorry

end grocery_store_more_expensive_per_can_l154_154647


namespace min_period_and_max_value_l154_154250

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

theorem min_period_and_max_value :
  (∀ x, f (x + π) = f x) ∧ (∀ x, f x ≤ 4) ∧ (∃ x, f x = 4) :=
by
  sorry

end min_period_and_max_value_l154_154250


namespace problem_1_problem_2_l154_154621

-- Define sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- First problem statement
theorem problem_1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  -- proof omitted
  sorry

-- Second problem statement
theorem problem_2 (a : ℝ) : (∅ ⊆ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  -- proof omitted
  sorry

end problem_1_problem_2_l154_154621


namespace derek_february_savings_l154_154925

theorem derek_february_savings :
  ∀ (savings : ℕ → ℕ),
  (savings 1 = 2) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 12 → savings (n + 1) = 2 * savings n) ∧
  (savings 12 = 4096) →
  savings 2 = 4 :=
by
  sorry

end derek_february_savings_l154_154925


namespace cube_edge_adjacency_l154_154725

def is_beautiful (f: Finset ℕ) := 
  ∃ a b c d, f = {a, b, c, d} ∧ a = b + c + d

def cube_is_beautiful (faces: Finset (Finset ℕ)) :=
  ∃ t1 t2 t3, t1 ∈ faces ∧ t2 ∈ faces ∧ t3 ∈ faces ∧
  is_beautiful t1 ∧ is_beautiful t2 ∧ is_beautiful t3

def valid_adjacency (v: ℕ) (n1 n2 n3: ℕ) := 
  v = 6 ∧ ((n1 = 2 ∧ n2 = 3 ∧ n3 = 5) ∨
           (n1 = 2 ∧ n2 = 3 ∧ n3 = 7) ∨
           (n1 = 3 ∧ n2 = 5 ∧ n3 = 7))

theorem cube_edge_adjacency : 
  ∀ faces: Finset (Finset ℕ), 
  ∃ v n1 n2 n3, 
  (v = 6 ∧ (valid_adjacency v n1 n2 n3)) ∧
  cube_is_beautiful faces := 
by
  -- Entails the proof, which is not required here
  sorry

end cube_edge_adjacency_l154_154725


namespace find_least_d_l154_154106

theorem find_least_d :
  ∃ d : ℕ, (d % 7 = 1) ∧ (d % 5 = 2) ∧ (d % 3 = 2) ∧ d = 92 :=
by 
  sorry

end find_least_d_l154_154106


namespace geometric_sequence_tenth_term_l154_154682

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (12 / 3) / 4
  let nth_term (n : ℕ) := a * r^(n-1)
  nth_term 10 = 4 :=
  by sorry

end geometric_sequence_tenth_term_l154_154682


namespace find_cost_price_l154_154864

/-- Define the given conditions -/
def selling_price : ℝ := 100
def profit_percentage : ℝ := 0.15
def cost_price : ℝ := 86.96

/-- Define the relationship between selling price and cost price -/
def relation (CP SP : ℝ) : Prop := SP = CP * (1 + profit_percentage)

/-- State the theorem based on the conditions and required proof -/
theorem find_cost_price 
  (SP : ℝ) (CP : ℝ) 
  (h1 : SP = selling_price) 
  (h2 : relation CP SP) : 
  CP = cost_price := 
by
  sorry

end find_cost_price_l154_154864


namespace closest_to_9_l154_154004

noncomputable def optionA : ℝ := 10.01
noncomputable def optionB : ℝ := 9.998
noncomputable def optionC : ℝ := 9.9
noncomputable def optionD : ℝ := 9.01
noncomputable def target : ℝ := 9

theorem closest_to_9 : 
  abs (optionD - target) < abs (optionA - target) ∧ 
  abs (optionD - target) < abs (optionB - target) ∧ 
  abs (optionD - target) < abs (optionC - target) := 
by
  sorry

end closest_to_9_l154_154004


namespace casey_saves_money_l154_154159

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l154_154159


namespace bankers_discount_l154_154393

/-- The banker’s gain on a sum due 3 years hence at 12% per annum is Rs. 360.
   The banker's discount is to be determined. -/
theorem bankers_discount (BG BD TD : ℝ) (R : ℝ := 12 / 100) (T : ℝ := 3) 
  (h1 : BG = 360) (h2 : BG = (BD * TD) / (BD - TD)) (h3 : TD = (P * R * T) / 100) 
  (h4 : BG = (TD * R * T) / 100) :
  BD = 562.5 :=
sorry

end bankers_discount_l154_154393


namespace find_d_l154_154135

noncomputable def single_point_graph (d : ℝ) : Prop :=
  ∃ x y : ℝ, 3 * x^2 + 2 * y^2 + 9 * x - 14 * y + d = 0

theorem find_d : single_point_graph 31.25 :=
sorry

end find_d_l154_154135


namespace cara_total_amount_owed_l154_154016

-- Define the conditions
def principal : ℝ := 54
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the simple interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the total amount owed calculation
def total_amount_owed (P R T : ℝ) : ℝ := P + (interest P R T)

-- The proof statement
theorem cara_total_amount_owed : total_amount_owed principal rate time = 56.70 := by
  sorry

end cara_total_amount_owed_l154_154016


namespace sin_double_angle_l154_154041

theorem sin_double_angle {x : ℝ} (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_l154_154041


namespace original_selling_price_is_990_l154_154553

theorem original_selling_price_is_990 
( P : ℝ ) -- original purchase price
( SP_1 : ℝ := 1.10 * P ) -- original selling price
( P_new : ℝ := 0.90 * P ) -- new purchase price
( SP_2 : ℝ := 1.17 * P ) -- new selling price
( h : SP_2 - SP_1 = 63 ) : SP_1 = 990 :=
by {
  -- This is just the statement, proof is not provided
  sorry
}

end original_selling_price_is_990_l154_154553


namespace max_det_A_l154_154434

open Real

-- Define the matrix and the determinant expression
noncomputable def A (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, 1, 1],
    ![1, 1 + cos θ, 1],
    ![1 + sin θ, 1, 1]
  ]

-- Lean statement to prove the maximum value of the determinant of matrix A
theorem max_det_A : ∃ θ : ℝ, (Matrix.det (A θ)) ≤ 1/2 := by
  sorry

end max_det_A_l154_154434


namespace sally_eats_sandwiches_l154_154286

theorem sally_eats_sandwiches
  (saturday_sandwiches : ℕ)
  (bread_per_sandwich : ℕ)
  (total_bread : ℕ)
  (one_sandwich_on_sunday : ℕ)
  (saturday_bread : saturday_sandwiches * bread_per_sandwich = 4)
  (total_bread_consumed : total_bread = 6)
  (bread_on_sundy : bread_per_sandwich = 2) :
  (total_bread - saturday_sandwiches * bread_per_sandwich) / bread_per_sandwich = one_sandwich_on_sunday :=
sorry

end sally_eats_sandwiches_l154_154286


namespace weighted_averages_correct_l154_154563

def group_A_boys : ℕ := 20
def group_B_boys : ℕ := 25
def group_C_boys : ℕ := 15

def group_A_weight : ℝ := 50.25
def group_B_weight : ℝ := 45.15
def group_C_weight : ℝ := 55.20

def group_A_height : ℝ := 160
def group_B_height : ℝ := 150
def group_C_height : ℝ := 165

def group_A_age : ℝ := 15
def group_B_age : ℝ := 14
def group_C_age : ℝ := 16

def group_A_athletic : ℝ := 0.60
def group_B_athletic : ℝ := 0.40
def group_C_athletic : ℝ := 0.75

noncomputable def total_boys : ℕ := group_A_boys + group_B_boys + group_C_boys

noncomputable def weighted_average_height : ℝ := 
    (group_A_boys * group_A_height + group_B_boys * group_B_height + group_C_boys * group_C_height) / total_boys

noncomputable def weighted_average_weight : ℝ := 
    (group_A_boys * group_A_weight + group_B_boys * group_B_weight + group_C_boys * group_C_weight) / total_boys

noncomputable def weighted_average_age : ℝ := 
    (group_A_boys * group_A_age + group_B_boys * group_B_age + group_C_boys * group_C_age) / total_boys

noncomputable def weighted_average_athletic : ℝ := 
    (group_A_boys * group_A_athletic + group_B_boys * group_B_athletic + group_C_boys * group_C_athletic) / total_boys

theorem weighted_averages_correct :
  weighted_average_height = 157.08 ∧
  weighted_average_weight = 49.36 ∧
  weighted_average_age = 14.83 ∧
  weighted_average_athletic = 0.5542 := 
  by
    sorry

end weighted_averages_correct_l154_154563


namespace similar_triangles_ratios_l154_154916

-- Define the context
variables {a b c a' b' c' : ℂ}

-- Define the statement of the problem
theorem similar_triangles_ratios (h_sim : ∃ z : ℂ, z ≠ 0 ∧ b - a = z * (b' - a') ∧ c - a = z * (c' - a')) :
  (b - a) / (c - a) = (b' - a') / (c' - a') :=
sorry

end similar_triangles_ratios_l154_154916


namespace solve_for_x_l154_154664

theorem solve_for_x (x : ℝ) : 64 = 4 * (16:ℝ)^(x - 2) → x = 3 :=
by 
  intro h
  sorry

end solve_for_x_l154_154664


namespace abs_inequality_solution_l154_154380

theorem abs_inequality_solution (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 < x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x < -5) := 
by
  sorry

end abs_inequality_solution_l154_154380


namespace find_a2_l154_154468

variable {a_n : ℕ → ℚ}

def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n-1) * d

theorem find_a2 (h_seq : arithmetic_seq a_n) (h3_5 : a_n 3 + a_n 5 = 15) (h6 : a_n 6 = 7) :
  a_n 2 = 8 := 
sorry

end find_a2_l154_154468


namespace trig_system_solution_l154_154254

theorem trig_system_solution (x y : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) (hy : 0 ≤ y ∧ y < 2 * Real.pi)
  (h1 : Real.sin x + Real.cos y = 0) (h2 : Real.cos x * Real.sin y = -1/2) :
    (x = Real.pi / 4 ∧ y = 5 * Real.pi / 4) ∨
    (x = 3 * Real.pi / 4 ∧ y = 3 * Real.pi / 4) ∨
    (x = 5 * Real.pi / 4 ∧ y = Real.pi / 4) ∨
    (x = 7 * Real.pi / 4 ∧ y = 7 * Real.pi / 4) := by
  sorry

end trig_system_solution_l154_154254


namespace perm_prime_count_12345_l154_154729

theorem perm_prime_count_12345 : 
  (∀ x : List ℕ, (x ∈ (List.permutations [1, 2, 3, 4, 5])) → 
    (10^4 * x.head! + 10^3 * x.tail.head! + 10^2 * x.tail.tail.head! + 10 * x.tail.tail.tail.head! + x.tail.tail.tail.tail.head!) % 3 = 0)
  → 
  0 = 0 :=
by
  sorry

end perm_prime_count_12345_l154_154729


namespace find_A_from_conditions_l154_154739

variable (A B C D : ℕ)
variable (h_distinct : A ≠ B) (h_distinct2 : C ≠ D)
variable (h_positive : A > 0) (h_positive2 : B > 0) (h_positive3 : C > 0) (h_positive4 : D > 0)
variable (h_product1 : A * B = 72)
variable (h_product2 : C * D = 72)
variable (h_condition : A - B = C * D)

theorem find_A_from_conditions :
  A = 3 :=
sorry

end find_A_from_conditions_l154_154739


namespace sum_of_cubes_pattern_l154_154265

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  sorry

end sum_of_cubes_pattern_l154_154265


namespace triangle_vertices_l154_154884

theorem triangle_vertices : 
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ x - y = -4 ∧ x = 2 / 3 ∧ y = 14 / 3) ∧ 
  (∃ (x y : ℚ), x - y = -4 ∧ y = -1 ∧ x = -5) ∧
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ y = -1 ∧ x = 7 / 2) :=
by
  sorry

end triangle_vertices_l154_154884


namespace paco_salty_cookies_left_l154_154098

-- Define the initial number of salty cookies Paco had
def initial_salty_cookies : ℕ := 26

-- Define the number of salty cookies Paco ate
def eaten_salty_cookies : ℕ := 9

-- The theorem statement that Paco had 17 salty cookies left
theorem paco_salty_cookies_left : initial_salty_cookies - eaten_salty_cookies = 17 := 
 by
  -- Here we skip the proof by adding sorry
  sorry

end paco_salty_cookies_left_l154_154098


namespace distinct_solutions_square_difference_l154_154633

theorem distinct_solutions_square_difference 
  (Φ φ : ℝ) (h1 : Φ^2 = Φ + 2) (h2 : φ^2 = φ + 2) (h_distinct : Φ ≠ φ) :
  (Φ - φ)^2 = 9 :=
  sorry

end distinct_solutions_square_difference_l154_154633


namespace find_circumcenter_l154_154890

-- Define a quadrilateral with vertices A, B, C, and D
structure Quadrilateral :=
  (A B C D : (ℝ × ℝ))

-- Define the coordinates of the circumcenter
def circumcenter (q : Quadrilateral) : ℝ × ℝ := (6, 1)

-- Given condition that A, B, C, and D are vertices of a quadrilateral
-- Prove that the circumcenter of the circumscribed circle is (6, 1)
theorem find_circumcenter (q : Quadrilateral) : 
  circumcenter q = (6, 1) :=
by sorry

end find_circumcenter_l154_154890


namespace probability_AB_together_l154_154281

theorem probability_AB_together : 
  let total_events := 6
  let ab_together_events := 4
  let probability := ab_together_events / total_events
  probability = 2 / 3 :=
by
  sorry

end probability_AB_together_l154_154281


namespace value_of_fraction_l154_154873

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end value_of_fraction_l154_154873


namespace ratio_used_to_total_apples_l154_154332

noncomputable def total_apples_bonnie : ℕ := 8
noncomputable def total_apples_samuel : ℕ := total_apples_bonnie + 20
noncomputable def eaten_apples_samuel : ℕ := total_apples_samuel / 2
noncomputable def used_for_pie_samuel : ℕ := total_apples_samuel - eaten_apples_samuel - 10

theorem ratio_used_to_total_apples : used_for_pie_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 1 ∧
                                     total_apples_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 7 := by
  sorry

end ratio_used_to_total_apples_l154_154332


namespace opposite_neg_inv_three_l154_154558

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end opposite_neg_inv_three_l154_154558


namespace inequality_holds_iff_l154_154544

theorem inequality_holds_iff (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → x^2 + (a - 4) * x + 4 > 0) ↔ a > 0 :=
by
  sorry

end inequality_holds_iff_l154_154544


namespace motorcycles_in_anytown_l154_154813

variable (t s m : ℕ) -- t: number of trucks, s: number of sedans, m: number of motorcycles
variable (r_trucks r_sedans r_motorcycles : ℕ) -- r_trucks : truck ratio, r_sedans : sedan ratio, r_motorcycles : motorcycle ratio
variable (n_sedans : ℕ) -- n_sedans: number of sedans

theorem motorcycles_in_anytown
  (h1 : r_trucks = 3) -- ratio of trucks
  (h2 : r_sedans = 7) -- ratio of sedans
  (h3 : r_motorcycles = 2) -- ratio of motorcycles
  (h4 : s = 9100) -- number of sedans
  (h5 : s = (r_sedans * n_sedans)) -- relationship between sedans and parts
  (h6 : t = (r_trucks * n_sedans)) -- relationship between trucks and parts
  (h7 : m = (r_motorcycles * n_sedans)) -- relationship between motorcycles and parts
  : m = 2600 := by
    sorry

end motorcycles_in_anytown_l154_154813


namespace find_Z_l154_154360

open Complex

-- Definitions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_Z (Z : ℂ) (h1 : abs Z = 3) (h2 : is_pure_imaginary (Z + (3 * Complex.I))) : Z = 3 * Complex.I :=
by
  sorry

end find_Z_l154_154360


namespace michael_monica_age_ratio_l154_154630

theorem michael_monica_age_ratio
  (x y : ℕ)
  (Patrick Michael Monica : ℕ)
  (h1 : Patrick = 3 * x)
  (h2 : Michael = 5 * x)
  (h3 : Monica = y)
  (h4 : y - Patrick = 64)
  (h5 : Patrick + Michael + Monica = 196) :
  Michael * 5 = Monica * 3 :=
by
  sorry

end michael_monica_age_ratio_l154_154630


namespace distinct_triangles_count_l154_154140

def num_combinations (n k : ℕ) : ℕ := n.choose k

def count_collinear_sets_in_grid (grid_size : ℕ) : ℕ :=
  let rows := grid_size
  let cols := grid_size
  let diagonals := 2
  rows + cols + diagonals

noncomputable def distinct_triangles_in_grid (grid_size n k : ℕ) : ℕ :=
  num_combinations n k - count_collinear_sets_in_grid grid_size

theorem distinct_triangles_count :
  distinct_triangles_in_grid 3 9 3 = 76 := 
by 
  sorry

end distinct_triangles_count_l154_154140


namespace squares_characterization_l154_154934

theorem squares_characterization (n : ℕ) (a b : ℤ) (h_cond : n + 1 = a^2 + (a + 1)^2 ∧ n + 1 = b^2 + 2 * (b + 1)^2) :
  ∃ k l : ℤ, 2 * n + 1 = k^2 ∧ 3 * n + 1 = l^2 :=
sorry

end squares_characterization_l154_154934


namespace min_value_proof_l154_154690

theorem min_value_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x - 2 * y + 3 * z = 0) : 3 = 3 :=
by
  sorry

end min_value_proof_l154_154690


namespace like_monomials_are_same_l154_154831

theorem like_monomials_are_same (m n : ℤ) (h1 : 2 * m + 4 = 8) (h2 : 2 * n - 3 = 5) : m = 2 ∧ n = 4 :=
by
  sorry

end like_monomials_are_same_l154_154831


namespace mr_slinkums_shipments_l154_154060

theorem mr_slinkums_shipments 
  (T : ℝ) 
  (h : (3 / 4) * T = 150) : 
  T = 200 := 
sorry

end mr_slinkums_shipments_l154_154060


namespace ride_count_l154_154700

noncomputable def initial_tickets : ℕ := 287
noncomputable def spent_on_games : ℕ := 134
noncomputable def earned_tickets : ℕ := 32
noncomputable def cost_per_ride : ℕ := 17

theorem ride_count (initial_tickets : ℕ) (spent_on_games : ℕ) (earned_tickets : ℕ) (cost_per_ride : ℕ) : 
  initial_tickets = 287 ∧ spent_on_games = 134 ∧ earned_tickets = 32 ∧ cost_per_ride = 17 → (initial_tickets - spent_on_games + earned_tickets) / cost_per_ride = 10 :=
by
  intros
  sorry

end ride_count_l154_154700


namespace sum_coefficients_l154_154923

theorem sum_coefficients (a1 a2 a3 a4 a5 : ℤ) (h : ∀ x : ℕ, a1 * (x - 1) ^ 4 + a2 * (x - 1) ^ 3 + a3 * (x - 1) ^ 2 + a4 * (x - 1) + a5 = x ^ 4) :
  a2 + a3 + a4 = 14 :=
  sorry

end sum_coefficients_l154_154923


namespace count_even_digits_in_512_base_7_l154_154937

def base7_representation (n : ℕ) : ℕ := 
  sorry  -- Assuming this function correctly computes the base-7 representation of a natural number

def even_digits_count (n : ℕ) : ℕ :=
  sorry  -- Assuming this function correctly counts the even digits in the base-7 representation

theorem count_even_digits_in_512_base_7 : 
  even_digits_count (base7_representation 512) = 0 :=
by
  sorry

end count_even_digits_in_512_base_7_l154_154937


namespace interest_cannot_be_determined_without_investment_amount_l154_154082

theorem interest_cannot_be_determined_without_investment_amount :
  ∀ (interest_rate : ℚ) (price : ℚ) (invested_amount : Option ℚ),
  interest_rate = 0.16 → price = 128 → invested_amount = none → False :=
by
  sorry

end interest_cannot_be_determined_without_investment_amount_l154_154082


namespace polar_coordinates_of_point_l154_154704

theorem polar_coordinates_of_point :
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  r = 4 ∧ theta = Real.pi / 3 :=
by
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  have h_r : r = 4 := by {
    -- Calculation for r
    sorry
  }
  have h_theta : theta = Real.pi / 3 := by {
    -- Calculation for theta
    sorry
  }
  exact ⟨h_r, h_theta⟩

end polar_coordinates_of_point_l154_154704


namespace solution_set_of_inequality_l154_154681

theorem solution_set_of_inequality :
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x | (0 < x ∧ x < 1) ∨ x > 2} :=
by sorry

end solution_set_of_inequality_l154_154681


namespace stack_logs_total_l154_154746

   theorem stack_logs_total (a l d : ℤ) (n : ℕ) (top_logs : ℕ) (h1 : a = 15) (h2 : l = 5) (h3 : d = -2) (h4 : n = ((l - a) / d).natAbs + 1) (h5 : top_logs = 5) : (n / 2 : ℤ) * (a + l) = 60 :=
   by
   sorry
   
end stack_logs_total_l154_154746


namespace algebraic_expression_independence_l154_154049

theorem algebraic_expression_independence (a b : ℝ) (h : ∀ x : ℝ, (x^2 + a*x - (b*x^2 - x - 3)) = 3) : a - b = -2 :=
by
  sorry

end algebraic_expression_independence_l154_154049


namespace cindy_gave_lisa_marbles_l154_154044

-- Definitions for the given conditions
def cindy_initial_marbles : ℕ := 20
def lisa_initial_marbles := cindy_initial_marbles - 5
def lisa_final_marbles := lisa_initial_marbles + 19

-- Theorem we need to prove
theorem cindy_gave_lisa_marbles :
  ∃ n : ℕ, lisa_final_marbles = lisa_initial_marbles + n ∧ n = 19 :=
by
  sorry

end cindy_gave_lisa_marbles_l154_154044


namespace fantasia_max_capacity_reach_l154_154721

def acre_per_person := 1
def land_acres := 40000
def base_population := 500
def population_growth_factor := 4
def years_per_growth_period := 20

def maximum_capacity := land_acres / acre_per_person

def population_at_time (years_from_2000 : ℕ) : ℕ :=
  base_population * population_growth_factor^(years_from_2000 / years_per_growth_period)

theorem fantasia_max_capacity_reach :
  ∃ t : ℕ, t = 60 ∧ population_at_time t = maximum_capacity := by sorry

end fantasia_max_capacity_reach_l154_154721


namespace chef_made_10_cakes_l154_154575

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end chef_made_10_cakes_l154_154575


namespace polynomial_possible_integer_roots_l154_154625

theorem polynomial_possible_integer_roots (b1 b2 : ℤ) :
  ∀ x : ℤ, (x ∣ 18) ↔ (x^3 + b2 * x^2 + b1 * x + 18 = 0) → 
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
by {
  sorry
}


end polynomial_possible_integer_roots_l154_154625


namespace value_depletion_rate_l154_154271

theorem value_depletion_rate (P F : ℝ) (t : ℝ) (r : ℝ) (h₁ : P = 1100) (h₂ : F = 891) (h₃ : t = 2) (decay_formula : F = P * (1 - r) ^ t) : r = 0.1 :=
by 
  sorry

end value_depletion_rate_l154_154271


namespace value_of_8b_l154_154132

theorem value_of_8b (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : b = 2 * a - 3) : 8 * b = -8 := by
  sorry

end value_of_8b_l154_154132


namespace problem_1_minimum_value_problem_2_range_of_a_l154_154363

noncomputable def e : ℝ := Real.exp 1  -- Definition of e as exp(1)

-- Question I:
-- Prove that the minimum value of the function f(x) = e^x - e*x - e is -e.
theorem problem_1_minimum_value :
  ∃ x : ℝ, (∀ y : ℝ, (Real.exp x - e * x - e) ≤ (Real.exp y - e * y - e))
  ∧ (Real.exp x - e * x - e) = -e := 
sorry

-- Question II:
-- Prove that the range of values for a such that f(x) = e^x - a*x - a >= 0 for all x is [0, 1].
theorem problem_2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (Real.exp x - a * x - a) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end problem_1_minimum_value_problem_2_range_of_a_l154_154363


namespace goodColoringsOfPoints_l154_154583

noncomputable def countGoodColorings (k m : ℕ) : ℕ :=
  (k * (k - 1) + 2) * 2 ^ m

theorem goodColoringsOfPoints :
  countGoodColorings 2011 2011 = (2011 * 2010 + 2) * 2 ^ 2011 :=
  by
    sorry

end goodColoringsOfPoints_l154_154583


namespace period_of_sin_sub_cos_l154_154109

open Real

theorem period_of_sin_sub_cos :
  ∃ T > 0, ∀ x, sin x - cos x = sin (x + T) - cos (x + T) ∧ T = 2 * π := sorry

end period_of_sin_sub_cos_l154_154109


namespace no_real_roots_of_quadratic_l154_154038

theorem no_real_roots_of_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end no_real_roots_of_quadratic_l154_154038


namespace exists_consecutive_integers_sum_cube_l154_154582

theorem exists_consecutive_integers_sum_cube :
  ∃ (n : ℤ), ∃ (k : ℤ), 1981 * (n + 990) = k^3 :=
by
  sorry

end exists_consecutive_integers_sum_cube_l154_154582


namespace total_number_of_workers_l154_154789

theorem total_number_of_workers 
  (W : ℕ) 
  (avg_all : ℕ) 
  (n_technicians : ℕ) 
  (avg_technicians : ℕ) 
  (avg_non_technicians : ℕ) :
  avg_all * W = avg_technicians * n_technicians + avg_non_technicians * (W - n_technicians) →
  avg_all = 8000 →
  n_technicians = 7 →
  avg_technicians = 12000 →
  avg_non_technicians = 6000 →
  W = 21 :=
by 
  intro h1 h2 h3 h4 h5
  sorry

end total_number_of_workers_l154_154789


namespace all_statements_imply_negation_l154_154994

theorem all_statements_imply_negation :
  let s1 := (true ∧ true ∧ false)
  let s2 := (false ∧ true ∧ true)
  let s3 := (true ∧ false ∧ true)
  let s4 := (false ∧ false ∧ true)
  (s1 → ¬(true ∧ true ∧ true)) ∧
  (s2 → ¬(true ∧ true ∧ true)) ∧
  (s3 → ¬(true ∧ true ∧ true)) ∧
  (s4 → ¬(true ∧ true ∧ true)) :=
by sorry

end all_statements_imply_negation_l154_154994


namespace pipe_q_fills_in_9_hours_l154_154400

theorem pipe_q_fills_in_9_hours (x : ℝ) :
  (1 / 3 + 1 / x + 1 / 18 = 1 / 2) → x = 9 :=
by {
  sorry
}

end pipe_q_fills_in_9_hours_l154_154400


namespace initial_velocity_is_three_l154_154718

noncomputable def displacement (t : ℝ) : ℝ :=
  3 * t - t^2

theorem initial_velocity_is_three : 
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_is_three_l154_154718


namespace factorize_x4_plus_81_l154_154900

theorem factorize_x4_plus_81 : 
  ∀ x : ℝ, 
    (x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9)) :=
by
  intro x
  sorry

end factorize_x4_plus_81_l154_154900


namespace rectangle_area_l154_154687

theorem rectangle_area (a : ℕ) (w l : ℕ) (h_square_area : a = 36) (h_square_side : w * w = a) (h_rectangle_length : l = 3 * w) : w * l = 108 :=
by
  -- Placeholder for proof
  sorry

end rectangle_area_l154_154687


namespace first_day_reduction_percentage_l154_154000

variables (P x : ℝ)

theorem first_day_reduction_percentage (h : P * (1 - x / 100) * 0.90 = 0.81 * P) : x = 10 :=
sorry

end first_day_reduction_percentage_l154_154000


namespace remaining_distance_l154_154731

-- Definitions of the given conditions
def D : ℕ := 500
def daily_alpha : ℕ := 30
def daily_beta : ℕ := 50
def effective_beta : ℕ := daily_beta / 2

-- Proving the theorem with given conditions
theorem remaining_distance (n : ℕ) (h : n = 25) :
  D - daily_alpha * n = 2 * (D - effective_beta * n) :=
by
  sorry

end remaining_distance_l154_154731


namespace jake_pure_alcohol_l154_154219

theorem jake_pure_alcohol (total_shots : ℕ) (shots_per_split : ℕ) (ounces_per_shot : ℚ) (purity : ℚ) :
  total_shots = 8 →
  shots_per_split = 2 →
  ounces_per_shot = 1.5 →
  purity = 0.5 →
  (total_shots / shots_per_split) * ounces_per_shot * purity = 3 := 
by
  sorry

end jake_pure_alcohol_l154_154219


namespace necessary_but_not_sufficient_condition_for_a_lt_neg_one_l154_154875

theorem necessary_but_not_sufficient_condition_for_a_lt_neg_one (a : ℝ) : 
  (1 / a > -1) ↔ (a < -1) :=
by sorry

end necessary_but_not_sufficient_condition_for_a_lt_neg_one_l154_154875


namespace totalGames_l154_154079

-- Define Jerry's original number of video games
def originalGames : ℕ := 7

-- Define the number of video games Jerry received for his birthday
def birthdayGames : ℕ := 2

-- Statement: Prove that the total number of games Jerry has now is 9
theorem totalGames : originalGames + birthdayGames = 9 := by
  sorry

end totalGames_l154_154079


namespace asian_population_percentage_in_west_l154_154009

theorem asian_population_percentage_in_west
    (NE MW South West : ℕ)
    (H_NE : NE = 2)
    (H_MW : MW = 3)
    (H_South : South = 2)
    (H_West : West = 6)
    : (West * 100) / (NE + MW + South + West) = 46 :=
sorry

end asian_population_percentage_in_west_l154_154009


namespace subtracted_value_l154_154186

theorem subtracted_value (s : ℕ) (h : s = 4) (x : ℕ) (h2 : (s + s^2 - x = 4)) : x = 16 :=
by
  sorry

end subtracted_value_l154_154186


namespace price_verification_l154_154803

noncomputable def price_on_hot_day : ℚ :=
  let P : ℚ := 225 / 172
  1.25 * P

theorem price_verification :
  (32 * 7 * (225 / 172) + 32 * 3 * (1.25 * (225 / 172)) - (32 * 10 * 0.75)) = 210 :=
sorry

end price_verification_l154_154803


namespace candy_distribution_l154_154440

theorem candy_distribution (candies : ℕ) (family_members : ℕ) (required_candies : ℤ) :
  (candies = 45) ∧ (family_members = 5) →
  required_candies = 0 :=
by sorry

end candy_distribution_l154_154440


namespace range_of_x_l154_154781

-- Defining the vectors as given in the conditions
def a (x : ℝ) : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (2, -1)

-- Defining the condition that the angle is obtuse
def is_obtuse (x : ℝ) : Prop := 
  let dot_product := (a x).1 * b.1 + (a x).2 * b.2
  dot_product < 0

-- Defining the condition that vectors are not in opposite directions
def not_opposite_directions (x : ℝ) : Prop := x ≠ -6

-- Proving the required range of x
theorem range_of_x (x : ℝ) :
  is_obtuse x → not_opposite_directions x → x < 3 / 2 :=
sorry

end range_of_x_l154_154781


namespace number_of_paths_l154_154666

-- Definition of vertices
inductive Vertex
| A | B | C | D | E | F | G

-- Edges based on the description
def edges : List (Vertex × Vertex) := [
  (Vertex.A, Vertex.G), (Vertex.G, Vertex.C), (Vertex.G, Vertex.D), (Vertex.C, Vertex.B),
  (Vertex.D, Vertex.C), (Vertex.D, Vertex.F), (Vertex.D, Vertex.E), (Vertex.E, Vertex.F),
  (Vertex.F, Vertex.B), (Vertex.C, Vertex.F), (Vertex.A, Vertex.C), (Vertex.A, Vertex.D)
]

-- Function to count paths from A to B without revisiting any vertex
def countPaths (start : Vertex) (goal : Vertex) (adj : List (Vertex × Vertex)) : Nat :=
sorry

-- The theorem statement
theorem number_of_paths : countPaths Vertex.A Vertex.B edges = 10 :=
sorry

end number_of_paths_l154_154666


namespace sum_groups_eq_250_l154_154702

-- Definitions for each sum
def sum1 : ℕ := 3 + 13 + 23 + 33 + 43
def sum2 : ℕ := 7 + 17 + 27 + 37 + 47

-- Theorem statement that the sum of these groups is 250
theorem sum_groups_eq_250 : sum1 + sum2 = 250 :=
by sorry

end sum_groups_eq_250_l154_154702


namespace mark_charged_more_hours_l154_154514

variable {p k m : ℕ}

theorem mark_charged_more_hours (h1 : p + k + m = 216)
                                (h2 : p = 2 * k)
                                (h3 : p = m / 3) :
                                m - k = 120 :=
sorry

end mark_charged_more_hours_l154_154514


namespace find_middle_integer_l154_154494

theorem find_middle_integer (a b c : ℕ) (h1 : a^2 = 97344) (h2 : c^2 = 98596) (h3 : c = a + 2) : b = a + 1 ∧ b = 313 :=
by
  sorry

end find_middle_integer_l154_154494


namespace sheets_in_height_l154_154268

theorem sheets_in_height (sheets_per_ream : ℕ) (thickness_per_ream : ℝ) (target_thickness : ℝ) 
  (h₀ : sheets_per_ream = 500) (h₁ : thickness_per_ream = 5.0) (h₂ : target_thickness = 7.5) :
  target_thickness / (thickness_per_ream / sheets_per_ream) = 750 :=
by sorry

end sheets_in_height_l154_154268


namespace eating_time_proof_l154_154794

noncomputable def combined_eating_time (time_fat time_thin weight : ℝ) : ℝ :=
  let rate_fat := 1 / time_fat
  let rate_thin := 1 / time_thin
  let combined_rate := rate_fat + rate_thin
  weight / combined_rate

theorem eating_time_proof :
  let time_fat := 12
  let time_thin := 40
  let weight := 5
  combined_eating_time time_fat time_thin weight = (600 / 13) :=
by
  -- placeholder for the proof
  sorry

end eating_time_proof_l154_154794


namespace plane_distance_last_10_seconds_l154_154404

theorem plane_distance_last_10_seconds (s : ℝ → ℝ) (h : ∀ t, s t = 60 * t - 1.5 * t^2) : 
  s 20 - s 10 = 150 := 
by 
  sorry

end plane_distance_last_10_seconds_l154_154404


namespace necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l154_154759

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditionally state that x > -3 is necessary for an acute angle
theorem necessary_condition_for_acute_angle (x : ℝ) :
  dot_product vector_a (vector_b x) > 0 → x > -3 := by
  sorry

-- Define the theorem for necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > -3) → (dot_product vector_a (vector_b x) > 0 ∧ x ≠ 4 / 3) := by
  sorry

end necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l154_154759


namespace coordinates_of_B_l154_154113

theorem coordinates_of_B (m : ℝ) (h : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) :=
by
  -- proof goes here
  sorry

end coordinates_of_B_l154_154113


namespace rational_function_sum_l154_154446

-- Define the problem conditions and the target equality
theorem rational_function_sum (p q : ℝ → ℝ) :
  (∀ x, (p x) / (q x) = (x - 1) / ((x + 1) * (x - 1))) ∧
  (∀ x ≠ -1, q x ≠ 0) ∧
  (q 2 = 3) ∧
  (p 2 = 1) →
  (p x + q x = x^2 + x - 2) := by
  sorry

end rational_function_sum_l154_154446


namespace candy_store_food_colouring_amount_l154_154334

theorem candy_store_food_colouring_amount :
  let lollipop_colour := 5 -- each lollipop uses 5ml of food colouring
  let hard_candy_colour := 20 -- each hard candy uses 20ml of food colouring
  let num_lollipops := 100 -- the candy store makes 100 lollipops in one day
  let num_hard_candies := 5 -- the candy store makes 5 hard candies in one day
  (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600 :=
by
  let lollipop_colour := 5
  let hard_candy_colour := 20
  let num_lollipops := 100
  let num_hard_candies := 5
  show (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600
  sorry

end candy_store_food_colouring_amount_l154_154334


namespace area_of_rectangle_ABCD_l154_154987

-- Definitions based on conditions
def side_length_smaller_square := 2
def area_smaller_square := side_length_smaller_square ^ 2
def side_length_larger_square := 3 * side_length_smaller_square
def area_larger_square := side_length_larger_square ^ 2
def area_rect_ABCD := 2 * area_smaller_square + area_larger_square

-- Lean theorem statement for the proof problem
theorem area_of_rectangle_ABCD : area_rect_ABCD = 44 := by
  sorry

end area_of_rectangle_ABCD_l154_154987


namespace rectangular_coords_transformation_l154_154142

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem rectangular_coords_transformation :
  let ρ := Real.sqrt (2 ^ 2 + (-3) ^ 2 + 6 ^ 2)
  let φ := Real.arccos (6 / ρ)
  let θ := Real.arctan (-3 / 2)
  sphericalToRectangular ρ (Real.pi + θ) φ = (-2, 3, 6) :=
by
  sorry

end rectangular_coords_transformation_l154_154142


namespace find_a3_l154_154853

noncomputable def geometric_term (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n-1)

noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (q^n - 1) / (q - 1)

theorem find_a3 (a : ℝ) (q : ℝ) (h_q : q = 3)
  (h_sum : geometric_sum a q 3 + geometric_sum a q 4 = 53 / 3) :
  geometric_term a q 3 = 3 :=
by
  sorry

end find_a3_l154_154853


namespace max_lines_between_points_l154_154498

noncomputable def maxLines (points : Nat) := 
  let deg := [1, 2, 3, 4, 5]
  (1 * (points - 1) + 2 * (points - 2) + 3 * (points - 3) + 4 * (points - 4) + 5 * (points - 5)) / 2

theorem max_lines_between_points :
  ∀ (n : Nat), n = 15 → maxLines n = 85 :=
by
  intros n hn
  sorry

end max_lines_between_points_l154_154498


namespace basketball_probability_third_shot_l154_154758

theorem basketball_probability_third_shot
  (p1 : ℚ) (p2_given_made1 : ℚ) (p2_given_missed1 : ℚ) (p3_given_made2 : ℚ) (p3_given_missed2 : ℚ) :
  p1 = 2 / 3 → p2_given_made1 = 2 / 3 → p2_given_missed1 = 1 / 3 → p3_given_made2 = 2 / 3 → p3_given_missed2 = 2 / 3 →
  (p1 * p2_given_made1 * p3_given_made2 + p1 * p2_given_missed1 * p3_given_misseds2 + 
   (1 - p1) * p2_given_made1 * p3_given_made2 + (1 - p1) * p2_given_missed1 * p3_given_missed2) = 14 / 27 :=
by
  sorry

end basketball_probability_third_shot_l154_154758


namespace metal_waste_l154_154733

theorem metal_waste (a b : ℝ) (h : a < b) :
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * radius^2
  let side_square := a / Real.sqrt 2
  let area_square := side_square^2
  area_rectangle - area_square = a * b - ( a ^ 2 ) / 2 := by
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * (radius ^ 2)
  let side_square := a / Real.sqrt 2
  let area_square := side_square ^ 2
  sorry

end metal_waste_l154_154733


namespace Tyler_needs_more_eggs_l154_154910

noncomputable def recipe_eggs : ℕ := 2
noncomputable def recipe_milk : ℕ := 4
noncomputable def num_people : ℕ := 8
noncomputable def eggs_in_fridge : ℕ := 3

theorem Tyler_needs_more_eggs (recipe_eggs recipe_milk num_people eggs_in_fridge : ℕ)
  (h1 : recipe_eggs = 2)
  (h2 : recipe_milk = 4)
  (h3 : num_people = 8)
  (h4 : eggs_in_fridge = 3) :
  (num_people / 4) * recipe_eggs - eggs_in_fridge = 1 :=
by
  sorry

end Tyler_needs_more_eggs_l154_154910


namespace correct_divisor_l154_154966

theorem correct_divisor (D : ℕ) (X : ℕ) (H1 : X = 70 * (D + 12)) (H2 : X = 40 * D) : D = 28 := 
by 
  sorry

end correct_divisor_l154_154966


namespace num_rooms_l154_154948

theorem num_rooms (r1 r2 w1 w2 p w_paint : ℕ) (h_r1 : r1 = 5) (h_r2 : r2 = 4) (h_w1 : w1 = 4) (h_w2 : w2 = 5)
    (h_p : p = 5) (h_w_paint : w_paint = 8) (h_total_walls_family : p * w_paint = (r1 * w1 + r2 * w2)) :
    (r1 + r2 = 9) :=
by
  sorry

end num_rooms_l154_154948


namespace max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l154_154569

theorem max_lg_sum_eq_one {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ u, u = Real.log x + Real.log y → u ≤ 1 :=
sorry

theorem min_inv_sum_eq_specific_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ v, v = (1 / x) + (1 / y) → v ≥ (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l154_154569


namespace prob_yellow_straight_l154_154885

variable {P : ℕ → ℕ → ℚ}
-- Defining the probabilities of the given events
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2
def prob_rose : ℚ := 1 / 4
def prob_daffodil : ℚ := 1 / 2
def prob_tulip : ℚ := 1 / 4
def prob_rose_straight : ℚ := 1 / 6
def prob_daffodil_curved : ℚ := 1 / 3
def prob_tulip_straight : ℚ := 1 / 8

/-- The probability of picking a yellow and straight-petaled flower is 1/6 -/
theorem prob_yellow_straight : P 1 1 = 1 / 6 := sorry

end prob_yellow_straight_l154_154885


namespace cloth_sold_l154_154580

theorem cloth_sold (C S P: ℝ) (N : ℕ) 
  (h1 : S = 3 * C)
  (h2 : P = 10 * S)
  (h3 : (200 : ℝ) = (P / (N * C)) * 100) : N = 15 := 
sorry

end cloth_sold_l154_154580


namespace pam_bags_equiv_gerald_bags_l154_154040

theorem pam_bags_equiv_gerald_bags :
  ∀ (total_apples pam_bags apples_per_gerald_bag : ℕ), 
    total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 → 
    (total_apples / pam_bags) / apples_per_gerald_bag = 3 :=
by
  intros total_apples pam_bags apples_per_gerald_bag h
  obtain ⟨ht, hp, hg⟩ : total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 := h
  sorry

end pam_bags_equiv_gerald_bags_l154_154040


namespace value_of_a_l154_154689

theorem value_of_a (a : ℝ) (h_neg : a < 0) (h_f : ∀ (x : ℝ), (0 < x ∧ x ≤ 1) → 
  (x + 4 * a / x - a < 0)) : a ≤ -1 / 3 := 
sorry

end value_of_a_l154_154689


namespace gain_percent_calculation_l154_154754

theorem gain_percent_calculation (gain_paise : ℕ) (cost_price_rupees : ℕ) (rupees_to_paise : ℕ)
  (h_gain_paise : gain_paise = 70)
  (h_cost_price_rupees : cost_price_rupees = 70)
  (h_rupees_to_paise : rupees_to_paise = 100) :
  ((gain_paise / rupees_to_paise) / cost_price_rupees) * 100 = 1 :=
by
  -- Placeholder to indicate the need for proof
  sorry

end gain_percent_calculation_l154_154754


namespace smallest_uv_non_factor_of_48_l154_154055

theorem smallest_uv_non_factor_of_48 :
  ∃ (u v : ℕ) (hu : u ∣ 48) (hv : v ∣ 48), u ≠ v ∧ ¬ (u * v ∣ 48) ∧ u * v = 18 :=
sorry

end smallest_uv_non_factor_of_48_l154_154055


namespace find_number_of_toonies_l154_154612

variable (L T : ℕ)

def condition1 : Prop := L + T = 10
def condition2 : Prop := L + 2 * T = 14

theorem find_number_of_toonies (h1 : condition1 L T) (h2 : condition2 L T) : T = 4 :=
by
  sorry

end find_number_of_toonies_l154_154612


namespace solve_equation_l154_154449

/-- 
  Given the equation:
    ∀ x, (x = 2 ∨ (3 < x ∧ x < 4)) ↔ (⌊(1/x) * ⌊x⌋^2⌋ = 2),
  where ⌊u⌋ represents the greatest integer less than or equal to u.
-/
theorem solve_equation (x : ℝ) : (x = 2 ∨ (3 < x ∧ x < 4)) ↔ ⌊(1/x) * ⌊x⌋^2⌋ = 2 := 
sorry

end solve_equation_l154_154449


namespace solve_equation_l154_154868

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end solve_equation_l154_154868


namespace n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l154_154644

theorem n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd :
  ∀ (n m : ℤ), (n^2 + m^3) % 2 ≠ 0 → (n + m) % 2 = 1 :=
by sorry

end n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l154_154644


namespace time_after_9999_seconds_l154_154918

theorem time_after_9999_seconds:
  let initial_hours := 5
  let initial_minutes := 45
  let initial_seconds := 0
  let added_seconds := 9999
  let total_seconds := initial_seconds + added_seconds
  let total_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let final_hours := (initial_hours + total_hours + (initial_minutes + remaining_minutes) / 60) % 24
  let final_minutes := (initial_minutes + remaining_minutes) % 60
  initial_hours = 5 →
  initial_minutes = 45 →
  initial_seconds = 0 →
  added_seconds = 9999 →
  final_hours = 8 ∧ final_minutes = 31 ∧ remaining_seconds = 39 :=
by
  intros
  sorry

end time_after_9999_seconds_l154_154918


namespace required_pumps_l154_154691

-- Define the conditions in Lean
variables (x a b n : ℝ)

-- Condition 1: x + 40a = 80b
def condition1 : Prop := x + 40 * a = 2 * 40 * b

-- Condition 2: x + 16a = 64b
def condition2 : Prop := x + 16 * a = 4 * 16 * b

-- Main theorem: Given the conditions, prove that n >= 6 satisfies the remaining requirement
theorem required_pumps (h1 : condition1 x a b) (h2 : condition2 x a b) : n >= 6 :=
by
  sorry

end required_pumps_l154_154691


namespace ammonium_bromide_total_weight_l154_154788

noncomputable def nitrogen_weight : ℝ := 14.01
noncomputable def hydrogen_weight : ℝ := 1.01
noncomputable def bromine_weight : ℝ := 79.90
noncomputable def ammonium_bromide_weight : ℝ := nitrogen_weight + 4 * hydrogen_weight + bromine_weight
noncomputable def moles : ℝ := 5
noncomputable def total_weight : ℝ := moles * ammonium_bromide_weight

theorem ammonium_bromide_total_weight :
  total_weight = 489.75 :=
by
  -- The proof is omitted.
  sorry

end ammonium_bromide_total_weight_l154_154788


namespace max_inscribed_triangle_area_sum_l154_154719

noncomputable def inscribed_triangle_area (a b : ℝ) (h_a : a = 12) (h_b : b = 13) : ℝ :=
  let s := min (a / (Real.sqrt 3 / 2)) (b / (1 / 2))
  (Real.sqrt 3 / 4) * s^2

theorem max_inscribed_triangle_area_sum :
  inscribed_triangle_area 12 13 (by rfl) (by rfl) = 48 * Real.sqrt 3 - 0 :=
by
  sorry

#eval 48 + 3 + 0
-- Expected Result: 51

end max_inscribed_triangle_area_sum_l154_154719


namespace divide_by_repeating_decimal_l154_154051

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l154_154051


namespace inequality_proof_l154_154205

theorem inequality_proof (a b : ℝ) (x y : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_x : 0 < x) (h_y : 0 < y) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) :=
sorry

end inequality_proof_l154_154205


namespace adult_ticket_cost_l154_154063

theorem adult_ticket_cost 
  (child_ticket_cost : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (adults_attended : ℕ)
  (children_tickets : ℕ)
  (adults_ticket_cost : ℕ)
  (h1 : child_ticket_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_cost = 1875)
  (h4 : adults_attended = 175)
  (h5 : children_tickets = total_tickets - adults_attended)
  (h6 : total_cost = adults_attended * adults_ticket_cost + children_tickets * child_ticket_cost) :
  adults_ticket_cost = 9 :=
sorry

end adult_ticket_cost_l154_154063


namespace sum_of_cubes_eq_neg2_l154_154162

theorem sum_of_cubes_eq_neg2 (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := 
sorry

end sum_of_cubes_eq_neg2_l154_154162


namespace father_children_problem_l154_154591

theorem father_children_problem {F C n : ℕ} 
  (hF_C : F = C) 
  (sum_ages_after_15_years : C + 15 * n = 2 * (F + 15)) 
  (father_age : F = 75) : 
  n = 7 :=
by
  sorry

end father_children_problem_l154_154591


namespace isosceles_triangle_largest_angle_l154_154741

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end isosceles_triangle_largest_angle_l154_154741


namespace slope_point_on_line_l154_154087

theorem slope_point_on_line (b : ℝ) (h1 : ∃ x, x + b = 30) (h2 : (b / (30 - b)) = 4) : b = 24 :=
  sorry

end slope_point_on_line_l154_154087


namespace rowing_speed_l154_154458

theorem rowing_speed :
  ∀ (initial_width final_width increase_per_10m : ℝ) (time_seconds : ℝ)
  (yards_to_meters : ℝ → ℝ) (width_increase_in_yards : ℝ) (distance_10m_segments : ℝ) 
  (total_distance : ℝ),
  initial_width = 50 →
  final_width = 80 →
  increase_per_10m = 2 →
  time_seconds = 30 →
  yards_to_meters 1 = 0.9144 →
  width_increase_in_yards = (final_width - initial_width) →
  width_increase_in_yards * (yards_to_meters 1) = 27.432 →
  distance_10m_segments = (width_increase_in_yards * (yards_to_meters 1)) / 10 →
  total_distance = distance_10m_segments * 10 →
  (total_distance / time_seconds) = 0.9144 :=
by
  intros initial_width final_width increase_per_10m time_seconds yards_to_meters 
        width_increase_in_yards distance_10m_segments total_distance
  sorry

end rowing_speed_l154_154458


namespace net_increase_in_wealth_l154_154870

-- Definitions for yearly changes and fees
def firstYearChange (initialAmt : ℝ) : ℝ := initialAmt * 1.75 - 0.02 * initialAmt * 1.75
def secondYearChange (amt : ℝ) : ℝ := amt * 0.7 - 0.02 * amt * 0.7
def thirdYearChange (amt : ℝ) : ℝ := amt * 1.45 - 0.02 * amt * 1.45
def fourthYearChange (amt : ℝ) : ℝ := amt * 0.85 - 0.02 * amt * 0.85

-- Total Value after 4th year accounting all changes and fees
def totalAfterFourYears (initialAmt : ℝ) : ℝ :=
  let afterFirstYear := firstYearChange initialAmt
  let afterSecondYear := secondYearChange afterFirstYear
  let afterThirdYear := thirdYearChange afterSecondYear
  fourthYearChange afterThirdYear

-- Capital gains tax calculation
def capitalGainsTax (initialAmt finalAmt : ℝ) : ℝ :=
  0.20 * (finalAmt - initialAmt)

-- Net value after taxes
def netValueAfterTaxes (initialAmt : ℝ) : ℝ :=
  let total := totalAfterFourYears initialAmt
  total - capitalGainsTax initialAmt total

-- Main theorem statement
theorem net_increase_in_wealth :
  ∀ (initialAmt : ℝ), netValueAfterTaxes initialAmt = initialAmt * 1.31408238206 := sorry

end net_increase_in_wealth_l154_154870


namespace possible_winning_scores_count_l154_154804

def total_runners := 15
def total_score := (total_runners * (total_runners + 1)) / 2

def min_score := 15
def max_potential_score := 39

def is_valid_winning_score (score : ℕ) : Prop :=
  min_score ≤ score ∧ score ≤ max_potential_score

theorem possible_winning_scores_count : 
  ∃ scores : Finset ℕ, ∀ score ∈ scores, is_valid_winning_score score ∧ Finset.card scores = 25 := 
sorry

end possible_winning_scores_count_l154_154804


namespace jeremy_goal_product_l154_154525

theorem jeremy_goal_product 
  (g1 g2 g3 g4 g5 : ℕ) 
  (total5 : g1 + g2 + g3 + g4 + g5 = 13)
  (g6 g7 : ℕ) 
  (h6 : g6 < 10) 
  (h7 : g7 < 10) 
  (avg6 : (13 + g6) % 6 = 0) 
  (avg7 : (13 + g6 + g7) % 7 = 0) :
  g6 * g7 = 15 := 
sorry

end jeremy_goal_product_l154_154525


namespace incorrect_option_C_l154_154461

def line (α : Type*) := α → Prop
def plane (α : Type*) := α → Prop

variables {α : Type*} (m n : line α) (a b : plane α)

def parallel (m n : line α) : Prop := ∀ x, m x → n x
def perpendicular (m n : line α) : Prop := ∃ x, m x ∧ n x

def lies_in (m : line α) (a : plane α) : Prop := ∀ x, m x → a x

theorem incorrect_option_C (h : lies_in m a) : ¬ (parallel m n ∧ lies_in m a → parallel n a) :=
sorry

end incorrect_option_C_l154_154461


namespace sum_of_constants_l154_154152

theorem sum_of_constants (x a b : ℤ) (h : x^2 - 10 * x + 15 = 0) 
    (h1 : (x + a)^2 = b) : a + b = 5 := 
sorry

end sum_of_constants_l154_154152


namespace max_ahead_distance_l154_154564

noncomputable def distance_run_by_alex (initial_distance ahead1 ahead_max_runs final_ahead : ℝ) : ℝ :=
  initial_distance + ahead1 + ahead_max_runs + final_ahead

theorem max_ahead_distance :
  let initial_distance := 200
  let ahead1 := 300
  let final_ahead := 440
  let total_road := 5000
  let distance_remaining := 3890
  let distance_run_alex := total_road - distance_remaining
  ∃ X : ℝ, distance_run_by_alex initial_distance ahead1 X final_ahead = distance_run_alex ∧ X = 170 :=
by
  intro initial_distance ahead1 final_ahead total_road distance_remaining distance_run_alex
  use 170
  simp [initial_distance, ahead1, final_ahead, total_road, distance_remaining, distance_run_alex, distance_run_by_alex]
  sorry

end max_ahead_distance_l154_154564


namespace linear_function_decreases_l154_154199

theorem linear_function_decreases (m b x : ℝ) (h : m < 0) : 
  ∃ y : ℝ, y = m * x + b ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) :=
by 
  sorry

end linear_function_decreases_l154_154199


namespace necessary_but_not_sufficient_condition_l154_154126

theorem necessary_but_not_sufficient_condition (a b : ℝ) : 
  (a > b → a + 1 > b) ∧ (∃ a b : ℝ, a + 1 > b ∧ ¬ a > b) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l154_154126


namespace total_payment_divisible_by_25_l154_154422

theorem total_payment_divisible_by_25 (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 9) : 
  (2005 + B * 1000) % 25 = 0 :=
by
  sorry

end total_payment_divisible_by_25_l154_154422


namespace percentage_of_female_officers_on_duty_l154_154031

-- Declare the conditions
def total_officers_on_duty : ℕ := 100
def female_officers_on_duty : ℕ := 50
def total_female_officers : ℕ := 250

-- The theorem to prove
theorem percentage_of_female_officers_on_duty :
  (female_officers_on_duty / total_female_officers) * 100 = 20 := 
sorry

end percentage_of_female_officers_on_duty_l154_154031


namespace find_a_and_union_l154_154772

noncomputable def A (a : ℝ) : Set ℝ := { -4, 2 * a - 1, a ^ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { a - 5, 1 - a, 9 }

theorem find_a_and_union {a : ℝ}
  (h : A a ∩ B a = {9}): 
  a = -3 ∧ A a ∪ B a = {-8, -7, -4, 4, 9} :=
by
  sorry

end find_a_and_union_l154_154772


namespace geometric_series_sum_infinity_l154_154242

theorem geometric_series_sum_infinity (a₁ : ℝ) (q : ℝ) (S₆ S₃ : ℝ)
  (h₁ : a₁ = 3)
  (h₂ : S₆ / S₃ = 7 / 8)
  (h₃ : S₆ = a₁ * (1 - q ^ 6) / (1 - q))
  (h₄ : S₃ = a₁ * (1 - q ^ 3) / (1 - q)) :
  ∑' i : ℕ, a₁ * q ^ i = 2 := by
  sorry

end geometric_series_sum_infinity_l154_154242


namespace village_population_percentage_l154_154909

theorem village_population_percentage (P0 P2 P1 : ℝ) (x : ℝ)
  (hP0 : P0 = 7800)
  (hP2 : P2 = 5265)
  (hP1 : P1 = P0 * (1 - x / 100))
  (hP2_eq : P2 = P1 * 0.75) :
  x = 10 :=
by
  sorry

end village_population_percentage_l154_154909


namespace smallest_a_condition_l154_154043

theorem smallest_a_condition
  (a b : ℝ)
  (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_eq : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a = 15 :=
sorry

end smallest_a_condition_l154_154043


namespace choose_two_items_proof_l154_154213

   def number_of_ways_to_choose_two_items (n : ℕ) : ℕ :=
     n * (n - 1) / 2

   theorem choose_two_items_proof (n : ℕ) : number_of_ways_to_choose_two_items n = (n * (n - 1)) / 2 :=
   by
     sorry
   
end choose_two_items_proof_l154_154213


namespace find_number_l154_154335

theorem find_number (x : ℝ) (h : 0.85 * x = (4 / 5) * 25 + 14) : x = 40 :=
sorry

end find_number_l154_154335


namespace ellipse_foci_coordinates_l154_154556

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
    x^2 / 16 + y^2 / 25 = 1 → (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3) :=
by
  sorry

end ellipse_foci_coordinates_l154_154556


namespace calories_per_person_l154_154761

theorem calories_per_person 
  (oranges : ℕ)
  (pieces_per_orange : ℕ)
  (people : ℕ)
  (calories_per_orange : ℝ)
  (h_oranges : oranges = 7)
  (h_pieces_per_orange : pieces_per_orange = 12)
  (h_people : people = 6)
  (h_calories_per_orange : calories_per_orange = 80.0) :
  (oranges * pieces_per_orange / people) * (calories_per_orange / pieces_per_orange) = 93.3338 :=
by
  sorry

end calories_per_person_l154_154761


namespace smallest_N_l154_154683

-- Definitions for conditions
variable (a b c : ℕ) (N : ℕ)

-- Define the conditions for the given problem
def valid_block (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) = 252

def block_volume (a b c : ℕ) : ℕ := a * b * c

-- The target theorem to be proved
theorem smallest_N (h : valid_block a b c) : N = 224 :=
  sorry

end smallest_N_l154_154683


namespace sum_of_tens_l154_154555

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l154_154555


namespace find_minutes_per_mile_l154_154822

-- Conditions
def num_of_movies : ℕ := 2
def avg_length_of_movie_hours : ℝ := 1.5
def total_distance_miles : ℝ := 15

-- Question and proof target
theorem find_minutes_per_mile :
  (num_of_movies * avg_length_of_movie_hours * 60) / total_distance_miles = 12 :=
by
  -- Insert the proof here (not required as per the task instructions)
  sorry

end find_minutes_per_mile_l154_154822


namespace sequence_formula_l154_154015

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 3^n) :
  ∀ n : ℕ, a n = (3^n - 1) / 2 :=
sorry

end sequence_formula_l154_154015


namespace find_y_l154_154155

theorem find_y (x y z : ℤ) (h₁ : x + y + z = 355) (h₂ : x - y = 200) (h₃ : x + z = 500) : y = -145 :=
by
  sorry

end find_y_l154_154155
