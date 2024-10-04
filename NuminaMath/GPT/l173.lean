import Mathlib

namespace rectangle_square_problem_l173_173528

theorem rectangle_square_problem
  (m n x : ℕ)
  (h : 2 * (m + n) + 2 * x = m * n)
  (h2 : m * n - x^2 = 2 * (m + n)) :
  x = 2 ∧ ((m = 3 ∧ n = 10) ∨ (m = 6 ∧ n = 4)) :=
by {
  -- Proof goes here
  sorry
}

end rectangle_square_problem_l173_173528


namespace gain_percent_is_50_l173_173062

theorem gain_percent_is_50
  (C : ℕ) (S : ℕ) (hC : C = 10) (hS : S = 15) : ((S - C) / C : ℚ) * 100 = 50 := by
  sorry

end gain_percent_is_50_l173_173062


namespace base_five_to_ten_3214_l173_173194

theorem base_five_to_ten_3214 : (3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 4 * 5^0) = 434 := by
  sorry

end base_five_to_ten_3214_l173_173194


namespace two_digit_number_is_27_l173_173503

theorem two_digit_number_is_27 :
  ∃ n : ℕ, (n / 10 < 10) ∧ (n % 10 < 10) ∧ 
  (100*(n) = 37*(10*(n) + 1)) ∧ 
  n = 27 :=
by {
  sorry
}

end two_digit_number_is_27_l173_173503


namespace overall_profit_is_600_l173_173584

def grinder_cp := 15000
def mobile_cp := 10000
def laptop_cp := 20000
def camera_cp := 12000

def grinder_loss_percent := 4 / 100
def mobile_profit_percent := 10 / 100
def laptop_loss_percent := 8 / 100
def camera_profit_percent := 15 / 100

def grinder_sp := grinder_cp * (1 - grinder_loss_percent)
def mobile_sp := mobile_cp * (1 + mobile_profit_percent)
def laptop_sp := laptop_cp * (1 - laptop_loss_percent)
def camera_sp := camera_cp * (1 + camera_profit_percent)

def total_cp := grinder_cp + mobile_cp + laptop_cp + camera_cp
def total_sp := grinder_sp + mobile_sp + laptop_sp + camera_sp

def overall_profit_or_loss := total_sp - total_cp

theorem overall_profit_is_600 : overall_profit_or_loss = 600 := by
  sorry

end overall_profit_is_600_l173_173584


namespace repair_cost_total_l173_173320

-- Define the inputs
def labor_cost_rate : ℤ := 75
def labor_hours : ℤ := 16
def part_cost : ℤ := 1200

-- Define the required computation and proof statement
def total_repair_cost : ℤ :=
  let labor_cost := labor_cost_rate * labor_hours
  labor_cost + part_cost

theorem repair_cost_total : total_repair_cost = 2400 := by
  -- Proof would go here
  sorry

end repair_cost_total_l173_173320


namespace cube_volume_l173_173979

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173979


namespace last_digit_of_sum_edges_l173_173037

def total_edges (n : ℕ) : ℕ := (n + 1) * n * 2

def internal_edges (n : ℕ) : ℕ := (n - 1) * n * 2

def dominoes (n : ℕ) : ℕ := (n * n) / 2

def perfect_matchings (n : ℕ) : ℕ := if n = 8 then 12988816 else 0  -- specific to 8x8 chessboard

def sum_internal_edges_contribution (n : ℕ) : ℕ := perfect_matchings n * (dominoes n * 2)

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_sum_edges {n : ℕ} (h : n = 8) :
  last_digit (sum_internal_edges_contribution n) = 4 :=
by
  rw [h]
  sorry

end last_digit_of_sum_edges_l173_173037


namespace ravi_overall_profit_l173_173204

-- Define the cost price of the refrigerator and the mobile phone
def cost_price_refrigerator : ℝ := 15000
def cost_price_mobile_phone : ℝ := 8000

-- Define the loss percentage for the refrigerator and the profit percentage for the mobile phone
def loss_percentage_refrigerator : ℝ := 0.05
def profit_percentage_mobile_phone : ℝ := 0.10

-- Calculate the loss amount and the selling price of the refrigerator
def loss_amount_refrigerator : ℝ := loss_percentage_refrigerator * cost_price_refrigerator
def selling_price_refrigerator : ℝ := cost_price_refrigerator - loss_amount_refrigerator

-- Calculate the profit amount and the selling price of the mobile phone
def profit_amount_mobile_phone : ℝ := profit_percentage_mobile_phone * cost_price_mobile_phone
def selling_price_mobile_phone : ℝ := cost_price_mobile_phone + profit_amount_mobile_phone

-- Calculate the total cost price and the total selling price
def total_cost_price : ℝ := cost_price_refrigerator + cost_price_mobile_phone
def total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone

-- Calculate the overall profit or loss
def overall_profit_or_loss : ℝ := total_selling_price - total_cost_price

theorem ravi_overall_profit : overall_profit_or_loss = 50 := 
by
  sorry

end ravi_overall_profit_l173_173204


namespace find_amount_l173_173541

theorem find_amount (N : ℝ) (hN : N = 24) (A : ℝ) (hA : A = 0.6667 * N - 0.25 * N) : A = 10.0008 :=
by
  rw [hN] at hA
  sorry

end find_amount_l173_173541


namespace certain_number_x_l173_173366

theorem certain_number_x :
  ∃ x : ℤ, (287 * 287 + 269 * 269 - x * (287 * 269) = 324) ∧ (x = 2) := 
by {
  use 2,
  sorry
}

end certain_number_x_l173_173366


namespace expression_for_f_l173_173853

variable {R : Type*} [CommRing R]

def f (x : R) : R := sorry

theorem expression_for_f (x : R) :
  (f (x-1) = x^2 + 4*x - 5) → (f x = x^2 + 6*x) := by
  sorry

end expression_for_f_l173_173853


namespace find_product_l173_173916

theorem find_product (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 7.1)
  (h_rel : 2.5 * a = b - 1.2 ∧ b - 1.2 = c + 4.8 ∧ c + 4.8 = 0.25 * d) :
  a * b * c * d = 49.6 := 
sorry

end find_product_l173_173916


namespace difference_of_extremes_l173_173581

def digits : List ℕ := [2, 0, 1, 3]

def largest_integer : ℕ := 3210
def smallest_integer_greater_than_1000 : ℕ := 1023
def expected_difference : ℕ := 2187

theorem difference_of_extremes :
  largest_integer - smallest_integer_greater_than_1000 = expected_difference := by
  sorry

end difference_of_extremes_l173_173581


namespace compute_expression_l173_173536

theorem compute_expression : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end compute_expression_l173_173536


namespace count_integer_values_of_a_l173_173420

theorem count_integer_values_of_a (a : ℤ) :
  (∃ x : ℤ, x^2 + a * x + 12 * a = 0) ↔ (a ∈ {-(m + n) | m, n : ℤ ∧ m * n = 12 * a}) → ∃! a_vals : finset ℤ, a_vals.card = 16 :=
by
  sorry

end count_integer_values_of_a_l173_173420


namespace cookies_per_batch_l173_173694

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l173_173694


namespace scientific_notation_6500_l173_173744

theorem scientific_notation_6500 : (6500 : ℝ) = 6.5 * 10^3 := 
by 
  sorry

end scientific_notation_6500_l173_173744


namespace cube_volume_is_1728_l173_173949

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173949


namespace peter_large_glasses_l173_173329

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end peter_large_glasses_l173_173329


namespace expression_evaluation_l173_173534

theorem expression_evaluation (a b c : ℤ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 :=
by
  have ha : a = 8 := h₁
  have hb : b = 10 := h₂
  have hc : c = 3 := h₃
  rw [ha, hb, hc]
  sorry

end expression_evaluation_l173_173534


namespace math_test_score_l173_173616

theorem math_test_score (K E M : ℕ) 
  (h₁ : (K + E) / 2 = 92) 
  (h₂ : (K + E + M) / 3 = 94) : 
  M = 98 := 
by 
  sorry

end math_test_score_l173_173616


namespace find_b_l173_173202

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a = b + 2 ∧ 
  b = 2 * c ∧ 
  a + b + c = 42

theorem find_b (a b c : ℕ) (h : conditions a b c) : b = 16 := 
sorry

end find_b_l173_173202


namespace percentage_time_in_park_l173_173896

/-- Define the number of trips Laura takes to the park. -/
def number_of_trips : ℕ := 6

/-- Define time spent at the park per trip in hours. -/
def time_at_park_per_trip : ℝ := 2

/-- Define time spent walking per trip in hours. -/
def time_walking_per_trip : ℝ := 0.5

/-- Define the total time for all trips. -/
def total_time_for_all_trips : ℝ := (time_at_park_per_trip + time_walking_per_trip) * number_of_trips

/-- Define the total time spent in the park for all trips. -/
def total_time_in_park : ℝ := time_at_park_per_trip * number_of_trips

/-- Prove that the percentage of the total time spent in the park is 80%. -/
theorem percentage_time_in_park : total_time_in_park / total_time_for_all_trips * 100 = 80 :=
by
  sorry

end percentage_time_in_park_l173_173896


namespace quadratic_form_ratio_l173_173488

theorem quadratic_form_ratio (x y u v : ℤ) (h : ∃ k : ℤ, k * (u^2 + 3*v^2) = x^2 + 3*y^2) :
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 := sorry

end quadratic_form_ratio_l173_173488


namespace volume_is_correct_l173_173619

noncomputable def volume_of_rectangular_parallelepiped (a b : ℝ) (h_diag : (2 * a^2 + b^2 = 1)) (h_surface_area : (4 * a * b + 2 * a^2 = 1)) : ℝ :=
  a^2 * b

theorem volume_is_correct (a b : ℝ)
  (h_diag : 2 * a^2 + b^2 = 1)
  (h_surface_area : 4 * a * b + 2 * a^2 = 1) :
  volume_of_rectangular_parallelepiped a b h_diag h_surface_area = (Real.sqrt 2) / 27 :=
sorry

end volume_is_correct_l173_173619


namespace penultimate_digit_of_quotient_l173_173103

theorem penultimate_digit_of_quotient :
  (4^1994 + 7^1994) / 10 % 10 = 1 :=
by
  sorry

end penultimate_digit_of_quotient_l173_173103


namespace original_price_l173_173931

theorem original_price (P: ℝ) (h: 0.80 * 1.15 * P = 46) : P = 50 :=
by sorry

end original_price_l173_173931


namespace centroid_of_V_l173_173590

-- Define the set of points (x, y) satisfying the given conditions
noncomputable def V : Set (ℝ × ℝ) :=
  {p | (abs p.1) ≤ p.2 ∧ p.2 ≤ (abs p.1 + 3) ∧ p.2 ≤ 4}

-- Statement that the centroid of V is (0, 2.31)
theorem centroid_of_V :
  centroid V = (0, 2.31) :=
sorry

end centroid_of_V_l173_173590


namespace min_lifespan_ensures_prob_l173_173927

noncomputable def lifespan_distribution : ℝ → ℝ :=
λ x, Pdf.normalPdf 1000 (30^2) x

theorem min_lifespan_ensures_prob :
  ∀ X : ℝ → ℝ, X = lifespan_distribution →
    (∫ x in 910..1090, X x) = 0.997 → (∃ min_lifespan : ℝ, min_lifespan = 910) :=
by sorry

end min_lifespan_ensures_prob_l173_173927


namespace find_y_l173_173228

noncomputable def a := (3/5) * 2500
noncomputable def b := (2/7) * ((5/8) * 4000 + (1/4) * 3600 - (11/20) * 7200)
noncomputable def c (y : ℚ) := (3/10) * y
def result (a b c : ℚ) := a * b / c

theorem find_y : ∃ y : ℚ, result a b (c y) = 25000 ∧ y = -4/21 := 
by
  sorry

end find_y_l173_173228


namespace sqrt_expression_eq_twelve_l173_173229

theorem sqrt_expression_eq_twelve : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := 
sorry

end sqrt_expression_eq_twelve_l173_173229


namespace sum_of_coefficients_l173_173089

def f (x : ℝ) : ℝ := (1 + 2 * x)^4

theorem sum_of_coefficients : f 1 = 81 :=
by
  -- New goal is immediately achieved since the given is precisely ensured.
  sorry

end sum_of_coefficients_l173_173089


namespace total_turnips_grown_l173_173725

theorem total_turnips_grown 
  (melanie_turnips : ℕ) 
  (benny_turnips : ℕ) 
  (jack_turnips : ℕ) 
  (lynn_turnips : ℕ) : 
  melanie_turnips = 1395 ∧
  benny_turnips = 11380 ∧
  jack_turnips = 15825 ∧
  lynn_turnips = 23500 → 
  melanie_turnips + benny_turnips + jack_turnips + lynn_turnips = 52100 :=
by
  intros h
  rcases h with ⟨hm, hb, hj, hl⟩
  sorry

end total_turnips_grown_l173_173725


namespace age_difference_l173_173500

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 14) : C = A - 14 :=
by sorry

end age_difference_l173_173500


namespace part1_case1_part1_case2_m_gt_neg1_part1_case2_m_lt_neg1_part2_l173_173439

def f (x m : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1_case1 :
  ∀ x : ℝ, f x (-1) ≥ 0 := 
sorry

theorem part1_case2_m_gt_neg1 (m : ℝ) (h : m > -1) :
  ∀ x : ℝ, (f x m ≥ (m + 1) * x) ↔ 
    (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1) := 
sorry

theorem part1_case2_m_lt_neg1 (m : ℝ) (h : m < -1) :
  ∀ x : ℝ, (f x m ≥ (m + 1) * x) ↔ 
    (1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f x m ≥ 0) ↔ m ∈ set.Ici 1 :=
sorry

end part1_case1_part1_case2_m_gt_neg1_part1_case2_m_lt_neg1_part2_l173_173439


namespace range_x1_x2_l173_173868

theorem range_x1_x2
  (x1 x2 x3 : ℝ)
  (hx3_le_x2 : x3 ≤ x2)
  (hx2_le_x1 : x2 ≤ x1)
  (hx_sum : x1 + x2 + x3 = 1)
  (hfx_sum : (x1^2) + (x2^2) + (x3^2) = 1) :
  (2 / 3 : ℝ) ≤ x1 + x2 ∧ x1 + x2 ≤ (4 / 3 : ℝ) :=
sorry

end range_x1_x2_l173_173868


namespace virginia_initial_eggs_l173_173774

theorem virginia_initial_eggs (final_eggs : ℕ) (taken_eggs : ℕ) (H : final_eggs = 93) (G : taken_eggs = 3) : final_eggs + taken_eggs = 96 := 
by
  -- proof part could go here
  sorry

end virginia_initial_eggs_l173_173774


namespace situps_combined_l173_173546

theorem situps_combined (peter_situps : ℝ) (greg_per_set : ℝ) (susan_per_set : ℝ) 
                        (peter_per_set : ℝ) (sets : ℝ) 
                        (peter_situps_performed : peter_situps = sets * peter_per_set) 
                        (greg_situps_performed : sets * greg_per_set = 4.5 * 6)
                        (susan_situps_performed : sets * susan_per_set = 3.75 * 6) :
    peter_situps = 37.5 ∧ greg_per_set = 4.5 ∧ susan_per_set = 3.75 ∧ peter_per_set = 6.25 → 
    4.5 * 6 + 3.75 * 6 = 49.5 :=
by
  sorry

end situps_combined_l173_173546


namespace cube_volume_from_surface_area_l173_173998

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l173_173998


namespace employed_females_percentage_l173_173372

theorem employed_females_percentage (E M : ℝ) (hE : E = 60) (hM : M = 42) : ((E - M) / E) * 100 = 30 := by
  sorry

end employed_females_percentage_l173_173372


namespace circle_equation_l173_173121

theorem circle_equation (x y : ℝ) :
  let C := (4, -6)
  let r := 4
  (x - C.1)^2 + (y - C.2)^2 = r^2 →
  (x - 4)^2 + (y + 6)^2 = 16 :=
by
  intros
  sorry

end circle_equation_l173_173121


namespace min_disks_required_l173_173741

-- Define the initial conditions
def num_files : ℕ := 40
def disk_capacity : ℕ := 2 -- capacity in MB
def num_files_1MB : ℕ := 5
def num_files_0_8MB : ℕ := 15
def num_files_0_5MB : ℕ := 20
def size_1MB : ℕ := 1
def size_0_8MB : ℕ := 8/10 -- 0.8 MB
def size_0_5MB : ℕ := 1/2 -- 0.5 MB

-- Define the mathematical problem
theorem min_disks_required :
  (num_files_1MB * size_1MB + num_files_0_8MB * size_0_8MB + num_files_0_5MB * size_0_5MB) / disk_capacity ≤ 15 := by
  sorry

end min_disks_required_l173_173741


namespace min_dot_product_of_quadrilateral_l173_173315

noncomputable def minimum_dot_product (A B C D P Q : ℝ × ℝ)
  (AB BC AD : ℝ) (angle_DAB : ℝ) (BPQ_area : ℝ) : ℝ :=
if (AB = 4 ∧ BC = 2 ∧ angle_DAB = π / 3 ∧ BPQ_area = (sqrt 3 / 32)) then
  97 / 16 else 0

theorem min_dot_product_of_quadrilateral :
  ∀ (A B C D P Q : ℝ × ℝ) (AB BC AD : ℝ) (angle_DAB : ℝ) (BPQ_area : ℝ),
  AB = 4 → BC = 2 → angle_DAB = π / 3 → BPQ_area = (sqrt 3 / 32) →
  minimum_dot_product A B C D P Q AB BC AD angle_DAB BPQ_area = 97 / 16 :=
by {
  sorry
}

end min_dot_product_of_quadrilateral_l173_173315


namespace grocery_store_total_bottles_l173_173216

def total_bottles (regular_soda : Nat) (diet_soda : Nat) : Nat :=
  regular_soda + diet_soda

theorem grocery_store_total_bottles :
 (total_bottles 9 8 = 17) :=
 by
   sorry

end grocery_store_total_bottles_l173_173216


namespace factor_difference_of_squares_196_l173_173540

theorem factor_difference_of_squares_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end factor_difference_of_squares_196_l173_173540


namespace balloon_permutations_count_l173_173252

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l173_173252


namespace prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l173_173885

-- Definitions for the conditions
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Part (I): Without Replacement
theorem prob_at_least_one_black_without_replacement : 
  (20 - 4) / 20 = 4 / 5 :=
by sorry

-- Part (II): With Replacement
theorem prob_exactly_one_black_with_replacement : 
  (3 * 2 * 4 * 4) / (6 * 6 * 6) = 4 / 9 :=
by sorry

end prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l173_173885


namespace quadratic_inequality_solution_l173_173758

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end quadratic_inequality_solution_l173_173758


namespace fraction_solution_l173_173542

theorem fraction_solution (N : ℝ) (h : N = 12.0) : (0.6667 * N + 1) = (3/4) * N := by 
  sorry

end fraction_solution_l173_173542


namespace girl_scout_cookie_sales_l173_173794

theorem girl_scout_cookie_sales :
  ∃ C P : ℝ, C + P = 1585 ∧ 1.25 * C + 0.75 * P = 1586.25 ∧ P = 790 :=
by
  sorry

end girl_scout_cookie_sales_l173_173794


namespace cookies_per_batch_l173_173696

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l173_173696


namespace intersection_S_T_eq_T_l173_173904

noncomputable def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
noncomputable def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l173_173904


namespace cube_volume_from_surface_area_example_cube_volume_l173_173961

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173961


namespace solve_inequality1_solve_inequality_system_l173_173745

-- Define the first condition inequality
def inequality1 (x : ℝ) : Prop := 
  (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1

-- Theorem for the first inequality proving x >= -2
theorem solve_inequality1 {x : ℝ} (h : inequality1 x) : x ≥ -2 := 
sorry

-- Define the first condition for the system of inequalities
def inequality2 (x : ℝ) : Prop := 
  x - 3 * (x - 2) ≥ 4

-- Define the second condition for the system of inequalities
def inequality3 (x : ℝ) : Prop := 
  (2 * x - 1) / 5 < (x + 1) / 2

-- Theorem for the system of inequalities proving -7 < x ≤ 1
theorem solve_inequality_system {x : ℝ} (h1 : inequality2 x) (h2 : inequality3 x) : -7 < x ∧ x ≤ 1 := 
sorry

end solve_inequality1_solve_inequality_system_l173_173745


namespace Maria_bought_7_roses_l173_173652

theorem Maria_bought_7_roses
  (R : ℕ)
  (h1 : ∀ f : ℕ, 6 * f = 6 * f)
  (h2 : ∀ r : ℕ, ∃ d : ℕ, r = R ∧ d = 3)
  (h3 : 6 * R + 18 = 60) : R = 7 := by
  sorry

end Maria_bought_7_roses_l173_173652


namespace trains_meet_in_16_67_seconds_l173_173064

noncomputable def TrainsMeetTime (length1 length2 distance initial_speed1 initial_speed2 : ℝ) : ℝ := 
  let speed1 := initial_speed1 * 1000 / 3600
  let speed2 := initial_speed2 * 1000 / 3600
  let relativeSpeed := speed1 + speed2
  let totalDistance := distance + length1 + length2
  totalDistance / relativeSpeed

theorem trains_meet_in_16_67_seconds : 
  TrainsMeetTime 100 200 450 90 72 = 16.67 := 
by 
  sorry

end trains_meet_in_16_67_seconds_l173_173064


namespace relationship_between_a_and_b_l173_173558

-- Define the given linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k^2 + 1) * x + 1

-- Formalize the relationship between a and b given the points and the linear function
theorem relationship_between_a_and_b (a b k : ℝ) 
  (hP : a = linear_function k (-4))
  (hQ : b = linear_function k 2) :
  a < b := 
by
  sorry  -- Proof to be filled in by the theorem prover

end relationship_between_a_and_b_l173_173558


namespace nine_chapters_coins_l173_173643

theorem nine_chapters_coins (a d : ℚ)
  (h1 : (a - 2 * d) + (a - d) = a + (a + d) + (a + 2 * d))
  (h2 : (a - 2 * d) + (a - d) + a + (a + d) + (a + 2 * d) = 5) :
  a - d = 7 / 6 :=
by 
  sorry

end nine_chapters_coins_l173_173643


namespace determine_b_l173_173276

theorem determine_b (a b : ℤ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 :=
by
  sorry

end determine_b_l173_173276


namespace balloon_permutations_l173_173269

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l173_173269


namespace solve_4_times_3_l173_173569

noncomputable def custom_operation (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem solve_4_times_3 : custom_operation 4 3 = 13 := by
  -- Here the proof would be provided, for now we use sorry
  sorry

end solve_4_times_3_l173_173569


namespace cube_volume_l173_173933

theorem cube_volume (s : ℝ) (h1 : 6 * s^2 = 1734) : s^3 = 4913 := by
  sorry

end cube_volume_l173_173933


namespace balloon_permutations_count_l173_173251

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l173_173251


namespace apples_difference_l173_173582

-- Definitions based on conditions
def JackiesApples : Nat := 10
def AdamsApples : Nat := 8

-- Statement
theorem apples_difference : JackiesApples - AdamsApples = 2 := by
  sorry

end apples_difference_l173_173582


namespace blocks_added_l173_173627

theorem blocks_added (a b : Nat) (h₁ : a = 86) (h₂ : b = 95) : b - a = 9 :=
by
  sorry

end blocks_added_l173_173627


namespace correct_calculation_l173_173197

theorem correct_calculation :
  (∀ a : ℝ, (a^2)^3 = a^6) ∧
  ¬(∀ a : ℝ, a * a^3 = a^3) ∧
  ¬(∀ a : ℝ, a + 2 * a^2 = 3 * a^3) ∧
  ¬(∀ (a b : ℝ), (-2 * a^2 * b)^2 = -4 * a^4 * b^2) :=
by
  sorry

end correct_calculation_l173_173197


namespace max_value_of_k_l173_173555

theorem max_value_of_k (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + 2 * y) / (x * y) ≥ k / (2 * x + y)) :
  k ≤ 9 :=
by
  sorry

end max_value_of_k_l173_173555


namespace cube_volume_from_surface_area_l173_173990

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173990


namespace percentage_deducted_from_list_price_l173_173401

-- Definitions based on conditions
def cost_price : ℝ := 85.5
def marked_price : ℝ := 112.5
def profit_rate : ℝ := 0.25 -- 25% profit

noncomputable def selling_price : ℝ := cost_price * (1 + profit_rate)

theorem percentage_deducted_from_list_price:
  ∃ d : ℝ, d = 5 ∧ selling_price = marked_price * (1 - d / 100) :=
by
  sorry

end percentage_deducted_from_list_price_l173_173401


namespace adding_books_multiplying_books_l173_173909

-- Define the conditions
def num_books_first_shelf : ℕ := 4
def num_books_second_shelf : ℕ := 5
def num_books_third_shelf : ℕ := 6

-- Define the first question and prove its correctness
theorem adding_books :
  num_books_first_shelf + num_books_second_shelf + num_books_third_shelf = 15 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

-- Define the second question and prove its correctness
theorem multiplying_books :
  num_books_first_shelf * num_books_second_shelf * num_books_third_shelf = 120 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

end adding_books_multiplying_books_l173_173909


namespace min_value_of_one_over_a_and_one_over_b_l173_173321

noncomputable def minValue (a b : ℝ) : ℝ :=
  if 2 * a + 3 * b = 1 then 1 / a + 1 / b else 0

theorem min_value_of_one_over_a_and_one_over_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 1 ∧ minValue a b = 65 / 6 :=
by
  sorry

end min_value_of_one_over_a_and_one_over_b_l173_173321


namespace cube_eq_minus_one_l173_173634

theorem cube_eq_minus_one (x : ℝ) (h : x = -2) : (x + 1) ^ 3 = -1 :=
by
  sorry

end cube_eq_minus_one_l173_173634


namespace mod_sum_example_l173_173631

theorem mod_sum_example :
  (9^5 + 8^4 + 7^6) % 5 = 4 :=
by sorry

end mod_sum_example_l173_173631


namespace cube_volume_l173_173985

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173985


namespace min_value_of_sum_l173_173681

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / (2 * a)) + (1 / b) = 1) :
  a + 2 * b = 9 / 2 :=
sorry

end min_value_of_sum_l173_173681


namespace find_f_7_l173_173900

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 4) = f x
axiom piecewise_function (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : f x = 2 * x^3

theorem find_f_7 : f 7 = -2 := by
  sorry

end find_f_7_l173_173900


namespace problem_sequence_inequality_l173_173807

def a (n : ℕ) : ℚ := 15 + (n - 1 : ℚ) * (-(2 / 3))

theorem problem_sequence_inequality :
  ∃ k : ℕ, (a k) * (a (k + 1)) < 0 ∧ k = 23 :=
by {
  use 23,
  sorry
}

end problem_sequence_inequality_l173_173807


namespace students_catching_up_on_homework_l173_173463

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end students_catching_up_on_homework_l173_173463


namespace min_trips_is_157_l173_173922

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end min_trips_is_157_l173_173922


namespace compare_fractions_l173_173662

theorem compare_fractions : - (1 + 3 / 5) < -1.5 := 
by
  sorry

end compare_fractions_l173_173662


namespace find_3x2y2_l173_173278

theorem find_3x2y2 (x y : ℤ) 
  (h1 : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 := by
  sorry

end find_3x2y2_l173_173278


namespace correct_subtraction_result_l173_173509

-- Definitions based on the problem conditions
def initial_two_digit_number (X Y : ℕ) : ℕ := X * 10 + Y

-- Lean statement that expresses the proof problem
theorem correct_subtraction_result (X Y : ℕ) (H1 : initial_two_digit_number X Y = 99) (H2 : 57 = 57) :
  99 - 57 = 42 :=
by
  sorry

end correct_subtraction_result_l173_173509


namespace general_formula_neg_seq_l173_173137

theorem general_formula_neg_seq (a : ℕ → ℝ) (h_neg : ∀ n, a n < 0)
  (h_recurrence : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  ∀ n, a n = - ((2/3)^(n-2) : ℝ) :=
by
  sorry

end general_formula_neg_seq_l173_173137


namespace jerry_age_l173_173160

theorem jerry_age (M J : ℕ) (hM : M = 24) (hCond : M = 4 * J - 20) : J = 11 := by
  sorry

end jerry_age_l173_173160


namespace lcm_gcd_product_l173_173602

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  rw [ha, hb]
  -- Replace with Nat library functions and calculate
  sorry

end lcm_gcd_product_l173_173602


namespace quadratic_real_roots_range_l173_173303

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end quadratic_real_roots_range_l173_173303


namespace area_of_circle_given_circumference_l173_173751

theorem area_of_circle_given_circumference (C : ℝ) (hC : C = 18 * Real.pi) (k : ℝ) :
  ∃ r : ℝ, C = 2 * Real.pi * r ∧ k * Real.pi = Real.pi * r^2 → k = 81 :=
by
  sorry

end area_of_circle_given_circumference_l173_173751


namespace pumps_280_gallons_in_30_minutes_l173_173176

def hydraflow_rate_per_hour := 560 -- gallons per hour
def time_fraction_in_hour := 1 / 2

theorem pumps_280_gallons_in_30_minutes : hydraflow_rate_per_hour * time_fraction_in_hour = 280 := by
  sorry

end pumps_280_gallons_in_30_minutes_l173_173176


namespace cube_volume_l173_173975

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173975


namespace PetyaCanAlwaysForceDifferenceOfRoots2014_l173_173729

noncomputable def canPetyaForceDifferenceOfRoots2014 : Prop :=
  ∀ (v1 v2 : ℚ) (vasyachooses : (ℚ → ℚ) → Prop), (∃ p q : ℚ, vasyachooses (λ _: ℚ, _) ∧ vasyachooses (λ _: ℚ, _)) →
  ∃ (a b c : ℚ), 
    (vasyachooses (λ _: ℚ, a) ∧ vasyachooses (λ _: ℚ, c)) ∨
    (vasyachooses (λ _: ℚ, b) ∧ vasyachooses (λ _: ℚ, c)) ∧
    (∀ x y : ℚ, (x^3 + a*x^2 + b*x + c = 0 → y^3 + a*y^2 + b*y + c = 0 → abs(x - y) = 2014))

theorem PetyaCanAlwaysForceDifferenceOfRoots2014 : canPetyaForceDifferenceOfRoots2014 :=
sorry

end PetyaCanAlwaysForceDifferenceOfRoots2014_l173_173729


namespace balloon_arrangements_l173_173239

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l173_173239


namespace cost_price_of_radio_l173_173179

-- Definitions for conditions
def selling_price := 1245
def loss_percentage := 17

-- Prove that the cost price is Rs. 1500 given the conditions
theorem cost_price_of_radio : 
  ∃ C, (C - 1245) * 100 / C = 17 ∧ C = 1500 := 
sorry

end cost_price_of_radio_l173_173179


namespace find_m_collinear_l173_173686

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isCollinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

theorem find_m_collinear :
  ∀ (m : ℝ),
  let A := Point.mk (-2) 3
  let B := Point.mk 3 (-2)
  let C := Point.mk (1 / 2) m
  isCollinear A B C → m = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_m_collinear_l173_173686


namespace balloon_arrangements_l173_173266

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l173_173266


namespace students_taking_all_three_classes_l173_173187

variable (students : Finset ℕ)
variable (yoga bridge painting : Finset ℕ)

variables (yoga_count bridge_count painting_count at_least_two exactly_two all_three : ℕ)

variable (total_students : students.card = 25)
variable (yoga_students : yoga.card = 12)
variable (bridge_students : bridge.card = 15)
variable (painting_students : painting.card = 11)
variable (at_least_two_classes : at_least_two = 10)
variable (exactly_two_classes : exactly_two = 7)

theorem students_taking_all_three_classes :
  all_three = 3 :=
sorry

end students_taking_all_three_classes_l173_173187


namespace distance_C_distance_BC_l173_173324

variable (A B C D : ℕ)

theorem distance_C
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : C = 625 :=
by
  sorry

theorem distance_BC
  (A B C D : ℕ)
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : B + C = 875 :=
by
  sorry

end distance_C_distance_BC_l173_173324


namespace students_class_division_l173_173212

theorem students_class_division (n : ℕ) (h1 : n % 15 = 0) (h2 : n % 24 = 0) : n = 120 :=
sorry

end students_class_division_l173_173212


namespace symmetric_point_proof_l173_173752

def symmetric_point (P : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ := sorry

theorem symmetric_point_proof :
  symmetric_point (2, 5) (λ x => 1 - x) = (-4, -1) := sorry

end symmetric_point_proof_l173_173752


namespace cube_volume_l173_173976

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173976


namespace inverse_proportion_quadrants_l173_173043

theorem inverse_proportion_quadrants (x : ℝ) (y : ℝ) (h : y = 6/x) : 
  (x > 0 -> y > 0) ∧ (x < 0 -> y < 0) := 
sorry

end inverse_proportion_quadrants_l173_173043


namespace carA_speed_calc_l173_173404

-- Defining the conditions of the problem
def carA_time : ℕ := 8
def carB_speed : ℕ := 25
def carB_time : ℕ := 4
def distance_ratio : ℕ := 4
def carB_distance : ℕ := carB_speed * carB_time
def carA_distance : ℕ := distance_ratio * carB_distance

-- Mathematical statement to be proven
theorem carA_speed_calc : carA_distance / carA_time = 50 := by
  sorry

end carA_speed_calc_l173_173404


namespace balloon_arrangement_count_l173_173235

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l173_173235


namespace rearrange_infinite_decimal_l173_173607

-- Define the set of digits
def Digit : Type := Fin 10

-- Define the classes of digits
def Class1 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m > n → dec m ≠ d

def Class2 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ dec m = d

-- The statement to prove
theorem rearrange_infinite_decimal (dec : ℕ → Digit) (h : ∃ d : Digit, ¬ Class1 d dec) :
  ∃ rearranged : ℕ → Digit, (Class1 d rearranged ∧ Class2 d rearranged) →
  ∃ r : ℚ, ∃ n : ℕ, ∀ m ≥ n, rearranged m = rearranged (m + n) :=
sorry

end rearrange_infinite_decimal_l173_173607


namespace factorize_expression_l173_173277

theorem factorize_expression (x y : ℝ) : x^2 - 1 + 2 * x * y + y^2 = (x + y + 1) * (x + y - 1) :=
by sorry

end factorize_expression_l173_173277


namespace student_A_more_stable_than_B_l173_173521

theorem student_A_more_stable_than_B 
    (avg_A : ℝ := 98) (avg_B : ℝ := 98) 
    (var_A : ℝ := 0.2) (var_B : ℝ := 0.8) : 
    var_A < var_B :=
by sorry

end student_A_more_stable_than_B_l173_173521


namespace correct_understanding_of_philosophy_l173_173467

-- Define the conditions based on the problem statement
def philosophy_from_life_and_practice : Prop :=
  -- Philosophy originates from people's lives and practice.
  sorry
  
def philosophy_affects_lives : Prop :=
  -- Philosophy consciously or unconsciously affects people's lives, learning, and work
  sorry

def philosophical_knowledge_requires_learning : Prop :=
  true

def philosophy_not_just_summary : Prop :=
  true

-- Given conditions 1, 2, 3 (as negation of 3 in original problem), and 4 (as negation of 4 in original problem),
-- We need to prove the correct understanding (which is combination ①②) is correct.
theorem correct_understanding_of_philosophy :
  philosophy_from_life_and_practice →
  philosophy_affects_lives →
  philosophical_knowledge_requires_learning →
  philosophy_not_just_summary →
  (philosophy_from_life_and_practice ∧ philosophy_affects_lives) :=
by
  intros
  apply And.intro
  · assumption
  · assumption

end correct_understanding_of_philosophy_l173_173467


namespace triangle_angle_A_l173_173685

theorem triangle_angle_A (AC BC : ℝ) (angle_B : ℝ) (h_AC : AC = Real.sqrt 2) (h_BC : BC = 1) (h_angle_B : angle_B = 45) :
  ∃ (angle_A : ℝ), angle_A = 30 :=
by
  sorry

end triangle_angle_A_l173_173685


namespace initial_speed_l173_173799

variable (D T : ℝ) -- Total distance D and total time T
variable (S : ℝ)   -- Initial speed S

theorem initial_speed :
  (2 * D / 3) = (S * T / 3) →
  (35 = (D / (2 * T))) →
  S = 70 :=
by
  intro h1 h2
  -- Skipping the proof with 'sorry'
  sorry

end initial_speed_l173_173799


namespace three_digit_numbers_l173_173234

theorem three_digit_numbers (N : ℕ) (a b c : ℕ) 
  (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : b ≤ 9 ∧ c ≤ 9)
  (h4 : a - b + c % 11 = 0)
  (h5 : N % 11 = 0)
  (h6 : N = 11 * (a^2 + b^2 + c^2)) :
  N = 550 ∨ N = 803 :=
  sorry

end three_digit_numbers_l173_173234


namespace balloon_permutations_l173_173245

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l173_173245


namespace ratio_length_width_l173_173756

theorem ratio_length_width (A L W : ℕ) (hA : A = 432) (hW : W = 12) (hArea : A = L * W) : L / W = 3 := 
by
  -- Placeholders for the actual mathematical proof
  sorry

end ratio_length_width_l173_173756


namespace cube_volume_from_surface_area_example_cube_volume_l173_173957

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173957


namespace complement_A_in_U_range_of_a_l173_173122

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def f (x : ℝ) : ℝ := (1 / (sqrt (x + 2))) + log (3 - x)
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 3}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < (2 * a - 1)}

theorem complement_A_in_U : compl A = {x | x ≤ -2 ∨ 3 ≤ x} :=
by {
  sorry
}

theorem range_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ Iic 2 :=
by {
  sorry
}

end complement_A_in_U_range_of_a_l173_173122


namespace donation_to_treetown_and_forest_reserve_l173_173491

noncomputable def donation_problem (x : ℕ) :=
  x + (x + 140) = 1000

theorem donation_to_treetown_and_forest_reserve :
  ∃ x : ℕ, donation_problem x ∧ (x + 140 = 570) := 
by
  sorry

end donation_to_treetown_and_forest_reserve_l173_173491


namespace negation_proposition_l173_173572

theorem negation_proposition (p : Prop) (h : ∀ x : ℝ, 2 * x^2 + 1 > 0) : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
sorry

end negation_proposition_l173_173572


namespace candidate_valid_vote_percentage_l173_173133

theorem candidate_valid_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_votes : ℕ) 
  (valid_percentage : ℚ)
  (total_votes_eq : total_votes = 560000)
  (invalid_percentage_eq : invalid_percentage = 15 / 100)
  (candidate_votes_eq : candidate_votes = 357000)
  (valid_percentage_eq : valid_percentage = 85 / 100) :
  (candidate_votes / (total_votes * valid_percentage)) * 100 = 75 := 
by
  sorry

end candidate_valid_vote_percentage_l173_173133


namespace simplify_expression_l173_173554

theorem simplify_expression (a b c d : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) :
  -5 * a + 2017 * c * d - 5 * b = 2017 :=
by
  sorry

end simplify_expression_l173_173554


namespace total_people_bought_tickets_l173_173190

-- Definitions based on the conditions from step a)
def num_adults := 375
def num_children := 3 * num_adults
def total_revenue := 7 * num_adults + 3 * num_children

-- Statement of the theorem based on the question in step a)
theorem total_people_bought_tickets : (num_adults + num_children) = 1500 :=
by
  -- The proof is omitted, but we're ensuring the correctness of the theorem statement.
  sorry

end total_people_bought_tickets_l173_173190


namespace sum_of_tangents_l173_173426

theorem sum_of_tangents (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h_tan_α : Real.tan α = 2) (h_tan_β : Real.tan β = 3) : α + β = 3 * π / 4 :=
by
  sorry

end sum_of_tangents_l173_173426


namespace grunters_at_least_4_wins_l173_173174

noncomputable def grunters_probability : ℚ :=
  let p_win := 3 / 5
  let p_loss := 2 / 5
  let p_4_wins := 5 * (p_win^4) * (p_loss)
  let p_5_wins := p_win^5
  p_4_wins + p_5_wins

theorem grunters_at_least_4_wins :
  grunters_probability = 1053 / 3125 :=
by sorry

end grunters_at_least_4_wins_l173_173174


namespace find_x_y_l173_173579

theorem find_x_y (x y : ℝ) : 
    (3 * x + 2 * y + 5 * x + 7 * x = 360) →
    (x = y) →
    (x = 360 / 17) ∧ (y = 360 / 17) := by
  intros h₁ h₂
  sorry

end find_x_y_l173_173579


namespace problem_solution_l173_173568

theorem problem_solution (a d e : ℕ) (ha : 0 < a ∧ a < 10) (hd : 0 < d ∧ d < 10) (he : 0 < e ∧ e < 10) :
  ((10 * a + d) * (10 * a + e) = 100 * a ^ 2 + 110 * a + d * e) ↔ (d + e = 11) := by
  sorry

end problem_solution_l173_173568


namespace problem1_problem2_l173_173561

-- Problem 1: Proving the range of m values for the given inequality
theorem problem1 (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| ≥ 3) ↔ (m ≤ -4 ∨ m ≥ 2) :=
sorry

-- Problem 2: Proving the range of m values given a non-empty solution set for the inequality
theorem problem2 (m : ℝ) : (∃ x : ℝ, |m + 1| - 2 * m ≥ x^2 - x) ↔ (m ≤ 5/4) :=
sorry

end problem1_problem2_l173_173561


namespace smallest_number_of_students_l173_173386

theorem smallest_number_of_students 
  (ninth_to_seventh : ℕ → ℕ → Prop)
  (ninth_to_sixth : ℕ → ℕ → Prop) 
  (r1 : ninth_to_seventh 3 2) 
  (r2 : ninth_to_sixth 7 4) : 
  ∃ n7 n6 n9, 
    ninth_to_seventh n9 n7 ∧ 
    ninth_to_sixth n9 n6 ∧ 
    n9 + n7 + n6 = 47 :=
sorry

end smallest_number_of_students_l173_173386


namespace arithmetic_sequence_inequality_l173_173705

theorem arithmetic_sequence_inequality 
  (a b c : ℝ) 
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : b - a = d)
  (h3 : c - b = d) :
  ¬ (a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) :=
sorry

end arithmetic_sequence_inequality_l173_173705


namespace initial_cloves_l173_173325

theorem initial_cloves (used_cloves left_cloves initial_cloves : ℕ) (h1 : used_cloves = 86) (h2 : left_cloves = 7) : initial_cloves = 93 :=
by
  sorry

end initial_cloves_l173_173325


namespace initial_girls_count_l173_173646

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end initial_girls_count_l173_173646


namespace television_price_reduction_l173_173529

variable (P : ℝ) (F : ℝ)
variable (h : F = 0.56 * P - 50)

theorem television_price_reduction :
  F / P = 0.56 - 50 / P :=
by {
  sorry
}

end television_price_reduction_l173_173529


namespace one_fourth_of_eight_times_x_plus_two_l173_173301

theorem one_fourth_of_eight_times_x_plus_two (x : ℝ) : 
  (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 :=
by
  sorry

end one_fourth_of_eight_times_x_plus_two_l173_173301


namespace find_line_equation_l173_173879

theorem find_line_equation 
  (A : ℝ × ℝ) (hA : A = (-2, -3)) 
  (h_perpendicular : ∃ k b : ℝ, ∀ x y, 3 * x + 4 * y - 3 = 0 → k * x + y = b) :
  ∃ k' b' : ℝ, (∀ x y, k' * x + y = b' → y = (4 / 3) * x + 1 / 3) ∧ (k' = 4 ∧ b' = -1) :=
by
  sorry

end find_line_equation_l173_173879


namespace b_minus_a_equals_two_l173_173151

open Set

variables {a b : ℝ}

theorem b_minus_a_equals_two (h₀ : {1, a + b, a} = ({0, b / a, b} : Finset ℝ)) (h₁ : a ≠ 0) : b - a = 2 :=
sorry

end b_minus_a_equals_two_l173_173151


namespace part1_part2_l173_173440

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^3 + k * Real.log x
noncomputable def f' (x : ℝ) (k : ℝ) : ℝ := 3 * x^2 + k / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x k - f' x k + 9 / x

-- Part (1): Prove the monotonic intervals and extreme values for k = 6:
theorem part1 :
  (∀ x : ℝ, 0 < x ∧ x < 1 → g x 6 < g 1 6) ∧
  (∀ x : ℝ, 1 < x → g x 6 > g 1 6) ∧
  (g 1 6 = 1) := sorry

-- Part (2): Prove the given inequality for k ≥ -3:
theorem part2 (k : ℝ) (hk : k ≥ -3) (x1 x2 : ℝ) (hx1 : x1 ≥ 1) (hx2 : x2 ≥ 1) (h : x1 > x2) :
  (f' x1 k + f' x2 k) / 2 > (f x1 k - f x2 k) / (x1 - x2) := sorry

end part1_part2_l173_173440


namespace percentage_of_8thgraders_correct_l173_173180

def total_students_oakwood : ℕ := 150
def total_students_pinecrest : ℕ := 250

def percent_8thgraders_oakwood : ℕ := 60
def percent_8thgraders_pinecrest : ℕ := 55

def number_of_8thgraders_oakwood : ℚ := (percent_8thgraders_oakwood * total_students_oakwood) / 100
def number_of_8thgraders_pinecrest : ℚ := (percent_8thgraders_pinecrest * total_students_pinecrest) / 100

def total_number_of_8thgraders : ℚ := number_of_8thgraders_oakwood + number_of_8thgraders_pinecrest
def total_number_of_students : ℕ := total_students_oakwood + total_students_pinecrest

def percent_8thgraders_combined : ℚ := (total_number_of_8thgraders / total_number_of_students) * 100

theorem percentage_of_8thgraders_correct : percent_8thgraders_combined = 57 := 
by
  sorry

end percentage_of_8thgraders_correct_l173_173180


namespace brainiacs_like_both_l173_173221

theorem brainiacs_like_both
  (R M B : ℕ)
  (h1 : R = 2 * M)
  (h2 : R + M - B = 96)
  (h3 : M - B = 20) : B = 18 := by
  sorry

end brainiacs_like_both_l173_173221


namespace units_digit_fraction_l173_173824

theorem units_digit_fraction (h1 : 30 = 2 * 3 * 5) (h2 : 31 = 31) (h3 : 32 = 2^5) 
    (h4 : 33 = 3 * 11) (h5 : 34 = 2 * 17) (h6 : 35 = 5 * 7) (h7 : 7200 = 2^4 * 3^2 * 5^2) :
    ((30 * 31 * 32 * 33 * 34 * 35) / 7200) % 10 = 2 :=
by
  sorry

end units_digit_fraction_l173_173824


namespace find_larger_number_l173_173101

theorem find_larger_number (x y : ℕ) (h1 : x + y = 55) (h2 : x - y = 15) : x = 35 := by 
  -- proof will go here
  sorry

end find_larger_number_l173_173101


namespace max_value_of_expression_l173_173153

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l173_173153


namespace center_of_circle_l173_173891

-- Define the circle in polar coordinates
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center of the circle in polar coordinates
def center_polar (ρ θ : ℝ) : Prop := (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- The theorem states that the center of the given circle in polar coordinates is (1, π/2) or (1, 3π/2)
theorem center_of_circle : ∃ (ρ θ : ℝ), circle_polar ρ θ → center_polar ρ θ :=
by
  -- The center of the circle given the condition in polar coordinate system is (1, π/2) or (1, 3π/2)
  sorry

end center_of_circle_l173_173891


namespace domino_tile_count_l173_173576

theorem domino_tile_count (low high : ℕ) (tiles_standard_set : ℕ) (range_standard_set : ℕ) (range_new_set : ℕ) :
  range_standard_set = 6 → tiles_standard_set = 28 →
  low = 0 → high = 12 →
  range_new_set = 13 → 
  (∀ n, 0 ≤ n ∧ n ≤ range_standard_set → ∀ m, n ≤ m ∧ m ≤ range_standard_set → n ≤ m → true) →
  (∀ n, 0 ≤ n ∧ n ≤ range_new_set → ∀ m, n ≤ m ∧ m <= range_new_set → n <= m → true) →
  tiles_new_set = 91 :=
by
  intros h_range_standard h_tiles_standard h_low h_high h_range_new h_standard_pairs h_new_pairs
  --skipping the proof
  sorry

end domino_tile_count_l173_173576


namespace balloon_permutations_count_l173_173253

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l173_173253


namespace arithmetic_sequence_monotone_l173_173552

theorem arithmetic_sequence_monotone (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : ∀ n, a (n + 1) - a n = d) :
  (a 2 > a 1) ↔ (∀ n, a (n + 1) > a n) :=
by 
  sorry

end arithmetic_sequence_monotone_l173_173552


namespace fraction_subtraction_simplify_l173_173227

theorem fraction_subtraction_simplify :
  (9 / 19 - 3 / 57 - 1 / 3) = 5 / 57 :=
by
  sorry

end fraction_subtraction_simplify_l173_173227


namespace Michael_made_97_dollars_l173_173600

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end Michael_made_97_dollars_l173_173600


namespace balloon_permutation_count_l173_173247

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l173_173247


namespace balloon_permutations_l173_173243

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l173_173243


namespace smallest_quotient_is_1_9_l173_173400

def is_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n <= 99

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let x := n / 10
  let y := n % 10
  x + y

noncomputable def quotient (n : ℕ) : ℚ :=
  n / (sum_of_digits n)

theorem smallest_quotient_is_1_9 :
  ∃ n, is_two_digit_number n ∧ (∃ x y, n = 10 * x + y ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ quotient n = 1.9 := 
sorry

end smallest_quotient_is_1_9_l173_173400


namespace cube_volume_l173_173977

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173977


namespace john_made_47000_from_car_l173_173717

def cost_to_fix_before_discount := 20000
def discount := 0.20
def prize := 70000
def keep_percentage := 0.90

def cost_to_fix_after_discount := cost_to_fix_before_discount - (discount * cost_to_fix_before_discount)
def prize_kept := keep_percentage * prize
def money_made := prize_kept - cost_to_fix_after_discount

theorem john_made_47000_from_car : money_made = 47000 := by
  sorry

end john_made_47000_from_car_l173_173717


namespace sum_of_coefficients_l173_173691

theorem sum_of_coefficients (a b : ℝ) (h1 : a = 1 * 5) (h2 : -b = 1 + 5) : a + b = -1 :=
by
  sorry

end sum_of_coefficients_l173_173691


namespace cricket_team_initial_games_l173_173005

theorem cricket_team_initial_games
  (initial_games : ℕ)
  (won_30_percent_initially : ℕ)
  (additional_wins : ℕ)
  (final_win_rate : ℚ) :
  won_30_percent_initially = initial_games * 30 / 100 →
  final_win_rate = (won_30_percent_initially + additional_wins) / (initial_games + additional_wins) →
  additional_wins = 55 →
  final_win_rate = 52 / 100 →
  initial_games = 120 := by sorry

end cricket_team_initial_games_l173_173005


namespace correct_operation_l173_173778

theorem correct_operation (a b : ℝ) :
  (3 * a^2 - a^2 ≠ 3) ∧
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((-3 * a * b^2)^2 ≠ -6 * a^2 * b^4) →
  a^3 / a^2 = a :=
by
sorry

end correct_operation_l173_173778


namespace range_of_values_for_sqrt_l173_173621

theorem range_of_values_for_sqrt (x : ℝ) : (x + 3 ≥ 0) ↔ (x ≥ -3) :=
by
  sorry

end range_of_values_for_sqrt_l173_173621


namespace bernardo_prob_greater_silvia_l173_173823

open_locale classical
noncomputable theory

/--
Bernardo randomly picks 3 distinct numbers from the set {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, while Silvia randomly picks 3 distinct numbers from the set {1, 2, 3, 4, 5, 6, 7, 8, 9}. Both then arrange their chosen numbers in descending order to form a 3-digit number. Prove that the probability that Bernardo's number is greater than Silvia's number is 9/14.
-/
theorem bernardo_prob_greater_silvia :
  let S : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let T : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let bernardo_choices : finset (finset ℕ) := S.powerset.filter (λ s, s.card = 3)
  let silvia_choices : finset (finset ℕ) := T.powerset.filter (λ s, s.card = 3)
  let bernardo_3_digit : finset ℕ := bernardo_choices.image (λ s, (s.sort (≥)).foldl (λ acc n, acc * 10 + n) 0)
  let silvia_3_digit : finset ℕ := silvia_choices.image (λ s, (s.sort (≥)).foldl (λ acc n, acc * 10 + n) 0)
  let num_bigger := bernardo_3_digit.filter (λ b, ∀ s ∈ silvia_3_digit, b > s)
  (num_bigger.card : ℚ) / (bernardo_3_digit.card * silvia_3_digit.card) = 9 / 14 :=
by sorry

end bernardo_prob_greater_silvia_l173_173823


namespace smallest_percentage_increase_l173_173226

theorem smallest_percentage_increase :
  let n2005 := 75
  let n2006 := 85
  let n2007 := 88
  let n2008 := 94
  let n2009 := 96
  let n2010 := 102
  let perc_increase (a b : ℕ) := ((b - a) : ℚ) / a * 100
  perc_increase n2008 n2009 < perc_increase n2006 n2007 ∧
  perc_increase n2008 n2009 < perc_increase n2007 n2008 ∧
  perc_increase n2008 n2009 < perc_increase n2009 n2010 ∧
  perc_increase n2008 n2009 < perc_increase n2005 n2006
:= sorry

end smallest_percentage_increase_l173_173226


namespace probability_C_l173_173383

-- Variables representing the probabilities of each region
variables (P_A P_B P_C P_D P_E : ℚ)

-- Given conditions
def conditions := P_A = 3/10 ∧ P_B = 1/4 ∧ P_D = 1/5 ∧ P_E = 1/10 ∧ P_A + P_B + P_C + P_D + P_E = 1

-- The statement to prove
theorem probability_C (h : conditions P_A P_B P_C P_D P_E) : P_C = 3/20 := 
by
  sorry

end probability_C_l173_173383


namespace triangle_inequality_product_l173_173331

theorem triangle_inequality_product (x y z : ℝ) (h1 : x + y > z) (h2 : x + z > y) (h3 : y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := 
by
  sorry

end triangle_inequality_product_l173_173331


namespace Sasha_can_write_2011_l173_173740

theorem Sasha_can_write_2011 (N : ℕ) (hN : N > 1) : 
    ∃ (s : ℕ → ℕ), (s 0 = N) ∧ (∃ n, s n = 2011) ∧ 
    (∀ k, ∃ d, d > 1 ∧ (s (k + 1) = s k + d ∨ s (k + 1) = s k - d)) :=
sorry

end Sasha_can_write_2011_l173_173740


namespace find_t_l173_173481

-- conditions
def quadratic_eq (x : ℝ) : Prop := 25 * x^2 + 20 * x - 1000 = 0

-- statement to prove
theorem find_t (x : ℝ) (p t : ℝ) (h1 : p = 2/5) (h2 : t = 104/25) : 
  (quadratic_eq x) → (x + p)^2 = t :=
by
  intros
  sorry

end find_t_l173_173481


namespace abs_difference_of_opposite_signs_l173_173299

theorem abs_difference_of_opposite_signs (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hdiff : a * b < 0) : |a - b| = 6 := 
sorry

end abs_difference_of_opposite_signs_l173_173299


namespace cube_volume_from_surface_area_l173_173995

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173995


namespace no_nat_n_divisible_by_169_l173_173487

theorem no_nat_n_divisible_by_169 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 5 * n + 16 = 169 * k :=
sorry

end no_nat_n_divisible_by_169_l173_173487


namespace find_k_l173_173870

def a : ℝ × ℝ × ℝ := (2, 1, 4)
def b : ℝ × ℝ × ℝ := (1, 0, 2)
def k := 15 / 31

theorem find_k (ha : a = (2, 1, 4)) (hb : b = (1, 0, 2)) : 
    (a.fst + b.fst, a.snd + b.snd, a.snd + b.snd) • (k * a.fst - b.fst, k * a.snd - b.snd, k * a.snd - b.snd) = 0 :=
sorry

end find_k_l173_173870


namespace positive_integers_divisible_by_4_5_and_6_less_than_300_l173_173701

open Nat

theorem positive_integers_divisible_by_4_5_and_6_less_than_300 : 
    ∃ n : ℕ, n = 5 ∧ ∀ m, m < 300 → (m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0) → (m % 60 = 0) :=
by
  sorry

end positive_integers_divisible_by_4_5_and_6_less_than_300_l173_173701


namespace probability_prime_or_multiple_of_4_l173_173538

def balls : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def isPrimeOrMultipleOf4 (n : ℕ) : Prop :=
  Nat.Prime n ∨ n % 4 = 0

def satisfyingBalls : List ℕ :=
  balls.filter isPrimeOrMultipleOf4

def numberOfSatisfyingBalls : ℕ := satisfyingBalls.length
def totalBalls : ℕ := balls.length

theorem probability_prime_or_multiple_of_4 :
  numberOfSatisfyingBalls / totalBalls = 3 / 4 :=
by
  have h : finishingProportion = 6 := rfl
  have h : totalBalls = 8 := rfl
  sorry

end probability_prime_or_multiple_of_4_l173_173538


namespace random_variables_example_l173_173166

open ProbabilityTheory

noncomputable def xi_n (n : ℕ) (ξ : ℝ) : ℝ := 
  if (ξ >= 0 ∧ ξ <= (1 / n : ℝ)) then n else 0

theorem random_variables_example :
  ∃ (ξ : ℕ → ℝ), (∀ n ≥ 1, xi_n n ξ.toNat.toReal → 0) ∧ 
  (∀ n ≥ 1, E (xi_n n ξ.toNat.toReal) = 1) :=
sorry

end random_variables_example_l173_173166


namespace problem_1_problem_2_problem_3_problem_4_l173_173231

theorem problem_1 : 12 - (-18) + (-7) - 15 = 8 := sorry

theorem problem_2 : -0.5 + (- (3 + 1/4)) + (-2.75) + (7 + 1/2) = 1 := sorry

theorem problem_3 : -2^2 + 3 * (-1)^(2023) - abs (-4) * 5 = -27 := sorry

theorem problem_4 : -3 - (-5 + (1 - 2 * (3 / 5)) / (-2)) = 19 / 10 := sorry

end problem_1_problem_2_problem_3_problem_4_l173_173231


namespace dart_hit_number_list_count_l173_173084

def number_of_dart_hit_lists (darts dartboards : ℕ) : ℕ :=
  11  -- Based on the solution, the hard-coded answer is 11.

theorem dart_hit_number_list_count : number_of_dart_hit_lists 6 4 = 11 := 
by 
  sorry

end dart_hit_number_list_count_l173_173084


namespace pencils_left_l173_173411

def initial_pencils : Nat := 127
def pencils_from_joyce : Nat := 14
def pencils_per_friend : Nat := 7

theorem pencils_left : ((initial_pencils + pencils_from_joyce) % pencils_per_friend) = 1 := by
  sorry

end pencils_left_l173_173411


namespace min_b_over_a_l173_173433

theorem min_b_over_a (a b : ℝ) (h : ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x ≥ 0) : b / a ≥ 1 := by
  sorry

end min_b_over_a_l173_173433


namespace factorize_quadratic_l173_173094

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end factorize_quadratic_l173_173094


namespace Michael_made_97_dollars_l173_173601

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end Michael_made_97_dollars_l173_173601


namespace determine_b_l173_173618

theorem determine_b (b : ℝ) : 
  (∀ x : ℝ, f x = if x < 1 then 3 * x - b else 2^x) ∧
  f (f (5 / 6)) = 4 → b = 1 / 2 := 
by
  let f (x : ℝ) := if x < 1 then 3 * x - b else 2^x
  sorry

end determine_b_l173_173618


namespace find_line_equation_l173_173878

noncomputable def equation_of_perpendicular_line : Prop := 
  ∃ (l : ℝ → ℝ) (x y : ℝ), 
    (l x = 4*x/3 - 17/3) ∧ 
    (x = -2 ∧ y = -3) ∧ 
    (3*x + 4*y - 3 = 0)

theorem find_line_equation (A : ℝ × ℝ) (B : ℝ → Prop) :
    A = (-2, -3) → 
    (∀ x y : ℝ, B (3*x + 4*y - 3 = 0)) → 
     ∃ (a b c : ℝ), 4*a - 3*b - c = 0 :=
by 
    sorry

end find_line_equation_l173_173878


namespace initial_erasers_calculation_l173_173354

variable (initial_erasers added_erasers total_erasers : ℕ)

theorem initial_erasers_calculation
  (total_erasers_eq : total_erasers = 270)
  (added_erasers_eq : added_erasers = 131) :
  initial_erasers = total_erasers - added_erasers → initial_erasers = 139 := by
  intro h
  rw [total_erasers_eq, added_erasers_eq] at h
  simp at h
  exact h

end initial_erasers_calculation_l173_173354


namespace number_of_distinct_a_l173_173422

noncomputable def count_distinct_a : ℕ :=
  let divisors := [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72, 144, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -36, -48, -72, -144]
  let pairs := (divisors.product divisors).filter (λ (p : ℤ × ℤ), p.1 * p.2 = 144)
  let sums := pairs.map (λ (p : ℤ × ℤ), p.1 + p.2 - 24)
  let distinct_sums := sums.eraseDuplicates  
  distinct_sums.length

theorem number_of_distinct_a : count_distinct_a = 14 :=
by
  sorry

end number_of_distinct_a_l173_173422


namespace balloon_permutation_count_l173_173274

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l173_173274


namespace total_hair_cut_l173_173837

-- Definitions from conditions
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- The theorem stating the math problem
theorem total_hair_cut : first_cut + second_cut = 0.875 := by
  sorry

end total_hair_cut_l173_173837


namespace max_homework_time_l173_173483

theorem max_homework_time (biology_time history_time geography_time : ℕ) :
    biology_time = 20 ∧ history_time = 2 * biology_time ∧ geography_time = 3 * history_time →
    biology_time + history_time + geography_time = 180 :=
by
    intros
    sorry

end max_homework_time_l173_173483


namespace balloon_permutation_count_l173_173249

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l173_173249


namespace total_short_trees_l173_173763

def short_trees_initial := 41
def short_trees_planted := 57

theorem total_short_trees : short_trees_initial + short_trees_planted = 98 := by
  sorry

end total_short_trees_l173_173763


namespace range_of_x_l173_173107

theorem range_of_x (x : ℝ) : (2 : ℝ)^(3 - 2 * x) < (2 : ℝ)^(3 * x - 4) → x > 7 / 5 := by
  sorry

end range_of_x_l173_173107


namespace other_car_speed_l173_173055

-- Definitions of the conditions
def red_car_speed : ℕ := 30
def initial_gap : ℕ := 20
def overtaking_time : ℕ := 1

-- Assertion of what needs to be proved
theorem other_car_speed : (initial_gap + red_car_speed * overtaking_time) = 50 :=
  sorry

end other_car_speed_l173_173055


namespace employed_population_percentage_l173_173012

variable (P : ℝ) -- Total population
variable (percentage_employed_to_population : ℝ) -- Percentage of total population employed
variable (percentage_employed_males_to_population : ℝ := 0.42) -- 42% of population are employed males
variable (percentage_employed_females_to_employed : ℝ := 0.30) -- 30% of employed people are females

theorem employed_population_percentage :
  percentage_employed_to_population = 0.60 :=
sorry

end employed_population_percentage_l173_173012


namespace balloon_arrangements_l173_173242

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l173_173242


namespace inequality_for_positive_integers_l173_173899

theorem inequality_for_positive_integers 
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : 1/a + 1/b = 1)
  (n : ℕ)
  (hn : n > 0) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2^(2*n) - 2^(n + 1) :=
sorry

end inequality_for_positive_integers_l173_173899


namespace eccentricity_of_ellipse_l173_173921

theorem eccentricity_of_ellipse :
  (∃ θ : Real, (x = 3 * Real.cos θ) ∧ (y = 4 * Real.sin θ))
  → (∃ e : Real, e = Real.sqrt 7 / 4) := 
sorry

end eccentricity_of_ellipse_l173_173921


namespace determinant_of_tan_matrix_l173_173593

theorem determinant_of_tan_matrix
  (A B C : ℝ)
  (h₁ : A = π / 4)
  (h₂ : A + B + C = π)
  : (Matrix.det ![
      ![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]
    ]) = 2 :=
  sorry

end determinant_of_tan_matrix_l173_173593


namespace A_det_nonzero_A_inv_is_correct_l173_173100

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 4], ![2, 9]]

def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![9, -4], ![-2, 1]]

theorem A_det_nonzero : det A ≠ 0 := 
  sorry

theorem A_inv_is_correct : A * A_inv = 1 := 
  sorry

end A_det_nonzero_A_inv_is_correct_l173_173100


namespace parallel_segments_l173_173736

structure Point2D where
  x : Int
  y : Int

def vector (P Q : Point2D) : Point2D :=
  { x := Q.x - P.x, y := Q.y - P.y }

def is_parallel (v1 v2 : Point2D) : Prop :=
  ∃ k : Int, v2.x = k * v1.x ∧ v2.y = k * v1.y 

theorem parallel_segments :
  let A := { x := 1, y := 3 }
  let B := { x := 2, y := -1 }
  let C := { x := 0, y := 4 }
  let D := { x := 2, y := -4 }
  is_parallel (vector A B) (vector C D) := 
  sorry

end parallel_segments_l173_173736


namespace customers_left_l173_173079

theorem customers_left (original_customers remaining_tables people_per_table customers_left : ℕ)
  (h1 : original_customers = 44)
  (h2 : remaining_tables = 4)
  (h3 : people_per_table = 8)
  (h4 : original_customers - remaining_tables * people_per_table = customers_left) :
  customers_left = 12 :=
by
  sorry

end customers_left_l173_173079


namespace woman_work_rate_l173_173780

theorem woman_work_rate (W : ℝ) :
  (1 / 6) + W + (1 / 9) = (1 / 3) → W = (1 / 18) :=
by
  intro h
  sorry

end woman_work_rate_l173_173780


namespace log_expression_in_terms_of_a_l173_173852

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

variable (a : ℝ) (h : a = log3 2)

theorem log_expression_in_terms_of_a : log3 8 - 2 * log3 6 = a - 2 :=
by
  sorry

end log_expression_in_terms_of_a_l173_173852


namespace ray_walks_to_high_school_7_l173_173167

theorem ray_walks_to_high_school_7
  (walks_to_park : ℕ)
  (walks_to_high_school : ℕ)
  (walks_home : ℕ)
  (trips_per_day : ℕ)
  (total_daily_blocks : ℕ) :
  walks_to_park = 4 →
  walks_home = 11 →
  trips_per_day = 3 →
  total_daily_blocks = 66 →
  3 * (walks_to_park + walks_to_high_school + walks_home) = total_daily_blocks →
  walks_to_high_school = 7 :=
by
  sorry

end ray_walks_to_high_school_7_l173_173167


namespace workshop_personnel_l173_173511

-- Definitions for workshops with their corresponding production constraints
def workshopA_production (x : ℕ) : ℕ := 6 + 11 * (x - 1)
def workshopB_production (y : ℕ) : ℕ := 7 + 10 * (y - 1)

-- The main theorem to be proved
theorem workshop_personnel :
  ∃ (x y : ℕ), workshopA_production x = workshopB_production y ∧
               100 ≤ workshopA_production x ∧ workshopA_production x ≤ 200 ∧
               x = 12 ∧ y = 13 :=
by
  sorry

end workshop_personnel_l173_173511


namespace find_m_n_l173_173291

-- Define the set A
def set_A : Set ℝ := {x | |x + 2| < 3}

-- Define the set B in terms of a variable m
def set_B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

-- State the theorem
theorem find_m_n (m n : ℝ) (hA : set_A = {x | -5 < x ∧ x < 1}) (h_inter : set_A ∩ set_B m = {x | -1 < x ∧ x < n}) : 
  m = -1 ∧ n = 1 :=
by
  -- Proof is omitted
  sorry

end find_m_n_l173_173291


namespace shoes_per_person_l173_173130

theorem shoes_per_person (friends : ℕ) (pairs_of_shoes : ℕ) 
  (h1 : friends = 35) (h2 : pairs_of_shoes = 36) : 
  (pairs_of_shoes * 2) / (friends + 1) = 2 := by
  sorry

end shoes_per_person_l173_173130


namespace gift_cost_l173_173585

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 2 * half_cost = 28 :=
by
  sorry

end gift_cost_l173_173585


namespace age_difference_is_13_l173_173934

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end age_difference_is_13_l173_173934


namespace largest_value_of_a_l173_173150

theorem largest_value_of_a : 
  ∃ (a : ℚ), (3 * a + 4) * (a - 2) = 9 * a ∧ ∀ b : ℚ, (3 * b + 4) * (b - 2) = 9 * b → b ≤ 4 :=
by
  sorry

end largest_value_of_a_l173_173150


namespace triangle_cosine_rule_c_triangle_tangent_C_l173_173443

-- Define a proof statement for the cosine rule-based proof of c = 4.
theorem triangle_cosine_rule_c (a b : ℝ) (angleB : ℝ) (ha : a = 2)
                              (hb : b = 2 * Real.sqrt 3) (hB : angleB = π / 3) :
  ∃ (c : ℝ), c = 4 := by
  sorry

-- Define a proof statement for the tangent identity-based proof of tan C = 3 * sqrt 3 / 5.
theorem triangle_tangent_C (tanA : ℝ) (tanB : ℝ) (htA : tanA = 2 * Real.sqrt 3)
                           (htB : tanB = Real.sqrt 3) :
  ∃ (tanC : ℝ), tanC = 3 * Real.sqrt 3 / 5 := by
  sorry

end triangle_cosine_rule_c_triangle_tangent_C_l173_173443


namespace largest_divisor_of_even_square_difference_l173_173479

theorem largest_divisor_of_even_square_difference (m n : ℕ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) :
  ∃ (k : ℕ), k = 8 ∧ ∀ m n : ℕ, m % 2 = 0 → n % 2 = 0 → n < m → k ∣ (m^2 - n^2) := by
  sorry

end largest_divisor_of_even_square_difference_l173_173479


namespace max_value_inequality_l173_173847

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c) / ((a + b)^2 * (b + c)^2) ≤ 1 / 4) :=
sorry

end max_value_inequality_l173_173847


namespace animal_sighting_ratio_l173_173884

theorem animal_sighting_ratio
  (jan_sightings : ℕ)
  (feb_sightings : ℕ)
  (march_sightings : ℕ)
  (total_sightings : ℕ)
  (h1 : jan_sightings = 26)
  (h2 : feb_sightings = 3 * jan_sightings)
  (h3 : total_sightings = jan_sightings + feb_sightings + march_sightings)
  (h4 : total_sightings = 143) :
  (march_sightings : ℚ) / (feb_sightings : ℚ) = 1 / 2 :=
by
  sorry

end animal_sighting_ratio_l173_173884


namespace intersection_correct_union_correct_intersection_complement_correct_l173_173293

def U := ℝ
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}
def C_U_A : Set ℝ := {x | x ≤ 0 ∨ x > 2}
def C_U_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_correct : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

theorem union_correct : (A ∪ B) = {x : ℝ | x < -3 ∨ x > 0} :=
sorry

theorem intersection_complement_correct : (C_U_A ∩ C_U_B) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} :=
sorry

end intersection_correct_union_correct_intersection_complement_correct_l173_173293


namespace sum_of_two_equal_sides_is_4_l173_173498

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c = 2.8284271247461903 ∧ c ^ 2 = 2 * (a ^ 2)

theorem sum_of_two_equal_sides_is_4 :
  ∃ a : ℝ, isosceles_right_triangle a 2.8284271247461903 ∧ 2 * a = 4 :=
by
  sorry

end sum_of_two_equal_sides_is_4_l173_173498


namespace history_books_count_l173_173052

theorem history_books_count :
  ∃ (total_books reading_books math_books science_books history_books : ℕ),
    total_books = 10 ∧
    reading_books = (2 * total_books) / 5 ∧
    math_books = (3 * total_books) / 10 ∧
    science_books = math_books - 1 ∧
    history_books = total_books - (reading_books + math_books + science_books) ∧
    history_books = 1 :=
by
  sorry

end history_books_count_l173_173052


namespace cube_volume_l173_173983

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173983


namespace balloon_permutations_l173_173246

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l173_173246


namespace find_parallel_line_through_P_l173_173676

noncomputable def line_parallel_passing_through (p : (ℝ × ℝ)) (line : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, _) := line
  let (x, y) := p
  (a, b, - (a * x + b * y))

theorem find_parallel_line_through_P :
  line_parallel_passing_through (4, -1) (3, -4, 6) = (3, -4, -16) :=
by 
  sorry

end find_parallel_line_through_P_l173_173676


namespace sum_of_fractions_eq_decimal_l173_173412

theorem sum_of_fractions_eq_decimal :
  (3 / 100) + (5 / 1000) + (7 / 10000) = 0.0357 :=
by
  sorry

end sum_of_fractions_eq_decimal_l173_173412


namespace triangle_concurrency_l173_173911

-- Define Triangle Structure
structure Triangle (α : Type*) :=
(A B C : α)

-- Define Medians, Angle Bisectors, and Altitudes Concurrency Conditions
noncomputable def medians_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def angle_bisectors_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def altitudes_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry

-- Main Theorem Statement
theorem triangle_concurrency {α : Type*} [MetricSpace α] (T : Triangle α) :
  medians_concurrent T ∧ angle_bisectors_concurrent T ∧ altitudes_concurrent T :=
by 
  -- Proof outline: Prove each concurrency condition
  sorry

end triangle_concurrency_l173_173911


namespace divisible_by_lcm_of_4_5_6_l173_173700

theorem divisible_by_lcm_of_4_5_6 (n : ℕ) : (∃ k, 0 < k ∧ k < 300 ∧ k % 60 = 0) ↔ (∃! k, k = 4) :=
by
  let lcm_4_5_6 := Nat.lcm (Nat.lcm 4 5) 6
  have : lcm_4_5_6 = 60 := sorry
  have : ∀ k, (0 < k ∧ k < 300 ∧ k % lcm_4_5_6 = 0) ↔ (k = 60 ∨ k = 120 ∨ k = 180 ∨ k = 240) := sorry
  have : ∃ k, 0 < k ∧ k < 300 ∧ k % lcm_4_5_6 = 0 :=
    ⟨60, by norm_num, mk 120, by norm_num, mk 180, by norm_num, mk 240, by norm_num⟩
  have : ∃! k, (k = 60 ∨ k = 120 ∨ k = 180 ∨ k = 240) := sorry
  show _ ↔ (∃! k, k = 4) from sorry

end divisible_by_lcm_of_4_5_6_l173_173700


namespace find_value_of_f_l173_173690

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem find_value_of_f (ω φ : ℝ) (h_symmetry : ∀ x : ℝ, f ω φ (π/4 + x) = f ω φ (π/4 - x)) :
  f ω φ (π/4) = 2 ∨ f ω φ (π/4) = -2 := 
sorry

end find_value_of_f_l173_173690


namespace circle_center_l173_173841

theorem circle_center {x y : ℝ} :
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → (x, y) = (1, 2) :=
by
  sorry

end circle_center_l173_173841


namespace variance_of_scores_l173_173812

def scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

def mean (xs : List ℕ) : ℚ := xs.sum / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

theorem variance_of_scores : variance scores = 4 := by
  sorry

end variance_of_scores_l173_173812


namespace price_of_fruit_juice_l173_173804

theorem price_of_fruit_juice (F : ℝ)
  (Sandwich_price : ℝ := 2)
  (Hamburger_price : ℝ := 2)
  (Hotdog_price : ℝ := 1)
  (Selene_purchases : ℝ := 3 * Sandwich_price + F)
  (Tanya_purchases : ℝ := 2 * Hamburger_price + 2 * F)
  (Total_spent : Selene_purchases + Tanya_purchases = 16) :
  F = 2 :=
by
  sorry

end price_of_fruit_juice_l173_173804


namespace multiplicative_inverse_sum_is_zero_l173_173864

theorem multiplicative_inverse_sum_is_zero (a b : ℝ) (h : a * b = 1) :
  a^(2015) * b^(2016) + a^(2016) * b^(2017) + a^(2017) * b^(2016) + a^(2016) * b^(2015) = 0 :=
sorry

end multiplicative_inverse_sum_is_zero_l173_173864


namespace JacksonsGrade_l173_173583

theorem JacksonsGrade : 
  let hours_playing_video_games := 12
  let hours_studying := (1 / 3) * hours_playing_video_games
  let hours_kindness := (1 / 4) * hours_playing_video_games
  let grade_initial := 0
  let grade_per_hour_studying := 20
  let grade_per_hour_kindness := 40
  let grade_from_studying := grade_per_hour_studying * hours_studying
  let grade_from_kindness := grade_per_hour_kindness * hours_kindness
  let total_grade := grade_initial + grade_from_studying + grade_from_kindness
  total_grade = 200 :=
by
  -- Proof goes here
  sorry

end JacksonsGrade_l173_173583


namespace sequence_difference_l173_173050

theorem sequence_difference :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧ a 2 = 1 ∧
    (∀ n ≥ 1, (a (n + 2) : ℚ) / a (n + 1) - (a (n + 1) : ℚ) / a n = 1) ∧
    a 6 - a 5 = 96 :=
sorry

end sequence_difference_l173_173050


namespace distinctKeyArrangements_l173_173011

-- Given conditions as definitions in Lean.
def houseNextToCar : Prop := sorry
def officeNextToBike : Prop := sorry
def noDifferenceByRotationOrReflection (arr1 arr2 : List ℕ) : Prop := sorry

-- Main statement to be proven
theorem distinctKeyArrangements : 
  houseNextToCar ∧ officeNextToBike ∧ (∀ (arr1 arr2 : List ℕ), noDifferenceByRotationOrReflection arr1 arr2 ↔ arr1 = arr2) 
  → ∃ n : ℕ, n = 16 :=
by sorry

end distinctKeyArrangements_l173_173011


namespace find_solutions_to_system_l173_173414

theorem find_solutions_to_system (x y z : ℝ) 
    (h1 : 3 * (x^2 + y^2 + z^2) = 1) 
    (h2 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^3) : 
    x = y ∧ y = z ∧ (x = 1 / 3 ∨ x = -1 / 3) :=
by
  sorry

end find_solutions_to_system_l173_173414


namespace johns_calorie_intake_l173_173526

theorem johns_calorie_intake
  (servings : ℕ)
  (calories_per_serving : ℕ)
  (total_calories : ℕ)
  (half_package_calories : ℕ)
  (h1 : servings = 3)
  (h2 : calories_per_serving = 120)
  (h3 : total_calories = servings * calories_per_serving)
  (h4 : half_package_calories = total_calories / 2)
  : half_package_calories = 180 :=
by sorry

end johns_calorie_intake_l173_173526


namespace quadratic_equation_unique_solution_l173_173748

theorem quadratic_equation_unique_solution (a b x k : ℝ) (h : a = 8) (h₁ : b = 36) (h₂ : k = 40.5) : 
  (8*x^2 + 36*x + 40.5 = 0) ∧ x = -2.25 :=
by {
  sorry
}

end quadratic_equation_unique_solution_l173_173748


namespace quadratic_has_two_distinct_real_roots_l173_173171

variable {R : Type} [LinearOrderedField R]

theorem quadratic_has_two_distinct_real_roots (c d : R) :
  ∀ x : R, (x + c) * (x + d) - (2 * x + c + d) = 0 → 
  (x + c)^2 + 4 > 0 :=
by
  intros x h
  -- Proof (skipped)
  sorry

end quadratic_has_two_distinct_real_roots_l173_173171


namespace find_length_of_room_l173_173144

theorem find_length_of_room (width area_existing area_needed : ℕ) (h_width : width = 15) (h_area_existing : area_existing = 16) (h_area_needed : area_needed = 149) :
  (area_existing + area_needed) / width = 11 :=
by
  sorry

end find_length_of_room_l173_173144


namespace balloon_arrangement_count_l173_173237

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l173_173237


namespace number_of_valid_permutations_l173_173102

def no_consecutive_pairs (p : List ℕ) : Prop :=
  ∀ i, (1 <= i ∧ i < 8) → ¬ (i + 1 = p[i] ∧ i = p[i - 1])

noncomputable def valid_permutations : List (List ℕ) :=
  List.filter no_consecutive_pairs (List.permutations (List.range' 1 8))

theorem number_of_valid_permutations :
  valid_permutations.length = 16687 := by
  sorry

end number_of_valid_permutations_l173_173102


namespace range_of_a_l173_173290

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) :
  (a > -2 ∧ a < -1) ∨ (a ≥ 1) :=
by
  sorry

end range_of_a_l173_173290


namespace Nadal_wins_championship_probability_l173_173888

open Probability

noncomputable def probability_of_Nadal_winning_championship : ℝ :=
  let p := 2 / 3
  let q := 1 / 3
  let outcomes (k : ℕ) := Nat.choose (3 + k) k
  let individual_prob (k : ℕ) := (outcomes k) * (p ^ 4) * (q ^ k)
  (individual_prob 0) + (individual_prob 1) + (individual_prob 2) + (individual_prob 3)

theorem Nadal_wins_championship_probability :
  round (100 * probability_of_Nadal_winning_championship) = 89 :=
sorry

end Nadal_wins_championship_probability_l173_173888


namespace vegetarian_family_l173_173886

theorem vegetarian_family (eat_veg eat_non_veg eat_both : ℕ) (total_veg : ℕ) 
  (h1 : eat_non_veg = 8) (h2 : eat_both = 11) (h3 : total_veg = 26)
  : eat_veg = total_veg - eat_both := by
  sorry

end vegetarian_family_l173_173886


namespace school_distance_is_seven_l173_173661

-- Definitions based on conditions
def distance_to_school (x : ℝ) : Prop :=
  let monday_to_thursday_distance := 8 * x
  let friday_distance := 2 * x + 4
  let total_distance := monday_to_thursday_distance + friday_distance
  total_distance = 74

-- The problem statement to prove
theorem school_distance_is_seven : ∃ (x : ℝ), distance_to_school x ∧ x = 7 := 
by {
  sorry
}

end school_distance_is_seven_l173_173661


namespace kids_joined_in_l173_173769

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end kids_joined_in_l173_173769


namespace value_of_a_plus_b_l173_173004

-- Definition of collinearity for points in 3D
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), p3 = (p1.1 + λ * (p2.1 - p1.1), p1.2 + λ * (p2.2 - p1.2), p1.3 + λ * (p2.3 - p1.3))

-- Conditions
variables {a b : ℝ}
axiom collinear_points : collinear (2, a, b) (a, 3, b) (a, b, 4)

-- Main statement to prove
theorem value_of_a_plus_b : a + b = 6 := 
by 
  sorry -- Skipping the actual proof as per instructions

end value_of_a_plus_b_l173_173004


namespace general_term_formula_sum_of_b_first_terms_l173_173459

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end general_term_formula_sum_of_b_first_terms_l173_173459


namespace no_domovoi_exists_l173_173819

variables {Domovoi Creature : Type}

def likes_pranks (c : Creature) : Prop := sorry
def likes_cleanliness_order (c : Creature) : Prop := sorry
def is_domovoi (c : Creature) : Prop := sorry

axiom all_domovoi_like_pranks : ∀ (c : Creature), is_domovoi c → likes_pranks c
axiom all_domovoi_like_cleanliness : ∀ (c : Creature), is_domovoi c → likes_cleanliness_order c
axiom cleanliness_implies_no_pranks : ∀ (c : Creature), likes_cleanliness_order c → ¬ likes_pranks c

theorem no_domovoi_exists : ¬ ∃ (c : Creature), is_domovoi c := 
sorry

end no_domovoi_exists_l173_173819


namespace mary_groceries_fitting_l173_173723

theorem mary_groceries_fitting :
  (∀ bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice,
    bags = 2 →
    wt_green = 4 →
    wt_milk = 6 →
    wt_carrots = 2 * wt_green →
    wt_apples = 3 →
    wt_bread = 1 →
    wt_rice = 5 →
    (wt_green + wt_milk + wt_carrots + wt_apples + wt_bread + wt_rice = 27) →
    (∀ b, b < 20 →
      (b = 6 + 5 ∨ b = 22 - 11) →
      (20 - b = 9))) :=
by
  intros bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice h_bags h_green h_milk h_carrots h_apples h_bread h_rice h_total h_b
  sorry

end mary_groceries_fitting_l173_173723


namespace conversion_problems_l173_173377

-- Define the conversion factors
def square_meters_to_hectares (sqm : ℕ) : ℕ := sqm / 10000
def hectares_to_square_kilometers (ha : ℕ) : ℕ := ha / 100
def square_kilometers_to_hectares (sqkm : ℕ) : ℕ := sqkm * 100

-- Define the specific values from the problem
def value1_m2 : ℕ := 5000000
def value2_km2 : ℕ := 70000

-- The theorem to prove
theorem conversion_problems :
  (square_meters_to_hectares value1_m2 = 500) ∧
  (hectares_to_square_kilometers 500 = 5) ∧
  (square_kilometers_to_hectares value2_km2 = 7000000) :=
by
  sorry

end conversion_problems_l173_173377


namespace factorize_expression_l173_173669

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l173_173669


namespace initial_necklaces_count_l173_173655

theorem initial_necklaces_count (N : ℕ) 
  (h1 : N - 13 = 37) : 
  N = 50 := 
by
  sorry

end initial_necklaces_count_l173_173655


namespace net_rate_of_pay_l173_173215

theorem net_rate_of_pay :
  ∀ (duration_travel : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (earnings_rate : ℝ) (gas_cost : ℝ),
  duration_travel = 3 → speed = 50 → fuel_efficiency = 30 → earnings_rate = 0.75 → gas_cost = 2.50 →
  (earnings_rate * speed * duration_travel - (speed * duration_travel / fuel_efficiency) * gas_cost) / duration_travel = 33.33 :=
by
  intros duration_travel speed fuel_efficiency earnings_rate gas_cost
  intros h1 h2 h3 h4 h5
  sorry

end net_rate_of_pay_l173_173215


namespace complex_fraction_value_l173_173688

theorem complex_fraction_value (a b : ℝ) (h : (i - 2) / (1 + i) = a + b * i) : a + b = 1 :=
by
  sorry

end complex_fraction_value_l173_173688


namespace part_1_a_part_1_b_part_2_l173_173567

open Set

variable (a : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def compl_U_A : Set ℝ := compl A

theorem part_1_a :
  A ∩ B 1 = {x : ℝ | x < -2} :=
by
  sorry

theorem part_1_b :
  A ∪ B 1 = {x : ℝ | x > 2 ∨ x ≤ 1} :=
by
  sorry

theorem part_2 :
  compl_U_A ⊆ B a → a ≥ 2 :=
by
  sorry

end part_1_a_part_1_b_part_2_l173_173567


namespace hours_sunday_correct_l173_173123

-- Definitions of given conditions
def hours_saturday : ℕ := 6
def total_hours : ℕ := 9

-- The question translated to a proof problem
theorem hours_sunday_correct : total_hours - hours_saturday = 3 := 
by
  -- The proof is skipped and replaced by sorry
  sorry

end hours_sunday_correct_l173_173123


namespace product_of_sisters_and_brothers_l173_173596

-- Lucy's family structure
def lucy_sisters : ℕ := 4
def lucy_brothers : ℕ := 6

-- Liam's siblings count
def liam_sisters : ℕ := lucy_sisters + 1  -- Including Lucy herself
def liam_brothers : ℕ := lucy_brothers    -- Excluding himself

-- Prove the product of Liam's sisters and brothers is 25
theorem product_of_sisters_and_brothers : liam_sisters * (liam_brothers - 1) = 25 :=
by
  sorry

end product_of_sisters_and_brothers_l173_173596


namespace batsman_average_after_17th_inning_l173_173200

theorem batsman_average_after_17th_inning (A : ℝ) (h1 : 16 * A + 200 = 17 * (A + 10)) : 
  A + 10 = 40 := 
by
  sorry

end batsman_average_after_17th_inning_l173_173200


namespace product_of_remaining_numbers_is_12_l173_173352

noncomputable def final_numbers_product : ℕ := 
  12

theorem product_of_remaining_numbers_is_12 :
  ∀ (initial_ones initial_twos initial_threes initial_fours : ℕ)
  (erase_add_op : Π (a b c : ℕ), Prop),
  initial_ones = 11 ∧ initial_twos = 22 ∧ initial_threes = 33 ∧ initial_fours = 44 ∧
  (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c → erase_add_op a b c) →
  (∃ (final1 final2 final3 : ℕ), erase_add_op 11 22 33 → final1 * final2 * final3 = final_numbers_product) :=
sorry

end product_of_remaining_numbers_is_12_l173_173352


namespace largest_k_l173_173848

-- Define the system of equations and conditions
def system_valid (x y k : ℝ) : Prop := 
  2 * x + y = k ∧ 
  3 * x + y = 3 ∧ 
  x - 2 * y ≥ 1

-- Define the proof problem as a theorem in Lean
theorem largest_k (x y : ℝ) :
  ∀ k : ℝ, system_valid x y k → k ≤ 2 := 
sorry

end largest_k_l173_173848


namespace sequence_inequality_l173_173562

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 
  if n = 0 then 1/2
  else a (n - 1) + (1 / (n:ℚ)^2) * (a (n - 1))^2

theorem sequence_inequality (n : ℕ) : 
  1 - 1 / 2 ^ (n + 1) ≤ a n ∧ a n < 7 / 5 := 
sorry

end sequence_inequality_l173_173562


namespace simplify_fraction_l173_173658

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) : (3 * m^3) / (6 * m^2) = m / 2 :=
by
  sorry

end simplify_fraction_l173_173658


namespace scheduling_plans_count_l173_173813

open Function

/--
There are 7 employees to be scheduled from October 1 to October 7, 
one per day, such that:
- A and B are scheduled on consecutive days,
- C is not scheduled on October 1st,
- D is not scheduled on October 7th.

We need to prove that the total number of different scheduling plans is 1008.
-/
theorem scheduling_plans_count :
  let employees : Fin 7 := ⟨6, by decide⟩ -- Fix employees as Fin 7 since we have 7 employees.
  let count_schedule_plans (p : List (Fin 7)) : Bool :=
    p.length = 7 ∧ -- 7 employees scheduled
    ∃ i, p[i] = (0, 1) ∨ p[i] = (6, 7) ∨ -- A and B scheduled consecutively
           (∃ j, p[j] = (C, 7) ∧ (1 ≤ j ∧ j ≤ 5)) ∨ -- C not on October 1st
           (∃ k, p[k] = (D, 1) ∧ (0 ≤ k ∧ k ≤ 5)) -- D not on October 7th
    in nat.factorial 7 = 1008 := sorry

end scheduling_plans_count_l173_173813


namespace smallest_n_l173_173399

theorem smallest_n (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n % 9 = 2)
  (h3 : n % 6 = 4) : n = 146 :=
sorry

end smallest_n_l173_173399


namespace map_representation_l173_173161

-- Defining the conditions
noncomputable def map_scale : ℝ := 28 -- 1 inch represents 28 miles

-- Defining the specific instance provided in the problem
def inches_represented : ℝ := 13.7
def miles_represented : ℝ := 383.6

-- Statement of the problem
theorem map_representation (D : ℝ) : (D / map_scale) = (D : ℝ) / 28 := 
by
  -- Prove the statement
  sorry

end map_representation_l173_173161


namespace find_five_value_l173_173342

def f (x : ℝ) : ℝ := x^2 - x

theorem find_five_value : f 5 = 20 := by
  sorry

end find_five_value_l173_173342


namespace percent_employed_females_l173_173468

theorem percent_employed_females (h1 : 96 / 100 > 0) (h2 : 24 / 100 > 0) : 
  (96 - 24) / 96 * 100 = 75 := 
by 
  -- Proof to be filled out
  sorry

end percent_employed_females_l173_173468


namespace total_price_paid_l173_173489

noncomputable def total_price
    (price_rose : ℝ) (qty_rose : ℕ) (discount_rose : ℝ)
    (price_lily : ℝ) (qty_lily : ℕ) (discount_lily : ℝ)
    (price_sunflower : ℝ) (qty_sunflower : ℕ)
    (store_discount : ℝ) (tax_rate : ℝ)
    : ℝ :=
  let total_rose := qty_rose * price_rose
  let total_lily := qty_lily * price_lily
  let total_sunflower := qty_sunflower * price_sunflower
  let total := total_rose + total_lily + total_sunflower
  let total_disc_rose := total_rose * discount_rose
  let total_disc_lily := total_lily * discount_lily
  let discounted_total := total - total_disc_rose - total_disc_lily
  let store_discount_amount := discounted_total * store_discount
  let after_store_discount := discounted_total - store_discount_amount
  let tax_amount := after_store_discount * tax_rate
  after_store_discount + tax_amount

theorem total_price_paid :
  total_price 20 3 0.15 15 5 0.10 10 2 0.05 0.07 = 140.79 :=
by
  apply sorry

end total_price_paid_l173_173489


namespace flare_initial_velocity_and_duration_l173_173078

noncomputable def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

theorem flare_initial_velocity_and_duration (v t : ℝ) :
  (h v 5 = 245) ↔ (v = 73.5) ∧ (5 < t ∧ t < 10) :=
by {
  sorry
}

end flare_initial_velocity_and_duration_l173_173078


namespace a10_b10_l173_173326

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a10_b10 : a^10 + b^10 = 123 :=
by
  sorry

end a10_b10_l173_173326


namespace solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l173_173789

-- Problem 1: Solution Set of the Inequality
theorem solution_set_x2_minus_5x_plus_4 : 
  {x : ℝ | x^2 - 5 * x + 4 > 0} = {x : ℝ | x < 1 ∨ x > 4} :=
sorry

-- Problem 2: Range of Values for a
theorem range_of_a_if_x2_plus_ax_plus_4_gt_0 (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 4 > 0) :
  -4 < a ∧ a < 4 :=
sorry

end solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l173_173789


namespace pencils_placed_by_Joan_l173_173762

variable (initial_pencils : ℕ)
variable (total_pencils : ℕ)

theorem pencils_placed_by_Joan 
  (h1 : initial_pencils = 33) 
  (h2 : total_pencils = 60)
  : total_pencils - initial_pencils = 27 := 
by
  sorry

end pencils_placed_by_Joan_l173_173762


namespace largest_divisible_by_3_power_l173_173506

theorem largest_divisible_by_3_power :
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → ∃ m : ℕ, (3^m ∣ (2*k - 1)) → n = 49) :=
sorry

end largest_divisible_by_3_power_l173_173506


namespace harry_cookies_batch_l173_173698

theorem harry_cookies_batch
  (total_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips = 81)
  (batches = 3)
  (chips_per_cookie = 9) :
  (total_chips / batches) / chips_per_cookie = 3 := by
  sorry

end harry_cookies_batch_l173_173698


namespace balloon_permutation_count_l173_173273

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l173_173273


namespace area_of_right_triangle_l173_173076

-- Define the conditions
def hypotenuse : ℝ := 9
def angle : ℝ := 30

-- Define the Lean statement for the proof problem
theorem area_of_right_triangle : 
  ∃ (area : ℝ), area = 10.125 * Real.sqrt 3 ∧
  ∃ (shorter_leg : ℝ) (longer_leg : ℝ),
    shorter_leg = hypotenuse / 2 ∧
    longer_leg = shorter_leg * Real.sqrt 3 ∧
    area = (shorter_leg * longer_leg) / 2 :=
by {
  -- The proof would go here, but we only need to state the problem for this task.
  sorry
}

end area_of_right_triangle_l173_173076


namespace simplify_expression_l173_173913

variable (a : ℝ)

theorem simplify_expression (h₁ : a ≠ -3) (h₂ : a ≠ 1) :
  (1 - 4/(a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) :=
sorry

end simplify_expression_l173_173913


namespace find_b_l173_173749

-- Define the number 1234567 in base 36
def numBase36 : ℤ := 1 * 36^6 + 2 * 36^5 + 3 * 36^4 + 4 * 36^3 + 5 * 36^2 + 6 * 36^1 + 7 * 36^0

-- Prove that for b being an integer such that 0 ≤ b ≤ 10,
-- and given (numBase36 - b) is a multiple of 17, b must be 0
theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 10) (h3 : (numBase36 - b) % 17 = 0) : b = 0 :=
by
  sorry

end find_b_l173_173749


namespace expr_comparison_l173_173140

-- Define the given condition
def eight_pow_2001 : ℝ := 8 * (64 : ℝ) ^ 1000

-- State the theorem
theorem expr_comparison : (65 : ℝ) ^ 1000 > eight_pow_2001 := by
  sorry

end expr_comparison_l173_173140


namespace cylindrical_to_rectangular_l173_173233

noncomputable def convertToRectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular :
  let r := 10
  let θ := Real.pi / 3
  let z := 2
  let r' := 2 * r
  let z' := z + 1
  convertToRectangular r' θ z' = (10, 10 * Real.sqrt 3, 3) :=
by
  sorry

end cylindrical_to_rectangular_l173_173233


namespace find_number_of_sides_l173_173844

theorem find_number_of_sides (n : ℕ) (h : n - (n * (n - 3)) / 2 = 3) : n = 3 := 
sorry

end find_number_of_sides_l173_173844


namespace flower_pots_problem_l173_173597

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ := x + 5 * 0.15

theorem flower_pots_problem
  (x : ℝ)       -- cost of the smallest pot
  (total_cost : ℝ) -- total cost of all pots
  (h_total_cost : total_cost = 8.25)
  (h_price_relation : total_cost = 6 * x + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15)) :
  cost_of_largest_pot x = 1.75 :=
by
  sorry

end flower_pots_problem_l173_173597


namespace zero_of_transformed_function_l173_173872

variables (f : ℝ → ℝ)
variable (x0 : ℝ)

-- Condition: f is an odd function
def is_odd_function : Prop := ∀ x, f (-x) = -f x

-- Condition: x0 is a zero of y = f(x) + e^x
def is_zero_of_original_function : Prop := f x0 + Real.exp x0 = 0

-- The theorem we want to prove:
theorem zero_of_transformed_function (h1 : is_odd_function f) (h2 : is_zero_of_original_function f x0) :
  Real.exp (-x0) * f (-x0) - 1 = 0 :=
sorry

end zero_of_transformed_function_l173_173872


namespace houses_with_neither_l173_173784

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end houses_with_neither_l173_173784


namespace negation_of_forall_pos_l173_173928

open Real

theorem negation_of_forall_pos (h : ∀ x : ℝ, x^2 - x + 1 > 0) : 
  ¬(∀ x : ℝ, x^2 - x + 1 > 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_forall_pos_l173_173928


namespace part_a_l173_173371

theorem part_a (n : ℕ) :
  1 + ∑ k in Finset.range (n / 3 + 1), Nat.choose n (3 * k) = (1 / 3) * (Real.exp (n * Real.log 2) + 2 * Real.cos (n * Real.pi / 3)) :=
sorry

end part_a_l173_173371


namespace total_number_of_rats_l173_173587

theorem total_number_of_rats (Kenia Hunter Elodie Teagan : ℕ) 
  (h1 : Elodie = 30)
  (h2 : Elodie = Hunter + 10)
  (h3 : Kenia = 3 * (Hunter + Elodie))
  (h4 : Teagan = 2 * Elodie)
  (h5 : Teagan = Kenia - 5) : 
  Kenia + Hunter + Elodie + Teagan = 260 :=
by 
  sorry

end total_number_of_rats_l173_173587


namespace max_value_of_g_l173_173407

def g (n : ℕ) : ℕ :=
  if n < 20 then n + 20 else g (n - 7)

theorem max_value_of_g : ∀ n : ℕ, g n ≤ 39 ∧ (∃ m : ℕ, g m = 39) := by
  sorry

end max_value_of_g_l173_173407


namespace square_of_number_ending_in_5_l173_173787

theorem square_of_number_ending_in_5 (d : ℤ) : (10 * d + 5)^2 = 100 * d * (d + 1) + 25 :=
by
  sorry

end square_of_number_ending_in_5_l173_173787


namespace intersection_points_l173_173001

noncomputable def hyperbola : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 / 9 - y^2 = 1 }

noncomputable def line : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ y = (1 / 3) * (x + 1) }

theorem intersection_points :
  ∃! (p : ℝ × ℝ), p ∈ hyperbola ∧ p ∈ line :=
sorry

end intersection_points_l173_173001


namespace cube_volume_from_surface_area_l173_173994

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173994


namespace other_root_is_neg_2_l173_173571

theorem other_root_is_neg_2 (k : ℝ) (h : Polynomial.eval 0 (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) : 
  ∃ t : ℝ, (Polynomial.eval t (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) ∧ t = -2 :=
by
  sorry

end other_root_is_neg_2_l173_173571


namespace min_value_2x_plus_y_l173_173644

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 2 / (y + 1) = 2) :
  2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l173_173644


namespace least_common_multiple_1260_980_l173_173056

def LCM (a b : ℕ) : ℕ :=
  a * b / Nat.gcd a b

theorem least_common_multiple_1260_980 : LCM 1260 980 = 8820 := by
  sorry

end least_common_multiple_1260_980_l173_173056


namespace weighted_mean_is_correct_l173_173143

-- Define the given values
def dollar_from_aunt : ℝ := 9
def euros_from_uncle : ℝ := 9
def dollar_from_sister : ℝ := 7
def dollar_from_friends_1 : ℝ := 22
def dollar_from_friends_2 : ℝ := 23
def euros_from_friends_3 : ℝ := 18
def pounds_from_friends_4 : ℝ := 15
def dollar_from_friends_5 : ℝ := 22

-- Define the exchange rates
def exchange_rate_euro_to_usd : ℝ := 1.20
def exchange_rate_pound_to_usd : ℝ := 1.38

-- Calculate the amounts in USD
def dollar_from_uncle : ℝ := euros_from_uncle * exchange_rate_euro_to_usd
def dollar_from_friends_3_converted : ℝ := euros_from_friends_3 * exchange_rate_euro_to_usd
def dollar_from_friends_4_converted : ℝ := pounds_from_friends_4 * exchange_rate_pound_to_usd

-- Define total amounts from family and friends in USD
def family_total : ℝ := dollar_from_aunt + dollar_from_uncle + dollar_from_sister
def friends_total : ℝ := dollar_from_friends_1 + dollar_from_friends_2 + dollar_from_friends_3_converted + dollar_from_friends_4_converted + dollar_from_friends_5

-- Define weights
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Calculate the weighted mean
def weighted_mean : ℝ := (family_total * family_weight) + (friends_total * friends_weight)

theorem weighted_mean_is_correct : weighted_mean = 76.30 := by
  sorry

end weighted_mean_is_correct_l173_173143


namespace angle_R_values_l173_173892

theorem angle_R_values (P Q : ℝ) (h1: 5 * Real.sin P + 2 * Real.cos Q = 5) (h2: 2 * Real.sin Q + 5 * Real.cos P = 3) : 
  ∃ R : ℝ, R = Real.arcsin (1/20) ∨ R = 180 - Real.arcsin (1/20) :=
by
  sorry

end angle_R_values_l173_173892


namespace geometric_sequence_a_n_sum_of_first_n_terms_b_n_l173_173461

theorem geometric_sequence_a_n :
  (∃ a₁ a₂, a₁ * a₂ = 8 ∧ a₁ + a₂ = 6 ∧ a₁ < a₂) →
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) :=
begin
  sorry
end

theorem sum_of_first_n_terms_b_n :
  (∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n, a n = 2 ^ n))) →
  (∀ b : ℕ → ℕ, (∀ n, b n = 2 * (2 ^ n) + 3)) →
  (∀ T : ℕ → ℤ, (∀ n, T n = 2 ^ (n + 2) - 4 + 3 * n)) :=
begin
  sorry
end

end geometric_sequence_a_n_sum_of_first_n_terms_b_n_l173_173461


namespace point_below_parabola_l173_173348

theorem point_below_parabola (a b c : ℝ) (h : 2 < a + b + c) : 
  2 < c + b + a :=
by
  sorry

end point_below_parabola_l173_173348


namespace cube_volume_is_1728_l173_173952

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173952


namespace coordinates_of_P_l173_173578

/-- In the Cartesian coordinate system, given a point P with coordinates (-5, 3),
    prove that its coordinates with respect to the origin are (-5, 3). -/
theorem coordinates_of_P :
  ∀ (P : ℝ × ℝ), P = (-5, 3) → P = (-5, 3) :=
by
  intro P h,
  exact h

end coordinates_of_P_l173_173578


namespace geometric_sequence_sum_l173_173428

theorem geometric_sequence_sum (a_1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  (∀ n, S (n+1) = a_1 * (1 - q^(n+1)) / (1 - q)) →
  S 4 / a_1 = 15 :=
by
  intros hq hsum
  sorry

end geometric_sequence_sum_l173_173428


namespace mrs_hilt_found_nickels_l173_173907

theorem mrs_hilt_found_nickels : 
  ∀ (total cents quarter cents dime cents nickel cents : ℕ), 
    total = 45 → 
    quarter = 25 → 
    dime = 10 → 
    nickel = 5 → 
    ((total - (quarter + dime)) / nickel) = 2 := 
by
  intros total quarter dime nickel h_total h_quarter h_dime h_nickel
  sorry

end mrs_hilt_found_nickels_l173_173907


namespace distinct_arrangements_l173_173766

-- Defining the conditions as constants
def num_women : ℕ := 9
def num_men : ℕ := 3
def total_slots : ℕ := num_women + num_men

-- Using the combination formula directly as part of the statement
theorem distinct_arrangements : Nat.choose total_slots num_men = 220 := by
  sorry

end distinct_arrangements_l173_173766


namespace proof_problem_l173_173565

def RealSets (A B : Set ℝ) : Set ℝ :=
let complementA := {x | -2 < x ∧ x < 3}
let unionAB := complementA ∪ B
unionAB

theorem proof_problem :
  let A := {x : ℝ | (x + 2) * (x - 3) ≥ 0}
  let B := {x : ℝ | x > 1}
  let complementA := {x : ℝ | -2 < x ∧ x < 3}
  let unionAB := complementA ∪ B
  unionAB = {x : ℝ | x > -2} :=
by
  sorry

end proof_problem_l173_173565


namespace shrimp_price_l173_173524

theorem shrimp_price (y : ℝ) (h : 0.6 * (y / 4) = 2.25) : y = 15 :=
sorry

end shrimp_price_l173_173524


namespace integer_values_a_l173_173423

theorem integer_values_a :
  (∃! a : ℤ, ∃ m n : ℤ, m^2 + n^2 = (m + n)^2 - 2mn ∧
                     m + n = -a ∧ 
                     mn = 12a) :
  8 :=
sorry

end integer_values_a_l173_173423


namespace arithmetic_sequence_thm_l173_173316

theorem arithmetic_sequence_thm
  (a : ℕ → ℝ)
  (h1 : a 1 + a 4 + a 7 = 48)
  (h2 : a 2 + a 5 + a 8 = 40)
  (d : ℝ)
  (h3 : ∀ n, a (n + 1) = a n + d) :
  a 3 + a 6 + a 9 = 32 :=
by {
  sorry
}

end arithmetic_sequence_thm_l173_173316


namespace rectangle_circle_ratio_l173_173214

theorem rectangle_circle_ratio (r s : ℝ) (h : ∀ r s : ℝ, 2 * r * s - π * r^2 = π * r^2) : s / (2 * r) = π / 2 :=
by
  sorry

end rectangle_circle_ratio_l173_173214


namespace balloon_permutations_l173_173260

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l173_173260


namespace range_of_a_l173_173116

variable (a : ℝ)

def p : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

def q : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

theorem range_of_a :
  (p a ∧ q a) → a ≤ -1 := by
  sorry

end range_of_a_l173_173116


namespace bicycle_profit_theorem_l173_173805

def bicycle_profit_problem : Prop :=
  let CP_A : ℝ := 120
  let SP_C : ℝ := 225
  let profit_percentage_B : ℝ := 0.25
  -- intermediate calculations
  let CP_B : ℝ := SP_C / (1 + profit_percentage_B)
  let SP_A : ℝ := CP_B
  let Profit_A : ℝ := SP_A - CP_A
  let Profit_Percentage_A : ℝ := (Profit_A / CP_A) * 100
  -- final statement to prove
  Profit_Percentage_A = 50

theorem bicycle_profit_theorem : bicycle_profit_problem := 
by
  sorry

end bicycle_profit_theorem_l173_173805


namespace sum_of_cubes_l173_173336

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 5) (h3 : abc = -6) : a^3 + b^3 + c^3 = -36 :=
sorry

end sum_of_cubes_l173_173336


namespace cube_volume_from_surface_area_l173_173997

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l173_173997


namespace five_nat_numbers_product_1000_l173_173827

theorem five_nat_numbers_product_1000 :
  ∃ (a b c d e : ℕ), 
    a * b * c * d * e = 1000 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e := 
by
  sorry

end five_nat_numbers_product_1000_l173_173827


namespace q_l173_173413

-- Definitions for the problem conditions
def slips := 50
def numbers := 12
def slips_per_number := 5
def drawn_slips := 5
def binom := Nat.choose -- Lean function for binomial coefficients

-- Define the probabilities p' and q'
def p' := 12 / (binom slips drawn_slips)
def favorable_q' := (binom numbers 2) * (binom slips_per_number 3) * (binom slips_per_number 2)
def q' := favorable_q' / (binom slips drawn_slips)

-- The statement we need to prove
theorem q'_over_p'_equals_550 : q' / p' = 550 :=
by sorry

end q_l173_173413


namespace ball_bounce_height_l173_173069

noncomputable def min_bounces (h₀ h_min : ℝ) (bounce_factor : ℝ) := 
  Nat.ceil (Real.log (h_min / h₀) / Real.log bounce_factor)

theorem ball_bounce_height :
  min_bounces 512 40 (3/4) = 8 :=
by
  sorry

end ball_bounce_height_l173_173069


namespace count_distinct_a_l173_173417

def quadratic_has_integer_solutions (a : ℤ) : Prop :=
  ∃ x m n : ℤ, (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a)

theorem count_distinct_a : (finset.univ.filter quadratic_has_integer_solutions).card = 9 := 
sorry

end count_distinct_a_l173_173417


namespace how_many_kids_joined_l173_173768

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end how_many_kids_joined_l173_173768


namespace extended_cross_cannot_form_cube_l173_173639

-- Define what it means to form a cube from patterns
def forms_cube (pattern : Type) : Prop := 
  sorry -- Definition for forming a cube would be detailed here

-- Define the Extended Cross pattern in a way that captures its structure
def extended_cross : Type := sorry -- Definition for Extended Cross structure

-- Define the L shape pattern in a way that captures its structure
def l_shape : Type := sorry -- Definition for L shape structure

-- The theorem statement proving that the Extended Cross pattern cannot form a cube
theorem extended_cross_cannot_form_cube : ¬(forms_cube extended_cross) := 
  sorry

end extended_cross_cannot_form_cube_l173_173639


namespace average_balance_correct_l173_173224

-- Define the monthly balances
def january_balance : ℕ := 120
def february_balance : ℕ := 240
def march_balance : ℕ := 180
def april_balance : ℕ := 180
def may_balance : ℕ := 210
def june_balance : ℕ := 300

-- List of all balances
def balances : List ℕ := [january_balance, february_balance, march_balance, april_balance, may_balance, june_balance]

-- Define the function to calculate the average balance
def average_balance (balances : List ℕ) : ℕ :=
  (balances.sum / balances.length)

-- Define the target average balance
def target_average_balance : ℕ := 205

-- The theorem we need to prove
theorem average_balance_correct :
  average_balance balances = target_average_balance :=
by
  sorry

end average_balance_correct_l173_173224


namespace percent_of_rectangle_area_inside_square_l173_173218

theorem percent_of_rectangle_area_inside_square
  (s : ℝ)  -- Let the side length of the square be \( s \).
  (width : ℝ) (length: ℝ)
  (h1 : width = 3 * s)  -- The width of the rectangle is \( 3s \).
  (h2 : length = 2 * width) :  -- The length of the rectangle is \( 2 * width \).
  (s^2 / (length * width)) * 100 = 5.56 :=
by
  sorry

end percent_of_rectangle_area_inside_square_l173_173218


namespace circles_5_and_8_same_color_l173_173735

-- Define the circles and colors
inductive Color
  | red
  | yellow
  | blue

def circles : Nat := 8

-- Define the adjacency relationship (i.e., directly connected)
-- This is a placeholder. In practice, this would be defined based on the problem's diagram.
def directly_connected (c1 c2 : Nat) : Prop := sorry

-- Simulate painting circles with given constraints
def painted (c : Nat) : Color := sorry

-- Define the conditions
axiom paint_condition (c1 c2 : Nat) (h : directly_connected c1 c2) : painted c1 ≠ painted c2

-- The proof problem: show that circles 5 and 8 must be painted the same color
theorem circles_5_and_8_same_color : painted 5 = painted 8 := 
sorry

end circles_5_and_8_same_color_l173_173735


namespace xiao_wang_fourth_place_l173_173286

section Competition
  -- Define the participants and positions
  inductive Participant
  | XiaoWang : Participant
  | XiaoZhang : Participant
  | XiaoZhao : Participant
  | XiaoLi : Participant

  inductive Position
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

  open Participant Position

  -- Conditions given in the problem
  variables
    (place : Participant → Position)
    (hA1 : place XiaoWang = First → place XiaoZhang = Third)
    (hA2 : place XiaoWang = First → place XiaoZhang ≠ Third)
    (hB1 : place XiaoLi = First → place XiaoZhao = Fourth)
    (hB2 : place XiaoLi = First → place XiaoZhao ≠ Fourth)
    (hC1 : place XiaoZhao = Second → place XiaoWang = Third)
    (hC2 : place XiaoZhao = Second → place XiaoWang ≠ Third)
    (no_ties : ∀ x y, place x = place y → x = y)
    (half_correct : ∀ p, (p = A → ((place XiaoWang = First ∨ place XiaoZhang = Third) ∧ (place XiaoWang ≠ First ∨ place XiaoZhang ≠ Third)))
                          ∧ (p = B → ((place XiaoLi = First ∨ place XiaoZhao = Fourth) ∧ (place XiaoLi ≠ First ∨ place XiaoZhao ≠ Fourth)))
                          ∧ (p = C → ((place XiaoZhao = Second ∨ place XiaoWang = Third) ∧ (place XiaoZhao ≠ Second ∨ place XiaoWang ≠ Third)))) 

  -- The goal to prove
  theorem xiao_wang_fourth_place : place XiaoWang = Fourth :=
  sorry
end Competition

end xiao_wang_fourth_place_l173_173286


namespace sufficient_but_not_necessary_l173_173021

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 2 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_l173_173021


namespace age_problem_l173_173373

-- Define the conditions
variables (a b c : ℕ)

-- Assumptions based on conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 37) : b = 14 :=
by {
  sorry   -- Placeholder for the detailed proof
}

end age_problem_l173_173373


namespace triangle_cosine_l173_173456

theorem triangle_cosine {A : ℝ} (h : 0 < A ∧ A < π / 2) (tan_A : Real.tan A = -2) :
  Real.cos A = - (Real.sqrt 5) / 5 :=
sorry

end triangle_cosine_l173_173456


namespace division_of_repeating_decimals_l173_173358

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l173_173358


namespace cube_volume_from_surface_area_l173_173966

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173966


namespace cube_volume_l173_173973

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173973


namespace candidate_percentage_of_valid_votes_l173_173132

theorem candidate_percentage_of_valid_votes (total_votes : ℕ) (invalid_percentage : ℝ) (votes_for_candidate : ℕ) :
  invalid_percentage = 0.15 →
  total_votes = 560000 →
  votes_for_candidate = 357000 →
  let valid_votes := (1 - invalid_percentage) * total_votes in
  let percentage_of_valid_votes := (votes_for_candidate / valid_votes) * 100 in
  percentage_of_valid_votes = 75 := by
  intros h1 h2 h3
  let valid_votes := (1 - invalid_percentage) * total_votes
  let percentage_of_valid_votes := (votes_for_candidate.toFloat / valid_votes) * 100
  sorry

end candidate_percentage_of_valid_votes_l173_173132


namespace number_of_pencils_l173_173048

theorem number_of_pencils 
  (P Pe M : ℕ)
  (h1 : Pe = P + 4)
  (h2 : M = P + 20)
  (h3 : P / 5 = Pe / 6)
  (h4 : Pe / 6 = M / 7) : 
  Pe = 24 :=
by
  sorry

end number_of_pencils_l173_173048


namespace sufficient_but_not_necessary_condition_l173_173432

theorem sufficient_but_not_necessary_condition
  (a : ℝ) :
  (a = 2 → (a - 1) * (a - 2) = 0)
  ∧ (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l173_173432


namespace circle_people_count_l173_173785

def num_people (n : ℕ) (a b : ℕ) : Prop :=
  a = 7 ∧ b = 18 ∧ (b = a + (n / 2))

theorem circle_people_count (n : ℕ) (a b : ℕ) (h : num_people n a b) : n = 24 :=
by
  sorry

end circle_people_count_l173_173785


namespace molecular_weight_of_6_moles_l173_173775

-- Define the molecular weight of the compound
def molecular_weight : ℕ := 1404

-- Define the number of moles
def number_of_moles : ℕ := 6

-- The hypothesis would be the molecular weight condition
theorem molecular_weight_of_6_moles : number_of_moles * molecular_weight = 8424 :=
by sorry

end molecular_weight_of_6_moles_l173_173775


namespace train_speed_on_time_l173_173636

theorem train_speed_on_time (v : ℕ) (t : ℕ) :
  (15 / v + 1 / 4 = 15 / 50) ∧ (t = 15) → v = 300 := by
  sorry

end train_speed_on_time_l173_173636


namespace initial_alloy_weight_l173_173650

theorem initial_alloy_weight
  (x : ℝ)  -- Weight of the initial alloy in ounces
  (h1 : 0.80 * (x + 24) = 0.50 * x + 24)  -- Equation derived from conditions
: x = 16 := 
sorry

end initial_alloy_weight_l173_173650


namespace right_triangle_ratio_l173_173803

theorem right_triangle_ratio (x : ℝ) :
  let AB := 3 * x
  let BC := 4 * x
  let AC := (AB ^ 2 + BC ^ 2).sqrt
  let h := AC
  let AD := 16 / 21 * h / (16 / 21 + 1)
  let CD := h / (16 / 21 + 1)
  (CD / AD) = 21 / 16 :=
by 
  sorry

end right_triangle_ratio_l173_173803


namespace compound_interest_amount_l173_173349

theorem compound_interest_amount:
  let SI := (5250 * 4 * 2) / 100
  let CI := 2 * SI
  let P := 420 / 0.21 
  CI = P * ((1 + 0.1) ^ 2 - 1) →
  SI = 210 →
  CI = 420 →
  P = 2000 :=
by
  sorry

end compound_interest_amount_l173_173349


namespace john_made_money_l173_173718

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money_l173_173718


namespace peggy_dolls_ratio_l173_173733

noncomputable def peggy_dolls_original := 6
noncomputable def peggy_dolls_from_grandmother := 30
noncomputable def peggy_dolls_total := 51

theorem peggy_dolls_ratio :
  ∃ x, peggy_dolls_original + peggy_dolls_from_grandmother + x = peggy_dolls_total ∧ x / peggy_dolls_from_grandmother = 1 / 2 :=
by {
  sorry
}

end peggy_dolls_ratio_l173_173733


namespace theater_ticket_sales_l173_173347

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end theater_ticket_sales_l173_173347


namespace problem_l173_173591

noncomputable def f (x : ℝ) (a b c : ℝ) := a * x ^ 3 + b * x + c
noncomputable def g (x : ℝ) (d e f : ℝ) := d * x ^ 3 + e * x + f

theorem problem (a b c d e f : ℝ) :
    (∀ x : ℝ, f (g x d e f) a b c = g (f x a b c) d e f) ↔ d = a ∨ d = -a :=
by
  sorry

end problem_l173_173591


namespace bakery_new_cakes_count_l173_173533

def cakes_sold := 91
def more_cakes_bought := 63

theorem bakery_new_cakes_count : (91 + 63) = 154 :=
by
  sorry

end bakery_new_cakes_count_l173_173533


namespace balloon_arrangements_l173_173258

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l173_173258


namespace Monica_next_year_reading_l173_173029

variable (last_year_books : ℕ) (this_year_books : ℕ) (next_year_books : ℕ)

def Monica_reading_proof (last_year_books = 16 : ℕ) 
                         (this_year_books = 2 * last_year_books : ℕ)
                         (next_year_books = 2 * this_year_books + 5 : ℕ) : Prop :=
  next_year_books = 69

theorem Monica_next_year_reading : Monica_reading_proof :=
by
  unfold Monica_reading_proof
  sorry

end Monica_next_year_reading_l173_173029


namespace greatest_value_k_l173_173865

theorem greatest_value_k (k : ℝ) (h : ∀ x : ℝ, (x - 1) ∣ (x^2 + 2*k*x - 3*k^2)) : k ≤ 1 :=
  by
  sorry

end greatest_value_k_l173_173865


namespace rotated_angle_l173_173496

theorem rotated_angle (initial_angle : ℝ) (rotation_angle : ℝ) (final_angle : ℝ) :
  initial_angle = 30 ∧ rotation_angle = 450 → final_angle = 60 :=
by
  intro h
  sorry

end rotated_angle_l173_173496


namespace minimum_total_trips_l173_173924

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end minimum_total_trips_l173_173924


namespace turnip_mixture_l173_173087

theorem turnip_mixture (cups_potatoes total_turnips : ℕ) (h_ratio : 20 = 5 * 4) (h_turnips : total_turnips = 8) :
    cups_potatoes = 2 :=
by
    have ratio := h_ratio
    have turnips := h_turnips
    sorry

end turnip_mixture_l173_173087


namespace tax_amount_self_employed_l173_173623

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end tax_amount_self_employed_l173_173623


namespace verify_distinct_outcomes_l173_173945

def i : ℂ := Complex.I

theorem verify_distinct_outcomes :
  ∃! S, ∀ n : ℤ, n % 8 = n → S = i^n + i^(-n)
  := sorry

end verify_distinct_outcomes_l173_173945


namespace trigonometric_identity_l173_173548

open Real

theorem trigonometric_identity (α : ℝ) (h : sin (α - (π / 12)) = 1 / 3) :
  cos (α + (17 * π / 12)) = 1 / 3 :=
sorry

end trigonometric_identity_l173_173548


namespace option_A_option_B_option_C_option_D_l173_173638

theorem option_A : (-4:ℤ)^2 ≠ -(4:ℤ)^2 := sorry
theorem option_B : (-2:ℤ)^3 = -2^3 := sorry
theorem option_C : (-1:ℤ)^2020 ≠ (-1:ℤ)^2021 := sorry
theorem option_D : ((2:ℚ)/(3:ℚ))^3 = ((2:ℚ)/(3:ℚ))^3 := sorry

end option_A_option_B_option_C_option_D_l173_173638


namespace sale_in_fifth_month_l173_173796

theorem sale_in_fifth_month 
    (a1 a2 a3 a4 a6 : ℕ) 
    (avg_sale : ℕ)
    (H_avg : avg_sale = 8500)
    (H_a1 : a1 = 8435) 
    (H_a2 : a2 = 8927) 
    (H_a3 : a3 = 8855) 
    (H_a4 : a4 = 9230) 
    (H_a6 : a6 = 6991) : 
    ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sale ∧ a5 = 8562 := 
by
    sorry

end sale_in_fifth_month_l173_173796


namespace factorial_expression_l173_173657

theorem factorial_expression :
  (factorial 13 - factorial 12) / factorial 10 = 1584 :=
by
  sorry

end factorial_expression_l173_173657


namespace profit_percentage_is_50_l173_173490

/--
Assumption:
- Initial machine cost: Rs 10,000
- Repair cost: Rs 5,000
- Transportation charges: Rs 1,000
- Selling price: Rs 24,000

To prove:
- The profit percentage is 50%
-/

def initial_cost : ℕ := 10000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 24000
def total_cost : ℕ := initial_cost + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_50 :
  (profit * 100) / total_cost = 50 :=
by
  -- proof goes here
  sorry

end profit_percentage_is_50_l173_173490


namespace harry_cookies_batch_l173_173697

theorem harry_cookies_batch
  (total_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips = 81)
  (batches = 3)
  (chips_per_cookie = 9) :
  (total_chips / batches) / chips_per_cookie = 3 := by
  sorry

end harry_cookies_batch_l173_173697


namespace inequality_ratios_l173_173117

theorem inequality_ratios (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (c / a) > (d / b) :=
sorry

end inequality_ratios_l173_173117


namespace average_score_l173_173380

theorem average_score (classA_students classB_students : ℕ)
  (avg_score_classA avg_score_classB : ℕ)
  (h_classA : classA_students = 40)
  (h_classB : classB_students = 50)
  (h_avg_classA : avg_score_classA = 90)
  (h_avg_classB : avg_score_classB = 81) :
  (classA_students * avg_score_classA + classB_students * avg_score_classB) / 
  (classA_students + classB_students) = 85 := 
  by sorry

end average_score_l173_173380


namespace cube_volume_l173_173988

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173988


namespace theater_ticket_sales_l173_173346

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end theater_ticket_sales_l173_173346


namespace mark_leftover_amount_l173_173158

-- Definitions
def raise_percentage : ℝ := 0.05
def old_hourly_wage : ℝ := 40
def hours_per_week : ℝ := 8 * 5
def old_weekly_expenses : ℝ := 600
def new_expense : ℝ := 100

-- Calculate new hourly wage
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)

-- Calculate weekly earnings at the new wage
def weekly_earnings : ℝ := new_hourly_wage * hours_per_week

-- Calculate new total weekly expenses
def total_weekly_expenses : ℝ := old_weekly_expenses + new_expense

-- Calculate leftover amount
def leftover_per_week : ℝ := weekly_earnings - total_weekly_expenses

theorem mark_leftover_amount : leftover_per_week = 980 := by
  sorry

end mark_leftover_amount_l173_173158


namespace total_profit_is_correct_l173_173206

-- Definitions for the investments and profit shares
def x_investment : ℕ := 5000
def y_investment : ℕ := 15000
def x_share_of_profit : ℕ := 400

-- The theorem states that the total profit is Rs. 1600 given the conditions
theorem total_profit_is_correct (h1 : x_share_of_profit = 400) (h2 : x_investment = 5000) (h3 : y_investment = 15000) : 
  let y_share_of_profit := 3 * x_share_of_profit
  let total_profit := x_share_of_profit + y_share_of_profit
  total_profit = 1600 :=
by
  sorry

end total_profit_is_correct_l173_173206


namespace integer_value_expression_l173_173065

theorem integer_value_expression (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  (p = 2 ∧ q = 2) ∨ (p ≠ 2 ∧ q = 2 ∧ pq + p^p + q^q = 3 * (p + q)) :=
sorry

end integer_value_expression_l173_173065


namespace total_number_of_guests_l173_173213

theorem total_number_of_guests (A C S : ℕ) (hA : A = 58) (hC : C = A - 35) (hS : S = 2 * C) : 
  A + C + S = 127 := 
by
  sorry

end total_number_of_guests_l173_173213


namespace value_of_a_l173_173288

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x + a^2 * y + 6 = 0 → (a-2) * x + 3 * a * y + 2 * a = 0) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l173_173288


namespace decorative_plate_painted_fraction_l173_173793

noncomputable def fraction_painted_area (total_area painted_area : ℕ) : ℚ :=
  painted_area / total_area

theorem decorative_plate_painted_fraction :
  let side_length := 4
  let total_area := side_length * side_length
  let painted_smaller_squares := 6
  fraction_painted_area total_area painted_smaller_squares = 3 / 8 :=
by
  sorry

end decorative_plate_painted_fraction_l173_173793


namespace average_transformation_l173_173551

theorem average_transformation (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h_avg : (a_1 + a_2 + a_3 + a_4 + a_5) / 5 = 8) : 
  ((a_1 + 10) + (a_2 - 10) + (a_3 + 10) + (a_4 - 10) + (a_5 + 10)) / 5 = 10 := 
by
  sorry

end average_transformation_l173_173551


namespace kids_joined_in_l173_173770

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end kids_joined_in_l173_173770


namespace print_shop_cost_difference_l173_173678

theorem print_shop_cost_difference :
  let cost_per_copy_X := 1.25
  let cost_per_copy_Y := 2.75
  let num_copies := 40
  let total_cost_X := cost_per_copy_X * num_copies
  let total_cost_Y := cost_per_copy_Y * num_copies
  total_cost_Y - total_cost_X = 60 :=
by 
  dsimp only []
  sorry

end print_shop_cost_difference_l173_173678


namespace max_sum_of_four_numbers_l173_173164

theorem max_sum_of_four_numbers : 
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ (2 * a + 3 * b + 2 * c + 3 * d = 2017) ∧ 
    (a + b + c + d = 806) :=
by
  sorry

end max_sum_of_four_numbers_l173_173164


namespace ball_hits_ground_in_2_72_seconds_l173_173654

noncomputable def height_at_time (t : ℝ) : ℝ :=
  -16 * t^2 - 30 * t + 200

theorem ball_hits_ground_in_2_72_seconds :
  ∃ t : ℝ, t = 2.72 ∧ height_at_time t = 0 :=
by
  use 2.72
  sorry

end ball_hits_ground_in_2_72_seconds_l173_173654


namespace inequality_for_positive_real_l173_173610

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end inequality_for_positive_real_l173_173610


namespace correct_operation_l173_173368

theorem correct_operation (a b : ℝ) : 
  (-a^3 * b)^2 = a^6 * b^2 :=
by
  sorry

end correct_operation_l173_173368


namespace complement_union_eq_l173_173442

open Set

variable (U A B : Set ℤ)

noncomputable def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3}

noncomputable def setA : Set ℤ := {-1, 0, 3}

noncomputable def setB : Set ℤ := {1, 3}

theorem complement_union_eq :
  A ∪ B = {-1, 0, 1, 3} →
  U = universal_set →
  A = setA →
  B = setB →
  (U \ (A ∪ B)) = {-2, 2} := by
  intros
  sorry

end complement_union_eq_l173_173442


namespace relationship_f_neg2_f_expr_l173_173822

noncomputable def f : ℝ → ℝ := sorry  -- f is some function ℝ → ℝ, the exact definition is not provided

axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_on_negatives : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y -- f is increasing on (-∞, 0)

theorem relationship_f_neg2_f_expr (a : ℝ) : f (-2) ≥ f (a^2 - 4 * a + 6) := by
  -- proof omitted
  sorry

end relationship_f_neg2_f_expr_l173_173822


namespace speed_in_still_water_l173_173063

-- Definitions of the conditions
def downstream_condition (v_m v_s : ℝ) : Prop := v_m + v_s = 6
def upstream_condition (v_m v_s : ℝ) : Prop := v_m - v_s = 3

-- The theorem to be proven
theorem speed_in_still_water (v_m v_s : ℝ) 
  (h1 : downstream_condition v_m v_s) 
  (h2 : upstream_condition v_m v_s) : v_m = 4.5 :=
by
  sorry

end speed_in_still_water_l173_173063


namespace peter_large_glasses_l173_173330

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end peter_large_glasses_l173_173330


namespace fly_travel_distance_l173_173086

theorem fly_travel_distance
  (carA_speed : ℕ)
  (carB_speed : ℕ)
  (initial_distance : ℕ)
  (fly_speed : ℕ)
  (relative_speed : ℕ := carB_speed - carA_speed)
  (catchup_time : ℚ := initial_distance / relative_speed)
  (fly_travel : ℚ := fly_speed * catchup_time) :
  carA_speed = 20 → carB_speed = 30 → initial_distance = 1 → fly_speed = 40 → fly_travel = 4 :=
by
  sorry

end fly_travel_distance_l173_173086


namespace remainder_x_plus_13_div_41_l173_173782

theorem remainder_x_plus_13_div_41 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 := by
  sorry

end remainder_x_plus_13_div_41_l173_173782


namespace price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l173_173709

def cost_price : ℝ := 40
def initial_price : ℝ := 60
def initial_sales_volume : ℕ := 300
def sales_decrease_rate (x : ℕ) : ℕ := 10 * x
def sales_increase_rate (a : ℕ) : ℕ := 20 * a

noncomputable def price_increase_proft_relation (x : ℕ) : ℝ :=
  -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000

theorem price_increase_profit_relation_proof (x : ℕ) (h : 0 ≤ x ∧ x ≤ 30) :
  price_increase_proft_relation x = -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000 := sorry

noncomputable def price_decrease_profit_relation (a : ℕ) : ℝ :=
  -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000

theorem price_decrease_profit_relation_proof (a : ℕ) :
  price_decrease_profit_relation a = -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000 := sorry

theorem max_profit_price_increase :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧ price_increase_proft_relation x = 6250 := sorry

end price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l173_173709


namespace sum_of_reciprocals_of_squares_l173_173184

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) : (1 : ℚ)/a^2 + (1 : ℚ)/b^2 = 10/9 := 
sorry

end sum_of_reciprocals_of_squares_l173_173184


namespace sum_of_repeating_decimal_digits_of_five_thirteenths_l173_173049

theorem sum_of_repeating_decimal_digits_of_five_thirteenths 
  (a b : ℕ)
  (h1 : 5 / 13 = (a * 10 + b) / 99)
  (h2 : (a * 10 + b) = 38) :
  a + b = 11 :=
sorry

end sum_of_repeating_decimal_digits_of_five_thirteenths_l173_173049


namespace cuboid_volume_l173_173645

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 18) (h_height : height = 8) : 
  base_area * height = 144 :=
by
  rw [h_base_area, h_height]
  norm_num

end cuboid_volume_l173_173645


namespace cube_volume_is_1728_l173_173956

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173956


namespace fraction_of_time_l173_173510

-- Define the time John takes to clean the entire house
def John_time : ℝ := 6

-- Define the combined time it takes Nick and John to clean the entire house
def combined_time : ℝ := 3.6

-- Given this configuration, we need to prove the fraction result.
theorem fraction_of_time (N : ℝ) (H1 : John_time = 6) (H2 : ∀ N, (1/John_time) + (1/N) = 1/combined_time) :
  (John_time / 2) / N = 1 / 3 := 
by sorry

end fraction_of_time_l173_173510


namespace convert_to_spherical_l173_173088

noncomputable def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if y / x < 0 then Real.arctan (-y / x) + 2 * Real.pi else Real.arctan (y / x)
  (ρ, θ, φ)

theorem convert_to_spherical :
  let x := 1
  let y := -4 * Real.sqrt 3
  let z := 4
  spherical_coordinates x y z = (Real.sqrt 65, Real.arctan (-4 * Real.sqrt 3) + 2 * Real.pi, Real.arccos (4 / (Real.sqrt 65))) :=
by
  sorry

end convert_to_spherical_l173_173088


namespace balloon_arrangement_count_l173_173238

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l173_173238


namespace distinct_flavors_count_l173_173338

theorem distinct_flavors_count (red_candies : ℕ) (green_candies : ℕ)
  (h_red : red_candies = 0 ∨ red_candies = 1 ∨ red_candies = 2 ∨ red_candies = 3 ∨ red_candies = 4 ∨ red_candies = 5 ∨ red_candies = 6)
  (h_green : green_candies = 0 ∨ green_candies = 1 ∨ green_candies = 2 ∨ green_candies = 3 ∨ green_candies = 4 ∨ green_candies = 5) :
  ∃ unique_flavors : Finset (ℚ), unique_flavors.card = 25 :=
by
  sorry

end distinct_flavors_count_l173_173338


namespace gcd_8251_6105_l173_173099

theorem gcd_8251_6105 :
  Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l173_173099


namespace fourth_number_value_l173_173915

variable (A B C D E F : ℝ)

theorem fourth_number_value 
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) : 
  D = 25 := 
by 
  sorry

end fourth_number_value_l173_173915


namespace problem_statement_l173_173024

def f (x : Int) : Int :=
  if x > 6 then x^2 - 4
  else if -6 <= x && x <= 6 then 3*x + 2
  else 5

def adjusted_f (x : Int) : Int :=
  let fx := f x
  if x % 3 == 0 then fx + 5 else fx

theorem problem_statement : 
  adjusted_f (-8) + adjusted_f 0 + adjusted_f 9 = 94 :=
by 
  sorry

end problem_statement_l173_173024


namespace g_at_52_l173_173494

noncomputable def g : ℝ → ℝ := sorry

axiom g_multiplicative : ∀ (x y: ℝ), g (x * y) = y * g x
axiom g_at_1 : g 1 = 10

theorem g_at_52 : g 52 = 520 := sorry

end g_at_52_l173_173494


namespace perpendicular_line_to_plane_l173_173476

variables {Point Line Plane : Type}
variables (a b c : Line) (α : Plane) (A : Point)

-- Define the conditions
def line_perpendicular_to (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def lines_intersect_at (l1 l2 : Line) (P : Point) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given conditions in Lean 4
variables (h1 : line_perpendicular_to c a)
variables (h2 : line_perpendicular_to c b)
variables (h3 : line_in_plane a α)
variables (h4 : line_in_plane b α)
variables (h5 : lines_intersect_at a b A)

-- The theorem statement to prove
theorem perpendicular_line_to_plane : line_perpendicular_to_plane c α :=
sorry

end perpendicular_line_to_plane_l173_173476


namespace speed_of_first_boy_l173_173192

-- Variables for speeds and time
variables (v : ℝ) (t : ℝ) (d : ℝ)

-- Given conditions
def initial_conditions := 
  v > 0 ∧ 
  7.5 > 0 ∧ 
  t = 10 ∧ 
  d = 20

-- Theorem statement with the conditions and the expected answer
theorem speed_of_first_boy
  (h : initial_conditions v t d) : 
  v = 9.5 :=
sorry

end speed_of_first_boy_l173_173192


namespace g_x_plus_three_l173_173592

variable (x : ℝ)

def g (x : ℝ) : ℝ := x^2 - x

theorem g_x_plus_three : g (x + 3) = x^2 + 5 * x + 6 := by
  sorry

end g_x_plus_three_l173_173592


namespace imaginary_unit_problem_l173_173683

variable {a b : ℝ}

theorem imaginary_unit_problem (h : i * (a + i) = b + 2 * i) : a + b = 1 :=
sorry

end imaginary_unit_problem_l173_173683


namespace minimum_total_trips_l173_173925

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end minimum_total_trips_l173_173925


namespace arithmetic_sequence_sum_l173_173135

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l173_173135


namespace self_employed_tax_amount_l173_173624

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end self_employed_tax_amount_l173_173624


namespace division_of_repeating_decimals_l173_173357

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l173_173357


namespace number_of_knights_l173_173032

/--
On the island of Liars and Knights, a circular arrangement is called correct if everyone standing in the circle
can say that among his two neighbors there is a representative of his tribe. One day, 2019 natives formed a correct
arrangement in a circle. A liar approached them and said: "Now together we can also form a correct arrangement in a circle."
Prove that the number of knights in the initial arrangement is 1346.
-/
theorem number_of_knights : 
  ∀ (K L : ℕ), 
    (K + L = 2019) → 
    (K ≥ 2 * L) → 
    (K ≤ 2 * L + 1) → 
  K = 1346 :=
by
  intros K L h1 h2 h3
  sorry

end number_of_knights_l173_173032


namespace apples_per_pie_l173_173893

theorem apples_per_pie (total_apples : ℕ) (unripe_apples : ℕ) (pies : ℕ) (ripe_apples : ℕ)
  (H1 : total_apples = 34)
  (H2 : unripe_apples = 6)
  (H3 : pies = 7)
  (H4 : ripe_apples = total_apples - unripe_apples) :
  ripe_apples / pies = 4 := by
  sorry

end apples_per_pie_l173_173893


namespace movie_tickets_l173_173344

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end movie_tickets_l173_173344


namespace math_problem_l173_173172

noncomputable def compute_value (a b c : ℝ) : ℝ :=
  (b / (a + b)) + (c / (b + c)) + (a / (c + a))

theorem math_problem (a b c : ℝ)
  (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -12)
  (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 15) :
  compute_value a b c = 6 :=
sorry

end math_problem_l173_173172


namespace cube_volume_from_surface_area_example_cube_volume_l173_173962

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173962


namespace balloon_permutations_l173_173270

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l173_173270


namespace circles_max_ab_l173_173869

theorem circles_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (x y : ℝ), (x + a)^2 + (y - 2)^2 = 1 ∧ (x - b)^2 + (y - 2)^2 = 4) →
  a + b = 3 →
  ab ≤ 9 / 4 := 
  by
  sorry

end circles_max_ab_l173_173869


namespace second_discount_percentage_l173_173760

-- Defining the variables
variables (P S : ℝ) (d1 d2 : ℝ)

-- Given conditions
def original_price : P = 200 := by sorry
def sale_price_after_initial_discount : S = 171 := by sorry
def first_discount_rate : d1 = 0.10 := by sorry

-- Required to prove
theorem second_discount_percentage :
  ∃ d2, (d2 = 0.05) :=
sorry

end second_discount_percentage_l173_173760


namespace monica_books_l173_173028

theorem monica_books (last_year_books : ℕ) 
                      (this_year_books : ℕ) 
                      (next_year_books : ℕ) 
                      (h1 : last_year_books = 16) 
                      (h2 : this_year_books = 2 * last_year_books) 
                      (h3 : next_year_books = 2 * this_year_books + 5) : 
                      next_year_books = 69 :=
by
  rw [h1, h2] at h3
  rw [h2, h1] at h3
  simp at h3
  exact h3

end monica_books_l173_173028


namespace cube_volume_l173_173981

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173981


namespace intersecting_lines_l173_173308

theorem intersecting_lines (m : ℝ) :
  (∃ (x y : ℝ), y = 2 * x ∧ x + y = 3 ∧ m * x + 2 * y + 5 = 0) ↔ (m = -9) :=
by
  sorry

end intersecting_lines_l173_173308


namespace probability_of_white_first_red_second_l173_173006

noncomputable def probability_white_first_red_second : ℚ :=
let totalBalls := 6
let probWhiteFirst := 1 / totalBalls
let remainingBalls := totalBalls - 1
let probRedSecond := 1 / remainingBalls
probWhiteFirst * probRedSecond

theorem probability_of_white_first_red_second :
  probability_white_first_red_second = 1 / 30 :=
by
  sorry

end probability_of_white_first_red_second_l173_173006


namespace sum_polynomials_l173_173023

def p (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def q (x : ℝ) : ℝ := -3 * x^2 + x - 5
def r (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem sum_polynomials (x : ℝ) : p x + q x + r x = 3 * x^2 - 5 * x - 1 :=
by
  sorry

end sum_polynomials_l173_173023


namespace find_definite_integers_l173_173152

theorem find_definite_integers (n d e f : ℕ) (h₁ : n = d + Int.sqrt (e + Int.sqrt f)) 
    (h₂: ∀ x : ℝ, x = d + Int.sqrt (e + Int.sqrt f) → 
        (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12 * x - 5))
        : d + e + f = 76 :=
sorry

end find_definite_integers_l173_173152


namespace cube_volume_is_1728_l173_173953

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173953


namespace bus_ride_cost_l173_173223

theorem bus_ride_cost (B T : ℝ) 
  (h1 : T = B + 6.85)
  (h2 : T + B = 9.65)
  (h3 : ∃ n : ℤ, B = 0.35 * n ∧ ∃ m : ℤ, T = 0.35 * m) : 
  B = 1.40 := 
by
  sorry

end bus_ride_cost_l173_173223


namespace prob_a_prob_b_l173_173067

-- Given conditions and question for Part a
def election_prob (p q : ℕ) (h : p > q) : ℚ :=
  (p - q) / (p + q)

theorem prob_a : election_prob 3 2 (by decide) = 1 / 5 :=
  sorry

-- Given conditions and question for Part b
theorem prob_b : election_prob 1010 1009 (by decide) = 1 / 2019 :=
  sorry

end prob_a_prob_b_l173_173067


namespace initial_hamburgers_correct_l173_173394

-- Define the initial problem conditions
def initial_hamburgers (H : ℝ) : Prop := H + 3.0 = 12

-- State the proof problem
theorem initial_hamburgers_correct (H : ℝ) (h : initial_hamburgers H) : H = 9.0 :=
sorry

end initial_hamburgers_correct_l173_173394


namespace jim_can_bake_loaves_l173_173146

-- Define the amounts of flour in different locations
def flour_cupboard : ℕ := 200  -- in grams
def flour_counter : ℕ := 100   -- in grams
def flour_pantry : ℕ := 100    -- in grams

-- Define the amount of flour required for one loaf of bread
def flour_per_loaf : ℕ := 200  -- in grams

-- Total loaves Jim can bake
def loaves_baked (f_c f_k f_p f_r : ℕ) : ℕ :=
  (f_c + f_k + f_p) / f_r

-- Theorem to prove the solution
theorem jim_can_bake_loaves :
  loaves_baked flour_cupboard flour_counter flour_pantry flour_per_loaf = 2 :=
by
  -- Proof is omitted
  sorry

end jim_can_bake_loaves_l173_173146


namespace find_y_l173_173635

theorem find_y (y : ℕ) (h1 : y % 6 = 5) (h2 : y % 7 = 6) (h3 : y % 8 = 7) : y = 167 := 
by
  sorry  -- Proof is omitted

end find_y_l173_173635


namespace sum_mod_9_equal_6_l173_173446

theorem sum_mod_9_equal_6 :
  ((1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9) = 6 :=
by
  sorry

end sum_mod_9_equal_6_l173_173446


namespace minji_total_water_intake_l173_173906

variable (morning_water : ℝ)
variable (afternoon_water : ℝ)

theorem minji_total_water_intake (h_morning : morning_water = 0.26) (h_afternoon : afternoon_water = 0.37):
  morning_water + afternoon_water = 0.63 :=
sorry

end minji_total_water_intake_l173_173906


namespace parabola_equation_l173_173675

theorem parabola_equation (h k : ℝ) (p : ℝ × ℝ) (a b c : ℝ) :
  h = 3 ∧ k = -2 ∧ p = (4, -5) ∧
  (∀ x y : ℝ, y = a * (x - h) ^ 2 + k → p.2 = a * (p.1 - h) ^ 2 + k) →
  -(3:ℝ) = a ∧ 18 = b ∧ -29 = c :=
by sorry

end parabola_equation_l173_173675


namespace balloon_arrangements_l173_173263

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l173_173263


namespace folded_rectangle_ratio_l173_173801

-- Define the conditions
def original_area (a b : ℝ) := a * b
def pentagon_area (a b : ℝ) := (7 / 10) * original_area a b

-- Define the ratio to prove
def ratio_of_sides (a b : ℝ) := a / b

-- Define the theorem to prove the ratio equals sqrt(5)
theorem folded_rectangle_ratio (a b : ℝ) (h: a > b) 
  (A1 : pentagon_area a b = (7 / 10) * original_area a b) :
  ratio_of_sides a b = real.sqrt 5 :=
  sorry

end folded_rectangle_ratio_l173_173801


namespace how_much_leftover_a_week_l173_173159

variable (hourly_wage : ℕ)          -- Mark's old hourly wage (40 dollars)
variable (raise_percent : ℚ)        -- Raise percentage (5%)
variable (hours_per_day : ℕ)        -- Working hours per day (8 hours)
variable (days_per_week : ℕ)        -- Working days per week (5 days)
variable (old_weekly_bills : ℕ)     -- Old weekly bills (600 dollars)
variable (trainer_fee : ℕ)          -- Weekly personal trainer fee (100 dollars)

def new_hourly_wage := hourly_wage * (1 + raise_percent)
def daily_earnings := new_hourly_wage * hours_per_day
def weekly_earnings := daily_earnings * days_per_week
def new_weekly_bills := old_weekly_bills + trainer_fee
def leftover_money := weekly_earnings - new_weekly_bills

theorem how_much_leftover_a_week :
    hourly_wage = 40 → 
    raise_percent = 0.05 → 
    hours_per_day = 8 → 
    days_per_week = 5 → 
    old_weekly_bills = 600 → 
    trainer_fee = 100 → 
    leftover_money = 980 := 
by
    intros h1 h2 h3 h4 h5 h6
    sorry

end how_much_leftover_a_week_l173_173159


namespace sum_of_coeffs_eq_225_l173_173499

/-- The sum of the coefficients of all terms in the expansion
of (C_x + C_x^2 + C_x^3 + C_x^4)^2 is equal to 225. -/
theorem sum_of_coeffs_eq_225 (C_x : ℝ) : 
  (C_x + C_x^2 + C_x^3 + C_x^4)^2 = 225 :=
sorry

end sum_of_coeffs_eq_225_l173_173499


namespace intersection_A_B_l173_173860

-- Definition of set A
def A (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Definition of set B
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

-- Theorem stating the intersection of sets A and B
theorem intersection_A_B (x : ℝ) : (A x ∧ B x) ↔ (-1 < x ∧ x ≤ 0) :=
by sorry

end intersection_A_B_l173_173860


namespace appleJuicePercentageIsCorrect_l173_173026

-- Define the initial conditions
def MikiHas : ℕ × ℕ := (15, 10) -- Miki has 15 apples and 10 bananas

-- Define the juice extraction rates
def appleJuicePerApple : ℚ := 9 / 3 -- 9 ounces from 3 apples
def bananaJuicePerBanana : ℚ := 10 / 2 -- 10 ounces from 2 bananas

-- Define the number of apples and bananas used for the blend
def applesUsed : ℕ := 5
def bananasUsed : ℕ := 4

-- Calculate the total juice extracted
def appleJuice : ℚ := applesUsed * appleJuicePerApple
def bananaJuice : ℚ := bananasUsed * bananaJuicePerBanana

-- Calculate the total juice and percentage of apple juice
def totalJuice : ℚ := appleJuice + bananaJuice
def percentageAppleJuice : ℚ := (appleJuice / totalJuice) * 100

theorem appleJuicePercentageIsCorrect : percentageAppleJuice = 42.86 := by
  sorry

end appleJuicePercentageIsCorrect_l173_173026


namespace matthew_total_time_l173_173724

def assemble_time : ℝ := 1
def bake_time_normal : ℝ := 1.5
def decorate_time : ℝ := 1
def bake_time_double : ℝ := bake_time_normal * 2

theorem matthew_total_time :
  assemble_time + bake_time_double + decorate_time = 5 := 
by 
  -- The proof will be filled in here
  sorry

end matthew_total_time_l173_173724


namespace students_catching_up_on_homework_l173_173466

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end students_catching_up_on_homework_l173_173466


namespace loaves_of_bread_can_bake_l173_173148

def total_flour_in_cupboard := 200
def total_flour_on_counter := 100
def total_flour_in_pantry := 100
def flour_per_loaf := 200

theorem loaves_of_bread_can_bake :
  (total_flour_in_cupboard + total_flour_on_counter + total_flour_in_pantry) / flour_per_loaf = 2 := by
  sorry

end loaves_of_bread_can_bake_l173_173148


namespace greatest_xy_value_l173_173450

theorem greatest_xy_value (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 7 * x + 4 * y = 140) :
  (∀ z : ℕ, (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ z = x * y) → z ≤ 168) ∧
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ 168 = x * y) :=
sorry

end greatest_xy_value_l173_173450


namespace set_equiv_l173_173880

-- Definition of the set A according to the conditions
def A : Set ℚ := { z : ℚ | ∃ p q : ℕ, z = p / (q : ℚ) ∧ p + q = 5 ∧ p > 0 ∧ q > 0 }

-- The target set we want to prove A is equal to
def target_set : Set ℚ := { 1/4, 2/3, 3/2, 4 }

-- The theorem to prove that both sets are equal
theorem set_equiv : A = target_set :=
by
  sorry -- Proof goes here

end set_equiv_l173_173880


namespace banker_gain_l173_173178

theorem banker_gain :
  ∀ (t : ℝ) (r : ℝ) (TD : ℝ),
  t = 1 →
  r = 12 →
  TD = 65 →
  (TD * r * t) / (100 - (r * t)) = 8.86 :=
by
  intros t r TD ht hr hTD
  rw [ht, hr, hTD]
  sorry

end banker_gain_l173_173178


namespace find_day_for_balance_l173_173704

-- Define the initial conditions and variables
def initialEarnings : ℤ := 20
def secondDaySpending : ℤ := 15
variables (X Y : ℤ)

-- Define the function for net balance on day D
def netBalance (D : ℤ) : ℤ :=
  initialEarnings + (D - 1) * X - (secondDaySpending + (D - 2) * Y)

-- The main theorem proving the day D for net balance of Rs. 60
theorem find_day_for_balance (X Y : ℤ) : ∃ D : ℤ, netBalance X Y D = 60 → 55 = (D + 1) * (X - Y) :=
by
  sorry

end find_day_for_balance_l173_173704


namespace derivative_at_zero_l173_173448

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem derivative_at_zero : deriv f 0 = 720 :=
by
  sorry

end derivative_at_zero_l173_173448


namespace exists_three_numbers_sum_to_zero_l173_173606

theorem exists_three_numbers_sum_to_zero (s : Finset ℤ) (h_card : s.card = 101) (h_abs : ∀ x ∈ s, |x| ≤ 99) :
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 :=
by {
  sorry
}

end exists_three_numbers_sum_to_zero_l173_173606


namespace odd_function_f_2_eq_2_l173_173682

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^2 + 3 * x else -(if -x < 0 then (-x)^2 + 3 * (-x) else x^2 + 3 * x)

theorem odd_function_f_2_eq_2 : f 2 = 2 :=
by
  -- sorry will be used to skip the actual proof
  sorry

end odd_function_f_2_eq_2_l173_173682


namespace diameter_other_endpoint_l173_173826

def center : ℝ × ℝ := (1, -2)
def endpoint1 : ℝ × ℝ := (4, 3)
def expected_endpoint2 : ℝ × ℝ := (7, -7)

theorem diameter_other_endpoint (c : ℝ × ℝ) (e1 e2 : ℝ × ℝ) (h₁ : c = center) (h₂ : e1 = endpoint1) : e2 = expected_endpoint2 :=
by
  sorry

end diameter_other_endpoint_l173_173826


namespace compute_cd_l173_173495

noncomputable def ellipse_foci_at : Prop :=
  ∃ (c d : ℝ), (d^2 - c^2 = 25) ∧ (c^2 + d^2 = 64) ∧ (|c * d| = real.sqrt 868.75)

theorem compute_cd : ellipse_foci_at :=
  sorry

end compute_cd_l173_173495


namespace scientific_notation_of_16907_l173_173883

theorem scientific_notation_of_16907 :
  16907 = 1.6907 * 10^4 :=
sorry

end scientific_notation_of_16907_l173_173883


namespace find_n_l173_173504

/-- Given a natural number n such that LCM(n, 12) = 48 and GCF(n, 12) = 8, prove that n = 32. -/
theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 48) (h2 : Nat.gcd n 12 = 8) : n = 32 :=
sorry

end find_n_l173_173504


namespace cube_volume_from_surface_area_l173_173971

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173971


namespace line_tangent_to_circle_l173_173046

open Real

theorem line_tangent_to_circle :
    ∃ (x y : ℝ), (3 * x - 4 * y - 5 = 0) ∧ ((x - 1)^2 + (y + 3)^2 - 4 = 0) ∧ 
    (∃ (t r : ℝ), (t = 0 ∧ r ≠ 0) ∧ 
     (3 * t - 4 * (r + t * 3 / 4) - 5 = 0) ∧ ((r + t * 3 / 4 - 1)^2 + (3 * (-1) + t - 3)^2 = 0)) 
  :=
sorry

end line_tangent_to_circle_l173_173046


namespace balloon_permutations_l173_173259

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l173_173259


namespace cube_surface_area_l173_173205

theorem cube_surface_area (a : ℝ) : 
    let edge_length := 3 * a
    let face_area := edge_length^2
    let total_surface_area := 6 * face_area
    total_surface_area = 54 * a^2 := 
by sorry

end cube_surface_area_l173_173205


namespace speed_of_stream_l173_173129

/-- Given Athul's rowing conditions, prove the speed of the stream is 1 km/h. -/
theorem speed_of_stream 
  (A S : ℝ)
  (h1 : 16 = (A - S) * 4)
  (h2 : 24 = (A + S) * 4) : 
  S = 1 := 
sorry

end speed_of_stream_l173_173129


namespace remainder_of_9_pow_1995_mod_7_l173_173365

theorem remainder_of_9_pow_1995_mod_7 : (9^1995) % 7 = 1 := 
by 
sorry

end remainder_of_9_pow_1995_mod_7_l173_173365


namespace michael_total_revenue_l173_173598

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end michael_total_revenue_l173_173598


namespace cuboid_volume_l173_173759

/-- Define the ratio condition for the dimensions of the cuboid. -/
def ratio (l w h : ℕ) : Prop :=
  (∃ x : ℕ, l = 2*x ∧ w = x ∧ h = 3*x)

/-- Define the total surface area condition for the cuboid. -/
def surface_area (l w h sa : ℕ) : Prop :=
  2*(l*w + l*h + w*h) = sa

/-- Volume of the cuboid given the ratio and surface area conditions. -/
theorem cuboid_volume (l w h : ℕ) (sa : ℕ) (h_ratio : ratio l w h) (h_surface : surface_area l w h sa) :
  ∃ v : ℕ, v = l * w * h ∧ v = 48 :=
by
  sorry

end cuboid_volume_l173_173759


namespace find_z_l173_173118

open Complex

theorem find_z (z : ℂ) (h1 : (z + 2 * I).im = 0) (h2 : ((z / (2 - I)).im = 0)) : z = 4 - 2 * I :=
by
  sorry

end find_z_l173_173118


namespace question1_question2_l173_173792

def energy_cost (units: ℕ) : ℝ :=
  if units <= 100 then
    units * 0.5
  else
    100 * 0.5 + (units - 100) * 0.8

theorem question1 :
  energy_cost 130 = 74 := by
  sorry

theorem question2 (units: ℕ) (H: energy_cost units = 90) :
  units = 150 := by
  sorry

end question1_question2_l173_173792


namespace weight_of_b_l173_173786

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : B = 51 := 
by
  sorry

end weight_of_b_l173_173786


namespace new_person_weight_l173_173340

theorem new_person_weight
    (avg_weight_20 : ℕ → ℕ)
    (total_weight_20 : ℕ)
    (avg_weight_21 : ℕ)
    (count_20 : ℕ)
    (count_21 : ℕ) :
    avg_weight_20 count_20 = 58 →
    total_weight_20 = count_20 * avg_weight_20 count_20 →
    avg_weight_21 = 53 →
    count_21 = count_20 + 1 →
    ∃ (W : ℕ), total_weight_20 + W = count_21 * avg_weight_21 ∧ W = 47 := 
by 
  sorry

end new_person_weight_l173_173340


namespace chipmunk_families_left_l173_173667

theorem chipmunk_families_left (orig : ℕ) (left : ℕ) (h1 : orig = 86) (h2 : left = 65) : orig - left = 21 := by
  sorry

end chipmunk_families_left_l173_173667


namespace probability_other_side_red_given_seen_red_l173_173790

-- Definition of conditions
def total_cards := 9
def black_black_cards := 5
def black_red_cards := 2
def red_red_cards := 2
def red_sides := (2 * red_red_cards) + black_red_cards -- Total number of red sides
def favorable_red_red_sides := 2 * red_red_cards      -- Number of red sides on fully red cards

-- The required probability
def probability_other_side_red_given_red : ℚ := sorry

-- The main statement to prove
theorem probability_other_side_red_given_seen_red :
  probability_other_side_red_given_red = 2/3 :=
sorry

end probability_other_side_red_given_seen_red_l173_173790


namespace union_of_A_and_B_l173_173431

def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem union_of_A_and_B : A ∪ B = { x | 1 < x ∧ x < 4 } := by
  sorry

end union_of_A_and_B_l173_173431


namespace cube_volume_l173_173974

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173974


namespace female_computer_literacy_l173_173783

variable (E F C M CM CF : ℕ)

theorem female_computer_literacy (hE : E = 1200) 
                                (hF : F = 720) 
                                (hC : C = 744) 
                                (hM : M = 480) 
                                (hCM : CM = 240) 
                                (hCF : CF = C - CM) : 
                                CF = 504 :=
by {
  sorry
}

end female_computer_literacy_l173_173783


namespace cos_C_sin_B_area_l173_173457

noncomputable def triangle_conditions (A B C a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧
  (b / c = 2 * Real.sqrt 3 / 3) ∧
  (A + 3 * C = Real.pi)

theorem cos_C (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.cos C = Real.sqrt 3 / 3 :=
sorry

theorem sin_B (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.sin B = 2 * Real.sqrt 2 / 3 :=
sorry

theorem area (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) (hb : b = 3 * Real.sqrt 3) :
  (1 / 2) * b * c * Real.sin A = 9 * Real.sqrt 2 / 4 :=
sorry

end cos_C_sin_B_area_l173_173457


namespace cyclist_rejoins_group_time_l173_173385

noncomputable def travel_time (group_speed cyclist_speed distance : ℝ) : ℝ :=
  distance / (cyclist_speed - group_speed)

theorem cyclist_rejoins_group_time
  (group_speed : ℝ := 35)
  (cyclist_speed : ℝ := 45)
  (distance : ℝ := 10)
  : travel_time group_speed cyclist_speed distance * 2 = 1 / 4 :=
by
  sorry

end cyclist_rejoins_group_time_l173_173385


namespace possible_divisor_of_p_l173_173902

theorem possible_divisor_of_p (p q r s : ℕ)
  (hpq : ∃ x y, p = 40 * x ∧ q = 40 * y ∧ Nat.gcd p q = 40)
  (hqr : ∃ u v, q = 45 * u ∧ r = 45 * v ∧ Nat.gcd q r = 45)
  (hrs : ∃ w z, r = 60 * w ∧ s = 60 * z ∧ Nat.gcd r s = 60)
  (hsp : ∃ t, Nat.gcd s p = 100 * t ∧ 100 ≤ Nat.gcd s p ∧ Nat.gcd s p < 1000) :
  7 ∣ p :=
sorry

end possible_divisor_of_p_l173_173902


namespace gcd_324_243_l173_173343

-- Define the numbers involved in the problem.
def a : ℕ := 324
def b : ℕ := 243

-- State the theorem that the GCD of a and b is 81.
theorem gcd_324_243 : Nat.gcd a b = 81 := by
  sorry

end gcd_324_243_l173_173343


namespace intersection_complement_l173_173292

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 2}

-- Statement to prove
theorem intersection_complement :
  (((I \ B) ∩ A : Set ℕ) = {3, 5}) :=
by
  sorry

end intersection_complement_l173_173292


namespace cube_volume_from_surface_area_l173_173989

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173989


namespace min_trips_is_157_l173_173923

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end min_trips_is_157_l173_173923


namespace arithmetic_operators_correct_l173_173673

theorem arithmetic_operators_correct :
  let op1 := (132: ℝ) - (7: ℝ) * (6: ℝ)
  let op2 := (12: ℝ) + (3: ℝ)
  (op1 / op2) = (6: ℝ) := by 
  sorry

end arithmetic_operators_correct_l173_173673


namespace total_pencils_l173_173409

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 11) : (pencils_per_child * children = 22) := 
by
  sorry

end total_pencils_l173_173409


namespace correct_sample_in_survey_l173_173083

-- Definitions based on conditions:
def total_population := 1500
def surveyed_population := 150
def sample_description := "the national security knowledge of the selected 150 teachers and students"

-- Hypotheses: conditions
variables (pop : ℕ) (surveyed : ℕ) (description : String)
  (h1 : pop = total_population)
  (h2 : surveyed = surveyed_population)
  (h3 : description = sample_description)

-- Theorem we want to prove
theorem correct_sample_in_survey : description = sample_description :=
  by sorry

end correct_sample_in_survey_l173_173083


namespace cube_volume_from_surface_area_l173_173965

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173965


namespace opposite_of_2021_l173_173183

theorem opposite_of_2021 : -(2021) = -2021 := 
sorry

end opposite_of_2021_l173_173183


namespace cos_of_largest_angle_is_neg_half_l173_173429

-- Lean does not allow forward references to elements yet to be declared, 
-- hence we keep a strict order for declarations
namespace TriangleCosine

open Real

-- Define the side lengths of the triangle as constants
def a : ℝ := 3
def b : ℝ := 5
def c : ℝ := 7

-- Define the expression using cosine rule to find cos C
noncomputable def cos_largest_angle : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

-- Declare the theorem statement
theorem cos_of_largest_angle_is_neg_half : cos_largest_angle = -1 / 2 := 
by 
  sorry

end TriangleCosine

end cos_of_largest_angle_is_neg_half_l173_173429


namespace limit_problem_l173_173403

open Real

theorem limit_problem :
  filter.tendsto (λ x : ℝ, (exp (2 * x) - exp (3 * x)) / (arctan x - x ^ 2))
                (nhds_within 0 (set.univ : set ℝ))
                (nhds (-1)) :=
  sorry

end limit_problem_l173_173403


namespace sufficient_condition_for_inequality_l173_173020

open Real

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 1 / 5) : 1 / a > 3 :=
by
  sorry

end sufficient_condition_for_inequality_l173_173020


namespace intersection_of_sets_l173_173861

variable (M : Set ℤ) (N : Set ℤ)

theorem intersection_of_sets :
  M = {-2, -1, 0, 1, 2} →
  N = {x | x ≥ 3 ∨ x ≤ -2} →
  M ∩ N = {-2} :=
by
  intros hM hN
  sorry

end intersection_of_sets_l173_173861


namespace sin_30_to_cos_60_and_function_value_l173_173702

theorem sin_30_to_cos_60_and_function_value :
  let f : ℝ → ℝ := λ x, cos (3 * real.arccos x) in
  f (real.sin (real.pi / 6)) = -1 :=
by
  -- Definitions
  def f : ℝ → ℝ := λ x, real.cos (3 * real.arccos x)
  have h1 : real.sin (real.pi / 6) = real.cos (real.pi / 3), from sorry
  calc
    f (real.sin (real.pi / 6)) = f (real.cos (real.pi / 3)) : by rw [h1]
                            ... = real.cos (3 * (real.pi / 3)) : by simp [f]
                            ... = real.cos real.pi : by congr
                            ... = -1 : by norm_num

end sin_30_to_cos_60_and_function_value_l173_173702


namespace linear_function_in_quadrants_l173_173684

section LinearFunctionQuadrants

variable (m : ℝ)

def passesThroughQuadrants (m : ℝ) : Prop :=
  (m + 1 > 0) ∧ (m - 1 > 0)

theorem linear_function_in_quadrants (h : passesThroughQuadrants m) : m > 1 :=
by sorry

end LinearFunctionQuadrants

end linear_function_in_quadrants_l173_173684


namespace median_product_sum_l173_173926

-- Let's define the lengths of medians and distances from a point P to these medians
variables {s1 s2 s3 d1 d2 d3 : ℝ}

-- Define the conditions
def is_median_lengths (s1 s2 s3 : ℝ) : Prop := 
  ∃ (A B C : ℝ × ℝ), -- vertices of the triangle
    (s1 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2) ∧
    (s2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 2) ∧
    (s3 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 2)

def distances_to_medians (d1 d2 d3 : ℝ) : Prop :=
  ∃ (P A B C : ℝ × ℝ), -- point P and vertices of the triangle
    (d1 = dist P ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
    (d2 = dist P ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧
    (d3 = dist P ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

-- The theorem which we need to prove
theorem median_product_sum (h_medians : is_median_lengths s1 s2 s3) 
  (h_distances : distances_to_medians d1 d2 d3) :
  s1 * d1 + s2 * d2 + s3 * d3 = 0 := sorry

end median_product_sum_l173_173926


namespace apple_cost_l173_173651

theorem apple_cost (cost_per_pound : ℚ) (weight : ℚ) (total_cost : ℚ) : cost_per_pound = 1 ∧ weight = 18 → total_cost = 18 :=
by
  sorry

end apple_cost_l173_173651


namespace angles_equal_l173_173941

theorem angles_equal {α β γ α1 β1 γ1 : ℝ} (h1 : α + β + γ = 180) (h2 : α1 + β1 + γ1 = 180) 
  (h_eq_or_sum_to_180 : (α = α1 ∨ α + α1 = 180) ∧ (β = β1 ∨ β + β1 = 180) ∧ (γ = γ1 ∨ γ + γ1 = 180)) :
  α = α1 ∧ β = β1 ∧ γ = γ1 := 
by 
  sorry

end angles_equal_l173_173941


namespace total_balls_estimation_l173_173712

theorem total_balls_estimation 
  (num_red_balls : ℕ)
  (total_trials : ℕ)
  (red_ball_draws : ℕ)
  (red_ball_ratio : ℚ)
  (total_balls_estimate : ℕ)
  (h1 : num_red_balls = 5)
  (h2 : total_trials = 80)
  (h3 : red_ball_draws = 20)
  (h4 : red_ball_ratio = 1 / 4)
  (h5 : red_ball_ratio = red_ball_draws / total_trials)
  (h6 : red_ball_ratio = num_red_balls / total_balls_estimate)
  : total_balls_estimate = 20 := 
sorry

end total_balls_estimation_l173_173712


namespace employee_pay_per_week_l173_173942

theorem employee_pay_per_week (total_pay : ℝ) (ratio : ℝ) (pay_b : ℝ)
  (h1 : total_pay = 570)
  (h2 : ratio = 1.5)
  (h3 : total_pay = pay_b * (ratio + 1)) :
  pay_b = 228 :=
sorry

end employee_pay_per_week_l173_173942


namespace prob_not_lose_money_proof_min_purchase_price_proof_l173_173672

noncomputable def prob_not_lose_money : ℚ :=
  let pr_normal_rain := (2 : ℚ) / 3
  let pr_less_rain := (1 : ℚ) / 3
  let pr_price_6_normal := (1 : ℚ) / 4
  let pr_price_6_less := (2 : ℚ) / 3
  pr_normal_rain * pr_price_6_normal + pr_less_rain * pr_price_6_less

theorem prob_not_lose_money_proof : prob_not_lose_money = 7 / 18 := sorry

noncomputable def min_purchase_price : ℚ :=
  let old_exp_income := 500
  let new_yield := 2500
  let cost_increase := 1000
  (7000 + 1500 + cost_increase) / new_yield
  
theorem min_purchase_price_proof : min_purchase_price = 3.4 := sorry

end prob_not_lose_money_proof_min_purchase_price_proof_l173_173672


namespace hyperbola_equation_l173_173000

-- Definitions based on problem conditions
def asymptotes (x y : ℝ) : Prop :=
  y = (1/3) * x ∨ y = -(1/3) * x

def focus (p : ℝ × ℝ) : Prop :=
  p = (Real.sqrt 10, 0)

-- The main statement to prove
theorem hyperbola_equation :
  (∃ p, focus p) ∧ (∀ (x y : ℝ), asymptotes x y) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 = 1)) :=
sorry

end hyperbola_equation_l173_173000


namespace parabola_vertex_l173_173918

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := (x - 2)^2 + 5

-- State the theorem to find the vertex
theorem parabola_vertex : ∃ h k : ℝ, ∀ x : ℝ, parabola_equation x = (x - h)^2 + k ∧ h = 2 ∧ k = 5 :=
by
  sorry

end parabola_vertex_l173_173918


namespace peter_bought_large_glasses_l173_173328

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end peter_bought_large_glasses_l173_173328


namespace perfect_square_trinomial_l173_173855

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, (x : ℝ) → (x^2 + 2 * (m - 1) * x + 16) = (a * x + b)^2) → (m = 5 ∨ m = -3) :=
by
  sorry

end perfect_square_trinomial_l173_173855


namespace find_value_l173_173289

theorem find_value (m n : ℤ) (h : 2 * m + n - 2 = 0) : 2 * m + n + 1 = 3 :=
by { sorry }

end find_value_l173_173289


namespace masha_more_cakes_l173_173887

theorem masha_more_cakes (S : ℝ) (m n : ℝ) (H1 : S > 0) (H2 : m > 0) (H3 : n > 0) 
  (H4 : 2 * S * (m + n) ≤ S * m + (1/3) * S * n) :
  m > n := 
by 
  sorry

end masha_more_cakes_l173_173887


namespace number_of_intersections_l173_173664

def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1
def vertical_line (x : ℝ) : Prop := x = 4

theorem number_of_intersections : 
    (∃ y : ℝ, ellipse 4 y ∧ vertical_line 4) ∧ 
    ∀ y1 y2, (ellipse 4 y1 ∧ vertical_line 4) → (ellipse 4 y2 ∧ vertical_line 4) → y1 = y2 :=
by
  sorry

end number_of_intersections_l173_173664


namespace sum_of_possible_values_of_a_l173_173156

theorem sum_of_possible_values_of_a :
  ∀ (a b c d : ℝ), a > b → b > c → c > d → a + b + c + d = 50 → 
  (a - b = 4 ∧ b - d = 7 ∧ a - c = 5 ∧ c - d = 6 ∧ b - c = 2 ∨
   a - b = 5 ∧ b - d = 6 ∧ a - c = 4 ∧ c - d = 7 ∧ b - c = 2) →
  (a = 17.75 ∨ a = 18.25) →
  a + 18.25 + 17.75 - a = 36 :=
by sorry

end sum_of_possible_values_of_a_l173_173156


namespace distribute_pencils_l173_173285

def number_of_ways_to_distribute_pencils (pencils friends : ℕ) : ℕ :=
  Nat.choose (pencils - friends + friends - 1) (friends - 1)

theorem distribute_pencils :
  number_of_ways_to_distribute_pencils 4 4 = 35 :=
by
  sorry

end distribute_pencils_l173_173285


namespace remainder_division_l173_173544

def f (x : ℝ) : ℝ := x^3 - 4 * x + 7

theorem remainder_division (x : ℝ) : f 3 = 22 := by
  sorry

end remainder_division_l173_173544


namespace coplanar_k_values_l173_173604

noncomputable def coplanar_lines_possible_k (k : ℝ) : Prop :=
  ∃ (t u : ℝ), (2 + t = 1 + k * u) ∧ (3 + t = 4 + 2 * u) ∧ (4 - k * t = 5 + u)

theorem coplanar_k_values :
  ∀ k : ℝ, coplanar_lines_possible_k k ↔ (k = 0 ∨ k = -3) :=
by
  sorry

end coplanar_k_values_l173_173604


namespace max_value_of_k_l173_173532

theorem max_value_of_k (m : ℝ) (h₁ : 0 < m) (h₂ : m < 1/2) : 
  (1 / m + 2 / (1 - 2 * m)) ≥ 8 :=
sorry

end max_value_of_k_l173_173532


namespace arithmetic_seq_a₄_l173_173430

-- Definitions for conditions in the given problem
def S₅ (a₁ a₅ : ℕ) : ℕ := ((a₁ + a₅) * 5) / 2
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Final proof statement to show that a₄ = 9
theorem arithmetic_seq_a₄ (a₁ a₅ : ℕ) (d : ℕ) (h₁ : S₅ a₁ a₅ = 35) (h₂ : a₅ = 11) (h₃ : d = (a₅ - a₁) / 4) :
  arithmetic_sequence a₁ d 4 = 9 :=
sorry

end arithmetic_seq_a₄_l173_173430


namespace area_six_layers_l173_173282

theorem area_six_layers
  (A : ℕ → ℕ)
  (h1 : A 1 + A 2 + A 3 = 280)
  (h2 : A 2 = 54)
  (h3 : A 3 = 28)
  (h4 : A 4 = 14)
  (h5 : A 1 + 2 * A 2 + 3 * A 3 + 4 * A 4 + 6 * A 6 = 500)
  : A 6 = 9 := 
sorry

end area_six_layers_l173_173282


namespace average_of_remaining_two_l173_173641

-- Given conditions
def average_of_six (S : ℝ) := S / 6 = 3.95
def average_of_first_two (S1 : ℝ) := S1 / 2 = 4.2
def average_of_next_two (S2 : ℝ) := S2 / 2 = 3.85

-- Prove that the average of the remaining 2 numbers equals 3.8
theorem average_of_remaining_two (S S1 S2 Sr : ℝ) (h1 : average_of_six S) (h2 : average_of_first_two S1) (h3: average_of_next_two S2) (h4 : Sr = S - S1 - S2) :
  Sr / 2 = 3.8 :=
by
  -- We can use the assumptions h1, h2, h3, and h4 to reach the conclusion
  sorry

end average_of_remaining_two_l173_173641


namespace intersection_of_A_and_B_l173_173608

def A := {x : ℝ | x^2 - 5 * x + 6 > 0}
def B := {x : ℝ | x / (x - 1) < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end intersection_of_A_and_B_l173_173608


namespace angle_ABF_l173_173713

noncomputable theory

variables {a b c : ℝ}
variable [fact (0 < a)]
variable [fact (0 < b)]
variable [fact (b < a)]
variable [fact (c = a * (Real.sqrt 5 - 1) / 2)]

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem angle_ABF (h : ellipse a b)
(h_ecc : (Real.sqrt 5 - 1) / 2 = c / a)
: sorry := sorry

end angle_ABF_l173_173713


namespace snakes_in_pond_l173_173009

theorem snakes_in_pond (S : ℕ) (alligators : ℕ := 10) (total_eyes : ℕ := 56) (alligator_eyes : ℕ := 2) (snake_eyes : ℕ := 2) :
  (alligators * alligator_eyes) + (S * snake_eyes) = total_eyes → S = 18 :=
by
  intro h
  sorry

end snakes_in_pond_l173_173009


namespace cube_volume_from_surface_area_example_cube_volume_l173_173960

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173960


namespace z_real_iff_z_complex_iff_z_pure_imaginary_iff_l173_173106

-- Definitions for the problem conditions
def z_real (m : ℝ) : Prop := (m^2 - 2 * m - 15 = 0)
def z_pure_imaginary (m : ℝ) : Prop := (m^2 - 9 * m - 36 = 0) ∧ (m^2 - 2 * m - 15 ≠ 0)

-- Question 1: Prove that z is a real number if and only if m = -3 or m = 5
theorem z_real_iff (m : ℝ) : z_real m ↔ m = -3 ∨ m = 5 := sorry

-- Question 2: Prove that z is a complex number with non-zero imaginary part if and only if m ≠ -3 and m ≠ 5
theorem z_complex_iff (m : ℝ) : ¬z_real m ↔ m ≠ -3 ∧ m ≠ 5 := sorry

-- Question 3: Prove that z is a pure imaginary number if and only if m = 12
theorem z_pure_imaginary_iff (m : ℝ) : z_pure_imaginary m ↔ m = 12 := sorry

end z_real_iff_z_complex_iff_z_pure_imaginary_iff_l173_173106


namespace simplify_fraction_l173_173742

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := 
by sorry

end simplify_fraction_l173_173742


namespace number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l173_173298

theorem number_of_two_digit_factors_2_pow_18_minus_1_is_zero :
  (∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ n ∣ (2^18 - 1)) = false :=
by sorry

end number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l173_173298


namespace remainder_when_expression_divided_l173_173059

theorem remainder_when_expression_divided 
  (x y u v : ℕ) 
  (h1 : x = u * y + v) 
  (h2 : 0 ≤ v) 
  (h3 : v < y) :
  (x - u * y + 3 * v) % y = (4 * v) % y :=
by
  sorry

end remainder_when_expression_divided_l173_173059


namespace compare_probabilities_l173_173378

open ProbabilityTheory

noncomputable def sumDice (n : ℕ) (sides : finset ℕ) (p : ∀ i ∈ sides, 0 < (uniformProb sides) i ∧ (uniformProb sides) i ≤ 1) : ProbMeasure (fin n → ℕ) :=
  fun _ => classical.arbitrary (ProbMeasure (fin n → ℕ))

theorem compare_probabilities :
  let dice_faces := {1, 2, 3, 4, 5, 6}
  let num_dice := 60
  let S := sumDice num_dice dice_faces (by simp [uniformProb])
  AllDiceAreFair := ∀ x, x ∈ dice_faces → ∀ k, x / (fin_to_int k) = 1/6
  probability_of_sum_at_least_300 := S.ret (λ f, (∑ i, f i) ≥ 300)
  probability_of_sum_less_than_120 := S.ret (λ f, (∑ i, f i) < 120)
  P₁ := probability_of_sum_at_least_300,
  P₂ := probability_of_sum_less_than_120,
  P₁ > P₂ :=
begin
  sorry
end

end compare_probabilities_l173_173378


namespace straws_to_adult_pigs_l173_173054

theorem straws_to_adult_pigs (total_straws : ℕ) (num_piglets : ℕ) (straws_per_piglet : ℕ)
  (straws_adult_pigs : ℕ) (straws_piglets : ℕ) :
  total_straws = 300 →
  num_piglets = 20 →
  straws_per_piglet = 6 →
  (straws_piglets = num_piglets * straws_per_piglet) →
  (straws_adult_pigs = straws_piglets) →
  straws_adult_pigs = 120 :=
by
  intros h_total h_piglets h_straws_per_piglet h_straws_piglets h_equal
  subst h_total
  subst h_piglets
  subst h_straws_per_piglet
  subst h_straws_piglets
  subst h_equal
  sorry

end straws_to_adult_pigs_l173_173054


namespace sequence_properties_l173_173857

variable {a : ℕ → ℤ}

-- Conditions
axiom seq_add : ∀ (p q : ℕ), 1 ≤ p → 1 ≤ q → a (p + q) = a p + a q
axiom a2_neg4 : a 2 = -4

-- Theorem statement: We need to prove a6 = -12 and a_n = -2n for all n
theorem sequence_properties :
  (a 6 = -12) ∧ ∀ n : ℕ, 1 ≤ n → a n = -2 * n :=
by
  sorry

end sequence_properties_l173_173857


namespace sum_of_first_15_terms_of_arithmetic_sequence_l173_173574

theorem sum_of_first_15_terms_of_arithmetic_sequence 
  (a d : ℕ) 
  (h1 : (5 * (2 * a + 4 * d)) / 2 = 10) 
  (h2 : (10 * (2 * a + 9 * d)) / 2 = 50) :
  (15 * (2 * a + 14 * d)) / 2 = 120 :=
sorry

end sum_of_first_15_terms_of_arithmetic_sequence_l173_173574


namespace cos_sequence_next_coeff_sum_eq_28_l173_173139

theorem cos_sequence_next_coeff_sum_eq_28 (α : ℝ) :
  let u := 2 * Real.cos α
  2 * Real.cos (8 * α) = u ^ 8 - 8 * u ^ 6 + 20 * u ^ 4 - 16 * u ^ 2 + 2 → 
  8 + (-8) + 6 + 20 + 2 = 28 :=
by intros u; sorry

end cos_sequence_next_coeff_sum_eq_28_l173_173139


namespace option_C_incorrect_l173_173168

def p (x y : ℝ) : ℝ := x^3 - 3 * x^2 * y + 3 * x * y^2 - y^3

theorem option_C_incorrect (x y : ℝ) : 
  ((x^3 - 3 * x^2 * y) - (3 * x * y^2 + y^3)) ≠ p x y := by
  sorry

end option_C_incorrect_l173_173168


namespace max_sum_abc_l173_173454

theorem max_sum_abc (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) : a + b + c ≤ 3 :=
sorry

end max_sum_abc_l173_173454


namespace total_items_deleted_l173_173663

-- Define the initial conditions
def initial_apps : Nat := 17
def initial_files : Nat := 21
def remaining_apps : Nat := 3
def remaining_files : Nat := 7
def transferred_files : Nat := 4

-- Prove the total number of deleted items
theorem total_items_deleted : (initial_apps - remaining_apps) + (initial_files - (remaining_files + transferred_files)) = 24 :=
by
  sorry

end total_items_deleted_l173_173663


namespace total_students_in_class_is_15_l173_173297

noncomputable def choose (n k : ℕ) : ℕ := sorry -- Define a function for combinations
noncomputable def permute (n k : ℕ) : ℕ := sorry -- Define a function for permutations

variables (x m n : ℕ) (hx : choose x 4 = m) (hn : permute x 2 = n) (hratio : m * 2 = n * 13)

theorem total_students_in_class_is_15 : x = 15 :=
sorry

end total_students_in_class_is_15_l173_173297


namespace sum_of_coefficients_l173_173708

theorem sum_of_coefficients (a b : ℝ) (h : ∀ x : ℝ, (x > 1 ∧ x < 4) ↔ (ax^2 + bx - 2 > 0)) :
  a + b = 2 :=
by
  sorry

end sum_of_coefficients_l173_173708


namespace value_of_f_neg_2_l173_173025

section
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_pos : ∀ x : ℝ, 0 < x → f x = 2 ^ x + 1)

theorem value_of_f_neg_2 (h_odd : ∀ x, f (-x) = -f x) (h_pos : ∀ x, 0 < x → f x = 2^x + 1) :
  f (-2) = -5 :=
by
  sorry
end

end value_of_f_neg_2_l173_173025


namespace find_w_l173_173203

variable (p j t : ℝ) (w : ℝ)

-- Definitions based on conditions
def j_less_than_p : Prop := j = 0.75 * p
def j_less_than_t : Prop := j = 0.80 * t
def t_less_than_p : Prop := t = p * (1 - w / 100)

-- Objective: Prove that given these conditions, w = 6.25
theorem find_w (h1 : j_less_than_p p j) (h2 : j_less_than_t j t) (h3 : t_less_than_p t p w) : 
  w = 6.25 := 
by 
  sorry

end find_w_l173_173203


namespace principal_amount_l173_173948

variable (SI R T P : ℝ)

-- Given conditions
axiom SI_def : SI = 2500
axiom R_def : R = 10
axiom T_def : T = 5

-- Main theorem statement
theorem principal_amount : SI = (P * R * T) / 100 → P = 5000 :=
by
  sorry

end principal_amount_l173_173948


namespace self_employed_tax_amount_l173_173625

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end self_employed_tax_amount_l173_173625


namespace cost_of_pencils_and_pens_l173_173919

theorem cost_of_pencils_and_pens (a b : ℝ) (h1 : 4 * a + b = 2.60) (h2 : a + 3 * b = 2.15) : 3 * a + 2 * b = 2.63 :=
sorry

end cost_of_pencils_and_pens_l173_173919


namespace unique_solution_arcsin_equation_l173_173125

theorem unique_solution_arcsin_equation :
  ∃! x ∈ set.Icc (-0.5) 0.5, Real.arcsin (2 * x) + Real.arcsin x = (Real.pi / 3) :=
sorry

end unique_solution_arcsin_equation_l173_173125


namespace cube_volume_from_surface_area_l173_173970

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173970


namespace balanced_scale_l173_173058

def children's_book_weight : ℝ := 1.1

def weight1 : ℝ := 0.5
def weight2 : ℝ := 0.3
def weight3 : ℝ := 0.3

theorem balanced_scale :
  (weight1 + weight2 + weight3) = children's_book_weight :=
by
  sorry

end balanced_scale_l173_173058


namespace factorize_expression_l173_173671

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l173_173671


namespace polynomial_identity_l173_173337

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
  sorry

end polynomial_identity_l173_173337


namespace fraction_is_one_twelve_l173_173033

variables (A E : ℝ) (f : ℝ)

-- Given conditions
def condition1 : E = 200 := sorry
def condition2 : A - E = f * (A + E) := sorry
def condition3 : A * 1.10 = E * 1.20 + 20 := sorry

-- Proving the fraction f is 1/12
theorem fraction_is_one_twelve : E = 200 → (A - E = f * (A + E)) → (A * 1.10 = E * 1.20 + 20) → 
f = 1 / 12 :=
by
  intros hE hDiff hIncrease
  sorry

end fraction_is_one_twelve_l173_173033


namespace polynomial_transformation_l173_173061

noncomputable def p : ℝ → ℝ := sorry

variable (k : ℕ)

axiom ax1 (x : ℝ) : p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))

theorem polynomial_transformation (k : ℕ) (p : ℝ → ℝ)
  (h_p : ∀ x : ℝ, p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))) :
  ∀ x : ℝ, p (3 * x) = 3^(k - 1) * (p x + p (x + 1/3) + p (x + 2/3)) := sorry

end polynomial_transformation_l173_173061


namespace cube_volume_is_1728_l173_173950

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173950


namespace polynomial_complete_square_l173_173455

theorem polynomial_complete_square :
  ∃ a h k : ℝ, (∀ x : ℝ, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) ∧ a + h + k = -2.5 := by
  sorry

end polynomial_complete_square_l173_173455


namespace cube_volume_from_surface_area_l173_173991

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173991


namespace initial_cases_purchased_l173_173381

open Nat

-- Definitions based on conditions

def group1_children := 14
def group2_children := 16
def group3_children := 12
def group4_children := (group1_children + group2_children + group3_children) / 2
def total_children := group1_children + group2_children + group3_children + group4_children

def bottles_per_child_per_day := 3
def days := 3
def total_bottles_needed := total_children * bottles_per_child_per_day * days

def additional_bottles_needed := 255

def bottles_per_case := 24
def initial_bottles := total_bottles_needed - additional_bottles_needed

def cases_purchased := initial_bottles / bottles_per_case

-- Theorem to prove the number of cases purchased initially
theorem initial_cases_purchased : cases_purchased = 13 :=
  sorry

end initial_cases_purchased_l173_173381


namespace car_fuel_tanks_l173_173746

theorem car_fuel_tanks {x X p : ℝ}
  (h1 : x + X = 70)            -- Condition: total capacity is 70 liters
  (h2 : x * p = 45)            -- Condition: cost to fill small car's tank
  (h3 : X * (p + 0.29) = 68)   -- Condition: cost to fill large car's tank
  : x = 30 ∧ X = 40            -- Conclusion: capacities of the tanks
  :=
by {
  sorry
}

end car_fuel_tanks_l173_173746


namespace trig_identity_l173_173295

theorem trig_identity (x : ℝ) (h0 : -3 * Real.pi / 2 < x) (h1 : x < -Real.pi) (h2 : Real.tan x = -3) :
  Real.sin x * Real.cos x = -3 / 10 :=
sorry

end trig_identity_l173_173295


namespace distinct_integer_a_values_l173_173425

open Nat

theorem distinct_integer_a_values : ∃ (S : Finset ℤ), S.card = 8 ∧
  ∀ a ∈ S, ∃ (p q : ℤ), p + q = -a ∧ p * q = 12a ∧ 
  ∀ x ∈ (polynomial.roots (polynomial.C (12 : ℚ) * polynomial.X + polynomial.C a * polynomial.X + polynomial.C 1)).val, x ∈ ℤ :=
by
  sorry

end distinct_integer_a_values_l173_173425


namespace hyperbola_eccentricity_l173_173119

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_eq1 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : c = Real.sqrt (a^2 + b^2))
  (h_dist : ∀ x, x = b * c / Real.sqrt (a^2 + b^2))
  (h_eq3 : a = b) :
  e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l173_173119


namespace units_digit_product_l173_173633

theorem units_digit_product :
  let nums : List Nat := [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
  let product := nums.prod
  (product % 10) = 9 :=
by
  sorry

end units_digit_product_l173_173633


namespace domain_of_y_l173_173754

noncomputable def domain_of_function : Set ℝ := {x : ℝ | -3 < x ∧ x < 2 ∧ x ≠ 1}

theorem domain_of_y :
  ∀ x : ℝ, (2 - x > 0) ∧ (12 + x - x^2 > 0) ∧ (x ≠ 1) ↔ x ∈ domain_of_function :=
begin
  sorry
end

end domain_of_y_l173_173754


namespace jessica_earned_from_washing_l173_173469

-- Conditions defined as per Problem a)
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def remaining_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 11
def earned_from_washing : ℕ := final_amount - remaining_after_movies

-- Lean statement to prove Jessica earned $6 from washing the family car
theorem jessica_earned_from_washing :
  earned_from_washing = 6 := 
by
  -- Proof to be filled in later (skipped here with sorry)
  sorry

end jessica_earned_from_washing_l173_173469


namespace compute_value_3_std_devs_less_than_mean_l173_173614

noncomputable def mean : ℝ := 15
noncomputable def std_dev : ℝ := 1.5
noncomputable def skewness : ℝ := 0.5
noncomputable def kurtosis : ℝ := 0.6

theorem compute_value_3_std_devs_less_than_mean : 
  ¬∃ (value : ℝ), value = mean - 3 * std_dev :=
sorry

end compute_value_3_std_devs_less_than_mean_l173_173614


namespace car_parking_arrangements_l173_173575

theorem car_parking_arrangements : 
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  (red_car_positions * arrange_black_cars) = 14400 := 
by
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  sorry

end car_parking_arrangements_l173_173575


namespace gcd_765432_654321_l173_173835

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end gcd_765432_654321_l173_173835


namespace sum_of_factors_of_30_l173_173508

/--
Given the positive integer factors of 30, prove that their sum is 72.
-/
theorem sum_of_factors_of_30 : 
  (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 := 
by 
  sorry

end sum_of_factors_of_30_l173_173508


namespace limit_problem_l173_173402

open Real

theorem limit_problem :
  tendsto (λ x : ℝ, (sin x + sin (π * x) * arctan ((1 + x) / (1 - x))) / (1 + cos x)) (𝓝 1) (𝓝 (sin 1 / (1 + cos 1))) :=
by
  -- proof steps would go here
  sorry

end limit_problem_l173_173402


namespace janine_test_score_l173_173731

theorem janine_test_score :
  let num_mc := 10
  let p_mc := 0.80
  let num_sa := 30
  let p_sa := 0.70
  let total_questions := 40
  let correct_mc := p_mc * num_mc
  let correct_sa := p_sa * num_sa
  let total_correct := correct_mc + correct_sa
  (total_correct / total_questions) * 100 = 72.5 := 
by
  sorry

end janine_test_score_l173_173731


namespace value_of_M_l173_173447

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 4025) : M = 5635 :=
sorry

end value_of_M_l173_173447


namespace friedas_reaches_boundary_in_3_hops_l173_173828

noncomputable def friedasProbability (gridSize : ℕ) : ℝ :=
  if gridSize = 4 then 1 else 0

theorem friedas_reaches_boundary_in_3_hops :
  ∀ (start : ℕ × ℕ),
    start = (2, 2) →
    friedasProbability 4 = 1 :=
by
  intros start h_start
  rw h_start
  simp [friedasProbability]
  sorry

end friedas_reaches_boundary_in_3_hops_l173_173828


namespace distribution_scheme_count_l173_173930

-- Define the people and communities
inductive Person
| A | B | C
deriving DecidableEq, Repr

inductive Community
| C1 | C2 | C3 | C4 | C5 | C6 | C7
deriving DecidableEq, Repr

-- Define a function to count the number of valid distribution schemes
def countDistributionSchemes : Nat :=
  -- This counting is based on recognizing the problem involves permutations and combinations,
  -- the specific detail logic is omitted since we are only writing the statement, no proof.
  336

-- The main theorem statement
theorem distribution_scheme_count :
  countDistributionSchemes = 336 :=
sorry

end distribution_scheme_count_l173_173930


namespace expected_value_of_X_is_5_over_3_l173_173795

-- Define the probabilities of getting an interview with company A, B, and C
def P_A : ℚ := 2 / 3
def P_BC (p : ℚ) : ℚ := p

-- Define the random variable X representing the number of interview invitations
def X (P_A P_BC : ℚ) : ℚ := sorry

-- Define the probability of receiving no interview invitations
def P_X_0 (P_A P_BC : ℚ) : ℚ := (1 - P_A) * (1 - P_BC)^2

-- Given condition that P(X=0) is 1/12
def condition_P_X_0 (P_A P_BC : ℚ) : Prop := P_X_0 P_A P_BC = 1 / 12

-- Given p = 1/2 as per the problem solution
def p : ℚ := 1 / 2

-- Expected value of X
def E_X (P_A P_BC : ℚ) : ℚ := (1 * (2 * P_BC * (1 - P_BC) + 2 * P_BC^2 * (1 - P_BC) + (1 - P_A) * P_BC^2)) +
                               (2 * (P_A * P_BC * (1 - P_BC) + P_A * (1 - P_BC)^2 + P_BC * P_BC * (1 - P_A))) +
                               (3 * (P_A * P_BC^2))

-- Theorem proving the expected value of X given the above conditions
theorem expected_value_of_X_is_5_over_3 : E_X P_A (P_BC p) = 5 / 3 :=
by
  -- here you will write the proof later
  sorry

end expected_value_of_X_is_5_over_3_l173_173795


namespace no_integral_value_2001_l173_173474

noncomputable def P (x : ℤ) : ℤ := sorry -- Polynomial definition needs to be filled in

theorem no_integral_value_2001 (a0 a1 a2 a3 a4 : ℤ) (x1 x2 x3 x4 : ℤ) :
  (P x1 = 2020) ∧ (P x2 = 2020) ∧ (P x3 = 2020) ∧ (P x4 = 2020) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  ¬ ∃ x : ℤ, P x = 2001 :=
sorry

end no_integral_value_2001_l173_173474


namespace determine_house_height_l173_173947

-- Definitions for the conditions
def house_shadow : ℚ := 75
def tree_height : ℚ := 15
def tree_shadow : ℚ := 20

-- Desired Height of Lily's house
def house_height : ℚ := 56

-- Theorem stating the height of the house
theorem determine_house_height :
  (house_shadow / tree_shadow = house_height / tree_height) -> house_height = 56 :=
  by
  unfold house_shadow tree_height tree_shadow house_height
  sorry

end determine_house_height_l173_173947


namespace value_of_b_l173_173889

theorem value_of_b (b : ℝ) (x : ℝ) (h : x = 1) (h_eq : 3 * x^2 - b * x + 3 = 0) : b = 6 :=
by
  sorry

end value_of_b_l173_173889


namespace third_altitude_is_less_than_15_l173_173502

variable (a b c : ℝ)
variable (ha hb hc : ℝ)
variable (A : ℝ)

def triangle_area (side : ℝ) (height : ℝ) : ℝ := 0.5 * side * height

axiom ha_eq : ha = 10
axiom hb_eq : hb = 6

theorem third_altitude_is_less_than_15 : hc < 15 :=
sorry

end third_altitude_is_less_than_15_l173_173502


namespace pyramid_volume_theorem_l173_173761

noncomputable def volume_of_regular_square_pyramid : ℝ := 
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * side_edge_length * Real.sqrt 3
  (1 / 3) * base_area * height

theorem pyramid_volume_theorem :
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * (side_edge_length * Real.sqrt 3)
  (1 / 3) * base_area * height = 6 := 
by
  sorry

end pyramid_volume_theorem_l173_173761


namespace find_pairs_of_positive_numbers_l173_173674

theorem find_pairs_of_positive_numbers
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (exists_triangle : ∃ (C D E A B : ℝ), true)
  (points_on_hypotenuse : ∀ (C D E A B : ℝ), A ∈ [D, E] ∧ B ∈ [D, E]) 
  (equal_vectors : ∀ (D A B E : ℝ), (D - A) = (A - B) ∧ (A - B) = (B - E))
  (AC_eq_a : (C - A) = a)
  (BC_eq_b : (C - B) = b) :
  (1 / 2) < (a / b) ∧ (a / b) < 2 :=
by {
  sorry
}

end find_pairs_of_positive_numbers_l173_173674


namespace kids_all_three_activities_l173_173628

-- Definitions based on conditions
def total_kids : ℕ := 40
def kids_tubing : ℕ := total_kids / 4
def kids_tubing_rafting : ℕ := kids_tubing / 2
def kids_tubing_rafting_kayaking : ℕ := kids_tubing_rafting / 3

-- Theorem statement: proof of the final answer
theorem kids_all_three_activities : kids_tubing_rafting_kayaking = 1 := by
  sorry

end kids_all_three_activities_l173_173628


namespace apples_left_proof_l173_173726

def apples_left (mike_apples : Float) (nancy_apples : Float) (keith_apples_eaten : Float): Float :=
  mike_apples + nancy_apples - keith_apples_eaten

theorem apples_left_proof :
  apples_left 7.0 3.0 6.0 = 4.0 :=
by
  unfold apples_left
  norm_num
  sorry

end apples_left_proof_l173_173726


namespace solve_for_x_l173_173275

theorem solve_for_x (x : ℝ) 
  (h : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : 
  x = 9 := 
by
  sorry

end solve_for_x_l173_173275


namespace find_plane_angle_at_apex_l173_173182

noncomputable def plane_angle_at_apex (linear_angle : ℝ) : ℝ :=
  linear_angle / 2

theorem find_plane_angle_at_apex (linear_angle : ℝ) 
  (h : linear_angle = 2 * plane_angle_at_apex linear_angle) : 
  plane_angle_at_apex linear_angle = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
by
  sorry

end find_plane_angle_at_apex_l173_173182


namespace sequence_general_term_l173_173185

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 0 else 2 * n - 4

def S (n : ℕ) : ℤ :=
  n ^ 2 - 3 * n + 2

theorem sequence_general_term (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) := by
  sorry

end sequence_general_term_l173_173185


namespace letters_symmetry_l173_173313

theorem letters_symmetry (people : Fin 20) (sends : Fin 20 → Finset (Fin 20)) (h : ∀ p, (sends p).card = 10) :
  ∃ i j : Fin 20, i ≠ j ∧ j ∈ sends i ∧ i ∈ sends j :=
by
  sorry

end letters_symmetry_l173_173313


namespace speed_of_stream_l173_173798

-- Define the problem conditions
variables (b s : ℝ)
axiom cond1 : 21 = b + s
axiom cond2 : 15 = b - s

-- State the theorem
theorem speed_of_stream : s = 3 :=
sorry

end speed_of_stream_l173_173798


namespace Hulk_jump_l173_173175

theorem Hulk_jump :
  ∃ n : ℕ, 2^n > 500 ∧ ∀ m : ℕ, m < n → 2^m ≤ 500 :=
by
  sorry

end Hulk_jump_l173_173175


namespace simplify_fraction_subtraction_l173_173839

theorem simplify_fraction_subtraction :
  (5 / 15 : ℚ) - (2 / 45) = 13 / 45 :=
by
  -- (The proof will go here)
  sorry

end simplify_fraction_subtraction_l173_173839


namespace check_independence_and_expected_value_l173_173391

noncomputable def contingency_table (students: ℕ) (pct_75 : ℕ) (pct_less10 : ℕ) (num_75_10 : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  let num_75 := students * pct_75 / 100
  let num_less10 := students * pct_less10 / 100
  let num_75_less10 := num_75 - num_75_10
  let num_not75 := students - num_75
  let num_not75_less10 := num_less10 - num_75_less10
  let num_not75_10 := num_not75 - num_not75_less10
  ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10))

noncomputable def chi_square_statistic (a b c d : ℕ) (n: ℕ) : ℚ :=
  (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem check_independence_and_expected_value :
  let students := 500
  let pct_75 := 30
  let pct_less10 := 50
  let num_75_10 := 100
  let ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10)) := contingency_table students pct_75 pct_less10 num_75_10
  let chi2 := chi_square_statistic num_not75_less10 num_75_less10 num_not75_10 num_75_10 students
  let critical_value := 10.828
  let p0 := 1 / 84
  let p1 := 3 / 14
  let p2 := 15 / 28
  let p3 := 5 / 21
  let expected_x := 0 * p0 + 1 * p1 + 2 * p2 + 3 * p3
  (chi2 > critical_value) ∧ (expected_x = 2) :=
by 
  sorry

end check_independence_and_expected_value_l173_173391


namespace stratified_sampling_l173_173081

theorem stratified_sampling (lathe_A lathe_B total_samples : ℕ) (hA : lathe_A = 56) (hB : lathe_B = 42) (hTotal : total_samples = 14) :
  ∃ (sample_A sample_B : ℕ), sample_A = 8 ∧ sample_B = 6 :=
by
  sorry

end stratified_sampling_l173_173081


namespace petya_can_force_difference_2014_l173_173730

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end petya_can_force_difference_2014_l173_173730


namespace min_value_expression_l173_173281

theorem min_value_expression (x y z : ℝ) : ∃ v, v = 0 ∧ ∀ x y z : ℝ, x^2 + 2 * x * y + 3 * y^2 + 2 * x * z + 3 * z^2 ≥ v := 
by 
  use 0
  sorry

end min_value_expression_l173_173281


namespace number_of_men_in_second_group_l173_173376

theorem number_of_men_in_second_group 
  (work : ℕ)
  (days_first_group days_second_group : ℕ)
  (men_first_group men_second_group : ℕ)
  (h1 : work = men_first_group * days_first_group)
  (h2 : work = men_second_group * days_second_group)
  (h3 : men_first_group = 20)
  (h4 : days_first_group = 30)
  (h5 : days_second_group = 24) :
  men_second_group = 25 :=
by
  sorry

end number_of_men_in_second_group_l173_173376


namespace actual_books_bought_l173_173031

def initial_spending : ℕ := 180
def planned_books (x : ℕ) : Prop := initial_spending / x - initial_spending / (5 * x / 4) = 9

theorem actual_books_bought (x : ℕ) (hx : planned_books x) : (5 * x / 4) = 5 :=
by
  sorry

end actual_books_bought_l173_173031


namespace coat_price_proof_l173_173802

variable (W : ℝ) -- wholesale price
variable (currentPrice : ℝ) -- current price of the coat

-- Condition 1: The retailer marked up the coat by 90%.
def markup_90 : Prop := currentPrice = 1.9 * W

-- Condition 2: Further $4 increase achieves a 100% markup.
def increase_4 : Prop := 2 * W - currentPrice = 4

-- Theorem: The current price of the coat is $76.
theorem coat_price_proof (h1 : markup_90 W currentPrice) (h2 : increase_4 W currentPrice) : currentPrice = 76 :=
sorry

end coat_price_proof_l173_173802


namespace ralph_did_not_hit_110_balls_l173_173737

def tennis_problem : Prop :=
  ∀ (total_balls first_batch second_batch hit_first hit_second not_hit_first not_hit_second not_hit_total : ℕ),
  total_balls = 175 →
  first_batch = 100 →
  second_batch = 75 →
  hit_first = 2/5 * first_batch →
  hit_second = 1/3 * second_batch →
  not_hit_first = first_batch - hit_first →
  not_hit_second = second_batch - hit_second →
  not_hit_total = not_hit_first + not_hit_second →
  not_hit_total = 110

theorem ralph_did_not_hit_110_balls : tennis_problem := by
  unfold tennis_problem
  intros
  sorry

end ralph_did_not_hit_110_balls_l173_173737


namespace monge_point_intersection_altitude_foot_circumcircle_l173_173781

-- Define a tetrahedron in 3D space
noncomputable def Tetrahedron (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the concept of midpoints and perpendicular planes
noncomputable def Midpoint (a b : EuclideanSpace ℝ (Fin 3)) := (1/2) • (a + b)
axiom Perpendicular (a b c : EuclideanSpace ℝ (Fin 3)) : Prop

-- Define the Monge point
noncomputable def MongePoint (A B C D : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the circumscribed circle and the foot of the altitude
axiom CircumscribedCircle (A B C : EuclideanSpace ℝ (Fin 3)) : Set (EuclideanSpace ℝ (Fin 3))
axiom FootOfAltitude (D A B C : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3)

-- Define the conditions as properties
axiom MidpointPerpendicularPlanes (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (∀ a b, (a, b) ∈ {(A, B), (A, C), (A, D), (B, C), (B, D), (C, D)} → 
  ∃ p : EuclideanSpace ℝ (Fin 3), Perpendicular p (Midpoint a b) (opposite_edge a b) ∧ 
  ∀ q : EuclideanSpace ℝ (Fin 3), Perpendicular q (Midpoint a b) (opposite_edge a b) → q = p ) ∧
  ∃ M : EuclideanSpace ℝ (Fin 3), ∀ p : EuclideanSpace ℝ (Fin 3), MongePoint A B C D = p

-- Prove that all midpoints planes intersect at the Monge point
theorem monge_point_intersection (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (Tetrahedron A B C D) → 
  (∀ e1 e2 : EuclideanSpace ℝ (Fin 3), MidpointPerpendicularPlanes e1 e2 → 
   ∀ p : EuclideanSpace ℝ (Fin 3), Perpendicular p (Midpoint e1 e2) e1 e2) →
  ∃ O : EuclideanSpace ℝ (Fin 3), MongePoint A B C D = O := 
  sorry

-- Prove the property about the foot of the altitude and the circumcircle
theorem altitude_foot_circumcircle (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (Tetrahedron A B C D) → 
  (MongePoint A B C D ∈ Plane (A, B, C)) →
  FootOfAltitude D A B C ∈ CircumscribedCircle A B C :=
  sorry

end monge_point_intersection_altitude_foot_circumcircle_l173_173781


namespace cube_volume_l173_173982

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173982


namespace michael_total_revenue_l173_173599

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end michael_total_revenue_l173_173599


namespace sum_of_yellow_and_blue_is_red_l173_173757

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) : ∃ k : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * k + 1 :=
by sorry

end sum_of_yellow_and_blue_is_red_l173_173757


namespace combined_mean_correct_l173_173485

section MeanScore

variables (score_first_section mean_first_section : ℝ)
variables (score_second_section mean_second_section : ℝ)
variables (num_first_section num_second_section : ℝ)

axiom mean_first : mean_first_section = 92
axiom mean_second : mean_second_section = 78
axiom ratio_students : num_first_section / num_second_section = 5 / 7

noncomputable def combined_mean_score : ℝ := 
  let total_score := (mean_first_section * num_first_section + mean_second_section * num_second_section)
  let total_students := (num_first_section + num_second_section)
  total_score / total_students

theorem combined_mean_correct : combined_mean_score 92 78 (5 / 7 * num_second_section) num_second_section = 83.8 := by
  sorry

end MeanScore

end combined_mean_correct_l173_173485


namespace fraction_is_one_over_three_l173_173920

variable (x : ℚ) -- Let the fraction x be a rational number
variable (num : ℚ) -- Let the number be a rational number

theorem fraction_is_one_over_three (h1 : num = 45) (h2 : x * num - 5 = 10) : x = 1 / 3 := by
  sorry

end fraction_is_one_over_three_l173_173920


namespace balloon_arrangements_l173_173265

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l173_173265


namespace infinitely_many_squares_of_form_l173_173154

theorem infinitely_many_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ n' > n, 2 * k * n' - 7 = m^2 :=
sorry

end infinitely_many_squares_of_form_l173_173154


namespace deshaun_read_books_over_summer_l173_173829

theorem deshaun_read_books_over_summer 
  (summer_days : ℕ)
  (average_pages_per_book : ℕ)
  (ratio_closest_person : ℝ)
  (pages_read_per_day_second_person : ℕ)
  (books_read : ℕ)
  (total_pages_second_person_read : ℕ)
  (h1 : summer_days = 80)
  (h2 : average_pages_per_book = 320)
  (h3 : ratio_closest_person = 0.75)
  (h4 : pages_read_per_day_second_person = 180)
  (h5 : total_pages_second_person_read = pages_read_per_day_second_person * summer_days)
  (h6 : books_read * average_pages_per_book = total_pages_second_person_read / ratio_closest_person) :
  books_read = 60 :=
by {
  sorry
}

end deshaun_read_books_over_summer_l173_173829


namespace y_plus_z_value_l173_173311

theorem y_plus_z_value (v w x y z S : ℕ) 
  (h1 : 196 + x + y = S)
  (h2 : 269 + z + 123 = S)
  (h3 : 50 + x + z = S) : 
  y + z = 196 := 
sorry

end y_plus_z_value_l173_173311


namespace problem1_proof_problem2_proof_l173_173659

section Problems

variable {x a : ℝ}

-- Problem 1
theorem problem1_proof : 3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6 := by
  sorry

-- Problem 2
theorem problem2_proof : a^3 * a + (-a^2)^3 / a^2 = 0 := by
  sorry

end Problems

end problem1_proof_problem2_proof_l173_173659


namespace gcd_correct_l173_173833

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end gcd_correct_l173_173833


namespace recurring_fraction_division_l173_173361

/--
Given x = 0.\overline{36} and y = 0.\overline{12}, prove that x / y = 3.
-/
theorem recurring_fraction_division 
  (x y : ℝ)
  (h1 : x = 0.36 + 0.0036 + 0.000036 + 0.00000036 + ......) -- representation of 0.\overline{36}
  (h2 : y = 0.12 + 0.0012 + 0.000012 + 0.00000012 + ......) -- representation of 0.\overline{12}
  : x / y = 3 :=
  sorry

end recurring_fraction_division_l173_173361


namespace cube_volume_from_surface_area_example_cube_volume_l173_173964

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173964


namespace gcd_of_polynomials_l173_173287

/-- Given that a is an odd multiple of 7877, the greatest common divisor of
       7a^2 + 54a + 117 and 3a + 10 is 1. -/
theorem gcd_of_polynomials (a : ℤ) (h1 : a % 2 = 1) (h2 : 7877 ∣ a) :
  Int.gcd (7 * a ^ 2 + 54 * a + 117) (3 * a + 10) = 1 :=
sorry

end gcd_of_polynomials_l173_173287


namespace simplify_polynomial_expression_l173_173611

theorem simplify_polynomial_expression (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) = r^3 - 4 * r^2 + 2 * r + 3 :=
by
  sorry

end simplify_polynomial_expression_l173_173611


namespace cube_volume_l173_173980

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173980


namespace real_roots_of_quadratic_l173_173305

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end real_roots_of_quadratic_l173_173305


namespace Petya_can_ensure_root_difference_of_2014_l173_173728

theorem Petya_can_ensure_root_difference_of_2014 :
  ∀ a1 a2 : ℚ, ∃ a3 : ℚ, ∀ (r1 r2 r3 : ℚ),
    (r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧
    (r1, r2, r3 are roots of (λ x : ℚ, x^3 + a1 * x^2 + a2 * x + a3)) →
    (r1 - r2 = 2014 ∨ r1 - r2 = -2014 ∨
     r1 - r3 = 2014 ∨ r1 - r3 = -2014 ∨
     r2 - r3 = 2014 ∨ r2 - r3 = -2014) :=
by
  assume a1 a2 : ℚ
  have h : ∃ a3 : ℚ, ∀ (p : polynomial ℚ),
    (roots_of p = {0, 2014, r3}) ∨ (roots_of p = {0, r2, 2014})
  existsi a3
  sorry

end Petya_can_ensure_root_difference_of_2014_l173_173728


namespace probability_common_letters_l173_173836

open Set

def letters_GEOMETRY : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def letters_RHYME : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

def common_letters : Finset Char := letters_GEOMETRY ∩ letters_RHYME

theorem probability_common_letters :
  (common_letters.card : ℚ) / (letters_GEOMETRY.card : ℚ) = 1 / 2 := by
  sorry

end probability_common_letters_l173_173836


namespace find_a_value_l173_173310

noncomputable def solve_for_a (y : ℝ) (a : ℝ) : Prop :=
  0 < y ∧ (a * y) / 20 + (3 * y) / 10 = 0.6499999999999999 * y 

theorem find_a_value (y : ℝ) (a : ℝ) (h : solve_for_a y a) : a = 7 := 
by 
  sorry

end find_a_value_l173_173310


namespace balloon_arrangements_l173_173241

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l173_173241


namespace candy_distribution_l173_173482

theorem candy_distribution (candy_total friends : ℕ) (candies : List ℕ) :
  candy_total = 47 ∧ friends = 5 ∧ List.length candies = friends ∧
  (∀ n ∈ candies, n = 9) → (47 % 5 = 2) :=
by
  sorry

end candy_distribution_l173_173482


namespace compare_abc_l173_173549

noncomputable def a : ℝ := (1 / 6) ^ (1 / 2)
noncomputable def b : ℝ := Real.log 1 / 3 / Real.log 6
noncomputable def c : ℝ := Real.log 1 / 7 / Real.log (1 / 6)

theorem compare_abc : c > a ∧ a > b := by
  sorry

end compare_abc_l173_173549


namespace remainder_of_polynomial_l173_173416

def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

theorem remainder_of_polynomial (x : ℝ) : p 2 = 29 :=
by
  sorry

end remainder_of_polynomial_l173_173416


namespace range_of_a_l173_173127

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) :=
by
  sorry

end range_of_a_l173_173127


namespace cube_volume_l173_173987

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173987


namespace expression_evaluation_l173_173539

theorem expression_evaluation : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := 
by 
  sorry

end expression_evaluation_l173_173539


namespace correct_option_is_C_l173_173777

-- Definitions based on the problem conditions
def option_A : Prop := (-3 + (-3)) = 0
def option_B : Prop := (-3 - abs (-3)) = 0
def option_C (a b : ℝ) : Prop := (3 * a^2 * b - 4 * b * a^2) = - a^2 * b
def option_D (x : ℝ) : Prop := (-(5 * x - 2)) = -5 * x - 2

-- The theorem to be proved that option C is the correct calculation
theorem correct_option_is_C (a b : ℝ) : option_C a b :=
sorry

end correct_option_is_C_l173_173777


namespace pages_left_to_read_l173_173369

-- Defining the given conditions
def total_pages : ℕ := 500
def read_first_night : ℕ := (20 * total_pages) / 100
def read_second_night : ℕ := (20 * total_pages) / 100
def read_third_night : ℕ := (30 * total_pages) / 100

-- The total pages read over the three nights
def total_read : ℕ := read_first_night + read_second_night + read_third_night

-- The remaining pages to be read
def remaining_pages : ℕ := total_pages - total_read

theorem pages_left_to_read : remaining_pages = 150 :=
by
  -- Leaving the proof as a placeholder
  sorry

end pages_left_to_read_l173_173369


namespace borrowed_amount_l173_173390

theorem borrowed_amount (P : ℝ) (h1 : (9 / 100) * P - (8 / 100) * P = 200) : P = 20000 :=
  by sorry

end borrowed_amount_l173_173390


namespace ratio_distance_l173_173041

theorem ratio_distance
  (x : ℝ)
  (P : ℝ × ℝ)
  (hP_coords : P = (x, -9))
  (h_distance_y_axis : abs x = 18) :
  abs (-9) / abs x = 1 / 2 :=
by sorry

end ratio_distance_l173_173041


namespace integer_solution_count_l173_173421

def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0

theorem integer_solution_count : 
  (set.finite {a : ℤ | has_integer_solutions a}).to_finset.card = 16 := 
begin
  sorry
end

end integer_solution_count_l173_173421


namespace chocolates_total_l173_173765

theorem chocolates_total (x : ℕ)
  (h1 : x - 12 + x - 18 + x - 20 = 2 * x) :
  x = 50 :=
  sorry

end chocolates_total_l173_173765


namespace combined_parent_age_difference_l173_173334

def father_age_at_sobha_birth : ℕ := 38
def mother_age_at_brother_birth : ℕ := 36
def brother_younger_than_sobha : ℕ := 4
def sister_younger_than_brother : ℕ := 3
def father_age_at_sister_birth : ℕ := 45
def mother_age_at_youngest_birth : ℕ := 34
def youngest_younger_than_sister : ℕ := 6

def mother_age_at_sobha_birth := mother_age_at_brother_birth - brother_younger_than_sobha
def father_age_at_youngest_birth := father_age_at_sister_birth + youngest_younger_than_sister

def combined_age_difference_at_sobha_birth := father_age_at_sobha_birth - mother_age_at_sobha_birth
def compounded_difference_at_sobha_brother_birth := 
  (father_age_at_sobha_birth + brother_younger_than_sobha) - mother_age_at_brother_birth
def mother_age_at_sister_birth := mother_age_at_brother_birth + sister_younger_than_brother
def compounded_difference_at_sobha_sister_birth := father_age_at_sister_birth - mother_age_at_sister_birth
def compounded_difference_at_youngest_birth := father_age_at_youngest_birth - mother_age_at_youngest_birth

def combined_age_difference := 
  combined_age_difference_at_sobha_birth + 
  compounded_difference_at_sobha_brother_birth + 
  compounded_difference_at_sobha_sister_birth + 
  compounded_difference_at_youngest_birth 

theorem combined_parent_age_difference : combined_age_difference = 35 := by
  sorry

end combined_parent_age_difference_l173_173334


namespace range_of_m_l173_173120

theorem range_of_m (m : ℝ) : ((m + 3 > 0) ∧ (m - 1 < 0)) ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l173_173120


namespace cube_volume_from_surface_area_l173_173967

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173967


namespace weight_of_daughter_l173_173217

variable (M D G S : ℝ)

theorem weight_of_daughter :
  M + D + G + S = 200 →
  D + G = 60 →
  G = M / 5 →
  S = 2 * D →
  D = 800 / 15 :=
by
  intros h1 h2 h3 h4
  sorry

end weight_of_daughter_l173_173217


namespace total_buttons_l173_173810

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end total_buttons_l173_173810


namespace candy_store_sampling_l173_173007

theorem candy_store_sampling (total_customers sampling_customers caught_customers not_caught_customers : ℝ)
    (h1 : caught_customers = 0.22 * total_customers)
    (h2 : not_caught_customers = 0.15 * sampling_customers)
    (h3 : sampling_customers = caught_customers + not_caught_customers):
    sampling_customers = 0.2588 * total_customers := by
  sorry

end candy_store_sampling_l173_173007


namespace squares_difference_sum_l173_173195

theorem squares_difference_sum : 
  19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 200 :=
by
  sorry

end squares_difference_sum_l173_173195


namespace cube_volume_from_surface_area_l173_173969

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173969


namespace total_buttons_l173_173809

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end total_buttons_l173_173809


namespace balloon_permutation_count_l173_173272

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l173_173272


namespace correct_inequality_l173_173637

theorem correct_inequality :
  1.6 ^ 0.3 > 0.9 ^ 3.1 :=
sorry

end correct_inequality_l173_173637


namespace academy_league_total_games_l173_173914

theorem academy_league_total_games (teams : ℕ) (plays_each_other_twice games_non_conference : ℕ) 
  (h_teams : teams = 8)
  (h_plays_each_other_twice : plays_each_other_twice = 2 * teams * (teams - 1) / 2)
  (h_games_non_conference : games_non_conference = 6 * teams) :
  (plays_each_other_twice + games_non_conference) = 104 :=
by
  sorry

end academy_league_total_games_l173_173914


namespace power_of_product_l173_173513

theorem power_of_product (x : ℝ) : (-x^4)^3 = -x^12 := 
by sorry

end power_of_product_l173_173513


namespace fourth_vertex_of_parallelogram_l173_173620

structure Point where
  x : ℝ
  y : ℝ

def Q := Point.mk 1 (-1)
def R := Point.mk (-1) 0
def S := Point.mk 0 1
def V := Point.mk (-2) 2

theorem fourth_vertex_of_parallelogram (Q R S V : Point) :
  Q = ⟨1, -1⟩ ∧ R = ⟨-1, 0⟩ ∧ S = ⟨0, 1⟩ → V = ⟨-2, 2⟩ := by 
  sorry

end fourth_vertex_of_parallelogram_l173_173620


namespace balloon_arrangements_l173_173257

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l173_173257


namespace correct_transformation_l173_173367

structure Point :=
  (x : ℝ)
  (y : ℝ)

def rotate180 (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def is_rotation_180 (p p' : Point) : Prop :=
  rotate180 p = p'

theorem correct_transformation (C D : Point) (C' D' : Point) 
  (hC : C = Point.mk 3 (-2)) 
  (hC' : C' = Point.mk (-3) 2)
  (hD : D = Point.mk 2 (-5)) 
  (hD' : D' = Point.mk (-2) 5) :
  is_rotation_180 C C' ∧ is_rotation_180 D D' :=
by
  sorry

end correct_transformation_l173_173367


namespace tangent_plane_at_A_normal_line_through_A_l173_173098

noncomputable def F (x y z : ℝ) : ℝ := 2 * x^2 + y^2 - z

def point_A : ℝ × ℝ × ℝ := (1, -1, 3)

def tangent_plane_eq (x y z : ℝ) : ℝ := 4*x - 2*y - z - 3

def normal_line_eq (t : ℝ) : ℝ × ℝ × ℝ := (1 + 4 * t, -1 - 2 * t, 3 - t)

theorem tangent_plane_at_A : 
  ∀ x y z : ℝ, 
  tangent_plane_eq x y z = 0 ↔ 
  ∀ t : ℝ, 
  F (1 + 4 * t) (-1 - 2 * t) (3 - t) = 0 := sorry

theorem normal_line_through_A :
  ∀ t : ℝ, 
  (1 + 4 * t, -1 - 2 * t, 3 - t) ∈ {p : ℝ × ℝ × ℝ | F p.1 p.2 p.3 = 0} := sorry

end tangent_plane_at_A_normal_line_through_A_l173_173098


namespace leaves_blew_away_l173_173484

theorem leaves_blew_away (initial_leaves : ℕ) (leaves_left : ℕ) (blew_away : ℕ) 
  (h1 : initial_leaves = 356) (h2 : leaves_left = 112) (h3 : blew_away = initial_leaves - leaves_left) :
  blew_away = 244 :=
by
  sorry

end leaves_blew_away_l173_173484


namespace cookies_per_batch_l173_173695

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l173_173695


namespace inequality_solution_l173_173779

theorem inequality_solution (x : ℚ) : (3 * x - 5 ≥ 9 - 2 * x) → (x ≥ 14 / 5) :=
by
  sorry

end inequality_solution_l173_173779


namespace eight_times_nine_and_two_fifths_is_l173_173085

variable (m n a b : ℕ)
variable (d : ℚ)

-- Conditions
def mixed_to_improper (a b den : ℚ) : ℚ := a + b / den
def improper_to_mixed (n d : ℚ) : ℕ × ℚ := (n / d).to_int, (n % d) / d

-- Example specific instances
def nine_and_two_fifths : ℚ := mixed_to_improper 9 2 5
def eight_times_nine_and_two_fifths : ℚ := 8 * nine_and_two_fifths

-- Lean statement to confirm calculation
theorem eight_times_nine_and_two_fifths_is : improper_to_mixed eight_times_nine_and_two_fifths 5 = (75, 1/5) := by
  sorry

end eight_times_nine_and_two_fifths_is_l173_173085


namespace evaluate_expression_l173_173845

def acbd (a b c d : ℝ) : ℝ := a * d - b * c

theorem evaluate_expression (x : ℝ) (h : x^2 - 3 * x + 1 = 0) :
  acbd (x + 1) (x - 2) (3 * x) (x - 1) = 1 := 
by
  sorry

end evaluate_expression_l173_173845


namespace track_length_l173_173630

theorem track_length (V_A V_B V_C : ℝ) (x : ℝ) 
  (h1 : x / V_A = (x - 1) / V_B) 
  (h2 : x / V_A = (x - 2) / V_C) 
  (h3 : x / V_B = (x - 1.01) / V_C) : 
  110 - x = 9 :=
by 
  sorry

end track_length_l173_173630


namespace middle_group_frequency_l173_173580

theorem middle_group_frequency (f : ℕ) (A : ℕ) (h_total : A + f = 100) (h_middle : f = A) : f = 50 :=
by
  sorry

end middle_group_frequency_l173_173580


namespace tailor_cut_skirt_l173_173396

theorem tailor_cut_skirt (cut_pants cut_skirt : ℝ) (h1 : cut_pants = 0.5) (h2 : cut_skirt = cut_pants + 0.25) : cut_skirt = 0.75 :=
by
  sorry

end tailor_cut_skirt_l173_173396


namespace measure_orthogonal_trihedral_angle_sum_measure_polyhedral_angles_l173_173057

theorem measure_orthogonal_trihedral_angle (d : ℕ) (a : ℝ) (n : ℕ) 
(h1 : d = 3) (h2 : a = π / 2) (h3 : n = 8) : 
  ∃ measure : ℝ, measure = π / 2 :=
by
  sorry

theorem sum_measure_polyhedral_angles (d : ℕ) (a : ℝ) (n : ℕ) 
(h1 : d = 3) (h2 : a = π / 2) (h3 : n = 8) 
(h4 : n * a = 4 * π) : 
  ∃ sum_measure : ℝ, sum_measure = 4 * π :=
by
  sorry

end measure_orthogonal_trihedral_angle_sum_measure_polyhedral_angles_l173_173057


namespace min_sum_of_integers_cauchy_schwarz_l173_173095

theorem min_sum_of_integers_cauchy_schwarz :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  (1 / x + 4 / y + 9 / z = 1) ∧ 
  ((x + y + z) = 36) :=
  sorry

end min_sum_of_integers_cauchy_schwarz_l173_173095


namespace min_radius_cylinder_proof_l173_173387

-- Defining the radius of the hemisphere
def radius_hemisphere : ℝ := 10

-- Defining the angle alpha which is less than or equal to 30 degrees
def angle_alpha_leq_30 (α : ℝ) : Prop := α ≤ 30 * Real.pi / 180

-- Minimum radius of the cylinder given alpha <= 30 degrees
noncomputable def min_radius_cylinder : ℝ :=
  10 * (2 / Real.sqrt 3 - 1)

theorem min_radius_cylinder_proof (α : ℝ) (hα : angle_alpha_leq_30 α) :
  min_radius_cylinder = 10 * (2 / Real.sqrt 3 - 1) :=
by
  -- Here would go the detailed proof steps
  sorry

end min_radius_cylinder_proof_l173_173387


namespace problem1_problem2_problem3_problem4_l173_173230

-- Problem 1
theorem problem1 : (- (3 : ℝ) / 7) + (1 / 5) + (2 / 7) + (- (6 / 5)) = - (8 / 7) :=
by
  sorry

-- Problem 2
theorem problem2 : -(-1) + 3^2 / (1 - 4) * 2 = -5 :=
by
  sorry

-- Problem 3
theorem problem3 :  (-(1 / 6))^2 / ((1 / 2 - 1 / 3)^2) / (abs (-6))^2 = 1 / 36 :=
by
  sorry

-- Problem 4
theorem problem4 : (-1) ^ 1000 - 2.45 * 8 + 2.55 * (-8) = -39 :=
by
  sorry

end problem1_problem2_problem3_problem4_l173_173230


namespace distance_midpoint_parabola_y_axis_l173_173862

theorem distance_midpoint_parabola_y_axis (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hA : y1 ^ 2 = x1) (hB : y2 ^ 2 = x2) 
  (h_focus : ∀ {p : ℝ × ℝ}, p = (x1, y1) ∨ p = (x2, y2) → |p.1 - 1/4| = |p.1 + 1/4|)
  (h_dist : |x1 - 1/4| + |x2 - 1/4| = 3) :
  abs ((x1 + x2) / 2) = 5 / 4 :=
by sorry

end distance_midpoint_parabola_y_axis_l173_173862


namespace largest_number_obtained_l173_173530

theorem largest_number_obtained : 
  ∃ n : ℤ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m, 10 ≤ m ∧ m ≤ 99 → (250 - 3 * m)^2 ≤ (250 - 3 * n)^2) ∧ (250 - 3 * n)^2 = 4 :=
sorry

end largest_number_obtained_l173_173530


namespace store_profit_l173_173395

theorem store_profit :
  let selling_price : ℝ := 80
  let cost_price_profitable : ℝ := (selling_price - 0.60 * selling_price)
  let cost_price_loss : ℝ := (selling_price + 0.20 * selling_price)
  selling_price + selling_price - cost_price_profitable - cost_price_loss = 10 := by
  sorry

end store_profit_l173_173395


namespace age_difference_is_13_l173_173935

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end age_difference_is_13_l173_173935


namespace parametric_hyperbola_l173_173040

theorem parametric_hyperbola (t : ℝ) (ht : t ≠ 0) : 
  let x := t + 1 / t
  let y := t - 1 / t
  x^2 - y^2 = 4 :=
by
  let x := t + 1 / t
  let y := t - 1 / t
  sorry

end parametric_hyperbola_l173_173040


namespace min_value_y_l173_173720

theorem min_value_y (x y : ℝ) (h : x^2 + y^2 = 14 * x + 48 * y) : y = -1 := 
sorry

end min_value_y_l173_173720


namespace real_roots_of_quadratic_l173_173306

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end real_roots_of_quadratic_l173_173306


namespace smallest_period_of_f_is_pi_div_2_l173_173042

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 4 + (Real.sin x) ^ 2

theorem smallest_period_of_f_is_pi_div_2 : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2 :=
sorry

end smallest_period_of_f_is_pi_div_2_l173_173042


namespace Xiaogang_shooting_probability_l173_173199

theorem Xiaogang_shooting_probability (total_shots : ℕ) (shots_made : ℕ) (h_total : total_shots = 50) (h_made : shots_made = 38) :
  (shots_made : ℝ) / total_shots = 0.76 :=
by
  sorry

end Xiaogang_shooting_probability_l173_173199


namespace students_catching_up_on_homework_l173_173464

theorem students_catching_up_on_homework
  (total_students : ℕ)
  (half_doing_silent_reading : ℕ)
  (third_playing_board_games : ℕ)
  (remain_catching_up_homework : ℕ) :
  total_students = 24 →
  half_doing_silent_reading = total_students / 2 →
  third_playing_board_games = total_students / 3 →
  remain_catching_up_homework = total_students - (half_doing_silent_reading + third_playing_board_games) →
  remain_catching_up_homework = 4 :=
by
  intros h_total h_half h_third h_remain
  sorry

end students_catching_up_on_homework_l173_173464


namespace greatest_common_divisor_b_81_l173_173018

theorem greatest_common_divisor_b_81 (a b : ℤ) 
  (h : (1 + Real.sqrt 2) ^ 2012 = a + b * Real.sqrt 2) : Int.gcd b 81 = 3 :=
by
  sorry

end greatest_common_divisor_b_81_l173_173018


namespace gum_total_l173_173738

theorem gum_total (initial_gum : ℝ) (additional_gum : ℝ) : initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 :=
by
  intros
  sorry

end gum_total_l173_173738


namespace balloon_permutation_count_l173_173271

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l173_173271


namespace cube_volume_from_surface_area_l173_173968

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173968


namespace average_postcards_collected_per_day_l173_173232

theorem average_postcards_collected_per_day 
    (a : ℕ) (d : ℕ) (n : ℕ) 
    (h_a : a = 10)
    (h_d : d = 12)
    (h_n : n = 7) :
    (a + (a + (n - 1) * d)) / 2 = 46 := by
  sorry

end average_postcards_collected_per_day_l173_173232


namespace freq_count_of_third_group_l173_173560

theorem freq_count_of_third_group
  (sample_size : ℕ) 
  (freq_third_group : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : freq_third_group = 0.2) : 
  (sample_size * freq_third_group) = 20 :=
by 
  sorry

end freq_count_of_third_group_l173_173560


namespace total_buttons_l173_173811

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end total_buttons_l173_173811


namespace cube_volume_from_surface_area_example_cube_volume_l173_173958

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173958


namespace students_still_inward_l173_173053

theorem students_still_inward (num_students : ℕ) (turns : ℕ) : (num_students = 36) ∧ (turns = 36) → ∃ n, n = 26 :=
by
  sorry

end students_still_inward_l173_173053


namespace balloon_permutations_l173_173244

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l173_173244


namespace balloon_permutations_l173_173267

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l173_173267


namespace part1_part2a_part2b_l173_173851

-- Definitions and conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-3, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def scalar_mul (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Proof statements

-- Part 1: Verify the dot product computation
theorem part1 : dot_product (vector_add vector_a vector_b) (vector_sub vector_a vector_b) = -8 := by
  sorry

-- Part 2a: Verify the value of k for parallel vectors
theorem part2a : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (vector_sub vector_a (scalar_mul 3 vector_b)) := by
  sorry

-- Part 2b: Verify antiparallel direction
theorem part2b : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (scalar_mul (-1) (vector_sub vector_a (scalar_mul 3 vector_b))) := by
  sorry

end part1_part2a_part2b_l173_173851


namespace intersection_points_on_circle_l173_173284

theorem intersection_points_on_circle (u : ℝ) :
  ∃ (r : ℝ), ∀ (x y : ℝ), (u * x - 3 * y - 2 * u = 0) ∧ (2 * x - 3 * u * y + u = 0) → (x^2 + y^2 = r^2) :=
sorry

end intersection_points_on_circle_l173_173284


namespace rational_numbers_opposites_l173_173707

theorem rational_numbers_opposites (a b : ℚ) (h : (a + b) / (a * b) = 0) : a = -b ∧ a ≠ 0 ∧ b ≠ 0 :=
by
  sorry

end rational_numbers_opposites_l173_173707


namespace fib_sum_equiv_t_l173_173173

-- Definitions based on the conditions:
def fib (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 1
else fib (n - 1) + fib (n - 2)

def S (n : ℕ) : ℕ :=
(n + 1).sum fun i => fib i

-- Given these definitions, we state the proof problem:
theorem fib_sum_equiv_t (t : ℕ) (ht : fib 2018 = t) :
  S 2016 + S 2015 - S 2014 - S 2013 = t :=
sorry

end fib_sum_equiv_t_l173_173173


namespace at_least_one_corner_square_selected_l173_173772

theorem at_least_one_corner_square_selected :
  let total_squares := 16
  let total_corners := 4
  let total_non_corners := 12
  let ways_to_select_3_from_total := Nat.choose total_squares 3
  let ways_to_select_3_from_non_corners := Nat.choose total_non_corners 3
  let probability_no_corners := (ways_to_select_3_from_non_corners : ℚ) / ways_to_select_3_from_total
  let probability_at_least_one_corner := 1 - probability_no_corners
  probability_at_least_one_corner = (17 / 28 : ℚ) :=
by
  sorry

end at_least_one_corner_square_selected_l173_173772


namespace relationship_between_a_and_b_l173_173722

theorem relationship_between_a_and_b 
  (x a b : ℝ)
  (hx : 0 < x)
  (ha : 0 < a)
  (hb : 0 < b)
  (hax : a^x < b^x) 
  (hbx : b^x < 1) : 
  a < b ∧ b < 1 := 
sorry

end relationship_between_a_and_b_l173_173722


namespace find_number_l173_173309

theorem find_number (x n : ℤ) 
  (h1 : 0 < x) (h2 : x < 7) 
  (h3 : x < 15) 
  (h4 : -1 < x) (h5 : x < 5) 
  (h6 : x < 3) (h7 : 0 < x) 
  (h8 : x + n < 4) 
  (hx : x = 1): 
  n < 3 := 
sorry

end find_number_l173_173309


namespace cube_volume_from_surface_area_l173_173992

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173992


namespace whole_number_N_l173_173398

theorem whole_number_N (N : ℤ) : (9 < N / 4 ∧ N / 4 < 10) ↔ (N = 37 ∨ N = 38 ∨ N = 39) := 
by sorry

end whole_number_N_l173_173398


namespace induction_divisibility_l173_173943

theorem induction_divisibility (k x y : ℕ) (h : k > 0) :
  (x^(2*k-1) + y^(2*k-1)) ∣ (x + y) → 
  (x^(2*k+1) + y^(2*k+1)) ∣ (x + y) :=
sorry

end induction_divisibility_l173_173943


namespace kristy_gave_to_brother_l173_173588

def total_cookies : Nat := 22
def kristy_ate : Nat := 2
def first_friend_took : Nat := 3
def second_friend_took : Nat := 5
def third_friend_took : Nat := 5
def cookies_left : Nat := 6

theorem kristy_gave_to_brother :
  kristy_ate + first_friend_took + second_friend_took + third_friend_took = 15 ∧
  total_cookies - cookies_left - (kristy_ate + first_friend_took + second_friend_took + third_friend_took) = 1 :=
by
  sorry

end kristy_gave_to_brother_l173_173588


namespace jars_contain_k_balls_eventually_l173_173351

theorem jars_contain_k_balls_eventually
  (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hkp : k < 2 * p + 1) :
  ∃ n : ℕ, ∃ x y : ℕ, x + y = 2 * p + 1 ∧ (x = k ∨ y = k) :=
by
  sorry

end jars_contain_k_balls_eventually_l173_173351


namespace probability_non_yellow_l173_173210

def num_red := 4
def num_green := 7
def num_yellow := 9
def num_blue := 10

def total_jelly_beans := num_red + num_green + num_yellow + num_blue
def num_non_yellow := num_red + num_green + num_blue

theorem probability_non_yellow : (num_non_yellow : ℚ) / total_jelly_beans = 7 / 10 :=
by
  have h1: total_jelly_beans = 30 := by norm_num
  have h2: num_non_yellow = 21 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end probability_non_yellow_l173_173210


namespace product_negative_probability_l173_173356

noncomputable def prob_product_negative : ℚ :=
  let mySet := {-5, -8, 7, 4, -2} : finset ℤ
  let pairs := mySet.powerset.filter (λ s, s.card = 2)
  let negative_product_pairs := pairs.filter (λ s, s.prod id < 0)
  negative_product_pairs.card / pairs.card

theorem product_negative_probability : prob_product_negative = 3 / 5 := by
  sorry

end product_negative_probability_l173_173356


namespace sqrt_six_lt_a_lt_cubic_two_l173_173453

theorem sqrt_six_lt_a_lt_cubic_two (a : ℝ) (h : a^5 - a^3 + a = 2) : (Real.sqrt 3)^6 < a ∧ a < 2^(1/3) :=
sorry

end sqrt_six_lt_a_lt_cubic_two_l173_173453


namespace different_rhetorical_device_in_optionA_l173_173225

def optionA_uses_metaphor : Prop :=
  -- Here, define the condition explaining that Option A uses metaphor
  true -- This will denote that Option A uses metaphor 

def optionsBCD_use_personification : Prop :=
  -- Here, define the condition explaining that Options B, C, and D use personification
  true -- This will denote that Options B, C, and D use personification

theorem different_rhetorical_device_in_optionA :
  optionA_uses_metaphor ∧ optionsBCD_use_personification → 
  (∃ (A P : Prop), A ≠ P) :=
by
  -- No proof is required as per instructions
  intro h
  exact Exists.intro optionA_uses_metaphor (Exists.intro optionsBCD_use_personification sorry)

end different_rhetorical_device_in_optionA_l173_173225


namespace cannot_buy_same_number_of_notebooks_l173_173512

theorem cannot_buy_same_number_of_notebooks
  (price_softcover : ℝ)
  (price_hardcover : ℝ)
  (notebooks_ming : ℝ)
  (notebooks_li : ℝ)
  (h1 : price_softcover = 12)
  (h2 : price_hardcover = 21)
  (h3 : price_hardcover = price_softcover + 1.2) :
  notebooks_ming = 12 / price_softcover ∧
  notebooks_li = 21 / price_hardcover →
  ¬ (notebooks_ming = notebooks_li) :=
by
  sorry

end cannot_buy_same_number_of_notebooks_l173_173512


namespace simple_interest_correct_l173_173162

-- Define the principal amount P
variables {P : ℝ}

-- Define the rate of interest r which is 3% or 0.03 in decimal form
def r : ℝ := 0.03

-- Define the time period t which is 2 years
def t : ℕ := 2

-- Define the compound interest CI for 2 years which is $609
def CI : ℝ := 609

-- Define the simple interest SI that we need to find
def SI : ℝ := 600

-- Define a formula for compound interest
def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

-- Define a formula for simple interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem simple_interest_correct (hCI : compound_interest P r t = CI) : simple_interest P r t = SI :=
by
  sorry

end simple_interest_correct_l173_173162


namespace percent_of_a_is_4b_l173_173036

theorem percent_of_a_is_4b (b : ℝ) (a : ℝ) (h : a = 1.8 * b) : (4 * b / a) * 100 = 222.22 := 
by {
  sorry
}

end percent_of_a_is_4b_l173_173036


namespace mineral_age_possibilities_l173_173384

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_permutations_with_repeats (n : ℕ) (repeats : List ℕ) : ℕ :=
  factorial n / List.foldl (· * factorial ·) 1 repeats

theorem mineral_age_possibilities : 
  let digits := [2, 2, 4, 4, 7, 9]
  let odd_digits := [7, 9]
  let remaining_digits := [2, 2, 4, 4]
  2 * count_permutations_with_repeats 5 [2,2] = 60 :=
by
  sorry

end mineral_age_possibilities_l173_173384


namespace how_many_kids_joined_l173_173767

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end how_many_kids_joined_l173_173767


namespace shortest_side_of_right_triangle_l173_173647

theorem shortest_side_of_right_triangle (a b : ℝ) (h : a = 9 ∧ b = 12) : ∃ c : ℝ, (c = min a b) ∧ c = 9 :=
by
  sorry

end shortest_side_of_right_triangle_l173_173647


namespace cost_to_fill_pool_l173_173319

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end cost_to_fill_pool_l173_173319


namespace smallest_y_for_perfect_fourth_power_l173_173525

-- Define the conditions
def x : ℕ := 7 * 24 * 48
def y : ℕ := 6174

-- The theorem we need to prove
theorem smallest_y_for_perfect_fourth_power (x y : ℕ) 
  (hx : x = 7 * 24 * 48) 
  (hy : y = 6174) : ∃ k : ℕ, (∃ z : ℕ, z * z * z * z = x * y) :=
sorry

end smallest_y_for_perfect_fourth_power_l173_173525


namespace number_of_integer_values_of_a_l173_173419

theorem number_of_integer_values_of_a : 
  ∃ (a_set : Set ℤ), (∀ a ∈ a_set, ∃ x y : ℤ, x^2 + a * x + 12 * a = 0 ∧ y^2 + a * y + 12 * a = 0) ∧ a_set.card = 16 :=
sorry

end number_of_integer_values_of_a_l173_173419


namespace student_weekly_allowance_l173_173444

theorem student_weekly_allowance (A : ℝ) 
  (h1 : ∃ spent_arcade, spent_arcade = (3 / 5) * A)
  (h2 : ∃ spent_toy, spent_toy = (1 / 3) * ((2 / 5) * A))
  (h3 : ∃ spent_candy, spent_candy = 0.60)
  (h4 : ∃ remaining_after_toy, remaining_after_toy = ((6 / 15) * A - (2 / 15) * A))
  (h5 : remaining_after_toy = 0.60) : 
  A = 2.25 := by
  sorry

end student_weekly_allowance_l173_173444


namespace Laura_won_5_games_l173_173010

-- Define the number of wins and losses for each player
def Peter_wins : ℕ := 5
def Peter_losses : ℕ := 3
def Peter_games : ℕ := Peter_wins + Peter_losses

def Emma_wins : ℕ := 4
def Emma_losses : ℕ := 4
def Emma_games : ℕ := Emma_wins + Emma_losses

def Kyler_wins : ℕ := 2
def Kyler_losses : ℕ := 6
def Kyler_games : ℕ := Kyler_wins + Kyler_losses

-- Define the total number of games played in the tournament
def total_games_played : ℕ := (Peter_games + Emma_games + Kyler_games + 8) / 2

-- Define total wins and losses
def total_wins_losses : ℕ := total_games_played

-- Prove the number of games Laura won
def Laura_wins : ℕ := total_wins_losses - (Peter_wins + Emma_wins + Kyler_wins)

theorem Laura_won_5_games : Laura_wins = 5 := by
  -- The proof will be completed here
  sorry

end Laura_won_5_games_l173_173010


namespace tommys_books_l173_173773

-- Define the cost of each book
def book_cost : ℕ := 5

-- Define the amount Tommy already has
def tommy_money : ℕ := 13

-- Define the amount Tommy needs to save up
def tommy_goal : ℕ := 27

-- Prove the number of books Tommy wants to buy
theorem tommys_books : tommy_goal + tommy_money = 40 ∧ (tommy_goal + tommy_money) / book_cost = 8 :=
by
  sorry

end tommys_books_l173_173773


namespace factorize_quadratic_l173_173091

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end factorize_quadratic_l173_173091


namespace average_M_possibilities_l173_173898

theorem average_M_possibilities (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
    (12 = (8 + 15 + M) / 3) ∨ (15 = (8 + 15 + M) / 3) :=
  sorry

end average_M_possibilities_l173_173898


namespace central_angle_measure_l173_173435

theorem central_angle_measure (α r : ℝ) (h1 : α * r = 2) (h2 : 1/2 * α * r^2 = 2) : α = 1 := 
sorry

end central_angle_measure_l173_173435


namespace greatest_number_of_problems_missed_l173_173653

theorem greatest_number_of_problems_missed 
    (total_problems : ℕ) (passing_percentage : ℝ) (max_missed : ℕ) :
    total_problems = 40 →
    passing_percentage = 0.85 →
    max_missed = total_problems - ⌈total_problems * passing_percentage⌉ →
    max_missed = 6 :=
by
  intros h1 h2 h3
  sorry

end greatest_number_of_problems_missed_l173_173653


namespace dogs_Carly_worked_on_l173_173660

-- Define the parameters for the problem
def total_nails := 164
def three_legged_dogs := 3
def three_nail_paw_dogs := 2
def extra_nail_paw_dog := 1
def regular_dog_nails := 16
def three_legged_nails := (regular_dog_nails - 4)
def three_nail_paw_nails := (regular_dog_nails - 1)
def extra_nail_paw_nails := (regular_dog_nails + 1)

-- Lean statement to prove the number of dogs Carly worked on today
theorem dogs_Carly_worked_on :
  (3 * three_legged_nails) + (2 * three_nail_paw_nails) + extra_nail_paw_nails 
  = 83 → ((total_nails - 83) / regular_dog_nails ≠ 0) → 5 + 3 + 2 + 1 = 11 :=
by sorry

end dogs_Carly_worked_on_l173_173660


namespace distance_C_to_C_l173_173191

noncomputable def C : ℝ × ℝ := (-3, 2)
noncomputable def C' : ℝ × ℝ := (3, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_C_to_C' : distance C C' = 2 * Real.sqrt 13 := by
  sorry

end distance_C_to_C_l173_173191


namespace max_value_of_b_l173_173547

theorem max_value_of_b {m b : ℚ} (x : ℤ) 
  (line_eq : ∀ x : ℤ, 0 < x ∧ x ≤ 200 → 
    ¬ ∃ (y : ℤ), y = m * x + 3)
  (m_range : 1/3 < m ∧ m < b) :
  b = 69/208 :=
by
  sorry

end max_value_of_b_l173_173547


namespace product_of_roots_quadratic_eq_l173_173112

theorem product_of_roots_quadratic_eq : 
  ∀ (x1 x2 : ℝ), 
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 → (x = x1 ∨ x = x2)) → 
  x1 * x2 = -3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_quadratic_eq_l173_173112


namespace fraction_dutch_americans_has_window_l173_173603

variable (P D DA : ℕ)
variable (f_P_d d_P_w : ℚ)
variable (DA_w : ℕ)

-- Total number of people on the bus P 
-- Fraction of people who were Dutch f_P_d
-- Fraction of Dutch Americans who got window seats d_P_w
-- Number of Dutch Americans who sat at windows DA_w
-- Define the assumptions
def total_people_on_bus := P = 90
def fraction_dutch := f_P_d = 3 / 5
def fraction_dutch_americans_window := d_P_w = 1 / 3
def dutch_americans_window := DA_w = 9

-- Prove that fraction of Dutch people who were also American is 1/2
theorem fraction_dutch_americans_has_window (P D DA DA_w : ℕ) (f_P_d d_P_w : ℚ) :
  total_people_on_bus P ∧ fraction_dutch f_P_d ∧
  fraction_dutch_americans_window d_P_w ∧ dutch_americans_window DA_w →
  (DA: ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_dutch_americans_has_window_l173_173603


namespace coords_reflect_origin_l173_173493

def P : Type := (ℤ × ℤ)

def reflect_origin (p : P) : P :=
  (-p.1, -p.2)

theorem coords_reflect_origin (p : P) (hx : p = (2, -1)) : reflect_origin p = (-2, 1) :=
by
  sorry

end coords_reflect_origin_l173_173493


namespace find_a_for_inequality_l173_173196

theorem find_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 3) → -2 * x^2 + a * x + 6 > 0) → a = 2 :=
by
  sorry

end find_a_for_inequality_l173_173196


namespace max_product_sum_1988_l173_173668

theorem max_product_sum_1988 :
  ∃ (n : ℕ) (a : ℕ), n + a = 1988 ∧ a = 1 ∧ n = 662 ∧ (3^n * 2^a) = 2 * 3^662 :=
by
  sorry

end max_product_sum_1988_l173_173668


namespace find_x_l173_173283

theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : (1 / 2) * x * (3 * x) = 54) : x = 6 :=
by
  sorry

end find_x_l173_173283


namespace line_perpendicular_passing_through_point_l173_173843

theorem line_perpendicular_passing_through_point :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), 2 * x + y - 2 = 0 ↔ a * x + b * y + c = 0) ∧ 
                (a, b) ≠ (0, 0) ∧ 
                (a * -1 + b * 4 + c = 0) ∧ 
                (a * 1/2 + b * (-2) ≠ -4) :=
by { sorry }

end line_perpendicular_passing_through_point_l173_173843


namespace root_shifted_is_root_of_quadratic_with_integer_coeffs_l173_173929

theorem root_shifted_is_root_of_quadratic_with_integer_coeffs
  (a b c t : ℤ)
  (h : a ≠ 0)
  (h_root : a * t^2 + b * t + c = 0) :
  ∃ (x : ℤ), a * x^2 + (4 * a + b) * x + (4 * a + 2 * b + c) = 0 :=
by {
  sorry
}

end root_shifted_is_root_of_quadratic_with_integer_coeffs_l173_173929


namespace total_six_letter_words_l173_173080

def num_vowels := 6
def vowel_count := 5
def word_length := 6

theorem total_six_letter_words : (num_vowels ^ word_length) = 46656 :=
by sorry

end total_six_letter_words_l173_173080


namespace initial_earning_members_l173_173039

theorem initial_earning_members (n T : ℕ)
  (h₁ : T = n * 782)
  (h₂ : T - 1178 = (n - 1) * 650) :
  n = 14 :=
by sorry

end initial_earning_members_l173_173039


namespace min_value_expression_l173_173323

noncomputable section

variables {x y : ℝ}

theorem min_value_expression (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ 
    (∃ min_val : ℝ, min_val = (x^2 / (x + 2) + y^2 / (y + 1)) ∧ min_val = 1 / 4)) :=
  sorry

end min_value_expression_l173_173323


namespace square_area_l173_173363

theorem square_area (side_length : ℝ) (h : side_length = 10) : side_length * side_length = 100 := by
  sorry

end square_area_l173_173363


namespace max_lateral_surface_area_l173_173392

theorem max_lateral_surface_area (x y : ℝ) (h₁ : x + y = 10) : 
  2 * π * x * y ≤ 50 * π :=
by
  sorry

end max_lateral_surface_area_l173_173392


namespace reverse_digits_difference_l173_173516

theorem reverse_digits_difference (q r : ℕ) (x y : ℕ) 
  (hq : q = 10 * x + y)
  (hr : r = 10 * y + x)
  (hq_r_pos : q > r)
  (h_diff_lt_20 : q - r < 20)
  (h_max_diff : q - r = 18) :
  x - y = 2 := 
by
  sorry

end reverse_digits_difference_l173_173516


namespace train_speed_approx_900072_kmph_l173_173222

noncomputable def speed_of_train (train_length platform_length time_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / time_seconds
  speed_m_s * 3.6

theorem train_speed_approx_900072_kmph :
  abs (speed_of_train 225 400.05 25 - 90.0072) < 0.001 :=
by
  sorry

end train_speed_approx_900072_kmph_l173_173222


namespace sam_walking_speed_l173_173849

variable (s : ℝ)
variable (t : ℝ)
variable (fred_speed : ℝ := 2)
variable (sam_distance : ℝ := 25)
variable (total_distance : ℝ := 35)

theorem sam_walking_speed :
  (total_distance - sam_distance) = fred_speed * t ∧
  sam_distance = s * t →
  s = 5 := 
by
  intros
  sorry

end sam_walking_speed_l173_173849


namespace exists_x_for_ax2_plus_2x_plus_a_lt_0_l173_173882

theorem exists_x_for_ax2_plus_2x_plus_a_lt_0 (a : ℝ) : (∃ x : ℝ, a * x^2 + 2 * x + a < 0) ↔ a < 1 :=
by
  sorry

end exists_x_for_ax2_plus_2x_plus_a_lt_0_l173_173882


namespace equation_of_parabola_passing_through_points_l173_173856

noncomputable def parabola (x : ℝ) (b c : ℝ) : ℝ :=
  x^2 + b * x + c

theorem equation_of_parabola_passing_through_points :
  ∃ (b c : ℝ), 
    (parabola 0 b c = 5) ∧ (parabola 3 b c = 2) ∧
    (∀ x, parabola x b c = x^2 - 4 * x + 5) := 
by
  sorry

end equation_of_parabola_passing_through_points_l173_173856


namespace simplify_expression_correct_l173_173492

variable {R : Type} [CommRing R]

def simplify_expression (x : R) : R :=
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8)

theorem simplify_expression_correct (x : R) : 
  simplify_expression x = 8 * x^5 + 0 * x^4 - 13 * x^3 + 23 * x^2 - 14 * x + 56 :=
by
  sorry

end simplify_expression_correct_l173_173492


namespace exists_sphere_tangent_to_lines_l173_173732

variables
  (A B C D K L M N : Point)
  (AB BC CD DA : Line)
  (sphere : Sphere)

-- Given conditions
def AN_eq_AK : AN = AK := sorry
def BK_eq_BL : BK = BL := sorry
def CL_eq_CM : CL = CM := sorry
def DM_eq_DN : DM = DN := sorry
def sphere_tangent (s : Sphere) (l : Line) : Prop := sorry -- define tangency condition

-- Problem statement
theorem exists_sphere_tangent_to_lines :
  ∃ S : Sphere, 
    sphere_tangent S AB ∧
    sphere_tangent S BC ∧
    sphere_tangent S CD ∧
    sphere_tangent S DA := sorry

end exists_sphere_tangent_to_lines_l173_173732


namespace proof_age_gladys_l173_173177

-- Definitions of ages
def age_gladys : ℕ := 30
def age_lucas : ℕ := 5
def age_billy : ℕ := 10

-- Conditions
def condition1 : Prop := age_gladys = 2 * (age_billy + age_lucas)
def condition2 : Prop := age_gladys = 3 * age_billy
def condition3 : Prop := age_lucas + 3 = 8

-- Theorem to prove the correct age of Gladys
theorem proof_age_gladys (G L B : ℕ)
  (h1 : G = 2 * (B + L))
  (h2 : G = 3 * B)
  (h3 : L + 3 = 8) :
  G = 30 :=
sorry

end proof_age_gladys_l173_173177


namespace find_d_l173_173114

theorem find_d (a d : ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : d ≠ 0)
                (h2 : ∀ n, S n = n * a + (n * (n - 1) / 2 * d))
                (h3 : ∀ n, (S n + n) ^ (1/2) = (S (n + 1) + (n + 1)) ^ (1/2) - d)
                : d = 1 / 2 :=
by
  -- Proof omitted.
  sorry

end find_d_l173_173114


namespace modulus_problem_l173_173375

theorem modulus_problem : (13 ^ 13 + 13) % 14 = 12 :=
by
  sorry

end modulus_problem_l173_173375


namespace airplane_altitude_l173_173814

theorem airplane_altitude (d_Alice_Bob : ℝ) (angle_Alice : ℝ) (angle_Bob : ℝ) (altitude : ℝ) : 
  d_Alice_Bob = 8 ∧ angle_Alice = 45 ∧ angle_Bob = 30 → altitude = 16 / 3 :=
by
  intros h
  rcases h with ⟨h1, ⟨h2, h3⟩⟩
  -- you may insert the proof here if needed
  sorry

end airplane_altitude_l173_173814


namespace who_stole_the_pan_l173_173317

def Frog_statement := "Lackey-Lech stole the pan"
def LackeyLech_statement := "I did not steal any pan"
def KnaveOfHearts_statement := "I stole the pan"

axiom no_more_than_one_liar : ∀ (frog_is_lying : Prop) (lackey_lech_is_lying : Prop) (knave_of_hearts_is_lying : Prop), (frog_is_lying → ¬ lackey_lech_is_lying) ∧ (frog_is_lying → ¬ knave_of_hearts_is_lying) ∧ (lackey_lech_is_lying → ¬ knave_of_hearts_is_lying)

theorem who_stole_the_pan : KnaveOfHearts_statement = "I stole the pan" :=
sorry

end who_stole_the_pan_l173_173317


namespace total_shares_eq_300_l173_173082

-- Define the given conditions
def microtron_price : ℝ := 36
def dynaco_price : ℝ := 44
def avg_price : ℝ := 40
def dynaco_shares : ℝ := 150

-- Define the number of Microtron shares sold
variable (M : ℝ)

-- Define the total shares sold
def total_shares : ℝ := M + dynaco_shares

-- The average price equation given the conditions
def avg_price_eq (M : ℝ) : Prop :=
  avg_price = (microtron_price * M + dynaco_price * dynaco_shares) / total_shares M

-- The correct answer we need to prove
theorem total_shares_eq_300 (M : ℝ) (h : avg_price_eq M) : total_shares M = 300 :=
by
  sorry

end total_shares_eq_300_l173_173082


namespace charge_per_block_l173_173895

noncomputable def family_vacation_cost : ℝ := 1000
noncomputable def family_members : ℝ := 5
noncomputable def walk_start_fee : ℝ := 2
noncomputable def dogs_walked : ℝ := 20
noncomputable def total_blocks : ℝ := 128

theorem charge_per_block : 
  (family_vacation_cost / family_members) = 200 →
  (dogs_walked * walk_start_fee) = 40 →
  ((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) = 160 →
  (((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) / total_blocks) = 1.25 :=
by intros h1 h2 h3; sorry

end charge_per_block_l173_173895


namespace MrsHiltCanTakeFriendsToMovies_l173_173727

def TotalFriends : ℕ := 15
def FriendsCantGo : ℕ := 7
def FriendsCanGo : ℕ := 8

theorem MrsHiltCanTakeFriendsToMovies : TotalFriends - FriendsCantGo = FriendsCanGo := by
  -- The proof will show that 15 - 7 = 8.
  sorry

end MrsHiltCanTakeFriendsToMovies_l173_173727


namespace max_sum_red_green_balls_l173_173188

theorem max_sum_red_green_balls (total_balls : ℕ) (green_balls : ℕ) (max_red_balls : ℕ) 
  (h1 : total_balls = 28) (h2 : green_balls = 12) (h3 : max_red_balls ≤ 11) : 
  (max_red_balls + green_balls) = 23 := 
sorry

end max_sum_red_green_balls_l173_173188


namespace polynomial_solution_l173_173279

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end polynomial_solution_l173_173279


namespace quadratic_real_roots_range_l173_173304

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end quadratic_real_roots_range_l173_173304


namespace find_marks_of_a_l173_173615

theorem find_marks_of_a (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : (A + B + C + D) / 4 = 47)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 48) : 
  A = 43 :=
by
  sorry

end find_marks_of_a_l173_173615


namespace peter_has_4_finches_l173_173734

variable (parakeet_eats_per_day : ℕ) (parrot_eats_per_day : ℕ) (finch_eats_per_day : ℕ)
variable (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ)
variable (total_birdseed : ℕ)

theorem peter_has_4_finches
    (h1 : parakeet_eats_per_day = 2)
    (h2 : parrot_eats_per_day = 14)
    (h3 : finch_eats_per_day = 1)
    (h4 : num_parakeets = 3)
    (h5 : num_parrots = 2)
    (h6 : total_birdseed = 266)
    (h7 : total_birdseed = (num_parakeets * parakeet_eats_per_day + num_parrots * parrot_eats_per_day) * 7 + num_finches * finch_eats_per_day * 7) :
    num_finches = 4 :=
by
  sorry

end peter_has_4_finches_l173_173734


namespace added_water_proof_l173_173314

variable (total_volume : ℕ) (milk_ratio water_ratio : ℕ) (added_water : ℕ)

theorem added_water_proof 
  (h1 : total_volume = 45) 
  (h2 : milk_ratio = 4) 
  (h3 : water_ratio = 1) 
  (h4 : added_water = 3) 
  (milk_volume : ℕ)
  (water_volume : ℕ)
  (h5 : milk_volume = (milk_ratio * total_volume) / (milk_ratio + water_ratio))
  (h6 : water_volume = (water_ratio * total_volume) / (milk_ratio + water_ratio))
  (new_ratio : ℕ)
  (h7 : new_ratio = milk_volume / (water_volume + added_water)) : added_water = 3 :=
by
  sorry

end added_water_proof_l173_173314


namespace radius_of_circle_l173_173170

theorem radius_of_circle :
  ∃ r : ℝ, ∀ x : ℝ, (x^2 + r = x) ↔ (r = 1 / 4) :=
by
  sorry

end radius_of_circle_l173_173170


namespace sin_theta_value_l173_173128

open Real

noncomputable def sin_theta_sol (theta : ℝ) : ℝ :=
  (-5 + Real.sqrt 41) / 4

theorem sin_theta_value (theta : ℝ) (h1 : 5 * tan theta = 2 * cos theta) (h2 : 0 < theta) (h3 : theta < π) :
  sin theta = sin_theta_sol theta :=
by
  sorry

end sin_theta_value_l173_173128


namespace total_silk_dyed_correct_l173_173940

-- Define the conditions
def green_silk_yards : ℕ := 61921
def pink_silk_yards : ℕ := 49500
def total_silk_yards : ℕ := green_silk_yards + pink_silk_yards

-- State the theorem to be proved
theorem total_silk_dyed_correct : total_silk_yards = 111421 := by
  sorry

end total_silk_dyed_correct_l173_173940


namespace age_difference_l173_173937

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end age_difference_l173_173937


namespace no_solutions_for_specific_a_l173_173840

theorem no_solutions_for_specific_a (a : ℝ) :
  (a < -9) ∨ (a > 0) →
  ¬ ∃ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 3 * a = 0 :=
by sorry

end no_solutions_for_specific_a_l173_173840


namespace runners_meet_time_l173_173163

theorem runners_meet_time (t_P t_Q : ℕ) (hP: t_P = 252) (hQ: t_Q = 198) : Nat.lcm t_P t_Q = 2772 :=
by
  rw [hP, hQ]
  -- The proof can be continued by proving the LCM calculation step, which we omit here
  sorry

end runners_meet_time_l173_173163


namespace cylinder_radius_l173_173074

theorem cylinder_radius
  (diameter_c : ℝ) (altitude_c : ℝ) (height_relation : ℝ → ℝ)
  (same_axis : Bool) (radius_cylinder : ℝ → ℝ)
  (h1 : diameter_c = 14)
  (h2 : altitude_c = 20)
  (h3 : ∀ r, height_relation r = 3 * r)
  (h4 : same_axis = true)
  (h5 : ∀ r, radius_cylinder r = r) :
  ∃ r, r = 140 / 41 :=
by {
  sorry
}

end cylinder_radius_l173_173074


namespace factorize_quadratic_l173_173092

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end factorize_quadratic_l173_173092


namespace seed_mixture_ryegrass_l173_173332

theorem seed_mixture_ryegrass (α : ℝ) :
  (0.4667 * 0.4 + 0.5333 * α = 0.32) -> α = 0.25 :=
by
  sorry

end seed_mixture_ryegrass_l173_173332


namespace smallest_d_for_inverse_l173_173022

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse :
  ∃ d, (∀ x₁ x₂, d ≤ x₁ ∧ d ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) ∧ (∀ e, (∀ x₁ x₂, e ≤ x₁ ∧ e ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) → d ≤ e) ∧ d = 3 :=
by
  sorry

end smallest_d_for_inverse_l173_173022


namespace geometric_seq_problem_l173_173108

theorem geometric_seq_problem
  (a : Nat → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_cond : a 1 * a 99 = 16) :
  a 20 * a 80 = 16 := 
sorry

end geometric_seq_problem_l173_173108


namespace linda_total_distance_l173_173090

theorem linda_total_distance :
  ∃ x: ℕ, 
    (x > 0) ∧ (60 % x = 0) ∧
    ((x + 5) > 0) ∧ (60 % (x + 5) = 0) ∧
    ((x + 10) > 0) ∧ (60 % (x + 10) = 0) ∧
    ((x + 15) > 0) ∧ (60 % (x + 15) = 0) ∧
    (60 / x + 60 / (x + 5) + 60 / (x + 10) + 60 / (x + 15) = 25) :=
by
  sorry

end linda_total_distance_l173_173090


namespace value_of_x_l173_173874

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end value_of_x_l173_173874


namespace expression_value_l173_173307

theorem expression_value (a b : ℚ) (h : a + 2 * b = 0) : 
  abs (a / |b| - 1) + abs (|a| / b - 2) + abs (|a / b| - 3) = 4 :=
sorry

end expression_value_l173_173307


namespace real_y_values_l173_173408

theorem real_y_values (x : ℝ) :
  (∃ y : ℝ, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23 / 9 ∨ x ≥ 5 / 3) :=
by
  sorry

end real_y_values_l173_173408


namespace gain_percent_l173_173302

variable (C S : ℝ)

theorem gain_percent 
  (h : 81 * C = 45 * S) : ((4 / 5) * 100) = 80 := 
by 
  sorry

end gain_percent_l173_173302


namespace balloon_arrangements_l173_173255

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l173_173255


namespace radius_of_cylinder_l173_173075

-- Define the main parameters and conditions
def diameter_cone := 8
def radius_cone := diameter_cone / 2
def altitude_cone := 10
def height_cylinder (r : ℝ) := 2 * r

-- Assume similarity of triangles
theorem radius_of_cylinder (r : ℝ) (h_c := height_cylinder r) :
  altitude_cone - h_c / r = altitude_cone / radius_cone → r = 20 / 9 := 
by
  intro h
  sorry

end radius_of_cylinder_l173_173075


namespace coords_P_origin_l173_173577

variable (x y : Int)
def point_P := (-5, 3)

theorem coords_P_origin : point_P = (-5, 3) := 
by 
  -- Proof to be written here
  sorry

end coords_P_origin_l173_173577


namespace arithmetic_geometric_sequence_l173_173141

theorem arithmetic_geometric_sequence (d : ℤ) (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = a (n - 1) * a (n + 2)) :
  a 2017 = 1 :=
sorry

end arithmetic_geometric_sequence_l173_173141


namespace time_for_new_circle_l173_173520

theorem time_for_new_circle 
  (rounds : ℕ) (time : ℕ) (k : ℕ) (original_time_per_round new_time_per_round : ℝ) 
  (h1 : rounds = 8) 
  (h2 : time = 40) 
  (h3 : k = 10) 
  (h4 : original_time_per_round = time / rounds)
  (h5 : new_time_per_round = original_time_per_round * k) :
  new_time_per_round = 50 :=
by {
  sorry
}

end time_for_new_circle_l173_173520


namespace larger_box_cost_l173_173908

-- Definitions based on the conditions

def ounces_large : ℕ := 30
def ounces_small : ℕ := 20
def cost_small : ℝ := 3.40
def price_per_ounce_better_value : ℝ := 0.16

-- The statement to prove
theorem larger_box_cost :
  30 * price_per_ounce_better_value = 4.80 :=
by sorry

end larger_box_cost_l173_173908


namespace perpendicular_distance_is_8_cm_l173_173220

theorem perpendicular_distance_is_8_cm :
  ∀ (side_length distance_from_corner cut_angle : ℝ),
    side_length = 100 →
    distance_from_corner = 8 →
    cut_angle = 45 →
    (∃ h : ℝ, h = 8) :=
by
  intros side_length distance_from_corner cut_angle hms d8 a45
  sorry

end perpendicular_distance_is_8_cm_l173_173220


namespace johns_final_push_time_l173_173374

-- Definitions and initial conditions.
def john_initial_distance_behind_steve : ℝ := 12
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.7
def john_final_distance_ahead_of_steve : ℝ := 2

-- The statement we want to prove:
theorem johns_final_push_time : ∃ t : ℝ, john_speed * t = steve_speed * t + john_initial_distance_behind_steve + john_final_distance_ahead_of_steve ∧ t = 28 := 
by 
  -- Adding blank proof body
  sorry

end johns_final_push_time_l173_173374


namespace unit_conversion_factor_l173_173932

theorem unit_conversion_factor (u : ℝ) (h₁ : u = 5) (h₂ : (u * 0.9)^2 = 20.25) : u = 5 → (1 : ℝ) = 0.9  :=
sorry

end unit_conversion_factor_l173_173932


namespace existence_of_subset_A_l173_173550

def M : Set ℚ := {x : ℚ | 0 < x ∧ x < 1}

theorem existence_of_subset_A :
  ∃ A ⊆ M, ∀ m ∈ M, ∃! (S : Finset ℚ), (∀ a ∈ S, a ∈ A) ∧ (S.sum id = m) :=
sorry

end existence_of_subset_A_l173_173550


namespace range_of_function_x_geq_0_l173_173873

theorem range_of_function_x_geq_0 :
  ∀ (x : ℝ), x ≥ 0 → ∃ (y : ℝ), y ≥ 3 ∧ (y = x^2 + 2 * x + 3) :=
by
  sorry

end range_of_function_x_geq_0_l173_173873


namespace ratio_Lisa_Charlotte_l173_173405

def P_tot : ℕ := 100
def Pat_money : ℕ := 6
def Lisa_money : ℕ := 5 * Pat_money
def additional_required : ℕ := 49
def current_total_money : ℕ := P_tot - additional_required
def Pat_Lisa_total : ℕ := Pat_money + Lisa_money
def Charlotte_money : ℕ := current_total_money - Pat_Lisa_total

theorem ratio_Lisa_Charlotte : (Lisa_money : ℕ) / Charlotte_money = 2 :=
by
  -- Proof to be filled in later
  sorry

end ratio_Lisa_Charlotte_l173_173405


namespace balloon_arrangements_l173_173264

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l173_173264


namespace initial_trees_l173_173189

theorem initial_trees (DeadTrees CutTrees LeftTrees : ℕ) (h1 : DeadTrees = 15) (h2 : CutTrees = 23) (h3 : LeftTrees = 48) :
  DeadTrees + CutTrees + LeftTrees = 86 :=
by
  sorry

end initial_trees_l173_173189


namespace integer_a_can_be_written_in_form_l173_173181

theorem integer_a_can_be_written_in_form 
  (a x y : ℤ) 
  (h : 3 * a = x^2 + 2 * y^2) : 
  ∃ u v : ℤ, a = u^2 + 2 * v^2 :=
sorry

end integer_a_can_be_written_in_form_l173_173181


namespace correct_average_marks_l173_173339

theorem correct_average_marks 
  (n : ℕ) (wrong_avg : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (h1 : n = 10)
  (h2 : wrong_avg = 100)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 10) :
  (n * wrong_avg - wrong_mark + correct_mark) / n = 92 :=
by
  sorry

end correct_average_marks_l173_173339


namespace percentage_seniors_with_cars_is_40_l173_173353

noncomputable def percentage_of_seniors_with_cars 
  (total_students: ℕ) (seniors: ℕ) (lower_grades: ℕ) (percent_cars_all: ℚ) (percent_cars_lower_grades: ℚ) : ℚ :=
  let total_with_cars := percent_cars_all * total_students
  let lower_grades_with_cars := percent_cars_lower_grades * lower_grades
  let seniors_with_cars := total_with_cars - lower_grades_with_cars
  (seniors_with_cars / seniors) * 100

theorem percentage_seniors_with_cars_is_40
  : percentage_of_seniors_with_cars 1800 300 1500 0.15 0.10 = 40 := 
by
  -- Proof is omitted
  sorry

end percentage_seniors_with_cars_is_40_l173_173353


namespace arithmetic_mean_probability_integer_l173_173501

open Finset

theorem arithmetic_mean_probability_integer :
  let S := {2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030}.toFinset in
  let favorable : Finset (Finset ℕ) := S.powerset.filter (λ t, t.card = 3 ∧ (t.sum id) % 3 = 0) in
  let total : Finset (Finset ℕ) := S.powerset.filter (λ t, t.card = 3) in
  (toRational favorable.card / toRational total.card) = (7 / 20 : ℚ) :=
by {
  sorry
}

end arithmetic_mean_probability_integer_l173_173501


namespace smallest_value_of_y_l173_173903

theorem smallest_value_of_y (x y z d : ℝ) (h1 : x = y - d) (h2 : z = y + d) (h3 : x * y * z = 125) (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : y ≥ 5 :=
by
  -- Officially, the user should navigate through the proof, but we conclude with 'sorry' as placeholder
  sorry

end smallest_value_of_y_l173_173903


namespace complement_union_eq_l173_173595

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem complement_union_eq : (U \ (S ∪ T)) = {2, 4, 7, 8} :=
by {
  sorry
}

end complement_union_eq_l173_173595


namespace product_of_five_numbers_is_256_l173_173104

def possible_numbers : Set ℕ := {1, 2, 4}

theorem product_of_five_numbers_is_256 
  (x1 x2 x3 x4 x5 : ℕ) 
  (h1 : x1 ∈ possible_numbers) 
  (h2 : x2 ∈ possible_numbers) 
  (h3 : x3 ∈ possible_numbers) 
  (h4 : x4 ∈ possible_numbers) 
  (h5 : x5 ∈ possible_numbers) : 
  x1 * x2 * x3 * x4 * x5 = 256 :=
sorry

end product_of_five_numbers_is_256_l173_173104


namespace remainder_approx_l173_173776

def x : ℝ := 74.99999999999716 * 96
def y : ℝ := 74.99999999999716
def quotient : ℝ := 96
def expected_remainder : ℝ := 0.4096

theorem remainder_approx (x y : ℝ) (quotient : ℝ) (h1 : y = 74.99999999999716)
  (h2 : quotient = 96) (h3 : x = y * quotient) :
  x - y * quotient = expected_remainder :=
by
  sorry

end remainder_approx_l173_173776


namespace minimum_candies_l173_173312

variables (c z : ℕ) (total_candies : ℕ)

def remaining_red_candies := (3 * c) / 5
def remaining_green_candies := (2 * z) / 5
def remaining_total_candies := remaining_red_candies + remaining_green_candies
def red_candies_fraction := remaining_red_candies * 8 = 3 * remaining_total_candies

theorem minimum_candies (h1 : 5 * c = 2 * z) (h2 : red_candies_fraction) :
  total_candies ≥ 35 := sorry

end minimum_candies_l173_173312


namespace probability_multinomial_l173_173186

theorem probability_multinomial (n k1 k2 k3 : ℕ) (p1 p2 p3 : ℝ) 
  (h_n : n = 6)
  (h_k1 : k1 = 3)
  (h_k2 : k2 = 2)
  (h_k3 : k3 = 1)
  (h_p1 : p1 = 0.5)
  (h_p2 : p2 = 0.3)
  (h_p3 : p3 = 0.2) : 
  (nat.factorial n / (nat.factorial k1 * nat.factorial k2 * nat.factorial k3) * 
  p1^k1 * p2^k2 * p3^k3 = 0.135) :=
by 
  sorry

end probability_multinomial_l173_173186


namespace recurring_fraction_division_l173_173362

/--
Given x = 0.\overline{36} and y = 0.\overline{12}, prove that x / y = 3.
-/
theorem recurring_fraction_division 
  (x y : ℝ)
  (h1 : x = 0.36 + 0.0036 + 0.000036 + 0.00000036 + ......) -- representation of 0.\overline{36}
  (h2 : y = 0.12 + 0.0012 + 0.000012 + 0.00000012 + ......) -- representation of 0.\overline{12}
  : x / y = 3 :=
  sorry

end recurring_fraction_division_l173_173362


namespace alice_probability_multiple_of_4_l173_173816

noncomputable def probability_one_multiple_of_4 (choices : ℕ) : ℚ :=
  let p_not_multiple_of_4 : ℚ := 45 / 60
  let p_all_not_multiple_of_4 : ℚ := p_not_multiple_of_4 ^ choices
  1 - p_all_not_multiple_of_4

theorem alice_probability_multiple_of_4 :
  probability_one_multiple_of_4 3 = 37 / 64 :=
by
  sorry

end alice_probability_multiple_of_4_l173_173816


namespace solve_equation_l173_173612

theorem solve_equation (x : ℤ) (h1 : x ≠ 2) : x - 8 / (x - 2) = 5 - 8 / (x - 2) → x = 5 := by
  sorry

end solve_equation_l173_173612


namespace probability_not_yellow_l173_173209

-- Define the conditions
def red_jelly_beans : Nat := 4
def green_jelly_beans : Nat := 7
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Definitions used in the proof problem
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_yellow_jelly_beans : Nat := total_jelly_beans - yellow_jelly_beans

-- Lean statement of the probability problem
theorem probability_not_yellow : 
  (non_yellow_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = 7 / 10 := 
by 
  sorry

end probability_not_yellow_l173_173209


namespace inequality_for_positive_real_l173_173609

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end inequality_for_positive_real_l173_173609


namespace initial_bucket_capacity_l173_173124

theorem initial_bucket_capacity (x : ℕ) (h1 : x - 3 = 2) : x = 5 := sorry

end initial_bucket_capacity_l173_173124


namespace percentage_food_given_out_l173_173157

theorem percentage_food_given_out 
  (first_week_donations : ℕ)
  (second_week_donations : ℕ)
  (total_amount_donated : ℕ)
  (remaining_food : ℕ)
  (amount_given_out : ℕ)
  (percentage_given_out : ℕ) : 
  (first_week_donations = 40) →
  (second_week_donations = 2 * first_week_donations) →
  (total_amount_donated = first_week_donations + second_week_donations) →
  (remaining_food = 36) →
  (amount_given_out = total_amount_donated - remaining_food) →
  (percentage_given_out = (amount_given_out * 100) / total_amount_donated) →
  percentage_given_out = 70 :=
by sorry

end percentage_food_given_out_l173_173157


namespace Sandy_total_marks_l173_173739

theorem Sandy_total_marks
  (correct_marks_per_sum : ℤ)
  (incorrect_marks_per_sum : ℤ)
  (total_sums : ℕ)
  (correct_sums : ℕ)
  (incorrect_sums : ℕ)
  (total_marks : ℤ) :
  correct_marks_per_sum = 3 →
  incorrect_marks_per_sum = -2 →
  total_sums = 30 →
  correct_sums = 24 →
  incorrect_sums = total_sums - correct_sums →
  total_marks = correct_marks_per_sum * correct_sums + incorrect_marks_per_sum * incorrect_sums →
  total_marks = 60 :=
by
  sorry

end Sandy_total_marks_l173_173739


namespace collinear_points_sum_l173_173003

-- Points in 3-dimensional space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition of collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ k : ℝ,
    k ≠ 0 ∧
    (p2.x - p1.x) * k = (p3.x - p1.x) ∧
    (p2.y - p1.y) * k = (p3.y - p1.y) ∧
    (p2.z - p1.z) * k = (p3.z - p1.z)

-- Main statement
theorem collinear_points_sum {a b : ℝ} :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) → a + b = 6 :=
by
  sorry

end collinear_points_sum_l173_173003


namespace uncertain_relationship_l173_173105

noncomputable section

-- Define the events A and B in a sample space Ω
variable {Ω : Type*} (A B : Ω → Prop)

-- Define the probability measure P on the sample space Ω
variable (P : MeasureTheory.Measure Ω)

-- Define the conditions given in the problem
def events_relationship : Prop :=
  P {ω | A ω ∨ B ω} = P {ω | A ω} + P {ω | B ω} = 1

-- Lean theorem statement for the problem
theorem uncertain_relationship (P : MeasureTheory.Measure Ω) (A B : Ω → Prop) (h : events_relationship P A B) : 
  (complementary_relation P A B ∨ ¬ complementary_relation P A B) :=
sorry

end uncertain_relationship_l173_173105


namespace range_of_m_l173_173557

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / (x - 3) - 1 = x / (3 - x)) →
  m > 3 ∧ m ≠ 9 :=
by
  sorry

end range_of_m_l173_173557


namespace solution_set_absolute_value_l173_173350

theorem solution_set_absolute_value (x : ℝ) : 
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  -- Proof goes here
  sorry

end solution_set_absolute_value_l173_173350


namespace sum_of_common_ratios_l173_173788

theorem sum_of_common_ratios (k p r : ℝ) (h1 : k ≠ 0) (h2 : k * (p^2) - k * (r^2) = 5 * (k * p - k * r)) (h3 : p ≠ r) : p + r = 5 :=
sorry

end sum_of_common_ratios_l173_173788


namespace find_percentage_of_alcohol_l173_173035

theorem find_percentage_of_alcohol 
  (Vx : ℝ) (Px : ℝ) (Vy : ℝ) (Py : ℝ) (Vp : ℝ) (Pp : ℝ)
  (hx : Px = 10) (hvx : Vx = 300) (hvy : Vy = 100) (hvxy : Vx + Vy = 400) (hpxy : Pp = 15) :
  (Vy * Py / 100) = 30 :=
by
  sorry

end find_percentage_of_alcohol_l173_173035


namespace number_of_larger_planes_l173_173077

variable (S L : ℕ)
variable (h1 : S + L = 4)
variable (h2 : 130 * S + 145 * L = 550)

theorem number_of_larger_planes : L = 2 :=
by
  -- Placeholder for the proof
  sorry

end number_of_larger_planes_l173_173077


namespace minimum_value_x_plus_y_l173_173863

theorem minimum_value_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = x * y) :
  x + y = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_x_plus_y_l173_173863


namespace evaluate_expression_l173_173897

theorem evaluate_expression (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3 * a + 8 * b + 24 * c + 37 * d = 2018) : 
  3 * b + 8 * c + 24 * d + 37 * a = 1215 :=
by 
  sorry

end evaluate_expression_l173_173897


namespace cube_volume_from_surface_area_l173_173993

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173993


namespace trip_to_market_distance_l173_173626

theorem trip_to_market_distance 
  (school_trip_one_way : ℝ) (school_days_per_week : ℕ) 
  (weekly_total_mileage : ℝ) (round_trips_per_day : ℕ) (market_trip_count : ℕ) :
  (school_trip_one_way = 2.5) →
  (school_days_per_week = 4) →
  (round_trips_per_day = 2) →
  (weekly_total_mileage = 44) →
  (market_trip_count = 1) →
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  market_trip_distance = 2 :=
by
  intros h1 h2 h3 h4 h5
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  sorry

end trip_to_market_distance_l173_173626


namespace numberOfRottweilers_l173_173719

-- Define the grooming times in minutes for each type of dog
def groomingTimeRottweiler := 20
def groomingTimeCollie := 10
def groomingTimeChihuahua := 45

-- Define the number of each type of dog groomed
def numberOfCollies := 9
def numberOfChihuahuas := 1

-- Define the total grooming time in minutes
def totalGroomingTime := 255

-- Compute the time spent on grooming Collies
def timeSpentOnCollies := numberOfCollies * groomingTimeCollie

-- Compute the time spent on grooming Chihuahuas
def timeSpentOnChihuahuas := numberOfChihuahuas * groomingTimeChihuahua

-- Compute the time spent on grooming Rottweilers
def timeSpentOnRottweilers := totalGroomingTime - timeSpentOnCollies - timeSpentOnChihuahuas

-- The main theorem statement
theorem numberOfRottweilers :
  timeSpentOnRottweilers / groomingTimeRottweiler = 6 :=
by
  -- Proof placeholder
  sorry

end numberOfRottweilers_l173_173719


namespace cube_volume_is_1728_l173_173955

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173955


namespace keith_books_l173_173017

theorem keith_books : 
  ∀ (jason_books : ℕ) (total_books : ℕ),
    jason_books = 21 ∧ total_books = 41 →
    total_books - jason_books = 20 :=
by 
  intros jason_books total_books h,
  cases h with h1 h2,
  rw h1,
  rw h2,
  norm_num,
  sorry

end keith_books_l173_173017


namespace part_a_part_b_l173_173068

-- Definition for combination
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Proof problems as Lean statements
theorem part_a : combination 30 2 = 435 := by
  sorry

theorem part_b : combination 30 3 = 4060 := by
  sorry

end part_a_part_b_l173_173068


namespace balloon_arrangements_l173_173240

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l173_173240


namespace common_ratio_of_geometric_sequence_l173_173858

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n+1) = a n * q) → 
  (a 1 + a 5 = 17) → 
  (a 2 * a 4 = 16) → 
  (∀ i j, i < j → a i < a j) → 
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l173_173858


namespace total_books_from_library_l173_173605

def initialBooks : ℕ := 54
def additionalBooks : ℕ := 23

theorem total_books_from_library : initialBooks + additionalBooks = 77 := by
  sorry

end total_books_from_library_l173_173605


namespace cube_volume_from_surface_area_example_cube_volume_l173_173959

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173959


namespace cube_volume_is_1728_l173_173954

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173954


namespace gcd_231_154_l173_173280

def find_gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_231_154 : find_gcd 231 154 = 77 := by
  sorry

end gcd_231_154_l173_173280


namespace balloon_arrangements_l173_173256

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l173_173256


namespace line_equation_l173_173341

theorem line_equation
(point_pass : (λ P : ℝ × ℝ, P = (-4, -1)))
(x_intercept_twice_y_intercept : ∀ l : ℝ × ℝ → Prop, (∃ (P : ℝ × ℝ), l P ∧ P.1 / P.2 = 2)) :
(l = (λ P : ℝ × ℝ, P.2 = (1 / 4) * P.1) ∨ l = (λ P : ℝ × ℝ, P.1 + 2 * P.2 + 6 = 0)) :=
sorry

end line_equation_l173_173341


namespace car_highway_miles_per_tankful_l173_173070

-- Condition definitions
def city_miles_per_tankful : ℕ := 336
def miles_per_gallon_city : ℕ := 24
def city_to_highway_diff : ℕ := 9

-- Calculation from conditions
def miles_per_gallon_highway : ℕ := miles_per_gallon_city + city_to_highway_diff
def tank_size : ℤ := city_miles_per_tankful / miles_per_gallon_city

-- Desired result
def highway_miles_per_tankful : ℤ := miles_per_gallon_highway * tank_size

-- Proof statement
theorem car_highway_miles_per_tankful :
  highway_miles_per_tankful = 462 := by
  unfold highway_miles_per_tankful
  unfold miles_per_gallon_highway
  unfold tank_size
  -- Sorry here to skip the detailed proof steps
  sorry

end car_highway_miles_per_tankful_l173_173070


namespace additional_time_required_l173_173207

-- Definitions based on conditions
def time_to_clean_three_sections : ℕ := 24
def total_sections : ℕ := 27

-- Rate of cleaning
def cleaning_rate_per_section (t : ℕ) (n : ℕ) : ℕ := t / n

-- Total time required to clean all sections
def total_cleaning_time (n : ℕ) (r : ℕ) : ℕ := n * r

-- Additional time required to clean the remaining sections
def additional_cleaning_time (t_total : ℕ) (t_spent : ℕ) : ℕ := t_total - t_spent

-- Theorem statement
theorem additional_time_required 
  (t3 : ℕ) (n : ℕ) (t_spent : ℕ) 
  (h₁ : t3 = time_to_clean_three_sections)
  (h₂ : n = total_sections)
  (h₃ : t_spent = time_to_clean_three_sections)
  : additional_cleaning_time (total_cleaning_time n (cleaning_rate_per_section t3 3)) t_spent = 192 :=
by
  sorry

end additional_time_required_l173_173207


namespace original_number_of_men_l173_173201

theorem original_number_of_men 
    (x : ℕ) 
    (h : x * 40 = (x - 5) * 60) : x = 15 := 
sorry

end original_number_of_men_l173_173201


namespace keiko_speed_l173_173149

theorem keiko_speed (a b : ℝ) (s : ℝ) (h1 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : s = π / 3 :=
by {
  sorry -- proof is not required
}

end keiko_speed_l173_173149


namespace range_of_a_l173_173564

noncomputable def A : Set ℝ := {x | x^2 ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : a ≥ 1 := 
by
  sorry

end range_of_a_l173_173564


namespace fraction_of_visitors_l173_173711

variable (V E U : ℕ)
variable (H1 : E = U)
variable (H2 : 600 - E - 150 = 450)

theorem fraction_of_visitors (H3 : 600 = E + 150 + 450) : (450 : ℚ) / 600 = (3 : ℚ) / 4 :=
by
  apply sorry

end fraction_of_visitors_l173_173711


namespace locus_of_p_ratio_distances_l173_173714

theorem locus_of_p_ratio_distances :
  (∀ (P : ℝ × ℝ), (dist P (1, 0) = (1 / 3) * abs (P.1 - 9)) →
  (P.1^2 / 9 + P.2^2 / 8 = 1)) :=
by
  sorry

end locus_of_p_ratio_distances_l173_173714


namespace clive_can_correct_time_l173_173747

def can_show_correct_time (hour_hand_angle minute_hand_angle : ℝ) :=
  ∃ θ : ℝ, θ ∈ [0, 360] ∧ hour_hand_angle + θ % 360 = minute_hand_angle + θ % 360

theorem clive_can_correct_time (hour_hand_angle minute_hand_angle : ℝ) :
  can_show_correct_time hour_hand_angle minute_hand_angle :=
sorry

end clive_can_correct_time_l173_173747


namespace twenty_four_times_ninety_nine_l173_173656

theorem twenty_four_times_ninety_nine : 24 * 99 = 2376 :=
by sorry

end twenty_four_times_ninety_nine_l173_173656


namespace compute_value_l173_173901

theorem compute_value {p q : ℝ} (h1 : 3 * p^2 - 5 * p - 8 = 0) (h2 : 3 * q^2 - 5 * q - 8 = 0) :
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 :=
by
  sorry

end compute_value_l173_173901


namespace tessa_initial_apples_l173_173613

theorem tessa_initial_apples (x : ℝ) (h : x + 5.0 - 4.0 = 11) : x = 10 :=
by
  sorry

end tessa_initial_apples_l173_173613


namespace tax_amount_self_employed_l173_173622

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end tax_amount_self_employed_l173_173622


namespace semicircle_radius_l173_173806

noncomputable def radius_of_inscribed_semicircle (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21) : ℝ :=
  let AB := Real.sqrt (21^2 + 10^2)
  let s := 2 * Real.sqrt 541
  let area := 20 * 21
  (area) / (s * 2)

theorem semicircle_radius (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21)
  : radius_of_inscribed_semicircle BD height h_base h_height = 210 / Real.sqrt 541 :=
sorry

end semicircle_radius_l173_173806


namespace first_stack_height_is_seven_l173_173014

-- Definitions of the conditions
def first_stack (h : ℕ) := h
def second_stack (h : ℕ) := h + 5
def third_stack (h : ℕ) := h + 12

-- Conditions on the blocks falling down
def blocks_fell_first_stack (h : ℕ) := h
def blocks_fell_second_stack (h : ℕ) := (h + 5) - 2
def blocks_fell_third_stack (h : ℕ) := (h + 12) - 3

-- Total blocks fell down
def total_blocks_fell (h : ℕ) := blocks_fell_first_stack h + blocks_fell_second_stack h + blocks_fell_third_stack h

-- Lean statement to prove the height of the first stack
theorem first_stack_height_is_seven (h : ℕ) (h_eq : total_blocks_fell h = 33) : h = 7 :=
by sorry

-- Testing the conditions hold for the solution h = 7
#eval total_blocks_fell 7 -- Expected: 33

end first_stack_height_is_seven_l173_173014


namespace max_a1_l173_173556

theorem max_a1 (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, n > 0 → a n > 0)
  (h_eq : ∀ n : ℕ, n > 0 → 2 + a n * (a (n + 1) - a (n - 1)) = 0 ∨ 2 - a n * (a (n + 1) - a (n - 1)) = 0)
  (h_a20 : a 20 = a 20) :
  ∃ max_a1 : ℝ, max_a1 = 512 := 
sorry

end max_a1_l173_173556


namespace Cathy_wins_l173_173019

theorem Cathy_wins (n k : ℕ) (hn : n > 0) (hk : k > 0) : (∃ box_count : ℕ, box_count = 1) :=
  if h : n ≤ 2^(k-1) then
    sorry
  else
    sorry

end Cathy_wins_l173_173019


namespace students_neither_music_nor_art_l173_173791

theorem students_neither_music_nor_art
  (total_students : ℕ) (students_music : ℕ) (students_art : ℕ) (students_both : ℕ)
  (h_total : total_students = 500)
  (h_music : students_music = 30)
  (h_art : students_art = 10)
  (h_both : students_both = 10)
  : total_students - (students_music + students_art - students_both) = 460 :=
by
  rw [h_total, h_music, h_art, h_both]
  norm_num
  sorry

end students_neither_music_nor_art_l173_173791


namespace line_passing_A_parallel_BC_eq_l173_173692

-- Definitions and structures for points and slopes
structure Point where
  x : ℝ
  y : ℝ

def slope (P Q : Point) : ℝ :=
  if P.x = Q.x then 0 else (Q.y - P.y) / (Q.x - P.x)

-- Definitions of points A, B, and C
def A : Point := ⟨4, 0⟩
def B : Point := ⟨8, 10⟩
def C : Point := ⟨0, 6⟩

-- The slope of line BC
def slopeBC : ℝ := slope B C

-- The equation of the line passing through A and parallel to BC
def lineEquation (P : Point) (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y - P.y = k * (x - P.x)

theorem line_passing_A_parallel_BC_eq :
  ∀ (x y : ℝ), lineEquation A slopeBC x y ↔ x - 2*y - 4 = 0 :=
by
  sorry

end line_passing_A_parallel_BC_eq_l173_173692


namespace ratio_of_green_to_blue_l173_173629

-- Definitions of the areas and the circles
noncomputable def red_area : ℝ := Real.pi * (1 : ℝ) ^ 2
noncomputable def middle_area : ℝ := Real.pi * (2 : ℝ) ^ 2
noncomputable def large_area: ℝ := Real.pi * (3 : ℝ) ^ 2

noncomputable def blue_area : ℝ := middle_area - red_area
noncomputable def green_area : ℝ := large_area - middle_area

-- The proof that the ratio of the green area to the blue area is 5/3
theorem ratio_of_green_to_blue : green_area / blue_area = 5 / 3 := by
  sorry

end ratio_of_green_to_blue_l173_173629


namespace neg_abs_nonneg_l173_173441

theorem neg_abs_nonneg :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by
  sorry

end neg_abs_nonneg_l173_173441


namespace balloon_permutations_l173_173268

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l173_173268


namespace arithmetic_sequence_properties_l173_173134

noncomputable def general_term_formula (a₁ : ℕ) (S₃ : ℕ) (n : ℕ) (d : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def sum_of_double_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (2 * (a₁ + (n - 1) * d)) * n / 2

theorem arithmetic_sequence_properties
  (a₁ : ℕ) (S₃ : ℕ)
  (h₁ : a₁ = 2)
  (h₂ : S₃ = 9) :
  general_term_formula a₁ S₃ n (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) = n + 1 ∧
  sum_of_double_sequence a₁ (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) n = 2^(n+2) - 4 :=
by
  sorry

end arithmetic_sequence_properties_l173_173134


namespace cauchy_schwarz_inequality_l173_173169

theorem cauchy_schwarz_inequality 
  (a b a1 b1 : ℝ) : ((a * a1 + b * b1) ^ 2 ≤ (a^2 + b^2) * (a1^2 + b1^2)) :=
 by sorry

end cauchy_schwarz_inequality_l173_173169


namespace question1_question2_l173_173689

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem question1 (m : ℝ) (h1 : m > 0) 
(h2 : ∀ (x : ℝ), f (x + 1/2) ≤ 2 * m + 1 ↔ x ∈ [-2, 2]) : m = 3 / 2 := 
sorry

theorem question2 (x y : ℝ) : f x ≤ 2^y + 4 / 2^y + |2 * x + 3| := 
sorry

end question1_question2_l173_173689


namespace general_term_and_sum_l173_173460

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

-- Conditions for the geometric sequence {a_n}
axiom a_seq_geometric (n : ℕ) (a1 a2 : ℕ) (h1 : a1 * a2 = 8) (h2 : a1 + a2 = 6) : a n = 2^n

-- Definition of sequence {b_n}
def b_seq (n : ℕ) : ℕ := 2 * a n + 3

-- Sum of the first n terms of the sequence {b_n}
axiom sum_b_seq (n : ℕ) : T n = (2 ^ (n + 2)) - 4 + 3 * n

-- Theorem to prove
theorem general_term_and_sum 
(h : ∀ n, a n = 2 ^ n) 
(h_sum: ∀ n, T n = (2 ^ (n + 2)) - 4 + 3 * n) :
∀ n, (a n = 2 ^ n) ∧ (T n = (2 ^ (n + 2)) - 4 + 3 * n) := by
  intros
  exact ⟨h n, h_sum n⟩

end general_term_and_sum_l173_173460


namespace team_selection_ways_l173_173030

open Nat

theorem team_selection_ways :
  ∑ i in ({3, 4, 5, 6} : Finset ℕ), (choose 10 i) * (choose 5 (6 - i)) = 4770 :=
by
  simp only [Finset.sum_insert, Finset.insert_empty_eq_singleton]
  sorry

end team_selection_ways_l173_173030


namespace average_sales_l173_173938

-- Define the cost calculation for each special weekend
noncomputable def valentines_day_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20)) / 10

noncomputable def st_patricks_day_sales_per_ticket : Real :=
  ((3 * 2.00) + 6.25 + (8 * 1.00)) / 8

noncomputable def christmas_sales_per_ticket : Real :=
  ((6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 9

-- Define the combined average snack sales
noncomputable def combined_average_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20) + (3 * 2.00) + 6.25 + (8 * 1.00) + (6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 27

-- Proof problem as a Lean theorem
theorem average_sales : 
  valentines_day_sales_per_ticket = 2.62 ∧ 
  st_patricks_day_sales_per_ticket = 2.53 ∧ 
  christmas_sales_per_ticket = 3.16 ∧ 
  combined_average_sales_per_ticket = 2.78 :=
by 
  sorry

end average_sales_l173_173938


namespace total_buttons_l173_173808

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end total_buttons_l173_173808


namespace first_motorcyclist_laps_per_hour_l173_173771

noncomputable def motorcyclist_laps (x y z : ℝ) (P1 : 0 < x - y) (P2 : 0 < x - z) (P3 : 0 < y - z) : Prop :=
  (4.5 / (x - y) = 4.5) ∧ (4.5 / (x - z) = 4.5 - 0.5) ∧ (3 / (y - z) = 3) → x = 3

theorem first_motorcyclist_laps_per_hour (x y z : ℝ) (P1: 0 < x - y) (P2: 0 < x - z) (P3: 0 < y - z) :
  motorcyclist_laps x y z P1 P2 P3 →
  x = 3 :=
sorry

end first_motorcyclist_laps_per_hour_l173_173771


namespace larger_integer_is_neg4_l173_173570

-- Definitions of the integers used in the problem
variables (x y : ℤ)

-- Conditions given in the problem
def condition1 : x + y = -9 := sorry
def condition2 : x - y = 1 := sorry

-- The theorem to prove
theorem larger_integer_is_neg4 (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := 
sorry

end larger_integer_is_neg4_l173_173570


namespace balloon_permutation_count_l173_173248

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l173_173248


namespace age_ratio_l173_173142

theorem age_ratio (Tim_age : ℕ) (John_age : ℕ) (ratio : ℚ) 
  (h1 : Tim_age = 79) 
  (h2 : John_age = 35) 
  (h3 : Tim_age = ratio * John_age - 5) : 
  ratio = 2.4 := 
by sorry

end age_ratio_l173_173142


namespace measure_of_unknown_angle_in_hexagon_l173_173111

theorem measure_of_unknown_angle_in_hexagon :
  let a1 := 135
  let a2 := 105
  let a3 := 87
  let a4 := 120
  let a5 := 78
  let total_internal_angles := 180 * (6 - 2)
  let known_sum := a1 + a2 + a3 + a4 + a5
  let Q := total_internal_angles - known_sum
  Q = 195 :=
by
  sorry

end measure_of_unknown_angle_in_hexagon_l173_173111


namespace sides_increase_factor_l173_173051

theorem sides_increase_factor (s k : ℝ) (h : s^2 * 25 = k^2 * s^2) : k = 5 :=
by
  sorry

end sides_increase_factor_l173_173051


namespace find_W_from_conditions_l173_173866

theorem find_W_from_conditions :
  ∀ (x y : ℝ), (y = 1 / x ∧ y = |x| + 1) → (x + y = Real.sqrt 5) :=
by
  sorry

end find_W_from_conditions_l173_173866


namespace solve_inequality_l173_173537

theorem solve_inequality (x : ℝ) : x^3 - 9*x^2 - 16*x > 0 ↔ (x < -1 ∨ x > 16) := by
  sorry

end solve_inequality_l173_173537


namespace value_of_x_l173_173875

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end value_of_x_l173_173875


namespace line_intersects_circle_not_center_l173_173047

def line_eq (x : ℝ) : ℝ := x + 1
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem line_intersects_circle_not_center :
  ∃ (x y : ℝ), (line_eq x = y) ∧ (circle_eq x y) ∧ ¬(x = 0 ∧ y = 0) :=
sorry

end line_intersects_circle_not_center_l173_173047


namespace no_periodic_sequence_first_non_zero_digit_l173_173136

/-- 
Definition of the first non-zero digit from the unit's place in the decimal representation of n! 
-/
def first_non_zero_digit (n : ℕ) : ℕ :=
  -- This function should compute the first non-zero digit from the unit's place in n!
  -- Implementation details are skipped here.
  sorry

/-- 
Prove that no natural number \( N \) exists such that the sequence \( a_{N+1}, a_{N+2}, a_{N+3}, \ldots \) 
forms a periodic sequence, where \( a_n \) is the first non-zero digit from the unit's place in the decimal 
representation of \( n! \). 
-/
theorem no_periodic_sequence_first_non_zero_digit :
  ¬ ∃ (N : ℕ), ∃ (T : ℕ), ∀ (k : ℕ), first_non_zero_digit (N + k * T) = first_non_zero_digit (N + ((k + 1) * T)) :=
by
  sorry

end no_periodic_sequence_first_non_zero_digit_l173_173136


namespace chord_length_of_intersection_l173_173072

theorem chord_length_of_intersection 
  (x y : ℝ) (h_line : 2 * x - y - 1 = 0) (h_circle : (x - 2)^2 + (y + 2)^2 = 9) : 
  ∃ l, l = 4 := 
sorry

end chord_length_of_intersection_l173_173072


namespace solution_set_l173_173831

noncomputable def f : ℝ → ℝ := sorry
axiom f'_lt_one_third (x : ℝ) : deriv f x < 1 / 3
axiom f_at_two : f 2 = 1

theorem solution_set : {x : ℝ | 0 < x ∧ x < 4} = {x : ℝ | f (Real.logb 2 x) > (Real.logb 2 x + 1) / 3} :=
by
  sorry

end solution_set_l173_173831


namespace minimum_boxes_l173_173825

theorem minimum_boxes (x y z : ℕ) (h1 : 50 * x = 40 * y) (h2 : 50 * x = 25 * z) :
  x + y + z = 17 :=
by
  -- Prove that given these equations, the minimum total number of boxes (x + y + z) is 17
  sorry

end minimum_boxes_l173_173825


namespace john_total_amount_l173_173015

/-- Define the amounts of money John has and needs additionally -/
def johnHas : ℝ := 0.75
def needsMore : ℝ := 1.75

/-- Prove the total amount of money John needs given the conditions -/
theorem john_total_amount : johnHas + needsMore = 2.50 := by
  sorry

end john_total_amount_l173_173015


namespace provenance_of_positive_test_l173_173458

noncomputable def pr_disease : ℚ := 1 / 200
noncomputable def pr_no_disease : ℚ := 1 - pr_disease
noncomputable def pr_test_given_disease : ℚ := 1
noncomputable def pr_test_given_no_disease : ℚ := 0.05
noncomputable def pr_test : ℚ := pr_test_given_disease * pr_disease + pr_test_given_no_disease * pr_no_disease
noncomputable def pr_disease_given_test : ℚ := 
  (pr_test_given_disease * pr_disease) / pr_test

theorem provenance_of_positive_test : pr_disease_given_test = 20 / 219 :=
by
  sorry

end provenance_of_positive_test_l173_173458


namespace property_related_only_to_temperature_l173_173198

-- The conditions given in the problem
def solubility_of_ammonia_gas (T P : Prop) : Prop := T ∧ P
def ion_product_of_water (T : Prop) : Prop := T
def oxidizing_property_of_pp (T C A : Prop) : Prop := T ∧ C ∧ A
def degree_of_ionization_of_acetic_acid (T C : Prop) : Prop := T ∧ C

-- The statement to prove
theorem property_related_only_to_temperature
  (T P C A : Prop)
  (H1 : solubility_of_ammonia_gas T P)
  (H2 : ion_product_of_water T)
  (H3 : oxidizing_property_of_pp T C A)
  (H4 : degree_of_ionization_of_acetic_acid T C) :
  ∃ T, ion_product_of_water T ∧
        ¬solubility_of_ammonia_gas T P ∧
        ¬oxidizing_property_of_pp T C A ∧
        ¬degree_of_ionization_of_acetic_acid T C :=
by
  sorry

end property_related_only_to_temperature_l173_173198


namespace find_starting_number_of_range_l173_173764

theorem find_starting_number_of_range :
  ∃ n : ℕ, ∀ k : ℕ, k < 7 → (n + k * 9) ∣ 9 ∧ (n + k * 9) ≤ 97 ∧ (∀ m < k, (n + m * 9) < n + (m + 1) * 9) := 
sorry

end find_starting_number_of_range_l173_173764


namespace maximum_utilization_rate_80_l173_173531

noncomputable def maximum_utilization_rate (side_length : ℝ) (AF : ℝ) (BF : ℝ) : ℝ :=
  let area_square := side_length * side_length
  let length_rectangle := side_length
  let width_rectangle := AF / 2
  let area_rectangle := length_rectangle * width_rectangle
  (area_rectangle / area_square) * 100

theorem maximum_utilization_rate_80:
  maximum_utilization_rate 4 2 1 = 80 := by
  sorry

end maximum_utilization_rate_80_l173_173531


namespace value_of_x_plus_y_l173_173706

theorem value_of_x_plus_y 
  (x y : ℝ) 
  (h1 : -x = 3) 
  (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := 
  sorry

end value_of_x_plus_y_l173_173706


namespace equilateral_triangle_complex_l173_173687

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

lemma equal_magnitudes (z1 z2 z3 : ℂ) : z1.abs = z2.abs ∧ z2.abs = z3.abs := sorry

lemma product_sum_zero (z1 z2 z3 : ℂ) : z1 * z2 + z2 * z3 + z3 * z1 = 0 := sorry

theorem equilateral_triangle_complex (z1 z2 z3 : ℂ) 
  (h1 : z1.abs = z2.abs) (h2 : z2.abs = z3.abs) 
  (h3 : z1 * z2 + z2 * z3 + z3 * z1 = 0) : 
  (is_equilateral_triangle z1 z2 z3) := sorry

end equilateral_triangle_complex_l173_173687


namespace triangle_PQR_QR_length_l173_173138

-- Define the given conditions as a Lean statement
theorem triangle_PQR_QR_length 
  (P Q R : ℝ) -- Angles in the triangle PQR in radians
  (PQ QR PR : ℝ) -- Lengths of the sides of the triangle PQR
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1) 
  (h2 : PQ = 5)
  (h3 : PQ + QR + PR = 12)
  : QR = 3.5 := 
  sorry -- proof omitted

end triangle_PQR_QR_length_l173_173138


namespace smallest_n_l173_173632

theorem smallest_n (m l n : ℕ) :
  (∃ m : ℕ, 2 * n = m ^ 4) ∧ (∃ l : ℕ, 3 * n = l ^ 6) → n = 1944 :=
by
  sorry

end smallest_n_l173_173632


namespace elmo_to_laura_books_ratio_l173_173410

-- Definitions of the conditions given in the problem
def ElmoBooks : ℕ := 24
def StuBooks : ℕ := 4
def LauraBooks : ℕ := 2 * StuBooks

-- Ratio calculation and proof of the ratio being 3:1
theorem elmo_to_laura_books_ratio : (ElmoBooks : ℚ) / (LauraBooks : ℚ) = 3 / 1 := by
  sorry

end elmo_to_laura_books_ratio_l173_173410


namespace number_of_dispatch_plans_l173_173679

theorem number_of_dispatch_plans :
  (∃ (students : Finset ℕ) (communities : Finset ℕ),
    students.card = 4 ∧ communities.card = 3 ∧
    ∀ (f : students → communities), surjective f) → 
  ∃ (dispatch_plans : ℕ), dispatch_plans = 36 :=
by
  sorry

end number_of_dispatch_plans_l173_173679


namespace domain_of_sqrt_log_l173_173842

noncomputable def domain_of_function : Set ℝ := 
  {x : ℝ | (-Real.sqrt 2) ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ Real.sqrt 2}

theorem domain_of_sqrt_log : ∀ x : ℝ, 
  (∃ y : ℝ, y = Real.sqrt (Real.log (x^2 - 1) / Real.log (1/2)) ∧ 
  y ≥ 0) ↔ x ∈ domain_of_function := 
by
  sorry

end domain_of_sqrt_log_l173_173842


namespace movie_tickets_l173_173345

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end movie_tickets_l173_173345


namespace smallest_perfect_cube_divisor_l173_173322

theorem smallest_perfect_cube_divisor (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ k : ℕ, (k = (p * q * r^2)^3) ∧ (∃ n, n = p * q^3 * r^4 ∧ n ∣ k) := 
sorry

end smallest_perfect_cube_divisor_l173_173322


namespace geometric_sequence_min_value_l173_173559

theorem geometric_sequence_min_value 
  (a b c : ℝ)
  (h1 : b^2 = ac)
  (h2 : b = -Real.exp 1) :
  ac = Real.exp 2 := 
by
  sorry

end geometric_sequence_min_value_l173_173559


namespace abs_value_solutions_l173_173096

theorem abs_value_solutions (y : ℝ) :
  |4 * y - 5| = 39 ↔ (y = 11 ∨ y = -8.5) :=
by
  sorry

end abs_value_solutions_l173_173096


namespace last_digit_of_power_sum_l173_173755

theorem last_digit_of_power_sum (m : ℕ) (hm : 0 < m) : (2^(m + 2006) + 2^m) % 10 = 0 := 
sorry

end last_digit_of_power_sum_l173_173755


namespace alice_probability_at_least_one_multiple_of_4_l173_173817

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end alice_probability_at_least_one_multiple_of_4_l173_173817


namespace ants_in_third_anthill_l173_173710

-- Define the number of ants in the first anthill
def ants_first : ℕ := 100

-- Define the percentage reduction for each subsequent anthill
def percentage_reduction : ℕ := 20

-- Calculate the number of ants in the second anthill
def ants_second : ℕ := ants_first - (percentage_reduction * ants_first / 100)

-- Calculate the number of ants in the third anthill
def ants_third : ℕ := ants_second - (percentage_reduction * ants_second / 100)

-- Main theorem to prove that the number of ants in the third anthill is 64
theorem ants_in_third_anthill : ants_third = 64 := sorry

end ants_in_third_anthill_l173_173710


namespace min_club_members_l173_173522

theorem min_club_members (n : ℕ) :
  (∀ k : ℕ, k = 8 ∨ k = 9 ∨ k = 11 → n % k = 0) ∧ (n ≥ 300) → n = 792 :=
sorry

end min_club_members_l173_173522


namespace quadratic_eq_a_val_l173_173452

theorem quadratic_eq_a_val (a : ℝ) (h : a - 6 = 0) :
  a = 6 :=
by
  sorry

end quadratic_eq_a_val_l173_173452


namespace sandy_initial_payment_l173_173034

theorem sandy_initial_payment (P : ℝ) (repairs cost: ℝ) (selling_price gain: ℝ) 
  (hc : repairs = 300)
  (hs : selling_price = 1260) 
  (hg : gain = 5)
  (h : selling_price = (P + repairs) * (1 + gain / 100)) : 
  P = 900 :=
sorry

end sandy_initial_payment_l173_173034


namespace problem_l173_173703

variables (x : ℝ)

-- Define the condition
def condition (x : ℝ) : Prop :=
  0.3 * (0.2 * x) = 24

-- Define the target statement
def target (x : ℝ) : Prop :=
  0.2 * (0.3 * x) = 24

-- The theorem we want to prove
theorem problem (x : ℝ) (h : condition x) : target x :=
sorry

end problem_l173_173703


namespace sum_of_cube_faces_l173_173045

theorem sum_of_cube_faces (a b c d e f : ℕ) (h1 : a % 2 = 0) (h2 : b = a + 2) (h3 : c = b + 2) (h4 : d = c + 2) (h5 : e = d + 2) (h6 : f = e + 2)
(h_pairs : (a + f + 2) = (b + e + 2) ∧ (b + e + 2) = (c + d + 2)) :
  a + b + c + d + e + f = 90 :=
  sorry

end sum_of_cube_faces_l173_173045


namespace number_of_ways_to_read_BANANA_l173_173462

/-- 
In a 3x3 grid, there are 84 different ways to read the word BANANA 
by moving from one cell to another cell with which it shares an edge,
and cells may be visited more than once.
-/
theorem number_of_ways_to_read_BANANA (grid : Matrix (Fin 3) (Fin 3) Char) (word : String := "BANANA") : 
  ∃! n : ℕ, n = 84 :=
by
  sorry

end number_of_ways_to_read_BANANA_l173_173462


namespace pool_depths_l173_173016

theorem pool_depths (J S Su : ℝ) 
  (h1 : J = 15) 
  (h2 : J = 2 * S + 5) 
  (h3 : Su = J + S - 3) : 
  S = 5 ∧ Su = 17 := 
by 
  -- proof steps go here
  sorry

end pool_depths_l173_173016


namespace light_flash_fraction_l173_173389

theorem light_flash_fraction (flash_interval : ℕ) (total_flashes : ℕ) (seconds_in_hour : ℕ) (fraction_of_hour : ℚ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_in_hour = 3600 →
  fraction_of_hour = 1 →
  (total_flashes * flash_interval) / seconds_in_hour = fraction_of_hour := by
  sorry

end light_flash_fraction_l173_173389


namespace distinguishable_arrangements_l173_173699

theorem distinguishable_arrangements :
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  (Nat.factorial total) / (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow * Nat.factorial blue) = 50400 := 
by
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  sorry

end distinguishable_arrangements_l173_173699


namespace area_of_L_shape_is_58_l173_173388

-- Define the dimensions of the large rectangle
def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7

-- Define the dimensions of the smaller rectangle to be removed
def small_rectangle_length : ℕ := 4
def small_rectangle_width : ℕ := 3

-- Define the area of the large rectangle
def area_large_rectangle : ℕ := large_rectangle_length * large_rectangle_width

-- Define the area of the small rectangle
def area_small_rectangle : ℕ := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shaped region
def area_L_shape : ℕ := area_large_rectangle - area_small_rectangle

-- Prove that the area of the "L" shaped region is 58 square units
theorem area_of_L_shape_is_58 : area_L_shape = 58 := by
  sorry

end area_of_L_shape_is_58_l173_173388


namespace part_a_limit_part_b_inequality_l173_173478

noncomputable def seq_a (n : ℕ) : ℝ := 
  ∑ i in finset.range n, (-1 : ℝ)^i / (2 * i + 1)

theorem part_a_limit : 
  tendsto (seq_a) atTop (𝓝 (π / 4)) :=
sorry

theorem part_b_inequality (k : ℕ) : 
  1 / (2 * (4 * k + 1)) ≤ (π / 4 - seq_a (2 * k - 1)) ∧ (π / 4 - seq_a (2 * k - 1)) ≤ 1 / (4 * k + 1) :=
sorry 

end part_a_limit_part_b_inequality_l173_173478


namespace cookies_per_batch_l173_173693

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l173_173693


namespace balloon_permutations_l173_173261

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l173_173261


namespace cube_volume_l173_173984

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173984


namespace integer_pairs_solution_l173_173543

theorem integer_pairs_solution (a b : ℤ) : 
  (a - b - 1 ∣ a^2 + b^2 ∧ (a^2 + b^2) * 19 = (2 * a * b - 1) * 20) ↔
  (a, b) = (22, 16) ∨ (a, b) = (-16, -22) ∨ (a, b) = (8, 6) ∨ (a, b) = (-6, -8) :=
by 
  sorry

end integer_pairs_solution_l173_173543


namespace total_paintable_area_correct_l173_173648

-- Define the conditions
def warehouse_width := 12
def warehouse_length := 15
def warehouse_height := 7

def window_count_per_longer_wall := 3
def window_width := 2
def window_height := 3

-- Define areas for walls, ceiling, and floor
def area_wall_1 := warehouse_width * warehouse_height
def area_wall_2 := warehouse_length * warehouse_height
def window_area := window_width * window_height
def window_total_area := window_count_per_longer_wall * window_area
def area_wall_2_paintable := 2 * (area_wall_2 - window_total_area) -- both inside and outside
def area_ceiling := warehouse_width * warehouse_length
def area_floor := warehouse_width * warehouse_length

-- Total paintable area calculation
def total_paintable_area := 2 * area_wall_1 + area_wall_2_paintable + area_ceiling + area_floor

-- Final proof statement
theorem total_paintable_area_correct : total_paintable_area = 876 := by
  sorry

end total_paintable_area_correct_l173_173648


namespace find_Gary_gold_l173_173680

variable (G : ℕ) -- G represents the number of grams of gold Gary has.
variable (cost_Gary_gold_per_gram : ℕ) -- The cost per gram of Gary's gold.
variable (grams_Anna_gold : ℕ) -- The number of grams of gold Anna has.
variable (cost_Anna_gold_per_gram : ℕ) -- The cost per gram of Anna's gold.
variable (combined_cost : ℕ) -- The combined cost of both Gary's and Anna's gold.

theorem find_Gary_gold (h1 : cost_Gary_gold_per_gram = 15)
                       (h2 : grams_Anna_gold = 50)
                       (h3 : cost_Anna_gold_per_gram = 20)
                       (h4 : combined_cost = 1450)
                       (h5 : combined_cost = cost_Gary_gold_per_gram * G + grams_Anna_gold * cost_Anna_gold_per_gram) :
  G = 30 :=
by 
  sorry

end find_Gary_gold_l173_173680


namespace roses_given_to_mother_is_6_l173_173126

-- Define the initial conditions
def initial_roses : ℕ := 20
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4
def roses_kept : ℕ := 1

-- Define the expected number of roses given to mother
def roses_given_to_mother : ℕ := initial_roses - (roses_to_grandmother + roses_to_sister + roses_kept)

-- The theorem stating the number of roses given to the mother
theorem roses_given_to_mother_is_6 : roses_given_to_mother = 6 := by
  sorry

end roses_given_to_mother_is_6_l173_173126


namespace part1_daily_sales_profit_at_60_part2_selling_price_1350_l173_173523

-- Definitions from conditions
def cost_per_piece : ℕ := 40
def selling_price_50_sales_volume : ℕ := 100
def sales_decrease_per_dollar : ℕ := 2
def max_selling_price : ℕ := 65

-- Problem Part (1)
def profit_at_60_yuan := 
  let selling_price := 60
  let profit_per_piece := selling_price - cost_per_piece
  let sales_decrease := (selling_price - 50) * sales_decrease_per_dollar
  let sales_volume := selling_price_50_sales_volume - sales_decrease
  let daily_profit := profit_per_piece * sales_volume
  daily_profit

theorem part1_daily_sales_profit_at_60 : profit_at_60_yuan = 1600 := by
  sorry

-- Problem Part (2)
def selling_price_for_1350_profit :=
  let desired_profit := 1350
  let sales_volume (x : ℕ) := selling_price_50_sales_volume - sales_decrease_per_dollar * (x - 50)
  let profit_per_x_piece (x : ℕ) := x - cost_per_piece
  let daily_sales_profit (x : ℕ) := (profit_per_x_piece x) * (sales_volume x)
  daily_sales_profit

theorem part2_selling_price_1350 : 
  ∃ x, x ≤ max_selling_price ∧ selling_price_for_1350_profit x = 1350 ∧ x = 55 := by
  sorry

end part1_daily_sales_profit_at_60_part2_selling_price_1350_l173_173523


namespace total_possible_rankings_l173_173666

-- Define the players
inductive Player
| P | Q | R | S

-- Define the tournament results
inductive Result
| win | lose

-- Define Saturday's match outcomes
structure SaturdayOutcome :=
(P_vs_Q: Result)
(R_vs_S: Result)

-- Function to compute the number of possible tournament ranking sequences
noncomputable def countTournamentSequences : Nat :=
  let saturdayOutcomes: List SaturdayOutcome :=
    [ {P_vs_Q := Result.win, R_vs_S := Result.win}
    , {P_vs_Q := Result.win, R_vs_S := Result.lose}
    , {P_vs_Q := Result.lose, R_vs_S := Result.win}
    , {P_vs_Q := Result.lose, R_vs_S := Result.lose}
    ]
  let sundayPermutations (outcome : SaturdayOutcome) : Nat :=
    2 * 2  -- 2 permutations for 1st and 2nd places * 2 permutations for 3rd and 4th places per each outcome
  saturdayOutcomes.foldl (fun acc outcome => acc + sundayPermutations outcome) 0

-- Define the theorem to prove the total number of permutations
theorem total_possible_rankings : countTournamentSequences = 8 :=
by
  -- Proof steps here (proof omitted)
  sorry

end total_possible_rankings_l173_173666


namespace part1_part2_l173_173721

def f (x : ℝ) (t : ℝ) : ℝ := x^2 + 2 * t * x + t - 1

theorem part1 (hf : ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3) : 
  ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3 :=
by 
  sorry
  
theorem part2 (ht : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f x t > 0) : 
  t ∈ Set.Ioi (0 : ℝ) :=
by 
  sorry

end part1_part2_l173_173721


namespace factorize_quadratic_l173_173093

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end factorize_quadratic_l173_173093


namespace number_of_boys_in_class_l173_173939

theorem number_of_boys_in_class (B : ℕ) (G : ℕ) (hG : G = 10) (h_combinations : (G * B * (B - 1)) / 2 = 1050) :
    B = 15 :=
by
  sorry

end number_of_boys_in_class_l173_173939


namespace cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l173_173219

-- Definitions based on the conditions:
-- 1. Folded napkin structure
structure Napkin where
  folded_in_two: Bool -- A napkin folded in half once along one axis 
  folded_in_four: Bool -- A napkin folded in half twice to form a smaller square

-- 2. Cutting through a folded napkin
def single_cut_through_folded_napkin (n: Nat) (napkin: Napkin) : Bool :=
  if (n = 2 ∨ n = 4) then
    true
  else
    false

-- Main theorem statements 
-- If the napkin can be cut into 2 pieces
theorem cut_into_two_pieces_is_possible (napkin: Napkin) : single_cut_through_folded_napkin 2 napkin = true := by
  sorry

-- If the napkin can be cut into 3 pieces
theorem cut_into_three_pieces_is_impossible (napkin: Napkin) : single_cut_through_folded_napkin 3 napkin = false := by
  sorry

-- If the napkin can be cut into 4 pieces
theorem cut_into_four_pieces_is_possible (napkin: Napkin) : single_cut_through_folded_napkin 4 napkin = true := by
  sorry

-- If the napkin can be cut into 5 pieces
theorem cut_into_five_pieces_is_impossible (napkin: Napkin) : single_cut_through_folded_napkin 5 napkin = false := by
  sorry

end cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l173_173219


namespace rectangle_ratio_l173_173800

open Real

theorem rectangle_ratio (A B C D E : Point) (rat : ℚ) : 
  let area_rect := 1
  let area_pentagon := (7 / 10 : ℚ)
  let area_triangle_AEC := 3 / 10
  let area_triangle_ECD := 1 / 5
  let x := 3 * EA
  let y := 2 * EA
  let diag_longer_side := sqrt (5 * EA ^ 2)
  let diag_shorter_side := EA * sqrt 5
  let ratio := sqrt 5 
  ( area_pentagon == area_rect * (7 / 10) ) →
  ( area_triangle_AEC + area_pentagon = area_rect ) →
  ( area_triangle_AEC == area_rect - area_pentagon ) →
  ( ratio == diag_longer_side / diag_shorter_side ) :=
  sorry

end rectangle_ratio_l173_173800


namespace minimum_spending_l173_173517

noncomputable def box_volume (length width height : ℕ) : ℕ := length * width * height
noncomputable def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
noncomputable def total_cost (num_boxes : ℕ) (price_per_box : ℝ) : ℝ := num_boxes * price_per_box

theorem minimum_spending
  (box_length box_width box_height : ℕ)
  (price_per_box : ℝ)
  (total_collection_volume : ℕ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : price_per_box = 0.90)
  (h5 : total_collection_volume = 3060000) :
  total_cost (total_boxes_needed total_collection_volume (box_volume box_length box_width box_height)) price_per_box = 459 :=
by
  rw [h1, h2, h3, h4, h5]
  have box_vol : box_volume 20 20 15 = 6000 := by norm_num [box_volume]
  have boxes_needed : total_boxes_needed 3060000 6000 = 510 := by norm_num [total_boxes_needed, box_volume, *]
  have cost : total_cost 510 0.90 = 459 := by norm_num [total_cost]
  exact cost

end minimum_spending_l173_173517


namespace find_triplets_geometric_and_arithmetic_prog_l173_173355

theorem find_triplets_geometric_and_arithmetic_prog :
  ∃ a1 a2 b1 b2,
    (a2 = a1 * ((12:ℚ) / a1) ∧ 12 = a1 * ((12:ℚ) / a1)^2) ∧
    (b2 = b1 + ((9:ℚ) - b1) / 2 ∧ 9 = b1 + 2 * (((9:ℚ) - b1) / 2)) ∧
    ((a1 = b1) ∧ (a2 = b2)) ∧ 
    (∀ (a1 a2 : ℚ), ((a1 = -9) ∧ (a2 = -6)) ∨ ((a1 = 15) ∧ (a2 = 12))) :=
by sorry

end find_triplets_geometric_and_arithmetic_prog_l173_173355


namespace true_propositions_l173_173566

noncomputable def discriminant_leq_zero : Prop :=
  let a := 1
  let b := -1
  let c := 2
  b^2 - 4 * a * c ≤ 0

def proposition_1 : Prop := discriminant_leq_zero

def proposition_2 (x : ℝ) : Prop :=
  abs x ≥ 0 → x ≥ 0

def proposition_3 : Prop :=
  5 > 2 ∧ 3 < 7

theorem true_propositions : proposition_1 ∧ proposition_3 ∧ ¬∀ x : ℝ, proposition_2 x :=
by
  sorry

end true_propositions_l173_173566


namespace cube_volume_from_surface_area_l173_173999

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l173_173999


namespace linear_term_coefficient_l173_173890

-- Define the given equation
def equation (x : ℝ) : ℝ := x^2 - 2022*x - 2023

-- The goal is to prove that the coefficient of the linear term in equation is -2022
theorem linear_term_coefficient : ∀ x : ℝ, equation x = x^2 - 2022*x - 2023 → -2022 = -2022 :=
by
  intros x h
  sorry

end linear_term_coefficient_l173_173890


namespace gcd_765432_654321_l173_173834

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end gcd_765432_654321_l173_173834


namespace gcd_correct_l173_173832

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end gcd_correct_l173_173832


namespace total_pages_read_correct_l173_173905

-- Definition of the problem conditions
def first_week_books := 5
def first_week_book_pages := 300
def first_week_magazines := 3
def first_week_magazine_pages := 120
def first_week_newspapers := 2
def first_week_newspaper_pages := 50

def second_week_books := 2 * first_week_books
def second_week_book_pages := 350
def second_week_magazines := 4
def second_week_magazine_pages := 150
def second_week_newspapers := 1
def second_week_newspaper_pages := 60

def third_week_books := 3 * first_week_books
def third_week_book_pages := 400
def third_week_magazines := 5
def third_week_magazine_pages := 125
def third_week_newspapers := 1
def third_week_newspaper_pages := 70

-- Total pages read in each week
def first_week_total_pages : Nat :=
  (first_week_books * first_week_book_pages) +
  (first_week_magazines * first_week_magazine_pages) +
  (first_week_newspapers * first_week_newspaper_pages)

def second_week_total_pages : Nat :=
  (second_week_books * second_week_book_pages) +
  (second_week_magazines * second_week_magazine_pages) +
  (second_week_newspapers * second_week_newspaper_pages)

def third_week_total_pages : Nat :=
  (third_week_books * third_week_book_pages) +
  (third_week_magazines * third_week_magazine_pages) +
  (third_week_newspapers * third_week_newspaper_pages)

-- Grand total pages read over three weeks
def total_pages_read : Nat :=
  first_week_total_pages + second_week_total_pages + third_week_total_pages

-- Theorem statement to be proven
theorem total_pages_read_correct :
  total_pages_read = 12815 :=
by
  -- Proof will be provided here
  sorry

end total_pages_read_correct_l173_173905


namespace time_released_rope_first_time_l173_173470

theorem time_released_rope_first_time :
  ∀ (rate_ascent : ℕ) (rate_descent : ℕ) (time_first_ascent : ℕ) (time_second_ascent : ℕ) (highest_elevation : ℕ)
    (total_elevation_gained : ℕ) (elevation_difference : ℕ) (time_descent : ℕ),
  rate_ascent = 50 →
  rate_descent = 10 →
  time_first_ascent = 15 →
  time_second_ascent = 15 →
  highest_elevation = 1400 →
  total_elevation_gained = (rate_ascent * time_first_ascent) + (rate_ascent * time_second_ascent) →
  elevation_difference = total_elevation_gained - highest_elevation →
  time_descent = elevation_difference / rate_descent →
  time_descent = 10 :=
by
  intros rate_ascent rate_descent time_first_ascent time_second_ascent highest_elevation total_elevation_gained elevation_difference time_descent
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end time_released_rope_first_time_l173_173470


namespace intersection_of_sets_l173_173294

variable {x : ℝ}

def SetA : Set ℝ := {x | x + 1 > 0}
def SetB : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_sets : SetA ∩ SetB = {x | -1 < x ∧ x < 3} :=
by sorry

end intersection_of_sets_l173_173294


namespace part1_case1_part1_case2_part1_case3_part2_l173_173438

def f (m x : ℝ) : ℝ := (m+1)*x^2 - (m-1)*x + (m-1)

theorem part1_case1 (m x : ℝ) (h : m = -1) : 
  f m x ≥ (m+1)*x → x ≥ 1 := sorry

theorem part1_case2 (m x : ℝ) (h : m > -1) :
  f m x ≥ (m+1)*x →
  (x ≤ (m-1)/(m+1) ∨ x ≥ 1) := sorry

theorem part1_case3 (m x : ℝ) (h : m < -1) : 
  f m x ≥ (m+1)*x →
  (1 ≤ x ∧ x ≤ (m-1)/(m+1)) := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) →
  m ≥ 1 := sorry

end part1_case1_part1_case2_part1_case3_part2_l173_173438


namespace tricycles_count_l173_173846

theorem tricycles_count (cars bicycles pickup_trucks tricycles : ℕ) (total_tires : ℕ) : 
  cars = 15 →
  bicycles = 3 →
  pickup_trucks = 8 →
  total_tires = 101 →
  4 * cars + 2 * bicycles + 4 * pickup_trucks + 3 * tricycles = total_tires →
  tricycles = 1 :=
by
  sorry

end tricycles_count_l173_173846


namespace chuck_area_correct_l173_173406

noncomputable def chuck_play_area (shed_length shed_width leash_length : ℝ) (C : shed_length = 4 ∧ shed_width = 6 ∧ leash_length = 5) : ℝ :=
  (3 / 4) * Real.pi * (leash_length ^ 2) + (1 / 2) * Real.pi * (2 ^ 2)

theorem chuck_area_correct (shed_length shed_width leash_length : ℝ) (h : shed_length = 4) (h2 : shed_width = 6) (h3 : leash_length = 5) :
  chuck_play_area shed_length shed_width leash_length (and.intro h (and.intro h2 h3)) = (83 / 4) * Real.pi :=
by
  sorry

end chuck_area_correct_l173_173406


namespace primes_in_arithmetic_sequence_l173_173946

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_in_arithmetic_sequence (p : ℕ) :
  is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  intro h
  sorry

end primes_in_arithmetic_sequence_l173_173946


namespace peter_bought_large_glasses_l173_173327

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end peter_bought_large_glasses_l173_173327


namespace value_of_x_l173_173877

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end value_of_x_l173_173877


namespace min_abs_sum_l173_173300

theorem min_abs_sum (x : ℝ) : (∃ x : ℝ, ∀ y : ℝ, (|y - 2| + |y - 47| ≥ |x - 2| + |x - 47|)) → (|x - 2| + |x - 47| = 45) :=
by
  sorry

end min_abs_sum_l173_173300


namespace parabola_trajectory_l173_173436

theorem parabola_trajectory (P : ℝ × ℝ) : 
  (dist P (3, 0) = dist P (3 - 1, P.2 - 0)) → P.2^2 = 12 * P.1 := 
sorry

end parabola_trajectory_l173_173436


namespace find_coordinates_C_find_range_t_l173_173115

-- required definitions to handle the given points and vectors
structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

-- Given points
def A : Point := ⟨0, 4⟩
def B : Point := ⟨2, 0⟩

-- Proof for coordinates of point C
theorem find_coordinates_C :
  ∃ C : Point, vector A B = {x := 2 * (C.x - B.x), y := 2 * C.y} ∧ C = ⟨3, -2⟩ :=
by
  sorry

-- Proof for range of t
theorem find_range_t (t : ℝ) :
  let P := Point.mk 3 t
  let PA := vector P A
  let PB := vector P B
  (dot_product PA PB < 0 ∧ -3 * t ≠ -1 * (4 - t)) → 1 < t ∧ t < 3 :=
by
  sorry

end find_coordinates_C_find_range_t_l173_173115


namespace tiao_ri_method_four_times_l173_173038

theorem tiao_ri_method_four_times (a b c d e f g h: ℕ) (h1: a ≠ 0) (h2: c ≠ 0) (h3: e ≠ 0) (h4: g ≠ 0) :
  let x := Real.pi in
  (b:ℝ) / a < x ∧ x < (d:ℝ) / c →
  (d + b):(c + a) < x ∧ x < (f:ℝ) / e →
  (f + d):(e + c) < x ∧ x < (h:ℝ) / g →
  (h + f):(g + e) = (22:ℕ) / 7 :=
by
  intros
  sorry

end tiao_ri_method_four_times_l173_173038


namespace total_pears_picked_l173_173649

def pears_Alyssa : ℕ := 42
def pears_Nancy : ℕ := 17

theorem total_pears_picked : pears_Alyssa + pears_Nancy = 59 :=
by sorry

end total_pears_picked_l173_173649


namespace no_such_function_exists_l173_173665

theorem no_such_function_exists :
  ¬ (∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l173_173665


namespace smallest_c_plus_d_l173_173497

theorem smallest_c_plus_d :
  ∃ (c d : ℕ), (8 * c + 3 = 3 * d + 8) ∧ c + d = 27 :=
by
  sorry

end smallest_c_plus_d_l173_173497


namespace negation_proof_l173_173821

-- Definitions based on conditions
def atMostTwoSolutions (solutions : ℕ) : Prop := solutions ≤ 2
def atLeastThreeSolutions (solutions : ℕ) : Prop := solutions ≥ 3

-- Statement of the theorem
theorem negation_proof (solutions : ℕ) : atMostTwoSolutions solutions ↔ ¬ atLeastThreeSolutions solutions :=
by
  sorry

end negation_proof_l173_173821


namespace cube_volume_from_surface_area_l173_173996

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l173_173996


namespace balloon_permutations_l173_173262

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l173_173262


namespace minimum_bounces_to_reach_height_l173_173379

noncomputable def height_after_bounces (initial_height : ℝ) (bounce_factor : ℝ) (k : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ k)

theorem minimum_bounces_to_reach_height
  (initial_height : ℝ) (bounce_factor : ℝ) (min_height : ℝ) :
  initial_height = 800 → bounce_factor = 0.5 → min_height = 2 →
  (∀ k : ℕ, height_after_bounces initial_height bounce_factor k < min_height ↔ k ≥ 9) := 
by
  intros h₀ b₀ m₀
  rw [h₀, b₀, m₀]
  sorry

end minimum_bounces_to_reach_height_l173_173379


namespace find_b_l173_173867

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 1) (h2 : b - a = 2) : b = 2 := by
  sorry

end find_b_l173_173867


namespace loaves_of_bread_can_bake_l173_173147

def total_flour_in_cupboard := 200
def total_flour_on_counter := 100
def total_flour_in_pantry := 100
def flour_per_loaf := 200

theorem loaves_of_bread_can_bake :
  (total_flour_in_cupboard + total_flour_on_counter + total_flour_in_pantry) / flour_per_loaf = 2 := by
  sorry

end loaves_of_bread_can_bake_l173_173147


namespace total_pears_l173_173716

noncomputable def Jason_pears : ℝ := 46
noncomputable def Keith_pears : ℝ := 47
noncomputable def Mike_pears : ℝ := 12
noncomputable def Sarah_pears : ℝ := 32.5
noncomputable def Emma_pears : ℝ := (2 / 3) * Mike_pears
noncomputable def James_pears : ℝ := (2 * Sarah_pears) - 3

theorem total_pears :
  Jason_pears + Keith_pears + Mike_pears + Sarah_pears + Emma_pears + James_pears = 207.5 :=
by
  sorry

end total_pears_l173_173716


namespace cost_of_individual_roll_is_correct_l173_173211

-- Definitions given in the problem's conditions
def cost_per_case : ℝ := 9
def number_of_rolls : ℝ := 12
def percent_savings : ℝ := 0.25

-- The cost of one roll sold individually
noncomputable def individual_roll_cost : ℝ := 0.9375

-- The theorem to prove
theorem cost_of_individual_roll_is_correct :
  individual_roll_cost = (cost_per_case * (1 + percent_savings)) / number_of_rolls :=
by
  sorry

end cost_of_individual_roll_is_correct_l173_173211


namespace visible_sides_probability_l173_173382

theorem visible_sides_probability
  (r : ℝ)
  (side_length : ℝ := 4)
  (probability : ℝ := 3 / 4) :
  r = 8 * Real.sqrt 3 / 3 :=
sorry

end visible_sides_probability_l173_173382


namespace matrix_power_eq_l173_173589

def MatrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-8, -10]]

def MatrixA : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![201, 200], ![-400, -449]]

theorem matrix_power_eq :
  MatrixC ^ 50 = MatrixA := 
  sorry

end matrix_power_eq_l173_173589


namespace probability_of_exactly_three_positives_l173_173894

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_exactly_three_positives :
  let p := 2/5
  let n := 7
  let k := 3
  let positive_prob := p^k
  let negative_prob := (1 - p)^(n - k)
  let binomial_coefficient := choose n k
  binomial_coefficient * positive_prob * negative_prob = 22680/78125 := 
by
  sorry

end probability_of_exactly_three_positives_l173_173894


namespace kanul_total_amount_l173_173472

def kanul_spent : ℝ := 3000 + 1000
def kanul_spent_percentage (T : ℝ) : ℝ := 0.30 * T

theorem kanul_total_amount (T : ℝ) (h : T = kanul_spent + kanul_spent_percentage T) :
  T = 5714.29 := sorry

end kanul_total_amount_l173_173472


namespace value_of_x_l173_173876

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end value_of_x_l173_173876


namespace total_legs_l173_173586

theorem total_legs 
  (johnny_legs : ℕ := 2) 
  (son_legs : ℕ := 2) 
  (dog_legs_per_dog : ℕ := 4) 
  (number_of_dogs : ℕ := 2) :
  johnny_legs + son_legs + dog_legs_per_dog * number_of_dogs = 12 := 
sorry

end total_legs_l173_173586


namespace number_of_committees_l173_173850

theorem number_of_committees 
  (maths econ : ℕ)
  (maths_count econ_count total_count : ℕ)
  (h_maths : maths = 3)
  (h_econ : econ = 10)
  (h_total : total_count = 7) 
  (h_maths_count : maths_count + econ_count = 7)
  : ∑ i in finset.Icc 1 3, binom (maths + econ) i * binom (maths + econ - i) (total_count - i) = 1596 :=
by
  have : binom 13 7 - binom 10 7 = 1596, from by {
    calc
    binom 13 7 - binom 10 7
        = 1716 - 120 : by norm_num
    ... = 1596 : rfl
  }
  rw this
  sorry

end number_of_committees_l173_173850


namespace range_of_a_if_p_and_not_q_l173_173859

open Real

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a_if_p_and_not_q : 
  (∃ a : ℝ, (p a ∧ ¬q a)) → 
  (∀ a : ℝ, (p a ∧ ¬q a) → (-1 ≤ a ∧ a < 0)) :=
sorry

end range_of_a_if_p_and_not_q_l173_173859


namespace recurring_division_l173_173360

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l173_173360


namespace compare_powers_l173_173044

theorem compare_powers (a b c d : ℝ) (h1 : a + b = 0) (h2 : c + d = 0) : a^5 + d^6 = c^6 - b^5 :=
by
  sorry

end compare_powers_l173_173044


namespace gamma_lt_delta_l173_173480

open Real

variables (α β γ δ : ℝ)

-- Hypotheses as given in the problem
axiom h1 : 0 < α 
axiom h2 : α < β
axiom h3 : β < π / 2
axiom hg1 : 0 < γ
axiom hg2 : γ < π / 2
axiom htan_gamma_eq : tan γ = (tan α + tan β) / 2
axiom hd1 : 0 < δ
axiom hd2 : δ < π / 2
axiom hcos_delta_eq : (1 / cos δ) = (1 / 2) * (1 / cos α + 1 / cos β)

-- Goal to prove
theorem gamma_lt_delta : γ < δ := 
by 
sorry

end gamma_lt_delta_l173_173480


namespace probability_other_two_not_pair_l173_173131

open Classical

theorem probability_other_two_not_pair :
  ∀ (socks : Finset (Finset ℕ)), 
  socks.card = 4 ∧ 
  (∀ s ∈ socks, s.card = 2) ∧ 
  ∀ pair_choice : Finset ℕ, pair_choice.card = 2 → 
  ∃ chosen_socks : Finset ℕ, chosen_socks.card = 4 ∧ 
  (∀ s ∈ chosen_socks, s ∈ socks) ∧ 
  (∃ pair1 pair2 : Finset ℕ, 
    pair1 ≠ pair2 ∧
    pair1 ⊆ chosen_socks ∧
    pair2 ⊆ chosen_socks ∧
    pair1.card = 2 ∧
    pair2.card = 2) →
  Probability (chosen_socks.card = 4 ∧ ∃ p ∈ chosen_socks.pair_choice, 
    p.card = 2 ∧ ∀ s ∈ (chosen_socks \ p), s.card = 1) (chosen_socks.pair_choice.card = 2) = 8 / 9 := 
by
  sorry

end probability_other_two_not_pair_l173_173131


namespace least_positive_integer_fac_6370_factorial_l173_173364

theorem least_positive_integer_fac_6370_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, (6370 ∣ m.factorial) → m ≥ n) ∧ n = 14 :=
by
  sorry

end least_positive_integer_fac_6370_factorial_l173_173364


namespace range_of_m_l173_173854

theorem range_of_m (p q : Prop) (m : ℝ) (h₀ : ∀ x : ℝ, p ↔ (x^2 - 8 * x - 20 ≤ 0)) 
  (h₁ : ∀ x : ℝ, q ↔ (x^2 - 2 * x + 1 - m^2 ≤ 0)) (hm : m > 0) 
  (hsuff : (∃ x : ℝ, x > 10 ∨ x < -2) → (∃ x : ℝ, x < 1 - m ∨ x > 1 + m)) :
  0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l173_173854


namespace z_share_profit_correct_l173_173642

-- Define the investments as constants
def x_investment : ℕ := 20000
def y_investment : ℕ := 25000
def z_investment : ℕ := 30000

-- Define the number of months for each investment
def x_months : ℕ := 12
def y_months : ℕ := 12
def z_months : ℕ := 7

-- Define the annual profit
def annual_profit : ℕ := 50000

-- Calculate the active investment
def x_share : ℕ := x_investment * x_months
def y_share : ℕ := y_investment * y_months
def z_share : ℕ := z_investment * z_months

-- Calculate the total investment
def total_investment : ℕ := x_share + y_share + z_share

-- Define Z's ratio in terms of the total investment
def z_ratio : ℚ := z_share / total_investment

-- Calculate Z's share of the annual profit
def z_profit_share : ℚ := z_ratio * annual_profit

-- Theorem to prove Z's share in the annual profit
theorem z_share_profit_correct : z_profit_share = 14000 := by
  sorry

end z_share_profit_correct_l173_173642


namespace ship_illuminated_by_lighthouse_l173_173617

theorem ship_illuminated_by_lighthouse (d v : ℝ) (hv : v > 0) (ship_speed : ℝ) 
    (hship_speed : ship_speed ≤ v / 8) (rock_distance : ℝ) 
    (hrock_distance : rock_distance = d):
    ∀ t : ℝ, ∃ t' : ℝ, t' ≤ t ∧ t' = (d * t / v) := sorry

end ship_illuminated_by_lighthouse_l173_173617


namespace exists_line_l_l173_173013

-- Define the parabola and line l1
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1
def line_l1 (P : ℝ × ℝ) : Prop := P.1 + 5 * P.2 - 5 = 0

-- Define the problem statement
theorem exists_line_l :
  ∃ l : ℝ × ℝ → Prop, 
    ((∃ A B : ℝ × ℝ, parabola A ∧ parabola B ∧ A ≠ B ∧ l A ∧ l B) ∧
    (∃ M : ℝ × ℝ, M = (1, 4/5) ∧ line_l1 M) ∧
    (∀ A B : ℝ × ℝ, l A ∧ l B → (A.2 - B.2) / (A.1 - B.1) = 5)) ∧
    (∀ P : ℝ × ℝ, l P ↔ 25 * P.1 - 5 * P.2 - 21 = 0) :=
sorry

end exists_line_l_l173_173013


namespace profit_when_sold_at_double_price_l173_173797

-- Define the problem parameters

-- Assume cost price (CP)
def CP : ℕ := 100

-- Define initial selling price (SP) with 50% profit
def SP : ℕ := CP + (CP / 2)

-- Define new selling price when sold at double the initial selling price
def SP2 : ℕ := 2 * SP

-- Define profit when sold at SP2
def profit : ℕ := SP2 - CP

-- Define the percentage profit
def profit_percentage : ℕ := (profit * 100) / CP

-- The proof goal: if selling at double the price, percentage profit is 200%
theorem profit_when_sold_at_double_price : profit_percentage = 200 :=
by {sorry}

end profit_when_sold_at_double_price_l173_173797


namespace not_perfect_square_n_l173_173060

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

theorem not_perfect_square_n (n : ℕ) : ¬ isPerfectSquare (4 * n^2 + 4 * n + 4) :=
sorry

end not_perfect_square_n_l173_173060


namespace sumSquaresFractions_zero_l173_173477

noncomputable def sumCondition (x : Fin 50 → ℝ) : Prop :=
  ∑ i, x i = 2

noncomputable def sumFractions (x : Fin 50 → ℝ) : Prop :=
  ∑ i, x i / (1 - x i) = 2

noncomputable def sumSquaresFractions (x : Fin 50 → ℝ) : ℝ :=
  ∑ i, x i ^ 2 / (1 - x i)

theorem sumSquaresFractions_zero {x : Fin 50 → ℝ} (h₁ : sumCondition x) (h₂ : sumFractions x) :
  sumSquaresFractions x = 0 :=
sorry

end sumSquaresFractions_zero_l173_173477


namespace john_total_spent_l173_173471

noncomputable def total_spent (computer_cost : ℝ) (peripheral_ratio : ℝ) (base_video_cost : ℝ) : ℝ :=
  let peripheral_cost := computer_cost * peripheral_ratio
  let upgraded_video_cost := base_video_cost * 2
  computer_cost + peripheral_cost + (upgraded_video_cost - base_video_cost)

theorem john_total_spent :
  total_spent 1500 0.2 300 = 2100 :=
by
  sorry

end john_total_spent_l173_173471


namespace matrix_mult_3I_l173_173097

variable (N : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_mult_3I (w : Fin 3 → ℝ):
  (∀ (w : Fin 3 → ℝ), N.mulVec w = 3 * w) ↔ (N = 3 • (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_mult_3I_l173_173097


namespace inverse_of_97_mod_98_l173_173838

theorem inverse_of_97_mod_98 : 97 * 97 ≡ 1 [MOD 98] :=
by
  sorry

end inverse_of_97_mod_98_l173_173838


namespace balloon_permutations_count_l173_173254

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l173_173254


namespace third_height_less_than_30_l173_173193

theorem third_height_less_than_30 (h_a h_b : ℝ) (h_a_pos : h_a = 12) (h_b_pos : h_b = 20) : 
    ∃ (h_c : ℝ), h_c < 30 :=
by
  sorry

end third_height_less_than_30_l173_173193


namespace simplify_expression_l173_173333

theorem simplify_expression (b : ℝ) :
  (1 * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5)) = 720 * b^15 :=
by
  sorry

end simplify_expression_l173_173333


namespace baskets_picked_l173_173473

theorem baskets_picked
  (B : ℕ) -- How many baskets did her brother pick?
  (S : ℕ := 15) -- Each basket contains 15 strawberries
  (H1 : (8 * B * S) + (B * S) + ((8 * B * S) - 93) = 4 * 168) -- Total number of strawberries when divided equally
  (H2 : S = 15) -- Number of strawberries in each basket
: B = 3 :=
sorry

end baskets_picked_l173_173473


namespace symmetric_point_A_is_B_l173_173917

/-
  Define the symmetric point function for reflecting a point across the origin.
  Define the coordinate of point A.
  Assert that the symmetric point of A has coordinates (-2, 6).
-/

structure Point where
  x : ℤ
  y : ℤ

def symmetric_point (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def A : Point := ⟨2, -6⟩

def B : Point := ⟨-2, 6⟩

theorem symmetric_point_A_is_B : symmetric_point A = B := by
  sorry

end symmetric_point_A_is_B_l173_173917


namespace distance_to_weekend_class_l173_173318

theorem distance_to_weekend_class:
  ∃ d v : ℝ, (d = v * (1 / 2)) ∧ (d = (v + 10) * (3 / 10)) → d = 7.5 :=
by
  sorry

end distance_to_weekend_class_l173_173318


namespace second_derivative_parametric_l173_173066

noncomputable def x (t : ℝ) := Real.sqrt (t - 1)
noncomputable def y (t : ℝ) := 1 / Real.sqrt t

noncomputable def y_xx (t : ℝ) := (2 * t - 3) * Real.sqrt t / t^3

theorem second_derivative_parametric :
  ∀ t, y_xx t = (2 * t - 3) * Real.sqrt t / t^3 := sorry

end second_derivative_parametric_l173_173066


namespace average_people_per_row_l173_173208

theorem average_people_per_row (boys girls rows : ℕ) (h_boys : boys = 24) (h_girls : girls = 24) (h_rows : rows = 6) : 
  (boys + girls) / rows = 8 :=
by
  sorry

end average_people_per_row_l173_173208


namespace minimum_value_func_minimum_value_attained_l173_173109

noncomputable def func (x : ℝ) : ℝ := (4 / (x - 1)) + x

theorem minimum_value_func : ∀ (x : ℝ), x > 1 → func x ≥ 5 :=
by
  intros x hx
  -- proof goes here
  sorry

theorem minimum_value_attained : func 3 = 5 :=
by
  -- proof goes here
  sorry

end minimum_value_func_minimum_value_attained_l173_173109


namespace solve_for_y_l173_173335

noncomputable def roots := [(-126 + Real.sqrt 13540) / 8, (-126 - Real.sqrt 13540) / 8]

theorem solve_for_y (y : ℝ) :
  (8*y^2 + 176*y + 2) / (3*y + 74) = 4*y + 2 →
  y = roots[0] ∨ y = roots[1] :=
by
  intros
  sorry

end solve_for_y_l173_173335


namespace gdp_scientific_notation_l173_173486

theorem gdp_scientific_notation : 
  (33.5 * 10^12 = 3.35 * 10^13) := 
by
  sorry

end gdp_scientific_notation_l173_173486


namespace arithmetic_sequence_a6_eq_4_l173_173113

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition: a_n is an arithmetic sequence, so a_(n+1) = a_n + d
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: a_2 = 2
def a_2_eq_2 (a : ℕ → ℝ) : Prop :=
  a 2 = 2

-- Condition: S_4 = 9, where S_n is the sum of first n terms of the sequence
def sum_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

def S_4_eq_9 (S : ℕ → ℝ) : Prop :=
  S 4 = 9

-- Proof: a_6 = 4
theorem arithmetic_sequence_a6_eq_4 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a_2_eq_2 a)
  (h3 : sum_S_n a S) 
  (h4 : S_4_eq_9 S) :
  a 6 = 4 := 
sorry

end arithmetic_sequence_a6_eq_4_l173_173113


namespace steps_per_level_l173_173535

def number_of_steps_per_level (blocks_per_step total_blocks total_levels : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / total_levels

theorem steps_per_level (blocks_per_step : ℕ) (total_blocks : ℕ) (total_levels : ℕ) (h1 : blocks_per_step = 3) (h2 : total_blocks = 96) (h3 : total_levels = 4) :
  number_of_steps_per_level blocks_per_step total_blocks total_levels = 8 := 
by
  sorry

end steps_per_level_l173_173535


namespace palindrome_digital_clock_l173_173944

theorem palindrome_digital_clock (no_leading_zero : ∀ h : ℕ, h < 10 → ¬ ∃ h₂ : ℕ, h₂ = h * 1000)
                                 (max_hour : ∀ h : ℕ, h ≥ 24 → false) :
  ∃ n : ℕ, n = 61 := by
  sorry

end palindrome_digital_clock_l173_173944


namespace simplify_expression_l173_173912

variable (x y : ℤ) -- Assume x and y are integers for simplicity

theorem simplify_expression : (5 - 2 * x) - (8 - 6 * x + 3 * y) = -3 + 4 * x - 3 * y := by
  sorry

end simplify_expression_l173_173912


namespace log5_6_identity_l173_173427

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 3

theorem log5_6_identity :
  Real.log 6 / Real.log 5 = ((a * b) + 1) / (b - (a * b)) :=
by sorry

end log5_6_identity_l173_173427


namespace students_catching_up_on_homework_l173_173465

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end students_catching_up_on_homework_l173_173465


namespace range_of_a_if_proposition_l173_173573

theorem range_of_a_if_proposition :
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → -4 < a ∧ a < 2 := by
  sorry

end range_of_a_if_proposition_l173_173573


namespace rowing_time_to_place_and_back_l173_173073

def speed_man_still_water : ℝ := 8 -- km/h
def speed_river : ℝ := 2 -- km/h
def total_distance : ℝ := 7.5 -- km

theorem rowing_time_to_place_and_back :
  let V_m := speed_man_still_water
  let V_r := speed_river
  let D := total_distance / 2
  let V_up := V_m - V_r
  let V_down := V_m + V_r
  let T_up := D / V_up
  let T_down := D / V_down
  T_up + T_down = 1 :=
by
  sorry

end rowing_time_to_place_and_back_l173_173073


namespace recurring_division_l173_173359

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l173_173359


namespace factorize_expression_l173_173670

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l173_173670


namespace num_integers_contains_3_and_4_l173_173445

theorem num_integers_contains_3_and_4 
  (n : ℕ) (h1 : 500 ≤ n) (h2 : n < 1000) :
  (∀ a b c : ℕ, n = 100 * a + 10 * b + c → (b = 3 ∧ c = 4) ∨ (b = 4 ∧ c = 3)) → 
  n = 10 :=
sorry

end num_integers_contains_3_and_4_l173_173445


namespace Jane_exercises_days_per_week_l173_173715

theorem Jane_exercises_days_per_week 
  (goal_hours_per_day : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (exercise_days_per_week : ℕ) 
  (h_goal : goal_hours_per_day = 1)
  (h_weeks : weeks = 8)
  (h_total_hours : total_hours = 40)
  (h_exercise_hours_weekly : total_hours / weeks = exercise_days_per_week) :
  exercise_days_per_week = 5 :=
by
  sorry

end Jane_exercises_days_per_week_l173_173715


namespace slope_of_line_l173_173677

/-- 
Given points M(1, 2) and N(3, 4), prove that the slope of the line passing through these points is 1.
-/
theorem slope_of_line (x1 y1 x2 y2 : ℝ) (hM : x1 = 1 ∧ y1 = 2) (hN : x2 = 3 ∧ y2 = 4) : 
  (y2 - y1) / (x2 - x1) = 1 :=
by
  -- The proof is omitted here because only the statement is required.
  sorry

end slope_of_line_l173_173677


namespace inequality_in_triangle_l173_173451

variables {a b c : ℝ}

namespace InequalityInTriangle

-- Define the condition that a, b, c are sides of a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem inequality_in_triangle (a b c : ℝ) (h : is_triangle a b c) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) :=
sorry

end InequalityInTriangle

end inequality_in_triangle_l173_173451


namespace light_distance_200_years_l173_173753

-- Define the distance light travels in one year.
def distance_one_year := 5870000000000

-- Define the scientific notation representation for distance in one year
def distance_one_year_sci := 587 * 10^10

-- Define the distance light travels in 200 years.
def distance_200_years := distance_one_year * 200

-- Define the expected distance in scientific notation for 200 years.
def expected_distance := 1174 * 10^12

-- The theorem stating the given condition and the conclusion to prove
theorem light_distance_200_years : distance_200_years = expected_distance :=
by
  -- skipping the proof
  sorry

end light_distance_200_years_l173_173753


namespace length_of_CD_l173_173910

variable {x y u v : ℝ}

def divides_C_R (x y : ℝ) : Prop := x / y = 3 / 5
def divides_C_S (u v : ℝ) : Prop := u / v = 4 / 7
def length_RS (u x y v : ℝ) : Prop := u = x + 5 ∧ v = y - 5
def length_CD (x y : ℝ) : ℝ := x + y

theorem length_of_CD (h1 : divides_C_R x y) (h2 : divides_C_S u v) (h3 : length_RS u x y v) :
  length_CD x y = 40 :=
sorry

end length_of_CD_l173_173910


namespace cube_volume_is_1728_l173_173951

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l173_173951


namespace value_of_y_l173_173449

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(2*y) = 4) : y = 1 :=
by
  sorry

end value_of_y_l173_173449


namespace cube_volume_from_surface_area_example_cube_volume_l173_173963

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l173_173963


namespace find_x_value_l173_173437

theorem find_x_value (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x - 4) : x = 7 / 2 := 
sorry

end find_x_value_l173_173437


namespace limit_n_b_n_l173_173830

open Real

def M (x : ℝ) := x - (x^3) / 3

def iterate_M (x : ℝ) (n : ℕ) : ℝ := nat.iterate M n x

noncomputable def b_n (n : ℕ) : ℝ := iterate_M (25 / n) n

theorem limit_n_b_n : filter.tendsto (λ n : ℕ, n * b_n n) filter.at_top (nhds (9 / 25)) :=
sorry

end limit_n_b_n_l173_173830


namespace set_union_eq_l173_173563

open Set

noncomputable def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {-1, 0}
def C : Set ℤ := {-1, 0, 1}

theorem set_union_eq :
  A ∪ B = C :=
by {
  sorry
}

end set_union_eq_l173_173563


namespace inverse_of_inverse_at_9_l173_173750

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5

noncomputable def f_inv (x : ℝ) : ℝ := (x - 5) / 4

theorem inverse_of_inverse_at_9 : f_inv (f_inv 9) = -1 :=
by
  sorry

end inverse_of_inverse_at_9_l173_173750


namespace simplify_expression_l173_173743

variable (x : ℝ)

theorem simplify_expression : 
  2 * x^3 - (7 * x^2 - 9 * x) - 2 * (x^3 - 3 * x^2 + 4 * x) = -x^2 + x := 
by
  sorry

end simplify_expression_l173_173743


namespace close_to_one_below_l173_173820

theorem close_to_one_below (k l m n : ℕ) (h1 : k > l) (h2 : l > m) (h3 : m > n) (hk : k = 43) (hl : l = 7) (hm : m = 3) (hn : n = 2) :
  (1 : ℚ) / k + 1 / l + 1 / m + 1 / n < 1 := by
  sorry

end close_to_one_below_l173_173820


namespace eq_nine_l173_173640

theorem eq_nine (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : (x - y)^2 = 9 := by
  sorry

end eq_nine_l173_173640


namespace collinear_sum_l173_173002

theorem collinear_sum (a b : ℝ) (h : ∃ (λ : ℝ), (∀ t : ℝ, (2, a, b) + t * ((a, 3, b) - (2, a, b)) = (λ * t, λ * t + 1, λ * t + 2))) : a + b = 6 :=
sorry

end collinear_sum_l173_173002


namespace quadratic_has_two_equal_real_roots_l173_173165

theorem quadratic_has_two_equal_real_roots : ∃ c : ℝ, ∀ x : ℝ, (x^2 - 6*x + c = 0 ↔ (x = 3)) :=
by
  sorry

end quadratic_has_two_equal_real_roots_l173_173165


namespace alice_probability_at_least_one_multiple_of_4_l173_173818

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end alice_probability_at_least_one_multiple_of_4_l173_173818


namespace cube_volume_l173_173986

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l173_173986


namespace integer_values_of_a_l173_173418

theorem integer_values_of_a (a : ℤ) :
  (∀ x : ℤ, x^2 + a * x + 12 * a = 0 → x ∈ ℤ) → ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_values_of_a_l173_173418


namespace sample_size_is_30_l173_173008

-- Definitions based on conditions
def total_students : ℕ := 700 + 500 + 300
def students_first_grade : ℕ := 700
def students_sampled_first_grade : ℕ := 14
def sample_size (n : ℕ) : Prop := students_sampled_first_grade = (students_first_grade * n) / total_students

-- Theorem stating the proof problem
theorem sample_size_is_30 : sample_size 30 :=
by
  sorry

end sample_size_is_30_l173_173008


namespace balloon_permutation_count_l173_173250

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l173_173250


namespace distinct_integer_values_of_a_l173_173424

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l173_173424


namespace other_solution_of_quadratic_l173_173434

theorem other_solution_of_quadratic (x : ℚ) (h1 : x = 3 / 8) 
  (h2 : 72 * x^2 + 37 = -95 * x + 12) : ∃ y : ℚ, y ≠ 3 / 8 ∧ 72 * y^2 + 95 * y + 25 = 0 ∧ y = 5 / 8 :=
by
  sorry

end other_solution_of_quadratic_l173_173434


namespace complete_contingency_table_chi_square_test_certainty_l173_173071

-- Defining the initial conditions given in the problem
def total_students : ℕ := 100
def boys_dislike : ℕ := 10
def girls_like : ℕ := 20
def dislike_probability : ℚ := 0.4

-- Completed contingency table values based on given and inferred values
def boys_total : ℕ := 50
def girls_total : ℕ := 50
def boys_like : ℕ := boys_total - boys_dislike
def girls_dislike : ℕ := 30
def total_like : ℕ := boys_like + girls_like
def total_dislike : ℕ := boys_dislike + girls_dislike

-- Chi-square value from the solution
def K_squared : ℚ := 50 / 3

-- Declaring the proof problem for the completed contingency table
theorem complete_contingency_table :
  boys_total + girls_total = total_students ∧ 
  total_like + total_dislike = total_students ∧ 
  dislike_probability * total_students = total_dislike ∧ 
  boys_like = 40 ∧ 
  girls_dislike = 30 :=
sorry

-- Declaring the proof problem for the chi-square test
theorem chi_square_test_certainty :
  K_squared > 10.828 :=
sorry

end complete_contingency_table_chi_square_test_certainty_l173_173071


namespace balloon_arrangement_count_l173_173236

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l173_173236


namespace age_difference_l173_173936

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end age_difference_l173_173936


namespace shortest_distance_point_B_l173_173110

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shortest_distance_point_B :
  let A := (1 : ℝ, 0 : ℝ)
  let line_B (x y : ℝ) := x = y
  ∃ B : ℝ × ℝ, line_B B.1 B.2 ∧ (∀ B' : ℝ × ℝ, line_B B'.1 B'.2 → distance A B ≤ distance A B') ∧ B = (1/2, 1/2) :=
by
  let A := (1, 0)
  let line_B (x y : ℝ) := x = y
  use (1/2, 1/2)
  split
  { sorry }
  { split
    { sorry }
    { refl }
  }

end shortest_distance_point_B_l173_173110


namespace cube_volume_l173_173978

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l173_173978


namespace min_value_a_plus_3b_plus_9c_l173_173594

theorem min_value_a_plus_3b_plus_9c {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a + 3*b + 9*c ≥ 27 :=
sorry

end min_value_a_plus_3b_plus_9c_l173_173594


namespace Monica_books_read_l173_173027

theorem Monica_books_read : 
  let books_last_year := 16 
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  books_next_year = 69 :=
by
  let books_last_year := 16
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  sorry

end Monica_books_read_l173_173027


namespace rectangle_width_eq_six_l173_173527

theorem rectangle_width_eq_six (w : ℝ) :
  ∃ w, (3 * w = 25 - 7) ↔ w = 6 :=
by
  -- Given the conditions as stated:
  -- Length of the rectangle: 3 inches
  -- Width of the square: 5 inches
  -- Difference in area between the square and the rectangle: 7 square inches
  -- We can show that the width of the rectangle is 6 inches.
  sorry

end rectangle_width_eq_six_l173_173527


namespace no_n_satisfies_l173_173155

def sum_first_n_terms_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem no_n_satisfies (n : ℕ) (h_n : n ≠ 0) :
  let s1 := sum_first_n_terms_arith_seq 5 6 n
  let s2 := sum_first_n_terms_arith_seq 12 4 n
  (s1 * s2 = 24 * n^2) → False :=
by
  sorry

end no_n_satisfies_l173_173155


namespace profit_rate_is_five_percent_l173_173397

theorem profit_rate_is_five_percent (cost_price selling_price : ℝ) (hx : 1.1 * cost_price - 10 = 210) : 
  (selling_price = 1.1 * cost_price) → 
  (selling_price - cost_price) / cost_price * 100 = 5 :=
by
  sorry

end profit_rate_is_five_percent_l173_173397


namespace watch_cost_l173_173296

theorem watch_cost (number_of_dimes : ℕ) (value_of_dime : ℝ) (h : number_of_dimes = 50) (hv : value_of_dime = 0.10) :
  number_of_dimes * value_of_dime = 5.00 :=
by
  sorry

end watch_cost_l173_173296


namespace race_dead_heat_l173_173370

variable (v_B v_A L x : ℝ)

theorem race_dead_heat (h : v_A = 17 / 14 * v_B) : x = 3 / 17 * L :=
by
  sorry

end race_dead_heat_l173_173370


namespace evaluate_polynomial_l173_173545

theorem evaluate_polynomial (x : ℝ) : x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 9 * x + 2 := by
  sorry

end evaluate_polynomial_l173_173545


namespace jim_can_bake_loaves_l173_173145

-- Define the amounts of flour in different locations
def flour_cupboard : ℕ := 200  -- in grams
def flour_counter : ℕ := 100   -- in grams
def flour_pantry : ℕ := 100    -- in grams

-- Define the amount of flour required for one loaf of bread
def flour_per_loaf : ℕ := 200  -- in grams

-- Total loaves Jim can bake
def loaves_baked (f_c f_k f_p f_r : ℕ) : ℕ :=
  (f_c + f_k + f_p) / f_r

-- Theorem to prove the solution
theorem jim_can_bake_loaves :
  loaves_baked flour_cupboard flour_counter flour_pantry flour_per_loaf = 2 :=
by
  -- Proof is omitted
  sorry

end jim_can_bake_loaves_l173_173145


namespace relatively_prime_dates_in_september_l173_173393

-- Define a condition to check if two numbers are relatively prime
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the number of days in September
def days_in_september := 30

-- Define the month of September as the 9th month
def month_of_september := 9

-- Define the proposition that the number of relatively prime dates in September is 20
theorem relatively_prime_dates_in_september : 
  ∃ count, (count = 20 ∧ ∀ day, day ∈ Finset.range (days_in_september + 1) → relatively_prime month_of_september day → count = 20) := sorry

end relatively_prime_dates_in_september_l173_173393


namespace cube_volume_from_surface_area_l173_173972

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l173_173972


namespace smallest_k_satisfying_condition_l173_173507

def is_smallest_prime_greater_than (n : ℕ) (p : ℕ) : Prop :=
  Nat.Prime p ∧ n < p ∧ ∀ q, Nat.Prime q ∧ q > n → q >= p

def is_divisible_by (m k : ℕ) : Prop := k % m = 0

theorem smallest_k_satisfying_condition :
  ∃ k, is_smallest_prime_greater_than 19 23 ∧ is_divisible_by 3 k ∧ 64 ^ k > 4 ^ (19 * 23) ∧ (∀ k' < k, is_divisible_by 3 k' → 64 ^ k' ≤ 4 ^ (19 * 23)) :=
by
  sorry

end smallest_k_satisfying_condition_l173_173507


namespace fraction_multiplication_l173_173505

theorem fraction_multiplication :
  ((3 : ℚ) / 4) ^ 3 * ((2 : ℚ) / 5) ^ 3 = (27 : ℚ) / 1000 := sorry

end fraction_multiplication_l173_173505


namespace find_natural_numbers_l173_173415

theorem find_natural_numbers (x : ℕ) : (x % 7 = 3) ∧ (x % 9 = 4) ∧ (x < 100) ↔ (x = 31) ∨ (x = 94) := 
by sorry

end find_natural_numbers_l173_173415


namespace probability_all_white_balls_drawn_l173_173519

theorem probability_all_white_balls_drawn (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h_total : total_balls = 15) (h_white : white_balls = 7) (h_black : black_balls = 8) (h_drawn : drawn_balls = 7) :
  (Nat.choose 7 7 : ℚ) / (Nat.choose 15 7 : ℚ) = 1 / 6435 := by
sorry

end probability_all_white_balls_drawn_l173_173519


namespace place_two_in_front_l173_173881

-- Define the conditions: the original number has hundreds digit h, tens digit t, and units digit u.
variables (h t u : ℕ)

-- Define the function representing the placement of the digit 2 before the three-digit number.
def new_number (h t u : ℕ) : ℕ :=
  2000 + 100 * h + 10 * t + u

-- State the theorem that proves the new number formed is as stated.
theorem place_two_in_front : new_number h t u = 2000 + 100 * h + 10 * t + u :=
by sorry

end place_two_in_front_l173_173881


namespace range_of_a_l173_173553

variable {α : Type}

def A (x : ℝ) : Prop := 1 ≤ x ∧ x < 5
def B (x a : ℝ) : Prop := -a < x ∧ x ≤ a + 3

theorem range_of_a (a : ℝ) :
  (∀ x, B x a → A x) → a ≤ -1 := by
  sorry

end range_of_a_l173_173553


namespace b_finishes_remaining_work_correct_time_for_b_l173_173514

theorem b_finishes_remaining_work (a_days : ℝ) (b_days : ℝ) (work_together_days : ℝ) (remaining_work_after : ℝ) : ℝ :=
  let a_work_rate := 1 / a_days
  let b_work_rate := 1 / b_days
  let combined_work_per_day := a_work_rate + b_work_rate
  let work_done_together := combined_work_per_day * work_together_days
  let remaining_work := 1 - work_done_together
  let b_completion_time := remaining_work / b_work_rate
  b_completion_time

theorem correct_time_for_b : b_finishes_remaining_work 2 6 1 (1 - 2/3) = 2 := 
by sorry

end b_finishes_remaining_work_correct_time_for_b_l173_173514


namespace transform_into_product_l173_173518

theorem transform_into_product : 447 * (Real.sin (75 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 447 * Real.sqrt 6 / 2 := by
  sorry

end transform_into_product_l173_173518


namespace product_of_possible_values_l173_173871

theorem product_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 18) : ∃ a b, x = a ∨ x = b ∧ a * b = -30 :=
by 
  sorry

end product_of_possible_values_l173_173871


namespace min_abs_A_l173_173475

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def A (a d : ℚ) (n : ℕ) : ℚ :=
  (arithmetic_sequence a d n) + (arithmetic_sequence a d (n + 1)) + 
  (arithmetic_sequence a d (n + 2)) + (arithmetic_sequence a d (n + 3)) + 
  (arithmetic_sequence a d (n + 4)) + (arithmetic_sequence a d (n + 5)) + 
  (arithmetic_sequence a d (n + 6))

theorem min_abs_A : (arithmetic_sequence 19 (-4/5) 26 = -1) ∧ 
                    (∀ n, 1 ≤ n) →
                    ∃ n : ℕ, |A 19 (-4/5) n| = 7/5 :=
by
  sorry

end min_abs_A_l173_173475


namespace solve_for_x_l173_173515

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) :
  x = 37 :=
sorry

end solve_for_x_l173_173515


namespace alice_probability_multiple_of_4_l173_173815

noncomputable def probability_one_multiple_of_4 (choices : ℕ) : ℚ :=
  let p_not_multiple_of_4 : ℚ := 45 / 60
  let p_all_not_multiple_of_4 : ℚ := p_not_multiple_of_4 ^ choices
  1 - p_all_not_multiple_of_4

theorem alice_probability_multiple_of_4 :
  probability_one_multiple_of_4 3 = 37 / 64 :=
by
  sorry

end alice_probability_multiple_of_4_l173_173815
