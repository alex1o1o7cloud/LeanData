import Mathlib

namespace triangle_area_l1073_107359

/-- 
  Given:
  - A smaller rectangle OABD with OA = 4 cm, AB = 4 cm
  - A larger rectangle ABEC with AB = 12 cm, BC = 12 cm
  - Point O at (0,0)
  - Point A at (4,0)
  - Point B at (16,0)
  - Point C at (16,12)
  - Point D at (4,12)
  - Point E is on the line from A to C
  
  Prove the area of the triangle CDE is 54 cm²
-/
theorem triangle_area (OA AB OB DE DC : ℕ) : 
  OA = 4 ∧ AB = 4 ∧ OB = 16 ∧ DE = 12 - 3 ∧ DC = 12 → (1 / 2) * DE * DC = 54 := by 
  intros h
  sorry

end triangle_area_l1073_107359


namespace two_digit_number_satisfying_conditions_l1073_107321

theorem two_digit_number_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 8 ∧
  ∀ p ∈ s, ∃ (a b : ℕ), p = (a, b) ∧
    (10 * a + b < 100) ∧
    (a ≥ 2) ∧
    (10 * a + b + 10 * b + a = 110) :=
by
  sorry

end two_digit_number_satisfying_conditions_l1073_107321


namespace greater_number_is_33_l1073_107311

theorem greater_number_is_33 (A B : ℕ) (hcf_11 : Nat.gcd A B = 11) (product_363 : A * B = 363) :
  max A B = 33 :=
by
  sorry

end greater_number_is_33_l1073_107311


namespace jason_spent_at_music_store_l1073_107398

theorem jason_spent_at_music_store 
  (cost_flute : ℝ) (cost_music_tool : ℝ) (cost_song_book : ℝ)
  (h1 : cost_flute = 142.46)
  (h2 : cost_music_tool = 8.89)
  (h3 : cost_song_book = 7) :
  cost_flute + cost_music_tool + cost_song_book = 158.35 :=
by
  -- assumption proof
  sorry

end jason_spent_at_music_store_l1073_107398


namespace problem1_problem2_l1073_107334

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Statement for the first proof
theorem problem1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  bc / a + ca / b + ab / c ≥ a + b + c :=
sorry

-- Statement for the second proof
theorem problem2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6 :=
sorry

end problem1_problem2_l1073_107334


namespace wire_ratio_is_one_l1073_107397

theorem wire_ratio_is_one (a b : ℝ) (h1 : a = b) : a / b = 1 := by
  -- The proof goes here
  sorry

end wire_ratio_is_one_l1073_107397


namespace find_a_b_l1073_107382

def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b :
  ∀ (a b : ℝ),
  (∀ x, (curve x a b) = x^2 + a * x + b) →
  (tangent_line 0 (curve 0 a b)) →
  (tangent_line x y → y = x + 1) →
  (tangent_line x y → ∃ m c, y = m * x + c ∧ m = 1 ∧ c = 1) →
  (∃ a b : ℝ, a = 1 ∧ b = 1) :=
by
  intros a b h_curve h_tangent_line h_tangent_line_form h_tangent_line_eq
  sorry

end find_a_b_l1073_107382


namespace winning_post_distance_l1073_107384

theorem winning_post_distance (v x : ℝ) (h₁ : x ≠ 0) (h₂ : v ≠ 0)
  (h₃ : 1.75 * v = v) 
  (h₄ : x = 1.75 * (x - 84)) : 
  x = 196 :=
by 
  sorry

end winning_post_distance_l1073_107384


namespace initial_holes_count_additional_holes_needed_l1073_107335

-- Defining the conditions as variables
def circumference : ℕ := 400
def initial_interval : ℕ := 50
def new_interval : ℕ := 40

-- Defining the problems

-- Problem 1: Calculate the number of holes for the initial interval
theorem initial_holes_count (circumference : ℕ) (initial_interval : ℕ) : 
  circumference % initial_interval = 0 → 
  circumference / initial_interval = 8 := 
sorry

-- Problem 2: Calculate the additional holes needed
theorem additional_holes_needed (circumference : ℕ) (initial_interval : ℕ) 
  (new_interval : ℕ) (lcm_interval : ℕ) :
  lcm new_interval initial_interval = lcm_interval →
  circumference % new_interval = 0 →
  circumference / new_interval - 
  (circumference / lcm_interval) = 8 :=
sorry

end initial_holes_count_additional_holes_needed_l1073_107335


namespace photograph_area_l1073_107305

def dimensions_are_valid (a b : ℕ) : Prop :=
a > 0 ∧ b > 0 ∧ (a + 4) * (b + 5) = 77

theorem photograph_area (a b : ℕ) (h : dimensions_are_valid a b) : (a * b = 18 ∨ a * b = 14) :=
by 
  sorry

end photograph_area_l1073_107305


namespace max_difference_of_mean_505_l1073_107338

theorem max_difference_of_mean_505 (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : (x + y) / 2 = 505) : 
  x - y ≤ 810 :=
sorry

end max_difference_of_mean_505_l1073_107338


namespace inequality_proof_l1073_107316

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y * z) / (Real.sqrt (2 * x^2 * (y + z))) + 
  (y^2 + z * x) / (Real.sqrt (2 * y^2 * (z + x))) + 
  (z^2 + x * y) / (Real.sqrt (2 * z^2 * (x + y))) ≥ 1 := 
sorry

end inequality_proof_l1073_107316


namespace largest_integral_x_l1073_107389

theorem largest_integral_x (x y : ℤ) (h1 : (1 : ℚ)/4 < x/7) (h2 : x/7 < (2 : ℚ)/3) (h3 : x + y = 10) : x = 4 :=
by
  sorry

end largest_integral_x_l1073_107389


namespace maximum_distance_point_to_line_l1073_107300

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m - 1) * x + m * y + 2 = 0

-- Statement of the problem to prove
theorem maximum_distance_point_to_line :
  ∀ (x y m : ℝ), circle_C x y → ∃ P : ℝ, line_l m x y → P = 6 :=
by 
  sorry

end maximum_distance_point_to_line_l1073_107300


namespace initial_legos_l1073_107329

-- Definitions and conditions
def legos_won : ℝ := 17.0
def legos_now : ℝ := 2097.0

-- The statement to prove
theorem initial_legos : (legos_now - legos_won) = 2080 :=
by sorry

end initial_legos_l1073_107329


namespace julia_song_download_l1073_107355

theorem julia_song_download : 
  let internet_speed := 20 -- in MBps
  let half_hour_in_minutes := 30
  let size_per_song := 5 -- in MB
  (internet_speed * 60 * half_hour_in_minutes) / size_per_song = 7200 :=
by
  sorry

end julia_song_download_l1073_107355


namespace winning_votes_l1073_107381

theorem winning_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 312) : 0.62 * V = 806 :=
by
  -- The proof should be written here, but we'll skip it as per the instructions.
  sorry

end winning_votes_l1073_107381


namespace males_only_in_band_l1073_107326

theorem males_only_in_band
  (females_in_band : ℕ)
  (males_in_band : ℕ)
  (females_in_orchestra : ℕ)
  (males_in_orchestra : ℕ)
  (females_in_both : ℕ)
  (total_students : ℕ)
  (total_students_in_either : ℕ)
  (hf_in_band : females_in_band = 120)
  (hm_in_band : males_in_band = 90)
  (hf_in_orchestra : females_in_orchestra = 100)
  (hm_in_orchestra : males_in_orchestra = 130)
  (hf_in_both : females_in_both = 80)
  (h_total_students : total_students = 260) :
  total_students_in_either = 260 → 
  (males_in_band - (90 + 130 + 80 - 260 - 120)) = 30 :=
by
  intros h_total_students_in_either
  sorry

end males_only_in_band_l1073_107326


namespace tiffany_total_bags_l1073_107324

-- Define the initial and additional bags correctly
def bags_on_monday : ℕ := 10
def bags_next_day : ℕ := 3
def bags_day_after : ℕ := 7

-- Define the total bags calculation
def total_bags (initial : ℕ) (next : ℕ) (after : ℕ) : ℕ :=
  initial + next + after

-- Prove that the total bags collected is 20
theorem tiffany_total_bags : total_bags bags_on_monday bags_next_day bags_day_after = 20 :=
by
  sorry

end tiffany_total_bags_l1073_107324


namespace yoongi_has_smallest_points_l1073_107383

def points_jungkook : ℕ := 6 + 3
def points_yoongi : ℕ := 4
def points_yuna : ℕ := 5

theorem yoongi_has_smallest_points : points_yoongi < points_jungkook ∧ points_yoongi < points_yuna :=
by
  sorry

end yoongi_has_smallest_points_l1073_107383


namespace find_x_from_percentage_l1073_107388

theorem find_x_from_percentage (x : ℝ) (h : 0.2 * 30 = 0.25 * x + 2) : x = 16 :=
sorry

end find_x_from_percentage_l1073_107388


namespace average_sale_six_months_l1073_107348

theorem average_sale_six_months :
  let sale1 := 2500
  let sale2 := 6500
  let sale3 := 9855
  let sale4 := 7230
  let sale5 := 7000
  let sale6 := 11915
  let total_sales := sale1 + sale2 + sale3 + sale4 + sale5 + sale6
  let num_months := 6
  (total_sales / num_months) = 7500 :=
by
  sorry

end average_sale_six_months_l1073_107348


namespace bubble_bath_per_guest_l1073_107375

def rooms_couple : ℕ := 13
def rooms_single : ℕ := 14
def total_bubble_bath : ℕ := 400

theorem bubble_bath_per_guest :
  (total_bubble_bath / (rooms_couple * 2 + rooms_single)) = 10 :=
by
  sorry

end bubble_bath_per_guest_l1073_107375


namespace namjoonKoreanScore_l1073_107345

variables (mathScore englishScore : ℝ) (averageScore : ℝ := 95) (koreanScore : ℝ)

def namjoonMathScore : Prop := mathScore = 100
def namjoonEnglishScore : Prop := englishScore = 95
def namjoonAverage : Prop := (koreanScore + mathScore + englishScore) / 3 = averageScore

theorem namjoonKoreanScore
  (H1 : namjoonMathScore 100)
  (H2 : namjoonEnglishScore 95)
  (H3 : namjoonAverage koreanScore 100 95 95) :
  koreanScore = 90 :=
by
  sorry

end namjoonKoreanScore_l1073_107345


namespace lines_intersection_example_l1073_107340

theorem lines_intersection_example (m b : ℝ) 
  (h1 : 8 = m * 4 + 2) 
  (h2 : 8 = 4 * 4 + b) : 
  b + m = -13 / 2 := 
by
  sorry

end lines_intersection_example_l1073_107340


namespace original_stickers_l1073_107307

theorem original_stickers (x : ℕ) (h₁ : x * 3 / 4 * 4 / 5 = 45) : x = 75 :=
by
  sorry

end original_stickers_l1073_107307


namespace total_distance_thrown_l1073_107350

theorem total_distance_thrown (D : ℝ) (total_distance : ℝ) 
  (h1 : total_distance = 20 * D + 60 * D) : 
  total_distance = 1600 := 
by
  sorry

end total_distance_thrown_l1073_107350


namespace garden_length_l1073_107379

theorem garden_length (P b l : ℕ) (h1 : P = 500) (h2 : b = 100) : l = 150 :=
by
  sorry

end garden_length_l1073_107379


namespace num_disks_to_sell_l1073_107346

-- Define the buying and selling price conditions.
def cost_per_disk := 6 / 5
def sell_per_disk := 7 / 4

-- Define the desired profit
def desired_profit := 120

-- Calculate the profit per disk.
def profit_per_disk := sell_per_disk - cost_per_disk

-- Statement of the problem: Determine number of disks to sell.
theorem num_disks_to_sell
  (h₁ : cost_per_disk = 6 / 5)
  (h₂ : sell_per_disk = 7 / 4)
  (h₃ : desired_profit = 120)
  (h₄ : profit_per_disk = 7 / 4 - 6 / 5) :
  ∃ disks_to_sell : ℕ, disks_to_sell = 219 ∧ 
  disks_to_sell * profit_per_disk ≥ 120 ∧
  (disks_to_sell - 1) * profit_per_disk < 120 :=
sorry

end num_disks_to_sell_l1073_107346


namespace find_line_l_l1073_107318

def line_equation (x y: ℤ) : Prop := x - 2 * y = 2

def scaling_transform_x (x: ℤ) : ℤ := x
def scaling_transform_y (y: ℤ) : ℤ := 2 * y

theorem find_line_l :
  ∀ (x y x' y': ℤ),
  x' = scaling_transform_x x →
  y' = scaling_transform_y y →
  line_equation x y →
  x' - y' = 2 := by
  sorry

end find_line_l_l1073_107318


namespace graph_of_equation_pair_of_lines_l1073_107354

theorem graph_of_equation_pair_of_lines (x y : ℝ) : x^2 - 9 * y^2 = 0 ↔ (x = 3 * y ∨ x = -3 * y) :=
by
  sorry

end graph_of_equation_pair_of_lines_l1073_107354


namespace arith_seq_ninth_term_value_l1073_107322

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end arith_seq_ninth_term_value_l1073_107322


namespace matchsticks_distribution_l1073_107349

open Nat

theorem matchsticks_distribution
  (length_sticks : ℕ)
  (width_sticks : ℕ)
  (length_condition : length_sticks = 60)
  (width_condition : width_sticks = 10)
  (total_sticks : ℕ)
  (total_sticks_condition : total_sticks = 60 * 11 + 10 * 61)
  (children_count : ℕ)
  (children_condition : children_count > 100)
  (division_condition : total_sticks % children_count = 0) :
  children_count = 127 := by
  sorry

end matchsticks_distribution_l1073_107349


namespace solve_for_z_l1073_107380

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2 + I) : z = (1 / 2) + (3 / 2) * I :=
  sorry

end solve_for_z_l1073_107380


namespace find_m_l1073_107391

-- Define the sets A and B and the conditions
def A : Set ℝ := {x | x ≥ 3}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Define the conditions on these sets
def conditions (m : ℝ) : Prop :=
  (∀ x, x ∈ A ∨ x ∈ B m) ∧ (∀ x, ¬(x ∈ A ∧ x ∈ B m))

-- State the theorem
theorem find_m : ∃ m : ℝ, conditions m ∧ m = 3 :=
  sorry

end find_m_l1073_107391


namespace add_one_five_times_l1073_107386

theorem add_one_five_times (m n : ℕ) (h : n = m + 5) : n - (m + 1) = 4 :=
by
  sorry

end add_one_five_times_l1073_107386


namespace root_of_quadratic_l1073_107351

theorem root_of_quadratic (a : ℝ) (ha : a ≠ 1) (hroot : (a-1) * 1^2 - a * 1 + a^2 = 0) : a = -1 := by
  sorry

end root_of_quadratic_l1073_107351


namespace cyclic_sum_inequality_l1073_107337

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) ≥ (2 / 3) * (a^2 + b^2 + c^2) :=
  sorry

end cyclic_sum_inequality_l1073_107337


namespace smallest_palindrome_div_3_5_l1073_107385

theorem smallest_palindrome_div_3_5 : ∃ n : ℕ, n = 50205 ∧ 
  (∃ a b c : ℕ, n = 5 * 10^4 + a * 10^3 + b * 10^2 + a * 10 + 5) ∧ 
  n % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 10000 ∧ 
  n < 100000 :=
by
  sorry

end smallest_palindrome_div_3_5_l1073_107385


namespace price_reduction_equation_l1073_107368

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end price_reduction_equation_l1073_107368


namespace mail_handling_in_six_months_l1073_107363

theorem mail_handling_in_six_months (daily_letters daily_packages days_per_month months : ℕ) :
  daily_letters = 60 →
  daily_packages = 20 →
  days_per_month = 30 →
  months = 6 →
  (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  -- Skipping the proof
  sorry

end mail_handling_in_six_months_l1073_107363


namespace area_square_B_l1073_107387

theorem area_square_B (a b : ℝ) (h1 : a^2 = 25) (h2 : abs (a - b) = 4) : b^2 = 81 :=
by
  sorry

end area_square_B_l1073_107387


namespace find_b_value_l1073_107308

noncomputable def find_b (p q : ℕ) : ℕ := p^2 + q^2

theorem find_b_value
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h_distinct : p ≠ q) (h_roots : p + q = 13 ∧ p * q = 22) :
  find_b p q = 125 :=
by
  sorry

end find_b_value_l1073_107308


namespace countNegativeValues_l1073_107304

-- Define the condition that sqrt(x + 122) is a positive integer
noncomputable def isPositiveInteger (n : ℤ) (x : ℤ) : Prop :=
  ∃ n : ℤ, (n > 0) ∧ (x + 122 = n * n)

-- Define the condition that x is negative
def isNegative (x : ℤ) : Prop :=
  x < 0

-- Prove the number of different negative values of x such that sqrt(x + 122) is a positive integer is 11
theorem countNegativeValues :
  ∃ x_set : Finset ℤ, (∀ x ∈ x_set, isNegative x ∧ isPositiveInteger x (x + 122)) ∧ x_set.card = 11 :=
sorry

end countNegativeValues_l1073_107304


namespace min_notebooks_needed_l1073_107333

variable (cost_pen cost_notebook num_pens discount_threshold : ℕ)

theorem min_notebooks_needed (x : ℕ)
    (h1 : cost_pen = 10)
    (h2 : cost_notebook = 4)
    (h3 : num_pens = 3)
    (h4 : discount_threshold = 100)
    (h5 : num_pens * cost_pen + x * cost_notebook ≥ discount_threshold) :
    x ≥ 18 := 
sorry

end min_notebooks_needed_l1073_107333


namespace cos_of_tan_l1073_107327

/-- Given a triangle ABC with angle A such that tan(A) = -5/12, prove cos(A) = -12/13. -/
theorem cos_of_tan (A : ℝ) (h : Real.tan A = -5 / 12) : Real.cos A = -12 / 13 := by
  sorry

end cos_of_tan_l1073_107327


namespace correct_equation_l1073_107314

noncomputable def team_a_initial := 96
noncomputable def team_b_initial := 72
noncomputable def team_b_final (x : ℕ) := team_b_initial - x
noncomputable def team_a_final (x : ℕ) := team_a_initial + x

theorem correct_equation (x : ℕ) : 
  (1 / 3 : ℚ) * (team_a_final x) = (team_b_final x) := 
  sorry

end correct_equation_l1073_107314


namespace find_additional_payment_l1073_107313

-- Definitions used from the conditions
def total_payments : ℕ := 52
def first_partial_payments : ℕ := 25
def second_partial_payments : ℕ := total_payments - first_partial_payments
def first_payment_amount : ℝ := 500
def average_payment : ℝ := 551.9230769230769

-- Condition in Lean
theorem find_additional_payment :
  let total_amount := average_payment * total_payments
  let first_payment_total := first_partial_payments * first_payment_amount
  ∃ x : ℝ, total_amount = first_payment_total + second_partial_payments * (first_payment_amount + x) → x = 100 :=
by
  sorry

end find_additional_payment_l1073_107313


namespace root_poly_ratio_c_d_l1073_107323

theorem root_poly_ratio_c_d (a b c d : ℝ)
  (h₁ : 1 + (-2) + 3 = 2)
  (h₂ : 1 * (-2) + (-2) * 3 + 3 * 1 = -5)
  (h₃ : 1 * (-2) * 3 = -6)
  (h_sum : -b / a = 2)
  (h_pair_prod : c / a = -5)
  (h_prod : -d / a = -6) :
  c / d = 5 / 6 := by
  sorry

end root_poly_ratio_c_d_l1073_107323


namespace children_got_on_the_bus_l1073_107309

-- Definitions
def original_children : ℕ := 26
def current_children : ℕ := 64

-- Theorem stating the problem
theorem children_got_on_the_bus : (current_children - original_children = 38) :=
by {
  sorry
}

end children_got_on_the_bus_l1073_107309


namespace triangle_ABCD_lengths_l1073_107336

theorem triangle_ABCD_lengths (AB BC CA : ℝ) (h_AB : AB = 20) (h_BC : BC = 40) (h_CA : CA = 49) :
  ∃ DA DC : ℝ, DA = 27.88 ∧ DC = 47.88 ∧
  (AB + DC = BC + DA) ∧ 
  (((AB^2 + BC^2 - CA^2) / (2 * AB * BC)) + ((DC^2 + DA^2 - CA^2) / (2 * DC * DA)) = 0) :=
sorry

end triangle_ABCD_lengths_l1073_107336


namespace number_of_possible_heights_is_680_l1073_107303

noncomputable def total_possible_heights : Nat :=
  let base_height := 200 * 3
  let max_additional_height := 200 * (20 - 3)
  let min_height := base_height
  let max_height := base_height + max_additional_height
  let number_of_possible_heights := (max_height - min_height) / 5 + 1
  number_of_possible_heights

theorem number_of_possible_heights_is_680 : total_possible_heights = 680 := by
  sorry

end number_of_possible_heights_is_680_l1073_107303


namespace find_breadth_of_cuboid_l1073_107366

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end find_breadth_of_cuboid_l1073_107366


namespace pencils_calculation_l1073_107332

def num_pencil_boxes : ℝ := 4.0
def pencils_per_box : ℝ := 648.0
def total_pencils : ℝ := 2592.0

theorem pencils_calculation : (num_pencil_boxes * pencils_per_box) = total_pencils := 
by
  sorry

end pencils_calculation_l1073_107332


namespace michael_twenty_dollar_bills_l1073_107347

/--
Michael has $280 dollars and each bill is $20 dollars.
We need to prove that the number of $20 dollar bills Michael has is 14.
-/
theorem michael_twenty_dollar_bills (total_money : ℕ) (bill_denomination : ℕ) (number_of_bills : ℕ) :
  total_money = 280 →
  bill_denomination = 20 →
  number_of_bills = total_money / bill_denomination →
  number_of_bills = 14 :=
by
  intros h1 h2 h3
  sorry

end michael_twenty_dollar_bills_l1073_107347


namespace time_for_C_alone_to_finish_the_job_l1073_107301

variable {A B C : ℝ} -- Declare work rates as real numbers

-- Define the conditions
axiom h1 : A + B = 1/15
axiom h2 : A + B + C = 1/10

-- Define the theorem to prove
theorem time_for_C_alone_to_finish_the_job : C = 1/30 :=
by
  apply sorry

end time_for_C_alone_to_finish_the_job_l1073_107301


namespace pump_X_time_l1073_107374

-- Definitions for the problem conditions.
variables (W : ℝ) (T_x : ℝ) (R_x R_y : ℝ)

-- Condition 1: Rate of pump X
def pump_X_rate := R_x = (W / 2) / T_x

-- Condition 2: Rate of pump Y
def pump_Y_rate := R_y = W / 18

-- Condition 3: Combined rate when both pumps work together for 3 hours to pump the remaining water
def combined_rate := (R_x + R_y) = (W / 2) / 3

-- The statement to prove
theorem pump_X_time : 
  pump_X_rate W T_x R_x →
  pump_Y_rate W R_y →
  combined_rate W R_x R_y →
  T_x = 9 :=
sorry

end pump_X_time_l1073_107374


namespace geese_more_than_ducks_l1073_107376

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end geese_more_than_ducks_l1073_107376


namespace total_earnings_l1073_107370

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end total_earnings_l1073_107370


namespace solve_price_per_litre_second_oil_l1073_107393

variable (P : ℝ)

def price_per_litre_second_oil :=
  10 * 55 + 5 * P = 15 * 58.67

theorem solve_price_per_litre_second_oil (h : price_per_litre_second_oil P) : P = 66.01 :=
  by
  sorry

end solve_price_per_litre_second_oil_l1073_107393


namespace pinwheel_area_eq_six_l1073_107312

open Set

/-- Define the pinwheel in a 6x6 grid -/
def is_midpoint (x y : ℤ) : Prop :=
  (x = 3 ∧ (y = 1 ∨ y = 5)) ∨ (y = 3 ∧ (x = 1 ∨ x = 5))

def is_center (x y : ℤ) : Prop :=
  x = 3 ∧ y = 3

def is_triangle_vertex (x y : ℤ) : Prop :=
  is_center x y ∨ is_midpoint x y

-- Main theorem statement
theorem pinwheel_area_eq_six :
  let pinwheel : Set (ℤ × ℤ) := {p | is_triangle_vertex p.1 p.2}
  ∀ A : ℝ, A = 6 :=
by sorry

end pinwheel_area_eq_six_l1073_107312


namespace product_of_three_numbers_l1073_107372

theorem product_of_three_numbers : 
  ∃ x y z : ℚ, x + y + z = 30 ∧ x = 3 * (y + z) ∧ y = 6 * z ∧ x * y * z = 23625 / 686 :=
by
  sorry

end product_of_three_numbers_l1073_107372


namespace inequality_proof_l1073_107341

variable (f : ℕ → ℕ → ℕ)

theorem inequality_proof :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
by sorry

end inequality_proof_l1073_107341


namespace parker_total_stamps_l1073_107361

-- Definitions based on conditions
def original_stamps := 430
def addie_stamps := 1890
def addie_fraction := 3 / 7
def stamps_added_by_addie := addie_fraction * addie_stamps

-- Theorem statement to prove the final number of stamps
theorem parker_total_stamps : original_stamps + stamps_added_by_addie = 1240 :=
by
  -- definitions instantiated above
  sorry  -- proof required

end parker_total_stamps_l1073_107361


namespace arithmetic_sequence_sum_l1073_107320

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (n : ℕ)
  (a1 : ℤ)
  (d : ℤ)
  (h1 : a1 = 2)
  (h2 : a_n 5 = a_n 1 + 4 * d)
  (h3 : a_n 3 = a_n 1 + 2 * d)
  (h4 : a_n 5 = 3 * a_n 3) :
  S_n 9 = -54 := 
by  
  sorry

end arithmetic_sequence_sum_l1073_107320


namespace range_of_m_l1073_107328

variables {m x : ℝ}

def p (m : ℝ) : Prop := (16 * (m - 2)^2 - 16 > 0) ∧ (m - 2 < 0)
def q (m : ℝ) : Prop := (9 * m^2 - 4 < 0)
def pq (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(q m)

theorem range_of_m (h : pq m) : m ≤ -2/3 ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end range_of_m_l1073_107328


namespace trapezoid_median_properties_l1073_107319

-- Define the variables
variables (a b x : ℝ)

-- State the conditions and the theorem
theorem trapezoid_median_properties (h1 : x = (2 * a) / 3) (h2 : x = b + 3) (h3 : x = (a + b) / 2) : x = 6 :=
by
  sorry

end trapezoid_median_properties_l1073_107319


namespace minimum_value_fraction_l1073_107377

theorem minimum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (2 / x + 1 / y) >= 2 * Real.sqrt 2 :=
sorry

end minimum_value_fraction_l1073_107377


namespace remainder_369963_div_6_is_3_l1073_107392

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def remainder_when_divided (a b : ℕ) (r : ℕ) : Prop := a % b = r

theorem remainder_369963_div_6_is_3 :
  remainder_when_divided 369963 6 3 :=
by
  have h₁ : 369963 % 2 = 1 := by
    sorry -- It is known that 369963 is not divisible by 2.
  have h₂ : 369963 % 3 = 0 := by
    sorry -- It is known that 369963 is divisible by 3.
  have h₃ : 369963 % 6 = 3 := by
    sorry -- From the above properties, derive that the remainder when 369963 is divided by 6 is 3.
  exact h₃

end remainder_369963_div_6_is_3_l1073_107392


namespace base_length_of_isosceles_triangle_l1073_107378

theorem base_length_of_isosceles_triangle (a b : ℕ) 
    (h₁ : a = 8) 
    (h₂ : 2 * a + b = 25) : 
    b = 9 :=
by
  -- This is the proof stub. Proof will be provided here.
  sorry

end base_length_of_isosceles_triangle_l1073_107378


namespace averages_correct_l1073_107310

variables (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
           marksChemistry totalChemistry marksBiology totalBiology 
           marksHistory totalHistory marksGeography totalGeography : ℕ)

variables (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ)

def Kamal_average_english : Prop :=
  marksEnglish = 76 ∧ totalEnglish = 120 ∧ avgEnglish = (marksEnglish / totalEnglish) * 100

def Kamal_average_math : Prop :=
  marksMath = 65 ∧ totalMath = 150 ∧ avgMath = (marksMath / totalMath) * 100

def Kamal_average_physics : Prop :=
  marksPhysics = 82 ∧ totalPhysics = 100 ∧ avgPhysics = (marksPhysics / totalPhysics) * 100

def Kamal_average_chemistry : Prop :=
  marksChemistry = 67 ∧ totalChemistry = 80 ∧ avgChemistry = (marksChemistry / totalChemistry) * 100

def Kamal_average_biology : Prop :=
  marksBiology = 85 ∧ totalBiology = 100 ∧ avgBiology = (marksBiology / totalBiology) * 100

def Kamal_average_history : Prop :=
  marksHistory = 92 ∧ totalHistory = 150 ∧ avgHistory = (marksHistory / totalHistory) * 100

def Kamal_average_geography : Prop :=
  marksGeography = 58 ∧ totalGeography = 75 ∧ avgGeography = (marksGeography / totalGeography) * 100

theorem averages_correct :
  ∀ (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
      marksChemistry totalChemistry marksBiology totalBiology 
      marksHistory totalHistory marksGeography totalGeography : ℕ),
  ∀ (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ),
  Kamal_average_english marksEnglish totalEnglish avgEnglish →
  Kamal_average_math marksMath totalMath avgMath →
  Kamal_average_physics marksPhysics totalPhysics avgPhysics →
  Kamal_average_chemistry marksChemistry totalChemistry avgChemistry →
  Kamal_average_biology marksBiology totalBiology avgBiology →
  Kamal_average_history marksHistory totalHistory avgHistory →
  Kamal_average_geography marksGeography totalGeography avgGeography →
  avgEnglish = 63.33 ∧ avgMath = 43.33 ∧ avgPhysics = 82 ∧
  avgChemistry = 83.75 ∧ avgBiology = 85 ∧ avgHistory = 61.33 ∧ avgGeography = 77.33 :=
by
  sorry

end averages_correct_l1073_107310


namespace f_value_at_5_l1073_107364

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 / 2 then 2 * x^2 else sorry

theorem f_value_at_5 (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 3)
  (h_definition : ∀ x, 0 ≤ x ∧ x ≤ 3 / 2 → f x = 2 * x^2) :
  f 5 = 2 :=
by
  sorry

end f_value_at_5_l1073_107364


namespace min_pairs_opponents_statement_l1073_107317

-- Problem statement definitions
variables (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2)

-- Required minimum number of pairs of opponents in a parliament
def min_pairs_opponents (h p : ℕ) : ℕ :=
  min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)

-- Proof statement
theorem min_pairs_opponents_statement (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2) :
  ∀ (hp : ℕ), ∃ (pairs : ℕ), 
    pairs = min_pairs_opponents h p :=
  sorry

end min_pairs_opponents_statement_l1073_107317


namespace triangle_area_PQR_l1073_107367

section TriangleArea

variables {a b c d : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
variables (hOppositeSides : (0 - c) * b - (a - 0) * d < 0)

theorem triangle_area_PQR :
  let P := (0, a)
  let Q := (b, 0)
  let R := (c, d)
  let area := (1 / 2) * (a * c + b * d - a * b)
  area = (1 / 2) * (a * c + b * d - a * b) := 
by
  sorry

end TriangleArea

end triangle_area_PQR_l1073_107367


namespace range_of_a_l1073_107371

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 :=
by
  sorry

end range_of_a_l1073_107371


namespace margie_driving_distance_l1073_107365

-- Define the constants given in the conditions
def mileage_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def total_money : ℝ := 25

-- Define the expected result/answer
def expected_miles : ℝ := 200

-- The theorem that needs to be proved
theorem margie_driving_distance :
  (total_money / cost_per_gallon) * mileage_per_gallon = expected_miles :=
by
  -- proof goes here
  sorry

end margie_driving_distance_l1073_107365


namespace find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l1073_107353

-- Define the arithmetic sequence
def a (n : ℕ) (d : ℤ) := 23 + n * d

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) (d : ℤ) := n * 23 + (n * (n - 1) / 2) * d

-- Prove the common difference is -4
theorem find_common_difference (d : ℤ) :
  a 5 d > 0 ∧ a 6 d < 0 → d = -4 := sorry

-- Prove the maximum value of the sum S_n of the first n terms
theorem max_sum_first_n_terms (S_n : ℕ) :
  S 6 -4 = 78 := sorry

-- Prove the maximum value of n such that S_n > 0
theorem max_n_Sn_positive (n : ℕ) :
  S n -4 > 0 → n ≤ 12 := sorry

end find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l1073_107353


namespace compute_expression_l1073_107395

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end compute_expression_l1073_107395


namespace total_cars_l1073_107343

-- Definitions of the conditions
def cathy_cars : Nat := 5

def carol_cars : Nat := 2 * cathy_cars

def susan_cars : Nat := carol_cars - 2

def lindsey_cars : Nat := cathy_cars + 4

-- The theorem statement (problem)
theorem total_cars : cathy_cars + carol_cars + susan_cars + lindsey_cars = 32 :=
by
  -- sorry is added to skip the proof
  sorry

end total_cars_l1073_107343


namespace contrapositive_proof_l1073_107306

theorem contrapositive_proof (x m : ℝ) :
  (m < 0 → (∃ r : ℝ, r * r + 3 * r + m = 0)) ↔
  (¬ (∃ r : ℝ, r * r + 3 * r + m = 0) → m ≥ 0) :=
by
  sorry

end contrapositive_proof_l1073_107306


namespace rose_needs_more_money_l1073_107339

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l1073_107339


namespace gcd_3060_561_l1073_107396

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 :=
by
  sorry

end gcd_3060_561_l1073_107396


namespace screws_per_pile_l1073_107357

-- Definitions based on the given conditions
def initial_screws : ℕ := 8
def multiplier : ℕ := 2
def sections : ℕ := 4

-- Derived values based on the conditions
def additional_screws : ℕ := initial_screws * multiplier
def total_screws : ℕ := initial_screws + additional_screws

-- Proposition statement
theorem screws_per_pile : total_screws / sections = 6 := by
  sorry

end screws_per_pile_l1073_107357


namespace total_questions_l1073_107394

theorem total_questions (f s k : ℕ) (hf : f = 36) (hs : s = 2 * f) (hk : k = (f + s) / 2) :
  2 * (f + s + k) = 324 :=
by {
  sorry
}

end total_questions_l1073_107394


namespace slope_symmetric_line_l1073_107331

  theorem slope_symmetric_line {l1 l2 : ℝ → ℝ} 
     (hl1 : ∀ x, l1 x = 2 * x + 3)
     (hl2_sym : ∀ x, l2 x = 2 * x + 3 -> l2 (-x) = -2 * x - 3) :
     ∀ x, l2 x = -2 * x + 3 :=
  sorry
  
end slope_symmetric_line_l1073_107331


namespace range_of_a_l1073_107362

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a ≤ abs (x - 5) + abs (x - 3)) → a ≤ 2 := by
  sorry

end range_of_a_l1073_107362


namespace train_speed_is_18_kmh_l1073_107373

noncomputable def speed_of_train (length_of_bridge length_of_train time : ℝ) : ℝ :=
  (length_of_bridge + length_of_train) / time * 3.6

theorem train_speed_is_18_kmh
  (length_of_bridge : ℝ)
  (length_of_train : ℝ)
  (time : ℝ)
  (h1 : length_of_bridge = 200)
  (h2 : length_of_train = 100)
  (h3 : time = 60) :
  speed_of_train length_of_bridge length_of_train time = 18 :=
by
  sorry

end train_speed_is_18_kmh_l1073_107373


namespace amy_local_calls_l1073_107369

theorem amy_local_calls (L I : ℕ) 
  (h1 : 2 * L = 5 * I)
  (h2 : 3 * L = 5 * (I + 3)) : 
  L = 15 :=
by
  sorry

end amy_local_calls_l1073_107369


namespace a_equals_bc_l1073_107360

theorem a_equals_bc (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x y : ℝ, f x * g y = a * x * y + b * x + c * y + 1) → a = b * c :=
sorry

end a_equals_bc_l1073_107360


namespace solve_cos_2x_eq_cos_x_plus_sin_x_l1073_107352

open Real

theorem solve_cos_2x_eq_cos_x_plus_sin_x :
  ∀ x : ℝ,
    (cos (2 * x) = cos x + sin x) ↔
    (∃ k : ℤ, x = k * π - π / 4) ∨ 
    (∃ k : ℤ, x = 2 * k * π) ∨
    (∃ k : ℤ, x = 2 * k * π - π / 2) := 
sorry

end solve_cos_2x_eq_cos_x_plus_sin_x_l1073_107352


namespace number_of_real_z5_is_10_l1073_107330

theorem number_of_real_z5_is_10 :
  ∃ S : Finset ℂ, (∀ z ∈ S, z ^ 30 = 1 ∧ (z ^ 5).im = 0) ∧ S.card = 10 :=
sorry

end number_of_real_z5_is_10_l1073_107330


namespace chess_tournament_l1073_107390

theorem chess_tournament (n : ℕ) (h : (n * (n - 1)) / 2 - ((n - 3) * (n - 4)) / 2 = 130) : n = 19 :=
sorry

end chess_tournament_l1073_107390


namespace average_salary_of_all_workers_l1073_107325

def totalTechnicians : Nat := 6
def avgSalaryTechnician : Nat := 12000
def restWorkers : Nat := 6
def avgSalaryRest : Nat := 6000
def totalWorkers : Nat := 12
def totalSalary := (totalTechnicians * avgSalaryTechnician) + (restWorkers * avgSalaryRest)

theorem average_salary_of_all_workers : totalSalary / totalWorkers = 9000 := 
by
    -- replace with mathematical proof once available
    sorry

end average_salary_of_all_workers_l1073_107325


namespace find_initial_lion_population_l1073_107344

-- Define the conditions as integers
def lion_cubs_per_month : ℕ := 5
def lions_die_per_month : ℕ := 1
def total_lions_after_one_year : ℕ := 148

-- Define a formula for calculating the initial number of lions
def initial_number_of_lions (net_increase : ℕ) (final_count : ℕ) (months : ℕ) : ℕ :=
  final_count - (net_increase * months)

-- Main theorem statement
theorem find_initial_lion_population : initial_number_of_lions (lion_cubs_per_month - lions_die_per_month) total_lions_after_one_year 12 = 100 :=
  sorry

end find_initial_lion_population_l1073_107344


namespace total_presents_l1073_107315

variables (ChristmasPresents BirthdayPresents EasterPresents HalloweenPresents : ℕ)

-- Given conditions
def condition1 : ChristmasPresents = 60 := sorry
def condition2 : BirthdayPresents = 3 * EasterPresents := sorry
def condition3 : EasterPresents = (ChristmasPresents / 2) - 10 := sorry
def condition4 : HalloweenPresents = BirthdayPresents - EasterPresents := sorry

-- Proof statement
theorem total_presents (h1 : ChristmasPresents = 60)
    (h2 : BirthdayPresents = 3 * EasterPresents)
    (h3 : EasterPresents = (ChristmasPresents / 2) - 10)
    (h4 : HalloweenPresents = BirthdayPresents - EasterPresents) :
    ChristmasPresents + BirthdayPresents + EasterPresents + HalloweenPresents = 180 :=
sorry

end total_presents_l1073_107315


namespace candy_sampling_percentage_l1073_107342

theorem candy_sampling_percentage (total_percentage caught_percentage not_caught_percentage : ℝ) 
  (h1 : caught_percentage = 22 / 100) 
  (h2 : total_percentage = 24.444444444444443 / 100) 
  (h3 : not_caught_percentage = 2.444444444444443 / 100) :
  total_percentage = caught_percentage + not_caught_percentage :=
by
  sorry

end candy_sampling_percentage_l1073_107342


namespace tan_beta_half_l1073_107399

theorem tan_beta_half (α β : ℝ)
    (h1 : Real.tan α = 1 / 3)
    (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
    Real.tan β = 1 / 2 := 
sorry

end tan_beta_half_l1073_107399


namespace oranges_kilos_bought_l1073_107302

-- Definitions based on the given conditions
variable (O A x : ℝ)

-- Definitions from conditions
def A_value : Prop := A = 29
def equation1 : Prop := x * O + 5 * A = 419
def equation2 : Prop := 5 * O + 7 * A = 488

-- The theorem we want to prove
theorem oranges_kilos_bought {O A x : ℝ} (A_value: A = 29) (h1: x * O + 5 * A = 419) (h2: 5 * O + 7 * A = 488) : x = 5 :=
by
  -- start of proof
  sorry  -- proof omitted

end oranges_kilos_bought_l1073_107302


namespace desired_percentage_of_alcohol_l1073_107358

theorem desired_percentage_of_alcohol 
  (original_volume : ℝ)
  (original_percentage : ℝ)
  (added_volume : ℝ)
  (added_percentage : ℝ)
  (final_percentage : ℝ) :
  original_volume = 6 →
  original_percentage = 0.35 →
  added_volume = 1.8 →
  added_percentage = 1.0 →
  final_percentage = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end desired_percentage_of_alcohol_l1073_107358


namespace range_a_implies_not_purely_imaginary_l1073_107356

def is_not_purely_imaginary (z : ℂ) : Prop :=
  z.re ≠ 0

theorem range_a_implies_not_purely_imaginary (a : ℝ) :
  ¬ is_not_purely_imaginary ⟨a^2 - a - 2, abs (a - 1) - 1⟩ ↔ a ≠ -1 :=
by
  sorry

end range_a_implies_not_purely_imaginary_l1073_107356
