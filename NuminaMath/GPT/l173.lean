import Mathlib

namespace team_points_behind_l173_17357

-- Define the points for Max, Dulce and the condition for Val
def max_points : ℕ := 5
def dulce_points : ℕ := 3
def combined_points_max_dulce : ℕ := max_points + dulce_points
def val_points : ℕ := 2 * combined_points_max_dulce

-- Define the total points for their team and the opponents' team
def their_team_points : ℕ := max_points + dulce_points + val_points
def opponents_team_points : ℕ := 40

-- Proof statement
theorem team_points_behind : opponents_team_points - their_team_points = 16 :=
by
  sorry

end team_points_behind_l173_17357


namespace parallelepiped_vectors_l173_17338

theorem parallelepiped_vectors (x y z : ℝ)
  (h1: ∀ (AB BC CC1 AC1 : ℝ), AC1 = AB + BC + CC1)
  (h2: ∀ (AB BC CC1 AC1 : ℝ), AC1 = x * AB + 2 * y * BC + 3 * z * CC1) :
  x + y + z = 11 / 6 :=
by
  -- This is where the proof would go, but as per the instruction we'll add sorry.
  sorry

end parallelepiped_vectors_l173_17338


namespace total_cookies_and_brownies_l173_17380

-- Define the conditions
def bagsOfCookies : ℕ := 272
def cookiesPerBag : ℕ := 45
def bagsOfBrownies : ℕ := 158
def browniesPerBag : ℕ := 32

-- Define the total cookies, total brownies, and total items
def totalCookies := bagsOfCookies * cookiesPerBag
def totalBrownies := bagsOfBrownies * browniesPerBag
def totalItems := totalCookies + totalBrownies

-- State the theorem to prove
theorem total_cookies_and_brownies : totalItems = 17296 := by
  sorry

end total_cookies_and_brownies_l173_17380


namespace min_value_hyperbola_l173_17349

theorem min_value_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ e : ℝ, e = 2 ∧ (b^2 = (e * a)^2 - a^2)) :
  (a * 3 + 1 / a) = 2 * Real.sqrt 3 :=
by
  sorry

end min_value_hyperbola_l173_17349


namespace Ned_washed_shirts_l173_17337

-- Definitions based on conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts
def not_washed_shirts : ℕ := 1
def washed_shirts : ℕ := total_shirts - not_washed_shirts

-- Statement to prove
theorem Ned_washed_shirts : washed_shirts = 29 := by
  sorry

end Ned_washed_shirts_l173_17337


namespace new_outsiders_count_l173_17316

theorem new_outsiders_count (total_people: ℕ) (initial_snackers: ℕ)
  (first_group_outsiders: ℕ) (first_group_leave_half: ℕ) 
  (second_group_leave_count: ℕ) (half_remaining_leave: ℕ) (final_snackers: ℕ) 
  (total_snack_eaters: ℕ) 
  (initial_snackers_eq: total_people = 200) 
  (snackers_eq: initial_snackers = 100) 
  (first_group_outsiders_eq: first_group_outsiders = 20) 
  (first_group_leave_half_eq: first_group_leave_half = 60) 
  (second_group_leave_count_eq: second_group_leave_count = 30) 
  (half_remaining_leave_eq: half_remaining_leave = 15) 
  (final_snackers_eq: final_snackers = 20) 
  (total_snack_eaters_eq: total_snack_eaters = 120): 
  (60 - (second_group_leave_count + half_remaining_leave + final_snackers)) = 40 := 
by sorry

end new_outsiders_count_l173_17316


namespace parallelogram_base_length_l173_17310

theorem parallelogram_base_length 
  (area : ℝ)
  (b h : ℝ)
  (h_area : area = 128)
  (h_altitude : h = 2 * b) 
  (h_area_eq : area = b * h) : 
  b = 8 :=
by
  -- Proof goes here
  sorry

end parallelogram_base_length_l173_17310


namespace infinite_nested_radical_solution_l173_17327

theorem infinite_nested_radical_solution (x : ℝ) (h : x = Real.sqrt (4 + 3 * x)) : x = 4 := 
by 
  sorry

end infinite_nested_radical_solution_l173_17327


namespace number_2018_location_l173_17364

-- Define the odd square pattern as starting positions of rows
def odd_square (k : ℕ) : ℕ := (2 * k - 1) ^ 2

-- Define the conditions in terms of numbers in each row
def start_of_row (n : ℕ) : ℕ := (2 * n - 1) ^ 2 + 1

def number_at_row_column (n m : ℕ) :=
  start_of_row n + (m - 1)

theorem number_2018_location :
  number_at_row_column 44 82 = 2018 :=
by
  sorry

end number_2018_location_l173_17364


namespace line_ellipse_intersection_l173_17393

-- Define the problem conditions and the proof problem statement.
theorem line_ellipse_intersection (k m : ℝ) : 
  (∀ x y, y - k * x - 1 = 0 → ((x^2 / 5) + (y^2 / m) = 1)) →
  (m ≥ 1) ∧ (m ≠ 5) ∧ (m < 5 ∨ m > 5) :=
sorry

end line_ellipse_intersection_l173_17393


namespace solve_for_a_l173_17370

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem solve_for_a (a : ℤ) : star a 3 = 18 → a = 15 := by
  intro h₁
  sorry

end solve_for_a_l173_17370


namespace line_through_P_with_intercepts_l173_17317

theorem line_through_P_with_intercepts (a b : ℝ) (P : ℝ × ℝ) (hP : P = (6, -1)) 
  (h1 : a = 3 * b) (ha : a = 1 / ((-b - 1) / 6) + 6) (hb : b = -6 * ((-b - 1) / 6) - 1) :
  (∀ x y, y = (-1 / 3) * x + 1 ∨ y = (-1 / 6) * x) :=
sorry

end line_through_P_with_intercepts_l173_17317


namespace tan_half_sum_sq_l173_17355

theorem tan_half_sum_sq (a b : ℝ) : 
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 → 
  ∃ (x : ℝ), (x = (Real.tan (a / 2) + Real.tan (b / 2))^2) ∧ (x = 6 ∨ x = 26) := 
by
  intro h
  sorry

end tan_half_sum_sq_l173_17355


namespace multiply_and_simplify_fractions_l173_17382

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l173_17382


namespace hexagon_angles_l173_17394

theorem hexagon_angles (a e : ℝ) (h1 : a = e - 60) (h2 : 4 * a + 2 * e = 720) :
  e = 160 :=
by
  sorry

end hexagon_angles_l173_17394


namespace gcf_4370_13824_l173_17368

/-- Define the two numbers 4370 and 13824 -/
def num1 := 4370
def num2 := 13824

/-- The statement that the GCF of num1 and num2 is 1 -/
theorem gcf_4370_13824 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcf_4370_13824_l173_17368


namespace cube_side_length_in_cone_l173_17309

noncomputable def side_length_of_inscribed_cube (r h : ℝ) : ℝ :=
  if r = 1 ∧ h = 3 then (3 * Real.sqrt 2) / (3 + Real.sqrt 2) else 0

theorem cube_side_length_in_cone :
  side_length_of_inscribed_cube 1 3 = (3 * Real.sqrt 2) / (3 + Real.sqrt 2) :=
by
  sorry

end cube_side_length_in_cone_l173_17309


namespace books_left_over_l173_17335

-- Define the conditions as variables in Lean
def total_books : ℕ := 1500
def new_shelf_capacity : ℕ := 28

-- State the theorem based on these conditions
theorem books_left_over : total_books % new_shelf_capacity = 14 :=
by
  sorry

end books_left_over_l173_17335


namespace problem_M_m_evaluation_l173_17342

theorem problem_M_m_evaluation
  (a b c d e : ℝ)
  (h : a < b)
  (h' : b < c)
  (h'' : c < d)
  (h''' : d < e)
  (h'''' : a < e) :
  (max (min a (max b c))
       (max (min a d) (max b e))) = e := 
by
  sorry

end problem_M_m_evaluation_l173_17342


namespace reducibility_implies_divisibility_l173_17348

theorem reducibility_implies_divisibility
  (a b c d l k : ℤ)
  (p q : ℤ)
  (h1 : a * l + b = k * p)
  (h2 : c * l + d = k * q) :
  k ∣ (a * d - b * c) :=
sorry

end reducibility_implies_divisibility_l173_17348


namespace F_2_f_3_equals_341_l173_17386

def f (a : ℕ) : ℕ := a^2 - 2
def F (a b : ℕ) : ℕ := b^3 - a

theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end F_2_f_3_equals_341_l173_17386


namespace symmetric_scanning_codes_count_l173_17375

structure Grid (n : ℕ) :=
  (cells : Fin n × Fin n → Bool)

def is_symmetric_90 (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - j, i)

def is_symmetric_reflection_mid_side (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - i, j) ∧ g.cells (i, j) = g.cells (i, 7 - j)

def is_symmetric_reflection_diagonal (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (j, i)

def has_at_least_one_black_and_one_white (g : Grid 8) : Prop :=
  ∃ i j, g.cells (i, j) ∧ ∃ i j, ¬g.cells (i, j)

noncomputable def count_symmetric_scanning_codes : ℕ :=
  (sorry : ℕ)

theorem symmetric_scanning_codes_count : count_symmetric_scanning_codes = 62 :=
  sorry

end symmetric_scanning_codes_count_l173_17375


namespace more_divisible_by_7_than_11_l173_17354

open Nat

theorem more_divisible_by_7_than_11 :
  let N := 10000
  let count_7_not_11 := (N / 7) - (N / 77)
  let count_11_not_7 := (N / 11) - (N / 77)
  count_7_not_11 > count_11_not_7 := 
  by
    let N := 10000
    let count_7_not_11 := (N / 7) - (N / 77)
    let count_11_not_7 := (N / 11) - (N / 77)
    sorry

end more_divisible_by_7_than_11_l173_17354


namespace black_balls_count_l173_17322

theorem black_balls_count
  (P_red P_white : ℝ)
  (Red_balls_count : ℕ)
  (h1 : P_red = 0.42)
  (h2 : P_white = 0.28)
  (h3 : Red_balls_count = 21) :
  ∃ B, B = 15 :=
by
  sorry

end black_balls_count_l173_17322


namespace minimum_value_x_squared_plus_y_squared_l173_17395

-- We define our main proposition in Lean
theorem minimum_value_x_squared_plus_y_squared (x y : ℝ) 
  (h : (x + 5)^2 + (y - 12)^2 = 196) : x^2 + y^2 ≥ 169 :=
sorry

end minimum_value_x_squared_plus_y_squared_l173_17395


namespace correct_calculation_of_exponentiation_l173_17315

theorem correct_calculation_of_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end correct_calculation_of_exponentiation_l173_17315


namespace find_f_of_9_l173_17332

theorem find_f_of_9 (α : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x ^ α)
  (h2 : f 2 = Real.sqrt 2) :
  f 9 = 3 :=
sorry

end find_f_of_9_l173_17332


namespace highest_probability_highspeed_rail_l173_17371

def total_balls : ℕ := 10
def beidou_balls : ℕ := 3
def tianyan_balls : ℕ := 2
def highspeed_rail_balls : ℕ := 5

theorem highest_probability_highspeed_rail :
  (highspeed_rail_balls : ℚ) / total_balls > (beidou_balls : ℚ) / total_balls ∧
  (highspeed_rail_balls : ℚ) / total_balls > (tianyan_balls : ℚ) / total_balls :=
by {
  -- Proof skipped
  sorry
}

end highest_probability_highspeed_rail_l173_17371


namespace find_roots_l173_17372

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_roots 
  (h_symm : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h_three_roots : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0)
  (h_zero_root : f 0 = 0) :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ f a = 0 ∧ f b = 0 :=
sorry

end find_roots_l173_17372


namespace intersection_of_M_and_N_l173_17328

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | x < 1}
def expected_intersection : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = expected_intersection :=
sorry

end intersection_of_M_and_N_l173_17328


namespace time_for_A_and_C_to_complete_work_l173_17397

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end time_for_A_and_C_to_complete_work_l173_17397


namespace prism_closed_polygonal_chain_impossible_l173_17381

theorem prism_closed_polygonal_chain_impossible
  (lateral_edges : ℕ)
  (base_edges : ℕ)
  (total_edges : ℕ)
  (h_lateral : lateral_edges = 171)
  (h_base : base_edges = 171)
  (h_total : total_edges = 513)
  (h_total_sum : total_edges = 2 * base_edges + lateral_edges) :
  ¬ (∃ f : Fin 513 → (ℝ × ℝ × ℝ), (f 513 = f 0) ∧
    ∀ i, ( f (i + 1) - f i = (1, 0, 0) ∨ f (i + 1) - f i = (0, 1, 0) ∨ f (i + 1) - f i = (0, 0, 1) ∨ f (i + 1) - f i = (0, 0, -1) )) :=
by
  sorry

end prism_closed_polygonal_chain_impossible_l173_17381


namespace baseball_singles_percentage_l173_17312

theorem baseball_singles_percentage :
  let total_hits := 50
  let home_runs := 2
  let triples := 3
  let doubles := 8
  let non_single_hits := home_runs + triples + doubles
  let singles := total_hits - non_single_hits
  let singles_percentage := (singles / total_hits) * 100
  singles = 37 ∧ singles_percentage = 74 :=
by
  sorry

end baseball_singles_percentage_l173_17312


namespace sum_y_coords_l173_17360

theorem sum_y_coords (h1 : ∃(y : ℝ), (0 + 3)^2 + (y - 5)^2 = 64) : 
  ∃ y1 y2 : ℝ, y1 + y2 = 10 ∧ (0, y1) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) ∧ 
                            (0, y2) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) := 
by
  sorry

end sum_y_coords_l173_17360


namespace _l173_17367

variables (a b c : ℝ)
-- Conditionally define the theorem giving the constraints in the context.
example (h1 : a < 0) (h2 : b < 0) (h3 : c > 0) : 
  abs a - abs (a + b) + abs (c - a) + abs (b - c) = 2 * c - a := by 
sorry

end _l173_17367


namespace circle_condition_m_l173_17303

theorem circle_condition_m (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_m_l173_17303


namespace bounds_for_a_l173_17333

theorem bounds_for_a (a : ℝ) (h_a : a > 0) :
  ∀ x : ℝ, 0 < x ∧ x < 17 → (3 / 4) * x = (5 / 6) * (17 - x) + a → a < (153 / 12) := 
sorry

end bounds_for_a_l173_17333


namespace ray_walks_to_high_school_7_l173_17301

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

end ray_walks_to_high_school_7_l173_17301


namespace jenni_age_l173_17385

theorem jenni_age 
    (B J : ℤ)
    (h1 : B + J = 70)
    (h2 : B - J = 32) : 
    J = 19 :=
by
  sorry

end jenni_age_l173_17385


namespace inflection_point_on_3x_l173_17305

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x
noncomputable def f' (x : ℝ) : ℝ := 3 + 4 * Real.cos x + Real.sin x
noncomputable def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

theorem inflection_point_on_3x {x0 : ℝ} (h : f'' x0 = 0) : (f x0) = 3 * x0 := by
  sorry

end inflection_point_on_3x_l173_17305


namespace max_xy_l173_17307

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  xy ≤ 1 / 4 := 
sorry

end max_xy_l173_17307


namespace crocus_bulbs_count_l173_17365

theorem crocus_bulbs_count (C D : ℕ) 
  (h1 : C + D = 55) 
  (h2 : 0.35 * (C : ℝ) + 0.65 * (D : ℝ) = 29.15) :
  C = 22 :=
sorry

end crocus_bulbs_count_l173_17365


namespace polynomial_value_l173_17351

theorem polynomial_value (a : ℝ) (h : a^2 + 2 * a = 1) : 
  2 * a^5 + 7 * a^4 + 5 * a^3 + 2 * a^2 + 5 * a + 1 = 4 :=
by
  sorry

end polynomial_value_l173_17351


namespace tops_count_l173_17399

def price_eq (C T : ℝ) : Prop := 3 * C + 6 * T = 1500 ∧ C + 12 * T = 1500

def tops_to_buy (C T : ℝ) (num_tops : ℝ) : Prop := 500 = 100 * num_tops

theorem tops_count (C T num_tops : ℝ) (h1 : price_eq C T) (h2 : tops_to_buy C T num_tops) : num_tops = 5 :=
by
  sorry

end tops_count_l173_17399


namespace takeoff_run_length_l173_17398

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end takeoff_run_length_l173_17398


namespace max_pots_l173_17396

theorem max_pots (x y z : ℕ) (h₁ : 3 * x + 4 * y + 9 * z = 100) (h₂ : 1 ≤ x) (h₃ : 1 ≤ y) (h₄ : 1 ≤ z) : 
  z ≤ 10 :=
sorry

end max_pots_l173_17396


namespace simplify_fraction_l173_17306

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end simplify_fraction_l173_17306


namespace preimage_of_4_3_is_2_1_l173_17331

theorem preimage_of_4_3_is_2_1 :
  ∃ (a b : ℝ), (a + 2 * b = 4) ∧ (2 * a - b = 3) ∧ (a = 2) ∧ (b = 1) :=
by
  exists 2
  exists 1
  constructor
  { sorry }
  constructor
  { sorry }
  constructor
  { sorry }
  { sorry }


end preimage_of_4_3_is_2_1_l173_17331


namespace sum_of_digits_l173_17320

def digits (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

def P := 1
def Q := 0
def R := 2
def S := 5
def T := 6

theorem sum_of_digits :
  digits P ∧ digits Q ∧ digits R ∧ digits S ∧ digits T ∧ 
  (10000 * P + 1000 * Q + 100 * R + 10 * S + T) * 4 = 41024 →
  P + Q + R + S + T = 14 :=
by
  sorry

end sum_of_digits_l173_17320


namespace kids_bike_wheels_l173_17392

theorem kids_bike_wheels
  (x : ℕ) 
  (h1 : 7 * 2 + 11 * x = 58) :
  x = 4 :=
sorry

end kids_bike_wheels_l173_17392


namespace nth_equation_proof_l173_17334

theorem nth_equation_proof (n : ℕ) (hn : n > 0) :
  (1 : ℝ) + (1 / (n : ℝ)) - (2 / (2 * n - 1)) = (2 * n^2 + n + 1) / (n * (2 * n - 1)) :=
by
  sorry

end nth_equation_proof_l173_17334


namespace determine_f_36_l173_17329

def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n, f (n + 1) > f n

def multiplicative (f : ℕ → ℕ) : Prop :=
  ∀ m n, f (m * n) = f m * f n

def special_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n, m > n → m^m = n^n → f m = n

theorem determine_f_36 (f : ℕ → ℕ)
  (H1: strictly_increasing f)
  (H2: multiplicative f)
  (H3: special_condition f)
  : f 36 = 1296 := 
sorry

end determine_f_36_l173_17329


namespace roof_area_l173_17373

theorem roof_area (w l : ℕ) (h1 : l = 4 * w) (h2 : l - w = 42) : l * w = 784 :=
by
  sorry

end roof_area_l173_17373


namespace factorize_expression_l173_17325

theorem factorize_expression (a b m : ℝ) :
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) :=
by sorry

end factorize_expression_l173_17325


namespace triangle_y_values_l173_17387

theorem triangle_y_values (y : ℕ) :
  (8 + 11 > y^2) ∧ (y^2 + 8 > 11) ∧ (y^2 + 11 > 8) ↔ y = 2 ∨ y = 3 ∨ y = 4 :=
by
  sorry

end triangle_y_values_l173_17387


namespace random_event_l173_17311

theorem random_event (a b : ℝ) (h1 : a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0):
  ¬ (∀ a b, a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0 → a + b < 0) :=
by
  sorry

end random_event_l173_17311


namespace boys_went_down_the_slide_total_l173_17353

/-- Conditions -/
def a : Nat := 87
def b : Nat := 46
def c : Nat := 29

/-- The main proof problem -/
theorem boys_went_down_the_slide_total :
  a + b + c = 162 :=
by
  sorry

end boys_went_down_the_slide_total_l173_17353


namespace infinite_series_sum_l173_17363

theorem infinite_series_sum :
  (∑' n : ℕ, n * (1/5)^n) = 5/16 :=
by sorry

end infinite_series_sum_l173_17363


namespace floor_condition_x_l173_17345

theorem floor_condition_x (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 48) : 8 ≤ x ∧ x < 49 / 6 := 
by 
  sorry

end floor_condition_x_l173_17345


namespace greatest_difference_47x_l173_17336

def is_multiple_of_4 (n : Nat) : Prop :=
  n % 4 = 0

def valid_digit (d : Nat) : Prop :=
  d < 10

theorem greatest_difference_47x :
  ∃ x y : Nat, (is_multiple_of_4 (470 + x) ∧ valid_digit x) ∧ (is_multiple_of_4 (470 + y) ∧ valid_digit y) ∧ (x < y) ∧ (y - x = 4) :=
sorry

end greatest_difference_47x_l173_17336


namespace impossibility_triplet_2002x2002_grid_l173_17388

theorem impossibility_triplet_2002x2002_grid: 
  ∀ (M : Matrix ℕ (Fin 2002) (Fin 2002)),
    (∀ i j : Fin 2002, ∃ (r1 r2 r3 : Fin 2002), 
      (M i r1 > 0 ∧ M i r2 > 0 ∧ M i r3 > 0) ∨ 
      (M r1 j > 0 ∧ M r2 j > 0 ∧ M r3 j > 0)) →
    ¬ (∀ i j : Fin 2002, ∃ (a b c : ℕ), 
      M i j = a ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
      (∃ (r1 r2 r3 : Fin 2002), 
        (M i r1 = a ∨ M i r1 = b ∨ M i r1 = c) ∧ 
        (M i r2 = a ∨ M i r2 = b ∨ M i r2 = c) ∧ 
        (M i r3 = a ∨ M i r3 = b ∨ M i r3 = c)) ∨
      (∃ (c1 c2 c3 : Fin 2002), 
        (M c1 j = a ∨ M c1 j = b ∨ M c1 j = c) ∧ 
        (M c2 j = a ∨ M c2 j = b ∨ M c2 j = c) ∧ 
        (M c3 j = a ∨ M c3 j = b ∨ M c3 j = c)))
:= sorry

end impossibility_triplet_2002x2002_grid_l173_17388


namespace power_of_7_mod_8_l173_17369

theorem power_of_7_mod_8 : 7^123 % 8 = 7 :=
by sorry

end power_of_7_mod_8_l173_17369


namespace volume_PABCD_l173_17308

noncomputable def volume_of_pyramid (AB BC : ℝ) (PA : ℝ) : ℝ :=
  (1 / 3) * (AB * BC) * PA

theorem volume_PABCD (AB BC : ℝ) (h_AB : AB = 10) (h_BC : BC = 5)
  (PA : ℝ) (h_PA : PA = 2 * BC) :
  volume_of_pyramid AB BC PA = 500 / 3 :=
by
  subst h_AB
  subst h_BC
  subst h_PA
  -- At this point, we assert that everything simplifies correctly.
  -- This fill in the details for the correct expressions.
  sorry

end volume_PABCD_l173_17308


namespace trader_sold_23_bags_l173_17374

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end trader_sold_23_bags_l173_17374


namespace fabric_delivered_on_monday_amount_l173_17383

noncomputable def cost_per_yard : ℝ := 2
noncomputable def earnings : ℝ := 140

def fabric_delivered_on_monday (x : ℝ) : Prop :=
  let tuesday := 2 * x
  let wednesday := (1 / 4) * tuesday
  let total_yards := x + tuesday + wednesday
  let total_earnings := total_yards * cost_per_yard
  total_earnings = earnings

theorem fabric_delivered_on_monday_amount : ∃ x : ℝ, fabric_delivered_on_monday x ∧ x = 20 :=
by sorry

end fabric_delivered_on_monday_amount_l173_17383


namespace intersection_in_fourth_quadrant_l173_17302

theorem intersection_in_fourth_quadrant (a : ℝ) : 
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ x > 0 ∧ y < 0) → a > 1 / 2 := 
by 
  sorry

end intersection_in_fourth_quadrant_l173_17302


namespace parallel_line_through_point_l173_17350

theorem parallel_line_through_point (x y : ℝ) :
  (∃ (b : ℝ), (∀ (x : ℝ), y = 2 * x + b) ∧ y = 2 * 1 - 4) :=
sorry

end parallel_line_through_point_l173_17350


namespace sum_faces_of_cube_l173_17340

-- Conditions in Lean 4
variables (a b c d e f : ℕ)

-- Sum of vertex labels
def vertex_sum := a * b * c + a * e * c + a * b * f + a * e * f +
                  d * b * c + d * e * c + d * b * f + d * e * f

-- Theorem statement
theorem sum_faces_of_cube (h : vertex_sum a b c d e f = 1001) :
  (a + d) + (b + e) + (c + f) = 31 :=
sorry

end sum_faces_of_cube_l173_17340


namespace ratio_daves_bench_to_weight_l173_17366

variables (wD bM bD bC : ℝ)

def daves_weight := wD = 175
def marks_bench_press := bM = 55
def marks_comparison_to_craig := bM = bC - 50
def craigs_comparison_to_dave := bC = 0.20 * bD

theorem ratio_daves_bench_to_weight
  (h1 : daves_weight wD)
  (h2 : marks_bench_press bM)
  (h3 : marks_comparison_to_craig bM bC)
  (h4 : craigs_comparison_to_dave bC bD) :
  (bD / wD) = 3 :=
by
  rw [daves_weight] at h1
  rw [marks_bench_press] at h2
  rw [marks_comparison_to_craig] at h3
  rw [craigs_comparison_to_dave] at h4
  -- Now we have:
  -- 1. wD = 175
  -- 2. bM = 55
  -- 3. bM = bC - 50
  -- 4. bC = 0.20 * bD
  -- We proceed to solve:
  sorry

end ratio_daves_bench_to_weight_l173_17366


namespace sample_frequency_in_range_l173_17321

theorem sample_frequency_in_range :
  let total_capacity := 100
  let freq_0_10 := 12
  let freq_10_20 := 13
  let freq_20_30 := 24
  let freq_30_40 := 15
  (freq_0_10 + freq_10_20 + freq_20_30 + freq_30_40) / total_capacity = 0.64 :=
by
  sorry

end sample_frequency_in_range_l173_17321


namespace Yeonseo_skirts_l173_17352

theorem Yeonseo_skirts
  (P : ℕ)
  (more_than_two_skirts : ∀ S : ℕ, S > 2)
  (more_than_two_pants : P > 2)
  (ways_to_choose : P + 3 = 7) :
  ∃ S : ℕ, S = 3 := by
  sorry

end Yeonseo_skirts_l173_17352


namespace binom_sub_floor_div_prime_l173_17313

theorem binom_sub_floor_div_prime {n p : ℕ} (hp : Nat.Prime p) (hpn : n ≥ p) : 
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end binom_sub_floor_div_prime_l173_17313


namespace value_proof_l173_17362

noncomputable def find_value (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x) : Prop :=
  2 * b - a + c = 195

theorem value_proof : ∃ (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x), find_value a b c h h_rat :=
  sorry

end value_proof_l173_17362


namespace sphere_volume_ratio_l173_17376

theorem sphere_volume_ratio (r1 r2 : ℝ) (S1 S2 V1 V2 : ℝ) 
(h1 : S1 = 4 * Real.pi * r1^2)
(h2 : S2 = 4 * Real.pi * r2^2)
(h3 : V1 = (4 / 3) * Real.pi * r1^3)
(h4 : V2 = (4 / 3) * Real.pi * r2^3)
(h_surface_ratio : S1 / S2 = 2 / 3) :
V1 / V2 = (2 * Real.sqrt 6) / 9 :=
by
  sorry

end sphere_volume_ratio_l173_17376


namespace determine_m_of_monotonically_increasing_function_l173_17330

theorem determine_m_of_monotonically_increasing_function 
  (m n : ℝ)
  (h : ∀ x, 12 * x ^ 2 + 2 * m * x + (m - 3) ≥ 0) :
  m = 6 := 
by 
  sorry

end determine_m_of_monotonically_increasing_function_l173_17330


namespace tax_percentage_first_tier_l173_17358

theorem tax_percentage_first_tier
  (car_price : ℝ)
  (total_tax : ℝ)
  (first_tier_level : ℝ)
  (second_tier_rate : ℝ)
  (first_tier_tax : ℝ)
  (T : ℝ)
  (h_car_price : car_price = 30000)
  (h_total_tax : total_tax = 5500)
  (h_first_tier_level : first_tier_level = 10000)
  (h_second_tier_rate : second_tier_rate = 0.15)
  (h_first_tier_tax : first_tier_tax = (T / 100) * first_tier_level) :
  T = 25 :=
by
  sorry

end tax_percentage_first_tier_l173_17358


namespace relationship_between_a_and_b_l173_17390

variable (a b : ℝ)

def in_interval (x : ℝ) := 0 < x ∧ x < 1

theorem relationship_between_a_and_b 
  (ha : in_interval a)
  (hb : in_interval b)
  (h : (1 - a) * b > 1 / 4) : a < b :=
sorry

end relationship_between_a_and_b_l173_17390


namespace greatest_integer_x_l173_17347

theorem greatest_integer_x (x : ℤ) : (5 : ℚ)/8 > (x : ℚ)/15 → x ≤ 9 :=
by {
  sorry
}

end greatest_integer_x_l173_17347


namespace smallest_multiple_of_seven_gt_neg50_l173_17384

theorem smallest_multiple_of_seven_gt_neg50 : ∃ (n : ℤ), n % 7 = 0 ∧ n > -50 ∧ ∀ (m : ℤ), m % 7 = 0 → m > -50 → n ≤ m :=
sorry

end smallest_multiple_of_seven_gt_neg50_l173_17384


namespace problem1_problem2_l173_17314

theorem problem1 (x : ℚ) (h : x - 2/11 = -1/3) : x = -5/33 :=
sorry

theorem problem2 : -2 - (-1/3 + 1/2) = -13/6 :=
sorry

end problem1_problem2_l173_17314


namespace linear_equation_in_one_variable_proof_l173_17377

noncomputable def is_linear_equation_in_one_variable (eq : String) : Prop :=
  eq = "3x = 2x" ∨ eq = "ax + b = 0"

theorem linear_equation_in_one_variable_proof :
  is_linear_equation_in_one_variable "3x = 2x" ∧ ¬is_linear_equation_in_one_variable "3x - (4 + 3x) = 2"
  ∧ ¬is_linear_equation_in_one_variable "x + y = 1" ∧ ¬is_linear_equation_in_one_variable "x^2 + 1 = 5" :=
by
  sorry

end linear_equation_in_one_variable_proof_l173_17377


namespace average_price_of_cow_l173_17343

theorem average_price_of_cow (total_price_cows_and_goats rs: ℕ) (num_cows num_goats: ℕ)
    (avg_price_goat: ℕ) (total_price: total_price_cows_and_goats = 1400)
    (num_cows_eq: num_cows = 2) (num_goats_eq: num_goats = 8)
    (avg_price_goat_eq: avg_price_goat = 60) :
    let total_price_goats := avg_price_goat * num_goats
    let total_price_cows := total_price_cows_and_goats - total_price_goats
    let avg_price_cow := total_price_cows / num_cows
    avg_price_cow = 460 :=
by
  sorry

end average_price_of_cow_l173_17343


namespace non_zero_number_is_9_l173_17359

theorem non_zero_number_is_9 (x : ℝ) (hx : x ≠ 0) (h : (x + x^2) / 2 = 5 * x) : x = 9 :=
sorry

end non_zero_number_is_9_l173_17359


namespace tiles_covering_the_floor_l173_17339

theorem tiles_covering_the_floor 
  (L W : ℕ) 
  (h1 : (∃ k, L = 10 * k) ∧ (∃ j, W = 10 * j))
  (h2 : W = 2 * L)
  (h3 : (L * L + W * W).sqrt = 45) :
  L * W = 810 :=
sorry

end tiles_covering_the_floor_l173_17339


namespace trigonometric_signs_problem_l173_17300

open Real

theorem trigonometric_signs_problem (k : ℤ) (θ α : ℝ) 
  (hα : α = 2 * k * π - π / 5)
  (h_terminal_side : ∃ m : ℤ, θ = α + 2 * m * π) :
  (sin θ / |sin θ|) + (cos θ / |cos θ|) + (tan θ / |tan θ|) = -1 := 
sorry

end trigonometric_signs_problem_l173_17300


namespace find_x_l173_17318

theorem find_x (x : ℝ) (h : (3 * x - 4) / 7 = 15) : x = 109 / 3 :=
by sorry

end find_x_l173_17318


namespace interest_rate_proof_l173_17379

noncomputable def compound_interest_rate (P A : ℝ) (t n : ℕ) : ℝ :=
  (((A / P)^(1 / (n * t))) - 1) * n

theorem interest_rate_proof :
  ∀ P A : ℝ, ∀ t n : ℕ, P = 1093.75 → A = 1183 → t = 2 → n = 1 →
  compound_interest_rate P A t n = 0.0399 :=
by
  intros P A t n hP hA ht hn
  rw [hP, hA, ht, hn]
  unfold compound_interest_rate
  sorry

end interest_rate_proof_l173_17379


namespace equivalence_mod_equivalence_divisible_l173_17344

theorem equivalence_mod (a b c : ℤ) :
  (∃ k : ℤ, a - b = k * c) ↔ (a % c = b % c) := by
  sorry

theorem equivalence_divisible (a b c : ℤ) :
  (a % c = b % c) ↔ (∃ k : ℤ, a - b = k * c) := by
  sorry

end equivalence_mod_equivalence_divisible_l173_17344


namespace vertex_sum_of_cube_l173_17323

noncomputable def cube_vertex_sum (a : Fin 8 → ℕ) : ℕ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7

def face_sums (a : Fin 8 → ℕ) : List ℕ :=
  [
    a 0 + a 1 + a 2 + a 3, -- first face
    a 0 + a 1 + a 4 + a 5, -- second face
    a 0 + a 3 + a 4 + a 7, -- third face
    a 1 + a 2 + a 5 + a 6, -- fourth face
    a 2 + a 3 + a 6 + a 7, -- fifth face
    a 4 + a 5 + a 6 + a 7  -- sixth face
  ]

def total_face_sum (a : Fin 8 → ℕ) : ℕ :=
  List.sum (face_sums a)

theorem vertex_sum_of_cube (a : Fin 8 → ℕ) (h : total_face_sum a = 2019) :
  cube_vertex_sum a = 673 :=
sorry

end vertex_sum_of_cube_l173_17323


namespace find_a_l173_17304

theorem find_a (x y a : ℝ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) (h3 : a * x - 3 * y = 23) : 
  a = 12.141 :=
by
  sorry

end find_a_l173_17304


namespace ten_digit_number_contains_repeated_digit_l173_17378

open Nat

theorem ten_digit_number_contains_repeated_digit
  (n : ℕ)
  (h1 : 10^9 ≤ n^2 + 1)
  (h2 : n^2 + 1 < 10^10) :
  ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ (digits 10 (n^2 + 1))) ∧ (d2 ∈ (digits 10 (n^2 + 1))) :=
sorry

end ten_digit_number_contains_repeated_digit_l173_17378


namespace length_of_QB_l173_17319

/-- 
Given a circle Q with a circumference of 16π feet, 
segment AB as its diameter, 
and the angle AQB of 120 degrees, 
prove that the length of segment QB is 8 feet.
-/
theorem length_of_QB (C : ℝ) (r : ℝ) (A B Q : ℝ) (angle_AQB : ℝ) 
  (h1 : C = 16 * Real.pi)
  (h2 : 2 * Real.pi * r = C)
  (h3 : angle_AQB = 120) 
  : QB = 8 :=
sorry

end length_of_QB_l173_17319


namespace probability_one_marble_each_color_l173_17389

theorem probability_one_marble_each_color :
  let total_marbles := 9
  let total_ways := Nat.choose total_marbles 3
  let favorable_ways := 3 * 3 * 3
  let probability := favorable_ways / total_ways
  probability = 9 / 28 :=
by
  sorry

end probability_one_marble_each_color_l173_17389


namespace total_items_purchased_l173_17326

/-- Proof that Ike and Mike buy a total of 9 items given the constraints. -/
theorem total_items_purchased
  (total_money : ℝ)
  (sandwich_cost : ℝ)
  (drink_cost : ℝ)
  (combo_factor : ℕ)
  (money_spent_on_sandwiches : ℝ)
  (number_of_sandwiches : ℕ)
  (number_of_drinks : ℕ)
  (num_free_sandwiches : ℕ) :
  total_money = 40 →
  sandwich_cost = 5 →
  drink_cost = 1.5 →
  combo_factor = 5 →
  number_of_sandwiches = 9 →
  number_of_drinks = 0 →
  money_spent_on_sandwiches = number_of_sandwiches * sandwich_cost →
  total_money = money_spent_on_sandwiches →
  num_free_sandwiches = number_of_sandwiches / combo_factor →
  number_of_sandwiches = number_of_sandwiches + num_free_sandwiches →
  number_of_sandwiches + number_of_drinks = 9 :=
by
  intros
  sorry

end total_items_purchased_l173_17326


namespace hexagon_classroom_students_l173_17391

-- Define the number of sleeping students
def num_sleeping_students (students_detected : Nat → Nat) :=
  students_detected 2 + students_detected 3 + students_detected 6

-- Define the condition that the sum of snore-o-meter readings is 7
def snore_o_meter_sum (students_detected : Nat → Nat) :=
  2 * students_detected 2 + 3 * students_detected 3 + 6 * students_detected 6 = 7

-- Proof that the number of sleeping students is 3 given the conditions
theorem hexagon_classroom_students : 
  ∀ (students_detected : Nat → Nat), snore_o_meter_sum students_detected → num_sleeping_students students_detected = 3 :=
by
  intro students_detected h
  sorry

end hexagon_classroom_students_l173_17391


namespace range_of_n_l173_17361

theorem range_of_n (m n : ℝ) (h : (m^2 - 2 * m)^2 + 4 * m^2 - 8 * m + 6 - n = 0) : n ≥ 3 :=
sorry

end range_of_n_l173_17361


namespace square_area_l173_17356

theorem square_area (p : ℕ) (h : p = 48) : (p / 4) * (p / 4) = 144 := by
  sorry

end square_area_l173_17356


namespace intersection_of_M_and_N_l173_17341

open Set

variable (M N : Set ℕ)

theorem intersection_of_M_and_N :
  M = {1, 2, 4, 8, 16} →
  N = {2, 4, 6, 8} →
  M ∩ N = {2, 4, 8} :=
by
  intros hM hN
  rw [hM, hN]
  ext x
  simp
  sorry

end intersection_of_M_and_N_l173_17341


namespace total_water_in_bucket_l173_17346

noncomputable def initial_gallons : ℝ := 3
noncomputable def added_gallons_1 : ℝ := 6.8
noncomputable def liters_to_gallons (liters : ℝ) : ℝ := liters / 3.78541
noncomputable def quart_to_gallons (quarts : ℝ) : ℝ := quarts / 4
noncomputable def added_gallons_2 : ℝ := liters_to_gallons 10
noncomputable def added_gallons_3 : ℝ := quart_to_gallons 4

noncomputable def total_gallons : ℝ :=
  initial_gallons + added_gallons_1 + added_gallons_2 + added_gallons_3

theorem total_water_in_bucket :
  abs (total_gallons - 13.44) < 0.01 :=
by
  -- convert amounts and perform arithmetic operations
  sorry

end total_water_in_bucket_l173_17346


namespace value_of_a_l173_17324

theorem value_of_a (k : ℝ) (a : ℝ) (b : ℝ) (h1 : a = k / b^2) (h2 : a = 10) (h3 : b = 24) :
  a = 40 :=
sorry

end value_of_a_l173_17324
