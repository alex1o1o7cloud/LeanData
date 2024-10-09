import Mathlib

namespace manuscript_pages_count_l973_97373

theorem manuscript_pages_count
  (P : ℕ)
  (cost_first_time : ℕ := 5 * P)
  (cost_once_revised : ℕ := 4 * 30)
  (cost_twice_revised : ℕ := 8 * 20)
  (total_cost : ℕ := 780)
  (h : cost_first_time + cost_once_revised + cost_twice_revised = total_cost) :
  P = 100 :=
sorry

end manuscript_pages_count_l973_97373


namespace range_of_b_l973_97355

theorem range_of_b (a b c : ℝ) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 24) : 
  1 ≤ b ∧ b ≤ 5 := 
sorry

end range_of_b_l973_97355


namespace moles_of_KI_formed_l973_97331

-- Define the given conditions
def moles_KOH : ℕ := 1
def moles_NH4I : ℕ := 1
def balanced_equation (KOH NH4I KI NH3 H2O : ℕ) : Prop :=
  (KOH = 1) ∧ (NH4I = 1) ∧ (KI = 1) ∧ (NH3 = 1) ∧ (H2O = 1)

-- The proof problem statement
theorem moles_of_KI_formed (h : balanced_equation moles_KOH moles_NH4I 1 1 1) : 
  1 = 1 :=
by sorry

end moles_of_KI_formed_l973_97331


namespace perpendicular_lines_have_given_slope_l973_97314

theorem perpendicular_lines_have_given_slope (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_have_given_slope_l973_97314


namespace simplify_sub_polynomials_l973_97349

def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 5 * r - 4
def g (r : ℝ) : ℝ := r^3 + 3 * r^2 + 7 * r - 2

theorem simplify_sub_polynomials (r : ℝ) : f r - g r = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end simplify_sub_polynomials_l973_97349


namespace beverage_price_function_l973_97304

theorem beverage_price_function (box_price : ℕ) (bottles_per_box : ℕ) (bottles_purchased : ℕ) (y : ℕ) :
  box_price = 55 →
  bottles_per_box = 6 →
  y = (55 * bottles_purchased) / 6 := 
sorry

end beverage_price_function_l973_97304


namespace solution_set_of_inequality_l973_97380

theorem solution_set_of_inequality (x : ℝ) : |5 * x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_of_inequality_l973_97380


namespace range_of_a_l973_97382

noncomputable def A : Set ℝ := {x | -2 ≤ x ∧ x < 4 }

noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (a : ℝ) : (B a ⊆ A) ↔ (0 ≤ a ∧ a < 3) := sorry

end range_of_a_l973_97382


namespace factorization_correct_l973_97315

theorem factorization_correct : 
  ∀ x : ℝ, (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) :=
by
  intros
  sorry

end factorization_correct_l973_97315


namespace dice_sum_probability_l973_97364

theorem dice_sum_probability (n : ℕ) (h : ∃ k : ℕ, (8 : ℕ) * k + k = 12) : n = 330 :=
sorry

end dice_sum_probability_l973_97364


namespace brick_wall_problem_l973_97334

theorem brick_wall_problem : 
  ∀ (B1 B2 B3 B4 B5 : ℕ) (d : ℕ),
  B1 = 38 →
  B1 + B2 + B3 + B4 + B5 = 200 →
  B2 = B1 - d →
  B3 = B1 - 2 * d →
  B4 = B1 - 3 * d →
  B5 = B1 - 4 * d →
  d = 1 :=
by
  intros B1 B2 B3 B4 B5 d h1 h2 h3 h4 h5 h6
  rw [h1] at h2
  sorry

end brick_wall_problem_l973_97334


namespace lollipops_left_for_becky_l973_97353
-- Import the Mathlib library

-- Define the conditions as given in the problem
def lemon_lollipops : ℕ := 75
def peppermint_lollipops : ℕ := 210
def watermelon_lollipops : ℕ := 6
def marshmallow_lollipops : ℕ := 504
def friends : ℕ := 13

-- Total number of lollipops
def total_lollipops : ℕ := lemon_lollipops + peppermint_lollipops + watermelon_lollipops + marshmallow_lollipops

-- Statement to prove that the remainder after distributing the total lollipops among friends is 2
theorem lollipops_left_for_becky : total_lollipops % friends = 2 := by
  -- Proof goes here
  sorry

end lollipops_left_for_becky_l973_97353


namespace square_side_length_l973_97347

theorem square_side_length (s : ℝ) (h : s^2 + s - 4 * s = 4) : s = 4 :=
sorry

end square_side_length_l973_97347


namespace problem_statement_l973_97362

noncomputable def two_arccos_equals_arcsin : Prop :=
  2 * Real.arccos (3 / 5) = Real.arcsin (24 / 25)

theorem problem_statement : two_arccos_equals_arcsin :=
  sorry

end problem_statement_l973_97362


namespace ab_value_l973_97323

theorem ab_value 
  (a b c : ℝ)
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2 * a - b) : 
  a * b = 17 := 
by 
  sorry

end ab_value_l973_97323


namespace find_a_l973_97312

theorem find_a (a : ℝ) (t : ℝ) :
  (4 = 1 + 3 * t) ∧ (3 = a * t^2 + 2) → a = 1 :=
by
  sorry

end find_a_l973_97312


namespace mixed_doubles_pairing_l973_97322

theorem mixed_doubles_pairing: 
  let males := 5
  let females := 4
  let choose_males := Nat.choose males 2
  let choose_females := Nat.choose females 2
  let arrangements := Nat.factorial 2
  choose_males * choose_females * arrangements = 120 := by
  sorry

end mixed_doubles_pairing_l973_97322


namespace speed_of_A_l973_97374

theorem speed_of_A :
  ∀ (v_A : ℝ), 
    (v_A * 2 + 7 * 2 = 24) → 
    v_A = 5 :=
by
  intro v_A
  intro h
  have h1 : v_A * 2 = 10 := by linarith
  have h2 : v_A = 5 := by linarith
  exact h2

end speed_of_A_l973_97374


namespace calculate_floor_100_p_l973_97392

noncomputable def max_prob_sum_7 : ℝ := 
  let p1 := 0.2
  let p6 := 0.1
  let p2_p5_p3_p4 := 0.7 - p1 - p6
  2 * (p1 * p6 + p2_p5_p3_p4 / 2 ^ 2)

theorem calculate_floor_100_p : ∃ p : ℝ, (⌊100 * max_prob_sum_7⌋ = 28) :=
  by
  sorry

end calculate_floor_100_p_l973_97392


namespace greatest_4_digit_number_l973_97396

theorem greatest_4_digit_number
  (n : ℕ)
  (h1 : n % 5 = 3)
  (h2 : n % 9 = 2)
  (h3 : 1000 ≤ n)
  (h4 : n < 10000) :
  n = 9962 := 
sorry

end greatest_4_digit_number_l973_97396


namespace area_of_square_KLMN_is_25_l973_97370

-- Given a square ABCD with area 25
def ABCD_area_is_25 : Prop :=
  ∃ s : ℝ, (s * s = 25)

-- Given points K, L, M, and N forming isosceles right triangles with the sides of the square
def isosceles_right_triangles_at_vertices (A B C D K L M N : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    (a = b) ∧ (c = d) ∧
    (K - A)^2 + (B - K)^2 = (A - B)^2 ∧  -- AKB
    (L - B)^2 + (C - L)^2 = (B - C)^2 ∧  -- BLC
    (M - C)^2 + (D - M)^2 = (C - D)^2 ∧  -- CMD
    (N - D)^2 + (A - N)^2 = (D - A)^2    -- DNA

-- Given that KLMN is a square
def KLMN_is_square (K L M N : ℝ) : Prop :=
  (K - L)^2 + (L - M)^2 = (M - N)^2 + (N - K)^2

-- Proving that the area of square KLMN is 25 given the conditions
theorem area_of_square_KLMN_is_25 (A B C D K L M N : ℝ) :
  ABCD_area_is_25 → isosceles_right_triangles_at_vertices A B C D K L M N → KLMN_is_square K L M N → ∃s, s * s = 25 :=
by
  intro h1 h2 h3
  sorry

end area_of_square_KLMN_is_25_l973_97370


namespace percentage_increase_l973_97385

theorem percentage_increase (use_per_six_months : ℝ) (new_annual_use : ℝ) : 
  use_per_six_months = 90 →
  new_annual_use = 216 →
  ((new_annual_use - 2 * use_per_six_months) / (2 * use_per_six_months)) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_increase_l973_97385


namespace Albert_more_rocks_than_Joshua_l973_97325

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_rocks_than_Joshua_l973_97325


namespace area_within_fence_is_328_l973_97339

-- Define the dimensions of the fenced area
def main_rectangle_length : ℝ := 20
def main_rectangle_width : ℝ := 18

-- Define the dimensions of the square cutouts
def cutout_length : ℝ := 4
def cutout_width : ℝ := 4

-- Calculate the areas
def main_rectangle_area : ℝ := main_rectangle_length * main_rectangle_width
def cutout_area : ℝ := cutout_length * cutout_width

-- Define the number of cutouts
def number_of_cutouts : ℝ := 2

-- Calculate the final area within the fence
def area_within_fence : ℝ := main_rectangle_area - number_of_cutouts * cutout_area

theorem area_within_fence_is_328 : area_within_fence = 328 := by
  -- This is a place holder for the proof, replace it with the actual proof
  sorry

end area_within_fence_is_328_l973_97339


namespace exists_student_not_wet_l973_97337

theorem exists_student_not_wet (n : ℕ) (students : Fin (2 * n + 1) → ℝ) (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → students i ≠ students j) : 
  ∃ i : Fin (2 * n + 1), ∀ j : Fin (2 * n + 1), (j ≠ i → students j ≠ students i) :=
  sorry

end exists_student_not_wet_l973_97337


namespace seven_pow_k_eq_two_l973_97386

theorem seven_pow_k_eq_two {k : ℕ} (h : 7 ^ (4 * k + 2) = 784) : 7 ^ k = 2 := 
by 
  sorry

end seven_pow_k_eq_two_l973_97386


namespace CatsFavoriteNumber_l973_97372

theorem CatsFavoriteNumber :
  ∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧ 
    (∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n = p1 * p2 * p3) ∧ 
    (∀ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      n ≠ a ∧ n ≠ b ∧ n ≠ c ∧ n ≠ d ∧
      a + b - c = d ∨ b + c - d = a ∨ c + d - a = b ∨ d + a - b = c →
      (a = 30 ∧ b = 42 ∧ c = 66 ∧ d = 78)) ∧
    (n = 70) := by
  sorry

end CatsFavoriteNumber_l973_97372


namespace geometric_series_sum_l973_97371

theorem geometric_series_sum : 
  ∑' n : ℕ, (5 / 3) * (-1 / 3) ^ n = (5 / 4) := by
  sorry

end geometric_series_sum_l973_97371


namespace basketball_games_won_difference_l973_97328

theorem basketball_games_won_difference :
  ∀ (total_games games_won games_lost difference_won_lost : ℕ),
  total_games = 62 →
  games_won = 45 →
  games_lost = 17 →
  difference_won_lost = games_won - games_lost →
  difference_won_lost = 28 :=
by
  intros total_games games_won games_lost difference_won_lost
  intros h_total h_won h_lost h_diff
  rw [h_won, h_lost] at h_diff
  exact h_diff

end basketball_games_won_difference_l973_97328


namespace tangent_division_l973_97366

theorem tangent_division (a b c d e : ℝ) (h0 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) :
  ∃ t1 t5 : ℝ, t1 = (a + b - c - d + e) / 2 ∧ t5 = (a - b - c + d + e) / 2 ∧ t1 + t5 = a :=
by
  sorry

end tangent_division_l973_97366


namespace union_M_N_equals_0_1_5_l973_97381

def M : Set ℝ := { x | x^2 - 6 * x + 5 = 0 }
def N : Set ℝ := { x | x^2 - 5 * x = 0 }

theorem union_M_N_equals_0_1_5 : M ∪ N = {0, 1, 5} := by
  sorry

end union_M_N_equals_0_1_5_l973_97381


namespace polygon_sides_l973_97354

-- Definitions based on the conditions provided
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

def sum_exterior_angles : ℝ := 360 

def condition (n : ℕ) : Prop :=
  sum_interior_angles n = 2 * sum_exterior_angles + 180

-- Main theorem based on the correct answer
theorem polygon_sides (n : ℕ) (h : condition n) : n = 7 :=
sorry

end polygon_sides_l973_97354


namespace concave_quadrilateral_area_l973_97375

noncomputable def area_of_concave_quadrilateral (AB BC CD AD : ℝ) (angle_BCD : ℝ) : ℝ :=
  let BD := Real.sqrt (BC * BC + CD * CD)
  let area_ABD := 0.5 * AB * BD
  let area_BCD := 0.5 * BC * CD
  area_ABD - area_BCD

theorem concave_quadrilateral_area :
  ∀ (AB BC CD AD : ℝ) (angle_BCD : ℝ),
    angle_BCD = Real.pi / 2 ∧ AB = 12 ∧ BC = 4 ∧ CD = 3 ∧ AD = 13 → 
    area_of_concave_quadrilateral AB BC CD AD angle_BCD = 24 :=
by
  intros AB BC CD AD angle_BCD h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end concave_quadrilateral_area_l973_97375


namespace xiaoming_original_phone_number_l973_97350

variable (d1 d2 d3 d4 d5 d6 : Nat)

def original_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

def upgraded_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  20000000 + 1000000 * d1 + 80000 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem xiaoming_original_phone_number :
  let x := original_phone_number d1 d2 d3 d4 d5 d6
  let x' := upgraded_phone_number d1 d2 d3 d4 d5 d6
  (x' = 81 * x) → (x = 282500) :=
by
  sorry

end xiaoming_original_phone_number_l973_97350


namespace depth_of_channel_l973_97390

theorem depth_of_channel (top_width bottom_width : ℝ) (area : ℝ) (h : ℝ) 
  (h_top : top_width = 14) (h_bottom : bottom_width = 8) (h_area : area = 770) :
  (1 / 2) * (top_width + bottom_width) * h = area → h = 70 :=
by
  intros h_trapezoid
  sorry

end depth_of_channel_l973_97390


namespace tank_fraction_full_l973_97333

theorem tank_fraction_full 
  (initial_fraction : ℚ)
  (full_capacity : ℚ)
  (added_water : ℚ)
  (initial_fraction_eq : initial_fraction = 3/4)
  (full_capacity_eq : full_capacity = 40)
  (added_water_eq : added_water = 5) :
  ((initial_fraction * full_capacity + added_water) / full_capacity) = 7/8 :=
by 
  sorry

end tank_fraction_full_l973_97333


namespace sin_cos_eq_one_l973_97338

open Real

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h2 : x < 2 * π) (h : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := 
by
  sorry

end sin_cos_eq_one_l973_97338


namespace area_of_highest_points_l973_97340

noncomputable def highest_point_area (u g : ℝ) : ℝ :=
  let x₁ := u^2 / (2 * g)
  let x₂ := 2 * u^2 / g
  (1/4) * ((x₂^2) - (x₁^2))

theorem area_of_highest_points (u g : ℝ) : highest_point_area u g = 3 * u^4 / (4 * g^2) :=
by
  sorry

end area_of_highest_points_l973_97340


namespace digits_sum_l973_97394

theorem digits_sum (P Q R : ℕ) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10)
  (h_eq : 100 * P + 10 * Q + R + 10 * Q + R = 1012) :
  P + Q + R = 20 :=
by {
  -- Implementation of the proof will go here
  sorry
}

end digits_sum_l973_97394


namespace initial_loss_percentage_l973_97302

theorem initial_loss_percentage 
  (C : ℝ) 
  (h1 : selling_price_one_pencil_20 = 1 / 20)
  (h2 : selling_price_one_pencil_10 = 1 / 10)
  (h3 : C = 1 / (10 * 1.30)) :
  (C - selling_price_one_pencil_20) / C * 100 = 35 :=
by
  sorry

end initial_loss_percentage_l973_97302


namespace tangent_points_l973_97376

noncomputable def curve (x : ℝ) : ℝ := x^3 - x - 1

theorem tangent_points (x y : ℝ) (h : y = curve x) (slope_line : ℝ) (h_slope : slope_line = -1/2)
  (tangent_perpendicular : (3 * x^2 - 1) = 2) :
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := sorry

end tangent_points_l973_97376


namespace intersection_point_l973_97336

variable (x y z t : ℝ)

-- Conditions
def line_parametric : Prop := 
  (x = 1 + 2 * t) ∧ 
  (y = 2) ∧ 
  (z = 4 + t)

def plane_equation : Prop :=
  x - 2 * y + 4 * z - 19 = 0

-- Problem statement
theorem intersection_point (h_line: line_parametric x y z t) (h_plane: plane_equation x y z):
  x = 3 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end intersection_point_l973_97336


namespace n_squared_plus_d_not_square_l973_97346

theorem n_squared_plus_d_not_square 
  (n : ℕ) (d : ℕ)
  (h_pos_n : n > 0) 
  (h_pos_d : d > 0) 
  (h_div : d ∣ 2 * n^2) : 
  ¬ ∃ m : ℕ, n^2 + d = m^2 := 
sorry

end n_squared_plus_d_not_square_l973_97346


namespace caps_eaten_correct_l973_97317

def initial_bottle_caps : ℕ := 34
def remaining_bottle_caps : ℕ := 26
def eaten_bottle_caps (k_i k_r : ℕ) : ℕ := k_i - k_r

theorem caps_eaten_correct :
  eaten_bottle_caps initial_bottle_caps remaining_bottle_caps = 8 :=
by
  sorry

end caps_eaten_correct_l973_97317


namespace a_2009_eq_1_a_2014_eq_0_l973_97387

section
variable (a : ℕ → ℕ)
variable (n : ℕ)

-- Condition 1: a_{4n-3} = 1
axiom cond1 : ∀ n : ℕ, a (4 * n - 3) = 1

-- Condition 2: a_{4n-1} = 0
axiom cond2 : ∀ n : ℕ, a (4 * n - 1) = 0

-- Condition 3: a_{2n} = a_n
axiom cond3 : ∀ n : ℕ, a (2 * n) = a n

-- Theorem: a_{2009} = 1
theorem a_2009_eq_1 : a 2009 = 1 := by
  sorry

-- Theorem: a_{2014} = 0
theorem a_2014_eq_0 : a 2014 = 0 := by
  sorry

end

end a_2009_eq_1_a_2014_eq_0_l973_97387


namespace restaurant_june_production_l973_97306

-- Define the given conditions
def daily_hot_dogs := 60
def daily_pizzas := daily_hot_dogs + 40
def june_days := 30
def daily_total := daily_hot_dogs + daily_pizzas
def june_total := daily_total * june_days

-- The goal is to prove that the total number of pizzas and hot dogs made in June is 4800
theorem restaurant_june_production : june_total = 4800 := by
  -- Sorry to skip proof
  sorry

end restaurant_june_production_l973_97306


namespace total_time_to_complete_project_l973_97391

-- Define the initial conditions
def initial_people : ℕ := 6
def initial_days : ℕ := 35
def fraction_completed : ℚ := 1 / 3

-- Define the additional conditions after more people joined
def additional_people : ℕ := initial_people
def total_people : ℕ := initial_people + additional_people
def remaining_fraction : ℚ := 1 - fraction_completed

-- Total time taken to complete the project
theorem total_time_to_complete_project (initial_people initial_days additional_people : ℕ) (fraction_completed remaining_fraction : ℚ)
  (h1 : initial_people * initial_days * fraction_completed = 1/3) 
  (h2 : additional_people = initial_people) 
  (h3 : total_people = initial_people + additional_people)
  (h4 : remaining_fraction = 1 - fraction_completed) : 
  (initial_days + (remaining_fraction / (total_people * (fraction_completed / (initial_people * initial_days)))) = 70) :=
sorry

end total_time_to_complete_project_l973_97391


namespace inequality_solution_l973_97398

noncomputable def solve_inequality (a : ℝ) : Set ℝ :=
  if a = 0 then 
    {x : ℝ | 1 < x}
  else if 0 < a ∧ a < 2 then 
    {x : ℝ | 1 < x ∧ x < (2 / a)}
  else if a = 2 then 
    ∅
  else if a > 2 then 
    {x : ℝ | (2 / a) < x ∧ x < 1}
  else 
    {x : ℝ | x < (2 / a)} ∪ {x : ℝ | 1 < x}

theorem inequality_solution (a : ℝ) :
  ∀ x : ℝ, (ax^2 - (a + 2) * x + 2 < 0) ↔ (x ∈ solve_inequality a) :=
sorry

end inequality_solution_l973_97398


namespace Angie_age_ratio_l973_97324

-- Define Angie's age as a variable
variables (A : ℕ)

-- Give the condition
def Angie_age_condition := A + 4 = 20

-- State the theorem to be proved
theorem Angie_age_ratio (h : Angie_age_condition A) : (A : ℚ) / (A + 4) = 4 / 5 := 
sorry

end Angie_age_ratio_l973_97324


namespace sequence_sum_general_term_l973_97397

theorem sequence_sum_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n) : ∀ n, a n = 2 * n :=
by 
  sorry

end sequence_sum_general_term_l973_97397


namespace intersection_M_N_l973_97307

def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def N : Set ℝ := { y | ∃ x : ℝ, y = x }

theorem intersection_M_N : (M ∩ N) = { y : ℝ | 0 ≤ y } :=
by
  sorry

end intersection_M_N_l973_97307


namespace total_weight_apples_l973_97363

variable (Minjae_weight : ℝ) (Father_weight : ℝ)

theorem total_weight_apples (h1 : Minjae_weight = 2.6) (h2 : Father_weight = 5.98) :
  Minjae_weight + Father_weight = 8.58 :=
by 
  sorry

end total_weight_apples_l973_97363


namespace number_of_ways_to_choose_l973_97332

-- Define the teachers and classes
def teachers : ℕ := 5
def classes : ℕ := 4
def choices (t : ℕ) : ℕ := classes

-- Formalize the problem statement
theorem number_of_ways_to_choose : (choices teachers) ^ teachers = 1024 :=
by
  -- We denote the computation of (4^5)
  sorry

end number_of_ways_to_choose_l973_97332


namespace length_of_crease_l973_97309

/-- 
  Given a rectangular piece of paper 8 inches wide that is folded such that one corner 
  touches the opposite side at an angle θ from the horizontal, and one edge of the paper 
  remains aligned with the base, 
  prove that the length of the crease L is given by L = 8 * tan θ / (1 + tan θ). 
--/
theorem length_of_crease (theta : ℝ) (h : 0 < theta ∧ theta < Real.pi / 2): 
  ∃ L : ℝ, L = 8 * Real.tan theta / (1 + Real.tan theta) :=
sorry

end length_of_crease_l973_97309


namespace initial_shares_bought_l973_97313

variable (x : ℕ) -- x is the number of shares Tom initially bought

-- Conditions:
def initial_cost_per_share : ℕ := 3
def num_shares_sold : ℕ := 10
def selling_price_per_share : ℕ := 4
def doubled_value_per_remaining_share : ℕ := 2 * initial_cost_per_share
def total_profit : ℤ := 40

-- Proving the number of shares initially bought
theorem initial_shares_bought (h : num_shares_sold * selling_price_per_share - x * initial_cost_per_share = total_profit) :
  x = 10 := by sorry

end initial_shares_bought_l973_97313


namespace alicia_tax_deduction_l973_97335

theorem alicia_tax_deduction (earnings_per_hour_in_cents : ℕ) (tax_rate : ℚ) 
  (h1 : earnings_per_hour_in_cents = 2500) (h2 : tax_rate = 0.02) : 
  earnings_per_hour_in_cents * tax_rate = 50 := 
  sorry

end alicia_tax_deduction_l973_97335


namespace given_conditions_l973_97369

theorem given_conditions :
  ∀ (t : ℝ), t > 0 → t ≠ 1 → 
  let x := t^(2/(t-1))
  let y := t^((t+1)/(t-1))
  ¬ ((y * x^(1/y) = x * y^(1/x)) ∨ (y * x^y = x * y^x) ∨ (y^x = x^y) ∨ (x^(x+y) = y^(x+y))) :=
by
  intros t ht_pos ht_ne_1 x_def y_def
  let x := x_def
  let y := y_def
  sorry

end given_conditions_l973_97369


namespace product_of_terms_l973_97378

variable (a : ℕ → ℝ)

-- Conditions: the sequence is geometric, a_1 = 1, a_10 = 3.
axiom geometric_sequence : ∀ n m : ℕ, a n * a m = a 1 * a (n + m - 1)

axiom a_1_eq_one : a 1 = 1
axiom a_10_eq_three : a 10 = 3

-- We need to prove that the product a_2a_3a_4a_5a_6a_7a_8a_9 = 81.
theorem product_of_terms : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end product_of_terms_l973_97378


namespace equal_tuesdays_and_fridays_l973_97330

theorem equal_tuesdays_and_fridays (days_in_month : ℕ) (days_of_week : ℕ) (extra_days : ℕ) (starting_days : Finset ℕ) :
  days_in_month = 30 → days_of_week = 7 → extra_days = 2 →
  starting_days = {0, 3, 6} →
  ∃ n : ℕ, n = 3 :=
by
  sorry

end equal_tuesdays_and_fridays_l973_97330


namespace greatest_b_not_in_range_l973_97348

theorem greatest_b_not_in_range : ∃ b : ℤ, b = 10 ∧ ∀ x : ℝ, x^2 + (b:ℝ) * x + 20 ≠ -7 := sorry

end greatest_b_not_in_range_l973_97348


namespace sheets_in_stack_l973_97326

theorem sheets_in_stack (h : 200 * t = 2.5) (h_pos : t > 0) : (5 / t) = 400 :=
by
  sorry

end sheets_in_stack_l973_97326


namespace average_speed_without_stoppages_l973_97345

variables (d : ℝ) (t : ℝ) (v_no_stop : ℝ)

-- The train stops for 12 minutes per hour
def stoppage_per_hour := 12 / 60
def moving_fraction := 1 - stoppage_per_hour

-- Given speed with stoppages is 160 km/h
def speed_with_stoppage := 160

-- Average speed of the train without stoppages
def speed_without_stoppage := speed_with_stoppage / moving_fraction

-- The average speed without stoppages should equal 200 km/h
theorem average_speed_without_stoppages : speed_without_stoppage = 200 :=
by
  unfold speed_without_stoppage
  unfold moving_fraction
  unfold stoppage_per_hour
  norm_num
  sorry

end average_speed_without_stoppages_l973_97345


namespace evaluate_expression_l973_97301

-- Define the greatest power of 2 and 3 that are factors of 360
def a : ℕ := 3 -- 2^3 is the greatest power of 2 that is a factor of 360
def b : ℕ := 2 -- 3^2 is the greatest power of 3 that is a factor of 360

theorem evaluate_expression : (1 / 4)^(b - a) = 4 := 
by 
  have h1 : a = 3 := rfl
  have h2 : b = 2 := rfl
  rw [h1, h2]
  simp
  sorry

end evaluate_expression_l973_97301


namespace sqrt_43_between_6_and_7_l973_97384

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 :=
by
  sorry

end sqrt_43_between_6_and_7_l973_97384


namespace students_play_neither_sport_l973_97359

def total_students : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def both_players : ℕ := 10

theorem students_play_neither_sport :
  total_students - (hockey_players + basketball_players - both_players) = 4 :=
by
  sorry

end students_play_neither_sport_l973_97359


namespace decision_has_two_exit_paths_l973_97356

-- Define types representing different flowchart symbols
inductive FlowchartSymbol
| Terminal
| InputOutput
| Process
| Decision

-- Define a function that states the number of exit paths given a flowchart symbol
def exit_paths (s : FlowchartSymbol) : Nat :=
  match s with
  | FlowchartSymbol.Terminal   => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process    => 1
  | FlowchartSymbol.Decision   => 2

-- State the theorem that Decision has two exit paths
theorem decision_has_two_exit_paths : exit_paths FlowchartSymbol.Decision = 2 := by
  sorry

end decision_has_two_exit_paths_l973_97356


namespace sequence_count_even_odd_l973_97383

/-- The number of 8-digit sequences such that no two adjacent digits have the same parity
    and the sequence starts with an even number. -/
theorem sequence_count_even_odd : 
  let choices_for_even := 5
  let choices_for_odd := 5
  let total_positions := 8
  (choices_for_even * (choices_for_odd * choices_for_even) ^ (total_positions / 2 - 1)) = 390625 :=
by
  sorry

end sequence_count_even_odd_l973_97383


namespace polynomial_division_l973_97358

variable (x : ℝ)

theorem polynomial_division :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) 
  = (x^2 + 4 * x - 15 + 25 / (x+1)) :=
by sorry

end polynomial_division_l973_97358


namespace functions_are_computable_l973_97311

def f1 : ℕ → ℕ := λ n => 0
def f2 : ℕ → ℕ := λ n => n + 1
def f3 : ℕ → ℕ := λ n => max 0 (n - 1)
def f4 : ℕ → ℕ := λ n => n % 2
def f5 : ℕ → ℕ := λ n => n * 2
def f6 : ℕ × ℕ → ℕ := λ (m, n) => if m ≤ n then 1 else 0

theorem functions_are_computable :
  (Computable f1) ∧
  (Computable f2) ∧
  (Computable f3) ∧
  (Computable f4) ∧
  (Computable f5) ∧
  (Computable f6) := by
  sorry

end functions_are_computable_l973_97311


namespace isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l973_97377

-- Definitions for number of valence electrons
def valence_electrons (atom : String) : ℕ :=
  if atom = "C" then 4
  else if atom = "N" then 5
  else if atom = "O" then 6
  else if atom = "F" then 7
  else if atom = "S" then 6
  else 0

-- Definitions for molecular valence count
def molecule_valence_electrons (molecule : List String) : ℕ :=
  molecule.foldr (λ x acc => acc + valence_electrons x) 0

-- Definitions for specific molecules
def N2_molecule := ["N", "N"]
def CO_molecule := ["C", "O"]
def N2O_molecule := ["N", "N", "O"]
def CO2_molecule := ["C", "O", "O"]
def NO2_minus_molecule := ["N", "O", "O"]
def SO2_molecule := ["S", "O", "O"]
def O3_molecule := ["O", "O", "O"]

-- Isoelectronic property definition
def isoelectronic (mol1 mol2 : List String) : Prop :=
  molecule_valence_electrons mol1 = molecule_valence_electrons mol2

theorem isoelectronic_problem_1_part_1 :
  isoelectronic N2_molecule CO_molecule := sorry

theorem isoelectronic_problem_1_part_2 :
  isoelectronic N2O_molecule CO2_molecule := sorry

theorem isoelectronic_problem_2 :
  isoelectronic NO2_minus_molecule SO2_molecule ∧
  isoelectronic NO2_minus_molecule O3_molecule := sorry

end isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l973_97377


namespace sequence_general_formula_l973_97300

/--
A sequence a_n is defined such that the first term a_1 = 3 and the recursive formula 
a_{n+1} = (3 * a_n - 4) / (a_n - 2).

We aim to prove that the general term of the sequence is given by:
a_n = ( (-2)^(n+2) - 1 ) / ( (-2)^n - 1 )
-/
theorem sequence_general_formula (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 3)
  (hr : ∀ n, a (n + 1) = (3 * a n - 4) / (a n - 2)) :
  a n = ( (-2:ℝ)^(n+2) - 1 ) / ( (-2:ℝ)^n - 1) :=
sorry

end sequence_general_formula_l973_97300


namespace greatest_multiple_of_5_l973_97393

theorem greatest_multiple_of_5 (y : ℕ) (h1 : y > 0) (h2 : y % 5 = 0) (h3 : y^3 < 8000) : y ≤ 15 :=
by {
  sorry
}

end greatest_multiple_of_5_l973_97393


namespace range_sum_of_h_l973_97399

noncomputable def h (x : ℝ) : ℝ := 5 / (5 + 3 * x^2)

theorem range_sum_of_h : 
  (∃ a b : ℝ, (∀ x : ℝ, 0 < h x ∧ h x ≤ 1) ∧ a = 0 ∧ b = 1 ∧ a + b = 1) :=
sorry

end range_sum_of_h_l973_97399


namespace curry_draymond_ratio_l973_97310

theorem curry_draymond_ratio :
  ∃ (curry draymond kelly durant klay : ℕ),
    draymond = 12 ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    curry + draymond + kelly + durant + klay = 69 ∧
    curry = 24 ∧ -- Curry's points calculated in the solution
    draymond = 12 → -- Draymond's points reaffirmed
    curry / draymond = 2 :=
by
  sorry

end curry_draymond_ratio_l973_97310


namespace tap_filling_time_l973_97343

theorem tap_filling_time (T : ℝ) 
  (h_total : (1 / 3) = (1 / T + 1 / 15 + 1 / 6)) : T = 10 := 
sorry

end tap_filling_time_l973_97343


namespace min_max_f_l973_97379

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - Real.cos x

theorem min_max_f :
  (∀ x, 2 * (Real.sin (x / 2))^2 = 1 - Real.cos x) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 5 / 4) :=
by 
  intros h x
  sorry

end min_max_f_l973_97379


namespace cubes_with_one_painted_side_l973_97389

theorem cubes_with_one_painted_side (side_length : ℕ) (one_cm_cubes : ℕ) : 
  side_length = 5 → one_cm_cubes = 54 :=
by 
  intro h 
  sorry

end cubes_with_one_painted_side_l973_97389


namespace sequence_is_geometric_l973_97303

def is_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S n = 3 * a n - 3

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ * r ^ n

theorem sequence_is_geometric (S : ℕ → ℝ) (a : ℕ → ℝ) :
  is_sequence_sum S a →
  (∃ a₁ : ℝ, ∃ r : ℝ, geometric_sequence a r a₁ ∧ a₁ = 3 / 2 ∧ r = 3 / 2) :=
by
  sorry

end sequence_is_geometric_l973_97303


namespace shopkeeper_profit_percentage_l973_97329

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (goods_lost_pct : ℝ)
  (loss_pct : ℝ)
  (remaining_goods : ℝ)
  (selling_price : ℝ)
  (profit_pct : ℝ)
  (h1 : cost_price = 100)
  (h2 : goods_lost_pct = 0.20)
  (h3 : loss_pct = 0.12)
  (h4 : remaining_goods = cost_price * (1 - goods_lost_pct))
  (h5 : selling_price = cost_price * (1 - loss_pct))
  (h6 : profit_pct = ((selling_price - remaining_goods) / remaining_goods) * 100) : 
  profit_pct = 10 := 
sorry

end shopkeeper_profit_percentage_l973_97329


namespace lcm_subtract100_correct_l973_97365

noncomputable def lcm1364_884_subtract_100 : ℕ :=
  let a := 1364
  let b := 884
  let lcm_ab := Nat.lcm a b
  lcm_ab - 100

theorem lcm_subtract100_correct : lcm1364_884_subtract_100 = 1509692 := by
  sorry

end lcm_subtract100_correct_l973_97365


namespace greatest_n_divides_l973_97321

theorem greatest_n_divides (m : ℕ) (hm : 0 < m) : 
  ∃ n : ℕ, (n = m^4 - m^2 + m) ∧ (m^2 + n) ∣ (n^2 + m) := 
by {
  sorry
}

end greatest_n_divides_l973_97321


namespace no_matching_formula_l973_97320

def formula_A (x : ℕ) : ℕ := 4 * x - 2
def formula_B (x : ℕ) : ℕ := x^3 - x^2 + 2 * x
def formula_C (x : ℕ) : ℕ := 2 * x^2
def formula_D (x : ℕ) : ℕ := x^2 + 2 * x + 1

theorem no_matching_formula :
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_A x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_B x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_C x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_D x)
  :=
by
  sorry

end no_matching_formula_l973_97320


namespace cistern_emptying_l973_97342

theorem cistern_emptying (h: (3 / 4) / 12 = 1 / 16) : (8 * (1 / 16) = 1 / 2) :=
by sorry

end cistern_emptying_l973_97342


namespace tomatoes_ready_for_sale_l973_97361

-- Define all conditions
def initial_shipment := 1000 -- kg of tomatoes on Friday
def sold_on_saturday := 300 -- kg of tomatoes sold on Saturday
def rotten_on_sunday := 200 -- kg of tomatoes rotted on Sunday
def additional_shipment := 2 * initial_shipment -- kg of tomatoes arrived on Monday

-- Define the final calculation to prove
theorem tomatoes_ready_for_sale : 
  initial_shipment - sold_on_saturday - rotten_on_sunday + additional_shipment = 2500 := 
by
  sorry

end tomatoes_ready_for_sale_l973_97361


namespace exam_paper_max_marks_l973_97327

/-- A candidate appearing for an examination has to secure 40% marks to pass paper i.
    The candidate secured 40 marks and failed by 20 marks.
    Prove that the maximum mark for paper i is 150. -/
theorem exam_paper_max_marks (p : ℝ) (s f : ℝ) (M : ℝ) (h1 : p = 0.40) (h2 : s = 40) (h3 : f = 20) (h4 : p * M = s + f) :
  M = 150 :=
sorry

end exam_paper_max_marks_l973_97327


namespace find_a_for_odd_function_l973_97344

noncomputable def f (a x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem find_a_for_odd_function (a : ℝ) :
  (∀ x : ℝ, f a x + f a (-x) = 0) ↔ a = -1 := sorry

end find_a_for_odd_function_l973_97344


namespace necessary_not_sufficient_condition_l973_97316

variable (a : ℝ) (D : Set ℝ)

def p : Prop := a ∈ D
def q : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ - a ≤ -3

theorem necessary_not_sufficient_condition (h : p a D → q a) : D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end necessary_not_sufficient_condition_l973_97316


namespace binomial_n_choose_n_sub_2_l973_97351

theorem binomial_n_choose_n_sub_2 (n : ℕ) (h : 2 ≤ n) : Nat.choose n (n - 2) = n * (n - 1) / 2 :=
by
  sorry

end binomial_n_choose_n_sub_2_l973_97351


namespace ben_paints_150_square_feet_l973_97305

-- Define the given conditions
def ratio_allen_ben : ℕ := 3
def ratio_ben_allen : ℕ := 5
def total_work : ℕ := 240

-- Define the total amount of parts
def total_parts : ℕ := ratio_allen_ben + ratio_ben_allen

-- Define the work per part
def work_per_part : ℕ := total_work / total_parts

-- Define the work done by Ben
def ben_parts : ℕ := ratio_ben_allen
def ben_work : ℕ := work_per_part * ben_parts

-- The statement to be proved
theorem ben_paints_150_square_feet : ben_work = 150 :=
by
  sorry

end ben_paints_150_square_feet_l973_97305


namespace race_cars_count_l973_97308

theorem race_cars_count:
  (1 / 7 + 1 / 3 + 1 / 5 = 0.6761904761904762) -> 
  (∀ N : ℕ, (1 / N = 1 / 7 ∨ 1 / N = 1 / 3 ∨ 1 / N = 1 / 5)) -> 
  (1 / 105 = 0.6761904761904762) :=
by
  intro h_sum_probs h_indiv_probs
  sorry

end race_cars_count_l973_97308


namespace hundred_chicken_problem_l973_97352

theorem hundred_chicken_problem :
  ∃ (x y : ℕ), x + y + 81 = 100 ∧ 5 * x + 3 * y + 81 / 3 = 100 := 
by
  sorry

end hundred_chicken_problem_l973_97352


namespace greatest_common_multiple_less_than_bound_l973_97368

-- Define the numbers and the bound
def num1 : ℕ := 15
def num2 : ℕ := 10
def bound : ℕ := 150

-- Define the LCM of num1 and num2
def lcm_num1_num2 : ℕ := Nat.lcm num1 num2

-- Define the greatest multiple of LCM less than bound
def greatest_multiple_less_than_bound (lcm : ℕ) (b : ℕ) : ℕ :=
  (b / lcm) * lcm

-- Main theorem
theorem greatest_common_multiple_less_than_bound :
  greatest_multiple_less_than_bound lcm_num1_num2 bound = 120 :=
by
  sorry

end greatest_common_multiple_less_than_bound_l973_97368


namespace arithmetic_sequence_30th_term_l973_97360

theorem arithmetic_sequence_30th_term :
  let a₁ := 4
  let d₁ := 6
  let n := 30
  (a₁ + (n - 1) * d₁) = 178 :=
by
  sorry

end arithmetic_sequence_30th_term_l973_97360


namespace expected_yield_correct_l973_97318

-- Conditions
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def step_length_ft : ℝ := 2.5
def yield_per_sqft_pounds : ℝ := 0.75

-- Related quantities
def garden_length_ft : ℝ := garden_length_steps * step_length_ft
def garden_width_ft : ℝ := garden_width_steps * step_length_ft
def garden_area_sqft : ℝ := garden_length_ft * garden_width_ft
def expected_yield_pounds : ℝ := garden_area_sqft * yield_per_sqft_pounds

-- Statement to prove
theorem expected_yield_correct : expected_yield_pounds = 2109.375 := by
  sorry

end expected_yield_correct_l973_97318


namespace correct_operation_l973_97319

theorem correct_operation (a : ℝ) : a^5 / a^2 = a^3 := by
  -- Proof steps will be supplied here
  sorry

end correct_operation_l973_97319


namespace expression_value_l973_97388

def a : ℝ := 0.96
def b : ℝ := 0.1

theorem expression_value : (a^3 - (b^3 / a^2) + 0.096 + b^2) = 0.989651 :=
by
  sorry

end expression_value_l973_97388


namespace num_people_in_group_l973_97341

-- Define constants and conditions
def cost_per_set : ℕ := 3  -- $3 to make 4 S'mores
def smores_per_set : ℕ := 4
def total_cost : ℕ := 18   -- $18 total cost
def smores_per_person : ℕ := 3

-- Calculate total S'mores that can be made
def total_sets : ℕ := total_cost / cost_per_set
def total_smores : ℕ := total_sets * smores_per_set

-- Proof problem statement
theorem num_people_in_group : (total_smores / smores_per_person) = 8 :=
by
  sorry

end num_people_in_group_l973_97341


namespace new_average_is_minus_one_l973_97357

noncomputable def new_average_of_deducted_sequence : ℤ :=
  let n := 15
  let avg := 20
  let seq_sum := n * avg
  let x := (seq_sum - (n * (n-1) / 2)) / n
  let deductions := (n-1) * n * 3 / 2
  let new_sum := seq_sum - deductions
  new_sum / n

theorem new_average_is_minus_one : new_average_of_deducted_sequence = -1 := 
  sorry

end new_average_is_minus_one_l973_97357


namespace mod_congruence_zero_iff_l973_97395

theorem mod_congruence_zero_iff
  (a b c d n : ℕ)
  (h1 : a * c ≡ 0 [MOD n])
  (h2 : b * c + a * d ≡ 0 [MOD n]) :
  b * c ≡ 0 [MOD n] ∧ a * d ≡ 0 [MOD n] :=
by
  sorry

end mod_congruence_zero_iff_l973_97395


namespace sufficient_and_necessary_condition_l973_97367

theorem sufficient_and_necessary_condition (x : ℝ) :
  x^2 - 4 * x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 :=
sorry

end sufficient_and_necessary_condition_l973_97367
