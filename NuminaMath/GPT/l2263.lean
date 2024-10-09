import Mathlib

namespace cos_sum_sin_sum_cos_diff_sin_diff_l2263_226317

section

variables (A B : ℝ)

-- Definition of cos and sin of angles
def cos (θ : ℝ) : ℝ := sorry
def sin (θ : ℝ) : ℝ := sorry

-- Cosine of the sum of angles
theorem cos_sum : cos (A + B) = cos A * cos B - sin A * sin B := sorry

-- Sine of the sum of angles
theorem sin_sum : sin (A + B) = sin A * cos B + cos A * sin B := sorry

-- Cosine of the difference of angles
theorem cos_diff : cos (A - B) = cos A * cos B + sin A * sin B := sorry

-- Sine of the difference of angles
theorem sin_diff : sin (A - B) = sin A * cos B - cos A * sin B := sorry

end

end cos_sum_sin_sum_cos_diff_sin_diff_l2263_226317


namespace work_hours_together_l2263_226333

theorem work_hours_together (t : ℚ) :
  (1 / 9) * (9 : ℚ) = 1 ∧ (1 / 12) * (12 : ℚ) = 1 ∧
  (7 / 36) * t + (1 / 9) * (15 / 4) = 1 → t = 3 :=
by
  sorry

end work_hours_together_l2263_226333


namespace find_digits_l2263_226346

theorem find_digits (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
(h2 : 1 ≤ A ∧ A ≤ 9)
(h3 : 1 ≤ B ∧ B ≤ 9)
(h4 : 1 ≤ C ∧ C ≤ 9)
(h5 : 1 ≤ D ∧ D ≤ 9)
(h6 : (10 * A + B) * (10 * C + B) = 111 * D)
(h7 : (10 * A + B) < (10 * C + B)) :
A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 :=
sorry

end find_digits_l2263_226346


namespace fraction_meaningful_iff_x_ne_pm1_l2263_226380

theorem fraction_meaningful_iff_x_ne_pm1 (x : ℝ) : (x^2 - 1 ≠ 0) ↔ (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end fraction_meaningful_iff_x_ne_pm1_l2263_226380


namespace sum_with_extra_five_l2263_226332

theorem sum_with_extra_five 
  (a b c : ℕ)
  (h1 : a + b = 31)
  (h2 : b + c = 48)
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 :=
by
  sorry

end sum_with_extra_five_l2263_226332


namespace solve_for_x_l2263_226363

theorem solve_for_x (x : ℚ) : (2/5 : ℚ) - (1/4 : ℚ) = 1/x → x = 20/3 :=
by
  intro h
  sorry

end solve_for_x_l2263_226363


namespace eliana_total_steps_l2263_226341

noncomputable def day1_steps : ℕ := 200 + 300
noncomputable def day2_steps : ℕ := 2 * day1_steps
noncomputable def day3_steps : ℕ := day1_steps + day2_steps + 100

theorem eliana_total_steps : day3_steps = 1600 := by
  sorry

end eliana_total_steps_l2263_226341


namespace white_cats_count_l2263_226359

theorem white_cats_count (total_cats : ℕ) (black_cats : ℕ) (gray_cats : ℕ) (white_cats : ℕ)
  (h1 : total_cats = 15)
  (h2 : black_cats = 10)
  (h3 : gray_cats = 3)
  (h4 : total_cats = black_cats + gray_cats + white_cats) : 
  white_cats = 2 := 
  by
    -- proof or sorry here
    sorry

end white_cats_count_l2263_226359


namespace centroid_coordinates_satisfy_l2263_226369

noncomputable def P : ℝ × ℝ := (2, 5)
noncomputable def Q : ℝ × ℝ := (-1, 3)
noncomputable def R : ℝ × ℝ := (4, -2)

noncomputable def S : ℝ × ℝ := (
  (P.1 + Q.1 + R.1) / 3,
  (P.2 + Q.2 + R.2) / 3
)

theorem centroid_coordinates_satisfy :
  4 * S.1 + 3 * S.2 = 38 / 3 :=
by
  -- Proof will be added here
  sorry

end centroid_coordinates_satisfy_l2263_226369


namespace problem1_problem2_l2263_226392

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

-- Problem 1
theorem problem1 (α : ℝ) (hα1 : Real.sin α = -1 / 2) (hα2 : Real.cos α = Real.sqrt 3 / 2) :
  f α = -3 := sorry

-- Problem 2
theorem problem2 (h0 : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -2) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -2 := sorry

end problem1_problem2_l2263_226392


namespace michael_total_weight_loss_l2263_226331

def weight_loss_march := 3
def weight_loss_april := 4
def weight_loss_may := 3

theorem michael_total_weight_loss : weight_loss_march + weight_loss_april + weight_loss_may = 10 := by
  sorry

end michael_total_weight_loss_l2263_226331


namespace total_amount_shared_l2263_226371

theorem total_amount_shared (a b c : ℕ) (h_ratio : a = 3 * b / 5 ∧ c = 9 * b / 5) (h_b : b = 50) : a + b + c = 170 :=
by sorry

end total_amount_shared_l2263_226371


namespace product_of_x_values_l2263_226342

noncomputable def find_product_of_x : ℚ :=
  let x1 := -20
  let x2 := -20 / 7
  (x1 * x2)

theorem product_of_x_values :
  (∃ x : ℚ, abs (20 / x + 4) = 3) ->
  find_product_of_x = 400 / 7 :=
by
  sorry

end product_of_x_values_l2263_226342


namespace shaded_triangle_area_l2263_226313

-- Definitions and conditions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

def larger_triangle_base : ℕ := grid_width
def larger_triangle_height : ℕ := grid_height - 1

def smaller_triangle_base : ℕ := 12
def smaller_triangle_height : ℕ := 3

-- The proof problem stating that the area of the smaller shaded triangle is 18 units
theorem shaded_triangle_area :
  (smaller_triangle_base * smaller_triangle_height) / 2 = 18 :=
by
  sorry

end shaded_triangle_area_l2263_226313


namespace first_candidate_votes_percentage_l2263_226375

theorem first_candidate_votes_percentage 
( total_votes : ℕ ) 
( second_candidate_votes : ℕ ) 
( P : ℕ ) 
( h1 : total_votes = 2400 ) 
( h2 : second_candidate_votes = 480 ) 
( h3 : (P/100 : ℝ) * total_votes + second_candidate_votes = total_votes ) : 
  P = 80 := 
sorry

end first_candidate_votes_percentage_l2263_226375


namespace proportion_of_solution_x_in_mixture_l2263_226384

-- Definitions for the conditions in given problem
def solution_x_contains_perc_a : ℚ := 0.20
def solution_y_contains_perc_a : ℚ := 0.30
def solution_z_contains_perc_a : ℚ := 0.40

def solution_y_to_z_ratio : ℚ := 3 / 2
def final_mixture_perc_a : ℚ := 0.25

-- Proving the proportion of solution x in the mixture equals 9/14
theorem proportion_of_solution_x_in_mixture
  (x y z : ℚ) (k : ℚ) (hx : x = 9 * k) (hy : y = 3 * k) (hz : z = 2 * k) :
  solution_x_contains_perc_a * x + solution_y_contains_perc_a * y + solution_z_contains_perc_a * z
  = final_mixture_perc_a * (x + y + z) →
  x / (x + y + z) = 9 / 14 :=
by
  intros h
  -- leaving the proof as a placeholder
  sorry

end proportion_of_solution_x_in_mixture_l2263_226384


namespace green_peaches_are_six_l2263_226379

/-- There are 5 red peaches in the basket. -/
def red_peaches : ℕ := 5

/-- There are 14 yellow peaches in the basket. -/
def yellow_peaches : ℕ := 14

/-- There are total of 20 green and yellow peaches in the basket. -/
def green_and_yellow_peaches : ℕ := 20

/-- The number of green peaches is calculated as the difference between the total number of green and yellow peaches and the number of yellow peaches. -/
theorem green_peaches_are_six :
  (green_and_yellow_peaches - yellow_peaches) = 6 :=
by
  sorry

end green_peaches_are_six_l2263_226379


namespace problem1_problem2_l2263_226345

theorem problem1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := 
by sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := 
by sorry

end problem1_problem2_l2263_226345


namespace find_x_l2263_226361

variables (x : ℝ)

theorem find_x : (x / 4) * 12 = 9 → x = 3 :=
by
  sorry

end find_x_l2263_226361


namespace total_students_in_circle_l2263_226337

theorem total_students_in_circle (N : ℕ) (h1 : ∃ (students : Finset ℕ), students.card = N)
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ b - a = N / 2): N = 18 :=
by
  sorry

end total_students_in_circle_l2263_226337


namespace valid_common_ratios_count_l2263_226348

noncomputable def num_valid_common_ratios (a₁ : ℝ) (q : ℝ) : ℝ :=
  let a₅ := a₁ * q^4
  let a₃ := a₁ * q^2
  if 2 * a₅ = 4 * a₁ + (-2) * a₃ then 1 else 0

theorem valid_common_ratios_count (a₁ : ℝ) : 
  (num_valid_common_ratios a₁ 1) + (num_valid_common_ratios a₁ (-1)) = 2 :=
by sorry

end valid_common_ratios_count_l2263_226348


namespace oranges_per_box_calculation_l2263_226339

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end oranges_per_box_calculation_l2263_226339


namespace time_after_3108_hours_l2263_226387

/-- The current time is 3 o'clock. On a 12-hour clock, 
 what time will it be 3108 hours from now? -/
theorem time_after_3108_hours : (3 + 3108) % 12 = 3 := 
by
  sorry

end time_after_3108_hours_l2263_226387


namespace time_taken_l2263_226350

-- Define the function T which takes the number of cats, the number of rats, and returns the time in minutes
def T (n m : ℕ) : ℕ := if n = m then 4 else sorry

-- The theorem states that, given n cats and n rats, the time taken is 4 minutes
theorem time_taken (n : ℕ) : T n n = 4 :=
by simp [T]

end time_taken_l2263_226350


namespace only_negative_integer_among_list_l2263_226354

namespace NegativeIntegerProblem

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem only_negative_integer_among_list :
  (∃ x, x ∈ [0, -1, 2, -1.5] ∧ (x < 0) ∧ is_integer x) ↔ (x = -1) :=
by
  sorry

end NegativeIntegerProblem

end only_negative_integer_among_list_l2263_226354


namespace proof_problem_l2263_226357

variable {R : Type*} [LinearOrderedField R]

theorem proof_problem 
  (a1 a2 a3 b1 b2 b3 : R)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : b1 < b2) (h4 : b2 < b3)
  (h_sum : a1 + a2 + a3 = b1 + b2 + b3)
  (h_pair_sum : a1 * a2 + a1 * a3 + a2 * a3 = b1 * b2 + b1 * b3 + b2 * b3)
  (h_a1_lt_b1 : a1 < b1) :
  (b2 < a2) ∧ (a3 < b3) ∧ (a1 * a2 * a3 < b1 * b2 * b3) ∧ ((1 - a1) * (1 - a2) * (1 - a3) > (1 - b1) * (1 - b2) * (1 - b3)) :=
by {
  sorry
}

end proof_problem_l2263_226357


namespace range_of_ab_l2263_226304

def circle_eq (x y : ℝ) := x^2 + y^2 + 2 * x - 4 * y + 1 = 0
def line_eq (a b x y : ℝ) := 2 * a * x - b * y + 2 = 0

theorem range_of_ab (a b : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq a b x y) ∧ (∃ x y : ℝ, x = -1 ∧ y = 2) →
  ab <= 1/4 := 
by
  sorry

end range_of_ab_l2263_226304


namespace pizza_payment_l2263_226397

theorem pizza_payment (n : ℕ) (cost : ℕ) (total : ℕ) 
  (h1 : n = 3) 
  (h2 : cost = 8) 
  (h3 : total = n * cost) : 
  total = 24 :=
by 
  rw [h1, h2] at h3 
  exact h3

end pizza_payment_l2263_226397


namespace share_of_y_l2263_226356

theorem share_of_y (A y z : ℝ)
  (hx : y = 0.45 * A)
  (hz : z = 0.30 * A)
  (h_total : A + y + z = 140) :
  y = 36 := by
  sorry

end share_of_y_l2263_226356


namespace number_of_players_in_association_l2263_226383

-- Define the variables and conditions based on the given problem
def socks_cost : ℕ := 6
def tshirt_cost := socks_cost + 8
def hat_cost := tshirt_cost - 3
def total_expenditure : ℕ := 4950
def cost_per_player := 2 * (socks_cost + tshirt_cost + hat_cost)

-- The statement to prove
theorem number_of_players_in_association :
  total_expenditure / cost_per_player = 80 := by
  sorry

end number_of_players_in_association_l2263_226383


namespace time_to_fill_pool_l2263_226396

theorem time_to_fill_pool :
  let R1 := 1
  let R2 := 1 / 2
  let R3 := 1 / 3
  let R4 := 1 / 4
  let R_total := R1 + R2 + R3 + R4
  let T := 1 / R_total
  T = 12 / 25 := 
by
  sorry

end time_to_fill_pool_l2263_226396


namespace nate_reading_percentage_l2263_226378

-- Given conditions
def total_pages := 400
def pages_to_read := 320

-- Calculate the number of pages he has already read
def pages_read := total_pages - pages_to_read

-- Prove the percentage of the book Nate has finished reading
theorem nate_reading_percentage : (pages_read / total_pages) * 100 = 20 := by
  sorry

end nate_reading_percentage_l2263_226378


namespace second_number_l2263_226344

theorem second_number (A B : ℝ) (h1 : 0.50 * A = 0.40 * B + 180) (h2 : A = 456) : B = 120 := 
by
  sorry

end second_number_l2263_226344


namespace major_axis_length_l2263_226309

theorem major_axis_length (radius : ℝ) (k : ℝ) (minor_axis : ℝ) (major_axis : ℝ)
  (cyl_radius : radius = 2)
  (minor_eq_diameter : minor_axis = 2 * radius)
  (major_longer : major_axis = minor_axis * (1 + k))
  (k_value : k = 0.25) :
  major_axis = 5 :=
by
  -- Proof omitted, using sorry
  sorry

end major_axis_length_l2263_226309


namespace decimal_arithmetic_l2263_226389

theorem decimal_arithmetic : 0.45 - 0.03 + 0.008 = 0.428 := by
  sorry

end decimal_arithmetic_l2263_226389


namespace plot_length_l2263_226325

theorem plot_length (b : ℝ) (cost_per_meter cost_total : ℝ)
  (h1 : cost_per_meter = 26.5) 
  (h2 : cost_total = 5300) 
  (h3 : (2 * (b + (b + 20)) * cost_per_meter) = cost_total) : 
  b + 20 = 60 := 
by 
  -- Proof here
  sorry

end plot_length_l2263_226325


namespace hyperbola_min_sum_dist_l2263_226310

open Real

theorem hyperbola_min_sum_dist (x y : ℝ) (F1 F2 A B : ℝ × ℝ) :
  -- Conditions for the hyperbola and the foci
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 6 = 1) →
  F1 = (-c, 0) →
  F2 = (c, 0) →
  -- Minimum value of |AF2| + |BF2|
  ∃ (l : ℝ × ℝ → Prop), l F1 ∧ (∃ A B, l A ∧ l B ∧ A = (-3, y_A) ∧ B = (-3, y_B) ) →
  |dist A F2| + |dist B F2| = 16 :=
by
  sorry

end hyperbola_min_sum_dist_l2263_226310


namespace rectangular_sheet_integers_l2263_226351

noncomputable def at_least_one_integer (a b : ℝ) : Prop :=
  ∃ i : ℤ, a = i ∨ b = i

theorem rectangular_sheet_integers (a b : ℝ)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_cut_lines : ∀ x y : ℝ, (∃ k : ℤ, x = k ∧ y = 1 ∨ y = k ∧ x = 1) → (∃ z : ℤ, x = z ∨ y = z)) :
  at_least_one_integer a b :=
sorry

end rectangular_sheet_integers_l2263_226351


namespace Teresa_age_when_Michiko_born_l2263_226329

theorem Teresa_age_when_Michiko_born 
  (Teresa_age : ℕ) (Morio_age : ℕ) (Michiko_born_age : ℕ) (Kenji_diff : ℕ)
  (Emiko_diff : ℕ) (Hideki_same_as_Kenji : Prop) (Ryuji_age_same_as_Morio : Prop)
  (h1 : Teresa_age = 59) 
  (h2 : Morio_age = 71) 
  (h3 : Morio_age = Michiko_born_age + 33)
  (h4 : Kenji_diff = 4)
  (h5 : Emiko_diff = 10)
  (h6 : Hideki_same_as_Kenji = True)
  (h7 : Ryuji_age_same_as_Morio = True) : 
  ∃ Michiko_age Hideki_age Michiko_Hideki_diff Teresa_birth_age,
    Michiko_age = 33 ∧ 
    Hideki_age = 29 ∧ 
    Michiko_Hideki_diff = 4 ∧ 
    Teresa_birth_age = 26 :=
sorry

end Teresa_age_when_Michiko_born_l2263_226329


namespace moles_of_H2O_formed_l2263_226366

def NH4NO3 (n : ℕ) : Prop := n = 1
def NaOH (n : ℕ) : Prop := ∃ m : ℕ, m = n
def H2O (n : ℕ) : Prop := n = 1

theorem moles_of_H2O_formed :
  ∀ (n : ℕ), NH4NO3 1 → NaOH n → H2O 1 := 
by
  intros n hNH4NO3 hNaOH
  exact sorry

end moles_of_H2O_formed_l2263_226366


namespace total_cards_in_box_l2263_226362

-- Definitions based on conditions
def xiaoMingCountsFaster (m h : ℕ) := 6 * h = 4 * m
def xiaoHuaForgets (h1 h2 : ℕ) := h1 + h2 = 112
def finalCardLeft (t : ℕ) := t - 1 = 112

-- Main theorem stating that the total number of cards is 353
theorem total_cards_in_box : ∃ N : ℕ, 
    (∃ m h1 h2 : ℕ,
        xiaoMingCountsFaster m h1 ∧
        xiaoHuaForgets h1 h2 ∧
        finalCardLeft N) ∧
    N = 353 :=
sorry

end total_cards_in_box_l2263_226362


namespace messenger_speed_l2263_226311

noncomputable def team_length : ℝ := 6

noncomputable def team_speed : ℝ := 5

noncomputable def total_time : ℝ := 0.5

theorem messenger_speed (x : ℝ) :
  (6 / (x + team_speed) + 6 / (x - team_speed) = total_time) →
  x = 25 := by
  sorry

end messenger_speed_l2263_226311


namespace division_of_exponents_l2263_226300

-- Define the conditions as constants and statements that we are concerned with
variables (x : ℝ)

-- The Lean 4 statement of the equivalent proof problem
theorem division_of_exponents (h₁ : x ≠ 0) : x^8 / x^2 = x^6 := 
sorry

end division_of_exponents_l2263_226300


namespace elements_map_to_4_l2263_226301

def f (x : ℝ) : ℝ := x^2

theorem elements_map_to_4 :
  { x : ℝ | f x = 4 } = {2, -2} :=
by
  sorry

end elements_map_to_4_l2263_226301


namespace intersection_eq_l2263_226315

noncomputable def U : Set ℝ := Set.univ
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Complement of B in U
def complement_B : Set ℝ := {x | x < 2 ∨ x ≥ 3}

-- Intersection of A and complement of B
def intersection : Set ℕ := {x ∈ A | ↑x < 2 ∨ ↑x ≥ 3}

theorem intersection_eq : intersection = {1, 3, 4} :=
by
  sorry

end intersection_eq_l2263_226315


namespace double_espresso_cost_l2263_226377

-- Define the cost of coffee, days, and total spent as constants
def iced_coffee : ℝ := 2.5
def total_days : ℝ := 20
def total_spent : ℝ := 110

-- Define the cost of double espresso as variable E
variable (E : ℝ)

-- The proposition to prove
theorem double_espresso_cost : (total_days * (E + iced_coffee) = total_spent) → (E = 3) :=
by
  sorry

end double_espresso_cost_l2263_226377


namespace sector_angle_sector_max_area_l2263_226352

-- Part (1)
theorem sector_angle (r l : ℝ) (α : ℝ) :
  2 * r + l = 10 → (1 / 2) * l * r = 4 → α = l / r → α = 1 / 2 :=
by
  intro h1 h2 h3
  sorry

-- Part (2)
theorem sector_max_area (r l : ℝ) (α S : ℝ) :
  2 * r + l = 40 → α = l / r → S = (1 / 2) * l * r →
  (∀ r' l' α' S', 2 * r' + l' = 40 → α' = l' / r' → S' = (1 / 2) * l' * r' → S ≤ S') →
  r = 10 ∧ α = 2 ∧ S = 100 :=
by
  intro h1 h2 h3 h4
  sorry

end sector_angle_sector_max_area_l2263_226352


namespace negation_of_prop_l2263_226386

theorem negation_of_prop (P : Prop) :
  (¬ ∀ x > 0, x - 1 ≥ Real.log x) ↔ ∃ x > 0, x - 1 < Real.log x :=
by
  sorry

end negation_of_prop_l2263_226386


namespace number_of_terms_in_sequence_l2263_226312

theorem number_of_terms_in_sequence : ∃ n : ℕ, 6 + (n-1) * 4 = 154 ∧ n = 38 :=
by
  sorry

end number_of_terms_in_sequence_l2263_226312


namespace natalie_height_l2263_226322

variable (height_Natalie height_Harpreet height_Jiayin : ℝ)
variable (h1 : height_Natalie = height_Harpreet)
variable (h2 : height_Jiayin = 161)
variable (h3 : (height_Natalie + height_Harpreet + height_Jiayin) / 3 = 171)

theorem natalie_height : height_Natalie = 176 :=
by 
  sorry

end natalie_height_l2263_226322


namespace determinant_of_trig_matrix_l2263_226347

theorem determinant_of_trig_matrix (α β : ℝ) : 
  Matrix.det ![
    ![Real.sin α, Real.cos α], 
    ![Real.cos β, Real.sin β]
  ] = -Real.cos (α - β) :=
by sorry

end determinant_of_trig_matrix_l2263_226347


namespace symmetric_point_coordinates_l2263_226368

theorem symmetric_point_coordinates (M : ℝ × ℝ) (N : ℝ × ℝ) (hM : M = (1, -2)) (h_sym : N = (-M.1, -M.2)) :
  N = (-1, 2) :=
by sorry

end symmetric_point_coordinates_l2263_226368


namespace tank_full_volume_l2263_226318

theorem tank_full_volume (x : ℝ) (h1 : 5 / 6 * x > 0) (h2 : 5 / 6 * x - 15 = 1 / 3 * x) : x = 30 :=
by
  -- The proof is omitted as per the requirement.
  sorry

end tank_full_volume_l2263_226318


namespace semicircle_area_l2263_226330

theorem semicircle_area (x : ℝ) (y : ℝ) (r : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : x^2 + y^2 = (2*r)^2) :
  (1/2) * π * r^2 = (13 * π) / 8 :=
by
  sorry

end semicircle_area_l2263_226330


namespace factorizations_of_2079_l2263_226334

theorem factorizations_of_2079 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 2079 ∧ (a, b) = (21, 99) ∨ (a, b) = (33, 63) :=
sorry

end factorizations_of_2079_l2263_226334


namespace cars_meet_first_time_l2263_226394

-- Definitions based on conditions
def car (t : ℕ) (v : ℕ) : ℕ := t * v
def car_meet (t : ℕ) (v1 v2 : ℕ) : Prop := ∃ n, v1 * t + v2 * t = n

-- Given conditions
variables (v_A v_B v_C v_D : ℕ) (pairwise_different : v_A ≠ v_B ∧ v_B ≠ v_C ∧ v_C ≠ v_D ∧ v_D ≠ v_A)
variables (t1 t2 t3 : ℕ) (time_AC : t1 = 7) (time_BD : t1 = 7) (time_AB : t2 = 53)
variables (condition1 : car_meet t1 v_A v_C) (condition2 : car_meet t1 v_B v_D)
variables (condition3 : ∃ k, (v_A - v_B) * t2 = k)

-- Theorem statement
theorem cars_meet_first_time : ∃ t, (t = 371) := sorry

end cars_meet_first_time_l2263_226394


namespace round_trip_and_car_percent_single_trip_and_motorcycle_percent_l2263_226381

noncomputable def totalPassengers := 100
noncomputable def roundTripPercent := 35
noncomputable def singleTripPercent := 100 - roundTripPercent

noncomputable def roundTripCarPercent := 40
noncomputable def roundTripMotorcyclePercent := 15
noncomputable def roundTripNoVehiclePercent := 60

noncomputable def singleTripCarPercent := 25
noncomputable def singleTripMotorcyclePercent := 10
noncomputable def singleTripNoVehiclePercent := 45

theorem round_trip_and_car_percent : 
  ((roundTripCarPercent / 100) * (roundTripPercent / 100) * totalPassengers) = 14 :=
by
  sorry

theorem single_trip_and_motorcycle_percent :
  ((singleTripMotorcyclePercent / 100) * (singleTripPercent / 100) * totalPassengers) = 6 :=
by
  sorry

end round_trip_and_car_percent_single_trip_and_motorcycle_percent_l2263_226381


namespace maximum_value_m_l2263_226365

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

noncomputable def exists_t_and_max_m (m : ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ x

theorem maximum_value_m : ∃ m : ℝ, exists_t_and_max_m m ∧ (∀ m' : ℝ, exists_t_and_max_m m' → m' ≤ 4) :=
by
  sorry

end maximum_value_m_l2263_226365


namespace marble_ratio_l2263_226373

theorem marble_ratio (A V X : ℕ) 
  (h1 : A + 5 = V - 5)
  (h2 : V + X = (A - X) + 30) : X / 5 = 2 :=
by
  sorry

end marble_ratio_l2263_226373


namespace original_price_l2263_226388

theorem original_price (p q: ℝ) (h₁ : p ≠ 0) (h₂ : q ≠ 0) : 
  let x := 20000 / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  (x : ℝ) * (1 - p^2 / 10000) * (1 - q^2 / 10000) = 2 :=
by
  sorry

end original_price_l2263_226388


namespace find_real_roots_of_PQ_l2263_226323

noncomputable def P (x b : ℝ) : ℝ := x^2 + x / 2 + b
noncomputable def Q (x c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_real_roots_of_PQ (b c d : ℝ)
  (h: ∀ x : ℝ, P x b * Q x c d = Q (P x b) c d)
  (h_d_zero: d = 0) :
  ∃ x : ℝ, P (Q x c d) b = 0 → x = (-c + Real.sqrt (c^2 + 2)) / 2 ∨ x = (-c - Real.sqrt (c^2 + 2)) / 2 :=
by
  sorry

end find_real_roots_of_PQ_l2263_226323


namespace members_in_third_shift_l2263_226355

-- Defining the given conditions
def total_first_shift : ℕ := 60
def percent_first_shift_pension : ℝ := 0.20

def total_second_shift : ℕ := 50
def percent_second_shift_pension : ℝ := 0.40

variable (T : ℕ)
def percent_third_shift_pension : ℝ := 0.10

def percent_total_pension_program : ℝ := 0.24

noncomputable def number_of_members_third_shift : ℕ :=
  T

-- Using the conditions to declare the theorem
theorem members_in_third_shift :
  ((60 * 0.20) + (50 * 0.40) + (number_of_members_third_shift T * percent_third_shift_pension)) / (60 + 50 + number_of_members_third_shift T) = percent_total_pension_program →
  number_of_members_third_shift T = 40 :=
sorry

end members_in_third_shift_l2263_226355


namespace eq1_eq2_eq3_eq4_l2263_226382

theorem eq1 : ∀ x : ℝ, x = 6 → 3 * x - 8 = x + 4 := by
  intros x hx
  rw [hx]
  sorry

theorem eq2 : ∀ x : ℝ, x = -2 → 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) := by
  intros x hx
  rw [hx]
  sorry

theorem eq3 : ∀ x : ℝ, x = -20 → (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 := by
  intros x hx
  rw [hx]
  sorry

theorem eq4 : ∀ y : ℝ, y = -1 → (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  intros y hy
  rw [hy]
  sorry

end eq1_eq2_eq3_eq4_l2263_226382


namespace aldehyde_formula_l2263_226336

-- Define the problem starting with necessary variables
variables (n : ℕ)

-- Given conditions
def general_formula_aldehyde (n : ℕ) : String :=
  "CₙH_{2n}O"

def mass_percent_hydrogen (n : ℕ) : ℚ :=
  (2 * n) / (14 * n + 16)

-- Given the percentage of hydrogen in the aldehyde
def given_hydrogen_percent : ℚ := 0.12

-- The main theorem
theorem aldehyde_formula :
  (exists n : ℕ, mass_percent_hydrogen n = given_hydrogen_percent ∧ n = 6) ->
  general_formula_aldehyde 6 = "C₆H_{12}O" :=
by
  sorry

end aldehyde_formula_l2263_226336


namespace eq_sum_of_factorial_fractions_l2263_226376

theorem eq_sum_of_factorial_fractions (b2 b3 b5 b6 b7 b8 : ℤ)
  (h2 : 0 ≤ b2 ∧ b2 < 2)
  (h3 : 0 ≤ b3 ∧ b3 < 3)
  (h5 : 0 ≤ b5 ∧ b5 < 5)
  (h6 : 0 ≤ b6 ∧ b6 < 6)
  (h7 : 0 ≤ b7 ∧ b7 < 7)
  (h8 : 0 ≤ b8 ∧ b8 < 8)
  (h_eq : (3 / 8 : ℚ) = (b2 / (2 * 1) + b3 / (3 * 2 * 1) + b5 / (5 * 4 * 3 * 2 * 1) +
                          b6 / (6 * 5 * 4 * 3 * 2 * 1) + b7 / (7 * 6 * 5 * 4 * 3 * 2 * 1) +
                          b8 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) : ℚ)) :
  b2 + b3 + b5 + b6 + b7 + b8 = 12 :=
by
  sorry

end eq_sum_of_factorial_fractions_l2263_226376


namespace fruit_basket_combinations_l2263_226319

theorem fruit_basket_combinations (apples oranges : ℕ) (ha : apples = 6) (ho : oranges = 12) : 
  (∃ (baskets : ℕ), 
    (∀ a, 1 ≤ a ∧ a ≤ apples → ∃ b, 2 ≤ b ∧ b ≤ oranges ∧ baskets = a * b) ∧ baskets = 66) :=
by {
  sorry
}

end fruit_basket_combinations_l2263_226319


namespace inequality_reciprocal_l2263_226302

theorem inequality_reciprocal (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_l2263_226302


namespace ratio_of_voters_l2263_226372

open Real

theorem ratio_of_voters (X Y : ℝ) (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) : X / Y = 2 :=
by
  sorry

end ratio_of_voters_l2263_226372


namespace six_inch_cube_value_is_2700_l2263_226328

noncomputable def value_of_six_inch_cube (value_four_inch_cube : ℕ) : ℕ :=
  let volume_four_inch_cube := 4^3
  let volume_six_inch_cube := 6^3
  let scaling_factor := volume_six_inch_cube / volume_four_inch_cube
  value_four_inch_cube * scaling_factor

theorem six_inch_cube_value_is_2700 : value_of_six_inch_cube 800 = 2700 := by
  sorry

end six_inch_cube_value_is_2700_l2263_226328


namespace total_wage_is_75_l2263_226338

noncomputable def wages_total (man_wage : ℕ) : ℕ :=
  let men := 5
  let women := (5 : ℕ)
  let boys := 8
  (man_wage * men) + (man_wage * men) + (man_wage * men)

theorem total_wage_is_75
  (W : ℕ)
  (man_wage : ℕ := 5)
  (h1 : 5 = W) 
  (h2 : W = 8) 
  : wages_total man_wage = 75 := by
  sorry

end total_wage_is_75_l2263_226338


namespace socks_selection_l2263_226343

theorem socks_selection :
  (Nat.choose 7 3) - (Nat.choose 6 3) = 15 :=
by sorry

end socks_selection_l2263_226343


namespace adam_lessons_on_monday_l2263_226340

theorem adam_lessons_on_monday :
  (∃ (time_monday time_tuesday time_wednesday : ℝ) (n_monday_lessons : ℕ),
    time_tuesday = 3 ∧
    time_wednesday = 2 * time_tuesday ∧
    time_monday + time_tuesday + time_wednesday = 12 ∧
    n_monday_lessons = time_monday / 0.5 ∧
    n_monday_lessons = 6) :=
by
  sorry

end adam_lessons_on_monday_l2263_226340


namespace coupon_calculation_l2263_226398

theorem coupon_calculation :
  let initial_stock : ℝ := 40.0
  let sold_books : ℝ := 20.0
  let coupons_per_book : ℝ := 4.0
  let remaining_books := initial_stock - sold_books
  let total_coupons := remaining_books * coupons_per_book
  total_coupons = 80.0 :=
by
  sorry

end coupon_calculation_l2263_226398


namespace aluminum_iodide_mass_produced_l2263_226353

theorem aluminum_iodide_mass_produced
  (mass_Al : ℝ) -- the mass of Aluminum used
  (molar_mass_Al : ℝ) -- molar mass of Aluminum
  (molar_mass_AlI3 : ℝ) -- molar mass of Aluminum Iodide
  (reaction_eq : ∀ (moles_Al : ℝ) (moles_AlI3 : ℝ), 2 * moles_Al = 2 * moles_AlI3) -- reaction equation which indicates a 1:1 molar ratio
  (mass_Al_value : mass_Al = 25.0) 
  (molar_mass_Al_value : molar_mass_Al = 26.98) 
  (molar_mass_AlI3_value : molar_mass_AlI3 = 407.68) :
  ∃ mass_AlI3 : ℝ, mass_AlI3 = 377.52 := by
  sorry

end aluminum_iodide_mass_produced_l2263_226353


namespace boat_stream_ratio_l2263_226327

-- Conditions: A man takes twice as long to row a distance against the stream as to row the same distance in favor of the stream.
theorem boat_stream_ratio (B S : ℝ) (h : ∀ (d : ℝ), d / (B - S) = 2 * (d / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l2263_226327


namespace fraction_transform_l2263_226326

theorem fraction_transform (x : ℕ) (h : 9 * (537 - x) = 463 + x) : x = 437 :=
by
  sorry

end fraction_transform_l2263_226326


namespace apples_to_pears_value_l2263_226399

/-- Suppose 1/2 of 12 apples are worth as much as 10 pears. -/
def apples_per_pears_ratio : ℚ := 10 / (1 / 2 * 12)

/-- Prove that 3/4 of 6 apples are worth as much as 7.5 pears. -/
theorem apples_to_pears_value : (3 / 4 * 6) * apples_per_pears_ratio = 7.5 := 
by
  sorry

end apples_to_pears_value_l2263_226399


namespace worker_savings_multiple_l2263_226306

variable (P : ℝ)

theorem worker_savings_multiple (h1 : P > 0) (h2 : 0.4 * P + 0.6 * P = P) : 
  (12 * 0.4 * P) / (0.6 * P) = 8 :=
by
  sorry

end worker_savings_multiple_l2263_226306


namespace quadratic_inequality_sufficient_necessary_l2263_226391

theorem quadratic_inequality_sufficient_necessary (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ 0 < a ∧ a < 4 :=
by
  -- proof skipped
  sorry

end quadratic_inequality_sufficient_necessary_l2263_226391


namespace deepak_wife_speed_l2263_226367

-- Definitions and conditions
def track_circumference_km : ℝ := 0.66
def deepak_speed_kmh : ℝ := 4.5
def time_to_meet_hr : ℝ := 0.08

-- Theorem statement
theorem deepak_wife_speed
  (track_circumference_km : ℝ)
  (deepak_speed_kmh : ℝ)
  (time_to_meet_hr : ℝ)
  (deepak_distance : ℝ := deepak_speed_kmh * time_to_meet_hr)
  (wife_distance : ℝ := track_circumference_km - deepak_distance)
  (wife_speed_kmh : ℝ := wife_distance / time_to_meet_hr) : 
  wife_speed_kmh = 3.75 :=
sorry

end deepak_wife_speed_l2263_226367


namespace decreasing_intervals_l2263_226307

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

theorem decreasing_intervals : 
  (∀ x y : ℝ, x < y → ((y < -1 ∨ x > -1) → f y < f x)) ∧
  (∀ x y : ℝ, x < y → (y ≥ -1 ∧ x ≤ -1 → f y < f x)) :=
by 
  intros;
  sorry

end decreasing_intervals_l2263_226307


namespace ivan_years_l2263_226349

theorem ivan_years (years months weeks days hours : ℕ) (h1 : years = 48) (h2 : months = 48)
    (h3 : weeks = 48) (h4 : days = 48) (h5 : hours = 48) :
    (53 : ℕ) = (years + (months / 12) + ((weeks * 7 + days) / 365) + ((hours / 24) / 365)) := by
  sorry

end ivan_years_l2263_226349


namespace train_length_l2263_226335

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_seconds : ℝ := 36
noncomputable def speed_m_s := speed_km_hr * (5/18 : ℝ)
noncomputable def distance := speed_m_s * time_seconds

-- Theorem statement
theorem train_length : distance = 600.12 := by
  sorry

end train_length_l2263_226335


namespace roots_of_poly_l2263_226374

noncomputable def poly (x : ℝ) : ℝ := x^3 - 4 * x^2 - x + 4

theorem roots_of_poly :
  (poly 1 = 0) ∧ (poly (-1) = 0) ∧ (poly 4 = 0) ∧
  (∀ x, poly x = 0 → x = 1 ∨ x = -1 ∨ x = 4) :=
by
  sorry

end roots_of_poly_l2263_226374


namespace real_roots_determinant_l2263_226385

variable (a b c k : ℝ)
variable (k_pos : k > 0)
variable (a_nonzero : a ≠ 0) 
variable (b_nonzero : b ≠ 0)
variable (c_nonzero : c ≠ 0)
variable (k_nonzero : k ≠ 0)

theorem real_roots_determinant : 
  ∃! x : ℝ, (Matrix.det ![![x, k * c, -k * b], ![-k * c, x, k * a], ![k * b, -k * a, x]] = 0) :=
sorry

end real_roots_determinant_l2263_226385


namespace range_of_a_l2263_226303

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 2) * x + 5

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 4 → f x a ≤ f (x+1) a) : a ≥ -2 := 
by
  sorry

end range_of_a_l2263_226303


namespace strawberries_weight_before_l2263_226305

variables (M D E B : ℝ)

noncomputable def total_weight_before (M D E : ℝ) := M + D - E

theorem strawberries_weight_before :
  ∀ (M D E : ℝ), M = 36 ∧ D = 16 ∧ E = 30 → total_weight_before M D E = 22 :=
by
  intros M D E h
  simp [total_weight_before, h]
  sorry

end strawberries_weight_before_l2263_226305


namespace bounded_expression_l2263_226395

theorem bounded_expression (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 :=
sorry

end bounded_expression_l2263_226395


namespace supplement_complement_l2263_226390

theorem supplement_complement (angle1 angle2 : ℝ) 
  (h_complementary : angle1 + angle2 = 90) : 
   180 - angle1 = 90 + angle2 := by
  sorry

end supplement_complement_l2263_226390


namespace arithmetic_sequence_integers_l2263_226321

theorem arithmetic_sequence_integers (a3 a18 : ℝ) (d : ℝ) (n : ℕ)
  (h3 : a3 = 14) (h18 : a18 = 23) (hd : d = 0.6)
  (hn : n = 2010) : 
  (∃ (k : ℕ), n = 5 * (k + 1) - 2) ∧ (k ≤ 401) :=
by
  sorry

end arithmetic_sequence_integers_l2263_226321


namespace percentage_increase_after_decrease_l2263_226316

variable (P : ℝ) (x : ℝ)

-- Conditions
def decreased_price : ℝ := 0.80 * P
def final_price_condition : Prop := 0.80 * P + (x / 100) * (0.80 * P) = 1.04 * P
def correct_answer : Prop := x = 30

-- The proof goal
theorem percentage_increase_after_decrease : final_price_condition P x → correct_answer x :=
by sorry

end percentage_increase_after_decrease_l2263_226316


namespace power_function_k_values_l2263_226308

theorem power_function_k_values (k : ℝ) : (∃ (a : ℝ), (k^2 - k - 5) = a) → (k = 3 ∨ k = -2) :=
by
  intro h
  have h1 : k^2 - k - 5 = 1 := sorry -- Using the condition that it is a power function
  have h2 : k^2 - k - 6 = 0 := by linarith -- Simplify the equation
  exact sorry -- Solve the quadratic equation

end power_function_k_values_l2263_226308


namespace find_m_l2263_226364

-- Define the conditions
def function_is_decreasing (m : ℝ) : Prop := 
  (m^2 - m - 1 = 1) ∧ (1 - m < 0)

-- The proof problem: prove m = 2 given the conditions
theorem find_m (m : ℝ) (h : function_is_decreasing m) : m = 2 := 
by
  sorry -- Proof to be filled in

end find_m_l2263_226364


namespace arithmetic_sequence_a5_l2263_226324

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : 
  a 5 = 7 :=
by
  -- proof to be filled later
  sorry

end arithmetic_sequence_a5_l2263_226324


namespace like_terms_mn_eq_neg1_l2263_226360

variable (m n : ℤ)

theorem like_terms_mn_eq_neg1
  (hx : m + 3 = 4)
  (hy : n + 3 = 1) :
  m + n = -1 :=
sorry

end like_terms_mn_eq_neg1_l2263_226360


namespace problem_part_1_problem_part_2_l2263_226393

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def tan_2x_when_parallel (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Prop :=
    Real.tan (2 * x) = 12 / 5

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2

def range_f_on_interval : Prop :=
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, -Real.sqrt 2 / 2 ≤ f x ∧ f x ≤ 1 / 2

theorem problem_part_1 (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Real.tan (2 * x) = 12 / 5 :=
by
  sorry

theorem problem_part_2 : range_f_on_interval :=
by
  sorry

end problem_part_1_problem_part_2_l2263_226393


namespace compute_usage_difference_l2263_226358

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end compute_usage_difference_l2263_226358


namespace power_function_passes_through_1_1_l2263_226370

theorem power_function_passes_through_1_1 (n : ℝ) : (1 : ℝ) ^ n = 1 :=
by
  -- Proof will go here
  sorry

end power_function_passes_through_1_1_l2263_226370


namespace circle_radius_l2263_226320

theorem circle_radius (M N : ℝ) (h1 : M / N = 20) :
  ∃ r : ℝ, M = π * r^2 ∧ N = 2 * π * r ∧ r = 40 :=
by
  sorry

end circle_radius_l2263_226320


namespace area_triangle_ABC_is_correct_l2263_226314

noncomputable def radius : ℝ := 4

noncomputable def angleABDiameter : ℝ := 30

noncomputable def ratioAM_MB : ℝ := 2 / 3

theorem area_triangle_ABC_is_correct :
  ∃ (area : ℝ), area = (180 * Real.sqrt 3) / 19 :=
by sorry

end area_triangle_ABC_is_correct_l2263_226314
