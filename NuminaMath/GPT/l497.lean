import Mathlib

namespace NUMINAMATH_GPT_hyperbola_range_of_k_l497_49757

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ (x y : ℝ), (x^2)/(k-3) + (y^2)/(k+3) = 1 ∧ 
  (k-3 < 0) ∧ (k+3 > 0)) → (-3 < k ∧ k < 3) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_range_of_k_l497_49757


namespace NUMINAMATH_GPT_angle_relation_l497_49723

theorem angle_relation
  (x y z w : ℝ)
  (h_sum : x + y + z + (360 - w) = 360) :
  x = w - y - z :=
by
  sorry

end NUMINAMATH_GPT_angle_relation_l497_49723


namespace NUMINAMATH_GPT_paint_per_color_equal_l497_49704

theorem paint_per_color_equal (total_paint : ℕ) (num_colors : ℕ) (paint_per_color : ℕ) : 
  total_paint = 15 ∧ num_colors = 3 → paint_per_color = 5 := by
  sorry

end NUMINAMATH_GPT_paint_per_color_equal_l497_49704


namespace NUMINAMATH_GPT_part1_part2_l497_49755

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l497_49755


namespace NUMINAMATH_GPT_total_rainfall_2019_to_2021_l497_49749

theorem total_rainfall_2019_to_2021 :
  let R2019 := 50
  let R2020 := R2019 + 5
  let R2021 := R2020 - 3
  12 * R2019 + 12 * R2020 + 12 * R2021 = 1884 :=
by
  sorry

end NUMINAMATH_GPT_total_rainfall_2019_to_2021_l497_49749


namespace NUMINAMATH_GPT_abs_eq_neg_iff_non_positive_l497_49794

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_abs_eq_neg_iff_non_positive_l497_49794


namespace NUMINAMATH_GPT_probability_nearest_odd_l497_49714

def is_odd_nearest (a b : ℝ) : Prop := ∃ k : ℤ, 2 * k + 1 = Int.floor ((a - b) / (a + b))

def is_valid (a b : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1

noncomputable def probability_odd_nearest : ℝ :=
  let interval_area := 1 -- the area of the unit square [0, 1] x [0, 1]
  let odd_area := 1 / 3 -- as derived from the geometric interpretation in the problem's solution
  odd_area / interval_area

theorem probability_nearest_odd (a b : ℝ) (h : is_valid a b) :
  probability_odd_nearest = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_nearest_odd_l497_49714


namespace NUMINAMATH_GPT_Patel_family_theme_park_expenses_l497_49706

def regular_ticket_price : ℝ := 12.5
def senior_discount : ℝ := 0.8
def child_discount : ℝ := 0.6
def senior_ticket_price := senior_discount * regular_ticket_price
def child_ticket_price := child_discount * regular_ticket_price

theorem Patel_family_theme_park_expenses :
  (2 * senior_ticket_price + 2 * child_ticket_price + 4 * regular_ticket_price) = 85 := by
  sorry

end NUMINAMATH_GPT_Patel_family_theme_park_expenses_l497_49706


namespace NUMINAMATH_GPT_martha_profit_l497_49780

theorem martha_profit :
  let loaves_baked := 60
  let cost_per_loaf := 1
  let morning_price := 3
  let afternoon_price := 3 * 0.75
  let evening_price := 2
  let morning_loaves := loaves_baked / 3
  let afternoon_loaves := (loaves_baked - morning_loaves) / 2
  let evening_loaves := loaves_baked - morning_loaves - afternoon_loaves
  let morning_revenue := morning_loaves * morning_price
  let afternoon_revenue := afternoon_loaves * afternoon_price
  let evening_revenue := evening_loaves * evening_price
  let total_revenue := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost := loaves_baked * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 85 := 
by
  sorry

end NUMINAMATH_GPT_martha_profit_l497_49780


namespace NUMINAMATH_GPT_brenda_friends_l497_49746

def total_slices (pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ := pizzas * slices_per_pizza
def total_people (total_slices : ℕ) (slices_per_person : ℕ) : ℕ := total_slices / slices_per_person
def friends (total_people : ℕ) : ℕ := total_people - 1

theorem brenda_friends (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (slices_per_person : ℕ) (pizzas_ordered : pizzas = 5) 
  (slices_per_pizza_value : slices_per_pizza = 4) 
  (slices_per_person_value : slices_per_person = 2) :
  friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) = 9 :=
by
  rw [pizzas_ordered, slices_per_pizza_value, slices_per_person_value]
  sorry

end NUMINAMATH_GPT_brenda_friends_l497_49746


namespace NUMINAMATH_GPT_jar_and_beans_weight_is_60_percent_l497_49702

theorem jar_and_beans_weight_is_60_percent
  (J B : ℝ)
  (h1 : J = 0.10 * (J + B))
  (h2 : ∃ x : ℝ, x = 0.5555555555555556 ∧ (J + x * B = 0.60 * (J + B))) :
  J + 0.5555555555555556 * B = 0.60 * (J + B) :=
by
  sorry

end NUMINAMATH_GPT_jar_and_beans_weight_is_60_percent_l497_49702


namespace NUMINAMATH_GPT_find_equation_of_line_l497_49734

open Real

noncomputable def equation_of_line : Prop :=
  ∃ c : ℝ, (∀ (x y : ℝ), (3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 → 2 * x + 3 * y + c = 0)) ∧
  ∃ x y : ℝ, 3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 ∧
              (2 * x + 3 * y + c = 0 → 6 * x + 9 * y - 7 = 0)

theorem find_equation_of_line : equation_of_line :=
sorry

end NUMINAMATH_GPT_find_equation_of_line_l497_49734


namespace NUMINAMATH_GPT_final_score_l497_49760

theorem final_score (questions_first_half questions_second_half : Nat)
  (points_correct points_incorrect : Int)
  (correct_first_half incorrect_first_half correct_second_half incorrect_second_half : Nat) :
  questions_first_half = 10 →
  questions_second_half = 15 →
  points_correct = 3 →
  points_incorrect = -1 →
  correct_first_half = 6 →
  incorrect_first_half = 4 →
  correct_second_half = 10 →
  incorrect_second_half = 5 →
  (points_correct * correct_first_half + points_incorrect * incorrect_first_half 
   + points_correct * correct_second_half + points_incorrect * incorrect_second_half) = 39 := 
by
  intros
  sorry

end NUMINAMATH_GPT_final_score_l497_49760


namespace NUMINAMATH_GPT_regular_pentagon_cannot_cover_floor_completely_l497_49728

theorem regular_pentagon_cannot_cover_floor_completely
  (hexagon_interior_angle : ℝ)
  (pentagon_interior_angle : ℝ)
  (square_interior_angle : ℝ)
  (triangle_interior_angle : ℝ)
  (hexagon_condition : 360 / hexagon_interior_angle = 3)
  (square_condition : 360 / square_interior_angle = 4)
  (triangle_condition : 360 / triangle_interior_angle = 6)
  (pentagon_condition : 360 / pentagon_interior_angle ≠ 3)
  (pentagon_condition2 : 360 / pentagon_interior_angle ≠ 4)
  (pentagon_condition3 : 360 / pentagon_interior_angle ≠ 6) :
  pentagon_interior_angle = 108 := 
  sorry

end NUMINAMATH_GPT_regular_pentagon_cannot_cover_floor_completely_l497_49728


namespace NUMINAMATH_GPT_area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l497_49707

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
  (1 / 2) * b * h

noncomputable def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

theorem area_triangle_ACD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 →
  area_of_triangle C 20 = 100 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

theorem area_trapezoid_ABCD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 → 
  area_trapezoid 24 10 24 = 260 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

end NUMINAMATH_GPT_area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l497_49707


namespace NUMINAMATH_GPT_rowing_time_ratio_l497_49730

theorem rowing_time_ratio
  (V_b : ℝ) (V_s : ℝ) (V_upstream : ℝ) (V_downstream : ℝ) (T_upstream T_downstream : ℝ)
  (h1 : V_b = 39) (h2 : V_s = 13)
  (h3 : V_upstream = V_b - V_s) (h4 : V_downstream = V_b + V_s)
  (h5 : T_upstream * V_upstream = T_downstream * V_downstream) :
  T_upstream / T_downstream = 2 := by
  sorry

end NUMINAMATH_GPT_rowing_time_ratio_l497_49730


namespace NUMINAMATH_GPT_Andy_has_4_more_candies_than_Caleb_l497_49744

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end NUMINAMATH_GPT_Andy_has_4_more_candies_than_Caleb_l497_49744


namespace NUMINAMATH_GPT_deepaks_age_l497_49716

theorem deepaks_age (R D : ℕ) (h1 : R / D = 5 / 2) (h2 : R + 6 = 26) : D = 8 := 
sorry

end NUMINAMATH_GPT_deepaks_age_l497_49716


namespace NUMINAMATH_GPT_height_of_pole_l497_49732

theorem height_of_pole (pole_shadow tree_shadow tree_height : ℝ) 
                       (ratio_equal : pole_shadow = 84 ∧ tree_shadow = 32 ∧ tree_height = 28) : 
                       round (tree_height * (pole_shadow / tree_shadow)) = 74 :=
by
  sorry

end NUMINAMATH_GPT_height_of_pole_l497_49732


namespace NUMINAMATH_GPT_desired_ellipse_properties_l497_49764

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2)/(a^2) + (x^2)/(b^2) = 1

def ellipse_has_foci (a b : ℝ) (c : ℝ) : Prop :=
  c^2 = a^2 - b^2

def desired_ellipse_passes_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  is_ellipse a b P.1 P.2

def foci_of_ellipse (a b : ℝ) (c : ℝ) : Prop :=
  ellipse_has_foci a b c

axiom given_ellipse_foci : foci_of_ellipse 3 2 (Real.sqrt 5)

theorem desired_ellipse_properties :
  desired_ellipse_passes_through_point 4 (Real.sqrt 11) (0, 4) ∧
  foci_of_ellipse 4 (Real.sqrt 11) (Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_desired_ellipse_properties_l497_49764


namespace NUMINAMATH_GPT_notAPrpos_l497_49782

def isProposition (s : String) : Prop :=
  s = "6 > 4" ∨ s = "If f(x) is a sine function, then f(x) is a periodic function." ∨ s = "1 ∈ {1, 2, 3}"

theorem notAPrpos (s : String) : ¬isProposition "Is a linear function an increasing function?" :=
by
  sorry

end NUMINAMATH_GPT_notAPrpos_l497_49782


namespace NUMINAMATH_GPT_no_solution_exists_l497_49713

theorem no_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x^y + 3 = y^x ∧ 3 * x^y = y^x + 8) :=
by
  intro h
  obtain ⟨eq1, eq2⟩ := h
  sorry

end NUMINAMATH_GPT_no_solution_exists_l497_49713


namespace NUMINAMATH_GPT_minimum_manhattan_distance_l497_49756

open Real

def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 2 + P.2^2 = 1

def line (Q : ℝ × ℝ) : Prop := 3 * Q.1 + 4 * Q.2 = 12

def manhattan_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem minimum_manhattan_distance :
  ∃ P Q, ellipse P ∧ line Q ∧
    ∀ P' Q', ellipse P' → line Q' → manhattan_distance P Q ≤ manhattan_distance P' Q' :=
  sorry

end NUMINAMATH_GPT_minimum_manhattan_distance_l497_49756


namespace NUMINAMATH_GPT_problem_statement_l497_49700

theorem problem_statement :
  let a := (List.range (60 / 12)).card
  let b := (List.range (60 / Nat.lcm (Nat.lcm 2 3) 4)).card
  (a - b) ^ 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l497_49700


namespace NUMINAMATH_GPT_even_sum_probability_l497_49776

-- Definition of probabilities for the first wheel
def prob_first_even : ℚ := 2 / 6
def prob_first_odd  : ℚ := 4 / 6

-- Definition of probabilities for the second wheel
def prob_second_even : ℚ := 3 / 8
def prob_second_odd  : ℚ := 5 / 8

-- The expected probability of the sum being even
theorem even_sum_probability : prob_first_even * prob_second_even + prob_first_odd * prob_second_odd = 13 / 24 := by
  sorry

end NUMINAMATH_GPT_even_sum_probability_l497_49776


namespace NUMINAMATH_GPT_combined_experience_l497_49742

noncomputable def james_experience : ℕ := 20
noncomputable def john_experience_8_years_ago : ℕ := 2 * (james_experience - 8)
noncomputable def john_current_experience : ℕ := john_experience_8_years_ago + 8
noncomputable def mike_experience : ℕ := john_current_experience - 16

theorem combined_experience :
  james_experience + john_current_experience + mike_experience = 68 :=
by
  sorry

end NUMINAMATH_GPT_combined_experience_l497_49742


namespace NUMINAMATH_GPT_PlayStation_cost_l497_49729

def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def price_per_game : ℝ := 7.5
def games_to_sell : ℕ := 20
def total_gift_money : ℝ := birthday_money + christmas_money
def total_games_money : ℝ := games_to_sell * price_per_game
def total_money : ℝ := total_gift_money + total_games_money

theorem PlayStation_cost : total_money = 500 := by
  sorry

end NUMINAMATH_GPT_PlayStation_cost_l497_49729


namespace NUMINAMATH_GPT_part_one_part_two_l497_49712

theorem part_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 := sorry

theorem part_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := sorry

end NUMINAMATH_GPT_part_one_part_two_l497_49712


namespace NUMINAMATH_GPT_not_perfect_square_l497_49778

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 3^n + 2 * 17^n := sorry

end NUMINAMATH_GPT_not_perfect_square_l497_49778


namespace NUMINAMATH_GPT_cupboard_selling_percentage_l497_49741

theorem cupboard_selling_percentage (CP SP : ℝ) (h1 : CP = 6250) (h2 : SP + 1500 = 6250 * 1.12) :
  ((CP - SP) / CP) * 100 = 12 := by
sorry

end NUMINAMATH_GPT_cupboard_selling_percentage_l497_49741


namespace NUMINAMATH_GPT_floors_above_l497_49733

theorem floors_above (dennis_floor charlie_floor frank_floor : ℕ)
  (h1 : dennis_floor = 6)
  (h2 : frank_floor = 16)
  (h3 : charlie_floor = frank_floor / 4) :
  dennis_floor - charlie_floor = 2 :=
by
  sorry

end NUMINAMATH_GPT_floors_above_l497_49733


namespace NUMINAMATH_GPT_lola_pop_tarts_baked_l497_49771

theorem lola_pop_tarts_baked :
  ∃ P : ℕ, (13 + P + 8) + (16 + 12 + 14) = 73 ∧ P = 10 := by
  sorry

end NUMINAMATH_GPT_lola_pop_tarts_baked_l497_49771


namespace NUMINAMATH_GPT_probability_of_negative_m_l497_49793

theorem probability_of_negative_m (m : ℤ) (h₁ : -2 ≤ m) (h₂ : m < (9 : ℤ) / 4) :
  ∃ (neg_count total_count : ℤ), 
    (neg_count = 2) ∧ (total_count = 5) ∧ (m ∈ {i : ℤ | -2 ≤ i ∧ i < 2 ∧ i < 9 / 4}) → 
    (neg_count / total_count = 2 / 5) :=
sorry

end NUMINAMATH_GPT_probability_of_negative_m_l497_49793


namespace NUMINAMATH_GPT_max_days_for_process_C_l497_49784

/- 
  A project consists of four processes: A, B, C, and D, which require 2, 5, x, and 4 days to complete, respectively.
  The following conditions are given:
  - A and B can start at the same time.
  - C can start after A is completed.
  - D can start after both B and C are completed.
  - The total duration of the project is 9 days.
  We need to prove that the maximum number of days required to complete process C is 3.
-/
theorem max_days_for_process_C
  (A B C D : ℕ)
  (hA : A = 2)
  (hB : B = 5)
  (hD : D = 4)
  (total_duration : ℕ)
  (h_total : total_duration = 9)
  (h_condition1 : A + C + D = total_duration) : 
  C = 3 :=
by
  rw [hA, hD, h_total] at h_condition1
  linarith

#check max_days_for_process_C

end NUMINAMATH_GPT_max_days_for_process_C_l497_49784


namespace NUMINAMATH_GPT_interest_rate_A_to_B_l497_49719

theorem interest_rate_A_to_B :
  ∀ (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ) (interest_C : ℝ) (interest_A : ℝ),
    principal = 3500 →
    rate_C = 0.13 →
    time = 3 →
    gain_B = 315 →
    interest_C = principal * rate_C * time →
    gain_B = interest_C - interest_A →
    interest_A = principal * (R / 100) * time →
    R = 10 := by
  sorry

end NUMINAMATH_GPT_interest_rate_A_to_B_l497_49719


namespace NUMINAMATH_GPT_cos_240_eq_neg_half_l497_49722

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_240_eq_neg_half_l497_49722


namespace NUMINAMATH_GPT_rate_per_sq_meter_l497_49708

theorem rate_per_sq_meter (length width : ℝ) (total_cost : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : total_cost = 16500) : 
  total_cost / (length * width) = 800 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_sq_meter_l497_49708


namespace NUMINAMATH_GPT_problem1_problem2_l497_49715

-- First problem
theorem problem1 (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := 
by sorry

-- Second problem
theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : ∃ k, a^x = k ∧ b^y = k ∧ c^z = k) (h_sum : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l497_49715


namespace NUMINAMATH_GPT_max_true_statements_l497_49799

theorem max_true_statements (c d : ℝ) : 
  (∃ n, 1 ≤ n ∧ n ≤ 5 ∧ 
    (n = (if (1/c > 1/d) then 1 else 0) +
          (if (c^2 < d^2) then 1 else 0) +
          (if (c > d) then 1 else 0) +
          (if (c > 0) then 1 else 0) +
          (if (d > 0) then 1 else 0))) → 
  n ≤ 3 := 
sorry

end NUMINAMATH_GPT_max_true_statements_l497_49799


namespace NUMINAMATH_GPT_smallest_square_factor_2016_l497_49738

theorem smallest_square_factor_2016 : ∃ n : ℕ, (168 = n) ∧ (∃ k : ℕ, k^2 = n) ∧ (2016 ∣ k^2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_square_factor_2016_l497_49738


namespace NUMINAMATH_GPT_fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l497_49774

def visitors_enjoyed_understood_fraction (E U : ℕ) (total_visitors no_enjoy_no_understood : ℕ) : Prop :=
  E = U ∧
  no_enjoy_no_understood = 110 ∧
  total_visitors = 440 ∧
  E = (total_visitors - no_enjoy_no_understood) / 2 ∧
  E = 165 ∧
  (E / total_visitors) = 3 / 8

theorem fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8 :
  ∃ (E U : ℕ), visitors_enjoyed_understood_fraction E U 440 110 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l497_49774


namespace NUMINAMATH_GPT_binomial_expression_value_l497_49703

theorem binomial_expression_value :
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end NUMINAMATH_GPT_binomial_expression_value_l497_49703


namespace NUMINAMATH_GPT_difference_of_same_prime_factors_l497_49773

theorem difference_of_same_prime_factors (n : ℕ) :
  ∃ a b : ℕ, a - b = n ∧ (a.primeFactors.card = b.primeFactors.card) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_same_prime_factors_l497_49773


namespace NUMINAMATH_GPT_infinite_series_evaluation_l497_49759

theorem infinite_series_evaluation :
  (∑' m : ℕ, ∑' n : ℕ, 1 / (m * n * (m + n + 2))) = 3 :=
  sorry

end NUMINAMATH_GPT_infinite_series_evaluation_l497_49759


namespace NUMINAMATH_GPT_find_share_of_C_l497_49765

-- Definitions and assumptions
def share_in_ratio (x : ℕ) : Prop :=
  let a := 2 * x
  let b := 3 * x
  let c := 4 * x
  a + b + c = 945

-- Statement to prove
theorem find_share_of_C :
  ∃ x : ℕ, share_in_ratio x ∧ 4 * x = 420 :=
by
  -- We skip the proof here.
  sorry

end NUMINAMATH_GPT_find_share_of_C_l497_49765


namespace NUMINAMATH_GPT_find_number_l497_49743

-- Define the main problem statement
theorem find_number (x : ℝ) (h : 0.50 * x = 0.80 * 150 + 80) : x = 400 := by
  sorry

end NUMINAMATH_GPT_find_number_l497_49743


namespace NUMINAMATH_GPT_average_speed_of_car_l497_49763

/-- The average speed of a car over four hours given specific distances covered each hour. -/
theorem average_speed_of_car
  (d1 d2 d3 d4 : ℝ)
  (t1 t2 t3 t4 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 40)
  (h3 : d3 = 60)
  (h4 : d4 = 100)
  (h5 : t1 = 1)
  (h6 : t2 = 1)
  (h7 : t3 = 1)
  (h8 : t4 = 1) :
  (d1 + d2 + d3 + d4) / (t1 + t2 + t3 + t4) = 55 :=
by sorry

end NUMINAMATH_GPT_average_speed_of_car_l497_49763


namespace NUMINAMATH_GPT_semicircles_problem_l497_49787

-- Define the problem in Lean
theorem semicircles_problem 
  (D : ℝ) -- Diameter of the large semicircle
  (N : ℕ) -- Number of small semicircles
  (r : ℝ) -- Radius of each small semicircle
  (H1 : D = 2 * N * r) -- Combined diameter of small semicircles is equal to the large semicircle's diameter
  (H2 : (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 10) -- Ratio of areas condition
  : N = 11 :=
   sorry -- Proof to be filled in later

end NUMINAMATH_GPT_semicircles_problem_l497_49787


namespace NUMINAMATH_GPT_prob_D_correct_l497_49751

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 3
def prob_C : ℚ := 1 / 6
def total_prob (prob_D : ℚ) : Prop := prob_A + prob_B + prob_C + prob_D = 1

theorem prob_D_correct : ∃ (prob_D : ℚ), total_prob prob_D ∧ prob_D = 1 / 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_prob_D_correct_l497_49751


namespace NUMINAMATH_GPT_tan_210_eq_neg_sqrt3_over_3_l497_49766

noncomputable def angle_210 : ℝ := 210 * (Real.pi / 180)
noncomputable def angle_30 : ℝ := 30 * (Real.pi / 180)

theorem tan_210_eq_neg_sqrt3_over_3 : Real.tan angle_210 = -Real.sqrt 3 / 3 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_tan_210_eq_neg_sqrt3_over_3_l497_49766


namespace NUMINAMATH_GPT_scientific_notation_gdp_2022_l497_49770

def gdp_2022_fujian : ℝ := 53100 * 10^9

theorem scientific_notation_gdp_2022 : 
  (53100 * 10^9) = 5.31 * 10^12 :=
by
  -- The proof is based on the understanding that 53100 * 10^9 can be rewritten as 5.31 * 10^12
  -- However, this proof is currently omitted with a placeholder.
  sorry

end NUMINAMATH_GPT_scientific_notation_gdp_2022_l497_49770


namespace NUMINAMATH_GPT_twenty_five_percent_of_2004_l497_49711

theorem twenty_five_percent_of_2004 : (1 / 4 : ℝ) * 2004 = 501 := by
  sorry

end NUMINAMATH_GPT_twenty_five_percent_of_2004_l497_49711


namespace NUMINAMATH_GPT_correct_operation_l497_49788

theorem correct_operation (a : ℝ) :
  (a^2)^3 = a^6 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l497_49788


namespace NUMINAMATH_GPT_regular_polygon_sides_l497_49797

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l497_49797


namespace NUMINAMATH_GPT_remainder_of_sum_of_first_150_numbers_l497_49727

def sum_of_first_n_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem remainder_of_sum_of_first_150_numbers :
  (sum_of_first_n_natural_numbers 150) % 5000 = 1275 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_first_150_numbers_l497_49727


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_segment_ratio_l497_49762

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ)
  (h₀ : 0 < x)
  (AB BC : ℝ)
  (h₁ : AB = 3 * x)
  (h₂ : BC = 4 * x) :
  ∃ AD DC : ℝ, AD / DC = 3 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_segment_ratio_l497_49762


namespace NUMINAMATH_GPT_cos_theta_equal_neg_inv_sqrt_5_l497_49767

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x - Real.cos x

theorem cos_theta_equal_neg_inv_sqrt_5 (θ : ℝ) (h_max : ∀ x : ℝ, f θ ≥ f x) : Real.cos θ = -1 / Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_theta_equal_neg_inv_sqrt_5_l497_49767


namespace NUMINAMATH_GPT_vector_subtraction_proof_l497_49736

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_proof_l497_49736


namespace NUMINAMATH_GPT_triangle_interior_angle_at_least_one_leq_60_l497_49769

theorem triangle_interior_angle_at_least_one_leq_60 {α β γ : ℝ} :
  α + β + γ = 180 →
  (α > 60 ∧ β > 60 ∧ γ > 60) → false :=
by
  intro hsum hgt
  have hα : α > 60 := hgt.1
  have hβ : β > 60 := hgt.2.1
  have hγ : γ > 60 := hgt.2.2
  have h_total: α + β + γ > 60 + 60 + 60 := add_lt_add (add_lt_add hα hβ) hγ
  linarith

end NUMINAMATH_GPT_triangle_interior_angle_at_least_one_leq_60_l497_49769


namespace NUMINAMATH_GPT_min_value_expression_l497_49752

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l497_49752


namespace NUMINAMATH_GPT_inequality_xy_l497_49775

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end NUMINAMATH_GPT_inequality_xy_l497_49775


namespace NUMINAMATH_GPT_sale_book_cost_l497_49777

variable (x : ℝ)

def fiveSaleBooksCost (x : ℝ) : ℝ :=
  5 * x

def onlineBooksCost : ℝ :=
  40

def bookstoreBooksCost : ℝ :=
  3 * 40

def totalCost (x : ℝ) : ℝ :=
  fiveSaleBooksCost x + onlineBooksCost + bookstoreBooksCost

theorem sale_book_cost :
  totalCost x = 210 → x = 10 := by
  sorry

end NUMINAMATH_GPT_sale_book_cost_l497_49777


namespace NUMINAMATH_GPT_sin_double_angle_l497_49739

-- Lean code to define the conditions and represent the problem
variable (α : ℝ)
variable (x y : ℝ) 
variable (r : ℝ := Real.sqrt (x^2 + y^2))

-- Given conditions
def point_on_terminal_side (x y : ℝ) (h : x = 1 ∧ y = -2) : Prop :=
  ∃ α, (⟨1, -2⟩ : ℝ × ℝ) = ⟨Real.cos α * (Real.sqrt (1^2 + (-2)^2)), Real.sin α * (Real.sqrt (1^2 + (-2)^2))⟩

-- The theorem to prove
theorem sin_double_angle (h : point_on_terminal_side 1 (-2) ⟨rfl, rfl⟩) : 
  Real.sin (2 * α) = -4 / 5 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l497_49739


namespace NUMINAMATH_GPT_platform_length_l497_49747

theorem platform_length
  (train_length : ℝ := 360) -- The train is 360 meters long
  (train_speed_kmh : ℝ := 45) -- The train runs at a speed of 45 km/hr
  (time_to_pass_platform : ℝ := 60) -- It takes 60 seconds to pass the platform
  (platform_length : ℝ) : platform_length = 390 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l497_49747


namespace NUMINAMATH_GPT_simplify_expression_l497_49710

theorem simplify_expression (x : ℝ) :
  x - 3 * (1 + x) + 4 * (1 - x)^2 - 5 * (1 + 3 * x) = 4 * x^2 - 25 * x - 4 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l497_49710


namespace NUMINAMATH_GPT_g_three_fifths_l497_49717

-- Given conditions
variable (g : ℝ → ℝ)
variable (h₀ : g 0 = 0)
variable (h₁ : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
variable (h₂ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
variable (h₃ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3)

-- Proof statement
theorem g_three_fifths : g (3 / 5) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_g_three_fifths_l497_49717


namespace NUMINAMATH_GPT_base_is_16_l497_49724

noncomputable def base_y_eq : Prop := ∃ base : ℕ, base ^ 8 = 4 ^ 16

theorem base_is_16 (base : ℕ) (h₁ : base ^ 8 = 4 ^ 16) : base = 16 :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_base_is_16_l497_49724


namespace NUMINAMATH_GPT_geometric_sequence_sum_l497_49750

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℚ),
  (∀ n, 3 * a (n + 1) + a n = 0) ∧
  a 2 = -2/3 ∧
  (a 0 + a 1 + a 2 + a 3 + a 4) = 122/81 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l497_49750


namespace NUMINAMATH_GPT_ceil_inequality_range_x_solve_eq_l497_49726

-- Definition of the mathematical ceiling function to comply with the condition a).
def ceil (a : ℚ) : ℤ := ⌈a⌉

-- Condition 1: Relationship between m and ⌈m⌉.
theorem ceil_inequality (m : ℚ) : m ≤ ceil m ∧ ceil m < m + 1 :=
sorry

-- Part 2.1: Range of x given {3x + 2} = 8.
theorem range_x (x : ℚ) (h : ceil (3 * x + 2) = 8) : 5 / 3 < x ∧ x ≤ 2 :=
sorry

-- Part 2.2: Solving {3x - 2} = 2x + 1/2
theorem solve_eq (x : ℚ) (h : ceil (3 * x - 2) = 2 * x + 1 / 2) : x = 7 / 4 ∨ x = 9 / 4 :=
sorry

end NUMINAMATH_GPT_ceil_inequality_range_x_solve_eq_l497_49726


namespace NUMINAMATH_GPT_find_x2_y2_l497_49731

theorem find_x2_y2 (x y : ℝ) (h₁ : (x + y)^2 = 9) (h₂ : x * y = -6) : x^2 + y^2 = 21 := 
by
  sorry

end NUMINAMATH_GPT_find_x2_y2_l497_49731


namespace NUMINAMATH_GPT_avogadro_constant_problem_l497_49740

theorem avogadro_constant_problem 
  (N_A : ℝ) -- Avogadro's constant
  (mass1 : ℝ := 18) (molar_mass1 : ℝ := 20) (moles1 : ℝ := mass1 / molar_mass1) 
  (atoms_D2O_molecules : ℝ := 2) (atoms_D2O : ℝ := moles1 * atoms_D2O_molecules * N_A)
  (mass2 : ℝ := 14) (molar_mass_N2CO : ℝ := 28) (moles2 : ℝ := mass2 / molar_mass_N2CO)
  (electrons_per_molecule : ℝ := 14) (total_electrons_mixture : ℝ := moles2 * electrons_per_molecule * N_A)
  (volume3 : ℝ := 2.24) (temp_unk : Prop := true) -- unknown temperature
  (pressure_unk : Prop := true) -- unknown pressure
  (carbonate_molarity : ℝ := 0.1) (volume_solution : ℝ := 1) (moles_carbonate : ℝ := carbonate_molarity * volume_solution) 
  (anions_carbonate_solution : ℝ := moles_carbonate * N_A) :
  (atoms_D2O ≠ 2 * N_A) ∧ (anions_carbonate_solution > 0.1 * N_A) ∧ (total_electrons_mixture = 7 * N_A) -> 
  True := sorry

end NUMINAMATH_GPT_avogadro_constant_problem_l497_49740


namespace NUMINAMATH_GPT_domain_composite_function_l497_49798

theorem domain_composite_function (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x = y) →
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f (2^x - 1) = y) :=
by
  sorry

end NUMINAMATH_GPT_domain_composite_function_l497_49798


namespace NUMINAMATH_GPT_range_of_a_l497_49795

def f (x : ℝ) : ℝ := x^3 + x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a * x) > 2) → 0 < a ∧ a < 4 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l497_49795


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l497_49785

noncomputable def max (x y : ℕ) : ℕ := if x > y then x else y

def valid_pair_count (k : ℕ) : ℕ := 2 * k + 1

def pairs_count (a b : ℕ) : ℕ := 
  valid_pair_count 5 * valid_pair_count 3 * valid_pair_count 2 * valid_pair_count 1

theorem number_of_ordered_pairs : pairs_count 2 3 = 1155 := 
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l497_49785


namespace NUMINAMATH_GPT_average_tree_height_l497_49758

def mixed_num_to_improper (whole: ℕ) (numerator: ℕ) (denominator: ℕ) : Rat :=
  whole + (numerator / denominator)

theorem average_tree_height 
  (elm : Rat := mixed_num_to_improper 11 2 3)
  (oak : Rat := mixed_num_to_improper 17 5 6)
  (pine : Rat := mixed_num_to_improper 15 1 2)
  (num_trees : ℕ := 3) :
  ((elm + oak + pine) / num_trees) = (15 : Rat) := 
  sorry

end NUMINAMATH_GPT_average_tree_height_l497_49758


namespace NUMINAMATH_GPT_mark_charged_more_hours_l497_49761

theorem mark_charged_more_hours (P K M : ℕ) 
  (h1 : P + K + M = 135)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 75 := by {

sorry
}

end NUMINAMATH_GPT_mark_charged_more_hours_l497_49761


namespace NUMINAMATH_GPT_hyperbola_eccentricity_sqrt5_l497_49792

noncomputable def eccentricity_of_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_sqrt5
  (a b : ℝ)
  (h : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (y = x^2 + 1) → (x, y) = (1, 2)) :
  eccentricity_of_hyperbola a b = Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_sqrt5_l497_49792


namespace NUMINAMATH_GPT_cyclist_wait_20_minutes_l497_49720

noncomputable def cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (time_passed_minutes : ℝ) : ℝ :=
  let time_passed_hours := time_passed_minutes / 60
  let distance := cyclist_speed * time_passed_hours
  let hiker_catch_up_time := distance / hiker_speed
  hiker_catch_up_time * 60

theorem cyclist_wait_20_minutes :
  cyclist_wait_time 5 20 5 = 20 :=
by
  -- Definitions according to given conditions
  let hiker_speed := 5 -- miles per hour
  let cyclist_speed := 20 -- miles per hour
  let time_passed_minutes := 5
  -- Required result
  let result_needed := 20
  -- Using the cyclist_wait_time function
  show cyclist_wait_time hiker_speed cyclist_speed time_passed_minutes = result_needed
  sorry

end NUMINAMATH_GPT_cyclist_wait_20_minutes_l497_49720


namespace NUMINAMATH_GPT_problems_per_page_l497_49705

theorem problems_per_page (total_problems finished_problems remaining_pages problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : total_problems - finished_problems = 14)
  (h5 : 14 = remaining_pages * problems_per_page) :
  problems_per_page = 7 := 
by
  sorry

end NUMINAMATH_GPT_problems_per_page_l497_49705


namespace NUMINAMATH_GPT_initial_person_count_l497_49768

theorem initial_person_count
  (avg_weight_increase : ℝ)
  (weight_old_person : ℝ)
  (weight_new_person : ℝ)
  (h1 : avg_weight_increase = 4.2)
  (h2 : weight_old_person = 65)
  (h3 : weight_new_person = 98.6) :
  ∃ n : ℕ, weight_new_person - weight_old_person = avg_weight_increase * n ∧ n = 8 := 
by
  sorry

end NUMINAMATH_GPT_initial_person_count_l497_49768


namespace NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l497_49779

def spinsters := 22
def cats := spinsters + 55

theorem ratio_of_spinsters_to_cats : (spinsters : ℝ) / (cats : ℝ) = 2 / 7 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l497_49779


namespace NUMINAMATH_GPT_number_of_sets_of_popcorn_l497_49737

theorem number_of_sets_of_popcorn (t p s : ℝ) (k : ℕ) 
  (h1 : t = 5)
  (h2 : p = 0.80 * t)
  (h3 : s = 0.50 * p)
  (h4 : 4 * t + 4 * s + k * p = 36) :
  k = 2 :=
by sorry

end NUMINAMATH_GPT_number_of_sets_of_popcorn_l497_49737


namespace NUMINAMATH_GPT_find_b_find_area_of_ABC_l497_49796

variable {a b c : ℝ}
variable {B : ℝ}

-- Given Conditions
def given_conditions (a b c B : ℝ) := a = 4 ∧ c = 3 ∧ B = Real.arccos (1 / 8)

-- Proving b = sqrt(22)
theorem find_b (h : given_conditions a b c B) : b = Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) :=
by
  sorry

-- Proving the area of triangle ABC
theorem find_area_of_ABC (h : given_conditions a b c B) 
  (sinB : Real.sin B = 3 * Real.sqrt 7 / 8) : 
  (1 / 2) * a * c * Real.sin B = 9 * Real.sqrt 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_b_find_area_of_ABC_l497_49796


namespace NUMINAMATH_GPT_perpendicular_lines_l497_49772

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l497_49772


namespace NUMINAMATH_GPT_find_line_through_M_and_parallel_l497_49791
-- Lean code to represent the proof problem

def M : Prop := ∃ (x y : ℝ), 3 * x + 4 * y - 5 = 0 ∧ 2 * x - 3 * y + 8 = 0 

def line_parallel : Prop := ∃ (m b : ℝ), 2 * m + b = 0

theorem find_line_through_M_and_parallel :
  M → line_parallel → ∃ (a b c : ℝ), (a = 2) ∧ (b = 1) ∧ (c = 0) :=
by
  intros hM hLineParallel
  sorry

end NUMINAMATH_GPT_find_line_through_M_and_parallel_l497_49791


namespace NUMINAMATH_GPT_find_range_of_m_l497_49790

open Real

-- Definition for proposition p (the discriminant condition)
def real_roots (m : ℝ) : Prop := (3 * 3) - 4 * m ≥ 0

-- Definition for proposition q (ellipse with foci on x-axis conditions)
def is_ellipse (m : ℝ) : Prop := 
  9 - m > 0 ∧ 
  m - 2 > 0 ∧ 
  9 - m > m - 2

-- Lean statement for the mathematically equivalent proof problem
theorem find_range_of_m (m : ℝ) : (real_roots m ∧ is_ellipse m) → (2 < m ∧ m ≤ 9 / 4) := 
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l497_49790


namespace NUMINAMATH_GPT_range_of_p_l497_49789

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_p_l497_49789


namespace NUMINAMATH_GPT_total_puff_pastries_l497_49753

theorem total_puff_pastries (batches trays puff_pastry volunteers : ℕ) 
  (h_batches : batches = 1) 
  (h_trays : trays = 8) 
  (h_puff_pastry : puff_pastry = 25) 
  (h_volunteers : volunteers = 1000) : 
  (volunteers * trays * puff_pastry) = 200000 := 
by 
  have h_total_trays : volunteers * trays = 1000 * 8 := by sorry
  have h_total_puff_pastries_per_volunteer : trays * puff_pastry = 8 * 25 := by sorry
  have h_total_puff_pastries : volunteers * trays * puff_pastry = 1000 * 8 * 25 := by sorry
  sorry

end NUMINAMATH_GPT_total_puff_pastries_l497_49753


namespace NUMINAMATH_GPT_smallest_natural_number_with_condition_l497_49781

theorem smallest_natural_number_with_condition {N : ℕ} :
  (N % 10 = 6) ∧ (4 * N = (6 * 10 ^ ((Nat.digits 10 (N / 10)).length) + (N / 10))) ↔ N = 153846 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_with_condition_l497_49781


namespace NUMINAMATH_GPT_boxes_used_l497_49718

-- Define the given conditions
def oranges_per_box : ℕ := 10
def total_oranges : ℕ := 2650

-- Define the proof statement
theorem boxes_used : total_oranges / oranges_per_box = 265 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_boxes_used_l497_49718


namespace NUMINAMATH_GPT_crayons_difference_l497_49748

theorem crayons_difference (total_crayons : ℕ) (given_crayons : ℕ) (lost_crayons : ℕ) (h1 : total_crayons = 589) (h2 : given_crayons = 571) (h3 : lost_crayons = 161) : (given_crayons - lost_crayons) = 410 := by
  sorry

end NUMINAMATH_GPT_crayons_difference_l497_49748


namespace NUMINAMATH_GPT_problem_ratio_l497_49754

-- Define the conditions
variables 
  (R : ℕ) 
  (Bill_problems : ℕ := 20) 
  (Frank_problems_per_type : ℕ := 30)
  (types : ℕ := 4)

-- State the problem to prove
theorem problem_ratio (h1 : 3 * R = Frank_problems_per_type * types) :
  R / Bill_problems = 2 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_problem_ratio_l497_49754


namespace NUMINAMATH_GPT_smallest_stamps_l497_49783

theorem smallest_stamps : ∃ S, 1 < S ∧ (S % 9 = 1) ∧ (S % 10 = 1) ∧ (S % 11 = 1) ∧ S = 991 :=
by
  sorry

end NUMINAMATH_GPT_smallest_stamps_l497_49783


namespace NUMINAMATH_GPT_problem1_problem2_l497_49721

theorem problem1 : 
  -(3^3) * ((-1 : ℚ)/ 3)^2 - 24 * (3/4 - 1/6 + 3/8) = -26 := 
by 
  sorry

theorem problem2 : 
  -(1^100 : ℚ) - (3/4) / (((-2)^2) * ((-1 / 4) ^ 2) - 1 / 2) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l497_49721


namespace NUMINAMATH_GPT_solution_l497_49725

variable (a : ℕ → ℝ)

noncomputable def pos_sequence (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → a k > 0

noncomputable def recursive_relation (n : ℕ) : Prop :=
  ∀ n : ℕ, (n > 0) → (n+1) * a (n+1)^2 - n * a n^2 + a (n+1) * a n = 0

noncomputable def sequence_condition (n : ℕ) : Prop :=
  a 1 = 1 ∧ pos_sequence a n ∧ recursive_relation a n

theorem solution : ∀ n : ℕ, n > 0 → sequence_condition a n → a n = 1 / n :=
by
  intros n hn h
  sorry

end NUMINAMATH_GPT_solution_l497_49725


namespace NUMINAMATH_GPT_smaller_of_two_integers_l497_49786

noncomputable def smaller_integer (m n : ℕ) : ℕ :=
if m < n then m else n

theorem smaller_of_two_integers :
  ∀ (m n : ℕ),
  100 ≤ m ∧ m < 1000 ∧ 100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200 →
  smaller_integer m n = 891 :=
by
  intros m n h
  -- Assuming m, n are positive three-digit integers and satisfy the condition
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2.1
  have h5 := h.2.2.2.2
  sorry

end NUMINAMATH_GPT_smaller_of_two_integers_l497_49786


namespace NUMINAMATH_GPT_find_length_of_wood_l497_49701

-- Definitions based on given conditions
def Area := 24  -- square feet
def Width := 6  -- feet

-- The mathematical proof problem turned into Lean 4 statement
theorem find_length_of_wood (h : Area = 24) (hw : Width = 6) : (Length : ℕ) ∈ {l | l = Area / Width ∧ l = 4} :=
by {
  sorry
}

end NUMINAMATH_GPT_find_length_of_wood_l497_49701


namespace NUMINAMATH_GPT_find_length_of_PC_l497_49709

theorem find_length_of_PC (P A B C D : ℝ × ℝ) (h1 : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 25)
                            (h2 : (P.1 - D.1)^2 + (P.2 - D.2)^2 = 36)
                            (h3 : (P.1 - B.1)^2 + (P.2 - B.2)^2 = 49)
                            (square_ABCD : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) :
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_PC_l497_49709


namespace NUMINAMATH_GPT_simplify_expression_l497_49735

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l497_49735


namespace NUMINAMATH_GPT_plane_equation_l497_49745

theorem plane_equation (A B C D x y z : ℤ) (h1 : A = 15) (h2 : B = -3) (h3 : C = 2) (h4 : D = -238) 
  (h5 : gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1) (h6 : A > 0) :
  A * x + B * y + C * z + D = 0 ↔ 15 * x - 3 * y + 2 * z - 238 = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l497_49745
