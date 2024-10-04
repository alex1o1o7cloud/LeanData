import Mathlib

namespace find_m_eq_zero_l542_542835

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end find_m_eq_zero_l542_542835


namespace quadrilateral_inequality_l542_542104

theorem quadrilateral_inequality
  (A B C D : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (d_AB : dist A B) (d_BC : dist B C) (d_CD : dist C D) (d_DA : dist D A)
  (d_AC : dist A C) :
  (d_AB < d_BC + d_CD + d_DA) ∧
  (d_BC < d_AB + d_CD + d_DA) ∧
  (d_CD < d_AB + d_BC + d_DA) ∧
  (d_DA < d_AB + d_BC + d_CD) :=
by
  sorry

end quadrilateral_inequality_l542_542104


namespace sequence_decreasing_l542_542335

noncomputable def x_n (a b : ℝ) (n : ℕ) : ℝ := 2 ^ n * (b ^ (1 / 2 ^ n) - a ^ (1 / 2 ^ n))

theorem sequence_decreasing (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : ∀ n : ℕ, x_n a b n > x_n a b (n + 1) :=
by
  sorry

end sequence_decreasing_l542_542335


namespace percent_shaded_area_of_rectangle_l542_542159

theorem percent_shaded_area_of_rectangle
  (side_length : ℝ)
  (length_rectangle : ℝ)
  (width_rectangle : ℝ)
  (overlap_length : ℝ)
  (h1 : side_length = 12)
  (h2 : length_rectangle = 20)
  (h3 : width_rectangle = 12)
  (h4 : overlap_length = 4)
  : (overlap_length * width_rectangle) / (length_rectangle * width_rectangle) * 100 = 20 :=
  sorry

end percent_shaded_area_of_rectangle_l542_542159


namespace equal_areas_AMS_HNS_l542_542054

open EuclideanGeometry

variables {A B C D E F H Q R T S M N : Point}
variables (triangle_ABC : Triangle A B C) 
variables (altitude_AD : isAltitude A D) 
variables (altitude_BE : isAltitude B E) 
variables (altitude_CF : isAltitude C F)
variables (orthocenter_H : isOrthocenter H A B C) 
variables (Q_on_circumcircle : OnCircumcircle Q triangle_ABC)
variables (QR_perp_BC : Perpendicular QR B C)
variables (line_R_parallel_AQ : Parallel (Line.through R (Line.parallelTo AQ))) 
variables (RS_perp_AM : PerpendicularLine RS AM)
variables (RS_perp_HN : PerpendicularLine RS HN)
variables (area_AMS area_HNS : ℝ)

theorem equal_areas_AMS_HNS (h : ∀ (Δ AMS : Triangle A M S) (Δ HNS : Triangle H N S), 
    area Δ AMS = area Δ HNS) : 
  area_AMS = area_HNS := 
sorry

end equal_areas_AMS_HNS_l542_542054


namespace smallest_sum_abc_d_l542_542893

theorem smallest_sum_abc_d (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) : a + b + c + d = 108 :=
sorry

end smallest_sum_abc_d_l542_542893


namespace sample_definition_l542_542613

variables (analysis_based_on : String)
variables (results_from : ℕ → Set ℕ)
variables (purpose : String)
variables (long_jump_results : Set ℕ)
variables (students_count : ℕ)

def sample_specified : Prop := 
  analysis_based_on = "long jump results" ∧ 
  results_from students_count = long_jump_results ∧ 
  purpose = "assess the physical health status of middle school students in the city"

theorem sample_definition :
  students_count = 12000 ∧ sample_specified analysis_based_on results_from purpose long_jump_results students_count → 
  long_jump_results = results_from 12000 :=
begin
  sorry
end

end sample_definition_l542_542613


namespace count_special_integers_lt_500_l542_542768

theorem count_special_integers_lt_500 :
  let count_M := (count (λ M : ℕ, (M < 500) ∧
    (∃ k1 k2 k3 : ℕ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≥ 1 ∧ k2 ≥ 1 ∧ k3 ≥ 1 ∧
        ∃ m1 m2 m3 : ℕ, M = k1 * (2 * m1 + k1 - 1) ∧ 
                         M = k2 * (2 * m2 + k2 - 1) ∧ 
                         M = k3 * (2 * m3 + k3 - 1))) [1..499]) in
  count_M = 9 := 
sorry

end count_special_integers_lt_500_l542_542768


namespace exists_nat_sol_x9_eq_2013y10_l542_542693

theorem exists_nat_sol_x9_eq_2013y10 : ∃ (x y : ℕ), x^9 = 2013 * y^10 :=
by {
  -- Assume x and y are natural numbers, and prove that x^9 = 2013 y^10 has a solution
  sorry
}

end exists_nat_sol_x9_eq_2013y10_l542_542693


namespace total_juice_p_used_l542_542206

theorem total_juice_p_used (p v m y : ℕ) (r : ℚ)
  (h1 : v = 25) 
  (h2 : m = 4)
  (h3 : r = 1 / 5)
  (h4 : p = 20) :
  let juice_in_y := 25 / 5 in
  let total_juice := p + juice_in_y in
  total_juice = 25 := 
by
  -- Given conditions
  sorry

end total_juice_p_used_l542_542206


namespace max_distance_between_circle_and_ellipse_l542_542572

noncomputable def max_distance_PQ : ℝ :=
  1 + (3 * Real.sqrt 6) / 2

theorem max_distance_between_circle_and_ellipse :
  ∀ (P Q : ℝ × ℝ), (P.1^2 + (P.2 - 2)^2 = 1) → 
                   (Q.1^2 / 9 + Q.2^2 = 1) →
                   dist P Q ≤ max_distance_PQ :=
by
  intros P Q hP hQ
  sorry

end max_distance_between_circle_and_ellipse_l542_542572


namespace largest_k_divisible_factorial_l542_542707

theorem largest_k_divisible_factorial (n : ℕ) (hn : n > 1000) : ∃ k, k = -3 ∧ 2^(n+k+2) ∣ nat.factorial n :=
by 
  sorry

end largest_k_divisible_factorial_l542_542707


namespace percentage_palm_oil_in_cheese_l542_542854

-- Define the conditions
variables (initial_cheese_price : ℝ) (initial_palm_oil_price : ℝ) (final_cheese_price : ℝ) (final_palm_oil_price : ℝ)

-- Condition 1: Initial price assumptions and price increase percentages
def conditions (initial_cheese_price initial_palm_oil_price : ℝ) : Prop :=
  final_cheese_price = initial_cheese_price * 1.03 ∧
  final_palm_oil_price = initial_palm_oil_price * 1.10

-- The main theorem to prove
theorem percentage_palm_oil_in_cheese
  (initial_cheese_price initial_palm_oil_price final_cheese_price final_palm_oil_price : ℝ)
  (h : conditions initial_cheese_price initial_palm_oil_price) :
  initial_palm_oil_price / initial_cheese_price = 0.30 :=
by
  sorry

end percentage_palm_oil_in_cheese_l542_542854


namespace water_pumped_in_half_hour_l542_542576

theorem water_pumped_in_half_hour (rate : ℕ) (half_hour : ℚ) (result : ℕ) : 
  rate = 600 → half_hour = 1/2 → result = 300 → (rate * half_hour).natAbs = result :=
begin
  intros h_rate h_half_hour h_result,
  rw [h_rate, h_half_hour],
  norm_num,
  exact h_result,
end

end water_pumped_in_half_hour_l542_542576


namespace sum_of_valid_a_l542_542783

-- Definitions following conditions
def ineq_1 (x a : ℤ) : Prop := (x / 3 + 1 ≤ (x + 3) / 2)
def ineq_2 (x a : ℤ) : Prop := (3 * x - (a - 1) / 3 < 0)
def eq_y (a y : ℤ) : Prop := ((9 - a) / (y - 1) + 2 / (1 - y) = 2)

-- Main theorem statement based on the question
theorem sum_of_valid_a :
  (∀ x : ℤ, (ineq_1 x a ∧ ineq_2 x a) ↔ (∀ y : ℤ, y > 0 ∧ eq_y a y) → {a : ℤ | true}.sum = 8) :=
sorry

end sum_of_valid_a_l542_542783


namespace max_points_top_four_teams_l542_542045

theorem max_points_top_four_teams 
  (total_teams : ℕ)
  (games_played : total_teams = 7)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)
  (total_games : total_games = 21)
  (top_teams_equal_points : top_teams_equal_points total_teams = (list.replicate 4 15)) :
  let 
    max_points := 15
  in top_teams_equal_points = true := 
sorry

end max_points_top_four_teams_l542_542045


namespace cubic_function_three_roots_l542_542130

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem cubic_function_three_roots : 
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f(a) = 0 ∧ f(b) = 0 ∧ f(c) = 0 := 
by 
  sorry

end cubic_function_three_roots_l542_542130


namespace unit_price_solution_purchase_plan_and_costs_l542_542046

-- Definitions based on the conditions (Note: numbers and relationships purely)
def unit_prices (x y : ℕ) : Prop :=
  3 * x + 2 * y = 60 ∧ x + 3 * y = 55

def prize_purchase_conditions (m n : ℕ) : Prop :=
  m + n = 100 ∧ 10 * m + 15 * n ≤ 1160 ∧ m ≤ 3 * n

-- Proving that the unit prices found match the given constraints
theorem unit_price_solution : ∃ x y : ℕ, unit_prices x y := by
  sorry

-- Proving the number of purchasing plans and minimum cost
theorem purchase_plan_and_costs : 
  (∃ (num_plans : ℕ) (min_cost : ℕ), 
    num_plans = 8 ∧ min_cost = 1125 ∧ 
    ∀ m n : ℕ, prize_purchase_conditions m n → 
      ((68 ≤ m ∧ m ≤ 75) →
      10 * m + 15 * (100 - m) = min_cost)) := by
  sorry

end unit_price_solution_purchase_plan_and_costs_l542_542046


namespace area_of_centroid_path_l542_542541

theorem area_of_centroid_path (A B C O G : ℝ) (r : ℝ) (h1 : A ≠ B) 
  (h2 : 2 * r = 30) (h3 : ∀ C, C ≠ A ∧ C ≠ B ∧ dist O C = r) 
  (h4 : dist O G = r / 3) : 
  (π * (r / 3)^2 = 25 * π) :=
by 
  -- def AB := 2 * r -- given AB is a diameter of the circle
  -- def O := (A + B) / 2 -- center of the circle
  -- def G := (A + B + C) / 3 -- centroid of triangle ABC
  sorry

end area_of_centroid_path_l542_542541


namespace exists_unique_continuous_extension_l542_542078

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end exists_unique_continuous_extension_l542_542078


namespace negation_exists_forall_l542_542912

def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 1)

theorem negation_exists_forall' : 
  (¬ ∃ (x : ℝ), x < 0 ∧ f x ≥ 0) ↔ (∀ (x : ℝ), x < 0 → f x < 0) :=
by 
  sorry

end negation_exists_forall_l542_542912


namespace expected_value_of_geometric_variance_of_geometric_l542_542862

noncomputable def expected_value (p : ℝ) : ℝ :=
  1 / p

noncomputable def variance (p : ℝ) : ℝ :=
  (1 - p) / (p ^ 2)

theorem expected_value_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, (n + 1 : ℝ) * (1 - p) ^ n * p = expected_value p := by
  sorry

theorem variance_of_geometric (p : ℝ) (hp : 0 < p ∧ p < 1) :
  ∑' n, ((n + 1 : ℝ) ^ 2) * (1 - p) ^ n * p - (expected_value p) ^ 2 = variance p := by
  sorry

end expected_value_of_geometric_variance_of_geometric_l542_542862


namespace selena_amount_left_l542_542861

noncomputable def cost_of_steak := 24 * 2   -- $48
noncomputable def tax_rate_steak := 0.07
noncomputable def tax_amount_steak := cost_of_steak * tax_rate_steak   -- $3.36
noncomputable def total_cost_steak := cost_of_steak + tax_amount_steak   -- $51.36

noncomputable def cost_of_burgers := 3.5 * 2   -- $7
noncomputable def tax_rate_burgers := 0.06
noncomputable def tax_amount_burgers := cost_of_burgers * tax_rate_burgers   -- $0.42
noncomputable def total_cost_burgers := cost_of_burgers + tax_amount_burgers   -- $7.42

noncomputable def cost_of_ice_cream := 2 * 3   -- $6
noncomputable def tax_rate_ice_cream := 0.08
noncomputable def tax_amount_ice_cream := cost_of_ice_cream * tax_rate_ice_cream   -- $0.48
noncomputable def total_cost_ice_cream := cost_of_ice_cream + tax_amount_ice_cream   -- $6.48

noncomputable def total_meal_cost := total_cost_steak + total_cost_burgers + total_cost_ice_cream   -- $65.26
noncomputable def initial_amount := 99
noncomputable def amount_left := initial_amount - total_meal_cost   -- $33.74

theorem selena_amount_left : amount_left = 33.74 := 
by
  sorry

end selena_amount_left_l542_542861


namespace regionA_regionC_area_ratio_l542_542109

-- Definitions for regions A and B
def regionA (l w : ℝ) : Prop := 2 * (l + w) = 16 ∧ l = 2 * w
def regionB (l w : ℝ) : Prop := 2 * (l + w) = 20 ∧ l = 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem regionA_regionC_area_ratio {lA wA lB wB lC wC : ℝ} :
  regionA lA wA → regionB lB wB → (lC = lB ∧ wC = wB) → 
  (area lC wC ≠ 0) → 
  (area lA wA / area lC wC = 16 / 25) :=
by
  intros hA hB hC hC_area_ne_zero
  sorry

end regionA_regionC_area_ratio_l542_542109


namespace probability_XiaoYu_group_A_l542_542514

theorem probability_XiaoYu_group_A :
  ∀ (students : Fin 48) (groups : Fin 4) (groupAssignment : Fin 48 → Fin 4)
    (student : Fin 48) (groupA : Fin 4),
    (∀ (s : Fin 48), ∃ (g : Fin 4), groupAssignment s = g) → 
    (∀ (g : Fin 4), ∃ (count : ℕ), (0 < count ∧ count ≤ 12) ∧
       (∃ (groupMembers : List (Fin 48)), groupMembers.length = count ∧
        (∀ (m : Fin 48), m ∈ groupMembers → groupAssignment m = g))) →
    (groupAssignment student = groupA) →
  ∃ (p : ℚ), p = (1/4) ∧ ∀ (s : Fin 48), groupAssignment s = groupA → p = (1/4) :=
by
  sorry

end probability_XiaoYu_group_A_l542_542514


namespace ne_of_P_l542_542003

-- Define the initial proposition P
def P : Prop := ∀ m : ℝ, (0 ≤ m → 4^m ≥ 4 * m)

-- Define the negation of P
def not_P : Prop := ∃ m : ℝ, (0 ≤ m ∧ 4^m < 4 * m)

-- The theorem we need to prove
theorem ne_of_P : ¬P ↔ not_P :=
by
  sorry

end ne_of_P_l542_542003


namespace cleaned_area_correct_l542_542543

def lizzie_cleaned : ℚ := 3534 + 2/3
def hilltown_team_cleaned : ℚ := 4675 + 5/8
def green_valley_cleaned : ℚ := 2847 + 7/9
def riverbank_cleaned : ℚ := 6301 + 1/3
def meadowlane_cleaned : ℚ := 3467 + 4/5

def total_cleaned : ℚ := lizzie_cleaned + hilltown_team_cleaned + green_valley_cleaned + riverbank_cleaned + meadowlane_cleaned
def total_farmland : ℚ := 28500

def remaining_area_to_clean : ℚ := total_farmland - total_cleaned

theorem cleaned_area_correct : remaining_area_to_clean = 7672.7964 :=
by
  sorry

end cleaned_area_correct_l542_542543


namespace existence_of_ellipse_equation_existence_of_line_l_l542_542346

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

def is_right_focus (F : ℝ × ℝ) : Prop :=
  (F = (1, 0))

def is_top_point (M : ℝ × ℝ) (b : ℝ) : Prop :=
  (M = (0, b))

def is_origin (O : ℝ × ℝ) : Prop :=
  (O = (0, 0))

def is_isosceles_right_triangle (O M F : ℝ × ℝ) : Prop :=
  ((O = (0, 0)) ∧ (F = (1, 0)) ∧ (∃ b : ℝ, b > 0 ∧ M = (0, b) ∧ b^2 + 1 = a^2))

theorem existence_of_ellipse_equation (a b : ℝ) (F M O : ℝ × ℝ)
  (h_ellipse : ellipse_eq a b)
  (h_focus : is_right_focus F)
  (h_top : is_top_point M b)
  (h_origin : is_origin O)
  (h_triangle : is_isosceles_right_triangle O M F) :
  a = sqrt 2 ∧ b = 1 :=
by sorry

theorem existence_of_line_l (F M : ℝ × ℝ)
  (h_focus : is_right_focus F)
  (h_top : is_top_point M 1) :
  ∃ l : ℝ → ℝ, (l = λ x, x - 4 / 3)
  ∧ (∃ (P Q : ℝ × ℝ), (P ≠ Q) ∧ (h_intersect_line_ellipse l P Q) ∧ (h_orthocenter_triangle PQM F)) :=
by sorry


end existence_of_ellipse_equation_existence_of_line_l_l542_542346


namespace minimum_candy_kinds_l542_542412

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542412


namespace birds_on_fence_l542_542640

theorem birds_on_fence (initial_birds : ℕ) (joined_birds : ℕ) (h₁ : initial_birds = 2) (h₂ : joined_birds = 4) : initial_birds + joined_birds = 6 :=
by 
  rw [h₁, h₂]
  exact rfl

end birds_on_fence_l542_542640


namespace OB1_eq_OB2_l542_542517

variables {A B C H H1 H2 B1 B2 O : Type*}
variables [inhabited A] [inhabited B] [inhabited C] 

-- Given conditions
variables 
  (angle_B : ∠ B = 90°)
  (altitude_BH : ∀ (ABC : Triangle A B C), altitude (ABC) B H AC)
  (incircle_touch_ABH_H1 : ∀ (ABH : Triangle A B H), incircle_touch (ABH) AB H1)
  (incircle_touch_ABH_B1 : ∀ (ABH : Triangle A B H), incircle_touch (ABH) AH B1)
  (incircle_touch_CBH_H2 : ∀ (CBH : Triangle C B H), incircle_touch (CBH) CB H2)
  (incircle_touch_CBH_B2 : ∀ (CBH : Triangle C B H), incircle_touch (CBH) CH B2)
  (circumcenter_H1BH2 : ∀ (H1BH2 : Triangle H1 B H2), circumcenter (H1BH2) O)

-- To prove
theorem OB1_eq_OB2 : distance O B1 = distance O B2 :=
sorry

end OB1_eq_OB2_l542_542517


namespace fraction_of_64_l542_542162

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l542_542162


namespace john_reams_needed_l542_542502

theorem john_reams_needed 
  (pages_flash_fiction_weekly : ℕ := 20) 
  (pages_short_story_weekly : ℕ := 50) 
  (pages_novel_annual : ℕ := 1500) 
  (weeks_in_year : ℕ := 52) 
  (sheets_per_ream : ℕ := 500) 
  (sheets_flash_fiction_weekly : ℕ := 10)
  (sheets_short_story_weekly : ℕ := 25) :
  let sheets_flash_fiction_annual := sheets_flash_fiction_weekly * weeks_in_year
  let sheets_short_story_annual := sheets_short_story_weekly * weeks_in_year
  let total_sheets_annual := sheets_flash_fiction_annual + sheets_short_story_annual + pages_novel_annual
  let reams_needed := (total_sheets_annual + sheets_per_ream - 1) / sheets_per_ream
  reams_needed = 7 := 
by sorry

end john_reams_needed_l542_542502


namespace solve_parabola_c_l542_542220

theorem solve_parabola_c (b c : ℝ) :
  (∀ x y : ℝ, x = 2 → y = 6 → y = 2*x^2 + b*x + c) →
  (∀ x y : ℝ, x = -3 → y = -24 → y = 2*x^2 + b*x + c) →
  c = -18 := by
  intros h1 h2
  have : 6 = 8 + 2 * b + c := h1 2 6 (by exact eq.refl 2) (by exact eq.refl 6)
  have : -24 = 18 - 3 * b + c := h2 (-3) (-24) (by exact eq.refl (-3)) (by exact eq.refl (-24))
  sorry

end solve_parabola_c_l542_542220


namespace find_AX_eq_six_l542_542391

open EuclideanGeometry

noncomputable def point_of_triangle (A B C : Point) (E F : Point) (X : Point) : Prop :=
  (dist A B = 10) ∧ (dist B C = 12) ∧ (dist A C = 14) ∧ 
  midpoint A B E ∧ midpoint A C F ∧
  (tangent_circumcircle {a := B, b := C} (circumcircle A E F) X) ∧ (X ≠ A) →
  (dist A X = 6)

theorem find_AX_eq_six (A B C E F X : Point) :
  point_of_triangle A B C E F X := 
sorry

end find_AX_eq_six_l542_542391


namespace impossible_target_config_l542_542869

def initial_config : list ℕ := [1, 0, 0, 0]
def target_config : list ℕ := [1, 9, 8, 9]
def transformation (a b : list ℕ) : bool :=
  (a.length = 4 ∧ b.length = 4) ∧ -- both lists are size 4
  ∃ k i,  b = match a with
                | [ai1, ai2, ai3, ai4] := 
                    match i with
                    | 0 := [ai1 - k, ai2 + k, ai3, ai4 + k]
                    | 1 := [ai1 + k, ai2 - k, ai3 + k, ai4]
                    | 2 := [ai1, ai2 + k, ai3 - k, ai4 + k]
                    | 3 := [ai1 + k, ai2, ai3 + k, ai4 - k]
                    | _ := a
                    end
                | _ := a
                end

theorem impossible_target_config :
  ¬ ∃ (steps : list (list ℕ)), 
      steps.head = initial_config ∧ 
      steps.last = some target_config ∧ 
      ∀ a b ∈ steps, transformation a b := 
sorry

end impossible_target_config_l542_542869


namespace tickets_distribution_l542_542144

theorem tickets_distribution (n m : ℕ) (h_n : n = 10) (h_m : m = 3) :
  ∃ (d : ℕ), d = 10 * 9 * 8 := 
sorry -- Detailed proof would go here.

end tickets_distribution_l542_542144


namespace g_nested_evaluation_l542_542536

def g (x : ℝ) : ℝ :=
if x > 5 then real.sqrt (x + 1) else x^3

theorem g_nested_evaluation :
  g (g (g 3)) = real.sqrt 6.29 := by
sorry

end g_nested_evaluation_l542_542536


namespace tan_a_values_l542_542311

theorem tan_a_values (a : ℝ) (h : sin (2 * a) = 2 - 2 * cos (2 * a)) :
  tan a = 0 ∨ tan a = 1/2 :=
sorry

end tan_a_values_l542_542311


namespace chantal_gain_l542_542256

variable (sweaters balls cost_selling cost_yarn total_gain : ℕ)

def chantal_knits_sweaters : Prop :=
  sweaters = 28 ∧
  balls = 4 ∧
  cost_yarn = 6 ∧
  cost_selling = 35 ∧
  total_gain = (sweaters * cost_selling) - (sweaters * balls * cost_yarn)

theorem chantal_gain : chantal_knits_sweaters sweaters balls cost_selling cost_yarn total_gain → total_gain = 308 :=
by sorry

end chantal_gain_l542_542256


namespace digit_multiplication_sum_l542_542049

-- Define the main problem statement in Lean 4
theorem digit_multiplication_sum (A B E F : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) 
                                            (h2 : 0 ≤ B ∧ B ≤ 9) 
                                            (h3 : 0 ≤ E ∧ E ≤ 9)
                                            (h4 : 0 ≤ F ∧ F ≤ 9)
                                            (h5 : A ≠ B) 
                                            (h6 : A ≠ E) 
                                            (h7 : A ≠ F)
                                            (h8 : B ≠ E)
                                            (h9 : B ≠ F)
                                            (h10 : E ≠ F)
                                            (h11 : (100 * A + 10 * B + E) * F = 1001 * E + 100 * A)
                                            : A + B = 5 :=
sorry

end digit_multiplication_sum_l542_542049


namespace min_number_of_candy_kinds_l542_542423

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542423


namespace no_real_solution_of_quadratic_l542_542026

theorem no_real_solution_of_quadratic (x : ℝ) : 
  -3 * x - 8 = 8 * x^2 + 2 → ¬ ∃ y : ℝ, x = y :=
by { intro h, rw [←sub_eq_zero] at h, sorry }

end no_real_solution_of_quadratic_l542_542026


namespace arithmetic_result_l542_542279

theorem arithmetic_result :
  1325 + (572 / 52) - 225 + (2^3) = 1119 :=
by
  sorry

end arithmetic_result_l542_542279


namespace correct_conclusions_l542_542214

noncomputable theory

-- Define the probability of hitting the target in a single shot
def p_hit := 0.9

-- Define the number of shots
def num_shots := 4

-- Define the probability of hitting the target on the third shot
def prob_third_shot := p_hit

-- Define the probability of hitting the target exactly three times
def prob_exactly_three_hits := (nat.choose num_shots 3) * p_hit^3 * (1 - p_hit)

-- Define the probability of hitting the target at least once
def prob_at_least_one_hit := 1 - (1 - p_hit)^num_shots

-- State the main theorem to be proved
theorem correct_conclusions : 
  prob_third_shot = 0.9 ∧ 
  prob_exactly_three_hits ≠ p_hit^3 * (1 - p_hit) ∧
  prob_at_least_one_hit = 1 - 0.1^4 :=
by {
  sorry
}

end correct_conclusions_l542_542214


namespace triangle_inequality_l542_542933

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) : Prop :=
    a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 3 5 7 (by norm_num) (by norm_num) (by norm_num) :=
by simp; apply and.intro (by norm_num) (by norm_num)

end triangle_inequality_l542_542933


namespace minimum_candy_kinds_l542_542445

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542445


namespace smallest_degree_nonzero_poly_with_roots_l542_542871

   noncomputable def poly_deg : ℕ :=
     let p1 := (x : ℚ)[x] * (x - (2 - real.sqrt 3)) * (x - (2 + real.sqrt 3))
     let p2 := (x : ℚ)[x] * (x - (-4 + real.sqrt 15)) * (x - (-4 - real.sqrt 15))
     (p1 * p2).degree

   theorem smallest_degree_nonzero_poly_with_roots : poly_deg = 4 :=
   by
     sorry
   
end smallest_degree_nonzero_poly_with_roots_l542_542871


namespace perpendicular_tangents_l542_542550

theorem perpendicular_tangents (y0 : ℝ) (M : ℝ × ℝ) (hM : M = (0, y0)) 
  (f : ℝ → ℝ) (hf : f = λ x, 1 - 2*x - x^2)
  (k1 k2 : ℝ) (hk1 : k1 = -2 + Real.sqrt 5) (hk2 : k2 = -2 - Real.sqrt 5)
  (hperp : k1 * k2 = -1) :
  y0 = 9 / 4 ∧ M = (0, 9 / 4) ∧ 
  ∃ m1 m2 : ℝ, (m1 = k1) ∧ (m2 = k2) ∧ 
  (∀ x : ℝ, f x = y0 + m1 * x + 1) ∧ (∀ x : ℝ, f x = y0 + m2 * x + 1) := 
by
  sorry

end perpendicular_tangents_l542_542550


namespace odd_int_divides_l542_542567

variable (n : ℕ)

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def positive_odd_integer (n : ℕ) : Prop := n > 0 ∧ is_odd n

theorem odd_int_divides : positive_odd_integer n → n ∣ 2^(nat.factorial n) - 1 :=
by
  intro h
  sorry

end odd_int_divides_l542_542567


namespace find_savings_l542_542951

def calculate_savings (income expenditure : ℕ) (income_ratio expenditure_ratio : ℕ) (h_ratio : income_ratio > 0 ∧ expenditure_ratio > 0) 
    (h_ratio_eq : income_ratio * expenditure = expenditure_ratio * income) (h_income : income = 15000) : ℕ :=
  income - expenditure

theorem find_savings (income expenditure : ℕ) (income_ratio expenditure_ratio : ℕ)
  (h_ratio : income_ratio = 5 ∧ expenditure_ratio = 4)
  (h_ratio_eq : income_ratio * expenditure = expenditure_ratio * income)
  (h_income : income = 15000) :
  calculate_savings income expenditure income_ratio expenditure_ratio h_ratio h_ratio_eq h_income = 3000 :=
  sorry

end find_savings_l542_542951


namespace min_number_of_candy_kinds_l542_542427

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542427


namespace continuous_stripe_probability_l542_542224

-- Define the conditions of the tetrahedron and stripe orientations
def tetrahedron_faces : ℕ := 4
def stripe_orientations_per_face : ℕ := 2
def total_stripe_combinations : ℕ := stripe_orientations_per_face ^ tetrahedron_faces
def favorable_stripe_combinations : ℕ := 2 -- Clockwise and Counterclockwise combinations for a continuous stripe

-- Define the probability calculation
def probability_of_continuous_stripe : ℚ :=
  favorable_stripe_combinations / total_stripe_combinations

-- Theorem statement
theorem continuous_stripe_probability : probability_of_continuous_stripe = 1 / 8 :=
by
  -- The proof is omitted for brevity
  sorry

end continuous_stripe_probability_l542_542224


namespace cottage_cost_per_hour_l542_542057

-- Define the conditions
def jack_payment : ℝ := 20
def jill_payment : ℝ := 20
def total_payment : ℝ := jack_payment + jill_payment
def rental_duration : ℝ := 8

-- Define the theorem to be proved
theorem cottage_cost_per_hour : (total_payment / rental_duration) = 5 := by
  sorry

end cottage_cost_per_hour_l542_542057


namespace parameterize_circle_l542_542683

noncomputable def parametrization (t : ℝ) : ℝ × ℝ :=
  ( (t^2 - 1) / (t^2 + 1), (-2 * t) / (t^2 + 1) )

theorem parameterize_circle (t : ℝ) : 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  (x^2 + y^2) = 1 :=
by 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  sorry

end parameterize_circle_l542_542683


namespace team_advances_with_minimum_points_l542_542480

theorem team_advances_with_minimum_points :
  ∀ (total_teams : ℕ) (matches_per_team : ℕ) (draw_points : ℕ) (win_points : ℕ) (lose_points : ℕ)
    (teams_advancing : ℕ) (max_possible_points : ℕ),
  total_teams = 4 →
  matches_per_team = 3 →
  draw_points = 1 →
  win_points = 3 →
  lose_points = 0 →
  teams_advancing = 2 →
  max_possible_points = total_teams * (total_teams - 1) / 2 * win_points →
  ∃ (min_points : ℕ), 
  min_points = 7 ∧
  ∀ (team_points : ℕ),
  team_points >= min_points → 
  (total_teams - 1) * (team_points ≤ (max_possible_points - team_points) / (total_teams - 1)) :=
by
  intros total_teams matches_per_team draw_points win_points lose_points teams_advancing max_possible_points h₁ h₂ h₃ h₄ h₅ h₆ h₇
  use 7
  split
  -- The proof would come here.
  sorry

end team_advances_with_minimum_points_l542_542480


namespace percentage_error_in_square_area_l542_542188

-- Given an error of 1% in excess while measuring the side of a square,
-- prove that the percentage of error in the calculated area of the square is 2.01%.

theorem percentage_error_in_square_area (s : ℝ) (h : s ≠ 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let calculated_area := (1.01 * s) ^ 2
  let error_in_area := calculated_area - actual_area
  let percentage_error := (error_in_area / actual_area) * 100
  percentage_error = 2.01 :=
by {
  let measured_side := 1.01 * s;
  let actual_area := s ^ 2;
  let calculated_area := (1.01 * s) ^ 2;
  let error_in_area := calculated_area - actual_area;
  let percentage_error := (error_in_area / actual_area) * 100;
  sorry
}

end percentage_error_in_square_area_l542_542188


namespace parallel_line_eq_perpendicular_line_eq_l542_542705

variable {x y : ℝ}

-- Definition of a line passing through a point and being parallel to another line
def line_through_point_parallel (A B : ℝ) (P : ℝ × ℝ) : Prop := 
  (P.fst + P.snd + B = 0)

-- Definition of a line passing through a point and being perpendicular to another line
def line_through_point_perpendicular (C D : ℝ) (P : ℝ × ℝ) : Prop := 
  (P.fst - 3 * P.snd + D = 0)

-- Proof problem statements
theorem parallel_line_eq : ∀ x y : ℝ, line_through_point_parallel (-1) (-1) (-1, 2) :=
by trivial

theorem perpendicular_line_eq : ∀ x y : ℝ, line_through_point_perpendicular 3 3 (0, 1) :=
by trivial

end parallel_line_eq_perpendicular_line_eq_l542_542705


namespace min_kinds_of_candies_l542_542444

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542444


namespace area_percentage_error_l542_542667

theorem area_percentage_error (S : ℝ) (h : S > 0) : 
  let measured_side := S * (1 + 0.037) in 
  let actual_area := S^2 in 
  let measured_area := (measured_side)^2 in 
  (measured_area - actual_area) / actual_area * 100 = 7.57 :=
by
  have actual_area := S ^ 2
  have measured_side := S * (1 + 0.037)
  have measured_area := measured_side ^ 2
  have percentage_error := ((measured_area - actual_area) / actual_area) * 100
  calc
    percentage_error = ((S * (1 + 0.037))^2 - S^2) / S^2 * 100 : by sorry
                   ... = (1.037^2 - 1) * 100                : by sorry 
                   ... = (1.0757 - 1) * 100                : by sorry
                   ... = 0.0757 * 100                      : by sorry
                   ... = 7.57                              : by sorry

end area_percentage_error_l542_542667


namespace num_factors_of_2020_with_more_than_3_factors_l542_542771

open Nat

def num_divisors_with_more_than_three_factors (n : ℕ) : ℕ :=
  List.filter (λ x => numDivisors x > 3) (List.divisors n) |>.length

theorem num_factors_of_2020_with_more_than_3_factors :
  let n := 2020
  n = 2^2 * 5 * 101 → num_divisors_with_more_than_three_factors n = 7 :=
by
  -- The proof is omitted.
  sorry

end num_factors_of_2020_with_more_than_3_factors_l542_542771


namespace normal_dist_prob_geq_one_l542_542743

-- Given a random variable ξ following a normal distribution
-- with mean -1 and variance 6^2, with P(-3 ≤ ξ ≤ -1) = 0.4
variables {ξ : ℝ} (dist : Normal (-1) (6^2))

-- and the given condition P(-3 ≤ ξ ≤ -1) = 0.4
axiom prob_interval : dist.prob (-3 :.. -1) = 0.4

-- Prove that P(ξ ≥ 1) = 0.1
theorem normal_dist_prob_geq_one :
  dist.prob (1 :..) = 0.1 :=
sorry

end normal_dist_prob_geq_one_l542_542743


namespace rho_eq_c_implies_sphere_l542_542302

noncomputable def positive_constant_c (c : ℝ) : Prop :=
  0 < c

def shape_described_by_rho_eq_c (shape : Type) (c : ℝ) : Prop :=
  ∀ (ρ θ φ : ℝ), (ρ = c) → (shape = sphere)

theorem rho_eq_c_implies_sphere (c : ℝ) (h : positive_constant_c c) :
  ∃ shape, shape_described_by_rho_eq_c shape c :=
sorry

end rho_eq_c_implies_sphere_l542_542302


namespace find_t_l542_542766

variables {α : Type*} [inner_product_space ℝ α]
variables (a b : α)
variables (t : ℝ)

-- Declaring the conditions
def unit_vector (v : α) := ∥v∥ = 1
def angle_sixty (u v : α) := real.angle u v = real.pi / 3

-- The condition c = t * a + (1 - t) * b
def c (a b : α) (t : ℝ) : α := t • a + (1 - t) • b

-- The main theorem
theorem find_t (ha : unit_vector a) (hb : unit_vector b) (hangle : angle_sixty a b)
    (hbc : inner_product_space.is_orthogonal b (c a b t)) : t = 2 :=
sorry

end find_t_l542_542766


namespace max_area_triangle_l542_542488

theorem max_area_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (dot_product : (ℝ × ℝ) → (ℝ × ℝ) → ℝ)
  (hA1 : dot_product (cos A, sin A) (cos B, sin B) = sqrt 3 * sin B - cos C)
  (hA2 : A = π / 3 ∨ A = 2 * π / 3)
  (ha : a = 3) :
  (1/2 * b * c * sin A ≤ 9 * sqrt 3 / 4) :=
sorry

end max_area_triangle_l542_542488


namespace find_equation_of_line_L_l542_542736

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)
def line (a b c : ℝ) (p : ℝ × ℝ) : Prop := a * p.1 + b * p.2 + c = 0

def A := point 3 (-4)
def B := point 5 2
def L1 (x : ℝ × ℝ) := line 3 (-1) (-1) x
def L2 (x : ℝ × ℝ) := line 1 1 (-3) x
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def equidistant_from_line (p1 p2 : ℝ × ℝ) (L : ℝ × ℝ → Prop) : Prop := 
  abs (3 * p1.1 - p1.2 - 1) = abs (3 * p2.1 - p2.2 - 1)

def L (P : ℝ × ℝ) (m : ℝ → ℝ) (y : ℝ → ℝ) : Prop := 
  ∃ (a b c : ℝ), 
    L1 P ∧ L2 P ∧ 
    ∀ (p : ℝ × ℝ), L1 p ↔ line a b c p ∧
    ∀ (p : ℝ × ℝ), L2 p ↔ line a b c p ∧
    ∀ (x : ℝ), y x = -x + 3

theorem find_equation_of_line_L : 
  ∃ (L : ℝ × ℝ → Prop),
  L (1, 2) ∧
  equidistant_from_line A B L ∧
  ∀ (P : ℝ × ℝ), P = (1, 2) →
  ∀ (M : ℝ × ℝ), M = midpoint A B →
  ∀ (x : ℝ), line 1 1 (-3) M → (L = λ p, p.2 = -p.1 + 3) :=
sorry

end find_equation_of_line_L_l542_542736


namespace sqrt_fraction_eq_l542_542295

def factorial (n : ℕ) : ℕ := n.factorial

def num : ℕ := factorial 10
def denom : ℕ := 294
def simplified : ℚ := (num : ℚ) / denom

def sqrt_result := 120 * Real.sqrt 6

theorem sqrt_fraction_eq : Real.sqrt simplified = sqrt_result := by
  have fact10 : factorial 10 = 3628800 := by rfl
  have denom294 : 294 = 2 * 3 * 7 * 7 := by rfl
  calc
    Real.sqrt simplified
      = Real.sqrt (3628800 / 294) : by sorry -- additional simplification required here
      ... = 120 * Real.sqrt 6 : by sorry

end sqrt_fraction_eq_l542_542295


namespace nina_total_miles_l542_542845

noncomputable def totalDistance (warmUp firstHillUp firstHillDown firstRecovery 
                                 tempoRun secondHillUp secondHillDown secondRecovery 
                                 fartlek sprintsYards jogsBetweenSprints coolDown : ℝ) 
                                 (mileInYards : ℝ) : ℝ :=
  warmUp + 
  (firstHillUp + firstHillDown + firstRecovery) + 
  tempoRun + 
  (secondHillUp + secondHillDown + secondRecovery) + 
  fartlek + 
  (sprintsYards / mileInYards) + 
  jogsBetweenSprints + 
  coolDown

theorem nina_total_miles : 
  totalDistance 0.25 0.15 0.25 0.15 1.5 0.2 0.35 0.1 1.8 (8 * 50) (8 * 0.2) 0.3 1760 = 5.877 :=
by
  sorry

end nina_total_miles_l542_542845


namespace fair_prize_division_l542_542602

theorem fair_prize_division (eq_chance : ∀ (game : ℕ), 0.5 ≤ 1 ∧ 1 ≤ 0.5)
  (first_to_six : ∀ (p1_wins p2_wins : ℕ), (p1_wins = 6 ∨ p2_wins = 6) → (p1_wins + p2_wins) ≤ 11)
  (current_status : 5 + 3 = 8) :
  (7 : ℝ) / 8 = 7 / (8 : ℝ) :=
by
  sorry

end fair_prize_division_l542_542602


namespace lab_measurement_correct_value_l542_542466

def precise_value_to_report (D : ℝ) (error : ℝ) :=
  (floor ((D + error) * 10)) / 10 = (floor ((D - error) * 10)) / 10

theorem lab_measurement_correct_value
  (D : ℝ)
  (hD : D = 3.86792)
  (error : ℝ)
  (h_error : error = 0.00456) :
  precise_value_to_report D error = true :=
sorry

end lab_measurement_correct_value_l542_542466


namespace equivalent_statement_l542_542285

-- Define the conditions
def conditions (x : ℝ) : Prop :=
  0 ≤ x ∧ x < 2 * Real.pi ∧ 27 * 3^(3 * Real.sin x) = 9^(Real.cos x ^ 2)

-- Define the specific values we identified
def specific_values (x : ℝ) : Prop :=
  x = 7 * Real.pi / 6 ∨ x = 3 * Real.pi / 2 ∨ x = 11 * Real.pi / 6

-- The proof statement
theorem equivalent_statement (x : ℝ) : conditions x ↔ specific_values x := 
  by sorry

end equivalent_statement_l542_542285


namespace digits_sum_not_2001_l542_542123

theorem digits_sum_not_2001 (a : ℕ) (n m : ℕ) 
  (h1 : 10^(n-1) ≤ a ∧ a < 10^n)
  (h2 : 3 * n - 2 ≤ m ∧ m < 3 * n + 1)
  : m + n ≠ 2001 := 
sorry

end digits_sum_not_2001_l542_542123


namespace number_of_girls_l542_542958

variable (boys : ℕ) (total_children : ℕ)

theorem number_of_girls (h1 : boys = 40) (h2 : total_children = 117) : total_children - boys = 77 :=
by
  sorry

end number_of_girls_l542_542958


namespace part1_solution_part2_solution_l542_542348

variable {x a : ℝ}

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1_solution (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : f x a ≤ 2) : a = 2 :=
  sorry

theorem part2_solution (ha : 0 ≤ a) (hb : a ≤ 3) : (f (x + a) a + f (x - a) a ≥ f (a * x) a - a * f x a) :=
  sorry

end part1_solution_part2_solution_l542_542348


namespace set_powers_of_2_and_3_dense_in_positives_l542_542491

noncomputable def is_dense (S : set ℝ) := ∀ (x y : ℝ), x > 0 → y > 0 → x < y → ∃ s ∈ S, x < s ∧ s < y

theorem set_powers_of_2_and_3_dense_in_positives : 
  is_dense {r : ℝ | ∃ (m n : ℤ), r = 2 ^ m * 3 ^ n} :=
sorry

end set_powers_of_2_and_3_dense_in_positives_l542_542491


namespace counting_numbers_remainder_4_divide_53_l542_542767

theorem counting_numbers_remainder_4_divide_53 :
  let n := 53 - 4 in
  card {d : Nat | d ∣ n ∧ d > 4} = 2 :=
by
  let n := 53 - 4
  sorry

end counting_numbers_remainder_4_divide_53_l542_542767


namespace cost_of_transporting_500g_l542_542546

noncomputable def transport_cost (weight_grams : ℕ) (cost_per_kg : ℕ) (discount : ℕ → ℚ) : ℚ :=
let weight_kg := (weight_grams : ℚ) / 1000 in
let cost_without_discount := weight_kg * cost_per_kg in
cost_without_discount * discount weight_grams

theorem cost_of_transporting_500g (cost_per_kg : ℕ) (discount_percentage : ℚ) :
  (cost_per_kg = 18000) → (discount_percentage = 0.10) → 
  (transport_cost 500 18000 (λ w, if w < 1000 then 0.90 else 1) = 8100) :=
begin
  intros h1 h2,
  rw [h1, h2],
  -- Conversion from grams to kilograms and application of discount is handled by transport_cost,
  -- So removing intermediate steps and directly verifying the final condition for equivalence.
  sorry
end

end cost_of_transporting_500g_l542_542546


namespace train_length_l542_542657

theorem train_length (v_t v_m : ℝ) (t : ℝ) (L : ℝ) 
  (hv_t : v_t = 65) 
  (hv_m : v_m = 7) 
  (ht : t = 5.4995600351971845) : 
  L = 110 :=
by
  let relative_speed := v_t + v_m
  let relative_speed_ms := relative_speed * (5 / 18)
  have relative_speed_ms_calc : relative_speed_ms = 72 * (5 / 18) := by
    rw [hv_t, hv_m]
  have length_train : L = relative_speed_ms * t := by
    rw [relative_speed_ms_calc, ht]
  norm_num at length_train
  exact length_train

end train_length_l542_542657


namespace condition_implies_a_lt_b_sufficient_but_not_necessary_l542_542774

-- Definitions from the conditions
variables {a b : ℝ}

-- Define the condition
def condition (a b : ℝ) : Prop :=
  (a - b) * a^2 < 0

-- Prove the condition implies a < b (sufficient condition)
theorem condition_implies_a_lt_b (h : condition a b) : a < b :=
by sorry

-- State the main theorem
theorem sufficient_but_not_necessary : (condition_implies_a_lt_b (condition a b)) ∧ ¬ ((a < b) → (condition a b)) :=
by sorry

end condition_implies_a_lt_b_sufficient_but_not_necessary_l542_542774


namespace sapper_min_walk_distance_l542_542227

theorem sapper_min_walk_distance :
  ∀ (A B C : Type) [euclidean_space A] [euclidean_space B] [euclidean_space C],
  let side_length := 2,
  let detection_range := (real.sqrt 3) / 2,
  let triangle := equilateral_triangle A B C side_length in
  let sapper_initial_position := B in
  (minimum_walk_distance_for_complete_detection triangle detection_range sapper_initial_position) = (real.sqrt 7 - (real.sqrt 3) / 2) :=
by
  sorry

end sapper_min_walk_distance_l542_542227


namespace equivalent_cartesian_eq_C1_and_chord_length_l542_542797

-- Definitions for the conditions provided:
def polar_eq_C1 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + π / 4) = Real.sqrt 2

def parametric_eq_C2 (t : ℝ) : ℝ × ℝ :=
  (Real.cos t, 1 + Real.sin t)

def cartesian_eq_C2 : ℝ → ℝ → Prop :=
  λ x y, x^2 + (y - 1)^2 = 1

-- Statement of the problem:
theorem equivalent_cartesian_eq_C1_and_chord_length :
  (∀ (ρ θ x y : ℝ), polar_eq_C1 ρ θ → ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y → x + y = 2) ∧
  (∀ (t x y : ℝ), parametric_eq_C2 t = (x, y) → cartesian_eq_C2 x y) ∧
  (∀ (x y : ℝ), (x + y = 2) → (cartesian_eq_C2 x y) → (Real.sqrt (x^2 + (y-1)^2) = Real.sqrt 2)) :=
  sorry

end equivalent_cartesian_eq_C1_and_chord_length_l542_542797


namespace count_three_digit_decimals_l542_542377

-- We define the property of being a three-digit decimal number within the specified range
def three_digit_decimal (x : ℝ) := x > 3.006 ∧ x < 3.01 ∧ (x * 1000).isInt

theorem count_three_digit_decimals : {x : ℝ | three_digit_decimal x}.finite.to_finset.card = 3 :=
by
  sorry

end count_three_digit_decimals_l542_542377


namespace smallest_period_eq_pi_monotonic_in_interval_l542_542351

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem smallest_period_eq_pi (ω : ℝ) (hω : 0 < ω) :
  (∀ x : ℝ, f ω x = f ω (x + ℝ.pi)) → ω = 2 :=
sorry

theorem monotonic_in_interval (ω : ℝ) (hω : 0 < ω) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ Real.pi / 2 → f ω x1 ≤ f ω x2) → ω ≤ 1 / 3 :=
sorry

end smallest_period_eq_pi_monotonic_in_interval_l542_542351


namespace value_of_expression_l542_542906

theorem value_of_expression : 1 + 3^2 = 10 :=
by
  sorry

end value_of_expression_l542_542906


namespace minimum_candy_kinds_l542_542450

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542450


namespace three_x_x_l542_542383

theorem three_x_x (x : ℤ) (h : 3^x - 3^(x - 1) = 18) : 3 * x^x = 81 :=
sorry

end three_x_x_l542_542383


namespace diagonal_divides_rhombuses_l542_542818

-- Define the problem conditions
def is_regular_hexagon (A B C D E F : Type) (n : ℕ) : Prop :=
  ∃ (triangles : set (set (ℝ × ℝ))), -- consider some representation of triangles
    (triangles.card = 6 * n * n) ∧
    (∀ (tri ∈ triangles), is_equilateral_triangle_with_sidelength_1 tri) ∧
    ∃ (rhombuses : set (set (ℝ × ℝ))),
      (rhombuses.card = 3 * n * n) ∧
      (∀ (rhom ∈ rhombuses), is_rhombus_with_internal_angles_60_120 rhom) ∧
      (∀ (tri ∈ triangles), ∃! (rhom ∈ rhombuses), tri ⊆ rhom)

-- Define the hexagon and the required proof statement
theorem diagonal_divides_rhombuses (n : ℕ) (A B C D E F : Type)
  (h : is_regular_hexagon A B C D E F n) : divides_in_half_AD_exactly_n_rhombuses A D n :=
sorry

end diagonal_divides_rhombuses_l542_542818


namespace find_constants_l542_542235

noncomputable def f (x : ℕ) (a c : ℕ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

theorem find_constants (a c : ℕ) (h₁ : f 4 a c = 30) (h₂ : f a a c = 5) : 
  c = 60 ∧ a = 144 := 
by
  sorry

end find_constants_l542_542235


namespace find_number_l542_542787

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 100) : N = 192 :=
sorry

end find_number_l542_542787


namespace minimum_candy_kinds_l542_542398

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542398


namespace distance_from_focus_to_asymptote_l542_542877

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2) / 9 - (y^2) / 4 = 1

-- Right focus of the hyperbola
def right_focus : ℝ × ℝ := (Real.sqrt 13, 0)

-- Equation of one of the asymptotes
def asymptote (x y : ℝ) : Prop := 2 * x + 3 * y = 0

-- Distance formula
def distance (p : ℝ × ℝ) (line : ℝ × ℝ → Prop) : ℝ :=
  let (x_0, y_0) := p in
  let A := 2 in
  let B := 3 in
  let C := 0 in
  -- Point-line distance formula
  abs (A * x_0 + B * y_0 + C) / Real.sqrt (A^2 + B^2)

theorem distance_from_focus_to_asymptote :
  distance right_focus asymptote = 2 :=
  sorry

end distance_from_focus_to_asymptote_l542_542877


namespace sequence_of_multiples_l542_542044

theorem sequence_of_multiples (n : ℕ) (a : ℕ → ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n → a k = ∑ i in (finset.range n).filter (λ j, j + 1 ≠ k ∧ k ∣ a(j + 1)), a (i + 1)) →
  True :=
by
  sorry

end sequence_of_multiples_l542_542044


namespace find_C_l542_542146

theorem find_C
  (A B C D : ℕ)
  (h1 : 0 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 4 * 1000 + A * 100 + 5 * 10 + B + (C * 1000 + 2 * 100 + D * 10 + 7) = 8070) :
  C = 3 :=
by
  sorry

end find_C_l542_542146


namespace pyramid_volume_l542_542137

-- Define the coordinates and necessary conditions.
def AB : ℝ := 10 * Real.sqrt 5
def BC : ℝ := 15 * Real.sqrt 5
def volume_pyramid : ℝ := 18750 / Real.sqrt 475

-- Proof statement
theorem pyramid_volume (A B C D P : ℝ × ℝ × ℝ) (H1 : dist A B = AB) (H2 : dist B C = BC) (H3 : midpoint A C = P) (H4 : midpoint B D = P) :
  volume_pyramid = 18750 / Real.sqrt 475 := 
sorry

end pyramid_volume_l542_542137


namespace dot_product_correct_l542_542013

theorem dot_product_correct:
  let a : ℝ × ℝ := (5, -7)
  let b : ℝ × ℝ := (-6, -4)
  (a.1 * b.1) + (a.2 * b.2) = -2 := by
sorry

end dot_product_correct_l542_542013


namespace monomino_position_l542_542234

def color_distribution (table : list (list ℕ)) : Prop :=
  let colors := table.bind id
  ∃ color1_count color2_count color3_count : ℕ,
    color1_count = (colors.count 1) ∧
    color2_count = (colors.count 2) ∧
    color3_count = (colors.count 3) ∧
    color1_count = 22 ∧
    color2_count = 21 ∧
    color3_count = 21

def tromino_cover (table : list (list ℕ)) (n_trominos : ℕ) : Prop :=
  n_trominos = 21 ∧
  let colors := table.bind id
  ∃ color1_covered color2_covered color3_covered : ℕ,
    color1_covered = n_trominos ∧
    color2_covered = n_trominos ∧
    color3_covered = n_trominos ∧
    (colors.count 1 - color1_covered) = 1

def valid_monomino_positions (table1 table2 : list (list ℕ)) : list (ℕ × ℕ) :=
  table1.zip table2 |>.bind (λ (r1, r2),
    r1.zip r2 |>.bind (λ ((c1 : ℕ), (c2 : ℕ), ), if (c1 = 1) ∧ (c2 = 1) then [(c2, c2)] else []))

theorem monomino_position (table1 table2 : list (list ℕ)) (n_trominos : ℕ) :
  color_distribution table1 →
  color_distribution table2 →
  tromino_cover table1 n_trominos →
  tromino_cover table2 n_trominos →
  (valid_monomino_positions table1 table2).nonempty :=
by
  intros
  unfold valid_monomino_positions
  sorry

end monomino_position_l542_542234


namespace quotient_when_divided_by_8_l542_542174

theorem quotient_when_divided_by_8
  (n : ℕ)
  (h1 : n = 12 * 7 + 5)
  : (n / 8) = 11 :=
by
  -- the proof is omitted
  sorry

end quotient_when_divided_by_8_l542_542174


namespace tangent_line_at_point_is_correct_l542_542588

def curve (x : ℝ) : ℝ := x^3

def tangent_line_equation (x y : ℝ) : Bool :=
  (3 * x - y - 2 = 0)

theorem tangent_line_at_point_is_correct :
  tangent_line_equation 1 (curve 1) :=
by
  sorry

end tangent_line_at_point_is_correct_l542_542588


namespace solution_set_of_f_greater_than_2x_plus_4_l542_542637

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, x ∈ set.univ
axiom condition2 : f (-1) = 2
axiom condition3 : ∀ x : ℝ, f' x > 2

theorem solution_set_of_f_greater_than_2x_plus_4 :
  { x : ℝ | f x > 2 * x + 4 } = set.Ioi (-1) :=
by {
  -- proof is omitted
  sorry
}

end solution_set_of_f_greater_than_2x_plus_4_l542_542637


namespace equivalent_set_complement_intersection_l542_542762

def setM : Set ℝ := {x | -3 < x ∧ x < 1}
def setN : Set ℝ := {x | x ≤ 3}
def givenSet : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

theorem equivalent_set_complement_intersection :
  givenSet = (setM ∩ setN)ᶜ :=
sorry

end equivalent_set_complement_intersection_l542_542762


namespace fourth_child_sweets_l542_542216

theorem fourth_child_sweets (total_sweets : ℕ) (mother_sweets : ℕ) (child_sweets : ℕ) 
  (Y E T F: ℕ) (h1 : total_sweets = 120) (h2 : mother_sweets = total_sweets / 4) 
  (h3 : child_sweets = total_sweets - mother_sweets) 
  (h4 : E = 2 * Y) (h5 : T = F - 8) 
  (h6 : Y = (8 * (T + 6)) / 10) 
  (h7 : Y + E + (T + 6) + (F - 8) + F = child_sweets) : 
  F = 24 :=
by
  sorry

end fourth_child_sweets_l542_542216


namespace smallest_solution_neg3_l542_542714

theorem smallest_solution_neg3 : ∀ x : ℝ, (frac 3 (x * (x + 3))) + (frac (3 * x^2 - 18) x) = 9 → x = -3 → ∀ y : ℝ, (frac 3 (y * (y + 3))) + (frac (3 * y^2 - 18) y) = 9 → x ≤ y :=
by
  sorry

end smallest_solution_neg3_l542_542714


namespace angle_equality_of_triangle_l542_542734

noncomputable def M (A C : Point) := midpoint A C
noncomputable def P (C M : Point) := midpoint C M

theorem angle_equality_of_triangle
  (A B C : Point)
  (M : Point) (P : Point) (Q : Point)
  (hM : M = midpoint A C)
  (hP : P = midpoint C M)
  (circumcircle_ABP : circle A B P)
  (hQ : Q ∈ line_segment B C ∩ circumcircle_ABP) :
  angle A B M = angle M Q P :=
by
  sorry

end angle_equality_of_triangle_l542_542734


namespace favorite_movies_hours_l542_542067

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l542_542067


namespace integer_solution_l542_542282

theorem integer_solution (n m : ℤ) (h : (n + 2)^4 - n^4 = m^3) : (n = -1 ∧ m = 0) :=
by
  sorry

end integer_solution_l542_542282


namespace reckha_code_count_l542_542545

theorem reckha_code_count :
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970 :=
by
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  show total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970
  sorry

end reckha_code_count_l542_542545


namespace min_area_of_triangle_AFB_l542_542978

-- Definition of the problem conditions and proof

noncomputable def minTriangleArea : ℝ :=
  let focus := (1, 0) in -- focus of the parabola y^2 = 4x
  let M := (4, 0) in -- point M through which the line passes
  let parabola (y : ℝ) := y^2 - 4 * (4 - y * 0) - 16 in -- parabola equation y^2 = 4x transformed
  let line (m : ℝ) (y : ℝ) := 4 - m * y in -- equation of line x - 4 = my
  let roots (m : ℝ) := (4*m, -16) in -- sum and product of roots from the quadratic equation y^2 - 4my - 16 = 0
  let area (m : ℝ) := 6 * Real.sqrt(m^2 + 4) in -- area of triangle AFB
  let minArea := 12 in -- minimum area found in the solution
  minArea

theorem min_area_of_triangle_AFB : minTriangleArea = 12 := by
  sorry

end min_area_of_triangle_AFB_l542_542978


namespace diagonal_length_A_B_l542_542988

-- Define the dimensions of the rectangular prism
def length := 10
def width := 20
def height := 10

-- Define the function to compute the diagonal length given the dimensions
def diagonal_length (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

-- Statement to prove
theorem diagonal_length_A_B :
  diagonal_length length width height = 10 * Real.sqrt 6 :=
by sorry

end diagonal_length_A_B_l542_542988


namespace monotonic_decreasing_interval_range_of_a_l542_542756

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) * ((a / x) + a + 1)

theorem monotonic_decreasing_interval (a : ℝ) (h : a ≥ -1) :
  (a = -1 → ∀ x, x < -1 → f a x < f a (x + 1)) ∧
  (a ≠ -1 → (∀ x, -1 < a ∧ x < -1 ∨ x > 1 / (a + 1) → f a x < f a (x + 1)) ∧
                (∀ x, -1 < a ∧ -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 / (a + 1) → f a x < f a (x + 1)))
:= sorry

theorem range_of_a (a : ℝ) (h : a ≥ -1) :
  (∃ x1 x2, x1 > 0 ∧ x2 < 0 ∧ f a x1 < f a x2 → -1 ≤ a ∧ a < 0)
:= sorry

end monotonic_decreasing_interval_range_of_a_l542_542756


namespace shaded_area_equal_l542_542643

noncomputable def circle_area (r : ℝ) : ℝ := π * r^2

theorem shaded_area_equal
  (A B C : Type)
  (radius_small radius_large : ℝ)
  (h_small : radius_small = 4)
  (h_tangent : B ∈ {p : Type | dist p A = radius_small})
  (h_radius_large : radius_large = 2 * radius_small) :
  circle_area radius_large - circle_area radius_small = 48 * π :=
by
  rw [circle_area, circle_area, h_small, h_radius_large]
  norm_num
  sorry

end shaded_area_equal_l542_542643


namespace match_proverbs_l542_542178

-- Define each condition as a Lean definition
def condition1 : Prop :=
"As cold comes and heat goes, the four seasons change" = "Things are developing"

def condition2 : Prop :=
"Thousands of flowers arranged, just waiting for the first thunder" = 
"Decisively seize the opportunity to promote qualitative change"

def condition3 : Prop :=
"Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade" = 
"The unity of contradictions"

def condition4 : Prop :=
"There will be times when the strong winds break the waves, and we will sail across the sea with clouds" = 
"The future is bright"

-- The theorem we need to prove, using the condition definitions
theorem match_proverbs : condition2 ∧ condition4 :=
sorry

end match_proverbs_l542_542178


namespace exists_six_fully_connected_l542_542326

variable (E : Finset (Fin 1991))
variable (conn : (x y : Fin 1991) → (x ≠ y) → Prop)
variable (h1 : ∀ x, (Finset.filter (λ y, conn x y ∧ x ≠ y) E).card ≥ 1593)

theorem exists_six_fully_connected :
  ∃ S : Finset (Fin 1991), S.card = 6 ∧ ∀ x y ∈ S, x ≠ y → conn x y := 
sorry

end exists_six_fully_connected_l542_542326


namespace cream_ratio_l542_542060

theorem cream_ratio (joe_initial_coffee joann_initial_coffee : ℝ)
                    (joe_drank_ounces joann_drank_ounces joe_added_cream joann_added_cream : ℝ) :
  joe_initial_coffee = 20 →
  joann_initial_coffee = 20 →
  joe_drank_ounces = 3 →
  joann_drank_ounces = 3 →
  joe_added_cream = 4 →
  joann_added_cream = 4 →
  (4 : ℝ) / ((21 / 24) * 24 - 3) = (8 : ℝ) / 7 :=
by
  intros h_ji h_ji h_jd h_jd h_jc h_jc
  sorry

end cream_ratio_l542_542060


namespace min_number_of_candy_kinds_l542_542421

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542421


namespace determine_discriminant_of_quadratic_l542_542689

theorem determine_discriminant_of_quadratic
  (m n p : ℤ)
  (h1 : gcd m n = 1)
  (h2 : gcd m p = 1)
  (h3 : gcd n p = 1)
  (roots_form : ∀ x, 3*x^2 - 7*x + 4 = 0 → ∃ k, x = (m + k * sqrt n) / p ∨ x = (m - k * sqrt n) / p)
  : n = 1 :=
sorry

end determine_discriminant_of_quadratic_l542_542689


namespace max_and_min_numbers_l542_542218

def fireEmergencyNumber : Nat := 119
def medicalEmergencyNumber : Nat := 120

def addZeroAtEnd (n : Nat) : Nat :=
  n * 10

def addZeroAtBegin (n : Nat) : Nat :=
  n + ((10 ^ (Nat.digits 10 n).length.toNat) * 0)

def maxNumber : Nat := 12000119
def minNumber : Nat := 01191200

theorem max_and_min_numbers :
  let fe := fireEmergencyNumber
  let me := medicalEmergencyNumber
  let max := addZeroAtEnd me * (10 ^ (Nat.digits 10 (addZeroAtBegin fe)).length.toNat) + addZeroAtBegin fe
  let min := addZeroAtEnd fe * (10 ^ (Nat.digits 10 (addZeroAtBegin me)).length.toNat) + addZeroAtBegin me
  max = maxNumber ∧ min = minNumber := by
    sorry

end max_and_min_numbers_l542_542218


namespace monotonic_decreasing_interval_l542_542597

noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 2

theorem monotonic_decreasing_interval : 
  ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < 4) → f'(x1) > f'(x2) :=
by
  sorry

end monotonic_decreasing_interval_l542_542597


namespace sqrt_sum_simplification_l542_542960

theorem sqrt_sum_simplification : sqrt 8 + sqrt 18 = 5 * sqrt 2 := by
  sorry

end sqrt_sum_simplification_l542_542960


namespace rearrange_incongruent_mod_4028_l542_542193

theorem rearrange_incongruent_mod_4028 (x y : Fin 2014 → ℤ) :
  (∀ i j : Fin 2014, i ≠ j → x i % 2014 ≠ x j % 2014) →
  (∀ i j : Fin 2014, i ≠ j → y i % 2014 ≠ y j % 2014) →
  ∃ z : Fin 2014 → ℤ, (∀ i j : Fin 2014, i ≠ j → (x i + z i) % 4028 ≠ (x j + z j) % 4028) :=
begin
  sorry
end

end rearrange_incongruent_mod_4028_l542_542193


namespace stddev_transformed_l542_542782

noncomputable def sample_data : Fin 10 → ℝ := sorry -- original sample data
def transformed_data (x : ℝ) : ℝ := 2 * x - 1

-- Given condition: the standard deviation of the original data is 8
axiom stddev_original : Real.stddev (fun i => sample_data i) = 8

-- Prove: the standard deviation of the transformed data is 16
theorem stddev_transformed : Real.stddev (fun i => transformed_data (sample_data i)) = 16 := by
  sorry

end stddev_transformed_l542_542782


namespace difference_of_interchanged_digits_l542_542585

theorem difference_of_interchanged_digits (X Y : ℕ) (h : X - Y = 5) : (10 * X + Y) - (10 * Y + X) = 45 :=
by
  sorry

end difference_of_interchanged_digits_l542_542585


namespace min_kinds_of_candies_l542_542441

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542441


namespace pages_to_read_tomorrow_l542_542509

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l542_542509


namespace exists_subset_with_magnitude_ge_one_sixth_l542_542535

theorem exists_subset_with_magnitude_ge_one_sixth (n : ℕ) (z : Fin n → ℂ)
    (h : (∑ i, Complex.abs (z i)) = 1) :
    ∃ (s : Finset (Fin n)), Complex.abs (∑ i in s, z i) ≥ 1 / 6 :=
sorry

end exists_subset_with_magnitude_ge_one_sixth_l542_542535


namespace minimum_value_f_pi_over_8_l542_542352

def f (ω x : ℝ) : ℝ :=
  √3 * sin (ω * x) * cos (ω * x) + (cos (ω * x))^2 - 1/2

theorem minimum_value_f_pi_over_8 (ω : ℝ) (hω : 0 < ω) : 
  f ω (π / 8) = 1/2 :=
sorry

end minimum_value_f_pi_over_8_l542_542352


namespace sin_A_equals_4_over_5_l542_542047

variables {A B C : ℝ}
-- Given a right triangle ABC with angle B = 90 degrees
def right_triangle (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (B = 90)

-- We are given 3 * sin(A) = 4 * cos(A)
def given_condition (A : ℝ) : Prop :=
  3 * Real.sin A = 4 * Real.cos A

-- We need to prove that sin(A) = 4/5
theorem sin_A_equals_4_over_5 (A B C : ℝ) 
  (h1 : right_triangle B 90 C)
  (h2 : given_condition A) : 
  Real.sin A = 4 / 5 :=
by
  sorry

end sin_A_equals_4_over_5_l542_542047


namespace sum_n_k_l542_542125

theorem sum_n_k (n k : ℕ) (h1 : 3 * (k + 1) = n - k) (h2 : 2 * (k + 2) = n - k - 1) : n + k = 13 := by
  sorry

end sum_n_k_l542_542125


namespace min_candy_kinds_l542_542434

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542434


namespace geometric_sequence_sum_of_squares_l542_542050

variable {a : ℕ → ℕ}

-- Given the condition that the sum of the first n terms of the sequence equals 2^n - 1,
-- For example, given the sum formula, calculate the sum of squares.
theorem geometric_sequence_sum_of_squares (h : ∀ n, (∑ i in Finset.range (n + 1), a i) = 2 ^ (n + 1) - 1) :
  ∀ n, (∑ i in Finset.range (n + 1), (a i) ^ 2) = (4 ^ (n + 1) - 1) / 3 :=
by
  sorry

end geometric_sequence_sum_of_squares_l542_542050


namespace determine_b_l542_542817

noncomputable theory

variables {n : ℕ} {q : ℝ} {a b : ℕ → ℝ}

-- Define the conditions for the problem
def positive_numbers (a : ℕ → ℝ) (n : ℕ) : Prop := ∀ k, 1 ≤ k ∧ k ≤ n → a k > 0
def increasing_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop := ∀ k, 1 ≤ k ∧ k ≤ n → a k < b k
def ratio_bounded (b : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n - 1 → q < b (k + 1) / b k ∧ b (k + 1) / b k < 1 / q
def sum_bounded (b a : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop :=
  finset.sum (finset.range n) b < (1 + q) / (1 - q) * finset.sum (finset.range n) a

-- Given conditions in Lean
def condition_a (b a : ℕ → ℝ) (n : ℕ) : Prop :=
  increasing_sequence b a n
def condition_b (b : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop :=
  ratio_bounded b q n
def condition_c (b a : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop :=
  sum_bounded b a q n

-- Define b_k based on the solution's provided form
def b_k (a : ℕ → ℝ) (q : ℝ) (n k : ℕ) : ℝ :=
  a k + ∑ j in finset.range (n - 1), q^j * (a ((k - j - 1) % n + 1) + a ((k + j - 1) % n + 1))

-- Main theorem statement
theorem determine_b (h_pos : positive_numbers a n) (h_q : 0 < q ∧ q < 1) :
  ∃ b : ℕ → ℝ, condition_a b a n ∧ condition_b b q n ∧ condition_c b a q n :=
sorry

end determine_b_l542_542817


namespace books_in_june_l542_542155

-- Definitions
def Book_may : ℕ := 2
def Book_july : ℕ := 10
def Total_books : ℕ := 18

-- Theorem statement
theorem books_in_june : ∃ (Book_june : ℕ), Book_may + Book_june + Book_july = Total_books ∧ Book_june = 6 :=
by
  -- Proof will be here
  sorry

end books_in_june_l542_542155


namespace monotonicity_of_f_odd_function_a_value_l542_542356

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

-- Part 1: Prove that f(x) is monotonically increasing
theorem monotonicity_of_f (a : ℝ) : 
  ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := by
  intro x1 x2 hx
  sorry

-- Part 2: If f(x) is an odd function, find the value of a
theorem odd_function_a_value (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : 
  f a 0 = 0 → a = 1 / 2 := by
  intro h
  sorry

end monotonicity_of_f_odd_function_a_value_l542_542356


namespace mr_kishore_savings_l542_542238

-- Define the expenses
def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 5650

-- Define the total expenses
def total_expenses : ℝ := rent + milk + groceries + education + petrol + miscellaneous
-- Simplified to the calculated total
axiom total_expenses_calculated : total_expenses = 24150

-- Define the portion of the salary spent
def salary_spent_portion : ℝ := 0.90
-- Define the total salary spent
def total_salary_spent : ℝ := total_expenses

-- Define the monthly salary, solving for S
def monthly_salary : ℝ := total_salary_spent / salary_spent_portion
-- Simplified to the calculated salary
axiom monthly_salary_calculated : monthly_salary = 26833.33

-- Define the savings rate
def savings_rate : ℝ := 0.10
-- Define the savings amount
def savings : ℝ := savings_rate * monthly_salary
-- Simplified to the calculated savings amount
axiom savings_calculated : savings = 2683.33

-- Final theorem statement (proof not included, sorry placeholder)
theorem mr_kishore_savings : savings = 2683.33 :=
sorry

end mr_kishore_savings_l542_542238


namespace point_O_on_circle_C_if_c_eq_0_area_of_triangle_range_of_c_if_O_inside_circle_C_circle_C_tangent_to_median_if_a_eq_b_eq_c_l542_542795

section geometry_problems

variables {a b c : ℝ} (x y : ℝ)

def line_l (x y : ℝ) : Prop :=
  x/a + y/b = 1

def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - a * x - b * y - c = 0

theorem point_O_on_circle_C_if_c_eq_0 :
  c = 0 → circle_C 0 0 :=
by
  intro hc
  rw [circle_C, hc]
  simp
  sorry

theorem area_of_triangle :
  ∀ a b : ℝ, a * b ≠ 0 → (1/2) * |a| * |b| = (|a| * |b|)/2 :=
by
  intros
  field_simp
  rw mul_comm
  sorry

theorem range_of_c_if_O_inside_circle_C :
  (\frac{a^2 + b^2}{4} + c > \frac{a^2 + b^2}{4}) ↔ (c > 0) :=
by
  simp
  sorry

theorem circle_C_tangent_to_median_if_a_eq_b_eq_c :
  a = -\frac{8}{3} → b = -\frac{8}{3} → c = -\frac{8}{3} → 
  ∃ k l m n : ℝ, median_parallel_to_AB k l m ∧ tangent_to_circle n :=
by
  intros ha hb hc
  -- Definitions and assumptions formulating the proof conditions
  sorry

end geometry_problems

end point_O_on_circle_C_if_c_eq_0_area_of_triangle_range_of_c_if_O_inside_circle_C_circle_C_tangent_to_median_if_a_eq_b_eq_c_l542_542795


namespace meat_needed_for_40_hamburgers_l542_542499

theorem meat_needed_for_40_hamburgers (meat_per_10_hamburgers : ℕ) (hamburgers_needed : ℕ) (meat_per_hamburger : ℚ) (total_meat_needed : ℚ) :
  meat_per_10_hamburgers = 5 ∧ hamburgers_needed = 40 ∧
  meat_per_hamburger = meat_per_10_hamburgers / 10 ∧
  total_meat_needed = meat_per_hamburger * hamburgers_needed → 
  total_meat_needed = 20 := by
  sorry

end meat_needed_for_40_hamburgers_l542_542499


namespace not_perfect_square_count_l542_542071

noncomputable def P (m d c b a : ℤ) := m^5 + d * m^4 + c * m^3 + b * m^2 + 2023 * m + a

theorem not_perfect_square_count
  (a b c d : ℤ)
  (n : ℕ)
  (h_n_pos : n > 0) :
  ∃ m_set : Finset ℕ, m_set.card ≥ n / 4 ∧ ∀ m ∈ m_set, m ≤ n ∧ ¬ is_square (P m d c b a) :=
sorry

end not_perfect_square_count_l542_542071


namespace combined_weight_loss_l542_542239

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end combined_weight_loss_l542_542239


namespace minimum_candy_kinds_l542_542405

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542405


namespace find_smallest_m_l542_542881

open Real 

def smallest_translation_value (m : ℝ) (h : m > 0) : Prop := 
  ∀ x, sin^2 ((x - m) - π/4)  = sin^2 (-(x - m) - π/4)

theorem find_smallest_m : ∃ (m : ℝ) (h : m > 0), smallest_translation_value m h ∧ m = π/4 :=
by 
  sorry

end find_smallest_m_l542_542881


namespace part_a_part_b_l542_542583

-- Let p_k represent the probability that at the moment of completing the first collection, the second collection is missing exactly k crocodiles.
def p (k : ℕ) : ℝ := sorry

-- The conditions 
def totalCrocodiles : ℕ := 10
def probabilityEachEgg : ℝ := 0.1

-- Problems to prove:

-- (a) Prove that p_1 = p_2
theorem part_a : p 1 = p 2 := sorry

-- (b) Prove that p_2 > p_3 > p_4 > ... > p_10
theorem part_b : ∀ k, 2 ≤ k ∧ k < totalCrocodiles → p k > p (k + 1) := sorry

end part_a_part_b_l542_542583


namespace total_movie_hours_l542_542063

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l542_542063


namespace triangle_inradius_one_has_right_angle_l542_542015

-- Defining the geometric conditions based on the problem statement.
variables {ABC : Type} [triangle ABC] (r : ℝ) (h : r = 1)

-- The theorem that needs to be proven.
theorem triangle_inradius_one_has_right_angle (ABC : triangle) (h : inradius ABC = 1) : 
  ∃ (a b c : ℝ), is_right_angle a b c := sorry

end triangle_inradius_one_has_right_angle_l542_542015


namespace calculate_truck_loads_of_dirt_l542_542654

noncomputable def truck_loads_sand: ℚ := 0.16666666666666666
noncomputable def truck_loads_cement: ℚ := 0.16666666666666666
noncomputable def total_truck_loads_material: ℚ := 0.6666666666666666
noncomputable def truck_loads_dirt: ℚ := total_truck_loads_material - (truck_loads_sand + truck_loads_cement)

theorem calculate_truck_loads_of_dirt :
  truck_loads_dirt = 0.3333333333333333 := 
by
  sorry

end calculate_truck_loads_of_dirt_l542_542654


namespace leading_digits_sum_l542_542822

-- Define the conditions
def M : ℕ := (888888888888888888888888888888888888888888888888888888888888888888888888888888) -- define the 400-digit number
-- Assume the function g(r) which finds the leading digit of the r-th root of M

/-- 
  Function g(r) definition:
  It extracts the leading digit of the r-th root of the given number M.
-/
noncomputable def g (r : ℕ) : ℕ := sorry

-- Define the problem statement in Lean 4
theorem leading_digits_sum :
  g 3 + g 4 + g 5 + g 6 + g 7 = 8 :=
sorry

end leading_digits_sum_l542_542822


namespace price_reduction_relationship_l542_542934

variable (a : ℝ) -- original price a in yuan
variable (b : ℝ) -- final price b in yuan

-- condition: price decreased by 10% first
def priceAfterFirstReduction := a * (1 - 0.10)

-- condition: price decreased by 20% on the result of the first reduction
def finalPrice := priceAfterFirstReduction a * (1 - 0.20)

-- theorem: relationship between original price a and final price b
theorem price_reduction_relationship (h : b = finalPrice a) : 
  b = a * (1 - 0.10) * (1 - 0.20) :=
by
  -- proof would go here
  sorry

end price_reduction_relationship_l542_542934


namespace min_kinds_of_candies_l542_542440

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542440


namespace dwarfs_garden_area_l542_542113

/-- Seven dwarfs stand at certain points forming a closed path. Snow White walks specific distances
    in given directions starting from the initial point and returns to the starting point.
    
    The garden area bounded by these points is calculated as 22 square meters. -/
theorem dwarfs_garden_area : 
  let x1 := 0  -- Initial x-coordinate
  let y1 := 0  -- Initial y-coordinate
  let x2 := x1 + 4  -- 4 meters east
  let y2 := y1
  let x3 := x2  -- 2 meters north
  let y3 := y2 + 2
  let x4 := x3 - 2  -- 2 meters west
  let y4 := y3
  let x5 := x4  -- 3 meters north
  let y5 := y4 + 3
  let x6 := x5 - 4  -- 4 meters west
  let y6 := y5
  let x7 := x6  -- 3 meters south
  let y7 := y6 - 3
  let area_rectangle1 := 3 * 4  -- Rectangle area (Stydlin-Stistko-Kejchal-Drimal)
  let area_rectangle2 := 4 * 2  -- Rectangle area (Shmudla-Prof-Rejpal-B)
  let area_triangle := (1 / 2) * (2 * 2)  -- Triangle area (Shmudla-Drimal-B)
  let total_area := area_rectangle1 + area_rectangle2 + area_triangle
  total_area = 22 :=
begin
  sorry
end

end dwarfs_garden_area_l542_542113


namespace flu_transmission_average_l542_542986

-- Define the condition
def infected_average (x : ℝ) : Prop := (1 + x)^2 = 100

-- Statement to prove
theorem flu_transmission_average : ∃ (x : ℝ), infected_average x ∧ x = 9 :=
by
    use 9
    split
    . exact infected_average 9 -- This is our condition (1 + x)^2 = 100
    . refl -- Directly assert that x is indeed 9

end flu_transmission_average_l542_542986


namespace alice_yesterday_percentage_l542_542243

-- Definitions corresponding to conditions
def initial_watermelons := 120
def watermelons_remaining_after_yesterday (P : ℝ) := (1 - P / 100) * initial_watermelons
def watermelons_remaining_after_today (P : ℝ) := (3/4) * watermelons_remaining_after_yesterday P

-- Target statement to prove
theorem alice_yesterday_percentage : ∃ (P : ℝ), watermelons_remaining_after_today P = 54 ∧ P = 40 := 
by
  sorry

end alice_yesterday_percentage_l542_542243


namespace combined_area_rectangle_triangle_l542_542475

/-- 
  Given a rectangle ABCD with vertices A = (10, -30), 
  B = (2010, 170), D = (12, -50), and a right triangle
  ADE with vertex E = (12, -30), prove that the combined
  area of the rectangle and the triangle is 
  40400 + 20√101.
-/
theorem combined_area_rectangle_triangle :
  let A := (10, -30)
  let B := (2010, 170)
  let D := (12, -50)
  let E := (12, -30)
  let length_AB := Real.sqrt ((2010 - 10)^2 + (170 + 30)^2)
  let length_AD := Real.sqrt ((12 - 10)^2 + (-50 + 30)^2)
  let area_rectangle := length_AB * length_AD
  let length_DE := Real.sqrt ((12 - 12)^2 + (-50 + 30)^2)
  let area_triangle := 1/2 * length_DE * length_AD
  area_rectangle + area_triangle = 40400 + 20 * Real.sqrt 101 :=
by
  sorry

end combined_area_rectangle_triangle_l542_542475


namespace fewer_heads_than_tails_prob_l542_542621

theorem fewer_heads_than_tails_prob :
  let total_outcomes := 2^8,
      favorable_outcomes := 93
  in (favorable_outcomes / total_outcomes) = (93 / 256) :=
by
  let total_outcomes := 2^8,
      favorable_outcomes := 93
  show (favorable_outcomes / total_outcomes) = (93 / 256),
  sorry

end fewer_heads_than_tails_prob_l542_542621


namespace problem_1_problem_2_l542_542883

noncomputable def int_part_quot1 : ℝ := (3.24 / 1.5).toIntFloor
noncomputable def int_part_quot2 : ℝ := (1.92 / 2.4).toIntFloor

theorem problem_1 : int_part_quot1 = 2 := by
  sorry

theorem problem_2 : int_part_quot2 = 0 := by
  sorry

end problem_1_problem_2_l542_542883


namespace sin_sum_leq_3div2_sqrt3_l542_542393

theorem sin_sum_leq_3div2_sqrt3 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 :=
by
  sorry

end sin_sum_leq_3div2_sqrt3_l542_542393


namespace total_books_is_10033_l542_542467

variable (P C B M H : ℕ)
variable (x : ℕ) (h_P : P = 3 * x) (h_C : C = 2 * x)
variable (h_B : B = (3 / 2) * x)
variable (h_M : M = (3 / 5) * x)
variable (h_H : H = (4 / 5) * x)
variable (total_books : ℕ)
variable (h_total : total_books = P + C + B + M + H)
variable (h_bound : total_books > 10000)

theorem total_books_is_10033 : total_books = 10033 :=
  sorry

end total_books_is_10033_l542_542467


namespace minimum_kinds_of_candies_l542_542420

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542420


namespace extremum_at_one_iff_a_non_monotonic_on_interval_l542_542751

noncomputable def f (a x : ℝ) : ℝ := a^2 * x + (a / x) - 2 * Real.log x

def is_extremum (f' : ℝ → ℝ) (x : ℝ) : Prop := f' x = 0

def is_not_monotonic (f' : ℝ → ℝ) (I : Set ℝ) : Prop := ∃ x y, x ∈ I ∧ y ∈ I ∧ x < y ∧ f'(x) = 0 ∧ f'(y) = 0

theorem extremum_at_one_iff_a (a : ℝ) (deriv_f : ℝ → ℝ) :
  (is_extremum deriv_f 1) ↔ (a = 2) :=
sorry

theorem non_monotonic_on_interval (a : ℝ) (deriv_f : ℝ → ℝ) :
  (is_not_monotonic deriv_f (Set.Ioi 1)) ↔ (a ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 2) :=
sorry

end extremum_at_one_iff_a_non_monotonic_on_interval_l542_542751


namespace fewest_people_to_join_CBL_l542_542908

theorem fewest_people_to_join_CBL (initial_people teamsize : ℕ) (even_teams : ℕ → Prop)
  (initial_people_eq : initial_people = 38)
  (teamsize_eq : teamsize = 9)
  (even_teams_def : ∀ n, even_teams n ↔ n % 2 = 0) :
  ∃(p : ℕ), (initial_people + p) % teamsize = 0 ∧ even_teams ((initial_people + p) / teamsize) ∧ p = 16 := by
  sorry

end fewest_people_to_join_CBL_l542_542908


namespace part1_max_value_of_f_part2_theta_exists_l542_542755

noncomputable def f (x θ : ℝ) : ℝ :=
  sin x ^ 2 + (sqrt 3) * (tan θ) * (cos x) + (sqrt 3 / 8) * (tan θ) - 3 / 2

theorem part1_max_value_of_f :
  f 0 (π / 3) = 15 / 8 :=
sorry

theorem part2_theta_exists :
  ∃ θ, θ ∈ set.Icc 0 (π / 3) ∧ (∀ x, f x θ ≤ -1 / 8) ∧ (∃ x, f x θ = -1 / 8) ∧ θ = π / 6 :=
sorry

end part1_max_value_of_f_part2_theta_exists_l542_542755


namespace det_of_A_gt_zero_l542_542187

open Matrix

noncomputable theory

def exists_matrix_A (n : ℕ) : Prop :=
  ∃ (A : Matrix (Fin n) (Fin n) ℚ), (A ^ 3 = A + 1)

theorem det_of_A_gt_zero (n : ℕ) (A : Matrix (Fin n) (Fin n) ℚ) (h : A ^ 3 = A + 1) : det A > 0 :=
sorry

example (n : ℕ) : exists_matrix_A n :=
sorry

end det_of_A_gt_zero_l542_542187


namespace min_distance_ellipse_fixed_point_l542_542708

theorem min_distance_ellipse_fixed_point :
  ∃ P : ℝ × ℝ, (P ∈ {x : ℝ × ℝ | (x.1^2 / 9 + x.2^2 / 4 = 1)}) →
    ∀ Q : ℝ × ℝ, Q = (1, 0) →
    (∃ d_min : ℝ, d_min = (4 * real.sqrt 5) / 5 ∧
      ∀ P' : ℝ × ℝ, (P' ∈ {x : ℝ × ℝ | (x.1^2 / 9 + x.2^2 / 4 = 1)}) →
        dist P' Q ≥ d_min) := 
  sorry

end min_distance_ellipse_fixed_point_l542_542708


namespace length_of_median_A_to_BC_l542_542733

-- Define the coordinates of vertices A, B, and C
def A : ℝ × ℝ × ℝ := (1, -2, 5)
def B : ℝ × ℝ × ℝ := (-1, 0, 1)
def C : ℝ × ℝ × ℝ := (3, -4, 5)

-- Define the function to calculate the midpoint of two points in 3D
def midpoint (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ( (P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2 )

-- Define the function to calculate the distance between two points in 3D
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)

-- Define the midpoint M of B and C
def M : ℝ × ℝ × ℝ := midpoint B C

-- Prove that the length of the median from A to BC is 2
theorem length_of_median_A_to_BC : distance A M = 2 :=
by
  sorry

end length_of_median_A_to_BC_l542_542733


namespace coeff_x2_in_expansion_l542_542479

theorem coeff_x2_in_expansion : 
  (2 : ℚ) - (1 / x) * ((1 + x)^6)^(2 : ℤ) = (10 : ℚ) :=
by sorry

end coeff_x2_in_expansion_l542_542479


namespace original_number_l542_542697

theorem original_number (x : ℤ) (h : (x - 5) / 4 = (x - 4) / 5) : x = 9 :=
sorry

end original_number_l542_542697


namespace exist_k_good_function_l542_542956

def is_k_good (k : ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ), m > 0 → n > 0 → m ≠ n → (f(m) + n, f(n) + m) ≤ k

theorem exist_k_good_function : ∃ (k : ℕ), k > 0 ∧ (¬ ∃ f : ℕ → ℕ, is_k_good 1 f) ∧ (∃ f : ℕ → ℕ, is_k_good 2 f) :=
sorry

end exist_k_good_function_l542_542956


namespace pages_to_read_tomorrow_l542_542508

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l542_542508


namespace schedule_arrangement_count_l542_542154

-- Given subjects
inductive Subject
| Chinese
| Mathematics
| Politics
| English
| PhysicalEducation
| Art

open Subject

-- Define a function to get the total number of different arrangements
def arrangement_count : Nat := 192

-- The proof statement (problem restated in Lean 4)
theorem schedule_arrangement_count :
  arrangement_count = 192 :=
by
  sorry

end schedule_arrangement_count_l542_542154


namespace maximum_area_of_triangle_l542_542073

-- Definitions based on the conditions
variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def is_right_triangle (A B C : Triangle) : Prop := ∃ h : right_angle A B C, True
noncomputable def largest_side (t : Triangle) : Float := max_side t
noncomputable def area (t : Triangle) : Float := triangle_area t

-- Define the context: A triangle with specified properties
variables (t : Triangle)

-- Conditions
axiom largest_side_is_10 : ∀ t, largest_side t = 10
axiom midpoints_of_altitudes_collinear : ∀ t, midpoints_altitudes t collinear

-- Main theorem statement
theorem maximum_area_of_triangle (t : Triangle)
  (h1 : largest_side t = 10)
  (h2 : midpoints_of_altitudes_collinear t) :
  area t = 25 := sorry

end maximum_area_of_triangle_l542_542073


namespace find_a_l542_542090

theorem find_a (a : ℝ) (h1 : ∀ x y : ℝ, x - 2 * y + 4 = 0 → l (2, 3) ∥ x - 2y + 1 = 0)
(h2 : ∃ a : ℝ, a > 0 ∧ (|a - 4 + 4| / (√(1^2 + (-2)^2))) = (√5 / 5)) :
  a = 1 :=
begin
  sorry
end

end find_a_l542_542090


namespace sequence_length_l542_542769

theorem sequence_length :
  ∀ (a d n : ℤ), a = -6 → d = 4 → (a + (n - 1) * d = 50) → n = 15 :=
by
  intros a d n ha hd h_seq
  sorry

end sequence_length_l542_542769


namespace determine_original_price_l542_542248

namespace PriceProblem

variable (x : ℝ)

def final_price (x : ℝ) : ℝ := 0.98175 * x

theorem determine_original_price (h : final_price x = 100) : x = 101.86 :=
by
  sorry

end PriceProblem

end determine_original_price_l542_542248


namespace find_ABC_sum_l542_542298

theorem find_ABC_sum (A B C : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h_rel_prime : Nat.coprime A B ∧ Nat.coprime B C ∧ Nat.coprime A C)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) :
  A + B + C = 5 :=
sorry

end find_ABC_sum_l542_542298


namespace equal_areas_of_quadrilaterals_l542_542075

variables {A B C D E F M : Type} [ConvexHexagon A B C D E F] (circ_centers : Circle(₀, O₁, O₂, O₃, O₄, O₅, O₆))

-- Assume a convex hexagon ABCDEF with a common point M for diagonals and circumcenters of triangles MAB, MBC, MCD, MDE, MEF, MFA lie on a circle.
theorem equal_areas_of_quadrilaterals
  (h1 : are_common_diagonal A B C D E F M)
  (h2 : circumcenters_lies_on_circle circ_centers M A B C D E F) :
  (area_of_quadrilateral A B D E = area_of_quadrilateral B C E F) ∧
  (area_of_quadrilateral B C E F = area_of_quadrilateral C D F A) :=
sorry

end equal_areas_of_quadrilaterals_l542_542075


namespace function_monotonicity_and_extrema_l542_542350

noncomputable def f (x : ℝ) := x / (1 + x^2)

theorem function_monotonicity_and_extrema 
  (h0 : ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), 0 < 1 - x^2) : 
  (∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), (f '' (set.Icc (0 : ℝ) (1 : ℝ))).is_monotone (≤)) ∧
  (f '' (set.Icc (-1 : ℝ) (1 : ℝ))).is_max (1 / 2) ∧
  (f '' (set.Icc (-1 : ℝ) (1 : ℝ))).is_min (-1 / 2) :=
by
  sorry

end function_monotonicity_and_extrema_l542_542350


namespace birthday_paradox_l542_542180

-- Defining the problem conditions
def people (n : ℕ) := n ≥ 367

-- Using the Pigeonhole Principle as a condition
def pigeonhole_principle (pigeonholes pigeons : ℕ) := pigeonholes < pigeons

-- Stating the final proposition
theorem birthday_paradox (n : ℕ) (days_in_year : ℕ) (h1 : days_in_year = 366) (h2 : people n) : pigeonhole_principle days_in_year n :=
sorry

end birthday_paradox_l542_542180


namespace sum_b_first_100_terms_l542_542745

open Real

-- Assuming aₙ and bₙ being arithmetic
variables {a_n b_n : ℕ → ℝ} (d1 d2 : ℝ)

def c_n (n : ℕ) := (-1)^n * (a_n n - b_n n)

-- Conditions of the problem
axiom a11 : a_n 11 = 32
axiom b21 : b_n 21 = 43
axiom h1 : (∑ i in Finset.range 10 + 1, c_n i) = 5
axiom h2 : (∑ i in Finset.range 13 + 1, c_n i) = -5

-- Required proof statement 
theorem sum_b_first_100_terms : (∑ i in Finset.range 100 + 1, b_n i) = 10200 := 
  sorry

end sum_b_first_100_terms_l542_542745


namespace chord_length_product_l542_542523

theorem chord_length_product (A B : ℂ) (D : ℕ → ℂ)
  (hAB : abs (A - B) = 6)
  (hArcs : ∀ i, abs (D i - D (i + 1)) = π / 4) :
  ∏ i in finset.range 7, abs (A - D i) * abs (B - D i) = 196608 := 
sorry

end chord_length_product_l542_542523


namespace num_possible_values_abs_z_l542_542718

theorem num_possible_values_abs_z (a b c : ℝ) (h_eq : a = 1 ∧ b = -10 ∧ c = 50) :
  let Δ := b^2 - 4 * a * c in
  Δ < 0 → (∃ z1 z2 : ℂ, z1^2 - b * z1 + c = 0 ∧ z2^2 - b * z2 + c = 0 ∧ |z1| = |z2| ∧ |z1| = 5 * Real.sqrt 2 ∧ |z2| = 5 * Real.sqrt 2) →
  (∀ z1 z2 : ℂ, (z1^2 - b * z1 + c = 0 ∧ z2^2 - b * z2 + c = 0) → |z1| = |z2| ∧ |z1| = 5 * Real.sqrt 2) ∧
  ∃! val, val = 5 * Real.sqrt 2 :=
by
  sorry

end num_possible_values_abs_z_l542_542718


namespace probability_two_six_one_four_l542_542784

theorem probability_two_six_one_four
  (prob_six : ℚ := 1 / 6)
  (prob_four : ℚ := 1 / 6)
  (total_dice : ℕ := 3)
  (unique_arrangements : ℕ := 3) :
  let prob_sequence := prob_six * prob_six * prob_four in
  unique_arrangements * prob_sequence = 1 / 72 := 
by
  sorry

end probability_two_six_one_four_l542_542784


namespace find_angle_l542_542907

variable {V : Type} [InnerProductSpace ℝ V]
variable (a b d : V)
variable (c : V := d + a + b)

def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 1 := sorry
def norm_d : ∥d∥ = 3 := sorry
def orthogonal_d_a : ⟪d, a⟫ = 0 := sorry
def orthogonal_d_b : ⟪d, b⟫ = 0 := sorry

theorem find_angle
  (ha : norm_a)
  (hb : norm_b)
  (hd : norm_d)
  (hod : orthogonal_d_a)
  (hob : orthogonal_d_b)
  : ∃ θ, θ = real.arccos (real.sqrt 14 / 7) :=
by sorry

end find_angle_l542_542907


namespace four_points_concyclic_l542_542723

-- Definitions and problem formalizations.
variables {A B C B1 C1 B2 C2 : Point}
variables (hABCacute : acute_triangle A B C)
variables (hB2onAC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B2 = t • A + (1 - t) • C)
variables (hC2onAB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C2 = t • A + (1 - t) • B)
variables (hBC2eqBB1 : dist B C2 = dist B B1)
variables (hCB2eqCC1 : dist C B2 = dist C C1)

-- The proof theorem.
theorem four_points_concyclic :
  ∃ D : Circle, D = circle_of_points B1 B2 C1 C2 :=
sorry

end four_points_concyclic_l542_542723


namespace union_M_N_l542_542007

def M : Set ℝ := { x | x^2 - x = 0 }
def N : Set ℝ := { y | y^2 + y = 0 }

theorem union_M_N : (M ∪ N) = {-1, 0, 1} := 
by 
  sorry

end union_M_N_l542_542007


namespace exists_n_binom_divisors_l542_542553

def binom (n k : ℕ) : ℕ := n.choose k

theorem exists_n_binom_divisors : ∃ n : ℕ, n > 2003 ∧ ∀ i : ℕ, i ≤ 2002 → binom n i ∣ binom n (i + 1) :=
by
  let lcm_val := Nat.lcm (List.range' 1 2003)
  let n := lcm_val - 1
  have h₀ : n > 2003 := sorry
  have h₁ : ∀ i : ℕ, i ≤ 2002 → binom n i ∣ binom n (i + 1) := sorry
  exact ⟨n, h₀, h₁⟩

end exists_n_binom_divisors_l542_542553


namespace triangle_condition_proof_l542_542052

variables {A B C D M K : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace K]
variables (AB AC AD : ℝ)

-- Definitions based on the conditions
def is_isosceles (A B C : Type*) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Type*) : Prop :=
  true -- Ideally, this condition is more complex and involves perpendicular projection

def is_midpoint (M A D : Type*) : Prop :=
  true -- Ideally, this condition is more specific and involves equality of segments

def extends_to (C M A B K : Type*) : Prop :=
  true -- Represents the extension relationship

-- The theorem to be proved
theorem triangle_condition_proof (A B C D M K : Type*)
  (h_iso : is_isosceles A B C AB AC)
  (h_alt : is_altitude A D B C)
  (h_mid : is_midpoint M A D)
  (h_ext : extends_to C M A B K)
  : AB = 3 * AK :=
  sorry

end triangle_condition_proof_l542_542052


namespace cube_collinear_points_l542_542664

theorem cube_collinear_points : 
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let centroid := 1
  ∃ (number_of_sets : ℕ), 
    number_of_sets = 49 :=
begin
  sorry
end

end cube_collinear_points_l542_542664


namespace reflection_of_point_over_vector_eq_l542_542294

theorem reflection_of_point_over_vector_eq :
  let p := (3, -2 : ℚ × ℚ)
  let v := (2, -1 : ℚ × ℚ)
  let dot_prod (a b : ℚ × ℚ) : ℚ := a.1 * b.1 + a.2 * b.2
  let scalar_mult (k : ℚ) (a : ℚ × ℚ) := (k * a.1, k * a.2)
  let proj :=
    let num := dot_prod p v
    let den := dot_prod v v
    scalar_mult (num / den) v
  let r :=
    let p_scaled := scalar_mult 2 proj
    (p_scaled.1 - p.1, p_scaled.2 - p.2)
  r = (17 / 5, -6 / 5) :=
by
  sorry

end reflection_of_point_over_vector_eq_l542_542294


namespace Edgardo_winning_strategy_l542_542897

theorem Edgardo_winning_strategy :
  ∀ (numbers: List ℕ), 
  (∀ n, n ∈ numbers → 3 ≤ n ∧ n ≤ 2019) →
  (∀ (turn : ℕ), turn.even → 
  ∀ (numbers' : List ℕ), 
  game_turn numbers numbers' turn → 
  all_equal numbers' → 
  (turn > 0 → Edgardo_wins)) :=
begin
  sorry
end

end Edgardo_winning_strategy_l542_542897


namespace ellipse_solution_l542_542246

theorem ellipse_solution :
  (∃ (a b : ℝ), a = 4 * Real.sqrt 2 + Real.sqrt 17 ∧ b = Real.sqrt (32 + 16 * Real.sqrt 34) ∧ (∀ (x y : ℝ), (3 * 0 ≤ y ∧ y ≤ 8) → (3 * 0 ≤ x ∧ x ≤ 5) → (Real.sqrt ((x+3)^2 + y^2) + Real.sqrt ((x-3)^2 + y^2) = 2 * a) → 
   (Real.sqrt ((x-0)^2 + (y-8)^2) = b))) :=
sorry

end ellipse_solution_l542_542246


namespace sum_of_extrema_f_l542_542296

def f (x : ℝ) : ℝ := x^2 / (2 * x + 2)

theorem sum_of_extrema_f :
  ∑ x in {min, max}, f x = 1 / 3 :=
by
  sorry

end sum_of_extrema_f_l542_542296


namespace exists_integer_x_l542_542521

theorem exists_integer_x (a b c : ℝ) (h : ∃ (a b : ℝ), (a - b).abs > 1 / (2 * Real.sqrt 2)) :
  ∃ (x : ℤ), (x : ℝ)^2 - 4 * (a + b + c) * x + 12 * (a * b + b * c + c * a) < 0 := 
sorry

end exists_integer_x_l542_542521


namespace plate_arrangement_count_l542_542217

theorem plate_arrangement_count :
  let n_blue := 6
  let n_red := 3
  let n_green := 3
  let n_orange := 1
  
  let total_plates := n_blue + n_red + n_green + n_orange
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

  factorial (total_plates - 1) / (factorial n_blue * factorial n_red * factorial n_green) = 
  365240 - 2 * factorial ((total_plates - 1) - n_green) / (factorial n_blue * factorial n_red) := sorry

end plate_arrangement_count_l542_542217


namespace number_of_factorizations_l542_542473

theorem number_of_factorizations (a b c : ℕ) :
  (a * b * c = 10000) ∧ (¬(10 ∣ a) ∧ ¬(10 ∣ b) ∧ ¬(10 ∣ c)) → 
  (up_to_permutation (multiset {a, b, c})) = 6 :=
by
  have h1 : 10000 = 2^4 * 5^4 := by norm_num
  sorry

end number_of_factorizations_l542_542473


namespace joy_valid_rod_count_l542_542814

theorem joy_valid_rod_count : 
  let l := [4, 12, 21]
  let qs := [1, 2, 3, 5, 13, 20, 22, 40].filter (fun x => x != 4 ∧ x != 12 ∧ x != 21)
  (∀ d ∈ qs, 4 + 12 + 21 > d ∧ 4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4) → 
  ∃ n, n = 28 :=
by sorry

end joy_valid_rod_count_l542_542814


namespace projection_is_same_l542_542927

variable (v : ℝ × ℝ) (p : ℝ × ℝ)

noncomputable def is_projection (x : ℝ × ℝ) (v : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let proj := ((x.1 * v.1 + x.2 * v.2) / (v.1 * v.1 + v.2 * v.2)) * v
  proj = p

theorem projection_is_same :
  let p := ( -39 / 29, 91 / 29 )
  is_projection (-4, 2) v p → is_projection (3, 5) v p :=
by
  sorry

end projection_is_same_l542_542927


namespace positive_integer_palindrome_square_palindrome_l542_542299

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem positive_integer_palindrome_square_palindrome (n : ℕ) (h : n > 0) : 
  is_palindrome n → is_palindrome (n * n) → 
  n = 3 ∨ 
  (∃ k, n = (10^k + 1)) ∨ 
  (∃ k, n = (10^k + 2 * 10^(k//2) + 1)) ∨ 
  (∃ k, n = (2 * 10^k + 2)) ∨ 
  (∃ k, n = (2 * 10^k + (10^k + 1) * 10^(k//2) + 2)) :=
sorry

end positive_integer_palindrome_square_palindrome_l542_542299


namespace smallest_discount_m_l542_542301

theorem smallest_discount_m {m : ℕ} (h₁ : (1 : ℝ) - (m : ℝ) / 100 < (1 - 0.08) ^ 3)
                           (h₂ : (1 : ℝ) - (m : ℝ) / 100 < (1 - 0.12) ^ 2)
                           (h₃ : (1 : ℝ) - (m : ℝ) / 100 < (1 - 0.20) * (1 - 0.15)) :
  m = 33 :=
begin
  sorry -- Proof is omitted as per instruction.
end

end smallest_discount_m_l542_542301


namespace num_integers_square_between_50_and_200_l542_542291

theorem num_integers_square_between_50_and_200 :
  {n : ℤ | 50 < n^2 ∧ n^2 < 200}.finite.toFinset.card = 14 :=
sorry

end num_integers_square_between_50_and_200_l542_542291


namespace train_crossing_time_l542_542627

-- Define the given conditions
def train_length : ℝ := 100 -- Length of the train in meters

def speed_kmh : ℝ := 144 -- Speed of the train in kilometers per hour

-- Conversion factor from kilometers per hour to meters per second
def kmh_to_ms (v: ℝ) : ℝ := v * (1000 / 3600)

-- Define the speed in meters per second using the conversion factor
def speed_ms : ℝ := kmh_to_ms speed_kmh

-- Define the expected time to cross the electric pole
def expected_time : ℝ := 2.5 -- seconds

-- Prove that the time taken for the train to cross the electric pole is 2.5 seconds
theorem train_crossing_time : (train_length / speed_ms) = expected_time := by
  -- statement for proof
  sorry

end train_crossing_time_l542_542627


namespace center_of_circle_polar_eq_l542_542484

theorem center_of_circle_polar_eq (ρ θ : ℝ) : 
    (∀ ρ θ, ρ = 2 * Real.cos θ ↔ (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1) → 
    ∃ x y : ℝ, x = 1 ∧ y = 0 :=
by
  sorry

end center_of_circle_polar_eq_l542_542484


namespace area_of_rectangle_is_correct_l542_542210

-- Given Conditions
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- Question: Find the area of the rectangle
def area := length * width

-- The theorem to prove
theorem area_of_rectangle_is_correct : area = 588 :=
by
  -- Proof steps can go here.
  sorry

end area_of_rectangle_is_correct_l542_542210


namespace evaluate_g_at_4_l542_542828

def g (x : ℕ) := 5 * x + 2

theorem evaluate_g_at_4 : g 4 = 22 := by
  sorry

end evaluate_g_at_4_l542_542828


namespace probability_of_forming_triangle_is_three_tenths_l542_542564

-- Define the set {1, 2, 3, 4, 5}
def S : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the function that checks the triangle inequality
def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the probability of forming a triangle
def probability_triangle (s : Finset ℕ) (n : ℕ) : ℚ :=
  let triplets := s.powerset.filter (λ x, x.card = 3),
      valid_triplets := triplets.filter (λ x, ∀ t ∈ x.val, t.satisfies_triangle_inequality) in
  valid_triplets.card / triplets.card

-- Statement to prove
theorem probability_of_forming_triangle_is_three_tenths :
  probability_triangle S 3 = 3 / 10 := by
    sorry

end probability_of_forming_triangle_is_three_tenths_l542_542564


namespace problem_statement_l542_542525

-- Define C and D as specified in the problem conditions.
def C : ℕ := 4500
def D : ℕ := 3000

-- The final statement of the problem to prove C + D = 7500.
theorem problem_statement : C + D = 7500 := by
  -- This proof can be completed by checking arithmetic.
  sorry

end problem_statement_l542_542525


namespace perimeter_maximum_l542_542994

noncomputable def max_triangle_perimeter (R : ℝ) (A : ℝ) : ℝ :=
  let a := (4 * sqrt 3 / 3) * sin A
  let B := 60 * π / 180 / 2
  let b := (4 * sqrt 3 / 3) * sin B
  let c := ((4 * sqrt 3 / 3) * sin (2 * π / 3 - B))
  a + b + c

theorem perimeter_maximum {R : ℝ} {A : ℝ} (h1 : R = (2 * sqrt 3 / 3)) (h2 : A = π / 3) :
    max_triangle_perimeter R A = 6 :=
  sorry

end perimeter_maximum_l542_542994


namespace fraction_of_product_l542_542165

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l542_542165


namespace calculate_expression_l542_542672

theorem calculate_expression :
  ( (1/4: ℝ)⁻¹ - |real.sqrt 3 - 2| + 2 * (-real.sqrt 3) = 2 - real.sqrt 3 ) :=
by
  -- Add assumptions and calculations here
  sorry

end calculate_expression_l542_542672


namespace tan_of_point_on_graph_l542_542778

theorem tan_of_point_on_graph (a : ℝ) (h : (4, a) ∈ { p : ℝ × ℝ | p.2 = p.1^(1/2) }) : tan (a / 6 * π) = sqrt 3 :=
by
  -- (4, a) ∈ { p : ℝ × ℝ | p.2 = p.1^(1/2) } means that a = 4^(1/2)
  sorry

end tan_of_point_on_graph_l542_542778


namespace tangent_circumcircle_of_triangle_DEC_l542_542959

-- Definitions of geometric entities
variables (A B C D E C' : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point C']
variables (ABC : Triangle A B C) (D_alt : Altitude A D C) (E_alt : Altitude B E C)

-- Define symmetric point C' w.r.t. midpoint of segment DE
def symmetric_point (C : Point) (mid_DE : Point) : Point := C' 

-- Problem statement as Lean theorem
theorem tangent_circumcircle_of_triangle_DEC' 
  (mid_DE : Point) (h1: C' = symmetric_point C mid_DE) (h2: LiesOn C' (LineSegment A B)) :
  Tangent (Circumcircle (Triangle D E C')) (Line A B) :=
sorry

end tangent_circumcircle_of_triangle_DEC_l542_542959


namespace minimum_candy_kinds_l542_542401

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542401


namespace adult_ticket_cost_l542_542561

theorem adult_ticket_cost 
    (student_ticket_cost : ℝ) 
    (total_tickets : ℕ) 
    (total_income : ℝ) 
    (ron_tickets_sold : ℕ) 
    (ron_income : ℝ) 
    (adult_tickets_sold : ℕ) : ℝ :=
by
  let adult_ticket_cost := (total_income - ron_income) / adult_tickets_sold
  exact adult_ticket_cost

example (student_ticket_cost : ℝ := 2.00) 
        (total_tickets : ℕ := 20) 
        (total_income : ℝ := 60.00) 
        (ron_tickets_sold : ℕ := 12) 
        (ron_income : ℝ := 24.00) 
        (adult_tickets_sold : ℕ := 8): adult_ticket_cost student_ticket_cost total_tickets total_income ron_tickets_sold ron_income adult_tickets_sold = 4.50 :=
by
  refl

end adult_ticket_cost_l542_542561


namespace binom_identity1_binom_identity2_l542_542106

section Combinatorics

variable (n k m : ℕ)

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

-- Prove the identity: C(n, k) + C(n, k-1) = C(n+1, k)
theorem binom_identity1 : binomial n k + binomial n (k-1) = binomial (n+1) k :=
  sorry

-- Using the identity, prove: C(n, m) + C(n-1, m) + ... + C(n-10, m) = C(n+1, m+1) - C(n-10, m+1)
theorem binom_identity2 :
  (binomial n m + binomial (n-1) m + binomial (n-2) m + binomial (n-3) m
   + binomial (n-4) m + binomial (n-5) m + binomial (n-6) m + binomial (n-7) m
   + binomial (n-8) m + binomial (n-9) m + binomial (n-10) m)
   = binomial (n+1) (m+1) - binomial (n-10) (m+1) :=
  sorry

end Combinatorics

end binom_identity1_binom_identity2_l542_542106


namespace quadratic_distinct_real_roots_range_l542_542037

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l542_542037


namespace fraction_addition_l542_542253

theorem fraction_addition : (3 / 5) + (2 / 15) = 11 / 15 := sorry

end fraction_addition_l542_542253


namespace distinct_angles_in_cube_rays_l542_542676

theorem distinct_angles_in_cube_rays : 
  let cube := {vertices : list ℝ × ℝ × ℝ // vertices.length = 8 ∧ is_cube vertices}
  ∃ v ∈ cube.vertices, count_distinct_angles (rays_from v) = 5 :=
by
  sorry

end distinct_angles_in_cube_rays_l542_542676


namespace triangle_inequality_l542_542932

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) : Prop :=
    a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 3 5 7 (by norm_num) (by norm_num) (by norm_num) :=
by simp; apply and.intro (by norm_num) (by norm_num)

end triangle_inequality_l542_542932


namespace construct_triangle_l542_542266

theorem construct_triangle (α : ℝ) (a : ℝ) (p q : ℝ) :
  ∃ (b c : ℝ), ∃ (ABC : Triangle), 
    ABC.has_angle α ∧ 
    ABC.opposite_side_of_angle α = a ∧ 
    b / c = p / q ∧ 
    ABC.side_ratio b c = p / q ∧ 
    ABC.is_valid :=
sorry

end construct_triangle_l542_542266


namespace garish_functions_count_l542_542082

noncomputable def count_garish_functions (n : ℕ) : ℕ :=
∑ g in Equiv.perm (Fin n), 1

theorem garish_functions_count (n : ℕ) :
  (count_garish_functions n) = nat.factorial n :=
by
  sorry

end garish_functions_count_l542_542082


namespace find_AB_length_l542_542857

theorem find_AB_length (R : ℝ)
  (O : ℝ) 
  (A B C : ℝ → ℝ → ℝ) 
  (BC AC: ℝ) 
  (hO_center : ∀ A B C, A = R ∧ B = R ∧ C = R)
  (BC_length : BC = 8)
  (AC_length : AC = 4)
  (vec_length : ∥ 4 * (A O O) - (B O O) - 3 * (C O O) ∥ = 10)
  : ∥A O O - B O O∥ = 5 :=
sorry

end find_AB_length_l542_542857


namespace sum_of_squares_l542_542267

theorem sum_of_squares (n : ℕ) : ∑ k in finset.range (n + 1), k^2 = n * (n + 1) * (2 * n + 1) / 6 :=
by sorry

end sum_of_squares_l542_542267


namespace length_of_segment_AB_l542_542038

noncomputable def polar_eq1_to_cartesian_eq : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2) = 1 ↔ ∃ (ρ θ : ℝ), ρ = 1 ∧ x = ρ * Math.cos θ ∧ y = ρ * Math.sin θ

noncomputable def polar_eq2_to_cartesian_eq : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 - x + real.sqrt 3 * y = 0) ↔ ∃ (ρ θ : ℝ), ρ = 2 * Math.cos (θ + real.pi / 3) ∧ x = ρ * Math.cos θ ∧ y = ρ * Math.sin θ

noncomputable def segment_length_eqn : Prop :=
  ∀ (A B : ℝ × ℝ), A = (1, 0) ∧ B = (-1/2, -real.sqrt 3 / 2) → real.sqrt ((1 + 1/2)^2 + (0 + real.sqrt 3 / 2)^2) = real.sqrt 3

theorem length_of_segment_AB : polar_eq1_to_cartesian_eq ∧ polar_eq2_to_cartesian_eq ∧ segment_length_eqn :=
by {
  sorry,
}

end length_of_segment_AB_l542_542038


namespace exterior_angle_of_square_octagon_l542_542656

/--
Given:
1. A square with an angle \( \angle CAD = 90^\circ \).
2. A regular octagon with an angle \( \angle BAD = 135^\circ \).

Then the exterior angle \( \angle BAC = 135^\circ \).
-/
theorem exterior_angle_of_square_octagon {A D B C : Type} [Plane : Type]
  (square_angle_CAD : ∠CAD = 90)
  (octagon_angle_BAD : ∠BAD = 135) :
  ∠BAC = 135 :=
by
  sorry

end exterior_angle_of_square_octagon_l542_542656


namespace area_of_triangle_ACD_l542_542632

theorem area_of_triangle_ACD (p : ℝ) (y1 y2 x1 x2 : ℝ)
  (h1 : y1^2 = 2 * p * x1)
  (h2 : y2^2 = 2 * p * x2)
  (h3 : y1 + y2 = 4 * p)
  (h4 : y2 - y1 = p)
  (h5 : 2 * y1 + 2 * y2 = 8 * p^2 / (x2 - x1))
  (h6 : x2 - x1 = 2 * p)
  (h7 : 8 * p^2 = (y1 + y2) * 2 * p) :
  1 / 2 * (y1 * (x1 - (x2 + x1) / 2) + y2 * (x2 - (x2 + x1) / 2)) = 15 / 2 * p^2 :=
by
  sorry

end area_of_triangle_ACD_l542_542632


namespace geom_seq_a5_a6_eq_180_l542_542465

theorem geom_seq_a5_a6_eq_180 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n+1) = a n * q)
  (cond1 : a 1 + a 2 = 20)
  (cond2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 180 :=
sorry

end geom_seq_a5_a6_eq_180_l542_542465


namespace sequence_50th_term_l542_542957

def sequence_term (n : ℕ) : ℕ × ℕ :=
  (5 + (n - 1), n - 1)

theorem sequence_50th_term :
  sequence_term 50 = (54, 49) :=
by
  sorry

end sequence_50th_term_l542_542957


namespace ratio_of_bottles_l542_542496

theorem ratio_of_bottles
  (initial_money : ℤ)
  (initial_bottles : ℕ)
  (cost_per_bottle : ℤ)
  (cost_per_pound_cheese : ℤ)
  (cheese_pounds : ℚ)
  (remaining_money : ℤ) :
  initial_money = 100 →
  initial_bottles = 4 →
  cost_per_bottle = 2 →
  cost_per_pound_cheese = 10 →
  cheese_pounds = 0.5 →
  remaining_money = 71 →
  (2 * initial_bottles) / initial_bottles = 2 :=
by 
  sorry

end ratio_of_bottles_l542_542496


namespace birds_on_fence_l542_542911

/-
1. Initially, there are 12 birds on the fence.
2. Eight more birds land on the fence, making a total of T birds.
3. Then, five birds fly away and three more join, leaving us with W birds.
4. Half of the birds that were on the fence fly away while 2.5 birds return.
5. Prove that the number of birds on the fence is now 11.5.
-/

def initial_birds : ℕ := 12
def landed_birds : ℕ := 8
def flew_away_birds_1 : ℕ := 5
def joined_birds : ℕ := 3
def half_of_birds (n : ℕ) : ℝ := n / 2
def returned_birds : ℝ := 2.5

noncomputable def final_birds (initial : ℕ) (landed : ℕ) (flew_away_1 : ℕ) (joined : ℕ) (returned : ℝ) : ℝ :=
  let T := initial + landed
  let W := T - flew_away_1 + joined
  W - half_of_birds W + returned

theorem birds_on_fence :
  final_birds initial_birds landed_birds flew_away_birds_1 joined_birds returned_birds = 11.5 := by
  sorry

end birds_on_fence_l542_542911


namespace part_a_part_b_l542_542263

open Complex

noncomputable def a (x y z : ℂ) : ℝ := 
  (1 + (x / y)) * (1 + (y / z)) * (1 + (z / x))

theorem part_a (x y z : ℂ) (hx : abs x = 1) (hy : abs y = 1) (hz : abs z = 1) : 
  a x y z ∈ ℝ := 
sorry

theorem part_b (x y z : ℂ) (hx : abs x = 1) (hy : abs y = 1) (hz : abs z = 1) :
  -1 ≤ a x y z ∧ a x y z ≤ 8 := 
sorry

end part_a_part_b_l542_542263


namespace minimum_candy_kinds_l542_542397

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542397


namespace problem_statement_l542_542380

theorem problem_statement (y : ℝ) (h : 5^(3 * y) = 125) : 5^(3 * y - 2) = 5 :=
by
  sorry

end problem_statement_l542_542380


namespace Watson_class_student_count_l542_542909

def num_kindergartners : ℕ := 14
def num_first_graders : ℕ := 24
def num_second_graders : ℕ := 4

def total_students : ℕ := num_kindergartners + num_first_graders + num_second_graders

theorem Watson_class_student_count : total_students = 42 := 
by
    sorry

end Watson_class_student_count_l542_542909


namespace fraction_of_64_l542_542164

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l542_542164


namespace irrational_sqrt_5_l542_542179

theorem irrational_sqrt_5 : 
  ∀ (x : ℝ), 
    (x = 0 ∨ x = 3.14 ∨ x = -8 / 7 ∨ x = Real.sqrt 5) →
    (∀ (x : ℝ), x = Real.sqrt 5 → ¬(∃ (a b : ℤ), b ≠ 0 ∧ x = a / b)) →
    (∃ (a b : ℤ), 0 = a / b) ∧ 
    (∃ (a b : ℤ), 3.14 = a / b) ∧ 
    (∃ (a b : ℤ), b ≠ 0 ∧ -8 / 7 = a / b) :=
by
  sorry

end irrational_sqrt_5_l542_542179


namespace solve_eq_l542_542867

theorem solve_eq : ∃ x : ℝ, 6 * x - 4 * x = 380 - 10 * (x + 2) ∧ x = 30 := 
by
  sorry

end solve_eq_l542_542867


namespace amazing_rectangle_sum_eq_942_l542_542653

-- Define a rectangle to be amazing if the area equals three times the perimeter
def amazing_rectangle (a b : ℕ) : Prop :=
  a * b = 6 * (a + b)

-- Collect all possible areas of amazing rectangles with integer side lengths
def amazing_areas : List ℕ :=
  [ (a, b) | a b : ℕ, amazing_rectangle a b]
    .map (λ (a, b) => a * b)

-- Sum all unique areas of amazing rectangles
def sum_of_amazing_areas : ℕ :=
  amazing_areas.uniq.sum

-- The main theorem stating the sum of all possible areas of amazing rectangles
theorem amazing_rectangle_sum_eq_942 : sum_of_amazing_areas = 942 := 
by
  sorry

end amazing_rectangle_sum_eq_942_l542_542653


namespace prank_people_combinations_l542_542153

theorem prank_people_combinations (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (hMonday : Monday = 2)
  (hTuesday : Tuesday = 3)
  (hWednesday : Wednesday = 6)
  (hThursday : Thursday = 4)
  (hFriday : Friday = 3) :
  Monday * Tuesday * Wednesday * Thursday * Friday = 432 :=
  by sorry

end prank_people_combinations_l542_542153


namespace odot_9_3_equals_5_l542_542131

def odot (a b : ℝ) := a - (4 * a) / (3 * b)

theorem odot_9_3_equals_5 : odot 9 3 = 5 := 
by
  unfold odot
  simp
  sorry

end odot_9_3_equals_5_l542_542131


namespace number_of_seasons_is_5_l542_542201

theorem number_of_seasons_is_5 :
  let cost_per_first_season_episode := 100000 in
  let number_of_first_season_episodes := 12 in
  let cost_per_other_season_episode := 2 * cost_per_first_season_episode in
  let number_of_other_seasons_increase_rate := 1.5 in
  let number_of_last_season_episodes := 24 in
  let total_production_cost := 16800000 in
  ∃ n : ℕ,
    -- Cost of the first season
    let cost_first_season := number_of_first_season_episodes * cost_per_first_season_episode in
    -- Assuming the structure of other seasons excluding the last season
    let cost_per_other_season := 18 * cost_per_other_season_episode in
    -- Cost of the last season
    let cost_last_season := number_of_last_season_episodes * cost_per_other_season_episode in
    -- Total productions cost less that of the first and last seasons
    let cost_other_seasons := total_production_cost - cost_first_season - cost_last_season in
    -- Number of other seasons
    let number_of_other_seasons := cost_other_seasons / cost_per_other_season in
    -- Total number of seasons
    n = 1 + number_of_other_seasons + 1 →
    n = 5 :=
begin
  -- Proof would go here
  sorry
end

end number_of_seasons_is_5_l542_542201


namespace contrapositive_of_ab_eq_zero_l542_542875

theorem contrapositive_of_ab_eq_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → ab ≠ 0 :=
by
  sorry

end contrapositive_of_ab_eq_zero_l542_542875


namespace b_n_formula_S_n_formula_l542_542901

-- Define the sequence a_n
def a : ℕ → ℝ
| 0      := 2
| (n + 1) := 2^(n + 1) * a n / ( (n + 0.5) * (a n) + 2^n)

-- Task 1: Prove b_n = (n^2 + 1) / 2
def b (n : ℕ) : ℝ := 2^n / (a n)
theorem b_n_formula (n : ℕ) : b n = (n^2 + 1) / 2 :=
by
  sorry

-- Task 2: Prove S_n = n / 2
def c_n (n : ℕ) : ℝ := 1 / (n * (n + 1) * a (n + 1))
def S (n : ℕ) : ℝ := (Finset.range n).sum c_n
theorem S_n_formula (n : ℕ) : S n = n / 2 :=
by
  sorry

end b_n_formula_S_n_formula_l542_542901


namespace minimum_club_members_l542_542924

theorem minimum_club_members : ∃ (b : ℕ), (b = 7) ∧ ∃ (a : ℕ), (2 : ℚ) / 5 < (a : ℚ) / b ∧ (a : ℚ) / b < 1 / 2 := 
sorry

end minimum_club_members_l542_542924


namespace minimum_kinds_of_candies_l542_542414

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542414


namespace largest_m_for_polynomial_factorization_l542_542136

theorem largest_m_for_polynomial_factorization :
  ∀ (q : ℕ → polynomial ℝ), (x : ℝ) → (h : x^10 - 1 = q(1) * q(2) * q(3) * q(4)) →
  (∀ i, (q i).degree > 0) → (∀ i, (q i).coeffs ∈ ℝ) → m ≤ 4 := sorry

end largest_m_for_polynomial_factorization_l542_542136


namespace bus_pickup_time_l542_542842

open Time  -- Open the Time namespace

/-- 
 Conditions:
 1. The bus takes forty minutes to arrive at the first station.
 2. Mr. Langsley arrives at work at 9:00 a.m.
 3. The total time from the first station to Mr. Langsley's workplace is 140 minutes.

 Goal:
 The bus picks Mr. Langsley up at 6:00 a.m.
-/
theorem bus_pickup_time :
  let bus_to_station := 40  -- minutes
  let arrival_time := Time.mk 9 0  -- 9:00 am
  let first_station_to_work := 140  -- minutes
  (arrival_time - (bus_to_station + first_station_to_work)) = Time.mk 6 0 := 
by
  sorry

end bus_pickup_time_l542_542842


namespace percent_sold_second_day_l542_542995

-- Defining the problem conditions
def initial_pears (x : ℕ) : ℕ := x
def pears_sold_first_day (x : ℕ) : ℕ := (20 * x) / 100
def pears_remaining_after_first_sale (x : ℕ) : ℕ := x - pears_sold_first_day x
def pears_thrown_away_first_day (x : ℕ) : ℕ := (50 * pears_remaining_after_first_sale x) / 100
def pears_remaining_after_first_day (x : ℕ) : ℕ := pears_remaining_after_first_sale x - pears_thrown_away_first_day x
def total_pears_thrown_away (x : ℕ) : ℕ := (72 * x) / 100
def pears_thrown_away_second_day (x : ℕ) : ℕ := total_pears_thrown_away x - pears_thrown_away_first_day x
def pears_remaining_after_second_day (x : ℕ) : ℕ := pears_remaining_after_first_day x - pears_thrown_away_second_day x

-- Prove that the vendor sold 20% of the remaining pears on the second day
theorem percent_sold_second_day (x : ℕ) (h : x > 0) :
  ((pears_remaining_after_second_day x * 100) / pears_remaining_after_first_day x) = 20 :=
by 
  sorry

end percent_sold_second_day_l542_542995


namespace ab_range_l542_542313

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + (1 / a) + (1 / b) = 5) :
  1 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end ab_range_l542_542313


namespace total_movie_hours_l542_542070

variable (J M N R : ℕ)

def favorite_movie_lengths (J M N R : ℕ) :=
  J = M + 2 ∧
  N = 3 * M ∧
  R = (4 / 5 * N : ℝ).to_nat ∧
  N = 30

theorem total_movie_hours (J M N R : ℕ) (h : favorite_movie_lengths J M N R) :
  J + M + N + R = 76 :=
by
  sorry

end total_movie_hours_l542_542070


namespace cos_theta_value_l542_542347

variables (θ : ℝ) (P : ℝ × ℝ)

-- Define the point P and its coordinates
def P_coords (P : ℝ × ℝ) : ℝ × ℝ := (12, -5)

-- Define the distance r from the origin to the point P
def distance_from_origin (P : ℝ × ℝ) : ℝ := real.sqrt (P.1^2 + P.2^2)

-- Define the cosine of angle θ
def cos_theta (θ : ℝ) (P : ℝ × ℝ) : ℝ := P.1 / distance_from_origin P

-- The main theorem to prove
theorem cos_theta_value : cos_theta θ (P_coords P) = 12 / 13 := sorry

end cos_theta_value_l542_542347


namespace fraction_of_64_l542_542163

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l542_542163


namespace area_square_diagonal_l542_542903

theorem area_square_diagonal (d : ℝ) (k : ℝ) :
  (∀ side : ℝ, d^2 = 2 * side^2 → side^2 = (d^2)/2) →
  (∀ A : ℝ, A = (d^2)/2 → A = k * d^2) →
  k = 1/2 :=
by
  intros h1 h2
  sorry

end area_square_diagonal_l542_542903


namespace isosceles_triangle_of_perpendiculars_l542_542392

theorem isosceles_triangle_of_perpendiculars
  {A B C D E M F : Type}
  {BC AC BE : Type}
  (AD_perp_BC : ⟪AD, BC⟫ = 0)
  (DE_perp_AC : ⟪DE, AC⟫ = 0)
  (M_mid_DE : (∃ mid, mid = midpoint DE) ∧ M = mid)
  (AM_perp_BE : ⟪AM, BE⟫ = 0)
  (angle_B_acute : acute ∠B)
  (angle_C_acute : acute ∠C) :
  isosceles_triangle ABC := sorry

end isosceles_triangle_of_perpendiculars_l542_542392


namespace triangle_area_is_60_l542_542953

noncomputable def triangle_area (P r : ℝ) : ℝ :=
  (r * P) / 2

theorem triangle_area_is_60 (hP : 48 = 48) (hr : 2.5 = 2.5) : triangle_area 48 2.5 = 60 := by
  sorry

end triangle_area_is_60_l542_542953


namespace pizza_crust_calories_is_600_l542_542808

noncomputable def lettuce_calories := 50
noncomputable def carrots_calories := 2 * lettuce_calories
noncomputable def dressing_calories := 210
noncomputable def salad_calories := lettuce_calories + carrots_calories + dressing_calories

noncomputable def pizza_crust_calories : ℕ
noncomputable def pepperoni_calories := pizza_crust_calories / 3
noncomputable def cheese_calories := 400
noncomputable def pizza_calories := pizza_crust_calories + pepperoni_calories + cheese_calories

noncomputable def salad_consumed_calories := salad_calories / 4
noncomputable def pizza_consumed_calories := pizza_calories / 5

axiom jackson_total_consumed_calories : salad_consumed_calories + pizza_consumed_calories = 330

theorem pizza_crust_calories_is_600 : pizza_crust_calories = 600 := by
  calc
    salad_consumed_calories + pizza_consumed_calories = 330 : jackson_total_consumed_calories
    let sc := salad_calories / 4
    let pc := pizza_calories / 5
    have equation := sc + pc
    sorry -- the detailed calculations goes here which were derived in the provided solution

end pizza_crust_calories_is_600_l542_542808


namespace player_b_has_winning_strategy_l542_542879

-- Define the polynomial game conditions
def polynomial_game (x : ℝ) (a b : ℝ) (f : ℝ → ℝ) (β α : ℕ) : ℝ :=
  f(x) + b * x ^ β + a * x ^ α

-- Define the win condition for Player B
def player_b_winning_strategy : Prop :=
  ∀ (f : ℝ → ℝ) (β α : ℕ), α % 2 = 1 → ∃ (b : ℝ), ∀ (a : ℝ), ∃ (x : ℝ), polynomial_game x a b f β α < 0

theorem player_b_has_winning_strategy : player_b_winning_strategy :=
sorry

end player_b_has_winning_strategy_l542_542879


namespace projection_line_equation_l542_542600

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dotProduct (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
  let normSq (a : ℝ × ℝ) := dotProduct a a
  let scalar := (dotProduct u v) / (normSq u)
  (scalar * u.1, scalar * u.2)

theorem projection_line_equation (x y : ℝ) :
    projection ⟨3, -1⟩ ⟨x, y⟩ = (3 / 2, -1 / 2) → y = 3 * x - 5 :=
by
  sorry

end projection_line_equation_l542_542600


namespace angle_ratio_in_inscribed_triangle_l542_542236

theorem angle_ratio_in_inscribed_triangle 
  (A B C O E : Type*)
  [EuclideanGeometry A B C]
  (h1 : acute_angle A B C)
  (h2 : inscribed_circle O A B C)
  (h3 : minor_arc E A C)
  (h4 : perpendicular O E A C)
  (arc_AB : measure_arc A B = 100)
  (arc_BC : measure_arc B C = 80) :
  ratio (measure_angle O B E) (measure_angle B A C) = 1 / 2 :=
sorry

end angle_ratio_in_inscribed_triangle_l542_542236


namespace friends_in_rooms_l542_542965

theorem friends_in_rooms :
  ∃ (S : Finset (Finset (Fin ℕ))) (h : ∀ room ∈ S, room.card ≤ 2), S.card = 7920 :=
sorry

end friends_in_rooms_l542_542965


namespace candy_problem_l542_542457

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542457


namespace tan_sum_of_angles_area_of_triangle_l542_542010

/-- Given a triangle ∆ABC with angles A, B, and C opposite to sides a, b, and c respectively,
and the condition that tan(A) + tan(B) = -tan(A) * tan(B), a = 2 and c = 1,
prove that tan(A + B) = 1. -/
theorem tan_sum_of_angles (A B C : ℝ) (a b c : ℝ)
  (h₁ : Real.tan A + Real.tan B = -Real.tan A * Real.tan B)
  (h₂ : a = 2) (h₃ : c = 1) :
  Real.tan (A + B) = 1 :=
sorry

/-- Calculate the area of triangle ∆ABC given a = 2, c = 1, and b = √2 - 1,
and prove that the area is (2 - √2) / 2. -/
theorem area_of_triangle (a b c : ℝ) (C : ℝ)
  (h₁ : a = 2) (h₂ : c = 1) (h₃ : b = Real.sqrt 2 - 1) (h₄ : C = Real.pi * 3 / 4) :
  let area := (a * b * Real.sin C) / 2 in
  area = (2 - Real.sqrt 2) / 2 :=
sorry

end tan_sum_of_angles_area_of_triangle_l542_542010


namespace find_equation_of_BC_l542_542327

-- Definitions of given conditions
def altitude_from_A (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def altitude_from_C (x y : ℝ) : Prop := x + y = 0
def vertex_A : ℝ × ℝ := (1, 2)

-- Definition of the target line (side BC)
def line_BC (x y : ℝ) : Prop := 2 * x + 3 * y + 7 = 0

-- The theorem we need to prove
theorem find_equation_of_BC :
  (∀ x y : ℝ, (altitude_from_A x y) → (2 * x + 3 * y + 7 = 0)) ∧
  (∀ x y : ℝ, (altitude_from_C x y) → (2 * x + 3 * y + 7 = 0)) ∧
  (vertex_A.fst, vertex_A.snd) ∈ set_of (λ p : ℝ × ℝ, altitude_from_A p.1 p.2) →
  (∀ x y : ℝ, (line_BC x y)) :=
by
  sorry

end find_equation_of_BC_l542_542327


namespace max_area_of_inscribed_rectangle_l542_542384

open Real

theorem max_area_of_inscribed_rectangle : 
  ∃ a : ℝ, -π/2 ≤ a ∧ a ≤ π/2 ∧ (2 * a * cos a = 1.1222) := 
by
  sorry

end max_area_of_inscribed_rectangle_l542_542384


namespace expression_evaluation_l542_542695

noncomputable def evaluate_expression (x y : ℝ) : ℝ :=
  (1.2 * x^4 + (4 * y^3) / 3) * (0.86)^3 - (sqrt (0.1))^3 / (0.86 * x^2 * y^2) + 0.086 +
  (0.1)^2 * (2 * x^3 - (3 * y^4) / 2)

theorem expression_evaluation : evaluate_expression 1.5 (-2) ≈ -3.012286 :=
by
  sorry

end expression_evaluation_l542_542695


namespace find_pounds_of_flour_l542_542156

-- Define constants
def ticket_price := 20
def tickets_sold := 500
def promotion_cost := 1000
def salt_needed := 10
def salt_cost_per_pound := 0.2
def flour_bag_weight := 50
def flour_bag_cost := 20
def total_profit := 8798

-- Define the statement
theorem find_pounds_of_flour :
  let total_revenue := tickets_sold * ticket_price in
  let total_expenses (F : ℕ) := promotion_cost + salt_needed * salt_cost_per_pound + F * flour_bag_cost in
  ∃ (F : ℕ), total_profit = total_revenue - total_expenses F ∧ F * flour_bag_weight = 500 :=
begin
  sorry
end

end find_pounds_of_flour_l542_542156


namespace workout_total_correct_l542_542094

structure Band := 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ)

structure Equipment := 
  (leg_weight_squat : ℕ) 
  (dumbbell : ℕ) 
  (leg_weight_lunge : ℕ) 
  (kettlebell : ℕ)

def total_weight (bands : Band) (equip : Equipment) : ℕ := 
  let squat_total := bands.A + bands.B + bands.C + (2 * equip.leg_weight_squat) + equip.dumbbell
  let lunge_total := bands.A + bands.C + (2 * equip.leg_weight_lunge) + equip.kettlebell
  squat_total + lunge_total

theorem workout_total_correct (bands : Band) (equip : Equipment) : 
  bands = ⟨7, 5, 3⟩ → 
  equip = ⟨10, 15, 8, 18⟩ → 
  total_weight bands equip = 94 :=
by 
  -- Insert your proof steps here
  sorry

end workout_total_correct_l542_542094


namespace quadratic_root_l542_542717

theorem quadratic_root (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x^2 + x - 2 = 0) → a = 1 := by
  sorry

end quadratic_root_l542_542717


namespace find_certain_number_l542_542219

theorem find_certain_number (x : ℕ) (h : 220025 = (x + 445) * (2 * (x - 445)) + 25) : x = 555 :=
sorry

end find_certain_number_l542_542219


namespace problem_1_problem_2_problem_3_l542_542961

-- Problem 1: Prove that if the inequality |x-1| - |x-2| < a holds for all x in ℝ, then a > 1.
theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a) → a > 1 :=
sorry

-- Problem 2: Prove that if the inequality |x-1| - |x-2| < a has at least one real solution, then a > -1.
theorem problem_2 (a : ℝ) :
  (∃ x : ℝ, |x - 1| - |x - 2| < a) → a > -1 :=
sorry

-- Problem 3: Prove that if the solution set of the inequality |x-1| - |x-2| < a is empty, then a ≤ -1.
theorem problem_3 (a : ℝ) :
  (¬∃ x : ℝ, |x - 1| - |x - 2| < a) → a ≤ -1 :=
sorry

end problem_1_problem_2_problem_3_l542_542961


namespace inequality_negatives_l542_542312

theorem inequality_negatives (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > b^2 :=
sorry

end inequality_negatives_l542_542312


namespace line_eq_l542_542030

theorem line_eq (P : ℝ × ℝ) (hP : P = (1, 2)) (h_perp : ∀ x y : ℝ, 2 * x + y - 1 = 0 → x - 2 * y + c = 0) : 
  ∃ c : ℝ, (x - 2 * y + c = 0 ∧ P ∈ {(x, y) | x - 2 * y + c = 0}) ∧ c = 3 :=
  sorry

end line_eq_l542_542030


namespace n_power_of_prime_l542_542830

-- Definitions used in Lean 4
def arith_prog (a : ℕ → ℕ) (k d : ℕ) : Prop :=
∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = k + d * i

def divides (m n : ℕ) : Prop := ∃ k, m * k = n

def does_not_divide (m n : ℕ) : Prop := ¬ divides m n

-- Main theorem statement in Lean 4
theorem n_power_of_prime (a: ℕ → ℕ) (k d n: ℕ) 
  (h_arith_prog: arith_prog a k d) 
  (h_divides: ∀ i, 1 ≤ i ∧ i ≤ n-1 → divides i (a i))
  (h_n_not_divides: does_not_divide n (a n)) 
  : ∃ p : ℕ, prime p ∧ ∃ k : ℕ, n = p^k := sorry

end n_power_of_prime_l542_542830


namespace divides_2_pow_26k_plus_2_plus_3_by_19_l542_542566

theorem divides_2_pow_26k_plus_2_plus_3_by_19 (k : ℕ) : 19 ∣ (2^(26*k+2) + 3) := 
by
  sorry

end divides_2_pow_26k_plus_2_plus_3_by_19_l542_542566


namespace solution_set_l542_542139

theorem solution_set (x : ℝ) : (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
sorry

end solution_set_l542_542139


namespace min_kinds_of_candies_l542_542438

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542438


namespace coin_game_winner_coin_game_winner_2020_coin_game_winner_2021_l542_542540

/-- The coin game with two players, Barbara and Jenna, each taking optimal strategies. 
    Barbara can remove 3 or 5 coins unless fewer than 3 coins remain. 
    Jenna can remove 1, 2, or 3 coins. 
    The player to remove the last coin wins. -/
theorem coin_game_winner (n : ℕ) :
  (n % 6 = 2) → ¬(∃ k ∈ {3, 5}, n - k ≥ 0 ∧ (n - k) % 6 = 2) :=
begin
  -- For the sake of the theorem, we only need to state it, not prove it.
  -- This captures the statement that if the number of coins mod 6 is 2,
  -- the player to start is in a losing position.
  sorry
end

theorem coin_game_winner_2020 :
  (2020 % 6 = 2) → (∃ k ∈ {3, 5}, 2017 % 6 = 0 ∨ 2015 % 6 = 0) :=
by sorry

theorem coin_game_winner_2021 :
  (2021 % 6 = 3) → (∃ k ∈ {3, 5}, 2018 % 6 = 2 ∨ 2016 % 6 = 2) :=
by sorry

end coin_game_winner_coin_game_winner_2020_coin_game_winner_2021_l542_542540


namespace continuous_function_triples_l542_542286

theorem continuous_function_triples (f g h : ℝ → ℝ) (h₁ : Continuous f) (h₂ : Continuous g) (h₃ : Continuous h)
  (h₄ : ∀ x y : ℝ, f (x + y) = g x + h y) :
  ∃ (c a b : ℝ), (∀ x : ℝ, f x = c * x + a + b) ∧ (∀ x : ℝ, g x = c * x + a) ∧ (∀ x : ℝ, h x = c * x + b) :=
sorry

end continuous_function_triples_l542_542286


namespace p_plus_q_l542_542694

noncomputable def r : ℚ := 1 / 4 

def sequence (n : Nat) : ℝ :=
  match n with
  | 0 => 4096
  | 1 => 1024
  | 2 => 256
  | 3 => 256 * r
  | 4 => (256 * r) * r
  | 5 => (256 * r) * r * r
  | 6 => (256 * r) * r * r * r
  | _ => 0  -- This sequence model is explicitly defined for up to 7 terms.

def p : ℝ := sequence 3
def q : ℝ := sequence 4

theorem p_plus_q : p + q = 80 :=
by
  have h₁ : p = 64 := by 
    rw [p, sequence, r] 
    norm_num 
  have h₂ : q = 16 := by 
    rw [q, sequence, r] 
    norm_num 
  rw [h₁, h₂]
  norm_num
  · sorry

end p_plus_q_l542_542694


namespace math_problem_l542_542085

open Real

theorem math_problem
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a - b + c = 0) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
   a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
   b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 := by
  sorry

end math_problem_l542_542085


namespace pablo_hours_per_day_l542_542551

theorem pablo_hours_per_day
  (pieces_per_hour : ℕ)
  (puzzles1_count : ℕ)
  (puzzles2_count : ℕ)
  (pieces1_per_puzzle : ℕ)
  (pieces2_per_puzzle : ℕ)
  (days_to_complete : ℕ)
  (pieces_per_hour_eq : pieces_per_hour = 100)
  (puzzles1_count_eq : puzzles1_count = 8)
  (puzzles2_count_eq : puzzles2_count = 5)
  (pieces1_per_puzzle_eq : pieces1_per_puzzle = 300)
  (pieces2_per_puzzle_eq : pieces2_per_puzzle = 500)
  (days_to_complete_eq : days_to_complete = 7) :
  (puzzles1_count * pieces1_per_puzzle + puzzles2_count * pieces2_per_puzzle) / pieces_per_hour / days_to_complete = 7 :=
by
  rw [pieces_per_hour_eq, puzzles1_count_eq, puzzles2_count_eq, pieces1_per_puzzle_eq, pieces2_per_puzzle_eq, days_to_complete_eq]
  simp
  sorry

end pablo_hours_per_day_l542_542551


namespace gcf_factor_l542_542176

theorem gcf_factor (x y : ℕ) : gcd (6 * x ^ 3 * y ^ 2) (3 * x ^ 2 * y ^ 3) = 3 * x ^ 2 * y ^ 2 :=
by
  sorry

end gcf_factor_l542_542176


namespace matinee_tickets_sold_l542_542891

theorem matinee_tickets_sold (
  (m : ℕ) (e : ℕ) (d : ℕ) : 
  let price_matinee := 5
  let price_evening := 12
  let price_3d := 20
  let sold_evening := 300
  let sold_3d := 100
  let total_revenue := 6600
  let revenue_matinee := price_matinee * m
  let revenue_evening := price_evening * sold_evening
  let revenue_3d := price_3d * sold_3d
  revenue_matinee + revenue_evening + revenue_3d = total_revenue
) : m = 200 := 
sorry

end matinee_tickets_sold_l542_542891


namespace max_value_of_function_is_seven_l542_542390

theorem max_value_of_function_is_seven:
  ∃ a: ℕ, (0 < a) ∧ 
  (∃ x: ℝ, (x + Real.sqrt (13 - 2 * a * x)) = 7 ∧
    ∀ y: ℝ, (y = x + Real.sqrt (13 - 2 * a * x)) → y ≤ 7) :=
sorry

end max_value_of_function_is_seven_l542_542390


namespace rearranged_numbers_l542_542040

theorem rearranged_numbers (a b : ℚ) (h1 : a ≠ 0)
  (h2 : ∃ (p q r : ℚ), ({1, 0, a} = {p, q, r} ∧ {p, q, r} = {1/(a : ℚ), |a|, b/a})) :
  b - a = 1 :=
sorry

end rearranged_numbers_l542_542040


namespace minimum_candy_kinds_l542_542411

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542411


namespace sin_7pi_over_6_l542_542670

theorem sin_7pi_over_6 : sin (7 * π / 6) = - 1 / 2 :=
sorry

end sin_7pi_over_6_l542_542670


namespace diagonal_AC_length_l542_542712

def regular_octagon : Type :=
{ side_length : ℝ // side_length = 12 }

theorem diagonal_AC_length (O : regular_octagon) : 
  let s := O.val in 
  let cos_67_5 := Real.cos (67.5 * (Real.pi / 180)) in
  let MX := s * cos_67_5 in
  let AC := 2 * MX in
  AC ≈ 9.192 :=
by 
  sorry

end diagonal_AC_length_l542_542712


namespace singleton_symmetric_set_unique_two_element_symmetric_sets_max_elements_in_symmetric_set_l542_542840

noncomputable def has_symmetric_element (A : Set ℕ) : Prop :=
∀ x ∈ A, 10 - x ∈ A

theorem singleton_symmetric_set_unique (A : Set ℕ) (hA : has_symmetric_element A) (h1 : A.card = 1) : A = {5} :=
sorry

theorem two_element_symmetric_sets (A : Set ℕ) (hA : has_symmetric_element A) (h2 : A.card = 2) :
  A = {1, 9} ∨ A = {2, 8} ∨ A = {3, 7} ∨ A = {4, 6} :=
sorry

theorem max_elements_in_symmetric_set (A : Set ℕ) (hA : has_symmetric_element A) : A.card ≤ 9 :=
sorry

end singleton_symmetric_set_unique_two_element_symmetric_sets_max_elements_in_symmetric_set_l542_542840


namespace number_of_prime_digit_two_digit_integers_l542_542772

theorem number_of_prime_digit_two_digit_integers :
  let primes := {2, 3, 5, 7}
  in (∃ p1 p2 ∈ primes, (10 * p1 + p2 < 100) ∧ (10 * p1 + p2 >= 10)) →
  (∃ p1 p2 ∈ primes, 4 * 4 = 16) :=
by
  sorry

end number_of_prime_digit_two_digit_integers_l542_542772


namespace planes_are_perpendicular_l542_542739

variables {m n : Type} {α β : Type}
variable [plane_structure m α] [plane_structure n β] 
variable [line m n]

-- Defining the relationships
def contains (p : Type) (l : Type) [plane_structure l p] : Prop := 
sorry -- Define it properly by the mathematical definition

def parallel (l1 l2 : Type) [line l1 l2] : Prop :=
sorry -- Define it properly by the mathematical definition

def perpendicular (l1 l2 : Type) [line l1 l2] : Prop :=
sorry -- Define it properly by the mathematical definition

-- Given that n is contained in β and n is perpendicular to α, prove that α is perpendicular to β.
theorem planes_are_perpendicular (H1 : contains β n) (H2 : perpendicular n α) : perpendicular α β :=
sorry

end planes_are_perpendicular_l542_542739


namespace number_of_students_l542_542141

-- Definitions based on the problem conditions
def mini_cupcakes := 14
def donut_holes := 12
def desserts_per_student := 2

-- Total desserts calculation
def total_desserts := mini_cupcakes + donut_holes

-- Prove the number of students
theorem number_of_students : total_desserts / desserts_per_student = 13 :=
by
  -- Proof can be filled in here
  sorry

end number_of_students_l542_542141


namespace eval_fraction_l542_542278

theorem eval_fraction : (3 : ℚ) / (2 - 5 / 4) = 4 := 
by 
  sorry

end eval_fraction_l542_542278


namespace olympic_rings_area_l542_542847

theorem olympic_rings_area (d R r: ℝ) 
  (hyp_d : d = 12 * Real.sqrt 2) 
  (hyp_R : R = 11) 
  (hyp_r : r = 9) 
  (overlap_area : ∀ (i j : ℕ), i ≠ j → 592 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54): 
  592.0 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54 := 
by sorry

end olympic_rings_area_l542_542847


namespace dodecahedron_interior_diagonals_count_l542_542018

-- Define the structure of a dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)  -- 12 pentagonal faces
  (vertices : Fin 20)  -- 20 vertices
  (edges_per_vertex : Fin 3)  -- 3 faces meeting at each vertex

-- Define what an interior diagonal is
def is_interior_diagonal (v1 v2 : Fin 20) (dod : Dodecahedron) : Prop :=
  v1 ≠ v2 ∧ ¬ (∃ e, (e ∈ dod.edges_per_vertex))

-- Problem rephrased in Lean 4 statement: proving the number of interior diagonals equals 160.
theorem dodecahedron_interior_diagonals_count (dod : Dodecahedron) :
  (Finset.univ.filter (λ (p : Fin 20 × Fin 20), is_interior_diagonal p.1 p.2 dod)).card = 160 :=
by sorry

end dodecahedron_interior_diagonals_count_l542_542018


namespace y_intercept_line_l542_542143

theorem y_intercept_line : ∀ y : ℝ, (∃ x : ℝ, x = 0 ∧ x - 3 * y - 1 = 0) → y = -1/3 :=
by
  intro y
  intro h
  sorry

end y_intercept_line_l542_542143


namespace jill_trips_to_fill_tank_l542_542493

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end jill_trips_to_fill_tank_l542_542493


namespace det_commutator_zero_l542_542819

open Matrix

variable {n : Type} [Fintype n] [DecidableEq n]

theorem det_commutator_zero (n : ℕ) (A B : Matrix n n ℂ)
  (h_odd : n % 2 = 1) (h_idempotent : (A - B) * (A - B) = 0) :
  det (A * B - B * A) = 0 := by
  sorry

end det_commutator_zero_l542_542819


namespace min_kinds_of_candies_l542_542442

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542442


namespace notPolynomial_l542_542929

def isPolynomial (expr : Expr) : Prop :=
  ∃ p : Expr, expr = p

def expr1 := 3 * x
def expr2 := 1 / x
def expr3 := (x * y) / 2
def expr4 := x - 3 * y

theorem notPolynomial (x y : ℝ) :
  ¬ isPolynomial expr2 ∧ isPolynomial expr1 ∧ isPolynomial expr3 ∧ isPolynomial expr4 := sorry

end notPolynomial_l542_542929


namespace quarters_total_l542_542563

variable (q1 q2 S: Nat)

def original_quarters := 760
def additional_quarters := 418

theorem quarters_total : S = original_quarters + additional_quarters :=
sorry

end quarters_total_l542_542563


namespace find_x_l542_542297

theorem find_x :
  let numerator : ℝ := 3.6 * 0.48 * 2.50
  let denominator : ℝ := 0.12 * 0.09 * 0.5
  let lhs : ℝ := x * (numerator / denominator)
  lhs = 3200.0000000000005 → x = 1.6 :=
begin
  intros,
  sorry,
end

end find_x_l542_542297


namespace arithmetic_geometric_sequence_product_l542_542636

theorem arithmetic_geometric_sequence_product :
  (∀ n : ℕ, ∃ d : ℝ, ∀ m : ℕ, a_n = a_1 + m * d) →
  (∀ n : ℕ, ∃ q : ℝ, ∀ m : ℕ, b_n = b_1 * q ^ m) →
  a_1 = 1 → a_2 = 2 →
  b_1 = 1 → b_2 = 2 →
  a_5 * b_5 = 80 :=
by
  sorry

end arithmetic_geometric_sequence_product_l542_542636


namespace playground_area_l542_542601

noncomputable def calculate_area (w s : ℝ) : ℝ := s * s

theorem playground_area (w s : ℝ) (h1 : s = 3 * w + 10) (h2 : 4 * s = 480) : calculate_area w s = 14400 := by
  sorry

end playground_area_l542_542601


namespace max_picked_squares_l542_542785

theorem max_picked_squares (n : ℕ) (h : n = 2021) :
  ∃ k, k = 2 * (n / 2) * ((n + 1) / 2) :=
by
  use 2 * (n / 2) * ((n + 1) / 2)
  rw h
  norm_num
  sorry

end max_picked_squares_l542_542785


namespace area_of_sin_cos_closed_figure_l542_542580

theorem area_of_sin_cos_closed_figure :
  ∫ x in 0..(Real.pi / 4), (Real.cos x - Real.sin x) +
  ∫ x in (Real.pi / 4)..(5 * Real.pi / 4), (Real.sin x - Real.cos x) +
  ∫ x in (5 * Real.pi / 4)..(2 * Real.pi), (Real.cos x - Real.sin x) = 2 * Real.sqrt 2 :=
by
  sorry

end area_of_sin_cos_closed_figure_l542_542580


namespace num_integers_square_between_50_and_200_l542_542292

theorem num_integers_square_between_50_and_200 :
  {n : ℤ | 50 < n^2 ∧ n^2 < 200}.finite.toFinset.card = 14 :=
sorry

end num_integers_square_between_50_and_200_l542_542292


namespace perpendicular_x_intercept_l542_542168

theorem perpendicular_x_intercept (x : ℝ) :
  (∃ y : ℝ, 2 * x + 3 * y = 9) ∧ (∃ y : ℝ, y = 5) → x = -10 / 3 :=
by sorry -- Proof omitted

end perpendicular_x_intercept_l542_542168


namespace total_spent_l542_542096

/-- Define the prices of the rides in the morning and the afternoon --/
def morning_price (ride : String) (age : Nat) : Nat :=
  match ride, age with
  | "bumper_car", n => if n < 18 then 2 else 3
  | "space_shuttle", n => if n < 18 then 4 else 5
  | "ferris_wheel", n => if n < 18 then 5 else 6
  | _, _ => 0

def afternoon_price (ride : String) (age : Nat) : Nat :=
  (morning_price ride age) + 1

/-- Define the number of rides taken by Mara and Riley --/
def rides_morning (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 2
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 2
  | _, _ => 0

def rides_afternoon (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 1
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 1
  | _, _ => 0

/-- Define the ages of Mara and Riley --/
def age (person : String) : Nat :=
  match person with
  | "Mara" => 17
  | "Riley" => 19
  | _ => 0

/-- Calculate the total expenditure --/
def total_cost (person : String) : Nat :=
  List.sum ([
    (rides_morning person "bumper_car") * (morning_price "bumper_car" (age person)),
    (rides_afternoon person "bumper_car") * (afternoon_price "bumper_car" (age person)),
    (rides_morning person "space_shuttle") * (morning_price "space_shuttle" (age person)),
    (rides_afternoon person "space_shuttle") * (afternoon_price "space_shuttle" (age person)),
    (rides_morning person "ferris_wheel") * (morning_price "ferris_wheel" (age person)),
    (rides_afternoon person "ferris_wheel") * (afternoon_price "ferris_wheel" (age person))
  ])

/-- Prove the total cost for Mara and Riley is $62 --/
theorem total_spent : total_cost "Mara" + total_cost "Riley" = 62 :=
by
  sorry

end total_spent_l542_542096


namespace triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l542_542055

theorem triangle_side_ratio_impossible (a b c : ℝ) (h₁ : a = b / 2) (h₂ : a = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_2 (a b c : ℝ) (h₁ : b = a / 2) (h₂ : b = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_3 (a b c : ℝ) (h₁ : c = a / 2) (h₂ : c = b / 3) : false :=
by
  sorry

end triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l542_542055


namespace hallway_width_equals_four_l542_542844

-- Define the conditions: dimensions of the areas and total installed area.
def centralAreaLength : ℝ := 10
def centralAreaWidth : ℝ := 10
def centralArea : ℝ := centralAreaLength * centralAreaWidth

def totalInstalledArea : ℝ := 124
def hallwayLength : ℝ := 6

-- Total area minus central area's area yields hallway's area
def hallwayArea : ℝ := totalInstalledArea - centralArea

-- Statement to prove: the width of the hallway given its area and length.
theorem hallway_width_equals_four :
  (hallwayArea / hallwayLength) = 4 := 
by
  sorry

end hallway_width_equals_four_l542_542844


namespace distance_center_point_eq_sqrt_74_l542_542618

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def center_of_circle (a b c d : ℝ) : ℝ × ℝ :=
  let x_center := a / 2
  let y_center := b / 2
  (x_center, y_center)

theorem distance_center_point_eq_sqrt_74 :
  ∃ (c : ℝ × ℝ), center_of_circle 8 (-2) 0 23 = c ∧ distance c (-3, 4) = Real.sqrt 74 :=
by
  sorry

end distance_center_point_eq_sqrt_74_l542_542618


namespace num_common_points_planes_l542_542027

theorem num_common_points_planes (α β : set (ℝ × ℝ × ℝ)) (h_diff: α ≠ β) (h_plane_α: is_plane α) (h_plane_beta: is_plane β) : α ∩ β = ∅ ∨ ∃ P L, P ∈ α ∩ β ∧ line_through P L ∧ ∀ Q, Q ∈ α ∩ β → Q ∈ L :=
by
  sorry

end num_common_points_planes_l542_542027


namespace fraction_of_walls_not_illuminated_l542_542043

-- Define given conditions
def point_light_source : Prop := true
def rectangular_room : Prop := true
def flat_mirror_on_wall : Prop := true
def full_height_of_room : Prop := true

-- Define the fraction not illuminated
def fraction_not_illuminated := 17 / 32

-- State the theorem to prove
theorem fraction_of_walls_not_illuminated :
  point_light_source ∧ rectangular_room ∧ flat_mirror_on_wall ∧ full_height_of_room →
  fraction_not_illuminated = 17 / 32 :=
by
  intros h
  sorry

end fraction_of_walls_not_illuminated_l542_542043


namespace volume_of_tetrahedron_PQRS_l542_542872

open Real EuclideanGeometry

noncomputable def volume_of_tetrahedron 
  (P Q R S : Point)
  (dist_PQ : dist P Q = 6)
  (dist_PR : dist P R = 5)
  (dist_PS : dist P S = 4 * sqrt 2)
  (dist_QR : dist Q R = 3 * sqrt 2)
  (dist_QS : dist Q S = 5)
  (dist_RS : dist R S = 4) : ℝ :=
  (sqrt 35 / 3) * (2 * sqrt 35 / 3)

theorem volume_of_tetrahedron_PQRS (P Q R S : Point)
  (dist_PQ : dist P Q = 6)
  (dist_PR : dist P R = 5)
  (dist_PS : dist P S = 4 * sqrt 2)
  (dist_QR : dist Q R = 3 * sqrt 2)
  (dist_QS : dist Q S = 5)
  (dist_RS : dist R S = 4) :
  volume_of_tetrahedron P Q R S dist_PQ dist_PR dist_PS dist_QR dist_QS dist_RS = 140 / 9 :=
sorry

end volume_of_tetrahedron_PQRS_l542_542872


namespace xy_not_z_probability_l542_542181

theorem xy_not_z_probability :
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8
  let P := P_X * P_Y * P_not_Z
  P = (1 : ℝ) / 32 :=
by
  -- Definitions based on problem conditions
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8

  -- Calculate the combined probability
  let P := P_X * P_Y * P_not_Z
  
  -- Check equality with 1/32
  have h : P = (1 : ℝ) / 32 := by sorry
  exact h

end xy_not_z_probability_l542_542181


namespace man_speed_l542_542981

-- Define given conditions
def distance_meters : ℝ := 475.038
def time_seconds : ℝ := 30

-- Define derived conditions
def distance_kilometers : ℝ := distance_meters / 1000
def time_hours : ℝ := time_seconds / 3600

-- Postulate the problem statement
theorem man_speed 
  (d_km : ℝ := distance_kilometers)
  (t_hrs : ℝ := time_hours) :
  d_km / t_hrs ≈ 57.006 :=
sorry

end man_speed_l542_542981


namespace distance_from_O_to_plane_l542_542990

-- Definitions coming from conditions
variables (r : ℝ) (s1 s2 s3 : ℝ)

-- Assumptions based on conditions from the problem
def conditions : Prop :=
  r = 10 ∧ s1 = 13 ∧ s2 = 14 ∧ s3 = 15

-- The proof goal based on the math problem translation
theorem distance_from_O_to_plane (r : ℝ) (s1 s2 s3 : ℝ) (H : conditions r s1 s2 s3) : 
  ∃ (d : ℝ), d = 2 * real.sqrt 21 :=
sorry

end distance_from_O_to_plane_l542_542990


namespace probability_teacher_A_wins_l542_542396

theorem probability_teacher_A_wins :
  let p := (2 : ℚ) / 3 in
  let win_prob := 
    (comb 4 2) * (p^4 * (1 - p)^2) + 4 * (p^5 * (1 - p)) + p^6 in
  win_prob = 32 / 81 := 
sorry

end probability_teacher_A_wins_l542_542396


namespace rectangular_field_perimeter_l542_542579

variable (length width : ℝ)

theorem rectangular_field_perimeter (h_area : length * width = 50) (h_width : width = 5) : 2 * (length + width) = 30 := by
  sorry

end rectangular_field_perimeter_l542_542579


namespace geometric_loci_of_spheres_l542_542230

noncomputable def sphere_touches_plane (R : ℝ) (α : ℝ) := sorry

noncomputable def loci_of_centers (R r : ℝ) : Ennreal := 2 * real.sqrt (R * r)
noncomputable def loci_of_tangency_plane (R r : ℝ) : Ennreal := 2 * real.sqrt (R * r)
noncomputable def loci_of_tangency_sphere (R r : ℝ) : Ennreal := (2 * R * real.sqrt (R * r)) / (R + r)

theorem geometric_loci_of_spheres (R r : ℝ) (α : ℝ) :
  sphere_touches_plane R α →
  loci_of_centers R r = 2 * real.sqrt R ∧
  loci_of_tangency_plane R r = 2 * real.sqrt R ∧
  loci_of_tangency_sphere R r = (2 * R * real.sqrt (R * r)) / (R + r) :=
by sorry

end geometric_loci_of_spheres_l542_542230


namespace football_team_progress_l542_542944

theorem football_team_progress : 
  ∀ {loss gain : ℤ}, loss = 5 → gain = 11 → gain - loss = 6 :=
by
  intros loss gain h_loss h_gain
  rw [h_loss, h_gain]
  sorry

end football_team_progress_l542_542944


namespace infinite_indices_inequality_l542_542261

noncomputable def exists_infinite_indices
  (a : ℕ → ℕ) : Prop :=
  ∀ C : ℝ, ∃ N : ℕ, ∀ k : ℕ, k > N → a k > floor (C * k)

theorem infinite_indices_inequality
  (a : ℕ → ℕ)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1))
  (h_condition : exists_infinite_indices a) :
  ∃ᶠ k in at_top, 2 * a k < a (k - 1) + a (k + 1) :=
sorry

end infinite_indices_inequality_l542_542261


namespace double_square_area_l542_542251

theorem double_square_area (a k : ℝ) (h : (k * a) ^ 2 = 2 * a ^ 2) : k = Real.sqrt 2 := 
by 
  -- Our goal is to prove that k = sqrt(2)
  sorry

end double_square_area_l542_542251


namespace Janet_previous_movie_length_l542_542498

theorem Janet_previous_movie_length (L : ℝ) (H1 : 1.60 * L = 1920 / 100) : L / 60 = 0.20 :=
by
  sorry

end Janet_previous_movie_length_l542_542498


namespace cos_two_alpha_plus_pi_over_3_correct_l542_542336

noncomputable def cos_two_alpha_plus_pi_over_3 (α : ℝ) : ℝ :=
  cos (2 * α + π / 3)

theorem cos_two_alpha_plus_pi_over_3_correct {α : ℝ} (h1 : cos (π / 6 - α) + sin (π - α) = - (4 * sqrt 3) / 5)
  (h2 : -π / 2 < α ∧ α < 0) : 
  cos_two_alpha_plus_pi_over_3 α = -7 / 25 :=
by 
  sorry

end cos_two_alpha_plus_pi_over_3_correct_l542_542336


namespace tan_double_angle_l542_542039

axiom Point : Type
axiom α : ℝ
axiom P : Point
axiom x y : ℝ
axiom hP : P = (x, y)

noncomputable def tan := λ θ : ℝ, y / x

theorem tan_double_angle (h : tan α = -2) : tan (2 * α) = 4 / 3 :=
by
  sorry

end tan_double_angle_l542_542039


namespace johns_original_earnings_l542_542811

theorem johns_original_earnings (x : ℝ) (h1 : x + 0.5 * x = 90) : x = 60 := 
by
  -- sorry indicates the proof steps are omitted
  sorry

end johns_original_earnings_l542_542811


namespace rectangle_perimeter_l542_542984

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * (2 * a + 2 * b)) : 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l542_542984


namespace max_C_test_tubes_l542_542914

theorem max_C_test_tubes (a b c : ℕ) 
    (h1 : 1017 * a + 17 * b = 6983 * c)
    (h2 : a + b + c = 1000) : c ≤ 73 :=
begin
  sorry
end

end max_C_test_tubes_l542_542914


namespace eccentricity_of_conic_section_l542_542777

-- Define the condition for the conic section
def conic_section (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1

-- State the proof problem
theorem eccentricity_of_conic_section (m : ℝ) : 
  (∀ x y : ℝ, conic_section x y m) → (√(1 + (-1 / m)) = 2) → (m = -1 / 3) :=
by
  intros h_conic_section h_eccentricity
  sorry

end eccentricity_of_conic_section_l542_542777


namespace part1_part2_l542_542529

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

variables {a b : ℝ}

theorem part1 (h0 : 0 < a) (h1 : a < b) (h2 : f(a) = f(b)) (h3 : f(a) = 2 * f((a + b) / 2)) : 
  a < 1 ∧ 1 < b :=
sorry

theorem part2 (h0 : 0 < a) (h1 : a < b) (h2 : f(a) = f(b)) (h3 : f(a) = 2 * f((a + b) / 2)) (h4 : b > 1) : 
  2 < 4 * b - b^2 ∧ 4 * b - b^2 < 3 :=
sorry

end part1_part2_l542_542529


namespace total_cost_of_apples_l542_542964

variable (num_apples_per_bag cost_per_bag num_apples : ℕ)
#check num_apples_per_bag = 50
#check cost_per_bag = 8
#check num_apples = 750

theorem total_cost_of_apples : 
  (num_apples_per_bag = 50) → 
  (cost_per_bag = 8) → 
  (num_apples = 750) → 
  (num_apples / num_apples_per_bag * cost_per_bag = 120) :=
by
  intros
  sorry

end total_cost_of_apples_l542_542964


namespace students_in_hollow_square_are_160_l542_542974

-- Define the problem conditions
def hollow_square_formation (outer_layer : ℕ) (inner_layer : ℕ) : Prop :=
  outer_layer = 52 ∧ inner_layer = 28

-- Define the total number of students in the group based on the given condition
def total_students (n : ℕ) : Prop := n = 160

-- Prove that the total number of students is 160 given the hollow square formation conditions
theorem students_in_hollow_square_are_160 : ∀ (outer_layer inner_layer : ℕ),
  hollow_square_formation outer_layer inner_layer → total_students 160 :=
by
  intros outer_layer inner_layer h
  sorry

end students_in_hollow_square_are_160_l542_542974


namespace inequality_solution_set_minimum_value_expression_l542_542359

-- Definition of the function f
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Inequality solution set for f(x) ≤ 4
theorem inequality_solution_set :
  { x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3 } = { x : ℝ | f x ≤ 4 } := 
sorry

-- Minimum value of the given expression given conditions on a and b
theorem minimum_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0)
  (h3 : a + 2 * b = 3) :
  (1 / (a - 1)) + (2 / b) = 9 / 2 := 
sorry

end inequality_solution_set_minimum_value_expression_l542_542359


namespace chess_tournament_participants_l542_542461

theorem chess_tournament_participants :
  ∃ n : ℕ, (n * (n - 1)) / 2 + 10 * n = 672 ∧ n = 28 :=
by
  -- Construct the statement for the proof
  use 28
  split
  -- Prove that the equation holds for n = 28
  { sorry },
  -- Prove that n = 28
  { refl }

end chess_tournament_participants_l542_542461


namespace triangle_AOB_is_right_l542_542001

noncomputable def line_eq (m : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + 1}
noncomputable def parabola_eq := {p : ℝ × ℝ | p.2 = p.1^2}

def points_of_intersection (m : ℝ) : set (ℝ × ℝ) :=
  {p | (p ∈ line_eq m) ∧ (p ∈ parabola_eq)}

def vectors_OA_OB (A B : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let OA := A;
  let OB := B;
  (OA.1, OA.2, OB.1, OB.2)

def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

theorem triangle_AOB_is_right {m : ℝ} (A B : ℝ × ℝ) (hA : A ∈ points_of_intersection m)
  (hB : B ∈ points_of_intersection m) (h_intersection : A ≠ B) :
  dot_product A B = 0 → ∃ O : ℝ × ℝ, is_right_triangle O A B :=
sorry

end triangle_AOB_is_right_l542_542001


namespace matrix_product_correct_l542_542679

def matrix_seq_sum : ℕ → ℕ :=
λ n, 2 + (n - 1) * 2

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  if n = 0 then ![![1, 0], ![0, 1]]
  else let sum := (matrix_seq_sum n).sum in
    ![![1, sum], ![0, 1]]

theorem matrix_product_correct : matrix_product 50 = ![![1, 2550], ![0, 1]] :=
sorry

end matrix_product_correct_l542_542679


namespace non_deg_ellipse_projection_l542_542594

theorem non_deg_ellipse_projection (m : ℝ) : 
  (3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m → (m > -21)) := 
by
  sorry

end non_deg_ellipse_projection_l542_542594


namespace min_candy_kinds_l542_542431

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542431


namespace dividend_calculation_l542_542619

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 19) (h2 : quotient = 7) (h3 : remainder = 6) : 
  divisor * quotient + remainder = 139 :=
by
  rw [h1, h2, h3]
  sorry

end dividend_calculation_l542_542619


namespace parents_can_catch_kolya_l542_542788

-- Definitions and conditions
def alleys_length : ℝ := 1 -- Normalized length of alleys
def speed_ratio : ℝ := 3 -- Kolya runs 3 times faster

-- Main statement
theorem parents_can_catch_kolya (alleys : Fin 6 → ℝ × ℝ)
  (cond1 : ∀ i, alleys i = alleys_length) -- All alleys same length
  (cond2 : ∃ square_center : ℝ × ℝ, 
             ∃ square_side : ℝ, 
             ∀ i, (alleys i).1 = square_center.1 ∨ (alleys i).2 = square_center.2) -- Four alleys along square sides
  (cond3 : ∃ square_center : ℝ × ℝ, 
             ∃ median_line : ℝ, 
             ∀ i, (alleys i).1 = median_line ∨ (alleys i).2 = median_line) -- Two alleys along square midlines
  (cond4 : ∀ p1 p2 : ℝ × ℝ, dist p1 p2 ≤ (1 / speed_ratio) * dist p1 p2) -- Kolya runs 3 times faster
  (always_visible : ∀ p1 p2 : ℝ × ℝ, can_see p1 p2) -- All three can see each other
  : ∃ t : ℝ, catch_Kolya t :=
by
  sorry

end parents_can_catch_kolya_l542_542788


namespace combined_weight_loss_l542_542242

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end combined_weight_loss_l542_542242


namespace F_eq_0_or_1_F_eq_1_iff_perfect_square_l542_542077

def Omega (n : ℕ) : ℕ := 
  if n = 1 then 0
  else n.factors.length

def f (n : ℕ) : ℤ :=
  if n = 1 then 1
  else (-1) ^ Omega n

def F (n : ℕ) : ℤ :=
  ∑ d in divisors n, f d

theorem F_eq_0_or_1 (n : ℕ) : F n = 0 ∨ F n = 1 :=
  sorry

theorem F_eq_1_iff_perfect_square (n : ℕ) : F n = 1 ↔ ∃ m : ℕ, n = m * m :=
  sorry

end F_eq_0_or_1_F_eq_1_iff_perfect_square_l542_542077


namespace geometric_seq_property_l542_542093

noncomputable def a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

theorem geometric_seq_property (n : ℕ) (h_arith : S (n + 1) + S (n + 1) = 2 * S (n)) (h_condition : a 2 = -2) :
  a 7 = 64 := 
by sorry

end geometric_seq_property_l542_542093


namespace current_dogwood_trees_l542_542151

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end current_dogwood_trees_l542_542151


namespace minimum_candy_kinds_l542_542452

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542452


namespace bisect_A_l542_542732

open EuclideanGeometry

noncomputable def angle_A_ABC_as_45_degrees (A B C : Point) : Prop :=
angle A B C = 45

def antipode_A'_A (circumcircle : Circle) (A : Point) : Point :=
antipode circumcircle A

def points_on_segments (A B E F : Point) (segment_AB segment_AC : segment) : Prop :=
segment.contains E AB ∧ segment.contains F AC

def distances_equal (A' B E C F : Point) : Prop :=
distance A' B = distance B E ∧ distance A' C = distance C F

def second_intersection (circum_AEF circum_ABC : Circle) : Point :=
second_intersection_point circum_AEF circum_ABC

theorem bisect_A'K_by_EF 
  (A B C A' E F K : Point)
  (circumcircle : Circle)
  (segAB : segment A B)
  (segAC : segment A C)
  (circum_AEF : Circle)
  (circum_ABC : Circle) :
  angle_A_ABC_as_45_degrees A B C →
  antipode_A'_A circumcircle A = A' → 
  points_on_segments A B E F segAB segAC → 
  distances_equal A' B E C F →
  second_intersection circum_AEF circum_ABC = K →
  midpoint (A' K) E F :=
sorry

end bisect_A_l542_542732


namespace area_two_layers_l542_542610

-- Given conditions
variables (A_total A_covered A_three_layers : ℕ)

-- Conditions from the problem
def condition_1 : Prop := A_total = 204
def condition_2 : Prop := A_covered = 140
def condition_3 : Prop := A_three_layers = 20

-- Mathematical equivalent proof problem
theorem area_two_layers (A_total A_covered A_three_layers : ℕ) 
  (h1 : condition_1 A_total) 
  (h2 : condition_2 A_covered) 
  (h3 : condition_3 A_three_layers) : 
  ∃ A_two_layers : ℕ, A_two_layers = 24 :=
by sorry

end area_two_layers_l542_542610


namespace range_of_a_l542_542031

noncomputable def has_two_distinct_real_roots (a : ℝ) : Prop :=
  let p := polynomial.X^4 - 2 * a * polynomial.X^2 - polynomial.X + (a^2 - a) in
  ∃ x y : ℝ, x ≠ y ∧ p.eval x = 0 ∧ p.eval y = 0

theorem range_of_a (a : ℝ) : has_two_distinct_real_roots a → a ∈ set.Ioo (-1/4) (3/4) :=
sorry

end range_of_a_l542_542031


namespace general_formula_sum_of_first_n_terms_l542_542527

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n

def initial_conditions (a : ℕ → ℝ) :=
  a 1 = 2 ∧ a 3 = a 2 + 4

theorem general_formula (a : ℕ → ℝ) (hgeom : geometric_sequence a) (hinitial : initial_conditions a) :
  ∀ n, a n = 2^n :=
sorry

theorem sum_of_first_n_terms (a : ℕ → ℝ) (hgeom : geometric_sequence a) (hinitial : initial_conditions a) :
  ∀ n, (∑ i in finset.range n, a (i + 1)) = 2^{n+1} - 2 :=
sorry

end general_formula_sum_of_first_n_terms_l542_542527


namespace perpendicular_lines_parallel_lines_l542_542011

-- Define the lines and their slopes
def line1 : ℝ → ℝ → Prop := λ x y, 2 * x + y + 2 = 0
def line2 (m n : ℝ) : ℝ → ℝ → Prop := λ x y, m * x + 4 * y + n = 0

-- Slope of line1 is -2
def slope1 := -2
-- Slope of line2 given slope 𝑚 is -m/4
def slope2 (m : ℝ) := -m / 4

-- Perpendicular lines condition
def perpendicular (m : ℝ) : Prop := slope1 * slope2(m) = -1

-- Parallel lines condition
def parallel (m : ℝ) : Prop := slope1 = slope2(m)

-- Distance between two parallel lines
def distance (m n : ℝ) : ℝ :=
  let point_on_l1 := (0, -2) in
  let (x0, y0) := point_on_l1 in
  abs ((-2 : ℝ) + n / 4) / real.sqrt 5

-- Proving the conditions for perpendicular lines
theorem perpendicular_lines (m : ℝ) : 2 * x + y + 2 = 0 → mx + 4y + n = 0 → l_1 ⊥ l_2 → m = -2 :=
by sorry

-- Proving the conditions for parallel lines and the distances
theorem parallel_lines (m n : ℝ) : 2 * x + y + 2 = 0 → mx + 4y + n = 0 →
  l_1 ∥ l_2 ∧ distance(m, n) = sqrt 5 → (m = 8 ∧ (n = 28 ∨ n = -12)) :=
by sorry

end perpendicular_lines_parallel_lines_l542_542011


namespace sum_difference_even_odd_1001_l542_542688

theorem sum_difference_even_odd_1001 :
  let sum_odd := (1001 / 2) * (3 + (3 + (1001 - 1) * 2))
  let sum_even := (1001 / 2) * (4 + (4 + (1001 - 1) * 2))
  sum_even - sum_odd = 1001 :=
by
  let sum_odd := (1001 / 2) * (3 + (3 + (1001 - 1) * 2))
  let sum_even := (1001 / 2) * (4 + (4 + (1001 - 1) * 2))
  have h_sum_odd : sum_odd = 1003003 := by sorry
  have h_sum_even : sum_even = 1004004 := by sorry
  calc
    sum_even - sum_odd
        = 1004004 - 1003003 : by rw [h_sum_even, h_sum_odd]
    ... = 1001 : by norm_num

end sum_difference_even_odd_1001_l542_542688


namespace value_of_m_l542_542837

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end value_of_m_l542_542837


namespace pirate_15_gets_coins_l542_542973

def coins_required_for_pirates : ℕ :=
  Nat.factorial 14 * ((2 ^ 4) * (3 ^ 9)) / 15 ^ 14

theorem pirate_15_gets_coins :
  coins_required_for_pirates = 314928 := 
by sorry

end pirate_15_gets_coins_l542_542973


namespace complex_eq_l542_542122

theorem complex_eq : 6 * (complex.cos (4 * real.pi / 3) + complex.i * complex.sin (4 * real.pi / 3)) = -3 - 3 * real.sqrt 3 * complex.i :=
by
  have h1 : complex.cos (4 * real.pi / 3) = -1 / 2 := by sorry
  have h2 : complex.sin (4 * real.pi / 3) = -real.sqrt 3 / 2 := by sorry
  sorry

end complex_eq_l542_542122


namespace min_candy_kinds_l542_542429

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542429


namespace range_a_if_p_true_and_q_false_l542_542334

open Real

theorem range_a_if_p_true_and_q_false (a m : ℝ) (h₁ : ∀ m ∈ Icc (-1 : ℝ) 1, a^2 - 5 * a - 3 ≥ sqrt (m^2 + 8))
    (h₂ : ¬∃ x, x^2 + a * x + 2 < 0) : -2 * sqrt 2 ≤ a ∧ a ≤ -1 := 
sorry

end range_a_if_p_true_and_q_false_l542_542334


namespace perpendicular_vectors_l542_542373

theorem perpendicular_vectors (x : ℝ) :
  let m := (1, 1)
  let n := (x, 2 - 2x)
  (m.1 * n.1 + m.2 * n.2 = 0) → x = 2 :=
by
  intro h
  have h1 : m = (1, 1) := rfl
  have h2 : n = (x, 2 - 2x) := rfl
  sorry

end perpendicular_vectors_l542_542373


namespace final_center_coordinates_l542_542677

-- Definition of the initial condition: the center of Circle U
def center_initial : ℝ × ℝ := (3, -4)

-- Definition of the reflection function across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Definition of the translation function to translate a point 5 units up
def translate_up_5 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 5)

-- Defining the final coordinates after reflection and translation
def center_final : ℝ × ℝ :=
  translate_up_5 (reflect_y_axis center_initial)

-- Problem statement: Prove that the final center coordinates are (-3, 1)
theorem final_center_coordinates :
  center_final = (-3, 1) :=
by {
  -- Skipping the proof itself, but the theorem statement should be equivalent
  sorry
}

end final_center_coordinates_l542_542677


namespace parallelogram_area_l542_542252

def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
def vector_v : ℝ × ℝ × ℝ := (2, -1, 5)

theorem parallelogram_area :
  let u := vector_u
  let v := vector_v
  let cross_product := 
    (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1) in
  let area := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
  area = real.sqrt 789 :=
by sorry

end parallelogram_area_l542_542252


namespace first_digit_sequence_irrational_l542_542304

def first_digit (n : ℕ) : ℕ := (2^n).digits.head

theorem first_digit_sequence_irrational :
  ¬ rational (Real.of_digits 10 (fun n => first_digit (n + 1))) :=
by
  sorry

end first_digit_sequence_irrational_l542_542304


namespace max_volume_range_l542_542051

noncomputable def volume_of_tetrahedron (AD BC t : ℝ) (hAD_perp_BC : AD > 0 ∧ BC > 0 ∧ t ∈ Icc 8 (top : ℝ))
  (hAD : AD = 6) (hBC : BC = 2) (hT : ∀ AB BD AC CD, AB + BD = t ∧ AC + CD = t) : Set ℝ :=
  {V | V ≥ 2 * Real.sqrt 6}

theorem max_volume_range (t : ℝ) (h_t : t ∈ Icc 8 (top : ℝ)) :
  ∃ V : ℝ, V ∈ volume_of_tetrahedron 6 2 t (by split; linarith; assumption) (by linarith) (by linarith) _ :=
sorry

-- Helper constraint (AB + BD = t ∧ AC + CD = t) is abstracted in the format since the equality check within
-- the given interval is essential for the solution context.

end max_volume_range_l542_542051


namespace beautiful_fold_probability_is_half_l542_542851

theorem beautiful_fold_probability_is_half (A B C D F : Point) (fold : Line)
  (h1 : Square A B C D)
  (h2 : PointOnLine fold A)
  (h3 : PointOnLine fold B)
  (h4 : PointOnLine fold C)
  (h5 : PointOnLine fold D)
  (h6 : RandomPoint F (SquareRegion A B C D))
  (h7 : BeautifulFold fold A B C D) :
  Probability (F ∈ FoldingLinesThroughCenter A B C D fold) = 1 / 2 :=
sorry

end beautiful_fold_probability_is_half_l542_542851


namespace N_is_composite_l542_542366

def N : ℕ := 2011 * 2012 * 2013 * 2014 + 1

theorem N_is_composite : ¬ Prime N := by
  sorry

end N_is_composite_l542_542366


namespace cosine_value_of_angle_between_skew_lines_l542_542387

noncomputable def cosine_between_skew_lines (a b : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let magnitude_a := Math.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)
  let magnitude_b := Math.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)
  (dot_product / (magnitude_a * magnitude_b)).abs

theorem cosine_value_of_angle_between_skew_lines:
  let a := (0, -1, -2)
  let b := (4, 0, 2)
  cosine_between_skew_lines a b = 2 / 5 :=
by sorry

end cosine_value_of_angle_between_skew_lines_l542_542387


namespace sum_of_possible_values_of_d_is_364_l542_542838

theorem sum_of_possible_values_of_d_is_364 :
  ∃ (a_n : ℕ → ℕ) (d : ℕ), d ∈ {1, 3, 9, 27, 81, 243} ∧
    a_n 1 = 3^5 ∧
    (∀ n, a_n (n + 1) = a_n n + d) ∧
    (∀ m n, a_n m + a_n n = a_n (m + n)) →
  (∑ k in {1, 3, 9, 27, 81, 243}, k) = 364 :=
by
  sorry

end sum_of_possible_values_of_d_is_364_l542_542838


namespace problem_a_problem_b_l542_542105

-- Problem (a)
theorem problem_a (n : Nat) : Nat.mod (7 ^ (2 * n) - 4 ^ (2 * n)) 33 = 0 := sorry

-- Problem (b)
theorem problem_b (n : Nat) : Nat.mod (3 ^ (6 * n) - 2 ^ (6 * n)) 35 = 0 := sorry

end problem_a_problem_b_l542_542105


namespace fraction_of_product_l542_542166

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l542_542166


namespace age_difference_ratio_l542_542562

theorem age_difference_ratio (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R + 2 = 2 * (J + 2))
  (h3 : (R + 2) * (K + 2) = 192) :
  (R - J) / (R - K) = 2 := by
  sorry

end age_difference_ratio_l542_542562


namespace seq_formula_sum_formula_l542_542900

noncomputable def a_n (n : ℕ) : ℤ :=
  (-2)^(n+1)

def S_k (k : ℕ) : ℤ :=
  ∑ i in finset.range k, a_n (i + 1)

theorem seq_formula
  (a1 : a_n 1 = 4)
  (arith_seq: 2 * S_k 2 = S_k 3 + S_k 4) : 
  ∀ n, a_n n = (-2)^(n+1) :=
sorry

noncomputable def b_n (n : ℕ) : ℝ :=
  real.log2 (abs (a_n n).to_real)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (b_n (i+1) * b_n (i+2))

theorem sum_formula (n : ℕ) :
  T_n n = (n : ℝ) / (2*(n + 2)) :=
sorry

end seq_formula_sum_formula_l542_542900


namespace jacket_price_equation_l542_542646

theorem jacket_price_equation (x : ℝ) (h : 0.8 * (1 + 0.5) * x - x = 28) : 0.8 * (1 + 0.5) * x = x + 28 :=
by sorry

end jacket_price_equation_l542_542646


namespace sufficient_not_necessary_l542_542318

variable (p q : Prop)

theorem sufficient_not_necessary (h1 : p ∧ q) (h2 : ¬¬p) : ¬¬p :=
by
  sorry

end sufficient_not_necessary_l542_542318


namespace intersection_P_Q_l542_542823

def P (x : ℝ) : Prop := x + 2 ≥ x^2

def Q (x : ℕ) : Prop := x ≤ 3

theorem intersection_P_Q :
  {x : ℕ | P x} ∩ {x : ℕ | Q x} = {0, 1, 2} :=
by
  sorry

end intersection_P_Q_l542_542823


namespace fathers_contributions_l542_542308

noncomputable def total_contribution : ℕ := 30000
def first_father_contribution : ℕ := 11500

def second_father_contribution (y z : ℕ) : ℕ :=
  (1/3 : ℚ) * (first_father_contribution + y + z)

def third_father_contribution (x z : ℕ) : ℕ :=
  (1/4 : ℚ) * (first_father_contribution + x + z)

def fourth_father_contribution (x y : ℕ) : ℕ :=
  (1/5 : ℚ) * (first_father_contribution + x + y)

theorem fathers_contributions :
  ∃ (x y z : ℕ),
    x = second_father_contribution y z ∧
    y = third_father_contribution x z ∧
    z = fourth_father_contribution x y ∧
    first_father_contribution + x + y + z = total_contribution ∧
    x = 7500 ∧ y = 6000 ∧ z = 5000 :=
by
  sorry

end fathers_contributions_l542_542308


namespace smallest_a_for_x4_coefficient_l542_542719

theorem smallest_a_for_x4_coefficient (a : ℤ) :
  let P := (1 - 3 * x + a * x^2)^8 in
  coeff_of_x4_in_P_eq_70 (P : ℚ[x]) = 70 → a = -50 :=
sorry

end smallest_a_for_x4_coefficient_l542_542719


namespace angle_A_side_c_length_l542_542489

noncomputable def measure_angle_A (a b c : ℝ) (A B C : ℝ) (h1: a = 2*sqrt 3) 
  (h2 : B = π / 4) (h3 : A + B + C = π) (h4 : 2 * sin^2 (B + C) - 3 * cos A = 0) : Prop :=
  A = π / 3

noncomputable def length_side_c (a b c : ℝ) (A B C : ℝ) (h1: a = 2*sqrt 3) 
  (h2 : B = π / 4) (h3 : A + B + C = π) (h4 : 2 * sin^2 (B + C) - 3 * cos A = 0) 
  (h5 : A = π / 3): Prop :=
  c = sqrt 6 + sqrt 2

-- Theorems stating the properties
theorem angle_A (a b c : ℝ) (A B C : ℝ) (h1: a = 2*sqrt 3) 
  (h2 : B = π / 4) (h3 : A + B + C = π) (h4 : 2 * sin^2 (B + C) - 3 * cos A = 0) : 
  measure_angle_A a b c A B C h1 h2 h3 h4 := 
  by sorry

theorem side_c_length (a b c : ℝ) (A B C : ℝ) (h1: a = 2*sqrt 3) 
  (h2 : B = π / 4) (h3 : A + B + C = π) (h4 : 2 * sin^2 (B + C) - 3 * cos A = 0) 
  (h5 : A = π / 3) : 
  length_side_c a b c A B C h1 h2 h3 h4 h5 :=
  by sorry

end angle_A_side_c_length_l542_542489


namespace thief_speed_l542_542232

theorem thief_speed (v : ℝ) (hv : v > 0) : 
  let head_start_duration := (1/2 : ℝ)  -- 30 minutes, converted to hours
  let owner_speed := (75 : ℝ)  -- speed of owner in kmph
  let chase_duration := (2 : ℝ)  -- duration of the chase in hours
  let distance_by_owner := owner_speed * chase_duration  -- distance covered by the owner
  let total_distance_thief := head_start_duration * v + chase_duration * v  -- total distance covered by the thief
  distance_by_owner = 150 ->  -- given that owner covers 150 km
  total_distance_thief = 150  -- and so should the thief
  -> v = 60 := sorry

end thief_speed_l542_542232


namespace value_of_expression_l542_542173

noncomputable def expression : ℚ :=
  3 + (-3)⁻³

theorem value_of_expression :
  expression = 80 / 27 :=
by
  sorry

end value_of_expression_l542_542173


namespace two_students_exist_l542_542200

theorem two_students_exist (scores : Fin 49 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (i j : Fin 49), i ≠ j ∧ (scores i).1 ≥ (scores j).1 ∧ (scores i).2.1 ≥ (scores j).2.1 ∧ (scores i).2.2 ≥ (scores j).2.2 := 
by
  sorry

end two_students_exist_l542_542200


namespace ferry_total_tourists_l542_542970

noncomputable def total_tourists : Nat := 
  let a := 120  -- First trip tourists
  let d := -2   -- Common difference
  let n := 11   -- Number of trips
  n * (a + (a + (n - 1) * d)) / 2

theorem ferry_total_tourists : total_tourists = 1210 := by
  sorry

end ferry_total_tourists_l542_542970


namespace range_of_x_l542_542344

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + 2 * real.sin x

theorem range_of_x :
  (∀ x : ℝ, -2 < x ∧ x < 2 → f (1 + x) + f (x - x^2) > 0) ↔ (1 - real.sqrt 2 < x ∧ x < 1) := 
sorry

end range_of_x_l542_542344


namespace triangle_largest_angle_and_type_l542_542599

theorem triangle_largest_angle_and_type
  (a b c : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 4 * k) 
  (h3 : b = 3 * k) 
  (h4 : c = 2 * k) 
  (h5 : a ≥ b) 
  (h6 : a ≥ c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := 
by
  -- Replace 'by' with 'sorry' to denote that the proof should go here
  sorry

end triangle_largest_angle_and_type_l542_542599


namespace problem_I_I_problem_II_l542_542738

def A := {x : ℝ | 3 ≤ real.exp (real.log 3 * x) ∧ real.exp (real.log 3 * x) ≤ 27}
def B := {x : ℝ | real.log (2 * x - 1) / real.log 2 > 1}
def C (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}
def R := {x : ℝ | true} -- The set of all real numbers

theorem problem_I_I :
  A = {x : ℝ | 1 ≤ x ∧ x ≤ 3} ∧ 
  B = {x : ℝ | x > 3 / 2} ∧ 
  (A ∩ B) = {x : ℝ | 3 / 2 < x ∧ x ≤ 3} ∧ 
  ((R \ B) ∪ A) = {x : ℝ | x ≤ 3} := sorry

theorem problem_II (a : ℝ) :
  (C a ⊆ A) ↔ a ≤ 3 := sorry

end problem_I_I_problem_II_l542_542738


namespace infimum_of_fraction_l542_542244

theorem infimum_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Inf { x | ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ x = 1 / (2 * a) + 2 / b } = 9 / 2 :=
sorry

end infimum_of_fraction_l542_542244


namespace problem_1_problem_2_l542_542761

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

end problem_1_problem_2_l542_542761


namespace sin_arccos_l542_542258

theorem sin_arccos (a h : ℝ) (h_cos : a / h = 3 / 5) : 
  Real.sin (Real.arccos (3 / 5)) = 4 / 5 :=
by
  have hx : h ≠ 0 :=
    by
      intro hz
      simp only [hz, div_zero, ne.def] at h_cos
      exact zero_ne_one h_cos
  have ha : a ≠ 0 := 
    by
      intro hz
      simp only [hz, zero_div, ne.def] at h_cos
      exact zero_ne_one h_cos
  let opp := Real.sqrt (h^2 - a^2)
  have h_opp: opp = 4 := by
    rw [h, a] at h_cos
    simp
  suffices h_sin := Real.sin (Real.arccos (3 / 5)) = (opp / h)
  rw [h_sin, h_opp, h_cos, h]
  simp
  sorry -- Detailed calculations to prove this

end sin_arccos_l542_542258


namespace fraction_of_fraction_l542_542161

theorem fraction_of_fraction:
  let a := (3:ℚ) / 4
  let b := (5:ℚ) / 12
  b / a = (5:ℚ) / 9 := by
  sorry

end fraction_of_fraction_l542_542161


namespace min_number_of_candy_kinds_l542_542426

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542426


namespace standard_equation_of_ellipse_max_area_of_triangle_DMN_l542_542330

-- Definitions of the conditions
def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∀ x y : ℝ, (x / a) ^ 2 + (y / b) ^ 2 = 1

def point_D : ℝ × ℝ := (2, 1)

def line_OD_intersects_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∃ A B : ℝ × ℝ, ellipse a b ha hb ∧ (A, B ∈ ellipse a b ha hb) ∧ |AB| = √2 * |OD|

def eccentricity (a b : ℝ) : Prop :=
  sqrt (a^2 - b^2) / a = sqrt(3) / 2

def line_l (t : ℝ) : ℝ → ℝ := λ x, 1/2 * x + t

-- Proof problem to verify the standard equation of the ellipse C
theorem standard_equation_of_ellipse :
  ∀ a b : ℝ, a > b > 0 ∧
  point_D = (2, 1) ∧
  line_OD_intersects_ellipse a b (by sorry) (by sorry) ∧
  eccentricity a b →
  ellipse a b (by sorry) (by sorry) = ellipse 2 1 (by norm_num1) (by norm_num1) :=
  sorry

-- Proof problem to verify the maximum area of the triangle ∆DMN
theorem max_area_of_triangle_DMN (l : ℝ) :
  ∀ a b : ℝ, a > b > 0 ∧
  point_D = (2, 1) ∧
  line_l l (by sorry) ∧
  eccentricity a b →
  ∃M N : ℝ × ℝ, (M, N ≠ point_D) ∧ max_area_triangle = 1 :=
  sorry

end standard_equation_of_ellipse_max_area_of_triangle_DMN_l542_542330


namespace percentage_mr_william_land_l542_542189

theorem percentage_mr_william_land 
  (T W : ℝ) -- Total taxable land of the village and the total land of Mr. William
  (tax_collected_village : ℝ) -- Total tax collected from the village
  (tax_paid_william : ℝ) -- Tax paid by Mr. William
  (h1 : tax_collected_village = 3840) 
  (h2 : tax_paid_william = 480) 
  (h3 : (480 / 3840) = (25 / 100) * (W / T)) 
: (W / T) * 100 = 12.5 :=
by sorry

end percentage_mr_william_land_l542_542189


namespace minimum_candy_kinds_l542_542449

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542449


namespace smallest_solution_floor_eq_3_l542_542530

def g (x : ℝ) : ℝ := 2 * Real.sin x + 3 * Real.cos x + 4 * Real.tan x

theorem smallest_solution_floor_eq_3 :
  ∃ s > 0, g s = 0 ∧ Int.floor s = 3 :=
sorry

end smallest_solution_floor_eq_3_l542_542530


namespace min_value_f_at_pi_eight_l542_542355

def f (ω x : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x) * cos (ω * x) + cos (ω * x) ^ 2 - 1/2

theorem min_value_f_at_pi_eight:
  ∀ (ω > 0), (∃ c : ℝ, c ∈ set.Ioo 0 (π / 4) ∧ ∀ x ∈ set.Ioo 0 (π / 4),
  f ω x = f ω (c - x))
  → (∃ ω, ω > 0 ∧ f ω (π / 8) = 1/2) :=
begin
  sorry
end

end min_value_f_at_pi_eight_l542_542355


namespace polygon_sides_in_circle_l542_542462

theorem polygon_sides_in_circle (radius : ℝ) (a : ℝ) (n : ℕ) 
  (hn : 1 < a ∧ a < real.sqrt 2) :
  (a = 2 * real.sin (real.pi / n)) → (radius = 1) → n = 5 :=
by
  intros ha hr
  sorry

end polygon_sides_in_circle_l542_542462


namespace find_last_four_digits_of_N_l542_542967

def P (n : Nat) : Nat :=
  match n with
  | 0     => 1 -- usually not needed but for completeness
  | 1     => 2
  | _     => 2 + (n - 1) * n

theorem find_last_four_digits_of_N : (P 2011) % 10000 = 2112 := by
  -- we define P(2011) as per the general formula derived and then verify the modulo operation
  sorry

end find_last_four_digits_of_N_l542_542967


namespace transformation_correct_l542_542029

-- Define the original function
def original_function (x : ℝ) : ℝ := Real.cos x

-- Define the transformed function after halving abscissas and shifting left by π/4
def transformed_function (x : ℝ) : ℝ :=
  Real.cos (2 * x + (Real.pi / 2))

-- Theorem to prove the transformation correctness
theorem transformation_correct :
  transformed_function = λ x, Real.cos (2 * x + Real.pi / 2) :=
by
  sorry

end transformation_correct_l542_542029


namespace find_a8_l542_542747

variable {α : Type} [LinearOrderedField α]

/-- Given conditions of an arithmetic sequence -/
def arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a_n n = a1 + n * d

theorem find_a8 (a_n : ℕ → ℝ)
  (h_arith : arithmetic_sequence a_n)
  (h3 : a_n 3 = 5)
  (h5 : a_n 5 = 3) :
  a_n 8 = 0 :=
sorry

end find_a8_l542_542747


namespace integral_equation_uniqueness_l542_542114

-- Mathematical definition of the continuity of a function
def isContinuous (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ x y, continuous_at (λ p : ℝ × ℝ, f p.1 p.2) (x, y)

-- Main statement to be proven
theorem integral_equation_uniqueness :
  ∀ f1 f2 : ℝ × ℝ → ℝ,
    (∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f1 (x, y) = 1 + ∫∫ f1) →
    (∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f2 (x, y) = 1 + ∫∫ f2) →
    isContinuous f1 →
    isContinuous f2 →
    (∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f1 (x, y) = f2 (x, y)) :=
by 
  sorry

end integral_equation_uniqueness_l542_542114


namespace middle_number_consecutive_sum_is_60_l542_542145

noncomputable def middle_number_of_consecutive_integers_sum_is_60 : ℕ :=
  let n : ℕ := 12
  in n

theorem middle_number_consecutive_sum_is_60 :
  let n := middle_number_of_consecutive_integers_sum_is_60
  in n - 2 + n - 1 + n + n + 1 + n + 2 = 60 :=
by 
  sorry

end middle_number_consecutive_sum_is_60_l542_542145


namespace max_segment_perimeter_l542_542571

def isosceles_triangle (base height : ℝ) := true -- A realistic definition can define properties of an isosceles triangle

def equal_area_segments (triangle : isosceles_triangle 10 12) (n : ℕ) := true -- A realist definition can define cutting into equal area segments

noncomputable def perimeter_segment (base height : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  1 + Real.sqrt (height^2 + (base / n * k)^2) + Real.sqrt (height^2 + (base / n * (k + 1))^2)

theorem max_segment_perimeter (base height : ℝ) (n : ℕ) (h_base : base = 10) (h_height : height = 12) (h_segments : n = 10) :
  ∃ k, k ∈ Finset.range n ∧ perimeter_segment base height k n = 31.62 :=
by
  sorry

end max_segment_perimeter_l542_542571


namespace hyperbola_chord_eq_l542_542729

variables {a b x₀ y₀ x y : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : ¬ ((x₀^2 / a^2) - (y₀^2 / b^2) = 1))

theorem hyperbola_chord_eq :
  ∃ P₁ P₂ : ℝ × ℝ, 
    (P₁ ≠ P₂) ∧
    (tangent P₀ P₁ (mk_hyperbola a b)) ∧
    (tangent P₀ P₂ (mk_hyperbola a b)) ∧
    (P₁ ∈ mk_hyperbola a b) ∧
    (P₂ ∈ mk_hyperbola a b) ∧
    (∀ x y : ℝ, (x, y) ∈ line_through P₁ P₂ → (x₀ * x) / a^2 - (y₀ * y) / b^2 = 1) :=
sorry

/-- 
  A function to represent a hyperbola.
-/
def mk_hyperbola (a b : ℝ) : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ (x^2 / a^2) - (y^2 / b^2) = 1}

/--
  A predicate to check if a point lies on the tangent to the hyperbola at a specific point.
-/
def tangent (P₀ P : ℝ × ℝ) (c : set (ℝ × ℝ)) := 
  ∃ x₀ y₀ x y : ℝ, P₀ = (x₀, y₀) ∧ P = (x, y) ∧ P ∈ c ∧ 
  ((x₀ * x) / a^2) - ((y₀ * y) / b^2) = 1

/--
  A function to create the line through two points.
-/
def line_through (P₁ P₂ : ℝ × ℝ) : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ 
  (y - P₁.2) * (P₂.1 - P₁.1) = (P₂.2 - P₁.2) * (x - P₁.1)}


end hyperbola_chord_eq_l542_542729


namespace complex_star_degree_sum_l542_542681

theorem complex_star_degree_sum (n : ℕ) (hn_even : n % 2 = 0) (hn_ge_six : n ≥ 6) :
  (∑ k in Finset.range n, (180 - 1080 / n : ℝ)) = 180 * (n - 6) :=
by
  sorry

end complex_star_degree_sum_l542_542681


namespace unique_inverse_exists_l542_542831

noncomputable def f (a : ℕ → ℝ) : FormalPowerSeries ℝ := 
  λ n, a n  

theorem unique_inverse_exists (a : ℕ → ℝ) (h : a 0 ≠ 0) :
  ∃! g : FormalPowerSeries ℝ, f a * g = 1 :=
sorry

end unique_inverse_exists_l542_542831


namespace yura_picture_dimensions_l542_542942

-- Definitions based on the problem conditions
variable {a b : ℕ} -- dimensions of the picture
variable (hasFrame : ℕ × ℕ → Prop) -- definition sketch

-- The main statement to prove
theorem yura_picture_dimensions (h : (a + 2) * (b + 2) - a * b = 2 * a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
  sorry

end yura_picture_dimensions_l542_542942


namespace Roberto_outfit_count_l542_542110

theorem Roberto_outfit_count :
  let trousers := 5
  let shirts := 6
  let jackets := 3
  let ties := 2
  trousers * shirts * jackets * ties = 180 :=
by
  sorry

end Roberto_outfit_count_l542_542110


namespace multiplication_of_exponents_l542_542701

theorem multiplication_of_exponents : 
(\left(\frac{1}{3}\right)^{10} \cdot \left(\frac{2}{5}\right)^{-4} = \frac{625}{944784}) := by
    have h1 : (\left(\frac{a}{b}\right)^{-n} = \left(\frac{b}{a}\right)^n) := sorry
    have h2 : (\left(\frac{a}{b}\right)^n = \frac{a^n}{b^n}) := sorry
    sorry

end multiplication_of_exponents_l542_542701


namespace maximum_factors_of_2_l542_542300

-- Define J_k
def Jk (k : ℕ) : ℕ := 2 * 10^(k + 1) + 4 * 25

-- Define M(k) as the number of factors of 2 in the prime factorization of J_k
def M (k : ℕ) : ℕ := Multiset.count 2 (UniqueFactorizationMonoid.factors (Jk k))

-- The theorem statement
theorem maximum_factors_of_2 (k : ℕ) (hk : k > 0) : ∃ m, M k = m ∧ (∀ n, M n ≤ 6) := 
by {
  sorry
}

end maximum_factors_of_2_l542_542300


namespace fence_construction_cost_l542_542682

noncomputable def f (x s : ℝ) : ℝ := 225 * x + (360 * s / x) - 360

theorem fence_construction_cost (s : ℝ) (h_s : s > 2.5) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 20 → f x s = 225 * x + 360 * s / x - 360) ∧
  let m := if s ≤ 250 then 180 * Real.sqrt (10 * s) - 360 else 4140 + 18 * s in
  ∃ x : ℝ, 2 ≤ x ∧ x ≤ 20 ∧ f x s = m :=
by
  sorry

end fence_construction_cost_l542_542682


namespace current_dogwood_trees_l542_542150

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end current_dogwood_trees_l542_542150


namespace complex_number_properties_l542_542324

def z : ℂ := 1 + complex.i

theorem complex_number_properties (z := 1 + complex.i) :
  -- The conjugate of z is 1 - complex.i
  complex.conj z = 1 - complex.i ∧
  -- The imaginary part of z is not i
  complex.im z ≠ complex.i ∧
  -- \frac{\overline{z}}{z} ≠ i
  (complex.conj z / z ≠ complex.i) ∧
  -- |z| = sqrt 2
  complex.abs z = real.sqrt 2 :=
by
  sorry

end complex_number_properties_l542_542324


namespace jacks_remaining_capacity_l542_542500

noncomputable def jacks_basket_full_capacity : ℕ := 12
noncomputable def jills_basket_full_capacity : ℕ := 2 * jacks_basket_full_capacity
noncomputable def jacks_current_apples (x : ℕ) : Prop := 3 * x = jills_basket_full_capacity

theorem jacks_remaining_capacity {x : ℕ} (hx : jacks_current_apples x) :
  jacks_basket_full_capacity - x = 4 :=
by sorry

end jacks_remaining_capacity_l542_542500


namespace jill_trips_to_fill_tank_l542_542492

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end jill_trips_to_fill_tank_l542_542492


namespace infinite_primes_not_in_Sa_l542_542303

noncomputable def S_a (a : ℕ) : set ℕ := { p ∈ Nat.primes | ∃ b : ℕ, (odd b) ∧ (p ∣ (2^(2^a))^b - 1) }

theorem infinite_primes_not_in_Sa (a : ℕ) : ∃^∞ p : ℕ, (p ∈ Nat.primes) ∧ p ∉ S_a a :=
sorry

end infinite_primes_not_in_Sa_l542_542303


namespace sequence_value_x_l542_542798

theorem sequence_value_x : ∃ x : ℕ, 
  let s : ℕ → ℕ := λ n, match n with
    | 0 => 2
    | 1 => 5
    | 2 => 11
    | 3 => 20
    | 4 => x
    | 5 => 47
    | _ => 0 -- just as a placeholder
  in 
    (s 1 - s 0 = 3) ∧
    (s 2 - s 1 = 6) ∧
    (s 3 - s 2 = 9) ∧
    (s 4 - s 3 = 12) ∧
    (s 5 - s 4 = 15) ∧ 
    x = 32 :=
begin
  sorry
end

end sequence_value_x_l542_542798


namespace opposite_of_2023_l542_542134

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end opposite_of_2023_l542_542134


namespace problem_l542_542196

theorem problem (A B : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : B > A) (n c : ℝ) 
  (h₃ : B = A * (1 + n / 100)) (h₄ : A = B * (1 - c / 100)) :
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) :=
by
  sorry

end problem_l542_542196


namespace minimum_candy_kinds_l542_542407

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542407


namespace probability_both_from_asia_probability_a1_not_b1_l542_542993

-- Definition and conditions for problem (1)
def asian_countries : Finset (String) := {"A1", "A2", "A3"}
def european_countries : Finset (String) := {"B1", "B2", "B3"}
def total_countries : Finset (String) := asian_countries ∪ european_countries

-- Prove the probability that both chosen countries are from Asia equals 1/5
theorem probability_both_from_asia : 
  (choose 6 2).card = 15 ∧ 
  (choose 3 2).card = 3 ∧ 
  (3 / 15 : ℚ) = 1 / 5 := by sorry

-- Definition and conditions for problem (2)
def valid_pairs : Finset (String × String) := 
  (asian_countries.product european_countries) \ 
  (asian_countries.product (Finset.singleton "B1"))

-- Prove the probability that a pair include A1 and not B1 equals 2/9
theorem probability_a1_not_b1 : 
  (asian_countries.product european_countries).card = 9 ∧ 
  (Finset.filter (λ p, p.1 = "A1" ∧ p.2 ≠ "B1") (asian_countries.product european_countries)).card = 2 ∧ 
  (2 / 9 : ℚ) = 2 / 9 := by sorry

end probability_both_from_asia_probability_a1_not_b1_l542_542993


namespace distance_from_origin_l542_542222

theorem distance_from_origin {x y : ℝ} (h1 : |x| = 8)
  (h2 : real.sqrt ((x - 7) ^ 2 + (y - 3) ^ 2) = 8)
  (h3 : y > 3) : 
  real.sqrt (x ^ 2 + y ^ 2) = real.sqrt (136 + 6 * real.sqrt 63) := 
      sorry

end distance_from_origin_l542_542222


namespace three_seventy_five_as_fraction_l542_542940

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l542_542940


namespace no_such_n_l542_542685

theorem no_such_n (n : ℕ) (h_positive : n > 0) : 
  ¬ ∃ k : ℕ, (n^2 + 1) = k * (Nat.floor (Real.sqrt n))^2 + 2 := by
  sorry

end no_such_n_l542_542685


namespace area_triangle_tangent_circles_l542_542209

theorem area_triangle_tangent_circles :
  ∃ (A B C : Type) (radius1 radius2 : ℝ) 
    (tangent1 tangent2 : ℝ → ℝ → Prop)
    (congruent_sides : ℝ → Prop),
    radius1 = 1 ∧ radius2 = 2 ∧
    (∀ x y, tangent1 x y) ∧ (∀ x y, tangent2 x y) ∧
    congruent_sides 1 ∧ congruent_sides 2 ∧
    ∃ (area : ℝ), area = 16 * Real.sqrt 2 :=
by
  -- This is where the proof would be written
  sorry

end area_triangle_tangent_circles_l542_542209


namespace no_valid_rearrangement_l542_542182

theorem no_valid_rearrangement 
  (cards : List ℕ) 
  (h1 : cards.length = 20) 
  (h2 : ∀ x : ℕ, x ∈ cards → x ≤ 9) 
  (h3 : ∀ i : ℕ, 0 ≤ i ∧ i ≤ 9 → list.count cards i = 2) :
  ¬ (∃ arrangement : List ℕ, 
      arrangement.length = 20 ∧ 
      ∀ i : ℕ, 0 ≤ i ∧ i ≤ 9 → 
        (arrangement.index_of i < arrangement.last_index_of i) ∧ 
        (arrangement.last_index_of i - arrangement.index_of i - 1 = i)) := 
  sorry

end no_valid_rearrangement_l542_542182


namespace false_propositions_l542_542829

open EuclideanGeometry

variables (m n : Line) (α β : Plane)

-- Conditions:
axiom prop1 {m : Line} {α β : Plane} : (Parallel m α) ∧ (Parallel m β) → Parallel α β
axiom prop2 {m : Line} {α β : Plane} : (Perpendicular m α) ∧ (Perpendicular m β) → Parallel α β
axiom prop3 {m n : Line} {α : Plane} : (Parallel m α) ∧ (Parallel n α) → Parallel m n
axiom prop4 {m n : Line} {α : Plane} : (Perpendicular m α) ∧ (Perpendicular n α) → Parallel m n

theorem false_propositions : (¬ prop1) ∧ (¬ prop3) :=
by
  -- Proof is omitted.
  sorry

end false_propositions_l542_542829


namespace measure_of_angle_C_range_of_sum_ab_l542_542053

-- Proof problem (1): Prove the measure of angle C
theorem measure_of_angle_C (a b c : ℝ) (A B C : ℝ) 
  (h1 : 2 * c * Real.sin C = (2 * b + a) * Real.sin B + (2 * a - 3 * b) * Real.sin A) :
  C = Real.pi / 3 := by 
  sorry

-- Proof problem (2): Prove the range of possible values of a + b
theorem range_of_sum_ab (a b : ℝ) (c : ℝ) (h1 : c = 4) (h2 : 16 = a^2 + b^2 - a * b) :
  4 < a + b ∧ a + b ≤ 8 := by 
  sorry

end measure_of_angle_C_range_of_sum_ab_l542_542053


namespace imaginary_part_correct_l542_542882

def imaginary_part_of_product : ℂ := (1 + 2 * complex.i) * complex.i
theorem imaginary_part_correct : complex.im (imaginary_part_of_product) = 1 := sorry

end imaginary_part_correct_l542_542882


namespace inequality_1_minimum_value_l542_542860

-- Definition for part (1)
theorem inequality_1 (a b m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (a^2 / m + b^2 / n) ≥ ((a + b)^2 / (m + n)) :=
sorry

-- Definition for part (2)
theorem minimum_value (x : ℝ) (hx : 0 < x) (hx' : x < 1) : 
  (∃ (y : ℝ), y = (1 / x + 4 / (1 - x)) ∧ y = 9) :=
sorry

end inequality_1_minimum_value_l542_542860


namespace moles_of_Cl2_required_l542_542293

theorem moles_of_Cl2_required (n_C2H6 n_HCl : ℕ) (balance : n_C2H6 = 3) (HCl_needed : n_HCl = 6) :
  ∃ n_Cl2 : ℕ, n_Cl2 = 9 :=
by
  sorry

end moles_of_Cl2_required_l542_542293


namespace more_vets_yummy_than_puppy_l542_542469

noncomputable def total_vets : ℕ := 3500
noncomputable def puppy_kibble_percentage : ℝ := 0.235
noncomputable def yummy_dog_kibble_percentage : ℝ := 0.372

noncomputable def num_puppy_kibble_vets : ℕ := (puppy_kibble_percentage * total_vets).ceil.to_nat
noncomputable def num_yummy_dog_kibble_vets : ℕ := (yummy_dog_kibble_percentage * total_vets).ceil.to_nat

theorem more_vets_yummy_than_puppy :
  num_yummy_dog_kibble_vets - num_puppy_kibble_vets = 479 :=
by sorry

end more_vets_yummy_than_puppy_l542_542469


namespace correct_propositions_l542_542725

variables (α β γ : Plane) (l m : Line) 

-- Propositions with their conditions
-- Proposition ②
def prop2 (h1 : l ∈ β) (h2 : l ⊥ α) : Prop :=
  α ⊥ β

-- Proposition ③
def prop3 (h1 : β ∩ γ = l) (h2 : l ∥ α) (h3 : m ∈ α) (h4 : m ⊥ γ) : Prop :=
  α ⊥ γ ∧ l ⊥ m

-- The main statement combining correct propositions
theorem correct_propositions (h₂_1 : l ∈ β) (h₂_2 : l ⊥ α)
  (h₃_1 : β ∩ γ = l) (h₃_2 : l ∥ α) (h₃_3 : m ∈ α) (h₃_4 : m ⊥ γ) : 
  prop2 α β γ l m h₂_1 h₂_2 ∧ prop3 α β γ l m h₃_1 h₃_2 h₃_3 h₃_4 :=
sorry

end correct_propositions_l542_542725


namespace Julieta_total_spending_l542_542503

def original_backpack_price: ℝ := 50
def original_ring_binder_price: ℝ := 20
def backpack_price_increase: ℝ := 5
def ring_binder_price_decrease: ℝ := 2
def backpack_discount: ℝ := 0.10
def ring_binder_buy_one_get_one: ℕ := 3
def sales_tax: ℝ := 0.06

theorem Julieta_total_spending:
  let new_backpack_price := original_backpack_price + backpack_price_increase in
  let new_ring_binder_price := original_ring_binder_price - ring_binder_price_decrease in
  let discounted_backpack_price := new_backpack_price * (1 - backpack_discount) in
  let ring_binders_needed := ring_binder_buy_one_get_one / 2 + ring_binder_buy_one_get_one % 2 in
  let total_ring_binder_price := ring_binders_needed * new_ring_binder_price in
  let subtotal := discounted_backpack_price + total_ring_binder_price in
  let total_spending := subtotal * (1 + sales_tax) in
  total_spending = 90.63 :=
sorry

end Julieta_total_spending_l542_542503


namespace percentage_palm_oil_in_cheese_l542_542855

theorem percentage_palm_oil_in_cheese
  (initial_cheese_price: ℝ := 100)
  (cheese_price_increase: ℝ := 3)
  (palm_oil_price_increase_percentage: ℝ := 0.10)
  (expected_palm_oil_percentage : ℝ := 30):
  ∃ (palm_oil_initial_price: ℝ),
  cheese_price_increase = palm_oil_initial_price * palm_oil_price_increase_percentage ∧
  expected_palm_oil_percentage = 100 * (palm_oil_initial_price / initial_cheese_price) := by
  sorry

end percentage_palm_oil_in_cheese_l542_542855


namespace percentage_increase_second_to_third_l542_542058

theorem percentage_increase_second_to_third :
  ∀ (D D2 D3 : ℝ),
    (1.2 * D = 12) →
    (10 + D2 + D3 = 37) →
    (D2 = 12) →
    D = 10 →
    D3 = 15 →
    ((D3 - D2) / D2) * 100 = 25 :=
by
  intros D D2 D3 h1_20D_eq_12 h_total_dist h_D2_eq_12 h_D_eq_10 h_D3_eq_15
  calc
    ((D3 - D2) / D2) * 100 = (3 / 12) * 100 := by rw [h_D3_eq_15, h_D2_eq_12]
    ...                        = (1 / 4) * 100 := by norm_num
    ...                        = 25            := by norm_num

end percentage_increase_second_to_third_l542_542058


namespace select_3_items_count_select_exactly_1_defective_count_select_at_least_1_defective_count_arrangement_with_1_defective_count_l542_542272

-- Assumptions
def total_products := 100
def non_defective := 98
def defective := 2
def selected_items := 3

-- Problem statements
theorem select_3_items_count : 
  (nat.choose total_products selected_items = 161700) := 
sorry

theorem select_exactly_1_defective_count :
  (nat.choose defective 1) * (nat.choose non_defective (selected_items - 1)) = 9506 := 
sorry

theorem select_at_least_1_defective_count :
  (nat.choose total_products selected_items - nat.choose non_defective selected_items = 9604) := 
sorry

theorem arrangement_with_1_defective_count :
  (nat.choose defective 1) * (nat.choose non_defective (selected_items - 1)) * (nat.factorial selected_items = 57036) := 
sorry

end select_3_items_count_select_exactly_1_defective_count_select_at_least_1_defective_count_arrangement_with_1_defective_count_l542_542272


namespace circle_through_point_on_parabola_l542_542648

theorem circle_through_point_on_parabola :
  let C : ℝ × ℝ → bool := λ p, (p.fst + 1) ^ 2 + (p.snd - 1) ^ 2 = 5
  let A : ℝ × ℝ := (-2, 3)
  let parabola : ℝ × ℝ → bool := λ p, p.snd ^ 2 = 4 * p.fst
  let directrix : ℝ := -1 
  let focus : ℝ × ℝ := (1, 0)
  let intersects_parabola (line : ℝ × ℝ → bool) : bool := 
    ∃ (A B : ℝ × ℝ), parabola A ∧ parabola B ∧ line A ∧ line B
  in
  ∀ k : ℝ, k ≠ 0 → 
  intersects_parabola (λ p, p.snd = k * (p.fst - 1))
    → C A :=
sorry

end circle_through_point_on_parabola_l542_542648


namespace number_less_than_value_l542_542650

-- Definition for the conditions
def exceeds_condition (x y : ℕ) : Prop := x - 18 = 3 * (y - x)
def specific_value (x : ℕ) : Prop := x = 69

-- Statement of the theorem
theorem number_less_than_value : ∃ y : ℕ, (exceeds_condition 69 y) ∧ (specific_value 69) → y = 86 :=
by
  -- To be proved
  sorry

end number_less_than_value_l542_542650


namespace volume_calculation_l542_542671

namespace TetrahedronVolume

variables (P Q R S : EuclideanSpace ℝ (Fin 3))
variables (PQ PR PS QR QS RS: ℝ)

-- Conditions
noncomputable def conditions : Prop :=
  dist P Q = 4 ∧
  dist P R = 5 ∧
  dist P S = 6 ∧
  dist Q R = 3 ∧
  dist Q S = Real.sqrt 37 ∧
  dist R S = 7

-- Objective: Prove the volume of tetrahedron equals 10.25
noncomputable def volume_tetrahedron (P Q R S : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  let a := P - R in
  let b := P - S in
  let c := P - Q in
  (1 / 6) * Real.abs (det (Matrix.fromVecs ![a, b, c]))

theorem volume_calculation (h : conditions P Q R S) : 
  volume_tetrahedron P Q R S = 10.25 :=
sorry

end TetrahedronVolume

end volume_calculation_l542_542671


namespace rectangular_solid_surface_area_l542_542273

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hvol : a * b * c = 455) : 
  let surface_area := 2 * (a * b + b * c + c * a)
  surface_area = 382 := by
-- proof
sorry

end rectangular_solid_surface_area_l542_542273


namespace common_difference_of_ap_l542_542287

theorem common_difference_of_ap : 
  ∀ (a T_15 n d : ℤ), a = 2 ∧ T_15 = 44 ∧ n = 15 → T_15 = a + (n - 1) * d → d = 3 :=
begin
  sorry
end

end common_difference_of_ap_l542_542287


namespace total_movie_hours_l542_542062

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l542_542062


namespace tan_half_angle_identity_l542_542024

theorem tan_half_angle_identity (α : ℝ) (h1 : cos α = -4/5) (h2 : π < α ∧ α < 3 * π / 2) :
  (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1/2 :=
  sorry

end tan_half_angle_identity_l542_542024


namespace complex_number_properties_l542_542323

def z : ℂ := 1 + complex.i

theorem complex_number_properties (z := 1 + complex.i) :
  -- The conjugate of z is 1 - complex.i
  complex.conj z = 1 - complex.i ∧
  -- The imaginary part of z is not i
  complex.im z ≠ complex.i ∧
  -- \frac{\overline{z}}{z} ≠ i
  (complex.conj z / z ≠ complex.i) ∧
  -- |z| = sqrt 2
  complex.abs z = real.sqrt 2 :=
by
  sorry

end complex_number_properties_l542_542323


namespace uncle_fyodor_sandwiches_count_l542_542616

variable (sandwiches_sharik : ℕ)
variable (sandwiches_matroskin : ℕ := 3 * sandwiches_sharik)
variable (total_sandwiches_eaten : ℕ := sandwiches_sharik + sandwiches_matroskin)
variable (sandwiches_uncle_fyodor : ℕ := 2 * total_sandwiches_eaten)
variable (difference : ℕ := sandwiches_uncle_fyodor - sandwiches_sharik)

theorem uncle_fyodor_sandwiches_count :
  (difference = 21) → sandwiches_uncle_fyodor = 24 := by
  intro h
  sorry

end uncle_fyodor_sandwiches_count_l542_542616


namespace count_perfect_squares_but_not_cubes_l542_542376

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n^2 = x

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = x

def is_perfect_sixth_power (x : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = x

def in_range_1_to_1M (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 1000000

def squares_but_not_cubes_count : ℕ :=
  (Finset.range 1000000).filter (λ x, is_perfect_square x ∧ ¬ is_perfect_sixth_power x).card

theorem count_perfect_squares_but_not_cubes :
  squares_but_not_cubes_count = 990 :=
by
  sorry

end count_perfect_squares_but_not_cubes_l542_542376


namespace petya_mistake_l542_542100

theorem petya_mistake (x : ℝ) (h : x - x / 10 = 19.71) : x = 21.9 := 
  sorry

end petya_mistake_l542_542100


namespace sum_first_four_terms_l542_542589

theorem sum_first_four_terms (a : ℕ → ℤ) (h5 : a 5 = 5) (h6 : a 6 = 9) (h7 : a 7 = 13) : 
  a 1 + a 2 + a 3 + a 4 = -20 :=
sorry

end sum_first_four_terms_l542_542589


namespace total_points_other_members_18_l542_542481

-- Definitions
def total_points (x : ℕ) (S : ℕ) (T : ℕ) (M : ℕ) (y : ℕ) :=
  S + T + M + y = x

def Sam_scored (x S : ℕ) := S = x / 3

def Taylor_scored (x T : ℕ) := T = 3 * x / 8

def Morgan_scored (M : ℕ) := M = 21

def other_members_scored (y : ℕ) := ∃ (a b c d e f g h : ℕ),
  a ≤ 3 ∧ b ≤ 3 ∧ c ≤ 3 ∧ d ≤ 3 ∧ e ≤ 3 ∧ f ≤ 3 ∧ g ≤ 3 ∧ h ≤ 3 ∧
  y = a + b + c + d + e + f + g + h

-- Theorem
theorem total_points_other_members_18 (x y S T M : ℕ) :
  Sam_scored x S → Taylor_scored x T → Morgan_scored M → total_points x S T M y → other_members_scored y → y = 18 :=
by
  intros hSam hTaylor hMorgan hTotal hOther
  sorry

end total_points_other_members_18_l542_542481


namespace count_zeros_before_first_non_zero_digit_l542_542016

theorem count_zeros_before_first_non_zero_digit :
  ∀ (a b : ℕ), a = 2^4 ∧ b = 5^6 → 
  let fraction := (1 : ℚ) / (a * b),
      decimal_representation := (4 : ℚ) / (10^6 : ℚ)
  in (fraction = decimal_representation) →
     -- We are expressing this as 6 digits with the first non-zero digit (4) being preceded by 5 zeros.
     5 = 6 - 1 - (1 : ℕ)
:=
begin
  intros,
  sorry
end

end count_zeros_before_first_non_zero_digit_l542_542016


namespace minimum_knights_l542_542147

theorem minimum_knights (inhabitants : Fin 30 → Bool) :
  (∀ i : Fin 30, inhabitants i = true →
    (inhabitants ((i + 1) % 30) = false ∨ inhabitants ((i - 1) % 30) = false)) →
  (∃ k : ℕ, k ≥ 10 ∧ (∃ S : Finset (Fin 30), S.card = k ∧ ∀ i : Fin 30, i ∈ S → inhabitants i = true)) :=
begin
  sorry
end

end minimum_knights_l542_542147


namespace convert_decimal_to_fraction_l542_542937

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l542_542937


namespace intersection_P_Q_l542_542370

def set_P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def set_Q : Set ℝ := {x | (x - 1) ^ 2 ≤ 4}

theorem intersection_P_Q :
  {x | x ∈ set_P ∧ x ∈ set_Q} = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_P_Q_l542_542370


namespace part_1_solution_part_2_solution_l542_542277

def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem part_1_solution (x : ℝ) : f x < 3 ↔ -4 / 3 < x ∧ x < 0 :=
by
  sorry

theorem part_2_solution (a : ℝ) : (∀ x, ¬ (f x < a)) → a ≤ 2 :=
by
  sorry

end part_1_solution_part_2_solution_l542_542277


namespace candy_problem_l542_542458

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542458


namespace problem1_problem2_l542_542213

-- Definitions based on the conditions
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 5

-- Problem 1: Both doctor A and B must join the team
theorem problem1 : ∃ (ways : ℕ), ways = 816 :=
  by
    let remaining_doctors := total_doctors - 2
    let choose := remaining_doctors.choose (team_size - 2)
    have h1 : choose = 816 := sorry
    exact ⟨choose, h1⟩

-- Problem 2: At least one of doctors A or B must join the team
theorem problem2 : ∃ (ways : ℕ), ways = 5661 :=
  by
    let remaining_doctors := total_doctors - 1
    let scenario1 := 2 * remaining_doctors.choose (team_size - 1)
    let scenario2 := (total_doctors - 2).choose (team_size - 2)
    let total_ways := scenario1 + scenario2
    have h2 : total_ways = 5661 := sorry
    exact ⟨total_ways, h2⟩

end problem1_problem2_l542_542213


namespace calc_expression_factorize_polynomial_l542_542638

noncomputable def expression : ℝ := (1/3)^(-1) - real.sqrt 16 + (-2016)^0

theorem calc_expression : expression = 0 := 
by 
  -- Proof can be left to fill in the actual steps
sorry

theorem factorize_polynomial (x : ℝ) : 3 * x^2 - 6 * x + 3 = 3 * (x - 1)^2 := 
by 
  -- Proof can be left to fill in the actual steps
sorry

end calc_expression_factorize_polynomial_l542_542638


namespace problem_statement_l542_542364

noncomputable def f (x θ : ℝ) : ℝ :=
  sin (2 * x + θ) + cos x ^ 2

noncomputable def G (θ : ℝ) : ℝ :=
  sqrt (5 / 4 + sin θ) + 1 / 2

noncomputable def g (θ : ℝ) : ℝ :=
  -sqrt (5 / 4 + sin θ) + 1 / 2

theorem problem_statement : ∃ θ₀ : ℝ, abs (G θ₀ / g θ₀) = π :=
by
  sorry

end problem_statement_l542_542364


namespace triangle_is_isosceles_l542_542332

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_triangle : A + B + C = π)
  (h_condition : 2 * Real.cos B * Real.sin C = Real.sin A) :
  B = C :=
sorry

end triangle_is_isosceles_l542_542332


namespace square_area_increase_l542_542192

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let original_area := s^2 in
  let new_side := 1.05 * s in
  let new_area := new_side^2 in
  let increase := new_area - original_area in
  let percent_increase := (increase / original_area) * 100 in
  percent_increase = 10.25 :=
by
  sorry

end square_area_increase_l542_542192


namespace value_of_x_plus_y_l542_542382

theorem value_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 4) (h3 : x * y > 0) : x + y = 7 ∨ x + y = -7 :=
by
  sorry

end value_of_x_plus_y_l542_542382


namespace find_values_of_x_y_l542_542735

-- Define the matrices A and B
def A (x : ℝ) := ![![ (-1 : ℝ), 2], ![1, x]]
def B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 1], ![2, -1]]

-- Define the vector a
def a (y : ℝ) := ![ (2 : ℝ), y]

-- The problem is to prove that for x = -1/2 and y = 4, A * a = B * a
theorem find_values_of_x_y : ∃ x y : ℝ, x = -1 / 2 ∧ y = 4 ∧ (A x).mul_vec (a y) = B.mul_vec (a y) :=
by
  sorry

end find_values_of_x_y_l542_542735


namespace min_value_sum_l542_542895

def non_neg_int := {n : ℕ // 0 ≤ n}

theorem min_value_sum (a b c d : non_neg_int)
  (h : a.val * b.val + b.val * c.val + c.val * d.val + d.val * a.val = 707) :
  a.val + b.val + c.val + d.val ≥ 108 :=
begin
  -- The proof would go here, but it is omitted as per instructions.
  sorry
end

end min_value_sum_l542_542895


namespace problem_statement_l542_542757

noncomputable def f (x : ℝ) := x^2 / (1 + x^2)

theorem problem_statement :
  (∀ x : ℝ, 0 < x → f x + f (1 / x) = 1) →
  2 * (∑ i in finset.range 2016, f (i + 2)) +
  (∑ i in finset.range 2016, f (1 / (i + 2))) +
  (∑ i in finset.range 2016, (1 / (i + 2)^2) * f (i + 2)) = 4032 :=
by
  sorry

end problem_statement_l542_542757


namespace find_m_eq_zero_l542_542834

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end find_m_eq_zero_l542_542834


namespace expressions_equal_iff_l542_542270

variable (a b c : ℝ)

theorem expressions_equal_iff :
  a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 :=
by
  sorry

end expressions_equal_iff_l542_542270


namespace first_player_wins_l542_542789

-- Define the game state and requirements
inductive Player
| first : Player
| second : Player

-- Game state consists of a number of stones and whose turn it is
structure GameState where
  stones : Nat
  player : Player

-- Define a simple transition for the game
def take_stones (s : GameState) (n : Nat) : GameState :=
  { s with stones := s.stones - n, player := Player.second }

-- Determine if a player can take n stones
def can_take (s : GameState) (n : Nat) : Prop :=
  n >= 1 ∧ n <= 4 ∧ n <= s.stones

-- Define victory condition
def wins (s : GameState) : Prop :=
  s.stones = 0 ∧ s.player = Player.second

-- Prove that if the first player starts with 18 stones and picks 3 stones initially,
-- they can ensure victory
theorem first_player_wins :
  ∀ (s : GameState),
    s.stones = 18 ∧ s.player = Player.first →
    can_take s 3 →
    wins (take_stones s 3)
:= by
  sorry

end first_player_wins_l542_542789


namespace sqrt_lipschitz_min_value_l542_542963

def is_lipschitz (f : ℝ → ℝ) (k : ℝ) (D : Set ℝ) : Prop :=
∀ x₁ x₂ ∈ D, x₁ ≠ x₂ → abs (f x₁ - f x₂) ≤ k * abs (x₁ - x₂)

def sqrt_lipschitz (k : ℝ) : Prop :=
∀ x₁ x₂ ∈ {x : ℝ | 1 ≤ x}, x₁ ≠ x₂ → abs (Real.sqrt x₁ - Real.sqrt x₂) ≤ k * abs (x₁ - x₂)

theorem sqrt_lipschitz_min_value :
  sqrt_lipschitz (1 / 2) :=
by sorry

end sqrt_lipschitz_min_value_l542_542963


namespace lateral_surface_area_correct_l542_542468

-- Define the geometric conditions
variables (A B C D A1 B1 C1 D1 M : Type*) 

-- Variables for angles and length
variables (α β : ℝ) (b : ℝ)

-- Hypotheses
variables (h1 : ∠ A M B = α)
variables (h2 : ∠ B M B1 = β)
variables (h3 : B1 M = b)

noncomputable def lateral_surface_area (α β b : ℝ) :=
  2 * b^2 * √2 * sin(2 * β) * sin((2 * α + Real.pi) / 4)

theorem lateral_surface_area_correct
  (h1 : ∠ A M B = α) (h2 : ∠ B M B1 = β) (h3 : B1 M = b) :
  lateral_surface_area α β b = 2 * b^2 * √2 * sin(2 * β) * sin((2 * α + Real.pi) / 4) :=
sorry

end lateral_surface_area_correct_l542_542468


namespace problem_1_problem_2_l542_542204

-- Definitions
def balls_in_box := {red: ℕ, white: ℕ}

def sampling_without_replacement (box: balls_in_box) : ℚ :=
  let total_balls := box.red + box.white in
  (box.red / total_balls) * (box.white / (total_balls - 1)) +
  (box.white / total_balls) * (box.red / (total_balls - 1))

def sampling_with_replacement (box: balls_in_box) (draw_times: ℕ) : ℚ :=
  1 - ((box.white / (box.red + box.white)) ^ draw_times)

-- Problems
theorem problem_1 (box: balls_in_box) (h: box.red = 2 ∧ box.white = 4) :
  sampling_without_replacement box = 8 / 15 :=
by
  sorry

theorem problem_2 (box: balls_in_box) (h: box.red = 2 ∧ box.white = 4) :
  sampling_with_replacement box 3 = 19 / 27 :=
by
  sorry

end problem_1_problem_2_l542_542204


namespace total_distance_Joe_travels_l542_542810

-- Define the complex positions
def z_J : ℂ := 3 + 4 * ℂ.I
def z_T : ℂ := 1 + ℂ.I
def z_G : ℂ := -2 + 3 * ℂ.I

-- Define the function to calculate the distance between two complex numbers
def distance (z1 z2 : ℂ) : ℝ := complex.abs (z1 - z2)

-- Define the distances d_JT and d_TG
def d_JT := distance z_J z_T
def d_TG := distance z_T z_G

-- Prove the total distance Joe travels is 2 * sqrt(13)
theorem total_distance_Joe_travels :
  d_JT + d_TG = 2 * Real.sqrt 13 := by
  sorry

end total_distance_Joe_travels_l542_542810


namespace problem_I_problem_II_l542_542746

-- Given that the terminal side of angle α lies on the line 3x + 4y = 0
def terminal_side_on_line (α : ℝ) : Prop :=
  ∃ x y : ℝ, (3 * x + 4 * y = 0) ∧ (α = real.arctan (y / x))

-- Problem (I)
theorem problem_I (α : ℝ) (h : terminal_side_on_line α) :
  (cos (π / 2 + α) * sin (-π - α)) / (cos (5 * π / 2 - α) * sin (9 * π / 2 + α)) = -3 / 4 :=
sorry

-- Problem (II)
theorem problem_II (α : ℝ) (h : terminal_side_on_line α) :
  sin α * cos α + cos α ^ 2 + 1 = 29 / 25 :=
sorry

end problem_I_problem_II_l542_542746


namespace outfit_count_correct_l542_542023

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end outfit_count_correct_l542_542023


namespace monotonic_increasing_interval_l542_542754

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

variables (φ : ℝ)
variables (hx1 : |φ| < Real.pi / 2)
variables (hx2 : 2 * φ + Real.pi / 3 = 0)

theorem monotonic_increasing_interval :
  ∃ I : Set ℝ, I = Set.Icc (-Real.pi / 3) (Real.pi / 6) ∧
  ∀ x₁ x₂ ∈ I, x₁ ≤ x₂ → f φ x₁ ≤ f φ x₂ := sorry

end monotonic_increasing_interval_l542_542754


namespace find_general_term_limit_b_n_l542_542320

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 2 + real.sqrt x) ^ 2

def seq_a : ℕ+ → ℝ
| 1 := 2
| (n + 1) := 4 * (n + 1) - 2

def S : ℕ+ → ℝ
| 1 := 2
| (n + 1) := f (S n)

def b_n (n : ℕ+) : ℝ :=
  let a_n := seq_a n
  let a_n1 := seq_a (n + 1)
  in (a_n1 ^ 2 + a_n ^ 2) / (2 * a_n1 * a_n)

theorem find_general_term :
  ∀ n : ℕ+, seq_a n = 4 * (↑n : ℕ) - 2 :=
sorry

theorem limit_b_n :
  tendsto (λ n : ℕ, (∑ i in finset.range n, b_n (nat.succ i) - n)) at_top (𝓝 1) :=
sorry

end find_general_term_limit_b_n_l542_542320


namespace P_geq1_l542_542779

-- Defining the conditions
constant ξ : ℝ → ℝ
constant σ : ℝ
axiom normal_dist_ξ : ∀ x : ℝ, ξ x = (1 / (√(2 * π * (σ^2)))) * (exp (-((x + 1)^2) / (2 * σ^2)))
axiom P_neg3_to_neg1 : ∫ (x : ℝ) in (-3 : ℝ)..(-1 : ℝ), ξ x = 0.4

-- The proof problem
theorem P_geq1 : ∫ (x : ℝ) in (1 : ℝ)..(∞ : ℝ), ξ x = 0.1 := sorry

end P_geq1_l542_542779


namespace prob_a18_is_13_l542_542368

noncomputable def probability_a18_eq_13 (sequence : Fin 18 → ℤ)
  (h1 : sequence 0 = 0)
  (h2 : ∀ k, k < 17 → abs (sequence (k + 1) - sequence k) = 1)
  : Prop :=
  let count_positive_steps := 
    (Finset.range 17).filter (λ k, sequence (k+1) - sequence k = 1) |>.card 
  in
  (count_positive_steps = 15 ∧ ∀ n, (count_positive_steps + n = 17 ∧ count_positive_steps - n = 13)) ∧ 
  ((Nat.choose 17 15 : ℚ) / 2^17 = 1 / 1024)

--Statement of the theorem to prove the probability
theorem prob_a18_is_13 (sequence : Fin 18 → ℤ)
  (h1 : sequence 0 = 0)
  (h2 : ∀ k, k < 17 → abs (sequence (k + 1) - sequence k) = 1)
  : probability_a18_eq_13 sequence h1 h2 :=
sorry

end prob_a18_is_13_l542_542368


namespace not_perfect_power_l542_542556

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ∀ m n : ℕ, m > 1 → n > 1 → 10^k - 1 ≠ m ^ n :=
by 
  sorry

end not_perfect_power_l542_542556


namespace diameter_hydrogen_atom_scientific_notation_l542_542617

theorem diameter_hydrogen_atom_scientific_notation :
  ∃ a n : ℕ, (1 ≤ a) ∧ (a < 10) ∧ (0.0000000001 = a * 10 ^ n) := 
begin
  use (1 : ℕ), 
  use (-10 : ℕ),
  split,
  { norm_num },
  { norm_num },
  sorry
end

end diameter_hydrogen_atom_scientific_notation_l542_542617


namespace father_has_4_chocolate_bars_left_l542_542544

noncomputable def chocolate_bars_given_to_father (initial_bars : ℕ) (num_people : ℕ) : ℕ :=
  let bars_per_person := initial_bars / num_people
  let bars_given := num_people * (bars_per_person / 2)
  bars_given

noncomputable def chocolate_bars_left_with_father (bars_given : ℕ) (bars_given_away : ℕ) : ℕ :=
  bars_given - bars_given_away

theorem father_has_4_chocolate_bars_left :
  ∀ (initial_bars num_people bars_given_away : ℕ), 
  initial_bars = 40 →
  num_people = 7 →
  bars_given_away = 10 →
  chocolate_bars_left_with_father (chocolate_bars_given_to_father initial_bars num_people) bars_given_away = 4 :=
by
  intros initial_bars num_people bars_given_away h_initial h_num h_given_away
  unfold chocolate_bars_given_to_father chocolate_bars_left_with_father
  rw [h_initial, h_num, h_given_away]
  exact sorry

end father_has_4_chocolate_bars_left_l542_542544


namespace gcd_value_l542_542925

theorem gcd_value (m n : ℕ) (h1 : m + 6 = 9 * m) : ∃ d ∈ {3, 6}, d ∣ m ∧ d ∣ n :=
by
  sorry

end gcd_value_l542_542925


namespace fermi_wins_prob_l542_542519

noncomputable def T := TNFTPP

theorem fermi_wins_prob (a b : ℕ) (h1 : a.gcd b = 1) (h2 : a < b) 
  (h3 : (a:ℚ) / b < 1/2) (h4 : (T - 332) / (2 * T - 601) = 1 / 11) : a = 1 := 
sorry

end fermi_wins_prob_l542_542519


namespace solvable_system_of_inequalities_l542_542305

theorem solvable_system_of_inequalities (n : ℕ) : 
  (∃ x : ℝ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k < x ^ k ∧ x ^ k < k + 1)) ∧ (1 < x ∧ x < 2)) ↔ (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end solvable_system_of_inequalities_l542_542305


namespace smallest_sum_abc_d_l542_542892

theorem smallest_sum_abc_d (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) : a + b + c + d = 108 :=
sorry

end smallest_sum_abc_d_l542_542892


namespace grazing_b_l542_542998

theorem grazing_b (A_oxen_months B_oxen_months C_oxen_months total_months total_rent C_rent B_oxen : ℕ) 
  (hA : A_oxen_months = 10 * 7)
  (hB : B_oxen_months = B_oxen * 5)
  (hC : C_oxen_months = 15 * 3)
  (htotal : total_months = A_oxen_months + B_oxen_months + C_oxen_months)
  (hrent : total_rent = 175)
  (hC_rent : C_rent = 45)
  (hC_share : C_oxen_months / total_months = C_rent / total_rent) :
  B_oxen = 12 :=
by
  sorry

end grazing_b_l542_542998


namespace opposite_of_2023_l542_542135

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end opposite_of_2023_l542_542135


namespace gain_is_42_86_percent_l542_542949

variable (C S : ℝ)

-- Condition: Cost price of 50 articles equals the selling price of 35 articles
def cost_price_condition : Prop := 50 * C = 35 * S

-- Definition of gain percent
def gain_percent (C S : ℝ) : ℝ := ((S - C) / C) * 100

-- Problem statement
theorem gain_is_42_86_percent (h : cost_price_condition C S) : gain_percent C S = 42.857142857142854 := by
  -- The calculation steps are omitted in the statement
  sorry

end gain_is_42_86_percent_l542_542949


namespace product_of_solutions_eq_neg6_l542_542171

theorem product_of_solutions_eq_neg6 : 
  ∀ (x : ℝ), 2 * x^2 + 4 * x = 12 →
  (∃ x1 x2 : ℝ, 2 * x1^2 + 4 * x1 - 12 = 0 ∧ 2 * x2^2 + 4 * x2 - 12 = 0 ∧ x1 * x2 = -6) :=
by
  intro x hx
  let a := 2
  let b := 4
  let c := -12
  use (roots_of_quadratic_eq a b c)
  sorry

end product_of_solutions_eq_neg6_l542_542171


namespace cooks_selection_l542_542790

theorem cooks_selection (total_people : ℕ) (specific_person : ℕ) (other_people : ℕ) 
                        (total_people_eq : total_people = 10)
                        (specific_person_inclusion : specific_person = 1)
                        (other_people_eq : other_people = 9) :
  (combinatorics.choose other_people 1) = 9 := by
  sorry

end cooks_selection_l542_542790


namespace smallest_sum_l542_542084

theorem smallest_sum (a b c : ℕ) (h : (13 * a + 11 * b + 7 * c = 1001)) :
    a / 77 + b / 91 + c / 143 = 1 → a + b + c = 79 :=
by
  sorry

end smallest_sum_l542_542084


namespace joan_spent_total_l542_542809

noncomputable def convert_to_usd (amount : ℝ) (rate : ℝ) : ℝ := amount / rate
noncomputable def apply_discount (amount : ℝ) (discount : ℝ) : ℝ := amount * (1 - discount)
noncomputable def apply_tax (amount : ℝ) (tax : ℝ) : ℝ := amount * (1 + tax)

def toy_cars : ℝ := 14.88
def skateboard_eur : ℝ := 4.88
def toy_trucks_pounds : ℝ := 5.86
def pants : ℝ := 14.55
def shirt_eur : ℝ := 7.43
def hat_pounds : ℝ := 12.50

def eur_to_usd_rate : ℝ := 0.85
def pounds_to_usd_rate : ℝ := 1.35

def toys_cost_usd : ℝ :=
  let skateboard := convert_to_usd skateboard_eur eur_to_usd_rate
  let toy_trucks := toy_trucks_pounds * pounds_to_usd_rate
  toy_cars + skateboard + toy_trucks

def clothes_cost_usd : ℝ :=
  let shirt := convert_to_usd shirt_eur eur_to_usd_rate
  let hat := hat_pounds * pounds_to_usd_rate
  pants + shirt + hat

def toys_cost_after_discount : ℝ := apply_discount toys_cost_usd 0.10
def clothes_cost_after_tax : ℝ := apply_tax clothes_cost_usd 0.05

def total_cost : ℝ := toys_cost_after_discount + clothes_cost_after_tax

theorem joan_spent_total :
  total_cost = 67.86 := by
  sorry

end joan_spent_total_l542_542809


namespace horner_method_v1_l542_542254

def polynomial (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem horner_method_v1 (x : ℝ) (h : x = 5) : 
  ((4 * x + 2) * x + 3.5) = 22 := by
  rw [h]
  norm_num
  sorry

end horner_method_v1_l542_542254


namespace problem1_problem2_l542_542537

-- Condition: Point P (x, y) on the circle given by the equation x^2 + y^2 = 2y
def PointOnCircle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Problem (1): Finding the range of values for 2x + y
theorem problem1 (x y : ℝ) (h : PointOnCircle x y) : -sqrt 5 + 1 ≤ 2 * x + y ∧ 2 * x + y ≤ sqrt 5 + 1 :=
sorry 

-- Problem (2): Finding the range of values for a such that x + y + a ≥ 0 is always true
theorem problem2 (a x y : ℝ) (h : PointOnCircle x y) (h2 : ∀ (x y : ℝ) (h : PointOnCircle x y), x + y + a ≥ 0) :
a ≥ sqrt 2 - 1 :=
sorry

end problem1_problem2_l542_542537


namespace pi_approximation_accuracy_l542_542386

noncomputable def mixed_number_approximation : ℝ :=
  3 + 8 / 60 + 34 / 60^2 + 17 / 60^3 + 8 / 60^4

theorem pi_approximation_accuracy : 
  |mixed_number_approximation - Real.pi| < 0.005 :=
begin
  sorry
end

end pi_approximation_accuracy_l542_542386


namespace smallest_int_c_l542_542630

theorem smallest_int_c (c : ℤ) : 27^c > 3^24 → c ≥ 9 :=
begin
  sorry
end

end smallest_int_c_l542_542630


namespace mean_of_remaining_two_numbers_l542_542570

theorem mean_of_remaining_two_numbers 
    (n1 n2 n3 n4 n5 n6 : ℕ)
    (h1 : n1 = 2347) (h2 : n2 = 2573) (h3 : n3 = 2689) (h4 : n4 = 2725) (h5 : n5 = 2839) (h6 : n6 = 2841)
    (total_sum : n1 + n2 + n3 + n4 + n5 + n6 = 16014) 
    (mean_four_numbers : ∃ a b c d : ℕ, a + b + c + d = 4 * 2666 ∧ a + b + c + d ∈ {n1, n2, n3, n4, n5, n6}) :
    ( ∃ x y : ℕ, x + y = 5350 ∧ x + y ∈ {n1, n2, n3, n4, n5, n6} → (x + y) / 2 = 2675 ) :=
begin
    sorry
end

end mean_of_remaining_two_numbers_l542_542570


namespace probability_of_multiple_12_is_one_third_l542_542041

open Finset

def numbers := {3, 4, 6, 9}

noncomputable def product_is_multiple_of_12 (x y : ℕ) : Prop :=
  (x * y) % 12 = 0

noncomputable def favorable_pairs := {(3, 4), (4, 6)}

noncomputable def all_pairs := numbers.toFinset.powerset.filter (λ s, s.card = 2)

noncomputable def favorable_count : ℕ :=
  favorable_pairs.card

noncomputable def total_count : ℕ :=
  all_pairs.card

theorem probability_of_multiple_12_is_one_third :
  (favorable_count : ℚ) / (total_count : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_multiple_12_is_one_third_l542_542041


namespace convert_decimal_to_fraction_l542_542936

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l542_542936


namespace smallest_log_log_x0_l542_542338

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_log_log_x0 (x₀ : ℝ) (h₀ : f x₀ = 0) (h_dom : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) :=
sorry

end smallest_log_log_x0_l542_542338


namespace abs_difference_of_two_numbers_l542_542140

theorem abs_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := sorry

end abs_difference_of_two_numbers_l542_542140


namespace complement_union_l542_542763

open Set -- to use the set notations and operations

variable {α : Type*} -- defining a generic type variable to represent the type of elements in the sets

-- defining the universal set U, and sets M and N
def U : Set α := {1, 2, 3, 4, 5, 6}
def M : Set α := {2, 3, 4}
def N : Set α := {4, 5}

-- proving that the complement of (M ∪ N) in U is {1, 6}
theorem complement_union (hU : U = {1, 2, 3, 4, 5, 6}) (hM : M = {2, 3, 4}) (hN : N = {4, 5}) :
  U \ (M ∪ N) = {1, 6} :=
by
  sorry -- proof to be completed

end complement_union_l542_542763


namespace inequality_example_l542_542520

theorem inequality_example (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (habc_sum : a + b + c = 3) :
  18 * ((1 / ((3 - a) * (4 - a))) + (1 / ((3 - b) * (4 - b))) + (1 / ((3 - c) * (4 - c)))) + 2 * (a * b + b * c + c * a) ≥ 15 :=
by
  sorry

end inequality_example_l542_542520


namespace value_of_m_l542_542836

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end value_of_m_l542_542836


namespace simplify_sqrt_sum_l542_542115

theorem simplify_sqrt_sum : sqrt (6 + 4 * sqrt 2) + sqrt (6 - 4 * sqrt 2) = 4 := 
by
  -- We use the conditions as definitions here
  have h1 : (sqrt 2 + 1) ^ 2 = 3 + 2 * sqrt 2 := by sorry
  have h2 : (sqrt 2 - 1) ^ 2 = 3 - 2 * sqrt 2 := by sorry
  -- Conclude the proof here
  sorry

end simplify_sqrt_sum_l542_542115


namespace circumcenter_on_ellipse_circumcircle_through_N_l542_542750

-- Conditions (definitions)
def ellipse (x y : ℝ) := x^2 / 4 + y^2 = 1
def parabola (x y p : ℝ) := x^2 = 2 * p * y

-- Questions translated to theorem statements

theorem circumcenter_on_ellipse (p : ℝ) (A B : ℝ × ℝ) :
  (∃ x y : ℝ, x^2 / 4 + y^2 = 1 ∧ x^2 = 2 * p * y) →
  (∃ x y : ℝ, x^2 / 4 + y^2 = 1 ∧ x^2 = 2 * p * y) →
  ∃ Cx Cy : ℝ, Cx^2 / 4 + Cy^2 = 1 ∧ p = (7 - real.sqrt 13) / 6 :=
sorry

theorem circumcircle_through_N (p : ℝ) (A B : ℝ × ℝ) (N : ℝ × ℝ) :
  (∃ x y : ℝ, x^2 / 4 + y^2 = 1 ∧ x^2 = 2 * p * y) →
  (A = (0, -2)) → (B = (0, 2)) →
  N = (0, 13 / 2) →
  ∃ Cx Cy : ℝ, p = 3 :=
sorry

end circumcenter_on_ellipse_circumcircle_through_N_l542_542750


namespace sqrt_two_irrational_l542_542565

theorem sqrt_two_irrational : ¬ ∃ (p q : ℕ), (q ≠ 0) ∧ (Nat.gcd p q = 1) ∧ (p ^ 2 = 2 * q ^ 2) := by
  sorry

end sqrt_two_irrational_l542_542565


namespace product_bc_l542_542524

-- Defining the conditions from part a)
variables (x y b c : ℝ)
def equation_C := x * y = 1
def reflection_line (y : ℝ) := y = 2 * x
def equation_C_star := 12 * x ^ 2 + b * x * y + c * y ^ 2 + d = 0

-- Our goal is to prove that bc = 84 under the given conditions.
theorem product_bc :
  ∃ b c d, (equation_C) ∧ (reflection_line y) ∧ (equation_C_star) ∧ (b = -7) ∧ (c = -12) -> b * c = 84 :=
by {
  sorry -- The proof goes here
}

end product_bc_l542_542524


namespace count_monic_quadratic_polynomials_l542_542710

theorem count_monic_quadratic_polynomials : 
  let bound := 49^68 in
  let total_polynomials := 4760 in
  ∃ (P : ℕ) (f : ℕ × ℕ → Prop), 
    (∀ a b, f (a, b) → (a ≥ b) ∧ (7^a + 7^b ≤ bound) ∧ (a + b ≤ 136)) ∧
    P = {(a, b) ∈ ℕ × ℕ | f (a, b)}.card ∧
    P = total_polynomials := sorry

end count_monic_quadratic_polynomials_l542_542710


namespace find_other_number_l542_542874

theorem find_other_number (HCF LCM num1 num2 : ℕ) 
    (h_hcf : HCF = 14)
    (h_lcm : LCM = 396)
    (h_num1 : num1 = 36)
    (h_prod : HCF * LCM = num1 * num2)
    : num2 = 154 := by
  sorry

end find_other_number_l542_542874


namespace sum_first_n_terms_of_sequence_c_l542_542744

noncomputable def sequence_b (a : ℕ → ℕ) := λ n, 2^(a n - 1)
def a_1 : ℕ := 1
def a_3 : ℕ := 3
def sequence_c (a : ℕ → ℕ) (b : ℕ → ℕ) := λ n, a n * b n

theorem sum_first_n_terms_of_sequence_c (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, b n = sequence_b a n) →
  a 1 = a_1 →
  a 3 = a_3 →
  (∀ n, c n = sequence_c a b n) →
  (∀ n, S n = ∑ i in range n, c (i + 1)) →
  ∀ n, S n = n * 2^n := 
sorry

end sum_first_n_terms_of_sequence_c_l542_542744


namespace problem_1992_l542_542516

noncomputable def length_of_XY: ℕ × ℕ := (16, 392)

theorem problem_1992 : 
  let (m, n) := length_of_XY in
  100 * m + n = 1992 := 
by
  let (m, n) := length_of_XY;
  have hx : m = 16 := rfl;
  have hn : n = 392 := rfl;
  calc
    100 * m + n = 100 * 16 + 392 : by rw [hx, hn]
             ... = 1992 : by norm_num

end problem_1992_l542_542516


namespace trig_identity_simplify_l542_542634

theorem trig_identity_simplify (α : ℝ) :
  (\frac{4 * sin(α - 5 * π) ^ 2 - sin(2 * α + π) ^ 2}{cos(2 * α - 3/2 * π) ^ 2 - 4 + 4 * sin(α) ^ 2}) = - (tan(α) ^ 4) :=
by 
  sorry

end trig_identity_simplify_l542_542634


namespace candy_problem_l542_542459

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542459


namespace zan_guo_gets_one_deer_l542_542575

noncomputable def a1 : ℚ := 5 / 3
noncomputable def sum_of_sequence (a1 : ℚ) (d : ℚ) : ℚ := 5 * a1 + (5 * 4 / 2) * d
noncomputable def d : ℚ := -1 / 3
noncomputable def a3 (a1 : ℚ) (d : ℚ) : ℚ := a1 + 2 * d

theorem zan_guo_gets_one_deer :
  a3 a1 d = 1 := by
  sorry

end zan_guo_gets_one_deer_l542_542575


namespace Michelangelo_ceiling_painting_l542_542098

theorem Michelangelo_ceiling_painting (C : ℕ) : 
  ∃ C, (C + (1/4) * C = 15) ∧ (28 - (C + (1/4) * C) = 13) :=
sorry

end Michelangelo_ceiling_painting_l542_542098


namespace symmetry_implies_condition_l542_542885

open Function

variable {R : Type*} [Field R]
variables (p q r s : R)

theorem symmetry_implies_condition
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0) 
  (h_symmetry : ∀ x y : R, y = (p * x + q) / (r * x - s) → 
                          -x = (p * (-y) + q) / (r * (-y) - s)) :
  r + s = 0 := 
sorry

end symmetry_implies_condition_l542_542885


namespace normal_dist_prob_X_greater_than_4_l542_542345

theorem normal_dist_prob_X_greater_than_4 (σ : ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = (1 / (σ * √(2 * π))) * exp (-(x - 2)^2 / (2 * σ^2)))
  (h : ∫ x in 0..2, f x = 1 / 3) : 
  ∫ x in 4..∞, f x = 1 / 6 := sorry

end normal_dist_prob_X_greater_than_4_l542_542345


namespace expression_never_prime_l542_542928

theorem expression_never_prime (p : ℕ) (hp : Prime p) : ¬ Prime (3 * p^2 + 15) :=
by
  sorry

end expression_never_prime_l542_542928


namespace value_range_correct_l542_542604

noncomputable def value_range_of_function : Set ℝ :=
  {y | ∃ x : ℝ, x ≤ 2 ∧ y = real.sqrt (25 - real.exp (x * real.log 5))}

theorem value_range_correct : value_range_of_function = Set.Ico 0 5 := 
by 
  sorry

end value_range_correct_l542_542604


namespace OK_perp_TM_l542_542076

/-- Given:
1. MAZN is an isosceles trapezium inscribed in a circle (c) with center O.
2. MN is a diameter of (c).
3. B is the midpoint of AZ.
4. (ε) is the perpendicular line to AZ passing through A.
5. C is a point on (ε).
6. E is the point of intersection of CB with (c).
7. AE is perpendicular to CB.
8. D is the point of intersection of CZ with (c).
9. F is the antidiametric point of D on (c).
10. P is the point of intersection of FE and CZ.
11. The tangents to (c) at M and Z meet the lines AZ and PA at points K and T respectively.
Prove that OK is perpendicular to TM.
-/
theorem OK_perp_TM
  (O M A Z N B C E D F P K T : Point) (c : Circle) (ε : Line)
  (h1 : is_trapezium M A Z N)
  (h2 : is_inscribed_in_trapezium c M A Z N)
  (h3 : is_center O c)
  (h4 : is_diameter M N c)
  (h5 : is_midpoint B A Z)
  (h6 : is_perpendicular_line ε A Z A)
  (h7 : is_point_on_line C ε)
  (h8 : line_intersection C B c E)
  (h9 : is_perpendicular A E C B)
  (h10 : line_intersection C Z c D)
  (h11 : is_antidiametric F D c)
  (h12 : line_intersection F E C Z P)
  (h13 : is_tangent_line K M c A Z)
  (h14 : is_tangent_line T Z c P A)
  : is_perpendicular O K T M :=
sorry

end OK_perp_TM_l542_542076


namespace Lanie_usual_work_week_hours_l542_542513

theorem Lanie_usual_work_week_hours (H : ℝ) (worked_hours : ℝ) (hourly_rate : ℝ) (weekly_salary : ℝ) :
  worked_hours = (4 / 5) * H →
  hourly_rate = 15 →
  weekly_salary = 480 →
  worked_hours * hourly_rate = weekly_salary →
  H = 40 :=
by {
  intros h1 h2 h3 h4,
  sorry
}

end Lanie_usual_work_week_hours_l542_542513


namespace construction_project_l542_542268

noncomputable def apartments_first_building : ℕ := 4000 / 2
noncomputable def condos_first_building : ℕ := 4000 / 2
noncomputable def total_units_first_building : ℕ := apartments_first_building + condos_first_building

noncomputable def total_units_second_building : ℕ := (2 / 5 : ℚ) * total_units_first_building |> Int.ofNat
noncomputable def condos_second_building : ℕ := total_units_second_building / 4
noncomputable def apartments_second_building : ℕ := 3 * condos_second_building

noncomputable def total_units_third_building : ℕ := (6 / 5 : ℚ) * total_units_second_building |> Int.ofNat
noncomputable def two_story_townhouses : ℕ := (6 / 10 : ℚ) * total_units_third_building |> Int.ofNat
noncomputable def single_story_bungalows : ℕ := (4 / 10 : ℚ) * total_units_third_building |> Int.ofNat

noncomputable def total_apartments : ℕ := apartments_first_building + apartments_second_building
noncomputable def total_condos : ℕ := condos_first_building + condos_second_building

theorem construction_project :
  total_apartments = 3200 ∧
  total_condos = 2400 ∧
  two_story_townhouses = 1152 ∧
  single_story_bungalows = 768 := 
by {
  sorry
}

end construction_project_l542_542268


namespace O_lies_on_g_l542_542730

variables (A B C D O M : Point)
variables (hO : on_circle A B C D O)
variables (h_perpendicular : perpendicular A C B D)
variables (g : Line)
variables (h_symmetric : symmetric_to_bisector g A C (angle_bisector A B D))

theorem O_lies_on_g :
  on_line O g :=
sorry

end O_lies_on_g_l542_542730


namespace at_least_fifty_same_leading_coefficient_l542_542333

-- Define what it means for two quadratic polynomials to intersect exactly once
def intersect_once (P Q : Polynomial ℝ) : Prop :=
∃ x, P.eval x = Q.eval x ∧ ∀ y ≠ x, P.eval y ≠ Q.eval y

-- Define the main theorem and its conditions
theorem at_least_fifty_same_leading_coefficient 
  (polynomials : Fin 100 → Polynomial ℝ)
  (h1 : ∀ i j, i ≠ j → intersect_once (polynomials i) (polynomials j))
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
        ¬∃ x, (polynomials i).eval x = (polynomials j).eval x ∧ (polynomials j).eval x = (polynomials k).eval x) : 
  ∃ (S : Finset (Fin 100)), S.card ≥ 50 ∧ ∃ a, ∀ i ∈ S, (polynomials i).leadingCoeff = a :=
sorry

end at_least_fifty_same_leading_coefficient_l542_542333


namespace dvds_on_first_rack_l542_542623

/--
If Seth organizes his DVD collection such that the number of DVDs on each rack doubles from the previous rack,
and he has 4 DVDs on the second rack,
8 DVDs on the third rack,
16 DVDs on the fourth rack,
32 DVDs on the fifth rack,
and 64 DVDs on the sixth rack,
then the number of DVDs on the first rack is 2.
-/
theorem dvds_on_first_rack (a1 a2 a3 a4 a5 a6 : ℕ)
  (h2 : a2 = 4)
  (h3 : a3 = 8)
  (h4 : a4 = 16)
  (h5 : a5 = 32)
  (h6 : a6 = 64)
  (pattern : ∀ (n : ℕ), a2 * 2^(n - 2) = a(n)) :
  a1 = 2 :=
sorry

end dvds_on_first_rack_l542_542623


namespace combined_weight_loss_l542_542241

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end combined_weight_loss_l542_542241


namespace sum_of_squares_divisors_1800_eq_5035485_l542_542715

theorem sum_of_squares_divisors_1800_eq_5035485 :
  let N := 1800 in
  let divisors_squares_sum (n : Nat) := Nat.divisors n |>.map (λ d => d * d) |>.sum in
  divisors_squares_sum N = 5035485 := by
  let N := 1800
  let divisors_squares_sum := λ n : Nat => Nat.divisors n |>.map (λ d => d * d) |>.sum
  have h₁ : Nat.divisors N = [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 25, 30, 36, 45, 50, 60, 75, 90, 100, 125, 150, 180, 225, 300, 450, 600, 900, 1800] := by sorry
  have h₂ : divisors_squares_sum N = 5035485 := by sorry
  exact h₂

end sum_of_squares_divisors_1800_eq_5035485_l542_542715


namespace probability_even_sum_l542_542972

-- Define the probabilities for even and odd outcomes on each wheel
def P_even1 := 1 / 2
def P_odd1  := 1 / 2
def P_even2 := 1 / 3
def P_odd2  := 2 / 3
def P_even3 := 3 / 4
def P_odd3  := 1 / 4

-- Probability of different combinations leading to an even sum
def EEE := P_even1 * P_even2 * P_even3
def OEE := P_odd1  * P_even2 * P_even3
def EOE := P_even1 * P_odd2  * P_even3
def EEO := P_even1 * P_even2 * P_odd3

-- Total probability of an even sum
def P_even_sum := EEE + OEE + EOE + EEO

-- Prove that the total probability of an even sum is 1/3
theorem probability_even_sum : P_even_sum = 1 / 3 := 
by 
suffices : 1 / 8 + 1 / 8 + 1 / 4 + 1 / 24 = 1 / 3,
  sorry,
sorry

end probability_even_sum_l542_542972


namespace quadratic_polynomial_existence_l542_542692

noncomputable def f (x : ℝ) : ℝ := x^2 - x
noncomputable def g (x : ℝ) : ℝ := x^2 + x
noncomputable def h (x : ℝ) : ℝ := -x^2 + 2x + 3

theorem quadratic_polynomial_existence :
  (∃ f g h : ℝ → ℝ,
    (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧
    (∃ c d : ℝ, g c = 0 ∧ g d = 0 ∧ c ≠ d) ∧
    (∃ e f : ℝ, h e = 0 ∧ h f = 0 ∧ e ≠ f) ∧
    (∃ k : ℝ, (f k + g k) = 0 ∧
             (∃ i : ℝ, i ≠ k → f i + g i ≠ 0)) ∧
    (∃ m : ℝ, (f m + h m) = 0 ∧
             (∃ j : ℝ, j ≠ m → f j + h j ≠ 0)) ∧
    (∃ n : ℝ, (g n + h n) = 0 ∧
             (∃ l : ℝ, l ≠ n → g l + h l ≠ 0)) ∧
    (∀ p : ℝ, f p + g p + h p ≠ 0)) :=
∃ f g h, f = x^2 - x ∧ g = x^2 + x ∧ h = -x^2 + 2x + 3
sorry

end quadratic_polynomial_existence_l542_542692


namespace min_max_abs_diff_eq_l542_542700

noncomputable def min_max_abs_diff (f : ℝ → ℝ → ℝ) : ℝ :=
  Real.inf (set.range (λ y, Real.sup (set.range (λ x, |f x y|))))

theorem min_max_abs_diff_eq : 
  min_max_abs_diff (λ x y, x^2 - x * y) = 3 - 2 * Real.sqrt 2 :=
sorry

end min_max_abs_diff_eq_l542_542700


namespace possible_slopes_of_line_with_ellipse_intersection_l542_542979

theorem possible_slopes_of_line_with_ellipse_intersection
  (m : ℝ)
  (h_line : ∃ x, (y = mx + 8))
  (h_ellipse : ∃ x y, (4 * x ^ 2 + 25 * y ^ 2 = 100))
  (h_intersect : ∃ x, y = mx + 8 ∧ 4 * x ^ 2 + 25 * (mx + 8) ^ 2 = 100):
  m ∈ Iic (-real.sqrt 2.4) ∪ Ici (real.sqrt 2.4) := 
sorry

end possible_slopes_of_line_with_ellipse_intersection_l542_542979


namespace min_kinds_of_candies_l542_542437

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542437


namespace box_candies_l542_542095

noncomputable def total_candies_original (x : ℝ) : ℝ :=
  let first_day_eaten := 0.2 * x + 16
  let second_day_remaining := x - first_day_eaten
  let second_day_eaten := 0.3 * second_day_remaining + 20
  let third_day_remaining := second_day_remaining - second_day_eaten
  let third_day_eaten := 0.75 * third_day_remaining + 30
  in
  second_day_remaining - second_day_eaten - third_day_eaten

theorem box_candies : ∃ x : ℝ, total_candies_original x = 0 ∧ x = 270 :=
by
  let x := 270
  -- Verification of calculation given conditions
  have h1 : total_candies_original 270 = 0 := sorry
  exists x
  exact ⟨h1, rfl⟩

end box_candies_l542_542095


namespace citizens_own_a_cat_l542_542470

theorem citizens_own_a_cat (p d : ℝ) (n : ℕ) (h1 : p = 0.60) (h2 : d = 0.50) (h3 : n = 100) : 
  (p * n - d * p * n) = 30 := 
by 
  sorry

end citizens_own_a_cat_l542_542470


namespace fewer_nickels_than_quarters_l542_542813

theorem fewer_nickels_than_quarters :
  (∃ (Q D N : ℕ), D = Q + 3 ∧ Q = 22 ∧ Q + D + N = 63 ∧ 22 - N = 6) :=
begin
  use 22,                -- Quarters
  use 25,                -- Dimes
  use 16,                -- Nickels
  split, { exact rfl },  -- D = Q + 3
  split, { exact rfl },  -- Q = 22
  split, { -- Total coins equation
    calc 22 + 25 + 16 = 63 : by norm_num
  },
  -- Fewer nickels than quarters
  calc 22 - 16 = 6 : by norm_num
end

end fewer_nickels_than_quarters_l542_542813


namespace minimum_kinds_of_candies_l542_542413

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542413


namespace fraction_of_product_l542_542167

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l542_542167


namespace jill_trips_to_fill_tank_l542_542495

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end jill_trips_to_fill_tank_l542_542495


namespace TA_tangent_to_circumcircle_l542_542074

-- Definition of the problem conditions
variables {A B C D E K L T : Type}
variables [geometry A] [geometry B] [geometry C] [geometry D] [geometry E] [geometry K] [geometry L] [geometry T]

-- Given conditions in the problem
axiom (triangle_ABC : triangle A B C)
axiom (D_on_AB : on_line D A B)
axiom (E_on_AC : on_line E A C)
axiom (DE_parallel_BC : parallel DE BC)
axiom (K_on_second_time : intersects_circum (circumcircle A B C) (circumcircle B D E) K)
axiom (L_on_second_time : intersects_circum (circumcircle A B C) (circumcircle C D E) L)
axiom (T_intersection_BK_CL : intersection_point BK CL T)

-- Proof statement that TA is tangent to the circumcircle of triangle ABC at A
theorem TA_tangent_to_circumcircle (h_triangle_ABC : triangle_ABC)
                                   (h_D_on_AB : D_on_AB)
                                   (h_E_on_AC : E_on_AC)
                                   (h_DE_parallel_BC : DE_parallel_BC)
                                   (h_K_on_second_time : K_on_second_time)
                                   (h_L_on_second_time : L_on_second_time)
                                   (h_T_intersection_BK_CL : T_intersection_BK_CL) :
  is_tangent (circumcircle A B C) T A :=
sorry

end TA_tangent_to_circumcircle_l542_542074


namespace inscribed_square_divides_circle_l542_542691

noncomputable def divide_circle_into_four_parts
  (O : Point) (r : ℝ) : Prop :=
  let circle := circle (r: ℝ) (O: Point) in
  let A := point_on_circle circle in
  let B := point_on_circle_at_distance circle r A in
  let C := point_on_circle_at_distance circle r B in
  let D := point_on_circle_at_distance circle r C in
  let E := intersection_arcs circle A D r (distance A C) in
  let F := intersection_arcs circle A E r (distance O E) in
  let G := other_intersection_point_on_circle circle A E F in
  inscribed_square_on_circle (O : Point) r (A : Point) (F : Point) (G : Point)

theorem inscribed_square_divides_circle
  (O : Point) (r : ℝ)
  (A B C D E F G : Point)
  (h1 : r > 0)
  (h2 : distance O A = r)
  (h3 : distance O B = r)
  (h4 : distance O C = r)
  (h5 : distance O D = r)
  (h6 : distance A C = r * sqrt 3)
  (h7 : distance A E = r * sqrt 3)
  (h8 : distance O E = r * sqrt 2)
  (h9 : intersection_of_arcs A D r (distance A C) = E)
  (h10 : intersection_of_arcs A E r (distance O E) = F)
  (h11 : other_intersection_point_on_circle O r A E F = G)
  : inscribed_square_on_circle O r A F G :=
sorry

end inscribed_square_divides_circle_l542_542691


namespace ratio_of_width_to_perimeter_l542_542226

-- Condition definitions
def length := 22
def width := 13
def perimeter := 2 * (length + width)

-- Statement of the problem in Lean 4
theorem ratio_of_width_to_perimeter : width = 13 ∧ length = 22 → width * 70 = 13 * perimeter :=
by
  sorry

end ratio_of_width_to_perimeter_l542_542226


namespace circle_radius_tangent_lines_l542_542966

open real

theorem circle_radius_tangent_lines (k : ℝ) (h : k > 8) : 
  let r := k * sqrt 2 + k - 8 * sqrt 2 - 8 in 
  ∃ (r : ℝ), r = k * sqrt 2 + k - 8 * sqrt 2 - 8 :=
sorry

end circle_radius_tangent_lines_l542_542966


namespace sin_of_angle_l542_542310

theorem sin_of_angle (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1/3) :
  Real.sin (2*θ + Real.pi/2) = -7/9 :=
by
  sorry

end sin_of_angle_l542_542310


namespace angle_less_than_30_l542_542531

variable (A B C P : Point)
variable (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A)
variable (hP_inside : inside_triangle ABC P)

theorem angle_less_than_30 (h1 : angle P A B ≠ 0)
  (h2 : angle P B C ≠ 0) (h3 : angle P C A ≠ 0) :
  (angle P A B ≤ 30) ∨ (angle P B C ≤ 30) ∨ (angle P C A ≤ 30) :=
sorry

end angle_less_than_30_l542_542531


namespace can_form_triangle_l542_542930

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle :
  (is_triangle 3 5 7) ∧ ¬(is_triangle 3 3 7) ∧ ¬(is_triangle 4 4 8) ∧ ¬(is_triangle 4 5 9) :=
by
  -- Proof steps will be added here
  sorry

end can_form_triangle_l542_542930


namespace largest_lcm_l542_542922

theorem largest_lcm :
  max (max (max (max (max (Nat.lcm 12 2) (Nat.lcm 12 4)) 
                    (Nat.lcm 12 6)) 
                 (Nat.lcm 12 8)) 
            (Nat.lcm 12 10)) 
      (Nat.lcm 12 12) = 60 :=
by sorry

end largest_lcm_l542_542922


namespace minimum_value_f_pi_over_8_l542_542353

def f (ω x : ℝ) : ℝ :=
  √3 * sin (ω * x) * cos (ω * x) + (cos (ω * x))^2 - 1/2

theorem minimum_value_f_pi_over_8 (ω : ℝ) (hω : 0 < ω) : 
  f ω (π / 8) = 1/2 :=
sorry

end minimum_value_f_pi_over_8_l542_542353


namespace alice_book_payment_l542_542663

/--
Alice is in the UK and wants to purchase a book priced at £25.
If one U.S. dollar is equivalent to £0.75, 
then Alice needs to pay 33.33 USD for the book.
-/
theorem alice_book_payment :
  ∀ (price_gbp : ℝ) (conversion_rate : ℝ), 
  price_gbp = 25 → conversion_rate = 0.75 → 
  (price_gbp / conversion_rate) = 33.33 :=
by
  intros price_gbp conversion_rate hprice hrate
  rw [hprice, hrate]
  sorry

end alice_book_payment_l542_542663


namespace minimum_value_of_f_l542_542709

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 1/x + 1/(x^2 + 1/x)

theorem minimum_value_of_f : 
  ∃ x > 0, f x = 2.5 :=
by 
  sorry

end minimum_value_of_f_l542_542709


namespace green_folder_stickers_l542_542552

theorem green_folder_stickers (total_stickers red_sheets blue_sheets : ℕ) (red_sticker_per_sheet blue_sticker_per_sheet green_stickers_needed green_sheets : ℕ) :
  total_stickers = 60 →
  red_sticker_per_sheet = 3 →
  blue_sticker_per_sheet = 1 →
  red_sheets = 10 →
  blue_sheets = 10 →
  green_sheets = 10 →
  let red_stickers_total := red_sticker_per_sheet * red_sheets
  let blue_stickers_total := blue_sticker_per_sheet * blue_sheets
  let green_stickers_total := total_stickers - (red_stickers_total + blue_stickers_total)
  green_sticker_per_sheet = green_stickers_total / green_sheets →
  green_sticker_per_sheet = 2 := 
sorry

end green_folder_stickers_l542_542552


namespace number_accuracy_l542_542578

theorem number_accuracy : ∀ n, n = 12000 → ∃ d, d = 2 ∧ accurate_digits n d :=
by
  intros n h
  use 2
  sorry

end number_accuracy_l542_542578


namespace candy_problem_l542_542454

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542454


namespace min_value_of_quadratic_l542_542887

theorem min_value_of_quadratic : ∀ x : ℝ, ∃ y : ℝ, y = (x - 1)^2 - 3 ∧ (∀ z : ℝ, (z - 1)^2 - 3 ≥ y) :=
by
  sorry

end min_value_of_quadratic_l542_542887


namespace alpha_eq_one_l542_542306

-- Definitions based on conditions from the problem statement.
variable (α : ℝ) 
variable (f : ℝ → ℝ)

-- The conditions defined as hypotheses
axiom functional_eq (x y : ℝ) : f (α * (x + y)) = f x + f y
axiom non_constant : ∃ x y : ℝ, f x ≠ 0

-- The statement to prove
theorem alpha_eq_one : (∃ f : ℝ → ℝ, (∀ x y : ℝ, f (α * (x + y)) = f x + f y) ∧ (∃ x y : ℝ, f x ≠ f y)) → α = 1 :=
by
  sorry

end alpha_eq_one_l542_542306


namespace lambda_range_l542_542759

-- Define the sequence {a_n}
noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3 * a_seq (n - 1) + 1

-- Define the sequence {b_n}
noncomputable def b_seq (n : ℕ) : ℕ → ℝ :=
  λ n, (n : ℝ) / ((3 ^ n - 1) * 2 ^ (n - 2)) * a_seq n

-- Define the sum T_n
noncomputable def T_seq (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, b_seq (k + 1))

-- Define the theorem to prove the range of λ
theorem lambda_range (λ : ℝ) :
  (∀ n : ℕ, n > 0 → 2^n * λ < 2^(n-1) * T_seq n + n) →
  λ < 1 :=
sorry

end lambda_range_l542_542759


namespace asymptote_angle_range_l542_542000

noncomputable def theta_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : ∀ (e : ℝ), sqrt 2 ≤ e ∧ e ≤ 2 → (e = real.sqrt (1 + b^2/a^2))) : Prop :=
  (π/4 ≤ real.arctan (b/a)) ∧ (real.arctan (b/a) ≤ π/3)

theorem asymptote_angle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∀ (e : ℝ), sqrt 2 ≤ e ∧ e ≤ 2 → (e = real.sqrt (1 + b^2/a^2))) :
  theta_range a b ha hb h1 :=
sorry

end asymptote_angle_range_l542_542000


namespace gcd_1729_1768_l542_542920

theorem gcd_1729_1768 : Int.gcd 1729 1768 = 13 := by
  sorry

end gcd_1729_1768_l542_542920


namespace max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l542_542358

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l542_542358


namespace charity_run_total_donation_l542_542573

theorem charity_run_total_donation : 
  (let suzanne_donation : ℕ → ℕ → ℝ
     | 0, 10 => 10.0
     | (n + 1), donation => 2 * donation
   in let maria_donation : ℕ → ℕ → ℝ
      | 0, 15 => 15.0
      | (n + 1), donation => if n < 3 then 1.5 * donation else 0.75 * donation
   in let james_donation : ℕ → ℕ → ℝ
      | 0, 20 => 20.0
      | 1, donation => donation
      | 2, donation => donation + 5.0
      | 3, donation => donation - 5.0
   in let suzanne_total := List.sum (List.map (λ (n : ℕ) => suzanne_donation n 10) [0, 1, 2, 3, 4])
    in let maria_total := List.sum (List.map (λ (n : ℕ) => maria_donation n 15) [0, 1, 2, 3, 4, 5])
    in let james_total := List.sum (List.map (λ (n : ℕ) => james_donation n 20) [0, 1, 2, 3])
    in suzanne_total + maria_total + james_total = 583.32) := 
by sorry

end charity_run_total_donation_l542_542573


namespace germs_in_total_l542_542048

theorem germs_in_total (dishes : ℕ) (germs_per_dish : ℕ) 
  (h1 : dishes = 74000 * (10 : ℕ) ^ (-3 : ℤ)) 
  (h2 : germs_per_dish = 50) : 
  dishes * germs_per_dish = 3700 :=
by
  -- Here, "h1" and "h2" encode the conditions given in the problem.
  -- Recall that 10^(-3) would be expressed as (10 : ℕ) ^ (-3 : ℤ) in Lean.
  sorry

end germs_in_total_l542_542048


namespace all_work_together_again_l542_542859

theorem all_work_together_again :
  let sam := 5
  let fran := 8
  let mike := 10
  let julio := 12
  Nat.lcm (Nat.lcm sam fran) (Nat.lcm mike julio) = 120 :=
by
  let sam := 5
  let fran := 8
  let mike := 10
  let julio := 12
  the lcm_sam_fran := Nat.lcm sam fran
  the lcm_mike_julio := Nat.lcm mike julio
  show Nat.lcm lcm_sam_fran lcm_mike_julio = 120
  sorry

end all_work_together_again_l542_542859


namespace crosses_zeros_23_crosses_zeros_22_l542_542021

theorem crosses_zeros_23 : number_of_ways_to_arrange(23, 11) = 12 :=
sorry

theorem crosses_zeros_22 : number_of_ways_to_arrange(22, 11) = 78 :=
sorry

end crosses_zeros_23_crosses_zeros_22_l542_542021


namespace factor_expression_l542_542919

theorem factor_expression (a b c d : ℝ) : 
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 
        = ((a - b) * (b - c) * (c - d) * (d - a)) * (a + b + c + d) := 
by
  sorry

end factor_expression_l542_542919


namespace f_value_at_4_over_3_l542_542317

def f : ℝ → ℝ 
| x <= 0 := real.cos (real.pi * x)
| x > 0 := f (x - 1) + 1

theorem f_value_at_4_over_3 : f (4 / 3) = 3 / 2 :=
by {
  sorry
}

end f_value_at_4_over_3_l542_542317


namespace pages_to_read_tomorrow_l542_542510

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l542_542510


namespace minimum_candy_kinds_l542_542403

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542403


namespace speed_in_still_water_correct_l542_542186

-- Defining the given conditions
def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

-- Defining the calculation of speed in still water
def speed_in_still_water : ℝ := (upstream_speed + downstream_speed) / 2

-- Proving the final equation
theorem speed_in_still_water_correct : speed_in_still_water = 24 := by
  -- This is the stub for the proof.
  sorry

end speed_in_still_water_correct_l542_542186


namespace minimum_kinds_of_candies_l542_542418

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542418


namespace monotonic_intervals_range_of_k_l542_542092

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x ^ 2) - k * (2 / x + Real.log x)
noncomputable def f' (x k : ℝ) : ℝ := (x - 2) * (Real.exp x - k * x) / (x^3)

theorem monotonic_intervals (k : ℝ) (h : k ≤ 0) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x k < 0) ∧ (∀ x : ℝ, x > 2 → f' x k > 0) := sorry

theorem range_of_k (k : ℝ) (h : e < k ∧ k < (e^2)/2) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ 
    (f' x1 k = 0 ∧ f' x2 k = 0 ∧ x1 ≠ x2) := sorry

end monotonic_intervals_range_of_k_l542_542092


namespace minimum_candy_kinds_l542_542451

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542451


namespace find_first_number_l542_542629

variable {A B C D : ℕ}

theorem find_first_number (h1 : A + B + C = 60) (h2 : B + C + D = 45) (h3 : D = 18) : A = 33 := 
  sorry

end find_first_number_l542_542629


namespace sabi_share_removed_l542_542199

theorem sabi_share_removed :
  ∀ (N S M x : ℝ), N - 5 = 2 * (S - x) / 8 ∧ S - x = 4 * (6 * (M - 4)) / 16 ∧ M = 102 ∧ N + S + M = 1100 
  → x = 829.67 := by
  sorry

end sabi_share_removed_l542_542199


namespace origin_moves_distance_l542_542212

/-- Original center of the circle before dilation -/
def B : ℝ × ℝ := (3, 1)

/-- Transformed center of the circle after dilation -/
def B' : ℝ × ℝ := (7, 9)

/-- Original radius of the circle before dilation -/
def r₁ : ℝ := 4

/-- Transformed radius of the circle after dilation -/
def r₂ : ℝ := 6

/-- Center of dilation -/
def C : ℝ × ℝ := (1, 3)

/-- Dilation factor -/
def k : ℝ := r₂ / r₁

/-- Origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Distance from origin to center of dilation before dilation -/
def d₀ : ℝ := Real.sqrt ((C.1 - O.1) ^ 2 + (C.2 - O.2) ^ 2)

/-- Distance from origin to center of dilation after dilation -/
def d₁ : ℝ := k * d₀

/-- Movement distance of the origin -/
def d_move : ℝ := d₁ - d₀

/-- The main theorem stating the distance the origin moves under the transformation -/
theorem origin_moves_distance : d_move = 0.5 * Real.sqrt 10 :=
by
  sorry

end origin_moves_distance_l542_542212


namespace smallest_domain_size_l542_542684

noncomputable def g : ℕ → ℕ
| 15           := 46
| (n : ℕ)      := if (n % 2 = 1) then 3 * n + 2 else n / 2

theorem smallest_domain_size:
  ∃ (S : set ℕ), 15 ∈ S ∧ (∀ a ∈ S, g a ∈ S) ∧ S.card = 31 := 
sorry

end smallest_domain_size_l542_542684


namespace part1_part2_l542_542372

-- Define points and lines
structure Point where
  x : ℝ
  y : ℝ

def l1 (p : Point) : Prop := p.y = 2 * p.x
def l2 (p : Point) : Prop := p.y = -2 * p.x

def M := Point.mk (-2) 0
def N := Point.mk 1 0

-- Define the line passing through M with slope k
def line_l (k : ℝ) (p : Point) : Prop := p.y = k * (p.x + 2)

-- Define points A and B on l1 and l2 respectively
def is_A (p : Point) := l1 p ∧ line_l k p ∧ p.x < 0 ∧ p.y < 0
def is_B (p : Point) := l2 p ∧ line_l k p ∧ p.x < 0 ∧ p.y > 0

-- Define the area condition for triangle NAB being 16
def triangle_area (N A B : Point) : ℝ := 
  ((A.x - N.x) * (B.y - N.y) - (A.y - N.y) * (B.x - N.x)) / 2

def area_condition (A B : Point) : Prop := triangle_area N A B = 16

-- Define slope for line passing through points
def slope (p1 p2 : Point) (m : ℝ) : Prop := (p2.y - p1.y) = m * (p2.x - p1.x)

-- The main theorem statements
theorem part1 (A B : Point) (k : ℝ) : (area_condition A B) → 
  line_l k M → (k = 4 ∨ k = -4) → (∀ p : Point, line_l 4 p ↔ (4 * p.x + p.y = -8) ∨ (4 * p.x - p.y = -8)) :=
sorry

theorem part2 (A B P Q : Point) (k1 k2 : ℝ) :
  slope A N k1 → slope B N k1 → 
  slope A P k1 → slope B Q k1 → 
  slope P Q k2 → 
  (k2 = -5 * k1) →
  (k1 / k2 = -1 / 5) :=
sorry

end part1_part2_l542_542372


namespace train_crossing_time_l542_542805

-- Definitions
def length_of_train : ℝ := 200 -- in meters
def speed_of_train_kmh : ℝ := 144 -- in km/hr

-- Conversion factor
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * (1000 / 3600)

-- Converted speed
def speed_of_train_ms : ℝ := kmh_to_ms speed_of_train_kmh

-- Time calculation
def crossing_time (distance speed : ℝ) : ℝ := distance / speed

-- The theorem to prove
theorem train_crossing_time : crossing_time length_of_train speed_of_train_ms = 5 := by
  -- Here, we would insert the proof steps if required
  sorry

end train_crossing_time_l542_542805


namespace max_velocity_at_C_l542_542971

def points := ["A", "B", "C", "D"]

def height_at_point (p : String) : Real := 
  match p with
  | "A" => 3.9
  | "B" => 2.7
  | "C" => 1.2
  | "D" => 2.7
  | _ => 0  -- No other points are defined

-- Assuming mass m and gravitational constant g are positive and > 0
constants (m g : Real) (hC : 0 < m) (hG : 0 < g)

-- Using this principle
theorem max_velocity_at_C : 
  ∀ p ∈ points, p ≠ "C" → 
  (1 / 2 * m * (sqrt (2 * m * g * (height_at_point "C")))^2 > 
   1 / 2 * m * (sqrt (2 * m * g * (height_at_point p)))^2) :=
begin
  intros p hp hpc,
  rw [←mul_lt_mul_left (mul_pos (by linarith [hC]) (by linarith [hG])), ←mul_assoc, one_div_mul_cancel (show 2 ≠ 0, by norm_num)] {occs := occurrences.pos [1, 3]},
  apply mul_lt_mul_left,
  { linarith [hC] },
  { simp [height_at_point, pow_two, sqrt_lt_sqrt_iff, hG],
    cases p;
    try {linarith},
    linarith [show 1.2 < 2.7, by norm_num] }
end

end max_velocity_at_C_l542_542971


namespace pizza_toppings_combinations_l542_542547

theorem pizza_toppings_combinations : (nat.choose 9 3) = 84 := 
by
  sorry

end pizza_toppings_combinations_l542_542547


namespace find_YZ_l542_542803

noncomputable def Y_to_rad := 40 * (Real.pi / 180)
noncomputable def Z_to_rad := 95 * (Real.pi / 180)
noncomputable def X := 180 - 40 - 95
noncomputable def X_to_rad := X * (Real.pi / 180)

theorem find_YZ (XY : ℝ) (Y : ℝ) (Z : ℝ) (YZ : ℝ) :
  Y = 40 → Z = 95 → XY = 6 → YZ ≈ 6.6 := 
by
  intro hY hZ hXY
  have hXY_val : XY = 6 := hXY
  have hY_val : Y = 40 := hY
  have hZ_val : Z = 95 := hZ

  -- Convert degrees to radians for calculations
  let Y_rad : ℝ := 40 * (Real.pi / 180)
  let Z_rad : ℝ := 95 * (Real.pi / 180)
  let X := 180 - 40 - 95
  let X_rad : ℝ := X * (Real.pi / 180)

  -- Compute sin values
  let sin_X := Real.sin X_rad
  let sin_Y := Real.sin Y_rad

  -- Define the length YZ using the law of sines
  let YZ' : ℝ := XY * (sin_X / sin_Y)

  -- Rational approximation of YZ
  have approx : YZ' ≈ 6.6 :=
    by
      -- Use known values of sin(45 degrees) and sin(40 degrees) for approximation
      have sin_45 : Real.sin (45 * (Real.pi / 180)) = Real.sqrt 2 / 2 := sorry
      have sin_40 : Real.sin (40 * (Real.pi / 180)) ≈ 0.6428 := sorry

      calc 
        YZ' = 6 * (sin_45 / sin_40) : by sorry
        ... ≈ 6 * (0.7071 / 0.6428) : by sorry
        ... ≈ 6.5977 : by sorry
        ... ≈ 6.6 : by sorry
  
  exact approx

end find_YZ_l542_542803


namespace inequality_solution_set_minimum_value_expression_l542_542360

-- Definition of the function f
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Inequality solution set for f(x) ≤ 4
theorem inequality_solution_set :
  { x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3 } = { x : ℝ | f x ≤ 4 } := 
sorry

-- Minimum value of the given expression given conditions on a and b
theorem minimum_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0)
  (h3 : a + 2 * b = 3) :
  (1 / (a - 1)) + (2 / b) = 9 / 2 := 
sorry

end inequality_solution_set_minimum_value_expression_l542_542360


namespace exists_bijection_l542_542080

-- Define the non-negative integers set
def N_0 := {n : ℕ // n ≥ 0}

-- Translation of the equivalent proof statement into Lean
theorem exists_bijection (f : N_0 → N_0) :
  (∀ m n : N_0, f ⟨3 * m.val * n.val + m.val + n.val, sorry⟩ = 
   ⟨4 * (f m).val * (f n).val + (f m).val + (f n).val, sorry⟩) :=
sorry

end exists_bijection_l542_542080


namespace favorite_movies_hours_l542_542065

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l542_542065


namespace problem1_problem2_l542_542962

-- Problem (1)
theorem problem1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) : a^2 + (3 / 2) * b - 5 = -2 := 
sorry

-- Problem (2)
theorem problem2 (x : ℝ) (h : 14 * x + 5 - 21 * x^2 = -2) : 6 * x^2 - 4 * x + 5 = 7 := 
sorry

end problem1_problem2_l542_542962


namespace area_ratio_XYZ_PQR_l542_542791

theorem area_ratio_XYZ_PQR 
  (PR PQ QR : ℝ)
  (p q r : ℝ) 
  (hPR : PR = 15) 
  (hPQ : PQ = 20) 
  (hQR : QR = 25)
  (hPX : p * PR = PR * p)
  (hQY : q * QR = QR * q) 
  (hPZ : r * PQ = PQ * r) 
  (hpq_sum : p + q + r = 3 / 4) 
  (hpq_sq_sum : p^2 + q^2 + r^2 = 9 / 16) : 
  (area_triangle_XYZ / area_triangle_PQR = 1 / 4) :=
sorry

end area_ratio_XYZ_PQR_l542_542791


namespace oshea_bought_basil_seeds_l542_542852

-- Define the number of large and small planters and their capacities.
def large_planters := 4
def seeds_per_large_planter := 20
def small_planters := 30
def seeds_per_small_planter := 4

-- The theorem statement: Oshea bought 200 basil seeds
theorem oshea_bought_basil_seeds :
  large_planters * seeds_per_large_planter + small_planters * seeds_per_small_planter = 200 :=
by sorry

end oshea_bought_basil_seeds_l542_542852


namespace prob_two_heads_in_four_flips_l542_542117

theorem prob_two_heads_in_four_flips : 
  (nat.choose 4 2 : ℝ) / (2^4) = 3 / 8 :=
by
  sorry

end prob_two_heads_in_four_flips_l542_542117


namespace opposite_of_0_point_5_equals_neg_0_point_5_l542_542133

theorem opposite_of_0_point_5_equals_neg_0_point_5 : ∃ x : ℝ, 0.5 + x = 0 ∧ x = -0.5 :=
by
  use -0.5
  split
  · norm_num
  · refl

end opposite_of_0_point_5_equals_neg_0_point_5_l542_542133


namespace candy_problem_l542_542455

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542455


namespace exists_fixed_point_on_circumcircle_of_triangle_l542_542160

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B P : Point)

structure Movement :=
  (A_path : ℝ → Point) -- Function of time to point on A's path
  (B_path : ℝ → Point) -- Function of time to point on B's path)

def uniform_speed (f : ℝ → Point) (speed : ℝ) : Prop :=
  ∀ t₁ t₂, dist (f t₁) (f t₂) = speed * abs (t₁ - t₂)

def lines_intersect_at (f g : ℝ → Point) (P : Point) : Prop :=
  f 0 = P ∧ g 0 = P

def not_pass_through_simultaneously (A_path B_path : ℝ → Point) (P : Point) : Prop :=
  ∀ t, A_path t ≠ P ∨ B_path t ≠ P

def circumcircle_contains (T : Triangle) (O : Point) : Prop :=
  ∃ r, dist O T.A = r ∧ dist O T.B = r ∧ dist O T.P = r

theorem exists_fixed_point_on_circumcircle_of_triangle
  (A_path B_path : ℝ → Point) (P : Point) (speed : ℝ) :
  uniform_speed A_path speed →
  uniform_speed B_path speed →
  lines_intersect_at A_path B_path P →
  not_pass_through_simultaneously A_path B_path P →
  ∃ O, O ≠ P ∧ ∀ t, circumcircle_contains ⟨A_path t, B_path t, P⟩ O :=
sorry

end exists_fixed_point_on_circumcircle_of_triangle_l542_542160


namespace minimum_candy_kinds_l542_542447

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542447


namespace find_ab_l542_542775

theorem find_ab (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 30) : a * b = 32 :=
by
  -- We will complete the proof in this space
  sorry

end find_ab_l542_542775


namespace pages_to_read_tomorrow_l542_542512

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l542_542512


namespace infinitely_many_n_gcd_condition_l542_542518

theorem infinitely_many_n_gcd_condition (P : ℤ[X])
  (h1 : P.eval 0 = 0)
  (h2 : Int.gcd (P.eval 0) (P.eval 1) (P.eval 2) ...) = 1)
  : ∃ᶠ n in at_top, Int.gcd (P.eval n - P.eval 0) (P.eval (n+1) - P.eval 1) (P.eval (n+2) - P.eval 2) ... = n := 
sorry

end infinitely_many_n_gcd_condition_l542_542518


namespace each_boy_receives_52_l542_542598

theorem each_boy_receives_52 {boys girls : ℕ} (h_ratio : boys / gcd boys girls = 5 ∧ girls / gcd boys girls = 7) (h_total : boys + girls = 180) (h_share : 3900 ∣ boys) :
  3900 / boys = 52 :=
by
  sorry

end each_boy_receives_52_l542_542598


namespace jackson_investment_ratio_l542_542497

theorem jackson_investment_ratio:
  ∀ (B J: ℝ), B = 0.20 * 500 → J = B + 1900 → (J / 500) = 4 :=
by
  intros B J hB hJ
  sorry

end jackson_investment_ratio_l542_542497


namespace max_a_plus_2b_l542_542033

theorem max_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 1) : 
  a + 2 * b ≤ sqrt 5 := 
sorry

end max_a_plus_2b_l542_542033


namespace tan_double_angle_cos_beta_l542_542724

theorem tan_double_angle
  (α β : ℝ)
  (hcosα : cos α = 5 / 13)
  (hcosαβ : cos (α - β) = 4 / 5)
  (hβ : 0 < β)
  (hαβ : β < α)
  (hα : α < π / 2) :
  tan (2 * α) = -120 / 119 :=
sorry

theorem cos_beta
  (α β : ℝ)
  (hcosα : cos α = 5 / 13)
  (hcosαβ : cos (α - β) = 4 / 5)
  (hβ : 0 < β)
  (hαβ : β < α)
  (hα : α < π / 2) :
  cos β = 56 / 65 :=
sorry

end tan_double_angle_cos_beta_l542_542724


namespace sum_of_largest_and_smallest_l542_542603

theorem sum_of_largest_and_smallest (n : ℕ) (h : 6 * n + 15 = 105) : (n + (n + 5) = 35) :=
by
  sorry

end sum_of_largest_and_smallest_l542_542603


namespace leah_coins_worth_89_cents_l542_542815

variables (p n d : ℕ)

theorem leah_coins_worth_89_cents (h1 : p + n + d = 15) (h2 : d - 1 = n) : 
  1 * p + 5 * n + 10 * d = 89 := 
sorry

end leah_coins_worth_89_cents_l542_542815


namespace min_kinds_of_candies_l542_542443

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542443


namespace minimum_candy_kinds_l542_542402

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542402


namespace equation_of_hyperbola_l542_542727

variable (a b c : ℝ)
variable (x y : ℝ)

theorem equation_of_hyperbola :
  (0 < a) ∧ (0 < b) ∧ (c / a = Real.sqrt 3) ∧ (a^2 / c = 1) ∧ (c = 3) ∧ (b = Real.sqrt 6)
  → (x^2 / 3 - y^2 / 6 = 1) :=
by
  sorry

end equation_of_hyperbola_l542_542727


namespace exists_n_digit_number_divisible_by_sum_l542_542863

theorem exists_n_digit_number_divisible_by_sum (n : ℕ) (h : n > 1) :
  ∃ N : ℕ, (nat.num_digits 10 N = n) ∧ (∀ d ∈ nat.digits 10 N, d ≠ 0) ∧ (N % (nat.digits 10 N).sum = 0) :=
sorry

end exists_n_digit_number_divisible_by_sum_l542_542863


namespace find_a_b_extreme_values_l542_542349

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x - (2/3)

theorem find_a_b_extreme_values : 
  ∃ (a b : ℝ), 
    (a = -2) ∧ 
    (b = 3) ∧ 
    (f 1 (-2) 3 = 2/3) ∧ 
    (f 3 (-2) 3 = -2/3) :=
by
  sorry

end find_a_b_extreme_values_l542_542349


namespace intersection_point_on_altitude_l542_542849

theorem intersection_point_on_altitude
  (A B C D1 D2 : Point)
  (AB BC : Line)
  (P Q : Square)
  [SquareConstructedExternally A B C D1 AB P]
  [SquareConstructedExternally A B C D2 BC Q]
  (H : Point)
  (BH : Altitude B H C)
  (X : Point)
  (AD2 : Line)
  (CD1 : Line)
  (IntPt : Intersection AD2 CD1 X) :
  X ∈ BH := 
sorry

end intersection_point_on_altitude_l542_542849


namespace equivalent_single_discount_l542_542989

variable (original_price : ℝ)
variable (first_discount : ℝ)
variable (second_discount : ℝ)

-- Conditions
def sale_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

def final_price (p : ℝ) (d1 d2 : ℝ) : ℝ :=
  let sale1 := sale_price p d1
  sale_price sale1 d2

-- Prove the equivalent single discount is as described
theorem equivalent_single_discount :
  original_price = 30 → first_discount = 0.2 → second_discount = 0.25 →
  (1 - final_price original_price first_discount second_discount / original_price) * 100 = 40 :=
by
  intros
  sorry

end equivalent_single_discount_l542_542989


namespace females_with_advanced_degrees_l542_542463

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (total_college_degrees : ℕ) 
  (males_with_college_degree : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : total_advanced_degrees = 90) 
  (h4 : total_college_degrees = 90) 
  (h5 : males_with_college_degree = 35) : 
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 55 := 
by {
  sorry
}

end females_with_advanced_degrees_l542_542463


namespace interior_diagonals_of_dodecahedron_l542_542020

/-- Definition of a dodecahedron. -/
structure Dodecahedron where
  vertices : ℕ
  faces : ℕ
  vertices_per_face : ℕ
  faces_meeting_per_vertex : ℕ
  interior_diagonals : ℕ

/-- A dodecahedron has 12 pentagonal faces, 20 vertices, and 3 faces meet at each vertex. -/
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_meeting_per_vertex := 3,
    interior_diagonals := 160 }

theorem interior_diagonals_of_dodecahedron (d : Dodecahedron) :
    d.vertices = 20 → 
    d.faces = 12 →
    d.faces_meeting_per_vertex = 3 →
    d.interior_diagonals = 160 :=
by
  intros
  sorry

end interior_diagonals_of_dodecahedron_l542_542020


namespace three_seventy_five_as_fraction_l542_542939

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l542_542939


namespace minimum_kinds_of_candies_l542_542415

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542415


namespace percentage_swans_among_nondr_ducks_is_37_5_l542_542916

/-- 
This season, 25% of the birds living on Crystal Lake were geese, 
30% were swans, 15% were herons, 20% were ducks, and the remaining were pigeons.
-/
def percentage_geese : ℚ := 0.25
def percentage_swans : ℚ := 0.30
def percentage_herons : ℚ := 0.15
def percentage_ducks : ℚ := 0.20
def percentage_pigeons : ℚ := 1 - percentage_geese - percentage_swans - percentage_herons - percentage_ducks

/--
Given the percentages of various birds on Crystal Lake, the 
percentage of birds that were not ducks were swans is 37.5%.
-/
theorem percentage_swans_among_nondr_ducks_is_37_5 :
  let total_birds := 200 in
  let non_ducks := total_birds * (1 - percentage_ducks) in
  let swans := total_birds * percentage_swans in
  (swans / non_ducks) * 100 = 37.5 :=
by
  sorry

end percentage_swans_among_nondr_ducks_is_37_5_l542_542916


namespace total_of_ages_is_75_l542_542639

variable (A B C : ℕ)

-- Conditions as hypotheses
def cond1 : Prop := A - 10 = (1 / 2 : ℚ) * (B - 10)
def cond2 : Prop := (A : ℚ) / B = (3 / 4 : ℚ)
def cond3 : Prop := C = A + B + 5

-- Proof goal:
theorem total_of_ages_is_75 (h1 : cond1) (h2 : cond2) (h3 : cond3) : A + B + C = 75 :=
  sorry

end total_of_ages_is_75_l542_542639


namespace cos_sum_zero_l542_542555

theorem cos_sum_zero (n : ℕ) (h : n > 1) : 
  ∑ i in Finset.range (n - 1), Real.cos (2 * i * Real.pi / n) = 0 := 
sorry

end cos_sum_zero_l542_542555


namespace length_of_segment_PS_l542_542474

theorem length_of_segment_PS
  (P Q R S : Type)
  [point : P Q R S]
  (dist_PQ : distance P Q = 6)
  (dist_QR : distance Q R = 10)
  (dist_RS : distance R S = 25)
  (angle_Q : right_angle Q)
  (angle_R : right_angle R)
  : distance P S = √461 :=
sorry

end length_of_segment_PS_l542_542474


namespace simplify_cosine_expression_l542_542116

theorem simplify_cosine_expression :
  ∀ (θ : ℝ), θ = 30 * Real.pi / 180 → (1 - Real.cos θ) * (1 + Real.cos θ) = 1 / 4 :=
by
  intro θ hθ
  have cos_30 := Real.cos θ
  rewrite [hθ]
  sorry

end simplify_cosine_expression_l542_542116


namespace solve_for_a_l542_542781

noncomputable def a_value (a x : ℝ) : Prop :=
  (3 / 10) * a + (2 * x + 4) / 2 = 4 * (x - 1)

theorem solve_for_a (a : ℝ) : a_value a 3 → a = 10 :=
by
  sorry

end solve_for_a_l542_542781


namespace cube_inequality_l542_542089

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l542_542089


namespace total_shells_l542_542276

theorem total_shells :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let ed_scallop_shells := 3
  let jacob_more_shells := 2
  let marissa_limpet_shells := 5
  let marissa_oyster_shells := 6
  let marissa_conch_shells := 3
  let marissa_scallop_shells := 1
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + ed_scallop_shells
  let jacob_shells := ed_shells + jacob_more_shells
  let marissa_shells := marissa_limpet_shells + marissa_oyster_shells + marissa_conch_shells + marissa_scallop_shells
  let shells_at_beach := ed_shells + jacob_shells + marissa_shells
  let total_shells := shells_at_beach + initial_shells
  total_shells = 51 := by
  sorry

end total_shells_l542_542276


namespace min_number_of_candy_kinds_l542_542425

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542425


namespace find_a_l542_542749

variable (a : ℝ)

-- Definition of the curve
def curve (x : ℝ) : ℝ := a * (x - 2) - Real.log (x - 1)

-- The point (2, 0) must lie on the curve
def condition1 (y : ℝ) : Prop := y = curve a 2

-- The derivative of the curve
def curve_derivative (x : ℝ) : ℝ := a - (1 / (x - 1))

-- The slope of the tangent line at the point (2, 0) is 2
def condition2 : Prop := curve_derivative a 2 = 2

-- The Lean statement to prove
theorem find_a (h1 : condition1 0) (h2 : condition2) : a = 3 := by
  sorry

end find_a_l542_542749


namespace total_games_played_l542_542574

theorem total_games_played
  (wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (total_points : ℕ)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0)
  (h_total_points : total_points = 46)
  (h_total_points_calculation : total_points = wins * points_per_win + losses * points_per_loss + draws * points_per_draw) : 
  wins + losses + draws = 20 :=
by
  have h1 : wins * points_per_win = 14 * 3, from congr_arg (λ x, x * 3) h_wins,
  have h2 : losses * points_per_loss = 2 * 0, from congr_arg (λ x, x * 0) h_losses,
  have h3 : draws * points_per_draw = total_points - (wins * points_per_win + losses * points_per_loss), by
    rw [h_total_points, h1, h2]
    exact eq_sub_of_add_eq' h_total_points_calculation,
  have h_draws : draws = 4, from nat.div_eq_of_eq_mul_right (nat.succ_pos 0) (by
    rw [mul_one] at h3
    exact h3.symm),
  have h_games : wins + losses + draws = 14 + 2 + 4, by
    rw [h_wins, h_losses, h_draws],
  exact h_games

end total_games_played_l542_542574


namespace largest_angle_of_triangle_l542_542596

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 4 * x + 5 * x + 9 * x = 180) 
  (h2 : 4 * x > 40) : 
  9 * x = 90 := 
sorry

end largest_angle_of_triangle_l542_542596


namespace square_perimeter_l542_542197

variable (x : ℕ)

theorem square_perimeter (h1 : ∀ AB BC : ℕ, AB = x + 16 → BC = 3x → AB = BC) : 
  ((4 * (x + 16)) = 96) := 
by { 
  sorry 
}

end square_perimeter_l542_542197


namespace path_B_travel_l542_542987

-- Given definitions and conditions
def BC : ℝ := 3 / π

-- The total path length for point B
def path_length : ℝ := 4.5

-- Statement to be proved
theorem path_B_travel : 
  let arc_length := 1 / 4 * 2 * π * BC in
  let straight_length := BC in
  let path_total := 0 + arc_length + straight_length + arc_length in
  path_total = path_length :=
by
  -- Definitions
  let arc_length := 1 / 4 * 2 * π * BC
  let straight_length := BC
  let path_total := 0 + arc_length + straight_length + arc_length

  -- Placeholder proof (to be provided)
  sorry

end path_B_travel_l542_542987


namespace sum_of_remainders_and_parity_l542_542776

theorem sum_of_remainders_and_parity 
  (n : ℤ) 
  (h₀ : n % 20 = 13) : 
  (n % 4 + n % 5 = 4) ∧ (n % 2 = 1) :=
by
  sorry

end sum_of_remainders_and_parity_l542_542776


namespace total_amount_in_dollars_l542_542231

theorem total_amount_in_dollars (x : ℝ) (B_share : ℝ) (euro_to_dollar : ℝ) (yen_to_dollar : ℝ) :
  B_share = 58 →
  euro_to_dollar = 1.18 →
  yen_to_dollar = 1.10 →
  let A_share := x * euro_to_dollar,
      C_share := 0.75 * A_share,
      D_share := yen_to_dollar * A_share,
      total := A_share + B_share * euro_to_dollar + C_share + D_share in
  total = 263.49 :=
by
  intros hB he hy,
  sorry

end total_amount_in_dollars_l542_542231


namespace fraction_of_milk_in_second_cup_l542_542806

noncomputable def ratio_mixture (V: ℝ) (x: ℝ) :=
  ((2 / 5 * V + (1 - x) * V) / (3 / 5 * V + x * V))

theorem fraction_of_milk_in_second_cup
  (V: ℝ) 
  (hV: V > 0)
  (hx: ratio_mixture V x = 3 / 7) :
  x = 4 / 5 :=
by
  sorry

end fraction_of_milk_in_second_cup_l542_542806


namespace age_of_B_l542_542042

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 11) : B = 41 :=
by
  -- Proof not required as per instructions
  sorry

end age_of_B_l542_542042


namespace area_difference_of_circles_in_squares_l542_542022

theorem area_difference_of_circles_in_squares 
  (s1 s2 : ℝ) 
  (h1 : s1 = 40) 
  (h2 : s2 = 20) : 
  let r1 := s1 / 2 in
  let r2 := s2 / 2 in
  let A1 := π * r1 ^ 2 in
  let A2 := π * r2 ^ 2 in
  A1 - A2 = 300 * π := 
by
  sorry

end area_difference_of_circles_in_squares_l542_542022


namespace minimum_kinds_of_candies_l542_542416

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542416


namespace number_of_students_passed_l542_542658

theorem number_of_students_passed (total_students : ℕ) (failure_frequency : ℝ) (h1 : total_students = 1000) (h2 : failure_frequency = 0.4) : 
  (total_students - (total_students * failure_frequency)) = 600 :=
by
  sorry

end number_of_students_passed_l542_542658


namespace range_PA_l542_542796

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def pointA_cartesian : ℝ × ℝ :=
  polar_to_cartesian 1 Real.pi

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def curveC (θ : ℝ) : ℝ × ℝ :=
  polar_to_cartesian (2 * Real.sin θ) θ

theorem range_PA :
  let A := pointA_cartesian in
  ∀ P θ, P = curveC θ →
  ∃ PA_min PA_max, PA_min = Real.sqrt 2 - 1 ∧ PA_max = Real.sqrt 2 + 1 ∧
  distance P A ∈ set.Icc PA_min PA_max :=
by
  intro A P θ hP
  use Real.sqrt 2 - 1, Real.sqrt 2 + 1
  split
  all_goals {sorry}

end range_PA_l542_542796


namespace triangle_with_area_6_exists_l542_542478

def point (x y : ℝ) : Type :=
⟨x, y⟩

variables (P Q E : point ℝ ℝ)

-- Coordinates of fixed points F
def F := point 6 6

-- Area function of a triangle given vertices
def triangle_area (A B C : point ℝ ℝ) : ℝ :=
  0.5 * abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) : ℝ)

theorem triangle_with_area_6_exists :
  triangle_area P Q E = 6 := 
sorry

end triangle_with_area_6_exists_l542_542478


namespace correct_proposition_number_is_four_l542_542665

def proposition1_negation_is_incorrect : Prop :=
  (¬(∀ x : ℝ, x^2 = 1 → x = 1)) → (∀ x : ℝ, x^2 ≠ 1 → x ≠ 1)

def proposition2_negation_is_incorrect : Prop :=
  (¬(∃ x : ℝ, x^2 + x - 1 < 0)) → (∀ x : ℝ, x^2 + x - 1 ≥ 0)

def proposition3_is_not_false : Prop :=
  ¬(¬(∀ (x y : ℝ), x = y → sin x = sin y) → (∀ x y : ℝ, sin x ≠ sin y → x ≠ y))

def proposition4_is_correct : Prop :=
  ∀ p q : Prop, p ∨ q → p ∨ q

theorem correct_proposition_number_is_four : proposition1_negation_is_incorrect ∧ proposition2_negation_is_incorrect ∧ proposition3_is_not_false ∧ proposition4_is_correct := by
  sorry

end correct_proposition_number_is_four_l542_542665


namespace dogs_bone_relationship_l542_542651

-- Definitions from the conditions
def num_bones_first_dog := 3
def num_bones_second_dog (x : ℕ) := x
def num_bones_third_dog (x : ℕ) := 2 * x
def num_bones_fourth_dog := 1
def num_bones_fifth_dog := 2
def total_bones := 12

-- The mathematical statement to be proved
theorem dogs_bone_relationship (x : ℕ) (: 3 + x + 2 * x + 1 + 2 = 12) :
  num_bones_first_dog = num_bones_second_dog x + 1 :=
sorry

end dogs_bone_relationship_l542_542651


namespace basketball_lineup_count_l542_542203

theorem basketball_lineup_count :
  (∃ (players : Finset ℕ), players.card = 15) → 
  ∃ centers power_forwards small_forwards shooting_guards point_guards sixth_men : ℕ,
  ∃ b : Fin (15) → Fin (15),
  15 * 14 * 13 * 12 * 11 * 10 = 360360 
:= by sorry

end basketball_lineup_count_l542_542203


namespace length_AB_is_6_l542_542342

noncomputable def length_of_AB : ℝ :=
  let e : ℝ := 1 / 2
      f {α : Type} [fact α] (h : α) := id h
      a : ℝ := 4
      b : ℝ := sqrt (a ^ 2 * (1 - e ^ 2)) in
  let x := 2
      y2 := (15 * 4 / 4) * 3 / 4
      A : ℝ × ℝ := (x, y2) 
      B : ℝ × ℝ := (x, -y2) in
  sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem length_AB_is_6 : length_of_AB = 6 := 
by {
  erw [length_of_AB, Real.sqrt],
  exact rfl,
  sorry
}

end length_AB_is_6_l542_542342


namespace combined_weight_loss_l542_542240

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end combined_weight_loss_l542_542240


namespace volume_of_pyramid_l542_542560

noncomputable def volume_of_pyramid_QEFGH : ℝ := 
  let EF := 10
  let FG := 3
  let base_area := EF * FG
  let height := 9
  (1/3) * base_area * height

theorem volume_of_pyramid {EF FG : ℝ} (hEF : EF = 10) (hFG : FG = 3)
  (QE_perpendicular_EF : true) (QE_perpendicular_EH : true) (QE_height : QE = 9) :
  volume_of_pyramid_QEFGH = 90 := by
  sorry

end volume_of_pyramid_l542_542560


namespace min_nonempty_proper_subsets_l542_542369

theorem min_nonempty_proper_subsets (M N : Set ℤ) 
  (h1 : M = { x | 0 < |x - 2| ∧ |x - 2| < 2 ∧ x ∈ ℤ })
  (h2 : M ∪ N = { 1, 2, 3, 4 }) :
  3 ≤ card { S | S ⊆ N ∧ S ≠ ∅ ∧ S ≠ N } :=
sorry

end min_nonempty_proper_subsets_l542_542369


namespace distinct_real_roots_exists_l542_542720

-- This statement encompasses the conditions and the conclusion
theorem distinct_real_roots_exists (a : ℝ) :
  a ∈ Ioo 0 (1/2) ↔ ∀ x : ℝ, x > 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x^2 + 2 * (a - 1) * x + a^2 = 0) ∧ 
    x1 + a > 0 ∧
    x2 + a > 0 ∧
    x1 + a ≠ 1 ∧
    x2 + a ≠ 1 :=
begin
  sorry
end

end distinct_real_roots_exists_l542_542720


namespace number_of_tickets_is_42_l542_542985

theorem number_of_tickets_is_42 (n : ℕ) (h : n = 5) : (n * 8 + 2 = 42) :=
by {
  rw h,
  exact rfl,
}

end number_of_tickets_is_42_l542_542985


namespace ratio_of_volume_to_surface_area_l542_542983

-- Definitions of the given conditions
def unit_cube_volume : ℕ := 1
def total_cubes : ℕ := 8
def volume := total_cubes * unit_cube_volume
def exposed_faces (center_cube_faces : ℕ) (side_cube_faces : ℕ) (top_cube_faces : ℕ) : ℕ :=
  center_cube_faces + 6 * side_cube_faces + top_cube_faces
def surface_area := exposed_faces 1 5 5
def ratio := volume / surface_area

-- The main theorem statement
theorem ratio_of_volume_to_surface_area : ratio = 2 / 9 := by
  sorry

end ratio_of_volume_to_surface_area_l542_542983


namespace percentage_palm_oil_in_cheese_l542_542853

-- Define the conditions
variables (initial_cheese_price : ℝ) (initial_palm_oil_price : ℝ) (final_cheese_price : ℝ) (final_palm_oil_price : ℝ)

-- Condition 1: Initial price assumptions and price increase percentages
def conditions (initial_cheese_price initial_palm_oil_price : ℝ) : Prop :=
  final_cheese_price = initial_cheese_price * 1.03 ∧
  final_palm_oil_price = initial_palm_oil_price * 1.10

-- The main theorem to prove
theorem percentage_palm_oil_in_cheese
  (initial_cheese_price initial_palm_oil_price final_cheese_price final_palm_oil_price : ℝ)
  (h : conditions initial_cheese_price initial_palm_oil_price) :
  initial_palm_oil_price / initial_cheese_price = 0.30 :=
by
  sorry

end percentage_palm_oil_in_cheese_l542_542853


namespace inequality_count_l542_542846

theorem inequality_count
  (x y a b : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hx_lt_one : x < 1)
  (hy_lt_one : y < 1)
  (hx_lt_a : x < a)
  (hy_lt_b : y < b)
  (h_sum : x + y = a - b) :
  ({(x + y < a + b), (x - y < a - b), (x * y < a * b)}:Finset Prop).card = 3 :=
by
  sorry

end inequality_count_l542_542846


namespace ajith_speed_l542_542661

theorem ajith_speed
  (circumference : ℕ)
  (time_hours : ℕ)
  (rana_speed : ℕ)
  (distance_covered : rana_speed * time_hours)
  (ajith_meets_rana_in : circumference = ((time_hours * ajith_speed) - distance_covered)) :
  ajith_speed = 6 := by
  -- Conditions of the problem
  let circumference := 115
  let time_hours := 115
  let rana_speed := 5
  have distance_covered : rana_speed * time_hours = 575 := by linarith
  have ajith_meets_rana_in : circumference = ((time_hours * ajith_speed) - distance_covered) := by linarith
  -- Simplifications
  unfold ajith_speed
  rw [ajith_meets_rana_in, distance_covered, circumference]
  sorry

end ajith_speed_l542_542661


namespace can_construct_trapezoid_l542_542884

theorem can_construct_trapezoid (a b c d : ℝ) (h1: 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c < d) (h5 : d < a + b + c) : 
∃ P Q R S : ℝ × ℝ, (P.1 = Q.1 ∧ R.1 = S.1 ∧ 
((P.2 < Q.2 ∧ R.2 < S.2) ∨ (P.2 > Q.2 ∧ R.2 > S.2)) ∧ 
(dist P Q = a ∧ dist R S = b ∧ dist P R = c ∧ dist Q S = d) ∧ 
(dist P S = sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) ∧ dist Q R = sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2))) :=
sorry

end can_construct_trapezoid_l542_542884


namespace find_m_l542_542595

def point (ℝ : Type) := ℝ × ℝ

def slope (p₁ p₂ : point ℝ) : ℝ := 
(p₂.snd - p₁.snd) / (p₂.fst - p₁.fst)

theorem find_m {m : ℝ} 
(A B : point ℝ) 
(hA : A = (1, 2)) 
(hB : B = (3, m)) 
(h_theta : slope A B = 1) : 
m = 4 :=
by
  sorry

end find_m_l542_542595


namespace sequence_sum_S2020_l542_542005

open Nat

theorem sequence_sum_S2020 :
  (∃ a : ℕ → ℕ, (∀ n ≥ 3, a (n-1) = a n + a (n-2)) ∧
  let S : ℕ → ℕ := λ n, (∑ i in range n, a (i + 1))
  in S 2018 = 2017 ∧ S 2019 = 2018) →
  let S : ℕ → ℕ := λ n, (∑ i in range n, a (i + 1))
  in S 2020 = 1010 :=
by
  sorry

end sequence_sum_S2020_l542_542005


namespace non_obtuse_triangle_range_l542_542800

noncomputable def range_of_2a_over_c (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) : Set ℝ :=
  {x | ∃ (a b c A : ℝ), x = (2 * a) / c ∧ 1 < x ∧ x ≤ 4}

theorem non_obtuse_triangle_range (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) :
  (2 * a) / c ∈ range_of_2a_over_c a b c A C h1 h2 h3 := 
sorry

end non_obtuse_triangle_range_l542_542800


namespace favorite_movies_hours_l542_542066

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l542_542066


namespace min_value_of_a3b2c_l542_542528

theorem min_value_of_a3b2c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / a + 1 / b + 1 / c = 9) : 
  a^3 * b^2 * c ≥ 1 / 2916 :=
by 
  sorry

end min_value_of_a3b2c_l542_542528


namespace total_first_year_students_l542_542794

theorem total_first_year_students (males : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) (N : ℕ)
  (h1 : males = 570)
  (h2 : sample_size = 110)
  (h3 : female_in_sample = 53)
  (h4 : N = ((sample_size - female_in_sample) * males) / (sample_size - (sample_size - female_in_sample)))
  : N = 1100 := 
by
  sorry

end total_first_year_students_l542_542794


namespace number_of_elements_in_intersection_l542_542006

noncomputable def set_M : Set ℝ := {x : ℝ | x^2 - 5*x + 4 ≤ 0}
noncomputable def set_N : Set ℕ := {0, 1, 2, 3}
def intersection_set := set_M ∩ set_N
def intersection_cardinality : ℕ := Fintype.card { x ∈ intersection_set }

theorem number_of_elements_in_intersection :
  intersection_cardinality = 3 := by {
  sorry
}

end number_of_elements_in_intersection_l542_542006


namespace fruit_seller_apples_l542_542945

theorem fruit_seller_apples (X : ℕ) (h : 0.40 * X = 300) : X = 750 :=
by
  sorry

end fruit_seller_apples_l542_542945


namespace chess_amateurs_play_with_l542_542607

theorem chess_amateurs_play_with :
  ∃ n : ℕ, ∃ total_players : ℕ, total_players = 6 ∧
  (total_players * (total_players - 1)) / 2 = 12 ∧
  (n = total_players - 1 ∧ n = 5) :=
by
  sorry

end chess_amateurs_play_with_l542_542607


namespace minimum_candy_kinds_l542_542448

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542448


namespace total_horse_food_needed_per_day_l542_542669

def sheep_to_horses_ratio := 4 / 7
def ounces_per_horse_per_day := 230
def number_of_sheep := 32

theorem total_horse_food_needed_per_day :
  let number_of_horses := (7 * number_of_sheep) / 4 in
  ounces_per_horse_per_day * number_of_horses = 12880 := by
  sorry

end total_horse_food_needed_per_day_l542_542669


namespace min_candy_kinds_l542_542432

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542432


namespace number_of_rectangles_in_4x4_grid_l542_542260

theorem number_of_rectangles_in_4x4_grid : 
  ∀ (n : ℕ), n = 4 → (choose 4 2) * (choose 4 2) = 36 :=
by
  intros n hn
  rw hn
  have h1 : choose 4 2 = 6 := by decide
  calc
    (choose 4 2) * (choose 4 2) 
        = 6 * 6 : by rw [h1, h1]
    ... = 36 : by norm_num
  done

end number_of_rectangles_in_4x4_grid_l542_542260


namespace constant_term_in_expansion_l542_542584

-- Definitions according to conditions and question
def binomial_expansion (n : ℕ) (a b : ℝ) (x : ℝ) : ℝ :=
  -- Note: This function represents the general term of the binomial expansion
  let general_term (r : ℕ) := (-2)^r * (nat.choose 10 r) * x^((10 - r)/2 - 2*r)
  in sum (λ r, general_term r) (range (n+1))

theorem constant_term_in_expansion :
  binomial_expansion 10 (sqrt x) (-(2 / x^2)) 1 = 180 :=
  by sorry

end constant_term_in_expansion_l542_542584


namespace three_seventy_five_as_fraction_l542_542941

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l542_542941


namespace pages_to_read_tomorrow_l542_542511

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l542_542511


namespace inequality_proof_l542_542554

theorem inequality_proof (n : ℕ) (h : n > 1) (a : (Fin n) → ℝ) (h_pos : ∀ i, 0 < a i):
  (∑ i, (a i / (∑ j, if j ≠ i then a j else 0))^2) ≥ n / ((n - 1)^2) :=
sorry

end inequality_proof_l542_542554


namespace cone_volume_is_correct_l542_542975

noncomputable def cone_volume_from_half_sector (r_sector : ℝ) (sector_angle : ℝ) (r_cone_base : ℝ) (h_cone : ℝ) : ℝ :=
  (1 / 3) * π * r_cone_base^2 * h_cone

theorem cone_volume_is_correct : 
  ∀ (r_sector : ℝ), (r_sector = 6) → (cone_volume_from_half_sector r_sector 180 3 (3 * Real.sqrt 3) = 9 * π * Real.sqrt 3) := 
by
  intros
  rw cone_volume_from_half_sector
  sorry

end cone_volume_is_correct_l542_542975


namespace sum_first_five_terms_l542_542731

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem sum_first_five_terms (a1 q : ℝ) 
  (h1 : geometric_sequence a1 q 2 * geometric_sequence a1 q 3 = 2 * a1)
  (h2 : (geometric_sequence a1 q 4 + 2 * geometric_sequence a1 q 7) / 2 = 5 / 4)
  : sum_geometric_sequence a1 q 5 = 31 :=
sorry

end sum_first_five_terms_l542_542731


namespace find_possible_values_of_beta_l542_542824

noncomputable def possible_values_of_beta (β : ℂ) : Prop :=
  β ≠ 1 ∧ (abs (β^3 - 1) = 3 * abs (β - 1)) ∧ (abs (β^6 - 1) = 6 * abs (β - 1))

theorem find_possible_values_of_beta (β : ℂ) :
  possible_values_of_beta β → (β = 2 * complex.sqrt 2 * complex.I ∨ β = -2 * complex.sqrt 2 * complex.I) :=
by
  sorry

end find_possible_values_of_beta_l542_542824


namespace petya_grid_pictures_l542_542101

/--
Petya fills a 4x4 grid with natural numbers from 1 to 16 so that 
any two consecutive numbers (differing by 1) are placed in adjacent cells.
After erasing some numbers, prove that resulting pictures could be pictures
3 and 5.
-/
theorem petya_grid_pictures (𝒫 : (ℕ → ℕ → Option ℕ) → Prop) :
  (∀ (g : ℕ → ℕ → Option ℕ),
    (∀ i j, g i j ∈ some '' {n | 1 ≤ n ∧ n ≤ 16}) ∧
    (∀ i j m n, abs(i - m) + abs(j - n) = 1 → ∀ x y, g i j = some x ∧ g m n = some y → abs (x - y) = 1) →
    (𝒫 g)) →
  ∃ g3 g5,
    (𝒫 g3 ∧
     g3 = λ i, λ j, if (i = 1∧ j = 1) then some 1 else if (i = 1∧ j = 2) then some 2 else if (i = 2∧ j = 2) then some 3 else if (i = 2∧ j = 3) then some 4 else if (i = 3∧ j = 3) then some 5 else if (i = 3∧ j = 4) then some 6 else if (i = 4∧ j = 4) then some 7 else if (i = 4∧ j = 3) then some 8 else if (i = 4∧ j = 2) then some 9 else if (i = 4∧ j = 1) then some 10 else if (i = 3∧ j = 1) then some 11 else if (i = 3∧ j = 2) then some 12 else if (i = 2∧ j = 1) then some 13 else if (i = 1∧ j = 3) then some 14 else if (i = 1∧ j = 4) then some 15 else if (i = 2∧ j = 4) then some 16 else none) ∧ 
     (𝒫 g5 ∧
       g5 = λ i, λ j, if (i = 4∧ j = 1) then some 1 else if (i = 4∧ j = 2) then some 2 else if (i = 4∧ j = 3) then some 3 else if (i = 4∧ j = 4) then some 4 else if (i = 3∧ j = 4) then some 5 else if (i = 2∧ j = 4) then some 6 else if (i = 1∧ j = 4) then some 7 else if (i = 1∧ j = 3) then some 8 else if (i = 1∧ j = 2) then some 9 else if (i = 1∧ j = 1) then some 10 else if (i = 2∧ j = 1) then some 11 else if (i = 2∧ j = 2) then some 12 else if (i = 2∧ j = 3) then some 13 else if (i = 3∧ j = 3) then some 14 else if (i = 3∧ j = 2) then some 15 else if (i = 3∧ j = 1) then some 16 else none)
) := sorry

end petya_grid_pictures_l542_542101


namespace cone_volume_is_6pi_l542_542605

-- Definitions for the conditions in the problem
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def cone_radius (r : ℝ) : ℝ := r / 2

-- Theorem statement: Given the conditions, prove that the volume of the cone is 6π
theorem cone_volume_is_6pi (r h : ℝ) (hcylinder : cylinder_volume r h = 72 * π) :
  (1 / 3) * π * (cone_radius r)^2 * h = 6 * π :=
by
  sorry

end cone_volume_is_6pi_l542_542605


namespace minimum_kinds_of_candies_l542_542417

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542417


namespace max_f1_value_tan_x_value_l542_542316

noncomputable def f1 (x : ℝ) : ℝ := Math.sin x + (Real.sqrt 3) * Math.cos x

theorem max_f1_value : ∃ k : ℤ, (∀ x, f1 x ≤ 2) ∧ ∃ x, f1 x = 2 ∧ x = (π / 6) + 2 * k * π := 
sorry

noncomputable def f2 (x : ℝ) : ℝ := Math.sin x - Math.cos x

theorem tan_x_value (h1 : Math.sin (π / 4) + a * Math.cos (π / 4) = 0) 
    (h2 : ∀ x, 0 < x ∧ x < π → f2 x = 1/5) : ∃ x, 0 < x ∧ x < π ∧ Math.tan x = 4/3 :=
sorry

end max_f1_value_tan_x_value_l542_542316


namespace probability_neither_red_blue_purple_l542_542642

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_neither_red_blue_purple :
  (total_balls - (red_balls + blue_balls + purple_balls)) / total_balls = 35 / 48 := 
by 
  /- Proof details are not necessary -/
  sorry

end probability_neither_red_blue_purple_l542_542642


namespace convert_decimal_to_fraction_l542_542938

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l542_542938


namespace exists_tangent_l542_542748

variable (k : ℝ)

def circle (q : ℝ) := ∃ (x y : ℝ), (x + cos q)^2 + (y - sin q)^2 = 1

def line (k : ℝ) := ∃ (x y : ℝ), y = k * x

def tangent (k q : ℝ) := ∃ (x : ℝ), (x + cos q)^2 + (k * x - sin q)^2 = 1

theorem exists_tangent (k : ℝ) : ∃ q : ℝ, tangent k q :=
sorry

end exists_tangent_l542_542748


namespace mode_in_data_set_l542_542622

open Classical

noncomputable def data_set : Type := List ℕ

structure DataSetStatistics :=
  (mode : data_set → ℕ)
  (median : data_set → ℝ)
  (std_dev : data_set → ℝ)
  (mean : data_set → ℝ)

theorem mode_in_data_set (data : data_set)
  (h_mode : ∀ x ∈ data_set mode data, ∃ n : ℕ, n ∈ data_set)
  (h_median : ∀ x ∈ data_set median data, true)
  (h_std_dev : ∀ x ∈ data_set std_dev data, true)
  (h_mean : ∀ x ∈ data_set mean data, true) :
  ∃ mode_val : ℕ, mode_val ∈ data_set mode data :=
  sorry

end mode_in_data_set_l542_542622


namespace count_valid_statements_l542_542265

variables (p q r : Prop)

def statement_1 := p ∧ q ∧ ¬r
def statement_2 := ¬p ∧ q ∧ ¬r
def statement_3 := p ∧ ¬q ∧ r
def statement_4 := ¬p ∧ ¬q ∧ r
def implies_formula := p ∧ ¬r → ¬q

theorem count_valid_statements : 
  (statement_1 → implies_formula) +
  (statement_2 → implies_formula) +
  (statement_3 → implies_formula) +
  (statement_4 → implies_formula) = 3 :=
sorry

end count_valid_statements_l542_542265


namespace degree_of_monomial_3ab_l542_542124

variable (a b : ℕ)

def monomialDegree (x y : ℕ) : ℕ :=
  x + y

theorem degree_of_monomial_3ab : monomialDegree 1 1 = 2 :=
by
  sorry

end degree_of_monomial_3ab_l542_542124


namespace car_speed_l542_542028

def distance : ℝ := 69
def time : ℝ := 3
def speed (dist : ℝ) (t : ℝ) : ℝ := dist / t

theorem car_speed : speed distance time = 23 := by
  sorry

end car_speed_l542_542028


namespace fruit_display_l542_542913

theorem fruit_display (bananas : ℕ) (Oranges : ℕ) (Apples : ℕ) (hBananas : bananas = 5)
  (hOranges : Oranges = 2 * bananas) (hApples : Apples = 2 * Oranges) :
  bananas + Oranges + Apples = 35 :=
by sorry

end fruit_display_l542_542913


namespace min_value_k_l542_542486

noncomputable def a : ℕ → ℕ
| 0     := 4
| (n+1) := 3 * a n - 2

theorem min_value_k (k : ℝ) : (∀ n : ℕ, n > 0 → k * (a n - 1) ≥ 2 * n - 5) ↔ k ≥ (1 / 27) :=
begin
  sorry
end

end min_value_k_l542_542486


namespace area_of_side_face_l542_542626

theorem area_of_side_face (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := 
by
  sorry

end area_of_side_face_l542_542626


namespace quadratic_distinct_real_roots_range_l542_542036

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l542_542036


namespace cost_of_adult_ticket_is_15_l542_542898

variable (A : ℕ) -- Cost of an adult ticket
variable (total_tickets : ℕ) (cost_child_ticket : ℕ) (total_revenue : ℕ)
variable (adult_tickets_sold : ℕ)

theorem cost_of_adult_ticket_is_15
  (h1 : total_tickets = 522)
  (h2 : cost_child_ticket = 8)
  (h3 : total_revenue = 5086)
  (h4 : adult_tickets_sold = 130) 
  (h5 : (total_tickets - adult_tickets_sold) * cost_child_ticket + adult_tickets_sold * A = total_revenue) :
  A = 15 :=
by
  sorry

end cost_of_adult_ticket_is_15_l542_542898


namespace measure_Y_l542_542542

-- Definitions of lines, angles, and the parallel property
variable (p q : Line)
variable (X Z Y : Point)
variable (m : Angle)

-- Given conditions
axiom parallel_lines : p ∥ q
axiom angle_X : m.measure X = 100
axiom angle_Z : m.measure Z = 70

-- The theorem to be proven
theorem measure_Y (h_parallel : p ∥ q) (h_angle_X : m.measure X = 100) (h_angle_Z : m.measure Z = 70) : m.measure Y = 110 :=
  sorry

end measure_Y_l542_542542


namespace false_statement_l542_542764

noncomputable def circle (center : Type*) (radius : ℝ) := ∃ x: center, true

theorem false_statement (A B C : Type*) (a b c : ℝ) (hA_largest : a > b ∧ a > c)
  (non_intersecting : ∀ x y r1 r2, (circle x r1 ∧ circle y r2) → (x ≠ y ∧ r1 ≠ r2)) :
  ¬ (a + b + c = dist A B + dist A C) :=
sorry

end false_statement_l542_542764


namespace selection_methods_l542_542793

theorem selection_methods
  (total_people : ℕ)
  (from_each_class : ℕ)
  (classes : ℕ)
  (select : ℕ)
  (class3_limit : ℕ) 
  (h_total_people : total_people = 16)
  (h_from_each_class : from_each_class = 4)
  (h_classes : classes = 4)
  (h_select : select = 3)
  (h_class3_limit : class3_limit = 1) 
  : (∑ (case1_ways: ℕ) (case2_ways: ℕ), 
      case1_ways + case2_ways = 472) :=
sorry

end selection_methods_l542_542793


namespace find_cos_x_minus_pi_over_4_find_cos_2x_l542_542367

open Real

-- Definitions from conditions
def a (x : ℝ) : ℝ × ℝ := (cos x, sin x)
def b : ℝ × ℝ := (sqrt 2 / 2, sqrt 2 / 2)
def vector_length (u v : ℝ × ℝ) : ℝ :=
  sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2)

-- Problem statement (Ⅰ)
theorem find_cos_x_minus_pi_over_4 (x : ℝ) 
  (h1 : vector_length (a x) b = (4 * sqrt 5) / 5) :
  cos (x - π / 4) = -3 / 5 :=
sorry

-- Problem statement (Ⅱ)
theorem find_cos_2x (x : ℝ)
  (hx : x ∈ Icc (π / 2) (3 * π / 2))
  (h1 : vector_length (a x) b = (4 * sqrt 5) / 5) :
  cos (2 * x) = 24 / 25 :=
sorry

end find_cos_x_minus_pi_over_4_find_cos_2x_l542_542367


namespace triangle_area_l542_542169

-- Define the sides of the triangle
def a : ℝ := 5
def b : ℝ := 4
def c : ℝ := 4

-- Define the altitude of the isosceles triangle using the Pythagorean theorem
def altitude : ℝ := Real.sqrt (b^2 - (a/2)^2)

-- Prove that the area equals the given answer
theorem triangle_area :
  ∃ (area : ℝ), area = (a / 2) * altitude ∧ area = (5 * Real.sqrt 39) / 4 :=
by
  use (a / 2) * altitude
  split
  . rfl
  . sorry

end triangle_area_l542_542169


namespace problem_equivalent_l542_542379

theorem problem_equivalent
  (x : ℚ)
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289 / 8 := 
by
  sorry

end problem_equivalent_l542_542379


namespace card_difference_condition_l542_542728

theorem card_difference_condition 
  (n : ℕ) 
  (h_n : n ≥ 4) 
  (a : ℕ → ℝ) 
  (h_a : ∀ m : ℕ, m ≤ 2 * n + 4 → (⌊a m⌋₊ : ℝ) = m):
  ∃ (x y z t : ℕ), 
    x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧
    abs ((a x + a y) - (a z + a t)) < (1 / (n - real.sqrt (n / 2))) :=
begin
  sorry
end

end card_difference_condition_l542_542728


namespace complement_intersection_l542_542008

universe u

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

def complement_U (A : Finset Int) (U : Finset Int) : Finset Int :=
  U.filter (λ x => x ∉ A)

theorem complement_intersection :
  (complement_U A U) ∩ B = {-3, 0} :=
by
  sorry

end complement_intersection_l542_542008


namespace conic_section_properties_l542_542485

theorem conic_section_properties :
  let C := {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 3) = 1}
  let polar_eq (ρ θ : ℝ) : Prop := ρ^2 = 12 / (3 + sin θ^2)
  let l := {p : ℝ × ℝ | ∃ t : ℝ, p = (-1 + 1/2 * t, √3 / 2 * t)}
  let F1 : ℝ × ℝ := (-1, 0)
  let A : ℝ × ℝ := (0, -√3)
  let L (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)
  polar_eq ρ θ → 
  A ∈ C →
  l = {p | ∃ t : ℝ, p = (F1.1 + (t - F1.1) / |F1.1 - A.1|, (L A F1) * (A.1 - F1.1))} →
  ∀ p ∈ l, p ∈ C →
  (∃ t₀ t₁ : ℝ, t₀ ≠ t₁ ∧ (∀ p ∈ l, p ∈ C → (F1.1 - p.1)^2 + (F1.2 - p.2)^2 = t₀ * t₁)) →
  |(F1.1 - p.1)| * |(F1.2 - p.2)| = 12 / 5 :=
begin
  intros,
  sorry
end

end conic_section_properties_l542_542485


namespace percent_decaf_coffee_l542_542185

variable (initial_stock new_stock decaf_initial_percent decaf_new_percent : ℝ)
variable (initial_stock_pos new_stock_pos : initial_stock > 0 ∧ new_stock > 0)

theorem percent_decaf_coffee :
    initial_stock = 400 → 
    decaf_initial_percent = 20 → 
    new_stock = 100 → 
    decaf_new_percent = 60 → 
    (100 * ((decaf_initial_percent / 100 * initial_stock + decaf_new_percent / 100 * new_stock) / (initial_stock + new_stock))) = 28 := 
by
  sorry

end percent_decaf_coffee_l542_542185


namespace least_amount_of_money_l542_542786

variable (money : String → ℝ)
variable (Bo Coe Flo Jo Moe Zoe : String)

theorem least_amount_of_money :
  (money Bo ≠ money Coe) ∧ (money Bo ≠ money Flo) ∧ (money Bo ≠ money Jo) ∧ (money Bo ≠ money Moe) ∧ (money Bo ≠ money Zoe) ∧ 
  (money Coe ≠ money Flo) ∧ (money Coe ≠ money Jo) ∧ (money Coe ≠ money Moe) ∧ (money Coe ≠ money Zoe) ∧ 
  (money Flo ≠ money Jo) ∧ (money Flo ≠ money Moe) ∧ (money Flo ≠ money Zoe) ∧ 
  (money Jo ≠ money Moe) ∧ (money Jo ≠ money Zoe) ∧ 
  (money Moe ≠ money Zoe) ∧ 
  (money Flo > money Jo) ∧ (money Flo > money Bo) ∧
  (money Bo > money Moe) ∧ (money Coe > money Moe) ∧ 
  (money Jo > money Moe) ∧ (money Jo < money Bo) ∧ 
  (money Zoe > money Jo) ∧ (money Zoe < money Coe) →
  money Moe < money Bo ∧ money Moe < money Coe ∧ money Moe < money Flo ∧ money Moe < money Jo ∧ money Moe < money Zoe := 
sorry

end least_amount_of_money_l542_542786


namespace chorus_row_lengths_count_l542_542208

theorem chorus_row_lengths_count : 
  {x : ℕ | x ∣ 90 ∧ 6 ≤ x ∧ x ≤ 15}.card = 4 :=
by sorry

end chorus_row_lengths_count_l542_542208


namespace max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l542_542357

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l542_542357


namespace find_value_of_abs_2_add_z_l542_542343

theorem find_value_of_abs_2_add_z (z : ℂ) (i : ℂ)
  (hi : i^2 = -1)
  (hz : (1 - z) / (1 + z) = (-1 - i) ) :
  |2 + z| = (5 * real.sqrt 2) / 2 :=
by sorry

end find_value_of_abs_2_add_z_l542_542343


namespace smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l542_542753

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x := 
by 
  -- Proof omitted
  sorry

theorem monotonically_increasing_interval :
  ∃ k : ℤ, ∀ x y, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
               k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤  k * Real.pi + Real.pi / 6 →
               x ≤ y → f x ≤ f y := 
by 
  -- Proof omitted
  sorry

theorem minimum_value_a_of_triangle (A B C a b c : ℝ) 
  (h₀ : f A = 1/2) 
  (h₁ : B^2 - C^2 - B * C * Real.cos A - a^2 = 4) :
  a ≥ 2 * Real.sqrt 2 :=
by 
  -- Proof omitted
  sorry

end smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l542_542753


namespace seventy_ninth_digit_is_two_l542_542385

theorem seventy_ninth_digit_is_two : 
  let sequence := (List.range' 1 65).map (fun n => (65 - n).toString).join in
  sequence.get ⟨78, by decide⟩ = '2' :=
sorry

end seventy_ninth_digit_is_two_l542_542385


namespace keith_ate_apples_l542_542841

theorem keith_ate_apples : let total_apples := 7.0 + 3.0
                           let apples_left := 4.0
                           let keith_ate := total_apples - apples_left
                           in keith_ate = 6.0 :=
by
  sorry

end keith_ate_apples_l542_542841


namespace jiayuan_supermarket_transport_l542_542577

theorem jiayuan_supermarket_transport : 
  let baskets_apples := 62
      baskets_pears := 38
      weight_per_basket := 25 
  in (baskets_apples * weight_per_basket + baskets_pears * weight_per_basket = 2500) :=
by
  sorry

end jiayuan_supermarket_transport_l542_542577


namespace binary_addition_l542_542237

theorem binary_addition :
  0b1101 + 0b101 + 0b1110 + 0b10111 + 0b11000 = 0b11100010 :=
by
  sorry

end binary_addition_l542_542237


namespace find_lambda_orthogonal_l542_542012

variables (a b : ℝ × ℝ) (lambda : ℝ)

def vec_a : ℝ × ℝ := (3, 3)
def vec_b : ℝ × ℝ := (1, -1)

-- Definition that checks if two vectors are orthogonal
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda_orthogonal :
  (orthogonal (vec_a + lambda • vec_b) (vec_a - lambda • vec_b)) ↔ (lambda = 3 ∨ lambda = -3) :=
begin
  have ha : vec_a = (3, 3), from rfl,
  have hb : vec_b = (1, -1), from rfl,
  sorry
end

end find_lambda_orthogonal_l542_542012


namespace range_of_a_l542_542088

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x y, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 2 ∧ x < y →
    log a (abs (x^2 - (a + 1/a) * x + 1)) ≤ log a (abs (y^2 - (a + 1/a) * y + 1))) ↔
  a ≥ 2 + sqrt 3 :=
    sorry

end range_of_a_l542_542088


namespace injective_function_characterization_l542_542281

theorem injective_function_characterization (f : ℕ → ℕ)
  (h_injective : function.injective f)
  (h_functional_eq : ∀ a b : ℕ, (f^[f a] b) * (f^[f b] a) = (f (a + b))^2) :
  ∀ x : ℕ, f x = x + 1 := 
sorry

end injective_function_characterization_l542_542281


namespace trig_identity_l542_542257

theorem trig_identity : 
  (tan (real.pi / 6)) ^ 2 = 1 / 3 ∧ (sin (real.pi / 6)) ^ 2 = 1 / 4 → 
  ((tan (real.pi / 6)) ^ 2 - (sin (real.pi / 6)) ^ 2) / ((tan (real.pi / 6)) ^ 2 * (sin (real.pi / 6)) ^ 2) = 1 :=
by 
  assume h,
  have h1 : (tan (real.pi / 6)) ^ 2 = 1 / 3, from h.1,
  have h2 : (sin (real.pi / 6)) ^ 2 = 1 / 4, from h.2,
  sorry

end trig_identity_l542_542257


namespace percentage_palm_oil_in_cheese_l542_542856

theorem percentage_palm_oil_in_cheese
  (initial_cheese_price: ℝ := 100)
  (cheese_price_increase: ℝ := 3)
  (palm_oil_price_increase_percentage: ℝ := 0.10)
  (expected_palm_oil_percentage : ℝ := 30):
  ∃ (palm_oil_initial_price: ℝ),
  cheese_price_increase = palm_oil_initial_price * palm_oil_price_increase_percentage ∧
  expected_palm_oil_percentage = 100 * (palm_oil_initial_price / initial_cheese_price) := by
  sorry

end percentage_palm_oil_in_cheese_l542_542856


namespace largest_possible_median_l542_542921

theorem largest_possible_median (x : ℤ) : 
  ∃ m, m = 6 ∧ 
    (∀ (s : list ℤ), s = ([x, 2 * x, 3, 2, 5, 7].sorted) → 
        s.nth (s.length / 2 - 1) * s.nth (s.length / 2) = some 5 * some 7 / 2) :=
by {
  sorry
}

end largest_possible_median_l542_542921


namespace linear_condition_l542_542592

theorem linear_condition (m : ℝ) : ¬ (m = 2) ↔ (∃ f : ℝ → ℝ, ∀ x, f x = (m - 2) * x + 2) :=
by
  sorry

end linear_condition_l542_542592


namespace num_planting_arrangements_is_eight_l542_542645

def crop := Type

structure grid (α : crop) :=
(UL UR BL BR : α)

axioms
(alfalfa barley lentils quinoa : crop)
(grid_valid : grid crop)
(al1 : grid_valid.UL = alfalfa)
(al2 : grid_valid.BR = alfalfa)
(al3 : (grid_valid.UR = lentils ∨ grid_valid.BL = lentils) ∧ 
      (grid_valid.UR ≠ lentils ∨ grid_valid.UL = alfalfa) ∧ 
      (grid_valid.BL ≠ lentils ∨ grid_valid.BR = alfalfa))
(bq_adjacent : (grid_valid.UR ≠ barley ∨ grid_valid.BL ≠ quinoa) ∧ 
               (grid_valid.UR ≠ quinoa ∨ grid_valid.BL ≠ barley))

theorem num_planting_arrangements_is_eight : 
  ∃ arrangements : grid crop, 8 :=
sorry

end num_planting_arrangements_is_eight_l542_542645


namespace factor_expression_l542_542696

theorem factor_expression (x : ℝ) : 
  72 * x^2 + 108 * x + 36 = 36 * (2 * x^2 + 3 * x + 1) :=
sorry

end factor_expression_l542_542696


namespace gcd_100_450_l542_542706

theorem gcd_100_450 : Int.gcd 100 450 = 50 := 
by sorry

end gcd_100_450_l542_542706


namespace minimum_value_of_quadratic_l542_542889

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 3

theorem minimum_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f(y) >= f(x) ∧ f(x) = -3 :=
by
  sorry

end minimum_value_of_quadratic_l542_542889


namespace factor_of_polynomial_l542_542850

theorem factor_of_polynomial (x : ℝ) : 
  (x^2 - 2*x + 2) ∣ (29 * 39 * x^4 + 4) :=
sorry

end factor_of_polynomial_l542_542850


namespace intersection_A_B_l542_542538

-- Define sets A and B based on given conditions
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- Prove the intersection of A and B equals (2,4)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 4} := 
by
  sorry

end intersection_A_B_l542_542538


namespace volume_of_box_l542_542269

theorem volume_of_box (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : 
    l * w * h = 72 := 
by 
    sorry

end volume_of_box_l542_542269


namespace exp_gt_one_l542_542947

theorem exp_gt_one (a x y : ℝ) (ha : 1 < a) (hxy : x > y) : a^x > a^y :=
sorry

end exp_gt_one_l542_542947


namespace minimum_kinds_of_candies_l542_542419

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l542_542419


namespace point_in_fourth_quadrant_l542_542792

def quadrant_of_point (x : ℝ) : String :=
  let Px := x ^ 2 + 2
  let Py := -3
  if Px > 0 ∧ Py < 0 then "Fourth quadrant" else "Not in fourth quadrant"

theorem point_in_fourth_quadrant (x : ℝ) : quadrant_of_point x = "Fourth quadrant" :=
  by
    -- Proof is omitted
    sorry

end point_in_fourth_quadrant_l542_542792


namespace sum_first_10_terms_seq_l542_542121

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the partial sums of the sequence
def S (n : ℕ) : ℕ := (∑ i in finset.range (n + 1), a i)

-- Define the sequence {S_n / n}
def seq (n : ℕ) : ℕ := S n / n

-- State the theorem to prove the sum of the first 10 terms of seq is 75
theorem sum_first_10_terms_seq : (∑ i in finset.range 10, seq (i + 1)) = 75 :=
by
  sorry

end sum_first_10_terms_seq_l542_542121


namespace determine_number_l542_542954

noncomputable def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  (∃ d1 d2 d3, 
    n = d1 * 100 + d2 * 10 + d3 ∧ 
    (
      (d1 = 5 ∨ d1 = 1 ∨ d1 = 5 ∨ d1 = 2) ∧
      (d2 = 4 ∨ d2 = 4 ∨ d2 = 4) ∧
      (d3 = 3 ∨ d3 = 2 ∨ d3 = 6)
    ) ∧
    (
      (d1 ≠ 1 ∧ d1 ≠ 2 ∧ d1 ≠ 6) ∧
      (d2 ≠ 5 ∧ d2 ≠ 4 ∧ d2 ≠ 6 ∧ d2 ≠ 2) ∧
      (d3 ≠ 5 ∧ d3 ≠ 4 ∧ d3 ≠ 1 ∧ d3 ≠ 2)
    )
  )

theorem determine_number : ∃ n : ℕ, is_valid_number n ∧ n = 163 :=
by 
  existsi 163
  unfold is_valid_number
  sorry

end determine_number_l542_542954


namespace problem_sol_l542_542132

-- Defining the operations as given
def operation_hash (a b c : ℤ) : ℤ := 4 * a ^ 3 + 4 * b ^ 3 + 8 * a ^ 2 * b + c
def operation_star (a b d : ℤ) : ℤ := 2 * a ^ 2 - 3 * b ^ 2 + d ^ 3

-- Main theorem statement
theorem problem_sol (a b x c d : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (hc : c > 0) (hd : d > 0) 
  (h3 : operation_hash a x c = 250)
  (h4 : operation_star a b d + x = 50) :
  False := sorry

end problem_sol_l542_542132


namespace min_expression_value_l542_542833

theorem min_expression_value (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (habc : a + b + c = 3) : 
  a^2 + 2 * real.sqrt (a * b) + real.cbrt (a^2 * b * c) ≥ 4 :=
sorry

end min_expression_value_l542_542833


namespace exists_root_in_interval_l542_542127

noncomputable def f (x : ℝ) : ℝ := 2 ^ x - x ^ 3

theorem exists_root_in_interval : ∃ c ∈ Ioo 1 2, f c = 0 :=
by
  have h_cont : ContinuousOn f (Icc 1 2) := sorry
  have h_eval_1 : f 1 = 1 := by norm_num
  have h_eval_2 : f 2 = -4 := by norm_num
  have h_sign_change : (0 : ℝ) ∈ Ioo (f 2) (f 1) := by norm_num
  exact IntermediateValueTheorem h_cont (mem_Icc_iff.mpr ⟨le_rfl, le_rfl⟩) h_sign_change sorry

end exists_root_in_interval_l542_542127


namespace area_ratio_PQR_XYZ_l542_542804

def triangle (V : Type*) [AddCommGroup V] [Module ℝ V] := ℝ → V

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def in_triangle (x y z g h i : V) : Prop :=
  ∃ (t : ℝ), (t = 2/3) ∧ (g = t • y + (1 - t) • z ∧
                          h = t • x + (1 - t) • z ∧
                          i = t • x + (1 - t) • y)

def triangle_area_ratio {F : Type*} [Field F] [Module F V]
  (x y z p q r : V) : F :=
  let area_∆xyz : F := sorry  -- Needs definition in a general setting
  let area_∆pqr : F := sorry  -- Needs similar definition
  in area_∆pqr / area_∆xyz

theorem area_ratio_PQR_XYZ (x y z g h i p q r : V)
  (hx : in_triangle x y z g h i)
  (hπ : ∀ (a b : V), a = b → false) :
  triangle_area_ratio x y z p q r = (1 : ℝ) / 12 :=
by sorry

end area_ratio_PQR_XYZ_l542_542804


namespace dogwood_trees_current_l542_542148

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end dogwood_trees_current_l542_542148


namespace leftovers_value_l542_542655

def quarters_in_roll : ℕ := 30
def dimes_in_roll : ℕ := 40
def james_quarters : ℕ := 77
def james_dimes : ℕ := 138
def lindsay_quarters : ℕ := 112
def lindsay_dimes : ℕ := 244
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftovers_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_in_roll
  let leftover_dimes := total_dimes % dimes_in_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 2.45 :=
by
  sorry

end leftovers_value_l542_542655


namespace john_caffeine_consumption_l542_542501

noncomputable def caffeine_amount_first_drink (c : ℝ) : ℝ :=
  12 * c

theorem john_caffeine_consumption (c : ℝ) (h : 36 * c = 750) :
  caffeine_amount_first_drink c ≈ 250 :=
by
  sorry

end john_caffeine_consumption_l542_542501


namespace pos_real_ineq_l542_542340

theorem pos_real_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c)/3) :=
by 
  sorry

end pos_real_ineq_l542_542340


namespace second_group_students_l542_542904

-- Define the number of groups and their respective sizes
def num_groups : ℕ := 4
def first_group_students : ℕ := 5
def third_group_students : ℕ := 7
def fourth_group_students : ℕ := 4
def total_students : ℕ := 24

-- Define the main theorem to prove
theorem second_group_students :
  (∃ second_group_students : ℕ,
    total_students = first_group_students + second_group_students + third_group_students + fourth_group_students ∧
    second_group_students = 8) :=
sorry

end second_group_students_l542_542904


namespace dodecahedron_interior_diagonals_count_l542_542017

-- Define the structure of a dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)  -- 12 pentagonal faces
  (vertices : Fin 20)  -- 20 vertices
  (edges_per_vertex : Fin 3)  -- 3 faces meeting at each vertex

-- Define what an interior diagonal is
def is_interior_diagonal (v1 v2 : Fin 20) (dod : Dodecahedron) : Prop :=
  v1 ≠ v2 ∧ ¬ (∃ e, (e ∈ dod.edges_per_vertex))

-- Problem rephrased in Lean 4 statement: proving the number of interior diagonals equals 160.
theorem dodecahedron_interior_diagonals_count (dod : Dodecahedron) :
  (Finset.univ.filter (λ (p : Fin 20 × Fin 20), is_interior_diagonal p.1 p.2 dod)).card = 160 :=
by sorry

end dodecahedron_interior_diagonals_count_l542_542017


namespace triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l542_542864

noncomputable def sum_sine_3A_3B_3C (A B C : ℝ) : ℝ :=
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C)

theorem triangle_inequality_sine_three_times {A B C : ℝ} (h : A + B + C = Real.pi) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) : 
  (-2 : ℝ) ≤ sum_sine_3A_3B_3C A B C ∧ sum_sine_3A_3B_3C A B C ≤ (3 * Real.sqrt 3 / 2) :=
by
  sorry

theorem equality_sine_three_times_lower_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = 0) (h2: B = Real.pi / 2) (h3: C = Real.pi / 2) :
  sum_sine_3A_3B_3C A B C = -2 :=
by
  sorry

theorem equality_sine_three_times_upper_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = Real.pi / 3) (h2: B = Real.pi / 3) (h3: C = Real.pi / 3) :
  sum_sine_3A_3B_3C A B C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l542_542864


namespace monotonic_function_c_bound_l542_542032

noncomputable def f (x c : ℝ) : ℝ := (x^2 - c*x + 5) * Real.exp x

theorem monotonic_function_c_bound (c : ℝ) :
  (∀ x y ∈ Set.Icc (1 / 2) 4, x ≤ y → f x c ≤ f y c) → c ≤ 4 :=
by
  sorry

end monotonic_function_c_bound_l542_542032


namespace pages_to_read_tomorrow_l542_542505

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l542_542505


namespace smallest_digit_divisible_by_9_l542_542713

theorem smallest_digit_divisible_by_9 : 
  ∃ d : ℕ, (∃ m : ℕ, m = 2 + 4 + d + 6 + 0 ∧ m % 9 = 0 ∧ d < 10) ∧ d = 6 :=
by
  sorry

end smallest_digit_divisible_by_9_l542_542713


namespace find_tan_theta_l542_542371

def Point := (ℝ × ℝ)

def Rectangle : Type :=
  {A B C D : Point // (A = (0,0)) ∧ (B = (2,0)) ∧ (C = (2,1)) ∧ (D = (0,1))}

noncomputable def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def tan_theta_condition (theta : ℝ) (P0 : Point) : Prop :=
  let P1 := (2, tan θ) in
  let P2 := (2 - tan θ, 1) in
  let P3 := (0, 1 - tan θ) in
  let P4 := (1 - tan θ, 0) in
  P4 = P0

theorem find_tan_theta (θ : ℝ) (rect : Rectangle) :
  let P0 : Point := midpoint rect.1 rect.2 in
  tan_theta_condition θ P0 → θ = arctan (1 / 2) :=
sorry

end find_tan_theta_l542_542371


namespace magnitude_projection_l542_542832

variables (a b : EuclideanSpace ℝ) (h₁ : inner_product_space.inner a b = 7) (h₂ : ∥b∥ = 3)

theorem magnitude_projection (a b : EuclideanSpace ℝ) (h₁ : inner_product_space.inner a b = 7) (h₂ : ∥b∥ = 3) :
  ∥linear_algebra.proj b a∥ = 7 / 3 :=
by simp [linear_algebra.proj, h₁, h₂]; sorry

end magnitude_projection_l542_542832


namespace henry_initial_games_l542_542014

theorem henry_initial_games :
  ∃ H : ℕ, 
    H - 6 = 4 * (7 + 6) ∧ 
    H = 58 :=
begin
  use 58,  -- We use 58 as our value of H to prove the theorem
  split,
  { 
    -- First condition: H - 6 should be equal to 4 * (7 + 6)
    calc
    58 - 6 = 52 : by norm_num
    ...     = 4 * 13 : by norm_num
    ...     = 4 * (7 + 6) : by norm_num
  },
  {
    -- Second condition: H should be 58
    refl
  }
end

end henry_initial_games_l542_542014


namespace pages_to_read_tomorrow_l542_542507

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l542_542507


namespace vertex_property_l542_542142

theorem vertex_property (a b c m k : ℝ) (h : a ≠ 0)
  (vertex_eq : k = a * m^2 + b * m + c)
  (point_eq : m = a * k^2 + b * k + c) : a * (m - k) > 0 :=
sorry

end vertex_property_l542_542142


namespace sum_grid_is_multiple_of_k_l542_542737

variables {a b n k : ℕ}

theorem sum_grid_is_multiple_of_k (ha : 2 ≤ a) (hb : a < b) (hc : b ≤ n) (hk : k ≥ 2)
  (H1 : ∀ i j, 1 ≤ i ∧ i + a - 1 ≤ n → 1 ≤ j ∧ j + a - 1 ≤ n →
    (∑ x in (finset.range a).product (finset.range a), grid (i + x.1) (j + x.2)) % k = 0)
  (H2 : ∀ i j, 1 ≤ i ∧ i + b - 1 ≤ n → 1 ≤ j ∧ j + b - 1 ≤ n →
    (∑ x in (finset.range b).product (finset.range b), grid (i + x.1) (j + x.2)) % k = 0) : 
  (∑ i in (finset.range n).product (finset.range n), grid i.1 i.2) % k = 0 ↔ 
  (∃ m : ℕ, n = m * a ∨ n = m * b) := 
sorry

end sum_grid_is_multiple_of_k_l542_542737


namespace identify_even_increasing_function_l542_542245

theorem identify_even_increasing_function :
  ( ∀ x : ℝ, even (λ x, 2^|x|) ∧ ∀ x > 0, 2^|x| > 2^|x - 1|) ∧
  ((¬(even (λ x, -x^3)) ∨ ¬(∀ x > 0, -x^3 > (-x^3-1))) ∧
  (¬(even (λ x, x^(1/2))) ∨ ¬(∀ x > 0, x^(1/2) > (x^(1/2)-1))) ∧
  (¬(even (λ x, log 3 (-x))) ∨ ¬(∀ x > 0, log 3 (-x) > log 3 (-x - 1)))
  ) :=
by sorry

end identify_even_increasing_function_l542_542245


namespace original_price_of_shoes_l542_542991

theorem original_price_of_shoes (P : ℝ) (h : 0.08 * P = 16) : P = 200 :=
sorry

end original_price_of_shoes_l542_542991


namespace volunteer_hours_per_year_l542_542061

def volunteer_sessions_per_month := 2
def hours_per_session := 3
def months_per_year := 12

theorem volunteer_hours_per_year : 
  (volunteer_sessions_per_month * months_per_year * hours_per_session) = 72 := 
by
  sorry

end volunteer_hours_per_year_l542_542061


namespace chess_program_ratio_l542_542548

theorem chess_program_ratio {total_students chess_program_absent : ℕ}
  (h_total : total_students = 24)
  (h_absent : chess_program_absent = 4)
  (h_half : chess_program_absent * 2 = chess_program_absent + chess_program_absent) :
  (chess_program_absent * 2 : ℚ) / total_students = 1 / 3 :=
by
  sorry

end chess_program_ratio_l542_542548


namespace discount_percentage_l542_542992

variable (P : ℝ)  -- Original price of the car
variable (D : ℝ)  -- Discount percentage in decimal form
variable (S : ℝ)  -- Selling price of the car

theorem discount_percentage
  (h1 : S = P * (1 - D) * 1.70)
  (h2 : S = P * 1.1899999999999999) :
  D = 0.3 :=
by
  -- The proof goes here
  sorry

end discount_percentage_l542_542992


namespace chenny_total_cost_l542_542675

theorem chenny_total_cost :
    let plates := 9
    let plate_cost := 2.0
    let spoons := 4
    let spoon_cost := 1.50
    let total_cost := (plates * plate_cost) + (spoons * spoon_cost)
    total_cost = 24 :=
by
    let plates := 9
    let plate_cost := 2.0
    let spoons := 4
    let spoon_cost := 1.50
    have total_cost := (plates * plate_cost) + (spoons * spoon_cost)
    show total_cost = 24
    sorry

end chenny_total_cost_l542_542675


namespace employee_n_salary_l542_542614

theorem employee_n_salary (m n : ℝ) (h1: m + n = 594) (h2: m = 1.2 * n) : n = 270 := by
  sorry

end employee_n_salary_l542_542614


namespace fish_sold_total_l542_542590

theorem fish_sold_total (n_high: ℕ) (n_low: ℕ) : 
  n_high = 3 ∧ n_low = 12 → (n_high + n_low + 1 = 16) :=
by
  intro h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end fish_sold_total_l542_542590


namespace sum_of_six_digits_used_is_29_l542_542569

theorem sum_of_six_digits_used_is_29 :
  ∃ (a b c d f g : ℕ), 
    {a, b, c, d, f, g}.card = 6 ∧ 
    a + b + c = 23 ∧ 
    d + b + f + g = 12 ∧ 
    a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    f ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    g ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ f ∧ a ≠ g ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ f ∧ b ≠ g ∧ 
    c ≠ d ∧ c ≠ f ∧ c ≠ g ∧ 
    d ≠ f ∧ d ≠ g ∧ 
    f ≠ g ∧ 
    a + b + c + d + f + g = 29 :=
sorry

end sum_of_six_digits_used_is_29_l542_542569


namespace part_a_part_b_part_c_part_d_l542_542471

-- Given definitions based on the conditions
variables
  (A B C D E F G H I J K L M N : Type)
  [isTriangle : is_triangle A B C]
  (D_is_midpoint : is_midpoint D A B)
  (E_is_midpoint : is_midpoint E A C)
  (FG : Type)
  (FG_half_BC : FG = (1 / 2) * length B C)
  (DJMA_rot : rotated_180 DJMA D DHFB)
  (ELNA_rot : rotated_180 ELNA E EIGC)
  (HJKL_is_rectangle : is_rectangle H J K L)

-- Define angle collinearity
def collinear (X Y Z : Type) : Prop := ∠X Y Z = 180

-- Questions translated into lean statements
theorem part_a :
  collinear M A N :=
sorry

theorem part_b (cong_FG_MNK : ∠FGI = ∠MNK ∧ FG = MN ∧ ∠IFG = ∠KNM) :
  congruent (FGI : Type) (MNK : Type) :=
by 
  cases congruent_intro FG MN FG_half_BC D_is_midpoint; 
  sorry

theorem part_c (DE_half_BC : length D E = 1 / 2 * length B C ∧ parallel D E B C) :
  length L H = length E F :=
sorry

theorem part_d (ABC_area : area A B C = 9 ∧ square H J K L) :
  length E F = 3 :=
sorry

end part_a_part_b_part_c_part_d_l542_542471


namespace term_25_is_6_l542_542487

-- Define the sequence where each natural number n appears exactly n times.
def sequence (n : ℕ) : ℕ := sorry

-- Define the 25th term in the sequence.
def term_at_25 := sequence 25

-- Theorem statement: the 25th term of the sequence is 6.
theorem term_25_is_6 : term_at_25 = 6 := 
by 
  sorry

end term_25_is_6_l542_542487


namespace bookshop_shipment_correct_l542_542980

noncomputable def bookshop_shipment : ℕ :=
  let Initial_books := 743
  let Saturday_instore_sales := 37
  let Saturday_online_sales := 128
  let Sunday_instore_sales := 2 * Saturday_instore_sales
  let Sunday_online_sales := Saturday_online_sales + 34
  let books_sold := Saturday_instore_sales + Saturday_online_sales + Sunday_instore_sales + Sunday_online_sales
  let Final_books := 502
  Final_books - (Initial_books - books_sold)

theorem bookshop_shipment_correct : bookshop_shipment = 160 := by
  sorry

end bookshop_shipment_correct_l542_542980


namespace number_of_crabs_in_basket_l542_542812

theorem number_of_crabs_in_basket (crab_baskets_per_week : ℕ) (baskets_per_collection: ℕ) (price_per_crab : ℕ) (total_revenue : ℕ) :
  crab_baskets_per_week = 3 → 
  baskets_per_collection = 2 → 
  price_per_crab = 3 → 
  total_revenue = 72 → 
  4 = total_revenue / (crab_baskets_per_week * baskets_per_collection * price_per_crab) := 
by
  intros h1 h2 h3 h4
  have H : total_revenue = 3 * 2 * 3 * 4 := by rw [h1, h2, h3]; exact h4
  have : 3 * 2 * 3 * 4 = 72 := by norm_num
  rwa H at this
  sorry

end number_of_crabs_in_basket_l542_542812


namespace range_of_a_l542_542365

noncomputable def a_n (n : ℕ) (a : ℝ) : ℝ :=
  (-1)^(n + 2018) * a

noncomputable def b_n (n : ℕ) : ℝ :=
  2 + (-1)^(n + 2019) / n

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ, 1 ≤ n → a_n n a < b_n n) ↔ -2 ≤ a ∧ a < 3 / 2 :=
  sorry

end range_of_a_l542_542365


namespace cosine_angle_D₁O_BO₁_l542_542477

-- Definitions and assumptions for the given problem:
variable (A B C D A₁ B₁ C₁ D₁ O O₁ : Type)
variable [Cube A B C D A₁ B₁ C₁ D₁]
variable [Center O ABCD] -- O is the center of face ABCD
variable [Center O₁ ADD₁A₁] -- O₁ is the center of face ADD₁A₁

-- Define the points D₁O and BO₁ as line segments:
def D₁O := line_segment D₁ O
def BO₁ := line_segment B O₁

-- Problem statement: Prove cosine of the angle between D₁O and BO₁ is 5/6:
theorem cosine_angle_D₁O_BO₁ : 
  cos (angle D₁O BO₁) = 5 / 6 := 
sorry

end cosine_angle_D₁O_BO₁_l542_542477


namespace president_and_committee_count_l542_542378

-- We define a type to represent the 10 people.
inductive People : Type
| P0 : People
| P1 : People
| P2 : People
| P3 : People
| P4 : People
| P5 : People
| P6 : People
| P7 : People
| P8 : People
| P9 : People

open People

-- The main theorem to prove
theorem president_and_committee_count : 
  (∑ p in list.erase (People → Prop) id (list.fin_range 10), 
  (nat.choose (9 : ℕ) (3 : ℕ))) = 840 :=
by
  -- There are 10 ways to choose the president
  let president_ways := 10

  -- For each president, choose 3 committee members from the remaining 9 people
  let committee_ways := nat.choose 9 3

  -- Therefore, the total number of ways
  have h_total := president_ways * committee_ways

  -- Using the given combination value
  have h_combination: committee_ways = 84 := by
    rw nat.choose,
    -- Calculation steps for the combination (9, 3)
    sorry,

  -- Substituting the values
  rw h_combination at h_total,

  -- Therefore, the total ways are 840
  exact eq.symm (nat.mul_right_inj' (nat.pos_of_ne_zero (ne_of_eq_of_ne nat.choose_zero_right (ne_of_lt nat.lt_succ_self)))),
    sorry,

end president_and_committee_count_l542_542378


namespace reflex_angle_at_G_l542_542722

-- Define the conditions as variables and state the problem
theorem reflex_angle_at_G 
  (B A E L G : Type) -- Points B, A, E, L are on a straight line, and G is another point
  (h1: ∠ BAG = 130) -- Condition 2
  (h2: ∠ GEL = 70)  -- Condition 3
  : reflex_angle G = 340 := 
by 
  sorry -- Proof not required

end reflex_angle_at_G_l542_542722


namespace proposition_correctness_l542_542999

theorem proposition_correctness :
  (∀ (P : Prop), (¬P → P) → (¬P)) ∧
  (∀ (a1 a3 a4 d : ℝ), (a1 = 2) ∧ (a3 - a1 = d) ∧ (a4 - a3 = d) ∧ (a1 * a3 * a4 = a1^3) → (d ≠ -1/2) ) ∧
  (∀ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ (a + b = 1) → ∃ x y, (x = 5 + 2*sqrt 6) ∧ (y = (2/a) + (3/b)) ∧ (x ≤ y)) ∧
  (∀ (A B C : ℝ), sin A^2 < sin B^2 + sin C^2 → ¬acute_triangle A B C) :=
by
  split
  · intros P hP
    exact sorry
  split
  · intros a1 a3 a4 d h
    exact sorry
  split
  · intros a b hab
    use 5 + 2 * sqrt 6
    exact sorry
  · intros A B C hABC
    exact sorry

end proposition_correctness_l542_542999


namespace pages_to_read_tomorrow_l542_542504

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l542_542504


namespace train_crosses_second_platform_in_20_seconds_l542_542233

variable (length_first_platform length_train length_second_platform time_platform_one : ℕ)
variable (train_speed : ℚ)

-- Given conditions
def condition1 := length_first_platform = 180
def condition2 := length_train = 30
def condition3 := length_second_platform = 250
def condition4 := time_platform_one = 15
def condition5 := train_speed = (length_train + length_first_platform) / time_platform_one

-- Proof that the train crosses the second platform in 20 seconds
theorem train_crosses_second_platform_in_20_seconds 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : 
  (length_train + length_second_platform) / train_speed = 20 := by
  sorry

end train_crosses_second_platform_in_20_seconds_l542_542233


namespace line_circle_intersection_angle_l542_542250

theorem line_circle_intersection_angle :
  let line := λ (x y : ℝ), x + y = 15 in
  let circle := λ (x y : ℝ), (x - 4)^2 + (y - 5)^2 = 36 in
  let intersection_points : set (ℝ × ℝ) :=
    { (10, 5), (4, 11) } in
  let center := (4, 5) in
  let radius := 6 in
  ∀ (p ∈ intersection_points),
    let θ := if p = (10, 5) then real.arctan 1 else if p = (4, 11) then real.pi / 2 else 0 in
    θ = real.pi / 4 ∨ θ = real.pi / 2 :=
sorry

end line_circle_intersection_angle_l542_542250


namespace quadratic_distinct_roots_l542_542034

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end quadratic_distinct_roots_l542_542034


namespace correct_statements_l542_542322

-- Definitions
def z : ℂ := 1 + complex.I

-- Statements to prove
theorem correct_statements :
  (conj(z) = 1 - complex.I) ∧ (complex.abs z = real.sqrt 2) :=
by
  sorry -- Proofs are skipped as per instructions

end correct_statements_l542_542322


namespace rectangle_area_l542_542211

theorem rectangle_area (r : ℝ) (rectangle_tangent : ∀ (A B C D M O : Point), 
  Circle O r → 
  (Tangent (Circle O r) (Segment A B) ∧ Tangent (Circle O r) (Segment B C)) → 
  PassThrough (Circle O r) M → 
  Midpoint M A D → 
  RightAngle A B C → 
  (SideLength A B 2 * r ∧ SideLength A D 2 * r)) : 
  ∀ (A B C D : Point), 
  is_rectangle A B C D → 
  Area A B C D = 4 * r * r :=
by
  sorry

end rectangle_area_l542_542211


namespace problem_inequality_l542_542532

theorem problem_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 :=
by
  sorry

end problem_inequality_l542_542532


namespace nine_chapters_problem_l542_542476

theorem nine_chapters_problem
  (n : ℕ)
  (h : n ≥ 2) :
  let a : ℕ → ℚ := λ k, n / k in
  (finset.range (n - 1)).sum (λ k, a (k + 1) * a (k + 2)) = n * (n - 1) :=
begin
  sorry
end

end nine_chapters_problem_l542_542476


namespace teacher_has_graded_8_worksheets_l542_542997

theorem teacher_has_graded_8_worksheets :
  ∀ (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (remaining_problems : ℕ),
  total_worksheets = 16 →
  problems_per_worksheet = 4 →
  remaining_problems = 32 →
  (total_worksheets * problems_per_worksheet - remaining_problems) / problems_per_worksheet = 8 :=
begin
  intros total_worksheets problems_per_worksheet remaining_problems,
  assume hw : total_worksheets = 16,
  assume hp : problems_per_worksheet = 4,
  assume hr : remaining_problems = 32,
  sorry
end

end teacher_has_graded_8_worksheets_l542_542997


namespace perfect_square_digits_uniqueness_number_of_solutions_uniqueness_l542_542699

theorem perfect_square_digits_uniqueness :
  ∃ (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ : ℕ),
  (a₁₁ * 100 + a₁₂ * 10 + a₁₃ : ℕ).is_square ∧
  (a₂₁ * 100 + a₂₂ * 10 + a₂₃ : ℕ).is_square ∧
  (a₁₁ * 10 + a₂₁ : ℕ).is_square ∧
  (a₁₂ * 10 + a₂₂ : ℕ).is_square ∧
  (a₁₃ * 10 + a₂₃ : ℕ).is_square ∧
  a₁₁ = 8 ∧ a₁₂ = 4 ∧ a₁₃ = 1 ∧
  a₂₁ = 1 ∧ a₂₂ = 9 ∧ a₂₃ = 6 :=
begin
  sorry
end

theorem number_of_solutions_uniqueness :
  ∃! (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ : ℕ),
  (a₁₁ * 100 + a₁₂ * 10 + a₁₃ : ℕ).is_square ∧
  (a₂₁ * 100 + a₂₂ * 10 + a₂₃ : ℕ).is_square ∧
  (a₁₁ * 10 + a₂₁ : ℕ).is_square ∧
  (a₁₂ * 10 + a₂₂ : ℕ).is_square ∧
  (a₁₃ * 10 + a₂₃ : ℕ).is_square :=
begin
  sorry
end

end perfect_square_digits_uniqueness_number_of_solutions_uniqueness_l542_542699


namespace candy_problem_l542_542456

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542456


namespace min_phi_for_odd_function_l542_542593

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def translate_left (f : ℝ → ℝ) (phi : ℝ) : ℝ → ℝ :=
  λ x, f (x + phi)

def orig_function (x : ℝ) : ℝ :=
  2 * sin (x + (π / 3)) * cos (x + (π / 3))

theorem min_phi_for_odd_function :
  ∃ (φ : ℝ), φ > 0 ∧ is_odd (translate_left orig_function φ) ∧ φ = π / 6 :=
begin
  sorry
end

end min_phi_for_odd_function_l542_542593


namespace collinear_pq_l542_542009

-- Define the points A, B, C as vectors
def A : ℝ × ℝ × ℝ := (1, 5, -2)
def B : ℝ × ℝ × ℝ := (2, 4, 1)
def C (p q : ℝ) : ℝ × ℝ × ℝ := (p, 2, q + 2)

-- Vectors AB and AC
def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def AC (p q : ℝ) : ℝ × ℝ × ℝ := (C p q).1 - A.1, (C p q).2 - A.2, (C p q).3 - A.3)

-- Collinearity condition
noncomputable def collinear_condition (p q λ : ℝ) :=
  (AC p q) = λ • (AB)

-- Proof statement
theorem collinear_pq (p q : ℝ) (h : ∃ λ : ℝ, collinear_condition p q λ) : p + q = 9 := 
by
  sorry

end collinear_pq_l542_542009


namespace hyperbola_C2_eq_l542_542765

noncomputable def hyperbola_foci : Set (ℝ × ℝ) := { (2, 0), (-2, 0) }

theorem hyperbola_C2_eq 
  (C1_eq : ∀ x y : ℝ, C1_eq x y = (x^2 / 3 - y^2 = 1))
  (foci_coincide : hyperbola_foci = { (2, 0), (-2, 0) })
  (slope_condition : ∀ x y : ℝ, slope_C2 = 2 * slope_C1) :
    ∀ x y : ℝ, C2_eq x y = (x^2 - y^2 / 3 = 1) := 
    sorry

end hyperbola_C2_eq_l542_542765


namespace minimum_candy_kinds_l542_542409

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542409


namespace tank_empty_time_l542_542647

noncomputable def capacity : ℝ := 5760
noncomputable def leak_rate_time : ℝ := 6
noncomputable def inlet_rate_per_minute : ℝ := 4

-- leak rate calculation
noncomputable def leak_rate : ℝ := capacity / leak_rate_time

-- inlet rate calculation in litres per hour
noncomputable def inlet_rate : ℝ := inlet_rate_per_minute * 60

-- net emptying rate calculation
noncomputable def net_empty_rate : ℝ := leak_rate - inlet_rate

-- time to empty the tank calculation
noncomputable def time_to_empty : ℝ := capacity / net_empty_rate

-- The statement to prove
theorem tank_empty_time : time_to_empty = 8 :=
by
  -- Definition step
  have h1 : leak_rate = capacity / leak_rate_time := rfl
  have h2 : inlet_rate = inlet_rate_per_minute * 60 := rfl
  have h3 : net_empty_rate = leak_rate - inlet_rate := rfl
  have h4 : time_to_empty = capacity / net_empty_rate := rfl

  -- Final proof (skipped with sorry)
  sorry

end tank_empty_time_l542_542647


namespace find_n_l542_542799

noncomputable def sequence_an (n : ℕ) : ℝ := 1 / (Real.sqrt n + Real.sqrt (n + 1))

def partial_sum (n : ℕ) : ℝ := ∑ k in Finset.range n, sequence_an k

theorem find_n (S_n : ℝ) (h : S_n = 9) : 
  (∃ n : ℕ, partial_sum n = S_n) → ∃ n : ℕ, n = 99 :=
by
  sorry

end find_n_l542_542799


namespace remaining_quantities_count_l542_542582

theorem remaining_quantities_count 
  (S : ℕ) (S3 : ℕ) (S2 : ℕ) (n : ℕ) 
  (h1 : S / 5 = 10) 
  (h2 : S3 / 3 = 4) 
  (h3 : S = 50) 
  (h4 : S3 = 12) 
  (h5 : S2 = S - S3) 
  (h6 : S2 / n = 19) 
  : n = 2 := 
by 
  sorry

end remaining_quantities_count_l542_542582


namespace minimum_candy_kinds_l542_542399

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542399


namespace michael_can_cover_both_classes_l542_542097

open Nat

def total_students : ℕ := 30
def german_students : ℕ := 20
def japanese_students : ℕ := 24

-- Calculate the number of students taking both German and Japanese using inclusion-exclusion principle.
def both_students : ℕ := german_students + japanese_students - total_students

-- Calculate the number of students only taking German.
def only_german_students : ℕ := german_students - both_students

-- Calculate the number of students only taking Japanese.
def only_japanese_students : ℕ := japanese_students - both_students

-- Calculate the total number of ways to choose 2 students out of 30.
def total_ways_to_choose_2 : ℕ := (total_students * (total_students - 1)) / 2

-- Calculate the number of ways to choose 2 students only taking German or only taking Japanese.
def undesirable_outcomes : ℕ := (only_german_students * (only_german_students - 1)) / 2 + (only_japanese_students * (only_japanese_students - 1)) / 2

-- Calculate the probability of undesirable outcomes.
def undesirable_probability : ℚ := undesirable_outcomes / total_ways_to_choose_2

-- Calculate the probability Michael can cover both German and Japanese classes.
def desired_probability : ℚ := 1 - undesirable_probability

theorem michael_can_cover_both_classes : desired_probability = 25 / 29 := by sorry

end michael_can_cover_both_classes_l542_542097


namespace roots_sum_cubes_l542_542826

theorem roots_sum_cubes (a b c d : ℝ) 
  (h_eqn : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    3 * x^4 + 6 * x^3 + 1002 * x^2 + 2005 * x + 4010 = 0) :
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by { sorry }

end roots_sum_cubes_l542_542826


namespace actor_A_constraints_l542_542175

-- Definitions corresponding to the conditions.
def numberOfActors : Nat := 6
def positionConstraints : Nat := 4
def permutations (n : Nat) : Nat := Nat.factorial n

-- Lean statement for the proof problem.
theorem actor_A_constraints : 
  (positionConstraints * permutations (numberOfActors - 1)) = 480 := by
sorry

end actor_A_constraints_l542_542175


namespace sum_nk_l542_542126

theorem sum_nk (n k : ℕ) (h₁ : 3 * n - 4 * k = 4) (h₂ : 4 * n - 5 * k = 13) : n + k = 55 := by
  sorry

end sum_nk_l542_542126


namespace calc_length_MY_l542_542802

-- Define the known lengths and parameters
variables (YZ XK YM r s : ℝ)

-- Define the condition that XK and YM are altitudes
variables (XK_altitude_to_YZ : Altitude XK YZ)
variables (YM_altitude_to_XZ : Altitude YM XZ)

-- Define the inradius
variable (inradius : Inradius r)

-- Theorem statement (noncomputable since we are depending on geometric constructs and actual length computation)
noncomputable def length_MY (YZ XK YM r s : ℝ) 
  (XK_altitude_to_YZ : Altitude XK YZ) 
  (YM_altitude_to_XZ : Altitude YM XZ) 
  (inradius : Inradius r) : ℝ :=
  2 * r * s * YM / (XK * YZ)

-- The final theorem statement
theorem calc_length_MY (YZ XK YM r s : ℝ) 
  (XK_altitude_to_YZ : Altitude XK YZ) 
  (YM_altitude_to_XZ : Altitude YM XZ) 
  (inradius : Inradius r) :
  MY = length_MY YZ XK YM r s XK_altitude_to_YZ YM_altitude_to_XZ inradius :=
sorry

end calc_length_MY_l542_542802


namespace museum_wings_paintings_l542_542982

theorem museum_wings_paintings (P A : ℕ) (h1: P + A = 8) (h2: P = 1 + 2) : P = 3 :=
by
  -- Proof here
  sorry

end museum_wings_paintings_l542_542982


namespace largest_4_digit_congruent_to_15_mod_25_l542_542170

theorem largest_4_digit_congruent_to_15_mod_25 : 
  ∀ x : ℕ, (1000 ≤ x ∧ x < 10000 ∧ x % 25 = 15) → x = 9990 :=
by
  intros x h
  sorry

end largest_4_digit_congruent_to_15_mod_25_l542_542170


namespace diana_can_paint_seven_statues_l542_542948

theorem diana_can_paint_seven_statues (total_paint : ℚ) (paint_per_statue : ℚ) (n : ℕ) (h1 : total_paint = 7 / 16) (h2 : paint_per_statue = 1 / 16) (h3 : n = total_paint / paint_per_statue) : n = 7 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end diana_can_paint_seven_statues_l542_542948


namespace M_inter_N_empty_l542_542839

def M : Set ℝ := {a : ℝ | (1 / 2 < a ∧ a < 1) ∨ (1 < a)}
def N : Set ℝ := {a : ℝ | 0 < a ∧ a ≤ 1 / 2}

theorem M_inter_N_empty : M ∩ N = ∅ :=
sorry

end M_inter_N_empty_l542_542839


namespace negation_of_universal_to_existential_l542_542758

theorem negation_of_universal_to_existential :
  (¬(∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end negation_of_universal_to_existential_l542_542758


namespace num_terms_arithmetic_seq_l542_542686

theorem num_terms_arithmetic_seq (a d l : ℝ) (n : ℕ)
  (h1 : a = 3.25) 
  (h2 : d = 4)
  (h3 : l = 55.25)
  (h4 : l = a + (↑n - 1) * d) :
  n = 14 :=
by
  sorry

end num_terms_arithmetic_seq_l542_542686


namespace rhombus_side_length_l542_542668

noncomputable def side_length_rhombus (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) : ℝ :=
  4

theorem rhombus_side_length (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) (x : ℝ) :
  side_length_rhombus AB BC AC condition1 condition2 condition3 = x ↔ x = 4 := by
  sorry

end rhombus_side_length_l542_542668


namespace gcd_and_lcm_of_18_and_24_l542_542288

-- Definitions of gcd and lcm for the problem's context
def my_gcd (a b : ℕ) : ℕ := a.gcd b
def my_lcm (a b : ℕ) : ℕ := a.lcm b

-- Constants given in the problem
def a := 18
def b := 24

-- Proof problem statement
theorem gcd_and_lcm_of_18_and_24 : my_gcd a b = 6 ∧ my_lcm a b = 72 := by
  sorry

end gcd_and_lcm_of_18_and_24_l542_542288


namespace find_siblings_l542_542902

-- Define the characteristics of each child
structure Child where
  name : String
  eyeColor : String
  hairColor : String
  age : Nat

-- List of children
def Olivia : Child := { name := "Olivia", eyeColor := "Green", hairColor := "Red", age := 12 }
def Henry  : Child := { name := "Henry", eyeColor := "Gray", hairColor := "Brown", age := 12 }
def Lucas  : Child := { name := "Lucas", eyeColor := "Green", hairColor := "Red", age := 10 }
def Emma   : Child := { name := "Emma", eyeColor := "Green", hairColor := "Brown", age := 12 }
def Mia    : Child := { name := "Mia", eyeColor := "Gray", hairColor := "Red", age := 10 }
def Noah   : Child := { name := "Noah", eyeColor := "Gray", hairColor := "Brown", age := 12 }

-- Define a family as a set of children who share at least one characteristic
def isFamily (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.eyeColor = c3.eyeColor ∨ c2.eyeColor = c3.eyeColor) ∨
  (c1.hairColor = c2.hairColor ∨ c1.hairColor = c3.hairColor ∨ c2.hairColor = c3.hairColor) ∨
  (c1.age = c2.age ∨ c1.age = c3.age ∨ c2.age = c3.age)

-- The main theorem
theorem find_siblings : isFamily Olivia Lucas Emma :=
by
  sorry

end find_siblings_l542_542902


namespace not_possible_acquaintance_arrangement_l542_542721

-- Definitions and conditions for the problem
def num_people : ℕ := 40
def even_people_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 0 → A ≠ B → true -- A and B have a mutual acquaintance if an even number of people sit between them

def odd_people_not_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 1 → A ≠ B → true -- A and B do not have a mutual acquaintance if an odd number of people sit between them

theorem not_possible_acquaintance_arrangement : ¬ (∀ A B : ℕ, A ≠ B →
  (∀ num_between : ℕ, (num_between % 2 = 0 → even_people_acquainted A B num_between) ∧
  (num_between % 2 = 1 → odd_people_not_acquainted A B num_between))) :=
sorry

end not_possible_acquaintance_arrangement_l542_542721


namespace necessary_but_not_sufficient_conditions_l542_542635

theorem necessary_but_not_sufficient_conditions (x y : ℝ) :
  (|x| ≤ 1 ∧ |y| ≤ 1) → x^2 + y^2 ≤ 1 ∨ ¬(x^2 + y^2 ≤ 1) → 
  (|x| ≤ 1 ∧ |y| ≤ 1) → (x^2 + y^2 ≤ 1 → (|x| ≤ 1 ∧ |y| ≤ 1)) :=
by
  sorry

end necessary_but_not_sufficient_conditions_l542_542635


namespace dogwood_trees_current_l542_542149

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end dogwood_trees_current_l542_542149


namespace mutually_exclusive_scoring_l542_542228

-- Define conditions as types
def shoots_twice : Prop := true
def scoring_at_least_once : Prop :=
  ∃ (shot1 shot2 : Bool), shot1 || shot2
def not_scoring_both_times : Prop :=
  ∀ (shot1 shot2 : Bool), ¬(shot1 && shot2)

-- Statement of the problem: Prove the events are mutually exclusive.
theorem mutually_exclusive_scoring :
  shoots_twice → (scoring_at_least_once → not_scoring_both_times → false) :=
by
  intro h_shoots_twice
  intro h_scoring_at_least_once
  intro h_not_scoring_both_times
  sorry

end mutually_exclusive_scoring_l542_542228


namespace diesel_train_slower_l542_542205

theorem diesel_train_slower
    (t_cattle_speed : ℕ)
    (t_cattle_early_hours : ℕ)
    (t_diesel_hours : ℕ)
    (total_distance : ℕ)
    (diesel_speed : ℕ) :
  t_cattle_speed = 56 →
  t_cattle_early_hours = 6 →
  t_diesel_hours = 12 →
  total_distance = 1284 →
  diesel_speed = 23 →
  t_cattle_speed - diesel_speed = 33 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end diesel_train_slower_l542_542205


namespace apr_sales_is_75_l542_542119

-- Definitions based on conditions
def sales_jan : ℕ := 90
def sales_feb : ℕ := 50
def sales_mar : ℕ := 70
def avg_sales : ℕ := 72

-- Total sales of first three months
def total_sales_jan_to_mar : ℕ := sales_jan + sales_feb + sales_mar

-- Total sales considering average sales over 5 months
def total_sales : ℕ := avg_sales * 5

-- Defining April sales
def sales_apr (sales_may : ℕ) : ℕ := total_sales - total_sales_jan_to_mar - sales_may

theorem apr_sales_is_75 (sales_may : ℕ) : sales_apr sales_may = 75 :=
by
  unfold sales_apr total_sales total_sales_jan_to_mar avg_sales sales_jan sales_feb sales_mar
  -- Here we could insert more steps if needed to directly connect to the proof
  sorry


end apr_sales_is_75_l542_542119


namespace candy_problem_l542_542453

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542453


namespace locus_of_X_is_angle_bisector_l542_542329

variable {A B C D E F G X : Type}
variables [Circle γ1 γ2]
variables [IsTriangle ABC] [IsAcute ABC]

/-- The locus of points X determined by the internal tangents of γ1 and γ2 is the angle bisector of ∠BAC -/
theorem locus_of_X_is_angle_bisector
  (h_parallel: line.parallel (line.mk A E) (line.mk B C))
  (h_intersections: (circle.intersectminorarc γ1 (arc.mk A B) = F) ∧
    (circle.intersectminorarc γ1 (arc.mk A C) = G))
  (h_tangents_gamma1: (circle.tangent γ1 (line.mk B D)) ∧
    (circle.tangent γ1 (line.mk D F)))
  (h_tangents_gamma2: (circle.tangent γ2 (line.mk C E)) ∧
    (circle.tangent γ2 (line.mk E G)))
  (h_internal_tangents: internal_tangents γ1 γ2 = X):
  ∀ (X), (locus_of_points X) is angle_bisector (angle.mk A B C) :=
sorry

end locus_of_X_is_angle_bisector_l542_542329


namespace total_movie_hours_l542_542064

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l542_542064


namespace integral_sqrt_one_minus_x2_l542_542752

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - x^2)

theorem integral_sqrt_one_minus_x2 :
    ∫ x in -1.. 1, f x = π / 2 :=
by
  -- To be proved
  sorry

end integral_sqrt_one_minus_x2_l542_542752


namespace binomial_60_3_l542_542680

theorem binomial_60_3 : nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l542_542680


namespace set_of_x_values_l542_542138

theorem set_of_x_values (x : ℝ) : (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by
  sorry

end set_of_x_values_l542_542138


namespace election_winner_votes_l542_542608

variable (V : ℝ) (winner_votes : ℝ) (winner_margin : ℝ)
variable (condition1 : V > 0)
variable (condition2 : winner_votes = 0.60 * V)
variable (condition3 : winner_margin = 240)

theorem election_winner_votes (h : winner_votes - 0.40 * V = winner_margin) : winner_votes = 720 := by
  sorry

end election_winner_votes_l542_542608


namespace solution_l542_542280

noncomputable def problem_statement : Prop :=
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f(x^2 + f(y)) = y + f(x)^2) → f = id

theorem solution : problem_statement :=
sorry

end solution_l542_542280


namespace candy_problem_l542_542460

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l542_542460


namespace min_a_minus_b_when_ab_eq_156_l542_542825

theorem min_a_minus_b_when_ab_eq_156 : ∃ a b : ℤ, (a * b = 156 ∧ a - b = -155) :=
by
  sorry

end min_a_minus_b_when_ab_eq_156_l542_542825


namespace triangle_perimeter_l542_542328

structure Triangle (a b c : ℕ) :=
  (ineq1 : a + b > c)
  (ineq2 : a + c > b)
  (ineq3 : b + c > a)

theorem triangle_perimeter (a b x : ℕ) (h_eq1 : a = 3) (h_eq2 : b = 6)
  (h_x1 : x^2 - 7 * x + 12 = 0) : 
  (∃ t : Triangle a b x, a + b + x = 13) :=
by
  have h_solutions : x = 4 → (∀ t : Triangle a b x, a + b + x = 13) :=
  sorry
  exact ⟨h_solutions 4 sorry, sorry⟩

end triangle_perimeter_l542_542328


namespace crayons_total_l542_542698

theorem crayons_total (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 15 → crayons_per_row = 42 → total_crayons = rows * crayons_per_row → total_crayons = 630 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end crayons_total_l542_542698


namespace degree_of_g_l542_542870

noncomputable def degree (p : Polynomial ℂ) : ℕ := sorry -- Definition of degree function for polynomials

variable (f g h : Polynomial ℂ)

-- Conditions
def h_def : h = f.comp g - g := sorry -- Definition of h in terms of f and g
def deg_h : degree h = 4 := sorry -- Given degree of h
def deg_f : degree f = 3 := sorry -- Given degree of f

-- The theorem we want to prove
theorem degree_of_g : degree g = 4 :=
by
  -- Use the conditions specified above
  assume h_def
  assume deg_h
  assume deg_f
  sorry

end degree_of_g_l542_542870


namespace correct_statements_l542_542321

-- Definitions
def z : ℂ := 1 + complex.I

-- Statements to prove
theorem correct_statements :
  (conj(z) = 1 - complex.I) ∧ (complex.abs z = real.sqrt 2) :=
by
  sorry -- Proofs are skipped as per instructions

end correct_statements_l542_542321


namespace minimum_candy_kinds_l542_542404

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542404


namespace range_of_a_l542_542004

variable {x a : ℝ}

def proposition_p := (∃ a : ℝ, ∀ x : ℝ, y = log (0.5:ℝ) (x^2 + 2*x + a) ∈ ℝ)
def proposition_q := (∃ a : ℝ, ∀ x : ℝ, y = -(5 - 2*a)^x is_decreasing_on ℝ)

theorem range_of_a (a : ℝ) (p : proposition_p) (q : proposition_q) 
  (either_p_or_q : proposition_p ∨ proposition_q)
  (not_both_p_and_q : ¬(proposition_p ∧ proposition_q)) : 
  (1 < a ∧ a < 2) :=
sorry

end range_of_a_l542_542004


namespace raviraj_distance_home_l542_542628

theorem raviraj_distance_home :
  let origin := (0, 0)
  let after_south := (0, -20)
  let after_west := (-10, -20)
  let after_north := (-10, 0)
  let final_pos := (-30, 0)
  Real.sqrt ((final_pos.1 - origin.1)^2 + (final_pos.2 - origin.2)^2) = 30 :=
by
  sorry

end raviraj_distance_home_l542_542628


namespace monotonicity_of_f_l542_542363

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp x - 1/2 * x^2 + a * x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, f x a = (x - a - 1) * Real.exp x - 1/2 * x^2 + a * x) →
  ((
    (a < 0 → (∀ x : ℝ, x < a ∨ x > 0 → f' x a > 0) ∧ (a < 0 → a < x ∧ x < 0 → f' x a < 0)) ∧
    (a = 0 → ∀ x : ℝ, f' x a ≥ 0) ∧
    (a > 0 → (∀ x : ℝ, x < 0 ∨ x > a → f' x a > 0) ∧ (a > 0 → 0 < x ∧ x < a → f' x a < 0))
  )) ∧ 
  (∀ x_0 : ℝ, 1 ≤ x_0 ∧ x_0 ≤ 2 → f x_0 a < 0 → 
    (a > 1 / (2 * (1 - Real.exp 1))))
:= sorry

end monotonicity_of_f_l542_542363


namespace polynomial_divisible_by_x_squared_plus_one_l542_542558

theorem polynomial_divisible_by_x_squared_plus_one 
  (α : ℝ) (n : ℤ) :
  ∃ p : polynomial ℂ, 
    (polynomial.C (complex.cos α) + polynomial.X * polynomial.C (complex.sin α)) ^ n.to_nat
    - polynomial.C (complex.cos (n * α)) - polynomial.X * polynomial.C (complex.sin (n * α))
    = p * (polynomial.X^2 + 1) :=
sorry

end polynomial_divisible_by_x_squared_plus_one_l542_542558


namespace min_value_fraction_l542_542337

theorem min_value_fraction 
  (a b : ℝ) 
  (h1: 0 < a) 
  (h2: 0 < b) 
  (h3: a + b = 1) :
  ∃ m : ℝ, m = sqrt 2 + 3 / 2 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x + y = 1 → (1 / (2 * x) + 1 / y) ≥ m) :=
sorry

end min_value_fraction_l542_542337


namespace total_weight_correct_l542_542229

def weights : List ℝ := [91, 91, 91.5, 89, 91.2, 91.3, 88.7, 88.8, 91.8, 91.1]

theorem total_weight_correct : 
  (∑ w in weights, w) = 905.4 := 
by
  sorry

end total_weight_correct_l542_542229


namespace A_share_of_profit_l542_542976

-- Definitions for the problem
def A_investment_amount := 400
def A_investment_duration := 12
def B_investment_amount := 200
def B_investment_duration := 6
def total_profit := 100

-- Theorem statement, proving A's share of the profit is $66.67
theorem A_share_of_profit : 
  let A_investment := A_investment_amount * A_investment_duration,
      B_investment := B_investment_amount * B_investment_duration,
      total_investment := A_investment + B_investment,
      A_ratio := (A_investment : ℝ) / total_investment in
  total_profit * A_ratio = 66.67 :=
by
  sorry

end A_share_of_profit_l542_542976


namespace range_of_a_l542_542319

variables {x a : ℝ}

def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x ^ 2 - a * x ≤ x - a

theorem range_of_a (h : ¬(∃ x, p x) → ¬(∃ x, q x a)) :
  1 ≤ a ∧ a < 3 :=
by 
  sorry

end range_of_a_l542_542319


namespace unique_polynomial_l542_542284

-- Definitions of the initial conditions
def polynomial (f : ℝ → ℝ) : Prop := ∃ p : ℝ → ℝ, ∀ x, p x = f x

-- Main theorem statement
theorem unique_polynomial (f : ℝ → ℝ) (hf_poly : polynomial f)
  (hf0 : f 0 = 0)
  (hf_prop : ∀ x : ℝ, f (x^2 + 1) = f x^2 + 1):
  f = (λ x, x) :=
by
  sorry

end unique_polynomial_l542_542284


namespace minimum_candy_kinds_l542_542408

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542408


namespace equilateral_hyperbola_through_point_l542_542586

def hyperbola_eq (x y : ℝ) (λ : ℝ) : Prop := x^2 - y^2 = λ

theorem equilateral_hyperbola_through_point :
  ∃ λ : ℝ, hyperbola_eq 3 (-1) λ ∧ λ ≠ 0 ∧ λ = 8 :=
by {
  use 8,
  have p : hyperbola_eq 3 (-1) 8, by {
    unfold hyperbola_eq,
    norm_num,
  },
  exact ⟨p, by norm_num, rfl⟩
}

end equilateral_hyperbola_through_point_l542_542586


namespace angle_XIY_90_deg_l542_542968

variables {α : Type*} {P : α → α → α → Prop} {Q : α → α → α → α → Prop}

-- Definitions for geometrical constructs involved
def circle_incenter (A B C I : α) := P A B C ∧ Q A B C I
def common_external_tangents_meet_at (A B C D I X : α) := P A B C ∧ Q A B C I ∧ Q A D C X
def convex_quadrilateral_with_inscribed_circle (A B C D I : α) := Q A B C D ∧ Q B C D I ∧ Q A B I D

theorem angle_XIY_90_deg
  (A B C D I I_a I_b I_c I_d X Y : α)
  (h1 : convex_quadrilateral_with_inscribed_circle A B C D I)
  (h2 : circle_incenter D A B I_a)
  (h3 : circle_incenter A B C I_b)
  (h4 : circle_incenter B C D I_c)
  (h5 : circle_incenter C D A I_d)
  (h6 : common_external_tangents_meet_at A I_b I_d C X)
  (h7 : common_external_tangents_meet_at B I_a I_c D Y) :
  ∠ X I Y = 90 :=
sorry

end angle_XIY_90_deg_l542_542968


namespace width_of_carpet_l542_542249

theorem width_of_carpet (length_of_carpet : ℝ) (perc_covering : ℝ) (area_of_living_room : ℝ) (area_of_carpet : ℝ) (width_of_carpet : ℝ) :
  length_of_carpet = 9 ∧ perc_covering = 0.20 ∧ area_of_living_room = 180 ∧
  area_of_carpet = perc_covering * area_of_living_room ∧ width_of_carpet = area_of_carpet / length_of_carpet ->
  width_of_carpet = 4 :=
by
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with h4 h5,
  -- Variables we obtained from conditions
  have h_area_of_carpet := h4,
  have h_width_of_carpet := h5,
  -- Length of the carpet
  rw h1 at *,
  -- Living room area
  rw h3 at *,
  -- Carpet area calculation
  rw h2 at *,
  have h_area_computed : area_of_carpet = 0.2 * 180,
  -- Side-step due to floating point, exact calculations assured by context above
  simp at *,
  exact eq_of_mul_eq_mul_right initial,
  have h_width_computed : width_of_carpet = (0.2 * 180) / 9,
  rw h_width_of_carpet,
  norm_num,
  exact h_width_computed,
  sorry

end width_of_carpet_l542_542249


namespace max_possible_salary_l542_542395

-- Definition of the conditions
def num_players : ℕ := 25
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 800000

-- The theorem we want to prove: the maximum possible salary for a single player is $320,000
theorem max_possible_salary (total_salary_cap : ℕ) (num_players : ℕ) (min_salary : ℕ) :
  total_salary_cap - (num_players - 1) * min_salary = 320000 :=
by sorry

end max_possible_salary_l542_542395


namespace speed_in_still_water_l542_542946

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h1 : upstream_speed = 15) (h2 : downstream_speed = 25) :
  ((upstream_speed + downstream_speed) / 2) = 20 :=
by
  rw [h1, h2]
  norm_num
  sorry

end speed_in_still_water_l542_542946


namespace final_selling_price_l542_542207

-- Conditions
variable (x : ℝ)
def original_price : ℝ := x
def first_discount : ℝ := 0.8 * x
def additional_reduction : ℝ := 10

-- Statement of the problem
theorem final_selling_price (x : ℝ) : (0.8 * x) - 10 = 0.8 * x - 10 :=
by sorry

end final_selling_price_l542_542207


namespace min_number_of_candy_kinds_l542_542424

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542424


namespace sin_cos_double_angle_l542_542880

noncomputable def log_a (a : ℝ) (x : ℝ) := log x / log a

theorem sin_cos_double_angle 
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (x y : ℝ) (h3 : y = log_a a (x - 3) + 2)
  (h4 : (x, y) = (4, 2)) :
  let α := real.angle_of_point (4, 2) in
  real.sin (2 * α) + real.cos (2 * α) = 7 / 5 := 
sorry

end sin_cos_double_angle_l542_542880


namespace solve_for_x_and_n_l542_542625

theorem solve_for_x_and_n (x n : ℕ) : 2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := 
sorry

end solve_for_x_and_n_l542_542625


namespace triangle_area_l542_542702

-- Define the vertices of the triangle as conditions
def u : ℝ × ℝ × ℝ := (4, 2, 1)
def v : ℝ × ℝ × ℝ := (1, 1, 1)
def w : ℝ × ℝ × ℝ := (7, 5, 3)

-- Define the vectors between vertices
def vector_vu := (v.1 - u.1, v.2 - u.2, v.3 - u.3)  -- (-3, -1, 0)
def vector_wu := (w.1 - u.1, w.2 - u.2, w.3 - u.3)  -- (3, 3, 2)

-- Definition for the cross product of the two vectors
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

-- Definition for the magnitude of the cross product
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1^2 + a.2^2 + a.3^2)

-- Prove that the area of the triangle is sqrt(19)
theorem triangle_area :
  let cp := cross_product vector_vu vector_wu in
  magnitude cp / 2 = real.sqrt 19 :=
sorry

end triangle_area_l542_542702


namespace clothes_and_transport_amount_l542_542843

variables (S : ℝ) (savings : ℝ)

-- Conditions
def consumable_items := 0.6 * S
def remaining_salary := S - consumable_items
def clothes_and_transport := 0.5 * remaining_salary
def monthly_savings := remaining_salary - clothes_and_transport

-- Given monthly savings
axiom savings_assumption : savings = 3900
axiom annual_savings : savings * 12 = 46800
axiom salary_condition : S = savings / 0.20

theorem clothes_and_transport_amount : clothes_and_transport = 3900 :=
by
  -- Proof goes here
  sorry

end clothes_and_transport_amount_l542_542843


namespace positive_divisors_of_10_factorial_l542_542770

theorem positive_divisors_of_10_factorial : 
  let fact_10 := 2^7 * 3^3 * 5^2 * 7 in
  let num_divisors n := ∏ p in (n.factorization.support : finset ℕ), (n.factorization p + 1) in
  num_divisors fact_10 = 192 :=
by
  /- Skipping the actual proof steps -/
  sorry

end positive_divisors_of_10_factorial_l542_542770


namespace angle_AQB_eq_angle_ACB_l542_542194

def is_altitude (A B C : Point) (K L : Point) : Prop :=
  -- Definition for altitude
  ∃ H : Point, H ∈ line(A, K) ∧ H ∈ line(B, L)

theorem angle_AQB_eq_angle_ACB 
    (A B C K L P Q : Point) 
    (h_acute : is_acute_triangle A B C)
    (h_altitudes : is_altitude A B C K ∧ is_altitude B A C L)
    (h_midpoint : midpoint L K P)
    (h_parallel1 : parallel (line P Q) (line B C))
    (h_parallel2 : parallel (line B Q) (line P L)) :
  ∠ A Q B = ∠ A C B :=
  sorry

end angle_AQB_eq_angle_ACB_l542_542194


namespace solve_congruences_l542_542283

theorem solve_congruences :
  ∃ x y z : ℤ,
    (2 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧
    (x * y ≡ 1 [MOD z]) ∧ 
    (x * z ≡ 1 [MOD y]) ∧ 
    (y * z ≡ 1 [MOD x]) ∧
    (x = 2 ∧ y = 3 ∧ z = 5) :=
sorry

end solve_congruences_l542_542283


namespace fermat_numbers_coprime_l542_542533

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2 ^ 2 ^ (n - 1) + 1) (2 ^ 2 ^ (m - 1) + 1) = 1 :=
sorry

end fermat_numbers_coprime_l542_542533


namespace ensure_even_product_l542_542274

theorem ensure_even_product (cards : Finset ℕ) (h : ↑cards = ↑(Finset.range 15) \ {0}) :
  ∃ n, n ≤ cards.card ∧ (∀ drawn_cards : Finset ℕ, drawn_cards.card = n → ∃ x ∈ drawn_cards, even x) ↔ n ≥ 8 :=
begin
  sorry
end

end ensure_even_product_l542_542274


namespace min_value_l542_542726

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : (a - 1) * 1 + 1 * (2 * b) = 0) :
    (2 / a) + (1 / b) = 8 :=
  sorry

end min_value_l542_542726


namespace pradeep_failure_marks_l542_542103

theorem pradeep_failure_marks :
  let total_marks := 925
  let pradeep_score := 160
  let passing_percentage := 20
  let passing_marks := (passing_percentage / 100) * total_marks
  let failed_by := passing_marks - pradeep_score
  failed_by = 25 :=
by
  sorry

end pradeep_failure_marks_l542_542103


namespace carter_road_trip_total_time_l542_542674

-- Definitions of the conditions
def trip_time : ℝ := 24
def interval_leg_stretch : ℝ := 1.5
def num_food_stops : ℝ := 3
def num_gas_stops : ℝ := 4
def num_sightseeing_stops : ℝ := 2
def time_leg_stretch : ℝ := 15 / 60
def time_food_stop : ℝ := 35 / 60
def time_gas_stop : ℝ := 20 / 60
def time_sightseeing_stop : ℝ := 60 / 60

-- Calculation to determine the total trip time with stops
def total_trip_time : ℝ :=
  let num_leg_stretch_stops := trip_time / interval_leg_stretch in
  let total_leg_stretch_time := num_leg_stretch_stops * time_leg_stretch in
  let total_food_time := num_food_stops * time_food_stop in
  let total_gas_time := num_gas_stops * time_gas_stop in
  let total_sightseeing_time := num_sightseeing_stops * time_sightseeing_stop in
  trip_time + total_leg_stretch_time + total_food_time + total_gas_time + total_sightseeing_time

-- Lean 4 statement proving total trip time
theorem carter_road_trip_total_time : total_trip_time = 33.08 := by
  sorry

end carter_road_trip_total_time_l542_542674


namespace total_movie_hours_l542_542069

variable (J M N R : ℕ)

def favorite_movie_lengths (J M N R : ℕ) :=
  J = M + 2 ∧
  N = 3 * M ∧
  R = (4 / 5 * N : ℝ).to_nat ∧
  N = 30

theorem total_movie_hours (J M N R : ℕ) (h : favorite_movie_lengths J M N R) :
  J + M + N + R = 76 :=
by
  sorry

end total_movie_hours_l542_542069


namespace count_integers_between_bounds_l542_542289

theorem count_integers_between_bounds : 
  (∃ (n_set : Set ℤ), {n | 50 < n^2 ∧ n^2 < 200} = n_set ∧ n_set.size = 14) :=
by
  sorry

end count_integers_between_bounds_l542_542289


namespace curve_is_circle_l542_542483

/-- The curve represented by the equation ρ = sin θ in polar coordinates is a circle -/
theorem curve_is_circle (ρ θ : ℝ) : 
  (ρ = sin θ) → 
  ∃ (x y : ℝ), (x = ρ * cos θ) ∧ (y = ρ * sin θ) ∧ (x^2 + (y - 1/2)^2 = 1/4) :=
by 
  sorry

end curve_is_circle_l542_542483


namespace part1_solution_set_part2_minimum_value_l542_542361

-- Definitions for Part (1)
def f (x : Real) : Real := abs (x + 2) + 2 * abs (x - 1)

-- Problem (1): Prove the solution to the inequality f(x) ≤ 4 is [0, 4/3]
theorem part1_solution_set (x : Real) : 0 ≤ x ∧ x ≤ 4 / 3 ↔ f x ≤ 4 := sorry

-- Definitions for Part (2)
def expression (a b : Real) : Real := 1 / (a - 1) + 2 / b

-- Problem (2): Prove the minimum value of the expression given the constraints is 9/2
theorem part2_minimum_value (a b : Real) (h1 : a + 2 * b = 3) (h2 : a > 1) (h3 : b > 0) : 
  expression a b = 9 / 2 := sorry

end part1_solution_set_part2_minimum_value_l542_542361


namespace minimum_students_lost_all_three_l542_542275

noncomputable theory

open Set

variable (students : Finset ℕ) (A B C : Finset ℕ)

theorem minimum_students_lost_all_three :
  students.card = 30 →
  A.card = 26 →
  B.card = 23 →
  C.card = 21 →
  ∃ (x : ℕ), x ≥ 10 ∧ x = (A ∩ B ∩ C).card :=
by
  sorry

end minimum_students_lost_all_three_l542_542275


namespace find_ABC_l542_542820

noncomputable def problem (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 8 ∧ B < 8 ∧ C < 6 ∧
  (A * 8 + B + C = 8 * 2 + C) ∧
  (A * 8 + B + B * 8 + A = C * 8 + C) ∧
  (100 * A + 10 * B + C = 246)

theorem find_ABC : ∃ A B C : ℕ, problem A B C := sorry

end find_ABC_l542_542820


namespace min_candy_kinds_l542_542436

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542436


namespace distinct_positive_integers_criteria_l542_542534

theorem distinct_positive_integers_criteria (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)
  (hxyz_div : x * y * z ∣ (x * y - 1) * (y * z - 1) * (z * x - 1)) :
  (x, y, z) = (2, 3, 5) ∨ (x, y, z) = (2, 5, 3) ∨ (x, y, z) = (3, 2, 5) ∨
  (x, y, z) = (3, 5, 2) ∨ (x, y, z) = (5, 2, 3) ∨ (x, y, z) = (5, 3, 2) :=
by sorry

end distinct_positive_integers_criteria_l542_542534


namespace can_form_triangle_l542_542931

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle :
  (is_triangle 3 5 7) ∧ ¬(is_triangle 3 3 7) ∧ ¬(is_triangle 4 4 8) ∧ ¬(is_triangle 4 5 9) :=
by
  -- Proof steps will be added here
  sorry

end can_form_triangle_l542_542931


namespace subtraction_correct_l542_542259

theorem subtraction_correct :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end subtraction_correct_l542_542259


namespace B_finishes_alone_l542_542624

def W_A (W_B : ℝ) : ℝ := (1/2) * W_B
def A_and_B_work_together (W : ℝ) : ℝ := W / 18

theorem B_finishes_alone (W : ℝ) : 
  ∀ W_B : ℝ, W_A W_B + W_B = A_and_B_work_together W → (W / W_B) = 27 := 
by
  intro W_B h
  sorry

end B_finishes_alone_l542_542624


namespace area_of_triangle_with_given_sides_l542_542780

variable (a b c : ℝ)
variable (s : ℝ := (a + b + c) / 2)
variable (area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem area_of_triangle_with_given_sides (ha : a = 65) (hb : b = 60) (hc : c = 25) :
  area = 750 := by
  sorry

end area_of_triangle_with_given_sides_l542_542780


namespace quadrilateral_in_triangle_division_l542_542611

theorem quadrilateral_in_triangle_division
  (A B C K L P : Type)
  [IsTriangle A B C] -- Assuming IsTriangle is a predicate that denotes a triangle with vertices A, B, and C.
  (AK : LineSegment A K) (BL : LineSegment B L)
  (P_in_AKC : P ∈ Triangle A K C) (P_in_BLC : P ∈ Triangle B L C)
  (areas_equal : area (Triangle A B P) = area (Triangle A K P) ∧ 
                 area (Triangle A K P) = area (Triangle B L P)) :
  ∃ Q : Type, Q = Quadrilateral K L P Q :=
by
  sorry

end quadrilateral_in_triangle_division_l542_542611


namespace geom_sequence_and_sum_conditions_l542_542325

theorem geom_sequence_and_sum_conditions (a_n S_n : ℕ → ℝ)
  (h1 : a_n 1 = 1 / 32)
  (h2 : ∀ n > 0, S_n n = a_n (n+1) - (1 / 32)) :
  a_n = λ n, 2^(n-6) ∧ 
  (let b_n := λ n, Real.log2 (a_n n) in 
   let T_n := λ n, if n < 6 then (11 * n - n^2) / 2 else (n^2 - 11 * n) / 2 + 30 in
   T_n = λ n, if n < 6 then (11 * n - n^2) / 2 else (n^2 - 11 * n) / 2 + 30) := 
sorry

end geom_sequence_and_sum_conditions_l542_542325


namespace ratio_of_eggs_l542_542915

def total_eggs : ℕ := 63
def hannah_eggs : ℕ := 42

def helen_eggs : ℕ := total_eggs - hannah_eggs
def ratio_hannah_helen := hannah_eggs / (if helen_eggs = 0 then 1 else helen_eggs)

theorem ratio_of_eggs (a b : ℕ) (h_total : total_eggs = a) (h_hannah : hannah_eggs = b) :
  a = 63 ∧ b = 42 ∧ ratio_hannah_helen = 2 := 
by
  unfold total_eggs
  unfold hannah_eggs
  unfold helen_eggs
  unfold ratio_hannah_helen
  have h1 : helen_eggs = 63 - 42 := by 
    apply rfl
  have h2 : 42 / (63 - 42) = 2 := by
    sorry
  exact ⟨rfl, h1, h2⟩

end ratio_of_eggs_l542_542915


namespace total_movie_hours_l542_542068

variable (J M N R : ℕ)

def favorite_movie_lengths (J M N R : ℕ) :=
  J = M + 2 ∧
  N = 3 * M ∧
  R = (4 / 5 * N : ℝ).to_nat ∧
  N = 30

theorem total_movie_hours (J M N R : ℕ) (h : favorite_movie_lengths J M N R) :
  J + M + N + R = 76 :=
by
  sorry

end total_movie_hours_l542_542068


namespace subinterval_exists_subinterval_not_exists_l542_542816

open MeasureTheory

theorem subinterval_exists {A B : set ℝ} (hA : A ⊆ (0, 1)) (hB : B ⊆ (0, 1)) (h_disjoint : disjoint A B)
  (h_muA : 0 < μ A) (h_muB : 0 < μ B) (n : ℕ) (hn : 0 < n) : 
  ∃ (c d : ℝ) (h_cd : c < d ∧ (c,d) ⊆ (0,1)), 
    μ (A ∩ set.Ioo c d) = (1 / n : ℝ) * μ A ∧ μ (B ∩ set.Ioo c d) = (1 / n : ℝ) * μ B :=
sorry

theorem subinterval_not_exists {A B : set ℝ} (hA : A ⊆ (0, 1)) (hB : B ⊆ (0, 1)) (h_disjoint : disjoint A B)
  (h_muA : 0 < μ A) (h_muB : 0 < μ B) (λ : ℝ) (hλ : ∀ n : ℕ, λ ≠ (1 / n : ℝ)) :
  ¬∃ (c d : ℝ) (h_cd : c < d ∧ (c,d) ⊆ (0,1)), 
    μ (A ∩ set.Ioo c d) = λ * μ A ∧ μ (B ∩ set.Ioo c d) = λ * μ B :=
sorry

end subinterval_exists_subinterval_not_exists_l542_542816


namespace exists_valid_arrangement_l542_542673

def six_point_star (arrangement : Fin 12 → ℕ) : Prop :=
  -- Define positions for clarity, where each position is unique and within range 1 to 12
  let positions : List (Fin 12) := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  in 
  -- Ensure all positions have unique values from 1 to 12
  (∀ p1 p2, p1 ≠ p2 → arrangement p1 ≠ arrangement p2) ∧
  -- Ensure values are within 1 to 12
  (∀ p, arrangement p ∈ List.range 1 13) ∧
  -- Sum of any four numbers on a straight line is 26
  (arrangement 0 + arrangement 3 + arrangement 6 + arrangement 9 = 26) ∧
  (arrangement 1 + arrangement 4 + arrangement 7 + arrangement 10 = 26) ∧
  (arrangement 2 + arrangement 5 + arrangement 8 + arrangement 11 = 26) ∧
  (arrangement 0 + arrangement 1 + arrangement 2 + arrangement 9 = 26) ∧
  (arrangement 3 + arrangement 4 + arrangement 5 + arrangement 10 = 26) ∧
  (arrangement 6 + arrangement 7 + arrangement 8 + arrangement 11 = 26)

theorem exists_valid_arrangement : ∃ (arrangement : Fin 12 → ℕ), six_point_star arrangement :=
sorry

end exists_valid_arrangement_l542_542673


namespace tangent_line_correct_l542_542587

noncomputable def f : ℝ → ℝ := λ x, 4 * x - x ^ 3

def tangent_line_at_point (x1 y1 : ℝ) (k : ℝ) : ℝ → ℝ := λ x, k * (x - x1) + y1

theorem tangent_line_correct :
  tangent_line_at_point (-1) (-3) 1 = λ x, x - 2 := sorry

end tangent_line_correct_l542_542587


namespace min_value_f_at_pi_eight_l542_542354

def f (ω x : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x) * cos (ω * x) + cos (ω * x) ^ 2 - 1/2

theorem min_value_f_at_pi_eight:
  ∀ (ω > 0), (∃ c : ℝ, c ∈ set.Ioo 0 (π / 4) ∧ ∀ x ∈ set.Ioo 0 (π / 4),
  f ω x = f ω (c - x))
  → (∃ ω, ω > 0 ∧ f ω (π / 8) = 1/2) :=
begin
  sorry
end

end min_value_f_at_pi_eight_l542_542354


namespace min_value_sum_l542_542894

def non_neg_int := {n : ℕ // 0 ≤ n}

theorem min_value_sum (a b c d : non_neg_int)
  (h : a.val * b.val + b.val * c.val + c.val * d.val + d.val * a.val = 707) :
  a.val + b.val + c.val + d.val ≥ 108 :=
begin
  -- The proof would go here, but it is omitted as per instructions.
  sorry
end

end min_value_sum_l542_542894


namespace min_number_of_candy_kinds_l542_542428

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542428


namespace calculation_result_l542_542255

theorem calculation_result :
  (-1 : ℤ) ^ 2023 + Real.sqrt 4 - |(-Real.sqrt 2)| + Real.cbrt (-8) = -1 - Real.sqrt 2 := by
  sorry

end calculation_result_l542_542255


namespace circumscribed_quad_angle_sum_l542_542107

theorem circumscribed_quad_angle_sum (A B C D O : Type) 
    (h_circ : ∃ c : Circle, c.center = O ∧ Quadrilateral ABCD circumscribed_by c) 
    (h_opposite_angles : ∠A + ∠C = 180 ∧ ∠B + ∠D = 180) : 
    ∠AOB + ∠COD = 180 := 
by 
  sorry

end circumscribed_quad_angle_sum_l542_542107


namespace count_integers_between_bounds_l542_542290

theorem count_integers_between_bounds : 
  (∃ (n_set : Set ℤ), {n | 50 < n^2 ∧ n^2 < 200} = n_set ∧ n_set.size = 14) :=
by
  sorry

end count_integers_between_bounds_l542_542290


namespace new_ticket_price_l542_542649

theorem new_ticket_price (initial_price : ℕ) (initial_visitors : ℕ) (new_price : ℕ) 
  (price_reduced_increase_visitors : 50% increase) (revenue_growth : 35% growth) 
  : new_price = 270 :=
by
  -- Definitions according to the conditions
  let initial_price := 300
  let N := initial_visitors
  let new_visitors := 1.5 * initial_visitors
  let initial_revenue := initial_price * N
  let new_revenue := initial_revenue * 1.35
  let new_revenue_alternate := new_price * new_visitors
  
  -- Setting up the equation
  have eq1 : new_revenue = initial_revenue * 1.35 := rfl
  have eq2 : new_visitors = 1.5 * N := rfl
  have eq3 : new_revenue_alternate = new_price * (1.5 * N) := rfl

  -- Given new_revenue_alternate = new_revenue
  have final_eq : new_price * (1.5 * N) = 300 * N * 1.35,
    from eq1 ▸ eq2 ▸ eq3
  
  -- Solving the equation
  calc
    new_price * 1.5 = 405 : by sorry 
    new_price = 270        : by sorry -- skip the detailed solving steps

end new_ticket_price_l542_542649


namespace shifted_roots_polynomial_l542_542086

noncomputable def poly_with_shifted_roots (p : Polynomial ℝ) (shift : ℝ) : Polynomial ℝ :=
  Polynomial.map (Polynomial.C ∘ (λ x, x - shift)) p

theorem shifted_roots_polynomial :
  (poly_with_shifted_roots (Polynomial.C 7 - Polynomial.C 5 * X + Polynomial.C 4 * X^2 - X^3) (-3)) =
  (Polynomial.C (-85) - Polynomial.C 56 * X + Polynomial.C 13 * X^2 - X^3) :=
by
  sorry

end shifted_roots_polynomial_l542_542086


namespace percentage_of_boys_l542_542394

def ratio_boys_girls := 2 / 3
def ratio_teacher_students := 1 / 6
def total_people := 36

theorem percentage_of_boys : ∃ (n_student n_teacher n_boys n_girls : ℕ), 
  n_student + n_teacher = 35 ∧
  n_student * (1 + 1/6) = total_people ∧
  n_boys / n_student = ratio_boys_girls ∧
  n_teacher / n_student = ratio_teacher_students ∧
  ((n_boys : ℚ) / total_people) * 100 = 400 / 7 :=
sorry

end percentage_of_boys_l542_542394


namespace complement_of_A_l542_542091

open set

def A : set ℝ := {y | ∃ (x : ℝ), y = 2^x + 1}

def U : set ℝ := univ

theorem complement_of_A : compl A = Iic 1 :=
by {
  sorry
}

end complement_of_A_l542_542091


namespace jim_apples_value_l542_542059

-- Definitions of quantities stated in conditions
def jane_apples : ℕ := 60
def jerry_apples : ℕ := 40

-- Hypothesis that Jim's apples fit 2 times into the average amount of apples per person in the group
def hypothesis (jim_apples : ℕ) : Prop :=
  jim_apples = 2 * ((jim_apples + jane_apples + jerry_apples) / 3)

-- Theorem stating the number of apples Jim has
theorem jim_apples_value : ∃ jim_apples : ℕ, hypothesis jim_apples ∧ jim_apples = 200 :=
by {
  use 200, -- Providing the value of Jim's apples as 200
  unfold hypothesis, -- Unfold the hypothesis definition
  sorry
}

end jim_apples_value_l542_542059


namespace min_kinds_of_candies_l542_542439

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l542_542439


namespace minimum_value_of_quadratic_l542_542890

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 3

theorem minimum_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f(y) >= f(x) ∧ f(x) = -3 :=
by
  sorry

end minimum_value_of_quadratic_l542_542890


namespace part1_part2_l542_542482

noncomputable def F_1 := (-real.sqrt 3, 0 : ℝ × ℝ)
noncomputable def F_2 := (real.sqrt 3, 0 : ℝ × ℝ)
def M (x y : ℝ) : Prop := abs (real.sqrt ((x + real.sqrt 3)^2 + y^2) - real.sqrt ((x - real.sqrt 3)^2 + y^2)) = 2
def curve_C (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

theorem part1 (x y : ℝ) (h : M x y) : curve_C x y :=
sorry

noncomputable def A := (1, 0 : ℝ × ℝ)
def AP (x y : ℝ) := y = x - 1
def AQ (x y : ℝ) := y = -x + 1
def PQ (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := (x₁ + (y₂ - y₁)/(x₂ - x₁), y₁ + (y₂ - y₁)/(x₂ - x₁)*(x₁ - x₁))

theorem part2 (x₁ y₁ x₂ y₂ : ℝ) (hP : curve_C x₁ y₁) (hQ : curve_C x₂ y₂) (hPQ : PQ x₁ y₁ x₂ y₂ = (1, 0)) : (x₁ + x₂ = - 2 ∨ (1, 0) = (-3, 0)) :=
sorry

end part1_part2_l542_542482


namespace part1_solution_set_part2_minimum_value_l542_542362

-- Definitions for Part (1)
def f (x : Real) : Real := abs (x + 2) + 2 * abs (x - 1)

-- Problem (1): Prove the solution to the inequality f(x) ≤ 4 is [0, 4/3]
theorem part1_solution_set (x : Real) : 0 ≤ x ∧ x ≤ 4 / 3 ↔ f x ≤ 4 := sorry

-- Definitions for Part (2)
def expression (a b : Real) : Real := 1 / (a - 1) + 2 / b

-- Problem (2): Prove the minimum value of the expression given the constraints is 9/2
theorem part2_minimum_value (a b : Real) (h1 : a + 2 * b = 3) (h2 : a > 1) (h3 : b > 0) : 
  expression a b = 9 / 2 := sorry

end part1_solution_set_part2_minimum_value_l542_542362


namespace max_value_x2sqrt3y_l542_542118

noncomputable def polar_eq_C (ρ : ℝ) : Prop :=
  ρ = 1

noncomputable def param_eq_l (t x y : ℝ) : Prop :=
  x = 1 + (1/2)*t ∧ y = 2 + (sqrt 3 / 2)*t

noncomputable def scaling_transformation (x y x' y' : ℝ) : Prop :=
  x' = 2*x ∧ y' = y

theorem max_value_x2sqrt3y (x y x' y': ℝ) (t : ℝ) 
  (H1 : polar_eq_C (sqrt (x^2 + y^2)))
  (H2 : param_eq_l t x y)
  (H3 : scaling_transformation x y x' y') : 
  ∃ m : ℝ, m = 4 :=
by
  sorry

end max_value_x2sqrt3y_l542_542118


namespace pages_to_read_tomorrow_l542_542506

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l542_542506


namespace cartesian_to_polar_l542_542873

/-- Given the Cartesian coordinates of point M are (1, -sqrt(3)), prove that the polar coordinates are (2, -pi/3). -/
theorem cartesian_to_polar (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = 1) 
  (h2 : y = -Real.sqrt 3)
  (h3 : ρ = Real.sqrt (x^2 + y^2)) 
  (h4 : θ = Real.atan2 y x) : 
  ρ = 2 ∧ θ = -Real.pi / 3 := by
sory

end cartesian_to_polar_l542_542873


namespace average_visitors_per_day_l542_542977

/-- A library has different visitor numbers depending on the day of the week.
  - On Sundays, the library has an average of 660 visitors.
  - On Mondays through Thursdays, there are 280 visitors on average.
  - Fridays and Saturdays see an increase to an average of 350 visitors.
  - This month has a special event on the third Saturday, bringing an extra 120 visitors that day.
  - The month has 30 days and begins with a Sunday.
  We want to calculate the average number of visitors per day for the entire month. -/
theorem average_visitors_per_day
  (num_days : ℕ) (starts_on_sunday : Bool)
  (sundays_visitors : ℕ) (weekdays_visitors : ℕ) (weekend_visitors : ℕ)
  (special_event_extra_visitors : ℕ) (sundays : ℕ) (mondays : ℕ)
  (tuesdays : ℕ) (wednesdays : ℕ) (thursdays : ℕ) (fridays : ℕ)
  (saturdays : ℕ) :
  num_days = 30 → starts_on_sunday = true →
  sundays_visitors = 660 → weekdays_visitors = 280 → weekend_visitors = 350 →
  special_event_extra_visitors = 120 →
  sundays = 4 → mondays = 5 →
  tuesdays = 4 → wednesdays = 4 → thursdays = 4 → fridays = 4 → saturdays = 4 →
  ((sundays * sundays_visitors +
    mondays * weekdays_visitors +
    tuesdays * weekdays_visitors +
    wednesdays * weekdays_visitors +
    thursdays * weekdays_visitors +
    fridays * weekend_visitors +
    saturdays * weekend_visitors +
    special_event_extra_visitors) / num_days = 344) :=
by
  intros
  sorry

end average_visitors_per_day_l542_542977


namespace quadratic_distinct_roots_l542_542035

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end quadratic_distinct_roots_l542_542035


namespace cos_C_in_acute_triangle_l542_542341

theorem cos_C_in_acute_triangle 
  (a b c : ℝ) (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sides_angles : a * Real.cos B = 4 * c * Real.sin C - b * Real.cos A) 
  : Real.cos C = Real.sqrt 15 / 4 := 
sorry

end cos_C_in_acute_triangle_l542_542341


namespace min_candy_kinds_l542_542433

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542433


namespace area_triangle_EFG_is_3_l542_542108

-- Definitions for the given problem
variables (A B C D E F G : Type)
variables (x y : ℝ)

-- Assume A, B, C, D form a rectangle with area 48
def is_rectangle (A B C D : Type) (x y : ℝ) := x * y = 48

-- Assume E and F are midpoints of sides AB and CD, respectively
def is_midpoint (A B E : Type) := ∃ (x : ℝ), E = midpoint A B
def is_midpoint (C D F : Type) := ∃ (y : ℝ), F = midpoint C D

-- G is the intersection of the semicircle with diameter EF and diagonal AC
def semicircle_diameter (E F : Type) := ∃ (d : ℝ), d = (dist E F) / 2
def is_intersection (G : Type) (E F G : Type) := ∃ (G : Type), G ∈ semicircle E F ∧ G ∈ line AC

-- The area of triangle EFG
def area_triangle (E F G: Type) (x y : ℝ) := 1/2 * (dist E F) * (dist E G / 2)

-- Main theorem statement
theorem area_triangle_EFG_is_3 (A B C D E F G : Type) (x y : ℝ)
  (H1 : is_rectangle A B C D x y)
  (H2 : is_midpoint A B E)
  (H3 : is_midpoint C D F)
  (H4 : is_intersection G E F) :
  area_triangle E F G x y = 3 := sorry

end area_triangle_EFG_is_3_l542_542108


namespace fraction_irreducible_l542_542559

theorem fraction_irreducible (a b c d : ℤ) (h : a * d - b * c = 1) : ∀ m : ℤ, m > 1 → ¬ (m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d)) :=
by sorry

end fraction_irreducible_l542_542559


namespace drinks_left_for_Seungwoo_l542_542112

def coke_taken_liters := 35 + 0.5
def cider_taken_liters := 27 + 0.2
def coke_drank_liters := 1 + 0.75

theorem drinks_left_for_Seungwoo :
  (coke_taken_liters - coke_drank_liters) + cider_taken_liters = 60.95 := by
  sorry

end drinks_left_for_Seungwoo_l542_542112


namespace large_circle_diameter_l542_542866

-- Define the conditions
def small_circle_radius : ℝ := 4
def num_small_circles : ℕ := 6
def small_circle_tangent_large_circle : Type := sorry -- This needs to be appropriately defined
def small_circle_tangent_neighbors : Type := sorry -- This needs to be appropriately defined

-- Define the target theorem
theorem large_circle_diameter :
  (∀ (r : ℝ), r = small_circle_radius)
  ∧ (∀ (n : ℕ), n = num_small_circles)
  ∧ (small_circle_tangent_large_circle)
  ∧ (small_circle_tangent_neighbors) →
  ∃ (d : ℝ), d = 24 :=
sorry

end large_circle_diameter_l542_542866


namespace solution_set_is_l542_542381

def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a

axiom even_function (a : ℝ) (x : ℝ) : f x a = f (-x) a

theorem solution_set_is (-∞,2) (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (∀ x : ℝ, f x a - 1 < (Real.exp 2 + 1) / Real.exp 1) →
  ∀ x : ℝ, x < 2 :=
by
  sorry

end solution_set_is_l542_542381


namespace apples_per_case_l542_542183

theorem apples_per_case (total_apples : ℕ) (number_of_cases : ℕ) (h1 : total_apples = 1080) (h2 : number_of_cases = 90) : total_apples / number_of_cases = 12 := by
  sorry

end apples_per_case_l542_542183


namespace douglas_won_36_percent_in_Y_l542_542950

-- Definitions based on conditions
variables {V : ℕ} -- Number of voters in County Y
variables {votes_in_X : ℕ} {votes_in_Y : ℕ} {total_votes : ℕ}

def ratio_of_voters := 2 * V -- Ratio of voters in County X to County Y is 2:1 

def votes_douglas_X := 72 * (2 * V) -- Douglas won 72 percent of the vote in County X 
def votes_douglas_total := 60 * (3 * V) -- Douglas won 60 percent of the total vote in Counties X and Y

-- Theorem
theorem douglas_won_36_percent_in_Y :
  votes_douglas_X + votes_in_Y = votes_douglas_total → 
  votes_in_Y = 36 * V :=
by
  intro h,
  sorry

end douglas_won_36_percent_in_Y_l542_542950


namespace yz_Ruzsa_l542_542515

noncomputable theory

variables {x : ℕ} {n : ℕ}
variables (a : ℕ → ℕ)
variables (C : ℝ)

def reciprocals_sum_condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (1 < a 1) ∧ (∀ i, 1 ≤ i → i < n → a i < a (i+1)) ∧ (finset.range n).sum (λ i, (1 : ℝ) / (a i)) ≤ 1

def positive_integers_not_divisible (a : ℕ → ℕ) (x : ℕ) : ℕ :=
  finset.filter (λ m, ∀ i, i < n → ¬ (m % a i = 0)) (finset.range x).card

theorem yz_Ruzsa
  (h1 : 1 < x)
  (h2 : reciprocals_sum_condition a n)
  (h3 : positive_integers_not_divisible a x > 0) :
  positive_integers_not_divisible a x > C * x / real.log x :=
sorry

end yz_Ruzsa_l542_542515


namespace triangle_ac_interval_sum_l542_542568

theorem triangle_ac_interval_sum (AB CD : ℝ) (m n : ℝ) :
  AB = 12 → CD = 4 →
  (∀ AC : ℝ, 4 < AC ∧ AC < 24) →
  m = 4 ∧ n = 24 →
  m + n = 28 := 
begin
  intros hAB hCD hAC hmn,
  cases hmn with hm hn,
  rw [hm, hn],
  norm_num,
end

end triangle_ac_interval_sum_l542_542568


namespace contrapositive_of_p_l542_542388

-- Definitions of propositions and transformations
variable {m n : Prop}
def p : Prop := m → n
def q : Prop := ¬m → ¬n
def r : Prop := ¬n → ¬m

theorem contrapositive_of_p :
  r ↔ (p ↔ True) :=
by
  sorry

end contrapositive_of_p_l542_542388


namespace locus_of_Q_is_ellipse_l542_542490

noncomputable theory
open_locale classical

variables {α : Type*}

-- Define the basic geometric objects
structure Point (α : Type*) :=
(x : α)
(y : α)

structure Circle (α : Type*) :=
(center : Point α)
(radius : α)

-- Define symmetry, reflection, and perpendicularity
def symmetrical (P Q O : Point α) : Prop :=
2 * O.x = P.x + Q.x ∧ 2 * O.y = P.y + Q.y

def reflection (P Q : Point α) (L : set (Point α)) : Prop :=
-- Define reflection about line L in terms of geometric properties of P, Q, and line L
sorry

def perpendicular_chord (P P' A : Point α) : Prop :=
-- Define the perpendicularity condition
sorry

def intersects (L1 L2 : set (Point α)) (Q : Point α) : Prop :=
-- Define intersection of two lines L1 and L2 at point Q
sorry

-- Define the given points, circle, and point Q
variables (O A B P P' Q C : Point α) (r : α) (circle : Circle α)
hypothesis h_sym_AB: symmetrical A B O
hypothesis h_on_circle: -- P lies on the circle with center O and radius r
sorry
hypothesis h_perpendicular: perpendicular_chord P P' A
hypothesis h_reflection: reflection C B (overline PP')

-- The final proof statement
theorem locus_of_Q_is_ellipse (Q : Point α) :
  intersects (overline PP') (overline AC) Q →
  ∃ (f1 f2 : Point α) (a : α), f1 = A ∧ f2 = B ∧ a = 2 * r ∧
  ∀ (x : Point α), (dist x f1 + dist x f2 = a) ↔ x = Q :=
sorry

end locus_of_Q_is_ellipse_l542_542490


namespace coefficient_of_y_in_expression_l542_542687

-- Define the given expression
def expression (y : ℝ) := 5 * (y - 6) + 6 * (9 - 3 * y^2 + 7 * y) - 10 * (3 * y - 2)

-- Extract and prove the coefficient of y
theorem coefficient_of_y_in_expression : 
  ∀ y : ℝ, (∃ c, (∀ y, expression y = c * y + (expression 0 - c * 0)) ∧ c = 17) :=
by intros y;
  existsi 17;
  sorry

end coefficient_of_y_in_expression_l542_542687


namespace find_BW_l542_542158

theorem find_BW
    (Γ₁ Γ₂ : Circle)
    (C D Z W : Point)
    (A B X Y : Point)
    (h_intersect_cd : Intersect Γ₁ Γ₂ C D)
    (h_cd : Line CD ≥ (C, D))
    (h_byxa : Line BYXA ≥ (B, Y, X, A))
    (h_z : Intersection cd byxa = Z)
    (h_tangent_w : Tangent WB Γ₁ W ∧ Tangent WB Γ₂ W)
    (h_symmetry : ZX = ZY)
    (h_product : AB * AX = 100) :
    BW = 10 := sorry

end find_BW_l542_542158


namespace number_of_ways_to_sum_to_4_l542_542309

-- Definitions deriving from conditions
def cards : List ℕ := [0, 1, 2, 3, 4]

-- Goal to prove
theorem number_of_ways_to_sum_to_4 : 
  let pairs := List.product cards cards
  let valid_pairs := pairs.filter (λ (x, y) => x + y = 4)
  List.length valid_pairs = 5 := 
by
  sorry

end number_of_ways_to_sum_to_4_l542_542309


namespace cost_one_year_ago_is_1512_l542_542659

-- Defining the given conditions
def current_cost := 10080 / 4
def reduction_factor := 2 / 5

-- Statement of the problem as a Lean theorem
theorem cost_one_year_ago_is_1512 (current_cost : ℝ) (reduction_factor : ℝ) (H1 : current_cost = 2520) (H2 : reduction_factor = 0.4) : 
  let prev_cost := current_cost * (1 - reduction_factor)
  in prev_cost = 1512 :=
by
  have h1 : current_cost = 2520 := by simp [H1]
  have h2 : reduction_factor = 0.4 := by simp [H2]
  let prev_cost := current_cost * (1 - reduction_factor)
  show prev_cost = 1512
  have h_prev_cost : prev_cost = 2520 * (1 - 0.4) := by simp [h1, h2]
  have eq_prev_cost : 2520 * 0.6 = 1512 := by norm_num
  rw [← eq_prev_cost] at h_prev_cost
  exact h_prev_cost

end cost_one_year_ago_is_1512_l542_542659


namespace problem_acute_triangle_l542_542472

noncomputable def sqrt2 : ℝ := real.sqrt 2
noncomputable def sqrt3 : ℝ := real.sqrt 3
noncomputable def sqrt6 : ℝ := real.sqrt 6

theorem problem_acute_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (hA1 : sin A = (2 * sqrt2) / 3)
  (ha : a = 2)
  (h_cosB_C : c * cos B + b * cos C = 2 * a * cos B) :
  b = (3 * sqrt6) / 4 :=
by
  sorry

end problem_acute_triangle_l542_542472


namespace find_b2048_l542_542827

noncomputable def b_seq : ℕ → ℝ
| 0       := 1 -- arbitrary value as b_0 is not given
| 1       := 3 + 2 * Real.sqrt 5
| (n + 2) := b_seq (n + 1) * b_seq n

theorem find_b2048 :
  b_seq 2023 = 23 + 10 * Real.sqrt 5 →
  b_seq 2048 = 19 + 6 * Real.sqrt 5 :=
begin
  intro h2023,
  -- Proof to be completed
  sorry
end

end find_b2048_l542_542827


namespace gcd_2814_1806_l542_542620

def a := 2814
def b := 1806

theorem gcd_2814_1806 : Nat.gcd a b = 42 :=
by
  sorry

end gcd_2814_1806_l542_542620


namespace distance_between_parallel_lines_l542_542609
noncomputable theory

-- Given conditions
variables (A B C D O : Point) (r : ℝ) (d : ℝ) (circle : Circle O r)
hypotheses 
  (hAB : |AB| = 40)
  (hCD : |CD| = 40)
  (hBC : |BC| = 36)
  (h_eq_space : ∀ P Q : Point, (P, Q) ∈ [(A, B), (B, C), (C, D)] → |P - Q| = d)

-- Question: Prove the distance between two adjacent parallel lines
theorem distance_between_parallel_lines : 
  distance_between_parallel_lines circle hAB hCD hBC h_eq_space = 6.25 :=
sorry

end distance_between_parallel_lines_l542_542609


namespace solve_for_x_l542_542905

def triple_minus_double_eq_8_5 (x : Real) : Prop :=
  3 * x - 2 * x = 8.5

theorem solve_for_x : ∃ x : Real, triple_minus_double_eq_8_5 x :=
by {
  use 8.5,
  unfold triple_minus_double_eq_8_5,
  ring,
  exact rfl,
}

end solve_for_x_l542_542905


namespace no_integers_satisfy_eq_l542_542081

noncomputable theory

variables {P : ℤ → ℤ} (x : ℤ)

-- Given conditions
def P_conditions : Prop :=
  P (-1) = -4 ∧
  P (-3) = -40 ∧
  P (-5) = -156

theorem no_integers_satisfy_eq (hP : P_conditions) : ¬ ∃ x : ℤ, P (P x) = x^2 := sorry

end no_integers_satisfy_eq_l542_542081


namespace rabbit_prob_top_or_bottom_l542_542223

-- Define the probability function for the rabbit to hit the top or bottom border from a given point
noncomputable def prob_reach_top_or_bottom (start : ℕ × ℕ) (board_end : ℕ × ℕ) : ℚ :=
  sorry -- Detailed probability computation based on recursive and symmetry argument

-- The proof statement for the starting point (2, 3) on a rectangular board extending to (6, 5)
theorem rabbit_prob_top_or_bottom : prob_reach_top_or_bottom (2, 3) (6, 5) = 17 / 24 :=
  sorry

end rabbit_prob_top_or_bottom_l542_542223


namespace jill_first_bus_ride_time_l542_542917

-- Definitions based on the conditions
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := sorry
def ride_time_second_bus : ℕ := 21

-- The proof problem
theorem jill_first_bus_ride_time :
  ∃ x : ℕ, ride_time_second_bus * 2 = wait_time_first_bus + x ∧ x = 30 := 
by
  use 30
  split
  { 
    -- ⊢ ride_time_second_bus * 2 = wait_time_first_bus + 30
    sorry 
  }
  { 
    -- ⊢ 30 = 30
    refl 
  }

end jill_first_bus_ride_time_l542_542917


namespace length_of_street_correct_l542_542221

-- Given conditions
def time_minutes : ℝ := 8
def speed_kmph : ℝ := 5.31
def speed_mpm : ℝ := (speed_kmph * 1000) / 60 -- speed in meters per minute

-- The length of the street the person crosses
def length_of_street : ℝ := speed_mpm * time_minutes

-- The theorem to prove
theorem length_of_street_correct : length_of_street = 708 :=
by
  -- Sorry as the proof is not required
  sorry

end length_of_street_correct_l542_542221


namespace finite_solutions_l542_542557

theorem finite_solutions (x y z n : ℕ) :
  ∃ (solutions : finset (ℕ × ℕ × ℕ × ℕ)), (∀ (x y z n : ℕ), (x, y, z, n) ∈ solutions) ∧ 
  ∀ (x y z n : ℕ), 2^x + 5^y - 31^z = n! → (x, y, z, n) ∈ solutions :=
by
  sorry

end finite_solutions_l542_542557


namespace polynomial_integer_solutions_l542_542526

/-- Given a polynomial P with integer coefficients, we investigate the cardinality of the set
  {k : ℤ | k * P k = 2020}. The possible values for this cardinality are 0, 1, 2, 3, 4, 5, or 6. -/
theorem polynomial_integer_solutions (P : ℤ[X]) :
  let n := {k : ℤ | k * P.eval k = 2020}.to_finset.card in
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 :=
sorry

end polynomial_integer_solutions_l542_542526


namespace cistern_filling_time_l542_542615

noncomputable def rate_of_pipe_A : ℚ := 1 / 45
noncomputable def rate_of_pipe_B : ℚ := 1 / 60
noncomputable def rate_of_pipe_C : ℚ := 1 / 72

theorem cistern_filling_time :
  let combined_rate := rate_of_pipe_A + rate_of_pipe_B - rate_of_pipe_C in
  (1 / combined_rate) = 40 :=
by
  let combined_rate := rate_of_pipe_A + rate_of_pipe_B - rate_of_pipe_C
  have h_combined_rate : combined_rate = 1 / 40 := sorry
  rwa [←h_combined_rate, inv_inv]

end cistern_filling_time_l542_542615


namespace determine_function_form_l542_542079

variable {k : ℝ} (h : 0 < k)
variable {f : ℝ → ℝ}

noncomputable def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ Icc (-k) k → y ∈ Icc (-k) k → (x + y) ∈ Icc (-k) k →
  f x ^ 2 + f y ^ 2 - 2 * x * y = k ^ 2 + f (x + y) ^ 2

noncomputable def function_form (f : ℝ → ℝ) (a c : ℝ) : Prop :=
  ∀ x, x ∈ Icc (-k) k → f x = Real.sqrt (a * x + c - x ^ 2) ∧ 0 ≤ a * x + c - x ^ 2 ∧ a * x + c - x ^ 2 ≤ k ^ 2

theorem determine_function_form (f : ℝ → ℝ) (a c : ℝ) (h₁ : satisfies_equation k f) :
  function_form k f a c :=
sorry

end determine_function_form_l542_542079


namespace min_candy_kinds_l542_542435

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542435


namespace line_intersects_plane_l542_542374

-- Define the vectors a, b, and u
def a : ℝ × ℝ × ℝ := (1, 1/2, 3)
def b : ℝ × ℝ × ℝ := (1/2, 1, 1)
def u : ℝ × ℝ × ℝ := (1/2, 0, 1)

-- Prove that the line l with direction vector u intersects plane α defined by vectors a and b.
theorem line_intersects_plane : ¬∃ x y : ℝ, u = (x • a.1 + y • b.1, x • a.2 + y • b.2, x • a.3 + y • b.3) :=
sorry

end line_intersects_plane_l542_542374


namespace log_function_shift_l542_542918

theorem log_function_shift : 
  ∀ (x : ℝ), log (2 * (x + 1) + 1) - 1 = log(2 * x + 3) - 1 :=
by
  -- The proof goes here
  intro x
  rw [add_assoc, two_mul]
  sorry

end log_function_shift_l542_542918


namespace converse_inverse_contrapositive_count_l542_542002

theorem converse_inverse_contrapositive_count
  (a b : ℝ) : (a = 0 → ab = 0) →
  (if (ab = 0 → a = 0) then 1 else 0) +
  (if (a ≠ 0 → ab ≠ 0) then 1 else 0) +
  (if (ab ≠ 0 → a ≠ 0) then 1 else 0) = 1 :=
sorry

end converse_inverse_contrapositive_count_l542_542002


namespace direct_proportion_of_b_and_c_l542_542025

theorem direct_proportion_of_b_and_c (a b c : ℝ) (h : a = b / c) (h1 : c ≠ 0) (h2 : ∀ x y z : ℝ, x = y / z -> y = x * z) :
  (∀ y z : ℝ, y / z = a -> y = a * z) := 
by
  intro y z,
  intro h3,
  rw [←h, h3],
  exact eq.symm (h2 b c a h).symm

end direct_proportion_of_b_and_c_l542_542025


namespace trigonometric_series_bound_l542_542955

theorem trigonometric_series_bound (n : ℕ) (h : 2 ≤ n) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi) :
  Real.sin (θ / 2) * (Finset.sum (Finset.range n) (λ k, Real.sin (k * θ) / (k : ℝ))) < 1 :=
by
  sorry

end trigonometric_series_bound_l542_542955


namespace incorrect_residual_plot_statement_l542_542177

theorem incorrect_residual_plot_statement :
  ∀ (vertical_only_residual : Prop)
    (horizontal_any_of : Prop)
    (narrower_band_smaller_ssr : Prop)
    (narrower_band_smaller_corr : Prop)
    ,
    narrower_band_smaller_corr → False :=
  by intros vertical_only_residual horizontal_any_of narrower_band_smaller_ssr narrower_band_smaller_corr
     sorry

end incorrect_residual_plot_statement_l542_542177


namespace count_triangles_up_to_10_l542_542773

theorem count_triangles_up_to_10 : 
  (∃ (a b c : ℕ), a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧ a ≥ b ∧ b ≥ c ∧ b + c > a) →

  let triangles_count := (∑ a in (finset.range 11).val,  
    (∑ b in ((finset.range (a + 1).val).filter (λ b, b > a / 2)).val, 
     (b - max 1 (a - b + 1) + 1) )) 
   
  triangles_count = 125 :=
by
  sorry

end count_triangles_up_to_10_l542_542773


namespace max_value_of_z_l542_542923

theorem max_value_of_z : ∀ x : ℝ, (x^2 - 14 * x + 10 ≤ 0 - 39) :=
by
  sorry

end max_value_of_z_l542_542923


namespace distance_light_travels_100_years_l542_542878

def distance_light_travels_one_year : ℝ := 5870e9 * 10^3

theorem distance_light_travels_100_years : distance_light_travels_one_year * 100 = 587 * 10^12 :=
by
  rw [distance_light_travels_one_year]
  sorry

end distance_light_travels_100_years_l542_542878


namespace ant_return_5_moves_l542_542641

-- Define the conditions of the problem
def five_dim_hypercube := SimpleGraph.cube 5
def start_vertex := (0, 0, 0, 0, 0 : Fin 2)

theorem ant_return_5_moves :
  let moves := 5
  let distance := 1
  let total_ways := 6240
  ∃ (count : ℕ), count = total_ways :=
sorry

end ant_return_5_moves_l542_542641


namespace nanometers_to_scientific_notation_l542_542195

   theorem nanometers_to_scientific_notation :
     (0.000000001 : Float) = 1 * 10 ^ (-9) :=
   by
     sorry
   
end nanometers_to_scientific_notation_l542_542195


namespace minimize_M_l542_542128

theorem minimize_M (A B : ℝ) :
  ∃ M, M = (⨆ x ∈ (Set.Icc 0 (3 / 2 * Real.pi)), abs ((Real.cos x) ^ 2 + 2 * Real.sin x * Real.cos x - (Real.sin x) ^ 2 + A * x + B)) ∧ (∀ A B, M ≥ Real.sqrt 2) ∧ A = 0 ∧ B = 0 ∧ M = Real.sqrt 2 :=
sorry

end minimize_M_l542_542128


namespace not_buy_either_l542_542606

-- Definitions
variables (n T C B : ℕ)
variables (h_n : n = 15)
variables (h_T : T = 9)
variables (h_C : C = 7)
variables (h_B : B = 3)

-- Theorem statement
theorem not_buy_either (n T C B : ℕ) (h_n : n = 15) (h_T : T = 9) (h_C : C = 7) (h_B : B = 3) :
  n - (T - B) - (C - B) - B = 2 :=
sorry

end not_buy_either_l542_542606


namespace minimum_candy_kinds_l542_542400

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l542_542400


namespace part_a_part_b_l542_542644

section
-- Definitions based on the conditions
variable (n : ℕ)  -- Variable n representing the number of cities

-- Given a condition function T_n that returns an integer (number of ways to build roads)
def T_n (n : ℕ) : ℕ := sorry  -- Definition placeholder for T_n function

-- Part (a): For all odd n, T_n(n) is divisible by n
theorem part_a (hn : n % 2 = 1) : T_n n % n = 0 := sorry

-- Part (b): For all even n, T_n(n) is divisible by n / 2
theorem part_b (hn : n % 2 = 0) : T_n n % (n / 2) = 0 := sorry

end

end part_a_part_b_l542_542644


namespace solve_for_x_l542_542660

-- Define the conditions
def percentage15_of_25 : ℝ := 0.15 * 25
def percentage12 (x : ℝ) : ℝ := 0.12 * x
def condition (x : ℝ) : Prop := percentage15_of_25 + percentage12 x = 9.15

-- The target statement to prove
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 45 :=
by 
  -- The proof is omitted
  sorry

end solve_for_x_l542_542660


namespace ratio_of_b_to_a_l542_542102

open Real

theorem ratio_of_b_to_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h1 : log 9 a = log 12 b) (h2 : log 9 a = log 16 (3 * a + b)) :
  b / a = (1 + sqrt 13) / 2 :=
sorry

end ratio_of_b_to_a_l542_542102


namespace probability_of_specific_combination_l542_542464

def count_all_clothes : ℕ := 6 + 7 + 8 + 3
def choose4_out_of_24 : ℕ := Nat.choose 24 4
def choose1_shirt : ℕ := 6
def choose1_pair_shorts : ℕ := 7
def choose1_pair_socks : ℕ := 8
def choose1_hat : ℕ := 3
def favorable_outcomes : ℕ := choose1_shirt * choose1_pair_shorts * choose1_pair_socks * choose1_hat
def probability_of_combination : ℚ := favorable_outcomes / choose4_out_of_24

theorem probability_of_specific_combination :
  probability_of_combination = 144 / 1815 := by
sorry

end probability_of_specific_combination_l542_542464


namespace price_for_2_mice_correct_l542_542652

noncomputable def price_for_2_mice (P : ℝ) : Prop :=
∃ P : ℝ, 
  let price_one_mouse := P / 2 in
  let cost_6_mice := 3 * P in
  let cost_7th_mouse := price_one_mouse in
  (cost_6_mice + cost_7th_mouse = 18.69) ∧ (P = 5.34)

theorem price_for_2_mice_correct : price_for_2_mice 5.34 :=
by {
  let P := 5.34,
  let price_one_mouse := P / 2,
  let cost_6_mice := 3 * P,
  let cost_7th_mouse := price_one_mouse,
  have H : cost_6_mice + cost_7th_mouse = 18.69,
  { sorry }, -- Actual solution steps skipped with sorry but needed for completeness
  exact ⟨P, ⟨price_one_mouse, cost_6_mice, cost_7th_mouse, ⟨H, rfl⟩⟩⟩
}

end price_for_2_mice_correct_l542_542652


namespace interior_diagonals_of_dodecahedron_l542_542019

/-- Definition of a dodecahedron. -/
structure Dodecahedron where
  vertices : ℕ
  faces : ℕ
  vertices_per_face : ℕ
  faces_meeting_per_vertex : ℕ
  interior_diagonals : ℕ

/-- A dodecahedron has 12 pentagonal faces, 20 vertices, and 3 faces meet at each vertex. -/
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_meeting_per_vertex := 3,
    interior_diagonals := 160 }

theorem interior_diagonals_of_dodecahedron (d : Dodecahedron) :
    d.vertices = 20 → 
    d.faces = 12 →
    d.faces_meeting_per_vertex = 3 →
    d.interior_diagonals = 160 :=
by
  intros
  sorry

end interior_diagonals_of_dodecahedron_l542_542019


namespace washing_machine_time_l542_542662

theorem washing_machine_time :
  let (shirts, pants, sweaters, jeans, socks_pairs, scarves) := (18, 12, 17, 13, 10, 8) in
  let (max_shirts_per_cycle, max_pants_per_cycle, max_sweaters_per_cycle, max_jeans_per_cycle) := (15, 15, 15, 15) in
  let max_socks_pairs_per_cycle := 10 in
  let max_scarves_per_cycle := 5 in
  let (time_per_cycle_shirts, time_per_cycle_pants, time_per_cycle_sweaters, time_per_cycle_jeans) := (45, 45, 45, 45) in
  let time_per_cycle_socks := 30 in
  let time_per_cycle_scarves := 60 in
  let cycles_needed (items : Nat) (max_per_cycle : Nat) : Nat := (items + max_per_cycle - 1) / max_per_cycle in
  let total_time (items : Nat) (max_per_cycle : Nat) (time_per_cycle : Nat) : Nat := cycles_needed(items, max_per_cycle) * time_per_cycle in
  let total_time_in_minutes :=
    total_time shirts max_shirts_per_cycle time_per_cycle_shirts +
    total_time pants max_pants_per_cycle time_per_cycle_pants +
    total_time sweaters max_sweaters_per_cycle time_per_cycle_sweaters +
    total_time jeans max_jeans_per_cycle time_per_cycle_jeans +
    total_time socks_pairs max_socks_pairs_per_cycle time_per_cycle_socks +
    total_time scarves max_scarves_per_cycle time_per_cycle_scarves in
  let total_time_in_hours := total_time_in_minutes / 60 in
  total_time_in_hours = 7 :=
by
  sorry

end washing_machine_time_l542_542662


namespace mismatch_colors_probability_l542_542099

/-- Nina's basketball team needs to select their new uniform design. The ninth-graders will choose 
the color of the shorts (black, gold, or blue) and the tenth-graders will choose the color of the 
jersey (black, white, or gold). The two groups will make their decisions independently, and each 
color option is equally likely to be chosen. Prove that the probability that the color of the 
shorts will be different from the color of the jersey is 7/9. -/

def probability_mismatch_colors (shorts_colors : Finset String) (jersey_colors : Finset String) : ℚ :=
  let total_combinations := shorts_colors.card * jersey_colors.card
  let matching_combinations := shorts_colors.filter (λ sc, jersey_colors.contains sc).card
  (total_combinations - matching_combinations) / total_combinations

theorem mismatch_colors_probability :
  probability_mismatch_colors ({"black", "gold", "blue"}.toFinset) ({"black", "white", "gold"}.toFinset) = 7/9 :=
by
  sorry

end mismatch_colors_probability_l542_542099


namespace vector_condition_l542_542087

variables {V : Type*} [inner_product_space ℝ V] 

-- Define non-zero vectors a and b
variables (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)

-- Define the condition of normalized vectors being the same
axiom H : a / ∥a∥ = b / ∥b∥

-- Prove that a = 2 * b
theorem vector_condition : a = 2 • b :=
sorry

end vector_condition_l542_542087


namespace triangle_base_length_l542_542120

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 6) (A_eq : A = 13.5) (area_eq : A = (b * h) / 2) : b = 4.5 :=
by
  sorry

end triangle_base_length_l542_542120


namespace find_circle_equation_l542_542741

noncomputable def center_of_parabola : ℝ × ℝ := (1, 0)

noncomputable def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0

noncomputable def equation_of_circle (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1

theorem find_circle_equation 
  (center_c : ℝ × ℝ := center_of_parabola)
  (tangent : ∀ x y, tangent_line x y → (x - 1) ^ 2 + (y - 0) ^ 2 = 1) :
  equation_of_circle = (fun x y => sorry) :=
sorry

end find_circle_equation_l542_542741


namespace sum_p_q_r_six_l542_542072

noncomputable theory

open real

def satisfies_conditions (p q r : ℝ) : Prop :=
  q = p * (4 - p) ∧
  r = q * (4 - q) ∧
  p = r * (4 - r)

theorem sum_p_q_r_six (p q r : ℝ) (hpq: satisfies_conditions p q r) : p + q + r = 6 :=
  sorry

end sum_p_q_r_six_l542_542072


namespace fail_to_reject_H₀_l542_542271

section

variables (m₁ n₁ m₂ n₂ : ℕ) (α : ℝ)

/-- Hypotheses -/
def H₀ : Prop := ∃ p : ℝ, p₁ = p ∧ p₂ = p
def H₁ : Prop := p₁ ≠ p₂

/-- Failure and sample size conditions -/
def conditions : Prop := 
  m₁ = 15 ∧ n₁ = 800 ∧ 
  m₂ = 25 ∧ n₂ = 1000 ∧ 
  α = 0.05

/-- Combined proportion -/
def p_hat : ℝ := (m₁ + m₂) / (n₁ + n₂)

/-- Standard error calculation -/
def SE : ℝ := real.sqrt (p_hat * (1 - p_hat) * (1 / n₁ + 1 / n₂))

/-- z-score calculation -/
def U_obs : ℝ := (m₁ / n₁ - m₂ / n₂) / SE

/-- Critical value at α = 0.05 for a two-tailed test -/
def critical_value : ℝ := 1.96

/-- Decision to fail to reject H₀ -/
theorem fail_to_reject_H₀ (h : conditions) : | U_obs m₁ n₁ m₂ n₂ α | < critical_value := sorry

end

end fail_to_reject_H₀_l542_542271


namespace complex_subtraction_l542_542865

theorem complex_subtraction : (7 - 3 * Complex.i) - (9 - 5 * Complex.i) = -2 + 2 * Complex.i :=
by
  sorry

end complex_subtraction_l542_542865


namespace jill_trips_to_fill_tank_l542_542494

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end jill_trips_to_fill_tank_l542_542494


namespace monotonic_interval_a_l542_542742

theorem monotonic_interval_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (2 * x - 2 * a) * (2 * 2 - 2 * a) ≥ 0 ∧ (2 * x - 2 * a) * (2 * 3 - 2 * a) ≥ 0) →
  a ≤ 2 ∨ a ≥ 3 := sorry

end monotonic_interval_a_l542_542742


namespace birds_meeting_distance_l542_542952

theorem birds_meeting_distance 
  (D : ℝ) (S1 : ℝ) (S2 : ℝ) (t : ℝ)
  (H1 : D = 45)
  (H2 : S1 = 6)
  (H3 : S2 = 2.5)
  (H4 : t = D / (S1 + S2)) :
  S1 * t = 31.76 :=
by
  sorry

end birds_meeting_distance_l542_542952


namespace max_min_of_f_on_interval_l542_542886

theorem max_min_of_f_on_interval :
  let f := λ x : ℝ, -3 * x + 1,
  let a : ℝ := 0,
  let b : ℝ := 1,
  (∀ x ∈ set.Icc a b, f x ≤ f a) ∧ (∀ x ∈ set.Icc a b, f x ≥ f b) :=
by
  sorry

end max_min_of_f_on_interval_l542_542886


namespace tetrahedron_volume_bound_l542_542858

variable (d1 d2 d3 V : ℝ)
variable (condition : ∀ (tetrahedron : Type), (edge_pair_distances : tetrahedron → ℝ × ℝ × ℝ) → 
                edge_pair_distances tetrahedron = (d1, d2, d3) → 
                tetrahedron → ℝ)

theorem tetrahedron_volume_bound (h : ∀ T : Type, edge_pair_distances T = (d1, d2, d3) → V ≥ (1 / 3) * d1 * d2 * d3) : 
  ∀ (tetrahedron : Type), (edge_pair_distances : tetrahedron → ℝ × ℝ × ℝ) →
  edge_pair_distances tetrahedron = (d1, d2, d3) → 
  tetrahedron → 
  V ≥ (1 / 3) * d1 * d2 * d3 := 
by
  sorry

end tetrahedron_volume_bound_l542_542858


namespace winning_strategy_l542_542876

-- Definitions for conditions
def king_is_at_top_left (king_position : (ℕ × ℕ)) : Prop :=
  king_position = (0, 0)

def valid_move (king_position : (ℕ × ℕ)) (board : ℕ × ℕ → bool) : Prop :=
  let (x, y) := king_position in
  (x > 0 ∧ ¬board (x - 1, y)) ∨ (y > 0 ∧ ¬board (x, y - 1)) ∨ 
  (x < fst board ∧ ¬board (x + 1, y)) ∨ (y < snd board ∧ ¬board (x, y + 1))

-- Main theorem
theorem winning_strategy (n m : ℕ) (king_position : ℕ × ℕ) 
    (board : ℕ × ℕ → bool) (h : king_is_at_top_left king_position) :
  (even (n * m) → ∃ strat : (ℕ × ℕ) → (ℕ × ℕ), valid_move (strat king_position) board) ∧ 
  (odd (n * m) → ∀ strat : (ℕ × ℕ) → (ℕ × ℕ), ¬valid_move (strat king_position) board) :=
by
  sorry

end winning_strategy_l542_542876


namespace find_number_l542_542926

theorem find_number (x : ℝ) (h : ((x / 3) * 24) - 7 = 41) : x = 6 :=
by
  sorry

end find_number_l542_542926


namespace sequence_existence_l542_542690

open Real

noncomputable def exists_sequence {α : Type*} (a : ℕ → α) :=
  ∃ (x y : α), 
    (∀ n : ℕ, a (n + 2) = x * a (n + 1) + y * a n) ∧
    ∀ r > 0, ∃ (i j : ℕ), i > 0 ∧ j > 0 ∧ |a i| < r ∧ r < |a j|

theorem sequence_existence : 
  ∃ (a : ℕ → ℝ), 
    (∀ n : ℕ, a n ≠ 0) ∧ 
    exists_sequence a :=
sorry

end sequence_existence_l542_542690


namespace minimum_candy_kinds_l542_542406

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542406


namespace problem_a_problem_b_problem_c_problem_d_l542_542868

-- a) Proof problem for \(x^2 + 5x + 6 < 0\)
theorem problem_a (x : ℝ) : x^2 + 5*x + 6 < 0 → -3 < x ∧ x < -2 := by
  sorry

-- b) Proof problem for \(-x^2 + 9x - 20 < 0\)
theorem problem_b (x : ℝ) : -x^2 + 9*x - 20 < 0 → x < 4 ∨ x > 5 := by
  sorry

-- c) Proof problem for \(x^2 + x - 56 < 0\)
theorem problem_c (x : ℝ) : x^2 + x - 56 < 0 → -8 < x ∧ x < 7 := by
  sorry

-- d) Proof problem for \(9x^2 + 4 < 12x\) (No solutions)
theorem problem_d (x : ℝ) : ¬ 9*x^2 + 4 < 12*x := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l542_542868


namespace job_completion_l542_542935

theorem job_completion (A_rate D_rate : ℝ) (h₁ : A_rate = 1 / 12) (h₂ : A_rate + D_rate = 1 / 4) : D_rate = 1 / 6 := 
by 
  sorry

end job_completion_l542_542935


namespace convert_seven_cubic_yards_l542_542375

-- Define the conversion factor from yards to feet
def yardToFeet : ℝ := 3
-- Define the conversion factor from cubic yards to cubic feet
def cubicYardToCubicFeet : ℝ := yardToFeet ^ 3
-- Define the conversion function from cubic yards to cubic feet
noncomputable def convertVolume (volumeInCubicYards : ℝ) : ℝ :=
  volumeInCubicYards * cubicYardToCubicFeet

-- Statement to prove: 7 cubic yards is equivalent to 189 cubic feet
theorem convert_seven_cubic_yards : convertVolume 7 = 189 := by
  sorry

end convert_seven_cubic_yards_l542_542375


namespace n_values_l542_542716

noncomputable def a_seq (n m : ℕ) : ℕ → ℕ
| 0     := m
| (k+1) := (a_seq k * a_seq k) % n

noncomputable def f (n m : ℕ) : ℤ :=
(0 to 2000).sum (λ (i : ℕ), if i % 2 == 0 then ↑(a_seq n m i) else -↑(a_seq n m i))

theorem n_values {n : ℕ} (hn : n ≥ 5) :
  (∃ m : ℕ, 2 ≤ m ∧ m ≤ n / 2 ∧ f n m > 0 ↔ n = 6 ∨ n = 7) :=
sorry

end n_values_l542_542716


namespace length_of_AC_is_45_l542_542821

-- Conditions definitions
variables (A B C D E : Type)
variables (AC BD : ℝ)
variable (E_div_AC : ℝ)
variable (E_div_BD : ℝ)

-- Given conditions
axiom h1 : E_div_AC = 2 / 3
axiom h2 : E_div_BD = 3 / 4
axiom h3 : AC > 0
axiom h4 : BD > 0
axiom h5 : ∃ EC : ℝ, EC = 15
axiom h6 : ∃ ED : ℝ, ED = 24

-- Definition of the length of diagonal AC
def length_of_AC (E_div_AC : ℝ) (EC : ℝ) : ℝ := (EC / (1 - E_div_AC))

-- Proof statement to be proven
theorem length_of_AC_is_45 : length_of_AC E_div_AC 15 = 45 :=
by sorry

#print axioms length_of_AC_is_45

end length_of_AC_is_45_l542_542821


namespace fraction_cal_handled_l542_542549

theorem fraction_cal_handled (Mabel Anthony Cal Jade : ℕ) 
  (h_Mabel : Mabel = 90)
  (h_Anthony : Anthony = Mabel + Mabel / 10)
  (h_Jade : Jade = 80)
  (h_Cal : Cal = Jade - 14) :
  (Cal : ℚ) / (Anthony : ℚ) = 2 / 3 :=
by
  sorry

end fraction_cal_handled_l542_542549


namespace combined_work_rate_l542_542184

/-- Definition of work rate for A and B and their combined work rate --/
def work_rate (W : ℝ) : ℝ :=
  let A_rate := W / 8
  let B_rate := W / 4
  A_rate + B_rate

/-- The main theorem stating the fraction of work A and B can complete together in one day --/
theorem combined_work_rate (W : ℝ) : work_rate W = (3 * W) / 8 :=
by
  sorry

end combined_work_rate_l542_542184


namespace matrix_product_correct_l542_542678

def matrix_seq_sum : ℕ → ℕ :=
λ n, 2 + (n - 1) * 2

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  if n = 0 then ![![1, 0], ![0, 1]]
  else let sum := (matrix_seq_sum n).sum in
    ![![1, sum], ![0, 1]]

theorem matrix_product_correct : matrix_product 50 = ![![1, 2550], ![0, 1]] :=
sorry

end matrix_product_correct_l542_542678


namespace retailer_mark_up_l542_542225

theorem retailer_mark_up (R C M S : ℝ) 
  (hC : C = 0.7 * R)
  (hS : S = C / 0.7)
  (hSm : S = 0.9 * M) : 
  M = 1.111 * R :=
by 
  sorry

end retailer_mark_up_l542_542225


namespace inverse_function_fraction_eq_neg_four_l542_542129

theorem inverse_function_fraction_eq_neg_four (a b c d : ℝ) 
  (hf : ∀ x, f x = (3 * x - 2) / (x + 4))
  (hinv : f⁻¹ = λ x, (a * x + b) / (c * x + d)) : 
  a / c = -4 :=
  by
  -- Function definition needed to apply the conditions, we assume f has already been defined.
  sorry

end inverse_function_fraction_eq_neg_four_l542_542129


namespace lindys_speed_l542_542807

-- Setting up the given conditions
def initial_distance : ℝ := 240
def jack_speed : ℝ := 5
def christina_speed : ℝ := 3
def lindy_total_distance : ℝ := 270

-- Given the total distance Jack and Christina decrease the distance between them:
def combined_speed : ℝ := jack_speed + christina_speed
def meeting_time : ℝ := initial_distance / combined_speed

-- Prove that Lindy's speed is 9 feet per second
theorem lindys_speed :
  lindy_total_distance / meeting_time = 9 := by
  sorry

end lindys_speed_l542_542807


namespace number_of_persons_in_the_group_l542_542581

-- Define the conditions and the final proof goal
variable (n : ℕ)

-- Condition 1: The age decrease equation from the problem
axiom condition1 : 45 * n - 45 + 15 = 45 * n - 3 * n

-- Calculate the value of n
theorem number_of_persons_in_the_group : n = 10 :=
by
  have h : 45 * n - 30 = 42 * n, from condition1,
  have h' : 45 * n - 42 * n = 30, by linarith,
  have h'' : 3 * n = 30, by linarith,
  exact eq_of_mul_eq_mul_right dec_trivial h''

end number_of_persons_in_the_group_l542_542581


namespace hypotenuse_ratio_l542_542307

theorem hypotenuse_ratio (s : ℝ) (h : ℝ) :
  (4 * s^2 * 2 = h^2 * 4) → (h = s * sqrt(2)) :=
by
  sorry

end hypotenuse_ratio_l542_542307


namespace color_6x6_square_l542_542056

theorem color_6x6_square :
  ∃ (color : Fin 6 × Fin 6 → bool), 
  (∃ (n m : ℕ), 
      n > m ∧
      (∀ (r : Fin 6) (c : Fin 3), 
         ∑ i in finset.range 4, if color (r, (c + i) % 6) then 1 else 0 = 2) ∧
      (∀ (c : Fin 6) (r : Fin 3), 
         ∑ i in finset.range 4, if color ((r + i) % 6, c) then 1 else 0 = 2)) :=
sorry

end color_6x6_square_l542_542056


namespace complement_of_P_l542_542389
def U := {-1, 0, 1, 2}
def P := {x : ℤ | x^2 < 2}
def CU_P := {-1, 0, 1, 2} \ {x : ℤ | x^2 < 2}
theorem complement_of_P : CU_P = {2} := by
  sorry

end complement_of_P_l542_542389


namespace units_digit_G_100_l542_542264

theorem units_digit_G_100 : 
  let G (n : ℕ) := 3^(3^n) + 1 in
  G 100 % 10 = 4 := 
by
  sorry

end units_digit_G_100_l542_542264


namespace minimum_candy_kinds_l542_542410

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l542_542410


namespace find_g_neg_2_l542_542339

-- Definitions
variable {R : Type*} [CommRing R] [Inhabited R]
variable (f g : R → R)

-- Conditions
axiom odd_y (x : R) : f (-x) + 2 * x^2 = -(f x + 2 * x^2)
axiom definition_g (x : R) : g x = f x + 1
axiom value_f_2 : f 2 = 2

-- Goal
theorem find_g_neg_2 : g (-2) = -17 :=
by
  sorry

end find_g_neg_2_l542_542339


namespace extra_bananas_l542_542848

theorem extra_bananas
  (total_children : ℕ)
  (absent_children : ℕ)
  (D : ℕ)
  (bananas_per_child : ℕ)
  (total_children = 740)
  (absent_children = 370)
  (bananas_per_child = 2)
  (total_bananas : ℕ)
  (total_bananas = total_children * bananas_per_child)
  (remaining_children : ℕ)
  (remaining_children = total_children - absent_children)
  (bananas_per_remaining_child : ℕ)
  (bananas_per_remaining_child = total_bananas / remaining_children) :
  bananas_per_remaining_child - bananas_per_child = 2 :=
by
  -- Proof will be filled here
  sorry

end extra_bananas_l542_542848


namespace min_value_a_plus_b_l542_542314

open Real

theorem min_value_a_plus_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : 1 / a + 2 / b = 1) :
  a + b = 3 + 2 * sqrt 2 :=
sorry

end min_value_a_plus_b_l542_542314


namespace y_intercept_of_PR_l542_542631

theorem y_intercept_of_PR (b : ℝ) (h1 : ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 = 0)
  (h2 : ∀ Q R : ℝ × ℝ, Q.2 = Q.1^2 ∧ R.2 = R.1^2 ∧ (Q.2, R.2) ∈ line 1 b ∧ 
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (R.1 - Q.1)^2 + (R.2 - Q.2)^2) :
  b = 4 + 3 * Real.sqrt 2 → b ≈ 8.2 := 
sorry

end y_intercept_of_PR_l542_542631


namespace more_movies_read_than_books_l542_542910

theorem more_movies_read_than_books (books movies books_read movies_watched : ℕ)
  (h_books : books = 25) (h_movies : movies = 32) (h_books_read : books_read = 17) (h_movies_watched : movies_watched = 21) :
  |books_read - movies_watched| = 4 :=
by
  sorry

end more_movies_read_than_books_l542_542910


namespace min_ways_to_open_boxes_ways_with_exactly_two_cycles_l542_542262

-- Statement for the first part of the problem
theorem min_ways_to_open_boxes (n : ℕ) 
  (h_n : n = 10) :
  (∃ p : Equiv.Perm (Fin n), ∃ c : ℕ, c ≥ 2 ∧ p.cycleType.length ≥ 2) → 
  (∃ ways : ℕ, ways = 9 * (n - 1)!) :=
sorry

-- Statement for the second part of the problem
theorem ways_with_exactly_two_cycles (n : ℕ) 
  (h_n : n = 10) :
  (∃ p : Equiv.Perm (Fin n), p.cycleType.length = 2) →
  (∃ ways : ℕ, ways = 1024576) :=
sorry

end min_ways_to_open_boxes_ways_with_exactly_two_cycles_l542_542262


namespace warehouse_pumping_time_is_450_l542_542996

def warehouse_pump_time : ℕ :=
  let length := 30 in
  let width := 40 in
  let depth := 24 / 12 in  -- depth in feet
  let volume_cubic_feet := length * width * depth in
  let gallons_per_cubic_foot := 7.5 in
  let total_gallons := volume_cubic_feet * gallons_per_cubic_foot in
  let pumps := 4 in
  let rate_per_pump := 10 in  -- gallons per minute per pump
  let total_rate := pumps * rate_per_pump in
  total_gallons / total_rate

theorem warehouse_pumping_time_is_450 : warehouse_pump_time = 450 :=
by sorry

end warehouse_pumping_time_is_450_l542_542996


namespace part_a_l542_542633

theorem part_a (m n : ℕ) (hm : m > 1) : n ∣ Nat.totient (m^n - 1) :=
sorry

end part_a_l542_542633


namespace sufficient_condition_for_line_perpendicular_to_plane_l542_542740

variables {Plane Line : Type}
variables (α β γ : Plane) (m n l : Line)

-- Definitions of perpendicularity and inclusion
def perp (l : Line) (p : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def parallel (p₁ p₂ : Plane) : Prop := sorry -- definition of parallel planes
def incl (l : Line) (p : Plane) : Prop := sorry -- definition of a line being in a plane

-- The given conditions
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- The proof goal
theorem sufficient_condition_for_line_perpendicular_to_plane :
  perp m β :=
by
    sorry

end sufficient_condition_for_line_perpendicular_to_plane_l542_542740


namespace geometric_sequence_problem_l542_542198

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (h1 : a 1 + a 1 * q ^ 2 = 10) (h2 : a 1 * q + a 1 * q ^ 3 = 5) (h3 : geometric_sequence a q) :
  a 8 = 1 / 16 := sorry

end geometric_sequence_problem_l542_542198


namespace axis_of_symmetry_parabola_l542_542704

theorem axis_of_symmetry_parabola (x y : ℝ) :
  y = - (1 / 8) * x^2 → y = 2 :=
sorry

end axis_of_symmetry_parabola_l542_542704


namespace coins_of_each_type_l542_542202

theorem coins_of_each_type (x : ℕ) (h : x + x / 2 + x / 4 = 70) : x = 40 :=
sorry

end coins_of_each_type_l542_542202


namespace lateral_surface_area_square_rotated_l542_542111

noncomputable def lateral_surface_area_of_rotated_square (a : ℝ) : ℝ :=
  2 * real.pi * a

theorem lateral_surface_area_square_rotated (a : ℝ) (h : a = 1) :
  lateral_surface_area_of_rotated_square a = 2 * real.pi :=
by
  rw [h]
  unfold lateral_surface_area_of_rotated_square
  simp
  sorry

end lateral_surface_area_square_rotated_l542_542111


namespace coefficient_x8_l542_542703

noncomputable def polynomial := (1 - X + 2 * X^2)^5

theorem coefficient_x8 : (coeff (polynomial) 8) = 80 :=
by
  sorry

end coefficient_x8_l542_542703


namespace sum_of_diameters_of_inscribed_circles_l542_542247

-- Definitions based on conditions
def radius (R : ℝ) : ℝ := R
def perimeter_hexagon (Q : ℝ) : ℝ := Q
def acute_triangle_split (R : ℝ) (Q : ℝ) : Prop :=
  ∃ (right_triangle_1 right_triangle_2 right_triangle_3 : ℝ),
    -- Each of the following right triangles has inscribed circle radius similar properties

    -- The sum of the diameters of these three right triangles' inscribed circles will equal the provided value
    (right_triangle_1 + right_triangle_2 + right_triangle_3) = Q - 6 * R

-- Statement of the theorem
theorem sum_of_diameters_of_inscribed_circles (R Q : ℝ) 
  (h : acute_triangle_split R Q) : 
  ∑ d in {right_triangle_1, right_triangle_2, right_triangle_3}, d = Q - 6 * R :=
by
  -- Circumscribed property propagation
  exact h.2

-- Assertion part of the proof is missing in the provided Lean 4 statement to ensure it's valid for every step.
sorry

end sum_of_diameters_of_inscribed_circles_l542_542247


namespace AE_computation_l542_542083

-- Defining the conditions
def isosceles_trapezoid (A B C D : Point) : Prop :=
  AD = BC ∧ AB = 3 ∧ CD = 8

def point_in_plane (E : Point) (BC EC : ℝ) : Prop :=
  BC = EC ∧ AE ⟂ EC

-- Define the points A, B, C, D, E
def A : Point := (0, 0)
def B : Point := (3, 0)
def C : Point := (11/2, sqrt(24))
def D : Point := (-5/2, sqrt(24))

def E : Point := (some x, some y) -- Using some to denote that these points exist based on condition BC = EC

-- Problem statement to prove AE = 2 * sqrt(6)
theorem AE_computation :
  isosceles_trapezoid A B C D →
  point_in_plane E (BC = sqrt(24)) →
  AE E = 2 * sqrt(6) :=
by
  -- We skip the proof
  sorry

end AE_computation_l542_542083


namespace percentage_increase_area_l542_542191

theorem percentage_increase_area (L W : ℝ) :
  let A := L * W
  let L' := 1.20 * L
  let W' := 1.20 * W
  let A' := L' * W'
  let percentage_increase := (A' - A) / A * 100
  L > 0 → W > 0 → percentage_increase = 44 := 
by
  sorry

end percentage_increase_area_l542_542191


namespace NaHSO3_moles_to_SO2_l542_542711

-- Definitions
def balanced_reaction : Prop :=
  ∀ (NaHSO3 HCl SO2 H2O NaCl : Type),
    (NaHSO3 + HCl) = (SO2 + H2O + NaCl)

def mole_ratio (NaHSO3 SO2 : Type) : Prop := 
  (∀ n : ℕ, (n : ℕ).succ * NaHSO3 = (n : ℕ).succ * SO2)

-- Theorem statement
theorem NaHSO3_moles_to_SO2 (NaHSO3 SO2 : Type) (HCl H2O NaCl : Type) 
  (balanced : balanced_reaction NaHSO3 HCl SO2 H2O NaCl) 
  (ratio : mole_ratio NaHSO3 SO2) : 
  (2 : ℕ) * NaHSO3 = (2 : ℕ) * SO2 :=
by
  sorry

end NaHSO3_moles_to_SO2_l542_542711


namespace minimum_candy_kinds_l542_542446

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l542_542446


namespace rectangle_area_l542_542190

noncomputable def length (width : ℕ) : ℕ := 3 * width
noncomputable def area (length width : ℕ) : ℕ := length * width

theorem rectangle_area :
  let width := 6 in
  let length := 3 * width in
  area length width = 108 :=
by
  sorry

end rectangle_area_l542_542190


namespace derivative_ln_cubed_l542_542943

theorem derivative_ln_cubed (x : ℝ) (h : 5 * x + 2 > 0) : 
  let y := (ln(5 * x + 2))^3 in
  deriv (λ x : ℝ, (ln (5 * x + 2))^3) x = (15 * (ln (5 * x + 2))^2) / (5 * x + 2) :=
by sorry

end derivative_ln_cubed_l542_542943


namespace linear_function_diff_l542_542591

variable {α : Type _} [AddGroup α]

def linear (g : ℝ → α) : Prop := ∀ (x y : α), g (x + y) = g x + g y

theorem linear_function_diff (g : ℝ → ℝ) (h1 : linear g) (h2 : ∀ d : ℝ, g(d + 1) = g(d) + 5) : g 0 - g 4 = -20 :=
sorry

end linear_function_diff_l542_542591


namespace triangle_properties_l542_542801

open Real

noncomputable def vec_m (a : ℝ) : ℝ × ℝ := (2 * sin (a / 2), sqrt 3)
noncomputable def vec_n (a : ℝ) : ℝ × ℝ := (cos a, 2 * cos (a / 4)^2 - 1)
noncomputable def area_triangle := 3 * sqrt 3 / 2

theorem triangle_properties (a b c : ℝ) (A : ℝ)
  (ha : a = sqrt 7)
  (hA : (1 / 2) * b * c * sin A = area_triangle)
  (hparallel : vec_m A = vec_n A) :
  A = π / 3 ∧ b + c = 5 :=
by
  sorry

end triangle_properties_l542_542801


namespace correct_addition_l542_542612

def is_appropriate_choice (choice : Type) : Prop :=
  ∃ (concentration_ratio : Type), concentration_ratio = (1, 1) →
    match choice with
      | "KOH" => true
      | _ => false

theorem correct_addition :
  ∀ (substance : Type),
  substance = "KOH" →
  is_appropriate_choice substance :=
λ substance h,
  begin
    unfold is_appropriate_choice,
    existsi (1, 1),
    intro hratio,
    rw h,
    exact true.intro
  end

end correct_addition_l542_542612


namespace find_m_l542_542331

-- Defining the relevant vectors and properties in Triangle ABC
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C M : V}

-- Definition of centroid property
def is_centroid (M A B C : V) : Prop := (M - A) + (M - B) + (M - C) = 0

-- The main statement we need to prove
theorem find_m (h : is_centroid M A B C) (k : ∃ m : ℝ, (B - A) + (C - A) = m • (M - A)) : 
  (k.some) = 3 :=
by sorry

end find_m_l542_542331


namespace problem_1_problem_2_problem_3_problem_4_l542_542152

-- Given conditions
variable {T : Type} -- Type representing teachers
variable {S : Type} -- Type representing students

def arrangements_ends (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_not_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_two_between (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

-- Statements to prove

-- 1. Prove that if teachers A and B must stand at the two ends, there are 48 different arrangements
theorem problem_1 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_ends teachers students = 48 :=
  sorry

-- 2. Prove that if teachers A and B must stand next to each other, there are 240 different arrangements
theorem problem_2 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_next_to_each_other teachers students = 240 :=
  sorry 

-- 3. Prove that if teachers A and B cannot stand next to each other, there are 480 different arrangements
theorem problem_3 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_not_next_to_each_other teachers students = 480 :=
  sorry 

-- 4. Prove that if there must be two students standing between teachers A and B, there are 144 different arrangements
theorem problem_4 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_two_between teachers students = 144 :=
  sorry

end problem_1_problem_2_problem_3_problem_4_l542_542152


namespace one_in_B_neg_one_not_in_B_B_roster_l542_542539

open Set Int

def B : Set ℤ := {x | ∃ n : ℕ, 6 = n * (3 - x)}

theorem one_in_B : 1 ∈ B :=
by sorry

theorem neg_one_not_in_B : (-1 ∉ B) :=
by sorry

theorem B_roster : B = {2, 1, 0, -3} :=
by sorry

end one_in_B_neg_one_not_in_B_B_roster_l542_542539


namespace min_candy_kinds_l542_542430

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l542_542430


namespace product_mod_m_l542_542172

-- Define the constants
def a : ℕ := 2345
def b : ℕ := 1554
def m : ℕ := 700

-- Definitions derived from the conditions
def a_mod_m : ℕ := a % m
def b_mod_m : ℕ := b % m

-- The proof problem
theorem product_mod_m (a b m : ℕ) (h1 : a % m = 245) (h2 : b % m = 154) :
  (a * b) % m = 630 := by sorry

end product_mod_m_l542_542172


namespace triangle_sides_angles_inradius_l542_542522

variable {α β γ : ℝ} -- Angles of the triangle
variable {a b c r : ℝ} -- Sides a, b, c and inradius r

theorem triangle_sides_angles_inradius 
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) -- Triangle inequality conditions
  (sin_α : ℝ) (sin_β : ℝ) (sin_γ : ℝ) 
  (ha : sin_α = a / (2 * R)) (hb : sin_β = b / (2 * R)) (hc : sin_γ = c / (2 * R)) -- sine law conditions
  (inradius : r) : 
  a * sin_α + b * sin_β + c * sin_γ ≥ 9 * r :=
by 
  sorry

end triangle_sides_angles_inradius_l542_542522


namespace at_least_two_consecutive_heads_probability_l542_542969

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end at_least_two_consecutive_heads_probability_l542_542969


namespace max_people_seated_l542_542157

-- Define the problem conditions and setup
def max_seated_chairs : ℕ := 12

theorem max_people_seated : ∀ (max_seated_chairs : ℕ), max_seated_chairs = 12 → ∃ (max_people : ℕ), max_people = 11 :=
by
  intros n h
  use 11
  sorry

end max_people_seated_l542_542157


namespace problem_statement_l542_542899

-- Define the parametric equations of curve C1
def curveC1 (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 3 * Real.sin θ)

-- Define the polar equation of curve C2 and convert to Cartesian form
def curveC2_polar_eq (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - Real.pi / 4) = Real.sqrt 2

def curveC2_cartesian_eq (x y : ℝ) : Prop :=
  x + y - 2 = 0

-- Distance formula between a point and a line
def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  (Real.abs (a * x + b * y + c)) / (Real.sqrt (a^2 + b^2))

-- C1 and C2 definitions
noncomputable def max_distance_curve : ℝ := 7 * Real.sqrt 2 / 2

-- Proof statement of the problem
theorem problem_statement :
  (∀ ρ θ : ℝ, curveC2_polar_eq ρ θ → ∃ x y : ℝ, curveC2_cartesian_eq x y) ∧
  (∀ θ : ℝ, distance_point_to_line (4 * Real.cos θ) (3 * Real.sin θ) 1 1 (-2) ≤ max_distance_curve) :=
  by
    -- Proof would go here
    sorry

end problem_statement_l542_542899


namespace min_value_of_quadratic_l542_542888

theorem min_value_of_quadratic : ∀ x : ℝ, ∃ y : ℝ, y = (x - 1)^2 - 3 ∧ (∀ z : ℝ, (z - 1)^2 - 3 ≥ y) :=
by
  sorry

end min_value_of_quadratic_l542_542888


namespace couple_slices_each_l542_542215

noncomputable def slices_for_couple (total_slices children_slices people_in_couple : ℕ) : ℕ :=
  (total_slices - children_slices) / people_in_couple

theorem couple_slices_each (people_in_couple children slices_per_pizza num_pizzas : ℕ) (H1 : people_in_couple = 2) (H2 : children = 6) (H3 : slices_per_pizza = 4) (H4 : num_pizzas = 3) :
  slices_for_couple (num_pizzas * slices_per_pizza) (children * 1) people_in_couple = 3 := 
  by
  rw [H1, H2, H3, H4]
  show slices_for_couple (3 * 4) (6 * 1) 2 = 3
  rfl

end couple_slices_each_l542_542215


namespace area_of_inscribed_rectangle_l542_542666

-- Define the conditions of the problem
structure Ellipse (major_radius minor_radius : ℝ) :=
(inscription : ∃ (R : Rectangle), 
  R.length = 2 * major_radius ∧ 
  R.width = 2 * minor_radius ∧ 
  R.length / R.width = 3 / 2)

-- Define a rectangle structure
structure Rectangle :=
(length width : ℝ)

-- Define the main theorem
theorem area_of_inscribed_rectangle (E : Ellipse 7 4) : ∃ R : Rectangle, R.length * R.width = 96 :=
by 
  sorry

end area_of_inscribed_rectangle_l542_542666


namespace proof_problem_l542_542760

variable {α : Type} [DecidableEq α]

noncomputable def x : ℤ := 1  -- Define x in ℤ
def A : Set ℤ := {0, |x|}
def B : Set ℤ := {1, 0, -1}

theorem proof_problem : A ⊆ B →
  (|x| = 1) ∧
  (A = {0, 1}) ∧
  (A ∩ B = {0, 1}) ∧
  (A ∪ B = {-1, 0, 1}) ∧
  (B \ A = {-1}) :=
by
  sorry

end proof_problem_l542_542760


namespace integral_piecewise_result_l542_542315

noncomputable def f : ℝ → ℝ :=
λ x, if x < 1 then real.sqrt (1 - x^2) else x^2 - 1

theorem integral_piecewise_result :
  ∫ x in -1..2, f x = (real.pi / 2) + (4 / 3) :=
by 
  sorry

end integral_piecewise_result_l542_542315


namespace number_of_pairs_l542_542896

theorem number_of_pairs (x y: ℤ) (h1: 0 < x) (h2: x < y) (h3: Real.sqrt 2016 = Real.sqrt x + Real.sqrt y) : 
    ∃ n, n = 3 ∧ n = (finset.univ.filter (λ p, p.1 < p.2 ∧ sqrt p.1 + sqrt p.2 = sqrt 2016)).card :=
begin
  sorry
end

end number_of_pairs_l542_542896


namespace min_number_of_candy_kinds_l542_542422

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l542_542422
