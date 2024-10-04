import Mathlib

namespace boys_to_girls_ratio_l228_228810

theorem boys_to_girls_ratio (S G B : ℕ) (h : (1/2 : ℚ) * G = (1/3 : ℚ) * S) :
  B / G = 1 / 2 :=
by sorry

end boys_to_girls_ratio_l228_228810


namespace tangent_ellipse_hyperbola_l228_228591

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end tangent_ellipse_hyperbola_l228_228591


namespace friend_owns_10_bicycles_l228_228386

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end friend_owns_10_bicycles_l228_228386


namespace sufficient_not_necessary_l228_228827

variable (x : ℝ)

theorem sufficient_not_necessary (h : |x| > 0) : (x > 0 ↔ true) :=
by 
  sorry

end sufficient_not_necessary_l228_228827


namespace smallest_d_l228_228015

theorem smallest_d (d t s : ℕ) (h1 : 3 * t - 4 * s = 2023)
                   (h2 : t = s + d) 
                   (h3 : 4 * s > 0)
                   (h4 : d % 3 = 0) :
                   d = 675 := sorry

end smallest_d_l228_228015


namespace find_number_l228_228734

theorem find_number (x : ℝ) : 
  10 * ((2 * (x * x + 2) + 3) / 5) = 50 → x = 3 := 
by
  sorry

end find_number_l228_228734


namespace product_of_g_xi_l228_228828

noncomputable def x1 : ℂ := sorry
noncomputable def x2 : ℂ := sorry
noncomputable def x3 : ℂ := sorry
noncomputable def x4 : ℂ := sorry
noncomputable def x5 : ℂ := sorry

def f (x : ℂ) : ℂ := x^5 + x^2 + 1
def g (x : ℂ) : ℂ := x^3 - 2

axiom roots_of_f (x : ℂ) : f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5

theorem product_of_g_xi : (g x1) * (g x2) * (g x3) * (g x4) * (g x5) = -243 := sorry

end product_of_g_xi_l228_228828


namespace time_spent_on_Type_A_problems_l228_228182

theorem time_spent_on_Type_A_problems (total_questions : ℕ)
  (exam_duration_minutes : ℕ) (type_A_problems : ℕ)
  (time_ratio_A_B : ℕ) : ℕ :=
  let total_questions = 200 
  let exam_duration_minutes = 180 
  let type_A_problems = 25 
  let time_ratio_A_B = 2 
  40 := 
  sorry

end time_spent_on_Type_A_problems_l228_228182


namespace find_larger_number_l228_228153

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 :=
sorry

end find_larger_number_l228_228153


namespace percentage_of_40_eq_140_l228_228459

theorem percentage_of_40_eq_140 (p : ℝ) (h : (p / 100) * 40 = 140) : p = 350 :=
sorry

end percentage_of_40_eq_140_l228_228459


namespace hyperbola_line_intersection_range_l228_228243

def hyperbola : ℝ → ℝ → Prop := λ x y, x^2 - y^2 = 4
def line (k : ℝ) : ℝ → ℝ → Prop := λ x y, y = k * (x - 1)

theorem hyperbola_line_intersection_range (k : ℝ) :
  let discriminant : ℝ := 4 * (4 - 3 * k^2) in
  (¬ (1 - k^2 = 0) ∧ discriminant > 0 → k ∈ Ioo (-2 * real.sqrt 3 / 3) (-1) ∪ Ioo (-1) 1 ∪ Ioo 1 (2 * real.sqrt 3 / 3))
  ∧ ((1 - k^2 = 0 ∨ (¬ (1 - k^2 = 0) ∧ discriminant = 0)) → k = 1 ∨ k = -1 ∨ k = 2 * real.sqrt 3 / 3 ∨ k = -2 * real.sqrt 3 / 3)
  ∧ (¬ (¬ (1 - k^2 = 0) ∧ discriminant > 0 ∨ (1 - k^2 = 0 ∨ (¬ (1 - k^2 = 0) ∧ discriminant = 0))) → k ∈ Iio (-2 * real.sqrt 3 / 3) ∪ Ioi (2 * real.sqrt 3 / 3)) := sorry

end hyperbola_line_intersection_range_l228_228243


namespace line_MN_fixed_point_l228_228525

section EllipseProof

variable {a b : ℝ} (h1 : a = 2 * Real.sqrt 2) (h2 : b = 2)

noncomputable def ellipse_eq : Prop := ∀ (x y : ℝ), (x^2 / 8 + y^2 / 4 = 1) ↔ (Real.sqrt (x^2 + y^2) = 1)

def fixed_point (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem line_MN_fixed_point (x1 y1 x2 y2 : ℝ) (h3 : x1^2 / 8 + y1^2 / 4 = 1) (h4 : x2^2 / 8 + y2^2 / 4 = 1) (h5 : y1 = k * x1 + 1) (h6 : y2 = k * x2 + 1) (h7 : k ≠ 0) : fixed_point 0 1 := sorry

end EllipseProof

end line_MN_fixed_point_l228_228525


namespace catering_budget_l228_228120

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end catering_budget_l228_228120


namespace product_of_solutions_t_squared_eq_49_l228_228899

theorem product_of_solutions_t_squared_eq_49 (t : ℝ) (h1 : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_solutions_t_squared_eq_49_l228_228899


namespace surface_area_of_solid_l228_228216

-- Definitions about the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_rectangular_solid (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c ∧ (a * b * c = 399)

-- Main statement of the problem
theorem surface_area_of_solid (a b c : ℕ) (h : is_rectangular_solid a b c) : 
  2 * (a * b + b * c + c * a) = 422 := sorry

end surface_area_of_solid_l228_228216


namespace smallest_common_multiple_l228_228515

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l228_228515


namespace a_plus_b_is_24_l228_228280

theorem a_plus_b_is_24 (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (a + 3 * b) = 550) : a + b = 24 :=
sorry

end a_plus_b_is_24_l228_228280


namespace hash_nesting_example_l228_228358

def hash (N : ℝ) : ℝ :=
  0.5 * N + 2

theorem hash_nesting_example : hash (hash (hash (hash 20))) = 5 :=
by
  sorry

end hash_nesting_example_l228_228358


namespace convert_to_base8_l228_228651

theorem convert_to_base8 (n : ℕ) (h : n = 1024) : 
  (∃ (d3 d2 d1 d0 : ℕ), n = d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0 ∧ d3 = 2 ∧ d2 = 0 ∧ d1 = 0 ∧ d0 = 0) :=
by
  sorry

end convert_to_base8_l228_228651


namespace wire_cut_min_area_l228_228084

theorem wire_cut_min_area :
  ∃ x : ℝ, 0 < x ∧ x < 100 ∧ S = π * (x / (2 * π))^2 + ((100 - x) / 4)^2 ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 100 → (π * (y / (2 * π))^2 + ((100 - y) / 4)^2 ≥ S)) ∧
  x = 100 * π / (16 + π) :=
sorry

end wire_cut_min_area_l228_228084


namespace pallets_total_l228_228203

theorem pallets_total (P : ℕ) (h1 : P / 2 + P / 4 + P / 5 + 1 = P) : P = 20 :=
by
  sorry

end pallets_total_l228_228203


namespace original_price_of_sarees_l228_228859

theorem original_price_of_sarees (P : ℝ) (h : 0.92 * 0.90 * P = 331.2) : P = 400 :=
by
  sorry

end original_price_of_sarees_l228_228859


namespace parabola_equation_l228_228932

theorem parabola_equation (a : ℝ) : 
(∀ x y : ℝ, y = x → y = a * x^2)
∧ (∃ P : ℝ × ℝ, P = (2, 2) ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) 
  → A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ y₁ = x₁ ∧ y₂ = x₂ ∧ x₂ = x₁ → 
  ∃ f : ℝ × ℝ, f.fst ≠ 0 ∧ f.snd = 0) →
  a = (1 : ℝ) / 7 := 
sorry

end parabola_equation_l228_228932


namespace classes_Mr_Gates_has_l228_228940

theorem classes_Mr_Gates_has (buns_per_package packages_bought students_per_class buns_per_student : ℕ) :
  buns_per_package = 8 → 
  packages_bought = 30 → 
  students_per_class = 30 → 
  buns_per_student = 2 → 
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 := 
by
  sorry

end classes_Mr_Gates_has_l228_228940


namespace find_c_l228_228241

-- Define that X follows normal distribution N(3, 1)
variable {X : ℝ → ℝ}
def normal_dist (μ σ : ℝ) (x : ℝ) : ℝ := 
  1 / (σ * real.sqrt (2 * real.pi)) * real.exp (-(x - μ)^2 / (2 * σ^2))

-- Given: X follows N(3,1)
axiom normal_X : ∀ x, X x = normal_dist 3 1 x

-- Prove: If P(X < 2 * c + 1) = P(X > c + 5), then c = 0.
theorem find_c (c : ℝ) : (∫ x in set.Iic (2 * c + 1), X x) = (∫ x in set.Ioi (c + 5), X x) → c = 0 := 
by
  sorry

end find_c_l228_228241


namespace calculate_f_at_2_l228_228532

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem calculate_f_at_2
  (a b : ℝ)
  (h_extremum : 3 + 2 * a + b = 0)
  (h_f1 : f 1 a b = 10) :
  f 2 a b = 18 :=
sorry

end calculate_f_at_2_l228_228532


namespace last_number_aryana_counts_l228_228584

theorem last_number_aryana_counts (a d : ℤ) (h_start : a = 72) (h_diff : d = -11) :
  ∃ n : ℕ, (a + n * d > 0) ∧ (a + (n + 1) * d ≤ 0) ∧ a + n * d = 6 := by
  sorry

end last_number_aryana_counts_l228_228584


namespace ellipse_eq_max_area_AEBF_l228_228925

open Real

section ellipse_parabola_problem

variables {a b : ℝ} (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (x y k : ℝ) {M : ℝ × ℝ} {AO BO : ℝ} 
  (b_pos : 0 < b) (a_gt_b : b < a) (MF1_dist : abs (y - 1) = 5 / 3) (M_on_parabola : x^2 = 4 * y)
  (M_on_ellipse : (y / a)^2 + (x / b)^2 = 1) (A : ℝ × ℝ) (B : ℝ × ℝ) (D : ℝ × ℝ)
  (E F : ℝ × ℝ) (A_on_x : A.1 = b ∧ A.2 = 0) (B_on_y : B.1 = 0 ∧ B.2 = a)
  (D_intersect : D.2 = k * D.1) (E_on_ellipse : (E.2 / a)^2 + (E.1 / b)^2 = 1) 
  (F_on_ellipse : (F.2 / a)^2 + (F.1 / b)^2 = 1)
  (k_pos : 0 < k)

theorem ellipse_eq :
  a = 2 ∧ b = sqrt 3 → (y^2 / (2:ℝ)^2 + x^2 / (sqrt 3:ℝ)^2 = 1) :=
sorry

theorem max_area_AEBF :
  (a = 2 ∧ b = sqrt 3) →
  ∃ max_area : ℝ, max_area = 2 * sqrt 6 :=
sorry

end ellipse_parabola_problem

end ellipse_eq_max_area_AEBF_l228_228925


namespace pencils_placed_by_dan_l228_228726

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end pencils_placed_by_dan_l228_228726


namespace paige_finished_problems_l228_228416

-- Define the conditions
def initial_problems : ℕ := 110
def problems_per_page : ℕ := 9
def remaining_pages : ℕ := 7

-- Define the statement we want to prove
theorem paige_finished_problems :
  initial_problems - (remaining_pages * problems_per_page) = 47 :=
by sorry

end paige_finished_problems_l228_228416


namespace greatest_value_of_sum_l228_228800

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + (1/x)^2) : x + 1/x ≤ Real.sqrt 15 :=
sorry

end greatest_value_of_sum_l228_228800


namespace smallest_B_to_divisible_3_l228_228697

-- Define the problem
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the digits in the integer
def digit_sum (B : ℕ) : ℕ := 8 + B + 4 + 6 + 3 + 5

-- Prove that the smallest digit B that makes 8B4,635 divisible by 3 is 1
theorem smallest_B_to_divisible_3 : ∃ B : ℕ, B ≥ 0 ∧ B ≤ 9 ∧ is_divisible_by_3 (digit_sum B) ∧ ∀ B' : ℕ, B' < B → ¬ is_divisible_by_3 (digit_sum B') ∧ B = 1 :=
sorry

end smallest_B_to_divisible_3_l228_228697


namespace calculation_correct_l228_228483

theorem calculation_correct : (35 / (8 + 3 - 5) - 2) * 4 = 46 / 3 := by
  sorry

end calculation_correct_l228_228483


namespace find_a100_find_a1983_l228_228449

open Nat

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ k, a (a k) = 3 * k

theorem find_a100 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 100 = 181 := 
sorry

theorem find_a1983 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 1983 = 3762 := 
sorry

end find_a100_find_a1983_l228_228449


namespace circle_equation_l228_228594

-- Definitions for the given conditions
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-1, 1)
def line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 2 = 0

-- Theorem statement for the proof problem
theorem circle_equation :
  ∃ (h k : ℝ), line (h, k) ∧ (h = 1) ∧ (k = 1) ∧
  ((h - 1)^2 + (k - 1)^2 = 4) :=
sorry

end circle_equation_l228_228594


namespace simplified_expression_l228_228068

variable {x y : ℝ}

theorem simplified_expression 
  (P : ℝ := x^2 + y^2) 
  (Q : ℝ := x^2 - y^2) : 
  ( (P + 3 * Q) / (P - Q) - (P - 3 * Q) / (P + Q) ) = (2 * x^4 - y^4) / (x^2 * y^2) := 
  by sorry

end simplified_expression_l228_228068


namespace train_speed_kmh_l228_228756

variable (length_of_train_meters : ℕ) (time_to_cross_seconds : ℕ)

theorem train_speed_kmh (h1 : length_of_train_meters = 50) (h2 : time_to_cross_seconds = 6) :
  (length_of_train_meters * 3600) / (time_to_cross_seconds * 1000) = 30 :=
by
  sorry

end train_speed_kmh_l228_228756


namespace Sara_has_8_balloons_l228_228021

theorem Sara_has_8_balloons (Tom_balloons Sara_balloons total_balloons : ℕ)
  (htom : Tom_balloons = 9)
  (htotal : Tom_balloons + Sara_balloons = 17) :
  Sara_balloons = 8 :=
by
  sorry

end Sara_has_8_balloons_l228_228021


namespace combined_weight_chihuahua_pitbull_greatdane_l228_228747

noncomputable def chihuahua_pitbull_greatdane_combined_weight (C P G : ℕ) : ℕ :=
  C + P + G

theorem combined_weight_chihuahua_pitbull_greatdane :
  ∀ (C P G : ℕ), P = 3 * C → G = 3 * P + 10 → G = 307 → chihuahua_pitbull_greatdane_combined_weight C P G = 439 :=
by
  intros C P G h1 h2 h3
  sorry

end combined_weight_chihuahua_pitbull_greatdane_l228_228747


namespace problem_1_l228_228742

variable (x : ℝ) (a : ℝ)

theorem problem_1 (h1 : x - 1/x = 3) (h2 : a = x^2 + 1/x^2) : a = 11 := sorry

end problem_1_l228_228742


namespace pipe_filling_problem_l228_228731

theorem pipe_filling_problem (x : ℝ) (h : (2 / 15) * x + (1 / 20) * (10 - x) = 1) : x = 6 :=
sorry

end pipe_filling_problem_l228_228731


namespace difference_in_savings_correct_l228_228403

def S_last_year : ℝ := 45000
def saved_last_year_pct : ℝ := 0.083
def raise_pct : ℝ := 0.115
def saved_this_year_pct : ℝ := 0.056

noncomputable def saved_last_year_amount : ℝ := saved_last_year_pct * S_last_year
noncomputable def S_this_year : ℝ := S_last_year * (1 + raise_pct)
noncomputable def saved_this_year_amount : ℝ := saved_this_year_pct * S_this_year
noncomputable def difference_in_savings : ℝ := saved_last_year_amount - saved_this_year_amount

theorem difference_in_savings_correct :
  difference_in_savings = 925.20 := by
  sorry

end difference_in_savings_correct_l228_228403


namespace first_player_wins_if_not_power_of_two_l228_228163

/-- 
  Prove that the first player can guarantee a win if and only if $n$ is not a power of two, under the given conditions. 
-/
theorem first_player_wins_if_not_power_of_two
  (n : ℕ) (h : n > 1) :
  (∃ k : ℕ, n = 2^k) ↔ false :=
sorry

end first_player_wins_if_not_power_of_two_l228_228163


namespace water_fee_part1_water_fee_part2_water_fee_usage_l228_228065

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end water_fee_part1_water_fee_part2_water_fee_usage_l228_228065


namespace moles_of_water_from_reaction_l228_228898

def moles_of_water_formed (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  nh4cl_moles -- Because 1:1 ratio of reactants producing water

theorem moles_of_water_from_reaction :
  moles_of_water_formed 3 3 = 3 := by
  -- Use the condition of the 1:1 reaction ratio derivable from the problem's setup.
  sorry

end moles_of_water_from_reaction_l228_228898


namespace james_sushi_rolls_l228_228553

def fish_for_sushi : ℕ := 40
def total_fish : ℕ := 400
def bad_fish_percentage : ℕ := 20

theorem james_sushi_rolls :
  let good_fish := total_fish - (bad_fish_percentage * total_fish / 100)
  good_fish / fish_for_sushi = 8 :=
by
  sorry

end james_sushi_rolls_l228_228553


namespace slices_left_l228_228709

variable (total_pieces: ℕ) (joe_fraction: ℚ) (darcy_fraction: ℚ)
variable (carl_fraction: ℚ) (emily_fraction: ℚ)

theorem slices_left 
  (h1 : total_pieces = 24)
  (h2 : joe_fraction = 1/3)
  (h3 : darcy_fraction = 1/4)
  (h4 : carl_fraction = 1/6)
  (h5 : emily_fraction = 1/8) :
  total_pieces - (total_pieces * joe_fraction + total_pieces * darcy_fraction + total_pieces * carl_fraction + total_pieces * emily_fraction) = 3 := 
  by 
  sorry

end slices_left_l228_228709


namespace find_lowest_temperature_l228_228853

noncomputable def lowest_temperature 
(T1 T2 T3 T4 T5 : ℝ) : ℝ :=
if h : T1 + T2 + T3 + T4 + T5 = 200 ∧ max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 = 50 then
   min (min (min T1 T2) (min T3 T4)) T5
else 
  0

theorem find_lowest_temperature (T1 T2 T3 T4 T5 : ℝ) 
  (h_avg : T1 + T2 + T3 + T4 + T5 = 200)
  (h_range : max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 ≤ 50) : 
  lowest_temperature T1 T2 T3 T4 T5 = 30 := 
sorry

end find_lowest_temperature_l228_228853


namespace Emilee_earns_25_l228_228265

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l228_228265


namespace third_factor_of_product_l228_228191

theorem third_factor_of_product (w : ℕ) (h_w_pos : w > 0) (h_w_168 : w = 168)
  (w_factors : (936 * w) = 2^5 * 3^3 * x)
  (h36_factors : 2^5 ∣ (936 * w)) (h33_factors : 3^3 ∣ (936 * w)) : 
  (936 * w) / (2^5 * 3^3) = 182 :=
by {
  -- This is a placeholder. The actual proof is omitted.
  sorry
}

end third_factor_of_product_l228_228191


namespace find_n_l228_228519

theorem find_n : ∀ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) → (n = -7 / 2) := by
  intro n h
  sorry

end find_n_l228_228519


namespace odd_prime_divides_seq_implies_power_of_two_divides_l228_228160

theorem odd_prime_divides_seq_implies_power_of_two_divides (a : ℕ → ℤ) (p n : ℕ)
  (h0 : a 0 = 2)
  (hk : ∀ k, a (k + 1) = 2 * (a k) ^ 2 - 1)
  (h_odd_prime : Nat.Prime p)
  (h_odd : p % 2 = 1)
  (h_divides : ↑p ∣ a n) :
  2^(n + 3) ∣ p^2 - 1 :=
sorry

end odd_prime_divides_seq_implies_power_of_two_divides_l228_228160


namespace prob_X_lt_0_l228_228531

noncomputable def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry
noncomputable def normal_pdf (μ σ : ℝ) : ℝ → ℝ := sorry

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Conditions
axiom normal_dist : ∀ X, X = normal_cdf 1 σ
axiom prob_condition : ∀ P, P (0 < X 1) = 0.4

-- Question to prove
theorem prob_X_lt_0 : P (X 0) = 0.1 :=
by
  sorry

end prob_X_lt_0_l228_228531


namespace prime_p_prime_p₁₀_prime_p₁₄_l228_228037

theorem prime_p_prime_p₁₀_prime_p₁₄ (p : ℕ) (h₀p : Nat.Prime p) 
  (h₁ : Nat.Prime (p + 10)) (h₂ : Nat.Prime (p + 14)) : p = 3 := by
  sorry

end prime_p_prime_p₁₀_prime_p₁₄_l228_228037


namespace number_of_logs_in_stack_l228_228350

theorem number_of_logs_in_stack : 
    let first_term := 15 in
    let last_term := 5 in
    let num_terms := first_term - last_term + 1 in
    let average := (first_term + last_term) / 2 in
    let sum := average * num_terms in
    sum = 110 :=
by
  sorry

end number_of_logs_in_stack_l228_228350


namespace ladder_rung_length_l228_228957

noncomputable def ladder_problem : Prop :=
  let total_height_ft := 50
  let spacing_in := 6
  let wood_ft := 150
  let feet_to_inches(ft : ℕ) : ℕ := ft * 12
  let total_height_in := feet_to_inches total_height_ft
  let wood_in := feet_to_inches wood_ft
  let number_of_rungs := total_height_in / spacing_in
  let length_of_each_rung := wood_in / number_of_rungs
  length_of_each_rung = 18

theorem ladder_rung_length : ladder_problem := sorry

end ladder_rung_length_l228_228957


namespace probability_x_lt_2y_l228_228047

noncomputable def probability_x_lt_2y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1/2) * 4 * 2
  let area_rectangle : ℚ := 4 * 2
  (area_triangle / area_rectangle)

theorem probability_x_lt_2y (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 2) :
  probability_x_lt_2y_in_rectangle = 1/2 := by
  sorry

end probability_x_lt_2y_l228_228047


namespace max_bishops_1000x1000_l228_228451

def bishop_max_non_attacking (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem max_bishops_1000x1000 : bishop_max_non_attacking 1000 = 1998 :=
by sorry

end max_bishops_1000x1000_l228_228451


namespace pencil_length_total_l228_228752

theorem pencil_length_total :
  (1.5 + 0.5 + 2 + 1.25 + 0.75 + 1.8 + 2.5 = 10.3) :=
by
  sorry

end pencil_length_total_l228_228752


namespace at_least_one_not_less_than_2_l228_228443

theorem at_least_one_not_less_than_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_2_l228_228443


namespace complex_computation_l228_228212

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l228_228212


namespace sufficient_condition_for_parallel_l228_228235

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Definitions of parallelism and perpendicularity
variable {Parallel Perpendicular : Line → Plane → Prop}
variable {ParallelLines : Line → Line → Prop}

-- Definition of subset relation
variable {Subset : Line → Plane → Prop}

-- Theorems or conditions
variables (a b : Line) (α β : Plane)

-- Assertion of the theorem
theorem sufficient_condition_for_parallel (h1 : ParallelLines a b) (h2 : Parallel b α) (h3 : ¬ Subset a α) : Parallel a α :=
sorry

end sufficient_condition_for_parallel_l228_228235


namespace largest_non_sum_of_36_and_composite_l228_228173

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l228_228173


namespace sequence_problem_l228_228257

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m k, n ≠ m → a n = a m + (n - m) * k

theorem sequence_problem
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 :=
sorry

end sequence_problem_l228_228257


namespace division_of_decimals_l228_228066

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := 
sorry

end division_of_decimals_l228_228066


namespace find_a_if_f_is_odd_function_l228_228247

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (a * 2^x - 2^(-x))

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_a_if_f_is_odd_function : 
  ∀ a : ℝ, is_odd_function (f a) → a = 1 :=
by
  sorry

end find_a_if_f_is_odd_function_l228_228247


namespace maria_change_l228_228411

def cost_per_apple : ℝ := 0.75
def number_of_apples : ℕ := 5
def amount_paid : ℝ := 10.0
def total_cost := number_of_apples * cost_per_apple
def change_received := amount_paid - total_cost

theorem maria_change :
  change_received = 6.25 :=
sorry

end maria_change_l228_228411


namespace distribution_ways_numbers_closer_to_center_not_smaller_sum_of_numbers_in_rings_equal_l228_228307

-- Conditions and Definitions
def parts_of_target := 10
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def center_value := 10

-- Problem (a)
theorem distribution_ways :
    (10.factorial = 3628800) :=
by
    sorry  -- Proof not required

-- Problem (b)
theorem numbers_closer_to_center_not_smaller :
    ∃ counts : ℕ,
    (counts = 4320) :=
by
    sorry  -- Proof not required

-- Problem (c)
theorem sum_of_numbers_in_rings_equal :
    ∃ counts : ℕ,
    (counts = 34560) :=
by
    sorry  -- Proof not required

end distribution_ways_numbers_closer_to_center_not_smaller_sum_of_numbers_in_rings_equal_l228_228307


namespace complete_square_l228_228329

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l228_228329


namespace div_by_7_l228_228981

theorem div_by_7 (n : ℕ) (h : n ≥ 1) : 7 ∣ (8^n + 6) :=
sorry

end div_by_7_l228_228981


namespace roots_geometric_progression_condition_l228_228722

theorem roots_geometric_progression_condition 
  (a b c : ℝ) 
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = -a)
  (h2 : x1 * x2 + x2 * x3 + x1 * x3 = b)
  (h3 : x1 * x2 * x3 = -c)
  (h4 : x2^2 = x1 * x3) :
  a^3 * c = b^3 :=
sorry

end roots_geometric_progression_condition_l228_228722


namespace heartsuit_4_6_l228_228101

-- Define the operation \heartsuit
def heartsuit (x y : ℤ) : ℤ := 5 * x + 3 * y

-- Prove that 4 \heartsuit 6 = 38 under the given operation definition
theorem heartsuit_4_6 : heartsuit 4 6 = 38 := by
  -- Using the definition of \heartsuit
  -- Calculation is straightforward and skipped by sorry
  sorry

end heartsuit_4_6_l228_228101


namespace marie_socks_problem_l228_228832

theorem marie_socks_problem (x y z : ℕ) : 
  x + y + z = 15 → 
  2 * x + 3 * y + 5 * z = 36 → 
  1 ≤ x → 
  1 ≤ y → 
  1 ≤ z → 
  x = 11 :=
by
  sorry

end marie_socks_problem_l228_228832


namespace find_a7_l228_228488

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l228_228488


namespace vertical_axis_residuals_of_residual_plot_l228_228548

theorem vertical_axis_residuals_of_residual_plot :
  ∀ (vertical_axis : Type), 
  (vertical_axis = Residuals ∨ 
   vertical_axis = SampleNumber ∨ 
   vertical_axis = EstimatedValue) →
  (vertical_axis = Residuals) :=
by
  sorry

end vertical_axis_residuals_of_residual_plot_l228_228548


namespace mark_percentage_increase_l228_228951

-- Given a game with the following conditions:
-- Condition 1: Samanta has 8 more points than Mark
-- Condition 2: Eric has 6 points
-- Condition 3: The total points of Samanta, Mark, and Eric is 32

theorem mark_percentage_increase (S M : ℕ) (h1 : S = M + 8) (h2 : 6 + S + M = 32) : 
  (M - 6) / 6 * 100 = 50 :=
sorry

end mark_percentage_increase_l228_228951


namespace boat_speed_still_water_l228_228183

/-- Proof that the speed of the boat in still water is 10 km/hr given the conditions -/
theorem boat_speed_still_water (V_b V_s : ℝ) 
  (cond1 : V_b + V_s = 15) 
  (cond2 : V_b - V_s = 5) : 
  V_b = 10 :=
by
  sorry

end boat_speed_still_water_l228_228183


namespace collinear_points_inverse_sum_half_l228_228253

theorem collinear_points_inverse_sum_half (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
    (collinear : (a - 2) * (b - 2) - (-2) * a = 0) : 
    1 / a + 1 / b = 1 / 2 := 
by
  sorry

end collinear_points_inverse_sum_half_l228_228253


namespace polygon_sides_l228_228414

-- Definition of the problem conditions
def interiorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)
def givenAngleSum (n : ℕ) : ℕ := 140 + 145 * (n - 1)

-- Problem statement: proving the number of sides
theorem polygon_sides (n : ℕ) (h : interiorAngleSum n = givenAngleSum n) : n = 10 :=
sorry

end polygon_sides_l228_228414


namespace petya_can_prevent_natural_sum_l228_228568

def petya_prevents_natural_sum : Prop :=
  ∀ (fractions : List ℚ),
    (∀ f ∈ fractions, ∃ n : ℕ, f = 1 / n) →
    ∃ m : ℕ, let new_fractions := (1 / m) :: fractions in
    ∀ k : ℕ, 
      k > 0 →
      let vasya_fractions := new_fractions.take k in
      ∑ i in vasya_fractions, id i ∉ ℕ

theorem petya_can_prevent_natural_sum : petya_prevents_natural_sum :=
sorry

end petya_can_prevent_natural_sum_l228_228568


namespace number_of_pecan_pies_is_4_l228_228627

theorem number_of_pecan_pies_is_4 (apple_pies pumpkin_pies total_pies pecan_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pumpkin_pies = 7) 
  (h3 : total_pies = 13) 
  (h4 : pecan_pies = total_pies - (apple_pies + pumpkin_pies)) 
  : pecan_pies = 4 := 
by 
  sorry

end number_of_pecan_pies_is_4_l228_228627


namespace find_first_term_l228_228268

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l228_228268


namespace amount_of_c_l228_228739

theorem amount_of_c (A B C : ℕ) (h1 : A + B + C = 350) (h2 : A + C = 200) (h3 : B + C = 350) : C = 200 :=
sorry

end amount_of_c_l228_228739


namespace find_a2_given_conditions_l228_228923

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem find_a2_given_conditions
  {a : ℕ → ℤ}
  (h_seq : is_arithmetic_sequence a)
  (h1 : a 3 + a 5 = 24)
  (h2 : a 7 - a 3 = 24) :
  a 2 = 0 :=
by
  sorry

end find_a2_given_conditions_l228_228923


namespace find_coefficients_l228_228706

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def h (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_coefficients :
  ∃ a b c : ℝ, (∀ s : ℝ, f s = 0 → h a b c (s^3) = 0) ∧
    (a, b, c) = (-6, -9, 20) :=
sorry

end find_coefficients_l228_228706


namespace find_angle_A_find_triangle_area_l228_228563

-- Definition of the problem
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
variables (ha : a = Real.sqrt 3)
variables (hc : sin C = (1 + Real.sqrt 3) / 2 * sin B)
variables (eq_cond : b * (sin B - sin C) + (c - a) * (sin A + sin C) = 0)
variable (A_value : A = π / 3)
variable (triangle_area : 1 / 2 * a * b * sin C = (3 + Real.sqrt 3) / 4)

-- Theorem statements
theorem find_angle_A : A = π / 3 :=
sorry

theorem find_triangle_area : 
  (1 / 2 * (Real.sqrt 3) * b * sin ((π / 3 + B) / 2) = (3 + Real.sqrt 3) / 4) :=
sorry

end find_angle_A_find_triangle_area_l228_228563


namespace ratio_of_members_l228_228481

theorem ratio_of_members (f m c : ℕ) 
  (h1 : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  2 * f + m = 3 * c :=
by
  sorry

end ratio_of_members_l228_228481


namespace rose_share_correct_l228_228985

-- Define the conditions
def purity_share (P : ℝ) : ℝ := P
def sheila_share (P : ℝ) : ℝ := 5 * P
def rose_share (P : ℝ) : ℝ := 3 * P
def total_rent := 5400

-- The theorem to be proven
theorem rose_share_correct (P : ℝ) (h : purity_share P + sheila_share P + rose_share P = total_rent) : 
  rose_share P = 1800 :=
  sorry

end rose_share_correct_l228_228985


namespace geometric_N_digit_not_20_l228_228375

-- Variables and definitions
variables (a b c : ℕ)

-- Given conditions
def geometric_progression (a b c : ℕ) : Prop :=
  ∃ q : ℚ, (b = q * a) ∧ (c = q * b)

def ends_with_20 (N : ℕ) : Prop := N % 100 = 20

-- Prove the main theorem
theorem geometric_N_digit_not_20 (h1 : geometric_progression a b c) (h2 : ends_with_20 (a^3 + b^3 + c^3 - 3 * a * b * c)) :
  False :=
sorry

end geometric_N_digit_not_20_l228_228375


namespace jenna_round_trip_pay_l228_228818

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end jenna_round_trip_pay_l228_228818


namespace arrange_squares_l228_228267

theorem arrange_squares (n : ℕ) (h : n ≥ 5) :
  ∃ arrangement : Fin n → Fin n × Fin n, 
    (∀ i j : Fin n, i ≠ j → 
      (arrangement i).fst + (arrangement i).snd = (arrangement j).fst + (arrangement j).snd
      ∨ (arrangement i).fst = (arrangement j).fst
      ∨ (arrangement i).snd = (arrangement j).snd) :=
sorry

end arrange_squares_l228_228267


namespace line_slope_and_point_l228_228749

noncomputable def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem line_slope_and_point (m b : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 5) (h₃ : y₀ = 2) (h₄ : y₀ = line_equation x₀ m b) :
  m + b = 14 :=
by
  sorry

end line_slope_and_point_l228_228749


namespace player_A_elimination_after_third_round_at_least_one_player_passes_all_l228_228462

-- Define probabilities for Player A's success in each round
def P_A1 : ℚ := 4 / 5
def P_A2 : ℚ := 3 / 4
def P_A3 : ℚ := 2 / 3

-- Define probabilities for Player B's success in each round
def P_B1 : ℚ := 2 / 3
def P_B2 : ℚ := 2 / 3
def P_B3 : ℚ := 1 / 2

-- Define theorems
theorem player_A_elimination_after_third_round :
  P_A1 * P_A2 * (1 - P_A3) = 1 / 5 := by
  sorry

theorem at_least_one_player_passes_all :
  1 - ((1 - (P_A1 * P_A2 * P_A3)) * (1 - (P_B1 * P_B2 * P_B3))) = 8 / 15 := by
  sorry


end player_A_elimination_after_third_round_at_least_one_player_passes_all_l228_228462


namespace given_conditions_implies_correct_answer_l228_228238

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end given_conditions_implies_correct_answer_l228_228238


namespace simplify_fraction_l228_228426

-- Define the fraction and the GCD condition
def fraction_numerator : ℕ := 66
def fraction_denominator : ℕ := 4356
def gcd_condition : ℕ := Nat.gcd fraction_numerator fraction_denominator

-- State the theorem that the fraction simplifies to 1/66 given the GCD condition
theorem simplify_fraction (h : gcd_condition = 66) : (fraction_numerator / fraction_denominator = 1 / 66) :=
  sorry

end simplify_fraction_l228_228426


namespace value_of_a2_l228_228233

theorem value_of_a2 (a : ℕ → ℤ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 2)
  (h2 : ∃ r : ℤ, a 3 = r * a 1 ∧ a 4 = r * a 3) :
  a 2 = -6 :=
by
  sorry

end value_of_a2_l228_228233


namespace trig_identity_1_trig_identity_2_l228_228918

theorem trig_identity_1 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.sin (3 * π / 2 + θ)) / 
  (3 * Real.sin (π / 2 - θ) - 2 * Real.sin (π + θ)) = 1 / 7 :=
by sorry

theorem trig_identity_2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (1 - Real.cos (2 * θ)) / 
  (Real.sin (2 * θ) + Real.cos (2 * θ)) = 8 :=
by sorry

end trig_identity_1_trig_identity_2_l228_228918


namespace workers_count_l228_228314

noncomputable def numberOfWorkers (W: ℕ) : Prop :=
  let old_supervisor_salary := 870
  let new_supervisor_salary := 690
  let avg_old := 430
  let avg_new := 410
  let total_after_old := (W + 1) * avg_old
  let total_after_new := 9 * avg_new
  total_after_old - old_supervisor_salary = total_after_new - new_supervisor_salary

theorem workers_count : numberOfWorkers 8 :=
by
  sorry

end workers_count_l228_228314


namespace solve_for_x_l228_228446

-- Define the problem with the given conditions
def sum_of_triangle_angles (x : ℝ) : Prop := x + 2 * x + 30 = 180

-- State the theorem
theorem solve_for_x : ∀ (x : ℝ), sum_of_triangle_angles x → x = 50 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l228_228446


namespace sequence_length_arithmetic_sequence_l228_228493

theorem sequence_length_arithmetic_sequence :
  ∀ (a d l n : ℕ), a = 5 → d = 3 → l = 119 → l = a + (n - 1) * d → n = 39 :=
by
  intros a d l n ha hd hl hln
  sorry

end sequence_length_arithmetic_sequence_l228_228493


namespace edges_sum_l228_228882

def edges_triangular_pyramid : ℕ := 6
def edges_triangular_prism : ℕ := 9

theorem edges_sum : edges_triangular_pyramid + edges_triangular_prism = 15 :=
by
  sorry

end edges_sum_l228_228882


namespace max_triangle_area_l228_228526

theorem max_triangle_area (a b c : ℝ) (h1 : b + c = 8) (h2 : a + b > c)
  (h3 : a + c > b) (h4 : b + c > a) :
  (a - b + c) * (a + b - c) ≤ 64 / 17 :=
by sorry

end max_triangle_area_l228_228526


namespace arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l228_228368

open Nat

variable (a : ℕ → ℝ)
variable (c : ℕ → ℝ)
variable (k b : ℝ)

-- Condition 1: sequence definition
def sequence_condition := ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + n + 1

-- Condition 2: initial value
def initial_value := a 1 = -1

-- Condition 3: c_n definition
def geometric_sequence_condition := ∀ n : ℕ, 0 < n → c (n + 1) / c n = 2

-- Problem 1: Arithmetic sequence parameters
theorem arith_sequence_parameters (h1 : sequence_condition a) (h2 : initial_value a) : a 1 = -3 ∧ 2 * (a 1 + 2) - a 1 - 7 = -1 :=
by sorry

-- Problem 2: Cannot be a geometric sequence
theorem not_geo_sequence (h1 : sequence_condition a) (h2 : initial_value a) : ¬ (∃ q, ∀ n : ℕ, 0 < n → a n * q = a (n + 1)) :=
by sorry

-- Problem 3: c_n is a geometric sequence and general term for a_n
theorem geo_sequence_and_gen_term (h1 : sequence_condition a) (h2 : initial_value a) 
    (h3 : ∀ n : ℕ, 0 < n → c n = a n + k * n + b)
    (hk : k = 1) (hb : b = 2) : sequence_condition a ∧ initial_value a :=
by sorry

end arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l228_228368


namespace range_of_m_value_of_m_l228_228093

-- Defining the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- The condition for having real roots
def has_real_roots (a b c : ℝ) := b^2 - 4 * a * c ≥ 0

-- First part: Range of values for m
theorem range_of_m (m : ℝ) : has_real_roots 1 (-2) (m - 1) ↔ m ≤ 2 := sorry

-- Second part: Finding m when x₁² + x₂² = 6x₁x₂
theorem value_of_m 
  (x₁ x₂ m : ℝ) (h₁ : quadratic_eq 1 (-2) (m - 1) x₁) (h₂ : quadratic_eq 1 (-2) (m - 1) x₂) 
  (h_sum : x₁ + x₂ = 2) (h_prod : x₁ * x₂ = m - 1) (h_condition : x₁^2 + x₂^2 = 6 * (x₁ * x₂)) : 
  m = 3 / 2 := sorry

end range_of_m_value_of_m_l228_228093


namespace probability_product_zero_probability_product_negative_l228_228378

def given_set : List ℤ := [-3, -2, -1, 0, 5, 6, 7]

def num_pairs : ℕ := 21

theorem probability_product_zero :
  (6 : ℚ) / num_pairs = 2 / 7 := sorry

theorem probability_product_negative :
  (9 : ℚ) / num_pairs = 3 / 7 := sorry

end probability_product_zero_probability_product_negative_l228_228378


namespace max_distance_of_MN_l228_228094

noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def curve_C_cartesian (x y : ℝ) := x^2 + y^2 - 2 * x

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ( -1 + (Real.sqrt 5 / 5) * t, (2 * Real.sqrt 5 / 5) * t)

def point_M : ℝ × ℝ := (0, 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def center_C : ℝ × ℝ := (1, 0)

theorem max_distance_of_MN :
  ∃ N : ℝ × ℝ, 
  ∀ (θ : ℝ), N = (curve_C_polar θ * Real.cos θ, curve_C_polar θ * Real.sin θ) →
  distance point_M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_of_MN_l228_228094


namespace gift_boxes_in_3_days_l228_228221
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_l228_228221


namespace ellipse_hyperbola_same_foci_l228_228856

theorem ellipse_hyperbola_same_foci (k : ℝ) (h1 : k > 0) :
  (∀ (x y : ℝ), (x^2 / 9 + y^2 / k^2 = 1) ↔ (x^2 / k - y^2 / 3 = 1)) → k = 2 :=
by
  sorry

end ellipse_hyperbola_same_foci_l228_228856


namespace geometric_progression_odd_last_term_l228_228465

theorem geometric_progression_odd_last_term :
  ∃ (n : ℕ), (∃ (a : ℕ) (r : ℚ), a = 10 ^ 2015 ∧ ∃ k : ℤ, r = k / a ∧ (10 ^ 2015) * r = odd_integer a n) → n = 8 :=
by sorry

def odd_integer (a : ℕ) (n : ℕ) (r : ℕ) : Prop :=
  (∃ (a1 an : ℕ), an = a1 * r^(n-1) ∧ a1 = 10^2015 ∧ (an % 2 = 1))

end geometric_progression_odd_last_term_l228_228465


namespace age_problem_l228_228254

variable (A B : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 5) : B = 35 := by
  sorry

end age_problem_l228_228254


namespace find_first_term_arithmetic_sequence_l228_228274

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l228_228274


namespace number_of_children_l228_228469

-- Definitions for the conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_amount : ℕ := 35

-- Theorem stating the proof problem
theorem number_of_children (A C T : ℕ) (hc: A = adult_ticket_cost) (ha: C = child_ticket_cost) (ht: T = total_amount) :
  (T - A) / C = 9 :=
by
  sorry

end number_of_children_l228_228469


namespace punctures_covered_l228_228623

theorem punctures_covered (P1 P2 P3 : ℝ) (h1 : 0 ≤ P1) (h2 : P1 < P2) (h3 : P2 < P3) (h4 : P3 < 3) :
    ∃ x, x ≤ P1 ∧ x + 2 ≥ P3 := 
sorry

end punctures_covered_l228_228623


namespace common_chord_of_circles_is_x_eq_y_l228_228500

theorem common_chord_of_circles_is_x_eq_y :
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x - 3 = 0) ∧ (x^2 + y^2 - 4 * y - 3 = 0) → (x = y) :=
by
  sorry

end common_chord_of_circles_is_x_eq_y_l228_228500


namespace P_2017_eq_14_l228_228428

def sumOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def numberOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.length

def P (n : Nat) : Nat :=
  sumOfDigits n + numberOfDigits n

theorem P_2017_eq_14 : P 2017 = 14 :=
by
  sorry

end P_2017_eq_14_l228_228428


namespace unique_positive_b_for_discriminant_zero_l228_228501

theorem unique_positive_b_for_discriminant_zero (c : ℝ) : 
  (∃! b : ℝ, b > 0 ∧ (b^2 + 1/b^2)^2 - 4 * c = 0) → c = 1 :=
by
  sorry

end unique_positive_b_for_discriminant_zero_l228_228501


namespace problem_solution_l228_228359

open Finset

def minimally_intersecting (A B C : Finset ℕ) : Prop :=
  |A ∩ B| = 1 ∧ |B ∩ C| = 1 ∧ |C ∩ A| = 1 ∧ (A ∩ B ∩ C) = ∅

def N : ℕ :=
  ((Finset.powerset (range 8)).filter (λ A B C, minimally_intersecting A B C)).card

theorem problem_solution : N % 1000 = 344 :=
sorry

end problem_solution_l228_228359


namespace jenna_round_trip_pay_l228_228821

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end jenna_round_trip_pay_l228_228821


namespace smallest_number_divisible_l228_228321

theorem smallest_number_divisible (x : ℕ) : 
  (∃ x, x + 7 % 8 = 0 ∧ x + 7 % 11 = 0 ∧ x + 7 % 24 = 0) ∧
  (∀ y, (y + 7 % 8 = 0 ∧ y + 7 % 11 = 0 ∧ y + 7 % 24 = 0) → 257 ≤ y) :=
by { sorry }

end smallest_number_divisible_l228_228321


namespace problem_f_g_comp_sum_l228_228966

-- Define the functions
def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

-- Define the statement we want to prove
theorem problem_f_g_comp_sum (x : ℚ) (h : x = 2) : f (g x) + g (f x) = 36 / 5 := by
  sorry

end problem_f_g_comp_sum_l228_228966


namespace james_age_when_tim_is_79_l228_228115

variable {James_age John_age Tim_age : ℕ}

theorem james_age_when_tim_is_79 (J_age J_age_at_23 J_age_diff J_age_j : ℕ) 
                                  (H1 : J_age = J_age_at_23 - J_age_diff)
                                  (H2 : John_age = 35)
                                  (H3 : James_age = 23)
                                  (age_diff:12: ℕ )
                                  (H4 : Tim_age = 2 * John_age - 5)
                                  (H5 : Tim_age = 79):
                                  J_age=30 :=
by
  sorry

end james_age_when_tim_is_79_l228_228115


namespace function_has_one_zero_l228_228691

-- Define the function f
def f (x m : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

-- State the theorem
theorem function_has_one_zero (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m = 0 ∨ m = -3 := 
sorry

end function_has_one_zero_l228_228691


namespace tank_loss_rate_after_first_repair_l228_228043

def initial_capacity : ℕ := 350000
def first_loss_rate : ℕ := 32000
def first_loss_duration : ℕ := 5
def second_loss_duration : ℕ := 10
def filling_rate : ℕ := 40000
def filling_duration : ℕ := 3
def missing_gallons : ℕ := 140000

noncomputable def first_repair_loss_rate := (initial_capacity - (first_loss_rate * first_loss_duration) + (filling_rate * filling_duration) - (initial_capacity - missing_gallons)) / second_loss_duration

theorem tank_loss_rate_after_first_repair : first_repair_loss_rate = 10000 := by sorry

end tank_loss_rate_after_first_repair_l228_228043


namespace emilee_earns_25_l228_228263

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l228_228263


namespace P_equals_neg12_l228_228365

def P (a b : ℝ) : ℝ :=
  (2 * a + 3 * b)^2 - (2 * a + b) * (2 * a - b) - 2 * b * (3 * a + 5 * b)

lemma simplified_P (a b : ℝ) : P a b = 6 * a * b :=
  by sorry

theorem P_equals_neg12 (a b : ℝ) (h : b = -2 / a) : P a b = -12 :=
  by sorry

end P_equals_neg12_l228_228365


namespace gumballs_remaining_l228_228056

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end gumballs_remaining_l228_228056


namespace number_of_people_and_price_l228_228954

theorem number_of_people_and_price 
  (x y : ℤ) 
  (h1 : 8 * x - y = 3) 
  (h2 : y - 7 * x = 4) : 
  x = 7 ∧ y = 53 :=
by
  sorry

end number_of_people_and_price_l228_228954


namespace smallest_multiple_l228_228509

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l228_228509


namespace geometric_sum_S_40_l228_228808

variable (S : ℕ → ℝ)

-- Conditions
axiom sum_S_10 : S 10 = 18
axiom sum_S_20 : S 20 = 24

-- Proof statement
theorem geometric_sum_S_40 : S 40 = 80 / 3 :=
by
  sorry

end geometric_sum_S_40_l228_228808


namespace dog_catches_fox_at_distance_l228_228873

def initial_distance : ℝ := 30
def dog_leap_distance : ℝ := 2
def fox_leap_distance : ℝ := 1
def dog_leaps_per_time_unit : ℝ := 2
def fox_leaps_per_time_unit : ℝ := 3

noncomputable def dog_speed : ℝ := dog_leaps_per_time_unit * dog_leap_distance
noncomputable def fox_speed : ℝ := fox_leaps_per_time_unit * fox_leap_distance
noncomputable def relative_speed : ℝ := dog_speed - fox_speed
noncomputable def time_to_catch := initial_distance / relative_speed
noncomputable def distance_dog_runs := time_to_catch * dog_speed

theorem dog_catches_fox_at_distance :
  distance_dog_runs = 120 :=
  by sorry

end dog_catches_fox_at_distance_l228_228873


namespace math_problem_l228_228088

noncomputable def x : ℝ := -2

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}
def C1 : Set ℝ := {1, 3}
def C2 : Set ℝ := {3, 4}

theorem math_problem
  (h1 : B x ⊆ A x) :
  x = -2 ∧ (B x ∪ C1 = A x ∨ B x ∪ C2 = A x) :=
by
  sorry

end math_problem_l228_228088


namespace winning_percentage_is_70_l228_228105

def percentage_of_votes (P : ℝ) : Prop :=
  ∃ (P : ℝ), (7 * P - 7 * (100 - P) = 280 ∧ 0 ≤ P ∧ P ≤ 100)

theorem winning_percentage_is_70 :
  percentage_of_votes 70 :=
by
  sorry

end winning_percentage_is_70_l228_228105


namespace parabola_vertex_y_l228_228455

theorem parabola_vertex_y (x : ℝ) : (∃ (h k : ℝ), (4 * (x - h)^2 + k = 4 * x^2 + 16 * x + 11) ∧ k = -5) := 
  sorry

end parabola_vertex_y_l228_228455


namespace jenna_round_trip_pay_l228_228820

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end jenna_round_trip_pay_l228_228820


namespace probability_different_grandchildren_count_l228_228285

theorem probability_different_grandchildren_count :
  let total_grandchildren := 12
  let total_variations := 2 ^ total_grandchildren
  let comb := Nat.choose total_grandchildren (total_grandchildren / 2)
  let prob_equal := comb / total_variations
  let prob_different := 1 - prob_equal
  prob_different = 793 / 1024 := by
sorry

end probability_different_grandchildren_count_l228_228285


namespace points_difference_l228_228643

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end points_difference_l228_228643


namespace vector_sum_l228_228937

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (-1, 0, 1)

-- Define the target vector c
def c : ℝ × ℝ × ℝ := (-1, 2, 5)

-- State the theorem to be proven
theorem vector_sum : a + (2:ℝ) • b = c :=
by 
  -- Not providing the proof, just adding a sorry
  sorry

end vector_sum_l228_228937


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l228_228171

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l228_228171


namespace smallest_pos_value_correct_l228_228906

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l228_228906


namespace solution_set_of_inequality_l228_228251

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality :
  (∀ x, f (x) = f (-x)) →               -- f(x) is even
  (∀ x y, 0 < x → x < y → f y ≤ f x) →   -- f(x) is monotonically decreasing on (0, +∞)
  f 2 = 0 →                              -- f(2) = 0
  {x : ℝ | (f x + f (-x)) / (3 * x) < 0} = 
    {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
by sorry

end solution_set_of_inequality_l228_228251


namespace area_of_large_rectangle_ABCD_l228_228717

-- Definitions for conditions and given data
def shaded_rectangle_area : ℕ := 2
def area_of_rectangle_ABCD (a b c : ℕ) : ℕ := a + b + c

-- The theorem to prove
theorem area_of_large_rectangle_ABCD
  (a b c : ℕ) 
  (h1 : shaded_rectangle_area = a)
  (h2 : shaded_rectangle_area = b)
  (h3 : a + b + c = 8) : 
  area_of_rectangle_ABCD a b c = 8 :=
by
  sorry

end area_of_large_rectangle_ABCD_l228_228717


namespace cos_theta_correct_projection_correct_l228_228681

noncomputable def vec_a : ℝ × ℝ := (2, 3)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (norm_a * norm_b)

noncomputable def projection (b : ℝ × ℝ) (cosθ : ℝ) : ℝ :=
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  norm_b * cosθ

theorem cos_theta_correct :
  cos_theta vec_a vec_b = 4 * Real.sqrt 65 / 65 :=
by
  sorry

theorem projection_correct :
  projection vec_b (cos_theta vec_a vec_b) = 8 * Real.sqrt 13 / 13 :=
by
  sorry

end cos_theta_correct_projection_correct_l228_228681


namespace max_height_of_basketball_l228_228457

def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

theorem max_height_of_basketball : ∃ t : ℝ, h t = 127 :=
by
  use 5
  sorry

end max_height_of_basketball_l228_228457


namespace race_positions_l228_228564

variable (nabeel marzuq arabi rafsan lian rahul : ℕ)

theorem race_positions :
  (arabi = 6) →
  (arabi = rafsan + 1) →
  (rafsan = rahul + 2) →
  (rahul = nabeel + 1) →
  (nabeel = marzuq + 6) →
  (marzuq = 8) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end race_positions_l228_228564


namespace third_student_gold_stickers_l228_228313

theorem third_student_gold_stickers:
  ∃ (n : ℕ), n = 41 ∧ 
  (∃ (a1 a2 a4 a5 a6 : ℕ), 
    a1 = 29 ∧ 
    a2 = 35 ∧ 
    a4 = 47 ∧ 
    a5 = 53 ∧ 
    a6 = 59 ∧ 
    a2 - a1 = 6 ∧ 
    a5 - a4 = 6 ∧ 
    ∀ k, k = 3 → n = a2 + 6) := 
sorry

end third_student_gold_stickers_l228_228313


namespace geometric_sequence_S5_equals_l228_228111

theorem geometric_sequence_S5_equals :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    a 1 = 1 → 
    (a 3 + a 4) / (a 1 + a 2) = 4 → 
    ((S5 : ℤ) = 31 ∨ (S5 : ℤ) = 11) :=
by
  sorry

end geometric_sequence_S5_equals_l228_228111


namespace tariffs_impact_but_no_timeframe_l228_228039

noncomputable def cost_of_wine_today : ℝ := 20.00
noncomputable def increase_percentage : ℝ := 0.25
noncomputable def bottles_count : ℕ := 5
noncomputable def price_increase_for_bottles : ℝ := 25.00

theorem tariffs_impact_but_no_timeframe :
  ¬ ∃ (t : ℝ), (cost_of_wine_today * (1 + increase_percentage) - cost_of_wine_today) * bottles_count = price_increase_for_bottles →
  (t = sorry) :=
by 
  sorry

end tariffs_impact_but_no_timeframe_l228_228039


namespace find_plane_speed_l228_228033

-- Defining the values in the problem
def distance_with_wind : ℝ := 420
def distance_against_wind : ℝ := 350
def wind_speed : ℝ := 23

-- The speed of the plane in still air
def plane_speed_in_still_air : ℝ := 253

-- Proof goal: Given the conditions, the speed of the plane in still air is 253 mph
theorem find_plane_speed :
  ∃ p : ℝ, (distance_with_wind / (p + wind_speed) = distance_against_wind / (p - wind_speed)) ∧ p = plane_speed_in_still_air :=
by
  use plane_speed_in_still_air
  have h : plane_speed_in_still_air = 253 := rfl
  sorry

end find_plane_speed_l228_228033


namespace jonah_first_intermission_lemonade_l228_228657

theorem jonah_first_intermission_lemonade :
  ∀ (l1 l2 l3 l_total : ℝ)
  (h1 : l2 = 0.42)
  (h2 : l3 = 0.25)
  (h3 : l_total = 0.92)
  (h4 : l_total = l1 + l2 + l3),
  l1 = 0.25 :=
by sorry

end jonah_first_intermission_lemonade_l228_228657


namespace no_possible_values_for_b_l228_228148

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_possible_values_for_b_l228_228148


namespace greg_needs_additional_amount_l228_228380

def total_cost : ℤ := 90
def saved_amount : ℤ := 57
def additional_amount_needed : ℤ := total_cost - saved_amount

theorem greg_needs_additional_amount :
  additional_amount_needed = 33 :=
by
  sorry

end greg_needs_additional_amount_l228_228380


namespace calc_expression_l228_228763

theorem calc_expression : 3 ^ 2022 * (1 / 3) ^ 2023 = 1 / 3 :=
by
  sorry

end calc_expression_l228_228763


namespace parabola_focus_coords_l228_228716

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus coordinates
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- The math proof problem statement
theorem parabola_focus_coords :
  ∀ x y, parabola x y → focus x y :=
by
  intros x y hp
  sorry

end parabola_focus_coords_l228_228716


namespace number_of_children_to_movies_l228_228470

theorem number_of_children_to_movies (adult_ticket_cost child_ticket_cost total_money : ℕ) 
  (h_adult_ticket_cost : adult_ticket_cost = 8) 
  (h_child_ticket_cost : child_ticket_cost = 3) 
  (h_total_money : total_money = 35) : 
  (total_money - adult_ticket_cost) / child_ticket_cost = 9 := 
by 
  have remaining_money : ℕ := total_money - adult_ticket_cost
  have h_remaining_money : remaining_money = 27 := by 
    rw [h_total_money, h_adult_ticket_cost]
    norm_num
  rw [h_remaining_money]
  exact nat.div_eq_of_eq_mul_right (by norm_num)
    (by norm_num)

end number_of_children_to_movies_l228_228470


namespace triangle_obtuse_of_eccentricities_l228_228252

noncomputable def is_obtuse_triangle (a b m : ℝ) : Prop :=
  a^2 + b^2 - m^2 < 0

theorem triangle_obtuse_of_eccentricities (a b m : ℝ) (ha : a > 0) (hm : m > b) (hb : b > 0)
  (ecc_cond : (Real.sqrt (a^2 + b^2) / a) * (Real.sqrt (m^2 - b^2) / m) > 1) :
  is_obtuse_triangle a b m := 
sorry

end triangle_obtuse_of_eccentricities_l228_228252


namespace student_in_16th_group_has_number_244_l228_228950

theorem student_in_16th_group_has_number_244 :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 800 ∧ ((k - 36) % 16 = 0) ∧ (n = 3 + (k - 36) / 16)) →
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ 800 ∧ ((m - 244) % 16 = 0) ∧ (16 = 3 + (m - 36) / 16) :=
by
  sorry

end student_in_16th_group_has_number_244_l228_228950


namespace sweetsies_remainder_l228_228659

-- Each definition used in Lean 4 statement should be directly from the conditions in a)
def number_of_sweetsies_in_one_bag (m : ℕ): Prop :=
  m % 8 = 5

theorem sweetsies_remainder (m : ℕ) (h : number_of_sweetsies_in_one_bag m) : 
  (4 * m) % 8 = 4 := by
  -- Proof will be provided here.
  sorry

end sweetsies_remainder_l228_228659


namespace james_can_make_sushi_rolls_l228_228552

def fish_per_sushi_roll : Nat := 40
def total_fish_bought : Nat := 400
def percentage_bad_fish : Real := 0.20

theorem james_can_make_sushi_rolls : 
  (total_fish_bought - Nat.floor((percentage_bad_fish * total_fish_bought : Real))) / fish_per_sushi_roll = 8 := 
by
  sorry

end james_can_make_sushi_rolls_l228_228552


namespace revenue_for_recent_quarter_l228_228476

noncomputable def previous_year_revenue : ℝ := 85.0
noncomputable def percentage_fall : ℝ := 43.529411764705884
noncomputable def recent_quarter_revenue : ℝ := previous_year_revenue - (previous_year_revenue * (percentage_fall / 100))

theorem revenue_for_recent_quarter : recent_quarter_revenue = 48.0 := 
by 
  sorry -- Proof is skipped

end revenue_for_recent_quarter_l228_228476


namespace ellipse_hyperbola_tangent_l228_228593

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end ellipse_hyperbola_tangent_l228_228593


namespace inverse_function_domain_l228_228830

noncomputable def f (x : ℝ) : ℝ := -3 + Real.log (x - 1) / Real.log 2

theorem inverse_function_domain :
  ∀ x : ℝ, x ≥ 5 → ∃ y : ℝ, f x = y ∧ y ≥ -1 :=
by
  intro x hx
  use f x
  sorry

end inverse_function_domain_l228_228830


namespace xiaohong_money_l228_228448

def cost_kg_pears (x : ℝ) := x

def cost_kg_apples (x : ℝ) := x + 1.1

theorem xiaohong_money (x : ℝ) (hx : 6 * x - 3 = 5 * (x + 1.1) - 4) : 6 * x - 3 = 24 :=
by sorry

end xiaohong_money_l228_228448


namespace smallest_positive_real_is_131_div_11_l228_228900

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l228_228900


namespace train_passing_platform_time_l228_228637

theorem train_passing_platform_time :
  (500 : ℝ) / (50 : ℝ) > 0 →
  (500 : ℝ) + (500 : ℝ) / ((500 : ℝ) / (50 : ℝ)) = 100 := by
  sorry

end train_passing_platform_time_l228_228637


namespace additional_cost_tv_ad_l228_228018

theorem additional_cost_tv_ad (in_store_price : ℝ) (payment : ℝ) (shipping : ℝ) :
  in_store_price = 129.95 → payment = 29.99 → shipping = 14.95 → 
  (4 * payment + shipping - in_store_price) * 100 = 496 :=
by
  intros h1 h2 h3
  sorry

end additional_cost_tv_ad_l228_228018


namespace leak_drain_time_l228_228346

theorem leak_drain_time :
  ∀ (P L : ℝ),
  P = 1/6 →
  P - L = 1/12 →
  (1/L) = 12 :=
by
  intros P L hP hPL
  sorry

end leak_drain_time_l228_228346


namespace max_value_of_f_l228_228029

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5

theorem max_value_of_f : ∃ x ∈ (Set.Icc (-2:ℝ) 2), ∀ y ∈ (Set.Icc (-2:ℝ) 2), f y ≤ f x ∧ f x = 3 := by
  sorry

end max_value_of_f_l228_228029


namespace solve_for_x_l228_228987

theorem solve_for_x : ∀ x, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ↔ x = -7 / 2 := by
  sorry

end solve_for_x_l228_228987


namespace min_sum_l228_228850

namespace MinimumSum

theorem min_sum (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hc : 98 * m = n^3) : m + n = 42 :=
sorry

end MinimumSum

end min_sum_l228_228850


namespace one_third_of_recipe_l228_228201

noncomputable def recipe_flour_required : ℚ := 7 + 3 / 4

theorem one_third_of_recipe : (1 / 3) * recipe_flour_required = (2 : ℚ) + 7 / 12 :=
by
  sorry

end one_third_of_recipe_l228_228201


namespace ants_meeting_time_l228_228728

open Real

theorem ants_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 3 2 := sorry

end ants_meeting_time_l228_228728


namespace horse_tile_system_l228_228397

theorem horse_tile_system (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + (1 / 3 : ℚ) * y = 100) : 
  ∃ (x y : ℕ), (x + y = 100) ∧ (3 * x + (1 / 3 : ℚ) * y = 100) :=
by sorry

end horse_tile_system_l228_228397


namespace proper_subsets_B_l228_228794

theorem proper_subsets_B (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | x^2 + 2*x + 1 = 0})
  (hA_singleton : A = {a})
  (hB : B = {x | x^2 + a*x = 0}) :
  a = -1 ∧ 
  B = {0, 1} ∧
  (∀ S, S ∈ ({∅, {0}, {1}} : Set (Set ℝ)) ↔ S ⊂ B) :=
by
  -- Proof not provided, only statement required.
  sorry

end proper_subsets_B_l228_228794


namespace smallest_multiple_of_6_and_15_l228_228512

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l228_228512


namespace jessa_cupcakes_l228_228554

-- Define the number of classes and students
def fourth_grade_classes : ℕ := 3
def students_per_fourth_grade_class : ℕ := 30
def pe_classes : ℕ := 1
def students_per_pe_class : ℕ := 50

-- Calculate the total number of cupcakes needed
def total_cupcakes_needed : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) +
  (pe_classes * students_per_pe_class)

-- Statement to prove
theorem jessa_cupcakes : total_cupcakes_needed = 140 :=
by
  sorry

end jessa_cupcakes_l228_228554


namespace factor_expression_l228_228776

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end factor_expression_l228_228776


namespace find_a7_l228_228489

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l228_228489


namespace daniel_biked_more_l228_228854

def miles_biked_after_4_hours_more (speed_plain_daniel : ℕ) (speed_plain_elsa : ℕ) (time_plain : ℕ) 
(speed_hilly_daniel : ℕ) (speed_hilly_elsa : ℕ) (time_hilly : ℕ) : ℕ :=
(speed_plain_daniel * time_plain + speed_hilly_daniel * time_hilly) - 
(speed_plain_elsa * time_plain + speed_hilly_elsa * time_hilly)

theorem daniel_biked_more : miles_biked_after_4_hours_more 20 18 3 16 15 1 = 7 :=
by
  sorry

end daniel_biked_more_l228_228854


namespace quadratic_equation_l228_228070

theorem quadratic_equation (a b c x1 x2 : ℝ) (hx1 : a * x1^2 + b * x1 + c = 0) (hx2 : a * x2^2 + b * x2 + c = 0) :
  ∃ y : ℝ, c * y^2 + b * y + a = 0 := 
sorry

end quadratic_equation_l228_228070


namespace power_sum_evaluation_l228_228869

theorem power_sum_evaluation :
  (-1)^(4^3) + 2^(3^2) = 513 :=
by
  sorry

end power_sum_evaluation_l228_228869


namespace expected_value_is_7_l228_228195

def win (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * (10 - n) else 10 - n

def fair_die_values := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def expected_value (values : List ℕ) (win : ℕ → ℕ) : ℚ :=
  (values.map (λ n => win n)).sum / values.length

theorem expected_value_is_7 :
  expected_value fair_die_values win = 7 := 
sorry

end expected_value_is_7_l228_228195


namespace number_of_8_tuples_l228_228653

-- Define the constraints for a_k
def valid_a (a : ℕ) (k : ℕ) : Prop := 0 ≤ a ∧ a ≤ k

-- Define the condition for the 8-tuple
def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) : Prop :=
  valid_a a1 1 ∧ valid_a a2 2 ∧ valid_a a3 3 ∧ valid_a a4 4 ∧ 
  (a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19)

theorem number_of_8_tuples : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ), valid_8_tuple a1 a2 a3 a4 b1 b2 b3 b4 := 
sorry

end number_of_8_tuples_l228_228653


namespace count_natural_numbers_perfect_square_l228_228381

theorem count_natural_numbers_perfect_square :
  ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (n1^2 - 19 * n1 + 91) = m^2 ∧ (n2^2 - 19 * n2 + 91) = k^2 ∧
  ∀ n : ℕ, (n^2 - 19 * n + 91) = p^2 → n = n1 ∨ n = n2 := sorry

end count_natural_numbers_perfect_square_l228_228381


namespace neighbors_receive_equal_mangoes_l228_228835

-- Definitions from conditions
def total_mangoes : ℕ := 560
def mangoes_sold : ℕ := total_mangoes / 2
def remaining_mangoes : ℕ := total_mangoes - mangoes_sold
def neighbors : ℕ := 8

-- The lean statement
theorem neighbors_receive_equal_mangoes :
  remaining_mangoes / neighbors = 35 :=
by
  -- This is where the proof would go, but we'll leave it with sorry for now.
  sorry

end neighbors_receive_equal_mangoes_l228_228835


namespace smallest_enclosing_sphere_radius_l228_228072

noncomputable def radius_of_enclosing_sphere (r : ℝ) : ℝ :=
  let s := 6 -- side length of the cube
  let d := s * Real.sqrt 3 -- space diagonal of the cube
  (d + 2 * r) / 2

theorem smallest_enclosing_sphere_radius :
  radius_of_enclosing_sphere 2 = 3 * Real.sqrt 3 + 2 :=
by
  -- skipping the proof with sorry
  sorry

end smallest_enclosing_sphere_radius_l228_228072


namespace tan_add_pi_over_3_l228_228802

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by
  sorry

end tan_add_pi_over_3_l228_228802


namespace vacuum_tube_pins_and_holes_l228_228475

theorem vacuum_tube_pins_and_holes :
  ∀ (pins holes : Finset ℕ), 
  pins = {1, 2, 3, 4, 5, 6, 7} →
  holes = {1, 2, 3, 4, 5, 6, 7} →
  (∃ (a : ℕ), ∀ k ∈ pins, ∃ b ∈ holes, (2 * k) % 7 = b) := by
  sorry

end vacuum_tube_pins_and_holes_l228_228475


namespace width_of_crate_l228_228461

theorem width_of_crate
  (r : ℝ) (h : ℝ) (w : ℝ)
  (h_crate : h = 6 ∨ h = 10 ∨ w = 6 ∨ w = 10)
  (r_tank : r = 4)
  (height_longest_crate : h > w)
  (maximize_volume : ∃ d : ℝ, d = 2 * r ∧ w = d) :
  w = 8 := 
sorry

end width_of_crate_l228_228461


namespace units_digit_of_sum_of_squares_2010_odds_l228_228322

noncomputable def sum_units_digit_of_squares (n : ℕ) : ℕ :=
  let units_digits := [1, 9, 5, 9, 1]
  List.foldl (λ acc x => (acc + x) % 10) 0 (List.map (λ i => units_digits.get! (i % 5)) (List.range (2 * n)))

theorem units_digit_of_sum_of_squares_2010_odds : sum_units_digit_of_squares 2010 = 0 := sorry

end units_digit_of_sum_of_squares_2010_odds_l228_228322


namespace prime_problem_l228_228317

open Nat

-- Definition of primes and conditions based on the problem
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The formalized problem and conditions
theorem prime_problem (p q s : ℕ) 
  (p_prime : is_prime p) 
  (q_prime : is_prime q) 
  (s_prime : is_prime s) 
  (h1 : p + q = s + 4) 
  (h2 : 1 < p) 
  (h3 : p < q) : 
  p = 2 :=
sorry

end prime_problem_l228_228317


namespace bulb_probability_gt4000_l228_228034

-- Definitions given in conditions
def P_X : ℝ := 0.60
def P_Y : ℝ := 0.40
def P_gt4000_X : ℝ := 0.59
def P_gt4000_Y : ℝ := 0.65

-- The proof statement
theorem bulb_probability_gt4000 : 
  (P_X * P_gt4000_X + P_Y * P_gt4000_Y) = 0.614 :=
  by
  sorry

end bulb_probability_gt4000_l228_228034


namespace jump_difference_l228_228435

def frog_jump := 39
def grasshopper_jump := 17

theorem jump_difference :
  frog_jump - grasshopper_jump = 22 := by
  sorry

end jump_difference_l228_228435


namespace converse_inverse_contrapositive_l228_228303

theorem converse (x y : ℤ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by sorry

theorem inverse (x y : ℤ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by sorry

theorem contrapositive (x y : ℤ) : (¬ (x = 3 ∧ y = 2)) → (¬ (x + y = 5)) :=
by sorry

end converse_inverse_contrapositive_l228_228303


namespace danny_wrappers_more_than_soda_cans_l228_228767

theorem danny_wrappers_more_than_soda_cans :
  (67 - 22 = 45) := sorry

end danny_wrappers_more_than_soda_cans_l228_228767


namespace total_books_l228_228971

-- Given conditions
def susan_books : Nat := 600
def lidia_books : Nat := 4 * susan_books

-- The theorem to prove
theorem total_books : susan_books + lidia_books = 3000 :=
by
  unfold susan_books lidia_books
  sorry

end total_books_l228_228971


namespace double_series_evaluation_l228_228775

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, if h : n ≥ m then 1 / (m * n * (m + n + 2)) else 0) = (Real.pi ^ 2) / 6 := sorry

end double_series_evaluation_l228_228775


namespace no_n_in_range_l228_228660

theorem no_n_in_range :
  ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 10467 % 7 := by
  sorry

end no_n_in_range_l228_228660


namespace min_even_integers_zero_l228_228729

theorem min_even_integers_zero (x y a b m n : ℤ)
(h1 : x + y = 28) 
(h2 : x + y + a + b = 46) 
(h3 : x + y + a + b + m + n = 64) : 
∃ e, e = 0 :=
by {
  -- The conditions assure the sums of pairs are even including x, y, a, b, m, n.
  sorry
}

end min_even_integers_zero_l228_228729


namespace number_of_triangles_in_lattice_l228_228100

-- Define the triangular lattice structure
def triangular_lattice_rows : List ℕ := [1, 2, 3, 4]

-- Define the main theorem to state the number of triangles
theorem number_of_triangles_in_lattice :
  let number_of_triangles := 1 + 2 + 3 + 6 + 10
  number_of_triangles = 22 :=
by
  -- here goes the proof, which we skip with "sorry"
  sorry

end number_of_triangles_in_lattice_l228_228100


namespace sally_lost_orange_balloons_l228_228573

theorem sally_lost_orange_balloons :
  ∀ (initial_orange_balloons lost_orange_balloons current_orange_balloons : ℕ),
  initial_orange_balloons = 9 →
  current_orange_balloons = 7 →
  lost_orange_balloons = initial_orange_balloons - current_orange_balloons →
  lost_orange_balloons = 2 :=
by
  intros initial_orange_balloons lost_orange_balloons current_orange_balloons
  intros h_init h_current h_lost
  rw [h_init, h_current] at h_lost
  exact h_lost

end sally_lost_orange_balloons_l228_228573


namespace find_abc_sum_l228_228824

theorem find_abc_sum {U : Type} 
  (a b c : ℕ)
  (ha : a = 26)
  (hb : b = 1)
  (hc : c = 32)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  a + b + c = 59 :=
by
  sorry

end find_abc_sum_l228_228824


namespace cos_triangle_inequality_l228_228841

theorem cos_triangle_inequality (α β γ : ℝ) (h_sum : α + β + γ = Real.pi) 
    (h_α : 0 < α) (h_β : 0 < β) (h_γ : 0 < γ) (h_α_lt : α < Real.pi) (h_β_lt : β < Real.pi) (h_γ_lt : γ < Real.pi) : 
    (Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α) ≤ 3 / 4 :=
by
  sorry

end cos_triangle_inequality_l228_228841


namespace partI_solution_set_partII_range_of_m_l228_228151

def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

theorem partI_solution_set (x : ℝ) :
  ∀ (x : ℝ), f x 5 ≤ 12 ↔ (-13 / 2 ≤ x ∧ x ≤ 11 / 2) :=
by
  sorry

theorem partII_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end partI_solution_set_partII_range_of_m_l228_228151


namespace chris_dana_shared_rest_days_l228_228764

/-- Chris's and Dana's working schedules -/
structure work_schedule where
  work_days : ℕ
  rest_days : ℕ

/-- Define Chris's and Dana's schedules -/
def Chris_schedule : work_schedule := { work_days := 5, rest_days := 2 }
def Dana_schedule : work_schedule := { work_days := 6, rest_days := 1 }

/-- Number of days to consider -/
def total_days : ℕ := 1200

/-- Combinatorial function to calculate the number of coinciding rest-days -/
noncomputable def coinciding_rest_days (schedule1 schedule2 : work_schedule) (days : ℕ) : ℕ :=
  (days / (Nat.lcm (schedule1.work_days + schedule1.rest_days) (schedule2.work_days + schedule2.rest_days)))

/-- The proof problem statement -/
theorem chris_dana_shared_rest_days : 
coinciding_rest_days Chris_schedule Dana_schedule total_days = 171 :=
by sorry

end chris_dana_shared_rest_days_l228_228764


namespace ratio_population_X_to_Z_l228_228867

-- Given definitions
def population_of_Z : ℕ := sorry
def population_of_Y : ℕ := 2 * population_of_Z
def population_of_X : ℕ := 5 * population_of_Y

-- Theorem to prove
theorem ratio_population_X_to_Z : population_of_X / population_of_Z = 10 :=
by
  sorry

end ratio_population_X_to_Z_l228_228867


namespace find_d_l228_228250

variables {x y z k d : ℝ}
variables {a : ℝ} (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
variables (h_ap : x * (y - z) + y * (z - x) + z * (x - y) = 0)
variables (h_sum : x * (y - z) + (y * (z - x) + d) + (z * (x - y) + 2 * d) = k)

theorem find_d : d = k / 3 :=
sorry

end find_d_l228_228250


namespace inequality_solution_l228_228077

theorem inequality_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x ∈ Set.Ioo (-2 : ℝ) (-1) ∨ x ∈ Set.Ioi 2) ↔ 
  (∃ x : ℝ, (x^2 + x - 2) / (x + 2) ≥ (3 / (x - 2)) + (3 / 2)) := by
  sorry

end inequality_solution_l228_228077


namespace correct_equation_l228_228619

variable (x : ℤ)
variable (cost_of_chickens : ℤ)

-- Condition 1: If each person contributes 9 coins, there will be an excess of 11 coins.
def condition1 : Prop := 9 * x - cost_of_chickens = 11

-- Condition 2: If each person contributes 6 coins, there will be a shortage of 16 coins.
def condition2 : Prop := 6 * x - cost_of_chickens = -16

-- The goal is to prove the correct equation given the conditions.
theorem correct_equation (h1 : condition1 (x) (cost_of_chickens)) (h2 : condition2 (x) (cost_of_chickens)) :
  9 * x - 11 = 6 * x + 16 :=
sorry

end correct_equation_l228_228619


namespace perfect_square_is_289_l228_228603

/-- The teacher tells a three-digit perfect square number by
revealing the hundreds digit to person A, the tens digit to person B,
and the units digit to person C, and tells them that all three digits
are different from each other. Each person only knows their own digit and
not the others. The three people have the following conversation:

Person A: I don't know what the perfect square number is.  
Person B: You don't need to say; I also know that you don't know.  
Person C: I already know what the number is.  
Person A: After hearing Person C, I also know what the number is.  
Person B: After hearing Person A also knows what the number is.

Given these conditions, the three-digit perfect square number is 289. -/
theorem perfect_square_is_289:
  ∃ n : ℕ, n^2 = 289 := by
  sorry

end perfect_square_is_289_l228_228603


namespace find_added_number_l228_228527

theorem find_added_number (a : ℕ → ℝ) (x : ℝ) (h_init : a 1 = 2) (h_a3 : a 3 = 6)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  (h_geom : (a 4 + x)^2 = (a 1 + x) * (a 5 + x)) : 
  x = -11 := 
sorry

end find_added_number_l228_228527


namespace percentage_of_filled_seats_l228_228545

theorem percentage_of_filled_seats (total_seats vacant_seats : ℕ) (h_total : total_seats = 600) (h_vacant : vacant_seats = 240) :
  (total_seats - vacant_seats) * 100 / total_seats = 60 :=
by
  sorry

end percentage_of_filled_seats_l228_228545


namespace greatest_common_divisor_of_72_and_m_l228_228730

-- Definitions based on the conditions
def is_power_of_prime (m : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m = p^k

-- Main theorem based on the question and conditions
theorem greatest_common_divisor_of_72_and_m (m : ℕ) :
  (Nat.gcd 72 m = 9) ↔ (m = 3^2) ∨ ∃ k, k ≥ 2 ∧ m = 3^k :=
by
  sorry

end greatest_common_divisor_of_72_and_m_l228_228730


namespace compute_expression_l228_228209

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l228_228209


namespace sphere_radius_volume_eq_surface_area_l228_228805

theorem sphere_radius_volume_eq_surface_area (r : ℝ) (h₁ : (4 / 3) * π * r^3 = 4 * π * r^2) : r = 3 :=
by
  sorry

end sphere_radius_volume_eq_surface_area_l228_228805


namespace find_positive_integers_l228_228781

theorem find_positive_integers 
    (a b : ℕ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (h1 : ∃ k1 : ℤ, (a^3 * b - 1) = k1 * (a + 1))
    (h2 : ∃ k2 : ℤ, (b^3 * a + 1) = k2 * (b - 1)) : 
    (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
sorry

end find_positive_integers_l228_228781


namespace Fedya_age_statement_l228_228030

theorem Fedya_age_statement (d a : ℕ) (today : ℕ) (birthday : ℕ) 
    (H1 : d + 2 = a) 
    (H2 : a + 2 = birthday + 3) 
    (H3 : birthday = today + 1) :
    ∃ sameYear y, (birthday < today + 2 ∨ today < birthday) ∧ ((sameYear ∧ y - today = 1) ∨ (¬ sameYear ∧ y - today = 0)) :=
by
  sorry

end Fedya_age_statement_l228_228030


namespace parking_lot_capacity_l228_228812

-- Definitions based on the conditions
def levels : ℕ := 5
def parkedCars : ℕ := 23
def moreCars : ℕ := 62
def capacityPerLevel : ℕ := parkedCars + moreCars

-- Proof problem statement
theorem parking_lot_capacity : levels * capacityPerLevel = 425 := by
  -- Proof omitted
  sorry

end parking_lot_capacity_l228_228812


namespace find_circle_center_l228_228499

-- The statement to prove that the center of the given circle equation is (1, -2)
theorem find_circle_center : 
  ∃ (h k : ℝ), 3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0 → (h, k) = (1, -2) := 
by
  sorry

end find_circle_center_l228_228499


namespace solve_problem_l228_228965
open Complex

noncomputable def problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  (a ≠ -1) ∧ (b ≠ -1) ∧ (c ≠ -1) ∧ (d ≠ -1) ∧ (ω ^ 4 = 1) ∧ (ω ≠ 1) ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω ^ 2)
  
theorem solve_problem {a b c d : ℝ} {ω : ℂ} (h : problem a b c d ω) : 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2) :=
sorry

end solve_problem_l228_228965


namespace total_time_to_virgo_l228_228167

def train_ride : ℝ := 5
def first_layover : ℝ := 1.5
def bus_ride : ℝ := 4
def second_layover : ℝ := 0.5
def first_flight : ℝ := 6
def third_layover : ℝ := 2
def second_flight : ℝ := 3 * bus_ride
def fourth_layover : ℝ := 3
def car_drive : ℝ := 3.5
def first_boat_ride : ℝ := 1.5
def fifth_layover : ℝ := 0.75
def second_boat_ride : ℝ := 2 * first_boat_ride - 0.5
def final_walk : ℝ := 1.25

def total_time : ℝ := train_ride + first_layover + bus_ride + second_layover + first_flight + third_layover + second_flight + fourth_layover + car_drive + first_boat_ride + fifth_layover + second_boat_ride + final_walk

theorem total_time_to_virgo : total_time = 44 := by
  simp [train_ride, first_layover, bus_ride, second_layover, first_flight, third_layover, second_flight, fourth_layover, car_drive, first_boat_ride, fifth_layover, second_boat_ride, final_walk, total_time]
  sorry

end total_time_to_virgo_l228_228167


namespace smallest_percent_both_coffee_tea_l228_228134

noncomputable def smallest_percent_coffee_tea (P_C P_T P_not_C_or_T : ℝ) : ℝ :=
  let P_C_or_T := 1 - P_not_C_or_T
  let P_C_and_T := P_C + P_T - P_C_or_T
  P_C_and_T

theorem smallest_percent_both_coffee_tea :
  smallest_percent_coffee_tea 0.9 0.85 0.15 = 0.9 :=
by
  sorry

end smallest_percent_both_coffee_tea_l228_228134


namespace complete_square_l228_228330

theorem complete_square (x : ℝ) : (x ^ 2 + 4 * x + 1 = 0) ↔ ((x + 2) ^ 2 = 3) :=
by {
  split,
  { intro h,
    sorry },
  { intro h,
    sorry }
}

end complete_square_l228_228330


namespace workers_in_workshop_l228_228184

theorem workers_in_workshop :
  (∀ (W : ℕ), 8000 * W = 12000 * 7 + 6000 * (W - 7) → W = 21) :=
by
  intro W h
  sorry

end workers_in_workshop_l228_228184


namespace solution_to_problem_l228_228540

theorem solution_to_problem (x y : ℕ) (h : (2*x - 5) * (2*y - 5) = 25) : x + y = 10 ∨ x + y = 18 := by
  sorry

end solution_to_problem_l228_228540


namespace James_age_l228_228114

-- Defining variables
variables (James John Tim : ℕ)
variables (h1 : James + 12 = John)
variables (h2 : Tim + 5 = 2 * John)
variables (h3 : Tim = 79)

-- Statement to prove James' age
theorem James_age : James = 25 :=
by {
  sorry
}

end James_age_l228_228114


namespace quadratic_equation_identify_l228_228177

theorem quadratic_equation_identify {a b c x : ℝ} :
  ((3 - 5 * x^2 = x) ↔ true) ∧
  ((3 / x + x^2 - 1 = 0) ↔ false) ∧
  ((a * x^2 + b * x + c = 0) ↔ (a ≠ 0)) ∧
  ((4 * x - 1 = 0) ↔ false) :=
by
  sorry

end quadratic_equation_identify_l228_228177


namespace milan_total_bill_correct_l228_228914

-- Define the monthly fee, the per minute rate, and the number of minutes used last month
def monthly_fee : ℝ := 2
def per_minute_rate : ℝ := 0.12
def minutes_used : ℕ := 178

-- Define the total bill calculation
def total_bill : ℝ := minutes_used * per_minute_rate + monthly_fee

-- The proof statement
theorem milan_total_bill_correct :
  total_bill = 23.36 := 
by
  sorry

end milan_total_bill_correct_l228_228914


namespace product_12_3460_l228_228157

theorem product_12_3460 : 12 * 3460 = 41520 :=
by
  sorry

end product_12_3460_l228_228157


namespace sum_of_odd_subsets_l228_228562

open Finset

noncomputable def capacity (X : Finset ℕ) : ℕ :=
  X.sum id

noncomputable def even_subsets_capacity (n : ℕ) : ℕ :=
  (powerset (range (n + 1))).filter (λ s, capacity s % 2 = 0).sum capacity

noncomputable def odd_subsets_capacity (n : ℕ) : ℕ :=
  (powerset (range (n + 1))).filter (λ s, capacity s % 2 = 1).sum capacity

theorem sum_of_odd_subsets (n : ℕ) (h : n ≥ 3) :
  odd_subsets_capacity n = 2^(n-3) * n * (n + 1) := 
sorry

end sum_of_odd_subsets_l228_228562


namespace max_min_value_l228_228231

noncomputable def f (A B x a b : ℝ) : ℝ :=
  A * Real.sqrt (x - a) + B * Real.sqrt (b - x)

theorem max_min_value (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha_lt_b : a < b) :
  (∀ x, a ≤ x ∧ x ≤ b → f A B x a b ≤ Real.sqrt ((A^2 + B^2) * (b - a))) ∧
  min (f A B a a b) (f A B b a b) ≤ f A B x a b :=
  sorry

end max_min_value_l228_228231


namespace find_brick_length_l228_228020

-- Conditions as given in the problem.
def wall_length : ℝ := 8
def wall_width : ℝ := 6
def wall_height : ℝ := 22.5
def number_of_bricks : ℕ := 6400
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- The volume of the wall in cubic centimeters.
def wall_volume_cm_cube : ℝ := (wall_length * 100) * (wall_width * 100) * (wall_height * 100)

-- Define the volume of one brick based on the unknown length L.
def brick_volume (L : ℝ) : ℝ := L * brick_width * brick_height

-- Define an equivalence for the total volume of the bricks to the volume of the wall.
theorem find_brick_length : 
  ∃ (L : ℝ), wall_volume_cm_cube = brick_volume L * number_of_bricks ∧ L = 2500 := 
by
  sorry

end find_brick_length_l228_228020


namespace stella_annual_income_l228_228429

-- Define the conditions
def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def total_months : ℕ := 12

-- The question: What is Stella's annual income last year?
def annual_income (monthly_income : ℕ) (worked_months : ℕ) : ℕ :=
  monthly_income * worked_months

-- Prove that Stella's annual income last year was $49190
theorem stella_annual_income : annual_income monthly_income (total_months - unpaid_leave_months) = 49190 :=
by
  sorry

end stella_annual_income_l228_228429


namespace calculate_flat_rate_shipping_l228_228354

noncomputable def flat_rate_shipping : ℝ :=
  17.00

theorem calculate_flat_rate_shipping
  (price_per_shirt : ℝ)
  (num_shirts : ℤ)
  (price_pack_socks : ℝ)
  (num_packs_socks : ℤ)
  (price_per_short : ℝ)
  (num_shorts : ℤ)
  (price_swim_trunks : ℝ)
  (num_swim_trunks : ℤ)
  (total_bill : ℝ)
  (total_items_cost : ℝ)
  (shipping_cost : ℝ) :
  price_per_shirt * num_shirts + 
  price_pack_socks * num_packs_socks + 
  price_per_short * num_shorts +
  price_swim_trunks * num_swim_trunks = total_items_cost →
  total_bill - total_items_cost = shipping_cost →
  total_items_cost > 50 → 
  0.20 * total_items_cost ≠ shipping_cost →
  flat_rate_shipping = 17.00 := 
sorry

end calculate_flat_rate_shipping_l228_228354


namespace range_of_a_minus_b_l228_228308

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem range_of_a_minus_b (a b : ℝ) (h1 : ∃ α β : ℝ, α ≠ β ∧ f α a b = 0 ∧ f β a b = 0)
  (h2 : ∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
                         (x2 - x1 = x3 - x2) ∧ (x3 - x2 = x4 - x3) ∧
                         f (x1^2 + 2 * x1 - 1) a b = 0 ∧
                         f (x2^2 + 2 * x2 - 1) a b = 0 ∧
                         f (x3^2 + 2 * x3 - 1) a b = 0 ∧
                         f (x4^2 + 2 * x4 - 1) a b = 0) :
  a - b ≤ 25 / 9 :=
sorry

end range_of_a_minus_b_l228_228308


namespace f1_neither_even_nor_odd_f2_min_value_l228_228453

noncomputable def f1 (x : ℝ) : ℝ :=
  x^2 + abs (x - 2) - 1

theorem f1_neither_even_nor_odd : ¬(∀ x : ℝ, f1 x = f1 (-x)) ∧ ¬(∀ x : ℝ, f1 x = -f1 (-x)) :=
sorry

noncomputable def f2 (x a : ℝ) : ℝ :=
  x^2 + abs (x - a) + 1

theorem f2_min_value (a : ℝ) :
  (if a < -1/2 then (∃ x, f2 x a = 3/4 - a)
  else if -1/2 ≤ a ∧ a ≤ 1/2 then (∃ x, f2 x a = a^2 + 1)
  else (∃ x, f2 x a = 3/4 + a)) :=
sorry

end f1_neither_even_nor_odd_f2_min_value_l228_228453


namespace gumballs_remaining_l228_228054

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end gumballs_remaining_l228_228054


namespace first_term_arithmetic_sum_l228_228272

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l228_228272


namespace sasha_quarters_l228_228845

theorem sasha_quarters (h₁ : 2.10 = 0.35 * q) : q = 6 := 
sorry

end sasha_quarters_l228_228845


namespace perimeter_of_resulting_figure_l228_228202

def side_length := 100
def original_square_perimeter := 4 * side_length
def rectangle_width := side_length
def rectangle_height := side_length / 2
def number_of_longer_sides_of_rectangles_touching := 4

theorem perimeter_of_resulting_figure :
  let new_perimeter := 3 * side_length + number_of_longer_sides_of_rectangles_touching * rectangle_height
  new_perimeter = 500 :=
by
  sorry

end perimeter_of_resulting_figure_l228_228202


namespace trapezoid_area_l228_228004

theorem trapezoid_area (AD BC : ℝ) (AD_eq : AD = 18) (BC_eq : BC = 2) (CD : ℝ) (h : CD = 10): 
  ∃ (CH : ℝ), CH = 6 ∧ (1 / 2) * (AD + BC) * CH = 60 :=
by
  sorry

end trapezoid_area_l228_228004


namespace fraction_boxes_loaded_by_day_crew_l228_228866

variables {D W_d : ℝ}

theorem fraction_boxes_loaded_by_day_crew
  (h1 : ∀ (D W_d: ℝ), D > 0 → W_d > 0 → ∃ (D' W_n : ℝ), (D' = 0.5 * D) ∧ (W_n = 0.8 * W_d))
  (h2 : ∃ (D W_d : ℝ), ∀ (D' W_n : ℝ), (D' = 0.5 * D) → (W_n = 0.8 * W_d) → 
        (D * W_d / (D * W_d + D' * W_n)) = (5 / 7)) :
  (∃ (D W_d : ℝ), D > 0 → W_d > 0 → (D * W_d)/(D * W_d + 0.5 * D * 0.8 * W_d) = (5/7)) := 
  sorry 

end fraction_boxes_loaded_by_day_crew_l228_228866


namespace amanda_average_speed_l228_228351

def amanda_distance1 : ℝ := 450
def amanda_time1 : ℝ := 7.5
def amanda_distance2 : ℝ := 420
def amanda_time2 : ℝ := 7

def total_distance : ℝ := amanda_distance1 + amanda_distance2
def total_time : ℝ := amanda_time1 + amanda_time2
def expected_average_speed : ℝ := 60

theorem amanda_average_speed :
  (total_distance / total_time) = expected_average_speed := by
  sorry

end amanda_average_speed_l228_228351


namespace abs_less_than_2_sufficient_but_not_necessary_l228_228432

theorem abs_less_than_2_sufficient_but_not_necessary (x : ℝ) : 
  (|x| < 2 → (x^2 - x - 6 < 0)) ∧ ¬(x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_less_than_2_sufficient_but_not_necessary_l228_228432


namespace crayons_ratio_l228_228557

theorem crayons_ratio (K B G J : ℕ) 
  (h1 : K = 2 * B)
  (h2 : B = 2 * G)
  (h3 : G = J)
  (h4 : K = 128)
  (h5 : J = 8) : 
  G / J = 4 :=
by
  sorry

end crayons_ratio_l228_228557


namespace boys_number_l228_228190

variable (M W B : ℕ)

-- Conditions
axiom h1 : M = W
axiom h2 : W = B
axiom h3 : M * 8 = 120

theorem boys_number :
  B = 15 := by
  sorry

end boys_number_l228_228190


namespace velocity_at_t1_l228_228340

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2 * t

-- Define the velocity function as the derivative of s
def velocity (t : ℝ) : ℝ := -2 * t + 2

-- Prove that the velocity at t = 1 is 0
theorem velocity_at_t1 : velocity 1 = 0 :=
by
  -- Apply the definition of velocity
    sorry

end velocity_at_t1_l228_228340


namespace Iris_total_spent_l228_228701

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end Iris_total_spent_l228_228701


namespace find_a_l228_228383

theorem find_a (a : ℝ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := sorry

end find_a_l228_228383


namespace solve_equation_l228_228582

theorem solve_equation (x : ℝ) : 2 * (x - 2)^2 = 6 - 3 * x ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_equation_l228_228582


namespace points_on_circle_l228_228364

theorem points_on_circle (t : ℝ) : (∃ (x y : ℝ), x = Real.cos (2 * t) ∧ y = Real.sin (2 * t) ∧ (x^2 + y^2 = 1)) := by
  sorry

end points_on_circle_l228_228364


namespace express_in_scientific_notation_l228_228836

-- Definitions based on problem conditions
def GDP_first_quarter : ℝ := 27017800000000

-- Main theorem statement that needs to be proved
theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), (GDP_first_quarter = a * 10 ^ b) ∧ (a = 2.70178) ∧ (b = 13) :=
by
  sorry -- Placeholder to indicate the proof is omitted

end express_in_scientific_notation_l228_228836


namespace multiply_72517_9999_l228_228333

theorem multiply_72517_9999 : 72517 * 9999 = 725097483 :=
by
  sorry

end multiply_72517_9999_l228_228333


namespace g_at_3_l228_228719

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) - x * g (3 ^ (-x)) = x) : g 3 = 0 :=
by
  sorry

end g_at_3_l228_228719


namespace friend_owns_10_bikes_l228_228389

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end friend_owns_10_bikes_l228_228389


namespace general_term_defines_sequence_l228_228935

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end general_term_defines_sequence_l228_228935


namespace rope_length_after_knots_l228_228444

def num_ropes : ℕ := 64
def length_per_rope : ℕ := 25
def length_reduction_per_knot : ℕ := 3
def num_knots : ℕ := num_ropes - 1
def initial_total_length : ℕ := num_ropes * length_per_rope
def total_reduction : ℕ := num_knots * length_reduction_per_knot
def final_rope_length : ℕ := initial_total_length - total_reduction

theorem rope_length_after_knots :
  final_rope_length = 1411 := by
  sorry

end rope_length_after_knots_l228_228444


namespace geometric_arithmetic_sum_l228_228091

open Real

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def arithmetic_mean (x y a b c : ℝ) : Prop :=
  2 * x = a + b ∧ 2 * y = b + c

theorem geometric_arithmetic_sum
  (a b c x y : ℝ)
  (habc : geometric_sequence a b c)
  (hxy : arithmetic_mean x y a b c)
  (hx_ne_zero : x ≠ 0)
  (hy_ne_zero : y ≠ 0) :
  (a / x) + (c / y) = 2 := 
by {
  sorry -- Proof omitted as per the prompt
}

end geometric_arithmetic_sum_l228_228091


namespace find_z_l228_228522

noncomputable def solve_for_z (i : ℂ) (z : ℂ) :=
  (2 - i) * z = i ^ 2021

theorem find_z (i z : ℂ) (h1 : solve_for_z i z) : 
  z = -1/5 + 2/5 * i := 
by 
  sorry

end find_z_l228_228522


namespace cube_and_reciprocal_l228_228537

theorem cube_and_reciprocal (m : ℝ) (hm : m + 1/m = 10) : m^3 + 1/m^3 = 970 := 
by
  sorry

end cube_and_reciprocal_l228_228537


namespace smallest_value_l228_228616

theorem smallest_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (v : ℝ), (∀ x y : ℝ, 0 < x → 0 < y → v ≤ (16 / x + 108 / y + x * y)) ∧ v = 36 :=
sorry

end smallest_value_l228_228616


namespace integral_transform_eq_l228_228738

open MeasureTheory

variable (f : ℝ → ℝ)

theorem integral_transform_eq (hf_cont : Continuous f) (hL_exists : ∃ L, ∫ x in (Set.univ : Set ℝ), f x = L) :
  ∃ L, ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = L :=
by
  cases' hL_exists with L hL
  use L
  have h_transform : ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = ∫ x in (Set.univ : Set ℝ), f x := sorry
  rw [h_transform]
  exact hL

end integral_transform_eq_l228_228738


namespace cotangent_identity_l228_228288

noncomputable def cotangent (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cotangent_identity (x : ℝ) (i : ℂ) (n : ℕ) (k : ℕ) (h : (0 < k) ∧ (k < n)) :
  ((x + i) / (x - i))^n = 1 → x = cotangent (k * Real.pi / n) := 
sorry

end cotangent_identity_l228_228288


namespace find_k_value_l228_228933

-- Define the lines l1 and l2 with given conditions
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the condition for the quadrilateral to be circumscribed by a circle
def is_circumscribed (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ line2 k x y ∧ 0 < x ∧ 0 < y

theorem find_k_value (k : ℝ) : is_circumscribed k → k = 3 := 
sorry

end find_k_value_l228_228933


namespace initial_percentage_of_gold_l228_228058

theorem initial_percentage_of_gold (x : ℝ) (h₁ : 48 * x / 100 + 12 = 40 * 60 / 100) : x = 25 :=
by
  sorry

end initial_percentage_of_gold_l228_228058


namespace marbles_shared_equally_l228_228604

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end marbles_shared_equally_l228_228604


namespace certain_number_is_two_l228_228181

variable (x : ℕ)  -- x is the certain number

-- Condition: Given that adding 6 incorrectly results in 8
axiom h1 : x + 6 = 8

-- The mathematically equivalent proof problem Lean statement
theorem certain_number_is_two : x = 2 :=
by
  sorry

end certain_number_is_two_l228_228181


namespace num_bikes_l228_228544

variable (C B : ℕ)

-- The given conditions
def num_cars : ℕ := 10
def num_wheels_total : ℕ := 44
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- The mathematical proof problem statement
theorem num_bikes :
  C = num_cars →
  B = ((num_wheels_total - (C * wheels_per_car)) / wheels_per_bike) →
  B = 2 :=
by
  intros hC hB
  rw [hC] at hB
  sorry

end num_bikes_l228_228544


namespace james_correct_take_home_pay_l228_228261

noncomputable def james_take_home_pay : ℝ :=
  let main_job_hourly_rate := 20
  let second_job_hourly_rate := main_job_hourly_rate * 0.8
  let main_job_hours := 30
  let main_job_overtime_hours := 5
  let second_job_hours := 15
  let side_gig_daily_rate := 100
  let side_gig_days := 2
  let tax_deductions := 200
  let federal_tax_rate := 0.18
  let state_tax_rate := 0.05

  let regular_main_job_hours := main_job_hours - main_job_overtime_hours
  let main_job_regular_pay := regular_main_job_hours * main_job_hourly_rate
  let main_job_overtime_pay := main_job_overtime_hours * main_job_hourly_rate * 1.5
  let total_main_job_pay := main_job_regular_pay + main_job_overtime_pay

  let total_second_job_pay := second_job_hours * second_job_hourly_rate
  let total_side_gig_pay := side_gig_daily_rate * side_gig_days

  let total_earnings := total_main_job_pay + total_second_job_pay + total_side_gig_pay
  let taxable_income := total_earnings - tax_deductions
  let federal_tax := taxable_income * federal_tax_rate
  let state_tax := taxable_income * state_tax_rate
  let total_taxes := federal_tax + state_tax
  total_earnings - total_taxes

theorem james_correct_take_home_pay : james_take_home_pay = 885.30 := by
  sorry

end james_correct_take_home_pay_l228_228261


namespace kittens_price_l228_228877

theorem kittens_price (x : ℕ) 
  (h1 : 2 * x + 5 = 17) : x = 6 := by
  sorry

end kittens_price_l228_228877


namespace first_term_arithmetic_sequence_l228_228279

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l228_228279


namespace sport_vs_std_ratio_comparison_l228_228698

/-- Define the ratios for the standard formulation. -/
def std_flavor_syrup_ratio := 1 / 12
def std_flavor_water_ratio := 1 / 30

/-- Define the conditions for the sport formulation. -/
def sport_water := 15 -- ounces of water in the sport formulation
def sport_syrup := 1 -- ounce of corn syrup in the sport formulation

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation. -/
def sport_flavor_water_ratio := std_flavor_water_ratio / 2

/-- Calculate the amount of flavoring in the sport formulation. -/
def sport_flavor := sport_water * sport_flavor_water_ratio

/-- The ratio of flavoring to corn syrup in the sport formulation. -/
def sport_flavor_syrup_ratio := sport_flavor / sport_syrup

/-- The proof problem statement. -/
theorem sport_vs_std_ratio_comparison : sport_flavor_syrup_ratio = 3 * std_flavor_syrup_ratio := 
by
  -- proof would go here
  sorry

end sport_vs_std_ratio_comparison_l228_228698


namespace no_integer_solution_l228_228988

theorem no_integer_solution : ¬ ∃ (x y : ℤ), x^2 - 7 * y = 10 :=
by
  sorry

end no_integer_solution_l228_228988


namespace megan_folders_count_l228_228132

theorem megan_folders_count (init_files deleted_files files_per_folder : ℕ) (h₁ : init_files = 93) (h₂ : deleted_files = 21) (h₃ : files_per_folder = 8) :
  (init_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end megan_folders_count_l228_228132


namespace trey_will_sell_bracelets_for_days_l228_228168

def cost : ℕ := 112
def price_per_bracelet : ℕ := 1
def bracelets_per_day : ℕ := 8

theorem trey_will_sell_bracelets_for_days :
  ∃ d : ℕ, d = cost / (price_per_bracelet * bracelets_per_day) ∧ d = 14 := by
  sorry

end trey_will_sell_bracelets_for_days_l228_228168


namespace sin_double_angle_plus_pi_over_2_l228_228371

theorem sin_double_angle_plus_pi_over_2 (θ : ℝ) (h : Real.cos θ = -1/3) :
  Real.sin (2 * θ + Real.pi / 2) = -7/9 :=
sorry

end sin_double_angle_plus_pi_over_2_l228_228371


namespace number_of_multiples_of_4_l228_228646

theorem number_of_multiples_of_4 (a b : ℤ) (h1 : 100 < a) (h2 : b < 500) (h3 : a % 4 = 0) (h4 : b % 4 = 0) : 
  ∃ n : ℤ, n = 99 :=
by
  sorry

end number_of_multiples_of_4_l228_228646


namespace shelby_initial_money_l228_228292

-- Definitions based on conditions
def cost_of_first_book : ℕ := 8
def cost_of_second_book : ℕ := 4
def cost_of_each_poster : ℕ := 4
def number_of_posters : ℕ := 2

-- Number to prove (initial money)
def initial_money : ℕ := 20

-- Theorem statement
theorem shelby_initial_money :
    (cost_of_first_book + cost_of_second_book + (number_of_posters * cost_of_each_poster)) = initial_money := by
    sorry

end shelby_initial_money_l228_228292


namespace blue_faces_cube_l228_228757

theorem blue_faces_cube (n : ℕ) (h1 : n > 0) (h2 : (6 * n^2) = 1 / 3 * 6 * n^3) : n = 3 :=
by
  -- we only need the statement for now; the proof is omitted.
  sorry

end blue_faces_cube_l228_228757


namespace tulip_count_l228_228783

theorem tulip_count (total_flowers : ℕ) (daisies : ℕ) (roses_ratio : ℚ)
  (tulip_count : ℕ) :
  total_flowers = 102 →
  daisies = 6 →
  roses_ratio = 5 / 6 →
  tulip_count = (total_flowers - daisies) * (1 - roses_ratio) →
  tulip_count = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end tulip_count_l228_228783


namespace state_tax_percentage_l228_228641

theorem state_tax_percentage (weekly_salary federal_percent health_insurance life_insurance parking_fee final_paycheck : ℝ)
  (h_weekly_salary : weekly_salary = 450)
  (h_federal_percent : federal_percent = 1/3)
  (h_health_insurance : health_insurance = 50)
  (h_life_insurance : life_insurance = 20)
  (h_parking_fee : parking_fee = 10)
  (h_final_paycheck : final_paycheck = 184) :
  (36 / 450) * 100 = 8 :=
by
  sorry

end state_tax_percentage_l228_228641


namespace range_of_b_l228_228370

variable (a b c : ℝ)

theorem range_of_b (h1 : a + b + c = 9) (h2 : a * b + b * c + c * a = 24) : 1 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l228_228370


namespace expression_value_l228_228496

noncomputable def evaluate_expression : ℝ :=
  Real.logb 2 (3 * 11 + Real.exp (4 - 8)) + 3 * Real.sin (Real.pi^2 - Real.sqrt ((6 * 4) / 3 - 4))

theorem expression_value : evaluate_expression = 3.832 := by
  sorry

end expression_value_l228_228496


namespace good_numbers_l228_228750

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ (n = 1 ∨ (nat.prime n ∧ n % 2 = 1)) :=
by
  sorry

end good_numbers_l228_228750


namespace polynomial_difference_of_squares_l228_228352

theorem polynomial_difference_of_squares:
  (∀ a b : ℝ, ¬ ∃ x1 x2 : ℝ, a^2 + (-b)^2 = (x1 - x2) * (x1 + x2)) ∧
  (∀ m n : ℝ, ¬ ∃ x1 x2 : ℝ, 5 * m^2 - 20 * m * n = (x1 - x2) * (x1 + x2)) ∧
  (∀ x y : ℝ, ¬ ∃ x1 x2 : ℝ, -x^2 - y^2 = (x1 - x2) * (x1 + x2)) →
  ∃ x1 x2 : ℝ, -x^2 + 9 = (x1 - x2) * (x1 + x2) :=
by 
  sorry

end polynomial_difference_of_squares_l228_228352


namespace line_through_fixed_point_l228_228664

theorem line_through_fixed_point (a : ℝ) :
  ∃ P : ℝ × ℝ, (P = (1, 2)) ∧ (∀ x y, a * x + y - a - 2 = 0 → P = (x, y)) ∧
  ((∃ a, x + y = a ∧ x = 1 ∧ y = 2) → (a = 3)) :=
by
  sorry

end line_through_fixed_point_l228_228664


namespace part_a_part_b_l228_228974

-- Define the cost variables for chocolates, popsicles, and lollipops
variables (C P L : ℕ)

-- Given conditions
axiom cost_relation1 : 3 * C = 2 * P
axiom cost_relation2 : 2 * L = 5 * C

-- Part (a): Prove that Mário can buy 5 popsicles with the money for 3 lollipops
theorem part_a : 
  (3 * L) / P = 5 :=
by sorry

-- Part (b): Prove that Mário can buy 11 chocolates with the money for 3 chocolates, 2 popsicles, and 2 lollipops combined
theorem part_b : 
  (3 * C + 2 * P + 2 * L) / C = 11 :=
by sorry

end part_a_part_b_l228_228974


namespace dodecahedron_probability_l228_228318

noncomputable def probability_endpoints_of_edge (total_vertices edges_per_vertex total_edges : ℕ) : ℚ :=
  let possible_choices := total_vertices - 1
  let favorable_outcomes := edges_per_vertex
  favorable_outcomes / possible_choices

theorem dodecahedron_probability :
  probability_endpoints_of_edge 20 3 30 = 3 / 19 := by
  sorry

end dodecahedron_probability_l228_228318


namespace beavers_build_dam_l228_228302

def num_beavers_first_group : ℕ := 20

theorem beavers_build_dam (B : ℕ) (t₁ : ℕ) (t₂ : ℕ) (n₂ : ℕ) :
  (B * t₁ = n₂ * t₂) → (B = num_beavers_first_group) := 
by
  -- Given
  let t₁ := 3
  let t₂ := 5
  let n₂ := 12

  -- Work equation
  assume h : B * t₁ = n₂ * t₂
  
  -- Correct answer
  have B_def : B = (n₂ * t₂) / t₁,
  exact h
   
  sorry

end beavers_build_dam_l228_228302


namespace illuminated_cube_surface_area_l228_228629

noncomputable def edge_length : ℝ := Real.sqrt (2 + Real.sqrt 3)
noncomputable def radius : ℝ := Real.sqrt 2
noncomputable def illuminated_area (a ρ : ℝ) : ℝ := Real.sqrt 3 * (Real.pi + 3)

theorem illuminated_cube_surface_area :
  illuminated_area edge_length radius = Real.sqrt 3 * (Real.pi + 3) := sorry

end illuminated_cube_surface_area_l228_228629


namespace isabel_spending_ratio_l228_228258

theorem isabel_spending_ratio :
  ∀ (initial_amount toy_cost remaining_amount : ℝ),
    initial_amount = 204 ∧
    toy_cost = initial_amount / 2 ∧
    remaining_amount = 51 →
    ((initial_amount - toy_cost - remaining_amount) / remaining_amount) = 1 / 2 :=
by
  intros
  sorry

end isabel_spending_ratio_l228_228258


namespace beaver_group_l228_228299

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end beaver_group_l228_228299


namespace december_19th_day_l228_228890

theorem december_19th_day (december_has_31_days : true)
  (december_1st_is_monday : true)
  (day_of_week : ℕ → ℕ) :
  day_of_week 19 = 5 :=
sorry

end december_19th_day_l228_228890


namespace sale_coupon_discount_l228_228049

theorem sale_coupon_discount
  (original_price : ℝ)
  (sale_price : ℝ)
  (price_after_coupon : ℝ)
  (h1 : sale_price = 0.5 * original_price)
  (h2 : price_after_coupon = 0.8 * sale_price) :
  (original_price - price_after_coupon) / original_price * 100 = 60 := by
sorry

end sale_coupon_discount_l228_228049


namespace overlap_coordinates_l228_228628

theorem overlap_coordinates :
  ∃ m n : ℝ, 
    (m + n = 6.8) ∧ 
    ((2 * (7 + m) / 2 - 3) = (3 + n) / 2) ∧ 
    ((2 * (7 + m) / 2 - 3) = - (m - 7) / 2) :=
by
  sorry

end overlap_coordinates_l228_228628


namespace largest_positive_integer_not_sum_of_36_and_composite_l228_228170

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l228_228170


namespace janet_family_needs_91_tickets_l228_228116

def janet_family_tickets (adults: ℕ) (children: ℕ) (roller_coaster_adult_tickets: ℕ) (roller_coaster_child_tickets: ℕ) 
  (giant_slide_adult_tickets: ℕ) (giant_slide_child_tickets: ℕ) (num_roller_coaster_rides_adult: ℕ) 
  (num_roller_coaster_rides_child: ℕ) (num_giant_slide_rides_adult: ℕ) (num_giant_slide_rides_child: ℕ) : ℕ := 
  (adults * roller_coaster_adult_tickets * num_roller_coaster_rides_adult) + 
  (children * roller_coaster_child_tickets * num_roller_coaster_rides_child) + 
  (1 * giant_slide_adult_tickets * num_giant_slide_rides_adult) + 
  (1 * giant_slide_child_tickets * num_giant_slide_rides_child)

theorem janet_family_needs_91_tickets :
  janet_family_tickets 2 2 7 5 4 3 3 2 5 3 = 91 := 
by 
  -- Calculations based on the given conditions (skipped in this statement)
  sorry

end janet_family_needs_91_tickets_l228_228116


namespace simplify_fraction_l228_228425

variable (k : ℤ)

theorem simplify_fraction (a b : ℤ)
  (hk : a = 2)
  (hb : b = 4) :
  (6 * k + 12) / 3 = 2 * k + 4 ∧ (a : ℚ) / (b : ℚ) = 1 / 2 := 
by
  sorry

end simplify_fraction_l228_228425


namespace factor_difference_of_squares_l228_228073

theorem factor_difference_of_squares (t : ℤ) : t^2 - 64 = (t - 8) * (t + 8) :=
by {
  sorry
}

end factor_difference_of_squares_l228_228073


namespace total_chips_eaten_l228_228960

theorem total_chips_eaten (dinner_chips after_dinner_chips : ℕ) (h1 : dinner_chips = 1) (h2 : after_dinner_chips = 2 * dinner_chips) : dinner_chips + after_dinner_chips = 3 := by
  sorry

end total_chips_eaten_l228_228960


namespace find_k_value_l228_228082

def line (k : ℝ) (x y : ℝ) : Prop := 3 - 2 * k * x = -4 * y

def on_line (k : ℝ) : Prop := line k 5 (-2)

theorem find_k_value (k : ℝ) : on_line k → k = -0.5 :=
by
  sorry

end find_k_value_l228_228082


namespace time_for_A_to_complete_work_l228_228746

theorem time_for_A_to_complete_work (W : ℝ) (A B C : ℝ) (W_pos : 0 < W) (B_work : B = W / 40) (C_work : C = W / 20) : 
  (10 * (W / A) + 10 * (W / B) + 10 * (W / C) = W) → A = W / 40 :=
by 
  sorry

end time_for_A_to_complete_work_l228_228746


namespace committee_formation_l228_228395

theorem committee_formation :
  let club_size := 15
  let num_roles := 2
  let num_members := 3
  let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
  total_ways = 60060 := by
    let club_size := 15
    let num_roles := 2
    let num_members := 3
    let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
    show total_ways = 60060
    sorry

end committee_formation_l228_228395


namespace final_point_P_after_transformations_l228_228980

noncomputable def point := (ℝ × ℝ)

def rotate_90_clockwise (p : point) : point :=
  (-p.2, p.1)

def reflect_across_x (p : point) : point :=
  (p.1, -p.2)

def P : point := (3, -5)

def Q : point := (5, -2)

def R : point := (5, -5)

theorem final_point_P_after_transformations : reflect_across_x (rotate_90_clockwise P) = (-5, 3) :=
by 
  sorry

end final_point_P_after_transformations_l228_228980


namespace number_of_parallelograms_l228_228027

-- Problem's condition
def side_length (n : ℕ) : Prop := n > 0

-- Required binomial coefficient (combination formula)
def binom (n k : ℕ) : ℕ := n.choose k

-- Total number of parallelograms in the tiling
theorem number_of_parallelograms (n : ℕ) (h : side_length n) : 
  3 * binom (n + 2) 4 = 3 * (n+2).choose 4 :=
by
  sorry

end number_of_parallelograms_l228_228027


namespace probability_first_spade_second_ace_l228_228862

theorem probability_first_spade_second_ace :
  let n : ℕ := 52
  let spades : ℕ := 13
  let aces : ℕ := 4
  let ace_of_spades : ℕ := 1
  let non_ace_spades : ℕ := spades - ace_of_spades
  (non_ace_spades / n : ℚ) * (aces / (n - 1) : ℚ) +
  (ace_of_spades / n : ℚ) * ((aces - 1) / (n - 1) : ℚ) =
  (1 / n : ℚ) :=
by {
  -- proof goes here
  sorry
}

end probability_first_spade_second_ace_l228_228862


namespace mutually_exclusive_not_necessarily_complementary_l228_228413

-- Define what it means for events to be mutually exclusive
def mutually_exclusive (E1 E2 : Prop) : Prop :=
  ¬ (E1 ∧ E2)

-- Define what it means for events to be complementary
def complementary (E1 E2 : Prop) : Prop :=
  (E1 ∨ E2) ∧ ¬ (E1 ∧ E2) ∧ (¬ E1 ∨ ¬ E2)

theorem mutually_exclusive_not_necessarily_complementary :
  ∀ E1 E2 : Prop, mutually_exclusive E1 E2 → ¬ complementary E1 E2 :=
sorry

end mutually_exclusive_not_necessarily_complementary_l228_228413


namespace doritos_ratio_l228_228566

noncomputable def bags_of_chips : ℕ := 80
noncomputable def bags_per_pile : ℕ := 5
noncomputable def piles : ℕ := 4

theorem doritos_ratio (D T : ℕ) (h1 : T = bags_of_chips)
  (h2 : D = piles * bags_per_pile) :
  (D : ℚ) / T = 1 / 4 := by
  sorry

end doritos_ratio_l228_228566


namespace points_difference_l228_228642

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end points_difference_l228_228642


namespace find_greater_number_l228_228998

theorem find_greater_number (a b : ℕ) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 :=
sorry

end find_greater_number_l228_228998


namespace area_difference_is_correct_l228_228357

noncomputable def area_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

noncomputable def area_circle (diameter : ℝ) : ℝ := (Real.pi * (diameter / 2) ^ 2)

noncomputable def combined_area_difference : ℝ :=
  (area_rectangle 11 11 + area_rectangle 5.5 11) - 
  (area_equilateral_triangle 6 + area_circle 4)
 
theorem area_difference_is_correct :
  |combined_area_difference - 153.35| < 0.001 :=
by
  sorry

end area_difference_is_correct_l228_228357


namespace total_distance_covered_l228_228633

theorem total_distance_covered :
  let speed_upstream := 12 -- km/h
  let time_upstream := 2 -- hours
  let speed_downstream := 38 -- km/h
  let time_downstream := 1 -- hour
  let distance_upstream := speed_upstream * time_upstream
  let distance_downstream := speed_downstream * time_downstream
  distance_upstream + distance_downstream = 62 := by
  sorry

end total_distance_covered_l228_228633


namespace product_gamma_l228_228964

noncomputable def gamma_params {α γ : ℝ} (xi : ℝ) (zeta : ℝ) :=
  xi ~ gamma_dist γ 1 ∧ zeta ~ beta_dist α (γ - α) ∧ 0 < α ∧ α < γ ∧ 0 < 1 ∧ indep xi zeta

lemma laplace_unique {α γ : ℝ} (xi zeta : ℝ) (h : gamma_params xi zeta) :
  laplace_transform (zeta * xi) = laplace_transform (gamma_dist γ 1) :=
sorry

theorem product_gamma {α γ λ: ℝ} (xi: ℝ) (zeta: ℝ) : 
  (xi ~ gamma_dist γ λ ∧ zeta ~ beta_dist α (γ - α) ∧ indep xi zeta ∧ 0 < α ∧ α < γ ∧ 0 < λ) 
  → (zeta * xi ∼ gamma_dist γ λ) :=
begin
  intros h,
  have h_gamma : gamma_params xi zeta,
  { split,
    exact h.1,
    split,
    exact h.2,
    split,
    exact h.3,
    exact h.4,
    exact h.5,
    exact h.6, },
  have h_laplace : laplace_transform (zeta * xi) = laplace_transform (gamma_dist γ 1),
  { apply laplace_unique,
    exact h_gamma, },
  admit
end

end product_gamma_l228_228964


namespace probability_of_exactly_one_hitting_l228_228026

variable (P_A_hitting B_A_hitting : ℝ)

theorem probability_of_exactly_one_hitting (hP_A : P_A_hitting = 0.6) (hP_B : B_A_hitting = 0.6) :
  ((P_A_hitting * (1 - B_A_hitting)) + ((1 - P_A_hitting) * B_A_hitting)) = 0.48 := 
by 
  sorry

end probability_of_exactly_one_hitting_l228_228026


namespace students_taking_neither_l228_228189

theorem students_taking_neither (total_students music art science music_and_art music_and_science art_and_science three_subjects : ℕ)
  (h1 : total_students = 800)
  (h2 : music = 80)
  (h3 : art = 60)
  (h4 : science = 50)
  (h5 : music_and_art = 30)
  (h6 : music_and_science = 25)
  (h7 : art_and_science = 20)
  (h8 : three_subjects = 15) :
  total_students - (music + art + science - music_and_art - music_and_science - art_and_science + three_subjects) = 670 :=
by sorry

end students_taking_neither_l228_228189


namespace family_reunion_kids_l228_228059

theorem family_reunion_kids (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h_adults : adults = 123) (h_tables : tables = 14) 
  (h_people_per_table : people_per_table = 12) :
  (tables * people_per_table - adults) = 45 :=
by
  sorry

end family_reunion_kids_l228_228059


namespace mutually_exclusive_not_opposite_l228_228892

open Probability

-- Define the people and cards
inductive Person : Type
| A | B | C | D
inductive Card : Type
| Red | Yellow | Blue | White

-- Define a random distribution of cards to people
def random_distribution : Person → Card → Prop :=
λ p c, true

def event_A_gets_red_card := random_distribution Person.A Card.Red
def event_B_gets_blue_card := random_distribution Person.B Card.Blue

theorem mutually_exclusive_not_opposite :
  MutuallyExclusive event_A_gets_red_card event_B_gets_blue_card ∧
  ¬ Opposite event_A_gets_red_card event_B_gets_blue_card := sorry

end mutually_exclusive_not_opposite_l228_228892


namespace find_value_l228_228707

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom symmetric_about_one : ∀ x, f (x - 1) = f (1 - x)
axiom equation_on_interval : ∀ x, 0 < x ∧ x < 1 → f x = 9^x

theorem find_value : f (5 / 2) + f 2 = -3 := 
by sorry

end find_value_l228_228707


namespace quadratic_roots_properties_l228_228078

-- Given the quadratic equation x^2 - 7x + 12 = 0
-- Prove that the absolute value of the difference of the roots is 1
-- Prove that the maximum value of the roots is 4

theorem quadratic_roots_properties :
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → abs (r1 - r2) = 1) ∧ 
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → max r1 r2 = 4) :=
by sorry

end quadratic_roots_properties_l228_228078


namespace min_points_in_set_M_l228_228166
-- Import the necessary library

-- Define the problem conditions and the result to prove
theorem min_points_in_set_M :
  ∃ (M : Finset ℝ) (C₁ C₂ C₃ C₄ C₅ C₆ C₇ : Finset ℝ),
  C₇.card = 7 ∧
  C₆.card = 6 ∧
  C₅.card = 5 ∧
  C₄.card = 4 ∧
  C₃.card = 3 ∧
  C₂.card = 2 ∧
  C₁.card = 1 ∧
  C₇ ⊆ M ∧
  C₆ ⊆ M ∧
  C₅ ⊆ M ∧
  C₄ ⊆ M ∧
  C₃ ⊆ M ∧
  C₂ ⊆ M ∧
  C₁ ⊆ M ∧
  M.card = 12 :=
sorry

end min_points_in_set_M_l228_228166


namespace Steven_has_more_peaches_l228_228401

variable (Steven_peaches : Nat) (Jill_peaches : Nat)
variable (h1 : Steven_peaches = 19) (h2 : Jill_peaches = 6)

theorem Steven_has_more_peaches : Steven_peaches - Jill_peaches = 13 :=
by
  sorry

end Steven_has_more_peaches_l228_228401


namespace tan_of_angle_in_fourth_quadrant_l228_228090

-- Define the angle α in the fourth quadrant in terms of its cosine value
variable (α : Real)
variable (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) -- fourth quadrant condition
variable (h2 : Real.cos α = 4/5) -- given condition

-- Define the proof problem that tan α equals -3/4 given the conditions
theorem tan_of_angle_in_fourth_quadrant (α : Real) (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) (h2 : Real.cos α = 4/5) : 
  Real.tan α = -3/4 :=
sorry

end tan_of_angle_in_fourth_quadrant_l228_228090


namespace computation_l228_228210

theorem computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  have h₁ : 27 = 3^3 := by rfl
  have h₂ : (3 : ℕ) ^ 4 = 81 := by norm_num
  have h₃ : 27^63 / 27^61 = (3^3)^63 / (3^3)^61 := by rw [h₁]
  rwa [← pow_sub, nat.sub_eq_iff_eq_add] at h₃
  have h4: 3 * 3^4 = 3^5 := by norm_num
  have h5: -486 = 3^5 - 3^6 := by norm_num
  exact h5
  sorry

end computation_l228_228210


namespace catering_budget_l228_228119

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end catering_budget_l228_228119


namespace number_divisible_by_19_l228_228570

theorem number_divisible_by_19 (n : ℕ) : (12000 + 3 * 10^n + 8) % 19 = 0 := 
by sorry

end number_divisible_by_19_l228_228570


namespace complex_computation_l228_228213

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l228_228213


namespace hours_per_day_l228_228038

theorem hours_per_day (H : ℕ) : 
  (42 * 12 * H = 30 * 14 * 6) → 
  H = 5 := by
  sorry

end hours_per_day_l228_228038


namespace cody_books_reading_l228_228486

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end cody_books_reading_l228_228486


namespace fixed_rate_calculation_l228_228875

theorem fixed_rate_calculation (f n : ℕ) (h1 : f + 4 * n = 220) (h2 : f + 7 * n = 370) : f = 20 :=
by
  sorry

end fixed_rate_calculation_l228_228875


namespace xyz_line_segments_total_length_l228_228103

noncomputable def total_length_XYZ : ℝ :=
  let length_X := 2 * Real.sqrt 2
  let length_Y := 2 + 2 * Real.sqrt 2
  let length_Z := 2 + Real.sqrt 2
  length_X + length_Y + length_Z

theorem xyz_line_segments_total_length : total_length_XYZ = 4 + 5 * Real.sqrt 2 := 
  sorry

end xyz_line_segments_total_length_l228_228103


namespace rods_in_one_mile_l228_228669

theorem rods_in_one_mile (chains_in_mile : ℕ) (rods_in_chain : ℕ) (mile_to_chain : 1 = 10 * chains_in_mile) (chain_to_rod : 1 = 22 * rods_in_chain) :
  1 * 220 = 10 * 22 :=
by sorry

end rods_in_one_mile_l228_228669


namespace gcd_polynomial_l228_228671

theorem gcd_polynomial (b : ℕ) (h : 570 ∣ b) : Nat.gcd (5 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l228_228671


namespace directrix_of_parabola_l228_228791

-- Define the given conditions
def parabola_focus_on_line (p : ℝ) := ∃ (x y : ℝ), y^2 = 2 * p * x ∧ 2 * x + 3 * y - 8 = 0

-- Define the statement to be proven
theorem directrix_of_parabola (p : ℝ) (h: parabola_focus_on_line p) : 
   ∃ (d : ℝ), d = -4 := 
sorry

end directrix_of_parabola_l228_228791


namespace Ella_food_each_day_l228_228396

variable {E : ℕ} -- Define E as the number of pounds of food Ella eats each day

def food_dog_eats (E : ℕ) : ℕ := 4 * E -- Definition of food the dog eats each day

def total_food_eaten_in_10_days (E : ℕ) : ℕ := 10 * E + 10 * (food_dog_eats E) -- Total food (Ella and dog) in 10 days

theorem Ella_food_each_day : total_food_eaten_in_10_days E = 1000 → E = 20 :=
by
  intros h -- Assume the given condition
  sorry -- Skip the actual proof

end Ella_food_each_day_l228_228396


namespace area_of_inscribed_rectangle_l228_228472

theorem area_of_inscribed_rectangle
  (s : ℕ) (R_area : ℕ)
  (h1 : s = 4) 
  (h2 : 2 * 4 + 1 * 1 + R_area = s * s) :
  R_area = 7 :=
by
  sorry

end area_of_inscribed_rectangle_l228_228472


namespace september_first_2021_was_wednesday_l228_228050

-- Defining the main theorem based on the conditions and the question
theorem september_first_2021_was_wednesday
  (doubledCapitalOnWeekdays : ∀ day : Nat, day = 0 % 7 → True)
  (sevenFiftyPercOnWeekends : ∀ day : Nat, day = 5 % 7 → True)
  (millionaireOnLastDayOfYear: ∀ day : Nat, day = 364 % 7 → True)
  : 1 % 7 = 3 % 7 := 
sorry

end september_first_2021_was_wednesday_l228_228050


namespace beavers_build_dam_l228_228301

def num_beavers_first_group : ℕ := 20

theorem beavers_build_dam (B : ℕ) (t₁ : ℕ) (t₂ : ℕ) (n₂ : ℕ) :
  (B * t₁ = n₂ * t₂) → (B = num_beavers_first_group) := 
by
  -- Given
  let t₁ := 3
  let t₂ := 5
  let n₂ := 12

  -- Work equation
  assume h : B * t₁ = n₂ * t₂
  
  -- Correct answer
  have B_def : B = (n₂ * t₂) / t₁,
  exact h
   
  sorry

end beavers_build_dam_l228_228301


namespace inequality_a_b_c_d_l228_228408

theorem inequality_a_b_c_d
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h₄ : a * b + b * c + c * d + d * a = 1) :
  (a ^ 3 / (b + c + d) + b ^ 3 / (c + d + a) + c ^ 3 / (a + b + d) + d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_a_b_c_d_l228_228408


namespace possible_values_of_a_l228_228678

def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a :
  {a | M a ⊆ P} = {1, -1, 0} :=
sorry

end possible_values_of_a_l228_228678


namespace water_fee_part1_water_fee_part2_water_fee_usage_l228_228064

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end water_fee_part1_water_fee_part2_water_fee_usage_l228_228064


namespace gumballs_remaining_l228_228053

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end gumballs_remaining_l228_228053


namespace sqrt_meaningful_range_l228_228858

theorem sqrt_meaningful_range (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 1)) ↔ x ≥ 1 :=
by 
  sorry

end sqrt_meaningful_range_l228_228858


namespace a679b_multiple_of_72_l228_228249

-- Define conditions
def is_divisible_by_8 (n : Nat) : Prop :=
  n % 8 = 0

def sum_of_digits_is_divisible_by_9 (n : Nat) : Prop :=
  (n.digits 10).sum % 9 = 0

-- Define the given problem
theorem a679b_multiple_of_72 (a b : Nat) : 
  is_divisible_by_8 (7 * 100 + 9 * 10 + b) →
  sum_of_digits_is_divisible_by_9 (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) → 
  a = 3 ∧ b = 2 :=
by 
  sorry

end a679b_multiple_of_72_l228_228249


namespace abs_lt_two_nec_but_not_suff_l228_228336

theorem abs_lt_two_nec_but_not_suff (x : ℝ) :
  (|x - 1| < 2) → (0 < x ∧ x < 3) ∧ ¬((0 < x ∧ x < 3) → (|x - 1| < 2)) := sorry

end abs_lt_two_nec_but_not_suff_l228_228336


namespace part1_part2_part3_l228_228474

-- Definitions from the problem
def initial_cost_per_bottle := 16
def initial_selling_price := 20
def initial_sales_volume := 60
def sales_decrease_per_yuan_increase := 5

def daily_sales_volume (x : ℕ) : ℕ :=
  initial_sales_volume - sales_decrease_per_yuan_increase * x

def profit_per_bottle (x : ℕ) : ℕ :=
  (initial_selling_price - initial_cost_per_bottle) + x

def daily_profit (x : ℕ) : ℕ :=
  daily_sales_volume x * profit_per_bottle x

-- The proofs we need to establish
theorem part1 (x : ℕ) : 
  daily_sales_volume x = 60 - 5 * x ∧ profit_per_bottle x = 4 + x :=
sorry

theorem part2 (x : ℕ) : 
  daily_profit x = 300 → x = 6 ∨ x = 2 :=
sorry

theorem part3 : 
  ∃ x : ℕ, ∀ y : ℕ, (daily_profit x < daily_profit y) → 
              (daily_profit x = 320 ∧ x = 4) :=
sorry

end part1_part2_part3_l228_228474


namespace dorms_and_students_l228_228983

theorem dorms_and_students (x : ℕ) :
  (4 * x + 19) % 6 ≠ 0 → ∃ s : ℕ, (x = 10 ∧ s = 59) ∨ (x = 11 ∧ s = 63) ∨ (x = 12 ∧ s = 67) :=
by
  sorry

end dorms_and_students_l228_228983


namespace arithmetic_mean_first_n_positive_integers_l228_228995

theorem arithmetic_mean_first_n_positive_integers (n : ℕ) (Sn : ℕ) (h : Sn = n * (n + 1) / 2) : 
  (Sn / n) = (n + 1) / 2 := by
  -- proof steps would go here
  sorry

end arithmetic_mean_first_n_positive_integers_l228_228995


namespace units_digit_of_2011_odd_squares_l228_228323

def units_digit_sum_squares_first_k_odd_integers (k : ℕ) : ℕ :=
  let odd_numbers := List.range k |>.map (λ n, 2*n + 1)
  let squares := odd_numbers.map (λ n, n^2)
  let total_sum := squares.sum
  total_sum % 10

theorem units_digit_of_2011_odd_squares : units_digit_sum_squares_first_k_odd_integers 2011 = 9 :=
by
  sorry

end units_digit_of_2011_odd_squares_l228_228323


namespace fraction_of_alvin_age_l228_228495

variable (A E F : ℚ)

-- Conditions
def edwin_older_by_six : Prop := E = A + 6
def total_age : Prop := A + E = 30.99999999
def age_relation_in_two_years : Prop := E + 2 = F * (A + 2) + 20

-- Statement to prove
theorem fraction_of_alvin_age
  (h1 : edwin_older_by_six A E)
  (h2 : total_age A E)
  (h3 : age_relation_in_two_years A E F) :
  F = 1 / 29 :=
sorry

end fraction_of_alvin_age_l228_228495


namespace perimeter_reduction_percentage_l228_228335

-- Given initial dimensions x and y
-- Initial Perimeter
def initial_perimeter (x y : ℝ) : ℝ := 2 * (x + y)

-- First reduction
def first_reduction_length (x : ℝ) : ℝ := 0.9 * x
def first_reduction_width (y : ℝ) : ℝ := 0.8 * y

-- New perimeter after first reduction
def new_perimeter_first (x y : ℝ) : ℝ := 2 * (first_reduction_length x + first_reduction_width y)

-- Condition: new perimeter is 88% of the initial perimeter
def perimeter_condition (x y : ℝ) : Prop := new_perimeter_first x y = 0.88 * initial_perimeter x y

-- Solve for x in terms of y
def solve_for_x (y : ℝ) : ℝ := 4 * y

-- Second reduction
def second_reduction_length (x : ℝ) : ℝ := 0.8 * x
def second_reduction_width (y : ℝ) : ℝ := 0.9 * y

-- New perimeter after second reduction
def new_perimeter_second (x y : ℝ) : ℝ := 2 * (second_reduction_length x + second_reduction_width y)

-- Proof statement
theorem perimeter_reduction_percentage (x y : ℝ) (h : perimeter_condition x y) : 
  new_perimeter_second x y = 0.82 * initial_perimeter x y :=
by
  sorry

end perimeter_reduction_percentage_l228_228335


namespace altitude_identity_l228_228023

variable {a b c d : ℝ}

def is_right_triangle (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2

def right_angle_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

def altitude_property (a b c d : ℝ) : Prop :=
  a * b = c * d

theorem altitude_identity (a b c d : ℝ) (h1: right_angle_triangle a b c) (h2: altitude_property a b c d) :
  1 / a^2 + 1 / b^2 = 1 / d^2 :=
sorry

end altitude_identity_l228_228023


namespace cost_to_fill_pool_l228_228961

noncomputable def pool_cost : ℝ :=
  let base_width := 6
  let top_width := 4
  let length := 20
  let depth := 10
  let conversion_factor := 25
  let price_per_liter := 3
  let tax_rate := 0.08
  let discount_rate := 0.05
  let volume := 0.5 * depth * (base_width + top_width) * length
  let liters := volume * conversion_factor
  let initial_cost := liters * price_per_liter
  let cost_with_tax := initial_cost * (1 + tax_rate)
  let final_cost := cost_with_tax * (1 - discount_rate)
  final_cost

theorem cost_to_fill_pool : pool_cost = 76950 := by
  sorry

end cost_to_fill_pool_l228_228961


namespace distance_between_houses_l228_228400

-- Definitions
def speed : ℝ := 2          -- Amanda's speed in miles per hour
def time : ℝ := 3           -- Time taken by Amanda in hours

-- The theorem to prove distance is 6 miles
theorem distance_between_houses : speed * time = 6 := by
  sorry

end distance_between_houses_l228_228400


namespace no_integer_roots_l228_228498

theorem no_integer_roots : ∀ x : ℤ, x^3 - 3 * x^2 - 16 * x + 20 ≠ 0 := by
  intro x
  sorry

end no_integer_roots_l228_228498


namespace min_inequality_l228_228406

theorem min_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz_sum : x + y + z = 9) :
    ( \frac{x^2 + y^2}{3*(x+y)} + \frac{x^2 + z^2}{3*(x+z)} + \frac{y^2 + z^2}{3*(y+z)} ) ≥ 3 :=
by
  sorry

end min_inequality_l228_228406


namespace peter_pizza_fraction_l228_228753

def pizza_slices : ℕ := 16
def peter_initial_slices : ℕ := 2
def shared_slices : ℕ := 2
def shared_with_paul : ℕ := shared_slices / 2
def total_slices_peter_ate := peter_initial_slices + shared_with_paul
def fraction_peter_ate : ℚ := total_slices_peter_ate / pizza_slices

theorem peter_pizza_fraction :
  fraction_peter_ate = 3 / 16 :=
by
  -- Leave space for the proof, which is not required.
  sorry

end peter_pizza_fraction_l228_228753


namespace productivity_after_repair_l228_228837

-- Define the initial productivity and the increase factor.
def original_productivity : ℕ := 10
def increase_factor : ℝ := 1.5

-- Define the expected productivity after the improvement.
def expected_productivity : ℝ := 25

-- The theorem we need to prove.
theorem productivity_after_repair :
  original_productivity * (1 + increase_factor) = expected_productivity := by
  sorry

end productivity_after_repair_l228_228837


namespace gumballs_remaining_l228_228055

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end gumballs_remaining_l228_228055


namespace completing_the_square_l228_228326

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l228_228326


namespace no_integer_solution_l228_228422

theorem no_integer_solution (x y z : ℤ) (n : ℕ) (h1 : Prime (x + y)) (h2 : Odd n) : ¬ (x^n + y^n = z^n) :=
sorry

end no_integer_solution_l228_228422


namespace hyperbola_asymptotes_l228_228069

-- Define the data for the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

-- Define the two equations for the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = 4 / 5 * x + 13 / 5
def asymptote2 (x y : ℝ) : Prop := y = -4 / 5 * x + 13 / 5

-- Theorem stating that the given asymptotes are correct for the hyperbola
theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, hyperbola_eq x y → (asymptote1 x y ∨ asymptote2 x y)) := 
by
  sorry

end hyperbola_asymptotes_l228_228069


namespace two_lines_perpendicular_to_same_plane_are_parallel_l228_228744

/- 
Problem: Let a, b be two lines, and α be a plane. Prove that if a ⊥ α and b ⊥ α, then a ∥ b.
-/

variables {Line Plane : Type} 

def is_parallel (l1 l2 : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry
def is_contained_in (l : Line) (p : Plane) : Prop := sorry

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane)
  (ha_perpendicular : is_perpendicular a α)
  (hb_perpendicular : is_perpendicular b α) :
  is_parallel a b :=
by
  sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l228_228744


namespace bridget_bakery_profit_l228_228063

theorem bridget_bakery_profit :
  let loaves := 36
  let cost_per_loaf := 1
  let morning_sale_price := 3
  let afternoon_sale_price := 1.5
  let late_afternoon_sale_price := 1
  
  let morning_loaves := (2/3 : ℝ) * loaves
  let morning_revenue := morning_loaves * morning_sale_price
  
  let remaining_after_morning := loaves - morning_loaves
  let afternoon_loaves := (1/2 : ℝ) * remaining_after_morning
  let afternoon_revenue := afternoon_loaves * afternoon_sale_price
  
  let late_afternoon_loaves := remaining_after_morning - afternoon_loaves
  let late_afternoon_revenue := late_afternoon_loaves * late_afternoon_sale_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := loaves * cost_per_loaf
  
  total_revenue - total_cost = 51 := by sorry

end bridget_bakery_profit_l228_228063


namespace option_b_does_not_represent_5x_l228_228179

theorem option_b_does_not_represent_5x (x : ℝ) : 
  (∀ a, a = 5 * x ↔ a = x + x + x + x + x) →
  (¬ (5 * x = x * x * x * x * x)) :=
by
  intro h
  -- Using sorry to skip the proof.
  sorry

end option_b_does_not_represent_5x_l228_228179


namespace emily_speed_l228_228223

theorem emily_speed (distance time : ℝ) (h1 : distance = 10) (h2 : time = 2) : (distance / time) = 5 := 
by sorry

end emily_speed_l228_228223


namespace shipping_cost_correct_l228_228410

-- Definitions of given conditions
def total_weight_of_fish : ℕ := 540
def weight_of_each_crate : ℕ := 30
def total_shipping_cost : ℚ := 27

-- Calculating the number of crates
def number_of_crates : ℕ := total_weight_of_fish / weight_of_each_crate

-- Definition of the target shipping cost per crate
def shipping_cost_per_crate : ℚ := total_shipping_cost / number_of_crates

-- Lean statement to prove the given problem
theorem shipping_cost_correct :
  shipping_cost_per_crate = 1.50 := by
  sorry

end shipping_cost_correct_l228_228410


namespace response_rate_percentage_l228_228343

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 240) (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := 
by 
  sorry

end response_rate_percentage_l228_228343


namespace intersection_P_Q_l228_228098

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l228_228098


namespace minimum_value_of_f_l228_228376

def f (x a : ℝ) : ℝ := abs (x + 1) + abs (a * x + 1)

theorem minimum_value_of_f (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 / 2) →
  (∃ x : ℝ, f x a = 3 / 2) →
  (a = -1 / 2 ∨ a = -2) :=
by
  intros h1 h2
  sorry

end minimum_value_of_f_l228_228376


namespace number_of_outcomes_for_champions_l228_228622

def num_events : ℕ := 3
def num_competitors : ℕ := 6
def total_possible_outcomes : ℕ := num_competitors ^ num_events

theorem number_of_outcomes_for_champions :
  total_possible_outcomes = 216 :=
by
  sorry

end number_of_outcomes_for_champions_l228_228622


namespace range_of_real_number_m_l228_228708

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end range_of_real_number_m_l228_228708


namespace improved_productivity_l228_228838

-- Let the initial productivity be a constant
def initial_productivity : ℕ := 10

-- Let the increase factor be a constant, represented as a rational number
def increase_factor : ℚ := 3 / 2

-- The goal is to prove that the current productivity equals 25 trees daily
theorem improved_productivity : initial_productivity + (initial_productivity * increase_factor).toNat = 25 := 
by
  sorry

end improved_productivity_l228_228838


namespace shooting_competition_hits_l228_228391

noncomputable def a1 : ℝ := 1
noncomputable def d : ℝ := 0.5
noncomputable def S_n (n : ℝ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

theorem shooting_competition_hits (n : ℝ) (h : S_n n = 7) : 25 - n = 21 :=
by
  -- sequence of proof steps
  sorry

end shooting_competition_hits_l228_228391


namespace find_a2016_l228_228934

theorem find_a2016 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : ∀ n : ℕ, a (n + 2) = a (n + 1) - a n) : a 2016 = -2 := 
by sorry

end find_a2016_l228_228934


namespace mod_product_example_l228_228849

theorem mod_product_example :
  ∃ m : ℤ, 256 * 738 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 ∧ m = 53 :=
by
  use 53
  sorry

end mod_product_example_l228_228849


namespace marcy_total_spears_l228_228973

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end marcy_total_spears_l228_228973


namespace part_1_part_2_l228_228366

noncomputable def f (a m x : ℝ) := a ^ m / x

theorem part_1 (a : ℝ) (m : ℝ) (H1 : a > 1) (H2 : ∀ x, x ∈ Set.Icc a (2*a) → f a m x ∈ Set.Icc (a^2) (a^3)) :
  a = 2 :=
sorry

theorem part_2 (t : ℝ) (s : ℝ) (H1 : ∀ x, x ∈ Set.Icc 0 s → (x + t) ^ 2 + 2 * (x + t) ≤ 3 * x) :
  s ∈ Set.Ioc 0 5 :=
sorry

end part_1_part_2_l228_228366


namespace problem_solution_l228_228520

theorem problem_solution (n : ℤ) : 
  (1 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4) → (n = -2) :=
by
  intro h
  sorry

end problem_solution_l228_228520


namespace trapezoid_area_is_correct_l228_228952

noncomputable def trapezoid_area (base_short : ℝ) (angle_adj : ℝ) (angle_diag : ℝ) : ℝ :=
  let width := 2 * base_short -- calculated width from angle_adj
  let height := base_short / Real.tan (angle_adj / 2 * Real.pi / 180)
  (base_short + base_short + width) * height / 2

theorem trapezoid_area_is_correct :
  trapezoid_area 2 135 150 = 2 :=
by
  sorry

end trapezoid_area_is_correct_l228_228952


namespace Jim_catches_Bob_in_20_minutes_l228_228207

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end Jim_catches_Bob_in_20_minutes_l228_228207


namespace beaver_group_count_l228_228297

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l228_228297


namespace find_divided_number_l228_228165

theorem find_divided_number:
  ∃ x : ℕ, (x % 127 = 6) ∧ (2037 % 127 = 5) ∧ x = 2038 :=
by
  sorry

end find_divided_number_l228_228165


namespace find_m_l228_228860

theorem find_m
  (x y : ℝ)
  (h1 : 100 = 300 * x + 200 * y)
  (h2 : 120 = 240 * x + 300 * y)
  (h3 : ∃ m : ℝ, 50 * 3 = 150 * x + m * y):
  ∃ m : ℝ, m = 450 :=
by
  sorry

end find_m_l228_228860


namespace first_term_arithmetic_sum_l228_228271

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l228_228271


namespace ellipse_hyperbola_tangent_l228_228592

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end ellipse_hyperbola_tangent_l228_228592


namespace danny_distance_to_work_l228_228768

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_l228_228768


namespace solve_for_x_l228_228580

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l228_228580


namespace ratio_of_c_d_l228_228887

theorem ratio_of_c_d 
  (x y c d : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c)
  (h2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 :=
by 
  sorry

end ratio_of_c_d_l228_228887


namespace find_a_l228_228667

noncomputable def A (a : ℝ) : Set ℝ := {2^a, 3}
def B : Set ℝ := {2, 3}
def C : Set ℝ := {1, 2, 3}

theorem find_a (a : ℝ) (h : A a ∪ B = C) : a = 0 :=
sorry

end find_a_l228_228667


namespace trapezoid_area_l228_228550

theorem trapezoid_area (AD BC AC : ℝ) (BD : ℝ) 
  (hAD : AD = 24) 
  (hBC : BC = 8) 
  (hAC : AC = 13) 
  (hBD : BD = 5 * Real.sqrt 17) : 
  (1 / 2 * (AD + BC) * Real.sqrt (AC^2 - (BC + (AD - BC) / 2)^2)) = 80 :=
by
  sorry

end trapezoid_area_l228_228550


namespace smallest_positive_x_l228_228902

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l228_228902


namespace replacement_paint_intensity_l228_228144

theorem replacement_paint_intensity 
  (P_original : ℝ) (P_new : ℝ) (f : ℝ) (I : ℝ) :
  P_original = 50 →
  P_new = 45 →
  f = 0.2 →
  0.8 * P_original + f * I = P_new →
  I = 25 :=
by
  intros
  sorry

end replacement_paint_intensity_l228_228144


namespace parallelogram_side_length_l228_228345

theorem parallelogram_side_length 
  (s : ℝ) 
  (A : ℝ)
  (angle : ℝ)
  (adj1 adj2 : ℝ) 
  (h : adj1 = s) 
  (h1 : adj2 = 2 * s) 
  (h2 : angle = 30)
  (h3 : A = 8 * Real.sqrt 3): 
  s = 2 * Real.sqrt 2 :=
by
  -- sorry to skip proofs
  sorry

end parallelogram_side_length_l228_228345


namespace probability_AEMC9_is_1_over_84000_l228_228399

-- Define possible symbols for each category.
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def nonVowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

-- Define the total number of possible license plates.
def totalLicensePlates : Nat := 
  (vowels.length) * (vowels.length - 1) * 
  (nonVowels.length) * (nonVowels.length - 1) * 
  (digits.length)

-- Define the number of favorable outcomes.
def favorableOutcomes : Nat := 1

-- Define the probability calculation.
noncomputable def probabilityAEMC9 : ℚ := favorableOutcomes / totalLicensePlates

-- The theorem to prove.
theorem probability_AEMC9_is_1_over_84000 :
  probabilityAEMC9 = 1 / 84000 := by
  sorry

end probability_AEMC9_is_1_over_84000_l228_228399


namespace number_of_points_on_line_l228_228291

theorem number_of_points_on_line (a b c d : ℕ) (h1 : a * b = 80) (h2 : c * d = 90) (h3 : a + b = c + d) :
  a + b + 1 = 22 :=
sorry

end number_of_points_on_line_l228_228291


namespace cos_theta_result_projection_result_l228_228938

variables (a b : ℝ × ℝ) (θ : ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
(dot_product a b) / ((magnitude a) * (magnitude b))

def projection (a b : ℝ × ℝ) : ℝ :=
magnitude b * cos_theta a b

theorem cos_theta_result : cos_theta (2, 3) (-2, 4) = 4 / Real.sqrt 65 :=
by sorry

theorem projection_result : projection (2, 3) (-2, 4) = 8 * Real.sqrt 13 / 13 :=
by sorry

end cos_theta_result_projection_result_l228_228938


namespace cost_of_largest_pot_is_2_52_l228_228131

/-
Mark bought a set of 6 flower pots of different sizes at a total pre-tax cost.
Each pot cost 0.4 more than the next one below it in size.
The total cost, including a sales tax of 7.5%, was $9.80.
Prove that the cost of the largest pot before sales tax was $2.52.
-/

def cost_smallest_pot (x : ℝ) : Prop :=
  let total_cost := x + (x + 0.4) + (x + 0.8) + (x + 1.2) + (x + 1.6) + (x + 2.0)
  let pre_tax_cost := total_cost / 1.075
  let pre_tax_total_cost := (9.80 / 1.075)
  (total_cost = 6 * x + 6 ∧ total_cost = pre_tax_total_cost) →
  (x + 2.0 = 2.52)

theorem cost_of_largest_pot_is_2_52 :
  ∃ x : ℝ, cost_smallest_pot x :=
sorry

end cost_of_largest_pot_is_2_52_l228_228131


namespace sequence_value_2016_l228_228922

theorem sequence_value_2016 (a : ℕ → ℕ) (h₁ : a 1 = 0) (h₂ : ∀ n, a (n + 1) = a n + 2 * n) : a 2016 = 2016 * 2015 :=
by 
  sorry

end sequence_value_2016_l228_228922


namespace cecile_apples_l228_228494

theorem cecile_apples (C D : ℕ) (h1 : D = C + 20) (h2 : C + D = 50) : C = 15 :=
by
  -- Proof steps would go here
  sorry

end cecile_apples_l228_228494


namespace jake_and_luke_items_l228_228260

theorem jake_and_luke_items :
  ∃ (p j : ℕ), 6 * p + 2 * j ≤ 50 ∧ (∀ (p' : ℕ), 6 * p' + 2 * j ≤ 50 → p' ≤ p) ∧ p + j = 9 :=
by
  sorry

end jake_and_luke_items_l228_228260


namespace solution_inequality_l228_228440

open Real

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- State the theorem for the given proof problem
theorem solution_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by
  sorry

end solution_inequality_l228_228440


namespace absolute_value_c_l228_228304

noncomputable def condition_polynomial (a b c : ℤ) : Prop :=
  a * (↑(Complex.ofReal 3) + Complex.I)^4 +
  b * (↑(Complex.ofReal 3) + Complex.I)^3 +
  c * (↑(Complex.ofReal 3) + Complex.I)^2 +
  b * (↑(Complex.ofReal 3) + Complex.I) +
  a = 0

noncomputable def coprime_integers (a b c : ℤ) : Prop :=
  Int.gcd (Int.gcd a b) c = 1

theorem absolute_value_c (a b c : ℤ) (h1 : condition_polynomial a b c) (h2 : coprime_integers a b c) :
  |c| = 97 :=
sorry

end absolute_value_c_l228_228304


namespace not_divisible_by_121_l228_228815

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 2014)) :=
sorry

end not_divisible_by_121_l228_228815


namespace no_natural_n_for_perfect_square_l228_228224

theorem no_natural_n_for_perfect_square :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 2007 + 4^n = k^2 :=
by {
  sorry  -- Proof omitted
}

end no_natural_n_for_perfect_square_l228_228224


namespace find_first_term_arithmetic_sequence_l228_228276

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l228_228276


namespace complex_number_z_l228_228409

theorem complex_number_z (i : ℂ) (z : ℂ) (hi : i * i = -1) (h : 2 * i / z = 1 - i) : z = -1 + i :=
by
  sorry

end complex_number_z_l228_228409


namespace beaver_group_count_l228_228298

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l228_228298


namespace not_mysterious_diff_consecutive_odd_l228_228356

/-- A mysterious number is defined as the difference of squares of two consecutive even numbers. --/
def is_mysterious (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2 * k + 2)^2 - (2 * k)^2

/-- The difference of the squares of two consecutive odd numbers. --/
def diff_squares_consecutive_odd (k : ℤ) : ℤ :=
  (2 * k + 1)^2 - (2 * k - 1)^2

/-- Prove that the difference of squares of two consecutive odd numbers is not a mysterious number. --/
theorem not_mysterious_diff_consecutive_odd (k : ℤ) : ¬ is_mysterious (Int.natAbs (diff_squares_consecutive_odd k)) :=
by
  sorry

end not_mysterious_diff_consecutive_odd_l228_228356


namespace op_correct_l228_228598

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end op_correct_l228_228598


namespace necessarily_negative_b_plus_3b_squared_l228_228842

theorem necessarily_negative_b_plus_3b_squared
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 1) :
  b + 3 * b^2 < 0 :=
sorry

end necessarily_negative_b_plus_3b_squared_l228_228842


namespace polygon_sides_from_diagonals_l228_228384

theorem polygon_sides_from_diagonals (n D : ℕ) (h1 : D = 15) (h2 : D = n * (n - 3) / 2) : n = 8 :=
by
  -- skipping proof
  sorry

end polygon_sides_from_diagonals_l228_228384


namespace grandma_olga_daughters_l228_228683

theorem grandma_olga_daughters :
  ∃ (D : ℕ), ∃ (S : ℕ),
  S = 3 ∧
  (∃ (total_grandchildren : ℕ), total_grandchildren = 33) ∧
  (∀ D', 6 * D' + 5 * S = 33 → D = D')
:=
sorry

end grandma_olga_daughters_l228_228683


namespace fruit_cost_l228_228471

theorem fruit_cost:
  let strawberry_cost := 2.20
  let cherry_cost := 6 * strawberry_cost
  let blueberry_cost := cherry_cost / 2
  let strawberries_count := 3
  let cherries_count := 4.5
  let blueberries_count := 6.2
  let total_cost := (strawberries_count * strawberry_cost) + (cherries_count * cherry_cost) + (blueberries_count * blueberry_cost)
  total_cost = 106.92 :=
by
  sorry

end fruit_cost_l228_228471


namespace students_both_courses_l228_228543

-- Definitions from conditions
def total_students : ℕ := 87
def students_french : ℕ := 41
def students_german : ℕ := 22
def students_neither : ℕ := 33

-- The statement we need to prove
theorem students_both_courses : (students_french + students_german - 9 + students_neither = total_students) → (9 = 96 - total_students) :=
by
  -- The proof would go here, but we leave it as sorry for now
  sorry

end students_both_courses_l228_228543


namespace length_of_train_correct_l228_228041

noncomputable def length_of_train (time_pass_man : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  relative_speed_ms * time_pass_man

theorem length_of_train_correct :
  length_of_train 29.997600191984642 60 6 = 449.96400287976963 := by
  sorry

end length_of_train_correct_l228_228041


namespace equivalent_operation_l228_228180

theorem equivalent_operation : 
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  (initial_op * multiply_3_2) = (5 / 4 : ℝ) :=
by
  -- setup operations
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  -- state the goal
  have h : (initial_op * multiply_3_2) = (5 / 4 : ℝ) := sorry
  exact h

end equivalent_operation_l228_228180


namespace smallest_pos_multiple_6_15_is_30_l228_228504

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l228_228504


namespace original_number_is_16_l228_228689

theorem original_number_is_16 (x : ℕ) : 213 * x = 3408 → x = 16 :=
by
  sorry

end original_number_is_16_l228_228689


namespace max_sum_of_circle_eq_eight_l228_228193

noncomputable def max_sum_of_integer_solutions (r : ℕ) : ℕ :=
  if r = 6 then 8 else 0

theorem max_sum_of_circle_eq_eight 
  (h1 : ∃ (x y : ℤ), (x - 1)^2 + (y - 1)^2 = 36 ∧ (r : ℕ) = 6) :
  max_sum_of_integer_solutions r = 8 := 
by
  sorry

end max_sum_of_circle_eq_eight_l228_228193


namespace Edmund_can_wrap_15_boxes_every_3_days_l228_228218

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end Edmund_can_wrap_15_boxes_every_3_days_l228_228218


namespace monomial_2024_l228_228755

def monomial (n : ℕ) : ℤ × ℕ := ((-1)^(n + 1) * (2 * n - 1), n)

theorem monomial_2024 :
  monomial 2024 = (-4047, 2024) :=
sorry

end monomial_2024_l228_228755


namespace pies_count_l228_228885

-- Definitions based on the conditions given in the problem
def strawberries_per_pie := 3
def christine_strawberries := 10
def rachel_strawberries := 2 * christine_strawberries

-- The theorem to prove
theorem pies_count : (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 := by
  sorry

end pies_count_l228_228885


namespace zain_has_80_coins_l228_228864

theorem zain_has_80_coins (emerie_quarters emerie_dimes emerie_nickels emerie_pennies emerie_half_dollars : ℕ)
  (h_quarters : emerie_quarters = 6) 
  (h_dimes : emerie_dimes = 7)
  (h_nickels : emerie_nickels = 5)
  (h_pennies : emerie_pennies = 10) 
  (h_half_dollars : emerie_half_dollars = 2) : 
  10 + emerie_quarters + 10 + emerie_dimes + 10 + emerie_nickels + 10 + emerie_pennies + 10 + emerie_half_dollars = 80 :=
by
  sorry

end zain_has_80_coins_l228_228864


namespace M_inter_N_is_1_2_l228_228795

-- Definitions based on given conditions
def M : Set ℝ := { y | ∃ x : ℝ, x > 0 ∧ y = 2^x }
def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Prove intersection of M and N is (1, 2]
theorem M_inter_N_is_1_2 :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end M_inter_N_is_1_2_l228_228795


namespace proof_problem_l228_228083

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := 
  (Real.sin (2 * x), 2 * Real.cos x ^ 2 - 1)

noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := 
  (Real.sin θ, Real.cos θ)

noncomputable def f (x θ : ℝ) : ℝ := 
  (vector_a x).1 * (vector_b θ).1 + (vector_a x).2 * (vector_b θ).2

theorem proof_problem 
  (θ : ℝ) 
  (hθ : 0 < θ ∧ θ < π) 
  (h1 : f (π / 6) θ = 1) 
  (x : ℝ) 
  (hx : -π / 6 ≤ x ∧ x ≤ π / 4) :
  θ = π / 3 ∧
  (∀ x, f x θ = f (x + π) θ) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≤ 1) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≥ -0.5) :=
by
  sorry

end proof_problem_l228_228083


namespace vector_perpendicular_l228_228680

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 3)
def vec_diff : ℝ × ℝ := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_perpendicular :
  dot_product vec_a vec_diff = 0 := by
  sorry

end vector_perpendicular_l228_228680


namespace no_primes_in_sequence_l228_228086

-- Definitions and conditions derived from the problem statement
variable (a : ℕ → ℕ) -- sequence of natural numbers
variable (increasing : ∀ n, a n < a (n + 1)) -- increasing sequence
variable (is_arith_or_geom : ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) ^ 2 = a n * a (n + 2))) -- arithmetic or geometric progression condition
variable (divisible_by_four : a 0 % 4 = 0 ∧ a 1 % 4 = 0) -- first two numbers divisible by 4

-- The statement to prove: no prime numbers exist in the sequence
theorem no_primes_in_sequence : ∀ n, ¬ (Nat.Prime (a n)) :=
by 
  sorry

end no_primes_in_sequence_l228_228086


namespace number_of_solid_figures_is_4_l228_228478

def is_solid_figure (shape : String) : Bool :=
  shape = "cone" ∨ shape = "cuboid" ∨ shape = "sphere" ∨ shape = "triangular prism"

def shapes : List String :=
  ["circle", "square", "cone", "cuboid", "line segment", "sphere", "triangular prism", "right-angled triangle"]

def number_of_solid_figures : Nat :=
  (shapes.filter is_solid_figure).length

theorem number_of_solid_figures_is_4 : number_of_solid_figures = 4 :=
  by sorry

end number_of_solid_figures_is_4_l228_228478


namespace solve_for_x_l228_228868

theorem solve_for_x (x : ℝ) (h : (x / 5) + 3 = 4) : x = 5 :=
by
  sorry

end solve_for_x_l228_228868


namespace total_number_of_students_l228_228741

namespace StudentRanking

def rank_from_right := 17
def rank_from_left := 5
def total_students (rank_from_right rank_from_left : ℕ) := rank_from_right + rank_from_left - 1

theorem total_number_of_students : total_students rank_from_right rank_from_left = 21 :=
by
  sorry

end StudentRanking

end total_number_of_students_l228_228741


namespace fraction_of_square_shaded_is_half_l228_228256

theorem fraction_of_square_shaded_is_half {s : ℝ} (h : s > 0) :
  let O := (0, 0)
  let P := (0, s)
  let Q := (s, s / 2)
  let area_square := s^2
  let area_triangle_OPQ := 1 / 2 * s^2 / 2
  let shaded_area := area_square - area_triangle_OPQ
  (shaded_area / area_square) = 1 / 2 :=
by
  sorry

end fraction_of_square_shaded_is_half_l228_228256


namespace algebraic_expression_value_l228_228682

variable {R : Type} [CommRing R]

theorem algebraic_expression_value (m n : R) (h1 : m - n = -2) (h2 : m * n = 3) :
  -m^3 * n + 2 * m^2 * n^2 - m * n^3 = -12 :=
sorry

end algebraic_expression_value_l228_228682


namespace solve_equation_l228_228143

theorem solve_equation {x : ℝ} (h : x ≠ -2) : (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) → x = 1 :=
by
  intro h_eq
  -- proof steps would go here
  sorry

end solve_equation_l228_228143


namespace sum_x_coordinates_Q4_is_3000_l228_228745

-- Let Q1 be a 150-gon with vertices having x-coordinates summing to 3000
def Q1_x_sum := 3000
def Q2_x_sum := Q1_x_sum
def Q3_x_sum := Q2_x_sum
def Q4_x_sum := Q3_x_sum

-- Theorem to prove the sum of the x-coordinates of the vertices of Q4 is 3000
theorem sum_x_coordinates_Q4_is_3000 : Q4_x_sum = 3000 := by
  sorry

end sum_x_coordinates_Q4_is_3000_l228_228745


namespace GCF_LCM_proof_l228_228407

-- Define GCF (greatest common factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (least common multiple)
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_proof :
  GCF (LCM 9 21) (LCM 14 15) = 21 :=
by
  sorry

end GCF_LCM_proof_l228_228407


namespace total_cost_is_90_l228_228699

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end total_cost_is_90_l228_228699


namespace distance_walked_is_18_miles_l228_228468

-- Defining the variables for speed, time, and distance
variables (x t d : ℕ)

-- Declaring the conditions given in the problem
def walked_distance_at_usual_rate : Prop :=
  d = x * t

def walked_distance_at_increased_rate : Prop :=
  d = (x + 1) * (3 * t / 4)

def walked_distance_at_decreased_rate : Prop :=
  d = (x - 1) * (t + 3)

-- The proof problem statement to show the distance walked is 18 miles
theorem distance_walked_is_18_miles
  (hx : walked_distance_at_usual_rate x t d)
  (hz : walked_distance_at_increased_rate x t d)
  (hy : walked_distance_at_decreased_rate x t d) :
  d = 18 := by
  sorry

end distance_walked_is_18_miles_l228_228468


namespace sum_of_five_distinct_integers_product_2022_l228_228158

theorem sum_of_five_distinct_integers_product_2022 :
  ∃ (a b c d e : ℤ), 
    a * b * c * d * e = 2022 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧ 
    (a + b + c + d + e = 342 ∨
     a + b + c + d + e = 338 ∨
     a + b + c + d + e = 336 ∨
     a + b + c + d + e = -332) :=
by 
  sorry

end sum_of_five_distinct_integers_product_2022_l228_228158


namespace solve_inequality_l228_228295

theorem solve_inequality (x : ℝ) : 
  (x / (x^2 + x - 6) ≥ 0) ↔ (x < -3) ∨ (x = 0) ∨ (0 < x ∧ x < 2) :=
by 
  sorry 

end solve_inequality_l228_228295


namespace isosceles_triangle_perimeter_l228_228393

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  ∃ (c : ℝ), (a = b ∧ 7 = c ∨ a = c ∧ 7 = b) ∧ a + b + c = 17 :=
by
  use 17
  sorry

end isosceles_triangle_perimeter_l228_228393


namespace express_in_scientific_notation_l228_228052

def scientific_notation_of_160000 : Prop :=
  160000 = 1.6 * 10^5

theorem express_in_scientific_notation : scientific_notation_of_160000 :=
  sorry

end express_in_scientific_notation_l228_228052


namespace batsman_average_excluding_highest_and_lowest_l228_228586

theorem batsman_average_excluding_highest_and_lowest (average : ℝ) (innings : ℕ) (highest_score : ℝ) (score_difference : ℝ) :
  average = 63 →
  innings = 46 →
  highest_score = 248 →
  score_difference = 150 →
  (average * innings - highest_score - (highest_score - score_difference)) / (innings - 2) = 58 :=
by
  intros h_average h_innings h_highest h_difference
  simp [h_average, h_innings, h_highest, h_difference]
  -- Here the detailed steps from the solution would come in to verify the simplification
  sorry

end batsman_average_excluding_highest_and_lowest_l228_228586


namespace sufficient_not_necessary_l228_228524

variable (x : ℝ)

theorem sufficient_not_necessary (h : x^2 - 3 * x + 2 > 0) : x > 2 → (∀ x : ℝ, x^2 - 3 * x + 2 > 0 ↔ x > 2 ∨ x < -1) :=
by
  sorry

end sufficient_not_necessary_l228_228524


namespace find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l228_228369

variables {m : ℝ} 
def point_on_y_axis (P : (ℝ × ℝ)) := P = (0, -3)
def point_distance_to_y_axis (P : (ℝ × ℝ)) := P = (6, 0) ∨ P = (-6, -6)
def point_in_third_quadrant_and_equidistant (P : (ℝ × ℝ)) := P = (-6, -6)

theorem find_coords_of_P_cond1 (P : ℝ × ℝ) (h : 2 * m + 4 = 0) : point_on_y_axis P ↔ P = (0, -3) :=
by {
  sorry
}

theorem find_coords_of_P_cond2 (P : ℝ × ℝ) (h : abs (2 * m + 4) = 6) : point_distance_to_y_axis P ↔ (P = (6, 0) ∨ P = (-6, -6)) :=
by {
  sorry
}

theorem find_coords_of_P_cond3 (P : ℝ × ℝ) (h1 : 2 * m + 4 < 0) (h2 : m - 1 < 0) (h3 : abs (2 * m + 4) = abs (m - 1)) : point_in_third_quadrant_and_equidistant P ↔ P = (-6, -6) :=
by {
  sorry
}

end find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l228_228369


namespace susan_bought_36_items_l228_228418

noncomputable def cost_per_pencil : ℝ := 0.25
noncomputable def cost_per_pen : ℝ := 0.80
noncomputable def pencils_bought : ℕ := 16
noncomputable def total_spent : ℝ := 20.0

theorem susan_bought_36_items :
  ∃ (pens_bought : ℕ), pens_bought * cost_per_pen + pencils_bought * cost_per_pencil = total_spent ∧ pencils_bought + pens_bought = 36 := 
sorry

end susan_bought_36_items_l228_228418


namespace arithmetic_seq_a4_l228_228530

-- Definition of an arithmetic sequence with the first three terms given.
def arithmetic_seq (a : ℕ → ℕ) :=
  a 0 = 2 ∧ a 1 = 4 ∧ a 2 = 6 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

-- The actual proof goal.
theorem arithmetic_seq_a4 : ∃ a : ℕ → ℕ, arithmetic_seq a ∧ a 3 = 8 :=
by
  sorry

end arithmetic_seq_a4_l228_228530


namespace find_swimming_speed_l228_228851

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end find_swimming_speed_l228_228851


namespace final_price_is_correct_l228_228415

def cost_cucumber : ℝ := 5
def cost_tomato : ℝ := cost_cucumber - 0.2 * cost_cucumber
def cost_bell_pepper : ℝ := cost_cucumber + 0.5 * cost_cucumber
def total_cost_before_discount : ℝ := 2 * cost_tomato + 3 * cost_cucumber + 4 * cost_bell_pepper
def final_price : ℝ := total_cost_before_discount - 0.1 * total_cost_before_discount

theorem final_price_is_correct : final_price = 47.7 := sorry

end final_price_is_correct_l228_228415


namespace hyperbola_problem_l228_228533

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : Prop :=
  c / a = 2 * Real.sqrt 3 / 3

def focal_distance (c a : ℝ) : Prop :=
  2 * a^2 = 3 * c

def point_on_hyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b P.1 P.2

def point_satisfies_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 2

noncomputable def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem hyperbola_problem (a b c : ℝ) (P F1 F2 : ℝ × ℝ) :
  (a > 0 ∧ b > 0) →
  eccentricity a c →
  focal_distance c a →
  point_on_hyperbola P a b →
  point_satisfies_condition P F1 F2 →
  distance F1 F2 = 2 * c →
  (distance P F1) * (distance P F2) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hyperbola_problem_l228_228533


namespace percentage_big_bottles_sold_l228_228880

-- Definitions of conditions
def total_small_bottles : ℕ := 6000
def total_big_bottles : ℕ := 14000
def small_bottles_sold_percentage : ℕ := 20
def total_bottles_remaining : ℕ := 15580

-- Theorem statement
theorem percentage_big_bottles_sold : 
  let small_bottles_sold := (small_bottles_sold_percentage * total_small_bottles) / 100
  let small_bottles_remaining := total_small_bottles - small_bottles_sold
  let big_bottles_remaining := total_bottles_remaining - small_bottles_remaining
  let big_bottles_sold := total_big_bottles - big_bottles_remaining
  (100 * big_bottles_sold) / total_big_bottles = 23 := 
by
  sorry

end percentage_big_bottles_sold_l228_228880


namespace john_school_year_hours_l228_228822

theorem john_school_year_hours (summer_earnings : ℝ) (summer_hours_per_week : ℝ) (summer_weeks : ℝ) (target_school_earnings : ℝ) (school_weeks : ℝ) :
  summer_earnings = 4000 → summer_hours_per_week = 40 → summer_weeks = 8 → target_school_earnings = 5000 → school_weeks = 25 →
  (target_school_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_weeks) = 16 :=
by
  sorry

end john_school_year_hours_l228_228822


namespace angles_arithmetic_progression_l228_228452

theorem angles_arithmetic_progression (A B C : ℝ) (h_sum : A + B + C = 180) :
  (B = 60) ↔ (A + C = 2 * B) :=
by
  sorry

end angles_arithmetic_progression_l228_228452


namespace MikeSalaryNow_l228_228911

-- Definitions based on conditions
def FredSalary  := 1000   -- Fred's salary five months ago
def MikeSalaryFiveMonthsAgo := 10 * FredSalary  -- Mike's salary five months ago
def SalaryIncreasePercent := 40 / 100  -- 40 percent salary increase
def SalaryIncrease := SalaryIncreasePercent * MikeSalaryFiveMonthsAgo  -- Increase in Mike's salary

-- Statement to be proved
theorem MikeSalaryNow : MikeSalaryFiveMonthsAgo + SalaryIncrease = 14000 :=
by
  -- Proof is skipped
  sorry

end MikeSalaryNow_l228_228911


namespace negative_integer_solution_l228_228437

theorem negative_integer_solution (N : ℤ) (hN : N^2 + N = -12) : N = -3 ∨ N = -4 :=
sorry

end negative_integer_solution_l228_228437


namespace simplify_expression_l228_228293

theorem simplify_expression (z y : ℝ) :
  (4 - 5 * z + 2 * y) - (6 + 7 * z - 3 * y) = -2 - 12 * z + 5 * y :=
by
  sorry

end simplify_expression_l228_228293


namespace smallest_pos_multiple_6_15_is_30_l228_228502

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l228_228502


namespace right_triangle_ratio_l228_228390

theorem right_triangle_ratio (a b c r s : ℝ) (h_right_angle : (a:ℝ)^2 + (b:ℝ)^2 = c^2)
  (h_perpendicular : ∀ h : ℝ, c = r + s)
  (h_ratio_ab : a / b = 2 / 5)
  (h_geometry_r : r = a^2 / c)
  (h_geometry_s : s = b^2 / c) :
  r / s = 4 / 25 :=
sorry

end right_triangle_ratio_l228_228390


namespace range_of_abscissa_of_P_l228_228673

noncomputable def line (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

noncomputable def circle (M : ℝ × ℝ) : Prop := (M.1 - 2) ^ 2 + (M.2 - 1) ^ 2 = 1

noncomputable def condition_on_P (P : ℝ × ℝ) : Prop :=
  line P ∧ ∃ M N : ℝ × ℝ, circle M ∧ circle N ∧ angle_eq60 M P N

theorem range_of_abscissa_of_P :
  ∃ x : set ℝ, ∀ P : ℝ × ℝ, condition_on_P P → P.1 ∈ set.range x → P.1 ∈ set.Icc 0 2 :=
sorry

end range_of_abscissa_of_P_l228_228673


namespace g_seven_l228_228536

def g (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem g_seven : g 7 = 17 / 23 := by
  sorry

end g_seven_l228_228536


namespace op_correct_l228_228597

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end op_correct_l228_228597


namespace sector_angle_solution_l228_228675

theorem sector_angle_solution (R α : ℝ) (h1 : 2 * R + α * R = 6) (h2 : (1/2) * R^2 * α = 2) : α = 1 ∨ α = 4 := 
sorry

end sector_angle_solution_l228_228675


namespace prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l228_228693

section ParkingProblem

variable (P_A_more_1_no_more_2 : ℚ) (P_A_more_than_14 : ℚ)

theorem prob_A_fee_exactly_6_yuan :
  (P_A_more_1_no_more_2 = 1/3) →
  (P_A_more_than_14 = 5/12) →
  (1 - (P_A_more_1_no_more_2 + P_A_more_than_14)) = 1/4 :=
by
  -- Skipping the proof
  intros _ _
  sorry

theorem prob_sum_fees_A_B_36_yuan :
  (1/4 : ℚ) = 1/4 :=
by
  -- Skipping the proof
  exact rfl

end ParkingProblem

end prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l228_228693


namespace compute_expression_l228_228208

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l228_228208


namespace initial_cards_l228_228062

theorem initial_cards (taken left initial : ℕ) (h1 : taken = 59) (h2 : left = 17) (h3 : initial = left + taken) : initial = 76 :=
by
  sorry

end initial_cards_l228_228062


namespace units_digit_sum_squares_of_odd_integers_l228_228324

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end units_digit_sum_squares_of_odd_integers_l228_228324


namespace cost_of_four_dozen_bananas_l228_228355

/-- Given that five dozen bananas cost $24.00,
    prove that the cost for four dozen bananas is $19.20. -/
theorem cost_of_four_dozen_bananas 
  (cost_five_dozen: ℝ)
  (rate: cost_five_dozen = 24) : 
  ∃ (cost_four_dozen: ℝ), cost_four_dozen = 19.2 := by
  sorry

end cost_of_four_dozen_bananas_l228_228355


namespace integral_root_of_equation_l228_228214

theorem integral_root_of_equation : 
  ∀ x : ℤ, (x - 8 / (x - 4)) = 2 - 8 / (x - 4) ↔ x = 2 := 
sorry

end integral_root_of_equation_l228_228214


namespace smallest_common_multiple_l228_228514

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l228_228514


namespace leak_rate_l228_228798

-- Definitions based on conditions
def initialWater : ℕ := 10   -- 10 cups
def finalWater : ℕ := 2      -- 2 cups
def firstThreeMilesWater : ℕ := 3 * 1    -- 1 cup per mile for first 3 miles
def lastMileWater : ℕ := 3               -- 3 cups during the last mile
def hikeDuration : ℕ := 2    -- 2 hours

-- Proving the leak rate
theorem leak_rate (drunkWater : ℕ) (leakedWater : ℕ) (leakRate : ℕ) :
  drunkWater = firstThreeMilesWater + lastMileWater ∧ 
  (initialWater - finalWater) = (drunkWater + leakedWater) ∧
  hikeDuration = 2 ∧ 
  leakRate = leakedWater / hikeDuration → leakRate = 1 :=
by
  intros h
  sorry

end leak_rate_l228_228798


namespace fraction_increase_l228_228266

theorem fraction_increase (m n a : ℕ) (h1 : m > n) (h2 : a > 0) : 
  (n : ℚ) / m < (n + a : ℚ) / (m + a) :=
by
  sorry

end fraction_increase_l228_228266


namespace gain_percent_is_80_l228_228740

noncomputable def cost_price : ℝ := 600
noncomputable def selling_price : ℝ := 1080
noncomputable def gain : ℝ := selling_price - cost_price
noncomputable def gain_percent : ℝ := (gain / cost_price) * 100

theorem gain_percent_is_80 :
  gain_percent = 80 := by
  sorry

end gain_percent_is_80_l228_228740


namespace smallest_solution_l228_228909

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l228_228909


namespace geometric_sequence_sufficient_condition_l228_228787

theorem geometric_sequence_sufficient_condition 
  (a_1 : ℝ) (q : ℝ) (h_a1 : a_1 < 0) (h_q : 0 < q ∧ q < 1) :
  ∀ n : ℕ, n > 0 -> a_1 * q^(n-1) < a_1 * q^n :=
sorry

end geometric_sequence_sufficient_condition_l228_228787


namespace find_rate_percent_l228_228608

def P : ℝ := 800
def SI : ℝ := 200
def T : ℝ := 4

theorem find_rate_percent (R : ℝ) :
  SI = P * R * T / 100 → R = 6.25 :=
by
  sorry

end find_rate_percent_l228_228608


namespace valid_triangles_from_10_points_l228_228002

noncomputable def number_of_valid_triangles (n : ℕ) (h : n = 10) : ℕ :=
  if n = 10 then 100 else 0

theorem valid_triangles_from_10_points :
  number_of_valid_triangles 10 rfl = 100 := 
sorry

end valid_triangles_from_10_points_l228_228002


namespace music_stand_cost_proof_l228_228816

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end music_stand_cost_proof_l228_228816


namespace sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l228_228863

theorem sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100 : 
  (15^25 + 5^25) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l228_228863


namespace solve_for_x_l228_228576

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l228_228576


namespace catch_up_time_l228_228108

theorem catch_up_time (x : ℕ) : 240 * x = 150 * x + 12 * 150 := by
  sorry

end catch_up_time_l228_228108


namespace problem1_problem2_l228_228240

-- Definitions for first problem
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem for first problem
theorem problem1 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : ∀ x, -3 ≤ x → x ≤ 3) (h : f (m + 1) > f (2 * m - 1)) :
  -1 ≤ m ∧ m < 2 :=
sorry

-- Definitions for second problem
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem for second problem
theorem problem2 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : odd_function f) (h3 : f 2 = 1) (h4 : ∀ x, -3 ≤ x → x ≤ 3) :
  ∀ x, f (x + 1) + 1 > 0 ↔ -3 < x ∧ x ≤ 2 :=
sorry

end problem1_problem2_l228_228240


namespace min_value_of_expression_l228_228917

/-- 
Given α and β are the two real roots of the quadratic equation x^2 - 2a * x + a + 6 = 0,
prove that the minimum value of (α - 1)^2 + (β - 1)^2 is 8.
-/
theorem min_value_of_expression (a α β : ℝ) (h1 : α ^ 2 - 2 * a * α + a + 6 = 0) (h2 : β ^ 2 - 2 * a * β + a + 6 = 0) :
  (α - 1)^2 + (β - 1)^2 ≥ 8 := 
sorry

end min_value_of_expression_l228_228917


namespace right_triangle_sides_l228_228332

theorem right_triangle_sides :
  (4^2 + 5^2 ≠ 6^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (6^2 + 8^2 ≠ 11^2) ∧
  (5^2 + 12^2 ≠ 23^2) :=
by
  repeat { sorry }

end right_triangle_sides_l228_228332


namespace Cody_reads_books_in_7_weeks_l228_228485

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end Cody_reads_books_in_7_weeks_l228_228485


namespace max_point_of_f_l228_228529

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Define the first derivative of the function
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 12

-- Define the second derivative of the function
def f_double_prime (x : ℝ) : ℝ := 6 * x

-- Prove that a = -2 is the maximum value point of f(x)
theorem max_point_of_f : ∃ a : ℝ, (f_prime a = 0) ∧ (f_double_prime a < 0) ∧ (a = -2) :=
sorry

end max_point_of_f_l228_228529


namespace aerith_seat_l228_228893

-- Let the seats be numbered 1 through 8
-- Assigned seats for Aerith, Bob, Chebyshev, Descartes, Euler, Fermat, Gauss, and Hilbert
variables (a b c d e f g h : ℕ)

-- Define the conditions described in the problem
axiom Bob_assigned : b = 1
axiom Chebyshev_assigned : c = g + 2
axiom Descartes_assigned : d = f - 1
axiom Euler_assigned : e = h - 4
axiom Fermat_assigned : f = d + 5
axiom Gauss_assigned : g = e + 1
axiom Hilbert_assigned : h = a - 3

-- Provide the proof statement to find whose seat Aerith sits
theorem aerith_seat : a = c := sorry

end aerith_seat_l228_228893


namespace solve_for_x_l228_228579

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l228_228579


namespace parallel_lines_a_eq_3_l228_228036

theorem parallel_lines_a_eq_3
  (a : ℝ)
  (l1 : a^2 * x - y + a^2 - 3 * a = 0)
  (l2 : (4 * a - 3) * x - y - 2 = 0)
  (h : ∀ x y, a^2 * x - y + a^2 - 3 * a = (4 * a - 3) * x - y - 2) :
  a = 3 :=
by
  sorry

end parallel_lines_a_eq_3_l228_228036


namespace largest_sum_is_8_over_15_l228_228884

theorem largest_sum_is_8_over_15 :
  max ((1 / 3) + (1 / 6)) (max ((1 / 3) + (1 / 7)) (max ((1 / 3) + (1 / 5)) (max ((1 / 3) + (1 / 9)) ((1 / 3) + (1 / 8))))) = 8 / 15 :=
sorry

end largest_sum_is_8_over_15_l228_228884


namespace petya_prevents_vasya_l228_228567

-- Define the nature of fractions and the players' turns
def is_natural_sum (fractions : List ℚ) : Prop :=
  (fractions.sum = ⌊fractions.sum⌋)

def petya_vasya_game_prevent (fractions : List ℚ) : Prop :=
  ∀ k : ℕ, ∀ additional_fractions : List ℚ, 
  (additional_fractions.length = k) →
  ¬ is_natural_sum (fractions ++ additional_fractions)

theorem petya_prevents_vasya : ∀ fractions : List ℚ, petya_vasya_game_prevent fractions :=
by
  sorry

end petya_prevents_vasya_l228_228567


namespace three_students_received_A_l228_228758

variables (A B C E D : Prop)
variables (h1 : A → B) (h2 : B → C) (h3 : C → E) (h4 : E → D)

theorem three_students_received_A :
  (A ∨ ¬A) ∧ (B ∨ ¬B) ∧ (C ∨ ¬C) ∧ (E ∨ ¬E) ∧ (D ∨ ¬D) ∧ (¬A ∧ ¬B) → (C ∧ E ∧ D) ∧ ¬A ∧ ¬B :=
by sorry

end three_students_received_A_l228_228758


namespace total_surface_area_of_cube_l228_228161

theorem total_surface_area_of_cube : 
  ∀ (s : Real), 
  (12 * s = 36) → 
  (s * Real.sqrt 3 = 3 * Real.sqrt 3) → 
  6 * s^2 = 54 := 
by
  intros s h1 h2
  sorry

end total_surface_area_of_cube_l228_228161


namespace smallest_solution_l228_228908

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l228_228908


namespace percentage_more_than_l228_228979

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end percentage_more_than_l228_228979


namespace no_increasing_sequence_with_unique_sum_l228_228656

theorem no_increasing_sequence_with_unique_sum :
  ¬ (∃ (a : ℕ → ℕ), (∀ n, 0 < a n) ∧ (∀ n, a n < a (n + 1)) ∧ 
  (∀ N, ∃ k ≥ N, ∀ m ≥ k, 
    (∃! (i j : ℕ), a i + a j = m))) := sorry

end no_increasing_sequence_with_unique_sum_l228_228656


namespace find_pairs_l228_228978

noncomputable def possibleValues (α β : ℝ) : Prop :=
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = -(Real.pi/3) + 2*l*Real.pi) ∨
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = (Real.pi/3) + 2*l*Real.pi)

theorem find_pairs (α β : ℝ) (h1 : Real.sin (α - β) = Real.sin α - Real.sin β)
  (h2 : Real.cos (α - β) = Real.cos α - Real.cos β) :
  possibleValues α β :=
sorry

end find_pairs_l228_228978


namespace box_weight_without_balls_l228_228316

theorem box_weight_without_balls :
  let number_of_balls := 30
  let weight_per_ball := 0.36
  let total_weight_with_balls := 11.26
  let total_weight_of_balls := number_of_balls * weight_per_ball
  let weight_of_box := total_weight_with_balls - total_weight_of_balls
  weight_of_box = 0.46 :=
by 
  sorry

end box_weight_without_balls_l228_228316


namespace proof_problem_l228_228714

noncomputable def problem : ℕ :=
  let p := 588
  let q := 0
  let r := 1
  p + q + r

theorem proof_problem
  (AB : ℝ) (P Q : ℝ) (AP BP PQ : ℝ) (angle_POQ : ℝ) 
  (h1 : AB = 1200)
  (h2 : AP + PQ = BP)
  (h3 : BP - Q = 600)
  (h4 : angle_POQ = 30)
  (h5 : PQ = 500)
  : problem = 589 := by
    sorry

end proof_problem_l228_228714


namespace silvia_shorter_route_l228_228121

theorem silvia_shorter_route :
  let jerry_distance := 3 + 4
  let silvia_distance := Real.sqrt (3^2 + 4^2)
  let percentage_reduction := ((jerry_distance - silvia_distance) / jerry_distance) * 100
  (28.5 ≤ percentage_reduction ∧ percentage_reduction < 30.5) →
  percentage_reduction = 30 := by
  intro h
  sorry

end silvia_shorter_route_l228_228121


namespace arithmetic_seq_intersection_quadrant_l228_228234

theorem arithmetic_seq_intersection_quadrant
  (a₁ : ℝ := 1)
  (d : ℝ := -1 / 2)
  (n : ℕ)
  (h₁ : ∃ n : ℕ, a₁ + n * d = aₙ ∧ aₙ > -1 ∧ aₙ < 1/8)
  (h₂ : n ∈ {n : ℕ | n = 3 ∨ n = 4}) :
  (aₙ = 0 ∧ n = 3) ∨ (aₙ = 1/2 ∧ n = 4) :=
by
  sorry

end arithmetic_seq_intersection_quadrant_l228_228234


namespace find_second_number_l228_228430

theorem find_second_number 
  (h1 : (20 + 40 + 60) / 3 = (10 + x + 45) / 3 + 5) :
  x = 50 :=
sorry

end find_second_number_l228_228430


namespace solve_for_x_l228_228581

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l228_228581


namespace laura_garden_daisies_l228_228311

/-
Laura's Garden Problem: Given the ratio of daisies to tulips is 3:4,
Laura currently has 32 tulips, and she plans to add 24 more tulips,
prove that Laura will have 42 daisies in total after the addition to
maintain the same ratio.
-/

theorem laura_garden_daisies (daisies tulips add_tulips : ℕ) (ratio_d : ℕ) (ratio_t : ℕ)
    (h1 : ratio_d = 3) (h2 : ratio_t = 4) (h3 : tulips = 32) (h4 : add_tulips = 24)
    (new_tulips : ℕ := tulips + add_tulips) :
  daisies = 42 :=
by
  sorry

end laura_garden_daisies_l228_228311


namespace marcy_total_spears_l228_228972

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end marcy_total_spears_l228_228972


namespace smallest_multiple_of_6_and_15_l228_228513

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l228_228513


namespace problem1_problem2_l228_228949

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : cos B - 2 * cos A = (2 * a - b) * cos C / c)
variable (h2 : a = 2 * b)

theorem problem1 : a / b = 2 :=
by sorry

theorem problem2 (h3 : A > π / 2) (h4 : c = 3) : 0 < b ∧ b < 3 :=
by sorry

end problem1_problem2_l228_228949


namespace jenna_round_trip_pay_l228_228819

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end jenna_round_trip_pay_l228_228819


namespace solve_for_x_l228_228282

variable (a b x : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (x_pos : x > 0)

theorem solve_for_x : (3 * a) ^ (3 * b) = (a ^ b) * (x ^ b) → x = 27 * a ^ 2 :=
by
  intro h_eq
  sorry

end solve_for_x_l228_228282


namespace tangent_line_eqn_l228_228227

theorem tangent_line_eqn 
  (x y : ℝ)
  (H_curve : y = x^3 + 3 * x^2 - 5)
  (H_point : (x, y) = (-1, -3)) :
  (3 * x + y + 6 = 0) := 
sorry

end tangent_line_eqn_l228_228227


namespace commute_proof_l228_228613

noncomputable def commute_problem : Prop :=
  let d : ℝ := 1.5 -- distance in miles
  let v_w : ℝ := 3 -- walking speed in miles per hour
  let v_t : ℝ := 20 -- train speed in miles per hour
  let walking_minutes : ℝ := (d / v_w) * 60 -- walking time in minutes
  let train_minutes : ℝ := (d / v_t) * 60 -- train time in minutes
  ∃ x : ℝ, walking_minutes = train_minutes + x + 25 ∧ x = 0.5

theorem commute_proof : commute_problem :=
  sorry

end commute_proof_l228_228613


namespace dylans_mom_hotdogs_l228_228939

theorem dylans_mom_hotdogs (hotdogs_total : ℕ) (helens_mom_hotdogs : ℕ) (dylans_mom_hotdogs : ℕ) 
  (h1 : hotdogs_total = 480) (h2 : helens_mom_hotdogs = 101) (h3 : hotdogs_total = helens_mom_hotdogs + dylans_mom_hotdogs) :
dylans_mom_hotdogs = 379 :=
by
  sorry

end dylans_mom_hotdogs_l228_228939


namespace smallest_multiple_l228_228506

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l228_228506


namespace gcd_factorials_l228_228663

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem gcd_factorials (n m : ℕ) (hn : n = 8) (hm : m = 10) :
  Nat.gcd (factorial n) (factorial m) = 40320 := by
  sorry

end gcd_factorials_l228_228663


namespace sum_of_neg_ints_l228_228997

theorem sum_of_neg_ints (xs : List Int) (h₁ : ∀ x ∈ xs, x < 0)
  (h₂ : ∀ x ∈ xs, 3 < |x| ∧ |x| < 6) : xs.sum = -9 :=
sorry

end sum_of_neg_ints_l228_228997


namespace max_value_a_plus_2b_l228_228674

theorem max_value_a_plus_2b {a b : ℝ} (h_positive : 0 < a ∧ 0 < b) (h_eqn : a^2 + 2 * a * b + 4 * b^2 = 6) :
  a + 2 * b ≤ 2 * Real.sqrt 2 :=
sorry

end max_value_a_plus_2b_l228_228674


namespace solve_equation_1_solve_equation_2_l228_228583

theorem solve_equation_1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 + 2 * x - 1 = 0 ↔ x = 1 / 3 ∨ x = -1 :=
by sorry

end solve_equation_1_solve_equation_2_l228_228583


namespace no_fraternity_member_is_club_member_thm_l228_228480

-- Definitions from the conditions
variable (Person : Type)
variable (Club : Person → Prop)
variable (Honest : Person → Prop)
variable (Student : Person → Prop)
variable (Fraternity : Person → Prop)

-- Hypotheses from the problem statements
axiom all_club_members_honest (p : Person) : Club p → Honest p
axiom some_students_not_honest : ∃ p : Person, Student p ∧ ¬ Honest p
axiom no_fraternity_member_is_club_member (p : Person) : Fraternity p → ¬ Club p

-- The theorem to be proven
theorem no_fraternity_member_is_club_member_thm : 
  ∀ p : Person, Fraternity p → ¬ Club p := 
by 
  sorry

end no_fraternity_member_is_club_member_thm_l228_228480


namespace maximum_x_plus_y_l228_228126

theorem maximum_x_plus_y (N x y : ℕ) 
  (hN : N = 19 * x + 95 * y) 
  (hp : ∃ k : ℕ, N = k^2) 
  (hN_le : N ≤ 1995) :
  x + y ≤ 86 :=
sorry

end maximum_x_plus_y_l228_228126


namespace beach_ball_properties_l228_228454

theorem beach_ball_properties :
  let d : ℝ := 18
  let r : ℝ := d / 2
  let surface_area : ℝ := 4 * π * r^2
  let volume : ℝ := (4 / 3) * π * r^3
  surface_area = 324 * π ∧ volume = 972 * π :=
by
  sorry

end beach_ball_properties_l228_228454


namespace pot_filling_time_l228_228010

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end pot_filling_time_l228_228010


namespace prime_divides_product_of_divisors_l228_228000

theorem prime_divides_product_of_divisors (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
(Hp : Nat.Prime p) (Hdiv : p ∣ (Finset.univ.prod a)) : 
∃ i : Fin n, p ∣ a i :=
sorry

end prime_divides_product_of_divisors_l228_228000


namespace total_non_overlapping_area_of_squares_l228_228711

theorem total_non_overlapping_area_of_squares 
  (side_length : ℕ) 
  (num_squares : ℕ)
  (overlapping_areas_count : ℕ)
  (overlapping_width : ℕ)
  (overlapping_height : ℕ)
  (total_area_with_overlap: ℕ)
  (final_missed_patch_ratio: ℕ)
  (final_adjustment: ℕ) 
  (total_area: ℕ :=  total_area_with_overlap-final_missed_patch_ratio ):
  side_length = 2 ∧ 
  num_squares = 4 ∧ 
  overlapping_areas_count = 3 ∧ 
  overlapping_width = 1 ∧ 
  overlapping_height = 2 ∧
  total_area_with_overlap = 16- 3  ∧
  final_missed_patch_ratio = 3-> 
  total_area = 13 := 
 by sorry

end total_non_overlapping_area_of_squares_l228_228711


namespace factorize_expression_l228_228074

theorem factorize_expression (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := 
by 
  sorry

end factorize_expression_l228_228074


namespace energy_stick_difference_l228_228994

variable (B D : ℕ)

theorem energy_stick_difference (h1 : B = D + 17) : 
  let B' := B - 3
  let D' := D + 3
  D' < B' →
  (B' - D') = 11 :=
by
  sorry

end energy_stick_difference_l228_228994


namespace area_of_triangle_PQR_l228_228762

-- Define the vertices P, Q, and R
def P : (Int × Int) := (-3, 2)
def Q : (Int × Int) := (1, 7)
def R : (Int × Int) := (3, -1)

-- Define the formula for the area of a triangle given vertices
def triangle_area (A B C : Int × Int) : Real :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the statement to prove
theorem area_of_triangle_PQR : triangle_area P Q R = 21 := 
  sorry

end area_of_triangle_PQR_l228_228762


namespace range_of_m_l228_228792

theorem range_of_m (m : ℝ) :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + m ≤ 0) → 1 < m := by
  sorry

end range_of_m_l228_228792


namespace volume_of_mixture_removed_replaced_l228_228870

noncomputable def volume_removed (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ): ℝ :=
  (1 - final_concentration / initial_milk) * initial_mixture

theorem volume_of_mixture_removed_replaced (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ) (V: ℝ):
  initial_mixture = 100 →
  initial_milk = 36 →
  final_concentration = 9 →
  V = 50 →
  volume_removed initial_mixture initial_milk final_concentration = V :=
by
  intros h1 h2 h3 h4
  have h5 : initial_mixture = 100 := h1
  have h6 : initial_milk = 36 := h2
  have h7 : final_concentration = 9 := h3
  rw [h5, h6, h7]
  sorry

end volume_of_mixture_removed_replaced_l228_228870


namespace solve_equation_l228_228600

theorem solve_equation :
  ∀ x : ℝ, (4 * x - 2 * x + 1 - 3 = 0) ↔ (x = 1 ∨ x = -1) :=
by
  intro x
  sorry

end solve_equation_l228_228600


namespace union_of_A_and_B_l228_228970

-- Define the sets A and B as given in the problem
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 4}

-- State the theorem to prove that A ∪ B = {0, 1, 2, 4}
theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_of_A_and_B_l228_228970


namespace inequality_proof_l228_228419

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := 
sorry

end inequality_proof_l228_228419


namespace domain_ln_l228_228306

theorem domain_ln (x : ℝ) (h : x - 1 > 0) : x > 1 := 
sorry

end domain_ln_l228_228306


namespace income_increase_is_60_percent_l228_228284

noncomputable def income_percentage_increase 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : ℝ :=
  (M - T) / T * 100

theorem income_increase_is_60_percent 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : 
  income_percentage_increase J T M h1 h2 = 60 :=
by
  sorry

end income_increase_is_60_percent_l228_228284


namespace find_k_l228_228374

theorem find_k (k : ℝ) (h : (-2)^2 - k * (-2) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l228_228374


namespace solve_for_x_l228_228574

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l228_228574


namespace p_satisfies_conditions_l228_228125

noncomputable def p (x : ℕ) : ℕ := sorry

theorem p_satisfies_conditions (h_monic : p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5) : 
  p 6 = 126 := sorry

end p_satisfies_conditions_l228_228125


namespace ae_length_l228_228888

theorem ae_length (AB CD AC AE : ℝ) (h: 2 * AE + 3 * AE = 34): 
  AE = 34 / 5 := by
  -- Proof steps will go here
  sorry

end ae_length_l228_228888


namespace mass_increase_l228_228492

theorem mass_increase (ρ₁ ρ₂ m₁ m₂ a₁ a₂ : ℝ) (cond1 : ρ₂ = 2 * ρ₁) 
                      (cond2 : a₂ = 2 * a₁) (cond3 : m₁ = ρ₁ * (a₁^3)) 
                      (cond4 : m₂ = ρ₂ * (a₂^3)) : 
                      ((m₂ - m₁) / m₁) * 100 = 1500 := by
  sorry

end mass_increase_l228_228492


namespace negative_expression_l228_228057

theorem negative_expression :
  -(-1) ≠ -1 ∧ (-1)^2 ≠ -1 ∧ |(-1)| ≠ -1 ∧ -|(-1)| = -1 :=
by
  sorry

end negative_expression_l228_228057


namespace percent_increase_l228_228612

theorem percent_increase (x : ℝ) (h : (1 / 2) * x = 1) : ((x - (1 / 2)) / (1 / 2)) * 100 = 300 := by
  sorry

end percent_increase_l228_228612


namespace number_of_valid_bases_l228_228150

-- Define the main problem conditions
def base_representation_digits (n b : ℕ) := 
  let digits := (n.to_digits b).length 
  digits

def valid_bases_for_base10_256 (b : ℕ) : Prop := 
  b ≥ 2 ∧ base_representation_digits 256 b = 4

-- Theorem statement
theorem number_of_valid_bases : 
  finset.card (finset.filter valid_bases_for_base10_256 (finset.range (256 + 1))) = 2 := 
sorry

end number_of_valid_bases_l228_228150


namespace find_constants_l228_228779

variable (x : ℝ)

theorem find_constants 
  (h : ∀ x, (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) = 
  (13.5 / (x - 4)) + (-27 / (x - 2)) + (-15 / (x - 2)^3)) :
  true :=
by {
  sorry
}

end find_constants_l228_228779


namespace first_term_arithmetic_sum_l228_228273

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l228_228273


namespace additional_discount_percentage_l228_228135

def initial_price : ℝ := 2000
def gift_cards : ℝ := 200
def initial_discount_rate : ℝ := 0.15
def final_price : ℝ := 1330

theorem additional_discount_percentage :
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  additional_discount_percentage = 11.33 :=
by
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  show additional_discount_percentage = 11.33
  sorry

end additional_discount_percentage_l228_228135


namespace amelia_drove_distance_on_Monday_l228_228996

theorem amelia_drove_distance_on_Monday 
  (total_distance : ℕ) (tuesday_distance : ℕ) (remaining_distance : ℕ)
  (total_distance_eq : total_distance = 8205) 
  (tuesday_distance_eq : tuesday_distance = 582) 
  (remaining_distance_eq : remaining_distance = 6716) :
  ∃ x : ℕ, x + tuesday_distance + remaining_distance = total_distance ∧ x = 907 :=
by
  sorry

end amelia_drove_distance_on_Monday_l228_228996


namespace only_option_A_is_quadratic_l228_228178

def is_quadratic (expr : ℚ[X]) : Prop :=
  ∃ a b c : ℚ, a ≠ 0 ∧ expr = a * X^2 + b * X + c

def option_A := -5 * X^2 - X + 3
def option_B := (3 / X) + X^2 - 1
def option_C (a b c : ℚ) := a * X^2 + b * X + c
def option_D := 4 * X - 1

theorem only_option_A_is_quadratic :
  is_quadratic option_A ∧ 
  ¬ is_quadratic option_B ∧
  ∀ (a b c : ℚ), ¬ is_quadratic (option_C a b c) ∧
  ¬ is_quadratic option_D :=
by
  sorry

end only_option_A_is_quadratic_l228_228178


namespace four_digit_arithmetic_sequence_l228_228748

theorem four_digit_arithmetic_sequence :
  ∃ (a b c d : ℕ), 1000 * a + 100 * b + 10 * c + d = 5555 ∨ 1000 * a + 100 * b + 10 * c + d = 2468 ∧
  (a + d = 10) ∧ (b + c = 10) ∧ (2 * b = a + c) ∧ (c - b = b - a) ∧ (d - c = c - b) ∧
  (1000 * d + 100 * c + 10 * b + a + 1000 * a + 100 * b + 10 * c + d = 11110) :=
sorry

end four_digit_arithmetic_sequence_l228_228748


namespace number_of_tables_l228_228587

-- Define conditions
def chairs_in_base5 : ℕ := 310  -- chairs in base-5
def chairs_base10 : ℕ := 3 * 5^2 + 1 * 5^1 + 0 * 5^0  -- conversion to base-10
def people_per_table : ℕ := 3

-- The theorem to prove
theorem number_of_tables : chairs_base10 / people_per_table = 26 := by
  -- include the automatic proof here
  sorry

end number_of_tables_l228_228587


namespace music_stand_cost_proof_l228_228817

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end music_stand_cost_proof_l228_228817


namespace largest_of_seven_consecutive_integers_l228_228602

theorem largest_of_seven_consecutive_integers (n : ℕ) (h : n > 0) (h_sum : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) = 2222) : (n + 6) = 320 :=
by sorry

end largest_of_seven_consecutive_integers_l228_228602


namespace compute_p2_q2_compute_p3_q3_l228_228559

variables (p q : ℝ)

theorem compute_p2_q2 (h1 : p * q = 15) (h2 : p + q = 8) : p^2 + q^2 = 34 :=
sorry

theorem compute_p3_q3 (h1 : p * q = 15) (h2 : p + q = 8) : p^3 + q^3 = 152 :=
sorry

end compute_p2_q2_compute_p3_q3_l228_228559


namespace boat_travel_distance_upstream_l228_228188

noncomputable def upstream_distance (v : ℝ) : ℝ :=
  let d := 2.5191640969412834 * (v + 3)
  d

theorem boat_travel_distance_upstream :
  ∀ v : ℝ, 
  (∀ D : ℝ, D / (v + 3) = 2.5191640969412834 → D / (v - 3) = D / (v + 3) + 0.5) → 
  upstream_distance 33.2299691632954 = 91.25 :=
by
  sorry

end boat_travel_distance_upstream_l228_228188


namespace laura_rental_cost_l228_228963

def rental_cost_per_day : ℝ := 30
def driving_cost_per_mile : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 300

theorem laura_rental_cost : rental_cost_per_day * days_rented + driving_cost_per_mile * miles_driven = 165 := by
  sorry

end laura_rental_cost_l228_228963


namespace abigail_fence_building_l228_228881

theorem abigail_fence_building :
  ∀ (initial_fences : Nat) (time_per_fence : Nat) (hours_building : Nat) (minutes_per_hour : Nat),
    initial_fences = 10 →
    time_per_fence = 30 →
    hours_building = 8 →
    minutes_per_hour = 60 →
    initial_fences + (minutes_per_hour / time_per_fence) * hours_building = 26 :=
by
  intros initial_fences time_per_fence hours_building minutes_per_hour
  sorry

end abigail_fence_building_l228_228881


namespace seafood_regular_price_l228_228045

theorem seafood_regular_price (y : ℝ) (h : y / 4 = 4) : 2 * y = 32 := by
  sorry

end seafood_regular_price_l228_228045


namespace embankment_construction_l228_228551

theorem embankment_construction :
  (∃ r : ℚ, 0 < r ∧ (1 / 2 = 60 * r * 3)) →
  (∃ t : ℕ, 1 = 45 * 1 / 360 * t) :=
by
  sorry

end embankment_construction_l228_228551


namespace largest_number_in_ratio_l228_228861

theorem largest_number_in_ratio (x : ℕ) (h : ((4 * x + 5 * x + 6 * x) / 3 : ℝ) = 20) : 6 * x = 24 := 
by 
  sorry

end largest_number_in_ratio_l228_228861


namespace log_stack_total_l228_228349

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end log_stack_total_l228_228349


namespace tan_theta_value_l228_228372

noncomputable def tan_theta (θ : ℝ) : ℝ :=
  if (0 < θ) ∧ (θ < 2 * Real.pi) ∧ (Real.cos (θ / 2) = 1 / 3) then
    (2 * (2 * Real.sqrt 2) / (1 - (2 * Real.sqrt 2) ^ 2))
  else
    0 -- added default value for well-definedness

theorem tan_theta_value (θ : ℝ) (h₀: 0 < θ) (h₁ : θ < 2 * Real.pi) (h₂ : Real.cos (θ / 2) = 1 / 3) : 
  tan_theta θ = -4 * Real.sqrt 2 / 7 :=
by
  sorry

end tan_theta_value_l228_228372


namespace expected_value_of_sum_of_marbles_l228_228246

-- Definitions corresponding to the conditions
def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def pairs := marbles.powerset.filter (λ s, s.card = 2)

def pair_sum (s : Finset ℕ) := s.sum id

-- Expected value calculation: There are 21 pairs
def total_sum_pairs := pairs.sum pair_sum

def expected_value := (total_sum_pairs : ℚ) / (pairs.card : ℚ)

-- The theorem that must be proven
theorem expected_value_of_sum_of_marbles :
  expected_value = 154 / 21 :=
by
  sorry

end expected_value_of_sum_of_marbles_l228_228246


namespace compute_radii_sum_l228_228823

def points_on_circle (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist A B) * (dist C D) = (dist A C) * (dist B D)

theorem compute_radii_sum :
  ∃ (r1 r2 : ℝ), points_on_circle (0,0) (-1,-1) (5,2) (6,2) r1
               ∧ points_on_circle (0,0) (-1,-1) (34,14) (35,14) r2
               ∧ r1 > 0
               ∧ r2 > 0
               ∧ r1 < r2
               ∧ r1^2 + r2^2 = 1381 :=
by {
  sorry -- proof not required
}

end compute_radii_sum_l228_228823


namespace arithmetic_difference_l228_228155

variable (S : ℕ → ℤ)
variable (n : ℕ)

-- Definitions as conditions from the problem
def is_arithmetic_sum (s : ℕ → ℤ) :=
  ∀ n : ℕ, s n = 2 * n ^ 2 - 5 * n

theorem arithmetic_difference :
  is_arithmetic_sum S →
  S 10 - S 7 = 87 :=
by
  intro h
  sorry

end arithmetic_difference_l228_228155


namespace same_sign_m_minus_n_opposite_sign_m_plus_n_l228_228367

-- Definitions and Conditions
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom abs_m_eq_4 : |m| = 4
axiom abs_n_eq_3 : |n| = 3

-- Part 1: Prove m - n when m and n have the same sign
theorem same_sign_m_minus_n :
  (m > 0 ∧ n > 0) ∨ (m < 0 ∧ n < 0) → (m - n = 1 ∨ m - n = -1) :=
by
  sorry

-- Part 2: Prove m + n when m and n have opposite signs
theorem opposite_sign_m_plus_n :
  (m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = 1 ∨ m + n = -1) :=
by
  sorry

end same_sign_m_minus_n_opposite_sign_m_plus_n_l228_228367


namespace hyperbola_condition_l228_228654

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m + 2)) + (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_condition_l228_228654


namespace find_a12_l228_228921

variable (a : ℕ → ℝ) (q : ℝ)
variable (h1 : ∀ n, a (n + 1) = a n * q)
variable (h2 : abs q > 1)
variable (h3 : a 1 + a 6 = 2)
variable (h4 : a 3 * a 4 = -15)

theorem find_a12 : a 11 = -25 / 3 :=
by sorry

end find_a12_l228_228921


namespace queenie_worked_4_days_l228_228712

-- Conditions
def daily_earning : ℕ := 150
def overtime_rate : ℕ := 5
def overtime_hours : ℕ := 4
def total_pay : ℕ := 770

-- Question
def number_of_days_worked (d : ℕ) : Prop := 
  daily_earning * d + overtime_rate * overtime_hours * d = total_pay

-- Theorem statement
theorem queenie_worked_4_days : ∃ d : ℕ, number_of_days_worked d ∧ d = 4 := 
by 
  use 4
  unfold number_of_days_worked 
  sorry

end queenie_worked_4_days_l228_228712


namespace find_a_l228_228434

-- Definitions
def parabola (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c
def vertex_property (a b c : ℤ) := 
  ∃ x y, x = 2 ∧ y = 5 ∧ y = parabola a b c x
def point_on_parabola (a b c : ℤ) := 
  ∃ x y, x = 1 ∧ y = 2 ∧ y = parabola a b c x

-- The main statement
theorem find_a {a b c : ℤ} (h_vertex : vertex_property a b c) (h_point : point_on_parabola a b c) : a = -3 :=
by {
  sorry
}

end find_a_l228_228434


namespace rectangle_width_solution_l228_228142

noncomputable def solve_rectangle_width (W L w l : ℝ) :=
  L = 2 * W ∧ 3 * w = W ∧ 2 * l = L ∧ 6 * l * w = 5400

theorem rectangle_width_solution (W L w l : ℝ) :
  solve_rectangle_width W L w l → w = 10 * Real.sqrt 3 :=
by
  sorry

end rectangle_width_solution_l228_228142


namespace find_ordered_triple_l228_228968

theorem find_ordered_triple :
  ∃ (a b c : ℝ), a > 2 ∧ b > 2 ∧ c > 2 ∧
    (a + b + c = 30) ∧
    ( (a = 13) ∧ (b = 11) ∧ (c = 6) ) ∧
    ( ( ( (a + 3)^2 / (b + c - 3) ) + ( (b + 5)^2 / (c + a - 5) ) + ( (c + 7)^2 / (a + b - 7) ) = 45 ) ) :=
sorry

end find_ordered_triple_l228_228968


namespace factor_expression_l228_228777

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end factor_expression_l228_228777


namespace tan_15_deg_product_l228_228687

theorem tan_15_deg_product : (1 + Real.tan 15) * (1 + Real.tan 15) = 2.1433 := by
  sorry

end tan_15_deg_product_l228_228687


namespace quadratic_form_m_neg3_l228_228804

theorem quadratic_form_m_neg3
  (m : ℝ)
  (h_exp : m^2 - 7 = 2)
  (h_coef : m ≠ 3) :
  m = -3 := by
  sorry

end quadratic_form_m_neg3_l228_228804


namespace total_work_completed_in_days_l228_228865

-- Define the number of days Amit can complete the work
def amit_days : ℕ := 15

-- Define the number of days Ananthu can complete the work
def ananthu_days : ℕ := 90

-- Define the number of days Amit worked
def amit_work_days : ℕ := 3

-- Calculate the amount of work Amit can do in one day
def amit_work_day_rate : ℚ := 1 / amit_days

-- Calculate the amount of work Ananthu can do in one day
def ananthu_work_day_rate : ℚ := 1 / ananthu_days

-- Calculate the total work completed
theorem total_work_completed_in_days :
  amit_work_days * amit_work_day_rate + (1 - amit_work_days * amit_work_day_rate) / ananthu_work_day_rate = 75 :=
by
  -- Placeholder for the proof
  sorry

end total_work_completed_in_days_l228_228865


namespace pair_solution_l228_228772

theorem pair_solution (a b : ℕ) (h_b_ne_1 : b ≠ 1) :
  (a + 1 ∣ a^3 * b - 1) → (b - 1 ∣ b^3 * a + 1) →
  (a, b) = (0, 0) ∨ (a, b) = (0, 2) ∨ (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3) :=
by
  sorry

end pair_solution_l228_228772


namespace polynomial_divisibility_l228_228423

theorem polynomial_divisibility (a : ℤ) (n : ℕ) (h_pos : 0 < n) : 
  (a ^ (2 * n + 1) + (a - 1) ^ (n + 2)) % (a ^ 2 - a + 1) = 0 :=
sorry

end polynomial_divisibility_l228_228423


namespace mn_minus_7_is_negative_one_l228_228538

def opp (x : Int) : Int := -x
def largest_negative_integer : Int := -1
def m := opp (-6)
def n := opp largest_negative_integer

theorem mn_minus_7_is_negative_one : m * n - 7 = -1 := by
  sorry

end mn_minus_7_is_negative_one_l228_228538


namespace water_percentage_l228_228042

theorem water_percentage (P : ℕ) : 
  let initial_volume := 300
  let final_volume := initial_volume + 100
  let desired_water_percentage := 70
  let water_added := 100
  let final_water_amount := desired_water_percentage * final_volume / 100
  let current_water_amount := P * initial_volume / 100

  current_water_amount + water_added = final_water_amount → 
  P = 60 :=
by sorry

end water_percentage_l228_228042


namespace miles_from_second_friend_to_work_l228_228771
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end miles_from_second_friend_to_work_l228_228771


namespace inequality_proof_l228_228128

theorem inequality_proof (x y z : ℝ) (n : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_sum : x + y + z = 1) :
  (x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n))) ≥ (3^n) / (3^(n+2) - 9) :=
by
  sorry

end inequality_proof_l228_228128


namespace cos_product_identity_l228_228743

noncomputable def L : ℝ := 3.418 * (Real.cos (2 * Real.pi / 31)) *
                               (Real.cos (4 * Real.pi / 31)) *
                               (Real.cos (8 * Real.pi / 31)) *
                               (Real.cos (16 * Real.pi / 31)) *
                               (Real.cos (32 * Real.pi / 31))

theorem cos_product_identity : L = 1 / 32 := by
  sorry

end cos_product_identity_l228_228743


namespace total_surface_area_of_rectangular_solid_is_422_l228_228217

noncomputable def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_prime_edge_length (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

theorem total_surface_area_of_rectangular_solid_is_422 :
  ∃ (a b c : ℕ), is_prime_edge_length a b c ∧ volume a b c = 399 ∧ surface_area a b c = 422 :=
begin
  sorry
end

end total_surface_area_of_rectangular_solid_is_422_l228_228217


namespace third_divisor_is_11_l228_228609

theorem third_divisor_is_11 (n : ℕ) (x : ℕ) : 
  n = 200 ∧ (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % x = 0 ∧ (n - 20) % 60 = 0 → 
  x = 11 :=
by
  sorry

end third_divisor_is_11_l228_228609


namespace find_years_in_future_l228_228436

theorem find_years_in_future 
  (S F : ℕ)
  (h1 : F = 4 * S + 4)
  (h2 : F = 44) :
  ∃ x : ℕ, F + x = 2 * (S + x) + 20 ∧ x = 4 :=
by 
  sorry

end find_years_in_future_l228_228436


namespace length_of_common_chord_l228_228607

-- Problem conditions
variables (r : ℝ) (h : r = 15)

-- Statement to prove
theorem length_of_common_chord : 2 * (r / 2 * Real.sqrt 3) = 15 * Real.sqrt 3 :=
by
  sorry

end length_of_common_chord_l228_228607


namespace unique_solution_conditions_l228_228361

-- Definitions based on the conditions
variables {x y a : ℝ}

def inequality_condition (x y a : ℝ) : Prop := 
  x^2 + y^2 + 2 * x ≤ 1

def equation_condition (x y a : ℝ) : Prop := 
  x - y = -a

-- Main Theorem Statement
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, inequality_condition x y a ∧ equation_condition x y a) ↔ (a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2) :=
sorry

end unique_solution_conditions_l228_228361


namespace find_dividend_l228_228080

-- Conditions
def quotient : ℕ := 4
def divisor : ℕ := 4

-- Dividend computation
def dividend (q d : ℕ) : ℕ := q * d

-- Theorem to prove
theorem find_dividend : dividend quotient divisor = 16 := 
by
  -- Placeholder for the proof, not needed as per instructions
  sorry

end find_dividend_l228_228080


namespace paving_stone_width_l228_228342

theorem paving_stone_width :
  ∀ (L₁ L₂ : ℝ) (n : ℕ) (length width : ℝ), 
    L₁ = 30 → L₂ = 16 → length = 2 → n = 240 →
    (L₁ * L₂ = n * (length * width)) → width = 1 :=
by
  sorry

end paving_stone_width_l228_228342


namespace mixed_number_expression_l228_228497

theorem mixed_number_expression :
  23 * (((1 + 2 / 3: ℚ) + (2 + 1 / 4: ℚ))) / ((1 + 1 / 2: ℚ) + (1 + 1 / 5: ℚ)) = 367 / 108 := by
  sorry

end mixed_number_expression_l228_228497


namespace average_monthly_increase_l228_228626

theorem average_monthly_increase (x : ℝ) (turnover_january turnover_march : ℝ)
  (h_jan : turnover_january = 2)
  (h_mar : turnover_march = 2.88)
  (h_growth : turnover_march = turnover_january * (1 + x) * (1 + x)) :
  x = 0.2 :=
by
  sorry

end average_monthly_increase_l228_228626


namespace ant_population_percentage_l228_228625

theorem ant_population_percentage (R : ℝ) 
  (h1 : 0.45 * R = 46.75) 
  (h2 : R * 0.55 = 46.75) : 
  R = 0.85 := 
by 
  sorry

end ant_population_percentage_l228_228625


namespace coefficient_of_quadratic_polynomial_l228_228229

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_of_quadratic_polynomial (a b c : ℝ) (h : a > 0) :
  |f a b c 1| = 2 ∧ |f a b c 2| = 2 ∧ |f a b c 3| = 2 →
  (a = 4 ∧ b = -16 ∧ c = 14) ∨ (a = 2 ∧ b = -6 ∧ c = 2) ∨ (a = 2 ∧ b = -10 ∧ c = 10) :=
by
  sorry

end coefficient_of_quadratic_polynomial_l228_228229


namespace average_possible_k_l228_228930

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end average_possible_k_l228_228930


namespace product_increase_l228_228112

theorem product_increase (a b c : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
  sorry

end product_increase_l228_228112


namespace price_of_first_metal_l228_228044

theorem price_of_first_metal (x : ℝ) 
  (h1 : (x + 96) / 2 = 82) : 
  x = 68 :=
by sorry

end price_of_first_metal_l228_228044


namespace maximize_x_minus_y_plus_z_l228_228248

-- Define the given condition as a predicate
def given_condition (x y z : ℝ) : Prop :=
  2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5

-- Define the statement we want to prove
theorem maximize_x_minus_y_plus_z :
  ∃ x y z : ℝ, given_condition x y z ∧ (x - y + z = 4) :=
by
  sorry

end maximize_x_minus_y_plus_z_l228_228248


namespace infinite_rational_points_in_region_l228_228228

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
  (∀ p ∈ S, (p.1 ^ 2 + p.2 ^ 2 ≤ 16) ∧ (p.1 ≤ 3) ∧ (p.2 ≤ 3) ∧ (p.1 > 0) ∧ (p.2 > 0)) ∧
  Set.Infinite S :=
sorry

end infinite_rational_points_in_region_l228_228228


namespace each_person_ate_slices_l228_228122

def slices_per_person (num_pizzas : ℕ) (slices_per_pizza : ℕ) (num_people : ℕ) : ℕ :=
  (num_pizzas * slices_per_pizza) / num_people

theorem each_person_ate_slices :
  ∀ (num_pizzas slices_per_pizza num_people : ℕ),
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  num_people = 6 →
  slices_per_person num_pizzas slices_per_pizza num_people = 4 :=
by
  intros num_pizzas slices_per_pizza num_people hpizzas hslices hpeople
  rw [hpizzas, hslices, hpeople]
  simp [slices_per_person]
  sorry

end each_person_ate_slices_l228_228122


namespace beaver_group_l228_228300

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end beaver_group_l228_228300


namespace closest_to_sin_2016_deg_is_neg_half_l228_228661

/-- Given the value of \( \sin 2016^\circ \), show that the closest number from the given options is \( -\frac{1}{2} \).
Options:
A: \( \frac{11}{2} \)
B: \( -\frac{1}{2} \)
C: \( \frac{\sqrt{2}}{2} \)
D: \( -1 \)
-/
theorem closest_to_sin_2016_deg_is_neg_half :
  let sin_2016 := Real.sin (2016 * Real.pi / 180)
  |sin_2016 - (-1 / 2)| < |sin_2016 - 11 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - Real.sqrt 2 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - (-1)| :=
by
  sorry

end closest_to_sin_2016_deg_is_neg_half_l228_228661


namespace scientific_notation_41600_l228_228477

theorem scientific_notation_41600 : (4.16 * 10^4) = 41600 := by
  sorry

end scientific_notation_41600_l228_228477


namespace real_z9_count_l228_228162

theorem real_z9_count (z : ℂ) (hz : z^18 = 1) : 
  (∃! z : ℂ, z^18 = 1 ∧ (z^9).im = 0) :=
sorry

end real_z9_count_l228_228162


namespace friend_owns_10_bikes_l228_228388

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end friend_owns_10_bikes_l228_228388


namespace prime_bound_l228_228127

-- The definition for the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry  -- placeholder for the primorial definition

-- The main theorem to prove
theorem prime_bound (n : ℕ) : nth_prime n ≤ 2 ^ 2 ^ (n - 1) := sorry

end prime_bound_l228_228127


namespace find_number_l228_228518

theorem find_number : ∃ n : ℕ, (∃ x : ℕ, x / 15 = 4 ∧ x^2 = n) ∧ n = 3600 := 
by
  sorry

end find_number_l228_228518


namespace inscribed_sphere_surface_area_l228_228666

theorem inscribed_sphere_surface_area (V S : ℝ) (hV : V = 2) (hS : S = 3) : 4 * Real.pi * (3 * V / S)^2 = 16 * Real.pi := by
  sorry

end inscribed_sphere_surface_area_l228_228666


namespace no_intersection_l228_228799

def f₁ (x : ℝ) : ℝ := abs (3 * x + 6)
def f₂ (x : ℝ) : ℝ := -abs (4 * x - 1)

theorem no_intersection : ∀ x, f₁ x ≠ f₂ x :=
by
  sorry

end no_intersection_l228_228799


namespace problem1_problem2_l228_228237

variable {α : ℝ}

-- Given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 3

-- Proof statements to be shown
theorem problem1 (h : tan_alpha α) : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by sorry

theorem problem2 (h : tan_alpha α) : Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6 :=
by sorry

end problem1_problem2_l228_228237


namespace find_a8_l228_228676

-- Define the arithmetic sequence and the given conditions
variable {α : Type} [AddCommGroup α] [MulAction ℤ α]

def is_arithmetic_sequence (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ}
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 5 + a 6 = 22
axiom h3 : a 3 = 7

theorem find_a8 : a 8 = 15 :=
by
  -- Proof omitted
  sorry

end find_a8_l228_228676


namespace solve_for_x_l228_228215

theorem solve_for_x (x : ℤ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 9) : x = 72 / 23 :=
by
  sorry

end solve_for_x_l228_228215


namespace sum_of_integers_eq_l228_228855

-- We define the conditions
variables (x y : ℕ)
-- The conditions specified in the problem
def diff_condition : Prop := x - y = 16
def prod_condition : Prop := x * y = 63

-- The theorem stating that given the conditions, the sum is 2*sqrt(127)
theorem sum_of_integers_eq : diff_condition x y → prod_condition x y → x + y = 2 * Real.sqrt 127 :=
by
  sorry

end sum_of_integers_eq_l228_228855


namespace factor_polynomial_l228_228677

variable {R : Type*} [CommRing R]

theorem factor_polynomial (a b c : R) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + ab + bc + ac)) :=
by
  sorry

end factor_polynomial_l228_228677


namespace kellys_apples_l228_228703

def apples_kelly_needs_to_pick := 49
def total_apples := 105

theorem kellys_apples :
  ∃ x : ℕ, x + apples_kelly_needs_to_pick = total_apples ∧ x = 56 :=
sorry

end kellys_apples_l228_228703


namespace Jim_catches_Bob_in_20_minutes_l228_228206

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end Jim_catches_Bob_in_20_minutes_l228_228206


namespace smallest_positive_real_is_131_div_11_l228_228901

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l228_228901


namespace exists_nonneg_poly_div_l228_228662

theorem exists_nonneg_poly_div (P : Polynomial ℝ) 
  (hP_pos : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ n, Q.coeff n ≥ 0) ∧ (∀ n, R.coeff n ≥ 0) ∧ (P = Q / R) := 
sorry

end exists_nonneg_poly_div_l228_228662


namespace probability_sin_in_interval_half_l228_228634

noncomputable def probability_sin_interval : ℝ :=
  let a := - (Real.pi / 2)
  let b := Real.pi / 2
  let interval_length := b - a
  (b - 0) / interval_length

theorem probability_sin_in_interval_half :
  probability_sin_interval = 1 / 2 := by
  sorry

end probability_sin_in_interval_half_l228_228634


namespace range_of_a_l228_228617

theorem range_of_a (a : ℝ) : (-1/Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1/Real.exp 1) :=
  sorry

end range_of_a_l228_228617


namespace find_arithmetic_mean_l228_228003

theorem find_arithmetic_mean (σ μ : ℝ) (hσ : σ = 1.5) (h : 11 = μ - 2 * σ) : μ = 14 :=
by
  sorry

end find_arithmetic_mean_l228_228003


namespace danny_watermelon_slices_l228_228766

theorem danny_watermelon_slices : 
  ∀ (x : ℕ), 3 * x + 15 = 45 -> x = 10 := by
  intros x h
  sorry

end danny_watermelon_slices_l228_228766


namespace square_area_on_parabola_l228_228348

theorem square_area_on_parabola (s : ℝ) (h : 0 < s) (hG : (3 + s)^2 - 6 * (3 + s) + 5 = -2 * s) : 
  (2 * s) * (2 * s) = 24 - 8 * Real.sqrt 5 := 
by 
  sorry

end square_area_on_parabola_l228_228348


namespace sqrt_expression_simplification_l228_228649

theorem sqrt_expression_simplification : 
  (Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5) = Real.sqrt 3 := 
  by
    sorry

end sqrt_expression_simplification_l228_228649


namespace Juanita_weekday_spending_l228_228796

/- Defining the variables and conditions in the problem -/

def Grant_spending : ℝ := 200
def Sunday_spending : ℝ := 2
def extra_spending : ℝ := 60

-- We need to prove that Juanita spends $0.50 per day from Monday through Saturday on newspapers.

theorem Juanita_weekday_spending :
  (∃ x : ℝ, 6 * 52 * x + 52 * 2 = Grant_spending + extra_spending) -> (∃ x : ℝ, x = 0.5) := by {
  sorry
}

end Juanita_weekday_spending_l228_228796


namespace ceil_of_fractional_square_l228_228894

theorem ceil_of_fractional_square :
  (Int.ceil ((- (7/4) + 1/4) ^ 2) = 3) :=
by
  sorry

end ceil_of_fractional_square_l228_228894


namespace similar_triangle_leg_l228_228635

theorem similar_triangle_leg (x : Real) : 
  (12 / x = 9 / 7) → x = 84 / 9 := by
  intro h
  sorry

end similar_triangle_leg_l228_228635


namespace sin_14pi_over_5_eq_sin_36_degree_l228_228075

noncomputable def sin_14pi_over_5 : ℝ :=
  Real.sin (14 * Real.pi / 5)

noncomputable def sin_36_degree : ℝ :=
  Real.sin (36 * Real.pi / 180)

theorem sin_14pi_over_5_eq_sin_36_degree :
  sin_14pi_over_5 = sin_36_degree :=
sorry

end sin_14pi_over_5_eq_sin_36_degree_l228_228075


namespace find_a7_l228_228491

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l228_228491


namespace expected_composite_selection_l228_228571

noncomputable def expected_composite_count : ℚ :=
  let total_numbers := 100
  let composite_numbers := 74
  let p := (composite_numbers : ℚ) / total_numbers
  let n := 5
  n * p

theorem expected_composite_selection :
  expected_composite_count = 37 / 10 := by
  sorry

end expected_composite_selection_l228_228571


namespace market_survey_l228_228102

theorem market_survey (X Y : ℕ) (h1 : X / Y = 9) (h2 : X + Y = 400) : X = 360 :=
by
  sorry

end market_survey_l228_228102


namespace journal_sessions_per_week_l228_228601

/-- Given that each student writes 4 pages in each session and will write 72 journal pages in 6 weeks, prove that there are 3 journal-writing sessions per week.
--/
theorem journal_sessions_per_week (pages_per_session : ℕ) (total_pages : ℕ) (weeks : ℕ) (sessions_per_week : ℕ) :
  pages_per_session = 4 →
  total_pages = 72 →
  weeks = 6 →
  total_pages = pages_per_session * sessions_per_week * weeks →
  sessions_per_week = 3 :=
by
  intros h1 h2 h3 h4
  sorry

end journal_sessions_per_week_l228_228601


namespace good_numbers_l228_228751

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_odd_prime (n : ℕ) : Prop :=
  Prime n ∧ n % 2 = 1

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_divisor (d + 1) (n + 1)

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ n = 1 ∨ is_odd_prime n :=
sorry

end good_numbers_l228_228751


namespace medal_award_count_l228_228107

theorem medal_award_count :
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  no_canadians_get_medals + one_canadian_gets_medal = 480 :=
by
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  show no_canadians_get_medals + one_canadian_gets_medal = 480
  -- here should be the steps skipped
  sorry

end medal_award_count_l228_228107


namespace original_price_sarees_l228_228599

theorem original_price_sarees
  (P : ℝ)
  (h : 0.90 * 0.85 * P = 378.675) :
  P = 495 :=
sorry

end original_price_sarees_l228_228599


namespace exists_xy_nat_divisible_l228_228420

theorem exists_xy_nat_divisible (n : ℕ) : ∃ x y : ℤ, (x^2 + y^2 - 2018) % n = 0 :=
by
  use 43, 13
  sorry

end exists_xy_nat_divisible_l228_228420


namespace sum_of_coefficients_l228_228146

theorem sum_of_coefficients (d : ℤ) (h : d ≠ 0) :
    let a := 3 + 2
    let b := 17 + 2
    let c := 10 + 5
    let e := 16 + 4
    a + b + c + e = 59 :=
by
  let a := 3 + 2
  let b := 17 + 2
  let c := 10 + 5
  let e := 16 + 4
  sorry

end sum_of_coefficients_l228_228146


namespace graph_representation_l228_228735

theorem graph_representation {x y : ℝ} (h : x^2 * (x - y - 2) = y^2 * (x - y - 2)) :
  ( ∃ a : ℝ, ∀ (x : ℝ), y = a * x ) ∨ 
  ( ∃ b : ℝ, ∀ (x : ℝ), y = b * x ) ∨ 
  ( ∃ c : ℝ, ∀ (x : ℝ), y = x - 2 ) ∧ 
  (¬ ∃ d : ℝ, ∀ (x : ℝ), y = d * x ∧ y = d * x - 2) :=
sorry

end graph_representation_l228_228735


namespace polynomial_transformation_l228_228385

theorem polynomial_transformation :
  ∀ (a h k : ℝ), (8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → a + h + k = -23.5 :=
by
  intros a h k h_eq
  sorry

end polynomial_transformation_l228_228385


namespace smallest_common_multiple_l228_228516

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l228_228516


namespace number_of_k_l228_228081

theorem number_of_k (n : ℕ) (hn : n = 2016) :
  (Finset.card { k : ℕ | k ∈ Finset.range (n + 1) ∧ (k : ZMod (n + 1 + 1) ^ n = 1) }) = Nat.totient n :=
by
  sorry

end number_of_k_l228_228081


namespace probability_x_lt_2y_in_rectangle_l228_228046

theorem probability_x_lt_2y_in_rectangle :
  let rectangle := set.Icc (0, 0) (4, 2)
  let region := {p : ℝ × ℝ | p ∈ rectangle ∧ p.1 < 2 * p.2}
  (measure_theory.measure.region.probability_of_region region rectangle = 1 / 2) :=
by
  sorry

end probability_x_lt_2y_in_rectangle_l228_228046


namespace gift_boxes_in_3_days_l228_228220
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_l228_228220


namespace crows_cannot_be_on_same_tree_l228_228025

theorem crows_cannot_be_on_same_tree :
  (∀ (trees : ℕ) (crows : ℕ),
   trees = 22 ∧ crows = 22 →
   (∀ (positions : ℕ → ℕ),
    (∀ i, 1 ≤ positions i ∧ positions i ≤ 2) →
    ∀ (move : (ℕ → ℕ) → (ℕ → ℕ)),
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     move pos i = pos i + positions (i + 1) ∨ move pos i = pos i - positions (i + 1)) →
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     pos i % trees = (move pos i) % trees) →
    ¬ (∃ (final_pos : ℕ → ℕ),
      (∀ i, final_pos i = 0 ∨ final_pos i = 22) ∧
      (∀ i j, final_pos i = final_pos j)
    )
  )
) :=
sorry

end crows_cannot_be_on_same_tree_l228_228025


namespace solve_for_a_l228_228242

theorem solve_for_a (a x : ℝ) (h1 : 3 * x - 5 = x + a) (h2 : x = 2) : a = -1 :=
by
  sorry

end solve_for_a_l228_228242


namespace original_number_is_3_l228_228431

theorem original_number_is_3 
  (A B C D E : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 8) 
  (h2 : (8 + B + C + D + E) / 5 = 9): 
  A = 3 :=
sorry

end original_number_is_3_l228_228431


namespace jim_catches_bob_in_20_minutes_l228_228204

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end jim_catches_bob_in_20_minutes_l228_228204


namespace complete_square_l228_228328

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l228_228328


namespace find_positive_x_l228_228363

theorem find_positive_x (x : ℝ) (h1 : x * ⌊x⌋ = 72) (h2 : x > 0) : x = 9 :=
by 
  sorry

end find_positive_x_l228_228363


namespace cookie_radius_l228_228585

theorem cookie_radius (x y : ℝ) : x^2 + y^2 + 28 = 6*x + 20*y → ∃ r, r = 9 :=
by
  sorry

end cookie_radius_l228_228585


namespace total_beats_together_in_week_l228_228713

theorem total_beats_together_in_week :
  let samantha_beats_per_min := 250
  let samantha_hours_per_day := 3
  let michael_beats_per_min := 180
  let michael_hours_per_day := 2.5
  let days_per_week := 5

  let samantha_beats_per_day := samantha_beats_per_min * 60 * samantha_hours_per_day
  let samantha_beats_per_week := samantha_beats_per_day * days_per_week
  let michael_beats_per_day := michael_beats_per_min * 60 * michael_hours_per_day
  let michael_beats_per_week := michael_beats_per_day * days_per_week
  let total_beats_per_week := samantha_beats_per_week + michael_beats_per_week

  total_beats_per_week = 360000 := 
by
  -- The proof will go here
  sorry

end total_beats_together_in_week_l228_228713


namespace sequence_third_term_l228_228720

theorem sequence_third_term (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end sequence_third_term_l228_228720


namespace total_amount_245_l228_228051

-- Define the conditions and the problem
theorem total_amount_245 (a : ℝ) (x y z : ℝ) (h1 : y = 0.45 * a) (h2 : z = 0.30 * a) (h3 : y = 63) :
  x + y + z = 245 := 
by
  -- Starting the proof (proof steps are unnecessary as per the procedure)
  sorry

end total_amount_245_l228_228051


namespace lebesgue_decomposition_l228_228825

open MeasureTheory

variables {E : Type*} [measurable_space E]
variable {ν : measure_theory.measure E}
variable {μ : measure_theory.measure E}

-- Define the Lebesgue decomposition theorem
theorem lebesgue_decomposition (μ ν : measure E) [sigma_finite ν] :
  ∃ (f : E → ℝ) 
    (D : set E) 
    (hf : measurable f) 
    (hD : ν D = 0),
    (∀ B ∈ measurable_set, μ B = ∫ x in B, f x ∂ν + μ (B \ D)) ∧
    (∀ (g : E → ℝ) (C : set E) 
      (hg : measurable g) 
      (hC : ν C = 0),
      (∀ B ∈ measurable_set, μ B = ∫ x in B, g x ∂ν + μ (B \ C)) → 
      μ (D \ C ∪ C \ D) = 0 ∧
      ν {x | f x ≠ g x} = 0 ) :=
sorry

end lebesgue_decomposition_l228_228825


namespace felix_trees_chopped_l228_228778

-- Given conditions
def cost_per_sharpening : ℕ := 8
def total_spent : ℕ := 48
def trees_per_sharpening : ℕ := 25

-- Lean statement of the problem
theorem felix_trees_chopped (h : total_spent / cost_per_sharpening * trees_per_sharpening >= 150) : True :=
by {
  -- This is just a placeholder for the proof.
  sorry
}

end felix_trees_chopped_l228_228778


namespace sufficient_but_not_necessary_condition_l228_228789

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (|a - b^2| + |b - a^2| ≤ 1) → ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ 
  ∃ (a b : ℝ), ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ ¬ (|a - b^2| + |b - a^2| ≤ 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l228_228789


namespace exists_k_plus_one_element_subset_l228_228788

theorem exists_k_plus_one_element_subset 
  {X : Type} {n K m α : ℕ} (S : finset (finset (fin X))) 
  (h1 : S.card = m)
  (h2 : ∀ s ∈ S, s.card = K)
  (hm : m > ((k - 1) * (n - k) + k) / k^2 * (n.choose (K-1))) :
  ∃ (T : finset (fin X)), T.card = k + 1 ∧ (∀ s ∈ T.powerset.filter (λ x, x.card = K), s ∈ S) :=
sorry

end exists_k_plus_one_element_subset_l228_228788


namespace average_k_of_polynomial_with_positive_integer_roots_l228_228928

-- Define the conditions and the final theorem

theorem average_k_of_polynomial_with_positive_integer_roots :
  (∑ i in {k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset, i) / 
  ({k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset.card : ℝ) = 15 :=
by
  sorry

end average_k_of_polynomial_with_positive_integer_roots_l228_228928


namespace isabel_pictures_l228_228618

theorem isabel_pictures
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (h_phone_pics : phone_pics = 2)
  (h_camera_pics : camera_pics = 4)
  (h_total_albums : total_albums = 3) :
  (phone_pics + camera_pics) / total_albums = 2 :=
by
  sorry

end isabel_pictures_l228_228618


namespace exists_x_y_divisible_l228_228421

theorem exists_x_y_divisible (n : ℕ) : ∃ (x y : ℤ), n ∣ (x^2 + y^2 - 2018) :=
by {
  use [43, 13],
  simp,
  sorry
}

end exists_x_y_divisible_l228_228421


namespace min_value_of_f_l228_228156

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_of_f : ∃ x : ℝ, f x = x^3 - 3 * x^2 + 1 ∧ (∀ y : ℝ, f y ≥ f 2) :=
by
  sorry

end min_value_of_f_l228_228156


namespace bob_paid_correctly_l228_228883

-- Define the variables involved
def alice_acorns : ℕ := 3600
def price_per_acorn : ℕ := 15
def multiplier : ℕ := 9
def total_amount_alice_paid : ℕ := alice_acorns * price_per_acorn

-- Define Bob's payment amount
def bob_payment : ℕ := total_amount_alice_paid / multiplier

-- The main theorem
theorem bob_paid_correctly : bob_payment = 6000 := by
  sorry

end bob_paid_correctly_l228_228883


namespace necessary_but_not_sufficient_l228_228373

-- Defining the problem in Lean 4 terms.
noncomputable def geom_seq_cond (a : ℕ → ℕ) (m n p q : ℕ) : Prop :=
  m + n = p + q → a m * a n = a p * a q

theorem necessary_but_not_sufficient (a : ℕ → ℕ) (m n p q : ℕ) (h : m + n = p + q) :
  geom_seq_cond a m n p q → ∃ b : ℕ → ℕ, (∀ n, b n = 0 → (m + n = p + q → b m * b n = b p * b q))
    ∧ (∀ n, ¬ (b n = 0 → ∀ q, b (q+1) / b q = b (q+1) / b q)) := sorry

end necessary_but_not_sufficient_l228_228373


namespace num_ordered_triples_l228_228879

/-
Let Q be a right rectangular prism with integral side lengths a, b, and c such that a ≤ b ≤ c, and b = 2023.
A plane parallel to one of the faces of Q cuts Q into two prisms, one of which is similar to Q, and both have nonzero volume.
Prove that the number of ordered triples (a, b, c) such that b = 2023 is 7.
-/

theorem num_ordered_triples (a c : ℕ) (h : a ≤ 2023 ∧ 2023 ≤ c) (ac_eq_2023_squared : a * c = 2023^2) :
  ∃ count, count = 7 :=
by {
  sorry
}

end num_ordered_triples_l228_228879


namespace odd_number_diff_squares_unique_l228_228986

theorem odd_number_diff_squares_unique (n : ℕ) (h : 0 < n) : 
  ∃! (x y : ℤ), (2 * n + 1) = x^2 - y^2 :=
by {
  sorry
}

end odd_number_diff_squares_unique_l228_228986


namespace Barkley_bones_l228_228759

def bones_per_month : ℕ := 10
def months : ℕ := 5
def bones_received : ℕ := bones_per_month * months
def bones_buried : ℕ := 42
def bones_available : ℕ := 8

theorem Barkley_bones :
  bones_received - bones_buried = bones_available := by sorry

end Barkley_bones_l228_228759


namespace inverse_proposition_false_l228_228009

theorem inverse_proposition_false (a b c : ℝ) : 
  ¬ (a > b → ((c ≠ 0) ∧ (a / (c * c)) > (b / (c * c))))
:= 
by 
  -- Outline indicating that the proof will follow from checking cases
  sorry

end inverse_proposition_false_l228_228009


namespace integral_f_equals_neg_third_l228_228539

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * c

theorem integral_f_equals_neg_third :
  (∫ x in (0 : ℝ)..(1 : ℝ), f x (∫ t in (0 : ℝ)..(1 : ℝ), f t (∫ t in (0 : ℝ)..(1 : ℝ), f t 0))) = -1/3 :=
by
  sorry

end integral_f_equals_neg_third_l228_228539


namespace gwendolyn_reading_time_l228_228684

/--
Gwendolyn can read 200 sentences in 1 hour. 
Each paragraph has 10 sentences. 
There are 20 paragraphs per page. 
The book has 50 pages. 
--/
theorem gwendolyn_reading_time : 
  let sentences_per_hour := 200
  let sentences_per_paragraph := 10
  let paragraphs_per_page := 20
  let pages := 50
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  (total_sentences / sentences_per_hour) = 50 := 
by
  let sentences_per_hour : ℕ := 200
  let sentences_per_paragraph : ℕ := 10
  let paragraphs_per_page : ℕ := 20
  let pages : ℕ := 50
  let sentences_per_page : ℕ := sentences_per_paragraph * paragraphs_per_page
  let total_sentences : ℕ := sentences_per_page * pages
  have h : (total_sentences / sentences_per_hour) = 50 := by sorry
  exact h

end gwendolyn_reading_time_l228_228684


namespace tom_trout_count_l228_228412

theorem tom_trout_count (M T : ℕ) (hM : M = 8) (hT : T = 2 * M) : T = 16 :=
by
  -- proof goes here
  sorry

end tom_trout_count_l228_228412


namespace find_x_l228_228696

variable (x : ℝ)
variable (s : ℝ)

-- Conditions as hypothesis
def square_perimeter_60 (s : ℝ) : Prop := 4 * s = 60
def triangle_area_150 (x s : ℝ) : Prop := (1 / 2) * x * s = 150
def height_equals_side (s : ℝ) : Prop := true

-- Proof problem statement
theorem find_x 
  (h1 : square_perimeter_60 s)
  (h2 : triangle_area_150 x s)
  (h3 : height_equals_side s) : 
  x = 20 := 
sorry

end find_x_l228_228696


namespace remainder_of_3x_minus_2y_mod_30_l228_228222

theorem remainder_of_3x_minus_2y_mod_30
  (p q : ℤ) (x y : ℤ)
  (hx : x = 60 * p + 53)
  (hy : y = 45 * q + 28) :
  (3 * x - 2 * y) % 30 = 13 :=
by 
  sorry

end remainder_of_3x_minus_2y_mod_30_l228_228222


namespace arithmetic_sequence_property_l228_228546

-- Define arithmetic sequence and given condition
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Lean 4 statement
theorem arithmetic_sequence_property {a : ℕ → ℝ} (h : arithmetic_sequence a) (h1 : a 6 = 30) : a 3 + a 9 = 60 :=
by
  sorry

end arithmetic_sequence_property_l228_228546


namespace salary_increase_difference_l228_228244

structure Person where
  name : String
  salary : ℕ
  raise_percent : ℕ
  investment_return : ℕ

def hansel := Person.mk "Hansel" 30000 10 5
def gretel := Person.mk "Gretel" 30000 15 4
def rapunzel := Person.mk "Rapunzel" 40000 8 6
def rumpelstiltskin := Person.mk "Rumpelstiltskin" 35000 12 7
def cinderella := Person.mk "Cinderella" 45000 7 8
def jack := Person.mk "Jack" 50000 6 10

def salary_increase (p : Person) : ℕ := p.salary * p.raise_percent / 100
def investment_return (p : Person) : ℕ := salary_increase p * p.investment_return / 100
def total_increase  (p : Person) : ℕ := salary_increase p + investment_return p

def problem_statement : Prop :=
  let hansel_increase := total_increase hansel
  let gretel_increase := total_increase gretel
  let rapunzel_increase := total_increase rapunzel
  let rumpelstiltskin_increase := total_increase rumpelstiltskin
  let cinderella_increase := total_increase cinderella
  let jack_increase := total_increase jack

  let highest_increase := max gretel_increase (max rumpelstiltskin_increase (max cinderella_increase (max rapunzel_increase (max jack_increase hansel_increase))))
  let lowest_increase := min gretel_increase (min rumpelstiltskin_increase (min cinderella_increase (min rapunzel_increase (min jack_increase hansel_increase))))

  highest_increase - lowest_increase = 1530

theorem salary_increase_difference : problem_statement := by
  sorry

end salary_increase_difference_l228_228244


namespace opposite_of_negative_rational_l228_228014

theorem opposite_of_negative_rational : - (-(4/3)) = (4/3) :=
by
  sorry

end opposite_of_negative_rational_l228_228014


namespace solve_for_x_l228_228575

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l228_228575


namespace sequence_sum_l228_228360

theorem sequence_sum (r z w : ℝ) (h1 : 4 * r = 1) (h2 : 256 * r = z) (h3 : z * r = w) : z + w = 80 :=
by
  -- Proceed with your proof here.
  -- sorry for skipping the proof part.
  sorry

end sequence_sum_l228_228360


namespace num_valid_8_tuples_is_1540_l228_228652

def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) := 
  (0 ≤ a1 ∧ a1 ≤ 1) ∧ 
  (0 ≤ a2 ∧ a2 ≤ 2) ∧ 
  (0 ≤ a3 ∧ a3 ≤ 3) ∧
  (0 ≤ a4 ∧ a4 ≤ 4) ∧ 
  a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19

theorem num_valid_8_tuples_is_1540 :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ // valid_8_tuple t.1 t.2.1 t.2.2.1 t.2.2.2.1 t.2.2.2.2.1 t.2.2.2.2.2.1 t.2.2.2.2.2.2.1 t.2.2.2.2.2.2.2 } = 1540 :=
by
  -- Proof goes here
  sorry

end num_valid_8_tuples_is_1540_l228_228652


namespace halfway_between_l228_228891

theorem halfway_between (a b : ℚ) (h₁ : a = 1/8) (h₂ : b = 1/3) : (a + b) / 2 = 11 / 48 := 
by
  sorry

end halfway_between_l228_228891


namespace flagpole_height_in_inches_l228_228624

theorem flagpole_height_in_inches
  (height_lamppost shadow_lamppost : ℚ)
  (height_flagpole shadow_flagpole : ℚ)
  (h₁ : height_lamppost = 50)
  (h₂ : shadow_lamppost = 12)
  (h₃ : shadow_flagpole = 18 / 12) :
  height_flagpole * 12 = 75 :=
by
  -- Note: To keep the theorem concise, proof steps are omitted
  sorry

end flagpole_height_in_inches_l228_228624


namespace count_valid_sequences_returning_rectangle_l228_228067

/-- The transformations that can be applied to the rectangle -/
inductive Transformation
| rot90   : Transformation
| rot180  : Transformation
| rot270  : Transformation
| reflYeqX  : Transformation
| reflYeqNegX : Transformation

/-- Apply a transformation to a point (x, y) -/
def apply_transformation (t : Transformation) (p : ℝ × ℝ) : ℝ × ℝ :=
match t with
| Transformation.rot90   => (-p.2,  p.1)
| Transformation.rot180  => (-p.1, -p.2)
| Transformation.rot270  => ( p.2, -p.1)
| Transformation.reflYeqX  => ( p.2,  p.1)
| Transformation.reflYeqNegX => (-p.2, -p.1)

/-- Apply a sequence of transformations to a list of points -/
def apply_sequence (seq : List Transformation) (points : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  seq.foldl (λ acc t => acc.map (apply_transformation t)) points

/-- Prove that there are exactly 12 valid sequences of three transformations that return the rectangle to its original position -/
theorem count_valid_sequences_returning_rectangle :
  let rectangle := [(0,0), (6,0), (6,2), (0,2)];
  let transformations := [Transformation.rot90, Transformation.rot180, Transformation.rot270, Transformation.reflYeqX, Transformation.reflYeqNegX];
  let seq_transformations := List.replicate 3 transformations;
  (seq_transformations.filter (λ seq => apply_sequence seq rectangle = rectangle)).length = 12 :=
sorry

end count_valid_sequences_returning_rectangle_l228_228067


namespace calendar_matrix_sum_l228_228338

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![5, 6, 7], 
    ![8, 9, 10], 
    ![11, 12, 13]]

def modified_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![m 0 2, m 0 1, m 0 0], 
    ![m 1 0, m 1 1, m 1 2], 
    ![m 2 2, m 2 1, m 2 0]]

def diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def edge_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 1 + m 0 2 + m 2 0 + m 2 1

def total_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  diagonal_sum m + edge_sum m

theorem calendar_matrix_sum :
  total_sum (modified_matrix initial_matrix) = 63 :=
by
  sorry

end calendar_matrix_sum_l228_228338


namespace range_of_a_l228_228936

def set_A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def set_B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a ≤ -2 ∨ (a > -2 ∧ a ≤ -1/2) ∨ a ≥ 2) := by
  sorry

end range_of_a_l228_228936


namespace football_points_difference_l228_228644

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end football_points_difference_l228_228644


namespace range_of_a_l228_228807

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
by
  sorry -- The proof is omitted as per the instructions.

end range_of_a_l228_228807


namespace tiles_needed_l228_228196

def floor9ₓ12_ft : Type := {l : ℕ × ℕ // l = (9, 12)}
def tile4ₓ6_inch : Type := {l : ℕ × ℕ // l = (4, 6)}

theorem tiles_needed (floor : floor9ₓ12_ft) (tile : tile4ₓ6_inch) : 
  ∃ tiles : ℕ, tiles = 648 :=
sorry

end tiles_needed_l228_228196


namespace prove_expression_value_l228_228528

theorem prove_expression_value (x y : ℝ) (h1 : 4 * x + y = 18) (h2 : x + 4 * y = 20) :
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
sorry

end prove_expression_value_l228_228528


namespace intersection_A_B_l228_228130

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | x > 0}

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {1} := 
by {
  sorry
}

end intersection_A_B_l228_228130


namespace inequality_for_natural_n_l228_228139

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n :=
by sorry

end inequality_for_natural_n_l228_228139


namespace like_terms_mn_l228_228945

theorem like_terms_mn (m n : ℕ) (h1 : -2 * x^m * y^2 = 2 * x^3 * y^n) : m * n = 6 :=
by {
  -- Add the statements transforming the assumptions into intermediate steps
  sorry
}

end like_terms_mn_l228_228945


namespace evaluate_expression_l228_228765

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2
  else -n / 2

theorem evaluate_expression : S 19 * S 31 + S 48 = 136 :=
by sorry

end evaluate_expression_l228_228765


namespace negation_of_sine_bound_l228_228095

theorem negation_of_sine_bound (p : ∀ x : ℝ, Real.sin x ≤ 1) : ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x₀ : ℝ, Real.sin x₀ > 1 := 
by 
  sorry

end negation_of_sine_bound_l228_228095


namespace total_hours_driven_l228_228992

def total_distance : ℝ := 55.0
def distance_in_one_hour : ℝ := 1.527777778

theorem total_hours_driven : (total_distance / distance_in_one_hour) = 36.00 :=
by
  sorry

end total_hours_driven_l228_228992


namespace first_term_arithmetic_sequence_l228_228277

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l228_228277


namespace sum_of_squares_ge_sum_of_products_l228_228560

theorem sum_of_squares_ge_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end sum_of_squares_ge_sum_of_products_l228_228560


namespace jogger_distance_l228_228690

theorem jogger_distance (t : ℝ) (h1 : (20 * t = 12 * t + 15)) : (12 * t = 22.5) := by
  -- We use the equation 20t = 12t + 15
  have h2 : 8 * t = 15 := by linarith
  -- Solve for t
  have h3 : t = 15 / 8 := by linarith
  -- Substitute t back into 12 * t
  show 12 * t = 22.5 by
    rw [h3]
    norm_num

end jogger_distance_l228_228690


namespace circle_graph_to_bar_graph_correct_l228_228920

theorem circle_graph_to_bar_graph_correct :
  ∀ (white black gray blue : ℚ) (w_proportion b_proportion g_proportion blu_proportion : ℚ),
    white = 1/2 →
    black = 1/4 →
    gray = 1/8 →
    blue = 1/8 →
    w_proportion = 1/2 →
    b_proportion = 1/4 →
    g_proportion = 1/8 →
    blu_proportion = 1/8 →
    white = w_proportion ∧ black = b_proportion ∧ gray = g_proportion ∧ blue = blu_proportion :=
by
sorry

end circle_graph_to_bar_graph_correct_l228_228920


namespace spike_hunts_20_crickets_per_day_l228_228145

/-- Spike the bearded dragon hunts 5 crickets every morning -/
def spike_morning_crickets : ℕ := 5

/-- Spike hunts three times the morning amount in the afternoon and evening -/
def spike_afternoon_evening_multiplier : ℕ := 3

/-- Total number of crickets Spike hunts per day -/
def spike_total_crickets_per_day : ℕ := spike_morning_crickets + spike_morning_crickets * spike_afternoon_evening_multiplier

/-- Prove that the total number of crickets Spike hunts per day is 20 -/
theorem spike_hunts_20_crickets_per_day : spike_total_crickets_per_day = 20 := 
by
  sorry

end spike_hunts_20_crickets_per_day_l228_228145


namespace sum_of_coefficients_l228_228017

theorem sum_of_coefficients (x : ℝ) : (∃ x : ℝ, 5 * x * (1 - x) = 3) → 5 + (-5) + 3 = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end sum_of_coefficients_l228_228017


namespace complete_square_l228_228331

theorem complete_square (x : ℝ) : (x ^ 2 + 4 * x + 1 = 0) ↔ ((x + 2) ^ 2 = 3) :=
by {
  split,
  { intro h,
    sorry },
  { intro h,
    sorry }
}

end complete_square_l228_228331


namespace cost_per_adult_meal_l228_228640

theorem cost_per_adult_meal (total_people : ℕ) (num_kids : ℕ) (total_cost : ℕ) (cost_per_kid : ℕ) :
  total_people = 12 →
  num_kids = 7 →
  cost_per_kid = 0 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids)) = 3 :=
by
  intros
  sorry

end cost_per_adult_meal_l228_228640


namespace fraction_difference_l228_228761

theorem fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := 
  sorry

end fraction_difference_l228_228761


namespace exists_integers_x_y_z_l228_228840

theorem exists_integers_x_y_z (n : ℕ) : 
  ∃ x y z : ℤ, (x^2 + y^2 + z^2 = 3^(2^n)) ∧ (Int.gcd x (Int.gcd y z) = 1) :=
sorry

end exists_integers_x_y_z_l228_228840


namespace no_possible_values_for_b_l228_228147

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_possible_values_for_b_l228_228147


namespace Cody_reads_books_in_7_weeks_l228_228484

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end Cody_reads_books_in_7_weeks_l228_228484


namespace proof_a_minus_b_l228_228535

def S (a : ℕ) : Set ℕ := {1, 2, a}
def T (b : ℕ) : Set ℕ := {2, 3, 4, b}

theorem proof_a_minus_b (a b : ℕ)
  (hS : S a = {1, 2, a})
  (hT : T b = {2, 3, 4, b})
  (h_intersection : S a ∩ T b = {1, 2, 3}) :
  a - b = 2 := by
  sorry

end proof_a_minus_b_l228_228535


namespace balloon_descent_rate_l228_228555

theorem balloon_descent_rate (D : ℕ) 
    (rate_of_ascent : ℕ := 50) 
    (time_chain_pulled_1 : ℕ := 15) 
    (time_chain_pulled_2 : ℕ := 15) 
    (time_chain_released_1 : ℕ := 10) 
    (highest_elevation : ℕ := 1400) :
    (time_chain_pulled_1 + time_chain_pulled_2) * rate_of_ascent - time_chain_released_1 * D = highest_elevation 
    → D = 10 := 
by 
  intro h
  sorry

end balloon_descent_rate_l228_228555


namespace a_plus_b_eq_neg1_l228_228803

theorem a_plus_b_eq_neg1 (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : a + b = -1 :=
by
  sorry

end a_plus_b_eq_neg1_l228_228803


namespace f_prime_at_pi_over_six_l228_228947

noncomputable def f (f'_0 : ℝ) (x : ℝ) : ℝ := (1/2)*x^2 + 2*f'_0*(Real.cos x) + x

theorem f_prime_at_pi_over_six (f'_0 : ℝ) (h : f'_0 = 1) :
  (deriv (f f'_0)) (Real.pi / 6) = Real.pi / 6 := by
  sorry

end f_prime_at_pi_over_six_l228_228947


namespace discriminant_of_quadratic_l228_228588

theorem discriminant_of_quadratic : 
  let a := 1
  let b := -7
  let c := 4
  Δ = b ^ 2 - 4 * a * c
  (b ^ 2 - 4 * a * c) = 33 :=
by
  -- definitions of a, b, and c
  let a := 1
  let b := -7
  let c := 4
  -- definition of Δ
  let Δ := b ^ 2 - 4 * a * c
  -- given the quadratic equation x^2 - 7x + 4 = 0, prove that Δ = 33
  show (b ^ 2 - 4 * a * c) = 33,
  -- proof is omitted
  sorry

end discriminant_of_quadratic_l228_228588


namespace trip_time_l228_228889

theorem trip_time (distance half_dist speed1 speed2 : ℝ) 
  (h_distance : distance = 360) 
  (h_half_distance : half_dist = distance / 2) 
  (h_speed1 : speed1 = 50) 
  (h_speed2 : speed2 = 45) : 
  (half_dist / speed1 + half_dist / speed2) = 7.6 := 
by
  -- Simplify the expressions based on provided conditions
  sorry

end trip_time_l228_228889


namespace value_of_f_sin_20_l228_228919

theorem value_of_f_sin_20 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.sin (3 * x)) :
  f (Real.sin (20 * Real.pi / 180)) = -1 / 2 :=
by sorry

end value_of_f_sin_20_l228_228919


namespace cab_speed_fraction_l228_228871

theorem cab_speed_fraction (S R : ℝ) (h1 : S * 40 = R * 48) : (R / S) = (5 / 6) :=
sorry

end cab_speed_fraction_l228_228871


namespace cost_of_camel_l228_228186

-- Define the cost of each animal as variables
variables (C H O E : ℝ)

-- Assume the given relationships as hypotheses
def ten_camels_eq_twentyfour_horses := (10 * C = 24 * H)
def sixteens_horses_eq_four_oxen := (16 * H = 4 * O)
def six_oxen_eq_four_elephants := (6 * O = 4 * E)
def ten_elephants_eq_140000 := (10 * E = 140000)

-- The theorem that we want to prove
theorem cost_of_camel (h1 : ten_camels_eq_twentyfour_horses C H)
                      (h2 : sixteens_horses_eq_four_oxen H O)
                      (h3 : six_oxen_eq_four_elephants O E)
                      (h4 : ten_elephants_eq_140000 E) :
  C = 5600 := sorry

end cost_of_camel_l228_228186


namespace rental_cost_equation_l228_228984

theorem rental_cost_equation (x : ℕ) (h : x > 0) :
  180 / x - 180 / (x + 2) = 3 :=
sorry

end rental_cost_equation_l228_228984


namespace friend_owns_10_bicycles_l228_228387

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end friend_owns_10_bicycles_l228_228387


namespace find_valid_pairs_l228_228225

-- Definitions and conditions:
def satisfies_equation (a b : ℤ) : Prop := a^2 + a * b - b = 2018

-- Correct answers:
def valid_pairs : List (ℤ × ℤ) :=
  [(2, 2014), (0, -2018), (2018, -2018), (-2016, 2014)]

-- Statement to prove:
theorem find_valid_pairs :
  ∀ (a b : ℤ), satisfies_equation a b ↔ (a, b) ∈ valid_pairs.toFinset := by
  sorry

end find_valid_pairs_l228_228225


namespace pencils_placed_by_dan_l228_228725

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end pencils_placed_by_dan_l228_228725


namespace slips_numbers_exist_l228_228990

theorem slips_numbers_exist (x y z : ℕ) (h₁ : x + y + z = 20) (h₂ : 5 * x + 3 * y = 46) : 
  (x = 4) ∧ (y = 10) ∧ (z = 6) :=
by {
  -- Technically, the actual proving steps should go here, but skipped due to 'sorry'
  sorry
}

end slips_numbers_exist_l228_228990


namespace num_adult_tickets_is_35_l228_228872

noncomputable def num_adult_tickets_sold (A C: ℕ): Prop :=
  A + C = 85 ∧ 5 * A + 2 * C = 275

theorem num_adult_tickets_is_35: ∃ A C: ℕ, num_adult_tickets_sold A C ∧ A = 35 :=
by
  -- Definitions based on the provided conditions
  sorry

end num_adult_tickets_is_35_l228_228872


namespace cos_double_angle_l228_228523

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = 1 / 3) : 
  Real.cos (2 * α) = -7 / 9 := 
by 
  sorry

end cos_double_angle_l228_228523


namespace percentage_failed_in_Hindi_l228_228547

-- Let Hindi_failed denote the percentage of students who failed in Hindi.
-- Let English_failed denote the percentage of students who failed in English.
-- Let Both_failed denote the percentage of students who failed in both Hindi and English.
-- Let Both_passed denote the percentage of students who passed in both subjects.

variables (Hindi_failed English_failed Both_failed Both_passed : ℝ)
  (H_condition1 : English_failed = 44)
  (H_condition2 : Both_failed = 22)
  (H_condition3 : Both_passed = 44)

theorem percentage_failed_in_Hindi:
  Hindi_failed = 34 :=
by 
  -- Proof goes here
  sorry

end percentage_failed_in_Hindi_l228_228547


namespace Iris_total_spent_l228_228702

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end Iris_total_spent_l228_228702


namespace sufficient_but_not_necessary_l228_228620

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ ¬(x^2 > 4 → x > 3) :=
by sorry

end sufficient_but_not_necessary_l228_228620


namespace smallest_number_is_27_l228_228727

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end smallest_number_is_27_l228_228727


namespace triangle_circle_fill_l228_228953

theorem triangle_circle_fill (A B C D : ℕ) : 
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9) →
  (B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9) →
  (C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9) →
  (D = 6 ∨ D = 7 ∨ D = 8 ∨ D = 9) →
  (A + B + 1 + 8 =  A + 4 + 3 + 7) →  (D + 4 + 2 + 5 = 5 + 1 + 8 + B) →
  (5 + 1 + 8 + 6 = 5 + C + 7 + 4 ) →
  (A = 6) ∧ (B = 8) ∧ (C = 7) ∧ (D = 9) := by
  sorry

end triangle_circle_fill_l228_228953


namespace stamp_book_gcd_l228_228959

theorem stamp_book_gcd (total1 total2 total3 : ℕ) 
    (h1 : total1 = 945) (h2 : total2 = 1260) (h3 : total3 = 630) : 
    ∃ d, d = Nat.gcd (Nat.gcd total1 total2) total3 ∧ d = 315 := 
by
  sorry

end stamp_book_gcd_l228_228959


namespace factorial_binomial_mod_l228_228847

theorem factorial_binomial_mod (p : ℕ) (hp : Nat.Prime p) : 
  ((Nat.factorial (2 * p)) / (Nat.factorial p * Nat.factorial p)) - 2 ≡ 0 [MOD p] :=
by
  sorry

end factorial_binomial_mod_l228_228847


namespace second_discount_percentage_is_20_l228_228353

theorem second_discount_percentage_is_20 
    (normal_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (first_discount_percentage : ℝ)
    (h1 : normal_price = 149.99999999999997)
    (h2 : final_price = 108)
    (h3 : first_discount_percentage = 10)
    (h4 : first_discount = normal_price * (first_discount_percentage / 100)) :
    (((normal_price - first_discount) - final_price) / (normal_price - first_discount)) * 100 = 20 := by
  sorry

end second_discount_percentage_is_20_l228_228353


namespace ab_minus_a_inv_b_l228_228688

theorem ab_minus_a_inv_b (a : ℝ) (b : ℚ) (h1 : a > 1) (h2 : 0 < (b : ℝ)) (h3 : (a ^ (b : ℝ)) + (a ^ (-(b : ℝ))) = 2 * Real.sqrt 2) :
  (a ^ (b : ℝ)) - (a ^ (-(b : ℝ))) = 2 := 
sorry

end ab_minus_a_inv_b_l228_228688


namespace largest_k_for_positive_root_l228_228319

theorem largest_k_for_positive_root : ∃ k : ℤ, k = 1 ∧ ∀ k' : ℤ, (k' > 1) → ¬ (∃ x > 0, 3 * x * (2 * k' * x - 5) - 2 * x^2 + 8 = 0) :=
by
  sorry

end largest_k_for_positive_root_l228_228319


namespace largest_not_sum_of_36_and_composite_l228_228169

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l228_228169


namespace positive_difference_arithmetic_sequence_l228_228174

theorem positive_difference_arithmetic_sequence :
  let a := 3
  let d := 5
  let a₁₀₀ := a + (100 - 1) * d
  let a₁₁₀ := a + (110 - 1) * d
  a₁₁₀ - a₁₀₀ = 50 :=
by
  sorry

end positive_difference_arithmetic_sequence_l228_228174


namespace exists_six_digit_no_identical_six_endings_l228_228655

theorem exists_six_digit_no_identical_six_endings :
  ∃ (A : ℕ), (100000 ≤ A ∧ A < 1000000) ∧ ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 500000) → 
  (∀ d, d ≠ 0 → d < 10 → (k * A) % 1000000 ≠ d * 111111) :=
by
  sorry

end exists_six_digit_no_identical_six_endings_l228_228655


namespace decimal_fraction_eq_l228_228005

theorem decimal_fraction_eq {b : ℕ} (hb : 0 < b) :
  (4 * b + 19 : ℚ) / (6 * b + 11) = 0.76 → b = 19 :=
by
  -- Proof goes here
  sorry

end decimal_fraction_eq_l228_228005


namespace simplify_expr_l228_228294

open Real

theorem simplify_expr (x : ℝ) (hx : 1 ≤ x) :
  sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 
  if x ≤ 2 then 2 else 2 * sqrt (x - 1) :=
by sorry

end simplify_expr_l228_228294


namespace p_adic_valuation_of_factorial_l228_228839

noncomputable def digit_sum (n p : ℕ) : ℕ :=
  -- Definition for sum of digits of n in base p
  sorry

def p_adic_valuation (n factorial : ℕ) (p : ℕ) : ℕ :=
  -- Representation of p-adic valuation of n!
  sorry

theorem p_adic_valuation_of_factorial (n p : ℕ) (hp: p > 1):
  p_adic_valuation n.factorial p = (n - digit_sum n p) / (p - 1) :=
sorry

end p_adic_valuation_of_factorial_l228_228839


namespace trisha_hourly_wage_l228_228024

theorem trisha_hourly_wage (annual_take_home_pay : ℝ) (percent_withheld : ℝ)
  (hours_per_week : ℝ) (weeks_per_year : ℝ) (hourly_wage : ℝ) :
  annual_take_home_pay = 24960 ∧ 
  percent_withheld = 0.20 ∧ 
  hours_per_week = 40 ∧ 
  weeks_per_year = 52 ∧ 
  hourly_wage = (annual_take_home_pay / (0.80 * (hours_per_week * weeks_per_year))) → 
  hourly_wage = 15 :=
by sorry

end trisha_hourly_wage_l228_228024


namespace fraction_of_dark_tiles_is_correct_l228_228631

def num_tiles_in_block : ℕ := 64
def num_dark_tiles : ℕ := 18
def expected_fraction_dark_tiles : ℚ := 9 / 32

theorem fraction_of_dark_tiles_is_correct :
  (num_dark_tiles : ℚ) / num_tiles_in_block = expected_fraction_dark_tiles := by
sorry

end fraction_of_dark_tiles_is_correct_l228_228631


namespace trail_mix_total_weight_l228_228760

def peanuts : ℝ := 0.16666666666666666
def chocolate_chips : ℝ := 0.16666666666666666
def raisins : ℝ := 0.08333333333333333
def trail_mix_weight : ℝ := 0.41666666666666663

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = trail_mix_weight :=
sorry

end trail_mix_total_weight_l228_228760


namespace correct_transformation_l228_228732

theorem correct_transformation (m : ℤ) (h : 2 * m - 1 = 3) : 2 * m = 4 :=
by
  sorry

end correct_transformation_l228_228732


namespace crown_cost_before_tip_l228_228198

theorem crown_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (crown_cost : ℝ) :
  total_paid = 22000 → tip_percentage = 0.10 → total_paid = crown_cost * (1 + tip_percentage) → crown_cost = 20000 :=
by
  sorry

end crown_cost_before_tip_l228_228198


namespace problem_solution_l228_228124

-- Definitions for the digits and arithmetic conditions
def is_digit (n : ℕ) : Prop := n < 10

-- Problem conditions stated in Lean
variables (A B C D E : ℕ)

-- Define the conditions
axiom digits_A : is_digit A
axiom digits_B : is_digit B
axiom digits_C : is_digit C
axiom digits_D : is_digit D
axiom digits_E : is_digit E

-- Subtraction result for second equation
axiom sub_eq : A - C = A

-- Additional conditions derived from the problem
axiom add_eq : (E + E = D)

-- Now, state the problem in Lean
theorem problem_solution : D = 8 :=
sorry

end problem_solution_l228_228124


namespace find_smallest_x_satisfying_condition_l228_228904

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l228_228904


namespace sqrt_floor_eq_sqrt_sqrt_floor_l228_228113

theorem sqrt_floor_eq_sqrt_sqrt_floor (a : ℝ) (h : a > 1) :
  Int.floor (Real.sqrt (Int.floor (Real.sqrt a))) = Int.floor (Real.sqrt (Real.sqrt a)) :=
sorry

end sqrt_floor_eq_sqrt_sqrt_floor_l228_228113


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l228_228172

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l228_228172


namespace find_a_of_cool_frog_meeting_l228_228962

-- Question and conditions
def frogs : ℕ := 16
def friend_probability : ℚ := 1 / 2
def cool_condition (f: ℕ → ℕ) : Prop := ∀ i, f i % 4 = 0

-- Example theorem where we need to find 'a'
theorem find_a_of_cool_frog_meeting :
  let a := 1167
  let b := 2 ^ 41
  ∀ (f: ℕ → ℕ), ∀ (p: ℚ) (h: p = friend_probability),
    (cool_condition f) →
    (∃ a b, a / b = p ∧ a % gcd a b = 0 ∧ gcd a b = 1) ∧ a = 1167 :=
by
  sorry

end find_a_of_cool_frog_meeting_l228_228962


namespace son_time_to_complete_work_l228_228199

noncomputable def man_work_rate : ℚ := 1 / 6
noncomputable def combined_work_rate : ℚ := 1 / 3

theorem son_time_to_complete_work :
  (1 / (combined_work_rate - man_work_rate)) = 6 := by
  sorry

end son_time_to_complete_work_l228_228199


namespace cube_volume_from_surface_area_l228_228447

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 150) : (S / 6) ^ (3 / 2) = 125 := by
  sorry

end cube_volume_from_surface_area_l228_228447


namespace numbers_divisible_by_3_but_not_2_l228_228941

theorem numbers_divisible_by_3_but_not_2 (n : ℕ) (h₀ : n < 100) : 
  (∃ m, 1 ≤ m ∧ m < 100 ∧ m % 3 = 0 ∧ m % 2 ≠ 0) ↔ n = 17 := 
by {
  sorry
}

end numbers_divisible_by_3_but_not_2_l228_228941


namespace no_real_sol_l228_228071

open Complex

theorem no_real_sol (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (↑(x.re) ≠ x ∨ ↑(y.re) ≠ y) → (x + y) / y ≠ x / (y + x) := by
  sorry

end no_real_sol_l228_228071


namespace roberto_current_salary_l228_228982

theorem roberto_current_salary (starting_salary current_salary : ℝ) (h₀ : starting_salary = 80000)
(h₁ : current_salary = (starting_salary * 1.4) * 1.2) : 
current_salary = 134400 := by
  sorry

end roberto_current_salary_l228_228982


namespace find_trigonometric_identity_l228_228790

-- Define the conditions for the given problem:
variables (a b c S : ℝ) (A : ℝ)

-- Assume the area of the triangle is given by:
-- S = a^2 - (b - c)^2
axiom area_condition : S = a^2 - (b - c)^2

-- Prove that given these conditions, the value of
-- sin A / (1 - cos A) equals 4.
theorem find_trigonometric_identity (h : S = a^2 - (b - c)^2) : 
  ∀ (A : ℝ), b ≠ 0 → c ≠ 0 → cos A ≠ 1 → sin A / (1 - cos A) = 4 :=
by
  sorry

end find_trigonometric_identity_l228_228790


namespace expected_value_of_sum_of_two_marbles_l228_228245

open Finset

noncomputable def choose2 (s:Finset ℕ) := s.powerset.filter (λ t, t.card = 2)

theorem expected_value_of_sum_of_two_marbles:
  let marbles := range 1 8 in
  let num_pairs := (choose2 marbles).card in
  let total_sum := (choose2 marbles).sum (λ t, t.sum id) in
  (total_sum:ℚ) / (num_pairs:ℚ) = 8 :=
by
  let marbles := range 1 8
  let num_pairs := (choose2 marbles).card
  let total_sum := (choose2 marbles).sum (λ t, t.sum id)
  have h1: num_pairs = 21, by sorry
  have h2: total_sum = 168, by sorry
  rw [h1, h2]
  norm_num

end expected_value_of_sum_of_two_marbles_l228_228245


namespace arrival_time_difference_l228_228639

-- Define the times in minutes, with 600 representing 10:00 AM.
def my_watch_time_planned := 600
def my_watch_fast := 5
def my_watch_slow := 10

def friend_watch_time_planned := 600
def friend_watch_fast := 5

-- Calculate actual arrival times.
def my_actual_arrival_time := my_watch_time_planned - my_watch_fast + my_watch_slow
def friend_actual_arrival_time := friend_watch_time_planned - friend_watch_fast

-- Prove the arrival times and difference.
theorem arrival_time_difference :
  friend_actual_arrival_time < my_actual_arrival_time ∧
  my_actual_arrival_time - friend_actual_arrival_time = 20 :=
by
  -- Proof terms can be filled in later.
  sorry

end arrival_time_difference_l228_228639


namespace sin_double_angle_l228_228668

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
by sorry

end sin_double_angle_l228_228668


namespace hawks_score_l228_228542

theorem hawks_score (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 22) : H = 30 :=
by
  sorry

end hawks_score_l228_228542


namespace train_speed_length_l228_228910

theorem train_speed_length (t1 t2 s : ℕ) (p : ℕ)
  (h1 : t1 = 7) 
  (h2 : t2 = 25) 
  (h3 : p = 378)
  (h4 : t2 - t1 = 18)
  (h5 : p / (t2 - t1) = 21) 
  (h6 : (p / (t2 - t1)) * t1 = 147) :
  (21, 147) = (21, 147) :=
by {
  sorry
}

end train_speed_length_l228_228910


namespace common_difference_l228_228085

variable (a : ℕ → ℝ)

def arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a1, ∀ n, a n = a1 + (n - 1) * d

def geometric_sequence (a1 a2 a5 : ℝ) : Prop :=
  a1 * (a1 + 4 * (a2 - a1)) = (a2 - a1)^2

theorem common_difference {d : ℝ} (hd : d ≠ 0)
  (h_arith : arithmetic a d)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geom : geometric_sequence (a 1) (a 2) (a 5)) :
  d = 2 :=
sorry

end common_difference_l228_228085


namespace distance_between_polar_points_l228_228814

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end distance_between_polar_points_l228_228814


namespace overall_percentage_decrease_l228_228572

-- Define the initial pay cut percentages as given in the conditions.
def first_pay_cut := 5.25 / 100
def second_pay_cut := 9.75 / 100
def third_pay_cut := 14.6 / 100
def fourth_pay_cut := 12.8 / 100

-- Define the single shot percentage decrease we want to prove.
def single_shot_decrease := 36.73 / 100

-- Calculate the cumulative multiplier from individual pay cuts.
def cumulative_multiplier := 
  (1 - first_pay_cut) * (1 - second_pay_cut) * (1 - third_pay_cut) * (1 - fourth_pay_cut)

-- Statement: Prove the overall percentage decrease using cumulative multiplier is equal to single shot decrease.
theorem overall_percentage_decrease :
  1 - cumulative_multiplier = single_shot_decrease :=
by sorry

end overall_percentage_decrease_l228_228572


namespace avg_k_of_positive_integer_roots_l228_228929

theorem avg_k_of_positive_integer_roots : 
  (∑ (k : ℕ) in ({25, 14, 11, 10} : Finset ℕ), k) / 4 = 15 := 
by sorry

end avg_k_of_positive_integer_roots_l228_228929


namespace marbles_shared_equally_l228_228605

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end marbles_shared_equally_l228_228605


namespace blue_paint_amount_l228_228916

theorem blue_paint_amount
  (blue_white_ratio : ℚ := 4 / 5)
  (white_paint : ℚ := 15)
  (blue_paint : ℚ) :
  blue_paint = 12 :=
by
  sorry

end blue_paint_amount_l228_228916


namespace sqrt_9_is_pm3_l228_228016

theorem sqrt_9_is_pm3 : {x : ℝ | x ^ 2 = 9} = {3, -3} := sorry

end sqrt_9_is_pm3_l228_228016


namespace like_terms_m_eq_2_l228_228801

theorem like_terms_m_eq_2 (m : ℕ) :
  (∀ (x y : ℝ), 3 * x^m * y^3 = 3 * x^2 * y^3) -> m = 2 :=
by
  intros _
  sorry

end like_terms_m_eq_2_l228_228801


namespace tuning_day_method_pi_l228_228152

variable (x : ℝ)

-- Initial bounds and approximations
def initial_bounds (π : ℝ) := 31 / 10 < π ∧ π < 49 / 15

-- Definition of the "Tuning Day Method"
def tuning_day_method (a b c d : ℕ) (a' b' : ℝ) := a' = (b + d) / (a + c)

theorem tuning_day_method_pi :
  ∀ π : ℝ, initial_bounds π →
  (31 / 10 < π ∧ π < 16 / 5) ∧ 
  (47 / 15 < π ∧ π < 63 / 20) ∧
  (47 / 15 < π ∧ π < 22 / 7) →
  22 / 7 = 22 / 7 :=
by
  sorry

end tuning_day_method_pi_l228_228152


namespace watch_cost_price_l228_228638

theorem watch_cost_price (C : ℝ) (h1 : 0.85 * C = SP1) (h2 : 1.06 * C = SP2) (h3 : SP2 - SP1 = 350) : 
  C = 1666.67 := 
  sorry

end watch_cost_price_l228_228638


namespace number_of_paths_K_to_L_l228_228398

-- Definition of the problem structure
def K : Type := Unit
def A : Type := Unit
def R : Type := Unit
def L : Type := Unit

-- Defining the number of paths between each stage
def paths_from_K_to_A := 2
def paths_from_A_to_R := 4
def paths_from_R_to_L := 8

-- The main theorem stating the number of paths from K to L
theorem number_of_paths_K_to_L : paths_from_K_to_A * 2 * 2 = 8 := by 
  sorry

end number_of_paths_K_to_L_l228_228398


namespace experienced_sailors_monthly_earnings_l228_228473

theorem experienced_sailors_monthly_earnings :
  let total_sailors : Nat := 17
  let inexperienced_sailors : Nat := 5
  let hourly_wage_inexperienced : Nat := 10
  let workweek_hours : Nat := 60
  let weeks_in_month : Nat := 4
  let experienced_sailors : Nat := total_sailors - inexperienced_sailors
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5)
  let weekly_earnings_experienced := hourly_wage_experienced * workweek_hours
  let total_weekly_earnings_experienced := weekly_earnings_experienced * experienced_sailors
  let monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_in_month
  monthly_earnings_experienced = 34560 := by
  sorry

end experienced_sailors_monthly_earnings_l228_228473


namespace marissa_initial_ribbon_l228_228833

theorem marissa_initial_ribbon (ribbon_per_box : ℝ) (number_of_boxes : ℝ) (ribbon_left : ℝ) : 
  (ribbon_per_box = 0.7) → (number_of_boxes = 5) → (ribbon_left = 1) → 
  (ribbon_per_box * number_of_boxes + ribbon_left = 4.5) :=
  by
    intros
    sorry

end marissa_initial_ribbon_l228_228833


namespace directrix_of_parabola_l228_228377

theorem directrix_of_parabola (p m : ℝ) (hp : p > 0)
  (hM_on_parabola : (4, m).fst ^ 2 = 2 * p * (4, m).snd)
  (hM_to_focus : dist (4, m) (p / 2, 0) = 6) :
  -p/2 = -2 :=
sorry

end directrix_of_parabola_l228_228377


namespace smallest_multiple_of_6_and_15_l228_228510

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l228_228510


namespace solve_cryptarithm_l228_228989

-- Definitions for digits mapped to letters
def C : ℕ := 9
def H : ℕ := 3
def U : ℕ := 5
def K : ℕ := 4
def T : ℕ := 1
def R : ℕ := 2
def I : ℕ := 0
def G : ℕ := 6
def N : ℕ := 8
def S : ℕ := 7

-- Function to evaluate the cryptarithm sum
def cryptarithm_sum : ℕ :=
  (C*10000 + H*1000 + U*100 + C*10 + K) +
  (T*10000 + R*1000 + I*100 + G*10 + G) +
  (T*10000 + U*1000 + R*100 + N*10 + S)

-- Equation checking the result
def cryptarithm_correct : Prop :=
  cryptarithm_sum = T*100000 + R*10000 + I*1000 + C*100 + K*10 + S

-- The theorem we want to prove
theorem solve_cryptarithm : cryptarithm_correct :=
by
  -- Proof steps would be filled here
  -- but for now, we just acknowledge it is a theorem
  sorry

end solve_cryptarithm_l228_228989


namespace hexagon_angle_U_l228_228394

theorem hexagon_angle_U 
  (F I U G E R : ℝ)
  (h1 : F = I) 
  (h2 : I = U)
  (h3 : G + E = 180)
  (h4 : R + U = 180)
  (h5 : F + I + G + U + R + E = 720) :
  U = 120 := by
  sorry

end hexagon_angle_U_l228_228394


namespace melanie_total_payment_l228_228133

noncomputable def totalCost (rentalCostPerDay : ℝ) (insuranceCostPerDay : ℝ) (mileageCostPerMile : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  (rentalCostPerDay * days) + (insuranceCostPerDay * days) + (mileageCostPerMile * miles)

theorem melanie_total_payment :
  totalCost 30 5 0.25 3 350 = 192.5 :=
by
  sorry

end melanie_total_payment_l228_228133


namespace exists_language_spoken_by_at_least_three_l228_228439

noncomputable def smallestValue_n (k : ℕ) : ℕ :=
  2 * k + 3

theorem exists_language_spoken_by_at_least_three (k n : ℕ) (P : Fin n → Set ℕ) (K : ℕ → ℕ) :
  (n = smallestValue_n k) →
  (∀ i, (K i) ≤ k) →
  (∀ (x y z : Fin n), ∃ l, l ∈ P x ∧ l ∈ P y ∧ l ∈ P z ∨ l ∈ P y ∧ l ∈ P z ∨ l ∈ P z ∧ l ∈ P x ∨ l ∈ P x ∧ l ∈ P y) →
  ∃ l, ∃ (a b c : Fin n), l ∈ P a ∧ l ∈ P b ∧ l ∈ P c :=
by
  intros h1 h2 h3
  sorry

end exists_language_spoken_by_at_least_three_l228_228439


namespace cylindrical_tank_volume_increase_l228_228194

theorem cylindrical_tank_volume_increase (k : ℝ) (H R : ℝ) 
  (hR : R = 10) (hH : H = 5)
  (condition : (π * (10 * k)^2 * 5 - π * 10^2 * 5) = (π * 10^2 * (5 + k) - π * 10^2 * 5)) :
  k = (1 + Real.sqrt 101) / 10 :=
by
  sorry

end cylindrical_tank_volume_increase_l228_228194


namespace chromatic_equals_flow_l228_228913

variables {G : Type*} [planar_multigraph G] -- Assuming G is a planar multigraph type
variables {G_star : Type*} [dual_graph G G_star] -- Assuming G_star is the dual graph of G

theorem chromatic_equals_flow (G : Type*) [planar_multigraph G]
  (G_star : Type*) [dual_graph G G_star] :
  chromatic_number G = flow_number G_star :=
sorry

end chromatic_equals_flow_l228_228913


namespace box_mass_calculation_l228_228040

variable (h₁ w₁ l₁ : ℝ) (m₁ : ℝ)
variable (h₂ w₂ l₂ density₁ density₂ : ℝ)

theorem box_mass_calculation
  (h₁_eq : h₁ = 3)
  (w₁_eq : w₁ = 4)
  (l₁_eq : l₁ = 6)
  (m₁_eq : m₁ = 72)
  (h₂_eq : h₂ = 1.5 * h₁)
  (w₂_eq : w₂ = 2.5 * w₁)
  (l₂_eq : l₂ = l₁)
  (density₂_eq : density₂ = 2 * density₁)
  (density₁_eq : density₁ = m₁ / (h₁ * w₁ * l₁)) :
  h₂ * w₂ * l₂ * density₂ = 540 := by
  sorry

end box_mass_calculation_l228_228040


namespace border_material_length_l228_228463

noncomputable def area (r : ℝ) : ℝ := (22 / 7) * r^2

theorem border_material_length (r : ℝ) (C : ℝ) (border : ℝ) : 
  area r = 616 →
  C = 2 * (22 / 7) * r →
  border = C + 3 →
  border = 91 :=
by
  intro h_area h_circumference h_border
  sorry

end border_material_length_l228_228463


namespace original_number_input_0_2_l228_228975

theorem original_number_input_0_2 (x : ℝ) (hx : x ≠ 0) (h : (1 / (1 / x - 1) - 1 = -0.75)) : x = 0.2 := 
sorry

end original_number_input_0_2_l228_228975


namespace correct_option_C_l228_228096

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 5}
def P : Set ℕ := {2, 4}

theorem correct_option_C : 3 ∈ U \ (M ∪ P) :=
by
  sorry

end correct_option_C_l228_228096


namespace proof_f_1_add_g_2_l228_228705

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 1

theorem proof_f_1_add_g_2 : f (1 + g 2) = 8 := by
  sorry

end proof_f_1_add_g_2_l228_228705


namespace gasoline_storage_l228_228123

noncomputable def total_distance : ℕ := 280 * 2

noncomputable def miles_per_segment : ℕ := 40

noncomputable def gasoline_consumption : ℕ := 8

noncomputable def total_segments : ℕ := total_distance / miles_per_segment

noncomputable def total_gasoline : ℕ := total_segments * gasoline_consumption

noncomputable def number_of_refills : ℕ := 14

theorem gasoline_storage (storage_capacity : ℕ) (h : number_of_refills * storage_capacity = total_gasoline) :
  storage_capacity = 8 :=
by
  sorry

end gasoline_storage_l228_228123


namespace first_term_arithmetic_sequence_l228_228278

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l228_228278


namespace minimum_value_of_fraction_l228_228362

theorem minimum_value_of_fraction (x : ℝ) (hx : x > 10) : ∃ m, m = 30 ∧ ∀ y > 10, (y * y) / (y - 10) ≥ m :=
by 
  sorry

end minimum_value_of_fraction_l228_228362


namespace stock_AB_increase_factor_l228_228797

-- Define the conditions as mathematical terms
def stock_A_initial := 300
def stock_B_initial := 300
def stock_C_initial := 300
def stock_C_final := stock_C_initial / 2
def total_final := 1350
def AB_combined_initial := stock_A_initial + stock_B_initial
def AB_combined_final := total_final - stock_C_final

-- The statement to prove that the factor by which stocks A and B increased in value is 2.
theorem stock_AB_increase_factor :
  AB_combined_final / AB_combined_initial = 2 :=
  by
    sorry

end stock_AB_increase_factor_l228_228797


namespace cannot_determine_b_l228_228239

theorem cannot_determine_b 
  (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 12.345) 
  (h_ineq : a > b ∧ b > c ∧ c > d) : 
  ¬((b = 12.345) ∨ (b > 12.345) ∨ (b < 12.345)) :=
sorry

end cannot_determine_b_l228_228239


namespace find_point_B_l228_228087

theorem find_point_B (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, -5)) 
  (ha : a = (2, 3)) 
  (hAB : B - A = 3 • a) : 
  B = (5, 4) := sorry

end find_point_B_l228_228087


namespace sum_divisible_by_10_l228_228137

theorem sum_divisible_by_10 :
    (111 ^ 111 + 112 ^ 112 + 113 ^ 113) % 10 = 0 :=
by
  sorry

end sum_divisible_by_10_l228_228137


namespace complex_expression_evaluation_l228_228089

noncomputable def imaginary_i := Complex.I 

theorem complex_expression_evaluation : 
  ((2 + imaginary_i) / (1 - imaginary_i)) - (1 - imaginary_i) = -1/2 + (5/2) * imaginary_i :=
by 
  sorry

end complex_expression_evaluation_l228_228089


namespace continuous_stripe_probability_l228_228774

def cube_stripe_probability : ℚ :=
  let stripe_combinations_per_face := 8
  let total_combinations := stripe_combinations_per_face ^ 6
  let valid_combinations := 4 * 3 * 8 * 64
  let probability := valid_combinations / total_combinations
  probability

theorem continuous_stripe_probability :
  cube_stripe_probability = 3 / 128 := by
  sorry

end continuous_stripe_probability_l228_228774


namespace simplifiedtown_path_difference_l228_228809

/-- In Simplifiedtown, all streets are 30 feet wide. Each enclosed block forms a square with 
each side measuring 400 feet. Sarah runs exactly next to the block on a path that is 400 feet 
from the block's inner edge while Maude runs on the outer edge of the street opposite to 
Sarah. Prove that Maude runs 120 feet more than Sarah for each lap around the block. -/
theorem simplifiedtown_path_difference :
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  maude_lap - sarah_lap = 120 :=
by
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  show maude_lap - sarah_lap = 120
  sorry

end simplifiedtown_path_difference_l228_228809


namespace factor_expression_l228_228886

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) :=
sorry

end factor_expression_l228_228886


namespace multiply_base5_234_75_l228_228595

def to_base5 (n : ℕ) : ℕ := 
  let rec helper (n : ℕ) (acc : ℕ) (multiplier : ℕ) : ℕ := 
    if n = 0 then acc
    else
      let d := n % 5
      let q := n / 5
      helper q (acc + d * multiplier) (multiplier * 10)
  helper n 0 1

def base5_multiplication (a b : ℕ) : ℕ :=
  to_base5 ((a * b : ℕ))

theorem multiply_base5_234_75 : base5_multiplication 234 75 = 450620 := 
  sorry

end multiply_base5_234_75_l228_228595


namespace relationship_f_l228_228927

-- Define the function f which is defined on the reals and even
variable (f : ℝ → ℝ)
-- Condition: f is an even function
axiom even_f : ∀ x, f (-x) = f x
-- Condition: (x₁ - x₂)[f(x₁) - f(x₂)] > 0 for all x₁, x₂ ∈ [0, +∞)
axiom increasing_cond : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem relationship_f : f (1/2) < f 1 ∧ f 1 < f (-2) := by
  sorry

end relationship_f_l228_228927


namespace minute_hand_only_rotates_l228_228479

-- Define what constitutes translation and rotation
def is_translation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p1 p2 : ℝ), motion p1 p2 → (∃ d : ℝ, ∀ t : ℝ, motion (p1 + t) (p2 + t) ∧ |p1 - p2| = d)

def is_rotation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p : ℝ), ∃ c : ℝ, ∃ r : ℝ, (∀ (t : ℝ), |p - c| = r)

-- Define the condition that the minute hand of a clock undergoes a specific motion
def minute_hand_motion (p : ℝ) (t : ℝ) : Prop :=
  -- The exact definition here would involve trigonometric representation
  sorry

-- The main proof statement
theorem minute_hand_only_rotates :
  is_rotation minute_hand_motion ∧ ¬ is_translation minute_hand_motion :=
sorry

end minute_hand_only_rotates_l228_228479


namespace largest_prime_factor_l228_228897

theorem largest_prime_factor (a b c d : ℕ) (ha : a = 20) (hb : b = 15) (hc : c = 10) (hd : d = 5) :
  ∃ p, Nat.Prime p ∧ p = 103 ∧ ∀ q, Nat.Prime q ∧ q ∣ (a^3 + b^4 - c^5 + d^6) → q ≤ p :=
by
  sorry

end largest_prime_factor_l228_228897


namespace Emilee_earns_25_l228_228264

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l228_228264


namespace smallest_side_of_triangle_l228_228140

variable {α : Type} [LinearOrderedField α]

theorem smallest_side_of_triangle (a b c : α) (h : a^2 + b^2 > 5*c^2) : c ≤ a ∧ c ≤ b :=
by
  sorry

end smallest_side_of_triangle_l228_228140


namespace prime_square_minus_one_divisible_by_24_l228_228281

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : 5 ≤ p) : 24 ∣ (p^2 - 1) := 
by 
sorry

end prime_square_minus_one_divisible_by_24_l228_228281


namespace number_of_pumpkin_pies_l228_228192

-- Definitions for the conditions
def apple_pies : ℕ := 2
def pecan_pies : ℕ := 4
def total_pies : ℕ := 13

-- The proof statement
theorem number_of_pumpkin_pies
  (h_apple : apple_pies = 2)
  (h_pecan : pecan_pies = 4)
  (h_total : total_pies = 13) : 
  total_pies - (apple_pies + pecan_pies) = 7 :=
by 
  sorry

end number_of_pumpkin_pies_l228_228192


namespace number_of_persons_in_room_l228_228255

theorem number_of_persons_in_room (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
by
  /- We have:
     n * (n - 1) / 2 = 78,
     We need to prove n = 13 -/
  sorry

end number_of_persons_in_room_l228_228255


namespace megan_popsicles_l228_228976

variable (t_rate : ℕ) (t_hours : ℕ)

def popsicles_eaten (rate: ℕ) (hours: ℕ) : ℕ :=
  60 * hours / rate

theorem megan_popsicles : popsicles_eaten 20 5 = 15 := by
  sorry

end megan_popsicles_l228_228976


namespace exp_mul_l228_228647

variable {a : ℝ}

-- Define a theorem stating the problem: proof that a^2 * a^3 = a^5
theorem exp_mul (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exp_mul_l228_228647


namespace find_m_l228_228534

open Set

variable (A B : Set ℝ) (m : ℝ)

theorem find_m (h : A = {-1, 2, 2 * m - 1}) (h2 : B = {2, m^2}) (h3 : B ⊆ A) : m = 1 := 
by
  sorry

end find_m_l228_228534


namespace rugby_team_new_avg_weight_l228_228164

noncomputable def new_average_weight (original_players : ℕ) (original_avg_weight : ℕ) 
  (new_player_weights : List ℕ) : ℚ :=
  let total_original_weight := original_players * original_avg_weight
  let total_new_weight := new_player_weights.foldl (· + ·) 0
  let new_total_weight := total_original_weight + total_new_weight
  let new_total_players := original_players + new_player_weights.length
  (new_total_weight : ℚ) / (new_total_players : ℚ)

theorem rugby_team_new_avg_weight :
  new_average_weight 20 180 [210, 220, 230] = 185.22 := by
  sorry

end rugby_team_new_avg_weight_l228_228164


namespace sanjay_homework_fraction_l228_228844

theorem sanjay_homework_fraction (x : ℚ) :
  (2 * x + 1) / 3 + 4 / 15 = 1 ↔ x = 3 / 5 :=
by
  sorry

end sanjay_homework_fraction_l228_228844


namespace find_first_term_l228_228270

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l228_228270


namespace range_of_b_l228_228931

theorem range_of_b (a b : ℝ) : 
  (∀ x : ℝ, -3 < x ∧ x < 1 → (1 - a) * x^2 - 4 * x + 6 > 0) ∧
  (∀ x : ℝ, 3 * x^2 + b * x + 3 ≥ 0) →
  (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end range_of_b_l228_228931


namespace ants_first_group_count_l228_228296

theorem ants_first_group_count :
    ∃ x : ℕ, 
        (∀ (w1 c1 a1 t1 w2 c2 a2 t2 : ℕ),
          w1 = 10 ∧ c1 = 600 ∧ a1 = x ∧ t1 = 5 ∧
          w2 = 5 ∧ c2 = 960 ∧ a2 = 20 ∧ t2 = 3 ∧ 
          (w1 * c1) / t1 = 1200 / a1 ∧ (w2 * c2) / t2 = 1600 / 20 →
             x = 15)
:= sorry

end ants_first_group_count_l228_228296


namespace diophantine_equation_solvable_l228_228337

theorem diophantine_equation_solvable (a : ℕ) (ha : 0 < a) : 
  ∃ (x y : ℤ), x^2 - y^2 = a^3 :=
by
  let x := (a * (a + 1)) / 2
  let y := (a * (a - 1)) / 2
  have hx : x^2 = (a * (a + 1) / 2 : ℤ)^2 := sorry
  have hy : y^2 = (a * (a - 1) / 2 : ℤ)^2 := sorry
  use x
  use y
  sorry

end diophantine_equation_solvable_l228_228337


namespace speed_ratio_l228_228006

def distance_to_work := 28
def speed_back := 14
def total_time := 6

theorem speed_ratio 
  (d : ℕ := distance_to_work) 
  (v_2 : ℕ := speed_back) 
  (t : ℕ := total_time) : 
  ∃ v_1 : ℕ, (d / v_1 + d / v_2 = t) ∧ (v_2 / v_1 = 2) :=
by 
  sorry

end speed_ratio_l228_228006


namespace main_theorem_l228_228404

-- Define the polynomial with integer coefficients
def P : ℤ[X] := sorry

-- Define the degree of the polynomial P
def deg_P : ℤ := P.degree.to_nat

-- Define the number of integer solutions k such that (P(k))^2 = 1
noncomputable def n_P : ℕ := (finset.filter (λ k : ℤ, (P.eval k)^2 = 1) finset.Icc (-P.degree.to_nat) P.degree.to_nat).card

-- Prove the statement
theorem main_theorem : deg_P ≥ 1 → n_P - deg_P ≤ 2 :=
by
  -- Proof omitted
  sorry

end main_theorem_l228_228404


namespace intersection_of_M_and_N_l228_228097

def M : Set ℝ := { x : ℝ | x^2 - x > 0 }
def N : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | x > 1 } :=
by
  sorry

end intersection_of_M_and_N_l228_228097


namespace add_to_both_num_and_denom_l228_228175

theorem add_to_both_num_and_denom (n : ℕ) : (4 + n) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end add_to_both_num_and_denom_l228_228175


namespace solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l228_228427

-- Problem 1: Prove the solutions to x^2 = 2
theorem solve_quad_eq1 : ∃ x : ℝ, x^2 = 2 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by
  sorry

-- Problem 2: Prove the solutions to 4x^2 - 1 = 0
theorem solve_quad_eq2 : ∃ x : ℝ, 4 * x^2 - 1 = 0 ∧ (x = 1/2 ∨ x = -1/2) :=
by
  sorry

-- Problem 3: Prove the solutions to (x-1)^2 - 4 = 0
theorem solve_quad_eq3 : ∃ x : ℝ, (x - 1)^2 - 4 = 0 ∧ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 4: Prove the solutions to 12 * (3 - x)^2 - 48 = 0
theorem solve_quad_eq4 : ∃ x : ℝ, 12 * (3 - x)^2 - 48 = 0 ∧ (x = 1 ∨ x = 5) :=
by
  sorry

end solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l228_228427


namespace triangular_number_19_l228_228310

def triangular_number (n : Nat) : Nat :=
  (n + 1) * (n + 2) / 2

theorem triangular_number_19 : triangular_number 19 = 210 := by
  sorry

end triangular_number_19_l228_228310


namespace cos_value_l228_228926

variable (α : ℝ)

theorem cos_value (h : Real.sin (π / 4 + α) = 2 / 3) : Real.cos (π / 4 - α) = 2 / 3 := 
by 
  sorry 

end cos_value_l228_228926


namespace solve_for_x_l228_228577

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l228_228577


namespace no_prime_divisible_by_57_l228_228686

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Given that 57 is equal to 3 times 19.--/
theorem no_prime_divisible_by_57 : ∀ p : ℕ, is_prime p → ¬ (57 ∣ p) :=
by
  sorry

end no_prime_divisible_by_57_l228_228686


namespace total_candidates_l228_228106

theorem total_candidates (T : ℝ) 
  (h1 : 0.45 * T = T * 0.45)
  (h2 : 0.38 * T = T * 0.38)
  (h3 : 0.22 * T = T * 0.22)
  (h4 : 0.12 * T = T * 0.12)
  (h5 : 0.09 * T = T * 0.09)
  (h6 : 0.10 * T = T * 0.10)
  (h7 : 0.05 * T = T * 0.05)
  (h_passed_english_alone : T - (0.45 * T - 0.12 * T - 0.10 * T + 0.05 * T) = 720) :
  T = 1000 :=
by
  sorry

end total_candidates_l228_228106


namespace find_first_term_arithmetic_sequence_l228_228275

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l228_228275


namespace vector_parallel_find_k_l228_228679

theorem vector_parallel_find_k (k : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h₁ : a = (3 * k + 1, 2)) 
  (h₂ : b = (k, 1)) 
  (h₃ : ∃ c : ℝ, a = c • b) : k = -1 := 
by 
  sorry

end vector_parallel_find_k_l228_228679


namespace man_speed_in_still_water_l228_228200

theorem man_speed_in_still_water
  (speed_of_current_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_meters : ℝ)
  (speed_of_current_ms : ℝ := speed_of_current_kmph * (1000 / 3600))
  (speed_downstream : ℝ := distance_meters / time_seconds) :
  speed_of_current_kmph = 3 →
  time_seconds = 13.998880089592832 →
  distance_meters = 70 →
  (speed_downstream = (25 / 6)) →
  (speed_downstream - speed_of_current_ms) * (3600 / 1000) = 15 :=
by
  intros h_speed_current h_time h_distance h_downstream
  sorry

end man_speed_in_still_water_l228_228200


namespace Edmund_can_wrap_15_boxes_every_3_days_l228_228219

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end Edmund_can_wrap_15_boxes_every_3_days_l228_228219


namespace crayons_left_l228_228417

def initial_crayons : ℕ := 253
def lost_or_given_away_crayons : ℕ := 70
def remaining_crayons : ℕ := 183

theorem crayons_left (initial_crayons : ℕ) (lost_or_given_away_crayons : ℕ) (remaining_crayons : ℕ) :
  initial_crayons - lost_or_given_away_crayons = remaining_crayons :=
by {
  sorry
}

end crayons_left_l228_228417


namespace danny_distance_to_work_l228_228769

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_l228_228769


namespace flower_beds_fraction_l228_228878

noncomputable def area_triangle (leg: ℝ) : ℝ := (leg * leg) / 2
noncomputable def area_rectangle (length width: ℝ) : ℝ := length * width
noncomputable def area_trapezoid (a b height: ℝ) : ℝ := ((a + b) * height) / 2

theorem flower_beds_fraction : 
  ∀ (leg len width a b height total_length: ℝ),
    a = 30 →
    b = 40 →
    height = 6 →
    total_length = 60 →
    leg = 5 →
    len = 20 →
    width = 5 →
    (area_rectangle len width + 2 * area_triangle leg) / (area_trapezoid a b height + area_rectangle len width) = 125 / 310 :=
by
  intros
  sorry

end flower_beds_fraction_l228_228878


namespace exists_a_b_l228_228569

theorem exists_a_b (n : ℕ) (hn : 0 < n) : ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := by
  sorry

end exists_a_b_l228_228569


namespace max_value_quadratic_l228_228320

theorem max_value_quadratic (r : ℝ) : 
  ∃ M, (∀ r, -3 * r^2 + 36 * r - 9 ≤ M) ∧ M = 99 :=
sorry

end max_value_quadratic_l228_228320


namespace triangle_dimensions_l228_228813

-- Define the problem in Lean 4
theorem triangle_dimensions (a m : ℕ) (h₁ : a = m + 4)
  (h₂ : (a + 12) * (m + 12) = 10 * a * m) : 
  a = 12 ∧ m = 8 := 
by
  sorry

end triangle_dimensions_l228_228813


namespace smallest_pos_multiple_6_15_is_30_l228_228505

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l228_228505


namespace cost_of_bananas_l228_228061

/-- We are given that the rate of bananas is $6 per 3 kilograms. -/
def rate_per_3_kg : ℝ := 6

/-- We need to find the cost for 12 kilograms of bananas. -/
def weight_in_kg : ℝ := 12

/-- We are asked to prove that the cost of 12 kilograms of bananas is $24. -/
theorem cost_of_bananas (rate_per_3_kg weight_in_kg : ℝ) :
  (weight_in_kg / 3) * rate_per_3_kg = 24 :=
by
  sorry

end cost_of_bananas_l228_228061


namespace toms_final_stamp_count_l228_228022

-- Definitions of the given conditions

def initial_stamps : ℕ := 3000
def mike_gift : ℕ := 17
def harry_gift : ℕ := 2 * mike_gift + 10
def sarah_gift : ℕ := 3 * mike_gift - 5
def damaged_stamps : ℕ := 37

-- Statement of the goal
theorem toms_final_stamp_count :
  initial_stamps + mike_gift + harry_gift + sarah_gift - damaged_stamps = 3070 :=
by
  sorry

end toms_final_stamp_count_l228_228022


namespace original_number_l228_228944

theorem original_number (x : ℕ) (h : x / 3 = 42) : x = 126 :=
sorry

end original_number_l228_228944


namespace number_of_valid_bases_l228_228149

-- Define the main problem conditions
def base_representation_digits (n b : ℕ) := 
  let digits := (n.to_digits b).length 
  digits

def valid_bases_for_base10_256 (b : ℕ) : Prop := 
  b ≥ 2 ∧ base_representation_digits 256 b = 4

-- Theorem statement
theorem number_of_valid_bases : 
  finset.card (finset.filter valid_bases_for_base10_256 (finset.range (256 + 1))) = 2 := 
sorry

end number_of_valid_bases_l228_228149


namespace tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l228_228665

-- Condition: Given tan(α) = 2
variable (α : ℝ) (h₀ : Real.tan α = 2)

-- Statement (1): Prove tan(2α + π/4) = 9
theorem tan_double_alpha_plus_pi_over_four :
  Real.tan (2 * α + Real.pi / 4) = 9 := by
  sorry

-- Statement (2): Prove (6 sin α + cos α) / (3 sin α - 2 cos α) = 13 / 4
theorem sin_cos_fraction :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 := by
  sorry

end tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l228_228665


namespace find_hanyoung_weight_l228_228099

variable (H J : ℝ)

def hanyoung_is_lighter (H J : ℝ) : Prop := H = J - 4
def sum_of_weights (H J : ℝ) : Prop := H + J = 88

theorem find_hanyoung_weight (H J : ℝ) (h1 : hanyoung_is_lighter H J) (h2 : sum_of_weights H J) : H = 42 :=
by
  sorry

end find_hanyoung_weight_l228_228099


namespace cody_books_reading_l228_228487

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end cody_books_reading_l228_228487


namespace certain_number_modulo_l228_228176

theorem certain_number_modulo (x : ℕ) : (57 * x) % 8 = 7 ↔ x = 1 := by
  sorry

end certain_number_modulo_l228_228176


namespace range_of_x_plus_y_l228_228236

open Real

theorem range_of_x_plus_y (x y : ℝ) (h : x - sqrt (x + 1) = sqrt (y + 1) - y) :
  -sqrt 5 + 1 ≤ x + y ∧ x + y ≤ sqrt 5 + 1 :=
by sorry

end range_of_x_plus_y_l228_228236


namespace units_digit_sum_squares_of_first_2011_odd_integers_l228_228325

-- Define the relevant conditions and given parameters
def first_n_odd_integers (n : ℕ) : List ℕ := List.range' 1 (2*n) (λ k, 2*k - 1)

def units_digit (n : ℕ) : ℕ := n % 10

def square_units_digit (n : ℕ) : ℕ := units_digit (n * n)

-- Prove the units digit of the sum of squares of the first 2011 odd positive integers
theorem units_digit_sum_squares_of_first_2011_odd_integers : 
  units_digit (List.sum (List.map (λ x, x * x) (first_n_odd_integers 2011))) = 1 :=
by
  -- Sorry skips the proof
  sorry

end units_digit_sum_squares_of_first_2011_odd_integers_l228_228325


namespace transform_polynomial_l228_228785

variables {x y : ℝ}

theorem transform_polynomial (h : y = x - 1 / x) :
  (x^6 + x^5 - 5 * x^4 + 2 * x^3 - 5 * x^2 + x + 1 = 0) ↔ (x^2 * (y^2 + y - 3) = 0) :=
sorry

end transform_polynomial_l228_228785


namespace arithmetic_seq_sum_l228_228695

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 5 + a 6 + a 7 = 1) : a 3 + a 9 = 2 / 3 :=
sorry

end arithmetic_seq_sum_l228_228695


namespace smallest_pos_multiple_6_15_is_30_l228_228503

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l228_228503


namespace probability_of_D_l228_228458

theorem probability_of_D (P : Type) (A B C D : P) 
  (pA pB pC pD : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/3) 
  (hC : pC = 1/6) 
  (hSum : pA + pB + pC + pD = 1) :
  pD = 1/4 :=
by 
  sorry

end probability_of_D_l228_228458


namespace emilee_earns_25_l228_228262

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l228_228262


namespace total_cupcakes_correct_l228_228019

def cupcakes_per_event : ℝ := 96.0
def num_events : ℝ := 8.0
def total_cupcakes : ℝ := cupcakes_per_event * num_events

theorem total_cupcakes_correct : total_cupcakes = 768.0 :=
by
  unfold total_cupcakes
  unfold cupcakes_per_event
  unfold num_events
  sorry

end total_cupcakes_correct_l228_228019


namespace jim_catches_bob_in_20_minutes_l228_228205

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end jim_catches_bob_in_20_minutes_l228_228205


namespace sequence_sum_difference_l228_228445

def sum_odd (n : ℕ) : ℕ := n * n
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sequence_sum_difference :
  sum_even 1500 - sum_odd 1500 + sum_triangular 1500 = 563628000 :=
by
  sorry

end sequence_sum_difference_l228_228445


namespace marks_lost_per_wrong_answer_l228_228392

theorem marks_lost_per_wrong_answer
    (total_questions : ℕ)
    (correct_questions : ℕ)
    (total_marks : ℕ)
    (marks_per_correct : ℕ)
    (marks_lost : ℕ)
    (x : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_questions = 44)
    (h3 : total_marks = 160)
    (h4 : marks_per_correct = 4)
    (h5 : marks_lost = 176 - total_marks)
    (h6 : marks_lost = x * (total_questions - correct_questions)) :
    x = 1 := by
  sorry

end marks_lost_per_wrong_answer_l228_228392


namespace function_above_x_axis_l228_228092

theorem function_above_x_axis (m : ℝ) : 
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) ↔ m < 2 + 2 * Real.sqrt 2 :=
sorry

end function_above_x_axis_l228_228092


namespace computation_l228_228650

theorem computation :
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2 / 3) = 41.65 :=
by
  sorry

end computation_l228_228650


namespace mikes_salary_l228_228912

theorem mikes_salary
  (fred_salary : ℝ)
  (mike_salary_increase_percent : ℝ)
  (mike_salary_factor : ℝ)
  (fred_salary_val : fred_salary = 1000)
  (mike_salary_factor_val : mike_salary_factor = 10)
  (mike_salary_increase_val : mike_salary_increase_percent = 40)
  : (10000 * (1 + mike_salary_increase_percent / 100)) = 14000 := 
by
  rw [fred_salary_val, mike_salary_factor_val, mike_salary_increase_val]
  norm_num
  sorry  -- Proof omitted

end mikes_salary_l228_228912


namespace volume_of_hemisphere_l228_228630

theorem volume_of_hemisphere (d : ℝ) (h : d = 10) : 
  let r := d / 2
  let V := (2 / 3) * π * r^3
  V = 250 / 3 * π := by
sorry

end volume_of_hemisphere_l228_228630


namespace prove_A_annual_savings_l228_228811

noncomputable def employee_A_annual_savings
  (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02) : ℝ :=
  let total_deductions := tax_rate + pension_rate + healthcare_rate
  let Income_after_deductions := A_income * (1 - total_deductions)
  let annual_savings := 12 * Income_after_deductions
  annual_savings

theorem prove_A_annual_savings : 
  ∀ (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02),
  employee_A_annual_savings A_income B_income C_income D_income C_income_val income_ratio tax_rate pension_rate healthcare_rate tax_rate_val pension_rate_val healthcare_rate_val = 232400.16 :=
by
  sorry

end prove_A_annual_savings_l228_228811


namespace difference_of_scores_correct_l228_228305

-- Define the parameters
def num_innings : ℕ := 46
def batting_avg : ℕ := 63
def highest_score : ℕ := 248
def reduced_avg : ℕ := 58
def excluded_innings : ℕ := num_innings - 2

-- Necessary calculations
def total_runs := batting_avg * num_innings
def reduced_total_runs := reduced_avg * excluded_innings
def sum_highest_lowest := total_runs - reduced_total_runs
def lowest_score := sum_highest_lowest - highest_score

-- The correct answer to prove
def expected_difference := highest_score - lowest_score
def correct_answer := 150

-- Define the proof problem
theorem difference_of_scores_correct :
  expected_difference = correct_answer := by
  sorry

end difference_of_scores_correct_l228_228305


namespace probability_green_or_blue_l228_228715

-- Define the properties of the 10-sided die
def total_faces : ℕ := 10
def red_faces : ℕ := 4
def yellow_faces : ℕ := 3
def green_faces : ℕ := 2
def blue_faces : ℕ := 1

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := green_faces + blue_faces

-- Define the probability function
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- The theorem to prove
theorem probability_green_or_blue :
  probability favorable_outcomes total_faces = 3 / 10 :=
by
  sorry

end probability_green_or_blue_l228_228715


namespace volume_of_rice_pile_l228_228993

theorem volume_of_rice_pile
  (arc_length_bottom : ℝ)
  (height : ℝ)
  (one_fourth_cone : ℝ)
  (approx_pi : ℝ)
  (h_arc : arc_length_bottom = 8)
  (h_height : height = 5)
  (h_one_fourth_cone : one_fourth_cone = 1/4)
  (h_approx_pi : approx_pi = 3) :
  ∃ V : ℝ, V = one_fourth_cone * (1 / 3) * π * (16^2 / π^2) * height :=
by
  sorry

end volume_of_rice_pile_l228_228993


namespace number_of_bottom_row_bricks_l228_228614

theorem number_of_bottom_row_bricks :
  ∃ (x : ℕ), (x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) ∧ x = 22 :=
by 
  sorry

end number_of_bottom_row_bricks_l228_228614


namespace discriminant_quadratic_eq_l228_228589

theorem discriminant_quadratic_eq : 
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  Δ = 33 :=
by
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  exact sorry

end discriminant_quadratic_eq_l228_228589


namespace second_quadrant_necessary_not_sufficient_l228_228136

open Classical

-- Definitions
def isSecondQuadrant (α : ℝ) : Prop := 90 < α ∧ α < 180
def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ 180 < α ∧ α < 270

-- The theorem statement
theorem second_quadrant_necessary_not_sufficient (α : ℝ) :
  (isSecondQuadrant α → isObtuseAngle α) ∧ ¬(isSecondQuadrant α ↔ isObtuseAngle α) :=
by
  sorry

end second_quadrant_necessary_not_sufficient_l228_228136


namespace largest_possible_M_l228_228780

theorem largest_possible_M (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_cond : x * y + y * z + z * x = 1) :
    ∃ M, ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = 1 → 
    (x / (1 + yz/x) + y / (1 + zx/y) + z / (1 + xy/z) ≥ M) → 
        M = 3 / (Real.sqrt 3 + 1) :=
by
  sorry        

end largest_possible_M_l228_228780


namespace range_of_m_l228_228541

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x^2 - x - m = 0) : m ≥ -1/4 :=
by
  sorry

end range_of_m_l228_228541


namespace geometry_problem_l228_228109

theorem geometry_problem
  (A B C D E : Type*)
  (BAC ABC ACB ADE ADC AEB DEB CDE : ℝ)
  (h₁ : ABC = 72)
  (h₂ : ACB = 90)
  (h₃ : CDE = 36)
  (h₄ : ADC = 180)
  (h₅ : AEB = 180) :
  DEB = 162 :=
sorry

end geometry_problem_l228_228109


namespace domain_of_function_l228_228007

theorem domain_of_function :
  ∀ x : ℝ, (1 / (1 - x) ≥ 0 ∧ 1 - x ≠ 0) ↔ (x < 1) :=
by
  sorry

end domain_of_function_l228_228007


namespace distinct_value_expression_l228_228615

def tri (a b : ℕ) : ℕ := min a b
def nabla (a b : ℕ) : ℕ := max a b

theorem distinct_value_expression (x : ℕ) : (nabla 5 (nabla 4 (tri x 4))) = 5 := 
by
  sorry

end distinct_value_expression_l228_228615


namespace factor_quadratic_expression_l228_228154

theorem factor_quadratic_expression (a b : ℤ) :
  (∃ a b : ℤ, (5 * a + 5 * b = -125) ∧ (a * b = -100) → (a + b = -25)) → (25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) := 
by
  sorry

end factor_quadratic_expression_l228_228154


namespace find_xy_l228_228558

theorem find_xy (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p * (x - y) = x * y ↔ (x, y) = (p^2 - p, p + 1) := by
  sorry

end find_xy_l228_228558


namespace calculateRemainingMoney_l228_228521

def initialAmount : ℝ := 100
def actionFiguresCount : ℕ := 3
def actionFigureOriginalPrice : ℝ := 12
def actionFigureDiscount : ℝ := 0.25
def boardGamesCount : ℕ := 2
def boardGamePrice : ℝ := 11
def puzzleSetsCount : ℕ := 4
def puzzleSetPrice : ℝ := 6
def salesTax : ℝ := 0.05

theorem calculateRemainingMoney :
  initialAmount - (
    (actionFigureOriginalPrice * (1 - actionFigureDiscount) * actionFiguresCount) +
    (boardGamePrice * boardGamesCount) +
    (puzzleSetPrice * puzzleSetsCount)
  ) * (1 + salesTax) = 23.35 :=
by
  sorry

end calculateRemainingMoney_l228_228521


namespace perimeter_of_new_figure_is_correct_l228_228848

-- Define the given conditions
def original_horizontal_segments := 16
def original_vertical_segments := 10
def original_side_length := 1
def new_side_length := 2

-- Define total lengths calculations
def total_horizontal_length (new_side_length original_horizontal_segments : ℕ) : ℕ :=
  original_horizontal_segments * new_side_length

def total_vertical_length (new_side_length original_vertical_segments : ℕ) : ℕ :=
  original_vertical_segments * new_side_length

-- Formulate the main theorem
theorem perimeter_of_new_figure_is_correct :
  total_horizontal_length new_side_length original_horizontal_segments + 
  total_vertical_length new_side_length original_vertical_segments = 52 := by
  sorry

end perimeter_of_new_figure_is_correct_l228_228848


namespace field_division_l228_228467

theorem field_division (A B : ℝ) (h1 : A + B = 700) (h2 : B - A = (1 / 5) * ((A + B) / 2)) : A = 315 :=
by
  sorry

end field_division_l228_228467


namespace smallest_common_multiple_l228_228517

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l228_228517


namespace smallest_multiple_l228_228508

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l228_228508


namespace paper_clips_in_morning_l228_228958

variable (p : ℕ) (used left : ℕ)

theorem paper_clips_in_morning (h1 : left = 26) (h2 : used = 59) (h3 : left = p - used) : p = 85 :=
by
  sorry

end paper_clips_in_morning_l228_228958


namespace cos_A_eq_find_a_l228_228956

variable {A B C a b c : ℝ}

-- Proposition 1: If in triangle ABC, b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then cos A = sqrt 6 / 4
theorem cos_A_eq (h : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) : Real.cos A = Real.sqrt 6 / 4 :=
sorry

-- Proposition 2: Given b = sqrt 6, B = 2 * A, and b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then a = 2
theorem find_a (h1 : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) (h2 : B = 2 * A) (h3 : b = Real.sqrt 6) : a = 2 :=
sorry

end cos_A_eq_find_a_l228_228956


namespace distance_from_A_to_B_l228_228977

theorem distance_from_A_to_B (d C1A C1B C2A C2B : ℝ) (h1 : C1A + C1B = d)
  (h2 : C2A + C2B = d) (h3 : (C1A = 2 * C1B) ∨ (C1B = 2 * C1A)) 
  (h4 : (C2A = 3 * C2B) ∨ (C2B = 3 * C2A))
  (h5 : |C2A - C1A| = 10) : d = 120 ∨ d = 24 :=
sorry

end distance_from_A_to_B_l228_228977


namespace ellie_loan_difference_l228_228658

noncomputable def principal : ℝ := 8000
noncomputable def simple_rate : ℝ := 0.10
noncomputable def compound_rate : ℝ := 0.08
noncomputable def time : ℝ := 5
noncomputable def compounding_periods : ℝ := 1

noncomputable def simple_interest_total (P r t : ℝ) : ℝ :=
  P + (P * r * t)

noncomputable def compound_interest_total (P r t n : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem ellie_loan_difference :
  (compound_interest_total principal compound_rate time compounding_periods) -
  (simple_interest_total principal simple_rate time) = -245.36 := 
  by sorry

end ellie_loan_difference_l228_228658


namespace proof_of_k_values_l228_228895

noncomputable def problem_statement : Prop :=
  ∀ k : ℝ,
    (∃ a b : ℝ, (6 * a^2 + 5 * a + k = 0 ∧ 6 * b^2 + 5 * b + k = 0 ∧ a ≠ b ∧
    |a - b| = 3 * (a^2 + b^2))) ↔ (k = 1 ∨ k = -20.717)

theorem proof_of_k_values : problem_statement :=
by sorry

end proof_of_k_values_l228_228895


namespace range_of_a_l228_228948

-- Define the conditions
def line1 (a x y : ℝ) : Prop := a * x + y - 4 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- The main theorem to state
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, line1 a x y ∧ line2 x y ∧ first_quadrant x y) ↔ -1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l228_228948


namespace smallest_a_satisfies_sin_condition_l228_228129

open Real

theorem smallest_a_satisfies_sin_condition :
  ∃ (a : ℝ), (∀ x : ℤ, sin (a * x + 0) = sin (45 * x)) ∧ 0 ≤ a ∧ ∀ b : ℝ, (∀ x : ℤ, sin (b * x + 0) = sin (45 * x)) ∧ 0 ≤ b → 45 ≤ b :=
by
  -- To be proved.
  sorry

end smallest_a_satisfies_sin_condition_l228_228129


namespace triangle_is_isosceles_l228_228692

theorem triangle_is_isosceles
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * c * Real.cos B)
    (h2 : b = c * Real.cos A) 
    (h3 : c = a * Real.cos C) 
    : a = b := 
sorry

end triangle_is_isosceles_l228_228692


namespace tangent_circles_radii_l228_228185

noncomputable def radii_of_tangent_circles (R r : ℝ) (h : R > r) : Set ℝ :=
  { x | x = (R * r) / ((Real.sqrt R + Real.sqrt r)^2) ∨ x = (R * r) / ((Real.sqrt R - Real.sqrt r)^2) }

theorem tangent_circles_radii (R r : ℝ) (h : R > r) :
  ∃ x, x ∈ radii_of_tangent_circles R r h := sorry

end tangent_circles_radii_l228_228185


namespace age_problem_solution_l228_228782

theorem age_problem_solution :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
  a1 + a2 + a3 = 54 ∧
  a5 - a4 = 5 ∧
  a3 + a4 + a5 = 78 ∧
  a2 - a1 = 7 ∧
  a1 + a5 = 44 ∧
  a1 = 13 ∧
  a2 = 20 ∧
  a3 = 21 ∧
  a4 = 26 ∧
  a5 = 31 :=
by
  -- We should skip the implementation because the solution is provided in the original problem.
  sorry

end age_problem_solution_l228_228782


namespace order_of_reading_amounts_l228_228694

variable (a b c d : ℝ)

theorem order_of_reading_amounts (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by
  sorry

end order_of_reading_amounts_l228_228694


namespace intersection_A_B_l228_228924

open Set

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def B : Set ℝ := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x} :=
by
  sorry

end intersection_A_B_l228_228924


namespace uncovered_area_frame_l228_228596

def length_frame : ℕ := 40
def width_frame : ℕ := 32
def length_photo : ℕ := 32
def width_photo : ℕ := 28

def area_frame (l_f w_f : ℕ) : ℕ := l_f * w_f
def area_photo (l_p w_p : ℕ) : ℕ := l_p * w_p

theorem uncovered_area_frame :
  area_frame length_frame width_frame - area_photo length_photo width_photo = 384 :=
by
  sorry

end uncovered_area_frame_l228_228596


namespace qinJiushao_value_l228_228718

/-- A specific function f(x) with given a and b -/
def f (x : ℤ) : ℤ :=
  x^5 + 47 * x^4 - 37 * x^2 + 1

/-- Qin Jiushao algorithm to find V3 at x = -1 -/
def qinJiushao (x : ℤ) : ℤ :=
  let V0 := 1
  let V1 := V0 * x + 47
  let V2 := V1 * x + 0
  let V3 := V2 * x - 37
  V3

theorem qinJiushao_value :
  qinJiushao (-1) = 9 :=
by
  sorry

end qinJiushao_value_l228_228718


namespace least_sum_four_primes_gt_10_l228_228334

theorem least_sum_four_primes_gt_10 : 
  ∃ (p1 p2 p3 p4 : ℕ), 
    p1 > 10 ∧ p2 > 10 ∧ p3 > 10 ∧ p4 > 10 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1 + p2 + p3 + p4 = 60 ∧
    ∀ (q1 q2 q3 q4 : ℕ), 
      q1 > 10 ∧ q2 > 10 ∧ q3 > 10 ∧ q4 > 10 ∧ 
      Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ Nat.Prime q4 ∧
      q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 →
      q1 + q2 + q3 + q4 ≥ 60 :=
by
  sorry

end least_sum_four_primes_gt_10_l228_228334


namespace sqrt_domain_l228_228857

theorem sqrt_domain (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
sorry

end sqrt_domain_l228_228857


namespace find_a7_l228_228490

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l228_228490


namespace sum_xyz_eq_neg7_l228_228001

theorem sum_xyz_eq_neg7 (x y z : ℝ)
  (h1 : x = y + z + 2)
  (h2 : y = z + x + 1)
  (h3 : z = x + y + 4) :
  x + y + z = -7 :=
by
  sorry

end sum_xyz_eq_neg7_l228_228001


namespace find_k_l228_228197

-- Define the function f as described in the problem statement
def f (n : ℕ) : ℕ := 
  if n % 2 = 1 then 
    n + 3 
  else 
    n / 2

theorem find_k (k : ℕ) (h_odd : k % 2 = 1) : f (f (f k)) = k → k = 1 :=
by {
  sorry
}

end find_k_l228_228197


namespace nth_odd_and_sum_first_n_odds_l228_228736

noncomputable def nth_odd (n : ℕ) : ℕ := 2 * n - 1

noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n ^ 2

theorem nth_odd_and_sum_first_n_odds :
  nth_odd 100 = 199 ∧ sum_first_n_odds 100 = 10000 :=
by
  sorry

end nth_odd_and_sum_first_n_odds_l228_228736


namespace probability_of_second_less_than_first_is_two_fifths_l228_228456

variable (Ω : Type) [Fintype Ω] [UniformProbabilityMeasure Ω]

-- Define the event spaces
noncomputable def draws : List (ℕ × ℕ) := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                                            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                                            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                                            (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
                                            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]

noncomputable def event_count (event : (ℕ × ℕ) → Prop) : ℕ :=
  (draws.filter event).length

-- Define the specific event
def second_less_than_first (pair: ℕ × ℕ) : Prop := pair.2 < pair.1

-- Calculate the probability
noncomputable def probability_second_less_than_first : ℚ :=
  (event_count Ω second_less_than_first : ℚ) / (Fintype.card Ω * Fintype.card Ω : ℚ)

theorem probability_of_second_less_than_first_is_two_fifths :
  probability_second_less_than_first = 2 / 5 := by
  sorry

end probability_of_second_less_than_first_is_two_fifths_l228_228456


namespace geometric_sum_n_equals_4_l228_228312

def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def S (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))
def sum_value : ℚ := 26 / 81

theorem geometric_sum_n_equals_4 (n : ℕ) (h : S n = sum_value) : n = 4 :=
by sorry

end geometric_sum_n_equals_4_l228_228312


namespace find_smallest_x_satisfying_condition_l228_228905

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l228_228905


namespace f_20_equals_97_l228_228829

noncomputable def f_rec (f : ℕ → ℝ) (n : ℕ) := (2 * f n + n) / 2

theorem f_20_equals_97 (f : ℕ → ℝ) (h₁ : f 1 = 2)
    (h₂ : ∀ n : ℕ, f (n + 1) = f_rec f n) : 
    f 20 = 97 :=
sorry

end f_20_equals_97_l228_228829


namespace option_C_correct_l228_228611

-- Define the base a and natural numbers m and n for exponents
variables {a : ℕ} {m n : ℕ}

-- Lean statement to prove (a^5)^3 = a^(5 * 3)
theorem option_C_correct : (a^5)^3 = a^(5 * 3) := 
by sorry

end option_C_correct_l228_228611


namespace solution_set_of_xf_gt_0_l228_228464

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_ineq : ∀ x : ℝ, x > 0 → f x < x * (deriv f x)
axiom f_at_one : f 1 = 0

theorem solution_set_of_xf_gt_0 : {x : ℝ | x * f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_xf_gt_0_l228_228464


namespace expression_B_between_2_and_3_l228_228784

variable (a b : ℝ)
variable (h : 3 * a = 5 * b)

theorem expression_B_between_2_and_3 : 2 < (|a + b| / b) ∧ (|a + b| / b) < 3 :=
by sorry

end expression_B_between_2_and_3_l228_228784


namespace common_ratio_of_geometric_sequence_is_4_l228_228232

theorem common_ratio_of_geometric_sequence_is_4 
  (a_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ) 
  (d : ℝ) 
  (h₁ : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h₂ : d ≠ 0)
  (h₃ : (a_n 3)^2 = (a_n 2) * (a_n 7)) :
  b_n 2 / b_n 1 = 4 :=
sorry

end common_ratio_of_geometric_sequence_is_4_l228_228232


namespace slope_of_tangent_line_l228_228028

theorem slope_of_tangent_line 
  (center point : ℝ × ℝ) 
  (h_center : center = (5, 3)) 
  (h_point : point = (8, 8)) 
  : (∃ m : ℚ, m = -3/5) :=
sorry

end slope_of_tangent_line_l228_228028


namespace range_of_abscissa_of_P_l228_228672

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

end range_of_abscissa_of_P_l228_228672


namespace geom_sum_eq_six_l228_228670

variable (a : ℕ → ℝ)
variable (r : ℝ) -- common ratio for geometric sequence

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a (n + 1) > 0
axiom given_eq : a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36

-- Proof statement
theorem geom_sum_eq_six : a 2 + a 5 = 6 :=
sorry

end geom_sum_eq_six_l228_228670


namespace min_value_expression_l228_228942

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) :
  ∃ (M : ℝ), M = (2 : ℝ) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → ((y / x) + (3 / (y + 1)) ≥ M)) :=
by
  use 2
  sorry

end min_value_expression_l228_228942


namespace ryegrass_percent_of_mixture_l228_228424

noncomputable def mixture_percent_ryegrass (X_rye Y_rye portion_X : ℝ) : ℝ :=
  let portion_Y := 1 - portion_X
  let total_rye := (X_rye * portion_X) + (Y_rye * portion_Y)
  total_rye * 100

theorem ryegrass_percent_of_mixture :
  let X_rye := 40 / 100 
  let Y_rye := 25 / 100
  let portion_X := 1 / 3
  mixture_percent_ryegrass X_rye Y_rye portion_X = 30 :=
by
  sorry

end ryegrass_percent_of_mixture_l228_228424


namespace total_cost_is_90_l228_228700

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end total_cost_is_90_l228_228700


namespace common_ratio_of_geometric_sequence_l228_228955

variable (a₁ q : ℝ)

def geometric_sequence (n : ℕ) := a₁ * q^n

theorem common_ratio_of_geometric_sequence
  (h_sum : geometric_sequence a₁ q 0 + geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 = 3 * a₁) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_sequence_l228_228955


namespace vasya_number_l228_228733

theorem vasya_number (a b c : ℕ) (h1 : 100 ≤ 100*a + 10*b + c) (h2 : 100*a + 10*b + c < 1000) 
  (h3 : a + c = 1) (h4 : a * b = 4) (h5 : a ≠ 0) : 100*a + 10*b + c = 140 :=
by
  sorry

end vasya_number_l228_228733


namespace inscribed_cube_volume_l228_228636

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end inscribed_cube_volume_l228_228636


namespace pencils_on_desk_l228_228723

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end pencils_on_desk_l228_228723


namespace new_rectangle_dimensions_l228_228347

theorem new_rectangle_dimensions (l w : ℕ) (h_l : l = 12) (h_w : w = 10) :
  ∃ l' w' : ℕ, l' = l ∧ w' = w / 2 ∧ l' = 12 ∧ w' = 5 :=
by
  sorry

end new_rectangle_dimensions_l228_228347


namespace time_to_fill_pot_l228_228013

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end time_to_fill_pot_l228_228013


namespace negation_of_exists_x_quad_eq_zero_l228_228309

theorem negation_of_exists_x_quad_eq_zero :
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0 ↔ ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0 :=
by sorry

end negation_of_exists_x_quad_eq_zero_l228_228309


namespace distance_from_home_to_school_l228_228773

theorem distance_from_home_to_school
  (x y : ℝ)
  (h1 : x = y / 3)
  (h2 : x = (y + 18) / 5) : x = 9 := 
by
  sorry

end distance_from_home_to_school_l228_228773


namespace prime_divisible_by_57_is_zero_l228_228685

open Nat

theorem prime_divisible_by_57_is_zero :
  (∀ p, Prime p → (57 ∣ p) → False) :=
by
  intro p hp hdiv
  have h57 : 57 = 3 * 19 := by norm_num
  have h1 : p = 57 ∨ p = 3 ∨ p = 19 := sorry
  have hp1 : p ≠ 57 := sorry
  have hp2 : p ≠ 3 := sorry
  have hp3 : p ≠ 19 := sorry
  exact Or.elim h1 hp1 (Or.elim hp2 hp3)


end prime_divisible_by_57_is_zero_l228_228685


namespace new_mean_correct_l228_228852

-- Define the original condition data
def initial_mean : ℝ := 42
def total_numbers : ℕ := 60
def discard1 : ℝ := 50
def discard2 : ℝ := 60
def increment : ℝ := 2

-- A function representing the new arithmetic mean
noncomputable def new_arithmetic_mean : ℝ :=
  let initial_sum := initial_mean * total_numbers
  let sum_after_discard := initial_sum - (discard1 + discard2)
  let sum_after_increment := sum_after_discard + (increment * (total_numbers - 2))
  sum_after_increment / (total_numbers - 2)

-- The theorem statement
theorem new_mean_correct : new_arithmetic_mean = 43.55 :=
by 
  sorry

end new_mean_correct_l228_228852


namespace average_of_w_x_z_eq_one_sixth_l228_228991

open Real

variable {w x y z t : ℝ}

theorem average_of_w_x_z_eq_one_sixth
  (h1 : 3 / w + 3 / x + 3 / z = 3 / (y + t))
  (h2 : w * x * z = y + t)
  (h3 : w * z + x * t + y * z = 3 * w + 3 * x + 3 * z) :
  (w + x + z) / 3 = 1 / 6 :=
by 
  sorry

end average_of_w_x_z_eq_one_sixth_l228_228991


namespace find_first_term_l228_228269

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l228_228269


namespace total_pictures_l228_228289

-- Definitions based on problem conditions
def Randy_pictures : ℕ := 5
def Peter_pictures : ℕ := Randy_pictures + 3
def Quincy_pictures : ℕ := Peter_pictures + 20
def Susan_pictures : ℕ := 2 * Quincy_pictures - 7
def Thomas_pictures : ℕ := Randy_pictures ^ 3

-- The proof statement
theorem total_pictures : Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by
  sorry

end total_pictures_l228_228289


namespace largest_possible_a_l228_228969

theorem largest_possible_a :
  ∀ (a b c d : ℕ), a < 3 * b ∧ b < 4 * c ∧ c < 5 * d ∧ d < 80 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d → a ≤ 4724 := by
  sorry

end largest_possible_a_l228_228969


namespace tangent_ellipse_hyperbola_l228_228590

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end tangent_ellipse_hyperbola_l228_228590


namespace pot_filling_time_l228_228011

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end pot_filling_time_l228_228011


namespace inequality_proof_l228_228561

theorem inequality_proof
  (a b x y z : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l228_228561


namespace smallest_multiple_l228_228507

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l228_228507


namespace sandy_final_fish_l228_228843

theorem sandy_final_fish :
  let Initial_fish := 26
  let Bought_fish := 6
  let Given_away_fish := 10
  let Babies_fish := 15
  let Final_fish := Initial_fish + Bought_fish - Given_away_fish + Babies_fish
  Final_fish = 37 :=
by
  sorry

end sandy_final_fish_l228_228843


namespace increase_number_correct_l228_228621

-- Definitions for the problem
def originalNumber : ℕ := 110
def increasePercent : ℝ := 0.5

-- Statement to be proved
theorem increase_number_correct : originalNumber + (originalNumber * increasePercent) = 165 := by
  sorry

end increase_number_correct_l228_228621


namespace value_of_expr_l228_228230

theorem value_of_expr (a : Int) (h : a = -2) : a + 1 = -1 := by
  -- Placeholder for the proof, assuming it's correct
  sorry

end value_of_expr_l228_228230


namespace input_command_is_INPUT_l228_228110

-- Define the commands
def PRINT : String := "PRINT"
def INPUT : String := "INPUT"
def THEN : String := "THEN"
def END : String := "END"

-- Define the properties of each command
def PRINT_is_output (cmd : String) : Prop :=
  cmd = PRINT

def INPUT_is_input (cmd : String) : Prop :=
  cmd = INPUT

def THEN_is_conditional (cmd : String) : Prop :=
  cmd = THEN

def END_is_end (cmd : String) : Prop :=
  cmd = END

-- Theorem stating that INPUT is the command associated with input operation
theorem input_command_is_INPUT : INPUT_is_input INPUT :=
by
  -- Proof goes here
  sorry

end input_command_is_INPUT_l228_228110


namespace find_m_l228_228379

-- Define the vectors a and b
def veca (m : ℝ) : ℝ × ℝ := (m, 4)
def vecb (m : ℝ) : ℝ × ℝ := (m + 4, 1)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition that the dot product of a and b is zero
def are_perpendicular (m : ℝ) : Prop :=
  dot_product (veca m) (vecb m) = 0

-- The goal is to prove that if a and b are perpendicular, then m = -2
theorem find_m (m : ℝ) (h : are_perpendicular m) : m = -2 :=
by {
  -- Proof will be filled here
  sorry
}

end find_m_l228_228379


namespace hyperbola_correct_l228_228104

noncomputable def hyperbola_properties : Prop :=
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  let b := Real.sqrt ((c^2) - (a^2))
  (h + k + a + b = 4 * Real.sqrt 3 + 6)

theorem hyperbola_correct : hyperbola_properties :=
by
  unfold hyperbola_properties
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  have b : ℝ := Real.sqrt ((c^2) - (a^2))
  sorry

end hyperbola_correct_l228_228104


namespace correct_statement_l228_228032

-- Define the conditions as assumptions

/-- Condition 1: To understand the service life of a batch of new energy batteries, a sampling survey can be used. -/
def condition1 : Prop := True

/-- Condition 2: If the probability of winning a lottery is 2%, then buying 50 of these lottery tickets at once will definitely win. -/
def condition2 : Prop := False

/-- Condition 3: If the average of two sets of data, A and B, is the same, SA^2=2.3, SB^2=4.24, then set B is more stable. -/
def condition3 : Prop := False

/-- Condition 4: Rolling a die with uniform density and getting a score of 0 is a certain event. -/
def condition4 : Prop := False

-- The main theorem to prove the correct statement is A
theorem correct_statement : condition1 = True ∧ condition2 = False ∧ condition3 = False ∧ condition4 = False :=
by
  constructor; repeat { try { exact True.intro }; try { exact False.elim (by sorry) } }

end correct_statement_l228_228032


namespace smallest_positive_x_l228_228903

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l228_228903


namespace volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l228_228721

theorem volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron (R r : ℝ) (h : r = R / 3) : 
  (4/3 * π * r^3) / (4/3 * π * R^3) = 1 / 27 :=
by
  sorry

end volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l228_228721


namespace mathematics_equivalent_proof_l228_228826

noncomputable def distinctRealNumbers (a b c d : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d

theorem mathematics_equivalent_proof (a b c d : ℝ)
  (H₀ : distinctRealNumbers a b c d)
  (H₁ : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 :=
sorry

end mathematics_equivalent_proof_l228_228826


namespace mark_new_phone_plan_cost_l228_228834

noncomputable def total_new_plan_cost (old_plan_cost old_internet_cost old_intl_call_cost : ℝ) (percent_increase_plan percent_increase_internet percent_decrease_intl : ℝ) : ℝ :=
  let new_plan_cost := old_plan_cost * (1 + percent_increase_plan)
  let new_internet_cost := old_internet_cost * (1 + percent_increase_internet)
  let new_intl_call_cost := old_intl_call_cost * (1 - percent_decrease_intl)
  new_plan_cost + new_internet_cost + new_intl_call_cost

theorem mark_new_phone_plan_cost :
  let old_plan_cost := 150
  let old_internet_cost := 50
  let old_intl_call_cost := 30
  let percent_increase_plan := 0.30
  let percent_increase_internet := 0.20
  let percent_decrease_intl := 0.15
  total_new_plan_cost old_plan_cost old_internet_cost old_intl_call_cost percent_increase_plan percent_increase_internet percent_decrease_intl = 280.50 :=
by
  sorry

end mark_new_phone_plan_cost_l228_228834


namespace solve_for_M_l228_228793

def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 2 * x + y = 2 ∧ x - y = 1 }

theorem solve_for_M : M = { (1, 0) } := by
  sorry

end solve_for_M_l228_228793


namespace insurance_covers_80_percent_l228_228565

def xray_cost : ℕ := 250
def mri_cost : ℕ := 3 * xray_cost
def total_cost : ℕ := xray_cost + mri_cost
def mike_payment : ℕ := 200
def insurance_coverage : ℕ := total_cost - mike_payment
def insurance_percentage : ℕ := (insurance_coverage * 100) / total_cost

theorem insurance_covers_80_percent : insurance_percentage = 80 := by
  -- Carry out the necessary calculations
  sorry

end insurance_covers_80_percent_l228_228565


namespace find_k_l228_228806

theorem find_k (k : ℝ) (h : (3, 1) ∈ {(x, y) | y = k * x - 2} ∧ k ≠ 0) : k = 1 :=
by sorry

end find_k_l228_228806


namespace plane_hit_probability_l228_228339

theorem plane_hit_probability :
  let P_A : ℝ := 0.3
  let P_B : ℝ := 0.5
  let P_not_A : ℝ := 1 - P_A
  let P_not_B : ℝ := 1 - P_B
  let P_both_miss : ℝ := P_not_A * P_not_B
  let P_plane_hit : ℝ := 1 - P_both_miss
  P_plane_hit = 0.65 :=
by
  sorry

end plane_hit_probability_l228_228339


namespace find_b_amount_l228_228450

theorem find_b_amount (A B : ℝ) (h1 : A + B = 100) (h2 : (3 / 10) * A = (1 / 5) * B) : B = 60 := 
by 
  sorry

end find_b_amount_l228_228450


namespace weekly_spending_l228_228441

-- Definitions based on the conditions outlined in the original problem
def weekly_allowance : ℝ := 50
def hours_per_week : ℕ := 30
def hourly_wage : ℝ := 9
def weeks_per_year : ℕ := 52
def first_year_allowance : ℝ := weekly_allowance * weeks_per_year
def second_year_earnings : ℝ := (hourly_wage * hours_per_week) * weeks_per_year
def total_car_cost : ℝ := 15000
def additional_needed : ℝ := 2000
def total_savings : ℝ := first_year_allowance + second_year_earnings

-- The amount Thomas needs over what he has saved
def total_needed : ℝ := total_savings + additional_needed
def amount_spent_on_self : ℝ := total_needed - total_car_cost
def total_weeks : ℕ := 2 * weeks_per_year

theorem weekly_spending :
  amount_spent_on_self / total_weeks = 35 := by
  sorry

end weekly_spending_l228_228441


namespace pencils_on_desk_l228_228724

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end pencils_on_desk_l228_228724


namespace number_of_bookshelves_l228_228482

-- Definitions based on the conditions
def books_per_shelf : ℕ := 2
def total_books : ℕ := 38

-- Statement to prove
theorem number_of_bookshelves (books_per_shelf total_books : ℕ) : total_books / books_per_shelf = 19 :=
by sorry

end number_of_bookshelves_l228_228482


namespace max_xy_value_l228_228442

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3 * x * y = 2015) : xy <= 403 :=
sorry

end max_xy_value_l228_228442


namespace cos_angle_equiv_370_l228_228896

open Real

noncomputable def find_correct_n : ℕ :=
  sorry

theorem cos_angle_equiv_370 (n : ℕ) (h : 0 ≤ n ∧ n ≤ 180) : cos (n * π / 180) = cos (370 * π / 180) → n = 10 :=
by
  sorry

end cos_angle_equiv_370_l228_228896


namespace football_points_difference_l228_228645

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end football_points_difference_l228_228645


namespace sum_xy_sum_inv_squared_geq_nine_four_l228_228786

variable {x y z : ℝ}

theorem sum_xy_sum_inv_squared_geq_nine_four (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z + z * x) * (1 / (x + y)^2 + 1 / (y + z)^2 + 1 / (z + x)^2) ≥ 9 / 4 :=
by sorry

end sum_xy_sum_inv_squared_geq_nine_four_l228_228786


namespace average_number_of_carnations_l228_228606

-- Define the conditions in Lean
def number_of_bouquet_1 : ℕ := 9
def number_of_bouquet_2 : ℕ := 14
def number_of_bouquet_3 : ℕ := 13
def total_bouquets : ℕ := 3

-- The main statement to be proved
theorem average_number_of_carnations : 
  (number_of_bouquet_1 + number_of_bouquet_2 + number_of_bouquet_3) / total_bouquets = 12 := 
by
  sorry

end average_number_of_carnations_l228_228606


namespace angle_equivalence_l228_228031

theorem angle_equivalence :
  ∃ k : ℤ, -495 + 360 * k = 225 :=
sorry

end angle_equivalence_l228_228031


namespace minimal_fencing_l228_228283

theorem minimal_fencing (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l ≥ 400) : 
  2 * (w + l) = 60 * Real.sqrt 2 :=
by
  sorry

end minimal_fencing_l228_228283


namespace catering_budget_total_l228_228117

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end catering_budget_total_l228_228117


namespace box_height_is_6_l228_228187

-- Defining the problem setup
variables (h : ℝ) (r_large r_small : ℝ)
variables (box_size : ℝ) (n_spheres : ℕ)

-- The conditions of the problem
def rectangular_box :=
  box_size = 5 ∧ r_large = 3 ∧ r_small = 1.5 ∧ n_spheres = 4 ∧
  (∀ k : ℕ, k < n_spheres → 
   ∃ C : ℝ, 
     (C = r_small) ∧ 
     -- Each smaller sphere is tangent to three sides of the box condition
     (C ≤ box_size))

def sphere_tangency (h r_large r_small : ℝ) :=
  h = 2 * r_large ∧ r_large + r_small = 4.5

def height_of_box (h : ℝ) := 2 * 3 = h

-- The mathematically equivalent proof problem
theorem box_height_is_6 (h : ℝ) (r_large : ℝ) (r_small : ℝ) (box_size : ℝ) (n_spheres : ℕ) 
  (conditions : rectangular_box box_size r_large r_small n_spheres) 
  (tangency : sphere_tangency h r_large r_small) :
  height_of_box h :=
by {
  -- Proof is omitted
  sorry
}

end box_height_is_6_l228_228187


namespace smallest_pos_value_correct_l228_228907

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l228_228907


namespace rectangular_plot_area_l228_228048

theorem rectangular_plot_area (P : ℝ) (L W : ℝ) (h1 : P = 24) (h2 : L = 2 * W) :
    A = 32 := by
  sorry

end rectangular_plot_area_l228_228048


namespace range_of_x_for_f_lt_0_l228_228382

noncomputable def f (x : ℝ) : ℝ := x^2 - x^(1/2)

theorem range_of_x_for_f_lt_0 :
  {x : ℝ | f x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end range_of_x_for_f_lt_0_l228_228382


namespace find_x_l228_228967

namespace MathProof

variables {a b x : ℝ}
variables (h1 : a > 0) (h2 : b > 0)

theorem find_x (h3 : (a^2)^(2 * b) = a^b * x^b) : x = a^3 :=
by sorry

end MathProof

end find_x_l228_228967


namespace smallest_multiple_of_6_and_15_l228_228511

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l228_228511


namespace ratio_A_B_correct_l228_228290

-- Define the shares of A, B, and C
def A_share := 372
def B_share := 93
def C_share := 62

-- Total amount distributed
def total_share := A_share + B_share + C_share

-- The ratio of A's share to B's share
def ratio_A_to_B := A_share / B_share

theorem ratio_A_B_correct : 
  total_share = 527 ∧ 
  ¬(B_share = (1 / 4) * C_share) ∧ 
  ratio_A_to_B = 4 := 
by
  sorry

end ratio_A_B_correct_l228_228290


namespace quadratic_h_value_l228_228159

theorem quadratic_h_value (p q r h : ℝ) (hq : p*x^2 + q*x + r = 5*(x - 3)^2 + 15):
  let new_quadratic := 4* (p*x^2 + q*x + r)
  let m := 20
  let k := 60
  new_quadratic = m * (x - h) ^ 2 + k → h = 3 := by
  sorry

end quadratic_h_value_l228_228159


namespace gyeonghun_climbing_l228_228259

variable (t_up t_down d_up d_down : ℝ)
variable (h1 : t_up + t_down = 4) 
variable (h2 : d_down = d_up + 2)
variable (h3 : t_up = d_up / 3)
variable (h4 : t_down = d_down / 4)

theorem gyeonghun_climbing (h1 : t_up + t_down = 4) (h2 : d_down = d_up + 2) (h3 : t_up = d_up / 3) (h4 : t_down = d_down / 4) :
  t_up = 2 :=
by
  sorry

end gyeonghun_climbing_l228_228259


namespace fourth_root_eq_solution_l228_228226

theorem fourth_root_eq_solution (x : ℝ) (h : Real.sqrt (Real.sqrt x) = 16 / (8 - Real.sqrt (Real.sqrt x))) : x = 256 := by
  sorry

end fourth_root_eq_solution_l228_228226


namespace multiply_binomials_l228_228286

theorem multiply_binomials :
  ∀ (x : ℝ), 
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 :=
by
  sorry

end multiply_binomials_l228_228286


namespace completing_the_square_l228_228327

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l228_228327


namespace tanks_difference_l228_228035

theorem tanks_difference (total_tanks german_tanks allied_tanks sanchalian_tanks : ℕ)
  (h_total : total_tanks = 115)
  (h_german_allied : german_tanks = 2 * allied_tanks + 2)
  (h_allied_sanchalian : allied_tanks = 3 * sanchalian_tanks + 1)
  (h_total_eq : german_tanks + allied_tanks + sanchalian_tanks = total_tanks) :
  german_tanks - sanchalian_tanks = 59 :=
sorry

end tanks_difference_l228_228035


namespace total_cost_is_15_75_l228_228632

def price_sponge : ℝ := 4.20
def price_shampoo : ℝ := 7.60
def price_soap : ℝ := 3.20
def tax_rate : ℝ := 0.05
def total_cost_before_tax : ℝ := price_sponge + price_shampoo + price_soap
def tax_amount : ℝ := tax_rate * total_cost_before_tax
def total_cost_including_tax : ℝ := total_cost_before_tax + tax_amount

theorem total_cost_is_15_75 : total_cost_including_tax = 15.75 :=
by sorry

end total_cost_is_15_75_l228_228632


namespace max_daily_sales_revenue_l228_228460

noncomputable def f (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t < 15 
  then (1 / 3) * t + 8
  else if 15 ≤ t ∧ t < 30 
  then -(1 / 3) * t + 18
  else 0

noncomputable def g (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 30
  then -t + 30
  else 0

noncomputable def W (t : ℕ) : ℝ :=
  f t * g t

theorem max_daily_sales_revenue : ∃ t : ℕ, W t = 243 :=
by
  existsi 3
  sorry

end max_daily_sales_revenue_l228_228460


namespace catering_budget_total_l228_228118

theorem catering_budget_total 
  (total_guests : ℕ)
  (guests_want_chicken guests_want_steak : ℕ)
  (cost_steak cost_chicken : ℕ) 
  (H1 : total_guests = 80)
  (H2 : guests_want_steak = 3 * guests_want_chicken)
  (H3 : cost_steak = 25)
  (H4 : cost_chicken = 18)
  (H5 : guests_want_chicken + guests_want_steak = 80) :
  (guests_want_chicken * cost_chicken + guests_want_steak * cost_steak = 1860) := 
by
  sorry

end catering_budget_total_l228_228118


namespace bowls_remaining_l228_228060

-- Definitions based on conditions.
def initial_collection : ℕ := 70
def reward_per_10_bowls : ℕ := 2
def total_customers : ℕ := 20
def customers_bought_20 : ℕ := total_customers / 2
def bowls_bought_per_customer : ℕ := 20
def total_bowls_bought : ℕ := customers_bought_20 * bowls_bought_per_customer
def reward_sets : ℕ := total_bowls_bought / 10
def total_reward_given : ℕ := reward_sets * reward_per_10_bowls

-- Theorem statement to be proved.
theorem bowls_remaining : initial_collection - total_reward_given = 30 :=
by
  sorry

end bowls_remaining_l228_228060


namespace solve_for_x_l228_228578

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l228_228578


namespace jade_transactions_l228_228710

theorem jade_transactions 
    (mabel_transactions : ℕ)
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = cal_transactions + 14) : 
    jade_transactions = 80 :=
sorry

end jade_transactions_l228_228710


namespace eval_expression_l228_228438

theorem eval_expression : 5 + 4 - 3 + 2 - 1 = 7 :=
by
  -- Mathematically, this statement holds by basic arithmetic operations.
  sorry

end eval_expression_l228_228438


namespace least_multiple_of_25_gt_475_l228_228737

theorem least_multiple_of_25_gt_475 : ∃ n : ℕ, n > 475 ∧ n % 25 = 0 ∧ ∀ m : ℕ, (m > 475 ∧ m % 25 = 0) → n ≤ m := 
  sorry

end least_multiple_of_25_gt_475_l228_228737


namespace math_problem_proof_l228_228915

noncomputable def question_to_equivalent_proof_problem : Prop :=
  ∃ (p q r : ℤ), 
    (p + q + r = 0) ∧ 
    (p * q + q * r + r * p = -2023) ∧ 
    (|p| + |q| + |r| = 84)

theorem math_problem_proof : question_to_equivalent_proof_problem := 
  by 
    -- proof goes here
    sorry

end math_problem_proof_l228_228915


namespace exists_valid_numbers_l228_228287

noncomputable def sum_of_numbers_is_2012_using_two_digits : Prop :=
  ∃ (a b c d : ℕ), (a < 1000) ∧ (b < 1000) ∧ (c < 1000) ∧ (d < 1000) ∧ 
                    (∀ n ∈ [a, b, c, d], ∃ x y, (x ≠ y) ∧ ((∀ d ∈ [n / 100 % 10, n / 10 % 10, n % 10], d = x ∨ d = y))) ∧
                    (a + b + c + d = 2012)

theorem exists_valid_numbers : sum_of_numbers_is_2012_using_two_digits :=
  sorry

end exists_valid_numbers_l228_228287


namespace max_weight_of_chocolates_l228_228846

def max_total_weight (chocolates : List ℕ) (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) : ℕ :=
300

theorem max_weight_of_chocolates (chocolates : List ℕ)
  (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) :
  max_total_weight chocolates H_wt H_div = 300 :=
sorry

end max_weight_of_chocolates_l228_228846


namespace miles_from_second_friend_to_work_l228_228770
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end miles_from_second_friend_to_work_l228_228770


namespace computation_l228_228211

theorem computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  have h₁ : 27 = 3^3 := by rfl
  have h₂ : (3 : ℕ) ^ 4 = 81 := by norm_num
  have h₃ : 27^63 / 27^61 = (3^3)^63 / (3^3)^61 := by rw [h₁]
  rwa [← pow_sub, nat.sub_eq_iff_eq_add] at h₃
  have h4: 3 * 3^4 = 3^5 := by norm_num
  have h5: -486 = 3^5 - 3^6 := by norm_num
  exact h5
  sorry

end computation_l228_228211


namespace annual_depletion_rate_l228_228344

theorem annual_depletion_rate
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (depletion_rate : ℝ)
  (h_initial_value : initial_value = 40000)
  (h_final_value : final_value = 36100)
  (h_time : time = 2)
  (decay_eq : final_value = initial_value * (1 - depletion_rate)^time) :
  depletion_rate = 0.05 :=
by 
  sorry

end annual_depletion_rate_l228_228344


namespace problem_proof_l228_228466

noncomputable def original_number_of_buses_and_total_passengers : Nat × Nat :=
  let k := 24
  let total_passengers := 529
  (k, total_passengers)

theorem problem_proof (k n : Nat) (h₁ : n = 22 + 23 / (k - 1)) (h₂ : 22 * k + 1 = n * (k - 1)) (h₃ : k ≥ 2) (h₄ : n ≤ 32) :
  (k, 22 * k + 1) = original_number_of_buses_and_total_passengers :=
by
  sorry

end problem_proof_l228_228466


namespace smallest_positive_integer_l228_228610

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 66666 * n = 3 :=
by
  sorry

end smallest_positive_integer_l228_228610


namespace p_iff_q_l228_228943

variable (a b : ℝ)

def p := a > 2 ∧ b > 3

def q := a + b > 5 ∧ (a - 2) * (b - 3) > 0

theorem p_iff_q : p a b ↔ q a b := by
  sorry

end p_iff_q_l228_228943


namespace gcd_expression_infinite_composite_pairs_exists_l228_228704

-- Part (a)
theorem gcd_expression (n : ℕ) (a : ℕ) (b : ℕ) (hn : n > 0) (ha : a > 0) (hb : b > 0) :
  Nat.gcd (n^a + 1) (n^b + 1) ≤ n^(Nat.gcd a b) + 1 :=
by
  sorry

-- Part (b)
theorem infinite_composite_pairs_exists (n : ℕ) (hn : n > 0) :
  ∃ (pairs : ℕ × ℕ → Prop), (∀ a b, pairs (a, b) → a > 1 ∧ b > 1 ∧ ∃ d, d > 1 ∧ a = d ∧ b = dn) ∧
  (∀ a b, pairs (a, b) → Nat.gcd (n^a + 1) (n^b + 1) = n^(Nat.gcd a b) + 1) ∧
  (∀ x y, x > 1 → y > 1 → x ∣ y ∨ y ∣ x → ¬pairs (x, y)) :=
by
  sorry

end gcd_expression_infinite_composite_pairs_exists_l228_228704


namespace line_in_plane_parallel_to_other_plane_l228_228138

variables {Point Line Plane : Type} [EuclideanSpace Point Line Plane]

theorem line_in_plane_parallel_to_other_plane 
  (α β : Plane) (a : Line) 
  (h1 : α ∥ β) 
  (h2 : a ∈ α) : 
  a ∥ β :=
sorry

end line_in_plane_parallel_to_other_plane_l228_228138


namespace curve_is_line_l228_228079

theorem curve_is_line (r θ x y : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x + y = 1 := by
  sorry

end curve_is_line_l228_228079


namespace probability_at_least_one_woman_l228_228876

-- Given definitions
def total_employees : ℕ := 10
def total_men : ℕ := 6
def total_women : ℕ := 4
def unavailable_man : ℕ := 1
def unavailable_woman : ℕ := 1
def available_men : ℕ := total_men - unavailable_man
def available_women : ℕ := total_women - unavailable_woman
def available_employees : ℕ := available_men + available_women
def selections : ℕ := 3

-- Binomial coefficient calculation
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability calculation
def prob_at_least_one_woman : ℚ := 
  1 - (binom available_men selections : ℚ) / binom available_employees selections

-- Statement to prove
theorem probability_at_least_one_woman :
  prob_at_least_one_woman = 23 / 28 :=
sorry

end probability_at_least_one_woman_l228_228876


namespace max_red_socks_l228_228874

-- Define r (red socks), b (blue socks), t (total socks), with the given constraints
def socks_problem (r b t : ℕ) : Prop :=
  t = r + b ∧
  t ≤ 2023 ∧
  (2 * r * (r - 1) + 2 * b * (b - 1)) = 2 * 5 * t * (t - 1)

-- State the theorem that the maximum number of red socks is 990
theorem max_red_socks : ∃ r b t, socks_problem r b t ∧ r = 990 :=
sorry

end max_red_socks_l228_228874


namespace meaningful_expression_iff_l228_228946

theorem meaningful_expression_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 3))) ↔ x > 3 := by
  sorry

end meaningful_expression_iff_l228_228946


namespace meat_needed_l228_228141

theorem meat_needed (meat_per_hamburger : ℚ) (h_meat : meat_per_hamburger = (3 : ℚ) / 8) : 
  (24 * meat_per_hamburger) = 9 :=
by
  sorry

end meat_needed_l228_228141


namespace quadrilateral_area_proof_l228_228754

noncomputable def quadrilateral_area_statement : Prop :=
  ∀ (a b : ℤ), a > b ∧ b > 0 ∧ 8 * (a - b) * (a - b) = 32 → a + b = 4

theorem quadrilateral_area_proof : quadrilateral_area_statement :=
sorry

end quadrilateral_area_proof_l228_228754


namespace time_to_fill_pot_l228_228012

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end time_to_fill_pot_l228_228012


namespace determine_ratio_l228_228008

def p (x : ℝ) : ℝ := (x - 4) * (x + 3)
def q (x : ℝ) : ℝ := (x - 4) * (x + 3)

theorem determine_ratio : q 1 ≠ 0 ∧ p 1 / q 1 = 1 := by
  have hq : q 1 ≠ 0 := by
    simp [q]
    norm_num
  have hpq : p 1 / q 1 = 1 := by
    simp [p, q]
    norm_num
  exact ⟨hq, hpq⟩

end determine_ratio_l228_228008


namespace width_of_wall_l228_228341

def volume_of_brick (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_wall (length width height : ℝ) : ℝ :=
  length * width * height

theorem width_of_wall
  (l_b w_b h_b : ℝ) (n : ℝ) (L H : ℝ)
  (volume_brick := volume_of_brick l_b w_b h_b)
  (total_volume_bricks := n * volume_brick) :
  volume_of_wall L (total_volume_bricks / (L * H)) H = total_volume_bricks :=
by
  sorry

end width_of_wall_l228_228341


namespace set_intersection_complement_l228_228831

variable (U : Set ℝ := Set.univ)
variable (M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)})
variable (N : Set ℝ := {x | 0 < x ∧ x < 2})

theorem set_intersection_complement :
  N ∩ (U \ M) = {x | 0 < x ∧ x ≤ 1} :=
  sorry

end set_intersection_complement_l228_228831


namespace g_function_ratio_l228_228433

theorem g_function_ratio (g : ℝ → ℝ) (h : ∀ c d : ℝ, c^3 * g d = d^3 * g c) (hg3 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := 
by
  sorry

end g_function_ratio_l228_228433


namespace cos_75_cos_15_plus_sin_75_sin_15_l228_228648

theorem cos_75_cos_15_plus_sin_75_sin_15 :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) + 
   Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)) = (1 / 2) := by
  sorry

end cos_75_cos_15_plus_sin_75_sin_15_l228_228648


namespace ratio_of_w_to_y_l228_228999

theorem ratio_of_w_to_y
  (w x y z : ℚ)
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 :=
by sorry

end ratio_of_w_to_y_l228_228999


namespace solve_complex_eq_l228_228076

theorem solve_complex_eq (z : ℂ) (h : z^2 = -100 - 64 * I) : z = 3.06 - 10.46 * I ∨ z = -3.06 + 10.46 * I :=
by
  sorry

end solve_complex_eq_l228_228076


namespace equal_distances_l228_228405

def Point := ℝ × ℝ × ℝ

def dist (p1 p2 : Point) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2

def A : Point := (-8, 0, 0)
def B : Point := (0, 4, 0)
def C : Point := (0, 0, -6)
def D : Point := (0, 0, 0)
def P : Point := (-4, 2, -3)

theorem equal_distances : dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D :=
by
  sorry

end equal_distances_l228_228405


namespace loot_box_cost_l228_228556

variable (C : ℝ) -- Declare cost of each loot box as a real number

-- Conditions (average value of items, money spent, loss)
def avg_value : ℝ := 3.5
def money_spent : ℝ := 40
def avg_loss : ℝ := 12

-- Derived equation
def equation := avg_value * (money_spent / C) = money_spent - avg_loss

-- Statement to prove
theorem loot_box_cost : equation C → C = 5 := by
  sorry

end loot_box_cost_l228_228556


namespace parametric_curve_C_line_tangent_to_curve_C_l228_228549

open Real

-- Definitions of the curve C and line l
def curve_C (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * cos θ + 1 = 0

def line_l (t α x y : ℝ) : Prop := x = 4 + t * sin α ∧ y = t * cos α ∧ 0 ≤ α ∧ α < π

-- Parametric equation of curve C
theorem parametric_curve_C :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π →
  ∃ x y : ℝ, (x = 2 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ ∧
              curve_C (sqrt (x^2 + y^2)) θ) :=
sorry

-- Tangency condition for line l and curve C
theorem line_tangent_to_curve_C :
  ∀ α : ℝ, 0 ≤ α ∧ α < π →
  (∃ t : ℝ, ∃ x y : ℝ, (line_l t α x y ∧ (x - 2)^2 + y^2 = 3 ∧
                        ((abs (2 * cos α - 4 * cos α) / sqrt (cos α ^ 2 + sin α ^ 2)) = sqrt 3)) →
                       (α = π / 6 ∧ x = 7 / 2 ∧ y = - sqrt 3 / 2)) :=
sorry

end parametric_curve_C_line_tangent_to_curve_C_l228_228549


namespace darius_age_is_8_l228_228402

def age_of_darius (jenna_age darius_age : ℕ) : Prop :=
  jenna_age = darius_age + 5

theorem darius_age_is_8 (jenna_age darius_age : ℕ) (h1 : jenna_age = darius_age + 5) (h2: jenna_age = 13) : 
  darius_age = 8 :=
by
  sorry

end darius_age_is_8_l228_228402


namespace olivia_wallet_final_amount_l228_228315

variable (initial_money : ℕ) (money_added : ℕ) (money_spent : ℕ)

theorem olivia_wallet_final_amount
  (h1 : initial_money = 100)
  (h2 : money_added = 148)
  (h3 : money_spent = 89) :
  initial_money + money_added - money_spent = 159 :=
  by 
    sorry

end olivia_wallet_final_amount_l228_228315
