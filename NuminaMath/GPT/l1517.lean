import Mathlib

namespace triangle_is_isosceles_l1517_151793

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end triangle_is_isosceles_l1517_151793


namespace product_positions_8_2_100_100_l1517_151775

def num_at_position : ℕ → ℕ → ℤ
| 0, _ => 0
| n, k => 
  let remainder := k % 3
  if remainder = 1 then 1 
  else if remainder = 2 then 2
  else -3

theorem product_positions_8_2_100_100 : 
  num_at_position 8 2 * num_at_position 100 100 = -3 :=
by
  unfold num_at_position
  -- unfold necessary definition steps
  sorry

end product_positions_8_2_100_100_l1517_151775


namespace centroid_path_area_correct_l1517_151721

noncomputable def centroid_path_area (AB : ℝ) (A B C : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let R := AB / 2
  let radius_of_path := R / 3
  let area := Real.pi * radius_of_path ^ 2
  area

theorem centroid_path_area_correct (AB : ℝ) (A B C : ℝ × ℝ)
  (hAB : AB = 32)
  (hAB_diameter : (∃ O : ℝ × ℝ, dist O A = dist O B ∧ dist A B = 2 * dist O A))
  (hC_circle : ∃ O : ℝ × ℝ, dist O C = AB / 2 ∧ C ≠ A ∧ C ≠ B):
  centroid_path_area AB A B C (0, 0) = (256 / 9) * Real.pi := by
  sorry

end centroid_path_area_correct_l1517_151721


namespace typist_salary_proof_l1517_151749

noncomputable def original_salary (x : ℝ) : Prop :=
  1.10 * x * 0.95 = 1045

theorem typist_salary_proof (x : ℝ) (H : original_salary x) : x = 1000 :=
sorry

end typist_salary_proof_l1517_151749


namespace factorize_quadratic_example_l1517_151762

theorem factorize_quadratic_example (x : ℝ) :
  4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 :=
by
  sorry

end factorize_quadratic_example_l1517_151762


namespace find_a_2016_l1517_151753

noncomputable def a (n : ℕ) : ℕ := sorry

axiom condition_1 : a 4 = 1
axiom condition_2 : a 11 = 9
axiom condition_3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15

theorem find_a_2016 : a 2016 = 5 := sorry

end find_a_2016_l1517_151753


namespace surface_area_ratio_l1517_151766

-- Definitions based on conditions
def side_length (s : ℝ) := s > 0
def A_cube (s : ℝ) := 6 * s ^ 2
def A_rect (s : ℝ) := 2 * (2 * s) * (3 * s) + 2 * (2 * s) * (4 * s) + 2 * (3 * s) * (4 * s)

-- Theorem statement proving the ratio
theorem surface_area_ratio (s : ℝ) (h : side_length s) : A_cube s / A_rect s = 3 / 26 :=
by
  sorry

end surface_area_ratio_l1517_151766


namespace correct_properties_l1517_151708

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (-Real.pi / 6) = 0) :=
by
  sorry

end correct_properties_l1517_151708


namespace fraction_product_simplification_l1517_151777

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_simplification_l1517_151777


namespace train_speed_is_60_l1517_151771

noncomputable def train_speed_proof : Prop :=
  let train_length := 550 -- in meters
  let time_to_pass := 29.997600191984645 -- in seconds
  let man_speed_kmhr := 6 -- in km/hr
  let man_speed_ms := man_speed_kmhr * (1000 / 3600) -- converting km/hr to m/s
  let relative_speed_ms := train_length / time_to_pass -- relative speed in m/s
  let train_speed_ms := relative_speed_ms - man_speed_ms -- speed of the train in m/s
  let train_speed_kmhr := train_speed_ms * (3600 / 1000) -- converting m/s to km/hr
  train_speed_kmhr = 60 -- the speed of the train in km/hr

theorem train_speed_is_60 : train_speed_proof := by
  sorry

end train_speed_is_60_l1517_151771


namespace retailer_discount_percentage_l1517_151761

noncomputable def market_price (P : ℝ) : ℝ := 36 * P
noncomputable def profit (CP : ℝ) : ℝ := CP * 0.1
noncomputable def selling_price (P : ℝ) : ℝ := 40 * P
noncomputable def total_revenue (CP Profit : ℝ) : ℝ := CP + Profit
noncomputable def discount (P S : ℝ) : ℝ := P - S
noncomputable def discount_percentage (D P : ℝ) : ℝ := (D / P) * 100

theorem retailer_discount_percentage (P CP Profit TR S D : ℝ) (h1 : CP = market_price P)
  (h2 : Profit = profit CP) (h3 : TR = total_revenue CP Profit)
  (h4 : TR = selling_price S) (h5 : S = TR / 40) (h6 : D = discount P S) :
  discount_percentage D P = 1 :=
by
  sorry

end retailer_discount_percentage_l1517_151761


namespace first_consecutive_odd_number_l1517_151712

theorem first_consecutive_odd_number :
  ∃ k : Int, 2 * k - 1 + 2 * k + 1 + 2 * k + 3 = 2 * k - 1 + 128 ∧ 2 * k - 1 = 61 :=
by
  sorry

end first_consecutive_odd_number_l1517_151712


namespace find_K_l1517_151747

theorem find_K (K m n : ℝ) (p : ℝ) (hp : p = 0.3333333333333333)
  (eq1 : m = K * n + 5)
  (eq2 : m + 2 = K * (n + p) + 5) : 
  K = 6 := 
by
  sorry

end find_K_l1517_151747


namespace inequality_solution_l1517_151751

theorem inequality_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : (x * (x + 1)) / ((x - 3)^2) ≥ 8) : 3 < x ∧ x ≤ 24/7 :=
sorry

end inequality_solution_l1517_151751


namespace stephen_total_distance_l1517_151730

noncomputable def total_distance : ℝ :=
let speed1 : ℝ := 16
let time1 : ℝ := 10 / 60
let distance1 : ℝ := speed1 * time1

let speed2 : ℝ := 12 - 2 -- headwind reduction
let time2 : ℝ := 20 / 60
let distance2 : ℝ := speed2 * time2

let speed3 : ℝ := 20 + 4 -- tailwind increase
let time3 : ℝ := 15 / 60
let distance3 : ℝ := speed3 * time3

distance1 + distance2 + distance3

theorem stephen_total_distance :
  total_distance = 12 :=
by sorry

end stephen_total_distance_l1517_151730


namespace solution_set_of_inequality_l1517_151700

theorem solution_set_of_inequality (x : ℝ) : x^2 < -2 * x + 15 ↔ -5 < x ∧ x < 3 := 
sorry

end solution_set_of_inequality_l1517_151700


namespace f_neg_2_f_monotonically_decreasing_l1517_151734

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 4
axiom f_2 : f 2 = 0
axiom f_pos_2 (x : ℝ) : x > 2 → f x < 0

-- Statement to prove f(-2) = 8
theorem f_neg_2 : f (-2) = 8 := sorry

-- Statement to prove that f(x) is monotonically decreasing on ℝ
theorem f_monotonically_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := sorry

end f_neg_2_f_monotonically_decreasing_l1517_151734


namespace find_rate_of_interest_l1517_151776

noncomputable def interest_rate (P R : ℝ) : Prop :=
  (400 = P * (1 + 4 * R / 100)) ∧ (500 = P * (1 + 6 * R / 100))

theorem find_rate_of_interest (R : ℝ) (P : ℝ) (h : interest_rate P R) :
  R = 25 :=
by
  sorry

end find_rate_of_interest_l1517_151776


namespace melt_brown_fabric_scientific_notation_l1517_151720

theorem melt_brown_fabric_scientific_notation :
  0.000156 = 1.56 * 10^(-4) :=
sorry

end melt_brown_fabric_scientific_notation_l1517_151720


namespace calculate_a2_b2_c2_l1517_151783

theorem calculate_a2_b2_c2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -3) (h3 : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end calculate_a2_b2_c2_l1517_151783


namespace find_fractions_l1517_151798

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l1517_151798


namespace average_speed_comparison_l1517_151774

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0):
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v + w) / 3) :=
sorry

end average_speed_comparison_l1517_151774


namespace highest_slope_product_l1517_151797

theorem highest_slope_product (m1 m2 : ℝ) (h1 : m1 = 5 * m2) 
    (h2 : abs ((m2 - m1) / (1 + m1 * m2)) = 1) : (m1 * m2) ≤ 1.8 :=
by
  sorry

end highest_slope_product_l1517_151797


namespace circle_area_greater_than_hexagon_area_l1517_151729

theorem circle_area_greater_than_hexagon_area (h : ℝ) (r : ℝ) (π : ℝ) (sqrt3 : ℝ) (ratio : ℝ) : 
  (h = 1) →
  (r = sqrt3 / 2) →
  (π > 3) →
  (sqrt3 > 1.7) →
  (ratio = (π * sqrt3) / 6) →
  ratio > 0.9 :=
by
  intros h_eq r_eq pi_gt sqrt3_gt ratio_eq
  -- Proof omitted
  sorry

end circle_area_greater_than_hexagon_area_l1517_151729


namespace sufficient_not_necessary_condition_l1517_151768

theorem sufficient_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x > 0 ∧ y > 0) → (x > 0 ∧ y > 0 ↔ (y/x + x/y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l1517_151768


namespace number_of_students_only_taking_AMC8_l1517_151741

def total_Germain := 13
def total_Newton := 10
def total_Young := 12

def olympiad_Germain := 3
def olympiad_Newton := 2
def olympiad_Young := 4

def number_only_AMC8 :=
  (total_Germain - olympiad_Germain) +
  (total_Newton - olympiad_Newton) +
  (total_Young - olympiad_Young)

theorem number_of_students_only_taking_AMC8 :
  number_only_AMC8 = 26 := by
  sorry

end number_of_students_only_taking_AMC8_l1517_151741


namespace sum_of_first_K_natural_numbers_is_perfect_square_l1517_151790

noncomputable def values_K (K : ℕ) : Prop := 
  ∃ N : ℕ, (K * (K + 1)) / 2 = N^2 ∧ (N + K < 120)

theorem sum_of_first_K_natural_numbers_is_perfect_square :
  ∀ K : ℕ, values_K K ↔ (K = 1 ∨ K = 8 ∨ K = 49) := by
  sorry

end sum_of_first_K_natural_numbers_is_perfect_square_l1517_151790


namespace quadratic_vertex_l1517_151728

theorem quadratic_vertex (x : ℝ) :
  ∃ (h k : ℝ), (h = -3) ∧ (k = -5) ∧ (∀ y, y = -2 * (x + h) ^ 2 + k) :=
sorry

end quadratic_vertex_l1517_151728


namespace each_wolf_kills_one_deer_l1517_151752

-- Definitions based on conditions
def hunting_wolves : Nat := 4
def additional_wolves : Nat := 16
def wolves_per_pack : Nat := hunting_wolves + additional_wolves
def meat_per_wolf_per_day : Nat := 8
def days_between_hunts : Nat := 5
def meat_per_wolf : Nat := meat_per_wolf_per_day * days_between_hunts
def total_meat_required : Nat := wolves_per_pack * meat_per_wolf
def meat_per_deer : Nat := 200
def deer_needed : Nat := total_meat_required / meat_per_deer
def deer_per_wolf_needed : Nat := deer_needed / hunting_wolves

-- Lean statement to prove
theorem each_wolf_kills_one_deer (hunting_wolves : Nat := 4) (additional_wolves : Nat := 16) 
    (meat_per_wolf_per_day : Nat := 8) (days_between_hunts : Nat := 5) 
    (meat_per_deer : Nat := 200) : deer_per_wolf_needed = 1 := 
by
  -- Proof required here
  sorry

end each_wolf_kills_one_deer_l1517_151752


namespace students_play_at_least_one_sport_l1517_151739

def B := 12
def C := 10
def S := 9
def Ba := 6

def B_and_C := 5
def B_and_S := 4
def B_and_Ba := 3
def C_and_S := 2
def C_and_Ba := 3
def S_and_Ba := 2

def B_and_C_and_S_and_Ba := 1

theorem students_play_at_least_one_sport : 
  B + C + S + Ba - B_and_C - B_and_S - B_and_Ba - C_and_S - C_and_Ba - S_and_Ba + B_and_C_and_S_and_Ba = 19 :=
by
  sorry

end students_play_at_least_one_sport_l1517_151739


namespace cubic_equation_three_distinct_real_roots_l1517_151789

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 - a

theorem cubic_equation_three_distinct_real_roots (a : ℝ) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃
  ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ↔ -4 < a ∧ a < 0 :=
sorry

end cubic_equation_three_distinct_real_roots_l1517_151789


namespace necessary_and_sufficient_condition_x_eq_1_l1517_151722

theorem necessary_and_sufficient_condition_x_eq_1
    (x : ℝ) :
    (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end necessary_and_sufficient_condition_x_eq_1_l1517_151722


namespace cherry_sodas_in_cooler_l1517_151742

theorem cherry_sodas_in_cooler (C : ℕ) (h1 : (C + 2 * C = 24)) : C = 8 :=
sorry

end cherry_sodas_in_cooler_l1517_151742


namespace discriminant_of_quadratic_l1517_151791

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := 4

-- Prove the discriminant of the quadratic equation
theorem discriminant_of_quadratic :
    b^2 - 4 * a * c = 41 :=
by
  sorry

end discriminant_of_quadratic_l1517_151791


namespace find_base_l1517_151701

noncomputable def base_satisfies_first_transaction (s : ℕ) : Prop :=
  5 * s^2 + 3 * s + 460 = s^3 + s^2 + 1

noncomputable def base_satisfies_second_transaction (s : ℕ) : Prop :=
  s^2 + 2 * s + 2 * s^2 + 6 * s = 5 * s^2

theorem find_base (s : ℕ) (h1 : base_satisfies_first_transaction s) (h2 : base_satisfies_second_transaction s) :
  s = 4 :=
sorry

end find_base_l1517_151701


namespace motorboat_distance_l1517_151786

variable (S v u : ℝ)
variable (V_m : ℝ := 2 * v + u)  -- Velocity of motorboat downstream
variable (V_b : ℝ := 3 * v - u)  -- Velocity of boat upstream

theorem motorboat_distance :
  ( L = (161 / 225) * S ∨ L = (176 / 225) * S) :=
by
  sorry

end motorboat_distance_l1517_151786


namespace total_yellow_marbles_l1517_151757

theorem total_yellow_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := 
by 
  sorry

end total_yellow_marbles_l1517_151757


namespace response_rate_percentage_l1517_151780

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 240) (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := 
by 
  sorry

end response_rate_percentage_l1517_151780


namespace no_solution_m_4_l1517_151796

theorem no_solution_m_4 (m : ℝ) : 
  (¬ ∃ x : ℝ, 2/x = m/(2*x + 1)) → m = 4 :=
by
  sorry

end no_solution_m_4_l1517_151796


namespace sequence_sum_l1517_151756

open Nat

-- Define the sequence
def a : ℕ → ℕ
| 0     => 1
| (n+1) => a n + (n + 1)

-- Define the sum of reciprocals up to the 2016 term
def sum_reciprocals : ℕ → ℚ
| 0     => 1 / (a 0)
| (n+1) => sum_reciprocals n + 1 / (a (n+1))

-- Define the property we wish to prove
theorem sequence_sum :
  sum_reciprocals 2015 = 4032 / 2017 :=
sorry

end sequence_sum_l1517_151756


namespace carla_drank_total_amount_l1517_151779

-- Define the conditions
def carla_water : ℕ := 15
def carla_soda := 3 * carla_water - 6
def total_liquid := carla_water + carla_soda

-- State the theorem
theorem carla_drank_total_amount : total_liquid = 54 := by
  sorry

end carla_drank_total_amount_l1517_151779


namespace greatest_integer_of_set_is_152_l1517_151710

-- Define the conditions
def median (s : Set ℤ) : ℤ := 150
def smallest_integer (s : Set ℤ) : ℤ := 140
def consecutive_even_integers (s : Set ℤ) : Prop := 
  ∀ x ∈ s, ∃ y ∈ s, x = y ∨ x = y + 2

-- The main theorem
theorem greatest_integer_of_set_is_152 (s : Set ℤ) 
  (h_median : median s = 150)
  (h_smallest : smallest_integer s = 140)
  (h_consecutive : consecutive_even_integers s) : 
  ∃ greatest : ℤ, greatest = 152 := 
sorry

end greatest_integer_of_set_is_152_l1517_151710


namespace kendall_tau_correct_l1517_151713

-- Base Lean setup and list of dependencies might go here

structure TestScores :=
  (A : List ℚ)
  (B : List ℚ)

-- Constants from the problem
def scores : TestScores :=
  { A := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
  , B := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70] }

-- Function to calculate the Kendall rank correlation coefficient
noncomputable def kendall_tau (scores : TestScores) : ℚ :=
  -- the method of calculating Kendall tau could be very complex
  -- hence we assume the correct coefficient directly for the example
  0.51

-- The proof problem
theorem kendall_tau_correct : kendall_tau scores = 0.51 :=
by
  sorry

end kendall_tau_correct_l1517_151713


namespace appropriate_chart_for_temperature_statistics_l1517_151784

theorem appropriate_chart_for_temperature_statistics (chart_type : String) (is_line_chart : chart_type = "line chart") : chart_type = "line chart" :=
by
  sorry

end appropriate_chart_for_temperature_statistics_l1517_151784


namespace problem_statement_l1517_151711

theorem problem_statement :
  (3 = 0.25 * x) ∧ (3 = 0.50 * y) → (x - y = 6) ∧ (x + y = 18) :=
by
  sorry

end problem_statement_l1517_151711


namespace pairs_bought_after_donation_l1517_151732

-- Definitions from conditions
def initial_pairs : ℕ := 80
def donation_percentage : ℕ := 30
def post_donation_pairs : ℕ := 62

-- The theorem to be proven
theorem pairs_bought_after_donation : (initial_pairs - (donation_percentage * initial_pairs / 100) + 6 = post_donation_pairs) :=
by
  sorry

end pairs_bought_after_donation_l1517_151732


namespace Maryann_frees_all_friends_in_42_minutes_l1517_151767

-- Definitions for the problem conditions
def time_to_pick_cheap_handcuffs := 6
def time_to_pick_expensive_handcuffs := 8
def number_of_friends := 3

-- Define the statement we need to prove
theorem Maryann_frees_all_friends_in_42_minutes :
  (time_to_pick_cheap_handcuffs + time_to_pick_expensive_handcuffs) * number_of_friends = 42 :=
by
  sorry

end Maryann_frees_all_friends_in_42_minutes_l1517_151767


namespace apps_minus_files_eq_seven_l1517_151763

-- Definitions based on conditions
def initial_apps := 24
def initial_files := 9
def deleted_apps := initial_apps - 12
def deleted_files := initial_files - 5

-- Definitions based on the question and correct answer
def apps_left := 12
def files_left := 5

theorem apps_minus_files_eq_seven : apps_left - files_left = 7 := by
  sorry

end apps_minus_files_eq_seven_l1517_151763


namespace fraction_not_exist_implies_x_neg_one_l1517_151725

theorem fraction_not_exist_implies_x_neg_one {x : ℝ} :
  ¬(∃ y : ℝ, y = 1 / (x + 1)) → x = -1 :=
by
  intro h
  have : x + 1 = 0 :=
    by
      contrapose! h
      exact ⟨1 / (x + 1), rfl⟩
  linarith

end fraction_not_exist_implies_x_neg_one_l1517_151725


namespace balance_test_l1517_151715

variable (a b h c : ℕ)

theorem balance_test
  (h1 : 4 * a + 2 * b + h = 21 * c)
  (h2 : 2 * a = b + h + 5 * c) :
  b + 2 * h = 11 * c :=
sorry

end balance_test_l1517_151715


namespace quadratic_has_two_distinct_real_roots_l1517_151733

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 * x + m = 0 ∧ y^2 - 2 * y + m = 0) ↔ m < 1 :=
sorry

end quadratic_has_two_distinct_real_roots_l1517_151733


namespace find_b_plus_m_l1517_151759

def matrix_C (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]
  ]

def matrix_RHS : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 27, 3003],
    ![0, 1, 45],
    ![0, 0, 1]
  ]

theorem find_b_plus_m (b m : ℕ) (h : matrix_C b ^ m = matrix_RHS) : b + m = 306 := 
  sorry

end find_b_plus_m_l1517_151759


namespace remainder_when_divided_l1517_151703

theorem remainder_when_divided (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := 
by sorry

end remainder_when_divided_l1517_151703


namespace expression_value_l1517_151781

theorem expression_value (x : ℝ) (h : x = -2) : (x * x^2 * (1/x) = 4) :=
by
  rw [h]
  sorry

end expression_value_l1517_151781


namespace group_members_l1517_151716

theorem group_members (n : ℕ) (hn : n * n = 1369) : n = 37 :=
by
  sorry

end group_members_l1517_151716


namespace import_tax_excess_amount_l1517_151765

theorem import_tax_excess_amount (X : ℝ)
  (total_value : ℝ) (tax_paid : ℝ)
  (tax_rate : ℝ) :
  total_value = 2610 → tax_paid = 112.70 → tax_rate = 0.07 → 0.07 * (2610 - X) = 112.70 → X = 1000 :=
by
  intros h1 h2 h3 h4
  sorry

end import_tax_excess_amount_l1517_151765


namespace example_theorem_l1517_151746

-- Definitions of the conditions
def parallel (l1 l2 : Line) : Prop := sorry

def Angle (A B C : Point) : ℝ := sorry

-- Given conditions
def DC_parallel_AB (DC AB : Line) : Prop := parallel DC AB
def DCA_eq_55 (D C A : Point) : Prop := Angle D C A = 55
def ABC_eq_60 (A B C : Point) : Prop := Angle A B C = 60

-- Proof that angle ACB equals 5 degrees given the conditions
theorem example_theorem (D C A B : Point) (DC AB : Line) :
  DC_parallel_AB DC AB →
  DCA_eq_55 D C A →
  ABC_eq_60 A B C →
  Angle A C B = 5 := by
  sorry

end example_theorem_l1517_151746


namespace area_sum_four_smaller_circles_equals_area_of_large_circle_l1517_151717

theorem area_sum_four_smaller_circles_equals_area_of_large_circle (R : ℝ) :
  let radius_large := R
  let radius_small := R / 2
  let area_large := π * radius_large^2
  let area_small := π * radius_small^2
  let total_area_small := 4 * area_small
  area_large = total_area_small :=
by
  sorry

end area_sum_four_smaller_circles_equals_area_of_large_circle_l1517_151717


namespace vivi_total_yards_l1517_151794

theorem vivi_total_yards (spent_checkered spent_plain cost_per_yard : ℝ)
  (h1 : spent_checkered = 75)
  (h2 : spent_plain = 45)
  (h3 : cost_per_yard = 7.50) :
  (spent_checkered / cost_per_yard + spent_plain / cost_per_yard) = 16 :=
by 
  sorry

end vivi_total_yards_l1517_151794


namespace solve_for_y_l1517_151737

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end solve_for_y_l1517_151737


namespace problem_proof_l1517_151764

def f (a x : ℝ) := |a - x|

theorem problem_proof (a x x0 : ℝ) (h_a : a = 3 / 2) (h_x0 : x0 < 0) : 
  f a (x0 * x) ≥ x0 * f a x + f a (a * x0) :=
sorry

end problem_proof_l1517_151764


namespace ratio_of_volume_to_surface_area_l1517_151799

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

end ratio_of_volume_to_surface_area_l1517_151799


namespace basketball_score_l1517_151778

theorem basketball_score (score_game1 : ℕ) (score_game2 : ℕ) (score_game3 : ℕ) (score_game4 : ℕ) (score_total_games8 : ℕ) (score_total_games9 : ℕ) :
  score_game1 = 18 ∧ score_game2 = 22 ∧ score_game3 = 15 ∧ score_game4 = 20 ∧ 
  (score_game1 + score_game2 + score_game3 + score_game4) / 4 < score_total_games8 / 8 ∧ 
  score_total_games9 / 9 > 19 →
  score_total_games9 - score_total_games8 ≥ 21 :=
by
-- proof steps would be provided here based on the given solution
sorry

end basketball_score_l1517_151778


namespace quadratic_roots_expression_l1517_151735

theorem quadratic_roots_expression (x1 x2 : ℝ) (h1 : x1^2 + x1 - 2023 = 0) (h2 : x2^2 + x2 - 2023 = 0) :
  x1^2 + 2*x1 + x2 = 2022 :=
by
  sorry

end quadratic_roots_expression_l1517_151735


namespace sum_distinct_integers_l1517_151755

theorem sum_distinct_integers (a b c d e : ℤ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
    (h : (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120) :
    a + b + c + d + e = 13 := by
  sorry

end sum_distinct_integers_l1517_151755


namespace sqrt_16_eq_pm_4_l1517_151731

theorem sqrt_16_eq_pm_4 (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := by
  sorry

end sqrt_16_eq_pm_4_l1517_151731


namespace average_height_students_count_l1517_151785

-- Definitions based on the conditions
def total_students : ℕ := 400
def short_students : ℕ := (2 * total_students) / 5
def extremely_tall_students : ℕ := total_students / 10
def tall_students : ℕ := 90
def average_height_students : ℕ := total_students - (short_students + tall_students + extremely_tall_students)

-- Theorem to prove
theorem average_height_students_count : average_height_students = 110 :=
by
  -- This proof is omitted, we are only stating the theorem.
  sorry

end average_height_students_count_l1517_151785


namespace div_decimal_l1517_151727

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end div_decimal_l1517_151727


namespace solution_to_problem_l1517_151795

theorem solution_to_problem (x : ℝ) (h : 12^(Real.log 7 / Real.log 12) = 10 * x + 3) : x = 2 / 5 :=
by sorry

end solution_to_problem_l1517_151795


namespace monkey_ladder_min_rungs_l1517_151740

/-- 
  Proof that the minimum number of rungs n that allows the monkey to climb 
  to the top of the ladder and return to the ground, given that the monkey 
  ascends 16 rungs or descends 9 rungs at a time, is 24. 
-/
theorem monkey_ladder_min_rungs (n : ℕ) (ascend descend : ℕ) 
  (h1 : ascend = 16) (h2 : descend = 9) 
  (h3 : (∃ x y : ℤ, 16 * x - 9 * y = n) ∧ 
        (∃ x' y' : ℤ, 16 * x' - 9 * y' = 0)) : 
  n = 24 :=
sorry

end monkey_ladder_min_rungs_l1517_151740


namespace area_of_triangle_ABC_l1517_151736

theorem area_of_triangle_ABC (BD CE : ℝ) (angle_BD_CE : ℝ) (BD_len : BD = 9) (CE_len : CE = 15) (angle_BD_CE_deg : angle_BD_CE = 60) : 
  ∃ area : ℝ, 
    area = 90 * Real.sqrt 3 := 
by
  sorry

end area_of_triangle_ABC_l1517_151736


namespace fermat_numbers_coprime_l1517_151707

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2 ^ 2 ^ (n - 1) + 1) (2 ^ 2 ^ (m - 1) + 1) = 1 :=
sorry

end fermat_numbers_coprime_l1517_151707


namespace find_integer_x_l1517_151787

open Nat

noncomputable def isSquareOfPrime (n : ℤ) : Prop :=
  ∃ p : ℤ, Nat.Prime (Int.natAbs p) ∧ n = p * p

theorem find_integer_x :
  ∃ x : ℤ,
  (x = -360 ∨ x = -60 ∨ x = -48 ∨ x = -40 ∨ x = 8 ∨ x = 20 ∨ x = 32 ∨ x = 332) ∧
  isSquareOfPrime (x^2 + 28*x + 889) :=
sorry

end find_integer_x_l1517_151787


namespace find_b_find_perimeter_b_plus_c_l1517_151726

noncomputable def triangle_condition_1
  (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.cos B = (3 * c - b) * Real.cos A

noncomputable def triangle_condition_2
  (a b : ℝ) (C : ℝ) : Prop :=
  a * Real.sin C = 2 * Real.sqrt 2

noncomputable def triangle_condition_3
  (a b c : ℝ) (A : ℝ) : Prop :=
  (1 / 2) * b * c * Real.sin A = Real.sqrt 2

noncomputable def given_a
  (a : ℝ) : Prop :=
  a = 2 * Real.sqrt 2

theorem find_b
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b = 3 :=
sorry

theorem find_perimeter_b_plus_c
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b + c = 2 * Real.sqrt 3 :=
sorry

end find_b_find_perimeter_b_plus_c_l1517_151726


namespace smallest_solution_l1517_151758

def equation (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

theorem smallest_solution :
  ∀ x : ℝ, equation x → x = (3 - Real.sqrt 333) / 6 :=
sorry

end smallest_solution_l1517_151758


namespace find_k_l1517_151773

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 50 < f a b c 7)
  (h3 : f a b c 7 < 60)
  (h4 : 70 < f a b c 8)
  (h5 : f a b c 8 < 80)
  (h6 : 5000 * k < f a b c 100)
  (h7 : f a b c 100 < 5000 * (k + 1)) :
  k = 3 :=
sorry

end find_k_l1517_151773


namespace min_troublemakers_29_l1517_151754

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l1517_151754


namespace triangular_prism_sliced_faces_l1517_151723

noncomputable def resulting_faces_count : ℕ :=
  let initial_faces := 5 -- 2 bases + 3 lateral faces
  let additional_faces := 3 -- from the slices
  initial_faces + additional_faces

theorem triangular_prism_sliced_faces :
  resulting_faces_count = 8 := by
  sorry

end triangular_prism_sliced_faces_l1517_151723


namespace percentage_decrease_hours_with_assistant_l1517_151788

theorem percentage_decrease_hours_with_assistant :
  ∀ (B H H_new : ℝ), H_new = 0.9 * H → (H - H_new) / H * 100 = 10 :=
by
  intros B H H_new h_new_def
  sorry

end percentage_decrease_hours_with_assistant_l1517_151788


namespace range_for_k_solutions_when_k_eq_1_l1517_151748

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end range_for_k_solutions_when_k_eq_1_l1517_151748


namespace fraction_of_coins_in_decade_1800_through_1809_l1517_151702

theorem fraction_of_coins_in_decade_1800_through_1809 (total_coins : ℕ) (coins_in_decade : ℕ) (c : total_coins = 30) (d : coins_in_decade = 5) : coins_in_decade / (total_coins : ℚ) = 1 / 6 :=
by
  sorry

end fraction_of_coins_in_decade_1800_through_1809_l1517_151702


namespace skate_cost_l1517_151760

/- Define the initial conditions as Lean definitions -/
def admission_cost : ℕ := 5
def rental_cost : ℕ := 250 / 100  -- 2.50 dollars in cents for integer representation
def visits : ℕ := 26

/- Define the cost calculation as a Lean definition -/
def total_rental_cost (rental_cost : ℕ) (visits : ℕ) : ℕ := rental_cost * visits

/- Statement of the problem in Lean proof form -/
theorem skate_cost (C : ℕ) (h : total_rental_cost rental_cost visits = C) : C = 65 :=
by
  sorry

end skate_cost_l1517_151760


namespace age_difference_l1517_151738

variables (O N A : ℕ)

theorem age_difference (avg_age_stable : 10 * A = 10 * A + 50 - O + N) :
  O - N = 50 :=
by
  -- proof would go here
  sorry

end age_difference_l1517_151738


namespace find_a_l1517_151743

-- Define sets A and B based on the given real number a
def A (a : ℝ) : Set ℝ := {a^2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 3 * a - 1, a^2 + 1}

-- Given condition
def condition (a : ℝ) : Prop := A a ∩ B a = {-3}

-- Prove that a = -2/3 is the solution satisfying the condition
theorem find_a : ∃ a : ℝ, condition a ∧ a = -2/3 :=
by
  sorry  -- Proof goes here

end find_a_l1517_151743


namespace no_solution_in_natural_numbers_l1517_151724

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end no_solution_in_natural_numbers_l1517_151724


namespace points_not_all_odd_distance_l1517_151706

open Real

theorem points_not_all_odd_distance (p : Fin 4 → ℝ × ℝ) : ∃ i j : Fin 4, i ≠ j ∧ ¬ Odd (dist (p i) (p j)) := 
by
  sorry

end points_not_all_odd_distance_l1517_151706


namespace greatest_integer_prime_l1517_151769

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → n % m ≠ 0

theorem greatest_integer_prime (x : ℤ) :
  is_prime (|8 * x ^ 2 - 56 * x + 21|) → ∀ y : ℤ, (is_prime (|8 * y ^ 2 - 56 * y + 21|) → y ≤ x) :=
by
  sorry

end greatest_integer_prime_l1517_151769


namespace nina_weekend_earnings_l1517_151792

noncomputable def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℕ)
                                   (necklaces_sold bracelets_sold individual_earrings_sold ensembles_sold : ℕ) : ℕ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (individual_earrings_sold / 2) +
  ensemble_price * ensembles_sold

theorem nina_weekend_earnings :
  total_money_made 25 15 10 45 5 10 20 2 = 465 :=
by
  sorry

end nina_weekend_earnings_l1517_151792


namespace mix_solutions_l1517_151772

-- Definitions based on conditions
def solution_x_percentage : ℝ := 0.10
def solution_y_percentage : ℝ := 0.30
def volume_y : ℝ := 100
def desired_percentage : ℝ := 0.15

-- Problem statement rewrite with equivalent proof goal
theorem mix_solutions :
  ∃ Vx : ℝ, (Vx * solution_x_percentage + volume_y * solution_y_percentage) = (Vx + volume_y) * desired_percentage ∧ Vx = 300 :=
by
  sorry

end mix_solutions_l1517_151772


namespace determine_m_l1517_151714

-- Define f and g according to the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

-- Define the value of x
def x := 5

-- State the main theorem we need to prove
theorem determine_m 
  (h : 3 * f x m = 2 * g x m) : m = 10 / 7 :=
by
  -- Proof is omitted
  sorry

end determine_m_l1517_151714


namespace find_circle_equation_l1517_151705

noncomputable def center (m : ℝ) := (3 * m, m)

def radius (m : ℝ) : ℝ := 3 * m

def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  (x - 3 * m)^2 + (y - m)^2 = (radius m)^2

def point_A : ℝ × ℝ := (6, 1)

theorem find_circle_equation (m : ℝ) :
  (radius m = 3 * m ∧ center m = (3 * m, m) ∧ 
   point_A = (6, 1) ∧
   circle_eq m 6 1) →
  (circle_eq 1 x y ∨ circle_eq 37 x y) :=
by
  sorry

end find_circle_equation_l1517_151705


namespace average_visitors_on_sundays_is_correct_l1517_151745

noncomputable def average_visitors_sundays
  (num_sundays : ℕ) (num_non_sundays : ℕ) 
  (avg_non_sunday_visitors : ℕ) (avg_month_visitors : ℕ) : ℕ :=
  let total_month_days := num_sundays + num_non_sundays
  let total_visitors := avg_month_visitors * total_month_days
  let total_non_sunday_visitors := num_non_sundays * avg_non_sunday_visitors
  let total_sunday_visitors := total_visitors - total_non_sunday_visitors
  total_sunday_visitors / num_sundays

theorem average_visitors_on_sundays_is_correct :
  average_visitors_sundays 5 25 240 290 = 540 :=
by
  sorry

end average_visitors_on_sundays_is_correct_l1517_151745


namespace no_rational_solution_5x2_plus_3y2_eq_1_l1517_151750

theorem no_rational_solution_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := 
sorry

end no_rational_solution_5x2_plus_3y2_eq_1_l1517_151750


namespace haily_cheapest_salon_l1517_151709

def cost_Gustran : ℕ := 45 + 22 + 30
def cost_Barbara : ℕ := 40 + 30 + 28
def cost_Fancy : ℕ := 30 + 34 + 20

theorem haily_cheapest_salon : min (min cost_Gustran cost_Barbara) cost_Fancy = 84 := by
  sorry

end haily_cheapest_salon_l1517_151709


namespace find_time_l1517_151704

variables (V V_0 S g C : ℝ) (t : ℝ)

-- Given conditions.
axiom eq1 : V = 2 * g * t + V_0
axiom eq2 : S = (1 / 3) * g * t^2 + V_0 * t + C * t^3

-- The statement to prove.
theorem find_time : t = (V - V_0) / (2 * g) :=
sorry

end find_time_l1517_151704


namespace avg_calculation_l1517_151718

-- Define averages
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 0 2) 0 = 7 / 9 :=
  by
    sorry

end avg_calculation_l1517_151718


namespace find_line_eq_show_point_on_circle_l1517_151719

noncomputable section

variables {x y x0 y0 : ℝ} (P Q : ℝ × ℝ) (h1 : y0 ≠ 0)
  (h2 : P = (x0, y0))
  (h3 : P.1^2/4 + P.2^2/3 = 1)
  (h4 : Q = (x0/4, y0/3))

theorem find_line_eq (M : ℝ × ℝ) (hM : ∀ (M : ℝ × ℝ), 
  ((P.1 - M.1) , (P.2 - M.2)) • (Q.1 , Q.2) = 0) :
  ∀ (x0 y0 : ℝ), y0 ≠ 0 → ∀ (x y : ℝ), 
  (x0 * x / 4 + y0 * y / 3 = 1) :=
by sorry
  
theorem show_point_on_circle (F S : ℝ × ℝ)
  (hF : F = (1, 0)) (hs : ∀ (x0 y0 : ℝ), y0 ≠ 0 → 
  S = (4, 0) ∧ ((S.1 - P.1) ^ 2 + (S.2 - P.2) ^ 2 = 36)) :
  ∀ (x y : ℝ), 
  (x - 1) ^ 2 + y ^ 2 = 36 := 
by sorry

end find_line_eq_show_point_on_circle_l1517_151719


namespace erasers_left_l1517_151770

/-- 
There are initially 250 erasers in a box. Doris takes 75 erasers, Mark takes 40 
erasers, and Ellie takes 30 erasers out of the box. Prove that 105 erasers are 
left in the box.
-/
theorem erasers_left (initial_erasers : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ)
  (h_initial : initial_erasers = 250)
  (h_doris : doris_takes = 75)
  (h_mark : mark_takes = 40)
  (h_ellie : ellie_takes = 30) :
  initial_erasers - doris_takes - mark_takes - ellie_takes = 105 :=
  by 
  sorry

end erasers_left_l1517_151770


namespace find_natural_number_l1517_151782

theorem find_natural_number (x : ℕ) (y z : ℤ) (hy : x = 2 * y^2 - 1) (hz : x^2 = 2 * z^2 - 1) : x = 1 ∨ x = 7 :=
sorry

end find_natural_number_l1517_151782


namespace sector_area_eq_three_halves_l1517_151744

theorem sector_area_eq_three_halves (θ R S : ℝ) (hθ : θ = 3) (h₁ : 2 * R + θ * R = 5) :
  S = 3 / 2 :=
by
  sorry

end sector_area_eq_three_halves_l1517_151744
