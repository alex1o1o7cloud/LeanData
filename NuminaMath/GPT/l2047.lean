import Mathlib

namespace students_playing_both_correct_l2047_204742

def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def neither_players : ℕ := 7
def students_playing_both : ℕ := 17

theorem students_playing_both_correct :
  total_students - neither_players = (football_players + long_tennis_players) - students_playing_both :=
by 
  sorry

end students_playing_both_correct_l2047_204742


namespace jenna_weight_lift_l2047_204766

theorem jenna_weight_lift:
  ∀ (n : Nat), (2 * 10 * 25 = 500) ∧ (15 * n >= 500) ∧ (n = Nat.ceil (500 / 15 : ℝ))
  → n = 34 := 
by
  intros n h
  have h₀ : 2 * 10 * 25 = 500 := h.1
  have h₁ : 15 * n >= 500 := h.2.1
  have h₂ : n = Nat.ceil (500 / 15 : ℝ) := h.2.2
  sorry

end jenna_weight_lift_l2047_204766


namespace group_formations_at_fair_l2047_204761

theorem group_formations_at_fair : 
  (Nat.choose 7 3) * (Nat.choose 4 4) = 35 := by
  sorry

end group_formations_at_fair_l2047_204761


namespace area_of_triangle_tangent_at_pi_div_two_l2047_204726

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem area_of_triangle_tangent_at_pi_div_two :
  let x := Real.pi / 2
  let slope := 1 + Real.cos x
  let point := (x, f x)
  let intercept_y := f x - slope * x
  let x_intercept := -intercept_y / slope
  let y_intercept := intercept_y
  (1 / 2) * x_intercept * y_intercept = 1 / 2 := 
by
  sorry

end area_of_triangle_tangent_at_pi_div_two_l2047_204726


namespace fraction_beans_remain_l2047_204765

theorem fraction_beans_remain (J B B_remain : ℝ) 
  (h1 : J = 0.10 * (J + B)) 
  (h2 : J + B_remain = 0.60 * (J + B)) : 
  B_remain / B = 5 / 9 := 
by 
  sorry

end fraction_beans_remain_l2047_204765


namespace one_fifth_of_five_times_nine_l2047_204780

theorem one_fifth_of_five_times_nine (a b : ℕ) (h1 : a = 5) (h2 : b = 9) : (1 / 5 : ℚ) * (a * b) = 9 := by
  sorry

end one_fifth_of_five_times_nine_l2047_204780


namespace mike_initial_marbles_l2047_204739

-- Defining the conditions
def gave_marble (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles
def marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) : ℕ := initial_marbles - given_marbles

-- Using the given conditions
def initial_mike_marbles : ℕ := 8
def given_marbles : ℕ := 4
def remaining_marbles : ℕ := 4

-- Proving the statement
theorem mike_initial_marbles :
  initial_mike_marbles - given_marbles = remaining_marbles :=
by
  -- The proof
  sorry

end mike_initial_marbles_l2047_204739


namespace transformations_result_l2047_204740

theorem transformations_result :
  ∃ (r g : ℕ), r + g = 15 ∧ 
  21 + r - 5 * g = 0 ∧ 
  30 - 2 * r + 2 * g = 24 :=
by
  sorry

end transformations_result_l2047_204740


namespace find_a_of_normal_vector_l2047_204778

theorem find_a_of_normal_vector (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y + 5 = 0) ∧ (∃ n : ℝ × ℝ, n = (a, a - 2)) → a = 6 := by
  sorry

end find_a_of_normal_vector_l2047_204778


namespace orange_sacks_after_95_days_l2047_204718

-- Define the conditions as functions or constants
def harvest_per_day : ℕ := 150
def discard_per_day : ℕ := 135
def days_of_harvest : ℕ := 95

-- State the problem formally
theorem orange_sacks_after_95_days :
  (harvest_per_day - discard_per_day) * days_of_harvest = 1425 := 
by 
  sorry

end orange_sacks_after_95_days_l2047_204718


namespace books_not_sold_l2047_204709

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end books_not_sold_l2047_204709


namespace problem_l2047_204784

-- Define the variable
variable (x : ℝ)

-- Define the condition
def condition := 3 * x - 1 = 8

-- Define the statement to be proven
theorem problem (h : condition x) : 150 * (1 / x) + 2 = 52 :=
  sorry

end problem_l2047_204784


namespace range_of_a_l2047_204764

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end range_of_a_l2047_204764


namespace find_a_l2047_204732

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem find_a (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = -1 ∨ a = (1 / 3) :=
sorry

end find_a_l2047_204732


namespace weight_difference_at_end_of_year_l2047_204772

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end weight_difference_at_end_of_year_l2047_204772


namespace find_denominator_l2047_204762

theorem find_denominator (y : ℝ) (x : ℝ) (h₀ : y > 0) (h₁ : 9 * y / 20 + 3 * y / x = 0.75 * y) : x = 10 :=
sorry

end find_denominator_l2047_204762


namespace amount_made_per_jersey_l2047_204750

-- Definitions based on conditions
def total_revenue_from_jerseys : ℕ := 25740
def number_of_jerseys_sold : ℕ := 156

-- Theorem statement
theorem amount_made_per_jersey : 
  total_revenue_from_jerseys / number_of_jerseys_sold = 165 := 
by
  sorry

end amount_made_per_jersey_l2047_204750


namespace basketball_team_initial_players_l2047_204713

theorem basketball_team_initial_players
  (n : ℕ)
  (h_average_initial : Real := 190)
  (height_nikolai : Real := 197)
  (height_peter : Real := 181)
  (h_average_new : Real := 188)
  (total_height_initial : Real := h_average_initial * n)
  (total_height_new : Real := total_height_initial - (height_nikolai - height_peter))
  (avg_height_new_calculated : Real := total_height_new / n) :
  n = 8 :=
by
  sorry

end basketball_team_initial_players_l2047_204713


namespace remainder_when_divided_by_29_l2047_204785

theorem remainder_when_divided_by_29 (N : ℤ) (k : ℤ) (h : N = 751 * k + 53) : 
  N % 29 = 24 := 
by 
  sorry

end remainder_when_divided_by_29_l2047_204785


namespace charles_travel_time_l2047_204715

theorem charles_travel_time (D S T : ℕ) (hD : D = 6) (hS : S = 3) : T = D / S → T = 2 :=
by
  intros h
  rw [hD, hS] at h
  simp at h
  exact h

end charles_travel_time_l2047_204715


namespace forgotten_angle_l2047_204759

theorem forgotten_angle {n : ℕ} (h₁ : 2070 = (n - 2) * 180 - angle) : angle = 90 :=
by
  sorry

end forgotten_angle_l2047_204759


namespace proof_problem_l2047_204725

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1) → (0 ≤ x2) → (x1 ≠ x2) → (x1 - x2) * (f x1 - f x2) > 0

theorem proof_problem (f : ℝ → ℝ) (hf_even : even_function f) (hf_condition : condition f) :
  f 1 < f (-2) ∧ f (-2) < f 3 := sorry

end proof_problem_l2047_204725


namespace zero_ending_of_A_l2047_204777

theorem zero_ending_of_A (A : ℕ) (h : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c ∣ A ∧ a + b + c = 8 → a * b * c = 10) : 
  (10 ∣ A) ∧ ¬(100 ∣ A) :=
by
  sorry

end zero_ending_of_A_l2047_204777


namespace find_a_l2047_204755

theorem find_a :
  ∃ (a : ℤ), (∀ (x y : ℤ),
    (∃ (m n : ℤ), (x - 8 + m * y) * (x + 3 + n * y) = x^2 + 7 * x * y + a * y^2 - 5 * x - 45 * y - 24) ↔ a = 6) := 
sorry

end find_a_l2047_204755


namespace value_of_a_b_squared_l2047_204717

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a - b = Real.sqrt 2
axiom h2 : a * b = 4

theorem value_of_a_b_squared : (a + b)^2 = 18 := by
   sorry

end value_of_a_b_squared_l2047_204717


namespace line_equation_l2047_204737

-- Define the points A and M
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 3 1
def M := Point.mk 4 (-3)

def symmetric_point (A M : Point) : Point :=
  Point.mk (2 * M.x - A.x) (2 * M.y - A.y)

def line_through_origin (B : Point) : Prop :=
  7 * B.x + 5 * B.y = 0

theorem line_equation (B : Point) (hB : B = symmetric_point A M) : line_through_origin B :=
  by
  sorry

end line_equation_l2047_204737


namespace car_A_speed_l2047_204768

theorem car_A_speed (s_A s_B : ℝ) (d_AB d_extra t : ℝ) (h_s_B : s_B = 50) (h_d_AB : d_AB = 40) (h_d_extra : d_extra = 8) (h_time : t = 6) 
(h_distance_traveled_by_car_B : s_B * t = 300) 
(h_distance_difference : d_AB + d_extra = 48) :
  s_A = 58 :=
by
  sorry

end car_A_speed_l2047_204768


namespace student_error_difference_l2047_204728

theorem student_error_difference (num : ℤ) (num_val : num = 480) : 
  (5 / 6 * num - 5 / 16 * num) = 250 := 
by 
  sorry

end student_error_difference_l2047_204728


namespace class_B_has_more_stable_grades_l2047_204792

-- Definitions based on conditions
def avg_score_class_A : ℝ := 85
def avg_score_class_B : ℝ := 85
def var_score_class_A : ℝ := 120
def var_score_class_B : ℝ := 90

-- Proving which class has more stable grades (lower variance indicates more stability)
theorem class_B_has_more_stable_grades :
  var_score_class_B < var_score_class_A :=
by
  -- The proof will need to show the given condition and establish the inequality
  sorry

end class_B_has_more_stable_grades_l2047_204792


namespace factorize_expression_l2047_204790

theorem factorize_expression (a x y : ℝ) : 2 * x * (a - 2) - y * (2 - a) = (a - 2) * (2 * x + y) := 
by 
  sorry

end factorize_expression_l2047_204790


namespace monomial_2023_l2047_204783

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^n * (n + 1), n)

theorem monomial_2023 :
  monomial 2023 = (-2024, 2023) :=
by
  sorry

end monomial_2023_l2047_204783


namespace running_track_diameter_l2047_204741

theorem running_track_diameter 
  (running_track_width : ℕ) 
  (garden_ring_width : ℕ) 
  (play_area_diameter : ℕ) 
  (h1 : running_track_width = 4) 
  (h2 : garden_ring_width = 6) 
  (h3 : play_area_diameter = 14) :
  (2 * ((play_area_diameter / 2) + garden_ring_width + running_track_width)) = 34 := 
by
  sorry

end running_track_diameter_l2047_204741


namespace boarders_joined_l2047_204723

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ)
  (final_boarders : ℕ) (x : ℕ)
  (ratio_initial : initial_boarders * 16 = initial_day_scholars * 7)
  (ratio_final : final_boarders * 2 = initial_day_scholars)
  (final_boarders_eq : final_boarders = initial_boarders + x)
  (initial_boarders_val : initial_boarders = 560)
  (initial_day_scholars_val : initial_day_scholars = 1280)
  (final_boarders_val : final_boarders = 640) :
  x = 80 :=
by
  sorry

end boarders_joined_l2047_204723


namespace range_of_a_l2047_204748

noncomputable def f (x a : ℝ) := x^2 + 2 * x - a
noncomputable def g (x : ℝ) := 2 * x + 2 * Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2, (1/e) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ e ∧ f x1 a = g x1 ∧ f x2 a = g x2) ↔ 
  1 < a ∧ a ≤ (1/(e^2)) + 2 := 
sorry

end range_of_a_l2047_204748


namespace sin_alpha_minus_beta_l2047_204779

variables (α β : ℝ)

theorem sin_alpha_minus_beta (h1 : (Real.tan α / Real.tan β) = 7 / 13) 
    (h2 : Real.sin (α + β) = 2 / 3) :
    Real.sin (α - β) = -1 / 5 := 
sorry

end sin_alpha_minus_beta_l2047_204779


namespace swap_values_l2047_204775

theorem swap_values (A B : ℕ) (h₁ : A = 10) (h₂ : B = 20) : 
    let C := A 
    let A := B 
    let B := C
    A = 20 ∧ B = 10 := by
  let C := A
  let A := B
  let B := C
  have h₃ : C = 10 := h₁
  have h₄ : A = 20 := h₂
  have h₅ : B = 10 := h₃
  exact And.intro h₄ h₅

end swap_values_l2047_204775


namespace solve_system_of_equations_l2047_204704

variable {x : Fin 15 → ℤ}

theorem solve_system_of_equations (h : ∀ i : Fin 15, 1 - x i * x ((i + 1) % 15) = 0) :
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) :=
by
  -- Here we put the proof, but it's omitted for now.
  sorry

end solve_system_of_equations_l2047_204704


namespace nadine_hosing_time_l2047_204734

theorem nadine_hosing_time (shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) 
  (h1 : shampoos = 3) (h2 : time_per_shampoo = 15) (h3 : total_cleaning_time = 55) : 
  ∃ t : ℕ, t = total_cleaning_time - shampoos * time_per_shampoo ∧ t = 10 := 
by
  sorry

end nadine_hosing_time_l2047_204734


namespace commute_time_x_l2047_204700

theorem commute_time_x (d : ℝ) (walk_speed : ℝ) (train_speed : ℝ) (extra_time : ℝ) (diff_time : ℝ) :
  d = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  diff_time = 10 →
  (diff_time : ℝ) * 60 = (d / walk_speed - (d / train_speed + extra_time / 60)) * 60 →
  extra_time = 15.5 :=
by
  sorry

end commute_time_x_l2047_204700


namespace domain_of_f_l2047_204720

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 6) / sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : {x : ℝ | ∃ y, y = f x} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_f_l2047_204720


namespace Karen_sold_boxes_l2047_204795

theorem Karen_sold_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by
  sorry

end Karen_sold_boxes_l2047_204795


namespace Oliver_9th_l2047_204799

def person := ℕ → Prop

axiom Ruby : person
axiom Oliver : person
axiom Quinn : person
axiom Pedro : person
axiom Nina : person
axiom Samuel : person
axiom place : person → ℕ → Prop

-- Conditions given in the problem
axiom Ruby_Oliver : ∀ n, place Ruby n → place Oliver (n + 7)
axiom Quinn_Pedro : ∀ n, place Quinn n → place Pedro (n - 2)
axiom Nina_Oliver : ∀ n, place Nina n → place Oliver (n + 3)
axiom Pedro_Samuel : ∀ n, place Pedro n → place Samuel (n - 3)
axiom Samuel_Ruby : ∀ n, place Samuel n → place Ruby (n + 2)
axiom Quinn_5th : place Quinn 5

-- Question: Prove that Oliver finished in 9th place
theorem Oliver_9th : place Oliver 9 :=
sorry

end Oliver_9th_l2047_204799


namespace scientific_notation_l2047_204758

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end scientific_notation_l2047_204758


namespace river_flow_rate_l2047_204770

-- Define the conditions
def depth : ℝ := 8
def width : ℝ := 25
def volume_per_min : ℝ := 26666.666666666668

-- The main theorem proving the rate at which the river is flowing
theorem river_flow_rate : (volume_per_min / (depth * width)) = 133.33333333333334 := by
  -- Express the area of the river's cross-section
  let area := depth * width
  -- Define the velocity based on the given volume and calculated area
  let velocity := volume_per_min / area
  -- Simplify and derive the result
  show velocity = 133.33333333333334
  sorry

end river_flow_rate_l2047_204770


namespace exists_pythagorean_triple_rational_k_l2047_204757

theorem exists_pythagorean_triple_rational_k (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ), (a^2 + b^2 = c^2) ∧ ((a + c : ℚ) / b = k) := by
  sorry

end exists_pythagorean_triple_rational_k_l2047_204757


namespace arctan_sum_of_roots_l2047_204752

theorem arctan_sum_of_roots (u v w : ℝ) (h1 : u + v + w = 0) (h2 : u * v + v * w + w * u = -10) (h3 : u * v * w = -11) :
  Real.arctan u + Real.arctan v + Real.arctan w = π / 4 :=
by
  sorry

end arctan_sum_of_roots_l2047_204752


namespace jims_final_paycheck_l2047_204747

noncomputable def final_paycheck (g r t h m b btr : ℝ) := 
  let retirement := g * r
  let gym := m / 2
  let net_before_bonus := g - retirement - t - h - gym
  let after_tax_bonus := b * (1 - btr)
  net_before_bonus + after_tax_bonus

theorem jims_final_paycheck :
  final_paycheck 1120 0.25 100 200 50 500 0.30 = 865 :=
by
  sorry

end jims_final_paycheck_l2047_204747


namespace find_n_l2047_204736

theorem find_n (n : ℕ) (h1 : n > 13) (h2 : (12 : ℚ) / (n - 1 : ℚ) = 1 / 3) : n = 37 := by
  sorry

end find_n_l2047_204736


namespace exists_xy_l2047_204722

open Classical

variable (f : ℝ → ℝ)

theorem exists_xy (h : ∃ x₀ y₀ : ℝ, f x₀ ≠ f y₀) : ∃ x y : ℝ, f (x + y) < f (x * y) :=
by
  sorry

end exists_xy_l2047_204722


namespace mean_of_set_is_12_point_8_l2047_204774

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l2047_204774


namespace cost_price_per_meter_l2047_204791

theorem cost_price_per_meter
  (S : ℝ) (L : ℝ) (C : ℝ) (total_meters : ℝ) (total_price : ℝ)
  (h1 : total_meters = 400) (h2 : total_price = 18000)
  (h3 : L = 5) (h4 : S = total_price / total_meters) 
  (h5 : C = S + L) :
  C = 50 :=
by
  sorry

end cost_price_per_meter_l2047_204791


namespace calculate_mean_score_l2047_204754

theorem calculate_mean_score (M SD : ℝ) 
  (h1 : M - 2 * SD = 60)
  (h2 : M + 3 * SD = 100) : 
  M = 76 :=
by
  sorry

end calculate_mean_score_l2047_204754


namespace find_specified_time_l2047_204735

theorem find_specified_time (distance : ℕ) (slow_time fast_time : ℕ → ℕ) (fast_is_double : ∀ x, fast_time x = 2 * slow_time x)
  (distance_value : distance = 900) (slow_time_eq : ∀ x, slow_time x = x + 1) (fast_time_eq : ∀ x, fast_time x = x - 3) :
  2 * (distance / (slow_time x)) = distance / (fast_time x) :=
by
  intros
  rw [distance_value, slow_time_eq, fast_time_eq]
  sorry

end find_specified_time_l2047_204735


namespace three_primes_sum_odd_l2047_204701

theorem three_primes_sum_odd (primes : Finset ℕ) (h_prime : ∀ p ∈ primes, Prime p) :
  primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} →
  (Nat.choose 9 3 / Nat.choose 10 3 : ℚ) = 7 / 10 := by
  -- Let the set of first ten prime numbers.
  -- As per condition, primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  -- Then show that the probability calculation yields 7/10
  sorry

end three_primes_sum_odd_l2047_204701


namespace solve_for_x_l2047_204724

theorem solve_for_x : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108 / 19 := 
by 
  sorry

end solve_for_x_l2047_204724


namespace oldest_bride_age_l2047_204796

theorem oldest_bride_age (G B : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) : B = 102 :=
by
  sorry

end oldest_bride_age_l2047_204796


namespace find_angle_x_l2047_204703

noncomputable def angle_x (angle_ABC angle_ACB angle_CDE : ℝ) : ℝ :=
  let angle_BAC := 180 - angle_ABC - angle_ACB
  let angle_ADE := 180 - angle_CDE
  let angle_EAD := angle_BAC
  let angle_AED := 180 - angle_ADE - angle_EAD
  180 - angle_AED

theorem find_angle_x (angle_ABC angle_ACB angle_CDE : ℝ) :
  angle_ABC = 70 → angle_ACB = 90 → angle_CDE = 42 → angle_x angle_ABC angle_ACB angle_CDE = 158 :=
by
  intros hABC hACB hCDE
  simp [angle_x, hABC, hACB, hCDE]
  sorry

end find_angle_x_l2047_204703


namespace sarah_socks_l2047_204746

theorem sarah_socks :
  ∃ (a b c : ℕ), a + b + c = 15 ∧ 2 * a + 4 * b + 5 * c = 45 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ (a = 8 ∨ a = 9) :=
by {
  sorry
}

end sarah_socks_l2047_204746


namespace find_m_n_l2047_204781

theorem find_m_n (a b : ℝ) (m n : ℤ) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l2047_204781


namespace parabola_directrix_l2047_204744

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l2047_204744


namespace edge_length_of_prism_l2047_204702

-- Definitions based on conditions
def rectangular_prism_edges : ℕ := 12
def total_edge_length : ℕ := 72

-- Proof problem statement
theorem edge_length_of_prism (num_edges : ℕ) (total_length : ℕ) (h1 : num_edges = rectangular_prism_edges) (h2 : total_length = total_edge_length) : 
  (total_length / num_edges) = 6 :=
by {
  -- The proof is omitted here as instructed
  sorry
}

end edge_length_of_prism_l2047_204702


namespace YZ_length_l2047_204776

theorem YZ_length : 
  ∀ (X Y Z : Type) 
  (angle_Y angle_Z angle_X : ℝ)
  (XZ YZ : ℝ),
  angle_Y = 45 ∧ angle_Z = 60 ∧ XZ = 6 →
  angle_X = 180 - angle_Y - angle_Z →
  YZ = XZ * (Real.sin angle_X / Real.sin angle_Y) →
  YZ = 3 * (Real.sqrt 6 + Real.sqrt 2) :=
by
  intros X Y Z angle_Y angle_Z angle_X XZ YZ
  intro h1 h2 h3
  sorry

end YZ_length_l2047_204776


namespace total_students_in_college_l2047_204769

theorem total_students_in_college 
  (girls : ℕ) 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (h_ratio : ratio_boys = 8) 
  (h_ratio_girls : ratio_girls = 5) 
  (h_girls : girls = 400) 
  : (ratio_boys * (girls / ratio_girls) + girls = 1040) := 
by 
  sorry

end total_students_in_college_l2047_204769


namespace find_value_of_a_plus_b_l2047_204793

noncomputable def A (a b : ℤ) : Set ℤ := {1, a, b}
noncomputable def B (a b : ℤ) : Set ℤ := {a, a^2, a * b}

theorem find_value_of_a_plus_b (a b : ℤ) (h : A a b = B a b) : a + b = -1 :=
by sorry

end find_value_of_a_plus_b_l2047_204793


namespace find_y_from_equation_l2047_204729

theorem find_y_from_equation :
  ∀ y : ℕ, (12 ^ 3 * 6 ^ 4) / y = 5184 → y = 432 :=
by
  sorry

end find_y_from_equation_l2047_204729


namespace elmer_more_than_penelope_l2047_204738

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end elmer_more_than_penelope_l2047_204738


namespace li_payment_l2047_204743

noncomputable def payment_li (daily_payment_per_unit : ℚ) (days_li_worked : ℕ) : ℚ :=
daily_payment_per_unit * days_li_worked

theorem li_payment (work_per_day : ℚ) (days_li_worked : ℕ) (days_extra_work : ℕ) 
  (difference_payment : ℚ) (daily_payment_per_unit : ℚ) (initial_nanual_workdays : ℕ) :
  work_per_day = 1 →
  days_li_worked = 2 →
  days_extra_work = 3 →
  difference_payment = 2700 →
  daily_payment_per_unit = difference_payment / (initial_nanual_workdays + (3 * 3)) → 
  payment_li daily_payment_per_unit days_li_worked = 450 := 
by 
  intros h_work_per_day h_days_li_worked h_days_extra_work h_diff_payment h_daily_payment 
  sorry

end li_payment_l2047_204743


namespace rectangle_height_l2047_204705

-- Define the given right-angled triangle with its legs and hypotenuse
variables {a b c d : ℝ}

-- Define the conditions: Right-angled triangle with legs a, b and hypotenuse c
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the height of the inscribed rectangle is d
def height_of_rectangle (a b d : ℝ) : Prop :=
  d = a + b

-- The problem statement: Prove that the height of the rectangle is the sum of the heights of the squares
theorem rectangle_height (a b c d : ℝ) (ht : right_angled_triangle a b c) : height_of_rectangle a b d :=
by
  sorry

end rectangle_height_l2047_204705


namespace dividend_calculation_l2047_204751

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 18) (h2 : quotient = 9) (h3 : remainder = 5) : 
  (divisor * quotient + remainder = 167) :=
by
  sorry

end dividend_calculation_l2047_204751


namespace total_pictures_l2047_204782

noncomputable def RandyPics : ℕ := 5
noncomputable def PeterPics : ℕ := RandyPics + 3
noncomputable def QuincyPics : ℕ := PeterPics + 20

theorem total_pictures :
  RandyPics + PeterPics + QuincyPics = 41 :=
by
  sorry

end total_pictures_l2047_204782


namespace base_10_to_base_7_equiv_base_10_to_base_7_678_l2047_204749

theorem base_10_to_base_7_equiv : (678 : ℕ) = 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 := 
by
  -- proof steps would go here
  sorry

theorem base_10_to_base_7_678 : "678 in base-7" = "1656" := 
by
  have h1 := base_10_to_base_7_equiv
  -- additional proof steps to show 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 = 1656 in base-7
  sorry

end base_10_to_base_7_equiv_base_10_to_base_7_678_l2047_204749


namespace John_next_birthday_age_l2047_204763

variable (John Mike Lucas : ℝ)

def John_is_25_percent_older_than_Mike := John = 1.25 * Mike
def Mike_is_30_percent_younger_than_Lucas := Mike = 0.7 * Lucas
def sum_of_ages_is_27_point_3_years := John + Mike + Lucas = 27.3

theorem John_next_birthday_age 
  (h1 : John_is_25_percent_older_than_Mike John Mike) 
  (h2 : Mike_is_30_percent_younger_than_Lucas Mike Lucas) 
  (h3 : sum_of_ages_is_27_point_3_years John Mike Lucas) : 
  John + 1 = 10 := 
sorry

end John_next_birthday_age_l2047_204763


namespace find_a_l2047_204716

theorem find_a
  (r1 r2 r3 : ℕ)
  (hr1 : r1 > 2) (hr2 : r2 > 2) (hr3 : r3 > 2)
  (a b c : ℤ)
  (hr : (Polynomial.X - Polynomial.C (r1 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r2 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r3 : ℤ)) = 
         Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
  (h : a + b + c + 1 = -2009) :
  a = -58 := sorry

end find_a_l2047_204716


namespace quadratic_function_range_l2047_204714

theorem quadratic_function_range (a b c : ℝ) (x y : ℝ) :
  (∀ x, x = -4 → y = a * (-4)^2 + b * (-4) + c → y = 3) ∧
  (∀ x, x = -3 → y = a * (-3)^2 + b * (-3) + c → y = -2) ∧
  (∀ x, x = -2 → y = a * (-2)^2 + b * (-2) + c → y = -5) ∧
  (∀ x, x = -1 → y = a * (-1)^2 + b * (-1) + c → y = -6) ∧
  (∀ x, x = 0 → y = a * 0^2 + b * 0 + c → y = -5) →
  (∀ x, x < -2 → y > -5) :=
sorry

end quadratic_function_range_l2047_204714


namespace paint_replacement_fractions_l2047_204707

variables {r b g : ℚ}

/-- Given the initial and replacement intensities and the final intensities of red, blue,
and green paints respectively, prove the fractions of the original amounts of each paint color
that were replaced. -/
theorem paint_replacement_fractions :
  (0.6 * (1 - r) + 0.3 * r = 0.4) ∧
  (0.4 * (1 - b) + 0.15 * b = 0.25) ∧
  (0.25 * (1 - g) + 0.1 * g = 0.18) →
  (r = 2/3) ∧ (b = 3/5) ∧ (g = 7/15) :=
by
  sorry

end paint_replacement_fractions_l2047_204707


namespace number_of_students_l2047_204773

theorem number_of_students (n : ℕ) (A : ℕ) 
  (h1 : A = 10 * n)
  (h2 : (A - 11 + 41) / n = 11) :
  n = 30 := 
sorry

end number_of_students_l2047_204773


namespace equilateral_triangle_area_decrease_l2047_204797

theorem equilateral_triangle_area_decrease (s : ℝ) (A : ℝ) (s_new : ℝ) (A_new : ℝ)
    (hA : A = 100 * Real.sqrt 3)
    (hs : s^2 = 400)
    (hs_new : s_new = s - 6)
    (hA_new : A_new = (Real.sqrt 3 / 4) * s_new^2) :
    (A - A_new) / A * 100 = 51 := by
  sorry

end equilateral_triangle_area_decrease_l2047_204797


namespace vector_addition_correct_dot_product_correct_l2047_204753

-- Define the two vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- Define the expected results
def a_plus_b_expected : ℝ × ℝ := (4, 3)
def a_dot_b_expected : ℝ := 5

-- Prove the sum of vectors a and b
theorem vector_addition_correct : a + b = a_plus_b_expected := by
  sorry

-- Prove the dot product of vectors a and b
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = a_dot_b_expected := by
  sorry

end vector_addition_correct_dot_product_correct_l2047_204753


namespace parity_of_pq_l2047_204787

theorem parity_of_pq (x y m n p q : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 0)
    (hx : x = p) (hy : y = q) (h1 : x - 1998 * y = n) (h2 : 1999 * x + 3 * y = m) :
    p % 2 = 0 ∧ q % 2 = 1 :=
by
  sorry

end parity_of_pq_l2047_204787


namespace irreducible_fraction_unique_l2047_204767

theorem irreducible_fraction_unique :
  ∃ (a b : ℕ), a = 5 ∧ b = 2 ∧ gcd a b = 1 ∧ (∃ n : ℕ, 10^n = a * b) :=
by
  sorry

end irreducible_fraction_unique_l2047_204767


namespace tree_F_height_l2047_204710

variable (A B C D E F : ℝ)

def height_conditions : Prop :=
  A = 150 ∧ -- Tree A's height is 150 feet
  B = (2 / 3) * A ∧ -- Tree B's height is 2/3 of Tree A's height
  C = (1 / 2) * B ∧ -- Tree C's height is 1/2 of Tree B's height
  D = C + 25 ∧ -- Tree D's height is 25 feet more than Tree C's height
  E = 0.40 * A ∧ -- Tree E's height is 40% of Tree A's height
  F = (B + D) / 2 -- Tree F's height is the average of Tree B's height and Tree D's height

theorem tree_F_height : height_conditions A B C D E F → F = 87.5 :=
by
  intros
  sorry

end tree_F_height_l2047_204710


namespace integrate_differential_eq_l2047_204721

theorem integrate_differential_eq {x y C : ℝ} {y' : ℝ → ℝ → ℝ} (h : ∀ x y, (4 * y - 3 * x - 5) * y' x y + 7 * x - 3 * y + 2 = 0) : 
    ∃ C : ℝ, ∀ x y : ℝ, 2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C :=
by
  sorry

end integrate_differential_eq_l2047_204721


namespace max_value_of_cubes_l2047_204798

theorem max_value_of_cubes 
  (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  x^3 + y^3 + z^3 ≤ 27 :=
  sorry

end max_value_of_cubes_l2047_204798


namespace find_a2_l2047_204727

def S (n : Nat) (a1 d : Int) : Int :=
  n * a1 + (n * (n - 1) * d) / 2

theorem find_a2 (a1 : Int) (d : Int) :
  a1 = -2010 ∧
  (S 2010 a1 d) / 2010 - (S 2008 a1 d) / 2008 = 2 →
  a1 + d = -2008 :=
by
  sorry

end find_a2_l2047_204727


namespace sum_geometric_sequence_first_eight_terms_l2047_204786

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l2047_204786


namespace candidates_count_l2047_204712

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 := 
sorry

end candidates_count_l2047_204712


namespace fraction_picked_l2047_204745

/--
An apple tree has three times as many apples as the number of plums on a plum tree.
Damien picks a certain fraction of the fruits from the trees, and there are 96 plums
and apples remaining on the tree. There were 180 apples on the apple tree before 
Damien picked any of the fruits. Prove that Damien picked 3/5 of the fruits from the trees.
-/
theorem fraction_picked (P F : ℝ) (h1 : 3 * P = 180) (h2 : (1 - F) * (180 + P) = 96) :
  F = 3 / 5 :=
by
  sorry

end fraction_picked_l2047_204745


namespace income_increase_is_60_percent_l2047_204733

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

end income_increase_is_60_percent_l2047_204733


namespace candidate_percentage_l2047_204788

variables (M T : ℝ)

theorem candidate_percentage (h1 : (P / 100) * T = M - 30) 
                             (h2 : (45 / 100) * T = M + 15)
                             (h3 : M = 120) : 
                             P = 30 := 
by 
  sorry

end candidate_percentage_l2047_204788


namespace ajhsme_1989_reappears_at_12_l2047_204730

def cycle_length_letters : ℕ := 6
def cycle_length_digits  : ℕ := 4
def target_position : ℕ := Nat.lcm cycle_length_letters cycle_length_digits

theorem ajhsme_1989_reappears_at_12 :
  target_position = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end ajhsme_1989_reappears_at_12_l2047_204730


namespace travel_time_correct_l2047_204711

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l2047_204711


namespace find_greater_number_l2047_204794

theorem find_greater_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x > y) : x = 25 := 
sorry

end find_greater_number_l2047_204794


namespace base9_to_base10_l2047_204756

def num_base9 : ℕ := 521 -- Represents 521_9
def base : ℕ := 9

theorem base9_to_base10 : 
  (1 * base^0 + 2 * base^1 + 5 * base^2) = 424 := 
by
  -- Sorry allows us to skip the proof.
  sorry

end base9_to_base10_l2047_204756


namespace find_a_l2047_204760

theorem find_a
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, x = 1 ∨ x = 2 → a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0)
  (h2 : a + b + c = 2) : 
  a = 12 := 
sorry

end find_a_l2047_204760


namespace soccer_players_count_l2047_204719

theorem soccer_players_count (total_socks : ℕ) (P : ℕ) 
  (h_total_socks : total_socks = 22)
  (h_each_player_contributes : ∀ p : ℕ, p = P → total_socks = 2 * P) :
  P = 11 :=
by
  sorry

end soccer_players_count_l2047_204719


namespace notebooks_cost_l2047_204789

theorem notebooks_cost 
  (P N : ℝ)
  (h1 : 96 * P + 24 * N = 520)
  (h2 : ∃ x : ℝ, 3 * P + x * N = 60)
  (h3 : P + N = 15.512820512820513) :
  ∃ x : ℕ, x = 4 :=
by
  sorry

end notebooks_cost_l2047_204789


namespace prob_two_girls_l2047_204708

variable (Pboy Pgirl : ℝ)

-- Conditions
def prob_boy : Prop := Pboy = 1 / 2
def prob_girl : Prop := Pgirl = 1 / 2

-- The theorem to be proven
theorem prob_two_girls (h₁ : prob_boy Pboy) (h₂ : prob_girl Pgirl) : (Pgirl * Pgirl) = 1 / 4 :=
by
  sorry

end prob_two_girls_l2047_204708


namespace dylan_trip_time_l2047_204706

def total_time_of_trip (d1 d2 d3 v1 v2 v3 b : ℕ) : ℝ :=
  let t1 := d1 / v1
  let t2 := d2 / v2
  let t3 := d3 / v3
  let time_riding := t1 + t2 + t3
  let time_breaks := b * 25 / 60
  time_riding + time_breaks

theorem dylan_trip_time :
  total_time_of_trip 400 150 700 50 40 60 3 = 24.67 :=
by
  unfold total_time_of_trip
  sorry

end dylan_trip_time_l2047_204706


namespace ratio_of_luxury_to_suv_l2047_204731

variable (E L S : Nat)

-- Conditions
def condition1 := E * 2 = L * 3
def condition2 := E * 1 = S * 4

-- The statement to prove
theorem ratio_of_luxury_to_suv 
  (h1 : condition1 E L)
  (h2 : condition2 E S) :
  L * 3 = S * 8 :=
by sorry

end ratio_of_luxury_to_suv_l2047_204731


namespace max_value_l2047_204771

theorem max_value (y : ℝ) (h : y ≠ 0) : 
  ∃ M, M = 1 / 25 ∧ 
       ∀ y ≠ 0,  ∀ value, value = y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) 
       → value ≤ M :=
sorry

end max_value_l2047_204771
