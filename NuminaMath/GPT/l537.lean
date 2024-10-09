import Mathlib

namespace motorcyclist_travel_time_l537_53714

-- Define the conditions and the proof goal:
theorem motorcyclist_travel_time :
  ∀ (z : ℝ) (t₁ t₂ t₃ : ℝ),
    t₂ = 60 →
    t₃ = 3240 →
    (t₃ - 5) / (z / 40 - z / t₁) = 10 →
    t₃ / (z / 40) = 10 + t₂ / (z / 60 - z / t₁) →
    t₁ = 80 :=
by
  intros z t₁ t₂ t₃ h1 h2 h3 h4
  sorry

end motorcyclist_travel_time_l537_53714


namespace determine_multiplier_l537_53791

theorem determine_multiplier (x : ℝ) : 125 * x - 138 = 112 → x = 2 :=
by
  sorry

end determine_multiplier_l537_53791


namespace abs_neg_three_eq_three_l537_53726

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
sorry

end abs_neg_three_eq_three_l537_53726


namespace min_cosine_largest_angle_l537_53761

theorem min_cosine_largest_angle (a b c : ℕ → ℝ) 
  (triangle_inequality: ∀ i, a i ≤ b i ∧ b i ≤ c i)
  (pythagorean_inequality: ∀ i, (a i)^2 + (b i)^2 ≥ (c i)^2)
  (A : ℝ := ∑' i, a i)
  (B : ℝ := ∑' i, b i)
  (C : ℝ := ∑' i, c i) :
  (A^2 + B^2 - C^2) / (2 * A * B) ≥ 1 - (Real.sqrt 2) :=
sorry

end min_cosine_largest_angle_l537_53761


namespace evaluate_expression_l537_53719

theorem evaluate_expression :
  ((gcd 54 42 |> lcm 36) * (gcd 78 66 |> gcd 90) + (lcm 108 72 |> gcd 66 |> gcd 84)) = 24624 := by
  sorry

end evaluate_expression_l537_53719


namespace smallest_sum_xy_l537_53782

theorem smallest_sum_xy (x y : ℕ) (hx : x ≠ y) (h : 0 < x ∧ 0 < y) (hxy : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_sum_xy_l537_53782


namespace right_triangle_point_selection_l537_53792

theorem right_triangle_point_selection : 
  let n := 200 
  let rows := 2
  (rows * (n - 22 + 1)) + 2 * (rows * (n - 122 + 1)) + (n * (2 * (n - 1))) = 80268 := 
by 
  let rows := 2
  let n := 200
  let case1a := rows * (n - 22 + 1)
  let case1b := 2 * (rows * (n - 122 + 1))
  let case2 := n * (2 * (n - 1))
  have h : case1a + case1b + case2 = 80268 := by sorry
  exact h

end right_triangle_point_selection_l537_53792


namespace combined_cost_increase_l537_53701

def original_bicycle_cost : ℝ := 200
def original_skates_cost : ℝ := 50
def bike_increase_percent : ℝ := 0.06
def skates_increase_percent : ℝ := 0.15

noncomputable def new_bicycle_cost : ℝ := original_bicycle_cost * (1 + bike_increase_percent)
noncomputable def new_skates_cost : ℝ := original_skates_cost * (1 + skates_increase_percent)
noncomputable def original_total_cost : ℝ := original_bicycle_cost + original_skates_cost
noncomputable def new_total_cost : ℝ := new_bicycle_cost + new_skates_cost
noncomputable def total_increase : ℝ := new_total_cost - original_total_cost
noncomputable def percent_increase : ℝ := (total_increase / original_total_cost) * 100

theorem combined_cost_increase : percent_increase = 7.8 := by
  sorry

end combined_cost_increase_l537_53701


namespace marvelous_class_student_count_l537_53735

theorem marvelous_class_student_count (g : ℕ) (jb : ℕ) (jg : ℕ) (j_total : ℕ) (jl : ℕ) (init_jb : ℕ) : 
  jb = g + 3 →  -- Number of boys
  jg = 2 * g + 1 →  -- Jelly beans received by each girl
  init_jb = 726 →  -- Initial jelly beans
  jl = 4 →  -- Leftover jelly beans
  j_total = init_jb - jl →  -- Jelly beans distributed
  (jb * jb + g * jg = j_total) → -- Total jelly beans distributed equation
  2 * g + 1 + g + jb = 31 := -- Total number of students
by
  sorry

end marvelous_class_student_count_l537_53735


namespace slices_of_bread_left_l537_53712

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l537_53712


namespace elois_banana_bread_l537_53766

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l537_53766


namespace converse_even_power_divisible_l537_53784

theorem converse_even_power_divisible (n : ℕ) (h_even : ∀ (k : ℕ), n = 2 * k → (3^n + 63) % 72 = 0) :
  (3^n + 63) % 72 = 0 → ∃ (k : ℕ), n = 2 * k :=
by sorry

end converse_even_power_divisible_l537_53784


namespace students_in_canteen_l537_53741

-- Definitions for conditions
def total_students : ℕ := 40
def absent_fraction : ℚ := 1 / 10
def classroom_fraction : ℚ := 3 / 4

-- Lean 4 statement
theorem students_in_canteen :
  let absent_students := (absent_fraction * total_students)
  let present_students := (total_students - absent_students)
  let classroom_students := (classroom_fraction * present_students)
  let canteen_students := (present_students - classroom_students)
  canteen_students = 9 := by
    sorry

end students_in_canteen_l537_53741


namespace sphere_radius_equals_three_l537_53724

noncomputable def radius_of_sphere : ℝ := 3

theorem sphere_radius_equals_three {R : ℝ} (h1 : 4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) : 
  R = radius_of_sphere :=
by
  sorry

end sphere_radius_equals_three_l537_53724


namespace correct_angle_calculation_l537_53787

theorem correct_angle_calculation (α β : ℝ) (hα : 0 < α ∧ α < 90) (hβ : 90 < β ∧ β < 180) :
    22.5 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 67.5 → 0.25 * (α + β) = 45.3 :=
by
  sorry

end correct_angle_calculation_l537_53787


namespace raft_people_with_life_jackets_l537_53774

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l537_53774


namespace Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l537_53755

theorem Mishas_fathers_speed (d : ℝ) (t : ℝ) (V : ℝ) 
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) :
  V = 1 :=
by
  sorry

theorem Mishas_fathers_speed_in_kmh (d : ℝ) (t : ℝ) (V : ℝ) (V_kmh : ℝ)
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) 
  (h4 : V_kmh = V * 60):
  V_kmh = 60 :=
by
  sorry

end Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l537_53755


namespace bottom_price_l537_53788

open Nat

theorem bottom_price (B T : ℕ) (h1 : T = B + 300) (h2 : 3 * B + 3 * T = 21000) : B = 3350 := by
  sorry

end bottom_price_l537_53788


namespace min_tables_42_l537_53707

def min_tables_needed (total_people : ℕ) (table_sizes : List ℕ) : ℕ :=
  sorry

theorem min_tables_42 :
  min_tables_needed 42 [4, 6, 8] = 6 :=
sorry

end min_tables_42_l537_53707


namespace birds_in_tree_l537_53729

def initialBirds : Nat := 14
def additionalBirds : Nat := 21
def totalBirds := initialBirds + additionalBirds

theorem birds_in_tree : totalBirds = 35 := by
  sorry

end birds_in_tree_l537_53729


namespace perfect_square_trinomial_l537_53794

theorem perfect_square_trinomial (a b m : ℝ) :
  (∃ x : ℝ, a^2 + mab + b^2 = (x + b)^2 ∨ a^2 + mab + b^2 = (x - b)^2) ↔ (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l537_53794


namespace determine_n_l537_53773

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem determine_n :
  (∃ n : ℕ, digit_sum (9 * (10^n - 1)) = 999 ∧ n = 111) :=
sorry

end determine_n_l537_53773


namespace find_integer_m_l537_53770

theorem find_integer_m 
  (m : ℤ) (h_pos : m > 0) 
  (h_intersect : ∃ (x y : ℤ), 17 * x + 7 * y = 1000 ∧ y = m * x + 2) : 
  m = 68 :=
by
  sorry

end find_integer_m_l537_53770


namespace is_factor_l537_53760

-- Define the polynomial
def poly (x : ℝ) := x^4 + 4 * x^2 + 4

-- Define a candidate for being a factor
def factor_candidate (x : ℝ) := x^2 + 2

-- Proof problem: prove that factor_candidate is a factor of poly
theorem is_factor : ∀ x : ℝ, poly x = factor_candidate x * factor_candidate x := 
by
  intro x
  unfold poly factor_candidate
  sorry

end is_factor_l537_53760


namespace ratio_of_areas_of_circles_l537_53768

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l537_53768


namespace range_of_a_l537_53752

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
by
  sorry

end range_of_a_l537_53752


namespace bananas_distribution_l537_53746

noncomputable def total_bananas : ℝ := 550.5
noncomputable def lydia_bananas : ℝ := 80.25
noncomputable def dawn_bananas : ℝ := lydia_bananas + 93
noncomputable def emily_bananas : ℝ := 198
noncomputable def donna_bananas : ℝ := emily_bananas / 2

theorem bananas_distribution :
  dawn_bananas = 173.25 ∧
  lydia_bananas = 80.25 ∧
  donna_bananas = 99 ∧
  emily_bananas = 198 ∧
  dawn_bananas + lydia_bananas + donna_bananas + emily_bananas = total_bananas :=
by
  sorry

end bananas_distribution_l537_53746


namespace fraction_sum_eq_l537_53745

theorem fraction_sum_eq : (7 / 10 : ℚ) + (3 / 100) + (9 / 1000) = 0.739 := sorry

end fraction_sum_eq_l537_53745


namespace inequality_am_gm_l537_53764

theorem inequality_am_gm (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by
  sorry

end inequality_am_gm_l537_53764


namespace misha_second_round_score_l537_53737

def misha_score_first_round (darts : ℕ) (score_per_dart_min : ℕ) : ℕ := 
  darts * score_per_dart_min

def misha_score_second_round (score_first : ℕ) (multiplier : ℕ) : ℕ := 
  score_first * multiplier

def misha_score_third_round (score_second : ℕ) (multiplier : ℚ) : ℚ := 
  score_second * multiplier

theorem misha_second_round_score (darts : ℕ) (score_per_dart_min : ℕ) (multiplier_second : ℕ) (multiplier_third : ℚ) 
  (h_darts : darts = 8) (h_score_per_dart_min : score_per_dart_min = 3) (h_multiplier_second : multiplier_second = 2) (h_multiplier_third : multiplier_third = 1.5) :
  misha_score_second_round (misha_score_first_round darts score_per_dart_min) multiplier_second = 48 :=
by sorry

end misha_second_round_score_l537_53737


namespace coloring_even_conditional_l537_53702

-- Define the problem parameters and constraints
def number_of_colorings (n : Nat) (even_red : Bool) (even_yellow : Bool) : Nat :=
  sorry  -- This function would contain the detailed computational logic.

-- Define the main theorem statement
theorem coloring_even_conditional (n : ℕ) (h1 : n > 0) : ∃ C : Nat, number_of_colorings n true true = C := 
by
  sorry  -- The proof would go here.


end coloring_even_conditional_l537_53702


namespace vertical_shirts_count_l537_53740

-- Definitions from conditions
def total_people : ℕ := 40
def checkered_shirts : ℕ := 7
def horizontal_shirts := 4 * checkered_shirts

-- Proof goal
theorem vertical_shirts_count :
  ∃ vertical_shirts : ℕ, vertical_shirts = total_people - (checkered_shirts + horizontal_shirts) ∧ vertical_shirts = 5 :=
sorry

end vertical_shirts_count_l537_53740


namespace table_relation_l537_53739

theorem table_relation (x y : ℕ) (hx : x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6) :
  (y = 3 ∧ x = 2) ∨ (y = 8 ∧ x = 3) ∨ (y = 15 ∧ x = 4) ∨ (y = 24 ∧ x = 5) ∨ (y = 35 ∧ x = 6) ↔ 
  y = x^2 - x + 2 :=
sorry

end table_relation_l537_53739


namespace eventually_repeating_last_two_digits_l537_53796

theorem eventually_repeating_last_two_digits (K : ℕ) : ∃ N : ℕ, ∃ t : ℕ, 
    (∃ s : ℕ, t = s * 77 + N) ∨ (∃ u : ℕ, t = u * 54 + N) ∧ (t % 100) / 10 = (t % 100) % 10 :=
sorry

end eventually_repeating_last_two_digits_l537_53796


namespace probability_two_dice_same_number_l537_53750

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l537_53750


namespace journal_sessions_per_week_l537_53711

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

end journal_sessions_per_week_l537_53711


namespace part1_part2_l537_53716

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.sin (x + Real.pi / 3) - 1

theorem part1 : f (5 * Real.pi / 6) = -2 := by
  sorry

variables {A : ℝ} (hA1 : A > 0) (hA2 : A ≤ Real.pi / 3) (hFA : f A = 8 / 5)

theorem part2 (h : A > 0 ∧ A ≤ Real.pi / 3 ∧ f A = 8 / 5) : f (A + Real.pi / 4) = 6 / 5 :=
by
  sorry

end part1_part2_l537_53716


namespace gold_bars_lost_l537_53713

-- Define the problem constants
def initial_bars : ℕ := 100
def friends : ℕ := 4
def bars_per_friend : ℕ := 20

-- Define the total distributed gold bars
def total_distributed : ℕ := friends * bars_per_friend

-- Define the number of lost gold bars
def lost_bars : ℕ := initial_bars - total_distributed

-- Theorem: Prove that the number of lost gold bars is 20
theorem gold_bars_lost : lost_bars = 20 := by
  sorry

end gold_bars_lost_l537_53713


namespace complete_the_square_3x2_9x_20_l537_53742

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l537_53742


namespace minimum_value_fraction_l537_53723

theorem minimum_value_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 :=
sorry

end minimum_value_fraction_l537_53723


namespace arcade_ticket_problem_l537_53777

-- Define all the conditions given in the problem
def initial_tickets : Nat := 13
def used_tickets : Nat := 8
def more_tickets_for_clothes : Nat := 10
def tickets_for_toys : Nat := 8
def tickets_for_clothes := tickets_for_toys + more_tickets_for_clothes

-- The proof statement (goal)
theorem arcade_ticket_problem : tickets_for_clothes = 18 := by
  -- This is where the proof would go
  sorry

end arcade_ticket_problem_l537_53777


namespace _l537_53730

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 1 1 1 := 
by {
  -- Prove using the triangle inequality theorem that the sides form a triangle.
  -- This part is left as an exercise to the reader.
  sorry
}

end _l537_53730


namespace prime_factors_and_divisors_6440_l537_53743

theorem prime_factors_and_divisors_6440 :
  ∃ (a b c d : ℕ), 6440 = 2^a * 5^b * 7^c * 23^d ∧ a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧
  (a + 1) * (b + 1) * (c + 1) * (d + 1) = 32 :=
by 
  sorry

end prime_factors_and_divisors_6440_l537_53743


namespace line_passes_through_fixed_point_l537_53765

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) (h : a + b = 1) (h1 : 2 * a * x - b * y = 1) : x = 1/2 ∧ y = -1 :=
by 
  sorry

end line_passes_through_fixed_point_l537_53765


namespace train_length_l537_53793

noncomputable def speed_kmph := 80
noncomputable def time_seconds := 5

 noncomputable def speed_mps := (speed_kmph * 1000) / 3600

 noncomputable def length_train : ℝ := speed_mps * time_seconds

theorem train_length : length_train = 111.1 := by
  sorry

end train_length_l537_53793


namespace initial_markup_percentage_l537_53734

-- Conditions:
-- 1. Initial price of the coat is $76.
-- 2. Increasing the price by $4 results in a 100% markup.
-- 3. A 100% markup implies the selling price is double the wholesale price.

theorem initial_markup_percentage (W : ℝ) (h1 : W + (76 - W) = 76)
  (h2 : 2 * W = 76 + 4) : (36 / 40) * 100 = 90 :=
by
  -- Using the conditions directly from the problem, we need to prove the theorem statement.
  sorry

end initial_markup_percentage_l537_53734


namespace max_neg_p_l537_53781

theorem max_neg_p (p : ℤ) (h1 : p < 0) (h2 : ∃ k : ℤ, 2001 + p = k^2) : p ≤ -65 :=
by
  sorry

end max_neg_p_l537_53781


namespace morse_code_sequences_l537_53749

theorem morse_code_sequences : 
  let number_of_sequences := 
        (2 ^ 1) + (2 ^ 2) + (2 ^ 3) + (2 ^ 4) + (2 ^ 5)
  number_of_sequences = 62 :=
by
  sorry

end morse_code_sequences_l537_53749


namespace helen_choc_chip_yesterday_l537_53772

variable (total_cookies morning_cookies : ℕ)

theorem helen_choc_chip_yesterday :
  total_cookies = 1081 →
  morning_cookies = 554 →
  total_cookies - morning_cookies = 527 := by
  sorry

end helen_choc_chip_yesterday_l537_53772


namespace eight_n_is_even_l537_53717

theorem eight_n_is_even (n : ℕ) (h : n = 7) : 8 * n = 56 :=
by {
  sorry
}

end eight_n_is_even_l537_53717


namespace fourth_divisor_of_9600_l537_53758

theorem fourth_divisor_of_9600 (x : ℕ) (h1 : ∀ (d : ℕ), d = 15 ∨ d = 25 ∨ d = 40 → 9600 % d = 0) 
  (h2 : 9600 / Nat.lcm (Nat.lcm 15 25) 40 = x) : x = 16 := by
  sorry

end fourth_divisor_of_9600_l537_53758


namespace seating_arrangements_l537_53753

theorem seating_arrangements (n : ℕ) (h_n : n = 6) (A B : Fin n) (h : A ≠ B) : 
  ∃ k : ℕ, k = 240 := 
by 
  sorry

end seating_arrangements_l537_53753


namespace number_of_acceptable_outfits_l537_53789

-- Definitions based on conditions
def total_shirts := 5
def total_pants := 4
def restricted_shirts := 2
def restricted_pants := 1

-- Defining the problem statement
theorem number_of_acceptable_outfits : 
  (total_shirts * total_pants - restricted_shirts * restricted_pants + restricted_shirts * (total_pants - restricted_pants)) = 18 :=
by sorry

end number_of_acceptable_outfits_l537_53789


namespace base_r_correct_l537_53715

theorem base_r_correct (r : ℕ) :
  (5 * r ^ 2 + 6 * r) + (4 * r ^ 2 + 2 * r) = r ^ 3 + r ^ 2 → r = 8 := 
by 
  sorry

end base_r_correct_l537_53715


namespace amount_of_solution_added_l537_53706

variable (x : ℝ)

-- Condition: The solution contains 90% alcohol
def solution_alcohol_amount (x : ℝ) : ℝ := 0.9 * x

-- Condition: Total volume of the new mixture after adding 16 liters of water
def total_volume (x : ℝ) : ℝ := x + 16

-- Condition: The percentage of alcohol in the new mixture is 54%
def new_mixture_alcohol_amount (x : ℝ) : ℝ := 0.54 * (total_volume x)

-- The proof goal: the amount of solution added is 24 liters
theorem amount_of_solution_added : new_mixture_alcohol_amount x = solution_alcohol_amount x → x = 24 :=
by
  sorry

end amount_of_solution_added_l537_53706


namespace solution_set_l537_53783

def inequality_solution (x : ℝ) : Prop :=
  4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9

theorem solution_set :
  { x : ℝ | inequality_solution x } = { x : ℝ | (63 / 26 : ℝ) < x ∧ x ≤ (28 / 11 : ℝ) } :=
by
  sorry

end solution_set_l537_53783


namespace final_sum_after_50_passes_l537_53700

theorem final_sum_after_50_passes
  (particip: ℕ) 
  (num_passes: particip = 50) 
  (init_disp: ℕ → ℤ) 
  (initial_condition : init_disp 0 = 1 ∧ init_disp 1 = 0 ∧ init_disp 2 = -1)
  (operations: Π (i : ℕ), 
    (init_disp 0 = 1 →
    init_disp 1 = 0 →
    (i % 2 = 0 → init_disp 2 = -1) →
    (i % 2 = 1 → init_disp 2 = 1))
  )
  : init_disp 0 + init_disp 1 + init_disp 2 = 0 :=
by
  sorry

end final_sum_after_50_passes_l537_53700


namespace three_tenths_of_number_l537_53759

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 15) : (3/10) * x = 54 :=
by
  sorry

end three_tenths_of_number_l537_53759


namespace cone_base_circumference_l537_53705

theorem cone_base_circumference (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 240) :
  (2 / 3) * (2 * Real.pi * r) = 8 * Real.pi :=
by
  have circle_circumference : ℝ := 2 * Real.pi * r
  sorry

end cone_base_circumference_l537_53705


namespace no_arithmetic_sequence_without_square_gt1_l537_53748

theorem no_arithmetic_sequence_without_square_gt1 (a d : ℕ) (h_d : d ≠ 0) :
  ¬(∀ n : ℕ, ∃ k : ℕ, k > 0 ∧ k ∈ {a + n * d | n : ℕ} ∧ ∀ m : ℕ, m > 1 → m * m ∣ k → false) := sorry

end no_arithmetic_sequence_without_square_gt1_l537_53748


namespace bracelet_ratio_l537_53727

-- Definition of the conditions
def initial_bingley_bracelets : ℕ := 5
def kelly_bracelets_given : ℕ := 16 / 4
def total_bracelets_after_receiving := initial_bingley_bracelets + kelly_bracelets_given
def bingley_remaining_bracelets : ℕ := 6
def bingley_bracelets_given := total_bracelets_after_receiving - bingley_remaining_bracelets

-- Lean 4 Statement
theorem bracelet_ratio : bingley_bracelets_given * 3 = total_bracelets_after_receiving := by
  sorry

end bracelet_ratio_l537_53727


namespace hall_area_l537_53751

theorem hall_area 
  (L W : ℝ)
  (h1 : W = 1/2 * L)
  (h2 : L - W = 10) : 
  L * W = 200 := 
sorry

end hall_area_l537_53751


namespace investment_rate_l537_53767

theorem investment_rate (total_investment : ℝ) (invest1 : ℝ) (rate1 : ℝ) (invest2 : ℝ) (rate2 : ℝ) (desired_income : ℝ) (remaining_investment : ℝ) (remaining_rate : ℝ) : 
( total_investment = 12000 ∧ invest1 = 5000 ∧ rate1 = 0.06 ∧ invest2 = 4000 ∧ rate2 = 0.035 ∧ desired_income = 700 ∧ remaining_investment = 3000 ) → remaining_rate = 0.0867 :=
by
  sorry

end investment_rate_l537_53767


namespace Q_ratio_eq_one_l537_53798

noncomputable def g (x : ℂ) : ℂ := x^2007 - 2 * x^2006 + 2

theorem Q_ratio_eq_one (Q : ℂ → ℂ) (s : ℕ → ℂ) (h_root : ∀ j : ℕ, j < 2007 → g (s j) = 0) 
  (h_Q : ∀ j : ℕ, j < 2007 → Q (s j + (1 / s j)) = 0) :
  Q 1 / Q (-1) = 1 := by
  sorry

end Q_ratio_eq_one_l537_53798


namespace simplify_expression_l537_53771

theorem simplify_expression (x : ℝ) (hx : x ≠ 4):
  (x^2 - 4 * x) / (x^2 - 8 * x + 16) = x / (x - 4) :=
by sorry

end simplify_expression_l537_53771


namespace impossible_digit_filling_l537_53778

theorem impossible_digit_filling (T : Fin 5 → Fin 8 → Fin 10) :
  (∀ d : Fin 10, (∃! r₁ r₂ r₃ r₄ : Fin 5, T r₁ = d ∧ T r₂ = d ∧ T r₃ = d ∧ T r₄ = d) ∧
                 (∃! c₁ c₂ c₃ c₄ : Fin 8, T c₁ = d ∧ T c₂ = d ∧ T c₃ = d ∧ T c₄ = d)) → False :=
by
  sorry

end impossible_digit_filling_l537_53778


namespace equation_of_trajectory_l537_53736

open Real

variable (P : ℝ → ℝ → Prop)
variable (C : ℝ → ℝ → Prop)
variable (L : ℝ → ℝ → Prop)

-- Definition of the fixed circle C
def fixed_circle (x y : ℝ) : Prop :=
  (x + 2) ^ 2 + y ^ 2 = 1

-- Definition of the fixed line L
def fixed_line (x y : ℝ) : Prop := 
  x = 1

noncomputable def moving_circle (P : ℝ → ℝ → Prop) (r : ℝ) : Prop :=
  ∃ x y : ℝ, P x y ∧ r > 0 ∧
  (∀ a b : ℝ, fixed_circle a b → ((x - a) ^ 2 + (y - b) ^ 2) = (r + 1) ^ 2) ∧
  (∀ a b : ℝ, fixed_line a b → (abs (x - a)) = (r + 1))

theorem equation_of_trajectory
  (P : ℝ → ℝ → Prop)
  (r : ℝ)
  (h : moving_circle P r) :
  ∀ x y : ℝ, P x y → y ^ 2 = -8 * x :=
by
  sorry

end equation_of_trajectory_l537_53736


namespace sequence_formula_l537_53738

theorem sequence_formula :
  ∀ (a : ℕ → ℕ),
  (a 1 = 11) ∧
  (a 2 = 102) ∧
  (a 3 = 1003) ∧
  (a 4 = 10004) →
  ∀ n, a n = 10^n + n := by
  sorry

end sequence_formula_l537_53738


namespace smallest_n_mod_l537_53747

theorem smallest_n_mod :
  ∃ n : ℕ, (23 * n ≡ 5678 [MOD 11]) ∧ (∀ m : ℕ, (23 * m ≡ 5678 [MOD 11]) → (0 < n) ∧ (n ≤ m)) :=
  by
  sorry

end smallest_n_mod_l537_53747


namespace product_xyz_l537_53710

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l537_53710


namespace increase_in_average_l537_53703

variable (A : ℝ)
variable (new_avg : ℝ := 44)
variable (score_12th_inning : ℝ := 55)
variable (total_runs_after_11 : ℝ := 11 * A)

theorem increase_in_average :
  ((total_runs_after_11 + score_12th_inning) / 12 - A = 1) :=
by
  sorry

end increase_in_average_l537_53703


namespace jake_has_more_balloons_l537_53709

-- Defining the given conditions as parameters
def initial_balloons_allan : ℕ := 2
def initial_balloons_jake : ℕ := 6
def additional_balloons_allan : ℕ := 3

-- Calculate total balloons each person has
def total_balloons_allan : ℕ := initial_balloons_allan + additional_balloons_allan
def total_balloons_jake : ℕ := initial_balloons_jake

-- Formalize the statement to be proved
theorem jake_has_more_balloons :
  total_balloons_jake - total_balloons_allan = 1 :=
by
  -- Proof will be added here
  sorry

end jake_has_more_balloons_l537_53709


namespace fraction_of_As_l537_53769

-- Define the conditions
def fraction_B (T : ℕ) := 1/4 * T
def fraction_C (T : ℕ) := 1/2 * T
def remaining_D : ℕ := 20
def total_students_approx : ℕ := 400

-- State the theorem
theorem fraction_of_As 
  (F : ℚ) : 
  ∀ T : ℕ, 
  T = F * T + fraction_B T + fraction_C T + remaining_D → 
  T = total_students_approx → 
  F = 1/5 :=
by
  intros
  sorry

end fraction_of_As_l537_53769


namespace trigonometric_identity_l537_53728

theorem trigonometric_identity : 
  (Real.cos (15 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) - Real.cos (75 * Real.pi / 180) * Real.sin (105 * Real.pi / 180))
  = -1 / 2 :=
by
  sorry

end trigonometric_identity_l537_53728


namespace ab_product_power_l537_53756

theorem ab_product_power (a b : ℤ) (n : ℕ) (h1 : (a * b)^n = 128 * 8) : n = 10 := by
  sorry

end ab_product_power_l537_53756


namespace sequence_term_l537_53790

theorem sequence_term (x : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, 2 / x n = 1 / x (n - 1) + 1 / x (n + 1))
  (h₁ : x 2 = 2 / 3)
  (h₂ : x 4 = 2 / 5) :
  x 10 = 2 / 11 := 
sorry

end sequence_term_l537_53790


namespace crossing_time_indeterminate_l537_53780

-- Define the lengths of the two trains.
def train_A_length : Nat := 120
def train_B_length : Nat := 150

-- Define the crossing time of the two trains when moving in the same direction.
def crossing_time_together : Nat := 135

-- Define a theorem to state that without additional information, the crossing time for a 150-meter train cannot be determined.
theorem crossing_time_indeterminate 
    (V120 V150 : Nat) 
    (H : V150 - V120 = 2) : 
    ∃ t, t > 0 -> t < 150 / V150 -> False :=
by 
    -- The proof is not provided.
    sorry

end crossing_time_indeterminate_l537_53780


namespace polynomial_product_l537_53795

theorem polynomial_product (a b c : ℝ) :
  a * (b - c) ^ 3 + b * (c - a) ^ 3 + c * (a - b) ^ 3 = (a - b) * (b - c) * (c - a) * (a + b + c) :=
by sorry

end polynomial_product_l537_53795


namespace larger_sphere_radius_l537_53754

theorem larger_sphere_radius (r : ℝ) (π : ℝ) (h : r^3 = 2) :
  r = 2^(1/3) :=
by
  sorry

end larger_sphere_radius_l537_53754


namespace length_of_common_chord_l537_53797

theorem length_of_common_chord (x y : ℝ) :
  (x + 1)^2 + (y - 3)^2 = 9 ∧ x^2 + y^2 - 4 * x + 2 * y - 11 = 0 → 
  ∃ l : ℝ, l = 24 / 5 :=
by
  sorry

end length_of_common_chord_l537_53797


namespace gcd_expression_l537_53720

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end gcd_expression_l537_53720


namespace divisible_iff_condition_l537_53731

theorem divisible_iff_condition (a b : ℤ) : 
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) :=
  sorry

end divisible_iff_condition_l537_53731


namespace max_gcd_lcm_condition_l537_53718

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l537_53718


namespace compute_difference_l537_53722

noncomputable def f (n : ℝ) : ℝ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem compute_difference (r : ℝ) : f r - f (r - 1) = r * (r + 1) * (r + 2) := by
  sorry

end compute_difference_l537_53722


namespace tax_percentage_excess_l537_53799

/--
In Country X, each citizen is taxed an amount equal to 15 percent of the first $40,000 of income,
plus a certain percentage of all income in excess of $40,000. A citizen of Country X is taxed a total of $8,000
and her income is $50,000.

Prove that the percentage of the tax on the income in excess of $40,000 is 20%.
-/
theorem tax_percentage_excess (total_tax : ℝ) (first_income : ℝ) (additional_income : ℝ) (income : ℝ) (tax_first_part : ℝ) (tax_rate_first_part : ℝ) (tax_rate_excess : ℝ) (tax_excess : ℝ) :
  total_tax = 8000 →
  first_income = 40000 →
  additional_income = 10000 →
  income = first_income + additional_income →
  tax_rate_first_part = 0.15 →
  tax_first_part = tax_rate_first_part * first_income →
  tax_excess = total_tax - tax_first_part →
  tax_rate_excess * additional_income = tax_excess →
  tax_rate_excess = 0.20 :=
by
  intro h_total_tax h_first_income h_additional_income h_income h_tax_rate_first_part h_tax_first_part h_tax_excess h_tax_equation
  sorry

end tax_percentage_excess_l537_53799


namespace gcd_7384_12873_l537_53757

theorem gcd_7384_12873 : Int.gcd 7384 12873 = 1 :=
by
  sorry

end gcd_7384_12873_l537_53757


namespace find_x_l537_53732

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : x * floor x = 50) : x = 7.142857 :=
by
  sorry

end find_x_l537_53732


namespace integer_solution_l537_53775

theorem integer_solution (x : ℤ) (h : x^2 < 3 * x) : x = 1 ∨ x = 2 :=
sorry

end integer_solution_l537_53775


namespace minimum_value_l537_53725

open Real

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log (2^x) + log (8^y) = log 2) :
  ∃ (v : ℝ), v = 4 ∧ ∀ u, (∀ x y, x > 0 ∧ y > 0 → log (2^x) + log (8^y) = log 2 → x + 3*y = 1 → u = 4) := sorry

end minimum_value_l537_53725


namespace coordinates_equality_l537_53708

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end coordinates_equality_l537_53708


namespace sin_480_deg_l537_53779

theorem sin_480_deg : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_deg_l537_53779


namespace jeff_current_cats_l537_53786

def initial_cats : ℕ := 20
def monday_found_kittens : ℕ := 2 + 3
def monday_stray_cats : ℕ := 4
def tuesday_injured_cats : ℕ := 1
def tuesday_health_issues_cats : ℕ := 2
def tuesday_family_cats : ℕ := 3
def wednesday_adopted_cats : ℕ := 4 * 2
def wednesday_pregnant_cats : ℕ := 2
def thursday_adopted_cats : ℕ := 3
def thursday_donated_cats : ℕ := 3
def friday_adopted_cats : ℕ := 2
def friday_found_cats : ℕ := 3

theorem jeff_current_cats : 
  initial_cats 
  + monday_found_kittens + monday_stray_cats 
  + (tuesday_injured_cats + tuesday_health_issues_cats + tuesday_family_cats)
  + (wednesday_pregnant_cats - wednesday_adopted_cats)
  + (thursday_donated_cats - thursday_adopted_cats)
  + (friday_found_cats - friday_adopted_cats) 
  = 30 := by
  sorry

end jeff_current_cats_l537_53786


namespace value_before_decrease_l537_53704

theorem value_before_decrease
  (current_value decrease : ℤ)
  (current_value_equals : current_value = 1460)
  (decrease_equals : decrease = 12) :
  current_value + decrease = 1472 :=
by
  -- We assume the proof to follow here.
  sorry

end value_before_decrease_l537_53704


namespace max_area_rect_l537_53733

noncomputable def maximize_area (l w : ℕ) : ℕ :=
  l * w

theorem max_area_rect (l w: ℕ) (hl_even : l % 2 = 0) (h_perim : 2*l + 2*w = 40) :
  maximize_area l w = 100 :=
by
  sorry 

end max_area_rect_l537_53733


namespace probability_of_two_accurate_forecasts_l537_53785

noncomputable def event_A : Type := {forecast : ℕ | forecast = 1}

def prob_A : ℝ := 0.9
def prob_A' : ℝ := 1 - prob_A

-- Define that there are 3 independent trials
def num_forecasts : ℕ := 3

-- Given
def probability_two_accurate (x : ℕ) : ℝ :=
if x = 2 then 3 * (prob_A^2 * prob_A') else 0

-- Statement to be proved
theorem probability_of_two_accurate_forecasts : probability_two_accurate 2 = 0.243 := by
  -- Proof will go here
  sorry

end probability_of_two_accurate_forecasts_l537_53785


namespace complex_multiplication_l537_53721

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  ((a + b * i) * (c + d * i)) = (-6 + 33 * i) :=
by
  have a := 3
  have b := -4
  have c := -6
  have d := 3
  sorry

end complex_multiplication_l537_53721


namespace edward_money_l537_53776

theorem edward_money (initial_amount spent1 spent2 : ℕ) (h_initial : initial_amount = 34) (h_spent1 : spent1 = 9) (h_spent2 : spent2 = 8) :
  initial_amount - (spent1 + spent2) = 17 :=
by
  sorry

end edward_money_l537_53776


namespace B_contribution_to_capital_l537_53762

theorem B_contribution_to_capital (A_capital : ℝ) (A_months : ℝ) (B_months : ℝ) (profit_ratio_A : ℝ) (profit_ratio_B : ℝ) (B_contribution : ℝ) :
  A_capital = 4500 →
  A_months = 12 →
  B_months = 5 →
  profit_ratio_A = 2 →
  profit_ratio_B = 3 →
  B_contribution = (4500 * 12 * 3) / (5 * 2) → 
  B_contribution = 16200 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end B_contribution_to_capital_l537_53762


namespace always_meaningful_fraction_l537_53763

theorem always_meaningful_fraction {x : ℝ} : (∀ x, ∃ option : ℕ, 
  (option = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) ∨ 
  (option = 2 ∧ True) ∨ 
  (option = 3 ∧ x ≠ 0) ∨ 
  (option = 4 ∧ x ≠ 1)) → option = 2 :=
sorry

end always_meaningful_fraction_l537_53763


namespace gasoline_reduction_l537_53744

theorem gasoline_reduction
  (P Q : ℝ)
  (h1 : 0 < P)
  (h2 : 0 < Q)
  (price_increase_percent : ℝ := 0.25)
  (spending_increase_percent : ℝ := 0.05)
  (new_price : ℝ := P * (1 + price_increase_percent))
  (new_total_cost : ℝ := (P * Q) * (1 + spending_increase_percent)) :
  100 - (100 * (new_total_cost / new_price) / Q) = 16 :=
by
  sorry

end gasoline_reduction_l537_53744
