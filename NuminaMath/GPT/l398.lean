import Mathlib

namespace prayer_ratio_l398_398293

theorem prayer_ratio (P_s : ℕ) :
    (6 * 20 + P_s = 6 * 10 + 2 * P_s + 20) → (P_s / 20 = 2) :=
begin
  intro h,
  sorry
end

end prayer_ratio_l398_398293


namespace michael_initial_fish_l398_398692

-- Define the conditions
def benGave : ℝ := 18.0
def totalFish : ℝ := 67

-- Define the statement to be proved
theorem michael_initial_fish :
  (totalFish - benGave) = 49 := by
  sorry

end michael_initial_fish_l398_398692


namespace largest_angle_of_triangle_ABC_l398_398247

theorem largest_angle_of_triangle_ABC (a b c : ℝ)
  (h₁ : a + b + 2 * c = a^2) 
  (h₂ : a + b - 2 * c = -1) : 
  ∃ C : ℝ, C = 120 :=
sorry

end largest_angle_of_triangle_ABC_l398_398247


namespace sum_of_solutions_eq_neg2000_l398_398075

theorem sum_of_solutions_eq_neg2000 : 
  let a := 1
  let b := 2000
  let c := -2001
  let roots_sum := -b / a
  solutions_sum := roots_sum = -2000 :=
by
  let a := 1
  let b := 2000
  let c := -2001
  let roots_sum := -b / a
  have h : roots_sum = -2000,
  {
    sorry
  }
  show solutions_sum = -2000, { exact h }

end sum_of_solutions_eq_neg2000_l398_398075


namespace max_months_with_five_sundays_l398_398413

/-- Prove that the maximum number of months in a year that can contain five Sundays is 5. -/
theorem max_months_with_five_sundays {days_in_week months_in_year weeks_in_common_year weeks_in_leap_year extra_days_in_common_year extra_days_in_leap_year :
  ℕ → ℕ → ℕ → ℕ → ℕ → ℕ}
  (d_week : days_in_week = 7)
  (m_year : months_in_year = 12)
  (w_common_year : weeks_in_common_year = 52)
  (w_leap_year : weeks_in_leap_year = 52)
  (e_common_year : extra_days_in_common_year = 1)
  (e_leap_year : extra_days_in_leap_year = 2)
  (days_in_year_common : ∀ (days_in_common_year), days_in_common_year = weeks_in_common_year * days_in_week + extra_days_in_common_year)
  (days_in_year_leap : ∀ (days_in_leap_year), days_in_leap_year = weeks_in_leap_year * days_in_week + extra_days_in_leap_year)
  (sundays_per_month : ℕ → ℕ → ℕ)
  (min_sundays_per_month : ∀ (days_in_month), 28 ≤ days_in_month → sundays_per_month days_in_week days_in_month ≥ 4)
  (total_sundays_common_year : ∀ (total_sundays), total_sundays = 52)
  (total_sundays_leap_year : ∀ (total_sundays_leap), total_sundays_leap = 53) :
  ∃ (max_sundays_common_year max_sundays_leap_year : ℕ),
  max_sundays_common_year = 4 ∧ max_sundays_leap_year = 5 :=
sorry

end max_months_with_five_sundays_l398_398413


namespace length_of_AP_l398_398115

theorem length_of_AP (P A B : ℝ) (hP_on_AB : P = A + B ∧ A ≥ 0 ∧ B ≥ 0) (hAP_squared : P^2 = A * B) (h_AB_eq_2 : A + B = 2) : 
  (AP = real.sqrt 5 - 1) := 
by
  sorry

end length_of_AP_l398_398115


namespace booking_rooms_needed_l398_398013

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398013


namespace sequence_general_term_l398_398098

-- Given a sequence {a_n} whose sum of the first n terms S_n = 2a_n - 1,
-- prove that the general formula for the n-th term a_n is 2^(n-1).

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
    (h₁ : ∀ n : ℕ, S n = 2 * a n - 1)
    (h₂ : S 1 = 1) : ∀ n : ℕ, a (n + 1) = 2 ^ n :=
by
  sorry

end sequence_general_term_l398_398098


namespace sum_abcd_eq_16_l398_398119

variable (a b c d : ℝ)

def cond1 : Prop := a^2 + b^2 + c^2 + d^2 = 250
def cond2 : Prop := a * b + b * c + c * a + a * d + b * d + c * d = 3

theorem sum_abcd_eq_16 (h1 : cond1 a b c d) (h2 : cond2 a b c d) : a + b + c + d = 16 := 
by 
  sorry

end sum_abcd_eq_16_l398_398119


namespace problem_statement_l398_398651

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {triangleABC : Prop}

def is_equilateral_triangle (A B C : ℝ) : Prop :=
  A = B ∧ B = C

def triangle_sides_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

noncomputable def cos_condition (A B C : ℝ) : Prop :=
  cos (A - B) * cos (B - C) * cos (C - A) = 1

theorem problem_statement (A B C a b c : ℝ) (habc : triangle_sides_condition a b c A B C) (h : cos_condition A B C) :
  is_equilateral_triangle A B C :=
sorry

end problem_statement_l398_398651


namespace perimeter_triangle_VWX_l398_398467

noncomputable def triangle_perimeter {A B C : Type*} [euclidean_space V] [inner_product_space ℝ V] 
  (P Q R : V) (height : ℝ) (side_length : ℝ) (midpoint_PR midpoint_RQ midpoint_QT : V) 
  (h_height : height = 20) (h_side_length : side_length = 10) 
  (h_PQ : dist P Q = side_length) (h_PR : dist P R = side_length) 
  (h_RQ : dist R Q = side_length) 
  (h_mid_PR: midpoint_PR = (P + R) / 2) 
  (h_mid_RQ: midpoint_RQ = (R + Q) / 2)
  (h_mid_QT: midpoint_QT = (Q + T) / 2) : ℝ :=
  dist midpoint_PR midpoint_RQ + dist midpoint_RQ midpoint_QT + dist midpoint_QT midpoint_PR 

theorem perimeter_triangle_VWX (P Q R T : ℝ) : 
  triangle_perimeter P Q R T 20 10 ((P + R) / 2) ((R + Q) / 2) ((Q + T) / 2) 20 10 10 10 5 5 + 10 * real.sqrt 5 :=
sorry

end perimeter_triangle_VWX_l398_398467


namespace ratio_of_sheep_to_horses_l398_398343

theorem ratio_of_sheep_to_horses (H : ℕ) (hH : 230 * H = 12880) (n_sheep : ℕ) (h_sheep : n_sheep = 56) :
  (n_sheep / H) = 1 := by
  sorry

end ratio_of_sheep_to_horses_l398_398343


namespace log_base2_075_l398_398076

theorem log_base2_075 : Real.logBase 2 0.75 = -0.4153 := 
by
  sorry

end log_base2_075_l398_398076


namespace ripe_oranges_count_l398_398797

/-- They harvest 52 sacks of unripe oranges per day. -/
def unripe_oranges_per_day : ℕ := 52

/-- After 26 days of harvest, they will have 2080 sacks of oranges. -/
def total_oranges_after_26_days : ℕ := 2080

/-- Define the number of sacks of ripe oranges harvested per day. -/
def ripe_oranges_per_day (R : ℕ) : Prop :=
  26 * (R + unripe_oranges_per_day) = total_oranges_after_26_days

/-- Prove that they harvest 28 sacks of ripe oranges per day. -/
theorem ripe_oranges_count : ripe_oranges_per_day 28 :=
by {
  -- This is where the proof would go
  sorry
}

end ripe_oranges_count_l398_398797


namespace probability_of_connection_l398_398067

theorem probability_of_connection (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  let num_pairs := 5 * 8 in
  let prob_no_connection := (1 - p) ^ num_pairs in
  1 - prob_no_connection = 1 - (1 - p) ^ 40 := 
by
  let num_pairs := 5 * 8
  have h_num_pairs : num_pairs = 40 := by norm_num
  rw h_num_pairs
  let prob_no_connection := (1 - p) ^ 40
  sorry

end probability_of_connection_l398_398067


namespace sum_of_angles_l398_398325

def arcs_in_circle := 16
def degrees_per_circle := 360

def angle_per_arc : ℝ := degrees_per_circle / arcs_in_circle
def central_angle_x (spans : ℕ) : ℝ := spans * angle_per_arc
def central_angle_y (spans : ℕ) : ℝ := spans * angle_per_arc
def inscribed_angle (central_angle : ℝ) : ℝ := central_angle / 2

def span_x := 4
def span_y := 5

theorem sum_of_angles : inscribed_angle (central_angle_x span_x) + inscribed_angle (central_angle_y span_y) = 101.25 :=
by
  -- Proof omitted
  sorry

end sum_of_angles_l398_398325


namespace area_of_triangle_ABC_l398_398977

theorem area_of_triangle_ABC : 
  let A := (1, 1)
  let B := (4, 1)
  let C := (1, 5)
  let area := 6
  (1:ℝ) * abs (1 * (1 - 5) + 4 * (5 - 1) + 1 * (1 - 1)) / 2 = area := 
by
  sorry

end area_of_triangle_ABC_l398_398977


namespace children_more_than_adults_l398_398078

-- Definitions based on given conditions
def price_per_child : ℚ := 4.50
def price_per_adult : ℚ := 6.75
def total_receipts : ℚ := 405
def number_of_children : ℕ := 48

-- Goal: Prove the number of children is 20 more than the number of adults.
theorem children_more_than_adults :
  ∃ (A : ℕ), (number_of_children - A) = 20 ∧ (price_per_child * number_of_children) + (price_per_adult * A) = total_receipts := by
  sorry

end children_more_than_adults_l398_398078


namespace problem_1_problem_2_l398_398588

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

-- (1) Prove that, given f(x) and a = -3, the solution set for f(x) ≥ 3 is (-∞, 1] ∪ [4, +∞)
theorem problem_1 (x : ℝ) : 
  (∃ (a : ℝ), a = -3 ∧ f x a ≥ 3) ↔ (x ≤ 1 ∨ x ≥ 4) :=
sorry

-- (2) Prove that for f(x) to be ≥ 3 for all x, the range of a is a ≥ 1 or a ≤ -5
theorem problem_2 : 
  (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≥ 1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l398_398588


namespace event_B_more_likely_l398_398703

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l398_398703


namespace range_of_a_l398_398108

variable {a : ℝ}
variable {x : ℝ}

def p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (hpq : p ∨ q) (hnpq : ¬(p ∧ q)) :
  a < 0 ∨ (1 / 4 < a ∧ a < 4) :=
sorry

end range_of_a_l398_398108


namespace johnsons_days_l398_398260

theorem johnsons_days (J : ℝ) (h1 : J > 0) (h2 : 1 / J + 1 / 40 = 1 / 8) : J = 10 :=
begin
  sorry
end

end johnsons_days_l398_398260


namespace possible_values_of_d_l398_398432

-- Let n be an odd positive integer with n > 1
variables {n : ℕ} (hn1 : n > 1) (hn2 : n % 2 = 1)

-- Let a_1, a_2, ..., a_n be positive integers such that gcd(a_1, a_2, ..., a_n) = 1
variables {a : ℕ → ℕ} (hpos : ∀ i, 1 ≤ a i)
(hgcd : gcd_list (list.fin_range n).map a = 1)

-- Define d = gcd(a_1^n + a_1 * a_2 * ... * a_n, a_2^n + a_1 * a_2 * ... * a_n, ..., a_n^n + a_1 * a_2 * ... * a_n)
noncomputable def prod_a : ℕ := list.prod (list.fin_range n).map a
noncomputable def sequence : ℕ → ℕ := λ i, (a i) ^ n + prod_a a
noncomputable def d : ℕ := gcd_list (list.fin_range n).map sequence

-- Show that the possible values of d are d = 1, d = 2
theorem possible_values_of_d : d a = 1 ∨ d a = 2 :=
sorry

end possible_values_of_d_l398_398432


namespace stable_table_quadruples_l398_398560

theorem stable_table_quadruples (n : ℕ) : 
  (∑ m in Finset.range (n + 1), (2 * m + 1) * (2 * m + 1) + 
  ∑ m in Finset.range (n + 1), (2 * (n + 1 - m) + 1) * (2 * (n + 1 - m) + 1)) = 
  (1/2 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3) := 
by
  sorry

end stable_table_quadruples_l398_398560


namespace no_positive_integer_solutions_l398_398542

theorem no_positive_integer_solutions (m : ℕ) (h_pos : m > 0) :
  ¬ ∃ x : ℚ, m * x^2 + 40 * x + m = 0 :=
by {
  -- the proof goes here
  sorry
}

end no_positive_integer_solutions_l398_398542


namespace calculate_f_neg3_l398_398960

noncomputable def f : ℝ → ℝ
| x => 
  if x ≥ 0 then x^2 + 3 * x 
  else f (x + 2)

theorem calculate_f_neg3 : f (-3) = 4 := by
  sorry

end calculate_f_neg3_l398_398960


namespace inequality_1_inequality_2_inequality_strict_1_inequality_strict_2_l398_398679

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def p : ℚ := sorry
noncomputable def q : ℚ := sorry

axiom positive_A : A > 0
axiom positive_B : B > 0
axiom rational_p : p ∈ ℚ
axiom rational_q : q ∈ ℚ

axiom relation_pq : (1 : ℚ) / p + (1 : ℚ) / q = 1

theorem inequality_1 (h_pgt1 : (p : ℝ) > 1) : A^(1/(p:ℝ)) * B^(1/(q:ℝ)) ≤ A / p + B / q := sorry
theorem inequality_2 (h_plt1 : (p : ℝ) < 1) : A^(1/(p:ℝ)) * B^(1/(q:ℝ)) ≥ A / p + B / q := sorry
theorem inequality_strict_1 (A_ne_B : A ≠ B) (h_pgt1 : (p : ℝ) > 1) : A^(1/(p:ℝ)) * B^(1/(q:ℝ)) < A / p + B / q := sorry
theorem inequality_strict_2 (A_ne_B : A ≠ B) (h_plt1 : (p : ℝ) < 1) : A^(1/(p:ℝ)) * B^(1/(q:ℝ)) > A / p + B / q := sorry

end inequality_1_inequality_2_inequality_strict_1_inequality_strict_2_l398_398679


namespace higher_sale_price_l398_398457

theorem higher_sale_price (initial_price : ℝ) (d1 d2 d3 d4 m : ℝ) :
  initial_price = 15000 →
  d1 = 0.25 →
  d2 = 0.15 →
  d3 = 0.05 →
  d4 = 0.40 →
  m = 0.30 →
  initial_price * (1 - d1) * (1 - d2) * (1 - d3) * (1 + m) >
  initial_price * (1 - d4) * (1 + m) :=
begin
  intros,
  sorry
end

end higher_sale_price_l398_398457


namespace probability_sum_15_l398_398387

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398387


namespace minimum_rooms_needed_fans_l398_398022

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398022


namespace minimum_rooms_needed_l398_398004

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398004


namespace sasha_pluck_leaves_l398_398478

theorem sasha_pluck_leaves (num_apple_trees num_poplar_trees num_unphotographed start_pluck : ℕ)
  (H1 : num_apple_trees = 17)
  (H2 : num_poplar_trees = 18)
  (H3 : num_unphotographed = 13)
  (H4 : start_pluck = 8) :
  let total_trees := num_apple_trees + num_poplar_trees in
  let end_photographed := 10 in
  let pluck_start_tree := start_pluck in
  22 = total_trees - end_photographed - (10 - pluck_start_tree) + 1 :=
sorry

end sasha_pluck_leaves_l398_398478


namespace triangle_area_l398_398266

theorem triangle_area (h : ℝ) (r : ℝ) (A : ℝ) (h_nonneg : 0 ≤ h) (r_nonneg : 0 ≤ r) :
  A = 1/2 * r * h^2 :=
begin
  sorry -- The proof follows from substituting base and height in the area formula
end

end triangle_area_l398_398266


namespace find_angle_E_l398_398724

variables (E F G H : Type) [quadrilateral E F G H] (measure : ∀ x : quadrilateral E F G H, ℝ)

-- Given conditions
def parallelogram_property (EFGH : Type) [quadrilateral E F G H] : Prop :=
  ∀ (α β : quadrilateral E F G H), quadrilateral.angle α = quadrilateral.angle β →
  (∃ (θ : ℝ), measure α = θ ∧ measure β = θ) ∧
  (∀ (α β : quadrilateral E F G H), (α + β = 180))

-- Given the specific measure of angle EGH
def measure_EGH (α : quadrilateral E F G H) (h_parallelogram : parallelogram_property E F G H) : Prop :=
  measure α = 80

-- Proof problem statement
theorem find_angle_E (h_parallelogram : parallelogram_property E F G H) (h_measure_EGH : measure_EGH (angle E G H) h_parallelogram) :
  measure (angle E F G H) = 80 :=
sorry

end find_angle_E_l398_398724


namespace nextBusyDay_l398_398727

theorem nextBusyDay (today_day : ℕ)
  (h_today : today_day = 13) -- Representing September 13 as day 13
  (sep_days : ℕ := 30) -- September has 30 days
  :
  ∃ next_day : ℕ, next_day = 13 ∧ next_day = (today_day + Nat.lcm 2 (Nat.lcm 3 5)) % sep_days → (today_day + sep_days - today_day) + 1 = 13 + 12 := 
begin
  sorry
end

end nextBusyDay_l398_398727


namespace circle_second_x_intercept_l398_398324

noncomputable def circle_intersects_x_axis_at_second_point : Prop :=
  ∃ (x : ℝ), (x ≠ 0) ∧ ((x - 4)^2 + (0 - 3)^2 = 25) ∧ x = 8

theorem circle_second_x_intercept :
  circle_intersects_x_axis_at_second_point :=
begin
  sorry
end

end circle_second_x_intercept_l398_398324


namespace tan_alpha_is_neg_5_over_12_l398_398944

variables (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_is_neg_5_over_12 : Real.tan α = -5/12 :=
by
  sorry

end tan_alpha_is_neg_5_over_12_l398_398944


namespace num_license_plates_l398_398995

-- Let's state the number of letters in the alphabet, vowels, consonants, and digits.
def num_letters : ℕ := 26
def num_vowels : ℕ := 5  -- A, E, I, O, U and Y is not a vowel
def num_consonants : ℕ := 21  -- Remaining letters including Y
def num_digits : ℕ := 10  -- 0 through 9

-- Prove the number of five-character license plates
theorem num_license_plates : 
  (num_consonants * num_consonants * num_vowels * num_vowels * num_digits) = 110250 :=
  by 
  sorry

end num_license_plates_l398_398995


namespace theo_cookies_l398_398791

theorem theo_cookies (cookies_per_time times_per_day total_cookies total_months : ℕ) (h1 : cookies_per_time = 13) (h2 : times_per_day = 3) (h3 : total_cookies = 2340) (h4 : total_months = 3) : (total_cookies / total_months) / (cookies_per_time * times_per_day) = 20 := 
by
  -- Placeholder for the proof
  sorry

end theo_cookies_l398_398791


namespace pow_mod_eq_one_l398_398818

theorem pow_mod_eq_one :
  ∀ (x m n : ℕ), (x ≡ 1 [MOD m]) → (x ^ n ≡ 1 [MOD m]) :=
by
  intros x m n h
  sorry

example : 101 ^ 37 % 100 = 1 :=
by
  have h1 : 101 ≡ 1 [MOD 100] := by norm_num
  have h2 := pow_mod_eq_one 101 100 37 h1
  exact nat.modeq.modeq_iff_dvd.mp h2

end pow_mod_eq_one_l398_398818


namespace selection_methods_eq_total_students_l398_398634

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end selection_methods_eq_total_students_l398_398634


namespace combined_speed_in_still_water_l398_398809

theorem combined_speed_in_still_water 
  (U1 D1 U2 D2 : ℝ) 
  (hU1 : U1 = 30) 
  (hD1 : D1 = 60) 
  (hU2 : U2 = 40) 
  (hD2 : D2 = 80) 
  : (U1 + D1) / 2 + (U2 + D2) / 2 = 105 := 
by 
  sorry

end combined_speed_in_still_water_l398_398809


namespace original_cube_side_length_l398_398443

noncomputable def small_cube_side_length : ℝ := 10
noncomputable def num_small_cubes : ℝ := 1000
noncomputable def original_cube_volume : ℝ := num_small_cubes * (small_cube_side_length ^ 3)

theorem original_cube_side_length :
  ∃ L : ℝ, L^3 = original_cube_volume ∧ L = 100 :=
by
  use 100
  split
  try { exact_pow_eq_pow_root_mul _ }
  sorry

end original_cube_side_length_l398_398443


namespace Fireflies_win_by_5_points_l398_398424

theorem Fireflies_win_by_5_points:
  ∀ (initial_hornets_score initial_fireflies_score additional_hornets_points additional_fireflies_points: ℕ),
    initial_hornets_score = 86 →
    initial_fireflies_score = 74 →
    additional_hornets_points = 2 * 2 →
    additional_fireflies_points = 7 * 3 →
    (initial_hornets_score + additional_hornets_points < initial_fireflies_score + additional_fireflies_points) →
    (initial_fireflies_score + additional_fireflies_points) - (initial_hornets_score + additional_hornets_points) = 5 :=
by
  intros initial_hornets_score initial_fireflies_score additional_hornets_points additional_fireflies_points
  assume h1 h2 h3 h4 h5
  sorry

end Fireflies_win_by_5_points_l398_398424


namespace isosceles_right_triangle_area_l398_398775

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l398_398775


namespace real_and_imag_parts_of_z_l398_398572

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem real_and_imag_parts_of_z :
  ∀ (i : ℂ), i * i = -1 → 
  ∀ (z : ℂ), z = i * (-1 + 2 * i) → real_part z = -2 ∧ imag_part z = -1 :=
by 
  intros i hi z hz
  sorry

end real_and_imag_parts_of_z_l398_398572


namespace mod_5_remainder_of_sum_skipping_sixth_number_l398_398887

-- Define the main problem statement
theorem mod_5_remainder_of_sum_skipping_sixth_number :
  let seq := List.range' (-3) 127
  let skipped_seq := seq.filter (fun x => ¬ (x - (-3)) % 6 == 5)
  let sum := skipped_seq.foldl (· + ·) 0
  sum % 5 = 0 :=
by
  sorry

end mod_5_remainder_of_sum_skipping_sixth_number_l398_398887


namespace simon_number_of_legos_l398_398731

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l398_398731


namespace arithmetic_progression_x_value_l398_398756

theorem arithmetic_progression_x_value (x: ℝ) (h1: 3*x - 1 - (2*x - 3) = 4*x + 1 - (3*x - 1)) : x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l398_398756


namespace passengers_initial_count_l398_398357

-- Let's define the initial number of passengers
variable (P : ℕ)

-- Given conditions:
def final_passengers (initial additional left : ℕ) : ℕ := initial + additional - left

-- The theorem statement to prove P = 28 given the conditions
theorem passengers_initial_count
  (final_count : ℕ)
  (h1 : final_count = 26)
  (h2 : final_passengers P 7 9 = final_count) 
  : P = 28 :=
by
  sorry

end passengers_initial_count_l398_398357


namespace ratio_of_areas_l398_398882

def regular_hexagon := Type

def vertices_star_are_vertices_hexagon
    (hex : regular_hexagon) : Prop :=
∀ (v : Fin 6), v ∈ (hex : regular_hexagon)

def hexagon_divided_18_small_triangles
    (hex : regular_hexagon) : Prop :=
∃ (triangles : Fin 18 → Type), true

def shaded_region_12_triangles
    (hex : regular_hexagon) : Prop :=
∃ (shaded_triangles : Fin 12 → Type), true

def blank_region_6_triangles
    (hex : regular_hexagon) : Prop :=
∃ (blank_triangles : Fin 6 → Type), true

theorem ratio_of_areas 
    (hex : regular_hexagon)
    (h1 : vertices_star_are_vertices_hexagon hex)
    (h2 : hexagon_divided_18_small_triangles hex)
    (h3 : shaded_region_12_triangles hex)
    (h4 : blank_region_6_triangles hex) :
    12 / 6 = 3 :=
by
  sorry

end ratio_of_areas_l398_398882


namespace student_grades_l398_398873

theorem student_grades (grades : List ℕ) (h1 : ∀ g ∈ grades, g ∈ {2, 3, 4, 5})
  (h2 : grades.length = 13) (h3 : (grades.sum / 13 : ℚ).den = 1) :
  ∃ g ∈ {2, 3, 4, 5}, grades.count g ≤ 2 :=
by
  sorry

end student_grades_l398_398873


namespace unique_two_scoop_sundaes_l398_398488

-- Define the problem conditions
def eight_flavors : Finset String := 
  {"Vanilla", "Chocolate", "Strawberry", "Mint", "Coffee", "Pistachio", "Lemon", "Blueberry"}

def vanilla_and_chocolate_together (S : Finset (Finset String)) : Prop :=
  ∀ s ∈ S, ("Vanilla" ∈ s ∧ "Chocolate" ∈ s) ∨ ("Vanilla" ∉ s ∧ "Chocolate" ∉ s)

-- Define the unique two-scoop calculation
theorem unique_two_scoop_sundaes : 
  ∃ S : Finset (Finset String), vanilla_and_chocolate_together S ∧ S.card = 7 :=
begin
  sorry
end

end unique_two_scoop_sundaes_l398_398488


namespace convert_500_to_base2_l398_398898

theorem convert_500_to_base2 :
  let n_base10 : ℕ := 500
  let n_base8 : ℕ := 7 * 64 + 6 * 8 + 4
  let n_base2 : ℕ := 1 * 256 + 1 * 128 + 1 * 64 + 1 * 32 + 1 * 16 + 0 * 8 + 1 * 4 + 0 * 2 + 0
  n_base10 = 500 ∧ n_base8 = 500 ∧ n_base2 = n_base8 :=
by
  sorry

end convert_500_to_base2_l398_398898


namespace problem1_problem2_l398_398929

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def M (a b c : ℝ) : ℝ := 
  max (f a b c (-2)) (max (f a b c 2) (max (f a b c 1) (f a b c (-1))))

def m (a b c : ℝ) : ℝ := 
  min (f a b c (-2)) (min (f a b c 2) (min (f a b c 1) (f a b c (-1))))

theorem problem1 (a b c : ℝ) (f0 : f a b c 0 = 2) (A1 A2 : A = {1, 2}) : 
  M a b c = 10 ∧ m a b c = 1 :=
sorry

theorem problem2 (a b : ℝ) (ha : a ≥ 1) (A2 : A = {2}) : 
  ∃ a, g a = 16 * a - 1 / (4 * a) ∧ gmin a = 63 / 4 :=
sorry

end problem1_problem2_l398_398929


namespace roots_of_quadratic_eq_l398_398151

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398151


namespace sum_a4_a5_a6_l398_398117

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (h1 : is_arithmetic_sequence a)
          (h2 : a 1 + a 2 + a 3 = 6)
          (h3 : a 7 + a 8 + a 9 = 24)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 15 :=
by
  sorry

end sum_a4_a5_a6_l398_398117


namespace total_strawberry_weight_l398_398690

def MarcosStrawberries : ℕ := 3
def DadsStrawberries : ℕ := 17

theorem total_strawberry_weight : MarcosStrawberries + DadsStrawberries = 20 := by
  sorry

end total_strawberry_weight_l398_398690


namespace find_cos_2alpha_find_sin_alpha_plus_pi_over_6_l398_398574

-- Define the conditions
variables (α : Real) (sin_alpha : Real)
  (h1 : Real.pi ≤ α ∧ α < 2 * Real.pi) -- α in the second quadrant
  (h2 : sin α = (Real.sqrt 15) / 4)

-- Define the statement for the first part
theorem find_cos_2alpha :
  ∃ β : Real, cos (2 * α) = -7 / 8 :=
by sorry
  
-- Define the statement for the second part
theorem find_sin_alpha_plus_pi_over_6 :
  ∃ β : Real, sin (α + Real.pi / 6) = (3 * Real.sqrt 5 - 1) / 8 :=
by sorry

end find_cos_2alpha_find_sin_alpha_plus_pi_over_6_l398_398574


namespace part1_part2_l398_398966

open Real

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - 1| - |x - a|

theorem part1 (a : ℝ) (h : a = 0) :
  {x : ℝ | f x a < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a < 1 → |(1 - 2 * a)^2 / 6| > 3 / 2) 
  : a < -1 :=
by
  sorry

end part1_part2_l398_398966


namespace remainder_is_three_l398_398699

def dividend : ℕ := 15
def divisor : ℕ := 3
def quotient : ℕ := 4

theorem remainder_is_three : dividend = (divisor * quotient) + Nat.mod dividend divisor := by
  sorry

end remainder_is_three_l398_398699


namespace maximum_F_value_l398_398280

open Real

noncomputable def F (a b c x : ℝ) := abs ((a * x^2 + b * x + c) * (c * x^2 + b * x + a))

theorem maximum_F_value (a b c : ℝ) (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1)
    (hfx : abs (a * x^2 + b * x + c) ≤ 1) :
    ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ F a b c x = 2 := 
  sorry

end maximum_F_value_l398_398280


namespace cyclic_quadrilateral_tangent_points_l398_398752

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

theorem cyclic_quadrilateral_tangent_points
  (P Q R S : A)
  (h1 : ∃ M, inscribed_circle_touches AC M ∧ M ∈ inscribed_circle ABC ∧ M ∈ inscribed_circle ACD)
  (h2 : inscribed_circle ABD P)
  (h3 : inscribed_circle BCD Q)
  : P = Q ∧ cyclic P Q R S :=
  sorry

-- Definitions used in the theorem
def inscribed_circle_touches (AC : Type) (M : Type) : Prop := sorry
def inscribed_circle (triangle : Type) (tangent_point : Type) : Prop := sorry
def cyclic (P Q R S : Type) : Prop := sorry

end cyclic_quadrilateral_tangent_points_l398_398752


namespace minimum_rooms_needed_l398_398007

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398007


namespace value_set_l398_398677

open Real Set

noncomputable def possible_values (a b c : ℝ) : Set ℝ :=
  {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ x = c / a + c / b}

theorem value_set (c : ℝ) (hc : c > 0) : possible_values a b c = Ici (2 * c) := by
  sorry

end value_set_l398_398677


namespace find_a_l398_398647

-- Define the parametric equations of lines l1 and l2
def l1_param (s : ℝ) : ℝ × ℝ := (2 * s + 1, s)
def l2_param (t : ℝ) (a : ℝ) : ℝ × ℝ := (a * t, 2 * t - 1)

-- Define the slopes of the general forms of the lines
def slope_l1 : ℝ := 1 / 2
def slope_l2 (a : ℝ) : ℝ := 2 / a

-- Theorem to prove that the lines are parallel implies a = 4
theorem find_a (a : ℝ) (h : slope_l1 = slope_l2 a) : a = 4 := by
  sorry

end find_a_l398_398647


namespace sasha_leaves_count_l398_398480

theorem sasha_leaves_count 
    (apple_trees : ℕ) 
    (poplar_trees : ℕ) 
    (masha_stop : ℕ) 
    (sasha_start : ℕ) 
    : apple_trees = 17 → 
      poplar_trees = 18 → 
      masha_stop = 10 → 
      sasha_start = 8 → 
      ∃ (total_leaves_plucked : ℕ), total_leaves_plucked = 22 :=
begin
  intros,
  have total_trees : ℕ := apple_trees + poplar_trees,
  have total_trees_after_masha : ℕ := total_trees - masha_stop,
  have total_trees_after_sasha : ℕ := total_trees - (sasha_start - 1),
  have total_leaves_plucked : ℕ := total_trees_after_sasha - (total_trees_after_masha - 13),
  use total_leaves_plucked,
  exact calc
    total_leaves_plucked = 35 - 7 - (35 - 10 - 13) : by simp [total_trees, total_trees_after_masha, total_trees_after_sasha]
                    ...  = total_leaves_plucked      : by sorry,
end

end sasha_leaves_count_l398_398480


namespace average_of_last_seven_results_l398_398318

variable (results : List ℕ) (result5 : ℕ)
variable (len11 := 11) (len5 := 5) (len7 := 7)
variable (avg11 := 42) (avg5 := 49) (result5 := 147)

def sum_results (l : List ℕ) : ℕ :=
  l.foldr (+) 0

theorem average_of_last_seven_results 
  (h₁ : sum_results results = avg11 * len11)
  (h₂ : sum_results (results.take len5) = avg5 * len5)
  (h₃ : result5 = (results.take len5).nth! 4)
  : (sum_results (results.drop (len11 - len7)) / len7 = 52) :=
sorry

end average_of_last_seven_results_l398_398318


namespace isosceles_right_triangle_area_l398_398773

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l398_398773


namespace find_z_l398_398111

-- Define the condition that \overline{z} is the conjugate of z
def conj (z : ℂ) : ℂ := complex.conj z

-- Given condition
def z_conjugate : ℂ := (1 : ℂ) * complex.I / (1 - complex.I)

-- The theorem to prove
theorem find_z (z : ℂ) (h : conj z = z_conjugate) : 
  z = - (1/2 : ℂ) - (1/2 : ℂ) * complex.I :=
begin
  sorry
end

end find_z_l398_398111


namespace find_radius_of_sphere_l398_398869

def radius_of_sphere (width : ℝ) (depth : ℝ) (r : ℝ) : Prop :=
  (width / 2) ^ 2 + (r - depth) ^ 2 = r ^ 2

theorem find_radius_of_sphere (r : ℝ) : radius_of_sphere 30 10 r → r = 16.25 :=
by
  intros h1
  -- sorry is a placeholder for the actual proof
  sorry

end find_radius_of_sphere_l398_398869


namespace systematic_sampling_sixth_group_l398_398448

theorem systematic_sampling_sixth_group
    (total_students : ℕ)
    (num_samples : ℕ)
    (student_number : ℕ → ℕ)
    (common_difference : ℕ) :
    total_students = 1000 →
    num_samples = 50 →
    (∀ n, student_number n = 1 + common_difference * (n - 1)) →
    common_difference = 20 →
    student_number 6 = 101 := 
by 
    intro h1 h2 h3 h4,
    sorry

end systematic_sampling_sixth_group_l398_398448


namespace angle_BAD_37_5_degrees_l398_398835
  
noncomputable def ∆ (A B C : Type) : triangle A B C := sorry

variables {A B C D : Point}
variables (h₁ : isosceles_triangle A B C)
variables (h₂ : ∠ACB = 30)
variables (h₃ : midpoint_segment B C D)
variables (h₄ : angle_bisector A B C A D)

theorem angle_BAD_37_5_degrees (h₁ : isosceles_triangle AC BC)
    (h₂ : ∠ACB = 30) (h₃ : midpoint_segment B C D) (h₄ : angle_bisector A B C A D) :
  ∠BAD = 37.5 := sorry

end angle_BAD_37_5_degrees_l398_398835


namespace event_B_more_likely_l398_398702

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l398_398702


namespace quadratic_roots_identity_l398_398180

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398180


namespace min_value_fraction_l398_398554

theorem min_value_fraction (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃ m, (∀ z, z = (1 / (x + 1) + 1 / y) → z ≥ m) ∧ m = (3 + 2 * Real.sqrt 2) / 2 :=
by
  sorry

end min_value_fraction_l398_398554


namespace squirrel_can_catch_nut_l398_398089

-- Definitions for the given conditions
def distance_gavrila_squirrel : Real := 3.75
def nut_velocity : Real := 5
def squirrel_jump_distance : Real := 1.8
def gravity_acceleration : Real := 10

-- Statement to be proved
theorem squirrel_can_catch_nut : ∃ t : Real, 
  let r_squared := (nut_velocity * t - distance_gavrila_squirrel)^2 + (gravity_acceleration * t^2 / 2)^2 in
  r_squared ≤ squirrel_jump_distance^2 :=
begin
  sorry
end

end squirrel_can_catch_nut_l398_398089


namespace inequality_proof_l398_398923

theorem inequality_proof (a b c : ℝ) (ha : a ≥ b) (hb : b ≥ c) (hc : c > 0) :
  b / a + c / b + a / c ≥ (1 / 3) * (a + b + c) * (1 / a + 1 / b + 1 / c) :=
by sorry

end inequality_proof_l398_398923


namespace max_MN_l398_398132

def f (x : ℝ) : ℝ := 2 * (Real.sin (π / 4 + x))^2
def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x)

theorem max_MN (a : ℝ) : |f a - g a| ≤ 3 := by
  sorry

end max_MN_l398_398132


namespace find_integers_l398_398414

theorem find_integers (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 := by
  sorry

end find_integers_l398_398414


namespace prob_sum_15_correct_l398_398378

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398378


namespace count_pos_even_multiples_of_12_less_than_5000_l398_398611

-- Define the conditions in terms appropriate for Lean
def isMultipleOf12 (n : ℕ) : Prop :=
  n % 12 = 0

def isEven (n : ℕ) : Prop :=
  n % 2 = 0

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the main statement
theorem count_pos_even_multiples_of_12_less_than_5000 :
  let count := (Finset.range 5000).filter (λ n, isEven n ∧ isMultipleOf12 n ∧ isPerfectSquare n)
  count.card = 5 :=
by
  let count := (Finset.range 5000).filter (λ n, isEven n ∧ isMultipleOf12 n ∧ isPerfectSquare n)
  admit

end count_pos_even_multiples_of_12_less_than_5000_l398_398611


namespace isosceles_triangle_area_l398_398767

theorem isosceles_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) 
  (h_eq : h = 6 * sqrt 2) 
  (h_leg : h = a * sqrt 2) 
  (area_eq : A = 1 / 2 * a^2) : 
  A = 18 :=
by
  sorry

end isosceles_triangle_area_l398_398767


namespace simplify_expression_l398_398310

variable (x : ℝ)

theorem simplify_expression (hx : x ∈ ℝ) : (x / (x + 1) - (3 * x) / (2 * (x + 1)) - 1 = (-3 * x - 2) / (2 * (x + 1))) := 
by 
  sorry

end simplify_expression_l398_398310


namespace quadratic_factors_l398_398344

theorem quadratic_factors {a b c : ℝ} (h : a = 1) (h_roots : (1:ℝ) + 2 = b ∧ (-1:ℝ) * 2 = c) :
  (x^2 - b * x + c) = (x - 1) * (x - 2) := by
  sorry

end quadratic_factors_l398_398344


namespace base_three_to_base_ten_l398_398749

theorem base_three_to_base_ten (n : ℕ) (h : n = 20121) : 
  let convert := 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 in
  convert = 178 :=
by
  have : convert = 162 + 0 + 9 + 6 + 1, by sorry
  show convert = 178, by sorry

end base_three_to_base_ten_l398_398749


namespace num_pos_int_solutions_l398_398142

theorem num_pos_int_solutions : {x : ℕ | 12 < -x + 17 ∧ x > 0}.card = 4 :=
by
  sorry

end num_pos_int_solutions_l398_398142


namespace angle_between_skew_lines_in_tetrahedron_l398_398225

theorem angle_between_skew_lines_in_tetrahedron
  (a : ℝ) 
  (h1 : a > 0) 
  (E F : Euclidean3.Point)
  (SA SB SC AB BC AC SE SF : Euclidean3.Vector)
  (h2 : SE = SA / 2) 
  (h3 : SF = BC / 2)
  (BE SF_vec : Euclidean3.Vector)
  (h4 : BE = SB + SA / 2) 
  (h5 : SF_vec = SB + BC / 2) 
  : ∃ θ : ℝ, θ = real.arccos (-2 / 3) :=
begin
  sorry
end

end angle_between_skew_lines_in_tetrahedron_l398_398225


namespace smallest_number_proof_l398_398799

noncomputable def smallest_number (a b c : ℕ) :=
  let sum := a + b + c
  let mean := sum / 3
  let sorted := (list.sort (≤) [a, b, c])
  (sorted.head!, sorted.nth! 1, sorted.nth! 2)

theorem smallest_number_proof (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30)
  (h_med : (list.sort (≤) [a, b, c]).nth! 1 = 28)
  (h_largest : (list.sort (≤) [a, b, c]).nth! 2 = 28 + 6) :
  (list.sort (≤) [a, b, c]).head! = 28 :=
by
  sorry

end smallest_number_proof_l398_398799


namespace train_speed_l398_398875

theorem train_speed (length_train length_platform time : ℕ) (h_train: length_train = 360) (h_platform: length_platform = 390) (h_time: time = 60) : 
  (length_train + length_platform) / time * 3.6 = 45 := 
  sorry

end train_speed_l398_398875


namespace unique_perpendicular_segment_exists_l398_398607

open EuclideanGeometry

-- Assume L1 and L2 are skew lines in Euclidean space
variable (L1 L2 : Line) (h_skew : ¬Intersect L1 L2 ∧ ¬Parallel L1 L2)

-- Theorem stating the existence and uniqueness of the perpendicular segment
theorem unique_perpendicular_segment_exists (h_skew : ¬ Intersect L1 L2 ∧ ¬ Parallel L1 L2) :
  ∃! (A : Point) (B : Point), (A ∈ L1 ∧ B ∈ L2) ∧ Perpendicular (segment A B) L1 ∧ Perpendicular (segment A B) L2 :=
sorry

end unique_perpendicular_segment_exists_l398_398607


namespace graph_of_equation_l398_398331

theorem graph_of_equation (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) := 
by sorry

end graph_of_equation_l398_398331


namespace squirrel_can_catch_nut_l398_398088

-- Define the initial distance between Gabriel and the squirrel.
def initial_distance : ℝ := 3.75

-- Define the speed of the nut.
def nut_speed : ℝ := 5.0

-- Define the jumping distance of the squirrel.
def squirrel_jump_distance : ℝ := 1.8

-- Define the acceleration due to gravity.
def gravity : ℝ := 10.0

-- Define the positions of the nut and the squirrel as functions of time.
def nut_position_x (t : ℝ) : ℝ := nut_speed * t
def squirrel_position_x : ℝ := initial_distance
def nut_position_y (t : ℝ) : ℝ := 0.5 * gravity * t^2

-- Define the squared distance between the nut and the squirrel.
def distance_squared (t : ℝ) : ℝ :=
  (nut_position_x t - squirrel_position_x)^2 + (nut_position_y t)^2

-- Prove that the minimum distance squared is less than or equal to the squirrel's jumping distance squared.
theorem squirrel_can_catch_nut : ∃ t : ℝ, distance_squared t ≤ squirrel_jump_distance^2 := by
  -- Sorry placeholder, as the proof is not required.
  sorry

end squirrel_can_catch_nut_l398_398088


namespace heaviest_vs_lightest_total_selling_price_l398_398352

-- Conditions
def standard_weight : ℕ := 25
def baskets : List (ℝ × ℕ) := [(-3, 1), (-1.5, 3), (-0.5, 2), (0, 1), (2, 1), (2.5, 2)]
def price_per_kg : ℝ := 3

-- Problem 1
theorem heaviest_vs_lightest : 
  let heaviest := 2.5
  let lightest := -3
  heaviest - lightest = 5.5 := by
    sorry

-- Problem 2
theorem total_selling_price :
  let total_deviation := (baskets.sumr (λ x, x.1 * x.2))
  let total_weight := (baskets.sumr (λ x, x.2) * standard_weight : ℝ) + total_deviation
  total_weight * price_per_kg = 745.5 := by
    sorry

end heaviest_vs_lightest_total_selling_price_l398_398352


namespace multiples_5_or_7_but_not_6_count_l398_398614

theorem multiples_5_or_7_but_not_6_count :
  let count_multiples (k : Nat) (N : Nat) :=
    Nat.floor (N / k) in
  let count_conditional (N : Nat) :=
    count_multiples 5 N + count_multiples 7 N - count_multiples 35 N -
    (count_multiples 6 N - count_multiples 30 N - count_multiples 42 N) in
  count_conditional 200 = 40 :=
by
  sorry

end multiples_5_or_7_but_not_6_count_l398_398614


namespace smallest_tic_tac_toe_winning_figure_has_7_cells_l398_398535

/-- Let a figure on graph paper be defined such that two players play tic-tac-toe on it.
The starting player will always be able to place three consecutive X's first. We must prove
that the smallest such figure consists of 7 cells. --/
theorem smallest_tic_tac_toe_winning_figure_has_7_cells :
  ∃ (figure : set (ℤ × ℤ)), (∀ (strategy : (ℤ × ℤ) → Prop), ∃ (starting_player_x_consecutive : (ℤ × ℤ) → Prop),
  (starting_player_x_consecutive strategy) ∧ (starting_player_x_consecutive → ∃ (a : ℕ), a = 7)) :=
sorry

end smallest_tic_tac_toe_winning_figure_has_7_cells_l398_398535


namespace sum_of_coefficients_l398_398074

-- Define the polynomial P(x)
def P (x : ℤ) : ℤ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

-- State the theorem we intend to prove
theorem sum_of_coefficients : P 1 = 2019 :=
by
  -- Proof omitted
  sorry

end sum_of_coefficients_l398_398074


namespace Adam_total_cost_l398_398475

theorem Adam_total_cost :
  let laptop1_cost := 500
  let laptop2_base_cost := 3 * laptop1_cost
  let discount := 0.15 * laptop2_base_cost
  let laptop2_cost := laptop2_base_cost - discount
  let external_hard_drive := 80
  let mouse := 20
  let software1 := 120
  let software2 := 2 * 120
  let insurance1 := 0.10 * laptop1_cost
  let insurance2 := 0.10 * laptop2_cost
  let total_cost1 := laptop1_cost + external_hard_drive + mouse + software1 + insurance1
  let total_cost2 := laptop2_cost + external_hard_drive + mouse + software2 + insurance2
  total_cost1 + total_cost2 = 2512.5 :=
by
  sorry

end Adam_total_cost_l398_398475


namespace even_quadruples_solution_l398_398271

def even_quadruples : Nat :=
  (Nat.choose 51 3)

theorem even_quadruples_solution :
  ∑ (x₁ x₂ x₃ x₄ : ℕ), (x₁ + x₂ + x₃ + x₄ = 104 ∧ x₁ % 2 = 0 ∧ x₂ % 2 = 0 ∧ x₃ % 2 = 0 ∧ x₄ % 2 = 0) / 100 = 208.25 :=
by
  sorry

end even_quadruples_solution_l398_398271


namespace min_rooms_needed_l398_398001

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l398_398001


namespace prove_a_eq_neg_2_l398_398947

theorem prove_a_eq_neg_2 (a : ℤ) (x : ℤ) 
  (h1 : a^2 * x - 20 = 0) 
  (h2 : x > 1 ∧ (∀k, 2 ≤ k ∧ k < x → ¬(k ∣ x))
  (h3 : |a * x - 7| > a^2)
  : a = -2 := 
sorry

end prove_a_eq_neg_2_l398_398947


namespace xy_sum_l398_398206

namespace ProofExample

variable (x y : ℚ)

def condition1 : Prop := (1 / x) + (1 / y) = 4
def condition2 : Prop := (1 / x) - (1 / y) = -6

theorem xy_sum : condition1 x y → condition2 x y → (x + y = -4 / 5) := by
  intros
  sorry

end ProofExample

end xy_sum_l398_398206


namespace angle_OMB_right_l398_398101

noncomputable def triangle (A B C : Type) : Prop :=
∃ (A B C : Point), triangle A B C

noncomputable def circle (O : Point) (A C : Point) : Prop :=
∃ (radius : Real), dist O A = radius ∧ dist O C = radius

noncomputable def intersects (circle : Circle) (segment : Line) (K N : Point) : Prop :=
intersects circle segment ∧ (∃ (distinct : K ≠ N))

noncomputable def circumcircle (triangle : Triangle) (circle : Circle) : Prop :=
circumcircle_of_triangle triangle circle

noncomputable def intersection (circle1 circle2 : Circle) (B M  : Point) : Prop :=
intersects circle1 B ∧ intersects circle2 B ∧ (∃ (M : Point), intersects circle1 M ∧ intersects circle2 M)

noncomputable def right_angle (O M B : Point) : Prop :=
angle O M B = 90

theorem angle_OMB_right {A B C O K N M : Point}
(triangle_ABC : triangle A B C)
(circle_O_AC : circle O A C)
(intersects_circle_segments_AB_BC : intersects (circle O A C) (line A B) K ∧ intersects (circle O A C) (line B C) N ∧ K ≠ N)
(circumcircles_intersect_at_B_and_M : circumcircle (triangle A B C) (circle O A C) ∧ circumcircle (triangle K B N) (circle O A C) ∧ intersection (circumcircle (triangle A B C)) (circumcircle (triangle K B N)) B M)
: right_angle O M B := sorry

end angle_OMB_right_l398_398101


namespace gcd_420_144_l398_398817

def prime_factorization_420 : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (7, 1)]
def prime_factorization_144 : List (ℕ × ℕ) := [(2, 4), (3, 2)]

theorem gcd_420_144 : Nat.gcd 420 144 = 12 :=
by
  have h₁ : Nat.factors 420 = [2, 2, 3, 5, 7] := by sorry
  have h₂ : Nat.factors 144 = [2, 2, 2, 2, 3, 3] := by sorry
  exact Nat.gcd_eq_gcd_summed h₁ h₂
  sorry

end gcd_420_144_l398_398817


namespace quadratic_real_roots_k_le_one_fourth_l398_398135

theorem quadratic_real_roots_k_le_one_fourth (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4 * k - 2) * x + k^2 = 0) ↔ k ≤ 1/4 :=
sorry

end quadratic_real_roots_k_le_one_fourth_l398_398135


namespace area_triangle_ABC_l398_398907

def area_of_30_60_90_triangle (AC : ℝ) (h : AC = 6) : ℝ :=
  by
    have triangle_ratio : 1:√3:2 := sorry -- Property of a 30-60-90 triangle
    let AB := AC * √3
    let Area := (1 / 2) * AB * AC
    exact Area

theorem area_triangle_ABC : area_of_30_60_90_triangle 6 rfl = 18 * √3 :=
  by 
    sorry

end area_triangle_ABC_l398_398907


namespace ship_speed_in_still_water_l398_398866

theorem ship_speed_in_still_water 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (x : ℝ) 
  (h1 : distance = 36)
  (h2 : time = 6)
  (h3 : current_speed = 3) 
  (h4 : (18 / (x + 3) + 18 / (x - 3) = 6)) 
  : x = 3 + 3 * Real.sqrt 2 :=
sorry

end ship_speed_in_still_water_l398_398866


namespace find_m_that_makes_f_odd_function_l398_398134

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (m : ℝ) : ℝ → ℝ := λ x : ℝ, (m^2 - 5*m + 7)*x^(m - 2)

theorem find_m_that_makes_f_odd_function :
  ∃ m : ℝ, is_odd_function (f m) ∧ m = 3 :=
by
  sorry

end find_m_that_makes_f_odd_function_l398_398134


namespace cost_of_27_pounds_l398_398715

def rate_per_pound : ℝ := 1
def weight_pounds : ℝ := 27

theorem cost_of_27_pounds :
  weight_pounds * rate_per_pound = 27 := 
by 
  -- sorry placeholder indicates that the proof is not provided
  sorry

end cost_of_27_pounds_l398_398715


namespace pendulum_faster_17_seconds_winter_l398_398796

noncomputable def pendulum_period (l g : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (l / g)

noncomputable def pendulum_seconds_faster_in_winter (T : ℝ) (l : ℝ) (g : ℝ) (shorten : ℝ) (hours : ℝ) : ℝ :=
  let summer_period := T
  let winter_length := l - shorten
  let winter_period := pendulum_period winter_length g
  let summer_cycles := (hours * 60 * 60) / summer_period
  let winter_cycles := (hours * 60 * 60) / winter_period
  winter_cycles - summer_cycles

theorem pendulum_faster_17_seconds_winter :
  let T := 1
  let l := 980 * (1 / (4 * Real.pi ^ 2))
  let g := 980
  let shorten := 0.01 / 100
  let hours := 24
  pendulum_seconds_faster_in_winter T l g shorten hours = 17 :=
by
  sorry

end pendulum_faster_17_seconds_winter_l398_398796


namespace correct_propositions_l398_398645

def proposition1 (l1 l2 l3 : Line) : Prop :=
parallel l1 l2 → parallel l1 l3 → parallel l2 l3

def proposition2 (l1 l2 l3 : Line) : Prop :=
perpendicular l1 l2 → perpendicular l1 l3 → parallel l2 l3

def proposition3 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
(l1 ∈ p1) → (l2 ∈ p2) → (p1 ≠ p2) → skew l1 l2

def proposition4 (q : Quadrilateral) : Prop :=
(∃ s1 s2 : Side, opposite_sides_equal q s1 s2) → parallelogram q

theorem correct_propositions :
  (proposition1 true true true) ∧
  (¬ proposition2 true true true) ∧
  (¬ proposition3 true true true) ∧
  (¬ proposition4 true) →
  1 = 1 :=
by
  sorry

end correct_propositions_l398_398645


namespace sin_double_angle_inequality_l398_398652

theorem sin_double_angle_inequality (A B C : ℝ) (h : A + B + C = π) :
  sin (2 * A) + sin (2 * B) + sin (2 * C) ≤ sin A + sin B + sin C :=
by
  sorry

end sin_double_angle_inequality_l398_398652


namespace solution_l398_398716

variable (hasMoney : ℕ)
variable (fullPrice : ℕ)
variable (saleSecond : ℕ)
variable (saleThird : ℕ)

def balloonSets := (fullPrice + saleSecond + saleThird)
def maxBalloonsOrvinCanBuy (money : ℕ) (setPrice : ℕ) (balloonsPerSet : ℕ) : ℕ :=
  (money / setPrice) * balloonsPerSet

theorem solution : 
  hasMoney = 120 ∧ 
  fullPrice = 4 ∧ 
  saleSecond = 2 ∧ 
  saleThird = 1 →
  maxBalloonsOrvinCanBuy hasMoney balloonSets 3 = 51 :=
by
  intros h
  cases h
  sorry

end solution_l398_398716


namespace min_rooms_needed_l398_398027

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398027


namespace probability_two_cards_sum_to_15_l398_398405

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398405


namespace andrea_height_proof_l398_398435

theorem andrea_height_proof (tree_height shadow_length : ℝ) (andrea_shadow : ℝ)
  (tree_height_casts_shadow : tree_height = 70)
  (shadow_length_of_tree : shadow_length = 14)
  (andrea_shadow_length : andrea_shadow = 3.5) :
  let ratio := tree_height / shadow_length in
  5 * andrea_shadow = 17.5 :=
by
  have ratio := tree_height / shadow_length
  have : ratio = 5 := by
    rw [tree_height_casts_shadow, shadow_length_of_tree]
    norm_num
  sorry

end andrea_height_proof_l398_398435


namespace minimum_distance_parabola_line_l398_398780

theorem minimum_distance_parabola_line : 
  let parabola := λ x : ℝ, - x ^ 2
  let line := 4 * x + 3 * y - 8 = 0
  ∀ (x y : ℝ), 
  (parabola x = y ∧ line) → y = - x ^ 2 ∧ 4 * x + 3 * y - 8 = 0
  (minimum_distance parabola line) = (4 / 3) :=
sorry

end minimum_distance_parabola_line_l398_398780


namespace event_B_more_likely_than_event_A_l398_398708

-- Definitions based on given conditions
def total_possible_outcomes := 6^3
def favorable_outcomes_B := (Nat.choose 6 3) * (Nat.factorial 3)
def prob_B := favorable_outcomes_B / total_possible_outcomes
def prob_A := 1 - prob_B

-- The theorem to be proved:
theorem event_B_more_likely_than_event_A (total_possible_outcomes = 216) 
    (favorable_outcomes_B = 120) 
    (prob_B = 5 / 9) 
    (prob_A = 4 / 9) :
    prob_B > prob_A := 
by {
    sorry
}

end event_B_more_likely_than_event_A_l398_398708


namespace definite_integral_sin8_l398_398431

-- Define the definite integral problem and the expected result in Lean.
theorem definite_integral_sin8:
  ∫ x in (Real.pi / 2)..Real.pi, (2^8 * (Real.sin x)^8) = 32 * Real.pi :=
  sorry

end definite_integral_sin8_l398_398431


namespace cheapest_option_l398_398984

/-
  Problem: Prove that gathering berries in the forest to make jam is
  the cheapest option for Grandmother Vasya.
-/

def gathering_berries_cost (transportation_cost_per_kg sugar_cost_per_kg : ℕ) := (40 + sugar_cost_per_kg : ℕ)
def buying_berries_cost (berries_cost_per_kg sugar_cost_per_kg : ℕ) := (150 + sugar_cost_per_kg : ℕ)
def buying_ready_made_jam_cost (ready_made_jam_cost_per_kg : ℕ) := (220 * 1.5 : ℕ)

theorem cheapest_option (transportation_cost_per_kg sugar_cost_per_kg berries_cost_per_kg ready_made_jam_cost_per_kg : ℕ) : 
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_berries_cost berries_cost_per_kg sugar_cost_per_kg ∧
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_ready_made_jam_cost ready_made_jam_cost_per_kg := 
by
  sorry

end cheapest_option_l398_398984


namespace verify_statements_l398_398959

-- Define the conditions
def slopes_product_condition (M : ℝ × ℝ) : Prop :=
  let F1 := (-4, 0)
  let F2 := (4, 0)
  let slope1 := (M.2 - F1.2) / (M.1 - F1.1)
  let slope2 := (M.2 - F2.2) / (M.1 - F2.1)
  slope1 * slope2 = -9 / 16

-- Define the foci for curve C
def left_focus : (ℝ × ℝ) := (-4, 0)
def right_focus : (ℝ × ℝ) := (4, 0)

-- Define the correct statements to be verified
def statement3 : Prop :=
  ∃ P : ℝ × ℝ, slopes_product_condition P ∧
    let PF1 := (P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2
    let PF2 := (P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2
    PF1 > PF2 ∧ PF1 / PF2 = (23 / 9) ^ 2

def statement4 : Prop :=
  let A := (1, 1)
  ∃ P : ℝ × ℝ, slopes_product_condition P ∧
    let PA := (P.1 - A.1)^2 + (P.2 - A.2)^2
    let PF1 := (P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2
    abs (fst PA) + abs (fst PF1) = 8 + sqrt (9 - 2 * sqrt 7)

-- The theorem statement encapsulating the mathematical proof problem
theorem verify_statements (P : ℝ × ℝ) (M : ℝ × ℝ) :
  slopes_product_condition M →
  (statement3 ∧ statement4) := 
sorry

end verify_statements_l398_398959


namespace percentage_discount_l398_398892

theorem percentage_discount (original_price sale_price : ℝ) (h1 : original_price = 25) (h2 : sale_price = 18.75) : 
  100 * (original_price - sale_price) / original_price = 25 := 
by
  -- Begin Proof
  sorry

end percentage_discount_l398_398892


namespace john_total_expense_l398_398660

-- Define variables
variables (M D : ℝ)

-- Define the conditions
axiom cond1 : M = 20 * D
axiom cond2 : M = 24 * (D - 3)

-- State the theorem to prove
theorem john_total_expense : M = 360 :=
by
  -- Add the proof steps here
  sorry

end john_total_expense_l398_398660


namespace decreasing_interval_of_f_l398_398334

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * log x

theorem decreasing_interval_of_f :
  (∀ x ∈ (Set.Ioc 0 1 : Set ℝ), 2*x - 2/x < 0) :=
by
  sorry

end decreasing_interval_of_f_l398_398334


namespace nth_equation_pattern_l398_398289

theorem nth_equation_pattern (n : ℕ) (hn : 0 < n) : n^2 - n = n * (n - 1) := by
  sorry

end nth_equation_pattern_l398_398289


namespace fraction_sum_product_roots_of_quadratic_l398_398161

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398161


namespace range_is_80_l398_398315

def dataSet : List ℕ := [60, 100, 80, 40, 20]

def minValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl min x xs

def maxValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl max x xs

def range (l : List ℕ) : ℕ :=
  maxValue l - minValue l

theorem range_is_80 : range dataSet = 80 :=
by
  sorry

end range_is_80_l398_398315


namespace samantha_last_name_length_l398_398307

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end samantha_last_name_length_l398_398307


namespace circumcenter_property_l398_398102

variable {Point : Type} [metric_space Point]

def perpendicular_bisector (A B : Point) : set Point := sorry
def circumcenter (A B C : Point) : Point := sorry

theorem circumcenter_property (A B C : Point) : 
  let O := circumcenter A B C in
  dist O A = dist O B ∧ dist O B = dist O C :=
sorry

end circumcenter_property_l398_398102


namespace carY_average_speed_l398_398496

-- Definitions and conditions
def carX_speed : ℝ := 35
def carX_travel_time_before_carY : ℝ := 72 / 60
def carX_distance_after_carY_started : ℝ := 49

-- Derived conditions
def carX_total_distance := (carX_speed * carX_travel_time_before_carY) + carX_distance_after_carY_started
def carX_time_for_49_miles := carX_distance_after_carY_started / carX_speed

-- Proof Statement
theorem carY_average_speed : 
  carX_total_distance / carX_time_for_49_miles = 65 :=
by
  sorry

end carY_average_speed_l398_398496


namespace chessboard_max_squares_l398_398998

def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

theorem chessboard_max_squares (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) : max_squares 1000 1000 = 1998 := 
by
  -- This is the theorem statement representing the maximum number of squares chosen
  -- in a 1000 x 1000 chessboard without having exactly three of them with two in the same row
  -- and two in the same column.
  sorry

end chessboard_max_squares_l398_398998


namespace power_function_at_3_l398_398210

theorem power_function_at_3 (a k : ℝ) (h : ∀ x, f x = a * x ^ k) (h1 : f 2 = 8) : f 3 = 27 := by
  sorry

end power_function_at_3_l398_398210


namespace hyperbola_properties_l398_398053

theorem hyperbola_properties :
  ∃ (a b : ℝ), (a ≠ 0 ∧ b ≠ 0) ∧
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 3) ^ 2 = 0 → (x / a) ^ 2 - (y / b) ^ 2 = 1) ∧
  (∀ x : ℝ, y = (1/2) * x ∨ y = -(1/2) * x) ∧
  (2 * b = √35) := sorry

end hyperbola_properties_l398_398053


namespace trapezium_height_l398_398048

theorem trapezium_height :
  ∀ (a b h : ℝ), a = 20 ∧ b = 18 ∧ (1 / 2) * (a + b) * h = 285 → h = 15 :=
by
  intros a b h hconds
  cases hconds with h1 hrem
  cases hrem with h2 harea
  simp at harea
  sorry

end trapezium_height_l398_398048


namespace sum_of_products_of_two_at_a_time_l398_398789

theorem sum_of_products_of_two_at_a_time (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a + b + c = 21) : 
  a * b + b * c + a * c = 100 := 
  sorry

end sum_of_products_of_two_at_a_time_l398_398789


namespace probability_of_odd_sum_rows_columns_l398_398782

open BigOperators

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_odd_sums : ℚ :=
  let even_arrangements := factorial 4
  let odd_positions := factorial 12
  let total_arrangements := factorial 16
  (even_arrangements * odd_positions : ℚ) / total_arrangements

theorem probability_of_odd_sum_rows_columns :
  probability_odd_sums = 1 / 1814400 :=
by
  sorry

end probability_of_odd_sum_rows_columns_l398_398782


namespace zoo_animal_count_l398_398471

def tiger_enclosures : ℕ := 4
def zebra_enclosures_per_tiger_enclosures : ℕ := 2
def zebra_enclosures : ℕ := tiger_enclosures * zebra_enclosures_per_tiger_enclosures
def giraffe_enclosures_per_zebra_enclosures : ℕ := 3
def giraffe_enclosures : ℕ := zebra_enclosures * giraffe_enclosures_per_zebra_enclosures
def tigers_per_enclosure : ℕ := 4
def zebras_per_enclosure : ℕ := 10
def giraffes_per_enclosure : ℕ := 2

def total_animals_in_zoo : ℕ := 
    (tiger_enclosures * tigers_per_enclosure) + 
    (zebra_enclosures * zebras_per_enclosure) + 
    (giraffe_enclosures * giraffes_per_enclosure)

theorem zoo_animal_count : total_animals_in_zoo = 144 := 
by
  -- proof would go here
  sorry

end zoo_animal_count_l398_398471


namespace quadratic_root_sum_and_product_l398_398193

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398193


namespace more_than_half_millet_on_thursday_l398_398695

-- Define the daily seed addition and consumption conditions
def initial_millet := 0.3
def initial_other_seeds := 0.7
def seeds_added_daily := 1.0
def millet_ratio := 0.3
def consumption_millet := 0.5
def consumption_other := 1.0

-- Define the amount of millet on each day after replenishment and bird consumption
def millet_after_replenish_and_consumption (day : ℕ) : ℝ :=
  if day = 1 then initial_millet
  else (millet_after_replenish_and_consumption (day - 1) * consumption_millet) + millet_ratio * seeds_added_daily

-- Define the condition for more than half of the seeds being millet
def more_than_half_millet (day : ℕ) : Prop :=
  let total_seeds := seeds_added_daily * day
  millet_after_replenish_and_consumption day > (total_seeds / 2)

-- Prove that more than half of the seeds are millet on the correct day, Thursday (day 4)
theorem more_than_half_millet_on_thursday : more_than_half_millet 4 :=
by {
  unfold more_than_half_millet, 
  unfold millet_after_replenish_and_consumption, 
  sorry -- Proof steps are skipped
}

end more_than_half_millet_on_thursday_l398_398695


namespace correct_propositions_l398_398229

theorem correct_propositions (a : ℕ → ℝ) (a_1 : ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ n, a n = a_1 + (n - 1) * d) →
  (∀ n, S n = n * a (n-1) + n * (n - 1) / 2 * d) →
  (let seq := λ n, (1/2) ^ (a n) in ∀ n, seq (n + 1) / seq n = (1/2) ^ d) ∧
  (a 10 = 3 ∧ S 7 = -7 → S 13 = 13) ∧
  (∀ n, S n = n * a n - n * (n - 1) / 2 * d) ∧
  (¬ ∃ n, d > 0 ∧ S n ≤ S (n + 1)) :=
sorry

end correct_propositions_l398_398229


namespace abs_f_sub_lt_abs_l398_398094

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

theorem abs_f_sub_lt_abs (a b : ℝ) (h : a ≠ b) : 
  |f a - f b| < |a - b| := 
by
  sorry

end abs_f_sub_lt_abs_l398_398094


namespace nine_digit_peak_numbers_count_l398_398209

theorem nine_digit_peak_numbers_count :
  ∃ (l : List ℕ), l.length = 9 ∧
    (∀ (i : Fin 5), l[5 + ↑i.succ] < l[5 + ↑i]) ∧
    (∀ (i : Fin 4), l[5 - ↑i.succ] > l[5 - ↑i]) ∧
    (∀ (i : Fin 9), 1 ≤ l[i] ∧ l[i] ≤ 9) ∧
    l.nodup ∧ 
    List.countP (λ l : List ℕ, 
      (∀ (i : Fin 5), l[(5 + i)].succ < l[5 + i]) ∧
      (∀ (i : Fin 4), l[(5 - i).succ] > l[5 - i])) = 11875 :=
sorry

end nine_digit_peak_numbers_count_l398_398209


namespace determine_omega_l398_398124

theorem determine_omega (ω : ℝ) (hω : ω > 0)
  (h1 : ∀ ⦃x1 x2 : ℝ⦄, -ω < x1 ∧ x1 < x2 ∧ x2 < ω → sin (ω * x1 + π / 4) < sin (ω * x2 + π / 4))
  (h2 : ∀ x : ℝ, sin (ω * x + π / 4) = sin (ω * (2 * ω - x) + π / 4)) :
  ω = sqrt (π) / 2 :=
sorry

end determine_omega_l398_398124


namespace line_relation_with_plane_l398_398979

variables {P : Type} [Infinite P] [MetricSpace P]

variables (a b : Line P) (α : Plane P)

-- Conditions
axiom intersecting_lines : ∃ p : P, p ∈ a ∧ p ∈ b
axiom line_parallel_plane : ∀ p : P, p ∈ a → p ∈ α

-- Theorem statement for the proof problem
theorem line_relation_with_plane : (∀ p : P, p ∈ b → p ∈ α) ∨ (∃ q : P, q ∈ α ∧ q ∈ b) :=
sorry

end line_relation_with_plane_l398_398979


namespace probability_two_white_balls_l398_398438

open Finset

theorem probability_two_white_balls (w b : ℕ) (hw : w = 7) (hb : b = 8) :
  (7 + 8 = 15) →
  (choose 7 2 / choose 15 2 = 1 / 5) :=
by
  intros
  sorry

end probability_two_white_balls_l398_398438


namespace sin_2490_eq_neg_half_cos_neg_52pi_over_3_eq_neg_half_l398_398836

theorem sin_2490_eq_neg_half : sin (2490 * real.pi / 180) = -1 / 2 :=
by
  sorry

theorem cos_neg_52pi_over_3_eq_neg_half : cos (-52 * real.pi / 3) = -1 / 2 :=
by
  sorry

end sin_2490_eq_neg_half_cos_neg_52pi_over_3_eq_neg_half_l398_398836


namespace probability_two_red_cards_l398_398458

theorem probability_two_red_cards :
  let deck_size := 60
  let suits := 5
  let cards_per_suit := 12
  let red_suits := 2
  let red_cards := red_suits * cards_per_suit
  let first_red_draw := red_cards / deck_size
  let second_red_draw := (red_cards - 1) / (deck_size - 1) in
  first_red_draw * second_red_draw = 92 / 590 :=
by
  -- Problem conditions
  have h1 : deck_size = 60 := rfl
  have h2 : suits = 5 := rfl
  have h3 : cards_per_suit = 12 := rfl
  have h4 : red_suits = 2 := rfl
  have h5 : red_cards = red_suits * cards_per_suit := rfl
  have h6 : red_cards = 24 := by simp [h4, h3]
  have h7 : first_red_draw = 24 / 60 := by simp [h1, h6]
  have h8 : second_red_draw = 23 / 59 := by simp [h1, h6]
  -- Calculation of final probability
  have h9 : first_red_draw * second_red_draw = (24 / 60) * (23 / 59) := by simp [h7, h8]
  have h10 : (24 * 23) / (60 * 59) = 552 / 3540 := by norm_num
  have h11 : 552 / 3540 = 92 / 590 := by norm_num
  -- Proving the final statement
  have result : first_red_draw * second_red_draw = 92 / 590 := by 
    rw [h9]
    rw [h10]
    rwa [h11]
  exact result

end probability_two_red_cards_l398_398458


namespace book_purchase_cost_l398_398804

/--
In a school book purchase scenario, where a total of 100 books (type A and B combined) are to be purchased, and the unit price of type A is $10 while type B is $8, prove that the cost of purchasing the type B books, given that x books of type A are purchased, equals 8 * (100 - x).
-/
theorem book_purchase_cost (x : ℕ) (h1 : x ≤ 100) :
  8 * (100 - x) = (total_books_cost : ℕ) :=
by
  have total_books := 100
  have price_per_B := 8
  have cost_of_B := price_per_B * (total_books - x)
  exact cost_of_B_def total_books cost_of_B sorry

end book_purchase_cost_l398_398804


namespace parallel_tan_x_max_f_value_l398_398092

-- Definitions for the conditions
def m (x : Real) : Vector Real := ⟨sin (x - π / 3), 1⟩
def n (x : Real) : Vector Real := ⟨cos x, 1⟩
def f (x : Real) : Real := (sin (x - π / 3)) * (cos x) + 1

-- Parallel condition
def parallel (u v : Vector Real) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Proof Problems
theorem parallel_tan_x (x : Real) (hx : parallel (m x) (n x)) : tan x = 2 + sqrt 3 := by
  sorry

theorem max_f_value :
  ∃ x ∈ Set.Icc 0 (π / 2), f x = (6 - sqrt 3) / 4 := by
  use π * 5 / 12
  split
  sorry -- Proof for x bounds: 0 ≤ π * 5 / 12 ≤ π / 2
  sorry -- Proof that f(x) = (6 - sqrt 3) / 4 when x = π * 5 / 12

end parallel_tan_x_max_f_value_l398_398092


namespace permutations_sum_divisible_by_37_l398_398744

theorem permutations_sum_divisible_by_37 (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
    ∃ k, (100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a) = 37 * k := 
by
  sorry

end permutations_sum_divisible_by_37_l398_398744


namespace rectangle_area_condition_l398_398360

theorem rectangle_area_condition (P Q R A B C D : Point) (d : ℝ) (h1 : d = 6)
  (h2 : circle P d)
  (h3 : circle Q d)
  (h4 : circle R d)
  (h5 : tangent P A B D C)
  (h6 : tangent Q A B D C)
  (h7 : tangent R A B D C)
  (h8 : collinear P Q R) :
  ∃ (area : ℝ), area = 72 :=
by
  -- conditions state PQR are circles with centers on sides of rectangle ABCD
  -- circle centered at Q has diameter 6 and passes through P and R
  -- given these, we calculate the area of the rectangle ABCD
  sorry

end rectangle_area_condition_l398_398360


namespace lines_parallel_or_coinciding_l398_398606

def relationship_between_lines 
  (v1 : ℝ × ℝ × ℝ) 
  (v2 : ℝ × ℝ × ℝ) 
  := if v1 = (-2, -2, -2) • v2 then "parallel or coinciding" else "unknown"

theorem lines_parallel_or_coinciding 
  (v1 : ℝ × ℝ × ℝ := (1, 2, 3)) 
  (v2 : ℝ × ℝ × ℝ := ((-1/2), -1, (-3/2))) 
  : relationship_between_lines v1 v2 = "parallel or coinciding" :=
by {
  -- Here, you could break down or transform the conditions further
  sorry
}

end lines_parallel_or_coinciding_l398_398606


namespace initial_amount_of_liquid_A_l398_398449

-- Definitions for liquids A and B and their ratios in the initial and modified mixtures
def initial_ratio_A_over_B : ℚ := 4 / 1
def final_ratio_A_over_B_after_replacement : ℚ := 2 / 3
def mixture_replacement_volume : ℚ := 30

-- Proof of the initial amount of liquid A
theorem initial_amount_of_liquid_A (x : ℚ) (A B : ℚ) (initial_mixture : ℚ) :
  (initial_ratio_A_over_B = 4 / 1) →
  (final_ratio_A_over_B_after_replacement = 2 / 3) →
  (mixture_replacement_volume = 30) →
  (A + B = 5 * x) →
  (A / B = 4 / 1) →
  ((A - 24) / (B - 6 + 30) = 2 / 3) →
  A = 48 :=
by {
  sorry
}

end initial_amount_of_liquid_A_l398_398449


namespace probability_sum_15_l398_398392

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398392


namespace minimum_value_proof_l398_398555

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (1 / (x + 1)) + (1 / y)

theorem minimum_value_proof (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) :
  minimum_value x y = (3 + 2 * real.sqrt 2) / 2 :=
sorry

end minimum_value_proof_l398_398555


namespace quadratic_root_identity_l398_398201

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398201


namespace variance_of_arithmetic_sequence_eq_20_div_3_l398_398917

noncomputable def arithmetic_sequence_variance : Real :=
  let x₁ := 1
  let common_difference := 1
  let sequence := List.range' x₁ 9 (fun i ↦ x₁ + i * common_difference)
  let mean := (sequence.sum) / sequence.length
  (sequence.map (fun x ↦ (x - mean) ^ 2)).sum / sequence.length

theorem variance_of_arithmetic_sequence_eq_20_div_3 :
  arithmetic_sequence_variance = 20 / 3 :=
by
  sorry

end variance_of_arithmetic_sequence_eq_20_div_3_l398_398917


namespace simon_legos_l398_398734

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l398_398734


namespace min_value_of_a_l398_398582

variable (a : ℝ)

def inequality_solution_set (a : ℝ) : Set ℝ :=
  { x : ℝ | (a * x - a^2 - 4) * (x - 4) > 0 }

def solution_set_size (a : ℝ) : ℕ :=
  Set.finite.toFinset (Set.finite (inequality_solution_set a)).card

theorem min_value_of_a (n : ℕ) : ∃ a : ℝ, solution_set_size a = n ∧ a = -2 :=
  sorry

end min_value_of_a_l398_398582


namespace line_through_P_l398_398908

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Axiom: A line passing through (x1, y1) with slope m has the equation: y - y1 = m * (x - x1)
-- Line passing through P, with slope 3/2 or in form with opposite intercepts

noncomputable def slope_line_eq (x1 y1 m : ℝ) : Prop :=
  ∀ x y, y - y1 = m * (x - x1)

noncomputable def intercept_line_eq (a : ℝ) : Prop :=
  ∀ x y, x / a + y / (-a) = 1

theorem line_through_P 
  (hP1 : P = (2, 3)) 
  (hSlope : slope_line_eq 2 3 (3 / 2)) 
  (hIntercept : intercept_line_eq (-1))
  : ∃ (a b c : ℝ), (a * b * c ≠ 0) ∧ 
    ((a = 3 ∧ b = -2 ∧ c = 0) ∨ (a = 1 ∧ b = -1 ∧ c = 1)) :=
by
  existsi (3, -2, 0)
  split
  sorry
  existsi (1, -1, 1)
  split
  sorry

end line_through_P_l398_398908


namespace binom_alternating_sum_l398_398037

theorem binom_alternating_sum :
  (∑ k in Finset.range 51, (if k % 2 = 0 then 1 else -1) * (Nat.choose 50 k)) = 0 :=
by
  sorry

end binom_alternating_sum_l398_398037


namespace problem1_solution_uniq_l398_398143

theorem problem1_solution_uniq : 
  ∃ x y z : ℝ, (x + y = 2) ∧ (xy - z^{2} = 1) ∧ (x = 1) ∧ (y = 1) ∧ (z = 0) :=
by
  sorry

end problem1_solution_uniq_l398_398143


namespace common_ratio_of_geometric_sequence_l398_398927

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) → 
  a 1 = 3 →
  (4 * a 1, 2 * a 2, a 3) = 
  let x := (4 * a 1) in 
  let y := (2 * a 2) in 
  let z := a 3 in
  (x, y, z) → 
  2 * a 2 - 4 * a 1 = a 3 - 2 * a 2 →
  q = 2 :=
by 
  sorry

end common_ratio_of_geometric_sequence_l398_398927


namespace range_of_x_coordinate_l398_398896

def ellipse := {P : ℝ × ℝ // (P.fst^2 / 4) + P.snd^2 = 1}

def is_obtuse (P : ℝ × ℝ) : Prop :=
  let F1 : ℝ × ℝ := (-√3, 0)
  let F2 : ℝ × ℝ := (√3, 0)
  (P.fst + √3) * (P.fst - √3) + P.snd^2 < 0

theorem range_of_x_coordinate (P : ellipse) (h : is_obtuse P.1) :
  -√(8/3) < P.1.1 ∧ P.1.1 < √(8/3) :=
sorry

end range_of_x_coordinate_l398_398896


namespace ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l398_398685

theorem ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : 0 < a * b * c)
  : a * b + b * c + c * a < (Real.sqrt (a * b * c)) / 2 + 1 / 4 := 
sorry

end ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l398_398685


namespace extreme_points_max_k_l398_398967

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (a * x^2) / 2 + a - x

def condition_1 (a : ℝ) : Prop := 0 < a ∧ a < 1 / Real.e

def condition_2 (k : ℕ) (x : ℝ) (a : ℝ) : Prop :=
  a = 2 ∧ 2 < x ∧ (k : ℝ) * (x - 2) + (2 - 2 * x - x^2) < f x a

theorem extreme_points (a : ℝ) : ∃ x₁ x₂, x₁ ≠ x₂ ∧
  x₁ ∈ ℝ ∧ x₂ ∈ ℝ ∧
  f x₁ a < f x₁ a ∧
  f x₂ a < f x₂ a → 
  condition_1 a := sorry

theorem max_k (k : ℕ) (a : ℝ) : 
  (∀ x : ℝ, 2 < x → condition_2 k x a) → 
  k ≤ 4 := sorry

end extreme_points_max_k_l398_398967


namespace work_increase_l398_398830

variable (W p : ℝ)
variable (hW : W > 0) (hp : p > 0)

theorem work_increase (h_absent : (1 / 3) * p > 0) (h_present : (2 / 3) * p > 0) :
  let initial_work := W / p
  let remaining_people := (2 / 3) * p
  let new_work := W / remaining_people
  let increase_in_work := new_work - initial_work
  increase_in_work = W / (2 * p) := by
  let h1 : remaining_people = (2 / 3) * p := rfl
  let h2 : new_work = 3 * W / (2 * p) := by rw [h1]; field_simp; ring
  let h3 : initial_work = W / p := rfl
  let h4 : increase_in_work = (3 * W / (2 * p)) - (W / p) := by rw [h2, h3]
  let h5 : W / (2 * p) = (3 * W - 2 * W) / (2 * p) := by field_simp; ring
  rw [h4, h5]; field_simp; ring
  sorry

end work_increase_l398_398830


namespace roots_of_quadratic_eq_l398_398155

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398155


namespace sine_transformation_l398_398518

variable (x : ℝ)

def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem sine_transformation :
  ∃ d, d = Real.pi / 4 ∧ ∀ x, f (x + d) = g x :=
by
  sorry

end sine_transformation_l398_398518


namespace range_of_a_l398_398630

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

noncomputable def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem range_of_a : 
  (∀ x ∈ Set.Iio 0, ∃ y, f x = g y a ∧ y = -x) →
  a < Real.sqrt (Real.exp 1) :=
  sorry

end range_of_a_l398_398630


namespace star_figure_area_ratio_l398_398845

-- Defining the circle with radius 3
def radius : ℝ := 3

-- The area of a circle with radius r
def area_circle (r : ℝ) : ℝ := π * r ^ 2

-- The area of a regular hexagon with side length s
def area_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s ^ 2

-- The given condition in the problem
def original_circle_area := area_circle radius
def hexagon_side_length := radius
def hexagon_area := area_hexagon hexagon_side_length

-- The given condition: radius is 3, resulting in hexagon formed by rearranged arcs
theorem star_figure_area_ratio :
  (hexagon_area - (6 * ((1/6) * π * radius ^ 2 - (1/2) * radius ^ 2 * Real.sin (π / 3)))) / original_circle_area = (81 * Real.sqrt 3 - 36 * π) / (36 * π) :=
by sorry

end star_figure_area_ratio_l398_398845


namespace probability_two_cards_sum_to_15_l398_398404

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398404


namespace sum_y_coordinates_of_other_two_vertices_l398_398999

-- Definitions for the problem
def point := (ℝ, ℝ)
def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Given points
def P1 : point := (5, 20)
def P2 : point := (15, -6)

-- Sum of the y-coordinates of the other two vertices
def sum_y_coordinates_of_opposite_vertices 
  (a b : point) : ℝ :=
  let mid := midpoint a b
  in 2 * mid.2

-- Statement to prove
theorem sum_y_coordinates_of_other_two_vertices :
  sum_y_coordinates_of_opposite_vertices P1 P2 = 14 :=
by
  sorry

end sum_y_coordinates_of_other_two_vertices_l398_398999


namespace largest_square_area_l398_398249

noncomputable theory

variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C] [normed_group A] [normed_group B] [normed_group C] [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C]
variables {AB AC BC : ℝ}

theorem largest_square_area
  (triangle_ABC : Type*)
  (right_angle_at_BAC : ∀ A B C, ∠BAC = π / 2)
  (AB_sq BC_sq AC_sq : ℝ)
  (additional_square_on_AC_ext : AC_sq)
  (sum_of_areas_of_squares : AB_sq + BC_sq + AC_sq + additional_square_on_AC_ext = 500) :
  AC_sq = 125 :=
by
sory

end largest_square_area_l398_398249


namespace percentage_boys_playing_soccer_is_correct_l398_398239

-- Definition of conditions 
def total_students := 420
def boys := 312
def soccer_players := 250
def girls_not_playing_soccer := 73

-- Calculated values based on conditions
def girls := total_students - boys
def girls_playing_soccer := girls - girls_not_playing_soccer
def boys_playing_soccer := soccer_players - girls_playing_soccer

-- Percentage of boys playing soccer
def percentage_boys_playing_soccer := (boys_playing_soccer / soccer_players) * 100

-- We assert the percentage of boys playing soccer is 86%
theorem percentage_boys_playing_soccer_is_correct : percentage_boys_playing_soccer = 86 := 
by
  -- Placeholder proof (use sorry as the proof is not required)
  sorry

end percentage_boys_playing_soccer_is_correct_l398_398239


namespace minimum_b_minus_a_l398_398598

def f (x : ℝ) := Real.log x - (1 / x)
def g (a b x : ℝ) := -a * x + b

theorem minimum_b_minus_a
    (a b : ℝ)
    (H1 : ∀ x : ℝ, x > 0 → deriv (fun x => f x - g a b x) x = 0)
    (H2 : ∃ m : ℝ, m > 0 ∧ g a b = f m + deriv f m * (x - m))
    : b - a = -1 := 
sorry

end minimum_b_minus_a_l398_398598


namespace quadratic_roots_vieta_l398_398169

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398169


namespace find_palindromic_number_l398_398739

def is_palindrome (n : ℕ) : Prop :=
  let str : String := toString n
  str = str.reverse

theorem find_palindromic_number (n : ℕ) :
  (n > 9) ∧ (n < 100) ∧ 
  is_palindrome (n * 91) ∧ 
  is_palindrome (n * 93) ∧ 
  is_palindrome (n * 95) ∧ 
  is_palindrome (n * 97) →
  n = 55 := sorry

end find_palindromic_number_l398_398739


namespace line_within_plane_l398_398455

variable (a : Set Point) (α : Set Point)

theorem line_within_plane : a ⊆ α :=
by
  sorry

end line_within_plane_l398_398455


namespace friend_balloons_count_l398_398425

-- Definitions of the conditions
def balloons_you_have : ℕ := 7
def balloons_difference : ℕ := 2

-- Proof problem statement
theorem friend_balloons_count : (balloons_you_have - balloons_difference) = 5 :=
by
  sorry

end friend_balloons_count_l398_398425


namespace Tonya_budget_allocation_l398_398363

def total_budget : ℝ := 300
def budget_per_sister : ℝ := total_budget / 3
def discounted_price_doll : ℝ := 18 - (18 * 0.1)
def discounted_price_board_game : ℝ := 25 - (25 * 0.05)
def discounted_price_lego_set : ℝ := 22 - (22 * 0.15)

def num_dolls := 6
def num_board_games := 4
def num_lego_sets := 5

def total_spent_on_dolls := num_dolls * discounted_price_doll
def total_spent_on_board_games := num_board_games * discounted_price_board_game
def total_spent_on_lego_sets := num_lego_sets * discounted_price_lego_set

def remaining_budget := total_budget - (
  total_spent_on_dolls +
  total_spent_on_board_games +
  total_spent_on_lego_sets
)

theorem Tonya_budget_allocation :
  total_spent_on_dolls + total_spent_on_board_games + total_spent_on_lego_sets + remaining_budget = total_budget :=
by
  sorry

end Tonya_budget_allocation_l398_398363


namespace fill_pool_cost_l398_398362

-- Define the different rates and costs
def rate_hose : ℕ := 100 -- gallons per hour
def rate_pump : ℕ := 150 -- gallons per hour
def cost_per_gallon_hose : ℚ := 0.01 / 10 -- cents per gallon
def cost_per_gallon_pump : ℚ := 0.01 / 8 -- cents per gallon
def time_hose : ℕ := 50 -- hours

theorem fill_pool_cost :
  let volume_pool := rate_hose * time_hose in
  let combined_rate := rate_hose + rate_pump in
  let time_to_fill := volume_pool / combined_rate in
  let cost_hose := (rate_hose * time_to_fill : ℚ) * cost_per_gallon_hose in
  let cost_pump := (rate_pump * time_to_fill : ℚ) * cost_per_gallon_pump in
  (cost_hose + cost_pump) / 100 = 5.75 :=  -- converting cents to dollars
by
  sorry

end fill_pool_cost_l398_398362


namespace geometry_proof_l398_398224

-- Given conditions
variables {SA SM SN : ℝ}
variables {pyramid : Type}
variables (is_regular_quadrilateral_pyramid : ∀ (X : pyramid), 
  (height_eq_base_side (X))) -- Regular quadrilateral pyramid implies heights and base side are equal
variables (M N D B A S C : pyramid) -- Points on the pyramid
variables (on_lateral_edges : M ∈ lateral_edge(S, D) ∧ N ∈ lateral_edge(S, B)) -- On lateral edges SD and SB
variables (perpendicular_AM_CN: (AM ⊥ CN)) -- AM and CN are mutually perpendicular

-- Prove the equality
theorem geometry_proof : 2 * SA * (SM + SN) = SA^2 + SM * SN :=
by sorry

end geometry_proof_l398_398224


namespace tens_digit_of_9_pow_1801_l398_398419

theorem tens_digit_of_9_pow_1801 : 
  ∀ n : ℕ, (9 ^ (1801) % 100) / 10 % 10 = 0 :=
by
  sorry

end tens_digit_of_9_pow_1801_l398_398419


namespace length_of_first_train_l398_398408

-- Definitions from the problem conditions
def speed_first_train_kmh : ℝ := 42
def speed_second_train_kmh : ℝ := 36
def length_second_train_m : ℝ := 280
def time_to_clear_s : ℝ := 18.460061656605934

-- Conversion factor
def kmh_to_ms (v_kmh : ℝ) : ℝ := v_kmh * (1000 / 3600)

def speed_first_train_ms : ℝ := kmh_to_ms speed_first_train_kmh
def speed_second_train_ms : ℝ := kmh_to_ms speed_second_train_kmh

-- Relative speed calculation
def relative_speed_ms : ℝ := speed_first_train_ms + speed_second_train_ms

-- Combined length calculation
def combined_length_m : ℝ := relative_speed_ms * time_to_clear_s

-- Theorem stating the length of the first train
theorem length_of_first_train : combined_length_m - length_second_train_m = 120 :=
sorry

end length_of_first_train_l398_398408


namespace booking_rooms_needed_l398_398011

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398011


namespace triangle_expression_range_l398_398216

theorem triangle_expression_range
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A)
  (h2 : A < 3 * Real.pi / 4)
  (h3 : c * Real.sin A = a * Real.cos C)
  (h4 : B = 3 * Real.pi / 4 - A)
  : (1 : ℝ) < sqrt 3 * Real.sin A - Real.cos (B + Real.pi / 4) ∧ sqrt 3 * Real.sin A - Real.cos (B + Real.pi / 4) ≤ 2 :=
by
  sorry

end triangle_expression_range_l398_398216


namespace least_possible_area_l398_398860

variable (x y : ℝ) (n : ℤ)

-- Conditions
def is_integer (x : ℝ) := ∃ k : ℤ, x = k
def is_half_integer (y : ℝ) := ∃ n : ℤ, y = n + 0.5

-- Problem statement in Lean 4
theorem least_possible_area (h1 : is_integer x) (h2 : is_half_integer y)
(h3 : 2 * (x + y) = 150) : ∃ A, A = 0 :=
sorry

end least_possible_area_l398_398860


namespace pet_shop_total_animals_l398_398460

theorem pet_shop_total_animals 
  (kittens : ℕ) (hamsters : ℕ) (birds : ℕ) 
  (hk : kittens = 32) (hh : hamsters = 15) (hb : birds = 30) : 
  kittens + hamsters + birds = 77 := 
by 
  rw [hk, hh, hb]
  norm_num

end pet_shop_total_animals_l398_398460


namespace polynomial_value_at_minus_2_l398_398409

-- Define the polynomial f(x)
def f (x : ℤ) := x^6 - 5 * x^5 + 6 * x^4 + x^2 + 3 * x + 2

-- Define the evaluation point
def x_val : ℤ := -2

-- State the theorem we want to prove
theorem polynomial_value_at_minus_2 : f x_val = 320 := 
by sorry

end polynomial_value_at_minus_2_l398_398409


namespace probability_two_cards_sum_15_from_standard_deck_l398_398394

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398394


namespace Samantha_last_name_length_l398_398304

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end Samantha_last_name_length_l398_398304


namespace length_of_BC_l398_398339

def triangle_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 20

def triangle_area (a b : ℝ) : Prop :=
  (1/2) * a * b * (Real.sqrt 3 / 2) = 10

theorem length_of_BC (a b c : ℝ) (h1 : triangle_perimeter a b c) (h2 : triangle_area a b) : c = 7 :=
  sorry

end length_of_BC_l398_398339


namespace function_monotonicity_l398_398963

open Real

theorem function_monotonicity (m : ℝ) (f : ℝ → ℝ)
  (h_fun : ∀ x, x ∈ Ici (-2) → f x = m * x ^ 2 - 2 * x + 3)
  (h_ineq : ∀ x1 x2, x1 ∈ Ici (-2) → x2 ∈ Ici (-2) → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) :
  m ∈ Icc (-(1 / 2)) 0 :=
sorry

end function_monotonicity_l398_398963


namespace sum_of_money_l398_398073

def SI (P R T : ℝ) : ℝ := P * R * T / 100

def CI (P R T : ℝ) : ℝ := P * (1 + R / 100) ^ T - P

theorem sum_of_money (P : ℝ) (R : ℝ) (T : ℝ) (h1 : R = 10) (h2 : T = 2) (h3 : CI P R T - SI P R T = 15) : P = 1500 :=
by
  sorry

end sum_of_money_l398_398073


namespace derivative_y_l398_398052

noncomputable def y (x : ℝ) :=
  (sinh x) / (4 * (cosh x)^4) + (3 * (sinh x)) / (8 * (cosh x)^2) + (3 / 8) * arctan (sinh x)

theorem derivative_y  (x : ℝ) :
  deriv y x = (1 - 3 * sinh(x)^2 - 3) / (4 * cosh(x)^5) := sorry

end derivative_y_l398_398052


namespace dataSetAverage_l398_398865

variable (x : ℕ) (S : Set ℕ)
-- Define the set S containing 1, x, 5, 7
def dataSet : Set ℕ := {1, x, 5, 7}

-- Define the conditions
def uniqueMode (S : Set ℕ) : Prop := -- placeholder for the definition of unique mode
sorry

def medianIsSix (S : List ℕ) : Prop := 
  S.nthLe 1 sorry = 6 -- Assumption that the nth element in sorted order is due to sorted input

theorem dataSetAverage (x : ℕ) 
  (h1: uniqueMode (dataSet x))
  (h2: medianIsSix (dataSet x).toList.sorted) 
  : (1 + x + 5 + 7) / 4 = 5 := sorry

end dataSetAverage_l398_398865


namespace hearing_aid_cost_l398_398659

theorem hearing_aid_cost
  (cost : ℝ)
  (insurance_coverage : ℝ)
  (personal_payment : ℝ)
  (total_aid_count : ℕ)
  (h : total_aid_count = 2)
  (h_insurance : insurance_coverage = 0.80)
  (h_personal_payment : personal_payment = 1000)
  (h_equation : personal_payment = (1 - insurance_coverage) * (total_aid_count * cost)) :
  cost = 2500 :=
by
  sorry

end hearing_aid_cost_l398_398659


namespace length_PQ_equals_external_common_tangent_l398_398828

theorem length_PQ_equals_external_common_tangent
  {O O' P Q : Point}
  (O_ext_O' : ∀ point_O', point_O' ∈ circle O' → point_O' ∉ circle O)
  (PQ_intersection : PQ_is_intersection_of_internal_with_external_common_tangents O O' P Q) :
  length_segment P Q = length_external_common_tangent O O' :=
sorry

end length_PQ_equals_external_common_tangent_l398_398828


namespace fraction_sum_product_roots_of_quadratic_l398_398163

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398163


namespace outer_boundary_diameter_l398_398846

def width_jogging_path : ℝ := 4
def width_garden_ring : ℝ := 10
def diameter_pond : ℝ := 12

theorem outer_boundary_diameter : 2 * (diameter_pond / 2 + width_garden_ring + width_jogging_path) = 40 := by
  sorry

end outer_boundary_diameter_l398_398846


namespace sum_of_values_satisfying_equation_l398_398418

noncomputable def sum_of_roots_of_quadratic (a b c : ℝ) : ℝ := -b / a

theorem sum_of_values_satisfying_equation :
  (∃ x : ℝ, (x^2 - 5 * x + 7 = 9)) →
  sum_of_roots_of_quadratic 1 (-5) (-2) = 5 :=
by
  sorry

end sum_of_values_satisfying_equation_l398_398418


namespace sum_of_squares_of_coefficients_l398_398820

theorem sum_of_squares_of_coefficients :
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2 = 550 :=
by
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  have hc4 : Polynomial.coeff p 4 = 5 := sorry
  have hc3 : Polynomial.coeff p 3 = 20 := sorry
  have hc2 : Polynomial.coeff p 2 = 10 := sorry
  have hc1 : Polynomial.coeff p 1 = 5 := sorry
  calc
    (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2
      = 5^2 + 20^2 + 10^2 + 5^2 : by rw [hc4, hc3, hc2, hc1]
      = 25 + 400 + 100 + 25 : by norm_num
      = 550 : by norm_num


end sum_of_squares_of_coefficients_l398_398820


namespace area_enclosed_by_parabola_and_line_l398_398529

def parabola (y : ℝ) : ℝ := (y^2) / 2
def line (y : ℝ) : ℝ := 4 - y

theorem area_enclosed_by_parabola_and_line :
  ∫ y in -4..2, (line y - parabola y) = 18 := 
by
  sorry

end area_enclosed_by_parabola_and_line_l398_398529


namespace smallest_n_for_terminating_decimal_l398_398416

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m : ℕ, (n = m → m > 0 → ∃ (a b : ℕ), n + 103 = 2^a * 5^b)) 
    ∧ n = 22 :=
sorry

end smallest_n_for_terminating_decimal_l398_398416


namespace number_of_possible_values_of_f_2020_2021_l398_398277

section problem

def N_gt_1 := {n : ℕ | n > 1}

def f : N_gt_1 → N_gt_1 := sorry -- The function f is unspecified in the proof

axiom f_multiplicative (m n : N_gt_1) : f (m * n) = f m * f n

axiom f_101_factorial : f ⟨ (finset.range 102).prod, sorry ⟩ = ⟨ (finset.range 102).prod, sorry ⟩

theorem number_of_possible_values_of_f_2020_2021 :
  (∃! k : ℕ, k = 66) :=
sorry

end problem

end number_of_possible_values_of_f_2020_2021_l398_398277


namespace sum_of_interior_angles_regular_polygon_l398_398779

theorem sum_of_interior_angles_regular_polygon (exterior_angle : ℝ) (h : exterior_angle = 30) :
  let n := 360 / exterior_angle in
  180 * (n - 2) = 1800 :=
by {
  have hn : n = 12 := by sorry,
  rw hn,
  simp,
  sorry
}

end sum_of_interior_angles_regular_polygon_l398_398779


namespace prob_sum_15_correct_l398_398373

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398373


namespace rearrange_to_square_l398_398219

structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

def remove_central_cutout (orig : Rectangle) (cutout : Rectangle) : ℕ :=
  (orig.width * orig.height) - (cutout.width * cutout.height)

theorem rearrange_to_square (orig : Rectangle) (cutout : Rectangle) :
  orig.width = 10 ∧ orig.height = 7 ∧ cutout.width = 1 ∧ cutout.height = 6 →
  remove_central_cutout orig cutout = 64 →
  ∃ (square : Rectangle), square.width = 8 ∧ square.height = 8 :=
by
  intro h₁ h₂
  exists ⟨8, 8⟩
  simp [remove_central_cutout] at h₂
  simp [h₂]
  sorry

end rearrange_to_square_l398_398219


namespace no_such_n_exists_l398_398669

theorem no_such_n_exists (w x y z : ℂ) (n : ℕ) (h_pos_n : n > 0)
    (h1 : 1/w + 1/x + 1/y + 1/z = 3)
    (h2 : w * x + w * y + w * z + x * y + x * z + y * z = 14)
    (h3 : (w + x)^3 + (w + y)^3 + (w + z)^3 + (x + y)^3 + (x + z)^3 + (y + z)^3 = 2160)
    (h4 : w + x + y + z + complex.I * complex.sqrt n ∈ ℝ) : false :=
by
  sorry

end no_such_n_exists_l398_398669


namespace remainder_of_3_pow_244_mod_5_l398_398415

theorem remainder_of_3_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l398_398415


namespace length_of_parallel_line_closer_to_base_is_12_sqrt_2_l398_398323

-- Given definitions and conditions:
def base_length: ℝ := 24
def area_ratio_middle: ℝ := 1 / 2  -- This line divides the triangle into two equal areas
def area_ratio_closer_line: ℝ := 1 / 2  -- The closer line then divides one of the halves into two equal areas as well

-- The proof statement:
theorem length_of_parallel_line_closer_to_base_is_12_sqrt_2 :
  let PQ_length := base_length * (Real.sqrt area_ratio_middle) in
  PQ_length = 12 * Real.sqrt 2 :=
by
  sorry

end length_of_parallel_line_closer_to_base_is_12_sqrt_2_l398_398323


namespace right_triangle_distances_l398_398640

theorem right_triangle_distances 
  {A B C M : Type*} {x d h : ℝ}
  (h_positive : h > 0)
  (d_nonnegative : d ≥ 0)
  (right_angle : ∠BAC = 90)
  (BC_eq : dist B C = 2 * d)
  (AC_eq : dist A C = h)
  (BM_MC_eq : dist B M + dist M C = sqrt (h^2 + 4 * d^2))
  (AM_eq : dist A M = x) :
  x = (h - sqrt (h^2 - 8 * d^2)) / 2 := 
sorry

end right_triangle_distances_l398_398640


namespace two_cards_sum_to_15_proof_l398_398381

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398381


namespace grandmother_cheapest_option_l398_398986

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end grandmother_cheapest_option_l398_398986


namespace acute_angle_parallel_vectors_l398_398956

theorem acute_angle_parallel_vectors 
  (α : ℝ) 
  (h_parallel : (∃ k : ℝ, k ≠ 0 ∧ (3 * real.cos α, 2) = (k * 3, k * 4 * real.sin α)))
  (h_acute : 0 < α ∧ α < real.pi / 2) : 
  α = real.pi / 4 :=
sorry

end acute_angle_parallel_vectors_l398_398956


namespace find_value_of_expression_l398_398268

noncomputable def root_finder (a b c : ℝ) : Prop :=
  a^3 - 30*a^2 + 65*a - 42 = 0 ∧
  b^3 - 30*b^2 + 65*b - 42 = 0 ∧
  c^3 - 30*c^2 + 65*c - 42 = 0

theorem find_value_of_expression {a b c : ℝ} (h : root_finder a b c) :
  a + b + c = 30 ∧ ab + bc + ca = 65 ∧ abc = 42 → 
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) = 770/43 :=
by
  sorry

end find_value_of_expression_l398_398268


namespace break_25_ruble_bill_l398_398805

theorem break_25_ruble_bill (x y z : ℕ) :
  (x + y + z = 11 ∧ 1 * x + 3 * y + 5 * z = 25) ↔ 
    (x = 4 ∧ y = 7 ∧ z = 0) ∨ 
    (x = 5 ∧ y = 5 ∧ z = 1) ∨ 
    (x = 6 ∧ y = 3 ∧ z = 2) ∨ 
    (x = 7 ∧ y = 1 ∧ z = 3) :=
sorry

end break_25_ruble_bill_l398_398805


namespace periodic_seq_root_of_quadratic_root_of_quadratic_eventually_periodic_l398_398864

noncomputable def seq (x0 : ℝ) : ℕ → ℝ
| 0 => x0
| (n + 1) => 1 / (seq x0 n - Real.floor (seq x0 n))

theorem periodic_seq_root_of_quadratic 
  (x0 : ℝ) (hx0 : x0 > 1) (hperiodic : ∃ p > 0, ∀ n, seq x0 (n + p) = seq x0 n) : 
  ∃ (a b c : ℤ), a ≠ 0 ∧ a * x0^2 + b * x0 + c = 0 := 
sorry

theorem root_of_quadratic_eventually_periodic 
  (x0 : ℝ) (hquad : ∃ (a b c : ℤ), a ≠ 0 ∧ a * x0^2 + b * x0 + c = 0) : 
  ∃ N, ∃ p > 0, ∀ n ≥ N, seq x0 (n + p) = seq x0 n := 
sorry

end periodic_seq_root_of_quadratic_root_of_quadratic_eventually_periodic_l398_398864


namespace symmetric_point_l398_398650

theorem symmetric_point (x y : ℝ) : 
  (∃ a b : ℝ, (0, 4) = (a, b) ∧ (∃ m n : ℝ, (m, n) = (3, 1) ∧ 
  (m = (2 * a - x) ∧ n = (2 * b - y) ∧ x - y + 1 = 0)) :=
sorry

end symmetric_point_l398_398650


namespace percent_relation_l398_398829

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end percent_relation_l398_398829


namespace limit_xn_eq_one_and_limit_xn_exp_n_eq_e_l398_398263

open Real Nat

def seq (x : ℕ → ℝ) : Prop :=
  x 1 = 2 ∧ ∀ n : ℕ, n > 0 → x (n + 1) = sqrt (x n + 1 / n)

theorem limit_xn_eq_one_and_limit_xn_exp_n_eq_e (x : ℕ → ℝ) (h : seq x) :
  (tendsto x at_top (𝓝 1)) ∧ tendsto (fun n => x n ^ n) at_top (𝓝 Real.exp 1) :=
sorry

end limit_xn_eq_one_and_limit_xn_exp_n_eq_e_l398_398263


namespace max_sin_y_minus_cos2_x_l398_398945

theorem max_sin_y_minus_cos2_x (x y : ℝ) (h : sin x + sin y = 1/3) :
  (sin y - cos x ^ 2) ≤ 4/9 :=
sorry

end max_sin_y_minus_cos2_x_l398_398945


namespace sale_price_of_sarees_l398_398346

noncomputable def successive_discount (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount, price * (1 - discount / 100)) original_price

theorem sale_price_of_sarees
  (original_price : ℝ)
  (discounts : List ℝ)
  (h₁ : original_price = 495)
  (h₂ : discounts = [15, 10, 5, 3]) :
  abs (successive_discount original_price discounts - 348.95) < 0.01 :=
by
  sorry

end sale_price_of_sarees_l398_398346


namespace intervals_of_increase_range_of_b_l398_398609

def vector_m (ω x : ℝ) : ℝ × ℝ := (sqrt 3 * real.sin (ω * x), 1)
def vector_n (ω x : ℝ) : ℝ × ℝ := (real.cos (ω * x), real.cos (ω * x)^2 + 1)
def f (ω x b : ℝ) : ℝ := (vector_m ω x).1 * (vector_n ω x).1 + (vector_m ω x).2 * (vector_n ω x).2 + b

theorem intervals_of_increase (ω : ℝ) (hω : 0 ≤ ω ∧ ω ≤ 3) :
    ∃ k : ℤ, ∀ x, f ω x 0 = sin (2 * ω * x + π / 6) + 3 / 2 + 0 →
        (∀ x ∈ set.Icc (k * π - π / 3) (k * π + π / 6), monotone_increasing (f ω x 0)) :=
sorry

theorem range_of_b (ω : ℝ) (b : ℝ) (hω : ω = 1) (h₁ : 0 ≤ ω ∧ ω ≤ 3)
    (h₂ : x ∈ set.Icc 0 (7 * π / 12)) :
    ∃ b : ℝ, f 1 0 (b) = f 1 x (b) →
    (b ∈ set.Icc (-2) ((sqrt 3 - 3) /2) ∪ {-5 / 2}) :=
sorry

end intervals_of_increase_range_of_b_l398_398609


namespace general_formula_correct_sequence_T_max_term_l398_398928

open Classical

noncomputable def geometric_sequence_term (n : ℕ) : ℝ :=
  if h : n > 0 then (-1)^(n-1) * (3 / 2^n)
  else 0

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  if h : n > 0 then 1 - (-1 / 2)^n
  else 0

noncomputable def sequence_T (n : ℕ) : ℝ :=
  geometric_sequence_sum n + 1 / geometric_sequence_sum n

theorem general_formula_correct :
  ∀ n : ℕ, n > 0 → geometric_sequence_term n = (-1)^(n-1) * (3 / 2^n) :=
sorry

theorem sequence_T_max_term :
  ∀ n : ℕ, n > 0 → sequence_T n ≤ sequence_T 1 ∧ sequence_T 1 = 13 / 6 :=
sorry

end general_formula_correct_sequence_T_max_term_l398_398928


namespace distance_between_points_l398_398493

-- Definitions of the points P and Q with their coordinates
def P (x1 k b : ℝ) : ℝ × ℝ := (x1, k * x1 + b)
def Q (x2 k b : ℝ) : ℝ × ℝ := (x2, k * x2 + b)

-- The statement we want to prove
theorem distance_between_points (x1 x2 k b : ℝ) :
  let P := P x1 k b;
      Q := Q x2 k b in
  dist P Q = |x1 - x2| * Real.sqrt (1 + k^2) :=
by
  sorry

end distance_between_points_l398_398493


namespace prob_roots_satisfy_cond_is_two_thirds_l398_398858

noncomputable def prob_roots_satisfy_cond : ℝ :=
  ∫ k in Icc 11 18, if ∃ x1 x2 : ℝ, (k^2 + 2 * k - 99) * x1^2 + (3 * k - 7) * x1 + 2 = 0 
    ∧ (k^2 + 2 * k - 99) * x2^2 + (3 * k - 7) * x2 + 2 = 0 ∧ x1 ≤ 2 * x2 then 1 else 0 / (18 - 11)

theorem prob_roots_satisfy_cond_is_two_thirds :
  prob_roots_satisfy_cond = 2 / 3 :=
sorry

end prob_roots_satisfy_cond_is_two_thirds_l398_398858


namespace two_cards_sum_to_15_proof_l398_398379

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398379


namespace power_multiplication_result_l398_398411

theorem power_multiplication_result :
  ( (8 / 9)^3 * (1 / 3)^3 * (2 / 5)^3 = (4096 / 2460375) ) :=
by
  sorry

end power_multiplication_result_l398_398411


namespace product_of_cubes_in_geometric_sequence_l398_398508

def is_geometric_sequence (a : ℕ) (r : ℕ) (terms : List ℕ) : Prop :=
  terms = [a, a * r, a * r^2, a * r^3, a * r^4, a * r^5]

def is_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k ^ 3 = n

def product_of_cubes (terms : List ℕ) : ℕ :=
  terms.foldl (λ acc term => if is_cube term then acc * term else acc) 1

theorem product_of_cubes_in_geometric_sequence :
  ∃ (terms : List ℕ), is_geometric_sequence 90 4 3 terms ∧ 
  terms.all (λ x => x < 150) ∧
  terms.sum = 450 ∧ 
  product_of_cubes terms = 90 :=
by
  sorry

end product_of_cubes_in_geometric_sequence_l398_398508


namespace probability_multiple_of_2_or_3_l398_398783

theorem probability_multiple_of_2_or_3 :
  let nums := (Finset.range 30).map (λ x => x + 1)
  let multiples_of_2 := nums.filter (λ x => x % 2 = 0)
  let multiples_of_3 := nums.filter (λ x => x % 3 = 0)
  let multiples_of_6 := nums.filter (λ x => x % 6 = 0)
  (multiples_of_2.card + multiples_of_3.card - multiples_of_6.card) / nums.card = 2 / 3 :=
by sorry

end probability_multiple_of_2_or_3_l398_398783


namespace magic_square_x_value_l398_398649

theorem magic_square_x_value :
  let S := 45 in
  ((∀ a b c d e, 
    a + b + c = b + d + e = c + e + a = S) ∧
  (S = 19 + 15 + 11) ∧
  (S = 19 + 14 + 12) ∧
  (S = 15 + 12 + x)) →
  x = 18 :=
by
  sorry

end magic_square_x_value_l398_398649


namespace sqrt_subtraction_l398_398821

theorem sqrt_subtraction : 
  let a := 49 + 81,
      b := 36 - 25
  in (Real.sqrt a - Real.sqrt b = Real.sqrt 130 - Real.sqrt 11) := by
  sorry

end sqrt_subtraction_l398_398821


namespace diagonals_parallel_to_sides_of_parallelograms_l398_398807

theorem diagonals_parallel_to_sides_of_parallelograms
  {P Q R S A A1 B B1 C C1 D D1 : Type*}
  [parallelogram ABCD]
  [parallelogram A1B1C1D1]
  (hA : A ∈ PQ)
  (hA1 : A1 ∈ PQ)
  (hB : B ∈ QR) 
  (hB1 : B1 ∈ QR)
  (hC : C ∈ RS)
  (hC1 : C1 ∈ RS)
  (hD : D ∈ SP)
  (hD1 : D1 ∈ SP)
  (h_parallel_ABCD : ∀ (X Y : Type*), (X, Y) ∈ ABCD.sides → X ∥ Y)
  (h_parallel_A1B1C1D1 : ∀ (X Y : Type*), (X, Y) ∈ A1B1C1D1.sides → X ∥ Y) :
  (diagonal PR).parallel_to_side AD ∧ 
  (diagonal QS).parallel_to_side BC :=
by
  sorry

end diagonals_parallel_to_sides_of_parallelograms_l398_398807


namespace distance_to_SFL_l398_398481

def distance_per_hour : ℕ := 27
def hours_travelled : ℕ := 3

theorem distance_to_SFL :
  (distance_per_hour * hours_travelled) = 81 := 
by
  sorry

end distance_to_SFL_l398_398481


namespace multiples_5_or_7_not_6_l398_398612

theorem multiples_5_or_7_not_6 (n : ℕ) : 
  (finset.card (finset.filter (λ x : ℕ, (x % 5 = 0 ∨ x % 7 = 0) ∧ x % 6 ≠ 0) (finset.range (n + 1))) = 39) :=
by
  let n := 200
  sorry

end multiples_5_or_7_not_6_l398_398612


namespace chess_pieces_missing_l398_398688

theorem chess_pieces_missing 
  (total_pieces : ℕ) (pieces_present : ℕ) (h1 : total_pieces = 32) (h2 : pieces_present = 28) : 
  total_pieces - pieces_present = 4 := 
by
  -- Sorry proof
  sorry

end chess_pieces_missing_l398_398688


namespace roots_of_quadratic_eq_l398_398156

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398156


namespace average_speed_ratio_l398_398428

theorem average_speed_ratio
  (time_eddy : ℕ)
  (time_freddy : ℕ)
  (distance_ab : ℕ)
  (distance_ac : ℕ)
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_ab = 570)
  (h4 : distance_ac = 300) :
  (distance_ab / time_eddy) / (distance_ac / time_freddy) = 38 / 15 := 
by
  sorry

end average_speed_ratio_l398_398428


namespace base_7_8_difference_l398_398492

theorem base_7_8_difference (a₇ : ℕ) (b₈ : ℕ) (a₇_eq : a₇ = 52430) (b₈_eq : b₈ = 4320) :
  (5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 3 * 7 + 0) - (4 * 8^3 + 3 * 8^2 + 2 * 8 + 0) = 10652 :=
by {
  have a_eq : 5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 3 * 7 + 0 = 12908 := by norm_num,
  have b_eq : 4 * 8^3 + 3 * 8^2 + 2 * 8 + 0 = 2256 := by norm_num,
  rw [a_eq, b_eq],
  exact nat.sub_eq_of_eq_add (by norm_num)
}

end base_7_8_difference_l398_398492


namespace geometric_arithmetic_sum_l398_398559

theorem geometric_arithmetic_sum {a : Nat → ℝ} {b : Nat → ℝ} 
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h_condition : a 3 * a 11 = 4 * a 7)
  (h_equal : a 7 = b 7) :
  b 5 + b 9 = 8 :=
sorry

end geometric_arithmetic_sum_l398_398559


namespace max_value_of_f_max_perimeter_of_triangle_l398_398284

-- Definitions:
def f (x : ℝ) : ℝ := cos (2 * x + (2 * Real.pi) / 3) + 2 * cos x ^ 2

-- Part I:
theorem max_value_of_f :
  (∀ x : ℝ, f(x) ≤ 2) ∧ (∀ x, f(x) = 2 ↔ ∃ k : ℤ, x = - Real.pi / 6 + k * Real.pi) :=
  sorry

-- Part II:
variables (A B C a b c : ℝ) (ha : a = 1)

def perimeter (a b c : ℝ) : ℝ := a + b + c
def in_triangle (A B C : ℝ) : Prop := A + B + C = Real.pi

theorem max_perimeter_of_triangle 
  (h : sin (2 * (B + C) - Real.pi / 6) = -1 / 2)
  (hBC : B + C = 2 * Real.pi / 3) :
  ∀ b c, perimeter 1 b c ≤ 3 :=
  sorry

end max_value_of_f_max_perimeter_of_triangle_l398_398284


namespace hyperbola_dot_product_zero_l398_398130

theorem hyperbola_dot_product_zero
  (a b x y : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_ecc : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 2) :
  let B := (-x, y)
  let C := (x, y)
  let A := (a, 0)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 0 :=
by
  sorry

end hyperbola_dot_product_zero_l398_398130


namespace two_cards_totaling_15_probability_l398_398365

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398365


namespace arithmetic_sequence_a20_l398_398946

theorem arithmetic_sequence_a20 (a : Nat → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l398_398946


namespace determine_f_and_f5_l398_398678

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation :
  ∀ (x y z : ℝ), f(x^2 + y * f(z) + 1) = x * f(x) + z * f(y) + 1

theorem determine_f_and_f5 : (∀ x : ℝ, f(x) = x) ∧ f(5) = 5 := by
  sorry

end determine_f_and_f5_l398_398678


namespace cannot_cover_chessboard_with_one_corner_removed_l398_398894

theorem cannot_cover_chessboard_with_one_corner_removed :
  ¬ (∃ (f : Fin (8*8 - 1) → Fin (64-1) × Fin (64-1)), 
        (∀ (i j : Fin (64-1)), 
          i ≠ j → f i ≠ f j) ∧ 
        (∀ (i : Fin (8 * 8 - 1)), 
          (f i).fst + (f i).snd = 2)) :=
by
  sorry

end cannot_cover_chessboard_with_one_corner_removed_l398_398894


namespace hundredth_smallest_of_S_l398_398276

def S : Set ℕ := {n | ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ n = 2^x + 2^y + 2^z}

noncomputable def smallest_elements (k : ℕ) : List ℕ :=
(List.range 1000000).filter (λ n, n ∈ S) |>.take k |>.sorted (<)

theorem hundredth_smallest_of_S : smallest_elements 100 |>.get? 99 = some 577 := by
  sorry

end hundredth_smallest_of_S_l398_398276


namespace minimum_value_proof_l398_398556

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (1 / (x + 1)) + (1 / y)

theorem minimum_value_proof (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) :
  minimum_value x y = (3 + 2 * real.sqrt 2) / 2 :=
sorry

end minimum_value_proof_l398_398556


namespace connie_earbuds_tickets_l398_398893

theorem connie_earbuds_tickets (total_tickets : ℕ) (koala_fraction : ℕ) (bracelet_tickets : ℕ) (earbud_tickets : ℕ) :
  total_tickets = 50 →
  koala_fraction = 2 →
  bracelet_tickets = 15 →
  (total_tickets / koala_fraction) + bracelet_tickets + earbud_tickets = total_tickets →
  earbud_tickets = 10 :=
by
  intros h_total h_koala h_bracelets h_sum
  sorry

end connie_earbuds_tickets_l398_398893


namespace mowing_time_l398_398693

def lawn : ℝ := 120 * 180
def mower_width_inch : ℝ := 30
def overlap_inch : ℝ := 6
def effective_width_inch := mower_width_inch - overlap_inch
def effective_width_foot : ℝ := effective_width_inch / 12
def lawn_width : ℝ := 180
def lawn_length : ℝ := 120
def strips := lawn_width / effective_width_foot
def total_distance : ℝ := strips * lawn_length
def mowing_speed : ℝ := 6000  -- feet per hour
def time_required := total_distance / mowing_speed

theorem mowing_time : time_required = 1.8 := by
  norm_num
  sorry

end mowing_time_l398_398693


namespace length_of_d_in_proportion_l398_398114

variable (a b c d : ℝ)

theorem length_of_d_in_proportion
  (h1 : a = 3) 
  (h2 : b = 2)
  (h3 : c = 6)
  (h_prop : a / b = c / d) : 
  d = 4 :=
by
  sorry

end length_of_d_in_proportion_l398_398114


namespace giraffe_statue_price_l398_398287

variable (G : ℕ) -- Price of a giraffe statue in dollars

-- Conditions as definitions in Lean 4
def giraffe_jade_usage := 120 -- grams
def elephant_jade_usage := 2 * giraffe_jade_usage -- 240 grams
def elephant_price := 350 -- dollars
def total_jade := 1920 -- grams
def additional_profit_with_elephants := 400 -- dollars

-- Prove that the price of a giraffe statue is $150
theorem giraffe_statue_price : 
  16 * G + additional_profit_with_elephants = 8 * elephant_price → G = 150 :=
by
  intro h
  sorry

end giraffe_statue_price_l398_398287


namespace quadratic_root_identity_l398_398200

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398200


namespace movie_friends_l398_398313

noncomputable def movie_only (M P G MP MG PG MPG : ℕ) : Prop :=
  let total_M := 20
  let total_P := 20
  let total_G := 5
  let total_students := 31
  (MP = 4) ∧ 
  (MG = 2) ∧ 
  (PG = 0) ∧ (MPG = 2) ∧ 
  (M + MP + MG + MPG = total_M) ∧ 
  (P + MP + PG + MPG = total_P) ∧ 
  (G + MG + PG + MPG = total_G) ∧ 
  (M + P + G + MP + MG + PG + MPG = total_students) ∧ 
  (M = 12)

theorem movie_friends (M P G MP MG PG MPG : ℕ) : movie_only M P G MP MG PG MPG := 
by 
  sorry

end movie_friends_l398_398313


namespace probability_two_white_balls_l398_398437

open Finset

theorem probability_two_white_balls (w b : ℕ) (hw : w = 7) (hb : b = 8) :
  (7 + 8 = 15) →
  (choose 7 2 / choose 15 2 = 1 / 5) :=
by
  intros
  sorry

end probability_two_white_balls_l398_398437


namespace projection_of_a_onto_b_l398_398110

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (2, -1, 2)

-- Function to calculate the dot product of 3-vectors
def dot (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Function to calculate the magnitude squared of a 3-vector
def mag_squared (v : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2 + v.3 * v.3

-- Function to calculate the scalar multiplication of a scalar with a 3-vector
def scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

-- Projection of a onto b
def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let c := (dot a b) / (mag_squared b)
  in scalar_mult c b

-- Main theorem statement (no proof)
theorem projection_of_a_onto_b :
  projection a b = (8/9, -4/9, 8/9) :=
by
  sorry

end projection_of_a_onto_b_l398_398110


namespace tangent_lines_to_circle_through_point_l398_398911

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (-1, 5)

theorem tangent_lines_to_circle_through_point :
  ∃ m c : ℝ, (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 4 → (m * x + y + c = 0 → (y = -m * x - c))) ∧
  (m = 5/12 ∧ c = -55/12) ∨ (m = 0 ∧ ∀ x : ℝ, x = -1) :=
sorry

end tangent_lines_to_circle_through_point_l398_398911


namespace probability_two_students_same_course_l398_398803

-- Define the conditions as a structure
structure Conditions where
  num_students : ℕ
  courses : List String
  student_register : ℕ → String

-- Define the problem as a theorem
theorem probability_two_students_same_course :
  ∀ (cond : Conditions), cond.num_students = 3 → cond.courses = ["basketball", "chess", "taekwondo"] →
  (∃ (p : ℚ), p = 2 / 3 ∧ 
    (let student_courses := List.map cond.student_register [1, 2, 3],
         grouped_students := student_courses.groupBy id in
     ∃! course, cond.num_students - 1 = (grouped_students.count (λ l => l.head = course)).length)) :=
sorry

end probability_two_students_same_course_l398_398803


namespace isosceles_right_triangle_area_l398_398772

-- Define the conditions as given in the problem statement
variables (h l : ℝ)
hypothesis (hypotenuse_rel : h = l * Real.sqrt 2)
hypothesis (hypotenuse_val : h = 6 * Real.sqrt 2)

-- Define the formula for the area of an isosceles right triangle
def area_of_isosceles_right_triangle (l : ℝ) : ℝ := (1 / 2) * l * l

-- Define the proof problem statement
theorem isosceles_right_triangle_area : 
  area_of_isosceles_right_triangle l = 18 :=
  sorry

end isosceles_right_triangle_area_l398_398772


namespace quadratic_roots_sum_product_l398_398183

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398183


namespace symmetry_and_monotonicity_triangle_length_l398_398592

-- Function definition
def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + sin (2 * x - π / 6) + cos (2 * x) + 1

-- Theorem 1: Symmetry center and monotonically increasing intervals
theorem symmetry_and_monotonicity :
  ∀ (k : ℤ),
    (∃ x : ℝ, f(x) = 1) ∧
    (∀ x : ℝ, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → f(x + π) ≤ f(x)) :=
sorry

-- Theorem 2: Triangle problem
theorem triangle_length 
  (f_A : ℝ := f (π / 6)) 
  (B : ℝ := π / 4) 
  (a : ℝ := sqrt 3) 
  (c : ℝ) : 
  f_A = 3 → c = (3 * sqrt 2 + sqrt 6) / 2 :=
sorry

end symmetry_and_monotonicity_triangle_length_l398_398592


namespace solve_system_of_inequalities_l398_398312

theorem solve_system_of_inequalities (x : ℝ) :
  (x + 1 < 5) ∧ (2 * x - 1) / 3 ≥ 1 ↔ 2 ≤ x ∧ x < 4 :=
by
  sorry

end solve_system_of_inequalities_l398_398312


namespace proof_AM1P1_plus_P1LP2_plus_P2N2B_eq_ABC_l398_398700

variable {k1 k2 : ℝ}
variable {A B C M1 M2 N1 N2 L P1 P2 : Type*}

axiom ratio_AM1_M1C : ∀ (AM1 M1C CN1 N1B : ℝ), AM1 / M1C = k1 ∧ CN1 / N1B = k1
axiom ratio_AM2_M2C : ∀ (AM2 M2C CN2 N2B : ℝ), AM2 / M2C = k2 ∧ CN2 / N2B = k2
axiom ratio_M1P1_P1N1 : ∀ (M1P1 P1N1 : ℝ), M1P1 / P1N1 = k1
axiom ratio_M2P2_P2N2 : ∀ (M2P2 P2N2 : ℝ), M2P2 / P2N2 = k2

theorem proof_AM1P1_plus_P1LP2_plus_P2N2B_eq_ABC
  (AM1P1 P1LP2 P2N2B ABC : ℝ)
  (h_AM1P1 : ratio_AM1_M1C AM1P1 P1LP2)
  (h_P1LP2 : ratio_AM2_M2C P1LP2 P2N2B)
  (h_P2N2B : ratio_M1P1_P1N1 P2N2B ABC)
  (h_ABC : ratio_M2P2_P2N2 P1LP2 P2N2B) :
  (AM1P1 ^ (1/3)) + (P1LP2 ^ (1/3)) + (P2N2B ^ (1/3)) = ABC ^ (1 / 3) :=
sorry

end proof_AM1P1_plus_P1LP2_plus_P2N2B_eq_ABC_l398_398700


namespace sum_of_r_l398_398667

def integer_part (r : ℝ) : ℤ := int.floor r
def fractional_part (r : ℝ) : ℝ := r - int.floor r

theorem sum_of_r (S : set ℝ)
  (hS : ∀ r ∈ S, 25 * fractional_part r + integer_part r = 125):
  ∑ r in S, r = 2837 :=
sorry

end sum_of_r_l398_398667


namespace f_eq_f_inv_at_2_5_l398_398513

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^2 - 5 * x - 3

-- State the theorem
theorem f_eq_f_inv_at_2_5 : ∀ x : ℝ, f x = 2.5 -> f x = 2x^2 - 5x - 3 :=
by
  sorry

end f_eq_f_inv_at_2_5_l398_398513


namespace distance_between_parallel_sides_l398_398046

/-- Define the lengths of the parallel sides and the area of the trapezium -/
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area : ℝ := 285

/-- Define the condition of the problem: the formula for the area of the trapezium -/
def area_of_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

/-- The problem: prove the distance between the parallel sides is 15 cm -/
theorem distance_between_parallel_sides (h : ℝ) : 
  area_of_trapezium length_side1 length_side2 h = area → h = 15 :=
by
  sorry

end distance_between_parallel_sides_l398_398046


namespace minimum_x_l398_398577

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem minimum_x (x : ℝ) (h : floor (x + 0.1) + floor (x + 0.2) + floor (x + 0.3) + floor (x + 0.4) + floor (x + 0.5) + floor (x + 0.6) + floor (x + 0.7) + floor (x + 0.8) + floor (x + 0.9) = 104) :
  x = 11.5 :=
sorry

end minimum_x_l398_398577


namespace diameter_of_circle_l398_398863

theorem diameter_of_circle (a b : ℕ) (r : ℝ) (h_a : a = 6) (h_b : b = 8) (h_triangle : a^2 + b^2 = r^2) : r = 10 :=
by 
  rw [h_a, h_b] at h_triangle
  sorry

end diameter_of_circle_l398_398863


namespace quadratic_root_sum_and_product_l398_398195

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398195


namespace quadratic_root_identity_l398_398202

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398202


namespace prob_contact_l398_398061

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l398_398061


namespace profit_calculation_l398_398622

theorem profit_calculation
  (P : ℝ)
  (h1 : 9 > 0)  -- condition that there are 9 employees
  (h2 : 0 < 0.10 ∧ 0.10 < 1) -- 10 percent profit is between 0 and 100%
  (h3 : 5 > 0)  -- condition that each employee gets $5
  (h4 : 9 * 5 = 45) -- total amount distributed among employees
  (h5 : 0.90 * P = 45) -- remaining profit to be distributed
  : P = 50 :=
sorry

end profit_calculation_l398_398622


namespace arccos_one_half_eq_pi_over_three_l398_398500

theorem arccos_one_half_eq_pi_over_three : 
  ∀ x : ℝ, cos x = 1/2 → arccos (1/2) = x :=
by
  sorry

end arccos_one_half_eq_pi_over_three_l398_398500


namespace arccos_one_half_l398_398504

theorem arccos_one_half :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_one_half_l398_398504


namespace multiples_5_or_7_but_not_6_count_l398_398615

theorem multiples_5_or_7_but_not_6_count :
  let count_multiples (k : Nat) (N : Nat) :=
    Nat.floor (N / k) in
  let count_conditional (N : Nat) :=
    count_multiples 5 N + count_multiples 7 N - count_multiples 35 N -
    (count_multiples 6 N - count_multiples 30 N - count_multiples 42 N) in
  count_conditional 200 = 40 :=
by
  sorry

end multiples_5_or_7_but_not_6_count_l398_398615


namespace quadratic_roots_sum_product_l398_398187

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398187


namespace town_population_growth_is_62_percent_l398_398218

noncomputable def population_growth_proof : ℕ := 
  let p := 22
  let p_square := p * p
  let pop_1991 := p_square
  let pop_2001 := pop_1991 + 150
  let pop_2011 := pop_2001 + 150
  let k := 28  -- Given that 784 = 28^2
  let pop_2011_is_perfect_square := k * k = pop_2011
  let percentage_increase := ((pop_2011 - pop_1991) * 100) / pop_1991
  if pop_2011_is_perfect_square then percentage_increase 
  else 0

theorem town_population_growth_is_62_percent :
  population_growth_proof = 62 :=
by
  sorry

end town_population_growth_is_62_percent_l398_398218


namespace count_sevens_in_houses_l398_398466

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.to_string.to_list.filter (λ c, c = Char.ofNat (d + '0'.toNat))).length

def count_digit_in_range (start end d : ℕ) : ℕ :=
  List.sum (List.map (λ x, count_digit x d) (List.range' start (end - start + 1)))

theorem count_sevens_in_houses (start end : ℕ) (h : end = 75) :
  count_digit_in_range start end 7 = 12 :=
by
  rw [h]
  -- Insert proof here
  sorry

end count_sevens_in_houses_l398_398466


namespace fraction_sum_product_roots_of_quadratic_l398_398162

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398162


namespace distance_between_paul_and_farthest_is_15_miles_l398_398476

-- Definitions of the conditions
def jay_speed : ℝ := 1 / 20      -- Jay's speed in miles per minute
def paul_speed : ℝ := 3 / 40     -- Paul's speed in miles per minute
def sam_speed : ℝ := 1.5 / 30    -- Sam's speed in miles per minute
def time_minutes : ℝ := 120      -- Total time in minutes (2 hours)

-- Definitions of distances based on speeds and time
def jay_distance : ℝ := jay_speed * time_minutes
def paul_distance : ℝ := paul_speed * time_minutes
def sam_distance : ℝ := sam_speed * time_minutes

-- Total distance between Paul and the farthest among Jay and Sam
def total_distance_between_paul_and_farthest : ℝ := paul_distance + max jay_distance sam_distance

-- The proof statement
theorem distance_between_paul_and_farthest_is_15_miles :
  total_distance_between_paul_and_farthest = 15 :=
by
  -- Define proofs for each utilized property
  sorry

end distance_between_paul_and_farthest_is_15_miles_l398_398476


namespace alpha_necessary_not_sufficient_for_beta_l398_398638

variables (A B P : Type) [metric_space P]

/-- Proposition Alpha: The sum of distances PA and PB is constant. -/
def Alpha (PA PB : P) : Prop := dist A PA + dist B PB = const

/-- Proposition Beta: The trajectory of point P is an ellipse with foci A and B. -/
def Beta (P : P) : Prop := is_ellipse_with_foci A B P

/-- Alpha is a necessary but not sufficient condition for Beta. -/
theorem alpha_necessary_not_sufficient_for_beta (P : P) (h : Alpha (P -> P) (P -> P)) :
  (Alpha (P -> P) (P -> P) → Beta P) ∧ ¬ (Beta P → Alpha (P -> P) (P -> P)) :=
begin
  sorry,
end

end alpha_necessary_not_sufficient_for_beta_l398_398638


namespace hexagon_sum_value_l398_398563

theorem hexagon_sum_value (k : ℕ) (hk : k > 4) :
  let S := k * k * (k * k + 1) / 2,
      total_contribution := 3 * k^2 - 3 * k - 3 in
  hexagon_value_sum k = S * total_contribution :=
sorry

end hexagon_sum_value_l398_398563


namespace smallest_number_is_28_l398_398801

theorem smallest_number_is_28 (a b c : ℕ) (h1 : (a + b + c) / 3 = 30) (h2 : b = 28) (h3 : b = c - 6) : a = 28 :=
by sorry

end smallest_number_is_28_l398_398801


namespace numBalancedStrings_l398_398095

/--
  Define a type for symbols and a function for computing @(S).
-/
inductive Symbol
| X : Symbol
| O : Symbol

def at (S : List Symbol) : Int :=
  S.count (· = Symbol.X) - S.count (· = Symbol.O)

/--
  Define a predicate for balanced strings.
-/
def isBalanced (S : List Symbol) : Prop :=
  ∀ (T : List Symbol), T.isSubList S → -2 ≤ at T ∧ at T ≤ 2

/--
  The main theorem to prove the number of balanced strings of length n is 2^n.
-/
theorem numBalancedStrings (n : Nat) : 
  ∃ N : Nat, N = 2^n ∧ ∀ S : List Symbol, S.length = n → isBalanced S ↔ S.length = N :=
by
  sorry

end numBalancedStrings_l398_398095


namespace exists_int_solution_l398_398267

theorem exists_int_solution (a : ℤ) (h : a < 0) :
  (a = -10 ∨ a = -4) →
  ∃ x : ℤ, a * x^2 - 2 * (a - 3) * x + (a - 2) = 0 :=
by
  intro ha
  cases ha
  case or.inl ha_eq { sorry }
  case or.inr ha_eq { sorry }

end exists_int_solution_l398_398267


namespace grade_appears_no_more_than_twice_l398_398871

variable {grades : Fin 13 → Fin 4}  -- Represent 13 grades, each being one of {2, 3, 4, 5}
variable count : ∀ g : Fin 4, Fin 13 -- Count occurrences of each grade
variable h_sum : ∑ i, (grades i) = sum -- Sum of grades
variable mean_is_int : ∑ i, (grades i) % 13 = 0  -- Mean is an integer

theorem grade_appears_no_more_than_twice :
  ∃ g : Fin 4, count g ≤ 2 := sorry

end grade_appears_no_more_than_twice_l398_398871


namespace fraction_addition_equivalence_l398_398823

theorem fraction_addition_equivalence : 
  ∃ (n : ℚ), ((4 + n) / (7 + n) = 7 / 9) ∧ n = 13 / 2 :=
begin
  sorry
end

end fraction_addition_equivalence_l398_398823


namespace eval_expression_correct_l398_398888

def eval_expression : ℤ :=
  -(-1) + abs (-1)

theorem eval_expression_correct : eval_expression = 2 :=
  by
    sorry

end eval_expression_correct_l398_398888


namespace product_a_5_to_a_100_l398_398916

def a_n (n : ℕ) (hn : n ≥ 5) : ℚ := ((n + 1)^3 - 1) / (n * (n^3 - 1))

theorem product_a_5_to_a_100 : 
  (∏ n in (finset.range 96).map (nat.add 5), a_n n (nat.add_le_add_right (nat.succ_pos 4) 5)) = 8309 / nat.factorial 100  := 
by 
  sorry

end product_a_5_to_a_100_l398_398916


namespace at_least_one_scheme_passes_l398_398314

open ProbabilityTheory

noncomputable def probability_at_least_one_scheme_passes (p_both_pass : ℝ) (independent_schemes : Prop) : ℝ :=
  if independent_schemes then 1 - (1 - p_both_pass) ^ 2 else 0

theorem at_least_one_scheme_passes (p_ab : ℝ) (h_independent : Prop) (h_p_ab : p_ab = 0.3) (h_independent_event : h_independent = true) :
  probability_at_least_one_scheme_passes p_ab h_independent = 0.51 :=
by
  sorry

end at_least_one_scheme_passes_l398_398314


namespace total_remaining_macaroons_l398_398548

-- Define initial macaroons count
def initial_red_macaroons : ℕ := 50
def initial_green_macaroons : ℕ := 40

-- Define macaroons eaten
def eaten_green_macaroons : ℕ := 15
def eaten_red_macaroons : ℕ := 2 * eaten_green_macaroons

-- Define remaining macaroons
def remaining_red_macaroons : ℕ := initial_red_macaroons - eaten_red_macaroons
def remaining_green_macaroons : ℕ := initial_green_macaroons - eaten_green_macaroons

-- Prove the total remaining macaroons
theorem total_remaining_macaroons : remaining_red_macaroons + remaining_green_macaroons = 45 := 
by
  -- Proof omitted
  sorry

end total_remaining_macaroons_l398_398548


namespace probability_two_cards_sum_to_15_l398_398406

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398406


namespace part1_part2_l398_398932

noncomputable def a_seq : ℕ → ℝ
| 0       => 3
| (n + 1) => (5 * a_seq n - 4) / (2 * a_seq n - 1)

noncomputable def b_seq : ℕ → ℝ
| n => (a_seq n - 1) / (a_seq n - 2)

def geom_seq (b : ℕ → ℝ) (r : ℝ) :=
∀ n, b (n + 1) = r * b n

def sum_first_n (f : ℕ → ℝ) (n : ℕ) :=
∑ i in finset.range n, f i

theorem part1 :
  geom_seq b_seq 3 :=
by
  unfold geom_seq
  intros n
  have h := a_seq (n + 1)
  norm_num
  sorry

theorem part2 (n : ℕ) :
  sum_first_n (λ k, k * b_seq k) n = 
  (1/2 : ℝ) + (↑n - 1/2) * (3 : ℝ)^n :=
by
  sorry

end part1_part2_l398_398932


namespace centroid_coincide_l398_398683

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points and conditions for the tetrahedron
variables {A B C D A' B' C' D' A'' B'' C'' D'' : V}
variables (k : ℝ)

-- Helper definitions for the conditions
def on_ray (P Q : V) : Prop := ∃ r : ℝ, r ≥ 0 ∧ P = r • Q

-- Assumptions based on the problem statements
variables (hAA' : on_ray A' A) (hBB' : on_ray B' B)
          (hCC' : on_ray C' C) (hDD' : on_ray D' D)
          (hA'' : on_ray A'' A') (hB'' : on_ray B'' B')
          (hC'' : on_ray C'' C') (hD'' : on_ray D'' D')
          (hEqualRatios : ∇ • A' = ∇ • B' ∧ ∇ • A'' = ∇ • A ∧ ∇ • B'' = ∇ • B
                           ∧ ∇ • C' = ∇ • C ∧ ∇ • D' = ∇ • D)

-- The main theorem to prove the equivalence of centroids
theorem centroid_coincide (hEqualRatios : AA' * AA'' = BB' * BB'' =
                                         CC' * CC'' = DD' * DD'') :
  centroid [A'', B'', C'', D''] = centroid [A, B, C, D] :=
sorry

end centroid_coincide_l398_398683


namespace prism_volume_l398_398322

theorem prism_volume (α β l : ℝ) : 
  ∀ (V_prism : ℝ),
  let h := l * Real.sin β,
      AB := l * Real.cos β,
      AD := (1 / 2) * l * Real.cos β,
      DC := AD * Real.cot (α / 2),
      S_ABC := (1 / 4) * l^2 * (Real.cos β)^2 * Real.cot (α / 2)
  in V_prism = S_ABC * h → V_prism = (1 / 8) * l^3 * Real.sin (2 * β) * Real.cos β * Real.cot (α / 2) :=
by
  sorry

end prism_volume_l398_398322


namespace area_of_triangle_triangle_area_l398_398332

theorem area_of_triangle {A B C P : Type} 
  (AP PB PQ : ℝ) 
  (r : ℝ)
  (hAP : AP = 28)
  (hPB : PB = 32)
  (hr : r = 30)
  :
  let s := (28 * 2 + 32 * 2 + PQ * 2) / 2 in
  let area := r * s in
  let herons_formula_area := real.sqrt((60 + PQ) * PQ * 28 * 32) in
  herons_formula_area = area :=
  sorry

-- Given conditions
noncomputable def AP : ℝ := 28
noncomputable def PB : ℝ := 32
noncomputable def r : ℝ := 30
let PQ := 2.079

-- Theorem statement
theorem triangle_area :
  let s := (28 * 2 + 32 * 2 + PQ * 2) / 2 in
  let area := r * s in
  area = 1862.37 :=
  sorry

end area_of_triangle_triangle_area_l398_398332


namespace PQ_parallel_AB_l398_398644

variables {A B C D M N P Q : Type} 
[metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M] [metric_space N] [metric_space P] [metric_space Q]

noncomputable def midpoint (x y : Type) [metric_space x] [metric_space y] : Type := sorry
noncomputable def radius (x y a: Type) [metric_space x] [metric_space y] [metric_space a]: Type := sorry

-- Given conditions
variables {tri_ABC : Type} [metric_space tri_ABC] (right_triangle : Prop)
variables (midpoint_D : Prop) 
variables (circle_M : Prop) (circle_N : Prop)
variables (P_intersect : Prop) (Q_intersect : Prop)

-- Prove that PQ is parallel to AB
theorem PQ_parallel_AB 
  (h1: right_triangle)
  (h2: midpoint_D)
  (h3: circle_M)
  (h4: circle_N)
  (h5: P_intersect)
  (h6: Q_intersect) : sorry :=
by
  sorry

end PQ_parallel_AB_l398_398644


namespace prime_divisor_condition_l398_398665

open Nat

noncomputable def x : ℕ → ℚ
| 1     := 7 / 2
| (n+1) := x n * (x n - 2)

theorem prime_divisor_condition (a b : ℕ) (h : coprime a b) 
  (h_seq : x 2021 = a / b) (p : ℕ) (hp : prime p) (hpa : p ∣ a) :
  3 ∣ (p - 1) ∨ p = 3 := 
sorry

end prime_divisor_condition_l398_398665


namespace food_price_increase_l398_398636

variable (N P : ℝ)

theorem food_price_increase :
  let N' := N * 0.91,
      P' := (1 / 0.91575) * P in
  N * P = N' * P' →
  (P' / P - 1) * 100 ≈ 9.89 :=
by
  intros _ _ _ _ h
  sorry

end food_price_increase_l398_398636


namespace prob_contact_l398_398062

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l398_398062


namespace actual_time_when_clock_shows_6_00_PM_l398_398259

-- Conditions
def John_Set_Time : Int := 8 * 60 -- 8:00 AM in minutes
def Clock_Show_Time_At_1_23_PM : Int := (8 + 6 + 0.5) * 60 -- 1:23 PM in minutes
def Time_Shown_At_1_23_PM : Int := 5 * 60 + 23 -- 5 hours and 23 minutes in minutes

-- Given that the clock loses time at a constant rate
-- We need to prove that the actual time when the clock shows 6:00 PM is 20:04 (in minutes from 8:00, which is 724.15 minutes)

theorem actual_time_when_clock_shows_6_00_PM :
  let rate := (Time_Shown_At_1_23_PM : ℝ) / (Clock_Show_Time_At_1_23_PM - John_Set_Time : ℝ)
  let time_shown_at_6_PM := (18 * 60 : ℝ) -- 6:00 PM in minutes
  let actual_time := time_shown_at_6_PM / rate + John_Set_Time
  actual_time ≈ (20 * 60 + 4 : ℝ) := 
by
  sorry

end actual_time_when_clock_shows_6_00_PM_l398_398259


namespace minimum_rooms_needed_fans_l398_398025

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398025


namespace part_1_part_2_l398_398308

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 4)

theorem part_1 (a : ℝ) (h : a = 3) :
  { x : ℝ | f x a ≥ 8 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | 1 ≤ x ∧ x ≤ 3 } ∪ { x : ℝ | x > 3 } := 
sorry

theorem part_2 (h : ∃ x : ℝ, f x a - abs (x + 2) ≤ 4) :
  -6 ≤ a ∧ a ≤ 2 :=
sorry

end part_1_part_2_l398_398308


namespace range_of_m_l398_398964

noncomputable def f (x : ℝ) (m : ℝ) :=
if x ≤ 2 then x^2 - m * (2 * x - 1) + m^2 else 2^(x + 1)

theorem range_of_m {m : ℝ} :
  (∀ x, f x m ≥ f 2 m) → (2 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l398_398964


namespace quadratic_roots_sum_product_l398_398188

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398188


namespace simon_legos_l398_398730

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l398_398730


namespace fraction_addition_equivalence_l398_398824

theorem fraction_addition_equivalence : 
  ∃ (n : ℚ), ((4 + n) / (7 + n) = 7 / 9) ∧ n = 13 / 2 :=
begin
  sorry
end

end fraction_addition_equivalence_l398_398824


namespace part_one_part_two_part_three_l398_398590

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

-- First part: Prove that f(x) + f(1 - x) = 1/2
theorem part_one (x : ℝ) : f(x) + f(1 - x) = 1 / 2 :=
sorry

-- Second part: Define the sequence a_n and prove a_n = (n + 1) / 4
def a_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), f (k / n)

theorem part_two (n : ℕ) : a_n n = (n + 1) / 4 :=
sorry

-- Third part: Define the sum of the first n terms of the sequence and prove the range of λ
def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, a_n k

theorem part_three (λ : ℝ) : (∀ n : ℕ, S_n n ≥ λ * a_n n) ↔ λ ∈ Iic 1 :=
sorry

end part_one_part_two_part_three_l398_398590


namespace sequence_properties_l398_398974

-- We define the initial condition and recurrence relation for the sequence {a_n}
def a : ℕ → ℝ
| 0 := 0 -- Define for n=0, {a_n} is only defined for n≥1 in the given problem, so 0 case is dummy
| 1 := 1
| (n + 1) := a n / (a n + 1)

-- Lean statement to prove the values of a_2, a_3, a_4 and the general term formula for a_n
theorem sequence_properties :
  a 2 = 1 / 2 ∧ 
  a 3 = 1 / 3 ∧ 
  a 4 = 1 / 4 ∧ 
  (∀ n : ℕ, 0 < n → ∃ k : ℕ, a n = 1 / k ∧ k = n) ∧
  (∀ n : ℕ, 1 < n → (1 / a (n + 1)) - (1 / a n) = 1) :=
by
  sorry

end sequence_properties_l398_398974


namespace polynomial_root_not_one_l398_398262

open Polynomial

noncomputable def P : Polynomial ℤ := sorry -- Let's assume P is some polynomial with integer coefficients

theorem polynomial_root_not_one (P : Polynomial ℤ) (n : ℤ) (hn : n ≠ 0) (hP : P ((n:ℤ)^2) = 0) (a : ℚ) (ha : a ≠ 0) :
  P (a^2) ≠ 1 :=
sorry

end polynomial_root_not_one_l398_398262


namespace two_cards_sum_to_15_proof_l398_398383

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398383


namespace find_remainder_l398_398576

theorem find_remainder (x y P Q : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 + y^4 = (P + 13) * (x + y) + Q) : Q = 8 :=
sorry

end find_remainder_l398_398576


namespace f_le_one_l398_398586

open Real

theorem f_le_one (x : ℝ) (hx : 0 < x) : (1 + log x) / x ≤ 1 := 
sorry

end f_le_one_l398_398586


namespace sum_of_solutions_abs_eq_l398_398631

theorem sum_of_solutions_abs_eq (x : ℝ) (h : |x - 25| = 50) : 
  ∃ a b : ℝ, (|a - 25| = 50) ∧ (|b - 25| = 50) ∧ (a + b = 50) :=
by
  -- The existence proof
  use 75, -25
  split; simp [abs_sub_comm]
  -- Provided conclusion
  sorry

end sum_of_solutions_abs_eq_l398_398631


namespace choose_4_diff_suit_rank_choose_6_with_all_suits_l398_398233

-- Part a) Proof Statement
theorem choose_4_diff_suit_rank : (13 * 12 * 11 * 10) = 17160 :=
by
  sorry

-- Part b) Proof Statement
theorem choose_6_with_all_suits : (4 * 13^3 * (13.choose 3) + (4.choose 2) * 13^2 * ((13.choose 2)^2)) = 8682544 :=
by
  sorry

end choose_4_diff_suit_rank_choose_6_with_all_suits_l398_398233


namespace avg_visitor_per_day_in_month_with_given_conditions_l398_398452

noncomputable def average_visitors_per_day 
  (visitors_sunday: ℕ) 
  (visitors_other_day: ℕ) 
  (days_in_month: ℕ) 
  (sundays_in_month: ℕ) 
: ℕ := 
  let other_days := days_in_month - sundays_in_month in
  (sundays_in_month * visitors_sunday + other_days * visitors_other_day) / days_in_month

theorem avg_visitor_per_day_in_month_with_given_conditions
  (visitors_sunday: ℕ) 
  (visitors_other_day: ℕ)
  (days_in_month: ℕ) 
  (sundays_in_month: ℕ)
  (other_days: ℕ := days_in_month - sundays_in_month)
  (total_visitors: ℕ := sundays_in_month * visitors_sunday + other_days * visitors_other_day)
  (avg_visitors_per_day: ℕ := total_visitors / days_in_month)
  (h1: visitors_sunday = 510)
  (h2: visitors_other_day = 240)
  (h3: days_in_month = 30)
  (h4: sundays_in_month = 4)
: avg_visitors_per_day = 276 :=
by
  have h5 : other_days = 26 := by simp [h3, h4]
  have h6 : total_visitors = 8280 := by simp [h1, h2, h4, h5]
  have h7 : avg_visitors_per_day = 276 := by simp [h6, h3]
  exact h7

end avg_visitor_per_day_in_month_with_given_conditions_l398_398452


namespace find_phi_l398_398593

-- Definitions and conditions
def f (x : ℝ) (ω φ : ℝ) := tan (ω * x + φ)

-- Given conditions
variables (ω φ : ℝ)
axiom omega_ne_zero : ω ≠ 0
axiom phi_bound : abs φ < (π / 2)
axiom points_adjacent_centers : (f (2 * π / 3) ω φ = 0) ∧ (f (7 * π / 6) ω φ = 0)
axiom monotonic_decreasing : ∀ x, (2 * π / 3 < x) ∧ (x < 4 * π / 3) → f x ω φ ≤ f (x + 1) ω φ

-- The goal to prove
theorem find_phi : φ = -π / 6 :=
sorry

end find_phi_l398_398593


namespace min_likes_both_l398_398808

-- Definitions corresponding to the conditions
def total_people : ℕ := 200
def likes_beethoven : ℕ := 160
def likes_chopin : ℕ := 150

-- Problem statement to prove
theorem min_likes_both : ∃ x : ℕ, x = 110 ∧ x = likes_beethoven - (total_people - likes_chopin) := by
  sorry

end min_likes_both_l398_398808


namespace parallel_line_perpendicular_planes_l398_398948

-- Definitions for line and planes
variables (ι : Type) (α β : Type)
variable [linear_space ι]  -- Assume ι is a line
variable [plane_space α]  -- Assume α is a distinct plane
variable [plane_space β]  -- Assume β is a distinct plane

-- Conditions: ι is a line; α and β are distinct planes
-- The statement to prove correct
theorem parallel_line_perpendicular_planes 
  (h1 : parallel ι α) 
  (h2 : perpendicular ι β) 
  : perpendicular α β := 
sorry

end parallel_line_perpendicular_planes_l398_398948


namespace probability_of_connection_l398_398068

theorem probability_of_connection (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  let num_pairs := 5 * 8 in
  let prob_no_connection := (1 - p) ^ num_pairs in
  1 - prob_no_connection = 1 - (1 - p) ^ 40 := 
by
  let num_pairs := 5 * 8
  have h_num_pairs : num_pairs = 40 := by norm_num
  rw h_num_pairs
  let prob_no_connection := (1 - p) ^ 40
  sorry

end probability_of_connection_l398_398068


namespace min_rooms_needed_l398_398030

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398030


namespace range_of_m_for_roots_greater_than_1_l398_398970

theorem range_of_m_for_roots_greater_than_1:
  ∀ m : ℝ, 
  (∀ x : ℝ, 8 * x^2 - (m - 1) * x + (m - 7) = 0 → 1 < x) ↔ 25 ≤ m :=
by
  sorry

end range_of_m_for_roots_greater_than_1_l398_398970


namespace time_A_left_work_l398_398436

theorem time_A_left_work : 
  ∀ (x : ℕ), (A_work_rate B_work_rate : ℚ) (days_B : ℕ), 
  A_work_rate = 1/45 → B_work_rate = 1/40 → days_B = 23 →
  (x * (85/1800) + days_B * (1/40) = 1) → x = 9 := 
by
  intros x A_work_rate B_work_rate days_B hA hB hDaysB hTotalWork
  sorry

end time_A_left_work_l398_398436


namespace collinear_points_vector_range_l398_398968

-- Proof Problem (I)
variables {a b : Vector} (t : ℝ) (OA OB OC : Vector)
  (noncollinear : ¬ collinear a b)
  (OA_eq : OA = a) (OB_eq : OB = t • b) (OC_eq : OC = (1/3) • (a + b))

theorem collinear_points :
  (t = 1/2 ↔ collinear_points OA OB OC) := by sorry

-- Proof Problem (II)
variables {a b : Vector} (x : ℝ)
  (norm_a : ∥a∥ = 1) (norm_b : ∥b∥ = 1)
  (dot_ab : a ⬝ b = -1/2)
  (x_range : -1 ≤ x ∧ x ≤ 1/2)

theorem vector_range :
  (∥a - x • b∥ = 1/2 ↔ ∥a - x • b∥ ∈ set.Icc (√(3/2)) (√(7/2))) := by sorry

end collinear_points_vector_range_l398_398968


namespace num_ticket_prices_divisors_l398_398447

theorem num_ticket_prices_divisors : 
  let x_values := {x : ℕ | x > 0 ∧ 56 % x = 0 ∧ 98 % x = 0 ∧ 84 % x = 0} 
  in x_values.card = 4 :=
by
  sorry

end num_ticket_prices_divisors_l398_398447


namespace probability_prime_ball_l398_398034

open Finset

theorem probability_prime_ball :
  let balls := {1, 2, 3, 4, 5, 6, 8, 9}
  let total := card balls
  let primes := {2, 3, 5}
  let primes_count := card primes
  (total = 8) → (primes ⊆ balls) → 
  primes_count = 3 → 
  primes_count / total = 3 / 8 :=
by
  intros
  sorry

end probability_prime_ball_l398_398034


namespace minimum_rooms_needed_fans_l398_398018

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398018


namespace fraction_sum_product_roots_of_quadratic_l398_398159

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398159


namespace sum_of_coordinates_l398_398743

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 4) :
  let x := 4
  let y := (f⁻¹ x) / 4
  x + y = 9 / 2 :=
by
  sorry

end sum_of_coordinates_l398_398743


namespace yoon_wins_first_six_games_l398_398257

theorem yoon_wins_first_six_games :
  (∀ n ∈ (Finset.range 6).map (λ x, x + 1), (1 - (1 / (n + 2))) = (n + 1) / (n + 2)) →
  (∀ n ∈ (Finset.range 6).map (λ x, x + 1), P_jae_wins_nth_game n = 1 / (n + 2)) →
  P_yoon_wins_all_6_games = 1 / 4 :=
by sorry

end yoon_wins_first_six_games_l398_398257


namespace minimum_rooms_needed_l398_398008

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398008


namespace omega_range_l398_398924

noncomputable def sin_function_monotonically_decreasing (ω : ℝ) : Prop :=
  ∀ x₁ x₂, (π / 2 < x₁ ∧ x₁ < π) → (π / 2 < x₂ ∧ x₂ < π) → (x₁ < x₂) → 
  (sin (ω * x₁ + π / 4) ≥ sin (ω * x₂ + π / 4))

theorem omega_range :
  {ω : ℝ | ω > 0 ∧ sin_function_monotonically_decreasing ω} = 
  {ω : ℝ | 1/2 ≤ ω ∧ ω ≤ 5/4} := 
  sorry

end omega_range_l398_398924


namespace closed_area_correct_l398_398317

noncomputable def closed_area : ℝ :=
  ∫ x in 2..3, (2 / (x^2 - 1))

theorem closed_area_correct :
  closed_area = Real.log (3 / 2) :=
by
  sorry

end closed_area_correct_l398_398317


namespace mary_flour_requirement_l398_398691

theorem mary_flour_requirement (total_flour required_flour: ℕ) (already_added: ℕ) : total_flour = 7 → already_added = 2 → required_flour = total_flour - already_added → required_flour = 5 :=
  by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw h3
  sorry

end mary_flour_requirement_l398_398691


namespace cistern_leak_empty_time_l398_398427

theorem cistern_leak_empty_time :
  (let R := 1 / 10 in
   let L := R - 1 / 12 in
   1 / L = 60) :=
by
  let R := 1 / 10
  let L := R - 1 / 12
  have hRL : L = 1 / 60 := by sorry
  show 1 / L = 60 from by sorry

end cistern_leak_empty_time_l398_398427


namespace probability_of_contact_l398_398055

noncomputable def probability_connection (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

theorem probability_of_contact (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let group1 := 5
  let group2 := 8
  let total_pairs := group1 * group2
  (total_pairs = 40) →
  (∀ i j, i ∈ fin group1 → j ∈ fin group2 → (¬ p = 1 → p = p)) → 
  probability_connection p = 1 - (1 - p) ^ 40 :=
by
  intros _ _ 
  sorry

end probability_of_contact_l398_398055


namespace composite_positive_integer_property_l398_398514

theorem composite_positive_integer_property :
  ∀ n : ℕ,
    (n > 1 ∧ (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ n = p * q)) →
    (∃ k : ℕ, ∃ divs : Finₓ k → ℕ,
      (divs 0 = 1) ∧
      (divs k = n) ∧
      (∀ i : Finₓ (k-1), divs i < divs (i + 1)) ∧
      (∀ i : Finₓ (k-1), divs (i + 1) - divs i = (i + 1)) →
    (n = 4)) :=
by
  sorry

end composite_positive_integer_property_l398_398514


namespace average_balance_proof_l398_398512

def monthly_balances : List ℕ := [120, 240, 180, 180, 300, 150]

def total_balance : ℕ := monthly_balances.sum

def number_of_months : ℕ := monthly_balances.length

def average_balance : ℕ := total_balance / number_of_months

theorem average_balance_proof : average_balance = 195 := by
  unfold monthly_balances total_balance number_of_months average_balance
  -- Calculate the sum of the monthly balances
  have h_sum : 120 + 240 + 180 + 180 + 300 + 150 = 1170 := rfl
  -- Calculate the number of months
  have h_length : 6 = 6 := rfl
  -- Average calculation
  have h_average : 1170 / 6 = 195 := by norm_num
  exact h_average

end average_balance_proof_l398_398512


namespace find_cheapest_option_l398_398990

variable (transportation_cost : ℕ) (berries_collected : ℕ)
          (cost_train_per_week : ℕ) (cost_berries_market : ℕ)
          (cost_sugar : ℕ) (jam_rate : ℚ) (cost_ready_made_jam : ℕ)
      
-- Define the cost of gathering 1.5 kg of jam
def option1_cost := (cost_train_per_week / berries_collected + cost_sugar) * jam_rate

-- Define the cost of buying berries and sugar to make 1.5 kg of jam
def option2_cost := (cost_berries_market + cost_sugar) * jam_rate

-- Define the cost of buying 1.5 kg of ready-made jam
def option3_cost := cost_ready_made_jam * jam_rate

theorem find_cheapest_option :
  option1_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam 
  < min (option2_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam)
        (option3_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam) :=
by
  unfold option1_cost option2_cost option3_cost
  have hc1 : (40 : ℕ) + 54 = 94 := by norm_num
  have hc2 : (150 : ℕ) + 54 = 204 := by norm_num
  have hc3 : (220 : ℕ) * (3/2) = 330 := by norm_num
  linarith
  sorry

end find_cheapest_option_l398_398990


namespace clocks_distance_property_l398_398698

/--
  There are 50 correctly working clocks on a table.
  Let Ai(t) and Bi(t+30) be the positions of the tips of the minute hands of clock i at times t and t+30 minutes, respectively.
  Let Oi be the center of clock i.
  Let O be the center of the table.
  We need to prove that at some moment, the sum of the distances from the center of the table to the tips
  of the minute hands will be greater than the sum of the distances from the center of the table to the centers of the clocks.
-/
theorem clocks_distance_property
  (n : ℕ) (h_n : n = 50)
  (O : ℝ × ℝ) (O_i : ℕ → ℝ × ℝ)
  (A_i B_i : ℕ → ℕ → ℝ × ℝ) :
  (∀ i, i < n → ∀ t, dist O O_i i ≤ (dist O (A_i i t) + dist O (B_i i (t + 30))) / 2) →
  ∃ t, (Σ i, dist O O_i i) < (Σ i, dist O (A_i i t)) ∨ (Σ i, dist O O_i i) < (Σ i, dist O (B_i i (t + 30))) :=
by
  intros h_triangle
  sorry

end clocks_distance_property_l398_398698


namespace determine_d_l398_398742

-- Define the polynomial equation and its properties
variables (a b c d : ℤ)
variables (h_gcd : Int.gcd a (Int.gcd b (Int.gcd c d)) = 1)
variables (root : ∀ i : ℕ, (3 + Complex.i)^i = (3 - Complex.i)^i)

-- The goal statement:
theorem determine_d (a b c d : ℤ) (h_gcd : Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) (root : ∀ i : ℕ, (3 + Complex.i)^i = (3 - Complex.i)^i) :
  |d| = 3 :=
sorry

end determine_d_l398_398742


namespace fraction_sum_product_roots_of_quadratic_l398_398157

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398157


namespace find_a_range_l398_398760

noncomputable def monotonic_func_a_range : Set ℝ :=
  {a : ℝ | ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → (3 * x^2 + a ≥ 0 ∨ 3 * x^2 + a ≤ 0)}

theorem find_a_range :
  monotonic_func_a_range = {a | a ≤ -27} ∪ {a | a ≥ 0} :=
by
  sorry

end find_a_range_l398_398760


namespace cartesian_eq_circle_intersection_product_MA_MB_l398_398522

-- Definition of the parametric equation of line l
def parametric_eq_line (t : ℝ) : ℝ × ℝ := (1 - (Real.sqrt 2 / 2) * t, 4 - (Real.sqrt 2 / 2) * t)

-- Definition of the polar coordinate equation of the circle
def polar_eq_circle (ρ θ : ℝ) : Prop := ρ = -4 * Real.cos θ

-- Point M with given coordinates
def point_M : ℝ × ℝ := (-2, 1)

theorem cartesian_eq_circle :
  (∀ t, let pt := parametric_eq_line t in pt.1 = 1 - (Real.sqrt 2 / 2) * t ∧ pt.2 = 4 - (Real.sqrt 2 / 2) * t) →
  (∀ ρ θ, polar_eq_circle ρ θ → ((ρ^2 = (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2) ∧ (-4 * Real.cos θ = ρ) → (ρ^2 = 16 * Real.cos θ^2)) →
  (∀ x y, (x + 2)^2 + y^2 = 4) :=
sorry

theorem intersection_product_MA_MB :
  (∀ t, let pt := parametric_eq_line t in pt.1 = 1 - (Real.sqrt 2 / 2) * t ∧ pt.2 = 4 - (Real.sqrt 2 / 2) * t) →
  (∀ x y, (x + 2)^2 + y^2 = 4) →
  (let M := point_M in
  (let t₁ t₂ : ℝ := sorry in
  let A B : ℝ × ℝ := sorry in
  |M.1 - A.1| * |M.1 - B.1| = 3)) :=
sorry

end cartesian_eq_circle_intersection_product_MA_MB_l398_398522


namespace two_cards_sum_to_15_proof_l398_398380

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398380


namespace EO1_eq_FO1_l398_398231

-- Assume the existence of a circle with specified chords and intersection points
variables {P : Type} [metric_space P] -- Assume P is a metric space for generality

-- O, O1, A, B, C, D, G, H, E, F are points in the metric space
variables {O O1 A B C D G H E F : P}

-- Definitions for midpoint and circle
def midpoint (X Y : P) : P := sorry  -- placeholder definition for midpoint

def is_chord (circle : set P) (X Y : P) : Prop := sorry -- placeholder for chord

def is_midpoint_of_chord (X Y M : P) : Prop := midpoint X Y = M

-- Assume O1 is the midpoint of chord GH in circle O
axiom GH_is_chord : is_chord (ball O r) G H
axiom O1_midpoint : is_midpoint_of_chord G H O1

-- Assume AB and CD are chords passing through O1 and intersect GH at E and F respectively
axiom AB_is_chord_passing_O1 : is_chord (ball O r) A B → A = O1 ∨ B = O1
axiom CD_is_chord_passing_O1 : is_chord (ball O r) C D → C = O1 ∨ D = O1
axiom E_intersects_GH : sorry -- Intersection point E on GH
axiom F_intersects_GH : sorry -- Intersection point F on GH

-- Problem statement in Lean
theorem EO1_eq_FO1 (h_circle : metric_space P) (r : ℝ) (O : P) 
  (GH_IS_CHORD: is_chord (ball O r) G H) 
  (O1_MID : is_midpoint_of_chord G H O1)
  (AB_IS_CHORD_O1: ∀ A B, is_chord (ball O r) A B → A = O1 ∨ B = O1)
  (CD_IS_CHORD_O1: ∀ C D, is_chord (ball O r) C D → C = O1 ∨ D = O1)
  (E_on_GH: sorry -- Intersection point E on GH) 
  (F_on_GH: sorry -- Intersection point F on GH)
  : dist E O1 = dist F O1 :=
sorry

end EO1_eq_FO1_l398_398231


namespace isosceles_right_triangle_area_l398_398774

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l398_398774


namespace length_of_24_l398_398537

def length_of_integer (k : ℕ) : ℕ :=
  k.factors.length

theorem length_of_24 : length_of_integer 24 = 4 :=
by
  sorry

end length_of_24_l398_398537


namespace find_middle_number_l398_398319

theorem find_middle_number
  (S1 S2 M : ℤ)
  (h1 : S1 = 6 * 5)
  (h2 : S2 = 6 * 7)
  (h3 : 13 * 9 = S1 + M + S2) :
  M = 45 :=
by
  -- proof steps would go here
  sorry

end find_middle_number_l398_398319


namespace multiples_5_or_7_not_6_l398_398613

theorem multiples_5_or_7_not_6 (n : ℕ) : 
  (finset.card (finset.filter (λ x : ℕ, (x % 5 = 0 ∨ x % 7 = 0) ∧ x % 6 ≠ 0) (finset.range (n + 1))) = 39) :=
by
  let n := 200
  sorry

end multiples_5_or_7_not_6_l398_398613


namespace hyperbola_equation_l398_398600

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ (k : ℝ), k = b / a ∧ k = √3)
    (h4 : ∃ (c : ℝ), c = 4 ∧ a^2 + b^2 = c^2)
    (h5 : b = √3 * a) : 
    (a = 2 ∧ b = 2 * √3) → (∀ x y : ℝ, (x^2/4) - (y^2/12) = 1 ∧ (x^2/a^2) - (y^2/b^2) = 1) :=
by
  intro ha_eq hb_eq
  specialize h4 (4) -- based on condition that focus aligns with parabola's directrix
  specialize h3 (√3) -- based on condition that asymptote passes through given point
  have ha := ha_eq.1
  have hb := hb_eq.2
  sorry

end hyperbola_equation_l398_398600


namespace inequality_abc_geq_36_l398_398309

theorem inequality_abc_geq_36 (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_prod : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 :=
by
  sorry

end inequality_abc_geq_36_l398_398309


namespace approximate_reading_l398_398329

theorem approximate_reading (x : ℝ) (h₁ : 8.5 ≤ x) (h₂ : x ≤ 9.0) : x ≈ 8.9 :=
by
  -- This is an approximate estimation problem, hence the use of ≈ (approx)
  -- The proof is based on visual approximation and hence ∃ y ~ x and y = 8.9
  sorry

end approximate_reading_l398_398329


namespace power_of_i_2016_l398_398837
-- Importing necessary libraries to handle complex numbers

theorem power_of_i_2016 (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) : 
  (i^2016 = 1) :=
sorry

end power_of_i_2016_l398_398837


namespace spend_on_laundry_detergent_l398_398474

def budget : ℕ := 60
def price_shower_gel : ℕ := 4
def num_shower_gels : ℕ := 4
def price_toothpaste : ℕ := 3
def remaining_budget : ℕ := 30

theorem spend_on_laundry_detergent : 
  (budget - remaining_budget) = (num_shower_gels * price_shower_gel + price_toothpaste) + 11 := 
by
  sorry

end spend_on_laundry_detergent_l398_398474


namespace two_cards_totaling_15_probability_l398_398367

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398367


namespace quadratic_roots_sum_product_l398_398182

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398182


namespace probability_sum_15_l398_398386

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398386


namespace minimum_rooms_needed_l398_398003

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398003


namespace determine_a_l398_398580

def f (x a : ℝ) : ℝ := x^2 - a * x + 3
def g (x a : ℝ) : ℝ := x^2 - a * log x

theorem determine_a :
  (∀ x ∈ set.Ioo 0 1, deriv (f x) a ≤ 0) ∧
  (∀ x ∈ set.Ioo 1 2, deriv (g x) a ≥ 0) →
  a = 2 :=
by
  sorry

end determine_a_l398_398580


namespace count_zeros_in_decimal_rep_l398_398516

theorem count_zeros_in_decimal_rep (n : ℕ) (h : n = 2^3 * 5^7) : 
  ∀ (a b : ℕ), (∃ (a : ℕ) (b : ℕ), n = 10^b ∧ a < 10^b) → 
  6 = b - 1 := by
  sorry

end count_zeros_in_decimal_rep_l398_398516


namespace least_a_divisible_by_960_l398_398624

theorem least_a_divisible_by_960:
  ∃ a : ℤ, (a^5 % 960 = 0) ∧ (∀ b : ℤ, (b^5 % 960 = 0) → a ≤ b) → a = 60 :=
begin
  sorry
end

end least_a_divisible_by_960_l398_398624


namespace booking_rooms_needed_l398_398015

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398015


namespace cos_double_angle_l398_398091

variable (α : ℝ)
variable (h : Real.cos α = 2/3)

theorem cos_double_angle : Real.cos (2 * α) = -1/9 :=
  by
  sorry

end cos_double_angle_l398_398091


namespace cement_tesss_street_l398_398725

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end cement_tesss_street_l398_398725


namespace minimum_rooms_needed_fans_l398_398024

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398024


namespace wire_length_around_two_poles_l398_398811

/-- Define the diameters of the poles -/
def diameter_pole1 := 4
def diameter_pole2 := 16

/-- Radii of the poles -/
def radius_pole1 := diameter_pole1 / 2
def radius_pole2 := diameter_pole2 / 2

/-- Distance between the centers of the poles -/
def center_distance := radius_pole1 + radius_pole2

/-- Difference in radii -/
def radius_difference := radius_pole2 - radius_pole1

/-- Pythagorean theorem to calculate the straight section of the wire -/
def straight_section_length := 2 * Real.sqrt (center_distance ^ 2 - radius_difference ^ 2)

/-- Curved sections of the wire -/
/-- Smaller circle arc length -/
def smaller_circle_arc_length := (1 / 4) * (2 * Real.pi * radius_pole1)

/-- Larger circle arc length -/
def larger_circle_arc_length := (3 / 4) * (2 * Real.pi * radius_pole2)

/-- Total wire length calculation -/
def total_wire_length := straight_section_length + smaller_circle_arc_length + larger_circle_arc_length

theorem wire_length_around_two_poles :
  total_wire_length = 16 + 13 * Real.pi :=
by
  sorry

end wire_length_around_two_poles_l398_398811


namespace bryden_receives_correct_amount_l398_398847

-- Definitions based on conditions
def collector_offer : ℝ := 1500 / 100 -- 1500% as a multiplier
def num_quarters : ℤ := 10 -- Bryden has ten state quarters
def face_value_per_quarter : ℝ := 0.25 -- Face value of each state quarter

-- Statement to prove Bryden will receive $37.5 for his state quarters
theorem bryden_receives_correct_amount : 
    (collector_offer * (num_quarters * face_value_per_quarter)) = 37.5 :=
by
    sorry

end bryden_receives_correct_amount_l398_398847


namespace random_walk_expected_distance_l398_398461

noncomputable def expected_distance_after_random_walk (n : ℕ) : ℚ :=
(sorry : ℚ) -- We'll define this in the proof

-- Proof problem statement in Lean 4
theorem random_walk_expected_distance :
  expected_distance_after_random_walk 6 = 15 / 8 :=
by 
  sorry

end random_walk_expected_distance_l398_398461


namespace quadratic_root_identity_l398_398199

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398199


namespace quadratic_roots_sum_product_l398_398181

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398181


namespace bug_total_distance_l398_398842

noncomputable def total_distance (r : ℕ) (d2 : ℕ) (d3 : ℝ) : ℝ := 
  let d1 := 2 * r
  let x := real.sqrt (d1^2 - d2^2)
  d1 + d2 + x

theorem bug_total_distance : total_distance 75 100 = 150 + 100 + 50 * real.sqrt 5 :=
by
  sorry

end bug_total_distance_l398_398842


namespace exists_points_within_distance_l398_398565

noncomputable def equilateral_triangle (side_length : ℝ) : Prop :=
∀ (A B C : ℝ × ℝ), 
  (dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length)

theorem exists_points_within_distance (A B C : ℝ × ℝ)
  (hABC : equilateral_triangle 2)
  (p1 p2 p3 p4 p5 : ℝ × ℝ)
  (h_in_triangle : ∀ p, p ∈ {p1, p2, p3, p4, p5} → p inside_triangle (A, B, C)) :
  ∃ (p q ∈ {p1, p2, p3, p4, p5}), dist p q ≤ 1 := 
sorry

end exists_points_within_distance_l398_398565


namespace variance_scale_l398_398933

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def variance (s : Finset α) : ℝ :=
 sorry -- definition of variance

theorem variance_scale (s : Finset ℝ) (k : ℝ) (v : ℝ) (h : variance s = v) :
  variance (s.map (fun x => k • x)) = k^2 * v :=
 sorry

end variance_scale_l398_398933


namespace hakeem_money_needed_l398_398992

variable (numArtichokesPer5oz : ℕ) (ozDipTarget : ℕ) (costPerArtichoke : ℝ)
-- Given conditions
def amountOfDip (numArtichokes : ℕ) : ℕ := (numArtichokes * 5) / numArtichokesPer5oz
def costForArtichokes (numArtichokes : ℕ) : ℝ := numArtichokes * costPerArtichoke

-- Main theorem
theorem hakeem_money_needed :
  numArtichokesPer5oz = 3 → 
  ozDipTarget = 20 →
  costPerArtichoke = 1.25 →
  costForArtichokes (ozDipTarget * numArtichokesPer5oz / 5) = 15.0 := 
by
  intros h1 h2 h3
  sorry

end hakeem_money_needed_l398_398992


namespace angle_of_inclination_l398_398040

-- The statement of the mathematically equivalent proof problem in Lean 4
theorem angle_of_inclination
  (k: ℝ)
  (α: ℝ)
  (line_eq: ∀ x, ∃ y, y = (k-1) * x + 2)
  (circle_eq: ∀ x y, x^2 + y^2 + k * x + 2 * y + k^2 = 0) :
  α = 3 * Real.pi / 4 :=
sorry -- Proof to be provided

end angle_of_inclination_l398_398040


namespace minimum_rooms_needed_l398_398006

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398006


namespace prob_contact_l398_398060

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l398_398060


namespace variance_scaled_data_l398_398937

-- Definition of variance
def variance (data : List ℝ) : ℝ :=
  let mean := (data.foldl (+) 0) / (data.length : ℝ)
  (data.map (λ x => (x - mean)^2)).foldl (+) 0 / (data.length : ℝ)

-- Original data set and its variance
def original_data := [x_1, x_2, ..., x_n]
def original_variance := 0.01

-- Prove that the variance of 10 times each data point is 1
theorem variance_scaled_data :
  variance (original_data.map (λ x => 10 * x)) = 1 := by
  sorry

end variance_scaled_data_l398_398937


namespace range_of_m_l398_398093

theorem range_of_m (a b c : ℝ) (m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 :=
sorry

end range_of_m_l398_398093


namespace distinct_z_values_count_l398_398118

theorem distinct_z_values_count : 
  ∃ (z_values : Finset ℕ), 
    (∀ x y : ℕ, 
      (100 ≤ x ∧ x ≤ 999) → 
      (100 ≤ y ∧ y ≤ 999) → 
      (y = x % 10 * 100 + x / 10 % 10 * 10 + x / 100) → 
      (z_values = Finset.image (λ (ac : ℕ × ℕ), 99 * (Nat.abs (ac.1 - ac.2)))
        (Finset.filter 
          (λ (ac : ℕ × ℕ), ac.1 ∈ Finset.range 10 ∧ ac.2 ∈ Finset.range 10)
          (Finset.product (Finset.range 10) (Finset.range 10)))) ∧ 
        (z_values.card = 10).

end distinct_z_values_count_l398_398118


namespace triangle_stick_sum_l398_398468

theorem triangle_stick_sum:
  let valid_n (n : ℕ) := 5 ≤ n ∧ n < 18 in
  (∑ n in (Finset.filter valid_n (Finset.range 18)), n) = 143 :=
by
  sorry

end triangle_stick_sum_l398_398468


namespace sqrt_subtraction_l398_398822

theorem sqrt_subtraction : 
  let a := 49 + 81,
      b := 36 - 25
  in (Real.sqrt a - Real.sqrt b = Real.sqrt 130 - Real.sqrt 11) := by
  sorry

end sqrt_subtraction_l398_398822


namespace problem_statement_l398_398919

noncomputable def five_digit_number_count : ℕ := 
  let digits := [5, 6, 7, 8, 9]
  let is_odd (n : ℕ) := n % 2 = 1
  let is_even (n : ℕ) := n % 2 = 0
  let valid_permutation (l : List ℕ) :=
    l.length = 5 ∧ 
    l.nodup ∧ 
    ∃ i, 1 ≤ i ∧ i + 2 < l.length ∧ 
    is_odd (l[i - 1]) ∧ is_even (l[i]) ∧ is_odd (l[i + 1])
  (List.permutations digits).countp valid_permutation

theorem problem_statement : five_digit_number_count = 48 := 
sorry

end problem_statement_l398_398919


namespace monotonically_increasing_interval_l398_398497

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≤ f y

noncomputable def resulting_function (x : ℝ) : ℝ :=
  sin (2 * x + π / 6)

theorem monotonically_increasing_interval :
  is_monotonically_increasing resulting_function (set.Ioo (-π / 3) (π / 6)) :=
sorry

end monotonically_increasing_interval_l398_398497


namespace minimum_area_of_square_on_parabola_l398_398881

theorem minimum_area_of_square_on_parabola :
  ∃ (A B C : ℝ × ℝ), 
  (∃ (x₁ x₂ x₃ : ℝ), (A = (x₁, x₁^2)) ∧ (B = (x₂, x₂^2)) ∧ (C = (x₃, x₃^2)) 
  ∧ x₁ < x₂ ∧ x₂ < x₃ 
  ∧ ∀ S : ℝ, (S = (1 + (x₃ + x₂)^2) * ((x₂ - x₃) - (x₃ - x₂))^2) → S ≥ 2) :=
sorry

end minimum_area_of_square_on_parabola_l398_398881


namespace david_twice_as_old_in_Y_years_l398_398900

variable (R D Y : ℕ)

-- Conditions
def rosy_current_age := R = 8
def david_is_older := D = R + 12
def twice_as_old_in_Y_years := D + Y = 2 * (R + Y)

-- Proof statement
theorem david_twice_as_old_in_Y_years
  (h1 : rosy_current_age R)
  (h2 : david_is_older R D)
  (h3 : twice_as_old_in_Y_years R D Y) :
  Y = 4 := sorry

end david_twice_as_old_in_Y_years_l398_398900


namespace correct_number_of_statements_is_2_l398_398084

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.sin (2 * x) - Real.sqrt 3 * Real.sin x ^ 2

def statement_1 := ∀ x, f (-π / 6 + x) = f (π / 2 - x)

def statement_2 := ∀ x, f (x + π) = f x

def statement_3 := ∀ x, x ∈ Set.Icc (π / 12) (7 * π / 12) → f x ≤ f (x + 1)

theorem correct_number_of_statements_is_2 :
  ¬statement_1 ∧ statement_2 ∧ statement_3 := by
  sorry

end correct_number_of_statements_is_2_l398_398084


namespace ratio_steel_iron_is_5_to_2_l398_398485

-- Definitions based on the given conditions
def amount_steel : ℕ := 35
def amount_iron : ℕ := 14

-- Main statement
theorem ratio_steel_iron_is_5_to_2 :
  (amount_steel / Nat.gcd amount_steel amount_iron) = 5 ∧
  (amount_iron / Nat.gcd amount_steel amount_iron) = 2 :=
by
  sorry

end ratio_steel_iron_is_5_to_2_l398_398485


namespace pen_price_l398_398886

theorem pen_price (x y : ℝ) (h1 : 2 * x + 3 * y = 49) (h2 : 3 * x + y = 49) : x = 14 :=
by
  -- Proof required here
  sorry

end pen_price_l398_398886


namespace equation_of_line1_equation_of_line2_l398_398564

-- Definitions for the circle and points
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9
def point_P := (2, 2)
def center_C := (1, 0)

-- Condition 1: when line l passes through the center of circle C
def line1 (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Condition 2: when chord AB is bisected by point P
def line2 (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- Proof problem for the first condition
theorem equation_of_line1 (x y : ℝ) (hP : point_P = (2, 2))
  (hC : center_C = (1, 0)) (hl : line1 2 2): circle x y → line1 x y :=
  by
    sorry

-- Proof problem for the second condition
theorem equation_of_line2 (x y : ℝ) (hP : point_P = (2, 2))
  (hl : line2 2 2): circle x y → line2 x y :=
  by
    sorry

end equation_of_line1_equation_of_line2_l398_398564


namespace number_of_factors_and_perfect_square_factors_l398_398610

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end number_of_factors_and_perfect_square_factors_l398_398610


namespace sequence_problem_l398_398100

def sequence (S_n : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  b 1 = 1 ∧ (∀ n, b (n + 1) = (1 / 3) * S_n n) ∧ (∀ n, S_n n = ∑ i in finset.range n, b (i + 1))

theorem sequence_problem (S_n : ℕ → ℝ) (b : ℕ → ℝ) (h1 : sequence S_n b) :
  (b 2 = 1 / 3 ∧ b 3 = 4 / 9 ∧ b 4 = 16 / 27) ∧
  (∀ n, b n = if n = 1 then 1 else 1 / 3 * ( 4 / 3 )^(n - 2)) ∧
  (∀ n, finset.sum (finset.range n) (λ i, b (2 * (i + 1))) = 3 / 7 * ( ( 4 / 3 )^(2 * n) - 1 )) :=
sorry

end sequence_problem_l398_398100


namespace event_B_more_likely_than_event_A_l398_398705

-- Definitions based on given conditions
def total_possible_outcomes := 6^3
def favorable_outcomes_B := (Nat.choose 6 3) * (Nat.factorial 3)
def prob_B := favorable_outcomes_B / total_possible_outcomes
def prob_A := 1 - prob_B

-- The theorem to be proved:
theorem event_B_more_likely_than_event_A (total_possible_outcomes = 216) 
    (favorable_outcomes_B = 120) 
    (prob_B = 5 / 9) 
    (prob_A = 4 / 9) :
    prob_B > prob_A := 
by {
    sorry
}

end event_B_more_likely_than_event_A_l398_398705


namespace even_and_decreasing_l398_398879

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_and_decreasing (f : ℝ → ℝ) (a b : ℝ) :
  (f = (λ x, x ^ (-2))) → (is_even f) → (is_monotonically_decreasing f a b) → f = (λ x, x ^ (-2)) := 
begin
  intros hf he hd,
  exact hf,
end

end even_and_decreasing_l398_398879


namespace min_rooms_needed_l398_398028

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398028


namespace two_cards_sum_to_15_proof_l398_398382

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398382


namespace find_k_and_formula_prove_Tn_l398_398099

-- Statement (1)
theorem find_k_and_formula (k : ℤ) (a : ℕ → ℤ) (S : ℕ → ℤ) (hS : ∀ n, S n = n^2 + k * n) :
  (a 1, a 4, a 13) form a geometric sequence ∧ (∀ n, a n = 2 * n + k - 1) → k = 2 ∧ (∀ n, a n = 2 * n - 1) := sorry

-- Statement (2)
theorem prove_Tn (a : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)
    (hS : ∀ n, a n = 2 * n - 1)
    (hb : ∀ n, b n = 4 / ((a n + 1) * (a (n+1) + 3)))
    (hT : ∀ n, T n = ∑ i in range n, b i) :
  T n < 5 / 12 := sorry

end find_k_and_formula_prove_Tn_l398_398099


namespace pythagorean_triple_example_l398_398483

noncomputable def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  is_pythagorean_triple 5 12 13 :=
by
  sorry

end pythagorean_triple_example_l398_398483


namespace sum_of_angles_l398_398525

theorem sum_of_angles (a b : ℝ) (ha : a = 45) (hb : b = 225) : a + b = 270 :=
by
  rw [ha, hb]
  norm_num -- Lean's built-in tactic to normalize numerical expressions

end sum_of_angles_l398_398525


namespace solve_for_x_and_y_l398_398205

theorem solve_for_x_and_y (x y : ℝ) 
  (h1 : 0.75 / x = 7 / 8)
  (h2 : x / y = 5 / 6) :
  x = 6 / 7 ∧ y = (6 / 7 * 6) / 5 :=
by
  sorry

end solve_for_x_and_y_l398_398205


namespace length_XZ_l398_398248

-- Definitions based on conditions
def Triangle (A B C : Type) := A ≠ B ∧ B ≠ C ∧ A ≠ C

variables (X Y Z : Type) [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Angle properties
def right_angle := 90 -- degrees

-- Given conditions
def angle_X_right : Prop := right_angle = 90
def hypotenuse_YZ : ℝ := 15
def tan_Z_eq_3sin_Z (XY Z : ℝ) : Prop := tan Z = 3 * sin Z

-- The proof statement (without the proof itself)
theorem length_XZ (XY XZ YZ : ℝ) (hYZX : hypotenuse_YZ = 15) (hangleX : angle_X_right) (htanZ : tan_Z_eq_3sin_Z XY Z) : XZ = 5 :=
sorry

end length_XZ_l398_398248


namespace problem_correct_options_l398_398354

open Finset

theorem problem_correct_options :
  ∃ (n m k l : ℕ), n ≠ 24 ∧ m = 18 ∧ k = 144 ∧ l = 9 :=
  let A := 4^4, -- This is the number of ways for option A, which is incorrect
      B := choose 4 2 * (choose 2 1 * factorial 2 + 1), -- Number of ways for option B
      C := choose 4 1 * (choose 3 1 * choose 2 2 * factorial 3 / (1 * factorial 1)), -- Number of ways for option C
      D := 3 * 3; -- Number of ways for option D
  by
  exact ⟨A, B, C, D, by simp [A], by simp [B], by simp [C], by simp [D]⟩
  sorry

end problem_correct_options_l398_398354


namespace two_cards_sum_to_15_proof_l398_398385

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398385


namespace min_rooms_needed_l398_398026

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398026


namespace find_three_points_in_circle_l398_398295

theorem find_three_points_in_circle :
  ∀ (points : Fin 51 → ℝ × ℝ), ∃ (c : ℝ × ℝ), ∃ (r : ℝ), r = 1/7 ∧ ∃ (subset : Fin 51) (h : subset ⊆ points), subset.card ≥ 3 ∧ ∀ p ∈ subset, dist p c ≤ r :=
begin
  sorry
end

end find_three_points_in_circle_l398_398295


namespace sum_of_squares_of_odd_integers_l398_398301

theorem sum_of_squares_of_odd_integers (a b c : ℕ) 
  (ha1 : odd a) (hb1 : odd b) (hc1 : odd c) 
  (h_diff1 : a ≠ b) (h_diff2 : b ≠ c) (h_diff3 : c ≠ a) :
  ∃ x1 x2 x3 x4 x5 x6 : ℕ, 
  (a*a + b*b + c*c + a + b + c + 3) = (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5 + x6 * x6) :=
begin
  sorry
end

end sum_of_squares_of_odd_integers_l398_398301


namespace grandmother_cheapest_option_l398_398987

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end grandmother_cheapest_option_l398_398987


namespace sum_of_ai_bk_eq_n_N_plus_1_l398_398278

open Nat

theorem sum_of_ai_bk_eq_n_N_plus_1 {n : ℕ} (a : Fin n.succ → ℕ) (hn : (∀ i, 0 < a i))
    (N : ℕ) (hN : N = Finset.univ.sup (λ i, a i) )
    (b : ℕ → ℕ) (hb : ∀ k, b k = Finset.card {i ∈ Finset.univ | a i ≤ k}) :
    (Finset.univ.sum a + Finset.range (N+1).sum b = n * (N + 1)) :=
by
  sorry

end sum_of_ai_bk_eq_n_N_plus_1_l398_398278


namespace isosceles_right_triangle_area_l398_398768

-- Define the conditions as given in the problem statement
variables (h l : ℝ)
hypothesis (hypotenuse_rel : h = l * Real.sqrt 2)
hypothesis (hypotenuse_val : h = 6 * Real.sqrt 2)

-- Define the formula for the area of an isosceles right triangle
def area_of_isosceles_right_triangle (l : ℝ) : ℝ := (1 / 2) * l * l

-- Define the proof problem statement
theorem isosceles_right_triangle_area : 
  area_of_isosceles_right_triangle l = 18 :=
  sorry

end isosceles_right_triangle_area_l398_398768


namespace variance_scale_l398_398935

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def variance (s : Finset α) : ℝ :=
 sorry -- definition of variance

theorem variance_scale (s : Finset ℝ) (k : ℝ) (v : ℝ) (h : variance s = v) :
  variance (s.map (fun x => k • x)) = k^2 * v :=
 sorry

end variance_scale_l398_398935


namespace probability_two_cards_sum_to_15_l398_398401

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398401


namespace expression_undefined_at_x_l398_398519

theorem expression_undefined_at_x (x : ℝ) : (x^2 - 18 * x + 81 = 0) → x = 9 :=
by {
  sorry
}

end expression_undefined_at_x_l398_398519


namespace negated_roots_quadratic_reciprocals_roots_quadratic_l398_398717

-- For (1)
theorem negated_roots_quadratic (x y : ℝ) : 
    (x^2 + 3 * x - 2 = 0) ↔ (y^2 - 3 * y - 2 = 0) :=
sorry

-- For (2)
theorem reciprocals_roots_quadratic (a b c x y : ℝ) (h : a ≠ 0) :
    (a * x^2 - b * x + c = 0) ↔ (c * y^2 - b * y + a = 0) :=
sorry

end negated_roots_quadratic_reciprocals_roots_quadratic_l398_398717


namespace coefficient_a5_l398_398621

theorem coefficient_a5 (x : ℝ) (a : Fin 11 → ℝ) (h : x^10 - x^5 = ∑ i in Finset.range 11, a i * (x - 1)^i) : a 5 = 251 :=
sorry

end coefficient_a5_l398_398621


namespace distribute_7_balls_into_4_boxes_l398_398617

-- Define the problem conditions
def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  if balls < boxes then 0 else Nat.choose (balls - 1) (boxes - 1)

-- Prove the specific case
theorem distribute_7_balls_into_4_boxes : number_of_ways_to_distribute_balls 7 4 = 20 :=
by
  -- Definition and proof to be filled
  sorry

end distribute_7_balls_into_4_boxes_l398_398617


namespace range_of_function_l398_398507

theorem range_of_function : 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 12) ∧ 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 18) ∧ 
  (∀ y : ℝ, (12 ≤ y ∧ y ≤ 18) → 
    ∃ x : ℝ, y = |x + 5| - |x - 3| + 4) :=
by
  sorry

end range_of_function_l398_398507


namespace trousers_cost_l398_398741

-- Definitions for the conditions
def total_amount : ℝ := 260
def cost_per_shirt : ℝ := 18.50
def num_shirts : ℝ := 2
def cost_per_additional_article : ℝ := 40
def num_additional_articles : ℝ := 4

-- Proving the cost of the pair of trousers
theorem trousers_cost :
  let total_shirts_cost := num_shirts * cost_per_shirt,
      remaining_after_shirts := total_amount - total_shirts_cost,
      total_additional_articles_cost := num_additional_articles * cost_per_additional_article,
      remaining_after_additional := remaining_after_shirts - total_additional_articles_cost in
  remaining_after_additional = 63 := by
  sorry

end trousers_cost_l398_398741


namespace win_probability_comparison_l398_398412

theorem win_probability_comparison :
  (4 * 256).inv > (56 * 16).inv :=
by sorry

end win_probability_comparison_l398_398412


namespace bus_time_one_way_l398_398254

-- define conditions
def walk_time_one_way := 5 -- 5 minutes for one walk
def total_annual_travel_time_hours := 365 -- 365 hours per year
def work_days_per_year := 365 -- works every day

-- convert annual travel time from hours to minutes
def total_annual_travel_time_minutes := total_annual_travel_time_hours * 60

-- calculate total daily travel time
def total_daily_travel_time := total_annual_travel_time_minutes / work_days_per_year

-- walking time per day
def total_daily_walking_time := (walk_time_one_way * 4)

-- total bus travel time per day
def total_daily_bus_time := total_daily_travel_time - total_daily_walking_time

-- one-way bus time
theorem bus_time_one_way : total_daily_bus_time / 2 = 20 := by
  sorry

end bus_time_one_way_l398_398254


namespace zero_of_f_when_a_half_no_extreme_value_when_a_ge_half_l398_398128

noncomputable def f (x a : ℝ) : ℝ := (x + 2) * Real.log x + a * x^2 - 4 * x + 7 * a

theorem zero_of_f_when_a_half :
  ∀ x : ℝ, f x (1/2) = 0 ↔ x = 1 :=
sorry

theorem no_extreme_value_when_a_ge_half :
  ∀ (a : ℝ), (1 / 2) ≤ a → ∀ x : ℝ, x > 0 → (∀ y, f' y ≥ 0) :=
sorry

end zero_of_f_when_a_half_no_extreme_value_when_a_ge_half_l398_398128


namespace find_x_l398_398980

variable (x : ℝ) 
def a : ℝ × ℝ × ℝ := (2 * x, 1, -2)
def b : ℝ × ℝ × ℝ := (x + 1, 3, -x)
def dot_product (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_x (h : dot_product (a x) (b x) = 3) : x = 0 ∨ x = -2 :=
by { sorry }

end find_x_l398_398980


namespace Samantha_last_name_length_l398_398305

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end Samantha_last_name_length_l398_398305


namespace B_initial_investment_l398_398841

variable (x : ℝ) -- B's initial investment

-- A's and B's initial investments
def A_init := 3000
def B_init := x

-- Investments after 8 months
def A_after_8_months := 2000
def B_after_8_months := x + 1000

-- Total investments over the year
def A_total := 8 * A_init + 4 * A_after_8_months
def B_total := 8 * B_init + 4 * (B_after_8_months)

-- Profits ratio
def total_profit := 840
def A_profit := 320
def B_profit := total_profit - A_profit

def investment_ratio := A_total / B_total
def profit_ratio := A_profit / B_profit

theorem B_initial_investment : B_init = 4000 :=
by
  sorry

end B_initial_investment_l398_398841


namespace ellipse_directrix_distance_l398_398106

noncomputable def distance_to_directrix (P : ℝ × ℝ) (F1 : ℝ × ℝ) (a b c : ℝ) (e : ℝ) (dist_P_F1 : ℝ) : ℝ := 
  dist_P_F1 / e

theorem ellipse_directrix_distance
  {P : ℝ × ℝ}
  {a b : ℝ}
  (ha : a = 2)
  (hb : b = 3√1.5)
  {e : ℝ}
  (he : e = c / a)
  {F1 : ℝ × ℝ}
  (hF1x : F1.1 = -c)
  (hF1y : F1.2 = 0)
  {dist_P_F1 : ℝ}
  (hdist_P_F1 : dist_P_F1 = 5 / 2) :
  distance_to_directrix P F1 2 3√1.5 1 / 2 (5 / 2) = 5 := 
by {
  -- Ellipse equation
  let ellipse_eq := ¬s.hole, ⟨a, b⟩, (ax : pow (P.1 / a) 2 + pow (P.2 / b) 1 = 1)
    = ⟩
  -- Distance calculation to focus
  drdeliveryproofitapplie(suswit)
elliptixissimpodist  -5 d signhafialsorry fo risveryidiom i.e. :

ellipse_eq :=dsfssimpoproof or arrivehxoxsorry  attritype is impattributesubtests⊕⟨ holea ,b,  x : pow (P.1 / a) is defective with error2 +nd pow ⟨ (F1apparitionedefect.Pith .2 / b) 1 judgmentmatrix
    ⟩ equalfirstdigit1 

end ellipse_directrix_distance_l398_398106


namespace second_derivative_parametrically_l398_398831

-- Definitions of the parametric equations
def x (t : ℝ) : ℝ := Real.cosh t
def y (t : ℝ) : ℝ := Real.sinh t ^ 2 / 3

-- Define the first derivatives
def x_t' (t : ℝ) : ℝ := Real.sinh t
def y_t' (t : ℝ) : ℝ := (2 / 3) * Real.cosh t / (Real.sinh t) ^ (1 / 3)

-- Compute y_x'
def y_x' (t : ℝ) : ℝ := y_t' t / x_t' t

-- Differentiate y_x' with respect to t
def y_x'_t' (t : ℝ) : ℝ := 
    let u := Real.sinh t
    let v := Real.cosh t
    -(2 / 9) * (3 + v^2) / u^3

-- Finally compute the second derivative
def y_xx'' (t : ℝ) : ℝ := y_x'_t' t / x_t' t

-- The theorem statement
theorem second_derivative_parametrically (t : ℝ) : y_xx'' t = -(2 * (3 + Real.cosh t ^ 2)) / (9 * Real.sinh t ^ 4) :=
by sorry

end second_derivative_parametrically_l398_398831


namespace reduced_price_is_25_l398_398862

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price (P : ℝ) := P * 0.85
noncomputable def amount_of_wheat_original (P : ℝ) := 500 / P
noncomputable def amount_of_wheat_reduced (P : ℝ) := 500 / (P * 0.85)

theorem reduced_price_is_25 : 
  ∃ (P : ℝ), reduced_price P = 25 ∧ (amount_of_wheat_reduced P = amount_of_wheat_original P + 3) :=
sorry

end reduced_price_is_25_l398_398862


namespace variance_scale_l398_398934

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def variance (s : Finset α) : ℝ :=
 sorry -- definition of variance

theorem variance_scale (s : Finset ℝ) (k : ℝ) (v : ℝ) (h : variance s = v) :
  variance (s.map (fun x => k • x)) = k^2 * v :=
 sorry

end variance_scale_l398_398934


namespace volume_of_prism_l398_398562

noncomputable def volume_of_triangular_prism
  (area_lateral_face : ℝ)
  (distance_cc1_to_lateral_face : ℝ) : ℝ :=
  area_lateral_face * distance_cc1_to_lateral_face

theorem volume_of_prism (area_lateral_face : ℝ) 
    (distance_cc1_to_lateral_face : ℝ)
    (h_area : area_lateral_face = 4)
    (h_distance : distance_cc1_to_lateral_face = 2):
  volume_of_triangular_prism area_lateral_face distance_cc1_to_lateral_face = 4 := by
  sorry

end volume_of_prism_l398_398562


namespace sin_double_angle_value_l398_398112

open Real

theorem sin_double_angle_value (x : ℝ) (h : sin (x + π / 4) = - 5 / 13) : sin (2 * x) = - 119 / 169 := 
sorry

end sin_double_angle_value_l398_398112


namespace cosine_identity_geometric_sequence_l398_398633

theorem cosine_identity_geometric_sequence 
  {A B C : ℝ}
  {a b c : ℝ}
  (h_geometric : b^2 = a * c) 
  (h_sin : sin B ^ 2 = sin A * sin C) :
  cos (A - C) + cos B + cos (2 * B) = 1 :=
sorry

end cosine_identity_geometric_sequence_l398_398633


namespace probability_prime_or_odd_l398_398905

-- Definition of a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Set of numbers from 1 to 8
def balls := {1, 2, 3, 4, 5, 6, 7, 8}

-- Predicate for the number being odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Predicate for the number being either prime or odd
def is_prime_or_odd (n : ℕ) : Prop := is_prime n ∨ is_odd n

-- Total number of favorable outcomes
def num_favorable_outcomes : ℕ := {n | n ∈ balls ∧ is_prime_or_odd n}.card

-- Total number of outcomes
def total_outcomes : ℕ := balls.card

-- The probability of drawing a ball that is either prime or odd
def probability : ℚ := num_favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem probability_prime_or_odd : probability = 5 / 8 :=
by
  sorry

end probability_prime_or_odd_l398_398905


namespace cosine_between_lines_l398_398246

noncomputable def A : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def A1 : ℝ × ℝ × ℝ := (0, 0, 2)
noncomputable def B1 : ℝ × ℝ × ℝ := (2, 0, 2)
noncomputable def C : ℝ × ℝ × ℝ := (0, 1, 0)

noncomputable def vector_AB1 := (B1.1 - A.1, B1.2 - A.2, B1.3 - A.3)
noncomputable def vector_A1C := (C.1 - A1.1, C.2 - A1.2, C.3 - A1.3)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cos_theta := (dot_product vector_AB1 vector_A1C) / (magnitude vector_AB1 * magnitude vector_A1C)

theorem cosine_between_lines : cos_theta = (real.sqrt 10) / 5 := by
  sorry

end cosine_between_lines_l398_398246


namespace circumcircle_incircle_inequality_l398_398812

theorem circumcircle_incircle_inequality
  (a b : ℝ)
  (h_a : a = 16)
  (h_b : b = 11)
  (R r : ℝ)
  (triangle_inequality : ∀ c : ℝ, 5 < c ∧ c < 27) :
  R ≥ 2.2 * r := sorry

end circumcircle_incircle_inequality_l398_398812


namespace find_hyperbola_equation_l398_398131

noncomputable def hyperbola_equation (a b : ℝ) := (x^2 / a^2 - y^2 / b^2 = 1)

theorem find_hyperbola_equation
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (asym1 : ∀ x y, y = (sqrt 3 / 3) * x)
  (vertex_dist : ∀ (a : ℝ), | sqrt 3 * a / sqrt 12 | = 1) :
  hyperbola_equation 2 ((2 * sqrt 3) / 3) :=
by
  sorry

end find_hyperbola_equation_l398_398131


namespace range_of_a_l398_398682

noncomputable def real_set := Set ℝ

def A : real_set := {x | 4 ≤ x ∧ x < 5}
def B (a : ℝ) : real_set := {x | a < x ∧ x ≤ 2 * a - 1}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 3 ≤ a ∧ a < 4 := by
  sorry

end range_of_a_l398_398682


namespace reflection_matrix_cube_eq_self_l398_398675

variable {F : Type} [Field F]
def reflection_matrix : Matrix (Fin 2) (Fin 2) F :=
  let v := ![1, 1] in
  2 • (v ⬝ vᵀ) / (v ⬝ᵥ v) - 1

theorem reflection_matrix_cube_eq_self (S : Matrix (Fin 2) (Fin 2) F) (hS : S = reflection_matrix) :
  S^3 = S :=
  sorry

end reflection_matrix_cube_eq_self_l398_398675


namespace num_undefined_values_l398_398083

-- Condition: Denominator is given as (x^2 + 2x - 3)(x - 3)(x + 1)
def denominator (x : ℝ) : ℝ := (x^2 + 2 * x - 3) * (x - 3) * (x + 1)

-- The Lean statement to prove the number of values of x for which the expression is undefined
theorem num_undefined_values : 
  ∃ (n : ℕ), (∀ x : ℝ, denominator x = 0 → (x = 1 ∨ x = -3 ∨ x = 3 ∨ x = -1)) ∧ n = 4 :=
by
  sorry

end num_undefined_values_l398_398083


namespace arithmetic_mean_of_smallest_twin_prime_pair_l398_398470

open Nat

/-- Definition of twin prime pair -/
def is_twin_prime_pair (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ q = p + 2

/-- The smallest twin prime pair is (3, 5) and the arithmetic mean of 3 and 5 is 4. -/
theorem arithmetic_mean_of_smallest_twin_prime_pair :
  ∃ (p q : ℕ), is_twin_prime_pair p q ∧ p = 3 ∧ q = 5 ∧ (p + q) / 2 = 4 :=
by
  sorry

end arithmetic_mean_of_smallest_twin_prime_pair_l398_398470


namespace find_solutions_equation_l398_398039

theorem find_solutions_equation :
  {x : ℝ | 1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 11 * x - 12) = 0}
  = {1, -12, 4, -3} :=
by
  sorry

end find_solutions_equation_l398_398039


namespace simon_number_of_legos_l398_398732

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l398_398732


namespace ellipse_semi_minor_axis_l398_398583

theorem ellipse_semi_minor_axis (b : ℝ) 
    (h1 : 0 < b) 
    (h2 : b < 5)
    (h_ellipse : ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1) 
    (h_eccentricity : 4 / 5 = 4 / 5) : b = 3 := 
sorry

end ellipse_semi_minor_axis_l398_398583


namespace trig_expression_value_l398_398146

theorem trig_expression_value
  (x : ℝ)
  (h : Real.tan (x + Real.pi / 4) = -3) :
  (Real.sin x + 2 * Real.cos x) / (3 * Real.sin x + 4 * Real.cos x) = 2 / 5 :=
by
  sorry

end trig_expression_value_l398_398146


namespace jack_and_jill_meet_distance_l398_398255

theorem jack_and_jill_meet_distance :
  ∀ (time : ℝ) (jack_starting_time : ℝ) (jill_rest_time : ℝ),
  (jack_speed_uphill jill_speed_uphill jack_speed_downhill jill_speed_downhill distance_uphill : ℝ),
  jack_starting_time = 1/4 ∧ 
  jill_rest_time = 1/12 ∧ 
  jack_speed_uphill = 14 ∧ 
  jill_speed_uphill = 15 ∧ 
  jack_speed_downhill = 18 ∧ 
  jill_speed_downhill = 21 ∧ 
  distance_uphill = 6 →
  let x := (129:ℝ)/28 in
  let y := 15 * (x - (1:ℝ)/4) in
  distance_top := 6 - y,
  distance_top = 327/32 :=
begin
  sorry
end

end jack_and_jill_meet_distance_l398_398255


namespace probability_of_contact_l398_398056

noncomputable def probability_connection (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

theorem probability_of_contact (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let group1 := 5
  let group2 := 8
  let total_pairs := group1 * group2
  (total_pairs = 40) →
  (∀ i j, i ∈ fin group1 → j ∈ fin group2 → (¬ p = 1 → p = p)) → 
  probability_connection p = 1 - (1 - p) ^ 40 :=
by
  intros _ _ 
  sorry

end probability_of_contact_l398_398056


namespace geometric_sequence_problem_l398_398240

variable (G : Type) [Field G]

noncomputable def geom_sequence (a1 a2 q : G) (n : ℕ) : G :=
  a1 * q ^ (n - 1) + a2 * q ^ (n - 2)

theorem geometric_sequence_problem (a1 a2 q : G) (h1 : a1 + a2 = 30) (h2 : geom_sequence a1 a2 q 3 + geom_sequence a1 a2 q 4 = 120) :
  geom_sequence a1 a2 q 5 + geom_sequence a1 a2 q 6 = 480 := by
  sorry

end geometric_sequence_problem_l398_398240


namespace prob_at_least_two_correct_guesses_correct_expected_score_correct_l398_398646

section GuessSongGame

variables (A_correct : ℝ)
variables (B_correct : ℝ)

-- Conditions
/-- The probability of the guest correctly guessing the title of each song in group A is 2/3 --/
def prob_A_guess_correct : ℝ := 2 / 3

/-- The probability of the guest correctly guessing the title of each song in group B is 1/2 --/
def prob_B_guess_correct : ℝ := 1 / 2

/-- The guest earns 1 point for each correct guess from group A --/
def points_per_A : ℝ := 1

/-- The guest earns 2 points for each correct guess from group B --/
def points_per_B : ℝ := 2

/-- The guest plays 2 melodies from each group A and B --/
noncomputable def total_melodies_A : ℕ := 2
noncomputable def total_melodies_B : ℕ := 2

/-- The probability that the guest guesses at least 2 song titles correctly --/
def prob_at_least_two_correct_guesses : ℝ :=
  1 - ((1 - prob_A_guess_correct)^2 * (1 - prob_B_guess_correct)^2) -
      (2 * (1 - prob_A_guess_correct) * prob_A_guess_correct * (prob_B_guess_correct)^2 + 
      ((1 - prob_A_guess_correct)^2 * 2 * prob_B_guess_correct * (1 - prob_B_guess_correct)))

theorem prob_at_least_two_correct_guesses_correct :
  prob_at_least_two_correct_guesses = 29 / 36 := by sorry

/-- The expected score of the guest --/
def expected_score : ℝ :=
  0 * (1 / 36) + 
  1 * (1 / 9) + 
  2 * (1 / 6) + 
  3 * (2 / 9) + 
  4 * (1 / 4) + 
  5 * (1 / 9) + 
  6 * (1 / 9)

theorem expected_score_correct : 
  expected_score = 10 / 3 := by sorry

end GuessSongGame

end prob_at_least_two_correct_guesses_correct_expected_score_correct_l398_398646


namespace area_ratio_l398_398250

variables {A B C D E M N : Type}

-- Assume we have points representing the vertices of the triangle and other significant points
variable [AffineSpace ℝ (Triangle A B C)]
variable [AffineSpace ℝ (Point D M E)]
variable [Midpoint (Point A E) (Point N)]

def k (area_ABC : ℝ) (area_MNE : ℝ) : ℝ :=
  area_MNE / area_ABC

theorem area_ratio (area_ABC area_MNE : ℝ)
  (h1 : is_centroid A B C D E M)
  (h2 : is_midpoint A E N)
  (h3 : area_MNE = (k area_ABC area_MNE) * area_ABC):
  k area_ABC area_MNE = 1 / 8 :=
begin
  sorry
end

end area_ratio_l398_398250


namespace fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l398_398290

-- Define the sequences as functions
def row1 (n : ℕ) : ℤ := (-2)^n
def row2 (n : ℕ) : ℤ := row1 n + 2
def row3 (n : ℕ) : ℤ := (-1) * (-2)^n

-- Theorems to be proven

-- (1) Prove the fourth term in row ① is 16 
theorem fourth_term_row1_is_16 : row1 4 = 16 := sorry

-- (1) Prove the nth term in row ① is (-2)^n
theorem nth_term_row1 (n : ℕ) : row1 n = (-2)^n := sorry

-- (2) Let the nth number in row ① be a, prove the nth number in row ② is a + 2
theorem nth_term_row2 (n : ℕ) : row2 n = row1 n + 2 := sorry

-- (3) If the sum of three consecutive numbers in row ③ is -192, find these numbers
theorem sum_three_consecutive_row3 : ∃ n : ℕ, row3 n + row3 (n + 1) + row3 (n + 2) = -192 ∧ 
  row3 n  = -64 ∧ row3 (n + 1) = 128 ∧ row3 (n + 2) = -256 := sorry

end fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l398_398290


namespace B_C_investment_ratio_l398_398472

variable (A B C : ℝ)
variable (total_profit profit_B : ℝ)
variable (investment_ratio : ℝ)

def investments (B_investment : ℝ) (C_investment : ℝ) : Prop :=
  A = 3 * B ∧
  B = investment_ratio * C ∧
  total_profit = 3300 ∧
  profit_B = 600 ∧
  profit_B / total_profit = B_investment / (A + B_investment + C_investment)

theorem B_C_investment_ratio (B_investment C_investment : ℝ) :
  investments A B C B_investment C_investment →
  B_investment / C_investment = 2 / 3 := 
by 
  sorry

end B_C_investment_ratio_l398_398472


namespace probability_Abby_Bridget_adjacent_l398_398473

theorem probability_Abby_Bridget_adjacent :
  let students := ["Abby", "Bridget"] ++ List.replicate 5 "classmate"
  let seats := List.range 8
  let total_ways := (Finset.univ : Finset (students.Perm)).card
  let favorable_ways :=
    (Finset.filter
      (λ perm : students.Perm,
          let seating := perm.to_list.take 8
          (∀ (i j : Fin 8), (seating.nth i = some "Abby" ∧ seating.nth j = some "Bridget" ∧
            ((i / 4 = j / 4 ∧ (i % 4 = j % 4 + 1 ∨ i % 4 + 1 = j % 4)) ∨ (i % 4 = j % 4 ∧ (i / 4 = j / 4 + 1 ∨ i / 4 + 1 = j / 4)))))
      ) Finset.univ
    ).card
  in favorable_ways / total_ways = 1 / 17 := sorry

end probability_Abby_Bridget_adjacent_l398_398473


namespace C1_polar_eq_area_MAB_l398_398648

-- Definitions
def C1_parametric (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)
def C2_polar (θ : ℝ) : ℝ := 4 * Real.cos θ
def ray_θ := Real.pi / 3
def point_M := (2 : ℝ, 0 : ℝ)

-- Proving the equivalences

-- Statement 1: Polar equation of C1
theorem C1_polar_eq (θ : ℝ) : (let (x, y) := C1_parametric θ in Real.sqrt (x^2 + y^2) = 4 * Real.sin θ) :=
sorry

-- Statement 2: Area of Δ MAB
theorem area_MAB : 
  let A := (4 * Real.sin ray_θ, ray_θ)
  let B := (4 * Real.cos ray_θ, ray_θ)
  let d := 2 * Real.sin ray_θ
  let AB := (B.1 - A.1)
  let S := (1/2 : ℝ) * AB * d
  S = 3 - Real.sqrt 3 :=
sorry

end C1_polar_eq_area_MAB_l398_398648


namespace max_unique_dance_counts_l398_398434

theorem max_unique_dance_counts (boys girls : ℕ) (positive_boys : boys = 29) (positive_girls : girls = 15) 
  (dances : ∀ b g, b ≤ boys → g ≤ girls → ℕ) :
  ∃ num_dances, num_dances = 29 := 
by
  sorry

end max_unique_dance_counts_l398_398434


namespace variance_scaled_data_l398_398938

-- Definition of variance
def variance (data : List ℝ) : ℝ :=
  let mean := (data.foldl (+) 0) / (data.length : ℝ)
  (data.map (λ x => (x - mean)^2)).foldl (+) 0 / (data.length : ℝ)

-- Original data set and its variance
def original_data := [x_1, x_2, ..., x_n]
def original_variance := 0.01

-- Prove that the variance of 10 times each data point is 1
theorem variance_scaled_data :
  variance (original_data.map (λ x => 10 * x)) = 1 := by
  sorry

end variance_scaled_data_l398_398938


namespace probability_two_cards_sum_to_15_l398_398402

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398402


namespace measure_angle_C_l398_398632

noncomputable def triangle_ABC (A B C D E H : Type) :=
  {a b c : A → ℝ // a + b + c = π}

variables {A B C D E H : Type}
variables [triangle_ABC A B C D E H]

variables [angle A = 70] {p : D ∈ segment A C}
variables (ae_bisects_angle : bisects AE angle A) 
variables (H_intersects : intersects BD AE H)
variables (ratio1 : ratio AH 3 HE 1)
variables (ratio2 : ratio BH 5 HD 3)

theorem measure_angle_C (A B C D E H : Type) [t : triangle_ABC A B C D E H]
  (angle_A : angle A = 70)
  (on_side_AC : D ∈ segment A C)
  (bisector_AE : bisects AE angle A)
  (intersection_H : intersects BD AE H)
  (ratio_AH_HE : ratio AH 3 HE 1)
  (ratio_BH_HD : ratio BH 5 HD 3)
  : angle C = 55 :=
  by
  sorry

end measure_angle_C_l398_398632


namespace find_a1_l398_398666

-- Definitions stemming from the conditions in the problem
def arithmetic_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

def is_geometric (a₁ a₃ a₆ : ℕ) : Prop :=
  ∃ r : ℕ, a₃ = r * a₁ ∧ a₆ = r^2 * a₁

theorem find_a1 :
  ∀ a₁ : ℕ,
    (arithmetic_seq a₁ 3 1 = a₁) ∧
    (arithmetic_seq a₁ 3 3 = a₁ + 6) ∧
    (arithmetic_seq a₁ 3 6 = a₁ + 15) ∧
    is_geometric a₁ (a₁ + 6) (a₁ + 15) →
    a₁ = 12 :=
by
  intros
  sorry

end find_a1_l398_398666


namespace evaluate_expression_l398_398261

noncomputable def a := Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def b := -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def d := -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 3 / 50 :=
by
  sorry

end evaluate_expression_l398_398261


namespace Homer_first_try_points_l398_398994

variable (x : ℕ)
variable (h1 : x + (x - 70) + 2 * (x - 70) = 1390)

theorem Homer_first_try_points : x = 400 := by
  sorry

end Homer_first_try_points_l398_398994


namespace f_eq_n_squared_minus_one_l398_398541

open Function

def f : ℕ → ℤ
| 1 := 0
| 2 := 3
| 3 := 8
| 4 := 15
| _ := sorry  -- generalized case not explicitly defined, since we are proving this based on given values

theorem f_eq_n_squared_minus_one :
  (f 1 = 0) →
  (f 2 = 3) →
  (f 3 = 8) →
  (f 4 = 15) →
  ∀ n, f n = n^2 - 1 :=
by
  intros h1 h2 h3 h4 n
  sorry

end f_eq_n_squared_minus_one_l398_398541


namespace expected_min_tau_n_n_limit_l398_398281

noncomputable def tau_n (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if h : ∃ k ∈ (finset.Icc 1 n), S k = 1 then finset.min' {k | k ∈ finset.Icc 1 n ∧ S k = 1} h else ⊤

noncomputable def min_tau_n_n (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  min (tau_n S n) n

theorem expected_min_tau_n_n_limit (S : ℕ → ℝ) (p q : ℝ) (hpq : p ≠ q):
  (λ n, E (min_tau_n_n S n)) ⟶ (if p > q then (p - q)⁻¹ else ⊤)ₙ :=
sorry

end expected_min_tau_n_n_limit_l398_398281


namespace fraction_sum_product_roots_of_quadratic_l398_398160

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398160


namespace english_test_question_count_l398_398294

theorem english_test_question_count (E : ℕ)
  (math_questions : ℕ := 40)
  (math_right_percentage : ℚ := 0.75)
  (english_right_percentage : ℚ := 0.98)
  (total_right : ℕ := 79)
  (math_right : ℕ := math_questions * math_right_percentage) :
  E * english_right_percentage = total_right - math_right → E = 50 :=
by
  intros h
  sorry

end english_test_question_count_l398_398294


namespace factor_and_divisor_statements_l398_398422

theorem factor_and_divisor_statements :
  (∃ n : ℕ, 25 = 5 * n) ∧
  ((∃ n : ℕ, 209 = 19 * n) ∧ ¬ (∃ n : ℕ, 63 = 19 * n)) ∧
  (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end factor_and_divisor_statements_l398_398422


namespace center_of_circle_l398_398753

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -3)) (h2 : (x2, y2) = (8, 9)) :
  midpoint (x1, y1) (x2, y2) = (5, 3) :=
by
  rw [h1, h2]
  simp [midpoint]
  sorry

end center_of_circle_l398_398753


namespace find_min_value_l398_398571

theorem find_min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) : 
  (1 / (2 * a)) + (2 / (b - 1)) ≥ 9 / 2 :=
by
  sorry

end find_min_value_l398_398571


namespace min_d_exists_l398_398718

theorem min_d_exists (a b c d : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ∃ d', d' = 503 ∧
  ∃ x1 x2 : ℚ, x1 ≠ x2 ∧
    3 * x1 + (abs (x1 - a) + abs (x1 - b) + abs (x1 - c) + abs (x1 - d')) = 3005 ∧
    3 * x2 + (abs (x2 - a) + abs (x2 - b) + abs (x2 - c) + abs (x2 - d')) = 3005 :=
begin
  sorry
end

end min_d_exists_l398_398718


namespace mean_of_α_X_l398_398082

open Finset

def M : Finset ℕ := range 1000 |>.map (λ x => x + 1)

def α (X : Finset ℕ) : ℕ :=
if X.nonempty then X.max' (X.nonempty) + X.min' (X.nonempty) else 0

def N : ℕ := ∑ X in M.powerset.filter (λ X, ¬X.isEmpty), α X

def f : ℚ :=
N.to_rat / (2^1000 - 1).to_rat

theorem mean_of_α_X :
  f = 1001 :=
sorry

end mean_of_α_X_l398_398082


namespace hexagon_coverage_percent_is_50_l398_398340

theorem hexagon_coverage_percent_is_50 :
  (∀ s : ℕ, let u := s ^ 2 in 
            let total_units := 16 * u in
            let hexagon_units := 8 * u in
            100 * hexagon_units / total_units = 50) :=
 by sorry
 
end hexagon_coverage_percent_is_50_l398_398340


namespace birds_flew_away_l398_398353

-- Define the initial and remaining birds
def original_birds : ℕ := 12
def remaining_birds : ℕ := 4

-- Define the number of birds that flew away
noncomputable def flew_away_birds : ℕ := original_birds - remaining_birds

-- State the theorem that the number of birds that flew away is 8
theorem birds_flew_away : flew_away_birds = 8 := by
  -- Lean expects a proof here. For now, we use sorry to indicate the proof is skipped.
  sorry

end birds_flew_away_l398_398353


namespace prob_sum_15_correct_l398_398376

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398376


namespace sum_roots_of_log_eqn_l398_398557

variable {a b : ℝ}

theorem sum_roots_of_log_eqn (h : (∀ x : ℝ, log (3*x) 3 + log 27 (3*x) = -4/3)) : a + b = 10/81 :=
sorry

end sum_roots_of_log_eqn_l398_398557


namespace intersection_of_sets_l398_398604

noncomputable def set_M : set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }
noncomputable def set_N : set ℝ := { y | ∃ x : ℝ, y = real.sqrt (3 - x^2) }

theorem intersection_of_sets :
  ∀ x : ℝ, x ∈ { x | (-1 : ℝ) ≤ x ∧ x ≤ real.sqrt 3 } ↔
    x ∈ { x | ∃ y : ℝ, y = x^2 - 1 } ∧ x ∈ { x | ∃ z : ℝ, z = real.sqrt (3 - x^2) } :=
sorry

end intersection_of_sets_l398_398604


namespace range_of_m_l398_398597

variable (a : ℝ) (h_a : a > 0)

def f (x : ℝ) : ℝ := Real.log (Real.exp x + a)

def f_inv (x : ℝ) : ℝ := Real.log (Real.exp x - a)

def f_prime (x : ℝ) : ℝ := (Real.exp x) / (Real.exp x + a)

theorem range_of_m (m : ℝ) :
  (∀ x, Real.log (3 * a) ≤ x ∧ x ≤ Real.log (4 * a) →
  |m - f_inv a x| + Real.log (f_prime a x) < 0) ↔
  Real.log (12 / 5 * a) < m ∧ m < Real.log (8 / 3 * a) :=
sorry

end range_of_m_l398_398597


namespace positive_difference_l398_398901

def sequence_A : ℕ :=
  ∑ i in finset.range 21, (2 * i + 2) * (2 * i + 3) + 42

def sequence_B : ℕ :=
  2 + ∑ i in finset.range 20, (2 * i + 3) * (2 * i + 4)

theorem positive_difference (A B : ℕ) (hA : A = sequence_A) (hB : B = sequence_B) : |A - B| = 800 :=
by
  sorry

end positive_difference_l398_398901


namespace sequence_sum_ge_neg_half_l398_398969

variable (n : ℕ) (a : ℕ → ℤ)

-- Conditions
def condition_1 : Prop := a 1 = 0
def condition_2 : Prop := ∀ i, 2 ≤ i → |a i| = |a (i - 1) + 1|

-- Theorem to prove
theorem sequence_sum_ge_neg_half {n : ℕ} (a : ℕ → ℤ) 
  (h1 : condition_1 a)
  (h2 : condition_2 n a) :
  (∑ i in Finset.range n, a (i + 1)) ≥ - (n / 2 : ℤ) :=
by
  sorry

end sequence_sum_ge_neg_half_l398_398969


namespace bacteria_initial_count_l398_398320

theorem bacteria_initial_count (t : ℕ) (final_count : ℕ) 
  (h1 : t = 180) 
  (h2 : final_count = 275562)
  (h3 : ∀ k : ℕ, ∀ n : ℕ, n * 3^k = final_count → k = 9) : 
  ∀ initial_count : ℕ, initial_count * 3^9 = final_count → initial_count = 14 :=
by 
  intros initial_count h_initial_count_eq
  have h: 3^9 = 19683 := by sorry
  rw [h, mul_assoc] at h_initial_count_eq
  linarith

end bacteria_initial_count_l398_398320


namespace evaluate_expression_l398_398814

theorem evaluate_expression :
  (3^4 + 3^4) / (3^(-4) + 3^(-4)) = 6561 :=
sorry

end evaluate_expression_l398_398814


namespace find_side_b_l398_398217

-- Given the side and angle conditions in the triangle
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ) 

-- Conditions provided in the problem
axiom side_a (h : a = 1) : True
axiom angle_B (h : B = Real.pi / 4) : True  -- 45 degrees in radians
axiom area_triangle (h : S = 2) : True

-- Final proof statement
theorem find_side_b (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : 
  b = 5 := sorry

end find_side_b_l398_398217


namespace prob_contact_l398_398059

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l398_398059


namespace arccos_one_half_eq_pi_over_three_l398_398499

theorem arccos_one_half_eq_pi_over_three : 
  ∀ x : ℝ, cos x = 1/2 → arccos (1/2) = x :=
by
  sorry

end arccos_one_half_eq_pi_over_three_l398_398499


namespace variance_scaled_data_l398_398936

-- Definition of variance
def variance (data : List ℝ) : ℝ :=
  let mean := (data.foldl (+) 0) / (data.length : ℝ)
  (data.map (λ x => (x - mean)^2)).foldl (+) 0 / (data.length : ℝ)

-- Original data set and its variance
def original_data := [x_1, x_2, ..., x_n]
def original_variance := 0.01

-- Prove that the variance of 10 times each data point is 1
theorem variance_scaled_data :
  variance (original_data.map (λ x => 10 * x)) = 1 := by
  sorry

end variance_scaled_data_l398_398936


namespace value_of_f_at_3_l398_398759

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem value_of_f_at_3 : f 3 = 15 :=
by
  -- This proof needs to be filled in
  sorry

end value_of_f_at_3_l398_398759


namespace isosceles_triangle_area_l398_398763

theorem isosceles_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) 
  (h_eq : h = 6 * sqrt 2) 
  (h_leg : h = a * sqrt 2) 
  (area_eq : A = 1 / 2 * a^2) : 
  A = 18 :=
by
  sorry

end isosceles_triangle_area_l398_398763


namespace shortest_distance_parabola_point_l398_398912

-- Conditions
def parabola_point (y : ℝ) : ℝ × ℝ := (y^2 / 4, y)
def given_point : ℝ × ℝ := (3, 6)

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement of the problem
theorem shortest_distance_parabola_point :
  ∃ P: ℝ × ℝ, P ∈ (λ y, parabola_point(y)) ∧ distance given_point P = Real.sqrt 5 :=
sorry

end shortest_distance_parabola_point_l398_398912


namespace min_value_of_reciprocal_sum_l398_398107

theorem min_value_of_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 2 * b = 1) (h2 : c + 2 * d = 1) :
  16 ≤ (1 / a) + 1 / (b * c * d) :=
by
  sorry

end min_value_of_reciprocal_sum_l398_398107


namespace part1_part2_l398_398961

noncomputable def f (x a : ℝ) : ℝ := Real.log (x + a)

theorem part1 {a : ℝ} (h : Deriv (f 1 a) = 1/2) : a = 1 :=
  by sorry

noncomputable def g (x m : ℝ) : ℝ := Real.log x + 1/2 * x^2 - m * x
noncomputable def h (x c b : ℝ) : ℝ := Real.log x - c * x^2 - b * x

theorem part2 (m c b : ℝ) (h_m : m ≥ 5/2) (x1 x2 : ℝ) (h_extremes : g x1 m = 0 ∧ g x2 m = 0 ∧ x1 < x2)
  (h_zeros : h x1 c b = 0 ∧ h x2 c b = 0) :
  (x1 - x2) * HasDerivAt (λ x, Real.log x - c * x^2 - b * x) ((x1 + x2) / 2) = -6/5 + Real.log 4 :=
  by sorry

end part1_part2_l398_398961


namespace identify_boys_l398_398336

def three_boys
  (Petya : Type)
  (Vasya : Type)
  (Vitya : Type)
  (name_of_boy : Petya ∨ Vasya ∨ Vitya → Prop) : Prop := 
  sorry

theorem identify_boys (Petya Vasya Vitya : Type) :
  ∀ (boy : Petya ∨ Vasya ∨ Vitya),
  ∃ (name : boy → Prop), 
  (boy = Petya ∧ ∀ q, Petya answers truthfully to q) ∨
  (boy = Vasya ∧ Vasya lies on the first question but then answers truthfully) ∨
  (boy = Vitya ∧ Vitya lies on the first two questions but then answers truthfully) →
  ∃ (name_of_boy : boy → Prop),
  three questions suffice to determine (boy name_of_boy) := 
sorry

end identify_boys_l398_398336


namespace simon_number_of_legos_l398_398733

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l398_398733


namespace find_range_of_a_l398_398241

-- Definitions and conditions
def pointA : ℝ × ℝ := (0, 3)
def lineL (x : ℝ) : ℝ := 2 * x - 4
def circleCenter (a : ℝ) : ℝ × ℝ := (a, 2 * a - 4)
def circleRadius : ℝ := 1

-- The range to prove
def valid_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12 / 5

-- Main theorem
theorem find_range_of_a (a : ℝ) (M : ℝ × ℝ)
  (on_circle : (M.1 - (circleCenter a).1)^2 + (M.2 - (circleCenter a).2)^2 = circleRadius^2)
  (condition_MA_MD : (M.1 - pointA.1)^2 + (M.2 - pointA.2)^2 = 4 * M.1^2 + 4 * M.2^2) :
  valid_range a :=
sorry

end find_range_of_a_l398_398241


namespace periodic_sequence_not_constant_l398_398855

theorem periodic_sequence_not_constant :
  ∃ (x : ℕ → ℤ), (∀ n : ℕ, x (n+1) = 2 * x n + 3 * x (n-1)) ∧ (∃ T > 0, ∀ n : ℕ, x (n+T) = x n) ∧ (∃ n m : ℕ, n ≠ m ∧ x n ≠ x m) :=
sorry

end periodic_sequence_not_constant_l398_398855


namespace number_of_correct_statements_l398_398484

theorem number_of_correct_statements :
  let statement1 := (∀ n : ℤ, n < 0 → -1 >= n)
  let statement2 := (∀ a : ℤ, |a| > 0 ∨ a = 0)
  let statement3 := (∀ distance : ℤ, distance = 10 → -distance = -5 → distance - 15) -- corrected to assume correct interpretation of the statement
  let statement4 := (∀ (l : List ℤ), l.count (λ n, n < 0) % 2 = 1 → l.prod < 0)
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) → 
  (∃ n, n = 3) :=
by
  sorry 

end number_of_correct_statements_l398_398484


namespace max_value_of_f_l398_398839

theorem max_value_of_f : 
  ∃ x ∈ Ioo(0, 1), ∀ y ∈ Ioo(0, 1), x(1 - x) ≥ y(1 - y) ∧ x(1 - x) = 1/4 :=
by
  sorry

end max_value_of_f_l398_398839


namespace sum_of_squares_positive_l398_398788

theorem sum_of_squares_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2 > 0) ∧ (b^2 + c^2 > 0) ∧ (c^2 + a^2 > 0) :=
by
  sorry

end sum_of_squares_positive_l398_398788


namespace vertex_closer_to_Q_than_P_l398_398251

-- Defining the points and polyhedron
structure Polyhedron where
  vertices : finset (EuclideanSpace ℝ 3)
  is_convex : polyhedron.vertices.is_convex

variables {P Q : EuclideanSpace ℝ 3} (poly : Polyhedron)

-- Given that P and Q are inside the polyhedron
axiom P_in_poly : P ∈ poly.vertices.convex_hull
axiom Q_in_poly : Q ∈ poly.vertices.convex_hull

theorem vertex_closer_to_Q_than_P : 
  ∃ (v : EuclideanSpace ℝ 3), v ∈ poly.vertices ∧ dist v Q < dist v P := 
sorry

end vertex_closer_to_Q_than_P_l398_398251


namespace prob_sum_15_correct_l398_398377

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398377


namespace minimum_rooms_needed_l398_398009

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398009


namespace range_of_m_empty_solution_set_inequality_l398_398954

theorem range_of_m_empty_solution_set_inequality (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 ≥ 0 → false) ↔ -4 < m ∧ m < 0 := 
sorry

end range_of_m_empty_solution_set_inequality_l398_398954


namespace minimum_rooms_needed_l398_398002

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398002


namespace proj_rotation_l398_398676

variables (v w : ℝ → ℝ → ℝ → ℝ) -- ℝ space vectors
variables (R : matrix (fin 3) (fin 3) ℝ) -- The rotation matrix

def proj (w v : matrix (fin 3) (fin 1) ℝ) : matrix (fin 3) (fin 1) ℝ :=
  ((vᵀ ⬝ w) / (wᵀ ⬝ w)) * w

noncomputable def rotation_matrix : matrix (fin 3) (fin 3) ℝ :=
  !![![(0 : ℝ), -1, 0], ![1, 0, 0], ![0, 0, 1]]

def v_proj_w : matrix (fin 3) (fin 1) ℝ := ![![2], ![-1], ![4]]

theorem proj_rotation :
  proj w (rotation_matrix ⬝ v) = ![![1], ![2], ![4]] :=
sorry

end proj_rotation_l398_398676


namespace triangular_pyramid_circumscribed_sphere_radius_l398_398639

theorem triangular_pyramid_circumscribed_sphere_radius (a b c : ℝ) :
  (∃ R, R = sqrt (a^2 + b^2 + c^2) / 2) :=
by
  use sqrt (a^2 + b^2 + c^2) / 2
  sorry

end triangular_pyramid_circumscribed_sphere_radius_l398_398639


namespace number_of_lines_l398_398243

theorem number_of_lines (n : ℕ) 
  (h1 : ∀ i j : ℕ, i < j → i < n ∧ j < n → True)  -- Every pair of lines intersects
  (h2 : ¬ ∃ p : finset (fin n), p.card = 4 ∧ ∀ x ∈ p, ∀ y ∈ p, x ≠ y → collinear x y z )  -- No four lines are concurrent
  (h3 : ∃ k : ℕ, k = 16)  -- There are 16 intersection points
  (h4 : ∃ m : ℕ, m = 6)  -- 6 of these points are intersections of three lines each
  : n = 8 := sorry

end number_of_lines_l398_398243


namespace contact_probability_l398_398064

theorem contact_probability (n m : ℕ) (p : ℝ) (h_n : n = 5) (h_m : m = 8) (hp : 0 ≤ p ∧ p ≤ 1) :
  (1 - (1 - p)^(n * m)) = 1 - (1 - p)^(40) :=
by
  rw [h_n, h_m]
  sorry

end contact_probability_l398_398064


namespace polynomial_sign_condition_l398_398536

def polynomial (x y z : ℤ) : ℤ := x^3 + 2 * y^3 + 4 * z^3 - 6 * x * y * z

theorem polynomial_sign_condition (a b c : ℤ) :
  ∀ (a b c : ℤ), sign (polynomial a b c) = sign (a + b * (2^(1/3) : ℝ) + c * (4^(1/3) : ℝ)) :=
sorry

end polynomial_sign_condition_l398_398536


namespace hexagon_perimeter_le_two_thirds_triangle_perimeter_l398_398844

variables {α : Type*} [linear_ordered_field α] {a b c : α}

theorem hexagon_perimeter_le_two_thirds_triangle_perimeter 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  let p := (a + b + c) / 2 in
  let hexagon_perimeter := 2 * (a * (p - a) / p + b * (p - b) / p + c * (p - c) / p) in
  hexagon_perimeter ≤ 2 / 3 * (a + b + c) :=
by
  sorry

end hexagon_perimeter_le_two_thirds_triangle_perimeter_l398_398844


namespace area_of_rhombus_l398_398050

theorem area_of_rhombus (x y : ℝ) (hx : x = 8 * Real.sqrt 3) (hy : y = 4 * Real.sqrt 3)
  (radius_PQR : Real.circumradius (x * Real.sqrt (x^2 + y^2)) x (x * Real.sqrt (x^2 + y^2)) = 10)
  (radius_PQS : Real.circumradius (x * Real.sqrt (x^2 + y^2)) y (x * Real.sqrt (x^2 + y^2)) = 20)
  : rhombus_area : ℝ :=
by 
  let area := x * y
  have harea : area = 96
  sorry

end area_of_rhombus_l398_398050


namespace quadratic_roots_sum_product_l398_398185

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398185


namespace arithmetic_sequence_sum_l398_398581

theorem arithmetic_sequence_sum :
  ∀(a_n : ℕ → ℕ) (S : ℕ → ℕ) (a_1 d : ℕ),
    (∀ n, a_n n = a_1 + (n - 1) * d) →
    (∀ n, S n = n * (a_1 + (n - 1) * d) / 2) →
    a_1 = 2 →
    S 4 = 20 →
    S 6 = 42 :=
by
  sorry

end arithmetic_sequence_sum_l398_398581


namespace logarithmic_inequality_l398_398921

noncomputable def log_a_b (a b : ℝ) := Real.log b / Real.log a

theorem logarithmic_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  log_a_b a b + log_a_b b c + log_a_b a c ≥ 3 :=
by
  sorry

end logarithmic_inequality_l398_398921


namespace monotonic_decreasing_intervals_range_of_a_l398_398121

def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x
def g (x : ℝ) : ℝ := x * Real.exp (-x)

theorem monotonic_decreasing_intervals :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3),
  ∀ y ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3),
  x < y → f y < f x := sorry

theorem range_of_a :
  ∀ x1 ∈ Set.Icc 1 3, ∀ x2 ∈ Set.Icc 0 (Real.pi / 2),
  ∀ a : ℝ, g x1 + a + 3 > f x2 ↔ a > -3 / Real.exp 3 := sorry

end monotonic_decreasing_intervals_range_of_a_l398_398121


namespace base_three_to_decimal_l398_398746

theorem base_three_to_decimal :
  let n := 20121 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 178 :=
by {
  sorry
}

end base_three_to_decimal_l398_398746


namespace probability_sum_15_l398_398391

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398391


namespace magnitude_of_m_l398_398605
-- Import necessary libraries

-- Define the vector
def m : ℝ × ℝ × ℝ := (1, -2, 2)

-- Define the magnitude function
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Prove the magnitude of vector m is 3
theorem magnitude_of_m : magnitude m = 3 := by
  sorry

end magnitude_of_m_l398_398605


namespace binary_to_base5_conversion_l398_398899

-- Define the binary number 10111 in Lean
def binary_10111 := 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Define the expected decimal result
def binary_10111_to_dec := 23

-- Define the expected base-5 result
def dec_23_to_base5 := 43

theorem binary_to_base5_conversion :
  binary_10111 = binary_10111_to_dec ∧
  (by calc 
    binary_10111_to_dec : ℕ 
    := 23 
    : 23 
    ÷ 5 = 4 : by rfl
    let rem1 := 23 % 5 in 
    4 ⊗ rem1 : ℕ 
    := 4 :  rem1 
    : 0 : 0 ∑d end) :=
sorry

end binary_to_base5_conversion_l398_398899


namespace quadratic_root_sum_and_product_l398_398189

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398189


namespace binomial_expansion_product_l398_398566

theorem binomial_expansion_product (a a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5)
  (h2 : (1 - (-1))^5 = a - a1 + a2 - a3 + a4 - a5) :
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := by
  sorry

end binomial_expansion_product_l398_398566


namespace squirrel_can_catch_nut_l398_398087

-- Define the initial distance between Gabriel and the squirrel.
def initial_distance : ℝ := 3.75

-- Define the speed of the nut.
def nut_speed : ℝ := 5.0

-- Define the jumping distance of the squirrel.
def squirrel_jump_distance : ℝ := 1.8

-- Define the acceleration due to gravity.
def gravity : ℝ := 10.0

-- Define the positions of the nut and the squirrel as functions of time.
def nut_position_x (t : ℝ) : ℝ := nut_speed * t
def squirrel_position_x : ℝ := initial_distance
def nut_position_y (t : ℝ) : ℝ := 0.5 * gravity * t^2

-- Define the squared distance between the nut and the squirrel.
def distance_squared (t : ℝ) : ℝ :=
  (nut_position_x t - squirrel_position_x)^2 + (nut_position_y t)^2

-- Prove that the minimum distance squared is less than or equal to the squirrel's jumping distance squared.
theorem squirrel_can_catch_nut : ∃ t : ℝ, distance_squared t ≤ squirrel_jump_distance^2 := by
  -- Sorry placeholder, as the proof is not required.
  sorry

end squirrel_can_catch_nut_l398_398087


namespace george_possible_change_sum_l398_398549

theorem george_possible_change_sum : 
  ∃ (change : ℕ), 
    change < 100 ∧
    (change % 25 = 3) ∧ 
    (change % 10 = 8) ∧ 
    (change = 28 ∨ change = 78) ∧ 
    ∑ c in {28, 78}, c = 106 := 
by
  sorry

end george_possible_change_sum_l398_398549


namespace power_of_two_with_nines_l398_398298

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ (n : ℕ), (2^n % 10^k) / 10^((10 * 5^k + k + 2 - k) / 2) = 9 :=
sorry

end power_of_two_with_nines_l398_398298


namespace plane_split_into_regions_l398_398511

theorem plane_split_into_regions :
  ∀ (x y : ℝ), (y = 3 * x ∨ y = (1 / 3) * x) → (plane_split_into_regions y x 8) :=
by
  sorry

def plane_split_into_regions (y x : ℝ) (n : ℝ) := 
  x ≠ 0 ∧ y ≠ 0 ∧ (y = 3 * x ∨ y = (1 / 3) * x) → (number_of_splits x y = n)

def number_of_splits (x y : ℝ) := 
  -- Function that calculates the number of regions created by lines y = 3x and y = 1/3x.
  -- This is a placeholder function to demonstrate the structure.
  if (y = 3 * x ∨ y = (1/3) * x) then 8 else 1

#check plane_split_into_regions

end plane_split_into_regions_l398_398511


namespace tim_average_sleep_is_correct_l398_398361

def total_weekday_sleep : ℝ :=
  (6 + 0.5) + (6 + 0.5) + (10 + 0.5) + (10 + 0.5) + (8 + 0.5)

def total_weekend_sleep : ℝ :=
  9 + 9

def total_weekly_sleep : ℝ :=
  total_weekday_sleep + total_weekend_sleep

def average_sleep_per_day (total_sleep : ℝ) : ℝ :=
  total_sleep / 7

theorem tim_average_sleep_is_correct :
  average_sleep_per_day total_weekly_sleep = 60.5 / 7 := 
by
  unfold total_weekday_sleep
  unfold total_weekend_sleep
  unfold total_weekly_sleep
  unfold average_sleep_per_day
  sorry

end tim_average_sleep_is_correct_l398_398361


namespace determine_k_l398_398918

theorem determine_k (k : ℝ) : 
  (∀ x : ℝ, (x^2 = 2 * x + k) → (∃ x0 : ℝ, ∀ x : ℝ, (x - x0)^2 = 0)) ↔ k = -1 :=
by 
  sorry

end determine_k_l398_398918


namespace total_cookies_is_correct_l398_398490

-- Define the number of people
def num_people : ℕ := 5

-- Define the number of cookies per person
def cookies_per_person : ℕ := 7

-- Define the total number of cookies prepared
def total_cookies : ℕ := num_people * cookies_per_person

-- Prove that the total number of cookies prepared is 35
theorem total_cookies_is_correct : total_cookies = 35 := by
  rw [total_cookies]
  unfold num_people cookies_per_person
  norm_num
  sorry

end total_cookies_is_correct_l398_398490


namespace phone_price_in_october_l398_398785

variable (a : ℝ) (P_October : ℝ) (r : ℝ)

noncomputable def price_in_january := a
noncomputable def price_in_october (a : ℝ) (r : ℝ) := a * r^9

theorem phone_price_in_october :
  r = 0.97 →
  P_October = price_in_october a r →
  P_October = a * (0.97)^9 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end phone_price_in_october_l398_398785


namespace power_function_value_at_4_l398_398629

-- Define the power function
def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

-- Given condition: the function passes through the point (2, sqrt(2))
def condition (a : ℝ) : Prop := power_function a 2 = Real.sqrt 2

-- The theorem to prove: the value of the function at x = 4 is 2, given the condition
theorem power_function_value_at_4 (a : ℝ) (h : condition a) : power_function a 4 = 2 :=
by
  -- We need to provide the proof here
  sorry

end power_function_value_at_4_l398_398629


namespace value_of_m_l398_398208

theorem value_of_m (m : ℝ) : (∀ (x : ℝ), (m-1) * x^2 + 5 * x + (m^2 - 1) = 0) → m = -1 :=
begin
  intros h,
  have h1 : m^2 - 1 = 0,
  from sorry, -- This comes from the condition that constant term is 0.
  have h2 : m - 1 ≠ 0,
  from sorry, -- This asserts that the coefficient of x^2 is non-zero.
  linarith, -- This will use both h1 and h2 to show m = -1.
end

end value_of_m_l398_398208


namespace eval_expr_l398_398889

theorem eval_expr : - (1 : ℝ) ^ 2023 - ( (π - 3) ^ 0 ) + ( (1 / 2) ^ (-1 : ℝ) ) + | 2 - sqrt 3 | + (6 / sqrt 2) - sqrt 18 = 2 - sqrt 3 :=
by sorry

end eval_expr_l398_398889


namespace extreme_values_of_f_l398_398126

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - 9 * x

theorem extreme_values_of_f :
  ∀ x ∈ Ioo (-2 : ℝ) 2, 
  (∀ y ∈ Ioo (-2 : ℝ) 2, f y ≤ 5 ∧ f y ≥ -2) ∧
  ∃ x₁ ∈ Ioo (-2 : ℝ) 2, f x₁ = 5 ∧ 
  ∃ x₂ ∈ Ioo (-2 : ℝ) 2, f x₂ = -2 :=
sorry

end extreme_values_of_f_l398_398126


namespace fraction_to_decimal_l398_398906

theorem fraction_to_decimal :
  (58 / 200 : ℝ) = 1.16 := by
  sorry

end fraction_to_decimal_l398_398906


namespace min_value_l398_398540

theorem min_value (x : ℝ) (h : x > 1) : ∃ m : ℝ, m = 2 * Real.sqrt 5 ∧ ∀ y : ℝ, y = Real.sqrt (x - 1) → (x = y^2 + 1) → (x + 4) / y = m :=
by
  sorry

end min_value_l398_398540


namespace b_minus_c_l398_398080

def a_n (n : ℕ) (hn : n > 1) : ℝ := 1 / Real.log 2017 / Real.log n

noncomputable def b : ℝ :=
  a_n 2 (by norm_num) + a_n 3 (by norm_num) + 
  a_n 4 (by norm_num) + a_n 5 (by norm_num) + 
  a_n 6 (by norm_num)

noncomputable def c : ℝ :=
  a_n 13 (by norm_num) + a_n 17 (by norm_num) + 
  a_n 19 (by norm_num) + a_n 23 (by norm_num) + 
  a_n 29 (by norm_num)

theorem b_minus_c : b - c = -Real.log (449.057638888889) / Real.log 2017 :=
by
  sorry

end b_minus_c_l398_398080


namespace tournament_teams_l398_398737

theorem tournament_teams (n : ℕ) (H : 240 = 2 * n * (n - 1)) : n = 12 := 
by sorry

end tournament_teams_l398_398737


namespace coordinates_of_A_l398_398941

-- Define initial coordinates of point A
def A : ℝ × ℝ := (-2, 4)

-- Define the transformation of moving 2 units upwards
def move_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

-- Define the transformation of moving 3 units to the left
def move_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Combine the transformations to get point A'
def A' : ℝ × ℝ :=
  move_left (move_up A 2) 3

-- The theorem stating that A' is (-5, 6)
theorem coordinates_of_A' : A' = (-5, 6) :=
by
  sorry

end coordinates_of_A_l398_398941


namespace minimum_value_of_g_l398_398077

def g : ℝ → ℝ :=
  λ x, (3 * x^2 + 9 * x + 18) / (4 * (1 + x))

theorem minimum_value_of_g :
  ∃ x₀ ≥ 0, g x₀ = (3 * Real.sqrt 11.25) / 2 ∧ (∀ x ≥ 0, g x ≥ (3 * Real.sqrt 11.25) / 2) :=
by
  sorry

end minimum_value_of_g_l398_398077


namespace find_m_div_n_l398_398104

variable (m n : ℝ)
variable (m_pos : 0 < m)
variable (n_pos : 0 < n)
variable (m_ne_n : m ≠ n)

-- Ellipse C: mx^2 + ny^2 = 1
-- Line L: x + y + 1 = 0
-- The slope of the line passing through the origin and the midpoint of segment AB is sqrt(2)/2.

theorem find_m_div_n (h : (m + n) ≠ 0)
  (midpoint_slope : ∃ M : ℝ × ℝ,
    (∀ A B : ℝ × ℝ,
      (A.1 + B.1 = -2 * n / (m + n)) ∧ (A.2 + B.2 = -2 * m / (m + n)) ∧
      M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
    M.2 / M.1 = sqrt 2 / 2) : m / n = sqrt 2 / 2 :=
sorry

end find_m_div_n_l398_398104


namespace quadratic_root_identity_l398_398204

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398204


namespace arccos_one_half_eq_pi_div_three_l398_398502

noncomputable def arccos_of_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = π / 3

theorem arccos_one_half_eq_pi_div_three : arccos_of_one_half_eq_pi_div_three :=
  by sorry

end arccos_one_half_eq_pi_div_three_l398_398502


namespace greatest_possible_q_minus_r_l398_398338

theorem greatest_possible_q_minus_r :
  ∃ (q r : ℕ), 945 = 21 * q + r ∧ 0 ≤ r ∧ r < 21 ∧ q - r = 45 :=
by
  sorry

end greatest_possible_q_minus_r_l398_398338


namespace cost_prices_proof_l398_398444

section
variables (C_t C_b O_c : ℝ)

-- Conditions
def paid_computer_table := 8450
def paid_bookshelf := 6250
def paid_chair := 3400

def markup_computer_table := 0.30
def markup_bookshelf := 0.25
def discount_chair := 0.15

-- Equivalent mathematical conditions
def computer_table_condition := paid_computer_table = C_t * (1 + markup_computer_table)
def bookshelf_condition := paid_bookshelf = C_b * (1 + markup_bookshelf)
def chair_condition := paid_chair = O_c * (1 - discount_chair)

-- Targeting to prove the cost prices
theorem cost_prices_proof :
  computer_table_condition C_t →
  bookshelf_condition C_b →
  chair_condition O_c →
  C_t = 6500 ∧ C_b = 5000 ∧ O_c = 4000 :=
by
  intros hc_t hc_b hc_c
  sorry
end

end cost_prices_proof_l398_398444


namespace quadratic_inequality_solution_l398_398086

theorem quadratic_inequality_solution :
  {x : ℝ | (x^2 - 50 * x + 576) ≤ 16} = {x : ℝ | 20 ≤ x ∧ x ≤ 28} :=
sorry

end quadratic_inequality_solution_l398_398086


namespace minimum_rooms_needed_fans_l398_398020

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398020


namespace macaroon_problem_l398_398545

def total_macaroons_remaining (red_baked green_baked red_ate green_ate : ℕ) : ℕ :=
  (red_baked - red_ate) + (green_baked - green_ate)

theorem macaroon_problem :
  let red_baked := 50 in
  let green_baked := 40 in
  let green_ate := 15 in
  let red_ate := 2 * green_ate in
  total_macaroons_remaining red_baked green_baked red_ate green_ate = 45 :=
by
  sorry

end macaroon_problem_l398_398545


namespace solve_equation_l398_398311

noncomputable def equation_solution (x : ℝ) : Prop :=
  (3 / 2) * log (x + 2)^2 / log (1 / 4) - 3 = log (4 - x)^3 / log (1 / 4) - log (x + 6)^3 / log 4

theorem solve_equation :
  ∀ x : ℝ, (x ≠ -2) ∧ (x < 4) ∧ (x > -6) → equation_solution x ↔ (x = 2 ∨ x = 1 - real.sqrt 33) :=
by {
  sorry -- Proof skipped
}

end solve_equation_l398_398311


namespace probability_sum_15_l398_398388

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398388


namespace scientific_notation_representation_l398_398833

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_representation_l398_398833


namespace probability_of_connection_l398_398069

theorem probability_of_connection (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  let num_pairs := 5 * 8 in
  let prob_no_connection := (1 - p) ^ num_pairs in
  1 - prob_no_connection = 1 - (1 - p) ^ 40 := 
by
  let num_pairs := 5 * 8
  have h_num_pairs : num_pairs = 40 := by norm_num
  rw h_num_pairs
  let prob_no_connection := (1 - p) ^ 40
  sorry

end probability_of_connection_l398_398069


namespace complement_of_M_l398_398674

open Set

def U : Set ℝ := univ
def M : Set ℝ := { x | x^2 - 2 * x > 0 }
def comp_M_Real := compl M

theorem complement_of_M :
  comp_M_Real = { x : ℝ | 0 ≤ x ∧ x ≤ 2 } :=
sorry

end complement_of_M_l398_398674


namespace sum_tangents_l398_398350

theorem sum_tangents (f : ℕ → ℝ) : (∀ k, f k = Real.tan ((4 * k + 1) * Real.pi / 180)) → (∑ k in Finset.range 45, f k) = 45 := 
sorry

end sum_tangents_l398_398350


namespace bernoulli_sum_eq_1_bernoulli_sum_gt_1_l398_398282

noncomputable def bernoulli_probability_1 {n : ℕ} (xi : Fin n → ℕ → Prop) (lambda : Fin n → ℝ) (Delta : ℝ) : ℝ :=
  ∑ i in Finset.univ, lambda i * Delta + O(Delta^2)

noncomputable def bernoulli_probability_gt_1 {n : ℕ} (xi : Fin n → ℕ → Prop) (lambda : Fin n → ℝ) (Delta : ℝ) : ℝ :=
  O(Delta^2)

theorem bernoulli_sum_eq_1 {n : ℕ} (xi : Fin n → ℕ → Prop) (lambda : Fin n → ℝ) (Delta : ℝ) 
  (h_independent: ∀ i j, i ≠ j → Prob.independent (xi i) (xi j))
  (h_bernoulli: ∀ i, ∀ x, Prob.b_bernoulli (xi i) x (1 - lambda i * Delta) (lambda i * Delta))
  (h_positive_delta : 0 < Delta)
  (h_positive_lambda : ∀ i, 0 < lambda i) :
  (Prob.measure (fun ω => (Finset.univ.sum (λ i, xi i ω)) = 1) 
  = (Finset.univ.sum lambda) * Delta + O(Delta^2)) := sorry

theorem bernoulli_sum_gt_1 {n : ℕ} (xi : Fin n → ℕ → Prop) (lambda : Fin n → ℝ) (Delta : ℝ) 
  (h_independent: ∀ i j, i ≠ j → Prob.independent (xi i) (xi j))
  (h_bernoulli: ∀ i, ∀ x, Prob.b_bernoulli (xi i) x (1 - lambda i * Delta) (lambda i * Delta))
  (h_positive_delta : 0 < Delta)
  (h_positive_lambda : ∀ i, 0 < lambda i) :
  (Prob.measure (fun ω => (Finset.univ.sum (λ i, xi i ω)) > 1) 
  = O(Delta^2)) := sorry

end bernoulli_sum_eq_1_bernoulli_sum_gt_1_l398_398282


namespace simplify_fraction_l398_398758

theorem simplify_fraction : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := 
by sorry

end simplify_fraction_l398_398758


namespace function_root_range_l398_398628

theorem function_root_range (k : ℝ) :
  (∃ x : ℝ, 2^(-|x|) - k = 0) → k ∈ set.Ioo 0 1 := 
by
  sorry

end function_root_range_l398_398628


namespace alice_rearrangements_time_l398_398877

-- Define the conditions
def alice_letters : ℕ := 5
def rearrangements_per_minute : ℕ := 15

-- State the proof problem
theorem alice_rearrangements_time :
  let total_rearrangements := Nat.factorial alice_letters
      total_minutes := total_rearrangements / rearrangements_per_minute
      total_hours := total_minutes / 60.0 
  in total_hours = (2.0 / 15.0) := 
by
  sorry

end alice_rearrangements_time_l398_398877


namespace alyssa_games_next_year_l398_398878

/-- Alyssa went to 11 games this year -/
def games_this_year : ℕ := 11

/-- Alyssa went to 13 games last year -/
def games_last_year : ℕ := 13

/-- Alyssa will go to a total of 39 games -/
def total_games : ℕ := 39

/-- Alyssa plans to go to 15 games next year -/
theorem alyssa_games_next_year : 
  games_this_year + games_last_year <= total_games ∧
  total_games - (games_this_year + games_last_year) = 15 := by {
  sorry
}

end alyssa_games_next_year_l398_398878


namespace isosceles_triangle_area_l398_398765

theorem isosceles_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) 
  (h_eq : h = 6 * sqrt 2) 
  (h_leg : h = a * sqrt 2) 
  (area_eq : A = 1 / 2 * a^2) : 
  A = 18 :=
by
  sorry

end isosceles_triangle_area_l398_398765


namespace find_largest_n_l398_398538

def gcd_powers_of_2 (x : ℕ) : ℕ :=
  if x % 2 = 1 then 1 else sorry  -- Based on the problem, but lean implementation might need advancement.

def T (n : ℕ) : ℕ :=
  ∑ k in Finset.range (2^n), gcd_powers_of_2 (2*k+1)

theorem find_largest_n : ∃ n, n < 500 ∧ T n = 2^n ∧ isPerfectSquare (2^n) ∧ ∀ m, m < 500 → T m = 2^m ∧ isPerfectSquare (2^m) → m ≤ n := 
by
  sorry

end find_largest_n_l398_398538


namespace compute_fraction_equation_l398_398498

theorem compute_fraction_equation :
  (8 * (2 / 3: ℚ)^4 + 2 = 290 / 81) :=
sorry

end compute_fraction_equation_l398_398498


namespace monotonic_intervals_and_extreme_values_l398_398127

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + x + 1)

theorem monotonic_intervals_and_extreme_values :
  (∀ x, 0 ≤ Real.exp x) ∧
  (derivative f x = Real.exp x * (x + 2) * (x + 1)) ∧
  ((∀ x, x < -2 → derivative f x > 0) ∧ 
   (∀ x, x > -1 → derivative f x > 0) ∧ 
   (∀ x, -2 < x ∧ x < -1 → derivative f x < 0)) ∧
  (f (-2) = 3 / Real.exp 2) ∧
  (f (-1) = 1 / Real.exp 1) := 
  sorry

end monotonic_intervals_and_extreme_values_l398_398127


namespace four_fours_to_seven_l398_398410

theorem four_fours_to_seven :
  (∃ eq1 eq2 : ℕ, eq1 ≠ eq2 ∧
    (eq1 = 4 + 4 - (4 / 4) ∧
     eq2 = 44 / 4 - 4 ∧ eq1 = 7 ∧ eq2 = 7)) :=
by
  existsi (4 + 4 - (4 / 4))
  existsi (44 / 4 - 4)
  sorry

end four_fours_to_seven_l398_398410


namespace sasha_pluck_leaves_l398_398477

theorem sasha_pluck_leaves (num_apple_trees num_poplar_trees num_unphotographed start_pluck : ℕ)
  (H1 : num_apple_trees = 17)
  (H2 : num_poplar_trees = 18)
  (H3 : num_unphotographed = 13)
  (H4 : start_pluck = 8) :
  let total_trees := num_apple_trees + num_poplar_trees in
  let end_photographed := 10 in
  let pluck_start_tree := start_pluck in
  22 = total_trees - end_photographed - (10 - pluck_start_tree) + 1 :=
sorry

end sasha_pluck_leaves_l398_398477


namespace evaluateExpression_correct_l398_398890

open Real

noncomputable def evaluateExpression : ℝ :=
  (-2)^2 + 2 * sin (π / 3) - tan (π / 3)

theorem evaluateExpression_correct : evaluateExpression = 4 :=
  sorry

end evaluateExpression_correct_l398_398890


namespace find_sum_a_b_l398_398212

-- Define the conditions as variables and hypotheses
variables {a b : ℝ}

-- The main theorem to prove
theorem find_sum_a_b
  (h1 : 3 * a + 2 * b = 3)
  (h2 : 3 * b + 2 * a = 2) :
  a + b = 1 := sorry

end find_sum_a_b_l398_398212


namespace probability_of_contact_l398_398057

noncomputable def probability_connection (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

theorem probability_of_contact (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let group1 := 5
  let group2 := 8
  let total_pairs := group1 * group2
  (total_pairs = 40) →
  (∀ i j, i ∈ fin group1 → j ∈ fin group2 → (¬ p = 1 → p = p)) → 
  probability_connection p = 1 - (1 - p) ^ 40 :=
by
  intros _ _ 
  sorry

end probability_of_contact_l398_398057


namespace rectangular_block_volume_l398_398861

/-- 
A rectangular wooden block that is 72 dm long is sawed into 3 equal parts, 
and its surface area increases by 48 dm². Prove that the volume of this wooden block is 864 cubic decimeters. 
-/
theorem rectangular_block_volume 
  (length : ℕ) (num_parts : ℕ) (surface_increase : ℕ) (num_surfaces : ℕ) (base_area : ℕ) (height : ℕ) :
  length = 72 →
  num_parts = 3 →
  surface_increase = 48 →
  num_surfaces = 4 →
  (surface_increase / num_surfaces = base_area) →
  (length * (surface_increase / num_surfaces) = 864) :=
by intros; calc sorry

end rectangular_block_volume_l398_398861


namespace circle_segment_area_l398_398681

theorem circle_segment_area (AB AC BC r: ℝ) (h1 : AB = 9)
  (h2: r = 9 / 2)
  (h3: BC = 2 * r)
  (h4 : ∠BAC = 90)
  (h5 : tangent_to_sides_circle (X Y: ℝ) (X' Y' : ℝ)): 
  let circle_area := π * r^2 / 4
  let triangle_area := 1 / 2 * r^2
  let desired_area := circle_area - triangle_area
  desired_area = 81 * (π - 2) / 16 :=
by
  sorry

end circle_segment_area_l398_398681


namespace farthest_vertex_of_dilated_triangle_l398_398364
open real

-- Definitions based on given conditions
def centroid := (4 : ℝ, -4 : ℝ)
def area := 9
def scale_factor := 3

-- Coordinates of the vertex of the image of triangle PQR that is farthest from the origin
def farthest_vertex_coords := (12 : ℝ, -12 + 6 * sqrt 3)

-- Proof statement
theorem farthest_vertex_of_dilated_triangle :
  let PQR_centroid := centroid,
      PQR_area := area,
      dilation_center := (0 : ℝ, 0 : ℝ),
      PQ_horizontal := true,
      dilated_triangle_vertex := farthest_vertex_coords in
  dilated_triangle_vertex = (12, -12 + 6 * sqrt 3) :=
by sorry

end farthest_vertex_of_dilated_triangle_l398_398364


namespace average_visitors_per_day_l398_398453

theorem average_visitors_per_day 
  (avg_visitors_sunday : ℕ)
  (avg_visitors_otherday : ℕ)
  (total_days : ℕ)
  (begins_with_sunday : Bool) :
  avg_visitors_sunday = 510 →
  avg_visitors_otherday = 240 →
  begins_with_sunday = true →
  total_days = 30 →
  (4 * avg_visitors_sunday + 26 * avg_visitors_otherday) / total_days = 276 := 
by
  intros h1 h2 h3 h4
  have h_sundays : 4 = total_days / 7 := by sorry
  have h_otherdays : 26 = 30 - 4 := by sorry
  have h_total_visitors : (4 * avg_visitors_sunday + 26 * avg_visitors_otherday) = 8280 := by sorry
  have h_avg_visitors : 8280 / total_days = 276 := by sorry
  assumption

end average_visitors_per_day_l398_398453


namespace total_revenue_calculation_l398_398487

-- Define the total number of etchings sold
def total_etchings : ℕ := 16

-- Define the number of etchings sold at $35 each
def etchings_sold_35 : ℕ := 9

-- Define the price per etching sold at $35
def price_per_etching_35 : ℕ := 35

-- Define the price per etching sold at $45
def price_per_etching_45 : ℕ := 45

-- Define the total revenue calculation
def total_revenue : ℕ :=
  let revenue_35 := etchings_sold_35 * price_per_etching_35
  let etchings_sold_45 := total_etchings - etchings_sold_35
  let revenue_45 := etchings_sold_45 * price_per_etching_45
  revenue_35 + revenue_45

-- Theorem stating the total revenue is $630
theorem total_revenue_calculation : total_revenue = 630 := by
  sorry

end total_revenue_calculation_l398_398487


namespace calculate_expr_l398_398891

open Real

theorem calculate_expr :
  |(-sqrt 2)| + (2016 + real.pi)^0 + ((-1) / 2)^(-1) - 2 * (sin (pi / 4)) = -1 :=
by
  sorry

end calculate_expr_l398_398891


namespace imaginary_part_of_z_l398_398925

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : I * z = 1 + I) : z.im = -1 := 
sorry

end imaginary_part_of_z_l398_398925


namespace quadratic_roots_vieta_l398_398165

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398165


namespace combinations_problem_l398_398426

open Nat

-- Definitions for combinations
def C (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

-- Condition: Number of ways to choose 2 sergeants out of 6
def C_6_2 : Nat := C 6 2

-- Condition: Number of ways to choose 20 soldiers out of 60
def C_60_20 : Nat := C 60 20

-- Theorem statement for the problem
theorem combinations_problem :
  3 * C_6_2 * C_60_20 = 3 * 15 * C 60 20 := by
  simp [C_6_2, C_60_20, C]
  sorry

end combinations_problem_l398_398426


namespace sum_of_squares_of_coefficients_l398_398819

theorem sum_of_squares_of_coefficients :
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2 = 550 :=
by
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  have hc4 : Polynomial.coeff p 4 = 5 := sorry
  have hc3 : Polynomial.coeff p 3 = 20 := sorry
  have hc2 : Polynomial.coeff p 2 = 10 := sorry
  have hc1 : Polynomial.coeff p 1 = 5 := sorry
  calc
    (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2
      = 5^2 + 20^2 + 10^2 + 5^2 : by rw [hc4, hc3, hc2, hc1]
      = 25 + 400 + 100 + 25 : by norm_num
      = 550 : by norm_num


end sum_of_squares_of_coefficients_l398_398819


namespace trapezium_area_l398_398328

/-- The area of a trapezium cross-section of a water channel -/
theorem trapezium_area (top_width bottom_width height : ℕ) (h_top_width : top_width = 12)
    (h_bottom_width : bottom_width = 8) (h_height : height = 70) :
    (1 / 2 : ℚ) * (top_width + bottom_width) * height = 700 :=
by
  rw [h_top_width, h_bottom_width, h_height]
  norm_num
  sorry

end trapezium_area_l398_398328


namespace extend_table_l398_398232

def valid_table (T : matrix (fin (n-2)) (fin n) ℕ) (n : ℕ) : Prop :=
  (∀ r : fin (n-2), ∀ c1 c2 : fin n, c1 ≠ c2 → T r c1 ≠ T r c2)
  ∧ (∀ c : fin n, ∀ r1 r2 : fin (n-2), r1 ≠ r2 → T r1 c ≠ T r2 c)
  ∧ (∀ r : fin (n-2), ∀ c : fin n, 1 ≤ T r c ∧ T r c ≤ n)

theorem extend_table (T : matrix (fin (n-2)) (fin n) ℕ) (n : ℕ) (h : n > 2) : 
  valid_table T n → ∃ T' : matrix (fin n) (fin n) ℕ, 
    (∀ r : fin n, ∀ c1 c2 : fin n, c1 ≠ c2 → T' r c1 ≠ T' r c2)
    ∧ (∀ c : fin n, ∀ r1 r2 : fin n, r1 ≠ r2 → T' r1 c ≠ T' r2 c)
    ∧ (∀ r : fin n, ∀ c : fin n, 1 ≤ T' r c ∧ T' r c ≤ n)
    ∧ (∀ r : fin (n-2), ∀ c : fin n, T r c = T' ⟨r.val, by sorry⟩ c) :=
by sorry

end extend_table_l398_398232


namespace area_of_square_l398_398296

theorem area_of_square (A B C D E F : Type)
  [square : square ABCD]
  (E_midpoint : midpoint D C E)
  (F_intersect : meets_between (line_through B E) (diagonal AC) F)
  (area_AFED_eq_45 : area_quadrilateral AFED = 45) :
  area_square ABCD = 108 :=
by sorry

end area_of_square_l398_398296


namespace find_b_value_l398_398762

-- Define the midpoint function
def midpoint (p1 p2 : (ℝ × ℝ)) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the points
def p1 : ℝ × ℝ := (2, 5)
def p2 : ℝ × ℝ := (8, 11)

-- Define the perpendicular bisector condition
def is_perpendicular_bisector (m : ℝ × ℝ) (b : ℝ) : Prop :=
  m.1 + m.2 = b

-- Define the statement that needs to be proven
theorem find_b_value : ∃ (b : ℝ), is_perpendicular_bisector (midpoint p1 p2) b ∧ b = 13 := by
  sorry

end find_b_value_l398_398762


namespace two_cards_totaling_15_probability_l398_398370

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398370


namespace football_team_right_handed_players_l398_398430

theorem football_team_right_handed_players (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 34) (h3 : ∀ t, t ∈ (finset.range throwers) → right_handed t) : 
  let non_throwers := total_players - throwers,
      left_handed_non_throwers := (1/3 : ℝ) * non_throwers,
      right_handed_non_throwers := (2/3 : ℝ) * non_throwers
  in right_handed_throwers + right_handed_non_throwers = 58 :=
by
  sorry

noncomputable theory

def right_handed (n : ℕ) : Prop := sorry -- This is just a placeholder to ensure the code compiles.

end football_team_right_handed_players_l398_398430


namespace min_rooms_needed_l398_398000

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l398_398000


namespace proof_of_neg_p_or_neg_q_l398_398953

variables (p q : Prop)

theorem proof_of_neg_p_or_neg_q (h₁ : ¬ (p ∧ q)) (h₂ : p ∨ q) : ¬ p ∨ ¬ q :=
  sorry

end proof_of_neg_p_or_neg_q_l398_398953


namespace number_of_good_points_l398_398926

-- Definitions of fixed triangle and other points
variables {A B C : Point} (triangle_ABC : Triangle A B C)
variables {D : Point} (hD : D ≠ A ∧ D ≠ B ∧ D ≠ C)

-- Definition of good points
def is_good_point (D : Point) : Prop :=
  ∃ (circle : Circle D A), 
  let intersect_A := circle.intersect_line (line_through A B),
      intersect_C := circle.intersect_line (line_through A C),
      A_b := (intersect_A \ {A}).choose,
      A_c := (intersect_C \ {A}).choose in
  let intersect_B_a := (circle.center_at B).intersect_line (line_through B C),
      B_a := (intersect_B_a \ {B}).choose,
      B_c := (circle.center_at C).intersect_line (line_through C A) \ {C}.choose in
  is_on_circle A_b A_c B_a B_c C_a C_b

-- Theorem statement for the number of good points
theorem number_of_good_points (triangle_ABC : Triangle A B C) : 
  ∃ n, (n = 2 ∨ n = 3 ∨ n = 4) ∧ 
      (card {D : Point | is_good_point D}) = n :=
begin
  sorry
end

end number_of_good_points_l398_398926


namespace perpendicular_divides_AH_l398_398297

-- Conditions
variables (O A B C H K L: Type) [incidence_geometry O] 

-- Definitions
def is_circumcenter (O : Type) (A B C: Type) [incidence_geometry O] : Prop := 
  circle_circum (A, B, C) O

def is_orthocenter (H : Type) (A B C: Type) [incidence_geometry H] : Prop :=
  orthocenter (A, B, C) H

def is_parallel (OH BC: Type) [parallel_geometry OH BC] : Prop :=
  parallel OH BC

def is_parallelogram (A B H K: Type) [parallelogram_geometry A B H K] : Prop :=
  parallelogram A B H K

def intersects (OK AC: Type) (L: Type) [intersection OK AC L] : Prop :=
  segment_intersection OK AC L

-- Proof Problem
theorem perpendicular_divides_AH (O A B C H K L: Type) [incidence_geometry O]
  (circumcenter_O : is_circumcenter O A B C)
  (orthocenter_H : is_orthocenter H A B C)
  (parallel_OH_BC : is_parallel OH BC)
  (parallelogram_ABHK : is_parallelogram A B H K)
  (intersection_OK_AC_L : intersects OK AC L) :
  divides_perpendicular AH L (1 : 1) :=
sorry

end perpendicular_divides_AH_l398_398297


namespace sum_of_subsets_eq_l398_398915

def sum_of_subsets (n : ℕ) : ℕ × ℕ :=
  let sets := { S : Finset ℕ | S.nonempty ∧ S ⊆ Finset.range (n + 1) }
  let σ (S : Finset ℕ) := S.sum id
  let π (S : Finset ℕ) := S.prod id
  (Finset.sum sets (λ S, σ S / π S), n^2 + 2 * n - (Finset.range n).sum (λ i, 1 / (i + 1)) * (n + 1))

theorem sum_of_subsets_eq (n : ℕ) : 
  (sum_of_subsets n).fst = (sum_of_subsets n).snd :=
sorry

end sum_of_subsets_eq_l398_398915


namespace quadratic_roots_identity_l398_398178

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398178


namespace line_product_l398_398851

theorem line_product (b m : Int) (h_b : b = -2) (h_m : m = 3) : m * b = -6 :=
by
  rw [h_b, h_m]
  norm_num

end line_product_l398_398851


namespace fraction_sum_product_roots_of_quadratic_l398_398158

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398158


namespace first_group_weavers_count_l398_398738

def rate_per_day_per_weaver (mats : ℕ) (days : ℕ) (weavers : ℕ) : ℚ := (mats : ℚ) / (days : ℚ) / (weavers : ℚ)

axiom condition1 : ∃ weavers1 : ℕ, rate_per_day_per_weaver 4 4 weavers1 = 1 / 4
axiom condition2 : 8 * (rate_per_day_per_weaver 16 8 8) = 2

theorem first_group_weavers_count : ∃ x : ℕ, rate_per_day_per_weaver 4 4 x = 1 / 4 ∧ x = 4 :=
begin
  sorry
end

end first_group_weavers_count_l398_398738


namespace minimum_rooms_needed_fans_l398_398023

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398023


namespace a_2_value_a_3_value_exists_lambda_arith_seq_general_term_formula_l398_398972

open BigOperators

-- Define the sequence
def a : ℕ → ℕ
| 1       := 5
| (n + 1) := 2 * a n + 2^(n + 1) - 1

-- Define the corresponding proof statements

theorem a_2_value : a 2 = 13 := sorry

theorem a_3_value : a 3 = 33 := sorry

theorem exists_lambda_arith_seq : ∃ λ : ℝ, λ = -1 ∧ ∀ n : ℕ, n ≥ 1 → 
  (λ (a (n + 1) + λ) / 2^(n + 1) - λ (a n + λ) / 2^n = 1) := sorry

theorem general_term_formula : ∀ n : ℕ, n ≥ 1 → a n = (n + 1) * 2^n + 1 := sorry

end a_2_value_a_3_value_exists_lambda_arith_seq_general_term_formula_l398_398972


namespace max_volume_proof_l398_398689

noncomputable def maximum_volume (cardboard_length cardboard_width : ℝ) : ℝ :=
  (cardboard_length - 2 * 3) * (cardboard_width - 2 * 3) * 3

theorem max_volume_proof (cardboard_length cardboard_width : ℝ) (h_length : cardboard_length = 30) (h_width : cardboard_width = 14) :
  maximum_volume cardboard_length cardboard_width = 576 :=
by
  rw [h_length, h_width]
  show (30 - 2 * 3) * (14 - 2 * 3) * 3 = 576
  norm_num
  trivial

end max_volume_proof_l398_398689


namespace max_T_n_at_n_4_l398_398096

variable {a : ℕ → ℝ}
variable {T : ℕ → ℝ}
variable {n : ℕ}

-- Assume a geometric sequence
axiom geo_seq (n : ℕ) : a (n + 1) = q * a n

-- Conditions given
axiom a_1 : a 1 = -24
axiom a_4 : a 4 = - (8 / 9)

-- Definition of the product of the first n terms of a geometric sequence
noncomputable def T_n (n : ℕ) : ℝ :=
  ∏ i in finset.range n, a (i + 1)

-- The theorem to be proven
theorem max_T_n_at_n_4 (q : ℝ) (h : q ^ 3 = 1 / 27) : 
  ∃ n : ℕ, T n = T ⟨4, by norm_num⟩ ∧ ∀ m, m ≠ 4 → T m < T ⟨4, by norm_num⟩ :=
  sorry

end max_T_n_at_n_4_l398_398096


namespace subset_exists_special_numbers_l398_398903

theorem subset_exists_special_numbers :
  ∀ (A : Finset ℕ) (hA : A = Finset.range 101),
    ∃ (S : Finset (Finset ℕ)) (hS : S.card = 7),
      ∃ (A_i : Finset ℕ) (hA_i : A_i ∈ S),
        (∃ a b c d ∈ A_i, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d) ∨
        (∃ a b c ∈ A_i, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b = 2 * c) := 
by {
  sorry
}

end subset_exists_special_numbers_l398_398903


namespace squirrel_can_catch_nut_l398_398090

-- Definitions for the given conditions
def distance_gavrila_squirrel : Real := 3.75
def nut_velocity : Real := 5
def squirrel_jump_distance : Real := 1.8
def gravity_acceleration : Real := 10

-- Statement to be proved
theorem squirrel_can_catch_nut : ∃ t : Real, 
  let r_squared := (nut_velocity * t - distance_gavrila_squirrel)^2 + (gravity_acceleration * t^2 / 2)^2 in
  r_squared ≤ squirrel_jump_distance^2 :=
begin
  sorry
end

end squirrel_can_catch_nut_l398_398090


namespace cake_mix_buyers_l398_398843

def total_buyers : Nat := 100
def muffin_buyers : Nat := 40
def both_buyers : Nat := 16
def probability_neither : Real := 0.26

theorem cake_mix_buyers : ∀ (C : Nat),
  N = floor (probability_neither * total_buyers) →
  N = 26 →
  C + muffin_buyers - both_buyers = total_buyers - N →
  C = 50 :=
by
  intros C h1 h2 h3
  sorry

end cake_mix_buyers_l398_398843


namespace cheapest_option_l398_398983

/-
  Problem: Prove that gathering berries in the forest to make jam is
  the cheapest option for Grandmother Vasya.
-/

def gathering_berries_cost (transportation_cost_per_kg sugar_cost_per_kg : ℕ) := (40 + sugar_cost_per_kg : ℕ)
def buying_berries_cost (berries_cost_per_kg sugar_cost_per_kg : ℕ) := (150 + sugar_cost_per_kg : ℕ)
def buying_ready_made_jam_cost (ready_made_jam_cost_per_kg : ℕ) := (220 * 1.5 : ℕ)

theorem cheapest_option (transportation_cost_per_kg sugar_cost_per_kg berries_cost_per_kg ready_made_jam_cost_per_kg : ℕ) : 
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_berries_cost berries_cost_per_kg sugar_cost_per_kg ∧
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_ready_made_jam_cost ready_made_jam_cost_per_kg := 
by
  sorry

end cheapest_option_l398_398983


namespace value_of_abc_l398_398755

-- Conditions
def cond1 (a b : ℤ) : Prop := ∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)
def cond2 (b c : ℤ) : Prop := ∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)

-- Theorem statement
theorem value_of_abc (a b c : ℤ) (h₁ : cond1 a b) (h₂ : cond2 b c) : a + b + c = 31 :=
sorry

end value_of_abc_l398_398755


namespace last_two_digits_28_l398_398299

theorem last_two_digits_28 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (2^(2*n) * (2^(2*n+1) - 1)) % 100 = 28 :=
by
  sorry

end last_two_digits_28_l398_398299


namespace quadratic_root_sum_and_product_l398_398191

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398191


namespace distance_AF_is_4_l398_398456

-- Define focus 'F' for the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Define the inclination angle of the line l
def inclination_angle : ℝ := real.pi / 3  -- 60 degrees in radians

-- Define the equation of the line passing through focus with given angle
def line (x : ℝ) : ℝ := real.sqrt (3) * (x - 1)

-- Define the equation of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Point A on the parabola in the first quadrant
def point_A (x y : ℝ) : Prop := parabola x y ∧ y = line x ∧ x > 0 ∧ y > 0

-- Distinguish x-coordinate of point A that satisfies point_A
noncomputable def x_coordinate_of_A := (λ x, x > 0 ∧ real.sqrt (3) * (x - 1) > 0 ∧ 3 * x^2 - 10 * x + 3 = 0)

-- Calculate the distance AF, knowing coordinate x of A
noncomputable def distance_AF (x : ℝ) : ℝ := x + 1

-- Prove that |AF| = 4
theorem distance_AF_is_4 : ∃ x y : ℝ, point_A x y → distance_AF x = 4 :=
by
  unfold point_A parabola focus line inclination_angle distance_AF x_coordinate_of_A
  use 3
  use real.sqrt(3) * 2
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }

end distance_AF_is_4_l398_398456


namespace parallel_segments_have_same_slope_l398_398757

noncomputable def slope (p1 p2 : (ℝ × ℝ)) : ℝ :=
(p2.2 - p1.2) / (p2.1 - p1.1)

theorem parallel_segments_have_same_slope (k : ℝ) :
  let A := (-4, 0)
  let B := (2, -2)
  let X := (0, 8)
  let Y := (18, k)
  slope A B = slope X Y → k = 2 :=
by
  intros
  unfold slope at *
  sorry

end parallel_segments_have_same_slope_l398_398757


namespace garden_ratio_correct_l398_398446

def garden_length : ℕ := 25
def garden_width : ℕ := 15

def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

def ratio_of_width_to_perimeter_simplified (w p : ℕ) : ℚ :=
  let ratio := w * 1.0 / p
  (w / (Nat.gcd w p)) * 1.0 / (p / (Nat.gcd w p))

theorem garden_ratio_correct :
  ratio_of_width_to_perimeter_simplified garden_width (perimeter garden_length garden_width) = 3 / 16 :=
by
  sorry

end garden_ratio_correct_l398_398446


namespace sum_f_values_l398_398591

def f (x : ℝ) := 4^x / (4^x + 1)

theorem sum_f_values : (finset.range 4033).sum (λ k, f (k - 2016 : ℤ)) = (4033 : ℝ) / 2 := 
by 
  sorry

end sum_f_values_l398_398591


namespace jog_distance_l398_398852

def point (x y : ℝ) : ℝ × ℝ := (x, y)

-- Given points
def Alex := point 3 (-15)
def Casey := point (-2) 18
def Jordan := point (-1/2) 5

-- Midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Horizontal distance calculation
def horizontal_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  abs (p1.1 - p2.1)

-- The theorem to prove
theorem jog_distance :
  horizontal_distance (midpoint Alex Casey) Jordan = 1 :=
by
  sorry

end jog_distance_l398_398852


namespace more_blue_count_l398_398228

-- Definitions based on the conditions given in the problem
def total_people : ℕ := 150
def more_green : ℕ := 95
def both_green_blue : ℕ := 35
def neither_green_blue : ℕ := 25

-- The Lean statement to prove the number of people who believe turquoise is "more blue"
theorem more_blue_count : 
  (total_people - neither_green_blue) - (more_green - both_green_blue) = 65 :=
by 
  sorry

end more_blue_count_l398_398228


namespace sequence_sign_change_l398_398602

theorem sequence_sign_change :
  (∃ k : ℕ, k = 23 ∧ 
    let a : ℕ → ℚ := λ n, if n = 1 then 15 else if n = 2 then 43/3 else (47/3) - (2/3) * n in
    a k * a (k + 1) < 0)
  :=
sorry

end sequence_sign_change_l398_398602


namespace correct_statement_l398_398423

-- Defining the statements as variables
variable (A B C D : Prop)

-- Conditions from the problem
def statement_A : Prop :=
  "A sample of 100 students was taken from 1500 students in the school to investigate their winter vacation reading habits, and the sample size is 1500"

def statement_B : Prop :=
  "It is suitable to use a sampling survey to understand the viewership rate of the Beijing Winter Olympics"

def statement_C : Prop :=
  "It is suitable to use a comprehensive survey to investigate the anti-collision ability of a batch of cars"

def statement_D : Prop :=
  "It is suitable to use a sampling survey for security checks before passengers board a plane"

-- Conditions clarification in this context
axiom condition_A : ¬ statement_A
axiom condition_B : statement_B
axiom condition_C : ¬ statement_C
axiom condition_D : ¬ statement_D

-- Theorem stating the correct answer
theorem correct_statement : B = statement_B :=
by
  rw [←condition_B]
  assume A C D,
  trivial


end correct_statement_l398_398423


namespace prism_volume_is_48_sqrt_3_l398_398625

def regular_triangular_prism_volume (r : ℝ) (h : ℝ) (a : ℝ) : ℝ :=
  (1 / 2) * a * a * real.sin (real.pi / 3) * h

theorem prism_volume_is_48_sqrt_3 :
  regular_triangular_prism_volume 2 4 (4 * real.sqrt 3) = 48 * real.sqrt 3 :=
by 
  sorry

end prism_volume_is_48_sqrt_3_l398_398625


namespace flowers_sold_l398_398550

theorem flowers_sold (lilacs roses gardenias tulips orchids : ℕ) 
  (hlilacs : lilacs = 15)
  (hroses : roses = 3 * lilacs)
  (hgardenias : gardenias = (lilacs / 2).round)
  (htulips : tulips = 2 * (roses + gardenias))
  (horchids : orchids = (roses + gardenias + tulips) / 3) :
  roses + lilacs + gardenias + tulips + orchids = 227 := 
by
  sorry

end flowers_sold_l398_398550


namespace point_in_fourth_quadrant_l398_398567
open Real

-- Conditions
def is_internal_angle (α : ℝ) : Prop := 0 < α ∧ α < π ∧ α ≠ π / 2

def x (α : ℝ) (h : is_internal_angle α) : ℝ := 1 / Real.sin α - Real.cos α

def y (α : ℝ) (h : is_internal_angle α) : ℝ := Real.sin α - |Real.tan α|

-- The proof problem
theorem point_in_fourth_quadrant (α : ℝ) (h : is_internal_angle α) :
  (0 < x α h) ∧ (y α h < 0) :=
by
  sorry

end point_in_fourth_quadrant_l398_398567


namespace max_value_of_expression_l398_398922

noncomputable def max_expression_value (x y : ℝ) : ℝ :=
  let expr := x^2 + 6 * y + 2
  14

theorem max_value_of_expression 
  (x y : ℝ) (h : x^2 + y^2 = 4) : ∃ (M : ℝ), M = 14 ∧ ∀ x y, x^2 + y^2 = 4 → x^2 + 6 * y + 2 ≤ M :=
  by
    use 14
    sorry

end max_value_of_expression_l398_398922


namespace intersection_A_B_l398_398139

-- Definitions of sets A and B
def A := { x : ℝ | x ≥ -1 }
def B := { y : ℝ | y < 1 }

-- Statement to prove the intersection of A and B
theorem intersection_A_B : A ∩ B = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l398_398139


namespace line_furthest_from_origin_l398_398909

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the slope of the line segment joining O and P
def slope_OP : ℝ := (P.2 - O.2) / (P.1 - O.1)

-- Define the slope of the line perpendicular to OP
def perpendicular_slope : ℝ := -1 / slope_OP

-- The equation of the line in point-slope form
def line_equation_point_slope (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.2 - P.2 = k * (p.1 - P.1)

-- The equation of the line in general form
def line_equation_general (x y : ℝ) : Prop :=
  2 * x + y - 5 = 0

-- The main statement to be proved
theorem line_furthest_from_origin :
  ∀ p : ℝ × ℝ, line_equation_point_slope P perpendicular_slope p ↔ line_equation_general p.1 p.2 :=
by
  intros p
  sorry

end line_furthest_from_origin_l398_398909


namespace max_probability_of_binomial_l398_398641

open ProbabilityTheory

def P (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem max_probability_of_binomial :
  ∀ k : ℕ, k ≤ 5 →
    P 5 k (1/4) ≤ P 5 1 (1/4) :=
by
  -- Proof is omitted
  sorry

end max_probability_of_binomial_l398_398641


namespace ab_non_positive_l398_398145

theorem ab_non_positive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l398_398145


namespace minimum_rooms_needed_l398_398005

open Nat

theorem minimum_rooms_needed (num_fans : ℕ) (num_teams : ℕ) (fans_per_room : ℕ)
    (h1 : num_fans = 100) 
    (h2 : num_teams = 3) 
    (h3 : fans_per_room = 3) 
    (h4 : num_fans > 0) 
    (h5 : ∀ n, 0 < n ∧ n <= num_teams → n % fans_per_room = 0 ∨ n % fans_per_room = 1 ∨ n % fans_per_room = 2) 
    : num_fans / fans_per_room + if num_fans % fans_per_room = 0 then 0 else 3 := 
by
  sorry

end minimum_rooms_needed_l398_398005


namespace find_train_speed_l398_398874
   
   noncomputable def train_speed (train_length : ℝ) (time_to_pass : ℝ) (trolley_speed_kmph : ℝ) : ℝ :=
     let trolley_speed_mps := (trolley_speed_kmph * 1000) / 3600
     let relative_speed_mps := train_length / time_to_pass
     let train_speed_mps := relative_speed_mps - trolley_speed_mps
     train_speed_mps * 3.6

   theorem find_train_speed :
     train_speed 110 5.4995600351971845 12 ≈ 60 :=
   sorry
   
end find_train_speed_l398_398874


namespace quadratic_roots_sum_product_l398_398184

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398184


namespace identity_proof_l398_398723

theorem identity_proof (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  ( (x - b) * (x - c) / ( (a - b) * (a - c) )
  + ( (x - c) * (x - a) / ( (b - c) * (b - a) ) 
  + ( (x - a) * (x - b) / ( (c - a) * (c - b) ) = 1
:= by
  sorry

end identity_proof_l398_398723


namespace kids_went_home_l398_398359

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℝ) (went_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : remaining_kids = 8.0) : went_home = 14.0 :=
by 
  sorry

end kids_went_home_l398_398359


namespace exists_equilateral_triangle_l398_398939

-- Defining a function to create a type for the color
inductive Color 
| black
| white

-- Defining the function to color the plane
def plane_coloring (c : ℝ × ℝ → Color) : Prop :=
  ∃ x y z : ℝ × ℝ, 
    (dist x y = 1 ∨ dist x y = real.sqrt 3) ∧
    (dist y z = 1 ∨ dist y z = real.sqrt 3) ∧
    (dist z x = 1 ∨ dist z x = real.sqrt 3) ∧
    (c x = c y ∧ c y = c z)

-- The theorem that corresponds to the proof problem
theorem exists_equilateral_triangle (c : ℝ × ℝ → Color) : plane_coloring c :=
sorry

end exists_equilateral_triangle_l398_398939


namespace tan_alpha_plus_pi_div_four_l398_398551

theorem tan_alpha_plus_pi_div_four (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := 
by
  sorry

end tan_alpha_plus_pi_div_four_l398_398551


namespace equivalent_proof_problem_l398_398270

theorem equivalent_proof_problem:
  ∃ n : ℤ, 0 ≤ n ∧ n < 37 ∧ (5 * n) % 37 = 1 → ((3 ^ n) ^ 3 - 3) % 37 = 35 :=
by
  sorry

end equivalent_proof_problem_l398_398270


namespace term_is_12th_l398_398136

-- Defining the sequence function
def sequence (n : ℕ) : ℝ := real.sqrt (4 * n + 2)

-- Proving that 5sqrt(2) is the 12th term of the sequence
theorem term_is_12th : sequence 12 = 5 * real.sqrt 2 := 
by 
  -- This is a placeholder for the proof
  sorry

end term_is_12th_l398_398136


namespace isosceles_triangle_area_l398_398766

theorem isosceles_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) 
  (h_eq : h = 6 * sqrt 2) 
  (h_leg : h = a * sqrt 2) 
  (area_eq : A = 1 / 2 * a^2) : 
  A = 18 :=
by
  sorry

end isosceles_triangle_area_l398_398766


namespace concentration_problem_l398_398421

theorem concentration_problem (initial_volume initial_concentration final_concentration water_removed : ℕ) :
  initial_volume = 24 ∧ final_concentration = 60 ∧ water_removed = 8 →
  ∃ initial_concentration : ℕ, (/* some properties related to given conditions would follow */) :=
begin
  sorry
end

end concentration_problem_l398_398421


namespace average_visitors_per_day_l398_398454

theorem average_visitors_per_day 
  (avg_visitors_sunday : ℕ)
  (avg_visitors_otherday : ℕ)
  (total_days : ℕ)
  (begins_with_sunday : Bool) :
  avg_visitors_sunday = 510 →
  avg_visitors_otherday = 240 →
  begins_with_sunday = true →
  total_days = 30 →
  (4 * avg_visitors_sunday + 26 * avg_visitors_otherday) / total_days = 276 := 
by
  intros h1 h2 h3 h4
  have h_sundays : 4 = total_days / 7 := by sorry
  have h_otherdays : 26 = 30 - 4 := by sorry
  have h_total_visitors : (4 * avg_visitors_sunday + 26 * avg_visitors_otherday) = 8280 := by sorry
  have h_avg_visitors : 8280 / total_days = 276 := by sorry
  assumption

end average_visitors_per_day_l398_398454


namespace circle_m_condition_l398_398754

theorem circle_m_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + m = 0) → m < 5 :=
by
  sorry

end circle_m_condition_l398_398754


namespace f_domain_f_1_plus_f_neg3_f_a_plus_1_l398_398120

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 3) + 1 / (x - 2)

theorem f_domain : {x : ℝ | (x + 3 ≥ 0) ∧ (x ≠ 2)} = { x : ℝ | x ≥ -3 ∧ x ≠ 2 } :=
by sorry

theorem f_1_plus_f_neg3 : f 1 + f (-3) = 4 / 5 :=
by sorry

variables {a : ℝ} (h1 : a > -4) (h2 : a ≠ 1)

theorem f_a_plus_1 : f (a + 1) = real.sqrt (a + 4) + 1 / (a - 1) :=
by sorry

end f_domain_f_1_plus_f_neg3_f_a_plus_1_l398_398120


namespace find_cheapest_option_l398_398989

variable (transportation_cost : ℕ) (berries_collected : ℕ)
          (cost_train_per_week : ℕ) (cost_berries_market : ℕ)
          (cost_sugar : ℕ) (jam_rate : ℚ) (cost_ready_made_jam : ℕ)
      
-- Define the cost of gathering 1.5 kg of jam
def option1_cost := (cost_train_per_week / berries_collected + cost_sugar) * jam_rate

-- Define the cost of buying berries and sugar to make 1.5 kg of jam
def option2_cost := (cost_berries_market + cost_sugar) * jam_rate

-- Define the cost of buying 1.5 kg of ready-made jam
def option3_cost := cost_ready_made_jam * jam_rate

theorem find_cheapest_option :
  option1_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam 
  < min (option2_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam)
        (option3_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam) :=
by
  unfold option1_cost option2_cost option3_cost
  have hc1 : (40 : ℕ) + 54 = 94 := by norm_num
  have hc2 : (150 : ℕ) + 54 = 204 := by norm_num
  have hc3 : (220 : ℕ) * (3/2) = 330 := by norm_num
  linarith
  sorry

end find_cheapest_option_l398_398989


namespace simon_legos_l398_398736

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l398_398736


namespace triangle_is_right_angle_if_orthocenter_at_vertex_l398_398211

noncomputable def orthocenter (A B C : Point) : Point := sorry -- definition for orthocenter

-- Define the statement as a theorem
theorem triangle_is_right_angle_if_orthocenter_at_vertex
  {A B C : Point}
  (h : orthocenter A B C = A ∨ orthocenter A B C = B ∨ orthocenter A B C = C) :
  (angle A B C = 90 ∨ angle B A C = 90 ∨ angle A C B = 90) :=
sorry

end triangle_is_right_angle_if_orthocenter_at_vertex_l398_398211


namespace collinear_iff_exists_scalar_l398_398826

-- Define non-zero vectors and collinearity
variable {V : Type} [AddCommGroup V] [Module ℝ V] (a b : V)

-- Define conditions
def non_zero_vector (v : V) := v ≠ 0
def collinear (a b : V) := ∃ λ : ℝ, b = λ • a

-- Theorem to prove
theorem collinear_iff_exists_scalar (a b : V) (h₁ : non_zero_vector a) (h₂ : non_zero_vector b) :
  collinear a b ↔ ∃ λ : ℝ, b = λ • a := 
sorry

end collinear_iff_exists_scalar_l398_398826


namespace prove_C1_C2_l398_398230

-- Given conditions
variables {A B C C₁ C₂ θ : ℝ}
-- Isosceles triangle with equal base angles
axiom is_isosceles_triangle (triangle_ABC : A = B) : True
-- Altitude from C
axiom altitude_divides_C (α : ℝ) : A = α ∧ B = α ∧ (C₁ = 90 - α ∧ C₂ = 90 - α)
-- External angle at C
axiom external_angle (θ₀ : ℝ) (θ₀ = 30) : θ = 30 ∧ θ = 2 * α

theorem prove_C1_C2 :
  (A = B) → (θ = 30) → (C₁ = 75 ∧ C₂ = 75) :=
by
  assume h1 : A = B
  assume h2 : θ = 30
  have altitude_split : ∃ α : ℝ, A = α ∧ B = α ∧ C₁ = 90 - α ∧ C₂ = 90 - α := sorry
  obtain ⟨α, hα1, hα2, hC₁, hC₂⟩ := altitude_split
  have external_angle_proof : θ = 2 * α := sorry
  rw [external_angle_proof] at h2
  have α_val : α = 15 := by linarith[h2]
  rw [←hα1, ←hα2] at α_val
  rw [hC₁, hC₂] 
  use 75 ∧ 75 sorry 

end prove_C1_C2_l398_398230


namespace bsnt_value_l398_398672

theorem bsnt_value (B S N T : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hT : 0 < T)
    (h1 : Real.log (B * S) / Real.log 10 + Real.log (B * N) / Real.log 10 = 3)
    (h2 : Real.log (N * T) / Real.log 10 + Real.log (N * S) / Real.log 10 = 4)
    (h3 : Real.log (S * T) / Real.log 10 + Real.log (S * B) / Real.log 10 = 5) :
    B * S * N * T = 10000 :=
sorry

end bsnt_value_l398_398672


namespace sum_of_ratios_l398_398697

theorem sum_of_ratios (n : ℕ) (x : ℕ → ℕ) (h1 : n ≥ 3)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ k : ℕ, (x (i % n) + x ((i + 2) % n)) = k * x ((i + 1) % n)) :
  2 * n ≤ finset.sum (finset.range n) (λ i, (x (i % n) + x ((i + 2) % n)) / x ((i + 1) % n)) ∧
  finset.sum (finset.range n) (λ i, (x (i % n) + x ((i + 2) % n)) / x ((i + 1) % n)) < 3 * n :=
by
  -- Proof omitted
  sorry

end sum_of_ratios_l398_398697


namespace rate_of_fencing_per_metre_l398_398316

noncomputable def area_hectares := 17.56
noncomputable def cost := 10398.369069916303
noncomputable def area_square_meters := 175600.0
noncomputable def pi := Real.pi

theorem rate_of_fencing_per_metre :
  let radius := Real.sqrt (area_square_meters / pi)
  let circumference := 2 * pi * radius
  let rate := cost / circumference
  rate ≈ 7.0 :=
by
  sorry

end rate_of_fencing_per_metre_l398_398316


namespace product_of_possible_N_l398_398884

theorem product_of_possible_N :
  ∃ (N1 N2 : ℤ), (∀ (P : ℤ), 
    let D := P + N1 
    let D_5 := D - 8 
    let P_5 := P + 7 in 
    abs (D_5 - P_5) = 3 ∨ 
    let D := P + N2 
    let D_5 := D - 8 
    let P_5 := P + 7 in 
    abs (D_5 - P_5) = 3) ∧ 
  (N1 = 18 ∨ N1 = 12) ∧ 
  (N2 = 18 ∨ N2 = 12) ∧ 
  (N1 ≠ N2) ∧ 
  (N1 * N2 = 216) :=
by
  sorry

end product_of_possible_N_l398_398884


namespace event_B_more_likely_than_event_A_l398_398707

-- Definitions based on given conditions
def total_possible_outcomes := 6^3
def favorable_outcomes_B := (Nat.choose 6 3) * (Nat.factorial 3)
def prob_B := favorable_outcomes_B / total_possible_outcomes
def prob_A := 1 - prob_B

-- The theorem to be proved:
theorem event_B_more_likely_than_event_A (total_possible_outcomes = 216) 
    (favorable_outcomes_B = 120) 
    (prob_B = 5 / 9) 
    (prob_A = 4 / 9) :
    prob_B > prob_A := 
by {
    sorry
}

end event_B_more_likely_than_event_A_l398_398707


namespace contact_probability_l398_398066

theorem contact_probability (n m : ℕ) (p : ℝ) (h_n : n = 5) (h_m : m = 8) (hp : 0 ≤ p ∧ p ≤ 1) :
  (1 - (1 - p)^(n * m)) = 1 - (1 - p)^(40) :=
by
  rw [h_n, h_m]
  sorry

end contact_probability_l398_398066


namespace probability_two_cards_sum_15_from_standard_deck_l398_398399

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398399


namespace circumscribed_and_inscribed_radii_product_l398_398079

theorem circumscribed_and_inscribed_radii_product (a b c : ℝ) (h1 : a = 26) (h2 : b = 28) (h3 : c = 30) :
  let p := (a + b + c) / 2,
      S := Real.sqrt (p * (p - a) * (p - b) * (p - c)),
      R := (a * b * c) / (4 * S),
      r := S / p
  in R * r = 130 :=
by
  intros
  rw [h1, h2, h3]
  let p := (26 + 28 + 30) / 2
  let S := Real.sqrt (p * (p - 26) * (p - 28) * (p - 30))
  let R := (26 * 28 * 30) / (4 * S)
  let r := S / p
  -- The rest of the proof would follow from this point.
  sorry

end circumscribed_and_inscribed_radii_product_l398_398079


namespace total_remaining_macaroons_l398_398547

-- Define initial macaroons count
def initial_red_macaroons : ℕ := 50
def initial_green_macaroons : ℕ := 40

-- Define macaroons eaten
def eaten_green_macaroons : ℕ := 15
def eaten_red_macaroons : ℕ := 2 * eaten_green_macaroons

-- Define remaining macaroons
def remaining_red_macaroons : ℕ := initial_red_macaroons - eaten_red_macaroons
def remaining_green_macaroons : ℕ := initial_green_macaroons - eaten_green_macaroons

-- Prove the total remaining macaroons
theorem total_remaining_macaroons : remaining_red_macaroons + remaining_green_macaroons = 45 := 
by
  -- Proof omitted
  sorry

end total_remaining_macaroons_l398_398547


namespace problem_l398_398269

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := (1/2) * (a n + b n + real.sqrt ((a n)^2 + (b n)^2))

noncomputable def b : ℕ → ℝ
| 0       := -3
| (n + 1) := (1/2) * (a n + b n - real.sqrt ((a n)^2 + (b n)^2))

theorem problem :
  (1 / a 1000) + (1 / b 1000) = (2 / 3) * (1 / 2) ^ 1000 :=
by
  sorry

end problem_l398_398269


namespace quadratic_roots_vieta_l398_398166

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398166


namespace sasha_leaves_count_l398_398479

theorem sasha_leaves_count 
    (apple_trees : ℕ) 
    (poplar_trees : ℕ) 
    (masha_stop : ℕ) 
    (sasha_start : ℕ) 
    : apple_trees = 17 → 
      poplar_trees = 18 → 
      masha_stop = 10 → 
      sasha_start = 8 → 
      ∃ (total_leaves_plucked : ℕ), total_leaves_plucked = 22 :=
begin
  intros,
  have total_trees : ℕ := apple_trees + poplar_trees,
  have total_trees_after_masha : ℕ := total_trees - masha_stop,
  have total_trees_after_sasha : ℕ := total_trees - (sasha_start - 1),
  have total_leaves_plucked : ℕ := total_trees_after_sasha - (total_trees_after_masha - 13),
  use total_leaves_plucked,
  exact calc
    total_leaves_plucked = 35 - 7 - (35 - 10 - 13) : by simp [total_trees, total_trees_after_masha, total_trees_after_sasha]
                    ...  = total_leaves_plucked      : by sorry,
end

end sasha_leaves_count_l398_398479


namespace cheapest_option_l398_398982

/-
  Problem: Prove that gathering berries in the forest to make jam is
  the cheapest option for Grandmother Vasya.
-/

def gathering_berries_cost (transportation_cost_per_kg sugar_cost_per_kg : ℕ) := (40 + sugar_cost_per_kg : ℕ)
def buying_berries_cost (berries_cost_per_kg sugar_cost_per_kg : ℕ) := (150 + sugar_cost_per_kg : ℕ)
def buying_ready_made_jam_cost (ready_made_jam_cost_per_kg : ℕ) := (220 * 1.5 : ℕ)

theorem cheapest_option (transportation_cost_per_kg sugar_cost_per_kg berries_cost_per_kg ready_made_jam_cost_per_kg : ℕ) : 
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_berries_cost berries_cost_per_kg sugar_cost_per_kg ∧
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_ready_made_jam_cost ready_made_jam_cost_per_kg := 
by
  sorry

end cheapest_option_l398_398982


namespace isosceles_right_triangle_area_l398_398769

-- Define the conditions as given in the problem statement
variables (h l : ℝ)
hypothesis (hypotenuse_rel : h = l * Real.sqrt 2)
hypothesis (hypotenuse_val : h = 6 * Real.sqrt 2)

-- Define the formula for the area of an isosceles right triangle
def area_of_isosceles_right_triangle (l : ℝ) : ℝ := (1 / 2) * l * l

-- Define the proof problem statement
theorem isosceles_right_triangle_area : 
  area_of_isosceles_right_triangle l = 18 :=
  sorry

end isosceles_right_triangle_area_l398_398769


namespace math_proof_problem_l398_398133

noncomputable def proof_problem : Prop :=
  ∃ (p : ℝ) (k m : ℝ), 
    (∀ (x y : ℝ), y^2 = 2 * p * x) ∧
    (p > 0) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (y1 * y2 = -8) ∧
      (x1 = 4 ∧ y1 = 0 ∨ x2 = 4 ∧ y2 = 0)) ∧
    (p = 1) ∧ 
    (∀ x0 : ℝ, 
      (2 * k * m = 1) ∧
      (∀ (x y : ℝ), y = k * x + m) ∧ 
      (∃ (r : ℝ), 
        ((x0 - r + 1 = 0) ∧
         (x0 - r * x0 + r^2 = 0))) ∧ 
       x0 = -1 / 2 )

theorem math_proof_problem : proof_problem := 
  sorry

end math_proof_problem_l398_398133


namespace james_total_socks_l398_398656

theorem james_total_socks :
  let red_pairs := 20
      black_pairs := red_pairs / 2
      total_red := red_pairs * 2
      total_black := black_pairs * 2
      combined_red_black := total_red + total_black
      white_socks := combined_red_black * 2
      total_socks := total_red + total_black + white_socks
  in total_socks = 180 :=
by
  sorry

end james_total_socks_l398_398656


namespace intersection_complement_l398_398976

open Finset

variable (U A B : Finset ℕ)
variable [DecidableEq ℕ]

def U := {2, 3, 4, 5, 6}
def A := {2, 3, 4}
def B := {2, 3, 5}

theorem intersection_complement :
    A ∩ (U \ B) = {4} :=
by
  sorry

end intersection_complement_l398_398976


namespace arithmetic_seq_slope_l398_398103

theorem arithmetic_seq_slope (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 + a 2 = 10) (h2 : a 3 + a 4 = 26) 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) : 
  (a (n + 1) - a n) / ((n + 1) - n) = 4 :=
by 
  obtain ⟨d, h_d⟩ := h_arith
  have key : a 2 = a 1 + d := h_d 1
  have key2 : a 3 = a 1 + 2 * d := (h_d 2).trans (congr_arg (λ t, a 1 + d + t) h_d 1)
  have key3 : a 4 = a 3 + d := h_d 3
  simp_rw [key, key2, key3] at h1 h2
  linarith
  sorry

end arithmetic_seq_slope_l398_398103


namespace prob_sum_15_correct_l398_398374

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398374


namespace smallest_m_for_power_of_two_l398_398138

open Set

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem smallest_m_for_power_of_two (X : Finset ℕ) (hX : X = (Finset.range 2002).erase 0) :
  ∃ m, (∀ W ⊆ X, W.card = m → ∃ u v ∈ W, is_power_of_two (u + v)) ∧ m = 999 :=
by
  sorry

end smallest_m_for_power_of_two_l398_398138


namespace arccos_one_half_l398_398503

theorem arccos_one_half :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_one_half_l398_398503


namespace direction_vector_of_line_l398_398445

theorem direction_vector_of_line : 
  ∀ (x y : ℝ), (∃ t : ℝ, x = 1 + 3 * t ∧ y = -1 + 4 * t) → (3, 4) ∈ ({(a, b) | ∃ t : ℝ, a = 3 * t ∧ b = 4 * t} : set (ℝ × ℝ)) :=
by
  intros x y h
  -- Proof steps would be filled in here
  sorry

end direction_vector_of_line_l398_398445


namespace find_z_in_sequence_l398_398244

theorem find_z_in_sequence (x y z a b : ℤ) 
  (h1 : b = 1)
  (h2 : a + b = 0)
  (h3 : y + a = 1)
  (h4 : z + y = 3)
  (h5 : x + z = 2) :
  z = 1 :=
sorry

end find_z_in_sequence_l398_398244


namespace arccos_one_half_eq_pi_div_three_l398_398506

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end arccos_one_half_eq_pi_div_three_l398_398506


namespace income_scientific_notation_l398_398116

theorem income_scientific_notation (avg_income_per_acre : ℝ) (acres : ℝ) (a n : ℝ) :
  avg_income_per_acre = 20000 →
  acres = 8000 → 
  (avg_income_per_acre * acres = a * 10 ^ n ↔ (a = 1.6 ∧ n = 8)) :=
by
  sorry

end income_scientific_notation_l398_398116


namespace largest_root_of_polynomial_l398_398897

theorem largest_root_of_polynomial (a0 a1 a2 a3 : ℝ)
  (h0 : |a0| ≤ 3) (h1 : |a1| ≤ 3) (h2 : |a2| ≤ 3) (h3 : |a3| ≤ 3) :
  ∃ r > 0, (r = 3) ∧ ∃ x : ℝ, (x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0 = 0) :=
begin
  sorry
end

end largest_root_of_polynomial_l398_398897


namespace curve_intersects_midline_at_unique_point_l398_398274

open Complex

noncomputable def intersection_point (a b c : ℝ) : ℝ × ℝ :=
  (1/2 : ℝ, (a + c + 2 * b) / 4)

theorem curve_intersects_midline_at_unique_point (a b c : ℝ) (z0 z1 z2 : ℂ)
  (h_z0 : z0 = ⟨0, a⟩)
  (h_z1 : z1 = ⟨1 / 2, b⟩)
  (h_z2 : z2 = ⟨1, c⟩) :
  ∃ t ∈ ℝ, let z := z0 * (Real.cos t) ^ 4 + 2 * z1 * (Real.cos t) ^ 2 * (Real.sin t) ^ 2 + z2 * (Real.sin t) ^ 4 in
  z.re = 1 / 2 ∧ z.im = (a + c + 2 * b) / 4 := 
sorry

end curve_intersects_midline_at_unique_point_l398_398274


namespace speed_difference_l398_398489

def anna_time_min := 15
def ben_time_min := 25
def distance_miles := 8

def anna_speed_mph := (distance_miles : ℚ) / (anna_time_min / 60 : ℚ)
def ben_speed_mph := (distance_miles : ℚ) / (ben_time_min / 60 : ℚ)

theorem speed_difference : (anna_speed_mph - ben_speed_mph : ℚ) = 12.8 := by {
  sorry
}

end speed_difference_l398_398489


namespace num_ordered_nine_tuples_l398_398942

theorem num_ordered_nine_tuples :
  ∃ (a : Fin 9 → ℕ), (∀ i j k : Fin 9, i < j → j < k → 
  (∃ l : Fin 9, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100)) ∧
  (card {a : Fin 9 → ℕ | 
    (∀ i j k : Fin 9, i < j → j < k → (∃ l : Fin 9, 
    l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100))} = 6) :=
sorry

end num_ordered_nine_tuples_l398_398942


namespace arccos_one_half_eq_pi_div_three_l398_398501

noncomputable def arccos_of_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = π / 3

theorem arccos_one_half_eq_pi_div_three : arccos_of_one_half_eq_pi_div_three :=
  by sorry

end arccos_one_half_eq_pi_div_three_l398_398501


namespace fencingCost_correct_l398_398530

noncomputable def pentagonSide1 : ℕ := 25
noncomputable def pentagonSide2 : ℕ := 35
noncomputable def pentagonSide3 : ℕ := 40
noncomputable def pentagonSide4 : ℕ := 45
noncomputable def pentagonSide5 : ℕ := 50

noncomputable def perimeter := pentagonSide1 + pentagonSide2 + pentagonSide3 + pentagonSide4 + pentagonSide5

noncomputable def rateA := 3.50
noncomputable def rateB := 2.25
noncomputable def rateC := 1.50

noncomputable def costA := perimeter * rateA
noncomputable def costB := perimeter * rateB
noncomputable def costC := perimeter * rateC

theorem fencingCost_correct :
  costA = 682.50 ∧
  costB = 438.75 ∧
  costC = 292.50 :=
by
  sorry

end fencingCost_correct_l398_398530


namespace max_m_value_l398_398570

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := exp x + b * x^2 + a

theorem max_m_value :
  ∀ (a b m : ℝ),
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → m ≤ g x a b ∧ g x a b ≤ m^2 - 2) →
  (∃ (a b : ℝ), a = 2 ∧ b = -1 ∧ g (-π / 4) a b = -1 ∧ (y = a * x + b + π / 2)) →
  m ≤ e ∧ m ≤ e + 1 :=
sorry

end max_m_value_l398_398570


namespace cartesian_of_polar_minimum_distance_l398_398236

-- Define the polar equation and show its Cartesian equivalent
theorem cartesian_of_polar (x y : ℝ) (h : 3 * (x^2 + y^2) = 12 * x - 10) :
  (x - 2)^2 + y^2 = 2 / 3 :=
by sorry

-- Define the curves and prove minimum distance
theorem minimum_distance (P Q : ℝ × ℝ)
  (hP : (P.1 - 2)^2 + P.2^2 = 2 / 3)
  (hQ : Q.1^2 / 16 + Q.2^2 / 4 = 1) :
  ∃ θ : ℝ, (min_dist_eq (distance P Q) = (sqrt (2 / 3))) :=
by sorry


end cartesian_of_polar_minimum_distance_l398_398236


namespace difference_of_lines_in_cm_l398_398524

def W : ℝ := 7.666666666666667
def B : ℝ := 3.3333333333333335
def inch_to_cm : ℝ := 2.54

theorem difference_of_lines_in_cm :
  (W * inch_to_cm) - (B * inch_to_cm) = 11.005555555555553 := 
sorry

end difference_of_lines_in_cm_l398_398524


namespace quadratic_roots_identity_l398_398179

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398179


namespace distance_between_parallel_sides_l398_398044

/-- Define the lengths of the parallel sides and the area of the trapezium -/
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area : ℝ := 285

/-- Define the condition of the problem: the formula for the area of the trapezium -/
def area_of_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

/-- The problem: prove the distance between the parallel sides is 15 cm -/
theorem distance_between_parallel_sides (h : ℝ) : 
  area_of_trapezium length_side1 length_side2 h = area → h = 15 :=
by
  sorry

end distance_between_parallel_sides_l398_398044


namespace original_number_is_13_l398_398853

theorem original_number_is_13 (x : ℝ) (h : 3 * (2 * x + 7) = 99) : x = 13 :=
sorry

end original_number_is_13_l398_398853


namespace arithmetic_sequence_sum_l398_398686

/-- 
  Define the conditions for the arithmetic sequence and 
  the sums of the first n terms S_n.
-/
variables {S_n : ℕ → ℝ} {a_n : ℕ → ℝ}

/--
  We set the given conditions.
-/
theorem arithmetic_sequence_sum (h1 : S_n 3 = 9)
                                (h2 : S_n 6 = 36) :
  (a_n 7 + a_n 8 + a_n 9) = 45 :=
sorry

end arithmetic_sequence_sum_l398_398686


namespace percentage_neither_language_l398_398694

noncomputable def total_diplomats : ℝ := 120
noncomputable def latin_speakers : ℝ := 20
noncomputable def russian_non_speakers : ℝ := 32
noncomputable def both_languages : ℝ := 0.10 * total_diplomats

theorem percentage_neither_language :
  let D := total_diplomats
  let L := latin_speakers
  let R := D - russian_non_speakers
  let LR := both_languages
  ∃ P, P = 100 * (D - (L + R - LR)) / D :=
by
  existsi ((total_diplomats - (latin_speakers + (total_diplomats - russian_non_speakers) - both_languages)) / total_diplomats * 100)
  sorry

end percentage_neither_language_l398_398694


namespace min_value_fraction_l398_398553

theorem min_value_fraction (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃ m, (∀ z, z = (1 / (x + 1) + 1 / y) → z ≥ m) ∧ m = (3 + 2 * Real.sqrt 2) / 2 :=
by
  sorry

end min_value_fraction_l398_398553


namespace range_of_function_l398_398517

theorem range_of_function : 
  ∀ x : ℝ, -real.sqrt 3 ≤ (real.sin (x + real.pi / 10) - real.cos (x + 4 * real.pi / 15)) ∧ (real.sin (x + real.pi / 10) - real.cos (x + 4 * real.pi / 15)) ≤ real.sqrt 3 :=
sorry

end range_of_function_l398_398517


namespace area_of_pentagon_AEDCB_l398_398303

structure Rectangle (A B C D : Type) :=
  (AB BC AD CD : ℕ)

def is_perpendicular (A E E' D : Type) : Prop := sorry

def area_of_triangle (AE DE : ℕ) : ℕ :=
  (AE * DE) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def area_of_pentagon (area_rect area_triangle : ℕ) : ℕ :=
  area_rect - area_triangle

theorem area_of_pentagon_AEDCB
  (A B C D E : Type)
  (h_rectangle : Rectangle A B C D)
  (h_perpendicular : is_perpendicular A E E D)
  (AE DE : ℕ)
  (h_ae : AE = 9)
  (h_de : DE = 12)
  : area_of_pentagon (area_of_rectangle 15 12) (area_of_triangle AE DE) = 126 := 
  sorry

end area_of_pentagon_AEDCB_l398_398303


namespace probability_two_cards_sum_15_from_standard_deck_l398_398395

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398395


namespace area_of_rectangle_l398_398930

-- Define the problem statement and conditions
theorem area_of_rectangle (p d : ℝ) :
  ∃ A : ℝ, (∀ (x y : ℝ), 2 * x + 2 * y = p ∧ x^2 + y^2 = d^2 → A = x * y) →
  A = (p^2 - 4 * d^2) / 8 :=
by 
  sorry

end area_of_rectangle_l398_398930


namespace term_2020_2187_is_1553rd_l398_398603

def is_valid_term (a n : ℕ) : Prop :=
  nat.gcd a 3 = 1 ∨ a = 2 ∧ n = 1

def sequence_term (idx : ℕ) : ℚ :=
  let n := nat.find (λ x, 3^x > idx) - 1 in
  let a := idx - (1 + list.sum (list.map (λ k, 2*3^k - 1) (list.range n))) in
  (a + 1) / 3^n

theorem term_2020_2187_is_1553rd :
  ∃ k, sequence_term k = 2020/2187 ∧ k = 1553 :=
by
  existsi 1553
  split
  sorry  -- skipping the proof as required in the task
  refl   -- by definition, the position is 1553

end term_2020_2187_is_1553rd_l398_398603


namespace quadratic_root_sum_and_product_l398_398190

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398190


namespace quadratic_roots_vieta_l398_398170

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398170


namespace length_of_each_reel_l398_398658

theorem length_of_each_reel
  (reels : ℕ)
  (sections : ℕ)
  (length_per_section : ℕ)
  (total_sections : ℕ)
  (h1 : reels = 3)
  (h2 : length_per_section = 10)
  (h3 : total_sections = 30)
  : (total_sections * length_per_section) / reels = 100 := 
by
  sorry

end length_of_each_reel_l398_398658


namespace first_train_speed_l398_398469

theorem first_train_speed:
  ∃ v : ℝ, 
    (∀ t : ℝ, t = 1 → (v * t) + (4 * v) = 200) ∧ 
    (∀ t : ℝ, t = 4 → 50 * t = 200) → 
    v = 40 :=
by {
 sorry
}

end first_train_speed_l398_398469


namespace booking_rooms_needed_l398_398014

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398014


namespace quadratic_root_sum_and_product_l398_398192

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398192


namespace infinite_lines_through_lattice_points_l398_398482

theorem infinite_lines_through_lattice_points : 
  ∀ (p : ℝ × ℝ), p = (10, 1 / 2) → ∃ (l : ℝ) (k : ℤ), {m : ℤ | ∃ n : ℤ, is_lattice_line (10, 1 / 2) m n} :=
sorry

end infinite_lines_through_lattice_points_l398_398482


namespace school_spent_on_grass_seeds_bottle_capacity_insufficient_l398_398840

-- Problem 1: Cost Calculation
theorem school_spent_on_grass_seeds (kg_seeds : ℝ) (cost_per_kg : ℝ) (total_cost : ℝ) 
  (h1 : kg_seeds = 3.3) (h2 : cost_per_kg = 9.48) :
  total_cost = 31.284 :=
  by
    sorry

-- Problem 2: Bottle Capacity
theorem bottle_capacity_insufficient (total_seeds : ℝ) (max_capacity_per_bottle : ℝ) (num_bottles : ℕ)
  (h1 : total_seeds = 3.3) (h2 : max_capacity_per_bottle = 0.35) (h3 : num_bottles = 9) :
  3.3 > 0.35 * 9 :=
  by
    sorry

end school_spent_on_grass_seeds_bottle_capacity_insufficient_l398_398840


namespace max_area_of_triangle_ABC_eq_3_add_sqrt2_l398_398561

noncomputable def max_area_triangle : ℝ :=
  let A := (-2 : ℝ, 0 : ℝ) in
  let B := (0 : ℝ, 2 : ℝ) in
  let C (θ : ℝ) := (Real.cos θ, -1 + Real.sin θ) in
  let vector_AB := (2 : ℝ, 2 : ℝ) in
  let base := Real.sqrt (2^2 + 2^2) in
  let d := (3 : ℝ) / Real.sqrt (1^2 + (-1)^2) in
  let r := 1 in
  let h := d + r in
  (1 / 2) * base * h

theorem max_area_of_triangle_ABC_eq_3_add_sqrt2 :
  max_area_triangle = 3 + Real.sqrt 2 :=
sorry

end max_area_of_triangle_ABC_eq_3_add_sqrt2_l398_398561


namespace isosceles_right_triangle_area_l398_398777

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l398_398777


namespace quadratic_roots_l398_398345

theorem quadratic_roots : ∀ x : ℝ, x * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 3) :=
by
  intro x
  split
  sorry

end quadratic_roots_l398_398345


namespace largest_square_advertisement_area_l398_398463

theorem largest_square_advertisement_area
  (width : ℝ) (height : ℝ) (border : ℝ)
  (hw : width = 9) (hh : height = 16) (hb : border = 1.5) :
  (max_square_area : ℝ) = 36 :=
by
  -- let side length of the square be s
  let s := min (width - 2 * border) (height - 2 * border)
  -- prove the side length is 6
  have hs : s = 6 := by sorry
  -- prove the area of the largest square advertisement is 36
  have : s * s = 36 := by sorry
  -- explicit return the value for max_square_area
  exact this

end largest_square_advertisement_area_l398_398463


namespace total_value_is_84_l398_398258

-- Definitions based on conditions
def number_of_stamps : ℕ := 21
def value_of_7_stamps : ℕ := 28
def stamps_per_7 : ℕ := 7
def stamp_value : ℤ := value_of_7_stamps / stamps_per_7
def total_value_of_collection : ℤ := number_of_stamps * stamp_value

-- Statement to prove the total value of the stamp collection
theorem total_value_is_84 : total_value_of_collection = 84 := by
  sorry

end total_value_is_84_l398_398258


namespace krishan_nandan_investment_l398_398662

def investment_ratio (k r₁ r₂ : ℕ) (N T Gn : ℕ) : Prop :=
  k = r₁ ∧ r₂ = 1 ∧ Gn = N * T ∧ k * N * 3 * T + Gn = 26000 ∧ Gn = 2000

/-- Given the conditions, the ratio of Krishan's investment to Nandan's investment is 4:1. -/
theorem krishan_nandan_investment :
  ∃ k N T Gn Gn_total : ℕ, 
    investment_ratio k 4 1 N T Gn  ∧ k * N * 3 * T = 24000 :=
by
  sorry

end krishan_nandan_investment_l398_398662


namespace sequence_term_l398_398955

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * n - 1

theorem sequence_term (n : ℕ) : a n = if n = 1 then 2 else 2 * n - 1 :=
begin
  have hS : ∀ n, S(n) = n^2 + 1, by { intro n, refl },
  have ha₁ : a 1 = 2, by { refl },
  have ha₂ : ∀ n, n ≥ 2 → a n = 2 * n - 1,
  { intros n hn,
    have h : S n = (n^2 + 1),
    { rw hS },
    have hpred : S (n - 1) = ((n - 1)^2 + 1),
    { rw hS },
    calc
      a n = S n - S (n - 1) : by sorry
      ... = (n^2 + 1) - ((n - 1)^2 + 1) : by rw [h, hpred]
      ... = 2 * n - 1 : by sorry },
  cases n,
  { exact ha₁ },
  { cases n,
    { exact ha₁ },
    { apply ha₂,
      linarith } }
end

end sequence_term_l398_398955


namespace radio_price_rank_l398_398883

theorem radio_price_rank (total_items : ℕ) (radio_position_highest : ℕ) (radio_position_lowest : ℕ) 
  (h1 : total_items = 40) (h2 : radio_position_highest = 17) : 
  radio_position_lowest = total_items - radio_position_highest + 1 :=
by
  sorry

end radio_price_rank_l398_398883


namespace quadratic_root_sum_and_product_l398_398196

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398196


namespace booking_rooms_needed_l398_398017

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398017


namespace verify_original_cost_l398_398876

-- Given conditions
def original_cost : ℝ := sorry
def decrease_percentage : ℝ := 0.24
def new_price : ℝ := 684
def expected_original_cost : ℝ := 900

-- The relationship between original cost and new price after decrease
theorem verify_original_cost :
  (1 - decrease_percentage) * original_cost = new_price →
  original_cost = expected_original_cost :=
by
  intros h
  rw [← one_mul (new_price / (1 - decrease_percentage))]
  rw [← mul_div_assoc]
  rw h
  sorry

end verify_original_cost_l398_398876


namespace total_students_school_l398_398745

noncomputable def total_students
  (total_donation : ℚ)
  (avg_donation_78 : ℚ)
  (num_students_78 : ℚ)
  (donation_9th : ℚ)
  (acceptance_rate_9th : ℚ)
  (num_students_9th : ℚ) : ℚ :=
  let avg_donation_9th := donation_9th * acceptance_rate_9th in
  let avg_donation_total := avg_donation_78 in
  total_donation / avg_donation_total

theorem total_students_school :
  let num_students_78 := 1 in
  let num_students_9th := 1 in
  total_students 13440 60 num_students_78 100 0.6 num_students_9th = 224 :=
sorry

end total_students_school_l398_398745


namespace holden_master_bath_size_l398_398141

theorem holden_master_bath_size (b n m : ℝ) (h_b : b = 309) (h_n : n = 918) (h : 2 * (b + m) = n) : m = 150 := by
  sorry

end holden_master_bath_size_l398_398141


namespace David_is_8_years_older_than_Scott_l398_398439

noncomputable def DavidAge : ℕ := 14 -- Since David was 8 years old, 6 years ago
noncomputable def RichardAge : ℕ := DavidAge + 6
noncomputable def ScottAge : ℕ := (RichardAge + 8) / 2 - 8
noncomputable def AgeDifference : ℕ := DavidAge - ScottAge

theorem David_is_8_years_older_than_Scott :
  AgeDifference = 8 :=
by
  sorry

end David_is_8_years_older_than_Scott_l398_398439


namespace two_cards_totaling_15_probability_l398_398369

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398369


namespace factor_expression_l398_398036

variable (a : ℝ)

theorem factor_expression : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) :=
by
  sorry

end factor_expression_l398_398036


namespace isosceles_right_triangle_area_l398_398770

-- Define the conditions as given in the problem statement
variables (h l : ℝ)
hypothesis (hypotenuse_rel : h = l * Real.sqrt 2)
hypothesis (hypotenuse_val : h = 6 * Real.sqrt 2)

-- Define the formula for the area of an isosceles right triangle
def area_of_isosceles_right_triangle (l : ℝ) : ℝ := (1 / 2) * l * l

-- Define the proof problem statement
theorem isosceles_right_triangle_area : 
  area_of_isosceles_right_triangle l = 18 :=
  sorry

end isosceles_right_triangle_area_l398_398770


namespace pears_remaining_l398_398661

theorem pears_remaining (K_picked : ℕ) (M_picked : ℕ) (S_picked : ℕ)
                        (K_gave : ℕ) (M_gave : ℕ) (S_gave : ℕ)
                        (hK_pick : K_picked = 47)
                        (hM_pick : M_picked = 12)
                        (hS_pick : S_picked = 22)
                        (hK_give : K_gave = 46)
                        (hM_give : M_gave = 5)
                        (hS_give : S_gave = 15) :
  (K_picked - K_gave) + (M_picked - M_gave) + (S_picked - S_gave) = 15 :=
by
  sorry

end pears_remaining_l398_398661


namespace isosceles_triangle_cross_sections_count_l398_398848

-- The number of cross-sections through vertex A of a regular tetrahedron ABCD, 
-- forming isosceles triangles with the base BCD and making an angle of 75 degrees,
-- is 18.
theorem isosceles_triangle_cross_sections_count :
  ∀ (A B C D : Point) (is_tetrahedron : regular_tetrahedron A B C D),
  let base := triangle B C D in
  ∃! cross_sections : set (triangle A (Point_on_base base) (Point_on_base base)),
    (∀ T ∈ cross_sections, isosceles T) ∧
    (∀ T ∈ cross_sections, angle_between T base = 75) ∧
    (finset.card cross_sections = 18) := sorry

end isosceles_triangle_cross_sections_count_l398_398848


namespace find_acute_angles_l398_398341

universe u

variables {α : Type u}

-- Let define the right-angled triangle with the right angle at C
structure RightAngledTriangle extends Triangle where
  C_right_angle : C_is_right_angle

-- Definitions
def centroid (T : RightAngledTriangle) := (T.median₁ ∩ T.median₂)
def incenter (T : RightAngledTriangle) := (T.inner_circle_center)
def inradius (T : RightAngledTriangle) := (T.inner_circle_radius)

def acute_angle1 (T : RightAngledTriangle) := T.angle_BAC
def acute_angle2 (T : RightAngledTriangle) := T.angle_ABC

-- Conditions of the problem
variables {T : RightAngledTriangle}
variables {M : Point} (hM : M = centroid T)
variables {O : Point} (hO : O = incenter T)
variables {r : ℝ} (hr : r = inradius T)

-- Additional conditions
lemma div_medians (T : RightAngledTriangle) : divides_medians_in_ratio T M 2 1 := sorry
lemma ab_length (T : RightAngledTriangle) (α : ℝ) : length_AB T = r * (Real.cot (α/2) + Real.cot (π/4 - α/2)) := sorry
lemma co_length (T : RightAngledTriangle) : distance C O = r * Real.sqrt 2 := sorry
lemma mo_length (T : RightAngledTriangle) : distance O M = r := sorry

-- The final theorem
theorem find_acute_angles (T : RightAngledTriangle) (hM : M = centroid T) (hO : O = incenter T) (hr : r = inradius T) :
  (acute_angle1 T) = π / 4 + Real.arccos ((4 * Real.sqrt 6 - 3 * Real.sqrt 2) / 6) ∨
  (acute_angle2 T) = π / 4 - Real.arccos ((4 * Real.sqrt 6 - 3 * Real.sqrt 2) / 6) := sorry

end find_acute_angles_l398_398341


namespace complex_number_solution_l398_398684

theorem complex_number_solution (z : ℂ) (h : (1 - complex.I) * z = 3 + complex.I) :
  z = 1 + 2 * complex.I :=
by 
  sorry

end complex_number_solution_l398_398684


namespace angle_bisectors_lengths_l398_398226

noncomputable def hypotenuse (a b : ℕ) : ℕ := Float.toNat (Float.sqrt (a * a + b * b))

noncomputable def angle_bisector_to_hypotenuse (a b c : ℕ) : Float :=
  Float.sqrt (a * b * ((a + b + c) / (a + b - c)))

noncomputable def angle_bisector_from_leg (a b : ℕ) : Float :=
  Float.sqrt (a * b * (1 - ((Float.ofNat (a - b)^2) / (Float.ofNat a)^2)))

/- Right-angle triangle conditions -/
def leg1 : ℕ := 3
def leg2 : ℕ := 4
def hyp := hypotenuse leg1 leg2

/-- Prove the lengths of the angle bisectors -/
theorem angle_bisectors_lengths : 
  angle_bisector_to_hypotenuse leg1 leg2 hyp = 12 * Float.sqrt 2 / 7 ∧ 
  angle_bisector_from_leg leg1 hyp = 4 * Float.sqrt 10 / 3 ∧ 
  angle_bisector_from_leg leg2 hyp = 3 * Float.sqrt 5 / 2 :=
by
  sorry

end angle_bisectors_lengths_l398_398226


namespace sum_S_2017_l398_398599

namespace SequenceSum

def a (n : ℕ) : ℤ := (-1)^(n-1) * (n + 1)

def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1) | i > 0, a i

theorem sum_S_2017 : S 2017 = 1010 := by
  sorry

end SequenceSum

end sum_S_2017_l398_398599


namespace contact_probability_l398_398063

theorem contact_probability (n m : ℕ) (p : ℝ) (h_n : n = 5) (h_m : m = 8) (hp : 0 ≤ p ∧ p ≤ 1) :
  (1 - (1 - p)^(n * m)) = 1 - (1 - p)^(40) :=
by
  rw [h_n, h_m]
  sorry

end contact_probability_l398_398063


namespace probability_sum_15_l398_398390

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398390


namespace probability_of_connection_l398_398070

theorem probability_of_connection (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  let num_pairs := 5 * 8 in
  let prob_no_connection := (1 - p) ^ num_pairs in
  1 - prob_no_connection = 1 - (1 - p) ^ 40 := 
by
  let num_pairs := 5 * 8
  have h_num_pairs : num_pairs = 40 := by norm_num
  rw h_num_pairs
  let prob_no_connection := (1 - p) ^ 40
  sorry

end probability_of_connection_l398_398070


namespace grade_appears_no_more_than_twice_l398_398870

variable {grades : Fin 13 → Fin 4}  -- Represent 13 grades, each being one of {2, 3, 4, 5}
variable count : ∀ g : Fin 4, Fin 13 -- Count occurrences of each grade
variable h_sum : ∑ i, (grades i) = sum -- Sum of grades
variable mean_is_int : ∑ i, (grades i) % 13 = 0  -- Mean is an integer

theorem grade_appears_no_more_than_twice :
  ∃ g : Fin 4, count g ≤ 2 := sorry

end grade_appears_no_more_than_twice_l398_398870


namespace min_value_expression_l398_398601

theorem min_value_expression (a b : ℝ) (h : a + 2 * b = 1) : 
  2^a + 4^b ≥ 2 * Real.sqrt 2 :=
begin
  sorry
end

end min_value_expression_l398_398601


namespace simon_legos_l398_398728

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l398_398728


namespace event_B_more_likely_l398_398701

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l398_398701


namespace find_omega_at_max_coordinate_l398_398950

theorem find_omega_at_max_coordinate (ω : ℝ) : 
  (∀ x ∈ set.Ioo 0 (π / 2), tan x = sin (ω * x) -> x = π / 4 ∧ tan x = 1) →
  ω = 2 :=
by 
  sorry

end find_omega_at_max_coordinate_l398_398950


namespace quadratic_root_identity_l398_398198

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398198


namespace find_t_for_perpendicular_vectors_l398_398140

theorem find_t_for_perpendicular_vectors :
  ∃ t : ℝ, let a := (3, 1) and b := (t, -3) in (a.1 * b.1 + a.2 * b.2 = 0) ∧ t = 1 :=
by
  sorry

end find_t_for_perpendicular_vectors_l398_398140


namespace find_range_m_l398_398981

def p (m : ℝ) : Prop := m > 2 ∨ m < -2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_m (h₁ : ¬ p m) (h₂ : q m) : (1 : ℝ) < m ∧ m ≤ 2 :=
by sorry

end find_range_m_l398_398981


namespace tangent_slope_at_one_l398_398913

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_slope_at_one : deriv f 1 = 2 * Real.exp 1 := sorry

end tangent_slope_at_one_l398_398913


namespace alex_not_read_probability_l398_398342

def probability_reads : ℚ := 5 / 8
def probability_not_reads : ℚ := 3 / 8

theorem alex_not_read_probability : (1 - probability_reads) = probability_not_reads := 
by
  sorry

end alex_not_read_probability_l398_398342


namespace value_of_a_l398_398420

theorem value_of_a (a : ℝ) : 
  (∀ (x : ℝ), (x < -4 ∨ x > 5) → x^2 + a * x + 20 > 0) → a = -1 :=
by
  sorry

end value_of_a_l398_398420


namespace find_k_range_m_l398_398587

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f (x k : ℝ) : ℝ := real.logb 4 (4^x + 1) + k * x

theorem find_k (h : is_even_function (λ x, f x k)) : k = 0 := sorry

theorem range_m (h : is_even_function (λ x, f x 0)) (m : ℝ) :
  ∃ x, f x 0 = m → m >= 1/2 := sorry

end find_k_range_m_l398_398587


namespace quadratic_no_real_roots_iff_m_gt_one_l398_398144

theorem quadratic_no_real_roots_iff_m_gt_one (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 :=
sorry

end quadratic_no_real_roots_iff_m_gt_one_l398_398144


namespace color_schemes_equivalence_l398_398407

noncomputable def number_of_non_equivalent_color_schemes (n : Nat) : Nat :=
  let total_ways := Nat.choose (n * n) 2
  -- Calculate the count for non-diametrically opposite positions (4 rotations)
  let non_diametric := (total_ways - 24) / 4
  -- Calculate the count for diametrically opposite positions (2 rotations)
  let diametric := 24 / 2
  -- Sum both counts
  non_diametric + diametric

theorem color_schemes_equivalence (n : Nat) (h : n = 7) : number_of_non_equivalent_color_schemes n = 300 :=
  by
    rw [h]
    sorry

end color_schemes_equivalence_l398_398407


namespace museum_trip_l398_398347

theorem museum_trip (bus1 bus2 bus3 bus4 : ℕ) (h1 : bus1 = 12) 
(h2 : bus2 = 2 * bus1) (h3 : bus3 = bus2 - 6) (h4 : bus4 = bus1 + 9) 
(h5 : ∀ (n : ℕ), n ∈ {bus1, bus2, bus3, bus4} → n ≤ 45) :
  ¬(∃ (n : ℕ), n ∈ {bus1, bus2, bus3, bus4} → n > 45) ∧ 
  bus1 + bus2 + bus3 + bus4 = 75 :=
by
  have h6 : bus2 = 24, from h2.symm ▸ h1 ▸ (by norm_num),
  have h7 : bus3 = 18, from h3.symm ▸ h6 ▸ (by norm_num),
  have h8 : bus4 = 21, from h4.symm ▸ h1 ▸ (by norm_num),
  split,
  { intro hcon,
    cases hcon with n hn,
    cases hn,
    all_goals {linarith}, },
  { dsimp at *,
    rw [h1, h6, h7, h8], norm_num }

end museum_trip_l398_398347


namespace hexagon_side_relation_l398_398221

noncomputable def hexagon (a b c d e f : ℝ) :=
  ∃ (i j k l m n : ℝ), 
    i = 120 ∧ j = 120 ∧ k = 120 ∧ l = 120 ∧ m = 120 ∧ n = 120 ∧  
    a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f ∧ f = a

theorem hexagon_side_relation
  (a b c d e f : ℝ)
  (ha : hexagon a b c d e f) :
  d - a = b - e ∧ b - e = f - c :=
by
  sorry

end hexagon_side_relation_l398_398221


namespace compare_log_values_l398_398113

theorem compare_log_values (a b c : ℝ) 
  (ha : a = log 5 (1 / 3)) 
  (hb : b = (1 : ℝ) / (1 / 6)) 
  (hc : c = log 5 4) : a < c ∧ c < b :=
by {
  sorry
}

end compare_log_values_l398_398113


namespace infinite_sequence_of_reals_contains_monotonic_subsequence_l398_398721

open Classical

theorem infinite_sequence_of_reals_contains_monotonic_subsequence
  (u : ℕ → ℝ) (h : ∀ n : ℕ, u n ∈ ℝ) :
  ∃ a : ℕ → ℕ, StrictMono a ∨ StrictAnti a :=
sorry

end infinite_sequence_of_reals_contains_monotonic_subsequence_l398_398721


namespace total_pens_bought_l398_398286

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 := 
sorry

end total_pens_bought_l398_398286


namespace cos_sum_identity_l398_398719

theorem cos_sum_identity (α : ℝ) : 
  cos α + cos (2 * α) + cos (6 * α) + cos (7 * α) = 4 * cos (α / 2) * cos (5 * α / 2) * cos (4 * α) := 
sorry

end cos_sum_identity_l398_398719


namespace sergey_min_cost_l398_398642

noncomputable def min_cost : ℕ := 3800

theorem sergey_min_cost :
  ∃ x y : ℕ, (3 * x + 20 * y = 15 * x + 5 * y) ∧ (5 ∣ x) ∧ (5 * 360 + 4 * 500 = min_cost) :=
by { use [5, 4], split, 
     { ring, },
     split,
     { exact dvd_refl 5, },
     { norm_num, } }

end sergey_min_cost_l398_398642


namespace max_triangle_area_l398_398235
noncomputable def parametric_curve := 
 {α : ℝ × ℝ // α.1 = sqrt 3 + 2 * cos α.2 ∧ α.2 = 1 + 2 * sin α.2}

noncomputable def polar_curve (ρ θ : ℝ) := ρ = 4 * sin (θ + π / 3)

theorem max_triangle_area 
 (α θ : ℝ) 
 (A B : α → ℝ × ℝ) 
 (h_param : A α = (sqrt 3 + 2 * cos α, 1 + 2 * sin α))
 (h_polar_eq : ∀ ρ, polar_curve ρ θ)
 (angle_AOB : A θ.1 = B (θ + π / 3)) :
 {δ : ℝ // δ = 3 * sqrt 3} :=
begin
  sorry
end

end max_triangle_area_l398_398235


namespace average_visitors_other_days_l398_398450

theorem average_visitors_other_days 
  (avg_sunday : ℕ) (avg_day : ℕ)
  (num_days : ℕ) (sunday_offset : ℕ)
  (other_days_count : ℕ) (total_days : ℕ) 
  (total_avg_visitors : ℕ)
  (sunday_avg_visitors : ℕ) :
  avg_sunday = 150 →
  avg_day = 125 →
  num_days = 30 →
  sunday_offset = 5 →
  total_days = 30 →
  total_avg_visitors * total_days =
    (sunday_offset * sunday_avg_visitors) + (other_days_count * avg_sunday) →
  125 = total_avg_visitors →
  150 = sunday_avg_visitors →
  other_days_count = num_days - sunday_offset →
  (125 * 30 = (5 * 150) + (other_days_count * avg_sunday)) →
  avg_sunday = 120 :=
by
  sorry

end average_visitors_other_days_l398_398450


namespace probability_two_cards_sum_15_from_standard_deck_l398_398397

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398397


namespace equations_of_lines_at_distance_l398_398330

theorem equations_of_lines_at_distance
  (l : affine.affineSpace ℝ ℝ)
  (x y : ℝ → ℝ)
  (c : ℝ)
  (dist_sqrt2_div2 : ℝ := real.sqrt 2 / 2) :
  (line_eqn : affine.affineSpace ℝ ℝ := λ p, x p - y p + 1 = 0) → 
  (∀ p ∈ line_eqn, abs (x p - y p + c - 1) / real.sqrt (x p ^ 2 + y p ^ 2) = dist_sqrt2_div2) →
  (c = 2 ∨ c = 0) → 
  (∀ p, (x p - y p + 2 = 0 ∨ x p - y p = 0)) := 
sorry

end equations_of_lines_at_distance_l398_398330


namespace shop_owner_gain_l398_398867

-- Define the conditions
def cost_price_silk_per_meter : ℕ := 30
def cost_price_linen_per_meter : ℕ := 45
def meters_silk_sold : ℕ := 15
def meters_linen_sold : ℕ := 20
def selling_price_equiv_meters_silk : ℕ := 10
def percent_profit_linen : ℕ := 20

-- Define the proof problem
theorem shop_owner_gain :
  let cost_price_silk := meters_silk_sold * cost_price_silk_per_meter in
  let selling_price_silk := (selling_price_equiv_meters_silk * cost_price_silk_per_meter) + cost_price_silk in
  let cost_price_linen := meters_linen_sold * cost_price_linen_per_meter in
  let profit_linen := (percent_profit_linen * cost_price_linen) / 100 in
  let selling_price_linen := cost_price_linen + profit_linen in
  let total_cost_price := cost_price_silk + cost_price_linen in
  let total_selling_price := selling_price_silk + selling_price_linen in
  let total_gain := total_selling_price - total_cost_price in
  (total_gain * 100 / total_cost_price : ℚ) = 35.56 :=
begin
  sorry
end

end shop_owner_gain_l398_398867


namespace min_rooms_needed_l398_398032

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398032


namespace sufficient_but_not_necessary_condition_l398_398940

def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0
def condition_q (x : ℝ) : Prop := |x - 2| < 1

theorem sufficient_but_not_necessary_condition : 
  (∀ x : ℝ, condition_p x → condition_q x) ∧ ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l398_398940


namespace number_of_scheduling_plans_l398_398794

def num_days : ℕ := 5
def num_exams : ℕ := 2

def valid_scheduling (s : List ℕ) : Prop :=
  s.length = num_exams ∧
  ∀ (i j: ℕ), i < j → s[i] < s[j] - 1

theorem number_of_scheduling_plans : 
  (∃ s : List (Fin num_days), valid_scheduling s) → (num_days.choose num_exams) - (num_days - num_exams) * 2 = 12 :=
sorry

end number_of_scheduling_plans_l398_398794


namespace find_a_11_l398_398097

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = n

theorem find_a_11 (a : ℕ → ℤ) (h : sequence a) : a 11 = 58 :=
by
  -- proof steps will go here
  sorry

end find_a_11_l398_398097


namespace problem1_arithmetic_sequence_problem2_max_k_problem2_k_is_3_l398_398973

-- Define the sequence a_n
def seq_a : ℕ → ℚ
| 0       := 2
| (n + 1) :=
  let S_n := (list.sum (list.map seq_a (list.range (n + 1)))) in
  -((S_n - 1) * (S_n - 1) / S_n)

-- Define the sum S_n of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℚ := (list.sum (list.map seq_a (list.range n)))

-- Problem 1: Prove that { 1 / (S n - 1) } is an arithmetic sequence
theorem problem1_arithmetic_sequence : ∃ d : ℚ, ∀ n : ℕ, 
  (1 / (S (n + 1) - 1)) - (1 / (S n - 1)) = d := sorry

-- Problem 2: Prove that the maximum value of k such that
-- (∏ i in finset.range n, S (i + 1) + 1) ≥ k * n is 3
theorem problem2_max_k (k : ℚ) : (∀ n : ℕ, ∏ i in finset.range n, (S (i + 1) + 1) ≥ k * (n : ℚ)) → k ≤ 3 := sorry

-- Verify that k = 3 satisfies the inequality
theorem problem2_k_is_3 : (∀ n : ℕ, ∏ i in finset.range n, (S (i + 1) + 1) ≥ 3 * (n : ℚ)) := sorry

end problem1_arithmetic_sequence_problem2_max_k_problem2_k_is_3_l398_398973


namespace focus_of_parabola_l398_398327

-- Define the equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8 * y

-- Define the focus of the parabola
def focus (x y : ℝ) : Prop := x = 0 ∧ y = 2

-- Prove that the coordinates of the focus of the parabola x^2 = 8y are (0, 2)
theorem focus_of_parabola : ∀ x y : ℝ, parabola x y → focus x y :=
by {
  intros x y h,
  sorry
}

end focus_of_parabola_l398_398327


namespace volume_of_inscribed_cube_l398_398895

-- Define the necessary properties and conditions of the pyramid and inscribed cube
axiom pyramid_base_side_length : ℝ
axiom pyramid_lateral_face_height : ℝ
axiom cube_volume : ℝ

-- Given conditions
def is_regular_hexagonal_base := pyramid_base_side_length = 2
def lateral_faces_are_isosceles := pyramid_lateral_face_height = 3
def cube_is_perfectly_inscribed := true

-- Prove the volume of the cube
theorem volume_of_inscribed_cube :
  is_regular_hexagonal_base ∧
  lateral_faces_are_isosceles ∧
  cube_is_perfectly_inscribed →
  cube_volume = 2 * real.sqrt 2 :=
by
  intro h,
  sorry

end volume_of_inscribed_cube_l398_398895


namespace winning_strategy_l398_398337

def number_on_blackboard : ℕ := 10 ^ 2007

def valid_operations (x : ℕ) (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ a * b = x

def can_replace (x : ℕ) : Prop :=
  ∃ a b, valid_operations x a b

def can_strike_off (x : ℕ) (y : ℕ) : Prop := 
  x = y ∨ x > 1 ∧ y > 1

def operation_possible (xs : list ℕ) : Prop :=
  ∃ x y, x ∈ xs ∧ y ∈ xs ∧ (valid_operations x y y ∨ can_strike_off x y)

def anne_has_winning_strategy : Prop :=
  ∀ board_state : list ℕ,
    number_on_blackboard ∈ board_state →
    ¬ operation_possible board_state →
    (anne_wins : ∃ n, valid_operations number_on_blackboard n n)

noncomputable def winning_strategy_for_anne : Prop :=
  anne_has_winning_strategy

-- ◾ Anne has a winning strategy
theorem winning_strategy (h : anne_has_winning_strategy) : winning_strategy_for_anne := sorry

end winning_strategy_l398_398337


namespace final_apples_quantity_l398_398348

variable (Q_initial Q_sold Q_bought : ℕ)

theorem final_apples_quantity (h1 : Q_initial = 280) (h2 : Q_sold = 132) (h3 : Q_bought = 145) :
  Q_initial - Q_sold + Q_bought = 293 :=
by
  rw [h1, h2, h3]
  sorry

end final_apples_quantity_l398_398348


namespace cubic_equation_of_quadratic_roots_l398_398544

theorem cubic_equation_of_quadratic_roots (p q x₁ x₂ : ℝ) (h : x₁^2 + p * x₁ + q = 0) (h' : x₂^2 + p * x₂ + q = 0) :
    Polynomial.eval y (Polynomial.Cubic.mk p q x₁ x₂) = y^3 - (p^2 - q) * y^2 + (p^2 * q - q^2) * y - q^3 :=
by
  sorry

end cubic_equation_of_quadratic_roots_l398_398544


namespace max_f_value_l398_398122

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (Real.pi / 2 + x) + Real.sin (Real.pi / 2 + x) ^ 2

theorem max_f_value : ∀ x ∈ Icc (-Real.pi) 0, f x ≤ 5 / 4 :=
begin
  sorry
end

end max_f_value_l398_398122


namespace quadratic_roots_identity_l398_398176

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398176


namespace event_B_more_likely_than_event_A_l398_398711

/-- Define the outcomes when rolling a die three times --/
def total_outcomes : ℕ := 6 ^ 3

/-- Define the number of ways to choose 3 different numbers from 6 --/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Define the number of ways to arrange 3 different numbers --/
def arrangements_3 : ℕ := 3.factorial

/-- Calculate the number of favorable outcomes for event B --/
def favorable_B : ℕ := choose_3_from_6 * arrangements_3

/-- Define the probability of event B --/
noncomputable def prob_B : ℝ := favorable_B / total_outcomes

/-- Define the probability of event A as the complement of event B --/
noncomputable def prob_A : ℝ := 1 - prob_B

/-- The theorem to prove that event B is more likely than event A --/
theorem event_B_more_likely_than_event_A : prob_B > prob_A :=
by
  sorry

end event_B_more_likely_than_event_A_l398_398711


namespace number_of_solutions_l398_398997

theorem number_of_solutions (n : ℤ) : 
  (∃ (count : ℤ), count = 9 ∧ 
    (∀ k, 5 < k ∧ k < 15 → k ∈ Ico 1 n ∧ (k + 10) * (k - 5) * (k - 15) < 0) ∧ 
    (∀ m, m ∈ Ico 1 n → (5 < m ∧ m < 15) ∨ ¬((m + 10) * (m - 5) * (m - 15) < 0))) :=
by
  -- We'll skip the proof part.
  sorry

end number_of_solutions_l398_398997


namespace trapezium_height_l398_398042

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end trapezium_height_l398_398042


namespace macaroon_problem_l398_398546

def total_macaroons_remaining (red_baked green_baked red_ate green_ate : ℕ) : ℕ :=
  (red_baked - red_ate) + (green_baked - green_ate)

theorem macaroon_problem :
  let red_baked := 50 in
  let green_baked := 40 in
  let green_ate := 15 in
  let red_ate := 2 * green_ate in
  total_macaroons_remaining red_baked green_baked red_ate green_ate = 45 :=
by
  sorry

end macaroon_problem_l398_398546


namespace find_radius_l398_398568

theorem find_radius (θ : ℝ) (r : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2)
  (h : ∃ l, l = { x // ∃ y, y = 2 * x - 2 ∧ (x = y)} ∧
        ∃ A : ℝ × ℝ, A = (1,4) ∧
        (∃ C : ℝ × ℝ, C = (-1, 2) ∧
        ∀ P : ℝ × ℝ, P = C ∧
        (∃ l : ℝ × ℝ → Prop, ( ∀ (x y : ℝ), l (x,y) = (x+1)^2+(y-2)^2 ) ∧
        (∀ P, (∃ (d : ℝ), d = abs((-2 -2 +2)/ sqrt(5)) ∧ 
        (d = r ∧ r > 0) )))) :
  r = 2 * Real.sqrt 5 / 5 := 
sorry

end find_radius_l398_398568


namespace axis_of_symmetry_l398_398965
open Real

def f (x : ℝ) : ℝ := sin x * cos (2 * x)

theorem axis_of_symmetry (x : ℝ) : 
  (x = -π / 4 → f (π / 2 + (-π / 4 - x)) ≠ f (π / 2 + x)) ∨
  (x = 0 → f (x) ≠ f (-x)) ∨
  (x = π / 4 → f (π / 2 + (π / 4 - x)) ≠ f (π / 2 + x)) ∨
  (x = π / 2 → f (π - x) = f (x)) :=
by 
  sorry

end axis_of_symmetry_l398_398965


namespace probability_sum_15_l398_398389

theorem probability_sum_15 :
  let total_cards := 52
  let valid_numbers := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := { (6, 9), (7, 8), (8, 7) }
  let probability := (16 + 16 + 12) / (52 * 51)
  probability = 11 / 663 :=
by
  sorry

end probability_sum_15_l398_398389


namespace find_x_l398_398527

theorem find_x :
  ∀ x : ℝ, (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
  8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) →
  x = 1486 / 225 :=
by
  sorry

end find_x_l398_398527


namespace pipe_fill_time_without_leak_l398_398857

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : T > 0) 
  (h2 : 1/T - 1/8 = 1/8) :
  T = 4 := 
sorry

end pipe_fill_time_without_leak_l398_398857


namespace soccer_shoes_cost_l398_398256

noncomputable def cost_of_socks : ℝ := 9.50
def num_pairs_of_socks : ℕ := 2
def total_sock_cost : ℝ := num_pairs_of_socks * cost_of_socks
def jack_has : ℝ := 40
def jack_needs_more : ℝ := 71
def total_required : ℝ := jack_has + jack_needs_more
def cost_of_shoes : ℝ := total_required - total_sock_cost

theorem soccer_shoes_cost :
  cost_of_shoes = 92 :=
by
  rw [cost_of_shoes, total_required, total_sock_cost, num_pairs_of_socks, cost_of_socks]
  norm_num
  sorry

end soccer_shoes_cost_l398_398256


namespace full_pages_l398_398726

theorem full_pages (total_photos : ℕ) (photos_per_page : ℕ) : 
  total_photos = 2176 → photos_per_page = 12 → total_photos / photos_per_page = 181 :=
by
  intro h1 h2
  rw [h1, h2]
  -- proof to be completed
  sorry

end full_pages_l398_398726


namespace base_three_to_base_ten_l398_398748

theorem base_three_to_base_ten (n : ℕ) (h : n = 20121) : 
  let convert := 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 in
  convert = 178 :=
by
  have : convert = 162 + 0 + 9 + 6 + 1, by sorry
  show convert = 178, by sorry

end base_three_to_base_ten_l398_398748


namespace problem1_problem2_problem3_l398_398125

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x + 2 * x

theorem problem1 (a : ℝ) : a = 1 → ∀ x : ℝ, f 1 x = x^2 - 3 * x + Real.log x → 
  (∀ x : ℝ, f 1 1 = -2) :=
by sorry

theorem problem2 (a : ℝ) (h : 0 < a) : (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → a ≥ 1 :=
by sorry

theorem problem3 (a : ℝ) : (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) → 0 ≤ a ∧ a ≤ 8 :=
by sorry

end problem1_problem2_problem3_l398_398125


namespace match_foci_of_parabola_and_hyperbola_l398_398949

noncomputable def focus_of_parabola (a : ℝ) : ℝ :=
a / 4

noncomputable def foci_of_hyperbola : Set ℝ :=
{2, -2}

theorem match_foci_of_parabola_and_hyperbola (a : ℝ) :
  focus_of_parabola a ∈ foci_of_hyperbola ↔ a = 8 ∨ a = -8 :=
by
  -- This is the placeholder for the proof.
  sorry

end match_foci_of_parabola_and_hyperbola_l398_398949


namespace emma_bank_account_balance_l398_398523

theorem emma_bank_account_balance
  (initial_balance : ℕ)
  (daily_spend : ℕ)
  (days_in_week : ℕ)
  (unit_bill : ℕ) :
  initial_balance = 100 → daily_spend = 8 → days_in_week = 7 → unit_bill = 5 →
  (initial_balance - daily_spend * days_in_week) % unit_bill = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end emma_bank_account_balance_l398_398523


namespace line_through_point_l398_398085

theorem line_through_point (k : ℝ) : (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end line_through_point_l398_398085


namespace fraction_sum_product_roots_of_quadratic_l398_398164

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l398_398164


namespace M_cap_N_l398_398975

open Set

noncomputable def M : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def N : Set ℤ := {x | -2 < x ∧ x < 5 ∧ (∀ y, y = x ∧ y ∈ ℤ)}

theorem M_cap_N :
  M ∩ N = {1, 2, 3, 4} :=
sorry

end M_cap_N_l398_398975


namespace R2_area_is_160_l398_398109

-- Define the initial conditions.
structure Rectangle :=
(width : ℝ)
(height : ℝ)

def R1 : Rectangle := { width := 4, height := 8 }

def similar (r1 r2 : Rectangle) : Prop :=
  r2.width / r2.height = r1.width / r1.height

def R2_diagonal := 20

-- Proving that the area of R2 is 160 square inches
theorem R2_area_is_160 (R2 : Rectangle)
  (h_similar : similar R1 R2)
  (h_diagonal : R2.width^2 + R2.height^2 = R2_diagonal^2) :
  R2.width * R2.height = 160 :=
  sorry

end R2_area_is_160_l398_398109


namespace maximize_profit_l398_398441

-- Define the conditions
variables (cost_price initial_price : ℝ)
variable (items_sold : ℝ)
variable (price_decrease items_increase : ℝ)

-- Assume the given conditions
def conditions : Prop :=
  cost_price = 60 ∧
  initial_price = 90 ∧
  items_sold = 30 ∧
  price_decrease = 1 ∧
  items_increase = 1

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  let selling_price := initial_price - x
  let number_of_items_sold := items_sold + x
  (selling_price - cost_price) * number_of_items_sold

-- Prove that the selling price at which profit is maximized is 90 yuan
theorem maximize_profit (h : conditions cost_price initial_price items_sold price_decrease items_increase) : 
  (∀ x : ℝ, profit cost_price initial_price items_sold x price_decrease items_increase ≤ profit cost_price initial_price items_sold 0 price_decrease items_increase) :=
sorry

end maximize_profit_l398_398441


namespace total_cost_of_fencing_l398_398051

noncomputable def pi : ℝ := Real.pi
def diameter : ℝ := 32
def cost_per_meter : ℝ := 2
def circumference (d : ℝ) : ℝ := pi * d

theorem total_cost_of_fencing : cost_per_meter * circumference diameter = 201.06 :=
by
  let C := circumference diameter
  have h1 : C = pi * diameter := rfl
  have h2 : C = 100.53 := by -- Here we would use lemmas that approximate the value of pi and calculations
    sorry
  show cost_per_meter * C = 201.06, from
    calc
      cost_per_meter * C = cost_per_meter * 100.53 : by rw h2
      ... = 201.06 : by norm_num

end total_cost_of_fencing_l398_398051


namespace work_done_correct_l398_398459

-- Define the force function
def F (x : ℝ) : ℝ := 3 * x^2

-- Define the work done by the force F(x) over the interval [0, 4]
def work_done : ℝ := ∫ x in 0..4, F x

-- The theorem asserting that the work done is 64 Joules
theorem work_done_correct : work_done = 64 := by
  sorry

end work_done_correct_l398_398459


namespace tan_value_expression_value_l398_398573

-- Definitions and conditions
variables (α : ℝ)
-- Condition 1: α is in the fourth quadrant implies that sin(α) is negative, and cos(α) is positive.
-- Condition 2: cos(α) = 3/5

-- First statement: Prove that tan(α) = -4/3
theorem tan_value (h1 : real.cos α = 3/5) (h2 : α ∈ Icc (3*π/2) (2*π)) :
  real.tan α = -4/3 :=
sorry

-- Second statement: Prove the expression evaluates to -1
theorem expression_value (h1 : real.cos α = 3/5) (h2 : α ∈ Icc (3*π/2) (2*π)) :
  (real.sin (3/2 * real.pi - α) + 2 * real.cos (α + real.pi / 2)) /
  (real.sin (α - real.pi) - 3 * real.cos (2 * real.pi - α)) = -1 :=
sorry

end tan_value_expression_value_l398_398573


namespace max_square_area_l398_398465

-- Define the dimensions of the rectangular sign
def rect_width : ℝ := 9
def rect_height : ℝ := 16

-- Define the border width
def border_width : ℝ := 1.5

-- Define the function to compute the maximum square side length
def max_square_side_length (width height border : ℝ) : ℝ :=
  min (width - 2 * border) (height - 2 * border)

-- Define the theorem to prove the max square area
theorem max_square_area : max_square_side_length rect_width rect_height border_width ^ 2 = 36 :=
by
  let side := max_square_side_length rect_width rect_height border_width
  have h1 : side = 6 := by linarith
  rw [h1]
  norm_num
  sorry

end max_square_area_l398_398465


namespace find_stock_face_value_l398_398816

theorem find_stock_face_value
  (cost_price : ℝ) -- Definition for the cost price
  (discount_rate : ℝ) -- Definition for the discount rate
  (brokerage_rate : ℝ) -- Definition for the brokerage rate
  (h1 : cost_price = 98.2) -- Condition: The cost price is 98.2
  (h2 : discount_rate = 0.02) -- Condition: The discount rate is 2%
  (h3 : brokerage_rate = 0.002) -- Condition: The brokerage rate is 1/5% (0.002)
  : ∃ X : ℝ, 0.982 * X = cost_price ∧ X = 100 := -- Theorem statement to prove
by
  -- Proof omitted
  sorry

end find_stock_face_value_l398_398816


namespace f_n_f_n_eq_n_l398_398265

def f : ℕ → ℕ := sorry
axiom f_def1 : f 1 = 1
axiom f_def2 : ∀ n ≥ 2, f n = n - f (f (n - 1))

theorem f_n_f_n_eq_n (n : ℕ) (hn : 0 < n) : f (n + f n) = n :=
by sorry

end f_n_f_n_eq_n_l398_398265


namespace event_B_more_likely_than_event_A_l398_398709

/-- Define the outcomes when rolling a die three times --/
def total_outcomes : ℕ := 6 ^ 3

/-- Define the number of ways to choose 3 different numbers from 6 --/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Define the number of ways to arrange 3 different numbers --/
def arrangements_3 : ℕ := 3.factorial

/-- Calculate the number of favorable outcomes for event B --/
def favorable_B : ℕ := choose_3_from_6 * arrangements_3

/-- Define the probability of event B --/
noncomputable def prob_B : ℝ := favorable_B / total_outcomes

/-- Define the probability of event A as the complement of event B --/
noncomputable def prob_A : ℝ := 1 - prob_B

/-- The theorem to prove that event B is more likely than event A --/
theorem event_B_more_likely_than_event_A : prob_B > prob_A :=
by
  sorry

end event_B_more_likely_than_event_A_l398_398709


namespace saved_percentage_this_year_l398_398663

variable (S : ℝ) -- Annual salary last year

-- Conditions
def saved_last_year := 0.06 * S
def salary_this_year := 1.20 * S
def saved_this_year := saved_last_year

-- The goal is to prove that the percentage saved this year is 5%
theorem saved_percentage_this_year :
  (saved_this_year / salary_this_year) * 100 = 5 :=
by sorry

end saved_percentage_this_year_l398_398663


namespace P2_coordinates_l398_398237

def point (x y : ℤ) : ℤ × ℤ := (x, y)

def translate_right (p : ℤ × ℤ) (n : ℤ) : ℤ × ℤ := (p.1 + n, p.2)

def rotate_clockwise_90 (p : ℤ × ℤ) : ℤ × ℤ := (p.2, -p.1)

def P : ℤ × ℤ := point (-5) 4

def P1 : ℤ × ℤ := translate_right P 8

def P2 : ℤ × ℤ := rotate_clockwise_90 P1

theorem P2_coordinates : P2 = (4, -3) :=
by
  unfold P
  unfold P1
  unfold P2
  simp
  -- proof omitted
  sorry

end P2_coordinates_l398_398237


namespace two_cards_totaling_15_probability_l398_398368

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398368


namespace decreasing_interval_of_f_l398_398335

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * log x

theorem decreasing_interval_of_f :
  (∀ x ∈ (Set.Ioc 0 1 : Set ℝ), 2*x - 2/x < 0) :=
by
  sorry

end decreasing_interval_of_f_l398_398335


namespace lim_Vn_over_n_l398_398668

noncomputable def u_k : ℕ → ℕ := sorry -- arbitrary sequence of integers

def V_n (n : ℕ) := {k | k ≤ n ∧ k ∈ (set.range u_k)}.card

theorem lim_Vn_over_n (h : ∑' k, 1 / (u_k k : ℝ) < ∞) :
  filter.tendsto (λ n, (V_n n : ℝ) / n) filter.at_top (nhds 0) :=
sorry

end lim_Vn_over_n_l398_398668


namespace smallest_number_proof_l398_398800

noncomputable def smallest_number (a b c : ℕ) :=
  let sum := a + b + c
  let mean := sum / 3
  let sorted := (list.sort (≤) [a, b, c])
  (sorted.head!, sorted.nth! 1, sorted.nth! 2)

theorem smallest_number_proof (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30)
  (h_med : (list.sort (≤) [a, b, c]).nth! 1 = 28)
  (h_largest : (list.sort (≤) [a, b, c]).nth! 2 = 28 + 6) :
  (list.sort (≤) [a, b, c]).head! = 28 :=
by
  sorry

end smallest_number_proof_l398_398800


namespace quadratic_roots_vieta_l398_398168

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398168


namespace lcm_gcd_product_lcm_gcd_product_three_l398_398832

-- Define the least common multiple (LCM) and greatest common divisor (GCD)
def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the LCM and GCD for three numbers
def lcm3 (a b c : ℕ) : ℕ := lcm (lcm a b) c
def gcd3 (a b c : ℕ) : ℕ := gcd (gcd a b) c

-- The first statement we want to prove
theorem lcm_gcd_product (a b : ℕ) : 
  lcm a b * gcd a b = a * b := sorry

-- The second statement we want to prove
theorem lcm_gcd_product_three (a b c : ℕ) : 
  (lcm3 a b c) * (gcd a b) * (gcd b c) * (gcd c a) / (gcd3 a b c) = a * b * c := sorry

end lcm_gcd_product_lcm_gcd_product_three_l398_398832


namespace porridge_amount_l398_398790

variables (x1 x2 x3 x4 x5 x6 : ℕ)
variables (T : ℕ)

-- Conditions from the problem
def condition1 := x3 = x1 + x2
def condition2 := x4 = x2 + x3
def condition3 := x5 = x3 + x4
def condition4 := x6 = x4 + x5
def condition5 := x5 = 10

-- The hypothesis that \( T \) is the total amount of porridge cooked
def total_porridge := T = x1 + x2 + x3 + x4 + x5 + x6

-- Proof statement
theorem porridge_amount : condition1 → condition2 → condition3 → condition4 → condition5 → total_porridge → T = 40 :=
by {
  intros,
  sorry,
}

end porridge_amount_l398_398790


namespace quadratic_root_identity_l398_398197

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398197


namespace ensemble_average_age_l398_398637

theorem ensemble_average_age (female_avg_age : ℝ) (num_females : ℕ) (male_avg_age : ℝ) (num_males : ℕ)
  (h1 : female_avg_age = 32) (h2 : num_females = 12) (h3 : male_avg_age = 40) (h4 : num_males = 18) :
  (num_females * female_avg_age + num_males * male_avg_age) / (num_females + num_males) =  36.8 :=
by sorry

end ensemble_average_age_l398_398637


namespace eval_expression_l398_398035

theorem eval_expression :
    (727 * 727) - (726 * 728) = 1 := by
  sorry

end eval_expression_l398_398035


namespace bamboo_probability_l398_398792

noncomputable def bambooLengths : List ℝ := [2.5, 2.6, 2.7, 2.8, 2.9]

def validPairs (l : List ℝ) (d : ℝ) : List (ℝ × ℝ) :=
  l.bind (λ x => l.filter (λ y => |x - y| = d).map (λ y => (x, y)))

def probabilityOfDiff (l : List ℝ) (d : ℝ) : ℝ :=
  (validPairs l d).length / (l.length.choose 2).toFloat

theorem bamboo_probability :
  probabilityOfDiff bambooLengths 0.3 = 0.2 :=
by
  sorry

end bamboo_probability_l398_398792


namespace two_cards_totaling_15_probability_l398_398366

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398366


namespace level_surfaces_and_value_l398_398054

-- Define the function
def func (x1 x2 x3 : ℝ) : ℝ :=
  Real.sqrt (36 - x1^2 - x2^2 - x3^2)

-- Prove the level surfaces and value of the function at point P(1, 1, 3)
theorem level_surfaces_and_value (x1 x2 x3 C : ℝ) (hx1 : x1 = 1) (hx2 : x2 = 1) (hx3 : x3 = 3) :
  (∃ (C : ℝ), func x1 x2 x3 = C ∧ (x1^2 + x2^2 + x3^2 = 36 - C^2)) ∧
  (func x1 x2 x3 = 5) :=
  by
    sorry

end level_surfaces_and_value_l398_398054


namespace event_B_more_likely_than_event_A_l398_398710

/-- Define the outcomes when rolling a die three times --/
def total_outcomes : ℕ := 6 ^ 3

/-- Define the number of ways to choose 3 different numbers from 6 --/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Define the number of ways to arrange 3 different numbers --/
def arrangements_3 : ℕ := 3.factorial

/-- Calculate the number of favorable outcomes for event B --/
def favorable_B : ℕ := choose_3_from_6 * arrangements_3

/-- Define the probability of event B --/
noncomputable def prob_B : ℝ := favorable_B / total_outcomes

/-- Define the probability of event A as the complement of event B --/
noncomputable def prob_A : ℝ := 1 - prob_B

/-- The theorem to prove that event B is more likely than event A --/
theorem event_B_more_likely_than_event_A : prob_B > prob_A :=
by
  sorry

end event_B_more_likely_than_event_A_l398_398710


namespace p_sufficient_not_necessary_for_q_l398_398273

variable (x : ℝ)

def p : Prop := x > 0
def q : Prop := x > -1

theorem p_sufficient_not_necessary_for_q : (p x → q x) ∧ ¬ (q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l398_398273


namespace proof_minimum_value_l398_398952

noncomputable def minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : Prop :=
  (1 / a + a / b) ≥ 1 + 2 * Real.sqrt 2

theorem proof_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : minimum_value_inequality a b h1 h2 h3 :=
  by
    sorry

end proof_minimum_value_l398_398952


namespace max_square_area_l398_398464

-- Define the dimensions of the rectangular sign
def rect_width : ℝ := 9
def rect_height : ℝ := 16

-- Define the border width
def border_width : ℝ := 1.5

-- Define the function to compute the maximum square side length
def max_square_side_length (width height border : ℝ) : ℝ :=
  min (width - 2 * border) (height - 2 * border)

-- Define the theorem to prove the max square area
theorem max_square_area : max_square_side_length rect_width rect_height border_width ^ 2 = 36 :=
by
  let side := max_square_side_length rect_width rect_height border_width
  have h1 : side = 6 := by linarith
  rw [h1]
  norm_num
  sorry

end max_square_area_l398_398464


namespace tenth_term_of_sequence_is_129_l398_398787

-- Define the sequence increment pattern
def sequence (n: ℕ) : ℕ :=
  match n with
  | 0 => 12
  | 1 => 13
  | 2 => 15
  | 3 => 17
  | 4 => 111
  | 5 => 113
  | 6 => 117
  | 7 => 119
  | 8 => 123
  | 9 => 123 + 6
  | _ => 0  -- Undefined for n >= 10 in the given pattern

-- The next term in sequence is 123 + 6, which equals 129.
theorem tenth_term_of_sequence_is_129 : sequence 9 = 129 := by
  unfold sequence
  sorry

end tenth_term_of_sequence_is_129_l398_398787


namespace length_of_first_object_is_correct_l398_398333

def length_of_first_object (a b : ℕ) : ℕ := Nat.gcd a b

theorem length_of_first_object_is_correct :
  ∀ (a b : ℕ), a = 225 → b = 780 → length_of_first_object a b = 15 :=
by
  intros a b ha hb
  rw [ha, hb]
  exact Nat.gcd_eq_right (by decide)
  exact sorry -- Proof of gcd calculation (omitting computational steps).

end length_of_first_object_is_correct_l398_398333


namespace socks_pairs_count_l398_398618

theorem socks_pairs_count :
  let white := 5
  let brown := 5
  let blue := 4
  let black := 2
  (nat.choose white 2 + nat.choose brown 2 + nat.choose blue 2 + nat.choose black 2) = 27 := by
  sorry

end socks_pairs_count_l398_398618


namespace third_player_game_count_l398_398798

theorem third_player_game_count (fp_games : ℕ) (sp_games : ℕ) (tp_games : ℕ) (total_games : ℕ) 
  (h1 : fp_games = 10) (h2 : sp_games = 21) (h3 : total_games = sp_games) 
  (h4 : total_games = fp_games + tp_games + 1): tp_games = 11 := 
  sorry

end third_player_game_count_l398_398798


namespace booking_rooms_needed_l398_398012

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398012


namespace number_of_packets_l398_398714

def ounces_in_packet : ℕ := 16 * 16 + 4
def ounces_in_ton : ℕ := 2500 * 16
def gunny_bag_capacity_in_ounces : ℕ := 13 * ounces_in_ton

theorem number_of_packets : gunny_bag_capacity_in_ounces / ounces_in_packet = 2000 :=
by
  sorry

end number_of_packets_l398_398714


namespace probability_sum_is_7_over_18_l398_398510

def first_die : list ℕ := [2, 3, 3, 4, 4, 5]
def second_die : list ℕ := [1, 2, 3, 5, 6, 7]

noncomputable def probability_sum_6_8_10 : ℚ :=
  let outcomes := (first_die.product second_die).filter (λ pair, pair.1 + pair.2 ∈ [6, 8, 10])
  (outcomes.length : ℚ) / (first_die.length * second_die.length : ℚ)

theorem probability_sum_is_7_over_18 :
  probability_sum_6_8_10 = 7 / 18 :=
sorry

end probability_sum_is_7_over_18_l398_398510


namespace roots_of_quadratic_eq_l398_398152

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398152


namespace true_discount_value_l398_398321

variables (BG R T TD : ℝ)
variables (h_bg : BG = 6)
variables (h_r : R = 12 / 100)
variables (h_t : T = 1)

theorem true_discount_value : TD = 50 :=
by
  -- Banker's Gain formula
  have bg_formula : BG = (TD * R * T),
  from sorry, -- Banker's Gain is the interest on True Discount

  -- Substituting known values
  have sub_values : 6 = TD * (12 / 100) * 1,
  from sorry,

  -- Solving for TD
  have solve_td : 6 * 100 / 12 = TD,
  from sorry,

  exact solve_td

end true_discount_value_l398_398321


namespace area_of_rectangle_A_is_88_l398_398291

theorem area_of_rectangle_A_is_88 
  (lA lB lC w wC : ℝ)
  (h1 : lB = lA + 2)
  (h2 : lB * w = lA * w + 22)
  (h3 : wC = w - 4)
  (AreaB : ℝ := lB * w)
  (AreaC : ℝ := lB * wC)
  (h4 : AreaC = AreaB - 40) : 
  (lA * w = 88) :=
sorry

end area_of_rectangle_A_is_88_l398_398291


namespace problem1_l398_398495

theorem problem1 (a b : ℝ) : 
  ((-2 * a) ^ 3 * (- (a * b^2)) ^ 3 - 4 * a * b^2 * (2 * a^5 * b^4 + (1 / 2) * a * b^3 - 5)) / (-2 * a * b) = a * b^4 - 10 * b :=
sorry

end problem1_l398_398495


namespace measure_angle_ADC_l398_398643

variable (A B C D : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Definitions for the angles
variable (angle_ABC angle_BCD angle_ADC : ℝ)

-- Conditions for the problem
axiom Angle_ABC_is_4_times_Angle_BCD : angle_ABC = 4 * angle_BCD
axiom Angle_BCD_ADC_sum_to_180 : angle_BCD + angle_ADC = 180

-- The theorem that we want to prove
theorem measure_angle_ADC (Angle_ABC_is_4_times_Angle_BCD: angle_ABC = 4 * angle_BCD)
    (Angle_BCD_ADC_sum_to_180: angle_BCD + angle_ADC = 180) : 
    angle_ADC = 144 :=
by
  sorry

end measure_angle_ADC_l398_398643


namespace algebraic_expression_l398_398552

noncomputable def given_condition (a : ℝ) : Prop := a + a⁻¹ = 3

theorem algebraic_expression (a : ℝ) (h : given_condition a) : a^2 + a⁻2 = 7 :=
by
  sorry

end algebraic_expression_l398_398552


namespace quadratic_roots_vieta_l398_398172

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398172


namespace lifespan_averaged_value_probability_of_replacement_l398_398440

open Real

noncomputable def normal_lifespan (mu sigma : ℝ) : Measure ℝ := NormalPDF mu sigma 

def lifespan_average (mu sigma : ℝ) : Prop :=
  let ξ := normal_lifespan mu sigma
  (Prob(ξ, fun t => t ≥ 12) = 0.8) →
  (Prob(ξ, fun t => t ≥ 24) = 0.2) →
  mu = 18

def replacement_probability (mu sigma : ℝ) (n : ℕ) : Prop :=
  let ξ := normal_lifespan mu sigma
  (Prob(ξ, fun t => t ≥ 12) = 0.8) →
  (Prob(ξ, fun t => t ≥ 24) = 0.2) →
  let prob_fail := 1 - 0.8
  let η := Binomial n prob_fail
  Prob(η, fun k => k ≥ 2) = 0.1808

theorem lifespan_averaged_value : lifespan_average 18 _
:= sorry

theorem probability_of_replacement : replacement_probability 18 _ 4
:= sorry

end lifespan_averaged_value_probability_of_replacement_l398_398440


namespace contact_probability_l398_398065

theorem contact_probability (n m : ℕ) (p : ℝ) (h_n : n = 5) (h_m : m = 8) (hp : 0 ≤ p ∧ p ≤ 1) :
  (1 - (1 - p)^(n * m)) = 1 - (1 - p)^(40) :=
by
  rw [h_n, h_m]
  sorry

end contact_probability_l398_398065


namespace problem_condition_l398_398081

theorem problem_condition (a : ℝ) (x : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) :
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end problem_condition_l398_398081


namespace minimum_rooms_needed_fans_l398_398019

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398019


namespace oil_amount_correct_l398_398358

-- Definitions based on the conditions in the problem
def initial_amount : ℝ := 0.16666666666666666
def additional_amount : ℝ := 0.6666666666666666
def final_amount : ℝ := 0.8333333333333333

-- Lean 4 statement to prove the given problem
theorem oil_amount_correct :
  initial_amount + additional_amount = final_amount :=
by
  sorry

end oil_amount_correct_l398_398358


namespace sum_first_10_terms_a_b_n_l398_398578

noncomputable def a (n : ℕ) : ℕ := a₁ + (n - 1)
noncomputable def b (n : ℕ) : ℕ := b₁ + (n - 1)
noncomputable def a_b_n (n : ℕ) : ℕ := a (b n)

axiom a₁_pos : a₁ ∈ ℕ ∧ 0 < a₁
axiom b₁_pos : b₁ ∈ ℕ ∧ 0 < b₁
axiom sum_eq_5 : a₁ + b₁ = 5
axiom a₁_gt_b₁ : a₁ > b₁

theorem sum_first_10_terms_a_b_n : 
  ∑ i in Finset.range 10, a_b_n i = 85 :=
sorry

end sum_first_10_terms_a_b_n_l398_398578


namespace minimum_value_exists_l398_398532

noncomputable def minimized_function (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

theorem minimum_value_exists :
  ∃ (x y : ℝ), minimized_function x y = minimized_function (4/3 - 2 * y/3) y :=
sorry

end minimum_value_exists_l398_398532


namespace quadratic_root_identity_l398_398203

theorem quadratic_root_identity : 
  (∀ x : ℝ, x^2 - 2 * x - 8 = 0 → (∃ x1 x2 : ℝ, x^2 - 2 * x - 8 = (x - x1) * (x - x2))) → 
  let x1 x2 : ℝ := classical.some (some_spec (quadratic_root_identity _)) in
  x1 + x2 = 2 ∧ x1 * x2 = -8 → 
  (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_root_identity_l398_398203


namespace vector_magnitude_sum_l398_398608

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_magnitude_sum
  (a b c : V)
  (h1 : a + b + c = 0)
  (h2 : (a - b) ⬝ c = 0)
  (h3 : a ⬝ b = 0)
  (h4 : ∥a∥ = 1) :
  ∥a∥^2 + ∥b∥^2 + ∥c∥^2 = 4 :=
begin
  sorry
end

end vector_magnitude_sum_l398_398608


namespace original_number_l398_398854

theorem original_number (x : ℝ) (h : x + 0.5 * x = 90) : x = 60 :=
by
  sorry

end original_number_l398_398854


namespace nth_equation_l398_398288

theorem nth_equation (n : ℕ) : ((list.finRange n).map (λ i, n + (i + 1))).prod = (2^n) * ((list.finRange n).map (λ i, 2 * i + 1)).prod :=
by
  sorry

end nth_equation_l398_398288


namespace minimum_rooms_needed_fans_l398_398021

def total_fans : Nat := 100
def fans_per_room : Nat := 3

def number_of_groups : Nat := 6
def fans_per_group : Nat := total_fans / number_of_groups
def remainder_fans_in_group : Nat := total_fans % number_of_groups

theorem minimum_rooms_needed_fans :
  fans_per_group * number_of_groups + remainder_fans_in_group = total_fans → 
  number_of_rooms total_fans ≤ 37 :=
sorry

def number_of_rooms (total_fans : Nat) : Nat :=
  let base_rooms := total_fans / fans_per_room
  let extra_rooms := if total_fans % fans_per_room > 0 then 1 else 0
  base_rooms + extra_rooms

end minimum_rooms_needed_fans_l398_398021


namespace event_B_more_likely_l398_398704

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l398_398704


namespace find_x_given_f_eq_3_l398_398589

theorem find_x_given_f_eq_3 (f : ℝ → ℝ)  (hf : ∀ x, f x = real.sqrt (x - 1)) (h : f 10 = 3) :  10 = 10 :=
by
  sorry

end find_x_given_f_eq_3_l398_398589


namespace quadratic_roots_identity_l398_398177

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398177


namespace sin_x_solution_l398_398619

variables {x : ℝ}

theorem sin_x_solution (h1 : sin x = -3/5) (h2 : π < x ∧ x < 3/2 * π) :
  x = π + Real.arcsin (3/5) :=
sorry

end sin_x_solution_l398_398619


namespace remaining_ball_is_green_l398_398795

theorem remaining_ball_is_green
    (R B G : ℕ)
    (hR : R = 10)
    (hB : B = 11)
    (hG : G = 12)
    (exchange_rule : ∀ R B G : ℕ, (R - 1) + (B - 1) + (G + 1) = R + B + G)
    : (R + B + G) % 3 = 1 + 2 + 0 := 
begin
    sorry
end

end remaining_ball_is_green_l398_398795


namespace roots_of_quadratic_eq_l398_398153

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398153


namespace new_circle_radius_shaded_region_l398_398806

theorem new_circle_radius_shaded_region {r1 r2 : ℝ} 
    (h1 : r1 = 35) 
    (h2 : r2 = 24) : 
    ∃ r : ℝ, π * r^2 = π * (r1^2 - r2^2) ∧ r = Real.sqrt 649 := 
by
  sorry

end new_circle_radius_shaded_region_l398_398806


namespace compute_5_a_div_b_plus_6_b_div_a_l398_398623

theorem compute_5_a_div_b_plus_6_b_div_a (a b : ℝ) (h1 : a = Real.log 25) (h2 : b = Real.log 36) :
  5 ^ (a / b) + 6 ^ (b / a) = 11 := 
sorry

end compute_5_a_div_b_plus_6_b_div_a_l398_398623


namespace max_sum_of_squares_l398_398670

variable (a b c d : ℝ)

theorem max_sum_of_squares :
  a + b = 17 ∧
  ab + c + d = 85 ∧
  ad + bc = 180 ∧
  cd = 105 →
  ∃ (max : ℝ), max = a^2 + b^2 + c^2 + d^2 ∧ max = 934 :=
begin
  sorry
end

end max_sum_of_squares_l398_398670


namespace Elberta_has_23_dollars_l398_398991

theorem Elberta_has_23_dollars :
  let granny_smith_amount := 63
  let anjou_amount := 1 / 3 * granny_smith_amount
  let elberta_amount := anjou_amount + 2
  elberta_amount = 23 := by
  sorry

end Elberta_has_23_dollars_l398_398991


namespace remaining_distance_l398_398654

-- Definitions based on the conditions
def total_distance : ℕ := 78
def first_leg : ℕ := 35
def second_leg : ℕ := 18

-- The theorem we want to prove
theorem remaining_distance : total_distance - (first_leg + second_leg) = 25 := by
  sorry

end remaining_distance_l398_398654


namespace concurrency_of_lines_l398_398252

theorem concurrency_of_lines 
  (A B C X A1 B1 C1 A2 B2 C2 : Point)
  (hX_in_triangle : X ∈ triangle A B C)
  (hA1_on_circumcircle : A1 ∈ circumcircle (triangle A B C))
  (hAX_intersects_circumcircle_at_A1 : line_through A X ∩ circumcircle (triangle A B C) = A1)
  (hA2_on_segment_BC : A2 ∈ segment B C)
  (hA2_touching_incircle : A2 ∈ incircle_segment_cutoff (arcBC (circumcircle (triangle A B C)) A1) BC)
  (hB2_on_segment_AC : B2 ∈ segment A C)
  (hB2_touching_incircle : B2 ∈ incircle_segment_cutoff (arcAC (circumcircle (triangle A B C)) B1) AC)
  (hC2_on_segment_AB : C2 ∈ segment A B)
  (hC2_touching_incircle : C2 ∈ incircle_segment_cutoff (arcAB (circumcircle (triangle A B C)) C1) AB) :
  concurrent_lines (line_through A A2) (line_through B B2) (line_through C C2) := 
sorry

end concurrency_of_lines_l398_398252


namespace booking_rooms_needed_l398_398016

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398016


namespace cartesian_circle_standard_eq_distance_PA_PB_eq_l398_398234

-- Definition of parametric line equations
def line_param_eq_x (t : ℝ) : ℝ := 1 + (ℝ.sqrt 2) / 2 * t
def line_param_eq_y (t : ℝ) : ℝ := 2 + (ℝ.sqrt 2) / 2 * t

-- Definition of polar equation of circle
def polar_circle_eq (theta : ℝ) : ℝ := 6 * Real.sin theta

-- Definition of Cartesian circle equation derived from polar form
def cartesian_circle_eq (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- The point P on line l
def point_P : ℝ × ℝ := (1, 2)

-- Prove the standard equation of circle in Cartesian coordinates
theorem cartesian_circle_standard_eq :
  ∀ (ρ θ : ℝ), polar_circle_eq θ = ρ ↔ cartesian_circle_eq (ρ * Real.cos θ) (ρ * Real.sin θ) := by
  sorry

-- Prove the distance |PA| + |PB| equals 2√7
theorem distance_PA_PB_eq :
  ∀ (A B : ℝ × ℝ), 
    (A = (line_param_eq_x (ℝ.sqrt 7), line_param_eq_y (ℝ.sqrt 7))) → 
    (B = (line_param_eq_x (-ℝ.sqrt 7), line_param_eq_y (-ℝ.sqrt 7))) →
    ∃ (PA : ℝ) (PB : ℝ), PA = Real.dist point_P A ∧ PB = Real.dist point_P B 
    ∧ PA + PB = 2 * ℝ.sqrt 7 := by
  sorry

end cartesian_circle_standard_eq_distance_PA_PB_eq_l398_398234


namespace convex_polygons_from_twelve_points_on_circle_l398_398978

theorem convex_polygons_from_twelve_points_on_circle : 
  let points := 12 in
  ∃ n, n = 4017 ∧ ∀ subset, (subset.card ≥ 3 → subset ⊆ finset.range points) := sorry

end convex_polygons_from_twelve_points_on_circle_l398_398978


namespace trapezium_height_l398_398041

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end trapezium_height_l398_398041


namespace minor_axis_length_six_l398_398105

-- Definitions of distances and ellipse properties
variable (E : Type) [Ellipse E]

variable (F : E → E → ℝ)

def major_vertex_distances (E : Type) [Ellipse E] (F : E → E → ℝ) : Prop :=
  ∃ x y : E, F x y = 9 ∧ F x y = 1

-- Proposition statement for the length of the minor axis
theorem minor_axis_length_six (h : major_vertex_distances E F) : minor_axis_length E = 6 :=
sorry

end minor_axis_length_six_l398_398105


namespace range_of_a_l398_398596

noncomputable def g (x a : ℝ) := x^3 - a*x^2 + 2

theorem range_of_a (a : ℝ) (h_a : a < 2) :
  (∃ x ∈ Icc (-2 : ℝ) 1, g x a = 0) ↔ a ∈ Icc (-3/2 : ℝ) 2 :=
by
  sorry

end range_of_a_l398_398596


namespace sum_abs_terms_20_l398_398569

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n • d

-- Define the sum of absolute values of the first 20 terms
def sum_abs_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (List.range (n + 1)).sum (λ k, |arithmetic_sequence a d k|)

-- State the theorem
theorem sum_abs_terms_20 :
  sum_abs_terms 12 (-2) 19 = 224 :=
sorry

end sum_abs_terms_20_l398_398569


namespace joe_paint_fraction_l398_398657

theorem joe_paint_fraction (total_paint first_week_fraction total_used second_week_fraction : ℕ) 
(h1 : total_paint = 360)
(h2 : first_week_fraction = 1/4)
(h3 : total_used = 180)
(h4 : total_used = total_paint * first_week_fraction + total_paint * second_week_fraction / 4 )
: second_week_fraction = 1 / 3 := 
begin 
  sorry
end

end joe_paint_fraction_l398_398657


namespace poly_properties_l398_398272

open Polynomial

noncomputable def p (x : ℝ) : ℝ := 
  monic_poly -- Assumption to define the polynomial general form

theorem poly_properties 
  (monic_p : monic p) 
  (deg_p : degree p = 7)
  (h0 : p(0) = 0) 
  (h1 : p(1) = 1) 
  (h2 : p(2) = 2)
  (h3 : p(3) = 3) 
  (h4 : p(4) = 4) 
  (h5 : p(5) = 5)
  (h6 : p(6) = 6): 
  p(7) = 5047 :=
by
  sorry

end poly_properties_l398_398272


namespace problem_statement_l398_398526

/-- Definition of the function f that relates the input n with floor functions -/
def f (n : ℕ) : ℤ :=
  n + ⌊(n : ℤ) / 6⌋ - ⌊(n : ℤ) / 2⌋ - ⌊2 * (n : ℤ) / 3⌋

/-- Prove the main statement -/
theorem problem_statement (n : ℕ) (hpos : 0 < n) :
  f n = 0 ↔ ∃ k : ℕ, n = 6 * k + 1 :=
sorry -- Proof goes here.

end problem_statement_l398_398526


namespace scientific_notation_representation_l398_398834

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_representation_l398_398834


namespace isosceles_right_triangle_area_l398_398771

-- Define the conditions as given in the problem statement
variables (h l : ℝ)
hypothesis (hypotenuse_rel : h = l * Real.sqrt 2)
hypothesis (hypotenuse_val : h = 6 * Real.sqrt 2)

-- Define the formula for the area of an isosceles right triangle
def area_of_isosceles_right_triangle (l : ℝ) : ℝ := (1 / 2) * l * l

-- Define the proof problem statement
theorem isosceles_right_triangle_area : 
  area_of_isosceles_right_triangle l = 18 :=
  sorry

end isosceles_right_triangle_area_l398_398771


namespace range_of_a_l398_398595

noncomputable def f (x : ℝ) : ℝ := (1 / (1 + x^2)) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) ↔
  (1 / Real.exp 1 ≤ a ∧ a ≤ (2 + Real.log 3) / 3) :=
sorry

end range_of_a_l398_398595


namespace matrixA_power_93_eq_A_l398_398664

def matrixA : Matrix 3 3 ℝ :=
  ![
    ![0, 0, 0],
    ![0, 0, -1],
    ![0, 1, 0]
  ]

theorem matrixA_power_93_eq_A : matrixA^93 = matrixA := by
  sorry

end matrixA_power_93_eq_A_l398_398664


namespace rhombus_area_l398_398931

theorem rhombus_area (s d1 d2 : ℝ)
  (h1 : s = Real.sqrt 113)
  (h2 : abs (d1 - d2) = 8)
  (h3 : s^2 = (d1 / 2)^2 + (d2 / 2)^2) :
  (d1 * d2) / 2 = 194 := by
  sorry

end rhombus_area_l398_398931


namespace probability_two_cards_sum_15_from_standard_deck_l398_398393

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398393


namespace max_sum_n_l398_398750

-- Define the conditions
variable (d a1 c : ℝ)
variable (an : ℕ → ℝ) -- Arithmetic sequence
variable (Sn : ℕ → ℝ) -- Sum of the first n terms
variable (d_pos : d < 0) -- Common difference d is negative
variable (a1_pos : a1 = - (21 / 2) * d ∧ a1 > 0) -- Derived from the inequality's roots
variable (ineq_sol : ∀ x, 0 ≤ x ∧ x ≤ 22 → (d / 2) * x^2 + (a1 - d / 2) * x + c ≥ 0)

-- Define the terms in the sequence
def a (n : ℕ) : ℝ := a1 + (n - 1) * d
def S (n : ℕ) : ℝ := n * (a1 + (a1 + (n - 1) * d)) / 2

-- The maximum value of n for which the sum of the first n terms is maximized
theorem max_sum_n : ∃ n : ℕ, n = 11 ∧ (∀ m : ℕ, S m ≤ S 11) :=
sorry

end max_sum_n_l398_398750


namespace shaded_square_area_fraction_l398_398696

noncomputable def fraction_of_area_inside_shaded_square : ℚ :=
  let side_length := Real.sqrt ((3 - 2)^2 + (3 - 2)^2)
  let shaded_area := side_length^2
  let total_area := 6^2
  shaded_area / total_area

theorem shaded_square_area_fraction (grid_size : ℕ) (p1 p2 p3 p4 : ℤ × ℤ)
  (h_grid : grid_size = 6) 
  (h_rotated : True)  -- Placeholder, as rotation visualization is conceptual
  (h_vertices : p1 = (2,2) ∧ p2 = (3,3) ∧ p3 = (2,4) ∧ p4 = (1,3)) :
  fraction_of_area_inside_shaded_square = 1 / 18 :=
by
  sorry

end shaded_square_area_fraction_l398_398696


namespace samantha_last_name_length_l398_398306

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end samantha_last_name_length_l398_398306


namespace prime_number_property_l398_398148

open Nat

-- Definition that p is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Conjecture to prove: if p is a prime number and p^4 - 3p^2 + 9 is also a prime number, then p = 2.
theorem prime_number_property (p : ℕ) (h1 : is_prime p) (h2 : is_prime (p^4 - 3*p^2 + 9)) : p = 2 :=
sorry

end prime_number_property_l398_398148


namespace probability_of_contact_l398_398058

noncomputable def probability_connection (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

theorem probability_of_contact (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let group1 := 5
  let group2 := 8
  let total_pairs := group1 * group2
  (total_pairs = 40) →
  (∀ i j, i ∈ fin group1 → j ∈ fin group2 → (¬ p = 1 → p = p)) → 
  probability_connection p = 1 - (1 - p) ^ 40 :=
by
  intros _ _ 
  sorry

end probability_of_contact_l398_398058


namespace part1_part2_part3_l398_398957

-- Given conditions
def P := (-3 : ℝ, real.sqrt 3)
def α := real.arctan (-real.sqrt 3 / 3) -- Finding α correctly

-- Part (1)
theorem part1 : real.tan α = -real.sqrt 3 / 3 :=
sorry

-- Part (2)
noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem part2 : determinant (real.sin α) (real.tan α) 1 (real.cos α) = real.sqrt 3 / 12 :=
sorry

-- Part (3)
noncomputable def f (x : ℝ) := real.cos x

noncomputable def y (x : ℝ) := real.sqrt 3 * f (real.pi / 2 - 2 * x) + 2 * (f x)^2

theorem part3 : ∃ (k : ℤ), y (k * real.pi + real.pi / 6) = 3 :=
sorry

end part1_part2_part3_l398_398957


namespace runner_F_placed_last_l398_398223

noncomputable def probability_F_last
  (A B C D E F : ℕ) (finished_between : ℕ → ℕ → ℕ → Prop) : ℚ :=
  if finished_between A B C ∧ finished_between B C D ∧ finished_between D E F then
    5 / 16
  else
    0

theorem runner_F_placed_last
  {A B C D E F : ℕ}
  (finished_between : ℕ → ℕ → ℕ → Prop)
  (h1 : finished_between A B C)
  (h2 : finished_between B C D)
  (h3 : finished_between D E F) :
  probability_F_last A B C D E F finished_between = 5 / 16 := by
  sorry

end runner_F_placed_last_l398_398223


namespace range_of_k_l398_398958

theorem range_of_k (M N : ℝ × ℝ) (k : ℝ) (h_parabola_M : M.2 = M.1^2) (h_parabola_N : N.2 = N.1^2)
  (h_symmetric : M.2 + N.2 = k * (M.1 + N.1) + 9) :
  k ∈ set.Ioo (-∞) (-1/4) ∪ set.Ioo (1/4) ∞ :=
sorry

end range_of_k_l398_398958


namespace minimum_colors_required_l398_398813

noncomputable def num_lines : ℕ := 2018

theorem minimum_colors_required (lines : finset (set (ℝ × ℝ))) :
  lines.card = num_lines →
  (∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → ∃! P, P ∈ l₁ ∧ P ∈ l₂) →
  (∀ l₁ l₂ l₃ ∈ lines, l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₃ ≠ l₁ → ¬∃ P, P ∈ l₁ ∧ P ∈ l₂ ∧ P ∈ l₃) →
  ∃ c : (ℝ × ℝ) → ℕ, (∀ l ∈ lines, ∀ P₁ P₂ ∈ l, segment P₁ P₂ ∩ intersections = ∅ → c P₁ ≠ c P₂) ∧ (∀ P, c P < 3) :=
sorry

end minimum_colors_required_l398_398813


namespace grandmother_cheapest_option_l398_398985

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end grandmother_cheapest_option_l398_398985


namespace find_m_value_l398_398534

noncomputable def polynomial := (x : ℝ) → 5 * x^3 - 3 * x^2 - 12 * x + m

theorem find_m_value (m : ℝ) : (polynomial 4 = 0) → m = -224 :=
by {
  sorry
}

end find_m_value_l398_398534


namespace increasing_interval_of_f_l398_398962

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem increasing_interval_of_f (φ : ℝ) (H1 : |φ| < Real.pi)
  (H2 : ∀ x : ℝ, f x φ ≤ |f (Real.pi / 6) φ|)
  (H3 : f (Real.pi / 2) φ > f Real.pi φ) :
  ∀ k : ℤ, ∀ x : ℝ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → 
    increasing (λ x, f x φ) :=
sorry -- proof will go here

end increasing_interval_of_f_l398_398962


namespace last_digit_of_sum_of_powers_l398_398528

theorem last_digit_of_sum_of_powers {a b c d : ℕ} 
  (h1 : a = 2311) (h2 : b = 5731) (h3 : c = 3467) (h4 : d = 6563) 
  : (a^b + c^d) % 10 = 4 := by
  sorry

end last_digit_of_sum_of_powers_l398_398528


namespace two_positive_roots_condition_l398_398585

theorem two_positive_roots_condition (a : ℝ) :
  (1 < a ∧ a ≤ 2) ∨ (a ≥ 10) ↔
  ∃ x1 x2 : ℝ, (1-a) * x1^2 + (a+2) * x1 - 4 = 0 ∧ 
               (1-a) * x2^2 + (a+2) * x2 - 4 = 0 ∧ 
               x1 > 0 ∧ x2 > 0 :=
sorry

end two_positive_roots_condition_l398_398585


namespace cube_side_length_of_paint_cost_l398_398751

theorem cube_side_length_of_paint_cost (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  cost_per_kg = 20 ∧ coverage_per_kg = 15 ∧ total_cost = 200 →
  6 * side_length ^ 2 = (total_cost / cost_per_kg) * coverage_per_kg →
  side_length = 5 :=
by
  intros h1 h2
  sorry

end cube_side_length_of_paint_cost_l398_398751


namespace number_of_kiwis_l398_398793

/-
There are 500 pieces of fruit in a crate. One fourth of the fruits are apples,
20% are oranges, one fifth are strawberries, and the rest are kiwis.
Prove that the number of kiwis is 175.
-/

theorem number_of_kiwis (total_fruits apples oranges strawberries kiwis : ℕ)
  (h1 : total_fruits = 500)
  (h2 : apples = total_fruits / 4)
  (h3 : oranges = 20 * total_fruits / 100)
  (h4 : strawberries = total_fruits / 5)
  (h5 : kiwis = total_fruits - (apples + oranges + strawberries)) :
  kiwis = 175 :=
sorry

end number_of_kiwis_l398_398793


namespace probability_two_cards_sum_to_15_l398_398403

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398403


namespace arccos_one_half_eq_pi_div_three_l398_398505

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end arccos_one_half_eq_pi_div_three_l398_398505


namespace log_eq_l398_398996

theorem log_eq (a : ℝ) (b : ℤ) (h_b : 2 ≤ b ∧ b ≤ 300) (h_a : a > 0) :
  (log b a)^2023 = log b (a^2023) → 897 :=
sorry

end log_eq_l398_398996


namespace find_line_equation_l398_398531

noncomputable def equation_of_line_through_M_with_double_slope_angle
  (M : (ℝ × ℝ)) (l : LinEq) : LinEq :=
    sorry

theorem find_line_equation
  (M : (ℝ × ℝ)) (hM : M = (1, 1))
  (l : LinEq) (hl : l = 2 * x - y = 0) :
  equation_of_line_through_M_with_double_slope_angle M l = 4 * x + 3 * y - 7 :=
sorry

end find_line_equation_l398_398531


namespace all_points_circle_l398_398539

variable (u : ℝ)

def x := (1 - u^4) / (1 + u^4)
def y := (2 * u^2) / (1 + u^4)

theorem all_points_circle : x u ^ 2 + y u ^ 2 = 1 := 
by sorry

end all_points_circle_l398_398539


namespace shopkeeper_sold_72_articles_l398_398868

noncomputable def number_of_articles_sold :=
  let C := ℝ
  let N := ℝ
  let cost_price_per_article := C
  let total_cost_price_60_articles := 60 * C
  let profit_percentage := 0.20
  let total_selling_price := total_cost_price_60_articles + profit_percentage * total_cost_price_60_articles
  ∀ N : ℝ, (N = total_selling_price / C) → N = 72

theorem shopkeeper_sold_72_articles : number_of_articles_sold :=
sorry

end shopkeeper_sold_72_articles_l398_398868


namespace min_rooms_needed_l398_398031

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398031


namespace coeff_x4y2_in_expansion_l398_398238

noncomputable def binom_expansion_coefficient : ℕ :=
  let x := 2
  let y := -1
  let coeff := 5
  binomial_coeff coeff 2 * binomial_coeff 3 1 * 2

theorem coeff_x4y2_in_expansion : binom_expansion_coefficient = 60 :=
by sorry

end coeff_x4y2_in_expansion_l398_398238


namespace chocolate_chips_per_cookie_l398_398993

theorem chocolate_chips_per_cookie 
  (total_chocolate_chips : ℕ)
  (batches : ℕ)
  (cookies_per_batch : ℕ)
  (H1 : total_chocolate_chips = 81)
  (H2 : batches = 3)
  (H3 : cookies_per_batch = 3) :
  total_chocolate_chips / (batches * cookies_per_batch) = 9 :=
by {
  simp [H1, H2, H3],
  norm_num,
  sorry
}

end chocolate_chips_per_cookie_l398_398993


namespace radius_tangent_circle_l398_398071

theorem radius_tangent_circle (r r1 r2 : ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 5)
    (h_concentric : true) : r = 1 := by
  -- Definitions are given as conditions
  have h1 := r1 -- radius of smaller concentric circle
  have h2 := r2 -- radius of larger concentric circle
  have h3 := h_concentric -- the circles are concentric
  have h4 := h_r1 -- r1 = 3
  have h5 := h_r2 -- r2 = 5
  sorry

end radius_tangent_circle_l398_398071


namespace sin_cos_alpha_l398_398920

theorem sin_cos_alpha (α : ℝ) (h : tan α = 4) : sin α * cos α = 4 / 17 :=
by
  sorry

end sin_cos_alpha_l398_398920


namespace suitable_for_systematic_sampling_l398_398880

-- Define the given conditions as a structure
structure SamplingProblem where
  option_A : String
  option_B : String
  option_C : String
  option_D : String

-- Define the equivalence theorem to prove Option C is the most suitable
theorem suitable_for_systematic_sampling (p : SamplingProblem) 
(hA: p.option_A = "Randomly selecting 8 students from a class of 48 students to participate in an activity")
(hB: p.option_B = "A city has 210 department stores, including 20 large stores, 40 medium stores, and 150 small stores. To understand the business situation of each store, a sample of 21 stores needs to be drawn")
(hC: p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions")
(hD: p.option_D = "Randomly selecting 10 students from 1200 high school students participating in a mock exam to understand the situation") :
  p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions" := 
sorry

end suitable_for_systematic_sampling_l398_398880


namespace probability_two_cards_sum_15_from_standard_deck_l398_398396

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398396


namespace cost_per_student_admission_l398_398326

-- Definitions based on the conditions.
def cost_to_rent_bus : ℕ := 100
def total_budget : ℕ := 350
def number_of_students : ℕ := 25

-- The theorem that we need to prove.
theorem cost_per_student_admission : (total_budget - cost_to_rent_bus) / number_of_students = 10 :=
by
  sorry

end cost_per_student_admission_l398_398326


namespace distance_between_parallel_sides_l398_398045

/-- Define the lengths of the parallel sides and the area of the trapezium -/
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area : ℝ := 285

/-- Define the condition of the problem: the formula for the area of the trapezium -/
def area_of_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

/-- The problem: prove the distance between the parallel sides is 15 cm -/
theorem distance_between_parallel_sides (h : ℝ) : 
  area_of_trapezium length_side1 length_side2 h = area → h = 15 :=
by
  sorry

end distance_between_parallel_sides_l398_398045


namespace tan_600_eq_neg_sqrt_3_l398_398533

theorem tan_600_eq_neg_sqrt_3 : Real.tan (600 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_600_eq_neg_sqrt_3_l398_398533


namespace acute_angle_vector_range_l398_398207

theorem acute_angle_vector_range (m : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (4, m)) 
  (acute : (a.1 * b.1 + a.2 * b.2) > 0) : 
  (m > -2) ∧ (m ≠ 8) := 
by 
  sorry

end acute_angle_vector_range_l398_398207


namespace sport_formulation_water_content_l398_398429

theorem sport_formulation_water_content :
  ∀ (f_s c_s w_s : ℕ) (f_p c_p w_p : ℕ),
    f_s / c_s = 1 / 12 →
    f_s / w_s = 1 / 30 →
    f_p / c_p = 1 / 4 →
    f_p / w_p = 1 / 60 →
    c_p = 4 →
    w_p = 60 := by
  sorry

end sport_formulation_water_content_l398_398429


namespace convex_polygon_in_rectangle_l398_398720

/-- A convex polygon -/
structure ConvexPolygon :=
(vertices : Set (ℝ × ℝ))
(convex : Set.convex ℝ vertices)
(area : ℝ)

theorem convex_polygon_in_rectangle (P : ConvexPolygon) (h_area : P.area = 1) :
  ∃ (R : Set (ℝ × ℝ)), (∃ (l w : ℝ), l * w = 2) ∧ (P.vertices ⊆ R) :=
by
  sorry

end convex_polygon_in_rectangle_l398_398720


namespace f_gt_f_l398_398594

open Real

noncomputable def f (x : ℝ) : ℝ := x - log x + (2 * x - 1) / x^2

noncomputable def f' (x : ℝ) : ℝ :=
  (deriv (λ x : ℝ, x - log x + (2 * x - 1) / x^2)) x

theorem f_gt_f'_plus_3_div_2 (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) :
  f x > f' x + 3 / 2 :=
sorry

end f_gt_f_l398_398594


namespace yvonnes_probability_l398_398827

open Classical

variables (P_X P_Y P_Z : ℝ)

theorem yvonnes_probability
  (h1 : P_X = 1/5)
  (h2 : P_Z = 5/8)
  (h3 : P_X * P_Y * (1 - P_Z) = 0.0375) :
  P_Y = 0.5 :=
by
  sorry

end yvonnes_probability_l398_398827


namespace event_B_more_likely_than_event_A_l398_398712

/-- Define the outcomes when rolling a die three times --/
def total_outcomes : ℕ := 6 ^ 3

/-- Define the number of ways to choose 3 different numbers from 6 --/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Define the number of ways to arrange 3 different numbers --/
def arrangements_3 : ℕ := 3.factorial

/-- Calculate the number of favorable outcomes for event B --/
def favorable_B : ℕ := choose_3_from_6 * arrangements_3

/-- Define the probability of event B --/
noncomputable def prob_B : ℝ := favorable_B / total_outcomes

/-- Define the probability of event A as the complement of event B --/
noncomputable def prob_A : ℝ := 1 - prob_B

/-- The theorem to prove that event B is more likely than event A --/
theorem event_B_more_likely_than_event_A : prob_B > prob_A :=
by
  sorry

end event_B_more_likely_than_event_A_l398_398712


namespace tangent_parallel_to_line_l398_398351

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (P : ℝ × ℝ) 
(hP : ∃ (x : ℝ), P = (x, f x))
(h_parallel : ∀ (x : ℝ), deriv f x = 4 ↔ P = (1, 0) ∨ P = (-1, -4)) :
P = (1, 0) ∨ P = (-1, -4) :=
by sorry

end tangent_parallel_to_line_l398_398351


namespace quadratic_root_sum_and_product_l398_398194

theorem quadratic_root_sum_and_product :
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 2 * x₁ - 8 = 0) (hx₂ : x₂^2 - 2 * x₂ - 8 = 0) :
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 := 
sorry

end quadratic_root_sum_and_product_l398_398194


namespace parallelogram_sides_l398_398856

theorem parallelogram_sides (a b : ℝ)
  (h1 : 2 * (a + b) = 32)
  (h2 : b - a = 8) :
  a = 4 ∧ b = 12 :=
by
  -- Proof is to be provided
  sorry

end parallelogram_sides_l398_398856


namespace quadratic_roots_identity_l398_398173

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398173


namespace middle_term_of_arithmetic_sequence_is_five_l398_398486

theorem middle_term_of_arithmetic_sequence_is_five (a : ℕ → ℝ) (n : ℕ) (h_arith_seq : ∀ i, a (i + 1) - a i = a 1 - a 0) 
  (h_length : n = 11) (h_sum_odd_terms : (∑ i in {0, 2, 4, 6, 8, 10}, a i) = 30) : 
  a 5 = 5 :=
by 
  sorry

end middle_term_of_arithmetic_sequence_is_five_l398_398486


namespace probability_two_cards_sum_15_from_standard_deck_l398_398398

-- Definitions
def standardDeck := {card : ℕ | 2 ≤ card ∧ card ≤ 10}
def validSum15 (pair : ℕ × ℕ) := pair.1 + pair.2 = 15

-- Problem statement
theorem probability_two_cards_sum_15_from_standard_deck :
  let totalCards := 52
  let numberCards := 4 * (10 - 1)
  (4 / totalCards) * (20 / (totalCards - 1)) = 100 / 663 := sorry

end probability_two_cards_sum_15_from_standard_deck_l398_398398


namespace roots_of_quadratic_eq_l398_398149

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398149


namespace find_cheapest_option_l398_398988

variable (transportation_cost : ℕ) (berries_collected : ℕ)
          (cost_train_per_week : ℕ) (cost_berries_market : ℕ)
          (cost_sugar : ℕ) (jam_rate : ℚ) (cost_ready_made_jam : ℕ)
      
-- Define the cost of gathering 1.5 kg of jam
def option1_cost := (cost_train_per_week / berries_collected + cost_sugar) * jam_rate

-- Define the cost of buying berries and sugar to make 1.5 kg of jam
def option2_cost := (cost_berries_market + cost_sugar) * jam_rate

-- Define the cost of buying 1.5 kg of ready-made jam
def option3_cost := cost_ready_made_jam * jam_rate

theorem find_cheapest_option :
  option1_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam 
  < min (option2_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam)
        (option3_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam) :=
by
  unfold option1_cost option2_cost option3_cost
  have hc1 : (40 : ℕ) + 54 = 94 := by norm_num
  have hc2 : (150 : ℕ) + 54 = 204 := by norm_num
  have hc3 : (220 : ℕ) * (3/2) = 330 := by norm_num
  linarith
  sorry

end find_cheapest_option_l398_398988


namespace triangle_cea_isosceles_bc_minus_ab_l398_398653

theorem triangle_cea_isosceles 
  (A B C E : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E]
  (angle_ABC : ℝ) (angle_ACB : ℝ) 
  (BE BA : ℝ) (BE_eq_BA : BE = BA)
  (angle_ABC_eq : angle_ABC = 20) 
  (angle_ACB_eq : angle_ACB = 40) :
  ∃ (CEA_isosceles : Prop), CEA_isosceles := 
begin
  sorry
end

theorem bc_minus_ab 
  (A B C E : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E]
  (length_bisector_BAC : ℝ) 
  (length_bisector_BAC_eq : length_bisector_BAC = 2) 
  (BC AB : ℝ) :
  BC - AB = 2 := 
begin
  sorry
end

end triangle_cea_isosceles_bc_minus_ab_l398_398653


namespace squares_common_area_tiled_l398_398722

noncomputable def common_area_tiled_with_squares {α : Type*} [Nonempty α] (A B : Set α) (O : α) (rotate : α → α) : Prop :=
∃ (Q Q' : Set α), ((Q ∪ Q' ∈ A ∧ O ∈ A) ∧
(Q = rotate O θ) ∧
(tiled_with_squares {x | x ∈ Q ∧ x ∈ Q'}))

/-- If two squares share a common center and one is rotated by an acute angle, then their common area forms an octagon that can be tiled with squares. -/
theorem squares_common_area_tiled (O : α) (Q Q' : Set α) (θ : ℝ) [RotationalSymmetry Q O] [RotationalSymmetry Q' O] [TiledWithSquares Q] :
  common_area_tiled_with_squares Q Q' O (rotate_by O θ) := sorry

end squares_common_area_tiled_l398_398722


namespace find_cos_gamma_l398_398673

variables (x y z : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : 0 < z)
variable (cos_alpha : ℝ := 2 / 5)
variable (cos_beta : ℝ := 1 / 4)

theorem find_cos_gamma :
  (cos_alpha^2 + cos_beta^2 + ((z / (x^2 + y^2 + z^2)^0.5)^2) = 1) →
  (cos_gamma : ℝ) = (311^0.5 / 20) := sorry

end find_cos_gamma_l398_398673


namespace prob_sum_15_correct_l398_398372

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398372


namespace polar_to_rectangular_l398_398951

def rectangular_coordinate_equation (x y : ℝ) : Prop := 
  √3 * x + y = 6

theorem polar_to_rectangular (ρ θ : ℝ) 
  (h : ρ * sin (θ + π / 3) = 3) : ∃ x y : ℝ, rectangular_coordinate_equation x y :=
by 
  sorry

end polar_to_rectangular_l398_398951


namespace exam_problem_solution_l398_398520

theorem exam_problem_solution :
  let Pa := 1/3
  let Pb := 1/4
  let Pc := 1/5
  let Pnone := (1 - Pa) * (1 - Pb) * (1 - Pc)
  let Pat_least_one := 1 - Pnone
  Pat_least_one = 3/5 :=
by
  -- formal proof would go here
  sorry

end exam_problem_solution_l398_398520


namespace avg_visitor_per_day_in_month_with_given_conditions_l398_398451

noncomputable def average_visitors_per_day 
  (visitors_sunday: ℕ) 
  (visitors_other_day: ℕ) 
  (days_in_month: ℕ) 
  (sundays_in_month: ℕ) 
: ℕ := 
  let other_days := days_in_month - sundays_in_month in
  (sundays_in_month * visitors_sunday + other_days * visitors_other_day) / days_in_month

theorem avg_visitor_per_day_in_month_with_given_conditions
  (visitors_sunday: ℕ) 
  (visitors_other_day: ℕ)
  (days_in_month: ℕ) 
  (sundays_in_month: ℕ)
  (other_days: ℕ := days_in_month - sundays_in_month)
  (total_visitors: ℕ := sundays_in_month * visitors_sunday + other_days * visitors_other_day)
  (avg_visitors_per_day: ℕ := total_visitors / days_in_month)
  (h1: visitors_sunday = 510)
  (h2: visitors_other_day = 240)
  (h3: days_in_month = 30)
  (h4: sundays_in_month = 4)
: avg_visitors_per_day = 276 :=
by
  have h5 : other_days = 26 := by simp [h3, h4]
  have h6 : total_visitors = 8280 := by simp [h1, h2, h4, h5]
  have h7 : avg_visitors_per_day = 276 := by simp [h6, h3]
  exact h7

end avg_visitor_per_day_in_month_with_given_conditions_l398_398451


namespace trapezium_height_l398_398049

theorem trapezium_height :
  ∀ (a b h : ℝ), a = 20 ∧ b = 18 ∧ (1 / 2) * (a + b) * h = 285 → h = 15 :=
by
  intros a b h hconds
  cases hconds with h1 hrem
  cases hrem with h2 harea
  simp at harea
  sorry

end trapezium_height_l398_398049


namespace simon_legos_l398_398729

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l398_398729


namespace poles_intersection_l398_398214

-- Define the known heights and distances
def heightOfIntersection (d h1 h2 x : ℝ) : ℝ := sorry

theorem poles_intersection :
  heightOfIntersection 120 30 60 40 = 20 := by
  sorry

end poles_intersection_l398_398214


namespace simon_legos_l398_398735

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l398_398735


namespace distinct_letters_permutations_count_l398_398616

-- Define a list of distinct letters
def letters : List Char := ['T', 'E₁', 'E₂', 'E₃', 'N₁', 'N₂', 'S₁', 'S₂']

-- we need to prove the number of permutations of these letters
theorem distinct_letters_permutations_count : 
  list.permutations letters |>.length = 40320 := 
by {
  sorry
}

end distinct_letters_permutations_count_l398_398616


namespace booking_rooms_needed_l398_398010

def team_partition := ℕ

def gender_partition := ℕ

def fans := ℕ → ℕ

variable (n : fans)

-- Conditions:
variable (num_fans : ℕ)
variable (num_rooms : ℕ)
variable (capacity : ℕ := 3)
variable (total_fans : num_fans = 100)
variable (team_groups : team_partition)
variable (gender_groups : gender_partition)
variable (teams : ℕ := 3)
variable (genders : ℕ := 2)

theorem booking_rooms_needed :
  (∀ fan : fans, fan team_partition + fan gender_partition ≤ num_rooms) ∧
  (∀ room : num_rooms, room * capacity ≥ fan team_partition + fan gender_partition) ∧
  (set.countable (fan team_partition + fan gender_partition)) ∧ 
  (total_fans = 100) ∧
  (team_groups = teams) ∧
  (gender_groups = genders) →
  (num_rooms = 37) :=
by
  sorry

end booking_rooms_needed_l398_398010


namespace isosceles_triangle_area_l398_398764

theorem isosceles_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) 
  (h_eq : h = 6 * sqrt 2) 
  (h_leg : h = a * sqrt 2) 
  (area_eq : A = 1 / 2 * a^2) : 
  A = 18 :=
by
  sorry

end isosceles_triangle_area_l398_398764


namespace functional_equation_solution_l398_398038

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) ↔ (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solution_l398_398038


namespace prob_sum_15_correct_l398_398375

def number_cards : List ℕ := [6, 7, 8, 9]

def count_pairs (s : ℕ) : ℕ :=
  (number_cards.filter (λ n, (s - n) ∈ number_cards)).length

def pair_prob (total_cards n : ℕ) : ℚ :=
  (count_pairs n : ℚ) * (4 : ℚ) / (total_cards : ℚ) * (4 - 1 : ℚ) / (total_cards - 1 : ℚ)

noncomputable def prob_sum_15 : ℚ :=
  pair_prob 52 15

theorem prob_sum_15_correct : prob_sum_15 = 16 / 663 := by
    sorry

end prob_sum_15_correct_l398_398375


namespace fixed_point_l398_398584

-- Definitions and conditions
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

def line (k m x y : ℝ) : Prop := y = k * x + m

-- The geometric properties involved
def passes_through (x y : ℝ) (P : ℝ × ℝ) := (x = P.1) ∧ (y = P.2)

def circle_with_diameter (M N : ℝ × ℝ) (A : ℝ × ℝ) : Prop := 
  let (x1, y1) := M in
  let (x2, y2) := N in
  (2 - x2) * (2 - x1) + y1 * y2 = 0

-- The fixed point to be proven
def is_fixed_point (k m : ℝ) : Prop := 
  (line k m (6 / 5) 0) ∧ (passes_through (6 / 5) 0 (6 / 5, 0))

-- Main theorem to prove
theorem fixed_point (k m x1 y1 x2 y2 : ℝ) 
  (hk : k ≠ 0) 
  (hline1 : line k m x1 y1) 
  (hline2 : line k m x2 y2) 
  (hellipse1 : ellipse x1 y1) 
  (hellipse2 : ellipse x2 y2) 
  (hcirc : circle_with_diameter (x1, y1) (x2, y2) (2, 0)) : 
  is_fixed_point k m :=
sorry

end fixed_point_l398_398584


namespace proof_problem_statement_l398_398302

noncomputable def proof_problem (x y: ℝ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ (∀ n : ℕ, n > 0 → (⌊x / y⌋ : ℝ) = ⌊↑n * x⌋ / ⌊↑n * y⌋) →
  (x = y ∨ (∃ k : ℤ, k ≠ 0 ∧ (x = k * y ∨ y = k * x)))

-- The formal statement of the problem
theorem proof_problem_statement (x y : ℝ) :
  proof_problem x y := by
  sorry

end proof_problem_statement_l398_398302


namespace probability_two_cards_sum_to_15_l398_398400

open ProbabilityTheory

noncomputable def probability_two_cards_sum_15 : ℚ :=
  let total_cards := 52
  let total_pairing_ways := 52 * 51 / 2
  let valid_pairing_ways := 4 * (4 + 4 + 4)  -- ways for cards summing to 15
  valid_pairing_ways / (total_pairing_ways)

theorem probability_two_cards_sum_to_15:
  probability_two_cards_sum_15 = (16 / 884) := by
  sorry

end probability_two_cards_sum_to_15_l398_398400


namespace invitations_sent_out_l398_398655

-- Define the conditions
def RSVPed (I : ℝ) : ℝ := 0.9 * I
def Showed_up (I : ℝ) : ℝ := 0.8 * RSVPed I
def No_gift : ℝ := 10
def Thank_you_cards : ℝ := 134

-- Prove the number of invitations
theorem invitations_sent_out : ∃ I : ℝ, Showed_up I - No_gift = Thank_you_cards ∧ I = 200 :=
by
  sorry

end invitations_sent_out_l398_398655


namespace roots_of_quadratic_eq_l398_398150

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398150


namespace regular_polygon_tan_sum_l398_398509

noncomputable def sum_tan_squared_angles {n : ℕ} (α : Fin n → ℝ) : ℝ :=
  ∑ i, (Real.tan (α i)) ^ 2

theorem regular_polygon_tan_sum (n : ℕ) (α : Fin n → ℝ) :
  let θ := (Real.pi / (2 * n))
  ∑ i in Finset.range n, (Real.tan (α ⟨i, by linarith⟩)) ^ 2 =
  2 * n * (Real.cos θ) ^ 2 / (Real.sin θ) ^ 4 :=
sorry

end regular_polygon_tan_sum_l398_398509


namespace quadratic_roots_sum_product_l398_398186

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l398_398186


namespace part_a_part_b_l398_398680

noncomputable def f_condition_a (f: ℝ → ℝ) : Prop :=
∀ (r: ℝ), ∇ (∇ f) (r : ℝ^3) = 0 → f = -C / r

noncomputable def f_condition_b (f: ℝ → ℝ) (C : ℝ) : Prop :=
∀ (r: ℝ), ∇ (λ r, f r * r) = 0 → f = C / r^3

theorem part_a {f: ℝ → ℝ} (h: f_condition_a f) : ∀ (r: ℝ), ∇ (∇ f) (r: ℝ^3) = 0 → f = -C / r := by
  sorry

theorem part_b {f: ℝ → ℝ} {C: ℝ} (h: f_condition_b f C) : ∀ (r: ℝ), ∇ (λ r, f r * r) = 0 → f = C / r^3 := by
  sorry

end part_a_part_b_l398_398680


namespace necessary_not_sufficient_condition_l398_398943

variable (a : ℝ) (D : Set ℝ)

def p : Prop := a ∈ D
def q : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ - a ≤ -3

theorem necessary_not_sufficient_condition (h : p a D → q a) : D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end necessary_not_sufficient_condition_l398_398943


namespace intersection_M_P_l398_398283

def M : Set ℝ := {0, 1, 2, 3}
def P : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_P : M ∩ P = {0, 1} := 
by
  -- You can fill in the proof here
  sorry

end intersection_M_P_l398_398283


namespace proof_problem_l398_398904

-- Conditions
def op1 := (15 + 3) / (8 - 2) = 3
def op2 := (9 + 4) / (14 - 7)

-- Statement
theorem proof_problem : op1 → op2 = 13 / 7 :=
by 
  intro h
  unfold op2
  sorry

end proof_problem_l398_398904


namespace multiple_of_a_l398_398543

theorem multiple_of_a (a : ℤ) (k : ℤ) :
  (97 * a^2 + 84 * a - 55 = k * a) ↔ 
  a ∈ ({1, 5, 11, 55, -1, -5, -11, -55} : set ℤ) :=
by sorry

end multiple_of_a_l398_398543


namespace Bryan_score_l398_398885

variable (Bryan Jen Sammy : ℤ)
variable (totalPoints mistakes : ℤ)
variable (totalPointsEq : totalPoints = 35)
variable (mistakesEq : mistakes = 7)
variable (JenScore : Jen = Bryan + 10)
variable (SammyScore : Sammy = Jen - 2)
variable (SammyMistakes : Sammy = totalPoints - mistakes)

theorem Bryan_score : Bryan = 20 := by
  have SammyScoreCorrect : Sammy = 28 := by
    rw [SammyMistakes, totalPointsEq, mistakesEq]
    norm_num
  have BryanFromSammy : Bryan = 28 - 8 := by
    rw [SammyScore, JenScore]
    rw [SammyScoreCorrect]
    norm_num
  exact BryanFromSammy

end Bryan_score_l398_398885


namespace problem_part1_problem_part2_l398_398575

noncomputable def a (n : ℕ) : ℝ := Classical.some (exists_unique_of_exists_of_unique
  (λ x : ℝ, x^3 + x / (n + 1).toReal = 1) 
  sorry) -- existence and uniqueness proof placeholder

theorem problem_part1 (n : ℕ) (hn : 0 < a n ∧ a n < 1):
  (a n < a (n + 1)) :=
sorry

theorem problem_part2 (n : ℕ) (hn : 0 < a n ∧ a n < 1) :
  (∑ k in Finset.range n, 1 / ((k + 2) ^ 2 * a (k + 1)) < a n) :=
sorry

end problem_part1_problem_part2_l398_398575


namespace seat_students_l398_398227

theorem seat_students (x : ℕ) (y : ℕ) 
  (h1 : x * y = 540) 
  (h2 : y ≥ 12) 
  (h3 : 20 ≤ x ∧ x ≤ 30) : 
  ∑ x in {x | x * (540 / x) = 540 ∧ 12 ≤ 540 / x ∧ 20 ≤ x ∧ x ≤ 30}, x = 77 :=
by 
  -- Proof skipped.
  sorry

end seat_students_l398_398227


namespace fifteenth_entry_of_sequence_l398_398914

def r_9 (n : ℕ) : ℕ := n % 9

theorem fifteenth_entry_of_sequence : 
   let seq := { n : ℕ | r_9 (3 * n) ≤ 4} 
   (finset.sort (≤) (finset.filter (λ (x : ℕ), x ∈ seq) (finset.range 100))).nth 14 = 29 :=
sorry

end fifteenth_entry_of_sequence_l398_398914


namespace ellipse_and_line_equations_l398_398433

section problem_statement

-- Define the focal distance and eccentricity
def focal_distance : ℝ := 1
def eccentricity : ℝ := 1 / 2

-- Define the center of the ellipse
def center : ℝ × ℝ := (0, 0)

-- Define the equation of the ellipse to be proved
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define point M on the line
def M : ℝ × ℝ := (0, 1)

-- Define the condition for line l intersecting ellipse C at points A and B
-- with the condition that \overrightarrow{AM} = 2\overrightarrow{MB}
def line_condition (m : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    (x1, y1) ≠ (0, 1) ∧ (x2, y2) ≠ (0, 1) ∧
    (y1 - 1) = m * (x1 - 0) ∧  (y2 -1) = m * (x2 - 0) ∧
    (x1^2 / 4) + (y1^2 / 3) = 1 ∧ (x2^2 / 4) + (y2^2 / 3) = 1 ∧ 
    (2 * (1 - y1 + y2) = y1 - 1) ∧ x1 = -2 * x2

-- Define the equation of line l to be proved
def line_equations (k : ℝ) : Prop :=
  k = 1 / 2 ∨ k = -1 / 2

theorem ellipse_and_line_equations :
  (∀ x y : ℝ, ellipse_equation x y) ∧ 
  (∃ m : ℝ, line_condition m) ∧ 
  (∀ k : ℝ, line_equations k) :=
begin
  sorry
end

end problem_statement

end ellipse_and_line_equations_l398_398433


namespace min_rooms_needed_l398_398033

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398033


namespace good_number_2013_l398_398213

def is_good_number (a : ℕ) : Prop :=
  (a.digits 10).sum = 6

def good_numbers := (List.range 10000).filter is_good_number

def a_n (n : ℕ) : ℕ := (good_numbers.get? n).get_or_else 0

theorem good_number_2013 {n : ℕ} (h : a_n n = 2013) : n = 51 :=
  sorry

end good_number_2013_l398_398213


namespace current_time_approx_10_13_l398_398253

theorem current_time_approx_10_13 :
  ∃ (x : ℝ), (x > 0) ∧ (x < 60) ∧
  6 * (6 + x) + 0.5 * (120 - x + 3) = 180 ∧
  abs (x - 12.6923) < 0.1 :=
begin
  -- proof goes here
  sorry
end

end current_time_approx_10_13_l398_398253


namespace parabola_equation_from_directrix_l398_398579

theorem parabola_equation_from_directrix (d : ℝ) (h : d = -7) : 
  ∃ a : ℝ, (y : ℝ) (x : ℝ), y^2 = a * x ∧ a = -28 :=
sorry

end parabola_equation_from_directrix_l398_398579


namespace isosceles_right_triangle_area_l398_398776

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l398_398776


namespace cake_pieces_l398_398810

-- Definition of the problem
def cake_cutting (cake : Type) (cuts : list (cake → cake)) : ℕ :=
  sorry -- This will define the structure for counting pieces of the cake

-- Condition definitions
def parallel_cuts (cake : Type) (cut1 cut2 : cake → cake) : Prop :=
  sorry -- This will define the relationship that describes parallel cuts

def intersecting_cuts (cake : Type) (cut1 cut2 : cake → cake) : Prop :=
  sorry -- This will define the relationship that describes intersecting cuts

-- Theorem stating the result based on the conditions
theorem cake_pieces (cake : Type) (cut1 cut2 : cake → cake) :
  (parallel_cuts cake cut1 cut2 → cake_cutting cake [cut1, cut2] = 3) ∧
  (intersecting_cuts cake cut1 cut2 → cake_cutting cake [cut1, cut2] = 4) :=
sorry

end cake_pieces_l398_398810


namespace intersection_empty_iff_l398_398072

variable {t x : ℝ}

def set_A : set ℝ := { x | abs (x - 2) ≤ 3 }
def set_B (t : ℝ) : set ℝ := { x | x < t }

theorem intersection_empty_iff (t : ℝ) :
  (set_A ∩ set_B t = ∅) ↔ (t ≤ -1) :=
by sorry

end intersection_empty_iff_l398_398072


namespace number_of_cows_is_26_l398_398220

-- Definition of the problem's conditions
def total_bags := 26
def total_days := 26
def consumption_one_cow := 1 / total_days

-- Definition of total consumption rate
def total_consumption_rate : ℝ := total_bags / total_days

-- Definition of consumption rate per cow
def one_cow_consumption_rate : ℝ := consumption_one_cow

-- Result to prove: Number of cows
def num_cows := total_consumption_rate / one_cow_consumption_rate

theorem number_of_cows_is_26 : num_cows = 26 :=
by
  sorry

end number_of_cows_is_26_l398_398220


namespace sum_of_consecutive_page_numbers_l398_398786

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end sum_of_consecutive_page_numbers_l398_398786


namespace largest_m_l398_398838

noncomputable def log_not_int (b : ℕ) (m : ℕ) : Prop :=
  ∀ k, m ≤ k ∧ k ≤ m + 2017 → ¬∃ n : ℕ, Real.log k / Real.log b = n

noncomputable def least_m (b : ℕ) : ℕ :=
  Nat.find (Exists.intro (b + 1) (log_not_int b (b + 1)))

theorem largest_m : ∀ (b : ℕ), b ≥ 2 → least_m b ≤ 2188 := by
  intro b h
  sorry

end largest_m_l398_398838


namespace range_of_b_over_a_l398_398971

theorem range_of_b_over_a (a b c : ℝ) (h : ∃ (r1 r2 r3 : ℝ), r1 * r2 = 1 ∧ r1 * r3 = 1 ∧ r2 * r3 ≠ 1 ∧ ∀ r, r ∈ {r1, r2, r3} → r < 1 ∨ r > 1) :
  -2 < b / a ∧ b / a < -0.5 :=
sorry

end range_of_b_over_a_l398_398971


namespace plant_cost_and_max_green_lily_students_l398_398245

-- Given conditions
def two_green_lily_three_spider_plants_cost (x y : ℕ) : Prop :=
  2 * x + 3 * y = 36

def one_green_lily_two_spider_plants_cost (x y : ℕ) : Prop :=
  x + 2 * y = 21

def total_students := 48

def cost_constraint (x y m : ℕ) : Prop :=
  9 * m + 6 * (48 - m) ≤ 378

-- Prove that x = 9, y = 6 and m ≤ 30
theorem plant_cost_and_max_green_lily_students :
  ∃ x y m : ℕ, two_green_lily_three_spider_plants_cost x y ∧ 
               one_green_lily_two_spider_plants_cost x y ∧ 
               cost_constraint x y m ∧ 
               x = 9 ∧ y = 6 ∧ m ≤ 30 :=
by
  sorry

end plant_cost_and_max_green_lily_students_l398_398245


namespace polynomial_evaluation_l398_398279

-- Define the polynomial p(x) and the conditions
noncomputable def p (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

-- Given conditions for p(1), p(2), p(3)
variables (a b c d : ℝ)
axiom h₁ : p 1 a b c d = 1993
axiom h₂ : p 2 a b c d = 3986
axiom h₃ : p 3 a b c d = 5979

-- The final proof statement
theorem polynomial_evaluation :
  (1 / 4) * (p 11 a b c d + p (-7) a b c d) = 5233 :=
sorry

end polynomial_evaluation_l398_398279


namespace distance_XT_l398_398859

theorem distance_XT {ABCD : Type*} (AB : ℝ) (height : ℝ) (ratio : ℝ) : 
  AB = 10 → height = 20 → ratio = 9 → let XT := 20 in XT = 20 ∧ (20 : ℤ) + (1 : ℤ) = 21 :=
begin
  intros h1 h2 h3,
  use 20,
  split,
  { refl },
  { ring }
end

end distance_XT_l398_398859


namespace largest_square_advertisement_area_l398_398462

theorem largest_square_advertisement_area
  (width : ℝ) (height : ℝ) (border : ℝ)
  (hw : width = 9) (hh : height = 16) (hb : border = 1.5) :
  (max_square_area : ℝ) = 36 :=
by
  -- let side length of the square be s
  let s := min (width - 2 * border) (height - 2 * border)
  -- prove the side length is 6
  have hs : s = 6 := by sorry
  -- prove the area of the largest square advertisement is 36
  have : s * s = 36 := by sorry
  -- explicit return the value for max_square_area
  exact this

end largest_square_advertisement_area_l398_398462


namespace two_cards_sum_to_15_proof_l398_398384

def probability_two_cards_sum_to_15 : ℚ := 32 / 884

theorem two_cards_sum_to_15_proof :
  let deck := { card | card ∈ set.range(2, 10 + 1) }
  ∀ (c1 c2 : ℕ), c1 ∈ deck → c2 ∈ deck → c1 ≠ c2 →  
  let chosen_cards := {c1, c2} in
  let sum := c1 + c2 in
  sum = 15 →
  (chosen_cards.probability = probability_two_cards_sum_to_15) :=
sorry

end two_cards_sum_to_15_proof_l398_398384


namespace trapezoid_circumcircle_radius_l398_398778

theorem trapezoid_circumcircle_radius :
  ∀ (BC AD height midline R : ℝ), 
  (BC / AD = (5 / 12)) →
  (height = 17) →
  (midline = height) →
  (midline = (BC + AD) / 2) →
  (BC = 10) →
  (AD = 24) →
  R = 13 :=
by
  intro BC AD height midline R
  intros h_ratio h_height h_midline_eq_height h_midline_eq_avg_bases h_BC h_AD
  -- Proof would go here, but it's skipped for now.
  sorry

end trapezoid_circumcircle_radius_l398_398778


namespace find_original_number_l398_398713

/-- Given that one less than the reciprocal of a number is 5/2, the original number must be -2/3. -/
theorem find_original_number (y : ℚ) (h : 1 - 1 / y = 5 / 2) : y = -2 / 3 :=
sorry

end find_original_number_l398_398713


namespace geometric_seq_log_sum_l398_398626

noncomputable def a_n : ℕ → ℝ := sorry -- Not assuming the specific function of a_n

theorem geometric_seq_log_sum :
  (∀ n m : ℕ, a_n * a_m = a_(n+1) * a_(m+1)) ∧ -- this encodes the geometric property
  (∀ n : ℕ, 0 < a_n) ∧ -- this encodes the positivity property
  (a_10 * a_11 + a_9 * a_12 = 2 * real.exp(5)) -- the given condition
  → (finset.sum (finset.range 20) (λ i, real.log (a_n i)) = 50) :=
by sorry

end geometric_seq_log_sum_l398_398626


namespace calculate_subtraction_l398_398491

def base9_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

def base6_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

theorem calculate_subtraction : base9_to_base10 324 - base6_to_base10 231 = 174 :=
  by sorry

end calculate_subtraction_l398_398491


namespace smallest_number_is_28_l398_398802

theorem smallest_number_is_28 (a b c : ℕ) (h1 : (a + b + c) / 3 = 30) (h2 : b = 28) (h3 : b = c - 6) : a = 28 :=
by sorry

end smallest_number_is_28_l398_398802


namespace necessary_not_sufficient_condition_for_monotonicity_l398_398123

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x < 2 then -x^2 + 2*a*x - 1 else Real.log (x-1) / Real.log a + 2*a

theorem necessary_not_sufficient_condition_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 2 ≤ a ∧ a ≤ 5 / 2 :=
sorry

end necessary_not_sufficient_condition_for_monotonicity_l398_398123


namespace quadratic_roots_identity_l398_398174

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398174


namespace quadratic_roots_vieta_l398_398167

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398167


namespace not_mapping_f4_l398_398137

open Set

def A : Set ℝ := Icc 0 8
def B : Set ℝ := Icc 0 4

def f1 (x : ℝ) : ℝ := (1/8) * x
def f2 (x : ℝ) : ℝ := (1/4) * x
def f3 (x : ℝ) : ℝ := (1/2) * x
def f4 (x : ℝ) : ℝ := x

theorem not_mapping_f4 : ¬ (∀ x ∈ A, f4 x ∈ B) :=
by 
  sorry

end not_mapping_f4_l398_398137


namespace vector_addition_proof_l398_398494

-- Considering the definitions:
def v1 := ⟨ 3, -2, 5 ⟩ : ℝ^3
def s := 2 : ℝ
def v2 := ⟨ -1, 4, -3 ⟩ : ℝ^3
def res := ⟨ 1, 6, -1 ⟩ : ℝ^3

-- The final proof statement:
theorem vector_addition_proof :
  v1 + s • v2 = res :=
by
  sorry

end vector_addition_proof_l398_398494


namespace prove_incorrect_judgments_l398_398356

-- Define conditions and propositions
variables {a b : ℝ} {A θ : ℝ} (p q : Prop)
local notation "sin " x => Real.sin x
local notation "cos " x => Real.cos x
local notation "tan " x => Real.tan x

-- Judgment 1 condition
def judgment1 := ∀ a b : ℝ, a + b ≠ 6 → (a ≠ 3 ∨ b ≠ 3) = false

-- Judgment 2 condition
def judgment2 := (p ∨ q) → p ∧ q = true

-- Judgment 3 condition
def judgment3 := (A > 30) → (sin A > 1/2) ∧ ¬(sin A > 1/2 ←→ 30 < A ∧ A < 150)

-- Judgment 4 condition
def judgment4 := let 
  a := (sin (2 * θ), cos θ),
  b := (cos θ, 1)
  in ((a.1 * b.2 = a.2 * b.1) ∨ tan θ ≠ 1/2) = true

noncomputable def incorrect_judgments : bool :=
  (judgment1 ∧ judgment2 ∧ judgment3 ∧ ¬judgment4) = (true)

-- Theorem stating that incorrect judgments are 1, 2, and 3 by the condition definitions
theorem prove_incorrect_judgments : incorrect_judgments = true :=
  sorry

end prove_incorrect_judgments_l398_398356


namespace hyperbola_circle_tangent_asymptote_l398_398627

theorem hyperbola_circle_tangent_asymptote (k : ℝ) (h₀ : k > 0) : 
  let asymptote_tangent_to_circle := (x^2 + (y - 2)^2 = 1 ∧ ∀ y : ℝ, y = k * x ∨ y = -k * x) in 
  let circle := (x^2 + (y - 2)^2 = 1) in
  asymptote_tangent_to_circle → k = real.sqrt 3 :=
by
  sorry

end hyperbola_circle_tangent_asymptote_l398_398627


namespace region_area_eq_72_l398_398815

theorem region_area_eq_72 : 
  let region := { (x, y) | |4 * x - 20| + |3 * y + 9| ≤ 6 } in
  set.area region = 72 :=
sorry

end region_area_eq_72_l398_398815


namespace sum_of_divisors_of_57_l398_398417

theorem sum_of_divisors_of_57 : 
  let divisors := [1, 3, 19, 57] in
  divisors.sum = 80 := 
by 
  sorry

end sum_of_divisors_of_57_l398_398417


namespace largest_value_a_plus_b_plus_c_l398_398264

open Nat
open Function

def sum_of_digits (n : ℕ) : ℕ :=
  (digits 10 n).sum

theorem largest_value_a_plus_b_plus_c :
  ∃ (a b c : ℕ),
    10 ≤ a ∧ a < 100 ∧
    100 ≤ b ∧ b < 1000 ∧
    1000 ≤ c ∧ c < 10000 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    (a + b + c = 10199) := sorry

end largest_value_a_plus_b_plus_c_l398_398264


namespace complement_set_l398_398687

theorem complement_set :
  let U := set.Ioo (-4 : ℝ) 4
  let A := {x : ℝ | -2 ≤ x ∧ x < 0}
  set.compl A ∩ U = set.Ioo (-4 : ℝ) (-2) ∪ set.Ico 0 4 :=
by {
  intros U A,
  sorry
}

end complement_set_l398_398687


namespace student_grades_l398_398872

theorem student_grades (grades : List ℕ) (h1 : ∀ g ∈ grades, g ∈ {2, 3, 4, 5})
  (h2 : grades.length = 13) (h3 : (grades.sum / 13 : ℚ).den = 1) :
  ∃ g ∈ {2, 3, 4, 5}, grades.count g ≤ 2 :=
by
  sorry

end student_grades_l398_398872


namespace angle_bd_obtuse_l398_398292

-- Define the geometrical setting of the problem
structure Triangle (α β γ : Type*) :=
(A B C : α)
(B_ext_C : β = γ)

def extension_laid_off (A B C D : Triangle ℝ) (h : B = D) : Prop :=
  ∃ (AC_ext : ℝ) (BC : ℝ), AC_ext > BC ∧ D = B_ext_C

theorem angle_bd_obtuse 
  (A B C D : Triangle ℝ)
  (h1 : AC > BC)
  (h2 : extension_laid_off A B C D (BD = BC)) :
  ∠ABD > 90 :=
sorry

end angle_bd_obtuse_l398_398292


namespace quarter_more_than_whole_l398_398825

theorem quarter_more_than_whole (x : ℝ) (h : x / 4 = 9 + x) : x = -12 :=
by
  sorry

end quarter_more_than_whole_l398_398825


namespace marbles_in_jar_l398_398850

theorem marbles_in_jar (M : ℕ) (h1 : ∀ n : ℕ, n = 20 → ∀ m : ℕ, m = M / n → ∀ a b : ℕ, a = n + 2 → b = m - 1 → ∀ k : ℕ, k = M / a → k = b) : M = 220 :=
by 
  sorry

end marbles_in_jar_l398_398850


namespace correct_option_l398_398671

variable {S : Type*} (A B : Set S) (a : S)

-- Conditions
variable (hA : A.Nonempty) (hB : B.Nonempty) (hAprop : A ⊂ Set.univ) (hBprop : B ⊂ Set.univ)
variable (haA : a ∈ A) (haB : a ∉ B)

-- Proof Goal
theorem correct_option : a ∈ (A ∩ Bᶜ) :=
sorry

end correct_option_l398_398671


namespace parabola_range_l398_398147

theorem parabola_range (a b c : ℝ) (h₁ : a < 0) :
  ∃ m M, (∀ x ∈ (Icc 0 1 : set ℝ), f(x) ∈ Icc c (-b^2 / (4 * a) + c)) :=
begin
  -- Definitions and proof would go here
  sorry
end

noncomputable def f (x : ℝ) := a * x^2 + b * x + c

end parabola_range_l398_398147


namespace triangle_AOB_area_l398_398222

noncomputable def point := ℝ × ℝ

noncomputable def origin : point := (0, 0)

noncomputable def A : point := (2, 2 * Real.pi / 3)

noncomputable def B : point := (3, Real.pi / 6)

def triangle_area (O A B : point) : ℝ :=
  let (r1, θ1) := A
  let (r2, θ2) := B
  let angle_AOB := θ1 - θ2
  let area := 0.5 * r1 * r2 * Real.sin angle_AOB
  area

theorem triangle_AOB_area (O A B : point) (hO : O = origin) (hA : A = (2, 2 * Real.pi / 3)) (hB : B = (3, Real.pi / 6)) :
  triangle_area O A B = 3 := by
  sorry

end triangle_AOB_area_l398_398222


namespace tangents_perpendicular_at_1_max_value_of_m_l398_398129

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g1 (x : ℝ) : ℝ := (x + 5) / (x + 1)
noncomputable def g2 (x : ℝ) (m : ℝ) : ℝ := m * (x - 1) / (x + 1)

-- Problem 1: Given functions f and g1, prove their tangents at x = 1 are perpendicular
theorem tangents_perpendicular_at_1 :
  (Deriv f 1) * (Deriv g1 1) = -1 := sorry

-- Problem 2: Given functions f and g2, prove that if ∀ x > 0, |f(x)| ≥ |g2(x, m)| then n = -1 and max value of m = 2
theorem max_value_of_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x > 0 → abs (f x) ≥ abs (g2 x m)) →
  n = -1 ∧ m ≤ 2 := sorry

end tangents_perpendicular_at_1_max_value_of_m_l398_398129


namespace find_m_l398_398761

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 4 :=
by 
  sorry

end find_m_l398_398761


namespace sophia_finished_more_pages_l398_398740

noncomputable def length_of_book : ℝ := 89.99999999999999

noncomputable def total_pages : ℕ := 90  -- Considering the practical purpose

noncomputable def finished_pages : ℕ := total_pages * 2 / 3

noncomputable def remaining_pages : ℕ := total_pages - finished_pages

theorem sophia_finished_more_pages :
  finished_pages - remaining_pages = 30 := 
  by
    -- Use sorry here as placeholder for the proof
    sorry

end sophia_finished_more_pages_l398_398740


namespace clover_count_l398_398635

theorem clover_count :
  let total_clovers := 500 in
  let four_leaved_percentage := 0.20 in
  let purple_four_leaved_fraction := 1 / 4 in
  let four_leaved_clovers := four_leaved_percentage * total_clovers in
  let purple_four_leaved_clovers := purple_four_leaved_fraction * four_leaved_clovers in
  purple_four_leaved_clovers = 25 :=
by
  sorry

end clover_count_l398_398635


namespace trapezium_height_l398_398043

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end trapezium_height_l398_398043


namespace trapezium_height_l398_398047

theorem trapezium_height :
  ∀ (a b h : ℝ), a = 20 ∧ b = 18 ∧ (1 / 2) * (a + b) * h = 285 → h = 15 :=
by
  intros a b h hconds
  cases hconds with h1 hrem
  cases hrem with h2 harea
  simp at harea
  sorry

end trapezium_height_l398_398047


namespace pentagon_diagonal_ratio_l398_398521

theorem pentagon_diagonal_ratio (ABCDE : convex_polygon) (h1 : ∀ diag ∈ diagonals ABCDE, ∃ side ∈ sides ABCDE, is_parallel diag side) :
  ∃ k, k = (1 + Real.sqrt 5) / 2 ∧ ∀ diag ∈ diagonals ABCDE, ∃ side ∈ sides ABCDE, parallel_side_ratio diag side = k :=
  sorry

end pentagon_diagonal_ratio_l398_398521


namespace number_of_valid_x_l398_398515

noncomputable def count_valid_x_values : ℕ :=
  let lower_bound : ℕ := 25
  let upper_bound : ℕ := 33  -- Since ⌊100/3⌋ = 33
  let valid_xs := (list.range' 25 (34 - 25))  -- List of integers [25..33] inclusive
  list.length valid_xs

theorem number_of_valid_x : count_valid_x_values = 9 := 
by
  -- Prove that count_valid_x_values = 9
  sorry

end number_of_valid_x_l398_398515


namespace find_d_l398_398784

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_d (d e : ℝ) (h1 : -(-6) / 3 = 2) (h2 : 3 + d + e - 6 = 9) (h3 : -d / 3 = 6) : d = -18 :=
by
  sorry

end find_d_l398_398784


namespace find_first_month_sale_l398_398849

/-- Given the sales for months two to six and the average sales over six months,
    prove the sale in the first month. -/
theorem find_first_month_sale
  (sales_2 : ℤ) (sales_3 : ℤ) (sales_4 : ℤ) (sales_5 : ℤ) (sales_6 : ℤ)
  (avg_sales : ℤ)
  (h2 : sales_2 = 5468) (h3 : sales_3 = 5568) (h4 : sales_4 = 6088)
  (h5 : sales_5 = 6433) (h6 : sales_6 = 5922) (h_avg : avg_sales = 5900) : 
  ∃ (sale_1 : ℤ), sale_1 = 5921 := 
by
  have total_sales : ℤ := avg_sales * 6
  have known_sales_sum : ℤ := sales_2 + sales_3 + sales_4 + sales_5
  use total_sales - known_sales_sum - sales_6
  sorry

end find_first_month_sale_l398_398849


namespace shorter_diagonal_from_obtuse_angle_l398_398300

-- Define the points and properties of the parallelogram
variables {A B C D O : Type} [Parallelogram A B C D O]
variables (AC BD : ℝ)

-- Definitions and given conditions
def diagonals_bisect_each_other : Prop := ∀ O, midpoint O A C ∧ midpoint O B D
def shorter_diagonal_condition : Prop := AC < BD
def obtuse_angle_condition {ABCD : Parallelogram A B C D O} : Prop := ∃ α : ℝ, α > (π / 2) ∧ angle_at_B_A_D ABCD α

-- The actual Lean statement to prove the shorter diagonal originates from the obtuse angle
theorem shorter_diagonal_from_obtuse_angle {ABCD : Parallelogram A B C D O} :
  diagonals_bisect_each_other A B C D O ∧ shorter_diagonal_condition AC BD →
  ∃ α : ℝ, α > (π / 2) ∧ angle_at_B_A_D ABCD α ∧ shorter_diagonal_is_AC A B C D α :=
sorry -- Proof is omitted

end shorter_diagonal_from_obtuse_angle_l398_398300


namespace even_gt_one_square_gt_l398_398620

theorem even_gt_one_square_gt (m : ℕ) (h_even : ∃ k : ℕ, m = 2 * k) (h_gt_one : m > 1) : m < m * m :=
by
  sorry

end even_gt_one_square_gt_l398_398620


namespace roots_of_quadratic_eq_l398_398154

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l398_398154


namespace base_three_to_decimal_l398_398747

theorem base_three_to_decimal :
  let n := 20121 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 178 :=
by {
  sorry
}

end base_three_to_decimal_l398_398747


namespace part_1_part_2_l398_398285

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def A_def : A = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  ext x
  sorry
  
def B_def : B = {x : ℝ | x^2 + 2*x - 3 > 0} := by
  ext x
  sorry

theorem part_1 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0} := by
  rw [hA, hB]
  sorry

theorem part_2 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  (compl A ∩ B) = {x | x > 1 ∨ x < -3} := by
  rw [hA, hB]
  sorry

end part_1_part_2_l398_398285


namespace factorial_236_trailing_zeros_l398_398781

theorem factorial_236_trailing_zeros : 
  let count_trailing_zeros := (236 / 5).floor + (236 / 25).floor + (236 / 125).floor
  in count_trailing_zeros = 57 :=
by
  sorry

end factorial_236_trailing_zeros_l398_398781


namespace number_of_ways_l398_398349

-- Define the conditions
def num_people : ℕ := 3
def num_sports : ℕ := 4

-- Prove the total number of different ways
theorem number_of_ways : num_sports ^ num_people = 64 := by
  sorry

end number_of_ways_l398_398349


namespace line_equation_through_P_l398_398910

theorem line_equation_through_P (P : ℝ × ℝ) (x y : ℝ) 
  (hP : P = (2, 1))
  (hAngle : ∀ θ, tan θ = 1/2 → tan (2 * θ) = 4/3)
  (hLine : x - 2 * y - 1 = 0) : 
  4 * x - 3 * y - 5 = 0 :=
sorry

end line_equation_through_P_l398_398910


namespace two_cards_totaling_15_probability_l398_398371

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l398_398371


namespace different_enrollment_methods_l398_398355

theorem different_enrollment_methods (n_students n_subjects : ℕ) :
  n_students = 4 → n_subjects = 3 → (n_subjects ^ n_students) = 81 :=
by
  intros h_students h_subjects
  rw [h_students, h_subjects]
  exact eq.refl 81

end different_enrollment_methods_l398_398355


namespace no_intersection_points_l398_398902

def intersection_points_eq_zero : Prop :=
∀ x y : ℝ, (y = abs (3 * x + 6)) ∧ (y = -abs (4 * x - 3)) → false

theorem no_intersection_points :
  intersection_points_eq_zero :=
by
  intro x y h
  cases h
  sorry

end no_intersection_points_l398_398902


namespace quadratic_roots_vieta_l398_398171

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l398_398171


namespace cricket_team_opponent_total_runs_l398_398442

theorem cricket_team_opponent_total_runs 
    (team_scores : Fin 12 → ℕ)
    (lost_matches_indexes : Fin 12 → Prop)
    (triple_score_indexes : Fin 12 → Prop)
    (half_score_indexes : Fin 12 → Prop)
    (lost_by_two : ∀ (i : Fin 12), lost_matches_indexes i → team_scores i + 2 = opponent_scores i)
    (triple_score : ∀ (i : Fin 12), triple_score_indexes i → team_scores i = 3 * opponent_scores i)
    (half_score : ∀ (i : Fin 12), half_score_indexes i → 2 * team_scores i = opponent_scores i)
    (total_matches_conditions : (∑ i, lost_matches_indexes i + ∑ i, triple_score_indexes i + ∑ i, half_score_indexes i = 12))
    (team_scores_all_matches : team_scores = ![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    : (∑ i, opponent_scores i) = 97 := sorry

end cricket_team_opponent_total_runs_l398_398442


namespace parallel_segments_in_2n_gon_l398_398275

theorem parallel_segments_in_2n_gon (n : ℕ) (P : Fin (2 * n) → Fin (2 * n)) (G : ∀ i, P (Fin.next i) ≠ P i) :
  (∃ i j : Fin (2 * n),
    (i ≠ j ∧ (P i + P (Fin.next i) = P j + P (Fin.next j) % (2 * n)))) := sorry

end parallel_segments_in_2n_gon_l398_398275


namespace cosine_of_angle_between_tangents_l398_398558

-- Definitions based on the conditions given in a)
def circle_eq (x y : ℝ) : Prop := x^2 - 2 * x + y^2 - 2 * y + 1 = 0
def P : ℝ × ℝ := (3, 2)

-- The main theorem to be proved
theorem cosine_of_angle_between_tangents (x y : ℝ)
  (hx : circle_eq x y) : 
  cos_angle_between_tangents := 
  sorry

end cosine_of_angle_between_tangents_l398_398558


namespace min_rooms_needed_l398_398029

-- Define the conditions of the problem
def max_people_per_room := 3
def total_fans := 100

inductive Gender | male | female
inductive Team | teamA | teamB | teamC

structure Fan :=
  (gender : Gender)
  (team : Team)

def fans : List Fan := List.replicate 100 ⟨Gender.male, Team.teamA⟩ -- Assuming a placeholder list of fans

-- Define the condition that each group based on gender and team must be considered separately
def groupFans (fans : List Fan) : List (List Fan) := 
  List.groupBy (fun f => (f.gender, f.team)) fans

-- Statement of the problem as a theorem in Lean 4
theorem min_rooms_needed :
  ∃ (rooms_needed : Nat), rooms_needed = 37 :=
by
  have h1 : ∀fan_group, (length fan_group) ≤ max_people_per_room → 1
  have h2 : ∀fan_group, (length fan_group) % max_people_per_room ≠ 0 goes to additional rooms properly.
  have h3 : total_fans = List.length fans
  -- calculations and conditions would follow matched to the above defined rules. 
  sorry

end min_rooms_needed_l398_398029


namespace quadratic_roots_identity_l398_398175

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l398_398175


namespace find_y_l398_398215

theorem find_y (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 :=
by sorry

end find_y_l398_398215


namespace min_distance_max_distance_l398_398242

-- Define the basic conditions of the problem
def satisfies_equation (x y : ℤ) : Prop := y^2 = 4 * x^2 - 15

-- Define the function to compute distance between two points
def distance (p1 p2 : ℤ × ℤ) : ℝ := 
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the statement for the minimum distance
theorem min_distance :
  ∀ (p1 p2 : ℤ × ℤ), 
  satisfies_equation p1.1 p1.2 →
  satisfies_equation p2.1 p2.2 →
  distance p1 p2 ≥ 2 := sorry

-- Define the statement for the maximum distance
theorem max_distance :
  ∃ (p1 p2 : ℤ × ℤ), 
  satisfies_equation p1.1 p1.2 ∧
  satisfies_equation p2.1 p2.2 ∧ 
  distance p1 p2 = 2 * Real.sqrt 65 := sorry

end min_distance_max_distance_l398_398242


namespace event_B_more_likely_than_event_A_l398_398706

-- Definitions based on given conditions
def total_possible_outcomes := 6^3
def favorable_outcomes_B := (Nat.choose 6 3) * (Nat.factorial 3)
def prob_B := favorable_outcomes_B / total_possible_outcomes
def prob_A := 1 - prob_B

-- The theorem to be proved:
theorem event_B_more_likely_than_event_A (total_possible_outcomes = 216) 
    (favorable_outcomes_B = 120) 
    (prob_B = 5 / 9) 
    (prob_A = 4 / 9) :
    prob_B > prob_A := 
by {
    sorry
}

end event_B_more_likely_than_event_A_l398_398706
