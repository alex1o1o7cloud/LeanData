import Mathlib

namespace supermarket_A_is_more_cost_effective_l477_47750

def price_A (kg : ℕ) : ℕ :=
  if kg <= 4 then kg * 10
  else 4 * 10 + (kg - 4) * 6

def price_B (kg : ℕ) : ℕ :=
  kg * 10 * 8 / 10

theorem supermarket_A_is_more_cost_effective :
  price_A 3 = 30 ∧ 
  price_A 5 = 46 ∧ 
  ∀ (x : ℕ), (x > 4) → price_A x = 6 * x + 16 ∧ 
  price_A 10 < price_B 10 :=
by 
  sorry

end supermarket_A_is_more_cost_effective_l477_47750


namespace smaller_angle_at_10_15_p_m_l477_47705

-- Definitions of conditions
def clock_hours : ℕ := 12
def degrees_per_hour : ℚ := 360 / clock_hours
def minute_hand_position : ℚ := (15 / 60) * 360
def hour_hand_position : ℚ := 10 * degrees_per_hour + (15 / 60) * degrees_per_hour
def absolute_difference : ℚ := |hour_hand_position - minute_hand_position|
def smaller_angle : ℚ := 360 - absolute_difference

-- Prove that the smaller angle is 142.5°
theorem smaller_angle_at_10_15_p_m : smaller_angle = 142.5 := by
  sorry

end smaller_angle_at_10_15_p_m_l477_47705


namespace pieces_of_wood_for_table_l477_47733

theorem pieces_of_wood_for_table :
  ∀ (T : ℕ), (24 * T + 48 * 8 = 672) → T = 12 :=
by
  intro T
  intro h
  sorry

end pieces_of_wood_for_table_l477_47733


namespace johnny_yellow_picks_l477_47768

variable (total_picks red_picks blue_picks yellow_picks : ℕ)

theorem johnny_yellow_picks
    (h_total_picks : total_picks = 3 * blue_picks)
    (h_half_red_picks : red_picks = total_picks / 2)
    (h_blue_picks : blue_picks = 12)
    (h_pick_sum : total_picks = red_picks + blue_picks + yellow_picks) :
    yellow_picks = 6 := by
  sorry

end johnny_yellow_picks_l477_47768


namespace sum_of_numbers_l477_47770

theorem sum_of_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 54) (h_ratio : a / b = 2 / 3) : a + b = 45 :=
by
  sorry

end sum_of_numbers_l477_47770


namespace shaded_region_area_l477_47763

-- Given conditions
def diagonal_PQ : ℝ := 10
def number_of_squares : ℕ := 20

-- Definition of the side length of the squares
noncomputable def side_length := diagonal_PQ / (4 * Real.sqrt 2)

-- Area of one smaller square
noncomputable def one_square_area := side_length * side_length

-- Total area of the shaded region
noncomputable def total_area_of_shaded_region := number_of_squares * one_square_area

-- The theorem to be proven
theorem shaded_region_area : total_area_of_shaded_region = 62.5 := by
  sorry

end shaded_region_area_l477_47763


namespace fraction_increase_by_two_l477_47785

theorem fraction_increase_by_two (x y : ℝ) : 
  (3 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * (3 * x * y) / (x + y) :=
by
  sorry

end fraction_increase_by_two_l477_47785


namespace monogramming_cost_per_stocking_l477_47725

noncomputable def total_stockings : ℕ := (5 * 5) + 4
noncomputable def price_per_stocking : ℝ := 20 - (0.10 * 20)
noncomputable def total_cost_of_stockings : ℝ := total_stockings * price_per_stocking
noncomputable def total_cost : ℝ := 1035
noncomputable def total_monogramming_cost : ℝ := total_cost - total_cost_of_stockings

theorem monogramming_cost_per_stocking :
  (total_monogramming_cost / total_stockings) = 17.69 :=
by
  sorry

end monogramming_cost_per_stocking_l477_47725


namespace ned_short_sleeve_shirts_l477_47762

theorem ned_short_sleeve_shirts (washed_shirts not_washed_shirts long_sleeve_shirts total_shirts : ℕ)
  (h1 : washed_shirts = 29) (h2 : not_washed_shirts = 1) (h3 : long_sleeve_shirts = 21)
  (h4 : total_shirts = washed_shirts + not_washed_shirts) :
  total_shirts - long_sleeve_shirts = 9 :=
by
  sorry

end ned_short_sleeve_shirts_l477_47762


namespace carl_garden_area_l477_47712

theorem carl_garden_area (total_posts : ℕ) (post_interval : ℕ) (x_posts_on_shorter : ℕ) (y_posts_on_longer : ℕ)
  (h1 : total_posts = 26)
  (h2 : post_interval = 5)
  (h3 : y_posts_on_longer = 2 * x_posts_on_shorter)
  (h4 : 2 * x_posts_on_shorter + 2 * y_posts_on_longer - 4 = total_posts) :
  (x_posts_on_shorter - 1) * post_interval * (y_posts_on_longer - 1) * post_interval = 900 := 
by
  sorry

end carl_garden_area_l477_47712


namespace find_number_l477_47761

theorem find_number (x : ℤ) (h : 7 * x + 37 = 100) : x = 9 :=
by
  sorry

end find_number_l477_47761


namespace books_sold_correct_l477_47791

-- Define the number of books sold by Matias, Olivia, and Luke on each day
def matias_monday := 7
def olivia_monday := 5
def luke_monday := 12

def matias_tuesday := 2 * matias_monday
def olivia_tuesday := 3 * olivia_monday
def luke_tuesday := luke_monday / 2

def matias_wednesday := 3 * matias_tuesday
def olivia_wednesday := 4 * olivia_tuesday
def luke_wednesday := luke_tuesday

-- Calculate the total books sold by each person over three days
def matias_total := matias_monday + matias_tuesday + matias_wednesday
def olivia_total := olivia_monday + olivia_tuesday + olivia_wednesday
def luke_total := luke_monday + luke_tuesday + luke_wednesday

-- Calculate the combined total of books sold by Matias, Olivia, and Luke
def combined_total := matias_total + olivia_total + luke_total

-- Prove the combined total equals 167
theorem books_sold_correct : combined_total = 167 := by
  sorry

end books_sold_correct_l477_47791


namespace quadratic_function_opens_downwards_l477_47773

theorem quadratic_function_opens_downwards (m : ℤ) (h1 : |m| = 2) (h2 : m + 1 < 0) : m = -2 := by
  sorry

end quadratic_function_opens_downwards_l477_47773


namespace circumradius_inradius_inequality_l477_47764

theorem circumradius_inradius_inequality (a b c R r : ℝ) (hR : R > 0) (hr : r > 0) :
  R / (2 * r) ≥ ((64 * a^2 * b^2 * c^2) / 
  ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end circumradius_inradius_inequality_l477_47764


namespace find_2g_x_l477_47729

theorem find_2g_x (g : ℝ → ℝ) (h : ∀ x > 0, g (3 * x) = 3 / (3 + x)) (x : ℝ) (hx : x > 0) :
  2 * g x = 18 / (9 + x) :=
sorry

end find_2g_x_l477_47729


namespace hania_age_in_five_years_l477_47715

-- Defining the conditions
variables (H S : ℕ)

-- First condition: Samir's age will be 20 in five years
def condition1 : Prop := S + 5 = 20

-- Second condition: Samir is currently half the age Hania was 10 years ago
def condition2 : Prop := S = (H - 10) / 2

-- The statement to prove: Hania's age in five years will be 45
theorem hania_age_in_five_years (H S : ℕ) (h1 : condition1 S) (h2 : condition2 H S) : H + 5 = 45 :=
sorry

end hania_age_in_five_years_l477_47715


namespace sum_of_edges_corners_faces_of_rectangular_prism_l477_47732

-- Definitions based on conditions
def rectangular_prism_edges := 12
def rectangular_prism_corners := 8
def rectangular_prism_faces := 6
def resulting_sum := rectangular_prism_edges + rectangular_prism_corners + rectangular_prism_faces

-- Statement we want to prove
theorem sum_of_edges_corners_faces_of_rectangular_prism :
  resulting_sum = 26 := 
by 
  sorry -- Placeholder for the proof

end sum_of_edges_corners_faces_of_rectangular_prism_l477_47732


namespace choose_3_out_of_13_l477_47782

theorem choose_3_out_of_13: (Nat.choose 13 3) = 286 :=
by
  sorry

end choose_3_out_of_13_l477_47782


namespace bottles_produced_by_10_machines_in_4_minutes_l477_47723

variable (rate_per_machine : ℕ)
variable (total_bottles_per_minute_six_machines : ℕ := 240)
variable (number_of_machines : ℕ := 6)
variable (new_number_of_machines : ℕ := 10)
variable (time_in_minutes : ℕ := 4)

theorem bottles_produced_by_10_machines_in_4_minutes :
  rate_per_machine = total_bottles_per_minute_six_machines / number_of_machines →
  (new_number_of_machines * rate_per_machine * time_in_minutes) = 1600 := 
sorry

end bottles_produced_by_10_machines_in_4_minutes_l477_47723


namespace eval_floor_abs_value_l477_47734

theorem eval_floor_abs_value : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry -- Proof is to be filled in

end eval_floor_abs_value_l477_47734


namespace zookeeper_configurations_l477_47765

theorem zookeeper_configurations :
  ∃ (configs : ℕ), configs = 3 ∧ 
  (∀ (r p : ℕ), 
    30 * r + 35 * p = 1400 ∧ p ≥ r → 
    ((r, p) = (7, 34) ∨ (r, p) = (14, 28) ∨ (r, p) = (21, 22))) :=
sorry

end zookeeper_configurations_l477_47765


namespace geometric_sequence_a4_l477_47721

-- Define the geometric sequence and known conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ) (q : ℝ)

-- Given conditions:
def a2_eq_4 : Prop := a 2 = 4
def a6_eq_16 : Prop := a 6 = 16

-- The goal is to show a 4 = 8 given the conditions
theorem geometric_sequence_a4 (h_seq : geometric_sequence a q)
  (h_a2 : a2_eq_4 a)
  (h_a6 : a6_eq_16 a) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l477_47721


namespace find_m_from_root_l477_47779

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end find_m_from_root_l477_47779


namespace f_odd_and_inequality_l477_47741

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem f_odd_and_inequality (x c : ℝ) : ∀ x c, 
  f x < c^2 - 3 * c + 3 := by 
  sorry

end f_odd_and_inequality_l477_47741


namespace range_of_a_l477_47798

theorem range_of_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2)
    (hxy : x * y = 2) (h : ∀ x y, 2 - x ≥ a / (4 - y)) : a ≤ 0 :=
sorry

end range_of_a_l477_47798


namespace remainder_when_690_div_170_l477_47722

theorem remainder_when_690_div_170 :
  ∃ r : ℕ, ∃ k l : ℕ, 
    gcd (690 - r) (875 - 25) = 170 ∧
    r = 690 % 170 ∧
    l = 875 / 170 ∧
    r = 10 :=
by 
  sorry

end remainder_when_690_div_170_l477_47722


namespace arithmetic_sequence_sum_l477_47744

theorem arithmetic_sequence_sum :
  let a := -3
  let d := 7
  let n := 10
  let s := n * (2 * a + (n - 1) * d) / 2
  s = 285 :=
by
  -- Details of the proof are omitted as per instructions
  sorry

end arithmetic_sequence_sum_l477_47744


namespace card_combinations_l477_47709

noncomputable def valid_card_combinations : List (ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)]

theorem card_combinations (a b c d : ℕ) (h : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (1, 2, 7, 8) ∈ valid_card_combinations ∨ 
  (1, 3, 6, 8) ∈ valid_card_combinations ∨ 
  (1, 4, 5, 8) ∈ valid_card_combinations ∨ 
  (2, 3, 6, 7) ∈ valid_card_combinations ∨ 
  (2, 4, 5, 7) ∈ valid_card_combinations ∨ 
  (3, 4, 5, 6) ∈ valid_card_combinations :=
sorry

end card_combinations_l477_47709


namespace find_breadth_of_rectangle_l477_47755

noncomputable def breadth_of_rectangle (s : ℝ) (π_approx : ℝ := 3.14) : ℝ :=
2 * s - 22

theorem find_breadth_of_rectangle (b s : ℝ) (π_approx : ℝ := 3.14) :
  4 * s = 2 * (22 + b) →
  π_approx * s / 2 + s = 29.85 →
  b = 1.22 :=
by
  intros h1 h2
  sorry

end find_breadth_of_rectangle_l477_47755


namespace volume_of_regular_tetrahedron_l477_47742

noncomputable def volume_of_tetrahedron (a H : ℝ) : ℝ :=
  (a^2 * H) / (6 * Real.sqrt 2)

theorem volume_of_regular_tetrahedron
  (d_face : ℝ)
  (d_edge : ℝ)
  (h : Real.sqrt 14 = d_edge)
  (h1 : 2 = d_face)
  (volume_approx : ℝ) :
  ∃ a H, (d_face = Real.sqrt ((H / 2)^2 + (a * Real.sqrt 3 / 6)^2) ∧ 
          d_edge = Real.sqrt ((H / 2)^2 + (a / (2 * Real.sqrt 3))^2) ∧ 
          Real.sqrt (volume_of_tetrahedron a H) = 533.38) :=
  sorry

end volume_of_regular_tetrahedron_l477_47742


namespace binomial_coefficient_10_3_l477_47788

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l477_47788


namespace find_theta_in_interval_l477_47746

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end find_theta_in_interval_l477_47746


namespace problem_proof_l477_47736

theorem problem_proof (p : ℕ) (hodd : p % 2 = 1) (hgt : p > 3):
  ((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 4) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p + 1) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 3) :=
by
  sorry

end problem_proof_l477_47736


namespace total_cakes_served_today_l477_47704

def cakes_served_lunch : ℕ := 6
def cakes_served_dinner : ℕ := 9
def total_cakes_served (lunch cakes_served_dinner : ℕ) : ℕ :=
  lunch + cakes_served_dinner

theorem total_cakes_served_today : total_cakes_served cakes_served_lunch cakes_served_dinner = 15 := 
by
  sorry

end total_cakes_served_today_l477_47704


namespace no_valid_partition_exists_l477_47751

namespace MathProof

-- Define the set of positive integers
def N := {n : ℕ // n > 0}

-- Define non-empty sets A, B, C which are disjoint and partition N
def valid_partition (A B C : N → Prop) : Prop :=
  (∃ a, A a) ∧ (∃ b, B b) ∧ (∃ c, C c) ∧
  (∀ n, A n → ¬ B n ∧ ¬ C n) ∧
  (∀ n, B n → ¬ A n ∧ ¬ C n) ∧
  (∀ n, C n → ¬ A n ∧ ¬ B n) ∧
  (∀ n, A n ∨ B n ∨ C n)

-- Define the conditions in the problem
def condition_1 (A B C : N → Prop) : Prop :=
  ∀ a b, A a → B b → C ⟨a.val + b.val + 1, by linarith [a.prop, b.prop]⟩

def condition_2 (A B C : N → Prop) : Prop :=
  ∀ b c, B b → C c → A ⟨b.val + c.val + 1, by linarith [b.prop, c.prop]⟩

def condition_3 (A B C : N → Prop) : Prop :=
  ∀ c a, C c → A a → B ⟨c.val + a.val + 1, by linarith [c.prop, a.prop]⟩

-- State the problem that no valid partition exists
theorem no_valid_partition_exists :
  ¬ ∃ (A B C : N → Prop), valid_partition A B C ∧
    condition_1 A B C ∧
    condition_2 A B C ∧
    condition_3 A B C :=
by
  sorry

end MathProof

end no_valid_partition_exists_l477_47751


namespace molecular_weight_is_correct_l477_47708

-- Define the masses of the individual isotopes
def H1 : ℕ := 1
def H2 : ℕ := 2
def O : ℕ := 16
def C : ℕ := 13
def N : ℕ := 15
def S : ℕ := 33

-- Define the molecular weight calculation
def molecular_weight : ℕ := (2 * H1) + H2 + O + C + N + S

-- The goal is to prove that the calculated molecular weight is 81
theorem molecular_weight_is_correct : molecular_weight = 81 :=
by 
  sorry

end molecular_weight_is_correct_l477_47708


namespace joseph_savings_ratio_l477_47710

theorem joseph_savings_ratio
    (thomas_monthly_savings : ℕ)
    (thomas_years_saving : ℕ)
    (total_savings : ℕ)
    (joseph_total_savings_is_total_minus_thomas : total_savings = thomas_monthly_savings * 12 * thomas_years_saving + (total_savings - thomas_monthly_savings * 12 * thomas_years_saving))
    (thomas_saves_each_month : thomas_monthly_savings = 40)
    (years_saving : thomas_years_saving = 6)
    (total_amount : total_savings = 4608) :
    (total_savings - thomas_monthly_savings * 12 * thomas_years_saving) / (12 * thomas_years_saving) / thomas_monthly_savings = 3 / 5 :=
by
  sorry

end joseph_savings_ratio_l477_47710


namespace sum_rational_irrational_not_rational_l477_47720

theorem sum_rational_irrational_not_rational (r i : ℚ) (hi : ¬ ∃ q : ℚ, i = q) : ¬ ∃ s : ℚ, r + i = s :=
by
  sorry

end sum_rational_irrational_not_rational_l477_47720


namespace ferry_P_travel_time_l477_47792

-- Definitions of conditions
def speed_P : ℝ := 6 -- speed of ferry P in km/h
def speed_diff_PQ : ℝ := 3 -- speed difference between ferry Q and ferry P in km/h
def travel_longer_Q : ℝ := 2 -- ferry Q travels a route twice as long as ferry P
def time_diff_PQ : ℝ := 1 -- time difference between ferry Q and ferry P in hours

-- Distance traveled by ferry P
def distance_P (t_P : ℝ) : ℝ := speed_P * t_P

-- Distance traveled by ferry Q
def distance_Q (t_P : ℝ) : ℝ := travel_longer_Q * (speed_P * t_P)

-- Speed of ferry Q
def speed_Q : ℝ := speed_P + speed_diff_PQ

-- Time taken by ferry Q
def time_Q (t_P : ℝ) : ℝ := t_P + time_diff_PQ

-- Main theorem statement
theorem ferry_P_travel_time (t_P : ℝ) : t_P = 3 :=
by
  have eq_Q : speed_Q * (time_Q t_P) = distance_Q t_P := sorry
  have eq_P : speed_P * t_P = distance_P t_P := sorry
  sorry

end ferry_P_travel_time_l477_47792


namespace remainder_140_div_k_l477_47787

theorem remainder_140_div_k (k : ℕ) (hk : k > 0) :
  (80 % k^2 = 8) → (140 % k = 2) :=
by
  sorry

end remainder_140_div_k_l477_47787


namespace difference_of_roots_l477_47707

theorem difference_of_roots : 
  let a := 6 + 3 * Real.sqrt 5
  let b := 3 + Real.sqrt 5
  let c := 1
  ∃ x1 x2 : ℝ, (a * x1^2 - b * x1 + c = 0) ∧ (a * x2^2 - b * x2 + c = 0) ∧ x1 ≠ x2 
  ∧ x1 > x2 ∧ (x1 - x2) = (Real.sqrt 6 - Real.sqrt 5) / 3 := 
sorry

end difference_of_roots_l477_47707


namespace hyperbola_parabola_focus_l477_47777

theorem hyperbola_parabola_focus (m : ℝ) :
  (m + (m - 2) = 4) → m = 3 :=
by
  intro h
  sorry

end hyperbola_parabola_focus_l477_47777


namespace hexagon_angles_sum_l477_47758

theorem hexagon_angles_sum (α β γ δ ε ζ : ℝ)
  (h1 : α + γ + ε = 180)
  (h2 : β + δ + ζ = 180) : 
  α + β + γ + δ + ε + ζ = 360 :=
by 
  sorry

end hexagon_angles_sum_l477_47758


namespace bisection_method_termination_condition_l477_47784

theorem bisection_method_termination_condition (x1 x2 : ℝ) (ε : ℝ) : Prop :=
  |x1 - x2| < ε

end bisection_method_termination_condition_l477_47784


namespace initial_volume_of_mixture_l477_47701

-- Define the conditions of the problem as hypotheses
variable (milk_ratio water_ratio : ℕ) (W : ℕ) (initial_mixture : ℕ)
variable (h1 : milk_ratio = 2) (h2 : water_ratio = 1)
variable (h3 : W = 60)
variable (h4 : water_ratio + milk_ratio = 3) -- The sum of the ratios used in the equation

theorem initial_volume_of_mixture : initial_mixture = 60 :=
by
  sorry

end initial_volume_of_mixture_l477_47701


namespace random_phenomenon_l477_47718

def is_certain_event (P : Prop) : Prop := ∀ h : P, true

def is_random_event (P : Prop) : Prop := ¬is_certain_event P

def scenario1 : Prop := ∀ pressure temperature : ℝ, (pressure = 101325) → (temperature = 100) → true
-- Under standard atmospheric pressure, water heated to 100°C will boil

def scenario2 : Prop := ∃ time : ℝ, true
-- Encountering a red light at a crossroads (which happens at random times)

def scenario3 (a b : ℝ) : Prop := true
-- For a rectangle with length and width a and b respectively, its area is a * b

def scenario4 : Prop := ∀ a b : ℝ, ∃ x : ℝ, a * x + b = 0
-- A linear equation with real coefficients always has one real root

theorem random_phenomenon : is_random_event scenario2 :=
by
  sorry

end random_phenomenon_l477_47718


namespace age_difference_l477_47702

theorem age_difference (B_age : ℕ) (A_age : ℕ) (X : ℕ) : 
  B_age = 42 → 
  A_age = B_age + 12 → 
  A_age + 10 = 2 * (B_age - X) → 
  X = 10 :=
by
  intros hB_age hA_age hEquation 
  -- define variables based on conditions
  have hB : B_age = 42 := hB_age
  have hA : A_age = B_age + 12 := hA_age
  have hEq : A_age + 10 = 2 * (B_age - X) := hEquation
  -- expected result
  sorry

end age_difference_l477_47702


namespace unit_cost_decreases_l477_47778

def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

theorem unit_cost_decreases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := 
by sorry


end unit_cost_decreases_l477_47778


namespace back_seat_people_l477_47786

/-- Define the number of seats on the left side of the bus --/
def left_side_seats : ℕ := 15

/-- Define the number of seats on the right side of the bus (3 fewer because of the rear exit door) --/
def right_side_seats : ℕ := left_side_seats - 3

/-- Define the number of people each seat can hold --/
def people_per_seat : ℕ := 3

/-- Define the total capacity of the bus --/
def total_capacity : ℕ := 90

/-- Define the total number of people that can sit on the regular seats (left and right sides) --/
def regular_seats_people := (left_side_seats + right_side_seats) * people_per_seat

/-- Theorem stating the number of people that can sit at the back seat --/
theorem back_seat_people : (total_capacity - regular_seats_people) = 9 := by
  sorry

end back_seat_people_l477_47786


namespace distance_after_four_steps_l477_47738

theorem distance_after_four_steps (total_distance : ℝ) (steps : ℕ) (steps_taken : ℕ) :
   total_distance = 25 → steps = 7 → steps_taken = 4 → (steps_taken * (total_distance / steps) = 100 / 7) :=
by
    intro h1 h2 h3
    rw [h1, h2, h3]
    simp
    sorry

end distance_after_four_steps_l477_47738


namespace triangle_sets_l477_47747

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end triangle_sets_l477_47747


namespace total_donuts_three_days_l477_47740

def donuts_on_Monday := 14

def donuts_on_Tuesday := donuts_on_Monday / 2

def donuts_on_Wednesday := 4 * donuts_on_Monday

def total_donuts := donuts_on_Monday + donuts_on_Tuesday + donuts_on_Wednesday

theorem total_donuts_three_days : total_donuts = 77 :=
  by
    sorry

end total_donuts_three_days_l477_47740


namespace find_distance_l477_47752

variable (y : ℚ) -- The circumference of the bicycle wheel
variable (x : ℚ) -- The distance between the village and the field

-- Condition 1: The circumference of the truck's wheel is 4/3 of the bicycle's wheel
def circum_truck_eq : Prop := (4 / 3 : ℚ) * y = y

-- Condition 2: The circumference of the truck's wheel is 2 meters shorter than the tractor's track
def circum_truck_less : Prop := (4 / 3 : ℚ) * y + 2 = y + 2

-- Condition 3: Truck's wheel makes 100 fewer revolutions than the bicycle's wheel
def truck_100_fewer : Prop := x / ((4 / 3 : ℚ) * y) = (x / y) - 100

-- Condition 4: Truck's wheel makes 150 more revolutions than the tractor track
def truck_150_more : Prop := x / ((4 / 3 : ℚ) * y) = (x / ((4 / 3 : ℚ) * y + 2)) + 150

theorem find_distance (y : ℚ) (x : ℚ) :
  circum_truck_eq y →
  circum_truck_less y →
  truck_100_fewer x y →
  truck_150_more x y →
  x = 600 :=
by
  intros
  sorry

end find_distance_l477_47752


namespace counter_represents_number_l477_47797

theorem counter_represents_number (a b : ℕ) : 10 * a + b = 10 * a + b := 
by 
  sorry

end counter_represents_number_l477_47797


namespace tile_border_ratio_l477_47716

theorem tile_border_ratio (n : ℕ) (t w : ℝ) (H1 : n = 30)
  (H2 : 900 * t^2 / (30 * t + 30 * w)^2 = 0.81) :
  w / t = 1 / 9 :=
by
  sorry

end tile_border_ratio_l477_47716


namespace count_4_digit_numbers_divisible_by_13_l477_47703

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l477_47703


namespace min_value_of_expression_l477_47726

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 4) :
  (9 / x + 1 / y + 25 / z) ≥ 20.25 :=
by 
  sorry

end min_value_of_expression_l477_47726


namespace rate_of_pipe_B_l477_47735

-- Definitions based on conditions
def tank_capacity : ℕ := 850
def pipe_A_rate : ℕ := 40
def pipe_C_rate : ℕ := 20
def cycle_time : ℕ := 3
def full_time : ℕ := 51

-- Prove that the rate of pipe B is 30 liters per minute
theorem rate_of_pipe_B (B : ℕ) : 
  (17 * (B + 20) = 850) → B = 30 := 
by 
  introv h1
  sorry

end rate_of_pipe_B_l477_47735


namespace dislikes_TV_and_books_l477_47783

-- The problem conditions
def total_people : ℕ := 800
def percent_dislikes_TV : ℚ := 25 / 100
def percent_dislikes_both : ℚ := 15 / 100

-- The expected answer
def expected_dislikes_TV_and_books : ℕ := 30

-- The proof problem statement
theorem dislikes_TV_and_books : 
  (total_people * percent_dislikes_TV) * percent_dislikes_both = expected_dislikes_TV_and_books := by 
  sorry

end dislikes_TV_and_books_l477_47783


namespace fraction_evaluation_l477_47756

theorem fraction_evaluation : (3 / 8 : ℚ) + 7 / 12 - 2 / 9 = 53 / 72 := by
  sorry

end fraction_evaluation_l477_47756


namespace simplify_fraction_l477_47724

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 :=
by
  sorry

end simplify_fraction_l477_47724


namespace positive_difference_l477_47754

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l477_47754


namespace square_side_length_l477_47771

theorem square_side_length 
  (AF DH BG AE : ℝ) 
  (AF_eq : AF = 7) 
  (DH_eq : DH = 4) 
  (BG_eq : BG = 5) 
  (AE_eq : AE = 1) 
  (area_EFGH : ℝ) 
  (area_EFGH_eq : area_EFGH = 78) : 
  (∃ s : ℝ, s^2 = 144) :=
by
  use 12
  sorry

end square_side_length_l477_47771


namespace parabola_symmetry_product_l477_47728

theorem parabola_symmetry_product (a p m : ℝ) 
  (hpr1 : a ≠ 0) 
  (hpr2 : p > 0) 
  (hpr3 : ∀ (x₀ y₀ : ℝ), y₀^2 = 2*p*x₀ → (a*(y₀ - m)^2 - 3*(y₀ - m) + 3 = x₀ + m)) :
  a * p * m = -3 := 
sorry

end parabola_symmetry_product_l477_47728


namespace fibonacci_expression_equality_l477_47780

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Statement to be proven
theorem fibonacci_expression_equality :
  (fibonacci 0 * fibonacci 2 + fibonacci 1 * fibonacci 3 + fibonacci 2 * fibonacci 4 +
  fibonacci 3 * fibonacci 5 + fibonacci 4 * fibonacci 6 + fibonacci 5 * fibonacci 7)
  - (fibonacci 1 ^ 2 + fibonacci 2 ^ 2 + fibonacci 3 ^ 2 + fibonacci 4 ^ 2 + fibonacci 5 ^ 2 + fibonacci 6 ^ 2)
  = 0 :=
by
  sorry

end fibonacci_expression_equality_l477_47780


namespace contrapositive_eq_l477_47781

variables (P Q : Prop)

theorem contrapositive_eq : (¬P → Q) ↔ (¬Q → P) := 
by {
    sorry
}

end contrapositive_eq_l477_47781


namespace max_pizzas_l477_47793

theorem max_pizzas (dough_available cheese_available sauce_available pepperoni_available mushroom_available olive_available sausage_available: ℝ)
  (dough_per_pizza cheese_per_pizza sauce_per_pizza toppings_per_pizza: ℝ)
  (total_toppings: ℝ)
  (toppings_per_pizza_sum: total_toppings = pepperoni_available + mushroom_available + olive_available + sausage_available)
  (dough_cond: dough_available = 200)
  (cheese_cond: cheese_available = 20)
  (sauce_cond: sauce_available = 20)
  (pepperoni_cond: pepperoni_available = 15)
  (mushroom_cond: mushroom_available = 5)
  (olive_cond: olive_available = 5)
  (sausage_cond: sausage_available = 10)
  (dough_per_pizza_cond: dough_per_pizza = 1)
  (cheese_per_pizza_cond: cheese_per_pizza = 1/4)
  (sauce_per_pizza_cond: sauce_per_pizza = 1/6)
  (toppings_per_pizza_cond: toppings_per_pizza = 1/3)
  : (min (dough_available / dough_per_pizza) (min (cheese_available / cheese_per_pizza) (min (sauce_available / sauce_per_pizza) (total_toppings / toppings_per_pizza))) = 80) :=
by
  sorry

end max_pizzas_l477_47793


namespace find_a_l477_47745

def A (x : ℝ) : Set ℝ := {1, 2, x^2 - 5 * x + 9}
def B (x a : ℝ) : Set ℝ := {3, x^2 + a * x + a}

theorem find_a (a x : ℝ) (hxA : A x = {1, 2, 3}) (h2B : 2 ∈ B x a) :
  a = -2/3 ∨ a = -7/4 :=
by sorry

end find_a_l477_47745


namespace minimum_value_expr_pos_reals_l477_47737

noncomputable def expr (a b : ℝ) := a^2 + b^2 + 2 * a * b + 1 / (a + b)^2

theorem minimum_value_expr_pos_reals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : 
  (expr a b) ≥ 2 :=
sorry

end minimum_value_expr_pos_reals_l477_47737


namespace number_of_workers_l477_47730

theorem number_of_workers 
  (W : ℕ) 
  (h1 : 750 * W = (5 * 900) + 700 * (W - 5)) : 
  W = 20 := 
by 
  sorry

end number_of_workers_l477_47730


namespace remainder_3n_plus_2_l477_47748

-- Define the condition
def n_condition (n : ℤ) : Prop := n % 7 = 5

-- Define the theorem to be proved
theorem remainder_3n_plus_2 (n : ℤ) (h : n_condition n) : (3 * n + 2) % 7 = 3 := 
by sorry

end remainder_3n_plus_2_l477_47748


namespace mutually_exclusive_event_is_D_l477_47719

namespace Problem

def event_A (n : ℕ) (defective : ℕ) : Prop := defective ≥ 2
def mutually_exclusive_event (n : ℕ) : Prop := (∀ (defective : ℕ), defective ≤ 1) ↔ (∀ (defective : ℕ), defective ≥ 2 → false)

theorem mutually_exclusive_event_is_D (n : ℕ) : mutually_exclusive_event n := 
by 
  sorry

end Problem

end mutually_exclusive_event_is_D_l477_47719


namespace cos_sin_gt_sin_cos_l477_47795

theorem cos_sin_gt_sin_cos (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi) : Real.cos (Real.sin x) > Real.sin (Real.cos x) :=
by
  sorry

end cos_sin_gt_sin_cos_l477_47795


namespace tan_theta_minus_pi_over_4_l477_47757

theorem tan_theta_minus_pi_over_4 
  (θ : Real) (h1 : π / 2 < θ ∧ θ < 2 * π)
  (h2 : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 := 
  sorry

end tan_theta_minus_pi_over_4_l477_47757


namespace area_of_triangle_LEF_l477_47799

noncomputable
def radius : ℝ := 10
def chord_length : ℝ := 10
def diameter_parallel_chord : Prop := True -- this condition ensures EF is parallel to LM
def LZ_length : ℝ := 20
def collinear_points : Prop := True -- this condition ensures L, M, O, Z are collinear

theorem area_of_triangle_LEF : 
  radius = 10 ∧
  chord_length = 10 ∧
  diameter_parallel_chord ∧
  LZ_length = 20 ∧ 
  collinear_points →
  (∃ area : ℝ, area = 50 * Real.sqrt 3) :=
by
  sorry

end area_of_triangle_LEF_l477_47799


namespace area_to_paint_l477_47753

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5
def window_height : ℕ := 2
def window_length : ℕ := 3

theorem area_to_paint : (wall_height * wall_length) - (door_height * door_length + window_height * window_length) = 129 := by
  sorry

end area_to_paint_l477_47753


namespace distance_A_to_B_l477_47743

theorem distance_A_to_B : 
  ∀ (D : ℕ),
    let boat_speed_with_wind := 21
    let boat_speed_against_wind := 17
    let time_for_round_trip := 7
    let stream_speed_ab := 3
    let stream_speed_ba := 2
    let effective_speed_ab := boat_speed_with_wind + stream_speed_ab
    let effective_speed_ba := boat_speed_against_wind - stream_speed_ba
    D / effective_speed_ab + D / effective_speed_ba = time_for_round_trip →
    D = 65 :=
by
  sorry

end distance_A_to_B_l477_47743


namespace value_of_a_minus_b_l477_47711

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  (a - b = -1) ∨ (a - b = -7) :=
by
  sorry

end value_of_a_minus_b_l477_47711


namespace charity_years_l477_47794

theorem charity_years :
  ∃! pairs : List (ℕ × ℕ), 
    (∀ (w m : ℕ), (w, m) ∈ pairs → 18 * w + 30 * m = 55 * 12) ∧
    pairs.length = 6 :=
by
  sorry

end charity_years_l477_47794


namespace binom_30_3_eq_4060_l477_47774

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l477_47774


namespace find_third_number_l477_47789

-- Definitions based on given conditions
def A : ℕ := 200
def C : ℕ := 100
def B : ℕ := 2 * C

-- The condition that the sum of A, B, and C is 500
def sum_condition : Prop := A + B + C = 500

-- The proof statement
theorem find_third_number : sum_condition → C = 100 := 
by
  have h1 : A = 200 := rfl
  have h2 : B = 2 * C := rfl
  have h3 : A + B + C = 500 := sorry
  sorry

end find_third_number_l477_47789


namespace smallest_next_divisor_l477_47775

def isOddFourDigitNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1000 ≤ n ∧ n < 10000

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => d > 0 ∧ n % d = 0)

theorem smallest_next_divisor (m : ℕ) (h₁ : isOddFourDigitNumber m) (h₂ : 437 ∈ divisors m) :
  ∃ k, k > 437 ∧ k ∈ divisors m ∧ k % 2 = 1 ∧ ∀ n, n > 437 ∧ n < k → n ∉ divisors m := by
  sorry

end smallest_next_divisor_l477_47775


namespace rational_solutions_quad_eq_iff_k_eq_4_l477_47713

theorem rational_solutions_quad_eq_iff_k_eq_4 (k : ℕ) (hk : 0 < k) : 
  (∃ x : ℚ, x^2 + 24/k * x + 9 = 0) ↔ k = 4 :=
sorry

end rational_solutions_quad_eq_iff_k_eq_4_l477_47713


namespace difference_of_numbers_is_21938_l477_47731

theorem difference_of_numbers_is_21938 
  (x y : ℕ) 
  (h1 : x + y = 26832) 
  (h2 : x % 10 = 0) 
  (h3 : y = x / 10 + 4) 
  : x - y = 21938 :=
sorry

end difference_of_numbers_is_21938_l477_47731


namespace squared_difference_l477_47776

theorem squared_difference (x y : ℝ) (h₁ : (x + y)^2 = 49) (h₂ : x * y = 8) : (x - y)^2 = 17 := 
by
  -- Proof omitted
  sorry

end squared_difference_l477_47776


namespace condition_implies_at_least_one_gt_one_l477_47760

theorem condition_implies_at_least_one_gt_one (a b : ℝ) :
  (a + b > 2 → (a > 1 ∨ b > 1)) ∧ ¬(a^2 + b^2 > 2 → (a > 1 ∨ b > 1)) :=
by
  sorry

end condition_implies_at_least_one_gt_one_l477_47760


namespace right_triangle_sets_l477_47714

theorem right_triangle_sets :
  ∃! (a b c : ℕ), 
    ((a = 5 ∧ b = 12 ∧ c = 13) ∧ a * a + b * b = c * c) ∧ 
    ¬(∃ a b c, (a = 3 ∧ b = 4 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 4 ∧ b = 5 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 5 ∧ b = 7 ∧ c = 9) ∧ a * a + b * b = c * c) :=
by {
  --- proof needed
  sorry
}

end right_triangle_sets_l477_47714


namespace repeating_decimal_356_fraction_l477_47769

noncomputable def repeating_decimal_356 := 3.0 + 56 / 99

theorem repeating_decimal_356_fraction : repeating_decimal_356 = 353 / 99 := by
  sorry

end repeating_decimal_356_fraction_l477_47769


namespace airplane_distance_difference_l477_47739

theorem airplane_distance_difference (a : ℕ) : 
  let against_wind_distance := (a - 20) * 3
  let with_wind_distance := (a + 20) * 4
  with_wind_distance - against_wind_distance = a + 140 :=
by
  sorry

end airplane_distance_difference_l477_47739


namespace combined_weight_l477_47706

variable (J S : ℝ)

-- Given conditions
def jake_current_weight := (J = 152)
def lose_weight_equation := (J - 32 = 2 * S)

-- Question: combined weight of Jake and his sister
theorem combined_weight (h1 : jake_current_weight J) (h2 : lose_weight_equation J S) : J + S = 212 :=
by
  sorry

end combined_weight_l477_47706


namespace avg_speed_between_B_and_C_l477_47749

noncomputable def avg_speed_from_B_to_C : ℕ := 20

theorem avg_speed_between_B_and_C
    (A_to_B_dist : ℕ := 120)
    (A_to_B_time : ℕ := 4)
    (B_to_C_dist : ℕ := 120) -- three-thirds of A_to_B_dist
    (C_to_D_dist : ℕ := 60) -- half of B_to_C_dist
    (C_to_D_time : ℕ := 2)
    (total_avg_speed : ℕ := 25)
    : avg_speed_from_B_to_C = 20 := 
  sorry

end avg_speed_between_B_and_C_l477_47749


namespace jane_age_l477_47727

theorem jane_age (j : ℕ) 
  (h₁ : ∃ (k : ℕ), j - 2 = k^2)
  (h₂ : ∃ (m : ℕ), j + 2 = m^3) :
  j = 6 :=
sorry

end jane_age_l477_47727


namespace sum_of_solutions_eq_l477_47790

theorem sum_of_solutions_eq :
  let A := 100
  let B := 3
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ = abs (B*x₁ - abs (A - B*x₁)) ∧ 
    x₂ = abs (B*x₂ - abs (A - B*x₂)) ∧ 
    x₃ = abs (B*x₃ - abs (A - B*x₃))) ∧ 
    (x₁ + x₂ + x₃ = (1900 : ℝ) / 7)) :=
by
  sorry

end sum_of_solutions_eq_l477_47790


namespace triangle_angle_sum_cannot_exist_l477_47796

theorem triangle_angle_sum (A : Real) (B : Real) (C : Real) :
    A + B + C = 180 :=
sorry

theorem cannot_exist (right_two_60 : ¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) 
    (scalene_100 : ∃ A B C : Real, A = 100 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A + B + C = 180)
    (isosceles_two_70 : ∃ A B C : Real, A = B ∧ A = 70 ∧ C = 180 - 2 * A ∧ A + B + C = 180)
    (equilateral_60 : ∃ A B C : Real, A = 60 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180)
    (one_90_two_50 : ¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :
  (¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) ∧
  (¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :=
by
  sorry

end triangle_angle_sum_cannot_exist_l477_47796


namespace total_people_in_group_l477_47766

-- Given conditions as definitions
def numChinese : Nat := 22
def numAmericans : Nat := 16
def numAustralians : Nat := 11

-- Statement of the theorem to prove
theorem total_people_in_group : (numChinese + numAmericans + numAustralians) = 49 :=
by
  -- proof goes here
  sorry

end total_people_in_group_l477_47766


namespace find_analytical_expression_function_increasing_inequality_solution_l477_47767

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Conditions
variables {a b x : ℝ}
axiom odd_function : ∀ x : ℝ, f a b (-x) = -f a b x
axiom half_value : f a b (1/2) = 2/5

-- Questions/Statements

-- 1. Analytical expression
theorem find_analytical_expression :
  ∃ a b, f a b x = x / (1 + x^2) := 
sorry

-- 2. Increasing function
theorem function_increasing :
  ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f 1 0 x1 < f 1 0 x2 := 
sorry

-- 3. Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 ((-1 + Real.sqrt 5) / 2)) → f 1 0 (x^2 - 1) + f 1 0 x < 0 := 
sorry

end find_analytical_expression_function_increasing_inequality_solution_l477_47767


namespace range_of_a_l477_47759

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + 1 + a * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end range_of_a_l477_47759


namespace tan_alpha_eq_one_third_cos2alpha_over_expr_l477_47717

theorem tan_alpha_eq_one_third_cos2alpha_over_expr (α : ℝ) (h : Real.tan α = 1/3) :
  (Real.cos (2 * α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8 / 15 :=
by
  -- This is the point where the proof steps will go, but we leave it as a placeholder.
  sorry

end tan_alpha_eq_one_third_cos2alpha_over_expr_l477_47717


namespace polygons_after_cuts_l477_47772

theorem polygons_after_cuts (initial_polygons : ℕ) (cuts : ℕ) 
  (initial_vertices : ℕ) (max_vertices_added_per_cut : ℕ) :
  (initial_polygons = 10) →
  (cuts = 51) →
  (initial_vertices = 100) →
  (max_vertices_added_per_cut = 4) →
  ∃ p, (p < 5 ∧ p ≥ 3) :=
by
  intros h_initial_polygons h_cuts h_initial_vertices h_max_vertices_added_per_cut
  -- proof steps would go here
  sorry

end polygons_after_cuts_l477_47772


namespace day50_yearM_minus1_is_Friday_l477_47700

-- Define weekdays
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Weekday

-- Define days of the week for specific days in given years
def day_of (d : Nat) (reference_day : Weekday) (reference_day_mod : Nat) : Weekday :=
  match (reference_day_mod + d - 1) % 7 with
  | 0 => Sunday
  | 1 => Monday
  | 2 => Tuesday
  | 3 => Wednesday
  | 4 => Thursday
  | 5 => Friday
  | 6 => Saturday
  | _ => Thursday -- This case should never occur due to mod 7

def day250_yearM : Weekday := Thursday
def day150_yearM1 : Weekday := Thursday

-- Theorem to prove
theorem day50_yearM_minus1_is_Friday :
    day_of 50 day250_yearM 6 = Friday :=
sorry

end day50_yearM_minus1_is_Friday_l477_47700
