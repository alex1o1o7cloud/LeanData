import Mathlib

namespace total_cost_magic_decks_l1935_193569

theorem total_cost_magic_decks (price_per_deck : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) :
  price_per_deck = 7 ∧ frank_decks = 3 ∧ friend_decks = 2 → 
  (price_per_deck * frank_decks + price_per_deck * friend_decks) = 35 :=
by
  sorry

end total_cost_magic_decks_l1935_193569


namespace determine_a_of_parallel_lines_l1935_193547

theorem determine_a_of_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x ↔ y = 3 * x + a) →
  (∀ x y : ℝ, y - 2 = (a - 3) * x ↔ y = (a - 3) * x + 2) →
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x → y - 2 = (a - 3) * x → 3 = a - 3) →
  a = 6 :=
by
  sorry

end determine_a_of_parallel_lines_l1935_193547


namespace range_of_a_l1935_193533

variable {a : ℝ}

theorem range_of_a (h : ∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) : -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l1935_193533


namespace find_value_of_p_l1935_193588

theorem find_value_of_p (p q r s t u v w : ℤ)
  (h1 : r + s = -2)
  (h2 : s + (-2) = 5)
  (h3 : t + u = 5)
  (h4 : u + v = 3)
  (h5 : v + w = 8)
  (h6 : w + t = 3)
  (h7 : q + r = s)
  (h8 : p + q = r) :
  p = -25 := by
  -- proof skipped
  sorry

end find_value_of_p_l1935_193588


namespace sequence_sum_129_l1935_193519

/-- 
  In an increasing sequence of four positive integers where the first three terms form an arithmetic
  progression and the last three terms form a geometric progression, and where the first and fourth
  terms differ by 30, the sum of the four terms is 129.
-/
theorem sequence_sum_129 :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a < a + d) ∧ (a + d < a + 2 * d) ∧ 
    (a + 2 * d < a + 30) ∧ 30 = (a + 30) - a ∧ 
    (a + d) * (a + 30) = (a + 2 * d) ^ 2 ∧ 
    a + (a + d) + (a + 2 * d) + (a + 30) = 129 :=
sorry

end sequence_sum_129_l1935_193519


namespace difference_in_elevation_difference_in_running_time_l1935_193557

structure Day :=
  (distance_km : ℝ) -- kilometers
  (pace_min_per_km : ℝ) -- minutes per kilometer
  (elevation_gain_m : ℝ) -- meters

def monday : Day := { distance_km := 9, pace_min_per_km := 6, elevation_gain_m := 300 }
def wednesday : Day := { distance_km := 4.816, pace_min_per_km := 5.5, elevation_gain_m := 150 }
def friday : Day := { distance_km := 2.095, pace_min_per_km := 7, elevation_gain_m := 50 }

noncomputable def calculate_running_time(day : Day) : ℝ :=
  day.distance_km * day.pace_min_per_km

noncomputable def total_elevation_gain(wednesday friday : Day) : ℝ :=
  wednesday.elevation_gain_m + friday.elevation_gain_m

noncomputable def total_running_time(wednesday friday : Day) : ℝ :=
  calculate_running_time wednesday + calculate_running_time friday

theorem difference_in_elevation :
  monday.elevation_gain_m - total_elevation_gain wednesday friday = 100 := by 
  sorry

theorem difference_in_running_time :
  calculate_running_time monday - total_running_time wednesday friday = 12.847 := by 
  sorry

end difference_in_elevation_difference_in_running_time_l1935_193557


namespace six_digit_squares_l1935_193583

theorem six_digit_squares :
    ∃ n m : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 100 ≤ m ∧ m ≤ 999 ∧ n = m^2 ∧ (n = 390625 ∨ n = 141376) :=
by
  sorry

end six_digit_squares_l1935_193583


namespace xena_escape_l1935_193563

theorem xena_escape
    (head_start : ℕ)
    (safety_distance : ℕ)
    (xena_speed : ℕ)
    (dragon_speed : ℕ)
    (effective_gap : ℕ := head_start - safety_distance)
    (speed_difference : ℕ := dragon_speed - xena_speed) :
    (time_to_safety : ℕ := effective_gap / speed_difference) →
    time_to_safety = 32 :=
by
  sorry

end xena_escape_l1935_193563


namespace intersection_point_unique_m_l1935_193590

theorem intersection_point_unique_m (m : ℕ) (h1 : m > 0)
  (x y : ℤ) (h2 : 13 * x + 11 * y = 700) (h3 : y = m * x - 1) : m = 6 :=
by
  sorry

end intersection_point_unique_m_l1935_193590


namespace arrangement_ways_l1935_193577

def green_marbles : Nat := 7
noncomputable def N_max_blue_marbles : Nat := 924

theorem arrangement_ways (N : Nat) (blue_marbles : Nat) (total_marbles : Nat)
  (h1 : total_marbles = green_marbles + blue_marbles) 
  (h2 : ∃ b_gap, b_gap = blue_marbles - (total_marbles - green_marbles - 1))
  (h3 : blue_marbles ≥ 6)
  : N = N_max_blue_marbles := 
sorry

end arrangement_ways_l1935_193577


namespace cost_of_building_fence_l1935_193592

-- Define the conditions
def area : ℕ := 289
def price_per_foot : ℕ := 60

-- Define the length of one side of the square (since area = side^2)
def side_length (a : ℕ) : ℕ := Nat.sqrt a

-- Define the perimeter of the square (since square has 4 equal sides)
def perimeter (s : ℕ) : ℕ := 4 * s

-- Define the cost of building the fence
def cost (p : ℕ) (ppf : ℕ) : ℕ := p * ppf

-- Prove that the cost of building the fence is Rs. 4080
theorem cost_of_building_fence : cost (perimeter (side_length area)) price_per_foot = 4080 := by
  -- Skip the proof steps
  sorry

end cost_of_building_fence_l1935_193592


namespace intersection_M_N_l1935_193512

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l1935_193512


namespace acute_triangle_sums_to_pi_over_4_l1935_193572

theorem acute_triangle_sums_to_pi_over_4 
    (A B : ℝ) 
    (hA : 0 < A ∧ A < π / 2) 
    (hB : 0 < B ∧ B < π / 2) 
    (h_sinA : Real.sin A = (Real.sqrt 5)/5) 
    (h_sinB : Real.sin B = (Real.sqrt 10)/10) : 
    A + B = π / 4 := 
sorry

end acute_triangle_sums_to_pi_over_4_l1935_193572


namespace compute_expression_l1935_193573

theorem compute_expression (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := 
  sorry

end compute_expression_l1935_193573


namespace sin_alpha_value_l1935_193585

theorem sin_alpha_value (α : ℝ) (h1 : Real.sin (α + π / 4) = 4 / 5) (h2 : α ∈ Set.Ioo (π / 4) (3 * π / 4)) :
  Real.sin α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end sin_alpha_value_l1935_193585


namespace lunch_cost_before_tip_l1935_193518

theorem lunch_cost_before_tip (tip_rate : ℝ) (total_spent : ℝ) (C : ℝ) : 
  tip_rate = 0.20 ∧ total_spent = 72.96 ∧ C + tip_rate * C = total_spent → C = 60.80 :=
by
  intro h
  sorry

end lunch_cost_before_tip_l1935_193518


namespace initial_ratio_of_stamps_l1935_193579

variable (K A : ℕ)

theorem initial_ratio_of_stamps (h1 : (K - 12) * 3 = (A + 12) * 4) (h2 : K - 12 = A + 44) : K/A = 5/3 :=
sorry

end initial_ratio_of_stamps_l1935_193579


namespace expected_adjacent_red_pairs_l1935_193535

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l1935_193535


namespace cannot_have_N_less_than_K_l1935_193507

theorem cannot_have_N_less_than_K (K N : ℕ) (hK : K > 2) (cards : Fin N → ℕ) (h_cards : ∀ i, cards i > 0) :
  ¬ (N < K) :=
sorry

end cannot_have_N_less_than_K_l1935_193507


namespace find_other_parallel_side_l1935_193506

theorem find_other_parallel_side 
  (a b d : ℝ) 
  (area : ℝ) 
  (h_area : area = 285) 
  (h_a : a = 20) 
  (h_d : d = 15)
  : (∃ x : ℝ, area = 1/2 * (a + x) * d ∧ x = 18) :=
by
  sorry

end find_other_parallel_side_l1935_193506


namespace numbers_are_perfect_squares_l1935_193591

/-- Prove that the numbers 49, 4489, 444889, ... obtained by inserting 48 into the 
middle of the previous number are perfect squares. -/
theorem numbers_are_perfect_squares :
  ∀ n : ℕ, ∃ k : ℕ, (k ^ 2) = (Int.ofNat ((20 * (10 : ℕ) ^ n + 1) / 3)) :=
by
  sorry

end numbers_are_perfect_squares_l1935_193591


namespace no_arithmetic_progression_40_terms_l1935_193537

noncomputable def is_arith_prog (f : ℕ → ℕ) (a : ℕ) (b : ℕ) : Prop :=
∀ n : ℕ, ∃ k : ℕ, f n = a + n * b

noncomputable def in_form_2m_3n (x : ℕ) : Prop :=
∃ m n : ℕ, x = 2^m + 3^n

theorem no_arithmetic_progression_40_terms :
  ¬ (∃ (a b : ℕ), ∀ n, n < 40 → in_form_2m_3n (a + n * b)) :=
sorry

end no_arithmetic_progression_40_terms_l1935_193537


namespace total_animals_l1935_193544

theorem total_animals : ∀ (D C R : ℕ), 
  C = 5 * D →
  R = D - 12 →
  R = 4 →
  (C + D + R = 100) :=
by
  intros D C R h1 h2 h3
  sorry

end total_animals_l1935_193544


namespace find_a_and_b_l1935_193530

theorem find_a_and_b (a b : ℝ) (h1 : b - 1/4 = (a + b) / 4 + b / 2) (h2 : 4 * a / 3 = (a + b) / 2)  :
  a = 3/2 ∧ b = 5/2 :=
by
  sorry

end find_a_and_b_l1935_193530


namespace ratio_of_adults_to_children_l1935_193510

-- Defining conditions as functions
def admission_fees_condition (a c : ℕ) : ℕ := 30 * a + 15 * c

-- Stating the problem
theorem ratio_of_adults_to_children (a c : ℕ) 
  (h1 : admission_fees_condition a c = 2250)
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 2 := 
sorry

end ratio_of_adults_to_children_l1935_193510


namespace starting_number_of_range_divisible_by_11_l1935_193526

theorem starting_number_of_range_divisible_by_11 (a : ℕ) : 
  a ≤ 79 ∧ (a + 22 = 77) ∧ ((a + 11) + 11 = 77) → a = 55 := 
by
  sorry

end starting_number_of_range_divisible_by_11_l1935_193526


namespace find_z_value_l1935_193534

theorem find_z_value (k : ℝ) (y z : ℝ) (h1 : (y = 2) → (z = 1)) (h2 : y ^ 3 * z ^ (1/3) = k) : 
  (y = 4) → z = 1 / 512 :=
by
  sorry

end find_z_value_l1935_193534


namespace min_quadratic_expression_value_l1935_193528

def quadratic_expression (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_quadratic_expression_value : 
  ∃ x : ℝ, quadratic_expression x = 2178 :=
sorry

end min_quadratic_expression_value_l1935_193528


namespace verify_trig_identity_l1935_193536

noncomputable def trig_identity_eqn : Prop :=
  2 * Real.sqrt (1 - Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4

theorem verify_trig_identity : trig_identity_eqn := by
  sorry

end verify_trig_identity_l1935_193536


namespace yoongi_calculation_l1935_193513

theorem yoongi_calculation (x : ℝ) (h : x - 5 = 30) : x / 7 = 5 :=
by
  sorry

end yoongi_calculation_l1935_193513


namespace probability_same_color_dice_l1935_193504

theorem probability_same_color_dice :
  let total_sides := 12
  let red_sides := 3
  let green_sides := 4
  let blue_sides := 2
  let yellow_sides := 3
  let prob_red := (red_sides / total_sides) ^ 2
  let prob_green := (green_sides / total_sides) ^ 2
  let prob_blue := (blue_sides / total_sides) ^ 2
  let prob_yellow := (yellow_sides / total_sides) ^ 2
  prob_red + prob_green + prob_blue + prob_yellow = 19 / 72 := 
by
  -- The proof goes here
  sorry

end probability_same_color_dice_l1935_193504


namespace sequence_bounded_l1935_193550

theorem sequence_bounded (a : ℕ → ℝ) :
  a 0 = 2 →
  (∀ n, a (n+1) = (2 * a n + 1) / (a n + 2)) →
  ∀ n, 1 < a n ∧ a n < 1 + 1 / 3^n :=
by
  intro h₀ h₁
  sorry

end sequence_bounded_l1935_193550


namespace distance_first_day_l1935_193570

theorem distance_first_day (total_distance : ℕ) (q : ℚ) (n : ℕ) (a : ℚ) : total_distance = 378 ∧ q = 1 / 2 ∧ n = 6 → a = 192 :=
by
  -- Proof omitted, just provide the statement
  sorry

end distance_first_day_l1935_193570


namespace ashok_average_marks_l1935_193582

variable (avg_5_subjects : ℕ) (marks_6th_subject : ℕ)
def total_marks_5_subjects := avg_5_subjects * 5
def total_marks_6_subjects := total_marks_5_subjects avg_5_subjects + marks_6th_subject
def avg_6_subjects := total_marks_6_subjects avg_5_subjects marks_6th_subject / 6

theorem ashok_average_marks (h1 : avg_5_subjects = 74) (h2 : marks_6th_subject = 50) : avg_6_subjects avg_5_subjects marks_6th_subject = 70 := by
  sorry

end ashok_average_marks_l1935_193582


namespace ratio_adult_child_l1935_193542

theorem ratio_adult_child (total_fee adults_fee children_fee adults children : ℕ) 
  (h1 : adults ≥ 1) (h2 : children ≥ 1) 
  (h3 : adults_fee = 30) (h4 : children_fee = 15) 
  (h5 : total_fee = 2250) 
  (h6 : adults_fee * adults + children_fee * children = total_fee) :
  (2 : ℚ) = adults / children :=
sorry

end ratio_adult_child_l1935_193542


namespace not_or_false_imp_and_false_l1935_193539

variable (p q : Prop)

theorem not_or_false_imp_and_false (h : ¬ (p ∨ q) = False) : ¬ (p ∧ q) :=
by
  sorry

end not_or_false_imp_and_false_l1935_193539


namespace ricardo_coins_difference_l1935_193589

theorem ricardo_coins_difference :
  ∃ (x y : ℕ), (x + y = 2020) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ ((5 * x + y) - (x + 5 * y) = 8072) :=
by
  sorry

end ricardo_coins_difference_l1935_193589


namespace policeman_hats_difference_l1935_193555

theorem policeman_hats_difference
  (hats_simpson : ℕ)
  (hats_obrien_now : ℕ)
  (hats_obrien_before : ℕ)
  (H : hats_simpson = 15)
  (H_hats_obrien_now : hats_obrien_now = 34)
  (H_hats_obrien_twice : hats_obrien_before = hats_obrien_now + 1) :
  hats_obrien_before - 2 * hats_simpson = 5 :=
by
  sorry

end policeman_hats_difference_l1935_193555


namespace boys_and_girls_at_bus_stop_l1935_193594

theorem boys_and_girls_at_bus_stop (H M : ℕ) 
  (h1 : H = 2 * (M - 15)) 
  (h2 : M - 15 = 5 * (H - 45)) : 
  H = 50 ∧ M = 40 := 
by 
  sorry

end boys_and_girls_at_bus_stop_l1935_193594


namespace sum_of_fourth_powers_is_three_times_square_l1935_193538

theorem sum_of_fourth_powers_is_three_times_square (n : ℤ) (h : n ≠ 0) :
  (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * (n^2 + 2)^2 :=
by
  sorry

end sum_of_fourth_powers_is_three_times_square_l1935_193538


namespace log_function_increasing_interval_l1935_193527

theorem log_function_increasing_interval (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x < y → y ≤ 3 → 4 - ax > 0 ∧ (4 - ax < 4 - ay)) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end log_function_increasing_interval_l1935_193527


namespace range_of_a_l1935_193514

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 1 < a ∧ a < 2 :=
by
  -- Insert the proof here
  sorry

end range_of_a_l1935_193514


namespace not_all_squares_congruent_l1935_193551

-- Define what it means to be a square
structure Square :=
  (side : ℝ)
  (angle : ℝ)
  (is_square : side > 0 ∧ angle = 90)

-- Define congruency of squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side ∧ s1.angle = s2.angle

-- The main statement to prove 
theorem not_all_squares_congruent : ∃ s1 s2 : Square, ¬ congruent s1 s2 :=
by
  sorry

end not_all_squares_congruent_l1935_193551


namespace line_k_x_intercept_l1935_193571

theorem line_k_x_intercept :
  ∀ (x y : ℝ), 3 * x - 5 * y + 40 = 0 ∧ 
  ∃ m' b', (m' = 4) ∧ (b' = 20 - 4 * 20) ∧ 
  (y = m' * x + b') →
  ∃ x_inter, (y = 0) → (x_inter = 15) := 
by
  sorry

end line_k_x_intercept_l1935_193571


namespace appropriate_import_range_l1935_193580

def mung_bean_import_range (p0 : ℝ) (p_desired_min p_desired_max : ℝ) (x : ℝ) : Prop :=
  p0 - (x / 100) ≤ p_desired_max ∧ p0 - (x / 100) ≥ p_desired_min

theorem appropriate_import_range : 
  ∃ x : ℝ, 600 ≤ x ∧ x ≤ 800 ∧ mung_bean_import_range 16 8 10 x :=
sorry

end appropriate_import_range_l1935_193580


namespace mikes_original_speed_l1935_193562

variable (x : ℕ) -- x is the original typing speed of Mike

-- Condition: After the accident, Mike's typing speed is 20 words per minute less
def currentSpeed : ℕ := x - 20

-- Condition: It takes Mike 18 minutes to type 810 words at his reduced speed
def typingTimeCondition : Prop := 18 * currentSpeed x = 810

-- Proof goal: Prove that Mike's original typing speed is 65 words per minute
theorem mikes_original_speed (h : typingTimeCondition x) : x = 65 := 
sorry

end mikes_original_speed_l1935_193562


namespace sum_of_decimals_as_fraction_l1935_193567

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l1935_193567


namespace cistern_fill_time_l1935_193575

theorem cistern_fill_time (hF : ∀ (F : ℝ), F = 1 / 3)
                         (hE : ∀ (E : ℝ), E = 1 / 5) : 
  ∃ (t : ℝ), t = 15 / 2 :=
by
  sorry

end cistern_fill_time_l1935_193575


namespace length_of_green_caterpillar_l1935_193524

def length_of_orange_caterpillar : ℝ := 1.17
def difference_in_length_between_caterpillars : ℝ := 1.83

theorem length_of_green_caterpillar :
  (length_of_orange_caterpillar + difference_in_length_between_caterpillars) = 3.00 :=
by
  sorry

end length_of_green_caterpillar_l1935_193524


namespace no_real_roots_smallest_m_l1935_193516

theorem no_real_roots_smallest_m :
  ∃ m : ℕ, m = 4 ∧
  ∀ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0 → ¬ ∃ x₀ : ℝ, 
  (3 * m - 2) * x₀^2 - 15 * x₀ + 7 = 0 ∧ 281 - 84 * m < 0 := sorry

end no_real_roots_smallest_m_l1935_193516


namespace perimeter_range_l1935_193561

variable (a b x : ℝ)
variable (a_gt_b : a > b)
variable (triangle_ineq : a - b < x ∧ x < a + b)

theorem perimeter_range : 2 * a < a + b + x ∧ a + b + x < 2 * (a + b) :=
by
  sorry

end perimeter_range_l1935_193561


namespace abc_inequality_l1935_193515

theorem abc_inequality (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (h : a * b + a * c + b * c = a + b + c) : 
  a + b + c + 1 ≥ 4 * a * b * c :=
by 
  sorry

end abc_inequality_l1935_193515


namespace find_number_that_satisfies_condition_l1935_193523

theorem find_number_that_satisfies_condition : ∃ x : ℝ, x / 3 + 12 = 20 ∧ x = 24 :=
by
  sorry

end find_number_that_satisfies_condition_l1935_193523


namespace number_of_boys_in_school_l1935_193578

-- Definition of percentages for Muslims, Hindus, and Sikhs
def percent_muslims : ℝ := 0.46
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10

-- Given number of boys in other communities
def boys_other_communities : ℝ := 136

-- The total number of boys in the school
def total_boys (B : ℝ) : Prop := B = 850

-- Proof statement (with conditions embedded)
theorem number_of_boys_in_school (B : ℝ) :
  percent_muslims * B + percent_hindus * B + percent_sikhs * B + boys_other_communities = B → 
  total_boys B :=
by
  sorry

end number_of_boys_in_school_l1935_193578


namespace diamond_sum_l1935_193559

def diamond (x : ℚ) : ℚ := (x^3 + 2 * x^2 + 3 * x) / 6

theorem diamond_sum : diamond 2 + diamond 3 + diamond 4 = 92 / 3 := by
  sorry

end diamond_sum_l1935_193559


namespace positive_solution_system_l1935_193576

theorem positive_solution_system (x1 x2 x3 x4 x5 : ℝ) (h1 : (x3 + x4 + x5)^5 = 3 * x1)
  (h2 : (x4 + x5 + x1)^5 = 3 * x2) (h3 : (x5 + x1 + x2)^5 = 3 * x3)
  (h4 : (x1 + x2 + x3)^5 = 3 * x4) (h5 : (x2 + x3 + x4)^5 = 3 * x5) :
  x1 > 0 → x2 > 0 → x3 > 0 → x4 > 0 → x5 > 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 ∧ (x1 = 1/3) :=
by 
  intros hpos1 hpos2 hpos3 hpos4 hpos5
  sorry

end positive_solution_system_l1935_193576


namespace family_spent_36_dollars_l1935_193553

def ticket_cost : ℝ := 5

def popcorn_cost : ℝ := 0.8 * ticket_cost

def soda_cost : ℝ := 0.5 * popcorn_cost

def tickets_bought : ℕ := 4

def popcorn_bought : ℕ := 2

def sodas_bought : ℕ := 4

def total_spent : ℝ :=
  (tickets_bought * ticket_cost) +
  (popcorn_bought * popcorn_cost) +
  (sodas_bought * soda_cost)

theorem family_spent_36_dollars : total_spent = 36 := by
  sorry

end family_spent_36_dollars_l1935_193553


namespace skittles_per_friend_l1935_193502

theorem skittles_per_friend (ts : ℕ) (nf : ℕ) (h1 : ts = 200) (h2 : nf = 5) : (ts / nf = 40) :=
by sorry

end skittles_per_friend_l1935_193502


namespace problem_statements_l1935_193541

noncomputable def f (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2

theorem problem_statements (x : ℝ) :
  (f x < g x) ∧
  ((f x)^2 + (g x)^2 ≥ 1) ∧
  (f (2 * x) = 2 * f x * g x) :=
by
  sorry

end problem_statements_l1935_193541


namespace solution_l1935_193586

theorem solution {a : ℕ → ℝ} 
  (h : a 1 = 1)
  (h2 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    a n - 4 * a (if n = 100 then 1 else n + 1) + 3 * a (if n = 99 then 1 else if n = 100 then 2 else n + 2) ≥ 0) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → a n = 1 :=
by
  sorry

end solution_l1935_193586


namespace reflection_across_x_axis_l1935_193597

theorem reflection_across_x_axis :
  let initial_point := (-3, 5)
  let reflected_point := (-3, -5)
  reflected_point = (initial_point.1, -initial_point.2) :=
by
  sorry

end reflection_across_x_axis_l1935_193597


namespace votes_for_sue_l1935_193517

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end votes_for_sue_l1935_193517


namespace numerator_denominator_added_l1935_193545

theorem numerator_denominator_added (n : ℕ) : (3 + n) / (5 + n) = 9 / 11 → n = 6 :=
by
  sorry

end numerator_denominator_added_l1935_193545


namespace compute_f_seven_halves_l1935_193508

theorem compute_f_seven_halves 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_shift : ∀ x, f (x + 2) = -f x)
  (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (7 / 2) = -1 / 2 :=
  sorry

end compute_f_seven_halves_l1935_193508


namespace neg_exponent_reciprocal_l1935_193554

theorem neg_exponent_reciprocal : (2 : ℝ) ^ (-1 : ℤ) = 1 / 2 := by
  -- Insert your proof here
  sorry

end neg_exponent_reciprocal_l1935_193554


namespace find_k_l1935_193501

-- Define a point and its translation
structure Point where
  x : ℕ
  y : ℕ

-- Original and translated points
def P : Point := { x := 5, y := 3 }
def P' : Point := { x := P.x - 4, y := P.y - 1 }

-- Given function with parameter k
def line (k : ℕ) (p : Point) : ℕ := (k * p.x) - 2

-- Prove the value of k
theorem find_k (k : ℕ) (h : line k P' = P'.y) : k = 4 :=
by
  sorry

end find_k_l1935_193501


namespace find_x_when_z_64_l1935_193565

-- Defining the conditions
def directly_proportional (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y^3
def inversely_proportional (y z : ℝ) : Prop := ∃ n : ℝ, y = n / z^2

theorem find_x_when_z_64 (x y z : ℝ) (m n : ℝ) (k : ℝ) (h1 : directly_proportional x y) 
    (h2 : inversely_proportional y z) (h3 : z = 64) (h4 : x = 8) (h5 : z = 16) : x = 1/256 := 
  sorry

end find_x_when_z_64_l1935_193565


namespace integer_solutions_l1935_193540

theorem integer_solutions (x y z : ℤ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x + y + z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1 / (x + y + z)) ↔ (z = -x - y) :=
sorry

end integer_solutions_l1935_193540


namespace probability_of_C_and_D_are_equal_l1935_193509

theorem probability_of_C_and_D_are_equal (h1 : Prob_A = 1/4) (h2 : Prob_B = 1/3) (h3 : total_prob = 1) (h4 : Prob_C = Prob_D) : 
  Prob_C = 5/24 ∧ Prob_D = 5/24 := by
  sorry

end probability_of_C_and_D_are_equal_l1935_193509


namespace more_sqft_to_mow_l1935_193543

-- Defining the parameters given in the original problem
def rate_per_sqft : ℝ := 0.10
def book_cost : ℝ := 150.0
def lawn_dimensions : ℝ × ℝ := (20, 15)
def num_lawns_mowed : ℕ := 3

-- The theorem stating how many more square feet LaKeisha needs to mow
theorem more_sqft_to_mow : 
  let area_one_lawn := (lawn_dimensions.1 * lawn_dimensions.2 : ℝ)
  let total_area_mowed := area_one_lawn * (num_lawns_mowed : ℝ)
  let money_earned := total_area_mowed * rate_per_sqft
  let remaining_amount := book_cost - money_earned
  let more_sqft_needed := remaining_amount / rate_per_sqft
  more_sqft_needed = 600 := 
by 
  sorry

end more_sqft_to_mow_l1935_193543


namespace sum_of_coefficients_l1935_193566

-- Definition of the polynomial
def P (x : ℝ) : ℝ := 5 * (2 * x ^ 9 - 3 * x ^ 6 + 4) - 4 * (x ^ 6 - 5 * x ^ 3 + 6)

-- Theorem stating the sum of the coefficients is 7
theorem sum_of_coefficients : P 1 = 7 := by
  sorry

end sum_of_coefficients_l1935_193566


namespace perfect_square_after_dividing_l1935_193560

theorem perfect_square_after_dividing (n : ℕ) (h : n = 16800) : ∃ m : ℕ, (n / 21) = m * m :=
by {
  sorry
}

end perfect_square_after_dividing_l1935_193560


namespace factory_workers_count_l1935_193549

theorem factory_workers_count :
  ∃ (F S_f : ℝ), 
    (F * S_f = 30000) ∧ 
    (30 * (S_f + 500) = 75000) → 
    (F = 15) :=
by
  sorry

end factory_workers_count_l1935_193549


namespace interest_years_proof_l1935_193556

theorem interest_years_proof :
  let interest_r800_first_2_years := 800 * 0.05 * 2
  let interest_r800_next_3_years := 800 * 0.12 * 3
  let total_interest_r800 := interest_r800_first_2_years + interest_r800_next_3_years
  let interest_r600_first_3_years := 600 * 0.07 * 3
  let interest_r600_next_n_years := 600 * 0.10 * n
  (interest_r600_first_3_years + interest_r600_next_n_years = total_interest_r800) ->
  n = 5 →
  3 + n = 8 :=
by
  sorry

end interest_years_proof_l1935_193556


namespace line_through_point_with_equal_intercepts_l1935_193532

theorem line_through_point_with_equal_intercepts 
  (x y k : ℝ) 
  (h1 : (3 : ℝ) + (-6 : ℝ) + k = 0 ∨ 2 * (3 : ℝ) + (-6 : ℝ) = 0) 
  (h2 : k = 0 ∨ x + y + k = 0) : 
  (x = 1 ∨ x = 2) ∧ (k = -3 ∨ k = 0) :=
sorry

end line_through_point_with_equal_intercepts_l1935_193532


namespace total_amount_divided_l1935_193525

theorem total_amount_divided (B_amount A_amount C_amount: ℝ) (h1 : A_amount = (1/3) * B_amount)
    (h2 : B_amount = 270) (h3 : B_amount = (1/4) * C_amount) :
    A_amount + B_amount + C_amount = 1440 :=
by
  sorry

end total_amount_divided_l1935_193525


namespace product_of_primes_sum_ten_l1935_193505

theorem product_of_primes_sum_ten :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ Prime p1 ∧ Prime p2 ∧ p1 + p2 = 10 ∧ p1 * p2 = 21 := 
by
  sorry

end product_of_primes_sum_ten_l1935_193505


namespace impossible_arrangement_l1935_193568

theorem impossible_arrangement (s : Finset ℕ) (h₁ : s = Finset.range 2018 \ {0})
  (h₂ : ∀ a ∈ s, ∀ b ∈ s, a ≠ b ∧ (b = a + 17 ∨ b = a + 21 ∨ b = a - 17 ∨ b = a - 21)) : False :=
by
  sorry

end impossible_arrangement_l1935_193568


namespace isosceles_triangle_perimeter_l1935_193596

-- Definitions for the conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Statement of the theorem
theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : valid_triangle a b c) :
  (a = 2 ∧ b = 4 ∧ c = 4 ∨ a = 4 ∧ b = 4 ∧ c = 2 ∨ a = 4 ∧ b = 2 ∧ c = 4) →
  a + b + c = 10 :=
by 
  sorry

end isosceles_triangle_perimeter_l1935_193596


namespace average_annual_growth_rate_l1935_193548

theorem average_annual_growth_rate (x : ℝ) (h : (1 + x)^2 = 1.20) : x < 0.1 :=
sorry

end average_annual_growth_rate_l1935_193548


namespace tan_sin_cos_log_expression_simplification_l1935_193599

-- Proof Problem 1 Statement in Lean 4
theorem tan_sin_cos (α : ℝ) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by
  sorry

-- Proof Problem 2 Statement in Lean 4
theorem log_expression_simplification : 
  Real.logb 3 (Real.sqrt 27) + Real.logb 10 25 + Real.logb 10 4 + 
  (7 : ℝ) ^ Real.logb 7 2 + (-9.8) ^ 0 = 13 / 2 :=
by
  sorry

end tan_sin_cos_log_expression_simplification_l1935_193599


namespace new_mean_of_five_numbers_l1935_193598

theorem new_mean_of_five_numbers (a b c d e : ℝ) 
  (h_mean : (a + b + c + d + e) / 5 = 25) :
  ((a + 5) + (b + 10) + (c + 15) + (d + 20) + (e + 25)) / 5 = 40 :=
by
  sorry

end new_mean_of_five_numbers_l1935_193598


namespace butterflies_left_correct_l1935_193500

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l1935_193500


namespace maximum_area_of_equilateral_triangle_in_rectangle_l1935_193584

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  (953 * Real.sqrt 3) / 16

theorem maximum_area_of_equilateral_triangle_in_rectangle :
  ∀ (a b : ℕ), a = 13 → b = 14 → maxEquilateralTriangleArea a b = (953 * Real.sqrt 3) / 16 :=
by
  intros a b h₁ h₂
  rw [h₁, h₂]
  apply rfl

end maximum_area_of_equilateral_triangle_in_rectangle_l1935_193584


namespace quadrilateral_segments_condition_l1935_193574

-- Define the lengths and their conditions
variables {a b c d : ℝ}

-- Define the main theorem with necessary and sufficient conditions
theorem quadrilateral_segments_condition (h_sum : a + b + c + d = 1.5)
    (h_order : a ≤ b) (h_order2 : b ≤ c) (h_order3 : c ≤ d) (h_ratio : d ≤ 3 * a) :
    (a ≥ 0.25 ∧ d < 0.75) ↔ (a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  sorry -- proof is omitted
}

end quadrilateral_segments_condition_l1935_193574


namespace installment_payment_l1935_193587

theorem installment_payment
  (cash_price : ℕ)
  (down_payment : ℕ)
  (first_four_months_payment : ℕ)
  (last_four_months_payment : ℕ)
  (installment_additional_cost : ℕ)
  (total_next_four_months_payment : ℕ)
  (H_cash_price : cash_price = 450)
  (H_down_payment : down_payment = 100)
  (H_first_four_months_payment : first_four_months_payment = 4 * 40)
  (H_last_four_months_payment : last_four_months_payment = 4 * 30)
  (H_installment_additional_cost : installment_additional_cost = 70)
  (H_total_next_four_months_payment_correct : 4 * total_next_four_months_payment = 4 * 35) :
  down_payment + first_four_months_payment + 4 * 35 + last_four_months_payment = cash_price + installment_additional_cost := 
by {
  sorry
}

end installment_payment_l1935_193587


namespace compare_abc_l1935_193546

noncomputable def a : ℝ :=
  (1/2) * Real.cos 16 - (Real.sqrt 3 / 2) * Real.sin 16

noncomputable def b : ℝ :=
  2 * Real.tan 14 / (1 + (Real.tan 14) ^ 2)

noncomputable def c : ℝ :=
  Real.sqrt ((1 - Real.cos 50) / 2)

theorem compare_abc : b > c ∧ c > a :=
  by sorry

end compare_abc_l1935_193546


namespace count_integer_solutions_l1935_193511

theorem count_integer_solutions : 
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ s) ↔ (x^3 + y^2 = 2*y + 1)) ∧ 
  s.card = 3 := 
by
  sorry

end count_integer_solutions_l1935_193511


namespace arithmetic_sequence_geometric_condition_l1935_193520

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + 3) 
  (h_geom : (a 1 + 6) ^ 2 = a 1 * (a 1 + 9)) : 
  a 2 = -9 :=
sorry

end arithmetic_sequence_geometric_condition_l1935_193520


namespace nancy_antacids_l1935_193521

theorem nancy_antacids :
  ∀ (x : ℕ),
  (3 * 3 + x * 2 + 1 * 2) * 4 = 60 → x = 2 :=
by
  sorry

end nancy_antacids_l1935_193521


namespace g_of_neg_two_l1935_193593

def f (x : ℝ) : ℝ := 4 * x - 9

def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_of_neg_two : g (-2) = 227 / 16 :=
by
  sorry

end g_of_neg_two_l1935_193593


namespace max_blocks_fit_l1935_193581

-- Define the dimensions of the block
def block_length := 2
def block_width := 3
def block_height := 1

-- Define the dimensions of the container box
def box_length := 4
def box_width := 3
def box_height := 3

-- Define the volume calculations
def volume (length width height : ℕ) : ℕ := length * width * height

def block_volume := volume block_length block_width block_height
def box_volume := volume box_length box_width box_height

-- The theorem to prove
theorem max_blocks_fit : (box_volume / block_volume) = 6 :=
by
  sorry

end max_blocks_fit_l1935_193581


namespace neg_two_is_negative_rational_l1935_193552

theorem neg_two_is_negative_rational : 
  (-2 : ℚ) < 0 ∧ ∃ (r : ℚ), r = -2 := 
by
  sorry

end neg_two_is_negative_rational_l1935_193552


namespace apples_used_l1935_193564

theorem apples_used (apples_before : ℕ) (apples_left : ℕ) (apples_used_for_pie : ℕ) 
                    (h1 : apples_before = 19) 
                    (h2 : apples_left = 4) 
                    (h3 : apples_used_for_pie = apples_before - apples_left) : 
  apples_used_for_pie = 15 :=
by
  -- Since we are instructed to leave the proof out, we put sorry here
  sorry

end apples_used_l1935_193564


namespace shaded_area_of_rotated_square_is_four_thirds_l1935_193522

noncomputable def common_shaded_area_of_rotated_square (β : ℝ) (h1 : 0 < β) (h2 : β < π / 2) (h_cos_beta : Real.cos β = 3 / 5) : ℝ :=
  let side_length := 2
  let area := side_length * side_length / 3 * 2
  area

theorem shaded_area_of_rotated_square_is_four_thirds
  (β : ℝ)
  (h1 : 0 < β)
  (h2 : β < π / 2)
  (h_cos_beta : Real.cos β = 3 / 5) :
  common_shaded_area_of_rotated_square β h1 h2 h_cos_beta = 4 / 3 :=
sorry

end shaded_area_of_rotated_square_is_four_thirds_l1935_193522


namespace find_a_l1935_193531

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : ∀ (x : ℝ), |x - a| < 1 → x ∈ {x | x = 2}) : a = 2 :=
sorry

end find_a_l1935_193531


namespace find_4a_plus_8b_l1935_193558

def quadratic_equation_x_solution (a b : ℝ) : Prop :=
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0

theorem find_4a_plus_8b (a b : ℝ) (h : quadratic_equation_x_solution a b) : 4 * a + 8 * b = -4 := 
  by
    sorry

end find_4a_plus_8b_l1935_193558


namespace simplest_radical_expression_l1935_193529

theorem simplest_radical_expression :
  let A := Real.sqrt 3
  let B := Real.sqrt 4
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 2)
  B = 2 :=
by
  sorry

end simplest_radical_expression_l1935_193529


namespace necessary_but_not_sufficient_l1935_193595

variable (a b : ℝ)

def proposition_A : Prop := a > 0
def proposition_B : Prop := a > b ∧ a⁻¹ > b⁻¹

theorem necessary_but_not_sufficient : (proposition_B a b → proposition_A a) ∧ ¬(proposition_A a → proposition_B a b) :=
by
  sorry

end necessary_but_not_sufficient_l1935_193595


namespace number_of_pairs_of_positive_integers_l1935_193503

theorem number_of_pairs_of_positive_integers 
    {m n : ℕ} (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m > n) (h_diff : m^2 - n^2 = 144) : 
    ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧ (∀ p ∈ pairs, p.1 > p.2 ∧ p.1^2 - p.2^2 = 144) :=
sorry

end number_of_pairs_of_positive_integers_l1935_193503
