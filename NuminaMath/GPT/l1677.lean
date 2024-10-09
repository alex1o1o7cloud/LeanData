import Mathlib

namespace sunzi_classic_l1677_167717

noncomputable def length_of_rope : ℝ := sorry
noncomputable def length_of_wood : ℝ := sorry
axiom first_condition : length_of_rope - length_of_wood = 4.5
axiom second_condition : length_of_wood - (1 / 2) * length_of_rope = 1

theorem sunzi_classic : 
  (length_of_rope - length_of_wood = 4.5) ∧ (length_of_wood - (1 / 2) * length_of_rope = 1) := 
by 
  exact ⟨first_condition, second_condition⟩

end sunzi_classic_l1677_167717


namespace numberOfBooks_correct_l1677_167738

variable (totalWeight : ℕ) (weightPerBook : ℕ)

def numberOfBooks (totalWeight weightPerBook : ℕ) : ℕ :=
  totalWeight / weightPerBook

theorem numberOfBooks_correct (h1 : totalWeight = 42) (h2 : weightPerBook = 3) :
  numberOfBooks totalWeight weightPerBook = 14 := by
  sorry

end numberOfBooks_correct_l1677_167738


namespace Kyle_is_25_l1677_167751

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l1677_167751


namespace no_geometric_progression_11_12_13_l1677_167797

theorem no_geometric_progression_11_12_13 :
  ∀ (b1 : ℝ) (q : ℝ) (k l n : ℕ), 
  (b1 * q ^ (k - 1) = 11) → 
  (b1 * q ^ (l - 1) = 12) → 
  (b1 * q ^ (n - 1) = 13) → 
  False :=
by
  intros b1 q k l n hk hl hn
  sorry

end no_geometric_progression_11_12_13_l1677_167797


namespace figure_at_1000th_position_position_of_1000th_diamond_l1677_167750

-- Define the repeating sequence
def repeating_sequence : List String := ["△", "Λ", "◇", "Λ", "⊙", "□"]

-- Lean 4 statement for (a)
theorem figure_at_1000th_position :
  repeating_sequence[(1000 % repeating_sequence.length) - 1] = "Λ" :=
by sorry

-- Define the arithmetic sequence for diamond positions
def diamond_position (n : Nat) : Nat :=
  3 + (n - 1) * 6

-- Lean 4 statement for (b)
theorem position_of_1000th_diamond :
  diamond_position 1000 = 5997 :=
by sorry

end figure_at_1000th_position_position_of_1000th_diamond_l1677_167750


namespace length_A_l1677_167728

open Real

theorem length_A'B'_correct {A B C A' B' : ℝ × ℝ} :
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (C.2 - A.2) / (C.1 - A.1) = ((B.2 - C.2) / (B.1 - C.1)) →
  (dist A' B') = 2.5 * sqrt 2 :=
by
  intros
  sorry

end length_A_l1677_167728


namespace problem1_problem2_l1677_167732

-- Problem 1
theorem problem1 (a b : ℝ) : (a + 2 * b)^2 - a * (a + 4 * b) = 4 * b^2 :=
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) : 
  (2 / (m - 1) + 1) / (2 * (m + 1) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end problem1_problem2_l1677_167732


namespace number_of_pupils_l1677_167755

theorem number_of_pupils (n : ℕ) (h1 : 79 - 45 = 34)
  (h2 : 34 = 1 / 2 * n) : n = 68 :=
by
  sorry

end number_of_pupils_l1677_167755


namespace abs_eq_five_l1677_167792

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  intro h
  sorry

end abs_eq_five_l1677_167792


namespace find_x_in_sequence_l1677_167765

theorem find_x_in_sequence :
  ∃ x : ℕ, x = 32 ∧
    2 + 3 = 5 ∧
    5 + 6 = 11 ∧
    11 + 9 = 20 ∧
    20 + (9 + 3) = x ∧
    x + (9 + 3 + 3) = 47 :=
by
  sorry

end find_x_in_sequence_l1677_167765


namespace max_possible_n_l1677_167789

theorem max_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 6400) : n ≤ 7 :=
by {
  sorry
}

end max_possible_n_l1677_167789


namespace quadratic_equation_iff_non_zero_coefficient_l1677_167753

theorem quadratic_equation_iff_non_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by
  sorry

end quadratic_equation_iff_non_zero_coefficient_l1677_167753


namespace find_a_plus_b_l1677_167702

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + ab = 0 → (x = -1 ∨ x = 4)) → a + b = -3 :=
by
  sorry

end find_a_plus_b_l1677_167702


namespace complex_eq_solution_l1677_167770

theorem complex_eq_solution (x y : ℝ) (i : ℂ) (h : (2 * x - 1) + i = y - (3 - y) * i) : 
  x = 5 / 2 ∧ y = 4 :=
  sorry

end complex_eq_solution_l1677_167770


namespace total_football_games_l1677_167782

theorem total_football_games (months : ℕ) (games_per_month : ℕ) (season_length : months = 17 ∧ games_per_month = 19) :
  (months * games_per_month) = 323 :=
by
  sorry

end total_football_games_l1677_167782


namespace josiah_hans_age_ratio_l1677_167735

theorem josiah_hans_age_ratio (H : ℕ) (J : ℕ) (hH : H = 15) (hSum : (J + 3) + (H + 3) = 66) : J / H = 3 :=
by
  sorry

end josiah_hans_age_ratio_l1677_167735


namespace degree_greater_than_2_l1677_167780

variable (P Q : ℤ[X]) -- P and Q are polynomials with integer coefficients

theorem degree_greater_than_2 (P_nonconstant : ¬(P.degree = 0))
  (Q_nonconstant : ¬(Q.degree = 0))
  (h : ∃ S : Finset ℤ, S.card ≥ 25 ∧ ∀ x ∈ S, (P.eval x) * (Q.eval x) = 2009) :
  P.degree > 2 ∧ Q.degree > 2 :=
by
  sorry

end degree_greater_than_2_l1677_167780


namespace log_6_15_expression_l1677_167734

theorem log_6_15_expression (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 15 / Real.log 6 = (b + 1 - a) / (a + b) :=
sorry

end log_6_15_expression_l1677_167734


namespace burger_share_l1677_167715

theorem burger_share (burger_length : ℝ) (brother_share : ℝ) (first_friend_share : ℝ) (second_friend_share : ℝ) (valentina_share : ℝ) :
  burger_length = 12 →
  brother_share = burger_length / 3 →
  first_friend_share = (burger_length - brother_share) / 4 →
  second_friend_share = (burger_length - brother_share - first_friend_share) / 2 →
  valentina_share = burger_length - (brother_share + first_friend_share + second_friend_share) →
  brother_share = 4 ∧ first_friend_share = 2 ∧ second_friend_share = 3 ∧ valentina_share = 3 :=
by
  intros
  sorry

end burger_share_l1677_167715


namespace find_m_l1677_167726

open Nat

def is_arithmetic (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i < n - 1, a (i + 2) - a (i + 1) = a (i + 1) - a i
def is_geometric (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i ≥ n, a (i + 1) * a n = a i * a (n + 1)
def sum_prod_condition (a : ℕ → ℤ) (m : ℕ) : Prop := a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

theorem find_m (a : ℕ → ℤ)
  (h1 : a 3 = -1)
  (h2 : a 7 = 4)
  (h3 : is_arithmetic a 6)
  (h4 : is_geometric a 5) :
  ∃ m : ℕ, m = 1 ∨ m = 3 ∧ sum_prod_condition a m := sorry

end find_m_l1677_167726


namespace natural_numbers_solution_l1677_167749

theorem natural_numbers_solution :
  ∃ (a b c d : ℕ), 
    ab = c + d ∧ a + b = cd ∧
    ((a, b, c, d) = (2, 2, 2, 2) ∨ (a, b, c, d) = (2, 3, 5, 1) ∨ 
     (a, b, c, d) = (3, 2, 5, 1) ∨ (a, b, c, d) = (2, 2, 1, 5) ∨ 
     (a, b, c, d) = (3, 2, 1, 5) ∨ (a, b, c, d) = (2, 3, 1, 5)) :=
by
  sorry

end natural_numbers_solution_l1677_167749


namespace three_city_population_l1677_167773

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l1677_167773


namespace value_of_c_l1677_167771

theorem value_of_c (b c : ℝ) (h1 : (x : ℝ) → (x + 4) * (x + b) = x^2 + c * x + 12) : c = 7 :=
by
  have h2 : 4 * b = 12 := by sorry
  have h3 : b = 3 := by sorry
  have h4 : c = b + 4 := by sorry
  rw [h3] at h4
  rw [h4]
  exact by norm_num

end value_of_c_l1677_167771


namespace constants_solution_l1677_167766

theorem constants_solution (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → 
    (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2)) ↔ 
    (A = 20 ∧ B = -15 ∧ C = -10) :=
by
  sorry

end constants_solution_l1677_167766


namespace circle_area_approx_error_exceeds_one_l1677_167724

theorem circle_area_approx_error_exceeds_one (r : ℝ) : 
  (3.14159 < Real.pi ∧ Real.pi < 3.14160) → 
  2 * r > 25 →  
  |(r * r * Real.pi - r * r * 3.14)| > 1 → 
  2 * r = 51 := 
by 
  sorry

end circle_area_approx_error_exceeds_one_l1677_167724


namespace time_to_cross_bridge_l1677_167712

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (time_min : ℝ) :
  speed_km_hr = 5 → length_m = 1250 → time_min = length_m / (speed_km_hr * 1000 / 60) → time_min = 15 :=
by
  intros h_speed h_length h_time
  rw [h_speed, h_length] at h_time
  -- Since 5 km/hr * 1000 / 60 = 83.33 m/min,
  -- substituting into equation gives us 1250 / 83.33 ≈ 15.
  sorry

end time_to_cross_bridge_l1677_167712


namespace boiling_point_fahrenheit_l1677_167709

-- Define the conditions as hypotheses
def boils_celsius : ℝ := 100
def melts_celsius : ℝ := 0
def melts_fahrenheit : ℝ := 32
def pot_temp_celsius : ℝ := 55
def pot_temp_fahrenheit : ℝ := 131

-- Theorem to prove the boiling point in Fahrenheit
theorem boiling_point_fahrenheit : ∀ (boils_celsius : ℝ) (melts_celsius : ℝ) (melts_fahrenheit : ℝ) 
                                    (pot_temp_celsius : ℝ) (pot_temp_fahrenheit : ℝ),
  boils_celsius = 100 →
  melts_celsius = 0 →
  melts_fahrenheit = 32 →
  pot_temp_celsius = 55 →
  pot_temp_fahrenheit = 131 →
  ∃ boils_fahrenheit : ℝ, boils_fahrenheit = 212 :=
by
  intros
  existsi 212
  sorry

end boiling_point_fahrenheit_l1677_167709


namespace intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l1677_167740

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

-- Define the conditions
variables (m : ℝ)
theorem intersection_points_of_quadratic :
    (quadratic m 1 = 0) ∧ (quadratic m 3 = 0) ↔ m ≠ 0 :=
sorry

theorem minimum_value_of_quadratic_in_range :
    ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → quadratic (-2) x ≥ -6 :=
sorry

theorem range_of_m_for_intersection_with_segment_PQ :
    ∀ (m : ℝ), (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic m x = (m + 4) / 2) ↔ 
    m ≤ -4 / 3 ∨ m ≥ 4 / 5 :=
sorry

end intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l1677_167740


namespace arithmetic_sequence_sum_equality_l1677_167741

variables {a_n : ℕ → ℝ} -- the arithmetic sequence
variables (S_n : ℕ → ℝ) -- the sum of the first n terms of the sequence

-- Define the conditions as hypotheses
def condition_1 (S_n : ℕ → ℝ) : Prop := S_n 3 = 3
def condition_2 (S_n : ℕ → ℝ) : Prop := S_n 6 = 15

-- Theorem statement
theorem arithmetic_sequence_sum_equality
  (h1 : condition_1 S_n)
  (h2 : condition_2 S_n)
  (a_n_formula : ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0))
  (S_n_formula : ∀ n, S_n n = n * (a_n 0 + (n - 1) * (a_n 1 - a_n 0) / 2)) :
  a_n 10 + a_n 11 + a_n 12 = 30 := sorry

end arithmetic_sequence_sum_equality_l1677_167741


namespace composite_sum_l1677_167763

theorem composite_sum (x y n : ℕ) (hx : x > 1) (hy : y > 1) (h : x^2 + x * y - y = n^2) :
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = x + y + 1 :=
sorry

end composite_sum_l1677_167763


namespace mona_drives_125_miles_l1677_167733

/-- Mona can drive 125 miles with $25 worth of gas, given the car mileage
    and the cost per gallon of gas. -/
theorem mona_drives_125_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (total_money : ℕ)
  (h_miles_per_gallon : miles_per_gallon = 25) (h_cost_per_gallon : cost_per_gallon = 5)
  (h_total_money : total_money = 25) :
  (total_money / cost_per_gallon) * miles_per_gallon = 125 :=
by
  sorry

end mona_drives_125_miles_l1677_167733


namespace kenneth_initial_money_l1677_167710

-- Define the costs of the items
def cost_baguette := 2
def cost_water := 1

-- Define the quantities bought
def baguettes_bought := 2
def water_bought := 2

-- Define the amount left after buying the items
def money_left := 44

-- Calculate the total cost
def total_cost := (baguettes_bought * cost_baguette) + (water_bought * cost_water)

-- Define the initial money Kenneth had
def initial_money := total_cost + money_left

-- Prove the initial money is $50
theorem kenneth_initial_money : initial_money = 50 := 
by 
  -- The proof part is omitted because it is not required.
  sorry

end kenneth_initial_money_l1677_167710


namespace molecular_weight_C4H10_l1677_167777

theorem molecular_weight_C4H10
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (C4H10_C_atoms : ℕ)
  (C4H10_H_atoms : ℕ)
  (moles : ℝ) : 
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  C4H10_C_atoms = 4 →
  C4H10_H_atoms = 10 →
  moles = 6 →
  (C4H10_C_atoms * atomic_weight_C + C4H10_H_atoms * atomic_weight_H) * moles = 348.72 :=
by
  sorry

end molecular_weight_C4H10_l1677_167777


namespace intersection_of_A_and_B_l1677_167707

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

-- State the theorem about the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
  sorry

end intersection_of_A_and_B_l1677_167707


namespace find_a_max_min_f_l1677_167703

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem find_a (a : ℝ) (h : (deriv (f a) 0 = 1)) : a = 1 :=
by sorry

noncomputable def f_one (x : ℝ) : ℝ := f 1 x

theorem max_min_f (h : ∀ x, 0 ≤ x → x ≤ 2 → deriv (f_one) x > 0) :
  (f_one 0 = 0) ∧ (f_one 2 = 2 * Real.exp 2) :=
by sorry

end find_a_max_min_f_l1677_167703


namespace length_of_platform_l1677_167775

theorem length_of_platform 
  (speed_train_kmph : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_man : ℝ)
  (conversion_factor : ℝ)
  (speed_train_mps : ℝ)
  (length_train : ℝ)
  (total_distance : ℝ)
  (length_platform : ℝ) :
  speed_train_kmph = 150 →
  time_cross_platform = 45 →
  time_cross_man = 20 →
  conversion_factor = (1000 / 3600) →
  speed_train_mps = speed_train_kmph * conversion_factor →
  length_train = speed_train_mps * time_cross_man →
  total_distance = speed_train_mps * time_cross_platform →
  length_platform = total_distance - length_train →
  length_platform = 1041.75 :=
by sorry

end length_of_platform_l1677_167775


namespace intersection_counts_l1677_167799

theorem intersection_counts (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = -x^2 + 4 * x - 3)
  (hg : ∀ x, g x = -f x)
  (hh : ∀ x, h x = f (-x))
  (c : ℕ) (hc : c = 2)
  (d : ℕ) (hd : d = 1):
  10 * c + d = 21 :=
by
  sorry

end intersection_counts_l1677_167799


namespace find_value_of_fraction_l1677_167747

variable {x y : ℝ}

theorem find_value_of_fraction (h1 : x > 0) (h2 : y > x) (h3 : y > 0) (h4 : x / y + y / x = 3) : 
  (x + y) / (y - x) = Real.sqrt 5 := 
by sorry

end find_value_of_fraction_l1677_167747


namespace average_of_two_intermediate_numbers_l1677_167744

theorem average_of_two_intermediate_numbers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
(h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_average : (a + b + c + d) / 4 = 5)
(h_max_diff: (max (max a b) (max c d) - min (min a b) (min c d) = 19)) :
  (a + b + c + d) - (max (max a b) (max c d)) - (min (min a b) (min c d)) = 5 :=
by
  -- The proof goes here
  sorry

end average_of_two_intermediate_numbers_l1677_167744


namespace jeffreys_total_steps_l1677_167711

-- Define the conditions
def effective_steps_per_pattern : ℕ := 1
def total_effective_distance : ℕ := 66
def steps_per_pattern : ℕ := 5

-- Define the proof problem
theorem jeffreys_total_steps : ∀ (N : ℕ), 
  N = (total_effective_distance * steps_per_pattern) := 
sorry

end jeffreys_total_steps_l1677_167711


namespace distinct_four_digit_integers_l1677_167795

open Nat

theorem distinct_four_digit_integers (count_digs_18 : ℕ) :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (∃ d1 d2 d3 d4 : ℕ,
      d1 * d2 * d3 * d4 = 18 ∧
      d1 > 0 ∧ d1 < 10 ∧
      d2 > 0 ∧ d2 < 10 ∧
      d3 > 0 ∧ d3 < 10 ∧
      d4 > 0 ∧ d4 < 10 ∧
      n = d1 * 1000 + d2 * 100 + d3 * 10 + d4)) →
  count_digs_18 = 24 :=
sorry

end distinct_four_digit_integers_l1677_167795


namespace percentage_of_25_of_fifty_percent_of_500_l1677_167721

-- Define the constants involved
def fifty_percent_of_500 := 0.50 * 500  -- 50% of 500

-- Prove the equivalence
theorem percentage_of_25_of_fifty_percent_of_500 : (25 / fifty_percent_of_500) * 100 = 10 := by
  -- Place proof steps here
  sorry

end percentage_of_25_of_fifty_percent_of_500_l1677_167721


namespace brownies_pieces_count_l1677_167748

-- Definitions of the conditions
def pan_length : ℕ := 24
def pan_width : ℕ := 15
def pan_area : ℕ := pan_length * pan_width -- pan_area = 360

def piece_length : ℕ := 3
def piece_width : ℕ := 2
def piece_area : ℕ := piece_length * piece_width -- piece_area = 6

-- Definition of the question and proving the expected answer
theorem brownies_pieces_count : (pan_area / piece_area) = 60 := by
  sorry

end brownies_pieces_count_l1677_167748


namespace value_of_f_eval_at_pi_over_12_l1677_167786

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem value_of_f_eval_at_pi_over_12 : f (Real.pi / 12) = (Real.sqrt 6) / 2 :=
by
  sorry

end value_of_f_eval_at_pi_over_12_l1677_167786


namespace determine_properties_range_of_m_l1677_167708

noncomputable def f (a x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem determine_properties (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  ((0 < a ∧ a < 1) → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 > f a x2) ∧
  (a > 1 → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) := 
sorry

theorem range_of_m (a m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_m_in_I : -1 < m ∧ m < 1) :
  f a (m - 1) + f a m < 0 ↔ 
  ((0 < a ∧ a < 1 → (1 / 2) < m ∧ m < 1) ∧
  (a > 1 → 0 < m ∧ m < (1 / 2))) := 
sorry

end determine_properties_range_of_m_l1677_167708


namespace complex_modulus_squared_l1677_167739

open Complex

theorem complex_modulus_squared (w : ℂ) (h : w^2 + abs w ^ 2 = 7 + 2 * I) : abs w ^ 2 = 53 / 14 :=
sorry

end complex_modulus_squared_l1677_167739


namespace tailor_time_l1677_167756

theorem tailor_time (x : ℝ) 
  (t_shirt : ℝ := x) 
  (t_pants : ℝ := 2 * x) 
  (t_jacket : ℝ := 3 * x) 
  (h_capacity : 2 * t_shirt + 3 * t_pants + 4 * t_jacket = 10) : 
  14 * t_shirt + 10 * t_pants + 2 * t_jacket = 20 :=
by
  sorry

end tailor_time_l1677_167756


namespace probability_neither_prime_nor_composite_l1677_167742

/-- Definition of prime number: A number is prime if it has exactly two distinct positive divisors -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of composite number: A number is composite if it has more than two positive divisors -/
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

/-- Given the number in the range 1 to 98 -/
def neither_prime_nor_composite (n : ℕ) : Prop := n = 1

/-- Probability function for uniform probability in a discrete sample space -/
def probability (event_occurrences total_possibilities : ℕ) : ℚ := event_occurrences / total_possibilities

theorem probability_neither_prime_nor_composite :
    probability 1 98 = 1 / 98 := by
  sorry

end probability_neither_prime_nor_composite_l1677_167742


namespace quadratic_roots_sum_squares_l1677_167723

theorem quadratic_roots_sum_squares {a b : ℝ} 
  (h₁ : a + b = -1) 
  (h₂ : a * b = -5) : 
  2 * a^2 + a + b^2 = 16 :=
by sorry

end quadratic_roots_sum_squares_l1677_167723


namespace isosceles_trapezoid_sides_length_l1677_167781

theorem isosceles_trapezoid_sides_length (b1 b2 A : ℝ) (h s : ℝ) 
  (hb1 : b1 = 11) (hb2 : b2 = 17) (hA : A = 56) :
  (A = 1/2 * (b1 + b2) * h) →
  (s ^ 2 = h ^ 2 + (b2 - b1) ^ 2 / 4) →
  s = 5 :=
by
  intro
  sorry

end isosceles_trapezoid_sides_length_l1677_167781


namespace sequence_term_is_correct_l1677_167778

theorem sequence_term_is_correct : ∀ (n : ℕ), (n = 7) → (2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) :=
by
  sorry

end sequence_term_is_correct_l1677_167778


namespace path_area_and_cost_correct_l1677_167783

def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_of_path : ℝ := 759.36
def cost_per_sqm : ℝ := 2
def total_cost : ℝ := 1518.72

theorem path_area_and_cost_correct :
    let length_with_path := length_field + 2 * path_width
    let width_with_path := width_field + 2 * path_width
    let area_with_path := length_with_path * width_with_path
    let area_field := length_field * width_field
    let calculated_area_of_path := area_with_path - area_field
    let calculated_total_cost := calculated_area_of_path * cost_per_sqm
    calculated_area_of_path = area_of_path ∧ calculated_total_cost = total_cost :=
by
    sorry

end path_area_and_cost_correct_l1677_167783


namespace domain_myFunction_l1677_167768

noncomputable def myFunction (x : ℝ) : ℝ :=
  (x^3 - 125) / (x + 125)

theorem domain_myFunction :
  {x : ℝ | ∀ y, y = myFunction x → x ≠ -125} = { x : ℝ | x ≠ -125 } := 
by
  sorry

end domain_myFunction_l1677_167768


namespace FerrisWheelCostIsSix_l1677_167752

structure AmusementPark where
  roller_coaster_cost : ℕ
  log_ride_cost : ℕ
  initial_tickets : ℕ
  additional_tickets_needed : ℕ

def ferris_wheel_cost (a : AmusementPark) : ℕ :=
  let total_needed := a.initial_tickets + a.additional_tickets_needed
  let total_ride_cost := a.roller_coaster_cost + a.log_ride_cost
  total_needed - total_ride_cost

theorem FerrisWheelCostIsSix (a : AmusementPark) 
  (h₁ : a.roller_coaster_cost = 5)
  (h₂ : a.log_ride_cost = 7)
  (h₃ : a.initial_tickets = 2)
  (h₄ : a.additional_tickets_needed = 16) :
  ferris_wheel_cost a = 6 :=
by
  -- proof omitted
  sorry

end FerrisWheelCostIsSix_l1677_167752


namespace transylvanian_human_truth_transylvanian_vampire_lie_l1677_167779

-- Definitions of predicates for human and vampire behavior
def is_human (A : Type) : Prop := ∀ (X : Prop), (A → X) → X
def is_vampire (A : Type) : Prop := ∀ (X : Prop), (A → X) → ¬X

-- Lean definitions for the problem
theorem transylvanian_human_truth (A : Type) (X : Prop) (h_human : is_human A) (h_says_true : A → X) :
  X :=
by sorry

theorem transylvanian_vampire_lie (A : Type) (X : Prop) (h_vampire : is_vampire A) (h_says_true : A → X) :
  ¬X :=
by sorry

end transylvanian_human_truth_transylvanian_vampire_lie_l1677_167779


namespace Matt_received_more_pencils_than_Lauren_l1677_167705

-- Definitions based on conditions
def total_pencils := 2 * 12
def pencils_to_Lauren := 6
def pencils_after_Lauren := total_pencils - pencils_to_Lauren
def pencils_left := 9
def pencils_to_Matt := pencils_after_Lauren - pencils_left

-- Formulate the problem statement
theorem Matt_received_more_pencils_than_Lauren (total_pencils := 24) (pencils_to_Lauren := 6) (pencils_after_Lauren := 18) (pencils_left := 9) (correct_answer := 3) :
  pencils_to_Matt - pencils_to_Lauren = correct_answer := 
by 
  sorry

end Matt_received_more_pencils_than_Lauren_l1677_167705


namespace smallest_number_divisible_by_11_and_conditional_modulus_l1677_167761

theorem smallest_number_divisible_by_11_and_conditional_modulus :
  ∃ n : ℕ, (n % 11 = 0) ∧ (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n % 7 = 2) ∧ n = 2102 :=
by
  sorry

end smallest_number_divisible_by_11_and_conditional_modulus_l1677_167761


namespace flowers_in_each_row_l1677_167701

theorem flowers_in_each_row (rows : ℕ) (total_remaining_flowers : ℕ) 
  (percentage_remaining : ℚ) (correct_rows : rows = 50) 
  (correct_remaining : total_remaining_flowers = 8000) 
  (correct_percentage : percentage_remaining = 0.40) :
  (total_remaining_flowers : ℚ) / percentage_remaining / (rows : ℚ) = 400 := 
by {
 sorry
}

end flowers_in_each_row_l1677_167701


namespace find_expression_l1677_167796

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_x2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem find_expression (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : symmetric_about_x2 f)
  (h3 : ∀ x, -2 < x ∧ x ≤ 2 → f x = -x^2 + 1) :
  ∀ x, -6 < x ∧ x < -2 → f x = -(x + 4)^2 + 1 :=
by
  sorry

end find_expression_l1677_167796


namespace pradeep_maximum_marks_l1677_167784

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.35 * M = 175) :
  M = 500 :=
by
  sorry

end pradeep_maximum_marks_l1677_167784


namespace coin_loading_impossible_l1677_167714

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l1677_167714


namespace trigonometric_identity_l1677_167790

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by 
  sorry

end trigonometric_identity_l1677_167790


namespace triangle_exists_among_single_color_sticks_l1677_167759

theorem triangle_exists_among_single_color_sticks
  (red yellow green : ℕ)
  (k y g K Y G : ℕ)
  (hk : k + y > G)
  (hy : y + g > K)
  (hg : g + k > Y)
  (hred : red = 100)
  (hyellow : yellow = 100)
  (hgreen : green = 100) :
  ∃ color : string, ∀ a b c : ℕ, (a = k ∨ a = K) → (b = k ∨ b = K) → (c = k ∨ c = K) → a + b > c :=
sorry

end triangle_exists_among_single_color_sticks_l1677_167759


namespace percentage_increase_on_friday_l1677_167757

theorem percentage_increase_on_friday (avg_books_per_day : ℕ) (friday_books : ℕ) (total_books_per_week : ℕ) (days_open : ℕ)
  (h1 : avg_books_per_day = 40)
  (h2 : total_books_per_week = 216)
  (h3 : days_open = 5)
  (h4 : friday_books > avg_books_per_day) :
  (((friday_books - avg_books_per_day) * 100) / avg_books_per_day) = 40 :=
sorry

end percentage_increase_on_friday_l1677_167757


namespace water_added_l1677_167700

theorem water_added (W X : ℝ) 
  (h1 : 45 / W = 2 / 1)
  (h2 : 45 / (W + X) = 6 / 5) : 
  X = 15 := 
by
  sorry

end water_added_l1677_167700


namespace coeff_x2_in_x_minus_1_pow_4_l1677_167791

theorem coeff_x2_in_x_minus_1_pow_4 :
  ∀ (x : ℝ), (∃ (p : ℕ), (x - 1) ^ 4 = p * x^2 + (other_terms) ∧ p = 6) :=
by sorry

end coeff_x2_in_x_minus_1_pow_4_l1677_167791


namespace new_avg_weight_l1677_167794

theorem new_avg_weight 
  (initial_avg_weight : ℝ)
  (initial_num_members : ℕ)
  (new_person1_weight : ℝ)
  (new_person2_weight : ℝ)
  (new_num_members : ℕ)
  (final_total_weight : ℝ)
  (final_avg_weight : ℝ) :
  initial_avg_weight = 48 →
  initial_num_members = 23 →
  new_person1_weight = 78 →
  new_person2_weight = 93 →
  new_num_members = initial_num_members + 2 →
  final_total_weight = (initial_avg_weight * initial_num_members) + new_person1_weight + new_person2_weight →
  final_avg_weight = final_total_weight / new_num_members →
  final_avg_weight = 51 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end new_avg_weight_l1677_167794


namespace pamTotalApples_l1677_167754

-- Define the given conditions
def applesPerGeraldBag : Nat := 40
def applesPerPamBag := 3 * applesPerGeraldBag
def pamBags : Nat := 10

-- Statement to prove
theorem pamTotalApples : pamBags * applesPerPamBag = 1200 :=
by
  sorry

end pamTotalApples_l1677_167754


namespace find_t_l1677_167785

theorem find_t (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
    (havg : (k + m + r + s + t) / 5 = 18)
    (hmed : r = 23) 
    (hpos_k : 0 < k)
    (hpos_m : 0 < m)
    (hpos_r : 0 < r)
    (hpos_s : 0 < s)
    (hpos_t : 0 < t) :
  t = 40 := sorry

end find_t_l1677_167785


namespace no_prime_divisible_by_56_l1677_167787

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end no_prime_divisible_by_56_l1677_167787


namespace rounding_to_one_decimal_place_l1677_167769

def number_to_round : Float := 5.049

def rounded_value : Float := 5.0

theorem rounding_to_one_decimal_place :
  (Float.round (number_to_round * 10) / 10) = rounded_value :=
by
  sorry

end rounding_to_one_decimal_place_l1677_167769


namespace prime_remainder_l1677_167722

theorem prime_remainder (p : ℕ) (k : ℕ) (h1 : Prime p) (h2 : p > 3) :
  (∃ k, p = 6 * k + 1 ∧ (p^3 + 17) % 24 = 18) ∨
  (∃ k, p = 6 * k - 1 ∧ (p^3 + 17) % 24 = 16) :=
by
  sorry

end prime_remainder_l1677_167722


namespace product_correct_l1677_167725

/-- Define the number and the digit we're interested in -/
def num : ℕ := 564823
def digit : ℕ := 4

/-- Define a function to calculate the local value of the digit 4 in the number 564823 -/
def local_value (n : ℕ) (d : ℕ) := if d = 4 then 40000 else 0

/-- Define a function to calculate the absolute value, although it is trivial for natural numbers -/
def abs_value (d : ℕ) := d

/-- Define the product of local value and absolute value of 4 in 564823 -/
def product := local_value num digit * abs_value digit

/-- Theorem stating that the product is as specified in the problem -/
theorem product_correct : product = 160000 :=
by
  sorry

end product_correct_l1677_167725


namespace hyunwoo_family_saving_l1677_167774

def daily_water_usage : ℝ := 215
def saving_factor : ℝ := 0.32

theorem hyunwoo_family_saving:
  daily_water_usage * saving_factor = 68.8 := by
  sorry

end hyunwoo_family_saving_l1677_167774


namespace sophie_germain_identity_l1677_167767

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4 * b^4 = (a^2 + 2 * a * b + 2 * b^2) * (a^2 - 2 * a * b + 2 * b^2) :=
by sorry

end sophie_germain_identity_l1677_167767


namespace tom_paid_amount_correct_l1677_167704

def kg (n : Nat) : Nat := n -- Just a type alias clarification

theorem tom_paid_amount_correct :
  ∀ (quantity_apples : Nat) (rate_apples : Nat) (quantity_mangoes : Nat) (rate_mangoes : Nat),
  quantity_apples = kg 8 →
  rate_apples = 70 →
  quantity_mangoes = kg 9 →
  rate_mangoes = 55 →
  (quantity_apples * rate_apples) + (quantity_mangoes * rate_mangoes) = 1055 :=
by
  intros quantity_apples rate_apples quantity_mangoes rate_mangoes
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end tom_paid_amount_correct_l1677_167704


namespace equation_solution_l1677_167764

theorem equation_solution (t : ℤ) : 
  ∃ y : ℤ, (21 * t + 2)^3 + 2 * (21 * t + 2)^2 + 5 = 21 * y :=
sorry

end equation_solution_l1677_167764


namespace eq_solutions_a2_eq_b_times_b_plus_7_l1677_167745

theorem eq_solutions_a2_eq_b_times_b_plus_7 (a b : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h : a^2 = b * (b + 7)) :
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end eq_solutions_a2_eq_b_times_b_plus_7_l1677_167745


namespace total_books_read_l1677_167776

-- Given conditions
variables (c s : ℕ) -- variable c represents the number of classes, s represents the number of students per class

-- Main statement to prove
theorem total_books_read (h1 : ∀ a, a = 7) (h2 : ∀ b, b = 12) :
  84 * c * s = 84 * c * s :=
by
  sorry

end total_books_read_l1677_167776


namespace general_term_formula_of_sequence_l1677_167736

theorem general_term_formula_of_sequence {a : ℕ → ℝ} (S : ℕ → ℝ)
  (hS : ∀ n, S n = (2 / 3) * a n + 1 / 3) :
  (∀ n, a n = (-2) ^ (n - 1)) :=
by
  sorry

end general_term_formula_of_sequence_l1677_167736


namespace line_through_two_points_l1677_167746

theorem line_through_two_points (x_1 y_1 x_2 y_2 x y : ℝ) :
  (x - x_1) * (y_2 - y_1) = (y - y_1) * (x_2 - x_1) :=
sorry

end line_through_two_points_l1677_167746


namespace points_lie_on_circle_l1677_167706

theorem points_lie_on_circle (s : ℝ) :
  ( (2 - s^2) / (2 + s^2) )^2 + ( 3 * s / (2 + s^2) )^2 = 1 :=
by sorry

end points_lie_on_circle_l1677_167706


namespace simplify_fraction_when_b_equals_4_l1677_167760

theorem simplify_fraction_when_b_equals_4 (b : ℕ) (h : b = 4) : (18 * b^4) / (27 * b^3) = 8 / 3 :=
by {
  -- we use the provided condition to state our theorem goals.
  sorry
}

end simplify_fraction_when_b_equals_4_l1677_167760


namespace largest_w_exists_l1677_167713

theorem largest_w_exists (w x y z : ℝ) (h1 : w + x + y + z = 25) (h2 : w * x + w * y + w * z + x * y + x * z + y * z = 2 * y + 2 * z + 193) :
  ∃ (w1 w2 : ℤ), w1 > 0 ∧ w2 > 0 ∧ ((w = w1 / w2) ∧ (w1 + w2 = 27)) :=
sorry

end largest_w_exists_l1677_167713


namespace number_of_outfits_l1677_167727

-- Definitions based on conditions a)
def trousers : ℕ := 5
def shirts : ℕ := 7
def jackets : ℕ := 3
def specific_trousers : ℕ := 2
def specific_jackets : ℕ := 2

-- Lean 4 theorem statement to prove the number of outfits
theorem number_of_outfits (trousers shirts jackets specific_trousers specific_jackets : ℕ) :
  (3 * jackets + specific_trousers * specific_jackets) * shirts = 91 :=
by
  sorry

end number_of_outfits_l1677_167727


namespace range_of_a_l1677_167720

def A : Set ℝ := { x | x^2 - 3 * x + 2 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 1 / (x - 3) < a }

theorem range_of_a (a : ℝ) : A ⊆ B a ↔ a > -1/2 :=
by sorry

end range_of_a_l1677_167720


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l1677_167716

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l1677_167716


namespace fixed_point_of_parabolas_l1677_167793

theorem fixed_point_of_parabolas 
  (t : ℝ) 
  (fixed_x fixed_y : ℝ) 
  (hx : fixed_x = 2) 
  (hy : fixed_y = 12) 
  (H : ∀ t : ℝ, ∃ y : ℝ, y = 3 * fixed_x^2 + t * fixed_x - 2 * t) : 
  ∃ y : ℝ, y = fixed_y :=
by
  sorry

end fixed_point_of_parabolas_l1677_167793


namespace problem_statement_l1677_167762

-- Define the function g and specify its properties
def g : ℕ → ℕ := sorry

axiom g_property (a b : ℕ) : g (a^2 + b^2) + g (a + b) = (g a)^2 + (g b)^2

-- Define the values of m and t that arise from the constraints on g(49)
def m : ℕ := 2
def t : ℕ := 106

-- Prove that the product m * t is 212
theorem problem_statement : m * t = 212 :=
by {
  -- Since g_property is an axiom, we use it to derive that
  -- g(49) can only take possible values 0 and 106,
  -- thus m = 2 and t = 106.
  sorry
}

end problem_statement_l1677_167762


namespace taxi_fare_l1677_167731

-- Define the necessary values and functions based on the problem conditions
def starting_price : ℝ := 6
def additional_charge_per_km : ℝ := 1.5
def distance (P : ℝ) : Prop := P > 6

-- Lean proposition to state the problem
theorem taxi_fare (P : ℝ) (hP : distance P) : 
  (starting_price + additional_charge_per_km * (P - 6)) = 1.5 * P - 3 := 
by 
  sorry

end taxi_fare_l1677_167731


namespace prove_inequality_l1677_167729

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end prove_inequality_l1677_167729


namespace find_integer_n_l1677_167719

theorem find_integer_n (n : ℤ) (hn : -150 < n ∧ n < 150) : (n = 80 ∨ n = -100) ↔ (Real.tan (n * Real.pi / 180) = Real.tan (1340 * Real.pi / 180)) :=
by 
  sorry

end find_integer_n_l1677_167719


namespace area_of_ABCD_proof_l1677_167737

noncomputable def point := ℝ × ℝ

structure Rectangle :=
  (A B C D : point)
  (angle_C_trisected_by_CE_CF : Prop)
  (E_on_AB : Prop)
  (F_on_AD : Prop)
  (AF : ℝ)
  (BE : ℝ)

def area_of_rectangle (rect : Rectangle) : ℝ :=
  let (x1, y1) := rect.A
  let (x2, y2) := rect.C
  (x2 - x1) * (y2 - y1)

theorem area_of_ABCD_proof :
  ∀ (ABCD : Rectangle),
    ABCD.angle_C_trisected_by_CE_CF →
    ABCD.E_on_AB →
    ABCD.F_on_AD →
    ABCD.AF = 2 →
    ABCD.BE = 6 →
    abs (area_of_rectangle ABCD - 150) < 1 :=
by
  sorry

end area_of_ABCD_proof_l1677_167737


namespace possible_numbers_erased_one_digit_reduce_sixfold_l1677_167788

theorem possible_numbers_erased_one_digit_reduce_sixfold (N : ℕ) :
  (∃ N' : ℕ, N = 6 * N' ∧ N % 10 ≠ 0 ∧ ¬N = N') ↔
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 ∨ N = 108 :=
by {
  sorry
}

end possible_numbers_erased_one_digit_reduce_sixfold_l1677_167788


namespace equivalent_proof_problem_l1677_167772

variable (a b d e c f g h : ℚ)

def condition1 : Prop := 8 = (6 / 100) * a
def condition2 : Prop := 6 = (8 / 100) * b
def condition3 : Prop := 9 = (5 / 100) * d
def condition4 : Prop := 7 = (3 / 100) * e
def condition5 : Prop := c = b / a
def condition6 : Prop := f = d / a
def condition7 : Prop := g = e / b

theorem equivalent_proof_problem (hac1 : condition1 a)
                                 (hac2 : condition2 b)
                                 (hac3 : condition3 d)
                                 (hac4 : condition4 e)
                                 (hac5 : condition5 a b c)
                                 (hac6 : condition6 a d f)
                                 (hac7 : condition7 b e g) :
    h = f + g ↔ h = (803 / 20) * c := 
by sorry

end equivalent_proof_problem_l1677_167772


namespace range_of_a_l1677_167718

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a * x + 3 ≥ a) ↔ -7 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l1677_167718


namespace painting_time_l1677_167798

theorem painting_time (rate_taylor rate_jennifer rate_alex : ℚ) 
  (h_taylor : rate_taylor = 1 / 12) 
  (h_jennifer : rate_jennifer = 1 / 10) 
  (h_alex : rate_alex = 1 / 15) : 
  ∃ t : ℚ, t = 4 ∧ (1 / t) = rate_taylor + rate_jennifer + rate_alex :=
by
  sorry

end painting_time_l1677_167798


namespace school_trip_seat_count_l1677_167743

theorem school_trip_seat_count :
  ∀ (classrooms students_per_classroom seats_per_bus : ℕ),
  classrooms = 87 →
  students_per_classroom = 58 →
  seats_per_bus = 29 →
  ∀ (total_students total_buses_needed : ℕ),
  total_students = classrooms * students_per_classroom →
  total_buses_needed = (total_students + seats_per_bus - 1) / seats_per_bus →
  seats_per_bus = 29 := by
  intros classrooms students_per_classroom seats_per_bus
  intros h1 h2 h3
  intros total_students total_buses_needed
  intros h4 h5
  sorry

end school_trip_seat_count_l1677_167743


namespace sum_of_reciprocals_l1677_167730

variable {x y : ℝ}

theorem sum_of_reciprocals (h1 : x + y = 4 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x + 1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l1677_167730


namespace sum_reciprocal_factors_of_12_l1677_167758

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l1677_167758
