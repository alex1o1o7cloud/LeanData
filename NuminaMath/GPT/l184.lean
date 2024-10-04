import Mathlib

namespace gcd_459_357_l184_184642

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l184_184642


namespace regression_equation_pos_corr_l184_184172

noncomputable def linear_regression (x y : ℝ) : ℝ := 0.4 * x + 2.5

theorem regression_equation_pos_corr (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (mean_x : ℝ := 2.5) (mean_y : ℝ := 3.5)
    (pos_corr : x * y > 0)
    (cond1 : mean_x = 2.5)
    (cond2 : mean_y = 3.5) :
    linear_regression mean_x mean_y = mean_y :=
by
  sorry

end regression_equation_pos_corr_l184_184172


namespace bears_in_shipment_l184_184663

theorem bears_in_shipment (initial_bears shipment_bears bears_per_shelf total_shelves : ℕ)
  (h1 : initial_bears = 17)
  (h2 : bears_per_shelf = 9)
  (h3 : total_shelves = 3)
  (h4 : total_shelves * bears_per_shelf = 27) :
  shipment_bears = 10 :=
by
  sorry

end bears_in_shipment_l184_184663


namespace work_completion_by_b_l184_184261

theorem work_completion_by_b (a_days : ℕ) (a_solo_days : ℕ) (a_b_combined_days : ℕ) (b_days : ℕ) :
  a_days = 12 ∧ a_solo_days = 3 ∧ a_b_combined_days = 5 → b_days = 15 :=
by
  sorry

end work_completion_by_b_l184_184261


namespace sum_interior_angles_of_regular_polygon_l184_184229

theorem sum_interior_angles_of_regular_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (n : ℝ)
  (h1 : exterior_angle = 45)
  (h2 : sum_exterior_angles = 360)
  (h3 : n = sum_exterior_angles / exterior_angle) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_interior_angles_of_regular_polygon_l184_184229


namespace divide_oranges_into_pieces_l184_184601

-- Definitions for conditions
def oranges : Nat := 80
def friends : Nat := 200
def pieces_per_friend : Nat := 4

-- Theorem stating the problem and the answer
theorem divide_oranges_into_pieces :
    (oranges > 0) → (friends > 0) → (pieces_per_friend > 0) →
    ((friends * pieces_per_friend) / oranges = 10) :=
by
  intros
  sorry

end divide_oranges_into_pieces_l184_184601


namespace men_in_first_group_l184_184339

noncomputable def first_group_men (x m b W : ℕ) : Prop :=
  let eq1 := 10 * x * m + 80 * b = W
  let eq2 := 2 * (26 * m + 48 * b) = W
  let eq3 := 4 * (15 * m + 20 * b) = W
  eq1 ∧ eq2 ∧ eq3

theorem men_in_first_group (m b W : ℕ) (h_condition : first_group_men 6 m b W) : 
  ∃ x, x = 6 :=
by
  sorry

end men_in_first_group_l184_184339


namespace common_ratio_is_two_l184_184271

theorem common_ratio_is_two (a r : ℝ) (h_pos : a > 0) 
  (h_sum : a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r)) : 
  r = 2 := 
by
  sorry

end common_ratio_is_two_l184_184271


namespace equilateral_triangle_surface_area_correct_l184_184164

noncomputable def equilateral_triangle_surface_area : ℝ :=
  let side_length := 2
  let A := (0, 0, 0)
  let B := (side_length, 0, 0)
  let C := (side_length / 2, (side_length * (Real.sqrt 3)) / 2, 0)
  let D := (side_length / 2, (side_length * (Real.sqrt 3)) / 6, 0)
  let folded_angle := 90
  let diagonal_length := Real.sqrt (1 + 1 + 3)
  let radius := diagonal_length / 2
  let surface_area := 4 * Real.pi * radius^2
  5 * Real.pi

theorem equilateral_triangle_surface_area_correct :
  equilateral_triangle_surface_area = 5 * Real.pi :=
by
  unfold equilateral_triangle_surface_area
  sorry -- proof omitted

end equilateral_triangle_surface_area_correct_l184_184164


namespace ratio_of_John_to_Mary_l184_184197

-- Definitions based on conditions
variable (J M T : ℕ)
variable (hT : T = 60)
variable (hJ : J = T / 2)
variable (hAvg : (J + M + T) / 3 = 35)

-- Statement to prove
theorem ratio_of_John_to_Mary : J / M = 2 := by
  -- Proof goes here
  sorry

end ratio_of_John_to_Mary_l184_184197


namespace solve_for_x_l184_184625

-- Step d: Lean 4 statement
theorem solve_for_x : 
  (∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2)) → (∃ x : ℚ, x = 1 / 3) :=
sorry

end solve_for_x_l184_184625


namespace integer_solutions_count_for_equation_l184_184392

theorem integer_solutions_count_for_equation :
  (∃ n : ℕ, (∀ x y : ℤ, (1/x + 1/y = 1/7) → (x ≠ 0) → (y ≠ 0) → n = 5 )) :=
sorry

end integer_solutions_count_for_equation_l184_184392


namespace shortest_distance_from_curve_to_line_l184_184396

noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_from_curve_to_line : 
  ∃ (x y : ℝ), y = curve x ∧ line x y ∧ 
  (∀ (x₀ y₀ : ℝ), y₀ = curve x₀ → ∃ (x₀ y₀ : ℝ), 
    y₀ = curve x₀ ∧ d = Real.sqrt 5) :=
sorry

end shortest_distance_from_curve_to_line_l184_184396


namespace total_trip_time_l184_184097

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end total_trip_time_l184_184097


namespace purse_multiple_of_wallet_l184_184561

theorem purse_multiple_of_wallet (W P : ℤ) (hW : W = 22) (hc : W + P = 107) : ∃ n : ℤ, n * W > P ∧ n = 4 :=
by
  sorry

end purse_multiple_of_wallet_l184_184561


namespace least_possible_b_l184_184074

theorem least_possible_b (a b : ℕ) (ha : a.prime) (hb : b.prime) (sum_90 : a + b = 90) (a_greater_b : a > b) : b = 7 :=
by
  sorry

end least_possible_b_l184_184074


namespace range_of_m_l184_184703

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  (¬(∃ u v : ℝ, u ≠ v ∧ 4*u^2 + 4*(m - 2)*u + 1 = 0 ∧ 4*v^2 + 4*(m - 2)*v + 1 = 0)) →
  m ∈ set.Ioo (-∞ : ℝ) (-2) ∪ set.Ioc 1 2 ∪ set.Ici 3 :=
sorry

end range_of_m_l184_184703


namespace find_a1_for_geometric_sequence_l184_184573

noncomputable def geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : geometric_sequence) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem find_a1_for_geometric_sequence (a : geometric_sequence)
  (h_geom : is_geometric_sequence a)
  (h1 : a 2 * a 5 = 2 * a 3)
  (h2 : (a 4 + a 6) / 2 = 5 / 4) :
  a 1 = 16 ∨ a 1 = -16 :=
sorry

end find_a1_for_geometric_sequence_l184_184573


namespace hamiltonian_cycle_exists_l184_184604

theorem hamiltonian_cycle_exists {G : Graph V} (n : ℕ) (hG : G.is_connected) 
  (hdeg : ∀ v : G.V, G.degree v ≥ n / 2) (hn : G.verts.card = n) : 
  G.is_hamiltonian :=
begin
  sorry
end

end hamiltonian_cycle_exists_l184_184604


namespace number_of_5_letter_words_with_at_least_one_vowel_l184_184883

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l184_184883


namespace ways_to_distribute_balls_l184_184893

theorem ways_to_distribute_balls :
  let balls : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
  let boxes : Finset ℕ := {0, 1, 2, 3}
  let choose_distinct (n k : ℕ) : ℕ := Nat.choose n k
  let distribution_patterns : List (ℕ × ℕ × ℕ × ℕ) := 
    [(6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0), 
     (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)]
  let ways_to_pattern (pattern : ℕ × ℕ × ℕ × ℕ) : ℕ :=
    match pattern with
    | (6,0,0,0) => 1
    | (5,1,0,0) => choose_distinct 6 5
    | (4,2,0,0) => choose_distinct 6 4 * choose_distinct 2 2
    | (4,1,1,0) => choose_distinct 6 4
    | (3,3,0,0) => choose_distinct 6 3 * choose_distinct 3 3 / 2
    | (3,2,1,0) => choose_distinct 6 3 * choose_distinct 3 2 * choose_distinct 1 1
    | (3,1,1,1) => choose_distinct 6 3
    | (2,2,2,0) => choose_distinct 6 2 * choose_distinct 4 2 * choose_distinct 2 2 / 6
    | (2,2,1,1) => choose_distinct 6 2 * choose_distinct 4 2 / 2
    | _ => 0
  let total_ways : ℕ := distribution_patterns.foldl (λ acc x => acc + ways_to_pattern x) 0
  total_ways = 182 := by
  sorry

end ways_to_distribute_balls_l184_184893


namespace spider_crawl_distance_l184_184550

theorem spider_crawl_distance :
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  abs (b - a) + abs (c - b) + abs (d - c) = 20 :=
by
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  sorry

end spider_crawl_distance_l184_184550


namespace sasha_total_items_l184_184379

/-
  Sasha bought pencils at 13 rubles each and pens at 20 rubles each,
  paying a total of 350 rubles. 
  Prove that the total number of pencils and pens Sasha bought is 23.
-/
theorem sasha_total_items
  (x y : ℕ) -- Define x as the number of pencils and y as the number of pens
  (H: 13 * x + 20 * y = 350) -- Given total cost condition
  : x + y = 23 := 
sorry

end sasha_total_items_l184_184379


namespace geometric_sequence_third_fourth_terms_l184_184868

theorem geometric_sequence_third_fourth_terms
  (a : ℕ → ℝ)
  (r : ℝ)
  (ha : ∀ n, a (n + 1) = r * a n)
  (hS2 : a 0 + a 1 = 3 * a 1) :
  (a 2 + a 3) / (a 0 + a 1) = 1 / 4 :=
by
  -- proof to be filled in
  sorry

end geometric_sequence_third_fourth_terms_l184_184868


namespace pinocchio_start_time_l184_184215

### Definitions for conditions

def pinocchio_arrival_time : ℚ := 22
def faster_arrival_time : ℚ := 21.5
def time_saved : ℚ := 0.5

### Proving when Pinocchio left the house

theorem pinocchio_start_time : 
  ∃ t : ℚ, pinocchio_arrival_time - t = 2.5 ∧ t = 19.5 :=
by
  sorry

end pinocchio_start_time_l184_184215


namespace expand_expression_l184_184447

theorem expand_expression (x y : ℝ) : 
  5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := 
by 
  sorry

end expand_expression_l184_184447


namespace cost_price_per_meter_l184_184528

-- Define the given conditions
def selling_price : ℕ := 8925
def meters : ℕ := 85
def profit_per_meter : ℕ := 35

-- Define the statement to be proved
theorem cost_price_per_meter :
  (selling_price - profit_per_meter * meters) / meters = 70 := 
by
  sorry

end cost_price_per_meter_l184_184528


namespace factory_minimize_salary_l184_184112

theorem factory_minimize_salary :
  ∃ x : ℕ, ∃ W : ℕ,
    x + (120 - x) = 120 ∧
    800 * x + 1000 * (120 - x) = W ∧
    120 - x ≥ 3 * x ∧
    x = 30 ∧
    W = 114000 :=
  sorry

end factory_minimize_salary_l184_184112


namespace compute_vector_expression_l184_184836

theorem compute_vector_expression :
  4 • (⟨3, -5⟩ : ℝ × ℝ) - 3 • (⟨2, -6⟩ : ℝ × ℝ) + 2 • (⟨0, 3⟩ : ℝ × ℝ) = (⟨6, 4⟩ : ℝ × ℝ) := 
sorry

end compute_vector_expression_l184_184836


namespace problem_statement_l184_184460

theorem problem_statement (x y z w : ℝ)
  (h1 : x + y + z + w = 0)
  (h7 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := 
sorry

end problem_statement_l184_184460


namespace team_t_speed_l184_184516

theorem team_t_speed (v t : ℝ) (h1 : 300 = v * t) (h2 : 300 = (v + 5) * (t - 3)) : v = 20 :=
by 
  sorry

end team_t_speed_l184_184516


namespace total_number_of_numbers_l184_184501

theorem total_number_of_numbers (avg : ℝ) (sum1 sum2 sum3 : ℝ) (N : ℝ) :
  avg = 3.95 →
  sum1 = 2 * 3.8 →
  sum2 = 2 * 3.85 →
  sum3 = 2 * 4.200000000000001 →
  avg = (sum1 + sum2 + sum3) / N →
  N = 6 :=
by
  intros h_avg h_sum1 h_sum2 h_sum3 h_total
  sorry

end total_number_of_numbers_l184_184501


namespace Q_share_of_profit_l184_184768

theorem Q_share_of_profit (P Q T : ℕ) (hP : P = 54000) (hQ : Q = 36000) (hT : T = 18000) : Q's_share = 7200 :=
by
  -- Definitions and conditions
  let P := 54000
  let Q := 36000
  let T := 18000
  have P_ratio := 3
  have Q_ratio := 2
  have ratio_sum := P_ratio + Q_ratio
  have Q's_share := (T * Q_ratio) / ratio_sum
  
  -- Q's share of the profit
  sorry

end Q_share_of_profit_l184_184768


namespace quadratic_identity_l184_184026

theorem quadratic_identity (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 :=
by
  sorry

end quadratic_identity_l184_184026


namespace largest_b_value_l184_184910

open Real

structure Triangle :=
(side_a side_b side_c : ℝ)
(a_pos : 0 < side_a)
(b_pos : 0 < side_b)
(c_pos : 0 < side_c)
(tri_ineq_a : side_a + side_b > side_c)
(tri_ineq_b : side_b + side_c > side_a)
(tri_ineq_c : side_c + side_a > side_b)

noncomputable def inradius (T : Triangle) : ℝ :=
  let s := (T.side_a + T.side_b + T.side_c) / 2
  let A := sqrt (s * (s - T.side_a) * (s - T.side_b) * (s - T.side_c))
  A / s

noncomputable def circumradius (T : Triangle) : ℝ :=
  let A := sqrt (((T.side_a + T.side_b + T.side_c) / 2) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_a) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_b) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_c))
  (T.side_a * T.side_b * T.side_c) / (4 * A)

noncomputable def condition_met (T1 T2 : Triangle) : Prop :=
  (inradius T1 / circumradius T1) = (inradius T2 / circumradius T2)

theorem largest_b_value :
  let T1 := Triangle.mk 8 11 11 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  ∃ b > 0, ∃ T2 : Triangle, T2.side_a = b ∧ T2.side_b = 1 ∧ T2.side_c = 1 ∧ b = 14 / 11 ∧ condition_met T1 T2 :=
  sorry

end largest_b_value_l184_184910


namespace minimize_sum_of_f_seq_l184_184859

def f (x : ℝ) : ℝ := x^2 - 8 * x + 10

def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem minimize_sum_of_f_seq
  (a : ℕ → ℝ)
  (h₀ : isArithmeticSequence a 1)
  (h₁ : a 1 = a₁)
  : f (a 1) + f (a 2) + f (a 3) = 3 * a₁^2 - 18 * a₁ + 30 →

  (∀ x, 3 * x^2 - 18 * x + 30 ≥ 3 * 3^2 - 18 * 3 + 30) →
  a₁ = 3 :=
by
  sorry

end minimize_sum_of_f_seq_l184_184859


namespace correct_mean_l184_184647

-- Definitions of conditions
def n : ℕ := 30
def mean_incorrect : ℚ := 140
def value_correct : ℕ := 145
def value_incorrect : ℕ := 135

-- The statement to be proved
theorem correct_mean : 
  let S_incorrect := mean_incorrect * n
  let Difference := value_correct - value_incorrect
  let S_correct := S_incorrect + Difference
  let mean_correct := S_correct / n
  mean_correct = 140.33 := 
by
  sorry

end correct_mean_l184_184647


namespace find_special_two_digit_integer_l184_184664

theorem find_special_two_digit_integer (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : (n + 3) % 3 = 0)
  (h3 : (n + 4) % 4 = 0)
  (h4 : (n + 5) % 5 = 0) :
  n = 60 := by
  sorry

end find_special_two_digit_integer_l184_184664


namespace quadratic_inequality_empty_set_l184_184030

theorem quadratic_inequality_empty_set (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 < 0)) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end quadratic_inequality_empty_set_l184_184030


namespace largest_divisor_of_five_consecutive_integers_product_correct_l184_184961

noncomputable def largest_divisor_of_five_consecutive_integers_product : ℕ :=
  120

theorem largest_divisor_of_five_consecutive_integers_product_correct :
  ∀ (n : ℕ), (∃ k : ℕ, k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ∧ 120 ∣ k) :=
sorry

end largest_divisor_of_five_consecutive_integers_product_correct_l184_184961


namespace area_of_field_l184_184633

theorem area_of_field (w l A : ℝ) 
    (h1 : l = 2 * w + 35) 
    (h2 : 2 * (w + l) = 700) : 
    A = 25725 :=
by sorry

end area_of_field_l184_184633


namespace finish_lollipops_in_6_days_l184_184880

variables (henry_alison_diff : ℕ) (alison_lollipops : ℕ) (diane_alison_ratio : ℕ) (lollipops_eaten_per_day : ℕ)
variables (days_needed : ℕ) (henry_lollipops : ℕ) (diane_lollipops : ℕ) (total_lollipops : ℕ)

-- Conditions as definitions
def condition_1 : Prop := henry_alison_diff = 30
def condition_2 : Prop := alison_lollipops = 60
def condition_3 : Prop := alison_lollipops * 2 = diane_lollipops
def condition_4 : Prop := lollipops_eaten_per_day = 45

-- Total lollipops calculation
def total_lollipops_calculated : ℕ := alison_lollipops + diane_lollipops + henry_lollipops

-- Days to finish lollipops calculation
def days_needed_calculated : ℕ := total_lollipops / lollipops_eaten_per_day

-- The theorem to prove
theorem finish_lollipops_in_6_days :
  condition_1 →
  condition_2 →
  condition_3 →
  condition_4 →
  henry_lollipops = alison_lollipops + 30 →
  total_lollipops_calculated = 270 →
  days_needed_calculated = 6 :=
by {
  sorry
}

end finish_lollipops_in_6_days_l184_184880


namespace set_has_one_element_iff_double_root_l184_184592

theorem set_has_one_element_iff_double_root (k : ℝ) :
  (∃ x, ∀ y, y^2 - k*y + 1 = 0 ↔ y = x) ↔ k = 2 ∨ k = -2 :=
by
  sorry

end set_has_one_element_iff_double_root_l184_184592


namespace minimum_raft_weight_l184_184988

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l184_184988


namespace integer_solutions_to_abs_equation_l184_184069

theorem integer_solutions_to_abs_equation :
  {p : ℤ × ℤ | abs (p.1 - 2) + abs (p.2 - 1) = 1} =
  {(3, 1), (1, 1), (2, 2), (2, 0)} :=
by
  sorry

end integer_solutions_to_abs_equation_l184_184069


namespace book_organizing_activity_l184_184801

theorem book_organizing_activity (x : ℕ) (h₁ : x > 0):
  (80 : ℝ) / (x + 5 : ℝ) = (70 : ℝ) / (x : ℝ) :=
sorry

end book_organizing_activity_l184_184801


namespace mimi_spent_on_clothes_l184_184207

noncomputable def total_cost : ℤ := 8000
noncomputable def cost_adidas : ℤ := 600
noncomputable def cost_nike : ℤ := 3 * cost_adidas
noncomputable def cost_skechers : ℤ := 5 * cost_adidas
noncomputable def cost_clothes : ℤ := total_cost - (cost_adidas + cost_nike + cost_skechers)

theorem mimi_spent_on_clothes :
  cost_clothes = 2600 :=
by
  sorry

end mimi_spent_on_clothes_l184_184207


namespace cats_not_eating_either_l184_184595

theorem cats_not_eating_either (total_cats : ℕ) (cats_liking_apples : ℕ) (cats_liking_fish : ℕ) (cats_liking_both : ℕ)
  (h1 : total_cats = 75) (h2 : cats_liking_apples = 15) (h3 : cats_liking_fish = 55) (h4 : cats_liking_both = 8) :
  ∃ cats_not_eating_either : ℕ, cats_not_eating_either = total_cats - (cats_liking_apples - cats_liking_both + cats_liking_fish - cats_liking_both + cats_liking_both) ∧ cats_not_eating_either = 13 :=
by
  sorry

end cats_not_eating_either_l184_184595


namespace A_runs_faster_l184_184264

variable (v_A v_B : ℝ)  -- Speed of A and B
variable (k : ℝ)       -- Factor by which A is faster than B

-- Conditions as definitions in Lean:
def speed_relation (k : ℝ) (v_A v_B : ℝ) : Prop := v_A = k * v_B
def start_difference : ℝ := 60
def race_course_length : ℝ := 80
def reach_finish_same_time (v_A v_B : ℝ) : Prop := (80 / v_A) = ((80 - start_difference) / v_B)

theorem A_runs_faster
  (h1 : speed_relation k v_A v_B)
  (h2 : reach_finish_same_time v_A v_B) : k = 4 :=
by
  sorry

end A_runs_faster_l184_184264


namespace intersection_M_P_l184_184328

def is_natural (x : ℤ) : Prop := x ≥ 0

def M (x : ℤ) : Prop := (x - 1)^2 < 4 ∧ is_natural x

def P := ({-1, 0, 1, 2, 3} : Set ℤ)

theorem intersection_M_P :
  {x : ℤ | M x} ∩ P = {0, 1, 2} :=
  sorry

end intersection_M_P_l184_184328


namespace temperature_on_Friday_l184_184628

variable (M T W Th F : ℝ)

def avg_M_T_W_Th := (M + T + W + Th) / 4 = 48
def avg_T_W_Th_F := (T + W + Th + F) / 4 = 46
def temp_Monday := M = 42

theorem temperature_on_Friday
  (h1 : avg_M_T_W_Th M T W Th)
  (h2 : avg_T_W_Th_F T W Th F) 
  (h3 : temp_Monday M) : F = 34 := by
  sorry

end temperature_on_Friday_l184_184628


namespace find_y_z_l184_184463

theorem find_y_z 
  (y z : ℝ) 
  (h_mean : (8 + 15 + 22 + 5 + y + z) / 6 = 12) 
  (h_diff : y - z = 6) : 
  y = 14 ∧ z = 8 := 
by
  sorry

end find_y_z_l184_184463


namespace total_area_of_removed_triangles_l184_184433

theorem total_area_of_removed_triangles (a b : ℝ)
  (square_side : ℝ := 16)
  (triangle_hypotenuse : ℝ := 8)
  (isosceles_right_triangle : a = b ∧ a^2 + b^2 = triangle_hypotenuse^2) :
  4 * (1 / 2 * a * b) = 64 :=
by
  -- Sketch of the proof:
  -- From the isosceles right triangle property and Pythagorean theorem,
  -- a^2 + b^2 = 8^2 ⇒ 2 * a^2 = 64 ⇒ a^2 = 32 ⇒ a = b = 4√2
  -- The area of one triangle is (1/2) * a * b = 16
  -- Total area of four such triangles is 4 * 16 = 64
  sorry

end total_area_of_removed_triangles_l184_184433


namespace number_of_integer_chords_through_point_l184_184216

theorem number_of_integer_chords_through_point {r : ℝ} {c : ℝ} 
    (hr: r = 13) (hc : c = 12) : 
    ∃ n : ℕ, n = 17 :=
by
  -- Suppose O is the center and P is a point inside the circle such that OP = 12
  -- Given radius r = 13, we need to show there are 17 different integer chord lengths
  sorry  -- Proof is omitted

end number_of_integer_chords_through_point_l184_184216


namespace perimeter_of_figure_l184_184440

def side_length : ℕ := 1
def num_vertical_stacks : ℕ := 2
def num_squares_per_stack : ℕ := 3
def gap_between_stacks : ℕ := 1
def squares_on_top : ℕ := 3
def squares_on_bottom : ℕ := 2

theorem perimeter_of_figure : 
  (2 * side_length * squares_on_top) + (2 * side_length * squares_on_bottom) + 
  (2 * num_squares_per_stack * num_vertical_stacks) + (2 * num_squares_per_stack * squares_on_top)
  = 22 :=
by
  sorry

end perimeter_of_figure_l184_184440


namespace five_letter_words_with_vowels_l184_184888

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l184_184888


namespace roots_of_quadratic_eq_l184_184791

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end roots_of_quadratic_eq_l184_184791


namespace bob_selling_price_per_muffin_l184_184833

variable (dozen_muffins_per_day : ℕ := 12)
variable (cost_per_muffin : ℝ := 0.75)
variable (weekly_profit : ℝ := 63)
variable (days_per_week : ℕ := 7)

theorem bob_selling_price_per_muffin : 
  let daily_cost := dozen_muffins_per_day * cost_per_muffin
  let weekly_cost := daily_cost * days_per_week
  let weekly_revenue := weekly_profit + weekly_cost
  let muffins_per_week := dozen_muffins_per_day * days_per_week
  let selling_price_per_muffin := weekly_revenue / muffins_per_week
  selling_price_per_muffin = 1.50 := 
by
  sorry

end bob_selling_price_per_muffin_l184_184833


namespace circle_trajectory_l184_184458

theorem circle_trajectory (x y : ℝ) (h1 : (x-5)^2 + (y+7)^2 = 16) (h2 : ∃ c : ℝ, c = ((x + 1 - 5)^2 + (y + 1 + 7)^2)): 
    ((x-5)^2+(y+7)^2 = 25 ∨ (x-5)^2+(y+7)^2 = 9) :=
by
  -- Proof is omitted
  sorry

end circle_trajectory_l184_184458


namespace equidistant_line_existence_l184_184567

def line_passing_through_intersection_and_equidistant (A B : Point) : Prop :=
  ∃ l : Line, 
  (A = ⟨-3, 1⟩ ∧ B = ⟨5, 7⟩) ∧
  (∃ x y : ℝ, 
    (2 * x + 7 * y - 4 = 0) ∧
    (7 * x - 21 * y - 1 = 0) ∧
    ((l = ⟨21 * x - 28 * y - 13, 0⟩ ∨ l = ⟨x, 1⟩) ∧ l.passes_through ⟨x, y⟩) ∧
    l.is_equidistant_from ⟨-3, 1⟩ ⟨5, 7⟩)

theorem equidistant_line_existence : ∃ l : line,
  line_passing_through_intersection_and_equidistant ⟨-3, 1⟩ ⟨5, 7⟩ := sorry

end equidistant_line_existence_l184_184567


namespace mike_earnings_first_job_l184_184367

def total_earnings := 160
def hours_second_job := 12
def hourly_wage_second_job := 9
def earnings_second_job := hours_second_job * hourly_wage_second_job
def earnings_first_job := total_earnings - earnings_second_job

theorem mike_earnings_first_job : 
  earnings_first_job = 160 - (12 * 9) := by
  -- omitted proof
  sorry

end mike_earnings_first_job_l184_184367


namespace eval_sum_l184_184242

theorem eval_sum : 333 + 33 + 3 = 369 :=
by
  sorry

end eval_sum_l184_184242


namespace conclusion1_conclusion2_l184_184123

theorem conclusion1 (x y a b : ℝ) (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b :=
sorry

theorem conclusion2 (x a : ℝ) (h1 : (x-1)*(x^2 + a*x + 1) - x^2 = x^3 - (a-1)*x^2 - (1-a)*x - 1) : a = 1 :=
sorry

end conclusion1_conclusion2_l184_184123


namespace sqrt_four_eq_two_or_neg_two_l184_184636

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 → (x = 2 ∨ x = -2) :=
sorry

end sqrt_four_eq_two_or_neg_two_l184_184636


namespace fries_remaining_time_l184_184832

def recommendedTime : ℕ := 5 * 60
def timeInOven : ℕ := 45
def remainingTime : ℕ := recommendedTime - timeInOven

theorem fries_remaining_time : remainingTime = 255 :=
by
  sorry

end fries_remaining_time_l184_184832


namespace inequality_proof_l184_184223

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) : 
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) :=
sorry

end inequality_proof_l184_184223


namespace number_of_integers_satisfying_inequality_l184_184153

theorem number_of_integers_satisfying_inequality :
  ∃ S : Finset ℤ, (∀ x ∈ S, x^2 < 9 * x) ∧ S.card = 8 :=
by
  sorry

end number_of_integers_satisfying_inequality_l184_184153


namespace percentage_increase_l184_184423

theorem percentage_increase (original final : ℝ) (h1 : original = 90) (h2 : final = 135) : ((final - original) / original) * 100 = 50 := 
by
  sorry

end percentage_increase_l184_184423


namespace men_in_first_group_l184_184340

-- Define the problem conditions
def men_and_boys_work (x m b : ℕ) :=
  let w := 10 * (x * m + 8 * b) in
  w = 2 * (26 * m + 48 * b) ∧
  4 * (15 * m + 20 * b) = w

-- Formal statement of the problem
theorem men_in_first_group (x m b : ℕ) :
  men_and_boys_work x m b → x = 6 :=
sorry

end men_in_first_group_l184_184340


namespace smallest_square_number_l184_184413

theorem smallest_square_number (x y : ℕ) (hx : ∃ a, x = a ^ 2) (hy : ∃ b, y = b ^ 3) 
  (h_simp: ∃ c d, x / (y ^ 3) = c ^ 3 / d ^ 2 ∧ c > 1 ∧ d > 1): x = 64 := by
  sorry

end smallest_square_number_l184_184413


namespace solutions_exist_iff_l184_184685

variable (a b : ℝ)

theorem solutions_exist_iff :
  (∃ x y : ℝ, (x^2 + y^2 + xy = a) ∧ (x^2 - y^2 = b)) ↔ (-2 * a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2 * a) :=
sorry

end solutions_exist_iff_l184_184685


namespace negation_of_universal_proposition_l184_184787

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
  sorry

end negation_of_universal_proposition_l184_184787


namespace raft_minimum_capacity_l184_184986

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l184_184986


namespace unique_solution_xp_eq_1_l184_184498

theorem unique_solution_xp_eq_1 (x p q : ℕ) (h1 : x ≥ 2) (h2 : p ≥ 2) (h3 : q ≥ 2):
  ((x + 1)^p - x^q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by 
  sorry

end unique_solution_xp_eq_1_l184_184498


namespace number_of_music_files_l184_184124

-- The conditions given in the problem
variable {M : ℕ} -- M is a natural number representing the initial number of music files

-- Conditions: Initial state and changes
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23
def remaining_files : ℕ := 2

-- Statement of the theorem
theorem number_of_music_files (h : M + initial_video_files - files_deleted = remaining_files) : M = 4 :=
  by
  -- Proof goes here
  sorry

end number_of_music_files_l184_184124


namespace case1_BL_case2_BL_l184_184799

variable (AD BD BL AL : ℝ)

theorem case1_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 3)
  (h₃ : AB = 6 * Real.sqrt 13)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 2 * AL)
  : BL = 16 * Real.sqrt 3 - 12 := by
  sorry

theorem case2_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 6)
  (h₃ : AB = 30)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 4 * AL)
  : BL = (16 * Real.sqrt 6 - 6) / 5 := by
  sorry

end case1_BL_case2_BL_l184_184799


namespace new_elephants_entry_rate_l184_184409

-- Definitions
def initial_elephants := 30000
def exodus_rate := 2880
def exodus_duration := 4
def final_elephants := 28980
def new_elephants_duration := 7

-- Prove that the rate of new elephants entering the park is 1500 elephants per hour
theorem new_elephants_entry_rate :
  let elephants_left_after_exodus := initial_elephants - exodus_rate * exodus_duration
  let new_elephants := final_elephants - elephants_left_after_exodus
  let new_entry_rate := new_elephants / new_elephants_duration
  new_entry_rate = 1500 :=
by
  sorry

end new_elephants_entry_rate_l184_184409


namespace Jim_paycheck_correct_l184_184483

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end Jim_paycheck_correct_l184_184483


namespace value_of_a_l184_184182

theorem value_of_a (a x : ℝ) (h1 : x = 2) (h2 : a * x = 4) : a = 2 :=
by
  sorry

end value_of_a_l184_184182


namespace find_c_for_circle_radius_5_l184_184701

theorem find_c_for_circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 + 8 * y + c = 0 
    → x^2 + 4 * x + y^2 + 8 * y = 5^2 - 25) 
  → c = -5 :=
by
  sorry

end find_c_for_circle_radius_5_l184_184701


namespace quadruple_exists_unique_l184_184699

def digits (x : Nat) : Prop := x ≤ 9

theorem quadruple_exists_unique :
  ∃ (A B C D: Nat),
    digits A ∧ digits B ∧ digits C ∧ digits D ∧
    A > B ∧ B > C ∧ C > D ∧
    (A * 1000 + B * 100 + C * 10 + D) -
    (D * 1000 + C * 100 + B * 10 + A) =
    (B * 1000 + D * 100 + A * 10 + C) ∧
    (A, B, C, D) = (7, 6, 4, 1) :=
by
  sorry

end quadruple_exists_unique_l184_184699


namespace range_of_f_l184_184585

noncomputable def f (x : ℤ) : ℤ := x ^ 2 + 1

def domain : Set ℤ := {-1, 0, 1, 2}

def range_f : Set ℤ := {1, 2, 5}

theorem range_of_f : Set.image f domain = range_f :=
by
  sorry

end range_of_f_l184_184585


namespace total_coins_is_16_l184_184645

theorem total_coins_is_16 (x y : ℕ) (h₁ : x ≠ y) (h₂ : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := 
sorry

end total_coins_is_16_l184_184645


namespace center_circle_sum_eq_neg1_l184_184296

theorem center_circle_sum_eq_neg1 
  (h k : ℝ) 
  (h_center : ∀ x y, (x - h)^2 + (y - k)^2 = 22) 
  (circle_eq : ∀ x y, x^2 + y^2 = 4*x - 6*y + 9) : 
  h + k = -1 := 
by 
  sorry

end center_circle_sum_eq_neg1_l184_184296


namespace tomatoes_picked_yesterday_l184_184544

-- Definitions corresponding to the conditions in the problem.
def initial_tomatoes : Nat := 160
def tomatoes_left_after_yesterday : Nat := 104

-- Statement of the problem proving the number of tomatoes picked yesterday.
theorem tomatoes_picked_yesterday : initial_tomatoes - tomatoes_left_after_yesterday = 56 :=
by
  sorry

end tomatoes_picked_yesterday_l184_184544


namespace translate_line_upwards_l184_184798

theorem translate_line_upwards {x y : ℝ} (h : y = -2 * x + 1) :
  y = -2 * x + 3 := by
  sorry

end translate_line_upwards_l184_184798


namespace minimum_distance_from_parabola_to_circle_l184_184865

noncomputable def minimum_distance_sum : ℝ :=
  let focus : ℝ × ℝ := (1, 0)
  let center : ℝ × ℝ := (0, 4)
  let radius : ℝ := 1
  let distance_from_focus_to_center : ℝ := Real.sqrt ((focus.1 - center.1)^2 + (focus.2 - center.2)^2)
  distance_from_focus_to_center - radius

theorem minimum_distance_from_parabola_to_circle : minimum_distance_sum = Real.sqrt 17 - 1 := by
  sorry

end minimum_distance_from_parabola_to_circle_l184_184865


namespace value_of_x2y_plus_xy2_l184_184336

-- Define variables x and y as real numbers
variables (x y : ℝ)

-- Define the conditions
def condition1 : Prop := x + y = -2
def condition2 : Prop := x * y = -3

-- Define the proof problem
theorem value_of_x2y_plus_xy2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 * y + x * y^2 = 6 := by
  sorry

end value_of_x2y_plus_xy2_l184_184336


namespace evaluate_expression_when_c_is_4_l184_184142

variable (c : ℕ)

theorem evaluate_expression_when_c_is_4 : (c = 4) → ((c^2 - c! * (c - 1)^c)^2 = 3715584) :=
by
  -- This is where the proof would go, but we only need to set up the statement.
  sorry

end evaluate_expression_when_c_is_4_l184_184142


namespace gcd_divisor_l184_184359

theorem gcd_divisor (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) (hrs : Nat.gcd r s = 60) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) 
  : 13 ∣ p :=
sorry

end gcd_divisor_l184_184359


namespace fraction_computation_l184_184134

theorem fraction_computation :
  ((11^4 + 324) * (23^4 + 324) * (35^4 + 324) * (47^4 + 324) * (59^4 + 324)) / 
  ((5^4 + 324) * (17^4 + 324) * (29^4 + 324) * (41^4 + 324) * (53^4 + 324)) = 295.615 := 
sorry

end fraction_computation_l184_184134


namespace total_area_rectangle_l184_184980

theorem total_area_rectangle (BF CF : ℕ) (A1 A2 x : ℕ) (h1 : BF = 3 * CF) (h2 : A1 = 3 * A2) (h3 : 2 * x = 96) (h4 : 48 = x) (h5 : A1 = 3 * 48) (h6 : A2 = 48) : A1 + A2 = 192 :=
  by sorry

end total_area_rectangle_l184_184980


namespace amount_C_l184_184997

theorem amount_C (A B C : ℕ) 
  (h₁ : A + B + C = 900) 
  (h₂ : A + C = 400) 
  (h₃ : B + C = 750) : 
  C = 250 :=
sorry

end amount_C_l184_184997


namespace right_triangle_acute_angles_l184_184960

variable (α β : ℝ)

noncomputable def prove_acute_angles (α β : ℝ) : Prop :=
  α + β = 90 ∧ 4 * α = 90

theorem right_triangle_acute_angles : 
  prove_acute_angles α β → α = 22.5 ∧ β = 67.5 := by
  sorry

end right_triangle_acute_angles_l184_184960


namespace calculate_expression_l184_184130

theorem calculate_expression :
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by sorry

end calculate_expression_l184_184130


namespace bananas_eaten_l184_184670

variable (initial_bananas : ℕ) (remaining_bananas : ℕ)

theorem bananas_eaten (initial_bananas remaining_bananas : ℕ) (h_initial : initial_bananas = 12) (h_remaining : remaining_bananas = 10) : initial_bananas - remaining_bananas = 2 := by
  -- Proof goes here
  sorry

end bananas_eaten_l184_184670


namespace algebraic_expression_value_l184_184333

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : 2 * a^2 - 4 * a + 2022 = 2024 := 
by 
  sorry

end algebraic_expression_value_l184_184333


namespace Emily_average_speed_l184_184141

noncomputable def Emily_run_distance : ℝ := 10

noncomputable def speed_first_uphill : ℝ := 4
noncomputable def distance_first_uphill : ℝ := 2

noncomputable def speed_first_downhill : ℝ := 6
noncomputable def distance_first_downhill : ℝ := 1

noncomputable def speed_flat_ground : ℝ := 5
noncomputable def distance_flat_ground : ℝ := 3

noncomputable def speed_second_uphill : ℝ := 4.5
noncomputable def distance_second_uphill : ℝ := 2

noncomputable def speed_second_downhill : ℝ := 6
noncomputable def distance_second_downhill : ℝ := 2

noncomputable def break_first : ℝ := 5 / 60
noncomputable def break_second : ℝ := 7 / 60
noncomputable def break_third : ℝ := 3 / 60

noncomputable def time_first_uphill : ℝ := distance_first_uphill / speed_first_uphill
noncomputable def time_first_downhill : ℝ := distance_first_downhill / speed_first_downhill
noncomputable def time_flat_ground : ℝ := distance_flat_ground / speed_flat_ground
noncomputable def time_second_uphill : ℝ := distance_second_uphill / speed_second_uphill
noncomputable def time_second_downhill : ℝ := distance_second_downhill / speed_second_downhill

noncomputable def total_running_time : ℝ := time_first_uphill + time_first_downhill + time_flat_ground + time_second_uphill + time_second_downhill
noncomputable def total_break_time : ℝ := break_first + break_second + break_third
noncomputable def total_time : ℝ := total_running_time + total_break_time

noncomputable def average_speed : ℝ := Emily_run_distance / total_time

theorem Emily_average_speed : abs (average_speed - 4.36) < 0.01 := by
  sorry

end Emily_average_speed_l184_184141


namespace sequence_term_306_l184_184599

theorem sequence_term_306 (a1 a2 : ℤ) (r : ℤ) (n : ℕ) (h1 : a1 = 7) (h2 : a2 = -7) (h3 : r = -1) (h4 : a2 = r * a1) : 
  ∃ a306 : ℤ, a306 = -7 ∧ a306 = a1 * r^305 :=
by
  use -7
  sorry

end sequence_term_306_l184_184599


namespace total_families_l184_184032

theorem total_families (F_2dogs F_1dog F_2cats total_animals total_families : ℕ) 
  (h1: F_2dogs = 15)
  (h2: F_1dog = 20)
  (h3: total_animals = 80)
  (h4: 2 * F_2dogs + F_1dog + 2 * F_2cats = total_animals) :
  total_families = F_2dogs + F_1dog + F_2cats := 
by 
  sorry

end total_families_l184_184032


namespace smallest_multiple_of_18_and_40_l184_184522

-- Define the conditions
def multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def multiple_of_40 (n : ℕ) : Prop := n % 40 = 0

-- Prove that the smallest number that meets the conditions is 360
theorem smallest_multiple_of_18_and_40 : ∃ n : ℕ, multiple_of_18 n ∧ multiple_of_40 n ∧ ∀ m : ℕ, (multiple_of_18 m ∧ multiple_of_40 m) → n ≤ m :=
  by
    let n := 360
    -- We have to prove that 360 is the smallest number that is a multiple of both 18 and 40
    sorry

end smallest_multiple_of_18_and_40_l184_184522


namespace hyperbola_center_coordinates_l184_184302

theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), 
  (∀ x y : ℝ, 
    ((4 * y - 6) ^ 2 / 36 - (5 * x - 3) ^ 2 / 49 = -1) ↔
    ((x - h) ^ 2 / ((7 / 5) ^ 2) - (y - k) ^ 2 / ((3 / 2) ^ 2) = 1)) ∧
  h = 3 / 5 ∧ k = 3 / 2 :=
by sorry

end hyperbola_center_coordinates_l184_184302


namespace num_factors_x_l184_184029

theorem num_factors_x (x : ℕ) (h : 2011^(2011^2012) = x^x) : ∃ n : ℕ, n = 2012 ∧  ∀ d : ℕ, d ∣ x -> d ≤ n :=
sorry

end num_factors_x_l184_184029


namespace answer_key_combinations_l184_184477

theorem answer_key_combinations : 
  (2^3 - 2) * 4^2 = 96 := 
by 
  -- Explanation about why it equals to this multi-step skipped, directly written as sorry.
  sorry

end answer_key_combinations_l184_184477


namespace necessary_condition_l184_184431

theorem necessary_condition :
  ∃ x : ℝ, (x < 0 ∨ x > 2) → (2 * x^2 - 5 * x - 3 ≥ 0) :=
sorry

end necessary_condition_l184_184431


namespace tennis_tournament_total_rounds_l184_184971

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end tennis_tournament_total_rounds_l184_184971


namespace stateA_selection_percentage_l184_184345

theorem stateA_selection_percentage :
  ∀ (P : ℕ), (∀ (n : ℕ), n = 8000) → (7 * 8000 / 100 = P * 8000 / 100 + 80) → P = 6 := by
  -- The proof steps go here
  sorry

end stateA_selection_percentage_l184_184345


namespace yellow_marbles_count_l184_184485

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end yellow_marbles_count_l184_184485


namespace area_of_triangle_ABC_circumcenter_of_triangle_ABC_l184_184711

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨2, 1⟩
def B : Point := ⟨4, 7⟩
def C : Point := ⟨8, 3⟩

def triangle_area (A B C : Point) : ℚ := by
  -- area calculation will be filled here
  sorry

def circumcenter (A B C : Point) : Point := by
  -- circumcenter calculation will be filled here
  sorry

theorem area_of_triangle_ABC : triangle_area A B C = 16 :=
  sorry

theorem circumcenter_of_triangle_ABC : circumcenter A B C = ⟨9/2, 7/2⟩ :=
  sorry

end area_of_triangle_ABC_circumcenter_of_triangle_ABC_l184_184711


namespace angle_between_hour_and_minute_hand_at_3_40_l184_184555

def angle_between_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (360 / 60) * minute
  let hour_angle := (360 / 12) + (30 / 60) * minute
  abs (minute_angle - hour_angle)

theorem angle_between_hour_and_minute_hand_at_3_40 : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_between_hour_and_minute_hand_at_3_40_l184_184555


namespace new_average_after_17th_l184_184424

def old_average (A : ℕ) (n : ℕ) : ℕ :=
  A -- A is the average before the 17th inning

def runs_in_17th : ℕ := 84 -- The score in the 17th inning

def average_increase : ℕ := 3 -- The increase in average after the 17th inning

theorem new_average_after_17th (A : ℕ) (n : ℕ) (h1 : n = 16) (h2 : old_average A n + average_increase = A + 3) :
  (old_average A n) + average_increase = 36 :=
by
  sorry

end new_average_after_17th_l184_184424


namespace integer_square_root_35_consecutive_l184_184907

theorem integer_square_root_35_consecutive : 
  ∃ n : ℕ, ∀ k : ℕ, n^2 ≤ k ∧ k < (n+1)^2 ∧ ((n + 1)^2 - n^2 = 35) ∧ (n = 17) := by 
  sorry

end integer_square_root_35_consecutive_l184_184907


namespace neg_p_is_exists_x_l184_184175

variable (x : ℝ)

def p : Prop := ∀ x, x^2 + x + 1 ≠ 0

theorem neg_p_is_exists_x : ¬ p ↔ ∃ x, x^2 + x + 1 = 0 := by
  sorry

end neg_p_is_exists_x_l184_184175


namespace mul_digits_example_l184_184600

theorem mul_digits_example (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : C = 2) (h8 : D = 5) : A + B = 2 := by
  sorry

end mul_digits_example_l184_184600


namespace inequality_proof_l184_184372

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9 / 16 :=
sorry

end inequality_proof_l184_184372


namespace no_solution_fraction_eq_l184_184571

theorem no_solution_fraction_eq {x m : ℝ} : 
  (∀ x, ¬ (1 - x = 0) → (2 - x) / (1 - x) = (m + x) / (1 - x) + 1) ↔ m = 0 := 
by
  sorry

end no_solution_fraction_eq_l184_184571


namespace order_large_pizzas_sufficient_l184_184831

def pizza_satisfaction (gluten_free_slices_per_large : ℕ) (medium_slices : ℕ) (small_slices : ℕ) 
                       (gluten_free_needed : ℕ) (dairy_free_needed : ℕ) :=
  let slices_gluten_free := small_slices
  let slices_dairy_free := 2 * medium_slices
  (slices_gluten_free < gluten_free_needed) → 
  let additional_slices_gluten_free := gluten_free_needed - slices_gluten_free
  let large_pizzas_gluten_free := (additional_slices_gluten_free + gluten_free_slices_per_large - 1) / gluten_free_slices_per_large
  large_pizzas_gluten_free = 1

theorem order_large_pizzas_sufficient :
  pizza_satisfaction 14 10 8 15 15 :=
by
  unfold pizza_satisfaction
  sorry

end order_large_pizzas_sufficient_l184_184831


namespace find_range_of_m_l184_184704

def equation1 (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0

def equation2 (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 → false

theorem find_range_of_m (m : ℝ) (h1 : equation1 m → m > 2) (h2 : equation2 m → 1 < m ∧ m < 3) :
  (equation1 m ∨ equation2 m) ∧ ¬(equation1 m ∧ equation2 m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end find_range_of_m_l184_184704


namespace cos_pi_minus_2alpha_eq_seven_over_twentyfive_l184_184856

variable (α : ℝ)

theorem cos_pi_minus_2alpha_eq_seven_over_twentyfive 
  (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π - 2 * α) = 7 / 25 := 
by
  sorry

end cos_pi_minus_2alpha_eq_seven_over_twentyfive_l184_184856


namespace maximum_distance_area_of_ring_l184_184247

def num_radars : ℕ := 9
def radar_radius : ℝ := 37
def ring_width : ℝ := 24

theorem maximum_distance (θ : ℝ) (hθ : θ = 20) 
  : (∀ d, d = radar_radius * (ring_width / 2 / (radar_radius^2 - (ring_width / 2)^2).sqrt)) →
    ( ∀ dist_from_center, dist_from_center = radar_radius / θ.sin) :=
sorry

theorem area_of_ring (θ : ℝ) (hθ : θ = 20) 
  : (∀ a, a = π * (ring_width * radar_radius * 2 / θ.tan)) →
    ( ∀ area, area = 1680 * π / θ.tan) :=
sorry

end maximum_distance_area_of_ring_l184_184247


namespace mean_goals_is_correct_l184_184190

theorem mean_goals_is_correct :
  let goals5 := 5
  let players5 := 4
  let goals6 := 6
  let players6 := 3
  let goals7 := 7
  let players7 := 2
  let goals8 := 8
  let players8 := 1
  let total_goals := goals5 * players5 + goals6 * players6 + goals7 * players7 + goals8 * players8
  let total_players := players5 + players6 + players7 + players8
  (total_goals / total_players : ℝ) = 6 :=
by
  -- The proof is omitted.
  sorry

end mean_goals_is_correct_l184_184190


namespace correct_product_l184_184198

theorem correct_product : 0.125 * 5.12 = 0.64 := sorry

end correct_product_l184_184198


namespace intersection_A_B_l184_184864

variable (x : ℝ)

def setA : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersection_A_B :
  { x | x^2 - 4*x - 5 < 0 } ∩ { x | -2 < x ∧ x < 2 } = { x | -1 < x ∧ x < 2 } :=
by
  -- Here would be the proof, but we use sorry to skip it
  sorry

end intersection_A_B_l184_184864


namespace minimum_k_conditions_l184_184805

theorem minimum_k_conditions (k : ℝ) :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k)) ↔ k = 3/2 :=
sorry

end minimum_k_conditions_l184_184805


namespace unstuck_rectangle_min_perimeter_l184_184546

open Real

/--
A rectangle that is inscribed in a larger rectangle (with one vertex on each side) is called unstuck if it is possible to rotate (however slightly) the smaller rectangle about its center within the confines of the larger. 
Of all the rectangles that can be inscribed unstuck in a 6 by 8 rectangle, the smallest perimeter has the form sqrt(N), for a positive integer N.
Prove that the smallest such N is 448.
-/
theorem unstuck_rectangle_min_perimeter :
  ∃ N : ℕ, (∃ P : ℝ, P = sqrt (N : ℝ)) ∧ (N = 448) :=
sorry

end unstuck_rectangle_min_perimeter_l184_184546


namespace stable_number_divisible_by_11_l184_184998

/-- Definition of a stable number as a three-digit number (cen, ten, uni) where
    each digit is non-zero, and the sum of any two digits is greater than the remaining digit.
-/
def is_stable_number (cen ten uni : ℕ) : Prop :=
cen ≠ 0 ∧ ten ≠ 0 ∧ uni ≠ 0 ∧
(cen + ten > uni) ∧ (cen + uni > ten) ∧ (ten + uni > cen)

/-- Function F defined for a stable number (cen ten uni). -/
def F (cen ten uni : ℕ) : ℕ := 10 * ten + cen + uni

/-- Function Q defined for a stable number (cen ten uni). -/
def Q (cen ten uni : ℕ) : ℕ := 10 * cen + ten + uni

/-- Statement to prove: Given a stable number s = 100a + 101b + 30 where 1 ≤ a ≤ 5 and 1 ≤ b ≤ 4,
    the expression 5 * F(s) + 2 * Q(s) is divisible by 11.
-/
theorem stable_number_divisible_by_11 (a b cen ten uni : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 5)
  (h_b : 1 ≤ b ∧ b ≤ 4)
  (h_s : 100 * a + 101 * b + 30 = 100 * cen + 10 * ten + uni)
  (h_stable : is_stable_number cen ten uni) :
  (5 * F cen ten uni + 2 * Q cen ten uni) % 11 = 0 :=
sorry

end stable_number_divisible_by_11_l184_184998


namespace opposite_signs_abs_larger_l184_184009

theorem opposite_signs_abs_larger (a b : ℝ) (h1 : a + b < 0) (h2 : a * b < 0) :
  (a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |b| > |a|) :=
sorry

end opposite_signs_abs_larger_l184_184009


namespace diameter_expression_l184_184078

noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (r ^ 3)

noncomputable def find_diameter (r1 : ℝ) : ℝ :=
  let volume1 := sphereVolume r1
  let volume2 := 3 * volume1
  let r2 := Real.cbrt (volume2 * 3 / (4 * Real.pi)) -- Solving for r in sphereVolume formula
  2 * r2   -- The diameter is twice the radius

theorem diameter_expression : 
  ∃ (a b : ℕ), b % 5 ≠ 0 ∧ find_diameter 6 = a * Real.cbrt b ∧ a + b = 18 :=
by {
  use [12, 6], -- Providing specific values for a and b
  split, {
    exact dec_trivial,   -- Proving b doesn't have perfect cube factors
  },
  split,
  {   
    have h1 : find_diameter (6 : ℝ) = 2 * Real.cbrt (2592 * 3 / 4),
    sorry,
    have h2 : 2 * Real.cbrt (2592 * 3 / 4) = 12 * Real.cbrt 6,
    sorry
  },
  {
    exact dec_trivial,  -- Directly proving a + b = 18 as a = 12 and b= 6
  }
}

end diameter_expression_l184_184078


namespace min_value_3x_plus_4y_l184_184184

variable (x y : ℝ)

theorem min_value_3x_plus_4y (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_plus_4y_l184_184184


namespace transformed_parabola_l184_184383

theorem transformed_parabola (x : ℝ) : 
  (λ x => -x^2 + 1) (x - 2) - 2 = - (x - 2)^2 - 1 := 
by 
  sorry 

end transformed_parabola_l184_184383


namespace calc_triangle_PQR_area_l184_184511

-- Given:
-- 1. Triangle ABC is a right triangle with right angle at A
-- 2. R is the midpoint of the hypotenuse BC
-- 3. Point P on AB such that CP = BP
-- 4. Point Q on BP such that triangle PQR is equilateral
-- 5. The area of triangle ABC is 27

theorem calc_triangle_PQR_area (A B C P Q R : ℝ^2) (hABC : A ≠ B) (hA90 : angle_fixed_90 A B C)
  (hR_midpoint : midpoint R B C) (hP_on_AB : P_is_on_AB P A B) 
  (hCP_eq_BP : distance C P = distance B P) (hQ_on_BP : Q_is_on_BP Q B P)
  (hPQR_eq_eq_tri : Equilateral_PQR P Q R) (h_area_ABC : area_triangle A B C = 27) :
  area_triangle P Q R = 9 / 2 :=
sorry

end calc_triangle_PQR_area_l184_184511


namespace intersection_points_of_graphs_l184_184782

open Real

theorem intersection_points_of_graphs (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃! x : ℝ, (f (x^3) = f (x^6)) ∧ (x = -1 ∨ x = 0 ∨ x = 1) :=
by
  -- Provide the structure of the proof
  sorry

end intersection_points_of_graphs_l184_184782


namespace length_of_platform_is_correct_l184_184529

noncomputable def length_of_platform : ℝ :=
  let train_length := 200 -- in meters
  let train_speed := 80 * 1000 / 3600 -- kmph to m/s
  let crossing_time := 22 -- in seconds
  (train_speed * crossing_time) - train_length

theorem length_of_platform_is_correct :
  length_of_platform = 2600 / 9 :=
by 
  -- proof would go here
  sorry

end length_of_platform_is_correct_l184_184529


namespace PQ_relationship_l184_184474

-- Define the sets P and Q
def P := {x : ℝ | x >= 5}
def Q := {x : ℝ | 5 <= x ∧ x <= 7}

-- Statement to be proved
theorem PQ_relationship : Q ⊆ P ∧ Q ≠ P :=
by
  sorry

end PQ_relationship_l184_184474


namespace non_neg_sum_of_squares_l184_184707

theorem non_neg_sum_of_squares (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (h : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
by
  sorry

end non_neg_sum_of_squares_l184_184707


namespace necklace_wire_length_l184_184040

theorem necklace_wire_length
  (spools : ℕ)
  (feet_per_spool : ℕ)
  (total_necklaces : ℕ)
  (h1 : spools = 3)
  (h2 : feet_per_spool = 20)
  (h3 : total_necklaces = 15) :
  (spools * feet_per_spool) / total_necklaces = 4 := by
  sorry

end necklace_wire_length_l184_184040


namespace treaty_of_versailles_original_day_l184_184783

-- Define the problem in Lean terms
def treatySignedDay : Nat -> Nat -> String
| 1919, 6 => "Saturday"
| _, _ => "Unknown"

-- Theorem statement
theorem treaty_of_versailles_original_day :
  treatySignedDay 1919 6 = "Saturday" :=
sorry

end treaty_of_versailles_original_day_l184_184783


namespace min_value_ineq_min_value_attainable_l184_184356

theorem min_value_ineq
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_sum : a + b + c = 9) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 :=
by {
  sorry,
}

theorem min_value_attainable :
  (a b c : ℝ)
  (h_eq : a = 3 ∧ b = 3 ∧ c = 3) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) = 9 :=
by {
  sorry,
}

end min_value_ineq_min_value_attainable_l184_184356


namespace relationship_among_a_b_c_l184_184579

noncomputable def a := Real.log 2 / 2
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 5 / 5

theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l184_184579


namespace range_of_a_opposite_sides_l184_184866

theorem range_of_a_opposite_sides (a : ℝ) :
  (3 * (-2) - 2 * 1 - a) * (3 * 1 - 2 * 1 - a) < 0 ↔ -8 < a ∧ a < 1 := by
  sorry

end range_of_a_opposite_sides_l184_184866


namespace find_profits_maximize_profit_week3_l184_184930

-- Defining the conditions of the problems
def week1_sales_A := 10
def week1_sales_B := 12
def week1_profit := 2000

def week2_sales_A := 20
def week2_sales_B := 15
def week2_profit := 3100

def total_sales_week3 := 25

-- Condition: Sales of type B exceed sales of type A but do not exceed twice the sales of type A
def sales_condition (x : ℕ) := (total_sales_week3 - x) > x ∧ (total_sales_week3 - x) ≤ 2 * x

-- Define the profits for types A and B
def profit_A (a b : ℕ) := week1_sales_A * a + week1_sales_B * b = week1_profit
def profit_B (a b : ℕ) := week2_sales_A * a + week2_sales_B * b = week2_profit

-- Define the profit function for week 3
def profit_week3 (a b x : ℕ) := a * x + b * (total_sales_week3 - x)

theorem find_profits : ∃ a b, profit_A a b ∧ profit_B a b :=
by
  use 80, 100
  sorry

theorem maximize_profit_week3 : 
  ∃ x y, 
  sales_condition x ∧ 
  x + y = total_sales_week3 ∧ 
  profit_week3 80 100 x = 2320 :=
by
  use 9, 16
  sorry

end find_profits_maximize_profit_week3_l184_184930


namespace problem1_problem2_l184_184966

-- Problem 1: Prove the simplification of an expression
theorem problem1 (x : ℝ) : (2*x + 1)^2 + x*(x-4) = 5*x^2 + 1 := 
by sorry

-- Problem 2: Prove the solution set for the system of inequalities
theorem problem2 (x : ℝ) (h1 : 3*x - 6 > 0) (h2 : (5 - x) / 2 < 1) : x > 3 := 
by sorry

end problem1_problem2_l184_184966


namespace garden_perimeter_ratio_l184_184661

theorem garden_perimeter_ratio (side_length : ℕ) (tripled_side_length : ℕ) (original_perimeter : ℕ) (new_perimeter : ℕ) (ratio : ℚ) :
  side_length = 50 →
  tripled_side_length = 3 * side_length →
  original_perimeter = 4 * side_length →
  new_perimeter = 4 * tripled_side_length →
  ratio = original_perimeter / new_perimeter →
  ratio = 1 / 3 :=
by
  sorry

end garden_perimeter_ratio_l184_184661


namespace domain_of_f_l184_184080

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x + 1 > 0 ∧ x + 1 ≠ 1} = {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0} :=
by 
  sorry

end domain_of_f_l184_184080


namespace correct_proposition_l184_184166

-- Definitions of the propositions p and q
def p : Prop := ∀ x : ℝ, (x > 1 → x > 2)
def q : Prop := ∀ x y : ℝ, (x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1)

-- The proof problem statement
theorem correct_proposition : ¬p ∧ q :=
by
  -- Assuming p is false (i.e., ¬p is true) and q is true
  sorry

end correct_proposition_l184_184166


namespace max_C_trees_l184_184949

theorem max_C_trees 
  (price_A : ℕ) (price_B : ℕ) (price_C : ℕ) (total_price : ℕ) (total_trees : ℕ)
  (h_price_ratio : 2 * price_B = 2 * price_A ∧ 3 * price_A = 2 * price_C)
  (h_price_A : price_A = 200)
  (h_total_price : total_price = 220120)
  (h_total_trees : total_trees = 1000) :
  ∃ (num_C : ℕ), num_C = 201 ∧ ∀ num_C', num_C' > num_C → 
  total_price < price_A * (total_trees - num_C') + price_C * num_C' :=
by
  sorry

end max_C_trees_l184_184949


namespace dogwood_tree_cut_count_l184_184343

theorem dogwood_tree_cut_count
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_left : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0)
  (h3 : trees_left = 2.0) :
  trees_part1 + trees_part2 - trees_left = 7.0 :=
by
  sorry

end dogwood_tree_cut_count_l184_184343


namespace difference_of_squares_l184_184101

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 :=
by
  sorry

end difference_of_squares_l184_184101


namespace sphere_tangency_relation_l184_184948

noncomputable def sphere_tangents (r R : ℝ) (h : R > r) :=
  (R >= (2 / (Real.sqrt 3) - 1) * r) ∧
  (∃ x, x = (R * (R + r - Real.sqrt (R^2 + 2 * R * r - r^2 / 3))) /
            (r + Real.sqrt (R^2 + 2 * R * r - r^2 / 3) - R)) 

theorem sphere_tangency_relation (r R: ℝ) (h : R > r) :
  sphere_tangents r R h :=
by
  sorry

end sphere_tangency_relation_l184_184948


namespace largest_number_of_right_angles_in_convex_octagon_l184_184519

theorem largest_number_of_right_angles_in_convex_octagon : 
  ∀ (angles : Fin 8 → ℝ), 
  (∀ i, 0 < angles i ∧ angles i < 180) → 
  (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 + angles 7 = 1080) → 
  ∃ k, k ≤ 6 ∧ (∀ i < 8, if angles i = 90 then k = 6 else true) := 
by 
  sorry

end largest_number_of_right_angles_in_convex_octagon_l184_184519


namespace max_marks_is_400_l184_184434

-- Given conditions
def passing_mark (M : ℝ) : ℝ := 0.30 * M
def student_marks : ℝ := 80
def marks_failed_by : ℝ := 40
def pass_marks : ℝ := student_marks + marks_failed_by

-- Statement to prove
theorem max_marks_is_400 (M : ℝ) (h : passing_mark M = pass_marks) : M = 400 :=
by sorry

end max_marks_is_400_l184_184434


namespace total_birds_in_marsh_l184_184035

-- Given conditions
def initial_geese := 58
def doubled_geese := initial_geese * 2
def ducks := 37
def swans := 15
def herons := 22

-- Prove that the total number of birds is 190
theorem total_birds_in_marsh : 
  doubled_geese + ducks + swans + herons = 190 := 
by
  sorry

end total_birds_in_marsh_l184_184035


namespace prob_not_spade_on_first_draw_is_three_quarters_l184_184033

-- Define number of total cards and number of spades
def total_cards : ℕ := 52
def spades : ℕ := 13

-- Define the event of drawing a spade
def prob_spade : ℚ := spades / total_cards

-- Define the event of not drawing a spade on the first draw
def prob_not_spade_first_draw : ℚ := 1 - prob_spade

-- The theorem statement 
theorem prob_not_spade_on_first_draw_is_three_quarters :
  prob_not_spade_first_draw = 3 / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end prob_not_spade_on_first_draw_is_three_quarters_l184_184033


namespace monotonic_intervals_a_leq_0_monotonic_intervals_a_gt_0_ln_sum_gt_2_l184_184018

noncomputable def f (x : ℝ) := 2 * Real.log x

def g (a x : ℝ) := (1 / 2) * a * x^2 + (2 * a - 1) * x

def h (a x : ℝ) := f x - g a x

theorem monotonic_intervals_a_leq_0 (a : ℝ) (h₀ : a ≤ 0) :
  ∀ x > 0, Monotone (λ x, h a x) :=
sorry

theorem monotonic_intervals_a_gt_0 (a : ℝ) (h₀ : a > 0) :
  ∀ x > 0, 
    (MonotoneOn (λ x, h a x) (set.Ioo 0 (1 / a))) ∧
    (AntitoneOn (λ x, h a x) (set.Ioi (1 / a))) :=
sorry
  
theorem ln_sum_gt_2 {a x₁ x₂ : ℝ} (hx : x₁ ≠ x₂)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (h₁ : f x₁ - a * x₁ = 0)
  (h₂ : f x₂ - a * x₂ = 0) :
  Real.log x₁ + Real.log x₂ > 2 :=
sorry

end monotonic_intervals_a_leq_0_monotonic_intervals_a_gt_0_ln_sum_gt_2_l184_184018


namespace perpendicular_bisector_l184_184697

theorem perpendicular_bisector (x y : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (h_line : x - 2 * y + 1 = 0) : 
  2 * x - y - 1 = 0 :=
sorry

end perpendicular_bisector_l184_184697


namespace infinite_zeros_in_S_l184_184019

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n % 4 = 0 then -↑(n + 1) else
  if n % 4 = 1 then ↑n else
  if n % 4 = 2 then ↑n else
  -↑(n + 1)

-- Define the sequence S_k as partial sum of a_n
def S : ℕ → ℤ
| 0       => a 0
| (n + 1) => S n + a (n + 1)

-- Proposition: S_k contains infinitely many zeros
theorem infinite_zeros_in_S : ∀ n : ℕ, ∃ m > n, S m = 0 := sorry

end infinite_zeros_in_S_l184_184019


namespace total_practice_hours_l184_184620

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l184_184620


namespace isosceles_right_triangle_solution_l184_184534

theorem isosceles_right_triangle_solution (a b : ℝ) (area : ℝ) 
  (h1 : a = b) (h2 : XY = a * Real.sqrt 2) (h3 : area = (1/2) * a * b) (h4 : area = 36) : 
  XY = 12 :=
by
  sorry

end isosceles_right_triangle_solution_l184_184534


namespace num_int_values_satisfying_inequality_l184_184155

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l184_184155


namespace calculate_area_of_region_l184_184128

theorem calculate_area_of_region :
  let region := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = 12}
  ∃ area, area = 17 * Real.pi
:= by
  sorry

end calculate_area_of_region_l184_184128


namespace real_part_is_neg4_l184_184464

def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_is_neg4 (i : ℂ) (h : i^2 = -1) :
  real_part_of_z ((3 + 4 * i) * i) = -4 := by
  sorry

end real_part_is_neg4_l184_184464


namespace minimum_raft_weight_l184_184990

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l184_184990


namespace raft_minimum_capacity_l184_184996

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l184_184996


namespace tan_sin_30_computation_l184_184681

theorem tan_sin_30_computation :
  let θ := 30 * Real.pi / 180 in
  Real.tan θ + 4 * Real.sin θ = (Real.sqrt 3 + 6) / 3 :=
by
  let θ := 30 * Real.pi / 180
  have sin_30 : Real.sin θ = 1 / 2 := by sorry
  have cos_30 : Real.cos θ = Real.sqrt 3 / 2 := by sorry
  have tan_30 : Real.tan θ = Real.sin θ / Real.cos θ := by sorry
  have sin_60 : Real.sin (2 * θ) = Real.sqrt 3 / 2 := by sorry
  sorry

end tan_sin_30_computation_l184_184681


namespace avg_integer_N_between_fractions_l184_184518

theorem avg_integer_N_between_fractions (N : ℕ) (h1 : (2 : ℚ) / 5 < N / 42) (h2 : N / 42 < 1 / 3) : 
  N = 15 := 
by
  sorry

end avg_integer_N_between_fractions_l184_184518


namespace equilateral_triangle_hyperbola_area_square_l184_184794

noncomputable def equilateral_triangle_area_square :
  {A B C : ℝ × ℝ // 
    A.1 * A.2 = 4 ∧ 
    B.1 * B.2 = 4 ∧ 
    C.1 * C.2 = 4 ∧ 
    (A.1 + B.1 + C.1) / 3 = 1 ∧ 
    (A.2 + B.2 + C.2) / 3 = 1 ∧ 
    dist A B = dist B C ∧ 
    dist B C = dist C A} → 
    ℝ :=
λ ⟨A, B, C, hA, hB, hC, hcentroid_x, hcentroid_y, hdist1, hdist2⟩, (6 * real.sqrt 3) ^ 2

theorem equilateral_triangle_hyperbola_area_square :
  ∀ t : {A B C : ℝ × ℝ // 
    A.1 * A.2 = 4 ∧ 
    B.1 * B.2 = 4 ∧ 
    C.1 * C.2 = 4 ∧ 
    (A.1 + B.1 + C.1) / 3 = 1 ∧ 
    (A.2 + B.2 + C.2) / 3 = 1 ∧ 
    dist A B = dist B C ∧ 
    dist B C = dist C A}, 
  equilateral_triangle_area_square t = 108 :=
sorry

end equilateral_triangle_hyperbola_area_square_l184_184794


namespace max_n_value_l184_184011

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h_ineq : 1/(a - b) + 1/(b - c) ≥ n/(a - c)) : n ≤ 4 := 
sorry

end max_n_value_l184_184011


namespace scientific_notation_of_42_trillion_l184_184902

theorem scientific_notation_of_42_trillion : (42.1 * 10^12) = 4.21 * 10^13 :=
by
  sorry

end scientific_notation_of_42_trillion_l184_184902


namespace determinant_difference_l184_184322

namespace MatrixDeterminantProblem

open Matrix

variables {R : Type*} [CommRing R]

theorem determinant_difference (a b c d : R) 
  (h : det ![![a, b], ![c, d]] = 15) :
  det ![![3 * a, 3 * b], ![3 * c, 3 * d]] - 
  det ![![3 * b, 3 * a], ![3 * d, 3 * c]] = 270 := 
by
  sorry

end MatrixDeterminantProblem

end determinant_difference_l184_184322


namespace village_population_decrease_rate_l184_184411

theorem village_population_decrease_rate :
  ∃ (R : ℝ), 15 * R = 18000 :=
by
  sorry

end village_population_decrease_rate_l184_184411


namespace find_positive_integer_n_l184_184690

noncomputable def is_largest_prime_divisor (p n : ℕ) : Prop :=
  (∃ k, n = p * k) ∧ ∀ q, Prime q ∧ q ∣ n → q ≤ p

noncomputable def is_least_prime_divisor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n ∧ ∀ q, Prime q ∧ q ∣ n → p ≤ q

theorem find_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ 
    (∃ p, is_largest_prime_divisor p (n^2 + 3) ∧ is_least_prime_divisor p (n^4 + 6)) ∧
    ∀ m : ℕ, m > 0 ∧ 
      (∃ q, is_largest_prime_divisor q (m^2 + 3) ∧ is_least_prime_divisor q (m^4 + 6)) → m = 3 :=
by sorry

end find_positive_integer_n_l184_184690


namespace map_distance_scaled_l184_184052

theorem map_distance_scaled (d_map : ℝ) (scale : ℝ) (d_actual : ℝ) :
  d_map = 8 ∧ scale = 1000000 → d_actual = 80 :=
by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end map_distance_scaled_l184_184052


namespace product_of_reciprocals_l184_184399

theorem product_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 :=
by
  sorry

end product_of_reciprocals_l184_184399


namespace nate_walks_past_per_minute_l184_184050

-- Define the conditions as constants
def rows_G := 15
def cars_per_row_G := 10
def rows_H := 20
def cars_per_row_H := 9
def total_minutes := 30

-- Define the problem statement
theorem nate_walks_past_per_minute :
  ((rows_G * cars_per_row_G) + (rows_H * cars_per_row_H)) / total_minutes = 11 := 
sorry

end nate_walks_past_per_minute_l184_184050


namespace raft_minimum_capacity_l184_184987

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l184_184987


namespace number_of_intersections_l184_184085

-- Definitions of the given curves.
def curve1 (x y : ℝ) : Prop := x^2 + 4*y^2 = 1
def curve2 (x y : ℝ) : Prop := 4*x^2 + y^2 = 4

-- Statement of the theorem
theorem number_of_intersections : ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2 := sorry

end number_of_intersections_l184_184085


namespace product_xyz_l184_184025

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) (h3 : x + 1 / z = 3) : x * y * z = 2 := 
by sorry

end product_xyz_l184_184025


namespace paper_strip_total_covered_area_l184_184570

theorem paper_strip_total_covered_area :
  let length := 12
  let width := 2
  let strip_count := 5
  let overlap_per_intersection := 4
  let intersection_count := 10
  let area_per_strip := length * width
  let total_area_without_overlap := strip_count * area_per_strip
  let total_overlap_area := intersection_count * overlap_per_intersection
  total_area_without_overlap - total_overlap_area = 80 := 
by
  sorry

end paper_strip_total_covered_area_l184_184570


namespace calculate_x_l184_184676

theorem calculate_x : 121 + 2 * 11 * 8 + 64 = 361 :=
by
  sorry

end calculate_x_l184_184676


namespace smallest_n_div_75_has_75_divisors_l184_184632

theorem smallest_n_div_75_has_75_divisors :
  ∃ n : ℕ, (n % 75 = 0) ∧ (n.factors.length = 75) ∧ (n / 75 = 432) :=
by
  sorry

end smallest_n_div_75_has_75_divisors_l184_184632


namespace logarithmic_inequality_l184_184318

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log (1 / 3) / Real.log 4

theorem logarithmic_inequality :
  Real.log a < (1 / 2)^b := by
  sorry

end logarithmic_inequality_l184_184318


namespace initial_cards_l184_184849

variable (x : ℕ)
variable (h1 : x - 3 = 2)

theorem initial_cards (x : ℕ) (h1 : x - 3 = 2) : x = 5 := by
  sorry

end initial_cards_l184_184849


namespace max_value_of_f_l184_184228

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.cos x - 3

theorem max_value_of_f : ∀ x : ℝ, f x ≤ -1/2 :=
by
  sorry

end max_value_of_f_l184_184228


namespace positive_b_3b_sq_l184_184063

variable (a b c : ℝ)

theorem positive_b_3b_sq (h1 : 0 < a ∧ a < 0.5) (h2 : -0.5 < b ∧ b < 0) (h3 : 1 < c ∧ c < 3) : b + 3 * b^2 > 0 :=
sorry

end positive_b_3b_sq_l184_184063


namespace irrational_implies_irrational_l184_184527

-- Define irrational number proposition
def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

-- Define the main proposition to prove
theorem irrational_implies_irrational (a : ℝ) : is_irrational (a - 2) → is_irrational a :=
by
  sorry

end irrational_implies_irrational_l184_184527


namespace side_lengths_of_triangle_l184_184016

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem side_lengths_of_triangle (m : ℝ) (a b c : ℝ) 
  (h1 : f m a > 0) 
  (h2 : f m b > 0) 
  (h3 : f m c > 0) 
  (h4 : f m a + f m b > f m c)
  (h5 : f m a + f m c > f m b)
  (h6 : f m b + f m c > f m a) :
  m ∈ Set.Ioo (7/5 : ℝ) 5 :=
sorry

end side_lengths_of_triangle_l184_184016


namespace find_intersection_pair_l184_184938

def cubic_function (x : ℝ) : ℝ := x^3 - 3*x + 2

def linear_function (x y : ℝ) : Prop := x + 4*y = 4

def intersection_points (x y : ℝ) : Prop := 
  linear_function x y ∧ y = cubic_function x

def sum_x_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.fst |>.sum

def sum_y_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.snd |>.sum

theorem find_intersection_pair (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : intersection_points x1 y1)
  (h2 : intersection_points x2 y2)
  (h3 : intersection_points x3 y3)
  (h_sum_x : sum_x_coord [(x1, y1), (x2, y2), (x3, y3)] = 0) :
  sum_y_coord [(x1, y1), (x2, y2), (x3, y3)] = 3 :=
sorry

end find_intersection_pair_l184_184938


namespace marys_garbage_bill_is_correct_l184_184762

noncomputable def weekly_cost_trash_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_cost_recycling_bin (price_per_bin : ℝ) (num_bins : ℕ) : ℝ :=
  price_per_bin * num_bins

noncomputable def weekly_total_cost (trash_cost : ℝ) (recycling_cost : ℝ) : ℝ :=
  trash_cost + recycling_cost

noncomputable def monthly_cost (weekly_cost : ℝ) (num_weeks : ℕ) : ℝ :=
  weekly_cost * num_weeks

noncomputable def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount_and_fine (total_cost : ℝ) (discount : ℝ) (fine : ℝ) : ℝ :=
  total_cost - discount + fine

theorem marys_garbage_bill_is_correct :
  let weekly_trash_cost := weekly_cost_trash_bin 10 2 in
  let weekly_recycling_cost := weekly_cost_recycling_bin 5 1 in
  let weekly_total := weekly_total_cost weekly_trash_cost weekly_recycling_cost in
  let monthly_total := monthly_cost weekly_total 4 in
  let senior_discount := discount monthly_total 0.18 in
  let fine := 20 in
  total_cost_after_discount_and_fine monthly_total senior_discount fine = 102 :=
by
  sorry

end marys_garbage_bill_is_correct_l184_184762


namespace max_projection_sum_l184_184974

-- Define the given conditions
def edge_length : ℝ := 2

def projection_front_view (length : ℝ) : Prop := length = edge_length
def projection_side_view (length : ℝ) : Prop := ∃ a : ℝ, a = length
def projection_top_view (length : ℝ) : Prop := ∃ b : ℝ, b = length

-- State the theorem
theorem max_projection_sum (a b : ℝ) (ha : projection_side_view a) (hb : projection_top_view b) :
  a + b ≤ 4 := sorry

end max_projection_sum_l184_184974


namespace problem1_problem2_l184_184968

-- Problem 1:
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

-- Problem 2:
theorem problem2 (α : ℝ) : 
  (Real.tan (2 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α + Real.pi) * Real.sin (-Real.pi + α)) = 1 :=
sorry

end problem1_problem2_l184_184968


namespace phone_not_answered_prob_l184_184738

noncomputable def P_not_answered_within_4_rings : ℝ :=
  let P1 := 1 - 0.1
  let P2 := 1 - 0.3
  let P3 := 1 - 0.4
  let P4 := 1 - 0.1
  P1 * P2 * P3 * P4

theorem phone_not_answered_prob : 
  P_not_answered_within_4_rings = 0.3402 := 
by 
  -- The detailed steps and proof will be implemented here 
  sorry

end phone_not_answered_prob_l184_184738


namespace edge_coloring_balance_l184_184426

-- Conditions: G is a connected graph with at least one vertex of odd degree.
variables {G : Type*} [fintype G] [graph G] [connected G]
           (h : ∃ v : G, odd (degree v))

-- Question: Prove that there exists an edge coloring such that the absolute difference between the number of red edges and blue edges at each vertex does not exceed 1.
theorem edge_coloring_balance :
  ∃ (coloring : edge_set G → ℕ),
    (∀ e, 0 ≤ coloring e ∧ coloring e ≤ 1) ∧  -- coloring is either 0 (red) or 1 (blue)
    (∀ v : G, abs((Σ (e ∈ edge_set G), if (coloring e = 0) then 1 else 0) -
                (Σ (e ∈ edge_set G), if (coloring e = 1) then 1 else 0)) ≤ 1) :=
sorry

end edge_coloring_balance_l184_184426


namespace show_spiders_l184_184376

noncomputable def spiders_found (ants : ℕ) (ladybugs_initial : ℕ) (ladybugs_fly_away : ℕ) (total_insects_remaining : ℕ) : ℕ :=
  let ladybugs_remaining := ladybugs_initial - ladybugs_fly_away
  let insects_observed := ants + ladybugs_remaining
  total_insects_remaining - insects_observed

theorem show_spiders
  (ants : ℕ := 12)
  (ladybugs_initial : ℕ := 8)
  (ladybugs_fly_away : ℕ := 2)
  (total_insects_remaining : ℕ := 21) :
  spiders_found ants ladybugs_initial ladybugs_fly_away total_insects_remaining = 3 := by
  sorry

end show_spiders_l184_184376


namespace speed_difference_l184_184120

theorem speed_difference (distance : ℕ) (time_jordan time_alex : ℕ) (h_distance : distance = 12) (h_time_jordan : time_jordan = 10) (h_time_alex : time_alex = 15) :
  (distance / (time_jordan / 60) - distance / (time_alex / 60) = 24) := by
  -- Lean code to correctly parse and understand the natural numbers, division, and maintain the theorem structure.
  sorry

end speed_difference_l184_184120


namespace students_remaining_l184_184615

theorem students_remaining (students_showed_up : ℕ) (students_checked_out : ℕ) (students_left : ℕ) :
  students_showed_up = 16 → students_checked_out = 7 → students_left = students_showed_up - students_checked_out → students_left = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end students_remaining_l184_184615


namespace number_of_herds_l184_184135

-- Definitions from the conditions
def total_sheep : ℕ := 60
def sheep_per_herd : ℕ := 20

-- The statement to prove
theorem number_of_herds : total_sheep / sheep_per_herd = 3 := by
  sorry

end number_of_herds_l184_184135


namespace locus_centers_of_circles_l184_184505

theorem locus_centers_of_circles (P : ℝ × ℝ) (a : ℝ) (a_pos : 0 < a):
  {O : ℝ × ℝ | dist O P = a} = {O : ℝ × ℝ | dist O P = a} :=
by
  sorry

end locus_centers_of_circles_l184_184505


namespace average_percentage_decrease_is_10_l184_184640

noncomputable def average_percentage_decrease (original_cost final_cost : ℝ) (n : ℕ) : ℝ :=
  1 - (final_cost / original_cost)^(1 / n)

theorem average_percentage_decrease_is_10
  (original_cost current_cost : ℝ)
  (n : ℕ)
  (h_original_cost : original_cost = 100)
  (h_current_cost : current_cost = 81)
  (h_n : n = 2) :
  average_percentage_decrease original_cost current_cost n = 0.1 :=
by
  -- The proof would go here if it were needed.
  sorry

end average_percentage_decrease_is_10_l184_184640


namespace new_volume_is_correct_l184_184657

variable (l w h : ℝ)

-- Conditions given in the problem
axiom volume : l * w * h = 4320
axiom surface_area : 2 * (l * w + w * h + h * l) = 1704
axiom edge_sum : 4 * (l + w + h) = 208

-- The proposition we need to prove:
theorem new_volume_is_correct : (l + 2) * (w + 2) * (h + 2) = 6240 :=
by
  -- Placeholder for the actual proof
  sorry

end new_volume_is_correct_l184_184657


namespace sum_of_distinct_products_l184_184084

theorem sum_of_distinct_products (G H : ℕ) (hG : G < 10) (hH : H < 10) :
  (3 * H + 8) % 8 = 0 ∧ ((6 + 2 + 8 + G + 4 + 0 + 9 + 3 + H + 8) % 9 = 0) →
  (G * H = 6 ∨ G * H = 48) →
  6 + 48 = 54 :=
by
  intros _ _
  sorry

end sum_of_distinct_products_l184_184084


namespace Haley_sweaters_l184_184178

theorem Haley_sweaters (machine_capacity loads shirts sweaters : ℕ) 
    (h_capacity : machine_capacity = 7)
    (h_loads : loads = 5)
    (h_shirts : shirts = 2)
    (h_sweaters_total : sweaters = loads * machine_capacity - shirts) :
  sweaters = 33 :=
by 
  rw [h_capacity, h_loads, h_shirts] at h_sweaters_total
  exact h_sweaters_total

end Haley_sweaters_l184_184178


namespace prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l184_184036

-- Problem 1: Proof of sin(C - B) = 1 given the trigonometric identity
theorem prove_sin_c_minus_b_eq_one
  (A B C : ℝ)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  : Real.sin (C - B) = 1 := 
sorry

-- Problem 2: Proof of CD/BC given the ratios AB:AD:AC and the trigonometric identity
theorem prove_cd_div_bc_eq
  (A B C : ℝ)
  (AB AD AC BC CD : ℝ)
  (h_ratio : AB / AD = Real.sqrt 3 / Real.sqrt 2)
  (h_ratio_2 : AB / AC = Real.sqrt 3 / 1)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  (h_D_on_BC : True) -- Placeholder for D lies on BC condition
  : CD / BC = (Real.sqrt 5 - 1) / 2 := 
sorry

end prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l184_184036


namespace base_conversion_and_addition_l184_184000

theorem base_conversion_and_addition :
  let a₈ : ℕ := 3 * 8^2 + 5 * 8^1 + 6 * 8^0
  let c₁₄ : ℕ := 4 * 14^2 + 12 * 14^1 + 3 * 14^0
  a₈ + c₁₄ = 1193 :=
by
  sorry

end base_conversion_and_addition_l184_184000


namespace positive_integer_triples_l184_184689

theorem positive_integer_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b ∣ (a + 1) ∧ c ∣ (b + 1) ∧ a ∣ (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1 ∨
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 5 ∧ b = 3 ∧ c = 4) :=
by
  sorry

end positive_integer_triples_l184_184689


namespace find_x_l184_184607

def x_y_conditions (x y : ℝ) : Prop :=
  x > y ∧
  x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40 ∧
  x * y + x + y = 8

theorem find_x (x y : ℝ) (h : x_y_conditions x y) : x = 3 + Real.sqrt 7 :=
by
  sorry

end find_x_l184_184607


namespace average_expression_l184_184785

-- Define a theorem to verify the given problem
theorem average_expression (E a : ℤ) (h1 : a = 34) (h2 : (E + (3 * a - 8)) / 2 = 89) : E = 84 :=
by
  -- Proof goes here
  sorry

end average_expression_l184_184785


namespace jancy_currency_notes_l184_184478

theorem jancy_currency_notes (x y : ℕ) (h1 : 70 * x + 50 * y = 5000) (h2 : y = 2) : x + y = 72 :=
by
  -- proof goes here
  sorry

end jancy_currency_notes_l184_184478


namespace total_food_for_guinea_pigs_l184_184775

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end total_food_for_guinea_pigs_l184_184775


namespace first_discount_percentage_l184_184394

-- Given conditions
def initial_price : ℝ := 390
def final_price : ℝ := 285.09
def second_discount : ℝ := 0.15

-- Definition for the first discount percentage
noncomputable def first_discount (D : ℝ) : ℝ :=
initial_price * (1 - D / 100) * (1 - second_discount)

-- Theorem statement
theorem first_discount_percentage : ∃ D : ℝ, first_discount D = final_price ∧ D = 13.99 :=
by
  sorry

end first_discount_percentage_l184_184394


namespace vampire_conversion_l184_184958

theorem vampire_conversion (x : ℕ) 
  (h_population : village_population = 300)
  (h_initial_vampires : initial_vampires = 2)
  (h_two_nights_vampires : 2 + 2 * x + x * (2 + 2 * x) = 72) :
  x = 5 :=
by
  -- Proof will be added here
  sorry

end vampire_conversion_l184_184958


namespace history_percentage_l184_184662

theorem history_percentage (H : ℕ) (math_percentage : ℕ := 72) (third_subject_percentage : ℕ := 69) (overall_average : ℕ := 75) :
  (math_percentage + H + third_subject_percentage) / 3 = overall_average → H = 84 :=
by
  intro h
  sorry

end history_percentage_l184_184662


namespace daisies_left_l184_184753

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9
def total_daisies : ℕ := initial_daisies + sister_daisies
def daisies_given_to_mother : ℕ := total_daisies / 2
def remaining_daisies : ℕ := total_daisies - daisies_given_to_mother

theorem daisies_left : remaining_daisies = 7 := by
  sorry

end daisies_left_l184_184753


namespace kimiko_age_l184_184212

noncomputable def K : ℚ := 28

theorem kimiko_age (O A : ℚ) (K : ℚ) 
  (h1 : O = 2 * K)
  (h2 : A = (3 / 4) * K)
  (h3 : (K + O + A) / 3 = 35) : K = 28 :=
by
  sorry

end kimiko_age_l184_184212


namespace pq_ratio_at_0_l184_184389

noncomputable def p (x : ℝ) : ℝ := -3 * (x + 4) * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 3)

theorem pq_ratio_at_0 : (p 0) / (q 0) = 0 := by
  sorry

end pq_ratio_at_0_l184_184389


namespace polygon_sides_l184_184730

theorem polygon_sides (h : 1440 = (n - 2) * 180) : n = 10 := 
by {
  -- Here, the proof would show the steps to solve the equation h and confirm n = 10
  sorry
}

end polygon_sides_l184_184730


namespace tangent_line_eqn_at_one_l184_184081

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_eqn_at_one :
  let k := (Real.exp 1)
  let p := (1, Real.exp 1)
  ∃ m b : ℝ, (m = k) ∧ (b = p.2 - m * p.1) ∧ (∀ x, f x = y → y = m * x + b) :=
sorry

end tangent_line_eqn_at_one_l184_184081


namespace remainder_of_sum_mod_l184_184896

theorem remainder_of_sum_mod (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2 * n) % 7 :=
by
  sorry

end remainder_of_sum_mod_l184_184896


namespace apple_cost_l184_184554

theorem apple_cost (rate_cost : ℕ) (rate_weight total_weight : ℕ) (h_rate : rate_cost = 5) (h_weight : rate_weight = 7) (h_total : total_weight = 21) :
  ∃ total_cost : ℕ, total_cost = 15 :=
by
  -- The proof will go here
  sorry

end apple_cost_l184_184554


namespace extreme_point_at_one_l184_184725

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end extreme_point_at_one_l184_184725


namespace negation_of_existence_l184_184314

theorem negation_of_existence (p : Prop) (h : ∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) : 
  ¬ (∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) ↔ 
  ∀ (c : ℝ), c > 0 → ¬ (∃ (x : ℝ), x^2 - x + c = 0) :=
by 
  sorry

end negation_of_existence_l184_184314


namespace solution_greater_iff_l184_184043

variables {c c' d d' : ℝ}
variables (hc : c ≠ 0) (hc' : c' ≠ 0)

theorem solution_greater_iff : (∃ x, x = -d / c) > (∃ x, x = -d' / c') ↔ (d' / c') < (d / c) :=
by sorry

end solution_greater_iff_l184_184043


namespace total_possible_arrangements_l184_184213

-- Define the subjects
inductive Subject : Type
| PoliticalScience
| Chinese
| Mathematics
| English
| PhysicalEducation
| Physics

open Subject

-- Define the condition that the first period cannot be Chinese
def first_period_cannot_be_chinese (schedule : Fin 6 → Subject) : Prop :=
  schedule 0 ≠ Chinese

-- Define the condition that the fifth period cannot be English
def fifth_period_cannot_be_english (schedule : Fin 6 → Subject) : Prop :=
  schedule 4 ≠ English

-- Define the schedule includes six unique subjects
def schedule_includes_all_subjects (schedule : Fin 6 → Subject) : Prop :=
  ∀ s : Subject, ∃ i : Fin 6, schedule i = s

-- Define the main theorem to prove the total number of possible arrangements
theorem total_possible_arrangements : 
  ∃ (schedules : List (Fin 6 → Subject)), 
  (∀ schedule, schedule ∈ schedules → 
    first_period_cannot_be_chinese schedule ∧ 
    fifth_period_cannot_be_english schedule ∧ 
    schedule_includes_all_subjects schedule) ∧ 
  schedules.length = 600 :=
sorry

end total_possible_arrangements_l184_184213


namespace no_fixed_point_implies_no_double_fixed_point_l184_184644

theorem no_fixed_point_implies_no_double_fixed_point (f : ℝ → ℝ) 
  (hf : Continuous f)
  (h : ∀ x : ℝ, f x ≠ x) :
  ∀ x : ℝ, f (f x) ≠ x :=
sorry

end no_fixed_point_implies_no_double_fixed_point_l184_184644


namespace option_B_correct_option_C_correct_l184_184252

-- Define the permutation coefficient
def A (n m : ℕ) : ℕ := n * (n-1) * (n-2) * (n-m+1)

-- Prove the equation for option B
theorem option_B_correct (n m : ℕ) : A (n+1) (m+1) - A n m = n^2 * A (n-1) (m-1) :=
by
  sorry

-- Prove the equation for option C
theorem option_C_correct (n m : ℕ) : A n m = n * A (n-1) (m-1) :=
by
  sorry

end option_B_correct_option_C_correct_l184_184252


namespace multiply_expression_l184_184496

theorem multiply_expression (x : ℝ) : (x^4 + 12 * x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end multiply_expression_l184_184496


namespace M_subset_N_iff_l184_184176

section
variables {a x : ℝ}

-- Definitions based on conditions in the problem
def M (a : ℝ) : Set ℝ := { x | x^2 - a * x - x < 0 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem M_subset_N_iff (a : ℝ) : M a ⊆ N ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry
end

end M_subset_N_iff_l184_184176


namespace max_servings_hot_chocolate_l184_184114

def recipe_servings : ℕ := 5
def chocolate_required : ℕ := 2 -- squares of chocolate required for 5 servings
def sugar_required : ℚ := 1 / 4 -- cups of sugar required for 5 servings
def water_required : ℕ := 1 -- cups of water required (not limiting)
def milk_required : ℕ := 4 -- cups of milk required for 5 servings

def chocolate_available : ℕ := 5 -- squares of chocolate Jordan has
def sugar_available : ℚ := 2 -- cups of sugar Jordan has
def milk_available : ℕ := 7 -- cups of milk Jordan has
def water_available_lots : Prop := True -- Jordan has lots of water (not limited)

def servings_from_chocolate := (chocolate_available / chocolate_required) * recipe_servings
def servings_from_sugar := (sugar_available / sugar_required) * recipe_servings
def servings_from_milk := (milk_available / milk_required) * recipe_servings

def max_servings (a b c : ℚ) : ℚ := min (min a b) c

theorem max_servings_hot_chocolate :
  max_servings servings_from_chocolate servings_from_sugar servings_from_milk = 35 / 4 :=
by
  sorry

end max_servings_hot_chocolate_l184_184114


namespace orthogonal_vectors_l184_184449

theorem orthogonal_vectors (x : ℝ) :
  (3 * x - 4 * 6 = 0) → x = 8 :=
by
  intro h
  sorry

end orthogonal_vectors_l184_184449


namespace percentage_increase_in_second_year_l184_184249

def initial_deposit : ℝ := 5000
def first_year_balance : ℝ := 5500
def two_year_increase_percentage : ℝ := 21
def second_year_increase_percentage : ℝ := 10

theorem percentage_increase_in_second_year
  (initial_deposit first_year_balance : ℝ) 
  (two_year_increase_percentage : ℝ) 
  (h1 : first_year_balance = initial_deposit + 500) 
  (h2 : (initial_deposit * (1 + two_year_increase_percentage / 100)) = initial_deposit * 1.21) 
  : second_year_increase_percentage = 10 := 
sorry

end percentage_increase_in_second_year_l184_184249


namespace simplify_expression_l184_184220

theorem simplify_expression (x : ℝ) : (3 * x) ^ 5 - (4 * x) * (x ^ 4) = 239 * x ^ 5 := 
by
  sorry

end simplify_expression_l184_184220


namespace sum_of_two_numbers_is_10_l184_184401

variable (a b : ℝ)

theorem sum_of_two_numbers_is_10
  (h1 : a + b = 10)
  (h2 : a - b = 8)
  (h3 : a^2 - b^2 = 80) :
  a + b = 10 :=
by
  sorry

end sum_of_two_numbers_is_10_l184_184401


namespace find_f_l184_184257

theorem find_f (q f : ℕ) (h_digit_q : q ≤ 9) (h_digit_f : f ≤ 9)
  (h_distinct : q ≠ f) 
  (h_div_by_36 : (457 * 1000 + q * 100 + 89 * 10 + f) % 36 = 0)
  (h_sum_3 : q + f = 3) :
  f = 2 :=
sorry

end find_f_l184_184257


namespace m_eq_n_is_necessary_but_not_sufficient_l184_184076

noncomputable def circle_condition (m n : ℝ) : Prop :=
  m = n ∧ m > 0

theorem m_eq_n_is_necessary_but_not_sufficient 
  (m n : ℝ) :
  (circle_condition m n → mx^2 + ny^2 = 3 → False) ∧
  (mx^2 + ny^2 = 3 → circle_condition m n) :=
by 
  sorry

end m_eq_n_is_necessary_but_not_sufficient_l184_184076


namespace sum_of_fractions_l184_184129

theorem sum_of_fractions :
  (2 / 8) + (4 / 8) + (6 / 8) + (8 / 8) + (10 / 8) + 
  (12 / 8) + (14 / 8) + (16 / 8) + (18 / 8) + (20 / 8) = 13.75 :=
by sorry

end sum_of_fractions_l184_184129


namespace value_of_a_minus_b_l184_184589

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = - (a + b)) :
  a - b = -2 ∨ a - b = -6 := sorry

end value_of_a_minus_b_l184_184589


namespace proof_prob_at_least_one_die_3_or_5_l184_184417

def probability_at_least_one_die_3_or_5 (total_outcomes : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem proof_prob_at_least_one_die_3_or_5 :
  let total_outcomes := 36
  let favorable_outcomes := 20
  probability_at_least_one_die_3_or_5 total_outcomes favorable_outcomes = 5 / 9 := 
by 
  sorry

end proof_prob_at_least_one_die_3_or_5_l184_184417


namespace Megan_acorns_now_l184_184365

def initial_acorns := 16
def given_away_acorns := 7
def remaining_acorns := initial_acorns - given_away_acorns

theorem Megan_acorns_now : remaining_acorns = 9 := by
  sorry

end Megan_acorns_now_l184_184365


namespace fish_market_customers_l184_184920

theorem fish_market_customers :
  let num_tuna := 10
  let weight_per_tuna := 200
  let weight_per_customer := 25
  let num_customers_no_fish := 20
  let total_tuna_weight := num_tuna * weight_per_tuna
  let num_customers_served := total_tuna_weight / weight_per_customer
  num_customers_served + num_customers_no_fish = 100 := 
by
  sorry

end fish_market_customers_l184_184920


namespace ratio_roots_l184_184637

theorem ratio_roots (p q r s : ℤ)
    (h1 : p ≠ 0)
    (h_roots : ∀ x : ℤ, (x = -1 ∨ x = 3 ∨ x = 4) → (p*x^3 + q*x^2 + r*x + s = 0)) : 
    (r : ℚ) / s = -5 / 12 :=
by sorry

end ratio_roots_l184_184637


namespace fenced_area_l184_184820

theorem fenced_area (L W : ℝ) (square_side triangle_leg : ℝ) :
  L = 20 ∧ W = 18 ∧ square_side = 4 ∧ triangle_leg = 3 →
  (L * W - square_side^2 - (1 / 2) * triangle_leg^2 = 339.5) := by
  intros h
  rcases h with ⟨hL, hW, hs, ht⟩
  rw [hL, hW, hs, ht]
  simp
  sorry

end fenced_area_l184_184820


namespace minimize_value_l184_184362

noncomputable def minimize_y (a b x : ℝ) : ℝ := (x - a) ^ 3 + (x - b) ^ 3

theorem minimize_value (a b : ℝ) : ∃ x : ℝ, minimize_y a b x = minimize_y a b a ∨ minimize_y a b x = minimize_y a b b :=
sorry

end minimize_value_l184_184362


namespace person_B_catches_up_after_meeting_point_on_return_l184_184536
noncomputable def distance_A := 46
noncomputable def speed_A := 15
noncomputable def speed_B := 40
noncomputable def initial_gap_time := 1

-- Prove that Person B catches up to Person A after 3/5 hours.
theorem person_B_catches_up_after : 
  ∃ x : ℚ, 40 * x = 15 * (x + 1) ∧ x = 3 / 5 := 
by
  sorry

-- Prove that they meet 10 kilometers away from point B on the return journey.
theorem meeting_point_on_return : 
  ∃ y : ℚ, (46 - y) / 15 - (46 + y) / 40 = 1 ∧ y = 10 := 
by 
  sorry

end person_B_catches_up_after_meeting_point_on_return_l184_184536


namespace assignment_statement_correct_l184_184506

-- Definitions for the conditions:
def cond_A : Prop := ∀ M : ℕ, (M = M + 3)
def cond_B : Prop := ∀ M : ℕ, (M = M + (3 - M))
def cond_C : Prop := ∀ M : ℕ, (M = M + 3)
def cond_D : Prop := true ∧ cond_A ∧ cond_B ∧ cond_C

-- Theorem statement proving the correct interpretation of the assignment is condition B
theorem assignment_statement_correct : cond_B :=
by
  sorry

end assignment_statement_correct_l184_184506


namespace probability_left_oar_works_l184_184540

structure Oars where
  P_L : ℝ -- Probability that the left oar works
  P_R : ℝ -- Probability that the right oar works
  
def independent_prob (o : Oars) : Prop :=
  o.P_L = o.P_R ∧ (1 - o.P_L) * (1 - o.P_R) = 0.16

theorem probability_left_oar_works (o : Oars) (h1 : independent_prob o) (h2 : 1 - (1 - o.P_L) * (1 - o.P_R) = 0.84) : o.P_L = 0.6 :=
by
  sorry

end probability_left_oar_works_l184_184540


namespace inequality_holds_for_all_x_l184_184899

theorem inequality_holds_for_all_x (m : ℝ) : (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by {
  sorry
}

end inequality_holds_for_all_x_l184_184899


namespace dhoni_toys_average_cost_l184_184841

theorem dhoni_toys_average_cost (A : ℝ) (h1 : ∃ x1 x2 x3 x4 x5, (x1 + x2 + x3 + x4 + x5) / 5 = A)
  (h2 : 5 * A = 5 * A)
  (h3 : ∃ x6, x6 = 16)
  (h4 : (5 * A + 16) / 6 = 11) : A = 10 :=
by
  sorry

end dhoni_toys_average_cost_l184_184841


namespace sherry_needs_bananas_l184_184219

/-
Conditions:
- Sherry wants to make 99 loaves.
- Her recipe makes enough batter for 3 loaves.
- The recipe calls for 1 banana per batch of 3 loaves.

Question:
- How many bananas does Sherry need?

Equivalent Proof Problem:
- Prove that given the conditions, the number of bananas needed is 33.
-/

def total_loaves : ℕ := 99
def loaves_per_batch : ℕ := 3
def bananas_per_batch : ℕ := 1

theorem sherry_needs_bananas :
  (total_loaves / loaves_per_batch) * bananas_per_batch = 33 :=
sorry

end sherry_needs_bananas_l184_184219


namespace infinite_a_exists_l184_184700

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ), ∀ (k : ℕ), ∃ (a : ℕ), n^6 + 3 * a = (n^2 + 3 * k)^3 := 
sorry

end infinite_a_exists_l184_184700


namespace tiffany_won_lives_l184_184245
-- Step d: Lean 4 statement incorporating the conditions and the proof goal


-- Define initial lives, lives won in the hard part and the additional lives won
def initial_lives : Float := 43.0
def additional_lives : Float := 27.0
def total_lives_after_wins : Float := 84.0

open Classical

theorem tiffany_won_lives (x : Float) :
    initial_lives + x + additional_lives = total_lives_after_wins →
    x = 14.0 :=
by
  intros h
  -- This "sorry" indicates that the proof is skipped.
  sorry

end tiffany_won_lives_l184_184245


namespace parallelogram_rectangle_l184_184266

/-- A quadrilateral is a parallelogram if both pairs of opposite sides are equal,
and it is a rectangle if its diagonals are equal. -/
structure Quadrilateral :=
  (side1 side2 side3 side4 : ℝ)
  (diag1 diag2 : ℝ)

structure Parallelogram extends Quadrilateral :=
  (opposite_sides_equal : side1 = side3 ∧ side2 = side4)

def is_rectangle (p : Parallelogram) : Prop :=
  p.diag1 = p.diag2 → (p.side1^2 + p.side2^2 = p.side3^2 + p.side4^2)

theorem parallelogram_rectangle (p : Parallelogram) : is_rectangle p :=
  sorry

end parallelogram_rectangle_l184_184266


namespace log_base_change_l184_184802

theorem log_base_change (a b : ℝ) (h₁ : Real.log 2 / Real.log 10 = a) (h₂ : Real.log 3 / Real.log 10 = b) :
    Real.log 18 / Real.log 5 = (a + 2 * b) / (1 - a) := by
  sorry

end log_base_change_l184_184802


namespace number_of_students_l184_184111

theorem number_of_students 
  (n : ℕ)
  (h1: 108 - 36 = 72)
  (h2: ∀ n > 0, 108 / n - 72 / n = 3) :
  n = 12 :=
sorry

end number_of_students_l184_184111


namespace min_value_l184_184357

theorem min_value
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 9) :
  ∃ x : ℝ, x = 9 ∧ x = min (λ x, (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)) := 
sorry

end min_value_l184_184357


namespace harris_flour_amount_l184_184407

noncomputable def flour_needed_by_cakes (cakes : ℕ) : ℕ := cakes * 100

noncomputable def traci_flour : ℕ := 500

noncomputable def total_cakes : ℕ := 9

theorem harris_flour_amount : flour_needed_by_cakes total_cakes - traci_flour = 400 := 
by
  sorry

end harris_flour_amount_l184_184407


namespace num_pairs_satisfying_equation_l184_184720

theorem num_pairs_satisfying_equation :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x^2 - y^2 = 204 :=
by
  sorry

end num_pairs_satisfying_equation_l184_184720


namespace arithmetic_sequence_sum_equality_l184_184363

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

end arithmetic_sequence_sum_equality_l184_184363


namespace math_dance_residents_l184_184609

theorem math_dance_residents (p a b : ℕ) (hp : Nat.Prime p) 
    (h1 : b ≥ 1) 
    (h2 : (a + b)^2 = (p + 1) * a + b) :
    b = 1 := by
  sorry

end math_dance_residents_l184_184609


namespace eggs_in_nests_l184_184749

theorem eggs_in_nests (x : ℕ) (h1 : 2 * x + 3 + 4 = 17) : x = 5 :=
by
  /- This is where the proof would go, but the problem only requires the statement -/
  sorry

end eggs_in_nests_l184_184749


namespace fraction_of_3_5_eq_2_15_l184_184412

theorem fraction_of_3_5_eq_2_15 : (2 / 15) / (3 / 5) = 2 / 9 := by
  sorry

end fraction_of_3_5_eq_2_15_l184_184412


namespace range_of_p_l184_184873

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end range_of_p_l184_184873


namespace solution_x_y_l184_184847

noncomputable def eq_values (x y : ℝ) := (
  x ≠ 0 ∧ x ≠ 1 ∧ y ≠ 0 ∧ y ≠ 3 ∧ (3/x + 2/y = 1/3)
)

theorem solution_x_y (x y : ℝ) (h : eq_values x y) : x = 9 * y / (y - 6) :=
sorry

end solution_x_y_l184_184847


namespace sum_of_numbers_l184_184214

theorem sum_of_numbers (x y : ℕ) (h1 : x = 18) (h2 : y = 2 * x - 3) : x + y = 51 :=
by
  sorry

end sum_of_numbers_l184_184214


namespace tennis_tournament_total_rounds_l184_184972

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end tennis_tournament_total_rounds_l184_184972


namespace calculate_correct_subtraction_l184_184722

theorem calculate_correct_subtraction (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 :=
by
  sorry

end calculate_correct_subtraction_l184_184722


namespace no_real_solutions_l184_184786

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
    (a = 0) ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a ^ 2 > 0) :=
by
  sorry

end no_real_solutions_l184_184786


namespace train_cross_bridge_time_l184_184531

noncomputable def time_to_cross_bridge (L_train : ℕ) (v_kmph : ℕ) (L_bridge : ℕ) : ℝ :=
  let v_mps := (v_kmph * 1000) / 3600
  let total_distance := L_train + L_bridge
  total_distance / v_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 145 54 660 = 53.67 := by
    sorry

end train_cross_bridge_time_l184_184531


namespace layers_tall_l184_184494

def total_cards (n_d c_d : ℕ) : ℕ := n_d * c_d
def layers (total c_l : ℕ) : ℕ := total / c_l

theorem layers_tall (n_d c_d c_l : ℕ) (hn_d : n_d = 16) (hc_d : c_d = 52) (hc_l : c_l = 26) : 
  layers (total_cards n_d c_d) c_l = 32 := by
  sorry

end layers_tall_l184_184494


namespace element_of_sequence_l184_184325

/-
Proving that 63 is an element of the sequence defined by aₙ = n² + 2n.
-/
theorem element_of_sequence (n : ℕ) (h : 63 = n^2 + 2 * n) : ∃ n : ℕ, 63 = n^2 + 2 * n :=
by
  sorry

end element_of_sequence_l184_184325


namespace even_suff_not_nec_l184_184812

theorem even_suff_not_nec (f g : ℝ → ℝ) 
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hg_even : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x + g x) = ((f + g) x) ∧ (∀ h : ℝ → ℝ, ∃ f g : ℝ → ℝ, h = f + g ∧ ∀ x : ℝ, (h (-x) = h x) ↔ (f (-x) = f x ∧ g (-x) = g x)) :=
by 
  sorry

end even_suff_not_nec_l184_184812


namespace two_digit_num_square_ends_in_self_l184_184444

theorem two_digit_num_square_ends_in_self {x : ℕ} (hx : 10 ≤ x ∧ x < 100) (hx0 : x % 10 ≠ 0) : 
  (x * x % 100 = x) ↔ (x = 25 ∨ x = 76) :=
sorry

end two_digit_num_square_ends_in_self_l184_184444


namespace students_recess_time_l184_184946

def initial_recess : ℕ := 20

def extra_minutes_as (as : ℕ) : ℕ := 4 * as
def extra_minutes_bs (bs : ℕ) : ℕ := 3 * bs
def extra_minutes_cs (cs : ℕ) : ℕ := 2 * cs
def extra_minutes_ds (ds : ℕ) : ℕ := ds
def extra_minutes_es (es : ℕ) : ℤ := - es
def extra_minutes_fs (fs : ℕ) : ℤ := -2 * fs

def total_recess (as bs cs ds es fs : ℕ) : ℤ :=
  initial_recess + 
  (extra_minutes_as as + extra_minutes_bs bs +
  extra_minutes_cs cs + extra_minutes_ds ds +
  extra_minutes_es es + extra_minutes_fs fs : ℤ)

theorem students_recess_time :
  total_recess 10 12 14 5 3 2 = 122 := by sorry

end students_recess_time_l184_184946


namespace product_congruent_three_mod_p_l184_184921

open BigOperators

theorem product_congruent_three_mod_p {p : ℕ} (hp : Nat.Prime p) (hp3 : p > 3) (hp_congruent : p % 3 = 2) :
  (∏ k in Finset.range (p - 1), (k^2 + k + 1) : ZMod p) = 3 := sorry

end product_congruent_three_mod_p_l184_184921


namespace fish_count_when_james_discovers_l184_184733

def fish_in_aquarium (initial_fish : ℕ) (bobbit_worm_eats : ℕ) (predatory_fish_eats : ℕ)
  (reproduction_rate : ℕ × ℕ) (days_1 : ℕ) (added_fish: ℕ) (days_2 : ℕ) : ℕ :=
  let predation_rate := bobbit_worm_eats + predatory_fish_eats
  let total_eaten_in_14_days := predation_rate * days_1
  let reproduction_events_in_14_days := days_1 / reproduction_rate.snd
  let fish_born_in_14_days := reproduction_events_in_14_days * reproduction_rate.fst
  let fish_after_14_days := initial_fish - total_eaten_in_14_days + fish_born_in_14_days
  let fish_after_14_days_non_negative := max fish_after_14_days 0
  let fish_after_addition := fish_after_14_days_non_negative + added_fish
  let total_eaten_in_7_days := predation_rate * days_2
  let reproduction_events_in_7_days := days_2 / reproduction_rate.snd
  let fish_born_in_7_days := reproduction_events_in_7_days * reproduction_rate.fst
  let fish_after_7_days := fish_after_addition - total_eaten_in_7_days + fish_born_in_7_days
  max fish_after_7_days 0

theorem fish_count_when_james_discovers :
  fish_in_aquarium 60 2 3 (2, 3) 14 8 7 = 4 :=
sorry

end fish_count_when_james_discovers_l184_184733


namespace keychain_arrangement_l184_184740

theorem keychain_arrangement (house car locker office key5 key6 : ℕ) :
  (∃ (A B : ℕ), house = A ∧ car = A ∧ locker = B ∧ office = B) →
  (∃ (arrangements : ℕ), arrangements = 24) :=
by
  sorry

end keychain_arrangement_l184_184740


namespace multiplication_of_powers_of_10_l184_184962

theorem multiplication_of_powers_of_10 : (10 : ℝ) ^ 65 * (10 : ℝ) ^ 64 = (10 : ℝ) ^ 129 := by
  sorry

end multiplication_of_powers_of_10_l184_184962


namespace option_C_correct_l184_184122

theorem option_C_correct (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1 / 2 :=
sorry

end option_C_correct_l184_184122


namespace roots_of_quadratic_eq_l184_184790

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end roots_of_quadratic_eq_l184_184790


namespace card_statements_are_false_l184_184265

theorem card_statements_are_false :
  ¬( ( (statements: ℕ) →
        (statements = 1 ↔ ¬statements = 1 ∧ ¬statements = 2 ∧ ¬statements = 3 ∧ ¬statements = 4 ∧ ¬statements = 5) ∧
        ( statements = 2 ↔ (statements = 1 ∨ statements = 3 ∨ statements = 4 ∨ statements = 5)) ∧
        (statements = 3 ↔ (statements = 1 ∧ statements = 2 ∧ (statements = 4 ∨ statements = 5) ) ) ∧
        (statements = 4 ↔ (statements = 1 ∧ statements = 2 ∧ statements = 3 ∧ statements != 5 ) ) ∧
        (statements = 5 ↔ (statements = 4 ) )
)) :=
sorry

end card_statements_are_false_l184_184265


namespace geometric_seq_a4_l184_184572

variable {a : ℕ → ℝ}

-- Definition: a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition
axiom h : a 2 * a 6 = 4

-- Theorem that needs to be proved
theorem geometric_seq_a4 (h_seq: is_geometric_sequence a) (h: a 2 * a 6 = 4) : a 4 = 2 ∨ a 4 = -2 := by
  sorry

end geometric_seq_a4_l184_184572


namespace Jim_paycheck_correct_l184_184482

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end Jim_paycheck_correct_l184_184482


namespace path_count_through_B_l184_184282

open SimpleGraph
open Finset

variable (A B C D F G I : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq F] [DecidableEq G] [DecidableEq I]

/-- The number of paths from A to C passing through B -/
theorem path_count_through_B 
    (ant_paths : A → B → C → D → F → G → I → ℕ) 
    (A_to_B_paths : ℕ := 2) 
    (B_to_C_paths : ℕ := 2) 
    : ant_paths A B C D F G I = A_to_B_paths * B_to_C_paths := 
begin
    sorry
end

end path_count_through_B_l184_184282


namespace cisco_spots_difference_l184_184330

theorem cisco_spots_difference :
  ∃ C G R : ℕ, R = 46 ∧ G = 5 * C ∧ G + C = 108 ∧ (23 - C) = 5 :=
by
  sorry

end cisco_spots_difference_l184_184330


namespace mary_garbage_bill_l184_184764

theorem mary_garbage_bill :
  let weekly_cost := 2 * 10 + 1 * 5,
      monthly_cost := weekly_cost * 4,
      discount := 0.18 * monthly_cost,
      discounted_monthly_cost := monthly_cost - discount,
      fine := 20,
      total_bill := discounted_monthly_cost + fine
  in total_bill = 102 :=
by
  let weekly_cost := 2 * 10 + 1 * 5
  let monthly_cost := weekly_cost * 4
  let discount := 0.18 * monthly_cost
  let discounted_monthly_cost := monthly_cost - discount
  let fine := 20
  let total_bill := discounted_monthly_cost + fine
  show total_bill = 102 from sorry

end mary_garbage_bill_l184_184764


namespace pictures_at_museum_l184_184808

-- Define the given conditions
def z : ℕ := 24
def k : ℕ := 14
def p : ℕ := 22

-- Define the number of pictures taken at the museum
def M : ℕ := 12

-- The theorem to be proven
theorem pictures_at_museum :
  z + M - k = p ↔ M = 12 :=
by
  sorry

end pictures_at_museum_l184_184808


namespace five_letter_words_with_vowel_l184_184884

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l184_184884


namespace triangle_sum_is_19_l184_184382

-- Defining the operation on a triangle
def triangle_op (a b c : ℕ) := a * b - c

-- Defining the vertices of the two triangles
def triangle1 := (4, 2, 3)
def triangle2 := (3, 5, 1)

-- Statement that the sum of the operation results is 19
theorem triangle_sum_is_19 :
  triangle_op (4) (2) (3) + triangle_op (3) (5) (1) = 19 :=
by
  -- Triangle 1 calculation: 4 * 2 - 3 = 8 - 3 = 5
  -- Triangle 2 calculation: 3 * 5 - 1 = 15 - 1 = 14
  -- Sum of calculations: 5 + 14 = 19
  sorry

end triangle_sum_is_19_l184_184382


namespace international_call_cost_per_minute_l184_184850

theorem international_call_cost_per_minute 
  (local_call_minutes : Nat)
  (international_call_minutes : Nat)
  (local_rate : Nat)
  (total_cost_cents : Nat) 
  (spent_dollars : Nat) 
  (spent_cents : Nat)
  (local_call_cost : Nat)
  (international_call_total_cost : Nat) : 
  local_call_minutes = 45 → 
  international_call_minutes = 31 → 
  local_rate = 5 → 
  total_cost_cents = spent_dollars * 100 → 
  spent_dollars = 10 → 
  local_call_cost = local_call_minutes * local_rate → 
  spent_cents = spent_dollars * 100 → 
  total_cost_cents = spent_cents →  
  international_call_total_cost = total_cost_cents - local_call_cost → 
  international_call_total_cost / international_call_minutes = 25 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end international_call_cost_per_minute_l184_184850


namespace find_ordered_pair_l184_184568

theorem find_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y)) 
  (h2 : x - y = (x - 2) + (y - 2)) : 
  (x = 5 ∧ y = 2) :=
by
  sorry

end find_ordered_pair_l184_184568


namespace at_least_3_babies_speak_l184_184186

noncomputable def probability_at_least_3_speak (p : ℚ) (n : ℕ) : ℚ := 
1 - (1 - p) ^ n - n * p * (1 - p) ^ (n - 1) - n * (n - 1) / 2 * p^2 * (1 - p) ^ (n - 2)

theorem at_least_3_babies_speak :
  probability_at_least_3_speak (1 / 5) 7 = 45349 / 78125 :=
by
  sorry

end at_least_3_babies_speak_l184_184186


namespace hess_law_delta_H298_l184_184239

def standardEnthalpyNa2O : ℝ := -416 -- kJ/mol
def standardEnthalpyH2O : ℝ := -286 -- kJ/mol
def standardEnthalpyNaOH : ℝ := -427.8 -- kJ/mol
def deltaH298 : ℝ := 2 * standardEnthalpyNaOH - (standardEnthalpyNa2O + standardEnthalpyH2O) 

theorem hess_law_delta_H298 : deltaH298 = -153.6 := by
  sorry

end hess_law_delta_H298_l184_184239


namespace line_through_point_equal_distance_l184_184171

noncomputable def line_equation (x0 y0 a b c x1 y1 : ℝ) : Prop :=
  (a * x0 + b * y0 + c = 0) ∧ (a * x1 + b * y1 + c = 0)

theorem line_through_point_equal_distance (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ (a b c : ℝ), 
    line_equation P.1 P.2 a b c A.1 A.2 ∧ 
    line_equation P.1 P.2 a b c B.1 B.2 ∧
    (a = 2) ∧ (b = 3) ∧ (c = -18) ∨
    (a = 2) ∧ (b = -1) ∧ (c = -2)
:=
sorry

end line_through_point_equal_distance_l184_184171


namespace carl_centroid_markable_l184_184757

def divides (m n : ℕ) : Prop := ∃ k, n = m * k

def rad (n : ℕ) : ℕ := n.coprime_part

-- The main problem statement as a Lean theorem
theorem carl_centroid_markable (n : ℕ) :
  ∃ m, (∃ (k : ℕ), m = k * rad (2 * n)) ∧ (∀ (points : list (ℝ × ℝ)), points.length = n →
    let centroid :=
      (list.sum (points.map prod.fst) / n, list.sum (points.map prod.snd) / n) in
    (∃ (a b : ℝ × ℝ), ∃ (div_points : list (ℝ × ℝ)),
      length div_points = m - 1 ∧
      (list.map_with_index (λ (i : ℕ) (_ : ℝ × ℝ), (1 - i / m, i / m)) div_points) = centroid)) :=
sorry

end carl_centroid_markable_l184_184757


namespace Jacob_has_48_graham_crackers_l184_184039

def marshmallows_initial := 6
def marshmallows_needed := 18
def marshmallows_total := marshmallows_initial + marshmallows_needed
def graham_crackers_per_smore := 2

def smores_total := marshmallows_total
def graham_crackers_total := smores_total * graham_crackers_per_smore

theorem Jacob_has_48_graham_crackers (h1 : marshmallows_initial = 6)
                                     (h2 : marshmallows_needed = 18)
                                     (h3 : graham_crackers_per_smore = 2)
                                     (h4 : marshmallows_total = marshmallows_initial + marshmallows_needed)
                                     (h5 : smores_total = marshmallows_total)
                                     (h6 : graham_crackers_total = smores_total * graham_crackers_per_smore) :
                                     graham_crackers_total = 48 :=
by
  sorry

end Jacob_has_48_graham_crackers_l184_184039


namespace sector_perimeter_l184_184169

theorem sector_perimeter (A θ r: ℝ) (hA : A = 2) (hθ : θ = 4) (hArea : A = (1/2) * r^2 * θ) : (2 * r + r * θ) = 6 :=
by 
  sorry

end sector_perimeter_l184_184169


namespace simplify_expression_l184_184066

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 :=
by sorry

end simplify_expression_l184_184066


namespace reservoir_percentage_before_storm_l184_184435

variable (total_capacity : ℝ)
variable (water_after_storm : ℝ := 220 + 110)
variable (percentage_after_storm : ℝ := 0.60)
variable (original_contents : ℝ := 220)

theorem reservoir_percentage_before_storm :
  total_capacity = water_after_storm / percentage_after_storm →
  (original_contents / total_capacity) * 100 = 40 :=
by
  sorry

end reservoir_percentage_before_storm_l184_184435


namespace result_of_y_minus_3x_l184_184031

theorem result_of_y_minus_3x (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 :=
sorry

end result_of_y_minus_3x_l184_184031


namespace faye_rows_l184_184688

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (rows_created : ℕ) :
  total_pencils = 12 → pencils_per_row = 4 → rows_created = 3 := by
  sorry

end faye_rows_l184_184688


namespace raft_min_capacity_l184_184992

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l184_184992


namespace sum_of_first_100_terms_l184_184163

theorem sum_of_first_100_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n+2) = a n + 1) : 
  (Finset.sum (Finset.range 100) a) = 2550 :=
sorry

end sum_of_first_100_terms_l184_184163


namespace johns_commute_distance_l184_184842

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end johns_commute_distance_l184_184842


namespace sum_squares_of_six_consecutive_even_eq_1420_l184_184964

theorem sum_squares_of_six_consecutive_even_eq_1420 
  (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) :
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420 :=
by
  sorry

end sum_squares_of_six_consecutive_even_eq_1420_l184_184964


namespace sum_of_angles_FC_correct_l184_184497

noncomputable def circleGeometry (A B C D E F : Point)
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E)
  (arcAB : ℝ) (arcDE : ℝ) : Prop :=
  let arcFull := 360;
  let angleF := 6;  -- Derived from the intersecting chords theorem
  let angleC := 36; -- Derived from the inscribed angle theorem
  arcAB = 60 ∧ arcDE = 72 ∧
  0 ≤ angleF ∧ 0 ≤ angleC ∧
  angleF + angleC = 42

theorem sum_of_angles_FC_correct (A B C D E F : Point) 
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) :
  circleGeometry A B C D E F onCircle 60 72 :=
by
  sorry  -- Proof to be filled

end sum_of_angles_FC_correct_l184_184497


namespace points_per_enemy_l184_184739

theorem points_per_enemy (total_enemies : ℕ) (destroyed_enemies : ℕ) (total_points : ℕ) 
  (h1 : total_enemies = 7)
  (h2 : destroyed_enemies = total_enemies - 2)
  (h3 : destroyed_enemies = 5)
  (h4 : total_points = 40) :
  total_points / destroyed_enemies = 8 :=
by
  sorry

end points_per_enemy_l184_184739


namespace impossible_measure_1_liter_with_buckets_l184_184468

theorem impossible_measure_1_liter_with_buckets :
  ¬ (∃ k l : ℤ, k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1) :=
by
  sorry

end impossible_measure_1_liter_with_buckets_l184_184468


namespace factor_expression_l184_184133

theorem factor_expression (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by
  sorry

end factor_expression_l184_184133


namespace part_I_part_II_l184_184493

def f (x a : ℝ) : ℝ := |2 * x + 1| + |2 * x - a| + a

theorem part_I (x : ℝ) (h₁ : f x 3 > 7) : sorry := sorry

theorem part_II (a : ℝ) (h₂ : ∀ (x : ℝ), f x a ≥ 3) : sorry := sorry

end part_I_part_II_l184_184493


namespace simplify_expr1_simplify_expr2_l184_184132

-- Defining the necessary variables as real numbers for the proof
variables (x y : ℝ)

-- Prove the first expression simplification
theorem simplify_expr1 : 
  (x + 2 * y) * (x - 2 * y) - x * (x + 3 * y) = -4 * y^2 - 3 * x * y :=
  sorry

-- Prove the second expression simplification
theorem simplify_expr2 : 
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4 * x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
  sorry

end simplify_expr1_simplify_expr2_l184_184132


namespace fiona_initial_seat_l184_184380

theorem fiona_initial_seat (greg hannah ian jane kayla lou : Fin 7)
  (greg_final : Fin 7 := greg + 3)
  (hannah_final : Fin 7 := hannah - 2)
  (ian_final : Fin 7 := jane)
  (jane_final : Fin 7 := ian)
  (kayla_final : Fin 7 := kayla + 1)
  (lou_final : Fin 7 := lou - 2)
  (fiona_final : Fin 7) :
  (fiona_final = 0 ∨ fiona_final = 6) →
  ∀ (fiona_initial : Fin 7), 
  (greg_final ≠ fiona_initial ∧ hannah_final ≠ fiona_initial ∧ ian_final ≠ fiona_initial ∧ 
   jane_final ≠ fiona_initial ∧ kayla_final ≠ fiona_initial ∧ lou_final ≠ fiona_initial) →
  fiona_initial = 0 :=
by
  sorry

end fiona_initial_seat_l184_184380


namespace x_squared_plus_y_squared_l184_184462

theorem x_squared_plus_y_squared (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
(h3 : x * y + x + y = 71)
(h4 : x^2 * y + x * y^2 = 880) :
x^2 + y^2 = 146 :=
sorry

end x_squared_plus_y_squared_l184_184462


namespace large_doll_cost_is_8_l184_184602

-- Define the cost of the large monkey doll
def cost_large_doll : ℝ := 8

-- Define the total amount spent
def total_spent : ℝ := 320

-- Define the price difference between large and small dolls
def price_difference : ℝ := 4

-- Define the count difference between buying small dolls and large dolls
def count_difference : ℝ := 40

theorem large_doll_cost_is_8 
    (h1 : total_spent = 320)
    (h2 : ∀ L, L - price_difference = 4)
    (h3 : ∀ L, (total_spent / (L - 4)) = (total_spent / L) + count_difference) :
    cost_large_doll = 8 := 
by 
  sorry

end large_doll_cost_is_8_l184_184602


namespace number_of_buckets_l184_184514

-- Defining the conditions
def total_mackerels : ℕ := 27
def mackerels_per_bucket : ℕ := 3

-- The theorem to prove
theorem number_of_buckets :
  total_mackerels / mackerels_per_bucket = 9 :=
sorry

end number_of_buckets_l184_184514


namespace total_practice_hours_l184_184623

def weekly_practice_hours : ℕ := 4
def weeks_in_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours : (weekly_practice_hours * weeks_in_month) * months = 80 := by
  -- Calculation for weekly practice in hours
  let monthly_hours := weekly_practice_hours * weeks_in_month
  -- Calculation for total practice in hours
  have total_hours : ℕ := monthly_hours * months
  have calculation : total_hours = 80 := 
    by simp [weekly_practice_hours, weeks_in_month, months, monthly_hours, total_hours]
  exact calculation

end total_practice_hours_l184_184623


namespace Rahim_books_l184_184772

variable (x : ℕ) (total_cost : ℕ) (total_books : ℕ) (average_price : ℚ)

def book_problem_conditions : Prop :=
  total_cost = 520 + 248 ∧
  total_books = x + 22 ∧
  average_price = 12

theorem Rahim_books (h : book_problem_conditions x total_cost total_books average_price):
  x = 42 :=
by
  sorry

end Rahim_books_l184_184772


namespace ones_digit_of_p_l184_184309

theorem ones_digit_of_p (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hseq : q = p + 4 ∧ r = p + 8 ∧ s = p + 12) (hpg : p > 5) : (p % 10) = 9 :=
by
  sorry

end ones_digit_of_p_l184_184309


namespace order_numbers_l184_184524

theorem order_numbers : (5 / 2 : ℝ) < (3 : ℝ) ∧ (3 : ℝ) < Real.sqrt (10) := 
by
  sorry

end order_numbers_l184_184524


namespace min_distance_pq_to_c3_l184_184581

def C1_polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 + 8*ρ*Real.cos θ - 6*ρ*Real.sin θ + 24 = 0

def C2_param_eq (θ : ℝ) : ℝ × ℝ :=
  (8 * Real.cos θ, 3 * Real.sin θ)

def P_coords: ℝ × ℝ :=
  (-4, 4)

def C3_line_eq (p : ℝ × ℝ) : Prop :=
  p.1 - 2 * p.2 - 7 = 0

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance_point_to_line (M : ℝ × ℝ) : ℝ :=
  (Real.sqrt 5 / 5) * Real.abs(4 * M.1 / 2 - 3 * M.2 / 2 - 13)

theorem min_distance_pq_to_c3 (θ φ : ℝ) (hcos : Real.cos φ = 4/5) (hsin : Real.sin φ = 3/5) :
  ∃ θ, distance_point_to_line (midpoint P_coords (C2_param_eq θ)) = 8 * Real.sqrt 5 / 5 := sorry

end min_distance_pq_to_c3_l184_184581


namespace alice_steps_l184_184280

noncomputable def num_sticks (n : ℕ) : ℕ :=
  (n + 1 : ℕ) ^ 2

theorem alice_steps (n : ℕ) (h : num_sticks n = 169) : n = 13 :=
by sorry

end alice_steps_l184_184280


namespace men_in_first_group_l184_184381

theorem men_in_first_group (M : ℕ) : (M * 18 = 27 * 24) → M = 36 :=
by
  sorry

end men_in_first_group_l184_184381


namespace page_copy_cost_l184_184349

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end page_copy_cost_l184_184349


namespace melanie_trout_l184_184918

theorem melanie_trout (M : ℕ) (h1 : 2 * M = 16) : M = 8 :=
by
  sorry

end melanie_trout_l184_184918


namespace mimi_spent_on_clothes_l184_184210

theorem mimi_spent_on_clothes :
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  total_spent - total_sneakers_cost = 2600 :=
by
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  show total_spent - total_sneakers_cost = 2600
  sorry

end mimi_spent_on_clothes_l184_184210


namespace even_sum_probability_correct_l184_184099

-- Definition: Calculate probabilities based on the given wheels
def even_probability_wheel_one : ℚ := 1/3
def odd_probability_wheel_one : ℚ := 2/3
def even_probability_wheel_two : ℚ := 1/4
def odd_probability_wheel_two : ℚ := 3/4

-- Probability of both numbers being even
def both_even_probability : ℚ := even_probability_wheel_one * even_probability_wheel_two

-- Probability of both numbers being odd
def both_odd_probability : ℚ := odd_probability_wheel_one * odd_probability_wheel_two

-- Final probability of the sum being even
def even_sum_probability : ℚ := both_even_probability + both_odd_probability

theorem even_sum_probability_correct : even_sum_probability = 7/12 := 
sorry

end even_sum_probability_correct_l184_184099


namespace value_is_correct_l184_184224

-- Define the mean and standard deviation
def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that value = 11.0
theorem value_is_correct : value = 11.0 := by
  sorry

end value_is_correct_l184_184224


namespace no_integer_solutions_l184_184061

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147 :=
by
  sorry

end no_integer_solutions_l184_184061


namespace isosceles_triangle_angle_l184_184507

open Real

theorem isosceles_triangle_angle (x : ℝ) (hx : 0 < x ∧ x < 90) :
  (∃ (α β : ℝ), (α = 11.25 ∧ β = 45) ∧ 
                (sin 3 x = sin x ∧ sin x = sin (5 * x)) ∧ 
                (x = α ∨ x = β)) :=
by 
  assume h,
  sorry

end isosceles_triangle_angle_l184_184507


namespace value_of_m_l184_184708

theorem value_of_m (a b m : ℝ)
    (h1: 2 ^ a = m)
    (h2: 5 ^ b = m)
    (h3: 1 / a + 1 / b = 1 / 2) :
    m = 100 :=
sorry

end value_of_m_l184_184708


namespace gcd_probability_is_one_l184_184955

open Set Nat

theorem gcd_probability_is_one :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (finset.powerset_len 2 (finset.image id S.to_finset)).card
  let non_rel_prime_pairs := 6
  (finset.card (finset.filter (λ (p : Finset ℕ), p.gcdₓ = 1) 
                                (finset.powerset_len 2 (finset.image id S.to_finset)))) / 
  total_pairs = 11 / 14 :=
sorry

end gcd_probability_is_one_l184_184955


namespace simplify_complex_fraction_l184_184778

open Complex

theorem simplify_complex_fraction :
  (⟨2, 2⟩ : ℂ) / (⟨-3, 4⟩ : ℂ) = (⟨-14 / 25, -14 / 25⟩ : ℂ) :=
by
  sorry

end simplify_complex_fraction_l184_184778


namespace domain_of_function_l184_184294

theorem domain_of_function : 
  {x : ℝ | x ≠ 1 ∧ x > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x)} :=
by
  sorry

end domain_of_function_l184_184294


namespace problem_inequality_solution_problem_prove_inequality_l184_184876

-- Function definition for f(x)
def f (x : ℝ) := |2 * x - 3| + |2 * x + 3|

-- Problem 1: Prove the solution set for the inequality f(x) ≤ 8
theorem problem_inequality_solution (x : ℝ) : f x ≤ 8 ↔ -2 ≤ x ∧ x ≤ 2 :=
sorry

-- Problem 2: Prove a + 2b + 3c ≥ 9 given conditions
theorem problem_prove_inequality (a b c : ℝ) (M : ℝ) (h1 : M = 6)
  (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 1 / a + 1 / (2 * b) + 1 / (3 * c) = M / 6) :
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_inequality_solution_problem_prove_inequality_l184_184876


namespace largest_value_of_x_l184_184303

theorem largest_value_of_x (x : ℝ) (h : |x - 8| = 15) : x ≤ 23 :=
by
  sorry -- Proof to be provided

end largest_value_of_x_l184_184303


namespace quadratic_value_at_3_l184_184525

theorem quadratic_value_at_3 (a b c : ℝ) :
  (a * (-2)^2 + b * (-2) + c = -13 / 2) →
  (a * (-1)^2 + b * (-1) + c = -4) →
  (a * 0^2 + b * 0 + c = -2.5) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 2^2 + b * 2 + c = -2.5) →
  (a * 3^2 + b * 3 + c = -4) :=
by
  sorry

end quadratic_value_at_3_l184_184525


namespace B_inter_A_complement_eq_one_l184_184718

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 3}
def A_complement : Set ℕ := I \ A

theorem B_inter_A_complement_eq_one : B ∩ A_complement = {1} := by
  sorry

end B_inter_A_complement_eq_one_l184_184718


namespace ice_cream_amount_l184_184668

/-- Given: 
    Amount of ice cream eaten on Friday night: 3.25 pints
    Total amount of ice cream eaten over both nights: 3.5 pints
    Prove: 
    Amount of ice cream eaten on Saturday night = 0.25 pints -/
theorem ice_cream_amount (friday_night saturday_night total : ℝ) (h_friday : friday_night = 3.25) (h_total : total = 3.5) : 
  saturday_night = total - friday_night → saturday_night = 0.25 :=
by
  intro h
  rw [h_total, h_friday] at h
  simp [h]
  sorry

end ice_cream_amount_l184_184668


namespace triangle_side_range_l184_184638

theorem triangle_side_range (x : ℝ) (h1 : x > 0) (h2 : x + (x + 1) + (x + 2) ≤ 12) :
  1 < x ∧ x ≤ 3 :=
by
  sorry

end triangle_side_range_l184_184638


namespace arithmetic_sequence_a5_l184_184870

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_a5 (a₁ d : ℝ) (h1 : a 2 a₁ d = 2 * a 3 a₁ d + 1) (h2 : a 4 a₁ d = 2 * a 3 a₁ d + 7) :
  a 5 a₁ d = 2 :=
by
  sorry

end arithmetic_sequence_a5_l184_184870


namespace min_chips_to_color_all_cells_l184_184828

def min_chips_needed (n : ℕ) : ℕ := n

theorem min_chips_to_color_all_cells (n : ℕ) :
  min_chips_needed n = n :=
sorry

end min_chips_to_color_all_cells_l184_184828


namespace paint_cost_per_liter_l184_184610

def cost_brush : ℕ := 20
def cost_canvas : ℕ := 3 * cost_brush
def min_liters : ℕ := 5
def total_earning : ℕ := 200
def total_profit : ℕ := 80
def total_cost : ℕ := total_earning - total_profit

theorem paint_cost_per_liter :
  (total_cost = cost_brush + cost_canvas + (5 * 8)) :=
by
  sorry

end paint_cost_per_liter_l184_184610


namespace Tod_drove_time_l184_184950

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end Tod_drove_time_l184_184950


namespace kylie_daisies_l184_184750

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end kylie_daisies_l184_184750


namespace find_set_A_l184_184467

def M : Set ℤ := {1, 3, 5, 7, 9}

def satisfiesCondition (A : Set ℤ) : Prop :=
  A ≠ ∅ ∧
  (∀ a ∈ A, a + 4 ∈ M) ∧
  (∀ a ∈ A, a - 4 ∈ M)

theorem find_set_A : ∃ A : Set ℤ, satisfiesCondition A ∧ A = {5} :=
  by
    sorry

end find_set_A_l184_184467


namespace percentage_increase_l184_184187

theorem percentage_increase (x : ℝ) (h1 : x = 99.9) : 
  ((x - 90) / 90) * 100 = 11 :=
by 
  -- Add the required proof steps here
  sorry

end percentage_increase_l184_184187


namespace quadratic_no_real_roots_l184_184235

theorem quadratic_no_real_roots :
  ∀ x : ℝ, ¬ (x^2 - 2 * x + 3 = 0) :=
by
  assume x,
  sorry

end quadratic_no_real_roots_l184_184235


namespace tan_sum_l184_184471

theorem tan_sum (x y : ℝ)
  (h1 : Real.sin x + Real.sin y = 72 / 65)
  (h2 : Real.cos x + Real.cos y = 96 / 65) : 
  Real.tan x + Real.tan y = 868 / 112 := 
by sorry

end tan_sum_l184_184471


namespace find_f1_plus_gneg1_l184_184461

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom relation : ∀ x : ℝ, f x - g x = (1 / 2) ^ x

-- Proof statement
theorem find_f1_plus_gneg1 : f 1 + g (-1) = -2 :=
by
  -- Proof goes here
  sorry

end find_f1_plus_gneg1_l184_184461


namespace money_distribution_l184_184742

-- Conditions
variable (A B x y : ℝ)
variable (h1 : x + 1/2 * y = 50)
variable (h2 : 2/3 * x + y = 50)

-- Problem statement
theorem money_distribution : x = A → y = B → (x + 1/2 * y = 50 ∧ 2/3 * x + y = 50) :=
by
  intro hx hy
  rw [hx, hy]
  exfalso -- using exfalso to skip proof body
  sorry

end money_distribution_l184_184742


namespace scarlett_initial_oil_amount_l184_184947

theorem scarlett_initial_oil_amount (x : ℝ) (h : x + 0.67 = 0.84) : x = 0.17 :=
by sorry

end scarlett_initial_oil_amount_l184_184947


namespace lucy_deposit_l184_184608

theorem lucy_deposit :
  ∃ D : ℝ, 
    let initial_balance := 65 
    let withdrawal := 4 
    let final_balance := 76 
    initial_balance + D - withdrawal = final_balance ∧ D = 15 :=
by
  -- sorry skips the proof
  sorry

end lucy_deposit_l184_184608


namespace power_sum_l184_184677

theorem power_sum
: (-2)^(2005) + (-2)^(2006) = 2^(2005) := by
  sorry

end power_sum_l184_184677


namespace baker_bakes_25_hours_per_week_mon_to_fri_l184_184651

-- Define the conditions
def loaves_per_hour_per_oven := 5
def number_of_ovens := 4
def weekend_baking_hours_per_day := 2
def total_weeks := 3
def total_loaves := 1740

-- Calculate the loaves per hour
def loaves_per_hour := loaves_per_hour_per_oven * number_of_ovens

-- Calculate the weekend baking hours in one week
def weekend_baking_hours_per_week := weekend_baking_hours_per_day * 2

-- Calculate the loaves baked on weekends in one week
def loaves_on_weekends_per_week := loaves_per_hour * weekend_baking_hours_per_week

-- Calculate the total loaves baked on weekends in 3 weeks
def loaves_on_weekends_total := loaves_on_weekends_per_week * total_weeks

-- Calculate the loaves baked from Monday to Friday in 3 weeks
def loaves_on_weekdays_total := total_loaves - loaves_on_weekends_total

-- Calculate the total hours baked from Monday to Friday in 3 weeks
def weekday_baking_hours_total := loaves_on_weekdays_total / loaves_per_hour

-- Calculate the number of hours baked from Monday to Friday in one week
def weekday_baking_hours_per_week := weekday_baking_hours_total / total_weeks

-- Proof statement
theorem baker_bakes_25_hours_per_week_mon_to_fri :
  weekday_baking_hours_per_week = 25 :=
by
  sorry

end baker_bakes_25_hours_per_week_mon_to_fri_l184_184651


namespace students_in_sixth_level_l184_184211

theorem students_in_sixth_level (S : ℕ)
  (h1 : ∃ S₄ : ℕ, S₄ = 4 * S)
  (h2 : ∃ S₇ : ℕ, S₇ = 2 * (4 * S))
  (h3 : S + 4 * S + 2 * (4 * S) = 520) :
  S = 40 :=
by
  sorry

end students_in_sixth_level_l184_184211


namespace sqrt_meaningful_iff_ge_eight_l184_184895

theorem sqrt_meaningful_iff_ge_eight (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 8)) ↔ x ≥ 8 := by
  sorry

end sqrt_meaningful_iff_ge_eight_l184_184895


namespace farmer_pays_per_acre_per_month_l184_184109

-- Define the conditions
def total_payment : ℕ := 600
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Define the problem to prove
theorem farmer_pays_per_acre_per_month :
  length_of_plot * width_of_plot / square_feet_per_acre > 0 ∧
  total_payment / (length_of_plot * width_of_plot / square_feet_per_acre) = 60 :=
by
  -- skipping the actual proof for now
  sorry

end farmer_pays_per_acre_per_month_l184_184109


namespace triangle_perimeter_ratio_l184_184551

theorem triangle_perimeter_ratio : 
  let side := 10
  let hypotenuse := Real.sqrt (side^2 + (side / 2) ^ 2)
  let triangle_perimeter := side + (side / 2) + hypotenuse
  let square_perimeter := 4 * side
  (triangle_perimeter / square_perimeter) = (15 + Real.sqrt 125) / 40 := 
by
  sorry

end triangle_perimeter_ratio_l184_184551


namespace group_contains_2007_l184_184613

theorem group_contains_2007 : 
  ∃ k, 2007 ∈ {a | (k * (k + 1)) / 2 < a ∧ a ≤ ((k + 1) * (k + 2)) / 2} ∧ k = 45 :=
by sorry

end group_contains_2007_l184_184613


namespace number_of_integers_satisfying_inequality_l184_184152

theorem number_of_integers_satisfying_inequality :
  ∃ S : Finset ℤ, (∀ x ∈ S, x^2 < 9 * x) ∧ S.card = 8 :=
by
  sorry

end number_of_integers_satisfying_inequality_l184_184152


namespace no_three_digit_numbers_meet_conditions_l184_184721

theorem no_three_digit_numbers_meet_conditions :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (n % 10 = 5) ∧ (n % 10 = 0) → false := 
by {
  sorry
}

end no_three_digit_numbers_meet_conditions_l184_184721


namespace find_original_strength_l184_184419

variable (original_strength : ℕ)
variable (total_students : ℕ := original_strength + 12)
variable (original_avg_age : ℕ := 40)
variable (new_students : ℕ := 12)
variable (new_students_avg_age : ℕ := 32)
variable (new_avg_age_reduction : ℕ := 4)
variable (new_avg_age : ℕ := original_avg_age - new_avg_age_reduction)

theorem find_original_strength (h : (original_avg_age * original_strength + new_students * new_students_avg_age) / total_students = new_avg_age) :
  original_strength = 12 := 
sorry

end find_original_strength_l184_184419


namespace Jim_paycheck_after_deductions_l184_184480

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end Jim_paycheck_after_deductions_l184_184480


namespace raft_minimum_capacity_l184_184995

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l184_184995


namespace pairs_satisfy_condition_l184_184293

theorem pairs_satisfy_condition (a b : ℝ) :
  (∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) →
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a_int b_int : ℤ, a = a_int ∧ b = b_int)) :=
by
  sorry

end pairs_satisfy_condition_l184_184293


namespace julia_played_with_kids_on_tuesday_l184_184351

theorem julia_played_with_kids_on_tuesday (total: ℕ) (monday: ℕ) (tuesday: ℕ) 
  (h1: total = 18) (h2: monday = 4) : 
  tuesday = (total - monday) :=
by
  sorry

end julia_played_with_kids_on_tuesday_l184_184351


namespace solution_set_inequality_l184_184943

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := 
by sorry

end solution_set_inequality_l184_184943


namespace reflect_point_value_l184_184939

theorem reflect_point_value (mx b : ℝ) 
  (start end_ : ℝ × ℝ)
  (Hstart : start = (2, 3))
  (Hend : end_ = (10, 7))
  (Hreflection : ∃ m b: ℝ, (end_.fst, end_.snd) = 
              (2 * ((5 / 2) - (1 / 2) * 3 * m - b), 2 * ((5 / 2) + (1 / 2) * 3)) ∧ m = -2)
  : m + b = 15 :=
sorry

end reflect_point_value_l184_184939


namespace tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l184_184008

theorem tanx_eq_2_sin2cos2 (x : ℝ) (h : Real.tan x = 2) : 
  (2 / 3) * (Real.sin x) ^ 2 + (1 / 4) * (Real.cos x) ^ 2 = 7 / 12 := 
by 
  sorry

theorem tanx_eq_2_cos_sin_ratio (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3 := 
by 
  sorry

end tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l184_184008


namespace positive_integer_a_l184_184341

theorem positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ (k : ℤ), (2 * a + 8) = k * (a + 1)) :
  a = 1 ∨ a = 2 ∨ a = 5 :=
by sorry

end positive_integer_a_l184_184341


namespace find_number_l184_184822

theorem find_number (x : ℝ) (h : 20 * (x / 5) = 40) : x = 10 :=
by
  sorry

end find_number_l184_184822


namespace min_value_is_nine_l184_184355

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end min_value_is_nine_l184_184355


namespace max_min_sum_eq_two_l184_184466

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 + Real.sqrt 2 * Real.sin (x + Real.pi / 4)) / (2 * x ^ 2 + Real.cos x)

theorem max_min_sum_eq_two (a b : ℝ) (h_max : ∀ x, f x ≤ a) (h_min : ∀ x, b ≤ f x) (h_max_val : ∃ x, f x = a) (h_min_val : ∃ x, f x = b) :
  a + b = 2 := 
sorry

end max_min_sum_eq_two_l184_184466


namespace unused_sector_angle_l184_184157

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)
noncomputable def central_angle (r base_circumference : ℝ) : ℝ := (base_circumference / (2 * Real.pi * r)) * 360
noncomputable def unused_angle (total_degrees used_angle : ℝ) : ℝ := total_degrees - used_angle

theorem unused_sector_angle (R : ℝ)
  (cone_radius := 15)
  (cone_volume := 675 * Real.pi)
  (total_circumference := 2 * Real.pi * R)
  (cone_height := 9)
  (slant_height := Real.sqrt (cone_radius^2 + cone_height^2))
  (base_circumference := 2 * Real.pi * cone_radius)
  (used_angle := central_angle slant_height base_circumference) :

  unused_angle 360 used_angle = 164.66 := by
  sorry

end unused_sector_angle_l184_184157


namespace third_discount_l184_184542

noncomputable def find_discount (P S firstDiscount secondDiscount D3 : ℝ) : Prop :=
  S = P * (1 - firstDiscount / 100) * (1 - secondDiscount / 100) * (1 - D3 / 100)

theorem third_discount (P : ℝ) (S : ℝ) (firstDiscount : ℝ) (secondDiscount : ℝ) (D3 : ℝ) 
  (HP : P = 9649.12) (HS : S = 6600)
  (HfirstDiscount : firstDiscount = 20) (HsecondDiscount : secondDiscount = 10) : 
  find_discount P S firstDiscount secondDiscount 5.01 :=
  by
  rw [HP, HS, HfirstDiscount, HsecondDiscount]
  sorry

end third_discount_l184_184542


namespace andrey_gifts_l184_184055

theorem andrey_gifts :
  ∃ (n : ℕ), ∀ (a : ℕ), n(n-2) = a(n-1) + 16 ∧ n = 18 :=
by {
  sorry
}

end andrey_gifts_l184_184055


namespace bacteria_growth_final_count_l184_184816

theorem bacteria_growth_final_count (initial_count : ℕ) (t : ℕ) 
(h1 : initial_count = 10) 
(h2 : t = 7) 
(h3 : ∀ n : ℕ, (n * 60) = t * 60 → 2 ^ n = 128) : 
(initial_count * 2 ^ t) = 1280 := 
by
  sorry

end bacteria_growth_final_count_l184_184816


namespace find_h_l184_184562

def infinite_sqrt_series (b : ℝ) : ℝ := sorry -- Placeholder for infinite series sqrt(b + sqrt(b + ...))

def diamond (a b : ℝ) : ℝ :=
  a^2 + infinite_sqrt_series b

theorem find_h (h : ℝ) : diamond 3 h = 12 → h = 6 :=
by
  intro h_condition
  -- Further steps will be used during proof
  sorry

end find_h_l184_184562


namespace boxes_sold_l184_184702

def case_size : ℕ := 12
def remaining_boxes : ℕ := 7

theorem boxes_sold (sold_boxes : ℕ) : ∃ n : ℕ, sold_boxes = n * case_size + remaining_boxes :=
sorry

end boxes_sold_l184_184702


namespace maximize_distance_l184_184852

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end maximize_distance_l184_184852


namespace fruit_seller_sp_l184_184270

theorem fruit_seller_sp (CP SP : ℝ)
    (h1 : SP = 0.75 * CP)
    (h2 : 19.93 = 1.15 * CP) :
    SP = 13.00 :=
by
  sorry

end fruit_seller_sp_l184_184270


namespace number_of_grey_birds_l184_184404

variable (G : ℕ)

def grey_birds_condition1 := G + 6
def grey_birds_condition2 := G / 2

theorem number_of_grey_birds
  (H1 : G + 6 + G / 2 = 66) :
  G = 40 :=
by
  sorry

end number_of_grey_birds_l184_184404


namespace rectangle_area_l184_184432

theorem rectangle_area (side_length width length : ℝ) (h_square_area : side_length^2 = 36)
  (h_width : width = side_length) (h_length : length = 2.5 * width) :
  width * length = 90 :=
by 
  sorry

end rectangle_area_l184_184432


namespace area_enclosed_by_region_l184_184563

theorem area_enclosed_by_region :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 6*y - 3 = 0) → 
  (∃ r : ℝ, r = 4 ∧ area = (π * r^2)) :=
by
  -- Starting proof setup
  sorry

end area_enclosed_by_region_l184_184563


namespace option_A_option_B_option_C_option_D_verify_options_l184_184807

open Real

-- Option A: Prove the maximum value of x(6-x) given 0 < x < 6 is 9.
theorem option_A (x : ℝ) (h1 : 0 < x) (h2 : x < 6) : 
  ∃ (max_value : ℝ), max_value = 9 ∧ ∀(y : ℝ), 0 < y ∧ y < 6 → y * (6 - y) ≤ max_value :=
sorry

-- Option B: Prove the minimum value of x^2 + 1/(x^2 + 3) for x in ℝ is not -1.
theorem option_B (x : ℝ) : ¬(∃ (min_value : ℝ), min_value = -1 ∧ ∀(y : ℝ), (y ^ 2) + 1 / (y ^ 2 + 3) ≥ min_value) :=
sorry

-- Option C: Prove the maximum value of xy given x + 2y + xy = 6 and x, y > 0 is 2.
theorem option_C (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y + x * y = 6) : 
  ∃ (max_value : ℝ), max_value = 2 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 2 * v + u * v = 6 → u * v ≤ max_value :=
sorry

-- Option D: Prove the minimum value of 2x + y given x + 4y + 4 = xy and x, y > 0 is 17.
theorem option_D (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 4 * y + 4 = x * y) : 
  ∃ (min_value : ℝ), min_value = 17 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 4 * v + 4 = u * v → 2 * u + v ≥ min_value :=
sorry

-- Combine to verify which options are correct
theorem verify_options (A_correct B_correct C_correct D_correct : Prop) :
  A_correct = true ∧ B_correct = false ∧ C_correct = true ∧ D_correct = true :=
sorry

end option_A_option_B_option_C_option_D_verify_options_l184_184807


namespace distinct_integers_sum_to_32_l184_184045

theorem distinct_integers_sum_to_32 
  (p q r s t : ℤ)
  (h_diff : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_eq : (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120) : 
  p + q + r + s + t = 32 := 
by 
  sorry

end distinct_integers_sum_to_32_l184_184045


namespace min_capacity_for_raft_l184_184983

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l184_184983


namespace ratio_of_15th_term_l184_184489

theorem ratio_of_15th_term (a d b e : ℤ) :
  (∀ n : ℕ, (n * (2 * a + (n - 1) * d)) / (n * (2 * b + (n - 1) * e)) = (7 * n^2 + 1) / (4 * n^2 + 27)) →
  (a + 14 * d) / (b + 14 * e) = 7 / 4 :=
by sorry

end ratio_of_15th_term_l184_184489


namespace andrey_gifts_l184_184058

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l184_184058


namespace benny_gave_seashells_l184_184284

theorem benny_gave_seashells (original_seashells : ℕ) (remaining_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 66) 
  (h2 : remaining_seashells = 14) 
  (h3 : original_seashells - remaining_seashells = given_seashells) : 
  given_seashells = 52 := 
by
  sorry

end benny_gave_seashells_l184_184284


namespace even_sin_condition_l184_184077

theorem even_sin_condition (φ : ℝ) : 
  (φ = -Real.pi / 2 → ∀ x : ℝ, sin (x + φ) = sin (-(x + φ))) ∧ 
  (∀ x : ℝ, sin (x + φ) = sin (-(x + φ)) → ∃ k : ℤ, φ = k * Real.pi + Real.pi / 2) :=
by
  sorry

end even_sin_condition_l184_184077


namespace harmonica_value_l184_184641

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end harmonica_value_l184_184641


namespace count_obtuse_triangle_values_k_l184_184397

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  if a ≥ b ∧ a ≥ c then a * a > b * b + c * c 
  else if b ≥ a ∧ b ≥ c then b * b > a * a + c * c
  else c * c > a * a + b * b

theorem count_obtuse_triangle_values_k :
  ∃! (k : ℕ), is_triangle 8 18 k ∧ is_obtuse_triangle 8 18 k :=
sorry

end count_obtuse_triangle_values_k_l184_184397


namespace soccer_team_probability_l184_184277

theorem soccer_team_probability :
  let total_players := 12
  let forwards := 6
  let defenders := 6
  let total_ways := Nat.choose total_players 2
  let defender_ways := Nat.choose defenders 2
  ∃ p : ℚ, p = defender_ways / total_ways ∧ p = 5 / 22 :=
sorry

end soccer_team_probability_l184_184277


namespace compare_diff_functions_l184_184017

variable {R : Type*} [LinearOrderedField R]
variable {f g : R → R}
variable (h_fg : ∀ x, f' x > g' x)
variable {x1 x2 : R}

theorem compare_diff_functions (h : x1 < x2) : f x1 - f x2 < g x1 - g x2 :=
  sorry

end compare_diff_functions_l184_184017


namespace parabola_vertex_l184_184934

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end parabola_vertex_l184_184934


namespace horner_multiplications_additions_l184_184517

def f (x : ℝ) : ℝ := 6 * x^6 + 5

def x : ℝ := 2

theorem horner_multiplications_additions :
  (6 : ℕ) = 6 ∧ (6 : ℕ) = 6 := 
by 
  sorry

end horner_multiplications_additions_l184_184517


namespace minimize_f_sum_l184_184858

noncomputable def f (x : ℝ) : ℝ := x^2 - 8*x + 10

theorem minimize_f_sum :
  ∃ a₁ : ℝ, (∀ a₂ a₃ : ℝ, a₂ = a₁ + 1 ∧ a₃ = a₁ + 2 →
    f(a₁) + f(a₂) + f(a₃) = 3 * a₁^2 - 18 * a₁ + 30) →
    (∀ b₁ : ℝ, (∀ b₂ b₃ : ℝ, b₂ = b₁ + 1 ∧ b₃ = b₁ + 2 →
      f(b₁) + f(b₂) + f(b₃) ≥ f(a₁) + f(a₂) + f(a₃)) ∧ a₁ = 3) :=
by 
  sorry

end minimize_f_sum_l184_184858


namespace remainder_of_polynomial_l184_184305

-- Define the polynomial
def P (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

-- State the theorem
theorem remainder_of_polynomial (x : ℝ) : P 3 = 50 := sorry

end remainder_of_polynomial_l184_184305


namespace negative_expression_l184_184940

noncomputable def U : ℝ := -2.5
noncomputable def V : ℝ := -0.8
noncomputable def W : ℝ := 0.4
noncomputable def X : ℝ := 1.0
noncomputable def Y : ℝ := 2.2

theorem negative_expression :
  (U - V < 0) ∧ ¬(U * V < 0) ∧ ¬((X / V) * U < 0) ∧ ¬(W / (U * V) < 0) ∧ ¬((X + Y) / W < 0) :=
by
  sorry

end negative_expression_l184_184940


namespace eq_inf_solutions_l184_184004

theorem eq_inf_solutions (a b : ℝ) : 
    (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -(4 / 3) * a := by
  sorry

end eq_inf_solutions_l184_184004


namespace tank_capacity_l184_184543

variable (c w : ℕ)

-- Conditions
def initial_fraction (w c : ℕ) : Prop := w = c / 7
def final_fraction (w c : ℕ) : Prop := (w + 2) = c / 5

-- The theorem statement
theorem tank_capacity : 
  initial_fraction w c → 
  final_fraction w c → 
  c = 35 := 
by
  sorry  -- indicates that the proof is not provided

end tank_capacity_l184_184543


namespace reciprocal_of_neg_1_point_5_l184_184634

theorem reciprocal_of_neg_1_point_5 : (1 / (-1.5) = -2 / 3) :=
by
  sorry

end reciprocal_of_neg_1_point_5_l184_184634


namespace people_visited_neither_l184_184104

-- Definitions based on conditions
def total_people : ℕ := 60
def visited_iceland : ℕ := 35
def visited_norway : ℕ := 23
def visited_both : ℕ := 31

-- Theorem statement
theorem people_visited_neither :
  total_people - (visited_iceland + visited_norway - visited_both) = 33 :=
by sorry

end people_visited_neither_l184_184104


namespace maximize_distance_l184_184853

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end maximize_distance_l184_184853


namespace ball_cost_l184_184929

theorem ball_cost (B C : ℝ) (h1 : 7 * B + 6 * C = 3800) (h2 : 3 * B + 5 * C = 1750) (hb : B = 500) : C = 50 :=
by
  sorry

end ball_cost_l184_184929


namespace totalAttendees_l184_184093

def numberOfBuses : ℕ := 8
def studentsPerBus : ℕ := 45
def chaperonesList : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]

theorem totalAttendees : 
    numberOfBuses * studentsPerBus + chaperonesList.sum = 389 := 
by
  sorry

end totalAttendees_l184_184093


namespace johns_initial_money_l184_184672

theorem johns_initial_money (X : ℝ) 
  (h₁ : (1 / 2) * X + (1 / 3) * X + (1 / 10) * X + 10 = X) : X = 150 :=
sorry

end johns_initial_money_l184_184672


namespace max_area_of_pen_l184_184311

theorem max_area_of_pen (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (x : ℝ), (3 * x + x = 60) ∧ (2 * x * x = 450) :=
by
  -- This theorem states that there exists an x such that
  -- the total perimeter with internal divider equals 60,
  -- and the total area of the two squares equals 450.
  use 15
  sorry

end max_area_of_pen_l184_184311


namespace initial_money_l184_184062

theorem initial_money {M : ℝ} (h : (M - 10) - (M - 10) / 4 = 15) : M = 30 :=
sorry

end initial_money_l184_184062


namespace extreme_point_a_zero_l184_184724

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end extreme_point_a_zero_l184_184724


namespace damian_serena_passing_times_l184_184838

/-- 
  Damian and Serena are running on a circular track for 40 minutes.
  Damian runs clockwise at 220 m/min on the inner lane with a radius of 45 meters.
  Serena runs counterclockwise at 260 m/min on the outer lane with a radius of 55 meters.
  They start on the same radial line.
  Prove that they pass each other exactly 184 times in 40 minutes. 
-/
theorem damian_serena_passing_times
  (time_run : ℕ)
  (damian_speed : ℕ)
  (serena_speed : ℕ)
  (damian_radius : ℝ)
  (serena_radius : ℝ)
  (start_same_line : Prop) :
  time_run = 40 →
  damian_speed = 220 →
  serena_speed = 260 →
  damian_radius = 45 →
  serena_radius = 55 →
  start_same_line →
  ∃ n : ℕ, n = 184 :=
by
  sorry

end damian_serena_passing_times_l184_184838


namespace no_unique_solution_l184_184575

theorem no_unique_solution (d : ℝ) (x y : ℝ) :
  (3 * (3 * x + 4 * y) = 36) ∧ (9 * x + 12 * y = d) ↔ d ≠ 36 := sorry

end no_unique_solution_l184_184575


namespace chris_money_before_birthday_l184_184290

variables {x : ℕ} -- Assuming we are working with natural numbers (non-negative integers)

-- Conditions
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Question
theorem chris_money_before_birthday : x = total_money_now - (grandmother_money + aunt_uncle_money + parents_money) :=
by
  sorry

end chris_money_before_birthday_l184_184290


namespace evaluate_expression_l184_184300

-- Define the ceiling of square roots for the given numbers
def ceil_sqrt_3 := 2
def ceil_sqrt_27 := 6
def ceil_sqrt_243 := 16

-- Main theorem statement
theorem evaluate_expression :
  ceil_sqrt_3 + ceil_sqrt_27 * 2 + ceil_sqrt_243 = 30 :=
by
  -- Sorry to indicate that the proof is skipped
  sorry

end evaluate_expression_l184_184300


namespace vector_expression_identity_l184_184469

variables (E : Type) [AddCommGroup E] [Module ℝ E]
variables (e1 e2 : E)
variables (a b : E)
variables (cond1 : a = (3 : ℝ) • e1 - (2 : ℝ) • e2) (cond2 : b = (e2 - (2 : ℝ) • e1))

theorem vector_expression_identity :
  (1 / 3 : ℝ) • a + b + a - (3 / 2 : ℝ) • b + 2 • b - a = -2 • e1 + (5 / 6 : ℝ) • e2 :=
sorry

end vector_expression_identity_l184_184469


namespace part_one_part_two_l184_184204

-- Definitions for the propositions
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1)

-- Theorems for the answers
theorem part_one (m : ℝ) : ¬ proposition_p m → m < 1 :=
by sorry

theorem part_two (m : ℝ) : ¬ (proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) → m < 1 ∨ (4 ≤ m ∧ m ≤ 6) :=
by sorry

end part_one_part_two_l184_184204


namespace correct_expression_l184_184806

theorem correct_expression (a b : ℝ) : (a - b) * (b + a) = a^2 - b^2 :=
by
  sorry

end correct_expression_l184_184806


namespace sum_of_first_n_odd_integers_eq_169_l184_184260

theorem sum_of_first_n_odd_integers_eq_169 (n : ℕ) 
  (h : n^2 = 169) : n = 13 :=
by sorry

end sum_of_first_n_odd_integers_eq_169_l184_184260


namespace total_food_for_guinea_pigs_l184_184776

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end total_food_for_guinea_pigs_l184_184776


namespace find_larger_number_l184_184509

theorem find_larger_number (a b : ℕ) (h1 : a + b = 96) (h2 : a = b + 12) : a = 54 :=
sorry

end find_larger_number_l184_184509


namespace factorization_correct_l184_184143

theorem factorization_correct (x y : ℝ) :
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2) * (x^2 - 3*y + 2) :=
by
  sorry

end factorization_correct_l184_184143


namespace tennis_tournament_rounds_needed_l184_184970

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end tennis_tournament_rounds_needed_l184_184970


namespace below_zero_notation_l184_184180

def celsius_above (x : ℤ) : String := "+" ++ toString x ++ "°C"
def celsius_below (x : ℤ) : String := "-" ++ toString x ++ "°C"

theorem below_zero_notation (h₁ : celsius_above 5 = "+5°C")
  (h₂ : ∀ x : ℤ, x > 0 → celsius_above x = "+" ++ toString x ++ "°C")
  (h₃ : ∀ x : ℤ, x > 0 → celsius_below x = "-" ++ toString x ++ "°C") :
  celsius_below 3 = "-3°C" :=
sorry

end below_zero_notation_l184_184180


namespace geometric_sequence_common_ratio_l184_184706

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 5/4) 
  (h_sequence : ∀ n, a n = a 1 * q ^ (n - 1)) : 
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l184_184706


namespace inequality_holds_iff_b_lt_a_l184_184442

theorem inequality_holds_iff_b_lt_a (a b : ℝ) :
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by
  sorry

end inequality_holds_iff_b_lt_a_l184_184442


namespace number_of_participants_2005_l184_184195

variable (participants : ℕ → ℕ)
variable (n : ℕ)

-- Conditions
def initial_participants := participants 2001 = 1000
def increase_till_2003 := ∀ n, 2001 ≤ n ∧ n ≤ 2003 → participants (n + 1) = 2 * participants n
def increase_from_2004 := ∀ n, n ≥ 2004 → participants (n + 1) = 2 * participants n + 500

-- Proof problem
theorem number_of_participants_2005 :
    initial_participants participants →
    increase_till_2003 participants →
    increase_from_2004 participants →
    participants 2005 = 17500 :=
by sorry

end number_of_participants_2005_l184_184195


namespace minimize_expression_l184_184307

theorem minimize_expression (x : ℝ) : 3 * x^2 - 12 * x + 1 ≥ 3 * 2^2 - 12 * 2 + 1 :=
by sorry

end minimize_expression_l184_184307


namespace find_roots_of_polynomial_l184_184696

theorem find_roots_of_polynomial :
  (∃ (a b : ℝ), 
    Multiplicity (polynomial.C a) (polynomial.C (Real.ofRat 2)) = 2 ∧ 
    Multiplicity (polynomial.C b) (polynomial.C (Real.ofRat 1)) = 1) ∧ 
  (x^3 - 7 * x^2 + 14 * x - 8 = 
    (x - 1) * (x - 2)^2) := sorry

end find_roots_of_polynomial_l184_184696


namespace product_equation_l184_184400

/-- Given two numbers x and y such that x + y = 20 and x - y = 4,
    the product of three times the larger number and the smaller number is 288. -/
theorem product_equation (x y : ℕ) (h1 : x + y = 20) (h2 : x - y = 4) (h3 : x > y) : 3 * x * y = 288 := 
sorry

end product_equation_l184_184400


namespace find_other_root_l184_184580

theorem find_other_root (a b : ℝ) (h₁ : 3^2 + 3 * a - 2 * a = 0) (h₂ : ∀ x, x^2 + a * x - 2 * a = 0 → (x = 3 ∨ x = b)) :
  b = 6 := 
sorry

end find_other_root_l184_184580


namespace minimize_difference_l184_184443

open BigOperators

/-- Given 16 numbers 1/2002, 1/2003, ..., 1/2017, partition them into two groups A and B such that 
the absolute difference of their sums is minimized. -/
theorem minimize_difference :
  let numbers := (list.range 16).map (λ i, 1 / (2002 + i : ℝ)),
      group1 := [0, 2, 4, 6, 8, 10, 12, 14].map (numbers.get ∘ id),
      group2 := [1, 3, 5, 7, 9, 11, 13, 15].map (numbers.get ∘ id),
      A := group1.sum,
      B := group2.sum
  in |A - B| = minimized :=
begin
  sorry
end

end minimize_difference_l184_184443


namespace mike_first_job_earnings_l184_184368

theorem mike_first_job_earnings (total_wages : ℕ) (hours_second_job : ℕ) (pay_rate_second_job : ℕ) 
  (second_job_earnings := hours_second_job * pay_rate_second_job) 
  (first_job_earnings := total_wages - second_job_earnings) :
  total_wages = 160 → hours_second_job = 12 → pay_rate_second_job = 9 → first_job_earnings = 52 := 
by 
  intros h₁ h₂ h₃ 
  unfold first_job_earnings second_job_earnings
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end mike_first_job_earnings_l184_184368


namespace members_who_play_both_sports_l184_184105

theorem members_who_play_both_sports 
  (N B T Neither BT : ℕ) 
  (h1 : N = 27)
  (h2 : B = 17)
  (h3 : T = 19)
  (h4 : Neither = 2)
  (h5 : BT = B + T - N + Neither) : 
  BT = 11 := 
by 
  have h6 : 17 + 19 - 27 + 2 = 11 := by norm_num
  rw [h2, h3, h1, h4, h6] at h5
  exact h5

end members_who_play_both_sports_l184_184105


namespace find_F_neg_a_l184_184421

-- Definitions of odd functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of F
def F (f g : ℝ → ℝ) (x : ℝ) := 3 * f x + 5 * g x + 2

theorem find_F_neg_a (f g : ℝ → ℝ) (a : ℝ)
  (hf : is_odd f) (hg : is_odd g) (hFa : F f g a = 3) : F f g (-a) = 1 :=
by
  sorry

end find_F_neg_a_l184_184421


namespace probability_of_p_satisfying_equation_l184_184179

theorem probability_of_p_satisfying_equation :
  (∃ (p : ℤ), 1 ≤ p ∧ p ≤ 20 ∧ ∃ (q : ℤ), p * q - 5 * p - 3 * q = -6) →
  rat.mk 4 20 = rat.mk 1 5 :=
begin
  sorry
end

end probability_of_p_satisfying_equation_l184_184179


namespace max_time_for_taxiing_is_15_l184_184927

-- Declare the function representing the distance traveled by the plane with respect to time
def distance (t : ℝ) : ℝ := 60 * t - 2 * t ^ 2

-- The main theorem stating the maximum time s the plane uses for taxiing
theorem max_time_for_taxiing_is_15 : ∃ s, ∀ t, distance t ≤ distance s ∧ s = 15 :=
by
  sorry

end max_time_for_taxiing_is_15_l184_184927


namespace at_least_one_not_less_than_two_l184_184491

theorem at_least_one_not_less_than_two
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (2 ≤ (y / x + y / z)) ∨ (2 ≤ (z / x + z / y)) ∨ (2 ≤ (x / z + x / y)) :=
sorry

end at_least_one_not_less_than_two_l184_184491


namespace find_2a_plus_b_l184_184759

open Real

theorem find_2a_plus_b (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
    (h1 : 4 * (cos a)^3 - 3 * (cos b)^3 = 2) 
    (h2 : 4 * cos (2 * a) + 3 * cos (2 * b) = 1) : 
    2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l184_184759


namespace total_practice_hours_l184_184621

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l184_184621


namespace factorization_of_M_l184_184687

theorem factorization_of_M :
  ∀ (x y z : ℝ), x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = 
  (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end factorization_of_M_l184_184687


namespace find_f4_l184_184312

noncomputable def f : ℝ → ℝ := sorry

theorem find_f4 (hf_odd : ∀ x : ℝ, f (-x) = -f x)
                (hf_property : ∀ x : ℝ, f (x + 2) = -f x) :
  f 4 = 0 :=
sorry

end find_f4_l184_184312


namespace exp_pi_gt_pi_exp_l184_184526

theorem exp_pi_gt_pi_exp (h : Real.pi > Real.exp 1) : Real.exp Real.pi > Real.pi ^ Real.exp 1 := by
  sorry

end exp_pi_gt_pi_exp_l184_184526


namespace number_of_yellow_marbles_l184_184487

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end number_of_yellow_marbles_l184_184487


namespace two_digit_integer_eq_55_l184_184591

theorem two_digit_integer_eq_55
  (c : ℕ)
  (h1 : c / 10 + c % 10 = 10)
  (h2 : (c / 10) * (c % 10) = 25) :
  c = 55 :=
  sorry

end two_digit_integer_eq_55_l184_184591


namespace total_go_stones_correct_l184_184091

-- Definitions based on the problem's conditions
def stones_per_bundle : Nat := 10
def num_bundles : Nat := 3
def white_stones : Nat := 16

-- A function that calculates the total number of go stones
def total_go_stones : Nat :=
  num_bundles * stones_per_bundle + white_stones

-- The theorem we want to prove
theorem total_go_stones_correct : total_go_stones = 46 :=
by
  sorry

end total_go_stones_correct_l184_184091


namespace kangaroo_chase_l184_184732

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end kangaroo_chase_l184_184732


namespace line_segments_cannot_form_triangle_l184_184513

theorem line_segments_cannot_form_triangle (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 7 = 21)
    (h3 : ∀ n, a n < a (n+1)) (h4 : ∀ i j k, a i + a j ≤ a k) :
    a 6 = 13 :=
    sorry

end line_segments_cannot_form_triangle_l184_184513


namespace original_length_l184_184652

-- Definitions based on conditions
def length_sawed_off : ℝ := 0.33
def remaining_length : ℝ := 0.08

-- The problem statement translated to a Lean 4 theorem
theorem original_length (L : ℝ) (h1 : L = length_sawed_off + remaining_length) : 
  L = 0.41 :=
by
  sorry

end original_length_l184_184652


namespace min_speed_to_arrive_before_cara_l184_184533

theorem min_speed_to_arrive_before_cara (d : ℕ) (sc : ℕ) (tc : ℕ) (sd : ℕ) (td : ℕ) (hd : ℕ) :
  d = 180 ∧ sc = 30 ∧ tc = d / sc ∧ hd = 1 ∧ td = tc - hd ∧ sd = d / td ∧ (36 < sd) :=
sorry

end min_speed_to_arrive_before_cara_l184_184533


namespace real_solution_x_condition_l184_184137

theorem real_solution_x_condition (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
by
  sorry

end real_solution_x_condition_l184_184137


namespace raft_minimum_capacity_l184_184994

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l184_184994


namespace karl_drove_420_miles_l184_184488

theorem karl_drove_420_miles :
  ∀ (car_mileage_per_gallon : ℕ)
    (tank_capacity : ℕ)
    (initial_drive_miles : ℕ)
    (gas_purchased : ℕ)
    (destination_tank_fraction : ℚ),
    car_mileage_per_gallon = 30 →
    tank_capacity = 16 →
    initial_drive_miles = 420 →
    gas_purchased = 10 →
    destination_tank_fraction = 3 / 4 →
    initial_drive_miles + (destination_tank_fraction * tank_capacity - (tank_capacity - (initial_drive_miles / car_mileage_per_gallon)) + gas_purchased) * car_mileage_per_gallon = 420 :=
by
  intros car_mileage_per_gallon tank_capacity initial_drive_miles gas_purchased destination_tank_fraction
  intro h1 -- car_mileage_per_gallon = 30
  intro h2 -- tank_capacity = 16
  intro h3 -- initial_drive_miles = 420
  intro h4 -- gas_purchased = 10
  intro h5 -- destination_tank_fraction = 3 / 4
  sorry

end karl_drove_420_miles_l184_184488


namespace parabola_intersects_x_axis_expression_l184_184014

theorem parabola_intersects_x_axis_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 2017 = 2018 := 
by 
  sorry

end parabola_intersects_x_axis_expression_l184_184014


namespace pencils_bought_at_cost_price_l184_184936

variable (C S : ℝ)
variable (n : ℕ)

theorem pencils_bought_at_cost_price (h1 : n * C = 8 * S) (h2 : S = 1.5 * C) : n = 12 := 
by sorry

end pencils_bought_at_cost_price_l184_184936


namespace range_of_m_n_l184_184044

noncomputable def tangent_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 → (m + 1) * x + (n + 1) * y - 2 = 0

theorem range_of_m_n (m n : ℝ) :
  tangent_condition m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end range_of_m_n_l184_184044


namespace range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l184_184317

theorem range_of_x_if_p_and_q_true (a : ℝ) (p q : ℝ → Prop) (h_a : a = 1) (h_p : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (h_q : ∀ x, q x ↔ (x-3)^2 < 1) (h_pq : ∀ x, p x ∧ q x) :
  ∀ x, 2 < x ∧ x < 3 :=
by
  sorry

theorem range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q (p q : ℝ → Prop) (h_neg : ∀ x, ¬p x → ¬q x) : 
  ∀ a : ℝ, a > 0 → (a ≥ 4/3 ∧ a ≤ 2) :=
by
  sorry

end range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l184_184317


namespace find_A_of_trig_max_bsquared_plus_csquared_l184_184342

-- Given the geometric conditions and trigonometric identities.

-- Prove: Given 2a * sin B = b * tan A, we have A = π / 3
theorem find_A_of_trig (a b c A B C : Real) (h1 : 2 * a * Real.sin B = b * Real.tan A) :
  A = Real.pi / 3 := sorry

-- Prove: Given a = 2, the maximum value of b^2 + c^2 is 8
theorem max_bsquared_plus_csquared (a b c A : Real) (hA : A = Real.pi / 3) (ha : a = 2) :
  b^2 + c^2 ≤ 8 :=
by
  have hcos : Real.cos A = 1 / 2 := by sorry
  have h : 4 = b^2 + c^2 - b * c * (1/2) := by sorry
  have hmax : b^2 + c^2 + b * c ≤ 8 := by sorry
  sorry -- Proof steps to reach the final result

end find_A_of_trig_max_bsquared_plus_csquared_l184_184342


namespace volume_invariant_l184_184923

noncomputable def volume_of_common_region (a b c : ℝ) : ℝ := (5/6) * a * b * c

theorem volume_invariant (a b c : ℝ) (P : ℝ × ℝ × ℝ) (hP : ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧ 0 ≤ z ∧ z ≤ c) :
  volume_of_common_region a b c = (5/6) * a * b * c :=
by sorry

end volume_invariant_l184_184923


namespace problem_l184_184428

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem problem (f : ℝ → ℝ) (h : ∀ (x y : ℝ), f (x * y) = f x + f y) : 
  (∀ x : ℝ, f x = log x / log 2) :=
sorry

end problem_l184_184428


namespace solution_l184_184846

def is_prime (n : ℕ) : Prop := ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

noncomputable def find_pairs : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ is_prime (a * b^2 / (a + b)) ∧ ((a = 6 ∧ b = 2) ∨ (a = 2 ∧ b = 6))

theorem solution :
  find_pairs := sorry

end solution_l184_184846


namespace sqrt_sum_abs_eq_l184_184454

theorem sqrt_sum_abs_eq (x : ℝ) :
    (Real.sqrt (x^2 + 6 * x + 9) + Real.sqrt (x^2 - 6 * x + 9)) = (|x - 3| + |x + 3|) := 
by 
  sorry

end sqrt_sum_abs_eq_l184_184454


namespace original_six_digit_number_is_285714_l184_184115

theorem original_six_digit_number_is_285714 
  (N : ℕ) 
  (h1 : ∃ x, N = 200000 + x ∧ 10 * x + 2 = 3 * (200000 + x)) :
  N = 285714 := 
sorry

end original_six_digit_number_is_285714_l184_184115


namespace tan_30_deg_plus_4_sin_30_deg_eq_l184_184679

theorem tan_30_deg_plus_4_sin_30_deg_eq :
  let sin30 := 1 / 2 in
  let cos30 := Real.sqrt 3 / 2 in
  let tan30 := sin30 / cos30 in
  tan30 + 4 * sin30 = (Real.sqrt 3 + 6) / 3 :=
by
  sorry

end tan_30_deg_plus_4_sin_30_deg_eq_l184_184679


namespace cone_height_correct_l184_184370

noncomputable def height_of_cone (R1 R2 R3 base_radius : ℝ) : ℝ :=
  if R1 = 20 ∧ R2 = 40 ∧ R3 = 40 ∧ base_radius = 21 then 28 else 0

theorem cone_height_correct :
  height_of_cone 20 40 40 21 = 28 :=
by sorry

end cone_height_correct_l184_184370


namespace range_of_a_l184_184149

theorem range_of_a
  (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 6)
  (y : ℝ) (hy : 0 < y)
  (h : (y / 4 - 2 * (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) :
  a ≤ 3 :=
sorry

end range_of_a_l184_184149


namespace reporters_not_covering_politics_l184_184653

def total_reporters : ℝ := 8000
def politics_local : ℝ := 0.12 + 0.08 + 0.08 + 0.07 + 0.06 + 0.05 + 0.04 + 0.03 + 0.02 + 0.01
def politics_non_local : ℝ := 0.15
def politics_total : ℝ := politics_local + politics_non_local

theorem reporters_not_covering_politics :
  1 - politics_total = 0.29 :=
by
  -- Required definition and intermediate proof steps.
  sorry

end reporters_not_covering_politics_l184_184653


namespace least_tiles_required_l184_184251

def room_length : ℕ := 7550
def room_breadth : ℕ := 2085
def tile_size : ℕ := 5
def total_area : ℕ := room_length * room_breadth
def tile_area : ℕ := tile_size * tile_size
def number_of_tiles : ℕ := total_area / tile_area

theorem least_tiles_required : number_of_tiles = 630270 := by
  sorry

end least_tiles_required_l184_184251


namespace sample_size_is_10_l184_184851

def product := Type

noncomputable def number_of_products : ℕ := 80
noncomputable def selected_products_for_quality_inspection : ℕ := 10

theorem sample_size_is_10 
  (N : ℕ) (sample_size : ℕ) 
  (hN : N = 80) 
  (h_sample_size : sample_size = 10) : 
  sample_size = 10 :=
by 
  sorry

end sample_size_is_10_l184_184851


namespace fraction_addition_l184_184131

theorem fraction_addition : (3 / 8) + (9 / 12) = 9 / 8 := sorry

end fraction_addition_l184_184131


namespace boots_cost_5_more_than_shoes_l184_184258

variable (S B : ℝ)

-- Conditions based on the problem statement
axiom h1 : 22 * S + 16 * B = 460
axiom h2 : 8 * S + 32 * B = 560

/-- Theorem to prove that the difference in cost between pairs of boots and pairs of shoes is $5 --/
theorem boots_cost_5_more_than_shoes : B - S = 5 :=
by
  sorry

end boots_cost_5_more_than_shoes_l184_184258


namespace complement_of_A_in_U_l184_184021

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set A
def A : Set ℕ := {3, 4, 5}

-- Statement to prove the complement of A with respect to U
theorem complement_of_A_in_U : U \ A = {1, 2, 6} := 
  by sorry

end complement_of_A_in_U_l184_184021


namespace find_ratio_of_three_numbers_l184_184346

noncomputable def ratio_of_three_numbers (A B C : ℝ) : Prop :=
  (A + B + C) / (A + B - C) = 4 / 3 ∧
  (A + B) / (B + C) = 7 / 6

theorem find_ratio_of_three_numbers (A B C : ℝ) (h₁ : ratio_of_three_numbers A B C) :
  A / C = 2 ∧ B / C = 5 :=
by
  sorry

end find_ratio_of_three_numbers_l184_184346


namespace cos_double_angle_l184_184745

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -Real.sqrt 3 / 2) : Real.cos (2 * α) = 1 / 2 :=
by
  sorry

end cos_double_angle_l184_184745


namespace book_read_ratio_l184_184809

variables (B : ℕ) -- Number of books Brad read last month

-- Conditions
def WilliamBooksLastMonth := 6
def BradBooksThisMonth := 8 
def WilliamBooksThisMonth := 2 * BradBooksThisMonth
def WilliamBooksTotal := WilliamBooksLastMonth + WilliamBooksThisMonth
def BradBooksTotal := WilliamBooksTotal - 4
def BradBooksLastMonth := BradBooksTotal - BradBooksThisMonth

-- The Ratio we need to prove
def Ratio := BradBooksLastMonth / WilliamBooksLastMonth

theorem book_read_ratio (h : BradBooksLastMonth = B) (hB : B = 10) :
  Ratio = 5 / 3 :=
by
  rw [←hB] 
  sorry

end book_read_ratio_l184_184809


namespace monotonic_intervals_of_f_g_minus_f_lt_3_l184_184862

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)

noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

-- Theorem 1: Monotonic intervals of f(x)
theorem monotonic_intervals_of_f : 
  (∀ x : ℝ, x ∈ set.Ioo (-∞) (-1 / Real.exp 1) → f' x < 0) ∧ 
  (∀ x : ℝ, x ∈ set.Ioo (-1 / Real.exp 1) 0 → f' x > 0) :=
sorry

-- Theorem 2: Proving g(x) - f(x) < 3
theorem g_minus_f_lt_3 (x : ℝ) : g(x) - f(x) < 3 :=
sorry

end monotonic_intervals_of_f_g_minus_f_lt_3_l184_184862


namespace find_x1_l184_184465

noncomputable def parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem find_x1 
  (a h k m x1 : ℝ)
  (h1 : parabola a h k (-1) = 2)
  (h2 : parabola a h k 1 = -2)
  (h3 : parabola a h k 3 = 2)
  (h4 : parabola a h k (-2) = m)
  (h5 : parabola a h k x1 = m) :
  x1 = 4 := 
sorry

end find_x1_l184_184465


namespace justin_tim_games_count_l184_184556

/-- 
At Barwell Middle School, a larger six-square league has 12 players, 
including Justin and Tim. Daily at recess, the twelve players form two six-square games,
each involving six players in no relevant order. Over a term, each possible 
configuration of six players happens exactly once. How many times did Justin and Tim 
end up in the same six-square game?
--/
theorem justin_tim_games_count :
  let total_players := 12 in
  let game_size := 6 in
  let justin_tim_teams_total := (@Nat.choose 10 4) in
  justin_tim_teams_total = 210 :=
by
  sorry

end justin_tim_games_count_l184_184556


namespace factorize_polynomial_l184_184001

theorem factorize_polynomial (x y : ℝ) : 
  (x^2 - y^2 - 2 * x - 4 * y - 3) = (x + y + 1) * (x - y - 3) :=
  sorry

end factorize_polynomial_l184_184001


namespace factorize_cubed_sub_four_l184_184686

theorem factorize_cubed_sub_four (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
by
  sorry

end factorize_cubed_sub_four_l184_184686


namespace intersection_x_value_l184_184241

/-- Prove that the x-value at the point of intersection of the lines
    y = 5x - 28 and 3x + y = 120 is 18.5 -/
theorem intersection_x_value :
  ∃ x y : ℝ, (y = 5 * x - 28) ∧ (3 * x + y = 120) ∧ (x = 18.5) :=
by
  sorry

end intersection_x_value_l184_184241


namespace largest_lcm_l184_184520

theorem largest_lcm :
  ∀ (a b c d e f : ℕ),
  a = Nat.lcm 18 2 →
  b = Nat.lcm 18 4 →
  c = Nat.lcm 18 6 →
  d = Nat.lcm 18 9 →
  e = Nat.lcm 18 12 →
  f = Nat.lcm 18 16 →
  max (max (max (max (max a b) c) d) e) f = 144 :=
by
  intros a b c d e f ha hb hc hd he hf
  sorry

end largest_lcm_l184_184520


namespace euler_lines_intersect_l184_184916

open EuclideanGeometry

theorem euler_lines_intersect
  {A B C P : Point}
  (h_in_triangle : in_triangle P A B C)
  (h_angle_APB : ∠APB = 120)
  (h_angle_BPC : ∠BPC = 120)
  (h_angle_CPA : ∠CPA = 120)
  (h_tri_angles_lt_120 : ∀ α ∈ {∠BAC, ∠ABC, ∠BCA}, α < 120) :
  let Δ_APB := (A, P, B),
      Δ_BPC := (B, P, C),
      Δ_CPA := (C, P, A),
      Euler_APB := euler_line Δ_APB,
      Euler_BPC := euler_line Δ_BPC,
      Euler_CPA := euler_line Δ_CPA
  in intersects_at_one_point Euler_APB Euler_BPC Euler_CPA :=
  sorry

end euler_lines_intersect_l184_184916


namespace original_rectangle_area_is_56_l184_184671

-- Conditions
def original_rectangle_perimeter := 30 -- cm
def smaller_rectangle_perimeter := 16 -- cm
def side_length_square := (original_rectangle_perimeter - smaller_rectangle_perimeter) / 2 -- Using the reduction logic

-- Computing the length and width of the original rectangle.
def width_original_rectangle := side_length_square
def length_original_rectangle := smaller_rectangle_perimeter / 2

-- The goal is to prove that the area of the original rectangle is 56 cm^2.

theorem original_rectangle_area_is_56:
  (length_original_rectangle - width_original_rectangle + width_original_rectangle) = 8 -- finding the length
  ∧ (length_original_rectangle * width_original_rectangle) = 56 := by
  sorry

end original_rectangle_area_is_56_l184_184671


namespace lambda_range_l184_184713

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem lambda_range (λ : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                   x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
                   (∀ x ≠ 0, f x =/ is real) ∧
                   Real.sqrt (f x) + 2 / Real.sqrt (f x) - λ = 0)
   ↔
   λ > Real.exp 1 + 2 / Real.exp 1 :=
sorry

end lambda_range_l184_184713


namespace linear_function_no_third_quadrant_l184_184598

theorem linear_function_no_third_quadrant :
  ∀ x y : ℝ, (y = -5 * x + 2023) → ¬ (x < 0 ∧ y < 0) := 
by
  intros x y h
  sorry

end linear_function_no_third_quadrant_l184_184598


namespace roots_of_polynomial_l184_184693

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l184_184693


namespace a_2015_eq_neg6_l184_184327

noncomputable def a : ℕ → ℤ
| 0 => 3
| 1 => 6
| (n+2) => a (n+1) - a n

theorem a_2015_eq_neg6 : a 2015 = -6 := 
by 
  sorry

end a_2015_eq_neg6_l184_184327


namespace compare_neg_frac1_l184_184422

theorem compare_neg_frac1 : (-3 / 7 : ℝ) < (-8 / 21 : ℝ) :=
sorry

end compare_neg_frac1_l184_184422


namespace inequality_holds_for_all_x_iff_l184_184728

theorem inequality_holds_for_all_x_iff (m : ℝ) :
  (∀ (x : ℝ), m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ -10 < m ∧ m ≤ 2 :=
by
  sorry

end inequality_holds_for_all_x_iff_l184_184728


namespace vertex_of_parabola_l184_184933

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end vertex_of_parabola_l184_184933


namespace number_of_possible_medians_l184_184354

open Set

variable (S : Finset ℤ)

def seven_elements : Set ℤ := {3, 5, 7, 13, 15, 17, 19}

theorem number_of_possible_medians
  (h1 : S.card = 11)
  (h2 : seven_elements ⊆ S) :
  ∃ medians : Finset ℤ, 
    (medians.card = 8 ∧ 
     ∀ m ∈ medians, 
     ∃ T : List ℤ, 
       T.length = 11 ∧ 
       T.nth 5 = some m ∧ 
       (∀ x ∈ T, x ∈ S)) :=
sorry

end number_of_possible_medians_l184_184354


namespace find_maximum_value_l184_184473

open Real

noncomputable def maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : ℝ :=
  2 + sqrt 5

theorem find_maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) > maximum_value a b c h₁ h₂ h₃ h₄ :=
by
  sorry

end find_maximum_value_l184_184473


namespace num_valid_permutations_l184_184588

theorem num_valid_permutations : 
  let digits := [2, 0, 2, 3]
  let num_2 := 2
  let total_permutations := Nat.factorial 4 / (Nat.factorial num_2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_start_2 := Nat.factorial 3
  let valid_start_3 := Nat.factorial 3 / Nat.factorial 2
  total_permutations = 12 ∧ valid_start_2 = 6 ∧ valid_start_3 = 3 ∧ (valid_start_2 + valid_start_3 = 9) := 
by
  sorry

end num_valid_permutations_l184_184588


namespace relationship_even_increasing_l184_184710

-- Even function definition
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- Monotonically increasing function definition on interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

variable {f : ℝ → ℝ}

-- The proof problem statement
theorem relationship_even_increasing (h_even : even_function f) (h_increasing : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
by
  sorry

end relationship_even_increasing_l184_184710


namespace find_third_number_l184_184385

theorem find_third_number (A B C : ℝ) (h1 : (A + B + C) / 3 = 48) (h2 : (A + B) / 2 = 56) : C = 32 :=
by sorry

end find_third_number_l184_184385


namespace quadratic_roots_opposite_l184_184574

theorem quadratic_roots_opposite (a : ℝ) (h : ∀ x1 x2 : ℝ, 
  (x1 + x2 = 0 ∧ x1 * x2 = a - 1) ∧
  (x1 - (-(x1)) = 0 ∧ x2 - x1 = 0)) :
  a = 0 :=
sorry

end quadratic_roots_opposite_l184_184574


namespace five_letter_words_with_vowels_l184_184887

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l184_184887


namespace PatriciaHighlightFilmTheorem_l184_184616

def PatriciaHighlightFilmProblem : Prop :=
  let point_guard_seconds := 130
  let shooting_guard_seconds := 145
  let small_forward_seconds := 85
  let power_forward_seconds := 60
  let center_seconds := 180
  let total_seconds := point_guard_seconds + shooting_guard_seconds + small_forward_seconds + power_forward_seconds + center_seconds
  let num_players := 5
  let average_seconds := total_seconds / num_players
  let average_minutes := average_seconds / 60
  average_minutes = 2

theorem PatriciaHighlightFilmTheorem : PatriciaHighlightFilmProblem :=
  by
    -- Proof goes here
    sorry

end PatriciaHighlightFilmTheorem_l184_184616


namespace nickels_eq_100_l184_184098

variables (P D N Q H DollarCoins : ℕ)

def conditions :=
  D = P + 10 ∧
  N = 2 * D ∧
  Q = 4 ∧
  P = 10 * Q ∧
  H = Q + 5 ∧
  DollarCoins = 3 * H ∧
  (P + 10 * D + 5 * N + 25 * Q + 50 * H + 100 * DollarCoins = 2000)

theorem nickels_eq_100 (h : conditions P D N Q H DollarCoins) : N = 100 :=
by {
  sorry
}

end nickels_eq_100_l184_184098


namespace range_of_p_l184_184874

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end range_of_p_l184_184874


namespace contrapositive_of_implication_l184_184586

theorem contrapositive_of_implication (p q : Prop) (h : p → q) : ¬q → ¬p :=
by {
  sorry
}

end contrapositive_of_implication_l184_184586


namespace line_of_symmetry_is_x_eq_0_l184_184475

variable (f : ℝ → ℝ)

theorem line_of_symmetry_is_x_eq_0 :
  (∀ y, f (10 + y) = f (10 - y)) → ( ∃ l, l = 0 ∧ ∀ x,  f (10 + l + x) = f (10 + l - x)) := 
by
  sorry

end line_of_symmetry_is_x_eq_0_l184_184475


namespace Tim_sleep_hours_l184_184246

theorem Tim_sleep_hours (x : ℕ) : 
  (x + x + 10 + 10 = 32) → x = 6 :=
by
  intro h
  sorry

end Tim_sleep_hours_l184_184246


namespace richard_older_than_david_l184_184815

theorem richard_older_than_david
  (R D S : ℕ)   -- ages of Richard, David, Scott
  (x : ℕ)       -- the number of years Richard is older than David
  (h1 : R = D + x)
  (h2 : D = S + 8)
  (h3 : R + 8 = 2 * (S + 8))
  (h4 : D = 14) : 
  x = 6 := sorry

end richard_older_than_david_l184_184815


namespace find_angle_A_correct_l184_184906

noncomputable def find_angle_A (BC AB angleC : ℝ) : ℝ :=
if BC = 3 ∧ AB = Real.sqrt 6 ∧ angleC = Real.pi / 4 then
  Real.pi / 3
else
  sorry

theorem find_angle_A_correct : find_angle_A 3 (Real.sqrt 6) (Real.pi / 4) = Real.pi / 3 :=
by
  -- proof goes here
  sorry

end find_angle_A_correct_l184_184906


namespace marble_theorem_l184_184158

noncomputable def marble_problem (M : ℝ) : Prop :=
  let M_Pedro : ℝ := 0.7 * M
  let M_Ebony : ℝ := 0.85 * M_Pedro
  let M_Jimmy : ℝ := 0.7 * M_Ebony
  (M_Jimmy / M) * 100 = 41.65

theorem marble_theorem (M : ℝ) : marble_problem M := 
by
  sorry

end marble_theorem_l184_184158


namespace that_three_digit_multiples_of_5_and_7_l184_184892

/-- 
Define the count_three_digit_multiples function, 
which counts the number of three-digit integers that are multiples of both 5 and 7.
-/
def count_three_digit_multiples : ℕ :=
  let lcm := Nat.lcm 5 7
  let first := (100 + lcm - 1) / lcm * lcm
  let last := 999 / lcm * lcm
  (last - first) / lcm + 1

/-- 
Theorem that states the number of positive three-digit integers that are multiples of both 5 and 7 is 26. 
-/
theorem three_digit_multiples_of_5_and_7 : count_three_digit_multiples = 26 := by
  sorry

end that_three_digit_multiples_of_5_and_7_l184_184892


namespace volume_of_rotated_segment_l184_184769

theorem volume_of_rotated_segment (a R : ℝ) (h : R > a / 2) :
  let V := 2 * π * ∫ x in -a/2..a/2, (R^2 - x^2)
  V = (π * a^3) / 6 :=
by
  -- Insert proof here
  sorry

end volume_of_rotated_segment_l184_184769


namespace avg_height_of_remaining_students_l184_184627

-- Define the given conditions
def avg_height_11_members : ℝ := 145.7
def number_of_members : ℝ := 11
def height_of_two_students : ℝ := 142.1

-- Define what we need to prove
theorem avg_height_of_remaining_students :
  (avg_height_11_members * number_of_members - 2 * height_of_two_students) / (number_of_members - 2) = 146.5 :=
by
  sorry

end avg_height_of_remaining_students_l184_184627


namespace remainder_problem_l184_184285

theorem remainder_problem :
  (1234567 % 135 = 92) ∧ ((92 * 5) % 27 = 1) := by
  sorry

end remainder_problem_l184_184285


namespace tod_driving_time_l184_184953
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end tod_driving_time_l184_184953


namespace shaded_trapezium_area_l184_184387

theorem shaded_trapezium_area :
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  area = 55 / 4 :=
by
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  show area = 55 / 4
  sorry

end shaded_trapezium_area_l184_184387


namespace subsetneq_M_N_l184_184049

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (x < 0) ∨ (x > 1 / 2)}

theorem subsetneq_M_N : M ⊂ N :=
by
  sorry

end subsetneq_M_N_l184_184049


namespace probability_at_least_three_red_l184_184272

def total_jellybeans := 15
def red_jellybeans := 6
def blue_jellybeans := 3
def white_jellybeans := 6
def picked_jellybeans := 4

-- Number of ways to pick k items from n items
noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways to pick 4 jellybeans from 15
noncomputable def total_ways := choose total_jellybeans picked_jellybeans

-- Number of ways to pick exactly 3 red jellybeans and 1 non-red jellybean
noncomputable def ways_three_red_one_nonred := choose red_jellybeans 3 * choose (total_jellybeans - red_jellybeans) 1

-- Number of ways to pick exactly 4 red jellybeans
noncomputable def ways_four_red := choose red_jellybeans 4

-- Total number of favorable outcomes for at least 3 red jellybeans
noncomputable def successful_outcomes := ways_three_red_one_nonred + ways_four_red

-- Probability of at least 3 red jellybeans out of 4 picked
noncomputable def probability := (successful_outcomes : ℚ) / total_ways

theorem probability_at_least_three_red :
  probability = (13 : ℚ) / 91 :=
begin
  sorry
end

end probability_at_least_three_red_l184_184272


namespace verify_formula_n1_l184_184814

theorem verify_formula_n1 (a : ℝ) (ha : a ≠ 1) : 1 + a = (a^3 - 1) / (a - 1) :=
by 
  sorry

end verify_formula_n1_l184_184814


namespace trigonometric_inequality_l184_184218

-- Define the necessary mathematical objects and structures:
noncomputable def sin (x : ℝ) : ℝ := sorry -- Assume sine function as given

-- The theorem statement
theorem trigonometric_inequality {x y z A B C : ℝ} 
  (hA : A + B + C = π) -- A, B, C are angles of a triangle
  :
  ((x + y + z) / 2) ^ 2 ≥ x * y * (sin A) ^ 2 + y * z * (sin B) ^ 2 + z * x * (sin C) ^ 2 :=
sorry

end trigonometric_inequality_l184_184218


namespace sum_of_possible_values_N_l184_184788

variable (a b c N : ℕ)

theorem sum_of_possible_values_N :
  (N = a * b * c) ∧ (N = 8 * (a + b + c)) ∧ (c = 2 * a + b) → N = 136 := 
by
  sorry

end sum_of_possible_values_N_l184_184788


namespace percentage_problem_l184_184727

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 400) : 1.20 * x = 2400 :=
by
  sorry

end percentage_problem_l184_184727


namespace find_number_l184_184304

-- Defining the constants provided and the related condition
def eight_percent_of (x: ℝ) : ℝ := 0.08 * x
def ten_percent_of_40 : ℝ := 0.10 * 40
def is_solution (x: ℝ) : Prop := (eight_percent_of x) + ten_percent_of_40 = 5.92

-- Theorem statement
theorem find_number : ∃ x : ℝ, is_solution x ∧ x = 24 :=
by sorry

end find_number_l184_184304


namespace gcd_4830_3289_l184_184503

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 :=
by sorry

end gcd_4830_3289_l184_184503


namespace fraction_simplification_l184_184221

theorem fraction_simplification (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  (x^2 + x) / (x^2 - 1) = x / (x - 1) :=
by
  -- Hint of expected development environment setting
  sorry

end fraction_simplification_l184_184221


namespace sphere_radius_in_cube_l184_184500

theorem sphere_radius_in_cube (r : ℝ) (n : ℕ) (side_length : ℝ) 
  (h1 : side_length = 2) 
  (h2 : n = 16)
  (h3 : ∀ (i : ℕ), i < n → (center_distance : ℝ) = 2 * r)
  (h4: ∀ (i : ℕ), i < n → (face_distance : ℝ) = r) : 
  r = 1 :=
by
  sorry

end sphere_radius_in_cube_l184_184500


namespace no_primes_of_form_2pow5m_plus_2powm_plus_1_l184_184760

theorem no_primes_of_form_2pow5m_plus_2powm_plus_1 {m : ℕ} (hm : m > 0) : ¬ (Prime (2^(5*m) + 2^m + 1)) :=
by
  sorry

end no_primes_of_form_2pow5m_plus_2powm_plus_1_l184_184760


namespace education_expenses_l184_184827

theorem education_expenses (rent milk groceries petrol miscellaneous savings total_salary education : ℝ) 
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_petrol : petrol = 2000)
  (h_miscellaneous : miscellaneous = 6100)
  (h_savings : savings = 2400)
  (h_saving_percentage : savings = 0.10 * total_salary)
  (h_total_salary : total_salary = savings / 0.10)
  (h_total_expenses : total_salary - savings = rent + milk + groceries + petrol + miscellaneous + education) :
  education = 2500 :=
by
  sorry

end education_expenses_l184_184827


namespace dara_employment_wait_time_l184_184231

theorem dara_employment_wait_time :
  ∀ (min_age current_jane_age years_later half_age_factor : ℕ), 
  min_age = 25 → 
  current_jane_age = 28 → 
  years_later = 6 → 
  half_age_factor = 2 →
  (min_age - (current_jane_age + years_later) / half_age_factor - years_later) = 14 :=
by
  intros min_age current_jane_age years_later half_age_factor 
  intros h_min_age h_current_jane_age h_years_later h_half_age_factor
  sorry

end dara_employment_wait_time_l184_184231


namespace diameter_of_circular_ground_l184_184267

noncomputable def radius_of_garden_condition (area_garden : ℝ) (broad_garden : ℝ) : ℝ :=
  let pi_val := Real.pi
  (area_garden / pi_val - broad_garden * broad_garden) / (2 * broad_garden)

-- Given conditions
variable (area_garden : ℝ := 226.19467105846502)
variable (broad_garden : ℝ := 2)

-- Goal to prove: diameter of the circular ground is 34 metres
theorem diameter_of_circular_ground : 2 * radius_of_garden_condition area_garden broad_garden = 34 :=
  sorry

end diameter_of_circular_ground_l184_184267


namespace solve_equation_l184_184068

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (3 - x^2) / (x + 2) + (2 * x^2 - 8) / (x^2 - 4) = 3 ↔ 
  x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
by
  sorry

end solve_equation_l184_184068


namespace female_students_selected_l184_184344

theorem female_students_selected (males females : ℕ) (p : ℚ) (h_males : males = 28)
  (h_females : females = 21) (h_p : p = 1 / 7) : females * p = 3 := by 
  sorry

end female_students_selected_l184_184344


namespace albert_age_l184_184667

theorem albert_age
  (A : ℕ)
  (dad_age : ℕ)
  (h1 : dad_age = 48)
  (h2 : dad_age - 4 = 4 * (A - 4)) :
  A = 15 :=
by
  sorry

end albert_age_l184_184667


namespace rhombus_area_correct_l184_184227

def rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_correct
  (d1 d2 : ℕ)
  (h1 : d1 = 70)
  (h2 : d2 = 160) :
  rhombus_area d1 d2 = 5600 := 
by
  sorry

end rhombus_area_correct_l184_184227


namespace prove_system_of_inequalities_l184_184070

theorem prove_system_of_inequalities : 
  { x : ℝ | x / (x - 2) ≥ 0 ∧ 2 * x + 1 ≥ 0 } = Set.Icc (-(1:ℝ)/2) 0 ∪ Set.Ioi 2 := 
by
  sorry

end prove_system_of_inequalities_l184_184070


namespace a_4_eq_15_l184_184587

noncomputable def a : ℕ → ℕ
| 0 => 1
| (n + 1) => 2 * a n + 1

theorem a_4_eq_15 : a 3 = 15 :=
by
  sorry

end a_4_eq_15_l184_184587


namespace last_digit_B_l184_184145

theorem last_digit_B 
  (B : ℕ) 
  (h : ∀ n : ℕ, n % 10 = (B - 287)^2 % 10 → n % 10 = 4) :
  (B = 5 ∨ B = 9) :=
sorry

end last_digit_B_l184_184145


namespace first_divisor_l184_184796

-- Definitions
def is_divisible_by (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

-- Theorem to prove
theorem first_divisor (x : ℕ) (h₁ : ∃ l, l = Nat.lcm x 35 ∧ is_divisible_by 1400 l ∧ 1400 / l = 8) : 
  x = 25 := 
sorry

end first_divisor_l184_184796


namespace find_b_l184_184654

theorem find_b (b c : ℝ) : 
  (-11 : ℝ) = (-1)^2 + (-1) * b + c ∧ 
  17 = 3^2 + 3 * b + c ∧ 
  6 = 2^2 + 2 * b + c → 
  b = 14 / 3 :=
by
  sorry

end find_b_l184_184654


namespace minimal_erasure_l184_184535

noncomputable def min_factors_to_erase : ℕ :=
  2016

theorem minimal_erasure:
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = g x) → 
    (∃ f' g' : ℝ → ℝ, (∀ x, f x ≠ g x) ∧ 
      ((∃ s : Finset ℕ, s.card = min_factors_to_erase ∧ (∀ i ∈ s, f' x = (x - i) * f x)) ∧ 
      (∃ t : Finset ℕ, t.card = min_factors_to_erase ∧ (∀ i ∈ t, g' x = (x - i) * g x)))) :=
by
  sorry

end minimal_erasure_l184_184535


namespace quadrilateral_diagonal_length_l184_184450

theorem quadrilateral_diagonal_length (D A₁ A₂ : ℝ) (hA₁ : A₁ = 9) (hA₂ : A₂ = 6) (Area : ℝ) (hArea : Area = 165) :
  (1/2) * D * (A₁ + A₂) = Area → D = 22 :=
by
  -- Use the given conditions and solve to obtain D = 22
  intros
  sorry

end quadrilateral_diagonal_length_l184_184450


namespace sean_less_points_than_combined_l184_184735

def tobee_points : ℕ := 4
def jay_points : ℕ := tobee_points + 6
def combined_points_tobee_jay : ℕ := tobee_points + jay_points
def total_team_points : ℕ := 26
def sean_points : ℕ := total_team_points - combined_points_tobee_jay

theorem sean_less_points_than_combined : (combined_points_tobee_jay - sean_points) = 2 := by
  sorry

end sean_less_points_than_combined_l184_184735


namespace winning_votes_cast_l184_184259

variable (V : ℝ) -- Total number of votes (real number)
variable (winner_votes_ratio : ℝ) -- Ratio for winner's votes
variable (votes_difference : ℝ) -- Vote difference due to winning

-- Conditions given
def election_conditions (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) : Prop :=
  winner_votes_ratio = 0.54 ∧
  votes_difference = 288

-- Proof problem: Proving the number of votes cast to the winning candidate is 1944
theorem winning_votes_cast (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) 
  (h : election_conditions V winner_votes_ratio votes_difference) :
  winner_votes_ratio * V = 1944 :=
by
  sorry

end winning_votes_cast_l184_184259


namespace alcohol_added_l184_184650

theorem alcohol_added (x : ℝ) :
  let initial_solution_volume := 40
  let initial_alcohol_percentage := 0.05
  let initial_alcohol_volume := initial_solution_volume * initial_alcohol_percentage
  let additional_water := 6.5
  let final_solution_volume := initial_solution_volume + x + additional_water
  let final_alcohol_percentage := 0.11
  let final_alcohol_volume := final_solution_volume * final_alcohol_percentage
  initial_alcohol_volume + x = final_alcohol_volume → x = 3.5 :=
by
  intros
  sorry

end alcohol_added_l184_184650


namespace hyperbola_eccentricity_l184_184913

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l184_184913


namespace range_of_x_l184_184894

-- Problem Statement
theorem range_of_x (x : ℝ) (h : 0 ≤ x - 8) : 8 ≤ x :=
by {
  sorry
}

end range_of_x_l184_184894


namespace maximum_f_value_l184_184441

noncomputable def otimes (a b : ℝ) : ℝ :=
if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
otimes (3 * x^2 + 6) (23 - x^2)

theorem maximum_f_value : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 4 :=
sorry

end maximum_f_value_l184_184441


namespace rainfall_on_Monday_l184_184908

theorem rainfall_on_Monday (rain_on_Tuesday : ℝ) (difference : ℝ) (rain_on_Tuesday_eq : rain_on_Tuesday = 0.2) (difference_eq : difference = 0.7) :
  ∃ rain_on_Monday : ℝ, rain_on_Monday = rain_on_Tuesday + difference := 
sorry

end rainfall_on_Monday_l184_184908


namespace solve_equation_one_solve_equation_two_l184_184926

theorem solve_equation_one (x : ℝ) : 3 * x + 7 = 32 - 2 * x → x = 5 :=
by
  intro h
  sorry

theorem solve_equation_two (x : ℝ) : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 → x = -1 :=
by
  intro h
  sorry

end solve_equation_one_solve_equation_two_l184_184926


namespace tim_total_trip_time_l184_184094

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end tim_total_trip_time_l184_184094


namespace tan_alpha_eq_one_third_l184_184023

variable (α : ℝ)

theorem tan_alpha_eq_one_third (h : Real.tan (α + Real.pi / 4) = 2) : Real.tan α = 1 / 3 :=
sorry

end tan_alpha_eq_one_third_l184_184023


namespace smallest_m_last_four_digits_l184_184358

theorem smallest_m_last_four_digits :
  ∃ m : ℕ, 
    (∃ k : ℕ, m = 4 * k) ∧ -- m is divisible by 4
    (∃ k : ℕ, m = 9 * k) ∧ -- m is divisible by 9
    (∀ d : ℕ, d ∣ m → d = 4 ∨ d = 9 ∨ d = 1) ∧ -- m's digits are only 4 and 9
    (∃ n₄ n₉ : ℕ, n₄ ≥ 2 ∧ n₉ ≥ 2 ∧ m.digits 10 = (list.replicate n₄ 4 ++ list.replicate n₉ 9).reverse) ∧ -- at least two 4's and two 9's
    (m % 10000 = 9494) -- last four digits are 9494
    :=
  sorry

end smallest_m_last_four_digits_l184_184358


namespace general_formula_a_sum_sn_l184_184316

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 * n

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ :=
  a n + 2 ^ (a n)

-- Define the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem general_formula_a :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_sn :
  ∀ n, S n = n * (n + 1) + (4^(n + 1) - 4) / 3 :=
sorry

end general_formula_a_sum_sn_l184_184316


namespace locus_of_intersections_l184_184373

-- Conditions definitions
variable {A B C : Point}
variable (O₁ O₂ : Circle)

-- Orthogonal circles
def orthogonal (O₁ O₂ : Circle) : Prop :=
  -- Definition of orthogonality of circles in terms of their intersections (details omitted)
  sorry

def passes_through (O : Circle) (P Q : Point) : Prop :=
  -- Definition of circle passing through points P and Q (details omitted)
  sorry

-- Main theorem
theorem locus_of_intersections (h₁ : orthogonal O₁ O₂)
                              (h₂ : passes_through O₁ A C)
                              (h₃ : passes_through O₂ B C) :
  (exists (L : locus_of_intersections_with_orthogonal_circles_through [A, C] [B, C]),
    is_circle L ∨ is_line L) :=
by
  sorry

end locus_of_intersections_l184_184373


namespace third_neigh_uses_100_more_l184_184826

def total_water : Nat := 1200
def first_neigh_usage : Nat := 150
def second_neigh_usage : Nat := 2 * first_neigh_usage
def fourth_neigh_remaining : Nat := 350

def third_neigh_usage := total_water - (first_neigh_usage + second_neigh_usage + fourth_neigh_remaining)
def diff_third_second := third_neigh_usage - second_neigh_usage

theorem third_neigh_uses_100_more :
  diff_third_second = 100 := by
  sorry

end third_neigh_uses_100_more_l184_184826


namespace range_of_ab_l184_184848

noncomputable def circle_equation (x y : ℝ) : Prop := (x^2 + y^2 + 2*x - 4*y + 1 = 0)

noncomputable def line_equation (a b x y : ℝ) : Prop := (2*a*x - b*y - 2 = 0)

def symmetric_with_respect_to (center_x center_y a b : ℝ) : Prop :=
  line_equation a b center_x center_y  -- check if the line passes through the center

theorem range_of_ab (a b : ℝ) (h_symm : symmetric_with_respect_to (-1) 2 a b) : 
  ∃ ab_max : ℝ, ab_max = 1/4 ∧ ∀ ab : ℝ, ab = (a * b) → ab ≤ ab_max :=
sorry

end range_of_ab_l184_184848


namespace percentage_invalid_votes_l184_184347

theorem percentage_invalid_votes
  (total_votes : ℕ)
  (votes_for_A : ℕ)
  (candidate_A_percentage : ℝ)
  (total_votes_count : total_votes = 560000)
  (votes_for_A_count : votes_for_A = 404600)
  (candidate_A_percentage_count : candidate_A_percentage = 0.85) :
  ∃ (x : ℝ), (x / 100) * total_votes = total_votes - votes_for_A / candidate_A_percentage ∧ x = 15 :=
by
  sorry

end percentage_invalid_votes_l184_184347


namespace possible_values_of_K_l184_184594

theorem possible_values_of_K (K N : ℕ) (h1 : K * (K + 1) = 2 * N^2) (h2 : N < 100) :
  K = 1 ∨ K = 8 ∨ K = 49 :=
sorry

end possible_values_of_K_l184_184594


namespace symmetric_scanning_codes_count_l184_184659

structure Grid (n : ℕ) :=
  (cells : Fin n × Fin n → Bool)

def is_symmetric_90 (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - j, i)

def is_symmetric_reflection_mid_side (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - i, j) ∧ g.cells (i, j) = g.cells (i, 7 - j)

def is_symmetric_reflection_diagonal (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (j, i)

def has_at_least_one_black_and_one_white (g : Grid 8) : Prop :=
  ∃ i j, g.cells (i, j) ∧ ∃ i j, ¬g.cells (i, j)

noncomputable def count_symmetric_scanning_codes : ℕ :=
  (sorry : ℕ)

theorem symmetric_scanning_codes_count : count_symmetric_scanning_codes = 62 :=
  sorry

end symmetric_scanning_codes_count_l184_184659


namespace arithmetic_sequence_sufficient_but_not_necessary_condition_l184_184717

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def a_1_a_3_equals_2a_2 (a : ℕ → ℤ) :=
  a 1 + a 3 = 2 * a 2

-- Statement of the mathematical problem
theorem arithmetic_sequence_sufficient_but_not_necessary_condition (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a_1_a_3_equals_2a_2 a ∧ (a_1_a_3_equals_2a_2 a → ¬ is_arithmetic_sequence a) :=
by
  sorry

end arithmetic_sequence_sufficient_but_not_necessary_condition_l184_184717


namespace distance_from_y_axis_l184_184629

theorem distance_from_y_axis (P : ℝ × ℝ) (x : ℝ) (hx : P = (x, -9)) 
  (h : (abs (P.2) = 1/2 * abs (P.1))) :
  abs x = 18 :=
by
  sorry

end distance_from_y_axis_l184_184629


namespace raft_minimum_capacity_l184_184985

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l184_184985


namespace remainder_of_6x_mod_9_l184_184420

theorem remainder_of_6x_mod_9 (x : ℕ) (h : x % 9 = 5) : (6 * x) % 9 = 3 :=
by
  sorry

end remainder_of_6x_mod_9_l184_184420


namespace lucky_license_plates_count_l184_184965

open Finset

def num_lucky_license_plates : ℕ :=
  let letters := {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'U', 'X'}
  let consonants := {'B', 'K', 'M', 'H', 'P', 'C', 'T', 'X'}
  let odd_digits := {1, 3, 5, 7, 9}
  let even_digits := {0, 2, 4, 6, 8}
  let num_letters := 12
  let num_consonants := 8
  let num_odd_digits := 5
  let num_even_digits := 5
  let num_digits := 10
  num_letters * num_odd_digits * num_digits * num_even_digits * num_consonants * num_letters

theorem lucky_license_plates_count :
  num_lucky_license_plates = 288000 := by
  sorry

end lucky_license_plates_count_l184_184965


namespace pages_revised_once_l184_184232

-- Definitions
def total_pages : ℕ := 200
def pages_revised_twice : ℕ := 20
def total_cost : ℕ := 1360
def cost_first_time : ℕ := 5
def cost_revision : ℕ := 3

theorem pages_revised_once (x : ℕ) (h1 : total_cost = 1000 + 3 * x + 120) : x = 80 := by
  sorry

end pages_revised_once_l184_184232


namespace water_percentage_in_dried_grapes_l184_184576

noncomputable def fresh_grape_weight : ℝ := 40  -- weight of fresh grapes in kg
noncomputable def dried_grape_weight : ℝ := 5  -- weight of dried grapes in kg
noncomputable def water_percentage_fresh : ℝ := 0.90  -- percentage of water in fresh grapes

noncomputable def water_weight_fresh : ℝ := fresh_grape_weight * water_percentage_fresh
noncomputable def solid_weight_fresh : ℝ := fresh_grape_weight * (1 - water_percentage_fresh)
noncomputable def water_weight_dried : ℝ := dried_grape_weight - solid_weight_fresh
noncomputable def water_percentage_dried : ℝ := (water_weight_dried / dried_grape_weight) * 100

theorem water_percentage_in_dried_grapes : water_percentage_dried = 20 := by
  sorry

end water_percentage_in_dried_grapes_l184_184576


namespace mean_rest_scores_l184_184975

theorem mean_rest_scores (n : ℕ) (h : 15 < n) 
  (overall_mean : ℝ := 10)
  (mean_of_fifteen : ℝ := 12)
  (total_score : ℝ := n * overall_mean): 
  (180 + p * (n - 15) = total_score) →
  p = (10 * n - 180) / (n - 15) :=
sorry

end mean_rest_scores_l184_184975


namespace factorial_multiple_square_l184_184002

theorem factorial_multiple_square (n : ℕ) (h : n > 0) : (Nat.factorial n) % (n^2) = 0 ↔ 
  n = 1 ∨ (n ≥ 6 ∧ ∃ k m : ℕ, (k > 1) ∧ (k < n) ∧ (m > 1) ∧ (m < n) ∧ k * m = n) :=
by 
  sorry

end factorial_multiple_square_l184_184002


namespace PlatformC_location_l184_184744

noncomputable def PlatformA : ℝ := 9
noncomputable def PlatformB : ℝ := 1 / 3
noncomputable def PlatformC : ℝ := 7
noncomputable def AB := PlatformA - PlatformB
noncomputable def AC := PlatformA - PlatformC

theorem PlatformC_location :
  AB = (13 / 3) * AC → PlatformC = 7 :=
by
  intro h
  simp [AB, AC, PlatformA, PlatformB, PlatformC] at h
  sorry

end PlatformC_location_l184_184744


namespace pollution_index_minimum_l184_184666

noncomputable def pollution_index (k a b : ℝ) (x : ℝ) : ℝ :=
  k * (a / (x ^ 2) + b / ((18 - x) ^ 2))

theorem pollution_index_minimum (k : ℝ) (h₀ : 0 < k) (h₁ : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 18) :
  ∀ a b x : ℝ, a = 1 → x = 6 → pollution_index k a b x = pollution_index k 1 8 6 :=
by
  intros a b x ha hx
  rw [ha, hx, pollution_index]
  sorry

end pollution_index_minimum_l184_184666


namespace total_practice_hours_l184_184622

def weekly_practice_hours : ℕ := 4
def weeks_in_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours : (weekly_practice_hours * weeks_in_month) * months = 80 := by
  -- Calculation for weekly practice in hours
  let monthly_hours := weekly_practice_hours * weeks_in_month
  -- Calculation for total practice in hours
  have total_hours : ℕ := monthly_hours * months
  have calculation : total_hours = 80 := 
    by simp [weekly_practice_hours, weeks_in_month, months, monthly_hours, total_hours]
  exact calculation

end total_practice_hours_l184_184622


namespace reese_practice_hours_l184_184618

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l184_184618


namespace number_divisibility_l184_184065

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end number_divisibility_l184_184065


namespace fraction_habitable_surface_l184_184338

noncomputable def fraction_land_not_covered_by_water : ℚ := 1 / 3
noncomputable def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_land_not_covered_by_water * fraction_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_habitable_surface_l184_184338


namespace time_for_second_train_to_cross_l184_184530

def length_first_train : ℕ := 100
def speed_first_train : ℕ := 10
def length_second_train : ℕ := 150
def speed_second_train : ℕ := 15
def distance_between_trains : ℕ := 50

def total_distance : ℕ := length_first_train + length_second_train + distance_between_trains
def relative_speed : ℕ := speed_second_train - speed_first_train

theorem time_for_second_train_to_cross :
  total_distance / relative_speed = 60 :=
by
  -- Definitions and intermediate steps would be handled in the proof here
  sorry

end time_for_second_train_to_cross_l184_184530


namespace michael_water_left_l184_184366

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end michael_water_left_l184_184366


namespace combined_stickers_l184_184088

theorem combined_stickers (k j a : ℕ) (h : 7 * j + 5 * a = 54) (hk : k = 42) (hk_ratio : k = 7 * 6) :
  j + a = 54 :=
by
  sorry

end combined_stickers_l184_184088


namespace range_of_k_l184_184901

theorem range_of_k (k : ℝ) (hₖ : 0 < k) :
  (∃ x : ℝ, 1 = x^2 + (k^2 / x^2)) → 0 < k ∧ k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l184_184901


namespace volume_intersection_pyramids_l184_184353

noncomputable section

-- Definitions of the pyramids P and Q
def pyramid_P_base := [(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0)]
def apex_P := (1, 1, 3)
def apex_Q := (2, 2, 3)

-- Theorem stating the volume of the intersection of P and Q is 27/4
theorem volume_intersection_pyramids :
  let P := mk_pyramid pyramid_P_base apex_P in
  let Q := mk_pyramid pyramid_P_base apex_Q in
  volume (intersection P.interior Q.interior) = 27 / 4 :=
sorry

end volume_intersection_pyramids_l184_184353


namespace general_form_of_equation_l184_184683

theorem general_form_of_equation : 
  ∀ x : ℝ, (x - 1) * (x - 2) = 4 → x^2 - 3 * x - 2 = 0 := by
  sorry

end general_form_of_equation_l184_184683


namespace ram_salary_percentage_more_l184_184374

theorem ram_salary_percentage_more (R r : ℝ) (h : r = 0.8 * R) :
  ((R - r) / r) * 100 = 25 := 
sorry

end ram_salary_percentage_more_l184_184374


namespace mimi_spent_on_clothes_l184_184208

noncomputable def total_cost : ℤ := 8000
noncomputable def cost_adidas : ℤ := 600
noncomputable def cost_nike : ℤ := 3 * cost_adidas
noncomputable def cost_skechers : ℤ := 5 * cost_adidas
noncomputable def cost_clothes : ℤ := total_cost - (cost_adidas + cost_nike + cost_skechers)

theorem mimi_spent_on_clothes :
  cost_clothes = 2600 :=
by
  sorry

end mimi_spent_on_clothes_l184_184208


namespace five_letter_words_with_at_least_one_vowel_l184_184890

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l184_184890


namespace circle_symmetric_equation_l184_184165

noncomputable def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

noncomputable def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

noncomputable def symmetric_condition (x y : ℝ) (L : ℝ × ℝ → Prop) : Prop := 
  L (y + 1, x - 1)

theorem circle_symmetric_equation :
  ∀ (x y : ℝ),
  circle1 (y + 1) (x - 1) →
  (x-2)^2 + (y+2)^2 = 1 :=
by
  intros x y h
  sorry

end circle_symmetric_equation_l184_184165


namespace cost_of_each_shirt_is_8_l184_184289

-- Define the conditions
variables (S : ℝ)
def shirts_cost := 4 * S
def pants_cost := 2 * 18
def jackets_cost := 2 * 60
def total_cost := shirts_cost S + pants_cost + jackets_cost
def carrie_pays := 94

-- The goal is to prove that S equals 8 given the conditions above
theorem cost_of_each_shirt_is_8
  (h1 : carrie_pays = total_cost S / 2) : S = 8 :=
sorry

end cost_of_each_shirt_is_8_l184_184289


namespace magic_square_sum_l184_184194

variable {a b c d e : ℕ}

-- Given conditions:
-- It's a magic square and the sums of the numbers in each row, column, and diagonal are equal.
-- Positions and known values specified:
theorem magic_square_sum (h : 15 + 24 = 18 + c ∧ 18 + c = 27 + a ∧ c = 21 ∧ a = 12 ∧ e = 17 ∧ d = 30 ∧ b = 25)
: d + e = 47 :=
by
  -- Sorry used to skip the proof
  sorry

end magic_square_sum_l184_184194


namespace sum_of_squares_not_7_mod_8_l184_184922

theorem sum_of_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 :=
sorry

end sum_of_squares_not_7_mod_8_l184_184922


namespace sum_of_two_pos_implies_one_pos_l184_184731

theorem sum_of_two_pos_implies_one_pos (x y : ℝ) (h : x + y > 0) : x > 0 ∨ y > 0 :=
  sorry

end sum_of_two_pos_implies_one_pos_l184_184731


namespace number_of_5_letter_words_with_at_least_one_vowel_l184_184882

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l184_184882


namespace percentage_of_profit_if_no_discount_l184_184824

-- Conditions
def discount : ℝ := 0.05
def profit_w_discount : ℝ := 0.216
def cost_price : ℝ := 100
def expected_profit : ℝ := 28

-- Proof statement
theorem percentage_of_profit_if_no_discount :
  ∃ (marked_price selling_price_no_discount : ℝ),
    selling_price_no_discount = marked_price ∧
    (marked_price - cost_price) / cost_price * 100 = expected_profit :=
by
  -- Definitions and logic will go here
  sorry

end percentage_of_profit_if_no_discount_l184_184824


namespace parabola_intercept_sum_l184_184937

theorem parabola_intercept_sum :
  let a := 6
  let b := 1
  let c := 2
  a + b + c = 9 :=
by
  sorry

end parabola_intercept_sum_l184_184937


namespace local_tax_deduction_in_cents_l184_184121

def aliciaHourlyWageInDollars : ℝ := 25
def taxDeductionRate : ℝ := 0.02
def aliciaHourlyWageInCents := aliciaHourlyWageInDollars * 100

theorem local_tax_deduction_in_cents :
  taxDeductionRate * aliciaHourlyWageInCents = 50 :=
by 
  -- Proof goes here
  sorry

end local_tax_deduction_in_cents_l184_184121


namespace calories_per_slice_l184_184140

theorem calories_per_slice
  (total_calories : ℕ)
  (portion_eaten : ℕ)
  (percentage_eaten : ℝ)
  (slices_in_cheesecake : ℕ)
  (calories_in_slice : ℕ) :
  total_calories = 2800 →
  percentage_eaten = 0.25 →
  portion_eaten = 2 →
  portion_eaten = percentage_eaten * slices_in_cheesecake →
  calories_in_slice = total_calories / slices_in_cheesecake →
  calories_in_slice = 350 :=
by
  intros
  sorry

end calories_per_slice_l184_184140


namespace rollo_guinea_pigs_food_l184_184773

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end rollo_guinea_pigs_food_l184_184773


namespace exists_prime_not_dividing_difference_l184_184840

theorem exists_prime_not_dividing_difference {m : ℕ} (hm : m ≠ 1) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬ p ∣ (n^n - m) := 
sorry

end exists_prime_not_dividing_difference_l184_184840


namespace LCM_GCD_product_l184_184275

theorem LCM_GCD_product (a b : ℕ) (h_a : a = 24) (h_b : b = 36) :
  Nat.lcm a b * Nat.gcd a b = 864 := by
  rw [h_a, h_b]
  have lcm_ab := Nat.lcm_eq (by repeat {exact dec_trivial})
  have gcd_ab := Nat.gcd_eq (by repeat {exact dec_trivial})
  exact calc
    Nat.lcm 24 36 * Nat.gcd 24 36 = 72 * 12 : by congr; apply lcm_ab; apply gcd_ab
    ... = 864 : by norm_num

end LCM_GCD_product_l184_184275


namespace sigma_eq_iff_exists_borel_functions_l184_184606

variables {Ω : Type*} [measurable_space Ω]
variables {α : Type*} [measurable_space α] {β : Type*} [measurable_space β]
variables (ξ : Ω → α) (ζ : Ω → β)

/-- σ(ξ) = σ(ζ) if and only if there exist Borel functions φ and ψ such that ξ = φ(ζ) and ζ = ψ(ξ) -/
theorem sigma_eq_iff_exists_borel_functions :
  measurable_space.generate_from (ξ '' set.univ) = measurable_space.generate_from (ζ '' set.univ) ↔
  (∃ (φ : β → α) (ψ : α → β), measurable φ ∧ measurable ψ ∧ ∀ ω, ξ ω = φ (ζ ω) ∧ ζ ω = ψ (ξ ω)) :=
sorry

end sigma_eq_iff_exists_borel_functions_l184_184606


namespace boat_speed_in_still_water_l184_184646

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by sorry

end boat_speed_in_still_water_l184_184646


namespace sum_ak_div_k2_ge_sum_inv_k_l184_184605

open BigOperators

theorem sum_ak_div_k2_ge_sum_inv_k
  (n : ℕ)
  (a : Fin n → ℕ)
  (hpos : ∀ k, 0 < a k)
  (hdist : Function.Injective a) :
  ∑ k : Fin n, (a k : ℝ) / (k + 1 : ℝ)^2 ≥ ∑ k : Fin n, 1 / (k + 1 : ℝ) := sorry

end sum_ak_div_k2_ge_sum_inv_k_l184_184605


namespace smallestThreeDigitNumberWithPerfectSquare_l184_184453

def isThreeDigitNumber (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

def formsPerfectSquare (a : ℕ) : Prop := ∃ n : ℕ, 1001 * a + 1 = n * n

theorem smallestThreeDigitNumberWithPerfectSquare :
  ∀ a : ℕ, isThreeDigitNumber a → formsPerfectSquare a → a = 183 :=
by
sorry

end smallestThreeDigitNumberWithPerfectSquare_l184_184453


namespace probability_coprime_l184_184956

open BigOperators

theorem probability_coprime (A : Finset ℕ) (h : A = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let pairs := { (a, b) ∈ (A ×ˢ A) | a < b }
  let coprime_pairs := pairs.filter (λ p, Nat.gcd p.1 p.2 = 1)
  coprime_pairs.card / pairs.card = 5 / 7 := by 
sorry

end probability_coprime_l184_184956


namespace geometric_sequence_fifth_term_is_32_l184_184510

-- Defining the geometric sequence conditions
variables (a r : ℝ)

def third_term := a * r^2 = 18
def fourth_term := a * r^3 = 24
def fifth_term := a * r^4

theorem geometric_sequence_fifth_term_is_32 (h1 : third_term a r) (h2 : fourth_term a r) : 
  fifth_term a r = 32 := 
by
  sorry

end geometric_sequence_fifth_term_is_32_l184_184510


namespace kara_total_water_intake_l184_184603

-- Define dosages and water intake per tablet
def medicationA_doses_per_day := 3
def medicationB_doses_per_day := 4
def medicationC_doses_per_day := 2
def medicationD_doses_per_day := 1

def water_per_tablet_A := 4
def water_per_tablet_B := 5
def water_per_tablet_C := 6
def water_per_tablet_D := 8

-- Compute weekly water intake
def weekly_water_intake_medication (doses_per_day water_per_tablet : ℕ) (days : ℕ) : ℕ :=
  doses_per_day * water_per_tablet * days

-- Total water intake for two weeks if instructions are followed perfectly
def total_water_no_errors :=
  2 * (weekly_water_intake_medication medicationA_doses_per_day water_per_tablet_A 7 +
       weekly_water_intake_medication medicationB_doses_per_day water_per_tablet_B 7 +
       weekly_water_intake_medication medicationC_doses_per_day water_per_tablet_C 7 +
       weekly_water_intake_medication medicationD_doses_per_day water_per_tablet_D 7)

-- Missed doses in second week
def missed_water_second_week :=
  3 * water_per_tablet_A +
  2 * water_per_tablet_B +
  2 * water_per_tablet_C +
  1 * water_per_tablet_D

-- Total water actually drunk over two weeks
def total_water_real :=
  total_water_no_errors - missed_water_second_week

-- Proof statement
theorem kara_total_water_intake :
  total_water_real = 686 :=
by
  sorry

end kara_total_water_intake_l184_184603


namespace other_root_is_minus_5_l184_184515

-- conditions
def polynomial (x : ℝ) := x^4 - x^3 - 18 * x^2 + 52 * x + (-40 : ℝ)
def r1 := 2
def f_of_r1_eq_zero : polynomial r1 = 0 := by sorry -- given condition

-- the proof problem
theorem other_root_is_minus_5 : ∃ r, polynomial r = 0 ∧ r ≠ r1 ∧ r = -5 :=
by
  sorry

end other_root_is_minus_5_l184_184515


namespace range_of_m_minimum_value_l184_184315

theorem range_of_m (m n : ℝ) (h : 2 * m - n = 3) (ineq : |m| + |n + 3| ≥ 9) : 
  m ≤ -3 ∨ m ≥ 3 := 
sorry

theorem minimum_value (m n : ℝ) (h : 2 * m - n = 3) : 
  ∃ c, c = 3 ∧ c = |(5 / 3) * m - (1 / 3) * n| + |(1 / 3) * m - (2 / 3) * n| := 
sorry

end range_of_m_minimum_value_l184_184315


namespace find_original_percentage_of_acid_l184_184415

noncomputable def percentage_of_acid (a w : ℕ) : ℚ :=
  (a : ℚ) / (a + w : ℚ) * 100

theorem find_original_percentage_of_acid (a w : ℕ) 
  (h1 : (a : ℚ) / (a + w + 2 : ℚ) = 1 / 4)
  (h2 : (a + 2 : ℚ) / (a + w + 4 : ℚ) = 2 / 5) : 
  percentage_of_acid a w = 33.33 :=
by 
  sorry

end find_original_percentage_of_acid_l184_184415


namespace problem_l184_184682

theorem problem
: 15 * (1 / 17) * 34 = 30 := by
  sorry

end problem_l184_184682


namespace garden_width_is_14_l184_184504

theorem garden_width_is_14 (w : ℝ) (h1 : ∃ (l : ℝ), l = 3 * w ∧ l * w = 588) : w = 14 :=
sorry

end garden_width_is_14_l184_184504


namespace hash_hash_hash_45_l184_184292

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_45 : hash (hash (hash 45)) = 7.56 :=
by
  sorry

end hash_hash_hash_45_l184_184292


namespace revenue_from_full_price_tickets_l184_184276

theorem revenue_from_full_price_tickets (f h p : ℕ) (h1 : f + h = 160) (h2 : f * p + h * (p / 2) = 2400) : f * p = 1600 :=
by
  sorry

end revenue_from_full_price_tickets_l184_184276


namespace find_speeds_of_A_and_B_l184_184263

noncomputable def speed_A_and_B (x y : ℕ) : Prop :=
  30 * x - 30 * y = 300 ∧ 2 * x + 2 * y = 300

theorem find_speeds_of_A_and_B : ∃ (x y : ℕ), speed_A_and_B x y ∧ x = 80 ∧ y = 70 :=
by
  sorry

end find_speeds_of_A_and_B_l184_184263


namespace eccentricity_bound_l184_184914

variables {a b c e : ℝ}

-- Definitions of the problem conditions
def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def line (x : ℝ) := 2 * x
def eccentricity (c a : ℝ) := c / a

-- Proof statement in Lean
theorem eccentricity_bound (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (c : ℝ)
  (h₃ : hyperbola x y a b)
  (h₄ : ∀ x, line x ≠ y) :
  1 < eccentricity c a ∧ eccentricity c a ≤ sqrt 5 :=
sorry

end eccentricity_bound_l184_184914


namespace calculate_f_at_2x_l184_184857

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem using the given condition and the desired result
theorem calculate_f_at_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end calculate_f_at_2x_l184_184857


namespace range_of_p_l184_184872

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end range_of_p_l184_184872


namespace solution_set_of_inequality_l184_184945

theorem solution_set_of_inequality :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := 
sorry

end solution_set_of_inequality_l184_184945


namespace lines_intersect_l184_184684

theorem lines_intersect (a b : ℝ) 
  (h₁ : ∃ y : ℝ, 4 = (3/4) * y + a ∧ y = 3)
  (h₂ : ∃ x : ℝ, 3 = (3/4) * x + b ∧ x = 4) :
  a + b = 7/4 :=
sorry

end lines_intersect_l184_184684


namespace farm_transaction_difference_l184_184614

theorem farm_transaction_difference
  (x : ℕ)
  (h_initial : 6 * x - 15 > 0) -- Ensure initial horses are enough to sell 15
  (h_ratio_initial : 6 * x = x * 6)
  (h_ratio_final : (6 * x - 15) = 3 * (x + 15)) :
  (6 * x - 15) - (x + 15) = 70 :=
by
  sorry

end farm_transaction_difference_l184_184614


namespace mimi_spent_on_clothes_l184_184209

theorem mimi_spent_on_clothes :
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  total_spent - total_sneakers_cost = 2600 :=
by
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  show total_spent - total_sneakers_cost = 2600
  sorry

end mimi_spent_on_clothes_l184_184209


namespace jogger_distance_l184_184337

theorem jogger_distance (t : ℝ) (h : 16 * t = 12 * t + 10) : 12 * t = 30 :=
by
  -- Definition and proof would go here
  --
  sorry

end jogger_distance_l184_184337


namespace quadratic_solution_exists_l184_184060

theorem quadratic_solution_exists (a b : ℝ) : ∃ (x : ℝ), (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 :=
by
  sorry

end quadratic_solution_exists_l184_184060


namespace toys_produced_each_day_l184_184963

-- Define the conditions
def total_weekly_production : ℕ := 8000
def days_worked_per_week : ℕ := 4
def daily_production : ℕ := total_weekly_production / days_worked_per_week

-- The statement to be proved
theorem toys_produced_each_day : daily_production = 2000 := sorry

end toys_produced_each_day_l184_184963


namespace complement_of_M_l184_184582

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}

theorem complement_of_M :
  (U \ M) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end complement_of_M_l184_184582


namespace box_count_neither_markers_nor_erasers_l184_184288

-- Define the conditions as parameters.
def total_boxes : ℕ := 15
def markers_count : ℕ := 10
def erasers_count : ℕ := 5
def both_count : ℕ := 4

-- State the theorem to be proven in Lean 4.
theorem box_count_neither_markers_nor_erasers : 
  total_boxes - (markers_count + erasers_count - both_count) = 4 := 
sorry

end box_count_neither_markers_nor_erasers_l184_184288


namespace solution_amount_of_solution_A_l184_184660

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x + y = 140)
variables (h2 : 0.40 * x + 0.90 * y = 0.80 * 140)

-- State the theorem
theorem solution_amount_of_solution_A : x = 28 :=
by
  -- Here, the proof would be provided, but we replace it with sorry
  sorry

end solution_amount_of_solution_A_l184_184660


namespace volume_removed_percentage_l184_184658

noncomputable def volume_rect_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def volume_cube (s : ℝ) : ℝ :=
  s * s * s

noncomputable def percent_removed (original_volume removed_volume : ℝ) : ℝ :=
  (removed_volume / original_volume) * 100

theorem volume_removed_percentage :
  let l := 18
  let w := 12
  let h := 10
  let cube_side := 4
  let num_cubes := 8
  let original_volume := volume_rect_prism l w h
  let removed_volume := num_cubes * volume_cube cube_side
  percent_removed original_volume removed_volume = 23.7 := 
sorry

end volume_removed_percentage_l184_184658


namespace extreme_point_at_one_l184_184726

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end extreme_point_at_one_l184_184726


namespace count_of_sequence_l184_184719

theorem count_of_sequence : 
  let a := 156
  let d := -6
  let final_term := 36
  (∃ n, a + (n - 1) * d = final_term) -> n = 21 := 
by
  sorry

end count_of_sequence_l184_184719


namespace rightmost_three_digits_of_7_pow_2011_l184_184803

theorem rightmost_three_digits_of_7_pow_2011 :
  (7 ^ 2011) % 1000 = 7 % 1000 :=
by
  sorry

end rightmost_three_digits_of_7_pow_2011_l184_184803


namespace apples_per_pie_l184_184075

theorem apples_per_pie
  (total_apples : ℕ) (apples_handed_out : ℕ) (remaining_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 96)
  (h2 : apples_handed_out = 42)
  (h3 : remaining_apples = total_apples - apples_handed_out)
  (h4 : remaining_apples = 54)
  (h5 : number_of_pies = 9) :
  remaining_apples / number_of_pies = 6 := by
  sorry

end apples_per_pie_l184_184075


namespace ice_cream_initial_amount_l184_184559

noncomputable def initial_ice_cream (milkshake_count : ℕ) : ℕ :=
  12 * milkshake_count

theorem ice_cream_initial_amount (m_i m_f : ℕ) (milkshake_count : ℕ) (I_f : ℕ) :
  m_i = 72 →
  m_f = 8 →
  milkshake_count = (m_i - m_f) / 4 →
  I_f = initial_ice_cream milkshake_count →
  I_f = 192 :=
by
  intros hmi hmf hcount hIc
  sorry

end ice_cream_initial_amount_l184_184559


namespace prob_at_least_one_gold_prob_gold_not_more_than_silver_l184_184297

/-
Definitions corresponding to conditions:
- Number of customers surveyed: 30
- Fraction of customers who did not win any prize: 2/3
- Fraction of winning customers who won the silver prize: 3/5
- The number of customers who did not win any prize: 20
- The number of winning customers: 10
- The number of silver prize winners: 6
- The number of gold prize winners: 4
-/

def num_customers_surveyed := 30
def frac_no_prize := (2 / 3 : ℚ)
def frac_silver_winners := (3 / 5 : ℚ)
def num_no_prize := 20
def num_prize_winners := 10
def num_silver_winners := 6
def num_gold_winners := 4

/-
The claims to prove:
1. Probability that at least one of the 3 selected customers won the gold prize is 73/203
2. Probability that the number of customers who won the gold prize among the 3 selected customers 
   is not more than the number of customers who won the silver prize is 157/203
-/

theorem prob_at_least_one_gold : 
  true → (73/203 : ℚ) = 1 - (130/203 : ℚ) := sorry

theorem prob_gold_not_more_than_silver : 
  true → (157/203 : ℚ) = (130/203 : ℚ) + (27/203 : ℚ) := sorry

end prob_at_least_one_gold_prob_gold_not_more_than_silver_l184_184297


namespace jessica_final_balance_l184_184999

variable {original_balance current_balance final_balance withdrawal1 withdrawal2 deposit1 deposit2 : ℝ}

theorem jessica_final_balance:
  (2 / 5) * original_balance = 200 → 
  current_balance = original_balance - 200 → 
  withdrawal1 = (1 / 3) * current_balance → 
  current_balance - withdrawal1 = current_balance - (1 / 3 * current_balance) → 
  deposit1 = (1 / 5) * (current_balance - (1 / 3 * current_balance)) → 
  final_balance = (current_balance - (1 / 3 * current_balance)) + deposit1 → 
  deposit2 / 7 * 3 = final_balance - (current_balance - (1 / 3 * current_balance) + deposit1) → 
  (final_balance + deposit2) = 420 :=
sorry

end jessica_final_balance_l184_184999


namespace distance_A_beats_B_l184_184255

theorem distance_A_beats_B
  (time_A time_B : ℝ)
  (dist : ℝ)
  (time_A_eq : time_A = 198)
  (time_B_eq : time_B = 220)
  (dist_eq : dist = 3) :
  (dist / time_A) * time_B - dist = 333 / 1000 :=
by
  sorry

end distance_A_beats_B_l184_184255


namespace least_f_e_l184_184200

theorem least_f_e (e : ℝ) (he : e > 0) : 
  ∃ f, (∀ (a b c d : ℝ), a^3 + b^3 + c^3 + d^3 ≤ e^2 * (a^2 + b^2 + c^2 + d^2) + f * (a^4 + b^4 + c^4 + d^4)) ∧ f = 1 / (4 * e^2) :=
sorry

end least_f_e_l184_184200


namespace prove_m_equals_9_given_split_l184_184455

theorem prove_m_equals_9_given_split (m : ℕ) (h : 1 < m) (h1 : m^3 = 73) : m = 9 :=
sorry

end prove_m_equals_9_given_split_l184_184455


namespace boat_stream_ratio_l184_184429

theorem boat_stream_ratio (B S : ℝ) (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l184_184429


namespace find_m_value_l184_184941

theorem find_m_value : 
  ∃ (m : ℝ), 
  (∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 1 ∧ (x - y + m = 0)) → m = -3 :=
by
  sorry

end find_m_value_l184_184941


namespace lemon_pie_degrees_l184_184737

-- Defining the constants
def total_students : ℕ := 45
def chocolate_pie : ℕ := 15
def apple_pie : ℕ := 10
def blueberry_pie : ℕ := 9

-- Defining the remaining students
def remaining_students := total_students - (chocolate_pie + apple_pie + blueberry_pie)

-- Half of the remaining students prefer cherry pie and half prefer lemon pie
def students_prefer_cherry : ℕ := remaining_students / 2
def students_prefer_lemon : ℕ := remaining_students / 2

-- Defining the degree measure function
def degrees (students : ℕ) := (students * 360) / total_students

-- Proof statement
theorem lemon_pie_degrees : degrees students_prefer_lemon = 48 := by
  sorry  -- proof omitted

end lemon_pie_degrees_l184_184737


namespace relationship_ab_l184_184321

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 1) = -f x)
variable (a b : ℝ)
variable (h_ex : ∃ x : ℝ, f (a + x) = f (b - x))

-- State the conclusion we need to prove
theorem relationship_ab : ∃ k : ℕ, k > 0 ∧ (a + b) = 2 * k + 1 :=
by
  sorry

end relationship_ab_l184_184321


namespace maximum_marks_l184_184206

theorem maximum_marks (M : ℝ) (h1 : 212 + 25 = 237) (h2 : 0.30 * M = 237) : M = 790 := 
by
  sorry

end maximum_marks_l184_184206


namespace maximum_area_of_rectangle_l184_184545

theorem maximum_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : ∃ A, A = 100 ∧ ∀ x' y', 2 * x' + 2 * y' = 40 → x' * y' ≤ A := by
  sorry

end maximum_area_of_rectangle_l184_184545


namespace problem1_problem2_l184_184299

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end problem1_problem2_l184_184299


namespace num_ways_to_turn_off_lights_l184_184274

-- Let's define our problem in terms of the conditions given
-- Define the total number of lights
def total_lights : ℕ := 12

-- Define that we need to turn off 3 lights
def lights_to_turn_off : ℕ := 3

-- Define that we have 10 possible candidates for being turned off 
def candidates := total_lights - 2

-- Define the gap consumption statement that effectively reduce choices to 7 lights
def effective_choices := candidates - lights_to_turn_off

-- Define the combination formula for the number of ways to turn off the lights
def num_ways := Nat.choose effective_choices lights_to_turn_off

-- Final statement to prove
theorem num_ways_to_turn_off_lights : num_ways = Nat.choose 7 3 :=
by
  sorry

end num_ways_to_turn_off_lights_l184_184274


namespace john_bought_metres_l184_184748

-- Define the conditions
def total_cost := 425.50
def cost_per_metre := 46.00

-- State the theorem
theorem john_bought_metres : total_cost / cost_per_metre = 9.25 :=
by
  sorry

end john_bought_metres_l184_184748


namespace tod_driving_time_l184_184952
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end tod_driving_time_l184_184952


namespace smaller_area_l184_184256

theorem smaller_area (x y : ℝ) 
  (h1 : x + y = 900)
  (h2 : y - x = (1 / 5) * (x + y) / 2) :
  x = 405 :=
sorry

end smaller_area_l184_184256


namespace quadratic_real_roots_k_eq_one_l184_184326

theorem quadratic_real_roots_k_eq_one 
  (k : ℕ) 
  (h_nonneg : k ≥ 0) 
  (h_real_roots : ∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) : 
  k = 1 := 
sorry

end quadratic_real_roots_k_eq_one_l184_184326


namespace find_angle_D_l184_184583

noncomputable def calculate_angle (A B C D : ℝ) : ℝ :=
  if (A + B = 180) ∧ (C = D) ∧ (A = 2 * D - 10) then D else 0

theorem find_angle_D (A B C D : ℝ) (h1: A + B = 180) (h2: C = D) (h3: A = 2 * D - 10) : D = 70 :=
by
  sorry

end find_angle_D_l184_184583


namespace goldie_worked_hours_last_week_l184_184470

variable (H : ℕ)
variable (money_per_hour : ℕ := 5)
variable (hours_this_week : ℕ := 30)
variable (total_earnings : ℕ := 250)

theorem goldie_worked_hours_last_week :
  H = (total_earnings - hours_this_week * money_per_hour) / money_per_hour :=
sorry

end goldie_worked_hours_last_week_l184_184470


namespace rectangle_ratio_l184_184308

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end rectangle_ratio_l184_184308


namespace yellow_marbles_count_l184_184484

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end yellow_marbles_count_l184_184484


namespace basketball_player_score_prob_l184_184973

variable (p : ℚ) (n : ℕ)

def independent_shots (prob : ℚ) (shots : ℕ) (score_prob : ℚ) : Prop :=
  ∀ (p : ℚ), score_prob = prob^(shots)

theorem basketball_player_score_prob
  (prob_shooting : ℚ) (score_target : ℚ) (shots : ℕ)
  (h_prob : prob_shooting = 0.7)
  (h_shots : shots = 3)
  (h_target : score_target = 0.343) :
  independent_shots prob_shooting shots score_target :=
begin
  assume prob,
  have h : score_target = prob_shooting ^ shots,
  {
    rw [h_prob, h_shots],
    norm_num,
  },
  exact h_target.symm ▸ h,
end

end basketball_player_score_prob_l184_184973


namespace distinct_sequences_l184_184331

theorem distinct_sequences : ∃ n : ℕ, n = 392 ∧
  (∃ seqs: Finset (List Char), seqs.card = n ∧
    ∀ s ∈ seqs, s.head = 'M' ∧ s.ilast ≠ 'A' ∧ s.length = 4 ∧ (∀ c ∈ s.tails, c.length ≤ 1 → c.nodup)) := 
begin
  sorry
end

end distinct_sequences_l184_184331


namespace find_a2_l184_184869

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end find_a2_l184_184869


namespace simplify_arithmetic_expr1_simplify_arithmetic_expr2_l184_184262

-- Problem 1 Statement
theorem simplify_arithmetic_expr1 (x y : ℝ) : 
  (x - 3 * y) - (y - 2 * x) = 3 * x - 4 * y :=
sorry

-- Problem 2 Statement
theorem simplify_arithmetic_expr2 (a b : ℝ) : 
  5 * a * b^2 - 3 * (2 * a^2 * b - 2 * (a^2 * b - 2 * a * b^2)) = -7 * a * b^2 :=
sorry

end simplify_arithmetic_expr1_simplify_arithmetic_expr2_l184_184262


namespace solve_for_x_l184_184136

def star (a b : ℕ) := a * b + a + b

theorem solve_for_x : ∃ x : ℕ, star 3 x = 27 ∧ x = 6 :=
by {
  sorry
}

end solve_for_x_l184_184136


namespace tamtam_blue_shells_l184_184626

theorem tamtam_blue_shells 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (orange_shells : ℕ)
  (H_total : total_shells = 65)
  (H_purple : purple_shells = 13)
  (H_pink : pink_shells = 8)
  (H_yellow : yellow_shells = 18)
  (H_orange : orange_shells = 14) :
  ∃ blue_shells : ℕ, blue_shells = 12 :=
by
  sorry

end tamtam_blue_shells_l184_184626


namespace part_a_part_b_l184_184537

theorem part_a (k : ℕ) : ∃ (a : ℕ → ℕ), (∀ i, i ≤ k → a i > 0) ∧ (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → a i < a j) ∧ (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) :=
sorry

theorem part_b : ∃ C > 0, ∀ a : ℕ → ℕ, (∀ k : ℕ, (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) → a 1 > (k : ℕ) ^ (C * k : ℕ)) :=
sorry

end part_a_part_b_l184_184537


namespace num_int_values_satisfying_inequality_l184_184154

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l184_184154


namespace ions_electron_shell_structure_l184_184456

theorem ions_electron_shell_structure
  (a b n m : ℤ) 
  (same_electron_shell_structure : a + n = b - m) :
  a + m = b - n :=
by
  sorry

end ions_electron_shell_structure_l184_184456


namespace geometric_sequence_common_ratio_l184_184042

theorem geometric_sequence_common_ratio
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : S 1 = a 1)
  (h2 : S 2 = a 1 + a 1 * q)
  (h3 : a 2 = a 1 * q)
  (h4 : a 3 = a 1 * q^2)
  (h5 : 3 * S 2 = a 3 - 2)
  (h6 : 3 * S 1 = a 2 - 2) :
  q = 4 :=
sorry

end geometric_sequence_common_ratio_l184_184042


namespace sealed_envelope_problem_l184_184547

theorem sealed_envelope_problem :
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) →
  ((n = 12 ∧ (n % 10 ≠ 2) ∧ n ≠ 35 ∧ (n % 10 ≠ 5)) ∨
   (n ≠ 12 ∧ (n % 10 ≠ 2) ∧ n = 35 ∧ (n % 10 = 5))) →
  ¬(n % 10 ≠ 5) :=
by
  sorry

end sealed_envelope_problem_l184_184547


namespace simplify_expression_l184_184624

theorem simplify_expression:
  (a = 2) ∧ (b = 1) →
  - (1 / 3 : ℚ) * (a^3 * b - a * b) 
  + a * b^3 
  - (a * b - b) / 2 
  - b / 2 
  + (1 / 3 : ℚ) * (a^3 * b) 
  = (5 / 3 : ℚ) := by 
  intros h
  simp [h.1, h.2]
  sorry

end simplify_expression_l184_184624


namespace sqrt_square_eq_17_l184_184238

theorem sqrt_square_eq_17 :
  (Real.sqrt 17) ^ 2 = 17 :=
sorry

end sqrt_square_eq_17_l184_184238


namespace minimum_value_l184_184146

noncomputable def expr (x y : ℝ) := x^2 + x * y + y^2 - 3 * y

theorem minimum_value :
  ∃ x y : ℝ, expr x y = -3 ∧
  ∀ x' y' : ℝ, expr x' y' ≥ -3 :=
sorry

end minimum_value_l184_184146


namespace proposition_2_proposition_3_only_propositions_2_and_3_are_correct_l184_184584

open Real

def f (x : ℝ) := 4 * sin (2 * x + π / 3)

theorem proposition_2 : ∀ x : ℝ, f x = 4 * cos (2 * x - π / 6) := sorry

theorem proposition_3 : ∃ x : ℝ, (f x = 0) ∧ ∀ x' : ℝ, f x' = f (2 * (-π / 6) - x') := sorry

theorem only_propositions_2_and_3_are_correct : 
  (proposition_2 ∧ proposition_3) ∧ 
  ¬ (∀ x1 x2 : ℝ, (f x1 = 0 ∧ f x2 = 0) → (x1 - x2) % π = 0) ∧ 
  ¬ (∃ x : ℝ, ∀ x' : ℝ, f x' = f (2 * x - π / 6)) := sorry

end proposition_2_proposition_3_only_propositions_2_and_3_are_correct_l184_184584


namespace b_is_perfect_square_l184_184755

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem b_is_perfect_square (a b : ℕ)
  (h_positive : 0 < a) (h_positive_b : 0 < b)
  (h_gcd_lcm_multiple : (Nat.gcd a b + Nat.lcm a b) % (a + 1) = 0)
  (h_le : b ≤ a) : is_perfect_square b :=
sorry

end b_is_perfect_square_l184_184755


namespace number_of_girls_l184_184904

-- Define the number of boys and girls as natural numbers
variable (B G : ℕ)

-- First condition: The number of girls is 458 more than the number of boys
axiom h1 : G = B + 458

-- Second condition: The total number of pupils is 926
axiom h2 : G + B = 926

-- The theorem to be proved: The number of girls is 692
theorem number_of_girls : G = 692 := by
  sorry

end number_of_girls_l184_184904


namespace min_value_x_fraction_l184_184160

theorem min_value_x_fraction (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ ∀ y > 1, y + 1 / (y - 1) ≥ m :=
by
  sorry

end min_value_x_fraction_l184_184160


namespace initial_mixtureA_amount_l184_184611

-- Condition 1: Mixture A is 20% oil and 80% material B by weight.
def oil_content (x : ℝ) : ℝ := 0.20 * x
def materialB_content (x : ℝ) : ℝ := 0.80 * x

-- Condition 2: 2 more kilograms of oil are added to a certain amount of mixture A
def oil_added := 2

-- Condition 3: 6 kilograms of mixture A must be added to make a 70% material B in the new mixture.
def mixture_added := 6

-- The total weight of the new mixture
def total_weight (x : ℝ) : ℝ := x + mixture_added + oil_added

-- The total amount of material B in the new mixture
def total_materialB (x : ℝ) : ℝ := 0.80 * x + 0.80 * mixture_added

-- The new mixture is supposed to be 70% material B.
def is_70_percent_materialB (x : ℝ) : Prop := total_materialB x = 0.70 * total_weight x

-- Proving x == 8 given the conditions
theorem initial_mixtureA_amount : ∃ x : ℝ, is_70_percent_materialB x ∧ x = 8 :=
by
  sorry

end initial_mixtureA_amount_l184_184611


namespace tim_total_trip_time_l184_184095

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end tim_total_trip_time_l184_184095


namespace parallelogram_midpoints_XY_square_l184_184597

theorem parallelogram_midpoints_XY_square (A B C D X Y : ℝ)
  (AB CD : ℝ) (BC DA : ℝ) (angle_D : ℝ)
  (mid_X : X = (B + C) / 2) (mid_Y : Y = (D + A) / 2)
  (h1: AB = 10) (h2: BC = 17) (h3: CD = 10) (h4 : angle_D = 60) :
  (XY ^ 2 = 219 / 4) :=
by
  sorry

end parallelogram_midpoints_XY_square_l184_184597


namespace find_n_l184_184754

variable {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
variable (h : 1/a + 1/b + 1/c = 1/(a + b + c))

theorem find_n (n : ℤ) : (∃ k : ℕ, n = 2 * k - 1) → 
  (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) :=
by
  sorry

end find_n_l184_184754


namespace y_intercept_of_line_l184_184843

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 6 * y = 24) : y = 4 := by
  sorry

end y_intercept_of_line_l184_184843


namespace circle_iff_m_gt_neg_1_over_2_l184_184593

noncomputable def represents_circle (m: ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + x + y - m = 0) → m > -1/2

theorem circle_iff_m_gt_neg_1_over_2 (m : ℝ) : represents_circle m ↔ m > -1/2 := by
  sorry

end circle_iff_m_gt_neg_1_over_2_l184_184593


namespace find_positive_integer_k_l184_184313

theorem find_positive_integer_k (p : ℕ) (hp : Prime p) (hp2 : Odd p) : 
  ∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, n * n = k - p * k ∧ k = ((p + 1) * (p + 1)) / 4 :=
by
  sorry

end find_positive_integer_k_l184_184313


namespace count_5_letter_words_with_at_least_one_vowel_l184_184885

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l184_184885


namespace rocky_training_miles_l184_184741

variable (x : ℕ)

theorem rocky_training_miles (h1 : x + 2 * x + 6 * x = 36) : x = 4 :=
by
  -- proof
  sorry

end rocky_training_miles_l184_184741


namespace last_digit_of_product_is_zero_l184_184038

theorem last_digit_of_product_is_zero : ∃ (B И H П У Х : ℕ), 
  B ≠ И ∧ B ≠ H ∧ B ≠ П ∧ B ≠ У ∧ B ≠ Х ∧ И ≠ H ∧ И ≠ П ∧ И ≠ У ∧ И ≠ Х ∧ 
  H ≠ П ∧ H ≠ У ∧ H ≠ Х ∧ П ≠ У ∧ П ≠ Х ∧ У ≠ Х ∧
  B ∈ {2, 3, 4, 5, 6, 7} ∧ И ∈ {2, 3, 4, 5, 6, 7} ∧ H ∈ {2, 3, 4, 5, 6, 7} ∧
  П ∈ {2, 3, 4, 5, 6, 7} ∧ У ∈ {2, 3, 4, 5, 6, 7} ∧ Х ∈ {2, 3, 4, 5, 6, 7} ∧
  (B * И * H * H * И * П * У * Х) % 10 = 0 := by
  sorry

end last_digit_of_product_is_zero_l184_184038


namespace number_of_valid_arrangements_l184_184191

def total_permutations (n : ℕ) : ℕ := n.factorial

def valid_permutations (total : ℕ) (block : ℕ) (specific_restriction : ℕ) : ℕ :=
  total - specific_restriction

theorem number_of_valid_arrangements : valid_permutations (total_permutations 5) 48 24 = 96 :=
by
  sorry

end number_of_valid_arrangements_l184_184191


namespace hyperbola_perimeter_l184_184083

-- Lean 4 statement
theorem hyperbola_perimeter (a b m : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (A B : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), (x,y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})
  (line_through_F1 : ∀ (x y : ℝ), x = F1.1)
  (A_B_on_hyperbola : (A.1^2/a^2 - A.2^2/b^2 = 1) ∧ (B.1^2/a^2 - B.2^2/b^2 = 1))
  (dist_AB : dist A B = m)
  (dist_relations : dist A F2 + dist B F2 - (dist A F1 + dist B F1) = 4 * a) : 
  dist A F2 + dist B F2 + dist A B = 4 * a + 2 * m :=
sorry

end hyperbola_perimeter_l184_184083


namespace integer_values_count_l184_184150

theorem integer_values_count (x : ℤ) : 
  (∃ n : ℕ, n = 8 ∧ {x : ℤ | x^2 < 9 * x}.to_finset.card = n) :=
by
  sorry

end integer_values_count_l184_184150


namespace janet_time_per_post_l184_184747

/-- Janet gets paid $0.25 per post she checks. She earns $90 per hour. 
    Prove that it takes her 10 seconds to check a post. -/
theorem janet_time_per_post
  (payment_per_post : ℕ → ℝ)
  (hourly_pay : ℝ)
  (posts_checked_hourly : ℕ)
  (secs_per_post : ℝ) :
  payment_per_post 1 = 0.25 →
  hourly_pay = 90 →
  hourly_pay = payment_per_post (posts_checked_hourly) →
  secs_per_post = 10 :=
sorry

end janet_time_per_post_l184_184747


namespace find_missing_number_l184_184446

theorem find_missing_number (x : ℕ) (h : 10111 - 10 * 2 * x = 10011) : x = 5 :=
sorry

end find_missing_number_l184_184446


namespace roots_reciprocal_sum_l184_184046

theorem roots_reciprocal_sum
  {a b c : ℂ}
  (h_roots : ∀ x : ℂ, (x - a) * (x - b) * (x - c) = x^3 - x + 1) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = -2 :=
by
  sorry

end roots_reciprocal_sum_l184_184046


namespace no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l184_184007

def P (x : ℝ) : Prop := x ^ 2 - 8 * x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_m_for_necessary_and_sufficient_condition :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
by sorry

theorem m_geq_3_for_necessary_condition :
  ∃ m : ℝ, (m ≥ 3) ∧ ∀ x : ℝ, S x m → P x :=
by sorry

end no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l184_184007


namespace triangle_ABC_angles_l184_184844

theorem triangle_ABC_angles :
  ∃ (θ φ ω : ℝ), θ = 36 ∧ φ = 72 ∧ ω = 72 ∧
  (ω + φ + θ = 180) ∧
  (2 * ω + θ = 180) ∧
  (φ = 2 * θ) :=
by
  sorry

end triangle_ABC_angles_l184_184844


namespace number_of_yellow_marbles_l184_184486

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end number_of_yellow_marbles_l184_184486


namespace monotonic_intervals_of_f_g_minus_f_less_than_3_l184_184863

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x < -1 / Real.exp 1 → f x < f (-1 / Real.exp 1) ∧ x > -1 / Real.exp 1 → f x > f (-1 / Real.exp 1) := sorry

theorem g_minus_f_less_than_3 :
  ∀ x : ℝ, x < 0 → g x - f x < 3 := sorry

end monotonic_intervals_of_f_g_minus_f_less_than_3_l184_184863


namespace tangent_line_characterization_l184_184761

theorem tangent_line_characterization 
  (α β m n : ℝ) 
  (h_pos_α : 0 < α) 
  (h_pos_β : 0 < β) 
  (h_alpha_beta : 1/α + 1/β = 1)
  (h_pos_m : 0 < m)
  (h_pos_n : 0 < n) :
  (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ x^α + y^α = 1 → mx + ny = 1) ↔ (m^β + n^β = 1) := 
sorry

end tangent_line_characterization_l184_184761


namespace largest_fully_communicating_sets_eq_l184_184549

noncomputable def largest_fully_communicating_sets :=
  let total_sets := Nat.choose 99 4
  let non_communicating_sets_per_pod := Nat.choose 48 3
  let total_non_communicating_sets := 99 * non_communicating_sets_per_pod
  total_sets - total_non_communicating_sets

theorem largest_fully_communicating_sets_eq : largest_fully_communicating_sets = 2051652 := by
  sorry

end largest_fully_communicating_sets_eq_l184_184549


namespace total_gifts_l184_184054

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l184_184054


namespace andrew_expected_distinct_colors_l184_184830

noncomputable def expected_distinct_colors_picks (balls: ℕ) (picks: ℕ) : ℚ :=
  let prob_not_picked_once := (balls - 1) / balls
  let prob_not_picked := prob_not_picked_once ^ picks
  let prob_picked := 1 - prob_not_picked
  balls * prob_picked

theorem andrew_expected_distinct_colors :
  (expected_distinct_colors_picks 10 4) = (3439 / 1000) :=
by sorry

end andrew_expected_distinct_colors_l184_184830


namespace reese_practice_hours_l184_184619

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l184_184619


namespace original_decimal_number_l184_184818

theorem original_decimal_number (x : ℝ) (h : 10 * x - x / 10 = 23.76) : x = 2.4 :=
sorry

end original_decimal_number_l184_184818


namespace max_distance_with_optimal_tire_swapping_l184_184854

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end max_distance_with_optimal_tire_swapping_l184_184854


namespace best_fitting_model_is_model3_l184_184903

-- Definitions of the coefficients of determination for the models
def R2_model1 : ℝ := 0.60
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.98
def R2_model4 : ℝ := 0.25

-- The best fitting effect corresponds to the highest R^2 value
theorem best_fitting_model_is_model3 :
  R2_model3 = max (max R2_model1 R2_model2) (max R2_model3 R2_model4) :=
by {
  -- Proofblock is skipped, using sorry
  sorry
}

end best_fitting_model_is_model3_l184_184903


namespace prime_p_range_l184_184024

open Classical

variable {p : ℤ} (hp_prime : Prime p)

def is_integer_root (a b c : ℤ) := 
  ∃ x y : ℤ, x * y = c ∧ x + y = -b

theorem prime_p_range (hp_roots : is_integer_root 1 p (-500 * p)) : 1 < p ∧ p ≤ 10 :=
by
  sorry

end prime_p_range_l184_184024


namespace initial_bottle_caps_l184_184839

theorem initial_bottle_caps (X : ℕ) (h1 : X - 60 + 58 = 67) : X = 69 := by
  sorry

end initial_bottle_caps_l184_184839


namespace tangent_curve_l184_184174

variable {k a b : ℝ}

theorem tangent_curve (h1 : 3 = (1 : ℝ)^3 + a * 1 + b)
(h2 : k = 2)
(h3 : k = 3 * (1 : ℝ)^2 + a) :
b = 3 :=
by
  sorry

end tangent_curve_l184_184174


namespace larger_number_is_33_l184_184648

theorem larger_number_is_33 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : max x y = 33 :=
sorry

end larger_number_is_33_l184_184648


namespace greatest_integer_c_l184_184451

theorem greatest_integer_c (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 0) → c = 6 :=
by
  sorry

end greatest_integer_c_l184_184451


namespace reciprocal_eq_self_l184_184942

theorem reciprocal_eq_self (x : ℝ) : (1 / x = x) ↔ (x = 1 ∨ x = -1) :=
sorry

end reciprocal_eq_self_l184_184942


namespace Tina_profit_correct_l184_184406

theorem Tina_profit_correct :
  ∀ (price_per_book cost_per_book books_per_customer total_customers : ℕ),
  price_per_book = 20 →
  cost_per_book = 5 →
  books_per_customer = 2 →
  total_customers = 4 →
  (price_per_book * (books_per_customer * total_customers) - 
   cost_per_book * (books_per_customer * total_customers) = 120) :=
by
  intros price_per_book cost_per_book books_per_customer total_customers
  sorry

end Tina_profit_correct_l184_184406


namespace valid_k_sum_correct_l184_184013

def sum_of_valid_k : ℤ :=
  (List.range 17).sum * 1734 + (List.range 17).sum * 3332

theorem valid_k_sum_correct : sum_of_valid_k = 5066 := by
  sorry

end valid_k_sum_correct_l184_184013


namespace increasing_sequence_and_limit_l184_184756

variables {a b : ℝ} {f g : ℝ → ℝ} (n : ℕ) [decidable_lt ℝ] 

noncomputable def I_n (a b : ℝ) (f g : ℝ → ℝ) (n : ℕ) : ℝ :=
  ∫ x in set.Icc a b, (f x) ^ (n + 1) / (g x) ^ n

theorem increasing_sequence_and_limit 
  (a b : ℝ) (f g : ℝ → ℝ)
  (h_ab : a < b)
  (h_f_cont : continuous_on f (set.Icc a b))
  (h_g_cont : continuous_on g (set.Icc a b))
  (h_f_pos : ∀ x ∈ set.Icc a b, 0 < f x)
  (h_g_pos : ∀ x ∈ set.Icc a b, 0 < g x)
  (h_int_eq : ∫ x in set.Icc a b, f x = ∫ x in set.Icc a b, g x)
  (h_f_ne_g : ∃ x ∈ set.Icc a b, f x ≠ g x) :
  (∀ n : ℕ, I_n a b f g (n + 1) > I_n a b f g n) ∧
  tendsto (λ n, I_n a b f g n) at_top at_top :=
begin
  sorry
end

end increasing_sequence_and_limit_l184_184756


namespace number_of_girls_more_than_boys_l184_184977

theorem number_of_girls_more_than_boys
    (total_students : ℕ)
    (number_of_boys : ℕ)
    (h1 : total_students = 485)
    (h2 : number_of_boys = 208) :
    total_students - number_of_boys - number_of_boys = 69 :=
by
    sorry

end number_of_girls_more_than_boys_l184_184977


namespace simplify_expression_l184_184067

variable (y : ℝ)

theorem simplify_expression : 
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + 3 * y ^ 8) = 
  15 * y ^ 13 - y ^ 12 + 3 * y ^ 11 + 15 * y ^ 10 - y ^ 9 - 6 * y ^ 8 :=
by
  sorry

end simplify_expression_l184_184067


namespace five_letter_words_with_vowel_l184_184886

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l184_184886


namespace minimize_y_at_x_l184_184291

noncomputable def minimize_y (a b x : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + 2 * (a - b) * x

theorem minimize_y_at_x (a b : ℝ) :
  ∃ x : ℝ, minimize_y a b x = minimize_y a b (b / 2) := by
  sorry

end minimize_y_at_x_l184_184291


namespace terry_problems_wrong_l184_184072

theorem terry_problems_wrong (R W : ℕ) 
  (h1 : R + W = 25) 
  (h2 : 4 * R - W = 85) : 
  W = 3 := 
by
  sorry

end terry_problems_wrong_l184_184072


namespace prove_a_lt_neg_one_l184_184205

variable {f : ℝ → ℝ} (a : ℝ)

-- Conditions:
-- 1. f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- 2. f has a period of 3
def has_period_three (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

-- 3. f(1) > 1
def f_one_gt_one (f : ℝ → ℝ) : Prop := f 1 > 1

-- 4. f(2) = a
def f_two_eq_a (f : ℝ → ℝ) (a : ℝ) : Prop := f 2 = a

-- Proof statement:
theorem prove_a_lt_neg_one (h1 : is_odd_function f) (h2 : has_period_three f)
  (h3 : f_one_gt_one f) (h4 : f_two_eq_a f a) : a < -1 :=
  sorry

end prove_a_lt_neg_one_l184_184205


namespace scientific_notation_of_213_million_l184_184553

theorem scientific_notation_of_213_million : ∃ (n : ℝ), (213000000 : ℝ) = 2.13 * 10^8 :=
by
  sorry

end scientific_notation_of_213_million_l184_184553


namespace graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l184_184390

theorem graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines :
  ∀ x y : ℝ, (x^2 - y^2 = 0) ↔ (y = x ∨ y = -x) := 
by
  sorry

end graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l184_184390


namespace part1_part2_l184_184860

open Complex

noncomputable def z0 : ℂ := 3 + 4 * Complex.I

theorem part1 (z1 : ℂ) (h : z1 * z0 = 3 * z1 + z0) : z1.im = -3/4 := by
  sorry

theorem part2 (x : ℝ) 
    (z : ℂ := (x^2 - 4 * x) + (x + 2) * Complex.I) 
    (z0_conj : ℂ := 3 - 4 * Complex.I) 
    (h : (z + z0_conj).re < 0 ∧ (z + z0_conj).im > 0) : 
    2 < x ∧ x < 3 :=
  by 
  sorry

end part1_part2_l184_184860


namespace five_letter_words_with_vowel_l184_184889

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l184_184889


namespace hyperbola_no_intersection_l184_184912

theorem hyperbola_no_intersection (a b e : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) :
  (√5 ≥ e ∧ 1 < e) → ∀ x y : ℝ, ¬ (y = 2 * x ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  intros h_intersect x y,
  sorry,
end

end hyperbola_no_intersection_l184_184912


namespace harvesting_days_l184_184028

theorem harvesting_days :
  (∀ (harvesters : ℕ) (days : ℕ) (mu : ℕ), 2 * 3 * (75 : ℕ) = 450) →
  (7 * 4 * (75 : ℕ) = 2100) :=
by
  sorry

end harvesting_days_l184_184028


namespace integer_part_sqrt_sum_l184_184162

theorem integer_part_sqrt_sum {a b c : ℤ} 
  (h_a : |a| = 4) 
  (h_b_sqrt : b^2 = 9) 
  (h_c_cubert : c^3 = -8) 
  (h_order : a > b ∧ b > c) 
  : (⌊ Real.sqrt (a + b + c) ⌋) = 2 := 
by 
  sorry

end integer_part_sqrt_sum_l184_184162


namespace fraction_simplification_l184_184448

theorem fraction_simplification :
  (2/5 + 3/4) / (4/9 + 1/6) = (207/110) := by
  sorry

end fraction_simplification_l184_184448


namespace least_prime_b_l184_184073

-- Define what it means for an angle to be a right triangle angle sum
def isRightTriangleAngleSum (a b : ℕ) : Prop := a + b = 90

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Formalize the goal: proving that the smallest possible b is 7
theorem least_prime_b (a b : ℕ) (h1 : isRightTriangleAngleSum a b) (h2 : isPrime a) (h3 : isPrime b) (h4 : a > b) : b = 7 :=
sorry

end least_prime_b_l184_184073


namespace min_value_x_plus_4y_l184_184168

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 2 * x * y) : x + 4 * y = 9 / 2 :=
by
  sorry

end min_value_x_plus_4y_l184_184168


namespace total_books_on_shelves_l184_184092

def num_shelves : ℕ := 520
def books_per_shelf : ℝ := 37.5

theorem total_books_on_shelves : num_shelves * books_per_shelf = 19500 :=
by
  sorry

end total_books_on_shelves_l184_184092


namespace power_equivalence_l184_184643

theorem power_equivalence (L : ℕ) : 32^4 * 4^5 = 2^L → L = 30 :=
by
  sorry

end power_equivalence_l184_184643


namespace total_number_of_items_l184_184810

-- Define the conditions as equations in Lean
def model_cars_price := 5
def model_trains_price := 8
def total_amount := 31

-- Initialize the variable definitions for number of cars and trains
variables (c t : ℕ)

-- The proof problem: Show that given the equation, the sum of cars and trains is 5
theorem total_number_of_items : (model_cars_price * c + model_trains_price * t = total_amount) → (c + t = 5) := by
  -- Proof steps would go here
  sorry

end total_number_of_items_l184_184810


namespace at_least_100_arcs_of_21_points_l184_184795

noncomputable def count_arcs (n : ℕ) (θ : ℝ) : ℕ :=
-- Please note this function needs to be defined appropriately, here we assume it computes the number of arcs of θ degrees or fewer between n points on a circle
sorry

theorem at_least_100_arcs_of_21_points :
  ∃ (n : ℕ), n = 21 ∧ count_arcs n (120 : ℝ) ≥ 100 :=
sorry

end at_least_100_arcs_of_21_points_l184_184795


namespace parabola_vertex_l184_184935

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end parabola_vertex_l184_184935


namespace min_value_x_fraction_l184_184159

theorem min_value_x_fraction (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ ∀ y > 1, y + 1 / (y - 1) ≥ m :=
by
  sorry

end min_value_x_fraction_l184_184159


namespace find_number_of_piles_l184_184495

theorem find_number_of_piles 
  (Q : ℕ) 
  (h1 : Q = Q) 
  (h2 : ∀ (piles : ℕ), piles = 3) 
  (total_coins : ℕ) 
  (h3 : total_coins = 30) 
  (e : 6 * Q = total_coins) :
  Q = 5 := 
by sorry

end find_number_of_piles_l184_184495


namespace find_number_l184_184051

theorem find_number (x : ℝ) : x = 7 ∧ x^2 + 95 = (x - 19)^2 :=
by
  sorry

end find_number_l184_184051


namespace calc_expr_correct_l184_184834

noncomputable def eval_expr : ℚ :=
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5

theorem calc_expr_correct : eval_expr = 12.5 :=
by
  sorry

end calc_expr_correct_l184_184834


namespace solution_set_of_fraction_inequality_l184_184944

theorem solution_set_of_fraction_inequality (a b x : ℝ) (h1: ∀ x, ax - b > 0 ↔ x ∈ Set.Iio 1) (h2: a < 0) (h3: a - b = 0) :
  ∀ x, (a * x + b) / (x - 2) > 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 2 := 
sorry

end solution_set_of_fraction_inequality_l184_184944


namespace find_C_l184_184119

theorem find_C
  (A B C : ℕ)
  (h1 : A + B + C = 1000)
  (h2 : A + C = 700)
  (h3 : B + C = 600) :
  C = 300 := by
  sorry

end find_C_l184_184119


namespace total_potatoes_l184_184479

theorem total_potatoes (jane_potatoes mom_potatoes dad_potatoes : Nat) 
  (h1 : jane_potatoes = 8)
  (h2 : mom_potatoes = 8)
  (h3 : dad_potatoes = 8) :
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  sorry

end total_potatoes_l184_184479


namespace simplify_expression_l184_184181

theorem simplify_expression (a b : ℂ) (x : ℂ) (hb : b ≠ 0) (ha : a ≠ b) (hx : x = a / b) :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  -- Proof goes here
  sorry

end simplify_expression_l184_184181


namespace part3_l184_184173

noncomputable def f (x a : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * Real.log x

theorem part3 (a : ℝ) : 
  (∀ x > 1, f x a > 0) ↔ a ∈ Set.Iic 0 := 
sorry

end part3_l184_184173


namespace distance_BF_l184_184012

-- Given the focus F of the parabola y^2 = 4x
def focus_of_parabola : (ℝ × ℝ) := (1, 0)

-- Points A and B lie on the parabola y^2 = 4x
def point_A (x y : ℝ) := y^2 = 4 * x
def point_B (x y : ℝ) := y^2 = 4 * x

-- The line through F intersects the parabola at points A and B, and |AF| = 2
def distance_AF : ℝ := 2

-- Prove that |BF| = 2
theorem distance_BF : ∀ (A B F : ℝ × ℝ), 
  A = (1, F.2) → 
  B = (1, -F.2) → 
  F = (1, 0) → 
  |A.1 - F.1| + |A.2 - F.2| = distance_AF → 
  |B.1 - F.1| + |B.2 - F.2| = 2 :=
by
  intros A B F hA hB hF d_AF
  sorry

end distance_BF_l184_184012


namespace problem_statement_l184_184334

-- Define the problem
theorem problem_statement (a b : ℝ) (h : a - b = 1 / 2) : -3 * (b - a) = 3 / 2 := 
  sorry

end problem_statement_l184_184334


namespace arithmetic_seq_a7_l184_184348

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 50) 
  (h_a5 : a 5 = 30) : 
  a 7 = 10 := 
by
  sorry

end arithmetic_seq_a7_l184_184348


namespace quadratic_no_real_roots_l184_184656

-- Define the quadratic polynomial f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions: f(x) = x has no real roots
theorem quadratic_no_real_roots (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
sorry

end quadratic_no_real_roots_l184_184656


namespace value_20_percent_greater_l184_184102

theorem value_20_percent_greater (x : ℝ) : (x = 88 * 1.20) ↔ (x = 105.6) :=
by
  sorry

end value_20_percent_greater_l184_184102


namespace max_value_expr_l184_184203

theorem max_value_expr (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) : 
  10 * x + 3 * y + 15 * z ≤ 9.455 :=
sorry

end max_value_expr_l184_184203


namespace integer_values_count_l184_184151

theorem integer_values_count (x : ℤ) : 
  (∃ n : ℕ, n = 8 ∧ {x : ℤ | x^2 < 9 * x}.to_finset.card = n) :=
by
  sorry

end integer_values_count_l184_184151


namespace floor_ceil_difference_l184_184675

theorem floor_ceil_difference : 
  let a := (18 / 5) * (-33 / 4)
  let b := ⌈(-33 / 4 : ℝ)⌉
  let c := (18 / 5) * (b : ℝ)
  let d := ⌈c⌉
  ⌊a⌋ - d = -2 :=
by
  sorry

end floor_ceil_difference_l184_184675


namespace kylie_daisies_l184_184751

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end kylie_daisies_l184_184751


namespace expression_divisible_by_264_l184_184217

theorem expression_divisible_by_264 (n : ℕ) (h : n > 1) : ∃ k : ℤ, 7^(2*n) - 4^(2*n) - 297 = 264 * k :=
by 
  sorry

end expression_divisible_by_264_l184_184217


namespace range_of_m_l184_184170

theorem range_of_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ 2 * x + m - 3 = 0) : m < 3 :=
sorry

end range_of_m_l184_184170


namespace equivalence_of_equation_and_conditions_l184_184254

open Real
open Set

-- Definitions for conditions
def condition1 (t : ℝ) : Prop := cos t ≠ 0
def condition2 (t : ℝ) : Prop := sin t ≠ 0
def condition3 (t : ℝ) : Prop := cos (2 * t) ≠ 0

-- The main statement to be proved
theorem equivalence_of_equation_and_conditions (t : ℝ) :
  ((sin t / cos t - cos t / sin t + 2 * (sin (2 * t) / cos (2 * t))) * (1 + cos (3 * t))) = 4 * sin (3 * t) ↔
  ((∃ k l : ℤ, t = (π / 5) * (2 * k + 1) ∧ k ≠ 5 * l + 2) ∨ (∃ n l : ℤ, t = (π / 3) * (2 * n + 1) ∧ n ≠ 3 * l + 1))
    ∧ condition1 t
    ∧ condition2 t
    ∧ condition3 t :=
by
  sorry

end equivalence_of_equation_and_conditions_l184_184254


namespace interest_calculated_years_l184_184398

variable (P T : ℝ)

-- Given conditions
def principal_sum_positive : Prop := P > 0
def simple_interest_condition : Prop := (P * 5 * T) / 100 = P / 5

-- Theorem statement
theorem interest_calculated_years (h1 : principal_sum_positive P) (h2 : simple_interest_condition P T) : T = 4 :=
  sorry

end interest_calculated_years_l184_184398


namespace prove_cuboid_properties_l184_184268

noncomputable def cuboid_length := 5
noncomputable def cuboid_width := 4
noncomputable def cuboid_height := 3

theorem prove_cuboid_properties :
  (min (cuboid_length * cuboid_width) (min (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 12) ∧
  (max (cuboid_length * cuboid_width) (max (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 20) ∧
  ((cuboid_length + cuboid_width + cuboid_height) * 4 = 48) ∧
  (2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 94) ∧
  (cuboid_length * cuboid_width * cuboid_height = 60) :=
by
  sorry

end prove_cuboid_properties_l184_184268


namespace ants_meeting_points_l184_184408

/-- Definition for the problem setup: two ants running at constant speeds around a circle. -/
structure AntsRunningCircle where
  laps_ant1 : ℕ
  laps_ant2 : ℕ

/-- Theorem stating that given the laps completed by two ants in opposite directions on a circle, 
    they will meet at a specific number of distinct points. -/
theorem ants_meeting_points 
  (ants : AntsRunningCircle)
  (h1 : ants.laps_ant1 = 9)
  (h2 : ants.laps_ant2 = 6) : 
    ∃ n : ℕ, n = 5 := 
by
  -- Proof goes here
  sorry

end ants_meeting_points_l184_184408


namespace polyhedron_value_l184_184976

theorem polyhedron_value (T H V E : ℕ) (h t : ℕ) 
  (F : ℕ) (h_eq : h = 10) (t_eq : t = 10)
  (F_eq : F = 20)
  (edges_eq : E = (3 * t + 6 * h) / 2)
  (vertices_eq : V = E - F + 2)
  (T_value : T = 2) (H_value : H = 2) :
  100 * H + 10 * T + V = 227 := by
  sorry

end polyhedron_value_l184_184976


namespace true_discount_l184_184386

theorem true_discount (BD SD : ℝ) (hBD : BD = 18) (hSD : SD = 90) : 
  ∃ TD : ℝ, TD = 15 ∧ TD = BD / (1 + BD / SD) :=
by
  have h1 : 1 + BD / SD = 1 + 18 / 90 := by rw [hBD, hSD]
  have h2 : 1 + 18 / 90 = 1 + 0.2 := by norm_num
  have h3 : 1 + 0.2 = 1.2 := by norm_num
  have h4 : BD / (1 + BD / SD) = 18 / 1.2 := by rw [h1, h3]
  have h5 : 18 / 1.2 = 15 := by norm_num
  use 15
  split
  · exact h5
  · exact h4

end true_discount_l184_184386


namespace smallest_prime_factor_of_difference_l184_184388

theorem smallest_prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 1 ≤ C ∧ C ≤ 9) (h_diff : A ≠ C) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 3 :=
by
  sorry

end smallest_prime_factor_of_difference_l184_184388


namespace rationalize_denominator_l184_184375

-- Problem statement
theorem rationalize_denominator :
  1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end rationalize_denominator_l184_184375


namespace stickers_after_birthday_l184_184350

-- Definitions based on conditions
def initial_stickers : Nat := 39
def birthday_stickers : Nat := 22

-- Theorem stating the problem we aim to prove
theorem stickers_after_birthday : initial_stickers + birthday_stickers = 61 :=
by 
  sorry

end stickers_after_birthday_l184_184350


namespace lollipop_problem_l184_184881

def Henry_lollipops (A : Nat) : Nat := A + 30
def Diane_lollipops (A : Nat) : Nat := 2 * A
def Total_days (H A D : Nat) (daily_rate : Nat) : Nat := (H + A + D) / daily_rate

theorem lollipop_problem
  (A : Nat) (H : Nat) (D : Nat) (daily_rate : Nat)
  (h₁ : A = 60)
  (h₂ : H = Henry_lollipops A)
  (h₃ : D = Diane_lollipops A)
  (h₄ : daily_rate = 45)
  : Total_days H A D daily_rate = 6 := by
  sorry

end lollipop_problem_l184_184881


namespace average_salary_of_feb_mar_apr_may_l184_184502

theorem average_salary_of_feb_mar_apr_may
  (avg_salary_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr_may: ℝ)
  (n_months: ℝ): 
  avg_salary_jan_feb_mar_apr = 8000 ∧ 
  salary_jan = 6100 ∧ 
  salary_may = 6500 ∧ 
  total_salary_feb_mar_apr = (avg_salary_jan_feb_mar_apr * 4 - salary_jan) ∧
  total_salary_feb_mar_apr_may = (total_salary_feb_mar_apr + salary_may) ∧
  n_months = (total_salary_feb_mar_apr_may / 8100) →
  n_months = 4 :=
by
  intros 
  sorry

end average_salary_of_feb_mar_apr_may_l184_184502


namespace decreasing_interval_of_f_l184_184295

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x)

theorem decreasing_interval_of_f :
  ∀ x y : ℝ, (1 < x ∧ x < y) → f y < f x :=
by
  sorry

end decreasing_interval_of_f_l184_184295


namespace exists_unique_line_prime_x_intercept_positive_y_intercept_l184_184193

/-- There is exactly one line with x-intercept that is a prime number less than 10 and y-intercept that is a positive integer not equal to 5, which passes through the point (5, 4) -/
theorem exists_unique_line_prime_x_intercept_positive_y_intercept (x_intercept : ℕ) (hx : Nat.Prime x_intercept) (hx_lt_10 : x_intercept < 10) (y_intercept : ℕ) (hy_pos : y_intercept > 0) (hy_ne_5 : y_intercept ≠ 5) :
  (∃ (a b : ℕ), a = x_intercept ∧ b = y_intercept ∧ (∀ p q : ℕ, p = 5 ∧ q = 4 → (p / a) + (q / b) = 1)) :=
sorry

end exists_unique_line_prime_x_intercept_positive_y_intercept_l184_184193


namespace lattice_point_distance_l184_184655

theorem lattice_point_distance (d : ℝ) : 
  (∃ (r : ℝ), r = 2020 ∧ (∀ (A B C D : ℝ), 
  A = 0 ∧ B = 4040 ∧ C = 2020 ∧ D = 4040) 
  ∧ (∃ (P Q : ℝ), P = 0.25 ∧ Q = 1)) → 
  d = 0.3 := 
by
  sorry

end lattice_point_distance_l184_184655


namespace arithmetic_sequence_sum_l184_184578

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)
variable (d : α)

-- Condition definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = d

def sum_condition (a : ℕ → α) : Prop :=
  a 2 + a 5 + a 8 = 39

-- The goal statement to prove
theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_sum : sum_condition a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
  sorry

end arithmetic_sequence_sum_l184_184578


namespace condition_of_inequality_l184_184577

theorem condition_of_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2 * (x + y - 1)) : x = 1 ∧ y = 1 :=
by
  sorry

end condition_of_inequality_l184_184577


namespace sufficient_but_not_necessary_condition_l184_184201

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 → ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) ∧
  ¬ (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 ↔ ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) := 
sorry

end sufficient_but_not_necessary_condition_l184_184201


namespace graph_passes_through_quadrants_l184_184082

theorem graph_passes_through_quadrants :
  ∀ x : ℝ, (4 * x + 2 > 0 → (x > 0)) ∨ (4 * x + 2 > 0 → (x < 0)) ∨ (4 * x + 2 < 0 → (x < 0)) :=
by
  intro x
  sorry

end graph_passes_through_quadrants_l184_184082


namespace age_ratio_in_1_year_l184_184244

variable (j m x : ℕ)

-- Conditions
def condition1 (j m : ℕ) : Prop :=
  j - 3 = 2 * (m - 3)

def condition2 (j m : ℕ) : Prop :=
  j - 5 = 3 * (m - 5)

-- Question
def age_ratio (j m x : ℕ) : Prop :=
  (j + x) * 2 = 3 * (m + x)

theorem age_ratio_in_1_year (j m x : ℕ) :
  condition1 j m → condition2 j m → age_ratio j m 1 :=
by
  sorry

end age_ratio_in_1_year_l184_184244


namespace businessmen_drink_none_l184_184436

open Finset

-- Definitions based on conditions
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def soda_drinkers : ℕ := 8
def coffee_and_tea_drinkers : ℕ := 6
def coffee_and_soda_drinkers : ℕ := 2
def tea_and_soda_drinkers : ℕ := 3
def all_three_drinkers : ℕ := 1

-- Theorem stating the number of businessmen who drank none of the beverages
theorem businessmen_drink_none :
  total_businessmen - (coffee_drinkers + tea_drinkers + soda_drinkers 
  - coffee_and_tea_drinkers - coffee_and_soda_drinkers - tea_and_soda_drinkers 
  + all_three_drinkers) = 4 :=
  sorry

end businessmen_drink_none_l184_184436


namespace price_per_acre_is_1863_l184_184819

-- Define the conditions
def totalAcres : ℕ := 4
def numLots : ℕ := 9
def pricePerLot : ℤ := 828
def totalRevenue : ℤ := numLots * pricePerLot
def totalCost (P : ℤ) : ℤ := totalAcres * P

-- The proof problem: Prove that the price per acre P is 1863
theorem price_per_acre_is_1863 (P : ℤ) (h : totalCost P = totalRevenue) : P = 1863 :=
by
  sorry

end price_per_acre_is_1863_l184_184819


namespace fraction_walk_is_three_twentieths_l184_184557

-- Define the various fractions given in the conditions
def fraction_bus : ℚ := 1 / 2
def fraction_auto : ℚ := 1 / 4
def fraction_bicycle : ℚ := 1 / 10

-- Defining the total fraction for students that do not walk
def total_not_walk : ℚ := fraction_bus + fraction_auto + fraction_bicycle

-- The remaining fraction after subtracting from 1
def fraction_walk : ℚ := 1 - total_not_walk

-- The theorem we want to prove that fraction_walk is 3/20
theorem fraction_walk_is_three_twentieths : fraction_walk = 3 / 20 := by
  sorry

end fraction_walk_is_three_twentieths_l184_184557


namespace solve_inequality_l184_184236

theorem solve_inequality :
  {x : ℝ | -x^2 + 5 * x > 6} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solve_inequality_l184_184236


namespace smallest_solution_eq_l184_184147

theorem smallest_solution_eq (x : ℝ) (h : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) :
  x = 4 - Real.sqrt 2 := 
  sorry

end smallest_solution_eq_l184_184147


namespace quadratic_inequality_solution_l184_184635

variable (a x : ℝ)

-- Define the quadratic expression and the inequality condition
def quadratic_inequality (a x : ℝ) : Prop := 
  x^2 - (2 * a + 1) * x + a^2 + a < 0

-- Define the interval in which the inequality holds
def solution_set (a x : ℝ) : Prop :=
  a < x ∧ x < a + 1

-- The main statement to be proven
theorem quadratic_inequality_solution :
  ∀ a x, quadratic_inequality a x ↔ solution_set a x :=
sorry

end quadratic_inequality_solution_l184_184635


namespace departure_sequences_count_l184_184979

noncomputable def total_departure_sequences (trains: Finset ℕ) (A B : ℕ) 
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6) 
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : ℕ := 6 * 6 * 6

-- The main theorem statement: given the conditions, prove the total number of different sequences is 216
theorem departure_sequences_count (trains: Finset ℕ) (A B : ℕ)
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6)
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : total_departure_sequences trains A B h hAB = 216 := 
by 
  sorry

end departure_sequences_count_l184_184979


namespace probability_of_consonant_initials_l184_184734

def number_of_students : Nat := 30
def alphabet_size : Nat := 26
def redefined_vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
def number_of_vowels : Nat := redefined_vowels.card
def number_of_consonants : Nat := alphabet_size - number_of_vowels

theorem probability_of_consonant_initials :
  (number_of_consonants : ℝ) / (number_of_students : ℝ) = 2/3 := 
by
  -- Proof goes here
  sorry

end probability_of_consonant_initials_l184_184734


namespace newton_method_approximation_bisection_method_approximation_l184_184617

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 3

theorem newton_method_approximation :
  let x0 := -1
  let x1 := x0 - f x0 / f' x0
  let x2 := x1 - f x1 / f' x1
  x2 = -7 / 5 := sorry

theorem bisection_method_approximation :
  let a := -2
  let b := -1
  let midpoint1 := (a + b) / 2
  let new_a := if f midpoint1 < 0 then midpoint1 else a
  let new_b := if f midpoint1 < 0 then b else midpoint1
  let midpoint2 := (new_a + new_b) / 2
  midpoint2 = -11 / 8 := sorry

end newton_method_approximation_bisection_method_approximation_l184_184617


namespace perpendicular_lines_m_value_l184_184900

def is_perpendicular (m : ℝ) : Prop :=
    let slope1 := 1 / 2
    let slope2 := -2 / m
    slope1 * slope2 = -1

theorem perpendicular_lines_m_value (m : ℝ) (h : is_perpendicular m) : m = 1 := by
    sorry

end perpendicular_lines_m_value_l184_184900


namespace expenditure_recording_l184_184183

theorem expenditure_recording (income expense : ℤ) (h1 : income = 100) (h2 : expense = -100)
  (h3 : income = -expense) : expense = -100 :=
by
  sorry

end expenditure_recording_l184_184183


namespace sum_of_inserted_numbers_in_progressions_l184_184243

theorem sum_of_inserted_numbers_in_progressions (x y : ℝ) (hx : 4 * (y / x) = x) (hy : 2 * y = x + 64) :
  x + y = 131 + 3 * Real.sqrt 129 :=
by
  sorry

end sum_of_inserted_numbers_in_progressions_l184_184243


namespace reasoning_is_invalid_l184_184878

-- Definitions based on conditions
variables {Line Plane : Type} (is_parallel_to : Line → Plane → Prop) (is_parallel_to' : Line → Line → Prop) (is_contained_in : Line → Plane → Prop)

-- Conditions
axiom major_premise (b : Line) (α : Plane) : is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a
axiom minor_premise1 (b : Line) (α : Plane) : is_parallel_to b α
axiom minor_premise2 (a : Line) (α : Plane) : is_contained_in a α

-- Conclusion
theorem reasoning_is_invalid : ∃ (a : Line) (b : Line) (α : Plane), ¬ (is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a) :=
sorry

end reasoning_is_invalid_l184_184878


namespace f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l184_184871

noncomputable def f (x : ℝ) : ℝ := if x > 0 then (Real.log (1 + x)) / x else 0

theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
sorry

theorem f_greater_than_2_div_x_plus_2 :
  ∀ x : ℝ, 0 < x → f x > 2 / (x + 2) :=
sorry

end f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l184_184871


namespace probability_gcd_one_l184_184957

-- Defining the domain of our problem: the set {1, 2, 3, ..., 8}
def S := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the selection of two distinct natural numbers from S
def select_two_distinct_from_S (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y

-- Defining the greatest common factor condition
def is_rel_prime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

-- Defining the probability computation (relatively prime pairs over total pairs)
def probability_rel_prime : ℚ :=
  (21 : ℚ) / 28  -- since 21 pairs are relatively prime out of 28 total pairs

-- The main theorem statement
theorem probability_gcd_one :
  probability_rel_prime = 3 / 4 :=
sorry

end probability_gcd_one_l184_184957


namespace candy_difference_l184_184457

theorem candy_difference (Frankie_candies Max_candies : ℕ) (hF : Frankie_candies = 74) (hM : Max_candies = 92) :
  Max_candies - Frankie_candies = 18 :=
by
  sorry

end candy_difference_l184_184457


namespace trail_length_l184_184558

variables (a b c d e : ℕ)

theorem trail_length : 
  a + b + c = 45 ∧
  b + d = 36 ∧
  c + d + e = 60 ∧
  a + d = 32 → 
  a + b + c + d + e = 69 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end trail_length_l184_184558


namespace raft_min_capacity_l184_184993

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l184_184993


namespace profit_function_profit_for_240_barrels_barrels_for_760_profit_l184_184630

-- Define fixed costs, cost price per barrel, and selling price per barrel as constants
def fixed_costs : ℝ := 200
def cost_price_per_barrel : ℝ := 5
def selling_price_per_barrel : ℝ := 8

-- Definitions for daily sales quantity (x) and daily profit (y)
def daily_sales_quantity (x : ℝ) : ℝ := x
def daily_profit (x : ℝ) : ℝ := (selling_price_per_barrel * x) - (cost_price_per_barrel * x) - fixed_costs

-- Prove the functional relationship y = 3x - 200
theorem profit_function (x : ℝ) : daily_profit x = 3 * x - fixed_costs :=
by sorry

-- Given sales quantity is 240 barrels, prove profit is 520 yuan
theorem profit_for_240_barrels : daily_profit 240 = 520 :=
by sorry

-- Given profit is 760 yuan, prove sales quantity is 320 barrels
theorem barrels_for_760_profit : ∃ (x : ℝ), daily_profit x = 760 ∧ x = 320 :=
by sorry

end profit_function_profit_for_240_barrels_barrels_for_760_profit_l184_184630


namespace Carly_fourth_week_running_distance_l184_184437

theorem Carly_fourth_week_running_distance :
  let week1_distance_per_day := 2
  let week2_distance_per_day := (week1_distance_per_day * 2) + 3
  let week3_distance_per_day := week2_distance_per_day * (9 / 7)
  let week4_intended_distance_per_day := week3_distance_per_day * 0.9
  let week4_actual_distance_per_day := week4_intended_distance_per_day * 0.5
  let week4_days_run := 5 -- due to 2 rest days
  (week4_actual_distance_per_day * week4_days_run) = 20.25 := 
by 
    -- We use sorry here to skip the proof
    sorry

end Carly_fourth_week_running_distance_l184_184437


namespace time_for_train_to_pass_jogger_l184_184978

noncomputable def time_to_pass (s_jogger s_train : ℝ) (d_headstart l_train : ℝ) : ℝ :=
  let speed_jogger := s_jogger * (1000 / 3600)
  let speed_train := s_train * (1000 / 3600)
  let relative_speed := speed_train - speed_jogger
  let total_distance := d_headstart + l_train
  total_distance / relative_speed

theorem time_for_train_to_pass_jogger :
  time_to_pass 12 60 360 180 = 40.48 :=
by
  sorry

end time_for_train_to_pass_jogger_l184_184978


namespace A_time_240m_race_l184_184811

theorem A_time_240m_race (t : ℕ) :
  (∀ t, (240 / t) = (184 / t) * (t + 7) ∧ 240 = 184 + ((184 * 7) / t)) → t = 23 :=
by
  sorry

end A_time_240m_race_l184_184811


namespace gcd_xyz_times_xyz_is_square_l184_184361

theorem gcd_xyz_times_xyz_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by
  sorry

end gcd_xyz_times_xyz_is_square_l184_184361


namespace height_large_cylinder_is_10_l184_184931

noncomputable def height_large_cylinder : ℝ :=
  let V_small := 13.5 * Real.pi
  let factor := 74.07407407407408
  let V_large := 100 * Real.pi
  factor * V_small / V_large

theorem height_large_cylinder_is_10 :
  height_large_cylinder = 10 :=
by
  sorry

end height_large_cylinder_is_10_l184_184931


namespace matrix_multiplication_is_correct_l184_184438

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![0, 2]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![15, -7], ![20, -16]]

theorem matrix_multiplication_is_correct : A ⬝ B = C :=
by
  sorry

end matrix_multiplication_is_correct_l184_184438


namespace maximum_pencils_l184_184981

-- Define the problem conditions
def red_pencil_cost := 27
def blue_pencil_cost := 23
def max_total_cost := 940
def max_diff := 10

-- Define the main theorem
theorem maximum_pencils (x y : ℕ) 
  (h1 : red_pencil_cost * x + blue_pencil_cost * y ≤ max_total_cost)
  (h2 : y - x ≤ max_diff)
  (hx_min : ∀ z : ℕ, z < x → red_pencil_cost * z + blue_pencil_cost * (z + max_diff) > max_total_cost):
  x = 14 ∧ y = 24 ∧ x + y = 38 := 
  sorry

end maximum_pencils_l184_184981


namespace alice_net_amount_spent_l184_184767

noncomputable def net_amount_spent : ℝ :=
  let price_per_pint := 4
  let sunday_pints := 4
  let sunday_cost := sunday_pints * price_per_pint

  let monday_discount := 0.1
  let monday_pints := 3 * sunday_pints
  let monday_price_per_pint := price_per_pint * (1 - monday_discount)
  let monday_cost := monday_pints * monday_price_per_pint

  let tuesday_discount := 0.2
  let tuesday_pints := monday_pints / 3
  let tuesday_price_per_pint := price_per_pint * (1 - tuesday_discount)
  let tuesday_cost := tuesday_pints * tuesday_price_per_pint

  let wednesday_returned_pints := tuesday_pints / 2
  let wednesday_refund := wednesday_returned_pints * tuesday_price_per_pint

  sunday_cost + monday_cost + tuesday_cost - wednesday_refund

theorem alice_net_amount_spent : net_amount_spent = 65.60 := by
  sorry

end alice_net_amount_spent_l184_184767


namespace range_of_p_l184_184875

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end range_of_p_l184_184875


namespace five_op_two_l184_184490

-- Definition of the operation
def op (a b : ℝ) := 3 * a + 4 * b

-- The theorem statement
theorem five_op_two : op 5 2 = 23 := by
  sorry

end five_op_two_l184_184490


namespace min_value_of_n_l184_184596

theorem min_value_of_n (n : ℕ) (k : ℚ) (h1 : k > 0.9999) 
    (h2 : 4 * n * (n - 1) * (1 - k) = 1) : 
    n = 51 :=
sorry

end min_value_of_n_l184_184596


namespace diameter_of_triple_sphere_l184_184079

noncomputable def radius_of_sphere : ℝ := 6

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

noncomputable def triple_volume_of_sphere (r : ℝ) : ℝ := 3 * volume_of_sphere r

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem diameter_of_triple_sphere (r : ℝ) (V1 V2 : ℝ) (a b : ℝ) 
  (h_r : r = radius_of_sphere)
  (h_V1 : V1 = volume_of_sphere r)
  (h_V2 : V2 = triple_volume_of_sphere r)
  (h_d : 12 * cube_root 3 = 2 * (6 * cube_root 3))
  : a + b = 15 :=
sorry

end diameter_of_triple_sphere_l184_184079


namespace bonnets_per_orphanage_correct_l184_184612

-- Definitions for each day's bonnet count
def monday_bonnets := 10
def tuesday_and_wednesday_bonnets := 2 * monday_bonnets
def thursday_bonnets := monday_bonnets + 5
def friday_bonnets := thursday_bonnets - 5
def saturday_bonnets := friday_bonnets - 8
def sunday_bonnets := 3 * saturday_bonnets

-- Total bonnets made in the week
def total_bonnets := 
  monday_bonnets +
  tuesday_and_wednesday_bonnets +
  thursday_bonnets +
  friday_bonnets +
  saturday_bonnets +
  sunday_bonnets

-- The number of orphanages
def orphanages := 10

-- Bonnets sent to each orphanage
def bonnets_per_orphanage := total_bonnets / orphanages

theorem bonnets_per_orphanage_correct :
  bonnets_per_orphanage = 6 :=
by
  sorry

end bonnets_per_orphanage_correct_l184_184612


namespace max_distance_with_optimal_tire_swapping_l184_184855

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end max_distance_with_optimal_tire_swapping_l184_184855


namespace triangle_area_l184_184784

theorem triangle_area : 
  ∀ (x y: ℝ), (x / 5 + y / 2 = 1) → (x = 5) ∨ (y = 2) → ∃ A : ℝ, A = 5 :=
by
  intros x y h1 h2
  -- Definitions based on the problem conditions
  have hx : x = 5 := sorry
  have hy : y = 2 := sorry
  have base := 5
  have height := 2
  have area := 1 / 2 * base * height
  use area
  sorry

end triangle_area_l184_184784


namespace initial_money_l184_184765

-- Let M represent the initial amount of money Mrs. Hilt had.
variable (M : ℕ)

-- Condition 1: Mrs. Hilt bought a pencil for 11 cents.
def pencil_cost : ℕ := 11

-- Condition 2: She had 4 cents left after buying the pencil.
def amount_left : ℕ := 4

-- Proof problem statement: Prove that M = 15 given the above conditions.
theorem initial_money (h : M = pencil_cost + amount_left) : M = 15 :=
by
  sorry

end initial_money_l184_184765


namespace max_value_l184_184047

theorem max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m ≤ 4, ∀ (z w : ℝ), z > 0 → w > 0 → (x + y = z + w) → (z^3 + w^3 ≥ x^3 + y^3 → 
  (z + w)^3 / (z^3 + w^3) ≤ m) :=
sorry

end max_value_l184_184047


namespace sequence_formula_l184_184716

open Nat

noncomputable def S : ℕ → ℤ
| n => n^2 - 2 * n + 2

noncomputable def a : ℕ → ℤ
| 0 => 1  -- note that in Lean, sequence indexing starts from 0
| (n+1) => 2*(n+1) - 3

theorem sequence_formula (n : ℕ) : 
  a n = if n = 0 then 1 else 2*n - 3 := by
  sorry

end sequence_formula_l184_184716


namespace container_volume_ratio_l184_184673

theorem container_volume_ratio (A B : ℕ) 
  (h1 : (3 / 4 : ℚ) * A = (5 / 8 : ℚ) * B) :
  (A : ℚ) / B = 5 / 6 :=
by
  admit
-- sorry

end container_volume_ratio_l184_184673


namespace students_like_basketball_or_cricket_or_both_l184_184189

theorem students_like_basketball_or_cricket_or_both :
  let basketball_lovers := 9
  let cricket_lovers := 8
  let both_lovers := 6
  basketball_lovers + cricket_lovers - both_lovers = 11 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l184_184189


namespace Tod_drove_time_l184_184951

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end Tod_drove_time_l184_184951


namespace intersection_point_l184_184800

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - 5
def g (x : ℝ) : ℝ := 2 * x^2 + 11

theorem intersection_point :
  ∃ x y : ℝ, f x = y ∧ g x = y ∧ x = 2 ∧ y = 19 := by
  sorry

end intersection_point_l184_184800


namespace find_a_l184_184476

noncomputable def unique_quad_solution (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1^2 - a * x1 + a = 1 → x2^2 - a * x2 + a = 1 → x1 = x2

theorem find_a (a : ℝ) (h : unique_quad_solution a) : a = 2 :=
sorry

end find_a_l184_184476


namespace integer_part_sqrt_sum_l184_184161

noncomputable def a := 4
noncomputable def b := 3
noncomputable def c := -2

theorem integer_part_sqrt_sum (h1 : |a| = 4) (h2 : b*b = 9) (h3 : c*c*c = -8) (h4 : a > b) (h5 : b > c) :
  int.sqrt (a + b + c) = 2 :=
by
  have h : a + b + c = 5 := sorry
  have h_sqrt : sqrt 5 = 2 := sorry
  exact h_sqrt

end integer_part_sqrt_sum_l184_184161


namespace absent_children_count_l184_184369

-- Definition of conditions
def total_children := 700
def bananas_per_child := 2
def bananas_extra := 2
def total_bananas := total_children * bananas_per_child

-- The proof goal
theorem absent_children_count (A P : ℕ) (h_P : P = total_children - A)
    (h_bananas : total_bananas = P * (bananas_per_child + bananas_extra)) : A = 350 :=
by
  -- Since this is a statement only, we place a sorry here to skip the proof.
  sorry

end absent_children_count_l184_184369


namespace fixed_point_exists_l184_184306

theorem fixed_point_exists (m : ℝ) :
  ∀ (x y : ℝ), (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 → x = 3 ∧ y = 1 :=
by
  sorry

end fixed_point_exists_l184_184306


namespace solve_problem1_solve_problem2_l184_184108

noncomputable def problem1 (α : ℝ) : Prop :=
  (Real.tan (Real.pi / 4 + α) = 1 / 2) →
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -2

noncomputable def problem2 : Prop :=
  Real.sin (Real.pi / 12) * Real.sin (5 * Real.pi / 12) = 1 / 4

-- theorems to be proved
theorem solve_problem1 (α : ℝ) : problem1 α := by
  sorry

theorem solve_problem2 : problem2 := by
  sorry

end solve_problem1_solve_problem2_l184_184108


namespace determine_a_l184_184010

-- Define the function f as given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 6

-- Formulate the proof statement
theorem determine_a (a : ℝ) (h : f a (-1) = 8) : a = -2 :=
by {
  sorry
}

end determine_a_l184_184010


namespace prob_A_eq_prob_B_l184_184188

-- Define the number of students and the number of tickets
def num_students : ℕ := 56
def num_tickets : ℕ := 56
def prize_tickets : ℕ := 1

-- Define the probability of winning the prize for a given student (A for first student, B for last student)
def prob_A := prize_tickets / num_tickets
def prob_B := prize_tickets / num_tickets

-- Statement to prove
theorem prob_A_eq_prob_B : prob_A = prob_B :=
by 
  -- We provide the statement to prove without the proof steps
  sorry

end prob_A_eq_prob_B_l184_184188


namespace alyssa_ate_limes_l184_184919

def mikes_limes : ℝ := 32.0
def limes_left : ℝ := 7.0

theorem alyssa_ate_limes : mikes_limes - limes_left = 25.0 := by
  sorry

end alyssa_ate_limes_l184_184919


namespace smaller_number_is_7_l184_184022

theorem smaller_number_is_7 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) (h3 : x ≤ y) (h4 : x ∣ 28) : x = 7 :=
  sorry

end smaller_number_is_7_l184_184022


namespace sum_of_xs_l184_184917

theorem sum_of_xs (x y z : ℂ) : (x + y * z = 8) ∧ (y + x * z = 12) ∧ (z + x * y = 11) → 
    ∃ S, ∀ (xi yi zi : ℂ), (xi + yi * zi = 8) ∧ (yi + xi * zi = 12) ∧ (zi + xi * yi = 11) →
        xi + yi + zi = S :=
by
  sorry

end sum_of_xs_l184_184917


namespace min_capacity_for_raft_l184_184982

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l184_184982


namespace solve_for_x_l184_184222

theorem solve_for_x (x : ℝ) : (5 * x + 9 * x = 350 - 10 * (x - 5)) -> x = 50 / 3 :=
by
  intro h
  sorry

end solve_for_x_l184_184222


namespace experiment_success_probability_l184_184797

/-- 
There are three boxes, each containing 10 balls. 
- The first box contains 7 balls marked 'A' and 3 balls marked 'B'.
- The second box contains 5 red balls and 5 white balls.
- The third box contains 8 red balls and 2 white balls.

The experiment consists of:
1. Drawing a ball from the first box.
2. If a ball marked 'A' is drawn, drawing from the second box.
3. If a ball marked 'B' is drawn, drawing from the third box.
The experiment is successful if the second ball drawn is red.

Prove that the probability of the experiment being successful is 0.59.
-/
theorem experiment_success_probability (P : ℝ) : 
  P = 0.59 :=
sorry

end experiment_success_probability_l184_184797


namespace greatest_possible_value_l184_184006

theorem greatest_possible_value :
  ∃ (u v : ℝ) (a b : ℝ),
  (u + v = u^2 + v^2) ∧ 
  (u + v = u^4 + v^4) ∧ 
  (u + v = u^6 + v^6) ∧ 
  (u + v = u^8 + v^8) ∧ 
  (u + v = u^{10} + v^{10}) ∧ 
  (u + v = u^{12} + v^{12}) ∧ 
  (u + v = u^{14} + v^{14}) ∧ 
  (u + v = u^{16} + v^{16}) ∧ 
  (u + v = u^{18} + v^{18}) ∧ 
  (is_root (X^2 - C a * X + C b) u) ∧ 
  (is_root (X^2 - C a * X + C b) v) ∧ 
  ( ∀ x y : ℝ, (x ≠ 1 ∨ y ≠ 1) ∨  ( x^2 - x + 1 = 0) ∧ (y^2 - y + 1 = 0) ∨  ( x =y) ) → 
  greatest_possible_value u v = 2 :=
  sorry

end greatest_possible_value_l184_184006


namespace Carla_final_position_l184_184287

-- Carla's initial position
def Carla_initial_position : ℤ × ℤ := (10, -10)

-- Function to calculate Carla's new position after each move
def Carla_move (pos : ℤ × ℤ) (direction : ℕ) (distance : ℤ) : ℤ × ℤ :=
  match direction % 4 with
  | 0 => (pos.1, pos.2 + distance)   -- North
  | 1 => (pos.1 + distance, pos.2)   -- East
  | 2 => (pos.1, pos.2 - distance)   -- South
  | 3 => (pos.1 - distance, pos.2)   -- West
  | _ => pos  -- This case will never happen due to the modulo operation

-- Recursive function to simulate Carla's journey
def Carla_journey : ℕ → ℤ × ℤ → ℤ × ℤ 
  | 0, pos => pos
  | n + 1, pos => 
    let next_pos := Carla_move pos n (2 + n / 2 * 2)
    Carla_journey n next_pos

-- Prove that after 100 moves, Carla's position is (-191, -10)
theorem Carla_final_position : Carla_journey 100 Carla_initial_position = (-191, -10) :=
sorry

end Carla_final_position_l184_184287


namespace pond_field_ratio_l184_184391

theorem pond_field_ratio (L W : ℕ) (pond_side : ℕ) (hL : L = 24) (hLW : L = 2 * W) (hPond : pond_side = 6) :
  pond_side * pond_side / (L * W) = 1 / 8 :=
by
  sorry

end pond_field_ratio_l184_184391


namespace f_at_zero_f_positive_f_increasing_l184_184427

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : true
axiom f_nonzero : f 0 ≠ 0
axiom f_pos_gt1 (x : ℝ) : x > 0 → f x > 1
axiom f_add (a b : ℝ) : f (a + b) = f a * f b

theorem f_at_zero : f 0 = 1 :=
sorry

theorem f_positive (x : ℝ) : f x > 0 :=
sorry

theorem f_increasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_at_zero_f_positive_f_increasing_l184_184427


namespace km_to_leaps_l184_184071

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end km_to_leaps_l184_184071


namespace sum_of_series_l184_184107

theorem sum_of_series : 
  (6 + 16 + 26 + 36 + 46) + (14 + 24 + 34 + 44 + 54) = 300 :=
by
  sorry

end sum_of_series_l184_184107


namespace neg_number_among_set_l184_184829

theorem neg_number_among_set :
  ∃ n ∈ ({5, 1, -2, 0} : Set ℤ), n < 0 ∧ n = -2 :=
by
  sorry

end neg_number_among_set_l184_184829


namespace r_investment_time_l184_184089

variables (P Q R Profit_p Profit_q Profit_r Tp Tq Tr : ℕ)
variables (h1 : P / Q = 7 / 5)
variables (h2 : Q / R = 5 / 4)
variables (h3 : Profit_p / Profit_q = 7 / 10)
variables (h4 : Profit_p / Profit_r = 7 / 8)
variables (h5 : Tp = 2)
variables (h6 : Tq = t)

theorem r_investment_time (t : ℕ) :
  ∃ Tr : ℕ, Tr = 4 :=
sorry

end r_investment_time_l184_184089


namespace boat_downstream_distance_l184_184539

variable (speed_still_water : ℤ) (speed_stream : ℤ) (time_downstream : ℤ)

theorem boat_downstream_distance
    (h₁ : speed_still_water = 24)
    (h₂ : speed_stream = 4)
    (h₃ : time_downstream = 4) :
    (speed_still_water + speed_stream) * time_downstream = 112 := by
  sorry

end boat_downstream_distance_l184_184539


namespace find_common_ratio_l184_184185

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variable {a : ℕ → ℝ} {q : ℝ}

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : a 2 + a 4 = 20)
  (h3 : a 3 + a 5 = 40) : q = 2 :=
by
  sorry

end find_common_ratio_l184_184185


namespace marys_garbage_bill_l184_184763

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end marys_garbage_bill_l184_184763


namespace polynomial_roots_correct_l184_184691

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l184_184691


namespace smallest_three_digit_number_with_property_l184_184452

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a ≤ 999 ∧ ∃ n : ℕ, 1001 * a + 1 = n^2 ∧ ∀ b : ℕ, 100 ≤ b ∧ b ≤ 999 ∧ (∃ m : ℕ, 1001 * b + 1 = m^2) → a ≤ b :=
begin
  sorry,
end

end smallest_three_digit_number_with_property_l184_184452


namespace solve_for_x_l184_184237

theorem solve_for_x : (∃ x : ℝ, 5 * x + 4 = -6) → x = -2 := 
by
  sorry

end solve_for_x_l184_184237


namespace Juanico_age_30_years_from_now_l184_184199

-- Definitions and hypothesis
def currentAgeGladys : ℕ := 30 -- Gladys's current age, since she will be 40 in 10 years
def currentAgeJuanico : ℕ := (1 / 2) * currentAgeGladys - 4 -- Juanico's current age based on Gladys's current age

theorem Juanico_age_30_years_from_now :
  currentAgeJuanico + 30 = 41 :=
by
  -- You would normally fill out the proof here, but we use 'sorry' to skip it.
  sorry

end Juanico_age_30_years_from_now_l184_184199


namespace inequality_division_l184_184472

variable (m n : ℝ)

theorem inequality_division (h : m > n) : (m / 4) > (n / 4) :=
sorry

end inequality_division_l184_184472


namespace measure_of_angle_A_range_of_b2_add_c2_div_a2_l184_184037

variable {A B C a b c : ℝ}
variable {S : ℝ}

theorem measure_of_angle_A
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) : 
  A = 2 * Real.pi / 3 :=
by
  sorry

theorem range_of_b2_add_c2_div_a2
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : A = 2 * Real.pi / 3) : 
  2 / 3 ≤ (b ^ 2 + c ^ 2) / a ^ 2 ∧ (b ^ 2 + c ^ 2) / a ^ 2 < 1 :=
by
  sorry

end measure_of_angle_A_range_of_b2_add_c2_div_a2_l184_184037


namespace best_is_man_l184_184273

structure Competitor where
  name : String
  gender : String
  age : Int
  is_twin : Bool

noncomputable def participants : List Competitor := [
  ⟨"man", "male", 30, false⟩,
  ⟨"sister", "female", 30, true⟩,
  ⟨"son", "male", 30, true⟩,
  ⟨"niece", "female", 25, false⟩
]

def are_different_gender (c1 c2 : Competitor) : Bool := c1.gender ≠ c2.gender
def has_same_age (c1 c2 : Competitor) : Bool := c1.age = c2.age

noncomputable def best_competitor : Competitor :=
  let best_candidate := participants[0] -- assuming "man" is the best for example's sake
  let worst_candidate := participants[2] -- assuming "son" is the worst for example's sake
  best_candidate

theorem best_is_man : best_competitor.name = "man" :=
by
  have h1 : are_different_gender (participants[0]) (participants[2]) := by sorry
  have h2 : has_same_age (participants[0]) (participants[2]) := by sorry
  exact sorry

end best_is_man_l184_184273


namespace TruckloadsOfSand_l184_184823

theorem TruckloadsOfSand (S : ℝ) (totalMat dirt cement : ℝ) 
  (h1 : totalMat = 0.67) 
  (h2 : dirt = 0.33) 
  (h3 : cement = 0.17) 
  (h4 : totalMat = S + dirt + cement) : 
  S = 0.17 := 
  by 
    sorry

end TruckloadsOfSand_l184_184823


namespace isosceles_triangle_angles_l184_184928

theorem isosceles_triangle_angles 
  (α r R : ℝ)
  (isosceles : α ∈ {β : ℝ | β = α})
  (circumference_relation : R = 3 * r) :
  (α = Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3)) ∨ 
   α = Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) ∧ 
  (
    180 = 2 * (Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3))) + 2 * α ∨
    180 = 2 * (Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) + 2 * α 
  ) :=
by sorry

end isosceles_triangle_angles_l184_184928


namespace cost_price_of_cloths_l184_184117

-- Definitions based on conditions
def SP_A := 8500 / 85
def Profit_A := 15
def CP_A := SP_A - Profit_A

def SP_B := 10200 / 120
def Profit_B := 12
def CP_B := SP_B - Profit_B

def SP_C := 4200 / 60
def Profit_C := 10
def CP_C := SP_C - Profit_C

-- Theorem to prove the cost prices
theorem cost_price_of_cloths :
    CP_A = 85 ∧
    CP_B = 73 ∧
    CP_C = 60 :=
by
    sorry

end cost_price_of_cloths_l184_184117


namespace meaningful_expression_l184_184523

theorem meaningful_expression (x : ℝ) : (∃ y, y = 5 / (Real.sqrt (x + 1))) ↔ x > -1 :=
by
  sorry

end meaningful_expression_l184_184523


namespace calc_expression_l184_184560

theorem calc_expression : (2019 / 2018) - (2018 / 2019) = 4037 / 4036 := 
by sorry

end calc_expression_l184_184560


namespace S_rational_iff_divides_l184_184911

-- Definition of "divides" for positive integers
def divides (m k : ℕ) : Prop := ∃ j : ℕ, k = m * j

-- Definition of the series S(m, k)
noncomputable def S (m k : ℕ) : ℝ := 
  ∑' n, 1 / (n * (m * n + k))

-- Proof statement
theorem S_rational_iff_divides (m k : ℕ) (hm : 0 < m) (hk : 0 < k) : 
  (∃ r : ℚ, S m k = r) ↔ divides m k :=
sorry

end S_rational_iff_divides_l184_184911


namespace warehouse_capacity_l184_184821

theorem warehouse_capacity (total_bins num_20_ton_bins cap_20_ton_bin cap_15_ton_bin : Nat) 
  (h1 : total_bins = 30) 
  (h2 : num_20_ton_bins = 12) 
  (h3 : cap_20_ton_bin = 20) 
  (h4 : cap_15_ton_bin = 15) : 
  total_bins * cap_20_ton_bin + (total_bins - num_20_ton_bins) * cap_15_ton_bin = 510 := 
by
  sorry

end warehouse_capacity_l184_184821


namespace xiao_ming_winning_probability_type1_given_winning_probability_l184_184126

open ProbabilityTheory

-- Defining the events
def A1 : Event := sorry
def A2 : Event := sorry
def A3 : Event := sorry
def B : Event := sorry

-- Given probabilities
def P_A1 : ℝ := 0.5
def P_A2 : ℝ := 0.25
def P_A3 : ℝ := 0.25
def P_B_given_A1 : ℝ := 0.3
def P_B_given_A2 : ℝ := 0.4
def P_B_given_A3 : ℝ := 0.5

-- Proof problem statement
theorem xiao_ming_winning_probability :
  P(B) = P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 := sorry

theorem type1_given_winning_probability :
  P(A1 | B) = (P_A1 * P_B_given_A1) / (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3) := sorry

end xiao_ming_winning_probability_type1_given_winning_probability_l184_184126


namespace dara_employment_waiting_time_l184_184230

theorem dara_employment_waiting_time :
  ∀ (D : ℕ),
  (∀ (Min_Age_Required : ℕ) (Current_Jane_Age : ℕ),
    Min_Age_Required = 25 →
    Current_Jane_Age = 28 →
    (D + 6 = 1 / 2 * (Current_Jane_Age + 6))) →
  (25 - D = 14) :=
by intros D Min_Age_Required Current_Jane_Age h1 h2 h3
   -- We are given that Min_Age_Required = 25
   rw h1 at *
   -- We are given that Current_Jane_Age = 28
   rw h2 at *
   -- We know from the condition that D satisfies the equation
   have h4: D + 6 = 0.5 * (28 + 6), from h3
   sorry

end dara_employment_waiting_time_l184_184230


namespace find_x_l184_184781

theorem find_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) : 
  x = (36 * Real.sqrt 5)^(4/11) := 
sorry

end find_x_l184_184781


namespace find_n_l184_184430

theorem find_n (n a b : ℕ) 
  (h1 : a > 1)
  (h2 : a ∣ n)
  (h3 : b > a)
  (h4 : b ∣ n)
  (h5 : ∀ m, 1 < m ∧ m < a → ¬ m ∣ n)
  (h6 : ∀ m, a < m ∧ m < b → ¬ m ∣ n)
  (h7 : n = a^a + b^b)
  : n = 260 :=
by sorry

end find_n_l184_184430


namespace james_total_spent_l184_184909

noncomputable def total_cost : ℝ :=
  let milk_price := 3.0
  let bananas_price := 2.0
  let bread_price := 1.5
  let cereal_price := 4.0
  let milk_tax := 0.20
  let bananas_tax := 0.15
  let bread_tax := 0.10
  let cereal_tax := 0.25
  let milk_total := milk_price * (1 + milk_tax)
  let bananas_total := bananas_price * (1 + bananas_tax)
  let bread_total := bread_price * (1 + bread_tax)
  let cereal_total := cereal_price * (1 + cereal_tax)
  milk_total + bananas_total + bread_total + cereal_total

theorem james_total_spent : total_cost = 12.55 :=
  sorry

end james_total_spent_l184_184909


namespace beef_weight_after_processing_l184_184548

noncomputable def initial_weight : ℝ := 840
noncomputable def lost_percentage : ℝ := 35
noncomputable def retained_percentage : ℝ := 100 - lost_percentage
noncomputable def final_weight : ℝ := retained_percentage / 100 * initial_weight

theorem beef_weight_after_processing : final_weight = 546 := by
  sorry

end beef_weight_after_processing_l184_184548


namespace union_of_P_and_Q_l184_184758

def P : Set ℝ := { x | |x| ≥ 3 }
def Q : Set ℝ := { y | ∃ x, y = 2^x - 1 }

theorem union_of_P_and_Q : P ∪ Q = { y | y ≤ -3 ∨ y > -1 } := by
  sorry

end union_of_P_and_Q_l184_184758


namespace poly_solution_l184_184393

-- Definitions for the conditions of the problem
def poly1 (d g : ℚ) := 5 * d ^ 2 - 4 * d + g
def poly2 (d h : ℚ) := 4 * d ^ 2 + h * d - 5
def product (d g h : ℚ) := 20 * d ^ 4 - 31 * d ^ 3 - 17 * d ^ 2 + 23 * d - 10

-- Statement of the problem: proving g + h = 7/2 given the conditions.
theorem poly_solution
  (g h : ℚ)
  (cond : ∀ d : ℚ, poly1 d g * poly2 d h = product d g h) :
  g + h = 7 / 2 :=
by
  sorry

end poly_solution_l184_184393


namespace total_volume_needed_l184_184100

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 12
def box_cost : ℕ := 50 -- in cents to avoid using floats
def total_spent : ℕ := 20000 -- $200 in cents

def volume_of_box : ℕ := box_length * box_width * box_height
def number_of_boxes : ℕ := total_spent / box_cost

theorem total_volume_needed : number_of_boxes * volume_of_box = 1920000 := by
  sorry

end total_volume_needed_l184_184100


namespace sum_of_reciprocals_six_l184_184240

theorem sum_of_reciprocals_six {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x) + (1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_six_l184_184240


namespace total_fish_l184_184924

theorem total_fish (goldfish bluefish : ℕ) (h1 : goldfish = 15) (h2 : bluefish = 7) : goldfish + bluefish = 22 := 
by
  sorry

end total_fish_l184_184924


namespace yoongi_age_l184_184253

theorem yoongi_age
  (H Y : ℕ)
  (h1 : Y = H - 2)
  (h2 : Y + H = 18) :
  Y = 8 :=
by
  sorry

end yoongi_age_l184_184253


namespace raft_min_capacity_l184_184991

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l184_184991


namespace number_of_teams_l184_184405

-- Define the necessary conditions and variables
variable (n : ℕ)
variable (num_games : ℕ)

-- Define the condition that each team plays each other team exactly once 
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The main theorem to prove
theorem number_of_teams (h : total_games n = 91) : n = 14 :=
sorry

end number_of_teams_l184_184405


namespace algebraic_expression_evaluation_l184_184861

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + x - 3 = 0) : x^3 + 2 * x^2 - 2 * x + 2 = 5 :=
by
  sorry

end algebraic_expression_evaluation_l184_184861


namespace solve_for_z_l184_184925

variable {z : ℂ}
def complex_i := Complex.I

theorem solve_for_z (h : 1 - complex_i * z = -1 + complex_i * z) : z = -complex_i := by
  sorry

end solve_for_z_l184_184925


namespace no_solution_for_equation_l184_184138

theorem no_solution_for_equation (x y z : ℤ) : x^3 + y^3 ≠ 9 * z + 5 := 
by
  sorry

end no_solution_for_equation_l184_184138


namespace andrey_gifts_l184_184056

theorem andrey_gifts :
  ∃ (n : ℕ), ∀ (a : ℕ), n(n-2) = a(n-1) + 16 ∧ n = 18 :=
by {
  sorry
}

end andrey_gifts_l184_184056


namespace number_of_cars_on_street_l184_184678

-- Definitions based on conditions
def cars_equally_spaced (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

def distance_between_first_and_last_car (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 242

def distance_between_cars (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

-- Given all conditions, prove n = 45
theorem number_of_cars_on_street (n : ℕ) :
  cars_equally_spaced n →
  distance_between_first_and_last_car n →
  distance_between_cars n →
  n = 45 :=
sorry

end number_of_cars_on_street_l184_184678


namespace gcd_of_1237_and_1849_l184_184144

def gcd_1237_1849 : ℕ := 1

theorem gcd_of_1237_and_1849 : Nat.gcd 1237 1849 = gcd_1237_1849 := by
  sorry

end gcd_of_1237_and_1849_l184_184144


namespace quadratic_inequality_l184_184712

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_inequality (h : f b c (-1) = f b c 3) : f b c 1 < c ∧ c < f b c 3 :=
by
  sorry

end quadratic_inequality_l184_184712


namespace minimum_raft_weight_l184_184989

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l184_184989


namespace arithmetic_seq_terms_greater_than_50_l184_184669

theorem arithmetic_seq_terms_greater_than_50 :
  let a_n (n : ℕ) := 17 + (n-1) * 4
  let num_terms := (19 - 10) + 1
  ∀ (a_n : ℕ → ℕ), ((a_n 1 = 17) ∧ (∃ k, a_n k = 89) ∧ (∀ n, a_n (n + 1) = a_n n + 4)) →
  ∃ m, m = num_terms ∧ ∀ n, (10 ≤ n ∧ n ≤ 19) → a_n n > 50 :=
by
  sorry

end arithmetic_seq_terms_greater_than_50_l184_184669


namespace common_divisors_sum_diff_l184_184086

theorem common_divisors_sum_diff (A B : ℤ) (h : Int.gcd A B = 1) : 
  {d : ℤ | d ∣ A + B ∧ d ∣ A - B} = {1, 2} :=
sorry

end common_divisors_sum_diff_l184_184086


namespace polynomial_roots_correct_l184_184692

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end polynomial_roots_correct_l184_184692


namespace tan_add_sin_l184_184680

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_add_sin (h1 : tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6))
  (h2 : Real.sin (Real.pi / 6) = 1 / 2)
  (h3 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  tan (Real.pi / 6) + 4 * Real.sin (Real.pi / 6) = (Real.sqrt 3 / 3) + 2 := 
sorry

end tan_add_sin_l184_184680


namespace car_speed_is_90_mph_l184_184541

-- Define the given conditions
def distance_yards : ℚ := 22
def time_seconds : ℚ := 0.5
def yards_per_mile : ℚ := 1760

-- Define the car's speed in miles per hour
noncomputable def car_speed_mph : ℚ := (distance_yards / yards_per_mile) * (3600 / time_seconds)

-- The theorem to be proven
theorem car_speed_is_90_mph : car_speed_mph = 90 := by
  sorry

end car_speed_is_90_mph_l184_184541


namespace balancing_point_is_vertex_l184_184352

-- Define a convex polygon and its properties
structure ConvexPolygon (n : ℕ) :=
(vertices : Fin n → Point)

-- Define a balancing point for a convex polygon
def is_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  -- Placeholder for the actual definition that the areas formed by drawing lines from Q to vertices of P are equal
  sorry

-- Define the uniqueness of the balancing point
def unique_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  ∀ R : Point, is_balancing_point P R → R = Q

-- Main theorem statement
theorem balancing_point_is_vertex (P : ConvexPolygon n) (Q : Point) 
  (h_balance : is_balancing_point P Q) (h_unique : unique_balancing_point P Q) : 
  ∃ i : Fin n, Q = P.vertices i :=
sorry

end balancing_point_is_vertex_l184_184352


namespace gcd_poly_correct_l184_184319

-- Define the conditions
def is_even_multiple_of (x k : ℕ) : Prop :=
  ∃ (n : ℕ), x = k * 2 * n

variable (b : ℕ)

-- Given condition
axiom even_multiple_7768 : is_even_multiple_of b 7768

-- Define the polynomials
def poly1 (b : ℕ) := 4 * b * b + 37 * b + 72
def poly2 (b : ℕ) := 3 * b + 8

-- Proof statement
theorem gcd_poly_correct : gcd (poly1 b) (poly2 b) = 8 :=
  sorry

end gcd_poly_correct_l184_184319


namespace sara_initial_peaches_l184_184377

variable (p : ℕ)

def initial_peaches (picked_peaches total_peaches : ℕ) :=
  total_peaches - picked_peaches

theorem sara_initial_peaches :
  initial_peaches 37 61 = 24 :=
by
  -- This follows directly from the definition of initial_peaches
  sorry

end sara_initial_peaches_l184_184377


namespace nacho_will_be_three_times_older_in_future_l184_184192

variable (N D x : ℕ)
variable (h1 : D = 5)
variable (h2 : N + D = 40)
variable (h3 : N + x = 3 * (D + x))

theorem nacho_will_be_three_times_older_in_future :
  x = 10 :=
by {
  -- Given conditions
  sorry
}

end nacho_will_be_three_times_older_in_future_l184_184192


namespace octagon_diag_20_algebraic_expr_positive_l184_184281

def octagon_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diag_20 : octagon_diagonals 8 = 20 := by
  -- Formula for diagonals is used here
  sorry

theorem algebraic_expr_positive (x : ℝ) : 2 * x^2 - 2 * x + 1 > 0 := by
  -- Complete the square to show it's always positive
  sorry

end octagon_diag_20_algebraic_expr_positive_l184_184281


namespace A_intersection_B_complement_l184_184364

noncomputable
def universal_set : Set ℝ := Set.univ

def set_A : Set ℝ := {x | x > 1}

def set_B : Set ℝ := {y | -1 < y ∧ y < 2}

def B_complement : Set ℝ := {y | y <= -1 ∨ y >= 2}

def intersection : Set ℝ := {x | x >= 2}

theorem A_intersection_B_complement :
  (set_A ∩ B_complement) = intersection :=
  sorry

end A_intersection_B_complement_l184_184364


namespace largest_possible_rational_root_l184_184804

noncomputable def rational_root_problem : Prop :=
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧
  ∀ p q : ℤ, (q ≠ 0) → (a * p^2 + b * p + c * q = 0) → 
  (p / q) ≤ -1 / 99

theorem largest_possible_rational_root : rational_root_problem :=
sorry

end largest_possible_rational_root_l184_184804


namespace simplify_expression_l184_184283

-- Define the constants and variables with required conditions
variables {x y z p q r : ℝ}

-- Assume the required distinctness conditions
axiom h1 : x ≠ p 
axiom h2 : y ≠ q 
axiom h3 : z ≠ r 

-- State the theorem to be proven
theorem simplify_expression (h : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (2 * (x - p) / (3 * (r - z))) * (2 * (y - q) / (3 * (p - x))) * (2 * (z - r) / (3 * (q - y))) = -8 / 27 :=
  sorry

end simplify_expression_l184_184283


namespace find_xyz_sum_l184_184879

theorem find_xyz_sum
  (x y z : ℝ)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 + x * y + y^2 = 108)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + z * x + x^2 = 124) :
  x * y + y * z + z * x = 48 := 
  sorry

end find_xyz_sum_l184_184879


namespace f_0_plus_f_1_l184_184324

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_neg1 : f (-1) = 2

theorem f_0_plus_f_1 : f 0 + f 1 = -2 :=
by
  sorry

end f_0_plus_f_1_l184_184324


namespace negate_universal_statement_l184_184813

theorem negate_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negate_universal_statement_l184_184813


namespace necessary_but_not_sufficient_l184_184226

theorem necessary_but_not_sufficient (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≠ b) : ab > 0 :=
  sorry

end necessary_but_not_sufficient_l184_184226


namespace cost_of_previous_hay_l184_184835

theorem cost_of_previous_hay
    (x : ℤ)
    (previous_hay_bales : ℤ)
    (better_quality_hay_cost : ℤ)
    (additional_amount_needed : ℤ)
    (better_quality_hay_bales : ℤ)
    (new_total_cost : ℤ) :
    previous_hay_bales = 10 ∧ 
    better_quality_hay_cost = 18 ∧ 
    additional_amount_needed = 210 ∧ 
    better_quality_hay_bales = 2 * previous_hay_bales ∧ 
    new_total_cost = better_quality_hay_bales * better_quality_hay_cost ∧ 
    new_total_cost - additional_amount_needed = 10 * x → 
    x = 15 := by
  sorry

end cost_of_previous_hay_l184_184835


namespace product_eq_5832_l184_184156

-- Define the integers A, B, C, D that satisfy the given conditions.
variables (A B C D : ℕ)

-- Define the conditions in the problem.
def conditions : Prop :=
  (A + B + C + D = 48) ∧
  (A + 3 = B - 3) ∧
  (A + 3 = C * 3) ∧
  (A + 3 = D / 3)

-- State the final theorem we want to prove.
theorem product_eq_5832 : conditions A B C D → A * B * C * D = 5832 :=
by 
  sorry

end product_eq_5832_l184_184156


namespace avg_reading_time_l184_184445

theorem avg_reading_time (emery_book_time serena_book_time emery_article_time serena_article_time : ℕ)
    (h1 : emery_book_time = 20)
    (h2 : emery_article_time = 2)
    (h3 : emery_book_time * 5 = serena_book_time)
    (h4 : emery_article_time * 3 = serena_article_time) :
    (emery_book_time + emery_article_time + serena_book_time + serena_article_time) / 2 = 64 := by
  sorry

end avg_reading_time_l184_184445


namespace trajectory_is_ellipse_l184_184048

noncomputable def trajectory_of_P (P : ℝ × ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.fst^2 + N.snd^2 = 8 ∧ 
                 ∃ (M : ℝ × ℝ), M.fst = 0 ∧ M.snd = N.snd ∧
                 P.fst = N.fst / 2 ∧ P.snd = N.snd

theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : trajectory_of_P P) : 
  P.fst^2 / 2 + P.snd^2 / 8 = 1 :=
by
  sorry

end trajectory_is_ellipse_l184_184048


namespace ny_mets_fans_count_l184_184103

variable (Y M R : ℕ) -- Variables representing number of fans
variable (k j : ℕ)   -- Helper variables for ratios

theorem ny_mets_fans_count :
  (Y = 3 * k) →
  (M = 2 * k) →
  (M = 4 * j) →
  (R = 5 * j) →
  (Y + M + R = 330) →
  (∃ (k j : ℕ), k = 2 * j) →
  M = 88 := sorry

end ny_mets_fans_count_l184_184103


namespace find_b_l184_184090

variable {a b c : ℚ}

theorem find_b (h1 : a + b + c = 117) (h2 : a + 8 = 4 * c) (h3 : b - 10 = 4 * c) : b = 550 / 9 := by
  sorry

end find_b_l184_184090


namespace quadratic_rewrite_l184_184378

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 4) (h2 : 2 * d * e = 20) (h3 : e^2 + f = -24) :
  d * e = 10 :=
sorry

end quadratic_rewrite_l184_184378


namespace rollo_guinea_pigs_food_l184_184774

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end rollo_guinea_pigs_food_l184_184774


namespace proof_problem_l184_184877

-- Condition for the first part: a quadratic inequality having a solution set
def quadratic_inequality (a : ℝ) :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * x^2 - 3 * x + 2 ≤ 0

-- Condition for the second part: the solution set of a rational inequality
def rational_inequality_solution (a : ℝ) (b : ℝ) :=
  ∀ x : ℝ, (x + 3) / (a * x - b) > 0 ↔ (x < -3 ∨ x > 2)

theorem proof_problem {a : ℝ} {b : ℝ} :
  (quadratic_inequality a → a = 1 ∧ b = 2) ∧ 
  (rational_inequality_solution 1 2) :=
by
  sorry

end proof_problem_l184_184877


namespace percent_decrease_apr_to_may_l184_184789

theorem percent_decrease_apr_to_may (P : ℝ) 
  (h1 : ∀ P : ℝ, P > 0 → (1.35 * P = P + 0.35 * P))
  (h2 : ∀ x : ℝ, P * (1.35 * (1 - x / 100) * 1.5) = 1.62000000000000014 * P)
  (h3 : 0 < x ∧ x < 100)
  : x = 20 :=
  sorry

end percent_decrease_apr_to_may_l184_184789


namespace divisibility_by_65_product_of_four_natural_numbers_l184_184770

def N : ℕ := 2^2022 + 1

theorem divisibility_by_65 : ∃ k : ℕ, N = 65 * k := by
  sorry

theorem product_of_four_natural_numbers :
  ∃ a b c d : ℕ, 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ N = a * b * c * d :=
  by sorry

end divisibility_by_65_product_of_four_natural_numbers_l184_184770


namespace min_capacity_for_raft_l184_184984

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l184_184984


namespace smallest_n_digit_sum_l184_184403

theorem smallest_n_digit_sum :
  ∃ n : ℕ, (∃ (arrangements : ℕ), arrangements > 1000000 ∧ arrangements = (1/2 * ((n + 1) * (n + 2)))) ∧ (1 + n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + n % 10 = 9) :=
sorry

end smallest_n_digit_sum_l184_184403


namespace three_pipes_time_l184_184639

variable (R : ℝ) (T : ℝ)

-- Condition: Two pipes fill the tank in 18 hours
def two_pipes_fill : Prop := 2 * R * 18 = 1

-- Question: How long does it take for three pipes to fill the tank?
def three_pipes_fill : Prop := 3 * R * T = 1

theorem three_pipes_time (h : two_pipes_fill R) : three_pipes_fill R 12 :=
by
  sorry

end three_pipes_time_l184_184639


namespace triangle_centroid_eq_l184_184041

-- Define the proof problem
theorem triangle_centroid_eq
  (P Q R G : ℝ × ℝ) -- Points P, Q, R, and G (the centroid of the triangle PQR)
  (centroid_eq : G = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)) -- Condition that G is the centroid
  (gp_sq_gq_sq_gr_sq_eq : dist G P ^ 2 + dist G Q ^ 2 + dist G R ^ 2 = 22) -- Given GP^2 + GQ^2 + GR^2 = 22
  : dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2 = 66 := -- Prove PQ^2 + PR^2 + QR^2 = 66
sorry -- Proof is omitted

end triangle_centroid_eq_l184_184041


namespace walking_rate_on_escalator_l184_184125

/-- If the escalator moves at 7 feet per second, is 180 feet long, and a person takes 20 seconds to cover this length, then the rate at which the person walks on the escalator is 2 feet per second. -/
theorem walking_rate_on_escalator 
  (escalator_rate : ℝ)
  (length : ℝ)
  (time : ℝ)
  (v : ℝ)
  (h_escalator_rate : escalator_rate = 7)
  (h_length : length = 180)
  (h_time : time = 20)
  (h_distance_formula : length = (v + escalator_rate) * time) :
  v = 2 :=
by
  sorry

end walking_rate_on_escalator_l184_184125


namespace find_N_is_20_l184_184003

theorem find_N_is_20 : ∃ (N : ℤ), ∃ (u v : ℤ), (N + 5 = u ^ 2) ∧ (N - 11 = v ^ 2) ∧ (N = 20) :=
by
  sorry

end find_N_is_20_l184_184003


namespace rhombus_area_l184_184323

theorem rhombus_area 
  (a b : ℝ)
  (side_length : ℝ)
  (diff_diag : ℝ)
  (h_side_len : side_length = Real.sqrt 89)
  (h_diff_diag : diff_diag = 6)
  (h_diag : a - b = diff_diag ∨ b - a = diff_diag)
  (h_side_eq : side_length = Real.sqrt (a^2 + b^2)) :
  (1 / 2 * a * b) * 4 = 80 :=
by
  sorry

end rhombus_area_l184_184323


namespace units_digit_product_l184_184148

theorem units_digit_product : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 := 
by
  sorry

end units_digit_product_l184_184148


namespace average_age_of_team_l184_184532

def total_age (A : ℕ) (N : ℕ) := A * N
def wicket_keeper_age (A : ℕ) := A + 3
def remaining_players_age (A : ℕ) (N : ℕ) (W : ℕ) := (total_age A N) - (A + W)

theorem average_age_of_team
  (A : ℕ)
  (N : ℕ)
  (H1 : N = 11)
  (H2 : A = 28)
  (W : ℕ)
  (H3 : W = wicket_keeper_age A)
  (H4 : (wicket_keeper_age A) = A + 3)
  : (remaining_players_age A N W) / (N - 2) = A - 1 :=
by
  rw [H1, H2, H3, H4]; sorry

end average_age_of_team_l184_184532


namespace count_perfect_squares_mul_36_l184_184891

theorem count_perfect_squares_mul_36 (n : ℕ) (h1 : n < 10^7) (h2 : ∃k, n = k^2) (h3 : 36 ∣ n) :
  ∃ m : ℕ, m = 263 :=
by
  sorry

end count_perfect_squares_mul_36_l184_184891


namespace quadratic_distinct_roots_l184_184898

theorem quadratic_distinct_roots (p q₁ q₂ : ℝ) 
  (h_eq : p = q₁ + q₂ + 1) :
  q₁ ≥ 1/4 → 
  (∃ x, x^2 + x + q₁ = 0 ∧ ∃ x', x' ≠ x ∧ x'^2 + x' + q₁ = 0) 
  ∨ 
  (∃ y, y^2 + p*y + q₂ = 0 ∧ ∃ y', y' ≠ y ∧ y'^2 + p*y' + q₂ = 0) :=
by 
  sorry

end quadratic_distinct_roots_l184_184898


namespace relatively_prime_probability_l184_184954

open Finset

theorem relatively_prime_probability :
  let s := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)
  in let pairs := s.val.powerset.filter (λ t, t.card = 2)
  in (pairs.count (λ t, (t : Multiset ℕ).gcd = 1)).toRational / pairs.card.toRational = 3 / 4 := 
by
  -- Prove that the probability is 3/4
  sorry

end relatively_prime_probability_l184_184954


namespace selected_female_athletes_l184_184116

-- Definitions based on conditions
def total_male_athletes := 56
def total_female_athletes := 42
def selected_male_athletes := 8
def male_to_female_ratio := 4 / 3

-- Problem statement: Prove that the number of selected female athletes is 6
theorem selected_female_athletes :
  selected_male_athletes * (3 / 4) = 6 :=
by 
  -- Placeholder for the proof
  sorry

end selected_female_athletes_l184_184116


namespace absolute_value_of_h_l184_184508

theorem absolute_value_of_h {h : ℝ} :
  (∀ x : ℝ, (x^2 + 2 * h * x = 3) → (∃ r s : ℝ, r + s = -2 * h ∧ r * s = -3 ∧ r^2 + s^2 = 10)) →
  |h| = 1 :=
by
  sorry

end absolute_value_of_h_l184_184508


namespace find_c_l184_184414

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -x^2 + c*x - 8 < 0 ↔ x ∈ set.Ioo (-∞:ℝ) 2 ∪ set.Ioo 6 (∞:ℝ)) → c = 8 :=
by
  sorry

end find_c_l184_184414


namespace trains_crossing_time_l184_184248

theorem trains_crossing_time
  (L speed1 speed2 : ℝ)
  (time_same_direction time_opposite_direction : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : time_same_direction = 40)
  (h4 : 2 * L = (speed1 - speed2) * 5/18 * time_same_direction) :
  time_opposite_direction = 8 := 
sorry

end trains_crossing_time_l184_184248


namespace percentage_of_water_in_mixture_l184_184106

-- Conditions
def percentage_water_LiquidA : ℝ := 0.10
def percentage_water_LiquidB : ℝ := 0.15
def percentage_water_LiquidC : ℝ := 0.25

def volume_LiquidA (v : ℝ) : ℝ := 4 * v
def volume_LiquidB (v : ℝ) : ℝ := 3 * v
def volume_LiquidC (v : ℝ) : ℝ := 2 * v

-- Proof
theorem percentage_of_water_in_mixture (v : ℝ) :
  (percentage_water_LiquidA * volume_LiquidA v + percentage_water_LiquidB * volume_LiquidB v + percentage_water_LiquidC * volume_LiquidC v) / (volume_LiquidA v + volume_LiquidB v + volume_LiquidC v) * 100 = 15 :=
by
  sorry

end percentage_of_water_in_mixture_l184_184106


namespace solve_inequality_l184_184569

theorem solve_inequality :
  {x : ℝ | 0 ≤ x ∧ x ≤ 1 } = {x : ℝ | x * (x - 1) ≤ 0} :=
by sorry

end solve_inequality_l184_184569


namespace solution_set_of_inequality_l184_184793

theorem solution_set_of_inequality (x : ℝ) : (|2 * x - 1| < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_of_inequality_l184_184793


namespace fraction_irreducible_l184_184064

theorem fraction_irreducible (n : ℤ) : Nat.gcd (18 * n + 3).natAbs (12 * n + 1).natAbs = 1 := 
sorry

end fraction_irreducible_l184_184064


namespace time_for_train_to_pass_platform_is_190_seconds_l184_184538

def trainLength : ℕ := 1200
def timeToCrossTree : ℕ := 120
def platformLength : ℕ := 700
def speed (distance time : ℕ) := distance / time
def distanceToCrossPlatform (trainLength platformLength : ℕ) := trainLength + platformLength
def timeToCrossPlatform (distance speed : ℕ) := distance / speed

theorem time_for_train_to_pass_platform_is_190_seconds
  (trainLength timeToCrossTree platformLength : ℕ) (h1 : trainLength = 1200) (h2 : timeToCrossTree = 120) (h3 : platformLength = 700) :
  timeToCrossPlatform (distanceToCrossPlatform trainLength platformLength) (speed trainLength timeToCrossTree) = 190 := by
  sorry

end time_for_train_to_pass_platform_is_190_seconds_l184_184538


namespace triangle_AC_length_l184_184746

open Real

theorem triangle_AC_length (A : ℝ) (AB AC S : ℝ) (h1 : A = π / 3) (h2 : AB = 2) (h3 : S = sqrt 3 / 2) : AC = 1 :=
by
  sorry

end triangle_AC_length_l184_184746


namespace students_passed_l184_184512

noncomputable def total_students : ℕ := 360
noncomputable def bombed : ℕ := (5 * total_students) / 12
noncomputable def not_bombed : ℕ := total_students - bombed
noncomputable def no_show : ℕ := (7 * not_bombed) / 15
noncomputable def remaining_after_no_show : ℕ := not_bombed - no_show
noncomputable def less_than_D : ℕ := 45
noncomputable def remaining_after_less_than_D : ℕ := remaining_after_no_show - less_than_D
noncomputable def technical_issues : ℕ := remaining_after_less_than_D / 8
noncomputable def passed_students : ℕ := remaining_after_less_than_D - technical_issues

theorem students_passed : passed_students = 59 := by
  sorry

end students_passed_l184_184512


namespace multiplication_correct_l184_184416

theorem multiplication_correct (x : ℤ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

end multiplication_correct_l184_184416


namespace simplify_expression_l184_184967

theorem simplify_expression :
  (Real.sin (Real.pi / 6) + (1 / 2) - 2007^0 + abs (-2) = 2) :=
by
  sorry

end simplify_expression_l184_184967


namespace volume_not_determined_l184_184566

noncomputable def tetrahedron_volume_not_unique 
  (area1 area2 area3 : ℝ) (circumradius : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∃ a' b' c', 
      (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧ 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)))

theorem volume_not_determined 
  (area1 area2 area3 circumradius: ℝ) 
  (h: tetrahedron_volume_not_unique area1 area2 area3 circumradius) : 
  ¬ ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∀ a' b' c', 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)) → 
      (a = a' ∧ b = b' ∧ c = c')) := 
by sorry

end volume_not_determined_l184_184566


namespace Jim_paycheck_after_deductions_l184_184481

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end Jim_paycheck_after_deductions_l184_184481


namespace banana_cantaloupe_cost_l184_184005

theorem banana_cantaloupe_cost {a b c d : ℕ} 
  (h1 : a + b + c + d = 20) 
  (h2 : d = 2 * a)
  (h3 : c = a - b) : b + c = 5 :=
sorry

end banana_cantaloupe_cost_l184_184005


namespace problem_l184_184167

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

def p : Prop := ∀ x : ℝ, x ≠ 0 → f x ≥ 4 ∧ (∃ x : ℝ, x > 0 ∧ f x = 4)

def q : Prop := ∀ (A B C : ℝ) (a b c : ℝ),
  A > B ↔ a > b

theorem problem : (¬p) ∧ q :=
sorry

end problem_l184_184167


namespace roots_of_polynomial_l184_184694

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l184_184694


namespace value_of_k_l184_184867

theorem value_of_k (k : ℤ) (h : (∀ x : ℤ, (x^2 - k * x - 6) = (x - 2) * (x + 3))) : k = -1 := by
  sorry

end value_of_k_l184_184867


namespace hyperbola_no_common_points_l184_184915

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_no_common_points (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 + (b^2 / a^2)))
  (h_slope : b / a < 2) :
  e = 2 :=
sorry

end hyperbola_no_common_points_l184_184915


namespace distance_between_foci_of_hyperbola_l184_184225

open Real

-- Definitions based on the given conditions
def asymptote1 (x : ℝ) : ℝ := x + 3
def asymptote2 (x : ℝ) : ℝ := -x + 5
def hyperbola_passes_through (x y : ℝ) : Prop := x = 4 ∧ y = 6
noncomputable def hyperbola_centre : (ℝ × ℝ) := (1, 4)

-- Definition of the hyperbola and the proof problem
theorem distance_between_foci_of_hyperbola (x y : ℝ) (hx : asymptote1 x = y) (hy : asymptote2 x = y) (hpass : hyperbola_passes_through 4 6) :
  2 * (sqrt (5 + 5)) = 2 * sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l184_184225


namespace equation1_solutions_equation2_solutions_l184_184780

theorem equation1_solutions (x : ℝ) : 3 * x^2 - 6 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

theorem equation2_solutions (x : ℝ) : x^2 + 4 * x - 1 = 0 ↔ (x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end equation1_solutions_equation2_solutions_l184_184780


namespace quadratic_roots_distinct_real_l184_184792

theorem quadratic_roots_distinct_real (a b c : ℝ) (h_eq : 2 * a = 2 ∧ 2 * b + -3 = b ∧ 2 * c + 1 = c) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ x : ℝ, (2 * x^2 + (-3) * x + 1 = 0) ↔ (x = x1 ∨ x = x2)) :=
by
  sorry

end quadratic_roots_distinct_real_l184_184792


namespace min_value_expression_l184_184705

noncomputable def sinSquare (θ : ℝ) : ℝ :=
  Real.sin (θ) ^ 2

theorem min_value_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : θ₁ > 0) (h₂ : θ₂ > 0) (h₃ : θ₃ > 0) (h₄ : θ₄ > 0)
  (sum_eq_pi : θ₁ + θ₂ + θ₃ + θ₄ = Real.pi) :
  (2 * sinSquare θ₁ + 1 / sinSquare θ₁) *
  (2 * sinSquare θ₂ + 1 / sinSquare θ₂) *
  (2 * sinSquare θ₃ + 1 / sinSquare θ₃) *
  (2 * sinSquare θ₄ + 1 / sinSquare θ₁) ≥ 81 := 
by
  sorry

end min_value_expression_l184_184705


namespace rectangular_prism_cut_corners_edges_l184_184139

def original_edges : Nat := 12
def corners : Nat := 8
def new_edges_per_corner : Nat := 3
def total_new_edges : Nat := corners * new_edges_per_corner

theorem rectangular_prism_cut_corners_edges :
  original_edges + total_new_edges = 36 := sorry

end rectangular_prism_cut_corners_edges_l184_184139


namespace exponent_product_l184_184335

theorem exponent_product (a : ℝ) (m n : ℕ)
  (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 :=
sorry

end exponent_product_l184_184335


namespace ratio_w_to_y_l184_184233

theorem ratio_w_to_y (w x y z : ℝ) (h1 : w / x = 4 / 3) (h2 : y / z = 5 / 3) (h3 : z / x = 1 / 5) :
  w / y = 4 :=
by
  sorry

end ratio_w_to_y_l184_184233


namespace number_of_boys_l184_184384

theorem number_of_boys (n : ℕ) (h1 : (n * 182 - 60) / n = 180): n = 30 :=
by
  sorry

end number_of_boys_l184_184384


namespace grain_milling_l184_184332

theorem grain_milling (W : ℝ) (h : 0.9 * W = 100) : W = 111.1 :=
sorry

end grain_milling_l184_184332


namespace ellipse_equation_fixed_point_l184_184459

theorem ellipse_equation (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (h₃ : (2^2)/(a^2) + 0/(b^2) = 1)
  (h₄ : (1/2)^2 = (1 - (b^2 / a^2))) :
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1 ↔
  x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem fixed_point (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (h₃ : (2^2)/(a^2) + 0/(b^2) = 1)
  (h₄ : (1/2)^2 = (1 - (b^2 / a^2)))
  (P : ℝ × ℝ) (h₅ : P.1 = -1)
  (MN : set (ℝ × ℝ)) (h₆ : ∀ M N ∈ MN, P = (M + N) / 2)
  (l : set (ℝ × ℝ)) (h₇ : ∀ Q ∈ MN, l ∈ perpThrough P Q) :
  fixed_point l (-1/4, 0) := by
  sorry

end ellipse_equation_fixed_point_l184_184459


namespace part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l184_184714

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem part1_smallest_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem part1_monotonic_interval :
  ∀ k : ℤ, ∀ x, (k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (k * Real.pi + Real.pi / 6) →
  ∃ (b a c : ℝ) (A : ℝ), b + c = 2 * a ∧ 2 * A = A + Real.pi / 3 ∧ 
  f A = 1 / 2 ∧ a = 3 * Real.sqrt 2 := 
sorry

theorem part2_value_of_a :
  ∀ (A b c : ℝ), 
  (∃ (a : ℝ), 2 * a = b + c ∧ 
  f A = 1 / 2 ∧ 
  b * c = 18 ∧ 
  Real.cos A = 1 / 2) → 
  ∃ a, a = 3 * Real.sqrt 2 := 
sorry

end part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l184_184714


namespace angle_between_line_and_plane_l184_184015

noncomputable def vector_angle (m n : ℝ) : ℝ := 120

theorem angle_between_line_and_plane (m n : ℝ) : 
  (vector_angle m n = 120) → (90 - (vector_angle m n - 90) = 30) :=
by sorry

end angle_between_line_and_plane_l184_184015


namespace train_speed_l184_184552

def train_length : ℕ := 110
def bridge_length : ℕ := 265
def crossing_time : ℕ := 30

def speed_in_m_per_s (d t : ℕ) : ℕ := d / t
def speed_in_km_per_hr (s : ℕ) : ℕ := s * 36 / 10

theorem train_speed :
  speed_in_km_per_hr (speed_in_m_per_s (train_length + bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l184_184552


namespace train_pass_bridge_in_approx_26_64_sec_l184_184286

noncomputable def L_train : ℝ := 240 -- Length of the train in meters
noncomputable def L_bridge : ℝ := 130 -- Length of the bridge in meters
noncomputable def Speed_train_kmh : ℝ := 50 -- Speed of the train in km/h
noncomputable def Speed_train_ms : ℝ := (Speed_train_kmh * 1000) / 3600 -- Speed of the train in m/s
noncomputable def Total_distance : ℝ := L_train + L_bridge -- Total distance to be covered by the train
noncomputable def Time : ℝ := Total_distance / Speed_train_ms -- Time to pass the bridge

theorem train_pass_bridge_in_approx_26_64_sec : |Time - 26.64| < 0.01 := by
  sorry

end train_pass_bridge_in_approx_26_64_sec_l184_184286


namespace sales_ratio_l184_184766

def large_price : ℕ := 60
def small_price : ℕ := 30
def last_month_large_paintings : ℕ := 8
def last_month_small_paintings : ℕ := 4
def this_month_sales : ℕ := 1200

theorem sales_ratio :
  (this_month_sales : ℕ) = 2 * (last_month_large_paintings * large_price + last_month_small_paintings * small_price) :=
by
  -- We will just state the proof steps as sorry.
  sorry

end sales_ratio_l184_184766


namespace daisies_left_l184_184752

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9
def total_daisies : ℕ := initial_daisies + sister_daisies
def daisies_given_to_mother : ℕ := total_daisies / 2
def remaining_daisies : ℕ := total_daisies - daisies_given_to_mother

theorem daisies_left : remaining_daisies = 7 := by
  sorry

end daisies_left_l184_184752


namespace vertex_of_parabola_l184_184932

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end vertex_of_parabola_l184_184932


namespace total_gifts_l184_184053

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l184_184053


namespace required_run_rate_l184_184034

def initial_run_rate : ℝ := 3.2
def overs_completed : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 50

theorem required_run_rate :
  (target_runs - initial_run_rate * overs_completed) / remaining_overs = 5 := 
by
  sorry

end required_run_rate_l184_184034


namespace compute_x_l184_184360

theorem compute_x 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 0.1)
  (hs1 : ∑' n, 4 * x^n = 4 / (1 - x))
  (hs2 : ∑' n, 4 * (10^n - 1) * x^n = 4 * (4 / (1 - x))) :
  x = 3 / 40 :=
by
  sorry

end compute_x_l184_184360


namespace inverse_function_b_value_l184_184320

theorem inverse_function_b_value (b : ℝ) :
  (∀ x, ∃ y, 2^x + b = y) ∧ (∃ x, ∃ y, (x, y) = (2, 5)) → b = 1 :=
by
  sorry

end inverse_function_b_value_l184_184320


namespace cistern_problem_l184_184425

theorem cistern_problem (T : ℝ) (h1 : (1 / 2 - 1 / T) = 1 / 2.571428571428571) : T = 9 :=
by
  sorry

end cistern_problem_l184_184425


namespace count_letters_with_both_l184_184736

theorem count_letters_with_both (a b c x : ℕ) 
  (h₁ : a = 24) 
  (h₂ : b = 7) 
  (h₃ : c = 40) 
  (H : a + b + x = c) : 
  x = 9 :=
by {
  -- Proof here
  sorry
}

end count_letters_with_both_l184_184736


namespace lcm_150_414_l184_184521

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 :=
by
  sorry

end lcm_150_414_l184_184521


namespace avg_price_of_towels_l184_184665

def towlesScenario (t1 t2 t3 : ℕ) (price1 price2 price3 : ℕ) : ℕ :=
  (t1 * price1 + t2 * price2 + t3 * price3) / (t1 + t2 + t3)

theorem avg_price_of_towels :
  towlesScenario 3 5 2 100 150 500 = 205 := by
  sorry

end avg_price_of_towels_l184_184665


namespace quadratic_equation_no_real_roots_l184_184234

theorem quadratic_equation_no_real_roots :
  ∀ (x : ℝ), ¬ (x^2 - 2 * x + 3 = 0) :=
by
  intro x
  sorry

end quadratic_equation_no_real_roots_l184_184234


namespace total_cows_l184_184269

/-- A farmer divides his herd of cows among his four sons.
The first son receives 1/3 of the herd, the second son receives 1/6,
the third son receives 1/9, and the rest goes to the fourth son,
who receives 12 cows. Calculate the total number of cows in the herd
-/
theorem total_cows (n : ℕ) (h1 : (n : ℚ) * (1 / 3) + (n : ℚ) * (1 / 6) + (n : ℚ) * (1 / 9) + 12 = n) : n = 54 := by
  sorry

end total_cows_l184_184269


namespace max_profit_price_l184_184113

-- Define the conditions
def hotel_rooms : ℕ := 50
def base_price : ℕ := 180
def price_increase : ℕ := 10
def expense_per_room : ℕ := 20

-- Define the price as a function of x
def room_price (x : ℕ) : ℕ := base_price + price_increase * x

-- Define the number of occupied rooms as a function of x
def occupied_rooms (x : ℕ) : ℕ := hotel_rooms - x

-- Define the profit function
def profit (x : ℕ) : ℕ := (room_price x - expense_per_room) * occupied_rooms x

-- The statement to be proven:
theorem max_profit_price : ∃ (x : ℕ), room_price x = 350 ∧ ∀ y : ℕ, profit y ≤ profit x :=
by
  sorry

end max_profit_price_l184_184113


namespace hcf_of_36_and_x_is_12_l184_184698

theorem hcf_of_36_and_x_is_12 (x : ℕ) (h : Nat.gcd 36 x = 12) : x = 48 :=
sorry

end hcf_of_36_and_x_is_12_l184_184698


namespace no_positive_integer_n_for_perfect_squares_l184_184771

theorem no_positive_integer_n_for_perfect_squares :
  ∀ (n : ℕ), 0 < n → ¬ (∃ a b : ℤ, (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2) :=
by
  sorry

end no_positive_integer_n_for_perfect_squares_l184_184771


namespace angle_difference_l184_184279

theorem angle_difference (X Y Z Z1 Z2 : ℝ) (h1 : Y = 2 * X) (h2 : X = 30) (h3 : Z1 + Z2 = Z) (h4 : Z1 = 60) (h5 : Z2 = 30) : Z1 - Z2 = 30 := 
by 
  sorry

end angle_difference_l184_184279


namespace freight_capacity_equation_l184_184110

theorem freight_capacity_equation
  (x : ℝ)
  (h1 : ∀ (capacity_large capacity_small : ℝ), capacity_large = capacity_small + 4)
  (h2 : ∀ (n_large n_small : ℕ), (n_large : ℝ) = 80 / (x + 4) ∧ (n_small : ℝ) = 60 / x → n_large = n_small) :
  (80 / (x + 4)) = (60 / x) :=
by
  sorry

end freight_capacity_equation_l184_184110


namespace total_trip_time_l184_184096

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end total_trip_time_l184_184096


namespace quadratic_equation_general_form_l184_184631

theorem quadratic_equation_general_form (x : ℝ) (h : 4 * x = x^2 - 8) : x^2 - 4 * x - 8 = 0 :=
sorry

end quadratic_equation_general_form_l184_184631


namespace tennis_tournament_rounds_needed_l184_184969

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end tennis_tournament_rounds_needed_l184_184969


namespace price_per_cup_l184_184298

theorem price_per_cup
  (num_trees : ℕ)
  (oranges_per_tree_g : ℕ)
  (oranges_per_tree_a : ℕ)
  (oranges_per_tree_m : ℕ)
  (oranges_per_cup : ℕ)
  (total_income : ℕ)
  (h_g : num_trees = 110)
  (h_a : oranges_per_tree_g = 600)
  (h_al : oranges_per_tree_a = 400)
  (h_m : oranges_per_tree_m = 500)
  (h_o : oranges_per_cup = 3)
  (h_income : total_income = 220000) :
  total_income / (((num_trees * oranges_per_tree_g) + (num_trees * oranges_per_tree_a) + (num_trees * oranges_per_tree_m)) / oranges_per_cup) = 4 :=
by
  repeat {sorry}

end price_per_cup_l184_184298


namespace roland_thread_length_l184_184499

noncomputable def length_initial : ℝ := 12
noncomputable def length_two_thirds : ℝ := (2 / 3) * length_initial
noncomputable def length_increased : ℝ := length_initial + length_two_thirds
noncomputable def length_half_increased : ℝ := (1 / 2) * length_increased
noncomputable def length_total : ℝ := length_increased + length_half_increased
noncomputable def length_inches : ℝ := length_total / 2.54

theorem roland_thread_length : length_inches = 11.811 :=
by sorry

end roland_thread_length_l184_184499


namespace final_number_after_increase_l184_184649

-- Define the original number and the percentage increase
def original_number : ℕ := 70
def increase_percentage : ℝ := 0.50

-- Define the function to calculate the final number after the increase
def final_number : ℝ := original_number * (1 + increase_percentage)

-- The proof statement that the final number is 105
theorem final_number_after_increase : final_number = 105 :=
by
  sorry

end final_number_after_increase_l184_184649


namespace directrix_parabola_l184_184845

theorem directrix_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 8 * x^2 + 5) : 
  ∃ c : ℝ, ∀ x, y x = 8 * x^2 + 5 ∧ c = 159 / 32 :=
by
  use 159 / 32
  repeat { sorry }

end directrix_parabola_l184_184845


namespace extreme_point_a_zero_l184_184723

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end extreme_point_a_zero_l184_184723


namespace problem1_problem2_l184_184709

variable (α : ℝ)

-- Equivalent problem 1
theorem problem1 (h : Real.tan α = 7) : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 := 
  sorry

-- Equivalent problem 2
theorem problem2 (h : Real.tan α = 7) : Real.sin α * Real.cos α = 7 / 50 := 
  sorry

end problem1_problem2_l184_184709


namespace ira_addition_olya_subtraction_addition_l184_184674

theorem ira_addition (x : ℤ) (h : (11 + x) / (41 + x : ℚ) = 3 / 8) : x = 7 :=
  sorry

theorem olya_subtraction_addition (y : ℤ) (h : (37 - y) / (63 + y : ℚ) = 3 / 17) : y = 22 :=
  sorry

end ira_addition_olya_subtraction_addition_l184_184674


namespace y_intercept_of_line_l184_184402

theorem y_intercept_of_line :
  ∃ y, (∀ x : ℝ, 2 * x - 3 * y = 6) ∧ (y = -2) :=
sorry

end y_intercept_of_line_l184_184402


namespace f_of_g_of_pi_div_two_l184_184020

def f (x : ℝ) : ℝ := x^3 + 2
def g (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem f_of_g_of_pi_div_two : f (g (Real.pi / 2)) = 127 := by
  sorry

end f_of_g_of_pi_div_two_l184_184020


namespace correct_average_of_corrected_number_l184_184418

theorem correct_average_of_corrected_number (num_list : List ℤ) (wrong_num correct_num : ℤ) (n : ℕ)
  (hn : n = 10)
  (haverage : (num_list.sum / n) = 5)
  (hwrong : wrong_num = 26)
  (hcorrect : correct_num = 36)
  (hnum_list_sum : num_list.sum + correct_num - wrong_num = num_list.sum + 10) :
  (num_list.sum + 10) / n = 6 :=
by
  sorry

end correct_average_of_corrected_number_l184_184418


namespace questionnaires_drawn_l184_184278

theorem questionnaires_drawn
  (units : ℕ → ℕ)
  (h_arithmetic : ∀ n, units (n + 1) - units n = units 1 - units 0)
  (h_total : units 0 + units 1 + units 2 + units 3 = 100)
  (h_unitB : units 1 = 20) :
  units 3 = 40 :=
by
  -- Proof would go here
  -- Establish that the arithmetic sequence difference is 10, then compute unit D (units 3)
  sorry

end questionnaires_drawn_l184_184278


namespace cube_negative_iff_l184_184492

theorem cube_negative_iff (x : ℝ) : x < 0 ↔ x^3 < 0 :=
sorry

end cube_negative_iff_l184_184492


namespace factorization_identity_l184_184301

theorem factorization_identity (a b : ℝ) : 
  -a^3 + 12 * a^2 * b - 36 * a * b^2 = -a * (a - 6 * b)^2 :=
by 
  sorry

end factorization_identity_l184_184301


namespace rhombus_diagonals_l184_184087

theorem rhombus_diagonals (p d_sum : ℝ) (h₁ : p = 100) (h₂ : d_sum = 62) :
  ∃ d₁ d₂ : ℝ, (d₁ + d₂ = d_sum) ∧ (d₁^2 + d₂^2 = (p/4)^2 * 4) ∧ ((d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48)) :=
by
  sorry

end rhombus_diagonals_l184_184087


namespace simplify_and_evaluate_l184_184779

theorem simplify_and_evaluate (m : ℤ) (h : m = -2) :
  let expr := (m / (m^2 - 9)) / (1 + (3 / (m - 3)))
  expr = 1 :=
by
  sorry

end simplify_and_evaluate_l184_184779


namespace determine_n_l184_184564

theorem determine_n (n : ℕ) (h : 17^(4 * n) = (1 / 17)^(n - 30)) : n = 6 :=
by {
  sorry
}

end determine_n_l184_184564


namespace distinct_roots_polynomial_l184_184202

theorem distinct_roots_polynomial (a b : ℂ) (h₁ : a ≠ b) (h₂: a^3 + 3*a^2 + a + 1 = 0) (h₃: b^3 + 3*b^2 + b + 1 = 0) :
  a^2 * b + a * b^2 + 3 * a * b = 1 :=
sorry

end distinct_roots_polynomial_l184_184202


namespace square_pieces_placement_l184_184059

theorem square_pieces_placement (n : ℕ) (H : n = 8) :
  {m : ℕ // m = 17} :=
sorry

end square_pieces_placement_l184_184059


namespace reduction_percentage_40_l184_184897

theorem reduction_percentage_40 (P : ℝ) : 
  1500 * 1.20 - (P / 100 * (1500 * 1.20)) = 1080 ↔ P = 40 :=
by
  sorry

end reduction_percentage_40_l184_184897


namespace vector_parallel_sum_l184_184177

theorem vector_parallel_sum (m n : ℝ) (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, -1, 3))
  (h_b : b = (4, m, n))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  m + n = 4 :=
sorry

end vector_parallel_sum_l184_184177


namespace area_after_shortening_other_side_l184_184777

-- Define initial dimensions of the index card
def initial_length := 5
def initial_width := 7
def initial_area := initial_length * initial_width

-- Define the area condition when one side is shortened by 2 inches
def shortened_side_length := initial_length - 2
def new_area_after_shortening_one_side := 21

-- Definition of the problem condition that results in 21 square inches area
def condition := 
  (shortened_side_length * initial_width = new_area_after_shortening_one_side)

-- Final statement
theorem area_after_shortening_other_side :
  condition →
  (initial_length * (initial_width - 2) = 25) :=
by
  intro h
  sorry

end area_after_shortening_other_side_l184_184777


namespace maria_correct_answers_l184_184905

theorem maria_correct_answers (x : ℕ) (n c d s : ℕ) (h1 : n = 30) (h2 : c = 20) (h3 : d = 5) (h4 : s = 325)
  (h5 : n = x + (n - x)) : 20 * x - 5 * (30 - x) = 325 → x = 19 :=
by 
  intros h_eq
  sorry

end maria_correct_answers_l184_184905


namespace probability_of_exactly_one_success_probability_of_at_least_one_success_l184_184371

variable (PA : ℚ := 1/2)
variable (PB : ℚ := 2/5)
variable (P_A_bar : ℚ := 1 - PA)
variable (P_B_bar : ℚ := 1 - PB)

theorem probability_of_exactly_one_success :
  PA * P_B_bar + PB * P_A_bar = 1/2 :=
sorry

theorem probability_of_at_least_one_success :
  1 - (P_A_bar * P_A_bar * P_B_bar * P_B_bar) = 91/100 :=
sorry

end probability_of_exactly_one_success_probability_of_at_least_one_success_l184_184371


namespace find_x_of_equation_l184_184027

-- Defining the condition and setting up the proof goal
theorem find_x_of_equation
  (h : (1/2)^25 * (1/x)^12.5 = 1/(18^25)) :
  x = 0.1577 := 
sorry

end find_x_of_equation_l184_184027


namespace circle_radius_l184_184395

-- Given conditions
def central_angle : ℝ := 225
def perimeter : ℝ := 83
noncomputable def pi_val : ℝ := Real.pi

-- Formula for the radius
noncomputable def radius : ℝ := 332 / (5 * pi_val + 8)

-- Prove that the radius is correct given the conditions
theorem circle_radius (theta : ℝ) (P : ℝ) (r : ℝ) (h_theta : theta = central_angle) (h_P : P = perimeter) :
  r = radius :=
sorry

end circle_radius_l184_184395


namespace inequality_holds_for_all_x_iff_range_m_l184_184729

theorem inequality_holds_for_all_x_iff_range_m (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ m ∈ Ioc (-10) 2 := by
  sorry

end inequality_holds_for_all_x_iff_range_m_l184_184729


namespace probability_at_least_one_female_probability_all_males_given_at_least_one_male_l184_184118

-- Definitions and conditions
def total_volunteers := 7
def male_volunteers := 4
def female_volunteers := 3
def selected_volunteers := 3

-- The number of ways to choose 3 out of 7 volunteers
def total_combinations := Nat.choose total_volunteers selected_volunteers

-- The number of ways to choose 3 males out of 4 male volunteers
def male_combinations := Nat.choose male_volunteers selected_volunteers

-- Definition of the event of selecting all males, and related probabilities
def P_all_males := (male_combinations : ℝ) / (total_combinations : ℝ)

-- Statements to prove
theorem probability_at_least_one_female :
  1 - P_all_males = 31 / 35 :=
by sorry

-- Probability of at least one male volunteer being selected
def female_only_combinations : ℝ := Nat.choose female_volunteers selected_volunteers

def P_at_least_one_male :=
  1 - (female_only_combinations / total_combinations)

-- Conditional probability of selecting all males given at least one male
theorem probability_all_males_given_at_least_one_male :
  P_all_males / P_at_least_one_male = 2 / 17 :=
by sorry

end probability_at_least_one_female_probability_all_males_given_at_least_one_male_l184_184118


namespace solution_for_equation_l184_184565

theorem solution_for_equation (m n : ℕ) (h : 0 < m ∧ 0 < n ∧ 2 * m^2 = 3 * n^3) :
  ∃ k : ℕ, 0 < k ∧ m = 18 * k^3 ∧ n = 6 * k^2 :=
by sorry

end solution_for_equation_l184_184565


namespace dot_product_eq_l184_184439

def vector1 : ℝ × ℝ := (-3, 0)
def vector2 : ℝ × ℝ := (7, 9)

theorem dot_product_eq :
  (vector1.1 * vector2.1 + vector1.2 * vector2.2) = -21 :=
by
  sorry

end dot_product_eq_l184_184439


namespace hexagon_sum_balanced_assignment_exists_l184_184959

-- Definitions based on the conditions
def is_valid_assignment (a b c d e f g : ℕ) : Prop :=
a + b + g = a + c + g ∧ a + b + g = a + d + g ∧ a + b + g = a + e + g ∧
a + b + g = b + c + g ∧ a + b + g = b + d + g ∧ a + b + g = b + e + g ∧
a + b + g = c + d + g ∧ a + b + g = c + e + g ∧ a + b + g = d + e + g

-- The theorem we want to prove
theorem hexagon_sum_balanced_assignment_exists :
  ∃ (a b c d e f g : ℕ), 
  (a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 2 ∨ b = 3 ∨ b = 5) ∧ 
  (c = 2 ∨ c = 3 ∨ c = 5) ∧ 
  (d = 2 ∨ d = 3 ∨ d = 5) ∧ 
  (e = 2 ∨ e = 3 ∨ e = 5) ∧
  (f = 2 ∨ f = 3 ∨ f = 5) ∧
  (g = 2 ∨ g = 3 ∨ g = 5) ∧
  is_valid_assignment a b c d e f g :=
sorry

end hexagon_sum_balanced_assignment_exists_l184_184959


namespace andrey_gifts_l184_184057

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l184_184057


namespace sum_of_altitudes_of_triangle_l184_184837

open Real

noncomputable def sum_of_altitudes (a b c : ℝ) : ℝ :=
  let inter_x := -c / a
  let inter_y := -c / b
  let vertex1 := (inter_x, 0)
  let vertex2 := (0, inter_y)
  let vertex3 := (0, 0)
  let area_triangle := (1 / 2) * abs (inter_x * inter_y)
  let altitude_x := abs inter_x
  let altitude_y := abs inter_y
  let altitude_line := abs c / sqrt (a ^ 2 + b ^ 2)
  altitude_x + altitude_y + altitude_line

theorem sum_of_altitudes_of_triangle :
  sum_of_altitudes 15 6 90 = 21 + 10 * sqrt (1 / 29) :=
by
  sorry

end sum_of_altitudes_of_triangle_l184_184837


namespace value_of_a_l184_184329

theorem value_of_a (a : ℝ) : 
  ({2, 3} : Set ℝ) ⊆ ({1, 2, a} : Set ℝ) → a = 3 :=
by
  sorry

end value_of_a_l184_184329


namespace fresh_grapes_water_content_l184_184310

theorem fresh_grapes_water_content:
  ∀ (P : ℝ), 
  (∀ (x y : ℝ), P = x) → 
  (∃ (fresh_grapes dry_grapes : ℝ), fresh_grapes = 25 ∧ dry_grapes = 3.125 ∧ 
  (100 - P) / 100 * fresh_grapes = 0.8 * dry_grapes ) → 
  P = 90 :=
by 
  sorry

end fresh_grapes_water_content_l184_184310


namespace find_roots_of_polynomial_l184_184695

theorem find_roots_of_polynomial :
  (∃ (a b : ℝ), 
    Multiplicity (polynomial.C a) (polynomial.C (Real.ofRat 2)) = 2 ∧ 
    Multiplicity (polynomial.C b) (polynomial.C (Real.ofRat 1)) = 1) ∧ 
  (x^3 - 7 * x^2 + 14 * x - 8 = 
    (x - 1) * (x - 2)^2) := sorry

end find_roots_of_polynomial_l184_184695


namespace bryan_total_books_and_magazines_l184_184127

-- Define the conditions
def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def bookshelves : ℕ := 29

-- Define the total books and magazines
def total_books : ℕ := books_per_shelf * bookshelves
def total_magazines : ℕ := magazines_per_shelf * bookshelves
def total_books_and_magazines : ℕ := total_books + total_magazines

-- The proof problem statement
theorem bryan_total_books_and_magazines : total_books_and_magazines = 2436 := 
by
  sorry

end bryan_total_books_and_magazines_l184_184127


namespace relationship_between_xyz_l184_184715

theorem relationship_between_xyz (x y z : ℝ) (h1 : x - z < y) (h2 : x + z > y) : -z < x - y ∧ x - y < z :=
by
  sorry

end relationship_between_xyz_l184_184715


namespace find_integer_n_l184_184250

theorem find_integer_n : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ 54126 % 23 = n :=
by
  use 13
  sorry

end find_integer_n_l184_184250


namespace fraction_of_number_l184_184590

theorem fraction_of_number (F : ℚ) (h : 0.5 * F * 120 = 36) : F = 3 / 5 :=
by
  sorry

end fraction_of_number_l184_184590


namespace elephant_entry_rate_l184_184410

def initial : ℕ := 30000
def exit_rate : ℕ := 2880
def exodus_hours : ℕ := 4
def final : ℕ := 28980
def entry_hours : ℕ := 7

theorem elephant_entry_rate :
  let elephants_left := exit_rate * exodus_hours
  let remaining_elephants := initial - elephants_left
  let new_elephants := final - remaining_elephants
  let entry_rate := new_elephants / entry_hours
  entry_rate = 1500 :=
by 
  rw [mul_comm exit_rate exodus_hours, mul_comm initial exodus_hours]
  rw [sub_eq_add_neg initial elephants_left, sub_eq_add_neg final remaining_elephants]
  exact sorry

end elephant_entry_rate_l184_184410


namespace correct_subtraction_result_l184_184743

-- Definition of numbers:
def tens_digit := 2
def ones_digit := 4
def correct_number := 10 * tens_digit + ones_digit
def incorrect_number := 59
def incorrect_result := 14
def Z := incorrect_result + incorrect_number

-- Statement of the theorem
theorem correct_subtraction_result : Z - correct_number = 49 :=
by
  sorry

end correct_subtraction_result_l184_184743


namespace price_of_other_frisbees_proof_l184_184825

noncomputable def price_of_other_frisbees (P : ℝ) : Prop :=
  ∃ x : ℝ, x + (60 - x) = 60 ∧ x ≥ 0 ∧ P * x + 4 * (60 - x) = 204 ∧ (60 - x) ≥ 24

theorem price_of_other_frisbees_proof : price_of_other_frisbees 3 :=
by
  sorry

end price_of_other_frisbees_proof_l184_184825


namespace tan_sum_product_l184_184196

theorem tan_sum_product (A B C : ℝ) (h_eq: Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := by
  sorry

end tan_sum_product_l184_184196


namespace find_radius_l184_184817

theorem find_radius (abbc: ℝ) (adbd: ℝ) (bccc: ℝ) (dcdd: ℝ) (R: ℝ)
  (h1: abbc = 4) (h2: adbd = 4) (h3: bccc = 2) (h4: dcdd = 1) :
  R = 5 :=
sorry

end find_radius_l184_184817
