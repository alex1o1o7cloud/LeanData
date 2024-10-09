import Mathlib

namespace length_of_bridge_l1248_124873

noncomputable def speed_kmhr_to_ms (v : ℕ) : ℝ := (v : ℝ) * (1000 / 3600)

noncomputable def distance_traveled (v : ℝ) (t : ℕ) : ℝ := v * (t : ℝ)

theorem length_of_bridge 
  (length_train : ℕ) -- 90 meters
  (speed_train_kmhr : ℕ) -- 45 km/hr
  (time_cross_bridge : ℕ) -- 30 seconds
  (conversion_factor : ℝ := 1000 / 3600) 
  : ℝ := 
  let speed_train_ms := speed_kmhr_to_ms speed_train_kmhr
  let total_distance := distance_traveled speed_train_ms time_cross_bridge
  total_distance - (length_train : ℝ)

example : length_of_bridge 90 45 30 = 285 := by
  sorry

end length_of_bridge_l1248_124873


namespace mom_approach_is_sampling_survey_l1248_124825

def is_sampling_survey (action : String) : Prop :=
  action = "tasting a little bit"

def is_census (action : String) : Prop :=
  action = "tasting the entire dish"

theorem mom_approach_is_sampling_survey :
  is_sampling_survey "tasting a little bit" :=
by {
  -- This follows from the given conditions directly.
  sorry
}

end mom_approach_is_sampling_survey_l1248_124825


namespace largest_value_fraction_l1248_124836

noncomputable def largest_value (x y : ℝ) : ℝ := (x + y) / x

theorem largest_value_fraction
  (x y : ℝ)
  (hx1 : -5 ≤ x)
  (hx2 : x ≤ -3)
  (hy1 : 3 ≤ y)
  (hy2 : y ≤ 5)
  (hy_odd : ∃ k : ℤ, y = 2 * k + 1) :
  largest_value x y = 0.4 :=
sorry

end largest_value_fraction_l1248_124836


namespace speed_of_faster_train_l1248_124804

theorem speed_of_faster_train
  (length_each_train : ℕ)
  (length_in_meters : length_each_train = 50)
  (speed_slower_train_kmh : ℝ)
  (speed_slower : speed_slower_train_kmh = 36)
  (pass_time_seconds : ℕ)
  (pass_time : pass_time_seconds = 36) :
  ∃ speed_faster_train_kmh, speed_faster_train_kmh = 46 :=
by
  sorry

end speed_of_faster_train_l1248_124804


namespace simplest_form_eq_a_l1248_124897

theorem simplest_form_eq_a (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + (a / (1 - a)))) = a :=
by sorry

end simplest_form_eq_a_l1248_124897


namespace jacket_purchase_price_l1248_124878

theorem jacket_purchase_price (P S SP : ℝ)
  (h1 : S = P + 0.40 * S)
  (h2 : SP = 0.80 * S)
  (h3 : SP - P = 18) :
  P = 54 :=
by
  sorry

end jacket_purchase_price_l1248_124878


namespace arithmetic_seq_solution_l1248_124840

variables (a : ℕ → ℤ) (d : ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) - a n = d

def seq_cond (a : ℕ → ℤ) (d : ℤ) : Prop :=
is_arithmetic_sequence a d ∧ (a 2 + a 6 = a 8)

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem arithmetic_seq_solution :
  ∀ (a : ℕ → ℤ) (d : ℤ), seq_cond a d → (a 2 - a 1 ≠ 0) → 
    (sum_first_n a 5 / a 5) = 3 :=
by
  intros a d h_cond h_d_ne_zero
  sorry

end arithmetic_seq_solution_l1248_124840


namespace average_of_last_four_numbers_l1248_124835

theorem average_of_last_four_numbers
  (avg_seven : ℝ) (avg_first_three : ℝ) (avg_last_four : ℝ)
  (h1 : avg_seven = 62) (h2 : avg_first_three = 55) :
  avg_last_four = 67.25 := 
by
  sorry

end average_of_last_four_numbers_l1248_124835


namespace prime_division_l1248_124830

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end prime_division_l1248_124830


namespace circle_equation_l1248_124895

theorem circle_equation (x y : ℝ) (h : ∀ x y : ℝ, x^2 + y^2 ≥ 64) :
  x^2 + y^2 - 64 = 0 ↔ x = 0 ∧ y = 0 :=
by
  sorry

end circle_equation_l1248_124895


namespace minimal_colors_l1248_124864

def complete_graph (n : ℕ) := Type

noncomputable def color_edges (G : complete_graph 2015) := ℕ → ℕ → ℕ

theorem minimal_colors (G : complete_graph 2015) (color : color_edges G) :
  (∀ {u v w : ℕ} (h1 : u ≠ v) (h2 : v ≠ w) (h3 : w ≠ u), color u v ≠ color v w ∧ color u v ≠ color u w ∧ color u w ≠ color v w) →
  ∃ C: ℕ, C = 2015 := 
sorry

end minimal_colors_l1248_124864


namespace quadratic_function_symmetry_l1248_124899

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := -x^2 + b * x + c

-- State the problem as a theorem
theorem quadratic_function_symmetry (b c : ℝ) (h_symm : ∀ x, f x b c = f (4 - x) b c) :
  f 2 b c > f 1 b c ∧ f 1 b c > f 4 b c :=
by
  -- Include a placeholder for the proof
  sorry

end quadratic_function_symmetry_l1248_124899


namespace fraction_is_one_fourth_l1248_124893

theorem fraction_is_one_fourth
  (f : ℚ)
  (m : ℕ)
  (h1 : (1 / 5) ^ m * f^2 = 1 / (10 ^ 4))
  (h2 : m = 4) : f = 1 / 4 := by
  sorry

end fraction_is_one_fourth_l1248_124893


namespace not_prime_for_all_n_ge_2_l1248_124810

theorem not_prime_for_all_n_ge_2 (n : ℕ) (hn : n ≥ 2) : ¬ Prime (2 * (n^3 + n + 1)) := 
by
  sorry

end not_prime_for_all_n_ge_2_l1248_124810


namespace max_robot_weight_l1248_124887

-- Definitions of the given conditions
def standard_robot_weight : ℕ := 100
def battery_weight : ℕ := 20
def min_payload : ℕ := 10
def max_payload : ℕ := 25
def min_robot_weight_extra : ℕ := 5
def min_robot_weight : ℕ := standard_robot_weight + min_robot_weight_extra

-- Definition for total minimum weight of the robot
def min_total_weight : ℕ := min_robot_weight + battery_weight + min_payload

-- Proposition for the maximum weight condition
theorem max_robot_weight :
  2 * min_total_weight = 270 :=
by
  -- Insert proof here
  sorry

end max_robot_weight_l1248_124887


namespace range_of_m_l1248_124817

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : y1 = x1^2 - 4*x1 + 3)
  (h2 : y2 = x2^2 - 4*x2 + 3) (h3 : -1 < x1) (h4 : x1 < 1)
  (h5 : m > 0) (h6 : m-1 < x2) (h7 : x2 < m) (h8 : y1 ≠ y2) :
  (2 ≤ m ∧ m ≤ 3) ∨ (m ≥ 6) :=
sorry

end range_of_m_l1248_124817


namespace other_solution_of_quadratic_l1248_124802

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop :=
  65 * x^2 - 104 * x + 31 = 0

-- Main theorem statement
theorem other_solution_of_quadratic :
  quadratic_eq (6 / 5) → quadratic_eq (5 / 13) :=
by
  intro h
  sorry

end other_solution_of_quadratic_l1248_124802


namespace non_working_games_count_l1248_124881

-- Definitions based on conditions
def total_games : Nat := 15
def total_earnings : Nat := 30
def price_per_game : Nat := 5

-- Definition to be proved
def working_games : Nat := total_earnings / price_per_game
def non_working_games : Nat := total_games - working_games

-- Statement to be proved
theorem non_working_games_count : non_working_games = 9 :=
by
  sorry

end non_working_games_count_l1248_124881


namespace correct_sentence_completion_l1248_124875

-- Define the possible options
inductive Options
| A : Options  -- "However he was reminded frequently"
| B : Options  -- "No matter he was reminded frequently"
| C : Options  -- "However frequently he was reminded"
| D : Options  -- "No matter he was frequently reminded"

-- Define the correctness condition
def correct_option : Options := Options.C

-- Define the proof problem
theorem correct_sentence_completion (opt : Options) : opt = correct_option :=
by sorry

end correct_sentence_completion_l1248_124875


namespace find_value_simplify_expression_l1248_124872

-- Define the first part of the problem
theorem find_value (α : ℝ) (h : Real.tan α = 1/3) : 
  (1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2)) = 2 / 3 := 
  sorry

-- Define the second part of the problem
theorem simplify_expression (α : ℝ) (h : Real.tan α = 1/3) : 
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / (Real.cos (-α - π) * Real.sin (-π - α)) = -1 := 
  sorry

end find_value_simplify_expression_l1248_124872


namespace cos_240_is_neg_half_l1248_124853

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l1248_124853


namespace solve_equation_l1248_124859

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔ x = -2 ∨ x = 1 / 3 :=
by
  sorry

end solve_equation_l1248_124859


namespace book_total_pages_l1248_124876

theorem book_total_pages (x : ℕ) (h1 : x * (3 / 5) * (3 / 8) = 36) : x = 120 := 
by
  -- Proof should be supplied here, but we only need the statement
  sorry

end book_total_pages_l1248_124876


namespace contradiction_example_l1248_124812

theorem contradiction_example (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2) : ¬ (a < 0 ∧ b < 0) :=
by
  -- The proof goes here, but we just need the statement
  sorry

end contradiction_example_l1248_124812


namespace even_function_symmetric_y_axis_l1248_124800

theorem even_function_symmetric_y_axis (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = f (-x) := by
  sorry

end even_function_symmetric_y_axis_l1248_124800


namespace xy_proposition_l1248_124828

theorem xy_proposition (x y : ℝ) : (x + y ≥ 5) → (x ≥ 3 ∨ y ≥ 2) :=
sorry

end xy_proposition_l1248_124828


namespace sara_grew_4_onions_l1248_124866

def onions_grown_by_sally : Nat := 5
def onions_grown_by_fred : Nat := 9
def total_onions_grown : Nat := 18

def onions_grown_by_sara : Nat :=
  total_onions_grown - (onions_grown_by_sally + onions_grown_by_fred)

theorem sara_grew_4_onions :
  onions_grown_by_sara = 4 :=
by
  sorry

end sara_grew_4_onions_l1248_124866


namespace area_of_shaded_region_l1248_124831

theorem area_of_shaded_region :
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  shaded_area = 22 :=
by
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  sorry

end area_of_shaded_region_l1248_124831


namespace fraction_halfway_between_l1248_124880

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l1248_124880


namespace zarnin_staffing_l1248_124834

open Finset

theorem zarnin_staffing :
  let total_resumes := 30
  let unsuitable_resumes := total_resumes / 3
  let suitable_resumes := total_resumes - unsuitable_resumes
  let positions := 5
  suitable_resumes = 20 → 
  positions = 5 → 
  Nat.factorial suitable_resumes / Nat.factorial (suitable_resumes - positions) = 930240 := by
  intro total_resumes unsuitable_resumes suitable_resumes positions h1 h2
  have hs : suitable_resumes = 20 := h1
  have hp : positions = 5 := h2
  sorry

end zarnin_staffing_l1248_124834


namespace total_monkeys_is_correct_l1248_124894

-- Define the parameters
variables (m n : ℕ)

-- Define the conditions as separate definitions
def monkeys_on_n_bicycles : ℕ := 3 * n
def monkeys_on_remaining_bicycles : ℕ := 5 * (m - n)

-- Define the total number of monkeys
def total_monkeys : ℕ := monkeys_on_n_bicycles n + monkeys_on_remaining_bicycles m n

-- State the theorem
theorem total_monkeys_is_correct : total_monkeys m n = 5 * m - 2 * n :=
by
  sorry

end total_monkeys_is_correct_l1248_124894


namespace find_m_value_l1248_124886

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ (x : ℝ), f x = 4 * x^2 - 3 * x + 5)
  (h2 : ∀ (x : ℝ), g x = 2 * x^2 - m * x + 8)
  (h3 : f 5 - g 5 = 15) :
  m = -17 / 5 :=
by
  sorry

end find_m_value_l1248_124886


namespace min_value_arith_geo_seq_l1248_124884

theorem min_value_arith_geo_seq (A B C D : ℕ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : 0 < D)
  (h_arith : C - B = B - A) (h_geo : C * C = B * D) (h_frac : 4 * C = 7 * B) :
  A + B + C + D = 97 :=
sorry

end min_value_arith_geo_seq_l1248_124884


namespace distance_from_sphere_center_to_triangle_plane_l1248_124822

theorem distance_from_sphere_center_to_triangle_plane :
  ∀ (O : ℝ × ℝ × ℝ) (r : ℝ) (a b c : ℝ), 
  r = 9 →
  a = 13 →
  b = 13 →
  c = 10 →
  (∀ (d : ℝ), d = distance_from_O_to_plane) →
  d = 8.36 :=
by
  intro O r a b c hr ha hb hc hd
  sorry

end distance_from_sphere_center_to_triangle_plane_l1248_124822


namespace rate_of_stream_l1248_124854

theorem rate_of_stream (v : ℝ) (h : 126 = (16 + v) * 6) : v = 5 :=
by 
  sorry

end rate_of_stream_l1248_124854


namespace compute_c_minus_d_cubed_l1248_124844

-- define c as the number of positive multiples of 12 less than 60
def c : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

-- define d as the number of positive integers less than 60 and a multiple of both 3 and 4
def d : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

theorem compute_c_minus_d_cubed : (c - d)^3 = 0 := by
  -- since c and d are computed the same way, (c - d) = 0
  -- hence, (c - d)^3 = 0^3 = 0
  sorry

end compute_c_minus_d_cubed_l1248_124844


namespace max_c_val_l1248_124803

theorem max_c_val (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : 2 * a * b = 2 * a + b) 
  (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 :=
sorry

end max_c_val_l1248_124803


namespace find_X_l1248_124847

theorem find_X (k : ℝ) (R1 R2 X1 X2 Y1 Y2 : ℝ) (h1 : R1 = k * (X1 / Y1)) (h2 : R1 = 10) (h3 : X1 = 2) (h4 : Y1 = 4) (h5 : R2 = 8) (h6 : Y2 = 5) : X2 = 2 :=
sorry

end find_X_l1248_124847


namespace volume_remaining_proof_l1248_124870

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end volume_remaining_proof_l1248_124870


namespace range_of_a_l1248_124871

open Set Real

def set_M (a : ℝ) : Set ℝ := { x | x * (x - a - 1) < 0 }
def set_N : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) : set_M a ⊆ set_N ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l1248_124871


namespace bus_speed_incl_stoppages_l1248_124806

theorem bus_speed_incl_stoppages (v_excl : ℝ) (minutes_stopped : ℝ) :
  v_excl = 64 → minutes_stopped = 13.125 →
  v_excl - (v_excl * (minutes_stopped / 60)) = 50 :=
by
  intro v_excl_eq minutes_stopped_eq
  rw [v_excl_eq, minutes_stopped_eq]
  have hours_stopped : ℝ := 13.125 / 60
  have distance_lost : ℝ := 64 * hours_stopped
  have v_incl := 64 - distance_lost
  sorry

end bus_speed_incl_stoppages_l1248_124806


namespace max_sum_cos_isosceles_triangle_l1248_124855

theorem max_sum_cos_isosceles_triangle :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ (2 * Real.cos α + Real.cos (π - 2 * α)) ≤ 1.5 :=
by
  sorry

end max_sum_cos_isosceles_triangle_l1248_124855


namespace angle_measure_of_three_times_complementary_l1248_124862

def is_complementary (α β : ℝ) : Prop := α + β = 90

def three_times_complement (α : ℝ) : Prop := 
  ∃ β : ℝ, is_complementary α β ∧ α = 3 * β

theorem angle_measure_of_three_times_complementary :
  ∀ α : ℝ, three_times_complement α → α = 67.5 :=
by sorry

end angle_measure_of_three_times_complementary_l1248_124862


namespace series_sum_l1248_124839

noncomputable def S (n : ℕ) : ℝ := 2^(n + 1) + n - 2

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem series_sum : 
  ∑' i, a i / 4^i = 4 / 3 :=
by 
  sorry

end series_sum_l1248_124839


namespace total_pears_l1248_124851

theorem total_pears (Alyssa_picked Nancy_picked : ℕ) (h₁ : Alyssa_picked = 42) (h₂ : Nancy_picked = 17) : Alyssa_picked + Nancy_picked = 59 :=
by
  sorry

end total_pears_l1248_124851


namespace max_value_of_x2_plus_y2_l1248_124811

open Real

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x - 2 * y + 2) : 
  x^2 + y^2 ≤ 6 + 4 * sqrt 2 :=
sorry

end max_value_of_x2_plus_y2_l1248_124811


namespace five_letter_words_with_one_consonant_l1248_124891

theorem five_letter_words_with_one_consonant :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E']
  let consonants := ['B', 'C', 'D', 'F']
  let total_words := (letters.length : ℕ)^5
  let vowel_only_words := (vowels.length : ℕ)^5
  total_words - vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_one_consonant_l1248_124891


namespace sum_first_9_terms_l1248_124815

noncomputable def sum_of_first_n_terms (a1 d : Int) (n : Int) : Int :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_9_terms (a1 d : ℤ) 
  (h1 : a1 + (a1 + 3 * d) + (a1 + 6 * d) = 39)
  (h2 : (a1 + 2 * d) + (a1 + 5 * d) + (a1 + 8 * d) = 27) :
  sum_of_first_n_terms a1 d 9 = 99 := by
  sorry

end sum_first_9_terms_l1248_124815


namespace number_of_buses_required_l1248_124888

def total_seats : ℕ := 28
def students_per_bus : ℝ := 14.0

theorem number_of_buses_required :
  (total_seats / students_per_bus) = 2 := 
by
  -- The actual proof is intentionally left out.
  sorry

end number_of_buses_required_l1248_124888


namespace decimal_to_base13_185_l1248_124898

theorem decimal_to_base13_185 : 
  ∀ n : ℕ, n = 185 → 
      ∃ a b c : ℕ, a * 13^2 + b * 13 + c = n ∧ 0 ≤ a ∧ a < 13 ∧ 0 ≤ b ∧ b < 13 ∧ 0 ≤ c ∧ c < 13 ∧ (a, b, c) = (1, 1, 3) := 
by
  intros n hn
  use 1, 1, 3
  sorry

end decimal_to_base13_185_l1248_124898


namespace largest_y_coordinate_on_graph_l1248_124807

theorem largest_y_coordinate_on_graph :
  ∀ x y : ℝ, (x / 7) ^ 2 + ((y - 3) / 5) ^ 2 = 0 → y ≤ 3 := 
by
  intro x y h
  sorry

end largest_y_coordinate_on_graph_l1248_124807


namespace serenity_total_shoes_l1248_124865

def pairs_of_shoes : ℕ := 3
def shoes_per_pair : ℕ := 2

theorem serenity_total_shoes : pairs_of_shoes * shoes_per_pair = 6 := by
  sorry

end serenity_total_shoes_l1248_124865


namespace average_of_all_digits_l1248_124852

theorem average_of_all_digits {a b : ℕ} (n : ℕ) (x y : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : n = 10) (h4 : x = 58) (h5 : y = 113) :
  ((a * x + b * y) / n = 80) :=
  sorry

end average_of_all_digits_l1248_124852


namespace x_axis_line_l1248_124885

variable (A B C : ℝ)

theorem x_axis_line (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : B ≠ 0 ∧ A = 0 ∧ C = 0 := by
  sorry

end x_axis_line_l1248_124885


namespace not_square_of_expression_l1248_124883

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ¬ ∃ m : ℤ, m * m = 2 * n * n + 2 - n :=
by
  sorry

end not_square_of_expression_l1248_124883


namespace sum_of_digits_l1248_124860

theorem sum_of_digits (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h3 : 34 * a + 42 * b = 142) : a + b = 4 := 
by
  sorry

end sum_of_digits_l1248_124860


namespace smallest_k_674_l1248_124890

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l1248_124890


namespace find_investment_period_l1248_124877

variable (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)

theorem find_investment_period (hP : P = 12000)
                               (hr : r = 0.10)
                               (hn : n = 2)
                               (hA : A = 13230) :
                               ∃ t : ℝ, A = P * (1 + r / n)^(n * t) ∧ t = 1 := 
by
  sorry

end find_investment_period_l1248_124877


namespace quilt_cost_l1248_124818

theorem quilt_cost :
  let length := 7
  let width := 8
  let cost_per_sq_ft := 40
  let area := length * width
  let total_cost := area * cost_per_sq_ft
  total_cost = 2240 :=
by
  sorry

end quilt_cost_l1248_124818


namespace cities_with_fewer_than_500000_residents_l1248_124858

theorem cities_with_fewer_than_500000_residents (P Q R : ℕ) 
  (h1 : P + Q + R = 100) 
  (h2 : P = 40) 
  (h3 : Q = 35) 
  (h4 : R = 25) : P + Q = 75 :=
by 
  sorry

end cities_with_fewer_than_500000_residents_l1248_124858


namespace elephants_ratio_l1248_124813

theorem elephants_ratio (x : ℝ) (w : ℝ) (g : ℝ) (total : ℝ) :
  w = 70 →
  total = 280 →
  g = x * w →
  w + g = total →
  x = 3 :=
by 
  intros h1 h2 h3 h4
  sorry

end elephants_ratio_l1248_124813


namespace problem_condition_l1248_124823

variable (x y z : ℝ)

theorem problem_condition (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end problem_condition_l1248_124823


namespace Jane_age_l1248_124874

theorem Jane_age (J A : ℕ) (h1 : J + A = 54) (h2 : J - A = 22) : A = 16 := 
by 
  sorry

end Jane_age_l1248_124874


namespace factory_workers_l1248_124867

-- Define parameters based on given conditions
def sewing_factory_x : ℤ := 1995
def shoe_factory_y : ℤ := 1575

-- Conditions based on the problem setup
def shoe_factory_of_sewing_factory := (15 * sewing_factory_x) / 19 = shoe_factory_y
def shoe_factory_plan_exceed := (3 * shoe_factory_y) / 7 < 1000
def sewing_factory_plan_exceed := (3 * sewing_factory_x) / 5 > 1000

-- Theorem stating the problem's assertion
theorem factory_workers (x y : ℤ) 
  (h1 : (15 * x) / 19 = y)
  (h2 : (4 * y) / 7 < 1000)
  (h3 : (3 * x) / 5 > 1000) : 
  x = 1995 ∧ y = 1575 :=
sorry

end factory_workers_l1248_124867


namespace closest_vector_l1248_124857

theorem closest_vector 
  (s : ℝ)
  (u b d : ℝ × ℝ × ℝ)
  (h₁ : u = (3, -2, 4) + s • (6, 4, 2))
  (h₂ : b = (1, 7, 6))
  (hdir : d = (6, 4, 2))
  (h₃ : (u - b) = (2 + 6 * s, -9 + 4 * s, -2 + 2 * s)) :
  ((2 + 6 * s) * 6 + (-9 + 4 * s) * 4 + (-2 + 2 * s) * 2) = 0 →
  s = 1 / 2 :=
by
  -- Skipping the proof, adding sorry
  sorry

end closest_vector_l1248_124857


namespace ants_total_l1248_124809

namespace Ants

-- Defining the number of ants each child finds based on the given conditions
def Abe_ants := 4
def Beth_ants := Abe_ants + Abe_ants
def CeCe_ants := 3 * Abe_ants
def Duke_ants := Abe_ants / 2
def Emily_ants := Abe_ants + (3 * Abe_ants / 4)
def Frances_ants := 2 * CeCe_ants

-- The total number of ants found by the six children
def total_ants := Abe_ants + Beth_ants + CeCe_ants + Duke_ants + Emily_ants + Frances_ants

-- The statement to prove
theorem ants_total: total_ants = 57 := by
  sorry

end Ants

end ants_total_l1248_124809


namespace trig_identity_l1248_124842

theorem trig_identity 
  (α : ℝ) 
  (h : Real.tan α = 1 / 3) : 
  Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) = 3 / 10 :=
sorry

end trig_identity_l1248_124842


namespace product_probability_probability_one_l1248_124868

def S : Set Int := {13, 57}

theorem product_probability (a b : Int) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : a ≠ b) : 
  (a * b > 15) := 
by 
  sorry

theorem probability_one : 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b > 15) ∧ 
  (∀ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b → a * b > 15) :=
by 
  sorry

end product_probability_probability_one_l1248_124868


namespace more_red_peaches_than_green_l1248_124896

-- Given conditions
def red_peaches : Nat := 17
def green_peaches : Nat := 16

-- Statement to prove
theorem more_red_peaches_than_green : red_peaches - green_peaches = 1 :=
by
  sorry

end more_red_peaches_than_green_l1248_124896


namespace max_x_minus_y_l1248_124833

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x^2 + y) :
  x - y ≤ 1 / Real.sqrt 24 :=
sorry

end max_x_minus_y_l1248_124833


namespace smallest_sum_of_squares_l1248_124821

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 217) : 
  x^2 + y^2 ≥ 505 :=
sorry

end smallest_sum_of_squares_l1248_124821


namespace correct_power_functions_l1248_124819

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, x ≠ 0 → f x = k * x^n

def f1 (x : ℝ) : ℝ := x^2 + 2
def f2 (x : ℝ) : ℝ := x^(1 / 2)
def f3 (x : ℝ) : ℝ := 2 * x^3
def f4 (x : ℝ) : ℝ := x^(3 / 4)
def f5 (x : ℝ) : ℝ := x^(1 / 3) + 1

theorem correct_power_functions :
  {f2, f4} = {f : ℝ → ℝ | is_power_function f} ∩ {f2, f4, f1, f3, f5} :=
by
  sorry

end correct_power_functions_l1248_124819


namespace correct_order_l1248_124861

noncomputable def f : ℝ → ℝ := sorry

axiom periodic : ∀ x : ℝ, f (x + 4) = f x
axiom increasing : ∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ < 2) → (0 ≤ x₂ ∧ x₂ ≤ 2) → x₁ < x₂ → f x₁ < f x₂
axiom symmetric : ∀ x : ℝ, f (x + 2) = f (2 - x)

theorem correct_order : f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  sorry

end correct_order_l1248_124861


namespace intersection_of_M_and_N_l1248_124845

-- Defining our sets M and N based on the conditions provided
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x^2 < 4 }

-- The statement we want to prove
theorem intersection_of_M_and_N :
  M ∩ N = { x | -2 < x ∧ x < 1 } :=
sorry

end intersection_of_M_and_N_l1248_124845


namespace molly_bike_miles_l1248_124882

def total_miles_ridden (daily_miles years_riding days_per_year : ℕ) : ℕ :=
  daily_miles * years_riding * days_per_year

theorem molly_bike_miles :
  total_miles_ridden 3 3 365 = 3285 :=
by
  -- The definition and theorem are provided; the implementation will be done by the prover.
  sorry

end molly_bike_miles_l1248_124882


namespace factory_production_l1248_124843

theorem factory_production (y x : ℝ) (h1 : y + 40 * x = 1.2 * y) (h2 : y + 0.6 * y * x = 2.5 * y) 
  (hx : x = 2.5) : y = 500 ∧ 1 + x = 3.5 :=
by
  sorry

end factory_production_l1248_124843


namespace apple_bags_l1248_124832

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l1248_124832


namespace square_area_l1248_124856

theorem square_area (perimeter : ℝ) (h : perimeter = 32) : 
  ∃ (side area : ℝ), side = perimeter / 4 ∧ area = side * side ∧ area = 64 := 
by
  sorry

end square_area_l1248_124856


namespace alpha_half_quadrant_l1248_124879

theorem alpha_half_quadrant (k : ℤ) (α : ℝ)
  (h : 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi) :
  (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < 2 * n * Real.pi) ∨
  (∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < (2 * n + 1) * Real.pi) :=
sorry

end alpha_half_quadrant_l1248_124879


namespace third_consecutive_odd_integers_is_fifteen_l1248_124814

theorem third_consecutive_odd_integers_is_fifteen :
  ∃ x : ℤ, (x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) ∧ (x + 2 + (x + 4) = x + 17) → (x + 4 = 15) :=
by
  sorry

end third_consecutive_odd_integers_is_fifteen_l1248_124814


namespace geometric_sequence_third_term_l1248_124827

theorem geometric_sequence_third_term (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := 
sorry

end geometric_sequence_third_term_l1248_124827


namespace malcolm_walked_uphill_l1248_124829

-- Define the conditions as variables and parameters
variables (x : ℕ)

-- Define the conditions given in the problem
def first_route_time := x + 2 * x + x
def second_route_time := 14 + 28
def time_difference := 18

-- Theorem statement - proving that Malcolm walked uphill for 6 minutes in the first route
theorem malcolm_walked_uphill : first_route_time - second_route_time = time_difference → x = 6 := by
  sorry

end malcolm_walked_uphill_l1248_124829


namespace find_n_l1248_124889

theorem find_n (n : ℕ) (h : n * n.factorial + n.factorial = 720) : n = 5 :=
sorry

end find_n_l1248_124889


namespace total_reduction_500_l1248_124837

noncomputable def total_price_reduction (P : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : ℝ :=
  let first_reduction := P * first_reduction_percent / 100
  let intermediate_price := P - first_reduction
  let second_reduction := intermediate_price * second_reduction_percent / 100
  let final_price := intermediate_price - second_reduction
  P - final_price

theorem total_reduction_500 (P : ℝ) (first_reduction_percent : ℝ)  (second_reduction_percent: ℝ) (h₁ : P = 500) (h₂ : first_reduction_percent = 5) (h₃ : second_reduction_percent = 4):
  total_price_reduction P first_reduction_percent second_reduction_percent = 44 := 
by
  sorry

end total_reduction_500_l1248_124837


namespace edge_length_of_box_l1248_124816

noncomputable def edge_length_cubical_box (num_cubes : ℕ) (edge_length_cube : ℝ) : ℝ :=
  if num_cubes = 8 ∧ edge_length_cube = 0.5 then -- 50 cm in meters
    1 -- The edge length of the cubical box in meters
  else
    0 -- Placeholder for other cases

theorem edge_length_of_box :
  edge_length_cubical_box 8 0.5 = 1 :=
sorry

end edge_length_of_box_l1248_124816


namespace ratio_of_other_triangle_to_square_l1248_124826

noncomputable def ratio_of_triangle_areas (m : ℝ) : ℝ :=
  let side_of_square := 2
  let area_of_square := side_of_square ^ 2
  let area_of_smaller_triangle := m * area_of_square
  let r := area_of_smaller_triangle / (side_of_square / 2)
  let s := side_of_square * side_of_square / r
  let area_of_other_triangle := side_of_square * s / 2
  area_of_other_triangle / area_of_square

theorem ratio_of_other_triangle_to_square (m : ℝ) (h : m > 0) :
  ratio_of_triangle_areas m = 1 / (4 * m) :=
sorry

end ratio_of_other_triangle_to_square_l1248_124826


namespace similar_triangle_perimeter_l1248_124805

theorem similar_triangle_perimeter :
  ∀ (a b c : ℝ), a = 7 ∧ b = 7 ∧ c = 12 →
  ∀ (d : ℝ), d = 30 →
  ∃ (p : ℝ), p = 65 ∧ 
  (∃ a' b' c' : ℝ, (a' = 17.5 ∧ b' = 17.5 ∧ c' = d) ∧ p = a' + b' + c') :=
by sorry

end similar_triangle_perimeter_l1248_124805


namespace exists_infinite_arith_prog_exceeding_M_l1248_124808

def sum_of_digits(n : ℕ) : ℕ :=
n.digits 10 |> List.sum

theorem exists_infinite_arith_prog_exceeding_M (M : ℝ) :
  ∃ (a d : ℕ), ¬ (10 ∣ d) ∧ (∀ n : ℕ, a + n * d > 0) ∧ (∀ n : ℕ, sum_of_digits (a + n * d) > M) := by
sorry

end exists_infinite_arith_prog_exceeding_M_l1248_124808


namespace product_of_roots_l1248_124838

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end product_of_roots_l1248_124838


namespace triangle_max_area_proof_l1248_124841

open Real

noncomputable def triangle_max_area (A B C : ℝ) (AB : ℝ) (tanA tanB : ℝ) : Prop :=
  AB = 4 ∧ tanA * tanB = 3 / 4 → ∃ S : ℝ, S = 2 * sqrt 3

theorem triangle_max_area_proof (A B C : ℝ) (tanA tanB : ℝ) (AB : ℝ) : 
  triangle_max_area A B C AB tanA tanB :=
by
  sorry

end triangle_max_area_proof_l1248_124841


namespace max_obtuse_in_convex_quadrilateral_l1248_124850

-- Definition and problem statement
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

theorem max_obtuse_in_convex_quadrilateral (a b c d : ℝ) :
  convex_quadrilateral a b c d →
  (is_obtuse a → (is_obtuse b → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse b → (is_obtuse a → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse c → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse d))) →
  (is_obtuse d → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse c))) :=
by
  intros h_convex h1 h2 h3 h4
  sorry

end max_obtuse_in_convex_quadrilateral_l1248_124850


namespace difference_in_soda_bottles_l1248_124869

def diet_soda_bottles : ℕ := 4
def regular_soda_bottles : ℕ := 83

theorem difference_in_soda_bottles :
  regular_soda_bottles - diet_soda_bottles = 79 :=
by
  sorry

end difference_in_soda_bottles_l1248_124869


namespace house_prices_and_yields_l1248_124801

theorem house_prices_and_yields :
  ∃ x y : ℝ, 
    (425 = (y / 100) * x) ∧ 
    (459 = ((y - 0.5) / 100) * (6/5) * x) ∧ 
    (x = 8500) ∧ 
    (y = 5) ∧ 
    ((6/5) * x = 10200) ∧ 
    (y - 0.5 = 4.5) :=
by
  sorry

end house_prices_and_yields_l1248_124801


namespace time_for_A_l1248_124863

-- Given rates of pipes A, B, and C filling the tank
variable (A B C : ℝ)

-- Condition 1: Tank filled by all three pipes in 8 hours
def combined_rate := (A + B + C = 1/8)

-- Condition 2: Pipe C is twice as fast as B
def rate_C := (C = 2 * B)

-- Condition 3: Pipe B is twice as fast as A
def rate_B := (B = 2 * A)

-- Question: To prove that pipe A alone will take 56 hours to fill the tank
theorem time_for_A (h₁ : combined_rate A B C) (h₂ : rate_C B C) (h₃ : rate_B A B) : 
  1 / A = 56 :=
by {
  sorry
}

end time_for_A_l1248_124863


namespace right_triangle_hypotenuse_length_l1248_124848

theorem right_triangle_hypotenuse_length (a b c : ℝ) (h₀ : a = 7) (h₁ : b = 24) (h₂ : a^2 + b^2 = c^2) : c = 25 :=
by
  rw [h₀, h₁] at h₂
  -- This step will simplify the problem
  sorry

end right_triangle_hypotenuse_length_l1248_124848


namespace polynomial_simplification_l1248_124820

theorem polynomial_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := 
by
  sorry

end polynomial_simplification_l1248_124820


namespace triangle_relation_l1248_124849

theorem triangle_relation (A B C a b : ℝ) (h : 4 * A = B ∧ B = C) (hABC : A + B + C = 180) : 
  a^3 + b^3 = 3 * a * b^2 := 
by 
  sorry

end triangle_relation_l1248_124849


namespace more_uniform_team_l1248_124824

-- Define the parameters and the variances
def average_height := 1.85
def variance_team_A := 0.32
def variance_team_B := 0.26

-- Main theorem statement
theorem more_uniform_team : variance_team_B < variance_team_A → "Team B" = "Team with more uniform heights" :=
by
  -- Placeholder for the actual proof
  sorry

end more_uniform_team_l1248_124824


namespace area_of_quadrilateral_PQRS_l1248_124892

noncomputable def calculate_area_of_quadrilateral_PQRS (PQ PR : ℝ) (PS_corrected : ℝ) : ℝ :=
  let area_ΔPQR := (1/2) * PQ * PR
  let RS := Real.sqrt (PR^2 - PQ^2)
  let area_ΔPRS := (1/2) * PR * RS
  area_ΔPQR + area_ΔPRS

theorem area_of_quadrilateral_PQRS :
  let PQ := 8
  let PR := 10
  let PS_corrected := Real.sqrt (PQ^2 + PR^2)
  calculate_area_of_quadrilateral_PQRS PQ PR PS_corrected = 70 := 
by
  sorry

end area_of_quadrilateral_PQRS_l1248_124892


namespace balance_balls_l1248_124846

variables (R B O P : ℝ)

-- Conditions
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 6 * B
axiom h3 : 8 * B = 6 * P

-- Proof problem
theorem balance_balls : 5 * R + 3 * O + 3 * P = 20 * B :=
by sorry

end balance_balls_l1248_124846
