import Mathlib

namespace equivalence_of_negation_l1519_151923

-- Define the statement for the negation
def negation_stmt := ¬ ∃ x0 : ℝ, x0 ≤ 0 ∧ x0^2 ≥ 0

-- Define the equivalent statement after negation
def equivalent_stmt := ∀ x : ℝ, x ≤ 0 → x^2 < 0

-- The theorem stating that the negation_stmt is equivalent to equivalent_stmt
theorem equivalence_of_negation : negation_stmt ↔ equivalent_stmt := 
sorry

end equivalence_of_negation_l1519_151923


namespace sin_alpha_sub_beta_cos_beta_l1519_151955

variables (α β : ℝ)
variables (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
variables (h1 : Real.sin α = 3 / 5)
variables (h2 : Real.tan (α - β) = -1 / 3)

theorem sin_alpha_sub_beta : Real.sin (α - β) = - Real.sqrt 10 / 10 :=
by
  sorry

theorem cos_beta : Real.cos β = 9 * Real.sqrt 10 / 50 :=
by
  sorry

end sin_alpha_sub_beta_cos_beta_l1519_151955


namespace james_prom_total_cost_l1519_151958

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end james_prom_total_cost_l1519_151958


namespace inequality_true_l1519_151919

variable (a b : ℝ)

theorem inequality_true (h : a > b ∧ b > 0) : (b^2 / a) < (a^2 / b) := by
  sorry

end inequality_true_l1519_151919


namespace cost_of_baking_soda_l1519_151940

-- Definitions of the condition
def students : ℕ := 23
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def total_cost_of_supplies : ℕ := 184

-- Main statement to prove
theorem cost_of_baking_soda : 
  (∀ (students : ℕ) (cost_of_bow : ℕ) (cost_of_vinegar : ℕ) (total_cost_of_supplies : ℕ),
    total_cost_of_supplies = students * (cost_of_bow + cost_of_vinegar) + students) → 
  total_cost_of_supplies = 23 * (5 + 2) + 23 → 
  184 = 23 * (5 + 2 + 1) :=
by
  sorry

end cost_of_baking_soda_l1519_151940


namespace multiply_63_57_l1519_151951

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l1519_151951


namespace chocolate_oranges_initial_l1519_151992

theorem chocolate_oranges_initial (p_c p_o G n_c x : ℕ) 
  (h_candy_bar_price : p_c = 5) 
  (h_orange_price : p_o = 10) 
  (h_goal : G = 1000) 
  (h_candy_bars_sold : n_c = 160) 
  (h_equation : G = p_o * x + p_c * n_c) : 
  x = 20 := 
by
  sorry

end chocolate_oranges_initial_l1519_151992


namespace min_value_a_2b_l1519_151900

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 :=
sorry

end min_value_a_2b_l1519_151900


namespace symmetric_line_eq_l1519_151946

theorem symmetric_line_eq (x y : ℝ) :  
  (x - 2 * y + 3 = 0) → (x + 2 * y + 3 = 0) :=
sorry

end symmetric_line_eq_l1519_151946


namespace solution_set_of_inequality_system_l1519_151967

theorem solution_set_of_inequality_system (x : ℝ) : 
  (x + 5 < 4) ∧ (3 * x + 1 ≥ 2 * (2 * x - 1)) ↔ (x < -1) :=
  by
  sorry

end solution_set_of_inequality_system_l1519_151967


namespace no_square_with_odd_last_two_digits_l1519_151969

def last_two_digits_odd (n : ℤ) : Prop :=
  (n % 10) % 2 = 1 ∧ ((n / 10) % 10) % 2 = 1

theorem no_square_with_odd_last_two_digits (n : ℤ) (k : ℤ) :
  (k^2 = n) → last_two_digits_odd n → False :=
by
  -- A placeholder for the proof
  sorry

end no_square_with_odd_last_two_digits_l1519_151969


namespace quoted_value_of_stock_l1519_151912

theorem quoted_value_of_stock (F P : ℝ) (h1 : F > 0) (h2 : P = 1.25 * F) : 
  (0.10 * F) / P = 0.08 := 
sorry

end quoted_value_of_stock_l1519_151912


namespace least_whole_number_for_ratio_l1519_151925

theorem least_whole_number_for_ratio :
  ∃ x : ℕ, (6 - x) * 21 < (7 - x) * 16 ∧ x = 3 :=
by
  sorry

end least_whole_number_for_ratio_l1519_151925


namespace product_area_perimeter_square_EFGH_l1519_151998

theorem product_area_perimeter_square_EFGH:
  let E := (5, 5)
  let F := (5, 1)
  let G := (1, 1)
  let H := (1, 5)
  let side_length := 4
  let area := side_length * side_length
  let perimeter := 4 * side_length
  area * perimeter = 256 :=
by
  sorry

end product_area_perimeter_square_EFGH_l1519_151998


namespace min_x_plus_y_l1519_151995

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
by
  sorry

end min_x_plus_y_l1519_151995


namespace x_squared_eq_r_floor_x_has_2_or_3_solutions_l1519_151928

theorem x_squared_eq_r_floor_x_has_2_or_3_solutions (r : ℝ) (hr : r > 2) : 
  ∃! (s : Finset ℝ), s.card = 2 ∨ s.card = 3 ∧ ∀ x ∈ s, x^2 = r * (⌊x⌋) :=
by
  sorry

end x_squared_eq_r_floor_x_has_2_or_3_solutions_l1519_151928


namespace sequence_formula_l1519_151931

noncomputable def a : ℕ → ℕ
| 0       => 2
| (n + 1) => a n ^ 2 - n * a n + 1

theorem sequence_formula (n : ℕ) : a n = n + 2 :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end sequence_formula_l1519_151931


namespace distinct_integers_are_squares_l1519_151905

theorem distinct_integers_are_squares
  (n : ℕ) 
  (h_n : n = 2000) 
  (x : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → x i ≠ x j)
  (h_product_square : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ∃ (m : ℕ), x i * x j * x k = m^2) :
  ∀ i : Fin n, ∃ (m : ℕ), x i = m^2 := 
sorry

end distinct_integers_are_squares_l1519_151905


namespace distinct_patterns_4x4_three_squares_l1519_151999

noncomputable def count_distinct_patterns : ℕ :=
  sorry

theorem distinct_patterns_4x4_three_squares :
  count_distinct_patterns = 12 :=
by sorry

end distinct_patterns_4x4_three_squares_l1519_151999


namespace rectangle_area_increase_l1519_151917

theorem rectangle_area_increase (x y : ℕ) 
  (hxy : x * y = 180) 
  (hperimeter : 2 * x + 2 * y = 54) : 
  (x + 6) * (y + 6) = 378 :=
by sorry

end rectangle_area_increase_l1519_151917


namespace anne_carries_total_weight_l1519_151945

-- Definitions for the conditions
def weight_female_cat : ℕ := 2
def weight_male_cat : ℕ := 2 * weight_female_cat

-- Problem statement
theorem anne_carries_total_weight : weight_female_cat + weight_male_cat = 6 :=
by
  sorry

end anne_carries_total_weight_l1519_151945


namespace identify_genuine_coins_l1519_151978

section IdentifyGenuineCoins

variables (coins : Fin 25 → ℝ) 
          (is_genuine : Fin 25 → Prop) 
          (is_counterfeit : Fin 25 → Prop)

-- Conditions
axiom coin_total : ∀ i, is_genuine i ∨ is_counterfeit i
axiom genuine_count : ∃ s : Finset (Fin 25), s.card = 22 ∧ ∀ i ∈ s, is_genuine i
axiom counterfeit_count : ∃ t : Finset (Fin 25), t.card = 3 ∧ ∀ i ∈ t, is_counterfeit i
axiom genuine_weight : ∃ w : ℝ, ∀ i, is_genuine i → coins i = w
axiom counterfeit_weight : ∃ c : ℝ, ∀ i, is_counterfeit i → coins i = c
axiom counterfeit_lighter : ∀ (w c : ℝ), (∃ i, is_genuine i → coins i = w) ∧ (∃ j, is_counterfeit j → coins j = c) → c < w

-- Theorem: Identifying 6 genuine coins using two weighings
theorem identify_genuine_coins : ∃ s : Finset (Fin 25), s.card = 6 ∧ ∀ i ∈ s, is_genuine i :=
sorry

end IdentifyGenuineCoins

end identify_genuine_coins_l1519_151978


namespace arithmetic_contains_geometric_l1519_151987

theorem arithmetic_contains_geometric (a b : ℚ) (h : a^2 + b^2 ≠ 0) :
  ∃ (q : ℚ) (c : ℚ) (n₀ : ℕ) (n : ℕ → ℕ), (∀ k : ℕ, n (k+1) = n k + c * q^k) ∧
  ∀ k : ℕ, ∃ r : ℚ, a + b * n k = r * q^k :=
sorry

end arithmetic_contains_geometric_l1519_151987


namespace initial_numbers_unique_l1519_151968

theorem initial_numbers_unique 
  (A B C A' B' C' : ℕ) 
  (h1: 1 ≤ A ∧ A ≤ 50) 
  (h2: 1 ≤ B ∧ B ≤ 50) 
  (h3: 1 ≤ C ∧ C ≤ 50) 
  (final_ana : 104 = 2 * A + B + C)
  (final_beto : 123 = A + 2 * B + C)
  (final_caio : 137 = A + B + 2 * C) : 
  A = 13 ∧ B = 32 ∧ C = 46 :=
sorry

end initial_numbers_unique_l1519_151968


namespace Juan_has_498_marbles_l1519_151963

def ConnieMarbles : Nat := 323
def JuanMoreMarbles : Nat := 175
def JuanMarbles : Nat := ConnieMarbles + JuanMoreMarbles

theorem Juan_has_498_marbles : JuanMarbles = 498 := by
  sorry

end Juan_has_498_marbles_l1519_151963


namespace number_of_steaks_needed_l1519_151941

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end number_of_steaks_needed_l1519_151941


namespace find_e_l1519_151920

theorem find_e (d e f : ℝ) (h1 : f = 5)
  (h2 : -d / 3 = -f)
  (h3 : -f = 1 + d + e + f) :
  e = -26 := 
by
  sorry

end find_e_l1519_151920


namespace necklace_cost_l1519_151970

def bead_necklaces := 3
def gemstone_necklaces := 3
def total_necklaces := bead_necklaces + gemstone_necklaces
def total_earnings := 36

theorem necklace_cost :
  (total_earnings / total_necklaces) = 6 :=
by
  -- Proof goes here
  sorry

end necklace_cost_l1519_151970


namespace plane_stops_at_20_seconds_l1519_151914

/-- The analytical expression of the function of the distance s the plane travels during taxiing 
after landing with respect to the time t is given by s = -1.5t^2 + 60t. 

Prove that the plane stops after taxiing for 20 seconds. -/

noncomputable def plane_distance (t : ℝ) : ℝ :=
  -1.5 * t^2 + 60 * t

theorem plane_stops_at_20_seconds :
  ∃ t : ℝ, t = 20 ∧ plane_distance t = plane_distance (20 : ℝ) :=
by
  sorry

end plane_stops_at_20_seconds_l1519_151914


namespace fraction_equality_l1519_151981

theorem fraction_equality (x : ℝ) :
  (4 + 2 * x) / (7 + 3 * x) = (2 + 3 * x) / (4 + 5 * x) ↔ x = -1 ∨ x = -2 := by
  sorry

end fraction_equality_l1519_151981


namespace total_subjects_l1519_151962

theorem total_subjects (subjects_monica subjects_marius subjects_millie : ℕ)
  (h1 : subjects_monica = 10)
  (h2 : subjects_marius = subjects_monica + 4)
  (h3 : subjects_millie = subjects_marius + 3) :
  subjects_monica + subjects_marius + subjects_millie = 41 :=
by
  sorry

end total_subjects_l1519_151962


namespace Sue_chewing_gums_count_l1519_151902

theorem Sue_chewing_gums_count (S : ℕ) 
  (hMary : 5 = 5) 
  (hSam : 10 = 10) 
  (hTotal : 5 + 10 + S = 30) : S = 15 := 
by {
  sorry
}

end Sue_chewing_gums_count_l1519_151902


namespace compare_f_minus1_f_1_l1519_151996

variable (f : ℝ → ℝ)

-- Given conditions
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x : ℝ, f x = x^2 + 2 * x * (f 2 - 2 * x))

-- Goal statement
theorem compare_f_minus1_f_1 : f (-1) > f 1 :=
by sorry

end compare_f_minus1_f_1_l1519_151996


namespace xy_product_l1519_151929

-- Define the proof problem with the conditions and required statement
theorem xy_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy_distinct : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 := 
  sorry

end xy_product_l1519_151929


namespace table_tennis_possible_outcomes_l1519_151952

-- Two people are playing a table tennis match. The first to win 3 games wins the match.
-- The match continues until a winner is determined.
-- Considering all possible outcomes (different numbers of wins and losses for each player are considered different outcomes),
-- prove that there are a total of 30 possible outcomes.

theorem table_tennis_possible_outcomes : 
  ∃ total_outcomes : ℕ, total_outcomes = 30 := 
by
  -- We need to prove that the total number of possible outcomes is 30
  sorry

end table_tennis_possible_outcomes_l1519_151952


namespace geometric_sequence_ratio_l1519_151960

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n+1) = q * a n)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 :=
sorry

end geometric_sequence_ratio_l1519_151960


namespace geometric_sequence_sum_l1519_151910

theorem geometric_sequence_sum {a : ℕ → ℝ} (h : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) 
  (h_cond : (1 / (a 2 * a 4)) + (2 / (a 4 * a 4)) + (1 / (a 4 * a 6)) = 81) :
  (1 / a 3) + (1 / a 5) = 9 :=
sorry

end geometric_sequence_sum_l1519_151910


namespace cricket_count_l1519_151921

theorem cricket_count (x : ℕ) (h : x + 11 = 18) : x = 7 :=
by sorry

end cricket_count_l1519_151921


namespace arithmetic_sequence_geometric_property_l1519_151949

theorem arithmetic_sequence_geometric_property (a : ℕ → ℤ) (d : ℤ) (h_d : d = 2)
  (h_a3 : a 3 = a 1 + 4) (h_a4 : a 4 = a 1 + 6)
  (geo_seq : (a 1 + 4) * (a 1 + 4) = a 1 * (a 1 + 6)) :
  a 2 = -6 := sorry

end arithmetic_sequence_geometric_property_l1519_151949


namespace final_price_lower_than_budget_l1519_151944

theorem final_price_lower_than_budget :
  let budget := 1500
  let T := 750 -- budget equally split for TV
  let S := 750 -- budget equally split for Sound System
  let TV_price_with_discount := (T - 150) * 0.80
  let SoundSystem_price_with_discount := S * 0.85
  let combined_price_before_tax := TV_price_with_discount + SoundSystem_price_with_discount
  let final_price_with_tax := combined_price_before_tax * 1.08
  budget - final_price_with_tax = 293.10 :=
by
  sorry

end final_price_lower_than_budget_l1519_151944


namespace pufferfish_count_l1519_151937

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end pufferfish_count_l1519_151937


namespace indeterminate_4wheelers_l1519_151961

-- Define conditions and the main theorem to state that the number of 4-wheelers cannot be uniquely determined.
theorem indeterminate_4wheelers (x y : ℕ) (h : 2 * x + 4 * y = 58) : ∃ k : ℤ, y = ((29 : ℤ) - k - x) / 2 :=
by
  sorry

end indeterminate_4wheelers_l1519_151961


namespace necessarily_positive_l1519_151932

-- Conditions
variables {x y z : ℝ}

-- Statement to prove
theorem necessarily_positive (h1 : 0 < x) (h2 : x < 1) (h3 : -2 < y) (h4 : y < 0) (h5 : 2 < z) (h6 : z < 3) :
  0 < y + 2 * z :=
sorry

end necessarily_positive_l1519_151932


namespace least_number_of_pairs_l1519_151918

theorem least_number_of_pairs :
  let students := 100
  let messages_per_student := 50
  ∃ (pairs_of_students : ℕ), pairs_of_students = 50 := sorry

end least_number_of_pairs_l1519_151918


namespace astronaut_days_on_orbius_l1519_151904

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end astronaut_days_on_orbius_l1519_151904


namespace fraction_comparison_l1519_151994

theorem fraction_comparison : 
  (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) :=
sorry

end fraction_comparison_l1519_151994


namespace consecutive_weights_sum_to_63_l1519_151972

theorem consecutive_weights_sum_to_63 : ∃ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 63 :=
by
  sorry

end consecutive_weights_sum_to_63_l1519_151972


namespace school_competition_students_l1519_151975

theorem school_competition_students (n : ℤ)
  (h1 : 100 < n) 
  (h2 : n < 200) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2) 
  (h5 : n % 6 = 2) :
  n = 122 ∨ n = 182 :=
sorry

end school_competition_students_l1519_151975


namespace normal_line_eq_l1519_151965

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem normal_line_eq (x_0 : ℝ) (h : x_0 = 1) :
  ∃ y_0 : ℝ, y_0 = f x_0 ∧ 
  ∀ x y : ℝ, y = -(x - 1) + y_0 ↔ f 1 = 0 ∧ y = -x + 1 :=
by
  sorry

end normal_line_eq_l1519_151965


namespace sin_C_value_l1519_151903

noncomputable def triangle_sine_proof (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : Real := by
  -- Utilizing the Law of Sines and given conditions to find sin C
  sorry

theorem sin_C_value (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : triangle_sine_proof A B C a b c hB hb = Real.sqrt 3 / 6 := by
  sorry

end sin_C_value_l1519_151903


namespace number_of_subsets_including_1_and_10_l1519_151924

def A : Set ℕ := {a : ℕ | ∃ x y z : ℕ, a = 2^x * 3^y * 5^z}
def B : Set ℕ := {b : ℕ | b ∈ A ∧ 1 ≤ b ∧ b ≤ 10}

theorem number_of_subsets_including_1_and_10 :
  ∃ (s : Finset (Finset ℕ)), (∀ x ∈ s, 1 ∈ x ∧ 10 ∈ x) ∧ s.card = 128 := by
  sorry

end number_of_subsets_including_1_and_10_l1519_151924


namespace coordinates_of_P_l1519_151971

def point (x y : ℝ) := (x, y)

def A : (ℝ × ℝ) := point 1 1
def B : (ℝ × ℝ) := point 4 0

def vector_sub (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2)

def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem coordinates_of_P
  (P : ℝ × ℝ)
  (hP : vector_sub P A = scalar_mult 3 (vector_sub B P)) :
  P = (11 / 2, -1 / 2) :=
by
  sorry

end coordinates_of_P_l1519_151971


namespace necessarily_negative_l1519_151974

theorem necessarily_negative
  (a b c : ℝ)
  (ha : -2 < a ∧ a < -1)
  (hb : 0 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 0) :
  b + c < 0 :=
sorry

end necessarily_negative_l1519_151974


namespace range_of_f_l1519_151947

noncomputable def f (x : Real) : Real :=
  if x ≤ 1 then 2 * x + 1 else Real.log x + 1

theorem range_of_f (x : Real) : f x + f (x + 1) > 1 ↔ (x > -(3 / 4)) :=
  sorry

end range_of_f_l1519_151947


namespace binary_remainder_div_8_l1519_151979

theorem binary_remainder_div_8 (n : ℕ) (h : n = 0b101100110011) : n % 8 = 3 :=
by sorry

end binary_remainder_div_8_l1519_151979


namespace solve_for_x_l1519_151988

theorem solve_for_x (x : ℝ) (h : 1 / 3 + 1 / x = 2 / 3) : x = 3 :=
sorry

end solve_for_x_l1519_151988


namespace average_age_after_person_leaves_l1519_151909

theorem average_age_after_person_leaves 
  (initial_people : ℕ) 
  (initial_average_age : ℕ) 
  (person_leaving_age : ℕ) 
  (remaining_people : ℕ) 
  (new_average_age : ℝ)
  (h1 : initial_people = 7) 
  (h2 : initial_average_age = 32) 
  (h3 : person_leaving_age = 22) 
  (h4 : remaining_people = 6) :
  new_average_age = 34 := 
by 
  sorry

end average_age_after_person_leaves_l1519_151909


namespace factorial_equation_solution_l1519_151953

theorem factorial_equation_solution (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → (a, b, c) = (3, 3, 4) :=
by
  sorry

end factorial_equation_solution_l1519_151953


namespace length_of_AB_l1519_151973

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def slope_of_line : ℝ := Real.tan (Real.pi / 6)

-- Equation of the line in point-slope form
noncomputable def line_eq (x : ℝ) : ℝ :=
  (slope_of_line * x) + 1

-- Intersection points of the line with the parabola y = (1/4)x^2
noncomputable def parabola_eq (x : ℝ) : ℝ :=
  (1/4) * x ^ 2

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, 
    (A.2 = parabola_eq A.1) ∧
    (B.2 = parabola_eq B.1) ∧ 
    (A.2 = line_eq A.1) ∧
    (B.2 = line_eq B.1) ∧
    ((((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ^ (1 / 2)) = 16 / 3) :=
by
  sorry

end length_of_AB_l1519_151973


namespace connie_earbuds_tickets_l1519_151915

theorem connie_earbuds_tickets (total_tickets : ℕ) (koala_fraction : ℕ) (bracelet_tickets : ℕ) (earbud_tickets : ℕ) :
  total_tickets = 50 →
  koala_fraction = 2 →
  bracelet_tickets = 15 →
  (total_tickets / koala_fraction) + bracelet_tickets + earbud_tickets = total_tickets →
  earbud_tickets = 10 :=
by
  intros h_total h_koala h_bracelets h_sum
  sorry

end connie_earbuds_tickets_l1519_151915


namespace friend_saves_per_week_l1519_151986

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end friend_saves_per_week_l1519_151986


namespace total_pears_l1519_151977

theorem total_pears (S P C : ℕ) (hS : S = 20) (hP : P = (S - S / 2)) (hC : C = (P + P / 5)) : S + P + C = 42 :=
by
  -- We state the theorem with the given conditions and the goal of proving S + P + C = 42.
  sorry

end total_pears_l1519_151977


namespace functional_equation_solution_l1519_151939

theorem functional_equation_solution {
  f : ℝ → ℝ
} (h : ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2 * y * f x + (f y)^2) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = x + 1) :=
sorry

end functional_equation_solution_l1519_151939


namespace range_of_m_l1519_151991

open Set

-- Definitions and conditions
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0
def neg_p (x : ℝ) : Prop := ¬ p x
def neg_q (x m : ℝ) : Prop := ¬ q x m

-- Theorem statement
theorem range_of_m (x m : ℝ) (h₁ : ¬ p x → ¬ q x m) (h₂ : m > 0) : m ≥ 9 :=
  sorry

end range_of_m_l1519_151991


namespace mini_marshmallows_count_l1519_151916

theorem mini_marshmallows_count (total_marshmallows large_marshmallows : ℕ) (h1 : total_marshmallows = 18) (h2 : large_marshmallows = 8) :
  total_marshmallows - large_marshmallows = 10 :=
by 
  sorry

end mini_marshmallows_count_l1519_151916


namespace sequence_and_sum_problems_l1519_151948

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n-1) * d) / 2

def geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := b * r^(n-1)

noncomputable def sum_geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := 
(if r = 1 then b * n
 else b * (r^n - 1) / (r - 1))

theorem sequence_and_sum_problems :
  (∀ n : ℕ, arithmetic_sequence 19 (-2) n = 21 - 2 * n) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 19 (-2) n = 20 * n - n^2) ∧
  (∀ n : ℕ, ∃ a_n : ℤ, (geometric_sequence 1 3 n + (a_n - geometric_sequence 1 3 n) = 21 - 2 * n + 3^(n-1)) ∧
    sum_geometric_sequence 1 3 n = (sum_arithmetic_sequence 19 (-2) n + (3^n - 1) / 2))
:= by
  sorry

end sequence_and_sum_problems_l1519_151948


namespace max_value_proof_l1519_151982

noncomputable def maximum_value (x y z : ℝ) : ℝ := 
  (2/x) + (1/y) - (2/z) + 2

theorem max_value_proof {x y z : ℝ} 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0):
  maximum_value x y z ≤ 3 :=
sorry

end max_value_proof_l1519_151982


namespace import_rate_for_rest_of_1997_l1519_151954

theorem import_rate_for_rest_of_1997
    (import_1996: ℝ)
    (import_first_two_months_1997: ℝ)
    (excess_imports_1997: ℝ)
    (import_rate_first_two_months: ℝ)
    (expected_total_imports_1997: ℝ)
    (remaining_imports_1997: ℝ)
    (R: ℝ):
    excess_imports_1997 = 720e6 →
    expected_total_imports_1997 = import_1996 + excess_imports_1997 →
    remaining_imports_1997 = expected_total_imports_1997 - import_first_two_months_1997 →
    10 * R = remaining_imports_1997 →
    R = 180e6 :=
by
    intros h_import1996 h_import_first_two_months h_excess_imports h_import_rate_first_two_months 
           h_expected_total_imports h_remaining_imports h_equation
    sorry

end import_rate_for_rest_of_1997_l1519_151954


namespace xiaochun_age_l1519_151930

theorem xiaochun_age
  (x y : ℕ)
  (h1 : x = y - 18)
  (h2 : 2 * (x + 3) = y + 3) :
  x = 15 :=
sorry

end xiaochun_age_l1519_151930


namespace luis_finish_fourth_task_l1519_151934

-- Define the starting and finishing times
def start_time : ℕ := 540  -- 9:00 AM is 540 minutes from midnight
def finish_third_task : ℕ := 750  -- 12:30 PM is 750 minutes from midnight
def duration_one_task : ℕ := (750 - 540) / 3  -- Time for one task

-- Define the problem statement
theorem luis_finish_fourth_task :
  start_time = 540 →
  finish_third_task = 750 →
  3 * duration_one_task = finish_third_task - start_time →
  finish_third_task + duration_one_task = 820 :=
by
  -- You can place the proof for the theorem here
  sorry

end luis_finish_fourth_task_l1519_151934


namespace percent_exceed_not_ticketed_l1519_151956

-- Defining the given conditions
def total_motorists : ℕ := 100
def percent_exceed_limit : ℕ := 50
def percent_with_tickets : ℕ := 40

-- Calculate the number of motorists exceeding the limit and receiving tickets
def motorists_exceed_limit := total_motorists * percent_exceed_limit / 100
def motorists_with_tickets := total_motorists * percent_with_tickets / 100

-- Theorem: Percentage of motorists exceeding the limit but not receiving tickets
theorem percent_exceed_not_ticketed : 
  (motorists_exceed_limit - motorists_with_tickets) * 100 / motorists_exceed_limit = 20 := 
by
  sorry

end percent_exceed_not_ticketed_l1519_151956


namespace max_popsicles_with_10_dollars_l1519_151984

def price (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 3 then 2
  else if n = 5 then 3
  else if n = 7 then 4
  else 0

theorem max_popsicles_with_10_dollars : ∀ (a b c d : ℕ),
  a * price 1 + b * price 3 + c * price 5 + d * price 7 = 10 →
  a + 3 * b + 5 * c + 7 * d ≤ 17 :=
sorry

end max_popsicles_with_10_dollars_l1519_151984


namespace roots_cubic_sum_cubes_l1519_151980

theorem roots_cubic_sum_cubes (a b c : ℝ) 
    (h1 : 6 * a^3 - 803 * a + 1606 = 0)
    (h2 : 6 * b^3 - 803 * b + 1606 = 0)
    (h3 : 6 * c^3 - 803 * c + 1606 = 0) :
    (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := 
by
  sorry

end roots_cubic_sum_cubes_l1519_151980


namespace domain_of_v_l1519_151938

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (Real.cos x)

theorem domain_of_v :
  (∀ x : ℝ, (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < x ∧ x < 2 * n * Real.pi + Real.pi / 2) ↔ 
    ∀ x : ℝ, ∀ x_in_domain : ℝ, (0 < Real.cos x ∧ 1 / Real.sqrt (Real.cos x) = x_in_domain)) :=
sorry

end domain_of_v_l1519_151938


namespace cone_rolls_path_l1519_151908

theorem cone_rolls_path (r h m n : ℝ) (rotations : ℕ) 
  (h_rotations : rotations = 20)
  (h_ratio : h / r = 3 * Real.sqrt 133)
  (h_m : m = 3)
  (h_n : n = 133) : 
  m + n = 136 := 
by sorry

end cone_rolls_path_l1519_151908


namespace number_of_b_objects_l1519_151936

theorem number_of_b_objects
  (total_objects : ℕ) 
  (a_objects : ℕ) 
  (b_objects : ℕ) 
  (h1 : total_objects = 35) 
  (h2 : a_objects = 17) 
  (h3 : total_objects = a_objects + b_objects) :
  b_objects = 18 :=
by
  sorry

end number_of_b_objects_l1519_151936


namespace rectangular_plot_area_l1519_151911

theorem rectangular_plot_area (P : ℝ) (L W : ℝ) (h1 : P = 24) (h2 : L = 2 * W) :
    A = 32 := by
  sorry

end rectangular_plot_area_l1519_151911


namespace sum_of_roots_l1519_151993

theorem sum_of_roots (p q : ℝ) (h_eq : 2 * p + 3 * q = 6) (h_roots : ∀ x : ℝ, x ^ 2 - p * x + q = 0) : p = 2 := by
sorry

end sum_of_roots_l1519_151993


namespace min_value_of_x_plus_y_l1519_151957

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l1519_151957


namespace largest_positive_real_root_bound_l1519_151997

theorem largest_positive_real_root_bound (b0 b1 b2 : ℝ)
  (h_b0 : abs b0 ≤ 1) (h_b1 : abs b1 ≤ 1) (h_b2 : abs b2 ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b2 * r^2 + b1 * r + b0 = 0 ∧ 1.5 < r ∧ r < 2 := 
sorry

end largest_positive_real_root_bound_l1519_151997


namespace sunflower_seeds_more_than_half_on_day_three_l1519_151906

-- Define the initial state and parameters
def initial_sunflower_seeds : ℚ := 0.4
def initial_other_seeds : ℚ := 0.6
def daily_added_sunflower_seeds : ℚ := 0.2
def daily_added_other_seeds : ℚ := 0.3
def daily_sunflower_eaten_factor : ℚ := 0.7
def daily_other_eaten_factor : ℚ := 0.4

-- Define the recurrence relations for sunflower seeds and total seeds
def sunflower_seeds (n : ℕ) : ℚ :=
  match n with
  | 0     => initial_sunflower_seeds
  | (n+1) => daily_sunflower_eaten_factor * sunflower_seeds n + daily_added_sunflower_seeds

def total_seeds (n : ℕ) : ℚ := 1 + (n : ℚ) * 0.5

-- Define the main theorem stating that on Tuesday (Day 3), sunflower seeds are more than half
theorem sunflower_seeds_more_than_half_on_day_three : sunflower_seeds 2 / total_seeds 2 > 0.5 :=
by
  -- Formal proof will go here
  sorry

end sunflower_seeds_more_than_half_on_day_three_l1519_151906


namespace perpendicular_line_equation_l1519_151983

theorem perpendicular_line_equation (x y : ℝ) (h : 2 * x + y + 3 = 0) (hx : ∃ c : ℝ, x - 2 * y + c = 0) :
  (c = 7 ↔ ∀ p : ℝ × ℝ, p = (-1, 3) → (p.1 - 2 * p.2 + 7 = 0)) :=
sorry

end perpendicular_line_equation_l1519_151983


namespace remainder_when_divided_by_6_l1519_151926

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l1519_151926


namespace quadratic_function_points_relationship_l1519_151966

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end quadratic_function_points_relationship_l1519_151966


namespace bobbit_worm_days_l1519_151989

variable (initial_fish : ℕ)
variable (fish_added : ℕ)
variable (fish_eaten_per_day : ℕ)
variable (week_days : ℕ)
variable (final_fish : ℕ)
variable (d : ℕ)

theorem bobbit_worm_days (h1 : initial_fish = 60)
                         (h2 : fish_added = 8)
                         (h3 : fish_eaten_per_day = 2)
                         (h4 : week_days = 7)
                         (h5 : final_fish = 26) :
  60 - 2 * d + 8 - 2 * week_days = 26 → d = 14 :=
by {
  sorry
}

end bobbit_worm_days_l1519_151989


namespace evaluate_expression_is_41_l1519_151942

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end evaluate_expression_is_41_l1519_151942


namespace restaurant_customers_prediction_l1519_151976

def total_customers_saturday (breakfast_customers_friday lunch_customers_friday dinner_customers_friday : ℝ) : ℝ :=
  let breakfast_customers_saturday := 2 * breakfast_customers_friday
  let lunch_customers_saturday := lunch_customers_friday + 0.25 * lunch_customers_friday
  let dinner_customers_saturday := dinner_customers_friday - 0.15 * dinner_customers_friday
  breakfast_customers_saturday + lunch_customers_saturday + dinner_customers_saturday

theorem restaurant_customers_prediction :
  let breakfast_customers_friday := 73
  let lunch_customers_friday := 127
  let dinner_customers_friday := 87
  total_customers_saturday breakfast_customers_friday lunch_customers_friday dinner_customers_friday = 379 := 
by
  sorry

end restaurant_customers_prediction_l1519_151976


namespace painting_cost_3x_l1519_151907

-- Define the dimensions of the original room and the painting cost
variables (L B H : ℝ)
def cost_of_painting (area : ℝ) : ℝ := 350

-- Create a definition for the calculation of area
def paint_area (L B H : ℝ) : ℝ := 2 * (L * H + B * H)

-- Define the new dimensions
def new_dimensions (L B H : ℝ) : ℝ × ℝ × ℝ := (3 * L, 3 * B, 3 * H)

-- Create a definition for the calculation of the new area
def new_paint_area (L B H : ℝ) : ℝ := 18 * (paint_area L B H)

-- Calculate the new cost
def new_cost (L B H : ℝ) : ℝ := 18 * cost_of_painting (paint_area L B H)

-- The theorem to be proved
theorem painting_cost_3x (L B H : ℝ) : new_cost L B H = 6300 :=
by 
  simp [new_cost, cost_of_painting, paint_area]
  sorry

end painting_cost_3x_l1519_151907


namespace leo_amount_after_settling_debts_l1519_151964

theorem leo_amount_after_settling_debts (total_amount : ℝ) (ryan_share : ℝ) (ryan_owes_leo : ℝ) (leo_owes_ryan : ℝ) 
  (h1 : total_amount = 48) 
  (h2 : ryan_share = (2 / 3) * total_amount) 
  (h3 : ryan_owes_leo = 10) 
  (h4 : leo_owes_ryan = 7) : 
  (total_amount - ryan_share) + (ryan_owes_leo - leo_owes_ryan) = 19 :=
by
  sorry

end leo_amount_after_settling_debts_l1519_151964


namespace intersection_equal_l1519_151935

-- Define the sets M and N based on given conditions
def M : Set ℝ := {x : ℝ | x^2 - 3 * x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Define the intersection of M and N
def intersection : Set ℝ := {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)}

-- The statement to be proved
theorem intersection_equal : M ∩ N = intersection :=
by 
  sorry -- Skipping the proof

end intersection_equal_l1519_151935


namespace ms_cole_total_students_l1519_151943

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end ms_cole_total_students_l1519_151943


namespace four_op_two_l1519_151922

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end four_op_two_l1519_151922


namespace find_b_from_ellipse_l1519_151959

-- Definitions used in conditions
variables {F₁ F₂ : ℝ → ℝ} -- foci
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Conditions
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse a b P.1 P.2
def perpendicular_vectors (P : ℝ × ℝ) : Prop := true -- Simplified, use correct condition in detailed proof
def area_of_triangle (P : ℝ × ℝ) (F₁ F₂ : ℝ → ℝ) : ℝ := 9

-- The target statement
theorem find_b_from_ellipse (P : ℝ × ℝ) (condition1 : point_on_ellipse a b P)
  (condition2 : perpendicular_vectors P) 
  (condition3 : area_of_triangle P F₁ F₂ = 9) : 
  b = 3 := 
sorry

end find_b_from_ellipse_l1519_151959


namespace sin_45_eq_sqrt2_div_2_l1519_151901

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l1519_151901


namespace average_of_first_15_even_numbers_is_16_l1519_151913

-- Define the sum of the first 15 even numbers
def sum_first_15_even_numbers : ℕ :=
  2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30

-- Define the average of the first 15 even numbers
def average_of_first_15_even_numbers : ℕ :=
  sum_first_15_even_numbers / 15

-- Prove that the average is equal to 16
theorem average_of_first_15_even_numbers_is_16 : average_of_first_15_even_numbers = 16 :=
by
  -- Sorry placeholder for the proof
  sorry

end average_of_first_15_even_numbers_is_16_l1519_151913


namespace amount_paid_l1519_151950

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l1519_151950


namespace total_weight_full_bucket_l1519_151933

theorem total_weight_full_bucket (x y c d : ℝ) 
(h1 : x + 3/4 * y = c) 
(h2 : x + 1/3 * y = d) :
x + y = (8 * c - 3 * d) / 5 :=
sorry

end total_weight_full_bucket_l1519_151933


namespace price_of_each_pizza_l1519_151990

variable (P : ℝ)

theorem price_of_each_pizza (h1 : 4 * P + 5 = 45) : P = 10 := by
  sorry

end price_of_each_pizza_l1519_151990


namespace number_of_students_l1519_151985

/--
Statement: Several students are seated around a circular table. 
Each person takes one piece from a bag containing 120 pieces of candy 
before passing it to the next. Chris starts with the bag, takes one piece 
and also ends up with the last piece. Prove that the number of students
at the table could be 7 or 17.
-/
theorem number_of_students (n : Nat) (h : 120 > 0) :
  (∃ k, 119 = k * n ∧ n ≥ 1) → (n = 7 ∨ n = 17) :=
by
  sorry

end number_of_students_l1519_151985


namespace equivalent_single_discount_l1519_151927

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.25
noncomputable def coupon_discount : ℝ := 0.10
noncomputable def final_price : ℝ := 33.75

theorem equivalent_single_discount :
  (1 - final_price / original_price) * 100 = 32.5 :=
by
  sorry

end equivalent_single_discount_l1519_151927
