import Mathlib

namespace all_three_selected_l1391_139110

-- Define the probabilities
def P_R : ℚ := 6 / 7
def P_Rv : ℚ := 1 / 5
def P_Rs : ℚ := 2 / 3
def P_Rv_given_R : ℚ := 2 / 5
def P_Rs_given_Rv : ℚ := 1 / 2

-- The probability that all three are selected
def P_all : ℚ := P_R * P_Rv_given_R * P_Rs_given_Rv

-- Prove that the calculated probability is equal to the given answer
theorem all_three_selected : P_all = 6 / 35 :=
by
  sorry

end all_three_selected_l1391_139110


namespace determinant_property_l1391_139140

variable {R : Type} [CommRing R]
variable (x y z w : R)

theorem determinant_property 
  (h : x * w - y * z = 7) :
  (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by sorry

end determinant_property_l1391_139140


namespace expression_value_l1391_139159

theorem expression_value (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end expression_value_l1391_139159


namespace area_of_shaded_region_l1391_139165

-- Given conditions
def side_length := 8
def area_of_square := side_length * side_length
def area_of_triangle := area_of_square / 4

-- Lean 4 statement for the equivalence
theorem area_of_shaded_region : area_of_triangle = 16 :=
by
  sorry

end area_of_shaded_region_l1391_139165


namespace sequence_value_l1391_139121

theorem sequence_value (a b c d x : ℕ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 17) (h4 : d = 33)
  (h5 : b - a = 4) (h6 : c - b = 8) (h7 : d - c = 16) (h8 : x - d = 32) : x = 65 := by
  sorry

end sequence_value_l1391_139121


namespace max_ab_ac_bc_l1391_139102

noncomputable def maxValue (a b c : ℝ) := a * b + a * c + b * c

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : maxValue a b c ≤ 12 :=
by
  sorry

end max_ab_ac_bc_l1391_139102


namespace probability_of_picking_red_ball_l1391_139194

theorem probability_of_picking_red_ball (w r : ℕ) 
  (h1 : r > w) 
  (h2 : r < 2 * w) 
  (h3 : 2 * w + 3 * r = 60) : 
  r / (w + r) = 7 / 11 :=
sorry

end probability_of_picking_red_ball_l1391_139194


namespace largest_negative_is_l1391_139132

def largest_of_negatives (a b c d : ℚ) (largest : ℚ) : Prop := largest = max (max a b) (max c d)

theorem largest_negative_is (largest : ℚ) : largest_of_negatives (-2/3) (-2) (-1) (-5) largest → largest = -2/3 :=
by
  intro h
  -- We assume the definition and the theorem are sufficient to say largest = -2/3
  sorry

end largest_negative_is_l1391_139132


namespace total_cases_sold_is_correct_l1391_139170

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end total_cases_sold_is_correct_l1391_139170


namespace fourth_root_cubed_eq_729_l1391_139143

theorem fourth_root_cubed_eq_729 (x : ℝ) (hx : (x^(1/4))^3 = 729) : x = 6561 :=
  sorry

end fourth_root_cubed_eq_729_l1391_139143


namespace proposition_R_is_converse_negation_of_P_l1391_139186

variables (x y : ℝ)

def P : Prop := x + y = 0 → x = -y
def Q : Prop := ¬(x + y = 0) → x ≠ -y
def R : Prop := x ≠ -y → ¬(x + y = 0)

theorem proposition_R_is_converse_negation_of_P : R x y ↔ ¬P x y :=
by sorry

end proposition_R_is_converse_negation_of_P_l1391_139186


namespace angle_C_is_80_l1391_139146

-- Define the angles A, B, and C
def isoscelesTriangle (A B C : ℕ) : Prop :=
  -- Triangle ABC is isosceles with A = B, and C is 30 degrees more than A
  A = B ∧ C = A + 30 ∧ A + B + C = 180

-- Problem: Prove that angle C is 80 degrees given the conditions
theorem angle_C_is_80 (A B C : ℕ) (h : isoscelesTriangle A B C) : C = 80 :=
by sorry

end angle_C_is_80_l1391_139146


namespace angle_Z_is_120_l1391_139183

-- Define angles and lines
variables {p q : Prop} {X Y Z : ℝ}
variables (h_parallel : p ∧ q)
variables (hX : X = 100)
variables (hY : Y = 140)

-- Proof statement: Given the angles X and Y, we prove that angle Z is 120 degrees.
theorem angle_Z_is_120 (h_parallel : p ∧ q) (hX : X = 100) (hY : Y = 140) : Z = 120 := by 
  -- Here we would add the proof steps
  sorry

end angle_Z_is_120_l1391_139183


namespace negation_of_p_is_neg_p_l1391_139196

-- Define the proposition p
def p : Prop :=
  ∀ x > 0, (x + 1) * Real.exp x > 1

-- Define the negation of the proposition p
def neg_p : Prop :=
  ∃ x > 0, (x + 1) * Real.exp x ≤ 1

-- State the proof problem: negation of p is neg_p
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by
  -- Stating that ¬p is equivalent to neg_p
  sorry

end negation_of_p_is_neg_p_l1391_139196


namespace num_distinct_orders_of_targets_l1391_139169

theorem num_distinct_orders_of_targets : 
  let total_targets := 10
  let column_A_targets := 4
  let column_B_targets := 4
  let column_C_targets := 2
  (Nat.factorial total_targets) / 
  ((Nat.factorial column_A_targets) * (Nat.factorial column_B_targets) * (Nat.factorial column_C_targets)) = 5040 := 
by
  sorry

end num_distinct_orders_of_targets_l1391_139169


namespace inequality_a2b3c_l1391_139171

theorem inequality_a2b3c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end inequality_a2b3c_l1391_139171


namespace unique_solution_single_element_l1391_139134

theorem unique_solution_single_element (a : ℝ) 
  (h : ∀ x y : ℝ, (a * x^2 + a * x + 1 = 0) → (a * y^2 + a * y + 1 = 0) → x = y) : a = 4 := 
by
  sorry

end unique_solution_single_element_l1391_139134


namespace trajectory_of_M_l1391_139135

-- Define the two circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for the moving circle M being tangent to both circles
def isTangent (Mx My : ℝ) : Prop := 
  let distC1 := (Mx + 3)^2 + My^2
  let distC2 := (Mx - 3)^2 + My^2
  distC2 - distC1 = 4

-- The equation of the trajectory of M
theorem trajectory_of_M (Mx My : ℝ) (h : isTangent Mx My) : 
  Mx^2 - (My^2 / 8) = 1 ∧ Mx < 0 :=
sorry

end trajectory_of_M_l1391_139135


namespace average_height_of_four_people_l1391_139189

theorem average_height_of_four_people (
  h1 h2 h3 h4 : ℕ
) (diff12 : h2 = h1 + 2)
  (diff23 : h3 = h2 + 2)
  (diff34 : h4 = h3 + 6)
  (h4_eq : h4 = 83) :
  (h1 + h2 + h3 + h4) / 4 = 77 :=
by sorry

end average_height_of_four_people_l1391_139189


namespace main_theorem_l1391_139131

-- Let x be a real number
variable {x : ℝ}

-- Define the given identity
def identity (M₁ M₂ : ℝ) : Prop :=
  ∀ x, (50 * x - 42) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)

-- The proposition to prove the numerical value of M₁M₂
def prove_M1M2_value : Prop :=
  ∀ (M₁ M₂ : ℝ), identity M₁ M₂ → M₁ * M₂ = -6264

theorem main_theorem : prove_M1M2_value :=
  sorry

end main_theorem_l1391_139131


namespace football_team_practice_missed_days_l1391_139150

theorem football_team_practice_missed_days 
(daily_practice_hours : ℕ) 
(total_practice_hours : ℕ) 
(days_in_week : ℕ) 
(h1 : daily_practice_hours = 5) 
(h2 : total_practice_hours = 30) 
(h3 : days_in_week = 7) : 
days_in_week - (total_practice_hours / daily_practice_hours) = 1 := 
by 
  sorry

end football_team_practice_missed_days_l1391_139150


namespace arithmetic_sequence_root_arithmetic_l1391_139148

theorem arithmetic_sequence_root_arithmetic (a : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_root : ∀ x : ℝ, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) : 
  a 6 = -6 := 
by
  -- We skip the proof as per instructions
  sorry

end arithmetic_sequence_root_arithmetic_l1391_139148


namespace sarah_took_correct_amount_l1391_139161

-- Definition of the conditions
def total_cookies : Nat := 150
def neighbors_count : Nat := 15
def correct_amount_per_neighbor : Nat := 10
def remaining_cookies : Nat := 8
def first_neighbors_count : Nat := 14
def last_neighbor : String := "Sarah"

-- Calculations based on conditions
def total_cookies_taken : Nat := total_cookies - remaining_cookies
def correct_cookies_taken : Nat := first_neighbors_count * correct_amount_per_neighbor
def extra_cookies_taken : Nat := total_cookies_taken - correct_cookies_taken
def sarah_cookies : Nat := correct_amount_per_neighbor + extra_cookies_taken

-- Proof statement: Sarah took 12 cookies
theorem sarah_took_correct_amount : sarah_cookies = 12 := by
  sorry

end sarah_took_correct_amount_l1391_139161


namespace commute_time_l1391_139137

theorem commute_time (start_time : ℕ) (first_station_time : ℕ) (work_time : ℕ) 
  (h1 : start_time = 6 * 60) 
  (h2 : first_station_time = 40) 
  (h3 : work_time = 9 * 60) : 
  work_time - (start_time + first_station_time) = 140 :=
by
  sorry

end commute_time_l1391_139137


namespace sequence_general_term_correctness_l1391_139162

def sequenceGeneralTerm (n : ℕ) : ℤ :=
  if n % 2 = 1 then
    0
  else
    (-1) ^ (n / 2 + 1)

theorem sequence_general_term_correctness (n : ℕ) :
  (∀ m, sequenceGeneralTerm m = 0 ↔ m % 2 = 1) ∧
  (∀ k, sequenceGeneralTerm k = (-1) ^ (k / 2 + 1) ↔ k % 2 = 0) :=
by
  sorry

end sequence_general_term_correctness_l1391_139162


namespace sum_of_consecutive_integers_l1391_139177

theorem sum_of_consecutive_integers (x : ℤ) (h1 : x * (x + 1) + x + (x + 1) = 156) (h2 : x + 1 < 20) : x + (x + 1) = 23 :=
by
  sorry

end sum_of_consecutive_integers_l1391_139177


namespace marbles_count_l1391_139182

theorem marbles_count (initial_marble: ℕ) (bought_marble: ℕ) (final_marble: ℕ) 
  (h1: initial_marble = 53) (h2: bought_marble = 134) : 
  final_marble = initial_marble + bought_marble -> final_marble = 187 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

-- sorry is omitted as proof is given.

end marbles_count_l1391_139182


namespace value_of_a_l1391_139184

theorem value_of_a (a : ℚ) (h : a + a / 4 = 6 / 2) : a = 12 / 5 := by
  sorry

end value_of_a_l1391_139184


namespace complex_plane_squares_areas_l1391_139116

theorem complex_plane_squares_areas (z : ℂ) 
  (h1 : z^3 - z = i * (z^2 - z) ∨ z^3 - z = -i * (z^2 - z))
  (h2 : z^4 - z = i * (z^3 - z) ∨ z^4 - z = -i * (z^3 - z)) :
  ( ∃ A₁ A₂ : ℝ, (A₁ = 10 ∨ A₁ = 18) ∧ (A₂ = 10 ∨ A₂ = 18) ) := 
sorry

end complex_plane_squares_areas_l1391_139116


namespace length_of_XY_l1391_139115

theorem length_of_XY (A B C D P Q X Y : ℝ) (h₁ : A = B) (h₂ : C = D) 
  (h₃ : A + B = 13) (h₄ : C + D = 21) (h₅ : A + P = 7) 
  (h₆ : C + Q = 8) (h₇ : P ≠ Q) (h₈ : P + Q = 30) :
  ∃ k : ℝ, XY = 2 * k + 30 + 31 / 15 :=
by sorry

end length_of_XY_l1391_139115


namespace second_remainder_l1391_139108

theorem second_remainder (n : ℕ) : n = 210 ∧ n % 13 = 3 → n % 17 = 6 :=
by
  sorry

end second_remainder_l1391_139108


namespace extinction_probability_l1391_139111

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l1391_139111


namespace find_k_value_l1391_139168

variable (x y z k : ℝ)

theorem find_k_value (h : 7 / (x + y) = k / (x + z) ∧ k / (x + z) = 11 / (z - y)) :
  k = 18 :=
sorry

end find_k_value_l1391_139168


namespace length_of_BC_l1391_139187

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l1391_139187


namespace point_A_in_second_quadrant_l1391_139106

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l1391_139106


namespace jordan_more_novels_than_maxime_l1391_139173

def jordan_french_novels : ℕ := 130
def jordan_spanish_novels : ℕ := 20

def alexandre_french_novels : ℕ := jordan_french_novels / 10
def alexandre_spanish_novels : ℕ := 3 * jordan_spanish_novels

def camille_french_novels : ℕ := 2 * alexandre_french_novels
def camille_spanish_novels : ℕ := jordan_spanish_novels / 2

def total_french_novels : ℕ := jordan_french_novels + alexandre_french_novels + camille_french_novels

def maxime_french_novels : ℕ := total_french_novels / 2 - 5
def maxime_spanish_novels : ℕ := 2 * camille_spanish_novels

def jordan_total_novels : ℕ := jordan_french_novels + jordan_spanish_novels
def maxime_total_novels : ℕ := maxime_french_novels + maxime_spanish_novels

def novels_difference : ℕ := jordan_total_novels - maxime_total_novels

theorem jordan_more_novels_than_maxime : novels_difference = 51 :=
sorry

end jordan_more_novels_than_maxime_l1391_139173


namespace sin_300_eq_neg_sqrt_3_div_2_l1391_139109

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l1391_139109


namespace concession_stand_total_revenue_l1391_139160

theorem concession_stand_total_revenue :
  let hot_dog_price : ℝ := 1.50
  let soda_price : ℝ := 0.50
  let total_items_sold : ℕ := 87
  let hot_dogs_sold : ℕ := 35
  let sodas_sold := total_items_sold - hot_dogs_sold
  let revenue_from_hot_dogs := hot_dogs_sold * hot_dog_price
  let revenue_from_sodas := sodas_sold * soda_price
  revenue_from_hot_dogs + revenue_from_sodas = 78.50 :=
by {
  -- Proof will go here
  sorry
}

end concession_stand_total_revenue_l1391_139160


namespace find_integers_a_l1391_139104

theorem find_integers_a (a : ℤ) : 
  (∃ n : ℤ, (a^3 + 1 = (a - 1) * n)) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := 
sorry

end find_integers_a_l1391_139104


namespace not_invited_students_l1391_139198

-- Definition of the problem conditions
def students := 15
def direct_friends_of_mia := 4
def unique_friends_of_each_friend := 2

-- Problem statement
theorem not_invited_students : (students - (1 + direct_friends_of_mia + direct_friends_of_mia * unique_friends_of_each_friend) = 2) :=
by
  sorry

end not_invited_students_l1391_139198


namespace quadratic_expression_l1391_139185

theorem quadratic_expression (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 6) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := 
by sorry

end quadratic_expression_l1391_139185


namespace distance_between_points_A_B_l1391_139155

theorem distance_between_points_A_B :
  let A := (8, -5)
  let B := (0, 10)
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 17 :=
by
  let A := (8, -5)
  let B := (0, 10)
  sorry

end distance_between_points_A_B_l1391_139155


namespace complete_square_eqn_l1391_139133

theorem complete_square_eqn (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 → (x + d)^2 = e) → d + e = 5 :=
by
  sorry

end complete_square_eqn_l1391_139133


namespace evaluate_expression_l1391_139179

theorem evaluate_expression (a b : ℤ) (h_a : a = 1) (h_b : b = -2) : 
  2 * (a^2 - 3 * a * b + 1) - (2 * a^2 - b^2) + 5 * a * b = 8 :=
by
  sorry

end evaluate_expression_l1391_139179


namespace average_speed_of_trip_is_correct_l1391_139139

-- Definitions
def total_distance : ℕ := 450
def distance_part1 : ℕ := 300
def speed_part1 : ℕ := 20
def distance_part2 : ℕ := 150
def speed_part2 : ℕ := 15

-- The average speed problem
theorem average_speed_of_trip_is_correct :
  (total_distance : ℤ) / (distance_part1 / speed_part1 + distance_part2 / speed_part2 : ℤ) = 18 := by
  sorry

end average_speed_of_trip_is_correct_l1391_139139


namespace ratio_bananas_apples_is_3_to_1_l1391_139122

def ratio_of_bananas_to_apples (oranges apples bananas peaches total_fruit : ℕ) : ℚ :=
if oranges = 6 ∧ apples = oranges - 2 ∧ peaches = bananas / 2 ∧ total_fruit = 28
   ∧ 6 + apples + bananas + peaches = total_fruit then
    bananas / apples
else 0

theorem ratio_bananas_apples_is_3_to_1 : ratio_of_bananas_to_apples 6 4 12 6 28 = 3 := by
sorry

end ratio_bananas_apples_is_3_to_1_l1391_139122


namespace solve_for_y_l1391_139180

theorem solve_for_y (y : ℚ) : y - 1 / 2 = 1 / 6 - 2 / 3 + 1 / 4 → y = 1 / 4 := by
  intro h
  sorry

end solve_for_y_l1391_139180


namespace approximation_irrational_quotient_l1391_139120

theorem approximation_irrational_quotient 
  (r1 r2 : ℝ) (irrational : ¬ ∃ q : ℚ, r1 = q * r2) 
  (x : ℝ) (p : ℝ) (pos_p : p > 0) : 
  ∃ (k1 k2 : ℤ), |x - (k1 * r1 + k2 * r2)| < p :=
sorry

end approximation_irrational_quotient_l1391_139120


namespace find_n_l1391_139144

variable (n : ℚ)

theorem find_n (h : (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4)) : 
  n = -2 / 3 :=
by
  sorry

end find_n_l1391_139144


namespace total_money_l1391_139100

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end total_money_l1391_139100


namespace kanul_cash_percentage_l1391_139138

-- Define the conditions
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 1000
def total_amount : ℝ := 5714.29
def total_spent := raw_materials_cost + machinery_cost
def cash := total_amount - total_spent

-- The goal is to prove the percentage of the total amount as cash is 30%
theorem kanul_cash_percentage :
  (cash / total_amount) * 100 = 30 := 
sorry

end kanul_cash_percentage_l1391_139138


namespace find_side_a_l1391_139130

noncomputable def side_a (b : ℝ) (A : ℝ) (S : ℝ) : ℝ :=
  2 * S / (b * Real.sin A)

theorem find_side_a :
  let b := 2
  let A := Real.pi * 2 / 3 -- 120 degrees in radians
  let S := 2 * Real.sqrt 3
  side_a b A S = 4 :=
by
  let b := 2
  let A := Real.pi * 2 / 3
  let S := 2 * Real.sqrt 3
  show side_a b A S = 4
  sorry

end find_side_a_l1391_139130


namespace monomials_like_terms_l1391_139119

theorem monomials_like_terms (a b : ℤ) (h1 : a + 1 = 2) (h2 : b - 2 = 3) : a + b = 6 :=
sorry

end monomials_like_terms_l1391_139119


namespace line_equation_l1391_139128

theorem line_equation (a T : ℝ) (h : 0 < a ∧ 0 < T) :
  ∃ (x y : ℝ), (2 * T * x - a^2 * y + 2 * a * T = 0) :=
by
  sorry

end line_equation_l1391_139128


namespace find_x_from_ratio_l1391_139142

theorem find_x_from_ratio (x y k: ℚ) 
  (h1 : ∀ x y, (5 * x - 3) / (y + 20) = k) 
  (h2 : 5 * 1 - 3 = 2 * 22) (hy : y = 5) : 
  x = 58 / 55 := 
by 
  sorry

end find_x_from_ratio_l1391_139142


namespace x_in_A_neither_sufficient_nor_necessary_for_x_in_B_l1391_139157

def A : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem x_in_A_neither_sufficient_nor_necessary_for_x_in_B : ¬ ((∀ x, x ∈ A → x ∈ B) ∧ (∀ x, x ∈ B → x ∈ A)) := by
  sorry

end x_in_A_neither_sufficient_nor_necessary_for_x_in_B_l1391_139157


namespace value_of_x_l1391_139101

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def f_inv (y : ℝ) : ℝ := sorry -- Placeholder for the inverse of f

noncomputable def g (x : ℝ) : ℝ := 3 * f_inv x

theorem value_of_x (h : g 18 = 18) : x = 30 / 11 :=
by
  -- Proof is not required.
  sorry

end value_of_x_l1391_139101


namespace trip_time_is_correct_l1391_139164

noncomputable def total_trip_time : ℝ :=
  let wrong_direction_time := 100 / 60
  let return_time := 100 / 45
  let detour_time := 30 / 45
  let normal_trip_time := 300 / 60
  let stop_time := 2 * (15 / 60)
  wrong_direction_time + return_time + detour_time + normal_trip_time + stop_time

theorem trip_time_is_correct : total_trip_time = 10.06 :=
  by
    -- Proof steps are omitted
    sorry

end trip_time_is_correct_l1391_139164


namespace find_number_l1391_139152

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 10) : x = 5 :=
by
  sorry

end find_number_l1391_139152


namespace division_by_fraction_equiv_neg_multiplication_l1391_139105

theorem division_by_fraction_equiv_neg_multiplication (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 :=
by
  -- Proof would go here, but we use sorry to skip the proof for now.
  sorry

end division_by_fraction_equiv_neg_multiplication_l1391_139105


namespace smallest_prime_less_than_square_l1391_139192

theorem smallest_prime_less_than_square : ∃ p n : ℕ, Prime p ∧ p = n^2 - 20 ∧ p = 5 :=
by 
  sorry

end smallest_prime_less_than_square_l1391_139192


namespace repeating_decimal_sum_l1391_139199

theorem repeating_decimal_sum :
  let x := (1 : ℚ) / 3
  let y := (7 : ℚ) / 33
  x + y = 6 / 11 :=
  by
  sorry

end repeating_decimal_sum_l1391_139199


namespace solve_expression_l1391_139166

theorem solve_expression (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end solve_expression_l1391_139166


namespace fg_of_2_eq_81_l1391_139129

def f (x : ℝ) : ℝ := x ^ 2
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 1

theorem fg_of_2_eq_81 : f (g 2) = 81 := by
  sorry

end fg_of_2_eq_81_l1391_139129


namespace tetrahedron_solution_l1391_139113

noncomputable def num_triangles (a : ℝ) (E F G : ℝ → ℝ → ℝ) : ℝ :=
  if a > 3 then 3 else 0

theorem tetrahedron_solution (a : ℝ) (E F G : ℝ → ℝ → ℝ) :
  a > 3 → num_triangles a E F G = 3 := by
  sorry

end tetrahedron_solution_l1391_139113


namespace no_int_solutions_except_zero_l1391_139167

theorem no_int_solutions_except_zero 
  (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
by
  sorry

end no_int_solutions_except_zero_l1391_139167


namespace find_V_y_l1391_139118

-- Define the volumes and percentages given in the problem
def V_x : ℕ := 300
def percent_x : ℝ := 0.10
def percent_y : ℝ := 0.30
def desired_percent : ℝ := 0.22

-- Define the alcohol volumes in the respective solutions
def alcohol_x := percent_x * V_x
def total_volume (V_y : ℕ) := V_x + V_y
def desired_alcohol (V_y : ℕ) := desired_percent * (total_volume V_y)

-- Define our main statement
theorem find_V_y : ∃ (V_y : ℕ), alcohol_x + (percent_y * V_y) = desired_alcohol V_y ∧ V_y = 450 :=
by
  sorry

end find_V_y_l1391_139118


namespace find_y_given_conditions_l1391_139175

theorem find_y_given_conditions (x : ℤ) (y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 2) (h2 : x = -5) : y = 45 :=
by
  sorry

end find_y_given_conditions_l1391_139175


namespace mouse_jump_vs_grasshopper_l1391_139178

-- Definitions for jumps
def grasshopper_jump : ℕ := 14
def frog_jump : ℕ := grasshopper_jump + 37
def mouse_jump : ℕ := frog_jump - 16

-- Theorem stating the result
theorem mouse_jump_vs_grasshopper : mouse_jump - grasshopper_jump = 21 :=
by
  -- Skip the proof
  sorry

end mouse_jump_vs_grasshopper_l1391_139178


namespace als_initial_portion_l1391_139151

theorem als_initial_portion (a b c : ℝ)
  (h1 : a + b + c = 1200)
  (h2 : a - 150 + 3 * b + 3 * c = 1800) :
  a = 825 :=
sorry

end als_initial_portion_l1391_139151


namespace hyperbola_asymptotes_and_point_l1391_139117

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_asymptotes_and_point 
  (x y : ℝ)
  (asymptote1 : ∀ x, y = (1/2) * x)
  (asymptote2 : ∀ x, y = (-1/2) * x)
  (point : (x, y) = (4, Real.sqrt 2))
: hyperbola_equation x y :=
sorry

end hyperbola_asymptotes_and_point_l1391_139117


namespace arjun_starting_amount_l1391_139123

theorem arjun_starting_amount (X : ℝ) (h1 : Anoop_investment = 4000) (h2 : Anoop_months = 6) (h3 : Arjun_months = 12) (h4 : (X * 12) = (4000 * 6)) :
  X = 2000 :=
sorry

end arjun_starting_amount_l1391_139123


namespace train_crossing_time_l1391_139126

theorem train_crossing_time
    (train_speed_kmph : ℕ)
    (platform_length_meters : ℕ)
    (crossing_time_platform_seconds : ℕ)
    (crossing_time_man_seconds : ℕ)
    (train_speed_mps : ℤ)
    (train_length_meters : ℤ)
    (T : ℤ)
    (h1 : train_speed_kmph = 72)
    (h2 : platform_length_meters = 340)
    (h3 : crossing_time_platform_seconds = 35)
    (h4 : train_speed_mps = 20)
    (h5 : train_length_meters = 360)
    (h6 : train_length_meters = train_speed_mps * crossing_time_man_seconds)
    : T = 18 :=
by
  sorry

end train_crossing_time_l1391_139126


namespace intersection_A_B_l1391_139156

-- Definitions for sets A and B based on the problem conditions
def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.log (2 - x) }

-- Proof problem statement
theorem intersection_A_B : (A ∩ B) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l1391_139156


namespace man_l1391_139197

-- Define all given conditions using Lean definitions
def speed_with_current_wind : ℝ := 22
def speed_of_current : ℝ := 5
def wind_resistance_factor : ℝ := 0.15
def current_increase_factor : ℝ := 0.10

-- Define the key quantities (man's speed in still water, effective speed in still water, new current speed against)
def speed_in_still_water : ℝ := speed_with_current_wind - speed_of_current
def effective_speed_in_still_water : ℝ := speed_in_still_water - (wind_resistance_factor * speed_in_still_water)
def new_speed_of_current_against : ℝ := speed_of_current + (current_increase_factor * speed_of_current)

-- Proof goal: Prove that the man's speed against the current is 8.95 km/hr considering all the conditions
theorem man's_speed_against_current_is_correct : 
  (effective_speed_in_still_water - new_speed_of_current_against) = 8.95 := 
by
  sorry

end man_l1391_139197


namespace quadratic_has_distinct_real_roots_l1391_139188

theorem quadratic_has_distinct_real_roots :
  ∀ (x : ℝ), x^2 - 2 * x - 1 = 0 → (∃ Δ > 0, Δ = ((-2)^2 - 4 * 1 * (-1))) := by
  sorry

end quadratic_has_distinct_real_roots_l1391_139188


namespace dartboard_area_ratio_l1391_139172

theorem dartboard_area_ratio
  (side_length : ℝ)
  (h_side_length : side_length = 2)
  (t : ℝ)
  (q : ℝ)
  (h_t : t = (1 / 2) * (1 / (Real.sqrt 2)) * (1 / (Real.sqrt 2)))
  (h_q : q = ((side_length * side_length) - (8 * t)) / 4) :
  q / t = 2 := by
  sorry

end dartboard_area_ratio_l1391_139172


namespace min_xy_value_min_x_plus_y_value_l1391_139195

variable {x y : ℝ}

theorem min_xy_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : xy ≥ 64 := 
sorry

theorem min_x_plus_y_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : x + y ≥ 18 :=
sorry

end min_xy_value_min_x_plus_y_value_l1391_139195


namespace largest_k_for_3_in_g_l1391_139154

theorem largest_k_for_3_in_g (k : ℝ) :
  (∃ x : ℝ, 2*x^2 - 8*x + k = 3) ↔ k ≤ 11 :=
by
  sorry

end largest_k_for_3_in_g_l1391_139154


namespace neither_necessary_nor_sufficient_l1391_139136

def p (x y : ℝ) : Prop := x > 1 ∧ y > 1
def q (x y : ℝ) : Prop := x + y > 3

theorem neither_necessary_nor_sufficient :
  ¬ (∀ x y, q x y → p x y) ∧ ¬ (∀ x y, p x y → q x y) :=
by
  sorry

end neither_necessary_nor_sufficient_l1391_139136


namespace johns_personal_payment_l1391_139127

theorem johns_personal_payment 
  (cost_per_hearing_aid : ℕ)
  (num_hearing_aids : ℕ)
  (deductible : ℕ)
  (coverage_percent : ℕ)
  (coverage_limit : ℕ) 
  (total_payment : ℕ)
  (insurance_payment_over_limit : ℕ) : 
  cost_per_hearing_aid = 2500 ∧ 
  num_hearing_aids = 2 ∧ 
  deductible = 500 ∧ 
  coverage_percent = 80 ∧ 
  coverage_limit = 3500 →
  total_payment = cost_per_hearing_aid * num_hearing_aids - deductible →
  insurance_payment_over_limit = max 0 (coverage_percent * total_payment / 100 - coverage_limit) →
  (total_payment - min (coverage_percent * total_payment / 100) coverage_limit + deductible = 1500) :=
by
  intros
  sorry

end johns_personal_payment_l1391_139127


namespace evaluate_expression_l1391_139114

-- Define the conditions
def num : ℤ := 900^2
def a : ℤ := 306
def b : ℤ := 294
def denom : ℤ := a^2 - b^2

-- State the theorem to be proven
theorem evaluate_expression : (num : ℚ) / denom = 112.5 :=
by
  -- proof is skipped
  sorry

end evaluate_expression_l1391_139114


namespace intersection_P_Q_l1391_139153

-- Defining the two sets P and Q
def P := { x : ℤ | abs x ≤ 2 }
def Q := { x : ℝ | -1 < x ∧ x < 5/2 }

-- Statement to prove
theorem intersection_P_Q : 
  { x : ℤ | abs x ≤ 2 } ∩ { x : ℤ | -1 < ((x : ℝ)) ∧ ((x : ℝ)) < 5/2 } = {0, 1, 2} := sorry

end intersection_P_Q_l1391_139153


namespace number_of_members_l1391_139191

-- Definitions based on conditions in the problem
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def cap_cost : ℕ := tshirt_cost

def home_game_cost_per_member : ℕ := sock_cost + tshirt_cost
def away_game_cost_per_member : ℕ := sock_cost + tshirt_cost + cap_cost
def total_cost_per_member : ℕ := home_game_cost_per_member + away_game_cost_per_member

def total_league_cost : ℕ := 4324

-- Statement to be proved
theorem number_of_members (m : ℕ) (h : total_league_cost = m * total_cost_per_member) : m = 85 :=
sorry

end number_of_members_l1391_139191


namespace first_plot_germination_rate_l1391_139149

-- Define the known quantities and conditions
def plot1_seeds : ℕ := 300
def plot2_seeds : ℕ := 200
def plot2_germination_rate : ℚ := 35 / 100
def total_germination_percentage : ℚ := 26 / 100

-- Define a statement to prove the percentage of seeds that germinated in the first plot
theorem first_plot_germination_rate : 
  ∃ (x : ℚ), (x / 100) * plot1_seeds + (plot2_germination_rate * plot2_seeds) = total_germination_percentage * (plot1_seeds + plot2_seeds) ∧ x = 20 :=
by
  sorry

end first_plot_germination_rate_l1391_139149


namespace distance_from_A_to_B_l1391_139112

-- Definitions of the conditions
def avg_speed : ℝ := 25
def distance_AB (D : ℝ) : Prop := ∃ T : ℝ, D / (4 * T) = avg_speed ∧ D = 3 * (T * avg_speed)∧ (D / 2) = (T * avg_speed)

theorem distance_from_A_to_B : ∃ D : ℝ, distance_AB D ∧ D = 100 / 3 :=
by
  sorry

end distance_from_A_to_B_l1391_139112


namespace ratio_areas_of_circumscribed_circles_l1391_139158

theorem ratio_areas_of_circumscribed_circles (P : ℝ) (A B : ℝ)
  (h1 : ∃ (x : ℝ), P = 8 * x)
  (h2 : ∃ (s : ℝ), s = P / 3)
  (hA : A = (5 * (P^2) * Real.pi) / 128)
  (hB : B = (P^2 * Real.pi) / 27) :
  A / B = 135 / 128 := by
  sorry

end ratio_areas_of_circumscribed_circles_l1391_139158


namespace diamond_19_98_l1391_139145

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end diamond_19_98_l1391_139145


namespace fitness_club_alpha_is_more_advantageous_l1391_139107

-- Define the costs and attendance pattern constants
def yearly_cost_alpha : ℕ := 11988
def monthly_cost_beta : ℕ := 1299
def weeks_per_month : ℕ := 4

-- Define the attendance pattern
def attendance_pattern : List ℕ := [3 * weeks_per_month, 2 * weeks_per_month, 1 * weeks_per_month, 0 * weeks_per_month]

-- Compute the total visits in a year for regular attendance
def total_visits (patterns : List ℕ) : ℕ :=
  patterns.sum * 3

-- Compute the total yearly cost for Beta when considering regular attendance
def yearly_cost_beta (monthly_cost : ℕ) : ℕ :=
  monthly_cost * 12

-- Calculate cost per visit for each club with given attendance
def cost_per_visit (total_cost : ℕ) (total_visits : ℕ) : ℚ :=
  total_cost / total_visits

theorem fitness_club_alpha_is_more_advantageous :
  cost_per_visit yearly_cost_alpha (total_visits attendance_pattern) <
  cost_per_visit (yearly_cost_beta monthly_cost_beta) (total_visits attendance_pattern) :=
by
  sorry

end fitness_club_alpha_is_more_advantageous_l1391_139107


namespace cone_volume_l1391_139163

theorem cone_volume (lateral_area : ℝ) (angle : ℝ) 
  (h₀ : lateral_area = 20 * Real.pi)
  (h₁ : angle = Real.arccos (4/5)) : 
  (1/3) * Real.pi * (4^2) * 3 = 16 * Real.pi :=
by
  sorry

end cone_volume_l1391_139163


namespace line_parabola_intersection_l1391_139174

theorem line_parabola_intersection (k : ℝ) (M A B : ℝ × ℝ) (h1 : ¬ k = 0) 
  (h2 : M = (2, 0))
  (h3 : ∃ x y, (x = k * y + 2 ∧ (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧ (p = A ∨ p = B))) 
  : 1 / |dist M A|^2 + 1 / |dist M B|^2 = 1 / 4 := 
by 
  sorry

end line_parabola_intersection_l1391_139174


namespace find_k_l1391_139125

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem find_k (k : ℝ) (h : dot_product (k * a.1, k * a.2 + b.2) (3 * a.1, 3 * a.2 - b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end find_k_l1391_139125


namespace B_gives_C_100_meters_start_l1391_139181

-- Definitions based on given conditions
variables (Va Vb Vc : ℝ) (T : ℝ)

-- Assume the conditions based on the problem statement
def race_condition_1 := Va = 1000 / T
def race_condition_2 := Vb = 900 / T
def race_condition_3 := Vc = 850 / T

-- Theorem stating that B can give C a 100 meter start
theorem B_gives_C_100_meters_start
  (h1 : race_condition_1 Va T)
  (h2 : race_condition_2 Vb T)
  (h3 : race_condition_3 Vc T) :
  (Vb = (1000 - 100) / T) :=
by
  -- Utilize conditions h1, h2, and h3
  sorry

end B_gives_C_100_meters_start_l1391_139181


namespace compute_expression_l1391_139124

-- Define the operation a Δ b
def Delta (a b : ℝ) : ℝ := a^2 - 2 * b

theorem compute_expression :
  let x := 3 ^ (Delta 4 10)
  let y := 4 ^ (Delta 2 3)
  Delta x y = ( -819.125 / 6561) :=
by 
  sorry

end compute_expression_l1391_139124


namespace cartons_loaded_l1391_139103

def total_cartons : Nat := 50
def cans_per_carton : Nat := 20
def cans_left_to_load : Nat := 200

theorem cartons_loaded (C : Nat) (h : cans_per_carton ≠ 0) : 
  C = total_cartons - (cans_left_to_load / cans_per_carton) := by
  sorry

end cartons_loaded_l1391_139103


namespace least_m_value_l1391_139176

def recursive_sequence (x : ℕ → ℚ) : Prop :=
  x 0 = 3 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_value (x : ℕ → ℚ) (h : recursive_sequence x) : ∃ m, m > 0 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k, k > 0 → k < m → x k > 3 + 1 / 2^10 :=
sorry

end least_m_value_l1391_139176


namespace pages_left_to_read_correct_l1391_139141

def total_pages : Nat := 563
def pages_read : Nat := 147
def pages_left_to_read : Nat := 416

theorem pages_left_to_read_correct : total_pages - pages_read = pages_left_to_read := by
  sorry

end pages_left_to_read_correct_l1391_139141


namespace first_meet_at_starting_point_l1391_139147

-- Definitions
def track_length := 300
def speed_A := 2
def speed_B := 4

-- Theorem: A and B will meet at the starting point for the first time after 400 seconds.
theorem first_meet_at_starting_point : 
  (∃ (t : ℕ), t = 400 ∧ (
    (∃ (n : ℕ), n * (track_length * (speed_B - speed_A)) = t * (speed_A + speed_B) * track_length) ∨
    (∃ (m : ℕ), m * (track_length * (speed_B + speed_A)) = t * (speed_A - speed_B) * track_length))) := 
    sorry

end first_meet_at_starting_point_l1391_139147


namespace staff_members_attended_meeting_l1391_139190

theorem staff_members_attended_meeting
  (n_doughnuts_served : ℕ)
  (e_each_staff_member : ℕ)
  (n_doughnuts_left : ℕ)
  (h1 : n_doughnuts_served = 50)
  (h2 : e_each_staff_member = 2)
  (h3 : n_doughnuts_left = 12) :
  (n_doughnuts_served - n_doughnuts_left) / e_each_staff_member = 19 := 
by
  sorry

end staff_members_attended_meeting_l1391_139190


namespace maximum_tied_teams_round_robin_l1391_139193

noncomputable def round_robin_tournament_max_tied_teams (n : ℕ) : ℕ := 
  sorry

theorem maximum_tied_teams_round_robin (h : n = 8) : round_robin_tournament_max_tied_teams n = 7 :=
sorry

end maximum_tied_teams_round_robin_l1391_139193
