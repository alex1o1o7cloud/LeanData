import Mathlib

namespace sidney_cats_l651_65177

theorem sidney_cats (A : ℕ) :
  (4 * 7 * (3 / 4) + A * 7 = 42) →
  A = 3 :=
by
  intro h
  sorry

end sidney_cats_l651_65177


namespace color_circles_with_four_colors_l651_65175

theorem color_circles_with_four_colors (n : ℕ) (circles : Fin n → (ℝ × ℝ)) (radius : ℝ):
  (∀ i j, i ≠ j → dist (circles i) (circles j) ≥ 2 * radius) →
  ∃ f : Fin n → Fin 4, ∀ i j, dist (circles i) (circles j) < 2 * radius → f i ≠ f j :=
by
  sorry

end color_circles_with_four_colors_l651_65175


namespace initial_innings_count_l651_65154

theorem initial_innings_count (n T L : ℕ) 
  (h1 : T = 50 * n)
  (h2 : 174 = L + 172)
  (h3 : (T - 174 - L) = 48 * (n - 2)) :
  n = 40 :=
by 
  sorry

end initial_innings_count_l651_65154


namespace gcd_sequence_l651_65183

theorem gcd_sequence (n : ℕ) : gcd ((7^n - 1)/6) ((7^(n+1) - 1)/6) = 1 := by
  sorry

end gcd_sequence_l651_65183


namespace lighter_boxes_weight_l651_65123

noncomputable def weight_lighter_boxes (W L H : ℕ) : Prop :=
  L + H = 30 ∧
  (L * W + H * 20) / 30 = 18 ∧
  (H - 15) = 0 ∧
  (15 + L - H = 15 ∧ 15 * 16 = 15 * W)

theorem lighter_boxes_weight :
  ∃ W, ∀ L H, weight_lighter_boxes W L H → W = 16 :=
by sorry

end lighter_boxes_weight_l651_65123


namespace negation_necessary_but_not_sufficient_l651_65166

def P (x : ℝ) : Prop := |x - 2| ≥ 1
def Q (x : ℝ) : Prop := x^2 - 3 * x + 2 ≥ 0

theorem negation_necessary_but_not_sufficient (x : ℝ) :
  (¬ P x → ¬ Q x) ∧ ¬ (¬ Q x → ¬ P x) :=
by
  sorry

end negation_necessary_but_not_sufficient_l651_65166


namespace domain_of_function_l651_65122

def domain_of_f (x: ℝ) : Prop :=
x >= -1 ∧ x <= 48

theorem domain_of_function :
  ∀ x, (x + 1 >= 0 ∧ 7 - Real.sqrt (x + 1) >= 0 ∧ 4 - Real.sqrt (7 - Real.sqrt (x + 1)) >= 0)
  ↔ domain_of_f x := by
  sorry

end domain_of_function_l651_65122


namespace expression_simplification_l651_65104

theorem expression_simplification : 2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := 
by 
  sorry

end expression_simplification_l651_65104


namespace waiter_earning_correct_l651_65186

-- Definitions based on the conditions
def tip1 : ℝ := 25 * 0.15
def tip2 : ℝ := 22 * 0.18
def tip3 : ℝ := 35 * 0.20
def tip4 : ℝ := 30 * 0.10

def total_tips : ℝ := tip1 + tip2 + tip3 + tip4
def commission : ℝ := total_tips * 0.05
def net_tips : ℝ := total_tips - commission

-- Theorem statement
theorem waiter_earning_correct : net_tips = 16.82 := by
  sorry

end waiter_earning_correct_l651_65186


namespace circle_radius_proof_l651_65159

def circle_radius : Prop :=
  let D := -2
  let E := 3
  let F := -3 / 4
  let r := 1 / 2 * Real.sqrt (D^2 + E^2 - 4 * F)
  r = 2

theorem circle_radius_proof : circle_radius :=
  sorry

end circle_radius_proof_l651_65159


namespace find_k_l651_65139

theorem find_k
  (S : ℝ)    -- Distance between the village and city
  (x : ℝ)    -- Speed of the truck in km/h
  (y : ℝ)    -- Speed of the car in km/h
  (H1 : 18 = 0.75 * x - 0.75 * x ^ 2 / (x + y))  -- Condition that truck leaving earlier meets 18 km closer to the city
  (H2 : 24 = x * y / (x + y))      -- Intermediate step from solving the first condition
  : (k = 8) :=    -- We need to show that k = 8
  sorry

end find_k_l651_65139


namespace grasshopper_jump_l651_65128

theorem grasshopper_jump :
  ∃ (x y : ℤ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end grasshopper_jump_l651_65128


namespace distance_between_a_and_c_l651_65192

-- Given conditions
variables (a : ℝ)

-- Statement to prove
theorem distance_between_a_and_c : |a + 1| = |a - (-1)| :=
by sorry

end distance_between_a_and_c_l651_65192


namespace solution_of_inequality_l651_65169

noncomputable def solutionSet (a x : ℝ) : Set ℝ :=
  if a > 0 then {x | -a < x ∧ x < 3 * a}
  else if a < 0 then {x | 3 * a < x ∧ x < -a}
  else ∅

theorem solution_of_inequality (a x : ℝ) :
  (x^2 - 2 * a * x - 3 * a^2 < 0 ↔ x ∈ solutionSet a x) :=
sorry

end solution_of_inequality_l651_65169


namespace increase_circumference_l651_65100

theorem increase_circumference (d1 d2 : ℝ) (increase : ℝ) (P : ℝ) : 
  increase = 2 * Real.pi → 
  P = Real.pi * increase → 
  P = 2 * Real.pi ^ 2 := 
by 
  intros h_increase h_P
  rw [h_P, h_increase]
  sorry

end increase_circumference_l651_65100


namespace expected_value_is_0_point_25_l651_65188

-- Define the probabilities and earnings
def prob_roll_1 := 1/4
def earning_1 := 4
def prob_roll_2 := 1/4
def earning_2 := -3
def prob_roll_3_to_6 := 1/8
def earning_3_to_6 := 0

-- Define the expected value calculation
noncomputable def expected_value : ℝ := 
  (prob_roll_1 * earning_1) + 
  (prob_roll_2 * earning_2) + 
  (prob_roll_3_to_6 * earning_3_to_6) * 4  -- For 3, 4, 5, and 6

-- The theorem to be proved
theorem expected_value_is_0_point_25 : expected_value = 0.25 := by
  sorry

end expected_value_is_0_point_25_l651_65188


namespace files_more_than_apps_l651_65167

-- Defining the initial conditions
def initial_apps : ℕ := 11
def initial_files : ℕ := 3

-- Defining the conditions after some changes
def apps_left : ℕ := 2
def files_left : ℕ := 24

-- Statement to prove
theorem files_more_than_apps : (files_left - apps_left) = 22 := 
by
  sorry

end files_more_than_apps_l651_65167


namespace pens_given_away_l651_65148

theorem pens_given_away (initial_pens : ℕ) (pens_left : ℕ) (n : ℕ) (h1 : initial_pens = 56) (h2 : pens_left = 34) (h3 : n = initial_pens - pens_left) : n = 22 := by
  -- The proof is omitted
  sorry

end pens_given_away_l651_65148


namespace find_focus_parabola_l651_65102

theorem find_focus_parabola
  (x y : ℝ) 
  (h₁ : y = 9 * x^2 + 6 * x - 4) :
  ∃ (h k p : ℝ), (x + 1/3)^2 = 1/3 * (y + 5) ∧ 4 * p = 1/3 ∧ h = -1/3 ∧ k = -5 ∧ (h, k + p) = (-1/3, -59/12) :=
sorry

end find_focus_parabola_l651_65102


namespace coefficient_x3_expansion_l651_65151

open Finset -- To use binomial coefficients and summation

theorem coefficient_x3_expansion (x : ℝ) : 
  (2 + x) ^ 3 = 8 + 12 * x + 6 * x^2 + 1 * x^3 :=
by
  sorry

end coefficient_x3_expansion_l651_65151


namespace equal_real_roots_implies_m_l651_65199

theorem equal_real_roots_implies_m (m : ℝ) : (∃ (x : ℝ), x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) → m = 1/4 :=
by
  sorry

end equal_real_roots_implies_m_l651_65199


namespace number_of_even_ones_matrices_l651_65187

noncomputable def count_even_ones_matrices (m n : ℕ) : ℕ :=
if m = 0 ∨ n = 0 then 1 else 2^((m-1)*(n-1))

theorem number_of_even_ones_matrices (m n : ℕ) : 
  count_even_ones_matrices m n = 2^((m-1)*(n-1)) := sorry

end number_of_even_ones_matrices_l651_65187


namespace min_value_of_sum_squares_l651_65171

theorem min_value_of_sum_squares (a b : ℝ) (h : (9 / a^2) + (4 / b^2) = 1) : a^2 + b^2 ≥ 25 :=
sorry

end min_value_of_sum_squares_l651_65171


namespace total_turtles_l651_65181

variable (Kristen_turtles Kris_turtles Trey_turtles : ℕ)

-- Kristen has 12 turtles
def Kristen_turtles_count : Kristen_turtles = 12 := sorry

-- Kris has 1/4 the number of turtles Kristen has
def Kris_turtles_count (hK : Kristen_turtles = 12) : Kris_turtles = Kristen_turtles / 4 := sorry

-- Trey has 5 times as many turtles as Kris
def Trey_turtles_count (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) : Trey_turtles = 5 * Kris_turtles := sorry

-- Total number of turtles
theorem total_turtles (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) 
  (hT : Trey_turtles = 5 * Kris_turtles) : Kristen_turtles + Kris_turtles + Trey_turtles = 30 := sorry

end total_turtles_l651_65181


namespace investment_ratio_proof_l651_65140

noncomputable def investment_ratio {A_invest B_invest C_invest : ℝ} (profit total_profit : ℝ) (A_times_B : ℝ) : ℝ :=
  C_invest / (A_times_B * B_invest + B_invest + C_invest)

theorem investment_ratio_proof (A_invest B_invest C_invest : ℝ)
  (profit total_profit : ℝ) (A_times_B : ℝ) 
  (h_profit : total_profit = 55000)
  (h_C_share : profit = 15000.000000000002)
  (h_A_times_B : A_times_B = 3)
  (h_ratio_eq : A_times_B * B_invest + B_invest + C_invest = 11 * B_invest / 3) :
  (A_invest / C_invest = 2) :=
by
  sorry

end investment_ratio_proof_l651_65140


namespace solve_for_t_l651_65144

theorem solve_for_t (t : ℝ) (ht : (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4)) : 
  t = -3 := sorry

end solve_for_t_l651_65144


namespace intersection_A_B_l651_65182

def A : Set ℝ := { x | 2 * x^2 - 5 * x < 0 }
def B : Set ℝ := { x | 3^(x - 1) ≥ Real.sqrt 3 }

theorem intersection_A_B : A ∩ B = Set.Ico (3 / 2) (5 / 2) := 
by
  sorry

end intersection_A_B_l651_65182


namespace proof_l651_65174

-- Define proposition p
def p : Prop := ∀ x : ℝ, x < 0 → 2^x > x

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

theorem proof : p ∨ q :=
by
  have hp : p := 
    -- Here, you would provide the proof of p being true.
    sorry
  have hq : ¬ q :=
    -- Here, you would provide the proof of q being false, 
    -- i.e., showing that ∀ x, x^2 + x + 1 ≥ 0.
    sorry
  exact Or.inl hp

end proof_l651_65174


namespace parallelogram_angles_l651_65135

theorem parallelogram_angles (x y : ℝ) (h_sub : y = x + 50) (h_sum : x + y = 180) : x = 65 :=
by
  sorry

end parallelogram_angles_l651_65135


namespace circle_through_A_B_and_tangent_to_m_l651_65179

noncomputable def circle_equation (x y : ℚ) : Prop :=
  x^2 + (y - 1/3)^2 = 16/9

theorem circle_through_A_B_and_tangent_to_m :
  ∃ (c : ℚ × ℚ) (r : ℚ),
    (c = (0, 1/3)) ∧
    (r = 4/3) ∧
    (∀ (x y : ℚ),
      (x = 0 ∧ y = -1 ∨ x = 4/3 ∧ y = 1/3 → (x^2 + (y - 1/3)^2 = 16/9)) ∧
      (x = 4/3 → x = r)) :=
by
  sorry

end circle_through_A_B_and_tangent_to_m_l651_65179


namespace sum_of_sixth_powers_l651_65165

theorem sum_of_sixth_powers (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 0) 
  (h2 : α₁^2 + α₂^2 + α₃^2 = 2) 
  (h3 : α₁^3 + α₂^3 + α₃^3 = 4) : 
  α₁^6 + α₂^6 + α₃^6 = 7 :=
sorry

end sum_of_sixth_powers_l651_65165


namespace solve_for_y_l651_65198

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l651_65198


namespace maximum_value_of_f_l651_65190

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem maximum_value_of_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = (Real.pi / 6) + Real.sqrt 3 ∧ 
  ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f (Real.pi / 6) :=
by
  sorry

end maximum_value_of_f_l651_65190


namespace remainder_of_expression_l651_65130

theorem remainder_of_expression :
  let a := 2^206 + 206
  let b := 2^103 + 2^53 + 1
  a % b = 205 := 
sorry

end remainder_of_expression_l651_65130


namespace student_council_profit_l651_65196

def boxes : ℕ := 48
def erasers_per_box : ℕ := 24
def price_per_eraser : ℝ := 0.75

theorem student_council_profit :
  boxes * erasers_per_box * price_per_eraser = 864 := 
by
  sorry

end student_council_profit_l651_65196


namespace interval_contains_root_l651_65143

noncomputable def f (x : ℝ) : ℝ := 3^x - x^2

theorem interval_contains_root : ∃ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), f x = 0 :=
by
  have f_neg : f (-1) < 0 := by sorry
  have f_zero : f 0 > 0 := by sorry
  sorry

end interval_contains_root_l651_65143


namespace weighted_average_score_l651_65197

def weight (subject_mark : Float) (weight_percentage : Float) : Float :=
    subject_mark * weight_percentage

theorem weighted_average_score :
    (weight 61 0.2) + (weight 65 0.25) + (weight 82 0.3) + (weight 67 0.15) + (weight 85 0.1) = 71.6 := by
    sorry

end weighted_average_score_l651_65197


namespace original_number_is_64_l651_65106

theorem original_number_is_64 (x : ℕ) : 500 + x = 9 * x - 12 → x = 64 :=
by
  sorry

end original_number_is_64_l651_65106


namespace john_burritos_left_l651_65191

def total_burritos (b1 b2 b3 b4 : ℕ) : ℕ :=
  b1 + b2 + b3 + b4

def burritos_left_after_giving_away (total : ℕ) (fraction : ℕ) : ℕ :=
  total - (total / fraction)

def burritos_left_after_eating (burritos_left : ℕ) (burritos_per_day : ℕ) (days : ℕ) : ℕ :=
  burritos_left - (burritos_per_day * days)

theorem john_burritos_left :
  let b1 := 15
  let b2 := 20
  let b3 := 25
  let b4 := 5
  let total := total_burritos b1 b2 b3 b4
  let burritos_after_give_away := burritos_left_after_giving_away total 3
  let burritos_after_eating := burritos_left_after_eating burritos_after_give_away 3 10
  burritos_after_eating = 14 :=
by
  sorry

end john_burritos_left_l651_65191


namespace find_other_number_l651_65178

theorem find_other_number (a b : ℤ) (h1 : 2 * a + 3 * b = 100) (h2 : a = 28 ∨ b = 28) : a = 8 ∨ b = 8 :=
sorry

end find_other_number_l651_65178


namespace max_total_weight_of_chocolates_l651_65152

theorem max_total_weight_of_chocolates 
  (A B C : ℕ)
  (hA : A ≤ 100)
  (hBC : B - C ≤ 100)
  (hC : C ≤ 100)
  (h_distribute : A ≤ 100 ∧ (B - C) ≤ 100)
  : (A + B = 300) :=
by 
  sorry

end max_total_weight_of_chocolates_l651_65152


namespace hilt_books_transaction_difference_l651_65141

noncomputable def total_cost_paid (original_price : ℝ) (num_first_books : ℕ) (discount1 : ℝ) (num_second_books : ℕ) (discount2 : ℝ) : ℝ :=
  let cost_first_books := num_first_books * original_price * (1 - discount1)
  let cost_second_books := num_second_books * original_price * (1 - discount2)
  cost_first_books + cost_second_books

noncomputable def total_sale_amount (sale_price : ℝ) (interest_rate : ℝ) (num_books : ℕ) : ℝ :=
  let compounded_price := sale_price * (1 + interest_rate) ^ 1
  compounded_price * num_books

theorem hilt_books_transaction_difference : 
  let original_price := 11
  let num_first_books := 10
  let discount1 := 0.20
  let num_second_books := 5
  let discount2 := 0.25
  let sale_price := 25
  let interest_rate := 0.05
  let num_books := 15
  total_sale_amount sale_price interest_rate num_books - total_cost_paid original_price num_first_books discount1 num_second_books discount2 = 264.50 :=
by
  sorry

end hilt_books_transaction_difference_l651_65141


namespace output_correct_l651_65195

-- Define the initial values and assignments
def initial_a : ℕ := 1
def initial_b : ℕ := 2
def initial_c : ℕ := 3

-- Perform the assignments in sequence
def after_c_assignment : ℕ := initial_b
def after_b_assignment : ℕ := initial_a
def after_a_assignment : ℕ := after_c_assignment

-- Final values after all assignments
def final_a := after_a_assignment
def final_b := after_b_assignment
def final_c := after_c_assignment

-- Theorem statement
theorem output_correct :
  final_a = 2 ∧ final_b = 1 ∧ final_c = 2 :=
by {
  -- Proof is omitted
  sorry
}

end output_correct_l651_65195


namespace computer_additions_per_hour_l651_65119

def operations_per_second : ℕ := 15000
def additions_per_second : ℕ := operations_per_second / 2
def seconds_per_hour : ℕ := 3600

theorem computer_additions_per_hour : 
  additions_per_second * seconds_per_hour = 27000000 := by
  sorry

end computer_additions_per_hour_l651_65119


namespace phoenix_hike_length_l651_65189

theorem phoenix_hike_length (a b c d : ℕ)
  (h1 : a + b = 22)
  (h2 : b + c = 26)
  (h3 : c + d = 30)
  (h4 : a + c = 26) :
  a + b + c + d = 52 :=
sorry

end phoenix_hike_length_l651_65189


namespace find_coins_l651_65118

-- Definitions based on conditions
structure Wallet where
  coin1 : ℕ
  coin2 : ℕ
  h_total_value : coin1 + coin2 = 15
  h_not_five : coin1 ≠ 5 ∨ coin2 ≠ 5

-- Theorem statement based on the proof problem
theorem find_coins (w : Wallet) : (w.coin1 = 5 ∧ w.coin2 = 10) ∨ (w.coin1 = 10 ∧ w.coin2 = 5) := by
  sorry

end find_coins_l651_65118


namespace candy_sampling_l651_65111

theorem candy_sampling (total_customers caught_sampling not_caught_sampling : ℝ) :
  caught_sampling = 0.22 * total_customers →
  not_caught_sampling = 0.12 * (total_customers * sampling_percent) →
  (sampling_percent * total_customers = caught_sampling / 0.78) :=
by
  intros h1 h2
  sorry

end candy_sampling_l651_65111


namespace expand_product_l651_65149

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x ^ 2 + 52 * x + 84 := by
  sorry

end expand_product_l651_65149


namespace find_difference_l651_65160

-- Define the necessary constants and variables
variables (u v : ℝ)

-- Define the conditions
def condition1 := u + v = 360
def condition2 := u = (1/1.1) * v

-- Define the theorem to prove
theorem find_difference (h1 : condition1 u v) (h2 : condition2 u v) : v - u = 17 := 
sorry

end find_difference_l651_65160


namespace original_side_length_l651_65134

theorem original_side_length (x : ℝ) (h1 : (x - 6) * (x - 5) = 120) : x = 15 :=
sorry

end original_side_length_l651_65134


namespace triangle_inequality_l651_65193

theorem triangle_inequality (a b c Δ : ℝ) (h_Δ: Δ = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt (3) * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 :=
by
  sorry

end triangle_inequality_l651_65193


namespace tangent_from_origin_l651_65126

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (6, 14)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function that computes the length of the tangent from O to the circle passing through A, B, and C
noncomputable def tangent_length : ℝ :=
 sorry -- Placeholder for the actual calculation

-- The theorem we need to prove: The length of the tangent from O to the circle passing through A, B, and C is as calculated
theorem tangent_from_origin (L : ℕ) : 
  tangent_length = L := 
 sorry -- Placeholder for the proof

end tangent_from_origin_l651_65126


namespace mary_needs_6_cups_of_flour_l651_65136

-- Define the necessary constants according to the conditions.
def flour_needed : ℕ := 6
def sugar_needed : ℕ := 13
def flour_more_than_sugar : ℕ := 8

-- Define the number of cups of flour Mary needs to add.
def flour_to_add (flour_put_in : ℕ) : ℕ := flour_needed - flour_put_in

-- Prove that Mary needs to add 6 more cups of flour.
theorem mary_needs_6_cups_of_flour (flour_put_in : ℕ) (h : flour_more_than_sugar = 8): flour_to_add flour_put_in = 6 :=
by {
  sorry -- the proof is omitted.
}

end mary_needs_6_cups_of_flour_l651_65136


namespace Mika_stickers_l651_65158

theorem Mika_stickers
  (initial_stickers : ℕ)
  (bought_stickers : ℕ)
  (received_stickers : ℕ)
  (given_stickers : ℕ)
  (used_stickers : ℕ)
  (final_stickers : ℕ) :
  initial_stickers = 45 →
  bought_stickers = 53 →
  received_stickers = 35 →
  given_stickers = 19 →
  used_stickers = 86 →
  final_stickers = initial_stickers + bought_stickers + received_stickers - given_stickers - used_stickers →
  final_stickers = 28 :=
by
  intros
  sorry

end Mika_stickers_l651_65158


namespace lana_total_spending_l651_65162

noncomputable def general_admission_cost : ℝ := 6
noncomputable def vip_cost : ℝ := 10
noncomputable def premium_cost : ℝ := 15

noncomputable def num_general_admission_tickets : ℕ := 6
noncomputable def num_vip_tickets : ℕ := 2
noncomputable def num_premium_tickets : ℕ := 1

noncomputable def discount_general_admission : ℝ := 0.10
noncomputable def discount_vip : ℝ := 0.15

noncomputable def total_spending (gen_cost : ℝ) (vip_cost : ℝ) (prem_cost : ℝ) (gen_num : ℕ) (vip_num : ℕ) (prem_num : ℕ) (gen_disc : ℝ) (vip_disc : ℝ) : ℝ :=
  let general_cost := gen_cost * gen_num
  let general_discount := general_cost * gen_disc
  let discounted_general_cost := general_cost - general_discount
  let vip_cost_total := vip_cost * vip_num
  let vip_discount := vip_cost_total * vip_disc
  let discounted_vip_cost := vip_cost_total - vip_discount
  let premium_cost_total := prem_cost * prem_num
  discounted_general_cost + discounted_vip_cost + premium_cost_total

theorem lana_total_spending : total_spending general_admission_cost vip_cost premium_cost num_general_admission_tickets num_vip_tickets num_premium_tickets discount_general_admission discount_vip = 64.40 := 
sorry

end lana_total_spending_l651_65162


namespace moles_NaHCO3_combined_l651_65172

-- Define conditions as given in the problem
def moles_HNO3_combined := 1
def moles_NaNO3_result := 1

-- The chemical equation as a definition
def balanced_reaction (moles_NaHCO3 moles_HNO3 moles_NaNO3 : ℕ) : Prop :=
  moles_HNO3 = moles_NaNO3 ∧ moles_NaHCO3 = moles_HNO3

-- The proof problem statement
theorem moles_NaHCO3_combined :
  balanced_reaction 1 moles_HNO3_combined moles_NaNO3_result → 1 = 1 :=
by 
  sorry

end moles_NaHCO3_combined_l651_65172


namespace lcm_gcd_pairs_l651_65131

theorem lcm_gcd_pairs (a b : ℕ) :
  (lcm a b + gcd a b = (a * b) / 5) ↔
  (a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6) :=
sorry

end lcm_gcd_pairs_l651_65131


namespace fathers_age_more_than_three_times_son_l651_65117

variable (F S x : ℝ)

theorem fathers_age_more_than_three_times_son :
  F = 27 →
  F = 3 * S + x →
  F + 3 = 2 * (S + 3) + 8 →
  x = 3 :=
by
  intros hF h1 h2
  sorry

end fathers_age_more_than_three_times_son_l651_65117


namespace sum_coefficients_l651_65185

theorem sum_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℚ) :
  (1 - 2 * (1 : ℚ))^5 = a_0 + a_1 * (1 : ℚ) + a_2 * (1 : ℚ)^2 + a_3 * (1 : ℚ)^3 + a_4 * (1 : ℚ)^4 + a_5 * (1 : ℚ)^5 →
  (1 - 2 * (0 : ℚ))^5 = a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -2 :=
by
  sorry

end sum_coefficients_l651_65185


namespace crayons_loss_l651_65107

def initial_crayons : ℕ := 479
def final_crayons : ℕ := 134
def crayons_lost : ℕ := initial_crayons - final_crayons

theorem crayons_loss :
  crayons_lost = 345 := by
  sorry

end crayons_loss_l651_65107


namespace Sheila_attends_picnic_probability_l651_65150

theorem Sheila_attends_picnic_probability :
  let P_rain := 0.5
  let P_no_rain := 0.5
  let P_Sheila_goes_if_rain := 0.3
  let P_Sheila_goes_if_no_rain := 0.7
  let P_friend_agrees := 0.5
  (P_rain * P_Sheila_goes_if_rain + P_no_rain * P_Sheila_goes_if_no_rain) * P_friend_agrees = 0.25 := 
by
  sorry

end Sheila_attends_picnic_probability_l651_65150


namespace comparison1_comparison2_comparison3_l651_65125

theorem comparison1 : -3.2 > -4.3 :=
by sorry

theorem comparison2 : (1 : ℚ) / 2 > -(1 / 3) :=
by sorry

theorem comparison3 : (1 : ℚ) / 4 > 0 :=
by sorry

end comparison1_comparison2_comparison3_l651_65125


namespace number_of_terms_in_sequence_l651_65176

theorem number_of_terms_in_sequence :
  ∃ n : ℕ, (1 + 4 * (n - 1) = 2025) ∧ n = 507 := by
  sorry

end number_of_terms_in_sequence_l651_65176


namespace trey_uses_47_nails_l651_65156

variable (D : ℕ) -- total number of decorations
variable (nails thumbtacks sticky_strips : ℕ)

-- Conditions
def uses_nails := nails = (5 * D) / 8
def uses_thumbtacks := thumbtacks = (9 * D) / 80
def uses_sticky_strips := sticky_strips = 20
def total_decorations := (21 * D) / 80 = 20

-- Question: Prove that Trey uses 47 nails when the conditions hold
theorem trey_uses_47_nails (D : ℕ) (h1 : uses_nails D nails) (h2 : uses_thumbtacks D thumbtacks) (h3 : uses_sticky_strips sticky_strips) (h4 : total_decorations D) : nails = 47 :=  
by
  sorry

end trey_uses_47_nails_l651_65156


namespace total_hours_charged_l651_65108

variables (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) 
                            (h2 : P = (1/3 : ℚ) * (M : ℚ)) 
                            (h3 : M = K + 85) : 
  K + P + M = 153 := 
by 
  sorry

end total_hours_charged_l651_65108


namespace problem_proof_l651_65155

theorem problem_proof (x : ℝ) (h : x + 1/x = 3) : (x - 3) ^ 2 + 36 / (x - 3) ^ 2 = 12 :=
sorry

end problem_proof_l651_65155


namespace fraction_of_pizza_covered_by_pepperoni_l651_65124

theorem fraction_of_pizza_covered_by_pepperoni :
  ∀ (d_pizza d_pepperoni : ℝ) (n_pepperoni : ℕ) (overlap_fraction : ℝ),
  d_pizza = 16 ∧ d_pepperoni = d_pizza / 8 ∧ n_pepperoni = 32 ∧ overlap_fraction = 0.25 →
  (π * d_pepperoni^2 / 4 * (1 - overlap_fraction) * n_pepperoni) / (π * (d_pizza / 2)^2) = 3 / 8 :=
by
  intro d_pizza d_pepperoni n_pepperoni overlap_fraction
  intro h
  sorry

end fraction_of_pizza_covered_by_pepperoni_l651_65124


namespace shaded_area_square_l651_65133

theorem shaded_area_square (s : ℝ) (r : ℝ) (A : ℝ) :
  s = 4 ∧ r = 2 * Real.sqrt 2 → A = s^2 - 4 * (π * r^2 / 2) → A = 8 - 2 * π :=
by
  intros h₁ h₂
  sorry

end shaded_area_square_l651_65133


namespace sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l651_65120

open Real

noncomputable def problem_conditions (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧
  cos α = 3/5 ∧ cos (β + α) = 5/13

theorem sin_beta_value 
  {α β : ℝ} (h : problem_conditions α β) : 
  sin β = 16 / 65 :=
sorry

theorem sin2alpha_over_cos2alpha_plus_cos2alpha_value
  {α β : ℝ} (h : problem_conditions α β) : 
  (sin (2 * α)) / (cos α^2 + cos (2 * α)) = 12 :=
sorry

end sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l651_65120


namespace mackenzie_new_disks_l651_65138

noncomputable def price_new (U N : ℝ) : Prop := 6 * N + 2 * U = 127.92

noncomputable def disks_mackenzie_buys (U N x : ℝ) : Prop := x * N + 8 * U = 133.89

theorem mackenzie_new_disks (U N x : ℝ) (h1 : U = 9.99) (h2 : price_new U N) (h3 : disks_mackenzie_buys U N x) :
  x = 3 :=
by
  sorry

end mackenzie_new_disks_l651_65138


namespace calc_exponent_l651_65184

theorem calc_exponent (a b : ℕ) : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end calc_exponent_l651_65184


namespace james_coffee_weekdays_l651_65170

theorem james_coffee_weekdays :
  ∃ (c d : ℕ) (k : ℤ), (c + d = 5) ∧ 
                      (3 * c + 2 * d + 10 = k / 3) ∧ 
                      (k % 3 = 0) ∧ 
                      c = 2 :=
by 
  sorry

end james_coffee_weekdays_l651_65170


namespace find_n_l651_65132

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n + 4

-- Define the condition a_n = 13
def condition (n : ℕ) : Prop := a n = 13

-- Prove that under this condition, n = 3
theorem find_n (n : ℕ) (h : condition n) : n = 3 :=
by {
  sorry
}

end find_n_l651_65132


namespace max_cars_div_10_l651_65112

noncomputable def max_cars (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) : ℕ :=
  let k := 2000
  2000 -- Maximum number of cars passing the sensor

theorem max_cars_div_10 (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) :
  car_length = 5 →
  (∀ k : ℕ, distance_for_speed k = k) →
  (∀ k : ℕ, speed k = 10 * k) →
  (max_cars car_length distance_for_speed speed) = 2000 → 
  (max_cars car_length distance_for_speed speed) / 10 = 200 := by
  intros
  sorry

end max_cars_div_10_l651_65112


namespace final_population_l651_65129

theorem final_population (P0 : ℕ) (r1 r2 : ℝ) (P2 : ℝ) 
  (h0 : P0 = 1000)
  (h1 : r1 = 1.20)
  (h2 : r2 = 1.30)
  (h3 : P2 = P0 * r1 * r2) : 
  P2 = 1560 := 
sorry

end final_population_l651_65129


namespace employee_b_pay_l651_65137

theorem employee_b_pay (total_pay : ℝ) (ratio_ab : ℝ) (pay_b : ℝ) 
  (h1 : total_pay = 570)
  (h2 : ratio_ab = 1.5 * pay_b)
  (h3 : total_pay = ratio_ab + pay_b) :
  pay_b = 228 := 
sorry

end employee_b_pay_l651_65137


namespace y_payment_is_approximately_272_73_l651_65114

noncomputable def calc_y_payment : ℝ :=
  let total_payment : ℝ := 600
  let percent_x_to_y : ℝ := 1.2
  total_payment / (percent_x_to_y + 1)

theorem y_payment_is_approximately_272_73
  (total_payment : ℝ)
  (percent_x_to_y : ℝ)
  (h1 : total_payment = 600)
  (h2 : percent_x_to_y = 1.2) :
  calc_y_payment = 272.73 :=
by
  sorry

end y_payment_is_approximately_272_73_l651_65114


namespace problem_area_of_circle_l651_65173

noncomputable def circleAreaPortion : ℝ :=
  let r := Real.sqrt 59
  let theta := 135 * Real.pi / 180
  (theta / (2 * Real.pi)) * (Real.pi * r^2)

theorem problem_area_of_circle :
  circleAreaPortion = (177 / 8) * Real.pi := by
  sorry

end problem_area_of_circle_l651_65173


namespace sum_of_interior_angles_n_plus_4_l651_65145

    noncomputable def sum_of_interior_angles (sides : ℕ) : ℝ :=
      180 * (sides - 2)

    theorem sum_of_interior_angles_n_plus_4 (n : ℕ) (h : sum_of_interior_angles n = 2340) :
      sum_of_interior_angles (n + 4) = 3060 :=
    by
      sorry
    
end sum_of_interior_angles_n_plus_4_l651_65145


namespace kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l651_65105

/-- Conditions: -/
def kitchen_clock_gain_rate : ℝ := 1.5 -- minutes per hour
def bedroom_clock_lose_rate : ℝ := 0.5 -- minutes per hour
def synchronization_time : ℝ := 0 -- time in hours when both clocks were correct

/-- Problem 1: -/
theorem kitchen_clock_correct_again :
  ∃ t : ℝ, 1.5 * t = 720 :=
by {
  sorry
}

/-- Problem 2: -/
theorem bedroom_clock_correct_again :
  ∃ t : ℝ, 0.5 * t = 720 :=
by {
  sorry
}

/-- Problem 3: -/
theorem both_clocks_same_time_again :
  ∃ t : ℝ, 2 * t = 720 :=
by {
  sorry
}

end kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l651_65105


namespace find_c_l651_65142

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end find_c_l651_65142


namespace comparison_17_pow_14_31_pow_11_l651_65121

theorem comparison_17_pow_14_31_pow_11 : 17^14 > 31^11 :=
by
  sorry

end comparison_17_pow_14_31_pow_11_l651_65121


namespace car_passing_problem_l651_65110

noncomputable def maxCarsPerHourDividedBy10 : ℕ :=
  let unit_length (n : ℕ) := 5 * (n + 1)
  let cars_passed_in_one_hour (n : ℕ) := 10000 * n / unit_length n
  Nat.div (2000) (10)

theorem car_passing_problem : maxCarsPerHourDividedBy10 = 200 :=
  by
  sorry

end car_passing_problem_l651_65110


namespace theresa_sons_count_l651_65194

theorem theresa_sons_count (total_meat_left : ℕ) (meat_per_plate : ℕ) (frac_left : ℚ) (s : ℕ) :
  total_meat_left = meat_per_plate ∧ meat_per_plate * frac_left * s = 3 → s = 9 :=
by sorry

end theresa_sons_count_l651_65194


namespace percentage_value_l651_65103

variables {P a b c : ℝ}

theorem percentage_value (h1 : (P / 100) * a = 12) (h2 : (12 / 100) * b = 6) (h3 : c = b / a) : c = P / 24 :=
by
  sorry

end percentage_value_l651_65103


namespace greatest_possible_value_y_l651_65153

theorem greatest_possible_value_y
  (x y : ℤ)
  (h : x * y + 3 * x + 2 * y = -6) : 
  y ≤ 3 :=
by sorry

end greatest_possible_value_y_l651_65153


namespace find_second_candy_cost_l651_65127

theorem find_second_candy_cost :
  ∃ (x : ℝ), 
    (15 * 8 + 30 * x = 45 * 6) ∧
    x = 5 := by
  sorry

end find_second_candy_cost_l651_65127


namespace value_of_n_l651_65157

theorem value_of_n (n : ℤ) :
  (∀ x : ℤ, (x + n) * (x + 2) = x^2 + 2 * x + n * x + 2 * n → 2 + n = 0) → n = -2 := 
by
  intro h
  have h1 := h 0
  sorry

end value_of_n_l651_65157


namespace depth_of_tunnel_l651_65161

theorem depth_of_tunnel (a b area : ℝ) (h := (2 * area) / (a + b)) (ht : a = 15) (hb : b = 5) (ha : area = 400) :
  h = 40 :=
by
  sorry

end depth_of_tunnel_l651_65161


namespace solve_for_x_l651_65101

theorem solve_for_x (x : ℝ) :
  (x - 2)^6 + (x - 6)^6 = 64 → x = 3 ∨ x = 5 :=
by
  intros h
  sorry

end solve_for_x_l651_65101


namespace Julie_simple_interest_l651_65146

variable (S : ℝ) (r : ℝ) (A : ℝ) (C : ℝ)

def initially_savings (S : ℝ) := S = 784
def half_savings_in_each_account (S A : ℝ) := A = S / 2
def compound_interest_after_two_years (A r : ℝ) := A * (1 + r)^2 - A = 120

theorem Julie_simple_interest
  (S : ℝ) (r : ℝ) (A : ℝ)
  (h1 : initially_savings S)
  (h2 : half_savings_in_each_account S A)
  (h3 : compound_interest_after_two_years A r) :
  A * r * 2 = 112 :=
by 
  sorry

end Julie_simple_interest_l651_65146


namespace average_mark_of_second_class_l651_65147

/-- 
There is a class of 30 students with an average mark of 40. 
Another class has 50 students with an unknown average mark. 
The average marks of all students combined is 65. 
Prove that the average mark of the second class is 80.
-/
theorem average_mark_of_second_class (x : ℝ) (h1 : 30 * 40 + 50 * x = 65 * (30 + 50)) : x = 80 := 
sorry

end average_mark_of_second_class_l651_65147


namespace solve_for_y_l651_65168

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l651_65168


namespace problem_statement_l651_65109

variables {a b x : ℝ}

theorem problem_statement (h1 : x = a / b + 2) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + 2 * b) / (a - 2 * b) = x / (x - 4) := 
sorry

end problem_statement_l651_65109


namespace vehicle_value_this_year_l651_65164

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end vehicle_value_this_year_l651_65164


namespace equivalent_condition_for_continuity_l651_65113

theorem equivalent_condition_for_continuity {x c d : ℝ} (g : ℝ → ℝ) (h1 : g x = 5 * x - 3) (h2 : ∀ x, |g x - 1| < c → |x - 1| < d) (hc : c > 0) (hd : d > 0) : d ≤ c / 5 :=
sorry

end equivalent_condition_for_continuity_l651_65113


namespace impossible_result_l651_65116

noncomputable def f (a b : ℝ) (c : ℤ) (x : ℝ) : ℝ :=
  a * Real.sin x + b * x + c

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬(f a b c 1 = 1 ∧ f a b c (-1) = 2) :=
by {
  sorry
}

end impossible_result_l651_65116


namespace cargo_arrival_day_l651_65115

-- Definitions based on conditions
def navigation_days : Nat := 21
def customs_days : Nat := 4
def warehouse_days_from_today : Nat := 2
def departure_days_ago : Nat := 30

-- Definition represents the total transit time
def total_transit_days : Nat := navigation_days + customs_days + warehouse_days_from_today

-- Theorem to prove the cargo always arrives at the rural warehouse 1 day after leaving the port in Vancouver
theorem cargo_arrival_day : 
  (departure_days_ago - total_transit_days + warehouse_days_from_today = 1) :=
by
  -- Placeholder for the proof
  sorry

end cargo_arrival_day_l651_65115


namespace tod_trip_time_l651_65180

noncomputable def total_time (d1 d2 d3 d4 s1 s2 s3 s4 : ℝ) : ℝ :=
  d1 / s1 + d2 / s2 + d3 / s3 + d4 / s4

theorem tod_trip_time :
  total_time 55 95 30 75 40 50 20 60 = 6.025 :=
by 
  sorry

end tod_trip_time_l651_65180


namespace competition_winner_is_C_l651_65163

-- Define the type for singers
inductive Singer
| A | B | C | D
deriving DecidableEq

-- Assume each singer makes a statement
def statement (s : Singer) : Prop :=
  match s with
  | Singer.A => Singer.B ≠ Singer.C
  | Singer.B => Singer.A ≠ Singer.C
  | Singer.C => true
  | Singer.D => Singer.B ≠ Singer.D

-- Define that two and only two statements are true
def exactly_two_statements_are_true : Prop :=
  (statement Singer.A ∧ statement Singer.C ∧ ¬statement Singer.B ∧ ¬statement Singer.D) ∨
  (statement Singer.A ∧ statement Singer.D ∧ ¬statement Singer.B ∧ ¬statement Singer.C)

-- Define the winner
def winner : Singer := Singer.C

-- The main theorem to be proved
theorem competition_winner_is_C :
  exactly_two_statements_are_true → (winner = Singer.C) :=
by
  intro h
  exact sorry

end competition_winner_is_C_l651_65163
