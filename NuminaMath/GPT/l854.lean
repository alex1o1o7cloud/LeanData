import Mathlib

namespace juanitas_dessert_cost_is_correct_l854_85439

noncomputable def brownie_cost := 2.50
noncomputable def regular_scoop_cost := 1.00
noncomputable def premium_scoop_cost := 1.25
noncomputable def deluxe_scoop_cost := 1.50
noncomputable def syrup_cost := 0.50
noncomputable def nuts_cost := 1.50
noncomputable def whipped_cream_cost := 0.75
noncomputable def cherry_cost := 0.25
noncomputable def discount_tuesday := 0.10

noncomputable def total_cost_of_juanitas_dessert :=
    let discounted_brownie := brownie_cost * (1 - discount_tuesday)
    let ice_cream_cost := 2 * regular_scoop_cost + premium_scoop_cost
    let syrup_total := 2 * syrup_cost
    let additional_toppings := nuts_cost + whipped_cream_cost + cherry_cost
    discounted_brownie + ice_cream_cost + syrup_total + additional_toppings
   
theorem juanitas_dessert_cost_is_correct:
  total_cost_of_juanitas_dessert = 9.00 := by
  sorry

end juanitas_dessert_cost_is_correct_l854_85439


namespace shortest_tree_height_is_correct_l854_85434

-- Definitions of the tree heights
def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

-- Theorem statement
theorem shortest_tree_height_is_correct :
  shortest_tree_height = 50 :=
by
  sorry

end shortest_tree_height_is_correct_l854_85434


namespace determinant_is_zero_l854_85447

-- Define the matrix
def my_matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, x + z, y - z],
    ![1, x + y + z, y - z],
    ![1, x + z, x + y]]

-- Define the property to prove
theorem determinant_is_zero (x y z : ℝ) :
  Matrix.det (my_matrix x y z) = 0 :=
by sorry

end determinant_is_zero_l854_85447


namespace sin_product_ge_one_l854_85486

theorem sin_product_ge_one (x : ℝ) (n : ℤ) :
  (∀ α, |Real.sin α| ≤ 1) →
  ∀ x,
  (Real.sin x) * (Real.sin (1755 * x)) * (Real.sin (2011 * x)) ≥ 1 ↔
  ∃ n : ℤ, x = π / 2 + 2 * π * n := by {
    sorry
}

end sin_product_ge_one_l854_85486


namespace spherical_coordinates_neg_z_l854_85467

theorem spherical_coordinates_neg_z (x y z : ℝ) (h₀ : ρ = 5) (h₁ : θ = 3 * Real.pi / 4) (h₂ : φ = Real.pi / 3)
  (hx : x = ρ * Real.sin φ * Real.cos θ) 
  (hy : y = ρ * Real.sin φ * Real.sin θ) 
  (hz : z = ρ * Real.cos φ) : 
  (ρ, θ, π - φ) = (5, 3 * Real.pi / 4, 2 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_neg_z_l854_85467


namespace belfried_payroll_l854_85402

noncomputable def tax_paid (payroll : ℝ) : ℝ :=
  if payroll < 200000 then 0 else 0.002 * (payroll - 200000)

theorem belfried_payroll (payroll : ℝ) (h : tax_paid payroll = 400) : payroll = 400000 :=
by
  sorry

end belfried_payroll_l854_85402


namespace minimum_g_value_l854_85458

noncomputable def g (x : ℝ) := (9 * x^2 + 18 * x + 20) / (4 * (2 + x))

theorem minimum_g_value :
  ∀ x ≥ (1 : ℝ), g x = (47 / 16) := sorry

end minimum_g_value_l854_85458


namespace mirella_orange_books_read_l854_85471

-- Definitions based on the conditions in a)
def purpleBookPages : ℕ := 230
def orangeBookPages : ℕ := 510
def purpleBooksRead : ℕ := 5
def extraOrangePages : ℕ := 890

-- The total number of purple pages read
def purplePagesRead := purpleBooksRead * purpleBookPages

-- The number of orange books read
def orangeBooksRead (O : ℕ) := O * orangeBookPages

-- Statement to be proved
theorem mirella_orange_books_read (O : ℕ) :
  orangeBooksRead O = purplePagesRead + extraOrangePages → O = 4 :=
by
  sorry

end mirella_orange_books_read_l854_85471


namespace distance_MN_is_2R_l854_85469

-- Definitions for the problem conditions
variable (R : ℝ) (A B C M N : ℝ) (alpha : ℝ)
variable (AC AB : ℝ)

-- Assumptions based on the problem statement
axiom circle_radius (r : ℝ) : r = R
axiom chord_length_AC (ch_AC : ℝ) : ch_AC = AC
axiom chord_length_AB (ch_AB : ℝ) : ch_AB = AB
axiom distance_M_to_AC (d_M_AC : ℝ) : d_M_AC = AC
axiom distance_N_to_AB (d_N_AB : ℝ) : d_N_AB = AB
axiom angle_BAC (ang_BAC : ℝ) : ang_BAC = alpha

-- To prove: the distance between M and N is 2R
theorem distance_MN_is_2R : |MN| = 2 * R := sorry

end distance_MN_is_2R_l854_85469


namespace triangle_third_side_l854_85490

theorem triangle_third_side (AB AC AD : ℝ) (hAB : AB = 25) (hAC : AC = 30) (hAD : AD = 24) :
  ∃ BC : ℝ, (BC = 25 ∨ BC = 11) :=
by
  sorry

end triangle_third_side_l854_85490


namespace range_of_y_l854_85405

theorem range_of_y :
  ∀ (y x : ℝ), x = 4 - y → (-2 ≤ x ∧ x ≤ -1) → (5 ≤ y ∧ y ≤ 6) :=
by
  intros y x h1 h2
  sorry

end range_of_y_l854_85405


namespace compound_interest_rate_l854_85432

theorem compound_interest_rate
(SI : ℝ) (CI : ℝ) (P1 : ℝ) (r : ℝ) (t1 t2 : ℕ) (P2 R : ℝ)
(h1 : SI = (P1 * r * t1) / 100)
(h2 : SI = CI / 2)
(h3 : CI = P2 * (1 + R / 100) ^ t2 - P2)
(h4 : P1 = 3500)
(h5 : r = 6)
(h6 : t1 = 2)
(h7 : P2 = 4000)
(h8 : t2 = 2) : R = 10 := by
  sorry

end compound_interest_rate_l854_85432


namespace range_of_f_l854_85418

noncomputable def f (x : ℝ) : ℝ := 2^x
def valid_range (S : Set ℝ) : Prop := ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), f x ∈ S

theorem range_of_f : valid_range (Set.Icc (1 : ℝ) (8 : ℝ)) :=
sorry

end range_of_f_l854_85418


namespace total_market_cost_l854_85453

-- Defining the variables for the problem
def pounds_peaches : Nat := 5 * 3
def pounds_apples : Nat := 4 * 3
def pounds_blueberries : Nat := 3 * 3

def cost_per_pound_peach := 2
def cost_per_pound_apple := 1
def cost_per_pound_blueberry := 1

-- Defining the total costs
def cost_peaches : Nat := pounds_peaches * cost_per_pound_peach
def cost_apples : Nat := pounds_apples * cost_per_pound_apple
def cost_blueberries : Nat := pounds_blueberries * cost_per_pound_blueberry

-- Total cost
def total_cost : Nat := cost_peaches + cost_apples + cost_blueberries

-- Theorem to prove the total cost is $51.00
theorem total_market_cost : total_cost = 51 := by
  sorry

end total_market_cost_l854_85453


namespace dolls_count_l854_85463

theorem dolls_count (lisa_dolls : ℕ) (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ)
  (h1 : vera_dolls = 2 * lisa_dolls)
  (h2 : sophie_dolls = 2 * vera_dolls)
  (h3 : aida_dolls = 2 * sophie_dolls)
  (hl : lisa_dolls = 20) :
  aida_dolls + sophie_dolls + vera_dolls + lisa_dolls = 300 :=
by
  sorry

end dolls_count_l854_85463


namespace taco_variants_count_l854_85404

theorem taco_variants_count :
  let toppings := 8
  let meat_variants := 3
  let shell_variants := 2
  2 ^ toppings * meat_variants * shell_variants = 1536 := by
sorry

end taco_variants_count_l854_85404


namespace sum_geometric_sequence_l854_85456

theorem sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_a1 : a 1 = 1)
  (h_arithmetic : 4 * a 2 + a 4 = 2 * a 3) : 
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_geometric_sequence_l854_85456


namespace quarters_dimes_equivalence_l854_85473

theorem quarters_dimes_equivalence (m : ℕ) (h : 25 * 30 + 10 * 20 = 25 * 15 + 10 * m) : m = 58 :=
by
  sorry

end quarters_dimes_equivalence_l854_85473


namespace no_seating_in_four_consecutive_seats_l854_85497

theorem no_seating_in_four_consecutive_seats :
  let total_arrangements := Nat.factorial 10
  let grouped_arrangements := Nat.factorial 7 * Nat.factorial 4
  let acceptable_arrangements := total_arrangements - grouped_arrangements
  acceptable_arrangements = 3507840 :=
by
  sorry

end no_seating_in_four_consecutive_seats_l854_85497


namespace min_value_expression_l854_85400

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 :=
by
  sorry

end min_value_expression_l854_85400


namespace total_points_after_3_perfect_games_l854_85480

def perfect_score := 21
def number_of_games := 3

theorem total_points_after_3_perfect_games : perfect_score * number_of_games = 63 := 
by
  sorry

end total_points_after_3_perfect_games_l854_85480


namespace table_tennis_basketball_teams_l854_85459

theorem table_tennis_basketball_teams (X Y : ℕ)
  (h1 : X + Y = 50) 
  (h2 : 7 * Y = 3 * X)
  (h3 : 2 * (X - 8) = 3 * (Y + 8)) :
  X = 35 ∧ Y = 15 :=
by
  sorry

end table_tennis_basketball_teams_l854_85459


namespace range_of_a_l854_85413

  variable {A : Set ℝ} {B : Set ℝ}
  variable {a : ℝ}

  def A_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 2 * a - 4 }
  def B_def : Set ℝ := { x | -1 < x ∧ x < 6 }

  theorem range_of_a (h : A_def a ∩ B_def = A_def a) : a < 5 :=
  sorry
  
end range_of_a_l854_85413


namespace geometric_progression_nonzero_k_l854_85438

theorem geometric_progression_nonzero_k (k : ℝ) : k ≠ 0 ↔ (40*k)^2 = (10*k) * (160*k) := by sorry

end geometric_progression_nonzero_k_l854_85438


namespace partition_exists_iff_l854_85485

theorem partition_exists_iff (k : ℕ) :
  (∃ (A B : Finset ℕ), A ∪ B = Finset.range (1990 + k + 1) ∧ A ∩ B = ∅ ∧ 
  (A.sum id + 1990 * A.card = B.sum id + 1990 * B.card)) ↔ 
  (k % 4 = 3 ∨ (k % 4 = 0 ∧ k ≥ 92)) :=
by
  sorry

end partition_exists_iff_l854_85485


namespace total_candies_in_store_l854_85489

-- Define the quantities of chocolates in each box
def box_chocolates_1 := 200
def box_chocolates_2 := 320
def box_chocolates_3 := 500
def box_chocolates_4 := 500
def box_chocolates_5 := 768
def box_chocolates_6 := 768

-- Define the quantities of candies in each tub
def tub_candies_1 := 1380
def tub_candies_2 := 1150
def tub_candies_3 := 1150
def tub_candies_4 := 1720

-- Sum of all chocolates and candies
def total_chocolates := box_chocolates_1 + box_chocolates_2 + box_chocolates_3 + box_chocolates_4 + box_chocolates_5 + box_chocolates_6
def total_candies := tub_candies_1 + tub_candies_2 + tub_candies_3 + tub_candies_4
def total_store_candies := total_chocolates + total_candies

theorem total_candies_in_store : total_store_candies = 8456 := by
  sorry

end total_candies_in_store_l854_85489


namespace min_variance_l854_85423

/--
Given a sample x, 1, y, 5 with an average of 2,
prove that the minimum value of the variance of this sample is 3.
-/
theorem min_variance (x y : ℝ) 
  (h_avg : (x + 1 + y + 5) / 4 = 2) :
  3 ≤ (1 / 4) * ((x - 2) ^ 2 + (y - 2) ^ 2 + (1 - 2) ^ 2 + (5 - 2) ^ 2) :=
sorry

end min_variance_l854_85423


namespace no_integer_roots_l854_85461

theorem no_integer_roots (a b x : ℤ) : 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 ≠ 0 :=
sorry

end no_integer_roots_l854_85461


namespace teena_speed_l854_85487

theorem teena_speed (T : ℝ) : 
  (∀ (d₀ d_poe d_ahead : ℝ), 
    d₀ = 7.5 ∧ d_poe = 40 * 1.5 ∧ d_ahead = 15 →
    T = (d₀ + d_poe + d_ahead) / 1.5) → 
  T = 55 :=
by
  intros
  sorry

end teena_speed_l854_85487


namespace eval_expr_l854_85408

theorem eval_expr : (2.1 * (49.7 + 0.3)) + 15 = 120 :=
  by
  sorry

end eval_expr_l854_85408


namespace find_other_number_l854_85448

def integers_three_and_four_sum (a b : ℤ) : Prop :=
  3 * a + 4 * b = 131

def one_of_the_numbers_is (x : ℤ) : Prop :=
  x = 17

theorem find_other_number (a b : ℤ) (h1 : integers_three_and_four_sum a b) (h2 : one_of_the_numbers_is a ∨ one_of_the_numbers_is b) :
  (a = 21 ∨ b = 21) :=
sorry

end find_other_number_l854_85448


namespace combination_sum_l854_85414

theorem combination_sum :
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) = 34 :=
by
  sorry

end combination_sum_l854_85414


namespace remainder_31_l854_85470

theorem remainder_31 (x : ℤ) (h : x % 62 = 7) : (x + 11) % 31 = 18 := by
  sorry

end remainder_31_l854_85470


namespace sweets_ratio_l854_85454

theorem sweets_ratio (total_sweets : ℕ) (mother_ratio : ℚ) (eldest_sweets second_sweets : ℕ)
  (h1 : total_sweets = 27) (h2 : mother_ratio = 1 / 3) (h3 : eldest_sweets = 8) (h4 : second_sweets = 6) :
  let mother_sweets := mother_ratio * total_sweets
  let remaining_sweets := total_sweets - mother_sweets
  let other_sweets := eldest_sweets + second_sweets
  let youngest_sweets := remaining_sweets - other_sweets
  youngest_sweets / eldest_sweets = 1 / 2 :=
by
  sorry

end sweets_ratio_l854_85454


namespace sum_of_sequence_l854_85451

noncomputable def f (n x : ℝ) : ℝ := (1 / (8 * n)) * x^2 + 2 * n * x

theorem sum_of_sequence (n : ℕ) (hn : n > 0) :
  let a : ℝ := 1 / (8 * n)
  let b : ℝ := 2 * n
  let f' := 2 * a * ((-n : ℝ )) + b 
  ∃ S : ℝ, S = (n - 1) * 2^(n + 1) + 2 := 
sorry

end sum_of_sequence_l854_85451


namespace find_y_from_expression_l854_85422

theorem find_y_from_expression :
  ∀ y : ℕ, 2^10 + 2^10 + 2^10 + 2^10 = 4^y → y = 6 :=
by
  sorry

end find_y_from_expression_l854_85422


namespace Tom_marble_choices_l854_85417

theorem Tom_marble_choices :
  let total_marbles := 18
  let special_colors := 4
  let choose_one_from_special := (Nat.choose special_colors 1)
  let remaining_marbles := total_marbles - special_colors
  let choose_remaining := (Nat.choose remaining_marbles 5)
  choose_one_from_special * choose_remaining = 8008
:= sorry

end Tom_marble_choices_l854_85417


namespace first_term_of_sequence_l854_85419

theorem first_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end first_term_of_sequence_l854_85419


namespace cost_of_french_bread_is_correct_l854_85428

noncomputable def cost_of_sandwiches := 2 * 7.75
noncomputable def cost_of_salami := 4.00
noncomputable def cost_of_brie := 3 * cost_of_salami
noncomputable def cost_of_olives := 10.00 * (1/4)
noncomputable def cost_of_feta := 8.00 * (1/2)
noncomputable def total_cost_of_items := cost_of_sandwiches + cost_of_salami + cost_of_brie + cost_of_olives + cost_of_feta
noncomputable def total_spent := 40.00
noncomputable def cost_of_french_bread := total_spent - total_cost_of_items

theorem cost_of_french_bread_is_correct :
  cost_of_french_bread = 2.00 :=
by
  sorry

end cost_of_french_bread_is_correct_l854_85428


namespace find_f_seven_l854_85491

theorem find_f_seven 
  (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 3) = x^2 - 2 * x + 3) :
  f 7 = 3 := 
sorry

end find_f_seven_l854_85491


namespace problem_statement_l854_85455

theorem problem_statement (x : ℝ) (h : 2 * x^2 + 1 = 17) : 4 * x^2 + 1 = 33 :=
by sorry

end problem_statement_l854_85455


namespace max_sin_angle_F1PF2_on_ellipse_l854_85401

theorem max_sin_angle_F1PF2_on_ellipse
  (x y : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h : P ∈ {Q | Q.1^2 / 9 + Q.2^2 / 5 = 1})
  (F1_is_focus : F1 = (-2, 0))
  (F2_is_focus : F2 = (2, 0)) :
  ∃ sin_max, sin_max = 4 * Real.sqrt 5 / 9 := 
sorry

end max_sin_angle_F1PF2_on_ellipse_l854_85401


namespace sarah_more_than_cecily_l854_85406

theorem sarah_more_than_cecily (t : ℕ) (ht : t = 144) :
  let s := (1 / 3 : ℚ) * t
  let a := (3 / 8 : ℚ) * t
  let c := t - (s + a)
  s - c = 6 := by
  sorry

end sarah_more_than_cecily_l854_85406


namespace root_of_function_l854_85424

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem root_of_function (f : ℝ → ℝ) (x₀ : ℝ) (h₀ : odd_function f) (h₁ : f (x₀) = Real.exp (x₀)) :
  (f (-x₀) * Real.exp (-x₀) + 1 = 0) :=
by
  sorry

end root_of_function_l854_85424


namespace thabo_books_l854_85445

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end thabo_books_l854_85445


namespace trapezoid_prob_l854_85494

noncomputable def trapezoid_probability_not_below_x_axis : ℝ :=
  let P := (4, 4)
  let Q := (-4, -4)
  let R := (-10, -4)
  let S := (-2, 4)
  -- Coordinates of intersection points
  let T := (0, 0)
  let U := (-6, 0)
  -- Compute the probability
  (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40)

theorem trapezoid_prob :
  trapezoid_probability_not_below_x_axis = (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40) :=
sorry

end trapezoid_prob_l854_85494


namespace find_number_l854_85499

theorem find_number (a b : ℕ) (h₁ : a = 555) (h₂ : b = 445) :
  let S := a + b
  let D := a - b
  let Q := 2 * D
  let R := 30
  let N := (S * Q) + R
  N = 220030 := by
  sorry

end find_number_l854_85499


namespace neither_sufficient_nor_necessary_l854_85426

theorem neither_sufficient_nor_necessary 
  (a b c : ℝ) : 
  ¬ ((∀ x : ℝ, b^2 - 4 * a * c < 0 → a * x^2 + b * x + c > 0) ∧ 
     (∀ x : ℝ, a * x^2 + b * x + c > 0 → b^2 - 4 * a * c < 0)) := 
by
  sorry

end neither_sufficient_nor_necessary_l854_85426


namespace greatest_power_of_2_divides_10_1004_minus_4_502_l854_85481

theorem greatest_power_of_2_divides_10_1004_minus_4_502 :
  ∃ k, 10^1004 - 4^502 = 2^1007 * k :=
sorry

end greatest_power_of_2_divides_10_1004_minus_4_502_l854_85481


namespace valid_colorings_l854_85421

-- Define the coloring function and the condition
variable (f : ℕ → ℕ) -- f assigns a color (0, 1, or 2) to each natural number
variable (a b c : ℕ)
-- Colors are represented by 0, 1, or 2
variable (colors : Fin 3)

-- Define the condition to be checked
def valid_coloring : Prop :=
  ∀ a b c, 2000 * (a + b) = c → (f a = f b ∧ f b = f c) ∨ (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f a)

-- Now define the two possible valid ways of coloring
def all_same_color : Prop :=
  ∃ color, ∀ n, f n = color

def every_third_different : Prop :=
  (∀ k : ℕ, f (3 * k) = 0 ∧ f (3 * k + 1) = 1 ∧ f (3 * k + 2) = 2)

-- Prove that these are the only two valid ways
theorem valid_colorings :
  valid_coloring f →
  all_same_color f ∨ every_third_different f :=
sorry

end valid_colorings_l854_85421


namespace probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l854_85477

noncomputable def p_a : ℝ := 0.18
noncomputable def p_b : ℝ := 0.5
noncomputable def p_b_given_a : ℝ := 0.2
noncomputable def p_c : ℝ := 0.3
noncomputable def p_c_given_a : ℝ := 0.4
noncomputable def p_c_given_b : ℝ := 0.6

noncomputable def p_a_and_b : ℝ := p_a * p_b_given_a
noncomputable def p_a_and_b_and_c : ℝ := p_c_given_a * p_a_and_b
noncomputable def p_a_and_b_given_c : ℝ := p_a_and_b_and_c / p_c
noncomputable def p_a_and_c_given_b : ℝ := p_a_and_b_and_c / p_b

theorem probability_a_and_b_and_c : p_a_and_b_and_c = 0.0144 := by
  sorry

theorem probability_a_and_b_given_c : p_a_and_b_given_c = 0.048 := by
  sorry

theorem probability_a_and_c_given_b : p_a_and_c_given_b = 0.0288 := by
  sorry

end probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l854_85477


namespace min_value_x_3y_min_value_x_3y_iff_l854_85479

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y ≥ 25 :=
sorry

theorem min_value_x_3y_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y = 25 ↔ x = 10 ∧ y = 5 :=
sorry

end min_value_x_3y_min_value_x_3y_iff_l854_85479


namespace new_average_age_l854_85446

theorem new_average_age:
  ∀ (initial_avg_age new_persons_avg_age : ℝ) (initial_count new_persons_count : ℕ),
    initial_avg_age = 16 →
    new_persons_avg_age = 15 →
    initial_count = 20 →
    new_persons_count = 20 →
    (initial_avg_age * initial_count + new_persons_avg_age * new_persons_count) / 
    (initial_count + new_persons_count) = 15.5 :=
by
  intros initial_avg_age new_persons_avg_age initial_count new_persons_count
  intros h1 h2 h3 h4
  
  sorry

end new_average_age_l854_85446


namespace find_C_l854_85407

theorem find_C 
  (m n : ℝ)
  (C : ℝ)
  (h1 : m = 6 * n + C)
  (h2 : m + 2 = 6 * (n + 0.3333333333333333) + C) 
  : C = 0 := by
  sorry

end find_C_l854_85407


namespace greatest_x_lcm_l854_85462

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l854_85462


namespace cost_price_percentage_of_marked_price_l854_85416

theorem cost_price_percentage_of_marked_price
  (MP : ℝ) -- Marked Price
  (CP : ℝ) -- Cost Price
  (discount_percent : ℝ) (gain_percent : ℝ)
  (H1 : CP = (x / 100) * MP) -- Cost Price is x percent of Marked Price
  (H2 : discount_percent = 13) -- Discount percentage
  (H3 : gain_percent = 55.35714285714286) -- Gain percentage
  : x = 56 :=
sorry

end cost_price_percentage_of_marked_price_l854_85416


namespace store_earnings_l854_85425

theorem store_earnings (num_pencils : ℕ) (num_erasers : ℕ) (price_eraser : ℝ) 
  (multiplier : ℝ) (price_pencil : ℝ) (total_earnings : ℝ) :
  num_pencils = 20 →
  price_eraser = 1 →
  num_erasers = num_pencils * 2 →
  price_pencil = (price_eraser * num_erasers) * multiplier →
  multiplier = 2 →
  total_earnings = num_pencils * price_pencil + num_erasers * price_eraser →
  total_earnings = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end store_earnings_l854_85425


namespace divisible_by_27_l854_85466

theorem divisible_by_27 (n : ℕ) : 27 ∣ (2^(5*n+1) + 5^(n+2)) :=
by
  sorry

end divisible_by_27_l854_85466


namespace question_proof_l854_85495

theorem question_proof (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) : xy + y^2 = y^2 + y + 12 :=
by
  sorry

end question_proof_l854_85495


namespace calc_j_inverse_l854_85493

noncomputable def i : ℂ := Complex.I  -- Equivalent to i^2 = -1 definition of complex imaginary unit
noncomputable def j : ℂ := i + 1      -- Definition of j

theorem calc_j_inverse :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by 
  -- The statement here only needs to declare the equivalence, 
  -- without needing the proof
  sorry

end calc_j_inverse_l854_85493


namespace sin_18_eq_l854_85435

theorem sin_18_eq : ∃ x : Real, x = (Real.sin (Real.pi / 10)) ∧ x = (Real.sqrt 5 - 1) / 4 := by
  sorry

end sin_18_eq_l854_85435


namespace benjamin_collects_6_dozen_eggs_l854_85449

theorem benjamin_collects_6_dozen_eggs (B : ℕ) (h : B + 3 * B + (B - 4) = 26) : B = 6 :=
by sorry

end benjamin_collects_6_dozen_eggs_l854_85449


namespace old_hen_weight_unit_l854_85460

theorem old_hen_weight_unit (w : ℕ) (units : String) (opt1 opt2 opt3 opt4 : String)
  (h_opt1 : opt1 = "grams") (h_opt2 : opt2 = "kilograms") (h_opt3 : opt3 = "tons") (h_opt4 : opt4 = "meters") (h_w : w = 2) : 
  (units = opt2) :=
sorry

end old_hen_weight_unit_l854_85460


namespace total_cost_of_fencing_l854_85475

theorem total_cost_of_fencing (length breadth : ℕ) (cost_per_metre : ℕ) 
  (h1 : length = breadth + 20) 
  (h2 : length = 200) 
  (h3 : cost_per_metre = 26): 
  2 * (length + breadth) * cost_per_metre = 20140 := 
by sorry

end total_cost_of_fencing_l854_85475


namespace translate_right_l854_85478

-- Definition of the initial point and translation distance
def point_A : ℝ × ℝ := (2, -1)
def translation_distance : ℝ := 3

-- The proof statement
theorem translate_right (x_A y_A : ℝ) (d : ℝ) 
  (h1 : point_A = (x_A, y_A))
  (h2 : translation_distance = d) : 
  (x_A + d, y_A) = (5, -1) := 
sorry

end translate_right_l854_85478


namespace rationalize_denominator_l854_85484

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l854_85484


namespace first_divisor_l854_85498

theorem first_divisor (k : ℤ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 7 = 3) (h4 : k < 42) (hk : k = 17) : 5 ≤ 6 ∧ 5 ≤ 7 ∧ 5 = 5 :=
by {
  sorry
}

end first_divisor_l854_85498


namespace find_a_7_l854_85410

-- Define the arithmetic sequence conditions
variable {a : ℕ → ℤ} -- The sequence a_n
variable (a_4_eq : a 4 = 4)
variable (a_3_a_8_eq : a 3 + a 8 = 5)

-- Prove that a_7 = 1
theorem find_a_7 : a 7 = 1 := by
  sorry

end find_a_7_l854_85410


namespace no_solution_implies_a_eq_one_l854_85409

theorem no_solution_implies_a_eq_one (a : ℝ) : 
  ¬(∃ x y : ℝ, a * x + y = 1 ∧ x + y = 2) → a = 1 :=
by
  intro h
  sorry

end no_solution_implies_a_eq_one_l854_85409


namespace geometric_sequence_n_terms_l854_85444

/-- In a geometric sequence with the first term a₁ and common ratio q,
the number of terms n for which the nth term aₙ has a given value -/
theorem geometric_sequence_n_terms (a₁ aₙ q : ℚ) (n : ℕ)
  (h1 : a₁ = 9/8)
  (h2 : aₙ = 1/3)
  (h3 : q = 2/3)
  (h_seq : aₙ = a₁ * q^(n-1)) :
  n = 4 := sorry

end geometric_sequence_n_terms_l854_85444


namespace second_number_is_sixty_l854_85450

theorem second_number_is_sixty (x : ℕ) (h_sum : 2 * x + x + (2 / 3) * x = 220) : x = 60 :=
by
  sorry

end second_number_is_sixty_l854_85450


namespace remainder_of_expression_l854_85415

theorem remainder_of_expression (k : ℤ) (hk : 0 < k) :
  (4 * k * (2 + 4 + 4 * k) + 3) % 2 = 1 :=
by
  sorry

end remainder_of_expression_l854_85415


namespace product_of_conversions_l854_85474

-- Define the binary number 1101
def binary_number := 1101

-- Convert binary 1101 to decimal
def binary_to_decimal : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 212
def ternary_number := 212

-- Convert ternary 212 to decimal
def ternary_to_decimal : ℕ := 2 * 3^2 + 1 * 3^1 + 2 * 3^0

-- Statement to prove
theorem product_of_conversions : (binary_to_decimal) * (ternary_to_decimal) = 299 := by
  sorry

end product_of_conversions_l854_85474


namespace fraction_of_kiwis_l854_85464

theorem fraction_of_kiwis (total_fruits : ℕ) (num_strawberries : ℕ) (h₁ : total_fruits = 78) (h₂ : num_strawberries = 52) :
  (total_fruits - num_strawberries) / total_fruits = 1 / 3 :=
by
  -- proof to be provided, this is just the statement
  sorry

end fraction_of_kiwis_l854_85464


namespace parabola_equation_l854_85430

theorem parabola_equation (x y : ℝ) :
  (∃p : ℝ, x = 4 ∧ y = -2 ∧ (x^2 = -2 * p * y ∨ y^2 = 2 * p * x) → (x^2 = -8 * y ∨ y^2 = x)) :=
by
  sorry

end parabola_equation_l854_85430


namespace rate_of_interest_is_4_l854_85476

theorem rate_of_interest_is_4 (R : ℝ) : 
  ∀ P : ℝ, ∀ T : ℝ, P = 3000 → T = 5 → (P * R * T / 100 = P - 2400) → R = 4 :=
by
  sorry

end rate_of_interest_is_4_l854_85476


namespace surface_area_of_brick_l854_85431

namespace SurfaceAreaProof

def brick_length : ℝ := 8
def brick_width : ℝ := 6
def brick_height : ℝ := 2

theorem surface_area_of_brick :
  2 * (brick_length * brick_width + brick_length * brick_height + brick_width * brick_height) = 152 :=
by
  sorry

end SurfaceAreaProof

end surface_area_of_brick_l854_85431


namespace solution_set_inequality_l854_85427

noncomputable def f : ℝ → ℝ := sorry

variable {f : ℝ → ℝ}
variable (hf_diff : Differentiable ℝ f)
variable (hf_ineq : ∀ x, f x > deriv f x)
variable (hf_zero : f 0 = 2)

theorem solution_set_inequality : {x : ℝ | f x < 2 * Real.exp x} = {x | 0 < x} :=
by
  sorry

end solution_set_inequality_l854_85427


namespace smallest_magnitude_z_theorem_l854_85437

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem smallest_magnitude_z_theorem : 
  ∃ z : ℂ, (Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) ∧
  smallest_magnitude_z z = 36 / Real.sqrt 97 := 
sorry

end smallest_magnitude_z_theorem_l854_85437


namespace weeks_to_work_l854_85482

-- Definitions of conditions as per step a)
def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def brothers_total_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def job_pay_per_week : ℕ := 3
def total_ticket_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
def total_savings := isabelle_savings + brothers_total_savings
def remaining_amount := total_ticket_cost - total_savings

-- Theorem statement to match the question
theorem weeks_to_work : remaining_amount / job_pay_per_week = 10 := by
  -- Lean expects a proof here, replaced with sorry to skip it
  sorry

end weeks_to_work_l854_85482


namespace cost_per_yellow_ink_l854_85488

def initial_amount : ℕ := 50
def cost_per_black_ink : ℕ := 11
def num_black_inks : ℕ := 2
def cost_per_red_ink : ℕ := 15
def num_red_inks : ℕ := 3
def additional_amount_needed : ℕ := 43
def num_yellow_inks : ℕ := 2

theorem cost_per_yellow_ink :
  let total_cost_needed := initial_amount + additional_amount_needed
  let total_black_ink_cost := cost_per_black_ink * num_black_inks
  let total_red_ink_cost := cost_per_red_ink * num_red_inks
  let total_non_yellow_cost := total_black_ink_cost + total_red_ink_cost
  let total_yellow_ink_cost := total_cost_needed - total_non_yellow_cost
  let cost_per_yellow_ink := total_yellow_ink_cost / num_yellow_inks
  cost_per_yellow_ink = 13 :=
by
  sorry

end cost_per_yellow_ink_l854_85488


namespace range_of_m_l854_85442

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4 * x - m < 0 ∧ -1 ≤ x ∧ x ≤ 2) →
  (∃ x : ℝ, x^2 - x - 2 > 0) →
  (∀ x : ℝ, 4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) →
  m > 8 :=
sorry

end range_of_m_l854_85442


namespace range_of_k_l854_85436

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > k) → k > 3 := 
sorry

end range_of_k_l854_85436


namespace polynomial_properties_l854_85483

noncomputable def p (x : ℕ) : ℕ := 2 * x^3 + x + 4

theorem polynomial_properties :
  p 1 = 7 ∧ p 10 = 2014 := 
by
  -- Placeholder for proof
  sorry

end polynomial_properties_l854_85483


namespace max_value_m_l854_85429

noncomputable def max_m : ℝ := 10

theorem max_value_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = x + 2 * y) : x * y ≥ max_m - 2 :=
by
  sorry

end max_value_m_l854_85429


namespace product_of_roots_of_quartic_polynomial_l854_85412

theorem product_of_roots_of_quartic_polynomial :
  (∀ x : ℝ, (3 * x^4 - 8 * x^3 + x^2 - 10 * x - 24 = 0) → x = p ∨ x = q ∨ x = r ∨ x = s) →
  (p * q * r * s = -8) :=
by
  intros
  -- proof goes here
  sorry

end product_of_roots_of_quartic_polynomial_l854_85412


namespace min_41x_2y_eq_nine_l854_85465

noncomputable def min_value_41x_2y (x y : ℝ) : ℝ :=
  41*x + 2*y

theorem min_41x_2y_eq_nine (x y : ℝ) (h : ∀ n : ℕ, 0 < n →  n*x + (1/n)*y ≥ 1) :
  min_value_41x_2y x y = 9 :=
sorry

end min_41x_2y_eq_nine_l854_85465


namespace log5_x_equals_neg_two_log5_2_l854_85452

theorem log5_x_equals_neg_two_log5_2 (x : ℝ) (h : x = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3)) :
  Real.log x / Real.log 5 = -2 * (Real.log 2 / Real.log 5) :=
by
  sorry

end log5_x_equals_neg_two_log5_2_l854_85452


namespace desired_interest_percentage_l854_85433

-- Definitions based on conditions
def face_value : ℝ := 20
def dividend_rate : ℝ := 0.09  -- 9% converted to fraction
def market_value : ℝ := 15

-- The main statement
theorem desired_interest_percentage : 
  ((dividend_rate * face_value) / market_value) * 100 = 12 :=
by
  sorry

end desired_interest_percentage_l854_85433


namespace tan_theta_eq_l854_85492

variables (k θ : ℝ)

-- Condition: k > 0
axiom k_pos : k > 0

-- Condition: k * cos θ = 12
axiom k_cos_theta : k * Real.cos θ = 12

-- Condition: k * sin θ = 5
axiom k_sin_theta : k * Real.sin θ = 5

-- To prove: tan θ = 5 / 12
theorem tan_theta_eq : Real.tan θ = 5 / 12 := by
  sorry

end tan_theta_eq_l854_85492


namespace sum_of_distances_l854_85496

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_2 = d_1 + 5) (h2 : d_1 + d_2 = 13) :
  d_1 + d_2 = 13 :=
by sorry

end sum_of_distances_l854_85496


namespace hour_minute_hand_coincide_at_l854_85420

noncomputable def coinciding_time : ℚ :=
  90 / (6 - 0.5)

theorem hour_minute_hand_coincide_at : coinciding_time = 16 + 4 / 11 := 
  sorry

end hour_minute_hand_coincide_at_l854_85420


namespace opposite_of_negative_fraction_l854_85440

theorem opposite_of_negative_fraction :
  -(-1 / 2023) = (1 / 2023) :=
by
  sorry

end opposite_of_negative_fraction_l854_85440


namespace g_g_is_odd_l854_85472

def f (x : ℝ) : ℝ := x^3

def g (x : ℝ) : ℝ := f (f x)

theorem g_g_is_odd : ∀ x : ℝ, g (g (-x)) = -g (g x) :=
by 
-- proof will go here
sorry

end g_g_is_odd_l854_85472


namespace fourth_friend_age_is_8_l854_85403

-- Define the given data
variables (a1 a2 a3 a4 : ℕ)
variables (h_avg : (a1 + a2 + a3 + a4) / 4 = 9)
variables (h1 : a1 = 7) (h2 : a2 = 9) (h3 : a3 = 12)

-- Formalize the theorem to prove that the fourth friend's age is 8
theorem fourth_friend_age_is_8 : a4 = 8 :=
by
  -- Placeholder for the proof
  sorry

end fourth_friend_age_is_8_l854_85403


namespace fraction_relation_l854_85411

theorem fraction_relation (n d : ℕ) (h1 : (n + 1 : ℚ) / (d + 1) = 3 / 5) (h2 : (n : ℚ) / d = 5 / 9) :
  ∃ k : ℚ, d = k * 2 * n ∧ k = 9 / 10 :=
by
  sorry

end fraction_relation_l854_85411


namespace horse_bags_problem_l854_85441

theorem horse_bags_problem (x y : ℤ) 
  (h1 : x - 1 = y + 1) : 
  x + 1 = 2 * (y - 1) :=
sorry

end horse_bags_problem_l854_85441


namespace birds_problem_l854_85468

theorem birds_problem 
  (x y z : ℕ) 
  (h1 : x + y + z = 30) 
  (h2 : (1 / 3 : ℚ) * x + (1 / 2 : ℚ) * y + 2 * z = 30) 
  : x = 9 ∧ y = 10 ∧ z = 11 := 
  by {
  -- Proof steps would go here
  sorry
}

end birds_problem_l854_85468


namespace expand_polynomial_l854_85457

variable (x : ℝ)

theorem expand_polynomial :
  (7 * x - 3) * (2 * x ^ 3 + 5 * x ^ 2 - 4) = 14 * x ^ 4 + 29 * x ^ 3 - 15 * x ^ 2 - 28 * x + 12 := by
  sorry

end expand_polynomial_l854_85457


namespace Karen_tote_weight_l854_85443

variable (B T F : ℝ)
variable (Papers Laptop : ℝ)

theorem Karen_tote_weight (h1: T = 2 * B)
                         (h2: F = 2 * T)
                         (h3: Papers = (1 / 6) * F)
                         (h4: Laptop = T + 2)
                         (h5: F = B + Laptop + Papers):
  T = 12 := 
sorry

end Karen_tote_weight_l854_85443
