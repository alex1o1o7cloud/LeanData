import Mathlib

namespace height_of_cuboid_l647_64713

theorem height_of_cuboid (A l w : ℝ) (h : ℝ) (hA : A = 442) (hl : l = 7) (hw : w = 8) : h = 11 :=
by
  sorry

end height_of_cuboid_l647_64713


namespace y_expression_l647_64736

theorem y_expression (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x := 
by
  sorry

end y_expression_l647_64736


namespace david_savings_l647_64743

def lawn_rate_monday : ℕ := 14
def lawn_rate_wednesday : ℕ := 18
def lawn_rate_friday : ℕ := 20
def hours_per_day : ℕ := 2
def weekly_earnings : ℕ := (lawn_rate_monday * hours_per_day) + (lawn_rate_wednesday * hours_per_day) + (lawn_rate_friday * hours_per_day)

def tax_rate : ℚ := 0.10
def tax_paid (earnings : ℚ) : ℚ := earnings * tax_rate

def shoe_price : ℚ := 75
def discount : ℚ := 0.15
def discounted_shoe_price : ℚ := shoe_price * (1 - discount)

def money_remaining (earnings : ℚ) (tax : ℚ) (shoes : ℚ) : ℚ := earnings - tax - shoes

def gift_rate : ℚ := 1 / 3
def money_given_to_mom (remaining : ℚ) : ℚ := remaining * gift_rate

def final_savings (remaining : ℚ) (gift : ℚ) : ℚ := remaining - gift

theorem david_savings : 
  final_savings (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price) 
                (money_given_to_mom (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price)) 
  = 19.90 :=
by
  -- The proof goes here
  sorry

end david_savings_l647_64743


namespace max_value_of_z_l647_64715

theorem max_value_of_z (x y : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) :
  x^2 + y^2 ≤ 2 :=
by {
  sorry
}

end max_value_of_z_l647_64715


namespace domino_covering_l647_64797

theorem domino_covering (m n : ℕ) (m_eq : (m, n) ∈ [(5, 5), (4, 6), (3, 7), (5, 6), (3, 8)]) :
  (m * n % 2 = 1) ↔ (m = 5 ∧ n = 5) ∨ (m = 3 ∧ n = 7) :=
by
  sorry

end domino_covering_l647_64797


namespace cyclist_arrives_first_l647_64769

-- Definitions based on given conditions
def speed_cyclist (v : ℕ) := v
def speed_motorist (v : ℕ) := 5 * v

def distance_total (d : ℕ) := d
def distance_half (d : ℕ) := d / 2

def time_motorist_first_half (d v : ℕ) : ℕ := distance_half d / speed_motorist v

def remaining_distance_cyclist (d v : ℕ) := d - v * time_motorist_first_half d v

def speed_motorist_walking (v : ℕ) := v / 2

def time_motorist_second_half (d v : ℕ) := distance_half d / speed_motorist_walking v
def time_cyclist_remaining (d v : ℕ) : ℕ := remaining_distance_cyclist d v / speed_cyclist v

-- Comparison to prove cyclist arrives first
theorem cyclist_arrives_first (d v : ℕ) (hv : 0 < v) (hd : 0 < d) :
  time_cyclist_remaining d v < time_motorist_second_half d v :=
by sorry

end cyclist_arrives_first_l647_64769


namespace quadratic_solution_product_l647_64707

theorem quadratic_solution_product :
  let r := 9 / 2
  let s := -11
  (r + 4) * (s + 4) = -119 / 2 :=
by
  -- Define the quadratic equation and its solutions
  let r := 9 / 2
  let s := -11

  -- Prove the statement
  sorry

end quadratic_solution_product_l647_64707


namespace solution_set_transformation_l647_64751

noncomputable def solution_set_of_first_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 5 * x + b > 0}

noncomputable def solution_set_of_second_inequality (a b : ℝ) : Set ℝ :=
  {x | b * x^2 - 5 * x + a > 0}

theorem solution_set_transformation (a b : ℝ)
  (h : solution_set_of_first_inequality a b = {x | -3 < x ∧ x < 2}) :
  solution_set_of_second_inequality a b = {x | x < -3 ∨ x > 2} :=
by
  sorry

end solution_set_transformation_l647_64751


namespace monotonicity_condition_l647_64760

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, f a x ≥ f a 1) ↔ a ∈ Set.Ici 2 :=
by
  sorry

end monotonicity_condition_l647_64760


namespace simplify_expression_l647_64726

theorem simplify_expression (t : ℝ) (t_ne_zero : t ≠ 0) : (t^5 * t^3) / t^4 = t^4 := 
by
  sorry

end simplify_expression_l647_64726


namespace correct_operation_l647_64717

variable (a b m : ℕ)

theorem correct_operation :
  (3 * a^2 * 2 * a^2 ≠ 5 * a^2) ∧
  ((2 * a^2)^3 = 8 * a^6) ∧
  (m^6 / m^3 ≠ m^2) ∧
  ((a + b)^2 ≠ a^2 + b^2) →
  ((2 * a^2)^3 = 8 * a^6) :=
by
  intros
  sorry

end correct_operation_l647_64717


namespace range_of_a_l647_64764

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l647_64764


namespace intersection_M_N_l647_64782

def M : Set ℝ := { x | x < 2017 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l647_64782


namespace sum_sequence_formula_l647_64709

-- Define the sequence terms as a function.
def seq_term (x a : ℕ) (n : ℕ) : ℕ :=
x ^ (n + 1) + (n + 1) * a

-- Define the sum of the first nine terms of the sequence.
def sum_first_nine_terms (x a : ℕ) : ℕ :=
(x * (x ^ 9 - 1)) / (x - 1) + 45 * a

-- State the theorem to prove that the sum S is as expected.
theorem sum_sequence_formula (x a : ℕ) (h : x ≠ 1) : 
  sum_first_nine_terms x a = (x ^ 10 - x) / (x - 1) + 45 * a := by
  sorry

end sum_sequence_formula_l647_64709


namespace original_price_of_article_l647_64755

theorem original_price_of_article 
  (S : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : S = 25)
  (h2 : gain_percent = 1.5)
  (h3 : S = P + P * gain_percent) : 
  P = 10 :=
by 
  sorry

end original_price_of_article_l647_64755


namespace f_of_10_is_20_l647_64761

theorem f_of_10_is_20 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (3 * x + 1) = x^2 + 3 * x + 2) : f 10 = 20 :=
  sorry

end f_of_10_is_20_l647_64761


namespace probability_of_odd_number_l647_64738

theorem probability_of_odd_number (wedge1 wedge2 wedge3 wedge4 wedge5 : ℝ)
  (h_wedge1_split : wedge1/3 = wedge2) 
  (h_wedge2_twice_wedge1 : wedge2 = 2 * (wedge1/3))
  (h_wedge3 : wedge3 = 1/4)
  (h_wedge5 : wedge5 = 1/4)
  (h_total : wedge1/3 + wedge2 + wedge3 + wedge4 + wedge5 = 1) :
  wedge1/3 + wedge3 + wedge5 = 7 / 12 :=
by
  sorry

end probability_of_odd_number_l647_64738


namespace gcd_315_2016_l647_64725

def a : ℕ := 315
def b : ℕ := 2016

theorem gcd_315_2016 : Nat.gcd a b = 63 := 
by 
  sorry

end gcd_315_2016_l647_64725


namespace find_b_l647_64758

-- Define the slopes of the two lines derived from the given conditions
noncomputable def slope1 := -2 / 3
noncomputable def slope2 (b : ℚ) := -b / 3

-- Lean 4 statement to prove that for the lines to be perpendicular, b must be -9/2
theorem find_b (b : ℚ) (h_perpendicular: slope1 * slope2 b = -1) : b = -9 / 2 := by
  sorry

end find_b_l647_64758


namespace resulting_chemical_percentage_l647_64749

theorem resulting_chemical_percentage 
  (init_solution_pct : ℝ) (replacement_frac : ℝ) (replacing_solution_pct : ℝ) (resulting_solution_pct : ℝ) : 
  init_solution_pct = 0.85 →
  replacement_frac = 0.8181818181818182 →
  replacing_solution_pct = 0.30 →
  resulting_solution_pct = 0.40 :=
by
  intros h1 h2 h3
  sorry

end resulting_chemical_percentage_l647_64749


namespace highest_y_coordinate_l647_64781

theorem highest_y_coordinate (x y : ℝ) (h : (x^2 / 49 + (y-3)^2 / 25 = 0)) : y = 3 :=
by
  sorry

end highest_y_coordinate_l647_64781


namespace find_number_of_children_l647_64700

-- Definitions based on conditions
def decorative_spoons : Nat := 2
def new_set_large_spoons : Nat := 10
def new_set_tea_spoons : Nat := 15
def total_spoons : Nat := 39
def spoons_per_child : Nat := 3
def new_set_spoons := new_set_large_spoons + new_set_tea_spoons

-- The main statement to prove the number of children
theorem find_number_of_children (C : Nat) :
  3 * C + decorative_spoons + new_set_spoons = total_spoons → C = 4 :=
by
  -- Proof would go here
  sorry

end find_number_of_children_l647_64700


namespace percentage_profit_l647_64733

variable (total_crates : ℕ)
variable (total_cost : ℕ)
variable (lost_crates : ℕ)
variable (sell_price_per_crate : ℕ)

theorem percentage_profit (h1 : total_crates = 10) (h2 : total_cost = 160)
  (h3 : lost_crates = 2) (h4 : sell_price_per_crate = 25) :
  (8 * sell_price_per_crate - total_cost) * 100 / total_cost = 25 :=
by
  -- Definitions and steps to prove this can be added here.
  sorry

end percentage_profit_l647_64733


namespace infinite_geometric_series_sum_l647_64785

theorem infinite_geometric_series_sum (p q : ℝ)
  (h : (∑' n : ℕ, p / q ^ (n + 1)) = 5) :
  (∑' n : ℕ, p / (p^2 + q) ^ (n + 1)) = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) :=
sorry

end infinite_geometric_series_sum_l647_64785


namespace scientific_notation_350_million_l647_64775

theorem scientific_notation_350_million : 350000000 = 3.5 * 10^8 := 
  sorry

end scientific_notation_350_million_l647_64775


namespace div_by_133_l647_64778

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end div_by_133_l647_64778


namespace no_valid_n_l647_64714

theorem no_valid_n : ¬ ∃ (n : ℕ), (n > 0) ∧ (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by {
  sorry
}

end no_valid_n_l647_64714


namespace sequence_general_formula_l647_64765

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h₁ : a 1 = 2)
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) :
  ∀ n, a n = 1 + 2^(n - 1) := 
sorry

end sequence_general_formula_l647_64765


namespace copper_tin_alloy_weight_l647_64798

theorem copper_tin_alloy_weight :
  let c1 := (4/5 : ℝ) * 10 -- Copper in the first alloy
  let t1 := (1/5 : ℝ) * 10 -- Tin in the first alloy
  let c2 := (1/4 : ℝ) * 16 -- Copper in the second alloy
  let t2 := (3/4 : ℝ) * 16 -- Tin in the second alloy
  let x := ((3 * 14 - 24) / 2 : ℝ) -- Pure copper added
  let total_copper := c1 + c2 + x
  let total_tin := t1 + t2
  total_copper + total_tin = 35 := 
by
  sorry

end copper_tin_alloy_weight_l647_64798


namespace test_tube_full_with_two_amoebas_l647_64730

-- Definition: Each amoeba doubles in number every minute.
def amoeba_doubling (initial : Nat) (minutes : Nat) : Nat :=
  initial * 2 ^ minutes

-- Condition: Starting with one amoeba, the test tube is filled in 60 minutes.
def time_to_fill_one_amoeba := 60

-- Theorem: If two amoebas are placed in the test tube, it takes 59 minutes to fill.
theorem test_tube_full_with_two_amoebas : amoeba_doubling 2 59 = amoeba_doubling 1 time_to_fill_one_amoeba :=
by sorry

end test_tube_full_with_two_amoebas_l647_64730


namespace sum_of_factors_of_30_multiplied_by_2_equals_144_l647_64756

-- We define the factors of 30
def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- We define the function to multiply each factor by 2 and sum them
def sum_factors_multiplied_by_2 (factors : List ℕ) : ℕ :=
  factors.foldl (λ acc x => acc + 2 * x) 0

-- The final statement to be proven
theorem sum_of_factors_of_30_multiplied_by_2_equals_144 :
  sum_factors_multiplied_by_2 factors_of_30 = 144 :=
by sorry

end sum_of_factors_of_30_multiplied_by_2_equals_144_l647_64756


namespace minimum_c_value_l647_64704

theorem minimum_c_value
  (a b c k : ℕ) (h1 : b = a + k) (h2 : c = b + k) (h3 : a < b) (h4 : b < c) (h5 : k > 0) :
  c = 6005 :=
sorry

end minimum_c_value_l647_64704


namespace sheets_of_paper_in_each_box_l647_64763

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 70) 
  (h2 : 4 * (E - 20) = S) : 
  S = 120 := 
by 
  sorry

end sheets_of_paper_in_each_box_l647_64763


namespace problem1_problem2_l647_64762

-- Definitions and Lean statement for Problem 1
noncomputable def curve1 (x : ℝ) : ℝ := x / (2 * x - 1)
def point1 : ℝ × ℝ := (1, 1)
noncomputable def tangent_line1 (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem1 : tangent_line1 (point1.fst) (curve1 (point1.fst)) :=
sorry -- proof goes here

-- Definitions and Lean statement for Problem 2
def parabola (x : ℝ) : ℝ := x^2
def point2 : ℝ × ℝ := (2, 3)
noncomputable def tangent_line2a (x y : ℝ) : Prop := 2 * x - y - 1 = 0
noncomputable def tangent_line2b (x y : ℝ) : Prop := 6 * x - y - 9 = 0

theorem problem2 : (tangent_line2a point2.fst point2.snd ∨ tangent_line2b point2.fst point2.snd) :=
sorry -- proof goes here

end problem1_problem2_l647_64762


namespace mascot_sales_growth_rate_equation_l647_64766

-- Define the conditions
def march_sales : ℝ := 100000
def may_sales : ℝ := 115000
def growth_rate (x : ℝ) : Prop := x > 0

-- Define the equation to be proven
theorem mascot_sales_growth_rate_equation (x : ℝ) (h : growth_rate x) :
    10 * (1 + x) ^ 2 = 11.5 :=
sorry

end mascot_sales_growth_rate_equation_l647_64766


namespace condition_for_equation_l647_64777

theorem condition_for_equation (a b c d : ℝ) 
  (h : (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2)) : 
  a = c ∨ a^2 + d + 2 * b = 0 :=
by
  sorry

end condition_for_equation_l647_64777


namespace parabola_equation_line_AB_fixed_point_min_area_AMBN_l647_64720

-- Prove that the equation of the parabola is y^2 = 4x given the focus (1,0) for y^2 = 2px
theorem parabola_equation (p : ℝ) (h : p > 0) (foc : (1, 0) = (1, 2*p*1/4)):
  (∀ x y: ℝ, y^2 = 4*x ↔ y^2 = 2*p*x) := sorry

-- Prove that line AB passes through fixed point T(2,0) given conditions
theorem line_AB_fixed_point (A B : ℝ × ℝ) (hA : A.2^2 = 4*A.1) 
    (hB : B.2^2 = 4*B.1) (h : A.1*B.1 + A.2*B.2 = -4) :
  ∃ T : ℝ × ℝ, T = (2, 0) := sorry

-- Prove that minimum value of area Quadrilateral AMBN is 48
theorem min_area_AMBN (T : ℝ × ℝ) (A B M N : ℝ × ℝ)
    (hT : T = (2, 0)) (hA : A.2^2 = 4*A.1) (hB : B.2^2 = 4*B.1)
    (hM : M.2^2 = 4*M.1) (hN : N.2^2 = 4*N.1)
    (line_AB : A.1 * B.1 + A.2 * B.2 = -4) :
  ∀ (m : ℝ), T.2 = -(1/m)*T.1 + 2 → 
  ((1+m^2) * (1+1/m^2)) * ((m^2 + 2) * (1/m^2 + 2)) = 256 → 
  8 * 48 = 48 := sorry

end parabola_equation_line_AB_fixed_point_min_area_AMBN_l647_64720


namespace painting_methods_correct_l647_64750

def num_painting_methods : Nat := 72

theorem painting_methods_correct :
  let vertices : Fin 4 := by sorry -- Ensures there are four vertices
  let edges : Fin 4 := by sorry -- Ensures each edge has different colored endpoints
  let available_colors : Fin 4 := by sorry -- Ensures there are four available colors
  num_painting_methods = 72 :=
sorry

end painting_methods_correct_l647_64750


namespace find_subtracted_value_l647_64732

theorem find_subtracted_value (N V : ℤ) (hN : N = 12) (h : 4 * N - 3 = 9 * (N - V)) : V = 7 := 
by
  sorry

end find_subtracted_value_l647_64732


namespace jerome_time_6_hours_l647_64779

theorem jerome_time_6_hours (T: ℝ) (s_J: ℝ) (t_N: ℝ) (s_N: ℝ)
  (h1: s_J = 4) 
  (h2: t_N = 3) 
  (h3: s_N = 8): T = 6 :=
by
  -- Given s_J = 4, t_N = 3, and s_N = 8,
  -- we need to prove that T = 6.
  sorry

end jerome_time_6_hours_l647_64779


namespace ratio_sharks_to_pelicans_l647_64741

-- Define the conditions given in the problem
def original_pelican_count {P : ℕ} (h : (2/3 : ℚ) * P = 20) : Prop :=
  P = 30

-- Define the final ratio we want to prove
def shark_to_pelican_ratio (sharks pelicans : ℕ) : ℚ :=
  sharks / pelicans

theorem ratio_sharks_to_pelicans
  (P : ℕ) (h : (2/3 : ℚ) * P = 20) (number_sharks : ℕ) (number_pelicans : ℕ)
  (H_sharks : number_sharks = 60) (H_pelicans : number_pelicans = P)
  (H_original_pelicans : original_pelican_count h) :
  shark_to_pelican_ratio number_sharks number_pelicans = 2 :=
by
  -- proof skipped
  sorry

end ratio_sharks_to_pelicans_l647_64741


namespace acute_angle_proof_l647_64794

theorem acute_angle_proof
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : Real.cos (α + β) = Real.sin (α - β)) : α = π / 4 :=
  sorry

end acute_angle_proof_l647_64794


namespace fg_evaluation_l647_64727

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_evaluation : f (g 3) = 97 := by
  sorry

end fg_evaluation_l647_64727


namespace gcd_m_n_l647_64747

def m : ℕ := 333333
def n : ℕ := 888888888

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l647_64747


namespace find_values_of_expression_l647_64735

theorem find_values_of_expression (a b : ℝ) 
  (h : (2 * a) / (a + b) + b / (a - b) = 2) : 
  (∃ x : ℝ, x = (3 * a - b) / (a + 5 * b) ∧ (x = 3 ∨ x = 1)) :=
by 
  sorry

end find_values_of_expression_l647_64735


namespace olympiad_scores_l647_64753

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l647_64753


namespace original_rice_amount_l647_64796

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end original_rice_amount_l647_64796


namespace initial_amount_simple_interest_l647_64783

theorem initial_amount_simple_interest 
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1125)
  (hR : R = 0.10)
  (hT : T = 5) :
  A = P * (1 + R * T) → P = 750 := 
by
  sorry

end initial_amount_simple_interest_l647_64783


namespace line_eq_of_midpoint_and_hyperbola_l647_64723

theorem line_eq_of_midpoint_and_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : 9 * (8 : ℝ)^2 - 16 * (3 : ℝ)^2 = 144)
    (h2 : x1 + x2 = 16) (h3 : y1 + y2 = 6) (h4 : 9 * x1^2 - 16 * y1^2 = 144) (h5 : 9 * x2^2 - 16 * y2^2 = 144) :
    3 * (8 : ℝ) - 2 * (3 : ℝ) - 18 = 0 :=
by
  -- The proof steps would go here
  sorry

end line_eq_of_midpoint_and_hyperbola_l647_64723


namespace curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l647_64792

-- Definitions of the curves
def curve_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1
def curve_C2_parametric (theta : ℝ) (x y : ℝ) : Prop := (x = 4 * Real.cos theta) ∧ (y = 3 * Real.sin theta)
def curve_C3_polar (rho theta : ℝ) : Prop := rho * (Real.cos theta - 2 * Real.sin theta) = 7

-- Proving the mathematical equivalence:
theorem curve_C1_parametric_equiv (t : ℝ) : ∃ x y, curve_C1 x y ∧ (x = 3 + Real.cos t) ∧ (y = 2 + Real.sin t) :=
by sorry

theorem curve_C2_general_equiv (x y : ℝ) : (∃ theta, curve_C2_parametric theta x y) ↔ (x^2 / 16 + y^2 / 9 = 1) :=
by sorry

theorem curve_C3_rectangular_equiv (x y : ℝ) : (∃ rho theta, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ curve_C3_polar rho theta) ↔ (x - 2 * y - 7 = 0) :=
by sorry

theorem max_distance_C2_to_C3 : ∃ (d : ℝ), d = (2 * Real.sqrt 65 + 7 * Real.sqrt 5) / 5 :=
by sorry

end curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l647_64792


namespace find_integer_pairs_l647_64705

theorem find_integer_pairs :
  ∃ (n : ℤ) (a : ℤ) (b : ℤ),
    (∀ a b : ℤ, (∃ m : ℤ, a^2 - 4*b = m^2) ∧ (∃ k : ℤ, b^2 - 4*a = k^2) ↔ 
    (a = 0 ∧ ∃ n : ℤ, b = n^2) ∨
    (b = 0 ∧ ∃ n : ℤ, a = n^2) ∨
    (b > 0 ∧ ∃ a : ℤ, a^2 > 0 ∧ b = -1 - a) ∨
    (a > 0 ∧ ∃ b : ℤ, b^2 > 0 ∧ a = -1 - b) ∨
    (a = 4 ∧ b = 4) ∨
    (a = 5 ∧ b = 6) ∨
    (a = 6 ∧ b = 5)) :=
sorry

end find_integer_pairs_l647_64705


namespace angle_value_is_140_l647_64722

-- Definitions of conditions
def angle_on_straight_line_degrees (x y : ℝ) : Prop := x + y = 180

-- Main statement in Lean
theorem angle_value_is_140 (x : ℝ) (h₁ : angle_on_straight_line_degrees 40 x) : x = 140 :=
by
  -- Proof is omitted (not required as per instructions)
  sorry

end angle_value_is_140_l647_64722


namespace train_crossing_time_l647_64737

def train_length := 140
def train_speed_kmph := 45
def bridge_length := 235
def speed_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600
def total_distance := train_length + bridge_length
def train_speed := speed_to_mps train_speed_kmph
def time_to_cross := total_distance / train_speed

theorem train_crossing_time : time_to_cross = 30 := by
  sorry

end train_crossing_time_l647_64737


namespace training_days_l647_64767

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l647_64767


namespace total_cost_of_soup_l647_64768

theorem total_cost_of_soup 
  (pounds_beef : ℕ) (pounds_veg : ℕ) (cost_veg_per_pound : ℕ) (beef_price_multiplier : ℕ)
  (h1 : pounds_beef = 4)
  (h2 : pounds_veg = 6)
  (h3 : cost_veg_per_pound = 2)
  (h4 : beef_price_multiplier = 3):
  (pounds_veg * cost_veg_per_pound + pounds_beef * (cost_veg_per_pound * beef_price_multiplier)) = 36 :=
by
  sorry

end total_cost_of_soup_l647_64768


namespace cleaning_time_with_doubled_an_speed_l647_64757

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l647_64757


namespace power_inequality_l647_64702

theorem power_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : abs x < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end power_inequality_l647_64702


namespace range_of_a_l647_64744

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x_0 : ℝ, x_0^2 + (a - 1) * x_0 + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end range_of_a_l647_64744


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l647_64795

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l647_64795


namespace inverse_proportion_points_l647_64710

theorem inverse_proportion_points (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : x2 > 0)
  (h3 : y1 = -8 / x1)
  (h4 : y2 = -8 / x2) :
  y2 < 0 ∧ 0 < y1 :=
by
  sorry

end inverse_proportion_points_l647_64710


namespace angle_WYZ_correct_l647_64780

-- Define the angles as constants
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem statement asserting the solution
theorem angle_WYZ_correct :
  (angle_XYZ - angle_XYW = 21) := 
by
  -- This is where the proof would go, but we use 'sorry' as instructed
  sorry

end angle_WYZ_correct_l647_64780


namespace find_c_for_given_radius_l647_64718

theorem find_c_for_given_radius (c : ℝ) : (∃ x y : ℝ, (x^2 - 2 * x + y^2 + 6 * y + c = 0) ∧ ((x - 1)^2 + (y + 3)^2 = 25)) → c = -15 :=
by
  sorry

end find_c_for_given_radius_l647_64718


namespace number_of_other_values_l647_64724

def orig_value : ℕ := 2 ^ (2 ^ (2 ^ 2))

def other_values : Finset ℕ :=
  {2 ^ (2 ^ (2 ^ 2)), 2 ^ ((2 ^ 2) ^ 2), ((2 ^ 2) ^ 2) ^ 2, (2 ^ (2 ^ 2)) ^ 2, (2 ^ 2) ^ (2 ^ 2)}

theorem number_of_other_values :
  other_values.erase orig_value = {256} :=
by
  sorry

end number_of_other_values_l647_64724


namespace sara_steps_l647_64791

theorem sara_steps (n : ℕ) (h : n^2 ≤ 210) : n = 14 :=
sorry

end sara_steps_l647_64791


namespace lowest_selling_price_l647_64776

/-- Define the variables and constants -/
def production_cost_per_component := 80
def shipping_cost_per_component := 7
def fixed_costs_per_month := 16500
def components_per_month := 150

/-- Define the total variable cost -/
def total_variable_cost (production_cost_per_component shipping_cost_per_component : ℕ) (components_per_month : ℕ) :=
  (production_cost_per_component + shipping_cost_per_component) * components_per_month

/-- Define the total cost -/
def total_cost (variable_cost fixed_costs_per_month : ℕ) :=
  variable_cost + fixed_costs_per_month

/-- Define the lowest price per component -/
def lowest_price_per_component (total_cost components_per_month : ℕ) :=
  total_cost / components_per_month

/-- The main theorem to prove the lowest selling price required to cover all costs -/
theorem lowest_selling_price (production_cost shipping_cost fixed_costs components : ℕ)
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 7)
  (h3 : fixed_costs = 16500)
  (h4 : components = 150) :
  lowest_price_per_component (total_cost (total_variable_cost production_cost shipping_cost components) fixed_costs) components = 197 :=
by
  sorry

end lowest_selling_price_l647_64776


namespace cheesecakes_sold_l647_64770

theorem cheesecakes_sold
  (initial_display : Nat)
  (initial_fridge : Nat)
  (left_to_sell : Nat)
  (total_cheesecakes := initial_display + initial_fridge)
  (total_after_sales : Nat) :
  initial_display = 10 →
  initial_fridge = 15 →
  left_to_sell = 18 →
  total_after_sales = total_cheesecakes - left_to_sell →
  total_after_sales = 7 := sorry

end cheesecakes_sold_l647_64770


namespace joanne_trip_l647_64773

theorem joanne_trip (a b c x : ℕ) (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 100 * c + 10 * a + b - (100 * a + 10 * b + c) = 60 * x) : 
  a^2 + b^2 + c^2 = 51 :=
by
  sorry

end joanne_trip_l647_64773


namespace math_proof_problem_l647_64793

namespace Proofs

-- Definition of the arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop := 
  ∀ m n, a n = a m + (n - m) * (a (m + 1) - a m)

-- Conditions for the arithmetic sequence
def a_conditions (a : ℕ → ℤ) : Prop := 
  a 3 = -6 ∧ a 6 = 0

-- Definition of the geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop := 
  ∃ q, ∀ n, b (n + 1) = q * b n

-- Conditions for the geometric sequence
def b_conditions (b a : ℕ → ℤ) : Prop := 
  b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3

-- The general formula for {a_n}
def a_formula (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 12

-- The sum formula of the first n terms of {b_n}
def S_n_formula (b : ℕ → ℤ) (S_n : ℕ → ℤ) :=
  ∀ n, S_n n = 4 * (1 - 3^n)

-- The main theorem combining all
theorem math_proof_problem (a b : ℕ → ℤ) (S_n : ℕ → ℤ) :
  arithmetic_seq a →
  a_conditions a →
  geometric_seq b →
  b_conditions b a →
  (a_formula a ∧ S_n_formula b S_n) :=
by 
  sorry

end Proofs

end math_proof_problem_l647_64793


namespace jaya_amitabh_number_of_digits_l647_64728

-- Definitions
def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def digit_sum (n1 n2 : ℕ) : ℕ :=
  let (d1, d2) := (n1 % 10, n1 / 10)
  let (d3, d4) := (n2 % 10, n2 / 10)
  d1 + d2 + d3 + d4
def append_ages (j a : ℕ) : ℕ := 1000 * (j / 10) + 100 * (j % 10) + 10 * (a / 10) + (a % 10)
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Main theorem
theorem jaya_amitabh_number_of_digits 
  (j a : ℕ) 
  (hj : is_two_digit_number j)
  (ha : is_two_digit_number a)
  (h_sum : digit_sum j a = 7)
  (h_square : is_perfect_square (append_ages j a)) : 
  ∃ n : ℕ, String.length (toString (append_ages j a)) = 4 :=
by
  sorry

end jaya_amitabh_number_of_digits_l647_64728


namespace convert_speed_l647_64788

-- Definitions based on the given condition
def kmh_to_mps (kmh : ℝ) : ℝ := kmh * 0.277778

-- Theorem statement
theorem convert_speed : kmh_to_mps 84 = 23.33 :=
by
  -- Proof omitted
  sorry

end convert_speed_l647_64788


namespace remainder_of_number_divided_by_39_l647_64729

theorem remainder_of_number_divided_by_39 
  (N : ℤ) 
  (k m : ℤ) 
  (h₁ : N % 195 = 79) 
  (h₂ : N % 273 = 109) : 
  N % 39 = 1 :=
by 
  sorry

end remainder_of_number_divided_by_39_l647_64729


namespace tire_usage_l647_64752

theorem tire_usage (total_distance : ℕ) (num_tires : ℕ) (active_tires : ℕ) 
  (h1 : total_distance = 45000) 
  (h2 : num_tires = 5) 
  (h3 : active_tires = 4) 
  (equal_usage : (total_distance * active_tires) / num_tires = 36000) : 
  (∀ tire, tire < num_tires → used_miles_per_tire = 36000) := 
by
  sorry

end tire_usage_l647_64752


namespace w_identity_l647_64712

theorem w_identity (w : ℝ) (h_pos : w > 0) (h_eq : w - 1 / w = 5) : (w + 1 / w) ^ 2 = 29 := by
  sorry

end w_identity_l647_64712


namespace false_statement_is_D_l647_64786

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

def is_scalene_triangle (a b c : ℝ) : Prop :=
  (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def is_right_isosceles_triangle (a b c : ℝ) : Prop :=
  is_right_triangle a b c ∧ is_isosceles_triangle a b c

-- Statements derived from conditions
def statement_A : Prop := ∀ (a b c : ℝ), is_isosceles_triangle a b c → a = b ∨ b = c ∨ c = a
def statement_B : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2
def statement_C : Prop := ∀ (a b c : ℝ), is_scalene_triangle a b c → a ≠ b ∧ b ≠ c ∧ c ≠ a
def statement_D : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → is_isosceles_triangle a b c
def statement_E : Prop := ∀ (a b c : ℝ), is_right_isosceles_triangle a b c → ∃ (θ : ℝ), θ ≠ 90 ∧ θ = 45

-- Main theorem to be proved
theorem false_statement_is_D : statement_D = false :=
by
  sorry

end false_statement_is_D_l647_64786


namespace find_common_difference_l647_64745

section
variables (a1 a7 a8 a9 S5 S6 : ℚ) (d : ℚ)

/-- Given an arithmetic sequence with the sum of the first n terms S_n,
    if S_5 = a_8 + 5 and S_6 = a_7 + a_9 - 5, we need to find the common difference d. -/
theorem find_common_difference
  (h1 : S5 = a8 + 5)
  (h2 : S6 = a7 + a9 - 5)
  (h3 : S5 = 5 / 2 * (2 * a1 + 4 * d))
  (h4 : S6 = 6 / 2 * (2 * a1 + 5 * d))
  (h5 : a8 = a1 + 7 * d)
  (h6 : a7 = a1 + 6 * d)
  (h7 : a9 = a1 + 8 * d):
  d = -55 / 19 :=
by
  sorry
end

end find_common_difference_l647_64745


namespace simplify_fraction_l647_64701

theorem simplify_fraction (c : ℚ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := 
sorry

end simplify_fraction_l647_64701


namespace number_is_nine_l647_64754

theorem number_is_nine (x : ℤ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end number_is_nine_l647_64754


namespace sum_of_squares_of_chords_in_sphere_l647_64789

-- Defining variables
variables (R PO : ℝ)

-- Define the problem statement
theorem sum_of_squares_of_chords_in_sphere
  (chord_lengths_squared : ℝ)
  (H_chord_lengths_squared : chord_lengths_squared = 3 * R^2 - 2 * PO^2) :
  chord_lengths_squared = 3 * R^2 - 2 * PO^2 :=
by
  sorry -- proof is omitted

end sum_of_squares_of_chords_in_sphere_l647_64789


namespace sum_of_roots_l647_64748

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l647_64748


namespace total_plates_used_l647_64746

-- Definitions from the conditions
def number_of_people := 6
def meals_per_day_per_person := 3
def plates_per_meal_per_person := 2
def number_of_days := 4

-- Statement of the theorem
theorem total_plates_used : number_of_people * meals_per_day_per_person * plates_per_meal_per_person * number_of_days = 144 := 
by
  sorry

end total_plates_used_l647_64746


namespace bipin_chandan_age_ratio_l647_64708

-- Define the condition statements
def AlokCurrentAge : Nat := 5
def BipinCurrentAge : Nat := 6 * AlokCurrentAge
def ChandanCurrentAge : Nat := 7 + 3

-- Define the ages after 10 years
def BipinAgeAfter10Years : Nat := BipinCurrentAge + 10
def ChandanAgeAfter10Years : Nat := ChandanCurrentAge + 10

-- Define the ratio and the statement to prove
def AgeRatio := BipinAgeAfter10Years / ChandanAgeAfter10Years

-- The theorem to prove the ratio is 2
theorem bipin_chandan_age_ratio : AgeRatio = 2 := by
  sorry

end bipin_chandan_age_ratio_l647_64708


namespace meet_days_l647_64739

-- Definition of conditions
def person_a_days : ℕ := 5
def person_b_days : ℕ := 7
def person_b_early_departure : ℕ := 2

-- Definition of the number of days after A's start that they meet
variable {x : ℕ}

-- Statement to be proven
theorem meet_days (x : ℕ) : (x + 2 : ℚ) / person_b_days + x / person_a_days = 1 := sorry

end meet_days_l647_64739


namespace fraction_of_females_l647_64787

def local_soccer_league_female_fraction : Prop :=
  ∃ (males_last_year females_last_year : ℕ),
    males_last_year = 30 ∧
    (1.10 * males_last_year : ℝ) = 33 ∧
    (males_last_year + females_last_year : ℝ) * 1.15 = 52 ∧
    (females_last_year : ℝ) * 1.25 = 19 ∧
    (33 + 19 = 52)

theorem fraction_of_females
  : local_soccer_league_female_fraction → 
    ∃ (females fraction : ℝ),
    females = 19 ∧ 
    fraction = 19 / 52 :=
by
  sorry

end fraction_of_females_l647_64787


namespace library_books_count_l647_64740

def students_per_day : List ℕ := [4, 5, 6, 9]
def books_per_student : ℕ := 5
def total_books_given (students : List ℕ) (books_per_student : ℕ) : ℕ :=
  students.foldl (λ acc n => acc + n * books_per_student) 0

theorem library_books_count :
  total_books_given students_per_day books_per_student = 120 :=
by
  sorry

end library_books_count_l647_64740


namespace limit_fraction_l647_64711

theorem limit_fraction :
  ∀ ε > 0, ∃ (N : ℕ), ∀ n ≥ N, |((4 * n - 1) / (2 * n + 1) : ℚ) - 2| < ε := 
  by sorry

end limit_fraction_l647_64711


namespace find_a4b4_l647_64716

theorem find_a4b4 
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) :
  a4 * b4 = -6 :=
sorry

end find_a4b4_l647_64716


namespace orthogonal_circles_l647_64784

theorem orthogonal_circles (R1 R2 d : ℝ) :
  (d^2 = R1^2 + R2^2) ↔ (d^2 = R1^2 + R2^2) :=
by sorry

end orthogonal_circles_l647_64784


namespace inequality_range_l647_64742

theorem inequality_range (a : ℝ) : (-1 < a ∧ a ≤ 0) → ∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0 :=
by
  intro ha
  sorry

end inequality_range_l647_64742


namespace inequality_sum_l647_64759

theorem inequality_sum 
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2)
  (h2 : a2 ≥ a3)
  (h3 : a3 > 0)
  (h4 : b1 ≥ b2)
  (h5 : b2 ≥ b3)
  (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) :
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := 
sorry

end inequality_sum_l647_64759


namespace min_sum_xy_l647_64799

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (pos_x : 0 < x) (pos_y : 0 < y)
  (h : (1 : ℚ) / x + 1 / y = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l647_64799


namespace min_additional_packs_needed_l647_64706

-- Defining the problem conditions
def total_sticker_packs : ℕ := 40
def packs_per_basket : ℕ := 7

-- The statement to prove
theorem min_additional_packs_needed : 
  ∃ (additional_packs : ℕ), 
    (total_sticker_packs + additional_packs) % packs_per_basket = 0 ∧ 
    (total_sticker_packs + additional_packs) / packs_per_basket = 6 ∧ 
    additional_packs = 2 :=
by 
  sorry

end min_additional_packs_needed_l647_64706


namespace joan_mortgage_payment_l647_64774

noncomputable def geometric_series_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r^n) / (1 - r)

theorem joan_mortgage_payment : 
  ∃ n : ℕ, geometric_series_sum 100 3 n = 109300 ∧ n = 7 :=
by
  sorry

end joan_mortgage_payment_l647_64774


namespace series_pattern_l647_64772

theorem series_pattern :
    (3 / (1 * 2) * (1 / 2) + 4 / (2 * 3) * (1 / 2^2) + 5 / (3 * 4) * (1 / 2^3) + 6 / (4 * 5) * (1 / 2^4) + 7 / (5 * 6) * (1 / 2^5)) 
    = (1 - 1 / (6 * 2^5)) :=
  sorry

end series_pattern_l647_64772


namespace present_value_l647_64734

theorem present_value (BD TD PV : ℝ) (hBD : BD = 42) (hTD : TD = 36)
  (h : BD = TD + (TD^2 / PV)) : PV = 216 :=
sorry

end present_value_l647_64734


namespace find_pairs_l647_64731

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 1 + 5^a = 6^b → (a, b) = (1, 1) := by
  sorry

end find_pairs_l647_64731


namespace Jane_age_proof_l647_64790

theorem Jane_age_proof (D J : ℕ) (h1 : D + 6 = (J + 6) / 2) (h2 : D + 14 = 25) : J = 28 :=
by
  sorry

end Jane_age_proof_l647_64790


namespace distance_from_center_to_line_l647_64703

-- Define the conditions 
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 1

-- Define the assertion that we want to prove
theorem distance_from_center_to_line (ρ θ : ℝ) 
  (h_circle: circle_polar_eq ρ θ) 
  (h_line: line_polar_eq ρ θ) : 
  ∃ d : ℝ, d = (Real.sqrt 5) / 5 := 
sorry

end distance_from_center_to_line_l647_64703


namespace class_size_l647_64721

theorem class_size (n : ℕ) (h₁ : 60 - n > 0) (h₂ : (60 - n) / 2 = n) : n = 20 :=
by
  sorry

end class_size_l647_64721


namespace factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l647_64719

-- First factorization problem
theorem factor_3a3_minus_6a2_plus_3a (a : ℝ) : 
  3 * a ^ 3 - 6 * a ^ 2 + 3 * a = 3 * a * (a - 1) ^ 2 :=
by sorry

-- Second factorization problem
theorem factor_a2_minus_b2_x_minus_y (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
by sorry

-- Third factorization problem
theorem factor_16a_plus_b_sq_minus_9a_minus_b_sq (a b : ℝ) : 
  16 * (a + b) ^ 2 - 9 * (a - b) ^ 2 = (a + 7 * b) * (7 * a + b) :=
by sorry

end factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l647_64719


namespace find_original_faculty_count_l647_64771

variable (F : ℝ)
variable (final_count : ℝ := 195)
variable (first_year_reduction : ℝ := 0.075)
variable (second_year_increase : ℝ := 0.125)
variable (third_year_reduction : ℝ := 0.0325)
variable (fourth_year_increase : ℝ := 0.098)
variable (fifth_year_reduction : ℝ := 0.1465)

theorem find_original_faculty_count (h : F * (1 - first_year_reduction)
                                        * (1 + second_year_increase)
                                        * (1 - third_year_reduction)
                                        * (1 + fourth_year_increase)
                                        * (1 - fifth_year_reduction) = final_count) :
  F = 244 :=
by sorry

end find_original_faculty_count_l647_64771
