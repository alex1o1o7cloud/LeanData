import Mathlib

namespace angela_insects_l29_2962

theorem angela_insects:
  ∀ (A J D : ℕ), 
    A = J / 2 → 
    J = 5 * D → 
    D = 30 → 
    A = 75 :=
by
  intro A J D
  intro hA hJ hD
  sorry

end angela_insects_l29_2962


namespace hyperbola_eccentricity_l29_2966

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) 
(hyperbola_eq : ∀ (x y : ℝ), x^2 / 9 - y^2 / m = 1) 
(eccentricity : ∀ (e : ℝ), e = 2) 
: m = 27 :=
sorry

end hyperbola_eccentricity_l29_2966


namespace casper_initial_candies_l29_2979

theorem casper_initial_candies 
  (x : ℚ)
  (h1 : ∃ y : ℚ, y = x - (1/4) * x - 3) 
  (h2 : ∃ z : ℚ, z = y - (1/5) * y - 5) 
  (h3 : z - 10 = 10) : x = 224 / 3 :=
by
  sorry

end casper_initial_candies_l29_2979


namespace polynomial_remainder_l29_2982

theorem polynomial_remainder :
  ∀ (x : ℂ), (x^1010 % (x^4 - 1)) = x^2 :=
sorry

end polynomial_remainder_l29_2982


namespace length_AE_l29_2938

theorem length_AE (AB CD AC AE ratio : ℝ) 
  (h_AB : AB = 10) 
  (h_CD : CD = 15) 
  (h_AC : AC = 18) 
  (h_ratio : ratio = 2 / 3) 
  (h_areas : ∀ (areas : ℝ), areas = 2 / 3)
  : AE = 7.2 := 
sorry

end length_AE_l29_2938


namespace fran_avg_speed_l29_2944

theorem fran_avg_speed (Joann_speed : ℕ) (Joann_time : ℚ) (Fran_time : ℕ) (distance : ℕ) (s : ℚ) : 
  Joann_speed = 16 → 
  Joann_time = 3.5 → 
  Fran_time = 4 → 
  distance = Joann_speed * Joann_time → 
  distance = Fran_time * s → 
  s = 14 :=
by
  intros hJs hJt hFt hD hF
  sorry

end fran_avg_speed_l29_2944


namespace total_problems_l29_2909

-- Definitions based on conditions
def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def problems_per_page : ℕ := 4

-- Statement of the problem
theorem total_problems : math_pages + reading_pages * problems_per_page = 40 :=
by
  unfold math_pages reading_pages problems_per_page
  sorry

end total_problems_l29_2909


namespace min_weighings_to_order_four_stones_l29_2913

theorem min_weighings_to_order_four_stones : ∀ (A B C D : ℝ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D → ∃ n, n = 5 :=
by sorry

end min_weighings_to_order_four_stones_l29_2913


namespace simplify_expression_l29_2988

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end simplify_expression_l29_2988


namespace find_largest_number_l29_2940

theorem find_largest_number :
  let a := -(abs (-3) ^ 3)
  let b := -((-3) ^ 3)
  let c := (-3) ^ 3
  let d := -(3 ^ 3)
  b = 27 ∧ b > a ∧ b > c ∧ b > d := by
  sorry

end find_largest_number_l29_2940


namespace inequality_holds_for_all_real_l29_2917

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 1 ≥ 2 * |x| := sorry

end inequality_holds_for_all_real_l29_2917


namespace fallen_sheets_l29_2930

/-- The number of sheets that fell out of a book given the first page is 163
    and the last page contains the same digits but arranged in a different 
    order and ends with an even digit.
-/
theorem fallen_sheets (h1 : ∃ n, n = 163 ∧ 
                        ∃ m, m ≠ n ∧ (m = 316) ∧ 
                        m % 2 = 0 ∧ 
                        (∃ p1 p2 p3 q1 q2 q3, 
                         (p1, p2, p3) ≠ (q1, q2, q3) ∧ 
                         p1 ≠ q1 ∧ p2 ≠ q2 ∧ p3 ≠ q3 ∧ 
                         n = p1 * 100 + p2 * 10 + p3 ∧ 
                         m = q1 * 100 + q2 * 10 + q3)) :
  ∃ k, k = 77 :=
by
  sorry

end fallen_sheets_l29_2930


namespace Mike_got_18_cards_l29_2998

theorem Mike_got_18_cards (original_cards : ℕ) (total_cards : ℕ) : 
  original_cards = 64 → total_cards = 82 → total_cards - original_cards = 18 :=
by
  intros h1 h2
  sorry

end Mike_got_18_cards_l29_2998


namespace B_subset_A_iff_a_range_l29_2946

variable (a : ℝ)
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

theorem B_subset_A_iff_a_range :
  B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
by
  sorry

end B_subset_A_iff_a_range_l29_2946


namespace shirts_sold_l29_2932

theorem shirts_sold (initial_shirts remaining_shirts shirts_sold : ℕ) (h1 : initial_shirts = 49) (h2 : remaining_shirts = 28) : 
  shirts_sold = initial_shirts - remaining_shirts → 
  shirts_sold = 21 := 
by 
  sorry

end shirts_sold_l29_2932


namespace product_of_fractions_l29_2958

theorem product_of_fractions :
  (3/4) * (4/5) * (5/6) * (6/7) = 3/7 :=
by
  sorry

end product_of_fractions_l29_2958


namespace find_value_of_expression_l29_2980

theorem find_value_of_expression (x y z : ℚ)
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y = 10) : (x + y - z) / 3 = -4 / 9 :=
by sorry

end find_value_of_expression_l29_2980


namespace theater_ticket_problem_l29_2964

noncomputable def total_cost_proof (x : ℝ) : Prop :=
  let cost_adult_tickets := 10 * x
  let cost_child_tickets := 8 * (x / 2)
  let cost_senior_tickets := 4 * (0.75 * x)
  cost_adult_tickets + cost_child_tickets + cost_senior_tickets = 58.65

theorem theater_ticket_problem (x : ℝ) (h : 6 * x + 5 * (x / 2) + 3 * (0.75 * x) = 42) : 
  total_cost_proof x :=
by
  sorry

end theater_ticket_problem_l29_2964


namespace log5_x_l29_2972

theorem log5_x (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ (Real.log 16 / Real.log 2) ^ 2) :
    Real.log x / Real.log 5 = -16 / (Real.log 2 / Real.log 5) := by
  sorry

end log5_x_l29_2972


namespace matrix_equation_l29_2978

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 5], ![-6, -2]]
def p : ℤ := 2
def q : ℤ := -18

theorem matrix_equation :
  M * M = p • M + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  sorry

end matrix_equation_l29_2978


namespace part1_part2_l29_2968

noncomputable def Sn (a : ℕ → ℚ) (n : ℕ) (p : ℚ) : ℚ :=
4 * a n - p

theorem part1 (a : ℕ → ℚ) (S : ℕ → ℚ) (p : ℚ) (hp : p ≠ 0)
  (hS : ∀ n, S n = Sn a n p) : 
  ∃ q, ∀ n, a (n + 1) = q * a n :=
sorry

noncomputable def an_formula (n : ℕ) : ℚ := (4/3)^(n - 1)

theorem part2 (b : ℕ → ℚ) (a : ℕ → ℚ)
  (p : ℚ) (hp : p = 3)
  (hb : b 1 = 2)
  (ha1 : a 1 = 1) 
  (h_rec : ∀ n, b (n + 1) = b n + a n) :
  ∀ n, b n = 3 * ((4/3)^(n - 1)) - 1 :=
sorry

end part1_part2_l29_2968


namespace inverse_prop_l29_2911

theorem inverse_prop (a c : ℝ) : (∀ (a : ℝ), a > 0 → a * c^2 ≥ 0) → (∀ (x : ℝ), x * c^2 ≥ 0 → x > 0) :=
by
  sorry

end inverse_prop_l29_2911


namespace value_of_alpha_beta_l29_2904

variable (α β : ℝ)

-- Conditions
def quadratic_eq (x: ℝ) : Prop := x^2 + 2*x - 2005 = 0

-- Lean 4 statement
theorem value_of_alpha_beta 
  (hα : quadratic_eq α) 
  (hβ : quadratic_eq β)
  (sum_roots : α + β = -2) :
  α^2 + 3*α + β = 2003 :=
sorry

end value_of_alpha_beta_l29_2904


namespace most_likely_maximum_people_in_room_l29_2907

theorem most_likely_maximum_people_in_room :
  ∃ k, 1 ≤ k ∧ k ≤ 3000 ∧
    (∃ p : ℕ → ℕ → ℕ → ℕ, (p 1000 1000 1000) = 1019) ∧
    (∀ a b c : ℕ, a + b + c = 3000 → a ≤ 1019 ∧ b ≤ 1019 ∧ c ≤ 1019 → max a (max b c) = 1019) :=
sorry

end most_likely_maximum_people_in_room_l29_2907


namespace find_a_degree_l29_2956

-- Definitions from conditions
def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

-- Statement of the proof problem
theorem find_a_degree (a : ℕ) (h : monomial_degree 2 a = 6) : a = 4 :=
by
  sorry

end find_a_degree_l29_2956


namespace calculate_expression_l29_2922

theorem calculate_expression : 4 + (-8) / (-4) - (-1) = 7 := 
by 
  sorry

end calculate_expression_l29_2922


namespace license_plate_configurations_l29_2965

theorem license_plate_configurations :
  (3 * 10^4 = 30000) :=
by
  sorry

end license_plate_configurations_l29_2965


namespace members_on_fathers_side_are_10_l29_2914

noncomputable def members_father_side (total : ℝ) (ratio : ℝ) (members_mother_side_more: ℝ) : Prop :=
  let F := total / (1 + ratio)
  F = 10

theorem members_on_fathers_side_are_10 :
  ∀ (total : ℝ) (ratio : ℝ), 
  total = 23 → 
  ratio = 0.30 →
  members_father_side total ratio (ratio * total) :=
by
  intros total ratio htotal hratio
  have h1 : total = 23 := htotal
  have h2 : ratio = 0.30 := hratio
  rw [h1, h2]
  sorry

end members_on_fathers_side_are_10_l29_2914


namespace circumscribed_circle_center_location_l29_2933

structure Trapezoid where
  is_isosceles : Bool
  angle_base : ℝ
  angle_between_diagonals : ℝ

theorem circumscribed_circle_center_location (T : Trapezoid)
  (h1 : T.is_isosceles = true)
  (h2 : T.angle_base = 50)
  (h3 : T.angle_between_diagonals = 40) :
  ∃ loc : String, loc = "Outside" := by
  sorry

end circumscribed_circle_center_location_l29_2933


namespace transform_correct_l29_2919

variable {α : Type} [Mul α] [DecidableEq α]

theorem transform_correct (a b c : α) (h : a = b) : a * c = b * c :=
by sorry

end transform_correct_l29_2919


namespace new_home_fraction_l29_2951

variable {M H G : ℚ} -- Use ℚ (rational numbers)

def library_fraction (H : ℚ) (G : ℚ) (M : ℚ) : ℚ :=
  (1 / 3 * H + 2 / 5 * G + 1 / 2 * M) / M

theorem new_home_fraction (H_eq : H = 1 / 2 * M) (G_eq : G = 3 * H) :
  library_fraction H G M = 29 / 30 :=
by
  sorry

end new_home_fraction_l29_2951


namespace more_red_balls_l29_2954

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l29_2954


namespace tangent_line_eqn_of_sine_at_point_l29_2931

theorem tangent_line_eqn_of_sine_at_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  ∀ (p : ℝ × ℝ), p = (0, Real.sqrt 3 / 2) →
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x, f x = Real.sin (x + Real.pi / 3)) ∧
  (∀ x y, y = f x → a * x + b * y + c = 0 → x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry

end tangent_line_eqn_of_sine_at_point_l29_2931


namespace erica_earnings_l29_2902

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l29_2902


namespace base_k_to_decimal_l29_2955

theorem base_k_to_decimal (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
  sorry

end base_k_to_decimal_l29_2955


namespace girls_in_school_l29_2908

noncomputable def num_of_girls (total_students : ℕ) (sampled_students : ℕ) (sampled_diff : ℤ) : ℕ :=
  sorry

theorem girls_in_school :
  let total_students := 1600
  let sampled_students := 200
  let sampled_diff := 10
  num_of_girls total_students sampled_students sampled_diff = 760 :=
  sorry

end girls_in_school_l29_2908


namespace sub_base8_l29_2992

theorem sub_base8 : (1352 - 674) == 1456 :=
by sorry

end sub_base8_l29_2992


namespace find_other_discount_l29_2925

theorem find_other_discount (P F d1 : ℝ) (H₁ : P = 70) (H₂ : F = 61.11) (H₃ : d1 = 10) : ∃ (d2 : ℝ), d2 = 3 :=
by 
  -- The proof will be provided here.
  sorry

end find_other_discount_l29_2925


namespace fraction_simplify_l29_2957

variable (a b c : ℝ)

theorem fraction_simplify
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : a + 2 * b + 3 * c ≠ 0) :
  (a^2 + 4 * b^2 - 9 * c^2 + 4 * a * b) / (a^2 + 9 * c^2 - 4 * b^2 + 6 * a * c) =
  (a + 2 * b - 3 * c) / (a - 2 * b + 3 * c) := by
  sorry

end fraction_simplify_l29_2957


namespace distance_between_trees_l29_2920

theorem distance_between_trees (L : ℕ) (n : ℕ) (hL : L = 150) (hn : n = 11) (h_end_trees : n > 1) : 
  (L / (n - 1)) = 15 :=
by
  -- Replace with the appropriate proof
  sorry

end distance_between_trees_l29_2920


namespace num_C_atoms_in_compound_l29_2969

def num_H_atoms := 6
def num_O_atoms := 1
def molecular_weight := 58
def atomic_weight_C := 12
def atomic_weight_H := 1
def atomic_weight_O := 16

theorem num_C_atoms_in_compound : 
  ∃ (num_C_atoms : ℕ), 
    molecular_weight = (num_C_atoms * atomic_weight_C) + (num_H_atoms * atomic_weight_H) + (num_O_atoms * atomic_weight_O) ∧ 
    num_C_atoms = 3 :=
by
  -- To be proven
  sorry

end num_C_atoms_in_compound_l29_2969


namespace sample_size_l29_2977

theorem sample_size (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : 8 = n * 2 / 10) : n = 40 :=
by
  sorry

end sample_size_l29_2977


namespace bread_left_l29_2927

def initial_bread : ℕ := 1000
def bomi_ate : ℕ := 350
def yejun_ate : ℕ := 500

theorem bread_left : initial_bread - (bomi_ate + yejun_ate) = 150 :=
by
  sorry

end bread_left_l29_2927


namespace product_of_repeating145_and_11_equals_1595_over_999_l29_2905

-- Defining the repeating decimal as a fraction
def repeating145_as_fraction : ℚ :=
  145 / 999

-- Stating the main theorem
theorem product_of_repeating145_and_11_equals_1595_over_999 :
  11 * repeating145_as_fraction = 1595 / 999 :=
by
  sorry

end product_of_repeating145_and_11_equals_1595_over_999_l29_2905


namespace arithmetic_sequence_sum_l29_2923

theorem arithmetic_sequence_sum 
  (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n : ℕ, a n = 2 + (n - 5)) 
  (ha5 : a 5 = 2) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9) := 
by 
  sorry

end arithmetic_sequence_sum_l29_2923


namespace value_diff_l29_2900

theorem value_diff (a b : ℕ) (h1 : a * b = 2 * (a + b) + 14) (h2 : b = 8) : b - a = 3 :=
by
  sorry

end value_diff_l29_2900


namespace sin_neg_30_eq_neg_half_l29_2974

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end sin_neg_30_eq_neg_half_l29_2974


namespace find_ordered_pair_l29_2994

theorem find_ordered_pair (a b : ℚ) :
  a • (⟨2, 3⟩ : ℚ × ℚ) + b • (⟨-2, 5⟩ : ℚ × ℚ) = (⟨10, -8⟩ : ℚ × ℚ) →
  (a, b) = (17 / 8, -23 / 8) :=
by
  intro h
  sorry

end find_ordered_pair_l29_2994


namespace total_attendance_l29_2915

-- Defining the given conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 1
def total_amount_collected : ℕ := 50
def number_of_child_tickets : ℕ := 18

-- Formulating the proof problem
theorem total_attendance (A : ℕ) (C : ℕ) (H1 : C = number_of_child_tickets)
  (H2 : adult_ticket_cost * A + child_ticket_cost * C = total_amount_collected) :
  A + C = 22 := by
  sorry

end total_attendance_l29_2915


namespace minimum_perimeter_rectangle_l29_2950

theorem minimum_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ x y : ℝ, (x * y = S) ∧ (∀ u v : ℝ, (u * v = S) → (2 * (u + v) ≥ 4 * Real.sqrt S)) ∧ (x = Real.sqrt S ∧ y = Real.sqrt S) :=
by
  sorry

end minimum_perimeter_rectangle_l29_2950


namespace point_in_fourth_quadrant_l29_2903

theorem point_in_fourth_quadrant (m : ℝ) : 0 < m ∧ 2 - m < 0 ↔ m > 2 := 
by 
  sorry

end point_in_fourth_quadrant_l29_2903


namespace system1_solution_system2_solution_l29_2918

theorem system1_solution (x y : ℤ) (h1 : x - y = 2) (h2 : x + 1 = 2 * (y - 1)) :
  x = 7 ∧ y = 5 :=
sorry

theorem system2_solution (x y : ℤ) (h1 : 2 * x + 3 * y = 1) (h2 : (y - 1) * 3 = (x - 2) * 4) :
  x = 1 ∧ y = -1 / 3 :=
sorry

end system1_solution_system2_solution_l29_2918


namespace total_handshakes_l29_2975

theorem total_handshakes (twins_num : ℕ) (triplets_num : ℕ) (twins_sets : ℕ) (triplets_sets : ℕ) (h_twins : twins_sets = 9) (h_triplets : triplets_sets = 6) (h_twins_num : twins_num = 2 * twins_sets) (h_triplets_num: triplets_num = 3 * triplets_sets) (h_handshakes : twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2) = 882): 
  (twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2)) / 2 = 441 :=
by
  sorry

end total_handshakes_l29_2975


namespace sum_a2_a9_l29_2929

variable {a : ℕ → ℝ} -- Define the sequence a_n
variable {S : ℕ → ℝ} -- Define the sum sequence S_n

-- The conditions
def arithmetic_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = (n * (a 1 + a n)) / 2

axiom S_10 : arithmetic_sum S a 10
axiom S_10_value : S 10 = 100

-- The goal
theorem sum_a2_a9 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 10 = 100) (h2 : arithmetic_sum S a 10) :
  a 2 + a 9 = 20 := 
sorry

end sum_a2_a9_l29_2929


namespace diff_of_squares_div_l29_2981

theorem diff_of_squares_div (a b : ℤ) (h1 : a = 121) (h2 : b = 112) : 
  (a^2 - b^2) / (a - b) = a + b :=
by
  rw [h1, h2]
  rw [sub_eq_add_neg, add_comm]
  exact sorry

end diff_of_squares_div_l29_2981


namespace number_is_3034_l29_2959

theorem number_is_3034 (number : ℝ) (h : number - 1002 / 20.04 = 2984) : number = 3034 :=
sorry

end number_is_3034_l29_2959


namespace fractional_eq_solution_l29_2999

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l29_2999


namespace exists_sol_in_naturals_l29_2984

theorem exists_sol_in_naturals : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := 
sorry

end exists_sol_in_naturals_l29_2984


namespace candy_problem_l29_2961

theorem candy_problem 
  (weightA costA : ℕ) (weightB costB : ℕ) (avgPrice per100 : ℕ)
  (hA : weightA = 300) (hCostA : costA = 5)
  (hCostB : costB = 7) (hAvgPrice : avgPrice = 150) (hPer100 : per100 = 100)
  (totalCost : ℕ) (hTotalCost : totalCost = costA + costB)
  (totalWeight : ℕ) (hTotalWeight : totalWeight = (totalCost * per100) / avgPrice) :
  (totalWeight = weightA + weightB) -> 
  weightB = 500 :=
by {
  sorry
}

end candy_problem_l29_2961


namespace log_base_250_2662sqrt10_l29_2985

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variables (a b : ℝ)
variables (h1 : log_base 50 55 = a) (h2 : log_base 55 20 = b)

theorem log_base_250_2662sqrt10 : log_base 250 (2662 * Real.sqrt 10) = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) :=
by
  sorry

end log_base_250_2662sqrt10_l29_2985


namespace real_part_z_pow_2017_l29_2935

open Complex

noncomputable def z : ℂ := 1 + I

theorem real_part_z_pow_2017 : re (z ^ 2017) = 2 ^ 1008 := sorry

end real_part_z_pow_2017_l29_2935


namespace circles_intersect_l29_2947

theorem circles_intersect (m : ℝ) 
  (h₁ : ∃ x y, x^2 + y^2 = m) 
  (h₂ : ∃ x y, x^2 + y^2 + 6*x - 8*y + 21 = 0) : 
  9 < m ∧ m < 49 :=
by sorry

end circles_intersect_l29_2947


namespace number_of_members_l29_2942

-- Define the conditions
def knee_pad_cost : ℕ := 6
def jersey_cost : ℕ := knee_pad_cost + 7
def wristband_cost : ℕ := jersey_cost + 3
def cost_per_member : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)
def total_expenditure : ℕ := 4080

-- Prove the number of members in the club
theorem number_of_members (h1 : knee_pad_cost = 6)
                          (h2 : jersey_cost = 13)
                          (h3 : wristband_cost = 16)
                          (h4 : cost_per_member = 70)
                          (h5 : total_expenditure = 4080) :
                          total_expenditure / cost_per_member = 58 := 
by 
  sorry

end number_of_members_l29_2942


namespace find_length_second_platform_l29_2921

noncomputable def length_second_platform : Prop :=
  let train_length := 500  -- in meters
  let time_cross_platform := 35  -- in seconds
  let time_cross_pole := 8  -- in seconds
  let second_train_length := 250  -- in meters
  let time_cross_second_train := 45  -- in seconds
  let platform1_scale := 0.75
  let time_cross_platform1 := 27  -- in seconds
  let train_speed := train_length / time_cross_pole
  let platform1_length := train_speed * time_cross_platform1 - train_length
  let platform2_length := platform1_length / platform1_scale
  platform2_length = 1583.33

/- The proof is omitted -/
theorem find_length_second_platform : length_second_platform := sorry

end find_length_second_platform_l29_2921


namespace tan_alpha_value_l29_2971

theorem tan_alpha_value (α β : ℝ) (h₁ : Real.tan (α + β) = 3) (h₂ : Real.tan β = 2) : 
  Real.tan α = 1 / 7 := 
by 
  sorry

end tan_alpha_value_l29_2971


namespace find_m_l29_2973

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (m : ℕ)

theorem find_m (h1 : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
               (h2 : S (2 * m - 1) = 39)       -- sum of first (2m-1) terms
               (h3 : a (m - 1) + a (m + 1) - a m - 1 = 0)
               (h4 : m > 1) : 
               m = 20 :=
   sorry

end find_m_l29_2973


namespace sum_x_coordinates_Q4_is_3000_l29_2941

-- Let Q1 be a 150-gon with vertices having x-coordinates summing to 3000
def Q1_x_sum := 3000
def Q2_x_sum := Q1_x_sum
def Q3_x_sum := Q2_x_sum
def Q4_x_sum := Q3_x_sum

-- Theorem to prove the sum of the x-coordinates of the vertices of Q4 is 3000
theorem sum_x_coordinates_Q4_is_3000 : Q4_x_sum = 3000 := by
  sorry

end sum_x_coordinates_Q4_is_3000_l29_2941


namespace problem_l29_2952

noncomputable def number_of_regions_four_planes (h1 : True) (h2 : True) : ℕ := 14

theorem problem (h1 : True) (h2 : True) : number_of_regions_four_planes h1 h2 = 14 :=
by sorry

end problem_l29_2952


namespace bug_at_vertex_A_after_8_meters_l29_2912

theorem bug_at_vertex_A_after_8_meters (P : ℕ → ℚ) (h₀ : P 0 = 1)
(h : ∀ n, P (n + 1) = 1/3 * (1 - P n)) : 
P 8 = 1823 / 6561 := 
sorry

end bug_at_vertex_A_after_8_meters_l29_2912


namespace condition_on_a_and_b_l29_2901

theorem condition_on_a_and_b (a b p q : ℝ) 
    (h1 : (∀ x : ℝ, (x + a) * (x + b) = x^2 + p * x + q))
    (h2 : p > 0)
    (h3 : q < 0) :
    (a < 0 ∧ b > 0 ∧ b > -a) ∨ (a > 0 ∧ b < 0 ∧ a > -b) :=
by
  sorry

end condition_on_a_and_b_l29_2901


namespace software_price_l29_2910

theorem software_price (copies total_revenue : ℝ) (P : ℝ) 
  (h1 : copies = 1200)
  (h2 : 0.5 * copies * P + 0.6 * (2 / 3) * (copies - 0.5 * copies) * P + 0.25 * (copies - 0.5 * copies - (2 / 3) * (copies - 0.5 * copies)) * P = total_revenue)
  (h3 : total_revenue = 72000) :
  P = 80.90 :=
by
  sorry

end software_price_l29_2910


namespace least_prime_in_sum_even_set_of_7_distinct_primes_l29_2937

noncomputable def is_prime (n : ℕ) : Prop := sorry -- Assume an implementation of prime numbers

theorem least_prime_in_sum_even_set_of_7_distinct_primes {q : Finset ℕ} 
  (hq_distinct : q.card = 7) 
  (hq_primes : ∀ n ∈ q, is_prime n) 
  (hq_sum_even : q.sum id % 2 = 0) :
  ∃ m ∈ q, m = 2 :=
by
  sorry

end least_prime_in_sum_even_set_of_7_distinct_primes_l29_2937


namespace total_cost_8_dozen_pencils_2_dozen_notebooks_l29_2991

variable (P N : ℝ)

def eq1 : Prop := 3 * P + 4 * N = 60
def eq2 : Prop := P + N = 15.512820512820513

theorem total_cost_8_dozen_pencils_2_dozen_notebooks :
  eq1 P N ∧ eq2 P N → (96 * P + 24 * N = 520) :=
by
  sorry

end total_cost_8_dozen_pencils_2_dozen_notebooks_l29_2991


namespace student_correct_answers_l29_2967

theorem student_correct_answers (C W : ℕ) (h₁ : C + W = 50) (h₂ : 4 * C - W = 130) : C = 36 := 
by
  sorry

end student_correct_answers_l29_2967


namespace flight_time_is_10_hours_l29_2970

def time_watching_TV_episodes : ℕ := 3 * 25
def time_sleeping : ℕ := 4 * 60 + 30
def time_watching_movies : ℕ := 2 * (1 * 60 + 45)
def remaining_flight_time : ℕ := 45

def total_flight_time : ℕ := (time_watching_TV_episodes + time_sleeping + time_watching_movies + remaining_flight_time) / 60

theorem flight_time_is_10_hours : total_flight_time = 10 := by
  sorry

end flight_time_is_10_hours_l29_2970


namespace find_y_coordinate_of_P_l29_2963

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (-3, 2)
noncomputable def C : ℝ × ℝ := (3, 2)
noncomputable def D : ℝ × ℝ := (4, 0)
noncomputable def ell1 (P : ℝ × ℝ) : Prop := (P.1 + 4) ^ 2 / 25 + (P.2) ^ 2 / 9 = 1
noncomputable def ell2 (P : ℝ × ℝ) : Prop := (P.1 + 3) ^ 2 / 25 + ((P.2 - 2) ^ 2) / 16 = 1

theorem find_y_coordinate_of_P :
  ∃ y : ℝ,
    ell1 (0, y) ∧ ell2 (0, y) ∧
    y = 6 / 7 ∧
    6 + 7 = 13 :=
by
  sorry

end find_y_coordinate_of_P_l29_2963


namespace probability_of_divisibility_l29_2960

noncomputable def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def is_prime_digit_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, is_prime_digit d

noncomputable def is_divisible_by_3_and_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem probability_of_divisibility (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999 ∨ 10 ≤ n ∧ n ≤ 99) →
  is_prime_digit_number n →
  ¬ is_divisible_by_3_and_4 n :=
by
  intros h1 h2
  sorry

end probability_of_divisibility_l29_2960


namespace prove_A_plus_B_plus_1_l29_2924

theorem prove_A_plus_B_plus_1 (A B : ℤ) 
  (h1 : B = A + 2)
  (h2 : 2 * A^2 + A + 6 + 5 * B + 2 = 7 * (A + B + 1) + 5) :
  A + B + 1 = 15 :=
by 
  sorry

end prove_A_plus_B_plus_1_l29_2924


namespace concentrate_to_water_ratio_l29_2948

theorem concentrate_to_water_ratio :
  ∀ (c w : ℕ), (∀ c, w = 3 * c) → (35 * 3 = 105) → (1 / 3 = (1 : ℝ) / (3 : ℝ)) :=
by
  intros c w h1 h2
  sorry

end concentrate_to_water_ratio_l29_2948


namespace cubic_equation_root_sum_l29_2934

theorem cubic_equation_root_sum (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p * q + p * r + q * r = 11) (h3 : p * q * r = 6) :
  (p * q / r + p * r / q + q * r / p) = 49 / 6 := sorry

end cubic_equation_root_sum_l29_2934


namespace total_spots_l29_2906

-- Define the variables
variables (R C G S B : ℕ)

-- State the problem conditions
def conditions : Prop :=
  R = 46 ∧
  C = R / 2 - 5 ∧
  G = 5 * C ∧
  S = 3 * R ∧
  B = 2 * (G + S)

-- State the proof problem
theorem total_spots : conditions R C G S B → G + C + S + B = 702 :=
by
  intro h
  obtain ⟨hR, hC, hG, hS, hB⟩ := h
  -- The proof steps would go here
  sorry

end total_spots_l29_2906


namespace rationalize_denominator_XYZ_sum_l29_2936

noncomputable def a := (5 : ℝ)^(1/3)
noncomputable def b := (4 : ℝ)^(1/3)

theorem rationalize_denominator_XYZ_sum : 
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  X + Y + Z + W = 62 :=
by 
  sorry

end rationalize_denominator_XYZ_sum_l29_2936


namespace point_not_on_line_pq_neg_l29_2995

theorem point_not_on_line_pq_neg (p q : ℝ) (h : p * q < 0) : ¬ (21 * p + q = -101) := 
by sorry

end point_not_on_line_pq_neg_l29_2995


namespace polar_to_cartesian_l29_2926

theorem polar_to_cartesian :
  ∀ (ρ θ : ℝ), ρ = 3 ∧ θ = π / 6 → 
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  intro ρ θ
  rintro ⟨hρ, hθ⟩
  rw [hρ, hθ]
  sorry

end polar_to_cartesian_l29_2926


namespace range_of_a_l29_2953

def f (a x : ℝ) : ℝ := a * x^2 - 2 * x - |x^2 - a * x + 1|

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, x ≠ x₁ ∧ x ≠ x₂ → f a x ≠ 0

theorem range_of_a :
  { a : ℝ | has_exactly_two_zeros a } =
  { a : ℝ | (a < 0) ∨ (0 < a ∧ a < 1) ∨ (1 < a) } :=
sorry

end range_of_a_l29_2953


namespace spending_spring_months_l29_2990

variable (s_feb s_may : ℝ)

theorem spending_spring_months (h1 : s_feb = 2.8) (h2 : s_may = 5.6) : s_may - s_feb = 2.8 := 
by
  sorry

end spending_spring_months_l29_2990


namespace inequality_solution_l29_2928

theorem inequality_solution (x : ℝ) : (x^2 - x - 2 < 0) ↔ (-1 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_l29_2928


namespace student_rank_left_l29_2986

theorem student_rank_left {n m : ℕ} (h1 : n = 10) (h2 : m = 6) : (n - m + 1) = 5 := by
  sorry

end student_rank_left_l29_2986


namespace total_sentence_l29_2989

theorem total_sentence (base_rate : ℝ) (value_stolen : ℝ) (third_offense_increase : ℝ) (additional_years : ℕ) : 
  base_rate = 1 / 5000 → 
  value_stolen = 40000 → 
  third_offense_increase = 0.25 → 
  additional_years = 2 →
  (value_stolen * base_rate * (1 + third_offense_increase) + additional_years) = 12 := 
by
  intros
  sorry

end total_sentence_l29_2989


namespace neg_prop_p_equiv_l29_2939

open Classical

variable (x : ℝ)
def prop_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 0

theorem neg_prop_p_equiv : ¬ prop_p ↔ ∃ x : ℝ, x^2 + 1 < 0 := by
  sorry

end neg_prop_p_equiv_l29_2939


namespace union_A_B_compl_inter_A_B_l29_2949

-- Definitions based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2 * x - 9 ≥ 6 - 3 * x}

-- The first proof statement
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by
  sorry

-- The second proof statement
theorem compl_inter_A_B : U \ (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 4} := by
  sorry

end union_A_B_compl_inter_A_B_l29_2949


namespace supplement_of_angle_l29_2997

-- Condition: The complement of angle α is 54 degrees 32 minutes
theorem supplement_of_angle (α : ℝ) (h : α = 90 - (54 + 32 / 60)) :
  180 - α = 144 + 32 / 60 := by
sorry

end supplement_of_angle_l29_2997


namespace xyz_problem_l29_2945

theorem xyz_problem (x y : ℝ) (h1 : x + y - x * y = 155) (h2 : x^2 + y^2 = 325) : |x^3 - y^3| = 4375 := by
  sorry

end xyz_problem_l29_2945


namespace elephants_at_WePreserveForFuture_l29_2983

theorem elephants_at_WePreserveForFuture (E : ℕ) 
  (h1 : ∀ gest : ℕ, gest = 3 * E)
  (h2 : ∀ total : ℕ, total = E + 3 * E) 
  (h3 : total = 280) : 
  E = 70 := 
by
  sorry

end elephants_at_WePreserveForFuture_l29_2983


namespace cube_identity_l29_2943

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l29_2943


namespace not_both_hit_prob_l29_2976

-- Defining the probabilities
def prob_archer_A_hits : ℚ := 1 / 3
def prob_archer_B_hits : ℚ := 1 / 2

-- Defining event B as both hit the bullseye
def prob_both_hit : ℚ := prob_archer_A_hits * prob_archer_B_hits

-- Defining the complementary event of not both hitting the bullseye
def prob_not_both_hit : ℚ := 1 - prob_both_hit

theorem not_both_hit_prob : prob_not_both_hit = 5 / 6 := by
  -- This is the statement we are trying to prove.
  sorry

end not_both_hit_prob_l29_2976


namespace project_hours_l29_2987

variable (K P M : ℕ)

theorem project_hours
  (h1 : P + K + M = 144)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 80 :=
sorry

end project_hours_l29_2987


namespace problem1_l29_2993

theorem problem1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end problem1_l29_2993


namespace binary_multiplication_correct_l29_2996

theorem binary_multiplication_correct:
  let n1 := 29 -- binary 11101 is decimal 29
  let n2 := 13 -- binary 1101 is decimal 13
  let result := 303 -- binary 100101111 is decimal 303
  n1 * n2 = result :=
by
  -- Proof goes here
  sorry

end binary_multiplication_correct_l29_2996


namespace final_price_of_jacket_l29_2916

noncomputable def originalPrice : ℝ := 250
noncomputable def firstDiscount : ℝ := 0.60
noncomputable def secondDiscount : ℝ := 0.25

theorem final_price_of_jacket :
  let P := originalPrice
  let D1 := firstDiscount
  let D2 := secondDiscount
  let priceAfterFirstDiscount := P * (1 - D1)
  let finalPrice := priceAfterFirstDiscount * (1 - D2)
  finalPrice = 75 :=
by
  sorry

end final_price_of_jacket_l29_2916
