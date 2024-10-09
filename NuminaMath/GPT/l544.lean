import Mathlib

namespace garden_dimensions_l544_54431

theorem garden_dimensions
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l * w = 600) : 
  w = 10 * Real.sqrt 3 ∧ l = 20 * Real.sqrt 3 :=
by
  sorry

end garden_dimensions_l544_54431


namespace wilson_fraction_l544_54472

theorem wilson_fraction (N : ℝ) (result : ℝ) (F : ℝ) (h1 : N = 8) (h2 : result = 16 / 3) (h3 : N - F * N = result) : F = 1 / 3 := 
by
  sorry

end wilson_fraction_l544_54472


namespace initially_calculated_average_l544_54483

theorem initially_calculated_average :
  ∀ (S : ℕ), (S / 10 = 18) →
  ((S - 46 + 26) / 10 = 16) :=
by
  sorry

end initially_calculated_average_l544_54483


namespace sequence_properties_l544_54421

variable {Seq : Nat → ℕ}
-- Given conditions: Sn = an(an + 3) / 6
def Sn (n : ℕ) := Seq n * (Seq n + 3) / 6

theorem sequence_properties :
  (Seq 1 = 3) ∧ (Seq 2 = 9) ∧ (∀ n : ℕ, Seq (n+1) = 3 * (n + 1)) :=
by 
  have h1 : Sn 1 = (Seq 1 * (Seq 1 + 3)) / 6 := rfl
  have h2 : Sn 2 = (Seq 2 * (Seq 2 + 3)) / 6 := rfl
  sorry

end sequence_properties_l544_54421


namespace power_evaluation_l544_54436

theorem power_evaluation (x : ℕ) (h1 : 3^x = 81) : 3^(x+2) = 729 := by
  sorry

end power_evaluation_l544_54436


namespace greatest_number_of_bouquets_l544_54481

/--
Sara has 42 red flowers, 63 yellow flowers, and 54 blue flowers.
She wants to make bouquets with the same number of each color flower in each bouquet.
Prove that the greatest number of bouquets she can make is 21.
-/
theorem greatest_number_of_bouquets (red yellow blue : ℕ) (h_red : red = 42) (h_yellow : yellow = 63) (h_blue : blue = 54) :
  Nat.gcd (Nat.gcd red yellow) blue = 21 :=
by
  rw [h_red, h_yellow, h_blue]
  sorry

end greatest_number_of_bouquets_l544_54481


namespace meryll_questions_l544_54470

theorem meryll_questions (M P : ℕ) (h1 : (3/5 : ℚ) * M + (2/3 : ℚ) * P = 31) (h2 : P = 15) : M = 35 :=
sorry

end meryll_questions_l544_54470


namespace smallest_numbers_l544_54498

-- Define the problem statement
theorem smallest_numbers (m n : ℕ) :
  (∃ (m1 n1 m2 n2 : ℕ), 7 * m1^2 - 11 * n1^2 = 1 ∧ 7 * m2^2 - 11 * n2^2 = 5) ↔
  (7 * m^2 - 11 * n^2 = 1) ∨ (7 * m^2 - 11 * n^2 = 5) :=
by
  sorry

end smallest_numbers_l544_54498


namespace part3_conclusion_l544_54434

-- Definitions and conditions for the problem
def quadratic_function (a x : ℝ) : ℝ := (x - a)^2 + a - 1

-- Part 1: Given condition that (1, 2) lies on the graph of the quadratic function
def part1_condition (a : ℝ) := (quadratic_function a 1) = 2

-- Part 2: Given condition that the function has a minimum value of 2 for 1 ≤ x ≤ 4
def part2_condition (a : ℝ) := ∀ x, 1 ≤ x ∧ x ≤ 4 → quadratic_function a x ≥ 2

-- Part 3: Given condition (m, n) on the graph where m > 0 and m > 2a
def part3_condition (a m n : ℝ) := m > 0 ∧ m > 2 * a ∧ quadratic_function a m = n

-- Conclusion for Part 3: Prove that n > -5/4
theorem part3_conclusion (a m n : ℝ) (h : part3_condition a m n) : n > -5/4 := 
sorry  -- Proof required here

end part3_conclusion_l544_54434


namespace reflected_ray_eq_l544_54451

theorem reflected_ray_eq:
  ∀ (x y : ℝ), 
    (3 * x + 4 * y - 18 = 0) ∧ (3 * x + 2 * y - 12 = 0) →
    63 * x + 16 * y - 174 = 0 :=
by
  intro x y
  intro h
  sorry

end reflected_ray_eq_l544_54451


namespace cash_sales_amount_l544_54496

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end cash_sales_amount_l544_54496


namespace smallest_integer_cube_ends_in_392_l544_54400

theorem smallest_integer_cube_ends_in_392 : ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 392) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m :=
by 
  sorry

end smallest_integer_cube_ends_in_392_l544_54400


namespace find_f2_l544_54473

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x y : ℝ, x * f y = y * f x) (h10 : f 10 = 30) : f 2 = 6 := 
by
  sorry

end find_f2_l544_54473


namespace stamps_ratio_l544_54441

theorem stamps_ratio (orig_stamps_P : ℕ) (addie_stamps : ℕ) (final_stamps_P : ℕ) 
  (h₁ : orig_stamps_P = 18) (h₂ : addie_stamps = 72) (h₃ : final_stamps_P = 36) :
  (final_stamps_P - orig_stamps_P) / addie_stamps = 1 / 4 :=
by {
  sorry
}

end stamps_ratio_l544_54441


namespace squares_not_all_congruent_l544_54465

/-- Proof that the statement "all squares are congruent to each other" is false. -/
theorem squares_not_all_congruent : ¬(∀ (a b : ℝ), a = b ↔ a = b) :=
by 
  sorry

end squares_not_all_congruent_l544_54465


namespace inequality_1_inequality_2_l544_54443

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end inequality_1_inequality_2_l544_54443


namespace tangent_line_equation_l544_54408

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 5

def point_A : ℝ × ℝ := (1, -2)

theorem tangent_line_equation :
  ∀ x y : ℝ, (y = 4 * x - 6) ↔ (fderiv ℝ f (point_A.1) x = 4) ∧ (y = f (point_A.1) + 4 * (x - point_A.1)) := by
  sorry

end tangent_line_equation_l544_54408


namespace matchsticks_left_l544_54479

def initial_matchsticks : ℕ := 30
def matchsticks_needed_2 : ℕ := 5
def matchsticks_needed_0 : ℕ := 6
def num_2s : ℕ := 3
def num_0s : ℕ := 1

theorem matchsticks_left : 
  initial_matchsticks - (num_2s * matchsticks_needed_2 + num_0s * matchsticks_needed_0) = 9 :=
by sorry

end matchsticks_left_l544_54479


namespace solve_for_a_l544_54426

theorem solve_for_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
sorry

end solve_for_a_l544_54426


namespace value_of_3b_minus_a_l544_54499

theorem value_of_3b_minus_a :
  ∃ (a b : ℕ), (a > b) ∧ (a >= 0) ∧ (b >= 0) ∧ (∀ x : ℝ, (x - a) * (x - b) = x^2 - 16 * x + 60) ∧ (3 * b - a = 8) := 
sorry

end value_of_3b_minus_a_l544_54499


namespace minimum_cable_length_l544_54466

def station_positions : List ℝ := [0, 3, 7, 11, 14]

def total_cable_length (x : ℝ) : ℝ :=
  abs x + abs (x - 3) + abs (x - 7) + abs (x - 11) + abs (x - 14)

theorem minimum_cable_length :
  (∀ x : ℝ, total_cable_length x ≥ 22) ∧ total_cable_length 7 = 22 :=
by
  sorry

end minimum_cable_length_l544_54466


namespace pants_to_shirts_ratio_l544_54417

-- Conditions
def shirts : ℕ := 4
def total_clothes : ℕ := 16

-- Given P as the number of pants and S as the number of shorts
variable (P S : ℕ)

-- State the conditions as hypotheses
axiom shorts_half_pants : S = P / 2
axiom total_clothes_condition : 4 + P + S = 16

-- Question: Prove that the ratio of pants to shirts is 2
theorem pants_to_shirts_ratio : P = 2 * shirts :=
by {
  -- insert proof steps here
  sorry
}

end pants_to_shirts_ratio_l544_54417


namespace Emilee_earnings_l544_54411

theorem Emilee_earnings (J R_j T R_t E R_e : ℕ) :
  (R_j * J = 35) → 
  (R_t * T = 30) → 
  (R_j * J + R_t * T + R_e * E = 90) → 
  (R_e * E = 25) :=
by
  intros h1 h2 h3
  sorry

end Emilee_earnings_l544_54411


namespace cab_driver_income_l544_54474

theorem cab_driver_income (incomes : Fin 5 → ℝ)
  (h1 : incomes 0 = 250)
  (h2 : incomes 1 = 400)
  (h3 : incomes 2 = 750)
  (h4 : incomes 3 = 400)
  (avg_income : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 460) : 
  incomes 4 = 500 :=
sorry

end cab_driver_income_l544_54474


namespace inequality_proof_l544_54420

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
by 
  sorry

end inequality_proof_l544_54420


namespace quadratic_complete_square_l544_54469

theorem quadratic_complete_square (b m : ℝ) (h1 : b > 0)
    (h2 : (x : ℝ) → (x + m)^2 + 8 = x^2 + bx + 20) : b = 4 * Real.sqrt 3 :=
by
  sorry

end quadratic_complete_square_l544_54469


namespace sum_of_sides_of_regular_pentagon_l544_54494

theorem sum_of_sides_of_regular_pentagon (s : ℝ) (n : ℕ)
    (h : s = 15) (hn : n = 5) : 5 * 15 = 75 :=
sorry

end sum_of_sides_of_regular_pentagon_l544_54494


namespace max_ab_l544_54459

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ab ≤ 1 / 16 :=
by
  sorry

end max_ab_l544_54459


namespace small_cubes_with_painted_faces_l544_54471

-- Definitions based on conditions
def large_cube_edge : ℕ := 8
def small_cube_edge : ℕ := 2
def division_factor : ℕ := large_cube_edge / small_cube_edge
def total_small_cubes : ℕ := division_factor ^ 3

-- Proving the number of cubes with specific painted faces.
theorem small_cubes_with_painted_faces :
  (8 : ℤ) = 8 ∧ -- 8 smaller cubes with three painted faces
  (24 : ℤ) = 24 ∧ -- 24 smaller cubes with two painted faces
  (24 : ℤ) = 24 := -- 24 smaller cubes with one painted face
by
  sorry

end small_cubes_with_painted_faces_l544_54471


namespace min_value_of_sum_of_reciprocals_l544_54450

theorem min_value_of_sum_of_reciprocals 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.log (1 / a + 1 / b) / Real.log 4 = Real.log (1 / Real.sqrt (a * b)) / Real.log 2) : 
  1 / a + 1 / b ≥ 4 := 
by 
  sorry

end min_value_of_sum_of_reciprocals_l544_54450


namespace find_x_l544_54452

theorem find_x (x : ℚ) (h : ⌊x⌋ + x = 15/4) : x = 15/4 := by
  sorry

end find_x_l544_54452


namespace subtracted_number_from_32_l544_54449

theorem subtracted_number_from_32 (x : ℕ) (h : 32 - x = 23) : x = 9 := 
by 
  sorry

end subtracted_number_from_32_l544_54449


namespace determine_denominator_of_fraction_l544_54485

theorem determine_denominator_of_fraction (x : ℝ) (h : 57 / x = 0.0114) : x = 5000 :=
by
  sorry

end determine_denominator_of_fraction_l544_54485


namespace haley_collected_cans_l544_54424

theorem haley_collected_cans (C : ℕ) (h : C - 7 = 2) : C = 9 :=
by {
  sorry
}

end haley_collected_cans_l544_54424


namespace quadratic_roots_relation_l544_54486

theorem quadratic_roots_relation (m p q : ℝ) (h_m_ne_zero : m ≠ 0) (h_p_ne_zero : p ≠ 0) (h_q_ne_zero : q ≠ 0) :
  (∀ r1 r2 : ℝ, (r1 + r2 = -q ∧ r1 * r2 = m) → (3 * r1 + 3 * r2 = -m ∧ (3 * r1) * (3 * r2) = p)) →
  p / q = 27 :=
by
  intros h
  sorry

end quadratic_roots_relation_l544_54486


namespace equilateral_right_triangle_impossible_l544_54461
-- Import necessary library

-- Define the conditions and the problem statement
theorem equilateral_right_triangle_impossible :
  ¬(∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A = B ∧ B = C ∧ (A^2 + B^2 = C^2) ∧ (A + B + C = 180)) := sorry

end equilateral_right_triangle_impossible_l544_54461


namespace average_height_plants_l544_54497

theorem average_height_plants (h1 h3 : ℕ) (h1_eq : h1 = 27) (h3_eq : h3 = 9)
  (prop : ∀ (h2 h4 : ℕ), (h2 = h1 / 3 ∨ h2 = h1 * 3) ∧ (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧ (h4 = h3 / 3 ∨ h4 = h3 * 3)) : 
  ((27 + h2 + 9 + h4) / 4 = 12) :=
by 
  sorry

end average_height_plants_l544_54497


namespace sum_of_transformed_numbers_l544_54418

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_transformed_numbers_l544_54418


namespace curtains_length_needed_l544_54493

def room_height_feet : ℕ := 8
def additional_material_inches : ℕ := 5

def height_in_inches : ℕ := room_height_feet * 12

def total_length_curtains : ℕ := height_in_inches + additional_material_inches

theorem curtains_length_needed : total_length_curtains = 101 := by
  sorry

end curtains_length_needed_l544_54493


namespace cost_of_600_candies_l544_54460

-- Definitions based on conditions
def costOfBox : ℕ := 6       -- The cost of one box of 25 candies in dollars
def boxSize   : ℕ := 25      -- The number of candies in one box
def cost (n : ℕ) : ℕ := (n / boxSize) * costOfBox -- The cost function for n candies

-- Theorem to be proven
theorem cost_of_600_candies : cost 600 = 144 :=
by sorry

end cost_of_600_candies_l544_54460


namespace min_attempts_sufficient_a_l544_54433

theorem min_attempts_sufficient_a (n : ℕ) (h : n > 2)
  (good_batteries bad_batteries : ℕ)
  (h1 : good_batteries = n + 1)
  (h2 : bad_batteries = n)
  (total_batteries := 2 * n + 1) :
  (∃ attempts, attempts = n + 1) := sorry

end min_attempts_sufficient_a_l544_54433


namespace range_of_a_l544_54478

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a

def proposition_q (a : ℝ) : Prop := ∃ (x₀ : ℝ), x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : proposition_p a ∧ proposition_q a ↔ (a = 1 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_l544_54478


namespace minimum_value_4x_minus_y_l544_54414

theorem minimum_value_4x_minus_y (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 4 ≥ 0) (h3 : x ≤ 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), (x' - y' ≥ 0) → (x' + y' - 4 ≥ 0) → (x' ≤ 4) → 4 * x' - y' ≥ m :=
by
  sorry

end minimum_value_4x_minus_y_l544_54414


namespace remainder_when_nm_div_61_l544_54448

theorem remainder_when_nm_div_61 (n m : ℕ) (k j : ℤ):
  n = 157 * k + 53 → m = 193 * j + 76 → (n + m) % 61 = 7 := by
  intros h1 h2
  sorry

end remainder_when_nm_div_61_l544_54448


namespace find_x_value_l544_54416

theorem find_x_value (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
sorry

end find_x_value_l544_54416


namespace dogs_in_garden_l544_54410

theorem dogs_in_garden (D : ℕ) (ducks : ℕ) (total_feet : ℕ) (dogs_feet : ℕ) (ducks_feet : ℕ) 
  (h1 : ducks = 2) 
  (h2 : total_feet = 28)
  (h3 : dogs_feet = 4)
  (h4 : ducks_feet = 2) 
  (h_eq : dogs_feet * D + ducks_feet * ducks = total_feet) : 
  D = 6 := by
  sorry

end dogs_in_garden_l544_54410


namespace hispanic_population_in_west_l544_54492

theorem hispanic_population_in_west (p_NE p_MW p_South p_West : ℕ)
  (h_NE : p_NE = 4)
  (h_MW : p_MW = 5)
  (h_South : p_South = 12)
  (h_West : p_West = 20) :
  ((p_West : ℝ) / (p_NE + p_MW + p_South + p_West : ℝ)) * 100 = 49 :=
by sorry

end hispanic_population_in_west_l544_54492


namespace total_donations_l544_54490

-- Define the conditions
def started_donating_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the proof problem to show the total donation amount equals $432,000
theorem total_donations : (current_age - started_donating_age) * annual_donation = 432000 := 
by
  sorry

end total_donations_l544_54490


namespace sequence_an_general_formula_sum_bn_formula_l544_54404

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

axiom seq_Sn_eq_2an_minus_n : ∀ n : ℕ, n > 0 → S n + n = 2 * a n

theorem sequence_an_general_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, a n + 1 = 2 * (a (n - 1) + 1)) ∧ (a n = 2^n - 1) :=
sorry

theorem sum_bn_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, b n = n * a n + n) → T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_general_formula_sum_bn_formula_l544_54404


namespace game_A_greater_game_B_l544_54484

-- Defining the probabilities and independence condition
def P_H := 2 / 3
def P_T := 1 / 3
def independent_tosses := true

-- Game A Probability Definition
def P_A := (P_H ^ 3) + (P_T ^ 3)

-- Game B Probability Definition
def P_B := ((P_H ^ 2) + (P_T ^ 2)) ^ 2

-- Statement to be proved
theorem game_A_greater_game_B : P_A = (27:ℚ) / 81 ∧ P_B = (25:ℚ) / 81 ∧ ((27:ℚ) / 81 - (25:ℚ) / 81 = (2:ℚ) / 81) := 
by
  -- P_A has already been computed: 1/3 = 27/81
  -- P_B has already been computed: 25/81
  sorry

end game_A_greater_game_B_l544_54484


namespace intersecting_lines_l544_54435

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem intersecting_lines (x y : ℝ) : x ≠ 0 → y ≠ 0 → 
  (diamond x y = diamond y x) ↔ (y = x ∨ y = -x) := 
by
  sorry

end intersecting_lines_l544_54435


namespace sally_picked_peaches_l544_54406

theorem sally_picked_peaches (original_peaches total_peaches picked_peaches : ℕ)
  (h_orig : original_peaches = 13)
  (h_total : total_peaches = 55)
  (h_picked : picked_peaches = total_peaches - original_peaches) :
  picked_peaches = 42 :=
by
  sorry

end sally_picked_peaches_l544_54406


namespace three_digit_addition_l544_54419

theorem three_digit_addition (a b : ℕ) (h₁ : 307 = 300 + a * 10 + 7) (h₂ : 416 + 10 * (a * 1) + 7 = 700 + b * 10 + 3) (h₃ : (7 + b + 3) % 3 = 0) : a + b = 2 :=
by
  -- mock proof, since solution steps are not considered
  sorry

end three_digit_addition_l544_54419


namespace lowest_price_per_component_l544_54463

theorem lowest_price_per_component (cost_per_component shipping_per_component fixed_costs num_components : ℕ) 
  (h_cost_per_component : cost_per_component = 80)
  (h_shipping_per_component : shipping_per_component = 5)
  (h_fixed_costs : fixed_costs = 16500)
  (h_num_components : num_components = 150) :
  (cost_per_component + shipping_per_component) * num_components + fixed_costs = 29250 ∧
  29250 / 150 = 195 :=
by
  sorry

end lowest_price_per_component_l544_54463


namespace part_I_part_II_l544_54412

-- Problem conditions as definitions
variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a + b = 1)

-- Statement for part (Ⅰ)
theorem part_I : (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Statement for part (Ⅱ)
theorem part_II : (1 / (a ^ 2016)) + (1 / (b ^ 2016)) ≥ 2 ^ 2017 :=
by
  sorry

end part_I_part_II_l544_54412


namespace Q_at_1_eq_1_l544_54488

noncomputable def Q (x : ℚ) : ℚ := x^4 - 16*x^2 + 16

theorem Q_at_1_eq_1 : Q 1 = 1 := by
  sorry

end Q_at_1_eq_1_l544_54488


namespace intersection_M_N_l544_54447

open Set

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} :=
by
  sorry

end intersection_M_N_l544_54447


namespace percentage_girls_not_attended_college_l544_54476

-- Definitions based on given conditions
def total_boys : ℕ := 300
def total_girls : ℕ := 240
def percent_boys_not_attended_college : ℚ := 0.30
def percent_class_attended_college : ℚ := 0.70

-- The goal is to prove that the percentage of girls who did not attend college is 30%
theorem percentage_girls_not_attended_college 
  (total_boys : ℕ)
  (total_girls : ℕ)
  (percent_boys_not_attended_college : ℚ)
  (percent_class_attended_college : ℚ)
  (total_students := total_boys + total_girls)
  (boys_not_attended := percent_boys_not_attended_college * total_boys)
  (students_attended := percent_class_attended_college * total_students)
  (students_not_attended := total_students - students_attended)
  (girls_not_attended := students_not_attended - boys_not_attended) :
  (girls_not_attended / total_girls) * 100 = 30 := 
  sorry

end percentage_girls_not_attended_college_l544_54476


namespace overlapped_squares_area_l544_54455

/-- 
Theorem: The area of the figure formed by overlapping four identical squares, 
each with an area of \(3 \, \text{cm}^2\), and with an overlapping region 
that double-counts 6 small squares is \(10.875 \, \text{cm}^2\).
-/
theorem overlapped_squares_area (area_of_square : ℝ) (num_squares : ℕ) (overlap_small_squares : ℕ) :
  area_of_square = 3 → 
  num_squares = 4 → 
  overlap_small_squares = 6 →
  ∃ total_area : ℝ, total_area = (num_squares * area_of_square) - (overlap_small_squares * (area_of_square / 16)) ∧
                         total_area = 10.875 :=
by
  sorry

end overlapped_squares_area_l544_54455


namespace min_stamps_l544_54427

theorem min_stamps : ∃ (x y : ℕ), 5 * x + 7 * y = 35 ∧ x + y = 5 :=
by
  have : ∀ (x y : ℕ), 5 * x + 7 * y = 35 → x + y = 5 → True := sorry
  sorry

end min_stamps_l544_54427


namespace range_of_m_l544_54446

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry

end range_of_m_l544_54446


namespace woody_savings_l544_54464

-- Definitions from conditions
def console_cost : Int := 282
def weekly_allowance : Int := 24
def saving_weeks : Int := 10

-- Theorem to prove that the amount Woody already has is $42
theorem woody_savings :
  (console_cost - (weekly_allowance * saving_weeks)) = 42 := 
by
  sorry

end woody_savings_l544_54464


namespace valid_pair_l544_54468

-- Definitions of the animals
inductive Animal
| lion
| tiger
| leopard
| elephant

open Animal

-- Given conditions
def condition1 (selected : Animal → Prop) : Prop :=
  selected lion → selected tiger

def condition2 (selected : Animal → Prop) : Prop :=
  ¬selected leopard → ¬selected tiger

def condition3 (selected : Animal → Prop) : Prop :=
  selected leopard → ¬selected elephant

-- Main theorem to prove
theorem valid_pair (selected : Animal → Prop) (pair : Animal × Animal) :
  (pair = (tiger, leopard)) ↔ 
  (condition1 selected ∧ condition2 selected ∧ condition3 selected) :=
sorry

end valid_pair_l544_54468


namespace intersection_A_complement_B_l544_54489

open Set

noncomputable def A : Set ℝ := {2, 3, 4, 5, 6}
noncomputable def B : Set ℝ := {x | x^2 - 8 * x + 12 >= 0}
noncomputable def complement_B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem intersection_A_complement_B :
  A ∩ complement_B = {3, 4, 5} :=
sorry

end intersection_A_complement_B_l544_54489


namespace divisors_of_10_factorial_larger_than_9_factorial_l544_54444

theorem divisors_of_10_factorial_larger_than_9_factorial :
  ∃ n, n = 9 ∧ (∀ d, d ∣ (Nat.factorial 10) → d > (Nat.factorial 9) → d > (Nat.factorial 1) → n = 9) :=
sorry

end divisors_of_10_factorial_larger_than_9_factorial_l544_54444


namespace arithmetic_sequence_sum_l544_54415

noncomputable def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 + n * d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 2)
  (h3 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l544_54415


namespace largest_apartment_size_l544_54480

theorem largest_apartment_size (cost_per_sqft : ℝ) (budget : ℝ) (s : ℝ) 
    (h₁ : cost_per_sqft = 1.20) 
    (h₂ : budget = 600) 
    (h₃ : 1.20 * s = 600) : 
    s = 500 := 
  sorry

end largest_apartment_size_l544_54480


namespace gcd_of_60_and_75_l544_54401

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l544_54401


namespace value_of_expression_l544_54453

theorem value_of_expression (m a b c d : ℚ) 
  (hm : |m + 1| = 4)
  (hab : a = -b) 
  (hcd : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 :=
by
  sorry

end value_of_expression_l544_54453


namespace fib_divisibility_l544_54437

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

theorem fib_divisibility (m n : ℕ) (hm : 1 ≤ m) (hn : 1 < n) : 
  (fib (m * n - 1) - fib (n - 1) ^ m) % fib n ^ 2 = 0 :=
sorry

end fib_divisibility_l544_54437


namespace regular_polygon_sides_l544_54403

theorem regular_polygon_sides (O A B : Type) (angle_OAB : ℝ) 
  (h_angle : angle_OAB = 72) : 
  (360 / angle_OAB = 5) := 
by 
  sorry

end regular_polygon_sides_l544_54403


namespace total_clouds_l544_54422

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end total_clouds_l544_54422


namespace trigonometric_identity_cos24_cos36_sub_sin24_cos54_l544_54439

theorem trigonometric_identity_cos24_cos36_sub_sin24_cos54  :
  (Real.cos (24 * Real.pi / 180) * Real.cos (36 * Real.pi / 180) - Real.sin (24 * Real.pi / 180) * Real.cos (54 * Real.pi / 180) = 1 / 2) := by
  sorry

end trigonometric_identity_cos24_cos36_sub_sin24_cos54_l544_54439


namespace area_of_picture_l544_54457

theorem area_of_picture
  (paper_width : ℝ)
  (paper_height : ℝ)
  (left_margin : ℝ)
  (right_margin : ℝ)
  (top_margin_cm : ℝ)
  (bottom_margin_cm : ℝ)
  (cm_per_inch : ℝ)
  (converted_top_margin : ℝ := top_margin_cm * (1 / cm_per_inch))
  (converted_bottom_margin : ℝ := bottom_margin_cm * (1 / cm_per_inch))
  (picture_width : ℝ := paper_width - left_margin - right_margin)
  (picture_height : ℝ := paper_height - converted_top_margin - converted_bottom_margin)
  (area : ℝ := picture_width * picture_height)
  (h1 : paper_width = 8.5)
  (h2 : paper_height = 10)
  (h3 : left_margin = 1.5)
  (h4 : right_margin = 1.5)
  (h5 : top_margin_cm = 2)
  (h6 : bottom_margin_cm = 2.5)
  (h7 : cm_per_inch = 2.54)
  : area = 45.255925 :=
by sorry

end area_of_picture_l544_54457


namespace unique_positive_integer_appending_digits_eq_sum_l544_54445

-- Define the problem in terms of Lean types and properties
theorem unique_positive_integer_appending_digits_eq_sum :
  ∃! (A : ℕ), (A > 0) ∧ (∃ (B : ℕ), (0 ≤ B ∧ B < 1000) ∧ (1000 * A + B = (A * (A + 1)) / 2)) :=
sorry

end unique_positive_integer_appending_digits_eq_sum_l544_54445


namespace initial_red_marbles_l544_54425

variable (r g : ℝ)

def red_green_ratio_initial (r g : ℝ) : Prop := r / g = 5 / 3
def red_green_ratio_new (r g : ℝ) : Prop := (r + 15) / (g - 9) = 3 / 1

theorem initial_red_marbles (r g : ℝ) (h₁ : red_green_ratio_initial r g) (h₂ : red_green_ratio_new r g) : r = 52.5 := sorry

end initial_red_marbles_l544_54425


namespace first_number_is_38_l544_54402

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end first_number_is_38_l544_54402


namespace alexander_total_payment_l544_54442

variable (initialFee : ℝ) (dailyRent : ℝ) (costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ)

def totalCost (initialFee dailyRent costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  initialFee + (dailyRent * daysRented) + (costPerMile * milesDriven)

theorem alexander_total_payment :
  totalCost 15 30 0.25 3 350 = 192.5 :=
by
  unfold totalCost
  norm_num

end alexander_total_payment_l544_54442


namespace part_1_part_2_l544_54429

variable {a b : ℝ}

theorem part_1 (ha : a > 0) (hb : b > 0) : a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

theorem part_2 (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a * b^2 + a^2 * b :=
sorry

end part_1_part_2_l544_54429


namespace annual_income_of_a_l544_54487

-- Definitions based on the conditions
def monthly_income_ratio (a_income b_income : ℝ) : Prop := a_income / b_income = 5 / 2
def income_percentage (part whole : ℝ) : Prop := part / whole = 12 / 100
def c_monthly_income : ℝ := 15000
def b_monthly_income (c_income : ℝ) := c_income + 0.12 * c_income

-- The theorem to prove
theorem annual_income_of_a : ∀ (a_income b_income c_income : ℝ),
  monthly_income_ratio a_income b_income ∧
  b_income = b_monthly_income c_income ∧
  c_income = c_monthly_income →
  (a_income * 12) = 504000 :=
by
  -- Here we do not need to fill out the proof, so we use sorry
  sorry

end annual_income_of_a_l544_54487


namespace scissors_total_l544_54405

theorem scissors_total (initial_scissors : ℕ) (additional_scissors : ℕ) (h1 : initial_scissors = 54) (h2 : additional_scissors = 22) : 
  initial_scissors + additional_scissors = 76 :=
by
  sorry

end scissors_total_l544_54405


namespace optimal_rental_decision_optimal_purchase_decision_l544_54428

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end optimal_rental_decision_optimal_purchase_decision_l544_54428


namespace find_value_of_sum_of_squares_l544_54462

theorem find_value_of_sum_of_squares (x y : ℝ) (h : x^2 + y^2 + x^2 * y^2 - 4 * x * y + 1 = 0) :
  (x + y)^2 = 4 :=
sorry

end find_value_of_sum_of_squares_l544_54462


namespace total_cookies_dropped_throughout_entire_baking_process_l544_54458

def initially_baked_by_alice := 74 + 45 + 15
def initially_baked_by_bob := 7 + 32 + 18

def initially_dropped_by_alice := 5 + 8
def initially_dropped_by_bob := 10 + 6

def additional_baked_by_alice := 5 + 4 + 12
def additional_baked_by_bob := 22 + 36 + 14

def edible_cookies := 145

theorem total_cookies_dropped_throughout_entire_baking_process :
  initially_baked_by_alice + initially_baked_by_bob +
  additional_baked_by_alice + additional_baked_by_bob -
  edible_cookies = 139 := by
  sorry

end total_cookies_dropped_throughout_entire_baking_process_l544_54458


namespace johns_apartment_number_l544_54475

theorem johns_apartment_number (car_reg : Nat) (apartment_num : Nat) 
  (h_car_reg_sum : car_reg = 834205) 
  (h_car_digits : (8 + 3 + 4 + 2 + 0 + 5 = 22)) 
  (h_apartment_digits : ∃ (d1 d2 d3 : Nat), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 + d2 + d3 = 22) :
  apartment_num = 985 :=
by
  sorry

end johns_apartment_number_l544_54475


namespace find_pairs_l544_54495
open Nat

theorem find_pairs (x p : ℕ) (hp : p.Prime) (hxp : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (x = 1 ∧ p.Prime) ∨ (x = 2 ∧ p = 2) ∨ (x = 1 ∧ p.Prime) ∨ (x = 3 ∧ p = 3) := 
by
  sorry


end find_pairs_l544_54495


namespace edward_original_lawns_l544_54423

-- Definitions based on conditions
def dollars_per_lawn : ℕ := 4
def lawns_forgotten : ℕ := 9
def dollars_earned : ℕ := 32

-- The original number of lawns to mow
def original_lawns_to_mow (L : ℕ) : Prop :=
  dollars_per_lawn * (L - lawns_forgotten) = dollars_earned

-- The proof problem statement
theorem edward_original_lawns : ∃ L : ℕ, original_lawns_to_mow L ∧ L = 17 :=
by
  sorry

end edward_original_lawns_l544_54423


namespace greatest_integer_sum_of_integers_l544_54491

-- Definition of the quadratic function
def quadratic_expr (n : ℤ) : ℤ := n^2 - 15 * n + 56

-- The greatest integer n such that quadratic_expr n ≤ 0
theorem greatest_integer (n : ℤ) (h : quadratic_expr n ≤ 0) : n ≤ 8 := 
  sorry

-- All integers that satisfy the quadratic inequality
theorem sum_of_integers (sum_n : ℤ) (h : ∀ n : ℤ, 7 ≤ n ∧ n ≤ 8 → quadratic_expr n ≤ 0) 
  (sum_eq : sum_n = 7 + 8) : sum_n = 15 :=
  sorry

end greatest_integer_sum_of_integers_l544_54491


namespace general_term_formula_sum_of_b_first_terms_l544_54413

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end general_term_formula_sum_of_b_first_terms_l544_54413


namespace tim_same_age_tina_l544_54456

-- Define the ages of Tim and Tina
variables (x y : ℤ)

-- Given conditions
def condition_tim := x + 2 = 2 * (x - 2)
def condition_tina := y + 3 = 3 * (y - 3)

-- The goal is to prove that Tim is the same age as Tina
theorem tim_same_age_tina (htim : condition_tim x) (htina : condition_tina y) : x = y :=
by 
  sorry

end tim_same_age_tina_l544_54456


namespace base4_arithmetic_l544_54482

theorem base4_arithmetic : 
  ∀ (a b c : ℕ),
  a = 2 * 4^2 + 3 * 4^1 + 1 * 4^0 →
  b = 2 * 4^1 + 4 * 4^0 →
  c = 3 * 4^0 →
  (a * b) / c = 2 * 4^3 + 3 * 4^2 + 1 * 4^1 + 0 * 4^0 :=
by
  intros a b c ha hb hc
  sorry

end base4_arithmetic_l544_54482


namespace sound_pressure_level_l544_54440

theorem sound_pressure_level (p_0 p_1 p_2 p_3 : ℝ) (h_p0 : 0 < p_0)
  (L_p : ℝ → ℝ)
  (h_gasoline : 60 ≤ L_p p_1 ∧ L_p p_1 ≤ 90)
  (h_hybrid : 50 ≤ L_p p_2 ∧ L_p p_2 ≤ 60)
  (h_electric : L_p p_3 = 40)
  (h_L_p : ∀ p, L_p p = 20 * Real.log (p / p_0))
  : p_2 ≤ p_1 ∧ p_1 ≤ 100 * p_2 :=
by
  sorry

end sound_pressure_level_l544_54440


namespace solve_equation1_solve_equation2_l544_54407

-- Proof for equation (1)
theorem solve_equation1 : ∃ x : ℝ, 2 * (2 * x + 1) - (3 * x - 4) = 2 := by
  exists -4
  sorry

-- Proof for equation (2)
theorem solve_equation2 : ∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  exists -1
  sorry

end solve_equation1_solve_equation2_l544_54407


namespace no_arithmetic_progression_exists_l544_54477

theorem no_arithmetic_progression_exists 
  (a : ℕ) (d : ℕ) (a_n : ℕ → ℕ) 
  (h_seq : ∀ n, a_n n = a + n * d) :
  ¬ ∃ (a_n : ℕ → ℕ), (∀ n, a_n (n+1) > a_n n ∧ 
  ∀ n, (a_n n) * (a_n (n+1)) * (a_n (n+2)) * (a_n (n+3)) * (a_n (n+4)) * 
        (a_n (n+5)) * (a_n (n+6)) * (a_n (n+7)) * (a_n (n+8)) * (a_n (n+9)) % 
        ((a_n n) + (a_n (n+1)) + (a_n (n+2)) + (a_n (n+3)) + (a_n (n+4)) + 
         (a_n (n+5)) + (a_n (n+6)) + (a_n (n+7)) + (a_n (n+8)) + (a_n (n+9)) ) = 0 ) := 
sorry

end no_arithmetic_progression_exists_l544_54477


namespace stella_profit_l544_54438

-- Definitions based on the conditions
def number_of_dolls := 6
def price_per_doll := 8
def number_of_clocks := 4
def price_per_clock := 25
def number_of_glasses := 8
def price_per_glass := 6
def number_of_vases := 3
def price_per_vase := 12
def number_of_postcards := 10
def price_per_postcard := 3
def cost_of_merchandise := 250

-- Calculations based on given problem and solution
def revenue_from_dolls := number_of_dolls * price_per_doll
def revenue_from_clocks := number_of_clocks * price_per_clock
def revenue_from_glasses := number_of_glasses * price_per_glass
def revenue_from_vases := number_of_vases * price_per_vase
def revenue_from_postcards := number_of_postcards * price_per_postcard
def total_revenue := revenue_from_dolls + revenue_from_clocks + revenue_from_glasses + revenue_from_vases + revenue_from_postcards
def profit := total_revenue - cost_of_merchandise

-- Main theorem statement
theorem stella_profit : profit = 12 := by
  sorry

end stella_profit_l544_54438


namespace log_increasing_condition_log_increasing_not_necessary_l544_54409

theorem log_increasing_condition (a : ℝ) (h : a > 2) : a > 1 :=
by sorry

theorem log_increasing_not_necessary (a : ℝ) : ∃ b, (b > 1 ∧ ¬(b > 2)) :=
by sorry

end log_increasing_condition_log_increasing_not_necessary_l544_54409


namespace lara_bought_52_stems_l544_54432

-- Define the conditions given in the problem:
def flowers_given_to_mom : ℕ := 15
def flowers_given_to_grandma : ℕ := flowers_given_to_mom + 6
def flowers_in_vase : ℕ := 16

-- The total number of stems of flowers Lara bought should be:
def total_flowers_bought : ℕ := flowers_given_to_mom + flowers_given_to_grandma + flowers_in_vase

-- The main theorem to prove the total number of flowers Lara bought is 52:
theorem lara_bought_52_stems : total_flowers_bought = 52 := by
  sorry

end lara_bought_52_stems_l544_54432


namespace prove_weight_loss_l544_54467

variable (W : ℝ) -- Original weight
variable (x : ℝ) -- Percentage of weight lost

def weight_equation := W - (x / 100) * W + (2 / 100) * W = (89.76 / 100) * W

theorem prove_weight_loss (h : weight_equation W x) : x = 12.24 :=
by
  sorry

end prove_weight_loss_l544_54467


namespace find_sample_size_l544_54454

def sports_team (total: Nat) (soccer: Nat) (basketball: Nat) (table_tennis: Nat) : Prop :=
  total = soccer + basketball + table_tennis

def valid_sample_size (total: Nat) (n: Nat) :=
  (n > 0) ∧ (total % n == 0) ∧ (n % 6 == 0)

def systematic_sampling_interval (total: Nat) (n: Nat): Nat :=
  total / n

theorem find_sample_size :
  ∀ (total soccer basketball table_tennis: Nat),
  sports_team total soccer basketball table_tennis →
  total = 36 →
  soccer = 18 →
  basketball = 12 →
  table_tennis = 6 →
  (∃ n, valid_sample_size total n ∧ valid_sample_size (total - 1) (n + 1)) →
  ∃ n, n = 6 := by
  sorry

end find_sample_size_l544_54454


namespace range_of_a_l544_54430

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l544_54430
