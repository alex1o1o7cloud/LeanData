import Mathlib

namespace system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l889_88922

theorem system_of_equations_solution_non_negative (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x ≥ 0) (h4 : y ≥ 0) : x = 1 ∧ y = 0 :=
sorry

theorem system_of_equations_solution_positive_sum (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x + y > 0) : x = 1 ∧ y = 0 :=
sorry

end system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l889_88922


namespace MrMartinBought2Cups_l889_88925

theorem MrMartinBought2Cups (c b : ℝ) (x : ℝ) (h1 : 3 * c + 2 * b = 12.75)
                             (h2 : x * c + 5 * b = 14.00)
                             (hb : b = 1.5) :
  x = 2 :=
sorry

end MrMartinBought2Cups_l889_88925


namespace max_cards_with_digit_three_l889_88998

/-- There are ten cards each of the digits "3", "4", and "5". Choose any 8 cards such that their sum is 27. 
Prove that the maximum number of these cards that can be "3" is 6. -/
theorem max_cards_with_digit_three (c3 c4 c5 : ℕ) (hc3 : c3 + c4 + c5 = 8) (h_sum : 3 * c3 + 4 * c4 + 5 * c5 = 27) :
  c3 ≤ 6 :=
sorry

end max_cards_with_digit_three_l889_88998


namespace solve_equation_l889_88991

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end solve_equation_l889_88991


namespace tower_count_l889_88973

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binom (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def multinomialCoeff (n : Nat) (ks : List Nat) : Nat :=
  factorial n / List.foldr (fun k acc => acc * factorial k) 1 ks

theorem tower_count :
  let totalCubes := 9
  let usedCubes := 8
  let redCubes := 2
  let blueCubes := 3
  let greenCubes := 4
  multinomialCoeff totalCubes [redCubes, blueCubes, greenCubes] = 1260 :=
by
  sorry

end tower_count_l889_88973


namespace students_behind_yoongi_l889_88971

theorem students_behind_yoongi (total_students jungkoo_position students_between_jungkook_yoongi : ℕ) 
    (h1 : total_students = 20)
    (h2 : jungkoo_position = 3)
    (h3 : students_between_jungkook_yoongi = 5) : 
    (total_students - (jungkoo_position + students_between_jungkook_yoongi + 1)) = 11 :=
by
  sorry

end students_behind_yoongi_l889_88971


namespace solve_for_x_l889_88955

theorem solve_for_x (x : ℚ) (h : (1 / 2 - 1 / 3) = 3 / x) : x = 18 :=
sorry

end solve_for_x_l889_88955


namespace percentage_increase_area_l889_88932

theorem percentage_increase_area (L W : ℝ) :
  let A := L * W
  let L' := 1.20 * L
  let W' := 1.20 * W
  let A' := L' * W'
  let percentage_increase := (A' - A) / A * 100
  L > 0 → W > 0 → percentage_increase = 44 := 
by
  sorry

end percentage_increase_area_l889_88932


namespace proof_f_values_l889_88999

def f (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 7
  else
    x^2 - 2

theorem proof_f_values :
  f (-2) = 3 ∧ f (3) = 7 :=
by
  sorry

end proof_f_values_l889_88999


namespace complex_equality_l889_88977

theorem complex_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by sorry

end complex_equality_l889_88977


namespace sandy_jacket_price_l889_88972

noncomputable def discounted_shirt_price (initial_shirt_price discount_percentage : ℝ) : ℝ :=
  initial_shirt_price - (initial_shirt_price * discount_percentage / 100)

noncomputable def money_left (initial_money additional_money discounted_price : ℝ) : ℝ :=
  initial_money + additional_money - discounted_price

noncomputable def jacket_price_before_tax (remaining_money tax_percentage : ℝ) : ℝ :=
  remaining_money / (1 + tax_percentage / 100)

theorem sandy_jacket_price :
  let initial_money := 13.99
  let initial_shirt_price := 12.14
  let discount_percentage := 5.0
  let additional_money := 7.43
  let tax_percentage := 10.0
  
  let discounted_price := discounted_shirt_price initial_shirt_price discount_percentage
  let remaining_money := money_left initial_money additional_money discounted_price
  
  jacket_price_before_tax remaining_money tax_percentage = 8.99 := sorry

end sandy_jacket_price_l889_88972


namespace figure_100_squares_l889_88935

-- Define the initial conditions as given in the problem
def squares_in_figure (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 11
  | 2 => 25
  | 3 => 45
  | _ => sorry

-- Define the quadratic formula assumed from the problem conditions
def quadratic_formula (n : ℕ) : ℕ :=
  3 * n^2 + 5 * n + 3

-- Theorem: For figure 100, the number of squares is 30503
theorem figure_100_squares :
  squares_in_figure 100 = quadratic_formula 100 :=
by
  sorry

end figure_100_squares_l889_88935


namespace profit_per_meter_correct_l889_88900

noncomputable def total_selling_price := 6788
noncomputable def num_meters := 78
noncomputable def cost_price_per_meter := 58.02564102564102
noncomputable def total_cost_price := 4526 -- rounded total
noncomputable def total_profit := 2262 -- calculated total profit
noncomputable def profit_per_meter := 29

theorem profit_per_meter_correct :
  (total_selling_price - total_cost_price) / num_meters = profit_per_meter :=
by
  sorry

end profit_per_meter_correct_l889_88900


namespace min_T_tiles_needed_l889_88983

variable {a b c d : Nat}
variable (total_blocks : Nat := a + b + c + d)
variable (board_size : Nat := 8 * 10)
variable (block_size : Nat := 4)
variable (tile_types := ["T_horizontal", "T_vertical", "S_horizontal", "S_vertical"])
variable (conditions : Prop := total_blocks = 20 ∧ a + c ≥ 5)

theorem min_T_tiles_needed
    (h : conditions)
    (covering : total_blocks * block_size = board_size)
    (T_tiles : a ≥ 6) :
    a = 6 := sorry

end min_T_tiles_needed_l889_88983


namespace integer_satisfies_mod_and_range_l889_88996

theorem integer_satisfies_mod_and_range :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-150 ≡ n [ZMOD 25]) → n = 0 :=
by
  sorry

end integer_satisfies_mod_and_range_l889_88996


namespace lasagna_pieces_l889_88903

theorem lasagna_pieces (m a k r l : ℕ → ℝ)
  (hm : m 1 = 1)                -- Manny's consumption
  (ha : a 0 = 0)                -- Aaron's consumption
  (hk : ∀ n, k n = 2 * (m 1))   -- Kai's consumption
  (hr : ∀ n, r n = (1 / 2) * (m 1)) -- Raphael's consumption
  (hl : ∀ n, l n = 2 + (r n))   -- Lisa's consumption
  : m 1 + a 0 + k 1 + r 1 + l 1 = 6 :=
by
  -- Proof goes here
  sorry

end lasagna_pieces_l889_88903


namespace value_added_to_number_l889_88916

theorem value_added_to_number (n v : ℤ) (h1 : n = 9)
  (h2 : 3 * (n + 2) = v + n) : v = 24 :=
by
  sorry

end value_added_to_number_l889_88916


namespace number_of_students_l889_88950

theorem number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = 80 * N)
  (h2 : (T - 160) / (N - 8) = 90) :
  N = 56 :=
sorry

end number_of_students_l889_88950


namespace rhombus_side_length_l889_88930

-- Define the conditions including the diagonals and area of the rhombus
def diagonal_ratio (d1 d2 : ℝ) : Prop := d1 = 3 * d2
def area_rhombus (b : ℝ) (K : ℝ) : Prop := K = (1 / 2) * b * (3 * b)

-- Define the side length of the rhombus in terms of K
noncomputable def side_length (K : ℝ) : ℝ := Real.sqrt (5 * K / 3)

-- The main theorem statement
theorem rhombus_side_length (K : ℝ) (b : ℝ) (h1 : diagonal_ratio (3 * b) b) (h2 : area_rhombus b K) : 
  side_length K = Real.sqrt (5 * K / 3) := 
sorry

end rhombus_side_length_l889_88930


namespace chord_length_through_focus_l889_88943

theorem chord_length_through_focus (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1)
  (h_perp : (x = 1) ∨ (x = -1)) : abs (2 * y) = 3 :=
by {
  sorry
}

end chord_length_through_focus_l889_88943


namespace find_m_l889_88937

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)
noncomputable def dot_prod (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) (h : dot_prod a (b m) / magnitude a = 3) : m = sqrt 3 :=
by
  sorry

end find_m_l889_88937


namespace convert_cylindrical_to_rectangular_l889_88917

theorem convert_cylindrical_to_rectangular (r θ z x y : ℝ) (h_r : r = 5) (h_θ : θ = (3 * Real.pi) / 2) (h_z : z = 4)
    (h_x : x = r * Real.cos θ) (h_y : y = r * Real.sin θ) :
    (x, y, z) = (0, -5, 4) :=
by
    sorry

end convert_cylindrical_to_rectangular_l889_88917


namespace nes_sale_price_l889_88905

noncomputable def price_of_nes
    (snes_value : ℝ)
    (tradein_rate : ℝ)
    (cash_given : ℝ)
    (change_received : ℝ)
    (game_value : ℝ) : ℝ :=
  let tradein_credit := snes_value * tradein_rate
  let additional_cost := cash_given - change_received
  let total_cost := tradein_credit + additional_cost
  let nes_price := total_cost - game_value
  nes_price

theorem nes_sale_price 
  (snes_value : ℝ)
  (tradein_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (game_value : ℝ) :
  snes_value = 150 → tradein_rate = 0.80 → cash_given = 80 → change_received = 10 → game_value = 30 →
  price_of_nes snes_value tradein_rate cash_given change_received game_value = 160 := by
  intros
  sorry

end nes_sale_price_l889_88905


namespace months_to_survive_l889_88959

theorem months_to_survive (P_survive : ℝ) (initial_population : ℕ) (expected_survivors : ℝ) (n : ℕ)
  (h1 : P_survive = 5 / 6)
  (h2 : initial_population = 200)
  (h3 : expected_survivors = 115.74)
  (h4 : initial_population * (P_survive ^ n) = expected_survivors) :
  n = 3 :=
sorry

end months_to_survive_l889_88959


namespace johns_commute_distance_l889_88975

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end johns_commute_distance_l889_88975


namespace find_abscissas_l889_88982

theorem find_abscissas (x_A x_B : ℝ) (y_A y_B : ℝ) : 
  ((y_A = x_A^2) ∧ (y_B = x_B^2) ∧ (0, 15) = (0,  (5 * y_B + 3 * y_A) / 8) ∧ (5 * x_B + 3 * x_A = 0)) → 
  ((x_A = -5 ∧ x_B = 3) ∨ (x_A = 5 ∧ x_B = -3)) :=
by
  sorry

end find_abscissas_l889_88982


namespace four_corresponds_to_364_l889_88987

noncomputable def number_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 36
  | 3 => 363
  | 5 => 365
  | 36 => 2
  | _ => 0 -- Assume 0 as the default case

theorem four_corresponds_to_364 : number_pattern 4 = 364 :=
sorry

end four_corresponds_to_364_l889_88987


namespace quadratic_to_vertex_form_l889_88951

theorem quadratic_to_vertex_form :
  ∀ (x a h k : ℝ), (x^2 - 7*x = a*(x - h)^2 + k) → k = -49 / 4 :=
by
  intros x a h k
  sorry

end quadratic_to_vertex_form_l889_88951


namespace second_horse_revolutions_l889_88923

theorem second_horse_revolutions (r1 r2 d1: ℝ) (n1 n2: ℕ) 
  (h1: r1 = 30) (h2: d1 = 36) (h3: r2 = 10) 
  (h4: 2 * Real.pi * r1 * d1 = 2 * Real.pi * r2 * n2) : 
  n2 = 108 := 
by
   sorry

end second_horse_revolutions_l889_88923


namespace CodgerNeedsTenPairs_l889_88965

def CodgerHasThreeFeet : Prop := true

def ShoesSoldInPairs : Prop := true

def ShoesSoldInEvenNumberedPairs : Prop := true

def CodgerOwnsOneThreePieceSet : Prop := true

-- Main theorem stating Codger needs 10 pairs of shoes to have 7 complete 3-piece sets
theorem CodgerNeedsTenPairs (h1 : CodgerHasThreeFeet) (h2 : ShoesSoldInPairs)
  (h3 : ShoesSoldInEvenNumberedPairs) (h4 : CodgerOwnsOneThreePieceSet) : 
  ∃ pairsToBuy : ℕ, pairsToBuy = 10 := 
by {
  -- We have to prove codger needs 10 pairs of shoes to have 7 complete 3-piece sets
  sorry
}

end CodgerNeedsTenPairs_l889_88965


namespace Mark_jump_rope_hours_l889_88947

theorem Mark_jump_rope_hours 
  (record : ℕ) 
  (jump_rate : ℕ) 
  (seconds_per_hour : ℕ) 
  (h_record : record = 54000) 
  (h_jump_rate : jump_rate = 3) 
  (h_seconds_per_hour : seconds_per_hour = 3600) 
  : (record / jump_rate) / seconds_per_hour = 5 := 
by
  sorry

end Mark_jump_rope_hours_l889_88947


namespace range_of_4x_2y_l889_88934

theorem range_of_4x_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) :
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 := 
sorry

end range_of_4x_2y_l889_88934


namespace length_of_AB_l889_88946

theorem length_of_AB (V : ℝ) (r : ℝ) :
  V = 216 * Real.pi →
  r = 3 →
  ∃ (len_AB : ℝ), len_AB = 20 :=
by
  intros hV hr
  have volume_cylinder := V - 36 * Real.pi
  have height_cylinder := volume_cylinder / (Real.pi * r^2)
  exists height_cylinder
  exact sorry

end length_of_AB_l889_88946


namespace remaining_pencils_total_l889_88985

-- Definitions corresponding to the conditions:
def J : ℝ := 300
def J_d : ℝ := 0.30 * J
def J_r : ℝ := J - J_d

def V : ℝ := 2 * J
def V_d : ℝ := 125
def V_r : ℝ := V - V_d

def S : ℝ := 450
def S_d : ℝ := 0.60 * S
def S_r : ℝ := S - S_d

-- Proving the remaining pencils add up to the required amount:
theorem remaining_pencils_total : J_r + V_r + S_r = 865 := by
  sorry

end remaining_pencils_total_l889_88985


namespace optionD_is_quadratic_l889_88969

variable (x : ℝ)

-- Original equation in Option D
def optionDOriginal := (x^2 + 2 * x = 2 * x^2 - 1)

-- Rearranged form of Option D's equation
def optionDRearranged := (-x^2 + 2 * x + 1 = 0)

theorem optionD_is_quadratic : optionDOriginal x → optionDRearranged x :=
by
  intro h
  -- The proof steps would go here, but we use sorry to skip it
  sorry

end optionD_is_quadratic_l889_88969


namespace find_B_l889_88981

theorem find_B (A B : Nat) (hA : A ≤ 9) (hB : B ≤ 9) (h_eq : 6 * A + 10 * B + 2 = 77) : B = 1 :=
by
-- proof steps would go here
sorry

end find_B_l889_88981


namespace problem_given_conditions_l889_88924

theorem problem_given_conditions (x y z : ℝ) 
  (h : x / 3 = y / (-4) ∧ y / (-4) = z / 7) : (3 * x + y + z) / y = -3 := 
by 
  sorry

end problem_given_conditions_l889_88924


namespace third_dimension_of_box_l889_88945

theorem third_dimension_of_box (h : ℕ) (H : (151^2 - 150^2) * h + 151^2 = 90000) : h = 223 :=
sorry

end third_dimension_of_box_l889_88945


namespace sticks_in_100th_stage_l889_88940

theorem sticks_in_100th_stage : 
  ∀ (n a₁ d : ℕ), a₁ = 5 → d = 4 → n = 100 → a₁ + (n - 1) * d = 401 :=
by
  sorry

end sticks_in_100th_stage_l889_88940


namespace range_of_f_x_minus_2_l889_88910

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + 1 else if x > 0 then -(x + 1) else 0

theorem range_of_f_x_minus_2 :
  ∀ x : ℝ, f (x - 2) < 0 ↔ x ∈ Set.union (Set.Iio 1) (Set.Ioo 2 3) := by
sorry

end range_of_f_x_minus_2_l889_88910


namespace hexagon_internal_angle_A_l889_88997

theorem hexagon_internal_angle_A
  (B C D E F : ℝ) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) 
  (H : B + C + D + E + F + A = 720) : A = 120 := 
sorry

end hexagon_internal_angle_A_l889_88997


namespace greatest_even_integer_leq_z_l889_88915

theorem greatest_even_integer_leq_z (z : ℝ) (z_star : ℝ → ℝ)
  (h1 : ∀ z, z_star z = z_star (z - (z - z_star z))) -- (This is to match the definition given)
  (h2 : 6.30 - z_star 6.30 = 0.2999999999999998) : z_star 6.30 ≤ 6.30 := by
sorry

end greatest_even_integer_leq_z_l889_88915


namespace fraction_subtraction_simplified_l889_88914

theorem fraction_subtraction_simplified :
  (8 / 21 - 3 / 63) = 1 / 3 := 
by
  sorry

end fraction_subtraction_simplified_l889_88914


namespace find_d_l889_88968

noncomputable def polynomial_d (a b c d : ℤ) (p q r s : ℤ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
  1 + a + b + c + d = 2024 ∧
  (1 + p) * (1 + q) * (1 + r) * (1 + s) = 2024 ∧
  d = p * q * r * s

theorem find_d (a b c d : ℤ) (h : polynomial_d a b c d 7 10 22 11) : d = 17020 :=
  sorry

end find_d_l889_88968


namespace nancy_crayons_l889_88904

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end nancy_crayons_l889_88904


namespace problem_conditions_l889_88939

theorem problem_conditions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 3) :
  (x * y ≤ 9 / 8) ∧ (4 ^ x + 2 ^ y ≥ 4 * Real.sqrt 2) ∧ (x / y + 1 / x ≥ 2 / 3 + 2 * Real.sqrt 3 / 3) :=
by
  -- Proof goes here
  sorry

end problem_conditions_l889_88939


namespace license_plate_possibilities_count_l889_88966

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

theorem license_plate_possibilities_count : 
  (vowels.card * digits.card * 2 = 100) := 
by {
  -- vowels.card = 5 because there are 5 vowels.
  -- digits.card = 10 because there are 10 digits.
  -- 2 because the middle character must match either the first vowel or the last digit.
  sorry
}

end license_plate_possibilities_count_l889_88966


namespace relationship_between_y1_y2_l889_88919

theorem relationship_between_y1_y2 (y1 y2 : ℝ)
  (h1 : y1 = -2 * (-2) + 3)
  (h2 : y2 = -2 * 3 + 3) :
  y1 > y2 := by
  sorry

end relationship_between_y1_y2_l889_88919


namespace figures_can_be_drawn_l889_88936

structure Figure :=
  (degrees : List ℕ) -- List of degrees of the vertices in the graph associated with the figure.

-- Define a predicate to check if a figure can be drawn without lifting the pencil and without retracing
def canBeDrawnWithoutLifting (fig : Figure) : Prop :=
  let odd_degree_vertices := fig.degrees.filter (λ d => d % 2 = 1)
  odd_degree_vertices.length = 0 ∨ odd_degree_vertices.length = 2

-- Define the figures A, B, C, D with their degrees (examples, these should match the problem's context)
def figureA : Figure := { degrees := [2, 2, 2, 2] }
def figureB : Figure := { degrees := [2, 2, 2, 2, 4] }
def figureC : Figure := { degrees := [3, 3, 3, 3] }
def figureD : Figure := { degrees := [4, 4, 2, 2] }

-- State the theorem that figures A, B, and D can be drawn without lifting the pencil
theorem figures_can_be_drawn :
  canBeDrawnWithoutLifting figureA ∧ canBeDrawnWithoutLifting figureB ∧ canBeDrawnWithoutLifting figureD :=
  by sorry -- Proof to be completed

end figures_can_be_drawn_l889_88936


namespace find_k_l889_88994

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k
  (a b : vector)
  (h_a : a = (2, -1))
  (h_b : b = (-1, 4))
  (h_perpendicular : dot_product (a.1 - k * b.1, a.2 + 4 * k) (3, -5) = 0) :
  k = -11/17 := sorry

end find_k_l889_88994


namespace hidden_prime_average_correct_l889_88913

noncomputable def hidden_prime_average : ℚ :=
  (13 + 17 + 59) / 3

theorem hidden_prime_average_correct :
  hidden_prime_average = 29.6 :=
by
  sorry

end hidden_prime_average_correct_l889_88913


namespace sum_of_digits_of_10_pow_30_minus_36_l889_88929

def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem sum_of_digits_of_10_pow_30_minus_36 : 
  sum_of_digits (10^30 - 36) = 11 := 
by 
  -- proof goes here
  sorry

end sum_of_digits_of_10_pow_30_minus_36_l889_88929


namespace prob_selected_first_eq_third_l889_88967

noncomputable def total_students_first := 800
noncomputable def total_students_second := 600
noncomputable def total_students_third := 500
noncomputable def selected_students_third := 25
noncomputable def prob_selected_third := selected_students_third / total_students_third

theorem prob_selected_first_eq_third :
  (selected_students_third / total_students_third = 1 / 20) →
  (prob_selected_third = 1 / 20) :=
by
  intros h
  sorry

end prob_selected_first_eq_third_l889_88967


namespace rotameter_percentage_l889_88949

theorem rotameter_percentage (l_inch_flow : ℝ) (l_liters_flow : ℝ) (g_inch_flow : ℝ) (g_liters_flow : ℝ) :
  l_inch_flow = 2.5 → l_liters_flow = 60 → g_inch_flow = 4 → g_liters_flow = 192 → 
  (g_liters_flow / g_inch_flow) / (l_liters_flow / l_inch_flow) * 100 = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end rotameter_percentage_l889_88949


namespace cloak_change_in_silver_l889_88957

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l889_88957


namespace count_valid_three_digit_numbers_l889_88942

theorem count_valid_three_digit_numbers : 
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  valid_numbers = 90 :=
by
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  have h1 : valid_numbers = 900 - 810 := by rfl
  have h2 : 900 - 810 = 90 := by norm_num
  exact h1.trans h2

end count_valid_three_digit_numbers_l889_88942


namespace zero_point_interval_l889_88909

variable (f : ℝ → ℝ)
variable (f_deriv : ℝ → ℝ)
variable (e : ℝ)
variable (monotonic_f : MonotoneOn f (Set.Ioi 0))

noncomputable def condition1_property (x : ℝ) (h : 0 < x) : f (f x - Real.log x) = Real.exp 1 + 1 := sorry
noncomputable def derivative_property (x : ℝ) (h : 0 < x) : f_deriv x = (deriv f) x := sorry

theorem zero_point_interval :
  ∃ x ∈ Set.Ioo 1 2, f x - f_deriv x - e = 0 := sorry

end zero_point_interval_l889_88909


namespace range_of_fx1_l889_88926

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

theorem range_of_fx1 (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : f x1 a = 0) (h4 : f x2 a = 0) :
    f x1 a > (1 - 2 * Real.log 2) / 4 :=
sorry

end range_of_fx1_l889_88926


namespace fourth_boy_payment_l889_88961

theorem fourth_boy_payment (a b c d : ℝ) 
  (h₁ : a = (1 / 2) * (b + c + d)) 
  (h₂ : b = (1 / 3) * (a + c + d)) 
  (h₃ : c = (1 / 4) * (a + b + d)) 
  (h₄ : a + b + c + d = 60) : 
  d = 13 := 
sorry

end fourth_boy_payment_l889_88961


namespace angle_sum_triangle_l889_88993

theorem angle_sum_triangle (x : ℝ) 
  (h1 : 70 + 70 + x = 180) : 
  x = 40 :=
by
  sorry

end angle_sum_triangle_l889_88993


namespace octopus_leg_count_l889_88948

theorem octopus_leg_count :
  let num_initial_octopuses := 5
  let legs_per_normal_octopus := 8
  let num_removed_octopuses := 2
  let legs_first_mutant := 10
  let legs_second_mutant := 6
  let legs_third_mutant := 2 * legs_per_normal_octopus
  let num_initial_legs := num_initial_octopuses * legs_per_normal_octopus
  let num_removed_legs := num_removed_octopuses * legs_per_normal_octopus
  let num_mutant_legs := legs_first_mutant + legs_second_mutant + legs_third_mutant
  num_initial_legs - num_removed_legs + num_mutant_legs = 56 :=
by
  -- proof to be filled in later
  sorry

end octopus_leg_count_l889_88948


namespace arithmetic_mean_pq_l889_88902

variable (p q r : ℝ)

-- Definitions from conditions
def condition1 := (p + q) / 2 = 10
def condition2 := (q + r) / 2 = 26
def condition3 := r - p = 32

-- Theorem statement
theorem arithmetic_mean_pq : condition1 p q → condition2 q r → condition3 p r → (p + q) / 2 = 10 :=
by
  intros h1 h2 h3
  exact h1

end arithmetic_mean_pq_l889_88902


namespace evaluate_expression_right_to_left_l889_88989

variable (a b c d : ℝ)

theorem evaluate_expression_right_to_left:
  (a * b + c - d) = (a * (b + c - d)) :=
by {
  -- Group operations from right to left according to the given condition
  sorry
}

end evaluate_expression_right_to_left_l889_88989


namespace triangle_inequality_l889_88978

theorem triangle_inequality
  (α β γ a b c : ℝ)
  (h_angles_sum : α + β + γ = Real.pi)
  (h_pos_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 2 * (a / α + b / β + c / γ) := by
  sorry

end triangle_inequality_l889_88978


namespace point_in_fourth_quadrant_l889_88963

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end point_in_fourth_quadrant_l889_88963


namespace common_number_in_lists_l889_88908

theorem common_number_in_lists (nums : List ℚ) (h_len : nums.length = 9)
  (h_first_five_avg : (nums.take 5).sum / 5 = 7)
  (h_last_five_avg : (nums.drop 4).sum / 5 = 9)
  (h_total_avg : nums.sum / 9 = 73/9) :
  ∃ x, x ∈ nums.take 5 ∧ x ∈ nums.drop 4 ∧ x = 7 := 
sorry

end common_number_in_lists_l889_88908


namespace find_constant_t_l889_88995

theorem find_constant_t (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t + 5^n) ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ (a 1 = S 1) ∧ 
  (∃ q, ∀ n ≥ 1, a (n + 1) = q * a n) → 
  t = -1 := by
  sorry

end find_constant_t_l889_88995


namespace initial_women_count_l889_88931

-- Let x be the initial number of women.
-- Let y be the initial number of men.

theorem initial_women_count (x y : ℕ) (h1 : y = 2 * (x - 15)) (h2 : (y - 45) * 5 = (x - 15)) :
  x = 40 :=
by
  -- sorry to skip the proof
  sorry

end initial_women_count_l889_88931


namespace max_value_function_max_value_expression_l889_88938

theorem max_value_function (x a : ℝ) (hx : x > 0) (ha : a > 2 * x) : ∃ y : ℝ, y = (a^2) / 8 :=
by
  sorry

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 4) : 
   ∃ m : ℝ, m = 4 :=
by
  sorry

end max_value_function_max_value_expression_l889_88938


namespace number_of_boys_l889_88970

theorem number_of_boys
  (x y : ℕ) 
  (h1 : x + y = 43)
  (h2 : 24 * x + 27 * y = 1101) : 
  x = 20 := by
  sorry

end number_of_boys_l889_88970


namespace chocolates_exceeding_200_l889_88906

-- Define the initial amount of chocolates
def initial_chocolates : ℕ := 3

-- Define the function that computes the amount of chocolates on the nth day
def chocolates_on_day (n : ℕ) : ℕ := initial_chocolates * 3 ^ (n - 1)

-- Define the proof problem
theorem chocolates_exceeding_200 : ∃ (n : ℕ), chocolates_on_day n > 200 :=
by
  -- Proof required here
  sorry

end chocolates_exceeding_200_l889_88906


namespace prime_factor_difference_duodecimal_l889_88953

theorem prime_factor_difference_duodecimal (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 11) (hB : 0 ≤ B ∧ B ≤ 11) (h : A ≠ B) : 
  ∃ k : ℤ, (12 * A + B - (12 * B + A)) = 11 * k := 
by sorry

end prime_factor_difference_duodecimal_l889_88953


namespace quadratic_roots_property_l889_88986

theorem quadratic_roots_property (a b : ℝ)
  (h1 : a^2 - 2 * a - 1 = 0)
  (h2 : b^2 - 2 * b - 1 = 0)
  (ha_b_sum : a + b = 2)
  (ha_b_product : a * b = -1) :
  a^2 + 2 * b - a * b = 6 :=
sorry

end quadratic_roots_property_l889_88986


namespace solution_is_correct_l889_88954

-- Initial conditions
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.40
def target_concentration : ℝ := 0.50

-- Given that we start with 2.4 liters of pure alcohol in a 6-liter solution
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Expected result after adding x liters of pure alcohol
def final_solution_volume (x : ℝ) : ℝ := initial_volume + x
def final_pure_alcohol (x : ℝ) : ℝ := initial_pure_alcohol + x

-- Equation to prove
theorem solution_is_correct (x : ℝ) :
  (final_pure_alcohol x) / (final_solution_volume x) = target_concentration ↔ 
  x = 1.2 := 
sorry

end solution_is_correct_l889_88954


namespace total_time_is_three_hours_l889_88952

-- Define the conditions of the problem in Lean
def time_uber_house := 10
def time_uber_airport := 5 * time_uber_house
def time_check_bag := 15
def time_security := 3 * time_check_bag
def time_boarding := 20
def time_takeoff := 2 * time_boarding

-- Total time in minutes
def total_time_minutes := time_uber_house + time_uber_airport + time_check_bag + time_security + time_boarding + time_takeoff

-- Conversion from minutes to hours
def total_time_hours := total_time_minutes / 60

-- The theorem to prove
theorem total_time_is_three_hours : total_time_hours = 3 := by
  sorry

end total_time_is_three_hours_l889_88952


namespace train_length_is_135_l889_88941

noncomputable def length_of_train (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_is_135 :
  length_of_train 54 9 = 135 := 
by
  -- Conditions: 
  -- speed_kmh = 54
  -- time_sec = 9
  sorry

end train_length_is_135_l889_88941


namespace area_of_triangle_ABC_l889_88979

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (2, 2)
def C : Point := (2, 0)

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  triangle_area A B C = 2 :=
by
  sorry

end area_of_triangle_ABC_l889_88979


namespace greatest_value_of_x_l889_88907

theorem greatest_value_of_x (x : ℝ) : 
  (∃ (M : ℝ), (∀ y : ℝ, (y ^ 2 - 14 * y + 45 <= 0) → y <= M) ∧ (M ^ 2 - 14 * M + 45 <= 0)) ↔ M = 9 :=
by
  sorry

end greatest_value_of_x_l889_88907


namespace probability_three_heads_l889_88988

noncomputable def fair_coin_flip: ℝ := 1 / 2

theorem probability_three_heads :
  (fair_coin_flip * fair_coin_flip * fair_coin_flip) = 1 / 8 :=
by
  -- proof would go here
  sorry

end probability_three_heads_l889_88988


namespace vector_transitivity_l889_88901

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem vector_transitivity (h1 : a = b) (h2 : b = c) : a = c :=
by {
  sorry
}

end vector_transitivity_l889_88901


namespace cannot_reach_target_l889_88912

def initial_price : ℕ := 1
def annual_increment : ℕ := 1
def tripling_year (n : ℕ) : ℕ := 3 * n
def total_years : ℕ := 99
def target_price : ℕ := 152
def incremental_years : ℕ := 98

noncomputable def final_price (x : ℕ) : ℕ := 
  initial_price + incremental_years * annual_increment + tripling_year x - annual_increment

theorem cannot_reach_target (p : ℕ) (h : p = final_price p) : p ≠ target_price :=
sorry

end cannot_reach_target_l889_88912


namespace grassy_plot_width_l889_88958

/-- A rectangular grassy plot has a length of 100 m and a certain width. 
It has a gravel path 2.5 m wide all round it on the inside. The cost of gravelling 
the path at 0.90 rupees per square meter is 742.5 rupees. 
Prove that the width of the grassy plot is 60 meters. -/
theorem grassy_plot_width 
  (length : ℝ)
  (path_width : ℝ)
  (cost_per_sq_meter : ℝ)
  (total_cost : ℝ)
  (width : ℝ) : 
  length = 100 ∧ 
  path_width = 2.5 ∧ 
  cost_per_sq_meter = 0.9 ∧ 
  total_cost = 742.5 → 
  width = 60 := 
by sorry

end grassy_plot_width_l889_88958


namespace payment_correct_l889_88960

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end payment_correct_l889_88960


namespace not_beautiful_739_and_741_l889_88980

-- Define the function g and its properties
variable (g : ℤ → ℤ)

-- Condition: g(x) ≠ x
axiom g_neq_x (x : ℤ) : g x ≠ x

-- Definition of "beautiful"
def beautiful (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

-- The theorem to prove
theorem not_beautiful_739_and_741 :
  ¬ (beautiful g 739 ∧ beautiful g 741) :=
sorry

end not_beautiful_739_and_741_l889_88980


namespace number_of_divisors_of_2018_or_2019_is_7_l889_88918

theorem number_of_divisors_of_2018_or_2019_is_7 (h1 : Prime 673) (h2 : Prime 1009) : 
  Nat.card {d : Nat | d ∣ 2018 ∨ d ∣ 2019} = 7 := 
  sorry

end number_of_divisors_of_2018_or_2019_is_7_l889_88918


namespace Jolene_charge_per_car_l889_88928

theorem Jolene_charge_per_car (babysitting_families cars_washed : ℕ) (charge_per_family total_raised babysitting_earnings car_charge : ℕ) :
  babysitting_families = 4 →
  charge_per_family = 30 →
  cars_washed = 5 →
  total_raised = 180 →
  babysitting_earnings = babysitting_families * charge_per_family →
  car_charge = (total_raised - babysitting_earnings) / cars_washed →
  car_charge = 12 :=
by
  intros
  sorry

end Jolene_charge_per_car_l889_88928


namespace place_synthetic_method_l889_88920

theorem place_synthetic_method :
  "Synthetic Method" = "Direct Proof" :=
sorry

end place_synthetic_method_l889_88920


namespace Margie_distance_on_25_dollars_l889_88962

theorem Margie_distance_on_25_dollars
  (miles_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (amount_spent : ℝ) :
  miles_per_gallon = 40 →
  cost_per_gallon = 5 →
  amount_spent = 25 →
  (amount_spent / cost_per_gallon) * miles_per_gallon = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Margie_distance_on_25_dollars_l889_88962


namespace calculation_correct_l889_88990

theorem calculation_correct (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b :=
by sorry

end calculation_correct_l889_88990


namespace contest_score_difference_l889_88974

theorem contest_score_difference :
  let percent_50 := 0.05
  let percent_60 := 0.20
  let percent_70 := 0.25
  let percent_80 := 0.30
  let percent_90 := 1 - (percent_50 + percent_60 + percent_70 + percent_80)
  let mean := (percent_50 * 50) + (percent_60 * 60) + (percent_70 * 70) + (percent_80 * 80) + (percent_90 * 90)
  let median := 70
  median - mean = -4 :=
by
  sorry

end contest_score_difference_l889_88974


namespace quadratic_root_property_l889_88964

theorem quadratic_root_property (m n : ℝ)
  (hmn : m^2 + m - 2021 = 0)
  (hn : n^2 + n - 2021 = 0) :
  m^2 + 2 * m + n = 2020 :=
by sorry

end quadratic_root_property_l889_88964


namespace coordinates_of_point_l889_88956

theorem coordinates_of_point (x y : ℝ) (hx : x < 0) (hy : y > 0) (dx : |x| = 3) (dy : |y| = 2) :
  (x, y) = (-3, 2) := 
sorry

end coordinates_of_point_l889_88956


namespace prove_a_pow_minus_b_l889_88911

-- Definitions of conditions
variables (x a b : ℝ)

def condition_1 : Prop := x - a > 2
def condition_2 : Prop := 2 * x - b < 0
def solution_set_condition : Prop := -1 < x ∧ x < 1
def derived_a : Prop := a + 2 = -1
def derived_b : Prop := b / 2 = 1

-- The main theorem to prove
theorem prove_a_pow_minus_b (h1 : condition_1 x a) (h2 : condition_2 x b) (h3 : solution_set_condition x) (ha : derived_a a) (hb : derived_b b) : a^(-b) = (1 / 9) :=
by
  sorry

end prove_a_pow_minus_b_l889_88911


namespace find_dividend_l889_88976

theorem find_dividend :
  ∀ (Divisor Quotient Remainder : ℕ), Divisor = 15 → Quotient = 9 → Remainder = 5 → (Divisor * Quotient + Remainder) = 140 :=
by
  intros Divisor Quotient Remainder hDiv hQuot hRem
  subst hDiv
  subst hQuot
  subst hRem
  sorry

end find_dividend_l889_88976


namespace set_inter_complement_eq_l889_88921

-- Given conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 < 1}
def B : Set ℝ := {x | x^2 - 2 * x > 0}

-- Question translated to proof problem statement
theorem set_inter_complement_eq :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_inter_complement_eq_l889_88921


namespace ray_nickels_left_l889_88927

theorem ray_nickels_left (h1 : 285 % 5 = 0) (h2 : 55 % 5 = 0) (h3 : 3 * 55 % 5 = 0) (h4 : 45 % 5 = 0) : 
  285 / 5 - ((55 / 5) + (3 * 55 / 5) + (45 / 5)) = 4 := sorry

end ray_nickels_left_l889_88927


namespace rice_in_each_container_l889_88984

variable (weight_in_pounds : ℚ := 35 / 2)
variable (num_containers : ℕ := 4)
variable (pound_to_oz : ℕ := 16)

theorem rice_in_each_container :
  (weight_in_pounds * pound_to_oz) / num_containers = 70 :=
by
  sorry

end rice_in_each_container_l889_88984


namespace david_pushups_difference_l889_88944

-- Definitions based on conditions
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- The number of push-ups David did more than Zachary
def david_more_pushups_than_zachary (D : ℕ) := D - zachary_pushups

-- The theorem we need to prove
theorem david_pushups_difference :
  ∃ D : ℕ, D > zachary_pushups ∧ D + zachary_pushups = total_pushups ∧ david_more_pushups_than_zachary D = 58 :=
by
  -- We leave the proof as an exercise or for further filling.
  sorry

end david_pushups_difference_l889_88944


namespace pyramid_total_blocks_l889_88933

-- Define the number of layers in the pyramid
def num_layers : ℕ := 8

-- Define the block multiplier for each subsequent layer
def block_multiplier : ℕ := 5

-- Define the number of blocks in the top layer
def top_layer_blocks : ℕ := 3

-- Define the total number of sandstone blocks
def total_blocks_pyramid : ℕ :=
  let rec total_blocks (layer : ℕ) (blocks : ℕ) :=
    if layer = 0 then blocks
    else blocks + total_blocks (layer - 1) (blocks * block_multiplier)
  total_blocks (num_layers - 1) top_layer_blocks

theorem pyramid_total_blocks :
  total_blocks_pyramid = 312093 :=
by
  -- Proof omitted
  sorry

end pyramid_total_blocks_l889_88933


namespace charlies_age_22_l889_88992

variable (A : ℕ) (C : ℕ)

theorem charlies_age_22 (h1 : C = 2 * A + 8) (h2 : C = 22) : A = 7 := by
  sorry

end charlies_age_22_l889_88992
