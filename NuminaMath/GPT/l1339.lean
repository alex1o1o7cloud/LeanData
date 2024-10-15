import Mathlib

namespace NUMINAMATH_GPT_area_of_triangle_l1339_133908

theorem area_of_triangle (m : ℝ) 
  (h : ∀ x y : ℝ, ((m + 3) * x + y = 3 * m - 4) → 
                  (7 * x + (5 - m) * y - 8 ≠ 0)
  ) : ((m = -2) → (1/2) * 2 * 2 = 2) := 
by {
  sorry
}

end NUMINAMATH_GPT_area_of_triangle_l1339_133908


namespace NUMINAMATH_GPT_LanceCents_l1339_133976

noncomputable def MargaretCents : ℕ := 75
noncomputable def GuyCents : ℕ := 60
noncomputable def BillCents : ℕ := 60
noncomputable def TotalCents : ℕ := 265

theorem LanceCents (lanceCents : ℕ) :
  MargaretCents + GuyCents + BillCents + lanceCents = TotalCents → lanceCents = 70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_LanceCents_l1339_133976


namespace NUMINAMATH_GPT_prove_triangular_cake_volume_surface_area_sum_l1339_133921

def triangular_cake_volume_surface_area_sum_proof : Prop :=
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 2
  let base_area : ℝ := (1 / 2) * length * width
  let volume : ℝ := base_area * height
  let top_area : ℝ := base_area
  let side_area : ℝ := (1 / 2) * width * height
  let icing_area : ℝ := top_area + 3 * side_area
  volume + icing_area = 15

theorem prove_triangular_cake_volume_surface_area_sum : triangular_cake_volume_surface_area_sum_proof := by
  sorry

end NUMINAMATH_GPT_prove_triangular_cake_volume_surface_area_sum_l1339_133921


namespace NUMINAMATH_GPT_nearest_integer_to_expression_correct_l1339_133949

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end NUMINAMATH_GPT_nearest_integer_to_expression_correct_l1339_133949


namespace NUMINAMATH_GPT_total_games_played_l1339_133940

-- Define the number of teams
def num_teams : ℕ := 12

-- Define the number of games each team plays with each other team
def games_per_pair : ℕ := 4

-- The theorem stating the total number of games played
theorem total_games_played : num_teams * (num_teams - 1) / 2 * games_per_pair = 264 :=
by
  sorry

end NUMINAMATH_GPT_total_games_played_l1339_133940


namespace NUMINAMATH_GPT_initial_oranges_l1339_133909

theorem initial_oranges (O : ℕ) (h1 : (1 / 4 : ℚ) * (1 / 2 : ℚ) * O = 39) (h2 : (1 / 8 : ℚ) * (1 / 2 : ℚ) * O = 4 + 78 - (1 / 4 : ℚ) * (1 / 2 : ℚ) * O) :
  O = 96 :=
by
  sorry

end NUMINAMATH_GPT_initial_oranges_l1339_133909


namespace NUMINAMATH_GPT_fourth_term_geometric_progression_l1339_133942

theorem fourth_term_geometric_progression
  (x : ℝ)
  (h : ∀ n : ℕ, n ≥ 0 → (3 * x * (n : ℝ) + 3 * (n : ℝ)) = (6 * x * ((n - 1) : ℝ) + 6 * ((n - 1) : ℝ))) :
  (((3*x + 3)^2 = (6*x + 6) * x) ∧ x = -3) → (∀ n : ℕ, n = 4 → (2^(n-3) * (6*x + 6)) = -24) :=
by
  sorry

end NUMINAMATH_GPT_fourth_term_geometric_progression_l1339_133942


namespace NUMINAMATH_GPT_find_m_l1339_133991

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end NUMINAMATH_GPT_find_m_l1339_133991


namespace NUMINAMATH_GPT_quadratic_real_roots_l1339_133962

variable (a b : ℝ)

theorem quadratic_real_roots (h : ∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) : b ≤ -1/8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1339_133962


namespace NUMINAMATH_GPT_height_difference_l1339_133955

-- Definitions of the terms and conditions
variables {b h : ℝ} -- base and height of Triangle B
variables {b' h' : ℝ} -- base and height of Triangle A

-- Given conditions:
-- Triangle A's base is 10% greater than Triangle B's base
def base_relation (b' : ℝ) (b : ℝ) := b' = 1.10 * b

-- The area of Triangle A is 1% less than the area of Triangle B
def area_relation (b h b' h' : ℝ) := (1 / 2) * b' * h' = (1 / 2) * b * h - 0.01 * (1 / 2) * b * h

-- Proof statement
theorem height_difference (b h b' h' : ℝ) (H_base: base_relation b' b) (H_area: area_relation b h b' h') :
  h' = 0.9 * h := 
sorry

end NUMINAMATH_GPT_height_difference_l1339_133955


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1339_133971

theorem range_of_a_for_inequality : 
  ∃ a : ℝ, (∀ x : ℤ, (a * x - 1) ^ 2 < x ^ 2) ↔ 
    (a > -3 / 2 ∧ a ≤ -4 / 3) ∨ (4 / 3 ≤ a ∧ a < 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l1339_133971


namespace NUMINAMATH_GPT_point_Q_in_third_quadrant_l1339_133960

theorem point_Q_in_third_quadrant (m : ℝ) :
  (2 * m + 4 = 0 → (m - 3, m).fst < 0 ∧ (m - 3, m).snd < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_Q_in_third_quadrant_l1339_133960


namespace NUMINAMATH_GPT_discount_is_10_percent_l1339_133990

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end NUMINAMATH_GPT_discount_is_10_percent_l1339_133990


namespace NUMINAMATH_GPT_Gake_needs_fewer_boards_than_Tom_l1339_133925

noncomputable def Tom_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (3 * width_char + 2 * 6) / width_board

noncomputable def Gake_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (4 * width_char + 3 * 1) / width_board

theorem Gake_needs_fewer_boards_than_Tom :
  Gake_boards_needed < Tom_boards_needed :=
by
  -- Here you will put the actual proof steps
  sorry

end NUMINAMATH_GPT_Gake_needs_fewer_boards_than_Tom_l1339_133925


namespace NUMINAMATH_GPT_symmetrical_character_l1339_133906

def is_symmetrical (char : String) : Prop := 
  sorry  -- Here the definition for symmetry will be elaborated

theorem symmetrical_character : 
  let A : String := "坡"
  let B : String := "上"
  let C : String := "草"
  let D : String := "原"
  is_symmetrical C := 
  sorry

end NUMINAMATH_GPT_symmetrical_character_l1339_133906


namespace NUMINAMATH_GPT_x_gt_neg2_is_necessary_for_prod_lt_0_l1339_133901

theorem x_gt_neg2_is_necessary_for_prod_lt_0 (x : Real) :
  (x > -2) ↔ (((x + 2) * (x - 3)) < 0) → (x > -2) :=
by
  sorry

end NUMINAMATH_GPT_x_gt_neg2_is_necessary_for_prod_lt_0_l1339_133901


namespace NUMINAMATH_GPT_find_x_value_l1339_133943

noncomputable def solve_some_number (x : ℝ) : Prop :=
  let expr := (x - (8 / 7) * 5 + 10)
  expr = 13.285714285714286

theorem find_x_value : ∃ x : ℝ, solve_some_number x ∧ x = 9 := by
  sorry

end NUMINAMATH_GPT_find_x_value_l1339_133943


namespace NUMINAMATH_GPT_rhombus_area_outside_circle_l1339_133910

theorem rhombus_area_outside_circle (d : ℝ) (r : ℝ) (h_d : d = 10) (h_r : r = 3) : 
  (d * d / 2 - 9 * Real.pi) > 9 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_outside_circle_l1339_133910


namespace NUMINAMATH_GPT_prove_ax5_by5_l1339_133937

variables {a b x y : ℝ}

theorem prove_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 30)
                      (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 :=
sorry

end NUMINAMATH_GPT_prove_ax5_by5_l1339_133937


namespace NUMINAMATH_GPT_sale_price_after_discounts_l1339_133996

def original_price : ℝ := 400.00
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.10

theorem sale_price_after_discounts (orig : ℝ) (d1 d2 d3 : ℝ) :
  orig = original_price →
  d1 = discount1 →
  d2 = discount2 →
  d3 = discount3 →
  orig * (1 - d1) * (1 - d2) * (1 - d3) = 243.00 := by
  sorry

end NUMINAMATH_GPT_sale_price_after_discounts_l1339_133996


namespace NUMINAMATH_GPT_trip_total_time_l1339_133984

theorem trip_total_time 
  (x : ℕ) 
  (h1 : 30 * 5 = 150) 
  (h2 : 42 * x + 150 = 38 * (x + 5)) 
  (h3 : 38 = (150 + 42 * x) / (5 + x)) : 
  5 + x = 15 := by
  sorry

end NUMINAMATH_GPT_trip_total_time_l1339_133984


namespace NUMINAMATH_GPT_largest_m_l1339_133946

noncomputable def max_min_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : ℝ :=
  min (a * b) (min (b * c) (c * a))

theorem largest_m (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : max_min_ab_bc_ca a b c ha hb hc h1 h2 = 6.75 :=
by
  sorry

end NUMINAMATH_GPT_largest_m_l1339_133946


namespace NUMINAMATH_GPT_part1_monotonicity_part2_find_range_l1339_133945

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end NUMINAMATH_GPT_part1_monotonicity_part2_find_range_l1339_133945


namespace NUMINAMATH_GPT_eccentricity_proof_l1339_133993

variables (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
def ellipse_eq (x y: ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def circle_eq (x y: ℝ) := x^2 + y^2 = b^2

-- Conditions
def a_eq_3b : Prop := a = 3 * b
def major_minor_axis_relation : Prop := a^2 = b^2 + c^2

-- To prove
theorem eccentricity_proof 
  (h3 : a_eq_3b a b)
  (h4 : major_minor_axis_relation a b c) :
  (c / a) = (2 * Real.sqrt 2 / 3) := 
  sorry

end NUMINAMATH_GPT_eccentricity_proof_l1339_133993


namespace NUMINAMATH_GPT_unique_largest_negative_integer_l1339_133986

theorem unique_largest_negative_integer :
  ∃! x : ℤ, x = -1 ∧ (∀ y : ℤ, y < 0 → x ≥ y) :=
by
  sorry

end NUMINAMATH_GPT_unique_largest_negative_integer_l1339_133986


namespace NUMINAMATH_GPT_solution_ratio_l1339_133998

-- Describe the problem conditions
variable (a b : ℝ) -- amounts of solutions A and B

-- conditions
def proportion_A : ℝ := 0.20 -- Alcohol concentration in solution A
def proportion_B : ℝ := 0.60 -- Alcohol concentration in solution B
def final_proportion : ℝ := 0.40 -- Final alcohol concentration

-- Lean statement
theorem solution_ratio (h : 0.20 * a + 0.60 * b = 0.40 * (a + b)) : a = b := by
  sorry

end NUMINAMATH_GPT_solution_ratio_l1339_133998


namespace NUMINAMATH_GPT_set_intersection_complement_l1339_133950

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

theorem set_intersection_complement :
  ((U \ A) ∩ B) = {1, 3, 7} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1339_133950


namespace NUMINAMATH_GPT_time_difference_between_car_and_minivan_arrival_l1339_133963

variable (car_speed : ℝ := 40)
variable (minivan_speed : ℝ := 50)
variable (pass_time : ℝ := 1 / 6) -- in hours

theorem time_difference_between_car_and_minivan_arrival :
  (60 * (1 / 6 - (20 / 3 / 50))) = 2 := sorry

end NUMINAMATH_GPT_time_difference_between_car_and_minivan_arrival_l1339_133963


namespace NUMINAMATH_GPT_Cooper_age_l1339_133951

variable (X : ℕ)
variable (Dante : ℕ)
variable (Maria : ℕ)

theorem Cooper_age (h1 : Dante = 2 * X) (h2 : Maria = 2 * X + 1) (h3 : X + Dante + Maria = 31) : X = 6 :=
by
  -- Proof is omitted as indicated
  sorry

end NUMINAMATH_GPT_Cooper_age_l1339_133951


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l1339_133934

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l1339_133934


namespace NUMINAMATH_GPT_multiply_equality_l1339_133927

variable (a b c d e : ℝ)

theorem multiply_equality
  (h1 : a = 2994)
  (h2 : b = 14.5)
  (h3 : c = 173)
  (h4 : d = 29.94)
  (h5 : e = 1.45)
  (h6 : a * b = c) : d * e = 1.73 :=
sorry

end NUMINAMATH_GPT_multiply_equality_l1339_133927


namespace NUMINAMATH_GPT_james_sheets_of_paper_l1339_133936

noncomputable def sheets_of_paper (books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) : ℕ :=
  (books * pages_per_book) / (pages_per_side * sides_per_sheet)

theorem james_sheets_of_paper :
  sheets_of_paper 2 600 4 2 = 150 :=
by
  sorry

end NUMINAMATH_GPT_james_sheets_of_paper_l1339_133936


namespace NUMINAMATH_GPT_defective_units_shipped_for_sale_l1339_133933

theorem defective_units_shipped_for_sale (d p : ℝ) (h1 : d = 0.09) (h2 : p = 0.04) : (d * p * 100 = 0.36) :=
by 
  -- Assuming some calculation steps 
  sorry

end NUMINAMATH_GPT_defective_units_shipped_for_sale_l1339_133933


namespace NUMINAMATH_GPT_geom_seq_find_b3_l1339_133935

-- Given conditions
def is_geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def geom_seq_condition (b : ℕ → ℝ) : Prop :=
  is_geometric_seq b ∧ b 2 * b 3 * b 4 = 8

-- Proof statement: We need to prove that b 3 = 2
theorem geom_seq_find_b3 (b : ℕ → ℝ) (h : geom_seq_condition b) : b 3 = 2 :=
  sorry

end NUMINAMATH_GPT_geom_seq_find_b3_l1339_133935


namespace NUMINAMATH_GPT_twenty_four_point_game_l1339_133952

theorem twenty_four_point_game : (9 + 7) * 3 / 2 = 24 := by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_twenty_four_point_game_l1339_133952


namespace NUMINAMATH_GPT_calculate_y_position_l1339_133929

/--
Given a number line with equally spaced markings, if eight steps are taken from \( 0 \) to \( 32 \),
then the position \( y \) after five steps can be calculated.
-/
theorem calculate_y_position : 
    ∃ y : ℕ, (∀ (step length : ℕ), (8 * step = 32) ∧ (y = 5 * length) → y = 20) :=
by
  -- Provide initial definitions based on the conditions
  let step := 4
  let length := 4
  use (5 * length)
  sorry

end NUMINAMATH_GPT_calculate_y_position_l1339_133929


namespace NUMINAMATH_GPT_a_1995_eq_l1339_133954

def a_3 : ℚ := (2 + 3) / (1 + 6)

def a (n : ℕ) : ℚ :=
  if n = 3 then a_3
  else if n ≥ 4 then
    let a_n_minus_1 := a (n - 1)
    (a_n_minus_1 + n) / (1 + n * a_n_minus_1)
  else
    0 -- We only care about n ≥ 3 in this problem

-- The problem itself
theorem a_1995_eq :
  a 1995 = 1991009 / 1991011 :=
by
  sorry

end NUMINAMATH_GPT_a_1995_eq_l1339_133954


namespace NUMINAMATH_GPT_value_computation_l1339_133932

theorem value_computation (N : ℝ) (h1 : 1.20 * N = 2400) : 0.20 * N = 400 := 
by
  sorry

end NUMINAMATH_GPT_value_computation_l1339_133932


namespace NUMINAMATH_GPT_solve_equation_l1339_133911

theorem solve_equation (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) :
  ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1 )
  ↔ x = 5 / 4 ∨ x = -2 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1339_133911


namespace NUMINAMATH_GPT_inequality_solution_set_l1339_133977

theorem inequality_solution_set :
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1339_133977


namespace NUMINAMATH_GPT_relationship_between_a_b_l1339_133968

theorem relationship_between_a_b (a b x : ℝ) (h1 : 2 * x = a + b) (h2 : 2 * x^2 = a^2 - b^2) : 
  a = -b ∨ a = 3 * b :=
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_l1339_133968


namespace NUMINAMATH_GPT_math_competition_rankings_l1339_133989

noncomputable def rankings (n : ℕ) : ℕ → Prop := sorry

theorem math_competition_rankings :
  (∀ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    
    -- A's guesses
    (rankings A 1 → rankings B 3 ∧ rankings C 5) →
    -- B's guesses
    (rankings B 2 → rankings E 4 ∧ rankings D 5) →
    -- C's guesses
    (rankings C 3 → rankings A 1 ∧ rankings E 4) →
    -- D's guesses
    (rankings D 4 → rankings C 1 ∧ rankings D 2) →
    -- E's guesses
    (rankings E 5 → rankings A 3 ∧ rankings D 4) →
    -- Condition that each position is guessed correctly by someone
    (∃ i, rankings A i) ∧
    (∃ i, rankings B i) ∧
    (∃ i, rankings C i) ∧
    (∃ i, rankings D i) ∧
    (∃ i, rankings E i) →
    
    -- The actual placing according to derived solution
    rankings A 1 ∧ 
    rankings D 2 ∧ 
    rankings B 3 ∧ 
    rankings E 4 ∧ 
    rankings C 5) :=
sorry

end NUMINAMATH_GPT_math_competition_rankings_l1339_133989


namespace NUMINAMATH_GPT_sheila_hourly_wage_l1339_133957

-- Definition of conditions
def hours_per_day_mon_wed_fri := 8
def days_mon_wed_fri := 3
def hours_per_day_tue_thu := 6
def days_tue_thu := 2
def weekly_earnings := 432

-- Variables derived from conditions
def total_hours_mon_wed_fri := hours_per_day_mon_wed_fri * days_mon_wed_fri
def total_hours_tue_thu := hours_per_day_tue_thu * days_tue_thu
def total_hours_per_week := total_hours_mon_wed_fri + total_hours_tue_thu

-- Proof statement
theorem sheila_hourly_wage : (weekly_earnings / total_hours_per_week) = 12 := 
sorry

end NUMINAMATH_GPT_sheila_hourly_wage_l1339_133957


namespace NUMINAMATH_GPT_total_number_of_fleas_l1339_133995

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end NUMINAMATH_GPT_total_number_of_fleas_l1339_133995


namespace NUMINAMATH_GPT_total_pokemon_cards_l1339_133953

-- Definitions based on conditions
def jenny_cards : ℕ := 6
def orlando_cards : ℕ := jenny_cards + 2
def richard_cards : ℕ := 3 * orlando_cards

-- The theorem stating the total number of cards
theorem total_pokemon_cards : jenny_cards + orlando_cards + richard_cards = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_pokemon_cards_l1339_133953


namespace NUMINAMATH_GPT_find_abc_of_N_l1339_133980

theorem find_abc_of_N :
  ∃ N : ℕ, (N % 10000) = (N + 2) % 10000 ∧ 
            (N % 16 = 15 ∧ (N + 2) % 16 = 1) ∧ 
            ∃ abc : ℕ, (100 ≤ abc ∧ abc < 1000) ∧ 
            (N % 1000) = 100 * abc + 99 := sorry

end NUMINAMATH_GPT_find_abc_of_N_l1339_133980


namespace NUMINAMATH_GPT_smallest_positive_number_is_correct_l1339_133941

noncomputable def smallest_positive_number : ℝ := 20 - 5 * Real.sqrt 15

theorem smallest_positive_number_is_correct :
  ∀ n,
    (n = 12 - 3 * Real.sqrt 12 ∨ n = 3 * Real.sqrt 12 - 11 ∨ n = 20 - 5 * Real.sqrt 15 ∨ n = 55 - 11 * Real.sqrt 30 ∨ n = 11 * Real.sqrt 30 - 55) →
    n > 0 → smallest_positive_number ≤ n :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_number_is_correct_l1339_133941


namespace NUMINAMATH_GPT_inequality_proof_l1339_133904

noncomputable def problem_statement (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : Prop :=
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z))) ≤ 
  ((x + y + z) / 3) ^ (5 / 8)

-- The statement below is what needs to be proven.
theorem inequality_proof (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : problem_statement x y z positive_x positive_y positive_z condition :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1339_133904


namespace NUMINAMATH_GPT_geo_seq_sum_S4_l1339_133902

noncomputable def geom_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geo_seq_sum_S4 {a : ℝ} {q : ℝ} (h1 : a * q^2 - a = 15) (h2 : a * q - a = 5) :
  geom_seq_sum a q 4 = 75 :=
by
  sorry

end NUMINAMATH_GPT_geo_seq_sum_S4_l1339_133902


namespace NUMINAMATH_GPT_equation_of_line_l1339_133900

theorem equation_of_line (x y : ℝ) 
  (l1 : 4 * x + y + 6 = 0) 
  (l2 : 3 * x - 5 * y - 6 = 0) 
  (midpoint_origin : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * x₁ + y₁ + 6 = 0) ∧ 
    (3 * x₂ - 5 * y₂ - 6 = 0) ∧ 
    (x₁ + x₂ = 0) ∧ 
    (y₁ + y₂ = 0)) : 
  7 * x + 4 * y = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_l1339_133900


namespace NUMINAMATH_GPT_cab_company_charge_l1339_133979

-- Defining the conditions
def total_cost : ℝ := 23
def base_price : ℝ := 3
def distance_to_hospital : ℝ := 5

-- Theorem stating the cost per mile
theorem cab_company_charge : 
  (total_cost - base_price) / distance_to_hospital = 4 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_cab_company_charge_l1339_133979


namespace NUMINAMATH_GPT_abs_eq_neg_imp_nonpos_l1339_133913

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end NUMINAMATH_GPT_abs_eq_neg_imp_nonpos_l1339_133913


namespace NUMINAMATH_GPT_men_in_first_group_l1339_133914

theorem men_in_first_group (M : ℕ) (h1 : (M * 15) = (M + 0) * 15) (h2 : (15 * 36) = 540) : M = 36 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_men_in_first_group_l1339_133914


namespace NUMINAMATH_GPT_pipe_A_fill_time_l1339_133939

theorem pipe_A_fill_time (B C : ℝ) (hB : B = 8) (hC : C = 14.4) (hB_not_zero : B ≠ 0) (hC_not_zero : C ≠ 0) :
  ∃ (A : ℝ), (1 / A + 1 / B = 1 / C) ∧ A = 24 :=
by
  sorry

end NUMINAMATH_GPT_pipe_A_fill_time_l1339_133939


namespace NUMINAMATH_GPT_tamia_bell_pepper_pieces_l1339_133947

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end NUMINAMATH_GPT_tamia_bell_pepper_pieces_l1339_133947


namespace NUMINAMATH_GPT_closest_approx_of_q_l1339_133916

theorem closest_approx_of_q :
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  abs (q - 9.24) < 0.005 := 
by 
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  sorry

end NUMINAMATH_GPT_closest_approx_of_q_l1339_133916


namespace NUMINAMATH_GPT_commute_time_l1339_133992

theorem commute_time (d w t : ℝ) (x : ℝ) (h_distance : d = 1.5) (h_walking_speed : w = 3) (h_train_speed : t = 20)
  (h_extra_time : 30 = 4.5 + x + 2) : x = 25.5 :=
by {
  -- Add the statement of the proof
  sorry
}

end NUMINAMATH_GPT_commute_time_l1339_133992


namespace NUMINAMATH_GPT_geometric_series_solution_l1339_133924

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_solution_l1339_133924


namespace NUMINAMATH_GPT_volume_of_56_ounces_is_24_cubic_inches_l1339_133919

-- Given information as premises
def directlyProportional (V W : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ V = k * W

-- The specific conditions in the problem
def initial_volume := 48   -- in cubic inches
def initial_weight := 112  -- in ounces
def target_weight := 56    -- in ounces
def target_volume := 24    -- in cubic inches (the value we need to prove)

-- The theorem statement 
theorem volume_of_56_ounces_is_24_cubic_inches
  (h1 : directlyProportional initial_volume initial_weight)
  (h2 : directlyProportional target_volume target_weight)
  (h3 : target_weight = 56)
  (h4 : initial_volume = 48)
  (h5 : initial_weight = 112) :
  target_volume = 24 :=
sorry -- Proof not required as per instructions

end NUMINAMATH_GPT_volume_of_56_ounces_is_24_cubic_inches_l1339_133919


namespace NUMINAMATH_GPT_find_a_l1339_133926

noncomputable def a := 1/2

theorem find_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 1 - a^2 = 3/4) : a = 1/2 :=
sorry

end NUMINAMATH_GPT_find_a_l1339_133926


namespace NUMINAMATH_GPT_sqrt_221_range_l1339_133905

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end NUMINAMATH_GPT_sqrt_221_range_l1339_133905


namespace NUMINAMATH_GPT_johns_age_l1339_133930

-- Define the variables and conditions
def age_problem (j d : ℕ) : Prop :=
j = d - 34 ∧ j + d = 84

-- State the theorem to prove that John's age is 25
theorem johns_age : ∃ (j d : ℕ), age_problem j d ∧ j = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_johns_age_l1339_133930


namespace NUMINAMATH_GPT_five_digit_numbers_with_alternating_parity_l1339_133999

theorem five_digit_numbers_with_alternating_parity : 
  ∃ n : ℕ, n = 5625 ∧ ∀ (x : ℕ), (10000 ≤ x ∧ x < 100000) → 
    (∀ i, i < 4 → (((x / 10^i) % 10) % 2 ≠ ((x / 10^(i+1)) % 10) % 2)) ↔ 
    (x = 5625) := 
sorry

end NUMINAMATH_GPT_five_digit_numbers_with_alternating_parity_l1339_133999


namespace NUMINAMATH_GPT_find_number_l1339_133912

theorem find_number : ∃ x : ℝ, (x / 6 * 12 = 10) ∧ x = 5 :=
by
 sorry

end NUMINAMATH_GPT_find_number_l1339_133912


namespace NUMINAMATH_GPT_hyperbola_asymptotes_n_l1339_133987

theorem hyperbola_asymptotes_n {y x : ℝ} (n : ℝ) (H : ∀ x y, (y^2 / 16) - (x^2 / 9) = 1 → y = n * x ∨ y = -n * x) : n = 4/3 :=
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_n_l1339_133987


namespace NUMINAMATH_GPT_find_y_l1339_133938

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 12) (h2 : x = 6) : y = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1339_133938


namespace NUMINAMATH_GPT_find_value_l1339_133975

variable (N : ℝ)

def condition : Prop := (1 / 4) * (1 / 3) * (2 / 5) * N = 16

theorem find_value (h : condition N) : (1 / 3) * (2 / 5) * N = 64 :=
sorry

end NUMINAMATH_GPT_find_value_l1339_133975


namespace NUMINAMATH_GPT_smallest_fraction_numerator_l1339_133915

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (5 * b < 7 * a) ∧ 
    ∀ (a' b' : ℕ), (10 ≤ a' ∧ a' < 100) ∧ (10 ≤ b' ∧ b' < 100) ∧ (5 * b' < 7 * a') →
    (a * b' ≤ a' * b) → a = 68 :=
sorry

end NUMINAMATH_GPT_smallest_fraction_numerator_l1339_133915


namespace NUMINAMATH_GPT_smallest_y_l1339_133948

noncomputable def x : ℕ := 3 * 40 * 75

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k^3 = n

theorem smallest_y (y : ℕ) (hy : y = 3) :
  ∀ (x : ℕ), x = 3 * 40 * 75 → is_perfect_cube (x * y) :=
by
  intro x hx
  unfold is_perfect_cube
  exists 5 -- This is just a placeholder value; the proof would find the correct k
  sorry

end NUMINAMATH_GPT_smallest_y_l1339_133948


namespace NUMINAMATH_GPT_solve_equation_l1339_133922

theorem solve_equation : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7/4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1339_133922


namespace NUMINAMATH_GPT_g_1987_l1339_133928

def g (x : ℕ) : ℚ := sorry

axiom g_defined_for_all (x : ℕ) : true

axiom g1 : g 1 = 1

axiom g_rec (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b) + 1

theorem g_1987 : g 1987 = 2 := sorry

end NUMINAMATH_GPT_g_1987_l1339_133928


namespace NUMINAMATH_GPT_dogwood_tree_count_l1339_133944

theorem dogwood_tree_count (n d1 d2 d3 d4 d5: ℕ) 
  (h1: n = 39)
  (h2: d1 = 24)
  (h3: d2 = d1 / 2)
  (h4: d3 = 4 * d2)
  (h5: d4 = 5)
  (h6: d5 = 15):
  n + d1 + d2 + d3 + d4 + d5 = 143 :=
by
  sorry

end NUMINAMATH_GPT_dogwood_tree_count_l1339_133944


namespace NUMINAMATH_GPT_abs_diff_kth_power_l1339_133974

theorem abs_diff_kth_power (k : ℕ) (a b : ℤ) (x y : ℤ)
  (hk : 2 ≤ k)
  (ha : a ≠ 0) (hb : b ≠ 0)
  (hab_odd : (a + b) % 2 = 1)
  (hxy : 0 < |x - y| ∧ |x - y| ≤ 2)
  (h_eq : a^k * x - b^k * y = a - b) :
  ∃ m : ℤ, |a - b| = m^k :=
sorry

end NUMINAMATH_GPT_abs_diff_kth_power_l1339_133974


namespace NUMINAMATH_GPT_infinite_solutions_ax2_by2_eq_z3_l1339_133961

theorem infinite_solutions_ax2_by2_eq_z3 
  (a b : ℤ) 
  (coprime_ab : Int.gcd a b = 1) :
  ∃ (x y z : ℤ), (∀ n : ℤ, ∃ (x y z : ℤ), a * x^2 + b * y^2 = z^3 
  ∧ Int.gcd x y = 1) := 
sorry

end NUMINAMATH_GPT_infinite_solutions_ax2_by2_eq_z3_l1339_133961


namespace NUMINAMATH_GPT_people_to_right_of_taehyung_l1339_133917

-- Given conditions
def total_people : Nat := 11
def people_to_left_of_taehyung : Nat := 5

-- Question and proof: How many people are standing to Taehyung's right?
theorem people_to_right_of_taehyung : total_people - people_to_left_of_taehyung - 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_people_to_right_of_taehyung_l1339_133917


namespace NUMINAMATH_GPT_angle_complement_supplement_l1339_133958

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end NUMINAMATH_GPT_angle_complement_supplement_l1339_133958


namespace NUMINAMATH_GPT_factor_equivalence_l1339_133985

noncomputable def given_expression (x : ℝ) :=
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5)

noncomputable def target_form (x : ℝ) :=
  7 * x^2 * (x + 68 / 7)

theorem factor_equivalence (x : ℝ) : given_expression x = target_form x :=
by
  sorry

end NUMINAMATH_GPT_factor_equivalence_l1339_133985


namespace NUMINAMATH_GPT_problem_2023_divisible_by_consecutive_integers_l1339_133966

theorem problem_2023_divisible_by_consecutive_integers :
  ∃ (n : ℕ), (n = 2022 ∨ n = 2023 ∨ n = 2024) ∧ (2023^2023 - 2023^2021) % n = 0 :=
sorry

end NUMINAMATH_GPT_problem_2023_divisible_by_consecutive_integers_l1339_133966


namespace NUMINAMATH_GPT_initial_apples_count_l1339_133907

variable (initial_apples : ℕ)
variable (used_apples : ℕ := 2)
variable (bought_apples : ℕ := 23)
variable (final_apples : ℕ := 38)

theorem initial_apples_count :
  initial_apples - used_apples + bought_apples = final_apples ↔ initial_apples = 17 := by
  sorry

end NUMINAMATH_GPT_initial_apples_count_l1339_133907


namespace NUMINAMATH_GPT_find_inscription_l1339_133970

-- Definitions for the conditions
def identical_inscriptions (box1 box2 : String) : Prop :=
  box1 = box2

def conclusion_same_master (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini"

def cannot_identify_master (box : String) : Prop :=
  ¬(∀ (made_by : String → Prop), made_by "Bellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Cellini")

def single_casket_indeterminate (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini")

-- Inscription on the boxes
def inscription := "At least one of these boxes was made by Cellini's son."

-- The Lean statement for the proof
theorem find_inscription (box1 box2 : String)
  (h1 : identical_inscriptions box1 box2)
  (h2 : conclusion_same_master box1)
  (h3 : cannot_identify_master box1)
  (h4 : single_casket_indeterminate box1) :
  box1 = inscription :=
sorry

end NUMINAMATH_GPT_find_inscription_l1339_133970


namespace NUMINAMATH_GPT_weight_of_replaced_person_is_correct_l1339_133965

-- Define a constant representing the number of persons in the group.
def num_people : ℕ := 10
-- Define a constant representing the weight of the new person.
def new_person_weight : ℝ := 110
-- Define a constant representing the increase in average weight when the new person joins.
def avg_weight_increase : ℝ := 5
-- Define the weight of the person who was replaced.
noncomputable def replaced_person_weight : ℝ :=
  new_person_weight - num_people * avg_weight_increase

-- Prove that the weight of the replaced person is 60 kg.
theorem weight_of_replaced_person_is_correct : replaced_person_weight = 60 :=
by
  -- Skip the detailed proof steps.
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_is_correct_l1339_133965


namespace NUMINAMATH_GPT_least_k_divisible_by_2160_l1339_133994

theorem least_k_divisible_by_2160 (k : ℤ) : k^3 ∣ 2160 → k ≥ 60 := by
  sorry

end NUMINAMATH_GPT_least_k_divisible_by_2160_l1339_133994


namespace NUMINAMATH_GPT_original_cost_of_car_l1339_133959

noncomputable def original_cost (C : ℝ) : ℝ :=
  if h : C + 13000 ≠ 0 then (60900 - (C + 13000)) / (C + 13000) * 100 else 0

theorem original_cost_of_car 
  (C : ℝ) 
  (h1 : original_cost C = 10.727272727272727)
  (h2 : 60900 - (C + 13000) > 0) :
  C = 433500 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_car_l1339_133959


namespace NUMINAMATH_GPT_line_through_intersection_and_origin_l1339_133982

theorem line_through_intersection_and_origin :
  ∃ (x y : ℝ), (2*x + y = 3) ∧ (x + 4*y = 2) ∧ (x - 10*y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_intersection_and_origin_l1339_133982


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l1339_133967

theorem revenue_from_full_price_tickets (f h p : ℕ) 
    (h1 : f + h = 160) 
    (h2 : f * p + h * (p / 2) = 2400) 
    (h3 : h = 160 - f)
    (h4 : 2 * 2400 = 4800) :
  f * p = 800 := 
sorry

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l1339_133967


namespace NUMINAMATH_GPT_set_difference_correct_l1339_133997

-- Define the sets A and B
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}

-- Define the set difference A - B
def A_minus_B : Set ℤ := {x | x ∈ A ∧ x ∉ B} -- This is the operation A - B

-- The theorem stating the required proof
theorem set_difference_correct : A_minus_B = {1, 3, 9} :=
by {
  -- Proof goes here; however, we have requested no proof, so we put sorry.
  sorry
}

end NUMINAMATH_GPT_set_difference_correct_l1339_133997


namespace NUMINAMATH_GPT_fraction_of_income_to_taxes_l1339_133918

noncomputable def joe_income : ℕ := 2120
noncomputable def joe_taxes : ℕ := 848

theorem fraction_of_income_to_taxes : (joe_taxes / gcd joe_taxes joe_income) / (joe_income / gcd joe_taxes joe_income) = 106 / 265 := sorry

end NUMINAMATH_GPT_fraction_of_income_to_taxes_l1339_133918


namespace NUMINAMATH_GPT_smallest_n_divisible_by_2022_l1339_133969

theorem smallest_n_divisible_by_2022 (n : ℕ) (h1 : n > 1) (h2 : (n^7 - 1) % 2022 = 0) : n = 79 :=
sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_2022_l1339_133969


namespace NUMINAMATH_GPT_max_A_k_value_l1339_133973

noncomputable def A_k (k : ℕ) : ℝ := (19^k + 66^k) / k.factorial

theorem max_A_k_value : 
  ∃ k : ℕ, (∀ m : ℕ, (A_k m ≤ A_k k)) ∧ k = 65 :=
by
  sorry

end NUMINAMATH_GPT_max_A_k_value_l1339_133973


namespace NUMINAMATH_GPT_min_period_f_and_max_value_g_l1339_133981

open Real

noncomputable def f (x : ℝ) : ℝ := abs (sin x) + abs (cos x)
noncomputable def g (x : ℝ) : ℝ := sin x ^ 3 - sin x

theorem min_period_f_and_max_value_g :
  (∀ m : ℝ, (∀ x : ℝ, f (x + m) = f x) -> m = π / 2) ∧ 
  (∃ n : ℝ, ∀ x : ℝ, g x ≤ n ∧ (∃ x : ℝ, g x = n)) ∧ 
  (∃ mn : ℝ, mn = (π / 2) * (2 * sqrt 3 / 9)) := 
by sorry

end NUMINAMATH_GPT_min_period_f_and_max_value_g_l1339_133981


namespace NUMINAMATH_GPT_find_y_value_l1339_133923

theorem find_y_value (y : ℕ) : (1/8 * 2^36 = 2^33) ∧ (8^y = 2^(3 * y)) → y = 11 :=
by
  intros h
  -- additional elaboration to verify each step using Lean, skipped for simplicity
  sorry

end NUMINAMATH_GPT_find_y_value_l1339_133923


namespace NUMINAMATH_GPT_complementary_angles_positive_difference_l1339_133978

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end NUMINAMATH_GPT_complementary_angles_positive_difference_l1339_133978


namespace NUMINAMATH_GPT_find_multiple_of_sum_l1339_133983

-- Define the conditions and the problem statement in Lean
theorem find_multiple_of_sum (a b m : ℤ) 
  (h1 : b = 8) 
  (h2 : b - a = 3) 
  (h3 : a * b = 14 + m * (a + b)) : 
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_sum_l1339_133983


namespace NUMINAMATH_GPT_paidAmount_Y_l1339_133956

theorem paidAmount_Y (X Y : ℝ) (h1 : X + Y = 638) (h2 : X = 1.2 * Y) : Y = 290 :=
by
  sorry

end NUMINAMATH_GPT_paidAmount_Y_l1339_133956


namespace NUMINAMATH_GPT_inequality_iff_positive_l1339_133903

variable (x y : ℝ)

theorem inequality_iff_positive :
  x + y > abs (x - y) ↔ x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_inequality_iff_positive_l1339_133903


namespace NUMINAMATH_GPT_find_k_l1339_133988

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l1339_133988


namespace NUMINAMATH_GPT_alex_cakes_l1339_133972

theorem alex_cakes :
  let slices_first_cake := 8
  let slices_second_cake := 12
  let given_away_friends_first := slices_first_cake / 4
  let remaining_after_friends_first := slices_first_cake - given_away_friends_first
  let given_away_family_first := remaining_after_friends_first / 2
  let remaining_after_family_first := remaining_after_friends_first - given_away_family_first
  let stored_in_freezer_first := remaining_after_family_first / 4
  let remaining_after_freezer_first := remaining_after_family_first - stored_in_freezer_first
  let remaining_after_eating_first := remaining_after_freezer_first - 2
  
  let given_away_friends_second := slices_second_cake / 3
  let remaining_after_friends_second := slices_second_cake - given_away_friends_second
  let given_away_family_second := remaining_after_friends_second / 6
  let remaining_after_family_second := remaining_after_friends_second - given_away_family_second
  let stored_in_freezer_second := remaining_after_family_second / 4
  let remaining_after_freezer_second := remaining_after_family_second - stored_in_freezer_second
  let remaining_after_eating_second := remaining_after_freezer_second - 1

  remaining_after_eating_first + stored_in_freezer_first + remaining_after_eating_second + stored_in_freezer_second = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_alex_cakes_l1339_133972


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_value_l1339_133931

theorem arithmetic_sequence_a3_value 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + 2) 
  (h2 : (a 1 + 2)^2 = a 1 * (a 1 + 8)) : 
  a 2 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_value_l1339_133931


namespace NUMINAMATH_GPT_line_intersects_ellipse_two_points_l1339_133920

theorem line_intersects_ellipse_two_points (k b : ℝ) : 
  (-2 < b) ∧ (b < 2) ↔ ∀ x y : ℝ, (y = k * x + b) ↔ (x ^ 2 / 9 + y ^ 2 / 4 = 1) → true :=
sorry

end NUMINAMATH_GPT_line_intersects_ellipse_two_points_l1339_133920


namespace NUMINAMATH_GPT_planting_rate_l1339_133964

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end NUMINAMATH_GPT_planting_rate_l1339_133964
