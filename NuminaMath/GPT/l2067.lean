import Mathlib

namespace sum_of_two_numbers_l2067_206798

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  x + y = (16 * Real.sqrt 3) / 3 := 
sorry

end sum_of_two_numbers_l2067_206798


namespace eval_expression_l2067_206776

theorem eval_expression : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end eval_expression_l2067_206776


namespace Papi_Calot_has_to_buy_141_plants_l2067_206720

noncomputable def calc_number_of_plants : Nat :=
  let initial_plants := 7 * 18
  let additional_plants := 15
  initial_plants + additional_plants

theorem Papi_Calot_has_to_buy_141_plants :
  calc_number_of_plants = 141 :=
by
  sorry

end Papi_Calot_has_to_buy_141_plants_l2067_206720


namespace bicycle_trip_length_l2067_206796

def total_distance (days1 day1 miles1 day2 miles2: ℕ) : ℕ :=
  days1 * miles1 + day2 * miles2

theorem bicycle_trip_length :
  total_distance 12 12 1 6 = 150 :=
by
  sorry

end bicycle_trip_length_l2067_206796


namespace age_ratio_in_2_years_is_2_1_l2067_206785

-- Define the ages and conditions
def son_age (current_year : ℕ) : ℕ := 20
def man_age (current_year : ℕ) : ℕ := son_age current_year + 22

def son_age_in_2_years (current_year : ℕ) : ℕ := son_age current_year + 2
def man_age_in_2_years (current_year : ℕ) : ℕ := man_age current_year + 2

-- The theorem stating the ratio of the man's age to the son's age in two years is 2:1
theorem age_ratio_in_2_years_is_2_1 (current_year : ℕ) :
  man_age_in_2_years current_year = 2 * son_age_in_2_years current_year :=
by
  sorry

end age_ratio_in_2_years_is_2_1_l2067_206785


namespace not_a_cube_l2067_206765

theorem not_a_cube (a b : ℤ) : ¬ ∃ c : ℤ, a^3 + b^3 + 4 = c^3 := 
sorry

end not_a_cube_l2067_206765


namespace P_eq_Q_at_x_l2067_206797

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 2
def Q (x : ℝ) : ℝ := 0

theorem P_eq_Q_at_x :
  ∃ x : ℝ, P x = Q x ∧ x = 1 :=
by
  sorry

end P_eq_Q_at_x_l2067_206797


namespace first_player_guaranteed_win_l2067_206762

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem first_player_guaranteed_win (n : ℕ) (h : n > 1) : 
  ¬ is_power_of_two n ↔ ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ (∀ k : ℕ, m ≤ k + 1 → ∀ t, t ≤ m → ∃ r, r = k + 1 ∧ r <= m) → 
                                (∃ l : ℕ, (l = 1) → true) :=
sorry

end first_player_guaranteed_win_l2067_206762


namespace count_whole_numbers_in_interval_l2067_206791

open Real

theorem count_whole_numbers_in_interval : 
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℕ, (sqrt 7 < x ∧ x < exp 2) ↔ (3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end count_whole_numbers_in_interval_l2067_206791


namespace sector_area_l2067_206742

theorem sector_area (r : ℝ) (h1 : r = 2) (h2 : 2 * r + r * ((2 * π * r - 2) / r) = 4 * π) :
  (1 / 2) * r^2 * ((4 * π - 2) / r) = 4 * π - 2 :=
by
  sorry

end sector_area_l2067_206742


namespace calculate_ratio_l2067_206704

theorem calculate_ratio (l m n : ℝ) :
  let D := (l + 1, 1, 1)
  let E := (1, m + 1, 1)
  let F := (1, 1, n + 1)
  let AB_sq := 4 * ((n - m) ^ 2)
  let AC_sq := 4 * ((l - n) ^ 2)
  let BC_sq := 4 * ((m - l) ^ 2)
  (AB_sq + AC_sq + BC_sq + 3) / (l^2 + m^2 + n^2 + 3) = 8 := by
  sorry

end calculate_ratio_l2067_206704


namespace cost_per_ream_is_27_l2067_206782

-- Let ream_sheets be the number of sheets in one ream.
def ream_sheets : ℕ := 500

-- Let total_sheets be the total number of sheets needed.
def total_sheets : ℕ := 5000

-- Let total_cost be the total cost to buy the total number of sheets.
def total_cost : ℕ := 270

-- We need to prove that the cost per ream (in dollars) is 27.
theorem cost_per_ream_is_27 : (total_cost / (total_sheets / ream_sheets)) = 27 := 
by
  sorry

end cost_per_ream_is_27_l2067_206782


namespace find_coords_C_l2067_206789

-- Define the coordinates of given points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- The proof problem wrapped in a lean theorem
theorem find_coords_C (C : ℝ × ℝ) 
  (h1 : AB = AC) (h2 : (D.1, D.2) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) :
  C = (-1, 5) :=
sorry

end find_coords_C_l2067_206789


namespace brad_siblings_product_l2067_206700

theorem brad_siblings_product (S B : ℕ) (hS : S = 5) (hB : B = 7) : S * B = 35 :=
by
  have : S = 5 := hS
  have : B = 7 := hB
  sorry

end brad_siblings_product_l2067_206700


namespace like_terms_calc_l2067_206739

theorem like_terms_calc {m n : ℕ} (h1 : m + 2 = 6) (h2 : n + 1 = 3) : (- (m : ℤ))^3 + (n : ℤ)^2 = -60 :=
  sorry

end like_terms_calc_l2067_206739


namespace remainders_inequalities_l2067_206770

theorem remainders_inequalities
  (X Y M A B s t u : ℕ)
  (h1 : X > Y)
  (h2 : X = Y + 8)
  (h3 : X % M = A)
  (h4 : Y % M = B)
  (h5 : s = (X^2) % M)
  (h6 : t = (Y^2) % M)
  (h7 : u = (A * B)^2 % M) :
  s ≠ t ∧ t ≠ u ∧ s ≠ u :=
sorry

end remainders_inequalities_l2067_206770


namespace one_third_of_1206_is_201_percent_of_200_l2067_206745

theorem one_third_of_1206_is_201_percent_of_200 : 
  (1 / 3) * 1206 = 402 ∧ 402 / 200 = 201 / 100 :=
by
  sorry

end one_third_of_1206_is_201_percent_of_200_l2067_206745


namespace Amith_current_age_l2067_206764

variable (A D : ℕ)

theorem Amith_current_age
  (h1 : A - 5 = 3 * (D - 5))
  (h2 : A + 10 = 2 * (D + 10)) :
  A = 50 := by
  sorry

end Amith_current_age_l2067_206764


namespace incenter_coordinates_l2067_206752

theorem incenter_coordinates (p q r : ℝ) (h₁ : p = 8) (h₂ : q = 6) (h₃ : r = 10) :
  ∃ x y z : ℝ, x + y + z = 1 ∧ x = p / (p + q + r) ∧ y = q / (p + q + r) ∧ z = r / (p + q + r) ∧
  x = 1 / 3 ∧ y = 1 / 4 ∧ z = 5 / 12 :=
by
  sorry

end incenter_coordinates_l2067_206752


namespace blue_eyed_among_blondes_l2067_206792

variable (l g b a : ℝ)

-- Given: The proportion of blondes among blue-eyed people is greater than the proportion of blondes among all people.
axiom given_condition : a / g > b / l

-- Prove: The proportion of blue-eyed people among blondes is greater than the proportion of blue-eyed people among all people.
theorem blue_eyed_among_blondes (l g b a : ℝ) (h : a / g > b / l) : a / b > g / l :=
by
  sorry

end blue_eyed_among_blondes_l2067_206792


namespace expected_profit_correct_l2067_206758

-- Define the conditions
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit calculation
def expected_profit : ℝ := (winning_probability * prize) - ticket_cost

-- The theorem we want to prove
theorem expected_profit_correct : expected_profit = -1.5 := by
  sorry

end expected_profit_correct_l2067_206758


namespace find_b_value_l2067_206723

theorem find_b_value (x : ℝ) (h_neg : x < 0) (h_eq : 1 / (x + 1 / (x + 2)) = 2) : 
  x + 7 / 2 = 2 :=
sorry

end find_b_value_l2067_206723


namespace gas_volume_at_25_degrees_l2067_206774

theorem gas_volume_at_25_degrees :
  (∀ (T V : ℕ), (T = 40 → V = 30) →
  (∀ (k : ℕ), T = 40 - 5 * k → V = 30 - 6 * k) → 
  (25 = 40 - 5 * 3) → 
  (V = 30 - 6 * 3) → 
  V = 12) := 
by
  sorry

end gas_volume_at_25_degrees_l2067_206774


namespace cuboid_layers_l2067_206737

theorem cuboid_layers (V : ℕ) (n_blocks : ℕ) (volume_per_block : ℕ) (blocks_per_layer : ℕ)
  (hV : V = 252) (hvol : volume_per_block = 1) (hblocks : n_blocks = V / volume_per_block) (hlayer : blocks_per_layer = 36) :
  (n_blocks / blocks_per_layer) = 7 :=
by
  sorry

end cuboid_layers_l2067_206737


namespace relationship_between_f_l2067_206740

-- Given definitions
def quadratic_parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def axis_of_symmetry (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y ↔ x + y = 2 * α

-- The problem statement to prove in Lean 4
theorem relationship_between_f (a b c x : ℝ) (hpos : x > 0) (apos : a > 0) :
  axis_of_symmetry (quadratic_parabola a b c) 1 →
  quadratic_parabola a b c (3^x) > quadratic_parabola a b c (2^x) :=
by
  sorry

end relationship_between_f_l2067_206740


namespace range_of_m_l2067_206722

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_incr : ∀ x y, x < y → f x < f y) : 
  f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := 
by 
  sorry

end range_of_m_l2067_206722


namespace sculpture_height_correct_l2067_206763

/-- Define the conditions --/
def base_height_in_inches : ℝ := 4
def total_height_in_feet : ℝ := 3.1666666666666665
def inches_per_foot : ℝ := 12

/-- Define the conversion from feet to inches for the total height --/
def total_height_in_inches : ℝ := total_height_in_feet * inches_per_foot

/-- Define the height of the sculpture in inches --/
def sculpture_height_in_inches : ℝ := total_height_in_inches - base_height_in_inches

/-- The proof problem in Lean 4 statement --/
theorem sculpture_height_correct :
  sculpture_height_in_inches = 34 := by
  sorry

end sculpture_height_correct_l2067_206763


namespace probability_within_three_units_from_origin_l2067_206702

-- Define the properties of the square Q is selected from
def isInSquare (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -2 ∧ Q.1 ≤ 2 ∧ Q.2 ≥ -2 ∧ Q.2 ≤ 2

-- Define the condition of being within 3 units from the origin
def withinThreeUnits (Q: ℝ × ℝ) : Prop :=
  (Q.1)^2 + (Q.2)^2 ≤ 9

-- State the problem: Proving the probability is 1
theorem probability_within_three_units_from_origin : 
  ∀ (Q : ℝ × ℝ), isInSquare Q → withinThreeUnits Q := 
by 
  sorry

end probability_within_three_units_from_origin_l2067_206702


namespace arithmetic_geometric_sequence_formula_l2067_206787

theorem arithmetic_geometric_sequence_formula :
  ∃ (a d : ℝ), (3 * a = 6) ∧
  ((5 - d) * (15 + d) = 64) ∧
  (∀ (n : ℕ), n ≥ 3 → (∃ (b_n : ℝ), b_n = 2 ^ (n - 1))) :=
by
  sorry

end arithmetic_geometric_sequence_formula_l2067_206787


namespace largest_of_given_numbers_l2067_206733

theorem largest_of_given_numbers :
  (0.99 > 0.9099) ∧
  (0.99 > 0.9) ∧
  (0.99 > 0.909) ∧
  (0.99 > 0.9009) →
  ∀ (x : ℝ), (x = 0.99 ∨ x = 0.9099 ∨ x = 0.9 ∨ x = 0.909 ∨ x = 0.9009) → 
  x ≤ 0.99 :=
by
  sorry

end largest_of_given_numbers_l2067_206733


namespace range_of_absolute_difference_l2067_206757

theorem range_of_absolute_difference : (∃ x : ℝ, y = |x + 4| - |x - 5|) → y ∈ [-9, 9] :=
sorry

end range_of_absolute_difference_l2067_206757


namespace music_tool_cost_l2067_206713

noncomputable def flute_cost : ℝ := 142.46
noncomputable def song_book_cost : ℝ := 7
noncomputable def total_spent : ℝ := 158.35

theorem music_tool_cost :
    total_spent - (flute_cost + song_book_cost) = 8.89 :=
by
  sorry

end music_tool_cost_l2067_206713


namespace distance_between_A_B_is_16_l2067_206761

-- The given conditions are translated as definitions
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- The theorem stating the proof problem
theorem distance_between_A_B_is_16 :
  let A : ℝ × ℝ := (4, 8)
  let B : ℝ × ℝ := (4, -8)
  let d : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  d = 16 :=
by
  sorry

end distance_between_A_B_is_16_l2067_206761


namespace arithmetic_geometric_sequences_l2067_206721

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end arithmetic_geometric_sequences_l2067_206721


namespace plumber_total_cost_l2067_206783

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end plumber_total_cost_l2067_206783


namespace inequality_neg_3_l2067_206709

theorem inequality_neg_3 (a b : ℝ) : a < b → -3 * a > -3 * b :=
by
  sorry

end inequality_neg_3_l2067_206709


namespace half_day_division_l2067_206736

theorem half_day_division : 
  ∃ (n m : ℕ), n * m = 43200 ∧ (∃! (k : ℕ), k = 60) := sorry

end half_day_division_l2067_206736


namespace polynomial_remainder_division_l2067_206701

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (x^4 + 2 * x^2 - 3) % (x^2 + 3 * x + 2) = -21 * x - 21 := 
by
  sorry

end polynomial_remainder_division_l2067_206701


namespace petals_vs_wings_and_unvisited_leaves_l2067_206750

def flowers_petals_leaves := 5
def petals_per_flower := 2
def bees_wings := 3
def wings_per_bee := 4
def leaves_per_flower := 3
def visits_per_bee := 2
def total_flowers := flowers_petals_leaves
def total_bees := bees_wings

def total_petals : ℕ := total_flowers * petals_per_flower
def total_wings : ℕ := total_bees * wings_per_bee
def more_wings_than_petals := total_wings - total_petals

def total_leaves : ℕ := total_flowers * leaves_per_flower
def total_visits : ℕ := total_bees * visits_per_bee
def leaves_per_visit := leaves_per_flower
def visited_leaves : ℕ := min total_leaves (total_visits * leaves_per_visit)
def unvisited_leaves : ℕ := total_leaves - visited_leaves

theorem petals_vs_wings_and_unvisited_leaves :
  more_wings_than_petals = 2 ∧ unvisited_leaves = 0 :=
by
  sorry

end petals_vs_wings_and_unvisited_leaves_l2067_206750


namespace apples_in_basket_l2067_206779

-- Define the conditions in Lean
def four_times_as_many_apples (O A : ℕ) : Prop :=
  A = 4 * O

def emiliano_consumes (O A : ℕ) : Prop :=
  (2/3 : ℚ) * O + (2/3 : ℚ) * A = 50

-- Formulate the main proposition to prove there are 60 apples
theorem apples_in_basket (O A : ℕ) (h1 : four_times_as_many_apples O A) (h2 : emiliano_consumes O A) : A = 60 := 
by
  sorry

end apples_in_basket_l2067_206779


namespace repeating_six_as_fraction_l2067_206775

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l2067_206775


namespace find_b_value_l2067_206719

theorem find_b_value 
  (point1 : ℝ × ℝ) (point2 : ℝ × ℝ) (b : ℝ) 
  (h1 : point1 = (0, -2))
  (h2 : point2 = (1, 0))
  (h3 : (∃ m c, ∀ x y, y = m * x + c ↔ (x, y) = point1 ∨ (x, y) = point2))
  (h4 : ∀ x y, y = 2 * x - 2 → (x, y) = (7, b)) :
  b = 12 :=
sorry

end find_b_value_l2067_206719


namespace seashells_calculation_l2067_206708

theorem seashells_calculation :
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  leigh_seashells = 16 :=
by
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  show leigh_seashells = 16
  sorry

end seashells_calculation_l2067_206708


namespace inversely_proportional_x_y_l2067_206714

theorem inversely_proportional_x_y {x y k : ℝ}
    (h_inv_proportional : x * y = k)
    (h_k : k = 75)
    (h_y : y = 45) :
    x = 5 / 3 :=
by
  sorry

end inversely_proportional_x_y_l2067_206714


namespace freds_sister_borrowed_3_dimes_l2067_206728

-- Define the conditions
def original_dimes := 7
def remaining_dimes := 4

-- Define the question and answer
def borrowed_dimes := original_dimes - remaining_dimes

-- Statement to prove
theorem freds_sister_borrowed_3_dimes : borrowed_dimes = 3 := by
  sorry

end freds_sister_borrowed_3_dimes_l2067_206728


namespace jason_pokemon_cards_l2067_206780

-- Conditions
def initial_cards : ℕ := 13
def cards_given : ℕ := 9

-- Proof Statement
theorem jason_pokemon_cards (initial_cards cards_given : ℕ) : initial_cards - cards_given = 4 :=
by
  sorry

end jason_pokemon_cards_l2067_206780


namespace arithmetic_sequence_solution_l2067_206799

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
    (h1 : q > 0)
    (h2 : 2 * a 3 = a 5 - 3 * a 4) 
    (h3 : a 2 * a 4 * a 6 = 64) 
    (h4 : ∀ n, S_n n = (1 - q^n) / (1 - q) * a 1) :
    q = 2 ∧ (∀ n, S_n n = (2^n - 1) / 2) := 
  by
  sorry

end arithmetic_sequence_solution_l2067_206799


namespace find_valid_pairs_l2067_206727

theorem find_valid_pairs (x y : ℤ) : 
  (x^3 + y) % (x^2 + y^2) = 0 ∧ 
  (x + y^3) % (x^2 + y^2) = 0 ↔ 
  (x, y) = (1, 1) ∨ (x, y) = (1, 0) ∨ (x, y) = (1, -1) ∨ 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (-1, 1) ∨ 
  (x, y) = (-1, 0) ∨ (x, y) = (-1, -1) :=
sorry

end find_valid_pairs_l2067_206727


namespace bank_check_problem_l2067_206717

theorem bank_check_problem :
  ∃ (x y : ℕ), (0 ≤ y ∧ y ≤ 99) ∧ (y + (x : ℚ) / 100 - 0.05 = 2 * (x + (y : ℚ) / 100)) ∧ x = 31 ∧ y = 63 :=
by
  -- Definitions and Conditions
  sorry

end bank_check_problem_l2067_206717


namespace tangent_circles_locus_l2067_206749

noncomputable def locus_condition (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

theorem tangent_circles_locus (a b : ℝ) (r : ℝ) (h : locus_condition a b r) :
  a^2 + 7 * b^2 - 34 * a - 57 = 0 :=
sorry

end tangent_circles_locus_l2067_206749


namespace number_of_positive_integer_pairs_l2067_206718

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l2067_206718


namespace interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l2067_206773

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem interval_of_increase (a : ℝ) : 
  (∀ x : ℝ, 0 < a → (Real.exp x - a ≥ 0 ↔ x ≥ Real.log a)) ∧ 
  (∀ x : ℝ, a ≤ 0 → (Real.exp x - a ≥ 0)) :=
by sorry

theorem monotone_increasing (a : ℝ) (h : ∀ x : ℝ, Real.exp x - a ≥ 0) : 
  a ≤ 0 :=
by sorry

theorem monotonically_increasing_decreasing : 
  ∃ a : ℝ, (∀ x ≤ 0, Real.exp x - a ≤ 0) ∧ 
           (∀ x ≥ 0, Real.exp x - a ≥ 0) ↔ a = 1 :=
by sorry

end interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l2067_206773


namespace shaded_region_area_l2067_206703

-- Definitions based on given conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def squares_in_large_square : ℕ := 16

-- The area of the entire shaded region
def area_of_shaded_region : ℝ := 78.125

-- Theorem to prove 
theorem shaded_region_area 
  (num_squares : ℕ) 
  (diagonal_length : ℝ) 
  (squares_in_large_square : ℕ) : 
  (num_squares = 25) → 
  (diagonal_length = 10) → 
  (squares_in_large_square = 16) → 
  area_of_shaded_region = 78.125 := 
by {
  sorry -- proof to be filled
}

end shaded_region_area_l2067_206703


namespace difference_abs_eq_200_l2067_206756

theorem difference_abs_eq_200 (x y : ℤ) (h1 : x + y = 250) (h2 : y = 225) : |x - y| = 200 := sorry

end difference_abs_eq_200_l2067_206756


namespace average_speed_first_girl_l2067_206711

theorem average_speed_first_girl (v : ℝ) 
  (start_same_point : True)
  (opp_directions : True)
  (avg_speed_second_girl : ℝ := 3)
  (distance_after_12_hours : (v + avg_speed_second_girl) * 12 = 120) :
  v = 7 :=
by
  sorry

end average_speed_first_girl_l2067_206711


namespace part_a_part_b_l2067_206786

-- Part (a): Proving that 91 divides n^37 - n for all integers n
theorem part_a (n : ℤ) : 91 ∣ (n ^ 37 - n) := 
sorry

-- Part (b): Finding the largest k that divides n^37 - n for all integers n is 3276
theorem part_b (n : ℤ) : ∀ k : ℤ, (k > 0) → (∀ n : ℤ, k ∣ (n ^ 37 - n)) → k ≤ 3276 :=
sorry

end part_a_part_b_l2067_206786


namespace find_percentage_l2067_206748

variable (P : ℝ)
variable (num : ℝ := 70)
variable (result : ℝ := 25)

theorem find_percentage (h : ((P / 100) * num) - 10 = result) : P = 50 := by
  sorry

end find_percentage_l2067_206748


namespace min_value_l2067_206788

theorem min_value (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by sorry

end min_value_l2067_206788


namespace bridge_length_l2067_206730

/-- The length of the bridge that a train 110 meters long and traveling at 45 km/hr can cross in 30 seconds is 265 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (cross_time_sec : ℝ) (bridge_length : ℝ) :
  train_length = 110 ∧ train_speed_kmh = 45 ∧ cross_time_sec = 30 ∧ bridge_length = 265 → 
  (train_speed_kmh * (1000 / 3600) * cross_time_sec - train_length = bridge_length) :=
by
  sorry

end bridge_length_l2067_206730


namespace sunzi_wood_problem_l2067_206741

theorem sunzi_wood_problem (x : ℝ) :
  (∃ (length_of_rope : ℝ), length_of_rope = x + 4.5 ∧
    ∃ (half_length_of_rope : ℝ), half_length_of_rope = length_of_rope / 2 ∧ 
      (half_length_of_rope + 1 = x)) ↔ 
  (1 / 2 * (x + 4.5) = x - 1) :=
by
  sorry

end sunzi_wood_problem_l2067_206741


namespace joe_eggs_club_house_l2067_206735

theorem joe_eggs_club_house (C : ℕ) (h : C + 5 + 3 = 20) : C = 12 :=
by 
  sorry

end joe_eggs_club_house_l2067_206735


namespace arithmetic_sequence_c_d_sum_l2067_206738

theorem arithmetic_sequence_c_d_sum :
  let c := 19 + (11 - 3)
  let d := c + (11 - 3)
  c + d = 62 :=
by
  sorry

end arithmetic_sequence_c_d_sum_l2067_206738


namespace min_value_x_add_2y_l2067_206771

theorem min_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end min_value_x_add_2y_l2067_206771


namespace triangle_is_right_triangle_l2067_206744

theorem triangle_is_right_triangle (A B C : ℝ) (hC_eq_A_plus_B : C = A + B) (h_angle_sum : A + B + C = 180) : C = 90 :=
by
  sorry

end triangle_is_right_triangle_l2067_206744


namespace binom_1294_2_l2067_206794

def combination (n k : Nat) := n.choose k

theorem binom_1294_2 : combination 1294 2 = 836161 := by
  sorry

end binom_1294_2_l2067_206794


namespace total_length_of_sticks_l2067_206781

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end total_length_of_sticks_l2067_206781


namespace prove_correct_option_C_l2067_206790

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l2067_206790


namespace largest_reciprocal_l2067_206734

-- Definitions of the given numbers
def num1 := 1 / 6
def num2 := 2 / 7
def num3 := (2 : ℝ)
def num4 := (8 : ℝ)
def num5 := (1000 : ℝ)

-- The main problem: prove that the reciprocal of 1/6 is the largest
theorem largest_reciprocal :
  (1 / num1 > 1 / num2) ∧ (1 / num1 > 1 / num3) ∧ (1 / num1 > 1 / num4) ∧ (1 / num1 > 1 / num5) :=
by
  sorry

end largest_reciprocal_l2067_206734


namespace difference_between_number_and_its_3_5_l2067_206754

theorem difference_between_number_and_its_3_5 (x : ℕ) (h : x = 155) :
  x - (3 / 5 : ℚ) * x = 62 := by
  sorry

end difference_between_number_and_its_3_5_l2067_206754


namespace angle_MON_l2067_206769

theorem angle_MON (O M N : ℝ × ℝ) (D : ℝ) :
  (O = (0, 0)) →
  (M = (-2, 2)) →
  (N = (2, 2)) →
  (x^2 + y^2 + D * x - 4 * y = 0) →
  (D = 0) →
  ∃ θ : ℝ, θ = 90 :=
by
  sorry

end angle_MON_l2067_206769


namespace birgit_hiking_time_l2067_206766

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end birgit_hiking_time_l2067_206766


namespace lemuel_total_points_l2067_206784

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end lemuel_total_points_l2067_206784


namespace geometric_sequence_fourth_term_l2067_206759

theorem geometric_sequence_fourth_term (x : ℝ) (h1 : (2 * x + 2) ^ 2 = x * (3 * x + 3))
  (h2 : x ≠ -1) : (3*x + 3) * (3/2) = -27/2 :=
by
  sorry

end geometric_sequence_fourth_term_l2067_206759


namespace total_settings_weight_l2067_206726

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l2067_206726


namespace living_room_area_l2067_206706

-- Define the conditions
def carpet_area (length width : ℕ) : ℕ :=
  length * width

def percentage_coverage (carpet_area living_room_area : ℕ) : ℕ :=
  (carpet_area * 100) / living_room_area

-- State the problem
theorem living_room_area (A : ℕ) (carpet_len carpet_wid : ℕ) (carpet_coverage : ℕ) :
  carpet_len = 4 → carpet_wid = 9 → carpet_coverage = 20 →
  20 * A = 36 * 100 → A = 180 :=
by
  intros h_len h_wid h_coverage h_proportion
  sorry

end living_room_area_l2067_206706


namespace ian_lottery_win_l2067_206751

theorem ian_lottery_win 
  (amount_paid_to_colin : ℕ)
  (amount_left : ℕ)
  (amount_paid_to_helen : ℕ := 2 * amount_paid_to_colin)
  (amount_paid_to_benedict : ℕ := amount_paid_to_helen / 2)
  (total_debts_paid : ℕ := amount_paid_to_colin + amount_paid_to_helen + amount_paid_to_benedict)
  (total_money_won : ℕ := total_debts_paid + amount_left)
  (h1 : amount_paid_to_colin = 20)
  (h2 : amount_left = 20) :
  total_money_won = 100 := 
sorry

end ian_lottery_win_l2067_206751


namespace race_course_length_l2067_206760

variable (v d : ℝ)

theorem race_course_length (h1 : 4 * v > 0) (h2 : ∀ t : ℝ, t > 0 → (d / (4 * v)) = ((d - 72) / v)) : d = 96 := by
  sorry

end race_course_length_l2067_206760


namespace area_of_defined_region_l2067_206707

theorem area_of_defined_region : 
  ∃ (A : ℝ), (∀ x y : ℝ, |4 * x - 20| + |3 * y + 9| ≤ 6 → A = 9) :=
sorry

end area_of_defined_region_l2067_206707


namespace johns_money_left_l2067_206746

def dog_walking_days_in_april := 26
def earnings_per_day := 10
def money_spent_on_books := 50
def money_given_to_sister := 50

theorem johns_money_left : (dog_walking_days_in_april * earnings_per_day) - (money_spent_on_books + money_given_to_sister) = 160 := 
by
  sorry

end johns_money_left_l2067_206746


namespace xy_square_sum_l2067_206743

theorem xy_square_sum (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 132) : x^2 + y^2 = 1336 :=
by
  sorry

end xy_square_sum_l2067_206743


namespace not_eq_positive_integers_l2067_206767

theorem not_eq_positive_integers (a b : ℤ) (ha : a > 0) (hb : b > 0) : 
  a^3 + (a + b)^2 + b ≠ b^3 + a + 2 :=
by {
  sorry
}

end not_eq_positive_integers_l2067_206767


namespace calc_pow_product_l2067_206716

theorem calc_pow_product : (0.25 ^ 2023) * (4 ^ 2023) = 1 := 
  by 
  sorry

end calc_pow_product_l2067_206716


namespace sin_add_cos_l2067_206724

theorem sin_add_cos (s72 c18 c72 s18 : ℝ) (h1 : s72 = Real.sin (72 * Real.pi / 180)) (h2 : c18 = Real.cos (18 * Real.pi / 180)) (h3 : c72 = Real.cos (72 * Real.pi / 180)) (h4 : s18 = Real.sin (18 * Real.pi / 180)) :
  s72 * c18 + c72 * s18 = 1 :=
by 
  sorry

end sin_add_cos_l2067_206724


namespace line_equation_passes_through_and_has_normal_l2067_206768

theorem line_equation_passes_through_and_has_normal (x y : ℝ) 
    (H1 : ∃ l : ℝ → ℝ, l 3 = 4)
    (H2 : ∃ n : ℝ × ℝ, n = (1, 2)) : 
    x + 2 * y - 11 = 0 :=
sorry

end line_equation_passes_through_and_has_normal_l2067_206768


namespace brick_width_l2067_206712

theorem brick_width (length_courtyard : ℕ) (width_courtyard : ℕ) (num_bricks : ℕ) (brick_length : ℕ) (total_area : ℕ) (brick_area : ℕ) (w : ℕ)
  (h1 : length_courtyard = 1800)
  (h2 : width_courtyard = 1200)
  (h3 : num_bricks = 30000)
  (h4 : brick_length = 12)
  (h5 : total_area = length_courtyard * width_courtyard)
  (h6 : total_area = num_bricks * brick_area)
  (h7 : brick_area = brick_length * w) :
  w = 6 :=
by
  sorry

end brick_width_l2067_206712


namespace circle_radius_of_equal_area_l2067_206729

theorem circle_radius_of_equal_area (A B C D : Type) (r : ℝ) (π : ℝ) 
  (h_rect_area : 8 * 9 = 72)
  (h_circle_area : π * r ^ 2 = 36) :
  r = 6 / Real.sqrt π :=
by
  sorry

end circle_radius_of_equal_area_l2067_206729


namespace isabella_hair_length_l2067_206747

theorem isabella_hair_length (original : ℝ) (increase_percent : ℝ) (new_length : ℝ) 
    (h1 : original = 18) (h2 : increase_percent = 0.75) 
    (h3 : new_length = original + increase_percent * original) : 
    new_length = 31.5 := by sorry

end isabella_hair_length_l2067_206747


namespace no_integer_solutions_l2067_206753

theorem no_integer_solutions (x y : ℤ) : ¬ (3 * x^2 + 2 = y^2) :=
sorry

end no_integer_solutions_l2067_206753


namespace largest_value_p_l2067_206795

theorem largest_value_p 
  (p q r : ℝ) 
  (h1 : p + q + r = 10) 
  (h2 : p * q + p * r + q * r = 25) :
  p ≤ 20 / 3 :=
sorry

end largest_value_p_l2067_206795


namespace no_solution_m1_no_solution_m2_solution_m3_l2067_206705

-- Problem 1: No positive integer solutions for m = 1
theorem no_solution_m1 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ x * y * z := sorry

-- Problem 2: No positive integer solutions for m = 2
theorem no_solution_m2 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ 2 * x * y * z := sorry

-- Problem 3: Only solutions for m = 3 are x = y = z = k for some k
theorem solution_m3 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z := sorry

end no_solution_m1_no_solution_m2_solution_m3_l2067_206705


namespace ticket_price_l2067_206715

theorem ticket_price (P : ℝ) (h_capacity : 50 * P - 24 * P = 208) :
  P = 8 :=
sorry

end ticket_price_l2067_206715


namespace trapezoid_area_l2067_206710

/-- Given that the area of the outer square is 36 square units and the area of the inner square is 
4 square units, the area of one of the four congruent trapezoids formed between the squares is 8 
square units. -/
theorem trapezoid_area (outer_square_area inner_square_area : ℕ) 
  (h_outer : outer_square_area = 36)
  (h_inner : inner_square_area = 4) : 
  (outer_square_area - inner_square_area) / 4 = 8 :=
by sorry

end trapezoid_area_l2067_206710


namespace car_speed_l2067_206755

theorem car_speed {vp vc : ℚ} (h1 : vp = 7 / 2) (h2 : vc = 6 * vp) : 
  vc = 21 := 
by 
  sorry

end car_speed_l2067_206755


namespace sum_of_roots_eq_six_l2067_206732

variable (a b : ℝ)

theorem sum_of_roots_eq_six (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_roots_eq_six_l2067_206732


namespace bus_driver_regular_rate_l2067_206731

theorem bus_driver_regular_rate (R : ℝ) (h1 : 976 = (40 * R) + (14.32 * (1.75 * R))) : 
  R = 15 := 
by
  sorry

end bus_driver_regular_rate_l2067_206731


namespace parabola_focus_l2067_206793

theorem parabola_focus (a b c : ℝ) (h_eq : ∀ x : ℝ, 2 * x^2 + 8 * x - 1 = a * (x + b)^2 + c) :
  ∃ focus : ℝ × ℝ, focus = (-2, -71 / 8) :=
sorry

end parabola_focus_l2067_206793


namespace problem_distribution_count_l2067_206725

theorem problem_distribution_count : 12^6 = 2985984 := 
by
  sorry

end problem_distribution_count_l2067_206725


namespace union_complements_l2067_206777

open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Define the conditions
def condition_U : U = {1, 2, 3, 4, 5} := by
  sorry

def condition_A : A = {1, 2, 3} := by
  sorry

def condition_B : B = {2, 3, 4} := by
  sorry

-- Prove that (complement_U A) ∪ (complement_U B) = {1, 4, 5}
theorem union_complements :
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by
  sorry

end union_complements_l2067_206777


namespace simplify_division_l2067_206772

theorem simplify_division :
  (2 * 10^12) / (4 * 10^5 - 1 * 10^4) = 5.1282 * 10^6 :=
by
  -- problem statement
  sorry

end simplify_division_l2067_206772


namespace polygonal_pyramid_faces_l2067_206778

/-- A polygonal pyramid is a three-dimensional solid. Its base is a regular polygon. Each of the vertices of the polygonal base is connected to a single point, called the apex. The sum of the number of edges and the number of vertices of a particular polygonal pyramid is 1915. This theorem states that the number of faces of this pyramid is 639. -/
theorem polygonal_pyramid_faces (n : ℕ) (hn : 2 * n + (n + 1) = 1915) : n + 1 = 639 :=
by
  sorry

end polygonal_pyramid_faces_l2067_206778
