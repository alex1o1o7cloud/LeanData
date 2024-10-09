import Mathlib

namespace round_robin_cycles_l419_41923

-- Define the conditions
def teams : ℕ := 28
def wins_per_team : ℕ := 13
def losses_per_team : ℕ := 13
def total_teams_games := teams * (teams - 1) / 2
def sets_of_three_teams := (teams * (teams - 1) * (teams - 2)) / 6

-- Define the problem statement
theorem round_robin_cycles :
  -- We need to show that the number of sets of three teams {A, B, C} where A beats B, B beats C, and C beats A is 1092
  (sets_of_three_teams - (teams * (wins_per_team * (wins_per_team - 1)) / 2)) = 1092 :=
by
  sorry

end round_robin_cycles_l419_41923


namespace solve_prime_equation_l419_41953

theorem solve_prime_equation (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  x^3 + y^3 - 3 * x * y = p - 1 ↔
  (x = 1 ∧ y = 0 ∧ p = 2) ∨
  (x = 0 ∧ y = 1 ∧ p = 2) ∨
  (x = 2 ∧ y = 2 ∧ p = 5) := 
sorry

end solve_prime_equation_l419_41953


namespace find_third_side_of_triangle_l419_41956

theorem find_third_side_of_triangle (a b : ℝ) (A : ℝ) (h1 : a = 6) (h2 : b = 10) (h3 : A = 18) (h4 : ∃ C, 0 < C ∧ C < π / 2 ∧ A = 0.5 * a * b * Real.sin C) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 22 :=
by
  sorry

end find_third_side_of_triangle_l419_41956


namespace gloves_selection_l419_41967

theorem gloves_selection (total_pairs : ℕ) (total_gloves : ℕ) (num_to_select : ℕ) 
    (total_ways : ℕ) (no_pair_ways : ℕ) : 
    total_pairs = 4 → 
    total_gloves = 8 → 
    num_to_select = 4 → 
    total_ways = (Nat.choose total_gloves num_to_select) → 
    no_pair_ways = 2^total_pairs → 
    (total_ways - no_pair_ways) = 54 :=
by
  intros
  sorry

end gloves_selection_l419_41967


namespace value_of_x_plus_y_l419_41983

theorem value_of_x_plus_y (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l419_41983


namespace exists_unique_i_l419_41951

theorem exists_unique_i (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) 
  (a : ℤ) (ha1 : 2 ≤ a) (ha2 : a ≤ p - 2) : 
  ∃! (i : ℤ), 2 ≤ i ∧ i ≤ p - 2 ∧ (i * a) % p = 1 ∧ Nat.gcd (i.natAbs) (a.natAbs) = 1 :=
sorry

end exists_unique_i_l419_41951


namespace batsman_average_after_17th_inning_l419_41922

theorem batsman_average_after_17th_inning
  (A : ℕ)  -- average after the 16th inning
  (h1 : 16 * A + 300 = 17 * (A + 10)) :
  A + 10 = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l419_41922


namespace find_fraction_l419_41950

noncomputable def distinct_real_numbers (a b : ℝ) : Prop :=
  a ≠ b

noncomputable def equation_condition (a b : ℝ) : Prop :=
  (2 * a / (3 * b)) + ((a + 12 * b) / (3 * b + 12 * a)) = (5 / 3)

theorem find_fraction (a b : ℝ) (h1 : distinct_real_numbers a b) (h2 : equation_condition a b) : a / b = -93 / 49 :=
by
  sorry

end find_fraction_l419_41950


namespace average_speed_of_the_car_l419_41916

noncomputable def averageSpeed (d1 d2 d3 d4 t1 t2 t3 t4 : ℝ) : ℝ :=
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := t1 + t2 + t3 + t4
  totalDistance / totalTime

theorem average_speed_of_the_car :
  averageSpeed 30 35 65 (40 * 0.5) (30 / 45) (35 / 55) 1 0.5 = 54 := 
  by 
    sorry

end average_speed_of_the_car_l419_41916


namespace sqrt_diff_inequality_l419_41904

open Real

theorem sqrt_diff_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt a - sqrt (a - 1) < sqrt (a - 2) - sqrt (a - 3) :=
sorry

end sqrt_diff_inequality_l419_41904


namespace license_plate_count_correct_l419_41930

def rotokas_letters : Finset Char := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U'}

def valid_license_plate_count : ℕ :=
  let first_letter_choices := 2 -- Letters A or E
  let last_letter_fixed := 1 -- Fixed as P
  let remaining_letters := rotokas_letters.erase 'V' -- Exclude V
  let second_letter_choices := (remaining_letters.erase 'P').card - 1 -- Exclude P and first letter
  let third_letter_choices := second_letter_choices - 1
  let fourth_letter_choices := third_letter_choices - 1
  2 * 9 * 8 * 7

theorem license_plate_count_correct :
  valid_license_plate_count = 1008 := by
  sorry

end license_plate_count_correct_l419_41930


namespace find_value_l419_41946

-- Given points A(a, 1), B(2, b), and C(3, 4).
variables (a b : ℝ)

-- Given condition from the problem
def condition : Prop := (3 * a + 4 = 6 + 4 * b)

-- The target is to find 3a - 4b
def target : ℝ := 3 * a - 4 * b

theorem find_value (h : condition a b) : target a b = 2 := 
by sorry

end find_value_l419_41946


namespace starting_number_l419_41925

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end starting_number_l419_41925


namespace paige_finished_problems_at_school_l419_41963

-- Definitions based on conditions
def math_problems : ℕ := 43
def science_problems : ℕ := 12
def total_problems : ℕ := math_problems + science_problems
def problems_left : ℕ := 11

-- The main theorem we need to prove
theorem paige_finished_problems_at_school : total_problems - problems_left = 44 := by
  sorry

end paige_finished_problems_at_school_l419_41963


namespace number_times_frac_eq_cube_l419_41943

theorem number_times_frac_eq_cube (x : ℕ) : x * (1/6)^2 = 6^3 → x = 7776 :=
by
  intro h
  -- skipped proof
  sorry

end number_times_frac_eq_cube_l419_41943


namespace abs_inequality_solution_set_l419_41901

theorem abs_inequality_solution_set (x : ℝ) : 
  (|2 * x - 3| ≤ 1) ↔ (1 ≤ x ∧ x ≤ 2) := 
by
  sorry

end abs_inequality_solution_set_l419_41901


namespace rotated_angle_l419_41914

theorem rotated_angle (initial_angle : ℝ) (rotation_angle : ℝ) (final_angle : ℝ) :
  initial_angle = 30 ∧ rotation_angle = 450 → final_angle = 60 :=
by
  intro h
  sorry

end rotated_angle_l419_41914


namespace moles_of_Cl2_l419_41921

def chemical_reaction : Prop :=
  ∀ (CH4 Cl2 HCl : ℕ), 
  (CH4 = 1) → 
  (HCl = 4) →
  -- Given the balanced equation: CH4 + 2Cl2 → CHCl3 + 4HCl
  (CH4 + 2 * Cl2 = CH4 + 2 * Cl2) →
  (4 * HCl = 4 * HCl) → -- This asserts the product side according to the balanced equation
  (Cl2 = 2)

theorem moles_of_Cl2 (CH4 Cl2 HCl : ℕ) (hCH4 : CH4 = 1) (hHCl : HCl = 4)
  (h_balanced : CH4 + 2 * Cl2 = CH4 + 2 * Cl2) (h_product : 4 * HCl = 4 * HCl) :
  Cl2 = 2 := by {
    sorry
}

end moles_of_Cl2_l419_41921


namespace cuboid_surface_area_500_l419_41954

def surface_area (w l h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

theorem cuboid_surface_area_500 :
  ∀ (w l h : ℝ), w = 4 → l = w + 6 → h = l + 5 →
  surface_area w l h = 500 :=
by
  intros w l h hw hl hh
  unfold surface_area
  rw [hw, hl, hh]
  norm_num
  sorry

end cuboid_surface_area_500_l419_41954


namespace value_subtracted_from_result_l419_41985

theorem value_subtracted_from_result (N V : ℕ) (hN : N = 1152) (h: (N / 6) - V = 3) : V = 189 :=
by
  sorry

end value_subtracted_from_result_l419_41985


namespace certain_number_l419_41952

theorem certain_number (G : ℕ) (N : ℕ) (H1 : G = 129) 
  (H2 : N % G = 9) (H3 : 2206 % G = 13) : N = 2202 :=
by
  sorry

end certain_number_l419_41952


namespace range_of_a_l419_41919

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - |x + 1| + 2 * a ≥ 0) ↔ a ∈ (Set.Ici ((Real.sqrt 3 + 1) / 4)) := by
  sorry

end range_of_a_l419_41919


namespace sector_angle_l419_41978

theorem sector_angle (r l : ℝ) (h1 : l + 2 * r = 6) (h2 : 1/2 * l * r = 2) : 
  l / r = 1 ∨ l / r = 4 := 
sorry

end sector_angle_l419_41978


namespace six_digit_squares_l419_41998

theorem six_digit_squares (x y : ℕ) 
  (h1 : y < 1000)
  (h2 : (1000 * x + y) < 1000000)
  (h3 : y * (y - 1) = 1000 * x)
  (mod8 : y * (y - 1) ≡ 0 [MOD 8])
  (mod125 : y * (y - 1) ≡ 0 [MOD 125]) :
  (1000 * x + y = 390625 ∨ 1000 * x + y = 141376) :=
sorry

end six_digit_squares_l419_41998


namespace associate_professors_bring_2_pencils_l419_41981

theorem associate_professors_bring_2_pencils (A B P : ℕ) 
  (h1 : A + B = 5)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 5)
  : P = 2 :=
by {
  -- Proof goes here
  sorry
}

end associate_professors_bring_2_pencils_l419_41981


namespace number_of_permissible_sandwiches_l419_41958

theorem number_of_permissible_sandwiches (b m c : ℕ) (h : b = 5) (me : m = 7) (ch : c = 6) 
  (no_ham_cheddar : ∀ bread, ¬(bread = ham ∧ cheese = cheddar))
  (no_turkey_swiss : ∀ bread, ¬(bread = turkey ∧ cheese = swiss)) : 
  5 * 7 * 6 - (5 * 1 * 1) - (5 * 1 * 1) = 200 := 
by 
  sorry

end number_of_permissible_sandwiches_l419_41958


namespace possible_values_of_a2b_b2c_c2a_l419_41927

theorem possible_values_of_a2b_b2c_c2a (a b c : ℝ) (h : a + b + c = 1) : ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
by
  sorry

end possible_values_of_a2b_b2c_c2a_l419_41927


namespace xy_positive_l419_41915

theorem xy_positive (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 :=
sorry

end xy_positive_l419_41915


namespace find_b_value_l419_41984

theorem find_b_value (x y z : ℝ) (u t : ℕ) (h_pos_xyx : x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ t > 0)
  (h1 : (x + y - z) / z = 1) (h2 : (x - y + z) / y = 1) (h3 : (-x + y + z) / x = 1) 
  (ha : (x + y) * (y + z) * (z + x) / (x * y * z) = 8) (hu_t : u + t + u * t = 34) : (u + t = 10) :=
by
  sorry

end find_b_value_l419_41984


namespace greatest_of_six_consecutive_mixed_numbers_l419_41948

theorem greatest_of_six_consecutive_mixed_numbers (A : ℚ) :
  let B := A + 1
  let C := A + 2
  let D := A + 3
  let E := A + 4
  let F := A + 5
  (A + B + C + D + E + F = 75.5) →
  F = 15 + 1/12 :=
by {
  sorry
}

end greatest_of_six_consecutive_mixed_numbers_l419_41948


namespace tangency_splits_segments_l419_41908

def pentagon_lengths (a b c d e : ℕ) (h₁ : a = 1) (h₃ : c = 1) (x1 x2 : ℝ) :=
x1 + x2 = b ∧ x1 = 1/2 ∧ x2 = 1/2

theorem tangency_splits_segments {a b c d e : ℕ} (h₁ : a = 1) (h₃ : c = 1) :
    ∃ x1 x2 : ℝ, pentagon_lengths a b c d e h₁ h₃ x1 x2 :=
    by 
    sorry

end tangency_splits_segments_l419_41908


namespace tom_strokes_over_par_l419_41933

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l419_41933


namespace problem_statement_l419_41971

noncomputable def f (a x : ℝ) : ℝ := a^x + a^(-x)

theorem problem_statement (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a 1 = 3) :
  f a 0 + f a 1 + f a 2 = 12 :=
sorry

end problem_statement_l419_41971


namespace beth_students_proof_l419_41988

-- Let initial := 150
-- Let joined := 30
-- Let left := 15
-- final := initial + joined - left
-- Prove final = 165

def beth_final_year_students (initial joined left final : ℕ) : Prop :=
  initial = 150 ∧ joined = 30 ∧ left = 15 ∧ final = initial + joined - left

theorem beth_students_proof : ∃ final, beth_final_year_students 150 30 15 final ∧ final = 165 :=
by
  sorry

end beth_students_proof_l419_41988


namespace mala_usha_speed_ratio_l419_41992

noncomputable def drinking_speed_ratio (M U : ℝ) (tM tU : ℝ) (fracU : ℝ) (total_bottle : ℝ) : ℝ :=
  let U_speed := fracU * total_bottle / tU
  let M_speed := (total_bottle - fracU * total_bottle) / tM
  M_speed / U_speed

theorem mala_usha_speed_ratio :
  drinking_speed_ratio (3/50) (1/50) 10 20 (4/10) 1 = 3 :=
by
  sorry

end mala_usha_speed_ratio_l419_41992


namespace factorize_expression_l419_41965

variable (m n : ℝ)

theorem factorize_expression : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 :=
by
  sorry

end factorize_expression_l419_41965


namespace simplify_expression_l419_41990

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l419_41990


namespace range_of_x_in_function_l419_41939

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end range_of_x_in_function_l419_41939


namespace difference_longest_shortest_worm_l419_41944

theorem difference_longest_shortest_worm
  (A B C D E : ℝ)
  (hA : A = 0.8)
  (hB : B = 0.1)
  (hC : C = 1.2)
  (hD : D = 0.4)
  (hE : E = 0.7) :
  (max C (max A (max E (max D B))) - min B (min D (min E (min A C)))) = 1.1 :=
by
  sorry

end difference_longest_shortest_worm_l419_41944


namespace spacesMovedBeforeSetback_l419_41940

-- Let's define the conditions as local constants
def totalSpaces : ℕ := 48
def firstTurnMove : ℕ := 8
def thirdTurnMove : ℕ := 6
def remainingSpacesToWin : ℕ := 37
def setback : ℕ := 5

theorem spacesMovedBeforeSetback (x : ℕ) : 
  (firstTurnMove + thirdTurnMove) + x - setback + remainingSpacesToWin = totalSpaces →
  x = 28 := by
  sorry

end spacesMovedBeforeSetback_l419_41940


namespace right_triangle_unique_value_l419_41924

theorem right_triangle_unique_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
(h1 : a + b + c = (1/2) * a * b) (h2 : c^2 = a^2 + b^2) : a + b - c = 4 :=
by
  sorry

end right_triangle_unique_value_l419_41924


namespace percent_answered_second_correctly_l419_41995

theorem percent_answered_second_correctly
  (nA : ℝ) (nAB : ℝ) (n_neither : ℝ) :
  nA = 0.80 → nAB = 0.60 → n_neither = 0.05 → 
  (nA + nB - nAB + n_neither = 1) → 
  ((1 - n_neither) = nA + nB - nAB) → 
  nB = 0.75 :=
by
  intros h1 h2 h3 hUnion hInclusion
  sorry

end percent_answered_second_correctly_l419_41995


namespace circle_standard_equation_l419_41977

theorem circle_standard_equation (x y : ℝ) :
  let center_x := 2
  let center_y := -1
  let radius := 3
  (center_x = 2) ∧ (center_y = -1) ∧ (radius = 3) → (x - center_x) ^ 2 + (y - center_y) ^ 2 = radius ^ 2 :=
by
  intros
  sorry

end circle_standard_equation_l419_41977


namespace plants_per_row_l419_41906

theorem plants_per_row (P : ℕ) (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : yield_per_plant = 20)
  (h3 : total_yield = 6000)
  (h4 : rows * yield_per_plant * P = total_yield) : 
  P = 10 :=
by 
  sorry

end plants_per_row_l419_41906


namespace div_5_implies_one_div_5_l419_41937

theorem div_5_implies_one_div_5 (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by 
  sorry

end div_5_implies_one_div_5_l419_41937


namespace cat_food_weight_l419_41903

theorem cat_food_weight (x : ℝ) :
  let bags_of_cat_food := 2
  let bags_of_dog_food := 2
  let ounces_per_pound := 16
  let total_ounces_of_pet_food := 256
  let dog_food_extra_weight := 2
  (ounces_per_pound * (bags_of_cat_food * x + bags_of_dog_food * (x + dog_food_extra_weight))) = total_ounces_of_pet_food
  → x = 3 :=
by
  sorry

end cat_food_weight_l419_41903


namespace least_number_to_make_divisible_l419_41947

theorem least_number_to_make_divisible (k : ℕ) (h : 1202 + k = 1204) : (2 ∣ 1204) := 
by
  sorry

end least_number_to_make_divisible_l419_41947


namespace div_pow_eq_l419_41935

theorem div_pow_eq : 23^11 / 23^5 = 148035889 := by
  sorry

end div_pow_eq_l419_41935


namespace unbroken_seashells_l419_41966

theorem unbroken_seashells (total broken : ℕ) (h1 : total = 7) (h2 : broken = 4) : total - broken = 3 :=
by
  -- Proof goes here…
  sorry

end unbroken_seashells_l419_41966


namespace initial_population_l419_41949

theorem initial_population (P : ℝ)
  (h1 : P * 1.25 * 0.75 = 18750) : P = 20000 :=
sorry

end initial_population_l419_41949


namespace janeth_balloons_count_l419_41994

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end janeth_balloons_count_l419_41994


namespace vertex_of_parabola_l419_41964

theorem vertex_of_parabola (c d : ℝ) (h₁ : ∀ x, -x^2 + c*x + d ≤ 0 ↔ (x ≤ -1 ∨ x ≥ 7)) : 
  ∃ v : ℝ × ℝ, v = (3, 16) :=
by
  sorry

end vertex_of_parabola_l419_41964


namespace fraction_to_decimal_17_625_l419_41910

def fraction_to_decimal (num : ℕ) (den : ℕ) : ℚ := num / den

theorem fraction_to_decimal_17_625 : fraction_to_decimal 17 625 = 272 / 10000 := by
  sorry

end fraction_to_decimal_17_625_l419_41910


namespace quadratic_inequality_hold_l419_41987

theorem quadratic_inequality_hold (α : ℝ) (h : 0 ≤ α ∧ α ≤ π) :
    (∀ x : ℝ, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔ 
    (α ∈ Set.Icc 0 (π / 6) ∨ α ∈ Set.Icc (5 * π / 6) π) :=
sorry

end quadratic_inequality_hold_l419_41987


namespace calculate_present_worth_l419_41955

variable (BG : ℝ) (r : ℝ) (t : ℝ)

theorem calculate_present_worth (hBG : BG = 24) (hr : r = 0.10) (ht : t = 2) : 
  ∃ PW : ℝ, PW = 120 := 
by
  sorry

end calculate_present_worth_l419_41955


namespace walter_hushpuppies_per_guest_l419_41928

variables (guests hushpuppies_per_batch time_per_batch total_time : ℕ)

def batches (total_time time_per_batch : ℕ) : ℕ :=
  total_time / time_per_batch

def total_hushpuppies (batches hushpuppies_per_batch : ℕ) : ℕ :=
  batches * hushpuppies_per_batch

def hushpuppies_per_guest (total_hushpuppies guests : ℕ) : ℕ :=
  total_hushpuppies / guests

theorem walter_hushpuppies_per_guest :
  ∀ (guests hushpuppies_per_batch time_per_batch total_time : ℕ),
    guests = 20 →
    hushpuppies_per_batch = 10 →
    time_per_batch = 8 →
    total_time = 80 →
    hushpuppies_per_guest (total_hushpuppies (batches total_time time_per_batch) hushpuppies_per_batch) guests = 5 :=
by 
  intros _ _ _ _ h_guests h_hpb h_tpb h_tt
  sorry

end walter_hushpuppies_per_guest_l419_41928


namespace triangle_area_DEF_l419_41905

def point : Type := ℝ × ℝ

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

theorem triangle_area_DEF :
  let base : ℝ := abs (D.1 - E.1)
  let height : ℝ := abs (F.2 - 2)
  let area := 1/2 * base * height
  area = 30 := 
by 
  sorry

end triangle_area_DEF_l419_41905


namespace asymptotes_of_hyperbola_l419_41918

theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 25 = 1) →
    (y = (5 / 4) * x ∨ y = -(5 / 4) * x)) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l419_41918


namespace cone_base_radius_l419_41932

theorem cone_base_radius (angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) 
(h1 : angle = 216)
(h2 : sector_radius = 15)
(h3 : 2 * π * base_radius = (3 / 5) * 2 * π * sector_radius) :
base_radius = 9 := 
sorry

end cone_base_radius_l419_41932


namespace total_gulbis_l419_41959

theorem total_gulbis (dureums fish_per_dureum : ℕ) (h1 : dureums = 156) (h2 : fish_per_dureum = 20) : dureums * fish_per_dureum = 3120 :=
by
  sorry

end total_gulbis_l419_41959


namespace largest_common_term_l419_41945

theorem largest_common_term (b : ℕ) (h1 : b ≡ 1 [MOD 3]) (h2 : b ≡ 2 [MOD 10]) (h3 : b < 300) : b = 290 :=
sorry

end largest_common_term_l419_41945


namespace merchant_marked_price_l419_41982

variable (L C M S : ℝ)

-- Conditions
def condition1 : Prop := C = 0.7 * L
def condition2 : Prop := C = 0.7 * S
def condition3 : Prop := S = 0.8 * M

-- The main statement
theorem merchant_marked_price (h1 : condition1 L C) (h2 : condition2 C S) (h3 : condition3 S M) : M = 1.25 * L :=
by
  sorry

end merchant_marked_price_l419_41982


namespace smallest_benches_l419_41957

theorem smallest_benches (N : ℕ) (h1 : ∃ n, 8 * n = 40 ∧ 10 * n = 40) : N = 20 :=
sorry

end smallest_benches_l419_41957


namespace num_cubes_with_more_than_one_blue_face_l419_41929

-- Define the parameters of the problem
def block_length : ℕ := 5
def block_width : ℕ := 3
def block_height : ℕ := 1

def total_cubes : ℕ := 15
def corners : ℕ := 4
def edges : ℕ := 6
def middles : ℕ := 5

-- Define the condition that the total number of cubes painted on more than one face.
def cubes_more_than_one_blue_face : ℕ := corners + edges

-- Prove that the number of cubes painted on more than one face is 10
theorem num_cubes_with_more_than_one_blue_face :
  cubes_more_than_one_blue_face = 10 :=
by
  show (4 + 6) = 10
  sorry

end num_cubes_with_more_than_one_blue_face_l419_41929


namespace length_of_first_train_is_140_l419_41973

theorem length_of_first_train_is_140 
  (speed1 : ℝ) (speed2 : ℝ) (time_to_cross : ℝ) (length2 : ℝ) 
  (h1 : speed1 = 60) 
  (h2 : speed2 = 40) 
  (h3 : time_to_cross = 12.239020878329734) 
  (h4 : length2 = 200) : 
  ∃ (length1 : ℝ), length1 = 140 := 
by
  sorry

end length_of_first_train_is_140_l419_41973


namespace present_cost_after_discount_l419_41974

theorem present_cost_after_discount 
  (X : ℝ) (P : ℝ) 
  (h1 : X - 4 = (0.80 * P) / 3) 
  (h2 : P = 3 * X)
  :
  0.80 * P = 48 :=
by
  sorry

end present_cost_after_discount_l419_41974


namespace unique_prime_p_l419_41907

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 2)) : p = 3 := 
by 
  sorry

end unique_prime_p_l419_41907


namespace range_of_a_l419_41999

noncomputable def f (a x : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) → a ≤ 4 :=
by
  sorry

end range_of_a_l419_41999


namespace af_cd_ratio_l419_41926

theorem af_cd_ratio (a b c d e f : ℝ) 
  (h1 : a * b * c = 130) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 750) 
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 2 / 3 := 
by
  sorry

end af_cd_ratio_l419_41926


namespace correct_conclusions_l419_41934

theorem correct_conclusions :
  (∀ n : ℤ, n < -1 -> n < -1) ∧
  (¬ ∀ a : ℤ, abs (a + 2022) > 0) ∧
  (∀ a b : ℤ, a + b = 0 -> a * b < 0) ∧
  (∀ n : ℤ, abs n = n -> n ≥ 0) :=
sorry

end correct_conclusions_l419_41934


namespace vanilla_syrup_cost_l419_41980

theorem vanilla_syrup_cost :
  ∀ (unit_cost_drip : ℝ) (num_drip : ℕ)
    (unit_cost_espresso : ℝ) (num_espresso : ℕ)
    (unit_cost_latte : ℝ) (num_lattes : ℕ)
    (unit_cost_cold_brew : ℝ) (num_cold_brews : ℕ)
    (unit_cost_cappuccino : ℝ) (num_cappuccino : ℕ)
    (total_cost : ℝ) (vanilla_cost : ℝ),
  unit_cost_drip = 2.25 →
  num_drip = 2 →
  unit_cost_espresso = 3.50 →
  num_espresso = 1 →
  unit_cost_latte = 4.00 →
  num_lattes = 2 →
  unit_cost_cold_brew = 2.50 →
  num_cold_brews = 2 →
  unit_cost_cappuccino = 3.50 →
  num_cappuccino = 1 →
  total_cost = 25.00 →
  vanilla_cost =
    total_cost -
    ((unit_cost_drip * num_drip) +
    (unit_cost_espresso * num_espresso) +
    (unit_cost_latte * (num_lattes - 1)) +
    (unit_cost_cold_brew * num_cold_brews) +
    (unit_cost_cappuccino * num_cappuccino)) →
  vanilla_cost = 0.50 := sorry

end vanilla_syrup_cost_l419_41980


namespace negation_of_universal_proposition_l419_41997

noncomputable def f (n : Nat) : Set ℕ := sorry

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, f n ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n, m ≤ n) ↔
  ∃ n_0 : ℕ, f n_0 ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n_0, m ≤ n_0 :=
sorry

end negation_of_universal_proposition_l419_41997


namespace last_digit_of_7_to_the_7_l419_41913

theorem last_digit_of_7_to_the_7 :
  (7 ^ 7) % 10 = 3 :=
by
  sorry

end last_digit_of_7_to_the_7_l419_41913


namespace original_cost_of_dress_l419_41975

theorem original_cost_of_dress (x : ℝ) 
  (h1 : x / 2 - 10 < x)
  (h2 : x - (x / 2 - 10) = 80) : 
  x = 140 := 
sorry

end original_cost_of_dress_l419_41975


namespace solve_for_x_l419_41936

theorem solve_for_x (x : ℤ) (h : 3 * x - 5 = 4 * x + 10) : x = -15 :=
sorry

end solve_for_x_l419_41936


namespace sqrt_eq_sum_iff_l419_41996

open Real

theorem sqrt_eq_sum_iff (a b : ℝ) : sqrt (a^2 + b^2) = a + b ↔ (a * b = 0) ∧ (a + b ≥ 0) :=
by
  sorry

end sqrt_eq_sum_iff_l419_41996


namespace value_of_x_l419_41991

theorem value_of_x (z : ℕ) (y : ℕ) (x : ℕ) 
  (h₁ : y = z / 5)
  (h₂ : x = y / 2)
  (h₃ : z = 60) : 
  x = 6 :=
by
  sorry

end value_of_x_l419_41991


namespace expression_value_l419_41962

theorem expression_value : 
  (2 ^ 1501 + 5 ^ 1502) ^ 2 - (2 ^ 1501 - 5 ^ 1502) ^ 2 = 20 * 10 ^ 1501 := 
by
  sorry

end expression_value_l419_41962


namespace pete_backwards_speed_l419_41979

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end pete_backwards_speed_l419_41979


namespace discount_per_bear_l419_41993

/-- Suppose the price of the first bear is $4.00 and Wally pays $354.00 for 101 bears.
 Prove that the discount per bear after the first bear is $0.50. -/
theorem discount_per_bear 
  (price_first : ℝ) (total_bears : ℕ) (total_paid : ℝ) (price_rest_bears : ℝ )
  (h1 : price_first = 4.0) (h2 : total_bears = 101) (h3 : total_paid = 354.0) : 
  (price_first + (total_bears - 1) * price_rest_bears - total_paid) / (total_bears - 1) = 0.50 :=
sorry

end discount_per_bear_l419_41993


namespace length_of_solution_set_l419_41917

variable {a b : ℝ}

theorem length_of_solution_set (h : ∀ x : ℝ, a ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ b → 12 = (b - a) / 3) : b - a = 36 :=
sorry

end length_of_solution_set_l419_41917


namespace decrypt_encryption_l419_41970

-- Encryption function description
def encrypt_digit (d : ℕ) : ℕ := 10 - (d * 7 % 10)

def encrypt_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let encrypted_digits := digits.map encrypt_digit
  encrypted_digits.foldr (λ d acc => d + acc * 10) 0
  
noncomputable def digit_match (d: ℕ) : ℕ :=
  match d with
  | 0 => 0 | 1 => 3 | 2 => 8 | 3 => 1 | 4 => 6 | 5 => 5
  | 6 => 8 | 7 => 1 | 8 => 4 | 9 => 7 | _ => 0

theorem decrypt_encryption:
encrypt_number 891134 = 473392 :=
by
  sorry

end decrypt_encryption_l419_41970


namespace green_pill_cost_l419_41960

variable (x : ℝ) -- cost of a green pill in dollars
variable (y : ℝ) -- cost of a pink pill in dollars
variable (total_cost : ℝ) -- total cost for 21 days

theorem green_pill_cost
  (h1 : x = y + 2) -- a green pill costs $2 more than a pink pill
  (h2 : total_cost = 819) -- total cost for 21 days is $819
  (h3 : ∀ n, n = 21 ∧ total_cost / n = (x + y)) :
  x = 20.5 :=
by
  sorry

end green_pill_cost_l419_41960


namespace finalCostCalculation_l419_41941

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end finalCostCalculation_l419_41941


namespace percent_of_g_is_a_l419_41902

theorem percent_of_g_is_a (a b c d e f g : ℤ) (h1 : (a + b + c + d + e + f + g) / 7 = 9)
: (a / g) * 100 = 50 := 
sorry

end percent_of_g_is_a_l419_41902


namespace smallest_zarks_l419_41931

theorem smallest_zarks (n : ℕ) : (n^2 > 15 * n) → (n ≥ 16) := sorry

end smallest_zarks_l419_41931


namespace find_first_offset_l419_41986

theorem find_first_offset 
  (diagonal : ℝ) (second_offset : ℝ) (area : ℝ) (first_offset : ℝ)
  (h_diagonal : diagonal = 20)
  (h_second_offset : second_offset = 4)
  (h_area : area = 90)
  (h_area_formula : area = (diagonal * (first_offset + second_offset)) / 2) :
  first_offset = 5 :=
by 
  rw [h_diagonal, h_second_offset, h_area] at h_area_formula 
  -- This would be the place where you handle solving the formula using the given conditions
  sorry

end find_first_offset_l419_41986


namespace smallest_a1_value_l419_41961

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then 29 / 98 else if n > 0 then 15 * a_seq (n - 1) - 2 * n else 0

theorem smallest_a1_value :
  (∃ f : ℕ → ℝ, (∀ n > 0, f n = 15 * f (n - 1) - 2 * n) ∧ (∀ n, f n > 0) ∧ (f 1 = 29 / 98)) :=
sorry

end smallest_a1_value_l419_41961


namespace find_m_value_l419_41920

noncomputable def pyramid_property (m : ℕ) : Prop :=
  let n1 := 3
  let n2 := 9
  let n3 := 6
  let r2_1 := m + n1
  let r2_2 := n1 + n2
  let r2_3 := n2 + n3
  let r3_1 := r2_1 + r2_2
  let r3_2 := r2_2 + r2_3
  let top := r3_1 + r3_2
  top = 54

theorem find_m_value : ∃ m : ℕ, pyramid_property m ∧ m = 12 := by
  sorry

end find_m_value_l419_41920


namespace evaluate_expression_l419_41968

theorem evaluate_expression : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end evaluate_expression_l419_41968


namespace moe_mowing_time_l419_41912

noncomputable def effective_swath_width_inches : ℝ := 30 - 6
noncomputable def effective_swath_width_feet : ℝ := (effective_swath_width_inches / 12)
noncomputable def lawn_width : ℝ := 180
noncomputable def lawn_length : ℝ := 120
noncomputable def walking_rate : ℝ := 4500
noncomputable def total_strips : ℝ := lawn_width / effective_swath_width_feet
noncomputable def total_distance : ℝ := total_strips * lawn_length
noncomputable def time_required : ℝ := total_distance / walking_rate

theorem moe_mowing_time :
  time_required = 2.4 := by
  sorry

end moe_mowing_time_l419_41912


namespace closest_integer_to_cube_root_of_500_l419_41909

theorem closest_integer_to_cube_root_of_500 :
  ∃ n : ℤ, (∀ m : ℤ, |m^3 - 500| ≥ |8^3 - 500|) := 
sorry

end closest_integer_to_cube_root_of_500_l419_41909


namespace rectangle_length_is_4_l419_41972

theorem rectangle_length_is_4 (w l : ℝ) (h_length : l = w + 3) (h_area : l * w = 4) : l = 4 := 
sorry

end rectangle_length_is_4_l419_41972


namespace max_cells_cut_diagonals_l419_41989

theorem max_cells_cut_diagonals (board_size : ℕ) (k : ℕ) (internal_cells : ℕ) :
  board_size = 9 →
  internal_cells = (board_size - 2) ^ 2 →
  64 = internal_cells →
  V = internal_cells + k →
  E = 4 * k →
  k ≤ 21 :=
by
  sorry

end max_cells_cut_diagonals_l419_41989


namespace marseille_hairs_l419_41938

theorem marseille_hairs (N : ℕ) (M : ℕ) (hN : N = 2000000) (hM : M = 300001) :
  ∃ k, k ≥ 7 ∧ ∃ b : ℕ, b ≤ M ∧ b > 0 ∧ ∀ i ≤ M, ∃ l : ℕ, l ≥ k → l ≤ (N / M + 1) :=
by
  sorry

end marseille_hairs_l419_41938


namespace average_weight_increase_l419_41969

theorem average_weight_increase 
  (w_old : ℝ) (w_new : ℝ) (n : ℕ) 
  (h1 : w_old = 65) 
  (h2 : w_new = 93) 
  (h3 : n = 8) : 
  (w_new - w_old) / n = 3.5 := 
by 
  sorry

end average_weight_increase_l419_41969


namespace find_solution_to_inequality_l419_41911

open Set

noncomputable def inequality_solution : Set ℝ := {x : ℝ | 0.5 ≤ x ∧ x < 2 ∨ 3 ≤ x}

theorem find_solution_to_inequality :
  {x : ℝ | (x^2 + 1) / (x - 2) + (2 * x + 3) / (2 * x - 1) ≥ 4} = inequality_solution := 
sorry

end find_solution_to_inequality_l419_41911


namespace arithmetic_geometric_product_l419_41976

theorem arithmetic_geometric_product :
  let a (n : ℕ) := 2 * n - 1
  let b (n : ℕ) := 2 ^ (n - 1)
  b (a 1) * b (a 3) * b (a 5) = 4096 :=
by 
  sorry

end arithmetic_geometric_product_l419_41976


namespace factorize_x_squared_minus_1_l419_41942

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l419_41942


namespace smallest_positive_debt_resolvable_l419_41900

theorem smallest_positive_debt_resolvable :
  ∃ D : ℤ, D > 0 ∧ (D = 250 * p + 175 * g + 125 * s ∧ 
  (∀ (D' : ℤ), D' > 0 → (∃ p g s : ℤ, D' = 250 * p + 175 * g + 125 * s) → D' ≥ D)) := 
sorry

end smallest_positive_debt_resolvable_l419_41900
