import Mathlib

namespace find_second_number_l885_88505

theorem find_second_number (x : ℕ) : 9548 + x = 3362 + 13500 → x = 7314 := by
  sorry

end find_second_number_l885_88505


namespace teacher_already_graded_worksheets_l885_88530

-- Define the conditions
def num_worksheets : ℕ := 9
def problems_per_worksheet : ℕ := 4
def remaining_problems : ℕ := 16
def total_problems := num_worksheets * problems_per_worksheet

-- Define the required proof
theorem teacher_already_graded_worksheets :
  (total_problems - remaining_problems) / problems_per_worksheet = 5 :=
by sorry

end teacher_already_graded_worksheets_l885_88530


namespace marta_should_buy_84_ounces_l885_88542

/-- Definition of the problem's constants and assumptions --/
def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def bag_capacity : ℕ := 49
def num_bags : ℕ := 3

-- Marta wants to put the same number of apples and oranges in each bag
def equal_fruit (A O : ℕ) := A = O

-- Each bag should hold up to 49 ounces of fruit
def bag_limit (n : ℕ) := 4 * n + 3 * n ≤ 49

-- Marta's total apple weight based on the number of apples per bag and number of bags
def total_apple_weight (A : ℕ) : ℕ := (A * 3 * 4)

/-- Statement of the proof problem: 
Marta should buy 84 ounces of apples --/
theorem marta_should_buy_84_ounces : total_apple_weight 7 = 84 :=
by
  sorry

end marta_should_buy_84_ounces_l885_88542


namespace strawberry_picking_l885_88500

theorem strawberry_picking 
  (e : ℕ) (n : ℕ) (p : ℕ) (A : ℕ) (w : ℕ) 
  (h1 : e = 4) 
  (h2 : n = 3) 
  (h3 : p = 20) 
  (h4 : A = 128) 
  : w = 7 :=
by 
  -- proof steps to be filled in
  sorry

end strawberry_picking_l885_88500


namespace lcm_inequality_l885_88535

open Nat

-- Assume positive integers n and m, with n > m
theorem lcm_inequality (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n > m) :
  Nat.lcm m n + Nat.lcm (m+1) (n+1) ≥ 2 * m * Real.sqrt n := 
  sorry

end lcm_inequality_l885_88535


namespace red_ball_count_l885_88506

theorem red_ball_count (w : ℕ) (f : ℝ) (total : ℕ) (r : ℕ) 
  (hw : w = 60)
  (hf : f = 0.25)
  (ht : total = w / (1 - f))
  (hr : r = total * f) : 
  r = 20 :=
by 
  -- Lean doesn't require a proof for the problem statement
  sorry

end red_ball_count_l885_88506


namespace div_simplify_l885_88554

theorem div_simplify (a b : ℝ) (h : a ≠ 0) : (8 * a * b) / (2 * a) = 4 * b :=
by
  sorry

end div_simplify_l885_88554


namespace number_of_devices_bought_l885_88539

-- Define the essential parameters
def original_price : Int := 800000
def discounted_price : Int := 450000
def total_discount : Int := 16450000

-- Define the main statement to prove
theorem number_of_devices_bought : (total_discount / (original_price - discounted_price) = 47) :=
by
  -- The essential proof is skipped here with sorry
  sorry

end number_of_devices_bought_l885_88539


namespace gcd_84_120_eq_12_l885_88519

theorem gcd_84_120_eq_12 : Int.gcd 84 120 = 12 := by
  sorry

end gcd_84_120_eq_12_l885_88519


namespace sum_of_cubes_l885_88581

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 5) (h3 : abc = -6) : a^3 + b^3 + c^3 = -36 :=
sorry

end sum_of_cubes_l885_88581


namespace retirement_year_2020_l885_88579

-- Given conditions
def femaleRetirementAge := 55
def initialRetirementYear (birthYear : ℕ) := birthYear + femaleRetirementAge
def delayedRetirementYear (baseYear additionalYears : ℕ) := baseYear + additionalYears

def postponementStep := 3
def delayStartYear := 2018
def retirementAgeIn2045 := 65
def retirementYear (birthYear : ℕ) : ℕ :=
  let originalRetirementYear := initialRetirementYear birthYear
  let delayYears := ((originalRetirementYear - delayStartYear) / postponementStep) + 1
  delayedRetirementYear originalRetirementYear delayYears

-- Main theorem to prove
theorem retirement_year_2020 : retirementYear 1964 = 2020 := sorry

end retirement_year_2020_l885_88579


namespace find_a_plus_2b_l885_88514

open Real

theorem find_a_plus_2b 
  (a b : ℝ) 
  (ha : 0 < a ∧ a < π / 2) 
  (hb : 0 < b ∧ b < π / 2) 
  (h1 : 4 * (sin a)^2 + 3 * (sin b)^2 = 1) 
  (h2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0) :
  a + 2 * b = π / 2 :=
sorry

end find_a_plus_2b_l885_88514


namespace election_ratio_l885_88568

theorem election_ratio (X Y : ℝ) 
  (h : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) : 
  X / Y = 2 :=
by sorry

end election_ratio_l885_88568


namespace rose_bushes_planted_l885_88551

-- Define the conditions as variables
variable (current_bushes planted_bushes total_bushes : Nat)
variable (h1 : current_bushes = 2) (h2 : total_bushes = 6)
variable (h3 : total_bushes = current_bushes + planted_bushes)

theorem rose_bushes_planted : planted_bushes = 4 := by
  sorry

end rose_bushes_planted_l885_88551


namespace ratio_of_x_to_y_l885_88528

theorem ratio_of_x_to_y (x y : ℤ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) : x / y = -20 :=
by
  sorry

end ratio_of_x_to_y_l885_88528


namespace octagon_side_length_eq_l885_88569

theorem octagon_side_length_eq (AB BC : ℝ) (AE FB s : ℝ) :
  AE = FB → AE < 5 → AB = 10 → BC = 12 →
  s = -11 + Real.sqrt 242 →
  EF = (10.5 - (Real.sqrt 242) / 2) :=
by
  -- Identified parameters and included all conditions from step a)
  intros h1 h2 h3 h4 h5
  -- statement of the theorem to be proven
  let EF := (10.5 - (Real.sqrt 242) / 2)
  sorry  -- placeholder for proof

end octagon_side_length_eq_l885_88569


namespace football_players_count_l885_88566

-- Define the given conditions
def total_students : ℕ := 39
def long_tennis_players : ℕ := 20
def both_sports : ℕ := 17
def play_neither : ℕ := 10

-- Define a theorem to prove the number of football players is 26
theorem football_players_count : 
  ∃ (F : ℕ), F = 26 ∧ 
  (total_students - play_neither) = (F - both_sports) + (long_tennis_players - both_sports) + both_sports :=
by {
  sorry
}

end football_players_count_l885_88566


namespace geometric_figure_perimeter_l885_88563

theorem geometric_figure_perimeter (A : ℝ) (n : ℝ) (area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  A = 216 ∧ n = 6 ∧ area = A / n ∧ side_length = Real.sqrt area ∧ perimeter = 2 * (3 * side_length + 2 * side_length) + 2 * side_length →
  perimeter = 72 := 
by 
  sorry

end geometric_figure_perimeter_l885_88563


namespace range_of_m_l885_88582

open Real Set

variable (x m : ℝ)

def p (x : ℝ) := (x + 1) * (x - 1) ≤ 0
def q (x m : ℝ) := (x + 1) * (x - (3 * m - 1)) ≤ 0 ∧ m > 0

theorem range_of_m (hpsuffq : ∀ x, p x → q x m) (hqnotsuffp : ∃ x, q x m ∧ ¬ p x) : m > 2 / 3 := by
  sorry

end range_of_m_l885_88582


namespace add_base8_l885_88508

-- Define x and y in base 8 and their sum in base 8
def x := 24 -- base 8
def y := 157 -- base 8
def result := 203 -- base 8

theorem add_base8 : (x + y) = result := 
by sorry

end add_base8_l885_88508


namespace each_dog_food_intake_l885_88556

theorem each_dog_food_intake (total_food : ℝ) (dog_count : ℕ) (equal_amount : ℝ) : total_food = 0.25 → dog_count = 2 → (total_food / dog_count) = equal_amount → equal_amount = 0.125 :=
by
  intros h1 h2 h3
  sorry

end each_dog_food_intake_l885_88556


namespace gnuff_tutoring_rate_l885_88534

theorem gnuff_tutoring_rate (flat_rate : ℕ) (total_paid : ℕ) (minutes : ℕ) :
  flat_rate = 20 → total_paid = 146 → minutes = 18 → (total_paid - flat_rate) / minutes = 7 :=
by
  intros
  sorry

end gnuff_tutoring_rate_l885_88534


namespace find_g_six_l885_88589

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_five : g 5 = 6

theorem find_g_six : g 6 = 36/5 := 
by 
  -- proof to be filled in
  sorry

end find_g_six_l885_88589


namespace angle_bisectors_triangle_l885_88592

theorem angle_bisectors_triangle
  (A B C I D K E : Type)
  (triangle : ∀ (A B C : Type), Prop)
  (is_incenter : ∀ (I A B C : Type), Prop)
  (is_on_arc_centered_at : ∀ (X Y : Type), Prop)
  (is_altitude_intersection : ∀ (X Y : Type), Prop)
  (angle_BIC : ∀ (B C : Type), ℝ)
  (angle_DKE : ∀ (D K E : Type), ℝ)
  (α β γ : ℝ)
  (h_sum_ang : α + β + γ = 180) :
  is_incenter I A B C →
  is_on_arc_centered_at D A → is_on_arc_centered_at K A → is_on_arc_centered_at E A →
  is_altitude_intersection E A →
  angle_BIC B C = 180 - (β + γ) / 2 →
  angle_DKE D K E = (360 - α) / 2 →
  angle_BIC B C + angle_DKE D K E = 270 :=
by sorry

end angle_bisectors_triangle_l885_88592


namespace make_polynomial_perfect_square_l885_88513

theorem make_polynomial_perfect_square (m : ℝ) :
  m = 196 → ∃ (f : ℝ → ℝ), ∀ x : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = (f x) ^ 2 :=
by
  sorry

end make_polynomial_perfect_square_l885_88513


namespace triangle_region_areas_l885_88518

open Real

theorem triangle_region_areas (A B C : ℝ) 
  (h1 : 20^2 + 21^2 = 29^2)
  (h2 : ∃ (triangle_area : ℝ), triangle_area = 210)
  (h3 : C > A)
  (h4 : C > B)
  : A + B + 210 = C := 
sorry

end triangle_region_areas_l885_88518


namespace roots_quadratic_expression_l885_88588

theorem roots_quadratic_expression (α β : ℝ) (hα : α^2 - 3 * α - 2 = 0) (hβ : β^2 - 3 * β - 2 = 0) :
    7 * α^4 + 10 * β^3 = 544 := 
sorry

end roots_quadratic_expression_l885_88588


namespace math_problem_l885_88586

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => if a n < 2 * n then a n + 1 else a n

theorem math_problem (n : ℕ) (hn : n > 0) (ha_inc : ∀ m, m > 0 → a m < a (m + 1)) 
  (ha_rec : ∀ m, m > 0 → a (m + 1) ≤ 2 * m) : 
  ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ n = a p - a q := sorry

end math_problem_l885_88586


namespace remainder_when_abc_divided_by_7_l885_88591

theorem remainder_when_abc_divided_by_7 (a b c : ℕ) (h0 : a < 7) (h1 : b < 7) (h2 : c < 7)
  (h3 : (a + 2 * b + 3 * c) % 7 = 0)
  (h4 : (2 * a + 3 * b + c) % 7 = 4)
  (h5 : (3 * a + b + 2 * c) % 7 = 4) :
  (a * b * c) % 7 = 6 := 
sorry

end remainder_when_abc_divided_by_7_l885_88591


namespace find_value_l885_88573

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end find_value_l885_88573


namespace track_length_proof_l885_88517

noncomputable def track_length : ℝ :=
  let x := 541.67
  x

theorem track_length_proof
  (p : ℝ)
  (q : ℝ)
  (h1 : p = 1 / 4)
  (h2 : q = 120)
  (h3 : ¬(p = q))
  (h4 : ∃ r : ℝ, r = 180)
  (speed_constant : ∃ b_speed, ∃ s_speed, b_speed * t = q ∧ s_speed * t = r) :
  track_length = 541.67 :=
sorry

end track_length_proof_l885_88517


namespace roof_length_width_difference_l885_88543

variable (w l : ℕ)

theorem roof_length_width_difference (h1 : l = 7 * w) (h2 : l * w = 847) : l - w = 66 :=
by 
  sorry

end roof_length_width_difference_l885_88543


namespace odd_number_as_difference_of_squares_l885_88502

theorem odd_number_as_difference_of_squares (n : ℤ) (h : ∃ k : ℤ, n = 2 * k + 1) :
  ∃ a b : ℤ, n = a^2 - b^2 :=
by
  sorry

end odd_number_as_difference_of_squares_l885_88502


namespace Dan_running_speed_is_10_l885_88595

noncomputable def running_speed
  (d : ℕ)
  (S : ℕ)
  (avg : ℚ) : ℚ :=
  let total_distance := 2 * d
  let total_time := d / (avg * 60) 
  let swim_time := d / S
  let run_time := total_time - swim_time
  total_distance / run_time

theorem Dan_running_speed_is_10
  (d S : ℕ)
  (avg : ℚ)
  (h1 : d = 4)
  (h2 : S = 6)
  (h3 : avg = 0.125) :
  running_speed d S (avg * 60) = 10 := by 
  sorry

end Dan_running_speed_is_10_l885_88595


namespace black_lambs_count_l885_88526

/-- Definition of the total number of lambs. -/
def total_lambs : Nat := 6048

/-- Definition of the number of white lambs. -/
def white_lambs : Nat := 193

/-- Prove that the number of black lambs is 5855. -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end black_lambs_count_l885_88526


namespace factorial_expression_l885_88523

theorem factorial_expression :
  7 * (Nat.factorial 7) + 6 * (Nat.factorial 6) + 2 * (Nat.factorial 6) = 41040 := by
  sorry

end factorial_expression_l885_88523


namespace value_2_std_dev_less_than_mean_l885_88584

-- Define the mean and standard deviation as constants
def mean : ℝ := 14.5
def std_dev : ℝ := 1.5

-- State the theorem (problem)
theorem value_2_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.5 := by
  sorry

end value_2_std_dev_less_than_mean_l885_88584


namespace calculation_result_l885_88562

theorem calculation_result:
  5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 :=
by
  sorry

end calculation_result_l885_88562


namespace shells_total_l885_88574

variable (x y : ℝ)

theorem shells_total (h1 : y = x + (x + 32)) : y = 2 * x + 32 :=
sorry

end shells_total_l885_88574


namespace find_solutions_l885_88597

theorem find_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + 4^y = 5^z ↔ (x = 3 ∧ y = 2 ∧ z = 2) ∨ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 11 ∧ y = 1 ∧ z = 3) :=
by sorry

end find_solutions_l885_88597


namespace seeds_total_l885_88565

variable (seedsInBigGarden : Nat)
variable (numSmallGardens : Nat)
variable (seedsPerSmallGarden : Nat)

theorem seeds_total (h1 : seedsInBigGarden = 36) (h2 : numSmallGardens = 3) (h3 : seedsPerSmallGarden = 2) : 
  seedsInBigGarden + numSmallGardens * seedsPerSmallGarden = 42 := by
  sorry

end seeds_total_l885_88565


namespace sufficient_but_not_necessary_for_ax_square_pos_l885_88544

variables (a x : ℝ)

theorem sufficient_but_not_necessary_for_ax_square_pos (h : a > 0) : 
  (a > 0 → ax^2 > 0) ∧ ((ax^2 > 0) → a > 0) :=
sorry

end sufficient_but_not_necessary_for_ax_square_pos_l885_88544


namespace inequality_xyz_l885_88545

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * ((x ^ 3 + y ^ 3 + z ^ 3) ^ (1 / 3)) :=
by
  sorry

end inequality_xyz_l885_88545


namespace jellybean_proof_l885_88561

def number_vanilla_jellybeans : ℕ := 120

def number_grape_jellybeans (V : ℕ) : ℕ := 5 * V + 50

def number_strawberry_jellybeans (V : ℕ) : ℕ := (2 * V) / 3

def total_number_jellybeans (V G S : ℕ) : ℕ := V + G + S

def cost_per_vanilla_jellybean : ℚ := 0.05

def cost_per_grape_jellybean : ℚ := 0.08

def cost_per_strawberry_jellybean : ℚ := 0.07

def total_cost_jellybeans (V G S : ℕ) : ℚ := 
  (cost_per_vanilla_jellybean * V) + 
  (cost_per_grape_jellybean * G) + 
  (cost_per_strawberry_jellybean * S)

theorem jellybean_proof :
  ∃ (V G S : ℕ), 
    V = number_vanilla_jellybeans ∧
    G = number_grape_jellybeans V ∧
    S = number_strawberry_jellybeans V ∧
    total_number_jellybeans V G S = 850 ∧
    total_cost_jellybeans V G S = 63.60 :=
by
  sorry

end jellybean_proof_l885_88561


namespace sum_of_terms_l885_88583

def geometric_sequence (a b c d : ℝ) :=
  ∃ q : ℝ, a = b / q ∧ c = b * q ∧ d = c * q

def symmetric_sequence_of_length_7 (s : Fin 8 → ℝ) :=
  ∀ i : Fin 8, s i = s (Fin.mk (7 - i) sorry)

def sequence_conditions (s : Fin 8 → ℝ) :=
  symmetric_sequence_of_length_7 s ∧
  geometric_sequence (s ⟨1,sorry⟩) (s ⟨2,sorry⟩) (s ⟨3,sorry⟩) (s ⟨4,sorry⟩) ∧
  s ⟨1,sorry⟩ = 2 ∧
  s ⟨3,sorry⟩ = 8

theorem sum_of_terms (s : Fin 8 → ℝ) (h : sequence_conditions s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = 44 ∨
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = -4 :=
sorry

end sum_of_terms_l885_88583


namespace second_player_wins_l885_88501

-- Piles of balls and game conditions
def two_pile_game (pile1 pile2 : ℕ) : Prop :=
  ∀ (player1_turn : ℕ → Prop) (player2_turn : ℕ → Prop),
    (∀ n : ℕ, player1_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player1's move
    (∀ n : ℕ, player2_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player2's move
    -- - Second player has a winning strategy
    ∃ (win_strategy : ℕ → ℕ), ∀ k : ℕ, player1_turn k → player2_turn (win_strategy k) 

-- Lean statement of the problem
theorem second_player_wins : ∀ (pile1 pile2 : ℕ), pile1 = 30 ∧ pile2 = 30 → two_pile_game pile1 pile2 :=
  by
    intros pile1 pile2 h
    sorry  -- Placeholder for the proof


end second_player_wins_l885_88501


namespace horner_rule_example_l885_88593

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_example : f 2 = 62 := by
  sorry

end horner_rule_example_l885_88593


namespace range_f_l885_88590

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x + 1)

theorem range_f : (Set.range f) = Set.univ := by
  sorry

end range_f_l885_88590


namespace logistics_company_freight_l885_88598

theorem logistics_company_freight :
  ∃ (x y : ℕ), 
    50 * x + 30 * y = 9500 ∧
    70 * x + 40 * y = 13000 ∧
    x = 100 ∧
    y = 140 :=
by
  -- The proof is skipped here
  sorry

end logistics_company_freight_l885_88598


namespace joan_final_oranges_l885_88532

def joan_oranges_initial := 75
def tom_oranges := 42
def sara_sold := 40
def christine_added := 15

theorem joan_final_oranges : joan_oranges_initial + tom_oranges - sara_sold + christine_added = 92 :=
by 
  sorry

end joan_final_oranges_l885_88532


namespace geometric_series_terms_l885_88507

theorem geometric_series_terms 
    (b1 q : ℝ)
    (h₁ : (b1^2 / (1 + q + q^2)) = 12)
    (h₂ : (b1^2 / (1 + q^2)) = (36 / 5)) :
    (b1 = 3 ∨ b1 = -3) ∧ q = -1/2 :=
by
  sorry

end geometric_series_terms_l885_88507


namespace expression_value_l885_88520

theorem expression_value (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
by
  sorry

end expression_value_l885_88520


namespace simplify_fraction_l885_88509

theorem simplify_fraction : 
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) :=
by
  -- Proof will go here
  sorry

end simplify_fraction_l885_88509


namespace solution_set_of_inequality_l885_88512

theorem solution_set_of_inequality :
  {x : ℝ | -1 < x ∧ x < 2} = {x : ℝ | (x - 2) / (x + 1) < 0} :=
sorry

end solution_set_of_inequality_l885_88512


namespace ratio_of_areas_l885_88557

theorem ratio_of_areas (r : ℝ) (h1 : r > 0) : 
  let OX := r / 3
  let area_OP := π * r ^ 2
  let area_OX := π * (OX) ^ 2
  (area_OX / area_OP) = 1 / 9 :=
by
  sorry

end ratio_of_areas_l885_88557


namespace intersection_M_N_l885_88546

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Prove the intersection of M and N equals (1, 2)
theorem intersection_M_N :
  ∀ x, x ∈ M ∩ N ↔ 1 < x ∧ x < 2 :=
by
  -- Skipping the proof here
  sorry

end intersection_M_N_l885_88546


namespace unique_triplet_exists_l885_88555

theorem unique_triplet_exists (a b p : ℕ) (hp : Nat.Prime p) : 
  (a + b)^p = p^a + p^b → (a = 1 ∧ b = 1 ∧ p = 2) :=
by sorry

end unique_triplet_exists_l885_88555


namespace jack_pays_back_expected_amount_l885_88525

-- Definitions from the conditions
def principal : ℝ := 1200
def interest_rate : ℝ := 0.10

-- Definition for proof
def interest : ℝ := principal * interest_rate
def total_amount : ℝ := principal + interest

-- Lean statement for the proof problem
theorem jack_pays_back_expected_amount : total_amount = 1320 := by
  sorry

end jack_pays_back_expected_amount_l885_88525


namespace quadrilateral_area_l885_88541

-- Define the number of interior and boundary points
def interior_points : ℕ := 5
def boundary_points : ℕ := 4

-- State the theorem to prove the area of the quadrilateral using Pick's Theorem
theorem quadrilateral_area : interior_points + (boundary_points / 2) - 1 = 6 := by sorry

end quadrilateral_area_l885_88541


namespace max_value_y_on_interval_l885_88559

noncomputable def y (x: ℝ) : ℝ := x^4 - 8 * x^2 + 2

theorem max_value_y_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y x = 11 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y z ≤ 11 := 
sorry

end max_value_y_on_interval_l885_88559


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l885_88515

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l885_88515


namespace abs_neg_three_l885_88564

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l885_88564


namespace snow_probability_l885_88560

theorem snow_probability :
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  1 - p_no_snow_week = 29 / 32 :=
by
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  have p_no_snow_week_eq : p_no_snow_week = 3 / 32 := sorry
  have p_snow_at_least_once_week : 1 - p_no_snow_week = 29 / 32 := sorry
  exact p_snow_at_least_once_week

end snow_probability_l885_88560


namespace find_m_l885_88510

theorem find_m (m x_1 x_2 : ℝ) 
  (h1 : x_1^2 + m * x_1 - 3 = 0) 
  (h2 : x_2^2 + m * x_2 - 3 = 0) 
  (h3 : x_1 + x_2 - x_1 * x_2 = 5) : 
  m = -2 :=
sorry

end find_m_l885_88510


namespace complex_number_arithmetic_l885_88575

theorem complex_number_arithmetic (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_number_arithmetic_l885_88575


namespace impossible_load_two_coins_l885_88521

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l885_88521


namespace original_price_proof_l885_88576

noncomputable def original_price (profit selling_price : ℝ) : ℝ :=
  (profit / 0.20)

theorem original_price_proof (P : ℝ) : 
  original_price 600 (P + 600) = 3000 :=
by
  sorry

end original_price_proof_l885_88576


namespace minimize_product_of_roots_of_quadratic_eq_l885_88537

theorem minimize_product_of_roots_of_quadratic_eq (k : ℝ) :
  (∃ x y : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) 
  → k = 25 / 8 :=
sorry

end minimize_product_of_roots_of_quadratic_eq_l885_88537


namespace b_range_condition_l885_88516

theorem b_range_condition (b : ℝ) : 
  -2 * Real.sqrt 6 < b ∧ b < 2 * Real.sqrt 6 ↔ (b^2 - 24) < 0 :=
by
  sorry

end b_range_condition_l885_88516


namespace candle_length_sum_l885_88578

theorem candle_length_sum (l s : ℕ) (x : ℤ) 
  (h1 : l = s + 32)
  (h2 : s = (5 * x)) 
  (h3 : l = (7 * (3 * x))) :
  l + s = 52 := 
sorry

end candle_length_sum_l885_88578


namespace Binkie_gemstones_l885_88599

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l885_88599


namespace min_value_expression_l885_88571

theorem min_value_expression (x y : ℝ) :
  ∃ m, (m = 104) ∧ (∀ x y : ℝ, (x + 3)^2 + 2 * (y - 2)^2 + 4 * (x - 7)^2 + (y + 4)^2 ≥ m) :=
sorry

end min_value_expression_l885_88571


namespace train_passing_time_l885_88524

-- Definitions based on the conditions
def length_T1 : ℕ := 800
def speed_T1_kmph : ℕ := 108
def length_T2 : ℕ := 600
def speed_T2_kmph : ℕ := 72

-- Converting kmph to mps
def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600
def speed_T1_mps : ℕ := convert_kmph_to_mps speed_T1_kmph
def speed_T2_mps : ℕ := convert_kmph_to_mps speed_T2_kmph

-- Calculating relative speed and total length
def relative_speed_T1_T2 : ℕ := speed_T1_mps - speed_T2_mps
def total_length_T1_T2 : ℕ := length_T1 + length_T2

-- Proving the time to pass
theorem train_passing_time : total_length_T1_T2 / relative_speed_T1_T2 = 140 := by
  sorry

end train_passing_time_l885_88524


namespace max_value_of_f_l885_88531

variable (n : ℕ)

-- Define the quadratic function with coefficients a, b, and c.
noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
axiom f_n : ∃ a b c, f n a b c = 6
axiom f_n1 : ∃ a b c, f (n + 1) a b c = 14
axiom f_n2 : ∃ a b c, f (n + 2) a b c = 14

-- The main goal is to prove the maximum value of f(x) is 15.
theorem max_value_of_f : ∃ a b c, (∀ x : ℝ, f x a b c ≤ 15) :=
by
  sorry

end max_value_of_f_l885_88531


namespace division_result_l885_88572

theorem division_result (a b : ℕ) (ha : a = 7) (hb : b = 3) :
    ((a^3 + b^3) / (a^2 - a * b + b^2) = 10) := 
by
  sorry

end division_result_l885_88572


namespace total_fruits_in_30_days_l885_88547

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end total_fruits_in_30_days_l885_88547


namespace perpendicular_lines_a_eq_neg6_l885_88503

theorem perpendicular_lines_a_eq_neg6 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, ax + 2*y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + 3*y - 2 = 0) 
  (h_perpendicular : ∀ m1 m2 : ℝ, m1 * m2 = -1) : 
  a = -6 := 
by 
  sorry

end perpendicular_lines_a_eq_neg6_l885_88503


namespace c_completion_days_l885_88533

noncomputable def work_rate (days: ℕ) := (1 : ℝ) / days

theorem c_completion_days : 
  ∀ (W : ℝ) (Ra Rb Rc : ℝ) (Dc : ℕ),
  Ra = work_rate 30 → Rb = work_rate 30 → Rc = work_rate Dc →
  (Ra + Rb + Rc) * 8 + (Ra + Rb) * 4 = W → 
  Dc = 40 :=
by
  intros W Ra Rb Rc Dc hRa hRb hRc hW
  sorry

end c_completion_days_l885_88533


namespace repeating_decimal_subtraction_simplified_l885_88538

theorem repeating_decimal_subtraction_simplified :
  let x := (567 / 999 : ℚ)
  let y := (234 / 999 : ℚ)
  let z := (891 / 999 : ℚ)
  x - y - z = -186 / 333 :=
by
  sorry

end repeating_decimal_subtraction_simplified_l885_88538


namespace least_repeating_block_of_8_over_11_l885_88585

theorem least_repeating_block_of_8_over_11 : (∃ n : ℕ, (∀ m : ℕ, m < n → ¬(∃ a b : ℤ, (10^m - 1) * (8 * 10^n - b * 11 * 10^(n - t)) = a * 11 * 10^(m - t))) ∧ n ≤ 2) :=
by
  sorry

end least_repeating_block_of_8_over_11_l885_88585


namespace coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l885_88536

def coprime_distinct_remainders (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : Prop :=
  ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
    (∀ (i : Fin m) (j : Fin k), ∀ (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k))

def not_coprime_congruent_product (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : Prop :=
  ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
    ∃ (i : Fin m) (j : Fin k) (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a s * b t) % (m * k)

-- Example statement to assert the existence of the above properties
theorem coprime_mk_has_distinct_products 
  (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : coprime_distinct_remainders m k coprime_mk :=
sorry

theorem not_coprime_mk_has_congruent_products 
  (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : not_coprime_congruent_product m k not_coprime_mk :=
sorry

end coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l885_88536


namespace PQRS_product_l885_88580

theorem PQRS_product :
  let P := (Real.sqrt 2012 + Real.sqrt 2013)
  let Q := (-Real.sqrt 2012 - Real.sqrt 2013)
  let R := (Real.sqrt 2012 - Real.sqrt 2013)
  let S := (Real.sqrt 2013 - Real.sqrt 2012)
  P * Q * R * S = 1 :=
by
  sorry

end PQRS_product_l885_88580


namespace least_value_of_a_l885_88529

theorem least_value_of_a (a : ℝ) (h : a^2 - 12 * a + 35 ≤ 0) : 5 ≤ a :=
by {
  sorry
}

end least_value_of_a_l885_88529


namespace grade_assignments_count_l885_88596

theorem grade_assignments_count (n : ℕ) (g : ℕ) (h : n = 15) (k : g = 4) : g^n = 1073741824 :=
by
  sorry

end grade_assignments_count_l885_88596


namespace min_value_fraction_l885_88504

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + 2 * y + 6) : 
  (∃ (z : ℝ), z = 1 / x + 1 / (2 * y) ∧ z ≥ 1 / 3) :=
sorry

end min_value_fraction_l885_88504


namespace perimeter_of_staircase_region_l885_88550

-- Definitions according to the conditions.
def staircase_region.all_right_angles : Prop := True -- Given condition that all angles are right angles.
def staircase_region.side_length : ℕ := 1 -- Given condition that the side length of each congruent side is 1 foot.
def staircase_region.total_area : ℕ := 120 -- Given condition that the total area of the region is 120 square feet.
def num_sides : ℕ := 12 -- Number of congruent sides.

-- The question is to prove that the perimeter of the region is 36 feet.
theorem perimeter_of_staircase_region : 
  (num_sides * staircase_region.side_length + 
    15 + -- length added to complete the larger rectangle assuming x = 15
    9   -- length added to complete the larger rectangle assuming y = 9
  ) = 36 := 
by
  -- Given and facts are already logically considered to prove (conditions and right angles are trivial)
  sorry

end perimeter_of_staircase_region_l885_88550


namespace a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l885_88511

-- Definitions for the point P and movements
def move (P : ℤ) (flip : Bool) : ℤ :=
  if flip then P + 1 else -P

-- Definitions for probabilities
def probability_of_event (events : ℕ) (successful : ℕ) : ℚ :=
  successful / events

def probability_a3_zero : ℚ :=
  probability_of_event 8 2  -- 2 out of 8 sequences lead to a3 = 0

def probability_a4_one : ℚ :=
  probability_of_event 16 2  -- 2 out of 16 sequences lead to a4 = 1

noncomputable def probability_an_n_minus_3 (n : ℕ) : ℚ :=
  if n < 3 then 0 else (n - 1) / (2 ^ n)

-- Statements to prove
theorem a3_probability_is_one_fourth : probability_a3_zero = 1/4 := by
  sorry

theorem a4_probability_is_one_eighth : probability_a4_one = 1/8 := by
  sorry

theorem an_n_minus_3_probability (n : ℕ) (hn : n ≥ 3) : probability_an_n_minus_3 n = (n - 1) / (2^n) := by
  sorry

end a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l885_88511


namespace restaurant_sales_l885_88558

theorem restaurant_sales (monday tuesday wednesday thursday : ℕ) 
  (h1 : monday = 40) 
  (h2 : tuesday = monday + 40) 
  (h3 : wednesday = tuesday / 2) 
  (h4 : monday + tuesday + wednesday + thursday = 203) : 
  thursday = wednesday + 3 := 
by sorry

end restaurant_sales_l885_88558


namespace bricks_of_other_types_l885_88587

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end bricks_of_other_types_l885_88587


namespace kelly_initial_apples_l885_88594

theorem kelly_initial_apples : ∀ (T P I : ℕ), T = 105 → P = 49 → I + P = T → I = 56 :=
by
  intros T P I ht hp h
  rw [ht, hp] at h
  linarith

end kelly_initial_apples_l885_88594


namespace find_capacity_l885_88570

noncomputable def pool_capacity (V1 V2 q : ℝ) : Prop :=
  V1 = q / 120 ∧ V2 = V1 + 50 ∧ V1 + V2 = q / 48

theorem find_capacity (q : ℝ) : ∃ V1 V2, pool_capacity V1 V2 q → q = 12000 :=
by 
  sorry

end find_capacity_l885_88570


namespace slips_with_3_l885_88522

variable (total_slips : ℕ) (expected_value : ℚ) (num_slips_with_3 : ℕ)

def num_slips_with_9 := total_slips - num_slips_with_3

def expected_value_calc (total_slips expected_value : ℚ) (num_slips_with_3 num_slips_with_9 : ℕ) : ℚ :=
  (num_slips_with_3 / total_slips) * 3 + (num_slips_with_9 / total_slips) * 9

theorem slips_with_3 (h1 : total_slips = 15) (h2 : expected_value = 5.4)
  (h3 : expected_value_calc total_slips expected_value num_slips_with_3 (num_slips_with_9 total_slips num_slips_with_3) = expected_value) :
  num_slips_with_3 = 9 :=
by
  rw [h1, h2] at h3
  sorry

end slips_with_3_l885_88522


namespace work_done_by_gas_l885_88552

theorem work_done_by_gas (n : ℕ) (R T0 Pa : ℝ) (V0 : ℝ) (W : ℝ) :
  -- Conditions
  n = 1 ∧
  R = 8.314 ∧
  T0 = 320 ∧
  Pa * V0 = n * R * T0 ∧
  -- Question Statement and Correct Answer
  W = Pa * V0 / 2 →
  W = 665 :=
by sorry

end work_done_by_gas_l885_88552


namespace number_added_multiplied_l885_88549

theorem number_added_multiplied (x : ℕ) (h : (7/8 : ℚ) * x = 28) : ((x + 16) * (5/16 : ℚ)) = 15 :=
by
  sorry

end number_added_multiplied_l885_88549


namespace relationship_between_abc_l885_88527

noncomputable def a : ℝ := (0.6 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (1.5 : ℝ)
noncomputable def c : ℝ := (1.5 : ℝ) ^ (0.6 : ℝ)

theorem relationship_between_abc : c > a ∧ a > b := sorry

end relationship_between_abc_l885_88527


namespace distance_AC_100_l885_88553

theorem distance_AC_100 (d_AB : ℝ) (t1 : ℝ) (t2 : ℝ) (AC : ℝ) (CB : ℝ) :
  d_AB = 150 ∧ t1 = 3 ∧ t2 = 12 ∧ d_AB = AC + CB ∧ AC / 3 = CB / 12 → AC = 100 := 
by
  sorry

end distance_AC_100_l885_88553


namespace polar_to_rectangular_l885_88567

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 6) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, -3 * Real.sqrt 3) :=
by
  -- Definitions and assertions from the conditions
  have cos_theta : Real.cos (5 * Real.pi / 3) = 1 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted
  have sin_theta : Real.sin (5 * Real.pi / 3) = - Real.sqrt 3 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted

  -- Proof that the converted coordinates match the expected result
  rw [hr, hθ, cos_theta, sin_theta]
  simp
  -- Detailed proof steps to verify (6 * (1 / 2), 6 * (- Real.sqrt 3 / 2)) = (3, -3 * Real.sqrt 3) omitted
  sorry

end polar_to_rectangular_l885_88567


namespace graph_is_hyperbola_l885_88540

theorem graph_is_hyperbola : ∀ (x y : ℝ), x^2 - 18 * y^2 - 6 * x + 4 * y + 9 = 0 → ∃ a b c d : ℝ, a * (x - b)^2 - c * (y - d)^2 = 1 :=
by
  -- Proof is omitted
  sorry

end graph_is_hyperbola_l885_88540


namespace integer_range_2014_l885_88577

theorem integer_range_2014 : 1000 < 2014 ∧ 2014 < 10000 := by
  sorry

end integer_range_2014_l885_88577


namespace second_day_hike_ratio_l885_88548

theorem second_day_hike_ratio (full_hike_distance first_day_distance third_day_distance : ℕ) 
(h_full_hike: full_hike_distance = 50)
(h_first_day: first_day_distance = 10)
(h_third_day: third_day_distance = 15) : 
(full_hike_distance - (first_day_distance + third_day_distance)) / full_hike_distance = 1 / 2 := by
  sorry

end second_day_hike_ratio_l885_88548
