import Mathlib

namespace problem_statement_l1443_144338

-- Define the necessary and sufficient conditions
def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ (¬ (P → Q))

-- Specific propositions in this scenario
def x_conditions (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Prove the given problem statement
theorem problem_statement (x : ℝ) : necessary_but_not_sufficient (x_conditions x) (x_equals_3 x) :=
  sorry

end problem_statement_l1443_144338


namespace find_integer_values_of_a_l1443_144324

theorem find_integer_values_of_a
  (x a b c : ℤ)
  (h : (x - a) * (x - 10) + 5 = (x + b) * (x + c)) :
  a = 4 ∨ a = 16 := by
    sorry

end find_integer_values_of_a_l1443_144324


namespace isosceles_triangle_largest_angle_l1443_144380

/-- 
  Given an isosceles triangle where one of the angles is 20% smaller than a right angle,
  prove that the measure of one of the two largest angles is 54 degrees.
-/
theorem isosceles_triangle_largest_angle 
  (A B C : ℝ) 
  (triangle_ABC : A + B + C = 180)
  (isosceles_triangle : A = B ∨ A = C ∨ B = C)
  (smaller_angle : A = 0.80 * 90) :
  A = 54 ∨ B = 54 ∨ C = 54 :=
sorry

end isosceles_triangle_largest_angle_l1443_144380


namespace scientific_notation_of_384_000_000_l1443_144314

theorem scientific_notation_of_384_000_000 :
  384000000 = 3.84 * 10^8 :=
sorry

end scientific_notation_of_384_000_000_l1443_144314


namespace compare_powers_l1443_144357

def n1 := 22^44
def n2 := 33^33
def n3 := 44^22

theorem compare_powers : n1 > n2 ∧ n2 > n3 := by
  sorry

end compare_powers_l1443_144357


namespace marbles_left_l1443_144355

def initial_marbles : ℕ := 100
def percent_t_to_Theresa : ℕ := 25
def percent_t_to_Elliot : ℕ := 10

theorem marbles_left (w t e : ℕ) (h_w : w = initial_marbles)
                                 (h_t : t = percent_t_to_Theresa)
                                 (h_e : e = percent_t_to_Elliot) : w - ((t * w) / 100 + (e * w) / 100) = 65 :=
by
  rw [h_w, h_t, h_e]
  sorry

end marbles_left_l1443_144355


namespace no_such_rectangle_exists_l1443_144331

theorem no_such_rectangle_exists :
  ¬(∃ (x y : ℝ), (∃ a b c d : ℕ, x = a + b * Real.sqrt 3 ∧ y = c + d * Real.sqrt 3) ∧ 
                (x * y = (3 * Real.sqrt 3) / 2 + n * (Real.sqrt 3 / 2))) :=
sorry

end no_such_rectangle_exists_l1443_144331


namespace joan_needs_more_flour_l1443_144354

-- Definitions for the conditions
def total_flour : ℕ := 7
def flour_added : ℕ := 3

-- The theorem stating the proof problem
theorem joan_needs_more_flour : total_flour - flour_added = 4 :=
by
  sorry

end joan_needs_more_flour_l1443_144354


namespace combined_molecular_weight_l1443_144310

theorem combined_molecular_weight 
  (atomic_weight_N : ℝ)
  (atomic_weight_O : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_C : ℝ)
  (moles_N2O3 : ℝ)
  (moles_H2O : ℝ)
  (moles_CO2 : ℝ)
  (molecular_weight_N2O3 : ℝ)
  (molecular_weight_H2O : ℝ)
  (molecular_weight_CO2 : ℝ)
  (weight_N2O3 : ℝ)
  (weight_H2O : ℝ)
  (weight_CO2 : ℝ)
  : 
  moles_N2O3 = 4 →
  moles_H2O = 3.5 →
  moles_CO2 = 2 →
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  atomic_weight_H = 1.01 →
  atomic_weight_C = 12.01 →
  molecular_weight_N2O3 = (2 * atomic_weight_N) + (3 * atomic_weight_O) →
  molecular_weight_H2O = (2 * atomic_weight_H) + atomic_weight_O →
  molecular_weight_CO2 = atomic_weight_C + (2 * atomic_weight_O) →
  weight_N2O3 = moles_N2O3 * molecular_weight_N2O3 →
  weight_H2O = moles_H2O * molecular_weight_H2O →
  weight_CO2 = moles_CO2 * molecular_weight_CO2 →
  weight_N2O3 + weight_H2O + weight_CO2 = 455.17 :=
by 
  intros;
  sorry

end combined_molecular_weight_l1443_144310


namespace red_markers_count_l1443_144382

-- Define the given conditions
def blue_markers : ℕ := 1028
def total_markers : ℕ := 3343

-- Define the red_makers calculation based on the conditions
def red_markers (total_markers blue_markers : ℕ) : ℕ := total_markers - blue_markers

-- Prove that the number of red markers is 2315 given the conditions
theorem red_markers_count : red_markers total_markers blue_markers = 2315 := by
  -- We can skip the proof for this demonstration
  sorry

end red_markers_count_l1443_144382


namespace rabbit_weight_l1443_144347

theorem rabbit_weight (a b c : ℕ) (h1 : a + b + c = 30) (h2 : a + c = 2 * b) (h3 : a + b = c) :
  a = 5 := by
  sorry

end rabbit_weight_l1443_144347


namespace sum_angles_of_two_triangles_l1443_144362

theorem sum_angles_of_two_triangles (a1 a3 a5 a2 a4 a6 : ℝ) 
  (hABC : a1 + a3 + a5 = 180) (hDEF : a2 + a4 + a6 = 180) : 
  a1 + a2 + a3 + a4 + a5 + a6 = 360 :=
by
  sorry

end sum_angles_of_two_triangles_l1443_144362


namespace vanessa_total_earnings_l1443_144332

theorem vanessa_total_earnings :
  let num_dresses := 7
  let num_shirts := 4
  let price_per_dress := 7
  let price_per_shirt := 5
  (num_dresses * price_per_dress + num_shirts * price_per_shirt) = 69 :=
by
  sorry

end vanessa_total_earnings_l1443_144332


namespace sum_of_number_and_its_square_is_20_l1443_144305

theorem sum_of_number_and_its_square_is_20 (n : ℕ) (h : n = 4) : n + n^2 = 20 :=
by
  sorry

end sum_of_number_and_its_square_is_20_l1443_144305


namespace sequence_formula_l1443_144377

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 2^n - 1 :=
by
  sorry

end sequence_formula_l1443_144377


namespace Q_value_l1443_144342

theorem Q_value (a b c P Q : ℝ) (h1 : a + b + c = 0)
    (h2 : (a^2 / (2 * a^2 + b * c)) + (b^2 / (2 * b^2 + a * c)) + (c^2 / (2 * c^2 + a * b)) = P - 3 * Q) : 
    Q = 8 := 
sorry

end Q_value_l1443_144342


namespace smallest_multiple_6_15_l1443_144341

theorem smallest_multiple_6_15 (b : ℕ) (hb1 : b % 6 = 0) (hb2 : b % 15 = 0) :
  ∃ (b : ℕ), (b > 0) ∧ (b % 6 = 0) ∧ (b % 15 = 0) ∧ (∀ x : ℕ, (x > 0) ∧ (x % 6 = 0) ∧ (x % 15 = 0) → x ≥ b) :=
sorry

end smallest_multiple_6_15_l1443_144341


namespace combined_total_score_is_correct_l1443_144329

-- Definitions of point values
def touchdown_points := 6
def extra_point_points := 1
def field_goal_points := 3

-- Hawks' Scores
def hawks_touchdowns := 4
def hawks_successful_extra_points := 2
def hawks_field_goals := 2

-- Eagles' Scores
def eagles_touchdowns := 3
def eagles_successful_extra_points := 3
def eagles_field_goals := 3

-- Calculations
def hawks_total_points := hawks_touchdowns * touchdown_points +
                          hawks_successful_extra_points * extra_point_points +
                          hawks_field_goals * field_goal_points

def eagles_total_points := eagles_touchdowns * touchdown_points +
                           eagles_successful_extra_points * extra_point_points +
                           eagles_field_goals * field_goal_points

def combined_total_score := hawks_total_points + eagles_total_points

-- The theorem that needs to be proved
theorem combined_total_score_is_correct : combined_total_score = 62 :=
by
  -- proof would go here
  sorry

end combined_total_score_is_correct_l1443_144329


namespace real_part_of_complex_div_l1443_144302

noncomputable def complexDiv (c1 c2 : ℂ) := c1 / c2

theorem real_part_of_complex_div (i_unit : ℂ) (h_i : i_unit = Complex.I) :
  (Complex.re (complexDiv (2 * i_unit) (1 + i_unit)) = 1) :=
by
  sorry

end real_part_of_complex_div_l1443_144302


namespace max_a_plus_b_l1443_144396

theorem max_a_plus_b (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) : a + b ≤ 14 / 5 := 
sorry

end max_a_plus_b_l1443_144396


namespace violet_needs_water_l1443_144353

/-- Violet needs 800 ml of water per hour hiked, her dog needs 400 ml of water per hour,
    and they can hike for 4 hours. We need to prove that Violet needs 4.8 liters of water
    for the hike. -/
theorem violet_needs_water (hiking_hours : ℝ)
  (violet_water_per_hour : ℝ)
  (dog_water_per_hour : ℝ)
  (violet_water_needed : ℝ)
  (dog_water_needed : ℝ)
  (total_water_needed_ml : ℝ)
  (total_water_needed_liters : ℝ) :
  hiking_hours = 4 ∧
  violet_water_per_hour = 800 ∧
  dog_water_per_hour = 400 ∧
  violet_water_needed = 3200 ∧
  dog_water_needed = 1600 ∧
  total_water_needed_ml = 4800 ∧
  total_water_needed_liters = 4.8 →
  total_water_needed_liters = 4.8 :=
by sorry

end violet_needs_water_l1443_144353


namespace perimeter_of_equilateral_triangle_l1443_144315

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l1443_144315


namespace assignment_problem_l1443_144360

theorem assignment_problem (a b c : ℕ) (h1 : a = 10) (h2 : b = 20) (h3 : c = 30) :
  let a := b
  let b := c
  let c := a
  a = 20 ∧ b = 30 ∧ c = 20 :=
by
  sorry

end assignment_problem_l1443_144360


namespace intersection_complement_A_B_l1443_144327

open Set

theorem intersection_complement_A_B :
  let A := {x : ℝ | x + 1 > 0}
  let B := {-2, -1, 0, 1}
  (compl A ∩ B : Set ℝ) = {-2, -1} :=
by
  sorry

end intersection_complement_A_B_l1443_144327


namespace john_has_leftover_correct_l1443_144375

-- Define the initial conditions
def initial_gallons : ℚ := 5
def given_away : ℚ := 18 / 7

-- Define the target result after subtraction
def remaining_gallons : ℚ := 17 / 7

-- The theorem statement
theorem john_has_leftover_correct :
  initial_gallons - given_away = remaining_gallons :=
by
  sorry

end john_has_leftover_correct_l1443_144375


namespace mary_age_l1443_144340

theorem mary_age (M F : ℕ) (h1 : F = 4 * M) (h2 : F - 3 = 5 * (M - 3)) : M = 12 :=
by
  sorry

end mary_age_l1443_144340


namespace train_speed_solution_l1443_144345

def train_speed_problem (L v : ℝ) (man_time platform_time : ℝ) (platform_length : ℝ) :=
  man_time = 12 ∧
  platform_time = 30 ∧
  platform_length = 180 ∧
  L = v * man_time ∧
  (L + platform_length) = v * platform_time

theorem train_speed_solution (L v : ℝ) (h : train_speed_problem L v 12 30 180) :
  v * 3.6 = 36 :=
by
  sorry

end train_speed_solution_l1443_144345


namespace calculate_expression_l1443_144336

theorem calculate_expression (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 :=
by
  sorry

end calculate_expression_l1443_144336


namespace height_of_block_l1443_144304

theorem height_of_block (h : ℝ) : 
  ((∃ (side : ℝ), ∃ (n : ℕ), side = 15 ∧ n = 10 ∧ 15 * 30 * h = n * side^3) → h = 75) := 
by
  intros
  sorry

end height_of_block_l1443_144304


namespace supplement_twice_angle_l1443_144317

theorem supplement_twice_angle (α : ℝ) (h : 180 - α = 2 * α) : α = 60 := by
  admit -- This is a placeholder for the actual proof

end supplement_twice_angle_l1443_144317


namespace reynald_volleyballs_l1443_144394

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

theorem reynald_volleyballs : volleyballs = 30 :=
by
  sorry

end reynald_volleyballs_l1443_144394


namespace find_y_perpendicular_l1443_144372

theorem find_y_perpendicular (y : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (2, y))
  (ha : a = (2, 1))
  (h_perp : (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0) :
  y = -4 :=
sorry

end find_y_perpendicular_l1443_144372


namespace solve_for_a_l1443_144344

theorem solve_for_a (a : ℝ) (h : a / 0.3 = 0.6) : a = 0.18 :=
by sorry

end solve_for_a_l1443_144344


namespace circle_center_l1443_144313

theorem circle_center {x y : ℝ} :
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → (x, y) = (1, 2) :=
by
  sorry

end circle_center_l1443_144313


namespace total_kids_played_l1443_144351

-- Definitions based on conditions
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Total kids calculation
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Theorem to prove
theorem total_kids_played (Julia : Prop) : totalKids = 34 :=
by
  -- Using sorry to skip the proof
  sorry

end total_kids_played_l1443_144351


namespace fraction_less_than_40_percent_l1443_144381

theorem fraction_less_than_40_percent (x : ℝ) (h1 : x * 180 = 48) (h2 : x < 0.4) : x = 4 / 15 :=
by
  sorry

end fraction_less_than_40_percent_l1443_144381


namespace green_hats_count_l1443_144330

theorem green_hats_count : ∃ G B : ℕ, B + G = 85 ∧ 6 * B + 7 * G = 540 ∧ G = 30 :=
by
  sorry

end green_hats_count_l1443_144330


namespace sum_of_squares_is_perfect_square_l1443_144374

theorem sum_of_squares_is_perfect_square (n p k : ℤ) : 
  (∃ m : ℤ, n^2 + p^2 + k^2 = m^2) ↔ (n * k = (p / 2)^2) :=
by
  sorry

end sum_of_squares_is_perfect_square_l1443_144374


namespace inequality_of_abc_l1443_144387

theorem inequality_of_abc (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_of_abc_l1443_144387


namespace find_integer_l1443_144349

noncomputable def least_possible_sum (x y z k : ℕ) : Prop :=
  2 * x = 5 * y ∧ 5 * y = 6 * z ∧ x + k + z = 26

theorem find_integer (x y z : ℕ) (h : least_possible_sum x y z 6) :
  6 = (26 - x - z) :=
  by {
    sorry
  }

end find_integer_l1443_144349


namespace sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l1443_144323

theorem sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^4 + t^2) = |t| * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l1443_144323


namespace parabola_points_l1443_144373

theorem parabola_points :
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} :=
by
  sorry

end parabola_points_l1443_144373


namespace selling_price_of_cycle_l1443_144391

def cost_price : ℝ := 1400
def loss_percentage : ℝ := 18

theorem selling_price_of_cycle : 
    (cost_price - (loss_percentage / 100) * cost_price) = 1148 := 
by
  sorry

end selling_price_of_cycle_l1443_144391


namespace new_person_weight_l1443_144397

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) : 
    avg_increase = 2.5 ∧ num_persons = 8 ∧ old_weight = 65 → 
    (old_weight + num_persons * avg_increase = 85) :=
by
  intro h
  sorry

end new_person_weight_l1443_144397


namespace combined_average_speed_l1443_144326

theorem combined_average_speed 
    (dA tA dB tB dC tC : ℝ)
    (mile_feet : ℝ)
    (hA : dA = 300) (hTA : tA = 6)
    (hB : dB = 400) (hTB : tB = 8)
    (hC : dC = 500) (hTC : tC = 10)
    (hMileFeet : mile_feet = 5280) :
    (1200 / 5280) / (24 / 3600) = 34.09 := 
by
  sorry

end combined_average_speed_l1443_144326


namespace probability_win_more_than_5000_l1443_144343

def boxes : Finset ℕ := {5, 500, 5000}
def keys : Finset (Finset ℕ) := { {5}, {500}, {5000} }

noncomputable def probability_correct_key (box : ℕ) : ℚ :=
  if box = 5000 then 1 / 3 else if box = 500 then 1 / 2 else 1

theorem probability_win_more_than_5000 :
    (probability_correct_key 5000) * (probability_correct_key 500) = 1 / 6 :=
by
  -- Proof is omitted
  sorry

end probability_win_more_than_5000_l1443_144343


namespace certain_percentage_l1443_144388

theorem certain_percentage (P : ℝ) : 
  0.15 * P * 0.50 * 4000 = 90 → P = 0.3 :=
by
  sorry

end certain_percentage_l1443_144388


namespace total_spent_l1443_144325

-- Constants representing the conditions from the problem
def cost_per_deck : ℕ := 8
def tom_decks : ℕ := 3
def friend_decks : ℕ := 5

-- Theorem stating the total amount spent by Tom and his friend
theorem total_spent : tom_decks * cost_per_deck + friend_decks * cost_per_deck = 64 := by
  sorry

end total_spent_l1443_144325


namespace cos_double_angle_tan_sum_angles_l1443_144393

variable (α β : ℝ)
variable (α_acute : 0 < α ∧ α < π / 2)
variable (β_acute : 0 < β ∧ β < π / 2)
variable (tan_alpha : Real.tan α = 4 / 3)
variable (sin_alpha_minus_beta : Real.sin (α - β) = - (Real.sqrt 5) / 5)

/- Prove that cos 2α = -7/25 given the conditions -/
theorem cos_double_angle :
  Real.cos (2 * α) = -7 / 25 :=
by
  sorry

/- Prove that tan (α + β) = -41/38 given the conditions -/
theorem tan_sum_angles :
  Real.tan (α + β) = -41 / 38 :=
by
  sorry

end cos_double_angle_tan_sum_angles_l1443_144393


namespace calculate_expression_l1443_144339

theorem calculate_expression :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = (3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end calculate_expression_l1443_144339


namespace find_correct_value_l1443_144399

theorem find_correct_value (k : ℕ) (h1 : 173 * 240 = 41520) (h2 : 41520 / 48 = 865) : k * 48 = 173 * 240 → k = 865 :=
by
  intros h
  sorry

end find_correct_value_l1443_144399


namespace pebble_difference_l1443_144370

-- Definitions and conditions
variables (x : ℚ) -- we use rational numbers for exact division
def Candy := 2 * x
def Lance := 5 * x
def Sandy := 4 * x
def condition1 := Lance = Candy + 10

-- Theorem statement
theorem pebble_difference (h : condition1) : Lance + Sandy - Candy = 30 :=
sorry

end pebble_difference_l1443_144370


namespace solution_set_l1443_144356
  
noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp (2 * x) + 1) - x

theorem solution_set (x : ℝ) :
  f (x + 2) > f (2 * x - 3) ↔ (1 / 3 < x ∧ x < 5) :=
by
  sorry

end solution_set_l1443_144356


namespace line_through_point_with_equal_intercepts_l1443_144311

/-- A line passing through point (-2, 3) and having equal intercepts
on the coordinate axes can have the equation y = -3/2 * x or x + y = 1. -/
theorem line_through_point_with_equal_intercepts (x y : Real) :
  (∃ (m : Real), (y = m * x) ∧ (y - m * (-2) = 3 ∧ y - m * 0 = 0))
  ∨ (∃ (a : Real), (x + y = a) ∧ (a = 1 ∧ (-2) + 3 = a)) :=
sorry

end line_through_point_with_equal_intercepts_l1443_144311


namespace inner_rectangle_length_l1443_144352

theorem inner_rectangle_length 
  (a b c : ℝ)
  (h1 : ∃ a1 a2 a3 : ℝ, a2 - a1 = a3 - a2)
  (w_inner : ℝ)
  (width_inner : w_inner = 2)
  (w_shaded : ℝ)
  (width_shaded : w_shaded = 1.5)
  (ar_prog : a = 2 * w_inner ∧ b = 3 * w_inner + 15 ∧ c = 3 * w_inner + 33)
  : ∀ x : ℝ, 2 * x = a → 3 * x + 15 = b → 3 * x + 33 = c → x = 3 :=
by
  sorry

end inner_rectangle_length_l1443_144352


namespace max_value_of_angle_B_l1443_144367

theorem max_value_of_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1: a + c = 2 * b)
  (h2: a^2 + b^2 - 2*a*b <= c^2 - 2*b*c - 2*a*c)
  (h3: A + B + C = π)
  (h4: 0 < A ∧ A < π) :  
  B ≤ π / 3 :=
sorry

end max_value_of_angle_B_l1443_144367


namespace solve_system_l1443_144308

theorem solve_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 7) : x + y = 5 :=
by
  sorry

end solve_system_l1443_144308


namespace dot_product_AB_BC_l1443_144364

theorem dot_product_AB_BC 
  (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a + c = 3)
  (cosB : ℝ)
  (h3 : cosB = 3 / 4) : 
  (a * c * (-cosB) = -3/2) :=
by 
  -- Given conditions
  sorry

end dot_product_AB_BC_l1443_144364


namespace maximum_value_of_m_l1443_144335

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end maximum_value_of_m_l1443_144335


namespace gym_distance_diff_l1443_144312

theorem gym_distance_diff (D G : ℕ) (hD : D = 10) (hG : G = 7) : G - D / 2 = 2 := by
  sorry

end gym_distance_diff_l1443_144312


namespace correct_number_of_paths_l1443_144385

-- Define the number of paths for each segment.
def paths_A_to_B : ℕ := 2
def paths_B_to_D : ℕ := 2
def paths_D_to_C : ℕ := 2
def direct_path_A_to_C : ℕ := 1

-- Define the function to calculate the total paths from A to C.
def total_paths_A_to_C : ℕ :=
  (paths_A_to_B * paths_B_to_D * paths_D_to_C) + direct_path_A_to_C

-- Prove that the total number of paths from A to C is 9.
theorem correct_number_of_paths : total_paths_A_to_C = 9 := by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end correct_number_of_paths_l1443_144385


namespace binary_101011_is_43_l1443_144307

def binary_to_decimal_conversion (b : Nat) : Nat := 
  match b with
  | 101011 => 43
  | _ => 0

theorem binary_101011_is_43 : binary_to_decimal_conversion 101011 = 43 := by
  sorry

end binary_101011_is_43_l1443_144307


namespace Shara_will_owe_money_l1443_144337

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end Shara_will_owe_money_l1443_144337


namespace pirates_share_l1443_144363

def initial_coins (N : ℕ) := N ≥ 3000 ∧ N ≤ 4000

def first_pirate (N : ℕ) := N - (2 + (N - 2) / 4)
def second_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def third_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def fourth_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)

def final_remaining (N : ℕ) :=
  let step1 := first_pirate N
  let step2 := second_pirate step1
  let step3 := third_pirate step2
  let step4 := fourth_pirate step3
  step4

theorem pirates_share (N : ℕ) (h : initial_coins N) :
  final_remaining N / 4 = 660 :=
by
  sorry

end pirates_share_l1443_144363


namespace smallest_hope_number_l1443_144376

def is_square (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k
def is_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k
def is_fifth_power (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k * k * k

def is_hope_number (n : ℕ) : Prop :=
  is_square (n / 8) ∧ is_cube (n / 9) ∧ is_fifth_power (n / 25)

theorem smallest_hope_number : ∃ n, is_hope_number n ∧ n = 2^15 * 3^20 * 5^12 :=
by
  sorry

end smallest_hope_number_l1443_144376


namespace subset_singleton_natural_l1443_144303

/-
  Problem Statement:
  Prove that the set {2} is a subset of the set of natural numbers.
-/

open Set

theorem subset_singleton_natural :
  {2} ⊆ (Set.univ : Set ℕ) :=
by
  sorry

end subset_singleton_natural_l1443_144303


namespace problem_solution_l1443_144395

noncomputable def alpha : ℝ := (3 + Real.sqrt 13) / 2
noncomputable def beta  : ℝ := (3 - Real.sqrt 13) / 2

theorem problem_solution : 7 * alpha ^ 4 + 10 * beta ^ 3 = 1093 :=
by
  -- Prove roots relation
  have hr1 : alpha * alpha - 3 * alpha - 1 = 0 := by sorry
  have hr2 : beta * beta - 3 * beta - 1 = 0 := by sorry
  -- Proceed to prove the required expression
  sorry

end problem_solution_l1443_144395


namespace find_roots_l1443_144333

theorem find_roots (x : ℝ) : x^2 - 2 * x - 2 / x + 1 / x^2 - 13 = 0 ↔ 
  (x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 ∨ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2) := by
  sorry

end find_roots_l1443_144333


namespace seonyeong_class_size_l1443_144371

theorem seonyeong_class_size :
  (12 * 4 + 3) - 12 = 39 :=
by
  sorry

end seonyeong_class_size_l1443_144371


namespace expected_winnings_correct_l1443_144383

def probability_1 := (1:ℚ) / 4
def probability_2 := (1:ℚ) / 4
def probability_3 := (1:ℚ) / 6
def probability_4 := (1:ℚ) / 6
def probability_5 := (1:ℚ) / 8
def probability_6 := (1:ℚ) / 8

noncomputable def expected_winnings : ℚ :=
  (probability_1 + probability_3 + probability_5) * 2 +
  (probability_2 + probability_4) * 4 +
  probability_6 * (-6 + 4)

theorem expected_winnings_correct : expected_winnings = 1.67 := by
  sorry

end expected_winnings_correct_l1443_144383


namespace printing_company_proportion_l1443_144322

theorem printing_company_proportion (x y : ℕ) :
  (28*x + 42*y) / (28*x) = 5/3 → x / y = 9 / 4 := by
  sorry

end printing_company_proportion_l1443_144322


namespace tan_beta_l1443_144319

noncomputable def tan_eq_2 (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) : Real :=
2

theorem tan_beta (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) :
  Real.tan β = tan_eq_2 α β h1 h2 := by
  sorry

end tan_beta_l1443_144319


namespace find_x_in_sequence_l1443_144320

theorem find_x_in_sequence :
  (∀ a b c d : ℕ, a * b * c * d = 120) →
  (a = 2) →
  (b = 4) →
  (d = 3) →
  ∃ x : ℕ, 2 * 4 * x * 3 = 120 ∧ x = 5 :=
sorry

end find_x_in_sequence_l1443_144320


namespace juan_original_number_l1443_144379

theorem juan_original_number (n : ℤ) 
  (h : ((2 * (n + 3) - 2) / 2) = 8) : 
  n = 6 := 
sorry

end juan_original_number_l1443_144379


namespace common_difference_l1443_144384

-- Define the arithmetic sequence with general term
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem common_difference (a₁ a₅ a₄ d : ℕ) 
  (h₁ : a₁ + a₅ = 10)
  (h₂ : a₄ = 7)
  (h₅ : a₅ = a₁ + 4 * d)
  (h₄ : a₄ = a₁ + 3 * d) :
  d = 2 :=
by
  sorry

end common_difference_l1443_144384


namespace sum_of_reciprocals_of_squares_roots_eq_14_3125_l1443_144309

theorem sum_of_reciprocals_of_squares_roots_eq_14_3125
  (α β γ : ℝ)
  (h1 : α + β + γ = 15)
  (h2 : α * β + β * γ + γ * α = 26)
  (h3 : α * β * γ = -8) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 14.3125 := 
by
  sorry

end sum_of_reciprocals_of_squares_roots_eq_14_3125_l1443_144309


namespace simplify_fraction_l1443_144366

-- Define the numerator and denominator
def numerator := 5^4 + 5^2
def denominator := 5^3 - 5

-- Define the simplified fraction
def simplified_fraction := 65 / 12

-- The proof problem statement
theorem simplify_fraction : (numerator / denominator) = simplified_fraction := 
by 
   -- Proof will go here
   sorry

end simplify_fraction_l1443_144366


namespace sum_of_first_6_terms_l1443_144316

-- Definitions based on given conditions
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)

-- The conditions provided in the problem
def condition_1 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 4
def condition_2 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 3 + arithmetic_sequence a1 d 5 = 10

-- The sum of the first 6 terms of the arithmetic sequence
def sum_first_6_terms (a1 d : ℤ) : ℤ := 6 * a1 + 15 * d

-- The theorem to prove
theorem sum_of_first_6_terms (a1 d : ℤ) 
  (h1 : condition_1 a1 d)
  (h2 : condition_2 a1 d) :
  sum_first_6_terms a1 d = 21 := sorry

end sum_of_first_6_terms_l1443_144316


namespace root_interval_k_l1443_144365

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_interval_k (k : ℤ) (h : ∃ ξ : ℝ, k < ξ ∧ ξ < k+1 ∧ f ξ = 0) : k = 0 :=
by
  sorry

end root_interval_k_l1443_144365


namespace graph_of_equation_is_two_lines_l1443_144389

theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, x^2 - 16*y^2 - 8*x + 16 = 0 ↔ (x = 4 + 4*y ∨ x = 4 - 4*y) :=
by
  sorry

end graph_of_equation_is_two_lines_l1443_144389


namespace points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l1443_144359

open Set

-- Define the point in the coordinate plane as a product of real numbers
def Point := ℝ × ℝ

-- Prove points with x = 3 form a vertical line
theorem points_on_x_eq_3_is_vertical_line : {p : Point | p.1 = 3} = {p : Point | ∀ y : ℝ, (3, y) = p} := sorry

-- Prove points with x < 3 lie to the left of x = 3
theorem points_with_x_lt_3 : {p : Point | p.1 < 3} = {p : Point | ∀ x y : ℝ, x < 3 → p = (x, y)} := sorry

-- Prove points with x > 3 lie to the right of x = 3
theorem points_with_x_gt_3 : {p : Point | p.1 > 3} = {p : Point | ∀ x y : ℝ, x > 3 → p = (x, y)} := sorry

-- Prove points with y = 2 form a horizontal line
theorem points_on_y_eq_2_is_horizontal_line : {p : Point | p.2 = 2} = {p : Point | ∀ x : ℝ, (x, 2) = p} := sorry

-- Prove points with y > 2 lie above y = 2
theorem points_with_y_gt_2 : {p : Point | p.2 > 2} = {p : Point | ∀ x y : ℝ, y > 2 → p = (x, y)} := sorry

end points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l1443_144359


namespace vertices_of_cube_l1443_144392

-- Given condition: geometric shape is a cube
def is_cube (x : Type) : Prop := true -- This is a placeholder declaration that x is a cube.

-- Question: How many vertices does a cube have?
-- Proof problem: Prove that the number of vertices of a cube is 8.
theorem vertices_of_cube (x : Type) (h : is_cube x) : true := 
  sorry

end vertices_of_cube_l1443_144392


namespace min_dot_product_l1443_144300

-- Define the conditions of the ellipse and focal points
variables (P : ℝ × ℝ)
def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define vectors
def OP (P : ℝ × ℝ) : ℝ × ℝ := P
def FP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 + 1, P.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Prove that the minimum value of the dot product is 2
theorem min_dot_product (hP : ellipse P.1 P.2) : 
  ∃ (P : ℝ × ℝ), dot_product (OP P) (FP P) = 2 := sorry

end min_dot_product_l1443_144300


namespace race_distance_l1443_144334

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end race_distance_l1443_144334


namespace ball_hits_ground_approx_time_l1443_144369

-- Conditions
def height (t : ℝ) : ℝ := -6.1 * t^2 + 4.5 * t + 10

-- Main statement to be proved
theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, (height t = 0) ∧ (abs (t - 1.70) < 0.01) :=
sorry

end ball_hits_ground_approx_time_l1443_144369


namespace inequality_transformation_l1443_144361

theorem inequality_transformation (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by {
  sorry
}

end inequality_transformation_l1443_144361


namespace max_value_cos2_sin_l1443_144318

noncomputable def max_cos2_sin (x : Real) : Real := 
  (Real.cos x) ^ 2 + Real.sin x

theorem max_value_cos2_sin : 
  ∃ x : Real, (-1 ≤ Real.sin x) ∧ (Real.sin x ≤ 1) ∧ 
    max_cos2_sin x = 5 / 4 :=
sorry

end max_value_cos2_sin_l1443_144318


namespace frac_val_of_x_y_l1443_144346

theorem frac_val_of_x_y (x y : ℝ) (h: (4 : ℝ) < (2 * x - 3 * y) / (2 * x + 3 * y) ∧ (2 * x - 3 * y) / (2 * x + 3 * y) < 8) (ht: ∃ t : ℤ, x = t * y) : x / y = -2 := 
by
  sorry

end frac_val_of_x_y_l1443_144346


namespace minimum_length_of_segment_PQ_l1443_144398

theorem minimum_length_of_segment_PQ:
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y + 1 = 0) → 
              (xy >= 2) → 
              (x - y >= 0) → 
              (y <= 1) → 
              ℝ) :=
sorry

end minimum_length_of_segment_PQ_l1443_144398


namespace intersection_points_eq_2_l1443_144328

def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
def eq2 (x y : ℝ) : Prop := (x + 2 * y - 3) * (3 * x - 4 * y + 6) = 0

theorem intersection_points_eq_2 : ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 2 := 
sorry

end intersection_points_eq_2_l1443_144328


namespace min_val_of_3x_add_4y_l1443_144386

theorem min_val_of_3x_add_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 
  (3 * x + 4 * y ≥ 5) ∧ (3 * x + 4 * y = 5 → x + 4 * y = 3) := 
by
  sorry

end min_val_of_3x_add_4y_l1443_144386


namespace smallest_constant_N_l1443_144368

-- Given that a, b, c are sides of a triangle and in arithmetic progression, prove that
-- (a^2 + b^2 + c^2) / (ab + bc + ca) ≥ 1.

theorem smallest_constant_N
  (a b c : ℝ)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality
  (hap : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d) -- Arithmetic progression
  : (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ≥ 1 := 
sorry

end smallest_constant_N_l1443_144368


namespace three_million_times_three_million_l1443_144301

theorem three_million_times_three_million : 
  (3 * 10^6) * (3 * 10^6) = 9 * 10^12 := 
by
  sorry

end three_million_times_three_million_l1443_144301


namespace jennie_total_rental_cost_l1443_144350

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end jennie_total_rental_cost_l1443_144350


namespace min_value_u_l1443_144358

theorem min_value_u (x y : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0)
  (h₂ : 2 * x + y = 6) : 
  ∀u, u = 4 * x ^ 2 + 3 * x * y + y ^ 2 - 6 * x - 3 * y -> 
  u ≥ 27 / 2 := sorry

end min_value_u_l1443_144358


namespace coin_difference_l1443_144378

theorem coin_difference : 
  ∀ (c : ℕ), c = 50 → 
  (∃ (n m : ℕ), 
    (n ≥ m) ∧ 
    (∃ (a b d e : ℕ), n = a + b + d + e ∧ 5 * a + 10 * b + 20 * d + 25 * e = c) ∧
    (∃ (p q r s : ℕ), m = p + q + r + s ∧ 5 * p + 10 * q + 20 * r + 25 * s = c) ∧ 
    (n - m = 8)) :=
by
  sorry

end coin_difference_l1443_144378


namespace distance_ratio_l1443_144321

theorem distance_ratio (D90 D180 : ℝ) 
  (h1 : D90 + D180 = 3600) 
  (h2 : D90 / 90 + D180 / 180 = 30) : 
  D90 / D180 = 1 := 
by 
  sorry

end distance_ratio_l1443_144321


namespace number_of_discounted_tickets_l1443_144390

def total_tickets : ℕ := 10
def full_price_ticket_cost : ℝ := 2.0
def discounted_ticket_cost : ℝ := 1.6
def total_spent : ℝ := 18.40

theorem number_of_discounted_tickets (F D : ℕ) : 
    F + D = total_tickets → 
    full_price_ticket_cost * ↑F + discounted_ticket_cost * ↑D = total_spent → 
    D = 4 :=
by
  intros h1 h2
  sorry

end number_of_discounted_tickets_l1443_144390


namespace value_of_expression_l1443_144348

def a : ℕ := 7
def b : ℕ := 5

theorem value_of_expression : (a^2 - b^2)^4 = 331776 := by
  sorry

end value_of_expression_l1443_144348


namespace log_comparison_l1443_144306

theorem log_comparison 
  (a : ℝ := 1 / 6 * Real.log 8)
  (b : ℝ := 1 / 2 * Real.log 5)
  (c : ℝ := Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := 
by
  sorry

end log_comparison_l1443_144306
