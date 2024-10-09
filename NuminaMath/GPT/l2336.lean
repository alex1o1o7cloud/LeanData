import Mathlib

namespace vanessas_mother_picked_14_carrots_l2336_233609

-- Define the problem parameters
variable (V : Nat := 17)  -- Vanessa picked 17 carrots
variable (G : Nat := 24)  -- Total good carrots
variable (B : Nat := 7)   -- Total bad carrots

-- Define the proof goal: Vanessa's mother picked 14 carrots
theorem vanessas_mother_picked_14_carrots : (G + B) - V = 14 := by
  sorry

end vanessas_mother_picked_14_carrots_l2336_233609


namespace gcd_102_238_eq_34_l2336_233667

theorem gcd_102_238_eq_34 :
  Int.gcd 102 238 = 34 :=
sorry

end gcd_102_238_eq_34_l2336_233667


namespace number_of_beakers_calculation_l2336_233680

-- Conditions
def solution_per_test_tube : ℕ := 7
def number_of_test_tubes : ℕ := 6
def solution_per_beaker : ℕ := 14

-- Total amount of solution
def total_solution : ℕ := solution_per_test_tube * number_of_test_tubes

-- Number of beakers is the fraction of total solution and solution per beaker
def number_of_beakers : ℕ := total_solution / solution_per_beaker

-- Statement of the problem
theorem number_of_beakers_calculation : number_of_beakers = 3 :=
by 
  -- Proof goes here
  sorry

end number_of_beakers_calculation_l2336_233680


namespace car_cost_l2336_233668

/--
A group of six friends planned to buy a car. They plan to share the cost equally. 
They had a car wash to help raise funds, which would be taken out of the total cost. 
The remaining cost would be split between the six friends. At the car wash, they earn $500. 
However, Brad decided not to join in the purchase of the car, and now each friend has to pay $40 more. 
What is the cost of the car?
-/
theorem car_cost 
  (C : ℝ) 
  (h1 : 6 * ((C - 500) / 5) = 5 * (C / 6 + 40)) : 
  C = 4200 := 
by 
  sorry

end car_cost_l2336_233668


namespace cube_colorings_distinguishable_l2336_233620

-- Define the problem
def cube_construction_distinguishable_ways : Nat :=
  30

-- The theorem we need to prove
theorem cube_colorings_distinguishable :
  ∃ (ways : Nat), ways = cube_construction_distinguishable_ways :=
by
  sorry

end cube_colorings_distinguishable_l2336_233620


namespace count_valid_A_l2336_233694

theorem count_valid_A : 
  ∃! (count : ℕ), count = 4 ∧ ∀ A : ℕ, (1 ≤ A ∧ A ≤ 9) → 
  (∃ x1 x2 : ℕ, x1 + x2 = 2 * A + 1 ∧ x1 * x2 = 2 * A ∧ x1 > 0 ∧ x2 > 0) → A = 1 ∨ A = 2 ∨ A = 3 ∨ A = 4 :=
sorry

end count_valid_A_l2336_233694


namespace foma_should_give_ierema_55_coins_l2336_233601

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l2336_233601


namespace evaluate_expression_l2336_233640

theorem evaluate_expression :
  - (20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 :=
by
  sorry

end evaluate_expression_l2336_233640


namespace find_width_of_chalkboard_l2336_233684

variable (w : ℝ) (l : ℝ)

-- Given conditions
def length_eq_twice_width (w l : ℝ) : Prop := l = 2 * w
def area_eq_eighteen (w l : ℝ) : Prop := w * l = 18

-- Theorem statement
theorem find_width_of_chalkboard (h1 : length_eq_twice_width w l) (h2 : area_eq_eighteen w l) : w = 3 :=
by sorry

end find_width_of_chalkboard_l2336_233684


namespace line_intersects_y_axis_at_l2336_233615

-- Define the two points the line passes through
structure Point (α : Type) :=
(x : α)
(y : α)

def p1 : Point ℤ := Point.mk 2 9
def p2 : Point ℤ := Point.mk 4 13

-- Define the function that describes the point where the line intersects the y-axis
def y_intercept : Point ℤ :=
  -- We are proving that the line intersects the y-axis at the point (0, 5)
  Point.mk 0 5

-- State the theorem to be proven
theorem line_intersects_y_axis_at (p1 p2 : Point ℤ) (yi : Point ℤ) :
  p1.x = 2 ∧ p1.y = 9 ∧ p2.x = 4 ∧ p2.y = 13 → yi = Point.mk 0 5 :=
by
  intros
  sorry

end line_intersects_y_axis_at_l2336_233615


namespace initial_price_of_phone_l2336_233656

theorem initial_price_of_phone (P : ℝ) (h : 0.20 * P = 480) : P = 2400 :=
sorry

end initial_price_of_phone_l2336_233656


namespace number_of_customers_who_did_not_want_tires_change_l2336_233626

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end number_of_customers_who_did_not_want_tires_change_l2336_233626


namespace jellybean_total_l2336_233611

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l2336_233611


namespace time_worked_on_thursday_l2336_233659

/-
  Given:
  - Monday: 3/4 hour
  - Tuesday: 1/2 hour
  - Wednesday: 2/3 hour
  - Friday: 75 minutes
  - Total (Monday to Friday): 4 hours = 240 minutes
  
  The time Mr. Willson worked on Thursday is 50 minutes.
-/

noncomputable def time_worked_monday : ℝ := (3 / 4) * 60
noncomputable def time_worked_tuesday : ℝ := (1 / 2) * 60
noncomputable def time_worked_wednesday : ℝ := (2 / 3) * 60
noncomputable def time_worked_friday : ℝ := 75
noncomputable def total_time_worked : ℝ := 4 * 60

theorem time_worked_on_thursday :
  time_worked_monday + time_worked_tuesday + time_worked_wednesday + time_worked_friday + 50 = total_time_worked :=
by
  sorry

end time_worked_on_thursday_l2336_233659


namespace algebraic_expression_value_l2336_233652

theorem algebraic_expression_value 
  (x1 x2 : ℝ)
  (h1 : x1^2 - x1 - 2022 = 0)
  (h2 : x2^2 - x2 - 2022 = 0) :
  x1^3 - 2022 * x1 + x2^2 = 4045 :=
by 
  sorry

end algebraic_expression_value_l2336_233652


namespace least_positive_three_digit_multiple_of_8_l2336_233657

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l2336_233657


namespace sum_of_eighth_powers_of_roots_l2336_233633

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let root_disc := Real.sqrt discriminant
  ((-b + root_disc) / (2 * a), (-b - root_disc) / (2 * a))

theorem sum_of_eighth_powers_of_roots :
  let (p, q) := quadratic_roots 1 (-Real.sqrt 7) 1
  p^2 + q^2 = 5 ∧ p^4 + q^4 = 23 ∧ p^8 + q^8 = 527 :=
by
  sorry

end sum_of_eighth_powers_of_roots_l2336_233633


namespace sum_of_arithmetic_sequence_2008_terms_l2336_233653

theorem sum_of_arithmetic_sequence_2008_terms :
  let a := -1776
  let d := 11
  let n := 2008
  let l := a + (n - 1) * d
  let S := (n / 2) * (a + l)
  S = 18599100 := by
  sorry

end sum_of_arithmetic_sequence_2008_terms_l2336_233653


namespace parabola_focus_hyperbola_equation_l2336_233665

-- Problem 1
theorem parabola_focus (p : ℝ) (h₀ : p > 0) (h₁ : 2 * p - 0 - 4 = 0) : p = 2 :=
by
  sorry

-- Problem 2
theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : b / a = 3 / 4) (h₃ : a^2 / a = 16 / 5) (h₄ : a^2 + b^2 = 1) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
by
  sorry

end parabola_focus_hyperbola_equation_l2336_233665


namespace minimum_pipe_length_l2336_233632

theorem minimum_pipe_length 
  (M S : ℝ × ℝ) 
  (horiz_dist : abs (M.1 - S.1) = 160)
  (vert_dist : abs (M.2 - S.2) = 120) :
  dist M S = 200 :=
by {
  sorry
}

end minimum_pipe_length_l2336_233632


namespace doughnuts_remaining_l2336_233634

theorem doughnuts_remaining 
  (total_doughnuts : ℕ)
  (total_staff : ℕ)
  (staff_3_doughnuts : ℕ)
  (doughnuts_eaten_by_3 : ℕ)
  (staff_2_doughnuts : ℕ)
  (doughnuts_eaten_by_2 : ℕ)
  (staff_4_doughnuts : ℕ)
  (doughnuts_eaten_by_4 : ℕ) :
  total_doughnuts = 120 →
  total_staff = 35 →
  staff_3_doughnuts = 15 →
  staff_2_doughnuts = 10 →
  doughnuts_eaten_by_3 = staff_3_doughnuts * 3 →
  doughnuts_eaten_by_2 = staff_2_doughnuts * 2 →
  staff_4_doughnuts = total_staff - (staff_3_doughnuts + staff_2_doughnuts) →
  doughnuts_eaten_by_4 = staff_4_doughnuts * 4 →
  total_doughnuts - (doughnuts_eaten_by_3 + doughnuts_eaten_by_2 + doughnuts_eaten_by_4) = 15 :=
by
  intros
  -- Proof goes here
  sorry

end doughnuts_remaining_l2336_233634


namespace m_and_n_relationship_l2336_233643

-- Define the function f
def f (x m : ℝ) := x^2 - 4*x + 4 + m

-- State the conditions and required proof
theorem m_and_n_relationship (m n : ℝ) (h_domain : ∀ x, 2 ≤ x ∧ x ≤ n → 2 ≤ f x m ∧ f x m ≤ n) :
  m^n = 8 :=
by
  -- Placeholder for the actual proof
  sorry

end m_and_n_relationship_l2336_233643


namespace length_of_segment_cutoff_l2336_233603

-- Define the parabola equation
def parabola (x y : ℝ) := y^2 = 4 * (x + 1)

-- Define the line passing through the focus and perpendicular to the x-axis
def line_through_focus_perp_x_axis (x y : ℝ) := x = 0

-- The actual length calculation lemma
lemma segment_length : 
  ∀ (x y : ℝ), parabola x y → line_through_focus_perp_x_axis x y → y = 2 ∨ y = -2 :=
by sorry

-- The final theorem which gives the length of the segment
theorem length_of_segment_cutoff (y1 y2 : ℝ) :
  ∀ (x : ℝ), parabola x y1 → parabola x y2 → line_through_focus_perp_x_axis x y1 → line_through_focus_perp_x_axis x y2 → (y1 = 2 ∨ y1 = -2) ∧ (y2 = 2 ∨ y2 = -2) → abs (y2 - y1) = 4 :=
by sorry

end length_of_segment_cutoff_l2336_233603


namespace exponentiation_division_l2336_233663

variable (a b : ℝ)

theorem exponentiation_division (a b : ℝ) : ((2 * a) / b) ^ 4 = (16 * a ^ 4) / (b ^ 4) := by
  sorry

end exponentiation_division_l2336_233663


namespace population_net_increase_l2336_233629

-- Definitions for birth and death rate, and the number of seconds in a day
def birth_rate : ℕ := 10
def death_rate : ℕ := 2
def seconds_in_day : ℕ := 86400

-- Calculate the population net increase in one day
theorem population_net_increase (birth_rate death_rate seconds_in_day : ℕ) :
  (seconds_in_day / 2) * birth_rate - (seconds_in_day / 2) * death_rate = 345600 :=
by
  sorry

end population_net_increase_l2336_233629


namespace rectangle_sides_l2336_233693

theorem rectangle_sides (a b : ℝ) (h1 : a < b) (h2 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 :=
by
  sorry

end rectangle_sides_l2336_233693


namespace log_relation_l2336_233607

theorem log_relation (a b : ℝ) (log_7 : ℝ → ℝ) (log_6 : ℝ → ℝ) (log_6_343 : log_6 343 = a) (log_7_18 : log_7 18 = b) :
  a = 6 / (b + 2 * log_7 2) :=
by
  sorry

end log_relation_l2336_233607


namespace polygon_sides_l2336_233636

theorem polygon_sides (interior_angle: ℝ) (sum_exterior_angles: ℝ) (n: ℕ) (h: interior_angle = 108) (h1: sum_exterior_angles = 360): n = 5 :=
by 
  sorry

end polygon_sides_l2336_233636


namespace ratio_Lisa_Charlotte_l2336_233638

def P_tot : ℕ := 100
def Pat_money : ℕ := 6
def Lisa_money : ℕ := 5 * Pat_money
def additional_required : ℕ := 49
def current_total_money : ℕ := P_tot - additional_required
def Pat_Lisa_total : ℕ := Pat_money + Lisa_money
def Charlotte_money : ℕ := current_total_money - Pat_Lisa_total

theorem ratio_Lisa_Charlotte : (Lisa_money : ℕ) / Charlotte_money = 2 :=
by
  -- Proof to be filled in later
  sorry

end ratio_Lisa_Charlotte_l2336_233638


namespace max_value_f_period_f_l2336_233697

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - (Real.cos x) ^ 4

theorem max_value_f : ∃ x : ℝ, (f x) = 1 / 4 :=
sorry

theorem period_f : ∃ p : ℝ, p = π / 2 ∧ ∀ x : ℝ, f (x + p) = f x :=
sorry

end max_value_f_period_f_l2336_233697


namespace apples_preference_count_l2336_233627

theorem apples_preference_count (total_people : ℕ) (total_angle : ℝ) (apple_angle : ℝ) 
  (h_total_people : total_people = 530) 
  (h_total_angle : total_angle = 360) 
  (h_apple_angle : apple_angle = 285) : 
  round ((total_people : ℝ) * (apple_angle / total_angle)) = 419 := 
by 
  sorry

end apples_preference_count_l2336_233627


namespace problem_statement_l2336_233621

theorem problem_statement :
  (81000 ^ 3) / (27000 ^ 3) = 27 :=
by sorry

end problem_statement_l2336_233621


namespace min_marked_cells_l2336_233696

theorem min_marked_cells (marking : Fin 15 → Fin 15 → Prop) :
  (∀ i : Fin 15, ∃ j : Fin 15, ∀ k : Fin 10, marking i (j + k % 15)) ∧
  (∀ j : Fin 15, ∃ i : Fin 15, ∀ k : Fin 10, marking (i + k % 15) j) →
  ∃s : Finset (Fin 15 × Fin 15), s.card = 20 ∧ ∀ i : Fin 15, (∃ j, (i, j) ∈ s ∨ (j, i) ∈ s) :=
sorry

end min_marked_cells_l2336_233696


namespace nine_div_one_plus_four_div_x_eq_one_l2336_233613

theorem nine_div_one_plus_four_div_x_eq_one (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end nine_div_one_plus_four_div_x_eq_one_l2336_233613


namespace dissimilar_terms_expansion_count_l2336_233681

noncomputable def num_dissimilar_terms_in_expansion (a b c d : ℝ) : ℕ :=
  let n := 8
  let k := 4
  Nat.choose (n + k - 1) (k - 1)

theorem dissimilar_terms_expansion_count : 
  num_dissimilar_terms_in_expansion a b c d = 165 := by
  sorry

end dissimilar_terms_expansion_count_l2336_233681


namespace chris_leftover_money_l2336_233690

def chris_will_have_leftover : Prop :=
  let video_game_cost := 60
  let candy_cost := 5
  let hourly_wage := 8
  let hours_worked := 9
  let total_earned := hourly_wage * hours_worked
  let total_cost := video_game_cost + candy_cost
  let leftover := total_earned - total_cost
  leftover = 7

theorem chris_leftover_money : chris_will_have_leftover := 
  by
    sorry

end chris_leftover_money_l2336_233690


namespace smallest_number_of_cubes_l2336_233614

noncomputable def container_cubes (length_ft : ℕ) (height_ft : ℕ) (width_ft : ℕ) (prime_inch : ℕ) : ℕ :=
  let length_inch := length_ft * 12
  let height_inch := height_ft * 12
  let width_inch := width_ft * 12
  (length_inch / prime_inch) * (height_inch / prime_inch) * (width_inch / prime_inch)

theorem smallest_number_of_cubes :
  container_cubes 60 24 30 3 = 2764800 :=
by
  sorry

end smallest_number_of_cubes_l2336_233614


namespace missed_angle_l2336_233699

theorem missed_angle (n : ℕ) (h1 : (n - 2) * 180 ≥ 3239) (h2 : n ≥ 3) : 3240 - 3239 = 1 :=
by
  sorry

end missed_angle_l2336_233699


namespace boat_speed_in_still_water_l2336_233622

theorem boat_speed_in_still_water
  (V_s : ℝ) (t : ℝ) (d : ℝ) (V_b : ℝ)
  (h_stream_speed : V_s = 4)
  (h_travel_time : t = 7)
  (h_distance : d = 196)
  (h_downstream_speed : d / t = V_b + V_s) :
  V_b = 24 :=
by
  sorry

end boat_speed_in_still_water_l2336_233622


namespace angle_B_sum_a_c_l2336_233612

theorem angle_B (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B) :
  B = π / 3 :=
  sorry

theorem sum_a_c (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B)
  (hB : B = π / 3) :
  a + c = Real.sqrt 15 :=
  sorry

end angle_B_sum_a_c_l2336_233612


namespace money_left_after_purchase_l2336_233635

def initial_money : ℕ := 7
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := 3

def total_spent : ℕ := cost_candy_bar + cost_chocolate
def money_left : ℕ := initial_money - total_spent

theorem money_left_after_purchase : 
  money_left = 2 := by
  sorry

end money_left_after_purchase_l2336_233635


namespace min_value_of_expression_l2336_233648

noncomputable def minValueExpr (a b c : ℝ) : ℝ :=
  a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem min_value_of_expression (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minValueExpr a b c >= 60 :=
by
  sorry

end min_value_of_expression_l2336_233648


namespace cement_used_tess_street_l2336_233605

-- Define the given conditions
def cement_used_lexi_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Define the statement to prove the amount of cement used for Tess's street
theorem cement_used_tess_street : total_cement_used - cement_used_lexi_street = 5.1 :=
by
  sorry

end cement_used_tess_street_l2336_233605


namespace abs_AB_l2336_233606

noncomputable def ellipse_foci (A B : ℝ) : Prop :=
  B^2 - A^2 = 25

noncomputable def hyperbola_foci (A B : ℝ) : Prop :=
  A^2 + B^2 = 64

theorem abs_AB (A B : ℝ) (h1 : ellipse_foci A B) (h2 : hyperbola_foci A B) :
  |A * B| = Real.sqrt 867.75 := 
sorry

end abs_AB_l2336_233606


namespace line_circle_intersections_l2336_233647

-- Define the line equation as a predicate
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- The goal is to prove the number of intersections of the line and the circle
theorem line_circle_intersections : (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧ 
                                   (∃ x y : ℝ, line_eq x y ∧ circle_eq x y ∧ x ≠ y) :=
sorry

end line_circle_intersections_l2336_233647


namespace cards_given_l2336_233655

def initial_cards : ℕ := 304
def remaining_cards : ℕ := 276
def given_cards : ℕ := initial_cards - remaining_cards

theorem cards_given :
  given_cards = 28 :=
by
  unfold given_cards
  unfold initial_cards
  unfold remaining_cards
  sorry

end cards_given_l2336_233655


namespace chess_tournament_games_l2336_233617

theorem chess_tournament_games (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_tournament_games_l2336_233617


namespace reciprocal_of_neg_1_point_5_l2336_233602

theorem reciprocal_of_neg_1_point_5 : (1 / (-1.5) = -2 / 3) :=
by
  sorry

end reciprocal_of_neg_1_point_5_l2336_233602


namespace xy_sum_is_one_l2336_233679

theorem xy_sum_is_one (x y : ℤ) (h1 : 2021 * x + 2025 * y = 2029) (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 :=
by sorry

end xy_sum_is_one_l2336_233679


namespace VincentLearnedAtCamp_l2336_233623

def VincentSongsBeforeSummerCamp : ℕ := 56
def VincentSongsAfterSummerCamp : ℕ := 74

theorem VincentLearnedAtCamp :
  VincentSongsAfterSummerCamp - VincentSongsBeforeSummerCamp = 18 := by
  sorry

end VincentLearnedAtCamp_l2336_233623


namespace vertical_asymptote_c_values_l2336_233637

theorem vertical_asymptote_c_values (c : ℝ) :
  (∃ x : ℝ, (x^2 - x - 6) = 0 ∧ (x^2 - 2*x + c) ≠ 0 ∧ ∀ y : ℝ, ((y ≠ x) → (x ≠ 3) ∧ (x ≠ -2)))
  → (c = -3 ∨ c = -8) :=
by sorry

end vertical_asymptote_c_values_l2336_233637


namespace Kishore_misc_expense_l2336_233618

theorem Kishore_misc_expense:
  let savings := 2400
  let percent_saved := 0.10
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let total_salary := savings / percent_saved 
  let total_spent := rent + milk + groceries + education + petrol
  total_salary - (total_spent + savings) = 6100 := 
by
  sorry

end Kishore_misc_expense_l2336_233618


namespace correct_calculation_l2336_233666

variable (a b : ℝ)

theorem correct_calculation : ((-a^2)^3 = -a^6) :=
by sorry

end correct_calculation_l2336_233666


namespace joy_can_choose_17_rods_for_quadrilateral_l2336_233650

theorem joy_can_choose_17_rods_for_quadrilateral :
  ∃ (possible_rods : Finset ℕ), 
    possible_rods.card = 17 ∧
    ∀ rod ∈ possible_rods, 
      rod > 0 ∧ rod <= 30 ∧
      (rod ≠ 3 ∧ rod ≠ 7 ∧ rod ≠ 15) ∧
      (rod > 15 - (3 + 7)) ∧
      (rod < 3 + 7 + 15) :=
by
  sorry

end joy_can_choose_17_rods_for_quadrilateral_l2336_233650


namespace veranda_area_l2336_233619

theorem veranda_area (room_length room_width veranda_length_width veranda_width_width : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_width = 2.5)
  (h4 : veranda_width_width = 3)
  : (room_length + 2 * veranda_length_width) * (room_width + 2 * veranda_width_width) - room_length * room_width = 204 :=
by
  simp [h1, h2, h3, h4]
  norm_num
  done

end veranda_area_l2336_233619


namespace min_value_of_fraction_sum_l2336_233686

theorem min_value_of_fraction_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 :=
sorry

end min_value_of_fraction_sum_l2336_233686


namespace marks_lost_per_wrong_answer_l2336_233671

theorem marks_lost_per_wrong_answer
    (total_questions : ℕ)
    (correct_questions : ℕ)
    (total_marks : ℕ)
    (marks_per_correct : ℕ)
    (marks_lost : ℕ)
    (x : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_questions = 44)
    (h3 : total_marks = 160)
    (h4 : marks_per_correct = 4)
    (h5 : marks_lost = 176 - total_marks)
    (h6 : marks_lost = x * (total_questions - correct_questions)) :
    x = 1 := by
  sorry

end marks_lost_per_wrong_answer_l2336_233671


namespace sandy_spent_on_repairs_l2336_233651

theorem sandy_spent_on_repairs (initial_cost : ℝ) (selling_price : ℝ) (gain_percent : ℝ) (repair_cost : ℝ) :
  initial_cost = 800 → selling_price = 1400 → gain_percent = 40 → selling_price = 1.4 * (initial_cost + repair_cost) → repair_cost = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_spent_on_repairs_l2336_233651


namespace remaining_tickets_l2336_233677

-- Define initial tickets and used tickets
def initial_tickets := 13
def used_tickets := 6

-- Declare the theorem we want to prove
theorem remaining_tickets (initial_tickets used_tickets : ℕ) (h1 : initial_tickets = 13) (h2 : used_tickets = 6) : initial_tickets - used_tickets = 7 :=
by
  sorry

end remaining_tickets_l2336_233677


namespace ratio_of_goals_l2336_233630

-- The conditions
def first_period_goals_kickers : ℕ := 2
def second_period_goals_kickers := 4
def first_period_goals_spiders := first_period_goals_kickers / 2
def second_period_goals_spiders := 2 * second_period_goals_kickers
def total_goals := first_period_goals_kickers + second_period_goals_kickers + first_period_goals_spiders + second_period_goals_spiders

-- The ratio to prove
def ratio_goals : ℕ := second_period_goals_kickers / first_period_goals_kickers

theorem ratio_of_goals : total_goals = 15 → ratio_goals = 2 := by
  intro h
  sorry

end ratio_of_goals_l2336_233630


namespace unique_solution_l2336_233624

theorem unique_solution (a b : ℤ) : 
  (a^6 + 1 ∣ b^11 - 2023 * b^3 + 40 * b) ∧ (a^4 - 1 ∣ b^10 - 2023 * b^2 - 41) 
  ↔ (a = 0 ∧ ∃ c : ℤ, b = c) := 
by 
  sorry

end unique_solution_l2336_233624


namespace circle_tangent_line_l2336_233664

theorem circle_tangent_line 
    (center : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) 
    (tangent_eq : ℝ) :
    center = (-1, 1) →
    line_eq 1 (-1)= 0 →
    tangent_eq = 2 :=
  let h := -1;
  let k := 1;
  let radius := Real.sqrt 2;
  sorry

end circle_tangent_line_l2336_233664


namespace arithmetic_sequence_properties_l2336_233673

noncomputable def arithmetic_sequence (a3 a5_a7_sum : ℝ) : Prop :=
  ∃ (a d : ℝ), a + 2*d = a3 ∧ 2*a + 10*d = a5_a7_sum

noncomputable def sequence_a_n (a d n : ℝ) : ℝ := a + (n - 1)*d

noncomputable def sum_S_n (a d n : ℝ) : ℝ := n/2 * (2*a + (n-1)*d)

noncomputable def sequence_b_n (a d n : ℝ) : ℝ := 1 / (sequence_a_n a d n ^ 2 - 1)

noncomputable def sum_T_n (a d n : ℝ) : ℝ :=
  (1 / 4) * (1 - 1/(n+1))

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 7 26) →
  (∀ n : ℕ+, sequence_a_n 3 2 n = 2 * n + 1) ∧
  (∀ n : ℕ+, sum_S_n 3 2 n = n^2 + 2 * n) ∧
  (∀ n : ℕ+, sum_T_n 3 2 n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_properties_l2336_233673


namespace no_preimage_iff_k_less_than_neg2_l2336_233662

theorem no_preimage_iff_k_less_than_neg2 (k : ℝ) :
  ¬∃ x : ℝ, x^2 - 2 * x - 1 = k ↔ k < -2 :=
sorry

end no_preimage_iff_k_less_than_neg2_l2336_233662


namespace simplify_sqrt7_pow6_l2336_233639

theorem simplify_sqrt7_pow6 : (Real.sqrt 7)^6 = (343 : Real) :=
by 
  -- we'll fill in the proof later
  sorry

end simplify_sqrt7_pow6_l2336_233639


namespace find_m_l2336_233692

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 :=
by
  intros h
  sorry

end find_m_l2336_233692


namespace total_spent_on_clothing_l2336_233641

-- Define the individual costs
def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

-- Define the proof problem to show the total cost
theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  sorry

end total_spent_on_clothing_l2336_233641


namespace side_length_of_square_l2336_233669

theorem side_length_of_square (s : ℝ) (h : s^2 = 2 * (4 * s)) : s = 8 := 
by
  sorry

end side_length_of_square_l2336_233669


namespace inequality_2_pow_ge_n_sq_l2336_233670

theorem inequality_2_pow_ge_n_sq (n : ℕ) (hn : n ≠ 3) : 2^n ≥ n^2 :=
sorry

end inequality_2_pow_ge_n_sq_l2336_233670


namespace S_is_positive_rationals_l2336_233644

variable {S : Set ℚ}

-- Defining the conditions as axioms
axiom cond1 (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : (a + b ∈ S) ∧ (a * b ∈ S)
axiom cond2 {r : ℚ} : (r ∈ S) ∨ (-r ∈ S) ∨ (r = 0)

-- The theorem to prove
theorem S_is_positive_rationals : S = { r : ℚ | r > 0 } := sorry

end S_is_positive_rationals_l2336_233644


namespace seeds_in_pots_l2336_233660

theorem seeds_in_pots (x : ℕ) (total_seeds : ℕ) (seeds_fourth_pot : ℕ) 
  (h1 : total_seeds = 10) (h2 : seeds_fourth_pot = 1) 
  (h3 : 3 * x + seeds_fourth_pot = total_seeds) : x = 3 :=
by
  sorry

end seeds_in_pots_l2336_233660


namespace ThaboRatio_l2336_233678

-- Define the variables
variables (P_f P_nf H_nf : ℕ)

-- Define the conditions as hypotheses
def ThaboConditions := P_f + P_nf + H_nf = 280 ∧ P_nf = H_nf + 20 ∧ H_nf = 55

-- State the theorem we want to prove
theorem ThaboRatio (h : ThaboConditions P_f P_nf H_nf) : (P_f / P_nf) = 2 :=
by sorry

end ThaboRatio_l2336_233678


namespace remainder_of_sum_l2336_233658

theorem remainder_of_sum (a b c : ℕ) (h1 : a % 15 = 8) (h2 : b % 15 = 12) (h3 : c % 15 = 13) : (a + b + c) % 15 = 3 := 
by
  sorry

end remainder_of_sum_l2336_233658


namespace ice_cream_ratio_l2336_233608

theorem ice_cream_ratio
    (T : ℕ)
    (W : ℕ)
    (hT : T = 12000)
    (hMultiple : ∃ k : ℕ, W = k * T)
    (hTotal : T + W = 36000) :
    W / T = 2 :=
by
  -- Proof is omitted, so sorry is used
  sorry

end ice_cream_ratio_l2336_233608


namespace insphere_radius_l2336_233688

theorem insphere_radius (V S : ℝ) (hV : V > 0) (hS : S > 0) : 
  ∃ r : ℝ, r = 3 * V / S := by
  sorry

end insphere_radius_l2336_233688


namespace correct_grammatical_phrase_l2336_233672

-- Define the conditions as lean definitions 
def number_of_cars_produced_previous_year : ℕ := sorry  -- number of cars produced in previous year
def number_of_cars_produced_2004 : ℕ := 3 * number_of_cars_produced_previous_year  -- number of cars produced in 2004

-- Define the theorem stating the correct phrase to describe the production numbers
theorem correct_grammatical_phrase : 
  (3 * number_of_cars_produced_previous_year = number_of_cars_produced_2004) → 
  ("three times as many cars" = "three times as many cars") := 
by
  sorry

end correct_grammatical_phrase_l2336_233672


namespace percentage_of_customers_purchased_l2336_233698

theorem percentage_of_customers_purchased (ad_cost : ℕ) (customers : ℕ) (price_per_sale : ℕ) (profit : ℕ)
  (h1 : ad_cost = 1000)
  (h2 : customers = 100)
  (h3 : price_per_sale = 25)
  (h4 : profit = 1000) :
  (profit / price_per_sale / customers) * 100 = 40 :=
by
  sorry

end percentage_of_customers_purchased_l2336_233698


namespace max_value_of_g_l2336_233687

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 ∧ (∃ x0, x0 = 1 ∧ g x0 = 3) :=
by
  sorry

end max_value_of_g_l2336_233687


namespace sum_denominators_l2336_233628

theorem sum_denominators (a b: ℕ) (h_coprime : Nat.gcd a b = 1) :
  (3:ℚ) / (5 * b) + (2:ℚ) / (9 * b) + (4:ℚ) / (15 * b) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 :=
by
  sorry

end sum_denominators_l2336_233628


namespace probability_of_shaded_triangle_l2336_233689

def triangle (name: String) := name

def triangles := ["AEC", "AEB", "BED", "BEC", "BDC", "ABD"]
def shaded_triangles := ["BEC", "BDC", "ABD"]

theorem probability_of_shaded_triangle :
  (shaded_triangles.length : ℚ) / (triangles.length : ℚ) = 1 / 2 := 
by
  sorry

end probability_of_shaded_triangle_l2336_233689


namespace ratio_is_one_half_l2336_233616

noncomputable def ratio_dresses_with_pockets (D : ℕ) (total_pockets : ℕ) (pockets_two : ℕ) (pockets_three : ℕ) :=
  ∃ (P : ℕ), D = 24 ∧ total_pockets = 32 ∧
  (P / 3) * 2 + (2 * P / 3) * 3 = total_pockets ∧ 
  P / D = 1 / 2

theorem ratio_is_one_half :
  ratio_dresses_with_pockets 24 32 2 3 :=
by 
  sorry

end ratio_is_one_half_l2336_233616


namespace train_cross_platform_time_l2336_233695

noncomputable def kmph_to_mps (s : ℚ) : ℚ :=
  (s * 1000) / 3600

theorem train_cross_platform_time :
  let train_length := 110
  let speed_kmph := 52
  let platform_length := 323.36799999999994
  let speed_mps := kmph_to_mps 52
  let total_distance := train_length + platform_length
  let time := total_distance / speed_mps
  time = 30 := 
by
  sorry

end train_cross_platform_time_l2336_233695


namespace find_x_of_equation_l2336_233600

theorem find_x_of_equation :
  ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 :=
by 
  sorry

end find_x_of_equation_l2336_233600


namespace angle_733_in_first_quadrant_l2336_233625

def in_first_quadrant (θ : ℝ) : Prop := 
  0 < θ ∧ θ < 90

theorem angle_733_in_first_quadrant :
  in_first_quadrant (733 % 360 : ℝ) :=
sorry

end angle_733_in_first_quadrant_l2336_233625


namespace sine_double_angle_inequality_l2336_233676

theorem sine_double_angle_inequality {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 4) : 
  Real.sin (2 * α) < 2 * Real.sin α :=
by
  sorry

end sine_double_angle_inequality_l2336_233676


namespace problem_statement_l2336_233685

variable {α : Type*} [LinearOrderedCommRing α]

theorem problem_statement (a b c d e : α) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : a * b^2 * c * d^4 * e < 0 :=
by
  sorry

end problem_statement_l2336_233685


namespace problem_statement_l2336_233675

open Complex

noncomputable def z : ℂ := ((1 - I)^2 + 3 * (1 + I)) / (2 - I)

theorem problem_statement :
  z = 1 + I ∧ (∀ (a b : ℝ), (z^2 + a * z + b = 1 - I) → (a = -3 ∧ b = 4)) :=
by
  sorry

end problem_statement_l2336_233675


namespace find_number_l2336_233631

theorem find_number (x : ℝ) :
  (10 + 30 + 50) / 3 = 30 →
  ((x + 40 + 6) / 3 = (10 + 30 + 50) / 3 - 8) →
  x = 20 :=
by
  intros h_avg1 h_avg2
  sorry

end find_number_l2336_233631


namespace value_of_ab_l2336_233691

theorem value_of_ab (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 :=
by
  sorry

end value_of_ab_l2336_233691


namespace minimum_students_lost_all_items_l2336_233654

def smallest_number (N A B C : ℕ) (x : ℕ) : Prop :=
  N = 30 ∧ A = 26 ∧ B = 23 ∧ C = 21 → x ≥ 10

theorem minimum_students_lost_all_items (N A B C : ℕ) : 
  smallest_number N A B C 10 := 
by {
  sorry
}

end minimum_students_lost_all_items_l2336_233654


namespace previous_year_height_l2336_233645

noncomputable def previous_height (H_current : ℝ) (g : ℝ) : ℝ :=
  H_current / (1 + g)

theorem previous_year_height :
  previous_height 147 0.05 = 140 :=
by
  unfold previous_height
  -- Proof steps would go here
  sorry

end previous_year_height_l2336_233645


namespace sequence_comparison_l2336_233604

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define geometric sequence
def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ (∀ n, b (n + 1) = b n * q) ∧ (∀ i, i ≥ 1 → b i > 0)

-- Main theorem to prove
theorem sequence_comparison {a b : ℕ → ℝ} (q : ℝ) (h_a_arith : arithmetic_sequence a) 
  (h_b_geom : geometric_sequence b q) (h_eq_1 : a 1 = b 1) (h_eq_11 : a 11 = b 11) :
  a 6 > b 6 :=
sorry

end sequence_comparison_l2336_233604


namespace additional_investment_interest_rate_l2336_233610

theorem additional_investment_interest_rate :
  let initial_investment := 2400
  let initial_rate := 0.05
  let additional_investment := 600
  let total_investment := initial_investment + additional_investment
  let desired_total_income := 0.06 * total_investment
  let income_from_initial := initial_rate * initial_investment
  let additional_income_needed := desired_total_income - income_from_initial
  let additional_rate := additional_income_needed / additional_investment
  additional_rate * 100 = 10 :=
by
  sorry

end additional_investment_interest_rate_l2336_233610


namespace classify_abc_l2336_233646

theorem classify_abc (a b c : ℝ) 
  (h1 : (a > 0 ∨ a < 0 ∨ a = 0) ∧ (b > 0 ∨ b < 0 ∨ b = 0) ∧ (c > 0 ∨ c < 0 ∨ c = 0))
  (h2 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ (a > 0 ∧ b = 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c = 0) ∨
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ b > 0 ∧ c < 0) ∨ (a = 0 ∧ b < 0 ∧ c > 0))
  (h3 : |a| = b^2 * (b - c)) : 
  a < 0 ∧ b > 0 ∧ c = 0 :=
by 
  sorry

end classify_abc_l2336_233646


namespace solve_real_eq_l2336_233682

theorem solve_real_eq (x : ℝ) :
  (8 * x ^ 2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by
  sorry

end solve_real_eq_l2336_233682


namespace symmetric_line_eq_l2336_233649

theorem symmetric_line_eq (x y : ℝ) (c : ℝ) (P : ℝ × ℝ)
  (h₁ : 3 * x - y - 4 = 0)
  (h₂ : P = (2, -1))
  (h₃ : 3 * x - y + c = 0)
  (h : 3 * 2 - (-1) + c = 0) : 
  c = -7 :=
by
  sorry

end symmetric_line_eq_l2336_233649


namespace parking_fines_l2336_233674

theorem parking_fines (total_citations littering_citations offleash_dog_citations parking_fines : ℕ) 
  (h1 : total_citations = 24) 
  (h2 : littering_citations = 4) 
  (h3 : offleash_dog_citations = 4) 
  (h4 : total_citations = littering_citations + offleash_dog_citations + parking_fines) : 
  parking_fines = 16 := 
by 
  sorry

end parking_fines_l2336_233674


namespace valueOf_seq_l2336_233683

variable (a : ℕ → ℝ)
variable (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
variable (h_positive : ∀ n : ℕ, a n > 0)
variable (h_arith_subseq : 2 * a 5 = a 3 + a 6)

theorem valueOf_seq (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arith_subseq : 2 * a 5 = a 3 + a 6) :
  (∃ q : ℝ, q = 1 ∨ q = (1 + Real.sqrt 5) / 2 ∧ (a 3 + a 5) / (a 4 + a 6) = 1 / q) → 
  (∃ q : ℝ, (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2) :=
by
  sorry

end valueOf_seq_l2336_233683


namespace intersection_M_P_l2336_233642

def is_natural (x : ℤ) : Prop := x ≥ 0

def M (x : ℤ) : Prop := (x - 1)^2 < 4 ∧ is_natural x

def P := ({-1, 0, 1, 2, 3} : Set ℤ)

theorem intersection_M_P :
  {x : ℤ | M x} ∩ P = {0, 1, 2} :=
  sorry

end intersection_M_P_l2336_233642


namespace correct_answer_is_B_l2336_233661

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end correct_answer_is_B_l2336_233661
