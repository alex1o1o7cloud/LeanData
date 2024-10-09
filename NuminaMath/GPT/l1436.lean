import Mathlib

namespace total_rehabilitation_centers_l1436_143687

def lisa_visits : ℕ := 6
def jude_visits (lisa : ℕ) : ℕ := lisa / 2
def han_visits (jude : ℕ) : ℕ := 2 * jude - 2
def jane_visits (han : ℕ) : ℕ := 2 * han + 6
def total_visits (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

theorem total_rehabilitation_centers :
  total_visits lisa_visits (jude_visits lisa_visits) (han_visits (jude_visits lisa_visits)) 
    (jane_visits (han_visits (jude_visits lisa_visits))) = 27 :=
by
  sorry

end total_rehabilitation_centers_l1436_143687


namespace problem_l1436_143625

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

variables (f : ℝ → ℝ)
variables (h_odd : is_odd_function f)
variables (h_f1 : f 1 = 5)
variables (h_period : ∀ x, f (x + 4) = -f x)

-- Prove that f(2012) + f(2015) = -5
theorem problem :
  f 2012 + f 2015 = -5 :=
sorry

end problem_l1436_143625


namespace quadratic_no_real_roots_l1436_143684

theorem quadratic_no_real_roots (m : ℝ) : (∀ x, x^2 - 2 * x + m ≠ 0) ↔ m > 1 := 
by sorry

end quadratic_no_real_roots_l1436_143684


namespace soda_choosers_l1436_143679

-- Definitions based on conditions
def total_people := 600
def soda_angle := 108
def full_circle := 360

-- Statement to prove the number of people who referred to soft drinks as "Soda"
theorem soda_choosers : total_people * (soda_angle / full_circle) = 180 :=
by
  sorry

end soda_choosers_l1436_143679


namespace simplify_and_evaluate_expr_l1436_143661

theorem simplify_and_evaluate_expr (x y : ℚ) (h1 : x = -3/8) (h2 : y = 4) :
  (x - 2 * y) ^ 2 + (x - 2 * y) * (x + 2 * y) - 2 * x * (x - y) = 3 :=
by
  sorry

end simplify_and_evaluate_expr_l1436_143661


namespace maximum_value_abs_difference_l1436_143681

theorem maximum_value_abs_difference (x y : ℝ) 
  (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : 
  |x - y + 1| ≤ 2 :=
sorry

end maximum_value_abs_difference_l1436_143681


namespace greatest_integer_radius_l1436_143650

theorem greatest_integer_radius (r : ℝ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
sorry

end greatest_integer_radius_l1436_143650


namespace problem_1_problem_2_l1436_143685

-- First Problem
theorem problem_1 (f : ℝ → ℝ) (a : ℝ) (h : ∃ x : ℝ, f x - 2 * |x - 7| ≤ 0) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → a ≥ -12 :=
by
  intros
  sorry

-- Second Problem
theorem problem_2 (f : ℝ → ℝ) (a m : ℝ) (h1 : a = 1) 
  (h2 : ∀ x : ℝ, f x + |x + 7| ≥ m) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → m ≤ 7 :=
by
  intros
  sorry

end problem_1_problem_2_l1436_143685


namespace tangent_line_equation_even_derived_l1436_143668

def f (x a : ℝ) : ℝ := x^3 + (a - 2) * x^2 + a * x - 1

def f' (x a : ℝ) : ℝ := 3 * x^2 + 2 * (a - 2) * x + a

theorem tangent_line_equation_even_derived (a : ℝ) (h : ∀ x : ℝ, f' x a = f' (-x) a) :
  5 * 1 - (f 1 a) - 3 = 0 :=
by
  sorry

end tangent_line_equation_even_derived_l1436_143668


namespace typhoon_tree_survival_l1436_143616

def planted_trees : Nat := 150
def died_trees : Nat := 92
def slightly_damaged_trees : Nat := 15

def total_trees_affected : Nat := died_trees + slightly_damaged_trees
def trees_survived_without_damages : Nat := planted_trees - total_trees_affected
def more_died_than_survived : Nat := died_trees - trees_survived_without_damages

theorem typhoon_tree_survival :
  more_died_than_survived = 49 :=
by
  -- Define the necessary computations and assertions
  let total_trees_affected := 92 + 15
  let trees_survived_without_damages := 150 - total_trees_affected
  let more_died_than_survived := 92 - trees_survived_without_damages
  -- Prove the statement
  have : total_trees_affected = 107 := rfl
  have : trees_survived_without_damages = 43 := rfl
  have : more_died_than_survived = 49 := rfl
  exact this

end typhoon_tree_survival_l1436_143616


namespace scientific_notation_correct_l1436_143637

def original_number : ℕ := 31900

def scientific_notation_option_A : ℝ := 3.19 * 10^2
def scientific_notation_option_B : ℝ := 0.319 * 10^3
def scientific_notation_option_C : ℝ := 3.19 * 10^4
def scientific_notation_option_D : ℝ := 0.319 * 10^5

theorem scientific_notation_correct :
  original_number = 31900 ∧ scientific_notation_option_C = 3.19 * 10^4 ∧ (original_number : ℝ) = scientific_notation_option_C := 
by 
  sorry

end scientific_notation_correct_l1436_143637


namespace probability_white_given_black_drawn_l1436_143639

-- Definitions based on the conditions
def num_white : ℕ := 3
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

def P (n : ℕ) : ℚ := n / total_balls

-- Event A: drawing a black ball on the first draw
def PA : ℚ := P num_black

-- Event B: drawing a white ball on the second draw
def PB_given_A : ℚ := num_white / (total_balls - 1)

-- Theorem statement
theorem probability_white_given_black_drawn :
  (PA * PB_given_A) / PA = 3 / 4 :=
by
  sorry

end probability_white_given_black_drawn_l1436_143639


namespace min_cost_to_win_l1436_143629

theorem min_cost_to_win (n : ℕ) : 
  (∀ m : ℕ, m = 0 →
  (∀ cents : ℕ, 
  (n = 5 * m ∨ n = m + 1) ∧ n > 2008 ∧ n % 100 = 42 → 
  cents = 35)) :=
sorry

end min_cost_to_win_l1436_143629


namespace solve_for_m_l1436_143612

def power_function_monotonic (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2 * m - 3 < 0)

theorem solve_for_m (m : ℝ) (h : power_function_monotonic m) : m = 2 :=
sorry

end solve_for_m_l1436_143612


namespace sum_roots_l1436_143634

theorem sum_roots :
  (∀ (x : ℂ), (3 * x^3 - 2 * x^2 + 4 * x - 15 = 0) → 
              x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (∀ (x : ℂ), (4 * x^3 - 16 * x^2 - 28 * x + 35 = 0) → 
              x = y₁ ∨ x = y₂ ∨ x = y₃) →
  (x₁ + x₂ + x₃ + y₁ + y₂ + y₃ = 14 / 3) :=
by
  sorry

end sum_roots_l1436_143634


namespace find_parabola_coeffs_l1436_143610

def parabola_vertex_form (a b c : ℝ) : Prop :=
  ∃ k:ℝ, k = c - b^2 / (4*a) ∧ k = 3

def parabola_through_point (a b c : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, x = 0 ∧ y = 1 ∧  y = a * x^2 + b * x + c

theorem find_parabola_coeffs :
  ∃ a b c : ℝ, parabola_vertex_form a b c ∧ parabola_through_point a b c ∧
  a = -1/2 ∧ b = 2 ∧ c = 1 :=
by
  sorry

end find_parabola_coeffs_l1436_143610


namespace jellybean_ratio_l1436_143663

-- Define the conditions
def Matilda_jellybeans := 420
def Steve_jellybeans := 84
def Matt_jellybeans := 10 * Steve_jellybeans

-- State the theorem to prove the ratio
theorem jellybean_ratio : (Matilda_jellybeans : Nat) / (Matt_jellybeans : Nat) = 1 / 2 :=
by
  sorry

end jellybean_ratio_l1436_143663


namespace greatest_n_for_xy_le_0_l1436_143691

theorem greatest_n_for_xy_le_0
  (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) :
  ∃ n : ℕ, (n = a * b ∧ ∃ x y : ℤ, n = a * x + b * y ∧ x * y ≤ 0) :=
sorry

end greatest_n_for_xy_le_0_l1436_143691


namespace quadratic_no_rational_solution_l1436_143656

theorem quadratic_no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ∀ (x : ℚ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end quadratic_no_rational_solution_l1436_143656


namespace arrows_from_530_to_535_l1436_143683

def cyclic_arrows (n : Nat) : Nat :=
  n % 5

theorem arrows_from_530_to_535 : 
  cyclic_arrows 530 = 0 ∧ cyclic_arrows 531 = 1 ∧ cyclic_arrows 532 = 2 ∧
  cyclic_arrows 533 = 3 ∧ cyclic_arrows 534 = 4 ∧ cyclic_arrows 535 = 0 :=
by
  sorry

end arrows_from_530_to_535_l1436_143683


namespace forty_percent_of_number_l1436_143643

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 0.40 * N = 120 :=
sorry

end forty_percent_of_number_l1436_143643


namespace solve_linear_equation_l1436_143658

theorem solve_linear_equation : ∀ x : ℝ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) → x = -1 / 11 :=
by
  intro x h
  -- Proof to be filled in
  sorry

end solve_linear_equation_l1436_143658


namespace original_price_l1436_143607

theorem original_price (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 60) 
  (h2 : rate_of_profit = 0.20) 
  (h3 : SP = CP * (1 + rate_of_profit)) : CP = 50 := by
  sorry

end original_price_l1436_143607


namespace pyramid_base_side_length_l1436_143645

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (side_length : ℝ)
  (h1 : area_lateral_face = 144)
  (h2 : slant_height = 24)
  (h3 : 144 = 0.5 * side_length * 24) : 
  side_length = 12 :=
by
  sorry

end pyramid_base_side_length_l1436_143645


namespace number_of_neighborhoods_l1436_143646

def street_lights_per_side : ℕ := 250
def roads_per_neighborhood : ℕ := 4
def total_street_lights : ℕ := 20000

theorem number_of_neighborhoods : 
  (total_street_lights / (2 * street_lights_per_side * roads_per_neighborhood)) = 10 :=
by
  -- proof to show that the number of neighborhoods is 10
  sorry

end number_of_neighborhoods_l1436_143646


namespace minimum_xy_l1436_143638

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/y = 1/2) : x * y ≥ 16 :=
sorry

end minimum_xy_l1436_143638


namespace probability_escher_consecutive_l1436_143696

def total_pieces : Nat := 12
def escher_pieces : Nat := 4

theorem probability_escher_consecutive :
  (Nat.factorial 9 * Nat.factorial 4 : ℚ) / Nat.factorial 12 = 1 / 55 := 
sorry

end probability_escher_consecutive_l1436_143696


namespace bridge_crossing_possible_l1436_143623

/-- 
  There are four people A, B, C, and D. 
  The time it takes for each of them to cross the bridge is 2, 4, 6, and 8 minutes respectively.
  No more than two people can be on the bridge at the same time.
  Prove that it is possible for all four people to cross the bridge in 10 minutes.
--/
theorem bridge_crossing_possible : 
  ∃ (cross : ℕ → ℕ), 
  cross 1 = 2 ∧ cross 2 = 4 ∧ cross 3 = 6 ∧ cross 4 = 8 ∧
  (∀ (t : ℕ), t ≤ 2 → cross 1 + cross 2 + cross 3 + cross 4 = 10) :=
by
  sorry

end bridge_crossing_possible_l1436_143623


namespace find_m_l1436_143641

-- Definitions for the sets
def setA (x : ℝ) : Prop := -2 < x ∧ x < 8
def setB (m : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3

-- Condition on the intersection
def intersection (m : ℝ) (a b : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3 ∧ -2 < x ∧ x < 8

-- Theorem statement
theorem find_m (m a b : ℝ) (h₀ : b - a = 3) (h₁ : ∀ x, intersection m a b x ↔ (a < x ∧ x < b)) : m = -2 ∨ m = 1 :=
sorry

end find_m_l1436_143641


namespace card_problem_l1436_143688

-- Define the variables
variables (x y : ℕ)

-- Conditions given in the problem
theorem card_problem 
  (h1 : x - 1 = y + 1) 
  (h2 : x + 1 = 2 * (y - 1)) : 
  x + y = 12 :=
sorry

end card_problem_l1436_143688


namespace Olivia_spent_25_dollars_l1436_143667

theorem Olivia_spent_25_dollars
    (initial_amount : ℕ)
    (final_amount : ℕ)
    (spent_amount : ℕ)
    (h_initial : initial_amount = 54)
    (h_final : final_amount = 29)
    (h_spent : spent_amount = initial_amount - final_amount) :
    spent_amount = 25 := by
  sorry

end Olivia_spent_25_dollars_l1436_143667


namespace Mikail_birthday_money_l1436_143670

theorem Mikail_birthday_money (x : ℕ) (h1 : x = 3 + 3 * 3) : 5 * x = 60 := 
by 
  sorry

end Mikail_birthday_money_l1436_143670


namespace mary_circus_change_l1436_143693

theorem mary_circus_change :
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  change = 15 :=
by
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  sorry

end mary_circus_change_l1436_143693


namespace number_of_common_points_l1436_143689

-- Define the circle equation
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the vertical line equation
def is_on_line (x : ℝ) : Prop :=
  x = 3

-- Prove that the number of distinct points common to both graphs is two
theorem number_of_common_points : 
  ∃ y1 y2 : ℝ, is_on_circle 3 y1 ∧ is_on_circle 3 y2 ∧ y1 ≠ y2 :=
by {
  sorry
}

end number_of_common_points_l1436_143689


namespace penny_nickel_dime_heads_probability_l1436_143665

def num_successful_outcomes : Nat :=
1 * 1 * 1 * 2

def total_possible_outcomes : Nat :=
2 ^ 4

def probability_event : ℚ :=
num_successful_outcomes / total_possible_outcomes

theorem penny_nickel_dime_heads_probability :
  probability_event = 1 / 8 := 
by
  sorry

end penny_nickel_dime_heads_probability_l1436_143665


namespace twenty_four_is_eighty_percent_of_what_number_l1436_143609

theorem twenty_four_is_eighty_percent_of_what_number (x : ℝ) (hx : 24 = 0.8 * x) : x = 30 :=
  sorry

end twenty_four_is_eighty_percent_of_what_number_l1436_143609


namespace divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l1436_143677

theorem divides_2_pow_n_sub_1 (n : ℕ) : 7 ∣ (2 ^ n - 1) ↔ 3 ∣ n := by
  sorry

theorem no_n_divides_2_pow_n_add_1 (n : ℕ) : ¬ 7 ∣ (2 ^ n + 1) := by
  sorry

end divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l1436_143677


namespace circles_tangent_l1436_143617

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

theorem circles_tangent :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end circles_tangent_l1436_143617


namespace rajas_monthly_income_l1436_143659

theorem rajas_monthly_income (I : ℝ) (h : 0.6 * I + 0.1 * I + 0.1 * I + 5000 = I) : I = 25000 :=
sorry

end rajas_monthly_income_l1436_143659


namespace solve_for_x_l1436_143671

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 4 * x = 0) (h₁ : x ≠ 0) : x = 4 := 
by
  sorry

end solve_for_x_l1436_143671


namespace find_ratio_l1436_143614
   
   -- Given Conditions
   variable (S T F : ℝ)
   variable (H1 : 30 + S + T + F = 450)
   variable (H2 : S > 30)
   variable (H3 : T > S)
   variable (H4 : F > T)
   
   -- The goal is to find the ratio S / 30
   theorem find_ratio :
     ∃ r : ℝ, r = S / 30 ↔ false :=
   by
     sorry
   
end find_ratio_l1436_143614


namespace not_p_and_not_p_and_q_implies_not_p_or_q_l1436_143627

theorem not_p_and_not_p_and_q_implies_not_p_or_q (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬(p ∨ q) :=
sorry

end not_p_and_not_p_and_q_implies_not_p_or_q_l1436_143627


namespace circle_through_points_eq_l1436_143603

noncomputable def circle_eqn (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_through_points_eq {h k r : ℝ} :
  circle_eqn h k r (-1) 0 ∧
  circle_eqn h k r 0 2 ∧
  circle_eqn h k r 2 0 → 
  (h = 2 / 3 ∧ k = 2 / 3 ∧ r^2 = 29 / 9) :=
sorry

end circle_through_points_eq_l1436_143603


namespace DE_plus_FG_equals_19_div_6_l1436_143615

theorem DE_plus_FG_equals_19_div_6
    (AB AC : ℝ)
    (BC : ℝ)
    (h_isosceles : AB = 2 ∧ AC = 2 ∧ BC = 1.5)
    (D E G F : ℝ)
    (h_parallel_DE_BC : D = E)
    (h_parallel_FG_BC : F = G)
    (h_same_perimeter : 2 + D = 2 + F ∧ 2 + F = 5.5 - F) :
    D + F = 19 / 6 := by
  sorry

end DE_plus_FG_equals_19_div_6_l1436_143615


namespace adam_books_l1436_143621

theorem adam_books (before_books total_shelves books_per_shelf after_books leftover_books bought_books : ℕ)
  (h_before: before_books = 56)
  (h_shelves: total_shelves = 4)
  (h_books_per_shelf: books_per_shelf = 20)
  (h_leftover: leftover_books = 2)
  (h_after: after_books = (total_shelves * books_per_shelf) + leftover_books)
  (h_difference: bought_books = after_books - before_books) :
  bought_books = 26 :=
by
  sorry

end adam_books_l1436_143621


namespace least_k_for_sum_divisible_l1436_143653

theorem least_k_for_sum_divisible (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, (∀ (xs : List ℕ), (xs.length = k) → (∃ ys : List ℕ, (ys.length % 2 = 0) ∧ (ys.sum % n = 0))) ∧ 
    (k = if n % 2 = 1 then 2 * n else n + 1)) :=
sorry

end least_k_for_sum_divisible_l1436_143653


namespace skating_minutes_needed_l1436_143648

-- Define the conditions
def minutes_per_day (day: ℕ) : ℕ :=
  if day ≤ 4 then 80 else if day ≤ 6 then 100 else 0

-- Define total skating time up to 6 days
def total_time_six_days := (4 * 80) + (2 * 100)

-- Prove that Gage needs to skate 180 minutes on the seventh day
theorem skating_minutes_needed : 
  (total_time_six_days + x = 7 * 100) → x = 180 :=
by sorry

end skating_minutes_needed_l1436_143648


namespace percentage_difference_l1436_143698

theorem percentage_difference :
  (0.50 * 56 - 0.30 * 50) = 13 := 
by
  -- sorry is used to skip the actual proof steps
  sorry 

end percentage_difference_l1436_143698


namespace parabola_focus_to_directrix_distance_correct_l1436_143640

def parabola_focus_to_directrix_distance (a : ℕ) (y x : ℝ) : Prop :=
  y^2 = 2 * x → a = 2 →  1 = 1

theorem parabola_focus_to_directrix_distance_correct :
  ∀ (a : ℕ) (y x : ℝ), parabola_focus_to_directrix_distance a y x :=
by
  unfold parabola_focus_to_directrix_distance
  intros
  sorry

end parabola_focus_to_directrix_distance_correct_l1436_143640


namespace solve_for_x_l1436_143660

theorem solve_for_x : ∃ x : ℝ, (9 - x) ^ 2 = x ^ 2 ∧ x = 4.5 :=
by
  sorry

end solve_for_x_l1436_143660


namespace problem1_problem2_l1436_143649

-- Problem 1
theorem problem1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 :=
by decide  -- automatically prove simple arithmetic

-- Problem 2
variables {x : ℝ} (hx1 : x ≠ 1) (hx2 : x ≠ -1)

theorem problem2 : ((x^2 / (x + 1)) - (1 / (x + 1))) * (x + 1) / (x - 1) = x + 1 :=
by sorry  -- proof to be completed

end problem1_problem2_l1436_143649


namespace place_sweet_hexagons_l1436_143669

def sweetHexagon (h : ℝ) : Prop := h = 1
def convexPolygon (A : ℝ) : Prop := A ≥ 1900000
def hexagonPlacementPossible (N : ℕ) : Prop := N ≤ 2000000

theorem place_sweet_hexagons:
  (∀ h, sweetHexagon h) →
  (∃ A, convexPolygon A) →
  (∃ N, hexagonPlacementPossible N) →
  True :=
by
  intros _ _ _ 
  exact True.intro

end place_sweet_hexagons_l1436_143669


namespace find_y_payment_l1436_143678

-- Definitions for the conditions in the problem
def total_payment (X Y : ℝ) : Prop := X + Y = 560
def x_is_120_percent_of_y (X Y : ℝ) : Prop := X = 1.2 * Y

-- Problem statement converted to a Lean proof problem
theorem find_y_payment (X Y : ℝ) (h1 : total_payment X Y) (h2 : x_is_120_percent_of_y X Y) : Y = 255 := 
by sorry

end find_y_payment_l1436_143678


namespace convert_A03_to_decimal_l1436_143636

theorem convert_A03_to_decimal :
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  hex_value = 2563 :=
by
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  have : hex_value = 2563 := sorry
  exact this

end convert_A03_to_decimal_l1436_143636


namespace greatest_length_of_cords_l1436_143694

theorem greatest_length_of_cords (a b c : ℝ) (h₁ : a = Real.sqrt 20) (h₂ : b = Real.sqrt 50) (h₃ : c = Real.sqrt 98) :
  ∃ (d : ℝ), d = 1 ∧ ∀ (k : ℝ), (k = a ∨ k = b ∨ k = c) → ∃ (n m : ℕ), k = d * (n : ℝ) ∧ d * (m : ℝ) = (m : ℝ) := by
sorry

end greatest_length_of_cords_l1436_143694


namespace cube_surface_area_l1436_143632

theorem cube_surface_area (a : ℕ) (h : a = 2) : 6 * a^2 = 24 := 
by
  sorry

end cube_surface_area_l1436_143632


namespace evaluate_expression_l1436_143618

theorem evaluate_expression : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3) := 
by
  sorry

end evaluate_expression_l1436_143618


namespace arithmetic_mean_18_27_45_l1436_143674

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end arithmetic_mean_18_27_45_l1436_143674


namespace largest_common_divisor_414_345_l1436_143631

theorem largest_common_divisor_414_345 : ∃ d, d ∣ 414 ∧ d ∣ 345 ∧ 
                                      (∀ e, e ∣ 414 ∧ e ∣ 345 → e ≤ d) ∧ d = 69 :=
by 
  sorry

end largest_common_divisor_414_345_l1436_143631


namespace polyhedron_calculation_l1436_143606

def faces := 32
def triangular := 10
def pentagonal := 8
def hexagonal := 14
def edges := 79
def vertices := 49
def T := 1
def P := 2

theorem polyhedron_calculation : 
  100 * P + 10 * T + vertices = 249 := 
sorry

end polyhedron_calculation_l1436_143606


namespace missing_fraction_l1436_143644

-- Definitions for the given fractions
def a := 1 / 3
def b := 1 / 2
def c := 1 / 5
def d := 1 / 4
def e := -9 / 20
def f := -2 / 15
def target_sum := 2 / 15 -- because 0.13333333333333333 == 2 / 15

-- Main theorem statement for the problem
theorem missing_fraction : a + b + c + d + e + f + -17 / 30 = target_sum :=
by
  simp [a, b, c, d, e, f, target_sum]
  sorry

end missing_fraction_l1436_143644


namespace sum_of_legs_of_similar_larger_triangle_l1436_143664

-- Define the conditions for the problem
def smaller_triangle_area : ℝ := 10
def larger_triangle_area : ℝ := 400
def smaller_triangle_hypotenuse : ℝ := 10

-- Define the correct answer (sum of the lengths of the legs of the larger triangle)
def sum_of_legs_of_larger_triangle : ℝ := 88.55

-- State the Lean theorem
theorem sum_of_legs_of_similar_larger_triangle :
  (∀ (A B C a b c : ℝ), 
    a * b / 2 = smaller_triangle_area ∧ 
    c = smaller_triangle_hypotenuse ∧
    C * C / 4 = larger_triangle_area / smaller_triangle_area ∧
    A / a = B / b ∧ 
    A^2 + B^2 = C^2 → 
    A + B = sum_of_legs_of_larger_triangle) :=
  by sorry

end sum_of_legs_of_similar_larger_triangle_l1436_143664


namespace find_m_l1436_143602

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l1436_143602


namespace minimum_value_abs_a_plus_2_abs_b_l1436_143620

open Real

theorem minimum_value_abs_a_plus_2_abs_b 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (x₁ x₂ x₃ : ℝ)
  (f_def : ∀ x, f x = x^3 + a*x^2 + b*x)
  (roots_cond : x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1)
  (equal_values : f x₁ = f x₂ ∧ f x₂ = f x₃) :
  ∃ minimum, minimum = (sqrt 3) ∧ (∀ (a b : ℝ), |a| + 2*|b| ≥ sqrt 3) :=
by
  sorry

end minimum_value_abs_a_plus_2_abs_b_l1436_143620


namespace percentage_of_y_l1436_143695

theorem percentage_of_y (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y :=
by
  sorry

end percentage_of_y_l1436_143695


namespace orthogonal_lines_solution_l1436_143672

theorem orthogonal_lines_solution (a b c d : ℝ)
  (h1 : b - a = 0)
  (h2 : c - a = 2)
  (h3 : 12 * d - a = 1)
  : d = 3 / 11 :=
by {
  sorry
}

end orthogonal_lines_solution_l1436_143672


namespace compute_expression_l1436_143608

theorem compute_expression :
  18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 :=
by
  sorry

end compute_expression_l1436_143608


namespace expand_expression_l1436_143624

theorem expand_expression : ∀ (x : ℝ), (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 :=
by
  intro x
  sorry

end expand_expression_l1436_143624


namespace num_chickens_is_one_l1436_143676

-- Define the number of dogs and the number of total legs
def num_dogs := 2
def total_legs := 10

-- Define the number of legs per dog and per chicken
def legs_per_dog := 4
def legs_per_chicken := 2

-- Define the number of chickens
def num_chickens := (total_legs - num_dogs * legs_per_dog) / legs_per_chicken

-- Prove that the number of chickens is 1
theorem num_chickens_is_one : num_chickens = 1 := by
  -- This is the proof placeholder
  sorry

end num_chickens_is_one_l1436_143676


namespace least_t_geometric_progression_exists_l1436_143635

open Real

theorem least_t_geometric_progression_exists :
  ∃ (t : ℝ),
  (∃ (α : ℝ), 0 < α ∧ α < π / 3 ∧
             (arcsin (sin α) = α ∧
              arcsin (sin (3 * α)) = 3 * α ∧
              arcsin (sin (8 * α)) = 8 * α) ∧
              (arcsin (sin (t * α)) = (some_ratio) * (arcsin (sin (8 * α))) )) ∧ 
   0 < t := 
by 
  sorry

end least_t_geometric_progression_exists_l1436_143635


namespace plus_signs_count_l1436_143697

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l1436_143697


namespace max_integer_solutions_l1436_143655

noncomputable def semi_centered (p : ℕ → ℤ) :=
  ∃ k : ℕ, p k = k + 50 - 50 * 50

theorem max_integer_solutions (p : ℕ → ℤ) (h1 : semi_centered p) (h2 : ∀ x : ℕ, ∃ c : ℤ, p x = c * x^2) (h3 : p 50 = 50) :
  ∃ n ≤ 6, ∀ k : ℕ, (p k = k^2) → k ∈ Finset.range (n+1) :=
sorry

end max_integer_solutions_l1436_143655


namespace find_pairs_of_positive_numbers_l1436_143692

theorem find_pairs_of_positive_numbers
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (exists_triangle : ∃ (C D E A B : ℝ), true)
  (points_on_hypotenuse : ∀ (C D E A B : ℝ), A ∈ [D, E] ∧ B ∈ [D, E]) 
  (equal_vectors : ∀ (D A B E : ℝ), (D - A) = (A - B) ∧ (A - B) = (B - E))
  (AC_eq_a : (C - A) = a)
  (BC_eq_b : (C - B) = b) :
  (1 / 2) < (a / b) ∧ (a / b) < 2 :=
by {
  sorry
}

end find_pairs_of_positive_numbers_l1436_143692


namespace commission_amount_l1436_143690

theorem commission_amount 
  (new_avg_commission : ℤ) (increase_in_avg : ℤ) (sales_count : ℤ) 
  (total_commission_before : ℤ) (total_commission_after : ℤ) : 
  new_avg_commission = 400 → increase_in_avg = 150 → sales_count = 6 → 
  total_commission_before = (sales_count - 1) * (new_avg_commission - increase_in_avg) → 
  total_commission_after = sales_count * new_avg_commission → 
  total_commission_after - total_commission_before = 1150 :=
by 
  sorry

end commission_amount_l1436_143690


namespace compound_interest_1200_20percent_3years_l1436_143601

noncomputable def compoundInterest (P r : ℚ) (n t : ℕ) : ℚ :=
  let A := P * (1 + r / n) ^ (n * t)
  A - P

theorem compound_interest_1200_20percent_3years :
  compoundInterest 1200 0.20 1 3 = 873.6 :=
by
  sorry

end compound_interest_1200_20percent_3years_l1436_143601


namespace convention_handshakes_l1436_143654

-- Introducing the conditions
def companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_reps : ℕ := companies * reps_per_company
def shakes_per_rep : ℕ := total_reps - 1 - (reps_per_company - 1)
def handshakes : ℕ := (total_reps * shakes_per_rep) / 2

-- Statement of the proof
theorem convention_handshakes : handshakes = 160 :=
by
  sorry  -- Proof is not required in this task.

end convention_handshakes_l1436_143654


namespace max_mean_BC_l1436_143642

theorem max_mean_BC (A_n B_n C_n A_total_weight B_total_weight C_total_weight : ℕ)
    (hA_mean : A_total_weight = 45 * A_n)
    (hB_mean : B_total_weight = 55 * B_n)
    (hAB_mean : (A_total_weight + B_total_weight) / (A_n + B_n) = 48)
    (hAC_mean : (A_total_weight + C_total_weight) / (A_n + C_n) = 50) :
    ∃ m : ℤ, m = 66 := by
  sorry

end max_mean_BC_l1436_143642


namespace new_assistant_draw_time_l1436_143682

-- Definitions based on conditions
def capacity : ℕ := 36
def halfway : ℕ := capacity / 2
def rate_top : ℕ := 1 / 6
def rate_bottom : ℕ := 1 / 4
def extra_time : ℕ := 24

-- The proof statement
theorem new_assistant_draw_time : 
  ∃ t : ℕ, ((capacity - (extra_time * rate_bottom * 1)) - halfway) = (t * rate_bottom * 1) ∧ t = 48 := by
sorry

end new_assistant_draw_time_l1436_143682


namespace number_of_women_l1436_143622

theorem number_of_women (w1 w2: ℕ) (m1 m2 d1 d2: ℕ)
    (h1: w2 = 5) (h2: m2 = 100) (h3: d2 = 1) 
    (h4: d1 = 3) (h5: m1 = 360)
    (h6: w1 * d1 = m1 * d2 / m2 * w2) : w1 = 6 :=
by
  sorry

end number_of_women_l1436_143622


namespace rectangle_area_correct_l1436_143604

noncomputable def rectangle_area (x: ℚ) : ℚ :=
  let length := 5 * x - 18
  let width := 25 - 4 * x
  length * width

theorem rectangle_area_correct (x: ℚ) (h1: 3.6 < x) (h2: x < 6.25) :
  rectangle_area (43 / 9) = (2809 / 81) := 
  by
    sorry

end rectangle_area_correct_l1436_143604


namespace fraction_difference_is_correct_l1436_143686

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l1436_143686


namespace distance_between_foci_of_hyperbola_l1436_143666

theorem distance_between_foci_of_hyperbola {x y : ℝ} (h : x ^ 2 - 4 * y ^ 2 = 4) :
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_hyperbola_l1436_143666


namespace problem_statement_l1436_143662

open Set

variable (a : ℕ)
variable (A : Set ℕ := {2, 3, 4})
variable (B : Set ℕ := {a + 2, a})

theorem problem_statement (hB : B ⊆ A) : (A \ B) = {3} :=
sorry

end problem_statement_l1436_143662


namespace initial_workers_l1436_143633

theorem initial_workers (W : ℕ) (work1 : ℕ) (work2 : ℕ) :
  (work1 = W * 8 * 30) →
  (work2 = (W + 35) * 6 * 40) →
  (work1 / 30 = work2 / 40) →
  W = 105 :=
by
  intros hwork1 hwork2 hprop
  sorry

end initial_workers_l1436_143633


namespace zero_in_interval_l1436_143630

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x + 2 * x - 8

theorem zero_in_interval : (f 3 < 0) ∧ (f 4 > 0) → ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end zero_in_interval_l1436_143630


namespace two_solutions_for_positive_integer_m_l1436_143675

theorem two_solutions_for_positive_integer_m :
  ∃ k : ℕ, k = 2 ∧ (∀ m : ℕ, 0 < m → 990 % (m^2 - 2) = 0 → m = 2 ∨ m = 3) := 
sorry

end two_solutions_for_positive_integer_m_l1436_143675


namespace classroom_books_l1436_143626

theorem classroom_books (students_group1 students_group2 books_per_student_group1 books_per_student_group2 books_brought books_lost : ℕ)
  (h1 : students_group1 = 20)
  (h2 : books_per_student_group1 = 15)
  (h3 : students_group2 = 25)
  (h4 : books_per_student_group2 = 10)
  (h5 : books_brought = 30)
  (h6 : books_lost = 7) :
  (students_group1 * books_per_student_group1 + students_group2 * books_per_student_group2 + books_brought - books_lost) = 573 := by
  sorry

end classroom_books_l1436_143626


namespace arith_seq_geom_seq_l1436_143647

theorem arith_seq_geom_seq (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : (a 9)^2 = a 5 * a 15) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end arith_seq_geom_seq_l1436_143647


namespace closest_clock_to_16_is_C_l1436_143657

noncomputable def closestTo16InMirror (clock : Char) : Bool :=
  clock = 'C'

theorem closest_clock_to_16_is_C : 
  (closestTo16InMirror 'A' = False) ∧ 
  (closestTo16InMirror 'B' = False) ∧ 
  (closestTo16InMirror 'C' = True) ∧ 
  (closestTo16InMirror 'D' = False) := 
by
  sorry

end closest_clock_to_16_is_C_l1436_143657


namespace remainder_of_exponentiation_is_correct_l1436_143680

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l1436_143680


namespace distance_between_ann_and_glenda_l1436_143613

def ann_distance : ℝ := 
  let speed1 := 6
  let time1 := 1
  let speed2 := 8
  let time2 := 1
  let break1 := 0
  let speed3 := 4
  let time3 := 1
  speed1 * time1 + speed2 * time2 + break1 * 0 + speed3 * time3

def glenda_distance : ℝ := 
  let speed1 := 8
  let time1 := 1
  let speed2 := 5
  let time2 := 1
  let break1 := 0
  let speed3 := 9
  let back_time := 0.5
  let back_distance := speed3 * back_time
  let continue_time := 0.5
  let continue_distance := speed3 * continue_time
  speed1 * time1 + speed2 * time2 + break1 * 0 + (-back_distance) + continue_distance

theorem distance_between_ann_and_glenda : 
  ann_distance + glenda_distance = 35.5 := 
by 
  sorry

end distance_between_ann_and_glenda_l1436_143613


namespace solution_set_of_inequality_l1436_143605

theorem solution_set_of_inequality:
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | (-7 < x ∧ x ≤ -1) ∨ (5 ≤ x ∧ x < 11)} :=
by
  sorry

end solution_set_of_inequality_l1436_143605


namespace fraction_of_B_amount_equals_third_of_A_amount_l1436_143628

variable (A B : ℝ)
variable (x : ℝ)

theorem fraction_of_B_amount_equals_third_of_A_amount
  (h1 : A + B = 1210)
  (h2 : B = 484)
  (h3 : (1 / 3) * A = x * B) : 
  x = 1 / 2 :=
sorry

end fraction_of_B_amount_equals_third_of_A_amount_l1436_143628


namespace winning_candidate_percentage_l1436_143699

theorem winning_candidate_percentage (total_membership: ℕ)
  (votes_cast: ℕ) (winning_percentage: ℝ) (h1: total_membership = 1600)
  (h2: votes_cast = 525) (h3: winning_percentage = 19.6875)
  : (winning_percentage / 100 * total_membership / votes_cast * 100 = 60) :=
by
  sorry

end winning_candidate_percentage_l1436_143699


namespace single_elimination_games_needed_l1436_143611

theorem single_elimination_games_needed (n : ℕ) (n_pos : n > 0) :
  (number_of_games_needed : ℕ) = n - 1 :=
by
  sorry

end single_elimination_games_needed_l1436_143611


namespace product_of_102_and_27_l1436_143651

theorem product_of_102_and_27 : 102 * 27 = 2754 :=
by
  sorry

end product_of_102_and_27_l1436_143651


namespace sugar_amount_l1436_143673

theorem sugar_amount (S F B : ℝ) 
    (h_ratio1 : S = F) 
    (h_ratio2 : F = 10 * B) 
    (h_ratio3 : F / (B + 60) = 8) : S = 2400 := 
by
  sorry

end sugar_amount_l1436_143673


namespace proportion_terms_l1436_143652

theorem proportion_terms (x v y z : ℤ) (a b c : ℤ)
  (h1 : x + v = y + z + a)
  (h2 : x^2 + v^2 = y^2 + z^2 + b)
  (h3 : x^4 + v^4 = y^4 + z^4 + c)
  (ha : a = 7) (hb : b = 21) (hc : c = 2625) :
  (x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) :=
by
  sorry

end proportion_terms_l1436_143652


namespace ratio_proof_l1436_143619

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : (a + 2 * b) / (3 * b + c) = 9 / 32 :=
by
  sorry

end ratio_proof_l1436_143619


namespace probability_div_int_l1436_143600

theorem probability_div_int
    (r : ℤ) (k : ℤ)
    (hr : -5 < r ∧ r < 10)
    (hk : 1 < k ∧ k < 8)
    (hk_prime : Nat.Prime (Int.natAbs k)) :
    ∃ p q : ℕ, (p = 3 ∧ q = 14) ∧ p / q = 3 / 14 := 
by {
  sorry
}

end probability_div_int_l1436_143600
