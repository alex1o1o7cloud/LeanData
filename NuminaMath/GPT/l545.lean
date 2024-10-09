import Mathlib

namespace minimal_perimeter_triangle_l545_54572

noncomputable def cos_P : ℚ := 3 / 5
noncomputable def cos_Q : ℚ := 24 / 25
noncomputable def cos_R : ℚ := -1 / 5

theorem minimal_perimeter_triangle
  (P Q R : ℝ) (a b c : ℕ)
  (h0 : a^2 + b^2 + c^2 - 2 * a * b * cos_P - 2 * b * c * cos_Q - 2 * c * a * cos_R = 0)
  (h1 : cos_P^2 + (1 - cos_P^2) = 1)
  (h2 : cos_Q^2 + (1 - cos_Q^2) = 1)
  (h3 : cos_R^2 + (1 - cos_R^2) = 1) :
  a + b + c = 47 :=
sorry

end minimal_perimeter_triangle_l545_54572


namespace total_circle_area_within_triangle_l545_54535

-- Define the sides of the triangle
def triangle_sides : Prop := ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5

-- Define the radii and center of the circles at each vertex of the triangle
def circle_centers_and_radii : Prop := ∃ (r : ℝ) (A B C : ℝ × ℝ), r = 1

-- The formal statement that we need to prove:
theorem total_circle_area_within_triangle :
  triangle_sides ∧ circle_centers_and_radii → 
  (total_area_of_circles_within_triangle = π / 2) := sorry

end total_circle_area_within_triangle_l545_54535


namespace prob_white_given_popped_l545_54524

-- Definitions for given conditions:
def P_white : ℚ := 1 / 2
def P_yellow : ℚ := 1 / 4
def P_blue : ℚ := 1 / 4

def P_popped_given_white : ℚ := 1 / 3
def P_popped_given_yellow : ℚ := 3 / 4
def P_popped_given_blue : ℚ := 2 / 3

-- Calculations derived from conditions:
def P_white_popped : ℚ := P_white * P_popped_given_white
def P_yellow_popped : ℚ := P_yellow * P_popped_given_yellow
def P_blue_popped : ℚ := P_blue * P_popped_given_blue

def P_popped : ℚ := P_white_popped + P_yellow_popped + P_blue_popped

-- Main theorem to be proved:
theorem prob_white_given_popped : (P_white_popped / P_popped) = 2 / 11 :=
by sorry

end prob_white_given_popped_l545_54524


namespace compare_neg_fractions_l545_54526

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l545_54526


namespace solve_first_equation_solve_second_equation_l545_54508

-- Statement for the first equation
theorem solve_first_equation : ∀ x : ℝ, x^2 - 3*x - 4 = 0 ↔ x = 4 ∨ x = -1 := by
  sorry

-- Statement for the second equation
theorem solve_second_equation : ∀ x : ℝ, x * (x - 2) = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
  sorry

end solve_first_equation_solve_second_equation_l545_54508


namespace problem_water_percentage_l545_54547

noncomputable def percentage_water_in_mixture 
  (volA volB volC volD : ℕ) 
  (pctA pctB pctC pctD : ℝ) : ℝ :=
  let total_volume := volA + volB + volC + volD
  let total_solution := volA * pctA + volB * pctB + volC * pctC + volD * pctD
  let total_water := total_volume - total_solution
  (total_water / total_volume) * 100

theorem problem_water_percentage :
  percentage_water_in_mixture 100 90 60 50 0.25 0.3 0.4 0.2 = 71.33 :=
by
  -- proof goes here
  sorry

end problem_water_percentage_l545_54547


namespace jenna_discount_l545_54500

def normal_price : ℝ := 50
def tickets_from_website : ℝ := 2 * normal_price
def scalper_initial_price_per_ticket : ℝ := 2.4 * normal_price
def scalper_total_initial : ℝ := 2 * scalper_initial_price_per_ticket
def friend_discounted_ticket : ℝ := 0.6 * normal_price
def total_price_five_tickets : ℝ := tickets_from_website + scalper_total_initial + friend_discounted_ticket
def amount_paid_by_friends : ℝ := 360

theorem jenna_discount : 
    total_price_five_tickets - amount_paid_by_friends = 10 :=
by
  -- The proof would go here, but we leave it as sorry for now.
  sorry

end jenna_discount_l545_54500


namespace expand_polynomial_l545_54588

theorem expand_polynomial : 
  ∀ (x : ℝ), (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 :=
by
  intro x
  sorry

end expand_polynomial_l545_54588


namespace pencil_ratio_l545_54592

theorem pencil_ratio (B G : ℕ) (h1 : ∀ (n : ℕ), n = 20) 
  (h2 : ∀ (n : ℕ), n = 40) 
  (h3 : ∀ (n : ℕ), n = 160) 
  (h4 : G = 20 + B)
  (h5 : B + 20 + G + 40 = 160) : 
  (B / 20) = 4 := 
  by sorry

end pencil_ratio_l545_54592


namespace maintain_constant_chromosomes_l545_54579

-- Definitions
def meiosis_reduces_chromosomes (original_chromosomes : ℕ) : ℕ := original_chromosomes / 2

def fertilization_restores_chromosomes (half_chromosomes : ℕ) : ℕ := half_chromosomes * 2

-- The proof problem
theorem maintain_constant_chromosomes (original_chromosomes : ℕ) (somatic_chromosomes : ℕ) :
  meiosis_reduces_chromosomes original_chromosomes = somatic_chromosomes / 2 ∧
  fertilization_restores_chromosomes (meiosis_reduces_chromosomes original_chromosomes) = somatic_chromosomes :=
sorry

end maintain_constant_chromosomes_l545_54579


namespace cars_produced_in_europe_l545_54514

theorem cars_produced_in_europe (total_cars : ℕ) (cars_in_north_america : ℕ) (cars_in_europe : ℕ) :
  total_cars = 6755 → cars_in_north_america = 3884 → cars_in_europe = total_cars - cars_in_north_america → cars_in_europe = 2871 :=
by
  -- necessary calculations and logical steps
  sorry

end cars_produced_in_europe_l545_54514


namespace find_a_of_extremum_l545_54596

theorem find_a_of_extremum (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : f x = x^3 + a*x^2 + b*x + a^2)
  (h2 : f' x = 3*x^2 + 2*a*x + b)
  (h3 : f' 1 = 0)
  (h4 : f 1 = 10) : a = 4 := by
  sorry

end find_a_of_extremum_l545_54596


namespace choose_9_3_eq_84_l545_54501

theorem choose_9_3_eq_84 : Nat.choose 9 3 = 84 :=
by
  sorry

end choose_9_3_eq_84_l545_54501


namespace magnitude_of_complex_l545_54551

def complex_number := Complex.mk 2 3 -- Define the complex number 2+3i

theorem magnitude_of_complex : Complex.abs complex_number = Real.sqrt 13 := by
  sorry

end magnitude_of_complex_l545_54551


namespace muffins_sold_in_afternoon_l545_54536

variable (total_muffins : ℕ)
variable (morning_muffins : ℕ)
variable (remaining_muffins : ℕ)

theorem muffins_sold_in_afternoon 
  (h1 : total_muffins = 20) 
  (h2 : morning_muffins = 12) 
  (h3 : remaining_muffins = 4) : 
  (total_muffins - remaining_muffins - morning_muffins) = 4 := 
by
  sorry

end muffins_sold_in_afternoon_l545_54536


namespace range_log_div_pow3_div3_l545_54525

noncomputable def log_div (x y : ℝ) : ℝ := Real.log (x / y)
noncomputable def log_div_pow3 (x y : ℝ) : ℝ := Real.log (x^3 / y^(1/2))
noncomputable def log_div_pow3_div3 (x y : ℝ) : ℝ := Real.log (x^3 / (3 * y))

theorem range_log_div_pow3_div3 
  (x y : ℝ) 
  (h1 : 1 ≤ log_div x y ∧ log_div x y ≤ 2)
  (h2 : 2 ≤ log_div_pow3 x y ∧ log_div_pow3 x y ≤ 3) 
  : Real.log (x^3 / (3 * y)) ∈ Set.Icc (26/15 : ℝ) 3 :=
sorry

end range_log_div_pow3_div3_l545_54525


namespace fraction_inequality_l545_54540

variables (a b m : ℝ)

theorem fraction_inequality (h1 : a > b) (h2 : m > 0) : (b + m) / (a + m) > b / a :=
sorry

end fraction_inequality_l545_54540


namespace exists_monomial_l545_54559

variables (x y : ℕ) -- Define x and y as natural numbers

theorem exists_monomial :
  ∃ (c : ℕ) (e_x e_y : ℕ), c = 3 ∧ e_x + e_y = 3 ∧ (c * x ^ e_x * y ^ e_y) = (3 * x ^ e_x * y ^ e_y) :=
by
  sorry

end exists_monomial_l545_54559


namespace hot_dogs_sold_next_innings_l545_54584

-- Defining the conditions
variables (total_initial hot_dogs_sold_first_innings hot_dogs_left : ℕ)

-- Given conditions that need to hold true
axiom initial_count : total_initial = 91
axiom first_innings_sold : hot_dogs_sold_first_innings = 19
axiom remaining_hot_dogs : hot_dogs_left = 45

-- Prove the number of hot dogs sold during the next three innings is 27
theorem hot_dogs_sold_next_innings : total_initial - (hot_dogs_sold_first_innings + hot_dogs_left) = 27 :=
by
  sorry

end hot_dogs_sold_next_innings_l545_54584


namespace remainder_mod_17_zero_l545_54570

theorem remainder_mod_17_zero :
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  ( (x1 % 17) * (x2 % 17) * (x3 % 17) * (x4 % 17) * (x5 % 17) * (x6 % 17) ) % 17 = 0 :=
by
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  sorry

end remainder_mod_17_zero_l545_54570


namespace total_digits_written_total_digit_1_appearances_digit_at_position_2016_l545_54590

-- Problem 1
theorem total_digits_written : 
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  digits_1_to_9 + digits_10_to_99 + digits_100_to_999 = 2889 := 
by
  sorry

-- Problem 2
theorem total_digit_1_appearances : 
  let digit_1_as_1_digit := 1
  let digit_1_as_2_digits := 10 + 9
  let digit_1_as_3_digits := 100 + 9 * 10 + 9 * 10
  digit_1_as_1_digit + digit_1_as_2_digits + digit_1_as_3_digits = 300 := 
by
  sorry

-- Problem 3
theorem digit_at_position_2016 : 
  let position_1_to_99 := 9 + 90 * 2
  let remaining_positions := 2016 - position_1_to_99
  let three_digit_positions := remaining_positions / 3
  let specific_number := 100 + three_digit_positions - 1
  specific_number % 10 = 8 := 
by
  sorry

end total_digits_written_total_digit_1_appearances_digit_at_position_2016_l545_54590


namespace solve_for_x_l545_54516

theorem solve_for_x (x : ℝ) : 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ↔ x = -22 / 13 := 
by 
  sorry

end solve_for_x_l545_54516


namespace jonah_total_raisins_l545_54550

-- Define the amounts of yellow and black raisins added
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- The main statement to be proved
theorem jonah_total_raisins : yellow_raisins + black_raisins = 0.7 :=
by 
  sorry

end jonah_total_raisins_l545_54550


namespace sequence_bound_l545_54548

theorem sequence_bound (a b c : ℕ → ℝ) :
  (a 0 = 1) ∧ (b 0 = 0) ∧ (c 0 = 0) ∧
  (∀ n, n ≥ 1 → a n = a (n-1) + c (n-1) / n) ∧
  (∀ n, n ≥ 1 → b n = b (n-1) + a (n-1) / n) ∧
  (∀ n, n ≥ 1 → c n = c (n-1) + b (n-1) / n) →
  ∀ n, n ≥ 1 → |a n - (n + 1) / 3| < 2 / Real.sqrt (3 * n) :=
by sorry

end sequence_bound_l545_54548


namespace jason_needs_87_guppies_per_day_l545_54523

def guppies_needed_per_day (moray_eel_guppies : Nat)
  (betta_fish_number : Nat) (betta_fish_guppies : Nat)
  (angelfish_number : Nat) (angelfish_guppies : Nat)
  (lionfish_number : Nat) (lionfish_guppies : Nat) : Nat :=
  moray_eel_guppies +
  betta_fish_number * betta_fish_guppies +
  angelfish_number * angelfish_guppies +
  lionfish_number * lionfish_guppies

theorem jason_needs_87_guppies_per_day :
  guppies_needed_per_day 20 5 7 3 4 2 10 = 87 := by
  sorry

end jason_needs_87_guppies_per_day_l545_54523


namespace total_cost_is_correct_l545_54546

def bus_ride_cost : ℝ := 1.75
def train_ride_cost : ℝ := bus_ride_cost + 6.35
def total_cost : ℝ := bus_ride_cost + train_ride_cost

theorem total_cost_is_correct : total_cost = 9.85 :=
by
  -- proof here
  sorry

end total_cost_is_correct_l545_54546


namespace triangle_inequality_l545_54552

variable (R r e f : ℝ)

theorem triangle_inequality (h1 : ∃ (A B C : ℝ × ℝ), true)
                            (h2 : true) :
  R^2 - e^2 ≥ 4 * (r^2 - f^2) :=
by sorry

end triangle_inequality_l545_54552


namespace problem_may_not_be_equal_l545_54542

-- Define the four pairs of expressions
def expr_A (a b : ℕ) := (a + b) = (b + a)
def expr_B (a : ℕ) := (3 * a) = (a + a + a)
def expr_C (a b : ℕ) := (3 * (a + b)) ≠ (3 * a + b)
def expr_D (a : ℕ) := (a ^ 3) = (a * a * a)

-- State the theorem stating that the expression in condition C may not be equal
theorem problem_may_not_be_equal (a b : ℕ) : (3 * (a + b)) ≠ (3 * a + b) :=
by
  sorry

end problem_may_not_be_equal_l545_54542


namespace cost_per_bag_l545_54531

theorem cost_per_bag
  (friends : ℕ)
  (payment_per_friend : ℕ)
  (total_bags : ℕ)
  (total_cost : ℕ)
  (h1 : friends = 3)
  (h2 : payment_per_friend = 5)
  (h3 : total_bags = 5)
  (h4 : total_cost = friends * payment_per_friend) :
  total_cost / total_bags = 3 :=
by {
  sorry
}

end cost_per_bag_l545_54531


namespace servings_per_day_l545_54521

-- Conditions
def week_servings := 21
def days_per_week := 7

-- Question and Answer
theorem servings_per_day : week_servings / days_per_week = 3 := 
by
  sorry

end servings_per_day_l545_54521


namespace remaining_pages_after_a_week_l545_54504

-- Define the conditions
def total_pages : Nat := 381
def pages_read_initial : Nat := 149
def pages_per_day : Nat := 20
def days : Nat := 7

-- Define the final statement to prove
theorem remaining_pages_after_a_week :
  let pages_left_initial := total_pages - pages_read_initial
  let pages_read_week := pages_per_day * days
  let pages_remaining := pages_left_initial - pages_read_week
  pages_remaining = 92 := by
  sorry

end remaining_pages_after_a_week_l545_54504


namespace probability_queen_then_diamond_l545_54595

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l545_54595


namespace k_is_perfect_square_l545_54518

theorem k_is_perfect_square (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (k : ℕ)
  (h_k : k = (m + n)^2 / (4 * m * (m - n)^2 + 4)) 
  (h_int_k : k * (4 * m * (m - n)^2 + 4) = (m + n)^2) :
  ∃ x : ℕ, k = x^2 := 
sorry

end k_is_perfect_square_l545_54518


namespace digit_a_solution_l545_54583

theorem digit_a_solution :
  ∃ a : ℕ, a000 + a998 + a999 = 22997 → a = 7 :=
sorry

end digit_a_solution_l545_54583


namespace coffee_shop_distance_l545_54502

theorem coffee_shop_distance (resort_distance mall_distance : ℝ) 
  (coffee_dist : ℝ)
  (h_resort_distance : resort_distance = 400) 
  (h_mall_distance : mall_distance = 700)
  (h_equidistant : ∀ S, (S - resort_distance) ^ 2 + resort_distance ^ 2 = S ^ 2 ∧ 
  (mall_distance - S) ^ 2 + resort_distance ^ 2 = S ^ 2 → coffee_dist = S):
  coffee_dist = 464 := 
sorry

end coffee_shop_distance_l545_54502


namespace trajectory_of_midpoint_l545_54561

theorem trajectory_of_midpoint (Q : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ)
  (h1 : Q.1^2 - Q.2^2 = 1)
  (h2 : N = (2 * P.1 - Q.1, 2 * P.2 - Q.2))
  (h3 : N.1 + N.2 = 2)
  (h4 : (P.2 - Q.2) / (P.1 - Q.1) = 1) :
  2 * P.1^2 - 2 * P.2^2 - 2 * P.1 + 2 * P.2 - 1 = 0 :=
  sorry

end trajectory_of_midpoint_l545_54561


namespace part1_m_n_part2_k_l545_54520

-- Definitions of vectors a, b, and c
def veca : ℝ × ℝ := (3, 2)
def vecb : ℝ × ℝ := (-1, 2)
def vecc : ℝ × ℝ := (4, 1)

-- Part (1)
theorem part1_m_n : 
  ∃ (m n : ℝ), (-m + 4 * n = 3) ∧ (2 * m + n = 2) :=
sorry

-- Part (2)
theorem part2_k : 
  ∃ (k : ℝ), (3 + 4 * k) * 2 - (-5) * (2 + k) = 0 :=
sorry

end part1_m_n_part2_k_l545_54520


namespace pythagorean_triple_third_number_l545_54529

theorem pythagorean_triple_third_number (x : ℕ) (h1 : x^2 + 8^2 = 17^2) : x = 15 :=
sorry

end pythagorean_triple_third_number_l545_54529


namespace polar_coordinates_of_point_l545_54587

open Real

theorem polar_coordinates_of_point :
  ∃ r θ : ℝ, r = 4 ∧ θ = 5 * π / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
           (∃ x y : ℝ, x = 2 ∧ y = -2 * sqrt 3 ∧ x = r * cos θ ∧ y = r * sin θ) :=
sorry

end polar_coordinates_of_point_l545_54587


namespace min_value_y_l545_54538

theorem min_value_y (x : ℝ) (hx : x > 3) : 
  ∃ y, (∀ x > 3, y = min_value) ∧ min_value = 5 :=
by 
  sorry

end min_value_y_l545_54538


namespace exit_forest_strategy_l545_54553

/-- A strategy ensuring the parachutist will exit the forest with a path length of less than 2.5l -/
theorem exit_forest_strategy (l : Real) : 
  ∃ (path_length : Real), path_length < 2.5 * l :=
by
  use 2.278 * l
  sorry

end exit_forest_strategy_l545_54553


namespace consecutive_numbers_average_l545_54539

theorem consecutive_numbers_average (a b c d e f g : ℕ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 9)
  (h2 : 2 * a = g) : 
  7 = 7 :=
by sorry

end consecutive_numbers_average_l545_54539


namespace neg_existence_of_ge_impl_universal_lt_l545_54506

theorem neg_existence_of_ge_impl_universal_lt : (¬ ∃ x : ℕ, x^2 ≥ x) ↔ ∀ x : ℕ, x^2 < x := 
sorry

end neg_existence_of_ge_impl_universal_lt_l545_54506


namespace pos_real_unique_solution_l545_54585

theorem pos_real_unique_solution (x : ℝ) (hx_pos : 0 < x) (h : (x - 3) / 8 = 5 / (x - 8)) : x = 16 :=
sorry

end pos_real_unique_solution_l545_54585


namespace square_side_length_in_right_triangle_l545_54522

theorem square_side_length_in_right_triangle
  (AC BC : ℝ)
  (h1 : AC = 3)
  (h2 : BC = 7)
  (right_triangle : ∃ A B C : ℝ × ℝ, A = (3, 0) ∧ B = (0, 7) ∧ C = (0, 0) ∧ (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = AC^2 + BC^2) :
  ∃ s : ℝ, s = 2.1 :=
by
  -- Proof goes here
  sorry

end square_side_length_in_right_triangle_l545_54522


namespace train_stoppage_time_l545_54534

-- Definitions from conditions
def speed_without_stoppages := 60 -- kmph
def speed_with_stoppages := 36 -- kmph

-- Main statement to prove
theorem train_stoppage_time : (60 - 36) / 60 * 60 = 24 := by
  sorry

end train_stoppage_time_l545_54534


namespace find_a7_l545_54558

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l545_54558


namespace common_solutions_for_y_l545_54554

theorem common_solutions_for_y (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x^2 - 3 * y = 12) ↔ (y = -4 ∨ y = 1) :=
by
  sorry

end common_solutions_for_y_l545_54554


namespace dimes_count_l545_54544

def num_dimes (total_in_cents : ℤ) (value_quarter value_dime value_nickel : ℤ) (num_each : ℤ) : Prop :=
  total_in_cents = num_each * (value_quarter + value_dime + value_nickel)

theorem dimes_count (num_each : ℤ) :
  num_dimes 440 25 10 5 num_each → num_each = 11 :=
by sorry

end dimes_count_l545_54544


namespace find_speed_of_goods_train_l545_54509

variable (v : ℕ) -- Speed of the goods train in km/h

theorem find_speed_of_goods_train
  (h1 : 0 < v) 
  (h2 : 6 * v + 4 * 90 = 10 * v) :
  v = 36 :=
by
  sorry

end find_speed_of_goods_train_l545_54509


namespace cubic_yards_to_cubic_feet_l545_54566

theorem cubic_yards_to_cubic_feet (yards_to_feet: 1 = 3): 6 * 27 = 162 := by
  -- We know from the setup that:
  -- 1 cubic yard = 27 cubic feet
  -- Hence,
  -- 6 cubic yards = 6 * 27 = 162 cubic feet
  sorry

end cubic_yards_to_cubic_feet_l545_54566


namespace min_avg_score_less_than_record_l545_54563

theorem min_avg_score_less_than_record
  (old_record_avg : ℝ := 287.5)
  (players : ℕ := 6)
  (rounds : ℕ := 12)
  (total_points_11_rounds : ℝ := 19350.5)
  (bonus_points_9_rounds : ℕ := 300) :
  ∀ final_round_avg : ℝ, (final_round_avg = (old_record_avg * players * rounds - total_points_11_rounds + bonus_points_9_rounds) / players) →
  old_record_avg - final_round_avg = 12.5833 :=
by {
  sorry
}

end min_avg_score_less_than_record_l545_54563


namespace total_lives_l545_54580

/-- Suppose there are initially 4 players, then 5 more players join. Each player has 3 lives.
    Prove that the total number of lives is equal to 27. -/
theorem total_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
  (h_initial : initial_players = 4) (h_additional : additional_players = 5) (h_lives : lives_per_player = 3) : 
  initial_players + additional_players = 9 ∧ 
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l545_54580


namespace natasha_time_to_top_l545_54556

theorem natasha_time_to_top (T : ℝ) 
  (descent_time : ℝ) 
  (whole_journey_avg_speed : ℝ) 
  (climbing_speed : ℝ) 
  (desc_time_condition : descent_time = 2) 
  (whole_journey_avg_speed_condition : whole_journey_avg_speed = 3.5) 
  (climbing_speed_condition : climbing_speed = 2.625) 
  (distance_to_top : ℝ := climbing_speed * T) 
  (avg_speed_condition : whole_journey_avg_speed = 2 * distance_to_top / (T + descent_time)) :
  T = 4 := by
  sorry

end natasha_time_to_top_l545_54556


namespace solution_set_M_inequality_ab_l545_54528

def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem solution_set_M :
  {x | -3 ≤ x ∧ x ≤ 1} = { x : ℝ | f x ≤ 4 } :=
sorry

theorem inequality_ab
  (a b : ℝ) (h1 : -3 ≤ a ∧ a ≤ 1) (h2 : -3 ≤ b ∧ b ≤ 1) :
  (a^2 + 2 * a - 3) * (b^2 + 2 * b - 3) ≥ 0 :=
sorry

end solution_set_M_inequality_ab_l545_54528


namespace spaghetti_cost_l545_54511

theorem spaghetti_cost (hamburger_cost french_fry_cost soda_cost spaghetti_cost split_payment friends : ℝ) 
(hamburger_count : ℕ) (french_fry_count : ℕ) (soda_count : ℕ) (friend_count : ℕ)
(h_split_payment : split_payment * friend_count = 25)
(h_hamburger_cost : hamburger_cost = 3 * hamburger_count)
(h_french_fry_cost : french_fry_cost = 1.20 * french_fry_count)
(h_soda_cost : soda_cost = 0.5 * soda_count)
(h_total_order_cost : hamburger_cost + french_fry_cost + soda_cost + spaghetti_cost = split_payment * friend_count) :
spaghetti_cost = 2.70 :=
by {
  sorry
}

end spaghetti_cost_l545_54511


namespace ratio_of_c_and_d_l545_54598

theorem ratio_of_c_and_d 
  (x y c d : ℝ)
  (h₁ : 4 * x - 2 * y = c)
  (h₂ : 6 * y - 12 * x = d) 
  (h₃ : d ≠ 0) : 
  c / d = -1 / 3 :=
by
  sorry

end ratio_of_c_and_d_l545_54598


namespace evenFunctionExists_l545_54541

-- Definitions based on conditions
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def passesThroughPoints (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = p.2

-- Example function
def exampleEvenFunction (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

-- Points to pass through
def givenPoints : List (ℝ × ℝ) := [(-1, 0), (0.5, 2.5), (3, 0)]

-- Theorem to be proven
theorem evenFunctionExists : 
  isEvenFunction exampleEvenFunction ∧ passesThroughPoints exampleEvenFunction givenPoints :=
by
  sorry

end evenFunctionExists_l545_54541


namespace prob_3_tails_in_8_flips_l545_54581

def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def probability_of_3_tails : ℚ :=
  unfair_coin_probability 8 3 (2/3)

theorem prob_3_tails_in_8_flips :
  probability_of_3_tails = 448 / 6561 :=
by
  sorry

end prob_3_tails_in_8_flips_l545_54581


namespace union_M_N_l545_54510

def M := {x : ℝ | -2 < x ∧ x < -1}
def N := {x : ℝ | (1 / 2 : ℝ)^x ≤ 4}

theorem union_M_N :
  M ∪ N = {x : ℝ | x ≥ -2} :=
sorry

end union_M_N_l545_54510


namespace florida_north_dakota_license_plate_difference_l545_54577

theorem florida_north_dakota_license_plate_difference :
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  florida_license_plates = north_dakota_license_plates :=
by
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  show florida_license_plates = north_dakota_license_plates
  sorry

end florida_north_dakota_license_plate_difference_l545_54577


namespace abs_diff_squares_l545_54569

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end abs_diff_squares_l545_54569


namespace simplify_polynomial_l545_54530

variable (p : ℝ)

theorem simplify_polynomial :
  (7 * p ^ 5 - 4 * p ^ 3 + 8 * p ^ 2 - 5 * p + 3) + (- p ^ 5 + 3 * p ^ 3 - 7 * p ^ 2 + 6 * p + 2) =
  6 * p ^ 5 - p ^ 3 + p ^ 2 + p + 5 :=
by
  sorry

end simplify_polynomial_l545_54530


namespace number_of_smaller_cubes_l545_54515

theorem number_of_smaller_cubes (edge : ℕ) (N : ℕ) (h_edge : edge = 5)
  (h_divisors : ∃ (a b c : ℕ), a + b + c = N ∧ a * 1^3 + b * 2^3 + c * 3^3 = edge^3 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  N = 22 :=
by
  sorry

end number_of_smaller_cubes_l545_54515


namespace david_more_pushups_than_zachary_l545_54578

-- Definitions based on conditions
def david_pushups : ℕ := 37
def zachary_pushups : ℕ := 7

-- Theorem statement proving the answer
theorem david_more_pushups_than_zachary : david_pushups - zachary_pushups = 30 := by
  sorry

end david_more_pushups_than_zachary_l545_54578


namespace fish_caught_together_l545_54505

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end fish_caught_together_l545_54505


namespace solution_set_of_inequality_l545_54582

theorem solution_set_of_inequality :
  { x : ℝ | x * (x - 1) ≤ 0 } = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
by
sorry

end solution_set_of_inequality_l545_54582


namespace probability_of_odd_numbers_l545_54565

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l545_54565


namespace reema_loan_period_l545_54557

theorem reema_loan_period (P SI : ℕ) (R : ℚ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = 6) : 
  ∃ T : ℕ, SI = (P * R * T) / 100 ∧ T = 6 :=
by
  sorry

end reema_loan_period_l545_54557


namespace friend_reading_time_l545_54594

def my_reading_time : ℕ := 120  -- It takes me 120 minutes to read the novella

def speed_ratio : ℕ := 3  -- My friend reads three times as fast as I do

theorem friend_reading_time : my_reading_time / speed_ratio = 40 := by
  -- Proof
  sorry

end friend_reading_time_l545_54594


namespace annual_interest_rate_l545_54503

variable (P : ℝ) (t : ℝ)
variable (h1 : t = 25)
variable (h2 : ∀ r : ℝ, P * 2 = P * (1 + r * t))

theorem annual_interest_rate : ∃ r : ℝ, P * 2 = P * (1 + r * t) ∧ r = 0.04 := by
  sorry

end annual_interest_rate_l545_54503


namespace time_for_train_to_pass_pole_l545_54543

-- Definitions based on conditions
def train_length_meters : ℕ := 160
def train_speed_kmph : ℕ := 72

-- The calculated speed in m/s
def train_speed_mps : ℕ := train_speed_kmph * 1000 / 3600

-- The calculation of time taken to pass the pole
def time_to_pass_pole : ℕ := train_length_meters / train_speed_mps

-- The theorem statement
theorem time_for_train_to_pass_pole : time_to_pass_pole = 8 := sorry

end time_for_train_to_pass_pole_l545_54543


namespace trader_gain_l545_54575

-- Conditions
def cost_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the cost price of a pen
def selling_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the selling price of a pen
def gain_percentage : ℝ := 0.40 -- 40% gain

-- Statement of the problem to prove
theorem trader_gain (C : ℝ) (N : ℕ) : 
  (100 : ℕ) * C * gain_percentage = N * C → 
  N = 40 :=
by
  sorry

end trader_gain_l545_54575


namespace lassis_from_mangoes_l545_54533

theorem lassis_from_mangoes (m l m' : ℕ) (h : m' = 18) (hlm : l / m = 8 / 3) : l / m' = 48 / 18 :=
by
  sorry

end lassis_from_mangoes_l545_54533


namespace surface_area_change_l545_54591

noncomputable def original_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

noncomputable def new_surface_area (l w h c : ℝ) : ℝ :=
  original_surface_area l w h - 
  (3 * (c * c)) + 
  (2 * c * c)

theorem surface_area_change (l w h c : ℝ) (hl : l = 5) (hw : w = 4) (hh : h = 3) (hc : c = 2) :
  new_surface_area l w h c = original_surface_area l w h - 8 :=
by 
  sorry

end surface_area_change_l545_54591


namespace positive_root_condition_negative_root_condition_zero_root_condition_l545_54599

variable (a b c : ℝ)

-- Condition for a positive root
theorem positive_root_condition : 
  ((a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c)) ↔ (∃ x : ℝ, x > 0 ∧ a * x = b - c) :=
sorry

-- Condition for a negative root
theorem negative_root_condition : 
  ((a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c)) ↔ (∃ x : ℝ, x < 0 ∧ a * x = b - c) :=
sorry

-- Condition for a root equal to zero
theorem zero_root_condition : 
  (a ≠ 0 ∧ b = c) ↔ (∃ x : ℝ, x = 0 ∧ a * x = b - c) :=
sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l545_54599


namespace number_of_roses_cut_l545_54567

-- Let's define the initial and final conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses Mary cut from her garden
def roses_cut := final_roses - initial_roses

-- Now, we state the theorem we aim to prove
theorem number_of_roses_cut : roses_cut = 10 :=
by
  -- Proof goes here
  sorry

end number_of_roses_cut_l545_54567


namespace part1_l545_54589

theorem part1 (a b c : ℤ) (h : a + b + c = 0) : a^3 + a^2 * c - a * b * c + b^2 * c + b^3 = 0 := 
sorry

end part1_l545_54589


namespace largest_club_size_is_four_l545_54537

variable {Player : Type} -- Assume Player is a type

-- Definition of the lesson-taking relation
variable (takes_lessons_from : Player → Player → Prop)

-- Club conditions
def club_conditions (A B C : Player) : Prop :=
  (takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ takes_lessons_from B C ∧ ¬takes_lessons_from C A) ∨ 
  (¬takes_lessons_from A B ∧ ¬takes_lessons_from B C ∧ takes_lessons_from C A)

theorem largest_club_size_is_four :
  ∀ (club : Finset Player),
  (∀ (A B C : Player), A ≠ B → B ≠ C → C ≠ A → A ∈ club → B ∈ club → C ∈ club → club_conditions takes_lessons_from A B C) →
  club.card ≤ 4 :=
sorry

end largest_club_size_is_four_l545_54537


namespace average_side_length_of_squares_l545_54549

theorem average_side_length_of_squares (a1 a2 a3 a4 : ℕ) 
(h1 : a1 = 36) (h2 : a2 = 64) (h3 : a3 = 100) (h4 : a4 = 144) :
(Real.sqrt a1 + Real.sqrt a2 + Real.sqrt a3 + Real.sqrt a4) / 4 = 9 := 
by
  sorry

end average_side_length_of_squares_l545_54549


namespace least_n_condition_l545_54545

-- Define the conditions and the question in Lean 4
def jackson_position (n : ℕ) : ℕ := sorry  -- Defining the position of Jackson after n steps

def expected_value (n : ℕ) : ℝ := sorry  -- Defining the expected value E_n

theorem least_n_condition : ∃ n : ℕ, (1 / expected_value n > 2017) ∧ (∀ m < n, 1 / expected_value m ≤ 2017) ∧ n = 13446 :=
by {
  -- Jackson starts at position 1
  -- The conditions described in the problem will be formulated here
  -- We need to show that the least n such that 1 / E_n > 2017 is 13446
  sorry
}

end least_n_condition_l545_54545


namespace zoe_remaining_pictures_l545_54562

-- Definitions for the problem conditions
def monday_pictures := 24
def tuesday_pictures := 37
def wednesday_pictures := 50
def thursday_pictures := 33
def friday_pictures := 44

def rate_first := 4
def rate_second := 5
def rate_third := 6
def rate_fourth := 3
def rate_fifth := 7

def days_colored (start_day : ℕ) (end_day := 6) := end_day - start_day

def remaining_pictures (total_pictures : ℕ) (rate_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pictures - (rate_per_day * days)

-- Main theorem statement
theorem zoe_remaining_pictures : 
  remaining_pictures monday_pictures rate_first (days_colored 1) +
  remaining_pictures tuesday_pictures rate_second (days_colored 2) +
  remaining_pictures wednesday_pictures rate_third (days_colored 3) +
  remaining_pictures thursday_pictures rate_fourth (days_colored 4) +
  remaining_pictures friday_pictures rate_fifth (days_colored 5) = 117 :=
  sorry

end zoe_remaining_pictures_l545_54562


namespace value_of_half_plus_five_l545_54519

theorem value_of_half_plus_five (n : ℕ) (h₁ : n = 20) : (n / 2) + 5 = 15 := 
by {
  sorry
}

end value_of_half_plus_five_l545_54519


namespace travel_time_between_resorts_l545_54517

theorem travel_time_between_resorts
  (num_cars : ℕ)
  (car_interval : ℕ)
  (opposing_encounter_time : ℕ)
  (travel_time : ℕ) :
  num_cars = 80 →
  car_interval = 15 →
  (opposing_encounter_time * 2 * car_interval / travel_time) = num_cars →
  travel_time = 20 :=
by
  sorry

end travel_time_between_resorts_l545_54517


namespace binary_to_decimal_is_1023_l545_54574

-- Define the binary number 1111111111 in terms of its decimal representation
def binary_to_decimal : ℕ :=
  (1 * 2^9 + 1 * 2^8 + 1 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0)

-- The theorem statement
theorem binary_to_decimal_is_1023 : binary_to_decimal = 1023 :=
by
  sorry

end binary_to_decimal_is_1023_l545_54574


namespace infinite_series_sum_l545_54568

theorem infinite_series_sum
  (a b : ℝ)
  (h1 : (∑' n : ℕ, a / (b ^ (n + 1))) = 4) :
  (∑' n : ℕ, a / ((a + b) ^ (n + 1))) = 4 / 5 := 
sorry

end infinite_series_sum_l545_54568


namespace floor_equation_solution_l545_54513

theorem floor_equation_solution (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (⌊ (a^2 : ℝ) / b ⌋ + ⌊ (b^2 : ℝ) / a ⌋ = ⌊ (a^2 + b^2 : ℝ) / (a * b) ⌋ + a * b) ↔
    (∃ n : ℕ, a = n ∧ b = n^2 + 1) ∨ (∃ n : ℕ, a = n^2 + 1 ∧ b = n) :=
sorry

end floor_equation_solution_l545_54513


namespace greendale_high_school_points_l545_54564

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l545_54564


namespace min_a_4_l545_54512

theorem min_a_4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 9 * x + y = x * y) : 
  4 * x + y ≥ 25 :=
sorry

end min_a_4_l545_54512


namespace sequence_formula_l545_54560

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n - 1) = 2^(n-1)) : a n = 2^n - 1 := 
sorry

end sequence_formula_l545_54560


namespace leah_probability_of_seeing_change_l545_54586

open Set

-- Define the length of each color interval
def green_duration := 45
def yellow_duration := 5
def red_duration := 35

-- Total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Leah's viewing intervals
def change_intervals : Set (ℕ × ℕ) :=
  {(40, 45), (45, 50), (80, 85)}

-- Probability calculation
def favorable_time := 15
def probability_of_change := (favorable_time : ℚ) / (total_cycle_duration : ℚ)

theorem leah_probability_of_seeing_change : probability_of_change = 3 / 17 :=
by
  -- We use sorry here as we are only required to state the theorem without proof.
  sorry

end leah_probability_of_seeing_change_l545_54586


namespace calculate_expression_l545_54571

theorem calculate_expression : (1000^2) / (252^2 - 248^2) = 500 := sorry

end calculate_expression_l545_54571


namespace solve_for_y_l545_54527

theorem solve_for_y (x y : ℤ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 :=
sorry

end solve_for_y_l545_54527


namespace y_range_l545_54555

theorem y_range (x y : ℝ) (h1 : 4 * x + y = 1) (h2 : -1 < x) (h3 : x ≤ 2) : -7 ≤ y ∧ y < -3 := 
by
  sorry

end y_range_l545_54555


namespace min_value_geometric_sequence_l545_54597

-- Definitions based on conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q

-- We need to state the problem using the above definitions
theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (s t : ℕ) 
  (h_seq : is_geometric_sequence a q) 
  (h_q : q ≠ 1) 
  (h_st : a s * a t = (a 5) ^ 2) 
  (h_s_pos : s > 0) 
  (h_t_pos : t > 0) 
  : 4 / s + 1 / (4 * t) = 5 / 8 := sorry

end min_value_geometric_sequence_l545_54597


namespace total_surface_area_of_cube_l545_54507

theorem total_surface_area_of_cube (edge_sum : ℕ) (h_edge_sum : edge_sum = 180) :
  ∃ (S : ℕ), S = 1350 := 
by
  sorry

end total_surface_area_of_cube_l545_54507


namespace PetyaColorsAll64Cells_l545_54532

-- Assuming a type for representing cell coordinates
structure Cell where
  row : ℕ
  col : ℕ

def isColored (c : Cell) : Prop := true  -- All cells are colored
def LShapedFigures : Set (Set Cell) := sorry  -- Define what constitutes an L-shaped figure

theorem PetyaColorsAll64Cells :
  (∀ tilesVector ∈ LShapedFigures, ¬∀ cell ∈ tilesVector, isColored cell) → (∀ c : Cell, c.row < 8 ∧ c.col < 8 ∧ isColored c) := sorry

end PetyaColorsAll64Cells_l545_54532


namespace sequence_is_geometric_not_arithmetic_l545_54573

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b / a = c / b

theorem sequence_is_geometric_not_arithmetic :
  ∀ (a₁ a₂ an : ℕ), a₁ = 3 ∧ a₂ = 9 ∧ an = 729 →
    ¬ is_arithmetic_sequence a₁ a₂ an ∧ is_geometric_sequence a₁ a₂ an :=
by
  intros a₁ a₂ an h
  sorry

end sequence_is_geometric_not_arithmetic_l545_54573


namespace box_height_l545_54593

variables (length width : ℕ) (cube_volume cubes total_volume : ℕ)
variable (height : ℕ)

theorem box_height :
  length = 12 →
  width = 16 →
  cube_volume = 3 →
  cubes = 384 →
  total_volume = cubes * cube_volume →
  total_volume = length * width * height →
  height = 6 :=
by
  intros
  sorry

end box_height_l545_54593


namespace radius_of_sphere_inscribed_in_box_l545_54576

theorem radius_of_sphere_inscribed_in_box (a b c s : ℝ)
  (h1 : a + b + c = 42)
  (h2 : 2 * (a * b + b * c + c * a) = 576)
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) :
  s = 3 * Real.sqrt 33 :=
by sorry

end radius_of_sphere_inscribed_in_box_l545_54576
