import Mathlib

namespace min_y_value_l176_17608

noncomputable def y (x : ℝ) : ℝ := x^2 + 16 * x + 20

theorem min_y_value : ∀ (x : ℝ), x ≥ -3 → y x ≥ -19 :=
by
  intro x hx
  sorry

end min_y_value_l176_17608


namespace Genevieve_drinks_pints_l176_17690

theorem Genevieve_drinks_pints :
  ∀ (total_gallons : ℝ) (num_thermoses : ℕ) (pints_per_gallon : ℝ) (genevieve_thermoses : ℕ),
  total_gallons = 4.5 → num_thermoses = 18 → pints_per_gallon = 8 → genevieve_thermoses = 3 →
  (genevieve_thermoses * ((total_gallons / num_thermoses) * pints_per_gallon) = 6) :=
by
  intros total_gallons num_thermoses pints_per_gallon genevieve_thermoses
  intros h1 h2 h3 h4
  sorry

end Genevieve_drinks_pints_l176_17690


namespace staffing_ways_l176_17691

def total_resumes : ℕ := 30
def unsuitable_resumes : ℕ := 10
def suitable_resumes : ℕ := total_resumes - unsuitable_resumes
def position_count : ℕ := 5

theorem staffing_ways :
  20 * 19 * 18 * 17 * 16 = 1860480 := by
  sorry

end staffing_ways_l176_17691


namespace number_100_in_row_15_l176_17627

theorem number_100_in_row_15 (A : ℕ) (H1 : 1 ≤ A)
  (H2 : ∀ n : ℕ, n > 0 → n ≤ 100 * A)
  (H3 : ∃ k : ℕ, 4 * A + 1 ≤ 31 ∧ 31 ≤ 5 * A ∧ k = 5):
  ∃ r : ℕ, (14 * A + 1 ≤ 100 ∧ 100 ≤ 15 * A ∧ r = 15) :=
by {
  sorry
}

end number_100_in_row_15_l176_17627


namespace expand_binomials_l176_17607

theorem expand_binomials (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 :=
by
  sorry

end expand_binomials_l176_17607


namespace count_factors_multiple_of_150_l176_17680

theorem count_factors_multiple_of_150 (n : ℕ) (h : n = 2^10 * 3^14 * 5^8) : 
  ∃ k, k = 980 ∧ ∀ d : ℕ, d ∣ n → 150 ∣ d → (d.factors.card = k) := sorry

end count_factors_multiple_of_150_l176_17680


namespace carlos_initial_blocks_l176_17624

theorem carlos_initial_blocks (g : ℕ) (l : ℕ) (total : ℕ) : g = 21 → l = 37 → total = g + l → total = 58 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carlos_initial_blocks_l176_17624


namespace second_smallest_integer_l176_17692

theorem second_smallest_integer (x y z w v : ℤ) (h_avg : (x + y + z + w + v) / 5 = 69)
  (h_median : z = 83) (h_mode : w = 85 ∧ v = 85) (h_range : 85 - x = 70) :
  y = 77 :=
by
  sorry

end second_smallest_integer_l176_17692


namespace polyomino_count_5_l176_17686

-- Definition of distinct polyomino counts for n = 2, 3, and 4.
def polyomino_count (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 5
  else 0

-- Theorem stating the distinct polyomino count for n = 5
theorem polyomino_count_5 : polyomino_count 5 = 12 :=
by {
  -- Proof steps would go here, but for now we use sorry.
  sorry
}

end polyomino_count_5_l176_17686


namespace problem_statement_l176_17606

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - 2 * a * x

theorem problem_statement (a : ℝ) (x1 x2 : ℝ) (h_a : a > 1) (h1 : x1 < x2) (h_extreme : f a x1 = 0 ∧ f a x2 = 0) : 
  f a x2 < -3/2 :=
sorry

end problem_statement_l176_17606


namespace tiling_possible_values_of_n_l176_17658

-- Define the sizes of the grid and the tiles
def grid_size : ℕ × ℕ := (9, 7)
def l_tile_size : ℕ := 3  -- L-shaped tile composed of three unit squares
def square_tile_size : ℕ := 4  -- square tile composed of four unit squares

-- Formalize the properties of the grid and the constraints for the tiling
def total_squares : ℕ := grid_size.1 * grid_size.2
def white_squares (n : ℕ) : ℕ := 3 * n
def black_squares (n : ℕ) : ℕ := n
def total_black_squares : ℕ := 20
def total_white_squares : ℕ := total_squares - total_black_squares

-- The main theorem statement
theorem tiling_possible_values_of_n (n : ℕ) : 
  (n = 2 ∨ n = 5 ∨ n = 8 ∨ n = 11 ∨ n = 14 ∨ n = 17 ∨ n = 20) ↔
  (3 * (total_white_squares - 2 * (20 - n)) / 3 + n = 23 ∧ n + (total_black_squares - n) = 20) :=
sorry

end tiling_possible_values_of_n_l176_17658


namespace find_n_l176_17603

theorem find_n (n : ℚ) : 1 / 2 + 2 / 3 + 3 / 4 + n / 12 = 2 ↔ n = 1 := by
  -- proof here
  sorry

end find_n_l176_17603


namespace eugene_swim_time_l176_17605

-- Define the conditions
variable (S : ℕ) -- Swim time on Sunday
variable (swim_time_mon : ℕ := 30) -- Swim time on Monday
variable (swim_time_tue : ℕ := 45) -- Swim time on Tuesday
variable (average_swim_time : ℕ := 34) -- Average swim time over three days

-- The total swim time over three days
def total_swim_time := S + swim_time_mon + swim_time_tue

-- The problem statement: Prove that given the conditions, Eugene swam for 27 minutes on Sunday.
theorem eugene_swim_time : total_swim_time S = 3 * average_swim_time → S = 27 := by
  -- Proof process will follow here
  sorry

end eugene_swim_time_l176_17605


namespace star_polygon_edges_congruent_l176_17662

theorem star_polygon_edges_congruent
  (n : ℕ)
  (α β : ℝ)
  (h1 : ∀ i j : ℕ, i ≠ j → (n = 133))
  (h2 : α = (5 / 14) * β)
  (h3 : n * (α + β) = 360) :
n = 133 :=
by sorry

end star_polygon_edges_congruent_l176_17662


namespace round_robin_teams_l176_17676

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 :=
sorry

end round_robin_teams_l176_17676


namespace tetrahedron_altitude_exsphere_eq_l176_17675

variable (h₁ h₂ h₃ h₄ r₁ r₂ r₃ r₄ : ℝ)

/-- The equality of the sum of the reciprocals of the heights and the radii of the exspheres of 
a tetrahedron -/
theorem tetrahedron_altitude_exsphere_eq :
  2 * (1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄) = (1 / r₁ + 1 / r₂ + 1 / r₃ + 1 / r₄) :=
sorry

end tetrahedron_altitude_exsphere_eq_l176_17675


namespace Micheal_completion_time_l176_17671

variable (W M A : ℝ)

-- Conditions
def condition1 (W M A : ℝ) : Prop := M + A = W / 20
def condition2 (W M A : ℝ) : Prop := A = (W - 14 * (M + A)) / 10

-- Goal
theorem Micheal_completion_time :
  (condition1 W M A) →
  (condition2 W M A) →
  M = W / 50 :=
by
  intros h1 h2
  sorry

end Micheal_completion_time_l176_17671


namespace smallest_n_for_nonzero_constant_term_l176_17634

theorem smallest_n_for_nonzero_constant_term : 
  ∃ n : ℕ, (∃ r : ℕ, n = 5 * r / 3) ∧ (n > 0) ∧ ∀ m : ℕ, (m > 0) → (∃ s : ℕ, m = 5 * s / 3) → n ≤ m :=
by sorry

end smallest_n_for_nonzero_constant_term_l176_17634


namespace f_neg_l176_17645

/-- Define f(x) as an odd function --/
def f : ℝ → ℝ := sorry

/-- The property of odd functions: f(-x) = -f(x) --/
axiom odd_fn_property (x : ℝ) : f (-x) = -f x

/-- Define the function for non-negative x --/
axiom f_nonneg (x : ℝ) (hx : 0 ≤ x) : f x = x + 1

/-- The goal is to determine f(x) when x < 0 --/
theorem f_neg (x : ℝ) (h : x < 0) : f x = x - 1 :=
by
  sorry

end f_neg_l176_17645


namespace polynomial_evaluation_l176_17682

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 3 * x - 10 = 0) (h2 : 0 < x) : 
  x^3 - 3 * x^2 - 10 * x + 5 = 5 :=
sorry

end polynomial_evaluation_l176_17682


namespace proof_math_problem_l176_17694

noncomputable def math_problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧ 
  ω^4 = 1 ∧ ω ≠ 1 ∧ 
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2)

theorem proof_math_problem (a b c d : ℝ) (ω : ℂ) (h: math_problem a b c d ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
sorry

end proof_math_problem_l176_17694


namespace tiles_required_for_floor_l176_17612

def tileDimensionsInFeet (width_in_inches : ℚ) (length_in_inches : ℚ) : ℚ × ℚ :=
  (width_in_inches / 12, length_in_inches / 12)

def area (length : ℚ) (width : ℚ) : ℚ :=
  length * width

noncomputable def numberOfTiles (floor_length : ℚ) (floor_width : ℚ) (tile_length : ℚ) (tile_width : ℚ) : ℚ :=
  (area floor_length floor_width) / (area tile_length tile_width)

theorem tiles_required_for_floor : numberOfTiles 10 15 (5/12) (2/3) = 540 := by
  sorry

end tiles_required_for_floor_l176_17612


namespace points_in_quadrants_l176_17677

theorem points_in_quadrants (x y : ℝ) (h₁ : y > 3 * x) (h₂ : y > 6 - x) : 
  (0 <= x ∧ 0 <= y) ∨ (x <= 0 ∧ 0 <= y) :=
by
  sorry

end points_in_quadrants_l176_17677


namespace result_of_dividing_295_by_5_and_adding_6_is_65_l176_17653

theorem result_of_dividing_295_by_5_and_adding_6_is_65 : (295 / 5) + 6 = 65 := by
  sorry

end result_of_dividing_295_by_5_and_adding_6_is_65_l176_17653


namespace fraction_pos_integer_l176_17679

theorem fraction_pos_integer (p : ℕ) (hp : 0 < p) : (∃ (k : ℕ), k = 1 + (2 * p + 53) / (3 * p - 8)) ↔ p = 3 := 
by
  sorry

end fraction_pos_integer_l176_17679


namespace powers_of_i_sum_l176_17664

theorem powers_of_i_sum :
  ∀ (i : ℂ), 
  (i^1 = i) ∧ (i^2 = -1) ∧ (i^3 = -i) ∧ (i^4 = 1) →
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 :=
by
  intros i h
  sorry

end powers_of_i_sum_l176_17664


namespace min_value_of_function_l176_17622

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  ∃ x, (x > -1) ∧ (x = 0) ∧ ∀ y, (y = x + (1 / (x + 1))) → y ≥ 1 := 
sorry

end min_value_of_function_l176_17622


namespace distribution_properties_l176_17641

theorem distribution_properties (m d j s k : ℝ) (h1 : True)
  (h2 : True)
  (h3 : True)
  (h4 : 68 ≤ 100 ∧ 68 ≥ 0) -- 68% being a valid percentage
  : j = 84 ∧ s = s ∧ k = k :=
by
  -- sorry is used to highlight the proof is not included
  sorry

end distribution_properties_l176_17641


namespace g_symmetric_l176_17698

theorem g_symmetric (g : ℝ → ℝ) (h₀ : ∀ x, x ≠ 0 → (g x + 3 * g (1 / x) = 4 * x ^ 2)) : 
  ∀ x : ℝ, x ≠ 0 → g x = g (-x) :=
by 
  sorry

end g_symmetric_l176_17698


namespace pascal_triangle_fifth_number_l176_17693

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l176_17693


namespace remainder_of_50_pow_2019_plus_1_mod_7_l176_17646

theorem remainder_of_50_pow_2019_plus_1_mod_7 :
  (50 ^ 2019 + 1) % 7 = 2 :=
by
  sorry

end remainder_of_50_pow_2019_plus_1_mod_7_l176_17646


namespace minimize_expression_l176_17678

theorem minimize_expression (x : ℝ) : 3 * x^2 - 12 * x + 1 ≥ 3 * 2^2 - 12 * 2 + 1 :=
by sorry

end minimize_expression_l176_17678


namespace halfway_fraction_l176_17643

theorem halfway_fraction : 
  ∃ (x : ℚ), x = 1/2 * ((2/3) + (4/5)) ∧ x = 11/15 :=
by
  sorry

end halfway_fraction_l176_17643


namespace find_f2_l176_17666

theorem find_f2 :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 2 * f x - 3 * f (1 / x) = x ^ 2) ∧ f 2 = 93 / 32) :=
sorry

end find_f2_l176_17666


namespace MsSatosClassRatioProof_l176_17640

variable (g b : ℕ) -- g is the number of girls, b is the number of boys

def MsSatosClassRatioProblem : Prop :=
  (g = b + 6) ∧ (g + b = 32) → g / b = 19 / 13

theorem MsSatosClassRatioProof : MsSatosClassRatioProblem g b := by
  sorry

end MsSatosClassRatioProof_l176_17640


namespace average_book_width_l176_17644

noncomputable def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]
def number_of_books : ℕ := 7
def total_sum_of_widths : ℚ := 34.5

theorem average_book_width :
  ((book_widths.sum) / number_of_books) = 241/49 :=
by
  sorry

end average_book_width_l176_17644


namespace outfits_count_l176_17626

def num_outfits (n : Nat) (total_colors : Nat) : Nat :=
  let total_combinations := n * n * n
  let undesirable_combinations := total_colors
  total_combinations - undesirable_combinations

theorem outfits_count : num_outfits 5 5 = 120 :=
  by
  sorry

end outfits_count_l176_17626


namespace solve_inequality_l176_17695

theorem solve_inequality : 
  {x : ℝ | -4 * x^2 + 7 * x + 2 < 0} = {x : ℝ | x < -1/4} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solve_inequality_l176_17695


namespace average_goals_per_game_l176_17652

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end average_goals_per_game_l176_17652


namespace find_a_l176_17611

theorem find_a (a : ℝ) (h : a ≠ 0) :
  (∀ x, -1 ≤ x ∧ x ≤ 4 → ax - a + 2 ≤ 7) →
  (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ ax - a + 2 = 7) →
  (a = 5/3 ∨ a = -5/2) :=
by
  sorry

end find_a_l176_17611


namespace solve_equation_l176_17628

theorem solve_equation :
  ∀ x : ℝ, (101 * x ^ 2 - 18 * x + 1) ^ 2 - 121 * x ^ 2 * (101 * x ^ 2 - 18 * x + 1) + 2020 * x ^ 4 = 0 ↔ 
    x = 1 / 18 ∨ x = 1 / 9 :=
by
  intro x
  sorry

end solve_equation_l176_17628


namespace parabola_hyperbola_focus_l176_17647

noncomputable def focus_left (p : ℝ) : ℝ × ℝ :=
  (-p / 2, 0)

theorem parabola_hyperbola_focus (p : ℝ) (hp : p > 0) : 
  focus_left p = (-2, 0) ↔ p = 4 :=
by 
  sorry

end parabola_hyperbola_focus_l176_17647


namespace binomial_parameters_l176_17655

theorem binomial_parameters
  (n : ℕ) (p : ℚ)
  (hE : n * p = 12) (hD : n * p * (1 - p) = 2.4) :
  n = 15 ∧ p = 4 / 5 :=
by
  sorry

end binomial_parameters_l176_17655


namespace one_greater_others_less_l176_17617

theorem one_greater_others_less {a b c : ℝ} (h1 : a > 0 ∧ b > 0 ∧ c > 0) (h2 : a * b * c = 1) (h3 : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
by
  sorry

end one_greater_others_less_l176_17617


namespace function_neither_even_nor_odd_l176_17604

noncomputable def f (x : ℝ) : ℝ := (4 * x ^ 3 - 3) / (x ^ 6 + 2)

theorem function_neither_even_nor_odd : 
  (∀ x : ℝ, f (-x) ≠ f x) ∧ (∀ x : ℝ, f (-x) ≠ -f x) :=
by
  sorry

end function_neither_even_nor_odd_l176_17604


namespace extremum_points_l176_17663

noncomputable def f (x1 x2 : ℝ) : ℝ := x1 * x2 / (1 + x1^2 * x2^2)

theorem extremum_points :
  (f 0 0 = 0) ∧
  (∀ x1 : ℝ, f x1 (-1 / x1) = -1 / 2) ∧
  (∀ x1 : ℝ, f x1 (1 / x1) = 1 / 2) ∧
  ∀ y1 y2 : ℝ, (f 0 0 < f y1 y2 → (0 < y1 ∧ 0 < y2)) ∧ 
             (f 0 0 > f y1 y2 → (0 > y1 ∧ 0 > y2)) :=
by
  sorry

end extremum_points_l176_17663


namespace min_value_of_function_l176_17638

open Real

theorem min_value_of_function (x y : ℝ) (h : 2 * x + 8 * y = 3) : ∃ (min_value : ℝ), min_value = -19 / 20 ∧ ∀ (x y : ℝ), 2 * x + 8 * y = 3 → x^2 + 4 * y^2 - 2 * x ≥ -19 / 20 :=
by
  sorry

end min_value_of_function_l176_17638


namespace operation_on_original_number_l176_17616

theorem operation_on_original_number (f : ℕ → ℕ) (x : ℕ) (h : 3 * (f x + 9) = 51) (hx : x = 4) :
  f x = 2 * x :=
by
  sorry

end operation_on_original_number_l176_17616


namespace solve_for_y_l176_17629

theorem solve_for_y (x y : ℝ) (h : 3 * x - 5 * y = 7) : y = (3 * x - 7) / 5 :=
sorry

end solve_for_y_l176_17629


namespace squido_oysters_l176_17631

theorem squido_oysters (S C : ℕ) (h1 : C ≥ 2 * S) (h2 : S + C = 600) : S = 200 :=
sorry

end squido_oysters_l176_17631


namespace range_of_a_l176_17625

open Real

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem range_of_a (a : ℝ) (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) :
  (f (a * sin θ) + f (1 - a) > 0) → a ≤ 1 :=
sorry

end range_of_a_l176_17625


namespace tickets_used_correct_l176_17609

def ferris_wheel_rides : ℕ := 7
def bumper_car_rides : ℕ := 3
def cost_per_ride : ℕ := 5

def total_rides : ℕ := ferris_wheel_rides + bumper_car_rides
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_correct : total_tickets_used = 50 := by
  sorry

end tickets_used_correct_l176_17609


namespace marbles_remainder_l176_17687

theorem marbles_remainder 
  (g r p : ℕ) 
  (hg : g % 8 = 5) 
  (hr : r % 7 = 2) 
  (hp : p % 7 = 4) : 
  (r + p + g) % 7 = 4 := 
sorry

end marbles_remainder_l176_17687


namespace cubes_with_odd_neighbors_in_5x5x5_l176_17602

theorem cubes_with_odd_neighbors_in_5x5x5 (unit_cubes : Fin 125 → ℕ) 
  (neighbors : ∀ (i : Fin 125), Fin 125 → Prop) : ∃ n, n = 62 := 
by
  sorry

end cubes_with_odd_neighbors_in_5x5x5_l176_17602


namespace inequality_proof_l176_17668

theorem inequality_proof (x y : ℝ) :
  abs ((x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2))) ≤ 1 / 2 := 
sorry

end inequality_proof_l176_17668


namespace positive_int_solution_is_perfect_square_l176_17642

variable (t n : ℤ)

theorem positive_int_solution_is_perfect_square (ht : ∃ n : ℕ, n > 0 ∧ n^2 + (4 * t - 1) * n + 4 * t^2 = 0) : ∃ k : ℕ, n = k^2 :=
  sorry

end positive_int_solution_is_perfect_square_l176_17642


namespace length_of_place_mat_l176_17636

noncomputable def length_of_mat
  (R : ℝ)
  (w : ℝ)
  (n : ℕ)
  (θ : ℝ) : ℝ :=
  2 * R * Real.sin (θ / 2)

theorem length_of_place_mat :
  ∃ y : ℝ, y = length_of_mat 5 1 7 (360 / 7) := by
  use 4.38
  sorry

end length_of_place_mat_l176_17636


namespace large_pizza_slices_l176_17683

-- Definitions and conditions based on the given problem
def slicesEatenByPhilAndre : ℕ := 9 * 2
def slicesLeft : ℕ := 2 * 2
def slicesOnSmallCheesePizza : ℕ := 8
def totalSlices : ℕ := slicesEatenByPhilAndre + slicesLeft

-- The theorem to be proven
theorem large_pizza_slices (slicesEatenByPhilAndre slicesLeft slicesOnSmallCheesePizza : ℕ) :
  slicesEatenByPhilAndre = 18 ∧ slicesLeft = 4 ∧ slicesOnSmallCheesePizza = 8 →
  totalSlices - slicesOnSmallCheesePizza = 14 :=
by
  intros h
  sorry

end large_pizza_slices_l176_17683


namespace minimum_soldiers_to_add_l176_17619

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l176_17619


namespace sale_in_fifth_month_l176_17620

-- Define the sale amounts and average sale required.
def sale_first_month : ℕ := 7435
def sale_second_month : ℕ := 7920
def sale_third_month : ℕ := 7855
def sale_fourth_month : ℕ := 8230
def sale_sixth_month : ℕ := 6000
def average_sale_required : ℕ := 7500

-- State the theorem to determine the sale in the fifth month.
theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 avg : ℕ)
  (h1 : s1 = sale_first_month)
  (h2 : s2 = sale_second_month)
  (h3 : s3 = sale_third_month)
  (h4 : s4 = sale_fourth_month)
  (h6 : s6 = sale_sixth_month)
  (havg : avg = average_sale_required) :
  s1 + s2 + s3 + s4 + s6 + x = 6 * avg →
  x = 7560 :=
by
  sorry

end sale_in_fifth_month_l176_17620


namespace right_angled_triangle_l176_17651
  
theorem right_angled_triangle (x : ℝ) (hx : 0 < x) :
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  a^2 + b^2 = c^2 :=
by
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  sorry

end right_angled_triangle_l176_17651


namespace pythagorean_triple_l176_17696

theorem pythagorean_triple {a b c : ℕ} (h : a * a + b * b = c * c) (gcd_abc : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ m n : ℕ, a = 2 * m * n ∧ b = m * m - n * n ∧ c = m * m + n * n :=
sorry

end pythagorean_triple_l176_17696


namespace y_relationship_range_of_x_l176_17614

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end y_relationship_range_of_x_l176_17614


namespace event_B_is_certain_l176_17630

-- Define the event that the sum of two sides of a triangle is greater than the third side
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the term 'certain event'
def certain_event (E : Prop) : Prop := E

/-- Prove that the event "the sum of two sides of a triangle is greater than the third side" is a certain event -/
theorem event_B_is_certain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  certain_event (triangle_inequality a b c) :=
sorry

end event_B_is_certain_l176_17630


namespace compute_g_five_times_l176_17657

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then - x^3 else x + 10

theorem compute_g_five_times (x : ℤ) (h : x = 2) : g (g (g (g (g x)))) = -8 := by
  sorry

end compute_g_five_times_l176_17657


namespace time_difference_l176_17600

-- Definitions for the conditions
def blocks_to_office : Nat := 12
def walk_time_per_block : Nat := 1 -- time in minutes
def bike_time_per_block : Nat := 20 / 60 -- time in minutes, converted from seconds

-- Definitions for the total times
def walk_time : Nat := blocks_to_office * walk_time_per_block
def bike_time : Nat := blocks_to_office * bike_time_per_block

-- Theorem statement
theorem time_difference : walk_time - bike_time = 8 := by
  -- Proof omitted
  sorry

end time_difference_l176_17600


namespace eggs_needed_for_recipe_l176_17618

noncomputable section

theorem eggs_needed_for_recipe 
  (total_eggs : ℕ) 
  (rotten_eggs : ℕ) 
  (prob_all_rotten : ℝ)
  (h_total : total_eggs = 36)
  (h_rotten : rotten_eggs = 3)
  (h_prob : prob_all_rotten = 0.0047619047619047615) 
  : (2 : ℕ) = 2 :=
by
  sorry

end eggs_needed_for_recipe_l176_17618


namespace common_difference_l176_17669

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : a 2 = 1 + d) (h4 : a 4 = 1 + 3 * d) (h5 : a 5 = 1 + 4 * d) 
  (h_geometric : (a 4)^2 = a 2 * a 5) 
  (h_nonzero : d ≠ 0) : 
  d = 1 / 5 :=
by sorry

end common_difference_l176_17669


namespace ellipse_minor_axis_length_l176_17637

theorem ellipse_minor_axis_length
  (semi_focal_distance : ℝ)
  (eccentricity : ℝ)
  (semi_focal_distance_eq : semi_focal_distance = 2)
  (eccentricity_eq : eccentricity = 2 / 3) :
  ∃ minor_axis_length : ℝ, minor_axis_length = 2 * Real.sqrt 5 :=
by
  sorry

end ellipse_minor_axis_length_l176_17637


namespace intersection_of_P_and_Q_l176_17639

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2 ∧ x ∈ Set.univ}
def Q : Set ℤ := {x | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {-2, -1, 0} :=
sorry

end intersection_of_P_and_Q_l176_17639


namespace largest_possible_red_socks_l176_17633

theorem largest_possible_red_socks (r b : ℕ) (h1 : 0 < r) (h2 : 0 < b)
  (h3 : r + b ≤ 2500) (h4 : r > b) :
  r * (r - 1) + b * (b - 1) = 3/5 * (r + b) * (r + b - 1) → r ≤ 1164 :=
by sorry

end largest_possible_red_socks_l176_17633


namespace polygon_sides_eq_six_l176_17654

theorem polygon_sides_eq_six (n : ℕ) (S_i S_e : ℕ) :
  S_i = 2 * S_e →
  S_e = 360 →
  (n - 2) * 180 = S_i →
  n = 6 :=
by
  sorry

end polygon_sides_eq_six_l176_17654


namespace rectangle_area_l176_17661

theorem rectangle_area (b l : ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end rectangle_area_l176_17661


namespace five_alpha_plus_two_beta_is_45_l176_17623

theorem five_alpha_plus_two_beta_is_45
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (tan_β : Real.tan β = 3 / 79) :
  5 * α + 2 * β = π / 4 :=
by
  sorry

end five_alpha_plus_two_beta_is_45_l176_17623


namespace triangle_length_product_square_l176_17665

theorem triangle_length_product_square 
  (a1 : ℝ) (b1 : ℝ) (c1 : ℝ) (a2 : ℝ) (b2 : ℝ) (c2 : ℝ) 
  (h1 : a1 * b1 / 2 = 3)
  (h2 : a2 * b2 / 2 = 4)
  (h3 : a1 = a2)
  (h4 : c1 = 2 * c2) 
  (h5 : c1^2 = a1^2 + b1^2)
  (h6 : c2^2 = a2^2 + b2^2) :
  (b1 * b2)^2 = (2304 / 25 : ℝ) :=
by
  sorry

end triangle_length_product_square_l176_17665


namespace nested_fraction_evaluation_l176_17621

theorem nested_fraction_evaluation : 
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))))) = (21 / 55) :=
by
  sorry

end nested_fraction_evaluation_l176_17621


namespace Aryan_owes_1200_l176_17674

variables (A K : ℝ) -- A represents Aryan's debt, K represents Kyro's debt

-- Condition 1: Aryan's debt is twice Kyro's debt
axiom condition1 : A = 2 * K

-- Condition 2: Aryan pays 60% of her debt
axiom condition2 : (0.60 * A) + (0.80 * K) = 1500 - 300

theorem Aryan_owes_1200 : A = 1200 :=
by
  sorry

end Aryan_owes_1200_l176_17674


namespace two_painters_days_l176_17635

-- Define the conditions and the proof problem
def five_painters_days : ℕ := 5
def days_per_five_painters : ℕ := 2
def total_painter_days : ℕ := five_painters_days * days_per_five_painters -- Total painter-days for the original scenario
def two_painters : ℕ := 2
def last_day_painter_half_day : ℕ := 1 -- Indicating that one painter works half a day on the last day
def last_day_work : ℕ := two_painters - last_day_painter_half_day / 2 -- Total work on the last day is equivalent to 1.5 painter-days

theorem two_painters_days : total_painter_days = 5 :=
by
  sorry -- Mathematical proof goes here

end two_painters_days_l176_17635


namespace solution_set_of_bx2_minus_ax_minus_1_gt_0_l176_17659

theorem solution_set_of_bx2_minus_ax_minus_1_gt_0
  (a b : ℝ)
  (h1 : ∀ (x : ℝ), 2 < x ∧ x < 3 ↔ x^2 - a * x - b < 0) :
  ∀ (x : ℝ), -1 / 2 < x ∧ x < -1 / 3 ↔ b * x^2 - a * x - 1 > 0 :=
by
  sorry

end solution_set_of_bx2_minus_ax_minus_1_gt_0_l176_17659


namespace checkered_square_division_l176_17688

theorem checkered_square_division (m n k d m1 n1 : ℕ) (h1 : m^2 = n * k)
  (h2 : d = Nat.gcd m n) (hm : m = m1 * d) (hn : n = n1 * d)
  (h3 : Nat.gcd m1 n1 = 1) : 
  ∃ (part_size : ℕ), 
    part_size = n ∧ (∃ (pieces : ℕ), pieces = k) ∧ m^2 = pieces * part_size := 
sorry

end checkered_square_division_l176_17688


namespace area_ratio_of_region_A_and_C_l176_17648

theorem area_ratio_of_region_A_and_C
  (pA : ℕ) (pC : ℕ) 
  (hA : pA = 16)
  (hC : pC = 24) :
  let sA := pA / 4
  let sC := pC / 6
  let areaA := sA * sA
  let areaC := (3 * Real.sqrt 3 / 2) * sC * sC
  (areaA / areaC) = (2 * Real.sqrt 3 / 9) :=
by
  sorry

end area_ratio_of_region_A_and_C_l176_17648


namespace no_real_solution_l176_17610

theorem no_real_solution (x y : ℝ) : x^3 + y^2 = 2 → x^2 + x * y + y^2 - y = 0 → false := 
by 
  intro h1 h2
  sorry

end no_real_solution_l176_17610


namespace eagle_speed_l176_17673

theorem eagle_speed (E : ℕ) 
  (falcon_speed : ℕ := 46)
  (pelican_speed : ℕ := 33)
  (hummingbird_speed : ℕ := 30)
  (total_distance : ℕ := 248)
  (flight_time : ℕ := 2)
  (falcon_distance := falcon_speed * flight_time)
  (pelican_distance := pelican_speed * flight_time)
  (hummingbird_distance := hummingbird_speed * flight_time) :
  2 * E + falcon_distance + pelican_distance + hummingbird_distance = total_distance →
  E = 15 :=
by
  -- Proof will be provided here
  sorry

end eagle_speed_l176_17673


namespace three_a_greater_three_b_l176_17632

variable (a b : ℝ)

theorem three_a_greater_three_b (h : a > b) : 3 * a > 3 * b :=
  sorry

end three_a_greater_three_b_l176_17632


namespace suit_price_after_discount_l176_17685

-- Definitions based on given conditions 
def original_price : ℝ := 200
def price_increase : ℝ := 0.30 * original_price
def new_price : ℝ := original_price + price_increase
def discount : ℝ := 0.30 * new_price
def final_price : ℝ := new_price - discount

-- The theorem
theorem suit_price_after_discount :
  final_price = 182 :=
by
  -- Here we would provide the proof if required
  sorry

end suit_price_after_discount_l176_17685


namespace sandy_savings_l176_17681

-- Definition and conditions
def last_year_savings (S : ℝ) : ℝ := 0.06 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_savings (S : ℝ) : ℝ := 1.8333333333333333 * last_year_savings S

-- The percentage P of this year's salary that Sandy saved
def this_year_savings_perc (S : ℝ) (P : ℝ) : Prop :=
  P * this_year_salary S = this_year_savings S

-- The proof statement: Sandy saved 10% of her salary this year
theorem sandy_savings (S : ℝ) (P : ℝ) (h: this_year_savings_perc S P) : P = 0.10 :=
  sorry

end sandy_savings_l176_17681


namespace find_starting_number_l176_17649

theorem find_starting_number (x : ℝ) (h : ((x - 2 + 4) / 1) / 2 * 8 = 77) : x = 17.25 := by
  sorry

end find_starting_number_l176_17649


namespace problem_statement_l176_17613

theorem problem_statement 
  (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := 
    sorry

end problem_statement_l176_17613


namespace calculate_expression_l176_17684

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l176_17684


namespace xyz_sum_eq_40_l176_17650

theorem xyz_sum_eq_40
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + x * z + x^2 = 91) :
  x * y + y * z + z * x = 40 :=
sorry

end xyz_sum_eq_40_l176_17650


namespace jade_more_transactions_l176_17672

theorem jade_more_transactions 
    (mabel_transactions : ℕ) 
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + (mabel_transactions / 10))
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = 82) :
    jade_transactions - cal_transactions = 16 :=
sorry

end jade_more_transactions_l176_17672


namespace range_of_a_l176_17697

open Real

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ , x^2 + a * x + 1 < 0) ↔ (-2 : ℝ) ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l176_17697


namespace inequality_proof_l176_17656

theorem inequality_proof 
  (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
  sorry

end inequality_proof_l176_17656


namespace find_room_width_l176_17601

theorem find_room_width
  (length : ℝ)
  (cost_per_sqm : ℝ)
  (total_cost : ℝ)
  (h_length : length = 10)
  (h_cost_per_sqm : cost_per_sqm = 900)
  (h_total_cost : total_cost = 42750) :
  ∃ width : ℝ, width = 4.75 :=
by
  sorry

end find_room_width_l176_17601


namespace scientific_notation_of_274M_l176_17670

theorem scientific_notation_of_274M :
  274000000 = 2.74 * 10^8 := 
by 
  sorry

end scientific_notation_of_274M_l176_17670


namespace trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l176_17699

-- Part a
theorem trihedral_sum_of_angles_le_sum_of_plane_angles
  (α β γ : ℝ) (ASB BSC CSA : ℝ)
  (h1 : α ≤ ASB)
  (h2 : β ≤ BSC)
  (h3 : γ ≤ CSA) :
  α + β + γ ≤ ASB + BSC + CSA :=
sorry

-- Part b
theorem trihedral_sum_of_angles_ge_half_sum_of_plane_angles
  (α_S β_S γ_S : ℝ) (ASB BSC CSA : ℝ) 
  (h_acute : ASB < (π / 2) ∧ BSC < (π / 2) ∧ CSA < (π / 2))
  (h1 : α_S ≥ (1/2) * ASB)
  (h2 : β_S ≥ (1/2) * BSC)
  (h3 : γ_S ≥ (1/2) * CSA) :
  α_S + β_S + γ_S ≥ (1/2) * (ASB + BSC + CSA) :=
sorry

end trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l176_17699


namespace problem_part1_problem_part2_problem_part3_l176_17667

section
variables (a b : ℚ)

-- Define the operation
def otimes (a b : ℚ) : ℚ := a * b + abs a - b

-- Prove the three statements
theorem problem_part1 : otimes (-5) 4 = -19 :=
sorry

theorem problem_part2 : otimes (otimes 2 (-3)) 4 = -7 :=
sorry

theorem problem_part3 : otimes 3 (-2) > otimes (-2) 3 :=
sorry
end

end problem_part1_problem_part2_problem_part3_l176_17667


namespace wait_time_difference_l176_17689

noncomputable def kids_waiting_for_swings : ℕ := 3
noncomputable def kids_waiting_for_slide : ℕ := 2 * kids_waiting_for_swings
noncomputable def wait_per_kid_swings : ℕ := 2 * 60 -- 2 minutes in seconds
noncomputable def wait_per_kid_slide : ℕ := 15 -- in seconds

noncomputable def total_wait_swings : ℕ := kids_waiting_for_swings * wait_per_kid_swings
noncomputable def total_wait_slide : ℕ := kids_waiting_for_slide * wait_per_kid_slide

theorem wait_time_difference : total_wait_swings - total_wait_slide = 270 := by
  sorry

end wait_time_difference_l176_17689


namespace multiply_101_self_l176_17615

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end multiply_101_self_l176_17615


namespace volume_of_prism_l176_17660

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 10) (hwh : w * h = 15) (hlh : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 :=
by
  sorry

end volume_of_prism_l176_17660
