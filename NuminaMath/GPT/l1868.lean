import Mathlib

namespace NUMINAMATH_GPT_total_oranges_and_apples_l1868_186831

-- Given conditions as definitions
def bags_with_5_oranges_and_7_apples (m : ℕ) : ℕ × ℕ :=
  (5 * m + 1, 7 * m)

def bags_with_9_oranges_and_7_apples (n : ℕ) : ℕ × ℕ :=
  (9 * n, 7 * n + 21)

theorem total_oranges_and_apples (m n : ℕ) (k : ℕ) 
  (h1 : (5 * m + 1, 7 * m) = (9 * n, 7 * n + 21)) 
  (h2 : 4 * n ≡ 1 [MOD 5]) : 85 = 36 + 49 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_and_apples_l1868_186831


namespace NUMINAMATH_GPT_stratified_sampling_grade12_l1868_186838

theorem stratified_sampling_grade12 (total_students grade12_students sample_size : ℕ) 
  (h_total : total_students = 2000) 
  (h_grade12 : grade12_students = 700) 
  (h_sample : sample_size = 400) : 
  (sample_size * grade12_students) / total_students = 140 := 
by 
  sorry

end NUMINAMATH_GPT_stratified_sampling_grade12_l1868_186838


namespace NUMINAMATH_GPT_jelly_beans_problem_l1868_186816

/-- Mrs. Wonderful's jelly beans problem -/
theorem jelly_beans_problem : ∃ n_girls n_boys : ℕ, 
  (n_boys = n_girls + 2) ∧
  ((n_girls ^ 2) + ((n_girls + 2) ^ 2) = 394) ∧
  (n_girls + n_boys = 28) :=
by
  sorry

end NUMINAMATH_GPT_jelly_beans_problem_l1868_186816


namespace NUMINAMATH_GPT_triangle_inequality_l1868_186827

variable {α : Type*} [LinearOrderedField α]

/-- Given a triangle ABC with sides a, b, c, circumradius R, 
exradii r_a, r_b, r_c, and given 2R ≤ r_a, we need to show that a > b, a > c, 2R > r_b, and 2R > r_c. -/
theorem triangle_inequality (a b c R r_a r_b r_c : α) (h₁ : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1868_186827


namespace NUMINAMATH_GPT_min_ν_of_cubic_eq_has_3_positive_real_roots_l1868_186851

open Real

noncomputable def cubic_eq (x θ : ℝ) : ℝ :=
  x^3 * sin θ - (sin θ + 2) * x^2 + 6 * x - 4

noncomputable def ν (θ : ℝ) : ℝ :=
  (9 * sin θ ^ 2 - 4 * sin θ + 3) / 
  ((1 - cos θ) * (2 * cos θ - 6 * sin θ - 3 * sin (2 * θ) + 2))

theorem min_ν_of_cubic_eq_has_3_positive_real_roots :
  (∀ x:ℝ, cubic_eq x θ = 0 → 0 < x) →
  ν θ = 621 / 8 :=
sorry

end NUMINAMATH_GPT_min_ν_of_cubic_eq_has_3_positive_real_roots_l1868_186851


namespace NUMINAMATH_GPT_find_pairs_l1868_186845

theorem find_pairs (a b : ℕ) :
  (1111 * a) % (11 * b) = 11 * (a - b) →
  140 ≤ (1111 * a) / (11 * b) ∧ (1111 * a) / (11 * b) ≤ 160 →
  (a, b) = (3, 2) ∨ (a, b) = (6, 4) ∨ (a, b) = (7, 5) ∨ (a, b) = (9, 6) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1868_186845


namespace NUMINAMATH_GPT_intersection_point_of_lines_l1868_186895

theorem intersection_point_of_lines :
  (∃ x y : ℝ, y = x ∧ y = -x + 2 ∧ (x = 1 ∧ y = 1)) :=
sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l1868_186895


namespace NUMINAMATH_GPT_sum_of_areas_l1868_186899

theorem sum_of_areas :
  (∑' n : ℕ, Real.pi * (1 / 9 ^ n)) = (9 * Real.pi) / 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_l1868_186899


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1868_186800

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1868_186800


namespace NUMINAMATH_GPT_tan_neg_five_pi_div_four_l1868_186886

theorem tan_neg_five_pi_div_four : Real.tan (- (5 * Real.pi / 4)) = -1 := 
sorry

end NUMINAMATH_GPT_tan_neg_five_pi_div_four_l1868_186886


namespace NUMINAMATH_GPT_cos_of_angle_l1868_186860

theorem cos_of_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (3 * Real.pi / 2 + 2 * θ) = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_cos_of_angle_l1868_186860


namespace NUMINAMATH_GPT_unique_solution_l1868_186847

noncomputable def valid_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + (2 - a) * x + 1 = 0 ∧ -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

theorem unique_solution (a : ℝ) :
  (valid_solutions a) ↔ (a = 4.5 ∨ (a < 0) ∨ (a > 16 / 3)) := 
sorry

end NUMINAMATH_GPT_unique_solution_l1868_186847


namespace NUMINAMATH_GPT_right_triangle_perimeter_l1868_186868

theorem right_triangle_perimeter (n : ℕ) (hn : Nat.Prime n) (x y : ℕ) 
  (h1 : y^2 = x^2 + n^2) : n + x + y = n + n^2 := by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l1868_186868


namespace NUMINAMATH_GPT_max_abs_sum_on_ellipse_l1868_186896

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 9) = 1 → |x| + |y| ≤ 5 :=
by sorry

end NUMINAMATH_GPT_max_abs_sum_on_ellipse_l1868_186896


namespace NUMINAMATH_GPT_initial_percentage_increase_l1868_186879

theorem initial_percentage_increase (W R : ℝ) (P : ℝ) 
  (h1 : R = W * (1 + P / 100)) 
  (h2 : R * 0.75 = W * 1.3500000000000001) : P = 80 := 
by
  sorry

end NUMINAMATH_GPT_initial_percentage_increase_l1868_186879


namespace NUMINAMATH_GPT_cos_monotonic_increasing_interval_l1868_186869

open Real

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6}

theorem cos_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ,
    (∃ y, y = cos (π / 3 - 2 * x)) →
    (monotonic_increasing_interval k x) :=
by
  sorry

end NUMINAMATH_GPT_cos_monotonic_increasing_interval_l1868_186869


namespace NUMINAMATH_GPT_foil_covered_prism_width_l1868_186891

theorem foil_covered_prism_width
    (l w h : ℕ)
    (inner_volume : l * w * h = 128)
    (width_length_relation : w = 2 * l)
    (width_height_relation : w = 2 * h) :
    (w + 2) = 10 := 
sorry

end NUMINAMATH_GPT_foil_covered_prism_width_l1868_186891


namespace NUMINAMATH_GPT_parabola_focus_l1868_186892

theorem parabola_focus (x y p : ℝ) (h_eq : y = 2 * x^2) (h_standard_form : x^2 = (1 / 2) * y) (h_p : p = 1 / 4) : 
    (0, p / 2) = (0, 1 / 8) := by
    sorry

end NUMINAMATH_GPT_parabola_focus_l1868_186892


namespace NUMINAMATH_GPT_center_of_circle_l1868_186883

-- Definition of the main condition: the given circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 = 10 * x - 4 * y + 14

-- Statement to prove: that x + y = 3 when (x, y) is the center of the circle described by circle_equation
theorem center_of_circle {x y : ℝ} (h : circle_equation x y) : x + y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_center_of_circle_l1868_186883


namespace NUMINAMATH_GPT_sum_of_numbers_l1868_186804

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1868_186804


namespace NUMINAMATH_GPT_find_quadratic_expression_l1868_186863

-- Define the quadratic function
def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

-- Define conditions
def intersects_x_axis_at_A (a b c : ℝ) : Prop :=
  quadratic a b c (-2) = 0

def intersects_x_axis_at_B (a b c : ℝ) : Prop :=
  quadratic a b c (1) = 0

def has_maximum_value (a : ℝ) : Prop :=
  a < 0

-- Define the target function
def f_expr (x : ℝ) : ℝ := -x^2 - x + 2

-- The theorem to be proved
theorem find_quadratic_expression :
  ∃ a b c, 
    intersects_x_axis_at_A a b c ∧
    intersects_x_axis_at_B a b c ∧
    has_maximum_value a ∧
    ∀ x, quadratic a b c x = f_expr x :=
sorry

end NUMINAMATH_GPT_find_quadratic_expression_l1868_186863


namespace NUMINAMATH_GPT_marion_paperclips_correct_l1868_186850

def yun_initial_paperclips := 30
def yun_remaining_paperclips (x : ℕ) : ℕ := (2 * x) / 5
def marion_paperclips (x y : ℕ) : ℕ := (4 * (yun_remaining_paperclips x)) / 3 + y
def y := 7

theorem marion_paperclips_correct : marion_paperclips yun_initial_paperclips y = 23 := by
  sorry

end NUMINAMATH_GPT_marion_paperclips_correct_l1868_186850


namespace NUMINAMATH_GPT_proportion_difference_l1868_186846

theorem proportion_difference : (0.80 * 40) - ((4 / 5) * 20) = 16 := 
by 
  sorry

end NUMINAMATH_GPT_proportion_difference_l1868_186846


namespace NUMINAMATH_GPT_roots_square_sum_l1868_186832

theorem roots_square_sum {a b c : ℝ} (h1 : 3 * a^3 + 2 * a^2 - 3 * a - 8 = 0)
                                  (h2 : 3 * b^3 + 2 * b^2 - 3 * b - 8 = 0)
                                  (h3 : 3 * c^3 + 2 * c^2 - 3 * c - 8 = 0)
                                  (sum_roots : a + b + c = -2/3)
                                  (product_pairs : a * b + b * c + c * a = -1) : 
  a^2 + b^2 + c^2 = 22 / 9 := by
  sorry

end NUMINAMATH_GPT_roots_square_sum_l1868_186832


namespace NUMINAMATH_GPT_log_base_problem_l1868_186867

noncomputable def log_of_base (base value : ℝ) : ℝ := Real.log value / Real.log base

theorem log_base_problem (x : ℝ) (h : log_of_base 16 (x - 3) = 1 / 4) : 1 / log_of_base (x - 3) 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_log_base_problem_l1868_186867


namespace NUMINAMATH_GPT_probability_second_third_different_colors_l1868_186835

def probability_different_colors (blue_chips : ℕ) (red_chips : ℕ) (yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + red_chips + yellow_chips
  let prob_diff :=
    ((blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips)) +
    ((red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips)) +
    ((yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips))
  prob_diff

theorem probability_second_third_different_colors :
  probability_different_colors 7 6 5 = 107 / 162 :=
by
  sorry

end NUMINAMATH_GPT_probability_second_third_different_colors_l1868_186835


namespace NUMINAMATH_GPT_four_digit_number_sum_eq_4983_l1868_186841

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

theorem four_digit_number_sum_eq_4983 (n : ℕ) :
  n + reverse_number n = 4983 ↔ n = 1992 ∨ n = 2991 :=
by sorry

end NUMINAMATH_GPT_four_digit_number_sum_eq_4983_l1868_186841


namespace NUMINAMATH_GPT_transmission_time_is_128_l1868_186877

def total_time (blocks chunks_per_block rate : ℕ) : ℕ :=
  (blocks * chunks_per_block) / rate

theorem transmission_time_is_128 :
  total_time 80 256 160 = 128 :=
  by
  sorry

end NUMINAMATH_GPT_transmission_time_is_128_l1868_186877


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_3_0_l1868_186825

theorem line_intersects_x_axis_at_3_0 : ∃ (x : ℝ), ∃ (y : ℝ), 2 * y + 5 * x = 15 ∧ y = 0 ∧ (x, y) = (3, 0) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_3_0_l1868_186825


namespace NUMINAMATH_GPT_find_m_l1868_186801

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem find_m (a b m : ℝ) (h1 : f m a b = 0) (h2 : 3 * m^2 + 2 * a * m + b = 0)
  (h3 : f (m / 3) a b = 1 / 2) (h4 : m ≠ 0) : m = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_find_m_l1868_186801


namespace NUMINAMATH_GPT_double_chess_first_player_can_draw_l1868_186830

-- Define the basic structure and rules of double chess
structure Game :=
  (state : Type)
  (move : state → state)
  (turn : ℕ → state → state)

-- Define the concept of double move
def double_move (g : Game) (s : g.state) : g.state :=
  g.move (g.move s)

-- Define a condition stating that the first player can at least force a draw
theorem double_chess_first_player_can_draw
  (game : Game)
  (initial_state : game.state)
  (double_move_valid : ∀ s : game.state, ∃ s' : game.state, s' = double_move game s) :
  ∃ draw : game.state, ∀ second_player_strategy : game.state → game.state, 
    double_move game initial_state = draw :=
  sorry

end NUMINAMATH_GPT_double_chess_first_player_can_draw_l1868_186830


namespace NUMINAMATH_GPT_find_term_number_l1868_186890

variable {α : ℝ} (b : ℕ → ℝ) (q : ℝ)

namespace GeometricProgression

noncomputable def geometric_progression (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ (n : ℕ), b (n + 1) = b n * q

noncomputable def satisfies_conditions (α : ℝ) (b : ℕ → ℝ) : Prop :=
  b 25 = 2 * Real.tan α ∧ b 31 = 2 * Real.sin α

theorem find_term_number (α : ℝ) (b : ℕ → ℝ) (q : ℝ) (hb : geometric_progression b q) (hc : satisfies_conditions α b) :
  ∃ n, b n = Real.sin (2 * α) ∧ n = 37 :=
sorry

end GeometricProgression

end NUMINAMATH_GPT_find_term_number_l1868_186890


namespace NUMINAMATH_GPT_cubic_difference_l1868_186858

theorem cubic_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a^3 - b^3 = 108 :=
sorry

end NUMINAMATH_GPT_cubic_difference_l1868_186858


namespace NUMINAMATH_GPT_regular_octagon_interior_angle_l1868_186818

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end NUMINAMATH_GPT_regular_octagon_interior_angle_l1868_186818


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l1868_186864

theorem inequality_and_equality_condition (a b : ℝ) (h : a < b) :
  a^3 - 3 * a ≤ b^3 - 3 * b + 4 ∧ (a = -1 ∧ b = 1 → a^3 - 3 * a = b^3 - 3 * b + 4) :=
sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l1868_186864


namespace NUMINAMATH_GPT_geometric_sequence_x_l1868_186805

theorem geometric_sequence_x (x : ℝ) (h : 1 * 9 = x^2) : x = 3 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_x_l1868_186805


namespace NUMINAMATH_GPT_evaluate_sum_l1868_186878

theorem evaluate_sum (a b c : ℝ) 
  (h : (a / (36 - a) + b / (49 - b) + c / (81 - c) = 9)) :
  (6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5.047) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_sum_l1868_186878


namespace NUMINAMATH_GPT_average_sale_l1868_186842

-- Defining the monthly sales as constants
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month3 : ℝ := 6855
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 7391

-- The final theorem statement to prove the average sale
theorem average_sale : (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6900 := 
by 
  sorry

end NUMINAMATH_GPT_average_sale_l1868_186842


namespace NUMINAMATH_GPT_milk_needed_for_cookies_l1868_186882

-- Define the given conditions
def liters_to_cups (liters : ℕ) : ℕ := liters * 4

def milk_per_cookies (cups cookies : ℕ) : ℚ := cups / cookies

-- Define the problem statement
theorem milk_needed_for_cookies (h1 : milk_per_cookies 20 30 = milk_per_cookies x 12) : x = 8 :=
sorry

end NUMINAMATH_GPT_milk_needed_for_cookies_l1868_186882


namespace NUMINAMATH_GPT_product_of_digits_of_non_divisible_number_l1868_186803

theorem product_of_digits_of_non_divisible_number:
  (¬ (3641 % 4 = 0)) →
  ((3641 % 10) * ((3641 / 10) % 10)) = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_product_of_digits_of_non_divisible_number_l1868_186803


namespace NUMINAMATH_GPT_find_num_candies_bought_l1868_186898

-- Conditions
def cost_per_candy := 80
def sell_price_per_candy := 100
def num_sold := 48
def profit := 800

-- Question equivalence
theorem find_num_candies_bought (x : ℕ) 
  (hc : cost_per_candy = 80)
  (hs : sell_price_per_candy = 100)
  (hn : num_sold = 48)
  (hp : profit = 800) :
  48 * 100 - 80 * x = 800 → x = 50 :=
  by
  sorry

end NUMINAMATH_GPT_find_num_candies_bought_l1868_186898


namespace NUMINAMATH_GPT_number_of_pieces_l1868_186854

def length_piece : ℝ := 0.40
def total_length : ℝ := 47.5

theorem number_of_pieces : ⌊total_length / length_piece⌋ = 118 := by
  sorry

end NUMINAMATH_GPT_number_of_pieces_l1868_186854


namespace NUMINAMATH_GPT_circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l1868_186808

-- Define the variables
variables {a b c : ℝ} {x y z : ℝ}
variables {α β γ : ℝ}

-- Circumcircle equation
theorem circumcircle_trilinear_eq :
  a * y * z + b * x * z + c * x * y = 0 :=
sorry

-- Incircle equation
theorem incircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt x) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

-- Excircle equation
theorem excircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt (-x)) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

end NUMINAMATH_GPT_circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l1868_186808


namespace NUMINAMATH_GPT_eight_natural_numbers_exist_l1868_186809

theorem eight_natural_numbers_exist :
  ∃ (n : Fin 8 → ℕ), (∀ i j : Fin 8, i ≠ j → ¬(n i ∣ n j)) ∧ (∀ i j : Fin 8, i ≠ j → n i ∣ (n j * n j)) :=
by 
  sorry

end NUMINAMATH_GPT_eight_natural_numbers_exist_l1868_186809


namespace NUMINAMATH_GPT_triangle_inequality_l1868_186843

variables {a b c x y z : ℝ}

theorem triangle_inequality 
  (h1 : ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h2 : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1868_186843


namespace NUMINAMATH_GPT_fourth_term_of_geometric_progression_l1868_186802

theorem fourth_term_of_geometric_progression (x : ℝ) (r : ℝ) 
  (h1 : (2 * x + 5) = r * x) 
  (h2 : (3 * x + 10) = r * (2 * x + 5)) : 
  (3 * x + 10) * r = -5 :=
by
  sorry

end NUMINAMATH_GPT_fourth_term_of_geometric_progression_l1868_186802


namespace NUMINAMATH_GPT_distance_from_point_to_focus_l1868_186873

theorem distance_from_point_to_focus (x0 : ℝ) (h1 : (2 * Real.sqrt 3)^2 = 4 * x0) :
    x0 + 1 = 4 := by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_focus_l1868_186873


namespace NUMINAMATH_GPT_value_of_item_l1868_186852

theorem value_of_item (a b m p : ℕ) (h : a ≠ b) (eq_capitals : a * x + m = b * x + p) : 
  x = (p - m) / (a - b) :=
by
  sorry

end NUMINAMATH_GPT_value_of_item_l1868_186852


namespace NUMINAMATH_GPT_john_avg_speed_l1868_186885

/-- John's average speed problem -/
theorem john_avg_speed (d : ℕ) (total_time : ℕ) (time1 : ℕ) (speed1 : ℕ) 
  (time2 : ℕ) (speed2 : ℕ) (time3 : ℕ) (x : ℕ) :
  d = 144 ∧ total_time = 120 ∧ time1 = 40 ∧ speed1 = 64 
  ∧ time2 = 40 ∧ speed2 = 70 ∧ time3 = 40 
  → (d = time1 * speed1 + time2 * speed2 + time3 * x / 60)
  → x = 82 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_john_avg_speed_l1868_186885


namespace NUMINAMATH_GPT_find_q_of_quadratic_with_roots_ratio_l1868_186806

theorem find_q_of_quadratic_with_roots_ratio {q : ℝ} :
  (∃ r1 r2 : ℝ, r1 ≠ 0 ∧ r2 ≠ 0 ∧ r1 / r2 = 3 / 1 ∧ r1 + r2 = -10 ∧ r1 * r2 = q) →
  q = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_find_q_of_quadratic_with_roots_ratio_l1868_186806


namespace NUMINAMATH_GPT_mean_value_of_quadrilateral_interior_angles_l1868_186859

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end NUMINAMATH_GPT_mean_value_of_quadrilateral_interior_angles_l1868_186859


namespace NUMINAMATH_GPT_fraction_paint_remaining_l1868_186872

theorem fraction_paint_remaining :
  let original_paint := 1
  let first_day_usage := original_paint / 4
  let paint_remaining_after_first_day := original_paint - first_day_usage
  let second_day_usage := paint_remaining_after_first_day / 2
  let paint_remaining_after_second_day := paint_remaining_after_first_day - second_day_usage
  let third_day_usage := paint_remaining_after_second_day / 3
  let paint_remaining_after_third_day := paint_remaining_after_second_day - third_day_usage
  paint_remaining_after_third_day = original_paint / 4 := 
by
  sorry

end NUMINAMATH_GPT_fraction_paint_remaining_l1868_186872


namespace NUMINAMATH_GPT_scout_troop_profit_calc_l1868_186862

theorem scout_troop_profit_calc
  (candy_bars : ℕ := 1200)
  (purchase_rate : ℚ := 3/6)
  (sell_rate : ℚ := 2/3) :
  (candy_bars * sell_rate - candy_bars * purchase_rate) = 200 :=
by
  sorry

end NUMINAMATH_GPT_scout_troop_profit_calc_l1868_186862


namespace NUMINAMATH_GPT_part1_problem_part2_problem_l1868_186819

/-- Given initial conditions and price adjustment, prove the expected number of helmets sold and the monthly profit. -/
theorem part1_problem (initial_price : ℕ) (initial_sales : ℕ) 
(price_reduction : ℕ) (sales_per_reduction : ℕ) (cost_price : ℕ) : 
  initial_price = 80 → initial_sales = 200 → price_reduction = 10 → 
  sales_per_reduction = 20 → cost_price = 50 → 
  (initial_sales + price_reduction * sales_per_reduction = 400) ∧ 
  ((initial_price - price_reduction - cost_price) * 
  (initial_sales + price_reduction * sales_per_reduction) = 8000) :=
by
  intros
  sorry

/-- Given initial conditions and profit target, prove the expected selling price of helmets. -/
theorem part2_problem (initial_price : ℕ) (initial_sales : ℕ) 
(cost_price : ℕ) (profit_target : ℕ) (x : ℕ) :
  initial_price = 80 → initial_sales = 200 → cost_price = 50 → 
  profit_target = 7500 → (x = 15) → 
  (initial_price - x = 65) :=
by
  intros
  sorry

end NUMINAMATH_GPT_part1_problem_part2_problem_l1868_186819


namespace NUMINAMATH_GPT_simplify_expression_l1868_186871

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1868_186871


namespace NUMINAMATH_GPT_product_of_fractions_l1868_186884

-- Definitions from the conditions
def a : ℚ := 2 / 3 
def b : ℚ := 3 / 5
def c : ℚ := 4 / 7
def d : ℚ := 5 / 9

-- Statement of the proof problem
theorem product_of_fractions : a * b * c * d = 8 / 63 := 
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1868_186884


namespace NUMINAMATH_GPT_paint_houses_l1868_186865

theorem paint_houses (time_per_house : ℕ) (hour_to_minute : ℕ) (hours_available : ℕ) 
  (h1 : time_per_house = 20) (h2 : hour_to_minute = 60) (h3 : hours_available = 3) :
  (hours_available * hour_to_minute) / time_per_house = 9 :=
by
  sorry

end NUMINAMATH_GPT_paint_houses_l1868_186865


namespace NUMINAMATH_GPT_pascal_fifth_element_row_20_l1868_186897

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end NUMINAMATH_GPT_pascal_fifth_element_row_20_l1868_186897


namespace NUMINAMATH_GPT_find_x_l1868_186880

variable (m k x Km2 mk : ℚ)

def valid_conditions (m k : ℚ) : Prop :=
  m > 2 * k ∧ k > 0

def initial_acid (m : ℚ) : ℚ :=
  (m*m)/100

def diluted_acid (m k x : ℚ) : ℚ :=
  ((2*m) - k) * (m + x) / 100

theorem find_x (m k : ℚ) (h : valid_conditions m k):
  ∃ x : ℚ, (m^2 = diluted_acid m k x) ∧ x = (k * m - m^2) / (2 * m - k) :=
sorry

end NUMINAMATH_GPT_find_x_l1868_186880


namespace NUMINAMATH_GPT_matrix_power_50_l1868_186857

def P : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![-4, -3]
]

theorem matrix_power_50 :
  P ^ 50 = ![
    ![1, 0],
    ![0, 1]
  ] :=
sorry

end NUMINAMATH_GPT_matrix_power_50_l1868_186857


namespace NUMINAMATH_GPT_base3_to_base10_equiv_l1868_186824

theorem base3_to_base10_equiv : 
  let repr := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  repr = 142 :=
by
  sorry

end NUMINAMATH_GPT_base3_to_base10_equiv_l1868_186824


namespace NUMINAMATH_GPT_Petya_can_determine_weight_l1868_186881

theorem Petya_can_determine_weight (n : ℕ) (distinct_weights : Fin n → ℕ) 
  (device : (Fin 10 → Fin n) → ℕ) (ten_thousand_weights : n = 10000)
  (no_two_same : (∀ i j : Fin n, i ≠ j → distinct_weights i ≠ distinct_weights j)) :
  ∃ i : Fin n, ∃ w : ℕ, distinct_weights i = w :=
by
  sorry

end NUMINAMATH_GPT_Petya_can_determine_weight_l1868_186881


namespace NUMINAMATH_GPT_peaches_in_each_basket_l1868_186853

variable (R : ℕ)

theorem peaches_in_each_basket (h : 6 * R = 96) : R = 16 :=
by
  sorry

end NUMINAMATH_GPT_peaches_in_each_basket_l1868_186853


namespace NUMINAMATH_GPT_car_speed_proof_l1868_186815

noncomputable def car_speed_in_kmh (rpm : ℕ) (circumference : ℕ) : ℕ :=
  (rpm * circumference * 60) / 1000

theorem car_speed_proof : 
  car_speed_in_kmh 400 1 = 24 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_proof_l1868_186815


namespace NUMINAMATH_GPT_manager_salary_l1868_186813

theorem manager_salary :
  let avg_salary_employees := 1500
  let num_employees := 20
  let new_avg_salary := 2000
  (new_avg_salary * (num_employees + 1) - avg_salary_employees * num_employees = 12000) :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_l1868_186813


namespace NUMINAMATH_GPT_multiple_rate_is_correct_l1868_186836

-- Define Lloyd's standard working hours per day
def regular_hours_per_day : ℝ := 7.5

-- Define Lloyd's standard hourly rate
def regular_rate : ℝ := 3.5

-- Define the total hours worked on a specific day
def total_hours_worked : ℝ := 10.5

-- Define the total earnings for that specific day
def total_earnings : ℝ := 42.0

-- Define the function to calculate the multiple of the regular rate for excess hours
noncomputable def multiple_of_regular_rate (r_hours : ℝ) (r_rate : ℝ) (t_hours : ℝ) (t_earnings : ℝ) : ℝ :=
  let regular_earnings := r_hours * r_rate
  let excess_hours := t_hours - r_hours
  let excess_earnings := t_earnings - regular_earnings
  (excess_earnings / excess_hours) / r_rate

-- The statement to be proved
theorem multiple_rate_is_correct : 
  multiple_of_regular_rate regular_hours_per_day regular_rate total_hours_worked total_earnings = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_multiple_rate_is_correct_l1868_186836


namespace NUMINAMATH_GPT_max_red_dominated_rows_plus_blue_dominated_columns_l1868_186833

-- Definitions of the problem conditions and statement
theorem max_red_dominated_rows_plus_blue_dominated_columns (m n : ℕ)
  (h1 : Odd m) (h2 : Odd n) (h3 : 0 < m ∧ 0 < n) :
  ∃ A : Finset (Fin m) × Finset (Fin n),
  (A.1.card + A.2.card = m + n - 2) :=
sorry

end NUMINAMATH_GPT_max_red_dominated_rows_plus_blue_dominated_columns_l1868_186833


namespace NUMINAMATH_GPT_find_A_l1868_186840

theorem find_A :
  ∃ A B : ℕ, A < 10 ∧ B < 10 ∧ 5 * 100 + A * 10 + 8 - (B * 100 + 1 * 10 + 4) = 364 ∧ A = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1868_186840


namespace NUMINAMATH_GPT_monthly_growth_rate_l1868_186848

theorem monthly_growth_rate (x : ℝ)
  (turnover_may : ℝ := 1)
  (turnover_july : ℝ := 1.21)
  (growth_rate_condition : (1 + x) ^ 2 = 1.21) :
  x = 0.1 :=
sorry

end NUMINAMATH_GPT_monthly_growth_rate_l1868_186848


namespace NUMINAMATH_GPT_total_cost_of_horse_and_saddle_l1868_186888

noncomputable def saddle_cost : ℝ := 1000
noncomputable def horse_cost : ℝ := 4 * saddle_cost
noncomputable def total_cost : ℝ := saddle_cost + horse_cost

theorem total_cost_of_horse_and_saddle :
    total_cost = 5000 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_horse_and_saddle_l1868_186888


namespace NUMINAMATH_GPT_quadratic_root_condition_l1868_186826

theorem quadratic_root_condition (k : ℝ) :
  (∀ (x : ℝ), x^2 + k * x + 4 * k^2 - 3 = 0 → ∃ x1 x2 : ℝ, x1 + x2 = (-k) ∧ x1 * x2 = 4 * k^2 - 3 ∧ x1 + x2 = x1 * x2) →
  k = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_condition_l1868_186826


namespace NUMINAMATH_GPT_volume_of_prism_l1868_186887

variable (l w h : ℝ)

def area1 (l w : ℝ) : ℝ := l * w
def area2 (w h : ℝ) : ℝ := w * h
def area3 (l h : ℝ) : ℝ := l * h
def volume (l w h : ℝ) : ℝ := l * w * h

axiom cond1 : area1 l w = 15
axiom cond2 : area2 w h = 20
axiom cond3 : area3 l h = 30

theorem volume_of_prism : volume l w h = 30 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1868_186887


namespace NUMINAMATH_GPT_proof_problem_l1868_186849

-- Definitions of parallel and perpendicular relationships for lines and planes
def parallel (α β : Type) : Prop := sorry
def perpendicular (α β : Type) : Prop := sorry
def contained_in (m : Type) (α : Type) : Prop := sorry

-- Variables representing lines and planes
variables (l m n : Type) (α β : Type)

-- Assumptions from the conditions in step a)
variables 
  (h1 : m ≠ l)
  (h2 : α ≠ β)
  (h3 : parallel m n)
  (h4 : perpendicular m α)
  (h5 : perpendicular n β)

-- The goal is to prove that the planes α and β are parallel under the given conditions
theorem proof_problem : parallel α β :=
sorry

end NUMINAMATH_GPT_proof_problem_l1868_186849


namespace NUMINAMATH_GPT_length_of_A_l1868_186822

structure Point := (x : ℝ) (y : ℝ)

noncomputable def length (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem length_of_A'B' (A A' B B' C : Point) 
    (hA : A = ⟨0, 6⟩)
    (hB : B = ⟨0, 10⟩)
    (hC : C = ⟨3, 6⟩)
    (hA'_line : A'.y = A'.x)
    (hB'_line : B'.y = B'.x) 
    (hA'C : ∃ m b, ((C.y = m * C.x + b) ∧ (C.y = b) ∧ (A.y = b))) 
    (hB'C : ∃ m b, ((C.y = m * C.x + b) ∧ (B.y = m * B.x + b)))
    : length A' B' = (12 / 7) * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_A_l1868_186822


namespace NUMINAMATH_GPT_greatest_prime_factor_of_15_l1868_186861

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_15_l1868_186861


namespace NUMINAMATH_GPT_compare_abc_l1868_186814

noncomputable def a : ℝ := - Real.logb 2 (1/5)
noncomputable def b : ℝ := Real.logb 8 27
noncomputable def c : ℝ := Real.exp (-3)

theorem compare_abc : a = Real.logb 2 5 ∧ 1 < b ∧ b < 2 ∧ c = Real.exp (-3) → a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l1868_186814


namespace NUMINAMATH_GPT_chickens_increased_l1868_186828

-- Definitions and conditions
def initial_chickens := 45
def chickens_bought_day1 := 18
def chickens_bought_day2 := 12
def total_chickens_bought := chickens_bought_day1 + chickens_bought_day2

-- Proof statement
theorem chickens_increased :
  total_chickens_bought = 30 :=
by
  sorry

end NUMINAMATH_GPT_chickens_increased_l1868_186828


namespace NUMINAMATH_GPT_triangle_smallest_angle_l1868_186874

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end NUMINAMATH_GPT_triangle_smallest_angle_l1868_186874


namespace NUMINAMATH_GPT_fraction_identity_l1868_186829

open Real

theorem fraction_identity
  (p q r : ℝ)
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 :=
  sorry

end NUMINAMATH_GPT_fraction_identity_l1868_186829


namespace NUMINAMATH_GPT_find_pairs_l1868_186837

theorem find_pairs (a b q r : ℕ) (h1 : a * b = q * (a + b) + r)
  (h2 : q^2 + r = 2011) (h3 : 0 ≤ r ∧ r < a + b) : 
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ (a = t ∧ b = t + 2012 ∨ a = t + 2012 ∧ b = t)) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1868_186837


namespace NUMINAMATH_GPT_lisa_total_distance_l1868_186810

-- Definitions for distances and counts of trips
def plane_distance : ℝ := 256.0
def train_distance : ℝ := 120.5
def bus_distance : ℝ := 35.2

def plane_trips : ℕ := 32
def train_trips : ℕ := 16
def bus_trips : ℕ := 42

-- Definition of total distance traveled
def total_distance_traveled : ℝ :=
  (plane_distance * plane_trips)
  + (train_distance * train_trips)
  + (bus_distance * bus_trips)

-- The statement to be proven
theorem lisa_total_distance :
  total_distance_traveled = 11598.4 := by
  sorry

end NUMINAMATH_GPT_lisa_total_distance_l1868_186810


namespace NUMINAMATH_GPT_number_of_integer_solutions_l1868_186820

theorem number_of_integer_solutions (h : ∀ n : ℤ, (2020 - n) ^ 2 / (2020 - n ^ 2) ≥ 0) :
  ∃! (m : ℤ), m = 90 := 
sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l1868_186820


namespace NUMINAMATH_GPT_value_of_a_l1868_186821

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x + a^2 * y + 6 = 0 → (a-2) * x + 3 * a * y + 2 * a = 0) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1868_186821


namespace NUMINAMATH_GPT_willie_cream_from_farm_l1868_186889

variable (total_needed amount_to_buy amount_from_farm : ℕ)

theorem willie_cream_from_farm :
  total_needed = 300 → amount_to_buy = 151 → amount_from_farm = total_needed - amount_to_buy → amount_from_farm = 149 := by
  intros
  sorry

end NUMINAMATH_GPT_willie_cream_from_farm_l1868_186889


namespace NUMINAMATH_GPT_P_in_first_quadrant_l1868_186839

def point_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

theorem P_in_first_quadrant (k : ℝ) (h : k > 0) : point_in_first_quadrant (3, k) :=
by
  sorry

end NUMINAMATH_GPT_P_in_first_quadrant_l1868_186839


namespace NUMINAMATH_GPT_weight_of_B_l1868_186807

theorem weight_of_B (A B C : ℝ)
(h1 : (A + B + C) / 3 = 45)
(h2 : (A + B) / 2 = 40)
(h3 : (B + C) / 2 = 41)
(h4 : 2 * A = 3 * B ∧ 5 * C = 3 * B)
(h5 : A + B + C = 144) :
B = 43.2 :=
sorry

end NUMINAMATH_GPT_weight_of_B_l1868_186807


namespace NUMINAMATH_GPT_no_function_satisfies_condition_l1868_186817

theorem no_function_satisfies_condition :
  ¬ ∃ (f: ℕ → ℕ), ∀ (n: ℕ), f (f n) = n + 2017 :=
by
  -- Proof details are omitted
  sorry

end NUMINAMATH_GPT_no_function_satisfies_condition_l1868_186817


namespace NUMINAMATH_GPT_aiden_nap_is_15_minutes_l1868_186870

def aiden_nap_duration_in_minutes (nap_in_hours : ℚ) (minutes_per_hour : ℕ) : ℚ :=
  nap_in_hours * minutes_per_hour

theorem aiden_nap_is_15_minutes :
  aiden_nap_duration_in_minutes (1/4) 60 = 15 := by
  sorry

end NUMINAMATH_GPT_aiden_nap_is_15_minutes_l1868_186870


namespace NUMINAMATH_GPT_gcd_triangular_number_l1868_186834

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem gcd_triangular_number (n : ℕ) (h : n > 2) :
  ∃ k, n = 12 * k + 2 → gcd (6 * triangular_number n) (n - 2) = 12 :=
  sorry

end NUMINAMATH_GPT_gcd_triangular_number_l1868_186834


namespace NUMINAMATH_GPT_fruit_display_total_l1868_186893

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end NUMINAMATH_GPT_fruit_display_total_l1868_186893


namespace NUMINAMATH_GPT_compute_expression_l1868_186823

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1868_186823


namespace NUMINAMATH_GPT_probability_correct_l1868_186844

noncomputable def probability_one_white_one_black
    (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (draw_balls : ℕ) :=
if (total_balls = 4) ∧ (white_balls = 2) ∧ (black_balls = 2) ∧ (draw_balls = 2) then
  (2 * 2) / (Nat.choose total_balls draw_balls : ℚ)
else
  0

theorem probability_correct:
  probability_one_white_one_black 4 2 2 2 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_correct_l1868_186844


namespace NUMINAMATH_GPT_james_calories_ratio_l1868_186811

theorem james_calories_ratio:
  ∀ (dancing_sessions_per_day : ℕ) (hours_per_session : ℕ) 
  (days_per_week : ℕ) (calories_per_hour_walking : ℕ) 
  (total_calories_dancing_per_week : ℕ),
  dancing_sessions_per_day = 2 →
  hours_per_session = 1/2 →
  days_per_week = 4 →
  calories_per_hour_walking = 300 →
  total_calories_dancing_per_week = 2400 →
  300 * 2 = 600 →
  (total_calories_dancing_per_week / (dancing_sessions_per_day * hours_per_session * days_per_week)) / calories_per_hour_walking = 2 :=
by
  sorry

end NUMINAMATH_GPT_james_calories_ratio_l1868_186811


namespace NUMINAMATH_GPT_isosceles_trapezoid_height_l1868_186855

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_height_l1868_186855


namespace NUMINAMATH_GPT_smallest_perimeter_even_integer_triangl_l1868_186856

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end NUMINAMATH_GPT_smallest_perimeter_even_integer_triangl_l1868_186856


namespace NUMINAMATH_GPT_smallest_digit_not_in_units_place_of_odd_l1868_186866

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end NUMINAMATH_GPT_smallest_digit_not_in_units_place_of_odd_l1868_186866


namespace NUMINAMATH_GPT_average_percentage_decrease_l1868_186894

theorem average_percentage_decrease :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 100 * (1 - x)^2 = 81 ∧ x = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_decrease_l1868_186894


namespace NUMINAMATH_GPT_reading_hours_l1868_186876

theorem reading_hours (h : ℕ) (lizaRate suzieRate : ℕ) (lizaPages suziePages : ℕ) 
  (hliza : lizaRate = 20) (hsuzie : suzieRate = 15) 
  (hlizaPages : lizaPages = lizaRate * h) (hsuziePages : suziePages = suzieRate * h) 
  (h_diff : lizaPages = suziePages + 15) : h = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_reading_hours_l1868_186876


namespace NUMINAMATH_GPT_puppy_weight_l1868_186875

variable (a b c : ℝ)

theorem puppy_weight :
  (a + b + c = 30) →
  (a + c = 3 * b) →
  (a + b = c) →
  a = 7.5 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_puppy_weight_l1868_186875


namespace NUMINAMATH_GPT_maria_earnings_l1868_186812

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end NUMINAMATH_GPT_maria_earnings_l1868_186812
