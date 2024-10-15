import Mathlib

namespace NUMINAMATH_GPT_div_240_of_prime_diff_l2042_204271

-- Definitions
def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p = n ∧ Prime p
def prime_with_two_digits (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- The theorem statement
theorem div_240_of_prime_diff (a b : ℕ) (ha : prime_with_two_digits a) (hb : prime_with_two_digits b) (h : a > b) :
  240 ∣ (a^4 - b^4) ∧ ∀ d : ℕ, (d ∣ (a^4 - b^4) → (∀ m n : ℕ, prime_with_two_digits m → prime_with_two_digits n → m > n → d ∣ (m^4 - n^4) ) → d ≤ 240) :=
by
  sorry

end NUMINAMATH_GPT_div_240_of_prime_diff_l2042_204271


namespace NUMINAMATH_GPT_num_sequences_of_student_helpers_l2042_204233

-- Define the conditions
def num_students : ℕ := 15
def num_meetings : ℕ := 3

-- Define the statement to prove
theorem num_sequences_of_student_helpers : 
  (num_students ^ num_meetings) = 3375 :=
by sorry

end NUMINAMATH_GPT_num_sequences_of_student_helpers_l2042_204233


namespace NUMINAMATH_GPT_range_of_a_l2042_204221

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 3 → deriv (f a) x ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2042_204221


namespace NUMINAMATH_GPT_quadratic_real_roots_l2042_204297

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l2042_204297


namespace NUMINAMATH_GPT_jovana_added_shells_l2042_204209

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h_initial : initial_amount = 5) 
  (h_final : final_amount = 17) 
  (h_equation : final_amount = initial_amount + added_amount) : 
  added_amount = 12 := 
by 
  sorry

end NUMINAMATH_GPT_jovana_added_shells_l2042_204209


namespace NUMINAMATH_GPT_num_clients_visited_garage_l2042_204230

theorem num_clients_visited_garage :
  ∃ (num_clients : ℕ), num_clients = 24 ∧
    ∀ (num_cars selections_per_car selections_per_client : ℕ),
        num_cars = 16 → selections_per_car = 3 → selections_per_client = 2 →
        (num_cars * selections_per_car) / selections_per_client = num_clients :=
by
  sorry

end NUMINAMATH_GPT_num_clients_visited_garage_l2042_204230


namespace NUMINAMATH_GPT_range_of_f_l2042_204225

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - (Real.sin x) ^ 2 - 2 * (Real.sin x) * (Real.cos x)

theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 2 ∧ f x ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_f_l2042_204225


namespace NUMINAMATH_GPT_cube_face_expression_l2042_204240

theorem cube_face_expression (a b c : ℤ) (h1 : 3 * a + 2 = 17) (h2 : 7 * b - 4 = 10) (h3 : a + 3 * b - 2 * c = 11) : 
  a - b * c = 5 :=
by sorry

end NUMINAMATH_GPT_cube_face_expression_l2042_204240


namespace NUMINAMATH_GPT_temperature_on_Friday_l2042_204229

-- Definitions of the temperatures on the days
variables {M T W Th F : ℝ}

-- Conditions given in the problem
def avg_temp_mon_thu (M T W Th : ℝ) : Prop := (M + T + W + Th) / 4 = 48
def avg_temp_tue_fri (T W Th F : ℝ) : Prop := (T + W + Th + F) / 4 = 46
def temp_mon (M : ℝ) : Prop := M = 44

-- Statement to prove
theorem temperature_on_Friday (h1 : avg_temp_mon_thu M T W Th)
                               (h2 : avg_temp_tue_fri T W Th F)
                               (h3 : temp_mon M) : F = 36 :=
sorry

end NUMINAMATH_GPT_temperature_on_Friday_l2042_204229


namespace NUMINAMATH_GPT_two_categorical_variables_l2042_204226

-- Definitions based on the conditions
def smoking (x : String) : Prop := x = "Smoking" ∨ x = "Not smoking"
def sick (y : String) : Prop := y = "Sick" ∨ y = "Not sick"

def category1 (z : String) : Prop := z = "Whether smoking"
def category2 (w : String) : Prop := w = "Whether sick"

-- The main proof statement
theorem two_categorical_variables : 
  (category1 "Whether smoking" ∧ smoking "Smoking" ∧ smoking "Not smoking") ∧
  (category2 "Whether sick" ∧ sick "Sick" ∧ sick "Not sick") →
  "Whether smoking, Whether sick" = "Whether smoking, Whether sick" :=
by
  sorry

end NUMINAMATH_GPT_two_categorical_variables_l2042_204226


namespace NUMINAMATH_GPT_preservation_time_at_33_degrees_l2042_204279

noncomputable def preservation_time (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  Real.exp (k * x + b)

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : Real.exp b = 192)
  (h2 : Real.exp (22 * k + b) = 48) :
  preservation_time 33 k b = 24 := by
  sorry

end NUMINAMATH_GPT_preservation_time_at_33_degrees_l2042_204279


namespace NUMINAMATH_GPT_cost_of_paintbrush_l2042_204290

noncomputable def cost_of_paints : ℝ := 4.35
noncomputable def cost_of_easel : ℝ := 12.65
noncomputable def amount_already_has : ℝ := 6.50
noncomputable def additional_amount_needed : ℝ := 12.00

-- Let's define the total cost needed and the total costs of items
noncomputable def total_cost_of_paints_and_easel : ℝ := cost_of_paints + cost_of_easel
noncomputable def total_amount_needed : ℝ := amount_already_has + additional_amount_needed

-- And now we can state our theorem that needs to be proved.
theorem cost_of_paintbrush : total_amount_needed - total_cost_of_paints_and_easel = 1.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_paintbrush_l2042_204290


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l2042_204222

theorem solve_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l2042_204222


namespace NUMINAMATH_GPT_total_passengers_correct_l2042_204266

-- Definition of the conditions
def passengers_on_time : ℕ := 14507
def passengers_late : ℕ := 213
def total_passengers : ℕ := passengers_on_time + passengers_late

-- Theorem statement
theorem total_passengers_correct : total_passengers = 14720 := by
  sorry

end NUMINAMATH_GPT_total_passengers_correct_l2042_204266


namespace NUMINAMATH_GPT_parallel_lines_l2042_204292

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end NUMINAMATH_GPT_parallel_lines_l2042_204292


namespace NUMINAMATH_GPT_find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l2042_204270

-- Define the ellipse and parameters
variables (a b c : ℝ) (x y : ℝ)
-- Conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)

-- Given conditions
def eccentricity (c a : ℝ) : Prop := c = a * (Real.sqrt 3 / 2)
def rhombus_area (a b : ℝ) : Prop := (1/2) * (2 * a) * (2 * b) = 4
def relation_a_b_c (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- Questions transformed into proof problems
def ellipse_equation (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def range_OA_OB (OA OB : ℝ) : Prop := OA * OB ∈ Set.union (Set.Icc (-(3/2)) 0) (Set.Ioo 0 (3/2))
def quadrilateral_area : ℝ := 4

-- Prove the results given the conditions
theorem find_equation_of_ellipse (a b c : ℝ) (h_ellipse : ellipse a b) (h_ecc : eccentricity c a) (h_area : rhombus_area a b) (h_rel : relation_a_b_c a b c) :
  ellipse_equation x y := by
  sorry

theorem find_range_OA_OB (OA OB : ℝ) (kAC kBD : ℝ) (h_mult : kAC * kBD = -(1/4)) :
  range_OA_OB OA OB := by
  sorry

theorem find_area_quadrilateral : quadrilateral_area = 4 := by
  sorry

end NUMINAMATH_GPT_find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l2042_204270


namespace NUMINAMATH_GPT_find_r_l2042_204260

theorem find_r (r : ℝ) (h1 : ∃ s : ℝ, 8 * x^3 - 4 * x^2 - 42 * x + 45 = 8 * (x - r)^2 * (x - s)) :
  r = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l2042_204260


namespace NUMINAMATH_GPT_jenny_kenny_see_each_other_l2042_204212

-- Definitions of conditions
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def paths_distance : ℝ := 300
def radius_building : ℝ := 75
def start_distance : ℝ := 300

-- Theorem statement
theorem jenny_kenny_see_each_other : ∃ t : ℝ, (t = 120) :=
by
  sorry

end NUMINAMATH_GPT_jenny_kenny_see_each_other_l2042_204212


namespace NUMINAMATH_GPT_average_speed_distance_div_time_l2042_204289

theorem average_speed_distance_div_time (distance : ℕ) (time_minutes : ℕ) (average_speed : ℕ) : 
  distance = 8640 → time_minutes = 36 → average_speed = distance / (time_minutes * 60) → average_speed = 4 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  assumption

end NUMINAMATH_GPT_average_speed_distance_div_time_l2042_204289


namespace NUMINAMATH_GPT_wheat_pile_weight_l2042_204215

noncomputable def weight_of_conical_pile
  (circumference : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  let r := circumference / (2 * 3.14)
  let volume := (1.0 / 3.0) * 3.14 * r^2 * height
  volume * density

theorem wheat_pile_weight :
  weight_of_conical_pile 12.56 1.2 30 = 150.72 :=
by
  sorry

end NUMINAMATH_GPT_wheat_pile_weight_l2042_204215


namespace NUMINAMATH_GPT_find_surcharge_l2042_204281

-- The property tax in 1996 is increased by 6% over the 1995 tax.
def increased_tax (T_1995 : ℝ) : ℝ := T_1995 * 1.06

-- Petersons' property tax for the year 1995 is $1800.
def T_1995 : ℝ := 1800

-- The Petersons' 1996 tax totals $2108.
def T_1996 : ℝ := 2108

-- Additional surcharge for a special project.
def surcharge (T_1996 : ℝ) (increased_tax : ℝ) : ℝ := T_1996 - increased_tax

theorem find_surcharge : surcharge T_1996 (increased_tax T_1995) = 200 := by
  sorry

end NUMINAMATH_GPT_find_surcharge_l2042_204281


namespace NUMINAMATH_GPT_largest_square_tile_for_board_l2042_204298

theorem largest_square_tile_for_board (length width gcd_val : ℕ) (h1 : length = 16) (h2 : width = 24) 
  (h3 : gcd_val = Int.gcd length width) : gcd_val = 8 := by
  sorry

end NUMINAMATH_GPT_largest_square_tile_for_board_l2042_204298


namespace NUMINAMATH_GPT_range_of_a_l2042_204250

def P (a : ℝ) : Set ℝ := { x : ℝ | a - 4 < x ∧ x < a + 4 }
def Q : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 < 0 }

theorem range_of_a (a : ℝ) : (∀ x, Q x → P a x) → -1 < a ∧ a < 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2042_204250


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l2042_204231

theorem problem_1 : 12 - (-18) + (-7) - 15 = 8 := sorry

theorem problem_2 : -0.5 + (- (3 + 1/4)) + (-2.75) + (7 + 1/2) = 1 := sorry

theorem problem_3 : -2^2 + 3 * (-1)^(2023) - abs (-4) * 5 = -27 := sorry

theorem problem_4 : -3 - (-5 + (1 - 2 * (3 / 5)) / (-2)) = 19 / 10 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l2042_204231


namespace NUMINAMATH_GPT_thirds_side_length_valid_l2042_204206

theorem thirds_side_length_valid (x : ℝ) (h1 : x > 5) (h2 : x < 13) : x = 12 :=
sorry

end NUMINAMATH_GPT_thirds_side_length_valid_l2042_204206


namespace NUMINAMATH_GPT_unattainable_y_l2042_204265

theorem unattainable_y (x : ℚ) (hx : x ≠ -4 / 3) : 
    ∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3 :=
sorry

end NUMINAMATH_GPT_unattainable_y_l2042_204265


namespace NUMINAMATH_GPT_team_E_not_played_against_team_B_l2042_204272

-- Define the teams
inductive Team
| A | B | C | D | E | F
deriving DecidableEq

open Team

-- Define the matches played by each team
def matches_played : Team → Nat
| A => 5
| B => 4
| C => 3
| D => 2
| E => 1
| F => 0

-- Define the pairwise matches function
def paired : Team → Team → Prop
| A, B => true
| A, C => true
| A, D => true
| A, E => true
| A, F => true
| B, C => true
| B, D => true
| B, F  => true
| _, _ => false

-- Define the theorem based on the conditions and question
theorem team_E_not_played_against_team_B :
  ¬ paired E B :=
by
  sorry

end NUMINAMATH_GPT_team_E_not_played_against_team_B_l2042_204272


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l2042_204213

theorem parallel_vectors_x_value :
  ∀ (x : ℝ), (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (2, x) → a.1 * b.2 = a.2 * b.1) → x = -4 :=
by
  intros x h
  have h_parallel := h (1, -2) (2, x) rfl rfl
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l2042_204213


namespace NUMINAMATH_GPT_max_ratio_two_digit_mean_50_l2042_204286

theorem max_ratio_two_digit_mean_50 : 
  ∀ (x y : ℕ), (10 ≤ x ∧ x ≤ 99) ∧ (10 ≤ y ∧ y ≤ 99) ∧ (x + y = 100) → ( x / y ) ≤ 99 := 
by
  intros x y h
  obtain ⟨hx, hy, hsum⟩ := h
  sorry

end NUMINAMATH_GPT_max_ratio_two_digit_mean_50_l2042_204286


namespace NUMINAMATH_GPT_find_a_l2042_204228

theorem find_a (a x y : ℝ)
    (h1 : a * x - 5 * y = 5)
    (h2 : x / (x + y) = 5 / 7)
    (h3 : x - y = 3) :
    a = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l2042_204228


namespace NUMINAMATH_GPT_discount_calculation_l2042_204217

noncomputable def cost_price : ℝ := 180
noncomputable def markup_percentage : ℝ := 0.4778
noncomputable def profit_percentage : ℝ := 0.20

noncomputable def marked_price (CP : ℝ) (MP_percent : ℝ) : ℝ := CP + (MP_percent * CP)
noncomputable def selling_price (CP : ℝ) (PP_percent : ℝ) : ℝ := CP + (PP_percent * CP)
noncomputable def discount (MP : ℝ) (SP : ℝ) : ℝ := MP - SP

theorem discount_calculation :
  discount (marked_price cost_price markup_percentage) (selling_price cost_price profit_percentage) = 50.004 :=
by
  sorry

end NUMINAMATH_GPT_discount_calculation_l2042_204217


namespace NUMINAMATH_GPT_min_students_l2042_204234

variable (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ)

def find_min_students : Prop :=
  H = 2 * L ∧ 
  M = L + H ∧ 
  e = L + M + H ∧ 
  e = 6 * L ∧ 
  L ≥ 1

theorem min_students (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ) : find_min_students L H M e → e = 6 := 
by 
  intro h 
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end NUMINAMATH_GPT_min_students_l2042_204234


namespace NUMINAMATH_GPT_sara_spent_on_rented_movie_l2042_204254

def total_spent_on_movies : ℝ := 36.78
def spent_on_tickets : ℝ := 2 * 10.62
def spent_on_bought_movie : ℝ := 13.95

theorem sara_spent_on_rented_movie : 
  (total_spent_on_movies - spent_on_tickets - spent_on_bought_movie = 1.59) := 
by sorry

end NUMINAMATH_GPT_sara_spent_on_rented_movie_l2042_204254


namespace NUMINAMATH_GPT_sum_of_numbers_given_average_l2042_204200

variable (average : ℝ) (n : ℕ) (sum : ℝ)

theorem sum_of_numbers_given_average (h1 : average = 4.1) (h2 : n = 6) (h3 : average = sum / n) :
  sum = 24.6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_given_average_l2042_204200


namespace NUMINAMATH_GPT_meal_cost_one_burger_one_shake_one_cola_l2042_204205

-- Define the costs of individual items
variables (B S C : ℝ)

-- Conditions based on given equations
def eq1 : Prop := 3 * B + 7 * S + C = 120
def eq2 : Prop := 4 * B + 10 * S + C = 160.50

-- Goal: Prove that the total cost of one burger, one shake, and one cola is $39
theorem meal_cost_one_burger_one_shake_one_cola :
  eq1 B S C → eq2 B S C → B + S + C = 39 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_meal_cost_one_burger_one_shake_one_cola_l2042_204205


namespace NUMINAMATH_GPT_elvin_fixed_monthly_charge_l2042_204262

theorem elvin_fixed_monthly_charge
  (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) : 
  F = 24 := 
sorry

end NUMINAMATH_GPT_elvin_fixed_monthly_charge_l2042_204262


namespace NUMINAMATH_GPT_non_degenerate_triangles_l2042_204284

theorem non_degenerate_triangles :
  let total_points := 16
  let collinear_points := 5
  let total_triangles := Nat.choose total_points 3
  let degenerate_triangles := 2 * Nat.choose collinear_points 3
  let nondegenerate_triangles := total_triangles - degenerate_triangles
  nondegenerate_triangles = 540 := 
by
  sorry

end NUMINAMATH_GPT_non_degenerate_triangles_l2042_204284


namespace NUMINAMATH_GPT_sum_of_edges_of_rectangular_solid_l2042_204263

theorem sum_of_edges_of_rectangular_solid 
  (a r : ℝ) 
  (volume_eq : (a / r) * a * (a * r) = 343) 
  (surface_area_eq : 2 * ((a^2 / r) + (a^2 * r) + a^2) = 294) 
  (gp : a / r > 0 ∧ a > 0 ∧ a * r > 0) :
  4 * ((a / r) + a + (a * r)) = 84 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_edges_of_rectangular_solid_l2042_204263


namespace NUMINAMATH_GPT_percent_increase_fifth_triangle_l2042_204243

noncomputable def initial_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.2
noncomputable def num_triangles : ℕ := 5

noncomputable def side_length (n : ℕ) : ℝ :=
  initial_side_length * growth_factor ^ (n - 1)

noncomputable def perimeter_length (n : ℕ) : ℝ :=
  3 * side_length n

noncomputable def percent_increase (n : ℕ) : ℝ :=
  ((perimeter_length n / perimeter_length 1) - 1) * 100

theorem percent_increase_fifth_triangle :
  percent_increase 5 = 107.4 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_fifth_triangle_l2042_204243


namespace NUMINAMATH_GPT_expansion_no_x2_term_l2042_204255

theorem expansion_no_x2_term (n : ℕ) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  ¬ ∃ (r : ℕ), 0 ≤ r ∧ r ≤ n ∧ n - 4 * r = 2 → n = 7 := by
  sorry

end NUMINAMATH_GPT_expansion_no_x2_term_l2042_204255


namespace NUMINAMATH_GPT_probability_satisfies_inequality_l2042_204295

/-- Define the conditions for the points (x, y) -/
def within_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5

def satisfies_inequality (x y : ℝ) : Prop :=
  x + 2 * y ≤ 6

/-- Compute the probability that a randomly selected point within the rectangle
also satisfies the inequality -/
theorem probability_satisfies_inequality : (∃ p : ℚ, p = 3 / 10) :=
sorry

end NUMINAMATH_GPT_probability_satisfies_inequality_l2042_204295


namespace NUMINAMATH_GPT_tangent_line_at_one_min_value_f_l2042_204224

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * |Real.log x - 1|

theorem tangent_line_at_one (a : ℝ) (h1 : a = 1) : 
  ∃ (m b : ℝ), (∀ x : ℝ, f x a = m * x + b) ∧ m = 1 ∧ b = 1 ∧ (x - y + 1 = 0) := 
sorry

theorem min_value_f (a : ℝ) (h1 : 0 < a) : 
  (1 ≤ x ∧ x < e)  →  (x - f x a <= 0) ∨  (∀ (x : ℝ), 
  (f x a = if 0 < a ∧ a ≤ 2 then 1 + a 
          else if 2 < a ∧ a ≤ 2 * Real.exp (2) then 3 * (a / 2)^2 - (a / 2)^2 * Real.log (a / 2) else 
          Real.exp 2) 
   ) := 
sorry

end NUMINAMATH_GPT_tangent_line_at_one_min_value_f_l2042_204224


namespace NUMINAMATH_GPT_eval_x_sq_minus_y_sq_l2042_204256

theorem eval_x_sq_minus_y_sq (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + 3 * y = 29) : 
  x^2 - y^2 = -45 :=
sorry

end NUMINAMATH_GPT_eval_x_sq_minus_y_sq_l2042_204256


namespace NUMINAMATH_GPT_factorize_expr_l2042_204278

theorem factorize_expr (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l2042_204278


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l2042_204201

theorem sum_of_squares_of_consecutive_integers
  (a : ℤ) (h : (a - 1) * a * (a + 1) = 10 * ((a - 1) + a + (a + 1))) :
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l2042_204201


namespace NUMINAMATH_GPT_largest_among_four_l2042_204251

theorem largest_among_four (a b : ℝ) (h : 0 < a ∧ a < b ∧ a + b = 1) :
  a^2 + b^2 = max (max (max a (1/2)) (2*a*b)) (a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_largest_among_four_l2042_204251


namespace NUMINAMATH_GPT_subtraction_example_l2042_204218

theorem subtraction_example : 3.57 - 1.45 = 2.12 :=
by 
  sorry

end NUMINAMATH_GPT_subtraction_example_l2042_204218


namespace NUMINAMATH_GPT_infinite_divisible_269_l2042_204202

theorem infinite_divisible_269 (a : ℕ → ℤ) (h₀ : a 0 = 2) (h₁ : a 1 = 15) 
  (h_recur : ∀ n : ℕ, a (n + 2) = 15 * a (n + 1) + 16 * a n) :
  ∃ infinitely_many k: ℕ, 269 ∣ a k :=
by
  sorry

end NUMINAMATH_GPT_infinite_divisible_269_l2042_204202


namespace NUMINAMATH_GPT_curve_is_line_l2042_204296

theorem curve_is_line : ∀ (r θ : ℝ), r = 2 / (2 * Real.sin θ - Real.cos θ) → ∃ m b, ∀ (x y : ℝ), x = r * Real.cos θ → y = r * Real.sin θ → y = m * x + b :=
by
  intros r θ h
  sorry

end NUMINAMATH_GPT_curve_is_line_l2042_204296


namespace NUMINAMATH_GPT_larry_wins_probability_eq_l2042_204241

-- Define the conditions
def larry_probability_knocks_off : ℚ := 1 / 3
def julius_probability_knocks_off : ℚ := 1 / 4
def larry_throws_first : Prop := True
def independent_events : Prop := True

-- Define the proof that Larry wins the game with probability 2/3
theorem larry_wins_probability_eq :
  larry_throws_first ∧ independent_events →
  larry_probability_knocks_off = 1/3 ∧ julius_probability_knocks_off = 1/4 →
  ∃ p : ℚ, p = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_larry_wins_probability_eq_l2042_204241


namespace NUMINAMATH_GPT_replace_square_l2042_204214

theorem replace_square (x : ℝ) (h : 10.0003 * x = 10000.3) : x = 1000 :=
sorry

end NUMINAMATH_GPT_replace_square_l2042_204214


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l2042_204249

theorem ratio_of_a_to_b (a b : ℝ) (h1 : 0.5 / 100 * a = 85) (h2 : 0.75 / 100 * b = 150) : a / b = 17 / 20 :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_ratio_of_a_to_b_l2042_204249


namespace NUMINAMATH_GPT_product_lcm_gcd_eq_product_original_numbers_l2042_204280

theorem product_lcm_gcd_eq_product_original_numbers :
  let a := 12
  let b := 18
  (Int.gcd a b) * (Int.lcm a b) = a * b :=
by
  sorry

end NUMINAMATH_GPT_product_lcm_gcd_eq_product_original_numbers_l2042_204280


namespace NUMINAMATH_GPT_price_of_computer_and_desk_l2042_204235

theorem price_of_computer_and_desk (x y : ℕ) 
  (h1 : 10 * x + 200 * y = 90000)
  (h2 : 12 * x + 120 * y = 90000) : 
  x = 6000 ∧ y = 150 :=
by
  sorry

end NUMINAMATH_GPT_price_of_computer_and_desk_l2042_204235


namespace NUMINAMATH_GPT_unique_triple_solution_l2042_204232

theorem unique_triple_solution (x y z : ℝ) :
  x = y^3 + y - 8 ∧ y = z^3 + z - 8 ∧ z = x^3 + x - 8 → (x, y, z) = (2, 2, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_triple_solution_l2042_204232


namespace NUMINAMATH_GPT_fraction_nonneg_if_x_ge_m8_l2042_204203

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end NUMINAMATH_GPT_fraction_nonneg_if_x_ge_m8_l2042_204203


namespace NUMINAMATH_GPT_problem_I_problem_II_l2042_204204

-- Question I
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 :=
by
  sorry

-- Question II
theorem problem_II (a : ℝ) : (∀ x : ℝ, |x - a| + |2 * x - 1| ≥ 2) ↔ (a ≤ -3/2 ∨ a ≥ 5/2) :=
by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2042_204204


namespace NUMINAMATH_GPT_sum_of_data_l2042_204248

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) : a + b + c = 96 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_data_l2042_204248


namespace NUMINAMATH_GPT_problem_l2042_204252

theorem problem (x y z : ℝ) (h : (x - z) ^ 2 - 4 * (x - y) * (y - z) = 0) : z + x - 2 * y = 0 :=
sorry

end NUMINAMATH_GPT_problem_l2042_204252


namespace NUMINAMATH_GPT_cost_of_bananas_is_two_l2042_204242

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bananas_is_two_l2042_204242


namespace NUMINAMATH_GPT_cannot_form_polygon_l2042_204220

-- Define the stick lengths as a list
def stick_lengths : List ℕ := List.range 100 |>.map (λ n => 2^n)

-- Define the condition for forming a polygon
def can_form_polygon (lst : List ℕ) : Prop :=
  ∃ subset, subset ⊆ lst ∧ subset.length ≥ 3 ∧ (∀ s ∈ subset, s < (subset.sum - s))

-- The theorem to be proved
theorem cannot_form_polygon : ¬ can_form_polygon stick_lengths :=
by 
  sorry

end NUMINAMATH_GPT_cannot_form_polygon_l2042_204220


namespace NUMINAMATH_GPT_train_leave_tunnel_l2042_204273

noncomputable def train_leave_time 
  (train_speed : ℝ) 
  (tunnel_length : ℝ) 
  (train_length : ℝ) 
  (enter_time : ℝ × ℝ) : ℝ × ℝ :=
  let speed_km_min := train_speed / 60
  let total_distance := train_length + tunnel_length
  let time_to_pass := total_distance / speed_km_min
  let enter_minutes := enter_time.1 * 60 + enter_time.2
  let leave_minutes := enter_minutes + time_to_pass
  let leave_hours := leave_minutes / 60
  let leave_remainder_minutes := leave_minutes % 60
  (leave_hours, leave_remainder_minutes)

theorem train_leave_tunnel : 
  train_leave_time 80 70 1 (5, 12) = (6, 5.25) := 
sorry

end NUMINAMATH_GPT_train_leave_tunnel_l2042_204273


namespace NUMINAMATH_GPT_find_parabola_equation_l2042_204294

-- Define the problem conditions
def parabola_vertex_at_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 0

def axis_of_symmetry_x_or_y (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ y, f 0 = y)

def passes_through_point (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop :=
  f pt.1 = pt.2

-- Define the specific forms we expect the equations of the parabola to take
def equation1 (x y : ℝ) : Prop :=
  y^2 = - (9 / 2) * x

def equation2 (x y : ℝ) : Prop :=
  x^2 = (4 / 3) * y

-- state the main theorem
theorem find_parabola_equation :
  ∃ f : ℝ → ℝ, parabola_vertex_at_origin f ∧ axis_of_symmetry_x_or_y f ∧ passes_through_point f (-2, 3) ∧
  (equation1 (-2) (f (-2)) ∨ equation2 (-2) (f (-2))) :=
sorry

end NUMINAMATH_GPT_find_parabola_equation_l2042_204294


namespace NUMINAMATH_GPT_abc_proof_l2042_204267

noncomputable def abc_value (a b c : ℝ) : ℝ :=
  a * b * c

theorem abc_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b = 24 * (3 ^ (1 / 3)))
  (h5 : a * c = 40 * (3 ^ (1 / 3)))
  (h6 : b * c = 16 * (3 ^ (1 / 3))) : 
  abc_value a b c = 96 * (15 ^ (1 / 2)) :=
sorry

end NUMINAMATH_GPT_abc_proof_l2042_204267


namespace NUMINAMATH_GPT_find_k_b_l2042_204246

noncomputable def symmetric_line_circle_intersection : Prop :=
  ∃ (k b : ℝ), 
    (∀ (x y : ℝ),  (y = k * x) ∧ ((x-1)^2 + y^2 = 1)) ∧ 
    (∀ (x y : ℝ), (x - y + b = 0)) →
    (k = -1 ∧ b = -1)

theorem find_k_b :
  symmetric_line_circle_intersection :=
  by
    -- omitted proof
    sorry

end NUMINAMATH_GPT_find_k_b_l2042_204246


namespace NUMINAMATH_GPT_exists_sum_of_digits_div_11_l2042_204276

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_sum_of_digits_div_11_l2042_204276


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2042_204253

variable (x y : ℝ)
variable (condition_x : x = 1/3)
variable (condition_y : y = -6)

theorem simplify_and_evaluate :
  3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + (3/2) * x^2 * y)) + 2 * (3 * x * y^2 - x * y) = -4 :=
by
  rw [condition_x, condition_y]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2042_204253


namespace NUMINAMATH_GPT_max_value_of_expression_l2042_204238

noncomputable def max_value_expr (a b c : ℝ) : ℝ :=
  a + b^2 + c^3

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  max_value_expr a b c ≤ 8 :=
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l2042_204238


namespace NUMINAMATH_GPT_fraction_changed_value_l2042_204261

theorem fraction_changed_value:
  ∀ (num denom : ℝ), num / denom = 0.75 →
  (num + 0.15 * num) / (denom - 0.08 * denom) = 0.9375 :=
by
  intros num denom h_fraction
  sorry

end NUMINAMATH_GPT_fraction_changed_value_l2042_204261


namespace NUMINAMATH_GPT_inequality_abc_l2042_204239

variables {a b c : ℝ}

theorem inequality_abc 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2) ∧ 
    (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := 
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l2042_204239


namespace NUMINAMATH_GPT_evaporation_fraction_l2042_204299

theorem evaporation_fraction (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1)
  (h : (1 - x) * (3 / 4) = 1 / 6) : x = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_evaporation_fraction_l2042_204299


namespace NUMINAMATH_GPT_monotonic_decreasing_range_l2042_204227

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ -1 :=
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_range_l2042_204227


namespace NUMINAMATH_GPT_statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l2042_204288

-- Define the notion of line and plane
def Line := Type
def Plane := Type

-- Define the relations: parallel, contained-in, and intersection
def parallel (a b : Line) : Prop := sorry
def contained_in (a : Line) (α : Plane) : Prop := sorry
def intersects_at (a : Line) (α : Plane) (P : Type) : Prop := sorry

-- Conditions translated into Lean
def cond1 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ contained_in b α → parallel a b
def cond2 (a : Line) (α : Plane) (b : Line) {P : Type} : Prop := intersects_at a α P ∧ contained_in b α → ¬ parallel a b
def cond3 (a : Line) (α : Plane) : Prop := ¬ contained_in a α → parallel a α
def cond4 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ parallel b α → parallel a b

-- The statements that need to be proved incorrect
theorem statement_1_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond1 a α b) := sorry
theorem statement_3_incorrect (a : Line) (α : Plane) : ¬ (cond3 a α) := sorry
theorem statement_4_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond4 a α b) := sorry

end NUMINAMATH_GPT_statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l2042_204288


namespace NUMINAMATH_GPT_sum_of_squares_divisible_by_three_l2042_204210

theorem sum_of_squares_divisible_by_three {a b : ℤ} 
  (h : 3 ∣ (a^2 + b^2)) : (3 ∣ a ∧ 3 ∣ b) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_divisible_by_three_l2042_204210


namespace NUMINAMATH_GPT_simplify_expression_l2042_204257

theorem simplify_expression (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x^3 + y^3 = 3 * (x + y)) :
  (x / y) + (y / x) - (3 / (x * y)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2042_204257


namespace NUMINAMATH_GPT_incorrect_option_D_l2042_204282

-- definition of geometric objects and their properties
def octahedron_faces : Nat := 8
def tetrahedron_can_be_cut_into_4_pyramids : Prop := True
def frustum_extension_lines_intersect_at_a_point : Prop := True
def rectangle_rotated_around_side_forms_cylinder : Prop := True

-- incorrect identification of incorrect statement
theorem incorrect_option_D : 
  (∃ statement : String, statement = "D" ∧ ¬rectangle_rotated_around_side_forms_cylinder)  → False :=
by
  -- Proof of incorrect identification is not required per problem instructions
  sorry

end NUMINAMATH_GPT_incorrect_option_D_l2042_204282


namespace NUMINAMATH_GPT_sequence_result_l2042_204277

theorem sequence_result (initial_value : ℕ) (total_steps : ℕ) 
    (net_effect_one_cycle : ℕ) (steps_per_cycle : ℕ) : 
    initial_value = 100 ∧ total_steps = 26 ∧ 
    net_effect_one_cycle = (15 - 12 + 3) ∧ steps_per_cycle = 3 
    → 
    ∀ (resulting_value : ℕ), resulting_value = 151 :=
by
  sorry

end NUMINAMATH_GPT_sequence_result_l2042_204277


namespace NUMINAMATH_GPT_minimum_students_using_both_l2042_204207

theorem minimum_students_using_both (n L T x : ℕ) 
  (H1: 3 * n = 7 * L) 
  (H2: 5 * n = 6 * T) 
  (H3: n = 42) 
  (H4: n = L + T - x) : 
  x = 11 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_students_using_both_l2042_204207


namespace NUMINAMATH_GPT_solve_expression_l2042_204269

theorem solve_expression : 3 ^ (1 ^ (0 ^ 2)) - ((3 ^ 1) ^ 0) ^ 2 = 2 := by
  sorry

end NUMINAMATH_GPT_solve_expression_l2042_204269


namespace NUMINAMATH_GPT_bowling_ball_surface_area_l2042_204274

theorem bowling_ball_surface_area (d : ℝ) (hd : d = 9) : 
  4 * Real.pi * (d / 2)^2 = 81 * Real.pi :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_bowling_ball_surface_area_l2042_204274


namespace NUMINAMATH_GPT_largest_distance_l2042_204216

noncomputable def max_distance_between_spheres 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) : ℝ :=
dist c1 c2 + r1 + r2

theorem largest_distance 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) 
  (h₁ : c1 = (-3, -15, 10))
  (h₂ : r1 = 24)
  (h₃ : c2 = (20, 18, -30))
  (h₄ : r2 = 95) : 
  max_distance_between_spheres c1 r1 c2 r2 = Real.sqrt 3218 + 119 := 
by
  sorry

end NUMINAMATH_GPT_largest_distance_l2042_204216


namespace NUMINAMATH_GPT_star_sub_correctness_l2042_204259

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_sub_correctness : (star 6 2) - (star 2 6) = -12 := by
  sorry

end NUMINAMATH_GPT_star_sub_correctness_l2042_204259


namespace NUMINAMATH_GPT_simplify_expr_l2042_204287

variable (x : ℝ)

theorem simplify_expr : (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l2042_204287


namespace NUMINAMATH_GPT_part1_part2_l2042_204293

open Real

variables (x a : ℝ)

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0

theorem part1 (h : a = 1) (h_pq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

theorem part2 (hpq : ∀ (a x : ℝ), ¬ p x a → ¬ q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2042_204293


namespace NUMINAMATH_GPT_evaluate_expression_l2042_204223

theorem evaluate_expression (a b c : ℤ)
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3 : ℚ) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l2042_204223


namespace NUMINAMATH_GPT_original_amount_l2042_204237

theorem original_amount {P : ℕ} {R : ℕ} {T : ℕ} (h1 : P = 1000) (h2 : T = 5) 
  (h3 : ∃ R, (1000 * (R + 5) * 5) / 100 + 1000 = 1750) : 
  1000 + (1000 * R * 5 / 100) = 1500 :=
by
  sorry

end NUMINAMATH_GPT_original_amount_l2042_204237


namespace NUMINAMATH_GPT_product_of_integers_is_eight_l2042_204258

-- Define three different positive integers a, b, c such that they sum to 7
def sum_to_seven (a b c : ℕ) : Prop := a + b + c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Prove that the product of these integers is 8
theorem product_of_integers_is_eight (a b c : ℕ) (h : sum_to_seven a b c) : a * b * c = 8 := by sorry

end NUMINAMATH_GPT_product_of_integers_is_eight_l2042_204258


namespace NUMINAMATH_GPT_initial_distance_between_A_and_B_l2042_204264

theorem initial_distance_between_A_and_B
  (start_time : ℕ)        -- time in hours, 1 pm
  (meet_time : ℕ)         -- time in hours, 3 pm
  (speed_A : ℕ)           -- speed of A in km/hr
  (speed_B : ℕ)           -- speed of B in km/hr
  (time_walked : ℕ)       -- time walked in hours
  (distance_A : ℕ)        -- distance covered by A in km
  (distance_B : ℕ)        -- distance covered by B in km
  (initial_distance : ℕ)  -- initial distance between A and B

  (h1 : start_time = 1)
  (h2 : meet_time = 3)
  (h3 : speed_A = 5)
  (h4 : speed_B = 7)
  (h5 : time_walked = meet_time - start_time)
  (h6 : distance_A = speed_A * time_walked)
  (h7 : distance_B = speed_B * time_walked)
  (h8 : initial_distance = distance_A + distance_B) :

  initial_distance = 24 :=
by
  sorry

end NUMINAMATH_GPT_initial_distance_between_A_and_B_l2042_204264


namespace NUMINAMATH_GPT_samuel_has_five_birds_l2042_204211

theorem samuel_has_five_birds
  (birds_berries_per_day : ℕ)
  (total_berries_in_4_days : ℕ)
  (n_birds : ℕ)
  (h1 : birds_berries_per_day = 7)
  (h2 : total_berries_in_4_days = 140)
  (h3 : n_birds * birds_berries_per_day * 4 = total_berries_in_4_days) :
  n_birds = 5 := by
  sorry

end NUMINAMATH_GPT_samuel_has_five_birds_l2042_204211


namespace NUMINAMATH_GPT_min_valid_subset_card_eq_l2042_204208

open Finset

def pairs (n : ℕ) : Finset (ℕ × ℕ) := 
  (range n).product (range n)

def valid_subset (X : Finset (ℕ × ℕ)) (n : ℕ) : Prop :=
  ∀ (seq : ℕ → ℕ), ∃ k, (seq k, seq (k+1)) ∈ X

theorem min_valid_subset_card_eq (n : ℕ) (h : n = 10) : 
  ∃ X : Finset (ℕ × ℕ), valid_subset X n ∧ X.card = 55 := 
by 
  sorry

end NUMINAMATH_GPT_min_valid_subset_card_eq_l2042_204208


namespace NUMINAMATH_GPT_problem_solution_l2042_204245

theorem problem_solution
  (y1 y2 y3 y4 y5 y6 y7 : ℝ)
  (h1 : y1 + 3*y2 + 5*y3 + 7*y4 + 9*y5 + 11*y6 + 13*y7 = 0)
  (h2 : 3*y1 + 5*y2 + 7*y3 + 9*y4 + 11*y5 + 13*y6 + 15*y7 = 10)
  (h3 : 5*y1 + 7*y2 + 9*y3 + 11*y4 + 13*y5 + 15*y6 + 17*y7 = 104) :
  7*y1 + 9*y2 + 11*y3 + 13*y4 + 15*y5 + 17*y6 + 19*y7 = 282 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l2042_204245


namespace NUMINAMATH_GPT_parallel_lines_coefficient_l2042_204291

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 2 * y + 2 = 0) → (3 * x - y - 2 = 0)) → a = -6 :=
  by
    sorry

end NUMINAMATH_GPT_parallel_lines_coefficient_l2042_204291


namespace NUMINAMATH_GPT_machines_needed_l2042_204283

variables (R x m N : ℕ) (h1 : 4 * R * 6 = x)
           (h2 : N * R * 6 = m * x)

theorem machines_needed : N = m * 4 :=
by sorry

end NUMINAMATH_GPT_machines_needed_l2042_204283


namespace NUMINAMATH_GPT_problems_per_worksheet_l2042_204285

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ)
    (h1 : total_worksheets = 16) (h2 : graded_worksheets = 8) (h3 : remaining_problems = 32) :
    remaining_problems / (total_worksheets - graded_worksheets) = 4 :=
by
  sorry

end NUMINAMATH_GPT_problems_per_worksheet_l2042_204285


namespace NUMINAMATH_GPT_problem_sin_cos_k_l2042_204275

open Real

theorem problem_sin_cos_k {k : ℝ} :
  (∃ x : ℝ, sin x ^ 2 + cos x + k = 0) ↔ -2 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_GPT_problem_sin_cos_k_l2042_204275


namespace NUMINAMATH_GPT_distribute_papers_l2042_204219

theorem distribute_papers (n m : ℕ) (h_n : n = 5) (h_m : m = 10) : 
  (m ^ n) = 100000 :=
by 
  rw [h_n, h_m]
  rfl

end NUMINAMATH_GPT_distribute_papers_l2042_204219


namespace NUMINAMATH_GPT_find_other_tax_l2042_204268

/-- Jill's expenditure breakdown and total tax conditions. -/
def JillExpenditure 
  (total : ℝ)
  (clothingPercent : ℝ)
  (foodPercent : ℝ)
  (otherPercent : ℝ)
  (clothingTaxPercent : ℝ)
  (foodTaxPercent : ℝ)
  (otherTaxPercent : ℝ)
  (totalTaxPercent : ℝ) :=
  (clothingPercent + foodPercent + otherPercent = 100) ∧
  (clothingTaxPercent = 4) ∧
  (foodTaxPercent = 0) ∧
  (totalTaxPercent = 5.2) ∧
  (total > 0)

/-- The goal is to find the tax percentage on other items which Jill paid, given the constraints. -/
theorem find_other_tax
  {total clothingAmt foodAmt otherAmt clothingTax foodTax otherTaxPercent totalTax : ℝ}
  (h_exp : JillExpenditure total 50 10 40 clothingTax foodTax otherTaxPercent totalTax) :
  otherTaxPercent = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_other_tax_l2042_204268


namespace NUMINAMATH_GPT_Jenny_older_than_Rommel_l2042_204236

theorem Jenny_older_than_Rommel :
  ∃ t r j, t = 5 ∧ r = 3 * t ∧ j = t + 12 ∧ (j - r = 2) := 
by
  -- We insert the proof here using sorry to skip the actual proof part.
  sorry

end NUMINAMATH_GPT_Jenny_older_than_Rommel_l2042_204236


namespace NUMINAMATH_GPT_ones_digit_11_pow_l2042_204247

theorem ones_digit_11_pow (n : ℕ) (hn : n > 0) : (11^n % 10) = 1 := by
  sorry

end NUMINAMATH_GPT_ones_digit_11_pow_l2042_204247


namespace NUMINAMATH_GPT_ratio_of_cube_dimensions_l2042_204244

theorem ratio_of_cube_dimensions (V_original V_larger : ℝ) (hV_org : V_original = 64) (hV_lrg : V_larger = 512) :
  (∃ r : ℝ, r^3 = V_larger / V_original) ∧ r = 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_cube_dimensions_l2042_204244
