import Mathlib

namespace NUMINAMATH_GPT_rhombus_diagonal_l1089_108980

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h : d1 * d2 = 2 * area) (hd2 : d2 = 21) (h_area : area = 157.5) : d1 = 15 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1089_108980


namespace NUMINAMATH_GPT_question1_question2_l1089_108903

theorem question1 (x : ℝ) : (1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x) ↔ (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4) :=
by sorry

theorem question2 (x a : ℝ) : ((x - a)/(x - a^2) < 0)
  ↔ (a = 0 ∨ a = 1 → false)
  ∨ (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a)
  ∨ ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) :=
by sorry

end NUMINAMATH_GPT_question1_question2_l1089_108903


namespace NUMINAMATH_GPT_t_shirt_price_increase_t_shirt_max_profit_l1089_108977

theorem t_shirt_price_increase (x : ℝ) : (x + 10) * (300 - 10 * x) = 3360 → x = 2 := 
by 
  sorry

theorem t_shirt_max_profit (x : ℝ) : (-10 * x^2 + 200 * x + 3000) = 4000 ↔ x = 10 := 
by 
  sorry

end NUMINAMATH_GPT_t_shirt_price_increase_t_shirt_max_profit_l1089_108977


namespace NUMINAMATH_GPT_small_rectangular_prisms_intersect_diagonal_l1089_108996

def lcm (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

def inclusion_exclusion (n : Nat) : Nat :=
  n / 2 + n / 3 + n / 5 - n / (2 * 3) - n / (3 * 5) - n / (5 * 2) + n / (2 * 3 * 5)

theorem small_rectangular_prisms_intersect_diagonal :
  ∀ (a b c : Nat) (L : Nat), a = 2 → b = 3 → c = 5 → L = 90 →
  lcm a b c = 30 → 3 * inclusion_exclusion (lcm a b c) = 66 :=
by
  intros
  sorry

end NUMINAMATH_GPT_small_rectangular_prisms_intersect_diagonal_l1089_108996


namespace NUMINAMATH_GPT_ball_total_distance_l1089_108971

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (bounces : ℕ) : ℝ :=
  let rec loop (height : ℝ) (total : ℝ) (remaining : ℕ) : ℝ :=
    if remaining = 0 then total
    else loop (height * bounce_factor) (total + height + height * bounce_factor) (remaining - 1)
  loop initial_height 0 bounces

theorem ball_total_distance : 
  total_distance 20 0.8 4 = 106.272 :=
by
  sorry

end NUMINAMATH_GPT_ball_total_distance_l1089_108971


namespace NUMINAMATH_GPT_bus_stops_time_per_hour_l1089_108908

theorem bus_stops_time_per_hour 
  (avg_speed_without_stoppages : ℝ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 75) 
  (h2 : avg_speed_with_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 28 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_time_per_hour_l1089_108908


namespace NUMINAMATH_GPT_quadratic_function_inequality_l1089_108944

variable (a x x₁ x₂ : ℝ)

def f (x : ℝ) := a * x^2 + 2 * a * x + 4

theorem quadratic_function_inequality
  (h₀ : 0 < a) (h₁ : a < 3)
  (h₂ : x₁ + x₂ = 0)
  (h₃ : x₁ < x₂) :
  f a x₁ < f a x₂ := 
sorry

end NUMINAMATH_GPT_quadratic_function_inequality_l1089_108944


namespace NUMINAMATH_GPT_price_equivalence_l1089_108961

theorem price_equivalence : 
  (∀ a o p : ℕ, 10 * a = 5 * o ∧ 4 * o = 6 * p) → 
  (∀ a o p : ℕ, 20 * a = 15 * p) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_price_equivalence_l1089_108961


namespace NUMINAMATH_GPT_more_sparrows_than_pigeons_l1089_108991

-- Defining initial conditions
def initial_sparrows := 3
def initial_starlings := 5
def initial_pigeons := 2
def additional_sparrows := 4
def additional_starlings := 2
def additional_pigeons := 3

-- Final counts after additional birds join
def final_sparrows := initial_sparrows + additional_sparrows
def final_pigeons := initial_pigeons + additional_pigeons

-- The statement to be proved
theorem more_sparrows_than_pigeons:
  final_sparrows - final_pigeons = 2 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_more_sparrows_than_pigeons_l1089_108991


namespace NUMINAMATH_GPT_count_odd_numbers_300_600_l1089_108953

theorem count_odd_numbers_300_600 : ∃ n : ℕ, n = 149 ∧ ∀ k : ℕ, (301 ≤ k ∧ k < 600 ∧ k % 2 = 1) ↔ (301 ≤ k ∧ k < 600 ∧ k % 2 = 1 ∧ k - 301 < n * 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_odd_numbers_300_600_l1089_108953


namespace NUMINAMATH_GPT_find_f_of_conditions_l1089_108900

theorem find_f_of_conditions (f : ℝ → ℝ) :
  (f 1 = 1) →
  (∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) →
  (∀ x : ℝ, f x = 3^x - 2^x) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_f_of_conditions_l1089_108900


namespace NUMINAMATH_GPT_sum_of_two_digit_divisors_l1089_108990

theorem sum_of_two_digit_divisors (d : ℕ) (h1 : 145 % d = 4) (h2 : 10 ≤ d ∧ d < 100) :
  d = 47 :=
by
  have hd : d ∣ 141 := sorry
  exact sorry

end NUMINAMATH_GPT_sum_of_two_digit_divisors_l1089_108990


namespace NUMINAMATH_GPT_green_peaches_count_l1089_108939

def red_peaches : ℕ := 17
def green_peaches (x : ℕ) : Prop := red_peaches = x + 1

theorem green_peaches_count (x : ℕ) (h : green_peaches x) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_green_peaches_count_l1089_108939


namespace NUMINAMATH_GPT_directrix_eqn_of_parabola_l1089_108974

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_directrix_eqn_of_parabola_l1089_108974


namespace NUMINAMATH_GPT_sheets_paper_150_l1089_108992

def num_sheets_of_paper (S : ℕ) (E : ℕ) : Prop :=
  (S - E = 50) ∧ (3 * E - S = 150)

theorem sheets_paper_150 (S E : ℕ) : num_sheets_of_paper S E → S = 150 :=
by
  sorry

end NUMINAMATH_GPT_sheets_paper_150_l1089_108992


namespace NUMINAMATH_GPT_age_difference_l1089_108998

theorem age_difference :
  let x := 5
  let prod_today := x * x
  let prod_future := (x + 1) * (x + 1)
  prod_future - prod_today = 11 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1089_108998


namespace NUMINAMATH_GPT_period_is_seven_l1089_108952

-- Define the conditions
def apples_per_sandwich (a : ℕ) := a = 4
def sandwiches_per_day (s : ℕ) := s = 10
def total_apples (t : ℕ) := t = 280

-- Define the question to prove the period
theorem period_is_seven (a s t d : ℕ) 
  (h1 : apples_per_sandwich a)
  (h2 : sandwiches_per_day s)
  (h3 : total_apples t)
  (h4 : d = t / (a * s)) 
  : d = 7 := 
sorry

end NUMINAMATH_GPT_period_is_seven_l1089_108952


namespace NUMINAMATH_GPT_sum_set_15_l1089_108946

noncomputable def sum_nth_set (n : ℕ) : ℕ :=
  let first_element := 1 + (n - 1) * n / 2
  let last_element := first_element + n - 1
  n * (first_element + last_element) / 2

theorem sum_set_15 : sum_nth_set 15 = 1695 :=
  by sorry

end NUMINAMATH_GPT_sum_set_15_l1089_108946


namespace NUMINAMATH_GPT_kathleen_money_left_l1089_108909

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end NUMINAMATH_GPT_kathleen_money_left_l1089_108909


namespace NUMINAMATH_GPT_equation_solutions_l1089_108964

theorem equation_solutions :
  ∀ x y : ℤ, x^2 + x * y + y^2 + x + y - 5 = 0 → (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -3) ∨ (x = -3 ∧ y = 1) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_equation_solutions_l1089_108964


namespace NUMINAMATH_GPT_age_difference_l1089_108943

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 14) : C = A - 14 :=
by sorry

end NUMINAMATH_GPT_age_difference_l1089_108943


namespace NUMINAMATH_GPT_games_attended_this_month_l1089_108920

theorem games_attended_this_month 
  (games_last_month games_next_month total_games games_this_month : ℕ)
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44)
  (h4 : games_last_month + games_this_month + games_next_month = total_games) : 
  games_this_month = 11 := by
  sorry

end NUMINAMATH_GPT_games_attended_this_month_l1089_108920


namespace NUMINAMATH_GPT_inequality_holds_l1089_108907

theorem inequality_holds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1089_108907


namespace NUMINAMATH_GPT_negation_of_zero_product_l1089_108989

theorem negation_of_zero_product (x y : ℝ) : (xy ≠ 0) → (x ≠ 0) ∧ (y ≠ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_zero_product_l1089_108989


namespace NUMINAMATH_GPT_jenny_kenny_reunion_time_l1089_108970

/-- Define initial conditions given in the problem --/
def jenny_initial_pos : ℝ × ℝ := (-60, 100)
def kenny_initial_pos : ℝ × ℝ := (-60, -100)
def building_radius : ℝ := 60
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def distance_apa : ℝ := 200
def initial_distance : ℝ := 200

theorem jenny_kenny_reunion_time : ∃ t : ℚ, 
  (t = (10 * (Real.sqrt 35)) / 7) ∧ 
  (17 = (10 + 7)) :=
by
  -- conditions to be used
  let jenny_pos (t : ℝ) := (-60 + 2 * t, 100)
  let kenny_pos (t : ℝ) := (-60 + 4 * t, -100)
  let circle_eq (x y : ℝ) := (x^2 + y^2 = building_radius^2)
  
  sorry

end NUMINAMATH_GPT_jenny_kenny_reunion_time_l1089_108970


namespace NUMINAMATH_GPT_triangle_area_l1089_108921

-- Define the conditions of the problem
variables (a b c : ℝ) (C : ℝ)
axiom cond1 : c^2 = a^2 + b^2 - 2 * a * b + 6
axiom cond2 : C = Real.pi / 3

-- Define the goal
theorem triangle_area : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1089_108921


namespace NUMINAMATH_GPT_vector_subtraction_result_l1089_108933

-- Defining the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- The main theorem stating that a - 2b results in the expected coordinates
theorem vector_subtraction_result :
  a - 2 • b = (7, -2) := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_result_l1089_108933


namespace NUMINAMATH_GPT_num_spacy_subsets_15_l1089_108949

def spacy_subsets (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | 3     => 4
  | n + 1 => spacy_subsets n + if n ≥ 2 then spacy_subsets (n - 2) else 1

theorem num_spacy_subsets_15 : spacy_subsets 15 = 406 := by
  sorry

end NUMINAMATH_GPT_num_spacy_subsets_15_l1089_108949


namespace NUMINAMATH_GPT_ella_days_11_years_old_l1089_108906

theorem ella_days_11_years_old (x y z : ℕ) (h1 : 40 * x + 44 * y + 48 * (180 - x - y) = 7920) (h2 : x + y + z = 180) (h3 : 2 * x + y = 180) : y = 60 :=
by {
  -- proof can be derived from the given conditions
  sorry
}

end NUMINAMATH_GPT_ella_days_11_years_old_l1089_108906


namespace NUMINAMATH_GPT_evaluate_expression_l1089_108960

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1089_108960


namespace NUMINAMATH_GPT_problem_statement_l1089_108927

theorem problem_statement (x : ℝ) (h1 : x = 3 ∨ x = -3) : 6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1089_108927


namespace NUMINAMATH_GPT_system_solution_l1089_108905

theorem system_solution:
  let k := 115 / 12 
  ∃ x y z: ℝ, 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    (x + k * y + 5 * z = 0) ∧
    (4 * x + k * y - 3 * z = 0) ∧
    (3 * x + 5 * y - 4 * z = 0) ∧ 
    ((1 : ℝ) / 15 = (x * z) / (y * y)) := 
by sorry

end NUMINAMATH_GPT_system_solution_l1089_108905


namespace NUMINAMATH_GPT_xyz_cubic_expression_l1089_108959

theorem xyz_cubic_expression (x y z a b c : ℝ) (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0) (h7 : a ≠ 0) (h8 : b ≠ 0) (h9 : c ≠ 0) :
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_xyz_cubic_expression_l1089_108959


namespace NUMINAMATH_GPT_percentage_of_one_pair_repeated_digits_l1089_108968

theorem percentage_of_one_pair_repeated_digits (n : ℕ) (h1 : 10000 ≤ n) (h2 : n ≤ 99999) :
  ∃ (percentage : ℝ), percentage = 56.0 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_one_pair_repeated_digits_l1089_108968


namespace NUMINAMATH_GPT_correct_formula_l1089_108976

-- Given conditions
def table : List (ℕ × ℕ) := [(2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Candidate formulas
def formulaA (x : ℕ) : ℕ := 2 * x - 4
def formulaB (x : ℕ) : ℕ := x^2 - 3 * x + 2
def formulaC (x : ℕ) : ℕ := x^3 - 3 * x^2 + 2 * x
def formulaD (x : ℕ) : ℕ := x^2 - 4 * x
def formulaE (x : ℕ) : ℕ := x^2 - 4

-- The statement to be proven
theorem correct_formula : ∀ (x y : ℕ), (x, y) ∈ table → y = formulaB x :=
by
  sorry

end NUMINAMATH_GPT_correct_formula_l1089_108976


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l1089_108962

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_circles_l1089_108962


namespace NUMINAMATH_GPT_symmetric_line_origin_l1089_108955

theorem symmetric_line_origin (a b : ℝ) :
  (∀ (m n : ℝ), a * m + 3 * n = 9 → -m + 3 * -n + b = 0) ↔ a = -1 ∧ b = -9 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_origin_l1089_108955


namespace NUMINAMATH_GPT_john_subtracts_79_l1089_108901

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end NUMINAMATH_GPT_john_subtracts_79_l1089_108901


namespace NUMINAMATH_GPT_multiply_by_15_is_225_l1089_108950

-- Define the condition
def number : ℕ := 15

-- State the theorem with the conditions and the expected result
theorem multiply_by_15_is_225 : 15 * number = 225 := by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_multiply_by_15_is_225_l1089_108950


namespace NUMINAMATH_GPT_total_exterior_angles_l1089_108985

-- Define that the sum of the exterior angles of any convex polygon is 360 degrees
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Given four polygons: a triangle, a quadrilateral, a pentagon, and a hexagon
def triangle_exterior_sum := sum_exterior_angles 3
def quadrilateral_exterior_sum := sum_exterior_angles 4
def pentagon_exterior_sum := sum_exterior_angles 5
def hexagon_exterior_sum := sum_exterior_angles 6

-- The total sum of the exterior angles of these four polygons combined
def total_exterior_angle_sum := 
  triangle_exterior_sum + 
  quadrilateral_exterior_sum + 
  pentagon_exterior_sum + 
  hexagon_exterior_sum

-- The final proof statement
theorem total_exterior_angles : total_exterior_angle_sum = 1440 := by
  sorry

end NUMINAMATH_GPT_total_exterior_angles_l1089_108985


namespace NUMINAMATH_GPT_weight_of_33rd_weight_l1089_108902

theorem weight_of_33rd_weight :
  ∃ a : ℕ → ℕ, (∀ k, a k < a (k+1)) ∧
               (∀ k ≤ 29, a k + a (k+3) = a (k+1) + a (k+2)) ∧
               a 2 = 9 ∧
               a 8 = 33 ∧
               a 32 = 257 :=
sorry

end NUMINAMATH_GPT_weight_of_33rd_weight_l1089_108902


namespace NUMINAMATH_GPT_find_width_of_room_l1089_108919

theorem find_width_of_room
    (length : ℝ) (area : ℝ)
    (h1 : length = 12) (h2 : area = 96) :
    ∃ width : ℝ, width = 8 ∧ area = length * width :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_room_l1089_108919


namespace NUMINAMATH_GPT_exercise_l1089_108916

theorem exercise (x y z : ℝ)
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : 1/x + 1/y + 1/z = 3/5) : x^2 + y^2 + z^2 = 488.4 :=
sorry

end NUMINAMATH_GPT_exercise_l1089_108916


namespace NUMINAMATH_GPT_pie_difference_l1089_108941

theorem pie_difference (p1 p2 : ℚ) (h1 : p1 = 5 / 6) (h2 : p2 = 2 / 3) : p1 - p2 = 1 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_pie_difference_l1089_108941


namespace NUMINAMATH_GPT_product_of_all_possible_values_l1089_108983

theorem product_of_all_possible_values (x : ℝ) (h : 2 * |x + 3| - 4 = 2) :
  ∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_of_all_possible_values_l1089_108983


namespace NUMINAMATH_GPT_rodney_probability_correct_guess_l1089_108942

noncomputable def two_digit_integer (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

noncomputable def tens_digit (n : ℕ) : Prop :=
  (n / 10 = 7 ∨ n / 10 = 8 ∨ n / 10 = 9)

noncomputable def units_digit_even (n : ℕ) : Prop :=
  (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8)

noncomputable def greater_than_seventy_five (n : ℕ) : Prop := n > 75

theorem rodney_probability_correct_guess (n : ℕ) :
  two_digit_integer n →
  tens_digit n →
  units_digit_even n →
  greater_than_seventy_five n →
  (∃ m, m = 1 / 12) :=
sorry

end NUMINAMATH_GPT_rodney_probability_correct_guess_l1089_108942


namespace NUMINAMATH_GPT_range_of_k_for_circle_l1089_108969

theorem range_of_k_for_circle (x y : ℝ) (k : ℝ) : 
  (x^2 + y^2 - 4*x + 2*y + 5*k = 0) → k < 1 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_k_for_circle_l1089_108969


namespace NUMINAMATH_GPT_part1_part2_l1089_108926

def set_A := {x : ℝ | x^2 + 2*x - 8 = 0}
def set_B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem part1 (a : ℝ) (h : a = 1) : 
  (set_A ∩ set_B a) = {-4} := by
  sorry

theorem part2 (a : ℝ) : 
  (set_A ∩ (set_B a) = set_B a) → (a < -1 ∨ a > 3) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1089_108926


namespace NUMINAMATH_GPT_problem_solution_l1089_108924

noncomputable def time_without_distraction : ℝ :=
  let rate_A := 1 / 10
  let rate_B := 0.75 * rate_A
  let rate_C := 0.5 * rate_A
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

noncomputable def time_with_distraction : ℝ :=
  let rate_A := 0.9 * (1 / 10)
  let rate_B := 0.9 * (0.75 * (1 / 10))
  let rate_C := 0.9 * (0.5 * (1 / 10))
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

theorem problem_solution :
  time_without_distraction = 40 / 9 ∧
  time_with_distraction = 44.44 / 9 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1089_108924


namespace NUMINAMATH_GPT_ratio_of_volumes_l1089_108951

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem ratio_of_volumes :
  let r_C := 10
  let h_C := 20
  let r_D := 18
  let h_D := 12
  volume_cone r_C h_C / volume_cone r_D h_D = 125 / 243 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1089_108951


namespace NUMINAMATH_GPT_fraction_disliking_but_liking_l1089_108912

-- Definitions based on conditions
def total_students : ℕ := 100
def like_dancing : ℕ := 70
def dislike_dancing : ℕ := total_students - like_dancing

def say_they_like_dancing (like_dancing : ℕ) : ℕ := (70 * like_dancing) / 100
def say_they_dislike_dancing (like_dancing : ℕ) : ℕ := like_dancing - say_they_like_dancing like_dancing

def dislike_and_say_dislike (dislike_dancing : ℕ) : ℕ := (80 * dislike_dancing) / 100
def say_dislike_but_like (like_dancing : ℕ) : ℕ := say_they_dislike_dancing like_dancing

def total_say_dislike : ℕ := dislike_and_say_dislike dislike_dancing + say_dislike_but_like like_dancing

noncomputable def fraction_like_but_say_dislike : ℚ := (say_dislike_but_like like_dancing : ℚ) / (total_say_dislike : ℚ)

theorem fraction_disliking_but_liking : fraction_like_but_say_dislike = 46.67 / 100 := 
by sorry

end NUMINAMATH_GPT_fraction_disliking_but_liking_l1089_108912


namespace NUMINAMATH_GPT_common_number_exists_l1089_108981

def sum_of_list (l : List ℚ) : ℚ := l.sum

theorem common_number_exists (l1 l2 : List ℚ) (commonNumber : ℚ) 
    (h1 : l1.length = 5) 
    (h2 : l2.length = 5) 
    (h3 : sum_of_list l1 / 5 = 7) 
    (h4 : sum_of_list l2 / 5 = 10) 
    (h5 : (sum_of_list l1 + sum_of_list l2 - commonNumber) / 9 = 74 / 9) 
    : commonNumber = 11 :=
sorry

end NUMINAMATH_GPT_common_number_exists_l1089_108981


namespace NUMINAMATH_GPT_value_2_std_dev_less_than_mean_l1089_108910

def mean : ℝ := 16.5
def std_dev : ℝ := 1.5

theorem value_2_std_dev_less_than_mean :
  mean - 2 * std_dev = 13.5 := by
  sorry

end NUMINAMATH_GPT_value_2_std_dev_less_than_mean_l1089_108910


namespace NUMINAMATH_GPT_basic_computer_price_l1089_108999

variable (C P : ℕ)

theorem basic_computer_price 
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3) : 
  C = 1500 := 
sorry

end NUMINAMATH_GPT_basic_computer_price_l1089_108999


namespace NUMINAMATH_GPT_not_solution_of_equation_l1089_108931

theorem not_solution_of_equation (a : ℝ) (h : a ≠ 0) : ¬ (a^2 * 1^2 + (a + 1) * 1 + 1 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_not_solution_of_equation_l1089_108931


namespace NUMINAMATH_GPT_sum_first_seven_terms_of_arith_seq_l1089_108948

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Conditions: a_2 = 10 and a_5 = 1
def a_2 := 10
def a_5 := 1

-- The sum of the first 7 terms of the sequence
theorem sum_first_seven_terms_of_arith_seq (a d : ℤ) :
  arithmetic_seq a d 1 = a_2 →
  arithmetic_seq a d 4 = a_5 →
  (7 * a + (7 * 6 / 2) * d = 28) :=
by
  sorry

end NUMINAMATH_GPT_sum_first_seven_terms_of_arith_seq_l1089_108948


namespace NUMINAMATH_GPT_max_sheets_one_participant_l1089_108929

theorem max_sheets_one_participant
  (n : ℕ) (avg_sheets : ℕ) (h1 : n = 40) (h2 : avg_sheets = 7) 
  (h3 : ∀ i : ℕ, i < n → 1 ≤ 1) : 
  ∃ max_sheets : ℕ, max_sheets = 241 :=
by
  sorry

end NUMINAMATH_GPT_max_sheets_one_participant_l1089_108929


namespace NUMINAMATH_GPT_neg_q_sufficient_not_necc_neg_p_l1089_108966

variable (p q : Prop)

theorem neg_q_sufficient_not_necc_neg_p (hp: p → q) (hnpq: ¬(q → p)) : (¬q → ¬p) ∧ (¬(¬p → ¬q)) :=
by
  sorry

end NUMINAMATH_GPT_neg_q_sufficient_not_necc_neg_p_l1089_108966


namespace NUMINAMATH_GPT_tiling_condition_l1089_108965

theorem tiling_condition (a b n : ℕ) : 
  (∃ f : ℕ → ℕ × ℕ, ∀ i < (a * b) / n, (f i).fst < a ∧ (f i).snd < b) ↔ (n ∣ a ∨ n ∣ b) :=
sorry

end NUMINAMATH_GPT_tiling_condition_l1089_108965


namespace NUMINAMATH_GPT_num_of_elements_l1089_108975

-- Lean statement to define and prove the problem condition
theorem num_of_elements (n S : ℕ) (h1 : (S + 26) / n = 5) (h2 : (S + 36) / n = 6) : n = 10 := by
  sorry

end NUMINAMATH_GPT_num_of_elements_l1089_108975


namespace NUMINAMATH_GPT_b_1001_value_l1089_108930

theorem b_1001_value (b : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) 
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 888 = 17 + Real.sqrt 11) : 
  b 1001 = 7 * Real.sqrt 11 - 20 := sorry

end NUMINAMATH_GPT_b_1001_value_l1089_108930


namespace NUMINAMATH_GPT_order_options_count_l1089_108911

/-- Define the number of options for each category -/
def drinks : ℕ := 3
def salads : ℕ := 2
def pizzas : ℕ := 5

/-- The theorem statement that we aim to prove -/
theorem order_options_count : drinks * salads * pizzas = 30 :=
by
  sorry -- Proof is skipped as instructed

end NUMINAMATH_GPT_order_options_count_l1089_108911


namespace NUMINAMATH_GPT_sum_first_15_nat_eq_120_l1089_108978

-- Define a function to sum the first n natural numbers
def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define the theorem to show that the sum of the first 15 natural numbers equals 120
theorem sum_first_15_nat_eq_120 : sum_natural_numbers 15 = 120 := 
  by
    sorry

end NUMINAMATH_GPT_sum_first_15_nat_eq_120_l1089_108978


namespace NUMINAMATH_GPT_farey_sequence_mediant_l1089_108928

theorem farey_sequence_mediant (a b x y c d : ℕ) (h₁ : a * y < b * x) (h₂ : b * x < y * c) (farey_consecutiveness: bx - ay = 1 ∧ cy - dx = 1) : (x / y) = (a+c) / (b+d) := 
by
  sorry

end NUMINAMATH_GPT_farey_sequence_mediant_l1089_108928


namespace NUMINAMATH_GPT_multiply_vars_l1089_108988

variables {a b : ℝ}

theorem multiply_vars : -3 * a * b * 2 * a = -6 * a^2 * b := by
  sorry

end NUMINAMATH_GPT_multiply_vars_l1089_108988


namespace NUMINAMATH_GPT_average_mark_of_excluded_students_l1089_108954

theorem average_mark_of_excluded_students
  (N : ℕ) (A A_remaining : ℕ)
  (num_excluded : ℕ)
  (hN : N = 9)
  (hA : A = 60)
  (hA_remaining : A_remaining = 80)
  (h_excluded : num_excluded = 5) :
  (N * A - (N - num_excluded) * A_remaining) / num_excluded = 44 :=
by
  sorry

end NUMINAMATH_GPT_average_mark_of_excluded_students_l1089_108954


namespace NUMINAMATH_GPT_tangent_line_and_curve_l1089_108987

theorem tangent_line_and_curve (a x0 : ℝ) 
  (h1 : ∀ (x : ℝ), x0 + a = 1) 
  (h2 : ∀ (y : ℝ), y = x0 + 1) 
  (h3 : ∀ (y : ℝ), y = Real.log (x0 + a)) 
  : a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_tangent_line_and_curve_l1089_108987


namespace NUMINAMATH_GPT_a_n_nonzero_l1089_108993

/-- Recurrence relation for the sequence a_n --/
def a : ℕ → ℤ
| 0 => 1
| 1 => 2
| (n + 2) => if (a n * a (n + 1)) % 2 = 1 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

/-- Proof that for all n, a_n is non-zero --/
theorem a_n_nonzero : ∀ n : ℕ, a n ≠ 0 := 
sorry

end NUMINAMATH_GPT_a_n_nonzero_l1089_108993


namespace NUMINAMATH_GPT_find_g_inverse_84_l1089_108947

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- Definition stating the goal
theorem find_g_inverse_84 : g⁻¹ 84 = 3 :=
sorry

end NUMINAMATH_GPT_find_g_inverse_84_l1089_108947


namespace NUMINAMATH_GPT_find_students_l1089_108945

theorem find_students (n : ℕ) (h1 : n % 8 = 5) (h2 : n % 6 = 1) (h3 : n < 50) : n = 13 :=
sorry

end NUMINAMATH_GPT_find_students_l1089_108945


namespace NUMINAMATH_GPT_correct_equation_l1089_108925

-- Definitions of the conditions
def contributes_5_coins (x : ℕ) (P : ℕ) : Prop :=
  5 * x + 45 = P

def contributes_7_coins (x : ℕ) (P : ℕ) : Prop :=
  7 * x + 3 = P

-- Mathematical proof problem
theorem correct_equation 
(x : ℕ) (P : ℕ) (h1 : contributes_5_coins x P) (h2 : contributes_7_coins x P) : 
5 * x + 45 = 7 * x + 3 := 
by
  sorry

end NUMINAMATH_GPT_correct_equation_l1089_108925


namespace NUMINAMATH_GPT_factorize_expression1_factorize_expression2_l1089_108982

section
variable (x y : ℝ)

theorem factorize_expression1 : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

theorem factorize_expression2 : 3 * x^3 - 12 * x^2 * y + 12 * x * y^2 = 3 * x * (x - 2 * y)^2 :=
sorry
end

end NUMINAMATH_GPT_factorize_expression1_factorize_expression2_l1089_108982


namespace NUMINAMATH_GPT_system_has_infinite_solutions_l1089_108979

theorem system_has_infinite_solutions :
  ∀ (x y : ℝ), (3 * x - 4 * y = 5) ↔ (6 * x - 8 * y = 10) ∧ (9 * x - 12 * y = 15) :=
by
  sorry

end NUMINAMATH_GPT_system_has_infinite_solutions_l1089_108979


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1089_108913

-- Define sequence and sum properties
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

/- Theorem Statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) 
  (h_seq : arithmetic_sequence a d) 
  (h_initial : a 1 = 31) 
  (h_S_eq : S 10 = S 22) :
  -- Part 1: Find S_n
  (∀ n, S n = 32 * n - n ^ 2) ∧
  -- Part 2: Maximum sum occurs at n = 16 and is 256
  (∀ n, S n ≤ 256 ∧ (S 16 = 256 → ∀ m ≠ 16, S m < 256)) :=
by
  -- proof to be provided here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1089_108913


namespace NUMINAMATH_GPT_expression_is_nonnegative_l1089_108914

noncomputable def expression_nonnegative (a b c d e : ℝ) : Prop :=
  (a - b) * (a - c) * (a - d) * (a - e) +
  (b - a) * (b - c) * (b - d) * (b - e) +
  (c - a) * (c - b) * (c - d) * (c - e) +
  (d - a) * (d - b) * (d - c) * (d - e) +
  (e - a) * (e - b) * (e - c) * (e - d) ≥ 0

theorem expression_is_nonnegative (a b c d e : ℝ) : expression_nonnegative a b c d e := 
by 
  sorry

end NUMINAMATH_GPT_expression_is_nonnegative_l1089_108914


namespace NUMINAMATH_GPT_value_of_s_l1089_108937

-- Define the variables as integers (they represent non-zero digits)
variables {a p v e s r : ℕ}

-- Define the conditions as hypotheses
theorem value_of_s (h1 : a + p = v) (h2 : v + e = s) (h3 : s + a = r) (h4 : p + e + r = 14) :
  s = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_s_l1089_108937


namespace NUMINAMATH_GPT_Joe_spent_800_on_hotel_l1089_108994

noncomputable def Joe'sExpenses : Prop :=
  let S := 6000 -- Joe's total savings
  let F := 1200 -- Expense on the flight
  let FD := 3000 -- Expense on food
  let R := 1000 -- Remaining amount after all expenses
  let H := S - R - (F + FD) -- Calculating hotel expense
  H = 800 -- We need to prove the hotel expense equals $800

theorem Joe_spent_800_on_hotel : Joe'sExpenses :=
by {
  -- Proof goes here; currently skipped
  sorry
}

end NUMINAMATH_GPT_Joe_spent_800_on_hotel_l1089_108994


namespace NUMINAMATH_GPT_divide_milk_into_equal_parts_l1089_108973

def initial_state : (ℕ × ℕ × ℕ) := (8, 0, 0)

def is_equal_split (state : ℕ × ℕ × ℕ) : Prop :=
  state.1 = 4 ∧ state.2 = 4

theorem divide_milk_into_equal_parts : 
  ∃ (state_steps : Fin 25 → ℕ × ℕ × ℕ),
  initial_state = state_steps 0 ∧
  is_equal_split (state_steps 24) :=
sorry

end NUMINAMATH_GPT_divide_milk_into_equal_parts_l1089_108973


namespace NUMINAMATH_GPT_compute_fraction_l1089_108957

theorem compute_fraction (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) (sum_eq : x + y + z = 12) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l1089_108957


namespace NUMINAMATH_GPT_ball_arrangement_divisibility_l1089_108958

theorem ball_arrangement_divisibility :
  ∀ (n : ℕ), (∀ (i : ℕ), i < n → (∃ j k l m : ℕ, j < k ∧ k < l ∧ l < m ∧ m < n ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ j
    ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m)) →
  ¬((n = 2021) ∨ (n = 2022) ∨ (n = 2023) ∨ (n = 2024)) :=
sorry

end NUMINAMATH_GPT_ball_arrangement_divisibility_l1089_108958


namespace NUMINAMATH_GPT_octavio_can_reach_3_pow_2023_l1089_108936

theorem octavio_can_reach_3_pow_2023 (n : ℤ) (hn : n ≥ 1) :
  ∃ (steps : ℕ → ℤ), steps 0 = n ∧ (∀ k, steps (k + 1) = 3 * (steps k)) ∧
  steps 2023 = 3 ^ 2023 :=
by
  sorry

end NUMINAMATH_GPT_octavio_can_reach_3_pow_2023_l1089_108936


namespace NUMINAMATH_GPT_cookies_batches_needed_l1089_108922

noncomputable def number_of_recipes (total_students : ℕ) (attendance_drop : ℝ) (cookies_per_batch : ℕ) : ℕ :=
  let remaining_students := (total_students : ℝ) * (1 - attendance_drop)
  let total_cookies := remaining_students * 2
  let recipes_needed := total_cookies / cookies_per_batch
  (Nat.ceil recipes_needed : ℕ)

theorem cookies_batches_needed :
  number_of_recipes 150 0.40 18 = 10 :=
by
  sorry

end NUMINAMATH_GPT_cookies_batches_needed_l1089_108922


namespace NUMINAMATH_GPT_students_enrolled_in_only_english_l1089_108984

theorem students_enrolled_in_only_english (total_students both_english_german total_german : ℕ) (h1 : total_students = 40) (h2 : both_english_german = 12) (h3 : total_german = 22) (h4 : ∀ s, s < 40) :
  (total_students - (total_german - both_english_german) - both_english_german) = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_students_enrolled_in_only_english_l1089_108984


namespace NUMINAMATH_GPT_angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l1089_108934

-- Definitions of the sides and conditions in triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: a + b = 6
axiom sum_of_sides : a + b = 6

-- Condition: Area of triangle ABC is 2 * sqrt(3)
axiom area_of_triangle : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3

-- Condition: a cos B + b cos A = 2c cos C
axiom cos_condition : (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C

-- Proof problem 1: Prove that C = π/3
theorem angle_C_is_pi_div_3 (h_cos : Real.cos C = 1/2) : C = Real.pi / 3 :=
sorry

-- Proof problem 2: Prove that c = 2 sqrt(3)
theorem side_c_is_2_sqrt_3 (h_sin : Real.sin C = Real.sqrt 3 / 2) : c = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l1089_108934


namespace NUMINAMATH_GPT_find_ac_and_area_l1089_108940

variables {a b c : ℝ} {A B C : ℝ}
variables (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
variables (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4)
variables (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2)

noncomputable def ac_value := 2

noncomputable def area_of_triangle_abc := (Real.sqrt 15) / 4

theorem find_ac_and_area (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
                         (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4) 
                         (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2):
  ac_value = 2 ∧
  area_of_triangle_abc = (Real.sqrt 15) / 4 := 
sorry

end NUMINAMATH_GPT_find_ac_and_area_l1089_108940


namespace NUMINAMATH_GPT_exponentiation_problem_l1089_108918

theorem exponentiation_problem : 10^6 * (10^2)^3 / 10^4 = 10^8 := 
by 
  sorry

end NUMINAMATH_GPT_exponentiation_problem_l1089_108918


namespace NUMINAMATH_GPT_proof_inequality_l1089_108986

variable {a b c d : ℝ}

theorem proof_inequality (h1 : a + b + c + d = 6) (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
sorry

end NUMINAMATH_GPT_proof_inequality_l1089_108986


namespace NUMINAMATH_GPT_part1_complement_intersection_part2_range_m_l1089_108904

open Set

-- Define set A
def A : Set ℝ := { x | -1 ≤ x ∧ x < 4 }

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2 }

-- Part (1): Prove the complement of the intersection for m = 3
theorem part1_complement_intersection :
  ∀ x : ℝ, x ∉ (A ∩ B 3) ↔ x < 3 ∨ x ≥ 4 :=
by
  sorry

-- Part (2): Prove the range of m for A ∩ B = ∅
theorem part2_range_m (m : ℝ) :
  (A ∩ B m = ∅) ↔ m < -3 ∨ m ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_part1_complement_intersection_part2_range_m_l1089_108904


namespace NUMINAMATH_GPT_city_a_location_l1089_108932

theorem city_a_location (ϕ A_latitude : ℝ) (m : ℝ) (h_eq_height : true)
  (h_shadows_3x : true) 
  (h_angle: true) (h_southern : A_latitude < 0) 
  (h_rad_lat : ϕ = abs A_latitude):

  ϕ = 45 ∨ ϕ = 7.14 :=
by 
  sorry

end NUMINAMATH_GPT_city_a_location_l1089_108932


namespace NUMINAMATH_GPT_f_value_at_4_l1089_108963

def f : ℝ → ℝ := sorry  -- Define f as a function from ℝ to ℝ

-- Specify the condition that f satisfies for all real numbers x
axiom f_condition (x : ℝ) : f (2^x) + x * f (2^(-x)) = 3

-- Statement to be proven: f(4) = -3
theorem f_value_at_4 : f 4 = -3 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_f_value_at_4_l1089_108963


namespace NUMINAMATH_GPT_tan_beta_minus_2alpha_l1089_108972

theorem tan_beta_minus_2alpha (alpha beta : ℝ) (h1 : Real.tan alpha = 2) (h2 : Real.tan (beta - alpha) = 3) : 
  Real.tan (beta - 2 * alpha) = 1 / 7 := 
sorry

end NUMINAMATH_GPT_tan_beta_minus_2alpha_l1089_108972


namespace NUMINAMATH_GPT_pay_docked_per_lateness_l1089_108938

variable (hourly_rate : ℤ) (work_hours : ℤ) (times_late : ℕ) (actual_pay : ℤ) 

theorem pay_docked_per_lateness (h_rate : hourly_rate = 30) 
                                (w_hours : work_hours = 18) 
                                (t_late : times_late = 3) 
                                (a_pay : actual_pay = 525) :
                                (hourly_rate * work_hours - actual_pay) / times_late = 5 :=
by
  sorry

end NUMINAMATH_GPT_pay_docked_per_lateness_l1089_108938


namespace NUMINAMATH_GPT_correct_calculation_result_l1089_108917

theorem correct_calculation_result 
  (P : Polynomial ℝ := -x^2 + x - 1) :
  (P + -3 * x) = (-x^2 - 2 * x - 1) :=
by
  -- Since this is just the proof statement, sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l1089_108917


namespace NUMINAMATH_GPT_evaluate_expression_eq_l1089_108923

theorem evaluate_expression_eq :
  let x := 2
  let y := -3
  let z := 7
  x^2 + y^2 - z^2 - 2 * x * y + 3 * z = -15 := by
    sorry

end NUMINAMATH_GPT_evaluate_expression_eq_l1089_108923


namespace NUMINAMATH_GPT_reciprocal_neg_half_l1089_108935

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_half_l1089_108935


namespace NUMINAMATH_GPT_complex_product_l1089_108915

theorem complex_product (i : ℂ) (h : i^2 = -1) :
  (3 - 4 * i) * (2 + 7 * i) = 34 + 13 * i :=
sorry

end NUMINAMATH_GPT_complex_product_l1089_108915


namespace NUMINAMATH_GPT_option_a_solution_l1089_108956

theorem option_a_solution (x y : ℕ) (h₁: x = 2) (h₂: y = 2) : 2 * x + y = 6 := by
sorry

end NUMINAMATH_GPT_option_a_solution_l1089_108956


namespace NUMINAMATH_GPT_range_of_a_l1089_108995

open Set

theorem range_of_a (a : ℝ) : (-3 < a ∧ a < -1) ↔ (∀ x, x < -1 ∨ 5 < x ∨ (a < x ∧ x < a+8)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1089_108995


namespace NUMINAMATH_GPT_problem_l1089_108967

theorem problem (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2007 = 2008 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1089_108967


namespace NUMINAMATH_GPT_mark_total_votes_l1089_108997

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end NUMINAMATH_GPT_mark_total_votes_l1089_108997
