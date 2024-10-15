import Mathlib

namespace NUMINAMATH_GPT_fish_remaining_l2349_234937

theorem fish_remaining
  (initial_guppies : ℕ)
  (initial_angelfish : ℕ)
  (initial_tiger_sharks : ℕ)
  (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ)
  (sold_angelfish : ℕ)
  (sold_tiger_sharks : ℕ)
  (sold_oscar_fish : ℕ)
  (initial_total : ℕ := initial_guppies + initial_angelfish + initial_tiger_sharks + initial_oscar_fish)
  (sold_total : ℕ := sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish)
  (remaining : ℕ := initial_total - sold_total) :
  initial_guppies = 94 →
  initial_angelfish = 76 →
  initial_tiger_sharks = 89 →
  initial_oscar_fish = 58 →
  sold_guppies = 30 →
  sold_angelfish = 48 →
  sold_tiger_sharks = 17 →
  sold_oscar_fish = 24 →
  remaining = 198 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  norm_num
  sorry

end NUMINAMATH_GPT_fish_remaining_l2349_234937


namespace NUMINAMATH_GPT_min_m_plus_n_l2349_234934

theorem min_m_plus_n (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - 2 * m - 3 * n = 20) : 
  m + n = 20 :=
sorry

end NUMINAMATH_GPT_min_m_plus_n_l2349_234934


namespace NUMINAMATH_GPT_sum_reciprocals_of_partial_fractions_l2349_234969

noncomputable def f (s : ℝ) : ℝ := s^3 - 20 * s^2 + 125 * s - 500

theorem sum_reciprocals_of_partial_fractions :
  ∀ (p q r A B C : ℝ),
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    f p = 0 ∧ f q = 0 ∧ f r = 0 ∧
    (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
      (1 / f s = A / (s - p) + B / (s - q) + C / (s - r))) →
    1 / A + 1 / B + 1 / C = 720 :=
sorry

end NUMINAMATH_GPT_sum_reciprocals_of_partial_fractions_l2349_234969


namespace NUMINAMATH_GPT_kevin_food_spending_l2349_234936

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end NUMINAMATH_GPT_kevin_food_spending_l2349_234936


namespace NUMINAMATH_GPT_polygon_perimeter_exposure_l2349_234999

theorem polygon_perimeter_exposure:
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let exposure_triangle_nonagon := triangle_sides + nonagon_sides - 2
  let other_polygons_adjacency := 2 * 5
  let exposure_other_polygons := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - other_polygons_adjacency
  exposure_triangle_nonagon + exposure_other_polygons = 30 :=
by sorry

end NUMINAMATH_GPT_polygon_perimeter_exposure_l2349_234999


namespace NUMINAMATH_GPT_length_stationary_l2349_234900

def speed : ℝ := 64.8
def time_pole : ℝ := 5
def time_stationary : ℝ := 25

def length_moving : ℝ := speed * time_pole
def length_combined : ℝ := speed * time_stationary

theorem length_stationary : length_combined - length_moving = 1296 :=
by
  sorry

end NUMINAMATH_GPT_length_stationary_l2349_234900


namespace NUMINAMATH_GPT_stock_status_after_limit_moves_l2349_234990

theorem stock_status_after_limit_moves (initial_value : ℝ) (h₁ : initial_value = 1)
  (limit_up_factor : ℝ) (h₂ : limit_up_factor = 1 + 0.10)
  (limit_down_factor : ℝ) (h₃ : limit_down_factor = 1 - 0.10) :
  (limit_up_factor^5 * limit_down_factor^5) < initial_value :=
by
  sorry

end NUMINAMATH_GPT_stock_status_after_limit_moves_l2349_234990


namespace NUMINAMATH_GPT_fourth_person_height_l2349_234931

theorem fourth_person_height (h : ℝ)
  (h2 : h + 2 = h₂)
  (h3 : h + 4 = h₃)
  (h4 : h + 10 = h₄)
  (average_height : (h + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 :=
by
  sorry

end NUMINAMATH_GPT_fourth_person_height_l2349_234931


namespace NUMINAMATH_GPT_inequality_proof_l2349_234951

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z) ^ 2 * (y * z + z * x + x * y) ^ 2 ≤ 
  3 * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) * (x^2 + x * y + y^2) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l2349_234951


namespace NUMINAMATH_GPT_average_marbles_of_other_colors_l2349_234995

theorem average_marbles_of_other_colors
  (clear_percentage : ℝ) (black_percentage : ℝ) (total_marbles_taken : ℕ)
  (h1 : clear_percentage = 0.4) (h2 : black_percentage = 0.2) :
  (total_marbles_taken : ℝ) * (1 - clear_percentage - black_percentage) = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_marbles_of_other_colors_l2349_234995


namespace NUMINAMATH_GPT_original_savings_l2349_234996

-- Define original savings as a variable
variable (S : ℝ)

-- Define the condition that 1/4 of the savings equals 200
def tv_cost_condition : Prop := (1 / 4) * S = 200

-- State the theorem that if the condition is satisfied, then the original savings are 800
theorem original_savings (h : tv_cost_condition S) : S = 800 :=
by
  sorry

end NUMINAMATH_GPT_original_savings_l2349_234996


namespace NUMINAMATH_GPT_least_number_to_multiply_l2349_234968

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end NUMINAMATH_GPT_least_number_to_multiply_l2349_234968


namespace NUMINAMATH_GPT_fraction_over_65_l2349_234985

def num_people_under_21 := 33
def fraction_under_21 := 3 / 7
def total_people (N : ℕ) := N > 50 ∧ N < 100
def num_people (N : ℕ) := num_people_under_21 = fraction_under_21 * N

theorem fraction_over_65 (N : ℕ) : 
  total_people N → num_people N → N = 77 ∧ ∃ x, (x / 77) = x / 77 :=
by
  intro hN hnum
  sorry

end NUMINAMATH_GPT_fraction_over_65_l2349_234985


namespace NUMINAMATH_GPT_maximum_marks_l2349_234918

theorem maximum_marks (M : ℝ) (h1 : 212 + 25 = 237) (h2 : 0.30 * M = 237) : M = 790 := 
by
  sorry

end NUMINAMATH_GPT_maximum_marks_l2349_234918


namespace NUMINAMATH_GPT_father_l2349_234950

theorem father's_age (M F : ℕ) (h1 : M = 2 * F / 5) (h2 : M + 6 = (F + 6) / 2) : F = 30 :=
by
  sorry

end NUMINAMATH_GPT_father_l2349_234950


namespace NUMINAMATH_GPT_red_car_speed_l2349_234970

noncomputable def speed_blue : ℕ := 80
noncomputable def speed_green : ℕ := 8 * speed_blue
noncomputable def speed_red : ℕ := 2 * speed_green

theorem red_car_speed : speed_red = 1280 := by
  unfold speed_red
  unfold speed_green
  unfold speed_blue
  sorry

end NUMINAMATH_GPT_red_car_speed_l2349_234970


namespace NUMINAMATH_GPT_solve_cubic_fraction_l2349_234945

noncomputable def problem_statement (x : ℝ) :=
  (x = (-(3:ℝ) + Real.sqrt 13) / 4) ∨ (x = (-(3:ℝ) - Real.sqrt 13) / 4)

theorem solve_cubic_fraction (x : ℝ) (h : (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4) : 
  problem_statement x :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_fraction_l2349_234945


namespace NUMINAMATH_GPT_remainder_mod_7_l2349_234902

theorem remainder_mod_7 : (9^7 + 8^8 + 7^9) % 7 = 3 :=
by sorry

end NUMINAMATH_GPT_remainder_mod_7_l2349_234902


namespace NUMINAMATH_GPT_parabola_focus_l2349_234935

theorem parabola_focus (a : ℝ) (h1 : ∀ x y, x^2 = a * y ↔ y = x^2 / a)
(h2 : focus_coordinates = (0, 5)) : a = 20 := 
sorry

end NUMINAMATH_GPT_parabola_focus_l2349_234935


namespace NUMINAMATH_GPT_positive_integer_solutions_eq_8_2_l2349_234912

-- Define the variables and conditions in the problem
def positive_integer_solution_count_eq (n m : ℕ) : Prop :=
  ∀ (x₁ x₂ x₃ x₄ : ℕ),
    x₂ = m →
    (x₁ + x₂ + x₃ + x₄ = n) →
    (x₁ > 0 ∧ x₃ > 0 ∧ x₄ > 0) →
    -- Number of positive integer solutions should be 10
    (x₁ + x₃ + x₄ = 6)

-- Statement of the theorem
theorem positive_integer_solutions_eq_8_2 : positive_integer_solution_count_eq 8 2 := sorry

end NUMINAMATH_GPT_positive_integer_solutions_eq_8_2_l2349_234912


namespace NUMINAMATH_GPT_new_class_mean_l2349_234984

theorem new_class_mean 
  (n1 n2 : ℕ) (mean1 mean2 : ℚ) 
  (h1 : n1 = 24) (h2 : n2 = 8) 
  (h3 : mean1 = 85/100) (h4 : mean2 = 90/100) :
  (n1 * mean1 + n2 * mean2) / (n1 + n2) = 345/400 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_new_class_mean_l2349_234984


namespace NUMINAMATH_GPT_combined_tax_rate_l2349_234911

-- Definitions of the problem conditions
def tax_rate_Mork : ℝ := 0.40
def tax_rate_Mindy : ℝ := 0.25

-- Asserts the condition that Mindy earned 4 times as much as Mork
def income_ratio (income_Mindy income_Mork : ℝ) := income_Mindy = 4 * income_Mork

-- The theorem to be proved: The combined tax rate is 28%.
theorem combined_tax_rate (income_Mork income_Mindy total_income total_tax : ℝ)
  (h_income_ratio : income_ratio income_Mindy income_Mork)
  (total_income_eq : total_income = income_Mork + income_Mindy)
  (total_tax_eq : total_tax = tax_rate_Mork * income_Mork + tax_rate_Mindy * income_Mindy) :
  total_tax / total_income = 0.28 := sorry

end NUMINAMATH_GPT_combined_tax_rate_l2349_234911


namespace NUMINAMATH_GPT_inequality_solution_set_range_of_m_l2349_234979

-- Proof Problem 1
theorem inequality_solution_set :
  {x : ℝ | -2 < x ∧ x < 4} = { x : ℝ | 2 * x^2 - 4 * x - 16 < 0 } :=
sorry

-- Proof Problem 2
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 :=
  sorry

end NUMINAMATH_GPT_inequality_solution_set_range_of_m_l2349_234979


namespace NUMINAMATH_GPT_student_A_incorrect_l2349_234977

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  let (cx, cy) := center
  let (px, py) := point
  (px - cx)^2 + (py - cy)^2 = radius^2

def center : ℝ × ℝ := (2, -3)
def radius : ℝ := 5
def point_A : ℝ × ℝ := (-2, -1)
def point_D : ℝ × ℝ := (5, 1)

theorem student_A_incorrect :
  ¬ is_on_circle center radius point_A ∧ is_on_circle center radius point_D :=
by
  sorry

end NUMINAMATH_GPT_student_A_incorrect_l2349_234977


namespace NUMINAMATH_GPT_cookies_from_dough_l2349_234916

theorem cookies_from_dough :
  ∀ (length width : ℕ), length = 24 → width = 18 →
  ∃ (side : ℕ), side = Nat.gcd length width ∧ (length / side) * (width / side) = 12 :=
by
  intros length width h_length h_width
  simp only [h_length, h_width]
  use Nat.gcd length width
  simp only [Nat.gcd_rec]
  sorry

end NUMINAMATH_GPT_cookies_from_dough_l2349_234916


namespace NUMINAMATH_GPT_option_C_correct_l2349_234952

variable (a b : ℝ)

theorem option_C_correct (h : a > b) : -15 * a < -15 * b := 
  sorry

end NUMINAMATH_GPT_option_C_correct_l2349_234952


namespace NUMINAMATH_GPT_find_y_intercept_l2349_234960

theorem find_y_intercept (a b : ℝ) (h1 : (3 : ℝ) ≠ (7 : ℝ))
  (h2 : -2 = a * 3 + b) (h3 : 14 = a * 7 + b) :
  b = -14 :=
sorry

end NUMINAMATH_GPT_find_y_intercept_l2349_234960


namespace NUMINAMATH_GPT_sum_of_squares_largest_multiple_of_7_l2349_234964

theorem sum_of_squares_largest_multiple_of_7
  (N : ℕ) (a : ℕ) (h1 : N = a^2 + (a + 1)^2 + (a + 2)^2)
  (h2 : N < 10000)
  (h3 : 7 ∣ N) :
  N = 8750 := sorry

end NUMINAMATH_GPT_sum_of_squares_largest_multiple_of_7_l2349_234964


namespace NUMINAMATH_GPT_books_at_end_l2349_234972

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end NUMINAMATH_GPT_books_at_end_l2349_234972


namespace NUMINAMATH_GPT_gcd_pair_sum_ge_prime_l2349_234989

theorem gcd_pair_sum_ge_prime
  (n : ℕ)
  (h_prime: Prime (2*n - 1))
  (a : Fin n → ℕ)
  (h_distinct: ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j) / Nat.gcd (a i) (a j) ≥ 2*n - 1 := sorry

end NUMINAMATH_GPT_gcd_pair_sum_ge_prime_l2349_234989


namespace NUMINAMATH_GPT_extremum_condition_l2349_234942

noncomputable def y (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (y a x = 0) ∧ ∀ x' > x, y a x' < y a x) → a < -3 :=
by
  sorry

end NUMINAMATH_GPT_extremum_condition_l2349_234942


namespace NUMINAMATH_GPT_purchasing_methods_count_l2349_234974

theorem purchasing_methods_count :
  ∃ n, n = 6 ∧
    ∃ (x y : ℕ), 
      60 * x + 70 * y ≤ 500 ∧
      x ≥ 3 ∧
      y ≥ 2 :=
sorry

end NUMINAMATH_GPT_purchasing_methods_count_l2349_234974


namespace NUMINAMATH_GPT_price_of_70_cans_l2349_234958

noncomputable def regular_price_per_can : ℝ := 0.55
noncomputable def discount_rate_case : ℝ := 0.25
noncomputable def bulk_discount_rate : ℝ := 0.10
noncomputable def cans_per_case : ℕ := 24
noncomputable def total_cans_purchased : ℕ := 70

theorem price_of_70_cans :
  let discounted_price_per_can := regular_price_per_can * (1 - discount_rate_case)
  let discounted_price_for_cases := 48 * discounted_price_per_can
  let bulk_discount := if 70 >= 3 * cans_per_case then discounted_price_for_cases * bulk_discount_rate else 0
  let final_price_for_cases := discounted_price_for_cases - bulk_discount
  let additional_cans := total_cans_purchased % cans_per_case
  let price_for_additional_cans := additional_cans * discounted_price_per_can
  final_price_for_cases + price_for_additional_cans = 26.895 :=
by sorry

end NUMINAMATH_GPT_price_of_70_cans_l2349_234958


namespace NUMINAMATH_GPT_find_number_l2349_234910

theorem find_number (N : ℕ) (h1 : N / 3 = 8) (h2 : N / 8 = 3) : N = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2349_234910


namespace NUMINAMATH_GPT_tom_killed_enemies_l2349_234973

-- Define the number of points per enemy
def points_per_enemy : ℝ := 10

-- Define the bonus threshold and bonus factor
def bonus_threshold : ℝ := 100
def bonus_factor : ℝ := 1.5

-- Define the total score achieved by Tom
def total_score : ℝ := 2250

-- Define the number of enemies killed by Tom
variable (E : ℝ)

-- The proof goal
theorem tom_killed_enemies 
  (h1 : E ≥ bonus_threshold)
  (h2 : bonus_factor * points_per_enemy * E = total_score) : 
  E = 150 :=
sorry

end NUMINAMATH_GPT_tom_killed_enemies_l2349_234973


namespace NUMINAMATH_GPT_odd_function_symmetry_l2349_234907

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then x^2 else sorry

theorem odd_function_symmetry (x : ℝ) (k : ℕ) (h1 : ∀ y, f (-y) = -f y)
  (h2 : ∀ y, f y = f (2 - y)) (h3 : ∀ y, 0 < y ∧ y ≤ 1 → f y = y^2) :
  k = 45 / 4 → f k = -9 / 16 :=
by
  intros _
  sorry

end NUMINAMATH_GPT_odd_function_symmetry_l2349_234907


namespace NUMINAMATH_GPT_rectangle_difference_l2349_234906

theorem rectangle_difference (L B : ℝ) (h1 : 2 * (L + B) = 266) (h2 : L * B = 4290) :
  L - B = 23 :=
sorry

end NUMINAMATH_GPT_rectangle_difference_l2349_234906


namespace NUMINAMATH_GPT_william_total_tickets_l2349_234919

def initial_tickets : ℕ := 15
def additional_tickets : ℕ := 3
def total_tickets : ℕ := initial_tickets + additional_tickets

theorem william_total_tickets :
  total_tickets = 18 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_william_total_tickets_l2349_234919


namespace NUMINAMATH_GPT_least_value_expression_l2349_234962

theorem least_value_expression : ∃ x : ℝ, ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094
∧ ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 := by
  sorry

end NUMINAMATH_GPT_least_value_expression_l2349_234962


namespace NUMINAMATH_GPT_rate_of_stream_l2349_234923

theorem rate_of_stream (x : ℝ) (h1 : ∀ (distance : ℝ), (24 : ℝ) > 0) (h2 : ∀ (distance : ℝ), (distance / (24 - x)) = 3 * (distance / (24 + x))) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_stream_l2349_234923


namespace NUMINAMATH_GPT_value_of_f_ln3_l2349_234944

def f : ℝ → ℝ := sorry

theorem value_of_f_ln3 (f_symm : ∀ x : ℝ, f (x + 1) = f (-x + 1))
  (f_exp : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.exp (-x)) :
  f (Real.log 3) = 3 * Real.exp (-2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_ln3_l2349_234944


namespace NUMINAMATH_GPT_solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l2349_234921

-- Define the conditions
variables (a b c x y z: ℝ) 

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) : Prop :=
  (a * y + b * x = c) ∧
  (c * x + a * z = b) ∧
  (b * z + c * y = a)

-- Define the general solution
def solution (a b c x y z : ℝ) : Prop :=
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧
  z = (a^2 + b^2 - c^2) / (2 * a * b)

-- Define the proof problem statement
theorem solve_system_of_equations_general (a b c x y z : ℝ) (h : system_of_equations a b c x y z) 
      (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : solution a b c x y z :=
  sorry

-- Special cases
theorem solve_system_of_equations_zero_case_1 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_2 (a b c x y z : ℝ) (h1 : a = 0) (h2 : b = 0) (h3: c ≠ 0) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_3 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : x = c / b ∧ 
      (c * x = b) :=
  sorry

-- Following special cases more concisely
theorem solve_system_of_equations_special_cases (a b c x y z : ℝ) 
      (h : system_of_equations a b c x y z) (h1: a = 0 ∨ b = 0 ∨ c = 0): 
      (∃ k : ℝ, x = k ∧ y = -k ∧ z = k)  
    ∨ (∃ k : ℝ, x = k ∧ y = k ∧ z = -k)
    ∨ (∃ k : ℝ, x = -k ∧ y = k ∧ z = k) :=
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l2349_234921


namespace NUMINAMATH_GPT_geometric_sequence_product_l2349_234966

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n : ℕ, a (n + 1) = r * a n)
variable (h_condition : a 5 * a 14 = 5)

theorem geometric_sequence_product :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l2349_234966


namespace NUMINAMATH_GPT_min_people_for_no_empty_triplet_60_l2349_234920

noncomputable def min_people_for_no_empty_triplet (total_chairs : ℕ) : ℕ :=
  if h : total_chairs % 3 = 0 then total_chairs / 3 else sorry

theorem min_people_for_no_empty_triplet_60 :
  min_people_for_no_empty_triplet 60 = 20 :=
by
  sorry

end NUMINAMATH_GPT_min_people_for_no_empty_triplet_60_l2349_234920


namespace NUMINAMATH_GPT_product_square_preceding_div_by_12_l2349_234954

theorem product_square_preceding_div_by_12 (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_product_square_preceding_div_by_12_l2349_234954


namespace NUMINAMATH_GPT_sin_cos_identity_l2349_234903

variable {α : ℝ}

/-- Given 1 / sin(α) + 1 / cos(α) = √3, then sin(α) * cos(α) = -1 / 3 -/
theorem sin_cos_identity (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) : 
  Real.sin α * Real.cos α = -1 / 3 := 
sorry

end NUMINAMATH_GPT_sin_cos_identity_l2349_234903


namespace NUMINAMATH_GPT_distinct_collections_proof_l2349_234978

noncomputable def distinct_collections_count : ℕ := 240

theorem distinct_collections_proof : distinct_collections_count = 240 := by
  sorry

end NUMINAMATH_GPT_distinct_collections_proof_l2349_234978


namespace NUMINAMATH_GPT_find_k_l2349_234929

-- Define the vector operations and properties

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vector_smul (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def vectors_parallel (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.2 = a.2 * b.1)

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Statement of the problem
theorem find_k (k : ℝ) : 
  vectors_parallel (vector_add (vector_smul k a) b) (vector_add a (vector_smul (-3) b)) 
  → k = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2349_234929


namespace NUMINAMATH_GPT_find_f_2023_l2349_234948

def is_strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f a < f b

theorem find_f_2023 (f : ℕ → ℕ)
  (h_inc : is_strictly_increasing f)
  (h_relation : ∀ m n : ℕ, f (n + f m) = f n + m + 1) :
  f 2023 = 2024 :=
sorry

end NUMINAMATH_GPT_find_f_2023_l2349_234948


namespace NUMINAMATH_GPT_sum_of_integers_between_neg20_5_and_10_5_l2349_234925

theorem sum_of_integers_between_neg20_5_and_10_5 :
  let a := -20
  let l := 10
  let n := (l - a) / 1 + 1
  let S := n / 2 * (a + l)
  S = -155 := by
{
  sorry
}

end NUMINAMATH_GPT_sum_of_integers_between_neg20_5_and_10_5_l2349_234925


namespace NUMINAMATH_GPT_combined_population_after_two_years_l2349_234940

def population_after_years (initial_population : ℕ) (yearly_changes : List (ℕ → ℕ)) : ℕ :=
  yearly_changes.foldl (fun pop change => change pop) initial_population

def townA_change_year1 (pop : ℕ) : ℕ :=
  pop + (pop * 8 / 100) + 200 - 100

def townA_change_year2 (pop : ℕ) : ℕ :=
  pop + (pop * 10 / 100) + 200 - 100

def townB_change_year1 (pop : ℕ) : ℕ :=
  pop - (pop * 2 / 100) + 50 - 200

def townB_change_year2 (pop : ℕ) : ℕ :=
  pop - (pop * 1 / 100) + 50 - 200

theorem combined_population_after_two_years :
  population_after_years 15000 [townA_change_year1, townA_change_year2] +
  population_after_years 10000 [townB_change_year1, townB_change_year2] = 27433 := 
  sorry

end NUMINAMATH_GPT_combined_population_after_two_years_l2349_234940


namespace NUMINAMATH_GPT_at_least_one_not_less_than_four_l2349_234994

theorem at_least_one_not_less_than_four 
( m n t : ℝ ) 
( h_m : 0 < m ) 
( h_n : 0 < n ) 
( h_t : 0 < t ) : 
∃ a, ( a = m + 4 / n ∨ a = n + 4 / t ∨ a = t + 4 / m ) ∧ 4 ≤ a :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_four_l2349_234994


namespace NUMINAMATH_GPT_find_t_l2349_234986

-- Given a quadratic equation
def quadratic_eq (x : ℝ) := 4 * x ^ 2 - 16 * x - 200

-- Completing the square to find t
theorem find_t : ∃ q t : ℝ, (x : ℝ) → (quadratic_eq x = 0) → (x + q) ^ 2 = t ∧ t = 54 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l2349_234986


namespace NUMINAMATH_GPT_length_of_faster_train_proof_l2349_234913

-- Definitions based on the given conditions
def faster_train_speed_kmh := 72 -- in km/h
def slower_train_speed_kmh := 36 -- in km/h
def time_to_cross_seconds := 18 -- in seconds

-- Conversion factor from km/h to m/s
def kmh_to_ms := 5 / 18

-- Define the relative speed in m/s
def relative_speed_ms := (faster_train_speed_kmh - slower_train_speed_kmh) * kmh_to_ms

-- Length of the faster train in meters
def length_of_faster_train := relative_speed_ms * time_to_cross_seconds

-- The theorem statement for the Lean prover
theorem length_of_faster_train_proof : length_of_faster_train = 180 := by
  sorry

end NUMINAMATH_GPT_length_of_faster_train_proof_l2349_234913


namespace NUMINAMATH_GPT_dave_books_about_outer_space_l2349_234927

theorem dave_books_about_outer_space (x : ℕ) 
  (H1 : 8 + 3 = 11) 
  (H2 : 11 * 6 = 66) 
  (H3 : 102 - 66 = 36) 
  (H4 : 36 / 6 = x) : 
  x = 6 := 
by
  sorry

end NUMINAMATH_GPT_dave_books_about_outer_space_l2349_234927


namespace NUMINAMATH_GPT_balcony_more_than_orchestra_l2349_234997

variables (O B : ℕ) (H1 : O + B = 380) (H2 : 12 * O + 8 * B = 3320)

theorem balcony_more_than_orchestra : B - O = 240 :=
by sorry

end NUMINAMATH_GPT_balcony_more_than_orchestra_l2349_234997


namespace NUMINAMATH_GPT_trigonometric_identity_l2349_234915

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2349_234915


namespace NUMINAMATH_GPT_cleaning_project_l2349_234949

theorem cleaning_project (x : ℕ) : 12 + x = 2 * (15 - x) := sorry

end NUMINAMATH_GPT_cleaning_project_l2349_234949


namespace NUMINAMATH_GPT_second_integer_is_ninety_point_five_l2349_234971

theorem second_integer_is_ninety_point_five
  (n : ℝ)
  (first_integer fourth_integer : ℝ)
  (h1 : first_integer = n - 2)
  (h2 : fourth_integer = n + 1)
  (h_sum : first_integer + fourth_integer = 180) :
  n = 90.5 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_second_integer_is_ninety_point_five_l2349_234971


namespace NUMINAMATH_GPT_num_children_attended_show_l2349_234947

def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_adults : ℕ := 183
def total_revenue : ℕ := 5122

theorem num_children_attended_show : ∃ C : ℕ, (num_adults * ticket_price_adult + C * ticket_price_child = total_revenue) ∧ C = 28 :=
by
  sorry

end NUMINAMATH_GPT_num_children_attended_show_l2349_234947


namespace NUMINAMATH_GPT_polynomial_equivalence_l2349_234928

theorem polynomial_equivalence (x y : ℝ) (h : y = x + 1/x) :
  (x^2 * (y^2 + 2*y - 5) = 0) ↔ (x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_equivalence_l2349_234928


namespace NUMINAMATH_GPT_lcm_1_to_10_l2349_234922

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end NUMINAMATH_GPT_lcm_1_to_10_l2349_234922


namespace NUMINAMATH_GPT_polygon_diagonals_l2349_234988

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_polygon_diagonals_l2349_234988


namespace NUMINAMATH_GPT_values_of_a_for_single_root_l2349_234981

theorem values_of_a_for_single_root (a : ℝ) :
  (∃ (x : ℝ), ax^2 - 4 * x + 2 = 0) ∧ (∀ (x1 x2 : ℝ), ax^2 - 4 * x1 + 2 = 0 → ax^2 - 4 * x2 + 2 = 0 → x1 = x2) ↔ a = 0 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_values_of_a_for_single_root_l2349_234981


namespace NUMINAMATH_GPT_area_ratio_l2349_234901

theorem area_ratio (l w h : ℝ) (h1 : w * h = 288) (h2 : l * w = 432) (h3 : l * w * h = 5184) :
  (l * h) / (l * w) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_area_ratio_l2349_234901


namespace NUMINAMATH_GPT_lcm_ac_least_value_l2349_234982

theorem lcm_ac_least_value (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : 
  Nat.lcm a c = 30 :=
sorry

end NUMINAMATH_GPT_lcm_ac_least_value_l2349_234982


namespace NUMINAMATH_GPT_range_of_a_l2349_234941

theorem range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, (4 * x - 3 * y - 2 = 0) → (x^2 + y^2 - 2 * a * x + 4 * y + a^2 - 12 = 0) → x ≠ y) :
  -6 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2349_234941


namespace NUMINAMATH_GPT_volume_of_cube_in_pyramid_l2349_234943

open Real

noncomputable def side_length_of_base := 2
noncomputable def height_of_equilateral_triangle := sqrt 6
noncomputable def cube_side_length := sqrt 6 / 3
noncomputable def volume_of_cube := cube_side_length ^ 3

theorem volume_of_cube_in_pyramid 
  (side_length_of_base : ℝ) (height_of_equilateral_triangle : ℝ) (cube_side_length : ℝ) :
  volume_of_cube = 2 * sqrt 6 / 9 := 
by
  sorry

end NUMINAMATH_GPT_volume_of_cube_in_pyramid_l2349_234943


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_mul_three_eq_3480_l2349_234955

theorem arithmetic_sequence_sum_mul_three_eq_3480 :
  let a := 50
  let d := 3
  let l := 95
  let n := ((l - a) / d + 1 : ℕ)
  let sum := n * (a + l) / 2
  3 * sum = 3480 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_mul_three_eq_3480_l2349_234955


namespace NUMINAMATH_GPT_measure_of_angle_l2349_234914

theorem measure_of_angle (x : ℝ) (h : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_l2349_234914


namespace NUMINAMATH_GPT_compute_fraction_l2349_234953

theorem compute_fraction : ((5 * 7) - 3) / 9 = 32 / 9 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_l2349_234953


namespace NUMINAMATH_GPT_fg_of_2_l2349_234991

def f (x : ℤ) : ℤ := 4 * x + 3
def g (x : ℤ) : ℤ := x ^ 3 + 1

theorem fg_of_2 : f (g 2) = 39 := by
  sorry

end NUMINAMATH_GPT_fg_of_2_l2349_234991


namespace NUMINAMATH_GPT_Igor_colored_all_cells_l2349_234961

theorem Igor_colored_all_cells (m n : ℕ) (h1 : 9 * m = 12 * n) (h2 : 0 < m ∧ m ≤ 4) (h3 : 0 < n ∧ n ≤ 3) :
  m = 4 ∧ n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_Igor_colored_all_cells_l2349_234961


namespace NUMINAMATH_GPT_task1_task2_task3_l2349_234992

noncomputable def f (x a : ℝ) := x^2 - 4 * x + a + 3
noncomputable def g (x m : ℝ) := m * x + 5 - 2 * m

theorem task1 (a m : ℝ) (h₁ : a = -3) (h₂ : m = 0) :
  (∃ x : ℝ, f x a - g x m = 0) ↔ x = -1 ∨ x = 5 :=
sorry

theorem task2 (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem task3 (m : ℝ) :
  (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 → ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ 0 = g x₂ m) ↔ m ≤ -3 ∨ 6 ≤ m :=
sorry

end NUMINAMATH_GPT_task1_task2_task3_l2349_234992


namespace NUMINAMATH_GPT_adam_simon_distance_100_l2349_234980

noncomputable def time_to_be_100_apart (x : ℝ) : Prop :=
  let distance_adam := 10 * x
  let distance_simon_east := 10 * x * (Real.sqrt 2 / 2)
  let distance_simon_south := 10 * x * (Real.sqrt 2 / 2)
  let total_eastward_separation := abs (distance_adam - distance_simon_east)
  let resultant_distance := Real.sqrt (total_eastward_separation^2 + distance_simon_south^2)
  resultant_distance = 100

theorem adam_simon_distance_100 : ∃ (x : ℝ), time_to_be_100_apart x ∧ x = 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_adam_simon_distance_100_l2349_234980


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l2349_234957

theorem solve_eq1 : (2 * (x - 3) = 3 * x * (x - 3)) → (x = 3 ∨ x = 2 / 3) :=
by
  intro h
  sorry

theorem solve_eq2 : (2 * x ^ 2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l2349_234957


namespace NUMINAMATH_GPT_distance_proof_l2349_234908

-- Definitions from the conditions
def avg_speed_to_retreat := 50
def avg_speed_back_home := 75
def total_round_trip_time := 10
def distance_between_home_and_retreat := 300

-- Theorem stating the problem
theorem distance_proof 
  (D : ℝ)
  (h1 : D / avg_speed_to_retreat + D / avg_speed_back_home = total_round_trip_time) :
  D = distance_between_home_and_retreat :=
sorry

end NUMINAMATH_GPT_distance_proof_l2349_234908


namespace NUMINAMATH_GPT_center_of_circle_is_correct_l2349_234904

-- Define the conditions as Lean functions and statements
def is_tangent (x y : ℝ) : Prop :=
  (3 * x + 4 * y = 48) ∨ (3 * x + 4 * y = -12)

def is_on_line (x y : ℝ) : Prop := x = y

-- Define the proof statement
theorem center_of_circle_is_correct (x y : ℝ) (h1 : is_tangent x y) (h2 : is_on_line x y) :
  (x, y) = (18 / 7, 18 / 7) :=
sorry

end NUMINAMATH_GPT_center_of_circle_is_correct_l2349_234904


namespace NUMINAMATH_GPT_weight_of_A_l2349_234933

theorem weight_of_A
  (W_A W_B W_C W_D W_E : ℕ)
  (H_A H_B H_C H_D : ℕ)
  (Age_A Age_B Age_C Age_D : ℕ)
  (hw1 : (W_A + W_B + W_C) / 3 = 84)
  (hh1 : (H_A + H_B + H_C) / 3 = 170)
  (ha1 : (Age_A + Age_B + Age_C) / 3 = 30)
  (hw2 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (hh2 : (H_A + H_B + H_C + H_D) / 4 = 172)
  (ha2 : (Age_A + Age_B + Age_C + Age_D) / 4 = 28)
  (hw3 : (W_B + W_C + W_D + W_E) / 4 = 79)
  (hh3 : (H_B + H_C + H_D + H_E) / 4 = 173)
  (ha3 : (Age_B + Age_C + Age_D + (Age_A - 3)) / 4 = 27)
  (hw4 : W_E = W_D + 7)
  : W_A = 79 := 
sorry

end NUMINAMATH_GPT_weight_of_A_l2349_234933


namespace NUMINAMATH_GPT_inequality_holds_l2349_234939

variable (a b c d : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)
variable (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2)

theorem inequality_holds (ha : a > 0)
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2349_234939


namespace NUMINAMATH_GPT_geometric_seq_comparison_l2349_234975

def geometric_seq_positive (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n+1) = a n * q

theorem geometric_seq_comparison (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_seq_positive a q) (h2 : q ≠ 1) (h3 : ∀ n, a n > 0) (h4 : q > 0) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end NUMINAMATH_GPT_geometric_seq_comparison_l2349_234975


namespace NUMINAMATH_GPT_original_average_l2349_234987

theorem original_average (n : ℕ) (k : ℕ) (new_avg : ℝ) 
  (h1 : n = 35) 
  (h2 : k = 5) 
  (h3 : new_avg = 125) : 
  (new_avg / k) = 25 :=
by
  rw [h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_original_average_l2349_234987


namespace NUMINAMATH_GPT_sum_of_cubes_l2349_234938

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l2349_234938


namespace NUMINAMATH_GPT_angle_B_eq_pi_div_3_l2349_234976

variables {A B C : ℝ} {a b c : ℝ}

/-- Given an acute triangle ABC, where sides a, b, c are opposite the angles A, B, and C respectively, 
    and given the condition b cos C + sqrt 3 * b sin C = a + c, prove that B = π / 3. -/
theorem angle_B_eq_pi_div_3 
  (h : ∀ (A B C : ℝ), 0 < A ∧ A < π / 2  ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (cond : b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c) :
  B = π / 3 := 
sorry

end NUMINAMATH_GPT_angle_B_eq_pi_div_3_l2349_234976


namespace NUMINAMATH_GPT_total_price_of_books_l2349_234965

theorem total_price_of_books
  (total_books : ℕ)
  (math_books_cost : ℕ)
  (history_books_cost : ℕ)
  (math_books_bought : ℕ)
  (total_books_eq : total_books = 80)
  (math_books_cost_eq : math_books_cost = 4)
  (history_books_cost_eq : history_books_cost = 5)
  (math_books_bought_eq : math_books_bought = 10) :
  (math_books_bought * math_books_cost + (total_books - math_books_bought) * history_books_cost = 390) := 
by
  sorry

end NUMINAMATH_GPT_total_price_of_books_l2349_234965


namespace NUMINAMATH_GPT_gathering_people_total_l2349_234956

theorem gathering_people_total (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 :=
by
  sorry

end NUMINAMATH_GPT_gathering_people_total_l2349_234956


namespace NUMINAMATH_GPT_trees_in_garden_l2349_234959

theorem trees_in_garden (yard_length : ℕ) (distance_between_trees : ℕ) (H1 : yard_length = 400) (H2 : distance_between_trees = 16) : 
  (yard_length / distance_between_trees) + 1 = 26 :=
by
  -- Adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_trees_in_garden_l2349_234959


namespace NUMINAMATH_GPT_no_sum_of_two_squares_l2349_234909

theorem no_sum_of_two_squares (n : ℤ) (h : n % 4 = 3) : ¬∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end NUMINAMATH_GPT_no_sum_of_two_squares_l2349_234909


namespace NUMINAMATH_GPT_convert_to_base10_sum_l2349_234926

def base8_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def base13_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 13^2 + d1 * 13^1 + d0 * 13^0

def convert_537_8 : Nat :=
  base8_to_dec 5 3 7

def convert_4C5_13 : Nat :=
  base13_to_dec 4 12 5

theorem convert_to_base10_sum : 
  convert_537_8 + convert_4C5_13 = 1188 := 
by 
  sorry

end NUMINAMATH_GPT_convert_to_base10_sum_l2349_234926


namespace NUMINAMATH_GPT_value_of_m_l2349_234967

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x + 1 else 2 * (-x) + 1

theorem value_of_m (m : ℝ) (heven : ∀ x : ℝ, f (-x) = f x)
  (hpos : ∀ x : ℝ, x ≥ 0 → f x = 2 * x + 1)
  (hfm : f m = 5) : m = 2 ∨ m = -2 :=
sorry

end NUMINAMATH_GPT_value_of_m_l2349_234967


namespace NUMINAMATH_GPT_quadratic_root_a_value_l2349_234963

theorem quadratic_root_a_value (a : ℝ) :
  (∃ x : ℝ, x = -2 ∧ x^2 + (3 / 2) * a * x - a^2 = 0) → (a = 1 ∨ a = -4) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_root_a_value_l2349_234963


namespace NUMINAMATH_GPT_find_b_l2349_234905

theorem find_b (b : ℝ) (y : ℝ) : (4 * 3 + 2 * y = b) ∧ (3 * 3 + 6 * y = 3 * b) → b = 27 :=
by
sorry

end NUMINAMATH_GPT_find_b_l2349_234905


namespace NUMINAMATH_GPT_num_of_3_digit_nums_with_one_even_digit_l2349_234998

def is_even (n : Nat) : Bool :=
  n % 2 == 0

def count_3_digit_nums_with_exactly_one_even_digit : Nat :=
  let even_digits := [0, 2, 4, 6, 8]
  let odd_digits := [1, 3, 5, 7, 9]
  -- Case 1: A is even, B and C are odd
  let case1 := 4 * 5 * 5
  -- Case 2: B is even, A and C are odd
  let case2 := 5 * 5 * 5
  -- Case 3: C is even, A and B are odd
  let case3 := 5 * 5 * 5
  case1 + case2 + case3

theorem num_of_3_digit_nums_with_one_even_digit : count_3_digit_nums_with_exactly_one_even_digit = 350 := by
  sorry

end NUMINAMATH_GPT_num_of_3_digit_nums_with_one_even_digit_l2349_234998


namespace NUMINAMATH_GPT_tan_half_sum_pi_over_four_l2349_234993

-- Define the problem conditions
variable (α : ℝ)
variable (h_cos : Real.cos α = -4 / 5)
variable (h_quad : α > π ∧ α < 3 * π / 2)

-- Define the theorem to prove
theorem tan_half_sum_pi_over_four (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_quad : α > π ∧ α < 3 * π / 2) :
  Real.tan (π / 4 + α / 2) = -1 / 2 := sorry

end NUMINAMATH_GPT_tan_half_sum_pi_over_four_l2349_234993


namespace NUMINAMATH_GPT_timothy_tea_cups_l2349_234932

theorem timothy_tea_cups (t : ℕ) (h : 6 * t + 60 = 120) : t + 12 = 22 :=
by
  sorry

end NUMINAMATH_GPT_timothy_tea_cups_l2349_234932


namespace NUMINAMATH_GPT_summation_indices_equal_l2349_234930

theorem summation_indices_equal
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i ≤ 100)
  (h_length : ∀ i, i < 16) :
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l := 
by {
  sorry
}

end NUMINAMATH_GPT_summation_indices_equal_l2349_234930


namespace NUMINAMATH_GPT_remove_terms_to_make_sum_l2349_234946

theorem remove_terms_to_make_sum (a b c d e f : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/5) (h₃ : c = 1/7) (h₄ : d = 1/9) (h₅ : e = 1/11) (h₆ : f = 1/13) :
  a + b + c + d + e + f - e - f = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_remove_terms_to_make_sum_l2349_234946


namespace NUMINAMATH_GPT_ethanol_concentration_l2349_234983

theorem ethanol_concentration
  (w1 : ℕ) (c1 : ℝ) (w2 : ℕ) (c2 : ℝ)
  (hw1 : w1 = 400) (hc1 : c1 = 0.30)
  (hw2 : w2 = 600) (hc2 : c2 = 0.80) :
  (c1 * w1 + c2 * w2) / (w1 + w2) = 0.60 := 
by
  sorry

end NUMINAMATH_GPT_ethanol_concentration_l2349_234983


namespace NUMINAMATH_GPT_gcd_128_144_256_l2349_234917

theorem gcd_128_144_256 : Nat.gcd (Nat.gcd 128 144) 256 = 128 :=
  sorry

end NUMINAMATH_GPT_gcd_128_144_256_l2349_234917


namespace NUMINAMATH_GPT_robot_possible_path_lengths_l2349_234924

theorem robot_possible_path_lengths (n : ℕ) (valid_path: ∀ (i : ℕ), i < n → (i % 4 = 0 ∨ i % 4 = 1 ∨ i % 4 = 2 ∨ i % 4 = 3)) :
  (n % 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_robot_possible_path_lengths_l2349_234924
