import Mathlib

namespace NUMINAMATH_GPT_min_additional_trains_needed_l985_98511

-- Definitions
def current_trains : ℕ := 31
def trains_per_row : ℕ := 8
def smallest_num_additional_trains (current : ℕ) (per_row : ℕ) : ℕ :=
  let next_multiple := ((current + per_row - 1) / per_row) * per_row
  next_multiple - current

-- Theorem
theorem min_additional_trains_needed :
  smallest_num_additional_trains current_trains trains_per_row = 1 :=
by
  sorry

end NUMINAMATH_GPT_min_additional_trains_needed_l985_98511


namespace NUMINAMATH_GPT_prob1_prob2_l985_98538

-- Problem 1
theorem prob1 (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4 :=
sorry

-- Problem 2
theorem prob2 (x y : ℝ) : (5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2 :=
sorry

end NUMINAMATH_GPT_prob1_prob2_l985_98538


namespace NUMINAMATH_GPT_probability_of_event_B_l985_98570

def fair_dice := { n : ℕ // 1 ≤ n ∧ n ≤ 8 }

def event_B (x y : fair_dice) : Prop := x.val = y.val + 2

def total_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 6

theorem probability_of_event_B : (favorable_outcomes : ℚ) / total_outcomes = 3/32 := by
  have h1 : (64 : ℚ) = 8 * 8 := by norm_num
  have h2 : (6 : ℚ) / 64 = 3 / 32 := by norm_num
  sorry

end NUMINAMATH_GPT_probability_of_event_B_l985_98570


namespace NUMINAMATH_GPT_winner_percentage_l985_98537

variable (votes_winner : ℕ) (win_by : ℕ)
variable (total_votes : ℕ)
variable (percentage_winner : ℕ)

-- Conditions
def conditions : Prop :=
  votes_winner = 930 ∧
  win_by = 360 ∧
  total_votes = votes_winner + (votes_winner - win_by) ∧
  percentage_winner = (votes_winner * 100) / total_votes

-- Theorem to prove
theorem winner_percentage (h : conditions votes_winner win_by total_votes percentage_winner) : percentage_winner = 62 :=
sorry

end NUMINAMATH_GPT_winner_percentage_l985_98537


namespace NUMINAMATH_GPT_sum_of_cubes_l985_98500

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = 5) : a^3 + b^3 + c^3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l985_98500


namespace NUMINAMATH_GPT_ned_initial_video_games_l985_98591

theorem ned_initial_video_games : ∀ (w t : ℕ), 7 * w = 63 ∧ t = w + 6 → t = 15 := by
  intro w t
  intro h
  sorry

end NUMINAMATH_GPT_ned_initial_video_games_l985_98591


namespace NUMINAMATH_GPT_transformed_mean_stddev_l985_98509

variables (n : ℕ) (x : Fin n → ℝ)

-- Given conditions
def mean_is_4 (mean : ℝ) : Prop :=
  mean = 4

def stddev_is_7 (stddev : ℝ) : Prop :=
  stddev = 7

-- Definitions for transformations and the results
def transformed_mean (mean : ℝ) : ℝ :=
  3 * mean + 2

def transformed_stddev (stddev : ℝ) : ℝ :=
  3 * stddev

-- The proof problem
theorem transformed_mean_stddev (mean stddev : ℝ) 
  (h_mean : mean_is_4 mean) 
  (h_stddev : stddev_is_7 stddev) :
  transformed_mean mean = 14 ∧ transformed_stddev stddev = 21 :=
by
  rw [h_mean, h_stddev]
  unfold transformed_mean transformed_stddev
  rw [← h_mean, ← h_stddev]
  sorry

end NUMINAMATH_GPT_transformed_mean_stddev_l985_98509


namespace NUMINAMATH_GPT_cost_of_each_soda_l985_98555

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_soda_l985_98555


namespace NUMINAMATH_GPT_faces_of_prism_with_24_edges_l985_98558

theorem faces_of_prism_with_24_edges (L : ℕ) (h1 : 3 * L = 24) : L + 2 = 10 := by
  sorry

end NUMINAMATH_GPT_faces_of_prism_with_24_edges_l985_98558


namespace NUMINAMATH_GPT_solve_rational_numbers_l985_98579

theorem solve_rational_numbers 
  (a b c d : ℚ)
  (h₁ : a + b + c = -1)
  (h₂ : a + b + d = -3)
  (h₃ : a + c + d = 2)
  (h₄ : b + c + d = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := 
by
  sorry

end NUMINAMATH_GPT_solve_rational_numbers_l985_98579


namespace NUMINAMATH_GPT_solve_quadratic_l985_98573

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_l985_98573


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l985_98539

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a ≠ 0) :
  (a > 2 ↔ |a - 1| > 1) ↔ (a > 2 → |a - 1| > 1) ∧ (a < 0 → |a - 1| > 1) ∧ (∃ x : ℝ, (|x - 1| > 1) ∧ x < 0 ∧ x ≠ a) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l985_98539


namespace NUMINAMATH_GPT_customers_at_start_l985_98517

def initial_customers (X : ℕ) : Prop :=
  let first_hour := X + 3
  let second_hour := first_hour - 6
  second_hour = 12

theorem customers_at_start {X : ℕ} : initial_customers X → X = 15 :=
by
  sorry

end NUMINAMATH_GPT_customers_at_start_l985_98517


namespace NUMINAMATH_GPT_valid_votes_l985_98510

theorem valid_votes (V : ℝ) 
  (h1 : 0.70 * V - 0.30 * V = 176): V = 440 :=
  sorry

end NUMINAMATH_GPT_valid_votes_l985_98510


namespace NUMINAMATH_GPT_arithmetic_mean_multiplied_correct_l985_98593

-- Define the fractions involved
def frac1 : ℚ := 3 / 4
def frac2 : ℚ := 5 / 8

-- Define the arithmetic mean and the final multiplication result
def mean_and_multiply_result : ℚ := ( (frac1 + frac2) / 2 ) * 3

-- Statement to prove that the calculated result is equal to 33/16
theorem arithmetic_mean_multiplied_correct : mean_and_multiply_result = 33 / 16 := 
by 
  -- Skipping the proof with sorry for the statement only requirement
  sorry

end NUMINAMATH_GPT_arithmetic_mean_multiplied_correct_l985_98593


namespace NUMINAMATH_GPT_parallelogram_base_length_l985_98569

theorem parallelogram_base_length
  (height : ℝ) (area : ℝ) (base : ℝ) 
  (h1 : height = 18) 
  (h2 : area = 576) 
  (h3 : area = base * height) : 
  base = 32 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l985_98569


namespace NUMINAMATH_GPT_length_of_boat_l985_98556

-- Define Josie's jogging variables and problem conditions
variables (L J B : ℝ)
axiom eqn1 : 130 * J = L + 130 * B
axiom eqn2 : 70 * J = L - 70 * B

-- The theorem to prove that the length of the boat L equals 91 steps (i.e., 91 * J)
theorem length_of_boat : L = 91 * J :=
by
  sorry

end NUMINAMATH_GPT_length_of_boat_l985_98556


namespace NUMINAMATH_GPT_total_employees_l985_98544

-- Definitions based on the conditions:
variables (N S : ℕ)
axiom condition1 : 75 % 100 * S = 75 / 100 * S
axiom condition2 : 65 % 100 * S = 65 / 100 * S
axiom condition3 : N - S = 40
axiom condition4 : 5 % 6 * N = 5 / 6 * N

-- The statement to be proven:
theorem total_employees (N S : ℕ)
    (h1 : 75 % 100 * S = 75 / 100 * S)
    (h2 : 65 % 100 * S = 65 / 100 * S)
    (h3 : N - S = 40)
    (h4 : 5 % 6 * N = 5 / 6 * N)
    : N = 240 :=
sorry

end NUMINAMATH_GPT_total_employees_l985_98544


namespace NUMINAMATH_GPT_chef_cooked_additional_wings_l985_98514

def total_chicken_wings_needed (friends : ℕ) (wings_per_friend : ℕ) : ℕ :=
  friends * wings_per_friend

def additional_chicken_wings (total_needed : ℕ) (already_cooked : ℕ) : ℕ :=
  total_needed - already_cooked

theorem chef_cooked_additional_wings :
  let friends := 4
  let wings_per_friend := 4
  let already_cooked := 9
  additional_chicken_wings (total_chicken_wings_needed friends wings_per_friend) already_cooked = 7 := by
  sorry

end NUMINAMATH_GPT_chef_cooked_additional_wings_l985_98514


namespace NUMINAMATH_GPT_countTwoLeggedBirds_l985_98504

def countAnimals (x y : ℕ) : Prop :=
  x + y = 200 ∧ 2 * x + 4 * y = 522

theorem countTwoLeggedBirds (x y : ℕ) (h : countAnimals x y) : x = 139 :=
by
  sorry

end NUMINAMATH_GPT_countTwoLeggedBirds_l985_98504


namespace NUMINAMATH_GPT_expression_evaluates_at_1_l985_98521

variable (x : ℚ)

def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

def substituted_expr (x : ℚ) : ℚ :=
  (original_expr (original_expr x) + 2) / (original_expr (original_expr x) - 3)

theorem expression_evaluates_at_1 :
  substituted_expr 1 = -1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluates_at_1_l985_98521


namespace NUMINAMATH_GPT_car_dealer_bmw_sales_l985_98545

theorem car_dealer_bmw_sales (total_cars : ℕ)
  (vw_percentage : ℝ)
  (toyota_percentage : ℝ)
  (acura_percentage : ℝ)
  (bmw_count : ℕ) :
  total_cars = 300 →
  vw_percentage = 0.10 →
  toyota_percentage = 0.25 →
  acura_percentage = 0.20 →
  bmw_count = total_cars * (1 - (vw_percentage + toyota_percentage + acura_percentage)) →
  bmw_count = 135 :=
by
  intros
  sorry

end NUMINAMATH_GPT_car_dealer_bmw_sales_l985_98545


namespace NUMINAMATH_GPT_num_distinct_values_for_sum_l985_98599

theorem num_distinct_values_for_sum (x y z : ℝ) 
  (h : (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0) :
  ∃ s : Finset ℝ, 
  (∀ x y z, (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0 → (x + y + z) ∈ s) ∧ 
  s.card = 7 :=
by sorry

end NUMINAMATH_GPT_num_distinct_values_for_sum_l985_98599


namespace NUMINAMATH_GPT_Bruce_paid_l985_98589

noncomputable def total_paid : ℝ :=
  let grapes_price := 9 * 70 * (1 - 0.10)
  let mangoes_price := 7 * 55 * (1 - 0.05)
  let oranges_price := 5 * 45 * (1 - 0.15)
  let apples_price := 3 * 80 * (1 - 0.20)
  grapes_price + mangoes_price + oranges_price + apples_price

theorem Bruce_paid (h : total_paid = 1316.25) : true :=
by
  -- This is where the proof would be
  sorry

end NUMINAMATH_GPT_Bruce_paid_l985_98589


namespace NUMINAMATH_GPT_cost_per_remaining_ticket_is_seven_l985_98574

def total_tickets : ℕ := 29
def nine_dollar_tickets : ℕ := 11
def total_cost : ℕ := 225
def nine_dollar_ticket_cost : ℕ := 9
def remaining_tickets : ℕ := total_tickets - nine_dollar_tickets

theorem cost_per_remaining_ticket_is_seven :
  (total_cost - nine_dollar_tickets * nine_dollar_ticket_cost) / remaining_tickets = 7 :=
  sorry

end NUMINAMATH_GPT_cost_per_remaining_ticket_is_seven_l985_98574


namespace NUMINAMATH_GPT_find_length_CD_m_plus_n_l985_98592

noncomputable def lengthAB : ℝ := 7
noncomputable def lengthBD : ℝ := 11
noncomputable def lengthBC : ℝ := 9

axiom angle_BAD_ADC : Prop
axiom angle_ABD_BCD : Prop

theorem find_length_CD_m_plus_n :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (CD = m / n) ∧ (m + n = 67) :=
sorry  -- Proof would be provided here

end NUMINAMATH_GPT_find_length_CD_m_plus_n_l985_98592


namespace NUMINAMATH_GPT_fraction_value_l985_98559

theorem fraction_value (x : ℝ) (h : 1 - 5 / x + 6 / x^3 = 0) : 3 / x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l985_98559


namespace NUMINAMATH_GPT_number_of_teachers_l985_98518

-- Definitions from the problem conditions
def num_students : Nat := 1500
def classes_per_student : Nat := 6
def classes_per_teacher : Nat := 5
def students_per_class : Nat := 25

-- The proof problem statement
theorem number_of_teachers : 
  (num_students * classes_per_student / students_per_class) / classes_per_teacher = 72 := by
  sorry

end NUMINAMATH_GPT_number_of_teachers_l985_98518


namespace NUMINAMATH_GPT_triangle_isosceles_or_right_l985_98581

theorem triangle_isosceles_or_right (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_side_constraint : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_condition: a^2 * c^2 - b^2 * c^2 = a^4 - b^4) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_isosceles_or_right_l985_98581


namespace NUMINAMATH_GPT_laptop_price_l985_98519

theorem laptop_price (upfront_percent : ℝ) (upfront_payment full_price : ℝ)
  (h1 : upfront_percent = 0.20)
  (h2 : upfront_payment = 240)
  (h3 : upfront_payment = upfront_percent * full_price) :
  full_price = 1200 := 
sorry

end NUMINAMATH_GPT_laptop_price_l985_98519


namespace NUMINAMATH_GPT_a_pow_5_mod_11_l985_98585

theorem a_pow_5_mod_11 (a : ℕ) : (a^5) % 11 = 0 ∨ (a^5) % 11 = 1 ∨ (a^5) % 11 = 10 :=
sorry

end NUMINAMATH_GPT_a_pow_5_mod_11_l985_98585


namespace NUMINAMATH_GPT_abe_age_equation_l985_98572

theorem abe_age_equation (a : ℕ) (x : ℕ) (h1 : a = 19) (h2 : a + (a - x) = 31) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_abe_age_equation_l985_98572


namespace NUMINAMATH_GPT_triangle_right_angle_solution_l985_98587

def is_right_angle (a b : ℝ × ℝ) : Prop := (a.1 * b.1 + a.2 * b.2 = 0)

def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem triangle_right_angle_solution (x : ℝ) (h1 : (2, -1) = (2, -1)) (h2 : (x, 3) = (x, 3)) : 
  is_right_angle (2, -1) (x, 3) ∨ 
  is_right_angle (2, -1) (vector_sub (x, 3) (2, -1)) ∨ 
  is_right_angle (x, 3) (vector_sub (x, 3) (2, -1)) → 
  x = 3 / 2 ∨ x = 4 :=
sorry

end NUMINAMATH_GPT_triangle_right_angle_solution_l985_98587


namespace NUMINAMATH_GPT_biscuits_initial_l985_98577

theorem biscuits_initial (F M A L B : ℕ) 
  (father_gave : F = 13) 
  (mother_gave : M = 15) 
  (brother_ate : A = 20) 
  (left_with : L = 40) 
  (remaining : B + F + M - A = L) :
  B = 32 := 
by 
  subst father_gave
  subst mother_gave
  subst brother_ate
  subst left_with
  simp at remaining
  linarith

end NUMINAMATH_GPT_biscuits_initial_l985_98577


namespace NUMINAMATH_GPT_find_number_l985_98546

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 :=
sorry

end NUMINAMATH_GPT_find_number_l985_98546


namespace NUMINAMATH_GPT_calculate_surface_area_of_modified_cube_l985_98524

-- Definitions of the conditions
def edge_length_of_cube : ℕ := 5
def side_length_of_hole : ℕ := 2

-- The main theorem statement to be proven
theorem calculate_surface_area_of_modified_cube :
  let original_surface_area := 6 * (edge_length_of_cube * edge_length_of_cube)
  let area_removed_by_holes := 6 * (side_length_of_hole * side_length_of_hole)
  let area_exposed_by_holes := 6 * 6 * (side_length_of_hole * side_length_of_hole)
  original_surface_area - area_removed_by_holes + area_exposed_by_holes = 270 :=
by
  sorry

end NUMINAMATH_GPT_calculate_surface_area_of_modified_cube_l985_98524


namespace NUMINAMATH_GPT_students_above_90_l985_98525

theorem students_above_90 (total_students : ℕ) (above_90_chinese : ℕ) (above_90_math : ℕ)
  (all_above_90_at_least_one_subject : total_students = 50 ∧ above_90_chinese = 33 ∧ above_90_math = 38 ∧ 
    ∀ (n : ℕ), n < total_students → (n < above_90_chinese ∨ n < above_90_math)) :
  (above_90_chinese + above_90_math - total_students) = 21 :=
by
  sorry

end NUMINAMATH_GPT_students_above_90_l985_98525


namespace NUMINAMATH_GPT_solve_system_of_eq_l985_98566

noncomputable def system_of_eq (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2 * x) ∧
  z = y^3 * (3 - 2 * y) ∧
  x = z^3 * (3 - 2 * z)

theorem solve_system_of_eq (x y z : ℝ) :
  system_of_eq x y z →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end NUMINAMATH_GPT_solve_system_of_eq_l985_98566


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l985_98580

theorem isosceles_triangle_sides (length_rope : ℝ) (one_side : ℝ) (a b : ℝ) :
  length_rope = 18 ∧ one_side = 5 ∧ a + a + one_side = length_rope ∧ b = one_side ∨ b + b + one_side = length_rope -> (a = 6.5 ∨ a = 5) ∧ (b = 6.5 ∨ b = 5) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l985_98580


namespace NUMINAMATH_GPT_NoahMealsCount_l985_98528

-- Definition of all the choices available to Noah
def MainCourses := ["Pizza", "Burger", "Pasta"]
def Beverages := ["Soda", "Juice"]
def Snacks := ["Apple", "Banana", "Cookie"]

-- Condition that Noah avoids soda with pizza
def isValidMeal (main : String) (beverage : String) : Bool :=
  not (main = "Pizza" ∧ beverage = "Soda")

-- Total number of valid meal combinations
def totalValidMeals : Nat :=
  (if isValidMeal "Pizza" "Juice" then 1 else 0) * Snacks.length +
  (Beverages.length - 1) * Snacks.length * (MainCourses.length - 1) + -- for Pizza
  Beverages.length * Snacks.length * 2 -- for Burger and Pasta

-- The theorem that Noah can buy 15 distinct meals
theorem NoahMealsCount : totalValidMeals = 15 := by
  sorry

end NUMINAMATH_GPT_NoahMealsCount_l985_98528


namespace NUMINAMATH_GPT_parabola_focus_l985_98584

theorem parabola_focus (h : ∀ x y : ℝ, y ^ 2 = -12 * x → True) : (-3, 0) = (-3, 0) :=
  sorry

end NUMINAMATH_GPT_parabola_focus_l985_98584


namespace NUMINAMATH_GPT_inscribed_circles_radii_sum_l985_98541

noncomputable def sum_of_radii (d : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 + r2 = d / 2

theorem inscribed_circles_radii_sum (d : ℝ) (h : d = 23) (r1 r2 : ℝ) (h1 : r1 + r2 = d / 2) :
  r1 + r2 = 23 / 2 :=
by
  rw [h] at h1
  exact h1

end NUMINAMATH_GPT_inscribed_circles_radii_sum_l985_98541


namespace NUMINAMATH_GPT_halt_duration_l985_98549

theorem halt_duration (avg_speed : ℝ) (distance : ℝ) (start_time end_time : ℝ) (halt_duration : ℝ) :
  avg_speed = 87 ∧ distance = 348 ∧ start_time = 9 ∧ end_time = 13.75 →
  halt_duration = (end_time - start_time) - (distance / avg_speed) → 
  halt_duration = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_halt_duration_l985_98549


namespace NUMINAMATH_GPT_evaluate_polynomial_at_three_l985_98576

def polynomial (x : ℕ) : ℕ :=
  x^6 + 2 * x^5 + 4 * x^3 + 5 * x^2 + 6 * x + 12

theorem evaluate_polynomial_at_three :
  polynomial 3 = 588 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_three_l985_98576


namespace NUMINAMATH_GPT_Ned_earning_money_l985_98527

def total_games : Nat := 15
def non_working_games : Nat := 6
def price_per_game : Nat := 7
def working_games : Nat := total_games - non_working_games
def total_money : Nat := working_games * price_per_game

theorem Ned_earning_money : total_money = 63 := by
  sorry

end NUMINAMATH_GPT_Ned_earning_money_l985_98527


namespace NUMINAMATH_GPT_ratio_equivalence_l985_98512

theorem ratio_equivalence (x : ℕ) (h1 : 3 / 12 = x / 16) : x = 4 :=
by sorry

end NUMINAMATH_GPT_ratio_equivalence_l985_98512


namespace NUMINAMATH_GPT_original_amount_of_water_l985_98534

variable {W : ℝ} -- Assume W is a real number representing the original amount of water

theorem original_amount_of_water (h1 : 30 * 0.02 = 0.6) (h2 : 0.6 = 0.06 * W) : W = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_amount_of_water_l985_98534


namespace NUMINAMATH_GPT_zero_in_interval_l985_98586

theorem zero_in_interval {b : ℝ} (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 2 * b * x - 3 * b + 1)
  (h₂ : b > 1/5)
  (h₃ : b < 1) :
  ∃ x, -1 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_in_interval_l985_98586


namespace NUMINAMATH_GPT_simplified_sum_l985_98548

theorem simplified_sum :
  (-2^2003) + (2^2004) + (-2^2005) - (2^2006) = 5 * (2^2003) :=
by
  sorry

end NUMINAMATH_GPT_simplified_sum_l985_98548


namespace NUMINAMATH_GPT_cottonwood_fiber_scientific_notation_l985_98540

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end NUMINAMATH_GPT_cottonwood_fiber_scientific_notation_l985_98540


namespace NUMINAMATH_GPT_slope_angle_135_l985_98516

theorem slope_angle_135 (x y : ℝ) : 
  (∃ (m b : ℝ), 3 * x + 3 * y + 1 = 0 ∧ y = m * x + b ∧ m = -1) ↔ 
  (∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ Real.tan α = -1 ∧ α = 135) :=
sorry

end NUMINAMATH_GPT_slope_angle_135_l985_98516


namespace NUMINAMATH_GPT_quadratic_expression_value_l985_98550

variables (α β : ℝ)
noncomputable def quadratic_root_sum (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) : Prop :=
  α + β = -2

theorem quadratic_expression_value (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) (h3 : α + β = -2) :
  α^2 + 3*α + β = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l985_98550


namespace NUMINAMATH_GPT_inverse_g_neg138_l985_98560

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_g_neg138 :
  g (-3) = -138 :=
by
  sorry

end NUMINAMATH_GPT_inverse_g_neg138_l985_98560


namespace NUMINAMATH_GPT_octagon_edge_length_from_pentagon_l985_98598

noncomputable def regular_pentagon_edge_length : ℝ := 16
def num_of_pentagon_edges : ℕ := 5
def num_of_octagon_edges : ℕ := 8

theorem octagon_edge_length_from_pentagon (total_length_thread : ℝ) :
  total_length_thread = num_of_pentagon_edges * regular_pentagon_edge_length →
  (total_length_thread / num_of_octagon_edges) = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_octagon_edge_length_from_pentagon_l985_98598


namespace NUMINAMATH_GPT_problem_statement_l985_98554

variables {Line Plane : Type}

-- Defining the perpendicular relationship between a line and a plane
def perp (a : Line) (α : Plane) : Prop := sorry

-- Defining the parallel relationship between two planes
def para (α β : Plane) : Prop := sorry

-- The main statement to prove
theorem problem_statement (a : Line) (α β : Plane) (h1 : perp a α) (h2 : perp a β) : para α β := 
sorry

end NUMINAMATH_GPT_problem_statement_l985_98554


namespace NUMINAMATH_GPT_algebraic_expression_evaluates_to_2_l985_98529

theorem algebraic_expression_evaluates_to_2 (x : ℝ) (h : x^2 + x - 5 = 0) : 
(x - 1)^2 - x * (x - 3) + (x + 2) * (x - 2) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluates_to_2_l985_98529


namespace NUMINAMATH_GPT_Anne_weight_l985_98523

-- Define variables
def Douglas_weight : ℕ := 52
def weight_difference : ℕ := 15

-- Theorem to prove
theorem Anne_weight : Douglas_weight + weight_difference = 67 :=
by sorry

end NUMINAMATH_GPT_Anne_weight_l985_98523


namespace NUMINAMATH_GPT_geometric_series_sum_l985_98553

theorem geometric_series_sum :
  let a := -2
  let r := 4
  let n := 10
  let S := (a * (r^n - 1)) / (r - 1)
  S = -699050 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l985_98553


namespace NUMINAMATH_GPT_bromine_atoms_in_compound_l985_98543

theorem bromine_atoms_in_compound
  (atomic_weight_H : ℕ := 1)
  (atomic_weight_Br : ℕ := 80)
  (atomic_weight_O : ℕ := 16)
  (total_molecular_weight : ℕ := 129) :
  ∃ (n : ℕ), total_molecular_weight = atomic_weight_H + n * atomic_weight_Br + 3 * atomic_weight_O ∧ n = 1 := 
by
  sorry

end NUMINAMATH_GPT_bromine_atoms_in_compound_l985_98543


namespace NUMINAMATH_GPT_pumps_fill_time_l985_98561

-- Definitions for the rates and the time calculation
def small_pump_rate : ℚ := 1 / 3
def large_pump_rate : ℚ := 4
def third_pump_rate : ℚ := 1 / 2

def total_pump_rate : ℚ := small_pump_rate + large_pump_rate + third_pump_rate

theorem pumps_fill_time :
  1 / total_pump_rate = 6 / 29 :=
by
  -- Definition of the rates has already been given.
  -- Here we specify the calculation for the combined rate and filling time.
  sorry

end NUMINAMATH_GPT_pumps_fill_time_l985_98561


namespace NUMINAMATH_GPT_power_identity_l985_98575

theorem power_identity (x y a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) : 10^(3*x + 2*y) = a^3 * b^2 := 
by 
  sorry

end NUMINAMATH_GPT_power_identity_l985_98575


namespace NUMINAMATH_GPT_relatively_prime_subsequence_exists_l985_98594

theorem relatively_prime_subsequence_exists :
  ∃ (s : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → Nat.gcd (2^(s i) - 3) (2^(s j) - 3) = 1) :=
by
  sorry

end NUMINAMATH_GPT_relatively_prime_subsequence_exists_l985_98594


namespace NUMINAMATH_GPT_watch_arrangement_count_l985_98547

noncomputable def number_of_satisfying_watch_arrangements : Nat :=
  let dial_arrangements := Nat.factorial 2
  let strap_arrangements := Nat.factorial 3
  dial_arrangements * strap_arrangements

theorem watch_arrangement_count :
  number_of_satisfying_watch_arrangements = 12 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_watch_arrangement_count_l985_98547


namespace NUMINAMATH_GPT_complex_number_expression_l985_98530

noncomputable def compute_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1)

theorem complex_number_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  compute_expression r h1 h2 = 5 :=
sorry

end NUMINAMATH_GPT_complex_number_expression_l985_98530


namespace NUMINAMATH_GPT_calc3aMinus4b_l985_98588

theorem calc3aMinus4b (a b : ℤ) (h1 : a * 1 - b * 2 = -1) (h2 : a * 1 + b * 2 = 7) : 3 * a - 4 * b = 1 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_calc3aMinus4b_l985_98588


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l985_98557

variable {α : Type} [LinearOrderedField α]

noncomputable def a_n (a1 d n : α) := a1 + (n - 1) * d

theorem arithmetic_sequence_sum (a1 d : α) (h1 : a_n a1 d 3 * a_n a1 d 11 = 5)
  (h2 : a_n a1 d 3 + a_n a1 d 11 = 3) : a_n a1 d 5 + a_n a1 d 6 + a_n a1 d 10 = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l985_98557


namespace NUMINAMATH_GPT_xiaohua_amount_paid_l985_98536

def cost_per_bag : ℝ := 18
def discount_rate : ℝ := 0.1
def price_difference : ℝ := 36

theorem xiaohua_amount_paid (x : ℝ) 
  (h₁ : 18 * (x+1) * (1 - 0.1) = 18 * x - 36) :
  18 * (x + 1) * (1 - 0.1) = 486 := 
sorry

end NUMINAMATH_GPT_xiaohua_amount_paid_l985_98536


namespace NUMINAMATH_GPT_rectangle_length_width_l985_98542

-- Given conditions
variables (L W : ℕ)

-- Condition 1: The area of the rectangular field is 300 square meters
def area_condition : Prop := L * W = 300

-- Condition 2: The perimeter of the rectangular field is 70 meters
def perimeter_condition : Prop := 2 * (L + W) = 70

-- Condition 3: One side of the rectangle is 20 meters
def side_condition : Prop := L = 20

-- Conclusion
def length_width_proof : Prop :=
  L = 20 ∧ W = 15

-- The final mathematical proof problem statement
theorem rectangle_length_width (L W : ℕ) 
  (h1 : area_condition L W) 
  (h2 : perimeter_condition L W) 
  (h3 : side_condition L) : 
  length_width_proof L W :=
sorry

end NUMINAMATH_GPT_rectangle_length_width_l985_98542


namespace NUMINAMATH_GPT_distance_between_towns_l985_98501

-- Define the custom scale for conversion
def scale_in_km := 1.05  -- 1 km + 50 meters as 1.05 km

-- Input distances on the map and their conversion
def map_distance_in_inches := 6 + 11/16

noncomputable def actual_distance_in_km : ℝ :=
  let distance_in_inches := (6 * 8 + 11) / 16
  distance_in_inches * (8 / 3)

theorem distance_between_towns :
  actual_distance_in_km = 17.85 := by
  -- Equivalent mathematical steps and tests here
  sorry

end NUMINAMATH_GPT_distance_between_towns_l985_98501


namespace NUMINAMATH_GPT_find_d_l985_98595

theorem find_d (d : ℕ) : (1059 % d = 1417 % d) ∧ (1059 % d = 2312 % d) ∧ (1417 % d = 2312 % d) ∧ (d > 1) → d = 179 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l985_98595


namespace NUMINAMATH_GPT_no_polygon_with_half_parallel_diagonals_l985_98563

open Set

noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

def is_parallel_diagonal (n i j : ℕ) : Bool := 
  -- Here, you should define the mathematical condition of a diagonal being parallel to a side
  ((j - i) % n = 0) -- This is a placeholder; the actual condition would depend on the precise geometric definition.

theorem no_polygon_with_half_parallel_diagonals (n : ℕ) (h1 : n ≥ 3) :
  ¬(∃ (k : ℕ), k = num_diagonals n ∧ (∀ (i j : ℕ), i < j ∧ is_parallel_diagonal n i j = true → k = num_diagonals n / 2)) :=
by
  sorry

end NUMINAMATH_GPT_no_polygon_with_half_parallel_diagonals_l985_98563


namespace NUMINAMATH_GPT_sqrt_and_cbrt_eq_self_l985_98526

theorem sqrt_and_cbrt_eq_self (x : ℝ) (h1 : x = Real.sqrt x) (h2 : x = x^(1/3)) : x = 0 := by
  sorry

end NUMINAMATH_GPT_sqrt_and_cbrt_eq_self_l985_98526


namespace NUMINAMATH_GPT_cuboid_height_l985_98507

theorem cuboid_height
  (volume : ℝ)
  (width : ℝ)
  (length : ℝ)
  (height : ℝ)
  (h_volume : volume = 315)
  (h_width : width = 9)
  (h_length : length = 7)
  (h_volume_eq : volume = length * width * height) :
  height = 5 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_height_l985_98507


namespace NUMINAMATH_GPT_tangent_line_eq_l985_98505

/-- The equation of the tangent line to the curve y = 2x * tan x at the point x = π/4 is 
    (2 + π/2) * x - y - π^2/4 = 0. -/
theorem tangent_line_eq : ∀ x y : ℝ, 
  (y = 2 * x * Real.tan x) →
  (x = Real.pi / 4) →
  ((2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0) :=
by
  intros x y h_curve h_point
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l985_98505


namespace NUMINAMATH_GPT_circumscribed_triangle_area_relation_l985_98502

theorem circumscribed_triangle_area_relation
  (a b c D E F : ℝ)
  (h₁ : a = 18) (h₂ : b = 24) (h₃ : c = 30)
  (triangle_right : a^2 + b^2 = c^2)
  (triangle_area : (1/2) * a * b = 216)
  (circle_area : π * (c / 2)^2 = 225 * π)
  (non_triangle_areas : D + E + 216 = F) :
  D + E + 216 = F :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_triangle_area_relation_l985_98502


namespace NUMINAMATH_GPT_no_integral_solutions_l985_98515

theorem no_integral_solutions : ∀ (x : ℤ), x^5 - 31 * x + 2015 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integral_solutions_l985_98515


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l985_98590

-- Definitions corresponding to conditions
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Theorem stating the relationship between p and q
theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) := 
by
  sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l985_98590


namespace NUMINAMATH_GPT_find_distance_l985_98564

-- Definitions of given conditions
def speed : ℝ := 65 -- km/hr
def time  : ℝ := 3  -- hr

-- Statement: The distance is 195 km given the speed and time.
theorem find_distance (speed : ℝ) (time : ℝ) : (speed * time = 195) :=
by
  sorry

end NUMINAMATH_GPT_find_distance_l985_98564


namespace NUMINAMATH_GPT_quadratic_roots_identity_l985_98582

variable (α β : ℝ)
variable (h1 : α^2 + 3*α - 7 = 0)
variable (h2 : β^2 + 3*β - 7 = 0)

-- The problem is to prove that α^2 + 4*α + β = 4
theorem quadratic_roots_identity :
  α^2 + 4*α + β = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l985_98582


namespace NUMINAMATH_GPT_hamburgers_second_day_l985_98571

theorem hamburgers_second_day (x H D : ℕ) (h1 : 3 * H + 4 * D = 10) (h2 : x * H + 3 * D = 7) (h3 : D = 1) (h4 : H = 2) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_hamburgers_second_day_l985_98571


namespace NUMINAMATH_GPT_function_increment_l985_98531

theorem function_increment (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 / x) : f 1.5 - f 2 = 1 / 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_function_increment_l985_98531


namespace NUMINAMATH_GPT_sam_initial_puppies_l985_98503

theorem sam_initial_puppies (gave_away : ℝ) (now_has : ℝ) (initially : ℝ) 
    (h1 : gave_away = 2.0) (h2 : now_has = 4.0) : initially = 6.0 :=
by
  sorry

end NUMINAMATH_GPT_sam_initial_puppies_l985_98503


namespace NUMINAMATH_GPT_greatest_prime_factor_of_144_l985_98535

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_144_l985_98535


namespace NUMINAMATH_GPT_original_deck_card_count_l985_98567

variable (r b : ℕ)

theorem original_deck_card_count (h1 : r / (r + b) = 1 / 4) (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_original_deck_card_count_l985_98567


namespace NUMINAMATH_GPT_first_prime_year_with_digit_sum_8_l985_98578

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem first_prime_year_with_digit_sum_8 :
  ∃ y : ℕ, y > 2015 ∧ sum_of_digits y = 8 ∧ is_prime y ∧
  ∀ z : ℕ, z > 2015 ∧ sum_of_digits z = 8 ∧ is_prime z → y ≤ z :=
sorry

end NUMINAMATH_GPT_first_prime_year_with_digit_sum_8_l985_98578


namespace NUMINAMATH_GPT_cos_seven_theta_l985_98506

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end NUMINAMATH_GPT_cos_seven_theta_l985_98506


namespace NUMINAMATH_GPT_num_nonnegative_real_values_l985_98583

theorem num_nonnegative_real_values :
  ∃ n : ℕ, ∀ x : ℝ, (x ≥ 0) → (∃ k : ℕ, (169 - (x^(1/3))) = k^2) → n = 27 := 
sorry

end NUMINAMATH_GPT_num_nonnegative_real_values_l985_98583


namespace NUMINAMATH_GPT_neg_prop_true_l985_98596

theorem neg_prop_true (a : ℝ) :
  ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) → ∃ a : ℝ, a > 2 ∧ a^2 ≥ 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_neg_prop_true_l985_98596


namespace NUMINAMATH_GPT_allison_rolls_greater_probability_l985_98520

theorem allison_rolls_greater_probability :
  let allison_roll : ℕ := 6
  let charlie_prob_less_6 := 5 / 6
  let mia_prob_rolls_3 := 4 / 6
  let combined_prob := charlie_prob_less_6 * (mia_prob_rolls_3)
  combined_prob = 5 / 9 := by
  sorry

end NUMINAMATH_GPT_allison_rolls_greater_probability_l985_98520


namespace NUMINAMATH_GPT_k_value_for_z_perfect_square_l985_98552

theorem k_value_for_z_perfect_square (Z K : ℤ) (h1 : 500 < Z ∧ Z < 1000) (h2 : K > 1) (h3 : Z = K * K^2) :
  ∃ K : ℤ, Z = 729 ∧ K = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_k_value_for_z_perfect_square_l985_98552


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_l985_98522

theorem standard_equation_of_ellipse
  (a b c : ℝ)
  (h_major_minor : 2 * a = 6 * b)
  (h_focal_distance : 2 * c = 8)
  (h_ellipse_relation : a^2 = b^2 + c^2) :
  (∀ x y : ℝ, (x^2 / 18 + y^2 / 2 = 1) ∨ (y^2 / 18 + x^2 / 2 = 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_standard_equation_of_ellipse_l985_98522


namespace NUMINAMATH_GPT_perimeter_of_rhombus_l985_98532

theorem perimeter_of_rhombus (d1 d2 : ℝ) (hd1 : d1 = 8) (hd2 : d2 = 30) :
  (perimeter : ℝ) = 4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) :=
by
  simp [hd1, hd2]
  sorry

end NUMINAMATH_GPT_perimeter_of_rhombus_l985_98532


namespace NUMINAMATH_GPT_percent_decrease_second_year_l985_98597

theorem percent_decrease_second_year
  (V_0 V_1 V_2 : ℝ)
  (p_2 : ℝ)
  (h1 : V_1 = V_0 * 0.7)
  (h2 : V_2 = V_1 * (1 - p_2 / 100))
  (h3 : V_2 = V_0 * 0.63) :
  p_2 = 10 :=
sorry

end NUMINAMATH_GPT_percent_decrease_second_year_l985_98597


namespace NUMINAMATH_GPT_technician_round_trip_completion_percentage_l985_98568

theorem technician_round_trip_completion_percentage :
  ∀ (d total_d : ℝ),
  d = 1 + (0.75 * 1) + (0.5 * 1) + (0.25 * 1) →
  total_d = 4 * 2 →
  (d / total_d) * 100 = 31.25 :=
by
  intros d total_d h1 h2
  sorry

end NUMINAMATH_GPT_technician_round_trip_completion_percentage_l985_98568


namespace NUMINAMATH_GPT_hypotenuse_length_l985_98562

theorem hypotenuse_length (a b c : ℝ) (h₁ : a + b + c = 40) (h₂ : 0.5 * a * b = 24) (h₃ : a^2 + b^2 = c^2) : c = 18.8 := sorry

end NUMINAMATH_GPT_hypotenuse_length_l985_98562


namespace NUMINAMATH_GPT_mandy_bike_time_l985_98533

-- Definitions of the ratios and time spent on yoga
def ratio_gym_bike : ℕ × ℕ := (2, 3)
def ratio_yoga_exercise : ℕ × ℕ := (2, 3)
def time_yoga : ℕ := 20

-- Theorem stating that Mandy will spend 18 minutes riding her bike
theorem mandy_bike_time (r_gb : ℕ × ℕ) (r_ye : ℕ × ℕ) (t_y : ℕ) 
  (h_rgb : r_gb = (2, 3)) (h_rye : r_ye = (2, 3)) (h_ty : t_y = 20) : 
  let t_e := (r_ye.snd * t_y) / r_ye.fst
  let t_part := t_e / (r_gb.fst + r_gb.snd)
  t_part * r_gb.snd = 18 := sorry

end NUMINAMATH_GPT_mandy_bike_time_l985_98533


namespace NUMINAMATH_GPT_power_equality_l985_98551

theorem power_equality (n : ℝ) : (9:ℝ)^4 = (27:ℝ)^n → n = (8:ℝ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_power_equality_l985_98551


namespace NUMINAMATH_GPT_base8_problem_l985_98513

/--
Let A, B, and C be non-zero and distinct digits in base 8 such that
ABC_8 + BCA_8 + CAB_8 = AAA0_8 and A + B = 2C.
Prove that B + C = 14 in base 8.
-/
theorem base8_problem (A B C : ℕ) 
    (h1 : A > 0 ∧ B > 0 ∧ C > 0)
    (h2 : A < 8 ∧ B < 8 ∧ C < 8)
    (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (bcd_sum : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) 
        = 8^3 * A + 8^2 * A + 8 * A)
    (sum_condition : A + B = 2 * C) :
    B + C = A + B := by {
  sorry
}

end NUMINAMATH_GPT_base8_problem_l985_98513


namespace NUMINAMATH_GPT_moneySpentOnPaintbrushes_l985_98508

def totalExpenditure := 90
def costOfCanvases := 40
def costOfPaints := costOfCanvases / 2
def costOfEasel := 15
def costOfOthers := costOfCanvases + costOfPaints + costOfEasel

theorem moneySpentOnPaintbrushes : totalExpenditure - costOfOthers = 15 := by
  sorry

end NUMINAMATH_GPT_moneySpentOnPaintbrushes_l985_98508


namespace NUMINAMATH_GPT_boys_in_biology_is_25_l985_98565

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end NUMINAMATH_GPT_boys_in_biology_is_25_l985_98565
