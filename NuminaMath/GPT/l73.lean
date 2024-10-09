import Mathlib

namespace more_flour_than_sugar_l73_7333

variable (total_flour : ℕ) (total_sugar : ℕ)
variable (flour_added : ℕ)

def additional_flour_needed (total_flour flour_added : ℕ) : ℕ :=
  total_flour - flour_added

theorem more_flour_than_sugar :
  additional_flour_needed 10 7 - 2 = 1 :=
by
  sorry

end more_flour_than_sugar_l73_7333


namespace find_marksman_hit_rate_l73_7393

-- Define the conditions
def independent_shots (p : ℝ) (n : ℕ) : Prop :=
  0 ≤ p ∧ p ≤ 1 ∧ (n ≥ 1)

def hit_probability (p : ℝ) (n : ℕ) : ℝ :=
  1 - (1 - p) ^ n

-- Stating the proof problem in Lean
theorem find_marksman_hit_rate (p : ℝ) (n : ℕ) 
  (h_independent : independent_shots p n) 
  (h_prob : hit_probability p n = 80 / 81) : 
  p = 2 / 3 :=
sorry

end find_marksman_hit_rate_l73_7393


namespace proportional_division_middle_part_l73_7330

theorem proportional_division_middle_part : 
  ∃ x : ℕ, x = 8 ∧ 5 * x = 40 ∧ 3 * x + 5 * x + 7 * x = 120 := 
by
  sorry

end proportional_division_middle_part_l73_7330


namespace find_m_l73_7323

open Real

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) (m : ℝ) : ℝ :=
  2 * cos (ω * x + ϕ) + m

theorem find_m (ω ϕ : ℝ) (hω : 0 < ω)
  (symmetry : ∀ t : ℝ,  f (π / 4 - t) ω ϕ m = f t ω ϕ m)
  (f_π_8 : f (π / 8) ω ϕ m = -1) :
  m = -3 ∨ m = 1 := 
sorry

end find_m_l73_7323


namespace calculate_f_at_2x_l73_7355

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem using the given condition and the desired result
theorem calculate_f_at_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end calculate_f_at_2x_l73_7355


namespace table_height_l73_7372

theorem table_height
  (l d h : ℤ)
  (h_eq1 : l + h - d = 36)
  (h_eq2 : 2 * l + h = 46)
  (l_eq_d : l = d) :
  h = 36 :=
by
  sorry

end table_height_l73_7372


namespace part1_part2_l73_7362

-- Part 1: Prove that x < -12 given the inequality 2(-3 + x) > 3(x + 2)
theorem part1 (x : ℝ) : 2 * (-3 + x) > 3 * (x + 2) → x < -12 := 
  by
  intro h
  sorry

-- Part 2: Prove that 0 ≤ x < 3 given the system of inequalities
theorem part2 (x : ℝ) : 
    (1 / 2) * (x + 1) < 2 ∧ (x + 2) / 2 ≥ (x + 3) / 3 → 0 ≤ x ∧ x < 3 :=
  by
  intro h
  sorry

end part1_part2_l73_7362


namespace rons_siblings_product_l73_7386

theorem rons_siblings_product
  (H_sisters : ℕ)
  (H_brothers : ℕ)
  (Ha_sisters : ℕ)
  (Ha_brothers : ℕ)
  (R_sisters : ℕ)
  (R_brothers : ℕ)
  (Harry_cond : H_sisters = 4 ∧ H_brothers = 6)
  (Harriet_cond : Ha_sisters = 4 ∧ Ha_brothers = 6)
  (Ron_cond_sisters : R_sisters = Ha_sisters)
  (Ron_cond_brothers : R_brothers = Ha_brothers + 2)
  : R_sisters * R_brothers = 32 := by
  sorry

end rons_siblings_product_l73_7386


namespace bobby_pizzas_l73_7365

theorem bobby_pizzas (B : ℕ) (h_slices : (1 / 4 : ℝ) * B = 3) (h_slices_per_pizza : 6 > 0) :
  B / 6 = 2 := by
  sorry

end bobby_pizzas_l73_7365


namespace range_of_a_l73_7396

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - 1 < 3) ∧ (x - a < 0) → (x < a)) → (a ≤ 2) :=
by
  intro h
  sorry

end range_of_a_l73_7396


namespace johny_total_travel_distance_l73_7383

def TravelDistanceSouth : ℕ := 40
def TravelDistanceEast : ℕ := TravelDistanceSouth + 20
def TravelDistanceNorth : ℕ := 2 * TravelDistanceEast
def TravelDistanceWest : ℕ := TravelDistanceNorth / 2

theorem johny_total_travel_distance
    (hSouth : TravelDistanceSouth = 40)
    (hEast  : TravelDistanceEast = 60)
    (hNorth : TravelDistanceNorth = 120)
    (hWest  : TravelDistanceWest = 60)
    (totalDistance : ℕ := TravelDistanceSouth + TravelDistanceEast + TravelDistanceNorth + TravelDistanceWest) :
    totalDistance = 280 := by
  sorry

end johny_total_travel_distance_l73_7383


namespace rationalize_denominator_l73_7399

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 15 = (7 / 3 : ℝ) * Real.sqrt 15 :=
by
  sorry

end rationalize_denominator_l73_7399


namespace unique_real_solution_l73_7345

theorem unique_real_solution (x y : ℝ) (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
sorry

end unique_real_solution_l73_7345


namespace unit_conversion_factor_l73_7392

theorem unit_conversion_factor (u : ℝ) (h₁ : u = 5) (h₂ : (u * 0.9)^2 = 20.25) : u = 5 → (1 : ℝ) = 0.9  :=
sorry

end unit_conversion_factor_l73_7392


namespace coins_in_bag_l73_7304

theorem coins_in_bag (x : ℝ) (h : x + 0.5 * x + 0.25 * x = 140) : x = 80 :=
by sorry

end coins_in_bag_l73_7304


namespace divisible_by_6_of_cubed_sum_div_by_18_l73_7382

theorem divisible_by_6_of_cubed_sum_div_by_18 (a b c : ℤ) 
  (h : a^3 + b^3 + c^3 ≡ 0 [ZMOD 18]) : (a * b * c) ≡ 0 [ZMOD 6] :=
sorry

end divisible_by_6_of_cubed_sum_div_by_18_l73_7382


namespace emily_expenditure_l73_7305

-- Define the conditions
def price_per_flower : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

-- Total flowers bought
def total_flowers (roses daisies : ℕ) : ℕ :=
  roses + daisies

-- Define the cost function
def cost (flowers price_per_flower : ℕ) : ℕ :=
  flowers * price_per_flower

-- Theorem to prove the total expenditure
theorem emily_expenditure : 
  cost (total_flowers roses_bought daisies_bought) price_per_flower = 12 :=
by
  sorry

end emily_expenditure_l73_7305


namespace second_number_value_l73_7313

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end second_number_value_l73_7313


namespace problem_1_solution_problem_2_solution_l73_7380

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 3) - abs (x - a)

-- Proof problem for question 1
theorem problem_1_solution (x : ℝ) : f x 2 ≤ -1/2 ↔ x ≥ 11/4 :=
by
  sorry

-- Proof problem for question 2
theorem problem_2_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ a) ↔ a ∈ Set.Iic (3/2) :=
by
  sorry

end problem_1_solution_problem_2_solution_l73_7380


namespace xy_pos_iff_div_pos_ab_leq_mean_sq_l73_7387

-- Definition for question 1
theorem xy_pos_iff_div_pos (x y : ℝ) : 
  (x * y > 0) ↔ (x / y > 0) :=
sorry

-- Definition for question 3
theorem ab_leq_mean_sq (a b : ℝ) : 
  a * b ≤ ((a + b) / 2) ^ 2 :=
sorry

end xy_pos_iff_div_pos_ab_leq_mean_sq_l73_7387


namespace check_sufficient_condition_for_eq_l73_7348

theorem check_sufficient_condition_for_eq (a b c : ℤ) (h : a = c - 1 ∧ b = a - 1) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 1 := 
by
  sorry

end check_sufficient_condition_for_eq_l73_7348


namespace fliers_left_l73_7347

theorem fliers_left (total : ℕ) (morning_fraction afternoon_fraction : ℚ) 
  (h1 : total = 1000)
  (h2 : morning_fraction = 1/5)
  (h3 : afternoon_fraction = 1/4) :
  let morning_sent := total * morning_fraction
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent
  remaining_after_afternoon = 600 :=
by
  sorry

end fliers_left_l73_7347


namespace diagonal_length_count_l73_7331

theorem diagonal_length_count :
  ∃ (x : ℕ) (h : (3 < x ∧ x < 22)), x = 18 := by
    sorry

end diagonal_length_count_l73_7331


namespace factorize_polynomial_value_of_x_cubed_l73_7377

-- Problem 1: Factorization
theorem factorize_polynomial (x : ℝ) : 42 * x^2 - 33 * x + 6 = 3 * (2 * x - 1) * (7 * x - 2) :=
sorry

-- Problem 2: Given condition and proof of x^3 + 1/x^3
theorem value_of_x_cubed (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^3 + 1 / x^3 = 18 :=
sorry

end factorize_polynomial_value_of_x_cubed_l73_7377


namespace determine_h_l73_7311

noncomputable def h (x : ℝ) : ℝ := -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2

theorem determine_h (x : ℝ) :
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 :=
by
  sorry

end determine_h_l73_7311


namespace minimum_racing_stripes_l73_7322

variable 
  (totalCars : ℕ) (carsWithoutAirConditioning : ℕ) 
  (maxCarsWithAirConditioningWithoutStripes : ℕ)

-- Defining specific problem conditions
def conditions (totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes : ℕ) : Prop :=
  totalCars = 100 ∧ 
  carsWithoutAirConditioning = 37 ∧ 
  maxCarsWithAirConditioningWithoutStripes = 59

-- The statement to be proved
theorem minimum_racing_stripes (h : conditions totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes) :
   exists (R : ℕ ), R = 4 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end minimum_racing_stripes_l73_7322


namespace find_number_l73_7359

theorem find_number (number : ℚ) 
  (H1 : 8 * 60 = 480)
  (H2 : number / 6 = 16 / 480) :
  number = 1 / 5 := 
by
  sorry

end find_number_l73_7359


namespace max_ab_real_positive_l73_7375

theorem max_ab_real_positive (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) : 
  ab ≤ 1 :=
sorry

end max_ab_real_positive_l73_7375


namespace a_4_value_l73_7327

-- Defining the polynomial (2x - 3)^6
def polynomial_expansion (x : ℝ) := (2 * x - 3) ^ 6

-- Given conditions polynomial expansion in terms of (x - 1)
def polynomial_coefficients (x : ℝ) (a : Fin 7 → ℝ) : ℝ :=
  a 0 + a 1 * (x - 1) + a 2 * (x - 1) ^ 2 + a 3 * (x - 1) ^ 3 + a 4 * (x - 1) ^ 4 +
  a 5 * (x - 1) ^ 5 + a 6 * (x - 1) ^ 6

-- The proof problem asking to show a_4 = 240
theorem a_4_value : 
  ∀ a : Fin 7 → ℝ, (∀ x : ℝ, polynomial_expansion x = polynomial_coefficients x a) → a 4 = 240 := by 
  sorry

end a_4_value_l73_7327


namespace solve_system_of_equations_l73_7356

theorem solve_system_of_equations (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^4 + y^4 - x^2 * y^2 = 13)
  (h2 : x^2 - y^2 + 2 * x * y = 1) :
  x = 1 ∧ y = 2 :=
sorry

end solve_system_of_equations_l73_7356


namespace lizette_quiz_average_l73_7318

theorem lizette_quiz_average
  (Q1 Q2 : ℝ)
  (Q3 : ℝ := 92)
  (h : (Q1 + Q2 + Q3) / 3 = 94) :
  (Q1 + Q2) / 2 = 95 := by
sorry

end lizette_quiz_average_l73_7318


namespace unique_prime_triplets_l73_7335

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end unique_prime_triplets_l73_7335


namespace value_of_x_sq_plus_inv_x_sq_l73_7398

theorem value_of_x_sq_plus_inv_x_sq (x : ℝ) (h : x + 1/x = 1.5) : x^2 + (1/x)^2 = 0.25 := 
by 
  sorry

end value_of_x_sq_plus_inv_x_sq_l73_7398


namespace smallest_possible_area_l73_7317

noncomputable def smallest_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) then l * w else 0

theorem smallest_possible_area : ∃ l w : ℕ, 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) ∧ smallest_area l w = 2100 := by
  sorry

end smallest_possible_area_l73_7317


namespace manuscript_page_count_l73_7307

-- Define the main statement
theorem manuscript_page_count
  (P : ℕ)
  (cost_per_page : ℕ := 10)
  (rev1_pages : ℕ := 30)
  (rev2_pages : ℕ := 20)
  (total_cost : ℕ := 1350)
  (cost_rev1 : ℕ := 15)
  (cost_rev2 : ℕ := 20) 
  (remaining_pages_cost : ℕ := 10 * (P - (rev1_pages + rev2_pages))) :
  (remaining_pages_cost + rev1_pages * cost_rev1 + rev2_pages * cost_rev2 = total_cost)
  → P = 100 :=
by
  sorry

end manuscript_page_count_l73_7307


namespace number_of_triangles_with_perimeter_20_l73_7367

-- Declare the condition: number of triangles with integer side lengths and perimeter of 20
def integerTrianglesWithPerimeter (n : ℕ) : ℕ :=
  (Finset.range (n/2 + 1)).card

/-- Prove that the number of triangles with integer side lengths and a perimeter of 20 is 8. -/
theorem number_of_triangles_with_perimeter_20 : integerTrianglesWithPerimeter 20 = 8 := 
  sorry

end number_of_triangles_with_perimeter_20_l73_7367


namespace solve_inequality_l73_7337

theorem solve_inequality (x : ℝ) : 2 - x < 1 → x > 1 := 
by
  sorry

end solve_inequality_l73_7337


namespace possible_slopes_of_line_intersecting_ellipse_l73_7388

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end possible_slopes_of_line_intersecting_ellipse_l73_7388


namespace power_function_value_at_quarter_l73_7315

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem power_function_value_at_quarter (α : ℝ) (h : f 4 α = 1 / 2) : f (1 / 4) α = 2 := 
  sorry

end power_function_value_at_quarter_l73_7315


namespace unit_digit_product_l73_7366

theorem unit_digit_product (n1 n2 n3 : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) :
  (n1 = 68) ∧ (n2 = 59) ∧ (n3 = 71) ∧ (a = 3) ∧ (b = 6) ∧ (c = 7) →
  (a ^ n1 * b ^ n2 * c ^ n3) % 10 = 8 := by
  sorry

end unit_digit_product_l73_7366


namespace no_positive_integers_satisfy_condition_l73_7312

theorem no_positive_integers_satisfy_condition :
  ∀ (n : ℕ), n > 0 → ¬∃ (a b m : ℕ), a > 0 ∧ b > 0 ∧ m > 0 ∧ 
  (a + b * Real.sqrt n) ^ 2023 = Real.sqrt m + Real.sqrt (m + 2022) := by
  sorry

end no_positive_integers_satisfy_condition_l73_7312


namespace hidden_dots_sum_l73_7343

-- Lean 4 equivalent proof problem definition
theorem hidden_dots_sum (d1 d2 d3 d4 : ℕ)
    (h1 : d1 ≠ d2 ∧ d1 + d2 = 7)
    (h2 : d3 ≠ d4 ∧ d3 + d4 = 7)
    (h3 : d1 = 2 ∨ d1 = 4 ∨ d2 = 2 ∨ d2 = 4)
    (h4 : d3 + 4 = 7) :
    d1 + 7 + 7 + d3 = 24 :=
sorry

end hidden_dots_sum_l73_7343


namespace original_price_of_shirt_l73_7349

variables (S C P : ℝ)

def shirt_condition := S = C / 3
def pants_condition := S = P / 2
def total_paid := 0.90 * S + 0.95 * C + P = 900

theorem original_price_of_shirt :
  shirt_condition S C →
  pants_condition S P →
  total_paid S C P →
  S = 900 / 5.75 :=
by
  sorry

end original_price_of_shirt_l73_7349


namespace max_matrix_det_l73_7351

noncomputable def matrix_det (θ : ℝ) : ℝ :=
  by
    let M := ![
      ![1, 1, 1],
      ![1, 1 + Real.sin θ ^ 2, 1],
      ![1 + Real.cos θ ^ 2, 1, 1]
    ]
    exact Matrix.det M

theorem max_matrix_det : ∃ θ : ℝ, matrix_det θ = 3/4 :=
  sorry

end max_matrix_det_l73_7351


namespace tom_initial_amount_l73_7344

variables (t s j : ℝ)

theorem tom_initial_amount :
  t + s + j = 1200 →
  t - 200 + 3 * s + 2 * j = 1800 →
  t = 400 :=
by
  intros h1 h2
  sorry

end tom_initial_amount_l73_7344


namespace remove_blue_to_get_80_percent_red_l73_7300

-- Definitions from the conditions
def total_balls : ℕ := 150
def red_balls : ℕ := 60
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℤ := 80

-- Lean statement of the proof problem
theorem remove_blue_to_get_80_percent_red :
  ∃ (x : ℕ), (x ≤ initial_blue_balls) ∧ (red_balls * 100 = desired_percentage_red * (total_balls - x)) → x = 75 := sorry

end remove_blue_to_get_80_percent_red_l73_7300


namespace kali_height_now_l73_7341

variable (K_initial J_initial : ℝ)
variable (K_growth J_growth : ℝ)
variable (J_current : ℝ)

theorem kali_height_now :
  J_initial = K_initial →
  J_growth = (2 / 3) * 0.3 * K_initial →
  K_growth = 0.3 * K_initial →
  J_current = 65 →
  J_current = J_initial + J_growth →
  K_current = K_initial + K_growth →
  K_current = 70.42 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end kali_height_now_l73_7341


namespace trigonometric_identity_l73_7364

theorem trigonometric_identity :
  (1 / Real.cos 80) - (Real.sqrt 3 / Real.cos 10) = 4 :=
by
  sorry

end trigonometric_identity_l73_7364


namespace find_2a_plus_b_l73_7381

open Function

noncomputable def f (a b : ℝ) (x : ℝ) := 2 * a * x - 3 * b
noncomputable def g (x : ℝ) := 5 * x + 4
noncomputable def h (a b : ℝ) (x : ℝ) := g (f a b x)
noncomputable def h_inv (x : ℝ) := 2 * x - 9

theorem find_2a_plus_b (a b : ℝ) (h_comp_inv_eq_id : ∀ x, h a b (h_inv x) = x) :
  2 * a + b = 1 / 15 := 
sorry

end find_2a_plus_b_l73_7381


namespace soccer_game_goals_l73_7397

theorem soccer_game_goals (A1_first_half A2_first_half B1_first_half B2_first_half : ℕ) 
  (h1 : A1_first_half = 8)
  (h2 : B1_first_half = A1_first_half / 2)
  (h3 : B2_first_half = A1_first_half)
  (h4 : A2_first_half = B2_first_half - 2) : 
  A1_first_half + A2_first_half + B1_first_half + B2_first_half = 26 :=
by
  -- The proof is not needed, so we use sorry to skip it.
  sorry

end soccer_game_goals_l73_7397


namespace range_of_m_l73_7391

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h : 4 / x + 1 / y = 1) :
  x + y ≥ m^2 + m + 3 ↔ -3 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l73_7391


namespace tom_total_amount_after_saving_l73_7321

theorem tom_total_amount_after_saving :
  let hourly_rate := 6.50
  let work_hours := 31
  let saving_rate := 0.10
  let total_earnings := hourly_rate * work_hours
  let amount_set_aside := total_earnings * saving_rate
  let amount_for_purchases := total_earnings - amount_set_aside
  amount_for_purchases = 181.35 :=
by
  sorry

end tom_total_amount_after_saving_l73_7321


namespace mixed_bead_cost_per_box_l73_7360

-- Definitions based on given conditions
def red_bead_cost : ℝ := 1.30
def yellow_bead_cost : ℝ := 2.00
def total_boxes : ℕ := 10
def red_boxes_used : ℕ := 4
def yellow_boxes_used : ℕ := 4

-- Theorem statement
theorem mixed_bead_cost_per_box :
  ((red_boxes_used * red_bead_cost) + (yellow_boxes_used * yellow_bead_cost)) / total_boxes = 1.32 :=
  by sorry

end mixed_bead_cost_per_box_l73_7360


namespace correct_calculation_result_l73_7309

theorem correct_calculation_result (x : ℤ) (h : 4 * x + 16 = 32) : (x / 4) + 16 = 17 := by
  sorry

end correct_calculation_result_l73_7309


namespace find_unknown_number_l73_7361

theorem find_unknown_number (x : ℝ) (h : (8 / 100) * x = 96) : x = 1200 :=
by
  sorry

end find_unknown_number_l73_7361


namespace evaluate_expression_l73_7306

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 := by
  sorry

end evaluate_expression_l73_7306


namespace LeRoy_should_pay_Bernardo_l73_7378

theorem LeRoy_should_pay_Bernardo 
    (initial_loan : ℕ := 100)
    (LeRoy_gas_expense : ℕ := 300)
    (LeRoy_food_expense : ℕ := 200)
    (Bernardo_accommodation_expense : ℕ := 500)
    (total_expense := LeRoy_gas_expense + LeRoy_food_expense + Bernardo_accommodation_expense)
    (shared_expense := total_expense / 2)
    (LeRoy_total_responsibility := shared_expense + initial_loan)
    (LeRoy_needs_to_pay := LeRoy_total_responsibility - (LeRoy_gas_expense + LeRoy_food_expense)) :
    LeRoy_needs_to_pay = 100 := 
by
    sorry

end LeRoy_should_pay_Bernardo_l73_7378


namespace find_c_minus_a_l73_7385

variable (a b c d e : ℝ)

-- Conditions
axiom avg_ab : (a + b) / 2 = 40
axiom avg_bc : (b + c) / 2 = 60
axiom avg_de : (d + e) / 2 = 80
axiom geom_mean : (a * b * d) = (b * c * e)

theorem find_c_minus_a : c - a = 40 := by
  sorry

end find_c_minus_a_l73_7385


namespace investment_calculation_l73_7332

noncomputable def initial_investment (final_amount : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  final_amount / ((1 + interest_rate / 100) ^ years)

theorem investment_calculation :
  initial_investment 504.32 3 12 = 359 :=
by
  sorry

end investment_calculation_l73_7332


namespace leila_total_cakes_l73_7357

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 := by
  sorry

end leila_total_cakes_l73_7357


namespace AplusBplusC_4_l73_7320

theorem AplusBplusC_4 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 1 ∧ Nat.gcd a c = 1 ∧ (a^2 + a * b + b^2 = c^2) ∧ (a + b + c = 4) :=
by
  sorry

end AplusBplusC_4_l73_7320


namespace problem_statement_l73_7308

def contrapositive {P Q : Prop} (h : P → Q) : ¬Q → ¬P :=
by sorry

def sufficient_but_not_necessary (P Q : Prop) : (P → Q) ∧ ¬(Q → P) :=
by sorry

def proposition_C (p q : Prop) : ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

def negate_exists (P : ℝ → Prop) : (∃ x : ℝ, P x) → ¬(∀ x : ℝ, ¬P x) :=
by sorry

theorem problem_statement : 
¬ (∀ (P Q : Prop), ¬(P ∧ Q) → (¬P ∨ ¬Q)) :=
by sorry

end problem_statement_l73_7308


namespace find_starting_number_l73_7371

theorem find_starting_number (S : ℤ) (n : ℤ) (sum_eq : 10 = S) (consec_eq : S = (20 / 2) * (n + (n + 19))) : 
  n = -9 := 
by
  sorry

end find_starting_number_l73_7371


namespace doughnuts_left_l73_7328

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end doughnuts_left_l73_7328


namespace direct_proportion_function_l73_7346

theorem direct_proportion_function (m : ℝ) (h : ∀ x : ℝ, -2*x + m = k*x → m = 0) : m = 0 :=
sorry

end direct_proportion_function_l73_7346


namespace max_n_value_l73_7384

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem max_n_value (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n → (2 * (n + 0.5) = a n + a (n + 1))) 
  (h2 : S a 63 = 2020) (h3 : a 2 < 3) : 63 ∈ { n : ℕ | S a n = 2020 } :=
sorry

end max_n_value_l73_7384


namespace farmer_total_acres_l73_7374

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l73_7374


namespace sum_of_coefficients_l73_7340

theorem sum_of_coefficients
  (d : ℝ)
  (g h : ℝ)
  (h1 : (8 * d^2 - 4 * d + g) * (5 * d^2 + h * d - 10) = 40 * d^4 - 75 * d^3 - 90 * d^2 + 5 * d + 20) :
  g + h = 15.5 :=
sorry

end sum_of_coefficients_l73_7340


namespace hyperbola_equation_l73_7310

-- Conditions
def center_origin (P : ℝ × ℝ) : Prop := P = (0, 0)
def focus_at (F : ℝ × ℝ) : Prop := F = (0, Real.sqrt 3)
def vertex_distance (d : ℝ) : Prop := d = Real.sqrt 3 - 1

-- Statement
theorem hyperbola_equation
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (d : ℝ)
  (h_center : center_origin center)
  (h_focus : focus_at focus)
  (h_vert_dist : vertex_distance d) :
  y^2 - (x^2 / 2) = 1 := 
sorry

end hyperbola_equation_l73_7310


namespace otimes_self_twice_l73_7353

def otimes (x y : ℝ) := x^2 - y^2

theorem otimes_self_twice (a : ℝ) : (otimes (otimes a a) (otimes a a)) = 0 :=
  sorry

end otimes_self_twice_l73_7353


namespace count_distinct_m_values_l73_7368

theorem count_distinct_m_values : 
  ∃ m_values : Finset ℤ, 
  (∀ x1 x2 : ℤ, x1 * x2 = 30 → (m_values : Set ℤ) = { x1 + x2 }) ∧ 
  m_values.card = 8 :=
by
  sorry

end count_distinct_m_values_l73_7368


namespace polar_coordinates_full_circle_l73_7369

theorem polar_coordinates_full_circle :
  ∀ (r : ℝ) (θ : ℝ), (r = 3 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → (r = 3 ∧ ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ↔ r = 3) :=
by
  intros r θ h
  sorry

end polar_coordinates_full_circle_l73_7369


namespace unique_n_in_range_satisfying_remainders_l73_7389

theorem unique_n_in_range_satisfying_remainders : 
  ∃! (n : ℤ) (k r : ℤ), 150 < n ∧ n < 250 ∧ 0 <= r ∧ r <= 6 ∧ n = 7 * k + r ∧ n = 9 * r + r :=
sorry

end unique_n_in_range_satisfying_remainders_l73_7389


namespace remaining_distance_proof_l73_7395

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l73_7395


namespace teams_in_each_group_l73_7342

theorem teams_in_each_group (n : ℕ) :
  (2 * (n * (n - 1) / 2) + 3 * n = 56) → n = 7 :=
by
  sorry

end teams_in_each_group_l73_7342


namespace real_solutions_infinite_l73_7339

theorem real_solutions_infinite : 
  ∃ (S : Set ℝ), (∀ x ∈ S, - (x^2 - 4) ≥ 0) ∧ S.Infinite :=
sorry

end real_solutions_infinite_l73_7339


namespace rowing_time_75_minutes_l73_7354

-- Definition of time duration Ethan rowed.
def EthanRowingTime : ℕ := 25  -- minutes

-- Definition of the time duration Frank rowed.
def FrankRowingTime : ℕ := 2 * EthanRowingTime  -- twice as long as Ethan.

-- Definition of the total rowing time.
def TotalRowingTime : ℕ := EthanRowingTime + FrankRowingTime

-- Theorem statement proving the total rowing time is 75 minutes.
theorem rowing_time_75_minutes : TotalRowingTime = 75 := by
  -- The proof is omitted.
  sorry

end rowing_time_75_minutes_l73_7354


namespace number_of_true_propositions_is_2_l73_7338

-- Definitions for the propositions
def original_proposition (x : ℝ) : Prop := x > -3 → x > -6
def converse_proposition (x : ℝ) : Prop := x > -6 → x > -3
def inverse_proposition (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive_proposition (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The theorem we need to prove
theorem number_of_true_propositions_is_2 :
  (∀ x, original_proposition x) ∧ (∀ x, contrapositive_proposition x) ∧ 
  ¬ (∀ x, converse_proposition x) ∧ ¬ (∀ x, inverse_proposition x) → 2 = 2 := 
sorry

end number_of_true_propositions_is_2_l73_7338


namespace parallel_vectors_x_val_l73_7302

open Real

theorem parallel_vectors_x_val (x : ℝ) :
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (x, 1/2)
  a.1 * b.2 = a.2 * b.1 →
  x = 3/8 := 
by
  intro h
  -- Use this line if you need to skip the proof
  sorry

end parallel_vectors_x_val_l73_7302


namespace option_one_correct_l73_7301

theorem option_one_correct (x : ℝ) : 
  (x ≠ 0 → x + |x| > 0) ∧ ¬((x + |x| > 0) → x ≠ 0) := 
by
  sorry

end option_one_correct_l73_7301


namespace range_of_set_l73_7334

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end range_of_set_l73_7334


namespace brokerage_percentage_l73_7376

theorem brokerage_percentage
  (f : ℝ) (d : ℝ) (c : ℝ) 
  (hf : f = 100)
  (hd : d = 0.08)
  (hc : c = 92.2)
  (h_disc_price : f - f * d = 92) :
  (c - (f - f * d)) / f * 100 = 0.2 := 
by
  sorry

end brokerage_percentage_l73_7376


namespace brenda_ends_with_12_skittles_l73_7336

def initial_skittles : ℕ := 7
def bought_skittles : ℕ := 8
def given_away_skittles : ℕ := 3

theorem brenda_ends_with_12_skittles :
  initial_skittles + bought_skittles - given_away_skittles = 12 := by
  sorry

end brenda_ends_with_12_skittles_l73_7336


namespace faster_train_cross_time_l73_7352

/-- Statement of the problem in Lean 4 -/
theorem faster_train_cross_time :
  let speed_faster_train_kmph := 72
  let speed_slower_train_kmph := 36
  let length_faster_train_m := 180
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18 : ℝ)
  let time_taken := length_faster_train_m / relative_speed_mps
  time_taken = 18 :=
by
  sorry

end faster_train_cross_time_l73_7352


namespace ram_gohul_work_days_l73_7325

theorem ram_gohul_work_days (ram_days gohul_days : ℕ) (H_ram: ram_days = 10) (H_gohul: gohul_days = 15): 
  (ram_days * gohul_days) / (ram_days + gohul_days) = 6 := 
by
  sorry

end ram_gohul_work_days_l73_7325


namespace trigonometric_identity_l73_7326

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := 
by 
  sorry

end trigonometric_identity_l73_7326


namespace part1_part2_l73_7329

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x * (Real.sin x + Real.cos x)) - 1 / 2

theorem part1 (α : ℝ) (hα1 : 0 < α ∧ α < Real.pi / 2) (hα2 : Real.sin α = Real.sqrt 2 / 2) :
  f α = 1 / 2 :=
sorry

theorem part2 :
  ∀ (k : ℤ), ∀ (x : ℝ),
  -((3 : ℝ) * Real.pi / 8) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 8) + k * Real.pi →
  MonotoneOn f (Set.Icc (-((3 : ℝ) * Real.pi / 8) + k * Real.pi) ((Real.pi / 8) + k * Real.pi)) :=
sorry

end part1_part2_l73_7329


namespace max_value_64_l73_7314

-- Define the types and values of gemstones
structure Gemstone where
  weight : ℕ
  value : ℕ

-- Introduction of the three types of gemstones
def gem1 : Gemstone := ⟨3, 9⟩
def gem2 : Gemstone := ⟨5, 16⟩
def gem3 : Gemstone := ⟨2, 5⟩

-- Maximum weight Janet can carry
def max_weight := 20

-- Problem statement: Proving the maximum value Janet can carry is $64
theorem max_value_64 (n1 n2 n3 : ℕ) (h1 : n1 ≥ 15) (h2 : n2 ≥ 15) (h3 : n3 ≥ 15) 
  (weight_limit : n1 * gem1.weight + n2 * gem2.weight + n3 * gem3.weight ≤ max_weight) : 
  n1 * gem1.value + n2 * gem2.value + n3 * gem3.value ≤ 64 :=
sorry

end max_value_64_l73_7314


namespace total_bricks_calculation_l73_7363

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end total_bricks_calculation_l73_7363


namespace value_of_M_l73_7370

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end value_of_M_l73_7370


namespace power_function_inverse_l73_7316

theorem power_function_inverse (f : ℝ → ℝ) (h₁ : f 2 = (Real.sqrt 2) / 2) : f⁻¹ 2 = 1 / 4 :=
by
  -- Lean proof will be filled here
  sorry

end power_function_inverse_l73_7316


namespace solve_system_eq_solve_system_ineq_l73_7394

-- For the system of equations:
theorem solve_system_eq (x y : ℝ) (h1 : x + 2 * y = 7) (h2 : 3 * x + y = 6) : x = 1 ∧ y = 3 :=
sorry

-- For the system of inequalities:
theorem solve_system_ineq (x : ℝ) (h1 : 2 * (x - 1) + 1 > -3) (h2 : x - 1 ≤ (1 + x) / 3) : -1 < x ∧ x ≤ 2 :=
sorry

end solve_system_eq_solve_system_ineq_l73_7394


namespace combined_share_a_c_l73_7373

-- Define the conditions
def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

-- The total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

-- The value of each part
def value_per_part : ℕ := total_money / total_parts

-- The shares of a and c
def share_a : ℕ := ratio_a * value_per_part
def share_c : ℕ := ratio_c * value_per_part

-- Prove that the combined share of a + c equals 6952
theorem combined_share_a_c : share_a + share_c = 6952 :=
by
  -- This is the proof placeholder
  sorry

end combined_share_a_c_l73_7373


namespace platform_length_l73_7390

theorem platform_length (train_length : ℝ) (time_cross_pole : ℝ) (time_cross_platform : ℝ) (speed : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_cross_pole = 18) 
  (h3 : time_cross_platform = 54)
  (h4 : speed = train_length / time_cross_pole) :
  train_length + (speed * time_cross_platform) - train_length = 600 := 
by
  sorry

end platform_length_l73_7390


namespace alcohol_percentage_proof_l73_7350

noncomputable def percentage_alcohol_new_mixture 
  (original_solution_volume : ℕ)
  (percent_A : ℚ)
  (concentration_A : ℚ)
  (percent_B : ℚ)
  (concentration_B : ℚ)
  (percent_C : ℚ)
  (concentration_C : ℚ)
  (water_added_volume : ℕ) : ℚ :=
((original_solution_volume * percent_A * concentration_A) +
 (original_solution_volume * percent_B * concentration_B) +
 (original_solution_volume * percent_C * concentration_C)) /
 (original_solution_volume + water_added_volume) * 100

theorem alcohol_percentage_proof : 
  percentage_alcohol_new_mixture 24 0.30 0.80 0.40 0.90 0.30 0.95 16 = 53.1 := 
by 
  sorry

end alcohol_percentage_proof_l73_7350


namespace ceil_sqrt_250_eq_16_l73_7319

theorem ceil_sqrt_250_eq_16 : ⌈Real.sqrt 250⌉ = 16 :=
by
  have h1 : (15 : ℝ) < Real.sqrt 250 := sorry
  have h2 : Real.sqrt 250 < 16 := sorry
  exact sorry

end ceil_sqrt_250_eq_16_l73_7319


namespace total_granola_bars_l73_7358

-- Problem conditions
def oatmeal_raisin_bars : ℕ := 6
def peanut_bars : ℕ := 8

-- Statement to prove
theorem total_granola_bars : oatmeal_raisin_bars + peanut_bars = 14 := 
by 
  sorry

end total_granola_bars_l73_7358


namespace circles_tangent_iff_l73_7379

noncomputable def C1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 }
noncomputable def C2 (m: ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.1 + 8 * p.2 + m = 0 }

theorem circles_tangent_iff (m: ℝ) : (∀ p ∈ C1, p ∈ C2 m → False) ↔ (m = -4 ∨ m = 16) := 
sorry

end circles_tangent_iff_l73_7379


namespace kelsey_video_count_l73_7303

variable (E U K : ℕ)

noncomputable def total_videos : ℕ := 411
noncomputable def ekon_videos : ℕ := E
noncomputable def uma_videos : ℕ := E + 17
noncomputable def kelsey_videos : ℕ := E + 43

theorem kelsey_video_count (E U K : ℕ) 
  (h1 : total_videos = ekon_videos + uma_videos + kelsey_videos)
  (h2 : uma_videos = ekon_videos + 17)
  (h3 : kelsey_videos = ekon_videos + 43)
  : kelsey_videos = 160 := 
sorry

end kelsey_video_count_l73_7303


namespace percent_of_employed_people_who_are_females_l73_7324

theorem percent_of_employed_people_who_are_females (p_employed p_employed_males : ℝ) 
  (h1 : p_employed = 64) (h2 : p_employed_males = 48) : 
  100 * (p_employed - p_employed_males) / p_employed = 25 :=
by
  sorry

end percent_of_employed_people_who_are_females_l73_7324
