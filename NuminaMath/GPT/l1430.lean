import Mathlib

namespace NUMINAMATH_GPT_perimeter_of_triangle_is_36_l1430_143033

variable (inradius : ℝ)
variable (area : ℝ)
variable (P : ℝ)

theorem perimeter_of_triangle_is_36 (h1 : inradius = 2.5) (h2 : area = 45) : 
  P / 2 * inradius = area → P = 36 :=
sorry

end NUMINAMATH_GPT_perimeter_of_triangle_is_36_l1430_143033


namespace NUMINAMATH_GPT_problem_statement_l1430_143078

noncomputable def polynomial_expansion (x : ℚ) : ℚ := (1 - 2 * x) ^ 8

theorem problem_statement :
  (8 * (1 - 2 * 1) ^ 7 * (-2)) = (a_1 : ℚ) + 2 * (a_2 : ℚ) + 3 * (a_3 : ℚ) + 4 * (a_4 : ℚ) +
  5 * (a_5 : ℚ) + 6 * (a_6 : ℚ) + 7 * (a_7 : ℚ) + 8 * (a_8 : ℚ) := by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1430_143078


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1430_143067

theorem solution_set_of_inequality :
  { x : ℝ | 2 / (x - 1) ≥ 1 } = { x : ℝ | 1 < x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1430_143067


namespace NUMINAMATH_GPT_pyramid_sphere_area_l1430_143096

theorem pyramid_sphere_area (a : ℝ) (PA PB PC : ℝ) 
  (h1 : PA = PB) (h2 : PA = 2 * PC) 
  (h3 : PA = 2 * a) (h4 : PB = 2 * a) 
  (h5 : 4 * π * (PA^2 + PB^2 + PC^2) / 9 = 9 * π) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_sphere_area_l1430_143096


namespace NUMINAMATH_GPT_find_k_l1430_143020

variable (x y k : ℝ)

-- Definition: the line equations and the intersection condition
def line1_eq (x y k : ℝ) : Prop := 3 * x - 2 * y = k
def line2_eq (x y : ℝ) : Prop := x - 0.5 * y = 10
def intersect_at_x (x : ℝ) : Prop := x = -6

-- The theorem we need to prove
theorem find_k (h1 : line1_eq x y k)
               (h2 : line2_eq x y)
               (h3 : intersect_at_x x) :
               k = 46 :=
sorry

end NUMINAMATH_GPT_find_k_l1430_143020


namespace NUMINAMATH_GPT_completion_time_l1430_143052

variables {P E : ℝ}
theorem completion_time (h1 : (20 : ℝ) * P * E / 2 = D * (2.5 * P * E)) : D = 4 :=
by
  -- Given h1 as the condition
  sorry

end NUMINAMATH_GPT_completion_time_l1430_143052


namespace NUMINAMATH_GPT_table_fill_impossible_l1430_143054

/-- Proposition: Given a 7x3 table filled with 0s and 1s, it is impossible to prevent any 2x2 submatrix from having all identical numbers. -/
theorem table_fill_impossible : 
  ¬ ∃ (M : (Fin 7) → (Fin 3) → Fin 2), 
      ∀ i j, (i < 6) → (j < 2) → 
              (M i j = M i.succ j) ∨ 
              (M i j = M i j.succ) ∨ 
              (M i j = M i.succ j.succ) ∨ 
              (M i.succ j = M i j.succ → M i j = M i.succ j.succ) :=
sorry

end NUMINAMATH_GPT_table_fill_impossible_l1430_143054


namespace NUMINAMATH_GPT_janet_acres_l1430_143039

-- Defining the variables and conditions
variable (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ)

-- Assigning the given values to the variables
def horseFertilizer := 5
def acreFertilizer := 400
def janetSpreadRate := 4
def janetHorses := 80
def fertilizingDays := 25

-- Main theorem stating the question and proving the answer
theorem janet_acres : 
  ∀ (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ),
  horse_production = 5 → 
  acre_requirement = 400 →
  spread_rate = 4 →
  num_horses = 80 →
  days = 25 →
  (spread_rate * days = 100) := 
by
  intros
  -- Proof would be inserted here
  sorry

end NUMINAMATH_GPT_janet_acres_l1430_143039


namespace NUMINAMATH_GPT_zoe_pictures_l1430_143071

theorem zoe_pictures (P : ℕ) (h1 : P + 16 = 44) : P = 28 :=
by sorry

end NUMINAMATH_GPT_zoe_pictures_l1430_143071


namespace NUMINAMATH_GPT_smaller_side_of_new_rectangle_is_10_l1430_143021

/-- We have a 10x25 rectangle that is divided into two congruent polygons and rearranged 
to form another rectangle. We need to prove that the length of the smaller side of the 
resulting rectangle is 10. -/
theorem smaller_side_of_new_rectangle_is_10 :
  ∃ (y x : ℕ), (y * x = 10 * 25) ∧ (y ≤ x) ∧ y = 10 := 
sorry

end NUMINAMATH_GPT_smaller_side_of_new_rectangle_is_10_l1430_143021


namespace NUMINAMATH_GPT_find_k_from_inequality_l1430_143058

variable (k x : ℝ)

theorem find_k_from_inequality (h : ∀ x ∈ Set.Ico (-2 : ℝ) 1, 1 + k / (x - 1) ≤ 0)
  (h₂: 1 + k / (-2 - 1) = 0) :
  k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_from_inequality_l1430_143058


namespace NUMINAMATH_GPT_inequality_proof_l1430_143003

noncomputable def x : ℝ := Real.exp (-1/2)
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 3

theorem inequality_proof : z > x ∧ x > y := by
  -- Conditions defined as follows:
  -- x = exp(-1/2)
  -- y = log(2) / log(5)
  -- z = log(3)
  -- To be proved:
  -- z > x > y
  sorry

end NUMINAMATH_GPT_inequality_proof_l1430_143003


namespace NUMINAMATH_GPT_charles_total_money_l1430_143073

-- Definitions based on the conditions in step a)
def number_of_pennies : ℕ := 6
def number_of_nickels : ℕ := 3
def value_of_penny : ℕ := 1
def value_of_nickel : ℕ := 5

-- Calculations in Lean terms
def total_pennies_value : ℕ := number_of_pennies * value_of_penny
def total_nickels_value : ℕ := number_of_nickels * value_of_nickel
def total_money : ℕ := total_pennies_value + total_nickels_value

-- The final proof statement based on step c)
theorem charles_total_money : total_money = 21 := by
  sorry

end NUMINAMATH_GPT_charles_total_money_l1430_143073


namespace NUMINAMATH_GPT_y_minus_x_is_7_l1430_143057

theorem y_minus_x_is_7 (x y : ℕ) (hx : x ≠ y) (h1 : 3 + y = 10) (h2 : 0 + x + 1 = 1) (h3 : 3 + 7 = 10) :
  y - x = 7 :=
by
  sorry

end NUMINAMATH_GPT_y_minus_x_is_7_l1430_143057


namespace NUMINAMATH_GPT_coordinates_B_l1430_143069

theorem coordinates_B (A B : ℝ × ℝ) (distance : ℝ) (A_coords : A = (-1, 3)) 
  (AB_parallel_x : A.snd = B.snd) (AB_distance : abs (A.fst - B.fst) = distance) :
  (B = (-6, 3) ∨ B = (4, 3)) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_B_l1430_143069


namespace NUMINAMATH_GPT_profit_percentage_mobile_l1430_143016

-- Definitions derived from conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 0.05
def total_profit : ℝ := 50
def selling_price_grinder := cost_price_grinder * (1 - loss_percentage_grinder)
def total_cost_price := cost_price_grinder + cost_price_mobile
def total_selling_price := total_cost_price + total_profit
def selling_price_mobile := total_selling_price - selling_price_grinder
def profit_mobile := selling_price_mobile - cost_price_mobile

-- The theorem to prove the profit percentage on the mobile phone is 10%
theorem profit_percentage_mobile : (profit_mobile / cost_price_mobile) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_mobile_l1430_143016


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1430_143051

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else if x > 0 then x - 2
  else 0

theorem solution_set_of_inequality :
  {x : ℝ | 2 * f x - 1 < 0} = {x | x < -3 / 2 ∨ (0 ≤ x ∧ x < 5 / 2)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1430_143051


namespace NUMINAMATH_GPT_quadratic_completing_square_l1430_143055

theorem quadratic_completing_square:
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 + 900 * x + 1800 = (x + b)^2 + c) ∧ (c / b = -446.22222) :=
by
  -- We'll skip the proof steps here
  sorry

end NUMINAMATH_GPT_quadratic_completing_square_l1430_143055


namespace NUMINAMATH_GPT_one_fourth_of_eight_times_x_plus_two_l1430_143023

theorem one_fourth_of_eight_times_x_plus_two (x : ℝ) : 
  (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_one_fourth_of_eight_times_x_plus_two_l1430_143023


namespace NUMINAMATH_GPT_binom_1300_2_l1430_143062

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end NUMINAMATH_GPT_binom_1300_2_l1430_143062


namespace NUMINAMATH_GPT_man_rate_still_water_l1430_143009

def speed_with_stream : ℝ := 6
def speed_against_stream : ℝ := 2

theorem man_rate_still_water : (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

end NUMINAMATH_GPT_man_rate_still_water_l1430_143009


namespace NUMINAMATH_GPT_fleas_initial_minus_final_l1430_143014

theorem fleas_initial_minus_final (F : ℕ) (h : F / 16 = 14) :
  F - 14 = 210 :=
sorry

end NUMINAMATH_GPT_fleas_initial_minus_final_l1430_143014


namespace NUMINAMATH_GPT_meal_combinations_l1430_143038

def menu_items : ℕ := 12
def special_dish_chosen : Prop := true

theorem meal_combinations : (special_dish_chosen → (menu_items - 1) * (menu_items - 1) = 121) :=
by
  sorry

end NUMINAMATH_GPT_meal_combinations_l1430_143038


namespace NUMINAMATH_GPT_harkamal_total_amount_l1430_143017

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end NUMINAMATH_GPT_harkamal_total_amount_l1430_143017


namespace NUMINAMATH_GPT_first_rocket_height_l1430_143022

theorem first_rocket_height (h : ℝ) (combined_height : ℝ) (second_rocket_height : ℝ) 
  (H1 : second_rocket_height = 2 * h) 
  (H2 : combined_height = h + second_rocket_height) 
  (H3 : combined_height = 1500) : h = 500 := 
by 
  -- The proof would go here but is not required as per the instruction.
  sorry

end NUMINAMATH_GPT_first_rocket_height_l1430_143022


namespace NUMINAMATH_GPT_find_sum_l1430_143056

theorem find_sum (P : ℕ) (h_total : P * (4/100 + 6/100 + 8/100) = 2700) : P = 15000 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l1430_143056


namespace NUMINAMATH_GPT_area_of_quadrilateral_NLMK_l1430_143075

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_quadrilateral_NLMK 
  (AB BC AC AK CN CL : ℝ)
  (h_AB : AB = 13)
  (h_BC : BC = 20)
  (h_AC : AC = 21)
  (h_AK : AK = 4)
  (h_CN : CN = 1)
  (h_CL : CL = 20 / 21) : 
  triangle_area AB BC AC - 
  (1 * CL / (BC * AC) * triangle_area AB BC AC) - 
  (9 * (BC - CN) / (AB * BC) * triangle_area AB BC AC) -
  (16 * 41 / (169 * 21) * triangle_area AB BC AC) = 
  493737 / 11830 := 
sorry

end NUMINAMATH_GPT_area_of_quadrilateral_NLMK_l1430_143075


namespace NUMINAMATH_GPT_average_first_two_l1430_143012

theorem average_first_two (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) = 16.8)
  (h2 : (c + d) = 4.6)
  (h3 : (e + f) = 7.4) : 
  (a + b) / 2 = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_average_first_two_l1430_143012


namespace NUMINAMATH_GPT_find_integer_k_l1430_143046

theorem find_integer_k {k : ℤ} :
  (∀ x : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 →
    (∃ m n : ℝ, m ≠ n ∧ m * n = 1 / (k^2 + 1) ∧ m + n = (4 - k) / (k^2 + 1) ∧
      ((1 < m ∧ n < 1) ∨ (1 < n ∧ m < 1)))) →
  k = -1 ∨ k = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_k_l1430_143046


namespace NUMINAMATH_GPT_jennifer_money_left_over_l1430_143010

theorem jennifer_money_left_over :
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  money_left = 16 :=
by
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  exact sorry

end NUMINAMATH_GPT_jennifer_money_left_over_l1430_143010


namespace NUMINAMATH_GPT_mouse_jump_distance_l1430_143013

theorem mouse_jump_distance
  (g f m : ℕ)
  (hg : g = 25)
  (hf : f = g + 32)
  (hm : m = f - 26) :
  m = 31 := by
  sorry

end NUMINAMATH_GPT_mouse_jump_distance_l1430_143013


namespace NUMINAMATH_GPT_problem_1_problem_2_l1430_143088

-- Definition of sets A and B as in the problem's conditions
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | x > 2 ∨ x < -2}
def C (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- Prove that A ∩ B is as described
theorem problem_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} := by
  sorry

-- Prove that a ≥ 6 given the conditions in the problem
theorem problem_2 (a : ℝ) : (A ⊆ C a) → a ≥ 6 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1430_143088


namespace NUMINAMATH_GPT_chemist_salt_solution_l1430_143037

theorem chemist_salt_solution (x : ℝ) 
  (hx : 0.60 * x = 0.20 * (1 + x)) : x = 0.5 :=
sorry

end NUMINAMATH_GPT_chemist_salt_solution_l1430_143037


namespace NUMINAMATH_GPT_wealth_ratio_l1430_143035

theorem wealth_ratio 
  (P W : ℝ)
  (hP_pos : 0 < P)
  (hW_pos : 0 < W)
  (pop_A : ℝ := 0.30 * P)
  (wealth_A : ℝ := 0.40 * W)
  (pop_B : ℝ := 0.20 * P)
  (wealth_B : ℝ := 0.25 * W)
  (avg_wealth_A : ℝ := wealth_A / pop_A)
  (avg_wealth_B : ℝ := wealth_B / pop_B) :
  avg_wealth_A / avg_wealth_B = 16 / 15 :=
by
  sorry

end NUMINAMATH_GPT_wealth_ratio_l1430_143035


namespace NUMINAMATH_GPT_imaginary_part_z1_mul_z2_l1430_143098

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_z1_mul_z2_l1430_143098


namespace NUMINAMATH_GPT_fish_price_eq_shrimp_price_l1430_143047

-- Conditions
variable (x : ℝ) -- regular price for a full pound of fish
variable (h1 : 0.6 * (x / 4) = 1.50) -- quarter-pound fish price after 60% discount
variable (shrimp_price : ℝ) -- price per pound of shrimp
variable (h2 : shrimp_price = 10) -- given shrimp price

-- Proof Statement
theorem fish_price_eq_shrimp_price (h1 : 0.6 * (x / 4) = 1.50) (h2 : shrimp_price = 10) :
  x = 10 ∧ x = shrimp_price :=
by
  sorry

end NUMINAMATH_GPT_fish_price_eq_shrimp_price_l1430_143047


namespace NUMINAMATH_GPT_complementary_angle_decrease_l1430_143018

theorem complementary_angle_decrease :
  (ratio : ℚ := 3 / 7) →
  let total_angle := 90
  let small_angle := (ratio * total_angle) / (1+ratio)
  let large_angle := total_angle - small_angle
  let new_small_angle := small_angle * 1.2
  let new_large_angle := total_angle - new_small_angle
  let decrease_percent := (large_angle - new_large_angle) / large_angle * 100
  decrease_percent = 8.57 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angle_decrease_l1430_143018


namespace NUMINAMATH_GPT_third_team_cups_l1430_143000

theorem third_team_cups (required_cups : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) :
  required_cups = 280 ∧ first_team = 90 ∧ second_team = 120 →
  third_team = required_cups - (first_team + second_team) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_third_team_cups_l1430_143000


namespace NUMINAMATH_GPT_length_of_third_side_l1430_143093

-- Definitions for sides and perimeter condition
variables (a b : ℕ) (h1 : a = 3) (h2 : b = 10) (p : ℕ) (h3 : p % 6 = 0)
variable (c : ℕ)

-- Definition for the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove the length of the third side
theorem length_of_third_side (h4 : triangle_inequality a b c)
  (h5 : p = a + b + c) : c = 11 :=
sorry

end NUMINAMATH_GPT_length_of_third_side_l1430_143093


namespace NUMINAMATH_GPT_together_complete_days_l1430_143049

-- Define the work rates of x and y
def work_rate_x := (1 : ℚ) / 30
def work_rate_y := (1 : ℚ) / 45

-- Define the combined work rate when x and y work together
def combined_work_rate := work_rate_x + work_rate_y

-- Define the number of days to complete the work together
def days_to_complete_work := 1 / combined_work_rate

-- The theorem we want to prove
theorem together_complete_days : days_to_complete_work = 18 := by
  sorry

end NUMINAMATH_GPT_together_complete_days_l1430_143049


namespace NUMINAMATH_GPT_sum_is_seventeen_l1430_143015

variable (x y : ℕ)

def conditions (x y : ℕ) : Prop :=
  x > y ∧ x - y = 3 ∧ x * y = 56

theorem sum_is_seventeen (x y : ℕ) (h: conditions x y) : x + y = 17 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_seventeen_l1430_143015


namespace NUMINAMATH_GPT_total_length_of_fence_l1430_143066

theorem total_length_of_fence (x : ℝ) (h1 : 2 * x * x = 1250) : 2 * x + 2 * x = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_fence_l1430_143066


namespace NUMINAMATH_GPT_g_at_5_l1430_143048

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 47 * x ^ 2 - 44 * x + 24

theorem g_at_5 : g 5 = 104 := by
  sorry

end NUMINAMATH_GPT_g_at_5_l1430_143048


namespace NUMINAMATH_GPT_pool_students_count_l1430_143007

noncomputable def total_students (total_women : ℕ) (female_students : ℕ) (extra_men : ℕ) (non_student_men : ℕ) : ℕ := 
  let total_men := total_women + extra_men
  let male_students := total_men - non_student_men
  female_students + male_students

theorem pool_students_count
  (total_women : ℕ := 1518)
  (female_students : ℕ := 536)
  (extra_men : ℕ := 525)
  (non_student_men : ℕ := 1257) :
  total_students total_women female_students extra_men non_student_men = 1322 := 
by
  sorry

end NUMINAMATH_GPT_pool_students_count_l1430_143007


namespace NUMINAMATH_GPT_unit_price_in_range_l1430_143053

-- Given definitions and conditions
def Q (x : ℝ) : ℝ := 220 - 2 * x
def f (x : ℝ) : ℝ := x * Q x

-- The desired range for the unit price to maintain a production value of at least 60 million yuan
def valid_unit_price_range (x : ℝ) : Prop := 50 < x ∧ x < 60

-- The main theorem that needs to be proven
theorem unit_price_in_range (x : ℝ) (h₁ : 0 < x) (h₂ : x < 500) (h₃ : f x ≥ 60 * 10^6) : valid_unit_price_range x :=
sorry

end NUMINAMATH_GPT_unit_price_in_range_l1430_143053


namespace NUMINAMATH_GPT_range_of_m_l1430_143019

theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, mx^2 - 6 * m * x + m + 8 ≥ 0) ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1430_143019


namespace NUMINAMATH_GPT_possible_lost_rectangle_area_l1430_143089

theorem possible_lost_rectangle_area (areas : Fin 10 → ℕ) (total_area : ℕ) (h_total : total_area = 65) :
  (∃ (i : Fin 10), (64 = total_area - areas i) ∨ (49 = total_area - areas i)) ↔
  (∃ (i : Fin 10), (areas i = 1) ∨ (areas i = 16)) :=
by
  sorry

end NUMINAMATH_GPT_possible_lost_rectangle_area_l1430_143089


namespace NUMINAMATH_GPT_work_together_days_l1430_143094

noncomputable def A_per_day := 1 / 78
noncomputable def B_per_day := 1 / 39

theorem work_together_days 
  (A : ℝ) (B : ℝ) 
  (hA : A = 1 / 78)
  (hB : B = 1 / 39) : 
  1 / (A + B) = 26 :=
by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_work_together_days_l1430_143094


namespace NUMINAMATH_GPT_factors_of_m_multiples_of_200_l1430_143085

theorem factors_of_m_multiples_of_200 (m : ℕ) (h : m = 2^12 * 3^10 * 5^9) : 
  (∃ k, 200 * k ≤ m ∧ ∃ a b c, k = 2^a * 3^b * 5^c ∧ 3 ≤ a ∧ a ≤ 12 ∧ 2 ≤ c ∧ c ≤ 9 ∧ 0 ≤ b ∧ b ≤ 10) := 
by sorry

end NUMINAMATH_GPT_factors_of_m_multiples_of_200_l1430_143085


namespace NUMINAMATH_GPT_combined_ratio_is_1_l1430_143061

-- Conditions
variables (V1 V2 M1 W1 M2 W2 : ℝ)
variables (x : ℝ)
variables (ratio_volumes ratio_milk_water_v1 ratio_milk_water_v2 : ℝ)

-- Given conditions as hypotheses
-- Condition: V1 / V2 = 3 / 5
-- Hypothesis 1: The volume ratio of the first and second vessels
def volume_ratio : Prop :=
  V1 / V2 = 3 / 5

-- Condition: M1 / W1 = 1 / 2 in first vessel
-- Hypothesis 2: The milk to water ratio in the first vessel
def milk_water_ratio_v1 : Prop :=
  M1 / W1 = 1 / 2

-- Condition: M2 / W2 = 3 / 2 in the second vessel
-- Hypothesis 3: The milk to water ratio in the second vessel
def milk_water_ratio_v2 : Prop :=
  M2 / W2 = 3 / 2

-- Definition: Total volumes of milk and water in the larger vessel
def total_milk_water_ratio : Prop :=
  (M1 + M2) / (W1 + W2) = 1 / 1

-- Main theorem: Given the ratios, the ratio of milk to water in the larger vessel is 1:1
theorem combined_ratio_is_1 :
  (volume_ratio V1 V2) →
  (milk_water_ratio_v1 M1 W1) →
  (milk_water_ratio_v2 M2 W2) →
  total_milk_water_ratio M1 W1 M2 W2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_combined_ratio_is_1_l1430_143061


namespace NUMINAMATH_GPT_sufficient_condition_for_ellipse_l1430_143099

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_ellipse_l1430_143099


namespace NUMINAMATH_GPT_max_quotient_l1430_143050

theorem max_quotient (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) : 
  (∀ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) → y / x ≤ 18) ∧ 
  (∃ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) ∧ y / x = 18) :=
by
  sorry

end NUMINAMATH_GPT_max_quotient_l1430_143050


namespace NUMINAMATH_GPT_remainder_8_pow_1996_mod_5_l1430_143091

theorem remainder_8_pow_1996_mod_5 :
  (8: ℕ) ≡ 3 [MOD 5] →
  3^4 ≡ 1 [MOD 5] →
  8^1996 ≡ 1 [MOD 5] :=
by
  sorry

end NUMINAMATH_GPT_remainder_8_pow_1996_mod_5_l1430_143091


namespace NUMINAMATH_GPT_opposite_of_neg_three_l1430_143004

theorem opposite_of_neg_three : -(-3) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_l1430_143004


namespace NUMINAMATH_GPT_total_boxes_correct_l1430_143027

noncomputable def friday_boxes : ℕ := 40

noncomputable def saturday_boxes : ℕ := 2 * friday_boxes - 10

noncomputable def sunday_boxes : ℕ := saturday_boxes / 2

noncomputable def monday_boxes : ℕ := 
  let extra_boxes := (25 * sunday_boxes + 99) / 100 -- (25/100) * sunday_boxes rounded to nearest integer
  sunday_boxes + extra_boxes

noncomputable def total_boxes : ℕ := 
  friday_boxes + saturday_boxes + sunday_boxes + monday_boxes

theorem total_boxes_correct : total_boxes = 189 := by
  sorry

end NUMINAMATH_GPT_total_boxes_correct_l1430_143027


namespace NUMINAMATH_GPT_quadratic_solution_l1430_143060

theorem quadratic_solution (x : ℝ) : x^2 - 5 * x - 6 = 0 ↔ (x = 6 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l1430_143060


namespace NUMINAMATH_GPT_sum_of_abcd_l1430_143090

theorem sum_of_abcd (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : ∀ x, x^2 - 8*a*x - 9*b = 0 → x = c ∨ x = d)
  (h2 : ∀ x, x^2 - 8*c*x - 9*d = 0 → x = a ∨ x = b) :
  a + b + c + d = 648 := sorry

end NUMINAMATH_GPT_sum_of_abcd_l1430_143090


namespace NUMINAMATH_GPT_vector_at_t5_l1430_143082

theorem vector_at_t5 :
  ∃ (a : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ),
    a + (1 : ℝ) • d = (2, -1, 3) ∧
    a + (4 : ℝ) • d = (8, -5, 11) ∧
    a + (5 : ℝ) • d = (10, -19/3, 41/3) := 
sorry

end NUMINAMATH_GPT_vector_at_t5_l1430_143082


namespace NUMINAMATH_GPT_imaginary_part_z_l1430_143034

open Complex

theorem imaginary_part_z : (im ((i - 1) / (i + 1))) = 1 :=
by
  -- The proof goes here, but it can be marked with sorry for now
  sorry

end NUMINAMATH_GPT_imaginary_part_z_l1430_143034


namespace NUMINAMATH_GPT_quiz_answer_keys_count_l1430_143070

noncomputable def count_answer_keys : ℕ :=
  (Nat.choose 10 5) * (Nat.factorial 6)

theorem quiz_answer_keys_count :
  count_answer_keys = 181440 := 
by
  -- Proof is skipped, using sorry
  sorry

end NUMINAMATH_GPT_quiz_answer_keys_count_l1430_143070


namespace NUMINAMATH_GPT_commission_percentage_l1430_143024

theorem commission_percentage 
  (cost_price : ℝ) (profit_percentage : ℝ) (observed_price : ℝ) (C : ℝ) 
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.10)
  (h3 : observed_price = 19.8) 
  (h4 : 1 + C / 100 = 19.8 / (cost_price * (1 + profit_percentage)))
  : C = 20 := 
by
  sorry

end NUMINAMATH_GPT_commission_percentage_l1430_143024


namespace NUMINAMATH_GPT_width_to_length_ratio_l1430_143076

variable (w : ℕ)

def length := 10
def perimeter := 36

theorem width_to_length_ratio
  (h_perimeter : 2 * w + 2 * length = perimeter) :
  w / length = 4 / 5 :=
by
  -- Skipping proof steps, putting sorry
  sorry

end NUMINAMATH_GPT_width_to_length_ratio_l1430_143076


namespace NUMINAMATH_GPT_calculate_expression_l1430_143079

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end NUMINAMATH_GPT_calculate_expression_l1430_143079


namespace NUMINAMATH_GPT_khalil_total_payment_l1430_143041

def cost_dog := 60
def cost_cat := 40
def cost_parrot := 70
def cost_rabbit := 50

def num_dogs := 25
def num_cats := 45
def num_parrots := 15
def num_rabbits := 10

def total_cost := num_dogs * cost_dog + num_cats * cost_cat + num_parrots * cost_parrot + num_rabbits * cost_rabbit

theorem khalil_total_payment : total_cost = 4850 := by
  sorry

end NUMINAMATH_GPT_khalil_total_payment_l1430_143041


namespace NUMINAMATH_GPT_hose_removal_rate_l1430_143011

theorem hose_removal_rate (w l d : ℝ) (capacity_fraction : ℝ) (drain_time : ℝ) 
  (h_w : w = 60) 
  (h_l : l = 150) 
  (h_d : d = 10) 
  (h_capacity_fraction : capacity_fraction = 0.80) 
  (h_drain_time : drain_time = 1200) : 
  ((w * l * d * capacity_fraction) / drain_time) = 60 :=
by
  -- the proof is omitted here
  sorry

end NUMINAMATH_GPT_hose_removal_rate_l1430_143011


namespace NUMINAMATH_GPT_fish_remaining_l1430_143044

def initial_fish : ℝ := 47.0
def given_away_fish : ℝ := 22.5

theorem fish_remaining : initial_fish - given_away_fish = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_fish_remaining_l1430_143044


namespace NUMINAMATH_GPT_men_apples_l1430_143029

theorem men_apples (M W : ℕ) (h1 : M = W - 20) (h2 : 2 * M + 3 * W = 210) : M = 30 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_men_apples_l1430_143029


namespace NUMINAMATH_GPT_largest_n_for_divisibility_l1430_143086

theorem largest_n_for_divisibility : 
  ∃ n : ℕ, (n + 12 ∣ n^3 + 150) ∧ (∀ m : ℕ, (m + 12 ∣ m^3 + 150) → m ≤ 246) :=
sorry

end NUMINAMATH_GPT_largest_n_for_divisibility_l1430_143086


namespace NUMINAMATH_GPT_find_numbers_l1430_143081

theorem find_numbers (x y : ℤ) (h1 : x > y) (h2 : x^2 - y^2 = 100) : 
  x = 26 ∧ y = 24 := 
  sorry

end NUMINAMATH_GPT_find_numbers_l1430_143081


namespace NUMINAMATH_GPT_triangle_perimeter_l1430_143032

-- Definitions for the conditions
def side_length1 : ℕ := 3
def side_length2 : ℕ := 6
def equation (x : ℤ) := x^2 - 6 * x + 8 = 0

-- Perimeter calculation given the sides form a triangle
theorem triangle_perimeter (x : ℤ) (h₁ : equation x) (h₂ : 3 + 6 > x) (h₃ : 3 + x > 6) (h₄ : 6 + x > 3) :
  3 + 6 + x = 13 :=
by sorry

end NUMINAMATH_GPT_triangle_perimeter_l1430_143032


namespace NUMINAMATH_GPT_time_for_model_M_l1430_143040

variable (T : ℝ) -- Time taken by model M computer to complete the task in minutes.
variable (n_m : ℝ := 12) -- Number of model M computers
variable (n_n : ℝ := 12) -- Number of model N computers
variable (time_n : ℝ := 18) -- Time taken by model N computer to complete the task in minutes

theorem time_for_model_M :
  n_m / T + n_n / time_n = 1 → T = 36 := by
sorry

end NUMINAMATH_GPT_time_for_model_M_l1430_143040


namespace NUMINAMATH_GPT_order_of_p_q_r_l1430_143065

theorem order_of_p_q_r (p q r : ℝ) (h1 : p = Real.sqrt 2) (h2 : q = Real.sqrt 7 - Real.sqrt 3) (h3 : r = Real.sqrt 6 - Real.sqrt 2) :
  p > r ∧ r > q :=
by
  sorry

end NUMINAMATH_GPT_order_of_p_q_r_l1430_143065


namespace NUMINAMATH_GPT_ellipse_equation_l1430_143043

theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) (d_max : ℝ) (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
    (h3 : e = Real.sqrt 3 / 2) (h4 : P = (0, 3 / 2)) (h5 : ∀ P1 : ℝ × ℝ, (P1.1 ^ 2 / a ^ 2 + P1.2 ^ 2 / b ^ 2 = 1) → 
    ∃ P2 : ℝ × ℝ, dist P P2 = d_max ∧ (P2.1 ^ 2 / a ^ 2 + P2.2 ^ 2 / b ^ 2 = 1)) :
  (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x ^ 2 / 4) + y ^ 2 ≤ 1) := by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1430_143043


namespace NUMINAMATH_GPT_poles_needed_l1430_143045

theorem poles_needed (L W : ℕ) (dist : ℕ)
  (hL : L = 90) (hW : W = 40) (hdist : dist = 5) :
  (2 * (L + W)) / dist = 52 :=
by 
  sorry

end NUMINAMATH_GPT_poles_needed_l1430_143045


namespace NUMINAMATH_GPT_lcm_25_35_50_l1430_143026

theorem lcm_25_35_50 : Nat.lcm (Nat.lcm 25 35) 50 = 350 := by
  sorry

end NUMINAMATH_GPT_lcm_25_35_50_l1430_143026


namespace NUMINAMATH_GPT_part1_part2_l1430_143036

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2
noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem part1 : ∃ xₘ : ℝ, (∀ x > 0, f x ≤ f xₘ) ∧ f xₘ = -1 :=
by sorry

theorem part2 (a : ℝ) : (∀ x > 0, f x + g x a ≥ 0) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1430_143036


namespace NUMINAMATH_GPT_slope_of_line_l1430_143059

-- Defining the conditions
def intersects_on_line (s x y : ℝ) : Prop :=
  (2 * x + 3 * y = 8 * s + 6) ∧ (x + 2 * y = 5 * s - 1)

-- Theorem stating that the slope of the line on which all intersections lie is 2
theorem slope_of_line {s x y : ℝ} :
  (∃ s x y, intersects_on_line s x y) → (∃ (m : ℝ), m = 2) :=
by sorry

end NUMINAMATH_GPT_slope_of_line_l1430_143059


namespace NUMINAMATH_GPT_number_of_polynomials_l1430_143002

-- Define conditions
def is_positive_integer (n : ℤ) : Prop :=
  5 * 151 * n > 0

-- Define the main theorem
theorem number_of_polynomials (n : ℤ) (h : is_positive_integer n) : 
  ∃ k : ℤ, k = ⌊n / 2⌋ + 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_polynomials_l1430_143002


namespace NUMINAMATH_GPT_find_n_value_l1430_143095

theorem find_n_value : ∃ n : ℤ, 3^3 - 7 = 4^2 + n ∧ n = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_find_n_value_l1430_143095


namespace NUMINAMATH_GPT_problem1_problem2_l1430_143064

theorem problem1 : (Real.sqrt 24 - Real.sqrt 18) - Real.sqrt 6 = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : 2 * Real.sqrt 12 * Real.sqrt (1 / 8) + 5 * Real.sqrt 2 = Real.sqrt 6 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1430_143064


namespace NUMINAMATH_GPT_total_fruits_l1430_143092

theorem total_fruits (Mike_fruits Matt_fruits Mark_fruits : ℕ)
  (Mike_receives : Mike_fruits = 3)
  (Matt_receives : Matt_fruits = 2 * Mike_fruits)
  (Mark_receives : Mark_fruits = Mike_fruits + Matt_fruits) :
  Mike_fruits + Matt_fruits + Mark_fruits = 18 := by
  sorry

end NUMINAMATH_GPT_total_fruits_l1430_143092


namespace NUMINAMATH_GPT_candidates_appeared_in_each_state_equals_7900_l1430_143074

theorem candidates_appeared_in_each_state_equals_7900 (x : ℝ) (h : 0.07 * x = 0.06 * x + 79) : x = 7900 :=
sorry

end NUMINAMATH_GPT_candidates_appeared_in_each_state_equals_7900_l1430_143074


namespace NUMINAMATH_GPT_solve_system_of_equations_l1430_143097

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1430_143097


namespace NUMINAMATH_GPT_points_five_from_origin_l1430_143077

theorem points_five_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_GPT_points_five_from_origin_l1430_143077


namespace NUMINAMATH_GPT_book_price_increase_percentage_l1430_143025

theorem book_price_increase_percentage :
  let P_original := 300
  let P_new := 480
  (P_new - P_original : ℝ) / P_original * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_book_price_increase_percentage_l1430_143025


namespace NUMINAMATH_GPT_find_a10_l1430_143031

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Given conditions
variables (a : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a2 : a 2 = 2) (h_a6 : a 6 = 10)

-- Goal to prove
theorem find_a10 : a 10 = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_a10_l1430_143031


namespace NUMINAMATH_GPT_find_teacher_age_l1430_143001

/-- Given conditions: 
1. The class initially has 30 students with an average age of 10.
2. One student aged 11 leaves the class.
3. The average age of the remaining 29 students plus the teacher is 11.
Prove that the age of the teacher is 30 years.
-/
theorem find_teacher_age (total_students : ℕ) (avg_age : ℕ) (left_student_age : ℕ) 
  (remaining_avg_age : ℕ) (teacher_age : ℕ) :
  total_students = 30 →
  avg_age = 10 →
  left_student_age = 11 →
  remaining_avg_age = 11 →
  289 + teacher_age = 29 * remaining_avg_age + teacher_age →
  teacher_age = 30 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_find_teacher_age_l1430_143001


namespace NUMINAMATH_GPT_positive_integer_iff_positive_real_l1430_143068

theorem positive_integer_iff_positive_real (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℕ, n > 0 ∧ abs ((x - 2 * abs x) * abs x) / x = n) ↔ x > 0 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_iff_positive_real_l1430_143068


namespace NUMINAMATH_GPT_difference_divisible_by_9_l1430_143080

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end NUMINAMATH_GPT_difference_divisible_by_9_l1430_143080


namespace NUMINAMATH_GPT_meaningful_expression_range_l1430_143030

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1430_143030


namespace NUMINAMATH_GPT_division_and_subtraction_l1430_143006

theorem division_and_subtraction : (23 ^ 11 / 23 ^ 8) - 15 = 12152 := by
  sorry

end NUMINAMATH_GPT_division_and_subtraction_l1430_143006


namespace NUMINAMATH_GPT_count_divisible_by_25_l1430_143008

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end NUMINAMATH_GPT_count_divisible_by_25_l1430_143008


namespace NUMINAMATH_GPT_product_roots_l1430_143084

noncomputable def root1 (x1 : ℝ) : Prop := x1 * Real.log x1 = 2006
noncomputable def root2 (x2 : ℝ) : Prop := x2 * Real.exp x2 = 2006

theorem product_roots (x1 x2 : ℝ) (h1 : root1 x1) (h2 : root2 x2) : x1 * x2 = 2006 := sorry

end NUMINAMATH_GPT_product_roots_l1430_143084


namespace NUMINAMATH_GPT_transmission_prob_correct_transmission_scheme_comparison_l1430_143063

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end NUMINAMATH_GPT_transmission_prob_correct_transmission_scheme_comparison_l1430_143063


namespace NUMINAMATH_GPT_sum_max_min_values_l1430_143087

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 32 / x

theorem sum_max_min_values :
  y 1 = 34 ∧ y 2 = 24 ∧ y 4 = 40 → ((y 4 + y 2) = 64) :=
by
  sorry

end NUMINAMATH_GPT_sum_max_min_values_l1430_143087


namespace NUMINAMATH_GPT_calculate_expression_l1430_143042

theorem calculate_expression :
  ((16^10 / 16^8) ^ 3 * 8 ^ 3) / 2 ^ 9 = 16777216 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1430_143042


namespace NUMINAMATH_GPT_range_of_a_l1430_143072

noncomputable def proof_problem (x : ℝ) (a : ℝ) : Prop :=
  (x^2 - 4*x + 3 < 0) ∧ (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + a < 0)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, proof_problem x a) ↔ a ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1430_143072


namespace NUMINAMATH_GPT_inequality_abc_l1430_143083

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2 * b + 3 * c) ^ 2 / (a ^ 2 + 2 * b ^ 2 + 3 * c ^ 2) ≤ 6 :=
sorry

end NUMINAMATH_GPT_inequality_abc_l1430_143083


namespace NUMINAMATH_GPT_max_blocks_fit_l1430_143028

-- Define the dimensions of the block and the box
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the volumes calculation
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

-- Define the dimensions of the block and the box
def block : Dimensions := { length := 3, width := 1, height := 2 }
def box : Dimensions := { length := 4, width := 3, height := 6 }

-- Prove that the maximum number of blocks that can fit in the box is 12
theorem max_blocks_fit : (volume box) / (volume block) = 12 := by sorry

end NUMINAMATH_GPT_max_blocks_fit_l1430_143028


namespace NUMINAMATH_GPT_find_original_revenue_l1430_143005

variable (currentRevenue : ℝ) (percentageDecrease : ℝ)
noncomputable def originalRevenue (currentRevenue : ℝ) (percentageDecrease : ℝ) : ℝ :=
  currentRevenue / (1 - percentageDecrease)

theorem find_original_revenue (h1 : currentRevenue = 48.0) (h2 : percentageDecrease = 0.3333333333333333) :
  originalRevenue currentRevenue percentageDecrease = 72.0 := by
  rw [h1, h2]
  unfold originalRevenue
  norm_num
  sorry

end NUMINAMATH_GPT_find_original_revenue_l1430_143005
