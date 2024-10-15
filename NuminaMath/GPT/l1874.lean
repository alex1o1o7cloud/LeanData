import Mathlib

namespace NUMINAMATH_GPT_complex_eq_sub_l1874_187404

open Complex

theorem complex_eq_sub {a b : ℝ} (h : (a : ℂ) + 2 * I = I * ((b : ℂ) - I)) : a - b = -3 := by
  sorry

end NUMINAMATH_GPT_complex_eq_sub_l1874_187404


namespace NUMINAMATH_GPT_two_pow_2014_mod_seven_l1874_187442

theorem two_pow_2014_mod_seven : 
  ∃ r : ℕ, 2 ^ 2014 ≡ r [MOD 7] → r = 2 :=
sorry

end NUMINAMATH_GPT_two_pow_2014_mod_seven_l1874_187442


namespace NUMINAMATH_GPT_solve_for_x_l1874_187457

theorem solve_for_x (x : ℚ) :
  (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 73 ↔ x = -647 / 177 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1874_187457


namespace NUMINAMATH_GPT_skating_average_l1874_187450

variable (minutesPerDay1 minutesPerDay2 : Nat)
variable (days1 days2 totalDays requiredAverage : Nat)

theorem skating_average :
  minutesPerDay1 = 80 →
  days1 = 6 →
  minutesPerDay2 = 100 →
  days2 = 2 →
  totalDays = 9 →
  requiredAverage = 95 →
  (minutesPerDay1 * days1 + minutesPerDay2 * days2 + x) / totalDays = requiredAverage →
  x = 175 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_skating_average_l1874_187450


namespace NUMINAMATH_GPT_intersection_setA_setB_l1874_187465

def setA := {x : ℝ | |x| < 1}
def setB := {x : ℝ | x^2 - 2 * x ≤ 0}

theorem intersection_setA_setB :
  {x : ℝ | 0 ≤ x ∧ x < 1} = setA ∩ setB :=
by
  sorry

end NUMINAMATH_GPT_intersection_setA_setB_l1874_187465


namespace NUMINAMATH_GPT_average_people_per_hour_l1874_187484

theorem average_people_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) (average_per_hour : ℕ) :
  total_people = 3000 ∧ days = 5 ∧ hours_per_day = 24 ∧ total_hours = days * hours_per_day ∧ average_per_hour = total_people / total_hours → 
  average_per_hour = 25 :=
by
  sorry

end NUMINAMATH_GPT_average_people_per_hour_l1874_187484


namespace NUMINAMATH_GPT_probability_of_not_red_l1874_187440

-- Definitions based on conditions
def total_number_of_jelly_beans : ℕ := 7 + 9 + 10 + 12 + 5
def number_of_non_red_jelly_beans : ℕ := 9 + 10 + 12 + 5

-- Proving the probability
theorem probability_of_not_red : 
  (number_of_non_red_jelly_beans : ℚ) / total_number_of_jelly_beans = 36 / 43 :=
by sorry

end NUMINAMATH_GPT_probability_of_not_red_l1874_187440


namespace NUMINAMATH_GPT_additional_telephone_lines_l1874_187436

theorem additional_telephone_lines :
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  let additional_lines := lines_seven_digits - lines_six_digits
  additional_lines = 81 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_additional_telephone_lines_l1874_187436


namespace NUMINAMATH_GPT_problem1_problem2_l1874_187443

variable (α : ℝ)

-- First problem statement
theorem problem1 (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6 / 13 :=
by 
  sorry

-- Second problem statement
theorem problem2 (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 16 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1874_187443


namespace NUMINAMATH_GPT_min_colors_correct_l1874_187497

def min_colors (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 2
  else 3

theorem min_colors_correct (n : Nat) : min_colors n = 
  if n = 1 then 1
  else if n = 2 then 2
  else 3 := by
  sorry

end NUMINAMATH_GPT_min_colors_correct_l1874_187497


namespace NUMINAMATH_GPT_final_rider_is_C_l1874_187456

def initial_order : List Char := ['A', 'B', 'C']

def leader_changes : Nat := 19
def third_place_changes : Nat := 17

def B_finishes_third (final_order: List Char) : Prop :=
  final_order.get! 2 = 'B'

def total_transpositions (a b : Nat) : Nat :=
  a + b

theorem final_rider_is_C (final_order: List Char) :
  B_finishes_third final_order →
  total_transpositions leader_changes third_place_changes % 2 = 0 →
  final_order = ['C', 'A', 'B'] → 
  final_order.get! 0 = 'C' :=
by
  sorry

end NUMINAMATH_GPT_final_rider_is_C_l1874_187456


namespace NUMINAMATH_GPT_fare_range_l1874_187458

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 8 else 8 + 1.5 * (x - 3)

theorem fare_range (x : ℝ) (hx : fare x = 16) : 8 ≤ x ∧ x < 9 :=
by
  sorry

end NUMINAMATH_GPT_fare_range_l1874_187458


namespace NUMINAMATH_GPT_probability_of_red_second_given_red_first_l1874_187499

-- Define the conditions as per the problem.
def total_balls := 5
def red_balls := 3
def yellow_balls := 2
def first_draw_red : ℚ := (red_balls : ℚ) / (total_balls : ℚ)
def both_draws_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of drawing a red ball in the second draw given the first was red.
def conditional_probability_red_second_given_first : ℚ :=
  both_draws_red / first_draw_red

-- The main statement to be proved.
theorem probability_of_red_second_given_red_first :
  conditional_probability_red_second_given_first = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_red_second_given_red_first_l1874_187499


namespace NUMINAMATH_GPT_problem_statement_l1874_187494

namespace GeometricRelations

variables {Line Plane : Type} [Nonempty Line] [Nonempty Plane]

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Given conditions
variables (m n : Line) (α β : Plane)

-- The theorem to be proven
theorem problem_statement 
  (h1 : perpendicular m β) 
  (h2 : parallel α β) : 
  perpendicular m α :=
sorry

end GeometricRelations

end NUMINAMATH_GPT_problem_statement_l1874_187494


namespace NUMINAMATH_GPT_gcd_division_steps_l1874_187412

theorem gcd_division_steps (a b : ℕ) (h₁ : a = 1813) (h₂ : b = 333) : 
  ∃ steps : ℕ, steps = 3 ∧ (Nat.gcd a b = 37) :=
by
  have h₁ : a = 1813 := h₁
  have h₂ : b = 333 := h₂
  sorry

end NUMINAMATH_GPT_gcd_division_steps_l1874_187412


namespace NUMINAMATH_GPT_unique_zero_function_l1874_187433

theorem unique_zero_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_zero_function_l1874_187433


namespace NUMINAMATH_GPT_percent_increase_salary_l1874_187434

theorem percent_increase_salary (new_salary increase : ℝ) (h_new_salary : new_salary = 90000) (h_increase : increase = 25000) :
  (increase / (new_salary - increase)) * 100 = 38.46 := by
  -- Given values
  have h1 : new_salary = 90000 := h_new_salary
  have h2 : increase = 25000 := h_increase
  -- Compute original salary
  let original_salary : ℝ := new_salary - increase
  -- Compute percent increase
  let percent_increase : ℝ := (increase / original_salary) * 100
  -- Show that the percent increase is 38.46
  have h3 : percent_increase = 38.46 := sorry
  exact h3

end NUMINAMATH_GPT_percent_increase_salary_l1874_187434


namespace NUMINAMATH_GPT_remaining_dimes_l1874_187464

-- Define the initial quantity of dimes Joan had
def initial_dimes : Nat := 5

-- Define the quantity of dimes Joan spent
def dimes_spent : Nat := 2

-- State the theorem we need to prove
theorem remaining_dimes : initial_dimes - dimes_spent = 3 := by
  sorry

end NUMINAMATH_GPT_remaining_dimes_l1874_187464


namespace NUMINAMATH_GPT_laura_needs_to_buy_flour_l1874_187422

/--
Laura is baking a cake and needs to buy ingredients.
Flour costs $4, sugar costs $2, butter costs $2.5, and eggs cost $0.5.
The cake is cut into 6 slices. Her mother ate 2 slices.
The dog ate the remaining cake, costing $6.
Prove that Laura needs to buy flour worth $4.
-/
theorem laura_needs_to_buy_flour
  (flour_cost sugar_cost butter_cost eggs_cost dog_ate_cost : ℝ)
  (cake_slices mother_ate_slices dog_ate_slices : ℕ)
  (H_flour : flour_cost = 4)
  (H_sugar : sugar_cost = 2)
  (H_butter : butter_cost = 2.5)
  (H_eggs : eggs_cost = 0.5)
  (H_dog_ate : dog_ate_cost = 6)
  (total_slices : cake_slices = 6)
  (mother_slices : mother_ate_slices = 2)
  (dog_slices : dog_ate_slices = 4) :
  flour_cost = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_laura_needs_to_buy_flour_l1874_187422


namespace NUMINAMATH_GPT_employee_pay_l1874_187491

theorem employee_pay (y : ℝ) (x : ℝ) (h1 : x = 1.2 * y) (h2 : x + y = 700) : y = 318.18 :=
by
  sorry

end NUMINAMATH_GPT_employee_pay_l1874_187491


namespace NUMINAMATH_GPT_find_k_l1874_187420

theorem find_k (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + (-1) * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1874_187420


namespace NUMINAMATH_GPT_coordinates_of_D_l1874_187488
-- Importing the necessary library

-- Defining the conditions as given in the problem
def AB : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (-1, 3)
def CD : ℝ × ℝ := (2 * 5, 2 * 3)

-- The target proof statement
theorem coordinates_of_D :
  ∃ D : ℝ × ℝ, CD = D - C ∧ D = (9, -3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_D_l1874_187488


namespace NUMINAMATH_GPT_betty_height_correct_l1874_187468

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end NUMINAMATH_GPT_betty_height_correct_l1874_187468


namespace NUMINAMATH_GPT_timmy_needs_speed_l1874_187489

variable (s1 s2 s3 : ℕ) (extra_speed : ℕ)

theorem timmy_needs_speed
  (h_s1 : s1 = 36)
  (h_s2 : s2 = 34)
  (h_s3 : s3 = 38)
  (h_extra_speed : extra_speed = 4) :
  (s1 + s2 + s3) / 3 + extra_speed = 40 := 
sorry

end NUMINAMATH_GPT_timmy_needs_speed_l1874_187489


namespace NUMINAMATH_GPT_tangent_and_normal_are_correct_at_point_l1874_187437

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

def tangent_line (x y : ℝ) : Prop :=
  2*x - 7*y + 19 = 0

def normal_line (x y : ℝ) : Prop :=
  7*x + 2*y - 13 = 0

theorem tangent_and_normal_are_correct_at_point
  (hx : point_on_curve 1 3) :
  tangent_line 1 3 ∧ normal_line 1 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_and_normal_are_correct_at_point_l1874_187437


namespace NUMINAMATH_GPT_family_gathering_total_people_l1874_187475

theorem family_gathering_total_people (P : ℕ) 
  (h1 : P / 2 = 10) : 
  P = 20 := by
  sorry

end NUMINAMATH_GPT_family_gathering_total_people_l1874_187475


namespace NUMINAMATH_GPT_soda_cost_l1874_187473

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_l1874_187473


namespace NUMINAMATH_GPT_most_significant_price_drop_l1874_187421

noncomputable def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => -1.00
  | 2 => 0.50
  | 3 => -3.00
  | 4 => 2.00
  | 5 => -1.50
  | 6 => -0.75
  | _ => 0.00 -- For any invalid month, we assume no price change

theorem most_significant_price_drop :
  ∀ m : ℕ, (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) →
  (∀ n : ℕ, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) →
  price_change m ≤ price_change n) → m = 3 :=
by
  intros m hm H
  sorry

end NUMINAMATH_GPT_most_significant_price_drop_l1874_187421


namespace NUMINAMATH_GPT_pqrs_product_l1874_187493

theorem pqrs_product :
  let P := (Real.sqrt 2010 + Real.sqrt 2009 + Real.sqrt 2008)
  let Q := (-Real.sqrt 2010 - Real.sqrt 2009 + Real.sqrt 2008)
  let R := (Real.sqrt 2010 - Real.sqrt 2009 - Real.sqrt 2008)
  let S := (-Real.sqrt 2010 + Real.sqrt 2009 - Real.sqrt 2008)
  P * Q * R * S = 1 := by
{
  sorry -- Proof is omitted as per the provided instructions.
}

end NUMINAMATH_GPT_pqrs_product_l1874_187493


namespace NUMINAMATH_GPT_sum_ab_eq_five_l1874_187424

theorem sum_ab_eq_five (a b : ℕ) (h : (∃ (ab : ℕ), ab = a * 10 + b ∧ 3 / 13 = ab / 100)) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_sum_ab_eq_five_l1874_187424


namespace NUMINAMATH_GPT_hyperbola_equation_l1874_187474

theorem hyperbola_equation {x y : ℝ} (h1 : x ^ 2 / 2 - y ^ 2 = 1) 
  (h2 : x = -2) (h3 : y = 2) : y ^ 2 / 2 - x ^ 2 / 4 = 1 :=
by sorry

end NUMINAMATH_GPT_hyperbola_equation_l1874_187474


namespace NUMINAMATH_GPT_glass_bowls_sold_l1874_187482

theorem glass_bowls_sold
  (BowlsBought : ℕ) (CostPricePerBowl SellingPricePerBowl : ℝ) (PercentageGain : ℝ)
  (CostPrice := BowlsBought * CostPricePerBowl)
  (SellingPrice : ℝ := (102 : ℝ) * SellingPricePerBowl)
  (gain := (SellingPrice - CostPrice) / CostPrice * 100) :
  PercentageGain = 8.050847457627118 →
  BowlsBought = 118 →
  CostPricePerBowl = 12 →
  SellingPricePerBowl = 15 →
  PercentageGain = gain →
  102 = 102 := by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_glass_bowls_sold_l1874_187482


namespace NUMINAMATH_GPT_total_sum_of_ages_is_correct_l1874_187444

-- Definition of conditions
def ageOfYoungestChild : Nat := 4
def intervals : Nat := 3

-- Total sum calculation
def sumOfAges (ageOfYoungestChild intervals : Nat) :=
  let Y := ageOfYoungestChild
  Y + (Y + intervals) + (Y + 2 * intervals) + (Y + 3 * intervals) + (Y + 4 * intervals)

theorem total_sum_of_ages_is_correct : sumOfAges 4 3 = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_sum_of_ages_is_correct_l1874_187444


namespace NUMINAMATH_GPT_train_pass_bridge_in_approx_26_64_sec_l1874_187459

noncomputable def L_train : ℝ := 240 -- Length of the train in meters
noncomputable def L_bridge : ℝ := 130 -- Length of the bridge in meters
noncomputable def Speed_train_kmh : ℝ := 50 -- Speed of the train in km/h
noncomputable def Speed_train_ms : ℝ := (Speed_train_kmh * 1000) / 3600 -- Speed of the train in m/s
noncomputable def Total_distance : ℝ := L_train + L_bridge -- Total distance to be covered by the train
noncomputable def Time : ℝ := Total_distance / Speed_train_ms -- Time to pass the bridge

theorem train_pass_bridge_in_approx_26_64_sec : |Time - 26.64| < 0.01 := by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_in_approx_26_64_sec_l1874_187459


namespace NUMINAMATH_GPT_range_of_b_over_a_l1874_187461

noncomputable def f (a b x : ℝ) : ℝ := (a * x - b / x - 2 * a) * Real.exp x

noncomputable def f' (a b x : ℝ) : ℝ := (b / x^2 + a * x - b / x - a) * Real.exp x

theorem range_of_b_over_a (a b : ℝ) (h₀ : a > 0) (h₁ : ∃ x : ℝ, 1 < x ∧ f a b x + f' a b x = 0) : 
  -1 < b / a := sorry

end NUMINAMATH_GPT_range_of_b_over_a_l1874_187461


namespace NUMINAMATH_GPT_remaining_course_distance_l1874_187410

def total_distance_km : ℝ := 10.5
def distance_to_break_km : ℝ := 1.5
def additional_distance_m : ℝ := 3730.0

theorem remaining_course_distance :
  let total_distance_m := total_distance_km * 1000
  let distance_to_break_m := distance_to_break_km * 1000
  let total_traveled_m := distance_to_break_m + additional_distance_m
  total_distance_m - total_traveled_m = 5270 := by
  sorry

end NUMINAMATH_GPT_remaining_course_distance_l1874_187410


namespace NUMINAMATH_GPT_proposition_D_correct_l1874_187425

theorem proposition_D_correct :
  ∀ x : ℝ, x^2 + x + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_proposition_D_correct_l1874_187425


namespace NUMINAMATH_GPT_condition_a_neither_necessary_nor_sufficient_for_b_l1874_187447

theorem condition_a_neither_necessary_nor_sufficient_for_b {x y : ℝ} (h : ¬(x = 1 ∧ y = 2)) (k : ¬(x + y = 3)) : ¬((x ≠ 1 ∧ y ≠ 2) ↔ (x + y ≠ 3)) :=
by
  sorry

end NUMINAMATH_GPT_condition_a_neither_necessary_nor_sufficient_for_b_l1874_187447


namespace NUMINAMATH_GPT_popsicle_sticks_difference_l1874_187409

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end NUMINAMATH_GPT_popsicle_sticks_difference_l1874_187409


namespace NUMINAMATH_GPT_cookie_percentage_increase_l1874_187495

theorem cookie_percentage_increase (cookies_Monday cookies_Tuesday cookies_Wednesday total_cookies : ℕ) 
  (h1 : cookies_Monday = 5)
  (h2 : cookies_Tuesday = 2 * cookies_Monday)
  (h3 : total_cookies = cookies_Monday + cookies_Tuesday + cookies_Wednesday)
  (h4 : total_cookies = 29) :
  (100 * (cookies_Wednesday - cookies_Tuesday) / cookies_Tuesday = 40) := 
by
  sorry

end NUMINAMATH_GPT_cookie_percentage_increase_l1874_187495


namespace NUMINAMATH_GPT_find_angle_x_l1874_187403

-- Definitions as conditions from the problem statement
def angle_PQR := 120
def angle_PQS (x : ℝ) := 2 * x
def angle_QRS (x : ℝ) := x

-- The theorem to prove
theorem find_angle_x (x : ℝ) (h1 : angle_PQR = 120) (h2 : angle_PQS x + angle_QRS x = angle_PQR) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_x_l1874_187403


namespace NUMINAMATH_GPT_simplify_fraction_l1874_187400

theorem simplify_fraction (a b : ℤ) (h : a = 2^6 + 2^4) (h1 : b = 2^5 - 2^2) : 
  (a / b : ℚ) = 20 / 7 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1874_187400


namespace NUMINAMATH_GPT_complement_intersection_l1874_187432

open Set

-- Definitions of sets
def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- The theorem statement
theorem complement_intersection :
  (U \ B) ∩ A = {2, 6} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1874_187432


namespace NUMINAMATH_GPT_arith_seq_a4_a10_l1874_187498

variable {a : ℕ → ℕ}
axiom hp1 : a 1 + a 2 + a 3 = 32
axiom hp2 : a 11 + a 12 + a 13 = 118

theorem arith_seq_a4_a10 :
  a 4 + a 10 = 50 :=
by
  have h1 : a 2 = 32 / 3 := sorry
  have h2 : a 12 = 118 / 3 := sorry
  have h3 : a 2 + a 12 = 50 := sorry
  exact sorry

end NUMINAMATH_GPT_arith_seq_a4_a10_l1874_187498


namespace NUMINAMATH_GPT_remainder_when_divided_by_19_l1874_187455

theorem remainder_when_divided_by_19 {N : ℤ} (h : N % 342 = 47) : N % 19 = 9 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_19_l1874_187455


namespace NUMINAMATH_GPT_find_width_l1874_187429

namespace RectangleProblem

variables {w l : ℝ}

-- Conditions
def length_is_three_times_width (w l : ℝ) : Prop := l = 3 * w
def sum_of_length_and_width_equals_three_times_area (w l : ℝ) : Prop := l + w = 3 * (l * w)

-- Theorem statement
theorem find_width (w l : ℝ) (h1 : length_is_three_times_width w l) (h2 : sum_of_length_and_width_equals_three_times_area w l) :
  w = 4 / 9 :=
sorry

end RectangleProblem

end NUMINAMATH_GPT_find_width_l1874_187429


namespace NUMINAMATH_GPT_inequality_solution_set_l1874_187419

theorem inequality_solution_set (x : ℝ) : 4 * x^2 - 4 * x + 1 ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1874_187419


namespace NUMINAMATH_GPT_dirk_profit_l1874_187418

theorem dirk_profit 
  (days : ℕ) 
  (amulets_per_day : ℕ) 
  (sale_price : ℕ) 
  (cost_price : ℕ) 
  (cut_percentage : ℕ) 
  (profit : ℕ) : 
  days = 2 → amulets_per_day = 25 → sale_price = 40 → cost_price = 30 → cut_percentage = 10 → profit = 300 :=
by
  intros h_days h_amulets_per_day h_sale_price h_cost_price h_cut_percentage
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_dirk_profit_l1874_187418


namespace NUMINAMATH_GPT_number_of_two_legged_birds_l1874_187427

theorem number_of_two_legged_birds
  (b m i : ℕ)  -- Number of birds (b), mammals (m), and insects (i)
  (h_heads : b + m + i = 300)  -- Condition on total number of heads
  (h_legs : 2 * b + 4 * m + 6 * i = 980)  -- Condition on total number of legs
  : b = 110 :=
by
  sorry

end NUMINAMATH_GPT_number_of_two_legged_birds_l1874_187427


namespace NUMINAMATH_GPT_Bennett_sales_l1874_187445

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end NUMINAMATH_GPT_Bennett_sales_l1874_187445


namespace NUMINAMATH_GPT_double_root_values_l1874_187486

theorem double_root_values (b₃ b₂ b₁ s : ℤ) (h : ∀ x : ℤ, (x * (x - s)) ∣ (x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 36)) 
  : s = -6 ∨ s = -3 ∨ s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 6 :=
sorry

end NUMINAMATH_GPT_double_root_values_l1874_187486


namespace NUMINAMATH_GPT_angles_equal_sixty_degrees_l1874_187430

/-- Given a triangle ABC with sides a, b, c and respective angles α, β, γ, and with circumradius R,
if the following equation holds:
    (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R),
prove that α = β = γ = 60 degrees. -/
theorem angles_equal_sixty_degrees 
  (a b c R : ℝ) 
  (α β γ : ℝ) 
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R)) :
  α = 60 ∧ β = 60 ∧ γ = 60 := 
sorry

end NUMINAMATH_GPT_angles_equal_sixty_degrees_l1874_187430


namespace NUMINAMATH_GPT_opposite_numbers_reciprocal_values_l1874_187439

theorem opposite_numbers_reciprocal_values (a b m n : ℝ) (h₁ : a + b = 0) (h₂ : m * n = 1) : 5 * a + 5 * b - m * n = -1 :=
by sorry

end NUMINAMATH_GPT_opposite_numbers_reciprocal_values_l1874_187439


namespace NUMINAMATH_GPT_smallest_x_for_perfect_cube_l1874_187452

theorem smallest_x_for_perfect_cube (x N : ℕ) (hN : 1260 * x = N^3) (h_fact : 1260 = 2^2 * 3^2 * 5 * 7): x = 7350 := sorry

end NUMINAMATH_GPT_smallest_x_for_perfect_cube_l1874_187452


namespace NUMINAMATH_GPT_bookseller_original_cost_l1874_187460

theorem bookseller_original_cost
  (x y z : ℝ)
  (h1 : 1.10 * x = 11.00)
  (h2 : 1.10 * y = 16.50)
  (h3 : 1.10 * z = 24.20) :
  x + y + z = 47.00 := by
  sorry

end NUMINAMATH_GPT_bookseller_original_cost_l1874_187460


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1874_187470

theorem rhombus_diagonal_length (area d1 d2 : ℝ) (h₁ : area = 24) (h₂ : d1 = 8) (h₃ : area = (d1 * d2) / 2) : d2 = 6 := 
by sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1874_187470


namespace NUMINAMATH_GPT_votes_cast_l1874_187402

theorem votes_cast (V : ℝ) (h1 : ∃ (x : ℝ), x = 0.35 * V) (h2 : ∃ (y : ℝ), y = x + 2100) : V = 7000 :=
by sorry

end NUMINAMATH_GPT_votes_cast_l1874_187402


namespace NUMINAMATH_GPT_crayons_count_l1874_187431

theorem crayons_count
  (crayons_given : Nat := 563)
  (crayons_lost : Nat := 558)
  (crayons_left : Nat := 332) :
  crayons_given + crayons_lost + crayons_left = 1453 := 
sorry

end NUMINAMATH_GPT_crayons_count_l1874_187431


namespace NUMINAMATH_GPT_cats_not_eating_cheese_or_tuna_l1874_187438

-- Define the given conditions
variables (n C T B : ℕ)

-- State the problem in Lean
theorem cats_not_eating_cheese_or_tuna 
  (h_n : n = 100)  
  (h_C : C = 25)  
  (h_T : T = 70)  
  (h_B : B = 15)
  : n - (C - B + T - B + B) = 20 := 
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_cats_not_eating_cheese_or_tuna_l1874_187438


namespace NUMINAMATH_GPT_percentage_of_import_tax_l1874_187417

noncomputable def total_value : ℝ := 2560
noncomputable def taxable_threshold : ℝ := 1000
noncomputable def import_tax : ℝ := 109.20

theorem percentage_of_import_tax :
  let excess_value := total_value - taxable_threshold
  let percentage_tax := (import_tax / excess_value) * 100
  percentage_tax = 7 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_import_tax_l1874_187417


namespace NUMINAMATH_GPT_Earl_owes_Fred_l1874_187485

-- Define initial amounts of money each person has
def Earl_initial : ℤ := 90
def Fred_initial : ℤ := 48
def Greg_initial : ℤ := 36

-- Define debts
def Fred_owes_Greg : ℤ := 32
def Greg_owes_Earl : ℤ := 40

-- Define the total money Greg and Earl have together after debts are settled
def Greg_Earl_total_after_debts : ℤ := 130

-- Define the final amounts after debts are settled
def Earl_final (E : ℤ) : ℤ := Earl_initial - E + Greg_owes_Earl
def Fred_final (E : ℤ) : ℤ := Fred_initial + E - Fred_owes_Greg
def Greg_final : ℤ := Greg_initial + Fred_owes_Greg - Greg_owes_Earl

-- Prove that the total money Greg and Earl have together after debts are settled is 130
theorem Earl_owes_Fred (E : ℤ) (H : Greg_final + Earl_final E = Greg_Earl_total_after_debts) : E = 28 := 
by sorry

end NUMINAMATH_GPT_Earl_owes_Fred_l1874_187485


namespace NUMINAMATH_GPT_inflation_two_years_real_rate_of_return_l1874_187415

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end NUMINAMATH_GPT_inflation_two_years_real_rate_of_return_l1874_187415


namespace NUMINAMATH_GPT_greatest_ln_2_l1874_187471

theorem greatest_ln_2 (x1 x2 x3 x4 : ℝ) (h1 : x1 = (Real.log 2) ^ 2) (h2 : x2 = Real.log (Real.log 2)) (h3 : x3 = Real.log (Real.sqrt 2)) (h4 : x4 = Real.log 2) 
  (h5 : Real.log 2 < 1) : 
  x4 = max x1 (max x2 (max x3 x4)) := by 
  sorry

end NUMINAMATH_GPT_greatest_ln_2_l1874_187471


namespace NUMINAMATH_GPT_system_solution_l1874_187407
-- importing the Mathlib library

-- define the problem with necessary conditions
theorem system_solution (x y : ℝ → ℝ) (x0 y0 : ℝ) 
    (h1 : ∀ t, deriv x t = y t) 
    (h2 : ∀ t, deriv y t = -x t) 
    (h3 : x 0 = x0)
    (h4 : y 0 = y0):
    (∀ t, x t = x0 * Real.cos t + y0 * Real.sin t) ∧ (∀ t, y t = -x0 * Real.sin t + y0 * Real.cos t) ∧ (∀ t, (x t)^2 + (y t)^2 = x0^2 + y0^2) := 
by 
    sorry

end NUMINAMATH_GPT_system_solution_l1874_187407


namespace NUMINAMATH_GPT_find_number_l1874_187426

theorem find_number (x : ℕ) (h : (537 - x) / (463 + x) = 1 / 9) : x = 437 :=
sorry

end NUMINAMATH_GPT_find_number_l1874_187426


namespace NUMINAMATH_GPT_solve_equation_l1874_187414

theorem solve_equation (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1874_187414


namespace NUMINAMATH_GPT_money_weed_eating_l1874_187472

-- Define the amounts and conditions
def money_mowing : ℕ := 68
def money_per_week : ℕ := 9
def weeks : ℕ := 9
def total_money : ℕ := money_per_week * weeks

-- Define the proof that the money made weed eating is 13 dollars
theorem money_weed_eating :
  total_money - money_mowing = 13 := sorry

end NUMINAMATH_GPT_money_weed_eating_l1874_187472


namespace NUMINAMATH_GPT_range_of_a_l1874_187480

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0) 
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1874_187480


namespace NUMINAMATH_GPT_max_value_x_y3_z4_l1874_187449

theorem max_value_x_y3_z4 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  x + y^3 + z^4 ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_x_y3_z4_l1874_187449


namespace NUMINAMATH_GPT_routes_from_Bristol_to_Carlisle_l1874_187405

-- Given conditions as definitions
def routes_Bristol_to_Birmingham : ℕ := 8
def routes_Birmingham_to_Manchester : ℕ := 5
def routes_Manchester_to_Sheffield : ℕ := 4
def routes_Sheffield_to_Newcastle : ℕ := 3
def routes_Newcastle_to_Carlisle : ℕ := 2

-- Define the total number of routes from Bristol to Carlisle
def total_routes_Bristol_to_Carlisle : ℕ := routes_Bristol_to_Birmingham *
                                            routes_Birmingham_to_Manchester *
                                            routes_Manchester_to_Sheffield *
                                            routes_Sheffield_to_Newcastle *
                                            routes_Newcastle_to_Carlisle

-- The theorem to be proved
theorem routes_from_Bristol_to_Carlisle :
  total_routes_Bristol_to_Carlisle = 960 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_routes_from_Bristol_to_Carlisle_l1874_187405


namespace NUMINAMATH_GPT_solve_equation_l1874_187413

theorem solve_equation : ∀ (x : ℝ), 2 * (x - 1) = 2 - (5 * x - 2) → x = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1874_187413


namespace NUMINAMATH_GPT_find_number_l1874_187478

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_number_l1874_187478


namespace NUMINAMATH_GPT_average_of_w_x_z_l1874_187477

theorem average_of_w_x_z (w x z y a : ℝ) (h1 : 2 / w + 2 / x + 2 / z = 2 / y)
  (h2 : w * x * z = y) (h3 : w + x + z = a) : (w + x + z) / 3 = a / 3 :=
by sorry

end NUMINAMATH_GPT_average_of_w_x_z_l1874_187477


namespace NUMINAMATH_GPT_min_k_l_sum_l1874_187401

theorem min_k_l_sum (k l : ℕ) (hk : 120 * k = l^3) (hpos_k : k > 0) (hpos_l : l > 0) :
  k + l = 255 :=
sorry

end NUMINAMATH_GPT_min_k_l_sum_l1874_187401


namespace NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l1874_187446

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 0 < x) (h : x + 1/x = Real.sqrt 2020) : x^2 + 1/x^2 = 2018 :=
sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l1874_187446


namespace NUMINAMATH_GPT_lcm_48_180_value_l1874_187454

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end NUMINAMATH_GPT_lcm_48_180_value_l1874_187454


namespace NUMINAMATH_GPT_man_is_older_by_20_l1874_187463

variables (M S : ℕ)
axiom h1 : S = 18
axiom h2 : M + 2 = 2 * (S + 2)

theorem man_is_older_by_20 :
  M - S = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_man_is_older_by_20_l1874_187463


namespace NUMINAMATH_GPT_complement_correct_l1874_187492

-- Define the universal set U
def U : Set ℤ := {x | -2 < x ∧ x ≤ 3}

-- Define the set A
def A : Set ℤ := {3}

-- Define the complement of A with respect to U
def complement_U_A : Set ℤ := {x | x ∈ U ∧ x ∉ A}

theorem complement_correct : complement_U_A = { -1, 0, 1, 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_correct_l1874_187492


namespace NUMINAMATH_GPT_horner_v4_at_2_l1874_187451

def horner (a : List Int) (x : Int) : Int :=
  a.foldr (fun ai acc => ai + x * acc) 0

noncomputable def poly_coeffs : List Int := [1, -12, 60, -160, 240, -192, 64]

theorem horner_v4_at_2 : horner poly_coeffs 2 = 80 := by
  sorry

end NUMINAMATH_GPT_horner_v4_at_2_l1874_187451


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l1874_187490

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x + 1)) ↔ x ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l1874_187490


namespace NUMINAMATH_GPT_find_m_l1874_187487

theorem find_m (m : ℝ) : (1 : ℝ) * (-4 : ℝ) + (2 : ℝ) * m = 0 → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1874_187487


namespace NUMINAMATH_GPT_total_tbs_of_coffee_l1874_187466

theorem total_tbs_of_coffee (guests : ℕ) (weak_drinkers : ℕ) (medium_drinkers : ℕ) (strong_drinkers : ℕ) 
                           (cups_per_weak_drinker : ℕ) (cups_per_medium_drinker : ℕ) (cups_per_strong_drinker : ℕ) 
                           (tbsp_per_cup_weak : ℕ) (tbsp_per_cup_medium : ℝ) (tbsp_per_cup_strong : ℕ) :
  guests = 18 ∧ 
  weak_drinkers = 6 ∧ 
  medium_drinkers = 6 ∧ 
  strong_drinkers = 6 ∧ 
  cups_per_weak_drinker = 2 ∧ 
  cups_per_medium_drinker = 3 ∧ 
  cups_per_strong_drinker = 1 ∧ 
  tbsp_per_cup_weak = 1 ∧ 
  tbsp_per_cup_medium = 1.5 ∧ 
  tbsp_per_cup_strong = 2 →
  (weak_drinkers * cups_per_weak_drinker * tbsp_per_cup_weak + 
   medium_drinkers * cups_per_medium_drinker * tbsp_per_cup_medium + 
   strong_drinkers * cups_per_strong_drinker * tbsp_per_cup_strong) = 51 :=
by
  sorry

end NUMINAMATH_GPT_total_tbs_of_coffee_l1874_187466


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l1874_187423

theorem product_of_repeating_decimal (p : ℝ) (h : p = 0.6666666666666667) : p * 6 = 4 :=
sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_l1874_187423


namespace NUMINAMATH_GPT_second_experimental_point_is_correct_l1874_187428

-- Define the temperature range
def lower_bound : ℝ := 1400
def upper_bound : ℝ := 1600

-- Define the golden ratio constant
def golden_ratio : ℝ := 0.618

-- Calculate the first experimental point using 0.618 method
def first_point : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

-- Calculate the second experimental point
def second_point : ℝ := upper_bound - (first_point - lower_bound)

-- Theorem stating the calculated second experimental point equals 1476.4
theorem second_experimental_point_is_correct :
  second_point = 1476.4 := by
  sorry

end NUMINAMATH_GPT_second_experimental_point_is_correct_l1874_187428


namespace NUMINAMATH_GPT_marco_total_time_l1874_187469

def marco_run_time (laps distance1 distance2 speed1 speed2 : ℕ ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  laps * (time1 + time2)

theorem marco_total_time :
  marco_run_time 7 150 350 3 4 = 962.5 :=
by
  sorry

end NUMINAMATH_GPT_marco_total_time_l1874_187469


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1874_187406

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1874_187406


namespace NUMINAMATH_GPT_points_above_y_eq_x_l1874_187441

theorem points_above_y_eq_x (x y : ℝ) : (y > x) → (y, x) ∈ {p : ℝ × ℝ | p.2 < p.1} :=
by
  intro h
  sorry

end NUMINAMATH_GPT_points_above_y_eq_x_l1874_187441


namespace NUMINAMATH_GPT_sqrt_of_neg_five_squared_l1874_187496

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_of_neg_five_squared_l1874_187496


namespace NUMINAMATH_GPT_sum_x_y_eq_8_l1874_187411

theorem sum_x_y_eq_8 (x y S : ℝ) (h1 : x + y = S) (h2 : y - 3 * x = 7) (h3 : y - x = 7.5) : S = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_y_eq_8_l1874_187411


namespace NUMINAMATH_GPT_sanda_exercise_each_day_l1874_187416

def exercise_problem (javier_exercise_daily sanda_exercise_total total_minutes : ℕ) (days_in_week : ℕ) :=
  javier_exercise_daily * days_in_week + sanda_exercise_total = total_minutes

theorem sanda_exercise_each_day 
  (javier_exercise_daily : ℕ := 50)
  (days_in_week : ℕ := 7)
  (total_minutes : ℕ := 620)
  (days_sanda_exercised : ℕ := 3): 
  ∃ (sanda_exercise_each_day : ℕ), exercise_problem javier_exercise_daily (sanda_exercise_each_day * days_sanda_exercised) total_minutes days_in_week → sanda_exercise_each_day = 90 :=
by 
  sorry

end NUMINAMATH_GPT_sanda_exercise_each_day_l1874_187416


namespace NUMINAMATH_GPT_correct_sum_l1874_187479

theorem correct_sum (a b c n : ℕ) (h_m_pos : 100 * a + 10 * b + c > 0) (h_n_pos : n > 0)
    (h_err_sum : 100 * a + 10 * c + b + n = 128) : 100 * a + 10 * b + c + n = 128 := 
by
  sorry

end NUMINAMATH_GPT_correct_sum_l1874_187479


namespace NUMINAMATH_GPT_total_coronavirus_cases_l1874_187448

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end NUMINAMATH_GPT_total_coronavirus_cases_l1874_187448


namespace NUMINAMATH_GPT_greatest_two_digit_product_12_l1874_187453

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end NUMINAMATH_GPT_greatest_two_digit_product_12_l1874_187453


namespace NUMINAMATH_GPT_possible_new_perimeters_l1874_187462

theorem possible_new_perimeters
  (initial_tiles := 8)
  (initial_shape := "L")
  (initial_perimeter := 12)
  (additional_tiles := 2)
  (new_perimeters := [12, 14, 16]) :
  True := sorry

end NUMINAMATH_GPT_possible_new_perimeters_l1874_187462


namespace NUMINAMATH_GPT_average_ab_l1874_187435

theorem average_ab {a b : ℝ} (h : (3 + 5 + 7 + a + b) / 5 = 15) : (a + b) / 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_ab_l1874_187435


namespace NUMINAMATH_GPT_vlad_taller_than_sister_l1874_187467

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end NUMINAMATH_GPT_vlad_taller_than_sister_l1874_187467


namespace NUMINAMATH_GPT_john_gallons_of_gas_l1874_187481

theorem john_gallons_of_gas
  (rental_cost : ℝ)
  (gas_cost_per_gallon : ℝ)
  (mile_cost : ℝ)
  (miles_driven : ℝ)
  (total_cost : ℝ)
  (rental_cost_val : rental_cost = 150)
  (gas_cost_per_gallon_val : gas_cost_per_gallon = 3.50)
  (mile_cost_val : mile_cost = 0.50)
  (miles_driven_val : miles_driven = 320)
  (total_cost_val : total_cost = 338) :
  ∃ gallons_of_gas : ℝ, gallons_of_gas = 8 :=
by
  sorry

end NUMINAMATH_GPT_john_gallons_of_gas_l1874_187481


namespace NUMINAMATH_GPT_mike_picked_12_pears_l1874_187476

theorem mike_picked_12_pears
  (jason_pears : ℕ)
  (keith_pears : ℕ)
  (total_pears : ℕ)
  (H1 : jason_pears = 46)
  (H2 : keith_pears = 47)
  (H3 : total_pears = 105) :
  (total_pears - (jason_pears + keith_pears)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_mike_picked_12_pears_l1874_187476


namespace NUMINAMATH_GPT_compute_b_l1874_187408

open Real

theorem compute_b
  (a : ℚ) 
  (b : ℚ) 
  (h₀ : (3 + sqrt 5) ^ 3 + a * (3 + sqrt 5) ^ 2 + b * (3 + sqrt 5) + 12 = 0) 
  : b = -14 :=
sorry

end NUMINAMATH_GPT_compute_b_l1874_187408


namespace NUMINAMATH_GPT_jesse_remaining_pages_l1874_187483

theorem jesse_remaining_pages (pages_read : ℕ)
  (h1 : pages_read = 83)
  (h2 : pages_read = (1 / 3 : ℝ) * total_pages)
  : pages_remaining = 166 :=
  by 
    -- Here we would build the proof, skipped with sorry
    sorry

end NUMINAMATH_GPT_jesse_remaining_pages_l1874_187483
