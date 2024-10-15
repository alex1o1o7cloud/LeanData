import Mathlib

namespace NUMINAMATH_GPT_animals_percentage_monkeys_l2288_228867

theorem animals_percentage_monkeys (initial_monkeys : ℕ) (initial_birds : ℕ) (birds_eaten : ℕ) (final_monkeys : ℕ) (final_birds : ℕ) : 
  initial_monkeys = 6 → 
  initial_birds = 6 → 
  birds_eaten = 2 → 
  final_monkeys = initial_monkeys → 
  final_birds = initial_birds - birds_eaten → 
  (final_monkeys * 100 / (final_monkeys + final_birds) = 60) := 
by intros
   sorry

end NUMINAMATH_GPT_animals_percentage_monkeys_l2288_228867


namespace NUMINAMATH_GPT_irrational_neg_pi_lt_neg_two_l2288_228863

theorem irrational_neg_pi_lt_neg_two (h1 : Irrational π) (h2 : π > 2) : Irrational (-π) ∧ -π < -2 := by
  sorry

end NUMINAMATH_GPT_irrational_neg_pi_lt_neg_two_l2288_228863


namespace NUMINAMATH_GPT_infinite_solutions_l2288_228893

theorem infinite_solutions (a : ℤ) (h_a : a > 1) 
  (h_sol : ∃ x y : ℤ, x^2 - a * y^2 = -1) : 
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ, (f n).fst^2 - a * (f n).snd^2 = -1 :=
sorry

end NUMINAMATH_GPT_infinite_solutions_l2288_228893


namespace NUMINAMATH_GPT_speed_of_second_train_correct_l2288_228879

noncomputable def length_first_train : ℝ := 140 -- in meters
noncomputable def length_second_train : ℝ := 160 -- in meters
noncomputable def time_to_cross : ℝ := 10.799136069114471 -- in seconds
noncomputable def speed_first_train : ℝ := 60 -- in km/hr
noncomputable def speed_second_train : ℝ := 40 -- in km/hr

theorem speed_of_second_train_correct :
  (length_first_train + length_second_train)/time_to_cross - (speed_first_train * (5/18)) = speed_second_train * (5/18) :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_correct_l2288_228879


namespace NUMINAMATH_GPT_equilibrium_table_n_max_l2288_228849

theorem equilibrium_table_n_max (table : Fin 2010 → Fin 2010 → ℕ) :
  (∃ n, ∀ (i j k l : Fin 2010),
      table i j + table k l = table i l + table k j ∧
      ∀ m ≤ n, (m = 0 ∨ m = 1)
  ) → n = 1 ∧ table (Fin.mk 0 (by norm_num)) (Fin.mk 0 (by norm_num)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_equilibrium_table_n_max_l2288_228849


namespace NUMINAMATH_GPT_pencils_count_l2288_228816

theorem pencils_count (P L : ℕ) (h₁ : 6 * P = 5 * L) (h₂ : L = P + 4) : L = 24 :=
by sorry

end NUMINAMATH_GPT_pencils_count_l2288_228816


namespace NUMINAMATH_GPT_sum_of_angles_of_circumscribed_quadrilateral_l2288_228889

theorem sum_of_angles_of_circumscribed_quadrilateral
  (EF GH : ℝ)
  (EF_central_angle : EF = 100)
  (GH_central_angle : GH = 120) :
  (EF / 2 + GH / 2) = 70 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_angles_of_circumscribed_quadrilateral_l2288_228889


namespace NUMINAMATH_GPT_base_b_expression_not_divisible_l2288_228810

theorem base_b_expression_not_divisible 
  (b : ℕ) : 
  (b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) →
  (2 * b^3 - 2 * b^2 + b - 1) % 5 ≠ 0 ↔ (b ≠ 6) :=
by
  sorry

end NUMINAMATH_GPT_base_b_expression_not_divisible_l2288_228810


namespace NUMINAMATH_GPT_pages_written_in_a_year_l2288_228805

def pages_per_friend_per_letter : ℕ := 3
def friends : ℕ := 2
def letters_per_week : ℕ := 2
def weeks_per_year : ℕ := 52

theorem pages_written_in_a_year : 
  (pages_per_friend_per_letter * friends * letters_per_week * weeks_per_year) = 624 :=
by
  sorry

end NUMINAMATH_GPT_pages_written_in_a_year_l2288_228805


namespace NUMINAMATH_GPT_min_value_fraction_l2288_228841

theorem min_value_fraction (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) : 
  ∃ a, (∀ x y, (x - 1 ≥ 0) ∧ (x - y + 1 ≤ 0) ∧ (x + y - 4 ≤ 0) → (x / (y + 1)) ≥ a) ∧ 
      (a = 1 / 4) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l2288_228841


namespace NUMINAMATH_GPT_range_of_a_l2288_228843

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + 4

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = 0 → (f a x = 0 → x > 0)) ↔ a > 3 := sorry

end NUMINAMATH_GPT_range_of_a_l2288_228843


namespace NUMINAMATH_GPT_smallest_sum_l2288_228823

-- First, we define the conditions as assumptions:
def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  2 * y = x + z

def is_geometric_sequence (x y z : ℕ) : Prop :=
  y ^ 2 = x * z

-- Given conditions
variables (A B C D : ℕ)
variables (hABC : is_arithmetic_sequence A B C) (hBCD : is_geometric_sequence B C D)
variables (h_ratio : 4 * C = 7 * B)

-- The main theorem to prove
theorem smallest_sum : A + B + C + D = 97 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l2288_228823


namespace NUMINAMATH_GPT_circle_equation_l2288_228882

-- Definitions for the given conditions
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-1, 1)
def line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 2 = 0

-- Theorem statement for the proof problem
theorem circle_equation :
  ∃ (h k : ℝ), line (h, k) ∧ (h = 1) ∧ (k = 1) ∧
  ((h - 1)^2 + (k - 1)^2 = 4) :=
sorry

end NUMINAMATH_GPT_circle_equation_l2288_228882


namespace NUMINAMATH_GPT_simple_interest_rate_l2288_228850

-- Define the conditions
def S : ℚ := 2500
def P : ℚ := 5000
def T : ℚ := 5

-- Define the proof problem
theorem simple_interest_rate (R : ℚ) (h : S = P * R * T / 100) : R = 10 := by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2288_228850


namespace NUMINAMATH_GPT_smallest_gcd_qr_l2288_228894

theorem smallest_gcd_qr {p q r : ℕ} (hpq : Nat.gcd p q = 300) (hpr : Nat.gcd p r = 450) : 
  ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 150 :=
by
  sorry

end NUMINAMATH_GPT_smallest_gcd_qr_l2288_228894


namespace NUMINAMATH_GPT_eagles_points_l2288_228822

theorem eagles_points (x y : ℕ) (h₁ : x + y = 82) (h₂ : x - y = 18) : y = 32 :=
sorry

end NUMINAMATH_GPT_eagles_points_l2288_228822


namespace NUMINAMATH_GPT_inverse_variation_solution_l2288_228803

theorem inverse_variation_solution :
  ∀ (x y k : ℝ),
    (x * y^3 = k) →
    (∃ k, x = 8 ∧ y = 1 ∧ k = 8) →
    (y = 2 → x = 1) :=
by
  intros x y k h1 h2 hy2
  sorry

end NUMINAMATH_GPT_inverse_variation_solution_l2288_228803


namespace NUMINAMATH_GPT_other_number_l2288_228899

theorem other_number (x : ℕ) (h : 27 + x = 62) : x = 35 :=
by
  sorry

end NUMINAMATH_GPT_other_number_l2288_228899


namespace NUMINAMATH_GPT_find_certain_number_l2288_228892

theorem find_certain_number (x : ℤ) (h : x - 5 = 4) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l2288_228892


namespace NUMINAMATH_GPT_shape_described_by_theta_eq_c_is_plane_l2288_228860

-- Definitions based on conditions in the problem
def spherical_coordinates (ρ θ φ : ℝ) := true

def is_plane_condition (θ c : ℝ) := θ = c

-- Statement to prove
theorem shape_described_by_theta_eq_c_is_plane (c : ℝ) :
  ∀ ρ θ φ : ℝ, spherical_coordinates ρ θ φ → is_plane_condition θ c → "Plane" = "Plane" :=
by sorry

end NUMINAMATH_GPT_shape_described_by_theta_eq_c_is_plane_l2288_228860


namespace NUMINAMATH_GPT_arithmetic_problem_l2288_228813

theorem arithmetic_problem : 987 + 113 - 1000 = 100 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l2288_228813


namespace NUMINAMATH_GPT_vectors_parallel_iff_l2288_228802

-- Define the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

-- Define what it means for two vectors to be parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement that we need to prove
theorem vectors_parallel_iff (m : ℝ) : parallel a (b m) ↔ m = 1 := by
  sorry

end NUMINAMATH_GPT_vectors_parallel_iff_l2288_228802


namespace NUMINAMATH_GPT_remaining_people_l2288_228808

def initial_football_players : ℕ := 13
def initial_cheerleaders : ℕ := 16
def quitting_football_players : ℕ := 10
def quitting_cheerleaders : ℕ := 4

theorem remaining_people :
  (initial_football_players - quitting_football_players) 
  + (initial_cheerleaders - quitting_cheerleaders) = 15 := by
    -- Proof steps would go here, if required
    sorry

end NUMINAMATH_GPT_remaining_people_l2288_228808


namespace NUMINAMATH_GPT_limit_sum_perimeters_l2288_228832

theorem limit_sum_perimeters (a : ℝ) : ∑' n : ℕ, (4 * a) * (1 / 2) ^ n = 8 * a :=
by sorry

end NUMINAMATH_GPT_limit_sum_perimeters_l2288_228832


namespace NUMINAMATH_GPT_solve_equation_l2288_228825

theorem solve_equation (x : ℝ) :
  x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3 * x + 2 > 0 ↔ x = -3 ∨ x = 0 ∨ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2288_228825


namespace NUMINAMATH_GPT_distance_between_towns_l2288_228869

variables (x y z : ℝ)

theorem distance_between_towns
  (h1 : x / 24 + y / 16 + z / 12 = 2)
  (h2 : x / 12 + y / 16 + z / 24 = 2.25) :
  x + y + z = 34 :=
sorry

end NUMINAMATH_GPT_distance_between_towns_l2288_228869


namespace NUMINAMATH_GPT_seconds_in_12_5_minutes_l2288_228880

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end NUMINAMATH_GPT_seconds_in_12_5_minutes_l2288_228880


namespace NUMINAMATH_GPT_friends_attended_birthday_l2288_228887

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_friends_attended_birthday_l2288_228887


namespace NUMINAMATH_GPT_egg_production_difference_l2288_228874

-- Define the conditions
def last_year_production : ℕ := 1416
def this_year_production : ℕ := 4636

-- Define the theorem statement
theorem egg_production_difference :
  this_year_production - last_year_production = 3220 :=
by
  sorry

end NUMINAMATH_GPT_egg_production_difference_l2288_228874


namespace NUMINAMATH_GPT_joe_flight_expense_l2288_228888

theorem joe_flight_expense
  (initial_amount : ℕ)
  (hotel_expense : ℕ)
  (food_expense : ℕ)
  (remaining_amount : ℕ)
  (flight_expense : ℕ)
  (h1 : initial_amount = 6000)
  (h2 : hotel_expense = 800)
  (h3 : food_expense = 3000)
  (h4 : remaining_amount = 1000)
  (h5 : flight_expense = initial_amount - remaining_amount - hotel_expense - food_expense) :
  flight_expense = 1200 :=
by
  sorry

end NUMINAMATH_GPT_joe_flight_expense_l2288_228888


namespace NUMINAMATH_GPT_solve_g_eq_5_l2288_228833

noncomputable def g (x : ℝ) : ℝ :=
if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solve_g_eq_5 : {x : ℝ | g x = 5} = {-3/4, 20/3} :=
by
  sorry

end NUMINAMATH_GPT_solve_g_eq_5_l2288_228833


namespace NUMINAMATH_GPT_clearance_sale_gain_percent_l2288_228829

theorem clearance_sale_gain_percent
  (SP : ℝ := 30)
  (gain_percent : ℝ := 25)
  (discount_percent : ℝ := 10)
  (CP : ℝ := SP/(1 + gain_percent/100)) :
  let Discount := discount_percent / 100 * SP
  let SP_sale := SP - Discount
  let Gain_during_sale := SP_sale - CP
  let Gain_percent_during_sale := (Gain_during_sale / CP) * 100
  Gain_percent_during_sale = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_clearance_sale_gain_percent_l2288_228829


namespace NUMINAMATH_GPT_even_function_is_a_4_l2288_228807

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_is_a_4 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 := by
  sorry

end NUMINAMATH_GPT_even_function_is_a_4_l2288_228807


namespace NUMINAMATH_GPT_price_per_ton_max_tons_l2288_228852

variable (x y m : ℝ)

def conditions := x = y + 100 ∧ 2 * x + y = 1700

theorem price_per_ton (h : conditions x y) : x = 600 ∧ y = 500 :=
  sorry

def budget_conditions := 10 * (600 - 100) + 1 * 500 ≤ 5600

theorem max_tons (h : budget_conditions) : 600 * m + 500 * (10 - m) ≤ 5600 → m ≤ 6 :=
  sorry

end NUMINAMATH_GPT_price_per_ton_max_tons_l2288_228852


namespace NUMINAMATH_GPT_smallest_number_condition_l2288_228864

theorem smallest_number_condition 
  (x : ℕ) 
  (h1 : ∃ k : ℕ, x - 6 = k * 12)
  (h2 : ∃ k : ℕ, x - 6 = k * 16)
  (h3 : ∃ k : ℕ, x - 6 = k * 18)
  (h4 : ∃ k : ℕ, x - 6 = k * 21)
  (h5 : ∃ k : ℕ, x - 6 = k * 28)
  (h6 : ∃ k : ℕ, x - 6 = k * 35)
  (h7 : ∃ k : ℕ, x - 6 = k * 39) 
  : x = 65526 :=
sorry

end NUMINAMATH_GPT_smallest_number_condition_l2288_228864


namespace NUMINAMATH_GPT_new_member_money_l2288_228830

variable (T M : ℝ)
variable (H1 : T / 7 = 20)
variable (H2 : (T + M) / 8 = 14)

theorem new_member_money : M = 756 :=
by
  sorry

end NUMINAMATH_GPT_new_member_money_l2288_228830


namespace NUMINAMATH_GPT_find_k_l2288_228881

theorem find_k (x : ℝ) (k : ℝ) (h : 2 * x - 3 = 3 * x - 2 + k) (h_solution : x = 2) : k = -3 := by
  sorry

end NUMINAMATH_GPT_find_k_l2288_228881


namespace NUMINAMATH_GPT_total_modules_in_stock_l2288_228839

-- Given conditions
def module_cost_high : ℝ := 10
def module_cost_low : ℝ := 3.5
def total_stock_value : ℝ := 45
def low_module_count : ℕ := 10

-- To be proved: total number of modules in stock
theorem total_modules_in_stock (x : ℕ) (y : ℕ) (h1 : y = low_module_count) 
  (h2 : module_cost_high * x + module_cost_low * y = total_stock_value) : 
  x + y = 11 := 
sorry

end NUMINAMATH_GPT_total_modules_in_stock_l2288_228839


namespace NUMINAMATH_GPT_find_m_value_l2288_228855

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def is_perpendicular (v1 v2 : vector) : Prop :=
  dot_product v1 v2 = 0

theorem find_m_value (a b : vector) (m : ℝ) (h: a = (2, -1)) (h2: b = (1, 3))
  (h3: is_perpendicular a (a.1 + m * b.1, a.2 + m * b.2)) : m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_value_l2288_228855


namespace NUMINAMATH_GPT_sum_of_possible_values_of_x_l2288_228811

-- Define the concept of an isosceles triangle with specific angles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the angle sum property of a triangle
def angle_sum_property (a b c : ℝ) : Prop := 
  a + b + c = 180

-- State the problem using the given conditions and the required proof
theorem sum_of_possible_values_of_x :
  ∀ (x : ℝ), 
    is_isosceles_triangle 70 70 x ∨
    is_isosceles_triangle 70 x x ∨
    is_isosceles_triangle x 70 70 →
    angle_sum_property 70 70 x →
    angle_sum_property 70 x x →
    angle_sum_property x 70 70 →
    (70 + 55 + 40) = 165 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_x_l2288_228811


namespace NUMINAMATH_GPT_max_height_piston_l2288_228814

theorem max_height_piston (M a P c_v g R: ℝ) (h : ℝ) 
  (h_pos : 0 < h) (M_pos : 0 < M) (a_pos : 0 < a) (P_pos : 0 < P)
  (c_v_pos : 0 < c_v) (g_pos : 0 < g) (R_pos : 0 < R) :
  h = (2 * P ^ 2) / (M ^ 2 * g * a ^ 2 * (1 + c_v / R) ^ 2) := sorry

end NUMINAMATH_GPT_max_height_piston_l2288_228814


namespace NUMINAMATH_GPT_least_possible_sections_l2288_228815

theorem least_possible_sections (A C N : ℕ) (h1 : 7 * A = 11 * C) (h2 : N = A + C) : N = 18 :=
sorry

end NUMINAMATH_GPT_least_possible_sections_l2288_228815


namespace NUMINAMATH_GPT_number_of_candies_l2288_228838

theorem number_of_candies (n : ℕ) (h1 : 11 ≤ n) (h2 : n ≤ 100) (h3 : n % 18 = 0) (h4 : n % 7 = 1) : n = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_candies_l2288_228838


namespace NUMINAMATH_GPT_shopkeeper_percentage_above_cost_l2288_228868

theorem shopkeeper_percentage_above_cost (CP MP SP : ℚ) 
  (h1 : CP = 100) 
  (h2 : SP = CP * 1.02)
  (h3 : SP = MP * 0.85) : 
  (MP - CP) / CP * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_shopkeeper_percentage_above_cost_l2288_228868


namespace NUMINAMATH_GPT_student_most_stable_l2288_228846

theorem student_most_stable (A B C : ℝ) (hA : A = 0.024) (hB : B = 0.08) (hC : C = 0.015) : C < A ∧ C < B := by
  sorry

end NUMINAMATH_GPT_student_most_stable_l2288_228846


namespace NUMINAMATH_GPT_Gumble_words_total_l2288_228817

noncomputable def num_letters := 25
noncomputable def exclude_B := 24

noncomputable def total_5_letters_or_less (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 5 then num_letters^n - exclude_B^n else 0

noncomputable def total_Gumble_words : ℕ :=
  (total_5_letters_or_less 1) + (total_5_letters_or_less 2) + (total_5_letters_or_less 3) +
  (total_5_letters_or_less 4) + (total_5_letters_or_less 5)

theorem Gumble_words_total :
  total_Gumble_words = 1863701 := by
  sorry

end NUMINAMATH_GPT_Gumble_words_total_l2288_228817


namespace NUMINAMATH_GPT_andrew_paid_total_l2288_228865

-- Define the quantities and rates
def quantity_grapes : ℕ := 14
def rate_grapes : ℕ := 54
def quantity_mangoes : ℕ := 10
def rate_mangoes : ℕ := 62

-- Define the cost calculations
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes
def total_cost : ℕ := cost_grapes + cost_mangoes

-- Prove the total amount paid is as expected
theorem andrew_paid_total : total_cost = 1376 := by
  sorry 

end NUMINAMATH_GPT_andrew_paid_total_l2288_228865


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_S9_l2288_228827

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = n * (a 1 + a n) / 2

-- Problem statement in Lean
theorem arithmetic_sequence_sum_S9 (h_seq : ∃ d, arithmetic_sequence a d) (h_a2 : a 2 = -2) (h_a8 : a 8 = 6) (h_S_def : sum_of_first_n_terms a S) : S 9 = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_sum_S9_l2288_228827


namespace NUMINAMATH_GPT_father_l2288_228837

theorem father's_age : 
  ∀ (M F : ℕ), 
  (M = (2 : ℚ) / 5 * F) → 
  (M + 10 = (1 : ℚ) / 2 * (F + 10)) → 
  F = 50 :=
by
  intros M F h1 h2
  sorry

end NUMINAMATH_GPT_father_l2288_228837


namespace NUMINAMATH_GPT_hyperbola_asymptote_y_eq_1_has_m_neg_3_l2288_228847

theorem hyperbola_asymptote_y_eq_1_has_m_neg_3
    (m : ℝ)
    (h1 : ∀ x y, (x^2 / (2 * m)) - (y^2 / m) = 1)
    (h2 : ∀ x, 1 = (x^2 / (2 * m))): m = -3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_y_eq_1_has_m_neg_3_l2288_228847


namespace NUMINAMATH_GPT_find_a_2016_l2288_228831

-- Given definition for the sequence sum
def sequence_sum (n : ℕ) : ℕ := n * n

-- Definition for a_n using the given sequence sum
def term (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

-- Stating the theorem that we need to prove
theorem find_a_2016 : term 2016 = 4031 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_2016_l2288_228831


namespace NUMINAMATH_GPT_dhoni_spent_300_dollars_l2288_228828

theorem dhoni_spent_300_dollars :
  ∀ (L S X : ℝ),
  L = 6 →
  S = L - 2 →
  (X / S) - (X / L) = 25 →
  X = 300 :=
by
intros L S X hL hS hEquation
sorry

end NUMINAMATH_GPT_dhoni_spent_300_dollars_l2288_228828


namespace NUMINAMATH_GPT_estimate_total_fish_in_pond_l2288_228890

theorem estimate_total_fish_in_pond :
  ∀ (total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample : ℕ),
  initial_sample_size = 100 →
  second_sample_size = 200 →
  tagged_in_second_sample = 10 →
  total_tagged_fish = 100 →
  (total_tagged_fish : ℚ) / (total_fish : ℚ) = tagged_in_second_sample / second_sample_size →
  total_fish = 2000 := by
  intros total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_estimate_total_fish_in_pond_l2288_228890


namespace NUMINAMATH_GPT_problem_solution_l2288_228859

variable {a b x y : ℝ}

-- Define the conditions as Lean assumptions
axiom cond1 : a * x + b * y = 3
axiom cond2 : a * x^2 + b * y^2 = 7
axiom cond3 : a * x^3 + b * y^3 = 16
axiom cond4 : a * x^4 + b * y^4 = 42

-- The main theorem statement: under these conditions, prove a * x^5 + b * y^5 = 99
theorem problem_solution : a * x^5 + b * y^5 = 99 := 
sorry -- proof omitted

end NUMINAMATH_GPT_problem_solution_l2288_228859


namespace NUMINAMATH_GPT_cleaning_task_sequences_correct_l2288_228853

section ChemistryClass

-- Total number of students
def total_students : ℕ := 15

-- Number of classes in a week
def classes_per_week : ℕ := 5

-- Calculate the number of valid sequences of task assignments
def num_valid_sequences : ℕ := total_students * (total_students - 1) * (total_students - 2) * (total_students - 3) * (total_students - 4)

theorem cleaning_task_sequences_correct :
  num_valid_sequences = 360360 :=
by
  unfold num_valid_sequences
  norm_num
  sorry

end ChemistryClass

end NUMINAMATH_GPT_cleaning_task_sequences_correct_l2288_228853


namespace NUMINAMATH_GPT_linear_coefficient_of_quadratic_term_is_negative_five_l2288_228820

theorem linear_coefficient_of_quadratic_term_is_negative_five (a b c : ℝ) (x : ℝ) :
  (2 * x^2 = 5 * x - 3) →
  (a = 2) →
  (b = -5) →
  (c = 3) →
  (a * x^2 + b * x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_linear_coefficient_of_quadratic_term_is_negative_five_l2288_228820


namespace NUMINAMATH_GPT_circle_diameter_l2288_228872

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l2288_228872


namespace NUMINAMATH_GPT_greatest_divisor_l2288_228861

theorem greatest_divisor :
  ∃ x, (∀ y : ℕ, y > 0 → x ∣ (7^y + 12*y - 1)) ∧ (∀ z, (∀ y : ℕ, y > 0 → z ∣ (7^y + 12*y - 1)) → z ≤ x) ∧ x = 18 :=
sorry

end NUMINAMATH_GPT_greatest_divisor_l2288_228861


namespace NUMINAMATH_GPT_sophist_statements_correct_l2288_228840

-- Definitions based on conditions
def num_knights : ℕ := 40
def num_liars : ℕ := 25

-- Statements made by the sophist
def sophist_statement1 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_knights = 40
def sophist_statement2 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_liars + 1 = 26

-- Theorem to be proved
theorem sophist_statements_correct :
  sophist_statement1 ∧ sophist_statement2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sophist_statements_correct_l2288_228840


namespace NUMINAMATH_GPT_min_value_correct_l2288_228870

noncomputable def min_value (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)] : ℝ :=
  if x + y = 1 then (a / x + b / y) else 0

theorem min_value_correct (a b x y : ℝ) [Fact (a > 0)] [Fact (b > 0)] [Fact (x > 0)] [Fact (y > 0)]
  (h : x + y = 1) : min_value a b x y = (Real.sqrt a + Real.sqrt b)^2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_correct_l2288_228870


namespace NUMINAMATH_GPT_maximum_value_of_f_l2288_228834

def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem maximum_value_of_f :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x a ≥ -2) → f 2 a = 25 :=
by
  intro a h
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l2288_228834


namespace NUMINAMATH_GPT_not_all_pieces_found_l2288_228875

theorem not_all_pieces_found (N : ℕ) (petya_tore : ℕ → ℕ) (vasya_tore : ℕ → ℕ) : 
  (∀ n, petya_tore n = n * 5 - n) →
  (∀ n, vasya_tore n = n * 9 - n) →
  1988 = N ∧ (N % 2 = 1) → false :=
by
  intros h_petya h_vasya h
  sorry

end NUMINAMATH_GPT_not_all_pieces_found_l2288_228875


namespace NUMINAMATH_GPT_Nancy_folders_l2288_228895

def n_initial : ℕ := 43
def n_deleted : ℕ := 31
def n_per_folder : ℕ := 6
def n_folders : ℕ := (n_initial - n_deleted) / n_per_folder

theorem Nancy_folders : n_folders = 2 := by
  sorry

end NUMINAMATH_GPT_Nancy_folders_l2288_228895


namespace NUMINAMATH_GPT_rationalize_denominator_l2288_228800

theorem rationalize_denominator : 
  let a := 32
  let b := 8
  let c := 2
  let d := 4
  (a / (c * Real.sqrt c) + b / (d * Real.sqrt c)) = (9 * Real.sqrt c) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2288_228800


namespace NUMINAMATH_GPT_exponentiation_product_rule_l2288_228883

theorem exponentiation_product_rule (a : ℝ) : (3 * a) ^ 2 = 9 * a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_product_rule_l2288_228883


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2288_228854

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → (m = 1) ∨ (m = -2)) → (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → false) :=
by
  intros hm h_perp h
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2288_228854


namespace NUMINAMATH_GPT_functional_eq_solution_l2288_228886

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_GPT_functional_eq_solution_l2288_228886


namespace NUMINAMATH_GPT_heath_plants_per_hour_l2288_228866

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end NUMINAMATH_GPT_heath_plants_per_hour_l2288_228866


namespace NUMINAMATH_GPT_inequality_solution_set_l2288_228856

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2288_228856


namespace NUMINAMATH_GPT_find_p_when_q_is_1_l2288_228821

-- Define the proportionality constant k and the relationship
variables {k p q : ℝ}
def inversely_proportional (k q p : ℝ) : Prop := p = k / (q + 2)

-- Given conditions
theorem find_p_when_q_is_1 (h1 : inversely_proportional k 4 1) : 
  inversely_proportional k 1 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_p_when_q_is_1_l2288_228821


namespace NUMINAMATH_GPT_arithmetic_sequence_n_equals_100_l2288_228877

theorem arithmetic_sequence_n_equals_100
  (a₁ : ℕ) (d : ℕ) (a_n : ℕ)
  (h₁ : a₁ = 1)
  (h₂ : d = 3)
  (h₃ : a_n = 298) :
  ∃ n : ℕ, a_n = a₁ + (n - 1) * d ∧ n = 100 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_equals_100_l2288_228877


namespace NUMINAMATH_GPT_pairs_satisfy_equation_l2288_228835

theorem pairs_satisfy_equation :
  ∀ (x n : ℕ), (x > 0 ∧ n > 0) ∧ 3 * 2 ^ x + 4 = n ^ 2 → (x, n) = (2, 4) ∨ (x, n) = (5, 10) ∨ (x, n) = (6, 14) :=
by
  sorry

end NUMINAMATH_GPT_pairs_satisfy_equation_l2288_228835


namespace NUMINAMATH_GPT_volunteer_assigned_probability_l2288_228873

theorem volunteer_assigned_probability :
  let volunteers := ["A", "B", "C", "D"]
  let areas := ["Beijing", "Zhangjiakou"]
  let total_ways := 14
  let favorable_ways := 6
  ∃ (p : ℚ), p = 6/14 → (1 / total_ways) * favorable_ways = 3/7
:= sorry

end NUMINAMATH_GPT_volunteer_assigned_probability_l2288_228873


namespace NUMINAMATH_GPT_determine_functions_l2288_228826

noncomputable def functional_eq_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x) ^ 2

theorem determine_functions (f : ℝ → ℝ) (h : functional_eq_condition f) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
sorry

end NUMINAMATH_GPT_determine_functions_l2288_228826


namespace NUMINAMATH_GPT_podium_height_l2288_228862

theorem podium_height (l w h : ℝ) (r s : ℝ) (H1 : r = l + h - w) (H2 : s = w + h - l) 
  (Hr : r = 40) (Hs : s = 34) : h = 37 :=
by
  sorry

end NUMINAMATH_GPT_podium_height_l2288_228862


namespace NUMINAMATH_GPT_eiffel_tower_vs_burj_khalifa_l2288_228898

-- Define the heights of the structures
def height_eiffel_tower : ℕ := 324
def height_burj_khalifa : ℕ := 830

-- Define the statement to be proven
theorem eiffel_tower_vs_burj_khalifa :
  height_burj_khalifa - height_eiffel_tower = 506 :=
by
  sorry

end NUMINAMATH_GPT_eiffel_tower_vs_burj_khalifa_l2288_228898


namespace NUMINAMATH_GPT_find_angle_and_perimeter_l2288_228842

open Real

variables {A B C a b c : ℝ}

/-- If (2a - c)sinA + (2c - a)sinC = 2bsinB in triangle ABC -/
theorem find_angle_and_perimeter
  (h1 : (2 * a - c) * sin A + (2 * c - a) * sin C = 2 * b * sin B)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (b_eq : b = 1) :
  B = π / 3 ∧ (sqrt 3 + 1 < a + b + c ∧ a + b + c ≤ 3) :=
sorry

end NUMINAMATH_GPT_find_angle_and_perimeter_l2288_228842


namespace NUMINAMATH_GPT_three_x_minus_five_y_l2288_228871

noncomputable def F : ℝ × ℝ :=
  let D := (15, 3)
  let E := (6, 8)
  ((D.1 + E.1) / 2, (D.2 + E.2) / 2)

theorem three_x_minus_five_y : (3 * F.1 - 5 * F.2) = 4 := by
  sorry

end NUMINAMATH_GPT_three_x_minus_five_y_l2288_228871


namespace NUMINAMATH_GPT_solve_system_of_equations_l2288_228896

theorem solve_system_of_equations (x y : ℚ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : x + 3 * y = 9) : 
  x = 42 / 11 ∧ y = 19 / 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_of_equations_l2288_228896


namespace NUMINAMATH_GPT_sum_opposite_sign_zero_l2288_228836

def opposite_sign (a b : ℝ) : Prop :=
(a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem sum_opposite_sign_zero {a b : ℝ} (h : opposite_sign a b) : a + b = 0 :=
sorry

end NUMINAMATH_GPT_sum_opposite_sign_zero_l2288_228836


namespace NUMINAMATH_GPT_weight_of_dried_grapes_l2288_228897

/-- The weight of dried grapes available from 20 kg of fresh grapes given the water content in fresh and dried grapes. -/
theorem weight_of_dried_grapes (W_fresh W_dried : ℝ) (fresh_weight : ℝ) (weight_dried : ℝ) :
  W_fresh = 0.9 → 
  W_dried = 0.2 → 
  fresh_weight = 20 →
  weight_dried = (0.1 * fresh_weight) / (1 - W_dried) → 
  weight_dried = 2.5 :=
by sorry

end NUMINAMATH_GPT_weight_of_dried_grapes_l2288_228897


namespace NUMINAMATH_GPT_age_double_after_5_years_l2288_228801

-- Defining the current ages of the brothers
def older_brother_age := 15
def younger_brother_age := 5

-- Defining the condition
def after_x_years (x : ℕ) := older_brother_age + x = 2 * (younger_brother_age + x)

-- The main theorem with the condition
theorem age_double_after_5_years : after_x_years 5 :=
by sorry

end NUMINAMATH_GPT_age_double_after_5_years_l2288_228801


namespace NUMINAMATH_GPT_g_g_g_3_eq_71_l2288_228885

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1 else 2 * n + 5

theorem g_g_g_3_eq_71 : g (g (g 3)) = 71 := 
by
  sorry

end NUMINAMATH_GPT_g_g_g_3_eq_71_l2288_228885


namespace NUMINAMATH_GPT_bicycle_final_price_l2288_228857

theorem bicycle_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (h1 : original_price = 200) (h2 : discount1 = 0.4) (h3 : discount2 = 0.2) :
  (original_price * (1 - discount1) * (1 - discount2)) = 96 :=
by
  -- sorry proof here
  sorry

end NUMINAMATH_GPT_bicycle_final_price_l2288_228857


namespace NUMINAMATH_GPT_trig_eqn_solution_l2288_228845

open Real

theorem trig_eqn_solution (x : ℝ) (n : ℤ) :
  sin x ≠ 0 →
  cos x ≠ 0 →
  sin x + cos x ≥ 0 →
  (sqrt (1 + tan x) = sin x + cos x) →
  ∃ k : ℤ, (x = k * π + π / 4) ∨ (x = k * π - π / 4) ∨ (x = (2 * k * π + 3 * π / 4)) :=
by
  sorry

end NUMINAMATH_GPT_trig_eqn_solution_l2288_228845


namespace NUMINAMATH_GPT_p_necessary_for_q_l2288_228844

-- Definitions
def p (a b : ℝ) : Prop := (a + b = 2) ∨ (a + b = -2)
def q (a b : ℝ) : Prop := a + b = 2

-- Statement of the problem
theorem p_necessary_for_q (a b : ℝ) : (p a b → q a b) ∧ ¬(q a b → p a b) := 
sorry

end NUMINAMATH_GPT_p_necessary_for_q_l2288_228844


namespace NUMINAMATH_GPT_total_people_in_church_l2288_228848

def c : ℕ := 80
def m : ℕ := 60
def f : ℕ := 60

theorem total_people_in_church : c + m + f = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_people_in_church_l2288_228848


namespace NUMINAMATH_GPT_multiplicative_inverse_of_550_mod_4319_l2288_228878

theorem multiplicative_inverse_of_550_mod_4319 :
  (48^2 + 275^2 = 277^2) → ((550 * 2208) % 4319 = 1) := by
  intro h
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_of_550_mod_4319_l2288_228878


namespace NUMINAMATH_GPT_triangle_area_eq_e_div_4_l2288_228824

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

noncomputable def tangent_line (x : ℝ) : ℝ :=
  let k := (Real.exp 1) * (x + 1)
  k * (x - 1) + Real.exp 1

theorem triangle_area_eq_e_div_4 :
  let area := (1 / 2) * Real.exp 1 * (1 / 2)
  area = (Real.exp 1) / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_eq_e_div_4_l2288_228824


namespace NUMINAMATH_GPT_max_students_equal_distribution_l2288_228858

-- Define the number of pens and pencils
def pens : ℕ := 1008
def pencils : ℕ := 928

-- Define the problem statement which asks for the GCD of the given numbers
theorem max_students_equal_distribution : Nat.gcd pens pencils = 16 :=
by 
  -- Lean's gcd computation can be used to confirm the result
  sorry

end NUMINAMATH_GPT_max_students_equal_distribution_l2288_228858


namespace NUMINAMATH_GPT_place_circle_no_overlap_l2288_228884

theorem place_circle_no_overlap 
    (rect_width rect_height : ℝ) (num_squares : ℤ) (square_size square_diameter : ℝ)
    (h_rect_dims : rect_width = 20 ∧ rect_height = 25)
    (h_num_squares : num_squares = 120)
    (h_square_size : square_size = 1)
    (h_circle_diameter : square_diameter = 1) : 
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ rect_width ∧ 0 ≤ y ∧ y ≤ rect_height ∧ 
    ∀ (square_x square_y : ℝ), 
      0 ≤ square_x ∧ square_x ≤ rect_width - square_size ∧ 
      0 ≤ square_y ∧ square_y ≤ rect_height - square_size → 
      (x - square_x)^2 + (y - square_y)^2 ≥ (square_diameter / 2)^2 :=
sorry

end NUMINAMATH_GPT_place_circle_no_overlap_l2288_228884


namespace NUMINAMATH_GPT_no_positive_integer_solution_l2288_228851

/-- Let \( p \) be a prime greater than 3 and \( x \) be an integer such that \( p \) divides \( x \).
    Then the equation \( x^2 - 1 = y^p \) has no positive integer solutions for \( y \). -/
theorem no_positive_integer_solution {p x y : ℕ} (hp : Nat.Prime p) (hgt : 3 < p) (hdiv : p ∣ x) :
  ¬∃ y : ℕ, (x^2 - 1 = y^p) ∧ (0 < y) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solution_l2288_228851


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l2288_228809

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : 2 * a + b = 24) : b = 10 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l2288_228809


namespace NUMINAMATH_GPT_weight_of_replaced_person_l2288_228804

theorem weight_of_replaced_person 
  (avg_increase : ℝ) (new_person_weight : ℝ) (n : ℕ) (original_weight : ℝ) 
  (h1 : avg_increase = 2.5)
  (h2 : new_person_weight = 95)
  (h3 : n = 8)
  (h4 : original_weight = new_person_weight - n * avg_increase) : 
  original_weight = 75 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l2288_228804


namespace NUMINAMATH_GPT_onion_rings_cost_l2288_228891

variable (hamburger_cost smoothie_cost total_payment change_received : ℕ)

theorem onion_rings_cost (h_hamburger : hamburger_cost = 4) 
                         (h_smoothie : smoothie_cost = 3) 
                         (h_total_payment : total_payment = 20) 
                         (h_change_received : change_received = 11) :
                         total_payment - change_received - hamburger_cost - smoothie_cost = 2 :=
by
  sorry

end NUMINAMATH_GPT_onion_rings_cost_l2288_228891


namespace NUMINAMATH_GPT_exists_powers_mod_eq_l2288_228812

theorem exists_powers_mod_eq (N : ℕ) (A : ℤ) : ∃ r s : ℕ, r ≠ s ∧ (A ^ r - A ^ s) % N = 0 :=
sorry

end NUMINAMATH_GPT_exists_powers_mod_eq_l2288_228812


namespace NUMINAMATH_GPT_candles_left_in_room_l2288_228876

-- Define the variables and conditions
def total_candles : ℕ := 40
def alyssa_used : ℕ := total_candles / 2
def remaining_candles_after_alyssa : ℕ := total_candles - alyssa_used
def chelsea_used : ℕ := (7 * remaining_candles_after_alyssa) / 10
def final_remaining_candles : ℕ := remaining_candles_after_alyssa - chelsea_used

-- The theorem we need to prove
theorem candles_left_in_room : final_remaining_candles = 6 := by
  sorry

end NUMINAMATH_GPT_candles_left_in_room_l2288_228876


namespace NUMINAMATH_GPT_vector_subtraction_l2288_228818

def a : ℝ × ℝ := (3, 5)
def b : ℝ × ℝ := (-2, 1)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)

theorem vector_subtraction : (a.1 - two_b.1, a.2 - two_b.2) = (7, 3) := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l2288_228818


namespace NUMINAMATH_GPT_squirrel_travel_distance_l2288_228806

theorem squirrel_travel_distance
  (height: ℝ)
  (circumference: ℝ)
  (vertical_rise: ℝ)
  (num_circuits: ℝ):
  height = 25 →
  circumference = 3 →
  vertical_rise = 5 →
  num_circuits = height / vertical_rise →
  (num_circuits * circumference) ^ 2 + height ^ 2 = 850 :=
by
  sorry

end NUMINAMATH_GPT_squirrel_travel_distance_l2288_228806


namespace NUMINAMATH_GPT_path_length_l2288_228819

theorem path_length (scale_ratio : ℕ) (map_path_length : ℝ) 
  (h1 : scale_ratio = 500)
  (h2 : map_path_length = 3.5) : 
  (map_path_length * scale_ratio = 1750) :=
sorry

end NUMINAMATH_GPT_path_length_l2288_228819
