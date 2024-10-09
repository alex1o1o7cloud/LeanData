import Mathlib

namespace poly_diff_independent_of_x_l1520_152083

theorem poly_diff_independent_of_x (x y: ℤ) (m n : ℤ) 
  (h1 : (1 - n = 0)) 
  (h2 : (m + 3 = 0)) :
  n - m = 4 := by
  sorry

end poly_diff_independent_of_x_l1520_152083


namespace joe_initial_paint_l1520_152074

noncomputable def total_paint (P : ℕ) : Prop :=
  let used_first_week := (1 / 4 : ℚ) * P
  let remaining_after_first := (3 / 4 : ℚ) * P
  let used_second_week := (1 / 6 : ℚ) * remaining_after_first
  let total_used := used_first_week + used_second_week
  total_used = 135

theorem joe_initial_paint (P : ℕ) (h : total_paint P) : P = 463 :=
sorry

end joe_initial_paint_l1520_152074


namespace poly_square_of_binomial_l1520_152013

theorem poly_square_of_binomial (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := 
by 
  sorry

end poly_square_of_binomial_l1520_152013


namespace tony_initial_amount_l1520_152087

-- Define the initial amount P
variable (P : ℝ)

-- Define the conditions
def initial_amount := P
def after_first_year := 1.20 * P
def after_half_taken := 0.60 * P
def after_second_year := 0.69 * P
def final_amount : ℝ := 690

-- State the theorem to prove
theorem tony_initial_amount : 
  (after_second_year P = final_amount) → (initial_amount P = 1000) :=
by 
  intro h
  sorry

end tony_initial_amount_l1520_152087


namespace necessary_but_not_sufficient_l1520_152030

theorem necessary_but_not_sufficient (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≠ b) : ab > 0 :=
  sorry

end necessary_but_not_sufficient_l1520_152030


namespace percentage_alcohol_final_l1520_152027

-- Let's define the given conditions
variable (A B totalVolume : ℝ)
variable (percentAlcoholA percentAlcoholB : ℝ)
variable (approxA : ℝ)

-- Assume the conditions
axiom condition1 : percentAlcoholA = 0.20
axiom condition2 : percentAlcoholB = 0.50
axiom condition3 : totalVolume = 15
axiom condition4 : approxA = 10
axiom condition5 : A = approxA
axiom condition6 : B = totalVolume - A

-- The proof statement
theorem percentage_alcohol_final : 
  (0.20 * A + 0.50 * B) / 15 * 100 = 30 :=
by 
  -- Introduce enough structure for Lean to handle the problem.
  sorry

end percentage_alcohol_final_l1520_152027


namespace intersection_points_count_l1520_152091

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) := sorry

end intersection_points_count_l1520_152091


namespace nth_equation_l1520_152077

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = (n - 1) * 10 + 1 :=
sorry

end nth_equation_l1520_152077


namespace HCF_a_b_LCM_a_b_l1520_152034

-- Given the HCF condition
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Given numbers
def a : ℕ := 210
def b : ℕ := 286

-- Given HCF condition
theorem HCF_a_b : HCF a b = 26 := by
  sorry

-- LCM definition based on the product and HCF
def LCM (a b : ℕ) : ℕ := (a * b) / HCF a b

-- Theorem to prove
theorem LCM_a_b : LCM a b = 2310 := by
  sorry

end HCF_a_b_LCM_a_b_l1520_152034


namespace gcd_48_180_l1520_152060

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  have f1 : 48 = 2^4 * 3 := by norm_num
  have f2 : 180 = 2^2 * 3^2 * 5 := by norm_num
  sorry

end gcd_48_180_l1520_152060


namespace partition_555_weights_l1520_152061

theorem partition_555_weights :
  ∃ A B C : Finset ℕ, 
  (∀ x ∈ A, x ∈ Finset.range (555 + 1)) ∧ 
  (∀ y ∈ B, y ∈ Finset.range (555 + 1)) ∧ 
  (∀ z ∈ C, z ∈ Finset.range (555 + 1)) ∧ 
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
  A ∪ B ∪ C = Finset.range (555 + 1) ∧ 
  A.sum id = 51430 ∧ B.sum id = 51430 ∧ C.sum id = 51430 := sorry

end partition_555_weights_l1520_152061


namespace intersection_point_of_lines_l1520_152092

noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 4

noncomputable def line2 (x : ℝ) : ℝ := -1 / 3 * x + 10 / 3

def point : ℝ × ℝ := (4, 2)

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), line1 x = y ∧ line2 x = y ∧ (x, y) = (2.2, 2.6) :=
by
  sorry

end intersection_point_of_lines_l1520_152092


namespace problem_1_problem_2_l1520_152099

-- Problem I
theorem problem_1 (x : ℝ) (h : |x - 2| + |x - 1| < 4) : (-1/2 : ℝ) < x ∧ x < 7/2 :=
sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 1| ≥ 2) : a ≤ -1 ∨ a ≥ 3 :=
sorry

end problem_1_problem_2_l1520_152099


namespace dice_number_divisible_by_7_l1520_152043

theorem dice_number_divisible_by_7 :
  ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) 
               ∧ (1001 * (100 * a + 10 * b + c)) % 7 = 0 :=
by
  sorry

end dice_number_divisible_by_7_l1520_152043


namespace red_ball_probability_l1520_152071

theorem red_ball_probability : 
  let red_A := 2
  let white_A := 3
  let red_B := 4
  let white_B := 1
  let total_A := red_A + white_A
  let total_B := red_B + white_B
  let prob_red_A := red_A / total_A
  let prob_white_A := white_A / total_A
  let prob_red_B_after_red_A := (red_B + 1) / (total_B + 1)
  let prob_red_B_after_white_A := red_B / (total_B + 1)
  (prob_red_A * prob_red_B_after_red_A + prob_white_A * prob_red_B_after_white_A) = 11 / 15 :=
by {
  sorry
}

end red_ball_probability_l1520_152071


namespace no_conf_of_7_points_and_7_lines_l1520_152080

theorem no_conf_of_7_points_and_7_lines (points : Fin 7 → Prop) (lines : Fin 7 → (Fin 7 → Prop)) :
  (∀ p : Fin 7, ∃ l₁ l₂ l₃ : Fin 7, lines l₁ p ∧ lines l₂ p ∧ lines l₃ p ∧ l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
  (∀ l : Fin 7, ∃ p₁ p₂ p₃ : Fin 7, lines l p₁ ∧ lines l p₂ ∧ lines l p₃ ∧ p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃) 
  → false :=
by
  sorry

end no_conf_of_7_points_and_7_lines_l1520_152080


namespace trig_identity_sum_l1520_152055

-- Define the trigonometric functions and their properties
def sin_210_eq : Real.sin (210 * Real.pi / 180) = - Real.sin (30 * Real.pi / 180) := by
  sorry

def cos_60_eq : Real.cos (60 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) := by
  sorry

-- The goal is to prove that the sum of these specific trigonometric values is 0
theorem trig_identity_sum : Real.sin (210 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) = 0 := by
  rw [sin_210_eq, cos_60_eq]
  sorry

end trig_identity_sum_l1520_152055


namespace correct_assignment_l1520_152001

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l1520_152001


namespace frustum_small_cone_height_is_correct_l1520_152026

noncomputable def frustum_small_cone_height (altitude : ℝ) 
                                             (lower_base_area : ℝ) 
                                             (upper_base_area : ℝ) : ℝ :=
  let r1 := Real.sqrt (lower_base_area / Real.pi)
  let r2 := Real.sqrt (upper_base_area / Real.pi)
  let H := 2 * altitude
  altitude

theorem frustum_small_cone_height_is_correct 
  (altitude : ℝ)
  (lower_base_area : ℝ)
  (upper_base_area : ℝ)
  (h1 : altitude = 16)
  (h2 : lower_base_area = 196 * Real.pi)
  (h3 : upper_base_area = 49 * Real.pi ) : 
  frustum_small_cone_height altitude lower_base_area upper_base_area = 16 := by
  sorry

end frustum_small_cone_height_is_correct_l1520_152026


namespace sum_mod_17_l1520_152014

theorem sum_mod_17 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 17 = 2 :=
by
  sorry

end sum_mod_17_l1520_152014


namespace total_fuel_proof_l1520_152084

def highway_consumption_60 : ℝ := 3 -- gallons per mile at 60 mph
def highway_consumption_70 : ℝ := 3.5 -- gallons per mile at 70 mph
def city_consumption_30 : ℝ := 5 -- gallons per mile at 30 mph
def city_consumption_15 : ℝ := 4.5 -- gallons per mile at 15 mph

def day1_highway_60_hours : ℝ := 2 -- hours driven at 60 mph on the highway
def day1_highway_70_hours : ℝ := 1 -- hours driven at 70 mph on the highway
def day1_city_30_hours : ℝ := 4 -- hours driven at 30 mph in the city

def day2_highway_70_hours : ℝ := 3 -- hours driven at 70 mph on the highway
def day2_city_15_hours : ℝ := 3 -- hours driven at 15 mph in the city
def day2_city_30_hours : ℝ := 1 -- hours driven at 30 mph in the city

def day3_highway_60_hours : ℝ := 1.5 -- hours driven at 60 mph on the highway
def day3_city_30_hours : ℝ := 3 -- hours driven at 30 mph in the city
def day3_city_15_hours : ℝ := 1 -- hours driven at 15 mph in the city

def total_fuel_consumption (c1 c2 c3 c4 : ℝ) (h1 h2 h3 h4 h5 h6 h7 h8 h9 : ℝ) :=
  (h1 * 60 * c1) + (h2 * 70 * c2) + (h3 * 30 * c3) + 
  (h4 * 70 * c2) + (h5 * 15 * c4) + (h6 * 30 * c3) +
  (h7 * 60 * c1) + (h8 * 30 * c3) + (h9 * 15 * c4)

theorem total_fuel_proof :
  total_fuel_consumption highway_consumption_60 highway_consumption_70 city_consumption_30 city_consumption_15
  day1_highway_60_hours day1_highway_70_hours day1_city_30_hours day2_highway_70_hours
  day2_city_15_hours day2_city_30_hours day3_highway_60_hours day3_city_30_hours day3_city_15_hours
  = 3080 := by
  sorry

end total_fuel_proof_l1520_152084


namespace least_number_to_add_l1520_152046

theorem least_number_to_add (n : ℕ) (H : n = 433124) : ∃ k, k = 15 ∧ (n + k) % 17 = 0 := by
  sorry

end least_number_to_add_l1520_152046


namespace number_of_correct_judgments_is_zero_l1520_152045

theorem number_of_correct_judgments_is_zero :
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧
  (¬ ∀ (x y : ℚ), -x = y → y = 1 / x) ∧
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) →
  0 = 0 :=
by
  intros h
  exact rfl

end number_of_correct_judgments_is_zero_l1520_152045


namespace solve_exponential_diophantine_equation_l1520_152031

theorem solve_exponential_diophantine_equation :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 → (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by {
  sorry
}

end solve_exponential_diophantine_equation_l1520_152031


namespace find_x_for_f_of_one_fourth_l1520_152012

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then 2^(-x) else Real.log x / Real.log 4 

-- Define the proof problem
theorem find_x_for_f_of_one_fourth : 
  ∃ x : ℝ, (f x = 1 / 4) ∧ (x = Real.sqrt 2)  :=
sorry

end find_x_for_f_of_one_fourth_l1520_152012


namespace gcd_stamps_pages_l1520_152057

def num_stamps_book1 : ℕ := 924
def num_stamps_book2 : ℕ := 1200

theorem gcd_stamps_pages : Nat.gcd num_stamps_book1 num_stamps_book2 = 12 := by
  sorry

end gcd_stamps_pages_l1520_152057


namespace necessary_but_not_sufficient_l1520_152063

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧ (∃ x : ℝ, |x| ≥ 1 ∧ ¬ (x > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l1520_152063


namespace cookies_on_first_plate_l1520_152007

theorem cookies_on_first_plate :
  ∃ a1 a2 a3 a4 a5 a6 : ℤ, 
  a2 = 7 ∧ 
  a3 = 10 ∧
  a4 = 14 ∧
  a5 = 19 ∧
  a6 = 25 ∧
  a2 = a1 + 2 ∧ 
  a3 = a2 + 3 ∧ 
  a4 = a3 + 4 ∧ 
  a5 = a4 + 5 ∧ 
  a6 = a5 + 6 ∧ 
  a1 = 5 :=
sorry

end cookies_on_first_plate_l1520_152007


namespace sphere_surface_area_from_box_l1520_152098

/--
Given a rectangular box with length = 2, width = 2, and height = 1,
prove that if all vertices of the rectangular box lie on the surface of a sphere,
then the surface area of the sphere is 9π.
--/
theorem sphere_surface_area_from_box :
  let length := 2
  let width := 2
  let height := 1
  ∃ (r : ℝ), ∀ (d := Real.sqrt (length^2 + width^2 + height^2)),
  r = d / 2 → 4 * Real.pi * r^2 = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_from_box_l1520_152098


namespace kendra_more_buttons_l1520_152073

theorem kendra_more_buttons {K M S : ℕ} (hM : M = 8) (hS : S = 22) (hHalfK : S = K / 2) :
  K - 5 * M = 4 :=
by
  sorry

end kendra_more_buttons_l1520_152073


namespace range_of_m_l1520_152010

open Set

noncomputable def M (m : ℝ) : Set ℝ := {x | x ≤ m}
noncomputable def N : Set ℝ := {y | y ≥ 1}

theorem range_of_m (m : ℝ) : M m ∩ N = ∅ → m < 1 := by
  intros h
  sorry

end range_of_m_l1520_152010


namespace net_rate_of_pay_l1520_152086

theorem net_rate_of_pay
  (hours_travelled : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℝ)
  (price_per_gallon : ℝ)
  (net_rate_of_pay : ℝ) :
  hours_travelled = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  price_per_gallon = 2.50 →
  net_rate_of_pay = 25 := by
  sorry

end net_rate_of_pay_l1520_152086


namespace cubic_eq_solutions_l1520_152082

theorem cubic_eq_solutions (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∀ x, x^3 + a * x^2 + b * x + c = 0 → (x = a ∨ x = -b ∨ x = c)) : (a, b, c) = (1, -1, -1) := 
by {
  -- Convert solution steps into a proof
  sorry
}

end cubic_eq_solutions_l1520_152082


namespace bugs_eat_total_flowers_l1520_152058

def num_bugs : ℝ := 2.0
def flowers_per_bug : ℝ := 1.5
def total_flowers_eaten : ℝ := 3.0

theorem bugs_eat_total_flowers : 
  (num_bugs * flowers_per_bug) = total_flowers_eaten := 
  by 
    sorry

end bugs_eat_total_flowers_l1520_152058


namespace total_selling_price_l1520_152068

theorem total_selling_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
    (h1 : original_price = 80) (h2 : discount_rate = 0.25) (h3 : tax_rate = 0.10) :
  let discount_amt := original_price * discount_rate
  let sale_price := original_price - discount_amt
  let tax_amt := sale_price * tax_rate
  let total_price := sale_price + tax_amt
  total_price = 66 := by
  sorry

end total_selling_price_l1520_152068


namespace max_red_tiles_l1520_152009

theorem max_red_tiles (n : ℕ) (color : ℕ → ℕ → color) :
    (∀ i j, color i j ≠ color (i + 1) j ∧ color i j ≠ color i (j + 1) ∧ color i j ≠ color (i + 1) (j + 1) 
           ∧ color i j ≠ color (i - 1) j ∧ color i j ≠ color i (j - 1) ∧ color i j ≠ color (i - 1) (j - 1)) 
    → ∃ m ≤ 2500, ∀ i j, (color i j = red ↔ i * n + j < m) :=
sorry

end max_red_tiles_l1520_152009


namespace problem_l1520_152097

def f (x : ℝ) := 5 * x^3

theorem problem : f 2012 + f (-2012) = 0 := 
by
  sorry

end problem_l1520_152097


namespace problem_conditions_l1520_152039

def G (m : ℕ) : ℕ := m % 10

theorem problem_conditions (a b c : ℕ) (non_neg_m : ∀ m : ℕ, 0 ≤ m) :
  -- Condition ①
  ¬ (G (a - b) = G a - G b) ∧
  -- Condition ②
  (a - b = 10 * c → G a = G b) ∧
  -- Condition ③
  (G (a * b * c) = G (G a * G b * G c)) ∧
  -- Condition ④
  ¬ (G (3^2015) = 9) :=
by sorry

end problem_conditions_l1520_152039


namespace total_drink_ounces_l1520_152053

def total_ounces_entire_drink (coke_parts sprite_parts md_parts coke_ounces : ℕ) : ℕ :=
  let total_parts := coke_parts + sprite_parts + md_parts
  let ounces_per_part := coke_ounces / coke_parts
  total_parts * ounces_per_part

theorem total_drink_ounces (coke_parts sprite_parts md_parts coke_ounces : ℕ) (coke_cond : coke_ounces = 8) (parts_cond : coke_parts = 4 ∧ sprite_parts = 2 ∧ md_parts = 5) : 
  total_ounces_entire_drink coke_parts sprite_parts md_parts coke_ounces = 22 :=
by
  sorry

end total_drink_ounces_l1520_152053


namespace combination_value_l1520_152065

theorem combination_value (m : ℕ) (h : (1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m))) : 
    Nat.choose 8 m = 28 := 
sorry

end combination_value_l1520_152065


namespace age_problem_l1520_152050

theorem age_problem :
  ∃ (x y z : ℕ), 
    x - y = 3 ∧
    z = 2 * x + 2 * y - 3 ∧
    z = x + y + 20 ∧
    x = 13 ∧
    y = 10 ∧
    z = 43 :=
by 
  sorry

end age_problem_l1520_152050


namespace largest_integral_x_l1520_152054

theorem largest_integral_x (x : ℤ) : (2 / 7 : ℝ) < (x / 6) ∧ (x / 6) < (7 / 9) → x = 4 :=
by
  sorry

end largest_integral_x_l1520_152054


namespace no_solutions_abs_eq_quadratic_l1520_152005

theorem no_solutions_abs_eq_quadratic (x : ℝ) : ¬ (|x - 3| = x^2 + 2 * x + 4) := 
by
  sorry

end no_solutions_abs_eq_quadratic_l1520_152005


namespace range_of_k_for_distinct_real_roots_l1520_152023

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k*x1^2 - 2*x1 - 1 = 0 ∧ k*x2^2 - 2*x2 - 1 = 0) → k > -1 ∧ k ≠ 0 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l1520_152023


namespace proof_problem_l1520_152008

variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be increasing on (-∞, 0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y < 0 → f x < f y

-- Define what it means for a function to be decreasing on (0, +∞)
def is_decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f y < f x

theorem proof_problem 
  (h_even : is_even_function f) 
  (h_inc_neg : is_increasing_on_neg f) : 
  (∀ x : ℝ, f (-x) - f x = 0) ∧ (is_decreasing_on_pos f) :=
by
  sorry

end proof_problem_l1520_152008


namespace arbitrarily_large_ratios_l1520_152037

open Nat

theorem arbitrarily_large_ratios (a : ℕ → ℕ) (h_distinct: ∀ m n, m ≠ n → a m ≠ a n)
  (h_no_100_ones: ∀ n, ¬ (∃ k, a n / 10^k % 10^100 = 10^100 - 1)):
  ∀ M : ℕ, ∃ n : ℕ, a n / n ≥ M :=
by
  sorry

end arbitrarily_large_ratios_l1520_152037


namespace minimum_value_expression_l1520_152000

theorem minimum_value_expression (x y z : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2) ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2)) :=
sorry

end minimum_value_expression_l1520_152000


namespace conveyor_belt_efficiencies_and_min_cost_l1520_152020

theorem conveyor_belt_efficiencies_and_min_cost :
  ∃ (efficiency_B efficiency_A : ℝ),
    efficiency_A = 1.5 * efficiency_B ∧
    18000 / efficiency_B - 18000 / efficiency_A = 10 ∧
    efficiency_B = 600 ∧
    efficiency_A = 900 ∧
    ∃ (cost_A cost_B : ℝ),
      cost_A = 8 * 20 ∧
      cost_B = 6 * 30 ∧
      cost_A = 160 ∧
      cost_B = 180 ∧
      cost_A < cost_B :=
by
  sorry

end conveyor_belt_efficiencies_and_min_cost_l1520_152020


namespace john_candies_l1520_152019

theorem john_candies (mark_candies : ℕ) (peter_candies : ℕ) (total_candies : ℕ) (equal_share : ℕ) (h1 : mark_candies = 30) (h2 : peter_candies = 25) (h3 : total_candies = 90) (h4 : equal_share * 3 = total_candies) : 
  (total_candies - mark_candies - peter_candies = 35) :=
by
  sorry

end john_candies_l1520_152019


namespace loss_percentage_on_first_book_l1520_152025

theorem loss_percentage_on_first_book 
    (C1 C2 SP : ℝ) 
    (H1 : C1 = 210) 
    (H2 : C1 + C2 = 360) 
    (H3 : SP = 1.19 * C2) 
    (H4 : SP = 178.5) :
    ((C1 - SP) / C1) * 100 = 15 :=
by
  sorry

end loss_percentage_on_first_book_l1520_152025


namespace triangle_inequality_l1520_152070

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l1520_152070


namespace max_area_of_rectangular_playground_l1520_152072

theorem max_area_of_rectangular_playground (P : ℕ) (hP : P = 160) :
  (∃ (x y : ℕ), 2 * (x + y) = P ∧ x * y = 1600) :=
by
  sorry

end max_area_of_rectangular_playground_l1520_152072


namespace no_blonde_girls_added_l1520_152093

-- Initial number of girls
def total_girls : Nat := 80
def initial_blonde_girls : Nat := 30
def black_haired_girls : Nat := 50

-- Number of blonde girls added
def blonde_girls_added : Nat := total_girls - black_haired_girls - initial_blonde_girls

theorem no_blonde_girls_added : blonde_girls_added = 0 :=
by
  sorry

end no_blonde_girls_added_l1520_152093


namespace second_gym_signup_fee_covers_4_months_l1520_152056

-- Define constants
def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def total_spent_first_year : ℕ := 650

-- Define the monthly fee of the second gym
def second_gym_monthly_fee : ℕ := 3 * cheap_gym_monthly_fee

-- Calculate the amount spent on the second gym
def spent_on_second_gym : ℕ := total_spent_first_year - (12 * cheap_gym_monthly_fee + cheap_gym_signup_fee)

-- Define the number of months the sign-up fee covers
def months_covered_by_signup_fee : ℕ := spent_on_second_gym / second_gym_monthly_fee

theorem second_gym_signup_fee_covers_4_months :
  months_covered_by_signup_fee = 4 :=
by
  sorry

end second_gym_signup_fee_covers_4_months_l1520_152056


namespace tank_salt_solution_l1520_152067

theorem tank_salt_solution (x : ℝ) (hx1 : 0.20 * x / (3 / 4 * x + 30) = 1 / 3) : x = 200 :=
by sorry

end tank_salt_solution_l1520_152067


namespace joan_seashells_total_l1520_152022

-- Definitions
def original_seashells : ℕ := 70
def additional_seashells : ℕ := 27
def total_seashells : ℕ := original_seashells + additional_seashells

-- Proof Statement
theorem joan_seashells_total : total_seashells = 97 := by
  sorry

end joan_seashells_total_l1520_152022


namespace speed_of_man_in_still_water_l1520_152016

variable (V_m V_s : ℝ)

/-- The speed of a man in still water -/
theorem speed_of_man_in_still_water (h_downstream : 18 = (V_m + V_s) * 3)
                                     (h_upstream : 12 = (V_m - V_s) * 3) :
    V_m = 5 := 
sorry

end speed_of_man_in_still_water_l1520_152016


namespace parallelogram_angle_H_l1520_152002

theorem parallelogram_angle_H (F H : ℝ) (h1 : F = 125) (h2 : F + H = 180) : H = 55 :=
by
  have h3 : H = 180 - F := by linarith
  rw [h1] at h3
  rw [h3]
  norm_num

end parallelogram_angle_H_l1520_152002


namespace bus_problem_l1520_152044

theorem bus_problem
  (initial_children : ℕ := 18)
  (final_total_children : ℕ := 25) :
  final_total_children - initial_children = 7 :=
by
  sorry

end bus_problem_l1520_152044


namespace intersection_with_y_axis_l1520_152088

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end intersection_with_y_axis_l1520_152088


namespace simplify_expr1_simplify_expr2_l1520_152049

theorem simplify_expr1 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
by
  sorry

end simplify_expr1_simplify_expr2_l1520_152049


namespace apples_per_pie_l1520_152078

/-- Let's define the parameters given in the problem -/
def initial_apples : ℕ := 62
def apples_given_to_students : ℕ := 8
def pies_made : ℕ := 6

/-- Define the remaining apples after handing out to students -/
def remaining_apples : ℕ := initial_apples - apples_given_to_students

/-- The statement we need to prove: each pie requires 9 apples -/
theorem apples_per_pie : remaining_apples / pies_made = 9 := by
  -- Add the proof here
  sorry

end apples_per_pie_l1520_152078


namespace axis_of_symmetry_of_f_l1520_152048

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

theorem axis_of_symmetry_of_f : (axis_of_symmetry : ℝ) = -1 :=
by
  sorry

end axis_of_symmetry_of_f_l1520_152048


namespace more_blue_marbles_l1520_152021

theorem more_blue_marbles (r_boxes b_boxes marbles_per_box : ℕ) 
    (red_total_eq : r_boxes * marbles_per_box = 70) 
    (blue_total_eq : b_boxes * marbles_per_box = 126) 
    (r_boxes_eq : r_boxes = 5) 
    (b_boxes_eq : b_boxes = 9) 
    (marbles_per_box_eq : marbles_per_box = 14) : 
    126 - 70 = 56 := 
by 
  sorry

end more_blue_marbles_l1520_152021


namespace paint_coverage_l1520_152011

theorem paint_coverage 
  (width height cost_per_quart money_spent area : ℕ)
  (cover : ℕ → ℕ → ℕ)
  (num_sides quarts_purchased : ℕ)
  (total_area num_quarts : ℕ)
  (sqfeet_per_quart : ℕ) :
  width = 5 
  → height = 4 
  → cost_per_quart = 2 
  → money_spent = 20 
  → num_sides = 2
  → cover width height = area
  → area * num_sides = total_area
  → money_spent / cost_per_quart = quarts_purchased
  → total_area / quarts_purchased = sqfeet_per_quart
  → total_area = 40 
  → quarts_purchased = 10 
  → sqfeet_per_quart = 4 :=
by 
  intros
  sorry

end paint_coverage_l1520_152011


namespace bacteria_growth_rate_l1520_152064

theorem bacteria_growth_rate (P : ℝ) (r : ℝ) : 
  (P * r ^ 25 = 2 * (P * r ^ 24) ) → r = 2 :=
by sorry

end bacteria_growth_rate_l1520_152064


namespace gcd_9009_14014_l1520_152096

-- Given conditions
def decompose_9009 : 9009 = 9 * 1001 := by sorry
def decompose_14014 : 14014 = 14 * 1001 := by sorry
def coprime_9_14 : Nat.gcd 9 14 = 1 := by sorry

-- Proof problem statement
theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  have h1 : 9009 = 9 * 1001 := decompose_9009
  have h2 : 14014 = 14 * 1001 := decompose_14014
  have h3 : Nat.gcd 9 14 = 1 := coprime_9_14
  sorry

end gcd_9009_14014_l1520_152096


namespace pascal_triangle_fifth_number_twentieth_row_l1520_152028

theorem pascal_triangle_fifth_number_twentieth_row : 
  (Nat.choose 20 4) = 4845 :=
by
  sorry

end pascal_triangle_fifth_number_twentieth_row_l1520_152028


namespace Jim_paycheck_after_deductions_l1520_152094

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end Jim_paycheck_after_deductions_l1520_152094


namespace find_sports_package_channels_l1520_152017

-- Defining the conditions
def initial_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def reduce_package_by : ℕ := 10
def supreme_sports_package : ℕ := 7
def final_channels : ℕ := 147

-- Defining the situation before the final step
def channels_after_reduction := initial_channels - channels_taken_away + channels_replaced - reduce_package_by
def channels_after_supreme := channels_after_reduction + supreme_sports_package

-- Prove the original sports package added 8 channels
theorem find_sports_package_channels : ∀ sports_package_added : ℕ,
  sports_package_added + channels_after_supreme = final_channels → sports_package_added = 8 :=
by
  intro sports_package_added
  intro h
  sorry

end find_sports_package_channels_l1520_152017


namespace line_contains_point_l1520_152090

theorem line_contains_point (k : ℝ) (x : ℝ) (y : ℝ) (H : 2 - 2 * k * x = -4 * y) : k = -1 ↔ (x = 3 ∧ y = -2) :=
by
  sorry

end line_contains_point_l1520_152090


namespace solve_for_diamond_l1520_152029

theorem solve_for_diamond (d : ℤ) (h : d * 9 + 5 = d * 10 + 2) : d = 3 :=
by
  sorry

end solve_for_diamond_l1520_152029


namespace real_polynomial_has_exactly_one_real_solution_l1520_152032

theorem real_polynomial_has_exactly_one_real_solution:
  ∀ a : ℝ, ∃! x : ℝ, x^3 - a * x^2 - 3 * a * x + a^2 - 1 = 0 := 
by
  sorry

end real_polynomial_has_exactly_one_real_solution_l1520_152032


namespace interest_rate_difference_correct_l1520_152075

noncomputable def interest_rate_difference (P r R T : ℝ) :=
  let I := P * r * T
  let I' := P * R * T
  (I' - I) = 140

theorem interest_rate_difference_correct:
  ∀ (P r R T : ℝ),
  P = 1000 ∧ T = 7 ∧ interest_rate_difference P r R T →
  (R - r) = 0.02 :=
by
  intros P r R T h
  sorry

end interest_rate_difference_correct_l1520_152075


namespace number_of_possible_triangles_with_side_5_not_shortest_l1520_152095

-- Define and prove the number of possible triangles (a, b, c) with a, b, c positive integers,
-- such that one side is length 5 and it is not the shortest side is 10.
theorem number_of_possible_triangles_with_side_5_not_shortest (a b c : ℕ) (h1: a + b > c) (h2: a + c > b) (h3: b + c > a) 
(h4: 0 < a) (h5: 0 < b) (h6: 0 < c) (h7: a = 5 ∨ b = 5 ∨ c = 5) (h8: ¬ (a < 5 ∧ b < 5 ∧ c < 5)) :
∃ n, n = 10 := 
sorry

end number_of_possible_triangles_with_side_5_not_shortest_l1520_152095


namespace remainder_when_divided_by_84_l1520_152089

/-- 
  Given conditions:
  x ≡ 11 [MOD 14]
  Find the remainder when x is divided by 84, which equivalently means proving: 
  x ≡ 81 [MOD 84]
-/

theorem remainder_when_divided_by_84 (x : ℤ) (h1 : x % 14 = 11) : x % 84 = 81 :=
by
  sorry

end remainder_when_divided_by_84_l1520_152089


namespace hyperbola_asymptotes_l1520_152059

theorem hyperbola_asymptotes:
  ∀ (x y : ℝ),
  ( ∀ y, y = (1 + (4 / 5) * x) ∨ y = (1 - (4 / 5) * x) ) →
  (y-1)^2 / 16 - x^2 / 25 = 1 →
  (∃ m b: ℝ, m > 0 ∧ m = 4/5 ∧ b = 1) := by
  sorry

end hyperbola_asymptotes_l1520_152059


namespace M_inter_N_l1520_152079

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.sqrt x + Real.log (1 - x) }

theorem M_inter_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_inter_N_l1520_152079


namespace range_of_t_l1520_152052

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (a t : ℝ) := 2 * a * t - t^2

theorem range_of_t (t : ℝ) (a : ℝ) (x : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x)
                   (h₂ : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 1 → f x₁ ≤ f x₂)
                   (h₃ : f (-1) = -1) (h₄ : -1 ≤ x ∧ x ≤ 1 → f x ≤ t^2 - 2 * a * t + 1)
                   (h₅ : -1 ≤ a ∧ a ≤ 1) :
  t ≥ 2 ∨ t = 0 ∨ t ≤ -2 := sorry

end range_of_t_l1520_152052


namespace average_sleep_is_8_l1520_152040

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end average_sleep_is_8_l1520_152040


namespace find_coefficients_l1520_152085

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^4 - 8 * a * x^3 + b * x^2 - 32 * c * x + 16 * c

theorem find_coefficients (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 x3 x4 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 ∧ P a b c x3 = 0 ∧ P a b c x4 = 0) →
  (b = 16 * a ∧ c = a) :=
by
  sorry

end find_coefficients_l1520_152085


namespace pythagorean_relationship_l1520_152015

theorem pythagorean_relationship (a b c : ℝ) (h : c^2 = a^2 + b^2) : c^2 = a^2 + b^2 :=
by
  sorry

end pythagorean_relationship_l1520_152015


namespace probability_white_balls_le_1_l1520_152081

-- Definitions and conditions
def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

-- Combinatorial computations
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculations based on the conditions
def total_combinations : ℕ := C total_balls selected_balls
def red_combinations : ℕ := C red_balls selected_balls
def white_combinations : ℕ := C white_balls 1 * C red_balls 2

-- Probability calculations
def P_xi_le_1 : ℚ :=
  (red_combinations / total_combinations : ℚ) +
  (white_combinations / total_combinations : ℚ)

-- Problem statement: Prove that the calculated probability is 4/5
theorem probability_white_balls_le_1 : P_xi_le_1 = 4 / 5 := 
  sorry

end probability_white_balls_le_1_l1520_152081


namespace cos_theta_value_l1520_152035

theorem cos_theta_value (θ : ℝ) (h_tan : Real.tan θ = -4/3) (h_range : 0 < θ ∧ θ < π) : Real.cos θ = -3/5 :=
by
  sorry

end cos_theta_value_l1520_152035


namespace ellipse_minimum_distance_point_l1520_152024

theorem ellipse_minimum_distance_point :
  ∃ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1) ∧ (∀ p, x - 2 * y - 12 = 0 → dist (x, y) p ≥ dist (2, -3) p) :=
sorry

end ellipse_minimum_distance_point_l1520_152024


namespace sum_of_coeffs_eq_59049_l1520_152003

-- Definition of the polynomial
def poly (x y z : ℕ) : ℕ :=
  (2 * x - 3 * y + 4 * z) ^ 10

-- Conjecture: The sum of the numerical coefficients in poly when x, y, and z are set to 1 is 59049
theorem sum_of_coeffs_eq_59049 : poly 1 1 1 = 59049 := by
  sorry

end sum_of_coeffs_eq_59049_l1520_152003


namespace set_contains_all_rationals_l1520_152069

variable (S : Set ℚ)
variable (h1 : (0 : ℚ) ∈ S)
variable (h2 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
variable (h3 : ∀ x ∈ S, x ≠ 0 → x ≠ 1 → 1 / (x * (x - 1)) ∈ S)

theorem set_contains_all_rationals : ∀ q : ℚ, q ∈ S :=
by
  sorry

end set_contains_all_rationals_l1520_152069


namespace find_alpha_l1520_152047

theorem find_alpha (α : Real) (hα : 0 < α ∧ α < π) :
  (∃ x : Real, (|2 * x - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * x| = Real.sin α) ∧ 
  ∀ y : Real, (|2 * y - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * y| = Real.sin α) → y = x) →
  α = π / 12 ∨ α = 11 * π / 12 :=
by
  sorry

end find_alpha_l1520_152047


namespace bert_earns_more_l1520_152033

def bert_toy_phones : ℕ := 8
def bert_price_per_phone : ℕ := 18
def tory_toy_guns : ℕ := 7
def tory_price_per_gun : ℕ := 20

theorem bert_earns_more : (bert_toy_phones * bert_price_per_phone) - (tory_toy_guns * tory_price_per_gun) = 4 := by
  sorry

end bert_earns_more_l1520_152033


namespace max_squares_covered_by_card_l1520_152076

theorem max_squares_covered_by_card : 
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  ∃ n, n = 9 :=
by
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  existsi 9
  sorry

end max_squares_covered_by_card_l1520_152076


namespace smallest_integer_y_l1520_152042

theorem smallest_integer_y (y : ℤ) (h : 7 - 5 * y < 22) : y ≥ -2 :=
by sorry

end smallest_integer_y_l1520_152042


namespace total_money_l1520_152051

theorem total_money (a b c : ℕ) (h_ratio : (a / 2) / (b / 3) / (c / 4) = 1) (h_c : c = 306) : 
  a + b + c = 782 := 
by sorry

end total_money_l1520_152051


namespace find_c_l1520_152066

variable (c : ℝ)

theorem find_c (h : c * (1 + 1/2 + 1/3 + 1/4) = 1) : c = 12 / 25 :=
by 
  sorry

end find_c_l1520_152066


namespace ms_tom_investment_l1520_152018

def invested_amounts (X Y : ℝ) : Prop :=
  X + Y = 100000 ∧ 0.17 * Y = 0.23 * X + 200 

theorem ms_tom_investment (X Y : ℝ) (h : invested_amounts X Y) : X = 42000 :=
by
  sorry

end ms_tom_investment_l1520_152018


namespace set_intersection_complement_l1520_152062

open Set

variable (U P Q: Set ℕ)

theorem set_intersection_complement (hU: U = {1, 2, 3, 4}) (hP: P = {1, 2}) (hQ: Q = {2, 3}) :
  P ∩ (U \ Q) = {1} :=
by
  sorry

end set_intersection_complement_l1520_152062


namespace problem_equivalent_proof_l1520_152006

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem_equivalent_proof : ((sqrt 3 - 2) ^ 0 - Real.logb 2 (sqrt 2)) = 1 / 2 :=
by
  sorry

end problem_equivalent_proof_l1520_152006


namespace remainder_n_plus_2023_l1520_152036

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 :=
sorry

end remainder_n_plus_2023_l1520_152036


namespace gcd_of_gx_and_x_l1520_152004

theorem gcd_of_gx_and_x (x : ℤ) (hx : x % 11739 = 0) :
  Int.gcd ((3 * x + 4) * (5 * x + 3) * (11 * x + 5) * (x + 11)) x = 3 :=
sorry

end gcd_of_gx_and_x_l1520_152004


namespace smallest_positive_x_l1520_152041

theorem smallest_positive_x
  (x : ℕ)
  (h1 : x % 3 = 2)
  (h2 : x % 7 = 6)
  (h3 : x % 8 = 7) : x = 167 :=
by
  sorry

end smallest_positive_x_l1520_152041


namespace no_non_similar_triangles_with_geometric_angles_l1520_152038

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end no_non_similar_triangles_with_geometric_angles_l1520_152038
