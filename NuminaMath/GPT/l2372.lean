import Mathlib

namespace general_term_of_sequence_l2372_237222

theorem general_term_of_sequence 
  (a : ℕ → ℝ)
  (log_a : ℕ → ℝ)
  (h1 : ∀ n, log_a n = Real.log (a n)) 
  (h2 : ∃ d, ∀ n, log_a (n + 1) - log_a n = d)
  (h3 : d = Real.log 3)
  (h4 : log_a 0 + log_a 1 + log_a 2 = 6 * Real.log 3) : 
  ∀ n, a n = 3 ^ n :=
by
  sorry

end general_term_of_sequence_l2372_237222


namespace tens_digit_23_pow_1987_l2372_237253

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l2372_237253


namespace find_real_solutions_l2372_237247

theorem find_real_solutions : 
  ∀ x : ℝ, 1 / ((x - 2) * (x - 3)) 
         + 1 / ((x - 3) * (x - 4)) 
         + 1 / ((x - 4) * (x - 5)) 
         = 1 / 8 ↔ x = 7 ∨ x = -2 :=
by
  intro x
  sorry

end find_real_solutions_l2372_237247


namespace money_left_after_expenses_l2372_237215

theorem money_left_after_expenses : 
  let salary := 150000.00000000003
  let food := salary * (1 / 5)
  let house_rent := salary * (1 / 10)
  let clothes := salary * (3 / 5)
  let total_spent := food + house_rent + clothes
  let money_left := salary - total_spent
  money_left = 15000.00000000000 :=
by
  sorry

end money_left_after_expenses_l2372_237215


namespace find_cost_price_l2372_237266

variable (C : ℝ)

theorem find_cost_price (h : 56 - C = C - 42) : C = 49 :=
by
  sorry

end find_cost_price_l2372_237266


namespace min_value_fraction_l2372_237227

variable {a b : ℝ}

theorem min_value_fraction (h₁ : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (1 / a + 4 / b) ≥ 9 :=
sorry

end min_value_fraction_l2372_237227


namespace odd_c_perfect_square_no_even_c_infinitely_many_solutions_l2372_237226

open Nat

/-- Problem (1): prove that if c is an odd number, then c is a perfect square given 
    c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem odd_c_perfect_square (a b c : ℕ) (h_eq : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) (h_odd : Odd c) : ∃ k : ℕ, c = k^2 :=
  sorry

/-- Problem (2): prove that there does not exist an even number c that satisfies 
    c(a c + 1)^2 = (5c + 2b)(2c + b) for some a and b -/
theorem no_even_c (a b : ℕ) : ∀ c : ℕ, Even c → ¬ (c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) :=
  sorry

/-- Problem (3): prove that there are infinitely many solutions of positive integers 
    (a, b, c) that satisfy c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem infinitely_many_solutions (n : ℕ) : ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b) :=
  sorry

end odd_c_perfect_square_no_even_c_infinitely_many_solutions_l2372_237226


namespace total_arrangements_l2372_237229

theorem total_arrangements (students communities : ℕ) 
  (h_students : students = 5) 
  (h_communities : communities = 3)
  (h_conditions :
    ∀(student : Fin students) (community : Fin communities), 
      true 
  ) : 150 = 150 :=
by sorry

end total_arrangements_l2372_237229


namespace dihedral_minus_solid_equals_expression_l2372_237286

-- Definitions based on the conditions provided.
noncomputable def sumDihedralAngles (P : Polyhedron) : ℝ := sorry
noncomputable def sumSolidAngles (P : Polyhedron) : ℝ := sorry
def numFaces (P : Polyhedron) : ℕ := sorry

-- Theorem statement we want to prove.
theorem dihedral_minus_solid_equals_expression (P : Polyhedron) :
  sumDihedralAngles P - sumSolidAngles P = 2 * Real.pi * (numFaces P - 2) :=
sorry

end dihedral_minus_solid_equals_expression_l2372_237286


namespace probability_right_triangle_in_3x3_grid_l2372_237275

theorem probability_right_triangle_in_3x3_grid : 
  let vertices := (3 + 1) * (3 + 1)
  let total_combinations := Nat.choose vertices 3
  let right_triangles_on_gridlines := 144
  let right_triangles_off_gridlines := 24 + 32
  let total_right_triangles := right_triangles_on_gridlines + right_triangles_off_gridlines
  (total_right_triangles : ℚ) / total_combinations = 5 / 14 :=
by 
  sorry

end probability_right_triangle_in_3x3_grid_l2372_237275


namespace downstream_distance_l2372_237272

-- Define the speeds and distances as constants or variables
def speed_boat := 30 -- speed in kmph
def speed_stream := 10 -- speed in kmph
def distance_upstream := 40 -- distance in km
def time_upstream := distance_upstream / (speed_boat - speed_stream) -- time in hours

-- Define the variable for the downstream distance
variable {D : ℝ}

-- The Lean 4 statement to prove that the downstream distance is the specified value
theorem downstream_distance : 
  (time_upstream = D / (speed_boat + speed_stream)) → D = 80 :=
by
  sorry

end downstream_distance_l2372_237272


namespace necessary_but_not_sufficient_condition_holds_l2372_237213

-- Let m be a real number
variable (m : ℝ)

-- Define the conditions
def condition_1 : Prop := (m + 3) * (2 * m + 1) < 0
def condition_2 : Prop := -(2 * m - 1) > m + 2
def condition_3 : Prop := m + 2 > 0

-- Define necessary but not sufficient condition
def necessary_but_not_sufficient : Prop :=
  -2 < m ∧ m < -1 / 3

-- Problem statement
theorem necessary_but_not_sufficient_condition_holds 
  (h1 : condition_1 m) 
  (h2 : condition_2 m) 
  (h3 : condition_3 m) : necessary_but_not_sufficient m :=
sorry

end necessary_but_not_sufficient_condition_holds_l2372_237213


namespace unsold_books_l2372_237295

-- Definitions from conditions
def books_total : ℕ := 150
def books_sold : ℕ := (2 / 3) * books_total
def book_price : ℕ := 5
def total_received : ℕ := 500

-- Proof statement
theorem unsold_books :
  (books_sold * book_price = total_received) →
  (books_total - books_sold = 50) :=
by
  sorry

end unsold_books_l2372_237295


namespace find_g_zero_l2372_237234

variable {g : ℝ → ℝ}

theorem find_g_zero (h : ∀ x y : ℝ, g (x + y) = g x + g y - 1) : g 0 = 1 :=
sorry

end find_g_zero_l2372_237234


namespace sum_of_invalid_domain_of_g_l2372_237210

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + (1 / (3 + (1 / x))))

theorem sum_of_invalid_domain_of_g : 
  (0 : ℝ) + (-1 / 3) + (-2 / 7) = -13 / 21 :=
by
  sorry

end sum_of_invalid_domain_of_g_l2372_237210


namespace compare_a_b_c_compare_explicitly_defined_a_b_c_l2372_237265

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end compare_a_b_c_compare_explicitly_defined_a_b_c_l2372_237265


namespace gumball_machine_l2372_237250

variable (R B G Y O : ℕ)

theorem gumball_machine : 
  (B = (1 / 2) * R) ∧
  (G = 4 * B) ∧
  (Y = (7 / 2) * B) ∧
  (O = (2 / 3) * (R + B)) ∧
  (R = (3 / 2) * Y) ∧
  (Y = 24) →
  (R + B + G + Y + O = 186) :=
sorry

end gumball_machine_l2372_237250


namespace cornflowers_count_l2372_237243

theorem cornflowers_count
  (n k : ℕ)
  (total_flowers : 9 * n + 17 * k = 70)
  (equal_dandelions_daisies : 5 * n = 7 * k) :
  (9 * n - 20 - 14 = 2) ∧ (17 * k - 20 - 14 = 0) :=
by
  sorry

end cornflowers_count_l2372_237243


namespace length_of_garden_l2372_237239

theorem length_of_garden (P B : ℕ) (hP : P = 1800) (hB : B = 400) : 
  ∃ L : ℕ, L = 500 ∧ P = 2 * (L + B) :=
by
  sorry

end length_of_garden_l2372_237239


namespace div_z_x_l2372_237221

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l2372_237221


namespace x_squared_plus_y_squared_l2372_237268

theorem x_squared_plus_y_squared (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
(h3 : x * y + x + y = 71)
(h4 : x^2 * y + x * y^2 = 880) :
x^2 + y^2 = 146 :=
sorry

end x_squared_plus_y_squared_l2372_237268


namespace parallel_lines_implies_m_opposite_sides_implies_m_range_l2372_237237

-- Definitions of the given lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Problem Part (I)
theorem parallel_lines_implies_m (m : ℝ) : 
  (∀ (x y : ℝ), l1 x y → false) ∧ (∀ (x2 y2 : ℝ), (x2, y2) = A m ∨ (x2, y2) = B m → false) →
  (∃ m, 2 * m + 3 = 0 ∧ m + 5 = 0) :=
sorry

-- Problem Part (II)
theorem opposite_sides_implies_m_range (m : ℝ) :
  ((2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0) →
  m ∈ Set.Ioo (-3/2 : ℝ) (5 : ℝ) :=
sorry

end parallel_lines_implies_m_opposite_sides_implies_m_range_l2372_237237


namespace temperature_drop_l2372_237254

-- Define the initial temperature and the drop in temperature
def initial_temperature : ℤ := -6
def drop : ℤ := 5

-- Define the resulting temperature after the drop
def resulting_temperature : ℤ := initial_temperature - drop

-- The theorem to be proved
theorem temperature_drop : resulting_temperature = -11 :=
by
  sorry

end temperature_drop_l2372_237254


namespace density_of_cone_in_mercury_l2372_237288

variable {h : ℝ} -- height of the cone
variable {ρ : ℝ} -- density of the cone
variable {ρ_m : ℝ} -- density of the mercury
variable {k : ℝ} -- proportion factor

-- Archimedes' principle applied to the cone floating in mercury
theorem density_of_cone_in_mercury (stable_eq: ∀ (V V_sub: ℝ), (ρ * V) = (ρ_m * V_sub))
(h_sub: h / k = (k - 1) / k) :
  ρ = ρ_m * ((k - 1)^3 / k^3) :=
by
  sorry

end density_of_cone_in_mercury_l2372_237288


namespace problem_f_val_l2372_237251

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_val (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (-x) = -f x)
  (h2 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3) :
  f 2015 = -1 :=
  sorry

end problem_f_val_l2372_237251


namespace average_age_of_women_l2372_237259

theorem average_age_of_women (A : ℝ) (W1 W2 : ℝ)
  (cond1 : 10 * (A + 6) - 10 * A = 60)
  (cond2 : W1 + W2 = 60 + 40) :
  (W1 + W2) / 2 = 50 := 
by
  sorry

end average_age_of_women_l2372_237259


namespace local_odd_function_range_of_a_l2372_237203

variable (f : ℝ → ℝ)
variable (a : ℝ)

def local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (-x₀) = -f x₀

theorem local_odd_function_range_of_a (hf : ∀ x, f x = -a * (2^x) - 4) :
  local_odd_function f → (-4 ≤ a ∧ a < 0) :=
by
  sorry

end local_odd_function_range_of_a_l2372_237203


namespace Roshesmina_pennies_l2372_237289

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end Roshesmina_pennies_l2372_237289


namespace value_of_a_l2372_237255

noncomputable def function_f (x a : ℝ) : ℝ := (x - a) ^ 2 + (Real.log x ^ 2 - 2 * a) ^ 2

theorem value_of_a (x0 : ℝ) (a : ℝ) (h1 : x0 > 0) (h2 : function_f x0 a ≤ 4 / 5) : a = 1 / 5 :=
sorry

end value_of_a_l2372_237255


namespace batsman_average_after_12th_innings_l2372_237241

-- Defining the conditions
def before_12th_innings_average (A : ℕ) : Prop :=
11 * A + 80 = 12 * (A + 2)

-- Defining the question and expected answer
def after_12th_innings_average : ℕ := 58

-- Proving the equivalence
theorem batsman_average_after_12th_innings (A : ℕ) (h : before_12th_innings_average A) : after_12th_innings_average = 58 :=
by
sorry

end batsman_average_after_12th_innings_l2372_237241


namespace circle_center_l2372_237230

theorem circle_center (x y : ℝ) :
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 →
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 8 ∧ h = 2 ∧ k = -1) :=
sorry

end circle_center_l2372_237230


namespace proof_of_problem_l2372_237238

noncomputable def problem_statement : Prop :=
  ∃ (x y z m : ℝ), (x > 0 ∧ y > 0 ∧ z > 0 ∧ x^3 * y^2 * z = 1 ∧ m = x + 2*y + 3*z ∧ m^3 = 72)

theorem proof_of_problem : problem_statement :=
sorry

end proof_of_problem_l2372_237238


namespace coefficient_x2y6_expansion_l2372_237287

theorem coefficient_x2y6_expansion :
  let x : ℤ := 1
  let y : ℤ := 1
  ∃ a : ℤ, a = -28 ∧ (a • x ^ 2 * y ^ 6) = (1 - y / x) * (x + y) ^ 8 :=
by
  sorry

end coefficient_x2y6_expansion_l2372_237287


namespace caffeine_per_energy_drink_l2372_237205

variable (amount_of_caffeine_per_drink : ℕ)

def maximum_safe_caffeine_per_day := 500
def drinks_per_day := 4
def additional_safe_amount := 20

theorem caffeine_per_energy_drink :
  4 * amount_of_caffeine_per_drink + additional_safe_amount = maximum_safe_caffeine_per_day →
  amount_of_caffeine_per_drink = 120 :=
by
  sorry

end caffeine_per_energy_drink_l2372_237205


namespace number_of_intersections_l2372_237244

   -- Definitions corresponding to conditions
   def C1 (x y : ℝ) : Prop := x^2 - y^2 + 4*y - 3 = 0
   def C2 (a x y : ℝ) : Prop := y = a*x^2
   def positive_real (a : ℝ) : Prop := a > 0

   -- Final statement converting the question, conditions, and correct answer into Lean code
   theorem number_of_intersections (a : ℝ) (ha : positive_real a) :
     ∃ (count : ℕ), (count = 4) ∧
     (∀ x y : ℝ, C1 x y → C2 a x y → True) := sorry
   
end number_of_intersections_l2372_237244


namespace polar_coordinate_conversion_l2372_237260

theorem polar_coordinate_conversion :
  ∃ (r θ : ℝ), (r = 2) ∧ (θ = 11 * Real.pi / 8) ∧ 
    ∀ (r1 θ1 : ℝ), (r1 = -2) ∧ (θ1 = 3 * Real.pi / 8) →
      (abs r1 = r) ∧ (θ1 + Real.pi = θ) :=
by
  sorry

end polar_coordinate_conversion_l2372_237260


namespace Nell_has_123_more_baseball_cards_than_Ace_cards_l2372_237207

def Nell_cards_diff (baseball_cards_new : ℕ) (ace_cards_new : ℕ) : ℕ :=
  baseball_cards_new - ace_cards_new

theorem Nell_has_123_more_baseball_cards_than_Ace_cards:
  (Nell_cards_diff 178 55) = 123 :=
by
  -- proof here
  sorry

end Nell_has_123_more_baseball_cards_than_Ace_cards_l2372_237207


namespace siamese_cats_initial_l2372_237242

theorem siamese_cats_initial (S : ℕ) : S + 25 - 45 = 18 -> S = 38 :=
by
  intro h
  sorry

end siamese_cats_initial_l2372_237242


namespace white_area_is_69_l2372_237200

def area_of_sign : ℕ := 6 * 20

def area_of_M : ℕ := 2 * (6 * 1) + 2 * 2

def area_of_A : ℕ := 2 * 4 + 1 * 2

def area_of_T : ℕ := 1 * 4 + 6 * 1

def area_of_H : ℕ := 2 * (6 * 1) + 1 * 3

def total_black_area : ℕ := area_of_M + area_of_A + area_of_T + area_of_H

def white_area (sign_area black_area : ℕ) : ℕ := sign_area - black_area

theorem white_area_is_69 : white_area area_of_sign total_black_area = 69 := by
  sorry

end white_area_is_69_l2372_237200


namespace part_i_part_ii_l2372_237216

open Real -- Open the Real number space

-- (i) Prove that for any real number x, there exist two points of the same color that are at a distance of x from each other
theorem part_i (color : Real × Real → Bool) :
  ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = color p2 ∧ dist p1 p2 = x :=
by
  sorry

-- (ii) Prove that there exists a color such that for every real number x, 
-- we can find two points of that color that are at a distance of x from each other
theorem part_ii (color : Real × Real → Bool) :
  ∃ c : Bool, ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = c ∧ color p2 = c ∧ dist p1 p2 = x :=
by
  sorry

end part_i_part_ii_l2372_237216


namespace evaluate_expression_l2372_237212

theorem evaluate_expression {x y : ℕ} (h₁ : 144 = 2^x * 3^y) (hx : x = 4) (hy : y = 2) : (1 / 7) ^ (y - x) = 49 := 
by
  sorry

end evaluate_expression_l2372_237212


namespace fraction_order_l2372_237218

theorem fraction_order :
  (25 / 19 : ℚ) < (21 / 16 : ℚ) ∧ (21 / 16 : ℚ) < (23 / 17 : ℚ) := by
  sorry

end fraction_order_l2372_237218


namespace gcd_288_123_l2372_237249

theorem gcd_288_123 : gcd 288 123 = 3 :=
by
  sorry

end gcd_288_123_l2372_237249


namespace max_a2b3c4_l2372_237232

noncomputable def maximum_value (a b c : ℝ) : ℝ := a^2 * b^3 * c^4

theorem max_a2b3c4 (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  maximum_value a b c ≤ 143327232 / 386989855 := sorry

end max_a2b3c4_l2372_237232


namespace box_white_balls_count_l2372_237282

/--
A box has exactly 100 balls, and each ball is either red, blue, or white.
Given that the box has 12 more blue balls than white balls,
and twice as many red balls as blue balls,
prove that the number of white balls is 16.
-/
theorem box_white_balls_count (W B R : ℕ) 
  (h1 : B = W + 12) 
  (h2 : R = 2 * B) 
  (h3 : W + B + R = 100) : 
  W = 16 := 
sorry

end box_white_balls_count_l2372_237282


namespace initial_ratio_of_milk_to_water_l2372_237211

-- Define the capacity of the can, the amount of milk added, and the ratio when full.
def capacity : ℕ := 72
def additionalMilk : ℕ := 8
def fullRatioNumerator : ℕ := 2
def fullRatioDenominator : ℕ := 1

-- Define the initial amounts of milk and water in the can.
variables (M W : ℕ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  M + W + additionalMilk = capacity ∧
  (M + additionalMilk) * fullRatioDenominator = fullRatioNumerator * W

-- Define the expected result, the initial ratio of milk to water in the can.
def expected_ratio : ℕ × ℕ :=
  (5, 3)

-- The theorem to prove the initial ratio of milk to water given the conditions.
theorem initial_ratio_of_milk_to_water (M W : ℕ) (h : conditions M W) :
  (M / Nat.gcd M W, W / Nat.gcd M W) = expected_ratio :=
sorry

end initial_ratio_of_milk_to_water_l2372_237211


namespace inequality_and_equality_condition_l2372_237246

variable {x y : ℝ}

theorem inequality_and_equality_condition
  (hx : 0 < x) (hy : 0 < y) :
  (x + y^2 / x ≥ 2 * y) ∧ (x + y^2 / x = 2 * y ↔ x = y) := sorry

end inequality_and_equality_condition_l2372_237246


namespace usual_time_to_office_l2372_237225

theorem usual_time_to_office (S T : ℝ) (h : T = 4 / 3 * (T + 8)) : T = 24 :=
by
  sorry

end usual_time_to_office_l2372_237225


namespace revenue_change_l2372_237202

theorem revenue_change (T C : ℝ) (T_new C_new : ℝ)
  (h1 : T_new = 0.81 * T)
  (h2 : C_new = 1.15 * C)
  (R : ℝ := T * C) : 
  ((T_new * C_new - R) / R) * 100 = -6.85 :=
by
  sorry

end revenue_change_l2372_237202


namespace policeman_catches_thief_l2372_237257

/-
  From a police station situated on a straight road infinite in both directions, a thief has stolen a police car.
  Its maximal speed equals 90% of the maximal speed of a police cruiser. When the theft is discovered some time
  later, a policeman starts to pursue the thief on a cruiser. However, the policeman does not know in which direction
  along the road the thief has gone, nor does he know how long ago the car has been stolen. The goal is to prove
  that it is possible for the policeman to catch the thief.
-/
theorem policeman_catches_thief (v : ℝ) (T₀ : ℝ) (o₀ : ℝ) :
  (0 < v) →
  (0 < T₀) →
  ∃ T p, T₀ ≤ T ∧ p ≤ v * T :=
sorry

end policeman_catches_thief_l2372_237257


namespace smallest_n_for_cookies_l2372_237208

theorem smallest_n_for_cookies :
  ∃ n : ℕ, 15 * n - 1 % 11 = 0 ∧ (∀ m : ℕ, 15 * m - 1 % 11 = 0 → n ≤ m) :=
sorry

end smallest_n_for_cookies_l2372_237208


namespace value_of_card_l2372_237236

/-- For this problem: 
    1. Matt has 8 baseball cards worth $6 each.
    2. He trades two of them to Jane in exchange for 3 $2 cards and a card of certain value.
    3. He makes a profit of $3.
    We need to prove that the value of the card that Jane gave to Matt apart from the $2 cards is $9. -/
theorem value_of_card (value_per_card traded_cards received_dollar_cards profit received_total_value : ℤ)
  (h1 : value_per_card = 6)
  (h2 : traded_cards = 2)
  (h3 : received_dollar_cards = 6)
  (h4 : profit = 3)
  (h5 : received_total_value = 15) :
  received_total_value - received_dollar_cards = 9 :=
by {
  -- This is just left as a placeholder to signal that the proof needs to be provided.
  sorry
}

end value_of_card_l2372_237236


namespace probability_of_two_queens_or_at_least_one_king_l2372_237223

def probability_two_queens_or_at_least_one_king : ℚ := 2 / 13

theorem probability_of_two_queens_or_at_least_one_king :
  let probability_two_queens := (4/52) * (3/51)
  let probability_exactly_one_king := (2 * (4/52) * (48/51))
  let probability_two_kings := (4/52) * (3/51)
  let probability_at_least_one_king := probability_exactly_one_king + probability_two_kings
  let total_probability := probability_two_queens + probability_at_least_one_king
  total_probability = probability_two_queens_or_at_least_one_king := 
by
  sorry

end probability_of_two_queens_or_at_least_one_king_l2372_237223


namespace apples_vs_cherries_l2372_237214

def pies_per_day : Nat := 12
def apple_days_per_week : Nat := 3
def cherry_days_per_week : Nat := 2

theorem apples_vs_cherries :
  (apple_days_per_week * pies_per_day) - (cherry_days_per_week * pies_per_day) = 12 := by
  sorry

end apples_vs_cherries_l2372_237214


namespace sum_x_coordinates_eq_3_l2372_237217

def f : ℝ → ℝ := sorry -- definition of the function f as given by the five line segments

theorem sum_x_coordinates_eq_3 :
  (∃ x1 x2 x3 : ℝ, (f x1 = x1 + 1 ∧ f x2 = x2 + 1 ∧ f x3 = x3 + 1) ∧ (x1 + x2 + x3 = 3)) :=
sorry

end sum_x_coordinates_eq_3_l2372_237217


namespace find_f2_l2372_237298

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := 
by
  sorry

end find_f2_l2372_237298


namespace pyramid_base_side_length_l2372_237271

theorem pyramid_base_side_length
  (area : ℝ)
  (slant_height : ℝ)
  (h : area = 90)
  (sh : slant_height = 15) :
  ∃ (s : ℝ), 90 = 1 / 2 * s * 15 ∧ s = 12 :=
by
  sorry

end pyramid_base_side_length_l2372_237271


namespace investment_ratio_l2372_237248

theorem investment_ratio (A_invest B_invest C_invest : ℝ) (F : ℝ) (total_profit B_share : ℝ)
  (h1 : A_invest = 3 * B_invest)
  (h2 : B_invest = F * C_invest)
  (h3 : total_profit = 7700)
  (h4 : B_share = 1400)
  (h5 : (B_invest / (A_invest + B_invest + C_invest)) * total_profit = B_share) :
  (B_invest / C_invest) = 2 / 3 := 
by
  sorry

end investment_ratio_l2372_237248


namespace angle_B_in_right_triangle_in_degrees_l2372_237204

def angleSum (A B C: ℝ) : Prop := A + B + C = 180

theorem angle_B_in_right_triangle_in_degrees (A B C : ℝ) (h1 : C = 90) (h2 : A = 35.5) (h3 : angleSum A B C) : B = 54.5 := 
by
  sorry

end angle_B_in_right_triangle_in_degrees_l2372_237204


namespace fraction_1790s_l2372_237279

def total_states : ℕ := 30
def states_1790s : ℕ := 16

theorem fraction_1790s : (states_1790s / total_states : ℚ) = 8 / 15 :=
by
  -- We claim that the fraction of states admitted during the 1790s is exactly 8/15
  sorry

end fraction_1790s_l2372_237279


namespace power_sum_eq_nine_l2372_237219

theorem power_sum_eq_nine {m n p q : ℕ} (h : ∀ x > 0, (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2 * n + p)^(2 * q) = 9 :=
sorry

end power_sum_eq_nine_l2372_237219


namespace daughters_meet_days_count_l2372_237285

noncomputable def days_elder_returns := 5
noncomputable def days_second_returns := 4
noncomputable def days_youngest_returns := 3

noncomputable def total_days := 100

-- Defining the count of individual and combined visits
noncomputable def count_individual_visits (period : ℕ) : ℕ := total_days / period
noncomputable def count_combined_visits (period1 : ℕ) (period2 : ℕ) : ℕ := total_days / Nat.lcm period1 period2
noncomputable def count_all_together_visits (periods : List ℕ) : ℕ := total_days / periods.foldr Nat.lcm 1

-- Specific counts
noncomputable def count_youngest_visits : ℕ := count_individual_visits days_youngest_returns
noncomputable def count_second_visits : ℕ := count_individual_visits days_second_returns
noncomputable def count_elder_visits : ℕ := count_individual_visits days_elder_returns

noncomputable def count_youngest_and_second : ℕ := count_combined_visits days_youngest_returns days_second_returns
noncomputable def count_youngest_and_elder : ℕ := count_combined_visits days_youngest_returns days_elder_returns
noncomputable def count_second_and_elder : ℕ := count_combined_visits days_second_returns days_elder_returns

noncomputable def count_all_three : ℕ := count_all_together_visits [days_youngest_returns, days_second_returns, days_elder_returns]

-- Final Inclusion-Exclusion principle application
noncomputable def days_at_least_one_returns : ℕ := 
  count_youngest_visits + count_second_visits + count_elder_visits
  - count_youngest_and_second
  - count_youngest_and_elder
  - count_second_and_elder
  + count_all_three

theorem daughters_meet_days_count : days_at_least_one_returns = 60 := by
  sorry

end daughters_meet_days_count_l2372_237285


namespace divisibility_3804_l2372_237258

theorem divisibility_3804 (n : ℕ) (h : 0 < n) :
    3804 ∣ ((n ^ 3 - n) * (5 ^ (8 * n + 4) + 3 ^ (4 * n + 2))) :=
sorry

end divisibility_3804_l2372_237258


namespace rectangle_perimeter_l2372_237252

-- We first define the side lengths of the squares and their relationships
def b1 : ℕ := 3
def b2 : ℕ := 9
def b3 := b1 + b2
def b4 := 2 * b1 + b2
def b5 := 3 * b1 + 2 * b2
def b6 := 3 * b1 + 3 * b2
def b7 := 4 * b1 + 3 * b2

-- Dimensions of the rectangle
def L := 37
def W := 52

-- Theorem to prove the perimeter of the rectangle
theorem rectangle_perimeter : 2 * L + 2 * W = 178 := by
  -- Proof will be provided here
  sorry

end rectangle_perimeter_l2372_237252


namespace intersection_value_l2372_237284

theorem intersection_value (x0 : ℝ) (h1 : -x0 = Real.tan x0) (h2 : x0 ≠ 0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
  sorry

end intersection_value_l2372_237284


namespace fractions_arith_l2372_237280

theorem fractions_arith : (3 / 50) + (2 / 25) - (5 / 1000) = 0.135 := by
  sorry

end fractions_arith_l2372_237280


namespace simplify_fraction_product_l2372_237277

theorem simplify_fraction_product : 
  (270 / 24) * (7 / 210) * (6 / 4) = 4.5 :=
by
  sorry

end simplify_fraction_product_l2372_237277


namespace simplify_expression_l2372_237267

noncomputable def sin_30 := 1 / 2
noncomputable def cos_30 := Real.sqrt 3 / 2

theorem simplify_expression :
  (sin_30 ^ 3 + cos_30 ^ 3) / (sin_30 + cos_30) = 1 - Real.sqrt 3 / 4 := sorry

end simplify_expression_l2372_237267


namespace age_ratio_4_years_hence_4_years_ago_l2372_237264

-- Definitions based on the conditions
def current_age_ratio (A B : ℕ) := 5 * B = 3 * A
def age_ratio_4_years_ago_4_years_hence (A B : ℕ) := A - 4 = B + 4

-- The main theorem to prove
theorem age_ratio_4_years_hence_4_years_ago (A B : ℕ) 
  (h1 : current_age_ratio A B) 
  (h2 : age_ratio_4_years_ago_4_years_hence A B) : 
  A + 4 = 3 * (B - 4) := 
sorry

end age_ratio_4_years_hence_4_years_ago_l2372_237264


namespace num_pos_int_solutions_2a_plus_3b_eq_15_l2372_237245

theorem num_pos_int_solutions_2a_plus_3b_eq_15 : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 2 * a + 3 * b = 15) ∧ 
  (∀ (a1 a2 b1 b2 : ℕ), 0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2 ∧ 
  (2 * a1 + 3 * b1 = 15) ∧ (2 * a2 + 3 * b2 = 15) → 
  ((a1 = 3 ∧ b1 = 3 ∨ a1 = 6 ∧ b1 = 1) ∧ (a2 = 3 ∧ b2 = 3 ∨ a2 = 6 ∧ b2 = 1))) := 
  sorry

end num_pos_int_solutions_2a_plus_3b_eq_15_l2372_237245


namespace calculate_difference_l2372_237269

variable (σ : ℝ) -- Let \square be represented by a real number σ
def correct_answer := 4 * (σ - 3)
def incorrect_answer := 4 * σ - 3
def difference := correct_answer σ - incorrect_answer σ

theorem calculate_difference : difference σ = -9 := by
  sorry

end calculate_difference_l2372_237269


namespace percent_profit_l2372_237274

theorem percent_profit (CP LP SP Profit : ℝ) 
  (hCP : CP = 100) 
  (hLP : LP = CP + 0.30 * CP)
  (hSP : SP = LP - 0.10 * LP) 
  (hProfit : Profit = SP - CP) : 
  (Profit / CP) * 100 = 17 :=
by
  sorry

end percent_profit_l2372_237274


namespace convert_to_scientific_notation_l2372_237262

def original_value : ℝ := 3462.23
def scientific_notation_value : ℝ := 3.46223 * 10^3

theorem convert_to_scientific_notation : 
  original_value = scientific_notation_value :=
sorry

end convert_to_scientific_notation_l2372_237262


namespace five_digit_number_unique_nonzero_l2372_237297

theorem five_digit_number_unique_nonzero (a b c d e : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) (h3 : (100 * a + 10 * b + c) * 7 = 100 * c + 10 * d + e) : a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 4 ∧ e = 6 :=
by
  sorry

end five_digit_number_unique_nonzero_l2372_237297


namespace annual_interest_rate_l2372_237276

noncomputable def compound_interest 
  (P : ℝ) (A : ℝ) (n : ℕ) (t : ℝ) (r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem annual_interest_rate 
  (P := 140) (A := 169.40) (n := 2) (t := 1) :
  ∃ r : ℝ, compound_interest P A n t r ∧ r = 0.2 :=
sorry

end annual_interest_rate_l2372_237276


namespace find_m_pure_imaginary_l2372_237294

noncomputable def find_m (m : ℝ) : ℝ := m

theorem find_m_pure_imaginary (m : ℝ) (h : (m^2 - 5 * m + 6 : ℂ) = 0) :
  find_m m = 2 :=
by
  sorry

end find_m_pure_imaginary_l2372_237294


namespace quadratic_sum_of_squares_l2372_237209

theorem quadratic_sum_of_squares (α β : ℝ) (h1 : α * β = 3) (h2 : α + β = 7) : α^2 + β^2 = 43 := 
by
  sorry

end quadratic_sum_of_squares_l2372_237209


namespace function_solution_l2372_237206

theorem function_solution (f : ℝ → ℝ) (H : ∀ x y : ℝ, 1 < x → 1 < y → f x - f y = (y - x) * f (x * y)) :
  ∃ k : ℝ, ∀ x : ℝ, 1 < x → f x = k / x :=
by
  sorry

end function_solution_l2372_237206


namespace problem1_problem2_problem3_l2372_237233

-- 1. Given: ∃ x ∈ ℤ, x^2 - 2x - 3 = 0
--    Show: ∀ x ∈ ℤ, x^2 - 2x - 3 ≠ 0
theorem problem1 : (∃ x : ℤ, x^2 - 2 * x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2 * x - 3 ≠ 0) := sorry

-- 2. Given: ∀ x ∈ ℝ, x^2 + 3 ≥ 2x
--    Show: ∃ x ∈ ℝ, x^2 + 3 < 2x
theorem problem2 : (∀ x : ℝ, x^2 + 3 ≥ 2 * x) ↔ (∃ x : ℝ, x^2 + 3 < 2 * x) := sorry

-- 3. Given: If x > 1 and y > 1, then x + y > 2
--    Show: If x ≤ 1 or y ≤ 1, then x + y ≤ 2
theorem problem3 : (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ↔ (∀ x y : ℝ, x ≤ 1 ∨ y ≤ 1 → x + y ≤ 2) := sorry

end problem1_problem2_problem3_l2372_237233


namespace balanced_number_example_l2372_237296

/--
A number is balanced if it is a three-digit number, all digits are different,
and it equals the sum of all possible two-digit numbers composed from its different digits.
-/
def isBalanced (n : ℕ) : Prop :=
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  (n = (10 * (n / 100) + (n / 10) % 10) + (10 * (n / 100) + n % 10) +
    (10 * ((n / 10) % 10) + n / 100) + (10 * ((n / 10) % 10) + n % 10) +
    (10 * (n % 10) + n / 100) + (10 * (n % 10) + ((n / 10) % 10)))

theorem balanced_number_example : isBalanced 132 :=
  sorry

end balanced_number_example_l2372_237296


namespace percentage_of_students_passed_l2372_237270

theorem percentage_of_students_passed
  (students_failed : ℕ)
  (total_students : ℕ)
  (H_failed : students_failed = 260)
  (H_total : total_students = 400)
  (passed := total_students - students_failed) :
  (passed * 100 / total_students : ℝ) = 35 := 
by
  -- proof steps would go here
  sorry

end percentage_of_students_passed_l2372_237270


namespace initial_welders_count_l2372_237283

theorem initial_welders_count
  (W : ℕ)
  (complete_in_5_days : W * 5 = 1)
  (leave_after_1_day : 12 ≤ W) 
  (remaining_complete_in_6_days : (W - 12) * 6 = 1) : 
  W = 72 :=
by
  -- proof steps here
  sorry

end initial_welders_count_l2372_237283


namespace bee_fraction_remaining_l2372_237292

theorem bee_fraction_remaining (N : ℕ) (L : ℕ) (D : ℕ) (hN : N = 80000) (hL : L = 1200) (hD : D = 50) :
  (N - (L * D)) / N = 1 / 4 :=
by
  sorry

end bee_fraction_remaining_l2372_237292


namespace calculate_polynomial_value_l2372_237263

theorem calculate_polynomial_value (a a1 a2 a3 a4 a5 : ℝ) : 
  (∀ x : ℝ, (1 - x)^2 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) → 
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by 
  intro h
  sorry

end calculate_polynomial_value_l2372_237263


namespace brett_blue_marbles_more_l2372_237231

theorem brett_blue_marbles_more (r b : ℕ) (hr : r = 6) (hb : b = 5 * r) : b - r = 24 := by
  rw [hr, hb]
  norm_num
  sorry

end brett_blue_marbles_more_l2372_237231


namespace problem_inequality_l2372_237293

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l2372_237293


namespace football_game_spectators_l2372_237235

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ) (h1 : total_wristbands = 234) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end football_game_spectators_l2372_237235


namespace sum_of_nine_consecutive_quotients_multiple_of_9_l2372_237240

def a (i : ℕ) : ℕ := (10^(2 * i) - 1) / 9
def q (i : ℕ) : ℕ := a i / 11
def s (i : ℕ) : ℕ := q i + q (i + 1) + q (i + 2) + q (i + 3) + q (i + 4) + q (i + 5) + q (i + 6) + q (i + 7) + q (i + 8)

theorem sum_of_nine_consecutive_quotients_multiple_of_9 (i n : ℕ) (h : n > 8) 
  (h2 : i ≤ n - 8) : s i % 9 = 0 :=
sorry

end sum_of_nine_consecutive_quotients_multiple_of_9_l2372_237240


namespace distinct_dress_designs_l2372_237290

theorem distinct_dress_designs : 
  let num_colors := 5
  let num_patterns := 6
  num_colors * num_patterns = 30 :=
by
  sorry

end distinct_dress_designs_l2372_237290


namespace range_f_l2372_237224

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x - Real.cos x

theorem range_f : Set.range f = Set.Icc (-2 : ℝ) 2 := 
by
  sorry

end range_f_l2372_237224


namespace fitted_bowling_ball_volume_correct_l2372_237273

noncomputable def volume_of_fitted_bowling_ball : ℝ :=
  let ball_radius := 12
  let ball_volume := (4/3) * Real.pi * ball_radius^3
  let hole1_radius := 1
  let hole1_volume := Real.pi * hole1_radius^2 * 6
  let hole2_radius := 1.25
  let hole2_volume := Real.pi * hole2_radius^2 * 6
  let hole3_radius := 2
  let hole3_volume := Real.pi * hole3_radius^2 * 6
  ball_volume - (hole1_volume + hole2_volume + hole3_volume)

theorem fitted_bowling_ball_volume_correct :
  volume_of_fitted_bowling_ball = 2264.625 * Real.pi := by
  -- proof would go here
  sorry

end fitted_bowling_ball_volume_correct_l2372_237273


namespace line_equation_l2372_237228

theorem line_equation
  (x y : ℝ)
  (h1 : 2 * x + y + 2 = 0)
  (h2 : 2 * x - y + 2 = 0)
  (h3 : ∀ x y, x + y = 0 → x - 1 = y): 
  x - y + 1 = 0 :=
sorry

end line_equation_l2372_237228


namespace translated_line_expression_l2372_237261

theorem translated_line_expression (x y : ℝ) (b : ℝ) :
  (∀ x y, y = 2 * x + 3 ∧ (5, 1).2 = 2 * (5, 1).1 + b) → y = 2 * x - 9 :=
by
  sorry

end translated_line_expression_l2372_237261


namespace simplify_expression_l2372_237256

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l2372_237256


namespace xy_sq_is_37_over_36_l2372_237281

theorem xy_sq_is_37_over_36 (x y : ℚ) (h : 2002 * (x - 1)^2 + |x - 12 * y + 1| = 0) : x^2 + y^2 = 37 / 36 :=
sorry

end xy_sq_is_37_over_36_l2372_237281


namespace value_of_f_l2372_237201

noncomputable
def f (k l m x : ℝ) : ℝ := k + m / (x - l)

theorem value_of_f (k l m : ℝ) (hk : k = -2) (hl : l = 2.5) (hm : m = 12) :
  f k l m (k + l + m) = -4 / 5 :=
by
  sorry

end value_of_f_l2372_237201


namespace jonas_socks_solution_l2372_237299

theorem jonas_socks_solution (p_s p_h n_p n_t n : ℕ) (h_ps : p_s = 20) (h_ph : p_h = 5) (h_np : n_p = 10) (h_nt : n_t = 10) :
  2 * (p_s * 2 + p_h * 2 + n_p + n_t) = 2 * (p_s * 2 + p_h * 2 + n_p + n_t + n * 2) :=
by
  -- skipping the proof part
  sorry

end jonas_socks_solution_l2372_237299


namespace smallest_product_of_digits_l2372_237278

theorem smallest_product_of_digits : 
  ∃ (a b c d : ℕ), 
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) ∧ 
  (∃ x y : ℕ, (x = a * 10 + c ∧ y = b * 10 + d) ∨ (x = a * 10 + d ∧ y = b * 10 + c) ∨ (x = b * 10 + c ∧ y = a * 10 + d) ∨ (x = b * 10 + d ∧ y = a * 10 + c)) ∧
  (∀ x1 y1 x2 y2 : ℕ, ((x1 = 34 ∧ y1 = 56 ∨ x1 = 35 ∧ y1 = 46) ∧ (x2 = 34 ∧ y2 = 56 ∨ x2 = 35 ∧ y2 = 46)) → x1 * y1 ≥ x2 * y2) ∧
  35 * 46 = 1610 :=
sorry

end smallest_product_of_digits_l2372_237278


namespace minimum_distance_on_circle_l2372_237220

open Complex

noncomputable def minimum_distance (z : ℂ) : ℝ :=
  abs (z - (1 + 2*I))

theorem minimum_distance_on_circle :
  ∀ z : ℂ, abs (z + 2 - 2*I) = 1 → minimum_distance z = 2 :=
by
  intros z hz
  -- Proof is omitted
  sorry

end minimum_distance_on_circle_l2372_237220


namespace angle_BDC_is_30_l2372_237291

theorem angle_BDC_is_30 
    (A E C B D : ℝ) 
    (hA : A = 50) 
    (hE : E = 60) 
    (hC : C = 40) : 
    BDC = 30 :=
by
  sorry

end angle_BDC_is_30_l2372_237291
