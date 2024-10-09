import Mathlib

namespace binom_600_600_l2133_213357

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l2133_213357


namespace harry_worked_41_hours_l2133_213332

def james_earnings (x : ℝ) : ℝ :=
  (40 * x) + (7 * 2 * x)

def harry_earnings (x : ℝ) (h : ℝ) : ℝ :=
  (24 * x) + (11 * 1.5 * x) + (2 * h * x)

def harry_hours_worked (h : ℝ) : ℝ :=
  24 + 11 + h

theorem harry_worked_41_hours (x : ℝ) (h : ℝ) 
  (james_worked : james_earnings x = 54 * x)
  (harry_paid_same : harry_earnings x h = james_earnings x) :
  harry_hours_worked h = 41 :=
by
  -- sorry is used to skip the proof steps
  sorry

end harry_worked_41_hours_l2133_213332


namespace m₁_m₂_relationship_l2133_213375

-- Defining the conditions
variables {Point Line : Type}
variables (intersect : Line → Line → Prop)
variables (coplanar : Line → Line → Prop)

-- Assumption that lines l₁ and l₂ are non-coplanar.
variables {l₁ l₂ : Line} (h_non_coplanar : ¬ coplanar l₁ l₂)

-- Assuming m₁ and m₂ both intersect with l₁ and l₂.
variables {m₁ m₂ : Line}
variables (h_intersect_m₁_l₁ : intersect m₁ l₁)
variables (h_intersect_m₁_l₂ : intersect m₁ l₂)
variables (h_intersect_m₂_l₁ : intersect m₂ l₁)
variables (h_intersect_m₂_l₂ : intersect m₂ l₂)

-- Statement to prove that m₁ and m₂ are either intersecting or non-coplanar.
theorem m₁_m₂_relationship :
  (¬ coplanar m₁ m₂) ∨ (∃ p : Point, (intersect m₁ m₂ ∧ intersect m₂ m₁)) :=
sorry

end m₁_m₂_relationship_l2133_213375


namespace sequence_value_238_l2133_213369

theorem sequence_value_238 (a : ℕ → ℚ) :
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → (n % 2 = 0 → a n = a (n - 1) / 2 + 1) ∧ (n % 2 = 1 → a n = 1 / a (n - 1))) ∧
  (∃ n, a n = 30 / 19) → ∃ n, a n = 30 / 19 ∧ n = 238 :=
by
  sorry

end sequence_value_238_l2133_213369


namespace solve_equation_l2133_213316

theorem solve_equation (x : ℝ) :
  ((x - 2)^2 - 4 = 0) ↔ (x = 4 ∨ x = 0) :=
by
  sorry

end solve_equation_l2133_213316


namespace solution_alcohol_content_l2133_213368

noncomputable def volume_of_solution_y_and_z (V: ℝ) : Prop :=
  let vol_X := 300.0
  let conc_X := 0.10
  let conc_Y := 0.30
  let conc_Z := 0.40
  let vol_Y := 2 * V
  let vol_new := vol_X + vol_Y + V
  let alcohol_new := conc_X * vol_X + conc_Y * vol_Y + conc_Z * V
  (alcohol_new / vol_new) = 0.22

theorem solution_alcohol_content : volume_of_solution_y_and_z 300.0 :=
by
  sorry

end solution_alcohol_content_l2133_213368


namespace magnitude_of_complex_l2133_213348

open Complex

theorem magnitude_of_complex : abs (Complex.mk (3/4) (-5/6)) = Real.sqrt (181) / 12 :=
by
  sorry

end magnitude_of_complex_l2133_213348


namespace percentage_of_rotten_bananas_l2133_213338

theorem percentage_of_rotten_bananas :
  ∀ (total_oranges total_bananas : ℕ) 
    (percent_rotten_oranges : ℝ) 
    (percent_good_fruits : ℝ), 
  total_oranges = 600 → total_bananas = 400 → 
  percent_rotten_oranges = 0.15 → percent_good_fruits = 0.89 → 
  (100 - (((percent_good_fruits * (total_oranges + total_bananas)) - 
  ((1 - percent_rotten_oranges) * total_oranges)) / total_bananas) * 100) = 5 := 
by
  intros total_oranges total_bananas percent_rotten_oranges percent_good_fruits 
  intro ho hb hro hpf 
  sorry

end percentage_of_rotten_bananas_l2133_213338


namespace sum_of_four_squares_l2133_213396

theorem sum_of_four_squares (a b c : ℕ) 
    (h1 : 2 * a + b + c = 27)
    (h2 : 2 * b + a + c = 25)
    (h3 : 3 * c + a = 39) : 4 * c = 44 := 
  sorry

end sum_of_four_squares_l2133_213396


namespace melissa_total_points_l2133_213303

-- Definition of the points scored per game and the number of games played.
def points_per_game : ℕ := 7
def number_of_games : ℕ := 3

-- The total points scored by Melissa is defined as the product of points per game and number of games.
def total_points_scored : ℕ := points_per_game * number_of_games

-- The theorem stating the verification of the total points scored by Melissa.
theorem melissa_total_points : total_points_scored = 21 := by
  -- The proof will be given here.
  sorry

end melissa_total_points_l2133_213303


namespace correlation_relationships_l2133_213359

-- Let's define the relationships as conditions
def volume_cube_edge_length (v e : ℝ) : Prop := v = e^3
def yield_fertilizer (yield fertilizer : ℝ) : Prop := True -- Assume linear correlation within a certain range
def height_age (height age : ℝ) : Prop := True -- Assume linear correlation within a certain age range
def expenses_income (expenses income : ℝ) : Prop := True -- Assume linear correlation
def electricity_consumption_price (consumption price unit_price : ℝ) : Prop := price = consumption * unit_price

-- We want to prove that the answers correspond correctly to the conditions:
theorem correlation_relationships :
  ∀ (v e yield fertilizer height age expenses income consumption price unit_price : ℝ),
  ¬ volume_cube_edge_length v e ∧ yield_fertilizer yield fertilizer ∧ height_age height age ∧ expenses_income expenses income ∧ ¬ electricity_consumption_price consumption price unit_price → 
  "D" = "②③④" :=
by
  intros
  sorry

end correlation_relationships_l2133_213359


namespace find_c_for_radius_6_l2133_213306

-- Define the circle equation and the radius condition.
theorem find_c_for_radius_6 (c : ℝ) :
  (∃ (x y : ℝ), x^2 + 8 * x + y^2 + 2 * y + c = 0) ∧ 6 = 6 -> c = -19 := 
by
  sorry

end find_c_for_radius_6_l2133_213306


namespace line_segment_parametric_curve_l2133_213363

noncomputable def parametric_curve (θ : ℝ) := 
  (2 + Real.cos θ ^ 2, 1 - Real.sin θ ^ 2)

theorem line_segment_parametric_curve : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → 
    ∃ x y : ℝ, (x, y) = parametric_curve θ ∧ 2 ≤ x ∧ x ≤ 3 ∧ x - y = 2) := 
sorry

end line_segment_parametric_curve_l2133_213363


namespace exists_person_who_knows_everyone_l2133_213320

variable {Person : Type}
variable (knows : Person → Person → Prop)
variable (n : ℕ)

-- Condition: In a company of 2n + 1 people, for any n people, there is another person different from them who knows each of them.
axiom knows_condition : ∀ (company : Finset Person) (h : company.card = 2 * n + 1), 
  (∀ (subset : Finset Person) (hs : subset.card = n), ∃ (p : Person), p ∉ subset ∧ ∀ q ∈ subset, knows p q)

-- Statement to be proven:
theorem exists_person_who_knows_everyone (company : Finset Person) (hcompany : company.card = 2 * n + 1) :
  ∃ p, ∀ q ∈ company, knows p q :=
sorry

end exists_person_who_knows_everyone_l2133_213320


namespace measure_of_angle_C_l2133_213310

theorem measure_of_angle_C (a b area : ℝ) (C : ℝ) :
  a = 5 → b = 8 → area = 10 →
  (1 / 2 * a * b * Real.sin C = area) →
  (C = Real.pi / 6 ∨ C = 5 * Real.pi / 6) := by
  intros ha hb harea hformula
  sorry

end measure_of_angle_C_l2133_213310


namespace value_of_a2_b2_l2133_213341

theorem value_of_a2_b2 (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a - i) * i = b - i) : a^2 + b^2 = 2 :=
by sorry

end value_of_a2_b2_l2133_213341


namespace product_of_x1_to_x13_is_zero_l2133_213371

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end product_of_x1_to_x13_is_zero_l2133_213371


namespace percent_defective_units_shipped_for_sale_l2133_213378

theorem percent_defective_units_shipped_for_sale 
  (P : ℝ) -- total number of units produced
  (h_defective : 0.06 * P = d) -- 6 percent of units are defective
  (h_shipped : 0.0024 * P = s) -- 0.24 percent of units are defective units shipped for sale
  : (s / d) * 100 = 4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l2133_213378


namespace physics_class_size_l2133_213326

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 100)
  (h2 : physics_only + math_only + both = total_students)
  (h3 : both = 10)
  (h4 : physics_only + both = 2 * (math_only + both)) :
  physics_only + both = 62 := 
by sorry

end physics_class_size_l2133_213326


namespace triangle_angle_sum_l2133_213373

theorem triangle_angle_sum {x : ℝ} (h : 60 + 5 * x + 3 * x = 180) : x = 15 :=
by
  sorry

end triangle_angle_sum_l2133_213373


namespace airplane_seat_count_l2133_213382

theorem airplane_seat_count (s : ℝ) 
  (h1 : 30 + 0.2 * s + 0.75 * s = s) : 
  s = 600 :=
sorry

end airplane_seat_count_l2133_213382


namespace fraction_sum_in_simplest_form_l2133_213395

theorem fraction_sum_in_simplest_form :
  ∃ a b : ℕ, a + b = 11407 ∧ 0.425875 = a / (b : ℝ) ∧ Nat.gcd a b = 1 :=
by
  sorry

end fraction_sum_in_simplest_form_l2133_213395


namespace total_students_in_class_l2133_213345

variable (K M Both Total : ℕ)

theorem total_students_in_class
  (hK : K = 38)
  (hM : M = 39)
  (hBoth : Both = 32)
  (hTotal : Total = K + M - Both) :
  Total = 45 := 
by
  rw [hK, hM, hBoth] at hTotal
  exact hTotal

end total_students_in_class_l2133_213345


namespace hyperbola_asymptote_b_l2133_213397

theorem hyperbola_asymptote_b {b : ℝ} (hb : b > 0) :
  (∀ x y : ℝ, x^2 - (y^2 / b^2) = 1 → (y = 2 * x)) → b = 2 := by
  sorry

end hyperbola_asymptote_b_l2133_213397


namespace max_profit_at_one_device_l2133_213353

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2

def fixed_monthly_cost : ℝ := 40

def material_cost_per_device : ℝ := 5

noncomputable def cost (x : ℕ) : ℝ := fixed_monthly_cost + material_cost_per_device * x

noncomputable def profit_function (x : ℕ) : ℝ := (revenue x) - (cost x)

noncomputable def marginal_profit_function (x : ℕ) : ℝ :=
  profit_function (x + 1) - profit_function x

theorem max_profit_at_one_device :
  marginal_profit_function 1 = 24.4 ∧
  ∀ x : ℕ, marginal_profit_function x ≤ 24.4 := sorry

end max_profit_at_one_device_l2133_213353


namespace sum_of_integers_l2133_213305

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 300) (h2 : m * (m + 1) * (m + 2) = 300) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 49 := 
by sorry

end sum_of_integers_l2133_213305


namespace difference_approx_l2133_213398

-- Let L be the larger number and S be the smaller number
variables (L S : ℝ)

-- Conditions given:
-- 1. L is approximately 1542.857
def approx_L : Prop := abs (L - 1542.857) < 1

-- 2. When L is divided by S, quotient is 8 and remainder is 15
def division_condition : Prop := L = 8 * S + 15

-- The theorem stating the difference L - S is approximately 1351.874
theorem difference_approx (hL : approx_L L) (hdiv : division_condition L S) :
  abs ((L - S) - 1351.874) < 1 :=
sorry

#check difference_approx

end difference_approx_l2133_213398


namespace man_twice_son_age_in_years_l2133_213366

theorem man_twice_son_age_in_years :
  ∀ (S M Y : ℕ),
  (M = S + 26) →
  (S = 24) →
  (M + Y = 2 * (S + Y)) →
  Y = 2 :=
by
  intros S M Y h1 h2 h3
  sorry

end man_twice_son_age_in_years_l2133_213366


namespace f_x_when_x_negative_l2133_213356

-- Define the properties of the function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → f x = x * (1 + x)

-- The theorem we want to prove
theorem f_x_when_x_negative (f : ℝ → ℝ) 
  (h1: odd_function f)
  (h2: f_definition f) : 
  ∀ x, x < 0 → f x = -x * (1 - x) :=
by
  sorry

end f_x_when_x_negative_l2133_213356


namespace provisions_last_60_days_l2133_213389

/-
A garrison of 1000 men has provisions for a certain number of days.
At the end of 15 days, a reinforcement of 1250 arrives, and it is now found that the provisions will last only for 20 days more.
Prove that the provisions were supposed to last initially for 60 days.
-/

def initial_provisions (D : ℕ) : Prop :=
  let initial_garrison := 1000
  let reinforcement_garrison := 1250
  let days_spent := 15
  let remaining_days := 20
  initial_garrison * (D - days_spent) = (initial_garrison + reinforcement_garrison) * remaining_days

theorem provisions_last_60_days (D : ℕ) : initial_provisions D → D = 60 := by
  sorry

end provisions_last_60_days_l2133_213389


namespace correct_operation_l2133_213351

variable (a b : ℝ)

theorem correct_operation : (a^2 * a^3 = a^5) :=
by sorry

end correct_operation_l2133_213351


namespace oil_drop_probability_l2133_213380

theorem oil_drop_probability :
  let r_circle := 1 -- radius of the circle in cm
  let side_square := 0.5 -- side length of the square in cm
  let area_circle := π * r_circle^2
  let area_square := side_square * side_square
  (area_square / area_circle) = 1 / (4 * π) :=
by
  sorry

end oil_drop_probability_l2133_213380


namespace triangle_height_l2133_213325

theorem triangle_height (b h : ℕ) (A : ℕ) (hA : A = 50) (hb : b = 10) :
  A = (1 / 2 : ℝ) * b * h → h = 10 := 
by
  sorry

end triangle_height_l2133_213325


namespace number_of_music_files_l2133_213394

-- The conditions given in the problem
variable {M : ℕ} -- M is a natural number representing the initial number of music files

-- Conditions: Initial state and changes
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23
def remaining_files : ℕ := 2

-- Statement of the theorem
theorem number_of_music_files (h : M + initial_video_files - files_deleted = remaining_files) : M = 4 :=
  by
  -- Proof goes here
  sorry

end number_of_music_files_l2133_213394


namespace length_dg_l2133_213302

theorem length_dg (a b k l S : ℕ) (h1 : S = 47 * (a + b)) 
                   (h2 : S = a * k) (h3 : S = b * l) (h4 : b = S / l) 
                   (h5 : a = S / k) (h6 : k * l = 47 * k + 47 * l + 2209) : 
  k = 2256 :=
by sorry

end length_dg_l2133_213302


namespace walk_to_bus_stop_usual_time_l2133_213337

variable (S : ℝ) -- assuming S is the usual speed, a positive real number
variable (T : ℝ) -- assuming T is the usual time, which we need to determine
variable (new_speed : ℝ := (4 / 5) * S) -- the new speed is 4/5 of usual speed
noncomputable def time_to_bus_at_usual_speed : ℝ := T -- time to bus stop at usual speed

theorem walk_to_bus_stop_usual_time :
  (time_to_bus_at_usual_speed S = 30) ↔ (S * (T + 6) = (4 / 5) * S * T) :=
by
  sorry

end walk_to_bus_stop_usual_time_l2133_213337


namespace geometric_sequence_sum_product_l2133_213374

theorem geometric_sequence_sum_product {a b c : ℝ} : 
  a + b + c = 14 → 
  a * b * c = 64 → 
  (a = 8 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) :=
by
  sorry

end geometric_sequence_sum_product_l2133_213374


namespace compute_value_l2133_213340

theorem compute_value : ((-120) - (-60)) / (-30) = 2 := 
by 
  sorry

end compute_value_l2133_213340


namespace turtles_on_Happy_Island_l2133_213355

theorem turtles_on_Happy_Island (L H : ℕ) (hL : L = 25) (hH : H = 2 * L + 10) : H = 60 :=
by
  sorry

end turtles_on_Happy_Island_l2133_213355


namespace total_fence_used_l2133_213308

-- Definitions based on conditions
variables {L W : ℕ}
def area (L W : ℕ) := L * W

-- Provided conditions as Lean definitions
def unfenced_side := 40
def yard_area := 240

-- The proof problem statement
theorem total_fence_used (L_eq : L = unfenced_side) (A_eq : area L W = yard_area) : (2 * W + L) = 52 :=
sorry

end total_fence_used_l2133_213308


namespace base9_first_digit_is_4_l2133_213361

-- Define the base three representation of y
def y_base3 : Nat := 112211

-- Function to convert a given number from base 3 to base 10
def base3_to_base10 (n : Nat) : Nat :=
  let rec convert (n : Nat) (acc : Nat) (place : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * (3 ^ place)) (place + 1)
  convert n 0 0

-- Compute the base 10 representation of y
def y_base10 : Nat := base3_to_base10 y_base3

-- Function to convert a given number from base 10 to base 9
def base10_to_base9 (n : Nat) : List Nat :=
  let rec convert (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc
    else convert (n / 9) ((n % 9) :: acc)
  convert n []

-- Compute the base 9 representation of y as a list of digits
def y_base9 : List Nat := base10_to_base9 y_base10

-- Get the first digit (most significant digit) of the base 9 representation of y
def first_digit_base9 (digits : List Nat) : Nat :=
  digits.headD 0

-- The statement to prove
theorem base9_first_digit_is_4 : first_digit_base9 y_base9 = 4 := by sorry

end base9_first_digit_is_4_l2133_213361


namespace gasoline_expense_l2133_213334

-- Definitions for the conditions
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10
def initial_amount : ℝ := 50
def amount_left_for_return_trip : ℝ := 36.35

-- Definition for the total gift cost
def total_gift_cost : ℝ := number_of_people * gift_cost_per_person

-- Definition for the total amount received from grandma
def total_grandma_gift : ℝ := number_of_people * grandma_gift_per_person

-- Definition for the total initial amount including the gift from grandma
def total_initial_amount_with_gift : ℝ := initial_amount + total_grandma_gift

-- Definition for remaining amount after spending on lunch and gifts
def remaining_after_known_expenses : ℝ := total_initial_amount_with_gift - lunch_cost - total_gift_cost

-- The Lean theorem to prove the gasoline expense
theorem gasoline_expense : remaining_after_known_expenses - amount_left_for_return_trip = 8 := by
  sorry

end gasoline_expense_l2133_213334


namespace intersection_complement_eq_empty_l2133_213390

open Set

variable {α : Type*} (M N U: Set α)

theorem intersection_complement_eq_empty (h : M ⊆ N) : M ∩ (compl N) = ∅ :=
sorry

end intersection_complement_eq_empty_l2133_213390


namespace pencils_bought_l2133_213343

theorem pencils_bought (total_spent notebook_cost ruler_cost pencil_cost : ℕ)
  (h_total : total_spent = 74)
  (h_notebook : notebook_cost = 35)
  (h_ruler : ruler_cost = 18)
  (h_pencil : pencil_cost = 7) :
  (total_spent - (notebook_cost + ruler_cost)) / pencil_cost = 3 :=
by
  sorry

end pencils_bought_l2133_213343


namespace gcd_decomposition_l2133_213399

open Polynomial

noncomputable def f : Polynomial ℚ := 4 * X ^ 4 - 2 * X ^ 3 - 16 * X ^ 2 + 5 * X + 9
noncomputable def g : Polynomial ℚ := 2 * X ^ 3 - X ^ 2 - 5 * X + 4

theorem gcd_decomposition :
  ∃ (u v : Polynomial ℚ), u * f + v * g = X - 1 :=
sorry

end gcd_decomposition_l2133_213399


namespace new_pressure_eq_l2133_213342

-- Defining the initial conditions and values
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3.5
def new_volume : ℝ := 10.5
def k : ℝ := initial_pressure * initial_volume

-- The statement to prove
theorem new_pressure_eq :
  ∃ p_new : ℝ, new_volume * p_new = k ∧ p_new = 8 / 3 :=
by
  use (8 / 3)
  sorry

end new_pressure_eq_l2133_213342


namespace greatest_savings_by_choosing_boat_l2133_213372

/-- Given the transportation costs:
     - plane cost: $600.00
     - boat cost: $254.00
     - helicopter cost: $850.00
    Prove that the greatest amount of money saved by choosing the boat over the other options is $596.00. -/
theorem greatest_savings_by_choosing_boat :
  let plane_cost := 600
  let boat_cost := 254
  let helicopter_cost := 850
  max (plane_cost - boat_cost) (helicopter_cost - boat_cost) = 596 :=
by
  sorry

end greatest_savings_by_choosing_boat_l2133_213372


namespace solve_for_x_l2133_213392

theorem solve_for_x (x : ℝ) :
  (2 * x - 30) / 3 = (5 - 3 * x) / 4 + 1 → x = 147 / 17 := 
by
  intro h
  sorry

end solve_for_x_l2133_213392


namespace sum_gcd_lcm_eq_4851_l2133_213379

theorem sum_gcd_lcm_eq_4851 (a b : ℕ) (ha : a = 231) (hb : b = 4620) :
  Nat.gcd a b + Nat.lcm a b = 4851 :=
by
  rw [ha, hb]
  sorry

end sum_gcd_lcm_eq_4851_l2133_213379


namespace problem_l2133_213317

-- Define the conditions
variables (x y : ℝ)
axiom h1 : 2 * x + y = 7
axiom h2 : x + 2 * y = 5

-- Statement of the problem
theorem problem : (2 * x * y) / 3 = 2 :=
by 
  -- Proof is omitted, but you should replace 'sorry' by the actual proof
  sorry

end problem_l2133_213317


namespace compute_expression_eq_162_l2133_213323

theorem compute_expression_eq_162 : 
  3 * 3^4 - 9^35 / 9^33 = 162 := 
by 
  sorry

end compute_expression_eq_162_l2133_213323


namespace find_equation_of_tangent_line_perpendicular_l2133_213377

noncomputable def tangent_line_perpendicular_to_curve (a b : ℝ) : Prop :=
  (∃ (P : ℝ × ℝ), P = (-1, -3) ∧ 2 * P.1 - 6 * P.2 + 1 = 0 ∧ P.2 = P.1^3 + 5 * P.1^2 - 5) ∧
  (-3) = 3 * (-1)^2 + 6 * (-1)

theorem find_equation_of_tangent_line_perpendicular :
  tangent_line_perpendicular_to_curve (-1) (-3) →
  ∀ x y : ℝ, 3 * x + y + 6 = 0 :=
by
  sorry

end find_equation_of_tangent_line_perpendicular_l2133_213377


namespace max_value_y_l2133_213388

open Real

theorem max_value_y (x : ℝ) (h : -1 < x ∧ x < 1) : 
  ∃ y_max, y_max = 0 ∧ ∀ y, y = x / (x - 1) + x → y ≤ y_max :=
by
  have y : ℝ := x / (x - 1) + x
  use 0
  sorry

end max_value_y_l2133_213388


namespace karen_savings_over_30_years_l2133_213318

theorem karen_savings_over_30_years 
  (P_exp : ℕ) (L_exp : ℕ) 
  (P_cheap : ℕ) (L_cheap : ℕ) 
  (T : ℕ)
  (hP_exp : P_exp = 300)
  (hL_exp : L_exp = 15)
  (hP_cheap : P_cheap = 120)
  (hL_cheap : L_cheap = 5)
  (hT : T = 30) : 
  (P_cheap * (T / L_cheap) - P_exp * (T / L_exp)) = 120 := 
by 
  sorry

end karen_savings_over_30_years_l2133_213318


namespace smallest_a_value_l2133_213354

theorem smallest_a_value 
  (a b c : ℚ) 
  (a_pos : a > 0)
  (vertex_condition : ∃(x₀ y₀ : ℚ), x₀ = -1/3 ∧ y₀ = -4/3 ∧ y = a * (x + x₀)^2 + y₀)
  (integer_condition : ∃(n : ℤ), a + b + c = n)
  : a = 3/16 := 
sorry

end smallest_a_value_l2133_213354


namespace intersection_l2133_213336

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / (2 * x - 6)

noncomputable def g (x : ℝ) (a b c d k : ℝ) : ℝ := -2 * x - 4 + k / (x - d)

theorem intersection (a b c k : ℝ) (h_d : d = 3) (h_k : k = 36) : 
  ∃ (x y : ℝ), x ≠ -3 ∧ (f x = g x 0 0 0 d k) ∧ (x, y) = (6.8, -32 / 19) :=
by
  sorry

end intersection_l2133_213336


namespace bridge_length_is_100_l2133_213335

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (wind_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let wind_speed_ms := wind_speed_kmh * 1000 / 3600
  let effective_speed_ms := train_speed_ms - wind_speed_ms
  let distance_covered := effective_speed_ms * crossing_time_s
  distance_covered - train_length

theorem bridge_length_is_100 :
  length_of_bridge 150 45 15 30 = 100 :=
by
  sorry

end bridge_length_is_100_l2133_213335


namespace sqrt_17_estimation_l2133_213304

theorem sqrt_17_estimation :
  4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := 
sorry

end sqrt_17_estimation_l2133_213304


namespace percentage_problem_l2133_213349

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by 
  sorry

end percentage_problem_l2133_213349


namespace smallest_abs_sum_l2133_213301

open Matrix

noncomputable def matrix_square_eq (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![a, b], ![c, d]] * ![![a, b], ![c, d]]

theorem smallest_abs_sum (a b c d : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
    (M_eq : matrix_square_eq a b c d = ![![9, 0], ![0, 9]]) :
    |a| + |b| + |c| + |d| = 8 :=
sorry

end smallest_abs_sum_l2133_213301


namespace find_k_for_circle_of_radius_8_l2133_213362

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end find_k_for_circle_of_radius_8_l2133_213362


namespace more_chickens_than_chicks_l2133_213393

-- Let's define the given conditions
def total : Nat := 821
def chicks : Nat := 267

-- The statement we need to prove
theorem more_chickens_than_chicks : (total - chicks) - chicks = 287 :=
by
  -- This is needed for the proof and not part of conditions
  -- Add sorry as a placeholder for proof steps 
  sorry

end more_chickens_than_chicks_l2133_213393


namespace jace_total_distance_l2133_213300

noncomputable def total_distance (s1 s2 s3 s4 s5 : ℝ) (t1 t2 t3 t4 t5 : ℝ) : ℝ :=
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5

theorem jace_total_distance :
  total_distance 50 65 60 75 55 3 4.5 2.75 1.8333 2.6667 = 891.67 := by
  sorry

end jace_total_distance_l2133_213300


namespace sunglasses_and_cap_probability_l2133_213358

/-
On a beach:
  - 50 people are wearing sunglasses.
  - 35 people are wearing caps.
  - The probability that randomly selected person wearing a cap is also wearing sunglasses is 2/5.
  
Prove that the probability that a randomly selected person wearing sunglasses is also wearing a cap is 7/25.
-/

theorem sunglasses_and_cap_probability :
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * total_caps
  (both / total_sunglasses) = (7 : ℚ) / 25 :=
by
  -- definitions
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * (total_caps : ℚ)
  have prob : (both / (total_sunglasses : ℚ)) = (7 : ℚ) / 25 := sorry
  exact prob

end sunglasses_and_cap_probability_l2133_213358


namespace compute_value_of_expression_l2133_213381

theorem compute_value_of_expression (p q : ℝ) (h1 : 3 * p^2 - 7 * p + 1 = 0) (h2 : 3 * q^2 - 7 * q + 1 = 0) :
  (9 * p^3 - 9 * q^3) / (p - q) = 46 :=
sorry

end compute_value_of_expression_l2133_213381


namespace part1_3kg_part2_5kg_part2_function_part3_compare_l2133_213315

noncomputable def supermarket_A_cost (x : ℝ) : ℝ :=
if x <= 4 then 10 * x
else 6 * x + 16

noncomputable def supermarket_B_cost (x : ℝ) : ℝ :=
8 * x

-- Proof that supermarket_A_cost 3 = 30
theorem part1_3kg : supermarket_A_cost 3 = 30 :=
by sorry

-- Proof that supermarket_A_cost 5 = 46
theorem part2_5kg : supermarket_A_cost 5 = 46 :=
by sorry

-- Proof that the cost function is correct
theorem part2_function (x : ℝ) : 
(0 < x ∧ x <= 4 → supermarket_A_cost x = 10 * x) ∧ 
(x > 4 → supermarket_A_cost x = 6 * x + 16) :=
by sorry

-- Proof that supermarket A is cheaper for 10 kg apples
theorem part3_compare : supermarket_A_cost 10 < supermarket_B_cost 10 :=
by sorry

end part1_3kg_part2_5kg_part2_function_part3_compare_l2133_213315


namespace midpoint_AB_l2133_213344

noncomputable def s (x t : ℝ) : ℝ := (x + t)^2 + (x - t)^2

noncomputable def CP (x : ℝ) : ℝ := x * Real.sqrt 3 / 2

theorem midpoint_AB (x : ℝ) (P : ℝ) : 
    (s x 0 = 2 * CP x ^ 2) ↔ P = x :=
by
    sorry

end midpoint_AB_l2133_213344


namespace number_of_students_like_basketball_but_not_table_tennis_l2133_213313

-- Given definitions
def total_students : Nat := 40
def students_like_basketball : Nat := 24
def students_like_table_tennis : Nat := 16
def students_dislike_both : Nat := 6

-- Proposition to prove
theorem number_of_students_like_basketball_but_not_table_tennis : 
  students_like_basketball - (students_like_basketball + students_like_table_tennis - (total_students - students_dislike_both)) = 18 := 
by
  sorry

end number_of_students_like_basketball_but_not_table_tennis_l2133_213313


namespace covering_percentage_77_l2133_213391

-- Definition section for conditions
def radius_of_circle (r a : ℝ) := 2 * r * Real.pi = 4 * a
def center_coincide (a b : ℝ) := a = b

-- Theorem to be proven
theorem covering_percentage_77
  (r a : ℝ)
  (h_radius: radius_of_circle r a)
  (h_center: center_coincide 0 0) : 
  (r^2 * Real.pi - 0.7248 * r^2) / (r^2 * Real.pi) * 100 = 77 := by
  sorry

end covering_percentage_77_l2133_213391


namespace point_p_final_position_l2133_213385

theorem point_p_final_position :
  let P_start := -2
  let P_right := P_start + 5
  let P_final := P_right - 4
  P_final = -1 :=
by
  sorry

end point_p_final_position_l2133_213385


namespace average_runs_in_30_matches_l2133_213350

theorem average_runs_in_30_matches (avg_runs_15: ℕ) (avg_runs_20: ℕ) 
    (matches_15: ℕ) (matches_20: ℕ)
    (h1: avg_runs_15 = 30) (h2: avg_runs_20 = 15)
    (h3: matches_15 = 15) (h4: matches_20 = 20) : 
    (matches_15 * avg_runs_15 + matches_20 * avg_runs_20) / (matches_15 + matches_20) = 25 := 
by 
  sorry

end average_runs_in_30_matches_l2133_213350


namespace smallest_right_triangle_area_l2133_213352

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l2133_213352


namespace remainder_problem_l2133_213367

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 2) (h2 : n = 197) : 197 % 16 = 5 := by
  sorry

end remainder_problem_l2133_213367


namespace sufficient_but_not_necessary_condition_l2133_213314

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (2 * x^2 + x - 1 ≥ 0) → (x ≥ 1/2) ∨ (x ≤ -1) :=
by
  -- The given inequality and condition imply this result.
  sorry

end sufficient_but_not_necessary_condition_l2133_213314


namespace range_of_quadratic_function_l2133_213386

theorem range_of_quadratic_function :
  ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), -x^2 - 4 * x + 1 ∈ Set.Icc (-11) (5) :=
by
  sorry

end range_of_quadratic_function_l2133_213386


namespace M_necessary_for_N_l2133_213311

def M (x : ℝ) : Prop := -1 < x ∧ x < 3
def N (x : ℝ) : Prop := 0 < x ∧ x < 3

theorem M_necessary_for_N : (∀ a : ℝ, N a → M a) ∧ (∃ b : ℝ, M b ∧ ¬N b) :=
by sorry

end M_necessary_for_N_l2133_213311


namespace maximum_value_l2133_213339

def expression (A B C : ℕ) : ℕ := A * B * C + A * B + B * C + C * A

theorem maximum_value (A B C : ℕ) 
  (h1 : A + B + C = 15) : 
  expression A B C ≤ 200 :=
sorry

end maximum_value_l2133_213339


namespace percentage_profits_to_revenues_l2133_213322

theorem percentage_profits_to_revenues (R P : ℝ) 
  (h1 : R > 0) 
  (h2 : P > 0)
  (h3 : 0.12 * R = 1.2 * P) 
  : P / R = 0.1 :=
by
  sorry

end percentage_profits_to_revenues_l2133_213322


namespace range_of_m_l2133_213376

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by
  sorry

end range_of_m_l2133_213376


namespace farmer_initial_productivity_l2133_213329

theorem farmer_initial_productivity (x : ℝ) (d : ℝ)
  (hx1 : d = 1440 / x)
  (hx2 : 2 * x + (d - 4) * 1.25 * x = 1440) :
  x = 120 :=
by
  sorry

end farmer_initial_productivity_l2133_213329


namespace solve_arcsin_sin_l2133_213327

theorem solve_arcsin_sin (x : ℝ) (h : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.arcsin (Real.sin (2 * x)) = x ↔ x = 0 ∨ x = Real.pi / 3 ∨ x = -Real.pi / 3 :=
by
  sorry

end solve_arcsin_sin_l2133_213327


namespace sunset_time_range_l2133_213312

theorem sunset_time_range (h : ℝ) :
  ¬(h ≥ 7) ∧ ¬(h ≤ 8) ∧ ¬(h ≤ 6) ↔ h ∈ Set.Ioi 8 :=
by
  sorry

end sunset_time_range_l2133_213312


namespace binomial_expansion_l2133_213328

theorem binomial_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (1 + 2 * 1)^5 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 ∧
  (1 + 2 * -1)^5 = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 → 
  a_0 + a_2 + a_4 = 121 :=
by
  intro h
  let h₁ := h.1
  let h₂ := h.2
  sorry

end binomial_expansion_l2133_213328


namespace fish_population_estimation_l2133_213360

theorem fish_population_estimation (N : ℕ) (h1 : 80 ≤ N)
  (h_tagged_returned : true)
  (h_second_catch : 80 ≤ N)
  (h_tagged_in_second_catch : 2 = 80 * 80 / N) :
  N = 3200 :=
by
  sorry

end fish_population_estimation_l2133_213360


namespace max_omega_l2133_213370

theorem max_omega (ω : ℕ) (T : ℝ) (h₁ : T = 2 * Real.pi / ω) (h₂ : 1 < T) (h₃ : T < 3) : ω = 6 :=
sorry

end max_omega_l2133_213370


namespace distance_between_lines_l2133_213384

/-- The graph of the function y = x^2 + ax + b is drawn on a board.
Let the parabola intersect the horizontal lines y = s and y = t at points A, B and C, D respectively,
with A B = 5 and C D = 11. Then the distance between the lines y = s and y = t is 24. -/
theorem distance_between_lines 
  (a b s t : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + a * x1 + b = s) ∧ (x2^2 + a * x2 + b = s) ∧ |x1 - x2| = 5)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ (x3^2 + a * x3 + b = t) ∧ (x4^2 + a * x4 + b = t) ∧ |x3 - x4| = 11) :
  |t - s| = 24 := 
by
  sorry

end distance_between_lines_l2133_213384


namespace paco_more_cookies_l2133_213383

def paco_cookies_difference
  (initial_cookies : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_given : ℕ) : ℕ :=
  cookies_eaten - cookies_given

theorem paco_more_cookies 
  (initial_cookies : ℕ)
  (cookies_eaten : ℕ)
  (cookies_given : ℕ)
  (h1 : initial_cookies = 17)
  (h2 : cookies_eaten = 14)
  (h3 : cookies_given = 13) :
  paco_cookies_difference initial_cookies cookies_eaten cookies_given = 1 :=
by
  rw [h2, h3]
  exact rfl

end paco_more_cookies_l2133_213383


namespace min_value_expression_l2133_213333

-- Let x and y be positive integers such that x^2 + y^2 - 2017 * x * y > 0 and it is not a perfect square.
theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_not_square : ¬ ∃ z : ℕ, (x^2 + y^2 - 2017 * x * y) = z^2) :
  x^2 + y^2 - 2017 * x * y > 0 → ∃ k : ℕ, k = 2019 ∧ ∀ m : ℕ, (m > 0 → ¬ ∃ z : ℤ, (x^2 + y^2 - 2017 * x * y) = z^2 ∧ x^2 + y^2 - 2017 * x * y < k) :=
sorry

end min_value_expression_l2133_213333


namespace coin_collection_l2133_213330

def initial_ratio (G S : ℕ) : Prop := G = S / 3
def new_ratio (G S : ℕ) (addedG : ℕ) : Prop := G + addedG = S / 2
def total_coins_after (G S addedG : ℕ) : ℕ := G + addedG + S

theorem coin_collection (G S : ℕ) (addedG : ℕ) 
  (h1 : initial_ratio G S) 
  (h2 : addedG = 15) 
  (h3 : new_ratio G S addedG) : 
  total_coins_after G S addedG = 135 := 
by {
  sorry
}

end coin_collection_l2133_213330


namespace butterfly_count_l2133_213387

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l2133_213387


namespace parallel_lines_a_eq_3_l2133_213309

theorem parallel_lines_a_eq_3
  (a : ℝ)
  (l1 : a^2 * x - y + a^2 - 3 * a = 0)
  (l2 : (4 * a - 3) * x - y - 2 = 0)
  (h : ∀ x y, a^2 * x - y + a^2 - 3 * a = (4 * a - 3) * x - y - 2) :
  a = 3 :=
by
  sorry

end parallel_lines_a_eq_3_l2133_213309


namespace g_is_odd_l2133_213307

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end g_is_odd_l2133_213307


namespace area_PTR_l2133_213346

-- Define points P, Q, R, S, and T
variables (P Q R S T : Type)

-- Assume QR is divided by points S and T in the given ratio
variables (QS ST TR : ℕ)
axiom ratio_condition : QS = 2 ∧ ST = 5 ∧ TR = 3

-- Assume the area of triangle PQS is given as 60 square centimeters
axiom area_PQS : ℕ
axiom area_PQS_value : area_PQS = 60

-- State the problem
theorem area_PTR : ∃ (area_PTR : ℕ), area_PTR = 90 :=
by
  sorry

end area_PTR_l2133_213346


namespace part1_part2_l2133_213364

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end part1_part2_l2133_213364


namespace find_b_from_conditions_l2133_213324

theorem find_b_from_conditions (x y z k : ℝ) (h1 : (x + y) / 2 = k) (h2 : (z + x) / 3 = k) (h3 : (y + z) / 4 = k) (h4 : x + y + z = 36) : x + y = 16 := 
by 
  sorry

end find_b_from_conditions_l2133_213324


namespace nth_equation_pattern_l2133_213319

theorem nth_equation_pattern (n : ℕ) (hn : 0 < n) : n^2 - n = n * (n - 1) := by
  sorry

end nth_equation_pattern_l2133_213319


namespace books_more_than_movies_l2133_213347

-- Define the number of movies and books in the "crazy silly school" series.
def num_movies : ℕ := 14
def num_books : ℕ := 15

-- State the theorem to prove there is 1 more book than movies.
theorem books_more_than_movies : num_books - num_movies = 1 :=
by 
  -- Proof is omitted.
  sorry

end books_more_than_movies_l2133_213347


namespace differential_solution_correct_l2133_213321

noncomputable def y (x : ℝ) : ℝ := (x + 1)^2

theorem differential_solution_correct : 
  (∀ x : ℝ, deriv (deriv y) x = 2) ∧ y 0 = 1 ∧ (deriv y 0) = 2 := 
by
  sorry

end differential_solution_correct_l2133_213321


namespace fraction_equality_l2133_213365

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end fraction_equality_l2133_213365


namespace binary_10101_to_decimal_l2133_213331

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ digit idx => digit * 2^idx) (List.range b.length) |>.sum

theorem binary_10101_to_decimal : binary_to_decimal [1, 0, 1, 0, 1] = 21 := by
  sorry

end binary_10101_to_decimal_l2133_213331
