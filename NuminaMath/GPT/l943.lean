import Mathlib

namespace least_positive_t_geometric_progression_l943_94369

noncomputable def least_positive_t( α : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) : ℝ :=
  9 - 4 * Real.sqrt 5

theorem least_positive_t_geometric_progression ( α t : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) :
  least_positive_t α h = t ↔
  ∃ r : ℝ, r > 0 ∧
    Real.arcsin (Real.sin α) = α ∧
    Real.arcsin (Real.sin (2 * α)) = 2 * α ∧
    Real.arcsin (Real.sin (7 * α)) = 7 * α ∧
    Real.arcsin (Real.sin (t * α)) = t * α ∧
    (α * r = 2 * α) ∧
    (2 * α * r = 7 * α ) ∧
    (7 * α * r = t * α) :=
sorry

end least_positive_t_geometric_progression_l943_94369


namespace find_x_plus_y_l943_94301

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end find_x_plus_y_l943_94301


namespace find_ages_l943_94359

theorem find_ages (P F M : ℕ) 
  (h1 : F - P = 31)
  (h2 : (F + 8) + (P + 8) = 69)
  (h3 : F - M = 4)
  (h4 : (P + 5) + (M + 5) = 65) :
  P = 11 ∧ F = 42 ∧ M = 38 :=
by
  sorry

end find_ages_l943_94359


namespace effective_annual_rate_of_interest_l943_94309

theorem effective_annual_rate_of_interest 
  (i : ℝ) (n : ℕ) (h_i : i = 0.10) (h_n : n = 2) : 
  (1 + i / n)^n - 1 = 0.1025 :=
by
  sorry

end effective_annual_rate_of_interest_l943_94309


namespace alex_correct_percentage_l943_94376

theorem alex_correct_percentage (y : ℝ) (hy_pos : y > 0) : 
  (5 / 7) * 100 = 71.43 := 
by
  sorry

end alex_correct_percentage_l943_94376


namespace girls_in_class_l943_94370

theorem girls_in_class (g b : ℕ) (h1 : g + b = 28) (h2 : g * 4 = b * 3) : g = 12 := by
  sorry

end girls_in_class_l943_94370


namespace find_a_for_inequality_l943_94306

theorem find_a_for_inequality (a : ℚ) :
  (∀ x : ℚ, (ax / (x - 1)) < 1 ↔ (x < 1 ∨ x > 2)) → a = 1/2 :=
by
  sorry

end find_a_for_inequality_l943_94306


namespace quadrilateral_ABCD_r_plus_s_l943_94344

noncomputable def AB_is (AB : Real) (r s : Nat) : Prop :=
  AB = r + Real.sqrt s

theorem quadrilateral_ABCD_r_plus_s :
  ∀ (BC CD AD : Real) (mA mB : ℕ) (r s : ℕ), 
  BC = 7 → 
  CD = 10 → 
  AD = 8 → 
  mA = 60 → 
  mB = 60 → 
  AB_is AB r s →
  r + s = 99 :=
by intros BC CD AD mA mB r s hBC hCD hAD hMA hMB hAB_is
   sorry

end quadrilateral_ABCD_r_plus_s_l943_94344


namespace drivers_schedule_l943_94346

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l943_94346


namespace ratio_of_wins_l943_94314

-- Definitions based on conditions
def W1 : ℕ := 15  -- Number of wins before first loss
def L : ℕ := 2    -- Total number of losses
def W2 : ℕ := 30 - W1  -- Calculate W2 based on W1 and total wins being 28 more than losses

-- Theorem statement: Prove the ratio of wins after her first loss to wins before her first loss is 1:1
theorem ratio_of_wins (h : W1 = 15 ∧ L = 2) : W2 / W1 = 1 := by
  sorry

end ratio_of_wins_l943_94314


namespace quadrilateral_area_is_48_l943_94381

structure Quadrilateral :=
  (PQ QR RS SP : ℝ)
  (angle_QRS angle_SPQ : ℝ)

def quadrilateral_example : Quadrilateral :=
{ PQ := 11, QR := 7, RS := 9, SP := 3, angle_QRS := 90, angle_SPQ := 90 }

noncomputable def area_of_quadrilateral (Q : Quadrilateral) : ℝ :=
  (1/2 * Q.PQ * Q.SP) + (1/2 * Q.QR * Q.RS)

theorem quadrilateral_area_is_48 (Q : Quadrilateral) (h1 : Q.PQ = 11) (h2 : Q.QR = 7) (h3 : Q.RS = 9) (h4 : Q.SP = 3) (h5 : Q.angle_QRS = 90) (h6 : Q.angle_SPQ = 90) :
  area_of_quadrilateral Q = 48 :=
by
  -- Here would be the proof
  sorry

end quadrilateral_area_is_48_l943_94381


namespace three_digit_integers_congruent_to_2_mod_4_l943_94329

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l943_94329


namespace weight_of_3_moles_of_CaI2_is_881_64_l943_94345

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
noncomputable def weight_3_moles_CaI2 : ℝ := 3 * molar_mass_CaI2

theorem weight_of_3_moles_of_CaI2_is_881_64 :
  weight_3_moles_CaI2 = 881.64 :=
by sorry

end weight_of_3_moles_of_CaI2_is_881_64_l943_94345


namespace range_of_values_for_a_l943_94363

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.sin x - (1 / 2) * Real.cos (2 * x) + a - (3 / a) + (1 / 2)

theorem range_of_values_for_a (a : ℝ) (ha : a ≠ 0) : 
  (∀ x : ℝ, f x a ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
by 
  let g (t : ℝ) : ℝ := t^2 + a * t + a - (3 / a)
  have h1 : g (-1) ≤ 0 := by sorry
  have h2 : g (1) ≤ 0 := by sorry
  sorry

end range_of_values_for_a_l943_94363


namespace function_decreases_iff_l943_94317

theorem function_decreases_iff (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (m - 3) * x1 + 4 > (m - 3) * x2 + 4) ↔ m < 3 :=
by
  sorry

end function_decreases_iff_l943_94317


namespace maximum_value_of_expression_l943_94332

noncomputable def maxValue (x y z : ℝ) : ℝ :=
(x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2)

theorem maximum_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  maxValue x y z ≤ 243 / 16 :=
sorry

end maximum_value_of_expression_l943_94332


namespace train_speed_is_126_kmh_l943_94392

noncomputable def train_speed_proof : Prop :=
  let length_meters := 560 / 1000           -- Convert length to kilometers
  let time_hours := 16 / 3600               -- Convert time to hours
  let speed := length_meters / time_hours   -- Calculate the speed
  speed = 126                               -- The speed should be 126 km/h

theorem train_speed_is_126_kmh : train_speed_proof := by 
  sorry

end train_speed_is_126_kmh_l943_94392


namespace quadratic_function_increasing_l943_94396

theorem quadratic_function_increasing (x : ℝ) : ((x - 1)^2 + 2 < (x + 1 - 1)^2 + 2) ↔ (x > 1) := by
  sorry

end quadratic_function_increasing_l943_94396


namespace A_completes_work_in_18_days_l943_94320

-- Define the conditions
def efficiency_A_twice_B (A B : ℕ → ℕ) : Prop := ∀ w, A w = 2 * B w
def same_work_time (A B C D : ℕ → ℕ) : Prop := 
  ∀ w t, A w + B w = C w + D w ∧ C t = 1 / 20 ∧ D t = 1 / 30

-- Define the key quantity to be proven
theorem A_completes_work_in_18_days (A B C D : ℕ → ℕ) 
  (h1 : efficiency_A_twice_B A B) 
  (h2 : same_work_time A B C D) : 
  ∀ w, A w = 1 / 18 :=
sorry

end A_completes_work_in_18_days_l943_94320


namespace vertical_asymptote_at_5_l943_94347

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 10) / (x - 5)

theorem vertical_asymptote_at_5 : ∃ a : ℝ, (a = 5) ∧ ∀ δ > 0, ∃ ε > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < ε → |f x| > δ :=
by
  sorry

end vertical_asymptote_at_5_l943_94347


namespace series_sum_equals_three_fourths_l943_94319

noncomputable def infinite_series_sum : ℝ :=
  (∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * (n + 1 + 1) * (n + 1 + 3)))

theorem series_sum_equals_three_fourths :
  infinite_series_sum = 3 / 4 :=
sorry

end series_sum_equals_three_fourths_l943_94319


namespace car_fuel_efficiency_l943_94399

theorem car_fuel_efficiency (distance gallons fuel_efficiency D : ℝ)
  (h₀ : fuel_efficiency = 40)
  (h₁ : gallons = 3.75)
  (h₂ : distance = 150)
  (h_eff : fuel_efficiency = distance / gallons) :
  fuel_efficiency = 40 ∧ (D / fuel_efficiency) = (D / 40) :=
by
  sorry

end car_fuel_efficiency_l943_94399


namespace part_I_part_II_l943_94378

def f (x a : ℝ) : ℝ := abs (3 * x + 2) - abs (2 * x + a)

theorem part_I (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = 4 / 3 :=
by
  sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≤ 0) ↔ (3 ≤ a ∨ a ≤ -7) :=
by
  sorry

end part_I_part_II_l943_94378


namespace difference_of_sums_1000_l943_94360

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_first_n_odd_not_divisible_by_5 (n : ℕ) : ℕ :=
  (n * n) - 5 * ((n / 5) * ((n / 5) + 1))

theorem difference_of_sums_1000 :
  (sum_first_n_even 1000) - (sum_first_n_odd_not_divisible_by_5 1000) = 51000 :=
by
  sorry

end difference_of_sums_1000_l943_94360


namespace num_outfits_l943_94349

-- Define the number of trousers, shirts, and jackets available
def num_trousers : Nat := 5
def num_shirts : Nat := 6
def num_jackets : Nat := 4

-- Define the main theorem
theorem num_outfits (t : Nat) (s : Nat) (j : Nat) (ht : t = num_trousers) (hs : s = num_shirts) (hj : j = num_jackets) :
  t * s * j = 120 :=
by 
  rw [ht, hs, hj]
  exact rfl

end num_outfits_l943_94349


namespace combined_mpg_l943_94335

def ray_mpg := 50
def tom_mpg := 20
def ray_miles := 100
def tom_miles := 200

theorem combined_mpg : 
  let ray_gallons := ray_miles / ray_mpg
  let tom_gallons := tom_miles / tom_mpg
  let total_gallons := ray_gallons + tom_gallons
  let total_miles := ray_miles + tom_miles
  total_miles / total_gallons = 25 :=
by
  sorry

end combined_mpg_l943_94335


namespace tim_bought_two_appetizers_l943_94343

-- Definitions of the conditions.
def total_spending : ℝ := 50
def portion_spent_on_entrees : ℝ := 0.80
def entree_cost : ℝ := total_spending * portion_spent_on_entrees
def appetizer_cost : ℝ := 5
def appetizer_spending : ℝ := total_spending - entree_cost

-- The statement to prove: that Tim bought 2 appetizers.
theorem tim_bought_two_appetizers :
  appetizer_spending / appetizer_cost = 2 := 
by
  sorry

end tim_bought_two_appetizers_l943_94343


namespace complex_division_example_l943_94384

-- Given conditions
def i : ℂ := Complex.I

-- The statement we need to prove
theorem complex_division_example : (1 + 3 * i) / (1 + i) = 2 + i :=
by
  sorry

end complex_division_example_l943_94384


namespace quadratic_equation_with_distinct_roots_l943_94313

theorem quadratic_equation_with_distinct_roots 
  (a p q b α : ℝ) 
  (hα1 : α ≠ 0) 
  (h_quad1 : α^2 + a * α + b = 0) 
  (h_quad2 : α^2 + p * α + q = 0) : 
  ∃ x : ℝ, x^2 - (b + q) * (a - p) / (q - b) * x + b * q * (a - p)^2 / (q - b)^2 = 0 :=
by
  sorry

end quadratic_equation_with_distinct_roots_l943_94313


namespace minimum_cost_is_correct_l943_94302

noncomputable def rectangular_area (length width : ℝ) : ℝ :=
  length * width

def flower_cost_per_sqft (flower : String) : ℝ :=
  match flower with
  | "Marigold" => 1.00
  | "Sunflower" => 1.75
  | "Tulip" => 1.25
  | "Orchid" => 2.75
  | "Iris" => 3.25
  | _ => 0.00

def min_garden_cost : ℝ :=
  let areas := [rectangular_area 5 2, rectangular_area 7 3, rectangular_area 5 5, rectangular_area 2 4, rectangular_area 5 4]
  let costs := [flower_cost_per_sqft "Orchid" * 8, 
                flower_cost_per_sqft "Iris" * 10, 
                flower_cost_per_sqft "Sunflower" * 20, 
                flower_cost_per_sqft "Tulip" * 21, 
                flower_cost_per_sqft "Marigold" * 25]
  costs.sum

theorem minimum_cost_is_correct :
  min_garden_cost = 140.75 :=
  by
    -- Proof omitted
    sorry

end minimum_cost_is_correct_l943_94302


namespace polynomial_integer_roots_a_value_l943_94364

open Polynomial

theorem polynomial_integer_roots_a_value (α β γ : ℤ) (a : ℤ) :
  (X - C α) * (X - C β) * (X - C γ) = X^3 - 2 * X^2 - 25 * X + C a →
  α + β + γ = 2 →
  α * β + α * γ + β * γ = -25 →
  a = -50 :=
by
  sorry

end polynomial_integer_roots_a_value_l943_94364


namespace evaluate_exponents_l943_94304

theorem evaluate_exponents :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := 
by
  sorry

end evaluate_exponents_l943_94304


namespace work_completion_days_l943_94398

theorem work_completion_days (A_time : ℝ) (A_efficiency : ℝ) (B_time : ℝ) (B_efficiency : ℝ) (C_time : ℝ) (C_efficiency : ℝ) :
  A_time = 60 → A_efficiency = 1.5 → B_time = 20 → B_efficiency = 1 → C_time = 30 → C_efficiency = 0.75 → 
  (1 / (A_efficiency / A_time + B_efficiency / B_time + C_efficiency / C_time)) = 10 := 
by
  intros A_time_eq A_efficiency_eq B_time_eq B_efficiency_eq C_time_eq C_efficiency_eq
  rw [A_time_eq, A_efficiency_eq, B_time_eq, B_efficiency_eq, C_time_eq, C_efficiency_eq]
  -- Proof omitted
  sorry

end work_completion_days_l943_94398


namespace find_value_of_a_l943_94327

-- Let a, b, and c be different numbers from {1, 2, 4}
def a_b_c_valid (a b c : ℕ) : Prop := 
  (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
  (a = 1 ∨ a = 2 ∨ a = 4) ∧ 
  (b = 1 ∨ b = 2 ∨ b = 4) ∧ 
  (c = 1 ∨ c = 2 ∨ c = 4)

-- The condition that (a / 2) / (b / c) equals 4 when evaluated
def expr_eq_four (a b c : ℕ) : Prop :=
  (a / 2 : ℚ) / (b / c : ℚ) = 4

-- Given the above conditions, prove that the value of 'a' is 4
theorem find_value_of_a (a b c : ℕ) (h_valid : a_b_c_valid a b c) (h_expr : expr_eq_four a b c) : a = 4 := 
  sorry

end find_value_of_a_l943_94327


namespace intersection_M_N_l943_94322

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := 3^x - 2

def M : Set ℝ := {x | f (g x) > 0}
def N : Set ℝ := {x | g x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | x < 1} :=
by sorry

end intersection_M_N_l943_94322


namespace solve_equation_l943_94380

theorem solve_equation :
  ∀ (x : ℝ), 
    x^3 + (Real.log 25 + Real.log 32 + Real.log 53) * x = (Real.log 23 + Real.log 35 + Real.log 52) * x^2 + 1 ↔ 
    x = Real.log 23 ∨ x = Real.log 35 ∨ x = Real.log 52 :=
by
  sorry

end solve_equation_l943_94380


namespace find_f_2011_l943_94324

def f: ℝ → ℝ :=
sorry

axiom f_periodicity (x : ℝ) : f (x + 3) = -f x
axiom f_initial_value : f 4 = -2

theorem find_f_2011 : f 2011 = 2 :=
by
  sorry

end find_f_2011_l943_94324


namespace minimum_value_of_expression_l943_94356

noncomputable def monotonic_function_property
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0): Prop :=
    (1 : ℝ) / a + 8 / b = 25

theorem minimum_value_of_expression 
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0) :
    (1 : ℝ) / a + 8 / b = 25 := 
sorry

end minimum_value_of_expression_l943_94356


namespace percentage_income_spent_on_clothes_l943_94397

-- Define the assumptions
def monthly_income : ℝ := 90000
def household_expenses : ℝ := 0.5 * monthly_income
def medicine_expenses : ℝ := 0.15 * monthly_income
def savings : ℝ := 9000

-- Define the proof statement
theorem percentage_income_spent_on_clothes :
  ∃ (clothes_expenses : ℝ),
    clothes_expenses = monthly_income - household_expenses - medicine_expenses - savings ∧
    (clothes_expenses / monthly_income) * 100 = 25 := 
sorry

end percentage_income_spent_on_clothes_l943_94397


namespace intersection_of_sets_example_l943_94316

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l943_94316


namespace time_differences_l943_94330

def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 8 -- minutes per mile
def lila_speed := 7 -- minutes per mile
def race_distance := 12 -- miles

noncomputable def malcolm_time := malcolm_speed * race_distance
noncomputable def joshua_time := joshua_speed * race_distance
noncomputable def lila_time := lila_speed * race_distance

theorem time_differences :
  joshua_time - malcolm_time = 24 ∧
  lila_time - malcolm_time = 12 :=
by
  sorry

end time_differences_l943_94330


namespace find_rate_percent_l943_94338

theorem find_rate_percent (SI P T : ℝ) (h : SI = (P * R * T) / 100) (H_SI : SI = 250) 
  (H_P : P = 1500) (H_T : T = 5) : R = 250 / 75 := by
  sorry

end find_rate_percent_l943_94338


namespace quotient_base5_l943_94315

theorem quotient_base5 (a b quotient : ℕ) 
  (ha : a = 2 * 5^3 + 4 * 5^2 + 3 * 5^1 + 1) 
  (hb : b = 2 * 5^1 + 3) 
  (hquotient : quotient = 1 * 5^2 + 0 * 5^1 + 3) :
  a / b = quotient :=
by sorry

end quotient_base5_l943_94315


namespace p_or_q_is_false_implies_p_and_q_is_false_l943_94361

theorem p_or_q_is_false_implies_p_and_q_is_false (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ((¬ (p ∧ q) → (p ∨ q ∨ ¬ (p ∨ q)))) := sorry

end p_or_q_is_false_implies_p_and_q_is_false_l943_94361


namespace locus_of_right_angle_vertex_l943_94310

variables {x y : ℝ}

/-- Given points M(-2,0) and N(2,0), if P(x,y) is the right-angled vertex of
  a right-angled triangle with MN as its hypotenuse, then the locus equation
  of P is given by x^2 + y^2 = 4 with the condition x ≠ ±2. -/
theorem locus_of_right_angle_vertex (h : x ≠ 2 ∧ x ≠ -2) :
  x^2 + y^2 = 4 :=
sorry

end locus_of_right_angle_vertex_l943_94310


namespace nitin_borrowed_amount_l943_94394

theorem nitin_borrowed_amount (P : ℝ) (I1 I2 I3 : ℝ) :
  (I1 = P * 0.06 * 3) ∧
  (I2 = P * 0.09 * 5) ∧
  (I3 = P * 0.13 * 3) ∧
  (I1 + I2 + I3 = 8160) →
  P = 8000 :=
by
  sorry

end nitin_borrowed_amount_l943_94394


namespace complex_quadrant_l943_94353

open Complex

-- Let complex number i be the imaginary unit
noncomputable def purely_imaginary (z : ℂ) : Prop := 
  z.re = 0

theorem complex_quadrant (z : ℂ) (a : ℂ) (hz : purely_imaginary z) (h : (2 + I) * z = 1 + a * I ^ 3) :
  (a + z).re > 0 ∧ (a + z).im < 0 :=
by 
  sorry

end complex_quadrant_l943_94353


namespace alice_bob_probability_l943_94387

noncomputable def probability_of_exactly_two_sunny_days : ℚ :=
  let p_sunny := 3 / 5
  let p_rain := 2 / 5
  3 * (p_sunny^2 * p_rain)

theorem alice_bob_probability :
  probability_of_exactly_two_sunny_days = 54 / 125 := 
sorry

end alice_bob_probability_l943_94387


namespace place_numbers_l943_94348

theorem place_numbers (a b c d : ℕ) (hab : Nat.gcd a b = 1) (hac : Nat.gcd a c = 1) 
  (had : Nat.gcd a d = 1) (hbc : Nat.gcd b c = 1) (hbd : Nat.gcd b d = 1) 
  (hcd : Nat.gcd c d = 1) :
  ∃ (bc ad ab cd abcd : ℕ), 
    bc = b * c ∧ ad = a * d ∧ ab = a * b ∧ cd = c * d ∧ abcd = a * b * c * d ∧
    Nat.gcd bc abcd > 1 ∧ Nat.gcd ad abcd > 1 ∧ Nat.gcd ab abcd > 1 ∧ 
    Nat.gcd cd abcd > 1 ∧
    Nat.gcd ab cd = 1 ∧ Nat.gcd ab ad = 1 ∧ Nat.gcd ab bc = 1 ∧ 
    Nat.gcd cd ad = 1 ∧ Nat.gcd cd bc = 1 ∧ Nat.gcd ad bc = 1 :=
by
  sorry

end place_numbers_l943_94348


namespace area_of_circle_given_circumference_l943_94371

theorem area_of_circle_given_circumference (C : ℝ) (hC : C = 18 * Real.pi) (k : ℝ) :
  ∃ r : ℝ, C = 2 * Real.pi * r ∧ k * Real.pi = Real.pi * r^2 → k = 81 :=
by
  sorry

end area_of_circle_given_circumference_l943_94371


namespace compute_product_l943_94368

-- Define the conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x^3 - y^3 = 35)

-- Define the theorem to be proved
theorem compute_product (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := 
sorry

end compute_product_l943_94368


namespace star_j_l943_94379

def star (x y : ℝ) : ℝ := x^3 - x * y

theorem star_j (j : ℝ) : star j (star j j) = 2 * j^3 - j^4 := 
by
  sorry

end star_j_l943_94379


namespace arithmetic_geometric_properties_l943_94351

noncomputable def arithmetic_seq (a₁ a₂ a₃ : ℝ) :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

noncomputable def geometric_seq (b₁ b₂ b₃ : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem arithmetic_geometric_properties (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  arithmetic_seq a₁ a₂ a₃ →
  geometric_seq b₁ b₂ b₃ →
  ¬(a₁ < a₂ ∧ a₂ > a₃) ∧
  (b₁ < b₂ ∧ b₂ > b₃) ∧
  (a₁ + a₂ < 0 → ¬(a₂ + a₃ < 0)) ∧
  (b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by
  sorry

end arithmetic_geometric_properties_l943_94351


namespace range_of_positive_integers_in_list_H_l943_94373

noncomputable def list_H_lower_bound : Int := -15
noncomputable def list_H_length : Nat := 30

theorem range_of_positive_integers_in_list_H :
  ∃(r : Nat), list_H_lower_bound + list_H_length - 1 = 14 ∧ r = 14 - 1 := 
by
  let upper_bound := list_H_lower_bound + Int.ofNat list_H_length - 1
  use (upper_bound - 1).toNat
  sorry

end range_of_positive_integers_in_list_H_l943_94373


namespace two_pow_n_plus_one_divisible_by_three_l943_94303

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end two_pow_n_plus_one_divisible_by_three_l943_94303


namespace identify_correct_statement_l943_94395

-- Definitions based on conditions
def population (athletes : ℕ) : Prop := athletes = 1000
def is_individual (athlete : ℕ) : Prop := athlete ≤ 1000
def is_sample (sampled_athletes : ℕ) (sample_size : ℕ) : Prop := sampled_athletes = 100 ∧ sample_size = 100

-- Theorem statement based on the conclusion
theorem identify_correct_statement (athletes : ℕ) (sampled_athletes : ℕ) (sample_size : ℕ)
    (h1 : population athletes) (h2 : ∀ a, is_individual a) (h3 : is_sample sampled_athletes sample_size) : 
    (sampled_athletes = 100) ∧ (sample_size = 100) :=
by
  sorry

end identify_correct_statement_l943_94395


namespace cubes_sum_to_91_l943_94328

theorem cubes_sum_to_91
  (a b : ℤ)
  (h : a^3 + b^3 = 91) : a * b = 12 :=
sorry

end cubes_sum_to_91_l943_94328


namespace minimum_value_of_expression_l943_94354

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x * y / z + z * x / y + y * z / x) * (x / (y * z) + y / (z * x) + z / (x * y))

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  expression x y z ≥ 9 :=
sorry

end minimum_value_of_expression_l943_94354


namespace distance_between_B_and_D_l943_94318

theorem distance_between_B_and_D (a b c d : ℝ) (h1 : |2 * a - 3 * c| = 1) (h2 : |2 * b - 3 * c| = 1) (h3 : |(2/3) * (d - a)| = 1) (h4 : a ≠ b) :
  |d - b| = 0.5 ∨ |d - b| = 2.5 :=
by
  sorry

end distance_between_B_and_D_l943_94318


namespace volume_of_cylinder_l943_94390

theorem volume_of_cylinder (r h : ℝ) (hr : r = 1) (hh : h = 2) (A : r * h = 4) : (π * r^2 * h = 2 * π) :=
by
  sorry

end volume_of_cylinder_l943_94390


namespace train_crossing_platform_l943_94325

/-- Given a train crosses a 100 m platform in 15 seconds, and the length of the train is 350 m,
    prove that the train takes 20 seconds to cross a second platform of length 250 m. -/
theorem train_crossing_platform (dist1 dist2 l_t t1 t2 : ℝ) (h1 : dist1 = 100) (h2 : dist2 = 250) (h3 : l_t = 350) (h4 : t1 = 15) :
  t2 = 20 :=
sorry

end train_crossing_platform_l943_94325


namespace sum_increased_consecutive_integers_product_990_l943_94341

theorem sum_increased_consecutive_integers_product_990 
  (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 990) :
  (a + 2) + (b + 2) + (c + 2) = 36 :=
sorry

end sum_increased_consecutive_integers_product_990_l943_94341


namespace race_distance_l943_94326

theorem race_distance (d x y z : ℝ) 
  (h1: d / x = (d - 25) / y)
  (h2: d / y = (d - 15) / z)
  (h3: d / x = (d - 35) / z) :
  d = 75 :=
sorry

end race_distance_l943_94326


namespace three_units_away_from_neg_one_l943_94365

def is_three_units_away (x : ℝ) (y : ℝ) : Prop := abs (x - y) = 3

theorem three_units_away_from_neg_one :
  { x : ℝ | is_three_units_away x (-1) } = {2, -4} := 
by
  sorry

end three_units_away_from_neg_one_l943_94365


namespace two_om_2om5_l943_94377

def om (a b : ℕ) : ℕ := a^b - b^a

theorem two_om_2om5 : om 2 (om 2 5) = 79 := by
  sorry

end two_om_2om5_l943_94377


namespace intersection_A_B_l943_94366

-- Define the sets A and B
def set_A : Set ℝ := { x | x^2 ≤ 1 }
def set_B : Set ℝ := { -2, -1, 0, 1, 2 }

-- The goal is to prove that the intersection of A and B is {-1, 0, 1}
theorem intersection_A_B : set_A ∩ set_B = ({-1, 0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l943_94366


namespace system_solution_conditions_l943_94323

theorem system_solution_conditions (α1 α2 α3 α4 : ℝ) :
  (α1 = α4 ∨ α2 = α3) ↔ 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 + x2 = α1 * α2 ∧
    x1 + x3 = α1 * α3 ∧
    x1 + x4 = α1 * α4 ∧
    x2 + x3 = α2 * α3 ∧
    x2 + x4 = α2 * α4 ∧
    x3 + x4 = α3 * α4 ∧
    x1 = x2 ∧
    x2 = x3 ∧
    x1 = α2^2 / 2 ∧
    x3 = α2^2 / 2 ∧
    x4 = α2 * α4 - (α2^2 / 2) ) :=
by sorry

end system_solution_conditions_l943_94323


namespace sum_of_digits_2_1989_and_5_1989_l943_94300

theorem sum_of_digits_2_1989_and_5_1989 
  (m n : ℕ) 
  (h1 : 10^(m-1) < 2^1989 ∧ 2^1989 < 10^m) 
  (h2 : 10^(n-1) < 5^1989 ∧ 5^1989 < 10^n) 
  (h3 : 2^1989 * 5^1989 = 10^1989) : 
  m + n = 1990 := 
sorry

end sum_of_digits_2_1989_and_5_1989_l943_94300


namespace product_range_l943_94340

theorem product_range (m b : ℚ) (h₀ : m = 3 / 4) (h₁ : b = 6 / 5) : 0 < m * b ∧ m * b < 1 :=
by
  sorry

end product_range_l943_94340


namespace negation_universal_proposition_l943_94350

theorem negation_universal_proposition : 
  (¬ ∀ x : ℝ, x^2 - x < 0) = ∃ x : ℝ, x^2 - x ≥ 0 :=
by
  sorry

end negation_universal_proposition_l943_94350


namespace nina_widgets_purchase_l943_94393

theorem nina_widgets_purchase (P : ℝ) (h1 : 8 * (P - 1) = 24) (h2 : 24 / P = 6) : true :=
by
  sorry

end nina_widgets_purchase_l943_94393


namespace Meena_cookies_left_l943_94337

def cookies_initial := 5 * 12
def cookies_sold_to_teacher := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

def cookies_left := cookies_initial - cookies_sold_to_teacher - cookies_bought_by_brock - cookies_bought_by_katy

theorem Meena_cookies_left : cookies_left = 15 := 
by 
  -- steps to be proven here
  sorry

end Meena_cookies_left_l943_94337


namespace problem_solution_l943_94386

theorem problem_solution : 
  (∃ (N : ℕ), (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N) → ∃ (N : ℕ), N = 5967 :=
by
  intro h
  sorry

end problem_solution_l943_94386


namespace find_x_l943_94362

noncomputable def x : ℝ := 80 / 9

theorem find_x
  (hx_pos : 0 < x)
  (hx_condition : x * (⌊x⌋₊ : ℝ) = 80) :
  x = 80 / 9 :=
by
  sorry

end find_x_l943_94362


namespace multiple_of_A_share_l943_94372

theorem multiple_of_A_share (a b c : ℤ) (hC : c = 84) (hSum : a + b + c = 427)
  (hEquality1 : ∃ x : ℤ, x * a = 4 * b) (hEquality2 : 7 * c = 4 * b) : ∃ x : ℤ, x = 3 :=
by {
  sorry
}

end multiple_of_A_share_l943_94372


namespace distinct_solutions_diff_l943_94389

theorem distinct_solutions_diff (r s : ℝ) 
  (h1 : r ≠ s) 
  (h2 : (5*r - 15)/(r^2 + 3*r - 18) = r + 3) 
  (h3 : (5*s - 15)/(s^2 + 3*s - 18) = s + 3) 
  (h4 : r > s) : 
  r - s = 13 :=
sorry

end distinct_solutions_diff_l943_94389


namespace geometric_sequence_problem_l943_94383

variable {a : ℕ → ℝ}

theorem geometric_sequence_problem (h1 : a 5 * a 7 = 2) (h2 : a 2 + a 10 = 3) : 
  (a 12 / a 4 = 1 / 2) ∨ (a 12 / a 4 = 2) := 
sorry

end geometric_sequence_problem_l943_94383


namespace peaches_thrown_away_l943_94388

variables (total_peaches fresh_percentage peaches_left : ℕ) (thrown_away : ℕ)
variables (h1 : total_peaches = 250) (h2 : fresh_percentage = 60) (h3 : peaches_left = 135)

theorem peaches_thrown_away :
  thrown_away = (total_peaches * (fresh_percentage / 100)) - peaches_left :=
sorry

end peaches_thrown_away_l943_94388


namespace unique_sum_of_three_distinct_positive_perfect_squares_l943_94334

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_positive_perfect_squares_that_sum_to (a b c sum : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a < b ∧ b < c ∧ a + b + c = sum

theorem unique_sum_of_three_distinct_positive_perfect_squares :
  (∃ a b c : ℕ, distinct_positive_perfect_squares_that_sum_to a b c 100) ∧
  (∀ a1 b1 c1 a2 b2 c2 : ℕ,
    distinct_positive_perfect_squares_that_sum_to a1 b1 c1 100 ∧
    distinct_positive_perfect_squares_that_sum_to a2 b2 c2 100 →
    (a1 = a2 ∧ b1 = b2 ∧ c1 = c2)) :=
by
  sorry

end unique_sum_of_three_distinct_positive_perfect_squares_l943_94334


namespace find_x_given_conditions_l943_94308

variable (x y z : ℝ)

theorem find_x_given_conditions
  (h1: x * y / (x + y) = 4)
  (h2: x * z / (x + z) = 9)
  (h3: y * z / (y + z) = 16)
  (h_pos: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_distinct: x ≠ y ∧ x ≠ z ∧ y ≠ z) :
  x = 384/21 :=
sorry

end find_x_given_conditions_l943_94308


namespace range_of_m_l943_94374

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x, (x^2 + 1) * (x^2 - 8 * x - 20) ≤ 0 → (x^2 - 2 * x + (1 - m^2)) ≤ 0) →
  m ≥ 9 := by
  sorry

end range_of_m_l943_94374


namespace solution_in_quadrants_I_and_II_l943_94367

theorem solution_in_quadrants_I_and_II (x y : ℝ) :
  (y > 3 * x) ∧ (y > 6 - 2 * x) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
by
  sorry

end solution_in_quadrants_I_and_II_l943_94367


namespace necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l943_94331

variable (x y : ℝ)

theorem necessary_but_not_sufficient (hx : x < y ∧ y < 0) : x^2 > y^2 :=
sorry

theorem not_sufficient (hx : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
sorry

-- Optional: Combining the two to create a combined theorem statement
theorem x2_gt_y2_iff_x_lt_y_lt_0 : (∀ x y : ℝ, x < y ∧ y < 0 → x^2 > y^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬ (x < y ∧ y < 0)) :=
sorry

end necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l943_94331


namespace extra_amount_spent_on_shoes_l943_94391

theorem extra_amount_spent_on_shoes (total_cost shirt_cost shoes_cost: ℝ) 
  (h1: total_cost = 300) (h2: shirt_cost = 97) 
  (h3: shoes_cost > 2 * shirt_cost)
  (h4: shirt_cost + shoes_cost = total_cost): 
  shoes_cost - 2 * shirt_cost = 9 :=
by
  sorry

end extra_amount_spent_on_shoes_l943_94391


namespace gamesNextMonth_l943_94355

def gamesThisMonth : ℕ := 11
def gamesLastMonth : ℕ := 17
def totalPlannedGames : ℕ := 44

theorem gamesNextMonth :
  (totalPlannedGames - (gamesThisMonth + gamesLastMonth) = 16) :=
by
  unfold totalPlannedGames
  unfold gamesThisMonth
  unfold gamesLastMonth
  sorry

end gamesNextMonth_l943_94355


namespace B_C_work_days_l943_94312

noncomputable def days_for_B_and_C {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) : ℝ :=
  30 / 7

theorem B_C_work_days {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) :
  days_for_B_and_C hA hA_B hA_B_C = 30 / 7 :=
sorry

end B_C_work_days_l943_94312


namespace ways_to_place_letters_l943_94357

-- defining the conditions of the problem
def num_letters : Nat := 4
def num_mailboxes : Nat := 3

-- the theorem we need to prove
theorem ways_to_place_letters : 
  (num_mailboxes ^ num_letters) = 81 := 
by 
  sorry

end ways_to_place_letters_l943_94357


namespace subtraction_digits_l943_94358

theorem subtraction_digits (a b c : ℕ) (h1 : c - a = 2) (h2 : b = c - 1) (h3 : 100 * a + 10 * b + c - (100 * c + 10 * b + a) = 802) :
a = 0 ∧ b = 1 ∧ c = 2 :=
by {
  -- The detailed proof steps will go here
  sorry
}

end subtraction_digits_l943_94358


namespace inequality_transformation_l943_94333

variable {a b : ℝ}

theorem inequality_transformation (h : a < b) : -a / 3 > -b / 3 :=
  sorry

end inequality_transformation_l943_94333


namespace smallest_possible_number_of_students_l943_94336

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l943_94336


namespace m_intersects_at_least_one_of_a_or_b_l943_94307

-- Definitions based on given conditions
variables {Plane : Type} {Line : Type} (α β : Plane) (a b m : Line)

-- Assume necessary conditions
axiom skew_lines (a b : Line) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom plane_intersection_is_line (p1 p2 : Plane) : Line
axiom intersects (l1 l2 : Line) : Prop

-- Given conditions
variables
  (h1 : skew_lines a b)               -- a and b are skew lines
  (h2 : line_in_plane a α)            -- a is contained in plane α
  (h3 : line_in_plane b β)            -- b is contained in plane β
  (h4 : plane_intersection_is_line α β = m)  -- α ∩ β = m

-- The theorem to prove the correct answer
theorem m_intersects_at_least_one_of_a_or_b :
  intersects m a ∨ intersects m b :=
sorry -- proof to be provided

end m_intersects_at_least_one_of_a_or_b_l943_94307


namespace linear_system_sum_l943_94305

theorem linear_system_sum (x y : ℝ) 
  (h1: x - y = 2) 
  (h2: y = 2): 
  x + y = 6 := 
sorry

end linear_system_sum_l943_94305


namespace exists_small_area_triangle_l943_94375

structure Point :=
(x : ℝ)
(y : ℝ)

def is_valid_point (p : Point) : Prop :=
(|p.x| ≤ 2) ∧ (|p.y| ≤ 2)

def no_three_collinear (points : List Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points →
  (p1 ≠ p2) → (p1 ≠ p3) → (p2 ≠ p3) →
  ((p1.y - p2.y) * (p1.x - p3.x) ≠ (p1.y - p3.y) * (p1.x - p2.x))

noncomputable def triangle_area (p1 p2 p3: Point) : ℝ :=
(abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) / 2

theorem exists_small_area_triangle (points : List Point)
  (h_valid : ∀ p ∈ points, is_valid_point p)
  (h_no_collinear : no_three_collinear points)
  (h_len : points.length = 6) :
  ∃ (p1 p2 p3: Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
  triangle_area p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l943_94375


namespace correct_proposition_l943_94385

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 + x else 1 - x

def prop_A := ∀ x : ℝ, f (Real.sin x) = -f (Real.sin (-x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_B := ∀ x : ℝ, f (Real.sin x) = f (Real.sin (-x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_C := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))
def prop_D := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))

theorem correct_proposition :
  (¬ prop_A ∧ ¬ prop_B ∧ prop_C ∧ ¬ prop_D) :=
sorry

end correct_proposition_l943_94385


namespace subtraction_division_l943_94339

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end subtraction_division_l943_94339


namespace fencing_problem_l943_94352

noncomputable def fencingRequired (L A W F : ℝ) := (A = L * W) → (F = 2 * W + L)

theorem fencing_problem :
  fencingRequired 25 880 35.2 95.4 :=
by
  sorry

end fencing_problem_l943_94352


namespace exists_univariate_polynomial_l943_94321

def polynomial_in_three_vars (P : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ,
  P x y z = P x y (x * y - z) ∧
  P x y z = P x (z * x - y) z ∧
  P x y z = P (y * z - x) y z

theorem exists_univariate_polynomial (P : ℝ → ℝ → ℝ → ℝ) (h : polynomial_in_three_vars P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x * y * z) :=
sorry

end exists_univariate_polynomial_l943_94321


namespace find_x_from_exponential_eq_l943_94311

theorem find_x_from_exponential_eq (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 6561) : x = 6 := 
sorry

end find_x_from_exponential_eq_l943_94311


namespace inequality_holds_for_m_l943_94382

theorem inequality_holds_for_m (n : ℕ) (m : ℕ) :
  (∀ a b : ℝ, (0 < a ∧ 0 < b) ∧ (a + b = 2) → (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by
  sorry

end inequality_holds_for_m_l943_94382


namespace radius_of_base_circle_of_cone_l943_94342

theorem radius_of_base_circle_of_cone (θ : ℝ) (r_sector : ℝ) (L : ℝ) (C : ℝ) (r_base : ℝ) :
  θ = 120 ∧ r_sector = 6 ∧ L = (θ / 360) * 2 * Real.pi * r_sector ∧ C = L ∧ C = 2 * Real.pi * r_base → r_base = 2 := by
  sorry

end radius_of_base_circle_of_cone_l943_94342
