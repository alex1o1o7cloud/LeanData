import Mathlib

namespace NUMINAMATH_GPT_selling_price_l1638_163867

theorem selling_price (CP P : ℝ) (hCP : CP = 320) (hP : P = 0.25) : CP + (P * CP) = 400 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_l1638_163867


namespace NUMINAMATH_GPT_surface_area_of_large_cube_correct_l1638_163894

-- Definition of the surface area problem

def edge_length_of_small_cube := 3 -- centimeters
def number_of_small_cubes := 27
def surface_area_of_large_cube (edge_length_of_small_cube : ℕ) (number_of_small_cubes : ℕ) : ℕ :=
  let edge_length_of_large_cube := edge_length_of_small_cube * (number_of_small_cubes^(1/3))
  6 * edge_length_of_large_cube^2

theorem surface_area_of_large_cube_correct :
  surface_area_of_large_cube edge_length_of_small_cube number_of_small_cubes = 486 := by
  sorry

end NUMINAMATH_GPT_surface_area_of_large_cube_correct_l1638_163894


namespace NUMINAMATH_GPT_cos_beta_acos_l1638_163815

theorem cos_beta_acos {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_cos_α : Real.cos α = 1 / 7) (h_cos_sum : Real.cos (α + β) = -11 / 14) :
  Real.cos β = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_beta_acos_l1638_163815


namespace NUMINAMATH_GPT_painter_remaining_time_l1638_163891

-- Define the initial conditions
def total_rooms : ℕ := 11
def hours_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Define the remaining rooms to paint
def remaining_rooms : ℕ := total_rooms - painted_rooms

-- Define the proof problem: the remaining time to paint the rest of the rooms
def remaining_hours : ℕ := remaining_rooms * hours_per_room

theorem painter_remaining_time :
  remaining_hours = 63 :=
sorry

end NUMINAMATH_GPT_painter_remaining_time_l1638_163891


namespace NUMINAMATH_GPT_nth_equation_holds_l1638_163883

theorem nth_equation_holds (n : ℕ) (h : 0 < n) :
  1 / (n + 2) + 2 / (n^2 + 2 * n) = 1 / n :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_holds_l1638_163883


namespace NUMINAMATH_GPT_k_value_l1638_163848

theorem k_value (k : ℝ) (x : ℝ) (y : ℝ) (hk : k^2 - 5 = -1) (hx : x > 0) (hy : y = (k - 1) * x^(k^2 - 5)) (h_dec : ∀ (x1 x2 : ℝ), x1 > 0 → x2 > x1 → (k - 1) * x2^(k^2 - 5) < (k - 1) * x1^(k^2 - 5)):
  k = 2 := by
  sorry

end NUMINAMATH_GPT_k_value_l1638_163848


namespace NUMINAMATH_GPT_inequality_proof_l1638_163804

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9 / 16 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1638_163804


namespace NUMINAMATH_GPT_reena_interest_paid_l1638_163814

-- Definitions based on conditions
def principal : ℝ := 1200
def rate : ℝ := 0.03
def time : ℝ := 3

-- Definition of simple interest calculation based on conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove that Reena paid $108 as interest
theorem reena_interest_paid : simple_interest principal rate time = 108 := by
  sorry

end NUMINAMATH_GPT_reena_interest_paid_l1638_163814


namespace NUMINAMATH_GPT_sum_of_edges_rectangular_solid_l1638_163843

theorem sum_of_edges_rectangular_solid
  (a r : ℝ)
  (hr : r ≠ 0)
  (volume_eq : (a / r) * a * (a * r) = 512)
  (surface_area_eq : 2 * ((a ^ 2) / r + a ^ 2 + (a ^ 2) * r) = 384)
  (geo_progression : true) : -- This is implicitly understood in the construction
  4 * ((a / r) + a + (a * r)) = 112 :=
by
  -- The proof will be placed here
  sorry

end NUMINAMATH_GPT_sum_of_edges_rectangular_solid_l1638_163843


namespace NUMINAMATH_GPT_license_plate_increase_factor_l1638_163844

def old_license_plates := 26^2 * 10^3
def new_license_plates := 26^3 * 10^4

theorem license_plate_increase_factor : (new_license_plates / old_license_plates) = 260 := by
  sorry

end NUMINAMATH_GPT_license_plate_increase_factor_l1638_163844


namespace NUMINAMATH_GPT_minimum_solutions_in_interval_l1638_163885

open Function Real

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define what it means for a function to be periodic
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- Main theorem statement
theorem minimum_solutions_in_interval :
  ∀ (f : ℝ → ℝ),
  is_even f → is_periodic f 3 → f 2 = 0 →
  (∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ x1 < 6 ∧ f x1 = 0 ∧
                     0 < x2 ∧ x2 < 6 ∧ f x2 = 0 ∧
                     0 < x3 ∧ x3 < 6 ∧ f x3 = 0 ∧
                     0 < x4 ∧ x4 < 6 ∧ f x4 = 0 ∧
                     x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :=
by
  sorry

end NUMINAMATH_GPT_minimum_solutions_in_interval_l1638_163885


namespace NUMINAMATH_GPT_smallest_fraction_denominator_l1638_163840

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end NUMINAMATH_GPT_smallest_fraction_denominator_l1638_163840


namespace NUMINAMATH_GPT_porter_monthly_earnings_l1638_163851

def daily_rate : ℕ := 8

def regular_days : ℕ := 5

def extra_day_rate : ℕ := daily_rate * 3 / 2  -- 50% increase on the daily rate

def weekly_earnings_with_overtime : ℕ := (daily_rate * regular_days) + extra_day_rate

def weeks_in_month : ℕ := 4

theorem porter_monthly_earnings : weekly_earnings_with_overtime * weeks_in_month = 208 :=
by
  sorry

end NUMINAMATH_GPT_porter_monthly_earnings_l1638_163851


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l1638_163825

structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def convertCylindricalToRectangular (c : CylindricalCoord) : RectangularCoord :=
  { x := c.r * Real.cos c.θ,
    y := c.r * Real.sin c.θ,
    z := c.z }

theorem cylindrical_to_rectangular :
  convertCylindricalToRectangular ⟨7, Real.pi / 3, -3⟩ = ⟨3.5, 7 * Real.sqrt 3 / 2, -3⟩ :=
by sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l1638_163825


namespace NUMINAMATH_GPT_cone_prism_ratio_l1638_163821

theorem cone_prism_ratio 
  (a b h_c h_p : ℝ) (hb_lt_a : b < a) : 
  (π * b * h_c) / (12 * a * h_p) = (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) :=
by
  sorry

end NUMINAMATH_GPT_cone_prism_ratio_l1638_163821


namespace NUMINAMATH_GPT_combined_mpg_l1638_163866

theorem combined_mpg (miles_alice : ℕ) (mpg_alice : ℕ) (miles_bob : ℕ) (mpg_bob : ℕ) :
  miles_alice = 120 ∧ mpg_alice = 30 ∧ miles_bob = 180 ∧ mpg_bob = 20 →
  (miles_alice + miles_bob) / ((miles_alice / mpg_alice) + (miles_bob / mpg_bob)) = 300 / 13 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_combined_mpg_l1638_163866


namespace NUMINAMATH_GPT_hyperbola_asymptotes_slope_l1638_163824

open Real

theorem hyperbola_asymptotes_slope (m : ℝ) : 
  (∀ x y : ℝ, (y ^ 2 / 16) - (x ^ 2 / 9) = 1 → (y = m * x ∨ y = -m * x)) → 
  m = 4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_slope_l1638_163824


namespace NUMINAMATH_GPT_coin_change_problem_l1638_163818

theorem coin_change_problem (d q h : ℕ) (n : ℕ) 
  (h1 : 2 * d + 5 * q + 10 * h = 240)
  (h2 : d ≥ 1)
  (h3 : q ≥ 1)
  (h4 : h ≥ 1) :
  n = 275 := 
sorry

end NUMINAMATH_GPT_coin_change_problem_l1638_163818


namespace NUMINAMATH_GPT_power_mod_result_l1638_163823

theorem power_mod_result :
  (47 ^ 1235 - 22 ^ 1235) % 8 = 7 := by
  sorry

end NUMINAMATH_GPT_power_mod_result_l1638_163823


namespace NUMINAMATH_GPT_determine_x_l1638_163808

theorem determine_x (x y : ℝ) (h : x / (x - 1) = (y^3 + 2 * y^2 - 1) / (y^3 + 2 * y^2 - 2)) : 
  x = y^3 + 2 * y^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l1638_163808


namespace NUMINAMATH_GPT_max_flowers_used_min_flowers_used_l1638_163875

-- Part (a) Setup
def max_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def max_T (C M : ℕ) := C + M

-- Given conditions
theorem max_flowers_used :
  (∀ C M : ℕ, max_flowers C M 36 → max_T C M = 36) :=
by sorry

-- Part (b) Setup
def min_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def min_T (C M : ℕ) := C + M

-- Given conditions
theorem min_flowers_used :
  (∀ C M : ℕ, min_flowers C M 48 → min_T C M = 24) :=
by sorry

end NUMINAMATH_GPT_max_flowers_used_min_flowers_used_l1638_163875


namespace NUMINAMATH_GPT_arithmetic_seq_sum_div_fifth_term_l1638_163833

open Int

/-- The sequence {a_n} is an arithmetic sequence with a non-zero common difference,
    given that a₂ + a₆ = a₈, prove that S₅ / a₅ = 3. -/
theorem arithmetic_seq_sum_div_fifth_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0)
  (h_condition : a 2 + a 6 = a 8) :
  ((5 * a 1 + 10 * d) / (a 1 + 4 * d) : ℚ) = 3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_div_fifth_term_l1638_163833


namespace NUMINAMATH_GPT_parities_of_E_10_11_12_l1638_163816

noncomputable def E : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| (n + 3) => 2 * (E (n + 2)) + (E n)

theorem parities_of_E_10_11_12 :
  (E 10 % 2 = 0) ∧ (E 11 % 2 = 1) ∧ (E 12 % 2 = 1) := 
  by
  sorry

end NUMINAMATH_GPT_parities_of_E_10_11_12_l1638_163816


namespace NUMINAMATH_GPT_sqrt_sub_sqrt_frac_eq_l1638_163838

theorem sqrt_sub_sqrt_frac_eq : (Real.sqrt 3) - (Real.sqrt (1 / 3)) = (2 * Real.sqrt 3) / 3 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sub_sqrt_frac_eq_l1638_163838


namespace NUMINAMATH_GPT_infinite_series_sum_l1638_163842

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) * (1 / 1000)^n = 3000000 / 998001 :=
by sorry

end NUMINAMATH_GPT_infinite_series_sum_l1638_163842


namespace NUMINAMATH_GPT_three_irreducible_fractions_prod_eq_one_l1638_163827

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end NUMINAMATH_GPT_three_irreducible_fractions_prod_eq_one_l1638_163827


namespace NUMINAMATH_GPT_geometric_seq_b6_l1638_163863

variable {b : ℕ → ℝ}

theorem geometric_seq_b6 (h1 : b 3 * b 9 = 9) (h2 : ∃ r, ∀ n, b (n + 1) = r * b n) : b 6 = 3 ∨ b 6 = -3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_b6_l1638_163863


namespace NUMINAMATH_GPT_triangle_inequality_range_isosceles_triangle_perimeter_l1638_163859

-- Define the parameters for the triangle
variables (AB BC AC a : ℝ)
variables (h_AB : AB = 8) (h_BC : BC = 2 * a + 2) (h_AC : AC = 22)

-- Define the lean proof problem for the given conditions
theorem triangle_inequality_range (h_triangle : AB = 8 ∧ BC = 2 * a + 2 ∧ AC = 22) :
  6 < a ∧ a < 14 := sorry

-- Define the isosceles condition and perimeter calculation
theorem isosceles_triangle_perimeter (h_isosceles : BC = AC) :
  perimeter = 52 := sorry

end NUMINAMATH_GPT_triangle_inequality_range_isosceles_triangle_perimeter_l1638_163859


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l1638_163841

def digit_representation (n m : ℕ) (a b : ℕ) :=
  n = 10 * a + b ∧
  m = 10 * b + a ∧
  n - m = 9 * (a * b) + 3

theorem sum_of_number_and_reverse :
  ∃ a b n m : ℕ, digit_representation n m a b ∧ n + m = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l1638_163841


namespace NUMINAMATH_GPT_zilla_savings_l1638_163853

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end NUMINAMATH_GPT_zilla_savings_l1638_163853


namespace NUMINAMATH_GPT_problem_statement_l1638_163846

variable (a b x : ℝ)

theorem problem_statement (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  a / (a - b) = x / (x - 1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1638_163846


namespace NUMINAMATH_GPT_jill_spending_on_clothing_l1638_163886

theorem jill_spending_on_clothing (C : ℝ) (T : ℝ)
  (h1 : 0.2 * T = 0.2 * T)
  (h2 : 0.3 * T = 0.3 * T)
  (h3 : (C / 100) * T * 0.04 + 0.3 * T * 0.08 = 0.044 * T) :
  C = 50 :=
by
  -- This line indicates the point where the proof would typically start
  sorry

end NUMINAMATH_GPT_jill_spending_on_clothing_l1638_163886


namespace NUMINAMATH_GPT_units_digit_product_odd_integers_10_to_110_l1638_163878

-- Define the set of odd integer numbers between 10 and 110
def oddNumbersInRange : List ℕ := List.filter (fun n => n % 2 = 1) (List.range' 10 101)

-- Define the set of relevant odd multiples of 5 within the range
def oddMultiplesOfFive : List ℕ := List.filter (fun n => n % 5 = 0) oddNumbersInRange

-- Prove that the product of all odd positive integers between 10 and 110 has units digit 5
theorem units_digit_product_odd_integers_10_to_110 :
  let product : ℕ := List.foldl (· * ·) 1 oddNumbersInRange
  product % 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_product_odd_integers_10_to_110_l1638_163878


namespace NUMINAMATH_GPT_six_digit_number_multiple_of_7_l1638_163898

theorem six_digit_number_multiple_of_7 (d : ℕ) (hd : d ≤ 9) :
  (∃ k : ℤ, 56782 + d * 10 = 7 * k) ↔ (d = 0 ∨ d = 7) := by
sorry

end NUMINAMATH_GPT_six_digit_number_multiple_of_7_l1638_163898


namespace NUMINAMATH_GPT_pipe_fills_tank_in_10_hours_l1638_163852

variables (pipe_rate leak_rate : ℝ)

-- Conditions
def combined_rate := pipe_rate - leak_rate
def leak_time := 30
def combined_time := 15

-- Express leak_rate from leak_time
noncomputable def leak_rate_def : ℝ := 1 / leak_time

-- Express pipe_rate from combined_time with leak_rate considered
noncomputable def pipe_rate_def : ℝ := 1 / combined_time + leak_rate_def

-- Theorem to be proved
theorem pipe_fills_tank_in_10_hours :
  (1 / pipe_rate_def) = 10 :=
by
  sorry

end NUMINAMATH_GPT_pipe_fills_tank_in_10_hours_l1638_163852


namespace NUMINAMATH_GPT_number_of_teams_l1638_163888

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_teams_l1638_163888


namespace NUMINAMATH_GPT_total_is_83_l1638_163813

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end NUMINAMATH_GPT_total_is_83_l1638_163813


namespace NUMINAMATH_GPT_find_x_l1638_163876

theorem find_x (p q r x : ℝ) (h1 : (p + q + r) / 3 = 4) (h2 : (p + q + r + x) / 4 = 5) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_x_l1638_163876


namespace NUMINAMATH_GPT_smaller_circle_radius_l1638_163800

theorem smaller_circle_radius (r R : ℝ) (hR : R = 10) (h : 2 * r = 2 * R) : r = 10 :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l1638_163800


namespace NUMINAMATH_GPT_rectangular_table_capacity_l1638_163877

variable (R : ℕ) -- The number of pupils a rectangular table can seat

-- Conditions
variable (rectangular_tables : ℕ)
variable (square_tables : ℕ)
variable (square_table_capacity : ℕ)
variable (total_pupils : ℕ)

-- Setting the values based on the conditions
axiom h1 : rectangular_tables = 7
axiom h2 : square_tables = 5
axiom h3 : square_table_capacity = 4
axiom h4 : total_pupils = 90

-- The proof statement
theorem rectangular_table_capacity :
  7 * R + 5 * 4 = 90 → R = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rectangular_table_capacity_l1638_163877


namespace NUMINAMATH_GPT_percentage_spent_on_hats_l1638_163801

def total_money : ℕ := 90
def cost_per_scarf : ℕ := 2
def number_of_scarves : ℕ := 18
def cost_of_scarves : ℕ := number_of_scarves * cost_per_scarf
def money_left_for_hats : ℕ := total_money - cost_of_scarves
def number_of_hats : ℕ := 2 * number_of_scarves

theorem percentage_spent_on_hats : 
  (money_left_for_hats : ℝ) / (total_money : ℝ) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_spent_on_hats_l1638_163801


namespace NUMINAMATH_GPT_min_value_x_plus_2y_l1638_163879

open Real

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_2y_l1638_163879


namespace NUMINAMATH_GPT_minimum_value_of_option_C_l1638_163890

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_option_C_l1638_163890


namespace NUMINAMATH_GPT_taco_truck_earnings_l1638_163873

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end NUMINAMATH_GPT_taco_truck_earnings_l1638_163873


namespace NUMINAMATH_GPT_area_of_paper_l1638_163836

theorem area_of_paper (L W : ℕ) (h1 : L + 2 * W = 34) (h2 : 2 * L + W = 38) : L * W = 140 := by
  sorry

end NUMINAMATH_GPT_area_of_paper_l1638_163836


namespace NUMINAMATH_GPT_solve_for_x_l1638_163889

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1638_163889


namespace NUMINAMATH_GPT_find_n_of_permut_comb_eq_l1638_163831

open Nat

theorem find_n_of_permut_comb_eq (n : Nat) (h : (n! / (n - 3)!) = 6 * (n! / (4! * (n - 4)!))) : n = 7 := by
  sorry

end NUMINAMATH_GPT_find_n_of_permut_comb_eq_l1638_163831


namespace NUMINAMATH_GPT_point_on_xOz_plane_l1638_163847

def point : ℝ × ℝ × ℝ := (1, 0, 4)

theorem point_on_xOz_plane : point.snd = 0 :=
by 
  -- Additional definitions and conditions might be necessary,
  -- but they should come directly from the problem statement:
  -- * Define conditions for being on the xOz plane.
  -- For the purpose of this example, we skip the proof.
  sorry

end NUMINAMATH_GPT_point_on_xOz_plane_l1638_163847


namespace NUMINAMATH_GPT_total_profit_is_35000_l1638_163828

-- Definitions based on the conditions
variables (IB TB : ℝ) -- IB: Investment of B, TB: Time period of B's investment
def IB_times_TB := IB * TB
def IA := 3 * IB
def TA := 2 * TB
def profit_share_B := IB_times_TB
def profit_share_A := 6 * IB_times_TB
variable (profit_B : ℝ)
def profit_B_val := 5000

-- Ensure these definitions are used
def total_profit := profit_share_A + profit_share_B

-- Lean 4 statement showing that the total profit is Rs 35000
theorem total_profit_is_35000 : total_profit = 35000 := by
  sorry

end NUMINAMATH_GPT_total_profit_is_35000_l1638_163828


namespace NUMINAMATH_GPT_sitting_break_frequency_l1638_163856

theorem sitting_break_frequency (x : ℕ) (h1 : 240 % x = 0) (h2 : 240 / 20 = 12) (h3 : 240 / x + 10 = 12) : x = 120 := 
sorry

end NUMINAMATH_GPT_sitting_break_frequency_l1638_163856


namespace NUMINAMATH_GPT_part_I_part_II_l1638_163865

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

theorem part_I (x : ℝ) : f x > 4 ↔ x < -1.5 ∨ x > 2.5 := 
sorry

theorem part_II (a : ℝ) : (∀ x, f x ≥ a) ↔ a ≤ 3 := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l1638_163865


namespace NUMINAMATH_GPT_johns_original_earnings_l1638_163849

-- Definitions from conditions
variables (x : ℝ) (raise_percentage : ℝ) (new_salary : ℝ)

-- Conditions
def conditions : Prop :=
  raise_percentage = 0.25 ∧ new_salary = 75 ∧ x + raise_percentage * x = new_salary

-- Theorem statement
theorem johns_original_earnings (h : conditions x 0.25 75) : x = 60 :=
sorry

end NUMINAMATH_GPT_johns_original_earnings_l1638_163849


namespace NUMINAMATH_GPT_emma_troy_wrapping_time_l1638_163860

theorem emma_troy_wrapping_time (emma_rate troy_rate total_task_time together_time emma_remaining_time : ℝ) 
  (h1 : emma_rate = 1 / 6) 
  (h2 : troy_rate = 1 / 8) 
  (h3 : total_task_time = 1) 
  (h4 : together_time = 2) 
  (h5 : emma_remaining_time = (total_task_time - (emma_rate + troy_rate) * together_time) / emma_rate) : 
  emma_remaining_time = 2.5 := 
sorry

end NUMINAMATH_GPT_emma_troy_wrapping_time_l1638_163860


namespace NUMINAMATH_GPT_polynomial_roots_ratio_l1638_163839

theorem polynomial_roots_ratio (a b c d : ℝ) (h₀ : a ≠ 0) 
    (h₁ : a * 64 + b * 16 + c * 4 + d = 0)
    (h₂ : -a + b - c + d = 0) : 
    (b + c) / a = -13 :=
by {
    sorry
}

end NUMINAMATH_GPT_polynomial_roots_ratio_l1638_163839


namespace NUMINAMATH_GPT_abc_inequality_l1638_163881

-- Required conditions and proof statement
theorem abc_inequality 
  {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 := 
sorry

end NUMINAMATH_GPT_abc_inequality_l1638_163881


namespace NUMINAMATH_GPT_trapezoidal_garden_solutions_l1638_163871

theorem trapezoidal_garden_solutions :
  ∃ (b1 b2 : ℕ), 
    (1800 = (60 * (b1 + b2)) / 2) ∧
    (b1 % 10 = 0) ∧ (b2 % 10 = 0) ∧
    (∃ (n : ℕ), n = 4) := 
sorry

end NUMINAMATH_GPT_trapezoidal_garden_solutions_l1638_163871


namespace NUMINAMATH_GPT_repeating_decimal_fraction_equiv_l1638_163872

noncomputable def repeating_decimal_to_fraction (x : ℚ) : Prop :=
  x = 0.4 + 37 / 990

theorem repeating_decimal_fraction_equiv : repeating_decimal_to_fraction (433 / 990) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_equiv_l1638_163872


namespace NUMINAMATH_GPT_correct_relation_l1638_163855

def A : Set ℝ := { x | x > 1 }

theorem correct_relation : 2 ∈ A := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_correct_relation_l1638_163855


namespace NUMINAMATH_GPT_blue_first_yellow_second_probability_l1638_163805

open Classical

-- Definition of initial conditions
def total_marbles : Nat := 3 + 4 + 9
def blue_marbles : Nat := 3
def yellow_marbles : Nat := 4
def pink_marbles : Nat := 9

-- Probability functions
def probability_first_blue : ℚ := blue_marbles / total_marbles
def probability_second_yellow_given_blue : ℚ := yellow_marbles / (total_marbles - 1)

-- Combined probability
def combined_probability_first_blue_second_yellow : ℚ := 
  probability_first_blue * probability_second_yellow_given_blue

-- Theorem statement
theorem blue_first_yellow_second_probability :
  combined_probability_first_blue_second_yellow = 1 / 20 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_blue_first_yellow_second_probability_l1638_163805


namespace NUMINAMATH_GPT_angle_bisectors_l1638_163817

open Real

noncomputable def r1 : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def r2 : ℝ × ℝ × ℝ := (0, 1, 1)

theorem angle_bisectors :
  ∃ (phi : ℝ), 0 ≤ phi ∧ phi ≤ π ∧ cos phi = 1 / 2 :=
sorry

end NUMINAMATH_GPT_angle_bisectors_l1638_163817


namespace NUMINAMATH_GPT_odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l1638_163837

theorem odd_solutions_eq_iff_a_le_neg3_or_a_ge3 (a : ℝ) :
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 5 ∧ (a - 3 * x^2 + Real.cos (9 * Real.pi * x / 2)) * Real.sqrt (3 - a * x) = 0) ↔ (a ≤ -3 ∨ a ≥ 3) := 
by
  sorry

end NUMINAMATH_GPT_odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l1638_163837


namespace NUMINAMATH_GPT_parametric_line_eq_l1638_163835

theorem parametric_line_eq (t : ℝ) :
  ∃ t : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
  (x = 3 * t + 5) ∧ (y = 6 * t - 7) → y = 2 * x - 17 :=
by
  sorry

end NUMINAMATH_GPT_parametric_line_eq_l1638_163835


namespace NUMINAMATH_GPT_rebecca_swimming_problem_l1638_163806

theorem rebecca_swimming_problem :
  ∃ D : ℕ, (D / 4 - D / 5) = 6 → D = 120 :=
sorry

end NUMINAMATH_GPT_rebecca_swimming_problem_l1638_163806


namespace NUMINAMATH_GPT_no_x_for_rational_sin_cos_l1638_163896

-- Define rational predicate
def is_rational (r : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ r = a / b

-- Define the statement of the problem
theorem no_x_for_rational_sin_cos :
  ∀ x : ℝ, ¬ (is_rational (Real.sin x + Real.sqrt 2) ∧ is_rational (Real.cos x - Real.sqrt 2)) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_no_x_for_rational_sin_cos_l1638_163896


namespace NUMINAMATH_GPT_ab_equals_five_l1638_163850

variable (a m b n : ℝ)

def arithmetic_seq (x y z : ℝ) : Prop :=
  2 * y = x + z

def geometric_seq (w x y z u : ℝ) : Prop :=
  x * x = w * y ∧ y * y = x * z ∧ z * z = y * u

theorem ab_equals_five
  (h1 : arithmetic_seq (-9) a (-1))
  (h2 : geometric_seq (-9) m b n (-1)) :
  a * b = 5 := sorry

end NUMINAMATH_GPT_ab_equals_five_l1638_163850


namespace NUMINAMATH_GPT_infinite_sum_equals_two_l1638_163809

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end NUMINAMATH_GPT_infinite_sum_equals_two_l1638_163809


namespace NUMINAMATH_GPT_consecutive_integer_quadratic_l1638_163830

theorem consecutive_integer_quadratic :
  ∃ (a b c : ℤ) (x₁ x₂ : ℤ),
  (a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0) ∧
  (a = 2 ∧ b = 0 ∧ c = -2) ∨ (a = -2 ∧ b = 0 ∧ c = 2) := sorry

end NUMINAMATH_GPT_consecutive_integer_quadratic_l1638_163830


namespace NUMINAMATH_GPT_monotonic_on_interval_l1638_163870

theorem monotonic_on_interval (k : ℝ) :
  (∀ x y : ℝ, x ≤ y → x ≤ 8 → y ≤ 8 → (4 * x ^ 2 - k * x - 8) ≤ (4 * y ^ 2 - k * y - 8)) ↔ (64 ≤ k) :=
sorry

end NUMINAMATH_GPT_monotonic_on_interval_l1638_163870


namespace NUMINAMATH_GPT_find_y_for_two_thirds_l1638_163897

theorem find_y_for_two_thirds (x y : ℝ) (h₁ : (2 / 3) * x + y = 10) (h₂ : x = 6) : y = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_y_for_two_thirds_l1638_163897


namespace NUMINAMATH_GPT_bus_seats_needed_l1638_163832

def members_playing_instruments : Prop :=
  let flute := 5
  let trumpet := 3 * flute
  let trombone := trumpet - 8
  let drum := trombone + 11
  let clarinet := 2 * flute
  let french_horn := trombone + 3
  let saxophone := (trumpet + trombone) / 2
  let piano := drum + 2
  let violin := french_horn - clarinet
  let guitar := 3 * flute
  let total_members := flute + trumpet + trombone + drum + clarinet + french_horn + saxophone + piano + violin + guitar
  total_members = 111

theorem bus_seats_needed : members_playing_instruments :=
by
  sorry

end NUMINAMATH_GPT_bus_seats_needed_l1638_163832


namespace NUMINAMATH_GPT_susie_total_savings_is_correct_l1638_163803

variable (initial_amount : ℝ) (year1_addition_pct : ℝ) (year2_addition_pct : ℝ) (interest_rate : ℝ)

def susies_savings (initial_amount year1_addition_pct year2_addition_pct interest_rate : ℝ) : ℝ :=
  let end_of_first_year := initial_amount + initial_amount * year1_addition_pct
  let first_year_interest := end_of_first_year * interest_rate
  let total_after_first_year := end_of_first_year + first_year_interest
  let end_of_second_year := total_after_first_year + total_after_first_year * year2_addition_pct
  let second_year_interest := end_of_second_year * interest_rate
  end_of_second_year + second_year_interest

theorem susie_total_savings_is_correct : 
  susies_savings 200 0.20 0.30 0.05 = 343.98 := 
by
  sorry

end NUMINAMATH_GPT_susie_total_savings_is_correct_l1638_163803


namespace NUMINAMATH_GPT_MrMartinSpent_l1638_163829

theorem MrMartinSpent : 
  ∀ (C B : ℝ), 
    3 * C + 2 * B = 12.75 → 
    B = 1.5 → 
    2 * C + 5 * B = 14 := 
by
  intros C B h1 h2
  sorry

end NUMINAMATH_GPT_MrMartinSpent_l1638_163829


namespace NUMINAMATH_GPT_manufacturing_department_degrees_l1638_163826

def percentage_of_circle (percentage : ℕ) (total_degrees : ℕ) : ℕ :=
  (percentage * total_degrees) / 100

theorem manufacturing_department_degrees :
  percentage_of_circle 30 360 = 108 :=
by
  sorry

end NUMINAMATH_GPT_manufacturing_department_degrees_l1638_163826


namespace NUMINAMATH_GPT_each_person_pays_50_97_l1638_163810

noncomputable def total_bill (original_bill : ℝ) (tip_percentage : ℝ) : ℝ :=
  original_bill + original_bill * tip_percentage

noncomputable def amount_per_person (total_bill : ℝ) (num_people : ℕ) : ℝ :=
  total_bill / num_people

theorem each_person_pays_50_97 :
  let original_bill := 139.00
  let number_of_people := 3
  let tip_percentage := 0.10
  let expected_amount := 50.97
  abs (amount_per_person (total_bill original_bill tip_percentage) number_of_people - expected_amount) < 0.01
:= sorry

end NUMINAMATH_GPT_each_person_pays_50_97_l1638_163810


namespace NUMINAMATH_GPT_xy_product_approx_25_l1638_163899

noncomputable def approx_eq (a b : ℝ) (ε : ℝ := 1e-6) : Prop :=
  |a - b| < ε

theorem xy_product_approx_25 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (hxy : x / y = 36) (hy : y = 0.8333333333333334) : approx_eq (x * y) 25 :=
by
  sorry

end NUMINAMATH_GPT_xy_product_approx_25_l1638_163899


namespace NUMINAMATH_GPT_adult_ticket_cost_l1638_163887

theorem adult_ticket_cost (C : ℝ) (h1 : ∀ (a : ℝ), a = C + 8)
  (h2 : ∀ (s : ℝ), s = C + 4)
  (h3 : 5 * C + 2 * (C + 8) + 2 * (C + 4) = 150) :
  ∃ (a : ℝ), a = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_adult_ticket_cost_l1638_163887


namespace NUMINAMATH_GPT_factorial_divisibility_l1638_163862

theorem factorial_divisibility {n : ℕ} (h : 2011^(2011) ∣ n!) : 2011^(2012) ∣ n! :=
sorry

end NUMINAMATH_GPT_factorial_divisibility_l1638_163862


namespace NUMINAMATH_GPT_least_x_l1638_163884

noncomputable def is_odd_prime (n : ℕ) : Prop :=
  n > 1 ∧ Prime n ∧ n % 2 = 1

theorem least_x (x p : ℕ) (hp : Prime p) (hx : x > 0) (hodd_prime : is_odd_prime (x / (12 * p))) : x = 72 := 
  sorry

end NUMINAMATH_GPT_least_x_l1638_163884


namespace NUMINAMATH_GPT_remainder_of_sum_of_integers_l1638_163882

theorem remainder_of_sum_of_integers (a b c : ℕ)
  (h₁ : a % 30 = 15) (h₂ : b % 30 = 5) (h₃ : c % 30 = 10) :
  (a + b + c) % 30 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_integers_l1638_163882


namespace NUMINAMATH_GPT_a_minus_b_is_15_l1638_163892

variables (a b c : ℝ)

-- Conditions from the problem statement
axiom cond1 : a = 1/3 * (b + c)
axiom cond2 : b = 2/7 * (a + c)
axiom cond3 : a + b + c = 540

-- The theorem we need to prove
theorem a_minus_b_is_15 : a - b = 15 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_is_15_l1638_163892


namespace NUMINAMATH_GPT_inequality_cube_of_greater_l1638_163861

variable {a b : ℝ}

theorem inequality_cube_of_greater (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_inequality_cube_of_greater_l1638_163861


namespace NUMINAMATH_GPT_Aaron_final_cards_l1638_163802

-- Definitions from conditions
def initial_cards_Aaron : Nat := 5
def found_cards_Aaron : Nat := 62

-- Theorem statement
theorem Aaron_final_cards : initial_cards_Aaron + found_cards_Aaron = 67 :=
by
  sorry

end NUMINAMATH_GPT_Aaron_final_cards_l1638_163802


namespace NUMINAMATH_GPT_distance_is_absolute_value_l1638_163834

noncomputable def distance_to_origin (x : ℝ) : ℝ := |x|

theorem distance_is_absolute_value (x : ℝ) : distance_to_origin x = |x| :=
by
  sorry

end NUMINAMATH_GPT_distance_is_absolute_value_l1638_163834


namespace NUMINAMATH_GPT_treShaun_marker_ink_left_l1638_163869

noncomputable def ink_left_percentage (marker_area : ℕ) (total_colored_area : ℕ) : ℕ :=
if total_colored_area >= marker_area then 0 else ((marker_area - total_colored_area) * 100) / marker_area

theorem treShaun_marker_ink_left :
  let marker_area := 3 * (4 * 4)
  let colored_area := (2 * (6 * 2) + 8 * 4)
  ink_left_percentage marker_area colored_area = 0 :=
by
  sorry

end NUMINAMATH_GPT_treShaun_marker_ink_left_l1638_163869


namespace NUMINAMATH_GPT_domain_of_c_l1638_163845

theorem domain_of_c (m : ℝ) :
  (∀ x : ℝ, 7*x^2 - 6*x + m ≠ 0) ↔ (m > (9 / 7)) :=
by
  -- you would typically put the proof here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_domain_of_c_l1638_163845


namespace NUMINAMATH_GPT_length_of_ship_l1638_163874

-- Variables and conditions
variables (E L S : ℝ)
variables (W : ℝ := 0.9) -- Wind reducing factor

-- Conditions as equations
def condition1 : Prop := 150 * E = L + 150 * S
def condition2 : Prop := 70 * E = L - 63 * S

-- Theorem to prove
theorem length_of_ship (hc1 : condition1 E L S) (hc2 : condition2 E L S) : L = (19950 / 213) * E :=
sorry

end NUMINAMATH_GPT_length_of_ship_l1638_163874


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_lines_l1638_163858

def line1 (x : ℝ) : ℝ := 5
def line2 (x : ℝ) : ℝ := 1 + x
def line3 (x : ℝ) : ℝ := 1 - x

theorem area_of_triangle_formed_by_lines :
  let A := (4, 5)
  let B := (-4, 5)
  let C := (0, 1)
  (1 / 2) * abs (4 * 5 + (-4) * 1 + 0 * 5 - (5 * (-4) + 1 * 4 + 5 * 0)) = 16 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_lines_l1638_163858


namespace NUMINAMATH_GPT_reflection_across_y_axis_coordinates_l1638_163857

def coordinates_after_reflection (x y : ℤ) : ℤ × ℤ :=
  (-x, y)

theorem reflection_across_y_axis_coordinates :
  coordinates_after_reflection (-3) 4 = (3, 4) :=
by
  sorry

end NUMINAMATH_GPT_reflection_across_y_axis_coordinates_l1638_163857


namespace NUMINAMATH_GPT_range_of_values_l1638_163864

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem range_of_values (x : ℝ) : f (2 * x) > f (x - 3) ↔ x < -3 ∨ x > 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_values_l1638_163864


namespace NUMINAMATH_GPT_invitations_sent_out_l1638_163893

-- Define the conditions
def RSVPed (I : ℝ) : ℝ := 0.9 * I
def Showed_up (I : ℝ) : ℝ := 0.8 * RSVPed I
def No_gift : ℝ := 10
def Thank_you_cards : ℝ := 134

-- Prove the number of invitations
theorem invitations_sent_out : ∃ I : ℝ, Showed_up I - No_gift = Thank_you_cards ∧ I = 200 :=
by
  sorry

end NUMINAMATH_GPT_invitations_sent_out_l1638_163893


namespace NUMINAMATH_GPT_fractions_lcm_l1638_163854

noncomputable def lcm_of_fractions_lcm (numerators : List ℕ) (denominators : List ℕ) : ℕ :=
  let lcm_nums := numerators.foldr Nat.lcm 1
  let gcd_denom := denominators.foldr Nat.gcd (denominators.headD 1)
  lcm_nums / gcd_denom

theorem fractions_lcm (hnum : List ℕ := [4, 5, 7, 9, 13, 16, 19])
                      (hdenom : List ℕ := [9, 7, 15, 13, 21, 35, 45]) :
  lcm_of_fractions_lcm hnum hdenom = 1244880 :=
by
  sorry

end NUMINAMATH_GPT_fractions_lcm_l1638_163854


namespace NUMINAMATH_GPT_rectangle_area_error_l1638_163895

/-
  Problem: 
  Given:
  1. One side of the rectangle is taken 20% in excess.
  2. The other side of the rectangle is taken 10% in deficit.
  Prove:
  The error percentage in the calculated area is 8%.
-/

noncomputable def error_percentage (L W : ℝ) := 
  let actual_area : ℝ := L * W
  let measured_length : ℝ := 1.20 * L
  let measured_width : ℝ := 0.90 * W
  let measured_area : ℝ := measured_length * measured_width
  ((measured_area - actual_area) / actual_area) * 100

theorem rectangle_area_error
  (L W : ℝ) : error_percentage L W = 8 := 
  sorry

end NUMINAMATH_GPT_rectangle_area_error_l1638_163895


namespace NUMINAMATH_GPT_triangle_is_right_l1638_163820

theorem triangle_is_right (A B C a b c : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
    (h₄ : A + B + C = π) (h_eq : a * (Real.cos C) + c * (Real.cos A) = b * (Real.sin B)) : B = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_right_l1638_163820


namespace NUMINAMATH_GPT_value_of_a_b_c_l1638_163811

theorem value_of_a_b_c (a b c : ℚ) (h₁ : |a| = 2) (h₂ : |b| = 2) (h₃ : |c| = 3) (h₄ : b < 0) (h₅ : 0 < a) :
  a + b + c = 3 ∨ a + b + c = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_b_c_l1638_163811


namespace NUMINAMATH_GPT_route_comparison_l1638_163807

noncomputable def t_X : ℝ := (8 / 40) * 60 -- time in minutes for Route X
noncomputable def t_Y1 : ℝ := (5.5 / 50) * 60 -- time in minutes for the normal speed segment of Route Y
noncomputable def t_Y2 : ℝ := (1 / 25) * 60 -- time in minutes for the construction zone segment of Route Y
noncomputable def t_Y3 : ℝ := (0.5 / 20) * 60 -- time in minutes for the park zone segment of Route Y
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3 -- total time in minutes for Route Y

theorem route_comparison : t_X - t_Y = 1.5 :=
by {
  -- Proof is skipped using sorry
  sorry
}

end NUMINAMATH_GPT_route_comparison_l1638_163807


namespace NUMINAMATH_GPT_express_vector_c_as_linear_combination_l1638_163822

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := (1, -1)
noncomputable def c : ℝ × ℝ := (2, 3)

theorem express_vector_c_as_linear_combination :
  ∃ x y : ℝ, c = (x * (1, 1).1 + y * (1, -1).1, x * (1, 1).2 + y * (1, -1).2) ∧
             x = 5 / 2 ∧ y = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_express_vector_c_as_linear_combination_l1638_163822


namespace NUMINAMATH_GPT_ratio_of_place_values_l1638_163880

-- Definitions based on conditions
def place_value_tens_digit : ℝ := 10
def place_value_hundredths_digit : ℝ := 0.01

-- Statement to prove
theorem ratio_of_place_values :
  (place_value_tens_digit / place_value_hundredths_digit) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_place_values_l1638_163880


namespace NUMINAMATH_GPT_Megan_not_lead_plays_l1638_163812

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end NUMINAMATH_GPT_Megan_not_lead_plays_l1638_163812


namespace NUMINAMATH_GPT_second_caterer_cheaper_l1638_163868

theorem second_caterer_cheaper (x : ℕ) :
  (150 + 18 * x > 250 + 14 * x) → x ≥ 26 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_caterer_cheaper_l1638_163868


namespace NUMINAMATH_GPT_find_a_l1638_163819

theorem find_a (a : ℝ) : (∃ x y : ℝ, 3 * x + a * y - 5 = 0 ∧ x = 1 ∧ y = 2) → a = 1 :=
by
  intro h
  match h with
  | ⟨x, y, hx, hx1, hy2⟩ => 
    have h1 : x = 1 := hx1
    have h2 : y = 2 := hy2
    rw [h1, h2] at hx
    sorry

end NUMINAMATH_GPT_find_a_l1638_163819
