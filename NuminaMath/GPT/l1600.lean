import Mathlib

namespace NUMINAMATH_GPT_a7_value_l1600_160071

theorem a7_value
  (a : ℕ → ℝ)
  (hx2 : ∀ n, n > 0 → a n ≠ 0)
  (slope_condition : ∀ n, n ≥ 2 → 2 * a n = 2 * a (n - 1) + 1)
  (point_condition : a 1 * 4 = 8) :
  a 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_a7_value_l1600_160071


namespace NUMINAMATH_GPT_symmetric_point_y_axis_l1600_160090

-- Define the original point P
def P : ℝ × ℝ := (1, 6)

-- Define the reflection across the y-axis
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.fst, point.snd)

-- Define the symmetric point with respect to the y-axis
def symmetric_point := reflect_y_axis P

-- Statement to prove
theorem symmetric_point_y_axis : symmetric_point = (-1, 6) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_symmetric_point_y_axis_l1600_160090


namespace NUMINAMATH_GPT_max_a2_plus_b2_l1600_160064

theorem max_a2_plus_b2 (a b : ℝ) 
  (h : abs (a - 1) + abs (a - 6) + abs (b + 3) + abs (b - 2) = 10) : 
  (a^2 + b^2) ≤ 45 :=
sorry

end NUMINAMATH_GPT_max_a2_plus_b2_l1600_160064


namespace NUMINAMATH_GPT_fraction_subtraction_simplify_l1600_160003

theorem fraction_subtraction_simplify :
  (9 / 19 - 3 / 57 - 1 / 3) = 5 / 57 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_simplify_l1600_160003


namespace NUMINAMATH_GPT_expand_product_l1600_160013

noncomputable def a (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
noncomputable def b (x : ℝ) : ℝ := x^2 + x + 3

theorem expand_product (x : ℝ) : (a x) * (b x) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1600_160013


namespace NUMINAMATH_GPT_penny_paid_amount_l1600_160087

-- Definitions based on conditions
def bulk_price : ℕ := 5
def minimum_spend : ℕ := 40
def tax_rate : ℕ := 1
def excess_pounds : ℕ := 32

-- Expression for total calculated cost
def total_pounds := (minimum_spend / bulk_price) + excess_pounds
def cost_before_tax := total_pounds * bulk_price
def total_tax := total_pounds * tax_rate
def total_cost := cost_before_tax + total_tax

-- Required proof statement
theorem penny_paid_amount : total_cost = 240 := 
by 
  sorry

end NUMINAMATH_GPT_penny_paid_amount_l1600_160087


namespace NUMINAMATH_GPT_last_four_digits_of_3_power_24000_l1600_160066

theorem last_four_digits_of_3_power_24000 (h : 3^800 ≡ 1 [MOD 2000]) : 3^24000 ≡ 1 [MOD 2000] :=
  by sorry

end NUMINAMATH_GPT_last_four_digits_of_3_power_24000_l1600_160066


namespace NUMINAMATH_GPT_lagrange_intermediate_value_l1600_160031

open Set

variable {a b : ℝ} (f : ℝ → ℝ)

-- Ensure that a < b for the interval [a, b]
axiom hab : a < b

-- Assume f is differentiable on [a, b]
axiom differentiable_on_I : DifferentiableOn ℝ f (Icc a b)

theorem lagrange_intermediate_value :
  ∃ (x0 : ℝ), x0 ∈ Ioo a b ∧ (deriv f x0) = (f a - f b) / (a - b) :=
sorry

end NUMINAMATH_GPT_lagrange_intermediate_value_l1600_160031


namespace NUMINAMATH_GPT_available_spaces_l1600_160007

noncomputable def numberOfBenches : ℕ := 50
noncomputable def capacityPerBench : ℕ := 4
noncomputable def peopleSeated : ℕ := 80

theorem available_spaces :
  let totalCapacity := numberOfBenches * capacityPerBench;
  let availableSpaces := totalCapacity - peopleSeated;
  availableSpaces = 120 := by
    sorry

end NUMINAMATH_GPT_available_spaces_l1600_160007


namespace NUMINAMATH_GPT_total_houses_in_neighborhood_l1600_160045

-- Definition of the function f
def f (x : ℕ) : ℕ := x^2 + 3*x

-- Given conditions
def x := 40

-- The theorem states that the total number of houses in Mariam's neighborhood is 1760.
theorem total_houses_in_neighborhood : (x + f x) = 1760 :=
by
  sorry

end NUMINAMATH_GPT_total_houses_in_neighborhood_l1600_160045


namespace NUMINAMATH_GPT_no_divisors_in_range_l1600_160029

theorem no_divisors_in_range : ¬ ∃ n : ℕ, 80 < n ∧ n < 90 ∧ n ∣ (3^40 - 1) :=
by sorry

end NUMINAMATH_GPT_no_divisors_in_range_l1600_160029


namespace NUMINAMATH_GPT_find_fz_l1600_160025

noncomputable def v (x y : ℝ) : ℝ :=
  3^x * Real.sin (y * Real.log 3)

theorem find_fz (x y : ℝ) (C : ℂ) (z : ℂ) (hz : z = x + y * Complex.I) :
  ∃ f : ℂ → ℂ, f z = 3^z + C :=
by
  sorry

end NUMINAMATH_GPT_find_fz_l1600_160025


namespace NUMINAMATH_GPT_intersection_A_B_l1600_160065

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1600_160065


namespace NUMINAMATH_GPT_quadratic_eq_has_distinct_real_roots_l1600_160070

theorem quadratic_eq_has_distinct_real_roots (c : ℝ) (h : c = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 ^ 2 - 3 * x1 + c = 0) ∧ (x2 ^ 2 - 3 * x2 + c = 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_eq_has_distinct_real_roots_l1600_160070


namespace NUMINAMATH_GPT_jungsoo_number_is_correct_l1600_160073

def J := (1 * 4) + (0.1 * 2) + (0.001 * 7)
def Y := 100 * J 
def S := Y + 0.05

theorem jungsoo_number_is_correct : S = 420.75 := by
  sorry

end NUMINAMATH_GPT_jungsoo_number_is_correct_l1600_160073


namespace NUMINAMATH_GPT_complement_of_A_in_S_l1600_160074

universe u

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_of_A_in_S : S \ A = {0, 1, 5} := 
by sorry

end NUMINAMATH_GPT_complement_of_A_in_S_l1600_160074


namespace NUMINAMATH_GPT_sqrt_11_bounds_l1600_160094

theorem sqrt_11_bounds : ∃ a : ℤ, a < Real.sqrt 11 ∧ Real.sqrt 11 < a + 1 ∧ a = 3 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_11_bounds_l1600_160094


namespace NUMINAMATH_GPT_range_of_m_l1600_160014

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (x^2 - 4*|x| + 5 - m = 0) → (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)) → (1 < m ∧ m < 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1600_160014


namespace NUMINAMATH_GPT_max_value_ratio_l1600_160042

theorem max_value_ratio (a b c: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_eq: a * (a + b + c) = b * c) :
  (a / (b + c) ≤ (Real.sqrt 2 - 1) / 2) :=
sorry -- proof omitted

end NUMINAMATH_GPT_max_value_ratio_l1600_160042


namespace NUMINAMATH_GPT_plane_equation_correct_l1600_160002

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 1, y := -1, z := 8 }
def B : Point3D := { x := -4, y := -3, z := 10 }
def C : Point3D := { x := -1, y := -1, z := 7 }

-- Define the vector BC
def vecBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

-- Define the equation of the plane
def planeEquation (P : Point3D) (normal : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  (normal.x, normal.y, normal.z, -(normal.x * P.x + normal.y * P.y + normal.z * P.z))

-- Calculate the equation of the plane passing through A and perpendicular to vector BC
def planeThroughAperpToBC : ℝ × ℝ × ℝ × ℝ :=
  let normal := vecBC B C
  planeEquation A normal

-- The expected result
def expectedPlaneEquation : ℝ × ℝ × ℝ × ℝ := (3, 2, -3, 23)

-- The theorem to be proved
theorem plane_equation_correct : planeThroughAperpToBC = expectedPlaneEquation := by
  sorry

end NUMINAMATH_GPT_plane_equation_correct_l1600_160002


namespace NUMINAMATH_GPT_joan_has_6_balloons_l1600_160006

theorem joan_has_6_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (h1 : initial_balloons = 8) (h2 : lost_balloons = 2) : initial_balloons - lost_balloons = 6 :=
sorry

end NUMINAMATH_GPT_joan_has_6_balloons_l1600_160006


namespace NUMINAMATH_GPT_price_of_case_l1600_160048

variables (bottles_per_day : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_spent : ℚ)

def total_bottles_consumed (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  bottles_per_day * days

def cases_needed (total_bottles : ℚ) (bottles_per_case : ℕ) : ℚ :=
  total_bottles / bottles_per_case

def price_per_case (total_spent : ℚ) (cases : ℚ) : ℚ :=
  total_spent / cases

theorem price_of_case (h1 : bottles_per_day = 1/2)
                      (h2 : days = 240)
                      (h3 : bottles_per_case = 24)
                      (h4 : total_spent = 60) :
  price_per_case total_spent (cases_needed (total_bottles_consumed bottles_per_day days) bottles_per_case) = 12 := 
sorry

end NUMINAMATH_GPT_price_of_case_l1600_160048


namespace NUMINAMATH_GPT_max_value_a_plus_b_l1600_160062

theorem max_value_a_plus_b
  (a b : ℝ)
  (h1 : 4 * a + 3 * b ≤ 10)
  (h2 : 3 * a + 5 * b ≤ 11) :
  a + b ≤ 156 / 55 :=
sorry

end NUMINAMATH_GPT_max_value_a_plus_b_l1600_160062


namespace NUMINAMATH_GPT_minimum_distance_l1600_160056

section MinimumDistance
open Real

noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := 2 * sqrt x
def t (x1 x2 : ℝ) := f x1 = g x2
def d (x1 x2 : ℝ) := abs (x2 - x1)

theorem minimum_distance : ∃ (x1 x2 : ℝ), t x1 x2 ∧ d x1 x2 = (1 - log 2) / 2 := 
sorry

end MinimumDistance

end NUMINAMATH_GPT_minimum_distance_l1600_160056


namespace NUMINAMATH_GPT_independent_variable_range_l1600_160016

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end NUMINAMATH_GPT_independent_variable_range_l1600_160016


namespace NUMINAMATH_GPT_white_tshirts_per_package_l1600_160018

theorem white_tshirts_per_package (p t : ℕ) (h1 : p = 28) (h2 : t = 56) :
  t / p = 2 :=
by 
  sorry

end NUMINAMATH_GPT_white_tshirts_per_package_l1600_160018


namespace NUMINAMATH_GPT_problem_1_problem_3_problem_4_l1600_160088

-- Definition of the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := (|x| * x) + (b * x) + c

-- Prove that when b > 0, f(x) is monotonically increasing on ℝ
theorem problem_1 (b c : ℝ) (h : b > 0) : 
  ∀ x y : ℝ, x < y → f x b c < f y b c :=
sorry

-- Prove that the graph of f(x) is symmetric about the point (0, c) when b = 0
theorem problem_3 (b c : ℝ) (h : b = 0) :
  ∀ x : ℝ, f x b c = f (-x) b c :=
sorry

-- Prove that when b < 0, f(x) = 0 can have three real roots
theorem problem_4 (b c : ℝ) (h : b < 0) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_3_problem_4_l1600_160088


namespace NUMINAMATH_GPT_we_the_people_cows_l1600_160023

theorem we_the_people_cows (W : ℕ) (h1 : ∃ H : ℕ, H = 3 * W + 2) (h2 : W + 3 * W + 2 = 70) : W = 17 :=
sorry

end NUMINAMATH_GPT_we_the_people_cows_l1600_160023


namespace NUMINAMATH_GPT_trees_planted_l1600_160075

def initial_trees : ℕ := 150
def total_trees_after_planting : ℕ := 225

theorem trees_planted (number_of_trees_planted : ℕ) : 
  number_of_trees_planted = total_trees_after_planting - initial_trees → number_of_trees_planted = 75 :=
by 
  sorry

end NUMINAMATH_GPT_trees_planted_l1600_160075


namespace NUMINAMATH_GPT_andrea_average_distance_per_day_l1600_160030

theorem andrea_average_distance_per_day
  (total_distance : ℕ := 168)
  (fraction_completed : ℚ := 3/7)
  (total_days : ℕ := 6)
  (days_completed : ℕ := 3) :
  (total_distance * (1 - fraction_completed) / (total_days - days_completed)) = 32 :=
by sorry

end NUMINAMATH_GPT_andrea_average_distance_per_day_l1600_160030


namespace NUMINAMATH_GPT_determine_k_if_even_function_l1600_160093

noncomputable def f (x k : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem determine_k_if_even_function (k : ℝ) (h_even: ∀ x : ℝ, f x k = f (-x) k ) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_if_even_function_l1600_160093


namespace NUMINAMATH_GPT_net_pay_rate_l1600_160078

def travelTime := 3 -- hours
def speed := 50 -- miles per hour
def fuelEfficiency := 25 -- miles per gallon
def earningsRate := 0.6 -- dollars per mile
def gasolineCost := 3 -- dollars per gallon

theorem net_pay_rate
  (travelTime : ℕ)
  (speed : ℕ)
  (fuelEfficiency : ℕ)
  (earningsRate : ℚ)
  (gasolineCost : ℚ)
  (h_time : travelTime = 3)
  (h_speed : speed = 50)
  (h_fuelEfficiency : fuelEfficiency = 25)
  (h_earningsRate : earningsRate = 0.6)
  (h_gasolineCost : gasolineCost = 3) :
  (earningsRate * speed * travelTime - (speed * travelTime / fuelEfficiency) * gasolineCost) / travelTime = 24 :=
by
  sorry

end NUMINAMATH_GPT_net_pay_rate_l1600_160078


namespace NUMINAMATH_GPT_evaluate_f_g3_l1600_160019

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 1
def g (x : ℝ) : ℝ := x + 3

theorem evaluate_f_g3 : f (g 3) = 97 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_g3_l1600_160019


namespace NUMINAMATH_GPT_grain_milling_necessary_pounds_l1600_160024

theorem grain_milling_necessary_pounds (x : ℝ) (h : 0.90 * x = 100) : x = 111 + 1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_grain_milling_necessary_pounds_l1600_160024


namespace NUMINAMATH_GPT_sequence_satisfies_n_squared_l1600_160099

theorem sequence_satisfies_n_squared (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) :
  ∀ n, a n = n^2 :=
by
  -- sorry
  sorry

end NUMINAMATH_GPT_sequence_satisfies_n_squared_l1600_160099


namespace NUMINAMATH_GPT_original_price_sarees_l1600_160055

theorem original_price_sarees
  (P : ℝ)
  (h : 0.90 * 0.85 * P = 378.675) :
  P = 495 :=
sorry

end NUMINAMATH_GPT_original_price_sarees_l1600_160055


namespace NUMINAMATH_GPT_part1_part2_l1600_160015

section 
variable {a b : ℚ}

-- Define the new operation as given in the condition
def odot (a b : ℚ) : ℚ := a * (a + b) - 1

-- Prove the given results
theorem part1 : odot 3 (-2) = 2 :=
by
  -- Proof omitted
  sorry

theorem part2 : odot (-2) (odot 3 5) = -43 :=
by
  -- Proof omitted
  sorry

end

end NUMINAMATH_GPT_part1_part2_l1600_160015


namespace NUMINAMATH_GPT_trees_occupy_area_l1600_160061

theorem trees_occupy_area
  (length : ℕ) (width : ℕ) (number_of_trees : ℕ)
  (h_length : length = 1000)
  (h_width : width = 2000)
  (h_trees : number_of_trees = 100000) :
  (length * width) / number_of_trees = 20 := 
by
  sorry

end NUMINAMATH_GPT_trees_occupy_area_l1600_160061


namespace NUMINAMATH_GPT_slower_speed_is_10_l1600_160035

-- Define the problem conditions
def walked_distance (faster_speed slower_speed actual_distance extra_distance : ℕ) : Prop :=
  actual_distance / slower_speed = (actual_distance + extra_distance) / faster_speed

-- Define main statement to prove
theorem slower_speed_is_10 (actual_distance : ℕ) (extra_distance : ℕ) (faster_speed : ℕ) (slower_speed : ℕ) :
  walked_distance faster_speed slower_speed actual_distance extra_distance ∧ 
  faster_speed = 15 ∧ extra_distance = 15 ∧ actual_distance = 30 → slower_speed = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_slower_speed_is_10_l1600_160035


namespace NUMINAMATH_GPT_simplify_expression_l1600_160051

theorem simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1600_160051


namespace NUMINAMATH_GPT_seokgi_walk_distance_correct_l1600_160037

-- Definitions of distances as per conditions
def entrance_to_temple_km : ℕ := 4
def entrance_to_temple_m : ℕ := 436
def temple_to_summit_m : ℕ := 1999

-- Total distance Seokgi walked in kilometers
def total_walked_km : ℕ := 12870

-- Proof statement
theorem seokgi_walk_distance_correct :
  ((entrance_to_temple_km * 1000 + entrance_to_temple_m) + temple_to_summit_m) * 2 / 1000 = total_walked_km / 1000 :=
by
  -- We will fill this in with the proof steps
  sorry

end NUMINAMATH_GPT_seokgi_walk_distance_correct_l1600_160037


namespace NUMINAMATH_GPT_no_real_roots_ffx_l1600_160027

theorem no_real_roots_ffx 
  (b c : ℝ) 
  (h : ∀ x : ℝ, (x^2 + (b - 1) * x + (c - 1) ≠ 0 ∨ ∀x: ℝ, (b - 1)^2 - 4 * (c - 1) < 0)) 
  : ∀ x : ℝ, (x^2 + bx + c)^2 + b * (x^2 + bx + c) + c ≠ x :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_ffx_l1600_160027


namespace NUMINAMATH_GPT_simplify_expression_l1600_160077

theorem simplify_expression :
  (1 / (1 / ((1 / 3)^1) + 1 / ((1 / 3)^2) + 1 / ((1 / 3)^3))) = 1 / 39 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1600_160077


namespace NUMINAMATH_GPT_train_speed_in_km_hr_l1600_160067

noncomputable def train_length : ℝ := 320
noncomputable def crossing_time : ℝ := 7.999360051195905
noncomputable def speed_in_meter_per_sec : ℝ := train_length / crossing_time
noncomputable def meter_per_sec_to_km_hr (speed_mps : ℝ) : ℝ := speed_mps * 3.6
noncomputable def expected_speed : ℝ := 144.018001125

theorem train_speed_in_km_hr :
  meter_per_sec_to_km_hr speed_in_meter_per_sec = expected_speed := by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_hr_l1600_160067


namespace NUMINAMATH_GPT_renaming_not_unnoticeable_l1600_160004

-- Define the conditions as necessary structures for cities and connections
structure City := (name : String)
structure Connection := (city1 city2 : City)

-- Definition of the king's list of connections
def kingList : List Connection := sorry  -- The complete list of connections

-- The renaming function represented generically
def rename (c1 c2 : City) : City := sorry  -- The renaming function which is unspecified here

-- The main theorem statement
noncomputable def renaming_condition (c1 c2 : City) : Prop :=
  -- This condition represents that renaming preserves the king's perception of connections
  ∀ c : City, sorry  -- The specific condition needs full details of renaming logic

-- The theorem to prove, which states that the renaming is not always unnoticeable
theorem renaming_not_unnoticeable : ∃ c1 c2 : City, ¬ renaming_condition c1 c2 := sorry

end NUMINAMATH_GPT_renaming_not_unnoticeable_l1600_160004


namespace NUMINAMATH_GPT_candy_remaining_l1600_160082

def initial_candy : ℝ := 1012.5
def talitha_took : ℝ := 283.7
def solomon_took : ℝ := 398.2
def maya_took : ℝ := 197.6

theorem candy_remaining : initial_candy - (talitha_took + solomon_took + maya_took) = 133 := 
by
  sorry

end NUMINAMATH_GPT_candy_remaining_l1600_160082


namespace NUMINAMATH_GPT_find_ratio_l1600_160081

-- Definitions
noncomputable def cost_per_gram_A : ℝ := 0.01
noncomputable def cost_per_gram_B : ℝ := 0.008
noncomputable def new_cost_per_gram_A : ℝ := 0.011
noncomputable def new_cost_per_gram_B : ℝ := 0.0072

def total_weight : ℝ := 1000

-- Theorem statement
theorem find_ratio (x y : ℝ) (h1 : x + y = total_weight)
    (h2 : cost_per_gram_A * x + cost_per_gram_B * y = new_cost_per_gram_A * x + new_cost_per_gram_B * y) :
    x / y = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_l1600_160081


namespace NUMINAMATH_GPT_minimum_cubes_required_l1600_160012

def box_length := 12
def box_width := 16
def box_height := 6
def cube_volume := 3

def volume_box := box_length * box_width * box_height

theorem minimum_cubes_required : volume_box / cube_volume = 384 := by
  sorry

end NUMINAMATH_GPT_minimum_cubes_required_l1600_160012


namespace NUMINAMATH_GPT_cubic_roots_c_over_d_l1600_160036

theorem cubic_roots_c_over_d (a b c d : ℤ) (h : a ≠ 0)
  (h_roots : ∃ r1 r2 r3, r1 = -1 ∧ r2 = 3 ∧ r3 = 4 ∧ 
              a * r1 * r2 * r3 + b * (r1 * r2 + r2 * r3 + r3 * r1) + c * (r1 + r2 + r3) + d = 0)
  : (c : ℚ) / d = 5 / 12 := 
sorry

end NUMINAMATH_GPT_cubic_roots_c_over_d_l1600_160036


namespace NUMINAMATH_GPT_power_function_passes_through_1_1_l1600_160038

theorem power_function_passes_through_1_1 (a : ℝ) : (1 : ℝ) ^ a = 1 := 
by
  sorry

end NUMINAMATH_GPT_power_function_passes_through_1_1_l1600_160038


namespace NUMINAMATH_GPT_right_angled_isosceles_triangle_third_side_length_l1600_160079

theorem right_angled_isosceles_triangle_third_side_length (a b c : ℝ) (h₀ : a = 50) (h₁ : b = 50) (h₂ : a + b + c = 160) : c = 60 :=
by
  -- TODO: Provide proof
  sorry

end NUMINAMATH_GPT_right_angled_isosceles_triangle_third_side_length_l1600_160079


namespace NUMINAMATH_GPT_sequence_values_l1600_160008

theorem sequence_values (x y z : ℕ) 
    (h1 : x = 14 * 3) 
    (h2 : y = x - 1) 
    (h3 : z = y * 3) : 
    x = 42 ∧ y = 41 ∧ z = 123 := by 
    sorry

end NUMINAMATH_GPT_sequence_values_l1600_160008


namespace NUMINAMATH_GPT_value_of_m_minus_n_over_n_l1600_160047

theorem value_of_m_minus_n_over_n (m n : ℚ) (h : (2/3 : ℚ) * m = (5/6 : ℚ) * n) :
  (m - n) / n = 1 / 4 := 
sorry

end NUMINAMATH_GPT_value_of_m_minus_n_over_n_l1600_160047


namespace NUMINAMATH_GPT_mary_no_torn_cards_l1600_160044

theorem mary_no_torn_cards
  (T : ℕ) -- number of Mary's initial torn baseball cards
  (initial_cards : ℕ := 18) -- initial baseball cards
  (fred_cards : ℕ := 26) -- baseball cards given by Fred
  (bought_cards : ℕ := 40) -- baseball cards bought
  (total_cards : ℕ := 84) -- total baseball cards Mary has now
  (h : initial_cards - T + fred_cards + bought_cards = total_cards)
  : T = 0 :=
by sorry

end NUMINAMATH_GPT_mary_no_torn_cards_l1600_160044


namespace NUMINAMATH_GPT_product_of_divisors_18_l1600_160086

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end NUMINAMATH_GPT_product_of_divisors_18_l1600_160086


namespace NUMINAMATH_GPT_Ali_money_left_l1600_160068

theorem Ali_money_left (initial_money : ℕ) 
  (spent_on_food_ratio : ℚ) 
  (spent_on_glasses_ratio : ℚ) 
  (spent_on_food : ℕ) 
  (left_after_food : ℕ) 
  (spent_on_glasses : ℕ) 
  (final_left : ℕ) :
    initial_money = 480 →
    spent_on_food_ratio = 1 / 2 →
    spent_on_food = initial_money * spent_on_food_ratio →
    left_after_food = initial_money - spent_on_food →
    spent_on_glasses_ratio = 1 / 3 →
    spent_on_glasses = left_after_food * spent_on_glasses_ratio →
    final_left = left_after_food - spent_on_glasses →
    final_left = 160 :=
by
  sorry

end NUMINAMATH_GPT_Ali_money_left_l1600_160068


namespace NUMINAMATH_GPT_p_pow_four_minus_one_divisible_by_ten_l1600_160005

theorem p_pow_four_minus_one_divisible_by_ten
  (p : Nat) (prime_p : Nat.Prime p) (h₁ : p ≠ 2) (h₂ : p ≠ 5) : 
  10 ∣ (p^4 - 1) := 
by
  sorry

end NUMINAMATH_GPT_p_pow_four_minus_one_divisible_by_ten_l1600_160005


namespace NUMINAMATH_GPT_total_wheels_in_neighborhood_l1600_160057

def cars_in_Jordan_driveway := 2
def wheels_per_car := 4
def spare_wheel := 1
def bikes_with_2_wheels := 3
def wheels_per_bike := 2
def bike_missing_rear_wheel := 1
def bike_with_training_wheel := 2 + 1
def trash_can_wheels := 2
def tricycle_wheels := 3
def wheelchair_main_wheels := 2
def wheelchair_small_wheels := 2
def wagon_wheels := 4
def roller_skates_total_wheels := 4
def roller_skates_missing_wheel := 1

def pickup_truck_wheels := 4
def boat_trailer_wheels := 2
def motorcycle_wheels := 2
def atv_wheels := 4

theorem total_wheels_in_neighborhood :
  (cars_in_Jordan_driveway * wheels_per_car + spare_wheel + bikes_with_2_wheels * wheels_per_bike + bike_missing_rear_wheel + bike_with_training_wheel + trash_can_wheels + tricycle_wheels + wheelchair_main_wheels + wheelchair_small_wheels + wagon_wheels + (roller_skates_total_wheels - roller_skates_missing_wheel)) +
  (pickup_truck_wheels + boat_trailer_wheels + motorcycle_wheels + atv_wheels) = 47 := by
  sorry

end NUMINAMATH_GPT_total_wheels_in_neighborhood_l1600_160057


namespace NUMINAMATH_GPT_gcd_of_powers_l1600_160000

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2016 - 1) (h2 : n = 2^2008 - 1) : 
  Nat.gcd m n = 255 :=
by
  -- (Definitions and steps are omitted as only the statement is required)
  sorry

end NUMINAMATH_GPT_gcd_of_powers_l1600_160000


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l1600_160058

def Point : Type := ℤ × ℤ

def in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def A : Point := (-3, 7)
def B : Point := (3, -7)
def C : Point := (3, 7)
def D : Point := (-3, -7)

theorem point_in_fourth_quadrant : in_fourth_quadrant B :=
by {
  -- skipping the proof steps for the purpose of this example
  sorry
}

end NUMINAMATH_GPT_point_in_fourth_quadrant_l1600_160058


namespace NUMINAMATH_GPT_pipes_fill_tank_in_1_5_hours_l1600_160054

theorem pipes_fill_tank_in_1_5_hours :
  (1 / 3 + 1 / 9 + 1 / 18 + 1 / 6) = (2 / 3) →
  (1 / (2 / 3)) = (3 / 2) :=
by sorry

end NUMINAMATH_GPT_pipes_fill_tank_in_1_5_hours_l1600_160054


namespace NUMINAMATH_GPT_inequality_ineqs_l1600_160021

theorem inequality_ineqs (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_cond : x * y + y * z + z * x = 1) :
  (27 / 4) * (x + y) * (y + z) * (z + x) 
  ≥ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2
  ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2 
  ≥ 
  6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_inequality_ineqs_l1600_160021


namespace NUMINAMATH_GPT_coins_problem_l1600_160092

theorem coins_problem : ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 7) ∧ (n % 11 = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_coins_problem_l1600_160092


namespace NUMINAMATH_GPT_A_alone_days_l1600_160095

theorem A_alone_days (A B C : ℝ) (hB: B = 9) (hC: C = 7.2) 
  (h: 1 / A + 1 / B + 1 / C = 1 / 2) : A = 2 :=
by
  rw [hB, hC] at h
  sorry

end NUMINAMATH_GPT_A_alone_days_l1600_160095


namespace NUMINAMATH_GPT_ambulance_ride_cost_correct_l1600_160028

noncomputable def total_bill : ℝ := 18000
noncomputable def medication_percentage : ℝ := 0.35
noncomputable def imaging_percentage : ℝ := 0.15
noncomputable def surgery_percentage : ℝ := 0.25
noncomputable def overnight_stays_percentage : ℝ := 0.10
noncomputable def doctors_fees_percentage : ℝ := 0.05

noncomputable def food_fee : ℝ := 300
noncomputable def consultation_fee : ℝ := 450
noncomputable def physical_therapy_fee : ℝ := 600

noncomputable def medication_cost : ℝ := medication_percentage * total_bill
noncomputable def imaging_cost : ℝ := imaging_percentage * total_bill
noncomputable def surgery_cost : ℝ := surgery_percentage * total_bill
noncomputable def overnight_stays_cost : ℝ := overnight_stays_percentage * total_bill
noncomputable def doctors_fees_cost : ℝ := doctors_fees_percentage * total_bill

noncomputable def percentage_based_costs : ℝ :=
  medication_cost + imaging_cost + surgery_cost + overnight_stays_cost + doctors_fees_cost

noncomputable def fixed_costs : ℝ :=
  food_fee + consultation_fee + physical_therapy_fee

noncomputable def total_known_costs : ℝ :=
  percentage_based_costs + fixed_costs

noncomputable def ambulance_ride_cost : ℝ :=
  total_bill - total_known_costs

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 450 := by
  sorry

end NUMINAMATH_GPT_ambulance_ride_cost_correct_l1600_160028


namespace NUMINAMATH_GPT_score_order_l1600_160026

variable (A B C D : ℕ)

theorem score_order
  (h1 : A + C = B + D)
  (h2 : B > D)
  (h3 : C > A + B) :
  C > B ∧ B > A ∧ A > D :=
by 
  sorry

end NUMINAMATH_GPT_score_order_l1600_160026


namespace NUMINAMATH_GPT_mean_proportional_l1600_160072

variable (a b c d : ℕ)
variable (x : ℕ)

def is_geometric_mean (a b : ℕ) (x : ℕ) := x = Int.sqrt (a * b)

theorem mean_proportional (h49 : a = 49) (h64 : b = 64) (h81 : d = 81)
  (h_geometric1 : x = 56) (h_geometric2 : c = 72) :
  c = 64 := sorry

end NUMINAMATH_GPT_mean_proportional_l1600_160072


namespace NUMINAMATH_GPT_shortest_chord_length_l1600_160033

theorem shortest_chord_length
  (x y : ℝ)
  (hx : x^2 + y^2 - 6 * x - 8 * y = 0)
  (point_on_circle : (3, 5) = (x, y)) :
  ∃ (length : ℝ), length = 4 * Real.sqrt 6 := 
by
  sorry

end NUMINAMATH_GPT_shortest_chord_length_l1600_160033


namespace NUMINAMATH_GPT_factors_of_expression_l1600_160041

def total_distinct_factors : ℕ :=
  let a := 10
  let b := 3
  let c := 2
  (a + 1) * (b + 1) * (c + 1)

theorem factors_of_expression :
  total_distinct_factors = 132 :=
by 
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_factors_of_expression_l1600_160041


namespace NUMINAMATH_GPT_hexagon_can_be_divided_into_congruent_triangles_l1600_160083

section hexagon_division

-- Definitions
variables {H : Type} -- H represents the type for hexagon

-- Conditions
variables (is_hexagon : H → Prop) -- A predicate stating that a shape is a hexagon
variables (lies_on_grid : H → Prop) -- A predicate stating that the hexagon lies on the grid
variables (can_cut_along_grid_lines : H → Prop) -- A predicate stating that cuts can only be made along the grid lines
variables (identical_figures : Type u → Prop) -- A predicate stating that the obtained figures must be identical
variables (congruent_triangles : Type u → Prop) -- A predicate stating that the obtained figures are congruent triangles
variables (area_division : H → Prop) -- A predicate stating that the area of the hexagon is divided equally

-- Theorem statement
theorem hexagon_can_be_divided_into_congruent_triangles (h : H)
  (H_is_hexagon : is_hexagon h)
  (H_on_grid : lies_on_grid h)
  (H_cut : can_cut_along_grid_lines h) :
  ∃ (F : Type u), identical_figures F ∧ congruent_triangles F ∧ area_division h :=
sorry

end hexagon_division

end NUMINAMATH_GPT_hexagon_can_be_divided_into_congruent_triangles_l1600_160083


namespace NUMINAMATH_GPT_find_b_l1600_160009

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end NUMINAMATH_GPT_find_b_l1600_160009


namespace NUMINAMATH_GPT_parabola_tangent_line_l1600_160001

theorem parabola_tangent_line (a : ℝ) : 
  (∀ x : ℝ, (y = ax^2 + 6 ↔ y = x)) → a = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_line_l1600_160001


namespace NUMINAMATH_GPT_shopkeeper_profit_percent_l1600_160049

theorem shopkeeper_profit_percent (cost_price profit : ℝ) (h1 : cost_price = 960) (h2 : profit = 40) : 
  (profit / cost_price) * 100 = 4.17 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percent_l1600_160049


namespace NUMINAMATH_GPT_cos_theta_value_l1600_160089

open Real

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

noncomputable def cos_theta (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1 ^ 2 + u.2 ^ 2) * Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem cos_theta_value :
  cos_theta a b = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_theta_value_l1600_160089


namespace NUMINAMATH_GPT_sampling_is_simple_random_l1600_160017

-- Definitions based on conditions
def total_students := 200
def students_sampled := 20
def sampling_method := "Simple Random Sampling"

-- The problem: given the random sampling of 20 students from 200, prove that the method is simple random sampling.
theorem sampling_is_simple_random :
  (total_students = 200 ∧ students_sampled = 20) → sampling_method = "Simple Random Sampling" := 
by
  sorry

end NUMINAMATH_GPT_sampling_is_simple_random_l1600_160017


namespace NUMINAMATH_GPT_day_of_week_150th_day_previous_year_l1600_160069

theorem day_of_week_150th_day_previous_year (N : ℕ) 
  (h1 : (275 % 7 = 4))  -- Thursday is 4th day of the week if starting from Sunday as 0
  (h2 : (215 % 7 = 4))  -- Similarly, Thursday is 4th day of the week
  : (150 % 7 = 6) :=     -- Proving the 150th day of year N-1 is a Saturday (Saturday as 6th day of the week)
sorry

end NUMINAMATH_GPT_day_of_week_150th_day_previous_year_l1600_160069


namespace NUMINAMATH_GPT_clocks_resynchronize_after_days_l1600_160010

/-- Arthur's clock gains 15 minutes per day. -/
def arthurs_clock_gain_per_day : ℕ := 15

/-- Oleg's clock gains 12 minutes per day. -/
def olegs_clock_gain_per_day : ℕ := 12

/-- The clocks display time in a 12-hour format, which is equivalent to 720 minutes. -/
def twelve_hour_format_in_minutes : ℕ := 720

/-- 
  After how many days will this situation first repeat given the 
  conditions of gain in Arthur's and Oleg's clocks and the 12-hour format.
-/
theorem clocks_resynchronize_after_days :
  ∃ (N : ℕ), N * arthurs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N * olegs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N = 240 :=
by
  sorry

end NUMINAMATH_GPT_clocks_resynchronize_after_days_l1600_160010


namespace NUMINAMATH_GPT_real_solution_count_l1600_160076

noncomputable def f (x : ℝ) : ℝ :=
  (1/(x - 1)) + (2/(x - 2)) + (3/(x - 3)) + (4/(x - 4)) + 
  (5/(x - 5)) + (6/(x - 6)) + (7/(x - 7)) + (8/(x - 8)) + 
  (9/(x - 9)) + (10/(x - 10))

theorem real_solution_count : ∃ n : ℕ, n = 11 ∧ 
  ∃ x : ℝ, f x = x :=
sorry

end NUMINAMATH_GPT_real_solution_count_l1600_160076


namespace NUMINAMATH_GPT_find_s_is_neg4_l1600_160080

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_is_neg4 : (∃ s : ℝ, g (-1) s = 0) ↔ (s = -4) :=
sorry

end NUMINAMATH_GPT_find_s_is_neg4_l1600_160080


namespace NUMINAMATH_GPT_difference_of_squares_l1600_160043

theorem difference_of_squares (a b : ℕ) (h₁ : a = 69842) (h₂ : b = 30158) :
  (a^2 - b^2) / (a - b) = 100000 :=
by
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1600_160043


namespace NUMINAMATH_GPT_Ashutosh_time_to_complete_job_l1600_160052

noncomputable def SureshWorkRate : ℝ := 1 / 15
noncomputable def AshutoshWorkRate (A : ℝ) : ℝ := 1 / A
noncomputable def SureshWorkIn9Hours : ℝ := 9 * SureshWorkRate

theorem Ashutosh_time_to_complete_job (A : ℝ) :
  (1 - SureshWorkIn9Hours) * AshutoshWorkRate A = 14 / 35 →
  A = 35 :=
by
  sorry

end NUMINAMATH_GPT_Ashutosh_time_to_complete_job_l1600_160052


namespace NUMINAMATH_GPT_lotus_leaves_not_odd_l1600_160022

theorem lotus_leaves_not_odd (n : ℕ) (h1 : n > 1) (h2 : ∀ t : ℕ, ∃ r : ℕ, 0 ≤ r ∧ r < n ∧ (t * (t + 1) / 2 - 1) % n = r) : ¬ Odd n :=
sorry

end NUMINAMATH_GPT_lotus_leaves_not_odd_l1600_160022


namespace NUMINAMATH_GPT_no_intersection_abs_functions_l1600_160020

open Real

theorem no_intersection_abs_functions : 
  ∀ f g : ℝ → ℝ, 
  (∀ x, f x = |2 * x + 5|) → 
  (∀ x, g x = -|3 * x - 2|) → 
  (∀ y, ∀ x1 x2, f x1 = y ∧ g x2 = y → y = 0 ∧ x1 = -5/2 ∧ x2 = 2/3 → (x1 ≠ x2)) → 
  (∃ x, f x = g x) → 
  false := 
  by
    intro f g hf hg h
    sorry

end NUMINAMATH_GPT_no_intersection_abs_functions_l1600_160020


namespace NUMINAMATH_GPT_find_x_l1600_160098

theorem find_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 :=
sorry

end NUMINAMATH_GPT_find_x_l1600_160098


namespace NUMINAMATH_GPT_weighted_average_correct_l1600_160039

-- Define the marks and credits for each subject
def marks_english := 90
def marks_mathematics := 92
def marks_physics := 85
def marks_chemistry := 87
def marks_biology := 85

def credits_english := 3
def credits_mathematics := 4
def credits_physics := 4
def credits_chemistry := 3
def credits_biology := 2

-- Define the weighted sum and total credits
def weighted_sum := marks_english * credits_english + marks_mathematics * credits_mathematics + marks_physics * credits_physics + marks_chemistry * credits_chemistry + marks_biology * credits_biology
def total_credits := credits_english + credits_mathematics + credits_physics + credits_chemistry + credits_biology

-- Prove that the weighted average is 88.0625
theorem weighted_average_correct : (weighted_sum.toFloat / total_credits.toFloat) = 88.0625 :=
by 
  sorry

end NUMINAMATH_GPT_weighted_average_correct_l1600_160039


namespace NUMINAMATH_GPT_intersection_M_N_l1600_160097

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1600_160097


namespace NUMINAMATH_GPT_number_of_members_l1600_160050

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end NUMINAMATH_GPT_number_of_members_l1600_160050


namespace NUMINAMATH_GPT_min_tosses_one_head_l1600_160040

theorem min_tosses_one_head (n : ℕ) (P : ℝ) (h₁ : P = 1 - (1 / 2) ^ n) (h₂ : P ≥ 15 / 16) : n ≥ 4 :=
by
  sorry -- Proof to be filled in.

end NUMINAMATH_GPT_min_tosses_one_head_l1600_160040


namespace NUMINAMATH_GPT_root_eq_neg_l1600_160053

theorem root_eq_neg {a : ℝ} (h : 3 * a - 9 < 0) : (a - 4) * (a - 5) > 0 :=
by
  sorry

end NUMINAMATH_GPT_root_eq_neg_l1600_160053


namespace NUMINAMATH_GPT_coal_consumption_rel_l1600_160063

variables (Q a x y : ℝ)
variables (h₀ : 0 < x) (h₁ : x < a) (h₂ : Q ≠ 0) (h₃ : a ≠ 0) (h₄ : a - x ≠ 0)

theorem coal_consumption_rel :
  y = Q / (a - x) - Q / a :=
sorry

end NUMINAMATH_GPT_coal_consumption_rel_l1600_160063


namespace NUMINAMATH_GPT_union_of_sets_l1600_160034

open Set

noncomputable def A (a : ℝ) : Set ℝ := {1, 2^a}
noncomputable def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) (h₁ : A a ∩ B a b = {1 / 2}) :
  A a ∪ B a b = {-1, 1 / 2, 1} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1600_160034


namespace NUMINAMATH_GPT_set_intersection_l1600_160085

theorem set_intersection (A B : Set ℝ)
  (hA : A = { x : ℝ | 1 < x ∧ x < 4 })
  (hB : B = { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }) :
  A ∩ (Set.univ \ B) = { x : ℝ | 3 < x ∧ x < 4 } :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1600_160085


namespace NUMINAMATH_GPT_minimum_route_length_l1600_160091

/-- 
Given a city with the shape of a 5 × 5 square grid,
prove that the minimum length of a route that covers each street exactly once and 
returns to the starting point is 68, considering each street can be walked any number of times. 
-/
theorem minimum_route_length (n : ℕ) (h1 : n = 5) : 
  ∃ route_length : ℕ, route_length = 68 := 
sorry

end NUMINAMATH_GPT_minimum_route_length_l1600_160091


namespace NUMINAMATH_GPT_each_student_contribution_l1600_160032

-- Definitions for conditions in the problem
def numberOfStudents : ℕ := 30
def totalAmount : ℕ := 480
def numberOfFridaysInTwoMonths : ℕ := 8

-- Statement to prove
theorem each_student_contribution (numberOfStudents : ℕ) (totalAmount : ℕ) (numberOfFridaysInTwoMonths : ℕ) : 
  totalAmount / (numberOfFridaysInTwoMonths * numberOfStudents) = 2 := 
by
  sorry

end NUMINAMATH_GPT_each_student_contribution_l1600_160032


namespace NUMINAMATH_GPT_person_birth_date_l1600_160059

theorem person_birth_date
  (x : ℕ)
  (h1 : 1937 - x = x^2 - x)
  (d m : ℕ)
  (h2 : 44 + m = d^2)
  (h3 : 0 < m ∧ m < 13)
  (h4 : d = 7 ∧ m = 5) :
  (x = 44 ∧ 1937 - (x + x^2) = 1892) ∧  d = 7 ∧ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_person_birth_date_l1600_160059


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1600_160011

theorem solve_equation1 (x : ℝ) : 3 * (x - 1)^3 = 24 ↔ x = 3 := by
  sorry

theorem solve_equation2 (x : ℝ) : (x - 3)^2 = 64 ↔ x = 11 ∨ x = -5 := by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1600_160011


namespace NUMINAMATH_GPT_robot_paths_from_A_to_B_l1600_160060

/-- Define a function that computes the number of distinct paths a robot can take -/
def distinctPaths (A B : ℕ × ℕ) : ℕ := sorry

/-- Proof statement: There are 556 distinct paths from A to B, given the movement conditions -/
theorem robot_paths_from_A_to_B (A B : ℕ × ℕ) (h_move : (A, B) = ((0, 0), (10, 10))) :
  distinctPaths A B = 556 :=
sorry

end NUMINAMATH_GPT_robot_paths_from_A_to_B_l1600_160060


namespace NUMINAMATH_GPT_tax_diminished_by_16_percent_l1600_160046

variables (T X : ℝ)

-- Condition: The new revenue is 96.6% of the original revenue
def new_revenue_effect : Prop :=
  (1.15 * (T - X) / 100) = (T / 100) * 0.966

-- Target: Prove that X is 16% of T
theorem tax_diminished_by_16_percent (h : new_revenue_effect T X) : X = 0.16 * T :=
sorry

end NUMINAMATH_GPT_tax_diminished_by_16_percent_l1600_160046


namespace NUMINAMATH_GPT_remainder_base12_2543_div_9_l1600_160096

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end NUMINAMATH_GPT_remainder_base12_2543_div_9_l1600_160096


namespace NUMINAMATH_GPT_roll_2_four_times_last_not_2_l1600_160084

def probability_of_rolling_2_four_times_last_not_2 : ℚ :=
  (1/6)^4 * (5/6)

theorem roll_2_four_times_last_not_2 :
  probability_of_rolling_2_four_times_last_not_2 = 5 / 7776 := 
by
  sorry

end NUMINAMATH_GPT_roll_2_four_times_last_not_2_l1600_160084
