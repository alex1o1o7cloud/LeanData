import Mathlib

namespace NUMINAMATH_GPT_pyramid_height_l1004_100436

noncomputable def height_of_pyramid : ℝ :=
  let perimeter := 32
  let pb := 12
  let side := perimeter / 4
  let fb := (side * Real.sqrt 2) / 2
  Real.sqrt (pb^2 - fb^2)

theorem pyramid_height :
  height_of_pyramid = 4 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_height_l1004_100436


namespace NUMINAMATH_GPT_part1_relationship_range_part2_maximize_profit_l1004_100406

variables {x y a : ℝ}
noncomputable def zongzi_profit (x : ℝ) : ℝ := -5 * x + 6000

-- Given conditions
def conditions (x : ℝ) : Prop :=
  100 ≤ x ∧ x ≤ 150

-- Part 1: Prove the functional relationship and range of x
theorem part1_relationship_range (x : ℝ) (h : conditions x) :
  zongzi_profit x = -5 * x + 6000 :=
  sorry

-- Part 2: Profit maximization given modified purchase price condition
noncomputable def modified_zongzi_profit (x : ℝ) (a : ℝ) : ℝ :=
  (a - 5) * x + 6000

def maximize_strategy (x a : ℝ) : Prop :=
  (0 < a ∧ a < 5 → x = 100) ∧ (5 ≤ a ∧ a < 10 → x = 150)

theorem part2_maximize_profit (a : ℝ) (ha : 0 < a ∧ a < 10) :
  ∃ x, conditions x ∧ maximize_strategy x a :=
  sorry

end NUMINAMATH_GPT_part1_relationship_range_part2_maximize_profit_l1004_100406


namespace NUMINAMATH_GPT_find_f3_l1004_100438

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 - c * x + 2

theorem find_f3 (a b c : ℝ)
  (h1 : f a b c (-3) = 9) :
  f a b c 3 = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_f3_l1004_100438


namespace NUMINAMATH_GPT_determine_p_l1004_100417

variable (x y z p : ℝ)

theorem determine_p (h1 : 8 / (x + y) = p / (x + z)) (h2 : p / (x + z) = 12 / (z - y)) : p = 20 :=
sorry

end NUMINAMATH_GPT_determine_p_l1004_100417


namespace NUMINAMATH_GPT_am_gm_inequality_l1004_100418

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1004_100418


namespace NUMINAMATH_GPT_unit_price_of_first_batch_minimum_selling_price_l1004_100434

-- Proof Problem 1
theorem unit_price_of_first_batch :
  (∃ x : ℝ, (3200 / x) * 2 = 7200 / (x + 10) ∧ x = 80) := 
  sorry

-- Proof Problem 2
theorem minimum_selling_price (x : ℝ) (hx : x = 80) :
  (40 * x + 80 * (x + 10) - 3200 - 7200 + 20 * 0.8 * x ≥ 3520) → 
  (∃ y : ℝ, y ≥ 120) :=
  sorry

end NUMINAMATH_GPT_unit_price_of_first_batch_minimum_selling_price_l1004_100434


namespace NUMINAMATH_GPT_max_value_PXQ_l1004_100450

theorem max_value_PXQ :
  ∃ (X P Q : ℕ), (XX = 10 * X + X) ∧ (10 * X + X) * X = 100 * P + 10 * X + Q ∧ 
  (X = 1 ∨ X = 5 ∨ X = 6) ∧ 
  (100 * P + 10 * X + Q) = 396 :=
sorry

end NUMINAMATH_GPT_max_value_PXQ_l1004_100450


namespace NUMINAMATH_GPT_parabolas_intersect_at_points_l1004_100433

theorem parabolas_intersect_at_points :
  ∀ (x y : ℝ), (y = 3 * x^2 - 12 * x - 9) ↔ (y = 2 * x^2 - 8 * x + 5) →
  (x, y) = (2 + 3 * Real.sqrt 2, 66 - 36 * Real.sqrt 2) ∨ (x, y) = (2 - 3 * Real.sqrt 2, 66 + 36 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_parabolas_intersect_at_points_l1004_100433


namespace NUMINAMATH_GPT_common_pasture_area_l1004_100475

variable (Area_Ivanov Area_Petrov Area_Sidorov Area_Vasilev Area_Ermolaev : ℝ)
variable (Common_Pasture : ℝ)

theorem common_pasture_area :
  Area_Ivanov = 24 ∧
  Area_Petrov = 28 ∧
  Area_Sidorov = 10 ∧
  Area_Vasilev = 20 ∧
  Area_Ermolaev = 30 →
  Common_Pasture = 17.5 :=
sorry

end NUMINAMATH_GPT_common_pasture_area_l1004_100475


namespace NUMINAMATH_GPT_greatest_price_drop_is_april_l1004_100424

-- Define the price changes for each month
def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => 1.00
  | 2 => -1.50
  | 3 => -0.50
  | 4 => -3.75 -- including the -1.25 adjustment
  | 5 => 0.50
  | 6 => -2.25
  | _ => 0 -- default case, although we only deal with months 1-6

-- Define a predicate for the month with the greatest drop
def greatest_drop_month (m : ℕ) : Prop :=
  m = 4

-- Main theorem: Prove that the month with the greatest price drop is April
theorem greatest_price_drop_is_april : greatest_drop_month 4 :=
by
  -- Use Lean tactics to prove the statement
  sorry

end NUMINAMATH_GPT_greatest_price_drop_is_april_l1004_100424


namespace NUMINAMATH_GPT_cylinder_heights_relation_l1004_100469

variables {r1 r2 h1 h2 : ℝ}

theorem cylinder_heights_relation 
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = (6 / 5) * r1) :
  h1 = 1.44 * h2 :=
by sorry

end NUMINAMATH_GPT_cylinder_heights_relation_l1004_100469


namespace NUMINAMATH_GPT_boys_from_Pine_l1004_100445

/-
We need to prove that the number of boys from Pine Middle School is 70
given the following conditions:
1. There were 150 students in total.
2. 90 were boys and 60 were girls.
3. 50 students were from Maple Middle School.
4. 100 students were from Pine Middle School.
5. 30 of the girls were from Maple Middle School.
-/
theorem boys_from_Pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h_total : total_students = 150) (h_boys : total_boys = 90)
  (h_girls : total_girls = 60) (h_maple : maple_students = 50)
  (h_pine : pine_students = 100) (h_maple_girls : maple_girls = 30) :
  total_boys - maple_students + maple_girls = 70 :=
by
  sorry

end NUMINAMATH_GPT_boys_from_Pine_l1004_100445


namespace NUMINAMATH_GPT_sum_lent_l1004_100470

theorem sum_lent (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ)
  (hR: R = 4) 
  (hT: T = 8) 
  (hI1 : I = P - 306) 
  (hI2 : I = P * R * T / 100) :
  P = 450 :=
by
  sorry

end NUMINAMATH_GPT_sum_lent_l1004_100470


namespace NUMINAMATH_GPT_gcd_two_powers_l1004_100402

noncomputable def gcd_expression (m n : ℕ) : ℕ :=
  Int.gcd (2^m + 1) (2^n - 1)

theorem gcd_two_powers (m n : ℕ) (hm : m > 0) (hn : n > 0) (odd_n : n % 2 = 1) : 
  gcd_expression m n = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_two_powers_l1004_100402


namespace NUMINAMATH_GPT_average_tomatoes_per_day_l1004_100486

theorem average_tomatoes_per_day :
  let t₁ := 120
  let t₂ := t₁ + 50
  let t₃ := 2 * t₂
  let t₄ := t₁ / 2
  (t₁ + t₂ + t₃ + t₄) / 4 = 172.5 := by
  sorry

end NUMINAMATH_GPT_average_tomatoes_per_day_l1004_100486


namespace NUMINAMATH_GPT_find_f_of_500_l1004_100478

theorem find_f_of_500
  (f : ℕ → ℕ)
  (h_pos : ∀ x y : ℕ, f x > 0 ∧ f y > 0) 
  (h_mul : ∀ x y : ℕ, f (x * y) = f x + f y) 
  (h_f10 : f 10 = 15)
  (h_f40 : f 40 = 23) :
  f 500 = 41 :=
sorry

end NUMINAMATH_GPT_find_f_of_500_l1004_100478


namespace NUMINAMATH_GPT_christmas_bonus_remainder_l1004_100473

theorem christmas_bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_christmas_bonus_remainder_l1004_100473


namespace NUMINAMATH_GPT_ben_gave_18_fish_l1004_100453

variable (initial_fish : ℕ) (total_fish : ℕ) (given_fish : ℕ)

theorem ben_gave_18_fish
    (h1 : initial_fish = 31)
    (h2 : total_fish = 49)
    (h3 : total_fish = initial_fish + given_fish) :
    given_fish = 18 :=
by
  sorry

end NUMINAMATH_GPT_ben_gave_18_fish_l1004_100453


namespace NUMINAMATH_GPT_change_factor_l1004_100430

theorem change_factor (avg1 avg2 : ℝ) (n : ℕ) (h_avg1 : avg1 = 40) (h_n : n = 10) (h_avg2 : avg2 = 80) : avg2 * (n : ℝ) / (avg1 * (n : ℝ)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_change_factor_l1004_100430


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1004_100443

noncomputable def a : ℝ := 2^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1004_100443


namespace NUMINAMATH_GPT_least_area_of_square_l1004_100488

theorem least_area_of_square :
  ∀ (s : ℝ), (3.5 ≤ s ∧ s < 4.5) → (s * s ≥ 12.25) :=
by
  intro s
  intro hs
  sorry

end NUMINAMATH_GPT_least_area_of_square_l1004_100488


namespace NUMINAMATH_GPT_trajectory_of_P_l1004_100444

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (2 * x - 3) ^ 2 + 4 * y ^ 2 = 1

theorem trajectory_of_P (m n x y : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : 2 * x = 3 + m ∧ 2 * y = n) : trajectory_equation x y :=
by 
  sorry

end NUMINAMATH_GPT_trajectory_of_P_l1004_100444


namespace NUMINAMATH_GPT_find_n_l1004_100458

-- Given Variables
variables (n x y : ℝ)

-- Given Conditions
axiom h1 : n * x = 6 * y
axiom h2 : x * y ≠ 0
axiom h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998

-- Conclusion
theorem find_n : n = 5 := sorry

end NUMINAMATH_GPT_find_n_l1004_100458


namespace NUMINAMATH_GPT_examine_points_l1004_100499

variable (Bryan Jen Sammy mistakes : ℕ)

def problem_conditions : Prop :=
  Bryan = 20 ∧ Jen = Bryan + 10 ∧ Sammy = Jen - 2 ∧ mistakes = 7

theorem examine_points (h : problem_conditions Bryan Jen Sammy mistakes) : ∃ total_points : ℕ, total_points = Sammy + mistakes :=
by {
  sorry
}

end NUMINAMATH_GPT_examine_points_l1004_100499


namespace NUMINAMATH_GPT_find_f_of_neg_1_l1004_100409

-- Define the conditions
variables (a b c : ℝ)
variables (g f : ℝ → ℝ)
axiom g_definition : ∀ x, g x = x^3 + a*x^2 + 2*x + 15
axiom f_definition : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c

axiom g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3)
axiom roots_of_g_are_roots_of_f : ∀ x, g x = 0 → f x = 0

-- Prove the value of f(-1) given the conditions
theorem find_f_of_neg_1 (a : ℝ) (b : ℝ) (c : ℝ) (g f : ℝ → ℝ)
  (h_g_def : ∀ x, g x = x^3 + a*x^2 + 2*x + 15)
  (h_f_def : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c)
  (h_g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3))
  (h_roots : ∀ x, g x = 0 → f x = 0) :
  f (-1) = 3733.25 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_f_of_neg_1_l1004_100409


namespace NUMINAMATH_GPT_management_sampled_count_l1004_100425

variable (total_employees salespeople management_personnel logistical_support staff_sample_size : ℕ)
variable (proportional_sampling : Prop)
variable (n_management_sampled : ℕ)

axiom h1 : total_employees = 160
axiom h2 : salespeople = 104
axiom h3 : management_personnel = 32
axiom h4 : logistical_support = 24
axiom h5 : proportional_sampling
axiom h6 : staff_sample_size = 20

theorem management_sampled_count : n_management_sampled = 4 :=
by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_management_sampled_count_l1004_100425


namespace NUMINAMATH_GPT_Intersection_A_B_l1004_100479

open Set

theorem Intersection_A_B :
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  show A ∩ B = {x : ℝ | -3 < x ∧ x < 1}
  sorry

end NUMINAMATH_GPT_Intersection_A_B_l1004_100479


namespace NUMINAMATH_GPT_interest_rate_decrease_l1004_100421

theorem interest_rate_decrease (initial_rate final_rate : ℝ) (x : ℝ) 
  (h_initial_rate : initial_rate = 2.25 * 0.01)
  (h_final_rate : final_rate = 1.98 * 0.01) :
  final_rate = initial_rate * (1 - x)^2 := 
  sorry

end NUMINAMATH_GPT_interest_rate_decrease_l1004_100421


namespace NUMINAMATH_GPT_sqrt_31_estimate_l1004_100451

theorem sqrt_31_estimate : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_31_estimate_l1004_100451


namespace NUMINAMATH_GPT_lex_read_pages_l1004_100482

theorem lex_read_pages (total_pages days : ℕ) (h1 : total_pages = 240) (h2 : days = 12) :
  total_pages / days = 20 :=
by sorry

end NUMINAMATH_GPT_lex_read_pages_l1004_100482


namespace NUMINAMATH_GPT_shaded_area_fraction_l1004_100405

theorem shaded_area_fraction :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let Q := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let R := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let S := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  let area_triangle := 1 / 2 * 2 * 2
  let shaded_area := 2 * area_triangle
  let total_area := 4 * 4
  shaded_area / total_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_fraction_l1004_100405


namespace NUMINAMATH_GPT_factorization_example_l1004_100487

open Function

theorem factorization_example (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by
  sorry

end NUMINAMATH_GPT_factorization_example_l1004_100487


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1004_100466

theorem arithmetic_geometric_mean_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1004_100466


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l1004_100403

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) ↔ (m = 7 ∨ m = -5) :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l1004_100403


namespace NUMINAMATH_GPT_product_is_correct_l1004_100464

def number : ℕ := 3460
def multiplier : ℕ := 12
def correct_product : ℕ := 41520

theorem product_is_correct : multiplier * number = correct_product := by
  sorry

end NUMINAMATH_GPT_product_is_correct_l1004_100464


namespace NUMINAMATH_GPT_minimum_area_of_Archimedean_triangle_l1004_100435

-- Define the problem statement with necessary conditions
theorem minimum_area_of_Archimedean_triangle (p : ℝ) (hp : p > 0) :
  ∃ (ABQ_area : ℝ), ABQ_area = p^2 ∧ 
    (∀ (A B Q : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * p * A.1) ∧
      (B.2 ^ 2 = 2 * p * B.1) ∧
      (0, 0) = (p / 2, p / 2) ∧
      (Q.2 = 0) → 
      ABQ_area = p^2) :=
sorry

end NUMINAMATH_GPT_minimum_area_of_Archimedean_triangle_l1004_100435


namespace NUMINAMATH_GPT_mechanic_charge_per_hour_l1004_100471

/-- Definitions based on provided conditions -/
def total_amount_paid : ℝ := 300
def part_cost : ℝ := 150
def hours : ℕ := 2

/-- Theorem stating the labor cost per hour is $75 -/
theorem mechanic_charge_per_hour (total_amount_paid part_cost hours : ℝ) : hours = 2 → part_cost = 150 → total_amount_paid = 300 → 
  (total_amount_paid - part_cost) / hours = 75 :=
by
  sorry

end NUMINAMATH_GPT_mechanic_charge_per_hour_l1004_100471


namespace NUMINAMATH_GPT_paper_plate_cup_cost_l1004_100446

variables (P C : ℝ)

theorem paper_plate_cup_cost (h : 100 * P + 200 * C = 6) : 20 * P + 40 * C = 1.20 :=
by sorry

end NUMINAMATH_GPT_paper_plate_cup_cost_l1004_100446


namespace NUMINAMATH_GPT_desired_cost_of_mixture_l1004_100474

theorem desired_cost_of_mixture 
  (w₈ : ℝ) (c₈ : ℝ) -- weight and cost per pound of the $8 candy
  (w₅ : ℝ) (c₅ : ℝ) -- weight and cost per pound of the $5 candy
  (total_w : ℝ) (desired_cost : ℝ) -- total weight and desired cost per pound of the mixture
  (h₁ : w₈ = 30) (h₂ : c₈ = 8) 
  (h₃ : w₅ = 60) (h₄ : c₅ = 5)
  (h₅ : total_w = w₈ + w₅)
  (h₆ : desired_cost = (w₈ * c₈ + w₅ * c₅) / total_w) :
  desired_cost = 6 := 
by
  sorry

end NUMINAMATH_GPT_desired_cost_of_mixture_l1004_100474


namespace NUMINAMATH_GPT_problem_conditions_l1004_100415

variables (a b : ℝ)
open Real

theorem problem_conditions (ha : a < 0) (hb : 0 < b) (hab : a + b > 0) :
  (a / b > -1) ∧ (abs a < abs b) ∧ (1 / a + 1 / b ≤ 0) ∧ ((a - 1) * (b - 1) < 1) := sorry

end NUMINAMATH_GPT_problem_conditions_l1004_100415


namespace NUMINAMATH_GPT_factor_expression_l1004_100489

theorem factor_expression (x : ℝ) : 
  3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1004_100489


namespace NUMINAMATH_GPT_initial_miles_correct_l1004_100481

-- Definitions and conditions
def miles_per_gallon : ℕ := 30
def gallons_per_tank : ℕ := 20
def current_miles : ℕ := 2928
def tanks_filled : ℕ := 2

-- Question: How many miles were on the car before the road trip?
def initial_miles : ℕ := current_miles - (miles_per_gallon * gallons_per_tank * tanks_filled)

-- Proof problem statement
theorem initial_miles_correct : initial_miles = 1728 :=
by
  -- Here we expect the proof, but are skipping it with 'sorry'
  sorry

end NUMINAMATH_GPT_initial_miles_correct_l1004_100481


namespace NUMINAMATH_GPT_degree_of_g_l1004_100480

noncomputable def poly_degree (p : Polynomial ℝ) : ℕ :=
  Polynomial.natDegree p

theorem degree_of_g
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ := f.comp g - g)
  (hf : poly_degree f = 3)
  (hh : poly_degree h = 8) :
  poly_degree g = 3 :=
sorry

end NUMINAMATH_GPT_degree_of_g_l1004_100480


namespace NUMINAMATH_GPT_value_of_x_for_g_equals_g_inv_l1004_100465

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_for_g_equals_g_inv_l1004_100465


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l1004_100439

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l1004_100439


namespace NUMINAMATH_GPT_standard_equation_of_parabola_l1004_100401

theorem standard_equation_of_parabola (focus : ℝ × ℝ): 
  (focus.1 - 2 * focus.2 - 4 = 0) → 
  ((focus = (4, 0) → (∃ a : ℝ, ∀ x y : ℝ, y^2 = 4 * a * x)) ∨
   (focus = (0, -2) → (∃ b : ℝ, ∀ x y : ℝ, x^2 = 4 * b * y))) :=
by
  sorry

end NUMINAMATH_GPT_standard_equation_of_parabola_l1004_100401


namespace NUMINAMATH_GPT_total_votes_cast_is_8200_l1004_100426

variable (V : ℝ) (h1 : 0.35 * V < V) (h2 : 0.35 * V + 2460 = 0.65 * V)

theorem total_votes_cast_is_8200 (V : ℝ)
  (h1 : 0.35 * V < V)
  (h2 : 0.35 * V + 2460 = 0.65 * V) :
  V = 8200 := by
sorry

end NUMINAMATH_GPT_total_votes_cast_is_8200_l1004_100426


namespace NUMINAMATH_GPT_initial_price_correct_l1004_100456

-- Definitions based on the conditions
def initial_price : ℝ := 3  -- Rs. 3 per kg
def new_price : ℝ := 5      -- Rs. 5 per kg
def reduction_in_consumption : ℝ := 0.4  -- 40%

-- The main theorem we need to prove
theorem initial_price_correct :
  initial_price = 3 :=
sorry

end NUMINAMATH_GPT_initial_price_correct_l1004_100456


namespace NUMINAMATH_GPT_homogeneous_variances_l1004_100467

noncomputable def sample_sizes : (ℕ × ℕ × ℕ) := (9, 13, 15)
noncomputable def sample_variances : (ℝ × ℝ × ℝ) := (3.2, 3.8, 6.3)
noncomputable def significance_level : ℝ := 0.05
noncomputable def degrees_of_freedom : ℕ := 2
noncomputable def V : ℝ := 1.43
noncomputable def critical_value : ℝ := 6.0

theorem homogeneous_variances :
  V < critical_value :=
by
  sorry

end NUMINAMATH_GPT_homogeneous_variances_l1004_100467


namespace NUMINAMATH_GPT_inequality_bounds_l1004_100483

theorem inequality_bounds (x y : ℝ) : |y - 3 * x| < 2 * x ↔ x > 0 ∧ x < y ∧ y < 5 * x := by
  sorry

end NUMINAMATH_GPT_inequality_bounds_l1004_100483


namespace NUMINAMATH_GPT_least_n_satisfies_condition_l1004_100484

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end NUMINAMATH_GPT_least_n_satisfies_condition_l1004_100484


namespace NUMINAMATH_GPT_sum_a10_a11_l1004_100427

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)

theorem sum_a10_a11 {a : ℕ → ℝ} (h_seq : geometric_sequence a)
  (h1 : a 1 + a 2 = 2)
  (h4 : a 4 + a 5 = 4) :
  a 10 + a 11 = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_a10_a11_l1004_100427


namespace NUMINAMATH_GPT_age_difference_l1004_100407

theorem age_difference :
  ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧
    (∀ x y : ℕ, (x = 10 * a + b) ∧ (y = 10 * b + a) → 
    (x + 5 = 2 * (y + 5)) ∧ ((10 * a + b) - (10 * b + a) = 18)) :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1004_100407


namespace NUMINAMATH_GPT_r_daily_earnings_l1004_100412

-- Given conditions as definitions
def daily_earnings (P Q R : ℕ) : Prop :=
(P + Q + R) * 9 = 1800 ∧ (P + R) * 5 = 600 ∧ (Q + R) * 7 = 910

-- Theorem statement corresponding to the problem
theorem r_daily_earnings : ∃ R : ℕ, ∀ P Q : ℕ, daily_earnings P Q R → R = 50 :=
by sorry

end NUMINAMATH_GPT_r_daily_earnings_l1004_100412


namespace NUMINAMATH_GPT_functions_eq_l1004_100477

open Function

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem functions_eq (h_surj : Surjective f) (h_inj : Injective g) (h_ge : ∀ n : ℕ, f n ≥ g n) : ∀ n : ℕ, f n = g n :=
sorry

end NUMINAMATH_GPT_functions_eq_l1004_100477


namespace NUMINAMATH_GPT_factor_polynomial_l1004_100491

theorem factor_polynomial (a : ℝ) : 74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1004_100491


namespace NUMINAMATH_GPT_room_width_is_12_l1004_100492

variable (w : ℝ)

def length_of_room : ℝ := 20
def width_of_veranda : ℝ := 2
def area_of_veranda : ℝ := 144

theorem room_width_is_12 :
  24 * (w + 4) - 20 * w = 144 → w = 12 := by
  sorry

end NUMINAMATH_GPT_room_width_is_12_l1004_100492


namespace NUMINAMATH_GPT_max_expression_value_l1004_100429

theorem max_expression_value (a b c d e f g h k : ℤ) 
  (ha : a = 1 ∨ a = -1)
  (hb : b = 1 ∨ b = -1)
  (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1)
  (he : e = 1 ∨ e = -1)
  (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1)
  (hh : h = 1 ∨ h = -1)
  (hk : k = 1 ∨ k = -1) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_expression_value_l1004_100429


namespace NUMINAMATH_GPT_European_to_American_swallow_ratio_l1004_100495

theorem European_to_American_swallow_ratio (a e : ℝ) (n_E : ℕ) 
  (h1 : a = 5)
  (h2 : 2 * n_E + n_E = 90)
  (h3 : 60 * a + 30 * e = 600) :
  e / a = 2 := 
by
  sorry

end NUMINAMATH_GPT_European_to_American_swallow_ratio_l1004_100495


namespace NUMINAMATH_GPT_num_initial_pairs_of_shoes_l1004_100494

theorem num_initial_pairs_of_shoes (lost_shoes remaining_pairs : ℕ)
  (h1 : lost_shoes = 9)
  (h2 : remaining_pairs = 20) :
  (initial_pairs : ℕ) = 25 :=
sorry

end NUMINAMATH_GPT_num_initial_pairs_of_shoes_l1004_100494


namespace NUMINAMATH_GPT_james_baked_multiple_l1004_100468

theorem james_baked_multiple (x : ℕ) (h1 : 115 ≠ 0) (h2 : 1380 = 115 * x) : x = 12 :=
sorry

end NUMINAMATH_GPT_james_baked_multiple_l1004_100468


namespace NUMINAMATH_GPT_sequence_infinite_coprime_l1004_100472

theorem sequence_infinite_coprime (a : ℤ) (h : a > 1) :
  ∃ (S : ℕ → ℕ), (∀ n m : ℕ, n ≠ m → Int.gcd (a^(S n + 1) + a^S n - 1) (a^(S m + 1) + a^S m - 1) = 1) :=
sorry

end NUMINAMATH_GPT_sequence_infinite_coprime_l1004_100472


namespace NUMINAMATH_GPT_line_bisects_circle_area_l1004_100460

theorem line_bisects_circle_area (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + b ↔ x^2 + y^2 - 2 * x - 4 * y + 4 = 0) → b = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_bisects_circle_area_l1004_100460


namespace NUMINAMATH_GPT_min_sets_bound_l1004_100428

theorem min_sets_bound (A : Type) (n k : ℕ) (S : Finset (Finset A))
  (h₁ : S.card = k)
  (h₂ : ∀ x y : A, x ≠ y → ∃ B ∈ S, (x ∈ B ∧ y ∉ B) ∨ (y ∈ B ∧ x ∉ B)) :
  2^k ≥ n :=
sorry

end NUMINAMATH_GPT_min_sets_bound_l1004_100428


namespace NUMINAMATH_GPT_solve_for_m_l1004_100419

-- Define the operation ◎ for real numbers a and b
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Lean statement for the proof problem
theorem solve_for_m (m : ℝ) (h : op (m + 1) (m - 2) = 16) : m = 3 ∨ m = -2 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l1004_100419


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l1004_100410

variable (S T I : ℝ)

-- Conditions
def tips_as_fraction_of_salary : Prop := T = (3/4) * S
def total_income : Prop := I = S + T

-- Theorem stating the proof problem
theorem fraction_of_income_from_tips 
  (h1 : tips_as_fraction_of_salary S T)
  (h2 : total_income S T I) : (T / I) = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l1004_100410


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1004_100496

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + x - 3 = 0) : x^3 + 2 * x^2 - 2 * x + 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1004_100496


namespace NUMINAMATH_GPT_fraction_addition_l1004_100420

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l1004_100420


namespace NUMINAMATH_GPT_characterize_set_A_l1004_100485

open Int

noncomputable def A : Set ℤ := { x | x^2 - 3 * x - 4 < 0 }

theorem characterize_set_A : A = {0, 1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_characterize_set_A_l1004_100485


namespace NUMINAMATH_GPT_train_speed_kmh_l1004_100463

theorem train_speed_kmh (T P: ℝ) (L: ℝ):
  (T = L + 320) ∧ (L = 18 * P) ->
  P = 20 -> 
  P * 3.6 = 72 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_kmh_l1004_100463


namespace NUMINAMATH_GPT_sum_of_all_potential_real_values_of_x_l1004_100442

/-- Determine the sum of all potential real values of x such that when the mean, median, 
and mode of the list [12, 3, 6, 3, 8, 3, x, 15] are arranged in increasing order, they 
form a non-constant arithmetic progression. -/
def sum_potential_x_values : ℚ :=
    let values := [12, 3, 6, 3, 8, 3, 15]
    let mean (x : ℚ) : ℚ := (50 + x) / 8
    let mode : ℚ := 3
    let median (x : ℚ) : ℚ := 
      if x ≤ 3 then 3.5 else if x < 6 then (x + 6) / 2 else 6
    let is_arithmetic_seq (a b c : ℚ) : Prop := 2 * b = a + c
    let valid_x_values : List ℚ := 
      (if is_arithmetic_seq mode 3.5 (mean (3.5)) then [] else []) ++
      (if is_arithmetic_seq mode 6 (mean 6) then [22] else []) ++
      (if is_arithmetic_seq mode (median (50 / 7)) (mean (50 / 7)) then [50 / 7] else [])
    (valid_x_values.sum)
theorem sum_of_all_potential_real_values_of_x :
  sum_potential_x_values = 204 / 7 :=
  sorry

end NUMINAMATH_GPT_sum_of_all_potential_real_values_of_x_l1004_100442


namespace NUMINAMATH_GPT_intersection_P_Q_l1004_100497

-- Define set P
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define set Q (using real numbers, but we will be interested in natural number intersections)
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- The intersection of P with Q in the natural numbers should be {1, 2}
theorem intersection_P_Q :
  {x : ℕ | x ∈ P ∧ (x : ℝ) ∈ Q} = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1004_100497


namespace NUMINAMATH_GPT_eighth_term_matchstick_count_l1004_100462

def matchstick_sequence (n : ℕ) : ℕ := (n + 1) * 3

theorem eighth_term_matchstick_count : matchstick_sequence 8 = 27 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_eighth_term_matchstick_count_l1004_100462


namespace NUMINAMATH_GPT_line_through_P0_perpendicular_to_plane_l1004_100448

-- Definitions of the given conditions
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P0 : Point3D := { x := 3, y := 4, z := 2 }

def plane (x y z : ℝ) : Prop := 8 * x - 4 * y + 5 * z - 4 = 0

-- The proof problem statement
theorem line_through_P0_perpendicular_to_plane :
  ∃ t : ℝ, (P0.x + 8 * t = x ∧ P0.y - 4 * t = y ∧ P0.z + 5 * t = z) ↔
    (∃ t : ℝ, x = 3 + 8 * t ∧ y = 4 - 4 * t ∧ z = 2 + 5 * t) → 
    (∃ t : ℝ, (x - 3) / 8 = t ∧ (y - 4) / -4 = t ∧ (z - 2) / 5 = t) := sorry

end NUMINAMATH_GPT_line_through_P0_perpendicular_to_plane_l1004_100448


namespace NUMINAMATH_GPT_no_solution_for_inequality_system_l1004_100476

theorem no_solution_for_inequality_system (x : ℝ) : 
  ¬ ((2 * x + 3 ≥ x + 11) ∧ (((2 * x + 5) / 3 - 1) < (2 - x))) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_inequality_system_l1004_100476


namespace NUMINAMATH_GPT_finite_steps_iff_power_of_2_l1004_100422

-- Define the conditions of the problem
def S (k n : ℕ) : ℕ := (k * (k + 1) / 2) % n

-- Define the predicate to check if the game finishes in finite number of steps
def game_completes (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < n → S (k + i) n ≠ S k n

-- The main statement to prove
theorem finite_steps_iff_power_of_2 (n : ℕ) : game_completes n ↔ ∃ t : ℕ, n = 2^t :=
sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_finite_steps_iff_power_of_2_l1004_100422


namespace NUMINAMATH_GPT_maria_min_score_fifth_term_l1004_100459

theorem maria_min_score_fifth_term (score1 score2 score3 score4 : ℕ) (avg_required : ℕ) 
  (h1 : score1 = 84) (h2 : score2 = 80) (h3 : score3 = 82) (h4 : score4 = 78)
  (h_avg_required : avg_required = 85) :
  ∃ x : ℕ, x ≥ 101 :=
by
  sorry

end NUMINAMATH_GPT_maria_min_score_fifth_term_l1004_100459


namespace NUMINAMATH_GPT_train_crossing_time_l1004_100416

noncomputable def length_train : ℝ := 250
noncomputable def length_bridge : ℝ := 150
noncomputable def speed_train_kmh : ℝ := 57.6
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

theorem train_crossing_time : 
  let total_length := length_train + length_bridge 
  let time := total_length / speed_train_ms 
  time = 25 := 
by 
  -- Convert all necessary units and parameters
  let length_train := (250 : ℝ)
  let length_bridge := (150 : ℝ)
  let speed_train_ms := (57.6 * (1000 / 3600) : ℝ)
  
  -- Compute the total length and time
  let total_length := length_train + length_bridge
  let time := total_length / speed_train_ms
  
  -- State the proof
  show time = 25
  { sorry }

end NUMINAMATH_GPT_train_crossing_time_l1004_100416


namespace NUMINAMATH_GPT_wilson_buys_3_bottles_of_cola_l1004_100400

theorem wilson_buys_3_bottles_of_cola
    (num_hamburgers : ℕ := 2) 
    (cost_per_hamburger : ℕ := 5) 
    (cost_per_cola : ℕ := 2) 
    (discount : ℕ := 4) 
    (total_paid : ℕ := 12) :
    num_hamburgers * cost_per_hamburger - discount + x * cost_per_cola = total_paid → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_wilson_buys_3_bottles_of_cola_l1004_100400


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1004_100440

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1004_100440


namespace NUMINAMATH_GPT_notebook_cost_proof_l1004_100455

-- Let n be the cost of the notebook and p be the cost of the pen.
variable (n p : ℝ)

-- Conditions:
def total_cost : Prop := n + p = 2.50
def notebook_more_pen : Prop := n = 2 + p

-- Theorem: Prove that the cost of the notebook is $2.25
theorem notebook_cost_proof (h1 : total_cost n p) (h2 : notebook_more_pen n p) : n = 2.25 := 
by 
  sorry

end NUMINAMATH_GPT_notebook_cost_proof_l1004_100455


namespace NUMINAMATH_GPT_sum_of_ages_in_5_years_l1004_100498

noncomputable def age_will_three_years_ago := 4
noncomputable def years_elapsed := 3
noncomputable def age_will_now := age_will_three_years_ago + years_elapsed
noncomputable def age_diane_now := 2 * age_will_now
noncomputable def years_into_future := 5
noncomputable def age_will_in_future := age_will_now + years_into_future
noncomputable def age_diane_in_future := age_diane_now + years_into_future

theorem sum_of_ages_in_5_years :
  age_will_in_future + age_diane_in_future = 31 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_in_5_years_l1004_100498


namespace NUMINAMATH_GPT_expected_value_in_classroom_l1004_100449

noncomputable def expected_pairs_next_to_each_other (boys girls : ℕ) : ℕ :=
  if boys = 9 ∧ girls = 14 ∧ boys + girls = 23 then
    10 -- Based on provided conditions and conclusion
  else
    0

theorem expected_value_in_classroom :
  expected_pairs_next_to_each_other 9 14 = 10 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_in_classroom_l1004_100449


namespace NUMINAMATH_GPT_tank_weight_when_full_l1004_100457

theorem tank_weight_when_full (p q : ℝ) (x y : ℝ)
  (h1 : x + (3/4) * y = p)
  (h2 : x + (1/3) * y = q) :
  x + y = (8/5) * p - (8/5) * q :=
by
  sorry

end NUMINAMATH_GPT_tank_weight_when_full_l1004_100457


namespace NUMINAMATH_GPT_cycle_reappear_l1004_100432

/-- Given two sequences with cycle lengths 6 and 4, prove the sequences will align on line number 12 -/
theorem cycle_reappear (l1 l2 : ℕ) (h1 : l1 = 6) (h2 : l2 = 4) :
  Nat.lcm l1 l2 = 12 := by
  sorry

end NUMINAMATH_GPT_cycle_reappear_l1004_100432


namespace NUMINAMATH_GPT_angle_same_terminal_side_l1004_100431

theorem angle_same_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 → α = 330 :=
by
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l1004_100431


namespace NUMINAMATH_GPT_chef_earns_less_than_manager_l1004_100490

theorem chef_earns_less_than_manager :
  let manager_wage := 7.50
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * 1.20
  (manager_wage - chef_wage) = 3.00 := by
    sorry

end NUMINAMATH_GPT_chef_earns_less_than_manager_l1004_100490


namespace NUMINAMATH_GPT_system_has_infinitely_many_solutions_l1004_100493

theorem system_has_infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ)), (∀ x y z : ℝ, (x + y = 2 ∧ xy - z^2 = 1) ↔ (x, y, z) ∈ S) ∧ S.Infinite :=
by
  sorry

end NUMINAMATH_GPT_system_has_infinitely_many_solutions_l1004_100493


namespace NUMINAMATH_GPT_find_focus_with_larger_x_coordinate_l1004_100408

noncomputable def focus_of_hyperbola_with_larger_x_coordinate : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 7
  let b := 9
  let c := Real.sqrt (a^2 + b^2)
  (h + c, k)

theorem find_focus_with_larger_x_coordinate :
  focus_of_hyperbola_with_larger_x_coordinate = (5 + Real.sqrt 130, 20) := by
  sorry

end NUMINAMATH_GPT_find_focus_with_larger_x_coordinate_l1004_100408


namespace NUMINAMATH_GPT_circle_standard_equation_l1004_100461

theorem circle_standard_equation {a : ℝ} :
  (∃ a : ℝ, a ≠ 0 ∧ (a = 2 * a - 3 ∨ a = 3 - 2 * a) ∧ 
  (((x - a)^2 + (y - (2 * a - 3))^2 = a^2) ∧ 
   ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1))) :=
sorry

end NUMINAMATH_GPT_circle_standard_equation_l1004_100461


namespace NUMINAMATH_GPT_marcie_cups_coffee_l1004_100404

theorem marcie_cups_coffee (S M T : ℕ) (h1 : S = 6) (h2 : S + M = 8) : M = 2 :=
by
  sorry

end NUMINAMATH_GPT_marcie_cups_coffee_l1004_100404


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1004_100413
open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1004_100413


namespace NUMINAMATH_GPT_Tod_drove_time_l1004_100411

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end NUMINAMATH_GPT_Tod_drove_time_l1004_100411


namespace NUMINAMATH_GPT_sum_of_cubes_correct_l1004_100423

noncomputable def expression_for_sum_of_cubes (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) : Prop :=
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + b^3 * d^3) / (a * b * c * d)

theorem sum_of_cubes_correct (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) :
  expression_for_sum_of_cubes x y z w a b c d hx hy hz hw ha hb hc hd hxy hxz hyz hxw :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_correct_l1004_100423


namespace NUMINAMATH_GPT_find_annual_interest_rate_l1004_100454

-- Define the given conditions
def principal : ℝ := 10000
def time : ℝ := 1  -- since 12 months is 1 year for annual rate
def simple_interest : ℝ := 800

-- Define the annual interest rate to be proved
def annual_interest_rate : ℝ := 0.08

-- The theorem stating the problem
theorem find_annual_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) : 
  P = principal → 
  T = time → 
  SI = simple_interest → 
  SI = P * annual_interest_rate * T := 
by
  intros hP hT hSI
  rw [hP, hT, hSI]
  unfold annual_interest_rate
  -- here's where we skip the proof
  sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l1004_100454


namespace NUMINAMATH_GPT_upgraded_video_card_multiple_l1004_100437

noncomputable def multiple_of_video_card_cost (computer_cost monitor_cost_peripheral_cost base_video_card_cost total_spent upgraded_video_card_cost : ℝ) : ℝ :=
  upgraded_video_card_cost / base_video_card_cost

theorem upgraded_video_card_multiple
  (computer_cost : ℝ)
  (monitor_cost_ratio : ℝ)
  (base_video_card_cost : ℝ)
  (total_spent : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : monitor_cost_ratio = 1/5)
  (h3 : base_video_card_cost = 300)
  (h4 : total_spent = 2100) :
  multiple_of_video_card_cost computer_cost (computer_cost * monitor_cost_ratio) base_video_card_cost total_spent (total_spent - (computer_cost + computer_cost * monitor_cost_ratio)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_upgraded_video_card_multiple_l1004_100437


namespace NUMINAMATH_GPT_graph_is_point_l1004_100414

theorem graph_is_point : ∀ x y : ℝ, x^2 + 3 * y^2 - 4 * x - 6 * y + 7 = 0 ↔ (x = 2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_graph_is_point_l1004_100414


namespace NUMINAMATH_GPT_second_quadrant_y_value_l1004_100441

theorem second_quadrant_y_value :
  ∀ (b : ℝ), (-3, b).2 > 0 → b = 2 :=
by
  sorry

end NUMINAMATH_GPT_second_quadrant_y_value_l1004_100441


namespace NUMINAMATH_GPT_find_y_l1004_100447

def G (a b c d : ℕ) : ℕ := a ^ b + c * d

theorem find_y (y : ℕ) : G 3 y 5 10 = 350 ↔ y = 5 := by
  sorry

end NUMINAMATH_GPT_find_y_l1004_100447


namespace NUMINAMATH_GPT_vector_k_range_l1004_100452

noncomputable def vector_length (v : (ℝ × ℝ)) : ℝ := (v.1 ^ 2 + v.2 ^ 2).sqrt

theorem vector_k_range :
  let a := (-2, 2)
  let b := (5, k)
  vector_length (a.1 + b.1, a.2 + b.2) ≤ 5 → -6 ≤ k ∧ k ≤ 2 := by
  sorry

end NUMINAMATH_GPT_vector_k_range_l1004_100452
