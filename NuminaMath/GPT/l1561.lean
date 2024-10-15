import Mathlib

namespace NUMINAMATH_GPT_speed_downstream_l1561_156134

def speed_in_still_water := 12 -- man in still water
def speed_of_stream := 6  -- speed of stream
def speed_upstream := 6  -- rowing upstream

theorem speed_downstream : 
  speed_in_still_water + speed_of_stream = 18 := 
by 
  sorry

end NUMINAMATH_GPT_speed_downstream_l1561_156134


namespace NUMINAMATH_GPT_farmer_harvested_correctly_l1561_156145

def estimated_harvest : ℕ := 213489
def additional_harvest : ℕ := 13257
def total_harvest : ℕ := 226746

theorem farmer_harvested_correctly :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end NUMINAMATH_GPT_farmer_harvested_correctly_l1561_156145


namespace NUMINAMATH_GPT_problem_solution_l1561_156143

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 8 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l1561_156143


namespace NUMINAMATH_GPT_solution_set_ineq_l1561_156125

theorem solution_set_ineq (x : ℝ) : (1 < x ∧ x ≤ 3) ↔ (x - 3) / (x - 1) ≤ 0 := sorry

end NUMINAMATH_GPT_solution_set_ineq_l1561_156125


namespace NUMINAMATH_GPT_eqn_solution_set_l1561_156159

theorem eqn_solution_set :
  {x : ℝ | x ^ 2 - 1 = 0} = {-1, 1} := 
sorry

end NUMINAMATH_GPT_eqn_solution_set_l1561_156159


namespace NUMINAMATH_GPT_number_of_students_exclusively_in_math_l1561_156116

variable (T M F K : ℕ)
variable (students_in_math students_in_foreign_language students_only_music : ℕ)
variable (students_not_in_music total_students_only_non_music : ℕ)

theorem number_of_students_exclusively_in_math (hT: T = 120) (hM: M = 82)
    (hF: F = 71) (hK: K = 20) :
    T - K = 100 →
    (M + F - 53 = T - K) →
    M - 53 = 29 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_students_exclusively_in_math_l1561_156116


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1561_156118

theorem system1_solution :
  ∃ x y : ℝ, 3 * x + 4 * y = 16 ∧ 5 * x - 8 * y = 34 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

theorem system2_solution :
  ∃ x y : ℝ, (x - 1) / 2 + (y + 1) / 3 = 1 ∧ x + y = 4 ∧ x = -1 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1561_156118


namespace NUMINAMATH_GPT_smallest_n_l1561_156156

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 4 * n = k1^2) (h2 : ∃ k2, 3 * n = k2^3) : n = 144 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1561_156156


namespace NUMINAMATH_GPT_ratio_is_1_to_3_l1561_156114

-- Definitions based on the conditions
def washed_on_wednesday : ℕ := 6
def washed_on_thursday : ℕ := 2 * washed_on_wednesday
def washed_on_friday : ℕ := washed_on_thursday / 2
def total_washed : ℕ := 26
def washed_on_saturday : ℕ := total_washed - washed_on_wednesday - washed_on_thursday - washed_on_friday

-- The ratio calculation
def ratio_saturday_to_wednesday : ℚ := washed_on_saturday / washed_on_wednesday

-- The theorem to prove
theorem ratio_is_1_to_3 : ratio_saturday_to_wednesday = 1 / 3 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_ratio_is_1_to_3_l1561_156114


namespace NUMINAMATH_GPT_min_distance_from_curve_to_focus_l1561_156103

noncomputable def minDistanceToFocus (x y θ : ℝ) : ℝ :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  a - c

theorem min_distance_from_curve_to_focus :
  ∀ θ : ℝ, minDistanceToFocus (2 * Real.cos θ) (3 * Real.sin θ) θ = 3 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_from_curve_to_focus_l1561_156103


namespace NUMINAMATH_GPT_CapeMay_more_than_twice_Daytona_l1561_156147

def Daytona_sharks : ℕ := 12
def CapeMay_sharks : ℕ := 32

theorem CapeMay_more_than_twice_Daytona : CapeMay_sharks - 2 * Daytona_sharks = 8 := by
  sorry

end NUMINAMATH_GPT_CapeMay_more_than_twice_Daytona_l1561_156147


namespace NUMINAMATH_GPT_man_speed_l1561_156152

theorem man_speed (distance_meters time_minutes : ℝ) (h_distance : distance_meters = 1250) (h_time : time_minutes = 15) :
  (distance_meters / 1000) / (time_minutes / 60) = 5 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_l1561_156152


namespace NUMINAMATH_GPT_remainder_21_l1561_156182

theorem remainder_21 (y : ℤ) (k : ℤ) (h : y = 288 * k + 45) : y % 24 = 21 := 
  sorry

end NUMINAMATH_GPT_remainder_21_l1561_156182


namespace NUMINAMATH_GPT_value_of_expression_l1561_156194

theorem value_of_expression 
  (triangle square : ℝ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1561_156194


namespace NUMINAMATH_GPT_product_modulo_25_l1561_156163

theorem product_modulo_25 : 
  (123 ≡ 3 [MOD 25]) → 
  (456 ≡ 6 [MOD 25]) → 
  (789 ≡ 14 [MOD 25]) → 
  (123 * 456 * 789 ≡ 2 [MOD 25]) := 
by 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_product_modulo_25_l1561_156163


namespace NUMINAMATH_GPT_reservoir_full_percentage_after_storm_l1561_156196

theorem reservoir_full_percentage_after_storm 
  (original_contents water_added : ℤ) 
  (percentage_full_before_storm: ℚ) 
  (total_capacity new_contents : ℚ) 
  (H1 : original_contents = 220 * 10^9) 
  (H2 : water_added = 110 * 10^9) 
  (H3 : percentage_full_before_storm = 0.40)
  (H4 : total_capacity = original_contents / percentage_full_before_storm)
  (H5 : new_contents = original_contents + water_added) :
  (new_contents / total_capacity) = 0.60 := 
by 
  sorry

end NUMINAMATH_GPT_reservoir_full_percentage_after_storm_l1561_156196


namespace NUMINAMATH_GPT_investment_amount_l1561_156177

theorem investment_amount (x y : ℝ) (hx : x ≤ 11000) (hy : 0.07 * x + 0.12 * y ≥ 2450) : x + y = 25000 := 
sorry

end NUMINAMATH_GPT_investment_amount_l1561_156177


namespace NUMINAMATH_GPT_relationship_between_D_and_A_l1561_156178

variables (A B C D : Prop)

def sufficient_not_necessary (P Q : Prop) : Prop := (P → Q) ∧ ¬ (Q → P)
def necessary_not_sufficient (P Q : Prop) : Prop := (Q → P) ∧ ¬ (P → Q)
def necessary_and_sufficient (P Q : Prop) : Prop := (P ↔ Q)

-- Conditions
axiom h1 : sufficient_not_necessary A B
axiom h2 : necessary_not_sufficient C B
axiom h3 : necessary_and_sufficient D C

-- Proof Goal
theorem relationship_between_D_and_A : necessary_not_sufficient D A :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_D_and_A_l1561_156178


namespace NUMINAMATH_GPT_real_life_distance_between_cities_l1561_156191

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end NUMINAMATH_GPT_real_life_distance_between_cities_l1561_156191


namespace NUMINAMATH_GPT_correct_calculation_l1561_156150

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1561_156150


namespace NUMINAMATH_GPT_sandy_initial_amount_l1561_156188

theorem sandy_initial_amount 
  (cost_shirt : ℝ) (cost_jacket : ℝ) (found_money : ℝ)
  (h1 : cost_shirt = 12.14) (h2 : cost_jacket = 9.28) (h3 : found_money = 7.43) : 
  (cost_shirt + cost_jacket + found_money = 28.85) :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_sandy_initial_amount_l1561_156188


namespace NUMINAMATH_GPT_ternary_1021_to_decimal_l1561_156100

-- Define the function to convert a ternary string to decimal
def ternary_to_decimal (n : String) : Nat :=
  n.foldr (fun c acc => acc * 3 + (c.toNat - '0'.toNat)) 0

-- The statement to prove
theorem ternary_1021_to_decimal : ternary_to_decimal "1021" = 34 := by
  sorry

end NUMINAMATH_GPT_ternary_1021_to_decimal_l1561_156100


namespace NUMINAMATH_GPT_minimum_d_exists_l1561_156168

open Nat

theorem minimum_d_exists :
  ∃ (a b c d e f g h i k : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ k ∧
                                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ k ∧
                                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ k ∧
                                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ k ∧
                                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ k ∧
                                f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ k ∧
                                g ≠ h ∧ g ≠ i ∧ g ≠ k ∧
                                h ≠ i ∧ h ≠ k ∧
                                i ≠ k ∧
                                d = a + 3 * (e + h) + k ∧
                                d = 20 :=
by
  sorry

end NUMINAMATH_GPT_minimum_d_exists_l1561_156168


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l1561_156102

theorem solve_quadratic_1 (x : Real) : x^2 - 2 * x - 4 = 0 ↔ (x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5) :=
by
  sorry

theorem solve_quadratic_2 (x : Real) : (x - 1)^2 = 2 * (x - 1) ↔ (x = 1 ∨ x = 3) :=
by
  sorry

theorem solve_quadratic_3 (x : Real) : (x + 1)^2 = 4 * x^2 ↔ (x = 1 ∨ x = -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l1561_156102


namespace NUMINAMATH_GPT_tournament_committees_l1561_156158

-- Assuming each team has 7 members
def team_members : Nat := 7

-- There are 5 teams
def total_teams : Nat := 5

-- The host team selects 3 members including at least one woman
def select_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 3
  let all_men_combinations := Nat.choose (team_members - 1) 3
  total_combinations - all_men_combinations

-- Each non-host team selects 2 members including at least one woman
def select_non_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 2
  let all_men_combinations := Nat.choose (team_members - 1) 2
  total_combinations - all_men_combinations

-- Total number of committees when one team is the host
def one_team_host_total_combinations (w m : Nat) : ℕ :=
  select_host_team_members w m * (select_non_host_team_members w m) ^ (total_teams - 1)

-- Total number of possible 11-member tournament committees
def total_committees (w m : Nat) : ℕ :=
  one_team_host_total_combinations w m * total_teams

theorem tournament_committees (w m : Nat) (hw : w ≥ 1) (hm : m ≤ 6) :
  total_committees w m = 97200 :=
by
  sorry

end NUMINAMATH_GPT_tournament_committees_l1561_156158


namespace NUMINAMATH_GPT_not_all_zero_implies_at_least_one_nonzero_l1561_156120

variable {a b c : ℤ}

theorem not_all_zero_implies_at_least_one_nonzero (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) : 
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 := 
by 
  sorry

end NUMINAMATH_GPT_not_all_zero_implies_at_least_one_nonzero_l1561_156120


namespace NUMINAMATH_GPT_loaned_books_l1561_156161

theorem loaned_books (initial_books : ℕ) (returned_percent : ℝ)
  (end_books : ℕ) (damaged_books : ℕ) (L : ℝ) :
  initial_books = 150 ∧
  returned_percent = 0.85 ∧
  end_books = 135 ∧
  damaged_books = 5 ∧
  0.85 * L + 5 + (initial_books - L) = end_books →
  L = 133 :=
by
  intros h
  rcases h with ⟨hb, hr, he, hd, hsum⟩
  repeat { sorry }

end NUMINAMATH_GPT_loaned_books_l1561_156161


namespace NUMINAMATH_GPT_trigonometric_inequality_for_tan_l1561_156131

open Real

theorem trigonometric_inequality_for_tan (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 
  1 + tan x < 1 / (1 - sin x) :=
sorry

end NUMINAMATH_GPT_trigonometric_inequality_for_tan_l1561_156131


namespace NUMINAMATH_GPT_sqrt_div_val_l1561_156127

theorem sqrt_div_val (n : ℕ) (h : n = 3600) : (Nat.sqrt n) / 15 = 4 := by 
  sorry

end NUMINAMATH_GPT_sqrt_div_val_l1561_156127


namespace NUMINAMATH_GPT_solve_for_a_b_c_d_l1561_156133

theorem solve_for_a_b_c_d :
  ∃ a b c d : ℕ, (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023 ∧ a^3 + b^3 + c^3 + d^3 = 43 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_a_b_c_d_l1561_156133


namespace NUMINAMATH_GPT_fraction_value_l1561_156170

theorem fraction_value (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l1561_156170


namespace NUMINAMATH_GPT_total_people_going_to_zoo_and_amusement_park_l1561_156165

theorem total_people_going_to_zoo_and_amusement_park :
  (7.0 * 45.0) + (5.0 * 56.0) = 595.0 :=
by
  sorry

end NUMINAMATH_GPT_total_people_going_to_zoo_and_amusement_park_l1561_156165


namespace NUMINAMATH_GPT_pollen_particle_diameter_in_scientific_notation_l1561_156154

theorem pollen_particle_diameter_in_scientific_notation :
  0.0000078 = 7.8 * 10^(-6) :=
by
  sorry

end NUMINAMATH_GPT_pollen_particle_diameter_in_scientific_notation_l1561_156154


namespace NUMINAMATH_GPT_f_has_two_zeros_l1561_156124

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_has_two_zeros (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := sorry

end NUMINAMATH_GPT_f_has_two_zeros_l1561_156124


namespace NUMINAMATH_GPT_common_root_exists_l1561_156112

theorem common_root_exists :
  ∃ x, (3 * x^4 + 13 * x^3 + 20 * x^2 + 17 * x + 7 = 0) ∧ (3 * x^4 + x^3 - 8 * x^2 + 11 * x - 7 = 0) → x = -7 / 3 := 
by
  sorry

end NUMINAMATH_GPT_common_root_exists_l1561_156112


namespace NUMINAMATH_GPT_percentage_of_men_attended_picnic_l1561_156169

variable (E : ℝ) (W M P : ℝ)
variable (H1 : M = 0.5 * E)
variable (H2 : W = 0.5 * E)
variable (H3 : 0.4 * W = 0.2 * E)
variable (H4 : 0.3 * E = P * M + 0.2 * E)

theorem percentage_of_men_attended_picnic : P = 0.2 :=
by sorry

end NUMINAMATH_GPT_percentage_of_men_attended_picnic_l1561_156169


namespace NUMINAMATH_GPT_relatively_prime_sequence_l1561_156135

theorem relatively_prime_sequence (k : ℤ) (hk : k > 1) :
  ∃ (a b : ℤ) (x : ℕ → ℤ),
    a > 0 ∧ b > 0 ∧
    (∀ n, x (n + 2) = x (n + 1) + x n) ∧
    x 0 = a ∧ x 1 = b ∧ ∀ n, gcd (x n) (4 * k^2 - 5) = 1 :=
by
  sorry

end NUMINAMATH_GPT_relatively_prime_sequence_l1561_156135


namespace NUMINAMATH_GPT_car_speed_l1561_156119

theorem car_speed (v : ℝ) (hv : (1 / v * 3600) = (1 / 40 * 3600) + 10) : v = 36 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_l1561_156119


namespace NUMINAMATH_GPT_fewest_printers_l1561_156105

/-!
# Fewest Printers Purchase Problem
Given two types of computer printers costing $350 and $200 per unit, respectively,
given that the company wants to spend equal amounts on both types of printers.
Prove that the fewest number of printers the company can purchase is 11.
-/

theorem fewest_printers (p1 p2 : ℕ) (h1 : p1 = 350) (h2 : p2 = 200) :
  ∃ n1 n2 : ℕ, p1 * n1 = p2 * n2 ∧ n1 + n2 = 11 := 
sorry

end NUMINAMATH_GPT_fewest_printers_l1561_156105


namespace NUMINAMATH_GPT_equation1_no_solution_equation2_solution_l1561_156113

/-- Prove that the equation (4-x)/(x-3) + 1/(3-x) = 1 has no solution. -/
theorem equation1_no_solution (x : ℝ) : x ≠ 3 → ¬ (4 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by intro hx; sorry

/-- Prove that the equation (x+1)/(x-1) - 6/(x^2-1) = 1 has solution x = 2. -/
theorem equation2_solution (x : ℝ) : x = 2 ↔ (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1 :=
by sorry

end NUMINAMATH_GPT_equation1_no_solution_equation2_solution_l1561_156113


namespace NUMINAMATH_GPT_increasing_function_range_a_l1561_156110

theorem increasing_function_range_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = if x > 1 then a^x else (4 - a/2)*x + 2) ∧
  (∀ x y, x < y → f x ≤ f y) →
  4 ≤ a ∧ a < 8 :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_range_a_l1561_156110


namespace NUMINAMATH_GPT_room_volume_l1561_156130

theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 :=
by sorry

end NUMINAMATH_GPT_room_volume_l1561_156130


namespace NUMINAMATH_GPT_min_convex_number_l1561_156111

noncomputable def minimum_convex_sets (A B C : ℝ × ℝ) : ℕ :=
  if A ≠ B ∧ B ≠ C ∧ C ≠ A then 3 else 4

theorem min_convex_number (A B C : ℝ × ℝ) (h : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  minimum_convex_sets A B C = 3 :=
by 
  sorry

end NUMINAMATH_GPT_min_convex_number_l1561_156111


namespace NUMINAMATH_GPT_ratio_of_volumes_l1561_156128
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1561_156128


namespace NUMINAMATH_GPT_gratuity_calculation_correct_l1561_156139

noncomputable def tax_rate (item: String): ℝ :=
  if item = "NY Striploin" then 0.10
  else if item = "Glass of wine" then 0.15
  else if item = "Dessert" then 0.05
  else if item = "Bottle of water" then 0.00
  else 0

noncomputable def base_price (item: String): ℝ :=
  if item = "NY Striploin" then 80
  else if item = "Glass of wine" then 10
  else if item = "Dessert" then 12
  else if item = "Bottle of water" then 3
  else 0

noncomputable def total_price_with_tax (item: String): ℝ :=
  base_price item + base_price item * tax_rate item

noncomputable def gratuity (item: String): ℝ :=
  total_price_with_tax item * 0.20

noncomputable def total_gratuity: ℝ :=
  gratuity "NY Striploin" + gratuity "Glass of wine" + gratuity "Dessert" + gratuity "Bottle of water"

theorem gratuity_calculation_correct :
  total_gratuity = 23.02 :=
by
  sorry

end NUMINAMATH_GPT_gratuity_calculation_correct_l1561_156139


namespace NUMINAMATH_GPT_one_third_of_flour_l1561_156199

-- Definition of the problem conditions
def initial_flour : ℚ := 5 + 2 / 3
def portion : ℚ := 1 / 3

-- Definition of the theorem to prove
theorem one_third_of_flour : portion * initial_flour = 1 + 8 / 9 :=
by {
  -- Placeholder proof
  sorry
}

end NUMINAMATH_GPT_one_third_of_flour_l1561_156199


namespace NUMINAMATH_GPT_range_of_m_for_line_to_intersect_ellipse_twice_l1561_156138

theorem range_of_m_for_line_to_intersect_ellipse_twice (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.2 = 4 * A.1 + m) ∧
   (B.2 = 4 * B.1 + m) ∧
   ((A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1) ∧
   ((B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1) ∧
   (A.1 + B.1) / 2 = 0 ∧ 
   (A.2 + B.2) / 2 = 4 * 0 + m) ↔
   - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13
 :=
sorry

end NUMINAMATH_GPT_range_of_m_for_line_to_intersect_ellipse_twice_l1561_156138


namespace NUMINAMATH_GPT_y_intercept_of_line_l1561_156183

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1561_156183


namespace NUMINAMATH_GPT_candy_bar_calories_unit_l1561_156162

-- Definitions based on conditions
def calories_unit := "calories per candy bar"

-- There are 4 units of calories in a candy bar
def units_per_candy_bar : ℕ := 4

-- There are 2016 calories in 42 candy bars
def total_calories : ℕ := 2016
def number_of_candy_bars : ℕ := 42

-- The statement to prove
theorem candy_bar_calories_unit : (total_calories / number_of_candy_bars = 48) → calories_unit = "calories per candy bar" :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_calories_unit_l1561_156162


namespace NUMINAMATH_GPT_shaded_area_in_octagon_l1561_156108

theorem shaded_area_in_octagon (s r : ℝ) (h_s : s = 4) (h_r : r = s / 2) :
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_semicircles := 8 * (π * r^2 / 2)
  area_octagon - area_semicircles = 32 * (1 + Real.sqrt 2) - 16 * π := by
  sorry

end NUMINAMATH_GPT_shaded_area_in_octagon_l1561_156108


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1561_156123

-- The conditions for quadrilateral ABCD
variables (a b c d e f m n S : ℝ)
variables (S_nonneg : 0 ≤ S)

-- Prove Part (a)
theorem part_a (a b c d e f : ℝ) (S : ℝ) (h : S ≤ 1/4 * (e^2 + f^2)) : S <= 1/4 * (e^2 + f^2) :=
by 
  exact h

-- Prove Part (b)
theorem part_b (a b c d e f m n S: ℝ) (h : S ≤ 1/2 * (m^2 + n^2)) : S <= 1/2 * (m^2 + n^2) :=
by 
  exact h

-- Prove Part (c)
theorem part_c (a b c d e f m n S: ℝ) (h : S ≤ 1/4 * (a + c) * (b + d)) : S <= 1/4 * (a + c) * (b + d) :=
by 
  exact h

#eval "This Lean code defines the correctness statement of each part of the problem."

end NUMINAMATH_GPT_part_a_part_b_part_c_l1561_156123


namespace NUMINAMATH_GPT_ratio_b_a_l1561_156186

theorem ratio_b_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a ≠ b) (h4 : a + b > 2 * a) (h5 : 2 * a > a) 
  (h6 : a + b > b) (h7 : a + 2 * a = b) : 
  b = a * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_b_a_l1561_156186


namespace NUMINAMATH_GPT_exists_infinite_subset_with_gcd_l1561_156190

/-- A set of natural numbers where each number is a product of at most 1987 primes -/
def is_bounded_product_set (A : Set ℕ) (k : ℕ) : Prop :=
  ∀ a ∈ A, ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ a = S.prod id ∧ S.card ≤ k

/-- Prove the existence of an infinite subset and a common gcd for any pair of its elements -/
theorem exists_infinite_subset_with_gcd (A : Set ℕ) (k : ℕ) (hk : k = 1987)
  (hA : is_bounded_product_set A k) (h_inf : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Subset B A ∧ Set.Infinite B ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b := 
sorry

end NUMINAMATH_GPT_exists_infinite_subset_with_gcd_l1561_156190


namespace NUMINAMATH_GPT_kate_candy_l1561_156132

variable (K : ℕ)
variable (R : ℕ) (B : ℕ) (M : ℕ)

-- Define the conditions
def robert_pieces := R = K + 2
def mary_pieces := M = R + 2
def bill_pieces := B = M - 6
def total_pieces := K + R + M + B = 20

-- The theorem to prove
theorem kate_candy :
  ∃ (K : ℕ), robert_pieces K R ∧ mary_pieces R M ∧ bill_pieces M B ∧ total_pieces K R M B ∧ K = 4 :=
sorry

end NUMINAMATH_GPT_kate_candy_l1561_156132


namespace NUMINAMATH_GPT_locus_of_P_is_ellipse_l1561_156104

-- Definitions and conditions
def circle_A (x y : ℝ) : Prop := (x + 3) ^ 2 + y ^ 2 = 100
def fixed_point_B : ℝ × ℝ := (3, 0)
def circle_P_passes_through_B (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 3) ^ 2 + center.2 ^ 2 = radius ^ 2
def circle_P_tangent_to_A_internally (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 + 3) ^ 2 + center.2 ^ 2 = (10 - radius) ^ 2

-- Statement of the problem to prove in Lean
theorem locus_of_P_is_ellipse :
  ∃ (foci_A B : ℝ × ℝ) (a b : ℝ), (foci_A = (-3, 0)) ∧ (foci_B = (3, 0)) ∧ (a = 5) ∧ (b = 4) ∧ 
  (∀ (x y : ℝ), (∃ (P : ℝ × ℝ) (radius : ℝ), circle_P_passes_through_B P radius ∧ circle_P_tangent_to_A_internally P radius ∧ P = (x, y)) ↔ 
  (x ^ 2) / 25 + (y ^ 2) / 16 = 1)
:=
sorry

end NUMINAMATH_GPT_locus_of_P_is_ellipse_l1561_156104


namespace NUMINAMATH_GPT_ladder_wood_sufficiency_l1561_156121

theorem ladder_wood_sufficiency
  (total_wood : ℝ)
  (rung_length_in: ℝ)
  (rung_distance_in: ℝ)
  (ladder_height_ft: ℝ)
  (total_wood_ft : total_wood = 300)
  (rung_length_ft : rung_length_in = 18 / 12)
  (rung_distance_ft : rung_distance_in = 6 / 12)
  (ladder_height_ft : ladder_height_ft = 50) :
  (∃ wood_needed : ℝ, wood_needed ≤ total_wood ∧ total_wood - wood_needed = 162.5) :=
sorry

end NUMINAMATH_GPT_ladder_wood_sufficiency_l1561_156121


namespace NUMINAMATH_GPT_sector_area_eq_4cm2_l1561_156126

variable (α : ℝ) (l : ℝ) (R : ℝ)
variable (h_alpha : α = 2) (h_l : l = 4) (h_R : R = l / α)

theorem sector_area_eq_4cm2
    (h_alpha : α = 2)
    (h_l : l = 4)
    (h_R : R = l / α) :
    (1/2 * l * R) = 4 := by
  sorry

end NUMINAMATH_GPT_sector_area_eq_4cm2_l1561_156126


namespace NUMINAMATH_GPT_million_to_scientific_notation_l1561_156109

theorem million_to_scientific_notation (population_henan : ℝ) (h : population_henan = 98.83 * 10^6) :
  population_henan = 9.883 * 10^7 :=
by sorry

end NUMINAMATH_GPT_million_to_scientific_notation_l1561_156109


namespace NUMINAMATH_GPT_final_withdrawal_amount_july_2005_l1561_156187

-- Define the conditions given in the problem
variables (a r : ℝ) (n : ℕ)

-- Define the recursive formula for deposits
def deposit_amount (n : ℕ) : ℝ :=
  if n = 0 then a else (deposit_amount (n - 1)) * (1 + r) + a

-- The problem statement translated to Lean
theorem final_withdrawal_amount_july_2005 :
  deposit_amount a r 5 = a / r * ((1 + r) ^ 6 - (1 + r)) :=
sorry

end NUMINAMATH_GPT_final_withdrawal_amount_july_2005_l1561_156187


namespace NUMINAMATH_GPT_second_group_persons_l1561_156151

open Nat

theorem second_group_persons
  (P : ℕ)
  (work_first_group : 39 * 24 * 5 = 4680)
  (work_second_group : P * 26 * 6 = 4680) :
  P = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_group_persons_l1561_156151


namespace NUMINAMATH_GPT_McKenna_stuffed_animals_count_l1561_156174

def stuffed_animals (M K T : ℕ) : Prop :=
  M + K + T = 175 ∧ K = 2 * M ∧ T = K + 5

theorem McKenna_stuffed_animals_count (M K T : ℕ) (h : stuffed_animals M K T) : M = 34 :=
by
  sorry

end NUMINAMATH_GPT_McKenna_stuffed_animals_count_l1561_156174


namespace NUMINAMATH_GPT_simplify_fraction_l1561_156185

variable (y b : ℚ)

theorem simplify_fraction : 
  (y+2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1561_156185


namespace NUMINAMATH_GPT_perpendicular_line_plane_implies_perpendicular_lines_l1561_156101

variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Assume inclusion of lines in planes, parallelism, and perpendicularity properties.
variables (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (subset : Line → Plane → Prop) (perpendicular_lines : Line → Line → Prop)

-- Given definitions based on the conditions
variable (is_perpendicular : perpendicular m α)
variable (is_subset : subset n α)

-- Prove that m is perpendicular to n
theorem perpendicular_line_plane_implies_perpendicular_lines
  (h1 : perpendicular m α)
  (h2 : subset n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_GPT_perpendicular_line_plane_implies_perpendicular_lines_l1561_156101


namespace NUMINAMATH_GPT_expected_number_of_returns_l1561_156171

noncomputable def expected_returns_to_zero : ℝ :=
  let p_move := 1 / 3
  let expected_value := -1 + (3 / (Real.sqrt 5))
  expected_value

theorem expected_number_of_returns : expected_returns_to_zero = (3 * Real.sqrt 5 - 5) / 5 :=
  by sorry

end NUMINAMATH_GPT_expected_number_of_returns_l1561_156171


namespace NUMINAMATH_GPT_find_y_l1561_156179

theorem find_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hrem : x % y = 11.52) (hdiv : x / y = 96.12) : y = 96 := 
sorry

end NUMINAMATH_GPT_find_y_l1561_156179


namespace NUMINAMATH_GPT_volume_of_cone_formed_by_sector_l1561_156157

theorem volume_of_cone_formed_by_sector :
  let radius := 6
  let sector_fraction := (5:ℝ) / 6
  let circumference := 2 * Real.pi * radius
  let cone_base_circumference := sector_fraction * circumference
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let slant_height := radius
  let cone_height := Real.sqrt (slant_height^2 - cone_base_radius^2)
  let volume := (1:ℝ) / 3 * Real.pi * (cone_base_radius^2) * cone_height
  volume = 25 / 3 * Real.pi * Real.sqrt 11 :=
by sorry

end NUMINAMATH_GPT_volume_of_cone_formed_by_sector_l1561_156157


namespace NUMINAMATH_GPT_find_integer_solutions_l1561_156140

theorem find_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + 1 / (x: ℝ)) * (1 + 1 / (y: ℝ)) * (1 + 1 / (z: ℝ)) = 2 ↔ (x = 2 ∧ y = 4 ∧ z = 15) ∨ (x = 2 ∧ y = 5 ∧ z = 9) ∨ (x = 2 ∧ y = 6 ∧ z = 7) ∨ (x = 3 ∧ y = 3 ∧ z = 8) ∨ (x = 3 ∧ y = 4 ∧ z = 5) := sorry

end NUMINAMATH_GPT_find_integer_solutions_l1561_156140


namespace NUMINAMATH_GPT_trains_crossing_time_l1561_156173

theorem trains_crossing_time :
  let length_of_each_train := 120 -- in meters
  let speed_of_each_train := 12 -- in km/hr
  let total_distance := length_of_each_train * 2
  let relative_speed := (speed_of_each_train * 1000 / 3600 * 2) -- in m/s
  total_distance / relative_speed = 36 := 
by
  -- Since we only need to state the theorem, the proof is omitted.
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l1561_156173


namespace NUMINAMATH_GPT_sixth_root_binomial_expansion_l1561_156136

theorem sixth_root_binomial_expansion :
  (2748779069441 = 1 * 150^6 + 6 * 150^5 + 15 * 150^4 + 20 * 150^3 + 15 * 150^2 + 6 * 150 + 1) →
  (2748779069441 = Nat.choose 6 6 * 150^6 + Nat.choose 6 5 * 150^5 + Nat.choose 6 4 * 150^4 + Nat.choose 6 3 * 150^3 + Nat.choose 6 2 * 150^2 + Nat.choose 6 1 * 150 + Nat.choose 6 0) →
  (Real.sqrt (2748779069441 : ℝ) = 151) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sixth_root_binomial_expansion_l1561_156136


namespace NUMINAMATH_GPT_knight_min_moves_l1561_156149

theorem knight_min_moves (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, k = 2 * (Nat.floor ((n + 1 : ℚ) / 3)) ∧
  (∀ m, (3 * m) ≥ (2 * (n - 1)) → ∃ l, l = 2 * m ∧ l ≥ k) :=
by
  sorry

end NUMINAMATH_GPT_knight_min_moves_l1561_156149


namespace NUMINAMATH_GPT_simplify_and_rationalize_l1561_156195

theorem simplify_and_rationalize : (1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l1561_156195


namespace NUMINAMATH_GPT_diamonds_in_G_10_l1561_156175

-- Define the sequence rule for diamonds in Gn
def diamonds_in_G (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

-- The main theorem to prove that the number of diamonds in G₁₀ is 218
theorem diamonds_in_G_10 : diamonds_in_G 10 = 218 := by
  sorry

end NUMINAMATH_GPT_diamonds_in_G_10_l1561_156175


namespace NUMINAMATH_GPT_baking_powder_now_l1561_156198

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_used : ℝ := 0.1

theorem baking_powder_now : 
  baking_powder_yesterday - baking_powder_used = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_baking_powder_now_l1561_156198


namespace NUMINAMATH_GPT_practice_minutes_l1561_156176

def month_total_days : ℕ := (2 * 6) + (2 * 7)

def piano_daily_minutes : ℕ := 25

def violin_daily_minutes := piano_daily_minutes * 3

def flute_daily_minutes := violin_daily_minutes / 2

theorem practice_minutes (piano_total : ℕ) (violin_total : ℕ) (flute_total : ℕ) :
  (26 * piano_daily_minutes = 650) ∧ 
  (20 * violin_daily_minutes = 1500) ∧ 
  (16 * flute_daily_minutes = 600) := by
  sorry

end NUMINAMATH_GPT_practice_minutes_l1561_156176


namespace NUMINAMATH_GPT_sara_caught_five_trout_l1561_156106

theorem sara_caught_five_trout (S M : ℕ) (h1 : M = 2 * S) (h2 : M = 10) : S = 5 :=
by
  sorry

end NUMINAMATH_GPT_sara_caught_five_trout_l1561_156106


namespace NUMINAMATH_GPT_smallest_base10_num_exists_l1561_156146

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end NUMINAMATH_GPT_smallest_base10_num_exists_l1561_156146


namespace NUMINAMATH_GPT_angle_terminal_side_eq_l1561_156167

noncomputable def has_same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_eq (k : ℤ) :
  has_same_terminal_side (- (Real.pi / 3)) (5 * Real.pi / 3) :=
by
  use 1
  sorry

end NUMINAMATH_GPT_angle_terminal_side_eq_l1561_156167


namespace NUMINAMATH_GPT_total_flour_l1561_156153

def cups_of_flour (flour_added : ℕ) (flour_needed : ℕ) : ℕ :=
  flour_added + flour_needed

theorem total_flour :
  ∀ (flour_added flour_needed : ℕ), flour_added = 3 → flour_needed = 6 → cups_of_flour flour_added flour_needed = 9 :=
by 
  intros flour_added flour_needed h_added h_needed
  rw [h_added, h_needed]
  rfl

end NUMINAMATH_GPT_total_flour_l1561_156153


namespace NUMINAMATH_GPT_range_of_a_l1561_156148

noncomputable def my_function (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (x + 1) ^ 2

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ → my_function a x₁ - my_function a x₂ ≥ 4 * (x₁ - x₂)) → a ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1561_156148


namespace NUMINAMATH_GPT_abs_function_le_two_l1561_156197

theorem abs_function_le_two {x : ℝ} (h : |x| ≤ 2) : |3 * x - x^3| ≤ 2 :=
sorry

end NUMINAMATH_GPT_abs_function_le_two_l1561_156197


namespace NUMINAMATH_GPT_exponentiation_product_l1561_156137

theorem exponentiation_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3 ^ a) ^ b = 3 ^ 3) : 3 ^ a * 3 ^ b = 3 ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_product_l1561_156137


namespace NUMINAMATH_GPT_no_solution_inequality_l1561_156107

theorem no_solution_inequality (m x : ℝ) (h1 : x - 2 * m < 0) (h2 : x + m > 2) : m ≤ 2 / 3 :=
  sorry

end NUMINAMATH_GPT_no_solution_inequality_l1561_156107


namespace NUMINAMATH_GPT_curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l1561_156166

-- Define the equation of the curve C
def curve_C (a x y : ℝ) : Prop := a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

-- Prove that curve C is a circle
theorem curve_C_is_circle (a : ℝ) (h : a ≠ 0) :
  ∃ (h_c : ℝ), ∃ (k : ℝ), ∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), curve_C a x y ↔ (x - h_c)^2 + (y - k)^2 = r^2
:= sorry

-- Prove that the area of triangle AOB is constant
theorem area_AOB_constant (a : ℝ) (h : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), (A = (2 * a, 0) ∧ B = (0, 4 / a)) ∧ 1/2 * (2 * a) * (4 / a) = 4
:= sorry

-- Find valid a and equation of curve C given conditions of line l and points M, N
theorem find_valid_a_and_curve_eq (a : ℝ) (h : a ≠ 0) :
  ∀ (M N : ℝ × ℝ), (|M.1 - 0| = |N.1 - 0| ∧ |M.2 - 0| = |N.2 - 0|) → (M.1 = N.1 ∧ M.2 = N.2) →
  y = -2 * x + 4 →  a = 2 ∧ ∀ (x y : ℝ), curve_C 2 x y ↔ x^2 + y^2 - 4 * x - 2 * y = 0
:= sorry

end NUMINAMATH_GPT_curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l1561_156166


namespace NUMINAMATH_GPT_high_card_point_value_l1561_156184

theorem high_card_point_value :
  ∀ (H L : ℕ), 
  (L = 1) →
  ∀ (high low total_points : ℕ), 
  (total_points = 5) →
  (high + (L + L + L) = total_points) →
  high = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_high_card_point_value_l1561_156184


namespace NUMINAMATH_GPT_symmetrical_character_l1561_156181

def symmetrical (char : String) : Prop :=
  -- Define a predicate symmetrical which checks if a given character
  -- is a symmetrical figure somehow. This needs to be implemented
  -- properly based on the graphical property of the character.
  sorry 

theorem symmetrical_character :
  ∀ (c : String), (c = "幸" → symmetrical c) ∧ 
                  (c = "福" → ¬ symmetrical c) ∧ 
                  (c = "惠" → ¬ symmetrical c) ∧ 
                  (c = "州" → ¬ symmetrical c) :=
by
  sorry

end NUMINAMATH_GPT_symmetrical_character_l1561_156181


namespace NUMINAMATH_GPT_students_prob_red_light_l1561_156155

noncomputable def probability_red_light_encountered (p1 p2 p3 : ℚ) : ℚ :=
  1 - ((1 - p1) * (1 - p2) * (1 - p3))

theorem students_prob_red_light :
  probability_red_light_encountered (1/2) (1/3) (1/4) = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_students_prob_red_light_l1561_156155


namespace NUMINAMATH_GPT_part_a_part_b_l1561_156142

variable (a b c : ℤ)
variable (h : a + b + c = 0)

theorem part_a : (a^4 + b^4 + c^4) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

theorem part_b : (a^100 + b^100 + c^100) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1561_156142


namespace NUMINAMATH_GPT_consecutive_numbers_expression_l1561_156172

theorem consecutive_numbers_expression (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y - 1) (h3 : z = 2) :
  2 * x + 3 * y + 3 * z = 8 * y - 1 :=
by
  -- substitute the conditions and simplify
  sorry

end NUMINAMATH_GPT_consecutive_numbers_expression_l1561_156172


namespace NUMINAMATH_GPT_min_value_expression_l1561_156117

theorem min_value_expression : ∃ (x y : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 ≥ 0) ∧ (∀ (x y : ℝ), x = 4 ∧ y = -3 → x^2 + y^2 - 8*x + 6*y + 25 = 0) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1561_156117


namespace NUMINAMATH_GPT_inequality_abc_l1561_156129

theorem inequality_abc
  (a b c : ℝ)
  (ha : 0 ≤ a) (ha_le : a ≤ 1)
  (hb : 0 ≤ b) (hb_le : b ≤ 1)
  (hc : 0 ≤ c) (hc_le : c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_abc_l1561_156129


namespace NUMINAMATH_GPT_gcd_factorials_l1561_156164

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_gcd_factorials_l1561_156164


namespace NUMINAMATH_GPT_value_set_of_t_l1561_156141

theorem value_set_of_t (t : ℝ) :
  (1 > 2 * (1) + 1 - t) ∧ (∀ x : ℝ, x^2 + (2*t-4)*x + 4 > 0) → 3 < t ∧ t < 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_value_set_of_t_l1561_156141


namespace NUMINAMATH_GPT_calc_expression_l1561_156192

theorem calc_expression (x y z : ℚ) (h1 : x = 1 / 3) (h2 : y = 2 / 3) (h3 : z = x * y) :
  3 * x^2 * y^5 * z^3 = 768 / 1594323 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1561_156192


namespace NUMINAMATH_GPT_alligators_not_hiding_l1561_156115

-- Definitions derived from conditions
def total_alligators : ℕ := 75
def hiding_alligators : ℕ := 19

-- Theorem statement matching the mathematically equivalent proof problem.
theorem alligators_not_hiding : (total_alligators - hiding_alligators) = 56 := by
  -- Sorry skips the proof. Replace with actual proof if required.
  sorry

end NUMINAMATH_GPT_alligators_not_hiding_l1561_156115


namespace NUMINAMATH_GPT_find_AC_l1561_156193

noncomputable def isTriangle (A B C : Type) : Type := sorry

-- Define angles and lengths.
variables (A B C : Type)
variables (angle_A angle_B : ℝ)
variables (BC AC : ℝ)

-- Assume the given conditions.
axiom angle_A_60 : angle_A = 60 * Real.pi / 180
axiom angle_B_45 : angle_B = 45 * Real.pi / 180
axiom BC_12 : BC = 12

-- Statement to prove.
theorem find_AC 
  (h_triangle : isTriangle A B C)
  (h_angle_A : angle_A = 60 * Real.pi / 180)
  (h_angle_B : angle_B = 45 * Real.pi / 180)
  (h_BC : BC = 12) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_find_AC_l1561_156193


namespace NUMINAMATH_GPT_cube_surface_area_l1561_156189

theorem cube_surface_area (V : ℝ) (hV : V = 64) : ∃ S : ℝ, S = 96 := 
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1561_156189


namespace NUMINAMATH_GPT_brian_final_cards_l1561_156160

-- Definitions of initial conditions
def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

-- The proof problem: Prove that the final number of cards is 62
theorem brian_final_cards : initial_cards - cards_taken + packs_bought * cards_per_pack = 62 :=
by
  -- Proof goes here, 'sorry' used to skip actual proof
  sorry

end NUMINAMATH_GPT_brian_final_cards_l1561_156160


namespace NUMINAMATH_GPT_solution_of_az_eq_b_l1561_156180

theorem solution_of_az_eq_b (a b z x y : ℝ) :
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬ ∃ y, 2 + y = (b + 1) * y) →
  az = b →
  z = 0 :=
by
  intros h1 h2 h3
  -- proof starts here
  sorry

end NUMINAMATH_GPT_solution_of_az_eq_b_l1561_156180


namespace NUMINAMATH_GPT_solve_for_x_l1561_156122

theorem solve_for_x (x : ℝ) (h : 1 / 4 - 1 / 6 = 4 / x) : x = 48 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l1561_156122


namespace NUMINAMATH_GPT_min_total_penalty_l1561_156144

noncomputable def min_penalty (B W R : ℕ) : ℕ :=
  min (B * W) (min (2 * W * R) (3 * R * B))

theorem min_total_penalty (B W R : ℕ) :
  min_penalty B W R = min (B * W) (min (2 * W * R) (3 * R * B)) := by
  sorry

end NUMINAMATH_GPT_min_total_penalty_l1561_156144
