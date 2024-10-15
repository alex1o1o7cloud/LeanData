import Mathlib

namespace NUMINAMATH_GPT_jack_jill_total_difference_l748_74883

theorem jack_jill_total_difference :
  let original_price := 90.00
  let discount_rate := 0.20
  let tax_rate := 0.06

  -- Jack's calculation
  let jack_total :=
    let price_with_tax := original_price * (1 + tax_rate)
    price_with_tax * (1 - discount_rate)
  
  -- Jill's calculation
  let jill_total :=
    let discounted_price := original_price * (1 - discount_rate)
    discounted_price * (1 + tax_rate)

  -- Equality check
  jack_total = jill_total := 
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_jack_jill_total_difference_l748_74883


namespace NUMINAMATH_GPT_baker_new_cakes_l748_74863

theorem baker_new_cakes :
  ∀ (initial_bought new_bought sold final : ℕ),
  initial_bought = 173 →
  sold = 86 →
  final = 190 →
  final = initial_bought + new_bought - sold →
  new_bought = 103 :=
by
  intros initial_bought new_bought sold final H_initial H_sold H_final H_eq
  sorry

end NUMINAMATH_GPT_baker_new_cakes_l748_74863


namespace NUMINAMATH_GPT_xy_condition_l748_74864

theorem xy_condition (x y z : ℝ) (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (posx : 0 < x) (posy : 0 < y) (posz : 0 < z) 
  (h : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_xy_condition_l748_74864


namespace NUMINAMATH_GPT_inequality_f_lt_g_range_of_a_l748_74862

def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

theorem inequality_f_lt_g :
  ∀ x : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (f x < g x ↔ (x < -5 ∨ x > 1)) :=
by
   sorry

theorem range_of_a :
  ∀ x a : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (2 * f x + g x > a * x) →
  (-4 ≤ a ∧ a < 9/4) :=
by
   sorry

end NUMINAMATH_GPT_inequality_f_lt_g_range_of_a_l748_74862


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l748_74886

theorem number_of_ordered_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ x ∈ S, (x.1 * x.2 = 64) ∧ (x.1 > 0) ∧ (x.2 > 0)) ∧ S.card = 7 := 
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l748_74886


namespace NUMINAMATH_GPT_sphere_surface_area_l748_74882

theorem sphere_surface_area (R r : ℝ) (h1 : 2 * OM = R) (h2 : ∀ r, π * r^2 = 3 * π) : 4 * π * R^2 = 16 * π :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l748_74882


namespace NUMINAMATH_GPT_slope_problem_l748_74801

theorem slope_problem (m : ℝ) (h₀ : m > 0) (h₁ : (3 - m) = m * (1 - m)) : m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_slope_problem_l748_74801


namespace NUMINAMATH_GPT_lcm_gcf_ratio_240_630_l748_74835

theorem lcm_gcf_ratio_240_630 :
  let a := 240
  let b := 630
  Nat.lcm a b / Nat.gcd a b = 168 := by
  sorry

end NUMINAMATH_GPT_lcm_gcf_ratio_240_630_l748_74835


namespace NUMINAMATH_GPT_line_equation_l748_74828

theorem line_equation (x y : ℝ) (hx : ∃ t : ℝ, t ≠ 0 ∧ x = t * -3) (hy : ∃ t : ℝ, t ≠ 0 ∧ y = t * 4) :
  4 * x - 3 * y + 12 = 0 := 
sorry

end NUMINAMATH_GPT_line_equation_l748_74828


namespace NUMINAMATH_GPT_boys_and_girls_are_equal_l748_74826

theorem boys_and_girls_are_equal (B G : ℕ) (h1 : B + G = 30)
    (h2 : ∀ b₁ b₂, b₁ ≠ b₂ → (0 ≤ b₁) ∧ (b₁ ≤ G - 1) → (0 ≤ b₂) ∧ (b₂ ≤ G - 1) → b₁ ≠ b₂)
    (h3 : ∀ g₁ g₂, g₁ ≠ g₂ → (0 ≤ g₁) ∧ (g₁ ≤ B - 1) → (0 ≤ g₂) ∧ (g₂ ≤ B - 1) → g₁ ≠ g₂) : 
    B = 15 ∧ G = 15 := by
  sorry

end NUMINAMATH_GPT_boys_and_girls_are_equal_l748_74826


namespace NUMINAMATH_GPT_orcs_per_squad_is_eight_l748_74816

-- Defining the conditions
def total_weight_of_swords := 1200
def weight_each_orc_can_carry := 15
def number_of_squads := 10

-- Proof statement to demonstrate the answer
theorem orcs_per_squad_is_eight :
  (total_weight_of_swords / weight_each_orc_can_carry) / number_of_squads = 8 := by
  sorry

end NUMINAMATH_GPT_orcs_per_squad_is_eight_l748_74816


namespace NUMINAMATH_GPT_find_x1_l748_74881

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) 
  (h2 : x4 ≤ x3) 
  (h3 : x3 ≤ x2) 
  (h4 : x2 ≤ x1) 
  (h5 : x1 ≤ 1) 
  (condition : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : 
  x1 = 4 / 5 := 
sorry

end NUMINAMATH_GPT_find_x1_l748_74881


namespace NUMINAMATH_GPT_largest_number_is_A_l748_74871

def numA : ℝ := 0.989
def numB : ℝ := 0.9879
def numC : ℝ := 0.98809
def numD : ℝ := 0.9807
def numE : ℝ := 0.9819

theorem largest_number_is_A :
  (numA > numB) ∧ (numA > numC) ∧ (numA > numD) ∧ (numA > numE) :=
by sorry

end NUMINAMATH_GPT_largest_number_is_A_l748_74871


namespace NUMINAMATH_GPT_relationship_between_abc_l748_74805

noncomputable def a : ℝ := Real.exp 0.9 + 1
def b : ℝ := 2.9
noncomputable def c : ℝ := Real.log (0.9 * Real.exp 3)

theorem relationship_between_abc : a > b ∧ b > c :=
by {
  sorry
}

end NUMINAMATH_GPT_relationship_between_abc_l748_74805


namespace NUMINAMATH_GPT_total_marks_more_than_physics_l748_74850

variable (P C M : ℕ)

theorem total_marks_more_than_physics :
  (P + C + M > P) ∧ ((C + M) / 2 = 75) → (P + C + M) - P = 150 := by
  intros h
  sorry

end NUMINAMATH_GPT_total_marks_more_than_physics_l748_74850


namespace NUMINAMATH_GPT_vector_sum_is_correct_l748_74827

-- Definitions for vectors a and b
def vector_a := (1, -2)
def vector_b (m : ℝ) := (2, m)

-- Condition for parallel vectors a and b
def parallel_vectors (m : ℝ) : Prop :=
  1 * m - (-2) * 2 = 0

-- Defining the target calculation for given m
def calculate_sum (m : ℝ) : ℝ × ℝ :=
  let a := vector_a
  let b := vector_b m
  (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)

-- Statement of the theorem to be proved
theorem vector_sum_is_correct (m : ℝ) (h : parallel_vectors m) : calculate_sum m = (7, -14) :=
by sorry

end NUMINAMATH_GPT_vector_sum_is_correct_l748_74827


namespace NUMINAMATH_GPT_problem_solution_l748_74839

theorem problem_solution :
  let m := 9
  let n := 20
  let lhs := (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)
  let rhs := 9 / 20
  lhs = rhs → 10 * m + n = 110 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l748_74839


namespace NUMINAMATH_GPT_gina_good_tipper_l748_74806

noncomputable def calculate_tip_difference (bill_in_usd : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (low_tip_rate : ℝ) (high_tip_rate : ℝ) (conversion_rate : ℝ) : ℝ :=
  let discounted_bill := bill_in_usd * (1 - discount_rate)
  let taxed_bill := discounted_bill * (1 + tax_rate)
  let low_tip := taxed_bill * low_tip_rate
  let high_tip := taxed_bill * high_tip_rate
  let difference_in_usd := high_tip - low_tip
  let difference_in_eur := difference_in_usd * conversion_rate
  difference_in_eur * 100

theorem gina_good_tipper : calculate_tip_difference 26 0.08 0.07 0.05 0.20 0.85 = 326.33 := 
by
  sorry

end NUMINAMATH_GPT_gina_good_tipper_l748_74806


namespace NUMINAMATH_GPT_M_inter_N_eq_singleton_l748_74887

def M (x y : ℝ) : Prop := x + y = 2
def N (x y : ℝ) : Prop := x - y = 4

theorem M_inter_N_eq_singleton :
  {p : ℝ × ℝ | M p.1 p.2} ∩ {p : ℝ × ℝ | N p.1 p.2} = { (3, -1) } :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_singleton_l748_74887


namespace NUMINAMATH_GPT_digits_divisibility_property_l748_74868

-- Definition: Example function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- Theorem: Prove the correctness of the given mathematical problem
theorem digits_divisibility_property:
  ∀ n : ℕ, (n = 18 ∨ n = 27 ∨ n = 45 ∨ n = 63) →
  (sum_of_digits n % 9 = 0) → (n % 9 = 0) := by
  sorry

end NUMINAMATH_GPT_digits_divisibility_property_l748_74868


namespace NUMINAMATH_GPT_cos_330_eq_sqrt3_over_2_l748_74859

theorem cos_330_eq_sqrt3_over_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_330_eq_sqrt3_over_2_l748_74859


namespace NUMINAMATH_GPT_total_students_in_class_l748_74810

theorem total_students_in_class (front_pos back_pos : ℕ) (H_front : front_pos = 23) (H_back : back_pos = 23) : front_pos + back_pos - 1 = 45 :=
by
  -- No proof required as per instructions
  sorry

end NUMINAMATH_GPT_total_students_in_class_l748_74810


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l748_74855

def is_isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 10) :
∃ c : ℝ, is_isosceles_triangle a b c ∧ perimeter a b c = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l748_74855


namespace NUMINAMATH_GPT_initial_amount_l748_74867

-- Define the conditions
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def num_small_glasses : ℕ := 8
def num_large_glasses : ℕ := 5
def change_left : ℕ := 1

-- Define the pieces based on conditions
def total_cost_small_glasses : ℕ := num_small_glasses * cost_small_glass
def total_cost_large_glasses : ℕ := num_large_glasses * cost_large_glass
def total_cost_glasses : ℕ := total_cost_small_glasses + total_cost_large_glasses

-- The theorem we need to prove
theorem initial_amount (h1 : total_cost_small_glasses = 24)
                       (h2 : total_cost_large_glasses = 25)
                       (h3 : total_cost_glasses = 49) : total_cost_glasses + change_left = 50 :=
by sorry

end NUMINAMATH_GPT_initial_amount_l748_74867


namespace NUMINAMATH_GPT_ellipse_eccentricity_l748_74844

theorem ellipse_eccentricity (x y : ℝ) (h : x^2 / 25 + y^2 / 9 = 1) : 
  let a := 5
  let b := 3
  let c := 4
  let e := c / a
  e = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l748_74844


namespace NUMINAMATH_GPT_find_d_value_l748_74857

theorem find_d_value (d : ℝ) :
  (∀ x, (8 * x^3 + 27 * x^2 + d * x + 55 = 0) → (2 * x + 5 = 0)) → d = 39.5 :=
by
  sorry

end NUMINAMATH_GPT_find_d_value_l748_74857


namespace NUMINAMATH_GPT_least_number_to_make_divisible_by_3_l748_74885

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_number_to_make_divisible_by_3 : ∃ k : ℕ, (∃ n : ℕ, 
  sum_of_digits 625573 ≡ 28 [MOD 3] ∧ 
  (625573 + k) % 3 = 0 ∧ 
  k = 2) :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_make_divisible_by_3_l748_74885


namespace NUMINAMATH_GPT_sandy_spent_money_l748_74819

theorem sandy_spent_money :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 :=
by
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  have total_spent : shorts + shirt + jacket = 33.56 := sorry
  exact total_spent

end NUMINAMATH_GPT_sandy_spent_money_l748_74819


namespace NUMINAMATH_GPT_sqrt_meaningful_implies_x_ge_2_l748_74854

theorem sqrt_meaningful_implies_x_ge_2 (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 := 
sorry

end NUMINAMATH_GPT_sqrt_meaningful_implies_x_ge_2_l748_74854


namespace NUMINAMATH_GPT_total_production_first_four_days_max_min_production_difference_total_wage_for_week_l748_74899

open Int

/-- Problem Statement -/
def planned_production : Int := 220

def production_change : List Int :=
  [5, -2, -4, 13, -10, 16, -9]

/-- Proof problem for total production in the first four days -/
theorem total_production_first_four_days :
  let first_four_days := production_change.take 4
  let total_change := first_four_days.sum
  let planned_first_four_days := planned_production * 4
  planned_first_four_days + total_change = 892 := 
by
  sorry

/-- Proof problem for difference in production between highest and lowest days -/
theorem max_min_production_difference :
  let max_change := production_change.maximum.getD 0
  let min_change := production_change.minimum.getD 0
  max_change - min_change = 26 := 
by
  sorry

/-- Proof problem for total wage calculation for the week -/
theorem total_wage_for_week :
  let total_change := production_change.sum
  let planned_week_total := planned_production * 7
  let actual_total := planned_week_total + total_change
  let base_wage := actual_total * 100
  let additional_wage := total_change * 20
  base_wage + additional_wage = 155080 := 
by
  sorry

end NUMINAMATH_GPT_total_production_first_four_days_max_min_production_difference_total_wage_for_week_l748_74899


namespace NUMINAMATH_GPT_average_speed_eq_l748_74875

variables (v₁ v₂ : ℝ) (t₁ t₂ : ℝ)

theorem average_speed_eq (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  ((v₁ * t₁) + (v₂ * t₂)) / (t₁ + t₂) = (v₁ + v₂) / 2 := 
sorry

end NUMINAMATH_GPT_average_speed_eq_l748_74875


namespace NUMINAMATH_GPT_volume_ratio_l748_74824

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) : 
  C / (A + B) = 23 / 12 :=
sorry

end NUMINAMATH_GPT_volume_ratio_l748_74824


namespace NUMINAMATH_GPT_remove_terms_to_get_two_thirds_l748_74809

noncomputable def sum_of_terms : ℚ := 
  (1/3) + (1/6) + (1/9) + (1/12) + (1/15) + (1/18)

noncomputable def sum_of_remaining_terms := 
  (1/3) + (1/6) + (1/9) + (1/18)

theorem remove_terms_to_get_two_thirds :
  sum_of_terms - (1/12 + 1/15) = (2/3) :=
by
  sorry

end NUMINAMATH_GPT_remove_terms_to_get_two_thirds_l748_74809


namespace NUMINAMATH_GPT_calc_result_l748_74841

theorem calc_result :
  12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end NUMINAMATH_GPT_calc_result_l748_74841


namespace NUMINAMATH_GPT_fish_price_relation_l748_74865

variables (b_c m_c b_v m_v : ℝ)

axiom cond1 : 3 * b_c + m_c = 5 * b_v
axiom cond2 : 2 * b_c + m_c = 3 * b_v + m_v

theorem fish_price_relation : 5 * m_v = b_c + 2 * m_c :=
by
  sorry

end NUMINAMATH_GPT_fish_price_relation_l748_74865


namespace NUMINAMATH_GPT_octagon_mass_l748_74815

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end NUMINAMATH_GPT_octagon_mass_l748_74815


namespace NUMINAMATH_GPT_arrow_in_48th_position_l748_74879

def arrow_sequence : List (String) := ["→", "↑", "↓", "←", "↘"]

theorem arrow_in_48th_position :
  arrow_sequence.get? ((48 % 5) - 1) = some "↓" :=
by
  norm_num
  sorry

end NUMINAMATH_GPT_arrow_in_48th_position_l748_74879


namespace NUMINAMATH_GPT_bryden_collection_value_l748_74849

-- Define the conditions
def face_value_half_dollar : ℝ := 0.5
def face_value_quarter : ℝ := 0.25
def num_half_dollars : ℕ := 5
def num_quarters : ℕ := 3
def multiplier : ℝ := 30

-- Define the problem statement as a theorem
theorem bryden_collection_value : 
  (multiplier * (num_half_dollars * face_value_half_dollar + num_quarters * face_value_quarter)) = 97.5 :=
by
  -- Proof is skipped since it's not required
  sorry

end NUMINAMATH_GPT_bryden_collection_value_l748_74849


namespace NUMINAMATH_GPT_determineHairColors_l748_74807

structure Person where
  name : String
  hairColor : String

def Belokurov : Person := { name := "Belokurov", hairColor := "" }
def Chernov : Person := { name := "Chernov", hairColor := "" }
def Ryzhev : Person := { name := "Ryzhev", hairColor := "" }

-- Define the possible hair colors
def Blonde : String := "Blonde"
def Brunette : String := "Brunette"
def RedHaired : String := "Red-Haired"

-- Define the conditions based on the problem statement
axiom hairColorConditions :
  Belokurov.hairColor ≠ Blonde ∧
  Belokurov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ Blonde ∧
  ∀ p : Person, p.hairColor = Brunette → p.name ≠ "Belokurov"

-- Define the uniqueness condition that each person has a different hair color
axiom uniqueHairColors :
  Belokurov.hairColor ≠ Chernov.hairColor ∧
  Belokurov.hairColor ≠ Ryzhev.hairColor ∧
  Chernov.hairColor ≠ Ryzhev.hairColor

-- Define the proof problem
theorem determineHairColors :
  Belokurov.hairColor = RedHaired ∧
  Chernov.hairColor = Blonde ∧
  Ryzhev.hairColor = Brunette := by
  sorry

end NUMINAMATH_GPT_determineHairColors_l748_74807


namespace NUMINAMATH_GPT_ratio_of_w_to_y_l748_74800

theorem ratio_of_w_to_y
  (w x y z : ℚ)
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_w_to_y_l748_74800


namespace NUMINAMATH_GPT_share_of_B_l748_74820

theorem share_of_B (x : ℕ) (A B C : ℕ) (h1 : A = 3 * B) (h2 : B = C + 25)
  (h3 : A + B + C = 645) : B = 134 :=
by
  sorry

end NUMINAMATH_GPT_share_of_B_l748_74820


namespace NUMINAMATH_GPT_bottles_needed_l748_74888

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end NUMINAMATH_GPT_bottles_needed_l748_74888


namespace NUMINAMATH_GPT_sum_of_A_and_B_l748_74842

theorem sum_of_A_and_B (A B : ℕ) (h1 : 7 - B = 3) (h2 : A - 5 = 4) (h_diff : A ≠ B) : A + B = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_A_and_B_l748_74842


namespace NUMINAMATH_GPT_simple_interest_correct_l748_74829

theorem simple_interest_correct (P R T : ℝ) (hP : P = 400) (hR : R = 12.5) (hT : T = 2) : 
  (P * R * T) / 100 = 50 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_simple_interest_correct_l748_74829


namespace NUMINAMATH_GPT_winning_strategy_for_B_l748_74848

theorem winning_strategy_for_B (N : ℕ) (h : N < 15) : N = 7 ↔ (∃ strategy : (Fin 6 → ℕ) → ℕ, ∀ f : Fin 6 → ℕ, (strategy f) % 1001 = 0) :=
by
  sorry

end NUMINAMATH_GPT_winning_strategy_for_B_l748_74848


namespace NUMINAMATH_GPT_initial_stock_before_shipment_l748_74880

-- Define the conditions for the problem
def initial_stock (total_shelves new_shipment_bears bears_per_shelf: ℕ) : ℕ :=
  let total_bears_on_shelves := total_shelves * bears_per_shelf
  total_bears_on_shelves - new_shipment_bears

-- State the theorem with the conditions
theorem initial_stock_before_shipment : initial_stock 2 10 7 = 4 := by
  -- Mathematically, the calculation details will be handled here
  sorry

end NUMINAMATH_GPT_initial_stock_before_shipment_l748_74880


namespace NUMINAMATH_GPT_carter_baseball_cards_l748_74814

theorem carter_baseball_cards (M C : ℕ) (h1 : M = 210) (h2 : M = C + 58) : C = 152 :=
by
  sorry

end NUMINAMATH_GPT_carter_baseball_cards_l748_74814


namespace NUMINAMATH_GPT_math_expression_identity_l748_74822

theorem math_expression_identity :
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_math_expression_identity_l748_74822


namespace NUMINAMATH_GPT_sum_eq_prod_nat_numbers_l748_74838

theorem sum_eq_prod_nat_numbers (A B C D E F : ℕ) :
  A + B + C + D + E + F = A * B * C * D * E * F →
  (A = 0 ∧ B = 0 ∧ C = 0 ∧ D = 0 ∧ E = 0 ∧ F = 0) ∨
  (A = 1 ∧ B = 1 ∧ C = 1 ∧ D = 1 ∧ E = 2 ∧ F = 6) :=
by
  sorry

end NUMINAMATH_GPT_sum_eq_prod_nat_numbers_l748_74838


namespace NUMINAMATH_GPT_factorization_correct_l748_74808

theorem factorization_correct (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end NUMINAMATH_GPT_factorization_correct_l748_74808


namespace NUMINAMATH_GPT_max_k_for_ineq_l748_74825

theorem max_k_for_ineq (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 > (m + n)^2) :
  m^3 + n^3 ≥ (m + n)^2 + 10 :=
sorry

end NUMINAMATH_GPT_max_k_for_ineq_l748_74825


namespace NUMINAMATH_GPT_fraction_spent_on_food_l748_74877

variable (salary : ℝ) (food_fraction rent_fraction clothes_fraction remaining_amount : ℝ)
variable (salary_condition : salary = 180000)
variable (rent_fraction_condition : rent_fraction = 1/10)
variable (clothes_fraction_condition : clothes_fraction = 3/5)
variable (remaining_amount_condition : remaining_amount = 18000)

theorem fraction_spent_on_food :
  rent_fraction * salary + clothes_fraction * salary + food_fraction * salary + remaining_amount = salary →
  food_fraction = 1/5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_spent_on_food_l748_74877


namespace NUMINAMATH_GPT_max_abc_l748_74818

def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

theorem max_abc (a b c n : ℕ) (hpos : n > 0) (h1 : 1 ≤ a ∧ a < 10) (h2 : 1 ≤ b ∧ b < 10) (h3 : 1 ≤ c ∧ c < 10) (h_eq : C_n c n - B_n b n = A_n a n ^ 2) :  a + b + c ≤ 18 :=
by sorry

end NUMINAMATH_GPT_max_abc_l748_74818


namespace NUMINAMATH_GPT_find_number_l748_74893

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end NUMINAMATH_GPT_find_number_l748_74893


namespace NUMINAMATH_GPT_milk_left_is_correct_l748_74802

def total_morning_milk : ℕ := 365
def total_evening_milk : ℕ := 380
def milk_sold : ℕ := 612
def leftover_milk_from_yesterday : ℕ := 15

def total_milk_left : ℕ :=
  (total_morning_milk + total_evening_milk - milk_sold) + leftover_milk_from_yesterday

theorem milk_left_is_correct : total_milk_left = 148 := by
  sorry

end NUMINAMATH_GPT_milk_left_is_correct_l748_74802


namespace NUMINAMATH_GPT_machine_A_time_to_produce_x_boxes_l748_74817

-- Definitions of the conditions
def machine_A_rate (T : ℕ) (x : ℕ) : ℚ := x / T
def machine_B_rate (x : ℕ) : ℚ := 2 * x / 5
def combined_rate (T : ℕ) (x : ℕ) : ℚ := (x / 2) 

-- The theorem statement
theorem machine_A_time_to_produce_x_boxes (x : ℕ) : 
  ∀ T : ℕ, 20 * (machine_A_rate T x + machine_B_rate x) = 10 * x → T = 10 :=
by
  intros T h
  sorry

end NUMINAMATH_GPT_machine_A_time_to_produce_x_boxes_l748_74817


namespace NUMINAMATH_GPT_inequality_comparison_l748_74876

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end NUMINAMATH_GPT_inequality_comparison_l748_74876


namespace NUMINAMATH_GPT_larger_number_is_l748_74843

-- Given definitions and conditions
def HCF (a b: ℕ) : ℕ := 23
def other_factor_1 : ℕ := 11
def other_factor_2 : ℕ := 12
def LCM (a b: ℕ) : ℕ := HCF a b * other_factor_1 * other_factor_2

-- Statement to be proven
theorem larger_number_is (a b: ℕ) (h: HCF a b = 23) (hA: a = 23 * 12) (hB: b ∣ a) : a = 276 :=
by { sorry }

end NUMINAMATH_GPT_larger_number_is_l748_74843


namespace NUMINAMATH_GPT_find_n_l748_74872

theorem find_n (a b : ℤ) (h₁ : a ≡ 25 [ZMOD 42]) (h₂ : b ≡ 63 [ZMOD 42]) :
  ∃ n, 200 ≤ n ∧ n ≤ 241 ∧ (a - b ≡ n [ZMOD 42]) ∧ n = 214 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l748_74872


namespace NUMINAMATH_GPT_correct_expression_l748_74889

theorem correct_expression :
  (2 + Real.sqrt 3 ≠ 2 * Real.sqrt 3) ∧ 
  (Real.sqrt 8 - Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6) ∧ 
  (Real.sqrt 27 / Real.sqrt 3 ≠ 9) := 
by
  sorry

end NUMINAMATH_GPT_correct_expression_l748_74889


namespace NUMINAMATH_GPT_sum_of_interior_angles_10th_polygon_l748_74831

theorem sum_of_interior_angles_10th_polygon (n : ℕ) (h1 : n = 10) : 
  180 * (n - 2) = 1440 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_10th_polygon_l748_74831


namespace NUMINAMATH_GPT_sum_of_ages_equal_to_grandpa_l748_74891

-- Conditions
def grandpa_age : Nat := 75
def grandchild_age_1 : Nat := 13
def grandchild_age_2 : Nat := 15
def grandchild_age_3 : Nat := 17

-- Main Statement
theorem sum_of_ages_equal_to_grandpa (t : Nat) :
  (grandchild_age_1 + t) + (grandchild_age_2 + t) + (grandchild_age_3 + t) = grandpa_age + t 
  ↔ t = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_ages_equal_to_grandpa_l748_74891


namespace NUMINAMATH_GPT_train_length_eq_l748_74884

-- Definitions
def train_speed_kmh : Float := 45
def crossing_time_s : Float := 30
def total_length_m : Float := 245

-- Theorem statement
theorem train_length_eq :
  ∃ (train_length bridge_length: Float),
  bridge_length = total_length_m - train_length ∧
  train_speed_kmh * 1000 / 3600 * crossing_time_s = train_length + bridge_length ∧
  train_length = 130 :=
by
  sorry

end NUMINAMATH_GPT_train_length_eq_l748_74884


namespace NUMINAMATH_GPT_probability_odd_and_multiple_of_5_l748_74851

/-- Given three distinct integers selected at random between 1 and 2000, inclusive, the probability that the product of the three integers is odd and a multiple of 5 is between 0.01 and 0.05. -/
theorem probability_odd_and_multiple_of_5 :
  ∃ p : ℚ, (0.01 < p ∧ p < 0.05) :=
sorry

end NUMINAMATH_GPT_probability_odd_and_multiple_of_5_l748_74851


namespace NUMINAMATH_GPT_oranges_count_l748_74836

def oranges_per_box : ℝ := 10
def boxes_per_day : ℝ := 2650
def total_oranges (x y : ℝ) : ℝ := x * y

theorem oranges_count :
  total_oranges oranges_per_box boxes_per_day = 26500 := 
  by sorry

end NUMINAMATH_GPT_oranges_count_l748_74836


namespace NUMINAMATH_GPT_finding_b_for_infinite_solutions_l748_74813

theorem finding_b_for_infinite_solutions :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 :=
by
  sorry

end NUMINAMATH_GPT_finding_b_for_infinite_solutions_l748_74813


namespace NUMINAMATH_GPT_broken_line_AEC_correct_l748_74874

noncomputable def length_of_broken_line_AEC 
  (side_length : ℝ)
  (height_of_pyramid : ℝ)
  (radius_of_equiv_circle : ℝ) 
  (length_AE : ℝ)
  (length_AEC : ℝ) : Prop :=
  side_length = 230.0 ∧
  height_of_pyramid = 146.423 ∧
  radius_of_equiv_circle = height_of_pyramid ∧
  length_AE = ((230.0 * 186.184) / 218.837) ∧
  length_AEC = 2 * length_AE ∧
  round (length_AEC * 100) = 39136

theorem broken_line_AEC_correct :
  length_of_broken_line_AEC 230 146.423 (146.423) 195.681 391.362 :=
by
  sorry

end NUMINAMATH_GPT_broken_line_AEC_correct_l748_74874


namespace NUMINAMATH_GPT_shopkeeper_total_cards_l748_74860

-- Conditions
def num_standard_decks := 3
def cards_per_standard_deck := 52
def num_tarot_decks := 2
def cards_per_tarot_deck := 72
def num_trading_sets := 5
def cards_per_trading_set := 100
def additional_random_cards := 27

-- Calculate total cards
def total_standard_cards := num_standard_decks * cards_per_standard_deck
def total_tarot_cards := num_tarot_decks * cards_per_tarot_deck
def total_trading_cards := num_trading_sets * cards_per_trading_set
def total_cards := total_standard_cards + total_tarot_cards + total_trading_cards + additional_random_cards

-- Proof statement
theorem shopkeeper_total_cards : total_cards = 827 := by
    sorry

end NUMINAMATH_GPT_shopkeeper_total_cards_l748_74860


namespace NUMINAMATH_GPT_abs_m_minus_1_greater_eq_abs_m_minus_1_l748_74892

theorem abs_m_minus_1_greater_eq_abs_m_minus_1 (m : ℝ) : |m - 1| ≥ |m| - 1 := 
sorry

end NUMINAMATH_GPT_abs_m_minus_1_greater_eq_abs_m_minus_1_l748_74892


namespace NUMINAMATH_GPT_angle_difference_l748_74811

theorem angle_difference (X Y Z Z1 Z2 : ℝ) (h1 : Y = 2 * X) (h2 : X = 30) (h3 : Z1 + Z2 = Z) (h4 : Z1 = 60) (h5 : Z2 = 30) : Z1 - Z2 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_angle_difference_l748_74811


namespace NUMINAMATH_GPT_greatest_perfect_square_power_of_3_under_200_l748_74866

theorem greatest_perfect_square_power_of_3_under_200 :
  ∃ n : ℕ, n < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ n = 3 ^ k) ∧ ∀ m : ℕ, (m < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ m = 3 ^ k)) → m ≤ n :=
  sorry

end NUMINAMATH_GPT_greatest_perfect_square_power_of_3_under_200_l748_74866


namespace NUMINAMATH_GPT_smallest_x_multiple_of_53_l748_74840

theorem smallest_x_multiple_of_53 :
  ∃ x : ℕ, (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_x_multiple_of_53_l748_74840


namespace NUMINAMATH_GPT_tape_for_small_box_l748_74821

theorem tape_for_small_box (S : ℝ) :
  (2 * 4) + (8 * 2) + (5 * S) + (2 + 8 + 5) = 44 → S = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tape_for_small_box_l748_74821


namespace NUMINAMATH_GPT_no_solution_prob1_l748_74856

theorem no_solution_prob1 : ¬ ∃ x : ℝ, x ≠ 2 ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_prob1_l748_74856


namespace NUMINAMATH_GPT_problem_statement_l748_74830

theorem problem_statement (a x m : ℝ) (h₀ : |a| ≤ 1) (h₁ : |x| ≤ 1) :
  (∀ x a, |x^2 - a * x - a^2| ≤ m) ↔ m ≥ 5/4 :=
sorry

end NUMINAMATH_GPT_problem_statement_l748_74830


namespace NUMINAMATH_GPT_point_not_on_graph_l748_74858

theorem point_not_on_graph :
  ¬ (1 * 5 = 6) :=
by 
  sorry

end NUMINAMATH_GPT_point_not_on_graph_l748_74858


namespace NUMINAMATH_GPT_find_f2_l748_74846

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- Condition: f'(x) = a
def f_derivative (a b x : ℝ) : ℝ := a

-- Given conditions
variables (a b : ℝ)
axiom h1 : f a b 1 = 2
axiom h2 : f_derivative a b 1 = 2

theorem find_f2 : f a b 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l748_74846


namespace NUMINAMATH_GPT_carl_watermelons_left_l748_74803

-- Define the conditions
def price_per_watermelon : ℕ := 3
def profit : ℕ := 105
def starting_watermelons : ℕ := 53

-- Define the main proof statement
theorem carl_watermelons_left :
  (starting_watermelons - (profit / price_per_watermelon) = 18) :=
sorry

end NUMINAMATH_GPT_carl_watermelons_left_l748_74803


namespace NUMINAMATH_GPT_smallest_positive_four_digit_equivalent_to_5_mod_8_l748_74890

theorem smallest_positive_four_digit_equivalent_to_5_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n % 8 = 5 ∧ n = 1005 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_four_digit_equivalent_to_5_mod_8_l748_74890


namespace NUMINAMATH_GPT_value_at_neg_9_over_2_l748_74845

def f : ℝ → ℝ := sorry 

axiom odd_function (x : ℝ) : f (-x) + f x = 0

axiom symmetric_y_axis (x : ℝ) : f (1 + x) = f (1 - x)

axiom functional_eq (x k : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hk : 0 ≤ k ∧ k ≤ 1) : f (k * x) + 1 = (f x + 1) ^ k

axiom f_at_1 : f 1 = - (1 / 2)

theorem value_at_neg_9_over_2 : f (- (9 / 2)) = 1 - (Real.sqrt 2) / 2 := 
sorry

end NUMINAMATH_GPT_value_at_neg_9_over_2_l748_74845


namespace NUMINAMATH_GPT_parabola_standard_equation_l748_74897

theorem parabola_standard_equation :
  ∃ p1 p2 : ℝ, p1 > 0 ∧ p2 > 0 ∧ (y^2 = 2 * p1 * x ∨ x^2 = 2 * p2 * y) ∧ ((6, 4) ∈ {(x, y) | y^2 = 2 * p1 * x} ∨ (6, 4) ∈ {(x, y) | x^2 = 2 * p2 * y}) := 
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_l748_74897


namespace NUMINAMATH_GPT_range_of_m_l748_74823

variable {x m : ℝ}
variable (q: ℝ → Prop) (p: ℝ → Prop)

-- Definition of q
def q_cond : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

-- Definition of p
def p_cond : Prop := |1 - (x - 1) / 3| ≤ 2

-- Statement of the proof problem
theorem range_of_m (h1 : ∀ x, q x → p x) (h2 : ∃ x, ¬p x → q x) 
  (h3 : m > 0) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l748_74823


namespace NUMINAMATH_GPT_picture_distance_l748_74898

theorem picture_distance (w t s p d : ℕ) (h1 : w = 25) (h2 : t = 2) (h3 : s = 1) (h4 : 2 * p + s = t + s + t) 
  (h5 : w = 2 * d + p) : d = 10 :=
by
  sorry

end NUMINAMATH_GPT_picture_distance_l748_74898


namespace NUMINAMATH_GPT_train_crosses_man_in_6_seconds_l748_74861

/-- A train of length 240 meters, traveling at a speed of 144 km/h, will take 6 seconds to cross a man standing on the platform. -/
theorem train_crosses_man_in_6_seconds
  (length_of_train : ℕ)
  (speed_of_train : ℕ)
  (conversion_factor : ℕ)
  (speed_in_m_per_s : ℕ)
  (time_to_cross : ℕ)
  (h1 : length_of_train = 240)
  (h2 : speed_of_train = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_in_m_per_s = speed_of_train * conversion_factor)
  (h5 : speed_in_m_per_s = 40)
  (h6 : time_to_cross = length_of_train / speed_in_m_per_s) :
  time_to_cross = 6 := by
  sorry

end NUMINAMATH_GPT_train_crosses_man_in_6_seconds_l748_74861


namespace NUMINAMATH_GPT_sqrt_6_approx_l748_74873

noncomputable def newton_iteration (x : ℝ) : ℝ :=
  (1 / 2) * x + (3 / x)

theorem sqrt_6_approx :
  let x0 : ℝ := 2
  let x1 : ℝ := newton_iteration x0
  let x2 : ℝ := newton_iteration x1
  let x3 : ℝ := newton_iteration x2
  abs (x3 - 2.4495) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_6_approx_l748_74873


namespace NUMINAMATH_GPT_find_quotient_l748_74894

def dividend : ℝ := 13787
def remainder : ℝ := 14
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89

theorem find_quotient :
  (dividend - remainder) / divisor = quotient :=
sorry

end NUMINAMATH_GPT_find_quotient_l748_74894


namespace NUMINAMATH_GPT_apples_hand_out_l748_74847

theorem apples_hand_out (t p a h : ℕ) (h_t : t = 62) (h_p : p = 6) (h_a : a = 9) : h = t - (p * a) → h = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_apples_hand_out_l748_74847


namespace NUMINAMATH_GPT_calculate_expression_l748_74870

theorem calculate_expression :
  (Int.floor ((15:ℚ)/8 * ((-34:ℚ)/4)) - Int.ceil ((15:ℚ)/8 * Int.floor ((-34:ℚ)/4))) = 0 := 
  by sorry

end NUMINAMATH_GPT_calculate_expression_l748_74870


namespace NUMINAMATH_GPT_find_d_plus_q_l748_74804

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + d * (n * (n - 1) / 2)

noncomputable def sum_geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁
  else b₁ * (q ^ n - 1) / (q - 1)

noncomputable def sum_combined_sequence (a₁ d b₁ q : ℝ) (n : ℕ) : ℝ :=
  sum_arithmetic_sequence a₁ d n + sum_geometric_sequence b₁ q n

theorem find_d_plus_q (a₁ d b₁ q : ℝ) (h_seq: ∀ n : ℕ, 0 < n → sum_combined_sequence a₁ d b₁ q n = n^2 - n + 2^n - 1) :
  d + q = 4 :=
  sorry

end NUMINAMATH_GPT_find_d_plus_q_l748_74804


namespace NUMINAMATH_GPT_cost_per_unit_range_of_type_A_purchases_maximum_profit_l748_74895

-- Definitions of the problem conditions
def cost_type_A : ℕ := 15
def cost_type_B : ℕ := 20

def profit_type_A : ℕ := 3
def profit_type_B : ℕ := 4

def budget_min : ℕ := 2750
def budget_max : ℕ := 2850

def total_units : ℕ := 150
def profit_min : ℕ := 565

-- Main proof statements as Lean theorems
theorem cost_per_unit : 
  ∃ (x y : ℕ), 
    2 * x + 3 * y = 90 ∧ 
    3 * x + y = 65 ∧ 
    x = cost_type_A ∧ 
    y = cost_type_B := 
sorry

theorem range_of_type_A_purchases : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 50 ∧ 
    budget_min ≤ cost_type_A * a + cost_type_B * (total_units - a) ∧ 
    cost_type_A * a + cost_type_B * (total_units - a) ≤ budget_max := 
sorry

theorem maximum_profit : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 35 ∧ 
    profit_min ≤ profit_type_A * a + profit_type_B * (total_units - a) ∧ 
    ¬∃ (b : ℕ), 
      30 ≤ b ∧ 
      b ≤ 35 ∧ 
      b ≠ a ∧ 
      profit_type_A * b + profit_type_B * (total_units - b) > profit_type_A * a + profit_type_B * (total_units - a) :=
sorry

end NUMINAMATH_GPT_cost_per_unit_range_of_type_A_purchases_maximum_profit_l748_74895


namespace NUMINAMATH_GPT_none_of_these_l748_74832

noncomputable def x (t : ℝ) : ℝ := t ^ (3 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t ^ ((t + 1) / (t - 1))

theorem none_of_these (t : ℝ) (ht_pos : t > 0) (ht_ne_one : t ≠ 1) :
  ¬ (y t ^ x t = x t ^ y t) ∧ ¬ (x t ^ x t = y t ^ y t) ∧
  ¬ (x t ^ (y t ^ x t) = y t ^ (x t ^ y t)) ∧ ¬ (x t ^ y t = y t ^ x t) :=
sorry

end NUMINAMATH_GPT_none_of_these_l748_74832


namespace NUMINAMATH_GPT_second_term_geometric_sequence_l748_74878

-- Given conditions
def a3 : ℕ := 12
def a4 : ℕ := 18
def q := a4 / a3 -- common ratio

-- Geometric progression definition
noncomputable def a2 := a3 / q

-- Theorem to prove
theorem second_term_geometric_sequence : a2 = 8 :=
by
  -- proof not required
  sorry

end NUMINAMATH_GPT_second_term_geometric_sequence_l748_74878


namespace NUMINAMATH_GPT_two_wheeler_wheels_l748_74837

-- Define the total number of wheels and the number of four-wheelers
def total_wheels : Nat := 46
def num_four_wheelers : Nat := 11

-- Define the number of wheels per vehicle type
def wheels_per_four_wheeler : Nat := 4
def wheels_per_two_wheeler : Nat := 2

-- Define the number of two-wheelers
def num_two_wheelers : Nat := (total_wheels - num_four_wheelers * wheels_per_four_wheeler) / wheels_per_two_wheeler

-- Proposition stating the number of wheels of the two-wheeler
theorem two_wheeler_wheels : wheels_per_two_wheeler * num_two_wheelers = 2 := by
  sorry

end NUMINAMATH_GPT_two_wheeler_wheels_l748_74837


namespace NUMINAMATH_GPT_y_pow_x_eq_nine_l748_74853

theorem y_pow_x_eq_nine (x y : ℝ) (h : x^2 + y^2 - 4 * x + 6 * y + 13 = 0) : y^x = 9 := by
  sorry

end NUMINAMATH_GPT_y_pow_x_eq_nine_l748_74853


namespace NUMINAMATH_GPT_sue_cost_l748_74833

def cost_of_car : ℝ := 2100
def total_days_in_week : ℝ := 7
def sue_days : ℝ := 3

theorem sue_cost : (cost_of_car * (sue_days / total_days_in_week)) = 899.99 :=
by
  sorry

end NUMINAMATH_GPT_sue_cost_l748_74833


namespace NUMINAMATH_GPT_oranges_for_juice_l748_74834

theorem oranges_for_juice 
  (bags : ℕ) (oranges_per_bag : ℕ) (rotten_oranges : ℕ) (oranges_sold : ℕ)
  (h_bags : bags = 10)
  (h_oranges_per_bag : oranges_per_bag = 30)
  (h_rotten_oranges : rotten_oranges = 50)
  (h_oranges_sold : oranges_sold = 220):
  (bags * oranges_per_bag - rotten_oranges - oranges_sold = 30) :=
by 
  sorry

end NUMINAMATH_GPT_oranges_for_juice_l748_74834


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l748_74812

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / (a + 1)) + (2 / (1 + b)) = 1) : 
  a + b ≥ 2 * Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l748_74812


namespace NUMINAMATH_GPT_yard_length_l748_74869

-- Define the given conditions
def num_trees : ℕ := 26
def dist_between_trees : ℕ := 13

-- Calculate the length of the yard
def num_gaps : ℕ := num_trees - 1
def length_of_yard : ℕ := num_gaps * dist_between_trees

-- Theorem statement: the length of the yard is 325 meters
theorem yard_length : length_of_yard = 325 := by
  sorry

end NUMINAMATH_GPT_yard_length_l748_74869


namespace NUMINAMATH_GPT_zero_point_six_six_six_is_fraction_l748_74852

def is_fraction (x : ℝ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = (n : ℝ) / (d : ℝ)

theorem zero_point_six_six_six_is_fraction:
  let sqrt_2_div_3 := (Real.sqrt 2) / 3
  let neg_sqrt_4 := - Real.sqrt 4
  let zero_point_six_six_six := 0.666
  let one_seventh := 1 / 7
  is_fraction zero_point_six_six_six :=
by sorry

end NUMINAMATH_GPT_zero_point_six_six_six_is_fraction_l748_74852


namespace NUMINAMATH_GPT_total_ice_cream_volume_l748_74896

def cone_height : ℝ := 10
def cone_radius : ℝ := 1.5
def cylinder_height : ℝ := 2
def cylinder_radius : ℝ := 1.5
def hemisphere_radius : ℝ := 1.5

theorem total_ice_cream_volume : 
  (1 / 3 * π * cone_radius ^ 2 * cone_height) +
  (π * cylinder_radius ^ 2 * cylinder_height) +
  (2 / 3 * π * hemisphere_radius ^ 3) = 14.25 * π :=
by sorry

end NUMINAMATH_GPT_total_ice_cream_volume_l748_74896
