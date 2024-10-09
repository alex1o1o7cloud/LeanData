import Mathlib

namespace meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l1777_177794

theorem meters_to_kilometers (h : 1 = 1000) : 6000 / 1000 = 6 := by
  sorry

theorem kilograms_to_grams (h : 1 = 1000) : (5 + 2) * 1000 = 7000 := by
  sorry

theorem centimeters_to_decimeters (h : 10 = 1) : (58 + 32) / 10 = 9 := by
  sorry

theorem hours_to_minutes (h : 60 = 1) : 3 * 60 + 30 = 210 := by
  sorry

end meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l1777_177794


namespace trapezoid_height_l1777_177786

variables (a b h : ℝ)

def is_trapezoid (a b h : ℝ) (angle_diag : ℝ) (angle_ext : ℝ) : Prop :=
a < b ∧ angle_diag = 90 ∧ angle_ext = 45

theorem trapezoid_height
  (a b : ℝ) (ha : a < b)
  (angle_diag : ℝ) (h_angle_diag : angle_diag = 90)
  (angle_ext : ℝ) (h_angle_ext : angle_ext = 45)
  (h_def : is_trapezoid a b h angle_diag angle_ext) :
  h = a * b / (b - a) :=
sorry

end trapezoid_height_l1777_177786


namespace incorrect_statement_l1777_177706

noncomputable def first_line_of_defense := "Skin and mucous membranes"
noncomputable def second_line_of_defense := "Antimicrobial substances and phagocytic cells in body fluids"
noncomputable def third_line_of_defense := "Immune organs and immune cells"
noncomputable def non_specific_immunity := "First and second line of defense"
noncomputable def specific_immunity := "Third line of defense"
noncomputable def d_statement := "The defensive actions performed by the three lines of defense in the human body are called non-specific immunity"

theorem incorrect_statement : d_statement ≠ specific_immunity ∧ d_statement ≠ non_specific_immunity := by
  sorry

end incorrect_statement_l1777_177706


namespace min_value_of_b_plus_2_div_a_l1777_177726

theorem min_value_of_b_plus_2_div_a (a : ℝ) (b : ℝ) (h₁ : 0 < a) 
  (h₂ : ∀ x : ℝ, 0 < x → (ax - 1) * (x^2 + bx - 4) ≥ 0) : 
  ∃ a' b', (a' > 0 ∧ b' = 4 * a' - 1 / a') ∧ b' + 2 / a' = 4 :=
by
  sorry

end min_value_of_b_plus_2_div_a_l1777_177726


namespace bears_in_shipment_l1777_177748

theorem bears_in_shipment (initial_bears shipment_bears bears_per_shelf total_shelves : ℕ)
  (h1 : initial_bears = 17)
  (h2 : bears_per_shelf = 9)
  (h3 : total_shelves = 3)
  (h4 : total_shelves * bears_per_shelf = 27) :
  shipment_bears = 10 :=
by
  sorry

end bears_in_shipment_l1777_177748


namespace score_of_29_impossible_l1777_177707

theorem score_of_29_impossible :
  ¬ ∃ (c u w : ℕ), c + u + w = 10 ∧ 3 * c + u = 29 :=
by {
  sorry
}

end score_of_29_impossible_l1777_177707


namespace find_x_y_n_l1777_177793

def is_reverse_digit (x y : ℕ) : Prop := 
  x / 10 = y % 10 ∧ x % 10 = y / 10

def is_two_digit_nonzero (z : ℕ) : Prop := 
  10 ≤ z ∧ z < 100

theorem find_x_y_n : 
  ∃ (x y n : ℕ), is_two_digit_nonzero x ∧ is_two_digit_nonzero y ∧ is_reverse_digit x y ∧ (x^2 - y^2 = 44 * n) ∧ (x + y + n = 93) :=
sorry

end find_x_y_n_l1777_177793


namespace smallest_number_of_butterflies_l1777_177779

theorem smallest_number_of_butterflies 
  (identical_groups : ℕ) 
  (groups_of_butterflies : ℕ) 
  (groups_of_fireflies : ℕ) 
  (groups_of_ladybugs : ℕ)
  (h1 : groups_of_butterflies = 44)
  (h2 : groups_of_fireflies = 17)
  (h3 : groups_of_ladybugs = 25)
  (h4 : identical_groups * (groups_of_butterflies + groups_of_fireflies + groups_of_ladybugs) % 60 = 0) :
  identical_groups * groups_of_butterflies = 425 :=
sorry

end smallest_number_of_butterflies_l1777_177779


namespace problem_real_numbers_l1777_177798

theorem problem_real_numbers (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) : 
  ((a / 3) ^ n + (b / 2) ^ n) ≥ 2 := 
sorry

end problem_real_numbers_l1777_177798


namespace factorize_expression_l1777_177750

-- Define the expression E
def E (x y z : ℝ) : ℝ := x^2 + x*y - x*z - y*z

-- State the theorem to prove \(E = (x + y)(x - z)\)
theorem factorize_expression (x y z : ℝ) : 
  E x y z = (x + y) * (x - z) := 
sorry

end factorize_expression_l1777_177750


namespace find_f_x_minus_1_l1777_177780

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem find_f_x_minus_1 (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end find_f_x_minus_1_l1777_177780


namespace abs_sum_inequality_solution_l1777_177719

theorem abs_sum_inequality_solution (x : ℝ) : 
  (|x - 5| + |x + 1| < 8) ↔ (-2 < x ∧ x < 6) :=
sorry

end abs_sum_inequality_solution_l1777_177719


namespace problem_condition_l1777_177742

variable {f : ℝ → ℝ}

theorem problem_condition (h_diff : Differentiable ℝ f) (h_ineq : ∀ x : ℝ, f x < iteratedDeriv 2 f x) : 
  e^2019 * f (-2019) < f 0 ∧ f 2019 > e^2019 * f 0 :=
by
  sorry

end problem_condition_l1777_177742


namespace oliver_siblings_l1777_177713

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)

def oliver := Child.mk "Oliver" "Gray" "Brown"
def charles := Child.mk "Charles" "Gray" "Red"
def diana := Child.mk "Diana" "Green" "Brown"
def olivia := Child.mk "Olivia" "Green" "Red"
def ethan := Child.mk "Ethan" "Green" "Red"
def fiona := Child.mk "Fiona" "Green" "Brown"

def sharesCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

def sameFamily (c1 c2 c3 : Child) : Prop :=
  sharesCharacteristic c1 c2 ∧
  sharesCharacteristic c2 c3 ∧
  sharesCharacteristic c3 c1

theorem oliver_siblings : 
  sameFamily oliver charles diana :=
by
  -- proof skipped
  sorry

end oliver_siblings_l1777_177713


namespace example_is_fraction_l1777_177770

def is_fraction (a b : ℚ) : Prop := ∃ x y : ℚ, a = x ∧ b = y ∧ y ≠ 0

-- Example condition relevant to the problem
theorem example_is_fraction (x : ℚ) : is_fraction x (x + 2) :=
by
  sorry

end example_is_fraction_l1777_177770


namespace oranges_sold_in_the_morning_eq_30_l1777_177746

variable (O : ℝ)  -- Denote the number of oranges Wendy sold in the morning

-- Conditions as assumptions
def price_per_apple : ℝ := 1.5
def price_per_orange : ℝ := 1
def morning_apples_sold : ℝ := 40
def afternoon_apples_sold : ℝ := 50
def afternoon_oranges_sold : ℝ := 40
def total_sales_for_day : ℝ := 205

-- Prove that O, satisfying the given conditions, equals 30
theorem oranges_sold_in_the_morning_eq_30 (h : 
    (morning_apples_sold * price_per_apple) +
    (O * price_per_orange) +
    (afternoon_apples_sold * price_per_apple) +
    (afternoon_oranges_sold * price_per_orange) = 
    total_sales_for_day
  ) : O = 30 :=
by
  sorry

end oranges_sold_in_the_morning_eq_30_l1777_177746


namespace cans_of_soda_l1777_177790

theorem cans_of_soda (S Q D : ℕ) : (4 * D * S) / Q = x :=
by
  sorry

end cans_of_soda_l1777_177790


namespace carter_students_received_grades_l1777_177769

theorem carter_students_received_grades
  (students_thompson : ℕ)
  (a_thompson : ℕ)
  (remaining_students_thompson : ℕ)
  (b_thompson : ℕ)
  (students_carter : ℕ)
  (ratio_A_thompson : ℚ)
  (ratio_B_thompson : ℚ)
  (A_carter : ℕ)
  (B_carter : ℕ) :
  students_thompson = 20 →
  a_thompson = 12 →
  remaining_students_thompson = 8 →
  b_thompson = 5 →
  students_carter = 30 →
  ratio_A_thompson = (a_thompson : ℚ) / students_thompson →
  ratio_B_thompson = (b_thompson : ℚ) / remaining_students_thompson →
  A_carter = ratio_A_thompson * students_carter →
  B_carter = (b_thompson : ℚ) / remaining_students_thompson * (students_carter - A_carter) →
  A_carter = 18 ∧ B_carter = 8 := 
by 
  intros;
  sorry

end carter_students_received_grades_l1777_177769


namespace largest_divisor_of_prime_squares_l1777_177754

theorem largest_divisor_of_prime_squares (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q < p) : 
  ∃ d : ℕ, ∀ p q : ℕ, Prime p → Prime q → q < p → d ∣ (p^2 - q^2) ∧ ∀ k : ℕ, (∀ p q : ℕ, Prime p → Prime q → q < p → k ∣ (p^2 - q^2)) → k ≤ d :=
by 
  use 2
  {
    sorry
  }

end largest_divisor_of_prime_squares_l1777_177754


namespace at_least_one_div_by_5_l1777_177789

-- Define natural numbers and divisibility by 5
def is_div_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- Proposition: If a, b are natural numbers and ab is divisible by 5, then at least one of a or b must be divisible by 5.
theorem at_least_one_div_by_5 (a b : ℕ) (h_ab : is_div_by_5 (a * b)) : is_div_by_5 a ∨ is_div_by_5 b :=
  by
    sorry

end at_least_one_div_by_5_l1777_177789


namespace g_is_odd_l1777_177758

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l1777_177758


namespace tan_of_cos_first_quadrant_l1777_177722

-- Define the angle α in the first quadrant and its cosine value
variable (α : ℝ) (h1 : 0 < α ∧ α < π/2) (hcos : Real.cos α = 2 / 3)

-- State the theorem
theorem tan_of_cos_first_quadrant : Real.tan α = Real.sqrt 5 / 2 := 
by
  sorry

end tan_of_cos_first_quadrant_l1777_177722


namespace sin_square_pi_over_4_l1777_177799

theorem sin_square_pi_over_4 (β : ℝ) (h : Real.sin (2 * β) = 2 / 3) : 
  Real.sin (β + π/4) ^ 2 = 5 / 6 :=
by
  sorry

end sin_square_pi_over_4_l1777_177799


namespace range_of_a_for_common_tangents_l1777_177767

theorem range_of_a_for_common_tangents :
  ∃ (a : ℝ), ∀ (x y : ℝ),
    ((x - 2)^2 + y^2 = 4) ∧ ((x - a)^2 + (y + 3)^2 = 9) →
    (-2 < a) ∧ (a < 6) := by
  sorry

end range_of_a_for_common_tangents_l1777_177767


namespace find_t_l1777_177762

def vector (α : Type) : Type := (α × α)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_t (t : ℝ) :
  let a : vector ℝ := (1, -1)
  let b : vector ℝ := (2, t)
  orthogonal a b → t = 2 := by
  sorry

end find_t_l1777_177762


namespace cos_difference_identity_l1777_177723

theorem cos_difference_identity (α : ℝ)
  (h : Real.sin (α + π / 6) + Real.cos α = - (Real.sqrt 3) / 3) :
  Real.cos (π / 6 - α) = -1 / 3 := 
sorry

end cos_difference_identity_l1777_177723


namespace arithmetic_sequence_sum_l1777_177735

-- Definitions used in the conditions
variable (a : ℕ → ℕ)
variable (n : ℕ)
variable (a_seq : Prop)
-- Declaring the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

noncomputable def a_5_is_2 : Prop := a 5 = 2

-- The statement we need to prove
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith_seq : is_arithmetic_sequence a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 := by
sorry

end arithmetic_sequence_sum_l1777_177735


namespace arithmetic_sequence_a13_l1777_177778

variable (a1 d : ℤ)

theorem arithmetic_sequence_a13 (h : a1 + 2 * d + a1 + 8 * d + a1 + 26 * d = 12) : a1 + 12 * d = 4 :=
by
  sorry

end arithmetic_sequence_a13_l1777_177778


namespace mans_speed_against_current_l1777_177785

/-- Given the man's speed with the current and the speed of the current, prove the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ) (speed_of_current : ℝ)
  (h1 : speed_with_current = 16)
  (h2 : speed_of_current = 3.2) :
  speed_with_current - 2 * speed_of_current = 9.6 :=
sorry

end mans_speed_against_current_l1777_177785


namespace a_value_intersection_l1777_177708

open Set

noncomputable def a_intersection_problem (a : ℝ) : Prop :=
  let A := { x : ℝ | x^2 < a^2 }
  let B := { x : ℝ | 1 < x ∧ x < 3 }
  let C := { x : ℝ | 1 < x ∧ x < 2 }
  A ∩ B = C → (a = 2 ∨ a = -2)

-- The theorem statement corresponding to the problem
theorem a_value_intersection (a : ℝ) :
  a_intersection_problem a :=
sorry

end a_value_intersection_l1777_177708


namespace percentage_increase_in_consumption_l1777_177737

-- Define the conditions
variables {T C : ℝ}  -- T: original tax, C: original consumption
variables (P : ℝ)    -- P: percentage increase in consumption

-- Non-zero conditions
variables (hT : T ≠ 0) (hC : C ≠ 0)

-- Define the Lean theorem
theorem percentage_increase_in_consumption 
  (h : 0.8 * (1 + P / 100) = 0.96) : 
  P = 20 :=
by
  sorry

end percentage_increase_in_consumption_l1777_177737


namespace remainder_of_power_mod_l1777_177755

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_power_mod_l1777_177755


namespace vehicle_height_limit_l1777_177704

theorem vehicle_height_limit (h : ℝ) (sign : String) (cond : sign = "Height Limit 4.5 meters") : h ≤ 4.5 :=
sorry

end vehicle_height_limit_l1777_177704


namespace largest_number_is_B_l1777_177736

-- Define the numbers as constants
def A : ℝ := 0.989
def B : ℝ := 0.998
def C : ℝ := 0.981
def D : ℝ := 0.899
def E : ℝ := 0.9801

-- State the theorem that B is the largest number
theorem largest_number_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  -- By comparison
  sorry

end largest_number_is_B_l1777_177736


namespace dhoni_savings_l1777_177734

theorem dhoni_savings :
  let earnings := 100
  let rent := 0.25 * earnings
  let dishwasher := rent - (0.10 * rent)
  let utilities := 0.15 * earnings
  let groceries := 0.20 * earnings
  let transportation := 0.12 * earnings
  let total_spent := rent + dishwasher + utilities + groceries + transportation
  earnings - total_spent = 0.055 * earnings :=
by
  sorry

end dhoni_savings_l1777_177734


namespace area_of_new_shape_l1777_177777

noncomputable def unit_equilateral_triangle_area : ℝ :=
  (1 : ℝ)^2 * Real.sqrt 3 / 4

noncomputable def area_removed_each_step (k : ℕ) : ℝ :=
  3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def total_removed_area : ℝ :=
  ∑' k, 3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def final_area := unit_equilateral_triangle_area - total_removed_area

theorem area_of_new_shape :
  final_area = Real.sqrt 3 / 10 := sorry

end area_of_new_shape_l1777_177777


namespace find_other_number_l1777_177741

def a : ℝ := 0.5
def d : ℝ := 0.16666666666666669
def b : ℝ := 0.3333333333333333

theorem find_other_number : a - d = b := by
  sorry

end find_other_number_l1777_177741


namespace part1_part2_l1777_177752

-- Part 1: Proving the value of a given f(x) = a/x + 1 and f(-2) = 0
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a / x + 1) (h2 : f (-2) = 0) : a = 2 := 
by 
-- Placeholder for the proof
sorry

-- Part 2: Proving the value of f(4) given f(x) = 6/x + 1
theorem part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = 6 / x + 1) : f 4 = 5 / 2 := 
by 
-- Placeholder for the proof
sorry

end part1_part2_l1777_177752


namespace find_number_l1777_177749

theorem find_number (x : ℝ) (h : 0.15 * 40 = 0.25 * x + 2) : x = 16 :=
by
  sorry

end find_number_l1777_177749


namespace polygon_quadrilateral_l1777_177756

theorem polygon_quadrilateral {n : ℕ} (h : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_quadrilateral_l1777_177756


namespace art_club_activity_l1777_177761

theorem art_club_activity (n p s b : ℕ) (h1 : n = 150) (h2 : p = 80) (h3 : s = 60) (h4 : b = 20) :
  (n - (p + s - b) = 30) :=
by
  sorry

end art_club_activity_l1777_177761


namespace students_arrangement_l1777_177732

theorem students_arrangement (B1 B2 S1 S2 T1 T2 C1 C2 : ℕ) :
  (B1 = B2 ∧ S1 ≠ S2 ∧ T1 ≠ T2 ∧ C1 ≠ C2) →
  (C1 ≠ C2) →
  (arrangements = 7200) :=
by
  sorry

end students_arrangement_l1777_177732


namespace combinations_count_l1777_177753

def colorChoices := 4
def decorationChoices := 3
def methodChoices := 3

theorem combinations_count : colorChoices * decorationChoices * methodChoices = 36 := by
  sorry

end combinations_count_l1777_177753


namespace triangle_centers_exist_l1777_177784

structure Triangle (α : Type _) [OrderedCommSemiring α] :=
(A B C : α × α)

noncomputable def circumcenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def incenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def excenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def centroid {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

theorem triangle_centers_exist {α : Type _} [OrderedCommSemiring α] (T : Triangle α) :
  ∃ K O Oc S : α × α, K = circumcenter T ∧ O = incenter T ∧ Oc = excenter T ∧ S = centroid T :=
by
  refine ⟨circumcenter T, incenter T, excenter T, centroid T, ⟨rfl, rfl, rfl, rfl⟩⟩

end triangle_centers_exist_l1777_177784


namespace triangle_side_lengths_inequality_iff_l1777_177725

theorem triangle_side_lengths_inequality_iff :
  {x : ℕ | 7 < x^2 ∧ x^2 < 17} = {3, 4} :=
by
  sorry

end triangle_side_lengths_inequality_iff_l1777_177725


namespace pure_alcohol_to_add_l1777_177773

-- Variables and known values
variables (x : ℝ) -- amount of pure alcohol added
def initial_volume : ℝ := 6 -- initial solution volume in liters
def initial_concentration : ℝ := 0.35 -- initial alcohol concentration
def target_concentration : ℝ := 0.50 -- target alcohol concentration

-- Conditions
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Statement of the problem
theorem pure_alcohol_to_add :
  (2.1 + x) / (initial_volume + x) = target_concentration ↔ x = 1.8 :=
by
  sorry

end pure_alcohol_to_add_l1777_177773


namespace calculate_expression_l1777_177781

theorem calculate_expression : 
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := 
by
  sorry

end calculate_expression_l1777_177781


namespace number_of_girls_l1777_177731

theorem number_of_girls
  (B G : ℕ)
  (ratio_condition : B * 8 = 5 * G)
  (total_condition : B + G = 260) :
  G = 160 :=
by
  sorry

end number_of_girls_l1777_177731


namespace no_odd_integer_trinomial_has_root_1_over_2022_l1777_177715

theorem no_odd_integer_trinomial_has_root_1_over_2022 :
  ¬ ∃ (a b c : ℤ), (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0)) :=
by
  sorry

end no_odd_integer_trinomial_has_root_1_over_2022_l1777_177715


namespace cricket_bat_cost_l1777_177743

variable (CP_A : ℝ) (CP_B : ℝ) (CP_C : ℝ)

-- Conditions
def CP_B_def : Prop := CP_B = 1.20 * CP_A
def CP_C_def : Prop := CP_C = 1.25 * CP_B
def CP_C_val : Prop := CP_C = 234

-- Theorem statement
theorem cricket_bat_cost (h1 : CP_B_def CP_A CP_B) (h2 : CP_C_def CP_B CP_C) (h3 : CP_C_val CP_C) : CP_A = 156 :=by
  sorry

end cricket_bat_cost_l1777_177743


namespace linda_age_l1777_177712

theorem linda_age
  (j k l : ℕ)       -- Ages of Jane, Kevin, and Linda respectively
  (h1 : j + k + l = 36)    -- Condition 1: j + k + l = 36
  (h2 : l - 3 = j)         -- Condition 2: l - 3 = j
  (h3 : k + 4 = (1 / 2 : ℝ) * (l + 4))  -- Condition 3: k + 4 = 1/2 * (l + 4)
  : l = 16 := 
sorry

end linda_age_l1777_177712


namespace inverse_matrix_l1777_177728

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end inverse_matrix_l1777_177728


namespace expression_in_parentheses_l1777_177710

theorem expression_in_parentheses (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) :
  ∃ expr : ℝ, xy * expr = -x^3 * y^2 ∧ expr = -x^2 * y :=
by
  sorry

end expression_in_parentheses_l1777_177710


namespace cube_painting_l1777_177763

theorem cube_painting (n : ℕ) (h1 : n > 3) 
  (h2 : 2 * (n-2) * (n-2) = 4 * (n-2)) :
  n = 4 :=
sorry

end cube_painting_l1777_177763


namespace average_loss_per_loot_box_l1777_177765

theorem average_loss_per_loot_box
  (cost_per_loot_box : ℝ := 5)
  (value_standard_item : ℝ := 3.5)
  (probability_rare_item_A : ℝ := 0.05)
  (value_rare_item_A : ℝ := 10)
  (probability_rare_item_B : ℝ := 0.03)
  (value_rare_item_B : ℝ := 15)
  (probability_rare_item_C : ℝ := 0.02)
  (value_rare_item_C : ℝ := 20) 
  : (cost_per_loot_box 
      - (0.90 * value_standard_item 
      + probability_rare_item_A * value_rare_item_A 
      + probability_rare_item_B * value_rare_item_B 
      + probability_rare_item_C * value_rare_item_C)) = 0.50 := by 
  sorry

end average_loss_per_loot_box_l1777_177765


namespace proof_by_contradiction_example_l1777_177796

theorem proof_by_contradiction_example (a b c : ℝ) (h : a < 3 ∧ b < 3 ∧ c < 3) : a < 1 ∨ b < 1 ∨ c < 1 := 
by
  have h1 : a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 := sorry
  sorry

end proof_by_contradiction_example_l1777_177796


namespace num_arithmetic_sequences_l1777_177705

theorem num_arithmetic_sequences (d : ℕ) (x : ℕ)
  (h_sum : 8 * x + 28 * d = 1080)
  (h_no180 : ∀ i, x + i * d ≠ 180)
  (h_pos : ∀ i, 0 < x + i * d)
  (h_less160 : ∀ i, x + i * d < 160)
  (h_not_equiangular : d ≠ 0) :
  ∃ n : ℕ, n = 3 :=
by sorry

end num_arithmetic_sequences_l1777_177705


namespace smallest_number_greater_than_l1777_177740

theorem smallest_number_greater_than : 
  ∀ (S : Set ℝ), S = {0.8, 0.5, 0.3} → 
  (∃ x ∈ S, x > 0.4 ∧ (∀ y ∈ S, y > 0.4 → x ≤ y)) → 
  x = 0.5 :=
by
  sorry

end smallest_number_greater_than_l1777_177740


namespace triangle_area_AC_1_AD_BC_circumcircle_l1777_177702

noncomputable def area_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_AC_1_AD_BC_circumcircle (A B C D E : ℝ × ℝ) (hAC : dist A C = 1)
  (hAD : dist A D = (2 / 3) * dist A B)
  (hMidE : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (hCircum : dist E ((A.1 + C.1) / 2, (A.2 + C.2) / 2) = 1 / 2) :
  area_triangle_ABC A B C = (Real.sqrt 5) / 6 :=
by
  sorry

end triangle_area_AC_1_AD_BC_circumcircle_l1777_177702


namespace dan_initial_money_l1777_177729

theorem dan_initial_money 
  (cost_chocolate : ℕ) 
  (cost_candy_bar : ℕ) 
  (h1 : cost_chocolate = 3) 
  (h2 : cost_candy_bar = 7)
  (h3 : cost_candy_bar - cost_chocolate = 4) : 
  cost_candy_bar + cost_chocolate = 10 := 
by
  sorry

end dan_initial_money_l1777_177729


namespace ice_cream_cones_sold_l1777_177744

theorem ice_cream_cones_sold (T W : ℕ) (h1 : W = 2 * T) (h2 : T + W = 36000) : T = 12000 :=
by
  sorry

end ice_cream_cones_sold_l1777_177744


namespace ab_value_l1777_177717

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end ab_value_l1777_177717


namespace factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l1777_177766

theorem factorize_3x_squared_minus_7x_minus_6 (x : ℝ) :
  3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2) :=
sorry

theorem factorize_6x_squared_minus_7x_minus_5 (x : ℝ) :
  6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5) :=
sorry

end factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l1777_177766


namespace solution_set_of_abs_inequality_l1777_177730

theorem solution_set_of_abs_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < 2.5 :=
by
  sorry

end solution_set_of_abs_inequality_l1777_177730


namespace relationship_between_vars_l1777_177776

-- Define the variables a, b, c, d as real numbers
variables (a b c d : ℝ)

-- Define the initial condition
def initial_condition := (a + 2 * b) / (2 * b + c) = (c + 2 * d) / (2 * d + a)

-- State the theorem to be proved
theorem relationship_between_vars (h : initial_condition a b c d) : 
  a = c ∨ a + c + 2 * (b + d) = 0 :=
sorry

end relationship_between_vars_l1777_177776


namespace complex_pow_sub_eq_zero_l1777_177721

namespace complex_proof

open Complex

def i : ℂ := Complex.I -- Defining i to be the imaginary unit

-- Stating the conditions as definitions
def condition := i^2 = -1

-- Stating the goal as a theorem
theorem complex_pow_sub_eq_zero (cond : condition) :
  (1 + 2 * i) ^ 24 - (1 - 2 * i) ^ 24 = 0 := 
by
  sorry

end complex_proof

end complex_pow_sub_eq_zero_l1777_177721


namespace smallest_n_boxes_l1777_177745

theorem smallest_n_boxes (n : ℕ) : (15 * n - 1) % 11 = 0 ↔ n = 3 :=
by
  sorry

end smallest_n_boxes_l1777_177745


namespace positive_real_solutions_l1777_177720

noncomputable def x1 := (75 + Real.sqrt 5773) / 2
noncomputable def x2 := (-50 + Real.sqrt 2356) / 2

theorem positive_real_solutions :
  ∀ x : ℝ, 
  0 < x → 
  (1/2 * (4*x^2 - 1) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10)) ↔ 
  (x = x1 ∨ x = x2) :=
by
  sorry

end positive_real_solutions_l1777_177720


namespace bake_sale_earnings_eq_400_l1777_177764

/-
  The problem statement derived from the given bake sale problem.
  We are to verify that the bake sale earned 400 dollars.
-/

def total_donation (bake_sale_earnings : ℕ) :=
  ((bake_sale_earnings - 100) / 2) + 10

theorem bake_sale_earnings_eq_400 (X : ℕ) (h : total_donation X = 160) : X = 400 :=
by
  sorry

end bake_sale_earnings_eq_400_l1777_177764


namespace scientific_notation_of_21500000_l1777_177733

theorem scientific_notation_of_21500000 :
  21500000 = 2.15 * 10^7 :=
by
  sorry

end scientific_notation_of_21500000_l1777_177733


namespace min_2x3y2z_l1777_177768

noncomputable def min_value (x y z : ℝ) : ℝ := 2 * (x^3) * (y^2) * z

theorem min_2x3y2z (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h : (1/x) + (1/y) + (1/z) = 9) :
  min_value x y z = 2 / 675 :=
sorry

end min_2x3y2z_l1777_177768


namespace find_a_minus_inverse_l1777_177701

-- Definition for the given condition
def condition (a : ℝ) : Prop := a + a⁻¹ = 6

-- Definition for the target value to be proven
def target_value (x : ℝ) : Prop := x = 4 * Real.sqrt 2 ∨ x = -4 * Real.sqrt 2

-- Theorem statement to be proved
theorem find_a_minus_inverse (a : ℝ) (ha : condition a) : target_value (a - a⁻¹) :=
by
  sorry

end find_a_minus_inverse_l1777_177701


namespace find_A_l1777_177795

theorem find_A : ∃ (A : ℕ), 
  (A > 0) ∧ (A ∣ (270 * 2 - 312)) ∧ (A ∣ (211 * 2 - 270)) ∧ 
  (∃ (rA rB rC : ℕ), 312 % A = rA ∧ 270 % A = rB ∧ 211 % A = rC ∧ 
                      rA = 2 * rB ∧ rB = 2 * rC ∧ A = 19) :=
by sorry

end find_A_l1777_177795


namespace tetrahedron_distance_sum_eq_l1777_177727

-- Defining the necessary conditions
variables {V K : ℝ}
variables {S_1 S_2 S_3 S_4 H_1 H_2 H_3 H_4 : ℝ}

axiom ratio_eq (i : ℕ) (Si : ℝ) (K : ℝ) : (Si / i = K)
axiom volume_eq : S_1 * H_1 + S_2 * H_2 + S_3 * H_3 + S_4 * H_4 = 3 * V

-- Main theorem stating that the desired result holds under the given conditions
theorem tetrahedron_distance_sum_eq :
  H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4 = 3 * V / K :=
by
have h1 : S_1 = K * 1 := by sorry
have h2 : S_2 = K * 2 := by sorry
have h3 : S_3 = K * 3 := by sorry
have h4 : S_4 = K * 4 := by sorry
have sum_eq : K * (H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4) = 3 * V := by sorry
exact sorry

end tetrahedron_distance_sum_eq_l1777_177727


namespace _l1777_177703

lemma right_triangle_angles (AB BC AC : ℝ) (α β : ℝ)
  (h1 : AB = 1) 
  (h2 : BC = Real.sin α)
  (h3 : AC = Real.cos α)
  (h4 : AB^2 = BC^2 + AC^2) -- Pythagorean theorem for the right triangle
  (h5 : α = (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1))) :
  β = 90 - (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) :=
sorry

end _l1777_177703


namespace complex_quadrant_l1777_177760

-- Define the imaginary unit
def i := Complex.I

-- Define the complex number z satisfying the given condition
variables (z : Complex)
axiom h : (3 - 2 * i) * z = 4 + 3 * i

-- Statement for the proof problem
theorem complex_quadrant (h : (3 - 2 * i) * z = 4 + 3 * i) : 
  (0 < z.re ∧ 0 < z.im) :=
sorry

end complex_quadrant_l1777_177760


namespace books_in_bin_after_transactions_l1777_177772

def initial_books : ℕ := 4
def sold_books : ℕ := 3
def added_books : ℕ := 10

def final_books (initial_books sold_books added_books : ℕ) : ℕ :=
  initial_books - sold_books + added_books

theorem books_in_bin_after_transactions :
  final_books initial_books sold_books added_books = 11 := by
  sorry

end books_in_bin_after_transactions_l1777_177772


namespace find_r_l1777_177775

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 20.7) : r = 10.7 := 
by 
  sorry 

end find_r_l1777_177775


namespace Kira_breakfast_time_l1777_177700

theorem Kira_breakfast_time :
  let sausages := 3
  let eggs := 6
  let time_per_sausage := 5
  let time_per_egg := 4
  (sausages * time_per_sausage + eggs * time_per_egg) = 39 :=
by
  sorry

end Kira_breakfast_time_l1777_177700


namespace sum_of_squares_divisible_by_sum_l1777_177759

theorem sum_of_squares_divisible_by_sum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h_bound : a < 2017 ∧ b < 2017 ∧ c < 2017)
    (h_mod : (a^3 - b^3) % 2017 = 0 ∧ (b^3 - c^3) % 2017 = 0 ∧ (c^3 - a^3) % 2017 = 0) :
    (a^2 + b^2 + c^2) % (a + b + c) = 0 :=
by
  sorry

end sum_of_squares_divisible_by_sum_l1777_177759


namespace sphere_radius_volume_eq_surface_area_l1777_177787

theorem sphere_radius_volume_eq_surface_area (r : ℝ) (h₁ : (4 / 3) * π * r^3 = 4 * π * r^2) : r = 3 :=
by
  sorry

end sphere_radius_volume_eq_surface_area_l1777_177787


namespace position_of_seventeen_fifteen_in_sequence_l1777_177711

theorem position_of_seventeen_fifteen_in_sequence :
  ∃ n : ℕ, (17 : ℚ) / 15 = (n + 3 : ℚ) / (n + 1) :=
sorry

end position_of_seventeen_fifteen_in_sequence_l1777_177711


namespace arithmetic_sequences_ratio_l1777_177738

theorem arithmetic_sequences_ratio (x y a1 a2 a3 b1 b2 b3 b4 : Real) (hxy : x ≠ y) 
  (h_arith1 : a1 = x + (y - x) / 4 ∧ a2 = x + 2 * (y - x) / 4 ∧ a3 = x + 3 * (y - x) / 4 ∧ y = x + 4 * (y - x) / 4)
  (h_arith2 : b1 = x - (y - x) / 2 ∧ b2 = x + (y - x) / 2 ∧ b3 = x + 2 * (y - x) / 2 ∧ y = x + 2 * (y - x) / 2 ∧ b4 = y + (y - x) / 2):
  (b4 - b3) / (a2 - a1) = 8 / 3 := 
sorry

end arithmetic_sequences_ratio_l1777_177738


namespace eval_expression_l1777_177709

def x : ℤ := 18 / 3 * 7^2 - 80 + 4 * 7

theorem eval_expression : -x = -242 := by
  sorry

end eval_expression_l1777_177709


namespace correct_operation_l1777_177716

theorem correct_operation (x : ℝ) (f : ℝ → ℝ) (h : ∀ x, (x / 10) = 0.01 * f x) : 
  f x = 10 * x :=
by
  sorry

end correct_operation_l1777_177716


namespace side_length_is_prime_l1777_177747

-- Define the integer side length of the square
variable (a : ℕ)

-- Define the conditions
def impossible_rectangle (m n : ℕ) : Prop :=
  m * n = a^2 ∧ m ≠ 1 ∧ n ≠ 1

-- Declare the theorem to be proved
theorem side_length_is_prime (h : ∀ m n : ℕ, impossible_rectangle a m n → false) : Nat.Prime a := sorry

end side_length_is_prime_l1777_177747


namespace cost_of_paving_floor_l1777_177771

-- Conditions
def length_of_room : ℝ := 8
def width_of_room : ℝ := 4.75
def rate_per_sq_metre : ℝ := 900

-- Statement to prove
theorem cost_of_paving_floor : (length_of_room * width_of_room * rate_per_sq_metre) = 34200 :=
by
  sorry

end cost_of_paving_floor_l1777_177771


namespace water_tank_equilibrium_l1777_177782

theorem water_tank_equilibrium :
  (1 / 15 : ℝ) + (1 / 10 : ℝ) - (1 / 6 : ℝ) = 0 :=
by
  sorry

end water_tank_equilibrium_l1777_177782


namespace value_of_x_l1777_177774

theorem value_of_x (x : ℝ) (h : x = 88 + 0.3 * 88) : x = 114.4 :=
by
  sorry

end value_of_x_l1777_177774


namespace contingency_fund_correct_l1777_177797

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l1777_177797


namespace find_fraction_l1777_177757

theorem find_fraction :
  ∀ (t k : ℝ) (frac : ℝ),
    t = frac * (k - 32) →
    t = 20 → 
    k = 68 → 
    frac = 5 / 9 :=
by
  intro t k frac h_eq h_t h_k
  -- Start from the conditions and end up showing frac = 5/9
  sorry

end find_fraction_l1777_177757


namespace scientific_notation_example_l1777_177739

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (3650000 : ℝ) = a * 10 ^ n :=
sorry

end scientific_notation_example_l1777_177739


namespace number_of_cases_in_top_level_l1777_177788

-- Definitions for the total number of soda cases
def pyramid_cases (n : ℕ) : ℕ :=
  n^2 + (n + 1)^2 + (n + 2)^2 + (n + 3)^2

-- Theorem statement: proving the number of cases in the top level
theorem number_of_cases_in_top_level (n : ℕ) (h : pyramid_cases n = 30) : n = 1 :=
by {
  sorry
}

end number_of_cases_in_top_level_l1777_177788


namespace sin_value_l1777_177724

theorem sin_value (α : ℝ) (h : Real.cos (α + π / 6) = - (Real.sqrt 2) / 10) : 
  Real.sin (2 * α - π / 6) = 24 / 25 :=
by
  sorry

end sin_value_l1777_177724


namespace largest_perfect_square_factor_of_882_l1777_177792

theorem largest_perfect_square_factor_of_882 : ∃ n, n * n = 441 ∧ ∀ m, m * m ∣ 882 → m * m ≤ 441 := 
by 
 sorry

end largest_perfect_square_factor_of_882_l1777_177792


namespace part1_part2_l1777_177751

noncomputable def problem1 (x y: ℕ) : Prop := 
  (2 * x + 3 * y = 44) ∧ (4 * x = 5 * y)

noncomputable def solution1 (x y: ℕ) : Prop :=
  (x = 10) ∧ (y = 8)

theorem part1 : ∃ x y: ℕ, problem1 x y → solution1 x y :=
by sorry

noncomputable def problem2 (a b: ℕ) : Prop := 
  25 * (10 * a + 8 * b) = 3500

noncomputable def solution2 (a b: ℕ) : Prop :=
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5))

theorem part2 : ∃ a b: ℕ, problem2 a b → solution2 a b :=
by sorry

end part1_part2_l1777_177751


namespace cost_price_l1777_177714

namespace ClothingDiscount

variables (x : ℝ)

def loss_condition (x : ℝ) : ℝ := 0.5 * x + 20
def profit_condition (x : ℝ) : ℝ := 0.8 * x - 40

def marked_price := { x : ℝ // loss_condition x = profit_condition x }

noncomputable def clothing_price : marked_price := 
    ⟨200, sorry⟩

theorem cost_price : loss_condition 200 = 120 :=
sorry

end ClothingDiscount

end cost_price_l1777_177714


namespace positive_number_sum_square_l1777_177718

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l1777_177718


namespace dave_final_tickets_l1777_177783

variable (initial_tickets_set1_won : ℕ) (initial_tickets_set1_lost : ℕ)
variable (initial_tickets_set2_won : ℕ) (initial_tickets_set2_lost : ℕ)
variable (multiplier_set3 : ℕ)
variable (initial_tickets_set3_lost : ℕ)
variable (used_tickets : ℕ)
variable (additional_tickets : ℕ)

theorem dave_final_tickets :
  let net_gain_set1 := initial_tickets_set1_won - initial_tickets_set1_lost
  let net_gain_set2 := initial_tickets_set2_won - initial_tickets_set2_lost
  let net_gain_set3 := multiplier_set3 * net_gain_set1 - initial_tickets_set3_lost
  let total_tickets_after_sets := net_gain_set1 + net_gain_set2 + net_gain_set3
  let tickets_after_buying := total_tickets_after_sets - used_tickets
  let final_tickets := tickets_after_buying + additional_tickets
  initial_tickets_set1_won = 14 →
  initial_tickets_set1_lost = 2 →
  initial_tickets_set2_won = 8 →
  initial_tickets_set2_lost = 5 →
  multiplier_set3 = 3 →
  initial_tickets_set3_lost = 15 →
  used_tickets = 25 →
  additional_tickets = 7 →
  final_tickets = 18 :=
by
  intros
  sorry

end dave_final_tickets_l1777_177783


namespace total_cost_is_103_l1777_177791

-- Base cost of the plan is 20 dollars
def base_cost : ℝ := 20

-- Cost per text message in dollars
def cost_per_text : ℝ := 0.10

-- Cost per minute over 25 hours in dollars
def cost_per_minute_over_limit : ℝ := 0.15

-- Number of text messages sent
def text_messages : ℕ := 200

-- Total hours talked
def hours_talked : ℝ := 32

-- Free minutes (25 hours)
def free_minutes : ℝ := 25 * 60

-- Calculating the extra minutes talked
def extra_minutes : ℝ := (hours_talked * 60) - free_minutes

-- Total cost
def total_cost : ℝ :=
  base_cost +
  (text_messages * cost_per_text) +
  (extra_minutes * cost_per_minute_over_limit)

-- Proving that the total cost is 103 dollars
theorem total_cost_is_103 : total_cost = 103 := by
  sorry

end total_cost_is_103_l1777_177791
