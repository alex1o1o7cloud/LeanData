import Mathlib

namespace gina_snake_mice_in_decade_l1810_181001

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end gina_snake_mice_in_decade_l1810_181001


namespace smallest_x_satisfying_expression_l1810_181038

theorem smallest_x_satisfying_expression :
  ∃ x : ℤ, (∃ k : ℤ, x^2 + x + 7 = k * (x - 2)) ∧ (∀ y : ℤ, (∃ k' : ℤ, y^2 + y + 7 = k' * (y - 2)) → y ≥ x) ∧ x = -11 :=
by
  sorry

end smallest_x_satisfying_expression_l1810_181038


namespace strongest_erosive_power_l1810_181089

-- Definition of the options
inductive Period where
  | MayToJune : Period
  | JuneToJuly : Period
  | JulyToAugust : Period
  | AugustToSeptember : Period

-- Definition of the eroding power function (stub)
def erosivePower : Period → ℕ
| Period.MayToJune => 1
| Period.JuneToJuly => 2
| Period.JulyToAugust => 3
| Period.AugustToSeptember => 1

-- Statement that July to August has the maximum erosive power
theorem strongest_erosive_power : erosivePower Period.JulyToAugust = 3 := 
by 
  sorry

end strongest_erosive_power_l1810_181089


namespace present_cost_after_two_years_l1810_181021

-- Defining variables and constants
def initial_cost : ℝ := 75
def inflation_rate : ℝ := 0.05
def first_year_increase1 : ℝ := 0.20
def first_year_decrease1 : ℝ := 0.20
def second_year_increase2 : ℝ := 0.30
def second_year_decrease2 : ℝ := 0.25

theorem present_cost_after_two_years : presents_cost = 77.40 :=
by
  let adjusted_initial_cost := initial_cost + (initial_cost * inflation_rate)
  let increased_cost_year1 := adjusted_initial_cost + (adjusted_initial_cost * first_year_increase1)
  let decreased_cost_year1 := increased_cost_year1 - (increased_cost_year1 * first_year_decrease1)
  let adjusted_cost_year1 := decreased_cost_year1 + (decreased_cost_year1 * inflation_rate)
  let increased_cost_year2 := adjusted_cost_year1 + (adjusted_cost_year1 * second_year_increase2)
  let decreased_cost_year2 := increased_cost_year2 - (increased_cost_year2 * second_year_decrease2)
  let presents_cost := decreased_cost_year2
  have h := (presents_cost : ℝ)
  have h := presents_cost
  sorry

end present_cost_after_two_years_l1810_181021


namespace central_angle_of_sector_l1810_181092

theorem central_angle_of_sector (r S α : ℝ) (h1 : r = 10) (h2 : S = 100)
  (h3 : S = 1/2 * α * r^2) : α = 2 :=
by
  -- Given radius r and area S, substituting into the formula for the area of the sector,
  -- we derive the central angle α.
  sorry

end central_angle_of_sector_l1810_181092


namespace kelly_initially_had_l1810_181076

def kelly_needs_to_pick : ℕ := 49
def kelly_will_have : ℕ := 105

theorem kelly_initially_had :
  kelly_will_have - kelly_needs_to_pick = 56 :=
by
  sorry

end kelly_initially_had_l1810_181076


namespace number_of_rods_in_one_mile_l1810_181034

theorem number_of_rods_in_one_mile :
  (1 : ℤ) * 6 * 60 = 360 :=
by
  sorry

end number_of_rods_in_one_mile_l1810_181034


namespace interest_rate_first_part_eq_3_l1810_181052

variable (T P1 P2 r2 I : ℝ)
variable (hT : T = 3400)
variable (hP1 : P1 = 1300)
variable (hP2 : P2 = 2100)
variable (hr2 : r2 = 5)
variable (hI : I = 144)

theorem interest_rate_first_part_eq_3 (r : ℝ) (h : (P1 * r) / 100 + (P2 * r2) / 100 = I) : r = 3 :=
by
  -- leaning in the proof
  sorry

end interest_rate_first_part_eq_3_l1810_181052


namespace border_area_is_72_l1810_181020

def livingRoomLength : ℝ := 12
def livingRoomWidth : ℝ := 10
def borderWidth : ℝ := 2

def livingRoomArea : ℝ := livingRoomLength * livingRoomWidth
def carpetLength : ℝ := livingRoomLength - 2 * borderWidth
def carpetWidth : ℝ := livingRoomWidth - 2 * borderWidth
def carpetArea : ℝ := carpetLength * carpetWidth
def borderArea : ℝ := livingRoomArea - carpetArea

theorem border_area_is_72 : borderArea = 72 := 
by
  sorry

end border_area_is_72_l1810_181020


namespace first_cyclist_speed_l1810_181035

theorem first_cyclist_speed (v₁ v₂ : ℕ) (c t : ℕ) 
  (h1 : v₂ = 8) 
  (h2 : c = 675) 
  (h3 : t = 45) 
  (h4 : v₁ * t + v₂ * t = c) : 
  v₁ = 7 :=
by {
  sorry
}

end first_cyclist_speed_l1810_181035


namespace largest_k_no_perpendicular_lines_l1810_181054

theorem largest_k_no_perpendicular_lines (n : ℕ) (h : 0 < n) :
  (∃ k, ∀ (l : Fin n → ℝ) (f : Fin n), (∀ i j, i ≠ j → l i ≠ -1 / (l j)) → k = Nat.ceil (n / 2)) :=
sorry

end largest_k_no_perpendicular_lines_l1810_181054


namespace orange_juice_percentage_l1810_181048

theorem orange_juice_percentage 
  (V : ℝ) 
  (W : ℝ) 
  (G : ℝ)
  (hV : V = 300)
  (hW: W = 0.4 * V)
  (hG: G = 105) : 
  (V - W - G) / V * 100 = 25 := 
by 
  -- We will need to use sorry to skip the proof and focus just on the statement
  sorry

end orange_juice_percentage_l1810_181048


namespace smallest_x_value_l1810_181083

theorem smallest_x_value : ∀ x : ℚ, (14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 → x = 4 / 5 :=
by
  intros x hx
  sorry

end smallest_x_value_l1810_181083


namespace third_team_pies_l1810_181027

theorem third_team_pies (total first_team second_team : ℕ) (h_total : total = 750) (h_first : first_team = 235) (h_second : second_team = 275) :
  total - (first_team + second_team) = 240 := by
  sorry

end third_team_pies_l1810_181027


namespace least_range_product_multiple_840_l1810_181075

def is_multiple (x y : Nat) : Prop :=
  ∃ k : Nat, y = k * x

theorem least_range_product_multiple_840 : 
  ∃ (a : Nat), a > 0 ∧ ∀ (n : Nat), (n = 3) → is_multiple 840 (List.foldr (· * ·) 1 (List.range' a n)) := 
by {
  sorry
}

end least_range_product_multiple_840_l1810_181075


namespace sin_cos_identity_l1810_181064

theorem sin_cos_identity (α β γ : ℝ) (h : α + β + γ = 180) :
    Real.sin α + Real.sin β + Real.sin γ = 
    4 * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) := 
  sorry

end sin_cos_identity_l1810_181064


namespace find_k_l1810_181043

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end find_k_l1810_181043


namespace largest_consecutive_integers_sum_to_45_l1810_181057

theorem largest_consecutive_integers_sum_to_45 (x n : ℕ) (h : 45 = n * (2 * x + n - 1) / 2) : n ≤ 9 :=
sorry

end largest_consecutive_integers_sum_to_45_l1810_181057


namespace complement_of_set_A_is_34_l1810_181049

open Set

noncomputable def U : Set ℕ := {n : ℕ | True}

noncomputable def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Complement of A in U
noncomputable def C_U_A : Set ℕ := U \ A

theorem complement_of_set_A_is_34 : C_U_A = {3, 4} :=
by sorry

end complement_of_set_A_is_34_l1810_181049


namespace equation_represents_circle_of_radius_8_l1810_181055

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end equation_represents_circle_of_radius_8_l1810_181055


namespace total_cost_of_constructing_the_path_l1810_181019

open Real

-- Define the conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_path_given : ℝ := 1518.72
def cost_per_sq_m : ℝ := 2

-- Define the total cost to be proven
def total_cost : ℝ := 3037.44

-- The statement to be proven
theorem total_cost_of_constructing_the_path :
  let outer_length := length_field + 2 * path_width
  let outer_width := width_field + 2 * path_width
  let total_area_incl_path := outer_length * outer_width
  let area_field := length_field * width_field
  let computed_area_path := total_area_incl_path - area_field
  let given_cost := area_path_given * cost_per_sq_m
  total_cost = given_cost := by
  sorry

end total_cost_of_constructing_the_path_l1810_181019


namespace quadratic_has_two_roots_l1810_181009

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end quadratic_has_two_roots_l1810_181009


namespace units_digit_of_n_squared_plus_2_n_is_7_l1810_181070

def n : ℕ := 2023 ^ 2 + 2 ^ 2023

theorem units_digit_of_n_squared_plus_2_n_is_7 : (n ^ 2 + 2 ^ n) % 10 = 7 := 
by
  sorry

end units_digit_of_n_squared_plus_2_n_is_7_l1810_181070


namespace triangles_in_divided_square_l1810_181032

theorem triangles_in_divided_square (V : ℕ) (marked_points : ℕ) (triangles : ℕ) 
  (h1 : V = 24) -- Vertices - 20 marked points and 4 vertices 
  (h2 : marked_points = 20) -- Marked points
  (h3 : triangles = F - 1) -- Each face (F) except the outer one is a triangle
  (h4 : V - E + F = 2) -- Euler's formula for planar graphs
  (h5 : E = (3*F + 1) / 2) -- Relationship between edges and faces
  (F : ℕ) -- Number of faces including the external face
  (E : ℕ) -- Number of edges
  : triangles = 42 := 
by 
  sorry

end triangles_in_divided_square_l1810_181032


namespace arithmetic_sequence_num_terms_l1810_181091

theorem arithmetic_sequence_num_terms (a_1 d S_n n : ℕ) 
  (h1 : a_1 = 4) (h2 : d = 3) (h3 : S_n = 650)
  (h4 : S_n = (n / 2) * (2 * a_1 + (n - 1) * d)) : n = 20 := by
  sorry

end arithmetic_sequence_num_terms_l1810_181091


namespace find_other_denomination_l1810_181090

theorem find_other_denomination
  (total_spent : ℕ)
  (twenty_bill_value : ℕ) (other_denomination_value : ℕ)
  (twenty_bill_count : ℕ) (other_bill_count : ℕ)
  (h1 : total_spent = 80)
  (h2 : twenty_bill_value = 20)
  (h3 : other_bill_count = 2)
  (h4 : twenty_bill_count = other_bill_count + 1)
  (h5 : total_spent = twenty_bill_value * twenty_bill_count + other_denomination_value * other_bill_count) : 
  other_denomination_value = 10 :=
by
  sorry

end find_other_denomination_l1810_181090


namespace max_and_min_W_l1810_181028

noncomputable def W (x y z : ℝ) : ℝ := 2 * x + 6 * y + 4 * z

theorem max_and_min_W {x y z : ℝ} (h1 : x + y + z = 1) (h2 : 3 * y + z ≥ 2) (h3 : 0 ≤ x ∧ x ≤ 1) (h4 : 0 ≤ y ∧ y ≤ 2) :
  ∃ (W_max W_min : ℝ), W_max = 6 ∧ W_min = 4 :=
by
  sorry

end max_and_min_W_l1810_181028


namespace NutsInThirdBox_l1810_181024

variable (x y z : ℝ)

theorem NutsInThirdBox (h1 : x = (y + z) - 6) (h2 : y = (x + z) - 10) : z = 16 := 
sorry

end NutsInThirdBox_l1810_181024


namespace final_number_is_50_l1810_181006

theorem final_number_is_50 (initial_ones initial_fours : ℕ) (h1 : initial_ones = 900) (h2 : initial_fours = 100) :
  ∃ (z : ℝ), (900 * (1:ℝ)^2 + 100 * (4:ℝ)^2) = z^2 ∧ z = 50 :=
by
  sorry

end final_number_is_50_l1810_181006


namespace pairs_symmetry_l1810_181013

theorem pairs_symmetry (N : ℕ) (hN : N > 2) :
  ∃ f : {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 > 2} ≃ 
           {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 < 2}, 
  true :=
sorry

end pairs_symmetry_l1810_181013


namespace average_people_per_hour_rounded_l1810_181060

def people_moving_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let total_hours := days * hours_per_day
  (total_people / total_hours : ℕ)

theorem average_people_per_hour_rounded :
  people_moving_per_hour 4500 5 24 = 38 := 
  sorry

end average_people_per_hour_rounded_l1810_181060


namespace w_share_l1810_181063

theorem w_share (k : ℝ) (w x y z : ℝ) (h1 : w = k) (h2 : x = 6 * k) (h3 : y = 2 * k) (h4 : z = 4 * k) (h5 : x - y = 1500):
  w = 375 := by
  /- Lean code to show w = 375 -/
  sorry

end w_share_l1810_181063


namespace part1_part2_l1810_181039

def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}
def C : Set ℝ := {x | -1 < x ∧ x < 4}

theorem part1 : A ∩ (B 3)ᶜ = Set.Icc 3 5 := by
  sorry

theorem part2 : A ∩ B m = C → m = 8 := by
  sorry

end part1_part2_l1810_181039


namespace exists_not_perfect_square_l1810_181053

theorem exists_not_perfect_square (a b c : ℤ) : ∃ (n : ℕ), n > 0 ∧ ¬ ∃ k : ℕ, n^3 + a * n^2 + b * n + c = k^2 :=
by
  sorry

end exists_not_perfect_square_l1810_181053


namespace max_possible_value_xv_l1810_181077

noncomputable def max_xv_distance (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) : ℝ :=
|x - v|

theorem max_possible_value_xv 
  (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  max_xv_distance x y z w v h1 h2 h3 h4 = 11 :=
sorry

end max_possible_value_xv_l1810_181077


namespace total_savings_l1810_181005

theorem total_savings :
  let josiah_daily := 0.25 
  let josiah_days := 24 
  let leah_daily := 0.50 
  let leah_days := 20 
  let megan_multiplier := 2
  let megan_days := 12 
  let josiah_savings := josiah_daily * josiah_days 
  let leah_savings := leah_daily * leah_days 
  let megan_daily := megan_multiplier * leah_daily 
  let megan_savings := megan_daily * megan_days 
  let total_savings := josiah_savings + leah_savings + megan_savings 
  total_savings = 28 :=
by
  sorry

end total_savings_l1810_181005


namespace percentage_decrease_is_correct_l1810_181042

variable (P : ℝ)

-- Condition 1: After the first year, the price increased by 30%
def price_after_first_year : ℝ := 1.30 * P

-- Condition 2: At the end of the 2-year period, the price of the painting is 110.5% of the original price
def price_after_second_year : ℝ := 1.105 * P

-- Condition 3: Let D be the percentage decrease during the second year
def D : ℝ := 0.15

-- Goal: Prove that the percentage decrease during the second year is 15%
theorem percentage_decrease_is_correct : 
  1.30 * P - D * 1.30 * P = 1.105 * P → D = 0.15 :=
by
  sorry

end percentage_decrease_is_correct_l1810_181042


namespace tan_product_l1810_181080

theorem tan_product : 
(1 + Real.tan (Real.pi / 60)) * (1 + Real.tan (Real.pi / 30)) * (1 + Real.tan (Real.pi / 20)) * (1 + Real.tan (Real.pi / 15)) * (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 10)) * (1 + Real.tan (Real.pi / 9)) * (1 + Real.tan (Real.pi / 6)) = 2^8 :=
by
  sorry 

end tan_product_l1810_181080


namespace smallest_positive_integer_n_l1810_181046

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (n > 0 ∧ 17 * n % 7 = 2) ∧ ∀ m : ℕ, (m > 0 ∧ 17 * m % 7 = 2) → n ≤ m := 
sorry

end smallest_positive_integer_n_l1810_181046


namespace wire_length_l1810_181069

theorem wire_length (r_sphere r_wire : ℝ) (h : ℝ) (V : ℝ)
  (h₁ : r_sphere = 24) (h₂ : r_wire = 16)
  (h₃ : V = 4 / 3 * Real.pi * r_sphere ^ 3)
  (h₄ : V = Real.pi * r_wire ^ 2 * h): 
  h = 72 := by
  -- we can use provided condition to show that h = 72, proof details omitted
  sorry

end wire_length_l1810_181069


namespace circle_center_radius_l1810_181097

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 2 ∧ k = -1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l1810_181097


namespace elsa_ends_with_145_marbles_l1810_181044

theorem elsa_ends_with_145_marbles :
  let initial := 150
  let after_breakfast := initial - 7
  let after_lunch := after_breakfast - 57
  let after_afternoon := after_lunch + 25
  let after_evening := after_afternoon + 85
  let after_exchange := after_evening - 9 + 6
  let final := after_exchange - 48
  final = 145 := by
    sorry

end elsa_ends_with_145_marbles_l1810_181044


namespace ryan_learning_hours_l1810_181012

theorem ryan_learning_hours :
  ∀ (e c s : ℕ) , (e = 6) → (s = 58) → (e = c + 3) → (c = 3) :=
by
  intros e c s he hs hc
  sorry

end ryan_learning_hours_l1810_181012


namespace range_of_m_l1810_181022

-- Define the quadratic function f
def f (a c x : ℝ) := a * x^2 - 2 * a * x + c

-- State the theorem
theorem range_of_m (a c : ℝ) (h : f a c 2017 < f a c (-2016)) (m : ℝ) 
  : f a c m ≤ f a c 0 → 0 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l1810_181022


namespace part1_part2_l1810_181071

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l1810_181071


namespace intersection_complement_l1810_181099

-- Declare variables for sets
variable (I A B : Set ℤ)

-- Define the universal set I
def universal_set : Set ℤ := { x | -3 < x ∧ x < 3 }

-- Define sets A and B
def set_A : Set ℤ := { -2, 0, 1 }
def set_B : Set ℤ := { -1, 0, 1, 2 }

-- Main theorem statement
theorem intersection_complement
  (hI : I = universal_set)
  (hA : A = set_A)
  (hB : B = set_B) :
  B ∩ (I \ A) = { -1, 2 } :=
sorry

end intersection_complement_l1810_181099


namespace solve_system_l1810_181000

theorem solve_system :
  ∃ x y : ℤ, (x - 3 * y = 7) ∧ (5 * x + 2 * y = 1) ∧ (x = 1) ∧ (y = -2) :=
by
  sorry

end solve_system_l1810_181000


namespace Jakes_brother_has_more_l1810_181062

-- Define the number of comic books Jake has
def Jake_comics : ℕ := 36

-- Define the total number of comic books Jake and his brother have together
def total_comics : ℕ := 87

-- Prove Jake's brother has 15 more comic books than Jake
theorem Jakes_brother_has_more : ∃ B, B > Jake_comics ∧ B + Jake_comics = total_comics ∧ B - Jake_comics = 15 :=
by
  sorry

end Jakes_brother_has_more_l1810_181062


namespace card_probability_l1810_181087

-- Definitions to capture the problem's conditions in Lean
def total_cards : ℕ := 52
def remaining_after_first : ℕ := total_cards - 1
def remaining_after_second : ℕ := total_cards - 2

def kings : ℕ := 4
def non_heart_kings : ℕ := 3
def non_kings_in_hearts : ℕ := 12
def spades_and_diamonds : ℕ := 26

-- Define probabilities for each step
def prob_first_king : ℚ := non_heart_kings / total_cards
def prob_second_heart : ℚ := non_kings_in_hearts / remaining_after_first
def prob_third_spade_or_diamond : ℚ := spades_and_diamonds / remaining_after_second

-- Calculate total probability
def total_probability : ℚ := prob_first_king * prob_second_heart * prob_third_spade_or_diamond

-- Theorem statement that encapsulates the problem
theorem card_probability : total_probability = 26 / 3675 :=
by sorry

end card_probability_l1810_181087


namespace total_cookies_l1810_181074

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l1810_181074


namespace find_some_number_l1810_181050

theorem find_some_number (n m : ℕ) (h : (n / 20) * (n / m) = 1) (n_eq_40 : n = 40) : m = 2 :=
by
  sorry

end find_some_number_l1810_181050


namespace tank_empties_in_4320_minutes_l1810_181086

-- Define the initial conditions
def tankVolumeCubicFeet: ℝ := 30
def inletPipeRateCubicInchesPerMin: ℝ := 5
def outletPipe1RateCubicInchesPerMin: ℝ := 9
def outletPipe2RateCubicInchesPerMin: ℝ := 8
def feetToInches: ℝ := 12

-- Conversion from cubic feet to cubic inches
def tankVolumeCubicInches: ℝ := tankVolumeCubicFeet * feetToInches^3

-- Net rate of emptying in cubic inches per minute
def netRateOfEmptying: ℝ := (outletPipe1RateCubicInchesPerMin + outletPipe2RateCubicInchesPerMin) - inletPipeRateCubicInchesPerMin

-- Time to empty the tank
noncomputable def timeToEmptyTank: ℝ := tankVolumeCubicInches / netRateOfEmptying

-- The theorem to prove
theorem tank_empties_in_4320_minutes :
  timeToEmptyTank = 4320 := by
  sorry

end tank_empties_in_4320_minutes_l1810_181086


namespace strawberry_candies_count_l1810_181096

theorem strawberry_candies_count (S G : ℕ) (h1 : S + G = 240) (h2 : G = S - 2) : S = 121 :=
by
  sorry

end strawberry_candies_count_l1810_181096


namespace largest_k_exists_l1810_181081

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end largest_k_exists_l1810_181081


namespace range_of_a_l1810_181098
noncomputable section

open Real

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * x + a + 2 > 0) : a > -1 :=
sorry

end range_of_a_l1810_181098


namespace car_trader_profit_l1810_181015

theorem car_trader_profit (P : ℝ) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.28000000000000004 * P
  let profit := selling_price - purchase_price
  let percentage_increase := (profit / purchase_price) * 100
  percentage_increase = 60 := 
by
  sorry

end car_trader_profit_l1810_181015


namespace probability_of_sum_17_is_correct_l1810_181030

def probability_sum_17 : ℚ :=
  let favourable_outcomes := 2
  let total_outcomes := 81
  favourable_outcomes / total_outcomes

theorem probability_of_sum_17_is_correct :
  probability_sum_17 = 2 / 81 :=
by
  -- The proof steps are not required for this task
  sorry

end probability_of_sum_17_is_correct_l1810_181030


namespace quilt_squares_count_l1810_181084

theorem quilt_squares_count (total_squares : ℕ) (additional_squares : ℕ)
  (h1 : total_squares = 4 * additional_squares)
  (h2 : additional_squares = 24) :
  total_squares = 32 :=
by
  -- Proof would go here
  -- The proof would involve showing that total_squares indeed equals 32 given h1 and h2
  sorry

end quilt_squares_count_l1810_181084


namespace problem_2535_l1810_181007

theorem problem_2535 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + b + (a^3 / b^2) + (b^3 / a^2) = 2535 := sorry

end problem_2535_l1810_181007


namespace basketball_not_table_tennis_l1810_181056

theorem basketball_not_table_tennis (total_students likes_basketball likes_table_tennis dislikes_all : ℕ) (likes_basketball_not_tt : ℕ) :
  total_students = 30 →
  likes_basketball = 15 →
  likes_table_tennis = 10 →
  dislikes_all = 8 →
  (likes_basketball - 3 = likes_basketball_not_tt) →
  likes_basketball_not_tt = 12 := by
  intros h_total h_basketball h_table_tennis h_dislikes h_eq
  sorry

end basketball_not_table_tennis_l1810_181056


namespace rocket_parachute_opens_l1810_181051

theorem rocket_parachute_opens (h t : ℝ) : h = -t^2 + 12 * t + 1 ∧ h = 37 -> t = 6 :=
by sorry

end rocket_parachute_opens_l1810_181051


namespace pentagon_area_l1810_181047

theorem pentagon_area 
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ) (side5 : ℝ)
  (h1 : side1 = 12) (h2 : side2 = 20) (h3 : side3 = 30) (h4 : side4 = 15) (h5 : side5 = 25)
  (right_angle : ∃ (a b : ℝ), a = side1 ∧ b = side5 ∧ a^2 + b^2 = (a + b)^2) : 
  ∃ (area : ℝ), area = 600 := 
  sorry

end pentagon_area_l1810_181047


namespace min_gennadys_needed_l1810_181036

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l1810_181036


namespace fox_initial_coins_l1810_181059

theorem fox_initial_coins :
  ∃ x : ℤ, x - 10 = 0 ∧ 2 * (x - 10) - 50 = 0 ∧ 2 * (2 * (x - 10) - 50) - 50 = 0 ∧
  2 * (2 * (2 * (x - 10) - 50) - 50) - 50 = 0 ∧ 2 * (2 * (2 * (2 * (x - 10) - 50) - 50) - 50) - 50 = 0 ∧
  x = 56 := 
by
  -- we skip the proof here
  sorry

end fox_initial_coins_l1810_181059


namespace chess_tournament_possible_l1810_181078

section ChessTournament

structure Player :=
  (name : String)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

def points (p : Player) : ℕ :=
  p.wins + p.draws / 2

def is_possible (A B C : Player) : Prop :=
  (points A > points B) ∧ (points A > points C) ∧
  (points C < points B) ∧
  (A.wins < B.wins) ∧ (A.wins < C.wins) ∧
  (C.wins > B.wins)

theorem chess_tournament_possible (A B C : Player) :
  is_possible A B C :=
  sorry

end ChessTournament

end chess_tournament_possible_l1810_181078


namespace incorrect_average_l1810_181088

theorem incorrect_average (S : ℕ) (A_correct : ℕ) (A_incorrect : ℕ) (S_correct : ℕ) 
  (h1 : S = 135)
  (h2 : A_correct = 19)
  (h3 : A_incorrect = (S + 25) / 10)
  (h4 : S_correct = (S + 55) / 10)
  (h5 : S_correct = A_correct) :
  A_incorrect = 16 :=
by
  -- The proof will go here, which is skipped with a 'sorry'
  sorry

end incorrect_average_l1810_181088


namespace lattice_points_non_visible_square_l1810_181023

theorem lattice_points_non_visible_square (n : ℕ) (h : n > 0) : 
  ∃ (a b : ℤ), ∀ (x y : ℤ), a < x ∧ x < a + n ∧ b < y ∧ y < b + n → Int.gcd x y > 1 :=
sorry

end lattice_points_non_visible_square_l1810_181023


namespace barbell_percentage_increase_l1810_181085

def old_barbell_cost : ℕ := 250
def new_barbell_cost : ℕ := 325

theorem barbell_percentage_increase :
  (new_barbell_cost - old_barbell_cost : ℚ) / old_barbell_cost * 100 = 30 := 
by
  sorry

end barbell_percentage_increase_l1810_181085


namespace x_coordinate_D_l1810_181040

noncomputable def find_x_coordinate_D (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ := 
  let l := -a * b
  let x := l / c
  x

theorem x_coordinate_D (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (D_on_parabola : d^2 = (a + b) * (d) + l)
  (lines_intersect_y_axis : ∃ l : ℝ, (a^2 = (b + a) * a + l) ∧ (b^2 = (b + a) * b + l) ∧ (c^2 = (d + c) * c + l)) :
  d = (a * b) / c :=
by sorry

end x_coordinate_D_l1810_181040


namespace unique_x2_range_of_a_l1810_181073

noncomputable def f (x : ℝ) (k a : ℝ) : ℝ :=
if x >= 0
then k*x + k*(1 - a^2)
else x^2 + (a^2 - 4*a)*x + (3 - a)^2

theorem unique_x2 (k a : ℝ) (x1 : ℝ) (hx1 : x1 ≠ 0) (hx2 : ∃ x2 : ℝ, x2 ≠ 0 ∧ x2 ≠ x1 ∧ f x2 k a = f x1 k a) :
f 0 k a = k*(1 - a^2) →
0 ≤ a ∧ a < 1 →
k = (3 - a)^2 / (1 - a^2) :=
sorry

variable (a : ℝ)

theorem range_of_a :
0 ≤ a ∧ a < 1 ↔ a^2 - 4*a ≤ 0 :=
sorry

end unique_x2_range_of_a_l1810_181073


namespace determine_prices_l1810_181018

variable (num_items : ℕ) (cost_keychains cost_plush : ℕ) (x : ℚ) (unit_price_keychains unit_price_plush : ℚ)

noncomputable def price_equation (x : ℚ) : Prop :=
  (cost_keychains / x) + (cost_plush / (1.5 * x)) = num_items

theorem determine_prices 
  (h1 : num_items = 15)
  (h2 : cost_keychains = 240)
  (h3 : cost_plush = 180)
  (h4 : price_equation num_items cost_keychains cost_plush x)
  (hx : x = 24) :
  unit_price_keychains = 24 ∧ unit_price_plush = 36 :=
  by
    sorry

end determine_prices_l1810_181018


namespace probability_of_region_l1810_181066

-- Definition of the bounds
def bounds (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8

-- Definition of the region where x + y <= 5
def region (x y : ℝ) : Prop := x + y ≤ 5

-- The proof statement
theorem probability_of_region : 
  (∃ (x y : ℝ), bounds x y ∧ region x y) →
  ∃ (p : ℚ), p = 3/8 :=
by sorry

end probability_of_region_l1810_181066


namespace solution_set_inequality_range_of_m_l1810_181061

def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Problem 1
theorem solution_set_inequality (x : ℝ) : 
  (f x 5 > 2) ↔ (-3 / 2 < x ∧ x < 3 / 2) :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (x^2 + 2 * x + 3) ∧ y = f x m) ↔ (m ≥ 4) :=
sorry

end solution_set_inequality_range_of_m_l1810_181061


namespace find_antecedent_l1810_181067

-- Condition: The ratio is 4:6, simplified to 2:3
def ratio (a b : ℕ) : Prop := (a / gcd a b) = 2 ∧ (b / gcd a b) = 3

-- Condition: The consequent is 30
def consequent (y : ℕ) : Prop := y = 30

-- The problem is to find the antecedent
def antecedent (x : ℕ) (y : ℕ) : Prop := ratio x y

-- The theorem to be proved
theorem find_antecedent:
  ∃ x : ℕ, consequent 30 → antecedent x 30 ∧ x = 20 :=
by
  sorry

end find_antecedent_l1810_181067


namespace inequality_abcd_l1810_181008

theorem inequality_abcd (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) :
    (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) >= 2 / 3) :=
by
  sorry

end inequality_abcd_l1810_181008


namespace initial_soccer_balls_l1810_181045

theorem initial_soccer_balls (x : ℝ) (h1 : 0.40 * x = y) (h2 : 0.20 * (0.60 * x) = z) (h3 : 0.80 * (0.60 * x) = 48) : x = 100 := by
  sorry

end initial_soccer_balls_l1810_181045


namespace zoe_bought_bottles_l1810_181016

theorem zoe_bought_bottles
  (initial_bottles : ℕ)
  (drank_bottles : ℕ)
  (current_bottles : ℕ)
  (initial_bottles_eq : initial_bottles = 42)
  (drank_bottles_eq : drank_bottles = 25)
  (current_bottles_eq : current_bottles = 47) :
  ∃ bought_bottles : ℕ, bought_bottles = 30 :=
by
  sorry

end zoe_bought_bottles_l1810_181016


namespace area_transformation_l1810_181031

variables {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x in a..b, g x = 12) :
  ∫ x in c..d, 4 * g (2 * x + 3) = 48 :=
by
  sorry

end area_transformation_l1810_181031


namespace revenue_per_investment_l1810_181072

theorem revenue_per_investment (Banks_investments : ℕ) (Elizabeth_investments : ℕ) (Elizabeth_revenue_per_investment : ℕ) (revenue_difference : ℕ) :
  Banks_investments = 8 →
  Elizabeth_investments = 5 →
  Elizabeth_revenue_per_investment = 900 →
  revenue_difference = 500 →
  ∃ (R : ℤ), R = (5 * 900 - 500) / 8 :=
by
  intros h1 h2 h3 h4
  let T_elizabeth := 5 * Elizabeth_revenue_per_investment
  let T_banks := T_elizabeth - revenue_difference
  let R := T_banks / 8
  use R
  sorry

end revenue_per_investment_l1810_181072


namespace smallest_n_divisibility_l1810_181004

theorem smallest_n_divisibility:
  ∃ (n : ℕ), n > 0 ∧ n^2 % 24 = 0 ∧ n^3 % 540 = 0 ∧ n = 60 :=
by
  sorry

end smallest_n_divisibility_l1810_181004


namespace verify_value_l1810_181029

theorem verify_value (a b c d m : ℝ) 
  (h₁ : a = -b) 
  (h₂ : c * d = 1) 
  (h₃ : |m| = 3) :
  3 * c * d + (a + b) / (c * d) - m = 0 ∨ 
  3 * c * d + (a + b) / (c * d) - m = 6 := 
sorry

end verify_value_l1810_181029


namespace hard_candy_food_coloring_l1810_181014

theorem hard_candy_food_coloring
  (lollipop_coloring : ℕ) (hard_candy_coloring : ℕ)
  (num_lollipops : ℕ) (num_hardcandies : ℕ)
  (total_coloring : ℕ)
  (H1 : lollipop_coloring = 8)
  (H2 : num_lollipops = 150)
  (H3 : num_hardcandies = 20)
  (H4 : total_coloring = 1800) :
  (20 * hard_candy_coloring + 150 * lollipop_coloring = total_coloring) → 
  hard_candy_coloring = 30 :=
by
  sorry

end hard_candy_food_coloring_l1810_181014


namespace factorize_n_squared_minus_nine_l1810_181017

theorem factorize_n_squared_minus_nine (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := 
sorry

end factorize_n_squared_minus_nine_l1810_181017


namespace average_score_l1810_181058

theorem average_score 
  (total_students : ℕ)
  (assigned_day_students_pct : ℝ)
  (makeup_day_students_pct : ℝ)
  (assigned_day_avg_score : ℝ)
  (makeup_day_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_students_pct = 0.70)
  (h3 : makeup_day_students_pct = 0.30)
  (h4 : assigned_day_avg_score = 0.60)
  (h5 : makeup_day_avg_score = 0.90) :
  (0.70 * 100 * 0.60 + 0.30 * 100 * 0.90) / 100 = 0.69 := 
sorry


end average_score_l1810_181058


namespace minimum_harmonic_sum_l1810_181025

theorem minimum_harmonic_sum
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
by
  sorry

end minimum_harmonic_sum_l1810_181025


namespace pave_square_with_tiles_l1810_181026

theorem pave_square_with_tiles (b c : ℕ) (h_right_triangle : (b > 0) ∧ (c > 0)) :
  (∃ (k : ℕ), k^2 = b^2 + c^2) ↔ (∃ (m n : ℕ), m * c * b = 2 * n^2 * (b^2 + c^2)) := 
sorry

end pave_square_with_tiles_l1810_181026


namespace time_fraction_reduced_l1810_181094

theorem time_fraction_reduced (T D : ℝ) (h1 : D = 30 * T) :
  D = 40 * ((3/4) * T) → 1 - (3/4) = 1/4 :=
sorry

end time_fraction_reduced_l1810_181094


namespace soft_drink_cost_l1810_181037

/-- Benny bought 2 soft drinks for a certain price each and 5 candy bars.
    He spent a total of $28. Each candy bar cost $4. 
    Prove that the cost of each soft drink was $4.
--/
theorem soft_drink_cost (S : ℝ) (h1 : 2 * S + 5 * 4 = 28) : S = 4 := 
by
  sorry

end soft_drink_cost_l1810_181037


namespace quadrilateral_is_parallelogram_l1810_181010

theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) 
  : (a = c ∧ b = d) → parallelogram :=
by
  sorry

end quadrilateral_is_parallelogram_l1810_181010


namespace probability_of_white_first_red_second_l1810_181065

noncomputable def probability_white_first_red_second : ℚ :=
let totalBalls := 6
let probWhiteFirst := 1 / totalBalls
let remainingBalls := totalBalls - 1
let probRedSecond := 1 / remainingBalls
probWhiteFirst * probRedSecond

theorem probability_of_white_first_red_second :
  probability_white_first_red_second = 1 / 30 :=
by
  sorry

end probability_of_white_first_red_second_l1810_181065


namespace books_total_pages_l1810_181082

theorem books_total_pages (x y z : ℕ) 
  (h1 : (2 / 3 : ℚ) * x - (1 / 3 : ℚ) * x = 20)
  (h2 : (3 / 5 : ℚ) * y - (2 / 5 : ℚ) * y = 15)
  (h3 : (3 / 4 : ℚ) * z - (1 / 4 : ℚ) * z = 30) : 
  x = 60 ∧ y = 75 ∧ z = 60 :=
by
  sorry

end books_total_pages_l1810_181082


namespace hannah_total_cost_l1810_181003

def price_per_kg : ℝ := 5
def discount_rate : ℝ := 0.4
def kilograms : ℝ := 10

theorem hannah_total_cost :
  (price_per_kg * (1 - discount_rate)) * kilograms = 30 := 
by
  sorry

end hannah_total_cost_l1810_181003


namespace min_value_of_translated_function_l1810_181011

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

theorem min_value_of_translated_function :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (Real.pi / 2) → ∀ (ϕ : ℝ), |ϕ| < (Real.pi / 2) →
  ∀ (k : ℤ), f (x + (Real.pi / 6)) (ϕ + (Real.pi / 3) + k * Real.pi) = f x ϕ →
  ∃ y : ℝ, y = - Real.sqrt 3 / 2 := sorry

end min_value_of_translated_function_l1810_181011


namespace systematic_sampling_method_l1810_181079

theorem systematic_sampling_method :
  ∀ (num_classes num_students_per_class selected_student : ℕ),
    num_classes = 12 →
    num_students_per_class = 50 →
    selected_student = 40 →
    (∃ (start_interval: ℕ) (interval: ℕ) (total_population: ℕ), 
      total_population > 100 ∧ start_interval < interval ∧ interval * num_classes = total_population ∧
      ∀ (c : ℕ), c < num_classes → (start_interval + c * interval) % num_students_per_class = selected_student - 1) →
    "Systematic Sampling" = "Systematic Sampling" :=
by
  intros num_classes num_students_per_class selected_student h_classes h_students h_selected h_conditions
  sorry

end systematic_sampling_method_l1810_181079


namespace kayden_total_processed_l1810_181041

-- Definition of the given conditions and final proof problem statement in Lean 4
variable (x : ℕ)  -- x is the number of cartons delivered to each customer

theorem kayden_total_processed (h : 4 * (x - 60) = 160) : 4 * x = 400 :=
by
  sorry

end kayden_total_processed_l1810_181041


namespace john_pays_2010_dollars_l1810_181093

-- Define the main problem as the number of ways to pay 2010$ using 2, 5, and 10$ notes.
theorem john_pays_2010_dollars :
  ∃ (count : ℕ), count = 20503 ∧
  ∀ (x y z : ℕ), (2 * x + 5 * y + 10 * z = 2010) → (x % 5 = 0) → (y % 2 = 0) → count = 20503 :=
by sorry

end john_pays_2010_dollars_l1810_181093


namespace number_of_homes_cleaned_l1810_181033

-- Define constants for the amount Mary earns per home and the total amount she made.
def amount_per_home := 46
def total_amount_made := 276

-- Prove that the number of homes Mary cleaned is 6 given the conditions.
theorem number_of_homes_cleaned : total_amount_made / amount_per_home = 6 :=
by
  sorry

end number_of_homes_cleaned_l1810_181033


namespace find_m_l1810_181002

noncomputable def m_value (m : ℝ) := 
  ((m ^ 2) - m - 1, (m ^ 2) - 2 * m - 1)

theorem find_m (m : ℝ) (h1 : (m ^ 2) - m - 1 = 1) (h2 : (m ^ 2) - 2 * m - 1 < 0) : 
  m = 2 :=
by sorry

end find_m_l1810_181002


namespace toms_dog_age_in_six_years_l1810_181068

-- Define the conditions as hypotheses
variables (B T D : ℕ)

-- Conditions
axiom h1 : B = 4 * D
axiom h2 : T = B - 3
axiom h3 : B + 6 = 30

-- The proof goal: Tom's dog's age in six years
theorem toms_dog_age_in_six_years : D + 6 = 12 :=
  sorry -- Proof is omitted based on the instructions

end toms_dog_age_in_six_years_l1810_181068


namespace lcm_18_30_eq_90_l1810_181095

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l1810_181095
