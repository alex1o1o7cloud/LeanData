import Mathlib

namespace dust_storm_acres_l1571_157121

def total_acres : ℕ := 64013
def untouched_acres : ℕ := 522
def dust_storm_covered : ℕ := total_acres - untouched_acres

theorem dust_storm_acres :
  dust_storm_covered = 63491 := by
  sorry

end dust_storm_acres_l1571_157121


namespace polynomial_root_recip_squares_l1571_157118

theorem polynomial_root_recip_squares (a b c : ℝ) 
  (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6):
  1 / a^2 + 1 / b^2 + 1 / c^2 = 49 / 36 :=
sorry

end polynomial_root_recip_squares_l1571_157118


namespace find_x_l1571_157158

theorem find_x (x y z : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : x + 10 = y - 10)
  (h3 : x + 10 = 3 * z) :
  x = 380 / 7 := 
  sorry

end find_x_l1571_157158


namespace conditional_without_else_l1571_157174

def if_then_else_statement (s: String) : Prop :=
  (s = "IF—THEN" ∨ s = "IF—THEN—ELSE")

theorem conditional_without_else : if_then_else_statement "IF—THEN" :=
  sorry

end conditional_without_else_l1571_157174


namespace monotonically_decreasing_iff_l1571_157148

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a^x

theorem monotonically_decreasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a y ≤ f a x) ↔ (3/8 ≤ a ∧ a < 2/3) :=
sorry

end monotonically_decreasing_iff_l1571_157148


namespace condition_on_a_l1571_157137

theorem condition_on_a (a : ℝ) : 
  (∀ x : ℝ, (5 * x - 3 < 3 * x + 5) → (x < a)) ↔ (a ≥ 4) :=
by
  sorry

end condition_on_a_l1571_157137


namespace find_dot_AP_BC_l1571_157195

-- Defining the lengths of the sides of the triangle.
def length_AB : ℝ := 13
def length_BC : ℝ := 14
def length_CA : ℝ := 15

-- Defining the provided dot product conditions at point P.
def dot_BP_CA : ℝ := 18
def dot_CP_BA : ℝ := 32

-- The target is to prove the final dot product.
theorem find_dot_AP_BC :
  ∃ (AP BC : ℝ), BC = 14 → dot_BP_CA = 18 → dot_CP_BA = 32 → (AP * BC = 14) :=
by
  -- proof goes here
  sorry

end find_dot_AP_BC_l1571_157195


namespace joe_out_of_money_after_one_month_worst_case_l1571_157107

-- Define the initial amount Joe has
def initial_amount : ℝ := 240

-- Define Joe's monthly subscription cost
def subscription_cost : ℝ := 15

-- Define the range of prices for buying games
def min_game_cost : ℝ := 40
def max_game_cost : ℝ := 60

-- Define the range of prices for selling games
def min_resale_price : ℝ := 20
def max_resale_price : ℝ := 40

-- Define the maximum number of games Joe can purchase per month
def max_games_per_month : ℕ := 3

-- Prove that Joe will be out of money after 1 month in the worst-case scenario
theorem joe_out_of_money_after_one_month_worst_case :
  initial_amount - 
  (max_games_per_month * max_game_cost - max_games_per_month * min_resale_price + subscription_cost) < 0 :=
by
  sorry

end joe_out_of_money_after_one_month_worst_case_l1571_157107


namespace copper_content_range_l1571_157175

theorem copper_content_range (x2 : ℝ) (y : ℝ) (h1 : 0 ≤ x2) (h2 : x2 ≤ 4 / 9) (hy : y = 0.4 + 0.075 * x2) : 
  40 ≤ 100 * y ∧ 100 * y ≤ 130 / 3 :=
by { sorry }

end copper_content_range_l1571_157175


namespace redistribution_l1571_157176

/-
Given:
- b = (12 / 13) * a
- c = (2 / 3) * b
- Person C will contribute 9 dollars based on the amount each person spent

Prove:
- Person C gives 6 dollars to Person A.
- Person C gives 3 dollars to Person B.
-/

theorem redistribution (a b c : ℝ) (h1 : b = (12 / 13) * a) (h2 : c = (2 / 3) * b) : 
  ∃ (x y : ℝ), x + y = 9 ∧ x = 6 ∧ y = 3 :=
by
  sorry

end redistribution_l1571_157176


namespace factor_expression_l1571_157167

theorem factor_expression (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by
  sorry

end factor_expression_l1571_157167


namespace green_or_yellow_probability_l1571_157116

-- Given the number of marbles of each color
def green_marbles : ℕ := 4
def yellow_marbles : ℕ := 3
def white_marbles : ℕ := 6

-- The total number of marbles
def total_marbles : ℕ := green_marbles + yellow_marbles + white_marbles

-- The number of favorable outcomes (green or yellow marbles)
def favorable_marbles : ℕ := green_marbles + yellow_marbles

-- The probability of drawing a green or yellow marble as a fraction
def probability_of_green_or_yellow : Rat := favorable_marbles / total_marbles

theorem green_or_yellow_probability :
  probability_of_green_or_yellow = 7 / 13 :=
by
  sorry

end green_or_yellow_probability_l1571_157116


namespace correct_propositions_l1571_157124

def Line := Type
def Plane := Type

variables (m n: Line) (α β γ: Plane)

-- Conditions from the problem statement
axiom perp (x: Line) (y: Plane): Prop -- x ⊥ y
axiom parallel (x: Line) (y: Plane): Prop -- x ∥ y
axiom perp_planes (x: Plane) (y: Plane): Prop -- x ⊥ y
axiom parallel_planes (x: Plane) (y: Plane): Prop -- x ∥ y

-- Given the conditions
axiom h1: perp m α
axiom h2: parallel n α
axiom h3: perp_planes α γ
axiom h4: perp_planes β γ
axiom h5: parallel_planes α β
axiom h6: parallel_planes β γ
axiom h7: parallel m α
axiom h8: parallel n α
axiom h9: perp m n
axiom h10: perp m γ

-- Lean statement for the problem: Prove that Propositions ① and ④ are correct.
theorem correct_propositions : (perp m n) ∧ (perp m γ) :=
by sorry -- Proof steps are not required.

end correct_propositions_l1571_157124


namespace set_points_quadrants_l1571_157194

theorem set_points_quadrants (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → 
  (y > 0 ∧ x > 0) ∨ (y > 0 ∧ x < 0) :=
by 
  sorry

end set_points_quadrants_l1571_157194


namespace combined_age_of_four_siblings_l1571_157106

theorem combined_age_of_four_siblings :
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  aaron_age + sister_age + henry_age + alice_age = 253 :=
by
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  have h1 : aaron_age + sister_age + henry_age + alice_age = 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) := by sorry
  have h2 : 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) = 253 := by sorry
  exact h1.trans h2

end combined_age_of_four_siblings_l1571_157106


namespace exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l1571_157126

noncomputable def f (x a : ℝ) : ℝ :=
if x > a then (x - 1)^3 else abs (x - 1)

theorem exists_no_minimum_value :
  ∃ a : ℝ, ¬ ∃ m : ℝ, ∀ x : ℝ, f x a ≥ m :=
sorry

theorem has_zeros_for_any_a (a : ℝ) : ∃ x : ℝ, f x a = 0 :=
sorry

theorem not_monotonically_increasing_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  ¬ ∀ x y : ℝ, 1 < x → x < y → y < a → f x a ≤ f y a :=
sorry

theorem exists_m_for_3_distinct_roots (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ m : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = m ∧ f x2 a = m ∧ f x3 a = m :=
sorry

end exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l1571_157126


namespace option_e_is_perfect_square_l1571_157192

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem option_e_is_perfect_square :
  is_perfect_square (4^10 * 5^5 * 6^10) :=
sorry

end option_e_is_perfect_square_l1571_157192


namespace necessary_but_not_sufficient_l1571_157155

theorem necessary_but_not_sufficient {a b c d : ℝ} (hcd : c > d) : 
  (a - c > b - d) → (a > b) ∧ ¬((a > b) → (a - c > b - d)) :=
by
  sorry

end necessary_but_not_sufficient_l1571_157155


namespace min_value_of_reciprocal_sum_l1571_157173

variable (m n : ℝ)
variable (a : ℝ × ℝ := (m, 1))
variable (b : ℝ × ℝ := (4 - n, 2))

theorem min_value_of_reciprocal_sum
  (h1 : m > 0) (h2 : n > 0)
  (h3 : a.1 * b.2 = a.2 * b.1) :
  (1/m + 8/n) = 9/2 :=
sorry

end min_value_of_reciprocal_sum_l1571_157173


namespace intersection_of_sets_l1571_157120

-- Defining set M
def M : Set ℝ := { x | x^2 + x - 2 < 0 }

-- Defining set N
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

-- Theorem stating the solution
theorem intersection_of_sets : M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_sets_l1571_157120


namespace abc_order_l1571_157122

noncomputable def a : ℝ := Real.log (3 / 2) - 3 / 2
noncomputable def b : ℝ := Real.log Real.pi - Real.pi
noncomputable def c : ℝ := Real.log 3 - 3

theorem abc_order : a > c ∧ c > b := by
  have h₁: a = Real.log (3 / 2) - 3 / 2 := rfl
  have h₂: b = Real.log Real.pi - Real.pi := rfl
  have h₃: c = Real.log 3 - 3 := rfl
  sorry

end abc_order_l1571_157122


namespace find_a_l1571_157117

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1)
  (h_diff : |a^2 - a| = 6) : a = 3 :=
sorry

end find_a_l1571_157117


namespace min_photos_needed_to_ensure_conditions_l1571_157193

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l1571_157193


namespace fill_tank_with_reduced_bucket_capacity_l1571_157102

theorem fill_tank_with_reduced_bucket_capacity (C : ℝ) :
    let original_buckets := 200
    let original_capacity := C
    let new_capacity := (4 / 5) * original_capacity
    let new_buckets := 250
    (original_buckets * original_capacity) = ((new_buckets) * new_capacity) :=
by
    sorry

end fill_tank_with_reduced_bucket_capacity_l1571_157102


namespace sean_divided_by_julie_is_2_l1571_157119

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l1571_157119


namespace ratio_of_full_boxes_l1571_157108

theorem ratio_of_full_boxes 
  (F H : ℕ)
  (boxes_count_eq : F + H = 20)
  (parsnips_count_eq : 20 * F + 10 * H = 350) :
  F / (F + H) = 3 / 4 := 
by
  -- proof will be placed here
  sorry

end ratio_of_full_boxes_l1571_157108


namespace determine_parity_of_f_l1571_157191

def parity_of_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = 0

theorem determine_parity_of_f (f : ℝ → ℝ) :
  (∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) →
  parity_of_f f :=
sorry

end determine_parity_of_f_l1571_157191


namespace F_double_prime_coordinates_correct_l1571_157112

structure Point where
  x : Int
  y : Int

def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_over_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := 6, y := -4 }

def F' : Point := reflect_over_y_axis F

def F'' : Point := reflect_over_x_axis F'

theorem F_double_prime_coordinates_correct : F'' = { x := -6, y := 4 } :=
  sorry

end F_double_prime_coordinates_correct_l1571_157112


namespace range_of_a_l1571_157103

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ 1 :=
sorry

end range_of_a_l1571_157103


namespace pascal_family_min_children_l1571_157185

-- We define the conditions b >= 3 and g >= 2
def b_condition (b : ℕ) : Prop := b >= 3
def g_condition (g : ℕ) : Prop := g >= 2

-- We state that the smallest number of children given these conditions is 5
theorem pascal_family_min_children (b g : ℕ) (hb : b_condition b) (hg : g_condition g) : b + g = 5 :=
sorry

end pascal_family_min_children_l1571_157185


namespace certain_amount_of_seconds_l1571_157130

theorem certain_amount_of_seconds (X : ℕ)
    (cond1 : 12 / X = 16 / 480) :
    X = 360 :=
by
  sorry

end certain_amount_of_seconds_l1571_157130


namespace bill_earnings_l1571_157164

theorem bill_earnings
  (milk_total : ℕ)
  (fraction : ℚ)
  (milk_to_butter_ratio : ℕ)
  (milk_to_sour_cream_ratio : ℕ)
  (butter_price_per_gallon : ℚ)
  (sour_cream_price_per_gallon : ℚ)
  (whole_milk_price_per_gallon : ℚ)
  (milk_for_butter : ℚ)
  (milk_for_sour_cream : ℚ)
  (remaining_milk : ℚ)
  (total_earnings : ℚ) :
  milk_total = 16 →
  fraction = 1/4 →
  milk_to_butter_ratio = 4 →
  milk_to_sour_cream_ratio = 2 →
  butter_price_per_gallon = 5 →
  sour_cream_price_per_gallon = 6 →
  whole_milk_price_per_gallon = 3 →
  milk_for_butter = milk_total * fraction / milk_to_butter_ratio →
  milk_for_sour_cream = milk_total * fraction / milk_to_sour_cream_ratio →
  remaining_milk = milk_total - 2 * (milk_total * fraction) →
  total_earnings = (remaining_milk * whole_milk_price_per_gallon) + 
                   (milk_for_sour_cream * sour_cream_price_per_gallon) + 
                   (milk_for_butter * butter_price_per_gallon) →
  total_earnings = 41 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end bill_earnings_l1571_157164


namespace correct_line_equation_l1571_157197

theorem correct_line_equation :
  ∃ (c : ℝ), (∀ (x y : ℝ), 2 * x - 3 * y + 4 = 0 → 2 * x - 3 * y + c = 0 ∧ 2 * (-1) - 3 * 2 + c = 0) ∧ c = 8 :=
by
  use 8
  sorry

end correct_line_equation_l1571_157197


namespace hyperbola_center_l1571_157100

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0 → (x, y) = (3, 5) :=
by
  sorry

end hyperbola_center_l1571_157100


namespace relationship_between_x_and_y_l1571_157123

variables (x y : ℝ)

theorem relationship_between_x_and_y (h1 : x + y > 2 * x) (h2 : x - y < 2 * y) : y > x := 
sorry

end relationship_between_x_and_y_l1571_157123


namespace solve_angle_CBO_l1571_157159

theorem solve_angle_CBO 
  (BAO CAO : ℝ) (CBO ABO : ℝ) (ACO BCO : ℝ) (AOC : ℝ) 
  (h1 : BAO = CAO) 
  (h2 : CBO = ABO) 
  (h3 : ACO = BCO) 
  (h4 : AOC = 110) 
  : CBO = 20 :=
by
  sorry

end solve_angle_CBO_l1571_157159


namespace find_b_l1571_157129

theorem find_b (a b : ℝ) (h1 : (-6) * a^2 = 3 * (4 * a + b))
  (h2 : a = 1) : b = -6 :=
by 
  sorry

end find_b_l1571_157129


namespace correct_average_l1571_157143

theorem correct_average 
(n : ℕ) (avg1 avg2 avg3 : ℝ): 
  n = 10 
  → avg1 = 40.2 
  → avg2 = avg1
  → avg3 = avg1
  → avg1 = avg3 :=
by 
  intros hn h_avg1 h_avg2 h_avg3
  sorry

end correct_average_l1571_157143


namespace compute_fraction_l1571_157127

def x : ℚ := 2 / 3
def y : ℚ := 3 / 2
def z : ℚ := 1 / 3

theorem compute_fraction :
  (1 / 3) * x^7 * y^5 * z^4 = 11 / 600 :=
by
  sorry

end compute_fraction_l1571_157127


namespace andy_coats_l1571_157113

-- Define the initial number of minks Andy buys
def initial_minks : ℕ := 30

-- Define the number of babies each mink has
def babies_per_mink : ℕ := 6

-- Define the total initial minks including babies
def total_initial_minks : ℕ := initial_minks * babies_per_mink + initial_minks

-- Define the number of minks set free by activists
def minks_set_free : ℕ := total_initial_minks / 2

-- Define the number of minks remaining after half are set free
def remaining_minks : ℕ := total_initial_minks - minks_set_free

-- Define the number of mink skins needed for one coat
def mink_skins_per_coat : ℕ := 15

-- Define the number of coats Andy can make
def coats_andy_can_make : ℕ := remaining_minks / mink_skins_per_coat

-- The theorem to prove the number of coats Andy can make
theorem andy_coats : coats_andy_can_make = 7 := by
  sorry

end andy_coats_l1571_157113


namespace find_pairs_l1571_157178

noncomputable def pairs_of_real_numbers (α β : ℝ) := 
  ∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
    (x + y^2 + z^3 + w^6 ≥ α * (x * y * z * w)^β)

theorem find_pairs (α β : ℝ) :
  (∃ x y z w : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
    (x + y^2 + z^3 + w^6 = α * (x * y * z * w)^β))
  →
  pairs_of_real_numbers α β :=
sorry

end find_pairs_l1571_157178


namespace tabby_average_speed_l1571_157125

noncomputable def overall_average_speed : ℝ := 
  let swimming_speed : ℝ := 1
  let cycling_speed : ℝ := 18
  let running_speed : ℝ := 6
  let time_swimming : ℝ := 2
  let time_cycling : ℝ := 3
  let time_running : ℝ := 2
  let distance_swimming := swimming_speed * time_swimming
  let distance_cycling := cycling_speed * time_cycling
  let distance_running := running_speed * time_running
  let total_distance := distance_swimming + distance_cycling + distance_running
  let total_time := time_swimming + time_cycling + time_running
  total_distance / total_time

theorem tabby_average_speed : overall_average_speed = 9.71 := sorry

end tabby_average_speed_l1571_157125


namespace monica_book_ratio_theorem_l1571_157133

/-
Given:
1. Monica read 16 books last year.
2. This year, she read some multiple of the number of books she read last year.
3. Next year, she will read 69 books.
4. Next year, she wants to read 5 more than twice the number of books she read this year.

Prove:
The ratio of the number of books she read this year to the number of books she read last year is 2.
-/

noncomputable def monica_book_ratio_proof : Prop :=
  let last_year_books := 16
  let next_year_books := 69
  ∃ (x : ℕ), (∃ (n : ℕ), x = last_year_books * n) ∧ (2 * x + 5 = next_year_books) ∧ (x / last_year_books = 2)

theorem monica_book_ratio_theorem : monica_book_ratio_proof :=
  by
    sorry

end monica_book_ratio_theorem_l1571_157133


namespace sheela_monthly_income_eq_l1571_157141

-- Defining the conditions
def sheela_deposit : ℝ := 4500
def percentage_of_income : ℝ := 0.28

-- Define Sheela's monthly income as I
variable (I : ℝ)

-- The theorem to prove
theorem sheela_monthly_income_eq : (percentage_of_income * I = sheela_deposit) → (I = 16071.43) :=
by
  sorry

end sheela_monthly_income_eq_l1571_157141


namespace determine_beta_l1571_157170

-- Define a structure for angles in space
structure Angle where
  measure : ℝ

-- Define the conditions
def alpha : Angle := ⟨30⟩
def parallel_sides (a b : Angle) : Prop := true  -- Simplification for the example, should be defined properly for general case

-- The theorem to be proved
theorem determine_beta (α β : Angle) (h1 : α = Angle.mk 30) (h2 : parallel_sides α β) : β = Angle.mk 30 ∨ β = Angle.mk 150 := by
  sorry

end determine_beta_l1571_157170


namespace integer_solutions_count_l1571_157111

theorem integer_solutions_count : ∃ (s : Finset ℤ), (∀ x ∈ s, x^2 - x - 2 ≤ 0) ∧ (Finset.card s = 4) :=
by
  sorry

end integer_solutions_count_l1571_157111


namespace population_reaches_target_l1571_157163

def initial_year : ℕ := 2020
def initial_population : ℕ := 450
def growth_period : ℕ := 25
def growth_factor : ℕ := 3
def target_population : ℕ := 10800

theorem population_reaches_target : ∃ (year : ℕ), year - initial_year = 3 * growth_period ∧ (initial_population * growth_factor ^ 3) >= target_population := by
  sorry

end population_reaches_target_l1571_157163


namespace elephant_entry_duration_l1571_157172

theorem elephant_entry_duration
  (initial_elephants : ℕ)
  (exodus_duration : ℕ)
  (leaving_rate : ℕ)
  (entering_rate : ℕ)
  (final_elephants : ℕ)
  (h_initial : initial_elephants = 30000)
  (h_exodus_duration : exodus_duration = 4)
  (h_leaving_rate : leaving_rate = 2880)
  (h_entering_rate : entering_rate = 1500)
  (h_final : final_elephants = 28980) :
  (final_elephants - (initial_elephants - (exodus_duration * leaving_rate))) / entering_rate = 7 :=
by
  sorry

end elephant_entry_duration_l1571_157172


namespace total_apples_picked_l1571_157152

-- Definitions based on conditions from part a)
def mike_apples : ℝ := 7.5
def nancy_apples : ℝ := 3.2
def keith_apples : ℝ := 6.1
def olivia_apples : ℝ := 12.4
def thomas_apples : ℝ := 8.6

-- The theorem we need to prove
theorem total_apples_picked : mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples = 37.8 := by
    sorry

end total_apples_picked_l1571_157152


namespace increase_in_sold_items_l1571_157165

variable (P N M : ℝ)
variable (discounted_price := 0.9 * P)
variable (increased_total_income := 1.17 * P * N)

theorem increase_in_sold_items (h: 0.9 * P * M = increased_total_income):
  M = 1.3 * N :=
  by sorry

end increase_in_sold_items_l1571_157165


namespace Q_div_P_eq_10_over_3_l1571_157166

noncomputable def solve_Q_over_P (P Q : ℤ) :=
  (Q / P = 10 / 3)

theorem Q_div_P_eq_10_over_3 (P Q : ℤ) (x : ℝ) :
  (∀ x, x ≠ 3 → x ≠ 4 → (P / (x + 3) + Q / (x^2 - 10 * x + 16) = (x^2 - 6 * x + 18) / (x^3 - 7 * x^2 + 14 * x - 48))) →
  solve_Q_over_P P Q :=
sorry

end Q_div_P_eq_10_over_3_l1571_157166


namespace parabola_equation_l1571_157179

theorem parabola_equation (x y : ℝ)
    (focus : x = 1 ∧ y = -2)
    (directrix : 5 * x + 2 * y = 10) :
    4 * x^2 - 20 * x * y + 25 * y^2 + 158 * x + 156 * y + 16 = 0 := 
by
  -- use the given conditions and intermediate steps to derive the final equation
  sorry

end parabola_equation_l1571_157179


namespace ratio_of_horses_to_cows_l1571_157149

/-- Let H and C be the initial number of horses and cows respectively.
Given that:
1. (H - 15) / (C + 15) = 7 / 3,
2. H - 15 = C + 75,
prove that the initial ratio of horses to cows is 4:1. -/
theorem ratio_of_horses_to_cows (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 7 / 3)
  (h2 : H - 15 = C + 75) :
  H / C = 4 :=
by
  sorry

end ratio_of_horses_to_cows_l1571_157149


namespace correct_calculation_l1571_157160

theorem correct_calculation (a : ℕ) :
  ¬ (a^3 + a^4 = a^7) ∧
  ¬ (2 * a - a = 2) ∧
  2 * a + a = 3 * a ∧
  ¬ (a^4 - a^3 = a) :=
by
  sorry

end correct_calculation_l1571_157160


namespace sum_of_fractions_correct_l1571_157196

def sum_of_fractions : ℚ := (4 / 3) + (8 / 9) + (18 / 27) + (40 / 81) + (88 / 243) - 5

theorem sum_of_fractions_correct : sum_of_fractions = -305 / 243 := by
  sorry -- proof to be provided

end sum_of_fractions_correct_l1571_157196


namespace hakeem_artichoke_dip_l1571_157161

theorem hakeem_artichoke_dip 
(total_money : ℝ)
(cost_per_artichoke : ℝ)
(artichokes_per_dip : ℕ)
(dip_per_three_artichokes : ℕ)
(h : total_money = 15)
(h₁ : cost_per_artichoke = 1.25)
(h₂ : artichokes_per_dip = 3)
(h₃ : dip_per_three_artichokes = 5) : 
total_money / cost_per_artichoke * (dip_per_three_artichokes / artichokes_per_dip) = 20 := 
sorry

end hakeem_artichoke_dip_l1571_157161


namespace square_root_of_16_is_pm_4_l1571_157169

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l1571_157169


namespace problem1_simplification_problem2_simplification_l1571_157189

theorem problem1_simplification : (3 / Real.sqrt 3 - (Real.sqrt 3) ^ 2 - Real.sqrt 27 + (abs (Real.sqrt 3 - 2))) = -1 - 3 * Real.sqrt 3 :=
  by
    sorry

theorem problem2_simplification (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  ((x + 2) / (x ^ 2 - 2 * x) - (x - 1) / (x ^ 2 - 4 * x + 4)) / ((x - 4) / x) = 1 / (x - 2) ^ 2 :=
  by
    sorry

end problem1_simplification_problem2_simplification_l1571_157189


namespace simplify_polynomial_l1571_157134

theorem simplify_polynomial :
  (3 * y - 2) * (6 * y ^ 12 + 3 * y ^ 11 + 6 * y ^ 10 + 3 * y ^ 9) =
  18 * y ^ 13 - 3 * y ^ 12 + 12 * y ^ 11 - 3 * y ^ 10 - 6 * y ^ 9 :=
by
  sorry

end simplify_polynomial_l1571_157134


namespace range_of_m_l1571_157154

variable {f : ℝ → ℝ}

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f x > f y

theorem range_of_m (hf_dec : is_decreasing f) (hf_odd : ∀ x, f (-x) = -f x) 
  (h : ∀ m, f (m - 1) + f (2 * m - 1) > 0) : ∀ m, m < 2 / 3 :=
by
  sorry

end range_of_m_l1571_157154


namespace jellybeans_count_l1571_157153

noncomputable def jellybeans_initial (y: ℝ) (n: ℕ) : ℝ :=
  y / (0.7 ^ n)

theorem jellybeans_count (y x: ℝ) (n: ℕ) (h: y = 24) (h2: n = 3) :
  x = 70 :=
by
  apply sorry

end jellybeans_count_l1571_157153


namespace find_ticket_price_l1571_157140

theorem find_ticket_price
  (P : ℝ) -- The original price of each ticket
  (h1 : 10 * 0.6 * P + 20 * 0.85 * P + 26 * P = 980) :
  P = 20 :=
sorry

end find_ticket_price_l1571_157140


namespace possible_values_of_c_l1571_157135

theorem possible_values_of_c (a b c : ℕ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : n = 729 * a + 81 * b + 36 + c) (h₂ : ∃ k, n = k^3) :
  c = 1 ∨ c = 8 :=
sorry

end possible_values_of_c_l1571_157135


namespace intersection_points_count_l1571_157187

theorem intersection_points_count : 
  ∃ (x1 y1 x2 y2 : ℝ), 
  (x1 - ⌊x1⌋)^2 + (y1 - 1)^2 = x1 - ⌊x1⌋ ∧ 
  y1 = 1/5 * x1 + 1 ∧ 
  (x2 - ⌊x2⌋)^2 + (y2 - 1)^2 = x2 - ⌊x2⌋ ∧ 
  y2 = 1/5 * x2 + 1 ∧ 
  (x1, y1) ≠ (x2, y2) :=
sorry

end intersection_points_count_l1571_157187


namespace cut_rectangle_to_square_l1571_157177

theorem cut_rectangle_to_square (a b : ℕ) (h₁ : a = 16) (h₂ : b = 9) :
  ∃ (s : ℕ), s * s = a * b ∧ s = 12 :=
by {
  sorry
}

end cut_rectangle_to_square_l1571_157177


namespace point_in_second_quadrant_l1571_157144

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

def problem_points : List (ℝ × ℝ) :=
  [(1, -2), (2, 1), (-2, -1), (-1, 2)]

theorem point_in_second_quadrant :
  ∃ (p : ℝ × ℝ), p ∈ problem_points ∧ is_in_second_quadrant p.1 p.2 := by
  use (-1, 2)
  sorry

end point_in_second_quadrant_l1571_157144


namespace radius_of_spheres_in_cube_l1571_157186

noncomputable def sphere_radius (sides: ℝ) (spheres: ℕ) (tangent_pairs: ℕ) (tangent_faces: ℕ): ℝ :=
  if sides = 2 ∧ spheres = 10 ∧ tangent_pairs = 2 ∧ tangent_faces = 3 then 0.5 else 0

theorem radius_of_spheres_in_cube : sphere_radius 2 10 2 3 = 0.5 :=
by
  -- This is the main theorem that states the radius of each sphere given the problem conditions.
  sorry

end radius_of_spheres_in_cube_l1571_157186


namespace trig_inequality_l1571_157162

noncomputable def a : ℝ := Real.sin (31 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (58 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (32 * Real.pi / 180)

theorem trig_inequality : c > b ∧ b > a := by
  sorry

end trig_inequality_l1571_157162


namespace max_value_log_div_x_l1571_157136

noncomputable def func (x : ℝ) := (Real.log x) / x

theorem max_value_log_div_x : ∃ x > 0, func x = 1 / Real.exp 1 ∧ 
(∀ t > 0, t ≠ x → func t ≤ func x) :=
sorry

end max_value_log_div_x_l1571_157136


namespace fibonacci_150_mod_7_l1571_157168

def fibonacci_mod_7 : Nat → Nat
| 0 => 0
| 1 => 1
| n + 2 => (fibonacci_mod_7 (n + 1) + fibonacci_mod_7 n) % 7

theorem fibonacci_150_mod_7 : fibonacci_mod_7 150 = 1 := 
by sorry

end fibonacci_150_mod_7_l1571_157168


namespace expression_greater_than_m_l1571_157145

theorem expression_greater_than_m (m : ℚ) : m + 2 > m :=
by sorry

end expression_greater_than_m_l1571_157145


namespace largest_five_digit_congruent_to_18_mod_25_l1571_157105

theorem largest_five_digit_congruent_to_18_mod_25 : 
  ∃ (x : ℕ), x < 100000 ∧ 10000 ≤ x ∧ x % 25 = 18 ∧ x = 99993 :=
by
  sorry

end largest_five_digit_congruent_to_18_mod_25_l1571_157105


namespace find_m_l1571_157180

noncomputable def union_sets (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∨ x ∈ B}

theorem find_m :
  ∀ (m : ℝ),
    (A = {1, 2 ^ m}) →
    (B = {0, 2}) →
    (union_sets A B = {0, 1, 2, 8}) →
    m = 3 :=
by
  intros m hA hB hUnion
  sorry

end find_m_l1571_157180


namespace parallelogram_properties_l1571_157128

noncomputable def perimeter (x y : ℤ) : ℝ :=
  2 * (5 + Real.sqrt ((x - 7) ^ 2 + (y - 3) ^ 2))

noncomputable def area (x y : ℤ) : ℝ :=
  5 * abs (y - 3)

theorem parallelogram_properties (x y : ℤ) (hx : x = 7) (hy : y = 7) :
  (perimeter x y + area x y) = 38 :=
by
  simp [perimeter, area, hx, hy]
  sorry

end parallelogram_properties_l1571_157128


namespace general_term_min_S9_and_S10_sum_b_seq_l1571_157199

-- Definitions for the arithmetic sequence {a_n}
def a_seq (n : ℕ) : ℤ := 2 * ↑n - 20

-- Conditions provided in the problem
def cond1 : Prop := a_seq 4 = -12
def cond2 : Prop := a_seq 8 = -4

-- The sum of the first n terms S_n of the arithmetic sequence {a_n}
def S_n (n : ℕ) : ℤ := n * (a_seq 1 + a_seq n) / 2

-- Definitions for the new sequence {b_n}
def b_seq (n : ℕ) : ℤ := 2^n - 20

-- The sum of the first n terms of the new sequence {b_n}
def T_n (n : ℕ) : ℤ := (2^(n + 1) - 2) - 20 * n

-- Lean 4 theorem statements
theorem general_term (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a_seq n = 2 * ↑n - 20 :=
sorry

theorem min_S9_and_S10 (h1 : cond1) (h2 : cond2) : S_n 9 = -90 ∧ S_n 10 = -90 :=
sorry

theorem sum_b_seq (n : ℕ) : ∀ k : ℕ, (k < n) → T_n k = (2^(k+1) - 20 * k - 2) :=
sorry

end general_term_min_S9_and_S10_sum_b_seq_l1571_157199


namespace find_n_for_primes_l1571_157104

def A_n (n : ℕ) : ℕ := 1 + 7 * (10^n - 1) / 9
def B_n (n : ℕ) : ℕ := 3 + 7 * (10^n - 1) / 9

theorem find_n_for_primes (n : ℕ) :
  (∀ n, n > 0 → (Nat.Prime (A_n n) ∧ Nat.Prime (B_n n)) ↔ n = 1) :=
sorry

end find_n_for_primes_l1571_157104


namespace find_y_when_x_is_1_l1571_157139

theorem find_y_when_x_is_1 (t : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = 1) : 
  y = 11 :=
by
  sorry

end find_y_when_x_is_1_l1571_157139


namespace smaller_two_digit_product_l1571_157151

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l1571_157151


namespace Joe_first_lift_weight_l1571_157110

variable (F S : ℝ)

theorem Joe_first_lift_weight (h1 : F + S = 600) (h2 : 2 * F = S + 300) : F = 300 := 
sorry

end Joe_first_lift_weight_l1571_157110


namespace golden_ratio_minus_one_binary_l1571_157171

theorem golden_ratio_minus_one_binary (n : ℕ → ℕ) (h_n : ∀ i, 1 ≤ n i)
  (h_incr : ∀ i, n i ≤ n (i + 1)): 
  (∀ k ≥ 4, n k ≤ 2^(k - 1) - 2) := 
by
  sorry

end golden_ratio_minus_one_binary_l1571_157171


namespace solve_for_y_l1571_157132

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l1571_157132


namespace combined_garden_area_l1571_157146

-- Definitions for the sizes and counts of the gardens.
def Mancino_gardens : ℕ := 4
def Marquita_gardens : ℕ := 3
def Matteo_gardens : ℕ := 2
def Martina_gardens : ℕ := 5

def Mancino_garden_area : ℕ := 16 * 5
def Marquita_garden_area : ℕ := 8 * 4
def Matteo_garden_area : ℕ := 12 * 6
def Martina_garden_area : ℕ := 10 * 3

-- The total combined area to be proven.
def total_area : ℕ :=
  (Mancino_gardens * Mancino_garden_area) +
  (Marquita_gardens * Marquita_garden_area) +
  (Matteo_gardens * Matteo_garden_area) +
  (Martina_gardens * Martina_garden_area)

-- Proof statement for the combined area.
theorem combined_garden_area : total_area = 710 :=
by sorry

end combined_garden_area_l1571_157146


namespace smallest_area_right_triangle_l1571_157181

theorem smallest_area_right_triangle (a b : ℕ) (h₁ : a = 4) (h₂ : b = 5) : 
  ∃ c, (c = 6 ∧ ∀ (x y : ℕ) (h₃ : x = 4 ∨ y = 4) (h₄ : x = 5 ∨ y = 5), c ≤ (x * y / 2)) :=
by {
  sorry
}

end smallest_area_right_triangle_l1571_157181


namespace curved_surface_area_cone_l1571_157101

variable (a α β : ℝ) (l := a * Real.sin α) (r := a * Real.cos β)

theorem curved_surface_area_cone :
  π * r * l = π * a^2 * Real.sin α * Real.cos β := by
  sorry

end curved_surface_area_cone_l1571_157101


namespace circle_radius_9_l1571_157184

theorem circle_radius_9 (k : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 81) → 
  (k = 94) :=
by
  sorry

end circle_radius_9_l1571_157184


namespace solve_quadratic_equation1_solve_quadratic_equation2_l1571_157138

theorem solve_quadratic_equation1 (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by sorry

theorem solve_quadratic_equation2 (x : ℝ) : (x + 3) * (x - 3) = 3 * (x + 3) ↔ x = -3 ∨ x = 6 :=
by sorry

end solve_quadratic_equation1_solve_quadratic_equation2_l1571_157138


namespace xyz_identity_l1571_157157

theorem xyz_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by
  sorry

end xyz_identity_l1571_157157


namespace reimbursement_diff_l1571_157183

/-- Let Tom, Emma, and Harry share equally the costs for a group activity.
- Tom paid $95
- Emma paid $140
- Harry paid $165
If Tom and Emma are to reimburse Harry to ensure all expenses are shared equally,
prove that e - t = -45 where e is the amount Emma gives Harry and t is the amount Tom gives Harry.
-/
theorem reimbursement_diff :
  let tom_paid := 95
  let emma_paid := 140
  let harry_paid := 165
  let total_cost := tom_paid + emma_paid + harry_paid
  let equal_share := total_cost / 3
  let t := equal_share - tom_paid
  let e := equal_share - emma_paid
  e - t = -45 :=
by {
  sorry
}

end reimbursement_diff_l1571_157183


namespace find_a_l1571_157131

-- Definition of the curve y = x^3 + ax + 1
def curve (x a : ℝ) : ℝ := x^3 + a * x + 1

-- Definition of the tangent line y = 2x + 1
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

-- The slope of the tangent line is 2
def slope_of_tangent_line (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + a

theorem find_a (a : ℝ) : 
  (∃ x₀, curve x₀ a = tangent_line x₀) ∧ (∃ x₀, slope_of_tangent_line x₀ a = 2) → a = 2 :=
by
  sorry

end find_a_l1571_157131


namespace least_positive_integer_l1571_157114

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 1) ∧ (a % 3 = 2) ∧ (a % 4 = 3) ∧ (a % 5 = 4) → a = 59 :=
by
  sorry

end least_positive_integer_l1571_157114


namespace tammy_weekly_distance_l1571_157198

-- Define the conditions.
def track_length : ℕ := 50
def loops_per_day : ℕ := 10
def days_in_week : ℕ := 7

-- Using the conditions, prove the total distance per week is 3500 meters.
theorem tammy_weekly_distance : (track_length * loops_per_day * days_in_week) = 3500 := by
  sorry

end tammy_weekly_distance_l1571_157198


namespace test_question_count_l1571_157150

theorem test_question_count :
  ∃ (x y : ℕ), x + y = 30 ∧ 5 * x + 10 * y = 200 ∧ x = 20 :=
by
  sorry

end test_question_count_l1571_157150


namespace stock_price_no_return_l1571_157142

/-- Define the increase and decrease factors. --/
def increase_factor := 117 / 100
def decrease_factor := 83 / 100

/-- Define the proof that the stock price cannot return to its initial value after any number of 
    increases and decreases. --/
theorem stock_price_no_return 
  (P0 : ℝ) (k l : ℕ) : 
  P0 * (increase_factor ^ k) * (decrease_factor ^ l) ≠ P0 :=
by
  sorry

end stock_price_no_return_l1571_157142


namespace parabola_equation_1_parabola_equation_2_l1571_157109

noncomputable def parabola_vertex_focus (vertex focus : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, (focus.1 = p / 2 ∧ focus.2 = 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 24 * x)

noncomputable def standard_parabola_through_point (point : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, ( ( point.1^2 = 2 * p * point.2 ∧ point.2 ≠ 0 ∧ point.1 ≠ 0) ∧ (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = y / 2) ) ∨
           ( ( point.2^2 = 2 * p * point.1 ∧ point.1 ≠ 0 ∧ point.2 ≠ 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) )

theorem parabola_equation_1 : parabola_vertex_focus (0, 0) (6, 0) := 
  sorry

theorem parabola_equation_2 : standard_parabola_through_point (1, 2) := 
  sorry

end parabola_equation_1_parabola_equation_2_l1571_157109


namespace sin_17pi_over_6_l1571_157190

theorem sin_17pi_over_6 : Real.sin (17 * Real.pi / 6) = 1 / 2 :=
by
  sorry

end sin_17pi_over_6_l1571_157190


namespace sin_squared_identity_l1571_157147

theorem sin_squared_identity :
  1 - 2 * (Real.sin (105 * Real.pi / 180))^2 = - (Real.sqrt 3) / 2 :=
by sorry

end sin_squared_identity_l1571_157147


namespace students_per_class_l1571_157188

theorem students_per_class :
  let buns_per_package := 8
  let packages := 30
  let buns_per_student := 2
  let classes := 4
  (packages * buns_per_package) / (buns_per_student * classes) = 30 :=
by
  sorry

end students_per_class_l1571_157188


namespace abs_tan_45_eq_sqrt3_factor_4x2_36_l1571_157115

theorem abs_tan_45_eq_sqrt3 : abs (1 - Real.sqrt 3) + Real.tan (Real.pi / 4) = Real.sqrt 3 := 
by 
  sorry

theorem factor_4x2_36 (x : ℝ) : 4 * x ^ 2 - 36 = 4 * (x + 3) * (x - 3) := 
by 
  sorry

end abs_tan_45_eq_sqrt3_factor_4x2_36_l1571_157115


namespace min_value_l1571_157182

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) : 
  ∃ c : ℝ, (c = 3 / 4) ∧ (∀ (a b c : ℝ), a = x ∧ b = y ∧ c = z → 
    (1/(a + 3*b) + 1/(b + 3*c) + 1/(c + 3*a)) ≥ c) :=
sorry

end min_value_l1571_157182


namespace prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l1571_157156

noncomputable def prob_first_firing_A : ℚ := 4 / 5
noncomputable def prob_first_firing_B : ℚ := 3 / 4
noncomputable def prob_first_firing_C : ℚ := 2 / 3

noncomputable def prob_second_firing : ℚ := 3 / 5

noncomputable def prob_only_one_success_first_firing :=
  prob_first_firing_A * (1 - prob_first_firing_B) * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * prob_first_firing_B * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * (1 - prob_first_firing_B) * prob_first_firing_C

theorem prob_only_one_success_first_firing_is_correct :
  prob_only_one_success_first_firing = 3 / 20 :=
by sorry

noncomputable def prob_success_after_both_firings_A := prob_first_firing_A * prob_second_firing
noncomputable def prob_success_after_both_firings_B := prob_first_firing_B * prob_second_firing
noncomputable def prob_success_after_both_firings_C := prob_first_firing_C * prob_second_firing

noncomputable def prob_all_success_after_both_firings :=
  prob_success_after_both_firings_A * prob_success_after_both_firings_B * prob_success_after_both_firings_C

theorem prob_all_success_after_both_firings_is_correct :
  prob_all_success_after_both_firings = 54 / 625 :=
by sorry

end prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l1571_157156
