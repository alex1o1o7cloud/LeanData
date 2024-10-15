import Mathlib

namespace NUMINAMATH_GPT_eccentricity_equals_2_l512_51256

variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
variables (eqn_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
variables (focus_F : F = (c, 0)) (imaginary_axis_B : B = (0, b))
variables (intersect_A : A = (c / 3, 2 * b / 3))
variables (vector_eqn : 3 * (A.1, A.2) = (F.1 + 2 * B.1, F.2 + 2 * B.2))
variables (asymptote_eqn : ∀ A1 A2 : ℝ, A2 = (b / a) * A1 → A = (A1, A2))

theorem eccentricity_equals_2 : (c / a = 2) :=
sorry

end NUMINAMATH_GPT_eccentricity_equals_2_l512_51256


namespace NUMINAMATH_GPT_roots_negative_condition_l512_51251

theorem roots_negative_condition (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_root : a * r^2 + b * r + c = 0) (h_neg : r = -s) : b = 0 := sorry

end NUMINAMATH_GPT_roots_negative_condition_l512_51251


namespace NUMINAMATH_GPT_value_proof_l512_51276

noncomputable def find_value (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x) : Prop :=
  2 * b - a + c = 195

theorem value_proof : ∃ (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x), find_value a b c h h_rat :=
  sorry

end NUMINAMATH_GPT_value_proof_l512_51276


namespace NUMINAMATH_GPT_has_two_roots_l512_51245

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end NUMINAMATH_GPT_has_two_roots_l512_51245


namespace NUMINAMATH_GPT_square_area_l512_51285

theorem square_area (p : ℕ) (h : p = 48) : (p / 4) * (p / 4) = 144 := by
  sorry

end NUMINAMATH_GPT_square_area_l512_51285


namespace NUMINAMATH_GPT_find_x_l512_51277

theorem find_x (x : ℝ) (h : (3 * x - 4) / 7 = 15) : x = 109 / 3 :=
by sorry

end NUMINAMATH_GPT_find_x_l512_51277


namespace NUMINAMATH_GPT_quadratic_equation_identify_l512_51248

theorem quadratic_equation_identify {a b c x : ℝ} :
  ((3 - 5 * x^2 = x) ↔ true) ∧
  ((3 / x + x^2 - 1 = 0) ↔ false) ∧
  ((a * x^2 + b * x + c = 0) ↔ (a ≠ 0)) ∧
  ((4 * x - 1 = 0) ↔ false) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_identify_l512_51248


namespace NUMINAMATH_GPT_int_sol_many_no_int_sol_l512_51262

-- Part 1: If there is one integer solution, there are at least three integer solutions
theorem int_sol_many (n : ℤ) (hn : n > 0) (x y : ℤ) 
  (hxy : x^3 - 3 * x * y^2 + y^3 = n) : 
  ∃ a b c d e f : ℤ, 
    (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (e, f) ≠ (x, y) ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) ∧ 
    a^3 - 3 * a * b^2 + b^3 = n ∧ 
    c^3 - 3 * c * d^2 + d^3 = n ∧ 
    e^3 - 3 * e * f^2 + f^3 = n :=
sorry

-- Part 2: When n = 2891, the equation has no integer solutions
theorem no_int_sol : ¬ ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end NUMINAMATH_GPT_int_sol_many_no_int_sol_l512_51262


namespace NUMINAMATH_GPT_total_items_purchased_l512_51287

/-- Proof that Ike and Mike buy a total of 9 items given the constraints. -/
theorem total_items_purchased
  (total_money : ℝ)
  (sandwich_cost : ℝ)
  (drink_cost : ℝ)
  (combo_factor : ℕ)
  (money_spent_on_sandwiches : ℝ)
  (number_of_sandwiches : ℕ)
  (number_of_drinks : ℕ)
  (num_free_sandwiches : ℕ) :
  total_money = 40 →
  sandwich_cost = 5 →
  drink_cost = 1.5 →
  combo_factor = 5 →
  number_of_sandwiches = 9 →
  number_of_drinks = 0 →
  money_spent_on_sandwiches = number_of_sandwiches * sandwich_cost →
  total_money = money_spent_on_sandwiches →
  num_free_sandwiches = number_of_sandwiches / combo_factor →
  number_of_sandwiches = number_of_sandwiches + num_free_sandwiches →
  number_of_sandwiches + number_of_drinks = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_items_purchased_l512_51287


namespace NUMINAMATH_GPT_total_fat_served_l512_51237

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end NUMINAMATH_GPT_total_fat_served_l512_51237


namespace NUMINAMATH_GPT_find_b_l512_51230

theorem find_b (a b : ℝ) (B C : ℝ)
    (h1 : a * b = 60 * Real.sqrt 3)
    (h2 : Real.sin B = Real.sin C)
    (h3 : 15 * Real.sqrt 3 = 1/2 * a * b * Real.sin C) :
  b = 2 * Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_find_b_l512_51230


namespace NUMINAMATH_GPT_ruby_siblings_l512_51244

structure Child :=
  (name : String)
  (eye_color : String)
  (hair_color : String)

def children : List Child :=
[
  {name := "Mason", eye_color := "Green", hair_color := "Red"},
  {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"},
  {name := "Fiona", eye_color := "Brown", hair_color := "Red"},
  {name := "Leo", eye_color := "Green", hair_color := "Blonde"},
  {name := "Ivy", eye_color := "Green", hair_color := "Red"},
  {name := "Carlos", eye_color := "Green", hair_color := "Blonde"}
]

def is_sibling_group (c1 c2 c3 : Child) : Prop :=
  (c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color) ∧
  (c2.eye_color = c3.eye_color ∨ c2.hair_color = c3.hair_color) ∧
  (c1.eye_color = c3.eye_color ∨ c1.hair_color = c3.hair_color)

theorem ruby_siblings :
  ∃ (c1 c2 : Child), 
    c1.name ≠ "Ruby" ∧ c2.name ≠ "Ruby" ∧
    c1 ≠ c2 ∧
    is_sibling_group {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"} c1 c2 ∧
    ((c1.name = "Leo" ∧ c2.name = "Carlos") ∨ (c1.name = "Carlos" ∧ c2.name = "Leo")) :=
by
  sorry

end NUMINAMATH_GPT_ruby_siblings_l512_51244


namespace NUMINAMATH_GPT_fuel_spending_reduction_l512_51269

-- Define the variables and the conditions
variable (x c : ℝ) -- x for efficiency and c for cost
variable (newEfficiency oldEfficiency newCost oldCost : ℝ)

-- Define the conditions
def conditions := (oldEfficiency = x) ∧ (newEfficiency = 1.75 * oldEfficiency)
                 ∧ (oldCost = c) ∧ (newCost = 1.3 * oldCost)

-- Define the expected reduction in cost
def expectedReduction : ℝ := 25.7142857142857 -- approximately 25 5/7 %

-- Define the assertion that Elmer will reduce his fuel spending by the expected reduction percentage
theorem fuel_spending_reduction : conditions x c oldEfficiency newEfficiency oldCost newCost →
  ((oldCost - (newCost / newEfficiency) * oldEfficiency) / oldCost) * 100 = expectedReduction :=
by
  sorry

end NUMINAMATH_GPT_fuel_spending_reduction_l512_51269


namespace NUMINAMATH_GPT_binom_sub_floor_div_prime_l512_51281

theorem binom_sub_floor_div_prime {n p : ℕ} (hp : Nat.Prime p) (hpn : n ≥ p) : 
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end NUMINAMATH_GPT_binom_sub_floor_div_prime_l512_51281


namespace NUMINAMATH_GPT_Marty_paint_combinations_l512_51267

theorem Marty_paint_combinations :
  let colors := 5 -- blue, green, yellow, black, white
  let styles := 3 -- brush, roller, sponge
  let invalid_combinations := 1 * 1 -- white paint with roller
  let total_combinations := (4 * styles) + (1 * (styles - 1))
  total_combinations = 14 :=
by
  -- Define the total number of combinations excluding the invalid one
  let colors := 5
  let styles := 3
  let invalid_combinations := 1 -- number of invalid combinations (white with roller)
  let valid_combinations := (4 * styles) + (1 * (styles - 1))
  show valid_combinations = 14
  {
    exact rfl -- This will assert that the valid_combinations indeed equals 14
  }

end NUMINAMATH_GPT_Marty_paint_combinations_l512_51267


namespace NUMINAMATH_GPT_sum_infinite_geometric_series_l512_51210

theorem sum_infinite_geometric_series (a r : ℚ) (h : a = 1) (h2 : r = 1/4) : 
  (∀ S, S = a / (1 - r) → S = 4 / 3) :=
by
  intros S hS
  rw [h, h2] at hS
  simp [hS]
  sorry

end NUMINAMATH_GPT_sum_infinite_geometric_series_l512_51210


namespace NUMINAMATH_GPT_count_valid_permutations_eq_X_l512_51211

noncomputable def valid_permutations_count : ℕ :=
sorry

theorem count_valid_permutations_eq_X : valid_permutations_count = X :=
sorry

end NUMINAMATH_GPT_count_valid_permutations_eq_X_l512_51211


namespace NUMINAMATH_GPT_new_outsiders_count_l512_51295

theorem new_outsiders_count (total_people: ℕ) (initial_snackers: ℕ)
  (first_group_outsiders: ℕ) (first_group_leave_half: ℕ) 
  (second_group_leave_count: ℕ) (half_remaining_leave: ℕ) (final_snackers: ℕ) 
  (total_snack_eaters: ℕ) 
  (initial_snackers_eq: total_people = 200) 
  (snackers_eq: initial_snackers = 100) 
  (first_group_outsiders_eq: first_group_outsiders = 20) 
  (first_group_leave_half_eq: first_group_leave_half = 60) 
  (second_group_leave_count_eq: second_group_leave_count = 30) 
  (half_remaining_leave_eq: half_remaining_leave = 15) 
  (final_snackers_eq: final_snackers = 20) 
  (total_snack_eaters_eq: total_snack_eaters = 120): 
  (60 - (second_group_leave_count + half_remaining_leave + final_snackers)) = 40 := 
by sorry

end NUMINAMATH_GPT_new_outsiders_count_l512_51295


namespace NUMINAMATH_GPT_intercepts_line_5x_minus_2y_minus_10_eq_0_l512_51213

theorem intercepts_line_5x_minus_2y_minus_10_eq_0 :
  ∃ a b : ℝ, (a = 2 ∧ b = -5) ∧ (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → 
     ((y = 0 ∧ x = a) ∨ (x = 0 ∧ y = b))) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_line_5x_minus_2y_minus_10_eq_0_l512_51213


namespace NUMINAMATH_GPT_boxes_of_nuts_purchased_l512_51204

theorem boxes_of_nuts_purchased (b : ℕ) (n : ℕ) (bolts_used : ℕ := 7 * 11 - 3) 
    (nuts_used : ℕ := 113 - bolts_used) (total_nuts : ℕ := nuts_used + 6) 
    (nuts_per_box : ℕ := 15) (h_bolts_boxes : b = 7) 
    (h_bolts_per_box : ∀ x, b * x = 77) 
    (h_nuts_boxes : ∃ x, n = x * nuts_per_box)
    : ∃ k, n = k * 15 ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_boxes_of_nuts_purchased_l512_51204


namespace NUMINAMATH_GPT_LCM_of_8_and_12_l512_51236

-- Definitions based on the provided conditions
def a : ℕ := 8
def x : ℕ := 12

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Conditions
def hcf_condition : HCF a x = 4 := by sorry
def x_condition : x = 12 := rfl

-- The proof statement
theorem LCM_of_8_and_12 : LCM a x = 24 :=
by
  have h1 : HCF a x = 4 := hcf_condition
  have h2 : x = 12 := x_condition
  rw [h2] at h1
  sorry

end NUMINAMATH_GPT_LCM_of_8_and_12_l512_51236


namespace NUMINAMATH_GPT_random_event_l512_51298

theorem random_event (a b : ℝ) (h1 : a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0):
  ¬ (∀ a b, a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0 → a + b < 0) :=
by
  sorry

end NUMINAMATH_GPT_random_event_l512_51298


namespace NUMINAMATH_GPT_team_points_behind_l512_51282

-- Define the points for Max, Dulce and the condition for Val
def max_points : ℕ := 5
def dulce_points : ℕ := 3
def combined_points_max_dulce : ℕ := max_points + dulce_points
def val_points : ℕ := 2 * combined_points_max_dulce

-- Define the total points for their team and the opponents' team
def their_team_points : ℕ := max_points + dulce_points + val_points
def opponents_team_points : ℕ := 40

-- Proof statement
theorem team_points_behind : opponents_team_points - their_team_points = 16 :=
by
  sorry

end NUMINAMATH_GPT_team_points_behind_l512_51282


namespace NUMINAMATH_GPT_sqrt_domain_l512_51206

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end NUMINAMATH_GPT_sqrt_domain_l512_51206


namespace NUMINAMATH_GPT_find_function_l512_51259

theorem find_function (f : ℝ → ℝ) :
  (∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)) →
  (∀ u : ℝ, 0 ≤ f u) →
  (∀ x : ℝ, f x = 0) := 
  by
    sorry

end NUMINAMATH_GPT_find_function_l512_51259


namespace NUMINAMATH_GPT_more_divisible_by_7_than_11_l512_51299

open Nat

theorem more_divisible_by_7_than_11 :
  let N := 10000
  let count_7_not_11 := (N / 7) - (N / 77)
  let count_11_not_7 := (N / 11) - (N / 77)
  count_7_not_11 > count_11_not_7 := 
  by
    let N := 10000
    let count_7_not_11 := (N / 7) - (N / 77)
    let count_11_not_7 := (N / 11) - (N / 77)
    sorry

end NUMINAMATH_GPT_more_divisible_by_7_than_11_l512_51299


namespace NUMINAMATH_GPT_train_speed_l512_51203

theorem train_speed (distance : ℝ) (time : ℝ) (distance_eq : distance = 270) (time_eq : time = 9)
  : (distance / time) * (3600 / 1000) = 108 :=
by 
  sorry

end NUMINAMATH_GPT_train_speed_l512_51203


namespace NUMINAMATH_GPT_evaluate_expression_l512_51243

theorem evaluate_expression :
  let c := (-2 : ℚ)
  let x := (2 : ℚ) / 5
  let y := (3 : ℚ) / 5
  let z := (-3 : ℚ)
  c * x^3 * y^4 * z^2 = (-11664) / 78125 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l512_51243


namespace NUMINAMATH_GPT_necessary_condition_l512_51252

theorem necessary_condition {x m : ℝ} 
  (p : |1 - (x - 1) / 3| ≤ 2)
  (q : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (hm : m > 0)
  (h_np_nq : ¬(|1 - (x - 1) / 3| ≤ 2) → ¬(x^2 - 2 * x + 1 - m^2 ≤ 0))
  : m ≥ 9 :=
sorry

end NUMINAMATH_GPT_necessary_condition_l512_51252


namespace NUMINAMATH_GPT_Yeonseo_skirts_l512_51274

theorem Yeonseo_skirts
  (P : ℕ)
  (more_than_two_skirts : ∀ S : ℕ, S > 2)
  (more_than_two_pants : P > 2)
  (ways_to_choose : P + 3 = 7) :
  ∃ S : ℕ, S = 3 := by
  sorry

end NUMINAMATH_GPT_Yeonseo_skirts_l512_51274


namespace NUMINAMATH_GPT_variance_of_data_l512_51225

def data : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.foldr (λ x acc => x + acc) 0) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.foldr (λ x acc => (x - m) ^ 2 + acc) 0) / l.length

theorem variance_of_data :
  variance data = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_data_l512_51225


namespace NUMINAMATH_GPT_exists_segment_l512_51216

theorem exists_segment (f : ℚ → ℤ) : 
  ∃ (a b c : ℚ), a ≠ b ∧ c = (a + b) / 2 ∧ f a + f b ≤ 2 * f c :=
by 
  sorry

end NUMINAMATH_GPT_exists_segment_l512_51216


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l512_51265

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 5*x - 14 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 7} :=
by
  -- proof to be filled here
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l512_51265


namespace NUMINAMATH_GPT_tiles_covering_the_floor_l512_51272

theorem tiles_covering_the_floor 
  (L W : ℕ) 
  (h1 : (∃ k, L = 10 * k) ∧ (∃ j, W = 10 * j))
  (h2 : W = 2 * L)
  (h3 : (L * L + W * W).sqrt = 45) :
  L * W = 810 :=
sorry

end NUMINAMATH_GPT_tiles_covering_the_floor_l512_51272


namespace NUMINAMATH_GPT_min_value_hyperbola_l512_51273

theorem min_value_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ e : ℝ, e = 2 ∧ (b^2 = (e * a)^2 - a^2)) :
  (a * 3 + 1 / a) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_hyperbola_l512_51273


namespace NUMINAMATH_GPT_difference_in_roses_and_orchids_l512_51200

theorem difference_in_roses_and_orchids
    (initial_roses : ℕ) (initial_orchids : ℕ) (initial_tulips : ℕ)
    (final_roses : ℕ) (final_orchids : ℕ) (final_tulips : ℕ)
    (ratio_roses_orchids_num : ℕ) (ratio_roses_orchids_den : ℕ)
    (ratio_roses_tulips_num : ℕ) (ratio_roses_tulips_den : ℕ)
    (h1 : initial_roses = 7)
    (h2 : initial_orchids = 12)
    (h3 : initial_tulips = 5)
    (h4 : final_roses = 11)
    (h5 : final_orchids = 20)
    (h6 : final_tulips = 10)
    (h7 : ratio_roses_orchids_num = 2)
    (h8 : ratio_roses_orchids_den = 5)
    (h9 : ratio_roses_tulips_num = 3)
    (h10 : ratio_roses_tulips_den = 5)
    (h11 : (final_roses : ℚ) / final_orchids = (ratio_roses_orchids_num : ℚ) / ratio_roses_orchids_den)
    (h12 : (final_roses : ℚ) / final_tulips = (ratio_roses_tulips_num : ℚ) / ratio_roses_tulips_den)
    : final_orchids - final_roses = 9 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_roses_and_orchids_l512_51200


namespace NUMINAMATH_GPT_baseball_singles_percentage_l512_51280

theorem baseball_singles_percentage :
  let total_hits := 50
  let home_runs := 2
  let triples := 3
  let doubles := 8
  let non_single_hits := home_runs + triples + doubles
  let singles := total_hits - non_single_hits
  let singles_percentage := (singles / total_hits) * 100
  singles = 37 ∧ singles_percentage = 74 :=
by
  sorry

end NUMINAMATH_GPT_baseball_singles_percentage_l512_51280


namespace NUMINAMATH_GPT_find_n_l512_51235

variable (x n : ℕ)
variable (y : ℕ) {h1 : y = 24}

theorem find_n
  (h1 : y = 24) 
  (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : 
  n = 6 := 
sorry

end NUMINAMATH_GPT_find_n_l512_51235


namespace NUMINAMATH_GPT_find_number_l512_51231

theorem find_number (x : ℕ) (h : 3 * (x + 2) = 24 + x) : x = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l512_51231


namespace NUMINAMATH_GPT_minimum_value_of_f_maximum_value_of_k_l512_51268

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f : ∃ x : ℝ, 0 < x ∧ f x = -1 / Real.exp 1 :=
sorry

theorem maximum_value_of_k : ∀ x > 2, ∀ k : ℤ, (f x ≥ k * x - 2 * (k + 1)) → k ≤ 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_maximum_value_of_k_l512_51268


namespace NUMINAMATH_GPT_range_of_a_l512_51227

theorem range_of_a (a : ℝ) (h : ∃ x1 x2, x1 ≠ x2 ∧ 3 * x1^2 + a = 0 ∧ 3 * x2^2 + a = 0) : a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l512_51227


namespace NUMINAMATH_GPT_maximum_value_of_rocks_l512_51257

theorem maximum_value_of_rocks (R6_val R3_val R2_val : ℕ)
  (R6_wt R3_wt R2_wt : ℕ)
  (num6 num3 num2 : ℕ) :
  R6_val = 16 →
  R3_val = 9 →
  R2_val = 3 →
  R6_wt = 6 →
  R3_wt = 3 →
  R2_wt = 2 →
  30 ≤ num6 →
  30 ≤ num3 →
  30 ≤ num2 →
  ∃ (x6 x3 x2 : ℕ),
    x6 ≤ 4 ∧
    x3 ≤ 4 ∧
    x2 ≤ 4 ∧
    (x6 * R6_wt + x3 * R3_wt + x2 * R2_wt ≤ 24) ∧
    (x6 * R6_val + x3 * R3_val + x2 * R2_val = 68) :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_rocks_l512_51257


namespace NUMINAMATH_GPT_swimming_pool_time_l512_51242

theorem swimming_pool_time 
  (empty_rate : ℕ) (fill_rate : ℕ) (capacity : ℕ) (final_volume : ℕ) (t : ℕ)
  (h_empty : empty_rate = 120 / 4) 
  (h_fill : fill_rate = 120 / 6) 
  (h_capacity : capacity = 120) 
  (h_final : final_volume = 90) 
  (h_eq : capacity - (empty_rate - fill_rate) * t = final_volume) :
  t = 3 := 
sorry

end NUMINAMATH_GPT_swimming_pool_time_l512_51242


namespace NUMINAMATH_GPT_vertex_C_path_length_equals_l512_51220

noncomputable def path_length_traversed_by_C (AB BC CA : ℝ) (PQ QR : ℝ) : ℝ :=
  let BC := 3  -- length of side BC is 3 inches
  let AB := 2  -- length of side AB is 2 inches
  let CA := 4  -- length of side CA is 4 inches
  let PQ := 8  -- length of side PQ of the rectangle is 8 inches
  let QR := 6  -- length of side QR of the rectangle is 6 inches
  4 * BC * Real.pi

theorem vertex_C_path_length_equals (AB BC CA PQ QR : ℝ) :
  AB = 2 ∧ BC = 3 ∧ CA = 4 ∧ PQ = 8 ∧ QR = 6 →
  path_length_traversed_by_C AB BC CA PQ QR = 12 * Real.pi :=
by
  intros h
  have hAB : AB = 2 := h.1
  have hBC : BC = 3 := h.2.1
  have hCA : CA = 4 := h.2.2.1
  have hPQ : PQ = 8 := h.2.2.2.1
  have hQR : QR = 6 := h.2.2.2.2
  simp [path_length_traversed_by_C, hAB, hBC, hCA, hPQ, hQR]
  sorry

end NUMINAMATH_GPT_vertex_C_path_length_equals_l512_51220


namespace NUMINAMATH_GPT_sum_faces_of_cube_l512_51296

-- Conditions in Lean 4
variables (a b c d e f : ℕ)

-- Sum of vertex labels
def vertex_sum := a * b * c + a * e * c + a * b * f + a * e * f +
                  d * b * c + d * e * c + d * b * f + d * e * f

-- Theorem statement
theorem sum_faces_of_cube (h : vertex_sum a b c d e f = 1001) :
  (a + d) + (b + e) + (c + f) = 31 :=
sorry

end NUMINAMATH_GPT_sum_faces_of_cube_l512_51296


namespace NUMINAMATH_GPT_point_on_x_axis_point_on_y_axis_l512_51208

section
-- Definitions for the conditions
def point_A (a : ℝ) : ℝ × ℝ := (a - 3, a ^ 2 - 4)

-- Proof for point A lying on the x-axis
theorem point_on_x_axis (a : ℝ) (h : (point_A a).2 = 0) :
  point_A a = (-1, 0) ∨ point_A a = (-5, 0) :=
sorry

-- Proof for point A lying on the y-axis
theorem point_on_y_axis (a : ℝ) (h : (point_A a).1 = 0) :
  point_A a = (0, 5) :=
sorry
end

end NUMINAMATH_GPT_point_on_x_axis_point_on_y_axis_l512_51208


namespace NUMINAMATH_GPT_greatest_integer_x_l512_51293

theorem greatest_integer_x (x : ℤ) : (5 : ℚ)/8 > (x : ℚ)/15 → x ≤ 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_integer_x_l512_51293


namespace NUMINAMATH_GPT_correct_calculation_of_exponentiation_l512_51294

theorem correct_calculation_of_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_of_exponentiation_l512_51294


namespace NUMINAMATH_GPT_intersection_of_sets_l512_51229

noncomputable def setA : Set ℕ := { x : ℕ | x^2 ≤ 4 * x ∧ x > 0 }

noncomputable def setB : Set ℕ := { x : ℕ | 2^x - 4 > 0 ∧ 2^x - 4 ≤ 4 }

theorem intersection_of_sets : { x ∈ setA | x ∈ setB } = {3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l512_51229


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l512_51290

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | x < 1}
def expected_intersection : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = expected_intersection :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l512_51290


namespace NUMINAMATH_GPT_deposit_on_Jan_1_2008_l512_51270

-- Let a be the initial deposit amount in yuan.
-- Let x be the annual interest rate.

def compound_interest (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  a * (1 + x) ^ n

theorem deposit_on_Jan_1_2008 (a : ℝ) (x : ℝ) : 
  compound_interest a x 5 = a * (1 + x) ^ 5 := 
by
  sorry

end NUMINAMATH_GPT_deposit_on_Jan_1_2008_l512_51270


namespace NUMINAMATH_GPT_books_left_over_l512_51283

-- Define the conditions as variables in Lean
def total_books : ℕ := 1500
def new_shelf_capacity : ℕ := 28

-- State the theorem based on these conditions
theorem books_left_over : total_books % new_shelf_capacity = 14 :=
by
  sorry

end NUMINAMATH_GPT_books_left_over_l512_51283


namespace NUMINAMATH_GPT_average_monthly_balance_l512_51218

-- Definitions for the monthly balances
def January_balance : ℝ := 120
def February_balance : ℝ := 240
def March_balance : ℝ := 180
def April_balance : ℝ := 180
def May_balance : ℝ := 160
def June_balance : ℝ := 200

-- The average monthly balance theorem statement
theorem average_monthly_balance : 
    (January_balance + February_balance + March_balance + April_balance + May_balance + June_balance) / 6 = 180 := 
by 
  sorry

end NUMINAMATH_GPT_average_monthly_balance_l512_51218


namespace NUMINAMATH_GPT_ratio_of_tax_revenue_to_cost_of_stimulus_l512_51223

-- Definitions based on the identified conditions
def bottom_20_percent_people (total_people : ℕ) : ℕ := (total_people * 20) / 100
def stimulus_per_person : ℕ := 2000
def total_people : ℕ := 1000
def government_profit : ℕ := 1600000

-- Cost of the stimulus
def cost_of_stimulus : ℕ := bottom_20_percent_people total_people * stimulus_per_person

-- Tax revenue returned to the government
def tax_revenue : ℕ := government_profit + cost_of_stimulus

-- The Proposition we need to prove
theorem ratio_of_tax_revenue_to_cost_of_stimulus :
  tax_revenue / cost_of_stimulus = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_tax_revenue_to_cost_of_stimulus_l512_51223


namespace NUMINAMATH_GPT_florist_picked_roses_l512_51234

def initial_roses : ℕ := 11
def sold_roses : ℕ := 2
def final_roses : ℕ := 41
def remaining_roses := initial_roses - sold_roses
def picked_roses := final_roses - remaining_roses

theorem florist_picked_roses : picked_roses = 32 :=
by
  -- This is where the proof would go, but we are leaving it empty on purpose
  sorry

end NUMINAMATH_GPT_florist_picked_roses_l512_51234


namespace NUMINAMATH_GPT_polynomial_value_l512_51289

theorem polynomial_value (a : ℝ) (h : a^2 + 2 * a = 1) : 
  2 * a^5 + 7 * a^4 + 5 * a^3 + 2 * a^2 + 5 * a + 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l512_51289


namespace NUMINAMATH_GPT_func_C_increasing_l512_51253

open Set

noncomputable def func_A (x : ℝ) : ℝ := 3 - x
noncomputable def func_B (x : ℝ) : ℝ := x^2 - x
noncomputable def func_C (x : ℝ) : ℝ := -1 / (x + 1)
noncomputable def func_D (x : ℝ) : ℝ := -abs x

theorem func_C_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → func_C x < func_C y := by
  sorry

end NUMINAMATH_GPT_func_C_increasing_l512_51253


namespace NUMINAMATH_GPT_sweets_leftover_candies_l512_51255

theorem sweets_leftover_candies (n : ℕ) (h : n % 8 = 5) : (3 * n) % 8 = 7 :=
sorry

end NUMINAMATH_GPT_sweets_leftover_candies_l512_51255


namespace NUMINAMATH_GPT_nadia_pies_l512_51219

variables (T R B S : ℕ)

theorem nadia_pies (h₁: R = T / 2) 
                   (h₂: B = R - 14) 
                   (h₃: S = (R + B) / 2) 
                   (h₄: T = R + B + S) :
                   R = 21 ∧ B = 7 ∧ S = 14 := 
  sorry

end NUMINAMATH_GPT_nadia_pies_l512_51219


namespace NUMINAMATH_GPT_eval_g_at_3_l512_51271

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem eval_g_at_3 : g 3 = 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_eval_g_at_3_l512_51271


namespace NUMINAMATH_GPT__l512_51291

variables (a b c : ℝ)
-- Conditionally define the theorem giving the constraints in the context.
example (h1 : a < 0) (h2 : b < 0) (h3 : c > 0) : 
  abs a - abs (a + b) + abs (c - a) + abs (b - c) = 2 * c - a := by 
sorry

end NUMINAMATH_GPT__l512_51291


namespace NUMINAMATH_GPT_factorize_expression_l512_51286

theorem factorize_expression (a b m : ℝ) :
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l512_51286


namespace NUMINAMATH_GPT_rectangular_garden_width_l512_51221

-- Define the problem conditions as Lean definitions
def rectangular_garden_length (w : ℝ) : ℝ := 3 * w
def rectangular_garden_area (w : ℝ) : ℝ := rectangular_garden_length w * w

-- This is the theorem we want to prove
theorem rectangular_garden_width : ∃ w : ℝ, rectangular_garden_area w = 432 ∧ w = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_garden_width_l512_51221


namespace NUMINAMATH_GPT_max_xy_l512_51279

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  xy ≤ 1 / 4 := 
sorry

end NUMINAMATH_GPT_max_xy_l512_51279


namespace NUMINAMATH_GPT_distance_between_polar_points_l512_51205

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_polar_points_l512_51205


namespace NUMINAMATH_GPT_trigonometric_signs_problem_l512_51284

open Real

theorem trigonometric_signs_problem (k : ℤ) (θ α : ℝ) 
  (hα : α = 2 * k * π - π / 5)
  (h_terminal_side : ∃ m : ℤ, θ = α + 2 * m * π) :
  (sin θ / |sin θ|) + (cos θ / |cos θ|) + (tan θ / |tan θ|) = -1 := 
sorry

end NUMINAMATH_GPT_trigonometric_signs_problem_l512_51284


namespace NUMINAMATH_GPT_number_of_trees_planted_l512_51264

-- Definition of initial conditions
def initial_trees : ℕ := 22
def final_trees : ℕ := 55

-- Theorem stating the number of trees planted
theorem number_of_trees_planted : final_trees - initial_trees = 33 := by
  sorry

end NUMINAMATH_GPT_number_of_trees_planted_l512_51264


namespace NUMINAMATH_GPT_sum_possible_x_values_in_isosceles_triangle_l512_51232

def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

def valid_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem sum_possible_x_values_in_isosceles_triangle :
  ∃ (x1 x2 x3 : ℝ), isosceles_triangle 80 x1 x1 ∧ isosceles_triangle x2 80 80 ∧ isosceles_triangle 80 x3 x3 ∧ 
  valid_triangle 80 x1 x1 ∧ valid_triangle x2 80 80 ∧ valid_triangle 80 x3 x3 ∧ 
  x1 + x2 + x3 = 150 :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_x_values_in_isosceles_triangle_l512_51232


namespace NUMINAMATH_GPT_company_x_installation_charge_l512_51240

theorem company_x_installation_charge:
  let price_X := 575
  let surcharge_X := 0.04 * price_X
  let installation_charge_X := 82.50
  let total_cost_X := price_X + surcharge_X + installation_charge_X
  let price_Y := 530
  let surcharge_Y := 0.03 * price_Y
  let installation_charge_Y := 93.00
  let total_cost_Y := price_Y + surcharge_Y + installation_charge_Y
  let savings := 41.60
  total_cost_X - total_cost_Y = savings → installation_charge_X = 82.50 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_company_x_installation_charge_l512_51240


namespace NUMINAMATH_GPT_max_happy_monkeys_l512_51215

-- Definitions for given problem
def pears := 20
def bananas := 30
def peaches := 40
def mandarins := 50
def fruits (x y : Nat) := x + y

-- The theorem to prove
theorem max_happy_monkeys : 
  ∃ (m : Nat), m = (pears + bananas + peaches) / 2 ∧ m ≤ mandarins :=
by
  sorry

end NUMINAMATH_GPT_max_happy_monkeys_l512_51215


namespace NUMINAMATH_GPT_range_of_n_l512_51275

theorem range_of_n (m n : ℝ) (h : (m^2 - 2 * m)^2 + 4 * m^2 - 8 * m + 6 - n = 0) : n ≥ 3 :=
sorry

end NUMINAMATH_GPT_range_of_n_l512_51275


namespace NUMINAMATH_GPT_at_least_one_vowel_l512_51217

-- Define the set of letters
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'I'}

-- Define the vowels within the set of letters
def vowels : Finset Char := {'A', 'E', 'I'}

-- Define the consonants within the set of letters
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

-- Function to count the total number of 3-letter words from a given set
def count_words (s : Finset Char) (length : Nat) : Nat :=
  s.card ^ length

-- Define the statement of the problem
theorem at_least_one_vowel : count_words letters 3 - count_words consonants 3 = 279 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_vowel_l512_51217


namespace NUMINAMATH_GPT_sum_of_fractions_l512_51249

theorem sum_of_fractions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) :
  f (1 / 8) + f (2 / 8) + f (3 / 8) + f (4 / 8) + 
  f (5 / 8) + f (6 / 8) + f (7 / 8) = 7 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l512_51249


namespace NUMINAMATH_GPT_length_of_QB_l512_51288

/-- 
Given a circle Q with a circumference of 16π feet, 
segment AB as its diameter, 
and the angle AQB of 120 degrees, 
prove that the length of segment QB is 8 feet.
-/
theorem length_of_QB (C : ℝ) (r : ℝ) (A B Q : ℝ) (angle_AQB : ℝ) 
  (h1 : C = 16 * Real.pi)
  (h2 : 2 * Real.pi * r = C)
  (h3 : angle_AQB = 120) 
  : QB = 8 :=
sorry

end NUMINAMATH_GPT_length_of_QB_l512_51288


namespace NUMINAMATH_GPT_total_area_of_tickets_is_3_6_m2_l512_51214

def area_of_one_ticket (side_length_cm : ℕ) : ℕ :=
  side_length_cm * side_length_cm

def total_tickets (people : ℕ) (tickets_per_person : ℕ) : ℕ :=
  people * tickets_per_person

def total_area_cm2 (area_per_ticket_cm2 : ℕ) (number_of_tickets : ℕ) : ℕ :=
  area_per_ticket_cm2 * number_of_tickets

def convert_cm2_to_m2 (area_cm2 : ℕ) : ℚ :=
  (area_cm2 : ℚ) / 10000

theorem total_area_of_tickets_is_3_6_m2 :
  let side_length := 30
  let people := 5
  let tickets_per_person := 8
  let one_ticket_area := area_of_one_ticket side_length
  let number_of_tickets := total_tickets people tickets_per_person
  let total_area_cm2 := total_area_cm2 one_ticket_area number_of_tickets
  let total_area_m2 := convert_cm2_to_m2 total_area_cm2
  total_area_m2 = 3.6 := 
by
  sorry

end NUMINAMATH_GPT_total_area_of_tickets_is_3_6_m2_l512_51214


namespace NUMINAMATH_GPT_tourists_left_l512_51254

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end NUMINAMATH_GPT_tourists_left_l512_51254


namespace NUMINAMATH_GPT_kim_paints_fewer_tiles_than_laura_l512_51212

-- Given conditions and definitions
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def total_tiles_per_15_minutes : ℕ := 375
def total_rate_per_minute : ℕ := total_tiles_per_15_minutes / 15
def kim_rate : ℕ := total_rate_per_minute - (don_rate + ken_rate + laura_rate)

-- Proof goal
theorem kim_paints_fewer_tiles_than_laura :
  laura_rate - kim_rate = 3 :=
by
  sorry

end NUMINAMATH_GPT_kim_paints_fewer_tiles_than_laura_l512_51212


namespace NUMINAMATH_GPT_exponent_on_right_side_l512_51258

theorem exponent_on_right_side (n : ℕ) (h : n = 17) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 :=
by
  sorry

end NUMINAMATH_GPT_exponent_on_right_side_l512_51258


namespace NUMINAMATH_GPT_different_outcomes_count_l512_51228

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 3

-- Define the proof statement
theorem different_outcomes_count : (num_competitions ^ num_students) = 81 := 
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_different_outcomes_count_l512_51228


namespace NUMINAMATH_GPT_boat_speed_still_water_l512_51260

/-- Proof that the speed of the boat in still water is 10 km/hr given the conditions -/
theorem boat_speed_still_water (V_b V_s : ℝ) 
  (cond1 : V_b + V_s = 15) 
  (cond2 : V_b - V_s = 5) : 
  V_b = 10 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l512_51260


namespace NUMINAMATH_GPT_line_through_point_perpendicular_y_axis_line_through_two_points_l512_51202

-- The first problem
theorem line_through_point_perpendicular_y_axis :
  ∃ (k : ℝ), ∀ (x : ℝ), k = 1 → y = k :=
sorry

-- The second problem
theorem line_through_two_points (x1 y1 x2 y2 : ℝ) (hA : (x1, y1) = (-4, 0)) (hB : (x2, y2) = (0, 6)) :
  ∃ (a b c : ℝ), (a, b, c) = (3, -2, 12) → ∀ (x y : ℝ), a * x + b * y + c = 0 :=
sorry

end NUMINAMATH_GPT_line_through_point_perpendicular_y_axis_line_through_two_points_l512_51202


namespace NUMINAMATH_GPT_water_leaked_l512_51201

theorem water_leaked (initial remaining : ℝ) (h_initial : initial = 0.75) (h_remaining : remaining = 0.5) :
  initial - remaining = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_water_leaked_l512_51201


namespace NUMINAMATH_GPT_nails_per_station_correct_l512_51222

variable (total_nails : ℕ) (total_stations : ℕ) (nails_per_station : ℕ)

theorem nails_per_station_correct :
  total_nails = 140 → total_stations = 20 → nails_per_station = total_nails / total_stations → nails_per_station = 7 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_nails_per_station_correct_l512_51222


namespace NUMINAMATH_GPT_sequence_formula_l512_51224

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : a 3 = 7) (h4 : a 4 = 15) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_sequence_formula_l512_51224


namespace NUMINAMATH_GPT_divides_seven_l512_51263

theorem divides_seven (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : Nat.gcd x y = 1) (h5 : x^2 + y^2 = z^4) : 7 ∣ x * y :=
by
  sorry

end NUMINAMATH_GPT_divides_seven_l512_51263


namespace NUMINAMATH_GPT_smallest_positive_integer_is_53_l512_51246

theorem smallest_positive_integer_is_53 :
  ∃ a : ℕ, a > 0 ∧ a % 3 = 2 ∧ a % 4 = 1 ∧ a % 5 = 3 ∧ a = 53 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_is_53_l512_51246


namespace NUMINAMATH_GPT_spheres_do_not_protrude_l512_51207

-- Define the basic parameters
variables (R r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ)
-- Assume conditions
axiom cylinder_height_diameter : h_cylinder = 2 * R
axiom cone_dimensions : h_cone = h_cylinder ∧ h_cone = R

-- The given radius relationship
axiom radius_relation : R = 3 * r

-- Prove the spheres do not protrude from the container
theorem spheres_do_not_protrude (R r h_cylinder h_cone : ℝ)
  (cylinder_height_diameter : h_cylinder = 2 * R)
  (cone_dimensions : h_cone = h_cylinder ∧ h_cone = R)
  (radius_relation : R = 3 * r) : r ≤ R / 2 :=
sorry

end NUMINAMATH_GPT_spheres_do_not_protrude_l512_51207


namespace NUMINAMATH_GPT_hyperbola_equation_chord_length_l512_51239

noncomputable def length_real_axis := 2
noncomputable def eccentricity := Real.sqrt 3
noncomputable def a := 1
noncomputable def b := Real.sqrt 2
noncomputable def hyperbola_eq (x y : ℝ) := x^2 - y^2 / 2 = 1

theorem hyperbola_equation : 
  (∀ x y : ℝ, hyperbola_eq x y ↔ x^2 - (y^2 / 2) = 1) :=
by
  intros x y
  sorry

theorem chord_length (m : ℝ) : 
  ∀ x1 x2 y1 y2 : ℝ, y1 = x1 + m → y2 = x2 + m →
    x1^2 - y1^2 / 2 = 1 → x2^2 - y2^2 / 2 = 1 →
    Real.sqrt (2 * ((x1 + x2)^2 - 4 * x1 * x2)) = 4 * Real.sqrt 2 →
    m = 1 ∨ m = -1 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_hyperbola_equation_chord_length_l512_51239


namespace NUMINAMATH_GPT_range_of_a_l512_51233

noncomputable def satisfies_system (a b c : ℝ) : Prop :=
  (a^2 - b * c - 8 * a + 7 = 0) ∧ (b^2 + c^2 + b * c - 6 * a + 6 = 0)

theorem range_of_a (a b c : ℝ) 
  (h : satisfies_system a b c) : 1 ≤ a ∧ a ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l512_51233


namespace NUMINAMATH_GPT_functional_equation_solution_l512_51226

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (f (xy - x)) + f (x + y) = y * f (x) + f (y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end NUMINAMATH_GPT_functional_equation_solution_l512_51226


namespace NUMINAMATH_GPT_solve_for_a_l512_51247

theorem solve_for_a (a : ℝ) (h_pos : a > 0) 
  (h_roots : ∀ x, x^2 - 2*a*x - 3*a^2 = 0 → (x = -a ∨ x = 3*a)) 
  (h_diff : |(-a) - (3*a)| = 8) : a = 2 := 
sorry

end NUMINAMATH_GPT_solve_for_a_l512_51247


namespace NUMINAMATH_GPT_range_p_l512_51261

open Set

def p (x : ℝ) : ℝ :=
  x^4 + 6*x^2 + 9

theorem range_p : range p = Ici 9 := by
  sorry

end NUMINAMATH_GPT_range_p_l512_51261


namespace NUMINAMATH_GPT_least_positive_integer_l512_51266

theorem least_positive_integer (n : ℕ) :
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ↔
  n = 83 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l512_51266


namespace NUMINAMATH_GPT_gcf_4370_13824_l512_51292

/-- Define the two numbers 4370 and 13824 -/
def num1 := 4370
def num2 := 13824

/-- The statement that the GCF of num1 and num2 is 1 -/
theorem gcf_4370_13824 : Nat.gcd num1 num2 = 1 := by
  sorry

end NUMINAMATH_GPT_gcf_4370_13824_l512_51292


namespace NUMINAMATH_GPT_part_a_part_b_l512_51238

theorem part_a (a : ℤ) (k : ℤ) (h : a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

theorem part_b (a b : ℤ) (m n : ℤ) (h1 : 2 + a = 11 * m) (h2 : 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l512_51238


namespace NUMINAMATH_GPT_infinite_nested_radical_solution_l512_51278

theorem infinite_nested_radical_solution (x : ℝ) (h : x = Real.sqrt (4 + 3 * x)) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_infinite_nested_radical_solution_l512_51278


namespace NUMINAMATH_GPT_parallelogram_base_length_l512_51297

theorem parallelogram_base_length 
  (area : ℝ)
  (b h : ℝ)
  (h_area : area = 128)
  (h_altitude : h = 2 * b) 
  (h_area_eq : area = b * h) : 
  b = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l512_51297


namespace NUMINAMATH_GPT_smallest_M_inequality_l512_51209

theorem smallest_M_inequality :
  ∃ M : ℝ, 
  M = 9 / (16 * Real.sqrt 2) ∧
  ∀ a b c : ℝ, 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ M * (a^2 + b^2 + c^2)^2 :=
by
  use 9 / (16 * Real.sqrt 2)
  sorry

end NUMINAMATH_GPT_smallest_M_inequality_l512_51209


namespace NUMINAMATH_GPT_school_year_days_l512_51250

theorem school_year_days :
  ∀ (D : ℕ),
  (9 = 5 * D / 100) →
  D = 180 := by
  intro D
  sorry

end NUMINAMATH_GPT_school_year_days_l512_51250


namespace NUMINAMATH_GPT_sum_of_coefficients_l512_51241

theorem sum_of_coefficients (a b c d e x : ℝ) (h : 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) :
  a + b + c + d + e = 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l512_51241
