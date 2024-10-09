import Mathlib

namespace div_by_six_l1543_154374

theorem div_by_six (n : ℕ) : 6 ∣ (17^n - 11^n) :=
by
  sorry

end div_by_six_l1543_154374


namespace bert_fraction_spent_l1543_154330

theorem bert_fraction_spent (f : ℝ) :
  let initial := 52
  let after_hardware := initial - initial * f
  let after_cleaners := after_hardware - 9
  let after_grocery := after_cleaners / 2
  let final := 15
  after_grocery = final → f = 1/4 :=
by
  intros h
  sorry

end bert_fraction_spent_l1543_154330


namespace f_2023_eq_1375_l1543_154368

-- Define the function f and the conditions
noncomputable def f : ℕ → ℕ := sorry

axiom f_ff_eq (n : ℕ) (h : n > 0) : f (f n) = 3 * n
axiom f_3n2_eq (n : ℕ) (h : n > 0) : f (3 * n + 2) = 3 * n + 1

-- Prove the specific value for f(2023)
theorem f_2023_eq_1375 : f 2023 = 1375 := sorry

end f_2023_eq_1375_l1543_154368


namespace potato_cost_l1543_154305

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l1543_154305


namespace bacon_vs_tomatoes_l1543_154325

theorem bacon_vs_tomatoes :
  let (n_b : ℕ) := 337
  let (n_t : ℕ) := 23
  n_b - n_t = 314 := by
  let n_b := 337
  let n_t := 23
  have h1 : n_b = 337 := rfl
  have h2 : n_t = 23 := rfl
  sorry

end bacon_vs_tomatoes_l1543_154325


namespace range_of_a_l1543_154336

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 4) : -3 ≤ a ∧ a ≤ 5 := 
sorry

end range_of_a_l1543_154336


namespace correct_percentage_fruits_in_good_condition_l1543_154399

noncomputable def percentage_fruits_in_good_condition
    (total_oranges : ℕ)
    (total_bananas : ℕ)
    (rotten_percentage_oranges : ℝ)
    (rotten_percentage_bananas : ℝ) : ℝ :=
let rotten_oranges := (rotten_percentage_oranges / 100) * total_oranges
let rotten_bananas := (rotten_percentage_bananas / 100) * total_bananas
let good_condition_oranges := total_oranges - rotten_oranges
let good_condition_bananas := total_bananas - rotten_bananas
let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
let total_fruits := total_oranges + total_bananas
(total_fruits_in_good_condition / total_fruits) * 100

theorem correct_percentage_fruits_in_good_condition :
  percentage_fruits_in_good_condition 600 400 15 4 = 89.4 := by
  sorry

end correct_percentage_fruits_in_good_condition_l1543_154399


namespace hyperbola_line_intersections_l1543_154363

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Conditions for intersecting the hyperbola at two points
def intersect_two_points (k : ℝ) : Prop := 
  k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∨ 
  k ∈ Set.Ioo (-1) 1 ∨ 
  k ∈ Set.Ioo 1 (2 * Real.sqrt 3 / 3)

-- Conditions for intersecting the hyperbola at exactly one point
def intersect_one_point (k : ℝ) : Prop := 
  k = 1 ∨ 
  k = -1 ∨ 
  k = 2 * Real.sqrt 3 / 3 ∨ 
  k = -2 * Real.sqrt 3 / 3

-- Proof that k is in the appropriate ranges
theorem hyperbola_line_intersections (k : ℝ) :
  ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ hyperbola x₁ y₁ ∧ line x₁ y₁ k ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ k) 
  → intersect_two_points k))
  ∧ ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x y : ℝ, (hyperbola x y ∧ line x y k ∧ (∀ x' y', hyperbola x' y' ∧ line x' y' k → (x' ≠ x ∨ y' ≠ y) = false)) 
  → intersect_one_point k)) := 
sorry

end hyperbola_line_intersections_l1543_154363


namespace algebra_expression_value_l1543_154364

theorem algebra_expression_value (x y : ℝ) (h : x = 2 * y + 1) : x^2 - 4 * x * y + 4 * y^2 = 1 := 
by 
  sorry

end algebra_expression_value_l1543_154364


namespace num_pairs_of_regular_polygons_l1543_154365

def num_pairs : Nat := 
  let pairs := [(7, 42), (6, 18), (5, 10), (4, 6)]
  pairs.length

theorem num_pairs_of_regular_polygons : num_pairs = 4 := 
  sorry

end num_pairs_of_regular_polygons_l1543_154365


namespace union_of_sets_eq_l1543_154398

variable (M N : Set ℕ)

theorem union_of_sets_eq (h1 : M = {1, 2}) (h2 : N = {2, 3}) : M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_eq_l1543_154398


namespace remainder_of_sum_div_18_l1543_154381

theorem remainder_of_sum_div_18 :
  let nums := [11065, 11067, 11069, 11071, 11073, 11075, 11077, 11079, 11081]
  let residues := [1, 3, 5, 7, 9, 11, 13, 15, 17]
  (nums.sum % 18) = 9 := by
    sorry

end remainder_of_sum_div_18_l1543_154381


namespace certain_number_unique_l1543_154338

-- Define the necessary conditions and statement
def is_certain_number (n : ℕ) : Prop :=
  (∃ k : ℕ, 25 * k = n) ∧ (∃ k : ℕ, 35 * k = n) ∧ 
  (n > 0) ∧ (∃ a b c : ℕ, 1 ≤ a * n ∧ a * n ≤ 1050 ∧ 1 ≤ b * n ∧ b * n ≤ 1050 ∧ 1 ≤ c * n ∧ c * n ≤ 1050 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem certain_number_unique :
  ∃ n : ℕ, is_certain_number n ∧ n = 350 :=
by 
  sorry

end certain_number_unique_l1543_154338


namespace boys_count_l1543_154321

-- Define the number of girls
def girls : ℕ := 635

-- Define the number of boys as being 510 more than the number of girls
def boys : ℕ := girls + 510

-- Prove that the number of boys in the school is 1145
theorem boys_count : boys = 1145 := by
  sorry

end boys_count_l1543_154321


namespace inequality_proof_l1543_154375

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_condition : a * b + b * c + c * d + d * a = 1) :
    (a ^ 3 / (b + c + d)) + (b ^ 3 / (c + d + a)) + (c ^ 3 / (a + b + d)) + (d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l1543_154375


namespace total_food_per_day_l1543_154332

def num_dogs : ℝ := 2
def food_per_dog_per_day : ℝ := 0.12

theorem total_food_per_day : (num_dogs * food_per_dog_per_day) = 0.24 :=
by sorry

end total_food_per_day_l1543_154332


namespace sides_of_rectangle_EKMR_l1543_154395

noncomputable def right_triangle_ACB (AC AB : ℕ) : Prop :=
AC = 3 ∧ AB = 4

noncomputable def rectangle_EKMR_area (area : ℚ) : Prop :=
area = 3/5

noncomputable def rectangle_EKMR_perimeter (x y : ℚ) : Prop :=
2 * (x + y) < 9

theorem sides_of_rectangle_EKMR (x y : ℚ) 
  (h_triangle : right_triangle_ACB 3 4)
  (h_area : rectangle_EKMR_area (3/5))
  (h_perimeter : rectangle_EKMR_perimeter x y) : 
  (x = 2 ∧ y = 3/10) ∨ (x = 3/10 ∧ y = 2) := 
sorry

end sides_of_rectangle_EKMR_l1543_154395


namespace simplify_and_evaluate_l1543_154393

theorem simplify_and_evaluate
  (a b : ℝ)
  (h : |a - 1| + (b + 2)^2 = 0) :
  ((2 * a + b)^2 - (2 * a + b) * (2 * a - b)) / (-1 / 2 * b) = 0 := 
sorry

end simplify_and_evaluate_l1543_154393


namespace john_lift_total_weight_l1543_154353

-- Define the conditions as constants
def initial_weight : ℝ := 135
def weight_increase : ℝ := 265
def bracer_factor : ℝ := 6

-- Define a theorem to prove the total weight John can lift
theorem john_lift_total_weight : initial_weight + weight_increase + (initial_weight + weight_increase) * bracer_factor = 2800 := by
  -- proof here
  sorry

end john_lift_total_weight_l1543_154353


namespace tangent_line_to_curve_determines_m_l1543_154360

theorem tangent_line_to_curve_determines_m :
  ∃ m : ℝ, (∀ x : ℝ, y = x ^ 4 + m * x) ∧ (2 * -1 + y' + 3 = 0) ∧ (y' = -2) → (m = 2) :=
by
  sorry

end tangent_line_to_curve_determines_m_l1543_154360


namespace nancy_picked_l1543_154320

variable (total_picked : ℕ) (alyssa_picked : ℕ)

-- Assuming the conditions given in the problem
def conditions := total_picked = 59 ∧ alyssa_picked = 42

-- Proving that Nancy picked 17 pears
theorem nancy_picked : conditions total_picked alyssa_picked → total_picked - alyssa_picked = 17 := by
  sorry

end nancy_picked_l1543_154320


namespace cost_price_of_article_l1543_154397

theorem cost_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) 
    (h1 : SP = 100) 
    (h2 : profit_percent = 0.20) 
    (h3 : SP = CP * (1 + profit_percent)) : 
    CP = 83.33 :=
by
  sorry

end cost_price_of_article_l1543_154397


namespace smallest_term_abs_l1543_154339

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem smallest_term_abs {a : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 > 0)
  (hS12 : (12 / 2) * (2 * a 1 + 11 * (a 2 - a 1)) > 0)
  (hS13 : (13 / 2) * (2 * a 1 + 12 * (a 2 - a 1)) < 0) :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 13 → n ≠ 7 → abs (a 6) > abs (a 1 + 6 * (a 2 - a 1)) :=
sorry

end smallest_term_abs_l1543_154339


namespace continuous_stripe_probability_l1543_154322

-- Definitions based on conditions from a)
def total_possible_combinations : ℕ := 4^6

def favorable_outcomes : ℕ := 12

def probability_of_continuous_stripe : ℚ := favorable_outcomes / total_possible_combinations

-- The theorem equivalent to prove the given problem
theorem continuous_stripe_probability :
  probability_of_continuous_stripe = 3 / 1024 :=
by
  sorry

end continuous_stripe_probability_l1543_154322


namespace ordered_pairs_count_l1543_154311

theorem ordered_pairs_count : ∃ (count : ℕ), count = 4 ∧
  ∀ (m n : ℕ), m > 0 → n > 0 → m ≥ n → m^2 - n^2 = 144 → (∃ (i : ℕ), i < count) := by
  sorry

end ordered_pairs_count_l1543_154311


namespace ratio_of_discretionary_income_l1543_154352

theorem ratio_of_discretionary_income 
  (salary : ℝ) (D : ℝ)
  (h_salary : salary = 3500)
  (h_discretionary : 0.15 * D = 105) :
  D / salary = 1 / 5 :=
by
  sorry

end ratio_of_discretionary_income_l1543_154352


namespace cans_per_person_day1_l1543_154394

theorem cans_per_person_day1
  (initial_cans : ℕ)
  (people_day1 : ℕ)
  (restock_day1 : ℕ)
  (people_day2 : ℕ)
  (cans_per_person_day2 : ℕ)
  (total_cans_given_away : ℕ) :
  initial_cans = 2000 →
  people_day1 = 500 →
  restock_day1 = 1500 →
  people_day2 = 1000 →
  cans_per_person_day2 = 2 →
  total_cans_given_away = 2500 →
  (total_cans_given_away - (people_day2 * cans_per_person_day2)) / people_day1 = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- condition trivially holds
  sorry

end cans_per_person_day1_l1543_154394


namespace geometric_sequence_ratio_l1543_154317

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 + a 8 = 15) 
  (h2 : a 3 * a 7 = 36) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  (a 19 / a 13 = 4) ∨ (a 19 / a 13 = 1 / 4) :=
by
  sorry

end geometric_sequence_ratio_l1543_154317


namespace compute_sqrt_fraction_l1543_154324

theorem compute_sqrt_fraction :
  (Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35))) = (256 / Real.sqrt 2049) :=
sorry

end compute_sqrt_fraction_l1543_154324


namespace basket_ratio_l1543_154331

variable (S A H : ℕ)

theorem basket_ratio 
  (alex_baskets : A = 8) 
  (hector_baskets : H = 2 * S) 
  (total_baskets : A + S + H = 80) : 
  (S : ℚ) / (A : ℚ) = 3 := 
by 
  sorry

end basket_ratio_l1543_154331


namespace remainder_of_sum_l1543_154334

theorem remainder_of_sum (k j : ℤ) (a b : ℤ) (h₁ : a = 60 * k + 53) (h₂ : b = 45 * j + 17) : ((a + b) % 15) = 5 :=
by
  sorry

end remainder_of_sum_l1543_154334


namespace find_center_of_tangent_circle_l1543_154380

theorem find_center_of_tangent_circle :
  ∃ (a b : ℝ), (abs a = 5) ∧ (abs b = 5) ∧ (4 * a - 3 * b + 10 = 25) ∧ (a = -5) ∧ (b = 5) :=
by {
  -- Here we would provide the proof in Lean, but for now, we state the theorem
  -- and leave the proof as an exercise.
  sorry
}

end find_center_of_tangent_circle_l1543_154380


namespace evaluate_expression_l1543_154300

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) :
  a^2 * b^3 * c = 5 / 256 :=
by
  rw [ha, hb, hc]
  norm_num

end evaluate_expression_l1543_154300


namespace Randy_trip_distance_l1543_154327

noncomputable def total_distance (x : ℝ) :=
  (x / 4) + 40 + 10 + (x / 6)

theorem Randy_trip_distance (x : ℝ) (h : total_distance x = x) : x = 600 / 7 :=
by
  sorry

end Randy_trip_distance_l1543_154327


namespace sum_divisible_by_4_l1543_154341

theorem sum_divisible_by_4 (a b c d x : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9) : 4 ∣ (a + b + c + d) :=
by
  sorry

end sum_divisible_by_4_l1543_154341


namespace matrix_not_invertible_l1543_154389

def is_not_invertible_matrix (y : ℝ) : Prop :=
  let a := 2 + y
  let b := 9
  let c := 4 - y
  let d := 10
  a * d - b * c = 0

theorem matrix_not_invertible (y : ℝ) : is_not_invertible_matrix y ↔ y = 16 / 19 :=
  sorry

end matrix_not_invertible_l1543_154389


namespace vector_relation_AD_l1543_154357

variables {P V : Type} [AddCommGroup V] [Module ℝ V]
variables (A B C D : P) (AB AC AD BC BD CD : V)
variables (hBC_CD : BC = 3 • CD)

theorem vector_relation_AD (h1 : BC = 3 • CD)
                           (h2 : AD = AB + BD)
                           (h3 : BD = BC + CD)
                           (h4 : BC = -AB + AC) :
  AD = - (1 / 3 : ℝ) • AB + (4 / 3 : ℝ) • AC :=
by
  sorry

end vector_relation_AD_l1543_154357


namespace least_number_of_stamps_l1543_154387

theorem least_number_of_stamps (s t : ℕ) (h : 5 * s + 7 * t = 50) : s + t = 8 :=
sorry

end least_number_of_stamps_l1543_154387


namespace jill_average_number_of_stickers_l1543_154312

def average_stickers (packs : List ℕ) : ℚ :=
  (packs.sum : ℚ) / packs.length

theorem jill_average_number_of_stickers :
  average_stickers [5, 7, 9, 9, 11, 15, 15, 17, 19, 21] = 12.8 :=
by
  sorry

end jill_average_number_of_stickers_l1543_154312


namespace common_ratio_is_2_l1543_154304

noncomputable def common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : ℝ :=
(a1 + 2 * d) / a1

theorem common_ratio_is_2 (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : 
    common_ratio a1 d h1 h2 = 2 :=
by
  -- Proof would go here
  sorry

end common_ratio_is_2_l1543_154304


namespace odd_power_of_7_plus_1_divisible_by_8_l1543_154384

theorem odd_power_of_7_plus_1_divisible_by_8 (n : ℕ) (h : n % 2 = 1) : (7 ^ n + 1) % 8 = 0 :=
by
  sorry

end odd_power_of_7_plus_1_divisible_by_8_l1543_154384


namespace maximum_even_integers_of_odd_product_l1543_154376

theorem maximum_even_integers_of_odd_product (a b c d e f g : ℕ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) (h5: e > 0) (h6: f > 0) (h7: g > 0) (hprod : a * b * c * d * e * f * g % 2 = 1): 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) ∧ (g % 2 = 1) :=
sorry

end maximum_even_integers_of_odd_product_l1543_154376


namespace find_k_l1543_154310

noncomputable def a : ℚ := sorry -- Represents positive rational number a
noncomputable def b : ℚ := sorry -- Represents positive rational number b

def minimal_period (x : ℚ) : ℕ := sorry -- Function to determine minimal period of a rational number

-- Conditions as definitions
axiom h1 : minimal_period a = 30
axiom h2 : minimal_period b = 30
axiom h3 : minimal_period (a - b) = 15

-- Statement to prove smallest natural number k such that minimal period of (a + k * b) is 15
theorem find_k : ∃ k : ℕ, minimal_period (a + k * b) = 15 ∧ ∀ n < k, minimal_period (a + n * b) ≠ 15 :=
sorry

end find_k_l1543_154310


namespace total_wages_of_12_men_l1543_154306

variable {M W B x y : Nat}
variable {total_wages : Nat}

-- Condition 1: 12 men do the work equivalent to W women
axiom work_equivalent_1 : 12 * M = W

-- Condition 2: 12 men do the work equivalent to 20 boys
axiom work_equivalent_2 : 12 * M = 20 * B

-- Condition 3: All together earn Rs. 450
axiom total_earnings : (12 * M) + (x * (12 * M / W)) + (y * (12 * M / (20 * B))) = 450

-- The theorem to prove
theorem total_wages_of_12_men : total_wages = 12 * M → false :=
by sorry

end total_wages_of_12_men_l1543_154306


namespace kurt_less_marbles_than_dennis_l1543_154328

theorem kurt_less_marbles_than_dennis
  (Laurie_marbles : ℕ)
  (Kurt_marbles : ℕ)
  (Dennis_marbles : ℕ)
  (h1 : Laurie_marbles = 37)
  (h2 : Laurie_marbles = Kurt_marbles + 12)
  (h3 : Dennis_marbles = 70) :
  Dennis_marbles - Kurt_marbles = 45 := by
  sorry

end kurt_less_marbles_than_dennis_l1543_154328


namespace bob_should_give_l1543_154351

theorem bob_should_give (alice_paid bob_paid charlie_paid : ℕ)
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_charlie : charlie_paid = 180) :
  bob_paid - (120 + 150 + 180) / 3 = 0 := 
by
  sorry

end bob_should_give_l1543_154351


namespace A_left_after_3_days_l1543_154309

def work_done_by_A_and_B_together (x : ℕ) : ℚ :=
  (1 / 21) * x + (1 / 28) * x

def work_done_by_B_alone (days : ℕ) : ℚ :=
  (1 / 28) * days

def total_work_done (x days_b_alone : ℕ) : ℚ :=
  work_done_by_A_and_B_together x + work_done_by_B_alone days_b_alone

theorem A_left_after_3_days :
  ∀ (x : ℕ), total_work_done x 21 = 1 ↔ x = 3 := by
  sorry

end A_left_after_3_days_l1543_154309


namespace total_people_museum_l1543_154386

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end total_people_museum_l1543_154386


namespace original_treadmill_price_l1543_154350

-- Given conditions in Lean definitions
def discount_rate : ℝ := 0.30
def plate_cost : ℝ := 50
def num_plates : ℕ := 2
def total_paid : ℝ := 1045

noncomputable def treadmill_price :=
  let plate_total := num_plates * plate_cost
  let treadmill_discount := (1 - discount_rate)
  (total_paid - plate_total) / treadmill_discount

theorem original_treadmill_price :
  treadmill_price = 1350 := by
  sorry

end original_treadmill_price_l1543_154350


namespace variance_of_data_set_l1543_154373

def data_set : List ℤ := [ -2, -1, 0, 3, 5 ]

def mean (l : List ℤ) : ℚ :=
  (l.sum / l.length)

def variance (l : List ℤ) : ℚ :=
  (1 / l.length) * (l.map (λ x => (x - mean l : ℚ)^2)).sum

theorem variance_of_data_set : variance data_set = 34 / 5 := by
  sorry

end variance_of_data_set_l1543_154373


namespace number_of_people_l1543_154382

-- Definitions based on the conditions
def total_cookies : ℕ := 420
def cookies_per_person : ℕ := 30

-- The goal is to prove the number of people is 14
theorem number_of_people : total_cookies / cookies_per_person = 14 :=
by
  sorry

end number_of_people_l1543_154382


namespace markup_rate_l1543_154316

variable (S : ℝ) (C : ℝ)
variable (profit_percent : ℝ := 0.12) (expense_percent : ℝ := 0.18)
variable (selling_price : ℝ := 8.00)

theorem markup_rate (h1 : C + profit_percent * S + expense_percent * S = S)
                    (h2 : S = selling_price) :
  ((S - C) / C) * 100 = 42.86 := by
  sorry

end markup_rate_l1543_154316


namespace volume_of_regular_quadrilateral_pyramid_l1543_154345

noncomputable def volume_of_pyramid (a : ℝ) : ℝ :=
  let x := 1 -- A placeholder to outline the structure
  let PM := (6 * a) / 5
  let V := (2 * a^3) / 5
  V

theorem volume_of_regular_quadrilateral_pyramid
  (a PM : ℝ)
  (h1 : PM = (6 * a) / 5)
  [InstReal : Nonempty (Real)] :
  volume_of_pyramid a = (2 * a^3) / 5 :=
by
  sorry

end volume_of_regular_quadrilateral_pyramid_l1543_154345


namespace third_speed_correct_l1543_154354

variable (total_time : ℝ := 11)
variable (total_distance : ℝ := 900)
variable (speed1_km_hr : ℝ := 3)
variable (speed2_km_hr : ℝ := 9)

noncomputable def convert_speed_km_hr_to_m_min (speed: ℝ) : ℝ := speed * 1000 / 60

noncomputable def equal_distance : ℝ := total_distance / 3

noncomputable def third_speed_m_min : ℝ :=
  let speed1_m_min := convert_speed_km_hr_to_m_min speed1_km_hr
  let speed2_m_min := convert_speed_km_hr_to_m_min speed2_km_hr
  let d := equal_distance
  300 / (total_time - (d / speed1_m_min + d / speed2_m_min))

noncomputable def third_speed_km_hr : ℝ := third_speed_m_min * 60 / 1000

theorem third_speed_correct : third_speed_km_hr = 6 := by
  sorry

end third_speed_correct_l1543_154354


namespace smallest_consecutive_even_sum_140_l1543_154391

theorem smallest_consecutive_even_sum_140 :
  ∃ (x : ℕ), (x % 2 = 0) ∧ (x + (x + 2) + (x + 4) + (x + 6) = 140) ∧ (x = 32) :=
by
  sorry

end smallest_consecutive_even_sum_140_l1543_154391


namespace num_divisible_by_7_200_to_400_l1543_154361

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l1543_154361


namespace taeyeon_height_proof_l1543_154347

noncomputable def seonghee_height : ℝ := 134.5
noncomputable def taeyeon_height : ℝ := seonghee_height * 1.06

theorem taeyeon_height_proof : taeyeon_height = 142.57 := 
by
  sorry

end taeyeon_height_proof_l1543_154347


namespace x_is_4286_percent_less_than_y_l1543_154385

theorem x_is_4286_percent_less_than_y (x y : ℝ) (h : y = 1.75 * x) : 
  ((y - x) / y) * 100 = 42.86 :=
by
  sorry

end x_is_4286_percent_less_than_y_l1543_154385


namespace simple_interest_correct_l1543_154383

-- Define the principal amount P
variables {P : ℝ}

-- Define the rate of interest r which is 3% or 0.03 in decimal form
def r : ℝ := 0.03

-- Define the time period t which is 2 years
def t : ℕ := 2

-- Define the compound interest CI for 2 years which is $609
def CI : ℝ := 609

-- Define the simple interest SI that we need to find
def SI : ℝ := 600

-- Define a formula for compound interest
def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

-- Define a formula for simple interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem simple_interest_correct (hCI : compound_interest P r t = CI) : simple_interest P r t = SI :=
by
  sorry

end simple_interest_correct_l1543_154383


namespace sum_of_angles_l1543_154301

theorem sum_of_angles (x u v : ℝ) (h1 : u = Real.sin x) (h2 : v = Real.cos x)
  (h3 : 0 ≤ x ∧ x ≤ 2 * Real.pi) 
  (h4 : Real.sin x ^ 4 - Real.cos x ^ 4 = (u - v) / (u * v)) 
  : x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 → (Real.pi / 4 + 5 * Real.pi / 4) = 3 * Real.pi / 2 := 
by
  intro h
  sorry

end sum_of_angles_l1543_154301


namespace roots_in_interval_l1543_154318

theorem roots_in_interval (P : Polynomial ℝ) (h : ∀ i, P.coeff i = 1 ∨ P.coeff i = 0 ∨ P.coeff i = -1) : 
  ∀ x : ℝ, P.eval x = 0 → -2 ≤ x ∧ x ≤ 2 :=
by {
  -- Proof omitted
  sorry
}

end roots_in_interval_l1543_154318


namespace kristy_initial_cookies_l1543_154314

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end kristy_initial_cookies_l1543_154314


namespace power_is_seventeen_l1543_154349

theorem power_is_seventeen (x : ℕ) : (1000^7 : ℝ) / (10^x) = (10000 : ℝ) ↔ x = 17 := by
  sorry

end power_is_seventeen_l1543_154349


namespace number_of_occupied_cars_l1543_154315

theorem number_of_occupied_cars (k : ℕ) (x y : ℕ) :
  18 * k / 9 = 2 * k → 
  3 * x + 2 * y = 12 → 
  x + y ≤ 18 → 
  18 - x - y = 13 :=
by sorry

end number_of_occupied_cars_l1543_154315


namespace joan_already_put_in_cups_l1543_154379

def recipe_cups : ℕ := 7
def cups_needed : ℕ := 4

theorem joan_already_put_in_cups : (recipe_cups - cups_needed = 3) :=
by
  sorry

end joan_already_put_in_cups_l1543_154379


namespace ribbon_cost_l1543_154342

variable (c_g c_m s : ℝ)

theorem ribbon_cost (h1 : 5 * c_g + s = 295) (h2 : 7 * c_m + s = 295) (h3 : 2 * c_m + c_g = 102) : s = 85 :=
sorry

end ribbon_cost_l1543_154342


namespace find_mean_l1543_154348

noncomputable def mean_of_normal_distribution (σ : ℝ) (value : ℝ) (std_devs : ℝ) : ℝ :=
value + std_devs * σ

theorem find_mean
  (σ : ℝ := 1.5)
  (value : ℝ := 12)
  (std_devs : ℝ := 2)
  (h : value = mean_of_normal_distribution σ (value - std_devs * σ) std_devs) :
  mean_of_normal_distribution σ value std_devs = 15 :=
sorry

end find_mean_l1543_154348


namespace inequality_implies_bounds_l1543_154390

open Real

theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, (exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → (0 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_implies_bounds_l1543_154390


namespace total_whales_observed_l1543_154319

-- Define the conditions
def trip1_male_whales : ℕ := 28
def trip1_female_whales : ℕ := 2 * trip1_male_whales
def trip1_total_whales : ℕ := trip1_male_whales + trip1_female_whales

def baby_whales_trip2 : ℕ := 8
def adult_whales_trip2 : ℕ := 2 * baby_whales_trip2
def trip2_total_whales : ℕ := baby_whales_trip2 + adult_whales_trip2

def trip3_male_whales : ℕ := trip1_male_whales / 2
def trip3_female_whales : ℕ := trip1_female_whales
def trip3_total_whales : ℕ := trip3_male_whales + trip3_female_whales

-- Prove the total number of whales observed
theorem total_whales_observed : trip1_total_whales + trip2_total_whales + trip3_total_whales = 178 := by
  -- Assuming all intermediate steps are correct
  sorry

end total_whales_observed_l1543_154319


namespace total_value_of_item_l1543_154346

theorem total_value_of_item (V : ℝ) 
  (h1 : ∃ V > 1000, 
              0.07 * (V - 1000) + 
              (if 55 > 50 then (55 - 50) * 0.15 else 0) + 
              0.05 * V = 112.70) :
  V = 1524.58 :=
by 
  sorry

end total_value_of_item_l1543_154346


namespace inequality_positive_integers_l1543_154372

theorem inequality_positive_integers (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 :=
sorry

end inequality_positive_integers_l1543_154372


namespace negation_of_proposition_l1543_154362

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℤ, 2 * x_0 + x_0 + 1 ≤ 0) ↔ ∀ x : ℤ, 2 * x + x + 1 > 0 :=
by sorry

end negation_of_proposition_l1543_154362


namespace ratio_of_part_to_whole_l1543_154392

theorem ratio_of_part_to_whole (N : ℝ) :
  (2/15) * N = 14 ∧ 0.40 * N = 168 → (14 / ((1/3) * (2/5) * N)) = 1 :=
by
  -- We assume the conditions given in the problem and need to prove the ratio
  intro h
  obtain ⟨h1, h2⟩ := h
  -- Establish equality through calculations
  sorry

end ratio_of_part_to_whole_l1543_154392


namespace sequence_sum_l1543_154358

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

noncomputable def S : ℕ → ℝ
| 0     => 0
| (n+1) => S n + a (n+1)

theorem sequence_sum : S 2017 = 1008 :=
by
  sorry

end sequence_sum_l1543_154358


namespace frank_fence_l1543_154355

theorem frank_fence (L W F : ℝ) (hL : L = 40) (hA : 320 = L * W) : F = 2 * W + L → F = 56 := by
  sorry

end frank_fence_l1543_154355


namespace part1_part2_l1543_154333

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem part1 (x : ℝ) : (∀ x, f x 2 ≤ x + 4 → (1 / 2 ≤ x ∧ x ≤ 7 / 2)) :=
by sorry

theorem part2 (x : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -5 ∨ a ≥ 3) :=
by sorry

end part1_part2_l1543_154333


namespace geometric_sequence_S5_eq_11_l1543_154326

theorem geometric_sequence_S5_eq_11 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (q : ℤ)
  (h1 : a 1 = 1)
  (h4 : a 4 = -8)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_S : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 5 = 11 := 
by
  -- Proof omitted
  sorry

end geometric_sequence_S5_eq_11_l1543_154326


namespace total_money_in_wallet_l1543_154329

-- Definitions of conditions
def initial_five_dollar_bills := 7
def initial_ten_dollar_bills := 1
def initial_twenty_dollar_bills := 3
def initial_fifty_dollar_bills := 1
def initial_one_dollar_coins := 8

def spent_groceries := 65
def paid_fifty_dollar_bill := 1
def paid_twenty_dollar_bill := 1
def received_five_dollar_bill_change := 1
def received_one_dollar_coin_change := 5

def received_twenty_dollar_bills_from_friend := 2
def received_one_dollar_bills_from_friend := 2

-- Proving total amount of money
theorem total_money_in_wallet : 
  initial_five_dollar_bills * 5 + 
  initial_ten_dollar_bills * 10 + 
  initial_twenty_dollar_bills * 20 + 
  initial_fifty_dollar_bills * 50 + 
  initial_one_dollar_coins * 1 - 
  spent_groceries + 
  received_five_dollar_bill_change * 5 + 
  received_one_dollar_coin_change * 1 + 
  received_twenty_dollar_bills_from_friend * 20 + 
  received_one_dollar_bills_from_friend * 1 
  = 150 := 
by
  -- This is where the proof would be located
  sorry

end total_money_in_wallet_l1543_154329


namespace simplest_form_of_expression_l1543_154377

theorem simplest_form_of_expression (c : ℝ) : ((3 * c + 5 - 3 * c) / 2) = 5 / 2 :=
by 
  sorry

end simplest_form_of_expression_l1543_154377


namespace man_speed_in_still_water_l1543_154371

theorem man_speed_in_still_water :
  ∃ (V_m V_s : ℝ), 
  V_m + V_s = 14 ∧ 
  V_m - V_s = 6 ∧ 
  V_m = 10 :=
by
  sorry

end man_speed_in_still_water_l1543_154371


namespace map_distance_l1543_154340

theorem map_distance
  (s d_m : ℝ) (d_r : ℝ)
  (h1 : s = 0.4)
  (h2 : d_r = 5.3)
  (h3 : d_m = 64) :
  (d_m * d_r / s) = 848 := by
  sorry

end map_distance_l1543_154340


namespace processing_plant_growth_eq_l1543_154323

-- Definition of the conditions given in the problem
def initial_amount : ℝ := 10
def november_amount : ℝ := 13
def growth_rate (x : ℝ) : ℝ := initial_amount * (1 + x)^2

-- Lean theorem statement to prove the equation
theorem processing_plant_growth_eq (x : ℝ) : 
  growth_rate x = november_amount ↔ initial_amount * (1 + x)^2 = 13 := 
by
  sorry

end processing_plant_growth_eq_l1543_154323


namespace division_result_l1543_154335

theorem division_result :
  3486 / 189 = 18.444444444444443 := by
  sorry

end division_result_l1543_154335


namespace problem1_problem2_l1543_154303

variable {A B C : ℝ} {AC BC : ℝ}

-- Condition: BC = 2AC
def condition1 (AC BC : ℝ) : Prop := BC = 2 * AC

-- Problem 1: Prove 4cos^2(B) - cos^2(A) = 3
theorem problem1 (h : condition1 AC BC) :
  4 * Real.cos B ^ 2 - Real.cos A ^ 2 = 3 :=
sorry

-- Problem 2: Prove the maximum value of (sin(A) / (2cos(B) + cos(A))) is 2/3 for A ∈ (0, π)
theorem problem2 (h : condition1 AC BC) (hA : 0 < A ∧ A < Real.pi) :
  ∃ t : ℝ, (t = Real.sin A / (2 * Real.cos B + Real.cos A) ∧ t ≤ 2/3) :=
sorry

end problem1_problem2_l1543_154303


namespace line_intersects_ellipse_l1543_154378

theorem line_intersects_ellipse (b : ℝ) : (∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + 1 → ((x^2 / 5) + (y^2 / b) = 1))
  ↔ b ∈ (Set.Ico 1 5 ∪ Set.Ioi 5) := by
sorry

end line_intersects_ellipse_l1543_154378


namespace find_number_of_male_students_l1543_154369

/- Conditions: 
 1. n ≡ 2 [MOD 4]
 2. n ≡ 1 [MOD 5]
 3. n > 15
 4. There are 15 female students
 5. There are more female students than male students
-/
theorem find_number_of_male_students (n : ℕ) (females : ℕ) (h1 : n % 4 = 2) (h2 : n % 5 = 1) (h3 : n > 15) (h4 : females = 15) (h5 : females > n - females) : (n - females) = 11 :=
by
  sorry

end find_number_of_male_students_l1543_154369


namespace complement_of_M_in_U_l1543_154302

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U : U \ M = {2, 3, 5} := by
  sorry

end complement_of_M_in_U_l1543_154302


namespace yellow_balls_in_bag_l1543_154359

theorem yellow_balls_in_bag (x : ℕ) (prob : 1 / (1 + x) = 1 / 4) :
  x = 3 :=
sorry

end yellow_balls_in_bag_l1543_154359


namespace arithmetic_sequence_general_formula_is_not_term_l1543_154396

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) (h17 : a 17 = 66) :
  ∀ n : ℕ, a n = 4 * n - 2 := sorry

theorem is_not_term (a : ℕ → ℤ) 
  (ha : ∀ n : ℕ, a n = 4 * n - 2) :
  ∀ k : ℤ, k = 88 → ¬ ∃ n : ℕ, a n = k := sorry

end arithmetic_sequence_general_formula_is_not_term_l1543_154396


namespace min_a_for_decreasing_f_l1543_154308

theorem min_a_for_decreasing_f {a : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 - a / (2 * Real.sqrt x) ≤ 0) →
  a ≥ 4 :=
sorry

end min_a_for_decreasing_f_l1543_154308


namespace length_of_the_train_is_120_l1543_154356

noncomputable def train_length (time: ℝ) (speed_km_hr: ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time

theorem length_of_the_train_is_120 :
  train_length 3.569962336897346 121 = 120 := by
  sorry

end length_of_the_train_is_120_l1543_154356


namespace coin_probability_l1543_154366

theorem coin_probability (p : ℚ) 
  (P_X_3 : ℚ := 10 * p^3 * (1 - p)^2)
  (P_X_4 : ℚ := 5 * p^4 * (1 - p))
  (P_X_5 : ℚ := p^5)
  (w : ℚ := P_X_3 + P_X_4 + P_X_5) :
  w = 5 / 16 → p = 1 / 4 :=
by
  sorry

end coin_probability_l1543_154366


namespace num_digits_sum_l1543_154313

theorem num_digits_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let num1 := 9643
  let num2 := A * 10 ^ 2 + 7 * 10 + 5
  let num3 := 5 * 10 ^ 2 + B * 10 + 2
  let sum := num1 + num2 + num3
  10^4 ≤ sum ∧ sum < 10^5 :=
by {
  sorry
}

end num_digits_sum_l1543_154313


namespace smallest_integer_proof_l1543_154343

def smallest_integer_condition (n : ℤ) : Prop := n^2 - 15 * n + 56 ≤ 0

theorem smallest_integer_proof :
  ∃ n : ℤ, smallest_integer_condition n ∧ ∀ m : ℤ, smallest_integer_condition m → n ≤ m :=
sorry

end smallest_integer_proof_l1543_154343


namespace rationalize_denominator_l1543_154367

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l1543_154367


namespace color_plane_no_unit_equilateral_same_color_l1543_154370

theorem color_plane_no_unit_equilateral_same_color :
  ∃ (coloring : ℝ × ℝ → ℕ), (∀ (A B C : ℝ × ℝ),
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    (coloring A ≠ coloring B ∨ coloring B ≠ coloring C ∨ coloring C ≠ coloring A)) :=
sorry

end color_plane_no_unit_equilateral_same_color_l1543_154370


namespace h_h_3_eq_2915_l1543_154344

def h (x : ℕ) : ℕ := 3 * x^2 + x + 1

theorem h_h_3_eq_2915 : h (h 3) = 2915 := by
  sorry

end h_h_3_eq_2915_l1543_154344


namespace root_quadratic_sum_product_l1543_154337

theorem root_quadratic_sum_product (x1 x2 : ℝ) (h1 : x1^2 - 2 * x1 - 5 = 0) (h2 : x2^2 - 2 * x2 - 5 = 0) 
  (h3 : x1 ≠ x2) : (x1 + x2 + 3 * (x1 * x2)) = -13 := 
by 
  sorry

end root_quadratic_sum_product_l1543_154337


namespace loan_interest_rate_l1543_154307

theorem loan_interest_rate (P SI T R : ℕ) (h1 : P = 900) (h2 : SI = 729) (h3 : T = R) :
  (SI = (P * R * T) / 100) -> R = 9 :=
by
  sorry

end loan_interest_rate_l1543_154307


namespace six_digit_number_property_l1543_154388

theorem six_digit_number_property :
  ∃ N : ℕ, N = 285714 ∧ (∃ x : ℕ, N = 2 * 10^5 + x ∧ M = 10 * x + 2 ∧ M = 3 * N) :=
by
  sorry

end six_digit_number_property_l1543_154388
