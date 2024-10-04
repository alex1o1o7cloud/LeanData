import Mathlib

namespace factorial_division_identity_l43_43157

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l43_43157


namespace product_xyz_l43_43200

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l43_43200


namespace Buratino_math_problem_l43_43915

theorem Buratino_math_problem (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 :=
by
  intro h
  sorry

end Buratino_math_problem_l43_43915


namespace largest_multiple_of_7_smaller_than_negative_85_l43_43088

theorem largest_multiple_of_7_smaller_than_negative_85 :
  ∃ (n : ℤ), (∃ (k : ℤ), n = 7 * k) ∧ n < -85 ∧ ∀ (m : ℤ), (∃ (k : ℤ), m = 7 * k) ∧ m < -85 → m ≤ n := 
by
  use -91
  split
  { use -13
    norm_num }
  split
  { exact dec_trivial }
  { intros m hm
    cases hm with k hk
    cases hk with hk1 hk2
    have hk3 : k < -12 := by linarith
    have hk4 : k ≤ -13 := int.floor_le $ hk3
    linarith }


end largest_multiple_of_7_smaller_than_negative_85_l43_43088


namespace tan_alpha_expression_value_l43_43726

-- (I) Prove that tan(α) = 4/3 under the given conditions
theorem tan_alpha (O A B C P : ℝ × ℝ) (α : ℝ)
  (hO : O = (0, 0))
  (hA : A = (Real.sin α, 1))
  (hB : B = (Real.cos α, 0))
  (hC : C = (-Real.sin α, 2))
  (hP : P = (2 * Real.cos α - Real.sin α, 1))
  (h_collinear : ∃ t : ℝ, C = t • (P.1, P.2)) :
  Real.tan α = 4 / 3 := sorry

-- (II) Prove the given expression under the condition tan(α) = 4/3
theorem expression_value (α : ℝ)
  (h_tan : Real.tan α = 4 / 3) :
  (Real.sin (2 * α) + Real.sin α) / (2 * Real.cos (2 * α) + 2 * Real.sin α * Real.sin α + Real.cos α) + Real.sin (2 * α) = 
  172 / 75 := sorry

end tan_alpha_expression_value_l43_43726


namespace factorial_div_eq_l43_43145
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l43_43145


namespace relationship_between_x_y_l43_43743

def in_interval (x : ℝ) : Prop := (Real.pi / 4) < x ∧ x < (Real.pi / 2)

noncomputable def x_def (α : ℝ) : ℝ := Real.sin α ^ (Real.log (Real.cos α) / Real.log α)

noncomputable def y_def (α : ℝ) : ℝ := Real.cos α ^ (Real.log (Real.sin α) / Real.log α)

theorem relationship_between_x_y (α : ℝ) (h : in_interval α) : 
  x_def α = y_def α := 
  sorry

end relationship_between_x_y_l43_43743


namespace range_a_range_b_l43_43722

def set_A : Set ℝ := {x | Real.log x / Real.log 2 > 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}
def set_C (b : ℝ) : Set ℝ := {x | b + 1 < x ∧ x < 2 * b + 1}

-- Part (1)
theorem range_a (a : ℝ) : (∀ x, x ∈ set_A → x ∈ set_B a) ↔ a ∈ Set.Iic 4 := sorry

-- Part (2)
theorem range_b (b : ℝ) : (set_A ∪ set_C b = set_A) ↔ b ∈ Set.Iic 0 ∪ Set.Ici 3 := sorry

end range_a_range_b_l43_43722


namespace bacteria_growth_time_l43_43929

theorem bacteria_growth_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (doubling_time : ℕ) :
  initial_bacteria = 1000 →
  final_bacteria = 128000 →
  doubling_time = 3 →
  (∃ t : ℕ, final_bacteria = initial_bacteria * 2 ^ (t / doubling_time) ∧ t = 21) :=
by
  sorry

end bacteria_growth_time_l43_43929


namespace net_profit_is_correct_l43_43267

-- Define the purchase price, markup, and overhead percentage
def purchase_price : ℝ := 48
def markup : ℝ := 55
def overhead_percentage : ℝ := 0.30

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the net profit calculation
def net_profit : ℝ := markup - overhead_cost

-- State the theorem
theorem net_profit_is_correct : net_profit = 40.60 :=
by
  sorry

end net_profit_is_correct_l43_43267


namespace retirement_savings_l43_43245

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end retirement_savings_l43_43245


namespace train_length_equals_750_l43_43416

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l43_43416


namespace max_profit_l43_43509

noncomputable def profit_function (x : ℕ) : ℝ :=
  if x ≤ 400 then
    300 * x - (1 / 2) * x^2 - 20000
  else
    60000 - 100 * x

theorem max_profit : 
  (∀ x ≥ 0, profit_function x ≤ 25000) ∧ (profit_function 300 = 25000) :=
by 
  sorry

end max_profit_l43_43509


namespace blue_books_count_l43_43943

def number_of_blue_books (R B : ℕ) (p : ℚ) : Prop :=
  R = 4 ∧ p = 3/14 → B^2 + 7 * B - 44 = 0

theorem blue_books_count :
  ∃ B : ℕ, number_of_blue_books 4 B (3/14) ∧ B = 4 :=
by
  sorry

end blue_books_count_l43_43943


namespace find_sum_of_x_and_reciprocal_l43_43728

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l43_43728


namespace circle_radius_of_complex_roots_l43_43847

theorem circle_radius_of_complex_roots (z : ℂ) (hz : (z - 1)^3 = 8 * z^3) : 
  ∃ r : ℝ, r = 1 / Real.sqrt 3 :=
by
  sorry

end circle_radius_of_complex_roots_l43_43847


namespace b6_b8_equals_16_l43_43346

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_arithmetic : ∃ d, ∀ n, a_seq (n + 1) = a_seq n + d
axiom b_geometric : ∃ r, ∀ n, b_seq (n + 1) = b_seq n * r
axiom a_nonzero : ∀ n, a_seq n ≠ 0
axiom a_eq : 2 * a_seq 3 - (a_seq 7)^2 + 2 * a_seq 11 = 0
axiom b7_eq_a7 : b_seq 7 = a_seq 7

theorem b6_b8_equals_16 : b_seq 6 * b_seq 8 = 16 := by
  sorry

end b6_b8_equals_16_l43_43346


namespace set_intersection_complement_l43_43572

open Set

noncomputable def A : Set ℝ := { x | abs (x - 1) > 2 }
noncomputable def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
noncomputable def notA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
noncomputable def targetSet : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem set_intersection_complement :
  (notA ∩ B) = targetSet :=
  by
  sorry

end set_intersection_complement_l43_43572


namespace age_of_B_l43_43795

theorem age_of_B (A B C : ℕ) 
  (h1 : (A + B + C) / 3 = 22)
  (h2 : (A + B) / 2 = 18)
  (h3 : (B + C) / 2 = 25) : 
  B = 20 := 
by
  sorry

end age_of_B_l43_43795


namespace John_l43_43910

theorem John's_earnings_on_Saturday :
  ∃ S : ℝ, (S + S / 2 + 20 = 47) ∧ (S = 18) := by
    sorry

end John_l43_43910


namespace problem1_problem2_l43_43136

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l43_43136


namespace average_income_A_B_l43_43073

theorem average_income_A_B (A B C : ℝ)
  (h1 : (B + C) / 2 = 5250)
  (h2 : (A + C) / 2 = 4200)
  (h3 : A = 3000) : (A + B) / 2 = 4050 :=
by
  sorry

end average_income_A_B_l43_43073


namespace part1_part2_l43_43181

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Statement for part 1
theorem part1 (a b : ℝ) (h1 : f a b (-1) = 0) (h2 : f a b 3 = 0) (h3 : a ≠ 0) :
  a = -1 ∧ b = 4 :=
sorry

-- Statement for part 2
theorem part2 (a b : ℝ) (h1 : f a b 1 = 2) (h2 : a + b = 1) (h3 : a > 0) (h4 : b > 0) :
  (∀ x > 0, 1 / a + 4 / b ≥ 9) :=
sorry

end part1_part2_l43_43181


namespace decimal_to_fraction_l43_43489

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43489


namespace train_length_problem_l43_43420

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l43_43420


namespace heximal_to_binary_k_value_l43_43587

theorem heximal_to_binary_k_value (k : ℕ) (h : 10 * (6^3) + k * 6 + 5 = 239) : 
  k = 3 :=
by
  sorry

end heximal_to_binary_k_value_l43_43587


namespace sandra_remaining_money_l43_43402

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end sandra_remaining_money_l43_43402


namespace candy_in_each_bag_l43_43867

theorem candy_in_each_bag (total_candy : ℕ) (bags : ℕ) (h1 : total_candy = 16) (h2 : bags = 2) : total_candy / bags = 8 :=
by {
    sorry
}

end candy_in_each_bag_l43_43867


namespace consignment_shop_total_items_l43_43931

variable (x y z t n : ℕ)

noncomputable def totalItems (n : ℕ) := n + n + n + 3 * n

theorem consignment_shop_total_items :
  ∃ (x y z t n : ℕ), 
    3 * n * y + n * x + n * z + n * t = 240 ∧
    t = 10 * n ∧
    z + x = y + t + 4 ∧
    x + y + 24 = t + z ∧
    y ≤ 6 ∧
    totalItems n = 18 :=
by
  sorry

end consignment_shop_total_items_l43_43931


namespace nancy_total_spending_l43_43112

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l43_43112


namespace find_m_and_n_l43_43024

theorem find_m_and_n (x y m n : ℝ) 
  (h1 : 5 * x - 2 * y = 3) 
  (h2 : m * x + 5 * y = 4) 
  (h3 : x - 4 * y = -3) 
  (h4 : 5 * x + n * y = 1) :
  m = -1 ∧ n = -4 :=
by
  sorry

end find_m_and_n_l43_43024


namespace bryce_received_15_raisins_l43_43740

theorem bryce_received_15_raisins (x : ℕ) (c : ℕ) (h1 : c = x - 10) (h2 : c = x / 3) : x = 15 :=
by
  sorry

end bryce_received_15_raisins_l43_43740


namespace arithmetic_mean_eq_one_l43_43015

theorem arithmetic_mean_eq_one 
  (x a b : ℝ) 
  (hx : x ≠ 0) 
  (hb : b ≠ 0) : 
  (1 / 2 * ((x + a + b) / x + (x - a - b) / x)) = 1 := by
  sorry

end arithmetic_mean_eq_one_l43_43015


namespace retirement_savings_l43_43244

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end retirement_savings_l43_43244


namespace equation_true_when_n_eq_2_l43_43827

theorem equation_true_when_n_eq_2 : (2 ^ (2 / 2)) = 2 :=
by
  sorry

end equation_true_when_n_eq_2_l43_43827


namespace circumcircle_eq_of_triangle_ABC_l43_43025

noncomputable def circumcircle_equation (A B C : ℝ × ℝ) : String := sorry

theorem circumcircle_eq_of_triangle_ABC :
  circumcircle_equation (4, 1) (-6, 3) (3, 0) = "x^2 + y^2 + x - 9y - 12 = 0" :=
sorry

end circumcircle_eq_of_triangle_ABC_l43_43025


namespace part_a_part_c_part_d_l43_43633

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l43_43633


namespace decimal_to_fraction_equivalence_l43_43481

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43481


namespace solve_for_x_l43_43925

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x = 600 - (4 * x + 6 * x) → x = 40 :=
by
  intro h
  sorry

end solve_for_x_l43_43925


namespace circle_standard_form1_circle_standard_form2_l43_43709

theorem circle_standard_form1 (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by
  sorry

theorem circle_standard_form2 (x y : ℝ) :
  4 * x^2 + 4 * y^2 - 8 * x + 4 * y - 11 = 0 ↔ (x - 1)^2 + (y + 1 / 2)^2 = 4 :=
by
  sorry

end circle_standard_form1_circle_standard_form2_l43_43709


namespace range_of_m_l43_43577

theorem range_of_m (m : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) →
  m < -1 / 2 :=
by
  sorry

end range_of_m_l43_43577


namespace tan_C_over_tan_A_max_tan_B_l43_43044

theorem tan_C_over_tan_A {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let tan_A := Real.tan A
  let tan_C := Real.tan C
  (Real.tan C / Real.tan A) = -3 :=
sorry

theorem max_tan_B {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let B := Real.arctan (Real.tan B)
  ∃ (x : ℝ), x = Real.tan B ∧ ∀ y, y = Real.tan B → y ≤ (Real.sqrt 3) / 3 :=
sorry

end tan_C_over_tan_A_max_tan_B_l43_43044


namespace count_triangles_l43_43575

-- Define the problem conditions
def num_small_triangles : ℕ := 11
def num_medium_triangles : ℕ := 4
def num_large_triangles : ℕ := 1

-- Define the main statement asserting the total number of triangles
theorem count_triangles (small : ℕ) (medium : ℕ) (large : ℕ) :
  small = num_small_triangles →
  medium = num_medium_triangles →
  large = num_large_triangles →
  small + medium + large = 16 :=
by
  intros h_small h_medium h_large
  rw [h_small, h_medium, h_large]
  sorry

end count_triangles_l43_43575


namespace hyperbola_condition_l43_43734

theorem hyperbola_condition (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1 → (-1 < k ∧ k < 1)) ∧ 
  ((-1 < k ∧ k < 1) → ∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1) :=
sorry

end hyperbola_condition_l43_43734


namespace total_money_shared_l43_43056

-- Let us define the conditions
def ratio (a b c : ℕ) : Prop := ∃ k : ℕ, (2 * k = a) ∧ (3 * k = b) ∧ (8 * k = c)

def olivia_share := 30

-- Our goal is to prove the total amount of money shared
theorem total_money_shared (a b c : ℕ) (h_ratio : ratio a b c) (h_olivia : a = olivia_share) :
    a + b + c = 195 :=
by
  sorry

end total_money_shared_l43_43056


namespace items_per_charge_is_five_l43_43980

-- Define the number of dog treats, chew toys, rawhide bones, and credit cards as constants.
def num_dog_treats := 8
def num_chew_toys := 2
def num_rawhide_bones := 10
def num_credit_cards := 4

-- Define the total number of items.
def total_items := num_dog_treats + num_chew_toys + num_rawhide_bones

-- Prove that the number of items per credit card charge is 5.
theorem items_per_charge_is_five :
  (total_items / num_credit_cards) = 5 :=
by
  -- Proof goes here (we use sorry to skip the actual proof)
  sorry

end items_per_charge_is_five_l43_43980


namespace track_circumference_is_180_l43_43133

noncomputable def track_circumference : ℕ :=
  let brenda_first_meeting_dist := 120
  let sally_second_meeting_dist := 180
  let brenda_speed_factor : ℕ := 2
  -- circumference of the track
  let circumference := 3 * brenda_first_meeting_dist / brenda_speed_factor
  circumference

theorem track_circumference_is_180 :
  track_circumference = 180 :=
by 
  sorry

end track_circumference_is_180_l43_43133


namespace polygon_interior_exterior_relation_l43_43367

theorem polygon_interior_exterior_relation (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n ≥ 3) :
  n = 8 :=
by
  sorry

end polygon_interior_exterior_relation_l43_43367


namespace least_value_of_x_l43_43752

theorem least_value_of_x 
  (x : ℕ) 
  (p : ℕ) 
  (hx : 0 < x) 
  (hp : Prime p) 
  (h : x = 2 * 11 * p) : x = 44 := 
by
  sorry

end least_value_of_x_l43_43752


namespace complex_identity_l43_43569

open Complex

noncomputable def z := 1 + 2 * I
noncomputable def z_inv := (1 - 2 * I) / 5
noncomputable def z_conj := 1 - 2 * I

theorem complex_identity : 
  (z + z_inv) * z_conj = (22 / 5 : ℂ) - (4 / 5) * I := 
by
  sorry

end complex_identity_l43_43569


namespace find_cookies_per_tray_l43_43170

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end find_cookies_per_tray_l43_43170


namespace petya_time_comparison_l43_43545

open Real

noncomputable def petya_planned_time (D V : ℝ) := D / V

noncomputable def petya_actual_time (D V : ℝ) :=
  let V1 := 1.25 * V
  let V2 := 0.80 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  T1 + T2

theorem petya_time_comparison (D V : ℝ) (hV : V > 0) : 
  petya_actual_time D V > petya_planned_time D V :=
by {
  let T := petya_planned_time D V
  let T_actual := petya_actual_time D V
  have h1 : petya_planned_time D V = D / V, by unfold petya_planned_time
  have h2 : petya_actual_time D V = (D * 41) / (40 * V), by {
      unfold petya_actual_time,
      have h3 : 1.25 * V = 5 * V / 4, by linarith,
      have h4 : 0.80 * V = 4 * V / 5, by linarith,
      rw [h3, h4],
      simp,
      linarith,
  },
  rw h1,
  rw h2,
  have h3 : (41 * D) / (40 * V) > D / V, by linarith,
  exact h3,
}

end petya_time_comparison_l43_43545


namespace new_ratio_of_boarders_to_day_scholars_l43_43079

theorem new_ratio_of_boarders_to_day_scholars
  (B_initial D_initial : ℕ)
  (B_initial_eq : B_initial = 560)
  (ratio_initial : B_initial / D_initial = 7 / 16)
  (new_boarders : ℕ)
  (new_boarders_eq : new_boarders = 80)
  (B_new : ℕ)
  (B_new_eq : B_new = B_initial + new_boarders)
  (D_new : ℕ)
  (D_new_eq : D_new = D_initial) :
  B_new / D_new = 1 / 2 :=
by
  sorry

end new_ratio_of_boarders_to_day_scholars_l43_43079


namespace balloon_permutations_l43_43992

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l43_43992


namespace possible_values_of_m_l43_43063

-- Proposition: for all real values of m, if for all real x, x^2 + 2x + 2 - m >= 0 holds, then m must be one of -1, 0, or 1

theorem possible_values_of_m (m : ℝ) 
  (h : ∀ (x : ℝ), x^2 + 2 * x + 2 - m ≥ 0) : m = -1 ∨ m = 0 ∨ m = 1 :=
sorry

end possible_values_of_m_l43_43063


namespace ratio_docking_to_license_l43_43607

noncomputable def Mitch_savings : ℕ := 20000
noncomputable def boat_cost_per_foot : ℕ := 1500
noncomputable def license_and_registration_fees : ℕ := 500
noncomputable def max_boat_length : ℕ := 12

theorem ratio_docking_to_license :
  let remaining_amount := Mitch_savings - license_and_registration_fees
  let cost_of_longest_boat := boat_cost_per_foot * max_boat_length
  let docking_fees := remaining_amount - cost_of_longest_boat
  docking_fees / license_and_registration_fees = 3 :=
by
  sorry

end ratio_docking_to_license_l43_43607


namespace inequality_a_inequality_c_inequality_d_l43_43644

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l43_43644


namespace roots_expression_eval_l43_43598

theorem roots_expression_eval (p q r : ℝ) 
  (h1 : p + q + r = 2)
  (h2 : p * q + q * r + r * p = -1)
  (h3 : p * q * r = -2)
  (hp : p^3 - 2 * p^2 - p + 2 = 0)
  (hq : q^3 - 2 * q^2 - q + 2 = 0)
  (hr : r^3 - 2 * r^2 - r + 2 = 0) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = 16 :=
sorry

end roots_expression_eval_l43_43598


namespace double_decker_bus_total_capacity_l43_43754

-- Define conditions for the lower floor seating
def lower_floor_left_seats : Nat := 15
def lower_floor_right_seats : Nat := 12
def lower_floor_priority_seats : Nat := 4

-- Each seat on the left and right side of the lower floor holds 2 people
def lower_floor_left_capacity : Nat := lower_floor_left_seats * 2
def lower_floor_right_capacity : Nat := lower_floor_right_seats * 2
def lower_floor_priority_capacity : Nat := lower_floor_priority_seats * 1

-- Define conditions for the upper floor seating
def upper_floor_left_seats : Nat := 20
def upper_floor_right_seats : Nat := 20
def upper_floor_back_capacity : Nat := 15

-- Each seat on the left and right side of the upper floor holds 3 people
def upper_floor_left_capacity : Nat := upper_floor_left_seats * 3
def upper_floor_right_capacity : Nat := upper_floor_right_seats * 3

-- Total capacity of lower and upper floors
def lower_floor_total_capacity : Nat := lower_floor_left_capacity + lower_floor_right_capacity + lower_floor_priority_capacity
def upper_floor_total_capacity : Nat := upper_floor_left_capacity + upper_floor_right_capacity + upper_floor_back_capacity

-- Assert the total capacity
def bus_total_capacity : Nat := lower_floor_total_capacity + upper_floor_total_capacity

-- Prove that the total bus capacity is 193 people
theorem double_decker_bus_total_capacity : bus_total_capacity = 193 := by
  sorry

end double_decker_bus_total_capacity_l43_43754


namespace cone_surface_area_l43_43682

-- Define the surface area formula for a cone with radius r and slant height l
theorem cone_surface_area (r l : ℝ) : 
  let S := π * r^2 + π * r * l
  S = π * r^2 + π * r * l :=
by sorry

end cone_surface_area_l43_43682


namespace nadine_hosing_time_l43_43052

theorem nadine_hosing_time (shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) 
  (h1 : shampoos = 3) (h2 : time_per_shampoo = 15) (h3 : total_cleaning_time = 55) : 
  ∃ t : ℕ, t = total_cleaning_time - shampoos * time_per_shampoo ∧ t = 10 := 
by
  sorry

end nadine_hosing_time_l43_43052


namespace find_m_value_l43_43565

theorem find_m_value
    (x y m : ℝ)
    (hx : x = -1)
    (hy : y = 2)
    (hxy : m * x + 2 * y = 1) :
    m = 3 :=
by
  sorry

end find_m_value_l43_43565


namespace angle_between_bisectors_l43_43613

theorem angle_between_bisectors (β γ : ℝ) (h_sum : β + γ = 130) : (β / 2) + (γ / 2) = 65 :=
by
  have h : β + γ = 130 := h_sum
  sorry

end angle_between_bisectors_l43_43613


namespace kevin_food_expenditure_l43_43781

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l43_43781


namespace find_students_that_got_As_l43_43555

variables (Emily Frank Grace Harry : Prop)

theorem find_students_that_got_As
  (cond1 : Emily → Frank)
  (cond2 : Frank → Grace)
  (cond3 : Grace → Harry)
  (cond4 : Harry → ¬ Emily)
  (three_A_students : ¬ (Emily ∧ Frank ∧ Grace ∧ Harry) ∧
                      (Emily ∧ Frank ∧ Grace ∧ ¬ Harry ∨
                       Emily ∧ Frank ∧ ¬ Grace ∧ Harry ∨
                       Emily ∧ ¬ Frank ∧ Grace ∧ Harry ∨
                       ¬ Emily ∧ Frank ∧ Grace ∧ Harry)) :
  (¬ Emily ∧ Frank ∧ Grace ∧ Harry) :=
by {
  sorry
}

end find_students_that_got_As_l43_43555


namespace instantaneous_acceleration_at_1_second_l43_43733

-- Assume the velocity function v(t) is given as:
def v (t : ℝ) : ℝ := t^2 + 2 * t + 3

-- We need to prove that the instantaneous acceleration at t = 1 second is 4 m/s^2.
theorem instantaneous_acceleration_at_1_second : 
  deriv v 1 = 4 :=
by 
  sorry

end instantaneous_acceleration_at_1_second_l43_43733


namespace factorial_division_identity_l43_43156

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l43_43156


namespace number_of_bags_needed_l43_43956

def cost_corn_seeds : ℕ := 50
def cost_fertilizers_pesticides : ℕ := 35
def cost_labor : ℕ := 15
def profit_percentage : ℝ := 0.10
def price_per_bag : ℝ := 11

theorem number_of_bags_needed (total_cost : ℕ) (total_revenue : ℝ) (num_bags : ℝ) :
  total_cost = cost_corn_seeds + cost_fertilizers_pesticides + cost_labor →
  total_revenue = ↑total_cost + (↑total_cost * profit_percentage) →
  num_bags = total_revenue / price_per_bag →
  num_bags = 10 := 
by
  sorry

end number_of_bags_needed_l43_43956


namespace range_of_a_l43_43586

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + 3 * a * x^2 + 3 * ((a + 2) * x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ deriv (f a) x = 0 ∧ deriv (f a) y = 0) ↔ a < -1 ∨ a > 2 :=
by
  sorry

end range_of_a_l43_43586


namespace man_is_older_l43_43309

-- Define present age of the son
def son_age : ℕ := 26

-- Define present age of the man (father)
axiom man_age : ℕ

-- Condition: in two years, the man's age will be twice the age of his son
axiom age_condition : man_age + 2 = 2 * (son_age + 2)

-- Prove that the man is 28 years older than his son
theorem man_is_older : man_age - son_age = 28 := sorry

end man_is_older_l43_43309


namespace common_ratio_of_geometric_sequence_l43_43563

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 1 + 4 * d = (a 0 + 16 * d) * (a 0 + 4 * d) / a 0 ) :
  (a 1 + 4 * d) / a 0 = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l43_43563


namespace percentage_employees_6_years_or_more_is_26_l43_43372

-- Define the units for different years of service
def units_less_than_2_years : ℕ := 4
def units_2_to_4_years : ℕ := 6
def units_4_to_6_years : ℕ := 7
def units_6_to_8_years : ℕ := 3
def units_8_to_10_years : ℕ := 2
def units_more_than_10_years : ℕ := 1

-- Define the total units
def total_units : ℕ :=
  units_less_than_2_years +
  units_2_to_4_years +
  units_4_to_6_years +
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- Define the units representing employees with 6 years or more of service
def units_6_years_or_more : ℕ :=
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- The goal is to prove that this percentage is 26%
theorem percentage_employees_6_years_or_more_is_26 :
  (units_6_years_or_more * 100) / total_units = 26 := by
  sorry

end percentage_employees_6_years_or_more_is_26_l43_43372


namespace extend_probability_measure_l43_43043

open MeasureTheory

variables {Ω : Type*} {ℱ : measurable_space Ω} {𝒫 : measure Ω} {C : set Ω}

/-- 
Assume ℱ is a measurable space on Ω, 𝒫 is a probability measure on ℱ, and 
C is a subset of Ω that does not belong to ℱ. 
--/
def extend_measure (ℱ : measurable_space Ω) (𝒫 : measure Ω) (C : set Ω) 
  (h𝒞 : ¬ measurable_set C) : Prop :=
  ∃ (𝒫' : measure Ω), 
  𝒫'.to_outer_measure.caratheodory = measurable_space.generate_from (ℱ.measurable_set' ∪ {C}) ∧
  ∀ (E : set Ω), 
  measurable_set E → 
  ℱ.measurable_set' E → 
  𝒫' E = 𝒫 E ∧
  -- Ensure countable additivity for the extended measure 𝒫'
  (∀ (A : ℕ → set Ω), 
  pairwise (disjoint on A) → 
  (∀ n, measurable_set (A n)) → 
  𝒫'.to_outer_measure (⋃ n, A n) = ∑' n, 𝒫'.to_outer_measure (A n))

noncomputable def extend_P {𝒫 : measure Ω} {ℱ : measurable_space Ω} {C : set Ω} 
  (h𝒞 : ¬ measurable_set C) : Prop :=
  extend_measure ℱ 𝒫 C h𝒞

theorem extend_probability_measure 
  (𝒫 : measure Ω) (ℱ : measurable_space Ω) (C : set Ω) 
  (hℱ : ℱ.measurable_set' = ℱ) 
  (h𝒞 : ¬ measurable_set C) : extend_P h𝒞 :=
sorry

end extend_probability_measure_l43_43043


namespace sum_c_d_eq_neg11_l43_43789

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x + 6) / (x^2 + c * x + d)

theorem sum_c_d_eq_neg11 (c d : ℝ) 
    (h₀ : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = 3 ∨ x = -4)) :
    c + d = -11 := 
sorry

end sum_c_d_eq_neg11_l43_43789


namespace problem_solution_l43_43885

def count_multiples_of_5_not_15 : ℕ := 
  let count_up_to (m n : ℕ) := n / m
  let multiples_of_5_up_to_300 := count_up_to 5 299
  let multiples_of_15_up_to_300 := count_up_to 15 299
  multiples_of_5_up_to_300 - multiples_of_15_up_to_300

theorem problem_solution : count_multiples_of_5_not_15 = 40 := by
  sorry

end problem_solution_l43_43885


namespace area_one_magnet_is_150_l43_43660

noncomputable def area_one_magnet : ℕ :=
  let length := 15
  let total_circumference := 70
  let combined_width := (total_circumference / 2 - length) / 2
  let width := combined_width
  length * width

theorem area_one_magnet_is_150 :
  area_one_magnet = 150 :=
by
  -- This will skip the actual proof for now
  sorry

end area_one_magnet_is_150_l43_43660


namespace profit_percentage_l43_43692

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 625) : 
  ((SP - CP) / CP) * 100 = 25 := 
by 
  sorry

end profit_percentage_l43_43692


namespace lottery_most_frequent_number_l43_43675

noncomputable def m (i : ℕ) : ℚ :=
  ((i - 1) * (90 - i) * (89 - i) * (88 - i)) / 6

theorem lottery_most_frequent_number :
  ∀ (i : ℕ), 2 ≤ i ∧ i ≤ 87 → m 23 ≥ m i :=
by 
  sorry -- Proof goes here. This placeholder allows the file to compile.

end lottery_most_frequent_number_l43_43675


namespace problem_solution_l43_43221

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4 * x^2 + 4 * x) / (x^3 + x^2 - 2 * x)

theorem problem_solution :
  let a := 1 in
  let b := 2 in
  let c := 1 in
  let d := 0 in
  a + 2 * b + 3 * c + 4 * d = 8 :=
by {
  let a := 1,
  let b := 2,
  let c := 1,
  let d := 0,
  show a + 2 * b + 3 * c + 4 * d = 8,
  calc
    a + 2 * b + 3 * c + 4 * d
    = 1 + 2 * 2 + 3 * 1 + 4 * 0 : by refl
  ... = 8 : by refl
}

end problem_solution_l43_43221


namespace functional_equation_zero_l43_43233

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (hx : ∀ x y : ℝ, f (x + y) = f x + f y) : f 0 = 0 :=
by
  sorry

end functional_equation_zero_l43_43233


namespace volume_multiplication_factor_l43_43747

variables (r h : ℝ) (π : ℝ := Real.pi)

def original_volume : ℝ := π * r^2 * h
def new_height : ℝ := 3 * h
def new_radius : ℝ := 2.5 * r
def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

theorem volume_multiplication_factor :
  new_volume r h / original_volume r h = 18.75 :=
by
  sorry

end volume_multiplication_factor_l43_43747


namespace total_onions_grown_l43_43773

theorem total_onions_grown :
  let onions_per_day_nancy := 3
  let days_worked_nancy := 4
  let onions_per_day_dan := 4
  let days_worked_dan := 6
  let onions_per_day_mike := 5
  let days_worked_mike := 5
  let onions_per_day_sasha := 6
  let days_worked_sasha := 4
  let onions_per_day_becky := 2
  let days_worked_becky := 6

  let total_onions_nancy := onions_per_day_nancy * days_worked_nancy
  let total_onions_dan := onions_per_day_dan * days_worked_dan
  let total_onions_mike := onions_per_day_mike * days_worked_mike
  let total_onions_sasha := onions_per_day_sasha * days_worked_sasha
  let total_onions_becky := onions_per_day_becky * days_worked_becky

  let total_onions := total_onions_nancy + total_onions_dan + total_onions_mike + total_onions_sasha + total_onions_becky

  total_onions = 97 :=
by
  -- proof goes here
  sorry

end total_onions_grown_l43_43773


namespace annual_interest_rate_l43_43114

theorem annual_interest_rate 
  (P A : ℝ) 
  (hP : P = 136) 
  (hA : A = 150) 
  : (A - P) / P = 0.10 :=
by sorry

end annual_interest_rate_l43_43114


namespace pedestrian_travel_time_l43_43970

noncomputable def travel_time (d : ℝ) (x y : ℝ) : ℝ :=
  d / x

theorem pedestrian_travel_time
  (d : ℝ)
  (x y : ℝ)
  (h1 : d = 1)
  (h2 : 3 * x = 1 - x - y)
  (h3 : (1 / 2) * (x + y) = 1 - x - y)
  : travel_time d x y = 9 := 
sorry

end pedestrian_travel_time_l43_43970


namespace odd_and_increasing_function_l43_43535

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f (x) ≤ f (y)

def function_D (x : ℝ) : ℝ := x * abs x

theorem odd_and_increasing_function : 
  (is_odd function_D) ∧ (is_increasing function_D) :=
sorry

end odd_and_increasing_function_l43_43535


namespace inequality_a_inequality_c_inequality_d_l43_43639

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l43_43639


namespace equation_solution_count_l43_43228

theorem equation_solution_count (n : ℕ) (h_pos : n > 0)
    (h_solutions : ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 28 ∧ ∀ (x y z : ℕ), (x, y, z) ∈ s → 2 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) :
    n = 17 ∨ n = 18 :=
sorry

end equation_solution_count_l43_43228


namespace perpendicular_slope_solution_l43_43872

theorem perpendicular_slope_solution (a : ℝ) :
  (∀ x y : ℝ, ax + (3 - a) * y + 1 = 0) →
  (∀ x y : ℝ, x - 2 * y = 0) →
  (l1_perp_l2 : ∀ x y : ℝ, ax + (3 - a) * y + 1 = 0 → x - 2 * y = 0 → False) →
  a = 2 :=
sorry

end perpendicular_slope_solution_l43_43872


namespace installment_payment_l43_43605

theorem installment_payment
  (cash_price : ℕ)
  (down_payment : ℕ)
  (first_four_months_payment : ℕ)
  (last_four_months_payment : ℕ)
  (installment_additional_cost : ℕ)
  (total_next_four_months_payment : ℕ)
  (H_cash_price : cash_price = 450)
  (H_down_payment : down_payment = 100)
  (H_first_four_months_payment : first_four_months_payment = 4 * 40)
  (H_last_four_months_payment : last_four_months_payment = 4 * 30)
  (H_installment_additional_cost : installment_additional_cost = 70)
  (H_total_next_four_months_payment_correct : 4 * total_next_four_months_payment = 4 * 35) :
  down_payment + first_four_months_payment + 4 * 35 + last_four_months_payment = cash_price + installment_additional_cost := 
by {
  sorry
}

end installment_payment_l43_43605


namespace slipper_cost_l43_43235

def original_price : ℝ := 50.00
def discount_rate : ℝ := 0.10
def embroidery_rate_per_shoe : ℝ := 5.50
def number_of_shoes : ℕ := 2
def shipping_cost : ℝ := 10.00

theorem slipper_cost :
  (original_price - original_price * discount_rate) + 
  (embroidery_rate_per_shoe * number_of_shoes) + 
  shipping_cost = 66.00 :=
by sorry

end slipper_cost_l43_43235


namespace problem1_problem2_l43_43135

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l43_43135


namespace yogurt_cost_l43_43274

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l43_43274


namespace max_value_A_l43_43501

noncomputable def A (x y : ℝ) : ℝ :=
  ((x^2 - y) * Real.sqrt (y + x^3 - x * y) + (y^2 - x) * Real.sqrt (x + y^3 - x * y) + 1) /
  ((x - y)^2 + 1)

theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  A x y ≤ 1 :=
sorry

end max_value_A_l43_43501


namespace compute_c_plus_d_l43_43045

variable {c d : ℝ}

-- Define the given polynomial equations
def poly_c (c : ℝ) := c^3 - 21*c^2 + 28*c - 70
def poly_d (d : ℝ) := 10*d^3 - 75*d^2 - 350*d + 3225

theorem compute_c_plus_d (hc : poly_c c = 0) (hd : poly_d d = 0) : c + d = 21 / 2 := sorry

end compute_c_plus_d_l43_43045


namespace decimal_to_fraction_l43_43473

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43473


namespace servings_needed_l43_43270

theorem servings_needed
  (pieces_per_serving : ℕ)
  (jared_consumption : ℕ)
  (three_friends_consumption : ℕ)
  (another_three_friends_consumption : ℕ)
  (last_four_friends_consumption : ℕ) : 
  pieces_per_serving = 60 →
  jared_consumption = 150 →
  three_friends_consumption = 3 * 80 →
  another_three_friends_consumption = 3 * 200 →
  last_four_friends_consumption = 4 * 100 →
  ∃ (s : ℕ), s = 24 :=
by
  intros
  sorry

end servings_needed_l43_43270


namespace equation_one_equation_two_l43_43618

-- Equation (1): Show that for the equation ⟦ ∀ x, (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3 ↔ x = 1 / 5) ⟧
theorem equation_one (x : ℝ) : (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x = 1 / 5) :=
sorry

-- Equation (2): Show that for the equation ⟦ ∀ x, ((4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false) ⟧
theorem equation_two (x : ℝ) : (4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false :=
sorry

end equation_one_equation_two_l43_43618


namespace total_number_of_cottages_is_100_l43_43854

noncomputable def total_cottages
    (x : ℕ) (n : ℕ) 
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25) 
    (h4 : x + 2 * x + n * x ≥ 70) : ℕ :=
x + 2 * x + n * x

theorem total_number_of_cottages_is_100 
    (x n : ℕ)
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25)
    (h4 : x + 2 * x + n * x ≥ 70)
    (h5 : ∃ m : ℕ, m = (x + 2 * x + n * x)) :
  total_cottages x n h1 h2 h3 h4 = 100 :=
by
  sorry

end total_number_of_cottages_is_100_l43_43854


namespace probability_of_six_being_largest_l43_43506

noncomputable def probability_six_is_largest : ℚ := sorry

theorem probability_of_six_being_largest (cards : Finset ℕ) (selected_cards : Finset ℕ) :
  cards = {1, 2, 3, 4, 5, 6, 7} →
  selected_cards ⊆ cards →
  selected_cards.card = 4 →
  (probability_six_is_largest = 2 / 7) := sorry

end probability_of_six_being_largest_l43_43506


namespace exists_x0_for_which_f_lt_g_l43_43769

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin x ^ 2017
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2017 + 2017 ^ x

theorem exists_x0_for_which_f_lt_g :
  ∃ x0 : ℝ, ∀ x : ℝ, x > x0 → f x < g x :=
sorry

end exists_x0_for_which_f_lt_g_l43_43769


namespace rohit_distance_from_start_l43_43829

noncomputable def rohit_final_position : ℕ × ℕ :=
  let start := (0, 0)
  let p1 := (start.1, start.2 - 25)       -- Moves 25 meters south.
  let p2 := (p1.1 + 20, p1.2)           -- Turns left (east) and moves 20 meters.
  let p3 := (p2.1, p2.2 + 25)           -- Turns left (north) and moves 25 meters.
  let result := (p3.1 + 15, p3.2)       -- Turns right (east) and moves 15 meters.
  result

theorem rohit_distance_from_start :
  rohit_final_position = (35, 0) :=
sorry

end rohit_distance_from_start_l43_43829


namespace sum_of_digits_l43_43951

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 7) :
  (∀ n m : ℕ, sum_of_digits (a ^ 2010 * b ^ 2012 * c) = 13) :=
by
  sorry

end sum_of_digits_l43_43951


namespace balloon_arrangements_correct_l43_43987

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l43_43987


namespace circumference_of_circle_l43_43704

theorem circumference_of_circle (R : ℝ) : 
  (C = 2 * Real.pi * R) :=
sorry

end circumference_of_circle_l43_43704


namespace find_n_l43_43701

theorem find_n (a1 a2 : ℕ) (s2 s1 : ℕ) (n : ℕ) :
    a1 = 12 →
    a2 = 3 →
    s2 = 3 * s1 →
    ∃ n : ℕ, a1 / (1 - a2/a1) = 16 ∧
             a1 / (1 - (a2 + n) / a1) = s2 →
             n = 6 :=
by
  intros
  sorry

end find_n_l43_43701


namespace part_a_part_c_part_d_l43_43630

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l43_43630


namespace first_player_win_condition_l43_43945

def player_one_wins (p q : ℕ) : Prop :=
  p % 5 = 0 ∨ p % 5 = 1 ∨ p % 5 = 4 ∨
  q % 5 = 0 ∨ q % 5 = 1 ∨ q % 5 = 4

theorem first_player_win_condition (p q : ℕ) :
  player_one_wins p q ↔
  (∃ (a b : ℕ), (a, b) = (p, q) ∧ (a % 5 = 0 ∨ a % 5 = 1 ∨ a % 5 = 4 ∨ 
                                     b % 5 = 0 ∨ b % 5 = 1 ∨ b % 5 = 4)) :=
sorry

end first_player_win_condition_l43_43945


namespace trig_cos_sum_l43_43103

open Real

theorem trig_cos_sum :
  cos (37 * (π / 180)) * cos (23 * (π / 180)) - sin (37 * (π / 180)) * sin (23 * (π / 180)) = 1 / 2 :=
by
  sorry

end trig_cos_sum_l43_43103


namespace total_sum_of_intersections_of_five_lines_l43_43218

theorem total_sum_of_intersections_of_five_lines : 
  let possible_intersections : List ℕ := [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
  (possible_intersections.sum = 53) :=
by
  sorry

end total_sum_of_intersections_of_five_lines_l43_43218


namespace identify_letter_X_l43_43057

-- Define the conditions
def date_behind_D (z : ℕ) : ℕ := z
def date_behind_E (z : ℕ) : ℕ := z + 1
def date_behind_F (z : ℕ) : ℕ := z + 14

-- Define the sum condition
def sum_date_E_F (z : ℕ) : ℕ := date_behind_E z + date_behind_F z

-- Define the target date behind another letter
def target_date_behind_another_letter (z : ℕ) : ℕ := z + 15

-- Theorem statement
theorem identify_letter_X (z : ℕ) :
  ∃ (x : Char), sum_date_E_F z = date_behind_D z + target_date_behind_another_letter z → x = 'X' :=
by
  -- The actual proof would go here; we'll defer it for now
  sorry

end identify_letter_X_l43_43057


namespace probability_route_X_is_8_over_11_l43_43900

-- Definitions for the graph paths and probabilities
def routes_from_A_to_B (X Y : Nat) : Nat := 2 + 6 + 3

def routes_passing_through_X (X Y : Nat) : Nat := 2 + 6

def probability_passing_through_X (total_routes passing_routes : Nat) : Rat :=
  (passing_routes : Rat) / (total_routes : Rat)

theorem probability_route_X_is_8_over_11 :
  let total_routes := routes_from_A_to_B 2 3
  let passing_routes := routes_passing_through_X 2 3
  probability_passing_through_X total_routes passing_routes = 8 / 11 :=
by
  -- Assumes correct route calculations from the conditions and aims to prove the probability value
  sorry

end probability_route_X_is_8_over_11_l43_43900


namespace twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l43_43813

theorem twenty_five_percent_less_than_80_is_twenty_five_percent_more_of (n : ℝ) (h : 1.25 * n = 80 - 0.25 * 80) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l43_43813


namespace bananas_more_than_pears_l43_43930

theorem bananas_more_than_pears (A P B : ℕ) (h1 : P = A + 2) (h2 : A + P + B = 19) (h3 : B = 9) : B - P = 3 :=
  by
  sorry

end bananas_more_than_pears_l43_43930


namespace regular_tetrahedron_l43_43312

-- Define the types for points and tetrahedrons
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Tetrahedron :=
(A B C D : Point)
(insphere : Point)

-- Conditions
def sphere_touches_at_angle_bisectors (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_altitudes (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_medians (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

-- Main theorem statement
theorem regular_tetrahedron (T : Tetrahedron)
  (h1 : sphere_touches_at_angle_bisectors T)
  (h2 : sphere_touches_at_altitudes T)
  (h3 : sphere_touches_at_medians T) :
  T.A = T.B ∧ T.A = T.C ∧ T.A = T.D := 
sorry

end regular_tetrahedron_l43_43312


namespace find_g_inverse_75_l43_43889

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 - 6

theorem find_g_inverse_75 : g⁻¹ 75 = 3 := sorry

end find_g_inverse_75_l43_43889


namespace red_shoes_drawn_l43_43944

-- Define the main conditions
def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4
def probability_red : ℝ := 0.21428571428571427

-- Problem statement in Lean
theorem red_shoes_drawn (x : ℕ) (hx : ↑x / total_shoes = probability_red) : x = 2 := by
  sorry

end red_shoes_drawn_l43_43944


namespace common_rational_root_l43_43799

theorem common_rational_root (a b c d e f g : ℚ) (p : ℚ) :
  (48 * p^4 + a * p^3 + b * p^2 + c * p + 16 = 0) ∧
  (16 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 48 = 0) ∧
  (∃ m n : ℤ, p = m / n ∧ Int.gcd m n = 1 ∧ n ≠ 1 ∧ p < 0 ∧ n > 0) →
  p = -1/2 :=
by
  sorry

end common_rational_root_l43_43799


namespace valid_three_digit_numbers_count_l43_43574

noncomputable def count_valid_numbers : ℕ :=
  let valid_first_digits := [2, 4, 6, 8].length
  let valid_other_digits := [0, 2, 4, 6, 8].length
  let total_even_digit_3_digit_numbers := valid_first_digits * valid_other_digits * valid_other_digits
  let no_4_or_8_first_digits := [2, 6].length
  let no_4_or_8_other_digits := [0, 2, 6].length
  let numbers_without_4_or_8 := no_4_or_8_first_digits * no_4_or_8_other_digits * no_4_or_8_other_digits
  let numbers_with_4_or_8 := total_even_digit_3_digit_numbers - numbers_without_4_or_8
  let valid_even_sum_count := 50  -- Assumed from the manual checking
  valid_even_sum_count

theorem valid_three_digit_numbers_count :
  count_valid_numbers = 50 :=
by
  sorry

end valid_three_digit_numbers_count_l43_43574


namespace additional_charge_l43_43965

variable (charge_first : ℝ) -- The charge for the first 1/5 of a mile
variable (total_charge : ℝ) -- Total charge for an 8-mile ride
variable (distance : ℝ) -- Total distance of the ride

theorem additional_charge 
  (h1 : charge_first = 3.50) 
  (h2 : total_charge = 19.1) 
  (h3 : distance = 8) :
  ∃ x : ℝ, x = 0.40 :=
  sorry

end additional_charge_l43_43965


namespace decimal_to_fraction_l43_43438

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43438


namespace cost_per_kg_paint_l43_43932

-- Define the basic parameters
variables {sqft_per_kg : ℝ} -- the area covered by 1 kg of paint
variables {total_cost : ℝ} -- the total cost to paint the cube
variables {side_length : ℝ} -- the side length of the cube
variables {num_faces : ℕ} -- the number of faces of the cube

-- Define the conditions given in the problem
def conditions (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) : Prop :=
  sqft_per_kg = 16 ∧
  total_cost = 876 ∧
  side_length = 8 ∧
  num_faces = 6

-- Define the statement to prove, which is the cost per kg of paint
theorem cost_per_kg_paint (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) :
  conditions sqft_per_kg total_cost side_length num_faces →
  ∃ cost_per_kg : ℝ, cost_per_kg = 36.5 :=
by
  sorry

end cost_per_kg_paint_l43_43932


namespace two_point_three_five_as_fraction_l43_43455

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43455


namespace possible_value_m_l43_43366

theorem possible_value_m (x m : ℝ) (h : ∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) : m ≥ -25 / 8 := sorry

end possible_value_m_l43_43366


namespace largest_multiple_of_7_smaller_than_neg_85_l43_43092

theorem largest_multiple_of_7_smaller_than_neg_85 :
  ∃ k : ℤ, 7 * k < -85 ∧ (∀ m : ℤ, 7 * m < -85 → 7 * m ≤ 7 * k) ∧ 7 * k = -91 :=
by
  simp only [exists_prop, and.assoc],
  sorry

end largest_multiple_of_7_smaller_than_neg_85_l43_43092


namespace max_sum_of_cubes_l43_43596

open Real

theorem max_sum_of_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * sqrt 5 :=
by
  sorry

end max_sum_of_cubes_l43_43596


namespace kabadi_kho_kho_players_l43_43685

theorem kabadi_kho_kho_players (total_players kabadi_only kho_kho_only both_games : ℕ)
  (h1 : kabadi_only = 10)
  (h2 : kho_kho_only = 40)
  (h3 : total_players = 50)
  (h4 : kabadi_only + kho_kho_only - both_games = total_players) :
  both_games = 0 := by
  sorry

end kabadi_kho_kho_players_l43_43685


namespace math_problem_l43_43622

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l43_43622


namespace unique_positive_b_for_one_solution_l43_43337

theorem unique_positive_b_for_one_solution
  (a : ℝ) (c : ℝ) :
  a = 3 →
  (∃! (b : ℝ), b > 0 ∧ (3 * (b + (1 / b)))^2 - 4 * c = 0 ) →
  c = 9 :=
by
  intros ha h
  -- Proceed to show that c must be 9
  sorry

end unique_positive_b_for_one_solution_l43_43337


namespace imaginary_unit_root_l43_43748

theorem imaginary_unit_root (a b : ℝ) (h : (Complex.I : ℂ) ^ 2 + a * Complex.I + b = 0) : a + b = 1 := by
  -- Since this is just the statement, we add a sorry to focus on the structure
  sorry

end imaginary_unit_root_l43_43748


namespace green_papayas_left_l43_43269

/-- Define the initial number of green papayas on the tree -/
def initial_green_papayas : ℕ := 14

/-- Define the number of papayas that turned yellow on Friday -/
def friday_yellow_papayas : ℕ := 2

/-- Define the number of papayas that turned yellow on Sunday -/
def sunday_yellow_papayas : ℕ := 2 * friday_yellow_papayas

/-- The remaining number of green papayas after Friday and Sunday -/
def remaining_green_papayas : ℕ := initial_green_papayas - friday_yellow_papayas - sunday_yellow_papayas

theorem green_papayas_left : remaining_green_papayas = 8 := by
  sorry

end green_papayas_left_l43_43269


namespace problem1_problem2_l43_43139

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l43_43139


namespace smallest_of_three_l43_43812

noncomputable def A : ℕ := 38 + 18
noncomputable def B : ℕ := A - 26
noncomputable def C : ℕ := B / 3

theorem smallest_of_three : C < A ∧ C < B := by
  sorry

end smallest_of_three_l43_43812


namespace unique_n_degree_polynomial_exists_l43_43064

theorem unique_n_degree_polynomial_exists (n : ℕ) (h : n > 0) :
  ∃! (f : Polynomial ℝ), Polynomial.degree f = n ∧
    f.eval 0 = 1 ∧
    ∀ x : ℝ, (x + 1) * (f.eval x)^2 - 1 = -((x + 1) * (f.eval (-x))^2 - 1) := 
sorry

end unique_n_degree_polynomial_exists_l43_43064


namespace flour_more_than_sugar_l43_43391

-- Define the conditions.
def sugar_needed : ℕ := 9
def total_flour_needed : ℕ := 14
def salt_needed : ℕ := 40
def flour_already_added : ℕ := 4

-- Define the target proof statement.
theorem flour_more_than_sugar :
  (total_flour_needed - flour_already_added) - sugar_needed = 1 :=
by
  -- sorry is used here to skip the proof.
  sorry

end flour_more_than_sugar_l43_43391


namespace r_at_6_l43_43047

-- Define the monic quintic polynomial r(x) with given conditions
def r (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + 2 

-- Given conditions
axiom r_1 : r 1 = 3
axiom r_2 : r 2 = 7
axiom r_3 : r 3 = 13
axiom r_4 : r 4 = 21
axiom r_5 : r 5 = 31

-- Proof goal
theorem r_at_6 : r 6 = 158 :=
by
  sorry

end r_at_6_l43_43047


namespace distance_symmetric_parabola_l43_43677

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def parabola (x : ℝ) : ℝ := 3 - x^2

theorem distance_symmetric_parabola (A B : ℝ × ℝ) 
  (hA : A.2 = parabola A.1) 
  (hB : B.2 = parabola B.1)
  (hSym : A.1 + A.2 = 0 ∧ B.1 + B.2 = 0) 
  (hDistinct : A ≠ B) :
  distance A B = 3 * sqrt 2 :=
by
  sorry

end distance_symmetric_parabola_l43_43677


namespace female_employees_l43_43896

theorem female_employees (total_employees male_employees : ℕ) 
  (advanced_degree_male_adv: ℝ) (advanced_degree_female_adv: ℝ) (prob: ℝ) 
  (h1 : total_employees = 450) 
  (h2 : male_employees = 300)
  (h3 : advanced_degree_male_adv = 0.10) 
  (h4 : advanced_degree_female_adv = 0.40)
  (h5 : prob = 0.4) : 
  ∃ F : ℕ, 0.10 * male_employees + (advanced_degree_female_adv * F + (1 - advanced_degree_female_adv) * F) / total_employees = prob ∧ F = 150 :=
by
  sorry

end female_employees_l43_43896


namespace find_value_of_a_l43_43881

noncomputable def value_of_a (a : ℝ) (hyp_asymptotes_tangent_circle : Prop) : Prop :=
  a = (Real.sqrt 3) / 3 → hyp_asymptotes_tangent_circle

theorem find_value_of_a (a : ℝ) (condition1 : 0 < a)
  (condition_hyperbola : ∀ x y, x^2 / a^2 - y^2 = 1)
  (condition_circle : ∀ x y, x^2 + y^2 - 4*y + 3 = 0)
  (hyp_asymptotes_tangent_circle : Prop) :
  value_of_a a hyp_asymptotes_tangent_circle := 
sorry

end find_value_of_a_l43_43881


namespace product_xyz_equals_zero_l43_43195

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l43_43195


namespace columns_contain_all_numbers_l43_43430

def rearrange (n m k : ℕ) (a : ℕ → ℕ) : ℕ → ℕ :=
  λ i => if i < n - m then a (i + m + 1)
         else if i < n - k - m then a (i - (n - m) + k + 1)
         else a (i - (n - k))

theorem columns_contain_all_numbers
  (n m k: ℕ)
  (h1 : n > 0)
  (h2 : m < n)
  (h3 : k < n)
  (a : ℕ → ℕ)
  (h4 : ∀ i : ℕ, i < n → a i = i + 1) :
  ∀ j : ℕ, j < n → ∃ i : ℕ, i < n ∧ rearrange n m k a i = j + 1 :=
by
  sorry

end columns_contain_all_numbers_l43_43430


namespace javier_first_throw_l43_43907

theorem javier_first_throw 
  (second third first : ℝ)
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := 
by sorry

end javier_first_throw_l43_43907


namespace reciprocal_sum_l43_43493

theorem reciprocal_sum (a b c d : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 3) (h4 : d = 4) : 
  (a / b + c / d)⁻¹ = (20 : ℚ) / 23 := 
by
  sorry

end reciprocal_sum_l43_43493


namespace digit_B_divisible_by_9_l43_43667

-- Defining the condition for B making 762B divisible by 9
theorem digit_B_divisible_by_9 (B : ℕ) : (15 + B) % 9 = 0 ↔ B = 3 := 
by
  sorry

end digit_B_divisible_by_9_l43_43667


namespace infinite_geometric_series_common_ratio_l43_43850

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end infinite_geometric_series_common_ratio_l43_43850


namespace distance_home_to_school_l43_43291

theorem distance_home_to_school :
  ∃ (D : ℝ) (T : ℝ), 
    3 * (T + 7 / 60) = D ∧
    6 * (T - 8 / 60) = D ∧
    D = 1.5 :=
by
  sorry

end distance_home_to_school_l43_43291


namespace soda_cost_l43_43772

theorem soda_cost (b s : ℕ) 
  (h₁ : 3 * b + 2 * s = 450) 
  (h₂ : 2 * b + 3 * s = 480) : 
  s = 108 := 
by
  sorry

end soda_cost_l43_43772


namespace a_equals_2t_squared_l43_43866

theorem a_equals_2t_squared {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4 * a = b^2) :
  ∃ t : ℕ, 0 < t ∧ a = 2 * t^2 :=
sorry

end a_equals_2t_squared_l43_43866


namespace infinite_series_sum_eq_two_l43_43326

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l43_43326


namespace find_distance_CD_l43_43078

noncomputable def distance_CD : ℝ :=
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (3, 6)
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

theorem find_distance_CD :
  ∀ (C D : ℝ × ℝ), 
  (C = (0, 0) ∧ D = (3, 6)) ∧ 
  (∃ x y : ℝ, (y^2 = 12 * x ∧ (x^2 + y^2 - 4 * x - 6 * y = 0))) → 
  distance_CD = 3 * Real.sqrt 5 :=
by
  sorry

end find_distance_CD_l43_43078


namespace verify_toothpick_count_l43_43662

def toothpick_problem : Prop :=
  let L := 45
  let W := 25
  let Mv := 8
  let Mh := 5
  -- Calculate the total number of vertical toothpicks
  let verticalToothpicks := (L + 1 - Mv) * W
  -- Calculate the total number of horizontal toothpicks
  let horizontalToothpicks := (W + 1 - Mh) * L
  -- Calculate the total number of toothpicks
  let totalToothpicks := verticalToothpicks + horizontalToothpicks
  -- Ensure the total matches the expected result
  totalToothpicks = 1895

theorem verify_toothpick_count : toothpick_problem :=
by
  sorry

end verify_toothpick_count_l43_43662


namespace find_points_l43_43260

noncomputable def f (x y z : ℝ) : ℝ := (x^2 + y^2 + z^2) / (x + y + z)

theorem find_points :
  (∃ (x₀ y₀ z₀ : ℝ), 0 < x₀^2 + y₀^2 + z₀^2 ∧ x₀^2 + y₀^2 + z₀^2 < 1 / 1999 ∧
    1.999 < f x₀ y₀ z₀ ∧ f x₀ y₀ z₀ < 2) :=
  sorry

end find_points_l43_43260


namespace quadratic_roots_l43_43074

theorem quadratic_roots {α p q : ℝ} (hα : 0 < α ∧ α ≤ 1) (hroots : ∃ x : ℝ, x^2 + p * x + q = 0) :
  ∃ x : ℝ, α * x^2 + p * x + q = 0 :=
by sorry

end quadratic_roots_l43_43074


namespace find_a_l43_43365

-- Define the conditions given in the problem
def binomial_term (r : ℕ) (a : ℝ) : ℝ :=
  Nat.choose 7 r * 2^(7-r) * (-a)^r

def coefficient_condition (a : ℝ) : Prop :=
  binomial_term 5 a = 84

-- The theorem stating the problem's solution
theorem find_a (a : ℝ) (h : coefficient_condition a) : a = -1 :=
  sorry

end find_a_l43_43365


namespace geometric_sequence_at_t_l43_43654

theorem geometric_sequence_at_t (a : ℕ → ℕ) (S : ℕ → ℕ) (t : ℕ) :
  (∀ n, a n = a 1 * (3 ^ (n - 1))) →
  a 1 = 1 →
  S t = (a 1 * (1 - 3 ^ t)) / (1 - 3) →
  S t = 364 →
  a t = 243 :=
by {
  sorry
}

end geometric_sequence_at_t_l43_43654


namespace concyclic_if_and_only_if_angle_bac_90_l43_43041

theorem concyclic_if_and_only_if_angle_bac_90
  {A B C D K L : Point}
  (ABC_scalene : ¬(A = B) ∧ ¬(B = C) ∧ ¬(A = C))
  (BC_largest : ∀ X Y Z : Point, distance X Y ≤ distance B C)
  (D_perpendicular : D ∈ line B C ∧ angle A D B = π / 2)
  (K_on_AB : K ∈ line A B)
  (L_on_AC : L ∈ line A C)
  (D_midpoint_KL : distance K D = distance D L) :
  (concyclic B K C L) ↔ (angle A B C = π / 2) :=
by sorry

end concyclic_if_and_only_if_angle_bac_90_l43_43041


namespace correct_system_of_equations_l43_43681

theorem correct_system_of_equations (x y : ℕ) :
  (8 * x - 3 = y ∧ 7 * x + 4 = y) ↔ 
  (8 * x - 3 = y ∧ 7 * x + 4 = y) := 
by 
  sorry

end correct_system_of_equations_l43_43681


namespace mrs_hilt_bakes_loaves_l43_43917

theorem mrs_hilt_bakes_loaves :
  let total_flour := 5
  let flour_per_loaf := 2.5
  (total_flour / flour_per_loaf) = 2 := 
by
  sorry

end mrs_hilt_bakes_loaves_l43_43917


namespace part1_part2_l43_43737

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := abs (x - 2) + 2
def g (m : R) (x : R) : R := m * abs x

theorem part1 (x : R) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem part2 (m : R) : (∀ x : R, f x ≥ g m x) → m ∈ Set.Iic (1 : R) := by
  sorry

end part1_part2_l43_43737


namespace range_of_ab_l43_43026

-- Given two positive numbers a and b such that ab = a + b + 3, we need to prove ab ≥ 9.

theorem range_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = a + b + 3) : 9 ≤ a * b :=
by
  sorry

end range_of_ab_l43_43026


namespace solve_mod_equation_l43_43999

theorem solve_mod_equation (x : ℤ) (h : 10 * x + 3 ≡ 7 [ZMOD 18]) : x ≡ 4 [ZMOD 9] :=
sorry

end solve_mod_equation_l43_43999


namespace yogurt_cost_l43_43272

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l43_43272


namespace regular_polygon_exterior_angle_l43_43536

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end regular_polygon_exterior_angle_l43_43536


namespace sampling_probabilities_equal_l43_43431

variables (total_items first_grade_items second_grade_items equal_grade_items substandard_items : ℕ)
variables (p_1 p_2 p_3 : ℚ)

-- Conditions given in the problem
def conditions := 
  total_items = 160 ∧ 
  first_grade_items = 48 ∧ 
  second_grade_items = 64 ∧ 
  equal_grade_items = 3 ∧ 
  substandard_items = 1 ∧ 
  p_1 = 1 / 8 ∧ 
  p_2 = 1 / 8 ∧ 
  p_3 = 1 / 8

-- The theorem to be proved
theorem sampling_probabilities_equal (h : conditions total_items first_grade_items second_grade_items equal_grade_items substandard_items p_1 p_2 p_3) :
  p_1 = p_2 ∧ p_2 = p_3 :=
sorry

end sampling_probabilities_equal_l43_43431


namespace part1_solution_set_part2_range_of_a_l43_43571

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end part1_solution_set_part2_range_of_a_l43_43571


namespace decimal_to_fraction_l43_43467

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43467


namespace xyz_product_value_l43_43193

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l43_43193


namespace slope_angle_of_line_l43_43558

theorem slope_angle_of_line (x y : ℝ) (θ : ℝ) : (x - y + 3 = 0) → θ = 45 := 
sorry

end slope_angle_of_line_l43_43558


namespace D_time_to_complete_job_l43_43496

-- Let A_rate be the rate at which A works (jobs per hour)
-- Let D_rate be the rate at which D works (jobs per hour)
def A_rate : ℚ := 1 / 3
def combined_rate : ℚ := 1 / 2

-- We need to prove that D_rate, the rate at which D works alone, is 1/6 jobs per hour
def D_rate := 1 / 6

-- And thus, that D can complete the job in 6 hours
theorem D_time_to_complete_job :
  (A_rate + D_rate = combined_rate) → (1 / D_rate) = 6 :=
by
  sorry

end D_time_to_complete_job_l43_43496


namespace green_apples_count_l43_43659

-- Definitions for the conditions
def total_apples : ℕ := 9
def red_apples : ℕ := 7

-- Theorem stating the number of green apples
theorem green_apples_count : total_apples - red_apples = 2 := by
  sorry

end green_apples_count_l43_43659


namespace win_sector_area_l43_43302

-- Defining the conditions
def radius : ℝ := 12
def total_area : ℝ := π * radius^2
def win_probability : ℝ := 1 / 3

-- Theorem to prove the area of the WIN sector
theorem win_sector_area : total_area * win_probability = 48 * π := by
  sorry

end win_sector_area_l43_43302


namespace Victor_Total_Money_l43_43085

-- Definitions for the conditions
def originalAmount : Nat := 10
def allowance : Nat := 8

-- The proof problem statement
theorem Victor_Total_Money : originalAmount + allowance = 18 := by
  sorry

end Victor_Total_Money_l43_43085


namespace print_time_correct_l43_43522

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l43_43522


namespace unique_solution_of_equation_l43_43338

theorem unique_solution_of_equation :
  ∃! (x : Fin 8 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + 
                                  (x 2 - x 3)^2 + (x 3 - x 4)^2 + 
                                  (x 4 - x 5)^2 + (x 5 - x 6)^2 + 
                                  (x 6 - x 7)^2 + (x 7)^2 = 1 / 9 :=
sorry

end unique_solution_of_equation_l43_43338


namespace sqrt_D_always_irrational_l43_43599

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end sqrt_D_always_irrational_l43_43599


namespace correct_statement_c_l43_43100

-- Definitions
variables {Point : Type*} {Line Plane : Type*}
variables (l m : Line) (α β : Plane)

-- Conditions
def parallel_planes (α β : Plane) : Prop := sorry  -- α ∥ β
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊥ α
def line_in_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊂ α
def line_perpendicular (l m : Line) : Prop := sorry  -- l ⊥ m

-- Theorem to be proven
theorem correct_statement_c 
  (α β : Plane) (l : Line)
  (h_parallel : parallel_planes α β)
  (h_perpendicular : perpendicular_line_plane l α) :
  ∀ (m : Line), line_in_plane m β → line_perpendicular m l := 
sorry

end correct_statement_c_l43_43100


namespace seq_a_n_100th_term_l43_43738

theorem seq_a_n_100th_term :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ 
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) ∧ 
  a 100 = -3 := 
sorry

end seq_a_n_100th_term_l43_43738


namespace random_event_proof_l43_43533

def is_certain_event (event: Prop) : Prop := ∃ h: event → true, ∃ h': true → event, true
def is_impossible_event (event: Prop) : Prop := event → false
def is_random_event (event: Prop) : Prop := ¬is_certain_event event ∧ ¬is_impossible_event event

def cond1 : Prop := sorry -- Yingying encounters a green light
def cond2 : Prop := sorry -- A non-transparent bag contains one ping-pong ball and two glass balls of the same size, and a ping-pong ball is drawn from it.
def cond3 : Prop := sorry -- You are currently answering question 12 of this test paper.
def cond4 : Prop := sorry -- The highest temperature in our city tomorrow will be 60°C.

theorem random_event_proof : 
  is_random_event cond1 ∧ 
  ¬is_random_event cond2 ∧ 
  ¬is_random_event cond3 ∧ 
  ¬is_random_event cond4 :=
by
  sorry

end random_event_proof_l43_43533


namespace intersection_point_l43_43040

noncomputable def f (x : ℝ) := (x^2 - 8 * x + 7) / (2 * x - 6)

noncomputable def g (a b c : ℝ) (x : ℝ) := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) :
  (∀ x, 2 * x - 6 = 0 <-> x ≠ 3) →
  ∃ (k : ℝ), (g a b c x = -2 * x - 4 + k / (x - 3)) →
  (f x = g a b c x) ∧ x ≠ -3 → x = 1 ∧ f 1 = 0 :=
by
  intros
  sorry

end intersection_point_l43_43040


namespace two_point_three_five_as_fraction_l43_43456

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43456


namespace bread_slices_leftover_l43_43249

-- Definitions based on conditions provided in the problem
def total_bread_slices : ℕ := 2 * 20
def total_ham_slices : ℕ := 2 * 8
def sandwiches_made : ℕ := total_ham_slices
def bread_slices_needed : ℕ := sandwiches_made * 2

-- Theorem we want to prove
theorem bread_slices_leftover : total_bread_slices - bread_slices_needed = 8 :=
by 
    -- Insert steps of proof here
    sorry

end bread_slices_leftover_l43_43249


namespace print_time_l43_43521

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l43_43521


namespace segment_length_BD_eq_CB_l43_43602

theorem segment_length_BD_eq_CB {AC CB BD x : ℝ}
  (h1 : AC = 4 * CB)
  (h2 : BD = CB)
  (h3 : CB = x) :
  BD = CB := 
by
  -- Proof omitted
  sorry

end segment_length_BD_eq_CB_l43_43602


namespace crafts_club_necklaces_l43_43609

theorem crafts_club_necklaces (members : ℕ) (total_beads : ℕ) (beads_per_necklace : ℕ)
  (h1 : members = 9) (h2 : total_beads = 900) (h3 : beads_per_necklace = 50) :
  (total_beads / beads_per_necklace) / members = 2 :=
by
  sorry

end crafts_club_necklaces_l43_43609


namespace total_students_in_Lansing_l43_43226

theorem total_students_in_Lansing :
  let num_schools_300 := 20
  let num_schools_350 := 30
  let num_schools_400 := 15
  let students_per_school_300 := 300
  let students_per_school_350 := 350
  let students_per_school_400 := 400
  (num_schools_300 * students_per_school_300 + num_schools_350 * students_per_school_350 + num_schools_400 * students_per_school_400 = 22500) := 
  sorry

end total_students_in_Lansing_l43_43226


namespace geometric_sequence_arithmetic_condition_l43_43717

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def positive_terms (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n > 0

def arithmetic_sequence_cond (a_n : ℕ → ℝ) : Prop :=
  a_n 2 - (1 / 2) * a_n 3 = (1 / 2) * a_n 3 - a_n 1

-- Problem: Prove the required ratio equals the given value
theorem geometric_sequence_arithmetic_condition
  (h_geo: is_geometric_sequence a_n q)
  (h_pos: positive_terms a_n)
  (h_arith: arithmetic_sequence_cond a_n)
  (h_q_ne_one: q ≠ 1) :
  (a_n 4 + a_n 5) / (a_n 3 + a_n 4) = (1 + Real.sqrt 5) / 2 :=
sorry

end geometric_sequence_arithmetic_condition_l43_43717


namespace sphere_radius_same_volume_l43_43510

noncomputable def tent_radius : ℝ := 3
noncomputable def tent_height : ℝ := 9

theorem sphere_radius_same_volume : 
  (4 / 3) * Real.pi * ( (20.25)^(1/3) )^3 = (1 / 3) * Real.pi * tent_radius^2 * tent_height :=
by
  sorry

end sphere_radius_same_volume_l43_43510


namespace max_area_circle_between_parallel_lines_l43_43166

theorem max_area_circle_between_parallel_lines : 
  ∀ (l₁ l₂ : ℝ → ℝ → Prop), 
    (∀ x y, l₁ x y ↔ 3*x - 4*y = 0) → 
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 20 = 0) → 
  ∃ A, A = 4 * Real.pi :=
by 
  sorry

end max_area_circle_between_parallel_lines_l43_43166


namespace cost_of_mozzarella_cheese_l43_43123

-- Define the problem conditions as Lean definitions
def blendCostPerKg : ℝ := 696.05
def romanoCostPerKg : ℝ := 887.75
def weightMozzarella : ℝ := 19
def weightRomano : ℝ := 18.999999999999986  -- Practically the same as 19 in context
def totalWeight : ℝ := weightMozzarella + weightRomano

-- Define the expected result for the cost per kilogram of mozzarella cheese
def expectedMozzarellaCostPerKg : ℝ := 504.40

-- Theorem statement to verify the cost of mozzarella cheese
theorem cost_of_mozzarella_cheese :
  weightMozzarella * (expectedMozzarellaCostPerKg : ℝ) + weightRomano * romanoCostPerKg = totalWeight * blendCostPerKg := by
  sorry

end cost_of_mozzarella_cheese_l43_43123


namespace x_intercept_of_line_l43_43864

theorem x_intercept_of_line (x y : ℚ) (h_eq : 4 * x + 7 * y = 28) (h_y : y = 0) : (x, y) = (7, 0) := 
by 
  sorry

end x_intercept_of_line_l43_43864


namespace linear_relationship_selling_price_maximize_profit_l43_43116

theorem linear_relationship (k b : ℝ)
  (h₁ : 36 = 12 * k + b)
  (h₂ : 34 = 13 * k + b) :
  y = -2 * x + 60 :=
by
  sorry

theorem selling_price (p c x : ℝ)
  (h₁ : x ≥ 10)
  (h₂ : x ≤ 19)
  (h₃ : x - 10 = (192 / (y + 10))) :
  x = 18 :=
by
  sorry

theorem maximize_profit (x w : ℝ)
  (h_max : x = 19)
  (h_profit : w = -2 * x^2 + 80 * x - 600) :
  w = 198 :=
by
  sorry

end linear_relationship_selling_price_maximize_profit_l43_43116


namespace isosceles_if_interior_angles_equal_l43_43412

-- Definition for a triangle
structure Triangle :=
  (A B C : Type)

-- Defining isosceles triangle condition
def is_isosceles (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Defining the angle equality condition
def interior_angles_equal (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Main theorem stating the contrapositive
theorem isosceles_if_interior_angles_equal (T : Triangle) : 
  interior_angles_equal T → is_isosceles T :=
by sorry

end isosceles_if_interior_angles_equal_l43_43412


namespace balloon_permutations_l43_43990

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l43_43990


namespace nancy_total_spent_l43_43106

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l43_43106


namespace theta_terminal_side_l43_43363

theorem theta_terminal_side (alpha : ℝ) (theta : ℝ) (h1 : alpha = 1560) (h2 : -360 < theta ∧ theta < 360) :
    (theta = 120 ∨ theta = -240) := by
  -- The proof steps would go here
  sorry

end theta_terminal_side_l43_43363


namespace balloon_permutations_l43_43993

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l43_43993


namespace table_sale_price_percentage_l43_43966

theorem table_sale_price_percentage (W : ℝ) : 
  let S := 1.4 * W
  let P := 0.65 * S
  P = 0.91 * W :=
by
  sorry

end table_sale_price_percentage_l43_43966


namespace decimal_to_fraction_l43_43462

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43462


namespace problem_l43_43564

def p : Prop := 0 % 2 = 0
def q : Prop := ¬(3 % 2 = 0)

theorem problem : p ∨ q :=
by
  sorry

end problem_l43_43564


namespace print_time_l43_43519

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l43_43519


namespace decimal_to_fraction_l43_43463

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43463


namespace coopers_daily_pie_count_l43_43329

-- Definitions of conditions
def total_pies_made_per_day (x : ℕ) : ℕ := x
def days := 12
def pies_eaten_by_ashley := 50
def remaining_pies := 34

-- Lean 4 statement of the problem to prove
theorem coopers_daily_pie_count (x : ℕ) : 
  12 * total_pies_made_per_day x - pies_eaten_by_ashley = remaining_pies → 
  x = 7 := 
by
  intro h
  -- Solution steps (not included in the theorem)
  -- Given proof follows from the Lean 4 statement
  sorry

end coopers_daily_pie_count_l43_43329


namespace range_of_a_l43_43561

variable (a : ℝ)

theorem range_of_a (h : ∀ x : ℤ, 2 * (x:ℝ)^2 - 17 * x + a ≤ 0 →  (x = 3 ∨ x = 4 ∨ x = 5)) : 
  30 < a ∧ a ≤ 33 :=
sorry

end range_of_a_l43_43561


namespace remainder_sum_first_150_l43_43096

-- Definitions based on the conditions
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Lean statement equivalent to the mathematical problem
theorem remainder_sum_first_150 :
  (sum_first_n 150) % 11250 = 75 :=
by 
sorry

end remainder_sum_first_150_l43_43096


namespace martin_correct_answers_l43_43058

theorem martin_correct_answers : 
  ∀ (Campbell_correct Kelsey_correct Martin_correct : ℕ), 
  Campbell_correct = 35 →
  Kelsey_correct = Campbell_correct + 8 →
  Martin_correct = Kelsey_correct - 3 →
  Martin_correct = 40 := 
by
  intros Campbell_correct Kelsey_correct Martin_correct h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  rw [h3]
  rfl

end martin_correct_answers_l43_43058


namespace length_of_train_is_750m_l43_43419

-- Defining the conditions
def train_and_platform_equal_length : Prop := ∀ (L : ℝ), (Length_of_train = L ∧ Length_of_platform = L)
def train_speed := 90 * (1000 / 3600)  -- Convert speed from km/hr to m/s
def crossing_time := 60  -- Time given in seconds

-- Definition for the length of the train
def Length_of_train := sorry -- Given that it should be derived

-- The proof problem statement
theorem length_of_train_is_750m : (train_and_platform_equal_length ∧ train_speed ∧ crossing_time → Length_of_train = 750) :=
by
  -- Proof is skipped
  sorry

end length_of_train_is_750m_l43_43419


namespace decimal_to_fraction_l43_43466

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43466


namespace largest_fraction_of_three_l43_43348

theorem largest_fraction_of_three (a b c : Nat) (h1 : Nat.gcd a 6 = 1)
  (h2 : Nat.gcd b 15 = 1) (h3 : Nat.gcd c 20 = 1)
  (h4 : (a * b * c) = 60) :
  max (a / 6) (max (b / 15) (c / 20)) = 5 / 6 :=
by
  sorry

end largest_fraction_of_three_l43_43348


namespace gcd_7384_12873_l43_43557

theorem gcd_7384_12873 : Int.gcd 7384 12873 = 1 :=
by
  sorry

end gcd_7384_12873_l43_43557


namespace problem_l43_43190

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l43_43190


namespace exists_k_not_divisible_l43_43065

theorem exists_k_not_divisible (a b c n : ℤ) (hn : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) :=
sorry

end exists_k_not_divisible_l43_43065


namespace black_circles_count_l43_43314

theorem black_circles_count (a1 d n : ℕ) (h1 : a1 = 2) (h2 : d = 1) (h3 : n = 16) :
  (n * (a1 + (n - 1) * d) / 2) + n ≤ 160 :=
by
  rw [h1, h2, h3]
  -- Here we will carry out the arithmetic to prove the statement
  sorry

end black_circles_count_l43_43314


namespace tangent_line_at_pi_l43_43330

theorem tangent_line_at_pi :
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.sin x) → 
  ∀ x, x = Real.pi →
  ∀ y, (y = -x + Real.pi) ↔
        (∀ x, y = -x + Real.pi) := 
  sorry

end tangent_line_at_pi_l43_43330


namespace decimal_to_fraction_l43_43474

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43474


namespace count_square_free_integers_l43_43030

def square_free_in_range_2_to_199 : Nat :=
  91

theorem count_square_free_integers :
  ∃ n : Nat, n = 91 ∧
  ∀ m : Nat, 2 ≤ m ∧ m < 200 →
  (∀ k : Nat, k^2 ∣ m → k^2 = 1) :=
by
  -- The proof will be filled here
  sorry

end count_square_free_integers_l43_43030


namespace ratio_PA_AB_l43_43371

theorem ratio_PA_AB (A B C P : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup P]
  (h1 : ∃ AC CB : ℕ, AC = 2 * CB)
  (h2 : ∃ PA AB : ℕ, PA = 4 * (AB / 5)) :
  PA / AB = 4 / 5 := sorry

end ratio_PA_AB_l43_43371


namespace complex_purely_imaginary_l43_43585

theorem complex_purely_imaginary (m : ℝ) :
  (m^2 - 3*m + 2 = 0) ∧ (m^2 - 2*m ≠ 0) → m = 1 :=
by {
  sorry
}

end complex_purely_imaginary_l43_43585


namespace butterflies_equal_distribution_l43_43899

theorem butterflies_equal_distribution (N : ℕ) : (∃ t : ℕ, 
    (N - t) % 8 = 0 ∧ (N - t) / 8 > 0) ↔ ∃ k : ℕ, N = 45 * k :=
by sorry

end butterflies_equal_distribution_l43_43899


namespace decimal_to_fraction_l43_43457

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43457


namespace range_of_f_l43_43815

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : set.range (λ x, if x = -2 then 0 else f x) = { y : ℝ | y ≠ 1 } :=
by
  sorry

end range_of_f_l43_43815


namespace yogurt_cost_l43_43277

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l43_43277


namespace part1_number_of_students_part2_probability_distribution_and_expected_value_part3_probability_at_least_one_prize_l43_43792

-- Part 1: Number of students participating from each class
theorem part1_number_of_students :
  let students_in_class := [30, 40, 20, 10]
  let total_students := sum students_in_class
  let sampling_ratio := 10 / total_students
  let participating_students := map (λ x, x * sampling_ratio) students_in_class
  participating_students = [3, 4, 2, 1] :=
by 
  sorry

-- Part 2: Probability distribution and expected value of X
open probability_theory

theorem part2_probability_distribution_and_expected_value :
  let X := λ (correct_answers : ℕ), if correct_answers ≤ 4 then correct_answers else 0
  let p_X := function.update (function.raise_to_fun ℕ ennreal) 4 0
  p_X 1 = 1 / 30 ∧
  p_X 2 = 3 / 10 ∧
  p_X 3 = 1 / 2 ∧
  p_X 4 = 1 / 6 ∧
  let E_X := ∑ i, i * p_X i
  E_X = 2.8 :=
by
  sorry

-- Part 3: Probability that at least one student from Class 1 will receive a prize
theorem part3_probability_at_least_one_prize :
  let p_correct := 1 / 3
  let p_incorrect := 2 / 3
  let p_prize := ∑ k in finset.range 5, (choose 4 k) * (p_correct ^ k) * (p_incorrect ^ (4 - k))
  let binom_dist := function.update (binomial 3 p_prize) 3 0
  let p_at_least_one := 1 - binom_dist 0
  p_at_least_one = 217 / 729 :=
by
  sorry

end part1_number_of_students_part2_probability_distribution_and_expected_value_part3_probability_at_least_one_prize_l43_43792


namespace price_of_adult_ticket_l43_43532

/--
Given:
1. The price of a child's ticket is half the price of an adult's ticket.
2. Janet buys tickets for 10 people, 4 of whom are children.
3. Janet buys a soda for $5.
4. With the soda, Janet gets a 20% discount on the total admission price.
5. Janet paid $197 in total for everything.

Prove that the price of an adult admission ticket is $30.
-/
theorem price_of_adult_ticket : 
  ∃ (A : ℝ), 
  (∀ (childPrice adultPrice total : ℝ),
    adultPrice = A →
    childPrice = A / 2 →
    total = adultPrice * 6 + childPrice * 4 →
    totalPriceWithDiscount = 192 →
    total / 0.8 = total + 5 →
    A = 30) :=
sorry

end price_of_adult_ticket_l43_43532


namespace sum_of_coordinates_l43_43774

-- Definitions of points and their coordinates
def pointC (x : ℝ) : ℝ × ℝ := (x, 8)
def pointD (x : ℝ) : ℝ × ℝ := (x, -8)

-- The goal is to prove that the sum of the four coordinate values of points C and D is 2x
theorem sum_of_coordinates (x : ℝ) :
  (pointC x).1 + (pointC x).2 + (pointD x).1 + (pointD x).2 = 2 * x :=
by
  sorry

end sum_of_coordinates_l43_43774


namespace barbara_weekly_allowance_l43_43317

theorem barbara_weekly_allowance (W C S : ℕ) (H : W = 100) (A : S = 20) (N : C = 16) :
  (W - S) / C = 5 :=
by
  -- definitions to match conditions
  have W_def : W = 100 := H
  have S_def : S = 20 := A
  have C_def : C = 16 := N
  sorry

end barbara_weekly_allowance_l43_43317


namespace part1_part2_case1_part2_case2_part2_case3_part3_l43_43876

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end part1_part2_case1_part2_case2_part2_case3_part3_l43_43876


namespace binary_addition_l43_43814

theorem binary_addition (a b : ℕ) :
  (a = (2^0 + 2^2 + 2^4 + 2^6)) → (b = (2^0 + 2^3 + 2^6)) →
  (a + b = 158) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end binary_addition_l43_43814


namespace every_integer_as_sum_of_squares_l43_43250

theorem every_integer_as_sum_of_squares (n : ℤ) : ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ n = (x^2 : ℤ) + (y^2 : ℤ) - (z^2 : ℤ) :=
by sorry

end every_integer_as_sum_of_squares_l43_43250


namespace problem_proof_l43_43343

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * x + 2 - x

-- Condition given in the problem
axiom h : ∃ a : ℝ, f a = 3

-- Theorem statement
theorem problem_proof : ∃ a : ℝ, f a = 3 → f (2 * a) = 7 :=
by
  sorry

end problem_proof_l43_43343


namespace count_not_divisible_by_5_or_7_l43_43358

theorem count_not_divisible_by_5_or_7 :
  let n := 1000
  let count_divisible_by (m : ℕ) := Nat.floor (999 / m)
  (999 - count_divisible_by 5 - count_divisible_by 7 + count_divisible_by 35) = 686 :=
by
  sorry

end count_not_divisible_by_5_or_7_l43_43358


namespace part_a_part_c_part_d_l43_43631

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l43_43631


namespace computer_hardware_contract_prob_l43_43266

theorem computer_hardware_contract_prob :
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  ∃ P_H : ℝ, P_at_least_one = P_H + P_S - P_H_and_S ∧ P_H = 0.8 :=
by
  -- Let definitions and initial conditions
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  -- Solve for P(H)
  let P_H := 0.8
  -- Show the proof of the calculation
  sorry

end computer_hardware_contract_prob_l43_43266


namespace Martin_correct_answers_l43_43059

theorem Martin_correct_answers (C K M : ℕ) 
  (h1 : C = 35)
  (h2 : K = C + 8)
  (h3 : M = K - 3) : 
  M = 40 :=
by
  sorry

end Martin_correct_answers_l43_43059


namespace ratio_cost_to_marked_price_l43_43512

theorem ratio_cost_to_marked_price (p : ℝ) (hp : p > 0) :
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 6) * selling_price
  cost_price / p = 5 / 8 :=
by 
  sorry

end ratio_cost_to_marked_price_l43_43512


namespace cos_at_min_distance_l43_43020

noncomputable def cosAtMinimumDistance (t : ℝ) (ht : t < 0) : ℝ :=
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  if distance = Real.sqrt 5 then
    x / distance
  else
    0 -- some default value given the condition distance is not sqrt(5), which is impossible in this context

theorem cos_at_min_distance (t : ℝ) (ht : t < 0) :
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  distance = Real.sqrt 5 → cosAtMinimumDistance t ht = - 2 * Real.sqrt 5 / 5 :=
by
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  sorry

end cos_at_min_distance_l43_43020


namespace tangent_line_at_point_inequality_proof_l43_43880

def f (t x : ℝ) : ℝ := t * x - (t - 1) * real.log x - t

theorem tangent_line_at_point (x : ℝ) (h : x = 1) : 
  ∀ t, t = 2 → (∃ (m b : ℝ), m = 1 ∧ b = -1 ∧ (∀ y, y = f x 1 → (y = x - 1))) :=
by
  sorry

theorem inequality_proof (t x : ℝ) (h1 : t ≤ 0) (h2 : x > 1) : 
  f t x < real.exp (x - 1) - 1 :=
by
  sorry

end tangent_line_at_point_inequality_proof_l43_43880


namespace no_real_solutions_l43_43617

open Real

theorem no_real_solutions :
  ¬(∃ x : ℝ, (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 = 0) := by
  sorry

end no_real_solutions_l43_43617


namespace z_rate_per_rupee_of_x_l43_43528

-- Given conditions as definitions in Lean 4
def x_share := 1 -- x gets Rs. 1 for this proof
def y_rate_per_rupee_of_x := 0.45
def y_share := 27
def total_amount := 105

-- The statement to prove
theorem z_rate_per_rupee_of_x :
  (105 - (1 * 60) - 27) / 60 = 0.30 :=
by
  sorry

end z_rate_per_rupee_of_x_l43_43528


namespace bread_leftover_after_sandwiches_l43_43248

def total_bread_slices (bread_packages: ℕ) (slices_per_package: ℕ) : ℕ :=
  bread_packages * slices_per_package

def total_ham_slices (ham_packages: ℕ) (slices_per_package: ℕ) : ℕ :=
  ham_packages * slices_per_package

def sandwiches_from_ham (ham_slices: ℕ) : ℕ :=
  ham_slices

def total_bread_used (sandwiches: ℕ) (bread_slices_per_sandwich: ℕ) : ℕ :=
  sandwiches * bread_slices_per_sandwich

def bread_leftover (total_bread: ℕ) (bread_used: ℕ) : ℕ :=
  total_bread - bread_used

theorem bread_leftover_after_sandwiches :
  let bread_packages := 2
  let bread_slices_per_package := 20
  let ham_packages := 2
  let ham_slices_per_package := 8
  let bread_slices_per_sandwich := 2 in
  bread_leftover
    (total_bread_slices bread_packages bread_slices_per_package)
    (total_bread_used
      (sandwiches_from_ham (total_ham_slices ham_packages ham_slices_per_package))
      bread_slices_per_sandwich) = 8 :=
by
  sorry

end bread_leftover_after_sandwiches_l43_43248


namespace sum_cotangents_equal_l43_43066

theorem sum_cotangents_equal (a b c S m_a m_b m_c S' : ℝ) (cot_A cot_B cot_C cot_A' cot_B' cot_C' : ℝ)
  (h1 : cot_A + cot_B + cot_C = (a^2 + b^2 + c^2) / (4 * S))
  (h2 : m_a^2 + m_b^2 + m_c^2 = 3 * (a^2 + b^2 + c^2) / 4)
  (h3 : S' = 3 * S / 4)
  (h4 : cot_A' + cot_B' + cot_C' = (m_a^2 + m_b^2 + m_c^2) / (4 * S')) :
  cot_A + cot_B + cot_C = cot_A' + cot_B' + cot_C' :=
by
  -- Proof is needed, but omitted here
  sorry

end sum_cotangents_equal_l43_43066


namespace meadow_total_revenue_correct_l43_43392

-- Define the given quantities and conditions as Lean definitions
def total_diapers : ℕ := 192000
def price_per_diaper : ℝ := 4.0
def bundle_discount : ℝ := 0.05
def purchase_discount : ℝ := 0.05
def tax_rate : ℝ := 0.10

-- Define a function that calculates the revenue from selling all the diapers
def calculate_revenue (total_diapers : ℕ) (price_per_diaper : ℝ) (bundle_discount : ℝ) 
    (purchase_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let gross_revenue := total_diapers * price_per_diaper
  let bundle_discounted_revenue := gross_revenue * (1 - bundle_discount)
  let purchase_discounted_revenue := bundle_discounted_revenue * (1 - purchase_discount)
  let taxed_revenue := purchase_discounted_revenue * (1 + tax_rate)
  taxed_revenue

-- The main theorem to prove that the calculated revenue matches the expected value
theorem meadow_total_revenue_correct : 
  calculate_revenue total_diapers price_per_diaper bundle_discount purchase_discount tax_rate = 762432 := 
by
  sorry

end meadow_total_revenue_correct_l43_43392


namespace inequality_a_inequality_c_inequality_d_l43_43642

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l43_43642


namespace cylinder_height_l43_43283

theorem cylinder_height
  (r : ℝ) (SA : ℝ) (h : ℝ)
  (h_radius : r = 3)
  (h_surface_area_given : SA = 30 * Real.pi)
  (h_surface_area_formula : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  h = 2 :=
by
  -- Proof can be written here
  sorry

end cylinder_height_l43_43283


namespace shooter_scores_l43_43373

theorem shooter_scores
    (x y z : ℕ)
    (hx : x + y + z > 11)
    (hscore: 8 * x + 9 * y + 10 * z = 100) :
    (x + y + z = 12) ∧ ((x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end shooter_scores_l43_43373


namespace inequality_system_solution_l43_43787

theorem inequality_system_solution (x : ℤ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) := sorry

end inequality_system_solution_l43_43787


namespace decimal_to_fraction_l43_43475

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43475


namespace maximum_value_of_a_l43_43009

theorem maximum_value_of_a :
  (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) → a ≤ 6 :=
by
  sorry

end maximum_value_of_a_l43_43009


namespace cos_angle_plus_pi_over_two_l43_43576

theorem cos_angle_plus_pi_over_two (α : ℝ) (h1 : Real.cos α = 1 / 5) (h2 : α ∈ Set.Icc (-2 * Real.pi) (-3 * Real.pi / 2) ∪ Set.Icc (0) (Real.pi / 2)) :
  Real.cos (α + Real.pi / 2) = 2 * Real.sqrt 6 / 5 :=
sorry

end cos_angle_plus_pi_over_two_l43_43576


namespace balloon_permutations_l43_43995

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l43_43995


namespace max_value_2ab_3bc_lemma_l43_43914

noncomputable def max_value_2ab_3bc (a b c : ℝ) : ℝ :=
  2 * a * b + 3 * b * c

theorem max_value_2ab_3bc_lemma
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 = 2) :
  max_value_2ab_3bc a b c ≤ 3 :=
sorry

end max_value_2ab_3bc_lemma_l43_43914


namespace geometric_sequence_sum_l43_43217

-- Let {a_n} be a geometric sequence such that S_2 = 7 and S_6 = 91. Prove that S_4 = 28

-- Define the sum of the first n terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 r : ℝ) : ℝ := a1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a1 r : ℝ) (h1 : S 2 a1 r = 7) (h2 : S 6 a1 r = 91) :
  S 4 a1 r = 28 := 
by 
  sorry

end geometric_sequence_sum_l43_43217


namespace expression_equivalence_l43_43099

theorem expression_equivalence (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) :=
by {
  sorry
}

end expression_equivalence_l43_43099


namespace set_representation_l43_43941

open Nat

def isInPositiveNaturals (x : ℕ) : Prop :=
  x ≠ 0

def isPositiveDivisor (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

theorem set_representation :
  {x | isInPositiveNaturals x ∧ isPositiveDivisor 6 (6 - x)} = {3, 4, 5} :=
by
  sorry

end set_representation_l43_43941


namespace fraction_zero_value_l43_43215

theorem fraction_zero_value (x : ℝ) (h : (3 - x) ≠ 0) : (x+2)/(3-x) = 0 ↔ x = -2 := by
  sorry

end fraction_zero_value_l43_43215


namespace positive_integer_iff_positive_real_l43_43559

theorem positive_integer_iff_positive_real (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℕ, n > 0 ∧ abs ((x - 2 * abs x) * abs x) / x = n) ↔ x > 0 :=
by
  sorry

end positive_integer_iff_positive_real_l43_43559


namespace box_height_l43_43964

theorem box_height (volume length width : ℝ) (h : ℝ) (h_volume : volume = 315) (h_length : length = 7) (h_width : width = 9) :
  h = 5 :=
by
  -- Proof would go here
  sorry

end box_height_l43_43964


namespace negation_of_exists_l43_43668

theorem negation_of_exists {x : ℝ} (h : ∃ x : ℝ, 3^x + x < 0) : ∀ x : ℝ, 3^x + x ≥ 0 :=
sorry

end negation_of_exists_l43_43668


namespace correct_operation_l43_43955

theorem correct_operation :
  (∀ a : ℝ, (a^4)^2 ≠ a^6) ∧
  (∀ a b : ℝ, (a - b)^2 ≠ a^2 - ab + b^2) ∧
  (∀ a b : ℝ, 6 * a^2 * b / (2 * a * b) = 3 * a) ∧
  (∀ a : ℝ, a^2 + a^4 ≠ a^6) :=
by {
  sorry
}

end correct_operation_l43_43955


namespace determine_k_l43_43731

noncomputable def p (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def q (x y : ℝ) : ℝ := Real.log (x - y)

def m (k : ℝ) : ℝ := 2 * k
def w (n : ℝ) : ℝ := n + 1

theorem determine_k (k : ℝ) (c : ℝ → ℝ → ℝ) (v : ℝ → ℝ → ℝ) (n : ℝ) :
  p 32 6 = k * c 32 6 ∧
  p 45 10 = m k * c 45 10 ∧
  q 15 5 = n * v 15 5 ∧
  q 28 7 = w n * v 28 7 →
  k = 1925 / 1976 :=
by
  sorry

end determine_k_l43_43731


namespace initial_workers_number_l43_43788

-- Define the initial problem
variables {W : ℕ} -- Number of initial workers
variables (Work1 : ℕ := W * 8) -- Work done for the first hole
variables (Work2 : ℕ := (W + 65) * 6) -- Work done for the second hole
variables (Depth1 : ℕ := 30) -- Depth of the first hole
variables (Depth2 : ℕ := 55) -- Depth of the second hole

-- Expressing the conditions and question
theorem initial_workers_number : 8 * W * 55 = 30 * (W + 65) * 6 → W = 45 :=
by
  sorry

end initial_workers_number_l43_43788


namespace carol_maximizes_at_0_75_l43_43316

def winning_probability (a b c : ℝ) : Prop :=
(0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (a < c ∧ c < b ∨ b < c ∧ c < a)

theorem carol_maximizes_at_0_75 :
  ∀ (a b : ℝ), (0 ≤ a ∧ a ≤ 1) → (0.25 ≤ b ∧ b ≤ 0.75) → (∃ c : ℝ, 0 ≤ c ∧ c ≤ 1 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → winning_probability a b x ≤ winning_probability a b 0.75)) :=
sorry

end carol_maximizes_at_0_75_l43_43316


namespace mt_product_l43_43382

noncomputable def g (x : ℝ) : ℝ := sorry

theorem mt_product
  (hg : ∀ (x y : ℝ), g (g x + y) = g x + g (g y + g (-x)) - x) : 
  ∃ m t : ℝ, m = 1 ∧ t = -5 ∧ m * t = -5 := 
by
  sorry

end mt_product_l43_43382


namespace decimal_to_fraction_equivalence_l43_43479

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43479


namespace inequality_a_inequality_c_inequality_d_l43_43640

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l43_43640


namespace proof_problem_l43_43385

variables {R : Type*} [Field R] (p q r u v w : R)

theorem proof_problem (h₁ : 15*u + q*v + r*w = 0)
                      (h₂ : p*u + 25*v + r*w = 0)
                      (h₃ : p*u + q*v + 50*w = 0)
                      (hp : p ≠ 15)
                      (hu : u ≠ 0) : 
                      (p / (p - 15) + q / (q - 25) + r / (r - 50)) = 1 := 
by sorry

end proof_problem_l43_43385


namespace combination_formula_l43_43148

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l43_43148


namespace students_with_one_talent_l43_43529

-- Define the given conditions
def total_students := 120
def cannot_sing := 30
def cannot_dance := 50
def both_skills := 10

-- Define the problem statement
theorem students_with_one_talent :
  (total_students - cannot_sing - both_skills) + (total_students - cannot_dance - both_skills) = 130 :=
by
  sorry

end students_with_one_talent_l43_43529


namespace two_point_three_five_as_fraction_l43_43453

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43453


namespace part1_part2_l43_43353

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem part1 (hx : f (-x) = 2 * f x) : f x ^ 2 = 2 / 5 := 
  sorry

theorem part2 : 
  ∀ k : ℤ, ∃ a b : ℝ, [a, b] = [2 * π * k + (5 * π / 6), 2 * π * k + (11 * π / 6)] ∧ 
  ∀ x : ℝ, x ∈ Set.Icc a b → ∀ y : ℝ, y = f (π / 12 - x) → 
  ∃ δ > 0, ∀ ε > 0, 0 < |x - y| ∧ |x - y| < δ → y < x := 
  sorry

end part1_part2_l43_43353


namespace speed_of_stream_l43_43690

-- Definitions of the problem's conditions
def downstream_distance := 72
def upstream_distance := 30
def downstream_time := 3
def upstream_time := 3

-- The unknowns
variables (b s : ℝ)

-- The effective speed equations based on the problem conditions
def effective_speed_downstream := b + s
def effective_speed_upstream := b - s

-- The core conditions of the problem
def condition1 : Prop := downstream_distance = effective_speed_downstream * downstream_time
def condition2 : Prop := upstream_distance = effective_speed_upstream * upstream_time

-- The problem statement transformed into a Lean theorem
theorem speed_of_stream (h1 : condition1) (h2 : condition2) : s = 7 := 
sorry

end speed_of_stream_l43_43690


namespace magician_red_marbles_taken_l43_43120

theorem magician_red_marbles_taken:
  ∃ R : ℕ, (20 - R) + (30 - 4 * R) = 35 ∧ R = 3 :=
by
  sorry

end magician_red_marbles_taken_l43_43120


namespace tangent_line_at_1_inequality_l43_43736

noncomputable def f : ℝ → ℝ := λ x, Real.exp x / x

def tangent_line_eq (x : ℝ) := (1 : ℝ) = 1 → (f 1) = Real.exp 1

theorem tangent_line_at_1 : tangent_line_eq 1 :=
by
  -- proof of the tangent line equation would go here
  sorry

theorem inequality (x : ℝ) (h : x ≠ 0) : 
  (1 / (x * f x) > 1 - x) :=
by
  -- proof of the inequality would go here
  sorry

end tangent_line_at_1_inequality_l43_43736


namespace find_common_ratio_l43_43621

variable {a : ℕ → ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

theorem find_common_ratio (h : geometric_sequence_q a q) : q = 2 :=
by
  sorry

end find_common_ratio_l43_43621


namespace product_of_numbers_l43_43664

theorem product_of_numbers (a b : ℕ) (hcf : ℕ := 12) (lcm : ℕ := 205) (ha : Nat.gcd a b = hcf) (hb : Nat.lcm a b = lcm) : a * b = 2460 := by
  sorry

end product_of_numbers_l43_43664


namespace registration_methods_l43_43368

theorem registration_methods :
  ∀ (interns : ℕ) (companies : ℕ), companies = 4 → interns = 5 → companies^interns = 1024 :=
by intros interns companies h1 h2; rw [h1, h2]; exact rfl

end registration_methods_l43_43368


namespace sum_of_p_and_q_l43_43183

-- Definitions for points and collinearity condition
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := {x := 1, y := 3, z := -2}
def B : Point3D := {x := 2, y := 5, z := 1}
def C (p q : ℝ) : Point3D := {x := p, y := 7, z := q - 2}

def collinear (A B C : Point3D) : Prop :=
  ∃ (k : ℝ), B.x - A.x = k * (C.x - A.x) ∧ B.y - A.y = k * (C.y - A.y) ∧ B.z - A.z = k * (C.z - A.z)

theorem sum_of_p_and_q (p q : ℝ) (h : collinear A B (C p q)) : p + q = 9 := by
  sorry

end sum_of_p_and_q_l43_43183


namespace decimal_to_fraction_l43_43469

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43469


namespace sugar_percentage_first_solution_l43_43243

theorem sugar_percentage_first_solution 
  (x : ℝ) (h1 : 0 < x ∧ x < 100) 
  (h2 : 17 = 3 / 4 * x + 1 / 4 * 38) : 
  x = 10 :=
sorry

end sugar_percentage_first_solution_l43_43243


namespace orange_balloons_count_l43_43615

variable (original_orange_balloons : ℝ)
variable (found_orange_balloons : ℝ)
variable (total_orange_balloons : ℝ)

theorem orange_balloons_count :
  original_orange_balloons = 9.0 →
  found_orange_balloons = 2.0 →
  total_orange_balloons = original_orange_balloons + found_orange_balloons →
  total_orange_balloons = 11.0 := by
  sorry

end orange_balloons_count_l43_43615


namespace exam_students_l43_43796

noncomputable def totalStudents (N : ℕ) (T : ℕ) := T = 70 * N
noncomputable def marksOfExcludedStudents := 5 * 50
noncomputable def remainingStudents (N : ℕ) := N - 5
noncomputable def remainingMarksCondition (N T : ℕ) := (T - marksOfExcludedStudents) / remainingStudents N = 90

theorem exam_students (N : ℕ) (T : ℕ) 
  (h1 : totalStudents N T) 
  (h2 : remainingMarksCondition N T) : 
  N = 10 :=
by 
  sorry

end exam_students_l43_43796


namespace melanie_gave_3_plums_to_sam_l43_43393

theorem melanie_gave_3_plums_to_sam 
  (initial_plums : ℕ) 
  (plums_left : ℕ) 
  (plums_given : ℕ) 
  (h1 : initial_plums = 7) 
  (h2 : plums_left = 4) 
  (h3 : plums_left + plums_given = initial_plums) : 
  plums_given = 3 :=
by 
  sorry

end melanie_gave_3_plums_to_sam_l43_43393


namespace factorial_div_eq_l43_43149

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l43_43149


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43635

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43635


namespace solve_fraction_equation_l43_43141

theorem solve_fraction_equation :
  ∀ (x : ℚ), (5 * x + 3) / (7 * x - 4) = 4128 / 4386 → x = 115 / 27 := by
  sorry

end solve_fraction_equation_l43_43141


namespace each_parent_payment_l43_43838

def original_salary : ℝ := 45000
def raise_percentage : ℝ := 0.2
def num_kids : ℕ := 9

def raise_amount : ℝ := original_salary * raise_percentage
def new_salary : ℝ := original_salary + raise_amount
def payment_per_parent : ℝ := new_salary / num_kids

theorem each_parent_payment (h1: raise_amount = 9000) (h2: new_salary = 54000) (h3: payment_per_parent = 6000) : payment_per_parent = 6000 :=
by
  sorry

end each_parent_payment_l43_43838


namespace problem_a_problem_b_problem_c_problem_d_l43_43614

theorem problem_a : 37.3 / (1 / 2) = 74.6 := by
  sorry

theorem problem_b : 0.45 - (1 / 20) = 0.4 := by
  sorry

theorem problem_c : (33 / 40) * (10 / 11) = 0.75 := by
  sorry

theorem problem_d : 0.375 + (1 / 40) = 0.4 := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l43_43614


namespace largest_value_of_x_not_defined_l43_43862

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b*b - 4*a*c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2*a)
  let x2 := (-b - sqrt_discriminant) / (2*a)
  (x1, x2)

noncomputable def largest_root : ℝ :=
  let (x1, x2) := quadratic_formula 4 (-81) 49
  if x1 > x2 then x1 else x2

theorem largest_value_of_x_not_defined :
  largest_root = 19.6255 :=
by
  sorry

end largest_value_of_x_not_defined_l43_43862


namespace problem_a_problem_c_problem_d_l43_43648

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l43_43648


namespace decimal_to_fraction_l43_43487

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43487


namespace min_attempts_to_pair_keys_suitcases_l43_43014

theorem min_attempts_to_pair_keys_suitcases (n : ℕ) : ∃ p : ℕ, (∀ (keyOpen : Fin n → Fin n), ∃ f : (Fin n × Fin n) → Bool, ∀ (i j : Fin n), i ≠ j → (keyOpen i = j ↔ f (i, j) = tt)) ∧ p = Nat.choose n 2 := by
  sorry

end min_attempts_to_pair_keys_suitcases_l43_43014


namespace population_increase_l43_43211

theorem population_increase (initial_population final_population: ℝ) (r: ℝ) : 
  initial_population = 14000 →
  final_population = 16940 →
  final_population = initial_population * (1 + r) ^ 2 →
  r = 0.1 :=
by
  intros h_initial h_final h_eq
  sorry

end population_increase_l43_43211


namespace difference_thursday_tuesday_l43_43921

-- Define the amounts given on each day
def amount_tuesday : ℕ := 8
def amount_wednesday : ℕ := 5 * amount_tuesday
def amount_thursday : ℕ := amount_wednesday + 9

-- Problem statement: prove that the difference between Thursday's and Tuesday's amount is $41
theorem difference_thursday_tuesday : amount_thursday - amount_tuesday = 41 := by
  sorry

end difference_thursday_tuesday_l43_43921


namespace fraction_of_tea_in_final_cup2_is_5_over_8_l43_43281

-- Defining the initial conditions and the transfers
structure CupContents where
  tea : ℚ
  milk : ℚ

def initialCup1 : CupContents := { tea := 6, milk := 0 }
def initialCup2 : CupContents := { tea := 0, milk := 3 }

def transferOneThird (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let teaTransferred := (1 / 3) * cup1.tea
  ( { cup1 with tea := cup1.tea - teaTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk } )

def transferOneFourth (cup2 : CupContents) (cup1 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup2.tea + cup2.milk
  let amountTransferred := (1 / 4) * mixedTotal
  let teaTransferred := amountTransferred * (cup2.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup2.milk / mixedTotal)
  ( { tea := cup1.tea + teaTransferred, milk := cup1.milk + milkTransferred },
    { tea := cup2.tea - teaTransferred, milk := cup2.milk - milkTransferred } )

def transferOneHalf (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup1.tea + cup1.milk
  let amountTransferred := (1 / 2) * mixedTotal
  let teaTransferred := amountTransferred * (cup1.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup1.milk / mixedTotal)
  ( { tea := cup1.tea - teaTransferred, milk := cup1.milk - milkTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk + milkTransferred } )

def finalContents (cup1 cup2 : CupContents) : CupContents × CupContents :=
  let (cup1Transferred, cup2Transferred) := transferOneThird cup1 cup2
  let (cup1Mixed, cup2Mixed) := transferOneFourth cup2Transferred cup1Transferred
  transferOneHalf cup1Mixed cup2Mixed

-- Statement to be proved
theorem fraction_of_tea_in_final_cup2_is_5_over_8 :
  ((finalContents initialCup1 initialCup2).snd.tea / ((finalContents initialCup1 initialCup2).snd.tea + (finalContents initialCup1 initialCup2).snd.milk) = 5 / 8) :=
sorry

end fraction_of_tea_in_final_cup2_is_5_over_8_l43_43281


namespace find_operation_l43_43268

theorem find_operation (a b : ℝ) (h_a : a = 0.137) (h_b : b = 0.098) :
  ((a + b) ^ 2 - (a - b) ^ 2) / (a * b) = 4 :=
by
  sorry

end find_operation_l43_43268


namespace sum_consecutive_powers_of_2_divisible_by_6_l43_43399

theorem sum_consecutive_powers_of_2_divisible_by_6 (n : ℕ) :
  ∃ k : ℕ, 2^n + 2^(n+1) = 6 * k :=
sorry

end sum_consecutive_powers_of_2_divisible_by_6_l43_43399


namespace range_of_a_l43_43588

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by {
  sorry -- Proof is not required as per instructions.
}

end range_of_a_l43_43588


namespace largest_cuts_9x9_l43_43284

theorem largest_cuts_9x9 (k : ℕ) (V E F : ℕ) (hV : V = 81) (hE : E = 4 * k) (hF : F = 1 + 2 * k)
  (hEuler : V - E + F ≥ 2) : k ≤ 21 :=
by
  sorry

end largest_cuts_9x9_l43_43284


namespace min_value_of_expr_l43_43005

def expr (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

theorem min_value_of_expr : ∃ x y : ℝ, expr x y = -2 / 3 :=
by
  sorry

end min_value_of_expr_l43_43005


namespace simple_interest_double_in_4_years_interest_25_percent_l43_43129

theorem simple_interest_double_in_4_years_interest_25_percent :
  ∀ {P : ℕ} (h : P > 0), ∃ (R : ℕ), R = 25 ∧ P + P * R * 4 / 100 = 2 * P :=
by
  sorry

end simple_interest_double_in_4_years_interest_25_percent_l43_43129


namespace total_sand_correct_l43_43531

-- Define the conditions as variables and equations:
variables (x : ℕ) -- original days scheduled to complete
variables (total_sand : ℕ) -- total amount of sand in tons

-- Define the conditions in the problem:
def original_daily_amount := 15  -- tons per day as scheduled
def actual_daily_amount := 20  -- tons per day in reality
def days_ahead := 3  -- days finished ahead of schedule

-- Equation representing the planned and actual transportation:
def planned_sand := original_daily_amount * x
def actual_sand := actual_daily_amount * (x - days_ahead)

-- The goal is to prove:
theorem total_sand_correct : planned_sand = actual_sand → total_sand = 180 :=
by
  sorry

end total_sand_correct_l43_43531


namespace c_S_power_of_2_l43_43763

variables (m : ℕ) (S : String)

-- condition: m > 1
def is_valid_m (m : ℕ) : Prop := m > 1

-- function c(S)
def c (S : String) : ℕ := sorry  -- actual implementation is skipped

-- function to check if a number represented by a string is divisible by m
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

-- Property that c(S) can take only powers of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem c_S_power_of_2 (m : ℕ) (S : String) (h1 : is_valid_m m) :
  is_power_of_two (c S) :=
sorry

end c_S_power_of_2_l43_43763


namespace infinite_series_equals_two_l43_43327

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l43_43327


namespace minimum_value_of_expression_l43_43744

theorem minimum_value_of_expression (x : ℝ) (h : x > 2) : 
  ∃ y, (∀ z, z > 2 → (z^2 - 4 * z + 5) / (z - 2) ≥ y) ∧ 
       y = 2 :=
by
  sorry

end minimum_value_of_expression_l43_43744


namespace mean_weight_players_l43_43942

/-- Definitions for the weights of the players and proving the mean weight. -/
def weights : List ℕ := [62, 65, 70, 73, 73, 76, 78, 79, 81, 81, 82, 84, 87, 89, 89, 89, 90, 93, 95]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

theorem mean_weight_players : mean weights = 80.84 := by
  sorry

end mean_weight_players_l43_43942


namespace question1_question2_l43_43016

noncomputable def minimum_value (x y : ℝ) : ℝ := (1 / x) + (1 / y)

theorem question1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) : 
  minimum_value x y = 2 :=
sorry

theorem question2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
  (x + 1) * (y + 1) ≠ 5 :=
sorry

end question1_question2_l43_43016


namespace red_shoe_probability_l43_43680

variable (red_shoes : ℕ) (green_shoes : ℕ)

def total_shoes (red_shoes: ℕ) (green_shoes: ℕ) : ℕ := red_shoes + green_shoes
def first_red_prob (red_shoes: ℕ) (total_shoes: ℕ) : ℚ := red_shoes / total_shoes
def second_red_prob (red_shoes: ℕ) (total_shoes: ℕ) : ℚ := (red_shoes - 1) / (total_shoes - 1)

theorem red_shoe_probability (h1 : red_shoes = 5) (h2 : green_shoes = 4) :
  first_red_prob red_shoes (total_shoes red_shoes green_shoes) * second_red_prob red_shoes (total_shoes red_shoes green_shoes) = 5 / 18 := by
  let total := total_shoes red_shoes green_shoes
  have h_total : total = 9 := by
    rw [h1, h2]
    norm_num
  sorry

end red_shoe_probability_l43_43680


namespace exponential_difference_l43_43186

theorem exponential_difference (f : ℕ → ℕ) (x : ℕ) (h : f x = 3^x) : f (x + 2) - f x = 8 * f x :=
by sorry

end exponential_difference_l43_43186


namespace ratio_green_to_yellow_l43_43395

theorem ratio_green_to_yellow (yellow fish blue fish green fish total fish : ℕ) 
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_total : total = yellow + blue + green)
  (h_aquarium_total : total = 42) : 
  green / yellow = 2 := 
sorry

end ratio_green_to_yellow_l43_43395


namespace problem_quadrilateral_inscribed_in_circle_l43_43525

theorem problem_quadrilateral_inscribed_in_circle
  (r : ℝ)
  (AB BC CD DA : ℝ)
  (h_radius : r = 300 * Real.sqrt 2)
  (h_AB : AB = 300)
  (h_BC : BC = 150)
  (h_CD : CD = 150) :
  DA = 750 :=
sorry

end problem_quadrilateral_inscribed_in_circle_l43_43525


namespace original_number_l43_43223

variable (x : ℝ)

theorem original_number (h1 : x - x / 10 = 37.35) : x = 41.5 := by
  sorry

end original_number_l43_43223


namespace curve_is_line_l43_43865

theorem curve_is_line : ∀ (r θ : ℝ), r = 2 / (2 * Real.sin θ - Real.cos θ) → ∃ m b, ∀ (x y : ℝ), x = r * Real.cos θ → y = r * Real.sin θ → y = m * x + b :=
by
  intros r θ h
  sorry

end curve_is_line_l43_43865


namespace problem_equivalent_proof_l43_43610

def sequence_row1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence_row2 (n : ℕ) : ℤ := sequence_row1 n - 1
def sequence_row3 (n : ℕ) : ℤ := (-2)^n - sequence_row2 n

theorem problem_equivalent_proof :
  let a := sequence_row1 7
  let b := sequence_row2 7
  let c := sequence_row3 7
  a - b + c = -254 :=
by
  sorry

end problem_equivalent_proof_l43_43610


namespace regular_polygon_exterior_angle_l43_43539

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end regular_polygon_exterior_angle_l43_43539


namespace suitable_comprehensive_survey_l43_43286

theorem suitable_comprehensive_survey :
  ¬(A = "comprehensive") ∧ ¬(B = "comprehensive") ∧ (C = "comprehensive") ∧ ¬(D = "comprehensive") → 
  suitable_survey = "C" :=
by
  sorry

end suitable_comprehensive_survey_l43_43286


namespace line_symmetric_y_axis_eqn_l43_43075

theorem line_symmetric_y_axis_eqn (x y : ℝ) : 
  (∀ x y : ℝ, x - y + 1 = 0 → x + y - 1 = 0) := 
sorry

end line_symmetric_y_axis_eqn_l43_43075


namespace petya_time_comparison_l43_43543

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l43_43543


namespace find_two_digit_number_l43_43957

theorem find_two_digit_number
  (X : ℕ)
  (h1 : 57 + (10 * X + 6) = 123)
  (h2 : two_digit_number = 10 * X + 9) :
  two_digit_number = 69 :=
by
  sorry

end find_two_digit_number_l43_43957


namespace number_of_cities_from_group_B_l43_43842

theorem number_of_cities_from_group_B
  (total_cities : ℕ)
  (cities_in_A : ℕ)
  (cities_in_B : ℕ)
  (cities_in_C : ℕ)
  (sampled_cities : ℕ)
  (h1 : total_cities = cities_in_A + cities_in_B + cities_in_C)
  (h2 : total_cities = 24)
  (h3 : cities_in_A = 4)
  (h4 : cities_in_B = 12)
  (h5 : cities_in_C = 8)
  (h6 : sampled_cities = 6) :
  cities_in_B * sampled_cities / total_cities = 3 := 
  by 
    sorry

end number_of_cities_from_group_B_l43_43842


namespace largest_integer_less_than_120_with_remainder_5_div_8_l43_43165

theorem largest_integer_less_than_120_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 120 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 120 → m % 8 = 5 → m ≤ n :=
sorry

end largest_integer_less_than_120_with_remainder_5_div_8_l43_43165


namespace train_length_l43_43434

def relative_speed (v_fast v_slow : ℕ) : ℚ :=
  v_fast - v_slow

def convert_speed (speed : ℚ) : ℚ :=
  (speed * 1000) / 3600

def covered_distance (speed : ℚ) (time_seconds : ℚ) : ℚ :=
  speed * time_seconds

theorem train_length (L : ℚ) (v_fast v_slow : ℕ) (time_seconds : ℚ)
    (hf : v_fast = 42) (hs : v_slow = 36) (ht : time_seconds = 36)
    (hc : relative_speed v_fast v_slow * 1000 / 3600 * time_seconds = 2 * L) :
    L = 30 := by
  sorry

end train_length_l43_43434


namespace train_length_equals_750_l43_43417

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l43_43417


namespace girls_attended_festival_l43_43741

variable (g b : ℕ)

theorem girls_attended_festival :
  g + b = 1500 ∧ (2 / 3) * g + (1 / 2) * b = 900 → (2 / 3) * g = 600 := by
  sorry

end girls_attended_festival_l43_43741


namespace triangle_perimeter_l43_43212

-- Let the lengths of the sides of the triangle be a, b, c.
variables (a b c : ℕ)
-- To represent the sides with specific lengths as stated in the problem.
def side1 := 2
def side2 := 5

-- The condition that the third side must be an odd integer.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Setting up the third side based on the given conditions.
def third_side_odd (c : ℕ) : Prop := 3 < c ∧ c < 7 ∧ is_odd c

-- The perimeter of the triangle.
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The main theorem to prove.
theorem triangle_perimeter (c : ℕ) (h_odd : third_side_odd c) : perimeter side1 side2 c = 12 :=
by
  sorry

end triangle_perimeter_l43_43212


namespace infinite_sum_problem_l43_43324

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l43_43324


namespace yogurt_cost_l43_43273

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end yogurt_cost_l43_43273


namespace max_knights_l43_43241

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end max_knights_l43_43241


namespace decimal_to_fraction_l43_43472

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43472


namespace min_abs_sum_l43_43818

theorem min_abs_sum : ∃ x : ℝ, (|x + 1| + |x + 2| + |x + 6|) = 5 :=
sorry

end min_abs_sum_l43_43818


namespace budget_spent_on_utilities_l43_43687

noncomputable def budget_is_correct : Prop :=
  let total_budget := 100
  let salaries := 60
  let r_and_d := 9
  let equipment := 4
  let supplies := 2
  let degrees_in_circle := 360
  let transportation_degrees := 72
  let transportation_percentage := (transportation_degrees * total_budget) / degrees_in_circle
  let known_percentages := salaries + r_and_d + equipment + supplies + transportation_percentage
  let utilities_percentage := total_budget - known_percentages
  utilities_percentage = 5

theorem budget_spent_on_utilities : budget_is_correct :=
  sorry

end budget_spent_on_utilities_l43_43687


namespace sum_of_digits_is_13_l43_43891

theorem sum_of_digits_is_13:
  ∀ (a b c d : ℕ),
  b + c = 10 ∧
  c + d = 1 ∧
  a + d = 2 →
  a + b + c + d = 13 :=
by {
  sorry
}

end sum_of_digits_is_13_l43_43891


namespace total_candy_given_l43_43706

def candy_given_total (a b c : ℕ) : ℕ := a + b + c

def first_10_friends_candy (n : ℕ) := 10 * n

def next_7_friends_candy (n : ℕ) := 7 * (2 * n)

def remaining_friends_candy := 50

theorem total_candy_given (n : ℕ) (h1 : first_10_friends_candy 12 = 120)
  (h2 : next_7_friends_candy 12 = 168) (h3 : remaining_friends_candy = 50) :
  candy_given_total 120 168 50 = 338 := by
  sorry

end total_candy_given_l43_43706


namespace height_of_triangle_l43_43072

theorem height_of_triangle (base height area : ℝ) (h1 : base = 6) (h2 : area = 24) (h3 : area = 1 / 2 * base * height) : height = 8 :=
by sorry

end height_of_triangle_l43_43072


namespace christine_min_bottles_l43_43549

theorem christine_min_bottles
  (fluid_ounces_needed : ℕ)
  (bottle_volume_ml : ℕ)
  (fluid_ounces_per_liter : ℝ)
  (liters_in_milliliter : ℕ)
  (required_bottles : ℕ)
  (h1 : fluid_ounces_needed = 45)
  (h2 : bottle_volume_ml = 200)
  (h3 : fluid_ounces_per_liter = 33.8)
  (h4 : liters_in_milliliter = 1000)
  (h5 : required_bottles = 7) :
  required_bottles = ⌈(fluid_ounces_needed * liters_in_milliliter) / (bottle_volume_ml * fluid_ounces_per_liter)⌉ := by
  sorry

end christine_min_bottles_l43_43549


namespace power_comparison_l43_43495

theorem power_comparison : (9^20 : ℝ) < (9999^10 : ℝ) :=
sorry

end power_comparison_l43_43495


namespace different_values_count_l43_43006

theorem different_values_count (i : ℕ) (h : 1 ≤ i ∧ i ≤ 2015) : 
  ∃ l : Finset ℕ, (∀ j ∈ l, ∃ i : ℕ, (1 ≤ i ∧ i ≤ 2015) ∧ j = (i^2 / 2015)) ∧
  l.card = 2016 := 
sorry

end different_values_count_l43_43006


namespace angle_B_equiv_60_l43_43037

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A

theorem angle_B_equiv_60 
  (a b c A B C : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π)
  (h6 : 0 < B) (h7 : B < π)
  (h8 : 0 < C) (h9 : C < π)
  (h_triangle : A + B + C = π)
  (h_arith : triangle_condition a b c A B C) : 
  B = π / 3 :=
by
  sorry

end angle_B_equiv_60_l43_43037


namespace number_of_intersection_points_l43_43553

-- Definitions of the given lines
def line1 (x y : ℝ) : Prop := 6 * y - 4 * x = 2
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := -4 * x + 6 * y = 3

-- Definitions of the intersection points
def intersection1 (x y : ℝ) : Prop := line1 x y ∧ line2 x y
def intersection2 (x y : ℝ) : Prop := line2 x y ∧ line3 x y

-- Definition of the problem
theorem number_of_intersection_points : 
  (∃ x y : ℝ, intersection1 x y) ∧
  (∃ x y : ℝ, intersection2 x y) ∧
  (¬ ∃ x y : ℝ, line1 x y ∧ line3 x y) →
  (∃ z : ℕ, z = 2) :=
sorry

end number_of_intersection_points_l43_43553


namespace decimal_to_fraction_l43_43440

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43440


namespace smallest_y_value_l43_43817

theorem smallest_y_value : 
  ∀ y : ℝ, (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end smallest_y_value_l43_43817


namespace roots_range_of_a_l43_43351

theorem roots_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + (a - 2)*|x - 3| + 9 - 2*a = 0) ↔ a > 0 ∨ a = -2 :=
sorry

end roots_range_of_a_l43_43351


namespace factorial_div_combination_l43_43158

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l43_43158


namespace smallest_w_value_l43_43362

theorem smallest_w_value (x y z w : ℝ) 
    (hx : -2 ≤ x ∧ x ≤ 5) 
    (hy : -3 ≤ y ∧ y ≤ 7) 
    (hz : 4 ≤ z ∧ z ≤ 8) 
    (hw : w = x * y - z) : 
    w ≥ -23 :=
sorry

end smallest_w_value_l43_43362


namespace central_angle_of_sector_l43_43584

theorem central_angle_of_sector (r α : ℝ) (h_arc_length : α * r = 5) (h_area : 0.5 * α * r^2 = 5): α = 5 / 2 := by
  sorry

end central_angle_of_sector_l43_43584


namespace probability_full_house_after_rerolling_l43_43007

theorem probability_full_house_after_rerolling
  (a b c : ℕ)
  (h0 : a ≠ b)
  (h1 : c ≠ a)
  (h2 : c ≠ b) :
  (2 / 6 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_full_house_after_rerolling_l43_43007


namespace monotonic_increase_interval_l43_43076

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_increase_interval : ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (Real.log x) / x :=
by sorry

end monotonic_increase_interval_l43_43076


namespace decimal_to_fraction_l43_43437

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43437


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43634

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43634


namespace peya_time_comparison_l43_43546

variable (V D : ℝ) (hV : 0 < V) (hD : 0 < D)

def planned_time : ℝ := D / V
def increased_speed : ℝ := 1.25 * V
def decreased_speed : ℝ := 0.80 * V

def first_half_distance : ℝ := D / 2
def second_half_distance : ℝ := D / 2

def time_first_half : ℝ := first_half_distance / increased_speed
def time_second_half : ℝ := second_half_distance / decreased_speed

def actual_time : ℝ := time_first_half + time_second_half

theorem peya_time_comparison : actual_time V D = (41 * D) / (40 * V) > (D / V) :=
by {
  unfold actual_time,
  unfold time_first_half time_second_half,
  unfold first_half_distance second_half_distance,
  unfold increased_speed decreased_speed,
  unfold planned_time,
  sorry
}

end peya_time_comparison_l43_43546


namespace tessellation_coloring_l43_43374

theorem tessellation_coloring :
  ∀ (T : Type) (colors : T → ℕ) (adjacent : T → T → Prop),
    (∀ t1 t2, adjacent t1 t2 → colors t1 ≠ colors t2) → 
    (∃ c1 c2 c3, ∀ t, colors t = c1 ∨ colors t = c2 ∨ colors t = c3) :=
sorry

end tessellation_coloring_l43_43374


namespace additional_amount_deductibles_next_year_l43_43396

theorem additional_amount_deductibles_next_year :
  let avg_deductible : ℝ := 3000
  let inflation_rate : ℝ := 0.03
  let plan_a_rate : ℝ := 2 / 3
  let plan_b_rate : ℝ := 1 / 2
  let plan_c_rate : ℝ := 3 / 5
  let plan_a_percent : ℝ := 0.40
  let plan_b_percent : ℝ := 0.30
  let plan_c_percent : ℝ := 0.30
  let additional_a : ℝ := avg_deductible * plan_a_rate
  let additional_b : ℝ := avg_deductible * plan_b_rate
  let additional_c : ℝ := avg_deductible * plan_c_rate
  let weighted_additional : ℝ := (additional_a * plan_a_percent) + (additional_b * plan_b_percent) + (additional_c * plan_c_percent)
  let inflation_increase : ℝ := weighted_additional * inflation_rate
  let total_additional_amount : ℝ := weighted_additional + inflation_increase
  total_additional_amount = 1843.70 :=
sorry

end additional_amount_deductibles_next_year_l43_43396


namespace alpha_cubed_plus_5beta_plus_10_l43_43347

noncomputable def α: ℝ := sorry
noncomputable def β: ℝ := sorry

-- Given conditions
axiom roots_eq : ∀ x : ℝ, x^2 + 2 * x - 1 = 0 → (x = α ∨ x = β)
axiom sum_eq : α + β = -2
axiom prod_eq : α * β = -1

-- The theorem stating the desired result
theorem alpha_cubed_plus_5beta_plus_10 :
  α^3 + 5 * β + 10 = -2 :=
sorry

end alpha_cubed_plus_5beta_plus_10_l43_43347


namespace jerry_age_l43_43050

theorem jerry_age
  (M J : ℕ)
  (h1 : M = 2 * J + 5)
  (h2 : M = 21) :
  J = 8 :=
by
  sorry

end jerry_age_l43_43050


namespace extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l43_43261

theorem extremum_implies_derivative_zero {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_extremum : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :
  deriv f x₀ = 0 :=
sorry

theorem derivative_zero_not_implies_extremum {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_deriv_zero : deriv f x₀ = 0) :
  ¬ (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :=
sorry

end extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l43_43261


namespace farthings_in_a_pfennig_l43_43172

theorem farthings_in_a_pfennig (x : ℕ) (h : 54 - 2 * x = 7 * x) : x = 6 :=
by
  sorry

end farthings_in_a_pfennig_l43_43172


namespace largest_multiple_of_seven_smaller_than_neg_85_l43_43094

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l43_43094


namespace xyz_product_value_l43_43192

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l43_43192


namespace Tony_packs_of_pens_l43_43594

theorem Tony_packs_of_pens (T : ℕ) 
  (Kendra_packs : ℕ := 4) 
  (pens_per_pack : ℕ := 3) 
  (Kendra_keep : ℕ := 2) 
  (Tony_keep : ℕ := 2)
  (friends_pens : ℕ := 14) 
  (total_pens_given : Kendra_packs * pens_per_pack - Kendra_keep + 3 * T - Tony_keep = friends_pens) :
  T = 2 :=
by {
  sorry
}

end Tony_packs_of_pens_l43_43594


namespace largest_multiple_of_7_less_than_neg85_l43_43091

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l43_43091


namespace correct_calculation_for_A_l43_43822

theorem correct_calculation_for_A (x : ℝ) : (-2 * x) ^ 3 = -8 * x ^ 3 :=
by
  sorry

end correct_calculation_for_A_l43_43822


namespace brad_ate_six_halves_l43_43029

theorem brad_ate_six_halves (total_cookies : ℕ) (total_halves : ℕ) (greg_ate : ℕ) (halves_left : ℕ) (halves_brad_ate : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : total_halves = total_cookies * 2)
  (h3 : greg_ate = 4)
  (h4 : halves_left = 18)
  (h5 : total_halves - greg_ate - halves_brad_ate = halves_left) :
  halves_brad_ate = 6 :=
by
  sorry

end brad_ate_six_halves_l43_43029


namespace minimum_area_convex_quadrilateral_l43_43413

theorem minimum_area_convex_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  (∀ S_BOC S_AOD : ℝ, S_AOB * S_COD = S_BOC * S_AOD → 
    (S_AOB + S_BOC + S_COD + S_AOD) ≥ 25) := sorry

end minimum_area_convex_quadrilateral_l43_43413


namespace flour_already_put_in_l43_43606

def total_flour : ℕ := 8
def additional_flour_needed : ℕ := 6

theorem flour_already_put_in : total_flour - additional_flour_needed = 2 := by
  sorry

end flour_already_put_in_l43_43606


namespace unique_non_congruent_rectangle_with_conditions_l43_43527

theorem unique_non_congruent_rectangle_with_conditions :
  ∃! (w h : ℕ), 2 * (w + h) = 80 ∧ w * h = 400 :=
by
  sorry

end unique_non_congruent_rectangle_with_conditions_l43_43527


namespace relationship_between_M_and_N_l43_43869

theorem relationship_between_M_and_N (a b : ℝ) (M N : ℝ) 
  (hM : M = a^2 - a * b) 
  (hN : N = a * b - b^2) : M ≥ N :=
by sorry

end relationship_between_M_and_N_l43_43869


namespace total_weight_of_rings_l43_43762

theorem total_weight_of_rings :
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 :=
by
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  sorry

end total_weight_of_rings_l43_43762


namespace ball_arrangements_l43_43811

theorem ball_arrangements : 
  let red : ℕ := 6
  let green : ℕ := 3
  let total_balls : ℕ := red + green
  let selected_balls : ℕ := 4
  (finset.card (finset.univ.powerset.filter (λ s, s.card = selected_balls))) = 15 :=
by 
  sorry

end ball_arrangements_l43_43811


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l43_43953

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  sum_of_digits (2 ^ 2010 * 5 ^ 2012 * 7) = 13 :=
by {
  -- We'll insert the detailed proof here
  sorry
}

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l43_43953


namespace shopper_saves_more_l43_43804

-- Definitions and conditions
def cover_price : ℝ := 30
def percent_discount : ℝ := 0.25
def dollar_discount : ℝ := 5
def first_discounted_price : ℝ := cover_price * (1 - percent_discount)
def second_discounted_price : ℝ := first_discounted_price - dollar_discount
def first_dollar_discounted_price : ℝ := cover_price - dollar_discount
def second_percent_discounted_price : ℝ := first_dollar_discounted_price * (1 - percent_discount)

def additional_savings : ℝ := second_percent_discounted_price - second_discounted_price

-- Theorem stating the shopper saves 125 cents more with 25% first
theorem shopper_saves_more : additional_savings = 1.25 := by
  sorry

end shopper_saves_more_l43_43804


namespace car_total_distance_l43_43821

-- Define the arithmetic sequence (s_n) where a = 40 and d = -10
def car_travel (n : ℕ) : ℕ := if n > 0 then 40 - 10 * (n - 1) else 0

-- Define the sum of the first 'k' terms of the arithmetic sequence
noncomputable def sum_car_travel (k : ℕ) : ℕ :=
  ∑ i in Finset.range k, car_travel (i + 1)

-- Main theorem statement
theorem car_total_distance : sum_car_travel 4 = 100 :=
by
  sorry

end car_total_distance_l43_43821


namespace smallest_whole_number_larger_than_sum_l43_43714

noncomputable def mixed_number1 : ℚ := 3 + 2/3
noncomputable def mixed_number2 : ℚ := 4 + 1/4
noncomputable def mixed_number3 : ℚ := 5 + 1/5
noncomputable def mixed_number4 : ℚ := 6 + 1/6
noncomputable def mixed_number5 : ℚ := 7 + 1/7

noncomputable def sum_of_mixed_numbers : ℚ :=
  mixed_number1 + mixed_number2 + mixed_number3 + mixed_number4 + mixed_number5

theorem smallest_whole_number_larger_than_sum : 
  ∃ n : ℤ, (n : ℚ) > sum_of_mixed_numbers ∧ n = 27 :=
by
  sorry

end smallest_whole_number_larger_than_sum_l43_43714


namespace sat_marking_problem_l43_43288

-- Define the recurrence relation for the number of ways to mark questions without consecutive markings of the same letter.
def f : ℕ → ℕ
| 0     => 1
| 1     => 2
| 2     => 3
| (n+2) => f (n+1) + f n

-- Define that each letter marking can be done in 32 different ways.
def markWays : ℕ := 32

-- Define the number of questions to be 10.
def numQuestions : ℕ := 10

-- Calculate the number of sequences of length numQuestions with no consecutive same markings.
def numWays := f numQuestions

-- Prove that the number of ways results in 2^20 * 3^10 and compute 100m + n + p where m = 20, n = 10, p = 3.
theorem sat_marking_problem :
  (numWays ^ 5 = 2 ^ 20 * 3 ^ 10) ∧ (100 * 20 + 10 + 3 = 2013) :=
by
  sorry

end sat_marking_problem_l43_43288


namespace log_travel_time_24_l43_43377

noncomputable def time_for_log_to_travel (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) : ℝ :=
  D / v

theorem log_travel_time_24 (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) :
  time_for_log_to_travel D u v h1 h2 = 24 :=
sorry

end log_travel_time_24_l43_43377


namespace ratio_of_friday_to_thursday_l43_43968

theorem ratio_of_friday_to_thursday
  (wednesday_copies : ℕ)
  (total_copies : ℕ)
  (ratio : ℚ)
  (h1 : wednesday_copies = 15)
  (h2 : total_copies = 69)
  (h3 : ratio = 1 / 5) :
  (total_copies - wednesday_copies - 3 * wednesday_copies) / (3 * wednesday_copies) = ratio :=
by
  -- proof goes here
  sorry

end ratio_of_friday_to_thursday_l43_43968


namespace decimal_to_fraction_l43_43445

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43445


namespace problem1_problem2_problem3_problem4_problem5_problem6_l43_43651

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l43_43651


namespace factor_present_l43_43859

noncomputable def given_expr := (x^2 - y^2 - z^2 + 2 * y * z + x + y - z)

theorem factor_present:
  ∃ f: Polynomial ℤ, ∃ g: Polynomial ℤ, given_expr = f * g ∧ ( f = x - y + z + 1 ∨ g = x - y + z + 1 ) :=
sorry

end factor_present_l43_43859


namespace product_xyz_l43_43199

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l43_43199


namespace photo_album_requirement_l43_43515

-- Definition of the conditions
def pages_per_album : ℕ := 32
def photos_per_page : ℕ := 5
def total_photos : ℕ := 900

-- Calculation of photos per album
def photos_per_album := pages_per_album * photos_per_page

-- Calculation of required albums
noncomputable def albums_needed := (total_photos + photos_per_album - 1) / photos_per_album

-- Theorem to prove the required number of albums is 6
theorem photo_album_requirement : albums_needed = 6 :=
  by sorry

end photo_album_requirement_l43_43515


namespace number_of_sequences_l43_43305

-- Define the number of possible outcomes for a single coin flip
def coinFlipOutcomes : ℕ := 2

-- Define the number of flips
def numberOfFlips : ℕ := 8

-- Theorem statement: The number of distinct sequences when flipping a coin eight times is 256
theorem number_of_sequences (n : ℕ) (outcomes : ℕ) (h : outcomes = 2) (hn : n = 8) : outcomes ^ n = 256 := by
  sorry

end number_of_sequences_l43_43305


namespace c_minus_a_is_10_l43_43234

variable (a b c d k : ℝ)

theorem c_minus_a_is_10 (h1 : a + b = 90)
                        (h2 : b + c = 100)
                        (h3 : a + c + d = 180)
                        (h4 : a^2 + b^2 + c^2 + d^2 = k) :
  c - a = 10 :=
by sorry

end c_minus_a_is_10_l43_43234


namespace problem_a_problem_b_problem_c_l43_43626

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l43_43626


namespace minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l43_43870

theorem minimum_value_x_plus_four_over_x (x : ℝ) (h : x ≥ 2) : 
  x + 4 / x ≥ 4 :=
by sorry

theorem minimum_value_occurs_at_x_eq_2 : ∀ (x : ℝ), x ≥ 2 → (x + 4 / x = 4 ↔ x = 2) :=
by sorry

end minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l43_43870


namespace prob_no_risk_factors_given_no_X_l43_43897

open ProbabilityTheory

def total_population : ℕ := 200

def prob_one_risk_factor (X Y Z : Prop) : ℝ := 0.07
def prob_two_risk_factors (X Y Z : Prop) : ℝ := 0.12
def prob_all_three_given_X_Y (X Y Z : Prop) : ℝ := 0.4

theorem prob_no_risk_factors_given_no_X (X Y Z : Prop) 
  (h_pop : total_population = 200)
  (h_one : prob_one_risk_factor X Y Z = 0.07)
  (h_two : prob_two_risk_factors X Y Z = 0.12)
  (h_conditional : prob_all_three_given_X_Y X Y Z = 0.4) :
  (70 / 122 : ℝ) = sorry :=
sorry

end prob_no_risk_factors_given_no_X_l43_43897


namespace inequality_nonneg_real_l43_43913

theorem inequality_nonneg_real (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ (2 / (1 + a * b)) ∧ ((1 / (1 + a^2)) + (1 / (1 + b^2)) = (2 / (1 + a * b)) ↔ a = b) :=
sorry

end inequality_nonneg_real_l43_43913


namespace inequality_proof_l43_43017

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 :=
by
  sorry

end inequality_proof_l43_43017


namespace kevin_birth_year_l43_43415

theorem kevin_birth_year (year_first_amc: ℕ) (annual: ∀ n, year_first_amc + n = year_first_amc + n) (age_tenth_amc: ℕ) (year_tenth_amc: ℕ) (year_kevin_took_amc: ℕ) 
  (h_first_amc: year_first_amc = 1988) (h_age_tenth_amc: age_tenth_amc = 13) (h_tenth_amc: year_tenth_amc = year_first_amc + 9) (h_kevin_took_amc: year_kevin_took_amc = year_tenth_amc) :
  year_kevin_took_amc - age_tenth_amc = 1984 :=
by
  sorry

end kevin_birth_year_l43_43415


namespace combinedAgeIn5Years_l43_43049

variable (Amy Mark Emily : ℕ)

-- Conditions
def amyAge : ℕ := 15
def markAge : ℕ := amyAge + 7
def emilyAge : ℕ := 2 * amyAge

-- Proposition to be proved
theorem combinedAgeIn5Years :
  Amy = amyAge →
  Mark = markAge →
  Emily = emilyAge →
  (Amy + 5) + (Mark + 5) + (Emily + 5) = 82 :=
by
  intros hAmy hMark hEmily
  sorry

end combinedAgeIn5Years_l43_43049


namespace decimal_to_fraction_l43_43444

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43444


namespace sandra_money_left_l43_43404

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end sandra_money_left_l43_43404


namespace find_n_l43_43890

noncomputable def b_0 : ℝ := Real.cos (Real.pi / 18) ^ 2

noncomputable def b_n (n : ℕ) : ℝ :=
if n = 0 then b_0 else 4 * (b_n (n - 1)) * (1 - (b_n (n - 1)))

theorem find_n : ∀ n : ℕ, b_n n = b_0 → n = 24 := 
sorry

end find_n_l43_43890


namespace avg_age_new_students_l43_43794

theorem avg_age_new_students :
  ∀ (O A_old A_new_avg : ℕ) (A_new : ℕ),
    O = 12 ∧ A_old = 40 ∧ A_new_avg = (A_old - 4) ∧ A_new_avg = 36 →
    A_new * 12 = (24 * A_new_avg) - (O * A_old) →
    A_new = 32 :=
by
  intros O A_old A_new_avg A_new
  intro h
  rcases h with ⟨hO, hA_old, hA_new_avg, h36⟩
  sorry

end avg_age_new_students_l43_43794


namespace inequality_for_large_n_l43_43251

theorem inequality_for_large_n (n : ℕ) (hn : n > 1) : 
  (1 / Real.exp 1 - 1 / (n * Real.exp 1)) < (1 - 1 / n) ^ n ∧ (1 - 1 / n) ^ n < (1 / Real.exp 1 - 1 / (2 * n * Real.exp 1)) :=
sorry

end inequality_for_large_n_l43_43251


namespace problem_a_problem_b_problem_c_l43_43629

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l43_43629


namespace part1_part2_part3_l43_43877

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end part1_part2_part3_l43_43877


namespace product_xyz_l43_43202

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l43_43202


namespace conference_handshakes_l43_43684

-- Define the number of attendees at the conference
def attendees : ℕ := 10

-- Define the number of ways to choose 2 people from the attendees
-- This is equivalent to the combination formula C(10, 2)
def handshakes (n : ℕ) : ℕ := n.choose 2

-- Prove that the number of handshakes at the conference is 45
theorem conference_handshakes : handshakes attendees = 45 := by
  sorry

end conference_handshakes_l43_43684


namespace trigonometric_identity1_trigonometric_identity2_l43_43873

theorem trigonometric_identity1 (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin (Real.pi - θ) + Real.cos (θ - Real.pi)) / (Real.sin (θ + Real.pi) + Real.cos (θ + Real.pi)) = -1/3 :=
by
  sorry

theorem trigonometric_identity2 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4/5 :=
by
  sorry

end trigonometric_identity1_trigonometric_identity2_l43_43873


namespace complex_problem_l43_43513

theorem complex_problem (z : ℂ) (h : (i * z + z) = 2) : z = 1 - i :=
sorry

end complex_problem_l43_43513


namespace top_card_is_11_l43_43297

-- Define the initial configuration of cards
def initial_array : List (List Nat) := [
  [1, 2, 3, 4, 5, 6],
  [7, 8, 9, 10, 11, 12],
  [13, 14, 15, 16, 17, 18]
]

-- Perform the described sequence of folds
def fold1 (arr : List (List Nat)) : List (List Nat) := [
  [3, 4, 5, 6],
  [9, 10, 11, 12],
  [15, 16, 17, 18],
  [1, 2],
  [7, 8],
  [13, 14]
]

def fold2 (arr : List (List Nat)) : List (List Nat) := [
  [5, 6],
  [11, 12],
  [17, 18],
  [3, 4, 1, 2],
  [9, 10, 7, 8],
  [15, 16, 13, 14]
]

def fold3 (arr : List (List Nat)) : List (List Nat) := [
  [11, 12, 7, 8],
  [17, 18, 13, 14],
  [5, 6, 1, 2],
  [9, 10, 3, 4],
  [15, 16, 9, 10]
]

-- Define the final array after all the folds
def final_array := fold3 (fold2 (fold1 initial_array))

-- Statement to be proven
theorem top_card_is_11 : (final_array.head!.head!) = 11 := 
  by
    sorry -- Proof to be filled in

end top_card_is_11_l43_43297


namespace sum_series_eq_two_l43_43322

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l43_43322


namespace decimal_to_fraction_equivalence_l43_43483

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43483


namespace divide_triangle_in_half_l43_43174

def triangle_vertices : Prop :=
  let A := (0, 2)
  let B := (0, 0)
  let C := (10, 0)
  let base := 10
  let height := 2
  let total_area := (1 / 2) * base * height

  ∀ (a : ℝ),
  (1 / 2) * a * height = total_area / 2 → a = 5

theorem divide_triangle_in_half : triangle_vertices := 
  sorry

end divide_triangle_in_half_l43_43174


namespace intersection_of_sets_l43_43182

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2}) (hB : B = {1, 2, 3, 4}) :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l43_43182


namespace sum_from_1_to_15_fractions_l43_43142

def sum_of_fractions (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (k : ℚ) / 7

theorem sum_from_1_to_15_fractions :
  sum_of_fractions 15 = 120 / 7 :=
by
  sorry

end sum_from_1_to_15_fractions_l43_43142


namespace min_value_of_sum_of_reciprocals_l43_43013

theorem min_value_of_sum_of_reciprocals 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.log (1 / a + 1 / b) / Real.log 4 = Real.log (1 / Real.sqrt (a * b)) / Real.log 2) : 
  1 / a + 1 / b ≥ 4 := 
by 
  sorry

end min_value_of_sum_of_reciprocals_l43_43013


namespace quadratic_real_roots_l43_43562

theorem quadratic_real_roots (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 - 1 + m = 0 ∧ x2^2 + 2 * x2 - 1 + m = 0) ↔ m ≤ 2 :=
by
  sorry

end quadratic_real_roots_l43_43562


namespace custom_op_evaluation_l43_43580

def custom_op (x y : ℕ) : ℕ := x * y + x - y

theorem custom_op_evaluation : (custom_op 7 4) - (custom_op 4 7) = 6 := by
  sorry

end custom_op_evaluation_l43_43580


namespace smallest_prime_reversing_to_composite_l43_43713

theorem smallest_prime_reversing_to_composite (p : ℕ) :
  p = 23 ↔ (p < 100 ∧ p ≥ 10 ∧ Nat.Prime p ∧ 
  ∃ c, c < 100 ∧ c ≥ 10 ∧ ¬ Nat.Prime c ∧ c = (p % 10) * 10 + p / 10 ∧ (p / 10 = 2 ∨ p / 10 = 3)) :=
by
  sorry

end smallest_prime_reversing_to_composite_l43_43713


namespace lisa_balls_count_l43_43256

def stepNumber := 1729

def base7DigitsSum(x : Nat) : Nat :=
  x / 7 ^ 3 + (x % 343) / 7 ^ 2 + (x % 49) / 7 + x % 7

theorem lisa_balls_count (h1 : stepNumber = 1729) : base7DigitsSum stepNumber = 11 := by
  sorry

end lisa_balls_count_l43_43256


namespace painted_prisms_l43_43845

theorem painted_prisms (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 :=
by sorry

end painted_prisms_l43_43845


namespace xyz_product_neg4_l43_43205

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l43_43205


namespace sequence_is_constant_l43_43333

noncomputable def sequence_condition (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ 2 * m * n / (m ^ 2 + n ^ 2)

theorem sequence_is_constant (a : ℕ → ℝ) 
  (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 :=
by
  sorry

end sequence_is_constant_l43_43333


namespace problem_b_correct_l43_43342

theorem problem_b_correct (a b : ℝ) (h₁ : a < 0) (h₂ : 0 < b) (h₃ : b < 1) : (ab^2 > ab ∧ ab > a) :=
by
  sorry

end problem_b_correct_l43_43342


namespace cylinder_volume_tripled_and_radius_increased_l43_43746

theorem cylinder_volume_tripled_and_radius_increased :
  ∀ (r h : ℝ), let V := π * r^2 * h in
               let V_new := π * (2.5 * r)^2 * (3 * h) in
               V_new = 18.75 * V :=
by
  intros r h
  let V := π * r^2 * h
  let V_new := π * (2.5 * r)^2 * (3 * h)
  sorry

end cylinder_volume_tripled_and_radius_increased_l43_43746


namespace proof_problem_l43_43033

theorem proof_problem (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 + 2 * a * b = 64 :=
sorry

end proof_problem_l43_43033


namespace find_x_plus_inv_x_l43_43730

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l43_43730


namespace find_d_l43_43264

theorem find_d (c : ℝ) (d : ℝ) (h1 : c = 7)
  (h2 : (2, 6) ∈ { p : ℝ × ℝ | ∃ d, (p = (2, 6) ∨ p = (5, c) ∨ p = (d, 0)) ∧
           ∃ m, m = (0 - 6) / (d - 2) ∧ m = (c - 6) / (5 - 2) }) : 
  d = -16 :=
by
  sorry

end find_d_l43_43264


namespace zoo_visitors_l43_43611

theorem zoo_visitors (P : ℕ) (h : 3 * P = 3750) : P = 1250 :=
by 
  sorry

end zoo_visitors_l43_43611


namespace particle_speed_at_time_t_l43_43969

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + t + 1, 6 * t + 2)

theorem particle_speed_at_time_t (t : ℝ) :
  let dx := (position t).1
  let dy := (position t).2
  let vx := 6 * t + 1
  let vy := 6
  let speed := Real.sqrt (vx^2 + vy^2)
  speed = Real.sqrt (36 * t^2 + 12 * t + 37) :=
by
  sorry

end particle_speed_at_time_t_l43_43969


namespace calculate_n_l43_43705

theorem calculate_n (n : ℕ) : 3^n = 3 * 9^5 * 81^3 -> n = 23 :=
by
  -- Proof omitted
  sorry

end calculate_n_l43_43705


namespace exists_sum_of_divisibles_l43_43776

theorem exists_sum_of_divisibles : ∃ (a b: ℕ), a + b = 316 ∧ (13 ∣ a) ∧ (11 ∣ b) :=
by
  existsi 52
  existsi 264
  sorry

end exists_sum_of_divisibles_l43_43776


namespace problem1_problem2_l43_43138

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l43_43138


namespace lesser_number_is_14_l43_43808

theorem lesser_number_is_14 (x y : ℕ) (h₀ : x + y = 60) (h₁ : 4 * y - x = 10) : y = 14 :=
by 
  sorry

end lesser_number_is_14_l43_43808


namespace approximation_example1_approximation_example2_approximation_example3_l43_43289

theorem approximation_example1 (α β : ℝ) (hα : α = 0.0023) (hβ : β = 0.0057) :
  (1 + α) * (1 + β) = 1.008 := sorry

theorem approximation_example2 (α β : ℝ) (hα : α = 0.05) (hβ : β = -0.03) :
  (1 + α) * (10 + β) = 10.02 := sorry

theorem approximation_example3 (α β γ : ℝ) (hα : α = 0.03) (hβ : β = -0.01) (hγ : γ = -0.02) :
  (1 + α) * (1 + β) * (1 + γ) = 1 := sorry

end approximation_example1_approximation_example2_approximation_example3_l43_43289


namespace parent_payment_per_year_l43_43837

noncomputable def former_salary : ℕ := 45000
noncomputable def raise_percentage : ℕ := 20
noncomputable def number_of_kids : ℕ := 9

theorem parent_payment_per_year : 
  (former_salary + (raise_percentage * former_salary / 100)) / number_of_kids = 6000 := by
  sorry

end parent_payment_per_year_l43_43837


namespace jack_paid_20_l43_43761

-- Define the conditions
def numberOfSandwiches : Nat := 3
def costPerSandwich : Nat := 5
def changeReceived : Nat := 5

-- Define the total cost
def totalCost : Nat := numberOfSandwiches * costPerSandwich

-- Define the amount paid
def amountPaid : Nat := totalCost + changeReceived

-- Prove that the amount paid is 20
theorem jack_paid_20 : amountPaid = 20 := by
  -- You may assume the steps and calculations here, only providing the statement
  sorry

end jack_paid_20_l43_43761


namespace find_a7_l43_43340

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def Sn_for_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a7 (h_arith : arithmetic_sequence a)
  (h_sum_property : Sn_for_arithmetic_sequence a S)
  (h1 : a 2 + a 5 = 4)
  (h2 : S 7 = 21) :
  a 7 = 9 :=
sorry

end find_a7_l43_43340


namespace regular_polygon_exterior_angle_l43_43537

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end regular_polygon_exterior_angle_l43_43537


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43637

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43637


namespace sara_spent_on_hotdog_l43_43616

def total_cost_of_lunch: ℝ := 10.46
def cost_of_salad: ℝ := 5.10
def cost_of_hotdog: ℝ := total_cost_of_lunch - cost_of_salad

theorem sara_spent_on_hotdog :
  cost_of_hotdog = 5.36 := by
  sorry

end sara_spent_on_hotdog_l43_43616


namespace foreign_objects_total_sum_l43_43978

-- define the conditions
def dog_burrs : Nat := 12
def dog_ticks := 6 * dog_burrs
def dog_fleas := 3 * dog_ticks

def cat_burrs := 2 * dog_burrs
def cat_ticks := dog_ticks / 3
def cat_fleas := 4 * cat_ticks

-- calculate the total foreign objects
def total_dog := dog_burrs + dog_ticks + dog_fleas
def total_cat := cat_burrs + cat_ticks + cat_fleas

def total_objects := total_dog + total_cat

-- state the theorem
theorem foreign_objects_total_sum : total_objects = 444 := by
  sorry

end foreign_objects_total_sum_l43_43978


namespace kevin_food_expenditure_l43_43782

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l43_43782


namespace kibble_left_l43_43225

-- Define the initial amount of kibble
def initial_kibble := 3

-- Define the rate at which the cat eats kibble
def kibble_rate := 1 / 4

-- Define the time Kira was away
def time_away := 8

-- Define the amount of kibble eaten by the cat during the time away
def kibble_eaten := (time_away * kibble_rate)

-- Define the remaining kibble in the bowl
def remaining_kibble := initial_kibble - kibble_eaten

-- State and prove that the remaining amount of kibble is 1 pound
theorem kibble_left : remaining_kibble = 1 := by
  sorry

end kibble_left_l43_43225


namespace paint_red_faces_of_octahedral_die_l43_43132

theorem paint_red_faces_of_octahedral_die :
  (finset.univ.subset (finset.univ : finset (fin (8)))) ∧ -- Die has 8 faces
  (card ((finset.univ : finset (fin 8)).powerset.filter (λ s, s.card = 2)) = 28) ∧
  (card (finset.filter (λ (s : finset (fin 8)), set.sum s.val = 9) ((finset.univ : finset (fin 8)).powerset.filter (λ s, s.card = 2))) = 4) →
  24 :=
begin
  sorry,
end

end paint_red_faces_of_octahedral_die_l43_43132


namespace dishonest_dealer_uses_correct_weight_l43_43119

noncomputable def dishonest_dealer_weight (profit_percent : ℝ) (true_weight : ℝ) : ℝ :=
  true_weight - (profit_percent / 100 * true_weight)

theorem dishonest_dealer_uses_correct_weight :
  dishonest_dealer_weight 11.607142857142861 1 = 0.8839285714285714 :=
by
  -- We skip the proof here
  sorry

end dishonest_dealer_uses_correct_weight_l43_43119


namespace valid_outfits_count_l43_43031

noncomputable def number_of_valid_outfits (shirt_count: ℕ) (pant_colors: List String) (hat_count: ℕ) : ℕ :=
  let total_combinations := shirt_count * (pant_colors.length) * hat_count
  let matching_outfits := List.length (List.filter (λ c => c ∈ pant_colors) ["tan", "black", "blue", "gray"])
  total_combinations - matching_outfits

theorem valid_outfits_count :
    number_of_valid_outfits 8 ["tan", "black", "blue", "gray"] 8 = 252 := by
  sorry

end valid_outfits_count_l43_43031


namespace tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l43_43498

-- Definition for Part (a)
theorem tetrahedron_parallelepiped_areas 
  (S1 S2 S3 S4 P1 P2 P3 : ℝ)
  (h1 : true)
  (h2 : true) :
  S1^2 + S2^2 + S3^2 + S4^2 = P1^2 + P2^2 + P3^2 := 
sorry

-- Definition for Part (b)
theorem tetrahedron_heights_distances 
  (h1 h2 h3 h4 d1 d2 d3 : ℝ)
  (h : true) :
  (1/(h1^2)) + (1/(h2^2)) + (1/(h3^2)) + (1/(h4^2)) = (1/(d1^2)) + (1/(d2^2)) + (1/(d3^2)) := 
sorry

end tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l43_43498


namespace neg_real_root_condition_l43_43803

theorem neg_real_root_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (0 < a ∧ a ≤ 1) ∨ (a < 0) :=
by
  sorry

end neg_real_root_condition_l43_43803


namespace dietitian_lunch_fraction_l43_43308

theorem dietitian_lunch_fraction
  (total_calories : ℕ)
  (recommended_calories : ℕ)
  (extra_calories : ℕ)
  (h1 : total_calories = 40)
  (h2 : recommended_calories = 25)
  (h3 : extra_calories = 5)
  : (recommended_calories + extra_calories) / total_calories = 3 / 4 :=
by
  sorry

end dietitian_lunch_fraction_l43_43308


namespace solve_inequality_1_range_of_m_l43_43023

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 3) + m

theorem solve_inequality_1 : {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} := sorry

theorem range_of_m (m : ℝ) (h : m > 4) : ∃ x : ℝ, f x < g x m := sorry

end solve_inequality_1_range_of_m_l43_43023


namespace necessary_but_not_sufficient_condition_l43_43874

theorem necessary_but_not_sufficient_condition {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  ((a + b > 1) ↔ (ab > 1)) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l43_43874


namespace find_b_l43_43167

theorem find_b (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : b = 4 :=
  sorry

end find_b_l43_43167


namespace roofing_cost_per_foot_l43_43916

theorem roofing_cost_per_foot:
  ∀ (total_feet needed_feet free_feet : ℕ) (total_cost : ℕ),
  needed_feet = 300 →
  free_feet = 250 →
  total_cost = 400 →
  needed_feet - free_feet = 50 →
  total_cost / (needed_feet - free_feet) = 8 :=
by sorry

end roofing_cost_per_foot_l43_43916


namespace find_three_digit_number_l43_43691

theorem find_three_digit_number (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (h_sum : 122 * a + 212 * b + 221 * c = 2003) :
  100 * a + 10 * b + c = 345 :=
by
  sorry

end find_three_digit_number_l43_43691


namespace proof_of_inequality_l43_43213

theorem proof_of_inequality (a : ℝ) (h : (∃ x : ℝ, x - 2 * a + 4 = 0 ∧ x < 0)) :
  (a - 3) * (a - 4) > 0 :=
by
  sorry

end proof_of_inequality_l43_43213


namespace find_pool_length_l43_43410

noncomputable def pool_length : ℝ :=
  let drain_rate := 60 -- cubic feet per minute
  let width := 40 -- feet
  let depth := 10 -- feet
  let capacity_percent := 0.80
  let drain_time := 800 -- minutes
  let drained_volume := drain_rate * drain_time -- cubic feet
  let full_capacity := drained_volume / capacity_percent -- cubic feet
  let length := full_capacity / (width * depth) -- feet
  length

theorem find_pool_length : pool_length = 150 := by
  sorry

end find_pool_length_l43_43410


namespace decimal_to_fraction_l43_43436

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43436


namespace number_of_students_surveyed_l43_43757

noncomputable def M : ℕ := 60
noncomputable def N : ℕ := 90
noncomputable def B : ℕ := M / 3

theorem number_of_students_surveyed : M + B + N = 170 := by
  rw [M, N, B]
  norm_num
  sorry

end number_of_students_surveyed_l43_43757


namespace value_of_y_minus_x_l43_43581

theorem value_of_y_minus_x (x y : ℝ) (h1 : abs (x + 1) = 3) (h2 : abs y = 5) (h3 : -y / x > 0) :
  y - x = -7 ∨ y - x = 9 :=
sorry

end value_of_y_minus_x_l43_43581


namespace solve_inequalities_l43_43254

theorem solve_inequalities (x : ℤ) :
  (1 ≤ x ∧ x < 3) ↔ 
  ((↑x - 1) / 2 < (↑x : ℝ) / 3 ∧ 2 * (↑x : ℝ) - 5 ≤ 3 * (↑x : ℝ) - 6) :=
by
  sorry

end solve_inequalities_l43_43254


namespace tom_hockey_games_l43_43946

theorem tom_hockey_games (g_this_year g_last_year : ℕ) 
  (h1 : g_this_year = 4)
  (h2 : g_last_year = 9) 
  : g_this_year + g_last_year = 13 := 
by
  sorry

end tom_hockey_games_l43_43946


namespace combined_weight_difference_l43_43911

def john_weight : ℕ := 81
def roy_weight : ℕ := 79
def derek_weight : ℕ := 91
def samantha_weight : ℕ := 72

theorem combined_weight_difference :
  derek_weight - samantha_weight = 19 :=
by
  sorry

end combined_weight_difference_l43_43911


namespace decimal_to_fraction_l43_43486

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43486


namespace scientific_notation_of_20000_l43_43258

def number : ℕ := 20000

theorem scientific_notation_of_20000 : number = 2 * 10 ^ 4 :=
by
  sorry

end scientific_notation_of_20000_l43_43258


namespace decimal_to_fraction_l43_43490

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43490


namespace max_value_l43_43028

open Real

/-- Given vectors a, b, and c, and real numbers m and n such that m * a + n * b = c,
prove that the maximum value for (m - 3)^2 + n^2 is 16. --/
theorem max_value
  (α : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (m n : ℝ)
  (ha : a = (1, 1))
  (hb : b = (1, -1))
  (hc : c = (sqrt 2 * cos α, sqrt 2 * sin α))
  (h : m * a.1 + n * b.1 = c.1 ∧ m * a.2 + n * b.2 = c.2) :
  (m - 3)^2 + n^2 ≤ 16 :=
by
  sorry

end max_value_l43_43028


namespace pairs_xy_solution_sum_l43_43554

theorem pairs_xy_solution_sum :
  ∃ (x y : ℝ) (a b c d : ℕ), 
    x + y = 5 ∧ 2 * x * y = 5 ∧ 
    (x = (5 + Real.sqrt 15) / 2 ∨ x = (5 - Real.sqrt 15) / 2) ∧ 
    a = 5 ∧ b = 1 ∧ c = 15 ∧ d = 2 ∧ a + b + c + d = 23 :=
by
  sorry

end pairs_xy_solution_sum_l43_43554


namespace geometric_series_common_ratio_l43_43849

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end geometric_series_common_ratio_l43_43849


namespace chemist_solution_l43_43835

theorem chemist_solution (x : ℝ) (h1 : ∃ x, 0 < x) 
  (h2 : x + 1 > 1) : 0.60 * x = 0.10 * (x + 1) → x = 0.2 := by
  sorry

end chemist_solution_l43_43835


namespace find_a_l43_43720

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : (6 * a * (-1) + 6) = 4) : 
  a = 10 / 3 :=
by {
  sorry
}

end find_a_l43_43720


namespace initial_profit_percentage_l43_43298

theorem initial_profit_percentage
  (CP : ℝ)
  (h1 : CP = 2400)
  (h2 : ∀ SP : ℝ, 15 / 100 * CP = 120 + SP) :
  ∃ P : ℝ, (P / 100) * CP = 10 :=
by
  sorry

end initial_profit_percentage_l43_43298


namespace dimensions_of_triangle_from_square_l43_43125

theorem dimensions_of_triangle_from_square :
  ∀ (a : ℝ) (triangle : ℝ × ℝ × ℝ), 
    a = 10 →
    triangle = (a, a, a * Real.sqrt 2) →
    triangle = (10, 10, 10 * Real.sqrt 2) :=
by
  intros a triangle a_eq triangle_eq
  -- Proof
  sorry

end dimensions_of_triangle_from_square_l43_43125


namespace deceased_member_income_l43_43898

theorem deceased_member_income
  (initial_income_4_members : ℕ)
  (initial_members : ℕ := 4)
  (initial_average_income : ℕ := 840)
  (final_income_3_members : ℕ)
  (remaining_members : ℕ := 3)
  (final_average_income : ℕ := 650)
  (total_income_initial : initial_income_4_members = initial_average_income * initial_members)
  (total_income_final : final_income_3_members = final_average_income * remaining_members)
  (income_deceased : ℕ) :
  income_deceased = initial_income_4_members - final_income_3_members :=
by
  -- sorry indicates this part of the proof is left as an exercise
  sorry

end deceased_member_income_l43_43898


namespace remainder_sum_first_150_div_11250_l43_43097

theorem remainder_sum_first_150_div_11250 : 
  let S := 150 * 151 / 2 
  in S % 11250 = 75 := 
by
  let S := 11325
  have hSum : S = 11325 := by rfl
  show S % 11250 = 75
  sorry

end remainder_sum_first_150_div_11250_l43_43097


namespace outfit_choices_l43_43344

theorem outfit_choices (tops pants : ℕ) (TopsCount : tops = 4) (PantsCount : pants = 3) :
  tops * pants = 12 := by
  sorry

end outfit_choices_l43_43344


namespace travel_time_l43_43751

noncomputable def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

theorem travel_time
  (speed_kmh : ℝ)
  (distance_m : ℝ) :
  speed_kmh = 63 →
  distance_m = 437.535 →
  (distance_m / convert_kmh_to_mps speed_kmh) = 25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end travel_time_l43_43751


namespace slower_train_speed_l43_43665

-- Define the given conditions
def speed_faster_train : ℝ := 50  -- km/h
def length_faster_train : ℝ := 75.006  -- meters
def passing_time : ℝ := 15  -- seconds

-- Conversion factor
def mps_to_kmph : ℝ := 3.6

-- Define the problem to be proved
theorem slower_train_speed : 
  ∃ speed_slower_train : ℝ, 
    speed_slower_train = speed_faster_train - (75.006 / 15) * mps_to_kmph := 
  by
    exists 31.99856
    sorry

end slower_train_speed_l43_43665


namespace win_sector_area_l43_43303

-- Given Conditions
def radius := 12  -- radius of the circle in cm
def probability_of_winning := 1 / 3  -- probability of winning on one spin

-- Calculate the total area of the circle
def total_area_circle : ℝ := real.pi * radius^2

-- Calculate the area of the WIN sector
def area_of_win_sector : ℝ := probability_of_winning * total_area_circle

-- Proof Statement
theorem win_sector_area : area_of_win_sector = 48 * real.pi :=
by
  -- proof steps will go here
  sorry

end win_sector_area_l43_43303


namespace calculate_expression_l43_43857

theorem calculate_expression :
  (-1: ℤ) ^ 53 + 2 ^ (4 ^ 4 + 3 ^ 3 - 5 ^ 2) = -1 + 2 ^ 258 := 
by
  sorry

end calculate_expression_l43_43857


namespace estimate_time_pm_l43_43904

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end estimate_time_pm_l43_43904


namespace factorial_div_eq_l43_43144
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l43_43144


namespace find_x_plus_inv_x_l43_43729

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l43_43729


namespace ceil_and_floor_difference_l43_43179

theorem ceil_and_floor_difference (x : ℝ) (ε : ℝ) 
  (h_cond : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) (h_eps : 0 < ε ∧ ε < 1) :
  ⌈x + ε⌉ - (x + ε) = 1 - ε :=
sorry

end ceil_and_floor_difference_l43_43179


namespace decimal_to_fraction_l43_43491

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43491


namespace balloon_permutations_l43_43989

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l43_43989


namespace sale_in_third_month_l43_43967

def sales_in_months (m1 m2 m3 m4 m5 m6 : Int) : Prop :=
  m1 = 5124 ∧
  m2 = 5366 ∧
  m4 = 6124 ∧
  m6 = 4579 ∧
  (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 5400

theorem sale_in_third_month (m5 : Int) :
  (∃ m3 : Int, sales_in_months 5124 5366 m3 6124 m5 4579 → m3 = 11207) :=
sorry

end sale_in_third_month_l43_43967


namespace points_five_units_away_from_neg_one_l43_43424

theorem points_five_units_away_from_neg_one (x : ℝ) :
  |x + 1| = 5 ↔ x = 4 ∨ x = -6 :=
by
  sorry

end points_five_units_away_from_neg_one_l43_43424


namespace yogurt_cost_l43_43275

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l43_43275


namespace max_knights_l43_43242

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end max_knights_l43_43242


namespace f_neg_example_l43_43381

-- Definitions and conditions given in the problem
def f : ℚ → ℚ := sorry

axiom condition1 (a b : ℚ) (ha : a > 0) (hb : b > 0) : f (a * b) = f a + f b
axiom condition2 (p : ℕ) (hp : nat.prime p) : f (p) = p

-- This is the statement that corresponds to the problem's question and conclusion.
theorem f_neg_example : f (25 / 11) < 0 :=
sorry

end f_neg_example_l43_43381


namespace largest_multiple_of_seven_smaller_than_neg_85_l43_43095

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l43_43095


namespace infinite_series_converges_to_3_l43_43320

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l43_43320


namespace bears_in_stock_initially_l43_43315

theorem bears_in_stock_initially 
  (shipment_bears : ℕ) (shelf_bears : ℕ) (shelves_used : ℕ)
  (total_bears_shelved : shipment_bears + shelf_bears * shelves_used = 24) : 
  (24 - shipment_bears = 6) :=
by
  exact sorry

end bears_in_stock_initially_l43_43315


namespace solve_abs_eq_l43_43164

theorem solve_abs_eq (x : ℝ) : |2*x - 6| = 3*x + 6 ↔ x = 0 :=
by 
  sorry

end solve_abs_eq_l43_43164


namespace cos_four_alpha_sub_9pi_over_2_l43_43716

open Real

theorem cos_four_alpha_sub_9pi_over_2 (α : ℝ) 
  (cond : 4.53 * (1 + cos (2 * α - 2 * π) + cos (4 * α + 2 * π) - cos (6 * α - π)) /
                  (cos (2 * π - 2 * α) + 2 * cos (2 * α + π) ^ 2 - 1) = 2 * cos (2 * α)) :
  cos (4 * α - 9 * π / 2) = cos (4 * α - π / 2) :=
by sorry

end cos_four_alpha_sub_9pi_over_2_l43_43716


namespace problem_a_problem_c_problem_d_l43_43646

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l43_43646


namespace sum_digits_2_pow_2010_5_pow_2012_7_l43_43949

theorem sum_digits_2_pow_2010_5_pow_2012_7 :
  digit_sum (2^2010 * 5^2012 * 7) = 13 :=
by
  sorry

end sum_digits_2_pow_2010_5_pow_2012_7_l43_43949


namespace part_a_l43_43294

theorem part_a {d m b : ℕ} (h_d : d = 41) (h_m : m = 28) (h_b : b = 15) :
    d - b + m - b + b = 54 :=
  by sorry

end part_a_l43_43294


namespace loan_amount_l43_43775

theorem loan_amount (R T SI : ℕ) (hR : R = 7) (hT : T = 7) (hSI : SI = 735) : 
  ∃ P : ℕ, P = 1500 := 
by 
  sorry

end loan_amount_l43_43775


namespace expected_potato_yield_l43_43394

-- Definitions based on the conditions
def steps_length : ℕ := 3
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def yield_rate : ℚ := 3 / 4

-- Calculate the dimensions in feet
def garden_length_feet : ℕ := garden_length_steps * steps_length
def garden_width_feet : ℕ := garden_width_steps * steps_length

-- Calculate the area in square feet
def garden_area_feet : ℕ := garden_length_feet * garden_width_feet

-- Calculate the expected yield in pounds
def expected_yield_pounds : ℚ := garden_area_feet * yield_rate

-- The theorem to prove the expected yield
theorem expected_potato_yield :
  expected_yield_pounds = 3037.5 := by
  sorry  -- Proof is omitted as per the instructions.

end expected_potato_yield_l43_43394


namespace equivalent_fraction_l43_43927

theorem equivalent_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = -3) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 38 / 13 :=
by
  sorry

end equivalent_fraction_l43_43927


namespace factorial_quotient_l43_43154

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l43_43154


namespace james_writes_pages_per_hour_l43_43222

theorem james_writes_pages_per_hour (hours_per_night : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) (total_hours : ℕ) :
  hours_per_night = 3 → 
  days_per_week = 7 → 
  weeks = 7 → 
  total_pages = 735 → 
  total_hours = 147 → 
  total_hours = hours_per_night * days_per_week * weeks → 
  total_pages / total_hours = 5 :=
by sorry

end james_writes_pages_per_hour_l43_43222


namespace bullet_train_speed_is_70kmph_l43_43300

noncomputable def bullet_train_speed (train_length time_man  : ℚ) (man_speed_kmph : ℕ) : ℚ :=
  let man_speed_ms : ℚ := man_speed_kmph * 1000 / 3600
  let relative_speed : ℚ := train_length / time_man
  let train_speed_ms : ℚ := relative_speed - man_speed_ms
  train_speed_ms * 3600 / 1000

theorem bullet_train_speed_is_70kmph :
  bullet_train_speed 160 7.384615384615384 8 = 70 :=
by {
  -- Proof is omitted
  sorry
}

end bullet_train_speed_is_70kmph_l43_43300


namespace smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l43_43948

def is_not_prime (n : ℕ) := ¬ Prime n
def is_not_square (n : ℕ) := ∀ m : ℕ, m * m ≠ n
def no_prime_factors_less_than (n k : ℕ) := ∀ p : ℕ, Prime p → p < k → ¬ p ∣ n
def smallest_integer_prop (n : ℕ) := is_not_prime n ∧ is_not_square n ∧ no_prime_factors_less_than n 60

theorem smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60 : ∃ n : ℕ, smallest_integer_prop n ∧ n = 4087 :=
by
  sorry

end smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l43_43948


namespace sum_of_squares_of_distances_is_constant_l43_43369

variable {r1 r2 : ℝ}
variable {x y : ℝ}

theorem sum_of_squares_of_distances_is_constant
  (h1 : r1 < r2)
  (h2 : x^2 + y^2 = r1^2) :
  let PA := (x - r2)^2 + y^2
  let PB := (x + r2)^2 + y^2
  PA + PB = 2 * r1^2 + 2 * r2^2 :=
by
  sorry

end sum_of_squares_of_distances_is_constant_l43_43369


namespace price_after_discount_eq_cost_price_l43_43071

theorem price_after_discount_eq_cost_price (m : Real) :
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  price_after_discount = m :=
by
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  sorry

end price_after_discount_eq_cost_price_l43_43071


namespace product_xyz_equals_zero_l43_43197

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l43_43197


namespace find_p_over_q_l43_43715

variables (x y p q : ℚ)

theorem find_p_over_q (h1 : (7 * x + 6 * y) / (x - 2 * y) = 27)
                      (h2 : x / (2 * y) = p / q) :
                      p / q = 3 / 2 :=
sorry

end find_p_over_q_l43_43715


namespace decimal_to_fraction_equivalence_l43_43480

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43480


namespace greatest_value_of_x_is_20_l43_43070

noncomputable def greatest_multiple_of_4 (x : ℕ) : Prop :=
  (x % 4 = 0 ∧ x^2 < 500 ∧ ∀ y : ℕ, (y % 4 = 0 ∧ y^2 < 500) → y ≤ x)

theorem greatest_value_of_x_is_20 : greatest_multiple_of_4 20 :=
  by 
  sorry

end greatest_value_of_x_is_20_l43_43070


namespace son_work_time_l43_43292

theorem son_work_time (M S : ℝ) 
  (hM : M = 1 / 4)
  (hCombined : M + S = 1 / 3) : 
  S = 1 / 12 :=
by
  sorry

end son_work_time_l43_43292


namespace matching_trio_probability_l43_43688

open Classical

def cards := 52
def removed_cards := 3
def remaining_cards := 49
def total_ways_to_choose_3_cards := Nat.choose remaining_cards 3
def number_of_successful_trios := (11 * Nat.choose 4 3) + (Nat.choose 3 3)

theorem matching_trio_probability 
  (m n : ℕ) 
  (rel_prime_m_n : Nat.gcd m n = 1)
  (frac_m_n : m / n = number_of_successful_trios / total_ways_to_choose_3_cards)
  : m + n = 18469 :=
by
  sorry

end matching_trio_probability_l43_43688


namespace arith_seq_s14_gt_0_l43_43018

variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of an arithmetic sequence
variable {a : ℕ → ℝ} -- a_n is the nth term of the arithmetic sequence
variable {d : ℝ} -- d is the common difference of the arithmetic sequence

-- Conditions
variable (a_7_lt_0 : a 7 < 0)
variable (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0)

-- Assertion
theorem arith_seq_s14_gt_0 (a_7_lt_0 : a 7 < 0) (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0) : S 14 > 0 := by
  sorry

end arith_seq_s14_gt_0_l43_43018


namespace smaller_number_l43_43429

theorem smaller_number (x y : ℝ) (h1 : x + y = 16) (h2 : x - y = 4) (h3 : x * y = 60) : y = 6 :=
sorry

end smaller_number_l43_43429


namespace maximum_value_A_l43_43500

theorem maximum_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  ( (x^2 - y) * real.sqrt (y + x^3 - x * y) + (y^2 - x) * real.sqrt (x + y^3 - x * y) + 1 ) /
  ( (x - y)^2 + 1 ) ≤ 1 :=
sorry

end maximum_value_A_l43_43500


namespace tan_alpha_over_tan_beta_l43_43723

theorem tan_alpha_over_tan_beta (α β : ℝ) (h1 : Real.sin (α + β) = 2 / 3) (h2 : Real.sin (α - β) = 1 / 3) :
  (Real.tan α / Real.tan β = 3) :=
sorry

end tan_alpha_over_tan_beta_l43_43723


namespace age_difference_l43_43809

variable (A B C : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 13) : (A + B) - (B + C) = 13 := by
  sorry

end age_difference_l43_43809


namespace find_k_percent_l43_43010

theorem find_k_percent (k : ℝ) : 0.2 * 30 = 6 → (k / 100) * 25 = 6 → k = 24 := by
  intros h1 h2
  sorry

end find_k_percent_l43_43010


namespace decimal_to_fraction_equivalence_l43_43484

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43484


namespace car_speed_in_second_hour_l43_43427

theorem car_speed_in_second_hour (x : ℕ) : 84 = (98 + x) / 2 → x = 70 := 
sorry

end car_speed_in_second_hour_l43_43427


namespace find_angle_D_l43_43341

theorem find_angle_D 
  (A B C D : ℝ) 
  (h1 : A + B = 180) 
  (h2 : C = D + 10) 
  (h3 : A = 50)
  : D = 20 := by
  sorry

end find_angle_D_l43_43341


namespace geometric_sequence_iff_q_neg_one_l43_43882

theorem geometric_sequence_iff_q_neg_one {p q : ℝ} (h1 : p ≠ 0) (h2 : p ≠ 1)
  (S : ℕ → ℝ) (hS : ∀ n, S n = p^n + q) :
  (∃ (a : ℕ → ℝ), (∀ n, a (n+1) = (p - 1) * p^n) ∧ (∀ n, a (n+1) = S (n+1) - S n) ∧
                    (∀ n, a (n+1) / a n = p)) ↔ q = -1 :=
sorry

end geometric_sequence_iff_q_neg_one_l43_43882


namespace decimal_to_fraction_l43_43476

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43476


namespace circles_tangent_internally_l43_43331

noncomputable def circle_center_radius (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let h := -a / 2
  let k := -b / 2
  let r := (h * h + k * k - c).sqrt
  (h, k, r)

theorem circles_tangent_internally :
  ∀ (x y : ℝ), 
    (x ^ 2 + y ^ 2 + 2 * x + 4 * y + 1 = 0) → 
    (x ^ 2 + y ^ 2 - 4 * x + 4 * y - 17 = 0) → 
    (∃ h1 k1 r1 h2 k2 r2, 
      (h1, k1, r1) = circle_center_radius 2 4 1 ∧ 
      (h2, k2, r2) = circle_center_radius (-4) 4 (-17) ∧
      (real.sqrt ((h1 - h2) ^ 2 + (k1 - k2) ^ 2) = (r1 - r2).abs)) := 
sorry

end circles_tangent_internally_l43_43331


namespace basketball_competition_l43_43000

theorem basketball_competition:
  (∃ x : ℕ, (0 ≤ x) ∧ (x ≤ 12) ∧ (3 * x - (12 - x) ≥ 28)) := by
  sorry

end basketball_competition_l43_43000


namespace yogurt_production_cost_l43_43278

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l43_43278


namespace f_negative_l43_43566

-- Let f be a function defined on the real numbers
variable (f : ℝ → ℝ)

-- Conditions: f is odd and given form for non-negative x
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2 * x

theorem f_negative (x : ℝ) (hx : x < 0) : f x = -x^2 + 2 * x := by
  sorry

end f_negative_l43_43566


namespace ellipse_foci_distance_2sqrt21_l43_43540

noncomputable def ellipse_foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance_2sqrt21 :
  let center : ℝ × ℝ := (5, 2)
  let a := 5
  let b := 2
  ellipse_foci_distance a b = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_2sqrt21_l43_43540


namespace pay_docked_per_lateness_l43_43356

variable (hourly_rate : ℤ) (work_hours : ℤ) (times_late : ℕ) (actual_pay : ℤ) 

theorem pay_docked_per_lateness (h_rate : hourly_rate = 30) 
                                (w_hours : work_hours = 18) 
                                (t_late : times_late = 3) 
                                (a_pay : actual_pay = 525) :
                                (hourly_rate * work_hours - actual_pay) / times_late = 5 :=
by
  sorry

end pay_docked_per_lateness_l43_43356


namespace factorial_division_identity_l43_43155

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l43_43155


namespace largest_multiple_of_7_less_than_neg85_l43_43090

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l43_43090


namespace shanna_initial_tomato_plants_l43_43924

theorem shanna_initial_tomato_plants (T : ℕ) 
  (h1 : 56 = (T / 2) * 7 + 2 * 7 + 3 * 7) : 
  T = 6 :=
by sorry

end shanna_initial_tomato_plants_l43_43924


namespace Petya_running_time_l43_43544

theorem Petya_running_time (D V : ℝ) 
  (hV_pos : 0 < V) (hD_pos : 0 < D):
  let T := D / V in
  let V1 := 1.25 * V in
  let V2 := 0.8 * V in
  let T1 := (D / 2) / V1 in
  let T2 := (D / 2) / V2 in
  let T_actual := T1 + T2 in
  T_actual > T :=
by
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  let T_actual := T1 + T2
  have : T_actual = (2 * D) / (5 * V) + (5 * D) / (8 * V) := by 
  sorry
  have : T_actual = 41 * D / (40 * V) := by 
  sorry
  have : T = D / V := by 
  sorry
  show 41 * D / (40 * V) > D / V := by 
  sorry

end Petya_running_time_l43_43544


namespace problem1_problem2_problem3_problem4_problem5_problem6_l43_43653

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l43_43653


namespace tomatoes_ready_for_sale_l43_43390

-- Define all conditions
def initial_shipment := 1000 -- kg of tomatoes on Friday
def sold_on_saturday := 300 -- kg of tomatoes sold on Saturday
def rotten_on_sunday := 200 -- kg of tomatoes rotted on Sunday
def additional_shipment := 2 * initial_shipment -- kg of tomatoes arrived on Monday

-- Define the final calculation to prove
theorem tomatoes_ready_for_sale : 
  initial_shipment - sold_on_saturday - rotten_on_sunday + additional_shipment = 2500 := 
by
  sorry

end tomatoes_ready_for_sale_l43_43390


namespace problem_l43_43188

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l43_43188


namespace graded_worksheets_before_l43_43130

-- Definitions based on conditions
def initial_worksheets : ℕ := 34
def additional_worksheets : ℕ := 36
def total_worksheets : ℕ := 63

-- Equivalent proof problem statement
theorem graded_worksheets_before (x : ℕ) (h₁ : initial_worksheets - x + additional_worksheets = total_worksheets) : x = 7 :=
by sorry

end graded_worksheets_before_l43_43130


namespace kevin_food_spending_l43_43783

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l43_43783


namespace decimal_to_fraction_l43_43459

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43459


namespace infinite_sum_problem_l43_43323

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l43_43323


namespace max_african_team_wins_max_l43_43219

-- Assume there are n African teams and (n + 9) European teams.
-- Each pair of teams plays exactly once.
-- European teams won nine times as many matches as African teams.
-- Prove that the maximum number of matches that a single African team might have won is 11.

theorem max_african_team_wins_max (n : ℕ) (k : ℕ) (n_african_wins : ℕ) (n_european_wins : ℕ)
  (h1 : n_african_wins = (n * (n - 1)) / 2) 
  (h2 : n_european_wins = ((n + 9) * (n + 8)) / 2 + k)
  (h3 : n_european_wins = 9 * (n_african_wins + (n * (n + 9) - k))) :
  ∃ max_wins, max_wins = 11 := by
  sorry

end max_african_team_wins_max_l43_43219


namespace sqrt_product_simplifies_l43_43703

theorem sqrt_product_simplifies (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := 
by
  sorry

end sqrt_product_simplifies_l43_43703


namespace periodic_points_1989_l43_43502

section

variables {m : ℕ} (hm : m > 1)
def unit_circle := {z : ℂ // complex.abs z = 1}
def f (z : unit_circle) : unit_circle := ⟨z.val^m, by { rw [complex.abs_pow, z.property, one_pow], exact one_eq_one }⟩
def f_iter (k : ℕ) (z : unit_circle) : unit_circle := nat.iterate f k z

def is_periodic_point (n : ℕ) (z : unit_circle) : Prop :=
(f_iter m hm n z = z) ∧ ∀ i < n, f_iter m hm i z ≠ z

def count_periodic_points (n : ℕ) : ℕ :=
(fix : ℕ := (m ^ n - 1)) - (fix 117) - (fix 153) - (fix 663) + (fix 51) + (fix 39) + (fix 9) - (fix 3)

theorem periodic_points_1989 :
  count_periodic_points 1989 m = m ^ 1989 - m ^ 117 - m ^ 153 - m ^ 663 + m ^ 51 + m ^ 39 + m ^ 9 - m ^ 3 := 
sorry

end

end periodic_points_1989_l43_43502


namespace factorial_div_eq_l43_43150

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l43_43150


namespace tim_took_rulers_l43_43082

theorem tim_took_rulers (initial_rulers : ℕ) (remaining_rulers : ℕ) (rulers_taken : ℕ) :
  initial_rulers = 46 → remaining_rulers = 21 → rulers_taken = initial_rulers - remaining_rulers → rulers_taken = 25 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end tim_took_rulers_l43_43082


namespace find_b_l43_43887

-- Define the conditions as given in the problem
def poly1 (x : ℝ) : ℝ := x^2 - 2 * x - 1
def poly2 (x a b : ℝ) : ℝ := a * x^3 + b * x^2 + 1

-- Define the problem statement using these conditions
theorem find_b (a b : ℤ) (h : ∀ x, poly1 x = 0 → poly2 x a b = 0) : b = -3 :=
sorry

end find_b_l43_43887


namespace ratio_of_sums_l43_43425

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def square_of_sum (n : ℕ) : ℚ :=
  ((n * (n + 1)) / 2) ^ 2

theorem ratio_of_sums (n : ℕ) (h : n = 25) :
  sum_of_squares n / square_of_sum n = 1 / 19 :=
by
  have hn : n = 25 := h
  rw [hn]
  dsimp [sum_of_squares, square_of_sum]
  have : (25 * (25 + 1) * (2 * 25 + 1)) / 6 = 5525 := by norm_num
  have : ((25 * (25 + 1)) / 2) ^ 2 = 105625 := by norm_num
  norm_num
  sorry

end ratio_of_sums_l43_43425


namespace azalea_profit_l43_43846

def num_sheep : Nat := 200
def wool_per_sheep : Nat := 10
def price_per_pound : Nat := 20
def shearer_cost : Nat := 2000

theorem azalea_profit : 
  (num_sheep * wool_per_sheep * price_per_pound) - shearer_cost = 38000 := 
by
  sorry

end azalea_profit_l43_43846


namespace value_of_b_minus_d_squared_l43_43208

theorem value_of_b_minus_d_squared (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := 
by 
  sorry

end value_of_b_minus_d_squared_l43_43208


namespace problem_solution_l43_43379

theorem problem_solution (a b c : ℤ)
  (h1 : ∀ x : ℤ, |x| ≠ |a|)
  (h2 : ∀ x : ℤ, x^2 ≠ b^2)
  (h3 : ∀ x : ℤ, x * c ≤ 1):
  a + b + c = 0 :=
by sorry

end problem_solution_l43_43379


namespace symmetric_point_origin_l43_43375

theorem symmetric_point_origin (m : ℤ) : 
  (symmetry_condition : (3, m - 2) = (-(-3), -5)) → m = -3 :=
by
  sorry

end symmetric_point_origin_l43_43375


namespace number_of_boys_l43_43128

theorem number_of_boys (x : ℕ) (y : ℕ) (h1 : x + y = 8) (h2 : y > x) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end number_of_boys_l43_43128


namespace decimal_to_fraction_l43_43448

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43448


namespace find_2a_plus_b_l43_43768

theorem find_2a_plus_b (a b : ℝ) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (h3 : 5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2)
  (h4 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 3) :
  2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l43_43768


namespace geometric_sequence_problem_l43_43759

variable {α : Type*} [LinearOrder α] [Field α]

def is_geometric_sequence (a : ℕ → α) :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

theorem geometric_sequence_problem (a : ℕ → α) (r : α) (h1 : a 1 = 1) (h2 : is_geometric_sequence a) (h3 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 7 = 4 :=
by
  sorry

end geometric_sequence_problem_l43_43759


namespace taxi_trip_distance_l43_43908

theorem taxi_trip_distance
  (initial_fee : ℝ)
  (per_segment_charge : ℝ)
  (segment_distance : ℝ)
  (total_charge : ℝ)
  (segments_traveled : ℝ)
  (total_miles : ℝ) :
  initial_fee = 2.25 →
  per_segment_charge = 0.3 →
  segment_distance = 2/5 →
  total_charge = 4.95 →
  total_miles = segments_traveled * segment_distance →
  segments_traveled = (total_charge - initial_fee) / per_segment_charge →
  total_miles = 3.6 :=
by
  intros h_initial_fee h_per_segment_charge h_segment_distance h_total_charge h_total_miles h_segments_traveled
  sorry

end taxi_trip_distance_l43_43908


namespace vertex_of_quadratic_l43_43259

def quadratic_vertex (a b c : ℝ) : ℝ × ℝ :=
  (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem vertex_of_quadratic :
  let y := -1 * (x + 1)^2 - 8
  (quadratic_vertex (-1) 2 7) = (-1, -8) :=
sorry

end vertex_of_quadratic_l43_43259


namespace ambulance_reachable_area_l43_43035

theorem ambulance_reachable_area :
  let travel_time_minutes := 8
  let travel_time_hours := (travel_time_minutes : ℝ) / 60
  let speed_on_road := 60 -- speed in miles per hour
  let speed_off_road := 10 -- speed in miles per hour
  let distance_on_road := speed_on_road * travel_time_hours
  distance_on_road = 8 → -- this verifies the distance covered on road
  let area := (2 * distance_on_road) ^ 2
  area = 256 := sorry

end ambulance_reachable_area_l43_43035


namespace ram_account_balance_first_year_l43_43282

theorem ram_account_balance_first_year :
  let initial_deposit := 1000
  let interest_first_year := 100
  initial_deposit + interest_first_year = 1100 :=
by
  sorry

end ram_account_balance_first_year_l43_43282


namespace nancy_total_spending_l43_43109

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l43_43109


namespace find_number_l43_43209

variable (a n : ℝ)

theorem find_number (h1: 2 * a = 3 * n) (h2: a * n ≠ 0) (h3: (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 :=
sorry

end find_number_l43_43209


namespace xyz_product_value_l43_43194

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l43_43194


namespace probability_intersection_first_quadrant_l43_43721

theorem probability_intersection_first_quadrant :
  let l₁ (x y : ℝ) := x - 2 * y - 1 = 0
  let l₂ (a b x y : ℝ) := ax - by + 1 = 0
  let points_in_first_quadrant (a b : ℝ) : Prop := a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧
    (∃ x y, l₁ x y ∧ l₂ a b x y ∧ x > 0 ∧ y > 0)
  (Finset.card (Finset.filter (λ (ab : ℝ × ℝ), points_in_first_quadrant ab.1 ab.2)
    (Finset.product (Finset.range 6) (Finset.range 6)))) / 36 = 1/6 := 
sorry

end probability_intersection_first_quadrant_l43_43721


namespace track_circumference_l43_43102

variable (A B : Nat → ℝ)
variable (speedA speedB : ℝ)
variable (x : ℝ) -- half the circumference of the track
variable (y : ℝ) -- the circumference of the track

theorem track_circumference
  (x_pos : 0 < x)
  (y_def : y = 2 * x)
  (start_opposite : A 0 = 0 ∧ B 0 = x)
  (B_first_meet_150 : ∃ t₁, B t₁ = 150 ∧ A t₁ = x - 150)
  (A_second_meet_90 : ∃ t₂, A t₂ = 2 * x - 90 ∧ B t₂ = x + 90) :
  y = 720 := 
by 
  sorry

end track_circumference_l43_43102


namespace find_polynomials_l43_43332

-- Definition of polynomials in Lean
noncomputable def polynomials : Type := Polynomial ℝ

-- Main theorem statement
theorem find_polynomials : 
  ∀ p : polynomials, 
    (∀ x : ℝ, p.eval (5 * x) ^ 2 - 3 = p.eval (5 * x^2 + 1)) → 
    (p.eval 0 ≠ 0 → (∃ c : ℝ, (p = Polynomial.C c) ∧ (c = (1 + Real.sqrt 13) / 2 ∨ c = (1 - Real.sqrt 13) / 2))) ∧ 
    (p.eval 0 = 0 → ∀ x : ℝ, p.eval x = 0) :=
by
  sorry

end find_polynomials_l43_43332


namespace proposition_verification_l43_43935

-- Definitions and Propositions
def prop1 : Prop := (∀ x, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x, x ≠ 1 ∧ x^2 - 3 * x + 2 = 0)
def prop2 : Prop := (∀ x, ¬ (x^2 - 3 * x + 2 = 0 → x = 1) → (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0))
def prop3 : Prop := ¬ (∃ x > 0, x^2 + x + 1 < 0) → (∀ x ≤ 0, x^2 + x + 1 ≥ 0)
def prop4 : Prop := ¬ (∃ p q : Prop, (p ∨ q) → ¬p ∧ ¬q)

-- Final theorem statement
theorem proposition_verification : prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by 
  sorry

end proposition_verification_l43_43935


namespace gcd_a_b_eq_one_l43_43087

def a : ℕ := 130^2 + 240^2 + 350^2
def b : ℕ := 131^2 + 241^2 + 349^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l43_43087


namespace arrange_students_l43_43974

theorem arrange_students (students : Fin 7 → Prop) : 
  ∃ arrangements : ℕ, arrangements = 140 :=
by
  -- Define selection of 6 out of 7
  let selection_ways := Nat.choose 7 6
  -- Define arrangement of 6 into two groups of 3 each
  let arrangement_ways := (Nat.choose 6 3) * (Nat.choose 3 3)
  -- Calculate total arrangements by multiplying the two values
  let total_arrangements := selection_ways * arrangement_ways
  use total_arrangements
  simp [selection_ways, arrangement_ways, total_arrangements]
  exact rfl

end arrange_students_l43_43974


namespace inequality_of_abc_l43_43600

theorem inequality_of_abc (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_of_abc_l43_43600


namespace simplify_tan_expression_l43_43405

theorem simplify_tan_expression :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have tan_add : Real.tan (10 * Real.pi / 180 + 35 * Real.pi / 180) = (Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180)) / (1 - Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180)) := by sorry
  have eq : Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180) = 1 - Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180) := by
    rw [← tan_add, tan_45]
    field_simp
    ring
  have res : (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 1 + (Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180)) + Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180) := by ring
  rw [eq, res]
  ring
  sorry

end simplify_tan_expression_l43_43405


namespace product_of_distances_l43_43871

-- Definitions based on the conditions
def curve (x y : ℝ) : Prop := x * y = 2

-- The theorem to prove
theorem product_of_distances (x y : ℝ) (h : curve x y) : abs x * abs y = 2 := by
  -- This is where the proof would go
  sorry

end product_of_distances_l43_43871


namespace factor_poly_l43_43556

theorem factor_poly (x : ℤ) :
  (x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1)) :=
by
  sorry

end factor_poly_l43_43556


namespace absolute_sum_of_roots_l43_43008

theorem absolute_sum_of_roots (d e f n : ℤ) (h1 : d + e + f = 0) (h2 : d * e + e * f + f * d = -2023) : |d| + |e| + |f| = 98 := 
sorry

end absolute_sum_of_roots_l43_43008


namespace count_positive_integers_satisfying_properties_l43_43265

theorem count_positive_integers_satisfying_properties :
  (∃ n : ℕ, ∀ N < 2007,
    (N % 2 = 1) ∧
    (N % 3 = 2) ∧
    (N % 4 = 3) ∧
    (N % 5 = 4) ∧
    (N % 6 = 5) → n = 33) :=
by
  sorry

end count_positive_integers_satisfying_properties_l43_43265


namespace minimum_percentage_increase_mean_l43_43499

def mean (s : List ℤ) : ℚ :=
  (s.sum : ℚ) / s.length

theorem minimum_percentage_increase_mean (F : List ℤ) (p1 p2 : ℤ) (F' : List ℤ)
  (hF : F = [ -4, -1, 0, 6, 9 ])
  (hp1 : p1 = 2) (hp2 : p2 = 3)
  (hF' : F' = [p1, p2, 0, 6, 9])
  : (mean F' - mean F) / mean F * 100 = 100 := 
sorry

end minimum_percentage_increase_mean_l43_43499


namespace decimal_to_fraction_l43_43485

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43485


namespace chewbacca_gum_pieces_l43_43318

theorem chewbacca_gum_pieces (y : ℚ)
  (h1 : ∀ x : ℚ, x ≠ 0 → (15 - y) = 15 * (25 + 2 * y) / 25) :
  y = 5 / 2 :=
by
  sorry

end chewbacca_gum_pieces_l43_43318


namespace fermat_little_theorem_l43_43683

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) (hcoprime : Int.gcd a p = 1) : 
  (a ^ (p - 1)) % p = 1 % p := 
sorry

end fermat_little_theorem_l43_43683


namespace decimal_to_fraction_l43_43465

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43465


namespace xyz_product_neg4_l43_43203

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l43_43203


namespace ratio_man_to_son_in_two_years_l43_43840

-- Define current ages and the conditions
def son_current_age : ℕ := 24
def man_current_age : ℕ := son_current_age + 26

-- Define ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- State the theorem
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 :=
by sorry

end ratio_man_to_son_in_two_years_l43_43840


namespace rectangle_width_decreased_by_33_percent_l43_43749

theorem rectangle_width_decreased_by_33_percent
  (L W A : ℝ)
  (hA : A = L * W)
  (newL : ℝ)
  (h_newL : newL = 1.5 * L)
  (W' : ℝ)
  (h_area_unchanged : newL * W' = A) : 
  (1 - W' / W) * 100 = 33.33 :=
by
  sorry

end rectangle_width_decreased_by_33_percent_l43_43749


namespace factorial_div_eq_l43_43143
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l43_43143


namespace geometric_progression_condition_l43_43707

theorem geometric_progression_condition {b : ℕ → ℝ} (b1_ne_b2 : b 1 ≠ b 2) (h : ∀ n, b (n + 2) = b n / b (n + 1)) :
  (∀ n, b (n+1) / b n = b 2 / b 1) ↔ b 1 = b 2^3 := sorry

end geometric_progression_condition_l43_43707


namespace mrs_sheridan_gave_away_14_cats_l43_43236

def num_initial_cats : ℝ := 17.0
def num_left_cats : ℝ := 3.0
def num_given_away (x : ℝ) : Prop := num_initial_cats - x = num_left_cats

theorem mrs_sheridan_gave_away_14_cats : num_given_away 14.0 :=
by
  sorry

end mrs_sheridan_gave_away_14_cats_l43_43236


namespace smallest_n_satisfying_condition_l43_43979

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (n > 1) ∧ (∀ i : ℕ, i ≥ 1 → i < n → (∃ k : ℕ, i + (i+1) = k^2)) ∧ n = 8 :=
sorry

end smallest_n_satisfying_condition_l43_43979


namespace jerky_batch_size_l43_43086

theorem jerky_batch_size
  (total_order_bags : ℕ)
  (initial_bags : ℕ)
  (days_to_fulfill : ℕ)
  (remaining_bags : ℕ := total_order_bags - initial_bags)
  (production_per_day : ℕ := remaining_bags / days_to_fulfill) :
  total_order_bags = 60 →
  initial_bags = 20 →
  days_to_fulfill = 4 →
  production_per_day = 10 :=
by
  intros
  sorry

end jerky_batch_size_l43_43086


namespace actual_area_of_park_l43_43310

-- Definitions of given conditions
def map_scale : ℕ := 250 -- scale: 1 inch = 250 miles
def map_length : ℕ := 6 -- length on map in inches
def map_width : ℕ := 4 -- width on map in inches

-- Definition of actual lengths
def actual_length : ℕ := map_length * map_scale -- actual length in miles
def actual_width : ℕ := map_width * map_scale -- actual width in miles

-- Theorem to prove the actual area
theorem actual_area_of_park : actual_length * actual_width = 1500000 := by
  -- By the conditions provided, the actual length and width in miles can be calculated directly:
  -- actual_length = 6 * 250 = 1500
  -- actual_width = 4 * 250 = 1000
  -- actual_area = 1500 * 1000 = 1500000
  sorry

end actual_area_of_park_l43_43310


namespace maximize_annual_profit_l43_43301

theorem maximize_annual_profit : 
  ∃ n : ℕ, n ≠ 0 ∧ (∀ m : ℕ, m ≠ 0 → (110 * n - (n * n + n) - 90) / n ≥ (110 * m - (m * m + m) - 90) / m) ↔ n = 5 := 
by 
  -- Proof steps would go here
  sorry

end maximize_annual_profit_l43_43301


namespace urn_contains_four_each_color_after_six_steps_l43_43702

noncomputable def probability_urn_four_each_color : ℚ := 2 / 7

def urn_problem (urn_initial : ℕ) (draws : ℕ) (final_urn : ℕ) (extra_balls : ℕ) : Prop :=
urn_initial = 2 ∧ draws = 6 ∧ final_urn = 8 ∧ extra_balls > 0

theorem urn_contains_four_each_color_after_six_steps :
  urn_problem 2 6 8 2 → probability_urn_four_each_color = 2 / 7 :=
by
  intro h
  cases h
  sorry

end urn_contains_four_each_color_after_six_steps_l43_43702


namespace problem_l43_43189

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l43_43189


namespace perfect_square_trinomial_m_l43_43725

theorem perfect_square_trinomial_m (m : ℤ) :
  ∀ y : ℤ, ∃ a : ℤ, (y^2 - m * y + 1 = (y + a) ^ 2) ∨ (y^2 - m * y + 1 = (y - a) ^ 2) → (m = 2 ∨ m = -2) :=
by 
  sorry

end perfect_square_trinomial_m_l43_43725


namespace compare_final_values_l43_43061

noncomputable def final_value_Almond (initial: ℝ): ℝ := (initial * 1.15) * 0.85
noncomputable def final_value_Bean (initial: ℝ): ℝ := (initial * 0.80) * 1.20
noncomputable def final_value_Carrot (initial: ℝ): ℝ := (initial * 1.10) * 0.90

theorem compare_final_values (initial: ℝ) (h_positive: 0 < initial):
  final_value_Almond initial < final_value_Bean initial ∧ 
  final_value_Bean initial < final_value_Carrot initial := by
  sorry

end compare_final_values_l43_43061


namespace max_number_of_children_l43_43937

theorem max_number_of_children (apples cookies chocolates : ℕ) (remaining_apples remaining_cookies remaining_chocolates : ℕ) 
  (h₁ : apples = 55) 
  (h₂ : cookies = 114) 
  (h₃ : chocolates = 83) 
  (h₄ : remaining_apples = 3) 
  (h₅ : remaining_cookies = 10) 
  (h₆ : remaining_chocolates = 5) : 
  gcd (apples - remaining_apples) (gcd (cookies - remaining_cookies) (chocolates - remaining_chocolates)) = 26 :=
by
  sorry

end max_number_of_children_l43_43937


namespace find_values_of_x_l43_43724

noncomputable def solution_x (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ 
  x^2 + 1/y = 13 ∧ 
  y^2 + 1/x = 8 ∧ 
  (x = Real.sqrt 13 ∨ x = -Real.sqrt 13)

theorem find_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) : x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by { sorry }

end find_values_of_x_l43_43724


namespace intersect_x_axis_once_l43_43732

theorem intersect_x_axis_once (k : ℝ) : 
  (∀ x : ℝ, (k - 3) * x^2 + 2 * x + 1 = 0 → x = 0) → (k = 3 ∨ k = 4) :=
by
  intro h
  sorry

end intersect_x_axis_once_l43_43732


namespace simple_interest_rate_l43_43750

theorem simple_interest_rate (P : ℝ) (R : ℝ) (SI : ℝ) (T : ℝ) (h1 : T = 4) (h2 : SI = P / 5) (h3 : SI = (P * R * T) / 100) : R = 5 := by
  sorry

end simple_interest_rate_l43_43750


namespace hexagon_longest_side_l43_43676

theorem hexagon_longest_side (x : ℝ) (h₁ : 6 * x = 20) (h₂ : x < 20 - x) : (10 / 3) ≤ x ∧ x < 10 :=
sorry

end hexagon_longest_side_l43_43676


namespace solve_equation_l43_43407

open Function

theorem solve_equation (m n : ℕ) (h_gcd : gcd m n = 2) (h_lcm : lcm m n = 4) :
  m * n = (gcd m n)^2 + lcm m n ↔ (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) :=
by
  sorry

end solve_equation_l43_43407


namespace nickys_pace_l43_43237

theorem nickys_pace (distance : ℝ) (head_start_time : ℝ) (cristina_pace : ℝ) 
    (time_before_catchup : ℝ) (nicky_distance : ℝ) :
    distance = 100 ∧ head_start_time = 12 ∧ cristina_pace = 5 
    ∧ time_before_catchup = 30 ∧ nicky_distance = 90 →
    nicky_distance / time_before_catchup = 3 :=
by
  sorry

end nickys_pace_l43_43237


namespace value_of_V3_l43_43666

def f (x : ℝ) : ℝ := 3 * x^5 + 8 * x^4 - 3 * x^3 + 5 * x^2 + 12 * x - 6

def horner (a : ℝ) : ℝ :=
  let V0 := 3
  let V1 := V0 * a + 8
  let V2 := V1 * a - 3
  let V3 := V2 * a + 5
  V3

theorem value_of_V3 : horner 2 = 55 :=
  by
    simp [horner]
    sorry

end value_of_V3_l43_43666


namespace math_problem_l43_43623

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l43_43623


namespace efficiency_ratio_l43_43497

variable {A B : ℝ}

theorem efficiency_ratio (hA : A = 1 / 30) (hAB : A + B = 1 / 20) : A / B = 2 :=
by
  sorry

end efficiency_ratio_l43_43497


namespace students_in_trumpet_or_trombone_l43_43296

theorem students_in_trumpet_or_trombone (h₁ : 0.5 + 0.12 = 0.62) : 
  0.5 + 0.12 = 0.62 :=
by
  exact h₁

end students_in_trumpet_or_trombone_l43_43296


namespace A_eq_three_l43_43619

theorem A_eq_three (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (A : ℤ)
  (h : A = ((a + 1 : ℕ) / (b : ℕ)) + (b : ℕ) / (a : ℕ)) : A = 3 := by
  sorry

end A_eq_three_l43_43619


namespace y1_lt_y2_of_linear_graph_l43_43903

/-- In the plane rectangular coordinate system xOy, if points A(2, y1) and B(5, y2) 
    lie on the graph of a linear function y = x + b (where b is a constant), then y1 < y2. -/
theorem y1_lt_y2_of_linear_graph (y1 y2 b : ℝ) (hA : y1 = 2 + b) (hB : y2 = 5 + b) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_graph_l43_43903


namespace surface_area_of_tunneled_cube_l43_43834

-- Definition of the initial cube and its properties.
def cube (side_length : ℕ) := side_length * side_length * side_length

-- Initial side length of the large cube
def large_cube_side : ℕ := 12

-- Each small cube side length
def small_cube_side : ℕ := 3

-- Number of small cubes that fit into the large cube
def num_small_cubes : ℕ := (cube large_cube_side) / (cube small_cube_side)

-- Number of cubes removed initially
def removed_cubes : ℕ := 27

-- Number of remaining cubes after initial removal
def remaining_cubes : ℕ := num_small_cubes - removed_cubes

-- Surface area of each unmodified small cube
def small_cube_surface : ℕ := 54

-- Additional surface area due to removal of center units
def additional_surface : ℕ := 24

-- Surface area of each modified small cube
def modified_cube_surface : ℕ := small_cube_surface + additional_surface

-- Total surface area before adjustment for shared faces
def total_surface_before_adjustment : ℕ := remaining_cubes * modified_cube_surface

-- Shared surface area to be subtracted
def shared_surface : ℕ := 432

-- Final surface area of the resulting figure
def final_surface_area : ℕ := total_surface_before_adjustment - shared_surface

-- Theorem statement
theorem surface_area_of_tunneled_cube : final_surface_area = 2454 :=
by {
  -- Proof required here
  sorry
}

end surface_area_of_tunneled_cube_l43_43834


namespace simplify_expression_l43_43766

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 3) : 
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  (1 / (b^2 + c^2 - 3 * a^2) + 1 / (a^2 + c^2 - 3 * b^2) + 1 / (a^2 + b^2 - 3 * c^2) = -3) :=
by
  intros
  sorry

end simplify_expression_l43_43766


namespace djibo_age_sum_years_ago_l43_43997

theorem djibo_age_sum_years_ago (x : ℕ) (h₁: 17 - x + 28 - x = 35) : x = 5 :=
by
  -- proof is omitted as per instructions
  sorry

end djibo_age_sum_years_ago_l43_43997


namespace g_f_neg3_eq_1741_l43_43227

def f (x : ℤ) : ℤ := x^3 - 3
def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg3_eq_1741 : g (f (-3)) = 1741 := 
by 
  sorry

end g_f_neg3_eq_1741_l43_43227


namespace determine_m_l43_43884

theorem determine_m (x y m : ℝ) 
  (h1 : 3 * x + 2 * y = 4 * m - 5) 
  (h2 : 2 * x + 3 * y = m) 
  (h3 : x + y = 2) : 
  m = 3 :=
sorry

end determine_m_l43_43884


namespace fish_population_estimate_l43_43801

theorem fish_population_estimate 
  (marked_initial : ℕ) (total_second_catch : ℕ) (marked_second_catch : ℕ) : 
  marked_initial = 30 → 
  total_second_catch = 50 → 
  marked_second_catch = 2 → 
  ∃ (x : ℕ), x = 750 :=
by
  intros
  use 750
  sorry

end fish_population_estimate_l43_43801


namespace part1_part2_part3_l43_43878

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end part1_part2_part3_l43_43878


namespace trajectory_moving_point_l43_43414

theorem trajectory_moving_point (x y : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 ↔ x^2 + y^2 = 1 := by
  sorry

end trajectory_moving_point_l43_43414


namespace mrs_sheridan_initial_cats_l43_43051

def cats_initial (cats_given_away : ℕ) (cats_left : ℕ) : ℕ :=
  cats_given_away + cats_left

theorem mrs_sheridan_initial_cats : cats_initial 14 3 = 17 :=
by
  sorry

end mrs_sheridan_initial_cats_l43_43051


namespace x_squared_minus_y_squared_l43_43207

theorem x_squared_minus_y_squared (x y : ℚ) (h₁ : x + y = 9 / 17) (h₂ : x - y = 1 / 51) : x^2 - y^2 = 1 / 289 :=
by
  sorry

end x_squared_minus_y_squared_l43_43207


namespace student_courses_last_year_l43_43127

variable (x : ℕ)
variable (courses_last_year : ℕ := x)
variable (avg_grade_last_year : ℕ := 100)
variable (courses_year_before : ℕ := 5)
variable (avg_grade_year_before : ℕ := 60)
variable (avg_grade_two_years : ℕ := 81)

theorem student_courses_last_year (h1 : avg_grade_last_year = 100)
                                   (h2 : courses_year_before = 5)
                                   (h3 : avg_grade_year_before = 60)
                                   (h4 : avg_grade_two_years = 81)
                                   (hc : ((5 * avg_grade_year_before) + (courses_last_year * avg_grade_last_year)) / (courses_year_before + courses_last_year) = avg_grade_two_years) :
                                   courses_last_year = 6 := by
  sorry

end student_courses_last_year_l43_43127


namespace part1_part2_case1_part2_case2_part2_case3_part3_l43_43875

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end part1_part2_case1_part2_case2_part2_case3_part3_l43_43875


namespace problem1_problem2_problem3_problem4_problem5_problem6_l43_43650

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l43_43650


namespace train_length_problem_l43_43421

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l43_43421


namespace sum_three_digit_even_integers_l43_43672

theorem sum_three_digit_even_integers :
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 247050 :=
by
  let a := 100
  let d := 2
  let l := 998
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  sorry

end sum_three_digit_even_integers_l43_43672


namespace decimal_to_fraction_l43_43439

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43439


namespace cupcakes_left_correct_l43_43984

-- Definitions based on conditions
def total_cupcakes : ℕ := 10 * 12 + 1 * 12 / 2
def total_students : ℕ := 48
def absent_students : ℕ := 6 
def field_trip_students : ℕ := 8
def teachers : ℕ := 2
def teachers_aids : ℕ := 2

-- Function to calculate the number of present people
def total_present_people : ℕ :=
  total_students - absent_students - field_trip_students + teachers + teachers_aids

-- Function to calculate the cupcakes left
def cupcakes_left : ℕ := total_cupcakes - total_present_people

-- The theorem to prove
theorem cupcakes_left_correct : cupcakes_left = 85 := 
by
  -- This is where the proof would go
  sorry

end cupcakes_left_correct_l43_43984


namespace expression_value_l43_43361

theorem expression_value (x y : ℝ) (h : x + y = -1) : x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l43_43361


namespace hemisphere_surface_area_l43_43656

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  (4 * Real.pi * r^2) / 2 + (Real.pi * r^2) = 300 * Real.pi := by
  sorry

end hemisphere_surface_area_l43_43656


namespace square_side_length_leq_half_l43_43790

theorem square_side_length_leq_half
    (l : ℝ)
    (h_square_inside_unit : l ≤ 1)
    (h_no_center_contain : ∀ (x y : ℝ), x^2 + y^2 > (l/2)^2 → (0.5 ≤ x ∨ 0.5 ≤ y)) :
    l ≤ 0.5 := 
sorry

end square_side_length_leq_half_l43_43790


namespace asymptote_hyperbola_condition_l43_43934

theorem asymptote_hyperbola_condition : 
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = 4/3 * x ∨ y = -4/3 * x)) ∧
  ¬(∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x → x^2 / 9 - y^2 / 16 = 1)) :=
by sorry

end asymptote_hyperbola_condition_l43_43934


namespace area_of_triangle_is_23_over_10_l43_43663

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  1/2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_triangle_is_23_over_10 :
  let A : ℚ × ℚ := (3, 3)
  let B : ℚ × ℚ := (5, 3)
  let C : ℚ × ℚ := (21 / 5, 19 / 5)
  area_of_triangle A.1 A.2 B.1 B.2 C.1 C.2 = 23 / 10 :=
by
  sorry

end area_of_triangle_is_23_over_10_l43_43663


namespace cheryl_initial_mms_l43_43140

theorem cheryl_initial_mms (lunch_mms : ℕ) (dinner_mms : ℕ) (sister_mms : ℕ) (total_mms : ℕ) 
  (h1 : lunch_mms = 7) (h2 : dinner_mms = 5) (h3 : sister_mms = 13) (h4 : total_mms = lunch_mms + dinner_mms + sister_mms) : 
  total_mms = 25 := 
by 
  rw [h1, h2, h3] at h4
  exact h4

end cheryl_initial_mms_l43_43140


namespace printing_time_345_l43_43518

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l43_43518


namespace inequality_solution_set_l43_43807

theorem inequality_solution_set (x : ℝ) : |x - 5| + |x + 3| ≤ 10 ↔ -4 ≤ x ∧ x ≤ 6 :=
by
  sorry

end inequality_solution_set_l43_43807


namespace distinct_solutions_subtract_eight_l43_43046

noncomputable def f (x : ℝ) : ℝ := (6 * x - 18) / (x^2 + 2 * x - 15)
noncomputable def equation := ∀ x, f x = x + 3

noncomputable def r_solutions (r s : ℝ) := (r > s) ∧ (f r = r + 3) ∧ (f s = s + 3)

theorem distinct_solutions_subtract_eight
  (r s : ℝ) (h : r_solutions r s) : r - s = 8 :=
sorry

end distinct_solutions_subtract_eight_l43_43046


namespace find_m_l43_43567

theorem find_m 
  (m : ℕ) 
  (hm_pos : 0 < m) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := 
sorry

end find_m_l43_43567


namespace rhombus_diagonal_l43_43694

theorem rhombus_diagonal (side : ℝ) (short_diag : ℝ) (long_diag : ℝ) 
  (h1 : side = 37) (h2 : short_diag = 40) :
  long_diag = 62 :=
sorry

end rhombus_diagonal_l43_43694


namespace problem_solution_l43_43886

theorem problem_solution (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 :=
sorry

end problem_solution_l43_43886


namespace angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l43_43230

-- Definitions according to the given conditions
variables (A B C D E F M N P : Point)
  (hexagon_regular : is_regular_hexagon A B C D E F)
  (is_midpoint_M : is_midpoint M C D)
  (is_midpoint_N : is_midpoint N D E)
  (intersection_P : intersection_point P (line_through A M) (line_through B N))

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_60_degrees 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  angle (line_through A M) (line_through B N) = 60 := 
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_triangle_ABP_eq_area_quadrilateral_MDNP 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  area (triangle A B P) = area (quadrilateral M D N P) := 
sorry

end angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l43_43230


namespace range_of_f_l43_43735

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem range_of_f : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ f x ∧ f x ≤ 2 :=
by
  intro x Hx
  sorry

end range_of_f_l43_43735


namespace exists_duplicate_parenthesizations_l43_43376

def expr : List Int := List.range' 1 (1991 + 1)

def num_parenthesizations : Nat := 2 ^ 995

def num_distinct_results : Nat := 3966067

theorem exists_duplicate_parenthesizations :
  num_parenthesizations > num_distinct_results :=
sorry

end exists_duplicate_parenthesizations_l43_43376


namespace Panikovsky_share_l43_43012

theorem Panikovsky_share :
  ∀ (horns hooves weight : ℕ) 
    (k δ : ℝ),
    horns = 17 →
    hooves = 2 →
    weight = 1 →
    (∀ h, h = k + δ) →
    (∀ wt, wt = k + 2 * δ) →
    (20 * k + 19 * δ) / 2 = 10 * k + 9.5 * δ →
    9 * k + 7.5 * δ = (9 * (k + δ) + 2 * k) →
    ∃ (Panikov_hearts Panikov_hooves : ℕ), 
    Panikov_hearts = 9 ∧ Panikov_hooves = 2 := 
by
  intros
  sorry

end Panikovsky_share_l43_43012


namespace find_inner_circle_radius_of_trapezoid_l43_43765

noncomputable def radius_of_inner_circle (k m n p : ℤ) : ℝ :=
  (-k + m * Real.sqrt n) / p

def is_equivalent (a b : ℝ) : Prop := a = b

theorem find_inner_circle_radius_of_trapezoid :
  ∃ (r : ℝ), is_equivalent r (radius_of_inner_circle 123 104 3 29) :=
by
  let r := radius_of_inner_circle 123 104 3 29
  have h1 :  (4^2 + (Real.sqrt (r^2 + 8 * r))^2 = (r + 4)^2) := sorry
  have h2 :  (3^2 + (Real.sqrt (r^2 + 6 * r))^2 = (r + 3)^2) := sorry
  have height_eq : Real.sqrt 13 = (Real.sqrt (r^2 + 6 * r) + Real.sqrt (r^2 + 8 * r)) := sorry
  use r
  exact sorry

end find_inner_circle_radius_of_trapezoid_l43_43765


namespace largest_multiple_of_7_smaller_than_negative_85_l43_43089

theorem largest_multiple_of_7_smaller_than_negative_85 :
  ∃ (n : ℤ), (∃ (k : ℤ), n = 7 * k) ∧ n < -85 ∧ ∀ (m : ℤ), (∃ (k : ℤ), m = 7 * k) ∧ m < -85 → m ≤ n := 
by
  use -91
  split
  { use -13
    norm_num }
  split
  { exact dec_trivial }
  { intros m hm
    cases hm with k hk
    cases hk with hk1 hk2
    have hk3 : k < -12 := by linarith
    have hk4 : k ≤ -13 := int.floor_le $ hk3
    linarith }


end largest_multiple_of_7_smaller_than_negative_85_l43_43089


namespace max_knights_seated_l43_43239

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end max_knights_seated_l43_43239


namespace value_of_work_clothes_l43_43126

theorem value_of_work_clothes (x y : ℝ) (h1 : x + 70 = 30 * y) (h2 : x + 20 = 20 * y) : x = 80 :=
by
  sorry

end value_of_work_clothes_l43_43126


namespace chessboard_edge_count_l43_43947

theorem chessboard_edge_count (n : ℕ) 
  (border_white : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ w : ℕ, w ≥ n)) 
  (border_black : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ b : ℕ, b ≥ n)) :
  ∃ e : ℕ, e ≥ n :=
sorry

end chessboard_edge_count_l43_43947


namespace measure_weights_l43_43919

theorem measure_weights (w1 w3 w7 : Nat) (h1 : w1 = 1) (h3 : w3 = 3) (h7 : w7 = 7) :
  ∃ s : Finset Nat, s.card = 7 ∧ 
    (1 ∈ s) ∧ (3 ∈ s) ∧ (7 ∈ s) ∧
    (4 ∈ s) ∧ (8 ∈ s) ∧ (10 ∈ s) ∧ 
    (11 ∈ s) := 
by
  sorry

end measure_weights_l43_43919


namespace balloon_permutations_l43_43996

theorem balloon_permutations : 
  let str := "BALLOON",
  let total_letters := 7,
  let repeated_L := 2,
  let repeated_O := 2,
  nat.factorial total_letters / (nat.factorial repeated_L * nat.factorial repeated_O) = 1260 := 
begin
  sorry
end

end balloon_permutations_l43_43996


namespace min_value_of_sum_of_squares_l43_43386

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x * y + y * z + x * z = 4) :
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_value_of_sum_of_squares_l43_43386


namespace sandra_remaining_money_l43_43401

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end sandra_remaining_money_l43_43401


namespace product_xyz_equals_zero_l43_43196

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l43_43196


namespace negation_of_existence_l43_43246

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 > 2) ↔ ∀ x : ℝ, x^2 ≤ 2 :=
by
  sorry

end negation_of_existence_l43_43246


namespace fraction_zero_solution_l43_43893

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end fraction_zero_solution_l43_43893


namespace factorial_quotient_l43_43152

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l43_43152


namespace no_solution_condition_l43_43710

theorem no_solution_condition (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := 
by
  sorry

end no_solution_condition_l43_43710


namespace profit_per_package_l43_43868

theorem profit_per_package
  (packages_first_center_per_day : ℕ)
  (packages_second_center_multiplier : ℕ)
  (weekly_profit : ℕ)
  (days_per_week : ℕ)
  (H1 : packages_first_center_per_day = 10000)
  (H2 : packages_second_center_multiplier = 3)
  (H3 : weekly_profit = 14000)
  (H4 : days_per_week = 7) :
  (weekly_profit / (packages_first_center_per_day * days_per_week + 
                    packages_second_center_multiplier * packages_first_center_per_day * days_per_week) : ℝ) = 0.05 :=
by
  sorry

end profit_per_package_l43_43868


namespace Andrew_runs_2_miles_each_day_l43_43397

theorem Andrew_runs_2_miles_each_day
  (A : ℕ)
  (Peter_runs : ℕ := A + 3)
  (total_miles_after_5_days : 5 * (A + Peter_runs) = 35) :
  A = 2 :=
by
  sorry

end Andrew_runs_2_miles_each_day_l43_43397


namespace jaya_rank_from_bottom_l43_43378

theorem jaya_rank_from_bottom (n t : ℕ) (h_n : n = 53) (h_t : t = 5) : n - t + 1 = 50 := by
  sorry

end jaya_rank_from_bottom_l43_43378


namespace inequality_proof_l43_43879

open Real

noncomputable def f (t x : ℝ) : ℝ := t * x - (t - 1) * log x - t

theorem inequality_proof (t x : ℝ) (h_t : t ≤ 0) (h_x : x > 1) : 
  f t x < exp (x - 1) - 1 :=
sorry

end inequality_proof_l43_43879


namespace minimum_experiments_fractional_method_l43_43514

/--
A pharmaceutical company needs to optimize the cultivation temperature for a certain medicinal liquid through bioassay.
The experimental range is set from 29℃ to 63℃, with an accuracy requirement of ±1℃.
Prove that the minimum number of experiments required to ensure the best cultivation temperature is found using the fractional method is 7.
-/
theorem minimum_experiments_fractional_method
  (range_start : ℕ)
  (range_end : ℕ)
  (accuracy : ℕ)
  (fractional_method : ∀ (range_start range_end accuracy: ℕ), ℕ) :
  range_start = 29 → range_end = 63 → accuracy = 1 → fractional_method range_start range_end accuracy = 7 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end minimum_experiments_fractional_method_l43_43514


namespace probability_xi_eq_1_l43_43620

-- Definitions based on conditions
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

-- Combinatorics function for choosing k items from n items
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Definition for probability P(ξ = 1)
def P_xi_eq_1 := 
  (C white_balls_bag_A 1 * C white_balls_bag_B 1 + C red_balls_bag_A 1 * C white_balls_bag_B 1) /
  (C (white_balls_bag_A + red_balls_bag_A) 1 * C (white_balls_bag_B + red_balls_bag_B) 1)

theorem probability_xi_eq_1 :
  P_xi_eq_1 = (C 8 1 * C 6 1 + C 4 1 * C 6 1) / (C 12 1 * C 12 1) :=
by
  sorry

end probability_xi_eq_1_l43_43620


namespace exists_prime_q_l43_43232

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) :
  ∃ q, Nat.Prime q ∧ ∀ n, ¬ (q ∣ n^p - p) := by
  sorry

end exists_prime_q_l43_43232


namespace nancy_total_spending_l43_43110

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l43_43110


namespace x_investment_amount_l43_43962

variable (X : ℝ)
variable (investment_y : ℝ := 15000)
variable (total_profit : ℝ := 1600)
variable (x_share : ℝ := 400)

theorem x_investment_amount :
  (total_profit - x_share) / investment_y = x_share / X → X = 5000 :=
by
  intro ratio
  have h1: 1200 / 15000 = 400 / 5000 :=
    by sorry
  have h2: X = 5000 :=
    by sorry
  exact h2

end x_investment_amount_l43_43962


namespace hyperbola_center_is_correct_l43_43335

theorem hyperbola_center_is_correct :
  ∃ h k : ℝ, (∀ x y : ℝ, ((4 * y + 8)^2 / 16^2) - ((5 * x - 15)^2 / 9^2) = 1 → x - h = 0 ∧ y + k = 0) ∧ h = 3 ∧ k = -2 :=
sorry

end hyperbola_center_is_correct_l43_43335


namespace score_order_l43_43756

variable (A B C D : ℕ)

theorem score_order
  (h1 : A + C = B + D)
  (h2 : B > D)
  (h3 : C > A + B) :
  C > B ∧ B > A ∧ A > D :=
by 
  sorry

end score_order_l43_43756


namespace kevin_food_expenditure_l43_43780

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l43_43780


namespace inscribed_circle_radius_l43_43670

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 8 8 5 = 38 / 21 :=
by
  sorry

end inscribed_circle_radius_l43_43670


namespace sin_minus_cos_eq_sqrt3_div2_l43_43019

theorem sin_minus_cos_eq_sqrt3_div2
  (α : ℝ) 
  (h_range : (Real.pi / 4) < α ∧ α < (Real.pi / 2))
  (h_sincos : Real.sin α * Real.cos α = 1 / 8) :
  Real.sin α - Real.cos α = Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_eq_sqrt3_div2_l43_43019


namespace proof_problem_l43_43124

noncomputable def problem_statement : Prop :=
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ 
    x1 + x2 + x3 + x4 = 8 ∧ 
    x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ 
    (x1 + x2) = 2 * 2 ∧ 
    (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 - 4 * 2 * (x1 + x2 + x3 + x4) + 4 * 4) = 4 ∧ 
    (x1 = 1 ∧ x2 = 1 ∧ x3 = 3 ∧ x4 = 3)

theorem proof_problem : problem_statement :=
sorry

end proof_problem_l43_43124


namespace multiple_of_six_and_nine_l43_43791

-- Definitions: x is a multiple of 6, y is a multiple of 9.
def is_multiple_of_six (x : ℤ) : Prop := ∃ m : ℤ, x = 6 * m
def is_multiple_of_nine (y : ℤ) : Prop := ∃ n : ℤ, y = 9 * n

-- Assertions: Given the conditions, prove the following.
theorem multiple_of_six_and_nine (x y : ℤ)
  (hx : is_multiple_of_six x) (hy : is_multiple_of_nine y) :
  ((∃ k : ℤ, x - y = 3 * k) ∧
   (∃ m n : ℤ, x = 6 * m ∧ y = 9 * n ∧ (2 * m - 3 * n) % 3 ≠ 0)) :=
by
  sorry

end multiple_of_six_and_nine_l43_43791


namespace kevin_food_spending_l43_43784

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l43_43784


namespace area_comparison_l43_43595

-- Define the side lengths of the triangles
def a₁ := 17
def b₁ := 17
def c₁ := 12

def a₂ := 17
def b₂ := 17
def c₂ := 16

-- Define the semiperimeters
def s₁ := (a₁ + b₁ + c₁) / 2
def s₂ := (a₂ + b₂ + c₂) / 2

-- Define the areas using Heron's formula
noncomputable def area₁ := (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)).sqrt
noncomputable def area₂ := (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)).sqrt

-- The theorem to prove
theorem area_comparison : area₁ < area₂ := sorry

end area_comparison_l43_43595


namespace inequality_a_inequality_c_inequality_d_l43_43641

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l43_43641


namespace infinite_series_equals_two_l43_43328

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l43_43328


namespace calories_per_strawberry_l43_43290

theorem calories_per_strawberry (x : ℕ) :
  (12 * x + 6 * 17 = 150) → x = 4 := by
  sorry

end calories_per_strawberry_l43_43290


namespace exists_l_l43_43295

theorem exists_l (n : ℕ) (h : n ≥ 4011^2) : ∃ l : ℤ, n < l^2 ∧ l^2 < (1 + 1/2005) * n := 
sorry

end exists_l_l43_43295


namespace find_number_l43_43797

theorem find_number (x : ℤ) (h : 2 * x - 8 = -12) : x = -2 :=
by
  sorry

end find_number_l43_43797


namespace triangle_area_inscribed_in_circle_l43_43530

theorem triangle_area_inscribed_in_circle :
  ∀ (x : ℝ), (2 * x)^2 + (3 * x)^2 = (4 * x)^2 → (5 = (4 * x) / 2) → (1/2 * (2 * x) * (3 * x) = 18.75) :=
by
  -- Assume all necessary conditions
  intros x h_ratio h_radius
  -- Skip the proof part using sorry
  sorry

end triangle_area_inscribed_in_circle_l43_43530


namespace sum_of_squares_of_consecutive_integers_l43_43168

theorem sum_of_squares_of_consecutive_integers :
  ∃ x : ℕ, x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) ∧ (x^2 + (x + 1)^2 + (x + 2)^2 = 77) :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l43_43168


namespace arithmetic_sequence_common_difference_l43_43428

-- Define the conditions
variables {S_3 a_1 a_3 : ℕ}
variables (d : ℕ)
axiom h1 : S_3 = 6
axiom h2 : a_3 = 4
axiom h3 : S_3 = 3 * (a_1 + a_3) / 2

-- Prove that the common difference d is 2
theorem arithmetic_sequence_common_difference :
  d = (a_3 - a_1) / 2 → d = 2 :=
by
  sorry -- Proof to be completed

end arithmetic_sequence_common_difference_l43_43428


namespace intersecting_lines_l43_43800

theorem intersecting_lines (a b : ℝ) (h1 : 3 = (1 / 3) * 6 + a) (h2 : 6 = (1 / 3) * 3 + b) : a + b = 6 :=
sorry

end intersecting_lines_l43_43800


namespace inequality_a_inequality_c_inequality_d_l43_43645

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l43_43645


namespace regular_polygon_interior_angle_ratio_l43_43806

theorem regular_polygon_interior_angle_ratio (r k : ℕ) (h1 : 180 - 360 / r = (5 : ℚ) / (3 : ℚ) * (180 - 360 / k)) (h2 : r = 2 * k) :
  r = 8 ∧ k = 4 :=
sorry

end regular_polygon_interior_angle_ratio_l43_43806


namespace percent_students_elected_to_learn_from_home_l43_43926

theorem percent_students_elected_to_learn_from_home (H : ℕ) : 
  (100 - H) / 2 = 30 → H = 40 := 
by
  sorry

end percent_students_elected_to_learn_from_home_l43_43926


namespace round_balloons_burst_l43_43038

theorem round_balloons_burst :
  let round_balloons := 5 * 20
  let long_balloons := 4 * 30
  let total_balloons := round_balloons + long_balloons
  let balloons_left := 215
  ((total_balloons - balloons_left) = 5) :=
by 
  sorry

end round_balloons_burst_l43_43038


namespace empty_subset_of_A_l43_43971

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A :=
by
  sorry

end empty_subset_of_A_l43_43971


namespace decimal_to_fraction_l43_43470

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43470


namespace complex_fraction_l43_43936

open Complex

/-- The given complex fraction \(\frac{5 - i}{1 - i}\) evaluates to \(3 + 2i\). -/
theorem complex_fraction : (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = ⟨3, 2⟩ :=
  by
  sorry

end complex_fraction_l43_43936


namespace distance_T_S_l43_43612

theorem distance_T_S : 
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  S - T = 25 :=
by
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  show S - T = 25
  sorry

end distance_T_S_l43_43612


namespace yogurt_production_cost_l43_43280

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l43_43280


namespace calculate_value_l43_43981

def a : ℤ := 3 * 4 * 5
def b : ℚ := 1/3 + 1/4 + 1/5

theorem calculate_value :
  (a : ℚ) * b = 47 := by
sorry

end calculate_value_l43_43981


namespace jungkook_needs_more_paper_l43_43593

def bundles : Nat := 5
def pieces_per_bundle : Nat := 8
def rows : Nat := 9
def sheets_per_row : Nat := 6

def total_pieces : Nat := bundles * pieces_per_bundle
def pieces_needed : Nat := rows * sheets_per_row
def pieces_missing : Nat := pieces_needed - total_pieces

theorem jungkook_needs_more_paper : pieces_missing = 14 := by
  sorry

end jungkook_needs_more_paper_l43_43593


namespace fruit_seller_l43_43689

theorem fruit_seller (A P : ℝ) (h1 : A = 700) (h2 : A * (100 - P) / 100 = 420) : P = 40 :=
sorry

end fruit_seller_l43_43689


namespace female_A_stand_end_both_female_not_at_ends_female_students_not_adjacent_female_A_right_of_B_l43_43104

open Nat
noncomputable theory
open_locale big_operators

-- Define the conditions
def num_male_students := 5
def num_female_students := 2

-- Define the first proof statement
theorem female_A_stand_end : 
  let total_ways := 2 * (num_male_students + num_female_students - 1)! in 
  total_ways = 1440 := by
  sorry

-- Define the second proof statement
theorem both_female_not_at_ends : 
  let choose_positions := (num_male_students - 1).choose num_female_students in
  let remaining_ways := (num_male_students - 1)! in
  let total_ways := choose_positions * remaining_ways in 
  total_ways = 2400 := by
  sorry

-- Define the third proof statement
theorem female_students_not_adjacent : 
  let male_permutations := num_male_students! in
  let gap_choices := (num_male_students + 1).choose num_female_students in
  let total_ways := male_permutations * gap_choices in 
  total_ways = 3600 := by
  sorry

-- Define the fourth proof statement
theorem female_A_right_of_B : 
  let total_permutations := (num_male_students + num_female_students)! in 
  let valid_arrangements := total_permutations / 2 in 
  valid_arrangements = 2520 := by
  sorry

end female_A_stand_end_both_female_not_at_ends_female_students_not_adjacent_female_A_right_of_B_l43_43104


namespace inequality_a_inequality_c_inequality_d_l43_43638

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l43_43638


namespace minimum_value_2x_plus_y_l43_43583

theorem minimum_value_2x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y + 6 = x * y) : 
  2 * x + y ≥ 12 := 
sorry

end minimum_value_2x_plus_y_l43_43583


namespace regular_polygon_exterior_angle_l43_43538

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end regular_polygon_exterior_angle_l43_43538


namespace xyz_product_neg4_l43_43204

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l43_43204


namespace simplify_tan_product_l43_43406

-- Mathematical Conditions
def tan_inv (x : ℝ) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry

-- Problem statement to be proven
theorem simplify_tan_product (x y : ℝ) (hx : tan_inv x = 10) (hy : tan_inv y = 35) :
  (1 + x) * (1 + y) = 2 :=
sorry

end simplify_tan_product_l43_43406


namespace gcd_increase_by_9_l43_43494

theorem gcd_increase_by_9 (m n d : ℕ) (h1 : d = Nat.gcd m n) (h2 : 9 * d = Nat.gcd (m + 6) n) : d = 3 ∨ d = 6 :=
by
  sorry

end gcd_increase_by_9_l43_43494


namespace weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l43_43081

noncomputable def Wa : ℕ := 
  let volume_a := (3/5) * 4
  let volume_b := (2/5) * 4
  let weight_b := 700
  let total_weight := 3280
  (total_weight - (weight_b * volume_b)) / volume_a

theorem weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900 :
  Wa = 900 := 
by
  sorry

end weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l43_43081


namespace apples_total_l43_43855

def benny_apples : ℕ := 2
def dan_apples : ℕ := 9
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_total : total_apples = 11 :=
by
    sorry

end apples_total_l43_43855


namespace range_of_f_l43_43816

noncomputable def f (x : ℝ) : ℝ := if x = -2 then 0 else (x^2 + 5 * x + 6) / (x + 2)

theorem range_of_f :
  set.range f = { y : ℝ | y ≠ 1 } :=
begin
  sorry
end

end range_of_f_l43_43816


namespace find_density_of_gold_l43_43048

theorem find_density_of_gold
  (side_length : ℝ)
  (gold_cost_per_gram : ℝ)
  (sale_factor : ℝ)
  (profit : ℝ)
  (density_of_gold : ℝ) :
  side_length = 6 →
  gold_cost_per_gram = 60 →
  sale_factor = 1.5 →
  profit = 123120 →
  density_of_gold = 19 :=
sorry

end find_density_of_gold_l43_43048


namespace min_value_of_quadratic_l43_43552

theorem min_value_of_quadratic :
  ∀ (x : ℝ), ∃ (z : ℝ), z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∃ c : ℝ, c = c → z = 12) :=
by
  sorry

end min_value_of_quadratic_l43_43552


namespace radius_ratio_l43_43839

variable (VL VS rL rS : ℝ)
variable (hVL : VL = 432 * Real.pi)
variable (hVS : VS = 0.275 * VL)

theorem radius_ratio (h1 : (4 / 3) * Real.pi * rL^3 = VL)
                     (h2 : (4 / 3) * Real.pi * rS^3 = VS) :
  rS / rL = 2 / 3 := by
  sorry

end radius_ratio_l43_43839


namespace g_of_minus_3_l43_43255

noncomputable def f (x : ℝ) : ℝ := 4 * x - 7
noncomputable def g (y : ℝ) : ℝ := 3 * ((y + 7) / 4) ^ 2 + 4 * ((y + 7) / 4) + 1

theorem g_of_minus_3 : g (-3) = 8 :=
by
  sorry

end g_of_minus_3_l43_43255


namespace total_cost_l43_43771

variables (p e n : ℕ) -- represent the costs of pencil, eraser, and notebook in cents

-- Given conditions
def conditions : Prop :=
  9 * p + 7 * e + 4 * n = 220 ∧
  p > n ∧ n > e ∧ e > 0

-- Prove the total cost
theorem total_cost (h : conditions p e n) : p + n + e = 26 :=
sorry

end total_cost_l43_43771


namespace bananas_to_pears_ratio_l43_43805

theorem bananas_to_pears_ratio (B P : ℕ) (hP : P = 50) (h1 : B + 10 = 160) (h2: ∃ k : ℕ, B = k * P) : B / P = 3 :=
by
  -- proof steps would go here
  sorry

end bananas_to_pears_ratio_l43_43805


namespace ratio_alan_to_ben_l43_43922

theorem ratio_alan_to_ben (A B L : ℕ) (hA : A = 48) (hL : L = 36) (hB : B = L / 3) : A / B = 4 := by
  sorry

end ratio_alan_to_ben_l43_43922


namespace xyz_product_neg4_l43_43206

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l43_43206


namespace sqrt_y_to_the_fourth_eq_256_l43_43674

theorem sqrt_y_to_the_fourth_eq_256 (y : ℝ) (h : (sqrt y)^4 = 256) : y = 16 := by
  sorry

end sqrt_y_to_the_fourth_eq_256_l43_43674


namespace two_point_three_five_as_fraction_l43_43452

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43452


namespace odd_n_divides_3n_plus_1_is_1_l43_43998

theorem odd_n_divides_3n_plus_1_is_1 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) (h3 : n ∣ 3^n + 1) : n = 1 :=
sorry

end odd_n_divides_3n_plus_1_is_1_l43_43998


namespace circular_garden_radius_l43_43117

theorem circular_garden_radius (r : ℝ) (h1 : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) : r = 12 :=
by sorry

end circular_garden_radius_l43_43117


namespace surface_area_of_cylinder_with_square_cross_section_l43_43021

theorem surface_area_of_cylinder_with_square_cross_section
  (side_length : ℝ) (h1 : side_length = 2) : 
  (2 * Real.pi * 2 + 2 * Real.pi * 1^2) = 6 * Real.pi :=
by
  rw [←h1]
  sorry

end surface_area_of_cylinder_with_square_cross_section_l43_43021


namespace yogurt_cost_l43_43276

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l43_43276


namespace problem_a_problem_b_problem_c_l43_43627

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l43_43627


namespace combination_formula_l43_43147

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l43_43147


namespace walter_exceptional_days_l43_43002

theorem walter_exceptional_days :
  ∃ (w b : ℕ), 
  b + w = 10 ∧ 
  3 * b + 5 * w = 36 ∧ 
  w = 3 :=
by
  sorry

end walter_exceptional_days_l43_43002


namespace count_paths_no_consecutive_restriction_l43_43357

theorem count_paths_no_consecutive_restriction : 
  let total_steps := 13,
      steps_up := 6,
      steps_right := 7
  in binomial total_steps steps_up = 1716 :=
by
  let total_steps := 13
  let steps_up := 6
  let steps_right := 7
  show binomial total_steps steps_up = 1716
  sorry

end count_paths_no_consecutive_restriction_l43_43357


namespace javier_first_throw_distance_l43_43906

-- Definitions based on the conditions
def distance_second_throw : ℕ := 150  -- As solved in the solution, the second throw is 150 meters

theorem javier_first_throw_distance
  (distance_second_throw : ℕ)
  (h_first_throw : 2 * distance_second_throw = 2 * 150)
  (h_third_throw : 4 * distance_second_throw = 4 * 150)
  (h_sum_throws : 2 * distance_second_throw + distance_second_throw + 4 * distance_second_throw = 1050) :
  2 * distance_second_throw = 300 :=
by
  -- Introduce variables for the throw distances
  let distance_first_throw := 2 * distance_second_throw
  let distance_third_throw := 4 * distance_second_throw
  -- Use the provided hypothesis and solve for the first throw distance
  have h_sum : distance_first_throw + distance_second_throw + distance_third_throw = 1050,
    from h_sum_throws
  sorry

end javier_first_throw_distance_l43_43906


namespace tiles_finite_initial_segment_l43_43069

theorem tiles_finite_initial_segment (S : ℕ → Prop) (hTiling : ∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧ S m) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → S n :=
by
  sorry

end tiles_finite_initial_segment_l43_43069


namespace area_of_triangle_ABC_l43_43861

variable (A B C : ℝ × ℝ)
variable (x1 y1 x2 y2 x3 y3 : ℝ)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC :
  let A := (1, 2)
  let B := (-2, 5)
  let C := (4, -2)
  area_of_triangle A B C = 1.5 :=
by
  sorry

end area_of_triangle_ABC_l43_43861


namespace domain_of_fx_l43_43933

theorem domain_of_fx {x : ℝ} : (2 * x) / (x - 1) = (2 * x) / (x - 1) ↔ x ∈ {y : ℝ | y ≠ 1} :=
by
  sorry

end domain_of_fx_l43_43933


namespace max_smoothie_servings_l43_43526

-- Define the constants based on the problem conditions
def servings_per_recipe := 4
def bananas_per_recipe := 3
def yogurt_per_recipe := 1 -- cup
def honey_per_recipe := 2 -- tablespoons
def strawberries_per_recipe := 2 -- cups

-- Define the total amount of ingredients Lynn has
def total_bananas := 12
def total_yogurt := 6 -- cups
def total_honey := 16 -- tablespoons (since 1 cup = 16 tablespoons)
def total_strawberries := 8 -- cups

-- Define the calculation for the number of servings each ingredient can produce
def servings_from_bananas := (total_bananas / bananas_per_recipe) * servings_per_recipe
def servings_from_yogurt := (total_yogurt / yogurt_per_recipe) * servings_per_recipe
def servings_from_honey := (total_honey / honey_per_recipe) * servings_per_recipe
def servings_from_strawberries := (total_strawberries / strawberries_per_recipe) * servings_per_recipe

-- Define the minimum number of servings that can be made based on all ingredients
def max_servings := min servings_from_bananas (min servings_from_yogurt (min servings_from_honey servings_from_strawberries))

theorem max_smoothie_servings : max_servings = 16 :=
by
  sorry

end max_smoothie_servings_l43_43526


namespace sides_of_figures_intersection_l43_43398

theorem sides_of_figures_intersection (n p q : ℕ) (h1 : p ≠ 0) (h2 : q ≠ 0) :
  p + q ≤ n + 4 :=
by sorry

end sides_of_figures_intersection_l43_43398


namespace balloons_remaining_each_friend_l43_43084

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end balloons_remaining_each_friend_l43_43084


namespace number_divided_by_four_l43_43285

variable (x : ℝ)

theorem number_divided_by_four (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by {
  sorry
}

end number_divided_by_four_l43_43285


namespace factorize_polynomial_l43_43712

theorem factorize_polynomial (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := 
by 
  sorry

end factorize_polynomial_l43_43712


namespace total_cost_of_mangoes_l43_43658

-- Definition of prices per dozen in one box
def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Number of dozens per box (constant for all boxes)
def dozens_per_box : ℕ := 10

-- Number of boxes
def number_of_boxes : ℕ := 36

-- Calculate the total cost of mangoes in all boxes
theorem total_cost_of_mangoes :
  (prices_per_dozen.sum * number_of_boxes = 3060) := by
  -- Proof goes here
  sorry

end total_cost_of_mangoes_l43_43658


namespace decimal_to_fraction_l43_43461

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43461


namespace product_xyz_equals_zero_l43_43198

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l43_43198


namespace linear_term_coefficient_l43_43550

theorem linear_term_coefficient : (x - 1) * (1 / x + x) ^ 6 = a + b * x + c * x^2 + d * x^3 + e * x^4 + f * x^5 + g * x^6 →
  b = 20 :=
by
  sorry

end linear_term_coefficient_l43_43550


namespace smallest_prime_dividing_sum_l43_43671

theorem smallest_prime_dividing_sum (h1 : Odd 7) (h2 : Odd 9) 
    (h3 : ∀ {a b : ℤ}, Odd a → Odd b → Even (a + b)) :
  ∃ p : ℕ, Prime p ∧ p ∣ (7 ^ 15 + 9 ^ 7) ∧ p = 2 := 
by
  sorry

end smallest_prime_dividing_sum_l43_43671


namespace boys_without_glasses_l43_43608

def total_students_with_glasses : ℕ := 36
def girls_with_glasses : ℕ := 21
def total_boys : ℕ := 30

theorem boys_without_glasses :
  total_boys - (total_students_with_glasses - girls_with_glasses) = 15 :=
by
  sorry

end boys_without_glasses_l43_43608


namespace quadratic_roots_properties_quadratic_roots_max_min_l43_43022

theorem quadratic_roots_properties (k : ℝ) (h : 2 ≤ k ∧ k ≤ 8)
  (x1 x2 : ℝ) (h_roots : x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) :
  (x1^2 + x2^2) = 16 * k - 30 :=
sorry

theorem quadratic_roots_max_min :
  (∀ k ∈ { k : ℝ | 2 ≤ k ∧ k ≤ 8 }, 
    ∃ (x1 x2 : ℝ), 
      (x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) 
      ∧ (x1^2 + x2^2) = (if k = 8 then 98 else if k = 2 then 2 else 16 * k - 30)) :=
sorry

end quadratic_roots_properties_quadratic_roots_max_min_l43_43022


namespace staircase_problem_l43_43432

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem staircase_problem (total_steps required_steps : ℕ) (num_two_steps : ℕ) :
  total_steps = 11 ∧ required_steps = 7 ∧ num_two_steps = 4 →
  C 7 4 = 35 :=
by
  intro h
  sorry

end staircase_problem_l43_43432


namespace polygon_interior_angle_144_proof_l43_43976

-- Definitions based on the conditions in the problem statement
def interior_angle (n : ℕ) : ℝ := 144
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- The problem statement as a Lean 4 theorem to prove n = 10
theorem polygon_interior_angle_144_proof : ∃ n : ℕ, interior_angle n = 144 ∧ sum_of_interior_angles n = n * 144 → n = 10 := by
  sorry

end polygon_interior_angle_144_proof_l43_43976


namespace range_of_a_l43_43590

theorem range_of_a (a : ℝ) :
  (∃ (M : ℝ × ℝ), (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧
    (M.1)^2 + (M.2 - 2)^2 + (M.1)^2 + (M.2)^2 = 10) → 
  0 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l43_43590


namespace fraction_equality_l43_43032

-- Defining the main problem statement
theorem fraction_equality (x y z : ℚ) (k : ℚ) 
  (h1 : x = 3 * k) (h2 : y = 5 * k) (h3 : z = 7 * k) :
  (y + z) / (3 * x - y) = 3 :=
by
  sorry

end fraction_equality_l43_43032


namespace largest_n_satisfying_equation_l43_43551

theorem largest_n_satisfying_equation :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ ∀ n : ℕ,
  (n * n = x * x + y * y + z * z + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 2 :=
by
  sorry

end largest_n_satisfying_equation_l43_43551


namespace ted_candy_bars_l43_43257

theorem ted_candy_bars (b : ℕ) (n : ℕ) (h : b = 5) (h2 : n = 3) : b * n = 15 :=
by
  sorry

end ted_candy_bars_l43_43257


namespace decimal_to_fraction_l43_43477

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43477


namespace two_point_three_five_as_fraction_l43_43450

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43450


namespace uma_fraction_part_l43_43400

theorem uma_fraction_part (r s t u : ℕ) 
  (hr : r = 6) 
  (hs : s = 5) 
  (ht : t = 7) 
  (hu : u = 8) 
  (shared_amount: ℕ)
  (hr_amount: shared_amount = r / 6)
  (hs_amount: shared_amount = s / 5)
  (ht_amount: shared_amount = t / 7)
  (hu_amount: shared_amount = u / 8) :
  ∃ total : ℕ, ∃ uma_total : ℕ, uma_total * 13 = 2 * total :=
sorry

end uma_fraction_part_l43_43400


namespace infinite_geometric_series_common_ratio_l43_43851

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end infinite_geometric_series_common_ratio_l43_43851


namespace two_point_three_five_as_fraction_l43_43451

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43451


namespace inequality_ab_bc_ca_max_l43_43831

theorem inequality_ab_bc_ca_max (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|))
  ≤ 1 + (1 / 3) * (a + b + c)^2 := sorry

end inequality_ab_bc_ca_max_l43_43831


namespace leap_years_count_l43_43841

theorem leap_years_count :
  let is_leap_year (y : ℕ) := (y % 900 = 150 ∨ y % 900 = 450) ∧ y % 100 = 0
  let range_start := 2100
  let range_end := 4200
  ∃ L, L = [2250, 2850, 3150, 3750, 4050] ∧ (∀ y ∈ L, is_leap_year y ∧ range_start ≤ y ∧ y ≤ range_end)
  ∧ L.length = 5 :=
by
  sorry

end leap_years_count_l43_43841


namespace intersection_complement_eq_l43_43883

open Set

variable (U A B : Set ℕ)
  
theorem intersection_complement_eq : 
  U = {0, 1, 2, 3, 4} → 
  A = {0, 1, 3} → 
  B = {2, 3} → 
  A ∩ (U \ B) = {0, 1} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end intersection_complement_eq_l43_43883


namespace total_distance_100_l43_43820

-- Definitions for the problem conditions:
def initial_velocity : ℕ := 40
def common_difference : ℕ := 10
def total_time (v₀ : ℕ) (d : ℕ) : ℕ := (v₀ / d) + 1  -- The total time until the car stops
def distance_traveled (v₀ : ℕ) (d : ℕ) : ℕ :=
  (v₀ * total_time v₀ d) - (d * total_time v₀ d * (total_time v₀ d - 1)) / 2

-- Statement to prove:
theorem total_distance_100 : distance_traveled initial_velocity common_difference = 100 := by
  sorry

end total_distance_100_l43_43820


namespace max_mom_money_difference_l43_43920

theorem max_mom_money_difference:
  let tuesday_amount := 8 in
  let wednesday_amount := 5 * tuesday_amount in
  let thursday_amount := wednesday_amount + 9 in
  (thursday_amount - tuesday_amount = 41) :=
by
  sorry

end max_mom_money_difference_l43_43920


namespace investment_total_l43_43958

theorem investment_total (x y : ℝ) (h₁ : 0.08 * x + 0.05 * y = 490) (h₂ : x = 3000 ∨ y = 3000) : x + y = 8000 :=
by
  sorry

end investment_total_l43_43958


namespace math_problem_l43_43624

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l43_43624


namespace xyz_product_value_l43_43191

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l43_43191


namespace find_a_l43_43360

variables {a b c : ℤ}

theorem find_a (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 5) : a = 3 :=
by
  sorry

end find_a_l43_43360


namespace books_leftover_l43_43589

theorem books_leftover :
  (1500 * 45) % 47 = 13 :=
by
  sorry

end books_leftover_l43_43589


namespace sandra_savings_l43_43067

theorem sandra_savings :
  let num_notepads := 8
  let original_price_per_notepad := 3.75
  let discount_rate := 0.25
  let discount_per_notepad := original_price_per_notepad * discount_rate
  let discounted_price_per_notepad := original_price_per_notepad - discount_per_notepad
  let total_cost_without_discount := num_notepads * original_price_per_notepad
  let total_cost_with_discount := num_notepads * discounted_price_per_notepad
  let total_savings := total_cost_without_discount - total_cost_with_discount
  total_savings = 7.50 :=
sorry

end sandra_savings_l43_43067


namespace find_multiple_l43_43940

theorem find_multiple (x m : ℝ) (h₁ : 10 * x = m * x - 36) (h₂ : x = -4.5) : m = 2 :=
by
  sorry

end find_multiple_l43_43940


namespace decrypt_message_base7_l43_43695

noncomputable def base7_to_base10 : Nat := 
  2 * 343 + 5 * 49 + 3 * 7 + 4 * 1

theorem decrypt_message_base7 : base7_to_base10 = 956 := 
by 
  sorry

end decrypt_message_base7_l43_43695


namespace part_a_part_c_part_d_l43_43632

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l43_43632


namespace bus_speed_l43_43004

theorem bus_speed (S : ℝ) (h1 : 36 = S * (2 / 3)) : S = 54 :=
by
sorry

end bus_speed_l43_43004


namespace area_of_BCD_l43_43901

theorem area_of_BCD (S_ABC : ℝ) (a_CD : ℝ) (h_ratio : ℝ) (h_ABC : ℝ) :
  S_ABC = 36 ∧ a_CD = 30 ∧ h_ratio = 0.5 ∧ h_ABC = 12 → 
  (1 / 2) * a_CD * (h_ratio * h_ABC) = 90 :=
by
  intros h
  sorry

end area_of_BCD_l43_43901


namespace decimal_to_fraction_equivalence_l43_43478

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43478


namespace goose_eggs_laied_l43_43852

theorem goose_eggs_laied (z : ℕ) (hatch_rate : ℚ := 2 / 3) (first_month_survival_rate : ℚ := 3 / 4) 
  (first_year_survival_rate : ℚ := 2 / 5) (geese_survived_first_year : ℕ := 126) :
  (hatch_rate * z) = 420 ∧ (first_month_survival_rate * 315 = 315) ∧ (first_year_survival_rate * 315 = 126) →
  z = 630 :=
by
  sorry

end goose_eggs_laied_l43_43852


namespace decimal_to_fraction_equivalence_l43_43482

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l43_43482


namespace sum_a1_to_a14_equals_zero_l43_43349

theorem sum_a1_to_a14_equals_zero 
  (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 : ℝ) 
  (h1 : (1 + x - x^2)^3 * (1 - 2 * x^2)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 + a13 * x^13 + a14 * x^14) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 1) 
  (h3 : a = 1) : 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 0 := by
  sorry

end sum_a1_to_a14_equals_zero_l43_43349


namespace equivalent_proposition_l43_43824

theorem equivalent_proposition (H : Prop) (P : Prop) (Q : Prop) (hpq : H → P → ¬ Q) : (H → ¬ Q → ¬ P) :=
by
  intro h nq np
  sorry

end equivalent_proposition_l43_43824


namespace myopia_relation_l43_43131

def myopia_data := 
  [(1.00, 100), (0.50, 200), (0.25, 400), (0.20, 500), (0.10, 1000)]

noncomputable def myopia_function (x : ℝ) : ℝ :=
  100 / x

theorem myopia_relation (h₁ : 100 = (1.00 : ℝ) * 100)
    (h₂ : 100 = (0.50 : ℝ) * 200)
    (h₃ : 100 = (0.25 : ℝ) * 400)
    (h₄ : 100 = (0.20 : ℝ) * 500)
    (h₅ : 100 = (0.10 : ℝ) * 1000) :
  (∀ x > 0, myopia_function x = 100 / x) ∧ (myopia_function 250 = 0.4) :=
by
  sorry

end myopia_relation_l43_43131


namespace paint_brush_ratio_l43_43121

theorem paint_brush_ratio 
  (s w : ℝ) 
  (h1 : s > 0) 
  (h2 : w > 0) 
  (h3 : (1 / 2) * w ^ 2 + (1 / 2) * (s - w) ^ 2 = (s ^ 2) / 3) 
  : s / w = 3 + Real.sqrt 3 :=
sorry

end paint_brush_ratio_l43_43121


namespace words_memorized_l43_43252

theorem words_memorized (x y z : ℕ) (h1 : x = 4 * (y + z) / 5) (h2 : x + y = 6 * z / 5) (h3 : 100 < x + y + z ∧ x + y + z < 200) : 
  x + y + z = 198 :=
by
  sorry

end words_memorized_l43_43252


namespace sandy_siding_cost_l43_43247

theorem sandy_siding_cost:
  let wall_width := 8
  let wall_height := 8
  let roof_width := 8
  let roof_height := 5
  let siding_width := 10
  let siding_height := 12
  let siding_cost := 30
  let wall_area := wall_width * wall_height
  let roof_side_area := roof_width * roof_height
  let roof_area := 2 * roof_side_area
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let required_sections := (total_area + siding_area - 1) / siding_area -- ceiling division
  let total_cost := required_sections * siding_cost
  total_cost = 60 :=
by
  sorry

end sandy_siding_cost_l43_43247


namespace half_angle_in_quadrant_l43_43178

theorem half_angle_in_quadrant (α : ℝ) (k : ℤ) (h : k * 360 + 90 < α ∧ α < k * 360 + 180) :
  ∃ n : ℤ, (n * 360 + 45 < α / 2 ∧ α / 2 < n * 360 + 90) ∨ (n * 360 + 225 < α / 2 ∧ α / 2 < n * 360 + 270) :=
by sorry

end half_angle_in_quadrant_l43_43178


namespace factorial_div_combination_l43_43159

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l43_43159


namespace print_time_correct_l43_43523

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l43_43523


namespace total_weight_is_correct_l43_43661

noncomputable def A (B : ℝ) : ℝ := 12 + (1/2) * B
noncomputable def B (C : ℝ) : ℝ := 8 + (1/3) * C
noncomputable def C (A : ℝ) : ℝ := 20 + 2 * A
noncomputable def NewWeightB (A B : ℝ) : ℝ := B + 0.15 * A
noncomputable def NewWeightA (A C : ℝ) : ℝ := A - 0.10 * C

theorem total_weight_is_correct (B C : ℝ) (h1 : A B = (C - 20) / 2)
  (h2 : B = 8 + (1/3) * C) 
  (h3 : C = 20 + 2 * A B) 
  (h4 : NewWeightB (A B) B = 38.35) 
  (h5 : NewWeightA (A B) C = 21.2) :
  NewWeightA (A B) C + NewWeightB (A B) B + C = 139.55 :=
sorry

end total_weight_is_correct_l43_43661


namespace expression_equality_l43_43345

-- Define the conditions
variables {a b x : ℝ}
variable (h1 : x = a / b)
variable (h2 : a ≠ 2 * b)
variable (h3 : b ≠ 0)

-- Define and state the theorem
theorem expression_equality : (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) :=
by 
  intros
  sorry

end expression_equality_l43_43345


namespace negation_proposition_l43_43423

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_proposition_l43_43423


namespace range_of_3a_minus_b_l43_43718

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 2 ≤ a + b ∧ a + b ≤ 5) (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
    -2 ≤ 3 * a - b ∧ 3 * a - b ≤ 7 := 
by 
  sorry

end range_of_3a_minus_b_l43_43718


namespace percentage_of_books_returned_l43_43122

theorem percentage_of_books_returned
  (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) (returned_books_percentage : ℚ) 
  (h1 : initial_books = 75) 
  (h2 : end_books = 68) 
  (h3 : loaned_books = 20)
  (h4 : returned_books_percentage = (end_books - (initial_books - loaned_books)) * 100 / loaned_books):
  returned_books_percentage = 65 := 
by
  sorry

end percentage_of_books_returned_l43_43122


namespace parallel_sufficient_not_necessary_l43_43101

def line := Type
def parallel (l1 l2 : line) : Prop := sorry
def in_plane (l : line) : Prop := sorry

theorem parallel_sufficient_not_necessary (a β : line) :
  (parallel a β → ∃ γ, in_plane γ ∧ parallel a γ) ∧
  ¬( (∃ γ, in_plane γ ∧ parallel a γ) → parallel a β ) :=
by sorry

end parallel_sufficient_not_necessary_l43_43101


namespace find_r_l43_43161

def cubic_function (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_r (p q r : ℝ) (h1 : cubic_function p q r (-1) = 0) :
  r = p - 2 :=
sorry

end find_r_l43_43161


namespace tangent_slope_at_one_l43_43426

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x + Real.sqrt x

theorem tangent_slope_at_one :
  (deriv f 1) = 3 / 2 :=
by
  sorry

end tangent_slope_at_one_l43_43426


namespace find_k_l43_43836

def f (n : ℤ) : ℤ :=
if n % 2 = 0 then n / 2 else n + 3

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_f_f_f_k : f (f (f k)) = 27) : k = 105 := by
  sorry

end find_k_l43_43836


namespace bags_of_chips_count_l43_43657

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end bags_of_chips_count_l43_43657


namespace peter_remaining_money_l43_43062

def initial_amount : Float := 500.0 
def sales_tax : Float := 0.05
def discount : Float := 0.10

def calculate_cost_with_tax (price_per_kilo: Float) (quantity: Float) (tax_rate: Float) : Float :=
  quantity * price_per_kilo * (1 + tax_rate)

def calculate_cost_with_discount (price_per_kilo: Float) (quantity: Float) (discount_rate: Float) : Float :=
  quantity * price_per_kilo * (1 - discount_rate)

def total_first_trip : Float :=
  calculate_cost_with_tax 2.0 6 sales_tax +
  calculate_cost_with_tax 3.0 9 sales_tax +
  calculate_cost_with_tax 4.0 5 sales_tax +
  calculate_cost_with_tax 5.0 3 sales_tax +
  calculate_cost_with_tax 3.50 2 sales_tax +
  calculate_cost_with_tax 4.25 7 sales_tax +
  calculate_cost_with_tax 6.0 4 sales_tax +
  calculate_cost_with_tax 5.50 8 sales_tax

def total_second_trip : Float :=
  calculate_cost_with_discount 1.50 2 discount +
  calculate_cost_with_discount 2.75 5 discount

def remaining_money (initial: Float) (first_trip: Float) (second_trip: Float) : Float :=
  initial - first_trip - second_trip

theorem peter_remaining_money : remaining_money initial_amount total_first_trip total_second_trip = 297.24 := 
  by
    -- Proof omitted
    sorry

end peter_remaining_money_l43_43062


namespace sampling_interval_is_9_l43_43505

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end sampling_interval_is_9_l43_43505


namespace eval_expression_l43_43162

open Real

noncomputable def e : ℝ := 2.71828

theorem eval_expression : abs (5 * e - 15) = 1.4086 := by
  sorry

end eval_expression_l43_43162


namespace geometric_series_common_ratio_l43_43848

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end geometric_series_common_ratio_l43_43848


namespace coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l43_43693

/- The problem's conditions and questions rephrased for Lean:
  1. Prove: if \( n \) is divisible by 3, then a valid coloring is possible.
  2. Prove: if a valid coloring is possible, then \( n \) is divisible by 3.
-/

def is_colorable (n : ℕ) : Prop :=
  ∃ (colors : Fin 3 → Fin n → Fin 3),
    ∀ (i j : Fin n), i ≠ j → (colors 0 i ≠ colors 0 j ∧ colors 1 i ≠ colors 1 j ∧ colors 2 i ≠ colors 2 j)

theorem coloring_possible_if_divisible_by_three (n : ℕ) (h : n % 3 = 0) : is_colorable n :=
  sorry

theorem divisible_by_three_if_coloring_possible (n : ℕ) (h : is_colorable n) : n % 3 = 0 :=
  sorry

end coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l43_43693


namespace total_number_of_workers_l43_43679

variables (W N : ℕ)
variables (average_salary_workers average_salary_techs average_salary_non_techs : ℤ)
variables (num_techs total_salary total_salary_techs total_salary_non_techs : ℤ)

theorem total_number_of_workers (h1 : average_salary_workers = 8000)
                               (h2 : average_salary_techs = 14000)
                               (h3 : num_techs = 7)
                               (h4 : average_salary_non_techs = 6000)
                               (h5 : total_salary = W * 8000)
                               (h6 : total_salary_techs = 7 * 14000)
                               (h7 : total_salary_non_techs = N * 6000)
                               (h8 : total_salary = total_salary_techs + total_salary_non_techs)
                               (h9 : W = 7 + N) : 
                               W = 28 :=
sorry

end total_number_of_workers_l43_43679


namespace relay_team_order_count_l43_43592

def num_ways_to_order_relay (total_members : Nat) (jordan_lap : Nat) : Nat :=
  if jordan_lap = total_members then (total_members - 1).factorial else 0

theorem relay_team_order_count : num_ways_to_order_relay 5 5 = 24 :=
by
  -- the proof would go here
  sorry

end relay_team_order_count_l43_43592


namespace number_of_lines_dist_l43_43895

theorem number_of_lines_dist {A B : ℝ × ℝ} (hA : A = (3, 0)) (hB : B = (0, 4)) : 
  ∃ n : ℕ, n = 3 ∧
  ∀ l : ℝ → ℝ → Prop, 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ A → dist A p = 2) ∧ 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ B → dist B p = 3) → n = 3 := 
by sorry

end number_of_lines_dist_l43_43895


namespace pencils_to_sell_l43_43843

/--
A store owner bought 1500 pencils at $0.10 each. 
Each pencil is sold for $0.25. 
He wants to make a profit of exactly $100. 
Prove that he must sell 1000 pencils to achieve this profit.
-/
theorem pencils_to_sell (total_pencils : ℕ) (cost_per_pencil : ℝ) (selling_price_per_pencil : ℝ) (desired_profit : ℝ)
  (h1 : total_pencils = 1500)
  (h2 : cost_per_pencil = 0.10)
  (h3 : selling_price_per_pencil = 0.25)
  (h4 : desired_profit = 100) :
  total_pencils * cost_per_pencil + desired_profit = 1000 * selling_price_per_pencil :=
by
  -- Since Lean code requires some proof content, we put sorry to skip it.
  sorry

end pencils_to_sell_l43_43843


namespace cost_formula_correct_l43_43844

def cost_of_ride (T : ℤ) : ℤ :=
  if T > 5 then 10 + 5 * T - 10 else 10 + 5 * T

theorem cost_formula_correct (T : ℤ) : cost_of_ride T = 10 + 5 * T - (if T > 5 then 10 else 0) := by
  sorry

end cost_formula_correct_l43_43844


namespace problem1_problem2_l43_43137

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l43_43137


namespace maria_gave_towels_l43_43287

def maria_towels (green_white total_left : Nat) : Nat :=
  green_white - total_left

theorem maria_gave_towels :
  ∀ (green white left given : Nat),
    green = 35 →
    white = 21 →
    left = 22 →
    given = 34 →
    maria_towels (green + white) left = given :=
by
  intros green white left given
  intros hgreen hwhite hleft hgiven
  rw [hgreen, hwhite, hleft, hgiven]
  sorry

end maria_gave_towels_l43_43287


namespace sum_arithmetic_series_l43_43819

theorem sum_arithmetic_series :
  let a1 := 1000
  let an := 5000
  let d := 4
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 3003000 := by
    sorry

end sum_arithmetic_series_l43_43819


namespace arithmetic_sequence_properties_l43_43175

variable {a : ℕ → ℕ}
variable {n : ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_sequence_properties 
  (a_3_eq_7 : a 3 = 7)
  (a_5_plus_a_7_eq_26 : a 5 + a 7 = 26) :
  (∃ a1 d, (a 1 = a1) ∧ (∀ n, a n = a1 + (n - 1) * d) ∧ d = 2) ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ S_n, S_n = n^2 + 2 * n) ∧ 
  ∀ T_n n, (∃ b : (ℕ → ℕ) → ℕ → ℕ, b a n = 1 / (a n ^ 2 - 1)) 
  → T_n = n / (4 * (n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l43_43175


namespace unfolded_side_view_of_cone_is_sector_l43_43080

theorem unfolded_side_view_of_cone_is_sector 
  (shape : Type)
  (curved_side : shape)
  (straight_side1 : shape)
  (straight_side2 : shape) 
  (condition1 : ∃ (s : shape), s = curved_side) 
  (condition2 : ∃ (s1 s2 : shape), s1 = straight_side1 ∧ s2 = straight_side2)
  : shape = sector :=
sorry

end unfolded_side_view_of_cone_is_sector_l43_43080


namespace louis_age_l43_43039

variable (L J M : ℕ) -- L for Louis, J for Jerica, and M for Matilda

theorem louis_age : 
  (M = 35) ∧ (M = J + 7) ∧ (J = 2 * L) → L = 14 := 
by 
  intro h 
  sorry

end louis_age_l43_43039


namespace combined_weight_of_new_students_l43_43411

theorem combined_weight_of_new_students 
  (avg_weight_orig : ℝ) (num_students_orig : ℝ) 
  (new_avg_weight : ℝ) (num_new_students : ℝ) 
  (total_weight_gain_orig : ℝ) (total_weight_loss_orig : ℝ)
  (total_weight_orig : ℝ := avg_weight_orig * num_students_orig) 
  (net_weight_change_orig : ℝ := total_weight_gain_orig - total_weight_loss_orig)
  (total_weight_after_change_orig : ℝ := total_weight_orig + net_weight_change_orig) 
  (total_students_after : ℝ := num_students_orig + num_new_students) 
  (total_weight_class_after : ℝ := new_avg_weight * total_students_after) : 
  total_weight_class_after - total_weight_after_change_orig = 586 :=
by
  sorry

end combined_weight_of_new_students_l43_43411


namespace largest_multiple_of_7_smaller_than_neg_85_l43_43093

theorem largest_multiple_of_7_smaller_than_neg_85 :
  ∃ k : ℤ, 7 * k < -85 ∧ (∀ m : ℤ, 7 * m < -85 → 7 * m ≤ 7 * k) ∧ 7 * k = -91 :=
by
  simp only [exists_prop, and.assoc],
  sorry

end largest_multiple_of_7_smaller_than_neg_85_l43_43093


namespace decimal_to_fraction_l43_43488

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l43_43488


namespace evaluate_expression_l43_43601

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 240 / 961 := 
by 
  sorry

end evaluate_expression_l43_43601


namespace distance_from_center_to_line_l43_43798

-- Define the circle and its center
def is_circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def center : (ℝ × ℝ) := (1, 0)

-- Define the line equation y = tan(30°) * x
def is_line (x y : ℝ) : Prop := y = (1 / Real.sqrt 3) * x

-- Function to compute the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C)) / Real.sqrt (A^2 + B^2)

-- The main theorem to be proven:
theorem distance_from_center_to_line : 
  distance_point_to_line center (1 / Real.sqrt 3) (-1) 0 = 1 / 2 :=
  sorry

end distance_from_center_to_line_l43_43798


namespace range_of_a_l43_43176

noncomputable
def proposition_p (x : ℝ) : Prop := abs (x - (3 / 4)) <= (1 / 4)
noncomputable
def proposition_q (x a : ℝ) : Prop := (x - a) * (x - a - 1) <= 0

theorem range_of_a :
  (∀ x : ℝ, proposition_p x → ∃ x : ℝ, proposition_q x a) ∧
  (∃ x : ℝ, ¬(proposition_p x → proposition_q x a )) →
  0 ≤ a ∧ a ≤ (1 / 2) :=
sorry

end range_of_a_l43_43176


namespace price_verification_l43_43830

noncomputable def price_on_hot_day : ℚ :=
  let P : ℚ := 225 / 172
  1.25 * P

theorem price_verification :
  (32 * 7 * (225 / 172) + 32 * 3 * (1.25 * (225 / 172)) - (32 * 10 * 0.75)) = 210 :=
sorry

end price_verification_l43_43830


namespace cos_double_beta_eq_24_over_25_l43_43560

theorem cos_double_beta_eq_24_over_25
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.cos (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (π / 2) π) :
  Real.cos (2 * β) = 24 / 25 := sorry

end cos_double_beta_eq_24_over_25_l43_43560


namespace range_of_a_l43_43352

noncomputable def f (x a : ℝ) := x^2 - a * x
noncomputable def g (x : ℝ) := Real.exp x
noncomputable def h (x : ℝ) := x - (Real.log x / x)

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (f x a = Real.log x)) ↔ (1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1) :=
by
  sorry

end range_of_a_l43_43352


namespace shaded_region_area_l43_43541

noncomputable def area_of_shaded_region (dodecagon_side_length: ℝ) (hexagon_side_length: ℝ): ℝ :=
  let base1 := dodecagon_side_length
  let height1 := dodecagon_side_length
  let area_triangle1 := 1/2 * base1 * height1
  
  let base2 := hexagon_side_length
  let height2 := hexagon_side_length / 2
  let area_triangle2 := 1/2 * base2 * height2

  3 * (area_triangle1 + area_triangle2)

theorem shaded_region_area :
  area_of_shaded_region 12 12 = 324 :=
by
  sorry

end shaded_region_area_l43_43541


namespace email_sequence_correct_l43_43003

theorem email_sequence_correct :
    ∀ (a b c d e f : Prop),
    (a → (e → (b → (c → (d → f))))) :=
by 
  sorry

end email_sequence_correct_l43_43003


namespace inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l43_43832

variables {x y g : ℝ}
variables (hx : 0 < x) (hy : 0 < y)
variable (hg : g = Real.sqrt (x * y))

theorem inf_geometric_mean_gt_3 :
  g ≥ 3 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) :=
by
  sorry

theorem inf_geometric_mean_le_2 :
  g ≤ 2 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) :=
by
  sorry

end inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l43_43832


namespace triangle_third_side_l43_43758

theorem triangle_third_side (DE DF : ℝ) (E F : ℝ) (EF : ℝ) 
    (h₁ : DE = 7) 
    (h₂ : DF = 21) 
    (h₃ : E = 3 * F) : EF = 14 * Real.sqrt 2 :=
sorry

end triangle_third_side_l43_43758


namespace part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l43_43503

-- Define the game rules and conditions for the proof
def takeMatches (total_matches : Nat) (taken_matches : Nat) : Nat :=
  total_matches - taken_matches

-- Part (a) statement
theorem part_a_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (b) statement
theorem part_b_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (c) generalized statement for game type (a)
theorem part_c_winner_a (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

-- Part (c) generalized statement for game type (b)
theorem part_c_winner_b (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

end part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l43_43503


namespace probability_of_D_given_T_l43_43928

-- Definitions based on the conditions given in the problem.
def pr_D : ℚ := 1 / 400
def pr_Dc : ℚ := 399 / 400
def pr_T_given_D : ℚ := 1
def pr_T_given_Dc : ℚ := 0.05
def pr_T : ℚ := pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Statement to prove 
theorem probability_of_D_given_T : pr_T ≠ 0 → (pr_T_given_D * pr_D) / pr_T = 20 / 419 :=
by
  intros h1
  unfold pr_T pr_D pr_Dc pr_T_given_D pr_T_given_Dc
  -- Mathematical steps are skipped in Lean by inserting sorry
  sorry

-- Check that the statement can be built successfully
example : pr_D = 1 / 400 := by rfl
example : pr_Dc = 399 / 400 := by rfl
example : pr_T_given_D = 1 := by rfl
example : pr_T_given_Dc = 0.05 := by rfl
example : pr_T = (1 * (1 / 400) + 0.05 * (399 / 400)) := by rfl

end probability_of_D_given_T_l43_43928


namespace problem1_problem2_problem3_problem4_problem5_problem6_l43_43652

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l43_43652


namespace factorial_div_eq_l43_43151

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l43_43151


namespace min_value_of_k_l43_43384

open Finset

variable {α : Type*} [DecidableEq α]

def symmetric_difference (A B : Finset α) := (A \ B) ∪ (B \ A)

theorem min_value_of_k (S T : Finset α) (hS : S.nonempty) (hT : T.nonempty) (h : (symmetric_difference S T).card = 1) : 
  (S.card + T.card) = 3 :=
sorry

end min_value_of_k_l43_43384


namespace delores_initial_money_l43_43985

def computer_price : ℕ := 400
def printer_price : ℕ := 40
def headphones_price : ℕ := 60
def discount_percentage : ℕ := 10
def left_money : ℕ := 10

theorem delores_initial_money :
  ∃ initial_money : ℕ,
    initial_money = printer_price + headphones_price + (computer_price - (discount_percentage * computer_price / 100)) + left_money :=
  sorry

end delores_initial_money_l43_43985


namespace sum_series_eq_two_l43_43321

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l43_43321


namespace factorial_quotient_l43_43153

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l43_43153


namespace tangent_expression_equals_two_l43_43833

noncomputable def eval_tangent_expression : ℝ :=
  (1 + Real.tan (3 * Real.pi / 180)) * (1 + Real.tan (42 * Real.pi / 180))

theorem tangent_expression_equals_two :
  eval_tangent_expression = 2 :=
by sorry

end tangent_expression_equals_two_l43_43833


namespace remainder_of_86_l43_43582

theorem remainder_of_86 {m : ℕ} (h1 : m ≠ 1) 
  (h2 : 69 % m = 90 % m) (h3 : 90 % m = 125 % m) : 86 % m = 2 := 
by
  sorry

end remainder_of_86_l43_43582


namespace decimal_to_fraction_l43_43442

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43442


namespace largest_d_l43_43350

variable (a b c d : ℝ)

theorem largest_d (h : a + 1 = b - 2 ∧ b - 2 = c + 3 ∧ c + 3 = d - 4) : 
  d >= a ∧ d >= b ∧ d >= c :=
by
  sorry

end largest_d_l43_43350


namespace volume_of_fuel_A_l43_43977

variables (V_A V_B : ℝ)

def condition1 := V_A + V_B = 212
def condition2 := 0.12 * V_A + 0.16 * V_B = 30

theorem volume_of_fuel_A :
  condition1 V_A V_B → condition2 V_A V_B → V_A = 98 :=
by
  intros h1 h2
  sorry

end volume_of_fuel_A_l43_43977


namespace hexagon_monochromatic_triangle_l43_43711

-- Define the probability of monochromatic triangles in a hexagon
noncomputable def probability_monochromatic_triangle : ℝ := 0.99683

-- Define the main theorem to prove the probability condition
theorem hexagon_monochromatic_triangle :
  ∃ (G : SimpleGraph (Fin 6)),
    (∀ e ∈ G.edgeSet, e.color = Green ∨ e.color = Yellow) →
    (random_colored_prob G ≈ probability_monochromatic_triangle) :=
by
  sorry

end hexagon_monochromatic_triangle_l43_43711


namespace baseball_card_total_percent_decrease_l43_43504

theorem baseball_card_total_percent_decrease :
  ∀ (original_value first_year_decrease second_year_decrease : ℝ),
  first_year_decrease = 0.60 →
  second_year_decrease = 0.10 →
  original_value > 0 →
  (original_value - original_value * first_year_decrease - (original_value * (1 - first_year_decrease)) * second_year_decrease) =
  original_value * (1 - 0.64) :=
by
  intros original_value first_year_decrease second_year_decrease h_first_year h_second_year h_original_pos
  sorry

end baseball_card_total_percent_decrease_l43_43504


namespace problem_l43_43034

theorem problem (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 6 = 976 :=
by
  sorry

end problem_l43_43034


namespace greatest_sum_of_other_two_roots_l43_43939

noncomputable def polynomial (x : ℝ) (k : ℝ) : ℝ :=
  x^3 - k * x^2 + 20 * x - 15

theorem greatest_sum_of_other_two_roots (k x1 x2 : ℝ) (h : polynomial 3 k = 0) (hx : x1 * x2 = 5)
  (h_prod_sum : 3 * x1 + 3 * x2 + x1 * x2 = 20) : x1 + x2 = 5 :=
by
  sorry

end greatest_sum_of_other_two_roots_l43_43939


namespace calculate_students_l43_43918

noncomputable def handshakes (m n : ℕ) : ℕ :=
  1/2 * (4 * 3 + 5 * (2 * (m - 2) + 2 * (n - 2)) + 8 * (m - 2) * (n - 2))

theorem calculate_students (m n : ℕ) (h_m : 3 ≤ m) (h_n : 3 ≤ n) (h_handshakes : handshakes m n = 1020) : m * n = 140 :=
by
  sorry

end calculate_students_l43_43918


namespace solution_set_product_positive_l43_43180

variable {R : Type*} [LinearOrderedField R]

def is_odd (f : R → R) : Prop := ∀ x : R, f (-x) = -f (x)

variable (f g : R → R)

noncomputable def solution_set_positive_f : Set R := { x | 4 < x ∧ x < 10 }
noncomputable def solution_set_positive_g : Set R := { x | 2 < x ∧ x < 5 }

theorem solution_set_product_positive :
  is_odd f →
  is_odd g →
  (∀ x, f x > 0 ↔ x ∈ solution_set_positive_f) →
  (∀ x, g x > 0 ↔ x ∈ solution_set_positive_g) →
  { x | f x * g x > 0 } = { x | (4 < x ∧ x < 5) ∨ (-5 < x ∧ x < -4) } :=
by
  sorry

end solution_set_product_positive_l43_43180


namespace cube_root_sum_is_integer_l43_43828

theorem cube_root_sum_is_integer :
  let a := (2 + (10 / 9) * Real.sqrt 3)^(1/3)
  let b := (2 - (10 / 9) * Real.sqrt 3)^(1/3)
  a + b = 2 := by
  sorry

end cube_root_sum_is_integer_l43_43828


namespace systematic_sampling_interval_l43_43433

theorem systematic_sampling_interval 
  (N : ℕ) (n : ℕ) (hN : N = 630) (hn : n = 45) :
  N / n = 14 :=
by {
  sorry
}

end systematic_sampling_interval_l43_43433


namespace inscribed_circle_radius_l43_43669

theorem inscribed_circle_radius (A B C : Point) (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 8) (hBC : BC = 5) : 
  radius_of_inscribed_circle_in_triangle ABC AB AC BC = 76 * Real.sqrt 10 / 21 :=
by {
  sorry -- Proof goes here.
}

end inscribed_circle_radius_l43_43669


namespace find_number_of_hens_l43_43511

def hens_and_cows_problem (H C : ℕ) : Prop :=
  (H + C = 50) ∧ (2 * H + 4 * C = 144)

theorem find_number_of_hens (H C : ℕ) (hc : hens_and_cows_problem H C) : H = 28 :=
by {
  -- We assume the problem conditions and skip the proof using sorry
  sorry
}

end find_number_of_hens_l43_43511


namespace problem_proof_l43_43982

def mixed_to_improper (a b c : ℚ) : ℚ := a + b / c

noncomputable def evaluate_expression : ℚ :=
  100 - (mixed_to_improper 3 1 8) / (mixed_to_improper 2 1 12 - 5 / 8) * (8 / 5 + mixed_to_improper 2 2 3)

theorem problem_proof : evaluate_expression = 636 / 7 := 
  sorry

end problem_proof_l43_43982


namespace problem_l43_43187

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l43_43187


namespace lois_final_books_l43_43388

-- Definitions for the conditions given in the problem.
def initial_books : ℕ := 40
def books_given_to_nephew (b : ℕ) : ℕ := b / 4
def books_remaining_after_giving (b_given : ℕ) (b : ℕ) : ℕ := b - b_given
def books_donated_to_library (b_remaining : ℕ) : ℕ := b_remaining / 3
def books_remaining_after_donating (b_donated : ℕ) (b_remaining : ℕ) : ℕ := b_remaining - b_donated
def books_purchased : ℕ := 3
def total_books (b_final_remaining : ℕ) (b_purchased : ℕ) : ℕ := b_final_remaining + b_purchased

-- Theorem stating: Given the initial conditions, Lois should have 23 books in the end.
theorem lois_final_books : 
  total_books 
    (books_remaining_after_donating (books_donated_to_library (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books)) 
    (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books))
    books_purchased = 23 :=
  by
    sorry  -- Proof omitted as per instructions.

end lois_final_books_l43_43388


namespace sequence_sum_square_l43_43983

-- Definition of the sum of the symmetric sequence.
def sequence_sum (n : ℕ) : ℕ :=
  (List.range' 1 (n+1)).sum + (List.range' 1 n).sum

-- The conjecture that the sum of the sequence equals n^2.
theorem sequence_sum_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end sequence_sum_square_l43_43983


namespace parabola_unique_solution_l43_43271

theorem parabola_unique_solution (a : ℝ) :
  (∃ x : ℝ, (0 ≤ x^2 + a * x + 5) ∧ (x^2 + a * x + 5 ≤ 4)) → (a = 2 ∨ a = -2) :=
by
  sorry

end parabola_unique_solution_l43_43271


namespace decimal_to_fraction_l43_43443

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43443


namespace appointment_schemes_count_l43_43171

/-- Given 5 male teachers and 4 female teachers, the total number of
different appointment schemes for selecting 3 teachers to serve as class
advisers (one adviser per class), with the requirement that both male and
female teachers must be included is 420. -/
theorem appointment_schemes_count :
  let male_teachers := 5
  let female_teachers := 4
  ∃ (ans : ℕ), 
  (ans = 420 ∧ 
    ((
      (nat.choose male_teachers 2) * 
      (nat.choose female_teachers 1) + 
      (nat.choose male_teachers 1) * 
      (nat.choose female_teachers 2)
    ) * nat.choose 3 3 = ans)) :=
by
  use 420
  sorry

end appointment_schemes_count_l43_43171


namespace base_not_divisible_by_5_l43_43011

def is_not_divisible_by_5 (c : ℤ) : Prop :=
  ¬(∃ k : ℤ, c = 5 * k)

def check_not_divisible_by_5 (b : ℤ) : Prop :=
  is_not_divisible_by_5 (3 * b^3 - 3 * b^2 - b)

theorem base_not_divisible_by_5 :
  check_not_divisible_by_5 6 ∧ check_not_divisible_by_5 8 :=
by 
  sorry

end base_not_divisible_by_5_l43_43011


namespace first_part_lending_years_l43_43699

-- Definitions and conditions from the problem
def total_sum : ℕ := 2691
def second_part : ℕ := 1656
def rate_first_part : ℚ := 3 / 100
def rate_second_part : ℚ := 5 / 100
def time_second_part : ℕ := 3

-- Calculated first part
def first_part : ℕ := total_sum - second_part

-- Prove that the number of years (n) the first part is lent is 8
theorem first_part_lending_years : 
  ∃ n : ℕ, (first_part : ℚ) * rate_first_part * n = (second_part : ℚ) * rate_second_part * time_second_part ∧ n = 8 :=
by
  -- Proof steps would go here
  sorry

end first_part_lending_years_l43_43699


namespace non_negative_solutions_l43_43077

theorem non_negative_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 := 
by {
  sorry
}

end non_negative_solutions_l43_43077


namespace paul_prays_more_than_bruce_l43_43060

-- Conditions as definitions in Lean 4
def prayers_per_day_paul := 20
def prayers_per_sunday_paul := 2 * prayers_per_day_paul
def prayers_per_day_bruce := prayers_per_day_paul / 2
def prayers_per_sunday_bruce := 2 * prayers_per_sunday_paul

def weekly_prayers_paul := 6 * prayers_per_day_paul + prayers_per_sunday_paul
def weekly_prayers_bruce := 6 * prayers_per_day_bruce + prayers_per_sunday_bruce

-- Statement of the proof problem
theorem paul_prays_more_than_bruce :
  (weekly_prayers_paul - weekly_prayers_bruce) = 20 := by
  sorry

end paul_prays_more_than_bruce_l43_43060


namespace fraction_zero_solution_l43_43894

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end fraction_zero_solution_l43_43894


namespace nearest_multiple_to_457_divisible_by_11_l43_43293

theorem nearest_multiple_to_457_divisible_by_11 : ∃ n : ℤ, (n % 11 = 0) ∧ (abs (457 - n) = 5) :=
by
  sorry

end nearest_multiple_to_457_divisible_by_11_l43_43293


namespace distinct_pairs_disjoint_subsets_l43_43573

theorem distinct_pairs_disjoint_subsets (n : ℕ) : 
  ∃ k, k = (3^n + 1) / 2 := 
sorry

end distinct_pairs_disjoint_subsets_l43_43573


namespace balloon_arrangements_correct_l43_43988

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l43_43988


namespace total_calculators_sold_l43_43299

theorem total_calculators_sold 
    (x y : ℕ)
    (h₁ : y = 35)
    (h₂ : 15 * x + 67 * y = 3875) :
    x + y = 137 :=
by 
  -- We will insert the proof here
  sorry

end total_calculators_sold_l43_43299


namespace balloon_permutations_l43_43994

theorem balloon_permutations : 
  let total_letters := 7 
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  total_permutations / adjustment = 1260 :=
by
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  let total_permutations := Nat.factorial total_letters
  let adjustment := Nat.factorial repetitions_L * Nat.factorial repetitions_O
  show total_permutations / adjustment = 1260 from sorry

end balloon_permutations_l43_43994


namespace f_25_over_11_neg_l43_43380

variable (f : ℚ → ℚ)
axiom f_mul : ∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b
axiom f_prime : ∀ p : ℕ, Prime p → f p = p

theorem f_25_over_11_neg : f (25 / 11) < 0 :=
by
  -- You can prove the necessary steps during interactive proof:
  -- Using primes 25 = 5^2 and 11 itself,
  -- f (25/11) = f 25 - f 11. Thus, f (25) = 2f(5) = 2 * 5 = 10 and f(11) = 11
  -- Therefore, f (25/11) = 10 - 11 = -1 < 0.
  sorry

end f_25_over_11_neg_l43_43380


namespace solve_equation_1_solve_equation_2_l43_43786

theorem solve_equation_1 (x : ℝ) : (x + 2) ^ 2 = 3 * (x + 2) ↔ x = -2 ∨ x = 1 := by
  sorry

theorem solve_equation_2 (x : ℝ) : x ^ 2 - 8 * x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13 := by
  sorry

end solve_equation_1_solve_equation_2_l43_43786


namespace total_sum_of_money_l43_43508

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end total_sum_of_money_l43_43508


namespace stratified_sampling_A_l43_43753

theorem stratified_sampling_A (A B C total_units : ℕ) (propA : A = 400) (propB : B = 300) (propC : C = 200) (units : total_units = 90) :
  let total_families := A + B + C
  let nA := (A * total_units) / total_families
  nA = 40 :=
by
  -- prove the theorem here
  sorry

end stratified_sampling_A_l43_43753


namespace cubic_polynomial_inequality_l43_43923

theorem cubic_polynomial_inequality
  (A B C : ℝ)
  (h : ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    A = -(a + b + c) ∧ B = ab + bc + ca ∧ C = -abc) :
  A^2 + B^2 + 18 * C > 0 :=
by
  sorry

end cubic_polynomial_inequality_l43_43923


namespace decimal_to_fraction_l43_43447

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43447


namespace christen_peeled_potatoes_l43_43184

open Nat

theorem christen_peeled_potatoes :
  ∀ (total_potatoes homer_rate homer_time christen_rate : ℕ) (combined_rate : ℕ),
    total_potatoes = 60 →
    homer_rate = 4 →
    homer_time = 6 →
    christen_rate = 6 →
    combined_rate = homer_rate + christen_rate →
    Nat.ceil ((total_potatoes - (homer_rate * homer_time)) / combined_rate * christen_rate) = 21 :=
by
  intros total_potatoes homer_rate homer_time christen_rate combined_rate
  intros htp hr ht cr cr_def
  rw [htp, hr, ht, cr, cr_def]
  sorry

end christen_peeled_potatoes_l43_43184


namespace find_number_l43_43745

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 := 
sorry

end find_number_l43_43745


namespace length_of_train_is_750m_l43_43418

-- Defining the conditions
def train_and_platform_equal_length : Prop := ∀ (L : ℝ), (Length_of_train = L ∧ Length_of_platform = L)
def train_speed := 90 * (1000 / 3600)  -- Convert speed from km/hr to m/s
def crossing_time := 60  -- Time given in seconds

-- Definition for the length of the train
def Length_of_train := sorry -- Given that it should be derived

-- The proof problem statement
theorem length_of_train_is_750m : (train_and_platform_equal_length ∧ train_speed ∧ crossing_time → Length_of_train = 750) :=
by
  -- Proof is skipped
  sorry

end length_of_train_is_750m_l43_43418


namespace problem_a_problem_b_problem_c_l43_43628

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l43_43628


namespace P_X_leq_36_eq_P_Y_leq_36_plan_arrive_before_734_bus_plan_arrive_before_740_bike_l43_43604

-- Definitions for the given conditions:
def mean_bus : ℝ := 30
def var_bus : ℝ := 36
def mean_bike : ℝ := 34
def var_bike : ℝ := 4

-- Assuming normal distributions for bus times and bike times:
def X : pmf ℝ := pmf.primitive (λ x, exp (-((x - mean_bus)^2) / (2 * var_bus)) / (sqrt (2 * π * var_bus)))
def Y : pmf ℝ := pmf.primitive (λ y, exp (-((y - mean_bike)^2) / (2 * var_bike)) / (sqrt (2 * π * var_bike)))

-- The theorem statements:
theorem P_X_leq_36_eq_P_Y_leq_36 : 
  (X.prob (λ x, x ≤ 36)) = (Y.prob (λ y, y ≤ 36)) :=
sorry

theorem plan_arrive_before_734_bus : 
  (X.prob (λ x, x ≤ 34)) > (Y.prob (λ y, y ≤ 34)) :=
sorry

theorem plan_arrive_before_740_bike : 
  (Y.prob (λ y, y ≤ 40)) > (X.prob (λ x, x ≤ 40)) :=
sorry

end P_X_leq_36_eq_P_Y_leq_36_plan_arrive_before_734_bus_plan_arrive_before_740_bike_l43_43604


namespace kevin_food_spending_l43_43785

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l43_43785


namespace sum_of_digits_of_expression_l43_43954

theorem sum_of_digits_of_expression :
  (sum_of_digits (nat_to_digits 10 (2^2010 * 5^2012 * 7))) = 13 :=
by
  sorry

end sum_of_digits_of_expression_l43_43954


namespace range_of_a_l43_43742

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end range_of_a_l43_43742


namespace printing_time_345_l43_43517

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l43_43517


namespace min_value_a_b_l43_43027

theorem min_value_a_b (x y a b : ℝ) (h1 : 2 * x - y + 2 ≥ 0) (h2 : 8 * x - y - 4 ≤ 0) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) (h5 : a > 0) (h6 : b > 0) (h7 : a * x + y = 8) : 
  a + b ≥ 4 :=
sorry

end min_value_a_b_l43_43027


namespace edward_candy_purchase_l43_43963

theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) 
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := 
by 
  sorry

end edward_candy_purchase_l43_43963


namespace exponential_inequality_l43_43888

theorem exponential_inequality (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m < a^n) : ¬ (m < n) := 
sorry

end exponential_inequality_l43_43888


namespace infinite_series_converges_to_3_l43_43319

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l43_43319


namespace factorial_div_combination_l43_43160

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l43_43160


namespace solve_equation_1_solve_equation_2_l43_43253

theorem solve_equation_1 (x : ℝ) :
  x^2 - 10 * x + 16 = 0 → x = 8 ∨ x = 2 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 3) = 6 - 2 * x → x = 3 ∨ x = -2 :=
by
  sorry

end solve_equation_1_solve_equation_2_l43_43253


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43636

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l43_43636


namespace closest_fraction_l43_43542

theorem closest_fraction :
  let won_france := (23 : ℝ) / 120
  let fractions := [ (1 : ℝ) / 4, (1 : ℝ) / 5, (1 : ℝ) / 6, (1 : ℝ) / 7, (1 : ℝ) / 8 ]
  ∃ closest : ℝ, closest ∈ fractions ∧ ∀ f ∈ fractions, abs (won_france - closest) ≤ abs (won_france - f)  :=
  sorry

end closest_fraction_l43_43542


namespace correct_proposition_l43_43578

theorem correct_proposition (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end correct_proposition_l43_43578


namespace trains_cross_time_l43_43961

theorem trains_cross_time 
  (len_train1 len_train2 : ℕ) 
  (speed_train1_kmph speed_train2_kmph : ℕ) 
  (len_train1_eq : len_train1 = 200) 
  (len_train2_eq : len_train2 = 300) 
  (speed_train1_eq : speed_train1_kmph = 70) 
  (speed_train2_eq : speed_train2_kmph = 50) 
  : (500 / (120 * 1000 / 3600)) = 15 := 
by sorry

end trains_cross_time_l43_43961


namespace win_sector_area_l43_43304

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end win_sector_area_l43_43304


namespace cards_given_to_Jeff_l43_43053

-- Definitions according to the conditions
def initial_cards : Nat := 304
def remaining_cards : Nat := 276

-- The proof problem
theorem cards_given_to_Jeff : initial_cards - remaining_cards = 28 :=
by
  sorry

end cards_given_to_Jeff_l43_43053


namespace inequality_a_inequality_c_inequality_d_l43_43643

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l43_43643


namespace hyperbola_asymptote_y_eq_1_has_m_neg_3_l43_43892

theorem hyperbola_asymptote_y_eq_1_has_m_neg_3
    (m : ℝ)
    (h1 : ∀ x y, (x^2 / (2 * m)) - (y^2 / m) = 1)
    (h2 : ∀ x, 1 = (x^2 / (2 * m))): m = -3 :=
by
  sorry

end hyperbola_asymptote_y_eq_1_has_m_neg_3_l43_43892


namespace yogurt_production_cost_l43_43279

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l43_43279


namespace present_age_of_son_l43_43678

theorem present_age_of_son :
  (∃ (S F : ℕ), F = S + 22 ∧ (F + 2) = 2 * (S + 2)) → ∃ (S : ℕ), S = 20 :=
by
  sorry

end present_age_of_son_l43_43678


namespace find_expression_value_l43_43185

theorem find_expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end find_expression_value_l43_43185


namespace decimal_to_fraction_l43_43468

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43468


namespace triangle_angle_ge_60_l43_43435

theorem triangle_angle_ge_60 {A B C : ℝ} (h : A + B + C = 180) :
  A < 60 ∧ B < 60 ∧ C < 60 → false :=
by
  sorry

end triangle_angle_ge_60_l43_43435


namespace point_on_parabola_l43_43262

theorem point_on_parabola (c m n x1 x2 : ℝ) (h : x1 < x2)
  (hx1 : x1^2 + 2*x1 + c = 0)
  (hx2 : x2^2 + 2*x2 + c = 0)
  (hp : n = m^2 + 2*m + c)
  (hn : n < 0) :
  x1 < m ∧ m < x2 :=
sorry

end point_on_parabola_l43_43262


namespace balloons_remaining_each_friend_l43_43083

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end balloons_remaining_each_friend_l43_43083


namespace kevin_food_expense_l43_43777

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l43_43777


namespace maximum_x_plus_7y_exists_Q_locus_l43_43036

noncomputable def Q_locus (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem maximum_x_plus_7y (M : ℝ × ℝ) (h : Q_locus M.fst M.snd) : 
  ∃ max_value, max_value = 18 :=
  sorry

theorem exists_Q_locus (x y : ℝ) : 
  (∃ (Q : ℝ × ℝ), Q_locus Q.fst Q.snd) :=
  sorry

end maximum_x_plus_7y_exists_Q_locus_l43_43036


namespace min_value_3x_4y_l43_43383

theorem min_value_3x_4y
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + 4 * y = 21 :=
sorry

end min_value_3x_4y_l43_43383


namespace problem_a_problem_c_problem_d_l43_43649

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l43_43649


namespace leftover_value_is_correct_l43_43311

def value_of_leftover_coins (total_quarters total_dimes quarters_per_roll dimes_per_roll : ℕ) : ℝ :=
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters * 0.25) + (leftover_dimes * 0.10)

def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40

def total_quarters : ℕ := michael_quarters + anna_quarters
def total_dimes : ℕ := michael_dimes + anna_dimes

theorem leftover_value_is_correct : 
  value_of_leftover_coins total_quarters total_dimes quarters_per_roll dimes_per_roll = 10.65 :=
by
  sorry

end leftover_value_is_correct_l43_43311


namespace orchestra_admission_l43_43422

theorem orchestra_admission (x v c t: ℝ) 
  -- Conditions
  (h1 : v = 1.25 * 1.6 * x)
  (h2 : c = 0.8 * x)
  (h3 : t = 0.4 * x)
  (h4 : v + c + t = 32) :
  -- Conclusion
  v = 20 ∧ c = 8 ∧ t = 4 :=
sorry

end orchestra_admission_l43_43422


namespace total_amount_shared_l43_43973

theorem total_amount_shared
  (A B C : ℕ)
  (h_ratio : A / 2 = B / 3 ∧ B / 3 = C / 8)
  (h_Ben_share : B = 30) : A + B + C = 130 :=
by
  -- Add placeholder for the proof.
  sorry

end total_amount_shared_l43_43973


namespace lois_books_count_l43_43389

variable (initial_books : ℕ)
variable (given_books_ratio : ℚ)
variable (donated_books_ratio : ℚ)
variable (purchased_books : ℕ)

theorem lois_books_count (h_initial : initial_books = 40)
  (h_given_ratio : given_books_ratio = 1 / 4)
  (h_donated_ratio : donated_books_ratio = 1 / 3)
  (h_purchased : purchased_books = 3) :
  let remaining_after_given := initial_books - initial_books * given_books_ratio in
  let remaining_after_donated := remaining_after_given - remaining_after_given * donated_books_ratio in
    remaining_after_donated + purchased_books = 23 := by
  sorry

end lois_books_count_l43_43389


namespace trajectory_of_point_l43_43364

theorem trajectory_of_point (x y : ℝ) (P A : ℝ × ℝ × ℝ) (hP : P = (x, y, 0)) (hA : A = (0, 0, 4)) (hPA : dist P A = 5) : 
  x^2 + y^2 = 9 :=
by sorry

end trajectory_of_point_l43_43364


namespace pages_per_day_read_l43_43856

theorem pages_per_day_read (start_date : ℕ) (end_date : ℕ) (total_pages : ℕ) (fraction_covered : ℚ) (pages_read : ℕ) (days : ℕ) :
  start_date = 1 →
  end_date = 12 →
  total_pages = 144 →
  fraction_covered = 2/3 →
  pages_read = fraction_covered * total_pages →
  days = end_date - start_date + 1 →
  pages_read / days = 8 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pages_per_day_read_l43_43856


namespace min_side_b_of_triangle_l43_43370

theorem min_side_b_of_triangle (A B C a b c : ℝ) 
  (h_arith_seq : 2 * B = A + C)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides_opposite : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h_given_eq : 3 * a * c + b^2 = 25) :
  b ≥ 5 / 2 :=
  sorry

end min_side_b_of_triangle_l43_43370


namespace find_x_l43_43098

theorem find_x : 
  (5 * 12 / (180 / 3) = 1) → (∃ x : ℕ, 1 + x = 81 ∧ x = 80) :=
by
  sorry

end find_x_l43_43098


namespace original_number_is_144_l43_43972

theorem original_number_is_144 (A B C : ℕ) (A_digit : A < 10) (B_digit : B < 10) (C_digit : C < 10)
  (h1 : 100 * A + 10 * B + B = 144)
  (h2 : A * B * B = 10 * A + C)
  (h3 : (10 * A + C) % 10 = C) : 100 * A + 10 * B + B = 144 := 
sorry

end original_number_is_144_l43_43972


namespace range_of_a_l43_43355

variables {a x : ℝ}

def P (a : ℝ) : Prop := ∀ x, ¬ (x^2 - (a + 1) * x + 1 ≤ 0)

def Q (a : ℝ) : Prop := ∀ x, |x - 1| ≥ a + 2

theorem range_of_a (a : ℝ) : 
  (¬ P a ∧ ¬ Q a) → a ≥ 1 :=
by
  sorry

end range_of_a_l43_43355


namespace decimal_to_fraction_l43_43449

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43449


namespace part1_solution_set_part2_range_of_a_l43_43570

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end part1_solution_set_part2_range_of_a_l43_43570


namespace nancy_total_spending_l43_43108

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l43_43108


namespace kevin_food_expense_l43_43779

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l43_43779


namespace ball_arrangement_count_l43_43810

theorem ball_arrangement_count : 
  let red_balls := 6
  let green_balls := 3
  let selected_balls := 4
  (number_of_arrangements red_balls green_balls selected_balls) = 15 :=
by
  sorry

def number_of_arrangements (red_balls : ℕ) (green_balls : ℕ) (selected_balls : ℕ) : ℕ :=
  let choose := λ n k : ℕ, nat.choose n k 
  let case1 := choose (red_balls) (selected_balls)  -- 4 Red Balls - 1 way
  let case2 := choose (red_balls) 3 * choose (green_balls) 1  -- 3 Red Balls and 1 Green Ball
  let case3 := choose (red_balls) 2 * choose (green_balls) 2  -- 2 Red Balls and 2 Green Balls
  let case4 := choose (red_balls) 1 * choose (green_balls) 3  -- 1 Red Ball and 3 Green Balls
  case1 + case2 + case3 + case4

end ball_arrangement_count_l43_43810


namespace unique_friendly_determination_l43_43603

def is_friendly (a b : ℕ → ℕ) : Prop :=
∀ n : ℕ, ∃ i j : ℕ, n = a i * b j ∧ ∀ (k l : ℕ), n = a k * b l → (i = k ∧ j = l)

theorem unique_friendly_determination {a b c : ℕ → ℕ} 
  (h_friend_a_b : is_friendly a b) 
  (h_friend_a_c : is_friendly a c) :
  b = c :=
sorry

end unique_friendly_determination_l43_43603


namespace nancy_total_spent_l43_43105

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l43_43105


namespace max_tan_A_minus_B_l43_43591

open Real

-- Given conditions
variables {A B C a b c : ℝ}

-- Assume the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- and the equation a * cos B - b * cos A = (3 / 5) * c holds.
def condition (a b c A B C : ℝ) : Prop :=
  a * cos B - b * cos A = (3 / 5) * c

-- Prove that the maximum value of tan(A - B) is 3/4
theorem max_tan_A_minus_B (a b c A B C : ℝ) (h : condition a b c A B C) :
  ∃ t : ℝ, t = tan (A - B) ∧ 0 ≤ t ∧ t ≤ 3 / 4 :=
sorry

end max_tan_A_minus_B_l43_43591


namespace nancy_total_spent_l43_43107

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l43_43107


namespace solve_x_l43_43673

noncomputable def x : ℝ := 4.7

theorem solve_x : (10 - x) ^ 2 = x ^ 2 + 6 :=
by
  sorry

end solve_x_l43_43673


namespace option_C_represents_same_function_l43_43534

-- Definitions of the functions from option C
def f (x : ℝ) := x^2 - 1
def g (t : ℝ) := t^2 - 1

-- The proof statement that needs to be proven
theorem option_C_represents_same_function :
  f = g :=
sorry

end option_C_represents_same_function_l43_43534


namespace math_problem_l43_43719

theorem math_problem
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := 
by
  sorry

end math_problem_l43_43719


namespace remaining_adults_fed_l43_43115

theorem remaining_adults_fed 
  (cans : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (remaining_cans : ℕ)
  (remaining_adults : ℕ) :
  (adults_per_can = 4) →
  (children_per_can = 6) →
  (initial_cans = 7) →
  (children_fed = 18) →
  (remaining_cans = initial_cans - children_fed / children_per_can) →
  (remaining_adults = remaining_cans * adults_per_can) →
  remaining_adults = 16 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end remaining_adults_fed_l43_43115


namespace triangle_XYZ_ratio_l43_43220

theorem triangle_XYZ_ratio (XZ YZ : ℝ)
  (hXZ : XZ = 9) (hYZ : YZ = 40)
  (XY : ℝ) (hXY : XY = Real.sqrt (XZ ^ 2 + YZ ^ 2))
  (ZD : ℝ) (hZD : ZD = Real.sqrt (XZ * YZ))
  (XJ YJ : ℝ) (hXJ : XJ = Real.sqrt (XZ * (XZ + 2 * ZD)))
  (hYJ : YJ = Real.sqrt (YZ * (YZ + 2 * ZD)))
  (ratio : ℝ) (h_ratio : ratio = (XJ + YJ + XY) / XY) :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ ratio = p / q ∧ p + q = 203 := sorry

end triangle_XYZ_ratio_l43_43220


namespace most_likely_sum_exceeding_twelve_l43_43307

-- Define a die with faces 0, 1, 2, 3, 4, 5
def die_faces : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define a function to get the sum of rolled results exceeding 12
noncomputable def sum_exceeds_twelve (rolls : List ℕ) : ℕ :=
  let sum := rolls.foldl (· + ·) 0
  if sum > 12 then sum else 0

-- Define a function to simulate the die roll until the sum exceeds 12
noncomputable def roll_die_until_exceeds_twelve : ℕ :=
  sorry -- This would contain the logic to simulate the rolling process

-- The theorem statement that the most likely value of the sum exceeding 12 is 13
theorem most_likely_sum_exceeding_twelve : roll_die_until_exceeds_twelve = 13 :=
  sorry

end most_likely_sum_exceeding_twelve_l43_43307


namespace tetrahedron_mistaken_sum_l43_43905

theorem tetrahedron_mistaken_sum :
  let edges := 6
  let vertices := 4
  let faces := 4
  let joe_count := vertices + 1  -- Joe counts one vertex twice
  edges + joe_count + faces = 15 := by
  sorry

end tetrahedron_mistaken_sum_l43_43905


namespace remove_terms_to_get_two_thirds_l43_43708

noncomputable def sum_of_terms : ℚ := 
  (1/3) + (1/6) + (1/9) + (1/12) + (1/15) + (1/18)

noncomputable def sum_of_remaining_terms := 
  (1/3) + (1/6) + (1/9) + (1/18)

theorem remove_terms_to_get_two_thirds :
  sum_of_terms - (1/12 + 1/15) = (2/3) :=
by
  sorry

end remove_terms_to_get_two_thirds_l43_43708


namespace average_cost_price_per_meter_l43_43224

noncomputable def average_cost_per_meter (total_cost total_meters : ℝ) : ℝ :=
  total_cost / total_meters

theorem average_cost_price_per_meter :
  let silk_cost := 416.25
  let silk_meters := 9.25
  let cotton_cost := 337.50
  let cotton_meters := 7.5
  let wool_cost := 378.0
  let wool_meters := 6.0
  let total_cost := silk_cost + cotton_cost + wool_cost
  let total_meters := silk_meters + cotton_meters + wool_meters
  average_cost_per_meter total_cost total_meters = 49.75 := by
  sorry

end average_cost_price_per_meter_l43_43224


namespace initial_boys_count_l43_43853

variable (q : ℕ) -- total number of children initially in the group
variable (b : ℕ) -- number of boys initially in the group

-- Initial condition: 60% of the group are boys initially
def initial_boys (q : ℕ) : ℕ := 6 * q / 10

-- Change after event: three boys leave, three girls join
def boys_after_event (b : ℕ) : ℕ := b - 3

-- After the event, the number of boys is 50% of the total group
def boys_percentage_after_event (b : ℕ) (q : ℕ) : Prop :=
  boys_after_event b = 5 * q / 10

theorem initial_boys_count :
  ∃ b q : ℕ, b = initial_boys q ∧ boys_percentage_after_event b q → b = 18 := 
sorry

end initial_boys_count_l43_43853


namespace decimal_to_fraction_l43_43460

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43460


namespace probability_all_flip_same_times_l43_43055

theorem probability_all_flip_same_times :
  (let P_oliver (n : ℕ) := (2/3)^(n-1) * (1/3))
  (let P_jayden (n : ℕ) := (3/4)^(n-1) * (1/4))
  (let P_mia (n : ℕ) := (4/5)^(n-1) * (1/5))
  let combined_probability (n : ℕ) := P_oliver n * P_jayden n * P_mia n
  let total_probability := ∑' n : ℕ, if n = 0 then 0 else combined_probability n
  total_probability = 1 / 36 :=
by
  -- Placeholder for proof
  sorry

end probability_all_flip_same_times_l43_43055


namespace pyramid_height_l43_43118

noncomputable def height_of_pyramid (h : ℝ) : Prop :=
  let cube_edge_length := 6
  let pyramid_base_edge_length := 12
  let V_cube := cube_edge_length ^ 3
  let V_pyramid := (1 / 3) * (pyramid_base_edge_length ^ 2) * h
  V_cube = V_pyramid → h = 4.5

theorem pyramid_height : height_of_pyramid 4.5 :=
by {
  sorry
}

end pyramid_height_l43_43118


namespace contrapositive_necessary_condition_l43_43359

theorem contrapositive_necessary_condition {p q : Prop} (h : p → q) : ¬p → ¬q :=
  by sorry

end contrapositive_necessary_condition_l43_43359


namespace solve_equation1_solve_equation2_solve_equation3_l43_43068

-- For equation x^2 + 2x = 5
theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 5 ↔ (x = -1 + Real.sqrt 6) ∨ (x = -1 - Real.sqrt 6) :=
sorry

-- For equation x^2 - 2x - 1 = 0
theorem solve_equation2 (x : ℝ) : x^2 - 2 * x - 1 = 0 ↔ (x = 1 + Real.sqrt 2) ∨ (x = 1 - Real.sqrt 2) :=
sorry

-- For equation 2x^2 + 3x - 5 = 0
theorem solve_equation3 (x : ℝ) : 2 * x^2 + 3 * x - 5 = 0 ↔ (x = -5 / 2) ∨ (x = 1) :=
sorry

end solve_equation1_solve_equation2_solve_equation3_l43_43068


namespace total_marbles_l43_43216

theorem total_marbles (ratio_red_blue_green_yellow : ℕ → ℕ → ℕ → ℕ → Prop) (total : ℕ) :
  (∀ r b g y, ratio_red_blue_green_yellow r b g y ↔ r = 1 ∧ b = 5 ∧ g = 3 ∧ y = 2) →
  (∃ y, y = 20) →
  (total = y * 11 / 2) →
  total = 110 :=
by
  intros ratio_condition yellow_condition total_condition
  sorry

end total_marbles_l43_43216


namespace arc_intercept_length_l43_43313

noncomputable def side_length : ℝ := 4
noncomputable def diagonal_length : ℝ := Real.sqrt (side_length^2 + side_length^2)
noncomputable def radius : ℝ := diagonal_length / 2
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def arc_length_one_side : ℝ := circumference / 4

theorem arc_intercept_length :
  arc_length_one_side = Real.sqrt 2 * Real.pi :=
by
  sorry

end arc_intercept_length_l43_43313


namespace oblique_area_l43_43214

theorem oblique_area (side_length : ℝ) (A_ratio : ℝ) (S_original : ℝ) (S_oblique : ℝ) 
  (h1 : side_length = 1) 
  (h2 : A_ratio = (Real.sqrt 2) / 4) 
  (h3 : S_original = side_length ^ 2) 
  (h4 : S_oblique / S_original = A_ratio) : 
  S_oblique = (Real.sqrt 2) / 4 :=
by 
  sorry

end oblique_area_l43_43214


namespace probability_four_friends_same_group_l43_43409

variable (Ω : Type) [Fintype Ω] (students : Finset Ω) (groups : Finset (Finset Ω))
variable [ProbTheory groups]
noncomputable def four_friends_probability (Dave Eve Frank Grace : Ω) (h : ∀ group ∈ groups, size group = 200):

  def probability_same_group : ℝ :=
  (1 / 4) * (1 / 4) * (1 / 4)

theorem probability_four_friends_same_group (Dave Eve Frank Grace : Ω) (h : ∀ group ∈ groups, size group = 200) :
  four_friends_probability Dave Eve Frank Grace h = 1 / 64
:= sorry

end probability_four_friends_same_group_l43_43409


namespace find_ratio_of_hyperbola_asymptotes_l43_43169

theorem find_ratio_of_hyperbola_asymptotes (a b : ℝ) (h : a > b) (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → |(2 * b / a)| = 1) : 
  a / b = 2 := 
by 
  sorry

end find_ratio_of_hyperbola_asymptotes_l43_43169


namespace geometric_sequence_a9_l43_43902

theorem geometric_sequence_a9
  (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 3 * a 6 = -32)
  (h2 : a 4 + a 5 = 4)
  (hq : ∃ n : ℤ, q = n)
  : a 10 = -256 := 
sorry

end geometric_sequence_a9_l43_43902


namespace john_total_spent_is_correct_l43_43909

noncomputable def john_spent_total (original_cost : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_cost := original_cost - (discount_rate / 100 * original_cost)
  let cost_with_tax := discounted_cost + (sales_tax_rate / 100 * discounted_cost)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_cost_with_tax := lightsaber_cost + (sales_tax_rate / 100 * lightsaber_cost)
  cost_with_tax + lightsaber_cost_with_tax

theorem john_total_spent_is_correct :
  john_spent_total 1200 20 8 = 3628.80 :=
by
  sorry

end john_total_spent_is_correct_l43_43909


namespace exists_ab_odd_n_exists_ab_odd_n_gt3_l43_43339

-- Define the required conditions
def gcd_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define a helper function to identify odd positive integers
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem exists_ab_odd_n (n : ℕ) (h : is_odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n :=
sorry

theorem exists_ab_odd_n_gt3 (n : ℕ) (h1 : is_odd n) (h2 : n > 3) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n ∧ n ∣ (a - b) = false :=
sorry

end exists_ab_odd_n_exists_ab_odd_n_gt3_l43_43339


namespace william_washing_time_l43_43823

theorem william_washing_time :
  let time_windows := 4
  let time_car_body := 7
  let time_tires := 4
  let time_waxing := 9
  let cars_washed := 2
  let suv_multiplier := 2
  let normal_car_time := time_windows + time_car_body + time_tires + time_waxing
  let total_normal_cars_time := cars_washed * normal_car_time
  let suv_time := suv_multiplier * normal_car_time
  let total_time := total_normal_cars_time + suv_time
  in total_time = 96 :=
by
  sorry

end william_washing_time_l43_43823


namespace value_of_a_plus_b_l43_43767

theorem value_of_a_plus_b (a b : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b) 
  (hg : ∀ x, g x = -3 * x + 2)
  (hgf : ∀ x, g (f x) = -2 * x - 3) :
  a + b = 7 / 3 :=
by
  sorry

end value_of_a_plus_b_l43_43767


namespace decimal_to_fraction_l43_43446

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l43_43446


namespace alices_number_l43_43700

theorem alices_number :
  ∃ (m : ℕ), (180 ∣ m) ∧ (45 ∣ m) ∧ (1000 ≤ m) ∧ (m ≤ 3000) ∧
    (m = 1260 ∨ m = 1440 ∨ m = 1620 ∨ m = 1800 ∨ m = 1980 ∨
     m = 2160 ∨ m = 2340 ∨ m = 2520 ∨ m = 2700 ∨ m = 2880) :=
by
  sorry

end alices_number_l43_43700


namespace parity_E_2021_2022_2023_l43_43697

-- Define the sequence with the given initial conditions and recurrence relation
def E : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 1
| (n + 3) := E (n + 2) + E (n + 1) + E n

-- Define the parities of numbers
def parity (n : ℕ) : Prop :=
  (n % 2 = 0)

-- The math problem rephrased: prove the correctness of parities for the specific indices
theorem parity_E_2021_2022_2023 :
  parity (E 2021) ∧ ¬ parity (E 2022) ∧ ¬ parity (E 2023) :=
by sorry

end parity_E_2021_2022_2023_l43_43697


namespace total_students_l43_43655

-- Define the condition that the sum of boys (75) and girls (G) is the total number of students (T)
def sum_boys_girls (G T : ℕ) := 75 + G = T

-- Define the condition that the number of girls (G) equals 75% of the total number of students (T)
def girls_percentage (G T : ℕ) := G = Nat.div (3 * T) 4

-- State the theorem that given the above conditions, the total number of students (T) is 300
theorem total_students (G T : ℕ) (h1 : sum_boys_girls G T) (h2 : girls_percentage G T) : T = 300 := 
sorry

end total_students_l43_43655


namespace capacity_of_first_bucket_is_3_l43_43507

variable (C : ℝ)

theorem capacity_of_first_bucket_is_3 
  (h1 : 48 / C = 48 / 3 - 4) : 
  C = 3 := 
  sorry

end capacity_of_first_bucket_is_3_l43_43507


namespace decimal_to_fraction_l43_43441

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l43_43441


namespace number_of_possible_values_for_m_l43_43263

noncomputable def log_2 (x : ℝ) : ℝ := log x / log 2

theorem number_of_possible_values_for_m :
  let m := λ (x : ℕ), x > 4 ∧ x < 1800
  in (set.count (set_of m)) = 1795 :=
by
  sorry

end number_of_possible_values_for_m_l43_43263


namespace find_b_l43_43860

def f (x : ℝ) : ℝ := 5 * x + 3

theorem find_b : ∃ b : ℝ, f b = -2 ∧ b = -1 := by
  have h : 5 * (-1 : ℝ) + 3 = -2 := by norm_num
  use -1
  simp [f, h]
  sorry

end find_b_l43_43860


namespace cost_to_fill_half_of_can_B_l43_43548

theorem cost_to_fill_half_of_can_B (r h : ℝ) (cost_fill_V : ℝ) (cost_fill_V_eq : cost_fill_V = 16)
  (V_radius_eq : 2 * r = radius_of_can_V)
  (V_height_eq: h / 2 = height_of_can_V) :
  cost_fill_half_of_can_B = 4 :=
by
  sorry

end cost_to_fill_half_of_can_B_l43_43548


namespace product_xyz_l43_43201

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l43_43201


namespace find_sum_of_x_and_reciprocal_l43_43727

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l43_43727


namespace find_n_solution_l43_43770

def product_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.prod

theorem find_n_solution : ∃ n : ℕ, n > 0 ∧ n^2 - 17 * n + 56 = product_of_digits n ∧ n = 4 := 
by
  sorry

end find_n_solution_l43_43770


namespace price_change_38_percent_l43_43698

variables (P : ℝ) (x : ℝ)
noncomputable def final_price := P * (1 - (x / 100)^2) * 0.9
noncomputable def target_price := 0.77 * P

theorem price_change_38_percent (h : final_price P x = target_price P):
  x = 38 := sorry

end price_change_38_percent_l43_43698


namespace collinear_implies_coplanar_l43_43975

-- Define the concept of collinear points in space
def collinear (p1 p2 p3 : Point) : Prop := ∃ l : Line, p1 ∈ l ∧ p2 ∈ l ∧ p3 ∈ l

-- Define the concept of coplanar points in space
def coplanar (s : Set Point) : Prop := ∃ p : Plane, ∀ x ∈ s, x ∈ p

-- State the problem conditions and conclusion in Lean statement
theorem collinear_implies_coplanar (p1 p2 p3 p4 : Point) :
  collinear p1 p2 p3 → coplanar {p1, p2, p3, p4} ∧ 
  ¬ (coplanar {p1, p2, p3, p4} → collinear p1 p2 p3) :=
sorry

end collinear_implies_coplanar_l43_43975


namespace two_point_three_five_as_fraction_l43_43454

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l43_43454


namespace steve_berry_picking_strategy_l43_43408

def berry_picking_goal_reached (monday_earnings tuesday_earnings total_goal: ℕ) : Prop :=
  monday_earnings + tuesday_earnings >= total_goal

def optimal_thursday_strategy (remaining_goal payment_per_pound total_capacity : ℕ) : ℕ :=
  if remaining_goal = 0 then 0 else total_capacity

theorem steve_berry_picking_strategy :
  let monday_lingonberries := 8
  let monday_cloudberries := 10
  let monday_blueberries := 30 - monday_lingonberries - monday_cloudberries
  let tuesday_lingonberries := 3 * monday_lingonberries
  let tuesday_cloudberries := 2 * monday_cloudberries
  let tuesday_blueberries := 5
  let lingonberry_rate := 2
  let cloudberry_rate := 3
  let blueberry_rate := 5
  let max_capacity := 30
  let total_goal := 150

  let monday_earnings := (monday_lingonberries * lingonberry_rate) + 
                         (monday_cloudberries * cloudberry_rate) + 
                         (monday_blueberries * blueberry_rate)
                         
  let tuesday_earnings := (tuesday_lingonberries * lingonberry_rate) + 
                          (tuesday_cloudberries * cloudberry_rate) +
                          (tuesday_blueberries * blueberry_rate)

  let total_earnings := monday_earnings + tuesday_earnings

  berry_picking_goal_reached monday_earnings tuesday_earnings total_goal ∧
  optimal_thursday_strategy (total_goal - total_earnings) blueberry_rate max_capacity = 30 
:= by {
  sorry
}

end steve_berry_picking_strategy_l43_43408


namespace solve_N_l43_43210

noncomputable def N (a b c d : ℝ) := (a + b) / c - d

theorem solve_N : 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  N a b c d = -1 :=
by 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  let n := N a b c d
  sorry

end solve_N_l43_43210


namespace probability_two_math_books_l43_43238

theorem probability_two_math_books (total_books math_books : ℕ) (total_books_eq : total_books = 5) (math_books_eq : math_books = 3) : 
  (nat.choose math_books 2 : ℚ) / (nat.choose total_books 2) = 3 / 10 := by
  rw [total_books_eq, math_books_eq]
  sorry

end probability_two_math_books_l43_43238


namespace fifty_percent_greater_l43_43960

theorem fifty_percent_greater (x : ℕ) (h : x = 88 + (88 / 2)) : x = 132 := 
by {
  sorry
}

end fifty_percent_greater_l43_43960


namespace sum_four_product_l43_43054

theorem sum_four_product (n : ℕ) (h : n > 0) :
  (∑ k in Finset.range n, k * (k + 1) * (k + 2) * (k + 3)) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) / 5 :=
by
  induction n with k hk
  . -- Base case goes here
    sorry
  . -- Inductive step goes here
    sorry

end sum_four_product_l43_43054


namespace truncated_cone_resistance_l43_43306

theorem truncated_cone_resistance (a b h : ℝ) (ρ : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_pos : 0 < h) :
  (∫ x in (0:ℝ)..h, ρ / (π * ((a + x * (b - a) / h) / 2) ^ 2)) = 4 * ρ * h / (π * a * b) := 
sorry

end truncated_cone_resistance_l43_43306


namespace base4_division_l43_43163

/-- Given in base 4:
2023_4 div 13_4 = 155_4
We need to prove the quotient is equal to 155_4.
-/
theorem base4_division (n m q r : ℕ) (h1 : n = 2 * 4^3 + 0 * 4^2 + 2 * 4^1 + 3 * 4^0)
    (h2 : m = 1 * 4^1 + 3 * 4^0)
    (h3 : q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0)
    (h4 : n = m * q + r)
    (h5 : 0 ≤ r ∧ r < m):
  q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0 := 
by
  sorry

end base4_division_l43_43163


namespace expected_value_of_winnings_is_5_l43_43959

namespace DiceGame

def sides : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 2 * roll else 0

noncomputable def expectedValue : ℚ :=
  (winnings 2 + winnings 4 + winnings 6 + winnings 8) / 8

theorem expected_value_of_winnings_is_5 :
  expectedValue = 5 := by
  sorry

end DiceGame

end expected_value_of_winnings_is_5_l43_43959


namespace largest_expression_l43_43597

noncomputable def x : ℝ := 10 ^ (-2024 : ℤ)

theorem largest_expression :
  let a := 5 + x
  let b := 5 - x
  let c := 5 * x
  let d := 5 / x
  let e := x / 5
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end largest_expression_l43_43597


namespace max_k_value_l43_43177

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∃ k : ℝ, (∀ m, 0 < m → m < 1/2 → (1/m + 2/(1-2*m) ≥ k)) ∧ k = 8) := 
sorry

end max_k_value_l43_43177


namespace range_x_l43_43568

variable {R : Type*} [LinearOrderedField R]

def monotone_increasing_on (f : R → R) (s : Set R) := ∀ ⦃a b⦄, a ≤ b → f a ≤ f b

theorem range_x 
    (f : R → R) 
    (h_mono : monotone_increasing_on f Set.univ) 
    (h_zero : f 1 = 0) 
    (h_ineq : ∀ x, f (x^2 + 3 * x - 3) < 0) :
  ∀ x, -4 < x ∧ x < 1 :=
by 
  sorry

end range_x_l43_43568


namespace max_knights_seated_l43_43240

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end max_knights_seated_l43_43240


namespace printing_time_345_l43_43516

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l43_43516


namespace problem_a_problem_c_problem_d_l43_43647

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l43_43647


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l43_43950

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l43_43950


namespace decimal_to_fraction_l43_43471

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l43_43471


namespace there_exists_triangle_part_two_l43_43231

noncomputable def exists_triangle (a b c : ℝ) : Prop :=
a > 0 ∧
4 * a - 8 * b + 4 * c ≥ 0 ∧
9 * a - 12 * b + 4 * c ≥ 0 ∧
2 * a ≤ 2 * b ∧
2 * b ≤ 3 * a ∧
b^2 ≥ a*c

theorem there_exists_triangle (a b c : ℝ) (h1 : a > 0)
  (h2 : 4 * a - 8 * b + 4 * c ≥ 0)
  (h3 : 9 * a - 12 * b + 4 * c ≥ 0)
  (h4 : 2 * a ≤ 2 * b)
  (h5 : 2 * b ≤ 3 * a)
  (h6 : b^2 ≥ a * c) : 
 a ≤ b ∧ b ≤ c ∧ a + b > c :=
sorry

theorem part_two (a b c : ℝ) (h1 : a > 0) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c < a + b) :
  ∃ h : a > 0, (a / (a + c) + b / (b + a) > c / (b + c)) :=
sorry

end there_exists_triangle_part_two_l43_43231


namespace decimal_to_fraction_l43_43464

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l43_43464


namespace sin_two_alpha_sub_pi_eq_24_div_25_l43_43173

noncomputable def pi_div_2 : ℝ := Real.pi / 2

theorem sin_two_alpha_sub_pi_eq_24_div_25
  (α : ℝ) 
  (h1 : pi_div_2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan (α + Real.pi / 4) = -1 / 7) : 
  Real.sin (2 * α - Real.pi) = 24 / 25 := 
sorry

end sin_two_alpha_sub_pi_eq_24_div_25_l43_43173


namespace new_students_average_age_l43_43793

theorem new_students_average_age :
  let O := 12 in
  let A_O := 40 in
  let N := 12 in
  let new_avg := A_O - 4 in
  let total_age_before := O * A_O in
  let total_age_after := (O + N) * new_avg in
  ∃ A_N : ℕ, total_age_before + N * A_N = total_age_after ∧ A_N = 32 :=
by
  let O := 12
  let A_O := 40
  let N := 12
  let new_avg := A_O - 4
  let total_age_before := O * A_O
  let total_age_after := (O + N) * new_avg
  use 32
  split
  · sorry
  · rfl

end new_students_average_age_l43_43793


namespace largest_base5_to_base7_l43_43912

-- Define the largest four-digit number in base-5
def largest_base5_four_digit_number : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

-- Convert this number to base-7
def convert_to_base7 (n : ℕ) : ℕ := 
  let d3 := n / (7^3)
  let r3 := n % (7^3)
  let d2 := r3 / (7^2)
  let r2 := r3 % (7^2)
  let d1 := r2 / (7^1)
  let r1 := r2 % (7^1)
  let d0 := r1
  (d3 * 10^3) + (d2 * 10^2) + (d1 * 10^1) + d0

-- Theorem to prove m in base-7
theorem largest_base5_to_base7 : 
  convert_to_base7 largest_base5_four_digit_number = 1551 :=
by 
  -- skip the proof
  sorry

end largest_base5_to_base7_l43_43912


namespace nancy_total_spending_l43_43111

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l43_43111


namespace combination_formula_l43_43146

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l43_43146


namespace minimum_time_for_xiang_qing_fried_eggs_l43_43825

-- Define the time taken for each individual step
def wash_scallions_time : ℕ := 1
def beat_eggs_time : ℕ := 1 / 2
def mix_egg_scallions_time : ℕ := 1
def wash_pan_time : ℕ := 1 / 2
def heat_pan_time : ℕ := 1 / 2
def heat_oil_time : ℕ := 1 / 2
def cook_dish_time : ℕ := 2

-- Define the total minimum time required
def minimum_time : ℕ := 5

-- The main theorem stating that the minimum time required is 5 minutes
theorem minimum_time_for_xiang_qing_fried_eggs :
  wash_scallions_time + beat_eggs_time + mix_egg_scallions_time + wash_pan_time + heat_pan_time + heat_oil_time + cook_dish_time = minimum_time := 
by sorry

end minimum_time_for_xiang_qing_fried_eggs_l43_43825


namespace simplify_fraction_l43_43579

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h_cond : y^3 - 1/x ≠ 0) :
  (x^3 - 1/y) / (y^3 - 1/x) = x / y :=
by
  sorry

end simplify_fraction_l43_43579


namespace students_in_school_at_least_225_l43_43755

-- Conditions as definitions
def students_in_band := 85
def students_in_sports := 200
def students_in_both := 60
def students_in_either := 225

-- The proof statement
theorem students_in_school_at_least_225 :
  students_in_band + students_in_sports - students_in_both = students_in_either :=
by
  -- This statement will just assert the correctness as per given information in the problem
  sorry

end students_in_school_at_least_225_l43_43755


namespace decimal_to_fraction_l43_43458

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l43_43458


namespace sandra_money_left_l43_43403

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end sandra_money_left_l43_43403


namespace sum_of_ages_l43_43229

-- Definitions based on conditions
def age_relation1 (a b c : ℕ) : Prop := a = 20 + b + c
def age_relation2 (a b c : ℕ) : Prop := a^2 = 2000 + (b + c)^2

-- The statement to be proven
theorem sum_of_ages (a b c : ℕ) (h1 : age_relation1 a b c) (h2 : age_relation2 a b c) : a + b + c = 80 :=
by
  sorry

end sum_of_ages_l43_43229


namespace epicenter_distance_l43_43001

noncomputable def distance_from_epicenter (v1 v2 Δt: ℝ) : ℝ :=
  Δt / ((1 / v2) - (1 / v1))

theorem epicenter_distance : 
  distance_from_epicenter 5.94 3.87 11.5 = 128 := 
by
  -- The proof will use calculations shown in the solution.
  sorry

end epicenter_distance_l43_43001


namespace bicycle_cost_price_l43_43696

theorem bicycle_cost_price 
  (CP_A : ℝ) 
  (H : CP_A * (1.20 * 0.85 * 1.30 * 0.90) = 285) : 
  CP_A = 285 / (1.20 * 0.85 * 1.30 * 0.90) :=
sorry

end bicycle_cost_price_l43_43696


namespace number_of_x_intercepts_l43_43986

def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem number_of_x_intercepts : ∃! (x : ℝ), ∃ (y : ℝ), parabola y = x ∧ y = 0 :=
by
  sorry

end number_of_x_intercepts_l43_43986


namespace first_divisor_is_13_l43_43686

theorem first_divisor_is_13 (x : ℤ) (h : (377 / x) / 29 * (1/4 : ℚ) / 2 = (1/8 : ℚ)) : x = 13 := by
  sorry

end first_divisor_is_13_l43_43686


namespace proof1_proof2_proof3_proof4_l43_43858

-- Define variables.
variable (m n x y z : ℝ)

-- Prove the expressions equalities.
theorem proof1 : (m + 2 * n) - (m - 2 * n) = 4 * n := sorry
theorem proof2 : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := sorry
theorem proof3 : 2 * x - 3 * (x - 2 * y + 3 * x) + 2 * (3 * x - 3 * y + 2 * z) = -4 * x + 4 * z := sorry
theorem proof4 : 8 * m^2 - (4 * m^2 - 2 * m - 4 * (2 * m^2 - 5 * m)) = 12 * m^2 - 18 * m := sorry

end proof1_proof2_proof3_proof4_l43_43858


namespace infinite_series_sum_eq_two_l43_43325

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l43_43325


namespace print_time_correct_l43_43524

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l43_43524


namespace complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l43_43354

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of a set
def C_A : Set ℕ := U \ A
def C_B : Set ℕ := U \ B

-- Questions rephrased as theorem statements
theorem complement_of_A_in_U : C_A = {2, 4, 5} := by sorry
theorem intersection_of_A_and_B : A ∩ B = ∅ := by sorry
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by sorry
theorem union_of_complements_of_A_and_B : C_A ∪ C_B = U := by sorry

end complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l43_43354


namespace plane_contains_points_l43_43336

def point := (ℝ × ℝ × ℝ)

def is_plane (A B C D : ℝ) (p : point) : Prop :=
  ∃ x y z, p = (x, y, z) ∧ A * x + B * y + C * z + D = 0

theorem plane_contains_points :
  ∃ A B C D : ℤ,
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    is_plane A B C D (2, -1, 3) ∧
    is_plane A B C D (0, -1, 5) ∧
    is_plane A B C D (-2, -3, 4) ∧
    A = 2 ∧ B = 5 ∧ C = -2 ∧ D = 7 :=
  sorry

end plane_contains_points_l43_43336


namespace math_problem_l43_43625

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l43_43625


namespace kevin_food_expense_l43_43778

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l43_43778


namespace min_digits_decimal_correct_l43_43492

noncomputable def min_digits_decimal : ℕ := 
  let n : ℕ := 123456789
  let d : ℕ := 2^26 * 5^4
  26 -- As per the problem statement

theorem min_digits_decimal_correct :
  let n := 123456789
  let d := 2^26 * 5^4
  ∀ x:ℕ, (∃ k:ℕ, n = k * 10^x) → x ≥ min_digits_decimal := 
by
  sorry

end min_digits_decimal_correct_l43_43492


namespace length_of_train_l43_43826

theorem length_of_train
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_mps : ℝ := speed_kmph * (1000 / 3600))
  (total_distance : ℝ := train_speed_mps * crossing_time)
  (train_length : ℝ := total_distance - platform_length)
  (h_speed : speed_kmph = 72)
  (h_platform : platform_length = 260)
  (h_time : crossing_time = 26)
  : train_length = 260 := by
  sorry

end length_of_train_l43_43826


namespace one_of_a_b_c_is_zero_l43_43387

theorem one_of_a_b_c_is_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 :=
by
  sorry

end one_of_a_b_c_is_zero_l43_43387


namespace print_time_l43_43520

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l43_43520


namespace sum_of_digits_of_expression_l43_43952

theorem sum_of_digits_of_expression :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7 in
  (n.digits.sum = 13) := 
by
  sorry

end sum_of_digits_of_expression_l43_43952


namespace expression_eq_l43_43042

variable {α β γ δ p q : ℝ}

-- Conditions from the problem
def roots_eq1 (α β p : ℝ) : Prop := ∀ x : ℝ, (x - α) * (x - β) = x^2 + p*x - 1
def roots_eq2 (γ δ q : ℝ) : Prop := ∀ x : ℝ, (x - γ) * (x - δ) = x^2 + q*x + 1

-- The proof statement where the expression is equated to p^2 - q^2
theorem expression_eq (h1: roots_eq1 α β p) (h2: roots_eq2 γ δ q) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = p^2 - q^2 := sorry

end expression_eq_l43_43042


namespace integer_solutions_k_l43_43863

theorem integer_solutions_k (k n m : ℤ) (h1 : k + 1 = n^2) (h2 : 16 * k + 1 = m^2) :
  k = 0 ∨ k = 3 :=
by sorry

end integer_solutions_k_l43_43863


namespace balloon_permutations_l43_43991

theorem balloon_permutations : ∀ (word : List Char) (n l o : ℕ),
  word = ['B', 'A', 'L', 'L', 'O', 'O', 'N'] → n = 7 → l = 2 → o = 2 →
  ∑ (perm : List Char), perm.permutations.count = (nat.factorial n / (nat.factorial l * nat.factorial o)) := sorry

end balloon_permutations_l43_43991


namespace polynomial_divisible_exists_l43_43764

theorem polynomial_divisible_exists (p : Polynomial ℤ) (a : ℕ → ℤ) (k : ℕ) 
  (h_inc : ∀ i j, i < j → a i < a j) (h_nonzero : ∀ i, i < k → p.eval (a i) ≠ 0) :
  ∃ a_0 : ℤ, ∀ i, i < k → p.eval (a i) ∣ p.eval a_0 := 
by
  sorry

end polynomial_divisible_exists_l43_43764


namespace min_value_P_l43_43802

-- Define the polynomial P
def P (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

-- Theorem statement: The minimum value of P(x, y) is -18
theorem min_value_P : ∃ (x y : ℝ), P x y = -18 := by
  sorry

end min_value_P_l43_43802


namespace valid_triplets_l43_43334

theorem valid_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_leq1 : a ≤ b) (h_leq2 : b ≤ c)
  (h_div1 : a ∣ (b + c)) (h_div2 : b ∣ (a + c)) (h_div3 : c ∣ (a + b)) :
  (a = b ∧ b = c) ∨ (a = b ∧ c = 2 * a) ∨ (b = 2 * a ∧ c = 3 * a) :=
sorry

end valid_triplets_l43_43334


namespace problem1_problem2_l43_43134

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l43_43134


namespace minimum_distance_of_AB_l43_43938

noncomputable def f (x : ℝ) := Real.exp x + 1
noncomputable def g (x : ℝ) := 2 * x - 1

theorem minimum_distance_of_AB :
  |(f (Real.log 2) - g (Real.log 2))| = 4 - 2 * Real.log 2 :=
sorry

end minimum_distance_of_AB_l43_43938


namespace nancy_total_spending_l43_43113

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l43_43113


namespace calculate_value_l43_43547

theorem calculate_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calculate_value_l43_43547


namespace vector_c_condition_l43_43739

variables (a b c : ℝ × ℝ)

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def is_parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * w.1, k * w.2)

theorem vector_c_condition (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (2, -3)) 
  (hc : c = (7 / 2, -7 / 4)) :
  is_perpendicular c a ∧ is_parallel b (a - c) :=
sorry

end vector_c_condition_l43_43739


namespace relationship_between_k_and_c_l43_43760

-- Define the functions and given conditions
def y1 (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c
def y2 (x : ℝ) (k : ℝ) : ℝ := k*x + 2

-- Define the vertex of y1
def vertex_y1 (c : ℝ) : ℝ × ℝ := (-1, c - 1)

-- State the main theorem
theorem relationship_between_k_and_c (k c : ℝ) (hk : k ≠ 0) :
  y2 (vertex_y1 c).1 k = (vertex_y1 c).2 → c + k = 3 :=
by
  sorry

end relationship_between_k_and_c_l43_43760
