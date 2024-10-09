import Mathlib

namespace required_moles_of_H2O_l2268_226801

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end required_moles_of_H2O_l2268_226801


namespace find_C_l2268_226810

theorem find_C
  (A B C : ℕ)
  (h1 : A + B + C = 1000)
  (h2 : A + C = 700)
  (h3 : B + C = 600) :
  C = 300 := by
  sorry

end find_C_l2268_226810


namespace hex_B3F_to_decimal_l2268_226854

-- Define the hexadecimal values of B, 3, F
def hex_B : ℕ := 11
def hex_3 : ℕ := 3
def hex_F : ℕ := 15

-- Prove the conversion of B3F_{16} to a base 10 integer equals 2879
theorem hex_B3F_to_decimal : (hex_B * 16^2 + hex_3 * 16^1 + hex_F * 16^0) = 2879 := 
by 
  -- calculation details skipped
  sorry

end hex_B3F_to_decimal_l2268_226854


namespace optionD_is_not_linear_system_l2268_226847

-- Define the equations for each option
def eqA1 (x y : ℝ) : Prop := 3 * x + 2 * y = 10
def eqA2 (x y : ℝ) : Prop := 2 * x - 3 * y = 5

def eqB1 (x y : ℝ) : Prop := 3 * x + 5 * y = 1
def eqB2 (x y : ℝ) : Prop := 2 * x - y = 4

def eqC1 (x y : ℝ) : Prop := x + 5 * y = 1
def eqC2 (x y : ℝ) : Prop := x - 5 * y = 2

def eqD1 (x y : ℝ) : Prop := x - y = 1
def eqD2 (x y : ℝ) : Prop := y + 1 / x = 3

-- Define the property of a linear equation
def is_linear (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, eq x y → a * x + b * y = c

-- State the theorem
theorem optionD_is_not_linear_system : ¬ (is_linear eqD1 ∧ is_linear eqD2) :=
by
  sorry

end optionD_is_not_linear_system_l2268_226847


namespace jasper_time_l2268_226816

theorem jasper_time {omar_time : ℕ} {omar_height : ℕ} {jasper_height : ℕ} 
  (h1 : omar_time = 12)
  (h2 : omar_height = 240)
  (h3 : jasper_height = 600)
  (h4 : ∃ t : ℕ, t = (jasper_height * omar_time) / (3 * omar_height))
  : t = 10 :=
by sorry

end jasper_time_l2268_226816


namespace asparagus_cost_correct_l2268_226864

def cost_asparagus (total_start: Int) (total_left: Int) (cost_bananas: Int) (cost_pears: Int) (cost_chicken: Int) : Int := 
  total_start - total_left - cost_bananas - cost_pears - cost_chicken

theorem asparagus_cost_correct :
  cost_asparagus 55 28 8 2 11 = 6 :=
by
  sorry

end asparagus_cost_correct_l2268_226864


namespace bridge_length_sufficient_l2268_226889

structure Train :=
  (length : ℕ) -- length of the train in meters
  (speed : ℚ) -- speed of the train in km/hr

def speed_in_m_per_s (speed_in_km_per_hr : ℚ) : ℚ :=
  speed_in_km_per_hr * 1000 / 3600

noncomputable def length_of_bridge (train1 train2 : Train) : ℚ :=
  let train1_speed_m_per_s := speed_in_m_per_s train1.speed
  let train2_speed_m_per_s := speed_in_m_per_s train2.speed
  let relative_speed := train1_speed_m_per_s + train2_speed_m_per_s
  let total_length := train1.length + train2.length
  let time_to_pass := total_length / relative_speed
  let distance_train1 := train1_speed_m_per_s * time_to_pass
  let distance_train2 := train2_speed_m_per_s * time_to_pass
  distance_train1 + distance_train2

theorem bridge_length_sufficient (train1 train2 : Train) (h1 : train1.length = 200) (h2 : train1.speed = 60) (h3 : train2.length = 150) (h4 : train2.speed = 45) :
  length_of_bridge train1 train2 ≥ 350.04 :=
  by
  sorry

end bridge_length_sufficient_l2268_226889


namespace high_heels_height_l2268_226817

theorem high_heels_height (x : ℝ) :
  let height := 157
  let lower_limbs := 95
  let golden_ratio := 0.618
  (95 + x) / (157 + x) = 0.618 → x = 5.3 :=
sorry

end high_heels_height_l2268_226817


namespace tan_double_angle_l2268_226823

variable {α β : ℝ}

theorem tan_double_angle (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 2) : Real.tan (2 * α) = -1 := by
  sorry

end tan_double_angle_l2268_226823


namespace wendy_makeup_time_l2268_226877

theorem wendy_makeup_time :
  ∀ (num_products wait_time total_time makeup_time : ℕ),
    num_products = 5 →
    wait_time = 5 →
    total_time = 55 →
    makeup_time = total_time - (num_products - 1) * wait_time →
    makeup_time = 35 :=
by
  intro num_products wait_time total_time makeup_time h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_makeup_time_l2268_226877


namespace find_angle_A_l2268_226867

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c) :
  A = Real.pi / 3 :=
sorry

end find_angle_A_l2268_226867


namespace equilateral_triangle_data_l2268_226803

theorem equilateral_triangle_data
  (A : ℝ)
  (b : ℝ)
  (ha : A = 450)
  (hb : b = 25)
  (equilateral : ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a) :
  ∃ (h P : ℝ), h = 36 ∧ P = 75 := by
  sorry

end equilateral_triangle_data_l2268_226803


namespace a_n_formula_b_n_formula_S_n_formula_l2268_226849

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1) + 3 * n
noncomputable def S_n (n : ℕ) : ℕ := 2^n - 1 + (3 * n^2 + 3 * n) / 2

theorem a_n_formula (n : ℕ) : a_n n = 3 * n := by
  unfold a_n
  rfl

theorem b_n_formula (n : ℕ) : b_n n = 2^(n-1) + 3 * n := by
  unfold b_n
  rfl

theorem S_n_formula (n : ℕ) : S_n n = 2^n - 1 + (3 * n^2 + 3 * n) / 2 := by
  unfold S_n
  rfl

end a_n_formula_b_n_formula_S_n_formula_l2268_226849


namespace cuckoo_chime_78_l2268_226800

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end cuckoo_chime_78_l2268_226800


namespace colored_sectors_overlap_l2268_226866

/--
Given two disks each divided into 1985 equal sectors, with 200 sectors on each disk colored arbitrarily,
and one disk is rotated by angles that are multiples of 360 degrees / 1985, 
prove that there are at least 80 positions where no more than 20 colored sectors coincide.
-/
theorem colored_sectors_overlap :
  ∀ (disks : ℕ → ℕ) (sectors_colored : ℕ),
  disks 1 = 1985 → disks 2 = 1985 →
  sectors_colored = 200 →
  ∃ (p : ℕ), p ≥ 80 ∧ (∀ (i : ℕ), (i < p → sectors_colored ≤ 20)) := 
sorry

end colored_sectors_overlap_l2268_226866


namespace staircase_steps_l2268_226846

theorem staircase_steps (x : ℕ) :
  x % 2 = 1 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 6 = 5 ∧
  x % 7 = 0 → 
  x ≡ 119 [MOD 420] :=
by
  sorry

end staircase_steps_l2268_226846


namespace solve_for_star_l2268_226811

theorem solve_for_star 
  (x : ℝ) 
  (h : 45 - (28 - (37 - (15 - x))) = 58) : 
  x = 19 :=
by
  -- Proof goes here. Currently incomplete, so we use sorry.
  sorry

end solve_for_star_l2268_226811


namespace sum_of_midpoints_l2268_226852

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end sum_of_midpoints_l2268_226852


namespace product_of_integers_l2268_226885

-- Define the conditions as variables in Lean
variables {x y : ℤ}

-- State the main theorem/proof
theorem product_of_integers (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end product_of_integers_l2268_226885


namespace quadratic_function_range_l2268_226881

-- Define the quadratic function and the domain
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + 1

-- State the proof problem
theorem quadratic_function_range : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → -8 ≤ quadratic_function x ∧ quadratic_function x ≤ 1 := 
by 
  intro x
  intro h
  sorry

end quadratic_function_range_l2268_226881


namespace customers_left_correct_l2268_226875

-- Define the initial conditions
def initial_customers : ℕ := 8
def remaining_customers : ℕ := 5

-- Define the statement regarding customers left
def customers_left : ℕ := initial_customers - remaining_customers

-- The theorem we need to prove
theorem customers_left_correct : customers_left = 3 := by
    -- Skipping the actual proof
    sorry

end customers_left_correct_l2268_226875


namespace math_problem_l2268_226882

theorem math_problem : (3 ^ 456) + (9 ^ 5 / 9 ^ 3) = 82 := 
by 
  sorry

end math_problem_l2268_226882


namespace unoccupied_volume_of_tank_l2268_226862

theorem unoccupied_volume_of_tank (length width height : ℝ) (num_marbles : ℕ) (marble_radius : ℝ) (fill_fraction : ℝ) :
    length = 12 → width = 12 → height = 15 → num_marbles = 5 → marble_radius = 1.5 → fill_fraction = 1/3 →
    (length * width * height * (1 - fill_fraction) - num_marbles * (4 / 3 * Real.pi * marble_radius^3) = 1440 - 22.5 * Real.pi) :=
by
  intros
  sorry

end unoccupied_volume_of_tank_l2268_226862


namespace alpha_beta_value_l2268_226873

variable (α β : ℝ)

def quadratic (x : ℝ) := x^2 + 2 * x - 2005

axiom roots_quadratic_eq : quadratic α = 0 ∧ quadratic β = 0

theorem alpha_beta_value :
  α^2 + 3 * α + β = 2003 :=
by sorry

end alpha_beta_value_l2268_226873


namespace rachel_plant_placement_l2268_226884

def num_ways_to_place_plants : ℕ :=
  let plants := ["basil", "basil", "aloe", "cactus"]
  let lamps := ["white", "white", "red", "red"]
  -- we need to compute the number of ways to place 4 plants under 4 lamps
  22

theorem rachel_plant_placement :
  num_ways_to_place_plants = 22 :=
by
  -- Proof omitted for brevity
  sorry

end rachel_plant_placement_l2268_226884


namespace megan_initial_cupcakes_l2268_226845

noncomputable def initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) : Nat :=
  packages * cupcakes_per_package + cupcakes_eaten

theorem megan_initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) :
  packages = 4 → cupcakes_per_package = 7 → cupcakes_eaten = 43 →
  initial_cupcakes packages cupcakes_per_package cupcakes_eaten = 71 :=
by
  intros
  simp [initial_cupcakes]
  sorry

end megan_initial_cupcakes_l2268_226845


namespace isabella_purchases_l2268_226857

def isabella_items_total (alexis_pants alexis_dresses isabella_pants isabella_dresses : ℕ) : ℕ :=
  isabella_pants + isabella_dresses

theorem isabella_purchases
  (alexis_pants : ℕ) (alexis_dresses : ℕ)
  (h_pants : alexis_pants = 21)
  (h_dresses : alexis_dresses = 18)
  (h_ratio : ∀ (x : ℕ), alexis_pants = 3 * x → alexis_dresses = 3 * x):
  isabella_items_total (21 / 3) (18 / 3) = 13 :=
by
  sorry

end isabella_purchases_l2268_226857


namespace paving_stone_width_l2268_226833

theorem paving_stone_width :
  let courtyard_length := 70
  let courtyard_width := 16.5
  let num_paving_stones := 231
  let paving_stone_length := 2.5
  let courtyard_area := courtyard_length * courtyard_width
  let total_area_covered := courtyard_area
  let paving_stone_width := total_area_covered / (paving_stone_length * num_paving_stones)
  paving_stone_width = 2 :=
by
  sorry

end paving_stone_width_l2268_226833


namespace arithmetic_geometric_sequence_problem_l2268_226855

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (q : ℚ)
  (h1 : ∀ n m : ℕ, a (n + m) = a n * (q ^ m))
  (h2 : a 2 * a 3 * a 4 = 27 / 64)
  (h3 : q = 2)
  (h4 : ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h5 : b 7 = a 5) : 
  b 3 + b 11 = 6 := 
sorry

end arithmetic_geometric_sequence_problem_l2268_226855


namespace sum_of_exponents_sqrt_l2268_226830

theorem sum_of_exponents_sqrt (a b c : ℕ) : 2 + 4 + 6 = 12 := by
  sorry

end sum_of_exponents_sqrt_l2268_226830


namespace division_problem_l2268_226848

theorem division_problem (n : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_div : divisor = 12) (h_quo : quotient = 9) (h_rem : remainder = 1) 
  (h_eq: n = divisor * quotient + remainder) : n = 109 :=
by
  sorry

end division_problem_l2268_226848


namespace distance_from_hut_to_station_l2268_226870

variable (t s : ℝ)

theorem distance_from_hut_to_station
  (h1 : s / 4 = t + 3 / 4)
  (h2 : s / 6 = t - 1 / 2) :
  s = 15 := by
  sorry

end distance_from_hut_to_station_l2268_226870


namespace line_equation_passes_through_l2268_226892

theorem line_equation_passes_through (a b : ℝ) (x y : ℝ) 
  (h_intercept : b = a + 1)
  (h_point : (6 * b) + (-2 * a) = a * b) :
  (x + 2 * y - 2 = 0 ∨ 2 * x + 3 * y - 6 = 0) := 
sorry

end line_equation_passes_through_l2268_226892


namespace initial_distances_l2268_226840

theorem initial_distances (x y : ℝ) 
  (h1: x^2 + y^2 = 400)
  (h2: (x - 6)^2 + (y - 8)^2 = 100) : 
  x = 12 ∧ y = 16 := 
by 
  sorry

end initial_distances_l2268_226840


namespace money_left_after_transactions_l2268_226880

-- Define the coin values and quantities
def dimes := 50
def quarters := 24
def nickels := 40
def pennies := 75

-- Define the item costs
def candy_bar_cost := 6 * 10 + 4 * 5 + 5
def lollipop_cost := 25 + 2 * 10 + 10 - 5 
def bag_of_chips_cost := 2 * 25 + 3 * 10 + 15
def bottle_of_soda_cost := 25 + 6 * 10 + 5 * 5 + 20 - 5

-- Define the number of items bought
def num_candy_bars := 6
def num_lollipops := 3
def num_bags_of_chips := 4
def num_bottles_of_soda := 2

-- Define the initial total money
def total_money := (dimes * 10) + (quarters * 25) + (nickels * 5) + (pennies)

-- Calculate the total cost of items
def total_cost := num_candy_bars * candy_bar_cost + num_lollipops * lollipop_cost + num_bags_of_chips * bag_of_chips_cost + num_bottles_of_soda * bottle_of_soda_cost

-- Calculate the money left after transactions
def money_left := total_money - total_cost

-- Theorem statement to prove
theorem money_left_after_transactions : money_left = 85 := by
  sorry

end money_left_after_transactions_l2268_226880


namespace trigonometric_identity_l2268_226815

theorem trigonometric_identity (α : ℝ) : 
  - (Real.sin α) + (Real.sqrt 3) * (Real.cos α) = 2 * (Real.sin (α + 2 * Real.pi / 3)) :=
by
  sorry

end trigonometric_identity_l2268_226815


namespace function_property_l2268_226878

variable (g : ℝ × ℝ → ℝ)
variable (cond : ∀ x y : ℝ, g (x, y) = - g (y, x))

theorem function_property (x : ℝ) : g (x, x) = 0 :=
by
  sorry

end function_property_l2268_226878


namespace reciprocal_relation_l2268_226808

theorem reciprocal_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := 
by
  sorry

end reciprocal_relation_l2268_226808


namespace problem1_problem2_problem3_l2268_226899

-- Problem Conditions
def inductive_reasoning (s: Sort _) (g: Sort _) : Prop := 
  ∀ (x: s → g), true 

def probabilistic_conclusion : Prop :=
  ∀ (x : Prop), true

def analogical_reasoning (a: Sort _) : Prop := 
  ∀ (x: a), true 

-- The Statements to be Proved
theorem problem1 : ¬ inductive_reasoning Prop Prop = true := 
sorry

theorem problem2 : probabilistic_conclusion = true :=
sorry 

theorem problem3 : ¬ analogical_reasoning Prop = true :=
sorry 

end problem1_problem2_problem3_l2268_226899


namespace students_in_circle_l2268_226853

theorem students_in_circle (n : ℕ) (h1 : n > 6) (h2 : n > 16) (h3 : n / 2 = 10) : n + 2 = 22 := by
  sorry

end students_in_circle_l2268_226853


namespace cone_angle_l2268_226895

theorem cone_angle (r l : ℝ) (α : ℝ)
  (h1 : 2 * Real.pi * r = Real.pi * l) 
  (h2 : Real.cos α = r / l) : α = Real.pi / 3 :=
by
  sorry

end cone_angle_l2268_226895


namespace intersection_A_B_l2268_226863

open Set

def setA : Set ℕ := {x | x - 4 < 0}
def setB : Set ℕ := {0, 1, 3, 4}

theorem intersection_A_B : setA ∩ setB = {0, 1, 3} := by
  sorry

end intersection_A_B_l2268_226863


namespace fettuccine_to_penne_ratio_l2268_226888

theorem fettuccine_to_penne_ratio
  (num_surveyed : ℕ)
  (num_spaghetti : ℕ)
  (num_ravioli : ℕ)
  (num_fettuccine : ℕ)
  (num_penne : ℕ)
  (h_surveyed : num_surveyed = 800)
  (h_spaghetti : num_spaghetti = 300)
  (h_ravioli : num_ravioli = 200)
  (h_fettuccine : num_fettuccine = 150)
  (h_penne : num_penne = 150) :
  num_fettuccine / num_penne = 1 :=
by
  sorry

end fettuccine_to_penne_ratio_l2268_226888


namespace price_and_max_units_proof_l2268_226813

/-- 
Given the conditions of purchasing epidemic prevention supplies: 
- 60 units of type A and 45 units of type B costing 1140 yuan
- 45 units of type A and 30 units of type B costing 840 yuan
- A total of 600 units with a cost not exceeding 8000 yuan

Prove:
1. The price of each unit of type A is 16 yuan, and type B is 4 yuan.
2. The maximum number of units of type A that can be purchased is 466.
--/
theorem price_and_max_units_proof 
  (x y : ℕ) 
  (m : ℕ)
  (h1 : 60 * x + 45 * y = 1140) 
  (h2 : 45 * x + 30 * y = 840) 
  (h3 : 16 * m + 4 * (600 - m) ≤ 8000) 
  (h4 : m ≤ 600) :
  x = 16 ∧ y = 4 ∧ m = 466 := 
by 
  sorry

end price_and_max_units_proof_l2268_226813


namespace forty_percent_of_number_l2268_226807

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : (40/100) * N = 192 :=
by
  sorry

end forty_percent_of_number_l2268_226807


namespace olympiad_problem_l2268_226859

variable (a b c d : ℕ)
variable (N : ℕ := a + b + c + d)

theorem olympiad_problem
  (h1 : (a + d) / (N:ℚ) = 0.5)
  (h2 : (b + d) / (N:ℚ) = 0.6)
  (h3 : (c + d) / (N:ℚ) = 0.7)
  : (d : ℚ) / N * 100 = 40 := by
  sorry

end olympiad_problem_l2268_226859


namespace positive_difference_arithmetic_sequence_l2268_226876

theorem positive_difference_arithmetic_sequence :
  let a := 3
  let d := 5
  let a₁₀₀ := a + (100 - 1) * d
  let a₁₁₀ := a + (110 - 1) * d
  a₁₁₀ - a₁₀₀ = 50 :=
by
  sorry

end positive_difference_arithmetic_sequence_l2268_226876


namespace coordinates_of_B_l2268_226858

structure Point where
  x : Float
  y : Float

def symmetricWithRespectToY (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem coordinates_of_B (A B : Point) 
  (hA : A.x = 2 ∧ A.y = -5)
  (h_sym : symmetricWithRespectToY A B) :
  B.x = -2 ∧ B.y = -5 :=
by
  sorry

end coordinates_of_B_l2268_226858


namespace ratio_of_40_to_8_l2268_226897

theorem ratio_of_40_to_8 : 40 / 8 = 5 := 
by
  sorry

end ratio_of_40_to_8_l2268_226897


namespace geom_seq_a6_value_l2268_226856

variable {α : Type _} [LinearOrderedField α]

theorem geom_seq_a6_value (a : ℕ → α) (q : α) 
(h_geom : ∀ n, a (n + 1) = a n * q)
(h_cond : a 4 + a 8 = π) : 
a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end geom_seq_a6_value_l2268_226856


namespace intersection_of_A_and_B_l2268_226824

def setA : Set ℝ := {x : ℝ | |x| > 1}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : setA ∩ setB = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_A_and_B_l2268_226824


namespace find_n_l2268_226861

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
by {
  sorry
}

end find_n_l2268_226861


namespace sunglasses_price_l2268_226887

theorem sunglasses_price (P : ℝ) 
  (buy_cost_per_pair : ℝ := 26) 
  (pairs_sold : ℝ := 10) 
  (sign_cost : ℝ := 20) :
  (pairs_sold * P - pairs_sold * buy_cost_per_pair) / 2 = sign_cost →
  P = 30 := 
by
  sorry

end sunglasses_price_l2268_226887


namespace students_total_l2268_226806

def num_girls : ℕ := 11
def num_boys : ℕ := num_girls + 5

theorem students_total : num_girls + num_boys = 27 := by
  sorry

end students_total_l2268_226806


namespace solve_for_y_l2268_226871

variable (x y z : ℝ)

theorem solve_for_y (h : 3 * x + 3 * y + 3 * z + 11 = 143) : y = 44 - x - z :=
by 
  sorry

end solve_for_y_l2268_226871


namespace complex_number_problem_l2268_226818

open Complex -- Open the complex numbers namespace

theorem complex_number_problem 
  (z1 z2 : ℂ) 
  (h_z1 : z1 = 2 - I) 
  (h_z2 : z2 = -I) : 
  z1 / z2 + Complex.abs z2 = 2 + 2 * I := by
-- Definitions and conditions directly from (a)
  rw [h_z1, h_z2] -- Replace z1 and z2 with their given values
  sorry -- Proof to be filled in place of the solution steps

end complex_number_problem_l2268_226818


namespace all_equal_l2268_226838

theorem all_equal (a : Fin 100 → ℝ) 
  (h1 : a 0 - 3 * a 1 + 2 * a 2 ≥ 0)
  (h2 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0)
  (h3 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0)
  -- ...
  (h99: a 98 - 3 * a 99 + 2 * a 0 ≥ 0)
  (h100: a 99 - 3 * a 0 + 2 * a 1 ≥ 0) : 
    ∀ i : Fin 100, a i = a 0 := 
by 
  sorry

end all_equal_l2268_226838


namespace rational_coefficients_terms_count_l2268_226839

theorem rational_coefficients_terms_count : 
  (∃ s : Finset ℕ, ∀ k ∈ s, k % 20 = 0 ∧ k ≤ 725 ∧ s.card = 37) :=
by
  -- Translates to finding the set of all k satisfying the condition and 
  -- ensuring it has a cardinality of 37.
  sorry

end rational_coefficients_terms_count_l2268_226839


namespace incorrect_relationship_f_pi4_f_pi_l2268_226832

open Real

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_derivative_exists : ∀ x : ℝ, DifferentiableAt ℝ f x
axiom f_derivative_lt_sin2x : ∀ x : ℝ, 0 < x → deriv f x < (sin x) ^ 2
axiom f_symmetric_property : ∀ x : ℝ, f (-x) + f x = 2 * (sin x) ^ 2

theorem incorrect_relationship_f_pi4_f_pi : ¬ (f (π / 4) < f π) :=
by sorry

end incorrect_relationship_f_pi4_f_pi_l2268_226832


namespace number_of_dvds_remaining_l2268_226835

def initial_dvds : ℕ := 850

def week1_rented : ℕ := (initial_dvds * 25) / 100
def week1_sold : ℕ := 15
def remaining_after_week1 : ℕ := initial_dvds - week1_rented - week1_sold

def week2_rented : ℕ := (remaining_after_week1 * 35) / 100
def week2_sold : ℕ := 25
def remaining_after_week2 : ℕ := remaining_after_week1 - week2_rented - week2_sold

def week3_rented : ℕ := (remaining_after_week2 * 50) / 100
def week3_sold : ℕ := (remaining_after_week2 - week3_rented) * 5 / 100
def remaining_after_week3 : ℕ := remaining_after_week2 - week3_rented - week3_sold

theorem number_of_dvds_remaining : remaining_after_week3 = 181 :=
by
  -- proof goes here
  sorry

end number_of_dvds_remaining_l2268_226835


namespace analyze_properties_l2268_226844

noncomputable def eq_condition (x a : ℝ) : Prop :=
x ≠ 0 ∧ a = (x - 1) / (x^2)

noncomputable def first_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x = 1

noncomputable def second_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x > 1

noncomputable def third_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x < 1

theorem analyze_properties (x a : ℝ) (h1 : eq_condition x a):
(first_condition x a) ∧ ¬(second_condition x a) ∧ ¬(third_condition x a) :=
by
  sorry

end analyze_properties_l2268_226844


namespace prove_all_perfect_squares_l2268_226821

noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k^2 = n

noncomputable def all_distinct (l : List ℕ) : Prop :=
l.Nodup

noncomputable def pairwise_products_are_perfect_squares (l : List ℕ) : Prop :=
∀ i j, i < l.length → j < l.length → i ≠ j → is_perfect_square (l.nthLe i sorry * l.nthLe j sorry)

theorem prove_all_perfect_squares :
  ∀ l : List ℕ, l.length = 25 →
  (∀ x ∈ l, x ≤ 1000 ∧ 0 < x) →
  all_distinct l →
  pairwise_products_are_perfect_squares l →
  ∀ x ∈ l, is_perfect_square x := 
by
  intros l h1 h2 h3 h4
  sorry

end prove_all_perfect_squares_l2268_226821


namespace bob_calories_consumed_l2268_226896

theorem bob_calories_consumed 
  (total_slices : ℕ)
  (half_slices : ℕ)
  (calories_per_slice : ℕ) 
  (H1 : total_slices = 8) 
  (H2 : half_slices = total_slices / 2) 
  (H3 : calories_per_slice = 300) : 
  half_slices * calories_per_slice = 1200 := 
by 
  sorry

end bob_calories_consumed_l2268_226896


namespace circle_area_from_circumference_l2268_226868

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l2268_226868


namespace unique_solution_l2268_226828

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem unique_solution (x y : ℕ) :
  is_prime x →
  is_odd y →
  x^2 + y = 2007 →
  (x = 2 ∧ y = 2003) :=
by
  sorry

end unique_solution_l2268_226828


namespace LittleRed_system_of_eqns_l2268_226814

theorem LittleRed_system_of_eqns :
  ∃ (x y : ℝ), (2/60) * x + (3/60) * y = 1.5 ∧ x + y = 18 :=
sorry

end LittleRed_system_of_eqns_l2268_226814


namespace problem_statement_l2268_226809

def product_of_first_n (n : ℕ) : ℕ := List.prod (List.range' 1 n)

def sum_of_first_n (n : ℕ) : ℕ := List.sum (List.range' 1 n)

theorem problem_statement : 
  let numerator := product_of_first_n 9  -- product of numbers 1 through 8
  let denominator := sum_of_first_n 9  -- sum of numbers 1 through 8
  numerator / denominator = 1120 :=
by {
  sorry
}

end problem_statement_l2268_226809


namespace find_a_evaluate_expr_l2268_226865

-- Given polynomials A and B
def A (a x y : ℝ) : ℝ := a * x^2 + 3 * x * y + 2 * |a| * x
def B (x y : ℝ) : ℝ := 2 * x^2 + 6 * x * y + 4 * x + y + 1

-- Statement part (1)
theorem find_a (a : ℝ) (x y : ℝ) (h : (2 * A a x y - B x y) = (2 * a - 2) * x^2 + (4 * |a| - 4) * x - y - 1) : a = -1 := 
  sorry

-- Expression for part (2)
def expr (a : ℝ) : ℝ := 3 * (-3 * a^2 - 2 * a) - (a^2 - 2 * (5 * a - 4 * a^2 + 1) - 2 * a)

-- Statement part (2)
theorem evaluate_expr : expr (-1) = -22 := 
  sorry

end find_a_evaluate_expr_l2268_226865


namespace omega_terms_sum_to_zero_l2268_226820

theorem omega_terms_sum_to_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 :=
by sorry

end omega_terms_sum_to_zero_l2268_226820


namespace speed_of_current_l2268_226872

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end speed_of_current_l2268_226872


namespace integer_solutions_abs_inequality_l2268_226874

-- Define the condition as a predicate
def abs_inequality_condition (x : ℝ) : Prop := |x - 4| ≤ 3

-- State the proposition
theorem integer_solutions_abs_inequality : ∃ (n : ℕ), n = 7 ∧ ∀ (x : ℤ), abs_inequality_condition x → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7) :=
sorry

end integer_solutions_abs_inequality_l2268_226874


namespace sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l2268_226831

theorem sufficient_but_not_necessary_condition_for_x_lt_3 (x : ℝ) : |x - 1| < 2 → x < 3 :=
by {
  sorry
}

theorem not_necessary_condition_for_x_lt_3 (x : ℝ) : (x < 3) → ¬(-1 < x ∧ x < 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l2268_226831


namespace minimum_value_of_f_l2268_226890

noncomputable def f (x y z : ℝ) := (x^2) / (1 + x) + (y^2) / (1 + y) + (z^2) / (1 + z)

theorem minimum_value_of_f (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
  (h7 : b * z + c * y = a) (h8 : a * z + c * x = b) (h9 : a * y + b * x = c) : 
  f x y z ≥ 1 / 2 :=
sorry

end minimum_value_of_f_l2268_226890


namespace part_A_part_C_part_D_l2268_226836

noncomputable def f : ℝ → ℝ := sorry -- define f with given properties

-- Given conditions
axiom mono_incr_on_neg1_0 : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x < y → f x < f y
axiom symmetry_about_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom symmetry_about_2_0 : ∀ x : ℝ, f (2 + x) = -f (2 - x)

-- Prove the statements
theorem part_A : f 0 = f (-2) := sorry
theorem part_C : ∀ x y : ℝ, 2 < x → x < 3 → 2 < y → y < 3 → x < y → f x > f y := sorry
theorem part_D : f 2021 > f 2022 ∧ f 2022 > f 2023 := sorry

end part_A_part_C_part_D_l2268_226836


namespace max_in_circle_eqn_l2268_226802

theorem max_in_circle_eqn : 
  ∀ (x y : ℝ), (x ≥ 0) → (y ≥ 0) → (4 * x + 3 * y ≤ 12) → (x - 1)^2 + (y - 1)^2 = 1 :=
by
  intros x y hx hy hineq
  sorry

end max_in_circle_eqn_l2268_226802


namespace quadratic_root_property_l2268_226841

theorem quadratic_root_property (a x1 x2 : ℝ) 
  (h_eq : ∀ x, a * x^2 - (3 * a + 1) * x + 2 * (a + 1) = 0)
  (h_distinct : x1 ≠ x2)
  (h_relation : x1 - x1 * x2 + x2 = 1 - a) : a = -1 :=
sorry

end quadratic_root_property_l2268_226841


namespace total_beetles_eaten_each_day_l2268_226822

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end total_beetles_eaten_each_day_l2268_226822


namespace f1_g1_eq_one_l2268_226883

-- Definitions of even and odd functions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given statement to be proved
theorem f1_g1_eq_one (f g : ℝ → ℝ) (h_even : even_function f) (h_odd : odd_function g)
    (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 :=
  sorry

end f1_g1_eq_one_l2268_226883


namespace hyperbola_eccentricity_l2268_226894

noncomputable def point_on_hyperbola (x y a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def focal_length (a b c : ℝ) : Prop :=
  2 * c = 4

noncomputable def eccentricity (e c a : ℝ) : Prop :=
  e = c / a

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_point_on_hyperbola : point_on_hyperbola 2 3 a b h_pos_a h_pos_b)
  (h_focal_length : focal_length a b c)
  : eccentricity e c a :=
sorry -- proof omitted

end hyperbola_eccentricity_l2268_226894


namespace value_of_a_l2268_226826

noncomputable def f (a : ℝ) (x : ℝ) := (x-1)*(x^2 - 3*x + a)

-- Define the condition that 1 is not a critical point
def not_critical (a : ℝ) : Prop := f a 1 ≠ 0

theorem value_of_a (a : ℝ) (h : not_critical a) : a = 2 := 
sorry

end value_of_a_l2268_226826


namespace domain_of_f_decreasing_on_interval_range_of_f_l2268_226851

noncomputable def f (x : ℝ) : ℝ := Real.log (3 + 2 * x - x^2) / Real.log 2

theorem domain_of_f :
  ∀ x : ℝ, (3 + 2 * x - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
by
  sorry

theorem decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), (1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3) →
  f x₂ < f x₁ :=
by
  sorry

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, -1 < x ∧ x < 3 ∧ y = f x) ↔ y ≤ 2 :=
by
  sorry

end domain_of_f_decreasing_on_interval_range_of_f_l2268_226851


namespace general_term_formula_l2268_226825

variable {a : ℕ → ℝ} -- Define the sequence as a function ℕ → ℝ

-- Conditions
axiom geom_seq (n : ℕ) (h : n ≥ 2): a (n + 1) = a 2 * (2 : ℝ) ^ (n - 1)
axiom a2_eq_2 : a 2 = 2
axiom a3_a4_cond : 2 * a 3 + a 4 = 16

theorem general_term_formula (n : ℕ) : a n = 2 ^ (n - 1) := by
  sorry -- Proof is not required

end general_term_formula_l2268_226825


namespace train_length_l2268_226829

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36 * 1000 / 3600) (h2 : time = 14.998800095992321) :
  speed * time = 149.99 :=
by {
  sorry
}

end train_length_l2268_226829


namespace max_area_rect_bamboo_fence_l2268_226819

theorem max_area_rect_bamboo_fence (a b : ℝ) (h : a + b = 10) : a * b ≤ 24 :=
by
  sorry

end max_area_rect_bamboo_fence_l2268_226819


namespace scientific_notation_of_203000_l2268_226891

-- Define the number
def n : ℝ := 203000

-- Define the representation of the number in scientific notation
def scientific_notation (a b : ℝ) : Prop := n = a * 10^b ∧ 1 ≤ a ∧ a < 10

-- The theorem to state 
theorem scientific_notation_of_203000 : ∃ a b : ℝ, scientific_notation a b ∧ a = 2.03 ∧ b = 5 :=
by
  use 2.03
  use 5
  sorry

end scientific_notation_of_203000_l2268_226891


namespace intersection_P_Q_intersection_complementP_Q_l2268_226805

-- Define the universal set U
def U := Set.univ (ℝ)

-- Define set P
def P := {x : ℝ | |x| > 2}

-- Define set Q
def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Complement of P with respect to U
def complement_P : Set ℝ := {x : ℝ | |x| ≤ 2}

theorem intersection_P_Q : P ∩ Q = ({x : ℝ | 2 < x ∧ x < 3}) :=
by {
  sorry
}

theorem intersection_complementP_Q : complement_P ∩ Q = ({x : ℝ | 1 < x ∧ x ≤ 2}) :=
by {
  sorry
}

end intersection_P_Q_intersection_complementP_Q_l2268_226805


namespace find_x_l2268_226842

theorem find_x (x : ℝ) : (x / 4 * 5 + 10 - 12 = 48) → (x = 40) :=
by
  sorry

end find_x_l2268_226842


namespace find_biology_marks_l2268_226869

variable (english mathematics physics chemistry average_marks : ℕ)

theorem find_biology_marks
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_physics : physics = 92)
  (h_chemistry : chemistry = 87)
  (h_average_marks : average_marks = 89) : 
  (english + mathematics + physics + chemistry + (445 - (english + mathematics + physics + chemistry))) / 5 = average_marks :=
by
  sorry

end find_biology_marks_l2268_226869


namespace quadratic_roots_value_l2268_226860

theorem quadratic_roots_value (d : ℝ) 
  (h : ∀ x : ℝ, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) : 
  d = 9.8 :=
by 
  sorry

end quadratic_roots_value_l2268_226860


namespace estimate_sqrt_expression_l2268_226879

theorem estimate_sqrt_expression :
  5 < 3 * Real.sqrt 5 - 1 ∧ 3 * Real.sqrt 5 - 1 < 6 :=
by
  sorry

end estimate_sqrt_expression_l2268_226879


namespace circle_radius_triple_area_l2268_226804

noncomputable def circle_radius (n : ℝ) : ℝ :=
  let r := (n * (Real.sqrt 3 + 1)) / 2
  r

theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) :
  r = (n * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end circle_radius_triple_area_l2268_226804


namespace prove_a₈_l2268_226812

noncomputable def first_term (a : ℕ → ℝ) : Prop := a 1 = 3
noncomputable def arithmetic_b (a b : ℕ → ℝ) : Prop := ∀ n, b n = a (n + 1) - a n
noncomputable def b_conditions (b : ℕ → ℝ) : Prop := b 3 = -2 ∧ b 10 = 12

theorem prove_a₈ (a b : ℕ → ℝ) (h1 : first_term a) (h2 : arithmetic_b a b) (h3 : b_conditions b) :
  a 8 = 3 :=
sorry

end prove_a₈_l2268_226812


namespace time_to_drain_tank_due_to_leak_l2268_226837

noncomputable def timeToDrain (P L : ℝ) : ℝ := (1 : ℝ) / L

theorem time_to_drain_tank_due_to_leak (P L : ℝ)
  (hP : P = 0.5)
  (hL : P - L = 5/11) :
  timeToDrain P L = 22 :=
by
  -- to state what needs to be proved here
  sorry

end time_to_drain_tank_due_to_leak_l2268_226837


namespace Robie_boxes_with_him_l2268_226898

-- Definition of the given conditions
def total_cards : Nat := 75
def cards_per_box : Nat := 10
def cards_not_placed : Nat := 5
def boxes_given_away : Nat := 2

-- Definition of the proof that Robie has 5 boxes with him
theorem Robie_boxes_with_him : ((total_cards - cards_not_placed) / cards_per_box) - boxes_given_away = 5 := by
  sorry

end Robie_boxes_with_him_l2268_226898


namespace max_take_home_pay_income_l2268_226834

theorem max_take_home_pay_income (x : ℤ) : 
  (1000 * 2 * 50) - 20 * 50^2 = 100000 := 
by 
  sorry

end max_take_home_pay_income_l2268_226834


namespace lucas_change_l2268_226893

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end lucas_change_l2268_226893


namespace rate_of_first_car_l2268_226827

theorem rate_of_first_car
  (r : ℕ) (h1 : 3 * r + 30 = 180) : r = 50 :=
sorry

end rate_of_first_car_l2268_226827


namespace half_hour_half_circle_half_hour_statement_is_true_l2268_226843

-- Definitions based on conditions
def half_circle_divisions : ℕ := 30
def small_divisions_per_minute : ℕ := 1
def total_small_divisions : ℕ := 60
def minutes_per_circle : ℕ := 60

-- Relation of small divisions and time taken
def time_taken_for_small_divisions (divs : ℕ) : ℕ := divs * small_divisions_per_minute

-- Theorem to prove the statement
theorem half_hour_half_circle : time_taken_for_small_divisions half_circle_divisions = 30 :=
by
  -- Given half circle covers 30 small divisions
  -- Each small division represents 1 minute
  -- Therefore, time taken for 30 divisions should be 30 minutes
  exact rfl

-- The final statement proving the truth of the condition
theorem half_hour_statement_is_true : 
  (time_taken_for_small_divisions half_circle_divisions = 30) → True :=
by
  intro h
  trivial

end half_hour_half_circle_half_hour_statement_is_true_l2268_226843


namespace compute_expression_l2268_226850

theorem compute_expression : 7^2 - 5 * 6 + 6^2 = 55 := by
  sorry

end compute_expression_l2268_226850


namespace transformed_quadratic_roots_l2268_226886

-- Definitions of the conditions
def quadratic_roots (a b : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + 3 = 0 → (x = -2) ∨ (x = 3)

-- Statement of the theorem
theorem transformed_quadratic_roots (a b : ℝ) :
  quadratic_roots a b →
  ∀ x : ℝ, a * (x + 2)^2 + b * (x + 2) + 3 = 0 → (x = -4) ∨ (x = 1) :=
sorry

end transformed_quadratic_roots_l2268_226886
