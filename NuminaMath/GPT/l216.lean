import Mathlib

namespace meal_cost_one_burger_one_shake_one_cola_l216_216635

-- Define the costs of individual items
variables (B S C : ℝ)

-- Conditions based on given equations
def eq1 : Prop := 3 * B + 7 * S + C = 120
def eq2 : Prop := 4 * B + 10 * S + C = 160.50

-- Goal: Prove that the total cost of one burger, one shake, and one cola is $39
theorem meal_cost_one_burger_one_shake_one_cola :
  eq1 B S C → eq2 B S C → B + S + C = 39 :=
by 
  intros 
  sorry

end meal_cost_one_burger_one_shake_one_cola_l216_216635


namespace cost_of_eraser_pencil_l216_216222

-- Define the cost of regular and short pencils
def cost_regular_pencil : ℝ := 0.5
def cost_short_pencil : ℝ := 0.4

-- Define the quantities sold
def quantity_eraser_pencils : ℕ := 200
def quantity_regular_pencils : ℕ := 40
def quantity_short_pencils : ℕ := 35

-- Define the total revenue
def total_revenue : ℝ := 194

-- Problem statement: Prove that the cost of a pencil with an eraser is 0.8
theorem cost_of_eraser_pencil (P : ℝ)
  (h : 200 * P + 40 * cost_regular_pencil + 35 * cost_short_pencil = total_revenue) :
  P = 0.8 := by
  sorry

end cost_of_eraser_pencil_l216_216222


namespace find_unknown_rate_l216_216852

theorem find_unknown_rate :
    let n := 7 -- total number of blankets
    let avg_price := 150 -- average price of the blankets
    let total_price := n * avg_price
    let cost1 := 3 * 100
    let cost2 := 2 * 150
    let remaining := total_price - (cost1 + cost2)
    remaining / 2 = 225 :=
by sorry

end find_unknown_rate_l216_216852


namespace percentage_decrease_in_sale_l216_216618

theorem percentage_decrease_in_sale (P Q : ℝ) (D : ℝ)
  (h1 : 1.80 * P * Q * (1 - D / 100) = 1.44 * P * Q) : 
  D = 20 :=
by
  -- Proof goes here
  sorry

end percentage_decrease_in_sale_l216_216618


namespace number_of_set_B_l216_216168

theorem number_of_set_B (U A B : Finset ℕ) (hU : U.card = 193) (hA_inter_B : (A ∩ B).card = 25) (hA : A.card = 110) (h_not_in_A_or_B : 193 - (A ∪ B).card = 59) : B.card = 49 := 
by
  sorry

end number_of_set_B_l216_216168


namespace find_angle_A_find_side_a_l216_216424

-- Define the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}
-- Assumption conditions in the problem
variables (h₁ : a * sin B = sqrt 3 * b * cos A)
variables (hb : b = 3)
variables (hc : c = 2)

-- Prove that A = π / 3 given the first condition
theorem find_angle_A : h₁ → A = π / 3 := by
  -- Proof is omitted
  sorry

-- Prove that a = sqrt 7 given b = 3, c = 2, and A = π / 3
theorem find_side_a : h₁ → hb → hc → a = sqrt 7 := by
  -- Proof is omitted
  sorry

end find_angle_A_find_side_a_l216_216424


namespace correct_multiple_l216_216075

theorem correct_multiple (n : ℝ) (m : ℝ) (h1 : n = 6) (h2 : m * n - 6 = 2 * n) : m * n = 18 :=
by
  sorry

end correct_multiple_l216_216075


namespace zachary_more_crunches_than_pushups_l216_216855

def zachary_pushups : ℕ := 46
def zachary_crunches : ℕ := 58
def zachary_crunches_more_than_pushups : ℕ := zachary_crunches - zachary_pushups

theorem zachary_more_crunches_than_pushups : zachary_crunches_more_than_pushups = 12 := by
  sorry

end zachary_more_crunches_than_pushups_l216_216855


namespace hexagon_tiling_colors_l216_216730

-- Problem Definition
theorem hexagon_tiling_colors (k l : ℕ) (hk : 0 < k ∨ 0 < l) : 
  ∃ n: ℕ, n = k^2 + k * l + l^2 :=
by
  sorry

end hexagon_tiling_colors_l216_216730


namespace pure_imaginary_z_squared_l216_216253

-- Formalization in Lean 4
theorem pure_imaginary_z_squared (a : ℝ) (h : a + (1 + a) * I = (1 + a) * I) : (a + (1 + a) * I)^2 = -1 :=
by
  sorry

end pure_imaginary_z_squared_l216_216253


namespace certain_number_is_four_l216_216500

theorem certain_number_is_four (k : ℕ) (h₁ : k = 16) : 64 / k = 4 :=
by
  sorry

end certain_number_is_four_l216_216500


namespace find_x_l216_216950

theorem find_x : ∃ (x : ℚ), (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end find_x_l216_216950


namespace middle_value_bounds_l216_216308

theorem middle_value_bounds (a b c : ℝ) (h1 : a + b + c = 10)
  (h2 : a > b) (h3 : b > c) (h4 : a - c = 3) : 
  7 / 3 < b ∧ b < 13 / 3 :=
by
  sorry

end middle_value_bounds_l216_216308


namespace seventh_term_of_arithmetic_sequence_l216_216192

theorem seventh_term_of_arithmetic_sequence (a d : ℤ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 6) : 
  a + 6 * d = 7 :=
by
  -- Proof omitted
  sorry

end seventh_term_of_arithmetic_sequence_l216_216192


namespace greatest_divisor_6215_7373_l216_216859

theorem greatest_divisor_6215_7373 : 
  Nat.gcd (6215 - 23) (7373 - 29) = 144 := by
  sorry

end greatest_divisor_6215_7373_l216_216859


namespace complement_of_A_with_respect_to_U_l216_216466

namespace SetTheory

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def C_UA : Set ℕ := {3, 4, 5}

theorem complement_of_A_with_respect_to_U :
  (U \ A) = C_UA := by
  sorry

end SetTheory

end complement_of_A_with_respect_to_U_l216_216466


namespace taxi_ride_cost_l216_216531

theorem taxi_ride_cost (base_fare : ℝ) (rate_per_mile : ℝ) (additional_charge : ℝ) (distance : ℕ) (cost : ℝ) :
  base_fare = 2 ∧ rate_per_mile = 0.30 ∧ additional_charge = 5 ∧ distance = 12 ∧ 
  cost = base_fare + (rate_per_mile * distance) + additional_charge → cost = 10.60 :=
by
  intros
  sorry

end taxi_ride_cost_l216_216531


namespace ab_zero_l216_216713

theorem ab_zero
  (a b : ℤ)
  (h : ∀ (m n : ℕ), ∃ (k : ℤ), a * (m : ℤ) ^ 2 + b * (n : ℤ) ^ 2 = k ^ 2) :
  a * b = 0 :=
sorry

end ab_zero_l216_216713


namespace sum_of_numbers_given_average_l216_216627

variable (average : ℝ) (n : ℕ) (sum : ℝ)

theorem sum_of_numbers_given_average (h1 : average = 4.1) (h2 : n = 6) (h3 : average = sum / n) :
  sum = 24.6 :=
by
  sorry

end sum_of_numbers_given_average_l216_216627


namespace number_of_squares_l216_216388

-- Define the conditions and the goal
theorem number_of_squares {x : ℤ} (hx0 : 0 ≤ x) (hx6 : x ≤ 6) {y : ℤ} (hy0 : -1 ≤ y) (hy : y ≤ 3 * x) :
  ∃ (n : ℕ), n = 123 :=
by 
  sorry

end number_of_squares_l216_216388


namespace g_of_zero_l216_216586

theorem g_of_zero (f g : ℤ → ℤ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) : 
  g 0 = -1 :=
by
  sorry

end g_of_zero_l216_216586


namespace rectangle_side_deficit_l216_216539

theorem rectangle_side_deficit (L W : ℝ) (p : ℝ)
  (h1 : 1.05 * L * (1 - p) * W - L * W = 0.8 / 100 * L * W)
  (h2 : 0 < L) (h3 : 0 < W) : p = 0.04 :=
by {
  sorry
}

end rectangle_side_deficit_l216_216539


namespace find_tan_alpha_plus_pi_div_12_l216_216207

theorem find_tan_alpha_plus_pi_div_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + Real.pi / 6)) :
  Real.tan (α + Real.pi / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end find_tan_alpha_plus_pi_div_12_l216_216207


namespace pizza_fraction_eaten_l216_216471

theorem pizza_fraction_eaten :
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  (a * (1 - r ^ n) / (1 - r)) = 63 / 128 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  sorry

end pizza_fraction_eaten_l216_216471


namespace determine_marbles_l216_216276

noncomputable def marbles_total (x : ℚ) := (4 * x + 2) + (2 * x) + (3 * x - 1)

theorem determine_marbles (x : ℚ) (h1 : marbles_total x = 47) :
  (4 * x + 2 = 202 / 9) ∧ (2 * x = 92 / 9) ∧ (3 * x - 1 = 129 / 9) :=
by
  sorry

end determine_marbles_l216_216276


namespace find_fourth_number_in_sequence_l216_216305

-- Define the conditions of the sequence
def first_number : ℤ := 1370
def second_number : ℤ := 1310
def third_number : ℤ := 1070
def fifth_number : ℤ := -6430

-- Define the differences
def difference1 : ℤ := second_number - first_number
def difference2 : ℤ := third_number - second_number

-- Define the ratio of differences
def ratio : ℤ := 4
def next_difference : ℤ := difference2 * ratio

-- Define the fourth number
def fourth_number : ℤ := third_number - (-next_difference)

-- Theorem stating the proof problem
theorem find_fourth_number_in_sequence : fourth_number = 2030 :=
by sorry

end find_fourth_number_in_sequence_l216_216305


namespace one_interior_angle_of_polygon_with_five_diagonals_l216_216290

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end one_interior_angle_of_polygon_with_five_diagonals_l216_216290


namespace number_of_functions_satisfying_conditions_l216_216474

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ s ∈ S, f (f (f s)) = s) ∧ (∀ s ∈ S, (f s - s) % 3 ≠ 0)

theorem number_of_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), f_conditions f) ∧ (∃! (n : ℕ), n = 288) :=
by
  sorry

end number_of_functions_satisfying_conditions_l216_216474


namespace part1_part2_l216_216365

-- Define the main condition of the farthest distance formula
def distance_formula (S h : ℝ) : Prop := S^2 = 1.7 * h

-- Define part 1: Given h = 1.7, prove S = 1.7
theorem part1
  (h : ℝ)
  (hyp : h = 1.7)
  : ∃ S : ℝ, distance_formula S h ∧ S = 1.7 :=
by
  sorry
  
-- Define part 2: Given S = 6.8 and height of eyes to ground 1.5, prove the height of tower = 25.7
theorem part2
  (S : ℝ)
  (h1 : ℝ)
  (height_eyes_to_ground : ℝ)
  (hypS : S = 6.8)
  (height_eyes_to_ground_eq : height_eyes_to_ground = 1.5)
  : ∃ h : ℝ, distance_formula S h ∧ (h - height_eyes_to_ground) = 25.7 :=
by
  sorry

end part1_part2_l216_216365


namespace find_prices_l216_216595

variables (C S : ℕ) -- Using natural numbers to represent rubles

theorem find_prices (h1 : C + S = 2500) (h2 : 4 * C + 3 * S = 8870) :
  C = 1370 ∧ S = 1130 :=
by
  sorry

end find_prices_l216_216595


namespace problem_I_II_l216_216249

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem problem_I_II
  (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : f 0 φ = 1 / 2) :
  ∃ T : ℝ, T = Real.pi ∧ φ = Real.pi / 6 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → (f x φ) ≥ -1 / 2) :=
by
  sorry

end problem_I_II_l216_216249


namespace sequence_sum_l216_216844

theorem sequence_sum (a : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, 0 < a n)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 2) = 1 + 1 / a n)
  (h₃ : a 2014 = a 2016) :
  a 13 + a 2016 = 21 / 13 + (1 + Real.sqrt 5) / 2 :=
sorry

end sequence_sum_l216_216844


namespace ammonium_chloride_potassium_hydroxide_ammonia_l216_216111

theorem ammonium_chloride_potassium_hydroxide_ammonia
  (moles_KOH : ℕ) (moles_NH3 : ℕ) (moles_NH4Cl : ℕ) 
  (reaction : moles_KOH = 3 ∧ moles_NH3 = moles_KOH ∧ moles_NH4Cl >= moles_KOH) : 
  moles_NH3 = 3 :=
by
  sorry

end ammonium_chloride_potassium_hydroxide_ammonia_l216_216111


namespace longest_side_of_region_l216_216441

theorem longest_side_of_region :
  (∃ (x y : ℝ), x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1) →
  (∃ (l : ℝ), l = Real.sqrt 130 / 3 ∧ 
    (l = Real.sqrt ((1 - 1)^2 + (4 - 1)^2) ∨ 
     l = Real.sqrt (((1 + 4 / 3) - 1)^2 + (1 - 1)^2) ∨ 
     l = Real.sqrt ((1 - (1 + 4 / 3))^2 + (1 - 1)^2))) :=
by
  sorry

end longest_side_of_region_l216_216441


namespace at_most_n_pairs_with_distance_d_l216_216864

theorem at_most_n_pairs_with_distance_d
  (n : ℕ) (hn : n ≥ 3)
  (points : Fin n → ℝ × ℝ)
  (d : ℝ)
  (hd : ∀ i j, i ≠ j → dist (points i) (points j) ≤ d)
  (dmax : ∃ i j, i ≠ j ∧ dist (points i) (points j) = d) :
  ∃ (pairs : Finset (Fin n × Fin n)), ∀ p ∈ pairs, dist (points p.1) (points p.2) = d ∧ pairs.card ≤ n := 
sorry

end at_most_n_pairs_with_distance_d_l216_216864


namespace smallest_irreducible_l216_216751

def is_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1

theorem smallest_irreducible : ∃ n : ℕ, is_irreducible n ∧ ∀ m : ℕ, m < n → ¬ is_irreducible m :=
  by
  exists 95
  sorry

end smallest_irreducible_l216_216751


namespace coin_flip_probability_difference_l216_216993

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l216_216993


namespace rowing_distance_l216_216062

noncomputable def effective_speed_with_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed + current_speed

noncomputable def effective_speed_against_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed - current_speed

noncomputable def distance (speed time : ℕ) : ℕ :=
  speed * time

theorem rowing_distance (rowing_speed current_speed total_time : ℕ) 
  (hrowing_speed : rowing_speed = 10)
  (hcurrent_speed : current_speed = 2)
  (htotal_time : total_time = 30) : 
  (distance 8 18) = 144 := 
by
  sorry

end rowing_distance_l216_216062


namespace sum_of_squares_of_consecutive_integers_l216_216668

theorem sum_of_squares_of_consecutive_integers
  (a : ℤ) (h : (a - 1) * a * (a + 1) = 10 * ((a - 1) + a + (a + 1))) :
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 :=
sorry

end sum_of_squares_of_consecutive_integers_l216_216668


namespace sqrt_expression_simplification_l216_216799

theorem sqrt_expression_simplification : 
  (Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5) = Real.sqrt 3 := 
  by
    sorry

end sqrt_expression_simplification_l216_216799


namespace right_triangle_acute_angles_l216_216294

variable (α β : ℝ)

noncomputable def prove_acute_angles (α β : ℝ) : Prop :=
  α + β = 90 ∧ 4 * α = 90

theorem right_triangle_acute_angles : 
  prove_acute_angles α β → α = 22.5 ∧ β = 67.5 := by
  sorry

end right_triangle_acute_angles_l216_216294


namespace find_e_m_l216_216554

variable {R : Type} [Field R]

def matrix_B (e : R) : Matrix (Fin 2) (Fin 2) R :=
  !![3, 4; 6, e]

theorem find_e_m (e m : R) (hB_inv : (matrix_B e)⁻¹ = m • (matrix_B e)) :
  e = -3 ∧ m = (1 / 11) := by
  sorry

end find_e_m_l216_216554


namespace analytic_expression_of_f_range_of_k_l216_216789

noncomputable def quadratic_function_minimum (a b : ℝ) : ℝ :=
a * (-1) ^ 2 + b * (-1) + 1

theorem analytic_expression_of_f (a b : ℝ) (ha : quadratic_function_minimum a b = 0)
  (hmin: -1 = -b / (2 * a)) : a = 1 ∧ b = 2 :=
by sorry

theorem range_of_k (k : ℝ) : ∃ k : ℝ, (k ∈ Set.Ici 3 ∨ k = 13 / 4) :=
by sorry

end analytic_expression_of_f_range_of_k_l216_216789


namespace max_value_64_l216_216967

-- Define the types and values of gemstones
structure Gemstone where
  weight : ℕ
  value : ℕ

-- Introduction of the three types of gemstones
def gem1 : Gemstone := ⟨3, 9⟩
def gem2 : Gemstone := ⟨5, 16⟩
def gem3 : Gemstone := ⟨2, 5⟩

-- Maximum weight Janet can carry
def max_weight := 20

-- Problem statement: Proving the maximum value Janet can carry is $64
theorem max_value_64 (n1 n2 n3 : ℕ) (h1 : n1 ≥ 15) (h2 : n2 ≥ 15) (h3 : n3 ≥ 15) 
  (weight_limit : n1 * gem1.weight + n2 * gem2.weight + n3 * gem3.weight ≤ max_weight) : 
  n1 * gem1.value + n2 * gem2.value + n3 * gem3.value ≤ 64 :=
sorry

end max_value_64_l216_216967


namespace range_of_sum_l216_216943

theorem range_of_sum (a b : ℝ) (h1 : -2 < a) (h2 : a < -1) (h3 : -1 < b) (h4 : b < 0) : 
  -3 < a + b ∧ a + b < -1 :=
by
  sorry

end range_of_sum_l216_216943


namespace divisibility_by_5_l216_216703

theorem divisibility_by_5 (n : ℕ) (h : 0 < n) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end divisibility_by_5_l216_216703


namespace jen_shooting_game_times_l216_216735

theorem jen_shooting_game_times (x : ℕ) (h1 : 5 * x + 9 = 19) : x = 2 := by
  sorry

end jen_shooting_game_times_l216_216735


namespace infinite_series_sum_l216_216437

theorem infinite_series_sum :
  ∑' n : ℕ, (1 / (n.succ * (n.succ + 2))) = 3 / 4 :=
by sorry

end infinite_series_sum_l216_216437


namespace construct_80_construct_160_construct_20_l216_216060

-- Define the notion of constructibility from an angle
inductive Constructible : ℝ → Prop
| base (a : ℝ) : a = 40 → Constructible a
| add (a b : ℝ) : Constructible a → Constructible b → Constructible (a + b)
| sub (a b : ℝ) : Constructible a → Constructible b → Constructible (a - b)

-- Lean statements for proving the constructibility
theorem construct_80 : Constructible 80 :=
sorry

theorem construct_160 : Constructible 160 :=
sorry

theorem construct_20 : Constructible 20 :=
sorry

end construct_80_construct_160_construct_20_l216_216060


namespace willam_tax_payment_correct_l216_216700

noncomputable def willamFarmTax : ℝ :=
  let totalTax := 3840
  let willamPercentage := 0.2777777777777778
  totalTax * willamPercentage

-- Lean theorem statement for the problem
theorem willam_tax_payment_correct : 
  willamFarmTax = 1066.67 :=
by
  sorry

end willam_tax_payment_correct_l216_216700


namespace farmer_cages_l216_216059

theorem farmer_cages (c : ℕ) (h1 : 164 + 6 = 170) (h2 : ∃ r : ℕ, c * r = 170) (h3 : ∃ r : ℕ, c * r > 164) :
  c = 10 :=
by
  sorry

end farmer_cages_l216_216059


namespace correct_statements_l216_216108

-- A quality inspector takes a sample from a uniformly moving production line every 10 minutes for a certain indicator test.
def statement1 := false -- This statement is incorrect because this is systematic sampling, not stratified sampling.

-- In the frequency distribution histogram, the sum of the areas of all small rectangles is 1.
def statement2 := true -- This is correct.

-- In the regression line equation \(\hat{y} = 0.2x + 12\), when the variable \(x\) increases by one unit, the variable \(y\) definitely increases by 0.2 units.
def statement3 := false -- This is incorrect because y increases on average by 0.2 units, not definitely.

-- For two categorical variables \(X\) and \(Y\), calculating the statistic \(K^2\) and its observed value \(k\), the larger the observed value \(k\), the more confident we are that “X and Y are related”.
def statement4 := true -- This is correct.

-- We need to prove that the correct statements are only statement2 and statement4.
theorem correct_statements : (statement1 = false ∧ statement2 = true ∧ statement3 = false ∧ statement4 = true) → (statement2 ∧ statement4) :=
by sorry

end correct_statements_l216_216108


namespace expand_expression_l216_216109

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l216_216109


namespace cubic_identity_l216_216184

variable (a b c : ℝ)
variable (h1 : a + b + c = 13)
variable (h2 : ab + ac + bc = 30)

theorem cubic_identity : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 :=
by 
  sorry

end cubic_identity_l216_216184


namespace total_cost_is_13_l216_216458

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l216_216458


namespace find_R_plus_S_l216_216802

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end find_R_plus_S_l216_216802


namespace hidden_dots_sum_l216_216988

-- Lean 4 equivalent proof problem definition
theorem hidden_dots_sum (d1 d2 d3 d4 : ℕ)
    (h1 : d1 ≠ d2 ∧ d1 + d2 = 7)
    (h2 : d3 ≠ d4 ∧ d3 + d4 = 7)
    (h3 : d1 = 2 ∨ d1 = 4 ∨ d2 = 2 ∨ d2 = 4)
    (h4 : d3 + 4 = 7) :
    d1 + 7 + 7 + d3 = 24 :=
sorry

end hidden_dots_sum_l216_216988


namespace samuel_has_five_birds_l216_216658

theorem samuel_has_five_birds
  (birds_berries_per_day : ℕ)
  (total_berries_in_4_days : ℕ)
  (n_birds : ℕ)
  (h1 : birds_berries_per_day = 7)
  (h2 : total_berries_in_4_days = 140)
  (h3 : n_birds * birds_berries_per_day * 4 = total_berries_in_4_days) :
  n_birds = 5 := by
  sorry

end samuel_has_five_birds_l216_216658


namespace negation_of_exists_l216_216676

theorem negation_of_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l216_216676


namespace max_S_value_max_S_value_achievable_l216_216036

theorem max_S_value (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) ≤ 8 / 27 :=
sorry

theorem max_S_value_achievable :
  ∃ (x y z w : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) = 8 / 27 :=
sorry

end max_S_value_max_S_value_achievable_l216_216036


namespace distance_between_points_eq_l216_216300

theorem distance_between_points_eq :
  let x1 := 2
  let y1 := -5
  let x2 := -8
  let y2 := 7
  let distance := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  distance = 2 * Real.sqrt 61 :=
by
  sorry

end distance_between_points_eq_l216_216300


namespace sum_of_digits_b_n_l216_216283

def a_n (n : ℕ) : ℕ := 10^(2^n) - 1

def b_n (n : ℕ) : ℕ :=
  List.prod (List.map a_n (List.range (n + 1)))

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_b_n (n : ℕ) : sum_of_digits (b_n n) = 9 * 2^n :=
  sorry

end sum_of_digits_b_n_l216_216283


namespace chess_player_total_games_l216_216269

noncomputable def total_games_played (W L : ℕ) : ℕ :=
  W + L

theorem chess_player_total_games :
  ∃ (W L : ℕ), W = 16 ∧ (L : ℚ) / W = 7 / 4 ∧ total_games_played W L = 44 :=
by
  sorry

end chess_player_total_games_l216_216269


namespace pyramid_volume_l216_216178

noncomputable def volume_pyramid (a b : ℝ) : ℝ :=
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2))

theorem pyramid_volume (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 < 4 * b^2) :
  volume_pyramid a b =
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2)) :=
sorry

end pyramid_volume_l216_216178


namespace intersection_M_N_l216_216369

-- Definitions:
def M := {x : ℝ | 0 ≤ x}
def N := {y : ℝ | -2 ≤ y}

-- The theorem statement:
theorem intersection_M_N : M ∩ N = {z : ℝ | 0 ≤ z} := sorry

end intersection_M_N_l216_216369


namespace truth_of_compound_proposition_l216_216353

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, x^2 > 0

theorem truth_of_compound_proposition : p ∧ ¬ q :=
by
  sorry

end truth_of_compound_proposition_l216_216353


namespace determine_h_l216_216959

noncomputable def h (x : ℝ) : ℝ := -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2

theorem determine_h (x : ℝ) :
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 :=
by
  sorry

end determine_h_l216_216959


namespace baker_total_cost_is_correct_l216_216665

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end baker_total_cost_is_correct_l216_216665


namespace probability_five_digit_palindrome_div_by_11_l216_216835

noncomputable
def five_digit_palindrome_div_by_11_probability : ℚ :=
  let total_palindromes := 900
  let valid_palindromes := 80
  valid_palindromes / total_palindromes

theorem probability_five_digit_palindrome_div_by_11 :
  five_digit_palindrome_div_by_11_probability = 2 / 25 := by
  sorry

end probability_five_digit_palindrome_div_by_11_l216_216835


namespace intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l216_216175

-- Define the solution sets A and B given conditions
def solution_set_A (a : ℝ) : Set ℝ :=
  { x | |x - 1| ≤ a }

def solution_set_B : Set ℝ :=
  { x | (x - 2) * (x + 2) > 0 }

theorem intersection_A_B_when_a_eq_2 :
  solution_set_A 2 ∩ solution_set_B = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ (a : ℝ), solution_set_A a ∩ solution_set_B = ∅ → 0 < a ∧ a ≤ 1 :=
by
  sorry

end intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l216_216175


namespace find_other_intersection_point_l216_216907

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end find_other_intersection_point_l216_216907


namespace opening_night_customers_l216_216475

theorem opening_night_customers
  (matinee_tickets : ℝ := 5)
  (evening_tickets : ℝ := 7)
  (opening_night_tickets : ℝ := 10)
  (popcorn_cost : ℝ := 10)
  (num_matinee_customers : ℝ := 32)
  (num_evening_customers : ℝ := 40)
  (total_revenue : ℝ := 1670) :
  ∃ x : ℝ, 
    (matinee_tickets * num_matinee_customers + 
    evening_tickets * num_evening_customers + 
    opening_night_tickets * x + 
    popcorn_cost * (num_matinee_customers + num_evening_customers + x) / 2 = total_revenue) 
    ∧ x = 58 := 
by
  use 58
  sorry

end opening_night_customers_l216_216475


namespace trader_allows_discount_l216_216757

-- Definitions for cost price, marked price, and selling price
variable (cp : ℝ)
def mp := cp + 0.12 * cp
def sp := cp - 0.01 * cp

-- The statement to prove
theorem trader_allows_discount :
  mp cp - sp cp = 13 :=
sorry

end trader_allows_discount_l216_216757


namespace cannot_form_polygon_l216_216644

-- Define the stick lengths as a list
def stick_lengths : List ℕ := List.range 100 |>.map (λ n => 2^n)

-- Define the condition for forming a polygon
def can_form_polygon (lst : List ℕ) : Prop :=
  ∃ subset, subset ⊆ lst ∧ subset.length ≥ 3 ∧ (∀ s ∈ subset, s < (subset.sum - s))

-- The theorem to be proved
theorem cannot_form_polygon : ¬ can_form_polygon stick_lengths :=
by 
  sorry

end cannot_form_polygon_l216_216644


namespace job_completion_time_l216_216542

theorem job_completion_time (initial_men : ℕ) (initial_days : ℕ) (extra_men : ℕ) (interval_days : ℕ) (total_days : ℕ) : 
  initial_men = 20 → 
  initial_days = 15 → 
  extra_men = 10 → 
  interval_days = 5 → 
  total_days = 12 → 
  ∀ n, (20 * 5 + (20 + 10) * 5 + (20 + 10 + 10) * n.succ = 300 → n + 10 + n.succ = 12) :=
by
  intro h1 h2 h3 h4 h5
  sorry

end job_completion_time_l216_216542


namespace factorial_inequality_l216_216299

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n! ≤ ((n+1)/2)^n := 
by {
  sorry
}

end factorial_inequality_l216_216299


namespace flour_qualification_l216_216508

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end flour_qualification_l216_216508


namespace solve_for_x_l216_216198

def δ (x : ℝ) : ℝ := 4 * x + 5
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_for_x (x : ℝ) (h : δ (φ x) = 4) : x = -17 / 20 := by
  sorry

end solve_for_x_l216_216198


namespace solve_f_g_f_3_l216_216762

def f (x : ℤ) : ℤ := 2 * x + 4

def g (x : ℤ) : ℤ := 5 * x + 2

theorem solve_f_g_f_3 :
  f (g (f 3)) = 108 := by
  sorry

end solve_f_g_f_3_l216_216762


namespace minimum_racing_stripes_l216_216968

variable 
  (totalCars : ℕ) (carsWithoutAirConditioning : ℕ) 
  (maxCarsWithAirConditioningWithoutStripes : ℕ)

-- Defining specific problem conditions
def conditions (totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes : ℕ) : Prop :=
  totalCars = 100 ∧ 
  carsWithoutAirConditioning = 37 ∧ 
  maxCarsWithAirConditioningWithoutStripes = 59

-- The statement to be proved
theorem minimum_racing_stripes (h : conditions totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes) :
   exists (R : ℕ ), R = 4 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end minimum_racing_stripes_l216_216968


namespace motion_of_Q_is_clockwise_with_2ω_l216_216467

variables {ω t : ℝ} {P Q : ℝ × ℝ}

def moving_counterclockwise (P : ℝ × ℝ) (ω t : ℝ) : Prop :=
  P = (Real.cos (ω * t), Real.sin (ω * t))

def motion_of_Q (x y : ℝ): ℝ × ℝ :=
  (-2 * x * y, y^2 - x^2)

def is_on_unit_circle (Q : ℝ × ℝ) : Prop :=
  Q.fst ^ 2 + Q.snd ^ 2 = 1

theorem motion_of_Q_is_clockwise_with_2ω 
  (P : ℝ × ℝ) (ω t : ℝ) (x y : ℝ) :
  moving_counterclockwise P ω t →
  P = (x, y) →
  is_on_unit_circle P →
  is_on_unit_circle (motion_of_Q x y) ∧
  Q = (x, y) →
  Q.fst = Real.cos (2 * ω * t + 3 * Real.pi / 2) ∧ 
  Q.snd = Real.sin (2 * ω * t + 3 * Real.pi / 2) :=
sorry

end motion_of_Q_is_clockwise_with_2ω_l216_216467


namespace sum_of_lengths_of_legs_of_larger_triangle_l216_216136

theorem sum_of_lengths_of_legs_of_larger_triangle
  (area_small : ℝ) (area_large : ℝ) (hypo_small : ℝ)
  (h_area_small : area_small = 18) (h_area_large : area_large = 288) (h_hypo_small : hypo_small = 10) :
  ∃ (sum_legs_large : ℝ), sum_legs_large = 52 :=
by
  sorry

end sum_of_lengths_of_legs_of_larger_triangle_l216_216136


namespace sequence_general_term_l216_216702

/-- The general term formula for the sequence 0.3, 0.33, 0.333, 0.3333, … is (1 / 3) * (1 - 1 / 10 ^ n). -/
theorem sequence_general_term (n : ℕ) : 
  (∃ a : ℕ → ℚ, (∀ n, a n = 0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1))) ↔
  ∀ n, (0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1)) = (1 / 3) * (1 - 1 / 10 ^ n) :=
sorry

end sequence_general_term_l216_216702


namespace derivative_of_y_l216_216233

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (2 * x)) ^ ((log (cos (2 * x))) / 4)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -((cos (2 * x)) ^ ((log (cos (2 * x))) / 4)) * (tan (2 * x)) * (log (cos (2 * x))) := by
    sorry

end derivative_of_y_l216_216233


namespace polynomial_root_sum_l216_216020

theorem polynomial_root_sum :
  ∃ a b c : ℝ,
    (∀ x : ℝ, Polynomial.eval x (Polynomial.X ^ 3 - 10 * Polynomial.X ^ 2 + 16 * Polynomial.X - 2) = 0) →
    a + b + c = 10 → ab + ac + bc = 16 → abc = 2 →
    (a / (bc + 2) + b / (ac + 2) + c / (ab + 2) = 4) := sorry

end polynomial_root_sum_l216_216020


namespace problem_1_solution_problem_2_solution_l216_216999

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 3) - abs (x - a)

-- Proof problem for question 1
theorem problem_1_solution (x : ℝ) : f x 2 ≤ -1/2 ↔ x ≥ 11/4 :=
by
  sorry

-- Proof problem for question 2
theorem problem_2_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ a) ↔ a ∈ Set.Iic (3/2) :=
by
  sorry

end problem_1_solution_problem_2_solution_l216_216999


namespace maximum_m_value_l216_216810

theorem maximum_m_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ m, m = 4 ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (1 / a + 1 / b) ≥ m) :=
sorry

end maximum_m_value_l216_216810


namespace tan_value_l216_216428

open Real

theorem tan_value (α : ℝ) (h : sin (5 * π / 6 - α) = sqrt 3 * cos (α + π / 6)) : 
  tan (α + π / 6) = sqrt 3 := 
  sorry

end tan_value_l216_216428


namespace calc_sub_neg_eq_add_problem_0_sub_neg_3_l216_216410

theorem calc_sub_neg_eq_add (a b : Int) : a - (-b) = a + b := by
  sorry

theorem problem_0_sub_neg_3 : 0 - (-3) = 3 := by
  exact calc_sub_neg_eq_add 0 3

end calc_sub_neg_eq_add_problem_0_sub_neg_3_l216_216410


namespace eval_expression_l216_216806

def base8_to_base10 (n : Nat) : Nat :=
  2 * 8^2 + 4 * 8^1 + 5 * 8^0

def base4_to_base10 (n : Nat) : Nat :=
  1 * 4^1 + 5 * 4^0

def base5_to_base10 (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 2 * 5^0

def base6_to_base10 (n : Nat) : Nat :=
  3 * 6^1 + 2 * 6^0

theorem eval_expression : 
  base8_to_base10 245 / base4_to_base10 15 - base5_to_base10 232 / base6_to_base10 32 = 15 :=
by sorry

end eval_expression_l216_216806


namespace prism_faces_l216_216736

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end prism_faces_l216_216736


namespace frustum_radius_l216_216063

theorem frustum_radius (C1 C2 l: ℝ) (S_lateral: ℝ) (r: ℝ) :
  (C1 = 2 * r * π) ∧ (C2 = 6 * r * π) ∧ (l = 3) ∧ (S_lateral = 84 * π) → (r = 7) :=
by
  sorry

end frustum_radius_l216_216063


namespace train_cross_time_l216_216888

-- Define the given conditions
def train_length : ℕ := 100
def train_speed_kmph : ℕ := 45
def total_length : ℕ := 275
def seconds_in_hour : ℕ := 3600
def meters_in_km : ℕ := 1000

-- Convert the speed from km/hr to m/s
noncomputable def train_speed_mps : ℚ := (train_speed_kmph * meters_in_km) / seconds_in_hour

-- The time to cross the bridge
noncomputable def time_to_cross (train_length total_length : ℕ) (train_speed_mps : ℚ) : ℚ :=
  total_length / train_speed_mps

-- The statement we want to prove
theorem train_cross_time : time_to_cross train_length total_length train_speed_mps = 30 :=
by
  sorry

end train_cross_time_l216_216888


namespace fraction_white_surface_area_l216_216303

-- Definitions for conditions
def larger_cube_side : ℕ := 3
def smaller_cube_count : ℕ := 27
def white_cube_count : ℕ := 19
def black_cube_count : ℕ := 8
def black_corners : Nat := 8
def faces_per_cube : ℕ := 6
def exposed_faces_per_corner : ℕ := 3

-- Theorem statement for proving the fraction of the white surface area
theorem fraction_white_surface_area : (30 : ℚ) / 54 = 5 / 9 :=
by 
  -- Add the proof steps here if necessary
  sorry

end fraction_white_surface_area_l216_216303


namespace train_travel_time_change_l216_216621

theorem train_travel_time_change 
  (t1 t2 : ℕ) (s1 s2 d : ℕ) 
  (h1 : t1 = 4) 
  (h2 : s1 = 50) 
  (h3 : s2 = 100) 
  (h4 : d = t1 * s1) :
  t2 = d / s2 → t2 = 2 :=
by
  intros
  sorry

end train_travel_time_change_l216_216621


namespace simplified_fraction_sum_l216_216759

theorem simplified_fraction_sum (n d : ℕ) (h_n : n = 144) (h_d : d = 256) : (9 + 16 = 25) := by
  have h1 : n = 2^4 * 3^2 := by sorry
  have h2 : d = 2^8 := by sorry
  have h3 : (n / gcd n d) = 9 := by sorry
  have h4 : (d / gcd n d) = 16 := by sorry
  exact rfl

end simplified_fraction_sum_l216_216759


namespace sum_of_coordinates_l216_216418

theorem sum_of_coordinates :
  let in_distance_from_line (p : (ℝ × ℝ)) (d : ℝ) (line_y : ℝ) : Prop := abs (p.2 - line_y) = d
  let in_distance_from_point (p1 p2 : (ℝ × ℝ)) (d : ℝ) : Prop := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = d^2
  ∃ (P1 P2 P3 P4 : ℝ × ℝ),
  in_distance_from_line P1 4 13 ∧ in_distance_from_point P1 (7, 13) 10 ∧
  in_distance_from_line P2 4 13 ∧ in_distance_from_point P2 (7, 13) 10 ∧
  in_distance_from_line P3 4 13 ∧ in_distance_from_point P3 (7, 13) 10 ∧
  in_distance_from_line P4 4 13 ∧ in_distance_from_point P4 (7, 13) 10 ∧
  (P1.1 + P2.1 + P3.1 + P4.1) + (P1.2 + P2.2 + P3.2 + P4.2) = 80 :=
sorry

end sum_of_coordinates_l216_216418


namespace largest_possible_b_l216_216594

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l216_216594


namespace some_number_is_five_l216_216196

theorem some_number_is_five (x : ℕ) (some_number : ℕ) (h1 : x = 5) (h2 : x / some_number + 3 = 4) : some_number = 5 := by
  sorry

end some_number_is_five_l216_216196


namespace pentagon_vertex_assignment_l216_216071

theorem pentagon_vertex_assignment :
  ∃ (x_A x_B x_C x_D x_E : ℝ),
    x_A + x_B = 1 ∧
    x_B + x_C = 2 ∧
    x_C + x_D = 3 ∧
    x_D + x_E = 4 ∧
    x_E + x_A = 5 ∧
    (x_A, x_B, x_C, x_D, x_E) = (1.5, -0.5, 2.5, 0.5, 3.5) := by
  sorry

end pentagon_vertex_assignment_l216_216071


namespace cos_alpha_minus_beta_l216_216706

theorem cos_alpha_minus_beta : 
  ∀ (α β : ℝ), 
  2 * Real.cos α - Real.cos β = 3 / 2 →
  2 * Real.sin α - Real.sin β = 2 →
  Real.cos (α - β) = -5 / 16 :=
by
  intros α β h1 h2
  sorry

end cos_alpha_minus_beta_l216_216706


namespace Petya_bonus_points_l216_216726

def bonus_points (p : ℕ) : ℕ :=
  if p < 1000 then
    (20 * p) / 100
  else if p ≤ 2000 then
    200 + (30 * (p - 1000)) / 100
  else
    200 + 300 + (50 * (p - 2000)) / 100

theorem Petya_bonus_points : bonus_points 2370 = 685 :=
by sorry

end Petya_bonus_points_l216_216726


namespace probability_left_oar_works_l216_216464

structure Oars where
  P_L : ℝ -- Probability that the left oar works
  P_R : ℝ -- Probability that the right oar works
  
def independent_prob (o : Oars) : Prop :=
  o.P_L = o.P_R ∧ (1 - o.P_L) * (1 - o.P_R) = 0.16

theorem probability_left_oar_works (o : Oars) (h1 : independent_prob o) (h2 : 1 - (1 - o.P_L) * (1 - o.P_R) = 0.84) : o.P_L = 0.6 :=
by
  sorry

end probability_left_oar_works_l216_216464


namespace value_of_b_conditioned_l216_216355

theorem value_of_b_conditioned
  (b: ℝ) 
  (h0 : 0 < b ∧ b < 7)
  (h1 : (1 / 2) * (8 - b) * (b - 8) / ((1 / 2) * (b / 2) * b) = 4 / 9):
  b = 4 := 
sorry

end value_of_b_conditioned_l216_216355


namespace inequality_solution_l216_216129

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := 3/4 ≤ x ∧ x ≤ 2

-- Theorem statement to prove the equivalence
theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ solution_set x := by
  sorry

end inequality_solution_l216_216129


namespace cost_of_each_hotdog_l216_216504

theorem cost_of_each_hotdog (number_of_hotdogs : ℕ) (total_cost : ℕ) (cost_per_hotdog : ℕ) 
    (h1 : number_of_hotdogs = 6) (h2 : total_cost = 300) : cost_per_hotdog = 50 :=
by
  have h3 : cost_per_hotdog = total_cost / number_of_hotdogs :=
    sorry -- here we would normally write the division step
  sorry -- here we would show that h3 implies cost_per_hotdog = 50, given h1 and h2

end cost_of_each_hotdog_l216_216504


namespace probability_of_red_ball_l216_216334

-- Define the conditions
def num_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

-- Calculate the probability
def probability_drawing_red_ball : ℚ := red_balls / num_balls

-- The theorem statement to be proven
theorem probability_of_red_ball : probability_drawing_red_ball = 2 / 3 :=
by
  sorry

end probability_of_red_ball_l216_216334


namespace maximal_p_sum_consecutive_l216_216291

theorem maximal_p_sum_consecutive (k : ℕ) (h1 : k = 31250) : 
  ∃ p a : ℕ, p * (2 * a + p - 1) = k ∧ ∀ p' a', (p' * (2 * a' + p' - 1) = k) → p' ≤ p := by
  sorry

end maximal_p_sum_consecutive_l216_216291


namespace wire_length_between_poles_l216_216514

theorem wire_length_between_poles :
  let d := 18  -- distance between the bottoms of the poles
  let h1 := 6 + 3  -- effective height of the shorter pole
  let h2 := 20  -- height of the taller pole
  let vertical_distance := h2 - h1 -- vertical distance between the tops of the poles
  let hypotenuse := Real.sqrt (d^2 + vertical_distance^2)
  hypotenuse = Real.sqrt 445 :=
by
  sorry

end wire_length_between_poles_l216_216514


namespace LeRoy_should_pay_Bernardo_l216_216980

theorem LeRoy_should_pay_Bernardo 
    (initial_loan : ℕ := 100)
    (LeRoy_gas_expense : ℕ := 300)
    (LeRoy_food_expense : ℕ := 200)
    (Bernardo_accommodation_expense : ℕ := 500)
    (total_expense := LeRoy_gas_expense + LeRoy_food_expense + Bernardo_accommodation_expense)
    (shared_expense := total_expense / 2)
    (LeRoy_total_responsibility := shared_expense + initial_loan)
    (LeRoy_needs_to_pay := LeRoy_total_responsibility - (LeRoy_gas_expense + LeRoy_food_expense)) :
    LeRoy_needs_to_pay = 100 := 
by
    sorry

end LeRoy_should_pay_Bernardo_l216_216980


namespace find_m_l216_216147

theorem find_m (m : ℕ) (h1 : List ℕ := [27, 32, 39, m, 46, 47])
            (h2 : List ℕ := [30, 31, 34, 41, 42, 45])
            (h3 : (39 + m) / 2 = 42) :
            m = 45 :=
by {
  sorry
}

end find_m_l216_216147


namespace hyperbola_asymptotes_m_value_l216_216997

theorem hyperbola_asymptotes_m_value : 
    (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 1) → (y = (3/4) * x ∨ y = -(3/4) * x)) := 
by sorry

end hyperbola_asymptotes_m_value_l216_216997


namespace profit_per_cake_l216_216312

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l216_216312


namespace length_of_each_piece_cm_l216_216737

theorem length_of_each_piece_cm 
  (total_length : ℝ) 
  (number_of_pieces : ℕ) 
  (htotal : total_length = 17) 
  (hpieces : number_of_pieces = 20) : 
  (total_length / number_of_pieces) * 100 = 85 := 
by
  sorry

end length_of_each_piece_cm_l216_216737


namespace max_distance_circle_to_line_l216_216348

open Real

theorem max_distance_circle_to_line :
  let circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y + 1 = 0
  let line_eq (x y : ℝ) := x - y = 2
  ∃ (M : ℝ), (∀ x y, circle_eq x y → ∀ (d : ℝ), (line_eq x y → M ≤ d)) ∧ M = sqrt 2 + 1 :=
by
  sorry

end max_distance_circle_to_line_l216_216348


namespace range_of_a_l216_216821

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by
  sorry

end range_of_a_l216_216821


namespace rowing_time_to_place_and_back_l216_216608

open Real

/-- Definitions of the problem conditions -/
def rowing_speed_still_water : ℝ := 5
def current_speed : ℝ := 1
def distance_to_place : ℝ := 2.4

/-- Proof statement: the total time taken to row to the place and back is 1 hour -/
theorem rowing_time_to_place_and_back :
  (distance_to_place / (rowing_speed_still_water + current_speed)) + 
  (distance_to_place / (rowing_speed_still_water - current_speed)) =
  1 := by
  sorry

end rowing_time_to_place_and_back_l216_216608


namespace verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l216_216839

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Definition for conversions from base 60 to base 10
def from_base_60 (d1 d0 : ℕ) : ℕ :=
  d1 * 60 + d0

-- Proof statements
theorem verify_21_base_60 : from_base_60 2 1 = 121 ∧ is_perfect_square 121 :=
by {
  sorry
}

theorem verify_1_base_60 : from_base_60 0 1 = 1 ∧ is_perfect_square 1 :=
by {
  sorry
}

theorem verify_2_base_60_not_square : from_base_60 0 2 = 2 ∧ ¬ is_perfect_square 2 :=
by {
  sorry
}

end verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l216_216839


namespace find_number_l216_216564

theorem find_number (x : ℝ) : (35 - x) * 2 + 12 = 72 → ((35 - x) * 2 + 12) / 8 = 9 → x = 5 :=
by
  -- assume the first condition
  intro h1
  -- assume the second condition
  intro h2
  -- the proof goes here
  sorry

end find_number_l216_216564


namespace investment_calculation_l216_216969

noncomputable def initial_investment (final_amount : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  final_amount / ((1 + interest_rate / 100) ^ years)

theorem investment_calculation :
  initial_investment 504.32 3 12 = 359 :=
by
  sorry

end investment_calculation_l216_216969


namespace pascal_sixth_element_row_20_l216_216559

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end pascal_sixth_element_row_20_l216_216559


namespace evaluate_expression_l216_216930

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 := by
  sorry

end evaluate_expression_l216_216930


namespace candy_per_packet_l216_216687

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l216_216687


namespace find_x_l216_216505

theorem find_x (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 := 
by 
  sorry

end find_x_l216_216505


namespace ratio_of_c_to_b_l216_216188

    theorem ratio_of_c_to_b (a b c : ℤ) (h0 : a = 0) (h1 : a < b) (h2 : b < c)
      (h3 : (a + b + c) / 3 = b / 2) : c / b = 1 / 2 :=
    by
      -- proof steps go here
      sorry
    
end ratio_of_c_to_b_l216_216188


namespace ram_gohul_work_days_l216_216918

theorem ram_gohul_work_days (ram_days gohul_days : ℕ) (H_ram: ram_days = 10) (H_gohul: gohul_days = 15): 
  (ram_days * gohul_days) / (ram_days + gohul_days) = 6 := 
by
  sorry

end ram_gohul_work_days_l216_216918


namespace sum_YNRB_l216_216804

theorem sum_YNRB :
  ∃ (R Y B N : ℕ),
    (RY = 10 * R + Y) ∧
    (BY = 10 * B + Y) ∧
    (111 * N = (10 * R + Y) * (10 * B + Y)) →
    (Y + N + R + B = 21) :=
sorry

end sum_YNRB_l216_216804


namespace find_some_value_l216_216797

theorem find_some_value (m n k : ℝ)
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + k = (n + 18) / 6 - 2 / 5) : 
  k = 3 :=
sorry

end find_some_value_l216_216797


namespace percent_of_employed_people_who_are_females_l216_216917

theorem percent_of_employed_people_who_are_females (p_employed p_employed_males : ℝ) 
  (h1 : p_employed = 64) (h2 : p_employed_males = 48) : 
  100 * (p_employed - p_employed_males) / p_employed = 25 :=
by
  sorry

end percent_of_employed_people_who_are_females_l216_216917


namespace average_weight_l216_216320

theorem average_weight (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 47) (h3 : B = 39) : (A + B + C) / 3 = 45 := 
  sorry

end average_weight_l216_216320


namespace treasure_in_heaviest_bag_l216_216413

theorem treasure_in_heaviest_bag (A B C D : ℝ) (h1 : A + B < C)
                                        (h2 : A + C = D)
                                        (h3 : A + D > B + C) : D > A ∧ D > B ∧ D > C :=
by 
  sorry

end treasure_in_heaviest_bag_l216_216413


namespace main_theorem_l216_216749

-- Declare nonzero complex numbers
variables {x y z : ℂ} 

-- State the conditions
def conditions (x y z : ℂ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + y + z = 30 ∧
  (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z

-- Prove the main statement given the conditions
theorem main_theorem (h : conditions x y z) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 33 :=
by
  sorry

end main_theorem_l216_216749


namespace solve_quadratic_inequality_l216_216637

theorem solve_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end solve_quadratic_inequality_l216_216637


namespace minimize_area_eq_l216_216588

theorem minimize_area_eq {l : ℝ → ℝ → Prop}
  (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (condition1 : l P.1 P.2)
  (condition2 : A.1 > 0 ∧ A.2 = 0)
  (condition3 : B.1 = 0 ∧ B.2 > 0)
  (line_eq : ∀ x y : ℝ, l x y ↔ (2 * x + y = 4)) :
  ∀ (a b : ℝ), a = 2 → b = 4 → 2 * P.1 + P.2 = 4 :=
by sorry

end minimize_area_eq_l216_216588


namespace bahs_for_1000_yahs_l216_216831

-- Definitions based on given conditions
def bahs_to_rahs_ratio (b r : ℕ) := 15 * b = 24 * r
def rahs_to_yahs_ratio (r y : ℕ) := 9 * r = 15 * y

-- Main statement to prove
theorem bahs_for_1000_yahs (b r y : ℕ) (h1 : bahs_to_rahs_ratio b r) (h2 : rahs_to_yahs_ratio r y) :
  1000 * y = 375 * b :=
by
  sorry

end bahs_for_1000_yahs_l216_216831


namespace five_digit_sine_rule_count_l216_216185

theorem five_digit_sine_rule_count :
    ∃ (count : ℕ), 
        (∀ (a b c d e : ℕ), 
          (a <  b) ∧
          (b >  c) ∧
          (c >  d) ∧
          (d <  e) ∧
          (a >  d) ∧
          (b >  e) ∧
          (∃ (num : ℕ), num = 10000 * a + 1000 * b + 100 * c + 10 * d + e))
        →
        count = 2892 :=
sorry

end five_digit_sine_rule_count_l216_216185


namespace final_sum_l216_216033

def Q (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 4

noncomputable def probability_condition_holds : ℝ :=
  by sorry

theorem final_sum :
  let m := 1
  let n := 1
  let o := 1
  let p := 0
  let q := 8
  (m + n + o + p + q) = 11 :=
  by
    sorry

end final_sum_l216_216033


namespace calc_exp_l216_216771

open Real

theorem calc_exp (x y : ℝ) : 
  (-(1/3) * (x^2) * y) ^ 3 = -(x^6 * y^3) / 27 := 
  sorry

end calc_exp_l216_216771


namespace g_eq_g_inv_l216_216714

-- Define the function g
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := (5 + Real.sqrt (1 + 8 * y)) / 4 -- simplified to handle the principal value

theorem g_eq_g_inv (x : ℝ) : g x = g_inv x → x = 1 := by
  -- Placeholder for proof
  sorry

end g_eq_g_inv_l216_216714


namespace remainder_of_101_pow_37_mod_100_l216_216115

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l216_216115


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l216_216436

-- Define a predicate for consecutive prime numbers
def is_consecutive_primes (a b c d : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧
  (b = a + 1 ∨ b = a + 2) ∧
  (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2)

-- Define the main problem statement
theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (a b c d : ℕ), is_consecutive_primes a b c d ∧ (a + b + c + d) % 5 = 0 ∧ ∀ (w x y z : ℕ), is_consecutive_primes w x y z ∧ (w + x + y + z) % 5 = 0 → a + b + c + d ≤ w + x + y + z :=
sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l216_216436


namespace greatest_whole_number_inequality_l216_216347

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end greatest_whole_number_inequality_l216_216347


namespace age_difference_l216_216697

-- Definitions based on the problem statement
def son_present_age : ℕ := 33

-- Represent the problem in terms of Lean
theorem age_difference (M : ℕ) (h : M + 2 = 2 * (son_present_age + 2)) : M - son_present_age = 35 :=
by
  sorry

end age_difference_l216_216697


namespace costume_processing_time_l216_216343

theorem costume_processing_time (x : ℕ) : 
  (300 - 60) / (2 * x) + 60 / x = 9 → (60 / x) + (240 / (2 * x)) = 9 :=
by
  sorry

end costume_processing_time_l216_216343


namespace class_books_transfer_l216_216600

theorem class_books_transfer :
  ∀ (A B n : ℕ), 
    A = 200 → B = 200 → 
    (B + n = 3/2 * (A - n)) →
    n = 40 :=
by sorry

end class_books_transfer_l216_216600


namespace solve_system_eq_solve_system_ineq_l216_216989

-- For the system of equations:
theorem solve_system_eq (x y : ℝ) (h1 : x + 2 * y = 7) (h2 : 3 * x + y = 6) : x = 1 ∧ y = 3 :=
sorry

-- For the system of inequalities:
theorem solve_system_ineq (x : ℝ) (h1 : 2 * (x - 1) + 1 > -3) (h2 : x - 1 ≤ (1 + x) / 3) : -1 < x ∧ x ≤ 2 :=
sorry

end solve_system_eq_solve_system_ineq_l216_216989


namespace smallest_possible_value_abs_sum_l216_216556

theorem smallest_possible_value_abs_sum : 
  ∀ (x : ℝ), 
    (|x + 3| + |x + 6| + |x + 7| + 2) ≥ 8 :=
by
  sorry

end smallest_possible_value_abs_sum_l216_216556


namespace product_sum_abcd_e_l216_216026

-- Define the individual numbers
def a : ℕ := 12
def b : ℕ := 25
def c : ℕ := 52
def d : ℕ := 21
def e : ℕ := 32

-- Define the sum of the numbers a, b, c, and d
def sum_abcd : ℕ := a + b + c + d

-- Prove that multiplying the sum by e equals 3520
theorem product_sum_abcd_e : sum_abcd * e = 3520 := by
  sorry

end product_sum_abcd_e_l216_216026


namespace car_mileage_city_l216_216319

theorem car_mileage_city {h c t : ℝ} (H1: 448 = h * t) (H2: 336 = c * t) (H3: c = h - 6) : c = 18 :=
sorry

end car_mileage_city_l216_216319


namespace part1_solution_set_part2_values_of_a_l216_216250

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l216_216250


namespace inequality_solution_l216_216485

theorem inequality_solution :
  {x : ℝ | (3 * x - 9) * (x - 4) / (x - 1) ≥ 0} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 3} ∪ {x : ℝ | x ≥ 4} :=
by
  sorry

end inequality_solution_l216_216485


namespace triple_angle_l216_216465

theorem triple_angle (α : ℝ) : 3 * α = α + α + α := 
by sorry

end triple_angle_l216_216465


namespace find_x_l216_216553

theorem find_x (y x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 := 
sorry

end find_x_l216_216553


namespace unrealistic_data_l216_216779

theorem unrealistic_data :
  let A := 1000
  let A1 := 265
  let A2 := 51
  let A3 := 803
  let A1U2 := 287
  let A2U3 := 843
  let A1U3 := 919
  let A1I2 := A1 + A2 - A1U2
  let A2I3 := A2 + A3 - A2U3
  let A3I1 := A3 + A1 - A1U3
  let U := A1 + A2 + A3 - A1I2 - A2I3 - A3I1
  let A1I2I3 := A - U
  A1I2I3 > A2 :=
by
   sorry

end unrealistic_data_l216_216779


namespace f_f_is_even_l216_216161

-- Let f be a function from reals to reals
variables {f : ℝ → ℝ}

-- Given that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem to prove
theorem f_f_is_even (h : is_even f) : is_even (fun x => f (f x)) :=
by
  intros
  unfold is_even at *
  -- at this point, we assume the function f is even,
  -- follow from the assumption, we can prove the result
  sorry

end f_f_is_even_l216_216161


namespace total_candies_darrel_took_l216_216638

theorem total_candies_darrel_took (r b x : ℕ) (h1 : r = 3 * b)
  (h2 : r - x = 4 * (b - x))
  (h3 : r - x - 12 = 5 * (b - x - 12)) : 2 * x = 48 := sorry

end total_candies_darrel_took_l216_216638


namespace total_travel_time_is_19_hours_l216_216845

-- Define the distances and speeds as constants
def distance_WA_ID := 640
def speed_WA_ID := 80
def distance_ID_NV := 550
def speed_ID_NV := 50

-- Define the times based on the given distances and speeds
def time_WA_ID := distance_WA_ID / speed_WA_ID
def time_ID_NV := distance_ID_NV / speed_ID_NV

-- Define the total time
def total_time := time_WA_ID + time_ID_NV

-- Prove that the total travel time is 19 hours
theorem total_travel_time_is_19_hours : total_time = 19 := by
  sorry

end total_travel_time_is_19_hours_l216_216845


namespace restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l216_216889

-- Defining the given conditions
noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P
noncomputable def post_tax_producer_price : ℝ := 64
noncomputable def per_unit_tax : ℝ := 90
noncomputable def elasticity_supply_no_tax (P_e : ℝ) (Q_e : ℝ) : ℝ :=
  1.5 * (-(4 * P_e / Q_e))

-- Supply function to be proven
noncomputable def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Total tax revenue to be proven
noncomputable def total_tax_revenue : ℝ := 6480

-- Optimal tax rate to be proven
noncomputable def optimal_tax_rate : ℝ := 60

-- Maximum tax revenue to be proven
noncomputable def maximum_tax_revenue : ℝ := 8640

-- Theorem statements that need to be proven
theorem restore_supply_function (P : ℝ) : 
  supply_function P = 6 * P - 312 := sorry

theorem determine_tax_revenue : 
  total_tax_revenue = 6480 := sorry

theorem determine_optimal_tax_rate : 
  optimal_tax_rate = 60 := sorry

theorem determine_maximum_tax_revenue : 
  maximum_tax_revenue = 8640 := sorry

end restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l216_216889


namespace greatest_sum_of_visible_numbers_l216_216395

/-- Definition of a cube with numbered faces -/
structure Cube where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  face4 : ℕ
  face5 : ℕ
  face6 : ℕ

/-- The cubes face numbers -/
def cube_numbers : List ℕ := [1, 2, 4, 8, 16, 32]

/-- Stacked cubes with maximized visible numbers sum -/
def maximize_visible_sum :=
  let cube1 := Cube.mk 1 2 4 8 16 32
  let cube2 := Cube.mk 1 2 4 8 16 32
  let cube3 := Cube.mk 1 2 4 8 16 32
  let cube4 := Cube.mk 1 2 4 8 16 32
  244

theorem greatest_sum_of_visible_numbers : maximize_visible_sum = 244 := 
  by
    sorry -- Proof to be done

end greatest_sum_of_visible_numbers_l216_216395


namespace range_of_m_l216_216006

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 2 * m - 3 ≥ 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l216_216006


namespace expected_value_is_correct_l216_216506

noncomputable def expected_value_max_two_rolls : ℝ :=
  let p_max_1 := (1/6) * (1/6)
  let p_max_2 := (2/6) * (2/6) - (1/6) * (1/6)
  let p_max_3 := (3/6) * (3/6) - (2/6) * (2/6)
  let p_max_4 := (4/6) * (4/6) - (3/6) * (3/6)
  let p_max_5 := (5/6) * (5/6) - (4/6) * (4/6)
  let p_max_6 := 1 - (5/6) * (5/6)
  1 * p_max_1 + 2 * p_max_2 + 3 * p_max_3 + 4 * p_max_4 + 5 * p_max_5 + 6 * p_max_6

theorem expected_value_is_correct :
  expected_value_max_two_rolls = 4.5 :=
sorry

end expected_value_is_correct_l216_216506


namespace rationalize_denominator_l216_216992

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 15 = (7 / 3 : ℝ) * Real.sqrt 15 :=
by
  sorry

end rationalize_denominator_l216_216992


namespace greatest_perfect_power_sum_l216_216234

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l216_216234


namespace more_flour_than_sugar_l216_216970

variable (total_flour : ℕ) (total_sugar : ℕ)
variable (flour_added : ℕ)

def additional_flour_needed (total_flour flour_added : ℕ) : ℕ :=
  total_flour - flour_added

theorem more_flour_than_sugar :
  additional_flour_needed 10 7 - 2 = 1 :=
by
  sorry

end more_flour_than_sugar_l216_216970


namespace find_200_digit_number_l216_216366

noncomputable def original_number_condition (N : ℕ) (c : ℕ) (k : ℕ) : Prop :=
  let m := 0
  let a := 2 * c
  let b := 3 * c
  k = 197 ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ N = 132 * c * 10^197

theorem find_200_digit_number :
  ∃ N c, original_number_condition N c 197 :=
by
  sorry

end find_200_digit_number_l216_216366


namespace xiaoLiangComprehensiveScore_l216_216868

-- Define the scores for the three aspects
def contentScore : ℝ := 88
def deliveryAbilityScore : ℝ := 95
def effectivenessScore : ℝ := 90

-- Define the weights for the three aspects
def contentWeight : ℝ := 0.5
def deliveryAbilityWeight : ℝ := 0.4
def effectivenessWeight : ℝ := 0.1

-- Define the comprehensive score
def comprehensiveScore : ℝ :=
  (contentScore * contentWeight) +
  (deliveryAbilityScore * deliveryAbilityWeight) +
  (effectivenessScore * effectivenessWeight)

-- The theorem stating that the comprehensive score equals 91
theorem xiaoLiangComprehensiveScore : comprehensiveScore = 91 := by
  -- proof here (omitted)
  sorry

end xiaoLiangComprehensiveScore_l216_216868


namespace a_4_value_l216_216933

-- Defining the polynomial (2x - 3)^6
def polynomial_expansion (x : ℝ) := (2 * x - 3) ^ 6

-- Given conditions polynomial expansion in terms of (x - 1)
def polynomial_coefficients (x : ℝ) (a : Fin 7 → ℝ) : ℝ :=
  a 0 + a 1 * (x - 1) + a 2 * (x - 1) ^ 2 + a 3 * (x - 1) ^ 3 + a 4 * (x - 1) ^ 4 +
  a 5 * (x - 1) ^ 5 + a 6 * (x - 1) ^ 6

-- The proof problem asking to show a_4 = 240
theorem a_4_value : 
  ∀ a : Fin 7 → ℝ, (∀ x : ℝ, polynomial_expansion x = polynomial_coefficients x a) → a 4 = 240 := by 
  sorry

end a_4_value_l216_216933


namespace visitors_yesterday_l216_216302

-- Definitions based on the given conditions
def visitors_today : ℕ := 583
def visitors_total : ℕ := 829

-- Theorem statement to prove the number of visitors the day before Rachel visited
theorem visitors_yesterday : ∃ v_yesterday: ℕ, v_yesterday = visitors_total - visitors_today ∧ v_yesterday = 246 :=
by
  sorry

end visitors_yesterday_l216_216302


namespace speed_of_stream_l216_216171

theorem speed_of_stream (v : ℝ) : (13 + v) * 4 = 68 → v = 4 :=
by
  intro h
  sorry

end speed_of_stream_l216_216171


namespace ratio_of_Steve_speeds_l216_216452

noncomputable def Steve_speeds_ratio : Nat := 
  let d := 40 -- distance in km
  let T := 6  -- total time in hours
  let v2 := 20 -- speed on the way back in km/h
  let t2 := d / v2 -- time taken on the way back in hours
  let t1 := T - t2 -- time taken on the way to work in hours
  let v1 := d / t1 -- speed on the way to work in km/h
  v2 / v1

theorem ratio_of_Steve_speeds :
  Steve_speeds_ratio = 2 := 
  by sorry

end ratio_of_Steve_speeds_l216_216452


namespace sum_of_dimensions_l216_216602

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 50) (h2 : A * C = 90) (h3 : B * C = 100) : A + B + C = 24 :=
  sorry

end sum_of_dimensions_l216_216602


namespace arithmetic_seq_necessary_not_sufficient_l216_216646

noncomputable def arithmetic_sequence (a b c : ℝ) : Prop :=
  a + c = 2 * b

noncomputable def proposition_B (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ (a / b) + (c / b) = 2

theorem arithmetic_seq_necessary_not_sufficient (a b c : ℝ) :
  (arithmetic_sequence a b c → proposition_B a b c) ∧ 
  (∃ a' b' c', arithmetic_sequence a' b' c' ∧ ¬ proposition_B a' b' c') := by
  sorry

end arithmetic_seq_necessary_not_sufficient_l216_216646


namespace expected_non_empty_urns_correct_l216_216573

open ProbabilityTheory

noncomputable def expected_non_empty_urns (n k : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n) ^ k)

theorem expected_non_empty_urns_correct (n k : ℕ) : expected_non_empty_urns n k = n * (1 - ((n - 1) / n) ^ k) :=
by 
  sorry

end expected_non_empty_urns_correct_l216_216573


namespace total_cost_for_round_trip_l216_216507

def time_to_cross_one_way : ℕ := 4 -- time in hours to cross the lake one way
def cost_per_hour : ℕ := 10 -- cost in dollars per hour

def total_time := time_to_cross_one_way * 2 -- total time in hours for a round trip
def total_cost := total_time * cost_per_hour -- total cost in dollars for the assistant

theorem total_cost_for_round_trip : total_cost = 80 := by
  repeat {sorry} -- Leaving the proof for now

end total_cost_for_round_trip_l216_216507


namespace evaluate_expression_l216_216778

theorem evaluate_expression : (64^(1 / 6) * 16^(1 / 4) * 8^(1 / 3) = 8) :=
by
  -- sorry added to skip the proof
  sorry

end evaluate_expression_l216_216778


namespace compute_exponent_multiplication_l216_216323

theorem compute_exponent_multiplication : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_exponent_multiplication_l216_216323


namespace puppies_adopted_each_day_l216_216084

variable (initial_puppies additional_puppies days total_puppies puppies_per_day : ℕ)

axiom initial_puppies_ax : initial_puppies = 9
axiom additional_puppies_ax : additional_puppies = 12
axiom days_ax : days = 7
axiom total_puppies_ax : total_puppies = initial_puppies + additional_puppies
axiom adoption_rate_ax : total_puppies / days = puppies_per_day

theorem puppies_adopted_each_day : 
  initial_puppies = 9 → additional_puppies = 12 → days = 7 → total_puppies = initial_puppies + additional_puppies → total_puppies / days = puppies_per_day → puppies_per_day = 3 :=
by
  intro initial_puppies_ax additional_puppies_ax days_ax total_puppies_ax adoption_rate_ax
  sorry

end puppies_adopted_each_day_l216_216084


namespace deepak_age_l216_216601

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 5 / 7)
  (h2 : A + 6 = 36) :
  D = 42 :=
by sorry

end deepak_age_l216_216601


namespace sequence_increasing_l216_216000

theorem sequence_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n + 3) : ∀ n, a (n + 1) > a n := 
by 
  sorry

end sequence_increasing_l216_216000


namespace intersection_is_3_l216_216614

open Set -- Open the Set namespace to use set notation

theorem intersection_is_3 {A B : Set ℤ} (hA : A = {1, 3}) (hB : B = {-1, 2, 3}) :
  A ∩ B = {3} :=
by {
-- Proof goes here
  sorry
}

end intersection_is_3_l216_216614


namespace wendy_adds_18_gallons_l216_216766

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end wendy_adds_18_gallons_l216_216766


namespace yellow_marbles_count_l216_216653

-- Definitions based on given conditions
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def black_marbles : ℕ := 1
def probability_black : ℚ := 1 / 28
def total_marbles : ℕ := 28

-- Problem statement to prove
theorem yellow_marbles_count :
  (total_marbles = blue_marbles + green_marbles + black_marbles + n) →
  (probability_black = black_marbles / total_marbles) →
  n = 12 :=
by
  intros; sorry

end yellow_marbles_count_l216_216653


namespace planes_parallel_if_any_line_parallel_l216_216239

axiom Plane : Type
axiom Line : Type
axiom contains : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Plane → Prop

theorem planes_parallel_if_any_line_parallel (α β : Plane)
  (h₁ : ∀ l, contains α l → parallel_lines l β) :
  parallel α β :=
sorry

end planes_parallel_if_any_line_parallel_l216_216239


namespace negation_of_p_l216_216288

-- Declare the proposition p as a condition
def p : Prop :=
  ∀ (x : ℝ), 0 ≤ x → x^2 + 4 * x + 3 > 0

-- State the problem
theorem negation_of_p : ¬ p ↔ ∃ (x : ℝ), 0 ≤ x ∧ x^2 + 4 * x + 3 ≤ 0 :=
by
  sorry

end negation_of_p_l216_216288


namespace sum_of_squares_l216_216248

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 :=
by
  sorry

end sum_of_squares_l216_216248


namespace pencils_total_l216_216744

-- Defining the conditions
def packs_to_pencils (packs : ℕ) : ℕ := packs * 12

def jimin_packs : ℕ := 2
def jimin_individual_pencils : ℕ := 7

def yuna_packs : ℕ := 1
def yuna_individual_pencils : ℕ := 9

-- Translating to Lean 4 statement
theorem pencils_total : 
  packs_to_pencils jimin_packs + jimin_individual_pencils + packs_to_pencils yuna_packs + yuna_individual_pencils = 52 := 
by
  sorry

end pencils_total_l216_216744


namespace train_stop_time_l216_216297

theorem train_stop_time (speed_no_stops speed_with_stops : ℕ) (time_per_hour : ℕ) (stoppage_time_per_hour : ℕ) :
  speed_no_stops = 45 →
  speed_with_stops = 30 →
  time_per_hour = 60 →
  stoppage_time_per_hour = 20 :=
by
  intros h1 h2 h3
  sorry

end train_stop_time_l216_216297


namespace martha_correct_guess_probability_l216_216150

namespace MarthaGuess

-- Definitions for the conditions
def height_guess_child_accurate : ℚ := 4 / 5
def height_guess_adult_accurate : ℚ := 5 / 6
def weight_guess_tight_clothing_accurate : ℚ := 3 / 4
def weight_guess_loose_clothing_accurate : ℚ := 7 / 10

-- Probabilities of incorrect guesses
def height_guess_child_inaccurate : ℚ := 1 - height_guess_child_accurate
def height_guess_adult_inaccurate : ℚ := 1 - height_guess_adult_accurate
def weight_guess_tight_clothing_inaccurate : ℚ := 1 - weight_guess_tight_clothing_accurate
def weight_guess_loose_clothing_inaccurate : ℚ := 1 - weight_guess_loose_clothing_accurate

-- Combined probability of guessing incorrectly for each case
def incorrect_prob_child_loose : ℚ := height_guess_child_inaccurate * weight_guess_loose_clothing_inaccurate
def incorrect_prob_adult_tight : ℚ := height_guess_adult_inaccurate * weight_guess_tight_clothing_inaccurate
def incorrect_prob_adult_loose : ℚ := height_guess_adult_inaccurate * weight_guess_loose_clothing_inaccurate

-- Total probability of incorrect guesses for all three cases
def total_incorrect_prob : ℚ := incorrect_prob_child_loose * incorrect_prob_adult_tight * incorrect_prob_adult_loose

-- Probability of at least one correct guess
def correct_prob_at_least_once : ℚ := 1 - total_incorrect_prob

-- Main theorem stating the final result
theorem martha_correct_guess_probability : correct_prob_at_least_once = 7999 / 8000 := by
  sorry

end MarthaGuess

end martha_correct_guess_probability_l216_216150


namespace square_octagon_can_cover_ground_l216_216027

def square_interior_angle := 90
def octagon_interior_angle := 135

theorem square_octagon_can_cover_ground :
  square_interior_angle + 2 * octagon_interior_angle = 360 :=
by
  -- Proof skipped with sorry
  sorry

end square_octagon_can_cover_ground_l216_216027


namespace solve_system_addition_l216_216572

theorem solve_system_addition (a b : ℝ) (h1 : 3 * a + 7 * b = 1977) (h2 : 5 * a + b = 2007) : a + b = 498 :=
by
  sorry

end solve_system_addition_l216_216572


namespace problem_statement_l216_216406

noncomputable def middle_of_three_consecutive (x : ℕ) : ℕ :=
  let y := x + 1
  let z := x + 2
  y

theorem problem_statement :
  ∃ x : ℕ, 
    (x + (x + 1) = 18) ∧ 
    (x + (x + 2) = 20) ∧ 
    ((x + 1) + (x + 2) = 23) ∧ 
    (middle_of_three_consecutive x = 7) :=
by
  sorry

end problem_statement_l216_216406


namespace triangle_angle_120_l216_216841

theorem triangle_angle_120 (a b c : ℝ) (B : ℝ) (hB : B = 120) :
  a^2 + a * c + c^2 - b^2 = 0 := by
sorry

end triangle_angle_120_l216_216841


namespace diamond_op_example_l216_216528

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y

theorem diamond_op_example : diamond_op 2 7 = 41 :=
by {
    -- proof goes here
    sorry
}

end diamond_op_example_l216_216528


namespace balls_in_base_l216_216073

theorem balls_in_base (n k : ℕ) (h1 : 165 = (n * (n + 1) * (n + 2)) / 6) (h2 : k = n * (n + 1) / 2) : k = 45 := 
by 
  sorry

end balls_in_base_l216_216073


namespace problem_l216_216200

theorem problem (p q r : ℝ) (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) : r = 2000 :=
by
  sorry

end problem_l216_216200


namespace div_by_90_l216_216832

def N : ℤ := 19^92 - 91^29

theorem div_by_90 : ∃ k : ℤ, N = 90 * k := 
sorry

end div_by_90_l216_216832


namespace average_weight_of_whole_class_l216_216079

def num_students_a : ℕ := 50
def num_students_b : ℕ := 70
def avg_weight_a : ℚ := 50
def avg_weight_b : ℚ := 70

theorem average_weight_of_whole_class :
  (num_students_a * avg_weight_a + num_students_b * avg_weight_b) / (num_students_a + num_students_b) = 61.67 := by
  sorry

end average_weight_of_whole_class_l216_216079


namespace houses_with_two_car_garage_l216_216370

theorem houses_with_two_car_garage
  (T P GP N G : ℕ)
  (hT : T = 90)
  (hP : P = 40)
  (hGP : GP = 35)
  (hN : N = 35)
  (hFormula : G + P - GP = T - N) :
  G = 50 :=
by
  rw [hT, hP, hGP, hN] at hFormula
  simp at hFormula
  exact hFormula

end houses_with_two_car_garage_l216_216370


namespace neither_sufficient_nor_necessary_l216_216620

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0 → ab > 0) ∧ (ab > 0 → a + b > 0)) :=
by {
  sorry
}

end neither_sufficient_nor_necessary_l216_216620


namespace triangle_side_value_l216_216497

theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 1)
  (h2 : b = 4)
  (h3 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h4 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = Real.sqrt 13 :=
sorry

end triangle_side_value_l216_216497


namespace solve_for_s_l216_216154

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * 3^s) 
  (h2 : 45 = m * 9^s) : 
  s = 2 :=
sorry

end solve_for_s_l216_216154


namespace addition_value_l216_216809

def certain_number : ℝ := 5.46 - 3.97

theorem addition_value : 5.46 + certain_number = 6.95 := 
  by 
    -- The proof would go here, but is replaced with sorry.
    sorry

end addition_value_l216_216809


namespace correlation_graph_is_scatter_plot_l216_216158

/-- The definition of a scatter plot graph -/
def scatter_plot_graph (x y : ℝ → ℝ) : Prop := 
  ∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)

/-- Prove that the graph representing a set of data for two variables with a correlation is called a "scatter plot" -/
theorem correlation_graph_is_scatter_plot (x y : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)) → 
  (scatter_plot_graph x y) :=
by
  sorry

end correlation_graph_is_scatter_plot_l216_216158


namespace part1_part2_part3_l216_216886

noncomputable def functional_relationship (x : ℝ) : ℝ := -x + 26

theorem part1 (x y : ℝ) (hx6 : x = 6 ∧ y = 20) (hx8 : x = 8 ∧ y = 18) (hx10 : x = 10 ∧ y = 16) :
  ∀ (x : ℝ), functional_relationship x = -x + 26 := 
by
  sorry

theorem part2 (x : ℝ) (h_price_range : 6 ≤ x ∧ x ≤ 12) : 
  14 ≤ functional_relationship x ∧ functional_relationship x ≤ 20 :=
by
  sorry

noncomputable def gross_profit (x : ℝ) : ℝ := x * (functional_relationship x - 4)

theorem part3 (hx : 1 ≤ x) (hy : functional_relationship x ≤ 10):
  gross_profit (16 : ℝ) = 120 :=
by
  sorry

end part1_part2_part3_l216_216886


namespace number_of_gigs_played_l216_216914

/-- Given earnings per gig for each band member and the total earnings, prove the total number of gigs played -/

def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer1_earnings : ℕ := 15
def backup_singer2_earnings : ℕ := 18
def backup_singer3_earnings : ℕ := 12
def total_earnings : ℕ := 3465

def total_earnings_per_gig : ℕ :=
  lead_singer_earnings +
  guitarist_earnings +
  bassist_earnings +
  drummer_earnings +
  keyboardist_earnings +
  backup_singer1_earnings +
  backup_singer2_earnings +
  backup_singer3_earnings

theorem number_of_gigs_played : (total_earnings / total_earnings_per_gig) = 21 := by
  sorry

end number_of_gigs_played_l216_216914


namespace angle_sum_proof_l216_216708

theorem angle_sum_proof (x y : ℝ) (h : 3 * x + 6 * x + (x + y) + 4 * y = 360) : x = 0 ∧ y = 72 :=
by {
  sorry
}

end angle_sum_proof_l216_216708


namespace infinite_divisible_269_l216_216667

theorem infinite_divisible_269 (a : ℕ → ℤ) (h₀ : a 0 = 2) (h₁ : a 1 = 15) 
  (h_recur : ∀ n : ℕ, a (n + 2) = 15 * a (n + 1) + 16 * a n) :
  ∃ infinitely_many k: ℕ, 269 ∣ a k :=
by
  sorry

end infinite_divisible_269_l216_216667


namespace solve_system_eq_l216_216739

theorem solve_system_eq (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x + y ≠ 0) 
  (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) :
  (xy / (x + y) = 1 / 3) ∧ (yz / (y + z) = 1 / 4) ∧ (zx / (z + x) = 1 / 5) →
  (x = 1 / 2) ∧ (y = 1) ∧ (z = 1 / 3) :=
  sorry

end solve_system_eq_l216_216739


namespace bugs_max_contacts_l216_216813

theorem bugs_max_contacts :
  ∃ a b : ℕ, (a + b = 2016) ∧ (a * b = 1008^2) :=
by
  sorry

end bugs_max_contacts_l216_216813


namespace maximum_value_l216_216858

theorem maximum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (a / (a + 1) + b / (b + 2) ≤ (5 - 2 * Real.sqrt 2) / 4) :=
sorry

end maximum_value_l216_216858


namespace specified_percentage_of_number_is_40_l216_216216

theorem specified_percentage_of_number_is_40 
  (N : ℝ) 
  (hN : (1 / 4) * (1 / 3) * (2 / 5) * N = 25) 
  (P : ℝ) 
  (hP : (P / 100) * N = 300) : 
  P = 40 := 
sorry

end specified_percentage_of_number_is_40_l216_216216


namespace S_equals_2_l216_216260

noncomputable def problem_S := 
  1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
  1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)

theorem S_equals_2 : problem_S = 2 := by
  sorry

end S_equals_2_l216_216260


namespace two_mathematicians_contemporaries_l216_216724

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end two_mathematicians_contemporaries_l216_216724


namespace original_price_of_shirt_l216_216981

variables (S C P : ℝ)

def shirt_condition := S = C / 3
def pants_condition := S = P / 2
def total_paid := 0.90 * S + 0.95 * C + P = 900

theorem original_price_of_shirt :
  shirt_condition S C →
  pants_condition S P →
  total_paid S C P →
  S = 900 / 5.75 :=
by
  sorry

end original_price_of_shirt_l216_216981


namespace arithmetic_series_sum_l216_216823

theorem arithmetic_series_sum :
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  sum = 418 := by {
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  have h₁ : n = 11 := by sorry
  have h₂ : sum = 418 := by sorry
  exact h₂
}

end arithmetic_series_sum_l216_216823


namespace unique_solution_triple_l216_216916

def satisfies_system (x y z : ℝ) :=
  x^3 = 3 * x - 12 * y + 50 ∧
  y^3 = 12 * y + 3 * z - 2 ∧
  z^3 = 27 * z + 27 * x

theorem unique_solution_triple (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 2 ∧ y = 4 ∧ z = 6) :=
by sorry

end unique_solution_triple_l216_216916


namespace prime_square_plus_eight_is_prime_l216_216278

theorem prime_square_plus_eight_is_prime (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 8)) : p = 3 :=
sorry

end prime_square_plus_eight_is_prime_l216_216278


namespace simplify_expression_l216_216879

-- Define general term for y
variable (y : ℤ)

-- Statement representing the given proof problem
theorem simplify_expression :
  4 * y + 5 * y + 6 * y + 2 = 15 * y + 2 := 
sorry

end simplify_expression_l216_216879


namespace steel_more_by_l216_216820

variable {S T C k : ℝ}
variable (k_greater_than_zero : k > 0)
variable (copper_weight : C = 90)
variable (S_twice_T : S = 2 * T)
variable (S_minus_C : S = C + k)
variable (total_eq : 20 * S + 20 * T + 20 * C = 5100)

theorem steel_more_by (k): k = 20 := by
  sorry

end steel_more_by_l216_216820


namespace trapezoid_area_pqrs_l216_216373

theorem trapezoid_area_pqrs :
  let P := (1, 1)
  let Q := (1, 4)
  let R := (6, 4)
  let S := (7, 1)
  let parallelogram := true -- indicates that PQ and RS are parallel
  let PQ := abs (Q.2 - P.2)
  let RS := abs (S.1 - R.1)
  let height := abs (R.1 - P.1)
  (1 / 2 : ℚ) * (PQ + RS) * height = 10 := by
  sorry

end trapezoid_area_pqrs_l216_216373


namespace trig_identity_l216_216743

theorem trig_identity (θ : ℝ) (h₁ : Real.tan θ = 2) :
  2 * Real.cos θ / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 :=
by
  sorry

end trig_identity_l216_216743


namespace quad_intersects_x_axis_l216_216010

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l216_216010


namespace fraction_tabs_closed_l216_216585

theorem fraction_tabs_closed (x : ℝ) (h₁ : 400 * (1 - x) * (3/5) * (1/2) = 90) : 
  x = 1 / 4 :=
by
  have := h₁
  sorry

end fraction_tabs_closed_l216_216585


namespace compute_b_l216_216548

-- Defining the polynomial and the root conditions
def poly (x a b : ℝ) := x^3 + a * x^2 + b * x + 21

theorem compute_b (a b : ℚ) (h1 : poly (3 + Real.sqrt 5) a b = 0) (h2 : poly (3 - Real.sqrt 5) a b = 0) : 
  b = -27.5 := 
sorry

end compute_b_l216_216548


namespace abs_abc_eq_one_l216_216897

theorem abs_abc_eq_one 
  (a b c : ℝ)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a)
  (h_eq : a + 1/b^2 = b + 1/c^2 ∧ b + 1/c^2 = c + 1/a^2) : 
  |a * b * c| = 1 := 
sorry

end abs_abc_eq_one_l216_216897


namespace equal_books_for_students_l216_216456

-- Define the conditions
def num_girls : ℕ := 15
def num_boys : ℕ := 10
def total_books : ℕ := 375
def books_for_girls : ℕ := 225
def books_for_boys : ℕ := total_books - books_for_girls -- Calculate books for boys

-- Define the theorem
theorem equal_books_for_students :
  books_for_girls / num_girls = 15 ∧ books_for_boys / num_boys = 15 :=
by
  sorry

end equal_books_for_students_l216_216456


namespace range_of_a_l216_216146

theorem range_of_a
  (a : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ (x1 * x2 = 2 * a + 6)) :
  a < -3 :=
by
  sorry

end range_of_a_l216_216146


namespace cost_of_dvd_player_l216_216426

/-- The ratio of the cost of a DVD player to the cost of a movie is 9:2.
    A DVD player costs $63 more than a movie.
    Prove that the cost of the DVD player is $81. -/
theorem cost_of_dvd_player 
(D M : ℝ)
(h1 : D = (9 / 2) * M)
(h2 : D = M + 63) : 
D = 81 := 
sorry

end cost_of_dvd_player_l216_216426


namespace find_height_of_box_l216_216090

-- Given the conditions
variables (h l w : ℝ)
variables (V : ℝ)

-- Conditions as definitions in Lean
def length_eq_height (h : ℝ) : ℝ := 3 * h
def length_eq_width (w : ℝ) : ℝ := 4 * w
def volume_eq (h l w : ℝ) : ℝ := l * w * h

-- The proof problem: Prove height of the box is 12 given the conditions
theorem find_height_of_box : 
  (∃ h l w, l = 3 * h ∧ l = 4 * w ∧ l * w * h = 3888) → h = 12 :=
by
  sorry

end find_height_of_box_l216_216090


namespace arithmetic_common_difference_l216_216128

variable {α : Type*} [LinearOrderedField α]

-- Definition of arithmetic sequence
def arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  a + (n - 1) * d

-- Definition of sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_common_difference (a10 : α) (s10 : α) (d : α) (a1 : α) :
  arithmetic_seq a1 d 10 = a10 →
  sum_arithmetic_seq a1 d 10 = s10 →
  d = 2 / 3 :=
by
  sorry

end arithmetic_common_difference_l216_216128


namespace no_such_function_exists_l216_216123

theorem no_such_function_exists (f : ℕ → ℕ) : ¬ (∀ n : ℕ, n ≥ 2 → f (f (n - 1)) = f (n + 1) - f n) :=
sorry

end no_such_function_exists_l216_216123


namespace molecular_weight_is_62_024_l216_216335

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_atoms_H : ℕ := 2
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

def molecular_weight_compound : ℝ :=
  num_atoms_H * atomic_weight_H + num_atoms_C * atomic_weight_C + num_atoms_O * atomic_weight_O

theorem molecular_weight_is_62_024 :
  molecular_weight_compound = 62.024 :=
by
  have h_H := num_atoms_H * atomic_weight_H
  have h_C := num_atoms_C * atomic_weight_C
  have h_O := num_atoms_O * atomic_weight_O
  have h_sum := h_H + h_C + h_O
  show molecular_weight_compound = 62.024
  sorry

end molecular_weight_is_62_024_l216_216335


namespace repeating_decimal_product_l216_216834

theorem repeating_decimal_product (x : ℚ) (h : x = 4 / 9) : x * 9 = 4 := 
by
  sorry

end repeating_decimal_product_l216_216834


namespace total_students_in_school_l216_216580

noncomputable def total_students (girls boys : ℕ) (ratio_girls boys_ratio : ℕ) : ℕ :=
  let parts := ratio_girls + boys_ratio
  let students_per_part := girls / ratio_girls
  students_per_part * parts

theorem total_students_in_school (girls : ℕ) (ratio_girls boys_ratio : ℕ) (h1 : ratio_girls = 5) (h2 : boys_ratio = 8) (h3 : girls = 160) :
  total_students girls boys_ratio ratio_girls = 416 :=
  by
  -- proof would go here
  sorry

end total_students_in_school_l216_216580


namespace cone_volume_l216_216932

theorem cone_volume (V_f : ℝ) (A1 A2 : ℝ) (V : ℝ)
  (h1 : V_f = 78)
  (h2 : A1 = 9 * A2) :
  V = 81 :=
sorry

end cone_volume_l216_216932


namespace nina_spends_70_l216_216255

-- Definitions of the quantities and prices
def toys := 3
def toy_price := 10
def basketball_cards := 2
def card_price := 5
def shirts := 5
def shirt_price := 6

-- Calculate the total amount spent
def total_spent := (toys * toy_price) + (basketball_cards * card_price) + (shirts * shirt_price)

-- Problem statement: Prove that the total amount spent is $70
theorem nina_spends_70 : total_spent = 70 := by
  sorry

end nina_spends_70_l216_216255


namespace find_brick_width_l216_216307

def SurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

theorem find_brick_width :
  ∃ width : ℝ, SurfaceArea 10 width 3 = 164 ∧ width = 4 :=
by
  sorry

end find_brick_width_l216_216307


namespace percentage_of_x_l216_216605

theorem percentage_of_x (x y : ℝ) (h1 : y = x / 4) (p : ℝ) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 :=
by sorry

end percentage_of_x_l216_216605


namespace work_completion_days_l216_216900

theorem work_completion_days (x : ℕ) 
  (h1 : (1 : ℚ) / x + 1 / 9 = 1 / 6) :
  x = 18 := 
sorry

end work_completion_days_l216_216900


namespace decimal_equivalent_one_quarter_power_one_l216_216329

theorem decimal_equivalent_one_quarter_power_one : (1 / 4 : ℝ) ^ 1 = 0.25 := by
  sorry

end decimal_equivalent_one_quarter_power_one_l216_216329


namespace least_odd_prime_factor_2027_l216_216510

-- Definitions for the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def order_divides (a n p : ℕ) : Prop := a ^ n % p = 1

-- Define lean function to denote the problem.
theorem least_odd_prime_factor_2027 :
  ∀ p : ℕ, 
  is_prime p → 
  order_divides 2027 12 p ∧ ¬ order_divides 2027 6 p → 
  p ≡ 1 [MOD 12] → 
  2027^6 + 1 % p = 0 → 
  p = 37 :=
by
  -- skipping proof steps
  sorry

end least_odd_prime_factor_2027_l216_216510


namespace sufficient_but_not_necessary_condition_l216_216011

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x * (x - 1) < 0 → x < 1) ∧ ¬(x < 1 → x * (x - 1) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l216_216011


namespace archer_score_below_8_probability_l216_216727

theorem archer_score_below_8_probability :
  ∀ (p10 p9 p8 : ℝ), p10 = 0.2 → p9 = 0.3 → p8 = 0.3 → 
  (1 - (p10 + p9 + p8) = 0.2) :=
by
  intros p10 p9 p8 hp10 hp9 hp8
  rw [hp10, hp9, hp8]
  sorry

end archer_score_below_8_probability_l216_216727


namespace number_of_tacos_l216_216219

-- Define the conditions and prove the statement
theorem number_of_tacos (T : ℕ) :
  (4 * 7 + 9 * T = 37) → T = 1 :=
by
  intro h
  sorry

end number_of_tacos_l216_216219


namespace candy_last_days_l216_216772

theorem candy_last_days (candy_neighbors candy_sister candy_per_day : ℕ)
  (h1 : candy_neighbors = 5)
  (h2 : candy_sister = 13)
  (h3 : candy_per_day = 9):
  (candy_neighbors + candy_sister) / candy_per_day = 2 :=
by
  sorry

end candy_last_days_l216_216772


namespace greatest_integer_gcd_18_is_6_l216_216399

theorem greatest_integer_gcd_18_is_6 (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 18 = 6) : n = 138 := 
sorry

end greatest_integer_gcd_18_is_6_l216_216399


namespace minimum_value_of_f_l216_216493

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

theorem minimum_value_of_f :
  exists (x : ℝ), x = 1 + 1 / Real.sqrt 3 ∧ ∀ (y : ℝ), f (1 + 1 / Real.sqrt 3) ≤ f y := sorry

end minimum_value_of_f_l216_216493


namespace remainder_of_3_pow_19_mod_5_l216_216798

theorem remainder_of_3_pow_19_mod_5 : (3 ^ 19) % 5 = 2 := by
  have h : 3 ^ 4 % 5 = 1 := by sorry
  sorry

end remainder_of_3_pow_19_mod_5_l216_216798


namespace floor_identity_l216_216279

theorem floor_identity (x : ℝ) : 
    (⌊(3 + x) / 6⌋ - ⌊(4 + x) / 6⌋ + ⌊(5 + x) / 6⌋ = ⌊(1 + x) / 2⌋ - ⌊(1 + x) / 3⌋) :=
by
  sorry

end floor_identity_l216_216279


namespace g_value_at_50_l216_216112

noncomputable def g (x : ℝ) : ℝ := (1 - x) / 2

theorem g_value_at_50 :
  (∀ x y : ℝ, 0 < x → 0 < y → 
  (x * g y - y * g x = g (x / y) + x - y)) →
  g 50 = -24.5 :=
by
  intro h
  have h_g : ∀ x : ℝ, 0 < x → g x = (1 - x) / 2 := 
    fun x x_pos => sorry -- g(x) derivation proof goes here
  exact sorry -- Final answer proof goes here

end g_value_at_50_l216_216112


namespace sin_alpha_second_quadrant_l216_216403

/-- Given angle α in the second quadrant such that tan(π - α) = 3/4, we need to prove that sin α = 3/5. -/
theorem sin_alpha_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.tan (π - α) = 3 / 4) : Real.sin α = 3 / 5 := by
  sorry

end sin_alpha_second_quadrant_l216_216403


namespace kittens_percentage_rounded_l216_216664

theorem kittens_percentage_rounded (total_cats female_ratio kittens_per_female cats_sold : ℕ) (h1 : total_cats = 6)
  (h2 : female_ratio = 2)
  (h3 : kittens_per_female = 7)
  (h4 : cats_sold = 9) : 
  ((12 : ℤ) * 100 / (18 : ℤ)).toNat = 67 := by
  -- Historical reference and problem specific values involved 
  sorry

end kittens_percentage_rounded_l216_216664


namespace mul_powers_same_base_l216_216722

theorem mul_powers_same_base : 2^2 * 2^3 = 2^5 :=
by sorry

end mul_powers_same_base_l216_216722


namespace rocky_running_ratio_l216_216720

theorem rocky_running_ratio (x y : ℕ) (h1 : x = 4) (h2 : 2 * x + y = 36) : y / (2 * x) = 3 :=
by
  sorry

end rocky_running_ratio_l216_216720


namespace Joan_seashells_l216_216579

theorem Joan_seashells (J_J : ℕ) (J : ℕ) (h : J + J_J = 14) (hJJ : J_J = 8) : J = 6 :=
by
  sorry

end Joan_seashells_l216_216579


namespace negation_of_no_slow_learners_attend_school_l216_216024

variable {α : Type}
variable (SlowLearner : α → Prop) (AttendsSchool : α → Prop)

-- The original statement
def original_statement : Prop := ∀ x, SlowLearner x → ¬ AttendsSchool x

-- The corresponding negation
def negation_statement : Prop := ∃ x, SlowLearner x ∧ AttendsSchool x

-- The proof problem statement
theorem negation_of_no_slow_learners_attend_school : 
  ¬ original_statement SlowLearner AttendsSchool ↔ negation_statement SlowLearner AttendsSchool := by
  sorry

end negation_of_no_slow_learners_attend_school_l216_216024


namespace exam_failure_l216_216143

structure ExamData where
  max_marks : ℕ
  passing_percentage : ℚ
  secured_marks : ℕ

def passing_marks (data : ExamData) : ℚ :=
  data.passing_percentage * data.max_marks

theorem exam_failure (data : ExamData)
  (h1 : data.max_marks = 150)
  (h2 : data.passing_percentage = 40 / 100)
  (h3 : data.secured_marks = 40) :
  (passing_marks data - data.secured_marks : ℚ) = 20 := by
    sorry

end exam_failure_l216_216143


namespace total_toads_l216_216840

def pond_toads : ℕ := 12
def outside_toads : ℕ := 6

theorem total_toads : pond_toads + outside_toads = 18 :=
by
  -- Proof goes here
  sorry

end total_toads_l216_216840


namespace mikes_remaining_cards_l216_216902

variable (original_number_of_cards : ℕ)
variable (sam_bought : ℤ)
variable (alex_bought : ℤ)

theorem mikes_remaining_cards :
  original_number_of_cards = 87 →
  sam_bought = 8 →
  alex_bought = 13 →
  original_number_of_cards - (sam_bought + alex_bought) = 66 :=
by
  intros h_original h_sam h_alex
  rw [h_original, h_sam, h_alex]
  norm_num

end mikes_remaining_cards_l216_216902


namespace at_least_one_did_not_land_stably_l216_216194

-- Define the propositions p and q
variables (p q : Prop)

-- Define the theorem to prove
theorem at_least_one_did_not_land_stably :
  (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by
  sorry

end at_least_one_did_not_land_stably_l216_216194


namespace marcus_savings_l216_216236

def MarcusMaxPrice : ℝ := 130
def ShoeInitialPrice : ℝ := 120
def DiscountPercentage : ℝ := 0.30
def FinalPrice : ℝ := ShoeInitialPrice - (DiscountPercentage * ShoeInitialPrice)
def Savings : ℝ := MarcusMaxPrice - FinalPrice

theorem marcus_savings : Savings = 46 := by
  sorry

end marcus_savings_l216_216236


namespace max_value_k_l216_216023

theorem max_value_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
(h4 : 4 = k^2 * (x^2 / y^2 + 2 + y^2 / x^2) + k^3 * (x / y + y / x)) : 
k ≤ 4 * (Real.sqrt 2) - 4 :=
by sorry

end max_value_k_l216_216023


namespace age_of_teacher_l216_216795

theorem age_of_teacher (S : ℕ) (T : Real) (n : ℕ) (average_student_age : Real) (new_average_age : Real) : 
  average_student_age = 14 → 
  new_average_age = 14.66 → 
  n = 45 → 
  S = average_student_age * n → 
  T = 44.7 :=
by
  sorry

end age_of_teacher_l216_216795


namespace AplusBplusC_4_l216_216945

theorem AplusBplusC_4 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 1 ∧ Nat.gcd a c = 1 ∧ (a^2 + a * b + b^2 = c^2) ∧ (a + b + c = 4) :=
by
  sorry

end AplusBplusC_4_l216_216945


namespace pattyCoinsValue_l216_216764

def totalCoins (q d : ℕ) : Prop := q + d = 30
def originalValue (q d : ℕ) : ℝ := 0.25 * q + 0.10 * d
def swappedValue (q d : ℕ) : ℝ := 0.10 * q + 0.25 * d
def valueIncrease (q : ℕ) : Prop := swappedValue q (30 - q) - originalValue q (30 - q) = 1.20

theorem pattyCoinsValue (q d : ℕ) (h1 : totalCoins q d) (h2 : valueIncrease q) : originalValue q d = 4.65 := 
by
  sorry

end pattyCoinsValue_l216_216764


namespace sum_of_5_and_8_l216_216270

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  rfl

end sum_of_5_and_8_l216_216270


namespace total_goals_l216_216763

-- Definitions
def louie_goals_last_match := 4
def louie_previous_goals := 40
def brother_multiplier := 2
def seasons := 3
def games_per_season := 50

-- Total number of goals scored by Louie and his brother
theorem total_goals : (louie_previous_goals + louie_goals_last_match) 
                      + (brother_multiplier * louie_goals_last_match * seasons * games_per_season) 
                      = 1244 :=
by sorry

end total_goals_l216_216763


namespace min_value_expression_l216_216214

theorem min_value_expression : ∃ x : ℝ, x = 300 ∧ ∀ y : ℝ, (y^2 - 600*y + 369) ≥ (300^2 - 600*300 + 369) := by
  use 300
  sorry

end min_value_expression_l216_216214


namespace otimes_self_twice_l216_216978

def otimes (x y : ℝ) := x^2 - y^2

theorem otimes_self_twice (a : ℝ) : (otimes (otimes a a) (otimes a a)) = 0 :=
  sorry

end otimes_self_twice_l216_216978


namespace typing_speed_ratio_l216_216132

theorem typing_speed_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.25 * M = 14) : M / T = 2 :=
by
  sorry

end typing_speed_ratio_l216_216132


namespace problem_f_neg2_l216_216860

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2007 + b * x + 1

theorem problem_f_neg2 (a b : ℝ) (h : f a b 2 = 2) : f a b (-2) = 0 :=
by
  sorry

end problem_f_neg2_l216_216860


namespace fixed_point_on_line_AB_always_exists_l216_216642

-- Define the line where P lies
def line (x y : ℝ) : Prop := x + 2 * y = 4

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4

-- Define the point P
def moving_point_P (x y : ℝ) : Prop := line x y

-- Define the function that checks if a point is a tangent to the ellipse
def is_tangent (x0 y0 x y : ℝ) : Prop :=
  moving_point_P x0 y0 → (x * x0 + 4 * y * y0 = 4)

-- Statement: There exists a fixed point (1, 1/2) through which the line AB always passes
theorem fixed_point_on_line_AB_always_exists :
  ∀ (P A B : ℝ × ℝ),
    moving_point_P P.1 P.2 →
    is_tangent P.1 P.2 A.1 A.2 →
    is_tangent P.1 P.2 B.1 B.2 →
    ∃ (F : ℝ × ℝ), F = (1, 1/2) ∧ (F.1 - A.1) / (F.2 - A.2) = (F.1 - B.1) / (F.2 - B.2) :=
by
  sorry

end fixed_point_on_line_AB_always_exists_l216_216642


namespace solve_inequality_l216_216686

theorem solve_inequality (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ( if 0 ≤ a ∧ a < 1 / 2 then (x > a ∧ x < 1 - a) else 
    if a = 1 / 2 then false else 
    if 1 / 2 < a ∧ a ≤ 1 then (x > 1 - a ∧ x < a) else false ) ↔ ((x - a) * (x + a - 1) < 0) :=
by
  sorry

end solve_inequality_l216_216686


namespace value_of_x_sq_plus_inv_x_sq_l216_216984

theorem value_of_x_sq_plus_inv_x_sq (x : ℝ) (h : x + 1/x = 1.5) : x^2 + (1/x)^2 = 0.25 := 
by 
  sorry

end value_of_x_sq_plus_inv_x_sq_l216_216984


namespace first_digit_l216_216177

-- Definitions and conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

def number (x y : ℕ) : ℕ := 653 * 100 + x * 10 + y

-- Main theorem
theorem first_digit (x y : ℕ) (h₁ : isDivisibleBy (number x y) 80) (h₂ : x + y = 2) : x = 2 :=
sorry

end first_digit_l216_216177


namespace relationship_y1_y2_l216_216862

theorem relationship_y1_y2 (x1 x2 y1 y2 : ℝ) 
  (h1: x1 > 0) 
  (h2: 0 > x2) 
  (h3: y1 = 2 / x1)
  (h4: y2 = 2 / x2) : 
  y1 > y2 :=
by
  sorry

end relationship_y1_y2_l216_216862


namespace base6_addition_correct_l216_216689

-- We define the numbers in base 6
def a_base6 : ℕ := 2 * 6^3 + 4 * 6^2 + 5 * 6^1 + 3 * 6^0
def b_base6 : ℕ := 1 * 6^4 + 6 * 6^3 + 4 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Define the expected result in base 6 and its base 10 equivalent
def result_base6 : ℕ := 2 * 6^4 + 5 * 6^3 + 5 * 6^2 + 4 * 6^1 + 5 * 6^0
def result_base10 : ℕ := 3881

-- The proof statement
theorem base6_addition_correct : (a_base6 + b_base6 = result_base6) ∧ (result_base6 = result_base10) := by
  sorry

end base6_addition_correct_l216_216689


namespace last_digit_of_power_sum_l216_216359

theorem last_digit_of_power_sum (m : ℕ) (hm : 0 < m) : (2^(m + 2006) + 2^m) % 10 = 0 := 
sorry

end last_digit_of_power_sum_l216_216359


namespace trains_clear_time_l216_216540

-- Definitions based on conditions
def length_train1 : ℕ := 160
def length_train2 : ℕ := 280
def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

-- Conversion factor from km/h to m/s
def kmph_to_mps (s : ℕ) : ℕ := s * 1000 / 3600

-- Computation of relative speed in m/s
def relative_speed_mps : ℕ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Total distance to be covered for the trains to clear each other
def total_distance : ℕ := length_train1 + length_train2

-- Time taken for the trains to clear each other
def time_to_clear_each_other : ℕ := total_distance / relative_speed_mps

-- Theorem stating that time taken is 22 seconds
theorem trains_clear_time : time_to_clear_each_other = 22 := by
  sorry

end trains_clear_time_l216_216540


namespace tan_105_degree_l216_216034

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l216_216034


namespace pqrs_sum_l216_216093

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end pqrs_sum_l216_216093


namespace trigonometry_identity_l216_216197

theorem trigonometry_identity (α : ℝ) (P : ℝ × ℝ) (h : P = (4, -3)) :
  let x := P.1
  let y := P.2
  let r := Real.sqrt (x^2 + y^2)
  x = 4 →
  y = -3 →
  r = 5 →
  Real.tan α = y / x := by
  intros x y r hx hy hr
  rw [hx, hy]
  simp [Real.tan, div_eq_mul_inv, mul_comm]
  sorry

end trigonometry_identity_l216_216197


namespace pencils_in_stock_at_end_of_week_l216_216613

def pencils_per_day : ℕ := 100
def days_per_week : ℕ := 5
def initial_pencils : ℕ := 80
def sold_pencils : ℕ := 350

theorem pencils_in_stock_at_end_of_week :
  (pencils_per_day * days_per_week + initial_pencils - sold_pencils) = 230 :=
by sorry  -- Proof will be filled in later

end pencils_in_stock_at_end_of_week_l216_216613


namespace stream_speed_l216_216891

theorem stream_speed (u v : ℝ) (h1 : 27 = 9 * (u - v)) (h2 : 81 = 9 * (u + v)) : v = 3 :=
by
  sorry

end stream_speed_l216_216891


namespace volume_relation_l216_216913

noncomputable def A (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
noncomputable def M (r : ℝ) : ℝ := 2 * Real.pi * r^3
noncomputable def C (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_relation (r : ℝ) : A r - M r + C r = 0 :=
by
  sorry

end volume_relation_l216_216913


namespace Z_equals_i_l216_216725

noncomputable def Z : ℂ := (Real.sqrt 2 - (Complex.I ^ 3)) / (1 - Real.sqrt 2 * Complex.I)

theorem Z_equals_i : Z = Complex.I := 
by 
  sorry

end Z_equals_i_l216_216725


namespace find_linear_function_passing_A_B_l216_216368

-- Conditions
def line_function (k b x : ℝ) : ℝ := k * x + b

theorem find_linear_function_passing_A_B :
  (∃ k b : ℝ, k ≠ 0 ∧ line_function k b 1 = 3 ∧ line_function k b 0 = -2) → 
  ∃ k b : ℝ, k = 5 ∧ b = -2 ∧ ∀ x : ℝ, line_function k b x = 5 * x - 2 :=
by
  -- Proof will be added here
  sorry

end find_linear_function_passing_A_B_l216_216368


namespace subtraction_example_l216_216628

theorem subtraction_example : 3.57 - 1.45 = 2.12 :=
by 
  sorry

end subtraction_example_l216_216628


namespace boat_speed_greater_than_stream_l216_216212

def boat_stream_speed_difference (S U V : ℝ) := 
  (S / (U - V)) - (S / (U + V)) + (S / (2 * V + 1)) = 1

theorem boat_speed_greater_than_stream 
  (S : ℝ) (U V : ℝ) 
  (h_dist : S = 1) 
  (h_time_diff : boat_stream_speed_difference S U V) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_stream_l216_216212


namespace find_m_l216_216923

open Real

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) (m : ℝ) : ℝ :=
  2 * cos (ω * x + ϕ) + m

theorem find_m (ω ϕ : ℝ) (hω : 0 < ω)
  (symmetry : ∀ t : ℝ,  f (π / 4 - t) ω ϕ m = f t ω ϕ m)
  (f_π_8 : f (π / 8) ω ϕ m = -1) :
  m = -3 ∨ m = 1 := 
sorry

end find_m_l216_216923


namespace yan_distance_ratio_l216_216790

theorem yan_distance_ratio 
  (w x y : ℝ)
  (h1 : y / w = x / w + (x + y) / (10 * w)) :
  x / y = 9 / 11 :=
by
  sorry

end yan_distance_ratio_l216_216790


namespace sum_S17_l216_216519

-- Definitions of the required arithmetic sequence elements.
variable (a1 d : ℤ)

-- Definition of the arithmetic sequence
def aₙ (n : ℤ) : ℤ := a1 + (n - 1) * d
def Sₙ (n : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Theorem for the problem statement
theorem sum_S17 : (aₙ a1 d 7 + aₙ a1 d 5) = (3 + aₙ a1 d 5) → (a1 + 8 * d = 3) → Sₙ a1 d 17 = 51 :=
by
  intros h1 h2
  sorry

end sum_S17_l216_216519


namespace visited_iceland_l216_216117

variable (total : ℕ) (visitedNorway : ℕ) (visitedBoth : ℕ) (visitedNeither : ℕ)

theorem visited_iceland (h_total : total = 50)
                        (h_visited_norway : visitedNorway = 23)
                        (h_visited_both : visitedBoth = 21)
                        (h_visited_neither : visitedNeither = 23) :
                        (total - (visitedNorway - visitedBoth + visitedNeither) = 25) :=
  sorry

end visited_iceland_l216_216117


namespace martins_travel_time_l216_216315

-- Declare the necessary conditions from the problem
variables (speed : ℝ) (distance : ℝ)
-- Define the conditions
def martin_speed := speed = 12 -- Martin's speed is 12 miles per hour
def martin_distance := distance = 72 -- Martin drove 72 miles

-- State the theorem to prove the time taken is 6 hours
theorem martins_travel_time (h1 : martin_speed speed) (h2 : martin_distance distance) : distance / speed = 6 :=
by
  -- To complete the problem statement, insert sorry to skip the actual proof
  sorry

end martins_travel_time_l216_216315


namespace solve_for_a_l216_216142

theorem solve_for_a (x a : ℤ) (h1 : x = 3) (h2 : x + 2 * a = -1) : a = -2 :=
by
  sorry

end solve_for_a_l216_216142


namespace child_grandmother_ratio_l216_216317

variable (G D C : ℕ)

axiom cond1 : G + D + C = 120
axiom cond2 : D + C = 60
axiom cond3 : D = 48

theorem child_grandmother_ratio : (C : ℚ) / G = 1 / 5 :=
by
  sorry

end child_grandmother_ratio_l216_216317


namespace time_at_simple_interest_l216_216070

theorem time_at_simple_interest 
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * (R + 5) / 100) * T = (P * (R / 100) * T) + 150) : 
  T = 10 := 
by 
  -- Proof is omitted.
  sorry

end time_at_simple_interest_l216_216070


namespace total_cost_is_correct_l216_216310

-- Define the number of total tickets and the number of children's tickets
def total_tickets : ℕ := 21
def children_tickets : ℕ := 16
def adult_tickets : ℕ := total_tickets - children_tickets

-- Define the cost of tickets for adults and children
def cost_per_adult_ticket : ℝ := 5.50
def cost_per_child_ticket : ℝ := 3.50

-- Define the total cost spent
def total_cost_spent : ℝ :=
  (adult_tickets * cost_per_adult_ticket) + (children_tickets * cost_per_child_ticket)

-- Prove that the total amount spent on tickets is $83.50
theorem total_cost_is_correct : total_cost_spent = 83.50 := by
  sorry

end total_cost_is_correct_l216_216310


namespace total_pears_picked_l216_216707

theorem total_pears_picked :
  let mike_pears := 8
  let jason_pears := 7
  let fred_apples := 6
  -- The total number of pears picked is the sum of Mike's and Jason's pears.
  mike_pears + jason_pears = 15 :=
by {
  sorry
}

end total_pears_picked_l216_216707


namespace maximum_value_expression_l216_216281

theorem maximum_value_expression (a b c : ℕ) (ha : 0 < a ∧ a ≤ 9) (hb : 0 < b ∧ b ≤ 9) (hc : 0 < c ∧ c ≤ 9) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (v : ℚ), v = (1 / (a + 2010 / (b + 1 / c : ℚ))) ∧ v ≤ (1 / 203) :=
sorry

end maximum_value_expression_l216_216281


namespace zoo_recovery_time_l216_216356

theorem zoo_recovery_time (lions rhinos recover_time : ℕ) (total_animals : ℕ) (total_time : ℕ)
    (h_lions : lions = 3) (h_rhinos : rhinos = 2) (h_recover_time : recover_time = 2)
    (h_total_animals : total_animals = lions + rhinos) (h_total_time : total_time = total_animals * recover_time) :
    total_time = 10 :=
by
  rw [h_lions, h_rhinos] at h_total_animals
  rw [h_total_animals, h_recover_time] at h_total_time
  exact h_total_time

end zoo_recovery_time_l216_216356


namespace min_value_frac_l216_216775

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (c : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → c ≤ 8 / x + 2 / y) ∧ c = 18 :=
sorry

end min_value_frac_l216_216775


namespace circle_radius_zero_l216_216563

theorem circle_radius_zero :
  ∀ (x y : ℝ),
    (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) →
    ((x - 1)^2 + (y - 2)^2 = 0) → 
    0 = 0 :=
by
  intros x y h_eq h_circle
  sorry

end circle_radius_zero_l216_216563


namespace marbles_left_l216_216405

-- Definitions and conditions
def marbles_initial : ℕ := 38
def marbles_lost : ℕ := 15

-- Statement of the problem
theorem marbles_left : marbles_initial - marbles_lost = 23 := by
  sorry

end marbles_left_l216_216405


namespace set_theory_problem_l216_216838

def U : Set ℤ := {x ∈ Set.univ | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_theory_problem : 
  (A ∩ B = {4}) ∧ 
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧ 
  (U \ (A ∪ C) = {6, 8, 10}) ∧ 
  ((U \ A) ∩ (U \ B) = {3}) := 
by 
  sorry

end set_theory_problem_l216_216838


namespace range_of_f_l216_216624

open Set

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

theorem range_of_f :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → 2 ≤ f x ∧ f x ≤ 3 :=
by
  intro x hx
  sorry

end range_of_f_l216_216624


namespace kali_height_now_l216_216957

variable (K_initial J_initial : ℝ)
variable (K_growth J_growth : ℝ)
variable (J_current : ℝ)

theorem kali_height_now :
  J_initial = K_initial →
  J_growth = (2 / 3) * 0.3 * K_initial →
  K_growth = 0.3 * K_initial →
  J_current = 65 →
  J_current = J_initial + J_growth →
  K_current = K_initial + K_growth →
  K_current = 70.42 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end kali_height_now_l216_216957


namespace total_lives_l216_216422

def initial_players := 25
def additional_players := 10
def lives_per_player := 15

theorem total_lives :
  (initial_players + additional_players) * lives_per_player = 525 := by
  sorry

end total_lives_l216_216422


namespace derivative_at_pi_div_3_l216_216756

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 2) * Real.sin x - Real.cos x

theorem derivative_at_pi_div_3 :
  deriv f (π / 3) = (1 / 2) * (1 + Real.sqrt 2 + Real.sqrt 3) :=
by
  sorry

end derivative_at_pi_div_3_l216_216756


namespace correct_calculation_result_l216_216922

theorem correct_calculation_result (x : ℤ) (h : 4 * x + 16 = 32) : (x / 4) + 16 = 17 := by
  sorry

end correct_calculation_result_l216_216922


namespace students_count_l216_216705

theorem students_count (S : ℕ) (num_adults : ℕ) (cost_student cost_adult total_cost : ℕ)
  (h1 : num_adults = 4)
  (h2 : cost_student = 5)
  (h3 : cost_adult = 6)
  (h4 : total_cost = 199) :
  5 * S + 4 * 6 = 199 → S = 35 := by
  sorry

end students_count_l216_216705


namespace problem_solution_l216_216425

variable (f : ℝ → ℝ)

noncomputable def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x < 1/2) ∨ (2 < x)

theorem problem_solution
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_1 : f 1 = 2) :
  ∀ x, f (Real.log x / Real.log 2) > 2 ↔ solution_set x :=
by
  sorry

end problem_solution_l216_216425


namespace geometric_sequence_a4_l216_216817

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ (n : ℕ), a (n + 1) = a n * r

def a_3a_5_is_64 (a : ℕ → ℝ) : Prop :=
  a 3 * a 5 = 64

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h1 : is_geometric_sequence a) (h2 : a_3a_5_is_64 a) : a 4 = 8 ∨ a 4 = -8 :=
by
  sorry

end geometric_sequence_a4_l216_216817


namespace calculate_value_l216_216457

theorem calculate_value : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end calculate_value_l216_216457


namespace minimum_vertical_distance_l216_216407

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 - 3 * x - 5

theorem minimum_vertical_distance :
  ∃ x : ℝ, (∀ y : ℝ, |absolute_value y - quadratic_function y| ≥ 4) ∧ (|absolute_value x - quadratic_function x| = 4) := 
sorry

end minimum_vertical_distance_l216_216407


namespace bird_mammal_difference_africa_asia_l216_216126

noncomputable def bird_families_to_africa := 42
noncomputable def bird_families_to_asia := 31
noncomputable def bird_families_to_south_america := 7

noncomputable def mammal_families_to_africa := 24
noncomputable def mammal_families_to_asia := 18
noncomputable def mammal_families_to_south_america := 15

noncomputable def reptile_families_to_africa := 15
noncomputable def reptile_families_to_asia := 9
noncomputable def reptile_families_to_south_america := 5

-- Calculate the total number of families migrating to Africa, Asia, and South America
noncomputable def total_families_to_africa := bird_families_to_africa + mammal_families_to_africa + reptile_families_to_africa
noncomputable def total_families_to_asia := bird_families_to_asia + mammal_families_to_asia + reptile_families_to_asia
noncomputable def total_families_to_south_america := bird_families_to_south_america + mammal_families_to_south_america + reptile_families_to_south_america

-- Calculate the combined total of bird and mammal families going to Africa
noncomputable def bird_and_mammal_families_to_africa := bird_families_to_africa + mammal_families_to_africa

-- Difference between bird and mammal families to Africa and total animal families to Asia
noncomputable def difference := bird_and_mammal_families_to_africa - total_families_to_asia

theorem bird_mammal_difference_africa_asia : difference = 8 := 
by
  sorry

end bird_mammal_difference_africa_asia_l216_216126


namespace hexahedron_volume_l216_216584

open Real

noncomputable def volume_of_hexahedron (AB A1B1 AA1 : ℝ) : ℝ :=
  let S_base := (3 * sqrt 3 / 2) * AB^2
  let S_top := (3 * sqrt 3 / 2) * A1B1^2
  let h := AA1
  (1 / 3) * h * (S_base + sqrt (S_base * S_top) + S_top)

theorem hexahedron_volume : volume_of_hexahedron 2 3 (sqrt 10) = 57 * sqrt 3 / 2 := by
  sorry

end hexahedron_volume_l216_216584


namespace a_alone_can_finish_in_60_days_l216_216733

variables (A B C : ℚ)

noncomputable def a_b_work_rate := A + B = 1/40
noncomputable def a_c_work_rate := A + 1/30 = 1/20

theorem a_alone_can_finish_in_60_days (A B C : ℚ) 
  (h₁ : a_b_work_rate A B) 
  (h₂ : a_c_work_rate A) : 
  A = 1/60 := 
sorry

end a_alone_can_finish_in_60_days_l216_216733


namespace electricity_price_per_kWh_l216_216423

theorem electricity_price_per_kWh (consumption_rate : ℝ) (hours_used : ℝ) (total_cost : ℝ) :
  consumption_rate = 2.4 → hours_used = 25 → total_cost = 6 →
  total_cost / (consumption_rate * hours_used) = 0.10 :=
by
  intros hc hh ht
  have h_energy : consumption_rate * hours_used = 60 :=
    by rw [hc, hh]; norm_num
  rw [ht, h_energy]
  norm_num

end electricity_price_per_kWh_l216_216423


namespace number_of_true_propositions_is_2_l216_216956

-- Definitions for the propositions
def original_proposition (x : ℝ) : Prop := x > -3 → x > -6
def converse_proposition (x : ℝ) : Prop := x > -6 → x > -3
def inverse_proposition (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive_proposition (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The theorem we need to prove
theorem number_of_true_propositions_is_2 :
  (∀ x, original_proposition x) ∧ (∀ x, contrapositive_proposition x) ∧ 
  ¬ (∀ x, converse_proposition x) ∧ ¬ (∀ x, inverse_proposition x) → 2 = 2 := 
sorry

end number_of_true_propositions_is_2_l216_216956


namespace margin_in_terms_of_selling_price_l216_216815

variable (C S M : ℝ) (n : ℕ) (h : M = (1 / 2) * (S - (1 / n) * C))

theorem margin_in_terms_of_selling_price :
  M = ((n - 1) / (2 * n - 1)) * S :=
sorry

end margin_in_terms_of_selling_price_l216_216815


namespace august_first_problem_answer_l216_216241

theorem august_first_problem_answer (A : ℕ)
  (h1 : 2 * A = B)
  (h2 : 3 * A - 400 = C)
  (h3 : A + B + C = 3200) : A = 600 :=
sorry

end august_first_problem_answer_l216_216241


namespace rickshaw_distance_l216_216487

theorem rickshaw_distance (km1_charge : ℝ) (rate_per_km : ℝ) (total_km : ℝ) (total_charge : ℝ) :
  km1_charge = 13.50 → rate_per_km = 2.50 → total_km = 13 → total_charge = 103.5 → (total_charge - km1_charge) / rate_per_km = 36 :=
by
  intro h1 h2 h3 h4
  -- We would fill in proof steps here, but skipping as required.
  sorry

end rickshaw_distance_l216_216487


namespace clubs_equal_students_l216_216163

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l216_216163


namespace largest_common_term_l216_216361

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end largest_common_term_l216_216361


namespace correct_calculation_l216_216008

theorem correct_calculation :
  (∀ a : ℝ, a^3 + a^2 ≠ a^5) ∧
  (∀ a : ℝ, a^3 / a^2 = a) ∧
  (∀ a : ℝ, 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ a : ℝ, (a - 2)^2 ≠ a^2 - 4) :=
by
  sorry

end correct_calculation_l216_216008


namespace certain_number_is_47_l216_216037

theorem certain_number_is_47 (x : ℤ) (h : 34 + x - 53 = 28) : x = 47 :=
by
  sorry

end certain_number_is_47_l216_216037


namespace m_plus_n_eq_five_l216_216106

theorem m_plus_n_eq_five (m n : ℝ) (h1 : m - 2 = 0) (h2 : 1 + n - 2 * m = 0) : m + n = 5 := 
  by 
  sorry

end m_plus_n_eq_five_l216_216106


namespace remainder_of_sum_of_primes_l216_216909

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l216_216909


namespace ab_range_l216_216461

theorem ab_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a * b = a + b) : 1 / 4 ≤ a * b :=
sorry

end ab_range_l216_216461


namespace total_tape_length_is_230_l216_216939

def tape_length (n : ℕ) (len_piece : ℕ) (overlap : ℕ) : ℕ :=
  len_piece + (n - 1) * (len_piece - overlap)

theorem total_tape_length_is_230 :
  tape_length 15 20 5 = 230 := 
    sorry

end total_tape_length_is_230_l216_216939


namespace no_solution_exists_l216_216122

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end no_solution_exists_l216_216122


namespace number_of_boys_in_school_l216_216049

theorem number_of_boys_in_school (total_students : ℕ) (sample_size : ℕ) 
(number_diff : ℕ) (ratio_boys_sample_girls_sample : ℚ) : 
total_students = 1200 → sample_size = 200 → number_diff = 10 →
ratio_boys_sample_girls_sample = 105 / 95 →
∃ (boys_in_school : ℕ), boys_in_school = 630 := by 
  sorry

end number_of_boys_in_school_l216_216049


namespace min_tries_to_get_blue_and_yellow_l216_216252

theorem min_tries_to_get_blue_and_yellow 
  (purple blue yellow : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5)
  (h_yellow : yellow = 11) :
  ∃ n, n = 9 ∧ (∀ tries, tries ≥ n → (∃ i j, (i ≤ purple ∧ j ≤ tries - i ∧ j ≤ blue) → (∃ k, k = tries - i - j ∧ k ≤ yellow))) :=
by sorry

end min_tries_to_get_blue_and_yellow_l216_216252


namespace polynomial_factorization_l216_216227

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end polynomial_factorization_l216_216227


namespace triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l216_216367

variable {a b c x : ℝ}

-- Part (1)
theorem triangle_ABC_is_isosceles (h : (a + b) * 1 ^ 2 - 2 * c * 1 + (a - b) = 0) : a = c :=
by 
  -- Proof omitted
  sorry

-- Part (2)
theorem roots_of_quadratic_for_equilateral (h_eq : a = b ∧ b = c ∧ c = a) : 
  (∀ x : ℝ, (a + a) * x ^ 2 - 2 * a * x + (a - a) = 0 → (x = 0 ∨ x = 1)) :=
by 
  -- Proof omitted
  sorry

end triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l216_216367


namespace percent_more_than_l216_216383

-- Definitions and conditions
variables (x y p : ℝ)

-- Condition: x is p percent more than y
def x_is_p_percent_more_than_y (x y p : ℝ) : Prop :=
  x = y + (p / 100) * y

-- The theorem to prove
theorem percent_more_than (h : x_is_p_percent_more_than_y x y p) :
  p = 100 * (x / y - 1) :=
sorry

end percent_more_than_l216_216383


namespace part_one_part_two_part_three_l216_216692

open Nat

def number_boys := 5
def number_girls := 4
def total_people := 9
def A_included := 1
def B_included := 1

theorem part_one : (number_boys.choose 2 * number_girls.choose 2) = 60 := sorry

theorem part_two : (total_people.choose 4 - (total_people - A_included - B_included).choose 4) = 91 := sorry

theorem part_three : (total_people.choose 4 - number_boys.choose 4 - number_girls.choose 4) = 120 := sorry

end part_one_part_two_part_three_l216_216692


namespace expected_rainfall_week_l216_216149

theorem expected_rainfall_week :
  let P_sun := 0.35
  let P_2 := 0.40
  let P_8 := 0.25
  let rainfall_2 := 2
  let rainfall_8 := 8
  let daily_expected := P_sun * 0 + P_2 * rainfall_2 + P_8 * rainfall_8
  let total_expected := 7 * daily_expected
  total_expected = 19.6 :=
by
  sorry

end expected_rainfall_week_l216_216149


namespace solution_1_solution_2_l216_216521

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

def critical_point_condition (a x : ℝ) : Prop :=
  (x = 1 / 4) → deriv (f a) x = 0

def pseudo_symmetry_point_condition (a : ℝ) (x0 : ℝ) : Prop :=
  let f' := fun x => 2 * x^2 - 5 * x + Real.log x
  let g := fun x => (4 * x0^2 - 5 * x0 + 1) / x0 * (x - x0) + 2 * x0^2 - 5 * x0 + Real.log x0
  ∀ x : ℝ, 
    (0 < x ∧ x < x0) → (f' x - g x < 0) ∧ 
    (x > x0) → (f' x - g x > 0)

theorem solution_1 (a : ℝ) (h1 : a > 0) (h2 : critical_point_condition a (1/4)) :
  a = 4 := 
sorry

theorem solution_2 (x0 : ℝ) (h1 : x0 = 1/2) :
  pseudo_symmetry_point_condition 4 x0 :=
sorry


end solution_1_solution_2_l216_216521


namespace total_income_l216_216455

theorem total_income (I : ℝ) (h1 : 0.10 * I * 2 + 0.20 * I + 0.06 * (I - 0.40 * I) = 0.46 * I) (h2 : 0.54 * I = 500) : I = 500 / 0.54 :=
by
  sorry

end total_income_l216_216455


namespace find_value_l216_216327

theorem find_value (a b : ℝ) (h : a + b + 1 = -2) : (a + b - 1) * (1 - a - b) = -16 := by
  sorry

end find_value_l216_216327


namespace largest_of_consecutive_non_prime_integers_l216_216392

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of consecutive non-prime sequence condition
def consecutive_non_prime_sequence (start : ℕ) : Prop :=
  ∀ i : ℕ, 0 ≤ i → i < 10 → ¬ is_prime (start + i)

theorem largest_of_consecutive_non_prime_integers :
  (∃ start, start + 9 < 50 ∧ consecutive_non_prime_sequence start) →
  (∃ start, start + 9 = 47) :=
by
  sorry

end largest_of_consecutive_non_prime_integers_l216_216392


namespace parallel_lines_slope_l216_216066

theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2 * y + 1 = 0 → ∀ x y : ℝ, x + y - 2 = 0 → True) → 
  a = 2 :=
by
  sorry

end parallel_lines_slope_l216_216066


namespace value_of_x_l216_216144

theorem value_of_x (x : ℝ) (h : (0.7 * x) - ((1 / 3) * x) = 110) : x = 300 :=
sorry

end value_of_x_l216_216144


namespace rectangle_area_l216_216427

theorem rectangle_area (r l b : ℝ) (h1: r = 30) (h2: l = (2 / 5) * r) (h3: b = 10) : 
  l * b = 120 := 
by
  sorry

end rectangle_area_l216_216427


namespace apples_left_l216_216696

-- Define the initial number of apples and the conditions
def initial_apples := 150
def percent_sold_to_jill := 20 / 100
def percent_sold_to_june := 30 / 100
def apples_given_to_teacher := 2

-- Formulate the problem statement in Lean
theorem apples_left (initial_apples percent_sold_to_jill percent_sold_to_june apples_given_to_teacher : ℕ) :
  let sold_to_jill := percent_sold_to_jill * initial_apples
  let remaining_after_jill := initial_apples - sold_to_jill
  let sold_to_june := percent_sold_to_june * remaining_after_jill
  let remaining_after_june := remaining_after_jill - sold_to_june
  let final_apples := remaining_after_june - apples_given_to_teacher
  final_apples = 82 := 
by 
  sorry

end apples_left_l216_216696


namespace valid_S2_example_l216_216167

def satisfies_transformation (S1 S2 : List ℕ) : Prop :=
  S2 = S1.map (λ n => (S1.count n : ℕ))

theorem valid_S2_example : 
  ∃ S1 : List ℕ, satisfies_transformation S1 [1, 2, 1, 1, 2] :=
by
  sorry

end valid_S2_example_l216_216167


namespace count_lattice_points_on_hyperbola_l216_216211

theorem count_lattice_points_on_hyperbola : 
  (∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S ↔ (p.1 ^ 2 - p.2 ^ 2 = 1800 ^ 2)) ∧ S.card = 150) :=
sorry

end count_lattice_points_on_hyperbola_l216_216211


namespace linear_equation_in_two_vars_example_l216_216375

def is_linear_equation_in_two_vars (eq : String) : Prop :=
  eq = "x + 4y = 6"

theorem linear_equation_in_two_vars_example :
  is_linear_equation_in_two_vars "x + 4y = 6" :=
by
  sorry

end linear_equation_in_two_vars_example_l216_216375


namespace divides_2_pow_26k_plus_2_plus_3_by_19_l216_216372

theorem divides_2_pow_26k_plus_2_plus_3_by_19 (k : ℕ) : 19 ∣ (2^(26*k+2) + 3) := 
by
  sorry

end divides_2_pow_26k_plus_2_plus_3_by_19_l216_216372


namespace minimum_value_of_objective_function_l216_216774

theorem minimum_value_of_objective_function :
  ∃ (x y : ℝ), x - y + 2 ≥ 0 ∧ 2 * x + 3 * y - 6 ≥ 0 ∧ 3 * x + 2 * y - 9 ≤ 0 ∧ (∀ (x' y' : ℝ), x' - y' + 2 ≥ 0 ∧ 2 * x' + 3 * y' - 6 ≥ 0 ∧ 3 * x' + 2 * y' - 9 ≤ 0 → 2 * x + 5 * y ≤ 2 * x' + 5 * y') ∧ 2 * x + 5 * y = 6 :=
sorry

end minimum_value_of_objective_function_l216_216774


namespace range_of_m_l216_216139

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - m * x^2

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ (∀ x, x = x₁ ∨ x = x₂ ∨ f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)

theorem range_of_m :
  ∀ m : ℝ, has_two_extreme_points (m) ↔ 0 < m ∧ m < 1 / 2 := 
by
  sorry

end range_of_m_l216_216139


namespace find_cos_minus_sin_l216_216746

-- Definitions from the conditions
variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)  -- Second quadrant
variable (h2 : Real.sin (2 * α) = -24 / 25)  -- Given sin 2α

-- Lean statement of the problem
theorem find_cos_minus_sin (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) :
  Real.cos α - Real.sin α = -7 / 5 := 
sorry

end find_cos_minus_sin_l216_216746


namespace tangent_line_at_0_2_is_correct_l216_216803

noncomputable def curve (x : ℝ) : ℝ := Real.exp (-2 * x) + 1

def tangent_line_at_0_2 (x : ℝ) : ℝ := -2 * x + 2

theorem tangent_line_at_0_2_is_correct :
  tangent_line_at_0_2 = fun x => -2 * x + 2 :=
by {
  sorry
}

end tangent_line_at_0_2_is_correct_l216_216803


namespace time_between_peanuts_l216_216518

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := 4
def flight_time_hours : ℕ := 2

theorem time_between_peanuts (peanuts_per_bag number_of_bags flight_time_hours : ℕ) (h1 : peanuts_per_bag = 30) (h2 : number_of_bags = 4) (h3 : flight_time_hours = 2) :
  (flight_time_hours * 60) / (peanuts_per_bag * number_of_bags) = 1 := by
  sorry

end time_between_peanuts_l216_216518


namespace p_necessary_not_sufficient_for_q_l216_216850

open Real

noncomputable def p (x : ℝ) : Prop := |x| < 3
noncomputable def q (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l216_216850


namespace no_five_consecutive_integers_with_fourth_powers_sum_l216_216610

theorem no_five_consecutive_integers_with_fourth_powers_sum:
  ∀ n : ℤ, n^4 + (n + 1)^4 + (n + 2)^4 + (n + 3)^4 ≠ (n + 4)^4 :=
by
  intros
  sorry

end no_five_consecutive_integers_with_fourth_powers_sum_l216_216610


namespace inequality_k_m_l216_216271

theorem inequality_k_m (k m : ℕ) (hk : 0 < k) (hm : 0 < m) (hkm : k > m) (hdiv : (k^3 - m^3) ∣ k * m * (k^2 - m^2)) :
  (k - m)^3 > 3 * k * m := 
by sorry

end inequality_k_m_l216_216271


namespace range_of_a_if_proposition_l216_216052

theorem range_of_a_if_proposition :
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → -4 < a ∧ a < 2 := by
  sorry

end range_of_a_if_proposition_l216_216052


namespace unique_real_solution_l216_216986

theorem unique_real_solution (x y : ℝ) (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
sorry

end unique_real_solution_l216_216986


namespace minimum_value_of_expression_l216_216268

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  x^2 + x * y + y^2 + 7

theorem minimum_value_of_expression :
  ∃ x y : ℝ, min_value_expression x y = 7 :=
by
  use 0, 0
  sorry

end minimum_value_of_expression_l216_216268


namespace sqrt_of_product_of_powers_l216_216525

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end sqrt_of_product_of_powers_l216_216525


namespace complement_of_intersection_l216_216695

-- Declare the universal set U
def U : Set ℤ := {-1, 1, 2, 3}

-- Declare the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the given quadratic equation
def is_solution (x : ℤ) : Prop := x^2 - 2 * x - 3 = 0
def B : Set ℤ := {x : ℤ | is_solution x}

-- The main theorem to prove
theorem complement_of_intersection (A_inter_B_complement : Set ℤ) :
  A_inter_B_complement = {1, 2, 3} :=
by
  sorry

end complement_of_intersection_l216_216695


namespace max_marks_l216_216498

theorem max_marks (M : ℝ) (h1 : 80 + 10 = 90) (h2 : 0.30 * M = 90) : M = 300 :=
by
  sorry

end max_marks_l216_216498


namespace no_valid_N_for_case1_valid_N_values_for_case2_l216_216341

variable (P R N : ℕ)
variable (N_less_than_40 : N < 40)
variable (avg_all : ℕ)
variable (avg_promoted : ℕ)
variable (avg_repeaters : ℕ)
variable (new_avg_promoted : ℕ)
variable (new_avg_repeaters : ℕ)

variables
  (promoted_condition : (71 * P + 56 * R) / N = 66)
  (increase_condition : (76 * P) / (P + R) = 75 ∧ (61 * R) / (P + R) = 59)
  (equation1 : 71 * P = 2 * R)
  (equation2: P + R = N)

-- Proof for part (a)
theorem no_valid_N_for_case1 
  (new_avg_promoted' : ℕ := 75) 
  (new_avg_repeaters' : ℕ := 59)
  : ∀ N, ¬ N < 40 ∨ ¬ ((76 * P) / (P + R) = new_avg_promoted' ∧ (61 * R) / (P + R) = new_avg_repeaters') := 
  sorry

-- Proof for part (b)
theorem valid_N_values_for_case2
  (possible_N_values : Finset ℕ := {6, 12, 18, 24, 30, 36})
  (new_avg_promoted'' : ℕ := 79)
  (new_avg_repeaters'' : ℕ := 47)
  : ∀ N, N ∈ possible_N_values ↔ (((76 * P) / (P + R) = new_avg_promoted'') ∧ (61 * R) / (P + R) = new_avg_repeaters'') := 
  sorry

end no_valid_N_for_case1_valid_N_values_for_case2_l216_216341


namespace find_missing_percentage_l216_216677

theorem find_missing_percentage (P : ℝ) : (P * 50 = 2.125) → (P * 100 = 4.25) :=
by
  sorry

end find_missing_percentage_l216_216677


namespace largest_possible_s_l216_216663

theorem largest_possible_s (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (h_angle : (101 : ℚ) / 97 * ((s - 2) * 180 / s : ℚ) = ((r - 2) * 180 / r : ℚ)) :
  s = 100 :=
by
  sorry

end largest_possible_s_l216_216663


namespace intersection_points_product_l216_216478

theorem intersection_points_product (x y : ℝ) :
  (x^2 - 2 * x + y^2 - 6 * y + 9 = 0) ∧ (x^2 - 8 * x + y^2 - 6 * y + 28 = 0) → x * y = 6 :=
by
  sorry

end intersection_points_product_l216_216478


namespace small_to_large_circle_ratio_l216_216055

theorem small_to_large_circle_ratio (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 5 * π * a^2) :
  a / b = 1 / Real.sqrt 6 :=
by
  sorry

end small_to_large_circle_ratio_l216_216055


namespace problem_intersecting_lines_l216_216453

theorem problem_intersecting_lines (c d : ℝ) :
  (3 : ℝ) = (1 / 3 : ℝ) * (6 : ℝ) + c ∧ (6 : ℝ) = (1 / 3 : ℝ) * (3 : ℝ) + d → c + d = 6 :=
by
  intros h
  sorry

end problem_intersecting_lines_l216_216453


namespace remaining_average_l216_216282

-- Definitions
def original_average (n : ℕ) (avg : ℝ) := n = 50 ∧ avg = 38
def discarded_numbers (a b : ℝ) := a = 45 ∧ b = 55

-- Proof Statement
theorem remaining_average (n : ℕ) (avg : ℝ) (a b : ℝ) (s : ℝ) :
  original_average n avg →
  discarded_numbers a b →
  s = (n * avg - (a + b)) / (n - 2) →
  s = 37.5 :=
by
  intros h_avg h_discard h_s
  sorry

end remaining_average_l216_216282


namespace charlie_book_pages_l216_216541

theorem charlie_book_pages :
  (2 * 40) + (4 * 45) + 20 = 280 :=
by 
  sorry

end charlie_book_pages_l216_216541


namespace total_number_of_students_l216_216678

theorem total_number_of_students (sample_size : ℕ) (first_year_selected : ℕ) (third_year_selected : ℕ) (second_year_students : ℕ) (second_year_selected : ℕ) (prob_selection : ℕ) :
  sample_size = 45 →
  first_year_selected = 20 →
  third_year_selected = 10 →
  second_year_students = 300 →
  second_year_selected = sample_size - first_year_selected - third_year_selected →
  prob_selection = second_year_selected / second_year_students →
  (sample_size / prob_selection) = 900 :=
by
  intros
  sorry

end total_number_of_students_l216_216678


namespace solution_largest_a_exists_polynomial_l216_216946

def largest_a_exists_polynomial : Prop :=
  ∃ (P : ℝ → ℝ) (a b c d e : ℝ),
    (∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + e) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 → 0 ≤ P x ∧ P x ≤ 1) ∧
    a = 4

theorem solution_largest_a_exists_polynomial : largest_a_exists_polynomial :=
  sorry

end solution_largest_a_exists_polynomial_l216_216946


namespace multiply_expression_l216_216680

theorem multiply_expression (x : ℝ) : 
  (x^4 + 49 * x^2 + 2401) * (x^2 - 49) = x^6 - 117649 :=
by
  sorry

end multiply_expression_l216_216680


namespace find_quantities_of_raib_ornaments_and_pendants_l216_216679

theorem find_quantities_of_raib_ornaments_and_pendants (x y : ℕ)
  (h1 : x + y = 90)
  (h2 : 40 * x + 25 * y = 2850) :
  x = 40 ∧ y = 50 :=
sorry

end find_quantities_of_raib_ornaments_and_pendants_l216_216679


namespace which_point_is_in_fourth_quadrant_l216_216013

def point (x: ℝ) (y: ℝ) : Prop := x > 0 ∧ y < 0

theorem which_point_is_in_fourth_quadrant :
  point 5 (-4) :=
by {
  -- proofs for each condition can be added,
  sorry
}

end which_point_is_in_fourth_quadrant_l216_216013


namespace john_needs_2_sets_l216_216041

-- Definition of the conditions
def num_bars_per_set : ℕ := 7
def total_bars : ℕ := 14

-- The corresponding proof problem statement
theorem john_needs_2_sets : total_bars / num_bars_per_set = 2 :=
by
  sorry

end john_needs_2_sets_l216_216041


namespace part1_part2_l216_216915

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x * (Real.sin x + Real.cos x)) - 1 / 2

theorem part1 (α : ℝ) (hα1 : 0 < α ∧ α < Real.pi / 2) (hα2 : Real.sin α = Real.sqrt 2 / 2) :
  f α = 1 / 2 :=
sorry

theorem part2 :
  ∀ (k : ℤ), ∀ (x : ℝ),
  -((3 : ℝ) * Real.pi / 8) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 8) + k * Real.pi →
  MonotoneOn f (Set.Icc (-((3 : ℝ) * Real.pi / 8) + k * Real.pi) ((Real.pi / 8) + k * Real.pi)) :=
sorry

end part1_part2_l216_216915


namespace distribute_papers_l216_216643

theorem distribute_papers (n m : ℕ) (h_n : n = 5) (h_m : m = 10) : 
  (m ^ n) = 100000 :=
by 
  rw [h_n, h_m]
  rfl

end distribute_papers_l216_216643


namespace max_number_of_small_boxes_l216_216684

def volume_of_large_box (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_small_box (length width height : ℕ) : ℕ :=
  length * width * height

def number_of_small_boxes (large_volume small_volume : ℕ) : ℕ :=
  large_volume / small_volume

theorem max_number_of_small_boxes :
  let large_box_length := 4 * 100  -- in cm
  let large_box_width := 2 * 100  -- in cm
  let large_box_height := 4 * 100  -- in cm
  let small_box_length := 4  -- in cm
  let small_box_width := 2  -- in cm
  let small_box_height := 2  -- in cm
  let large_volume := volume_of_large_box large_box_length large_box_width large_box_height
  let small_volume := volume_of_small_box small_box_length small_box_width small_box_height
  number_of_small_boxes large_volume small_volume = 2000000 := by
  -- Prove the statement
  sorry

end max_number_of_small_boxes_l216_216684


namespace thirteenth_result_is_878_l216_216701

-- Definitions based on the conditions
def avg_25_results : ℕ := 50
def num_25_results : ℕ := 25

def avg_first_12_results : ℕ := 14
def num_first_12_results : ℕ := 12

def avg_last_12_results : ℕ := 17
def num_last_12_results : ℕ := 12

-- Prove the 13th result is 878 given the above conditions.
theorem thirteenth_result_is_878 : 
  ((avg_25_results * num_25_results) - ((avg_first_12_results * num_first_12_results) + (avg_last_12_results * num_last_12_results))) = 878 :=
by
  sorry

end thirteenth_result_is_878_l216_216701


namespace jar_water_transfer_l216_216263

theorem jar_water_transfer
  (C_x : ℝ) (C_y : ℝ)
  (h1 : C_y = 1/2 * C_x)
  (WaterInX : ℝ)
  (WaterInY : ℝ)
  (h2 : WaterInX = 1/2 * C_x)
  (h3 : WaterInY = 1/2 * C_y) :
  WaterInX + WaterInY = 3/4 * C_x :=
by
  sorry

end jar_water_transfer_l216_216263


namespace kids_go_to_camp_l216_216582

theorem kids_go_to_camp (total_kids: Nat) (kids_stay_home: Nat) 
  (h1: total_kids = 1363293) (h2: kids_stay_home = 907611) : total_kids - kids_stay_home = 455682 :=
by
  have h_total : total_kids = 1363293 := h1
  have h_stay_home : kids_stay_home = 907611 := h2
  sorry

end kids_go_to_camp_l216_216582


namespace coloring_impossible_l216_216516

theorem coloring_impossible :
  ¬ ∃ (color : ℕ → Prop), (∀ n m : ℕ, (m = n + 5 → color n ≠ color m) ∧ (m = 2 * n → color n ≠ color m)) :=
sorry

end coloring_impossible_l216_216516


namespace cody_paid_17_l216_216408

-- Definitions for the conditions
def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def final_price_after_discount : ℝ := initial_cost * (1 + tax_rate) - discount
def cody_payment : ℝ := 17

-- The proof statement
theorem cody_paid_17 :
  cody_payment = (final_price_after_discount / 2) :=
by
  -- Proof steps, which we omit by using sorry
  sorry

end cody_paid_17_l216_216408


namespace applicant_overall_score_l216_216750

-- Definitions for the conditions
def writtenTestScore : ℝ := 80
def interviewScore : ℝ := 60
def weightWrittenTest : ℝ := 0.6
def weightInterview : ℝ := 0.4

-- Theorem statement
theorem applicant_overall_score : 
  (writtenTestScore * weightWrittenTest) + (interviewScore * weightInterview) = 72 := 
by
  sorry

end applicant_overall_score_l216_216750


namespace correct_answer_l216_216674

theorem correct_answer (x : ℝ) (h1 : 2 * x = 60) : x / 2 = 15 :=
by
  sorry

end correct_answer_l216_216674


namespace total_number_of_numbers_l216_216636

-- Definitions using the conditions from the problem
def sum_of_first_4_numbers : ℕ := 4 * 4
def sum_of_last_4_numbers : ℕ := 4 * 4
def average_of_all_numbers (n : ℕ) : ℕ := 3 * n
def fourth_number : ℕ := 11
def total_sum_of_numbers : ℕ := sum_of_first_4_numbers + sum_of_last_4_numbers - fourth_number

-- Theorem stating the problem
theorem total_number_of_numbers (n : ℕ) : total_sum_of_numbers = average_of_all_numbers n → n = 7 :=
by {
  sorry
}

end total_number_of_numbers_l216_216636


namespace economical_refuel_l216_216056

theorem economical_refuel (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  (x + y) / 2 > (2 * x * y) / (x + y) :=
sorry -- Proof omitted

end economical_refuel_l216_216056


namespace complement_of_M_wrt_U_l216_216782

-- Definitions of the sets U and M as given in the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

-- The goal is to show the complement of M w.r.t. U is {2, 4, 6}
theorem complement_of_M_wrt_U :
  (U \ M) = {2, 4, 6} := 
by
  sorry

end complement_of_M_wrt_U_l216_216782


namespace seq_a_eval_a4_l216_216483

theorem seq_a_eval_a4 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) : a 4 = 15 :=
sorry

end seq_a_eval_a4_l216_216483


namespace number_of_pairs_l216_216332

noncomputable def number_of_ordered_pairs (n : ℕ) : ℕ :=
  if n = 5 then 8 else 0

theorem number_of_pairs (f m: ℕ) : f ≥ 0 ∧ m ≥ 0 → number_of_ordered_pairs 5 = 8 :=
by
  intro h
  sorry

end number_of_pairs_l216_216332


namespace crossed_out_number_is_29_l216_216061

theorem crossed_out_number_is_29 : 
  ∀ n : ℕ, (11 * n + 66 - (325 - (12 * n + 66 - 325))) = 29 :=
by sorry

end crossed_out_number_is_29_l216_216061


namespace tangent_slope_at_point_552_32_l216_216547

noncomputable def slope_of_tangent_at_point (cx cy px py : ℚ) : ℚ :=
if py - cy = 0 then 
  0 
else 
  (px - cx) / (py - cy)

theorem tangent_slope_at_point_552_32 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 :=
by
  -- Conditions from problem
  have h1 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 := 
    sorry
  
  exact h1

end tangent_slope_at_point_552_32_l216_216547


namespace article_final_price_l216_216881

theorem article_final_price (list_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  first_discount = 0.1 → 
  second_discount = 0.01999999999999997 → 
  list_price = 70 → 
  ∃ final_price, final_price = 61.74 := 
by {
  sorry
}

end article_final_price_l216_216881


namespace more_girls_than_boys_l216_216866

def ratio_boys_girls (B G : ℕ) : Prop := B = (3/5 : ℚ) * G

def total_students (B G : ℕ) : Prop := B + G = 16

theorem more_girls_than_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : G - B = 4 :=
by
  sorry

end more_girls_than_boys_l216_216866


namespace brenda_ends_with_12_skittles_l216_216961

def initial_skittles : ℕ := 7
def bought_skittles : ℕ := 8
def given_away_skittles : ℕ := 3

theorem brenda_ends_with_12_skittles :
  initial_skittles + bought_skittles - given_away_skittles = 12 := by
  sorry

end brenda_ends_with_12_skittles_l216_216961


namespace find_radius_of_stationary_tank_l216_216818

theorem find_radius_of_stationary_tank
  (h_stationary : Real) (r_truck : Real) (h_truck : Real) (drop : Real) (V_truck : Real)
  (ht1 : h_stationary = 25)
  (ht2 : r_truck = 4)
  (ht3 : h_truck = 10)
  (ht4 : drop = 0.016)
  (ht5 : V_truck = π * r_truck ^ 2 * h_truck) :
  ∃ R : Real, π * R ^ 2 * drop = V_truck ∧ R = 100 :=
by
  sorry

end find_radius_of_stationary_tank_l216_216818


namespace intersection_with_y_axis_l216_216574

-- Define the original linear function
def original_function (x : ℝ) : ℝ := -2 * x + 3

-- Define the function after moving it up by 2 units
def moved_up_function (x : ℝ) : ℝ := original_function x + 2

-- State the theorem to prove the intersection with the y-axis
theorem intersection_with_y_axis : moved_up_function 0 = 5 :=
by
  sorry

end intersection_with_y_axis_l216_216574


namespace strawberries_per_box_l216_216251

-- Define the initial conditions
def initial_strawberries : ℕ := 42
def additional_strawberries : ℕ := 78
def number_of_boxes : ℕ := 6

-- Define the total strawberries based on the given conditions
def total_strawberries : ℕ := initial_strawberries + additional_strawberries

-- The theorem to prove the number of strawberries per box
theorem strawberries_per_box : total_strawberries / number_of_boxes = 20 :=
by
  -- Proof steps would go here, but we use sorry since it's not required
  sorry

end strawberries_per_box_l216_216251


namespace mark_saves_5_dollars_l216_216669

def cost_per_pair : ℤ := 50

def promotionA_total_cost (cost : ℤ) : ℤ :=
  cost + (cost / 2)

def promotionB_total_cost (cost : ℤ) : ℤ :=
  cost + (cost - 20)

def savings (totalB totalA : ℤ) : ℤ :=
  totalB - totalA

theorem mark_saves_5_dollars :
  savings (promotionB_total_cost cost_per_pair) (promotionA_total_cost cost_per_pair) = 5 := by
  sorry

end mark_saves_5_dollars_l216_216669


namespace find_a5_l216_216511

variable {a_n : ℕ → ℤ} -- Type of the arithmetic sequence
variable (d : ℤ)       -- Common difference of the sequence

-- Assuming the sequence is defined as an arithmetic progression
axiom arithmetic_seq (a d : ℤ) : ∀ n : ℕ, a_n n = a + n * d

theorem find_a5
  (h : a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 45):
  a_n 5 = 9 :=
by 
  sorry

end find_a5_l216_216511


namespace number_of_triangles_with_perimeter_20_l216_216973

-- Declare the condition: number of triangles with integer side lengths and perimeter of 20
def integerTrianglesWithPerimeter (n : ℕ) : ℕ :=
  (Finset.range (n/2 + 1)).card

/-- Prove that the number of triangles with integer side lengths and a perimeter of 20 is 8. -/
theorem number_of_triangles_with_perimeter_20 : integerTrianglesWithPerimeter 20 = 8 := 
  sorry

end number_of_triangles_with_perimeter_20_l216_216973


namespace power_function_inverse_l216_216952

theorem power_function_inverse (f : ℝ → ℝ) (h₁ : f 2 = (Real.sqrt 2) / 2) : f⁻¹ 2 = 1 / 4 :=
by
  -- Lean proof will be filled here
  sorry

end power_function_inverse_l216_216952


namespace solve_for_x_l216_216704

theorem solve_for_x (x : ℤ) (h : x + 1 = 4) : x = 3 :=
sorry

end solve_for_x_l216_216704


namespace original_price_of_dish_l216_216675

-- Define the variables and conditions explicitly
variables (P : ℝ)

-- John's payment after discount and tip over original price
def john_payment : ℝ := 0.9 * P + 0.15 * P

-- Jane's payment after discount and tip over discounted price
def jane_payment : ℝ := 0.9 * P + 0.135 * P

-- Given condition that John's payment is $0.63 more than Jane's
def payment_difference : Prop := john_payment P - jane_payment P = 0.63

theorem original_price_of_dish (h : payment_difference P) : P = 42 :=
by sorry

end original_price_of_dish_l216_216675


namespace second_number_value_l216_216936

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end second_number_value_l216_216936


namespace lizette_quiz_average_l216_216925

theorem lizette_quiz_average
  (Q1 Q2 : ℝ)
  (Q3 : ℝ := 92)
  (h : (Q1 + Q2 + Q3) / 3 = 94) :
  (Q1 + Q2) / 2 = 95 := by
sorry

end lizette_quiz_average_l216_216925


namespace alcohol_percentage_l216_216287

theorem alcohol_percentage (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) 
(h3 : (0.6 + (x / 100) * 6 = 2.4)) : x = 30 :=
by sorry

end alcohol_percentage_l216_216287


namespace find_x_for_g_equal_20_l216_216447

theorem find_x_for_g_equal_20 (g f : ℝ → ℝ) (h₁ : ∀ x, g x = 4 * (f⁻¹ x))
    (h₂ : ∀ x, f x = 30 / (x + 5)) :
    ∃ x, g x = 20 ∧ x = 3 := by
  sorry

end find_x_for_g_equal_20_l216_216447


namespace manager_salary_l216_216170

theorem manager_salary 
  (a : ℝ) (n : ℕ) (m_total : ℝ) (new_avg : ℝ) (m_avg_inc : ℝ)
  (h1 : n = 20) 
  (h2 : a = 1600) 
  (h3 : m_avg_inc = 100) 
  (h4 : new_avg = a + m_avg_inc)
  (h5 : m_total = n * a)
  (h6 : new_avg = (m_total + M) / (n + 1)) : 
  M = 3700 :=
by
  sorry

end manager_salary_l216_216170


namespace subset_N_M_l216_216910

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | x^2 - x < 0 }

-- The proof goal
theorem subset_N_M : N ⊆ M := by
  sorry

end subset_N_M_l216_216910


namespace remainder_of_2345678_div_5_l216_216371

theorem remainder_of_2345678_div_5 : (2345678 % 5) = 3 :=
by
  sorry

end remainder_of_2345678_div_5_l216_216371


namespace length_of_edge_l216_216651

-- Define all necessary conditions
def is_quadrangular_pyramid (e : ℝ) : Prop :=
  (8 * e = 14.8)

-- State the main theorem which is the equivalent proof problem
theorem length_of_edge (e : ℝ) (h : is_quadrangular_pyramid e) : e = 1.85 :=
by
  sorry

end length_of_edge_l216_216651


namespace total_votes_l216_216693

theorem total_votes (T F A : ℝ)
  (h1 : F = A + 68)
  (h2 : A = 0.40 * T)
  (h3 : T = F + A) :
  T = 340 :=
by sorry

end total_votes_l216_216693


namespace polynomial_solution_l216_216568

noncomputable def q (x : ℝ) : ℝ :=
  -20 / 93 * x^3 - 110 / 93 * x^2 - 372 / 93 * x - 525 / 93

theorem polynomial_solution :
  (q 1 = -11) ∧
  (q 2 = -15) ∧
  (q 3 = -25) ∧
  (q 5 = -65) :=
by
  sorry

end polynomial_solution_l216_216568


namespace number_of_people_study_only_cooking_l216_216593

def total_yoga : Nat := 25
def total_cooking : Nat := 18
def total_weaving : Nat := 10
def cooking_and_yoga : Nat := 5
def all_three : Nat := 4
def cooking_and_weaving : Nat := 5

theorem number_of_people_study_only_cooking :
  (total_cooking - (cooking_and_yoga + cooking_and_weaving - all_three)) = 12 :=
by
  sorry

end number_of_people_study_only_cooking_l216_216593


namespace jason_hours_saturday_l216_216435

def hours_after_school (x : ℝ) : ℝ := 4 * x
def hours_saturday (y : ℝ) : ℝ := 6 * y

theorem jason_hours_saturday 
  (x y : ℝ) 
  (total_hours : x + y = 18) 
  (total_earnings : 4 * x + 6 * y = 88) : 
  y = 8 :=
by 
  sorry

end jason_hours_saturday_l216_216435


namespace baker_cakes_total_l216_216401

def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

theorem baker_cakes_total : 
  (initial_cakes - cakes_sold) + additional_cakes = 111 := by
  sorry

end baker_cakes_total_l216_216401


namespace roots_purely_imaginary_l216_216661

open Complex

/-- 
  If m is a purely imaginary number, then the roots of the equation 
  8z^2 + 4i * z - m = 0 are purely imaginary.
-/
theorem roots_purely_imaginary (m : ℂ) (hm : m.im ≠ 0 ∧ m.re = 0) : 
  ∀ z : ℂ, 8 * z^2 + 4 * Complex.I * z - m = 0 → z.im ≠ 0 ∧ z.re = 0 :=
by
  sorry

end roots_purely_imaginary_l216_216661


namespace max_value_of_a1_l216_216318

theorem max_value_of_a1 (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h_distinct : ∀ i j, i ≠ j → (i ≠ a1 → i ≠ a2 → i ≠ a3 → i ≠ a4 → i ≠ a5 → i ≠ a6 → i ≠ a7)) 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 159) : a1 ≤ 19 :=
by
  sorry

end max_value_of_a1_l216_216318


namespace multiply_of_Mari_buttons_l216_216065

-- Define the variables and constants from the problem
def Mari_buttons : ℕ := 8
def Sue_buttons : ℕ := 22
def Kendra_buttons : ℕ := 2 * Sue_buttons

-- Statement that we need to prove
theorem multiply_of_Mari_buttons : ∃ (x : ℕ), Kendra_buttons = 8 * x + 4 ∧ x = 5 := by
  sorry

end multiply_of_Mari_buttons_l216_216065


namespace final_amoeba_is_blue_l216_216546

-- We define the initial counts of each type of amoeba
def initial_red : ℕ := 26
def initial_blue : ℕ := 31
def initial_yellow : ℕ := 16

-- We define the final count of amoebas
def final_amoebas : ℕ := 1

-- The type of the final amoeba (we're proving it's 'blue')
inductive AmoebaColor
| Red
| Blue
| Yellow

-- Given initial counts, we aim to prove the final amoeba is blue
theorem final_amoeba_is_blue :
  initial_red = 26 ∧ initial_blue = 31 ∧ initial_yellow = 16 ∧ final_amoebas = 1 → 
  ∃ c : AmoebaColor, c = AmoebaColor.Blue :=
by sorry

end final_amoeba_is_blue_l216_216546


namespace fedya_incorrect_l216_216336

theorem fedya_incorrect 
  (a b c d : ℕ) 
  (a_ends_in_9 : a % 10 = 9)
  (b_ends_in_7 : b % 10 = 7)
  (c_ends_in_3 : c % 10 = 3)
  (d_is_1 : d = 1) : 
  a ≠ b * c + d :=
by {
  sorry
}

end fedya_incorrect_l216_216336


namespace total_flour_l216_216345

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end total_flour_l216_216345


namespace container_solution_exists_l216_216083

theorem container_solution_exists (x y : ℕ) (h : 130 * x + 160 * y = 3000) : 
  (x = 12) ∧ (y = 9) :=
by sorry

end container_solution_exists_l216_216083


namespace Roselyn_initial_books_correct_l216_216911

variables (Roselyn_initial_books Mara_books Rebecca_books : ℕ)

-- Conditions
axiom A1 : Rebecca_books = 40
axiom A2 : Mara_books = 3 * Rebecca_books
axiom A3 : Roselyn_initial_books - (Rebecca_books + Mara_books) = 60

-- Proof statement
theorem Roselyn_initial_books_correct : Roselyn_initial_books = 220 :=
sorry

end Roselyn_initial_books_correct_l216_216911


namespace functional_eq_solve_l216_216275

theorem functional_eq_solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solve_l216_216275


namespace Sam_dimes_remaining_l216_216808

-- Define the initial and borrowed dimes
def initial_dimes_count : Nat := 8
def borrowed_dimes_count : Nat := 4

-- State the theorem
theorem Sam_dimes_remaining : (initial_dimes_count - borrowed_dimes_count) = 4 := by
  sorry

end Sam_dimes_remaining_l216_216808


namespace find_value_of_expression_l216_216948

variables (a b c : ℝ)

theorem find_value_of_expression
  (h1 : a ^ 4 * b ^ 3 * c ^ 5 = 18)
  (h2 : a ^ 3 * b ^ 5 * c ^ 4 = 8) :
  a ^ 5 * b * c ^ 6 = 81 / 2 :=
sorry

end find_value_of_expression_l216_216948


namespace Bella_bought_38_stamps_l216_216005

def stamps (n t r : ℕ) : ℕ :=
  n + t + r

theorem Bella_bought_38_stamps :
  ∃ (n t r : ℕ),
    n = 11 ∧
    t = n + 9 ∧
    r = t - 13 ∧
    stamps n t r = 38 := 
  by
  sorry

end Bella_bought_38_stamps_l216_216005


namespace marble_probability_l216_216088

theorem marble_probability (g w r b : ℕ) (h_g : g = 4) (h_w : w = 3) (h_r : r = 5) (h_b : b = 6) :
  (g + w + r + b = 18) → (g + w = 7) → (7 / 18 = 7 / 18) :=
by
  sorry

end marble_probability_l216_216088


namespace value_of_B_l216_216529

theorem value_of_B (B : ℝ) : 3 * B ^ 2 + 3 * B + 2 = 29 ↔ (B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2) :=
by sorry

end value_of_B_l216_216529


namespace pushups_count_l216_216785

theorem pushups_count :
  ∀ (David Zachary Hailey : ℕ),
    David = 44 ∧ (David = Zachary + 9) ∧ (Zachary = 2 * Hailey) ∧ (Hailey = 27) →
      (David = 63 ∧ Zachary = 54 ∧ Hailey = 27) :=
by
  intros David Zachary Hailey
  intro conditions
  obtain ⟨hDavid44, hDavid9Zachary, hZachary2Hailey, hHailey27⟩ := conditions
  sorry

end pushups_count_l216_216785


namespace investment_ratio_correct_l216_216551

variable (P Q : ℝ)
variable (investment_ratio: ℝ := 7 / 5)
variable (profit_ratio: ℝ := 7 / 10)
variable (time_p: ℝ := 7)
variable (time_q: ℝ := 14)

theorem investment_ratio_correct :
  (P * time_p) / (Q * time_q) = profit_ratio → (P / Q) = investment_ratio := 
by
  sorry

end investment_ratio_correct_l216_216551


namespace inequality_am_gm_l216_216360

variable {u v : ℝ}

theorem inequality_am_gm (hu : 0 < u) (hv : 0 < v) : u ^ 3 + v ^ 3 ≥ u ^ 2 * v + v ^ 2 * u := by
  sorry

end inequality_am_gm_l216_216360


namespace isosceles_triangle_sides_l216_216486

theorem isosceles_triangle_sides (a b c : ℕ) (h₁ : a + b + c = 10) (h₂ : (a = b ∨ b = c ∨ a = c)) 
  (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) : 
  (a = 3 ∧ b = 3 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 4) := 
by
  sorry

end isosceles_triangle_sides_l216_216486


namespace unique_handshakes_l216_216617

-- Define the circular arrangement and handshakes conditions
def num_people := 30
def handshakes_per_person := 2

theorem unique_handshakes : 
  (num_people * handshakes_per_person) / 2 = 30 :=
by
  -- Sorry is used here as a placeholder for the proof
  sorry

end unique_handshakes_l216_216617


namespace num_counting_numbers_dividing_52_leaving_remainder_7_l216_216415

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem num_counting_numbers_dividing_52_leaving_remainder_7 (n : ℕ) :
  (∃ n : ℕ, 59 ≡ 7 [MOD n]) → (n > 7 ∧ divides n 52) → n = 3 := 
sorry

end num_counting_numbers_dividing_52_leaving_remainder_7_l216_216415


namespace part_a_part_b_part_c_l216_216718

-- Define the conditions for Payneful pairs
def isPaynefulPair (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x ∈ Set.univ) ∧
  (∀ x, g x ∈ Set.univ) ∧
  (∀ x y, f (x + y) = f x * g y + g x * f y) ∧
  (∀ x y, g (x + y) = g x * g y - f x * f y) ∧
  (∃ a, f a ≠ 0)

-- Questions and corresponding proofs as Lean theorems
theorem part_a (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : f 0 = 0 ∧ g 0 = 1 := sorry

def h (f g : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ 2 + (g x) ^ 2

theorem part_b (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : h f g 5 * h f g (-5) = 1 := sorry

theorem part_c (f g : ℝ → ℝ) (hf : isPaynefulPair f g)
  (h_bound_f : ∀ x, -10 ≤ f x ∧ f x ≤ 10) (h_bound_g : ∀ x, -10 ≤ g x ∧ g x ≤ 10):
  h f g 2021 = 1 := sorry

end part_a_part_b_part_c_l216_216718


namespace rons_siblings_product_l216_216974

theorem rons_siblings_product
  (H_sisters : ℕ)
  (H_brothers : ℕ)
  (Ha_sisters : ℕ)
  (Ha_brothers : ℕ)
  (R_sisters : ℕ)
  (R_brothers : ℕ)
  (Harry_cond : H_sisters = 4 ∧ H_brothers = 6)
  (Harriet_cond : Ha_sisters = 4 ∧ Ha_brothers = 6)
  (Ron_cond_sisters : R_sisters = Ha_sisters)
  (Ron_cond_brothers : R_brothers = Ha_brothers + 2)
  : R_sisters * R_brothers = 32 := by
  sorry

end rons_siblings_product_l216_216974


namespace six_digit_divisible_by_72_l216_216264

theorem six_digit_divisible_by_72 (n m : ℕ) (h1 : n = 920160 ∨ n = 120168) :
  (∃ (x y : ℕ), 10 * x + y = 2016 ∧ (10^5 * x + n * 10 + m) % 72 = 0) :=
by
  sorry

end six_digit_divisible_by_72_l216_216264


namespace teams_in_each_group_l216_216987

theorem teams_in_each_group (n : ℕ) :
  (2 * (n * (n - 1) / 2) + 3 * n = 56) → n = 7 :=
by
  sorry

end teams_in_each_group_l216_216987


namespace algebraic_expression_value_l216_216938

theorem algebraic_expression_value
  (x y : ℚ)
  (h : |2 * x - 3 * y + 1| + (x + 3 * y + 5)^2 = 0) :
  (-2 * x * y)^2 * (-y^2) * 6 * x * y^2 = 192 :=
  sorry

end algebraic_expression_value_l216_216938


namespace average_price_of_fruit_l216_216476

theorem average_price_of_fruit :
  ∃ (A O : ℕ), A + O = 10 ∧ (40 * A + 60 * (O - 4)) / (A + O - 4) = 50 → 
  (40 * A + 60 * O) / 10 = 54 :=
by
  sorry

end average_price_of_fruit_l216_216476


namespace symmetric_about_y_axis_l216_216137

-- Condition: f is an odd function defined on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given that f is odd and F is defined as specified
theorem symmetric_about_y_axis (f : ℝ → ℝ)
  (hf : odd_function f) :
  ∀ x : ℝ, |f x| + f (|x|) = |f (-x)| + f (|x|) := 
by
  sorry

end symmetric_about_y_axis_l216_216137


namespace zebra_difference_is_zebra_l216_216091

/-- 
A zebra number is a non-negative integer in which the digits strictly alternate between even and odd.
Given two 100-digit zebra numbers, prove that their difference is still a 100-digit zebra number.
-/
theorem zebra_difference_is_zebra 
  (A B : ℕ) 
  (hA : (∀ i, (A / 10^i % 10) % 2 = i % 2) ∧ (A / 10^100 = 0) ∧ (A > 10^99))
  (hB : (∀ i, (B / 10^i % 10) % 2 = i % 2) ∧ (B / 10^100 = 0) ∧ (B > 10^99)) 
  : (∀ j, (((A - B) / 10^j) % 10) % 2 = j % 2) ∧ ((A - B) / 10^100 = 0) ∧ ((A - B) > 10^99) :=
sorry

end zebra_difference_is_zebra_l216_216091


namespace initial_overs_l216_216472

theorem initial_overs (initial_run_rate remaining_run_rate target runs initially remaining_overs : ℝ)
    (h_target : target = 282)
    (h_remaining_overs : remaining_overs = 40)
    (h_initial_run_rate : initial_run_rate = 3.6)
    (h_remaining_run_rate : remaining_run_rate = 6.15)
    (h_target_eq : initial_run_rate * initially + remaining_run_rate * remaining_overs = target) :
    initially = 10 :=
by
  sorry

end initial_overs_l216_216472


namespace range_of_a_for_decreasing_function_l216_216230

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 4 else 3 * a / x

theorem range_of_a_for_decreasing_function :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≥ f a x2) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end range_of_a_for_decreasing_function_l216_216230


namespace find_g_9_l216_216615

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l216_216615


namespace min_staff_members_l216_216225

theorem min_staff_members
  (num_male_students : ℕ)
  (num_benches_3_students : ℕ)
  (num_benches_4_students : ℕ)
  (num_female_students : ℕ)
  (total_students : ℕ)
  (total_seating_capacity : ℕ)
  (additional_seats_required : ℕ)
  (num_staff_members : ℕ)
  (h1 : num_female_students = 4 * num_male_students)
  (h2 : num_male_students = 29)
  (h3 : num_benches_3_students = 15)
  (h4 : num_benches_4_students = 14)
  (h5 : total_seating_capacity = 3 * num_benches_3_students + 4 * num_benches_4_students)
  (h6 : total_students = num_male_students + num_female_students)
  (h7 : additional_seats_required = total_students - total_seating_capacity)
  (h8 : num_staff_members = additional_seats_required)
  : num_staff_members = 44 := 
sorry

end min_staff_members_l216_216225


namespace chord_length_count_l216_216562

noncomputable def number_of_chords (d r : ℕ) : ℕ := sorry

theorem chord_length_count {d r : ℕ} (h1 : d = 12) (h2 : r = 13) :
  number_of_chords d r = 17 :=
sorry

end chord_length_count_l216_216562


namespace john_needs_392_tanks_l216_216873

/- Variables representing the conditions -/
def small_balloons : ℕ := 5000
def medium_balloons : ℕ := 5000
def large_balloons : ℕ := 5000

def small_balloon_volume : ℕ := 20
def medium_balloon_volume : ℕ := 30
def large_balloon_volume : ℕ := 50

def helium_tank_capacity : ℕ := 1000
def hydrogen_tank_capacity : ℕ := 1200
def mixture_tank_capacity : ℕ := 1500

/- Mathematical calculations -/
def helium_volume : ℕ := small_balloons * small_balloon_volume
def hydrogen_volume : ℕ := medium_balloons * medium_balloon_volume
def mixture_volume : ℕ := large_balloons * large_balloon_volume

def helium_tanks : ℕ := (helium_volume + helium_tank_capacity - 1) / helium_tank_capacity
def hydrogen_tanks : ℕ := (hydrogen_volume + hydrogen_tank_capacity - 1) / hydrogen_tank_capacity
def mixture_tanks : ℕ := (mixture_volume + mixture_tank_capacity - 1) / mixture_tank_capacity

def total_tanks : ℕ := helium_tanks + hydrogen_tanks + mixture_tanks

theorem john_needs_392_tanks : total_tanks = 392 :=
by {
  -- calculation proof goes here
  sorry
}

end john_needs_392_tanks_l216_216873


namespace right_triangle_sides_l216_216698

theorem right_triangle_sides (p m : ℝ)
  (hp : 0 < p)
  (hm : 0 < m) :
  ∃ a b c : ℝ, 
    a + b + c = 2 * p ∧
    a^2 + b^2 = c^2 ∧
    (1 / 2) * a * b = m^2 ∧
    c = (p^2 - m^2) / p ∧
    a = (p^2 + m^2 + Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) ∧
    b = (p^2 + m^2 - Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) := 
by
  sorry

end right_triangle_sides_l216_216698


namespace car_speed_first_hour_l216_216837

theorem car_speed_first_hour (x : ℕ) (hx : x = 65) : 
  let speed_second_hour := 45 
  let average_speed := 55
  (x + 45) / 2 = 55 
  :=
  by
  sorry

end car_speed_first_hour_l216_216837


namespace monotonic_increasing_interval_l216_216215

open Real

theorem monotonic_increasing_interval (k : ℤ) : 
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π ↔ 
    ∀ t, -π / 2 + 2 * k * π ≤ 2 * t - π / 3 ∧ 2 * t - π / 3 ≤ π / 2 + 2 * k * π :=
sorry

end monotonic_increasing_interval_l216_216215


namespace roots_of_quadratic_eq_l216_216710

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end roots_of_quadratic_eq_l216_216710


namespace sqrt_720_simplified_l216_216246

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l216_216246


namespace solve_system_of_equations_l216_216550

theorem solve_system_of_equations :
  ∀ x y : ℝ,
  (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ (y - x + 1 = x^2 - 3*x) ∧ (x ≠ 0) ∧ (x ≠ 3) →
  (x, y) = (-1, 2) ∨ (x, y) = (2, -1) ∨ (x, y) = (-2, 7) :=
by
  sorry

end solve_system_of_equations_l216_216550


namespace cobbler_mends_3_pairs_per_hour_l216_216526

def cobbler_hours_per_day_mon_thu := 8
def cobbler_hours_friday := 11 - 8
def cobbler_total_hours_week := 4 * cobbler_hours_per_day_mon_thu + cobbler_hours_friday
def cobbler_pairs_per_week := 105
def cobbler_pairs_per_hour := cobbler_pairs_per_week / cobbler_total_hours_week

theorem cobbler_mends_3_pairs_per_hour : cobbler_pairs_per_hour = 3 := 
by 
  -- Add the steps if necessary but in this scenario, we are skipping proof details
  sorry

end cobbler_mends_3_pairs_per_hour_l216_216526


namespace winner_won_by_324_votes_l216_216058

theorem winner_won_by_324_votes
  (total_votes : ℝ)
  (winner_percentage : ℝ)
  (winner_votes : ℝ)
  (h1 : winner_percentage = 0.62)
  (h2 : winner_votes = 837) :
  (winner_votes - (0.38 * total_votes) = 324) :=
by
  sorry

end winner_won_by_324_votes_l216_216058


namespace value_of_M_l216_216552

theorem value_of_M (M : ℝ) (H : 0.25 * M = 0.55 * 1500) : M = 3300 := 
by
  sorry

end value_of_M_l216_216552


namespace total_cost_computers_l216_216816

theorem total_cost_computers (B T : ℝ) 
  (cA : ℝ := 1.4 * B) 
  (cB : ℝ := B) 
  (tA : ℝ := T) 
  (tB : ℝ := T + 20) 
  (total_cost_A : ℝ := cA * tA)
  (total_cost_B : ℝ := cB * tB):
  total_cost_A = total_cost_B → 70 * B = total_cost_A := 
by
  sorry

end total_cost_computers_l216_216816


namespace impossible_result_l216_216822

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬ (∃ f1 f_1 : ℤ, f1 = a * Real.sin 1 + b + c ∧ f_1 = -a * Real.sin 1 - b + c ∧ (f1 = 1 ∧ f_1 = 2)) :=
by
  sorry

end impossible_result_l216_216822


namespace find_unknown_number_l216_216994

theorem find_unknown_number (x : ℝ) (h : (8 / 100) * x = 96) : x = 1200 :=
by
  sorry

end find_unknown_number_l216_216994


namespace trigonometric_identity_l216_216927

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := 
by 
  sorry

end trigonometric_identity_l216_216927


namespace photograph_goal_reach_l216_216438

-- Define the initial number of photographs
def initial_photos : ℕ := 250

-- Define the percentage splits initially
def beth_pct_init : ℝ := 0.40
def my_pct_init : ℝ := 0.35
def julia_pct_init : ℝ := 0.25

-- Define the photographs taken initially by each person
def beth_photos_init : ℕ := 100
def my_photos_init : ℕ := 88
def julia_photos_init : ℕ := 63

-- Confirm initial photographs sum
example (h : beth_photos_init + my_photos_init + julia_photos_init = 251) : true := 
by trivial

-- Define today's decreased productivity percentages
def beth_decrease_pct : ℝ := 0.35
def my_decrease_pct : ℝ := 0.45
def julia_decrease_pct : ℝ := 0.25

-- Define the photographs taken today by each person after decreases
def beth_photos_today : ℕ := 65
def my_photos_today : ℕ := 48
def julia_photos_today : ℕ := 47

-- Sum of photographs taken today
def total_photos_today : ℕ := 160

-- Define the initial plus today's needed photographs to reach goal
def goal_photos : ℕ := 650

-- Define the additional number of photographs needed
def additional_photos_needed : ℕ := 399 - total_photos_today

-- Final proof statement
theorem photograph_goal_reach : 
  (beth_photos_init + my_photos_init + julia_photos_init) + (beth_photos_today + my_photos_today + julia_photos_today) + additional_photos_needed = goal_photos := 
by sorry

end photograph_goal_reach_l216_216438


namespace borrowed_quarters_l216_216520

def original_quarters : ℕ := 8
def remaining_quarters : ℕ := 5

theorem borrowed_quarters : original_quarters - remaining_quarters = 3 :=
by
  sorry

end borrowed_quarters_l216_216520


namespace third_grade_contribution_fourth_grade_contribution_l216_216396

def first_grade := 20
def second_grade := 45
def third_grade := first_grade + second_grade - 17
def fourth_grade := 2 * third_grade - 36

theorem third_grade_contribution : third_grade = 48 := by
  sorry

theorem fourth_grade_contribution : fourth_grade = 60 := by
  sorry

end third_grade_contribution_fourth_grade_contribution_l216_216396


namespace sum_of_first_five_terms_l216_216086

theorem sum_of_first_five_terms 
  (a₂ a₃ a₄ : ℤ)
  (h1 : a₂ = 4)
  (h2 : a₃ = 7)
  (h3 : a₄ = 10) :
  ∃ a1 a5, a1 + a₂ + a₃ + a₄ + a5 = 35 :=
by
  sorry

end sum_of_first_five_terms_l216_216086


namespace prove_side_c_prove_sin_B_prove_area_circumcircle_l216_216429

-- Define the given conditions
def triangle_ABC (a b A : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3

-- Prove that side 'c' is equal to 3
theorem prove_side_c (h : triangle_ABC a b A) : c = 3 := by
  sorry

-- Prove that sin B is equal to \frac{\sqrt{21}}{7}
theorem prove_sin_B (h : triangle_ABC a b A) : Real.sin B = Real.sqrt 21 / 7 := by
  sorry

-- Prove that the area of the circumcircle is \frac{7\pi}{3}
theorem prove_area_circumcircle (h : triangle_ABC a b A) (R : ℝ) : 
  let circumcircle_area := Real.pi * R^2
  circumcircle_area = 7 * Real.pi / 3 := by
  sorry

end prove_side_c_prove_sin_B_prove_area_circumcircle_l216_216429


namespace find_natural_number_l216_216434

theorem find_natural_number (n : ℕ) (h1 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 3 ∨ d = 5 ∨ d = 9 ∨ d = 15)
  (h2 : 1 + 3 + 5 + 9 + 15 + n = 78) : n = 45 := sorry

end find_natural_number_l216_216434


namespace complement_intersection_eq_interval_l216_216558

open Set

noncomputable def M : Set ℝ := {x | 3 * x - 1 >= 0}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 1 / 2}

theorem complement_intersection_eq_interval :
  (M ∩ N)ᶜ = (Iio (1 / 3) ∪ Ici (1 / 2)) :=
by
  -- proof will go here in the actual development
  sorry

end complement_intersection_eq_interval_l216_216558


namespace dealer_cannot_prevent_l216_216515

theorem dealer_cannot_prevent (m n : ℕ) (h : m < 3 * n ∧ n < 3 * m) :
  ∃ (a b : ℕ), (a = 3 * b ∨ b = 3 * a) ∨ (a = 0 ∧ b = 0):=
sorry

end dealer_cannot_prevent_l216_216515


namespace sample_size_l216_216050

-- Definitions for the conditions
def ratio_A : Nat := 2
def ratio_B : Nat := 3
def ratio_C : Nat := 4
def stratified_sample_size : Nat := 9 -- Total parts in the ratio sum
def products_A_sample : Nat := 18 -- Sample contains 18 Type A products

-- We need to tie these conditions together and prove the size of the sample n
theorem sample_size (n : Nat) (ratio_A ratio_B ratio_C stratified_sample_size products_A_sample : Nat) :
  ratio_A = 2 → ratio_B = 3 → ratio_C = 4 → stratified_sample_size = 9 → products_A_sample = 18 → n = 81 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof body here
  sorry -- Placeholder for the proof

end sample_size_l216_216050


namespace girls_in_class_l216_216148

theorem girls_in_class (k : ℕ) (n_girls n_boys total_students : ℕ)
  (h1 : n_girls = 3 * k) (h2 : n_boys = 4 * k) (h3 : total_students = 35) 
  (h4 : n_girls + n_boys = total_students) : 
  n_girls = 15 :=
by
  -- The proof would normally go here, but is omitted per instructions.
  sorry

end girls_in_class_l216_216148


namespace system_solution_l216_216489

theorem system_solution (x y: ℝ) 
  (h1: x + y = 2) 
  (h2: 3 * x + y = 4) : 
  x = 1 ∧ y = 1 :=
sorry

end system_solution_l216_216489


namespace minimum_omega_l216_216262

noncomputable def f (omega phi x : ℝ) : ℝ := Real.sin (omega * x + phi)

theorem minimum_omega {omega : ℝ} (h_pos : omega > 0) (h_even : ∀ x : ℝ, f omega (Real.pi / 2) x = f omega (Real.pi / 2) (-x)) 
  (h_zero_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f omega (Real.pi / 2) x = 0) :
  omega ≥ 1 / 2 :=
sorry

end minimum_omega_l216_216262


namespace coefficient_of_neg2ab_is_neg2_l216_216885

-- Define the term -2ab
def term : ℤ := -2

-- Define the function to get the coefficient from term -2ab
def coefficient (t : ℤ) : ℤ := t

-- The theorem stating the coefficient of -2ab is -2
theorem coefficient_of_neg2ab_is_neg2 : coefficient term = -2 :=
by
  -- Proof can be filled later
  sorry

end coefficient_of_neg2ab_is_neg2_l216_216885


namespace remove_blue_to_get_80_percent_red_l216_216953

-- Definitions from the conditions
def total_balls : ℕ := 150
def red_balls : ℕ := 60
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℤ := 80

-- Lean statement of the proof problem
theorem remove_blue_to_get_80_percent_red :
  ∃ (x : ℕ), (x ≤ initial_blue_balls) ∧ (red_balls * 100 = desired_percentage_red * (total_balls - x)) → x = 75 := sorry

end remove_blue_to_get_80_percent_red_l216_216953


namespace find_x_l216_216800

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : y = 25) : x = 50 :=
by
  sorry

end find_x_l216_216800


namespace part1_part2_l216_216459

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
  sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → h a x ≤ if a ≥ 0 then 3*a + 3 else if -3 ≤ a then a + 3 else 0) :=
  sorry

end part1_part2_l216_216459


namespace age_of_15th_student_l216_216015

theorem age_of_15th_student : 
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  total_age_all_students - (total_age_first_group + total_age_second_group) = 16 :=
by
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  sorry

end age_of_15th_student_l216_216015


namespace inequality_solution_sets_min_value_exists_l216_216581

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- Existence of roots at -1 and n
def roots_of_quadratic (m : ℝ) (n : ℝ) : Prop :=
  m * (-1)^2 - 2 * (-1) - 3 = 0 ∧ m * n^2 - 2 * n - 3 = 0 ∧ m > 0

-- Main problem statements
theorem inequality_solution_sets (a : ℝ) (m : ℝ) (n : ℝ)
  (h1 : roots_of_quadratic m n) (h2 : m = 1) (h3 : n = 3) (h4 : a > 0) :
  if 0 < a ∧ a ≤ 1 then 
    ∀ x : ℝ, x > 2 / a ∨ x < 2
  else if 1 < a ∧ a < 2 then
    ∀ x : ℝ, x > 2 ∨ x < 2 / a
  else 
    False :=
sorry

theorem min_value_exists (a : ℝ) (m : ℝ)
  (h1 : 0 < a ∧ a < 1) (h2 : m = 1) (h3 : f (a^2) m - 3*a^3 = -5) :
  a = (Real.sqrt 5 - 1) / 2 :=
sorry

end inequality_solution_sets_min_value_exists_l216_216581


namespace distribute_diamonds_among_two_safes_l216_216872

theorem distribute_diamonds_among_two_safes (N : ℕ) :
  ∀ banker : ℕ, banker < 777 → ∃ s1 s2 : ℕ, s1 ≠ s2 ∧ s1 + s2 = N := sorry

end distribute_diamonds_among_two_safes_l216_216872


namespace part1_part2_l216_216599

open Set

namespace ProofProblem

variable (m : ℝ)

def A (m : ℝ) := {x : ℝ | 0 < x - m ∧ x - m < 3}
def B := {x : ℝ | x ≤ 0 ∨ x ≥ 3}

theorem part1 : (A 1 ∩ B) = {x : ℝ | 3 ≤ x ∧ x < 4} := by
  sorry

theorem part2 : (∀ m, (A m ∪ B) = B ↔ (m ≥ 3 ∨ m ≤ -3)) := by
  sorry

end ProofProblem

end part1_part2_l216_216599


namespace time_to_fill_pool_l216_216340

theorem time_to_fill_pool (V : ℕ) (n : ℕ) (r : ℕ) (fill_rate_per_hour : ℕ) :
  V = 24000 → 
  n = 4 →
  r = 25 → -- 2.5 gallons per minute expressed as 25/10 gallons
  fill_rate_per_hour = (n * r * 6) → -- since 6 * 10 = 60 (to convert per minute rate to per hour, we divide so r is 25 instead of 2.5)
  V / fill_rate_per_hour = 40 :=
by
  sorry

end time_to_fill_pool_l216_216340


namespace translation_coordinates_l216_216100

variable (A B A1 B1 : ℝ × ℝ)

theorem translation_coordinates
  (hA : A = (-1, 0))
  (hB : B = (1, 2))
  (hA1 : A1 = (2, -1))
  (translation_A : A1 = (A.1 + 3, A.2 - 1))
  (translation_B : B1 = (B.1 + 3, B.2 - 1)) :
  B1 = (4, 1) :=
sorry

end translation_coordinates_l216_216100


namespace friends_received_pebbles_l216_216442

-- Define the conditions as expressions
def total_weight_kg : ℕ := 36
def weight_per_pebble_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Convert the total weight from kilograms to grams
def total_weight_g : ℕ := total_weight_kg * 1000

-- Calculate the total number of pebbles
def total_pebbles : ℕ := total_weight_g / weight_per_pebble_g

-- Calculate the total number of friends who received pebbles
def number_of_friends : ℕ := total_pebbles / pebbles_per_friend

-- The theorem to prove the number of friends
theorem friends_received_pebbles : number_of_friends = 36 := by
  sorry

end friends_received_pebbles_l216_216442


namespace base_8_sum_units_digit_l216_216206

section
  def digit_in_base (n : ℕ) (base : ℕ) (d : ℕ) : Prop :=
  ((n % base) = d)

theorem base_8_sum_units_digit :
  let n1 := 63
  let n2 := 74
  let base := 8
  (digit_in_base n1 base 3) →
  (digit_in_base n2 base 4) →
  digit_in_base (n1 + n2) base 7 :=
by
  intro h1 h2
  -- placeholder for the detailed proof
  sorry
end

end base_8_sum_units_digit_l216_216206


namespace average_age_of_5_l216_216944

theorem average_age_of_5 (h1 : 19 * 15 = 285) (h2 : 9 * 16 = 144) (h3 : 15 = 71) :
    (285 - 144 - 71) / 5 = 14 :=
sorry

end average_age_of_5_l216_216944


namespace rectangular_prism_height_eq_17_l216_216870

-- Defining the lengths of the edges of the cubes and rectangular prism
def side_length_cube1 := 10
def edges_cube := 12
def length_rect_prism := 8
def width_rect_prism := 5

-- The total length of the wire used for each shape must be equal
def wire_length_cube1 := edges_cube * side_length_cube1
def wire_length_rect_prism (h : ℕ) := 4 * length_rect_prism + 4 * width_rect_prism + 4 * h

theorem rectangular_prism_height_eq_17 (h : ℕ) :
  wire_length_cube1 = wire_length_rect_prism h → h = 17 := 
by
  -- The proof goes here
  sorry

end rectangular_prism_height_eq_17_l216_216870


namespace rhombus_diagonal_sum_l216_216645

theorem rhombus_diagonal_sum
  (d1 d2 : ℝ)
  (h1 : d1 ≤ 6)
  (h2 : 6 ≤ d2)
  (side_len : ℝ)
  (h_side : side_len = 5)
  (rhombus_relation : d1^2 + d2^2 = 4 * side_len^2) :
  d1 + d2 ≤ 14 :=
sorry

end rhombus_diagonal_sum_l216_216645


namespace find_x_values_l216_216004

def f (x : ℝ) : ℝ := 3 * x^2 - 8

noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Placeholder for the inverse function

theorem find_x_values:
  ∃ x : ℝ, (f x = f_inv x) ↔ (x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6) := sorry

end find_x_values_l216_216004


namespace nursing_home_received_boxes_l216_216247

-- Each condition will be a definition in Lean 4.
def vitamins := 472
def supplements := 288
def total_boxes := 760

-- Statement of the proof problem in Lean
theorem nursing_home_received_boxes : vitamins + supplements = total_boxes := by
  sorry

end nursing_home_received_boxes_l216_216247


namespace no_integer_roots_l216_216454
open Polynomial

theorem no_integer_roots (p : Polynomial ℤ) (c1 c2 c3 : ℤ) (h1 : p.eval c1 = 1) (h2 : p.eval c2 = 1) (h3 : p.eval c3 = 1) (h_distinct : c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3) : ¬ ∃ a : ℤ, p.eval a = 0 :=
by
  sorry

end no_integer_roots_l216_216454


namespace num_red_balls_l216_216715

theorem num_red_balls (x : ℕ) (h1 : 60 = 60) (h2 : (x : ℝ) / (x + 60) = 0.25) : x = 20 :=
sorry

end num_red_balls_l216_216715


namespace vector_sum_length_l216_216826

open Real

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vector_angle_cosine (v w : ℝ × ℝ) : ℝ :=
dot_product v w / (vector_length v * vector_length w)

theorem vector_sum_length (a b : ℝ × ℝ)
  (ha : vector_length a = 2)
  (hb : vector_length b = 2)
  (hab_angle : vector_angle_cosine a b = cos (π / 3)):
  vector_length (a.1 + b.1, a.2 + b.2) = 2 * sqrt 3 :=
by sorry

end vector_sum_length_l216_216826


namespace dan_took_pencils_l216_216272

theorem dan_took_pencils (initial_pencils remaining_pencils : ℕ) (h_initial : initial_pencils = 34) (h_remaining : remaining_pencils = 12) : (initial_pencils - remaining_pencils) = 22 := 
by
  sorry

end dan_took_pencils_l216_216272


namespace midpoint_of_line_segment_on_hyperbola_l216_216101

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l216_216101


namespace sum_of_squares_divisible_by_three_l216_216657

theorem sum_of_squares_divisible_by_three {a b : ℤ} 
  (h : 3 ∣ (a^2 + b^2)) : (3 ∣ a ∧ 3 ∣ b) :=
by 
  sorry

end sum_of_squares_divisible_by_three_l216_216657


namespace heptagon_divisibility_impossible_l216_216064

theorem heptagon_divisibility_impossible (a b c d e f g : ℕ) :
  (b ∣ a ∨ a ∣ b) ∧ (c ∣ b ∨ b ∣ c) ∧ (d ∣ c ∨ c ∣ d) ∧ (e ∣ d ∨ d ∣ e) ∧
  (f ∣ e ∨ e ∣ f) ∧ (g ∣ f ∨ f ∣ g) ∧ (a ∣ g ∨ g ∣ a) →
  ¬((a ∣ c ∨ c ∣ a) ∧ (a ∣ d ∨ d ∣ a) ∧ (a ∣ e ∨ e ∣ a) ∧ (a ∣ f ∨ f ∣ a) ∧
    (a ∣ g ∨ g ∣ a) ∧ (b ∣ d ∨ d ∣ b) ∧ (b ∣ e ∨ e ∣ b) ∧ (b ∣ f ∨ f ∣ b) ∧
    (b ∣ g ∨ g ∣ b) ∧ (c ∣ e ∨ e ∣ c) ∧ (c ∣ f ∨ f ∣ c) ∧ (c ∣ g ∨ g ∣ c) ∧
    (d ∣ f ∨ f ∣ d) ∧ (d ∣ g ∨ g ∣ d) ∧ (e ∣ g ∨ g ∣ e)) :=
 by
  sorry

end heptagon_divisibility_impossible_l216_216064


namespace total_selling_price_is_correct_l216_216536

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

def discount : ℝ := discount_rate * original_price
def sale_price : ℝ := original_price - discount
def tax : ℝ := tax_rate * sale_price
def total_selling_price : ℝ := sale_price + tax

theorem total_selling_price_is_correct : total_selling_price = 96.6 := by
  sorry

end total_selling_price_is_correct_l216_216536


namespace final_score_correct_l216_216208

def innovation_score : ℕ := 88
def comprehensive_score : ℕ := 80
def language_score : ℕ := 75

def weight_innovation : ℕ := 5
def weight_comprehensive : ℕ := 3
def weight_language : ℕ := 2

def final_score : ℕ :=
  (innovation_score * weight_innovation + comprehensive_score * weight_comprehensive +
   language_score * weight_language) /
  (weight_innovation + weight_comprehensive + weight_language)

theorem final_score_correct :
  final_score = 83 :=
by
  -- proof goes here
  sorry

end final_score_correct_l216_216208


namespace tom_total_amount_after_saving_l216_216931

theorem tom_total_amount_after_saving :
  let hourly_rate := 6.50
  let work_hours := 31
  let saving_rate := 0.10
  let total_earnings := hourly_rate * work_hours
  let amount_set_aside := total_earnings * saving_rate
  let amount_for_purchases := total_earnings - amount_set_aside
  amount_for_purchases = 181.35 :=
by
  sorry

end tom_total_amount_after_saving_l216_216931


namespace cows_gift_by_friend_l216_216202

-- Define the base conditions
def initial_cows : Nat := 39
def cows_died : Nat := 25
def cows_sold : Nat := 6
def cows_increase : Nat := 24
def cows_bought : Nat := 43
def final_cows : Nat := 83

-- Define the computation to get the number of cows after each event
def cows_after_died : Nat := initial_cows - cows_died
def cows_after_sold : Nat := cows_after_died - cows_sold
def cows_after_increase : Nat := cows_after_sold + cows_increase
def cows_after_bought : Nat := cows_after_increase + cows_bought

-- Define the proof problem
theorem cows_gift_by_friend : (final_cows - cows_after_bought) = 8 := by
  sorry

end cows_gift_by_friend_l216_216202


namespace side_length_of_square_l216_216906

-- Define the conditions
def area_rectangle (length width : ℝ) : ℝ := length * width
def area_square (side : ℝ) : ℝ := side * side

-- Given conditions
def rect_length : ℝ := 2
def rect_width : ℝ := 8
def area_of_rectangle : ℝ := area_rectangle rect_length rect_width
def area_of_square : ℝ := area_of_rectangle

-- Main statement to prove
theorem side_length_of_square : ∃ (s : ℝ), s^2 = 16 ∧ s = 4 :=
by {
  -- use the conditions here
  sorry
}

end side_length_of_square_l216_216906


namespace sum_of_roots_eq_9_div_4_l216_216905

-- Define the values for the coefficients
def a : ℝ := -48
def b : ℝ := 108
def c : ℝ := -27

-- Define the quadratic equation and the function that represents the sum of the roots
def quadratic_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement of the problem: Prove the sum of the roots of the quadratic equation equals 9/4
theorem sum_of_roots_eq_9_div_4 : 
  (∀ x y : ℝ, quadratic_eq x = 0 → quadratic_eq y = 0 → x ≠ y → x + y = - (b/a)) → - (b / a) = 9 / 4 :=
by
  sorry

end sum_of_roots_eq_9_div_4_l216_216905


namespace charlie_paints_60_sqft_l216_216082

theorem charlie_paints_60_sqft (A B C : ℕ) (total_sqft : ℕ) (h_ratio : A = 3 ∧ B = 5 ∧ C = 2) (h_total : total_sqft = 300) : 
  C * (total_sqft / (A + B + C)) = 60 :=
by
  rcases h_ratio with ⟨rfl, rfl, rfl⟩
  rcases h_total with rfl
  sorry

end charlie_paints_60_sqft_l216_216082


namespace dragon_cake_votes_l216_216711

theorem dragon_cake_votes (W U D : ℕ) (x : ℕ) 
  (hW : W = 7) 
  (hU : U = 3 * W) 
  (hD : D = W + x) 
  (hTotal : W + U + D = 60) 
  (hx : x = D - W) : 
  x = 25 := 
by
  sorry

end dragon_cake_votes_l216_216711


namespace complement_of_M_l216_216377

open Set

def U : Set ℝ := univ
def M : Set ℝ := { x | x^2 - 2 * x > 0 }
def comp_M_Real := compl M

theorem complement_of_M :
  comp_M_Real = { x : ℝ | 0 ≤ x ∧ x ≤ 2 } :=
sorry

end complement_of_M_l216_216377


namespace train_pass_jogger_in_36_sec_l216_216901

noncomputable def time_to_pass_jogger (speed_jogger speed_train : ℝ) (lead_jogger len_train : ℝ) : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := lead_jogger + len_train
  total_distance / relative_speed

theorem train_pass_jogger_in_36_sec :
  time_to_pass_jogger 9 45 240 120 = 36 := by
  sorry

end train_pass_jogger_in_36_sec_l216_216901


namespace smallest_whole_number_for_inequality_l216_216470

theorem smallest_whole_number_for_inequality:
  ∃ (x : ℕ), (2 : ℝ) / 5 + (x : ℝ) / 9 > 1 ∧ ∀ (y : ℕ), (2 : ℝ) / 5 + (y : ℝ) / 9 > 1 → x ≤ y :=
by
  sorry

end smallest_whole_number_for_inequality_l216_216470


namespace exists_three_sticks_form_triangle_l216_216258

theorem exists_three_sticks_form_triangle 
  (l : Fin 5 → ℝ) 
  (h1 : ∀ i, 2 < l i) 
  (h2 : ∀ i, l i < 8) : 
  ∃ (i j k : Fin 5), i < j ∧ j < k ∧ 
    (l i + l j > l k) ∧ 
    (l j + l k > l i) ∧ 
    (l k + l i > l j) :=
sorry

end exists_three_sticks_form_triangle_l216_216258


namespace can_combine_with_sqrt2_l216_216874

theorem can_combine_with_sqrt2 :
  (∃ (x : ℝ), x = 2 * Real.sqrt 6 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧ ∃ (y : ℝ), y = Real.sqrt 2) :=
sorry

end can_combine_with_sqrt2_l216_216874


namespace largest_integer_n_apples_l216_216379

theorem largest_integer_n_apples (t : ℕ) (a : ℕ → ℕ) (h1 : t = 150) 
    (h2 : ∀ i : ℕ, 100 ≤ a i ∧ a i ≤ 130) :
  ∃ n : ℕ, n = 5 ∧ (∀ i j : ℕ, a i = a j → i = j → 5 ≤ i ∧ 5 ≤ j) :=
by
  sorry

end largest_integer_n_apples_l216_216379


namespace valid_permutations_remainder_l216_216578

def countValidPermutations : Nat :=
  let total := (Finset.range 3).sum (fun j =>
    Nat.choose 3 (j + 2) * Nat.choose 5 j * Nat.choose 7 (j + 3))
  total % 1000

theorem valid_permutations_remainder :
  countValidPermutations = 60 := 
  sorry

end valid_permutations_remainder_l216_216578


namespace sum_modulo_seven_l216_216296

theorem sum_modulo_seven (a b c : ℕ) (h1: a = 9^5) (h2: b = 8^6) (h3: c = 7^7) :
  (a + b + c) % 7 = 5 :=
by sorry

end sum_modulo_seven_l216_216296


namespace kelsey_video_count_l216_216971

variable (E U K : ℕ)

noncomputable def total_videos : ℕ := 411
noncomputable def ekon_videos : ℕ := E
noncomputable def uma_videos : ℕ := E + 17
noncomputable def kelsey_videos : ℕ := E + 43

theorem kelsey_video_count (E U K : ℕ) 
  (h1 : total_videos = ekon_videos + uma_videos + kelsey_videos)
  (h2 : uma_videos = ekon_videos + 17)
  (h3 : kelsey_videos = ekon_videos + 43)
  : kelsey_videos = 160 := 
sorry

end kelsey_video_count_l216_216971


namespace uncovered_side_length_l216_216284

theorem uncovered_side_length :
  ∃ (L : ℝ) (W : ℝ), L * W = 680 ∧ 2 * W + L = 146 ∧ L = 136 := by
  sorry

end uncovered_side_length_l216_216284


namespace integer_solutions_of_equation_l216_216534

theorem integer_solutions_of_equation :
  ∀ (x y : ℤ), (x^4 + y^4 = 3 * x^3 * y) → (x = 0 ∧ y = 0) := by
  intros x y h
  sorry

end integer_solutions_of_equation_l216_216534


namespace simplify_expression_l216_216099

theorem simplify_expression (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2) ^ 2) + Real.sqrt ((a - 8) ^ 2) = 6 :=
by
  sorry

end simplify_expression_l216_216099


namespace x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l216_216755

theorem x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero :
  ∃ (x : ℝ), (x = 1) → (x^2 + x - 2 = 0) ∧ (¬ (∀ (y : ℝ), y^2 + y - 2 = 0 → y = 1)) := by
  sorry

end x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l216_216755


namespace fraction_simplifies_l216_216738

def current_age_grant := 25
def current_age_hospital := 40

def age_in_five_years (current_age : Nat) : Nat := current_age + 5

def grant_age_in_5_years := age_in_five_years current_age_grant
def hospital_age_in_5_years := age_in_five_years current_age_hospital

def fraction_of_ages := grant_age_in_5_years / hospital_age_in_5_years

theorem fraction_simplifies : fraction_of_ages = (2 / 3) := by
  sorry

end fraction_simplifies_l216_216738


namespace border_collie_catches_ball_in_32_seconds_l216_216623

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end border_collie_catches_ball_in_32_seconds_l216_216623


namespace B_pow_101_eq_B_l216_216716

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]

-- State the theorem
theorem B_pow_101_eq_B : B^101 = B :=
  sorry

end B_pow_101_eq_B_l216_216716


namespace triangle_sum_correct_l216_216057

def triangle_op (a b c : ℕ) : ℕ :=
  a * b / c

theorem triangle_sum_correct :
  triangle_op 4 8 2 + triangle_op 5 10 5 = 26 :=
by
  sorry

end triangle_sum_correct_l216_216057


namespace second_quadrant_set_l216_216039

-- Define the set P of points in the second quadrant
def P : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 }

-- Statement of the problem: Prove that this definition accurately describes the set of all points in the second quadrant
theorem second_quadrant_set :
  P = { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 } :=
by
  sorry

end second_quadrant_set_l216_216039


namespace base_eight_to_base_ten_l216_216043

theorem base_eight_to_base_ten (n : ℕ) : 
  n = 3 * 8^1 + 1 * 8^0 → n = 25 :=
by
  intro h
  rw [mul_comm 3 (8^1), pow_one, mul_comm 1 (8^0), pow_zero, mul_one] at h
  exact h

end base_eight_to_base_ten_l216_216043


namespace opposite_of_three_l216_216346

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l216_216346


namespace gcd_75_225_l216_216292

theorem gcd_75_225 : Int.gcd 75 225 = 75 :=
by
  sorry

end gcd_75_225_l216_216292


namespace find_range_of_m_l216_216107

variable (x m : ℝ)

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + (4 * m - 3) > 0

def proposition_q (m : ℝ) : Prop := (∀ m > 2, m + 1 / (m - 2) ≥ 4) ∧ (∃ m, m + 1 / (m - 2) = 4)

def range_m : Set ℝ := {m | 1 < m ∧ m ≤ 2} ∪ {m | m ≥ 3}

theorem find_range_of_m
  (h_p : proposition_p m ∨ ¬proposition_p m)
  (h_q : proposition_q m ∨ ¬proposition_q m)
  (h_exclusive : (proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m))
  : m ∈ range_m := sorry

end find_range_of_m_l216_216107


namespace numerical_expression_as_sum_of_squares_l216_216648

theorem numerical_expression_as_sum_of_squares : 
  2 * (2009:ℕ)^2 + 2 * (2010:ℕ)^2 = (4019:ℕ)^2 + (1:ℕ)^2 := 
by
  sorry

end numerical_expression_as_sum_of_squares_l216_216648


namespace possible_slopes_of_line_intersecting_ellipse_l216_216996

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end possible_slopes_of_line_intersecting_ellipse_l216_216996


namespace min_value_sq_distance_l216_216685

theorem min_value_sq_distance {x y : ℝ} (h : x^2 + y^2 - 4 * x + 2 = 0) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x^2 + y^2 - 4 * x + 2 = 0 → x^2 + (y - 2)^2 ≥ m) :=
sorry

end min_value_sq_distance_l216_216685


namespace smallest_m_l216_216596

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 12 * p * p - m * p + 432 = 0) (h_sum : p + q = m / 12) (h_prod : p * q = 36) :
  m = 144 :=
by
  sorry

end smallest_m_l216_216596


namespace length_of_book_l216_216592

theorem length_of_book (A W L : ℕ) (hA : A = 50) (hW : W = 10) (hArea : A = L * W) : L = 5 := 
sorry

end length_of_book_l216_216592


namespace sum_not_prime_l216_216217

-- Definitions based on conditions:
variables {a b c d : ℕ}

-- Conditions:
axiom h_ab_eq_cd : a * b = c * d

-- Statement to prove:
theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) :=
sorry

end sum_not_prime_l216_216217


namespace count_valid_pairs_l216_216160

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 3 ∧ ∀ (m n : ℕ), m > n → n ≥ 4 → (m + n) ≤ 40 → (m - n)^2 = m + n → (m, n) ∈ [(10, 6), (15, 10), (21, 15)] := 
by {
  sorry 
}

end count_valid_pairs_l216_216160


namespace expected_red_hair_americans_l216_216619

theorem expected_red_hair_americans (prob_red_hair : ℝ) (sample_size : ℕ) :
  prob_red_hair = 1 / 6 → sample_size = 300 → (prob_red_hair * sample_size = 50) := by
  intros
  sorry

end expected_red_hair_americans_l216_216619


namespace max_rectangle_area_l216_216186

theorem max_rectangle_area (l w : ℕ) (h : 3 * l + 5 * w ≤ 50) : (l * w ≤ 35) :=
by sorry

end max_rectangle_area_l216_216186


namespace eval_expression_l216_216877

theorem eval_expression :
  6 - 9 * (1 / 2 - 3^3) * 2 = 483 := 
sorry

end eval_expression_l216_216877


namespace no_real_roots_l216_216235

noncomputable def polynomial (p : ℝ) (x : ℝ) : ℝ :=
  x^4 + 4 * p * x^3 + 6 * x^2 + 4 * p * x + 1

theorem no_real_roots (p : ℝ) :
  (p > -Real.sqrt 5 / 2) ∧ (p < Real.sqrt 5 / 2) ↔ ¬(∃ x : ℝ, polynomial p x = 0) := by
  sorry

end no_real_roots_l216_216235


namespace distinct_triangles_in_regular_ngon_l216_216748

theorem distinct_triangles_in_regular_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, t = n * (n-1) * (n-2) / 6 := 
sorry

end distinct_triangles_in_regular_ngon_l216_216748


namespace problem_I_problem_II_problem_III_l216_216587

-- The function f(x)
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1/2) * x^2 - a * Real.log x + b

-- Tangent line at x = 1
def tangent_condition (a : ℝ) (b : ℝ) :=
  1 - a = 3 ∧ f 1 a b = 0

-- Extreme point at x = 1
def extreme_condition (a : ℝ) :=
  1 - a = 0 

-- Monotonicity and minimum m
def inequality_condition (a m : ℝ) :=
  -2 ≤ a ∧ a < 0 ∧ ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 ≤ 2 ∧ 0 < x2 ∧ x2 ≤ 2 → 
  |f x1 a (0 : ℝ) - f x2 a 0| ≤ m * |1 / x1 - 1 / x2|

-- Proof problem 1
theorem problem_I : ∃ (a b : ℝ), tangent_condition a b → a = -2 ∧ b = -0.5 := sorry

-- Proof problem 2
theorem problem_II : ∃ (a : ℝ), extreme_condition a → a = 1 := sorry

-- Proof problem 3
theorem problem_III : ∃ (m : ℝ), inequality_condition (-2 : ℝ) m → m = 12 := sorry

end problem_I_problem_II_problem_III_l216_216587


namespace prime_factorization_2006_expr_l216_216131

theorem prime_factorization_2006_expr :
  let a := 2006
  let b := 669
  let c := 1593
  (a^2 * (b + c) - b^2 * (c + a) + c^2 * (a - b)) =
  2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 :=
by
  let a := 2006
  let b := 669
  let c := 1593
  have h1 : 2262 = b + c := by norm_num
  have h2 : 3599 = c + a := by norm_num
  have h3 : 1337 = a - b := by norm_num
  sorry

end prime_factorization_2006_expr_l216_216131


namespace total_applicants_is_40_l216_216381

def total_applicants (PS GPA_high Not_PS_GPA_low both : ℕ) : ℕ :=
  let PS_or_GPA_high := PS + GPA_high - both 
  PS_or_GPA_high + Not_PS_GPA_low

theorem total_applicants_is_40 :
  total_applicants 15 20 10 5 = 40 :=
by
  sorry

end total_applicants_is_40_l216_216381


namespace xy_pos_iff_div_pos_ab_leq_mean_sq_l216_216975

-- Definition for question 1
theorem xy_pos_iff_div_pos (x y : ℝ) : 
  (x * y > 0) ↔ (x / y > 0) :=
sorry

-- Definition for question 3
theorem ab_leq_mean_sq (a b : ℝ) : 
  a * b ≤ ((a + b) / 2) ^ 2 :=
sorry

end xy_pos_iff_div_pos_ab_leq_mean_sq_l216_216975


namespace unique_prime_triplets_l216_216963

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end unique_prime_triplets_l216_216963


namespace polynomial_sum_is_integer_l216_216081

-- Define the integer polynomial and the integers a and b
variables (f : ℤ[X]) (a b : ℤ)

-- The theorem statement
theorem polynomial_sum_is_integer :
  ∃ c : ℤ, f.eval (a - real.sqrt b) + f.eval (a + real.sqrt b) = c :=
sorry

end polynomial_sum_is_integer_l216_216081


namespace proportional_division_middle_part_l216_216920

theorem proportional_division_middle_part : 
  ∃ x : ℕ, x = 8 ∧ 5 * x = 40 ∧ 3 * x + 5 * x + 7 * x = 120 := 
by
  sorry

end proportional_division_middle_part_l216_216920


namespace largest_inscribed_rectangle_l216_216589

theorem largest_inscribed_rectangle {a b m : ℝ} (h : m ≥ b) :
  ∃ (base height area : ℝ),
    base = a * (b + m) / m ∧ 
    height = (b + m) / 2 ∧ 
    area = a * (b + m)^2 / (2 * m) :=
sorry

end largest_inscribed_rectangle_l216_216589


namespace find_average_of_xyz_l216_216488

variable (x y z k : ℝ)

def system_of_equations : Prop :=
  (2 * x + y - z = 26) ∧
  (x + 2 * y + z = 10) ∧
  (x - y + z = k)

theorem find_average_of_xyz (h : system_of_equations x y z k) : 
  (x + y + z) / 3 = (36 + k) / 6 :=
by sorry

end find_average_of_xyz_l216_216488


namespace increase_percentage_when_selfcheckout_broken_l216_216794

-- The problem conditions as variable definitions and declarations
def normal_complaints : ℕ := 120
def short_staffed_increase : ℚ := 1 / 3
def short_staffed_complaints : ℕ := normal_complaints + (normal_complaints / 3)
def total_complaints_three_days : ℕ := 576
def days : ℕ := 3
def both_conditions_complaints : ℕ := total_complaints_three_days / days

-- The theorem that we need to prove
theorem increase_percentage_when_selfcheckout_broken : 
  (both_conditions_complaints - short_staffed_complaints) * 100 / short_staffed_complaints = 20 := 
by
  -- This line sets up that the conclusion is true
  sorry

end increase_percentage_when_selfcheckout_broken_l216_216794


namespace cos_squared_value_l216_216758

theorem cos_squared_value (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) : 
  Real.cos (π / 3 - x) ^ 2 = 1 / 16 := 
sorry

end cos_squared_value_l216_216758


namespace solution_for_system_l216_216537
open Real

noncomputable def solve_system (a b x y : ℝ) : Prop :=
  (a * x + b * y = 7 ∧ b * x + a * y = 8)

noncomputable def solve_linear (a b m n : ℝ) : Prop :=
  (a * (m + n) + b * (m - n) = 7 ∧ b * (m + n) + a * (m - n) = 8)

theorem solution_for_system (a b : ℝ) : solve_system a b 2 3 → solve_linear a b (5/2) (-1/2) :=
by {
  sorry
}

end solution_for_system_l216_216537


namespace coins_in_bag_l216_216940

theorem coins_in_bag (x : ℝ) (h : x + 0.5 * x + 0.25 * x = 140) : x = 80 :=
by sorry

end coins_in_bag_l216_216940


namespace tom_seashells_l216_216224

theorem tom_seashells 
  (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) (h3 : total_seashells = days_at_beach * seashells_per_day) : 
  total_seashells = 35 := 
by
  rw [h1, h2] at h3 
  exact h3

end tom_seashells_l216_216224


namespace probability_two_yellow_apples_l216_216221

theorem probability_two_yellow_apples (total_apples : ℕ) (red_apples : ℕ) (green_apples : ℕ) (yellow_apples : ℕ) (choose : ℕ → ℕ → ℕ) (probability : ℕ → ℕ → ℝ) :
  total_apples = 10 →
  red_apples = 5 →
  green_apples = 3 →
  yellow_apples = 2 →
  choose total_apples 2 = 45 →
  choose yellow_apples 2 = 1 →
  probability (choose yellow_apples 2) (choose total_apples 2) = 1 / 45 := 
  by
  sorry

end probability_two_yellow_apples_l216_216221


namespace explicit_formula_for_f_l216_216856

theorem explicit_formula_for_f (f : ℕ → ℕ) (h₀ : f 0 = 0)
  (h₁ : ∀ (n : ℕ), n % 6 = 0 ∨ n % 6 = 1 → f (n + 1) = f n + 3)
  (h₂ : ∀ (n : ℕ), n % 6 = 2 ∨ n % 6 = 5 → f (n + 1) = f n + 1)
  (h₃ : ∀ (n : ℕ), n % 6 = 3 ∨ n % 6 = 4 → f (n + 1) = f n + 2)
  (n : ℕ) : f (6 * n) = 12 * n :=
by
  sorry

end explicit_formula_for_f_l216_216856


namespace positive_integer_divisibility_l216_216532

theorem positive_integer_divisibility :
  ∀ n : ℕ, 0 < n → (5^(n-1) + 3^(n-1) ∣ 5^n + 3^n) → n = 1 :=
by
  sorry

end positive_integer_divisibility_l216_216532


namespace replace_square_l216_216655

theorem replace_square (x : ℝ) (h : 10.0003 * x = 10000.3) : x = 1000 :=
sorry

end replace_square_l216_216655


namespace hexagon_diagonals_l216_216363

theorem hexagon_diagonals : (6 * (6 - 3)) / 2 = 9 := 
by 
  sorry

end hexagon_diagonals_l216_216363


namespace sum_xyz_eq_two_l216_216025

-- Define the variables x, y, and z to be real numbers
variables (x y z : ℝ)

-- Given condition
def condition : Prop :=
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0

-- The theorem to prove
theorem sum_xyz_eq_two (h : condition x y z) : x + y + z = 2 :=
sorry

end sum_xyz_eq_two_l216_216025


namespace symmetric_line_eq_l216_216781

theorem symmetric_line_eq (x y : ℝ) (h₁ : y = 3 * x + 4) : y = x → y = (1 / 3) * x - (4 / 3) :=
by
  sorry

end symmetric_line_eq_l216_216781


namespace arc_length_sector_l216_216460

theorem arc_length_sector (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 3) : 
  α * r = 2 * π / 3 := 
by 
  sorry

end arc_length_sector_l216_216460


namespace factor_problem_l216_216496

theorem factor_problem 
  (a b : ℕ) (h1 : a > b)
  (h2 : (∀ x, x^2 - 16 * x + 64 = (x - a) * (x - b))) 
  : 3 * b - a = 16 := by
  sorry

end factor_problem_l216_216496


namespace max_value_5x_minus_25x_l216_216765

noncomputable def max_value_of_expression : ℝ :=
  (1 / 4 : ℝ)

theorem max_value_5x_minus_25x :
  ∃ x : ℝ, ∀ y : ℝ, y = 5^x → (5^y - 25^y) ≤ max_value_of_expression :=
sorry

end max_value_5x_minus_25x_l216_216765


namespace six_digit_number_under_5_lakh_with_digit_sum_43_l216_216326

def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000
def under_500000 (n : ℕ) : Prop := n < 500000
def digit_sum (n : ℕ) : ℕ := (n / 100000) + (n / 10000 % 10) + (n / 1000 % 10) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem six_digit_number_under_5_lakh_with_digit_sum_43 :
  is_6_digit 499993 ∧ under_500000 499993 ∧ digit_sum 499993 = 43 :=
by 
  sorry

end six_digit_number_under_5_lakh_with_digit_sum_43_l216_216326


namespace solve_inequality_l216_216966

theorem solve_inequality (x : ℝ) : 2 - x < 1 → x > 1 := 
by
  sorry

end solve_inequality_l216_216966


namespace find_x_l216_216729

theorem find_x (x : ℝ) : 0.20 * x - (1 / 3) * (0.20 * x) = 24 → x = 180 :=
by
  intro h
  sorry

end find_x_l216_216729


namespace problem_l216_216018

theorem problem (a b : ℤ)
  (h1 : -2022 = -a)
  (h2 : -1 = -b) :
  a + b = 2023 :=
sorry

end problem_l216_216018


namespace minimum_students_using_both_l216_216631

theorem minimum_students_using_both (n L T x : ℕ) 
  (H1: 3 * n = 7 * L) 
  (H2: 5 * n = 6 * T) 
  (H3: n = 42) 
  (H4: n = L + T - x) : 
  x = 11 := 
by 
  sorry

end minimum_students_using_both_l216_216631


namespace max_largest_integer_l216_216397

theorem max_largest_integer (A B C D E : ℕ) 
  (h1 : A ≤ B) 
  (h2 : B ≤ C) 
  (h3 : C ≤ D) 
  (h4 : D ≤ E)
  (h5 : (A + B + C + D + E) / 5 = 60) 
  (h6 : E - A = 10) : 
  E ≤ 290 :=
sorry

end max_largest_integer_l216_216397


namespace range_of_function_l216_216409

theorem range_of_function (x : ℝ) : x ≠ 2 ↔ ∃ y, y = x / (x - 2) :=
sorry

end range_of_function_l216_216409


namespace sqrt_of_square_neg_five_eq_five_l216_216085

theorem sqrt_of_square_neg_five_eq_five :
  Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end sqrt_of_square_neg_five_eq_five_l216_216085


namespace possible_values_of_k_l216_216462

-- Definition of the proposition
def proposition (k : ℝ) : Prop :=
  ∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0

-- The main statement to prove in Lean 4
theorem possible_values_of_k (k : ℝ) : ¬ proposition k ↔ (k = 1 ∨ (1 < k ∧ k < 7)) :=
by 
  sorry

end possible_values_of_k_l216_216462


namespace find_geometric_sequence_element_l216_216095

theorem find_geometric_sequence_element (a b c d e : ℕ) (r : ℚ)
  (h1 : 2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100)
  (h2 : Nat.gcd a e = 1)
  (h3 : r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4)
  : c = 36 :=
  sorry

end find_geometric_sequence_element_l216_216095


namespace find_w_l216_216752

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end find_w_l216_216752


namespace trig_identity_l216_216393

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end trig_identity_l216_216393


namespace sum_of_x_and_y_l216_216717

theorem sum_of_x_and_y (x y : ℝ) (h : (x + y + 2)^2 + |2 * x - 3 * y - 1| = 0) : x + y = -2 :=
by
  sorry

end sum_of_x_and_y_l216_216717


namespace ninth_term_is_83_l216_216719

-- Definitions based on conditions
def a : ℕ := 3
def d : ℕ := 10
def arith_sequence (n : ℕ) : ℕ := a + n * d

-- Theorem to prove the 9th term is 83
theorem ninth_term_is_83 : arith_sequence 8 = 83 :=
by
  sorry

end ninth_term_is_83_l216_216719


namespace sequence_inequality_l216_216672

theorem sequence_inequality
  (n : ℕ) (h1 : 1 < n)
  (a : ℕ → ℕ)
  (h2 : ∀ i, i < n → a i < a (i + 1))
  (h3 : ∀ i, i < n - 1 → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) :
  a (n - 1) ≥ 2 * n ^ 2 - 1 :=
sorry

end sequence_inequality_l216_216672


namespace green_chips_count_l216_216068

def total_chips : ℕ := 60
def fraction_blue_chips : ℚ := 1 / 6
def num_red_chips : ℕ := 34

theorem green_chips_count :
  let num_blue_chips := total_chips * fraction_blue_chips
  let chips_not_green := num_blue_chips + num_red_chips
  let num_green_chips := total_chips - chips_not_green
  num_green_chips = 16 := by
    let num_blue_chips := total_chips * fraction_blue_chips
    let chips_not_green := num_blue_chips + num_red_chips
    let num_green_chips := total_chips - chips_not_green
    show num_green_chips = 16
    sorry

end green_chips_count_l216_216068


namespace single_burger_cost_l216_216232

-- Conditions
def total_cost : ℝ := 74.50
def total_burgers : ℕ := 50
def cost_double_burger : ℝ := 1.50
def double_burgers : ℕ := 49

-- Derived information
def cost_single_burger : ℝ := total_cost - (double_burgers * cost_double_burger)

-- Theorem: Prove the cost of a single burger
theorem single_burger_cost : cost_single_burger = 1.00 :=
by
  -- Proof goes here
  sorry

end single_burger_cost_l216_216232


namespace minimum_value_expression_l216_216626

-- Define the conditions in the problem
variable (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
variable (h3 : 2 * m + 2 * n = 2)

-- State the theorem proving the minimum value of the given expression
theorem minimum_value_expression : (1 / m + 2 / n) = 3 + 2 * Real.sqrt 2 := by
  sorry

end minimum_value_expression_l216_216626


namespace find_n_l216_216880

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 100 * n ≡ 85 [MOD 103]) : n = 6 := 
sorry

end find_n_l216_216880


namespace minimum_balls_ensure_20_single_color_l216_216029

def num_balls_to_guarantee_color (r g y b w k : ℕ) : ℕ :=
  let max_without_20 := 19 + 19 + 19 + 18 + 15 + 12
  max_without_20 + 1

theorem minimum_balls_ensure_20_single_color :
  num_balls_to_guarantee_color 30 25 25 18 15 12 = 103 := by
  sorry

end minimum_balls_ensure_20_single_color_l216_216029


namespace donuts_purchased_l216_216087

/-- John goes to a bakery every day for a four-day workweek and chooses between a 
    60-cent croissant or a 90-cent donut. At the end of the week, he spent a whole 
    number of dollars. Prove that he must have purchased 2 donuts. -/
theorem donuts_purchased (d c : ℕ) (h1 : d + c = 4) (h2 : 90 * d + 60 * c % 100 = 0) : d = 2 :=
sorry

end donuts_purchased_l216_216087


namespace greatest_possible_length_l216_216566

theorem greatest_possible_length (a b c : ℕ) (h1 : a = 28) (h2 : b = 45) (h3 : c = 63) : 
  Nat.gcd (Nat.gcd a b) c = 7 :=
by
  sorry

end greatest_possible_length_l216_216566


namespace trajectory_equation_l216_216890

theorem trajectory_equation (x y : ℝ) : x^2 + y^2 = 2 * |x| + 2 * |y| → x^2 + y^2 = 2 * |x| + 2 * |y| :=
by
  sorry

end trajectory_equation_l216_216890


namespace real_solutions_infinite_l216_216926

theorem real_solutions_infinite : 
  ∃ (S : Set ℝ), (∀ x ∈ S, - (x^2 - 4) ≥ 0) ∧ S.Infinite :=
sorry

end real_solutions_infinite_l216_216926


namespace diagonal_length_count_l216_216921

theorem diagonal_length_count :
  ∃ (x : ℕ) (h : (3 < x ∧ x < 22)), x = 18 := by
    sorry

end diagonal_length_count_l216_216921


namespace factorize_l216_216924

theorem factorize (a : ℝ) : 5*a^3 - 125*a = 5*a*(a + 5)*(a - 5) :=
sorry

end factorize_l216_216924


namespace triangle_area_is_correct_l216_216012

-- Defining the vertices of the triangle
def vertexA : ℝ × ℝ := (0, 0)
def vertexB : ℝ × ℝ := (0, 6)
def vertexC : ℝ × ℝ := (8, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Statement to prove
theorem triangle_area_is_correct : triangle_area vertexA vertexB vertexC = 24.0 := 
by
  sorry

end triangle_area_is_correct_l216_216012


namespace sandwich_cost_l216_216267

theorem sandwich_cost 
  (loaf_sandwiches : ℕ) (target_sandwiches : ℕ) 
  (bread_cost : ℝ) (meat_cost : ℝ) (cheese_cost : ℝ) 
  (cheese_coupon : ℝ) (meat_coupon : ℝ) (total_threshold : ℝ) 
  (discount_rate : ℝ)
  (h1 : loaf_sandwiches = 10) 
  (h2 : target_sandwiches = 50) 
  (h3 : bread_cost = 4) 
  (h4 : meat_cost = 5) 
  (h5 : cheese_cost = 4) 
  (h6 : cheese_coupon = 1) 
  (h7 : meat_coupon = 1) 
  (h8 : total_threshold = 60) 
  (h9 : discount_rate = 0.1) :
  ( ∃ cost_per_sandwich : ℝ, 
      cost_per_sandwich = 1.944 ) :=
  sorry

end sandwich_cost_l216_216267


namespace wechat_balance_l216_216503

def transaction1 : ℤ := 48
def transaction2 : ℤ := -30
def transaction3 : ℤ := -50

theorem wechat_balance :
  transaction1 + transaction2 + transaction3 = -32 :=
by
  -- placeholder for proof
  sorry

end wechat_balance_l216_216503


namespace negation_exists_l216_216691

theorem negation_exists (x : ℝ) (h : x ≥ 0) : (¬ (∀ x : ℝ, (x ≥ 0) → (2^x > x^2))) ↔ (∃ x₀ : ℝ, (x₀ ≥ 0) ∧ (2 ^ x₀ ≤ x₀^2)) := by
  sorry

end negation_exists_l216_216691


namespace number_of_children_l216_216513

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l216_216513


namespace cross_ratio_eq_one_implies_equal_points_l216_216965

-- Definitions corresponding to the points and hypothesis.
variable {A B C D : ℝ}
variable (h_line : collinear ℝ A B C D) (h_cross_ratio : cross_ratio A B C D = 1)

-- The theorem statement based on the given problem and solution.
theorem cross_ratio_eq_one_implies_equal_points :
  A = B ∨ C = D :=
sorry

end cross_ratio_eq_one_implies_equal_points_l216_216965


namespace simplify_tan_product_l216_216545

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l216_216545


namespace find_angle_C_max_area_l216_216780

-- Define the conditions as hypotheses
variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c = 2 * Real.sqrt 3)
variable (h2 : c * Real.cos B + (b - 2 * a) * Real.cos C = 0)

-- Problem (1): Prove that angle C is π/3
theorem find_angle_C : C = Real.pi / 3 :=
by
  sorry

-- Problem (2): Prove that the maximum area of triangle ABC is 3√3
theorem max_area : (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 :=
by
  sorry

end find_angle_C_max_area_l216_216780


namespace find_certain_number_l216_216783

theorem find_certain_number (x : ℕ) (h : (55 * x) % 8 = 7) : x = 1 := 
sorry

end find_certain_number_l216_216783


namespace parallel_vectors_x_value_l216_216654

theorem parallel_vectors_x_value :
  ∀ (x : ℝ), (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (2, x) → a.1 * b.2 = a.2 * b.1) → x = -4 :=
by
  intros x h
  have h_parallel := h (1, -2) (2, x) rfl rfl
  sorry

end parallel_vectors_x_value_l216_216654


namespace cos_double_angle_l216_216597

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l216_216597


namespace cylinder_water_depth_l216_216048

theorem cylinder_water_depth 
  (height radius : ℝ)
  (h_ge_zero : height ≥ 0)
  (r_ge_zero : radius ≥ 0)
  (total_height : height = 1200)
  (total_radius : radius = 100)
  (above_water_vol : 1 / 3 * π * radius^2 * height = 1 / 3 * π * radius^2 * 1200) :
  height - 800 = 400 :=
by
  -- Use provided constraints and logical reasoning on structures
  sorry

end cylinder_water_depth_l216_216048


namespace simplify_expression_l216_216848

theorem simplify_expression : 2 - 2 / (2 + Real.sqrt 5) + 2 / (2 - Real.sqrt 5) = 2 + 4 * Real.sqrt 5 :=
by sorry

end simplify_expression_l216_216848


namespace bicycle_cost_price_l216_216121

theorem bicycle_cost_price 
  (CP_A : ℝ) 
  (H : CP_A * (1.20 * 0.85 * 1.30 * 0.90) = 285) : 
  CP_A = 285 / (1.20 * 0.85 * 1.30 * 0.90) :=
sorry

end bicycle_cost_price_l216_216121


namespace largest_pot_cost_l216_216754

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ :=
  x + 5 * 0.15

theorem largest_pot_cost :
  ∃ (x : ℝ), (6 * x + 5 * 0.15 + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15) = 8.85) →
    cost_of_largest_pot x = 1.85 :=
by
  sorry

end largest_pot_cost_l216_216754


namespace find_perpendicular_slope_value_l216_216157

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value_l216_216157


namespace hot_dogs_served_today_l216_216836

-- Define the number of hot dogs served during lunch
def h_dogs_lunch : ℕ := 9

-- Define the number of hot dogs served during dinner
def h_dogs_dinner : ℕ := 2

-- Define the total number of hot dogs served today
def total_h_dogs : ℕ := h_dogs_lunch + h_dogs_dinner

-- Theorem stating that the total number of hot dogs served today is 11
theorem hot_dogs_served_today : total_h_dogs = 11 := by
  sorry

end hot_dogs_served_today_l216_216836


namespace metro_earnings_in_6_minutes_l216_216261

theorem metro_earnings_in_6_minutes 
  (ticket_cost : ℕ) 
  (tickets_per_minute : ℕ) 
  (duration_minutes : ℕ) 
  (earnings_in_one_minute : ℕ) 
  (earnings_in_six_minutes : ℕ) 
  (h1 : ticket_cost = 3) 
  (h2 : tickets_per_minute = 5) 
  (h3 : duration_minutes = 6) 
  (h4 : earnings_in_one_minute = tickets_per_minute * ticket_cost) 
  (h5 : earnings_in_six_minutes = earnings_in_one_minute * duration_minutes) 
  : earnings_in_six_minutes = 90 := 
by 
  -- Proof goes here
  sorry

end metro_earnings_in_6_minutes_l216_216261


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l216_216350

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l216_216350


namespace max_area_triangle_PAB_l216_216331

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 16) + (y^2 / 9) = 1

def point_A : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)

theorem max_area_triangle_PAB (P : ℝ × ℝ) (hP : ellipse_eq P.1 P.2) : 
  ∃ S, S = 6 * (sqrt 2 + 1) := 
sorry

end max_area_triangle_PAB_l216_216331


namespace min_value_expression_l216_216786

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = ( (x + 1) * (2 * y + 1) ) / (Real.sqrt (x * y)) ∧ min_val = 4 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_l216_216786


namespace Lauryn_employees_l216_216391

variables (M W : ℕ)

theorem Lauryn_employees (h1 : M = W - 20) (h2 : M + W = 180) : M = 80 :=
by {
    sorry
}

end Lauryn_employees_l216_216391


namespace shoes_difference_l216_216480

theorem shoes_difference : 
  ∀ (Scott_shoes Anthony_shoes Jim_shoes : ℕ), 
  Scott_shoes = 7 → 
  Anthony_shoes = 3 * Scott_shoes → 
  Jim_shoes = Anthony_shoes - 2 → 
  Anthony_shoes - Jim_shoes = 2 :=
by
  intros Scott_shoes Anthony_shoes Jim_shoes 
  intros h1 h2 h3 
  sorry

end shoes_difference_l216_216480


namespace expand_product_l216_216777

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end expand_product_l216_216777


namespace steve_speed_back_l216_216630

theorem steve_speed_back :
  ∀ (v : ℝ), v > 0 → (20 / v + 20 / (2 * v) = 6) → 2 * v = 10 := 
by
  intros v v_pos h
  sorry

end steve_speed_back_l216_216630


namespace initial_calculated_average_l216_216490

theorem initial_calculated_average (S : ℕ) (initial_average correct_average : ℕ) (num_wrongly_read correctly_read wrong_value correct_value : ℕ)
    (h1 : num_wrongly_read = 36) 
    (h2 : correctly_read = 26) 
    (h3 : correct_value = 6)
    (h4 : S = 10 * correct_value) :
    initial_average = (S - (num_wrongly_read - correctly_read)) / 10 → initial_average = 5 :=
sorry

end initial_calculated_average_l216_216490


namespace problem_1_problem_2_l216_216769

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * |x| - 2

theorem problem_1 : {x : ℝ | f x > 3} = {x : ℝ | x < -1 ∨ x > 5} :=
sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m ≤ 1 :=
sorry

end problem_1_problem_2_l216_216769


namespace work_days_for_A_l216_216569

/-- If A is thrice as fast as B and together they can do a work in 15 days, A alone can do the work in 20 days. -/
theorem work_days_for_A (Wb : ℕ) (Wa : ℕ) (H_wa : Wa = 3 * Wb) (H_total : (Wa + Wb) * 15 = Wa * 20) : A_work_days = 20 :=
by
  sorry

end work_days_for_A_l216_216569


namespace greatest_sum_l216_216673

theorem greatest_sum (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 :=
sorry

end greatest_sum_l216_216673


namespace union_complement_eq_l216_216443

open Set

-- Condition definitions
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem statement (what we want to prove)
theorem union_complement_eq :
  A ∪ compl B = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_complement_eq_l216_216443


namespace circles_tangent_iff_l216_216979

noncomputable def C1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 }
noncomputable def C2 (m: ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.1 + 8 * p.2 + m = 0 }

theorem circles_tangent_iff (m: ℝ) : (∀ p ∈ C1, p ∈ C2 m → False) ↔ (m = -4 ∨ m = 16) := 
sorry

end circles_tangent_iff_l216_216979


namespace solve_for_x_l216_216182

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  5 * y ^ 2 + 3 * y + 2 = 3 * (8 * x ^ 2 + y + 1) ↔ x = 1 / Real.sqrt 21 ∨ x = -1 / Real.sqrt 21 :=
by
  sorry

end solve_for_x_l216_216182


namespace irrational_lattice_point_exists_l216_216285

theorem irrational_lattice_point_exists (k : ℝ) (h_irrational : ¬ ∃ q r : ℚ, q / r = k)
  (ε : ℝ) (h_pos : ε > 0) : ∃ m n : ℤ, |m * k - n| < ε :=
by
  sorry

end irrational_lattice_point_exists_l216_216285


namespace smallest_sum_of_digits_l216_216477

theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), 
    (100 ≤ a ∧ a < 1000) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (∃ (d1 d2 d3 d4 d5 : ℕ), 
      (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ 
      (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ 
      (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ 
      (d4 ≠ d5) ∧ 
      S = a + b ∧ 100 ≤ S ∧ S < 1000 ∧ 
      (∃ (s : ℕ), 
        s = (S / 100) + ((S % 100) / 10) + (S % 10) ∧ 
        s = 3)) :=
sorry

end smallest_sum_of_digits_l216_216477


namespace first_divisor_l216_216557

theorem first_divisor (y : ℝ) (x : ℝ) (h1 : 320 / (y * 3) = x) (h2 : x = 53.33) : y = 2 :=
sorry

end first_divisor_l216_216557


namespace even_function_value_at_three_l216_216411

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- f is an even function
axiom h_even : ∀ x, f x = f (-x)

-- f(x) is defined as x^2 + x when x < 0
axiom h_neg_def : ∀ x, x < 0 → f x = x^2 + x

theorem even_function_value_at_three : f 3 = 6 :=
by {
  -- To be proved
  sorry
}

end even_function_value_at_three_l216_216411


namespace tables_needed_l216_216183

open Nat

def base7_to_base10 (n : Nat) : Nat := 
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

theorem tables_needed (attendees_base7 : Nat) (attendees_base10 : Nat) (tables : Nat) :
  attendees_base7 = 312 ∧ attendees_base10 = base7_to_base10 attendees_base7 ∧ attendees_base10 = 156 ∧ tables = attendees_base10 / 3 → tables = 52 := 
by
  intros
  sorry

end tables_needed_l216_216183


namespace num_valid_functions_l216_216851

theorem num_valid_functions :
  ∃! (f : ℤ → ℝ), 
  (f 1 = 1) ∧ 
  (∀ (m n : ℤ), f m ^ 2 - f n ^ 2 = f (m + n) * f (m - n)) ∧ 
  (∀ n : ℤ, f n = f (n + 2013)) :=
sorry

end num_valid_functions_l216_216851


namespace three_digit_reverse_sum_to_1777_l216_216491

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l216_216491


namespace mark_wait_time_l216_216416

theorem mark_wait_time (t1 t2 T : ℕ) (h1 : t1 = 4) (h2 : t2 = 20) (hT : T = 38) : 
  T - (t1 + t2) = 14 :=
by sorry

end mark_wait_time_l216_216416


namespace fraction_nonneg_if_x_ge_m8_l216_216616

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end fraction_nonneg_if_x_ge_m8_l216_216616


namespace benjamin_earns_more_l216_216046

noncomputable def additional_earnings : ℝ :=
  let P : ℝ := 75000
  let r : ℝ := 0.05
  let t_M : ℝ := 3
  let r_m : ℝ := r / 12
  let t_B : ℝ := 36
  let A_M : ℝ := P * (1 + r)^t_M
  let A_B : ℝ := P * (1 + r_m)^t_B
  A_B - A_M

theorem benjamin_earns_more : additional_earnings = 204 := by
  sorry

end benjamin_earns_more_l216_216046


namespace residue_mod_13_l216_216591

theorem residue_mod_13 : 
  (156 % 13 = 0) ∧ (52 % 13 = 0) ∧ (182 % 13 = 0) ∧ (26 % 13 = 0) →
  (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 :=
by
  intros h
  sorry

end residue_mod_13_l216_216591


namespace min_value_x_3y_l216_216376

noncomputable def min_value (x y : ℝ) : ℝ := x + 3 * y

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∃ (x y : ℝ), min_value x y = 18 + 21 * Real.sqrt 3 :=
sorry

end min_value_x_3y_l216_216376


namespace greatest_constant_right_triangle_l216_216549

theorem greatest_constant_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) (K : ℝ) 
    (hK : (a^2 + b^2) / (a^2 + b^2 + c^2) > K) : 
    K ≤ 1 / 2 :=
by 
  sorry

end greatest_constant_right_triangle_l216_216549


namespace product_of_digits_l216_216796

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : 8 ∣ (10 * A + B)) : A * B = 32 :=
sorry

end product_of_digits_l216_216796


namespace bank_exceeds_50_dollars_l216_216223

theorem bank_exceeds_50_dollars (a : ℕ := 5) (r : ℕ := 2) :
  ∃ n : ℕ, 5 * (2 ^ n - 1) > 5000 ∧ (n ≡ 9 [MOD 7]) :=
by
  sorry

end bank_exceeds_50_dollars_l216_216223


namespace fourth_root_of_25000000_eq_70_7_l216_216492

theorem fourth_root_of_25000000_eq_70_7 :
  Real.sqrt (Real.sqrt 25000000) = 70.7 :=
sorry

end fourth_root_of_25000000_eq_70_7_l216_216492


namespace minimum_value_l216_216080

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y / x = 1) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = (1 / x + x / y) → z ≥ m :=
sorry

end minimum_value_l216_216080


namespace parsnip_box_fullness_l216_216448

theorem parsnip_box_fullness (capacity : ℕ) (fraction_full : ℚ) (avg_boxes : ℕ) (avg_parsnips : ℕ) :
  capacity = 20 →
  fraction_full = 3 / 4 →
  avg_boxes = 20 →
  avg_parsnips = 350 →
  ∃ (full_boxes : ℕ) (non_full_boxes : ℕ) (parsnips_in_full_boxes : ℕ) (parsnips_in_non_full_boxes : ℕ)
    (avg_fullness_non_full_boxes : ℕ),
    full_boxes = fraction_full * avg_boxes ∧
    non_full_boxes = avg_boxes - full_boxes ∧
    parsnips_in_full_boxes = full_boxes * capacity ∧
    parsnips_in_non_full_boxes = avg_parsnips - parsnips_in_full_boxes ∧
    avg_fullness_non_full_boxes = parsnips_in_non_full_boxes / non_full_boxes ∧
    avg_fullness_non_full_boxes = 10 :=
by
  sorry

end parsnip_box_fullness_l216_216448


namespace amount_b_l216_216240

variable {a b : ℚ} -- a and b are rational numbers

theorem amount_b (h1 : a + b = 1210) (h2 : (4 / 15) * a = (2 / 5) * b) : b = 484 :=
sorry

end amount_b_l216_216240


namespace total_rooms_in_hotel_l216_216972

def first_wing_floors : ℕ := 9
def first_wing_halls_per_floor : ℕ := 6
def first_wing_rooms_per_hall : ℕ := 32

def second_wing_floors : ℕ := 7
def second_wing_halls_per_floor : ℕ := 9
def second_wing_rooms_per_hall : ℕ := 40

def third_wing_floors : ℕ := 12
def third_wing_halls_per_floor : ℕ := 4
def third_wing_rooms_per_hall : ℕ := 50

def first_wing_total_rooms : ℕ := 
  first_wing_floors * first_wing_halls_per_floor * first_wing_rooms_per_hall

def second_wing_total_rooms : ℕ := 
  second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall

def third_wing_total_rooms : ℕ := 
  third_wing_floors * third_wing_halls_per_floor * third_wing_rooms_per_hall

theorem total_rooms_in_hotel : 
  first_wing_total_rooms + second_wing_total_rooms + third_wing_total_rooms = 6648 := 
by 
  sorry

end total_rooms_in_hotel_l216_216972


namespace find_original_expression_l216_216440

theorem find_original_expression (a b c X : ℤ) :
  X + (a * b - 2 * b * c + 3 * a * c) = 2 * b * c - 3 * a * c + 2 * a * b →
  X = 4 * b * c - 6 * a * c + a * b :=
by
  sorry

end find_original_expression_l216_216440


namespace compare_probabilities_l216_216138

noncomputable def box_bad_coin_prob_method_one : ℝ := 1 - (0.99 ^ 10)
noncomputable def box_bad_coin_prob_method_two : ℝ := 1 - ((49 / 50) ^ 5)

theorem compare_probabilities : box_bad_coin_prob_method_one < box_bad_coin_prob_method_two := by
  sorry

end compare_probabilities_l216_216138


namespace calculation_result_l216_216398

theorem calculation_result : 2014 * (1/19 - 1/53) = 68 := by
  sorry

end calculation_result_l216_216398


namespace hilary_stalks_l216_216022

-- Define the given conditions
def ears_per_stalk : ℕ := 4
def kernels_per_ear_first_half : ℕ := 500
def kernels_per_ear_second_half : ℕ := 600
def total_kernels : ℕ := 237600

-- Average number of kernels per ear
def average_kernels_per_ear : ℕ := (kernels_per_ear_first_half + kernels_per_ear_second_half) / 2

-- Total number of ears based on total kernels
noncomputable def total_ears : ℕ := total_kernels / average_kernels_per_ear

-- Total number of stalks based on total ears
noncomputable def total_stalks : ℕ := total_ears / ears_per_stalk

-- The main theorem to prove
theorem hilary_stalks : total_stalks = 108 :=
by
  sorry

end hilary_stalks_l216_216022


namespace no_positive_integers_satisfy_condition_l216_216935

theorem no_positive_integers_satisfy_condition :
  ∀ (n : ℕ), n > 0 → ¬∃ (a b m : ℕ), a > 0 ∧ b > 0 ∧ m > 0 ∧ 
  (a + b * Real.sqrt n) ^ 2023 = Real.sqrt m + Real.sqrt (m + 2022) := by
  sorry

end no_positive_integers_satisfy_condition_l216_216935


namespace filled_sacks_count_l216_216883

-- Definitions from the problem conditions
def pieces_per_sack := 20
def total_pieces := 80

theorem filled_sacks_count : total_pieces / pieces_per_sack = 4 := 
by sorry

end filled_sacks_count_l216_216883


namespace valid_lineup_count_l216_216134

noncomputable def num_valid_lineups : ℕ :=
  let total_lineups := Nat.choose 18 8
  let unwanted_lineups := Nat.choose 14 4
  total_lineups - unwanted_lineups

theorem valid_lineup_count : num_valid_lineups = 42757 := by
  sorry

end valid_lineup_count_l216_216134


namespace total_action_figures_l216_216314

theorem total_action_figures (figures_per_shelf : ℕ) (number_of_shelves : ℕ) (h1 : figures_per_shelf = 10) (h2 : number_of_shelves = 8) : figures_per_shelf * number_of_shelves = 80 := by
  sorry

end total_action_figures_l216_216314


namespace isosceles_triangle_sum_x_l216_216135

noncomputable def sum_possible_values_of_x : ℝ :=
  let x1 : ℝ := 20
  let x2 : ℝ := 50
  let x3 : ℝ := 80
  x1 + x2 + x3

theorem isosceles_triangle_sum_x (x : ℝ) (h1 : x = 20 ∨ x = 50 ∨ x = 80) : sum_possible_values_of_x = 150 :=
  by
    sorry

end isosceles_triangle_sum_x_l216_216135


namespace average_student_headcount_l216_216228

def student_headcount_fall_0203 : ℕ := 11700
def student_headcount_fall_0304 : ℕ := 11500
def student_headcount_fall_0405 : ℕ := 11600

theorem average_student_headcount : 
  (student_headcount_fall_0203 + student_headcount_fall_0304 + student_headcount_fall_0405) / 3 = 11600 := by
  sorry

end average_student_headcount_l216_216228


namespace number_of_2_dollar_socks_l216_216193

theorem number_of_2_dollar_socks :
  ∃ (a b c : ℕ), (a + b + c = 15) ∧ (2 * a + 3 * b + 5 * c = 40) ∧ (a ≥ 1) ∧ (b ≥ 1) ∧ (c ≥ 1) ∧ (a = 7 ∨ a = 9 ∨ a = 11) :=
by {
  -- The details of the proof will go here, but we skip it for our requirements
  sorry
}

end number_of_2_dollar_socks_l216_216193


namespace teams_match_count_l216_216604

theorem teams_match_count
  (n : ℕ)
  (h : n = 6)
: (n * (n - 1)) / 2 = 15 := by
  sorry

end teams_match_count_l216_216604


namespace rate_of_rainfall_on_Monday_l216_216051

theorem rate_of_rainfall_on_Monday (R : ℝ) :
  7 * R + 4 * 2 + 2 * (2 * 2) = 23 → R = 1 := 
by
  sorry

end rate_of_rainfall_on_Monday_l216_216051


namespace wheat_pile_weight_l216_216640

noncomputable def weight_of_conical_pile
  (circumference : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  let r := circumference / (2 * 3.14)
  let volume := (1.0 / 3.0) * 3.14 * r^2 * height
  volume * density

theorem wheat_pile_weight :
  weight_of_conical_pile 12.56 1.2 30 = 150.72 :=
by
  sorry

end wheat_pile_weight_l216_216640


namespace james_take_home_pay_l216_216173

theorem james_take_home_pay :
  let main_hourly_rate := 20
  let second_hourly_rate := main_hourly_rate - (main_hourly_rate * 0.20)
  let main_hours := 30
  let second_hours := main_hours / 2
  let side_gig_earnings := 100 * 2
  let overtime_hours := 5
  let overtime_rate := main_hourly_rate * 1.5
  let irs_tax_rate := 0.18
  let state_tax_rate := 0.05
  
  -- Main job earnings
  let main_regular_earnings := main_hours * main_hourly_rate
  let main_overtime_earnings := overtime_hours * overtime_rate
  let main_total_earnings := main_regular_earnings + main_overtime_earnings
  
  -- Second job earnings
  let second_total_earnings := second_hours * second_hourly_rate
  
  -- Total earnings before taxes
  let total_earnings := main_total_earnings + second_total_earnings + side_gig_earnings
  
  -- Tax calculations
  let federal_tax := total_earnings * irs_tax_rate
  let state_tax := total_earnings * state_tax_rate
  let total_taxes := federal_tax + state_tax

  -- Total take home pay after taxes
  let take_home_pay := total_earnings - total_taxes

  take_home_pay = 916.30 := 
sorry

end james_take_home_pay_l216_216173


namespace four_person_apartments_l216_216342

theorem four_person_apartments : 
  ∃ x : ℕ, 
    (4 * (10 + 20 * 2 + 4 * x)) * 3 / 4 = 210 → x = 5 :=
by
  sorry

end four_person_apartments_l216_216342


namespace final_price_chocolate_l216_216412

-- Conditions
def original_cost : ℝ := 2.00
def discount : ℝ := 0.57

-- Question and answer
theorem final_price_chocolate : original_cost - discount = 1.43 :=
by
  sorry

end final_price_chocolate_l216_216412


namespace number_divisible_by_23_and_29_l216_216179

theorem number_divisible_by_23_and_29 (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  23 ∣ (200100 * a + 20010 * b + 2001 * c) ∧ 29 ∣ (200100 * a + 20010 * b + 2001 * c) :=
by
  sorry

end number_divisible_by_23_and_29_l216_216179


namespace find_value_b_l216_216191

-- Define the problem-specific elements
noncomputable def is_line_eqn (y x : ℝ) : Prop := y = 4 - 2 * x

theorem find_value_b (b : ℝ) (h₀ : b > 0) (h₁ : b < 2)
  (hP : ∀ y, is_line_eqn y 0 → y = 4)
  (hS : ∀ y, is_line_eqn y 2 → y = 0)
  (h_ratio : ∀ Q R S O P,
    Q = (2, 0) ∧ R = (2, 0) ∧ S = (2, 0) ∧ P = (0, 4) ∧ O = (0, 0) →
    4 / 9 = 4 / ((Q.1 - O.1) * (Q.1 - O.1)) →
    (Q.1 - O.1) / (P.2 - O.2) = 2 / 3) :
  b = 2 :=
sorry

end find_value_b_l216_216191


namespace translate_quadratic_vertex_right_l216_216205

theorem translate_quadratic_vertex_right : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = 2 * (x - 4)^2 - 3) ∧ 
  (∃ (g : ℝ → ℝ), (∀ x, g x = 2 * ((x - 1) - 3)^2 - 3))) → 
  (∃ v : ℝ × ℝ, v = (4, -3)) :=
sorry

end translate_quadratic_vertex_right_l216_216205


namespace triangle_base_length_l216_216598

theorem triangle_base_length (base : ℝ) (h1 : ∃ (side : ℝ), side = 6 ∧ (side^2 = (base * 12) / 2)) : base = 6 :=
sorry

end triangle_base_length_l216_216598


namespace range_of_m_l216_216257

theorem range_of_m :
  (∀ x : ℝ, (x > 0) → (x^2 - m * x + 4 ≥ 0)) ∧ (¬∃ x : ℝ, (x^2 - 2 * m * x + 7 * m - 10 = 0)) ↔ (2 < m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l216_216257


namespace max_sum_distinct_factors_2029_l216_216351

theorem max_sum_distinct_factors_2029 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2029 ∧ A + B + C = 297 :=
by
  sorry

end max_sum_distinct_factors_2029_l216_216351


namespace carol_first_six_l216_216788

-- A formalization of the probabilities involved when Alice, Bob, Carol,
-- and Dave take turns rolling a die, and the process repeats.
def probability_carol_first_six (prob_rolling_six : ℚ) : ℚ := sorry

theorem carol_first_six (prob_rolling_six : ℚ) (h : prob_rolling_six = 1/6) :
  probability_carol_first_six prob_rolling_six = 25 / 91 :=
sorry

end carol_first_six_l216_216788


namespace larger_integer_is_21_l216_216001

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l216_216001


namespace smallest_ab_41503_539_l216_216473

noncomputable def find_smallest_ab : (ℕ × ℕ) :=
  let a := 41503
  let b := 539
  (a, b)

theorem smallest_ab_41503_539 (a b : ℕ) (h : 7 * a^3 = 11 * b^5) (ha : a > 0) (hb : b > 0) :
  (a = 41503 ∧ b = 539) :=
  by
    -- Add sorry to skip the proof
    sorry

end smallest_ab_41503_539_l216_216473


namespace total_fireworks_correct_l216_216265

variable (fireworks_num fireworks_reg)
variable (fireworks_H fireworks_E fireworks_L fireworks_O)
variable (fireworks_square fireworks_triangle fireworks_circle)
variable (boxes fireworks_per_box : ℕ)

-- Given Conditions
def fireworks_years_2021_2023 : ℕ := 6 * 4 * 3
def fireworks_HAPPY_NEW_YEAR : ℕ := 5 * 11 + 6
def fireworks_geometric_shapes : ℕ := 4 + 3 + 12
def fireworks_HELLO : ℕ := 8 + 7 + 6 * 2 + 9
def fireworks_additional_boxes : ℕ := 100 * 10

-- Total Fireworks
def total_fireworks : ℕ :=
  fireworks_years_2021_2023 + 
  fireworks_HAPPY_NEW_YEAR + 
  fireworks_geometric_shapes + 
  fireworks_HELLO + 
  fireworks_additional_boxes

theorem total_fireworks_correct : 
  total_fireworks = 1188 :=
  by
  -- The proof is omitted.
  sorry

end total_fireworks_correct_l216_216265


namespace repeating_decimals_product_fraction_l216_216035

theorem repeating_decimals_product_fraction : 
  let x := 1 / 33
  let y := 9 / 11
  x * y = 9 / 363 := 
by
  sorry

end repeating_decimals_product_fraction_l216_216035


namespace sum_of_coefficients_l216_216949

theorem sum_of_coefficients
  (d : ℝ)
  (g h : ℝ)
  (h1 : (8 * d^2 - 4 * d + g) * (5 * d^2 + h * d - 10) = 40 * d^4 - 75 * d^3 - 90 * d^2 + 5 * d + 20) :
  g + h = 15.5 :=
sorry

end sum_of_coefficients_l216_216949


namespace distance_after_time_l216_216937

noncomputable def Adam_speed := 12 -- speed in mph
noncomputable def Simon_speed := 6 -- speed in mph
noncomputable def time_when_100_miles_apart := 100 / 15 -- hours

theorem distance_after_time (x : ℝ) : 
  (Adam_speed * x)^2 + (Simon_speed * x)^2 = 100^2 ->
  x = time_when_100_miles_apart := 
by
  sorry

end distance_after_time_l216_216937


namespace parallelogram_side_length_l216_216002

-- We need trigonometric functions and operations with real numbers.
open Real

theorem parallelogram_side_length (s : ℝ) 
  (h_side_lengths : s > 0 ∧ 3 * s > 0) 
  (h_angle : sin (30 / 180 * π) = 1 / 2) 
  (h_area : 3 * s * (s * sin (30 / 180 * π)) = 9 * sqrt 3) :
  s = 3 * sqrt 2 :=
by
  sorry

end parallelogram_side_length_l216_216002


namespace square_area_l216_216352

theorem square_area (l w x : ℝ) (h1 : 2 * (l + w) = 20) (h2 : l = x / 2) (h3 : w = x / 4) :
  x^2 = 1600 / 9 :=
by
  sorry

end square_area_l216_216352


namespace pd_distance_l216_216203

theorem pd_distance (PA PB PC PD : ℝ) (hPA : PA = 17) (hPB : PB = 15) (hPC : PC = 6) :
  PA^2 + PC^2 = PB^2 + PD^2 → PD = 10 :=
by
  sorry

end pd_distance_l216_216203


namespace sum_of_possible_values_l216_216543

theorem sum_of_possible_values {x : ℝ} :
  (3 * (x - 3)^2 = (x - 2) * (x + 5)) →
  (∃ (x1 x2 : ℝ), x1 + x2 = 10.5) :=
by sorry

end sum_of_possible_values_l216_216543


namespace shirts_per_minute_l216_216295

/--
An industrial machine made 8 shirts today and worked for 4 minutes today. 
Prove that the machine can make 2 shirts per minute.
-/
theorem shirts_per_minute (shirts_today : ℕ) (minutes_today : ℕ)
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  (shirts_today / minutes_today) = 2 :=
by sorry

end shirts_per_minute_l216_216295


namespace lindsey_squat_weight_l216_216499

theorem lindsey_squat_weight :
  let bandA := 7
  let bandB := 5
  let bandC := 3
  let leg_weight := 10
  let dumbbell := 15
  let total_weight := (2 * bandA) + (2 * bandB) + (2 * bandC) + (2 * leg_weight) + dumbbell
  total_weight = 65 :=
by
  sorry

end lindsey_squat_weight_l216_216499


namespace find_vector_from_origin_to_line_l216_216964

theorem find_vector_from_origin_to_line :
  ∃ t : ℝ, (3 * t + 1, 2 * t + 3) = (16, 32 / 3) ∧
  ∃ k : ℝ, (16, 32 / 3) = (3 * k, 2 * k) :=
sorry

end find_vector_from_origin_to_line_l216_216964


namespace tangent_line_equations_l216_216527

theorem tangent_line_equations (k b : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = k * x + b) ∧
    (∃ x₁, x₁^2 = k * x₁ + b) ∧ -- Tangency condition with C1: y = x²
    (∃ x₂, -(x₂ - 2)^2 = k * x₂ + b)) -- Tangency condition with C2: y = -(x-2)²
  → ((k = 0 ∧ b = 0) ∨ (k = 4 ∧ b = -4)) := sorry

end tangent_line_equations_l216_216527


namespace infinitely_many_good_numbers_seven_does_not_divide_good_number_l216_216565

-- Define what it means for a number to be good
def is_good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = n ∧ (a * b) ∣ (n^2 + n + 1)

-- Part (a): Show that there are infinitely many good numbers
theorem infinitely_many_good_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_good_number (f n) :=
sorry

-- Part (b): Show that if n is a good number, then 7 does not divide n
theorem seven_does_not_divide_good_number (n : ℕ) (h : is_good_number n) : ¬ (7 ∣ n) :=
sorry

end infinitely_many_good_numbers_seven_does_not_divide_good_number_l216_216565


namespace candy_remainder_l216_216209

theorem candy_remainder :
  38759863 % 6 = 1 :=
by
  sorry

end candy_remainder_l216_216209


namespace find_line_l216_216385

def point_on_line (P : ℝ × ℝ) (m b : ℝ) : Prop :=
  P.2 = m * P.1 + b

def intersection_points_distance (k m b : ℝ) : Prop :=
  |(k^2 - 4*k + 4) - (m*k + b)| = 6

noncomputable def desired_line (m b : ℝ) : Prop :=
  point_on_line (2, 3) m b ∧ ∀ (k : ℝ), intersection_points_distance k m b

theorem find_line : desired_line (-6) 15 := sorry

end find_line_l216_216385


namespace spadesuit_eval_l216_216776

def spadesuit (a b : ℤ) := abs (a - b)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 3 (spadesuit 8 12)) = 4 := 
by
  sorry

end spadesuit_eval_l216_216776


namespace find_other_number_l216_216723

theorem find_other_number (B : ℕ)
  (HCF : Nat.gcd 24 B = 12)
  (LCM : Nat.lcm 24 B = 312) :
  B = 156 :=
by
  sorry

end find_other_number_l216_216723


namespace direct_proportion_function_l216_216976

theorem direct_proportion_function (m : ℝ) (h : ∀ x : ℝ, -2*x + m = k*x → m = 0) : m = 0 :=
sorry

end direct_proportion_function_l216_216976


namespace evaluate_expression_l216_216824

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end evaluate_expression_l216_216824


namespace expression_equals_24_l216_216301

noncomputable def f : ℕ → ℝ := sorry

axiom f_add (m n : ℕ) : f (m + n) = f m * f n
axiom f_one : f 1 = 3

theorem expression_equals_24 :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 + (f 4^2 + f 8) / f 7 = 24 :=
by sorry

end expression_equals_24_l216_216301


namespace vector_subtraction_l216_216876

def a : Real × Real := (2, -1)
def b : Real × Real := (-2, 3)

theorem vector_subtraction :
  a.1 - 2 * b.1 = 6 ∧ a.2 - 2 * b.2 = -7 := by
  sorry

end vector_subtraction_l216_216876


namespace friends_behind_Yuna_l216_216846

def total_friends : ℕ := 6
def friends_in_front_of_Yuna : ℕ := 2

theorem friends_behind_Yuna : total_friends - friends_in_front_of_Yuna = 4 :=
by
  -- Proof goes here
  sorry

end friends_behind_Yuna_l216_216846


namespace real_solutions_unique_l216_216374

theorem real_solutions_unique (a b c : ℝ) :
  (2 * a - b = a^2 * b ∧ 2 * b - c = b^2 * c ∧ 2 * c - a = c^2 * a) →
  (a, b, c) = (-1, -1, -1) ∨ (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, 1, 1) :=
by
  sorry

end real_solutions_unique_l216_216374


namespace fraction_of_power_l216_216313

noncomputable def m : ℕ := 32^500

theorem fraction_of_power (h : m = 2^2500) : m / 8 = 2^2497 :=
by
  have hm : m = 2^2500 := h
  sorry

end fraction_of_power_l216_216313


namespace part_one_part_two_l216_216325

def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Question (1)
theorem part_one (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
sorry

-- Question (2)
theorem part_two (a : ℝ) (h : a ≥ 1) : ∀ (y : ℝ), (∃ x : ℝ, f x a = y) ↔ (∃ b : ℝ, y = b + 2 ∧ b ≥ a) := 
sorry

end part_one_part_two_l216_216325


namespace sin_X_value_l216_216583

theorem sin_X_value (a b X : ℝ) (h₁ : (1/2) * a * b * Real.sin X = 72) (h₂ : Real.sqrt (a * b) = 16) :
  Real.sin X = 9 / 16 := by
  sorry

end sin_X_value_l216_216583


namespace leigh_path_length_l216_216338

theorem leigh_path_length :
  let north := 10
  let south := 40
  let west := 60
  let east := 20
  let net_south := south - north
  let net_west := west - east
  let distance := Real.sqrt (net_south^2 + net_west^2)
  distance = 50 := 
by sorry

end leigh_path_length_l216_216338


namespace inequality_proof_l216_216420

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem inequality_proof (α β : ℝ) (m : ℕ) (h1 : 1 < α) (h2 : 1 < β) (h3 : m = 1) 
  (h4 : f α m + f β m = 2) : (4 / α) + (1 / β) ≥ 9 / 2 := by
  sorry

end inequality_proof_l216_216420


namespace problem1_problem2_l216_216792

-- Statement for Problem ①
theorem problem1 
: ( (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2) := by
  sorry

-- Statement for Problem ②
theorem problem2
: ((-99 - 11 / 12) * 24 = -2398) := by
  sorry

end problem1_problem2_l216_216792


namespace cubic_sum_of_roots_l216_216019

theorem cubic_sum_of_roots :
  ∀ (r s : ℝ), (r + s = 5) → (r * s = 6) → (r^3 + s^3 = 35) :=
by
  intros r s h₁ h₂
  sorry

end cubic_sum_of_roots_l216_216019


namespace solution_l216_216328

theorem solution (a b : ℝ) (h1 : a^2 + 2 * a - 2016 = 0) (h2 : b^2 + 2 * b - 2016 = 0) :
  a^2 + 3 * a + b = 2014 := 
sorry

end solution_l216_216328


namespace problem_I_problem_II_l216_216634

-- Question I
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 :=
by
  sorry

-- Question II
theorem problem_II (a : ℝ) : (∀ x : ℝ, |x - a| + |2 * x - 1| ≥ 2) ↔ (a ≤ -3/2 ∨ a ≥ 5/2) :=
by
  sorry

end problem_I_problem_II_l216_216634


namespace divisibility_by_5_l216_216479

theorem divisibility_by_5 (B : ℕ) (hB : B < 10) : (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := 
by
  sorry

end divisibility_by_5_l216_216479


namespace coeff_x_squared_l216_216495

theorem coeff_x_squared (n : ℕ) (t h : ℕ)
  (h_t : t = 4^n) 
  (h_h : h = 2^n) 
  (h_sum : t + h = 272)
  (C : ℕ → ℕ → ℕ) -- binomial coefficient notation, we'll skip the direct proof of properties for simplicity
  : (C 4 4) * (3^0) = 1 := 
by 
  /-
  Proof steps (informal, not needed in Lean statement):
  Since the sum of coefficients is t, we have t = 4^n.
  For the sum of binomial coefficients, we have h = 2^n.
  Given t + h = 272, solve for n:
    4^n + 2^n = 272 
    implies 2^n = 16, so n = 4.
  Substitute into the general term (\(T_{r+1}\):
    T_{r+1} = C_4^r * 3^(4-r) * x^((8+r)/6)
  For x^2 term, set (8+r)/6 = 2, yielding r = 4.
  The coefficient is C_4^4 * 3^0 = 1.
  -/
  sorry

end coeff_x_squared_l216_216495


namespace find_a_l216_216421

theorem find_a (f : ℝ → ℝ) (a x : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x - 5) 
  (h2 : f a = 6) : a = 7 / 4 := 
by
  sorry

end find_a_l216_216421


namespace staplers_left_is_correct_l216_216745

-- Define the initial conditions as constants
def initial_staplers : ℕ := 450
def stacie_reports : ℕ := 8 * 12 -- Stacie's reports in dozens converted to actual number
def jack_reports : ℕ := 9 * 12   -- Jack's reports in dozens converted to actual number
def laura_reports : ℕ := 50      -- Laura's individual reports

-- Define the stapler usage rates
def stacie_usage_rate : ℕ := 1                  -- Stacie's stapler usage rate (1 stapler per report)
def jack_usage_rate : ℕ := stacie_usage_rate / 2  -- Jack's stapler usage rate (half of Stacie's)
def laura_usage_rate : ℕ := stacie_usage_rate * 2 -- Laura's stapler usage rate (twice of Stacie's)

-- Define the usage calculations
def stacie_usage : ℕ := stacie_reports * stacie_usage_rate
def jack_usage : ℕ := jack_reports * jack_usage_rate
def laura_usage : ℕ := laura_reports * laura_usage_rate

-- Define total staplers used
def total_usage : ℕ := stacie_usage + jack_usage + laura_usage

-- Define the number of staplers left
def staplers_left : ℕ := initial_staplers - total_usage

-- Prove that the staplers left is 200
theorem staplers_left_is_correct : staplers_left = 200 := by
  unfold staplers_left initial_staplers total_usage stacie_usage jack_usage laura_usage
  unfold stacie_reports jack_reports laura_reports
  unfold stacie_usage_rate jack_usage_rate laura_usage_rate
  sorry   -- Place proof here

end staplers_left_is_correct_l216_216745


namespace back_seat_people_l216_216843

-- Define the problem conditions

def leftSideSeats : ℕ := 15
def seatDifference : ℕ := 3
def peoplePerSeat : ℕ := 3
def totalBusCapacity : ℕ := 88

-- Define the formula for calculating the people at the back seat
def peopleAtBackSeat := 
  totalBusCapacity - ((leftSideSeats * peoplePerSeat) + ((leftSideSeats - seatDifference) * peoplePerSeat))

-- The statement we need to prove
theorem back_seat_people : peopleAtBackSeat = 7 :=
by
  sorry

end back_seat_people_l216_216843


namespace parallel_vectors_l216_216830

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (P : a = (1, m) ∧ b = (m, 2) ∧ (a.1 / m = b.1 / 2)) :
  m = -Real.sqrt 2 ∨ m = Real.sqrt 2 :=
by
  sorry

end parallel_vectors_l216_216830


namespace hiring_manager_acceptance_l216_216555

theorem hiring_manager_acceptance :
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  k = 19 / 18 :=
by
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  show k = 19 / 18
  sorry

end hiring_manager_acceptance_l216_216555


namespace YooSeung_has_108_marbles_l216_216449

def YoungSoo_marble_count : ℕ := 21
def HanSol_marble_count : ℕ := YoungSoo_marble_count + 15
def YooSeung_marble_count : ℕ := 3 * HanSol_marble_count
def total_marble_count : ℕ := YoungSoo_marble_count + HanSol_marble_count + YooSeung_marble_count

theorem YooSeung_has_108_marbles 
  (h1 : YooSeung_marble_count = 3 * (YoungSoo_marble_count + 15))
  (h2 : HanSol_marble_count = YoungSoo_marble_count + 15)
  (h3 : total_marble_count = 165) :
  YooSeung_marble_count = 108 :=
by sorry

end YooSeung_has_108_marbles_l216_216449


namespace union_of_sets_l216_216857

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2}) (hB : B = {2, 3}) : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l216_216857


namespace inv_mod_997_l216_216382

theorem inv_mod_997 : ∃ x : ℤ, 0 ≤ x ∧ x < 997 ∧ (10 * x) % 997 = 1 := 
sorry

end inv_mod_997_l216_216382


namespace power_of_fraction_l216_216077

theorem power_of_fraction :
  (3 / 4) ^ 5 = 243 / 1024 :=
by sorry

end power_of_fraction_l216_216077


namespace total_cows_l216_216140

theorem total_cows (cows : ℕ) (h1 : cows / 3 + cows / 5 + cows / 6 + 12 = cows) : cows = 40 :=
sorry

end total_cows_l216_216140


namespace people_left_gym_l216_216190

theorem people_left_gym (initial : ℕ) (additional : ℕ) (current : ℕ) (H1 : initial = 16) (H2 : additional = 5) (H3 : current = 19) : (initial + additional - current) = 2 :=
by
  sorry

end people_left_gym_l216_216190


namespace gcd_possible_values_count_l216_216805

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l216_216805


namespace xiaoGong_walking_speed_l216_216814

-- Defining the parameters for the problem
def distance : ℕ := 1200
def daChengExtraSpeedPerMinute : ℕ := 20
def timeUntilMeetingForDaCheng : ℕ := 12
def timeUntilMeetingForXiaoGong : ℕ := 6 + timeUntilMeetingForDaCheng

-- The main statement to prove Xiao Gong's speed
theorem xiaoGong_walking_speed : ∃ v : ℕ, 12 * (v + daChengExtraSpeedPerMinute) + 18 * v = distance ∧ v = 32 :=
by
  sorry

end xiaoGong_walking_speed_l216_216814


namespace turtle_speed_l216_216445

theorem turtle_speed
  (hare_speed : ℝ)
  (race_distance : ℝ)
  (head_start : ℝ) :
  hare_speed = 10 → race_distance = 20 → head_start = 18 → 
  (race_distance / (head_start + race_distance / hare_speed) = 1) :=
by
  intros
  sorry

end turtle_speed_l216_216445


namespace child_ticket_cost_l216_216875

-- Define the conditions
def adult_ticket_cost : ℕ := 11
def total_people : ℕ := 23
def total_revenue : ℕ := 246
def children_count : ℕ := 7
def adults_count := total_people - children_count

-- Define the target to prove that the child ticket cost is 10
theorem child_ticket_cost (child_ticket_cost : ℕ) :
  16 * adult_ticket_cost + 7 * child_ticket_cost = total_revenue → 
  child_ticket_cost = 10 := by
  -- The proof is omitted
  sorry

end child_ticket_cost_l216_216875


namespace intersection_of_A_and_B_l216_216245

noncomputable def A : Set ℝ := {x | x^2 - 1 ≤ 0}

noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l216_216245


namespace jenny_kenny_see_each_other_l216_216629

-- Definitions of conditions
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def paths_distance : ℝ := 300
def radius_building : ℝ := 75
def start_distance : ℝ := 300

-- Theorem statement
theorem jenny_kenny_see_each_other : ∃ t : ℝ, (t = 120) :=
by
  sorry

end jenny_kenny_see_each_other_l216_216629


namespace arithmetic_sequence_S22_zero_l216_216089

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_S22_zero (a d : ℝ) (S : ℕ → ℝ) (h_arith_seq : ∀ n, S n = sum_of_first_n_terms a d n)
  (h1 : a > 0) (h2 : S 5 = S 17) :
  S 22 = 0 :=
by
  sorry

end arithmetic_sequence_S22_zero_l216_216089


namespace average_marks_l216_216468

theorem average_marks :
  let class1_students := 26
  let class1_avg_marks := 40
  let class2_students := 50
  let class2_avg_marks := 60
  let total_students := class1_students + class2_students
  let total_marks := (class1_students * class1_avg_marks) + (class2_students * class2_avg_marks)
  (total_marks / total_students : ℝ) = 53.16 := by
sorry

end average_marks_l216_216468


namespace gridiron_football_club_members_count_l216_216861

theorem gridiron_football_club_members_count :
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  total_expenditure / total_cost_per_member = 104 :=
by
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  sorry

end gridiron_football_club_members_count_l216_216861


namespace failed_by_35_l216_216102

variables (M S P : ℝ)
variables (hM : M = 153.84615384615384)
variables (hS : S = 45)
variables (hP : P = 0.52 * M)

theorem failed_by_35 (hM : M = 153.84615384615384) (hS : S = 45) (hP : P = 0.52 * M) : P - S = 35 :=
by
  sorry

end failed_by_35_l216_216102


namespace rowers_voted_l216_216040

variable (R : ℕ)

/-- Each rower votes for exactly 4 coaches out of 50 coaches,
and each coach receives exactly 7 votes.
Prove that the number of rowers is 88. -/
theorem rowers_voted (h1 : 50 * 7 = 4 * R) : R = 88 := by 
  sorry

end rowers_voted_l216_216040


namespace factor_expression_l216_216076

theorem factor_expression (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) :=
by
  sorry

end factor_expression_l216_216076


namespace length_of_platform_is_180_l216_216649

-- Define the train passing a platform and a man with given speeds and times
def train_pass_platform (speed : ℝ) (time_man time_platform : ℝ) (length_train length_platform : ℝ) :=
  time_man = length_train / speed ∧ 
  time_platform = (length_train + length_platform) / speed

-- Given conditions
noncomputable def train_length_platform :=
  ∃ length_platform,
    train_pass_platform 15 20 32 300 length_platform ∧
    length_platform = 180

-- The main theorem we want to prove
theorem length_of_platform_is_180 : train_length_platform :=
sorry

end length_of_platform_is_180_l216_216649


namespace no_real_solution_l216_216947

theorem no_real_solution (P : ℝ → ℝ) (h_cont : Continuous P) (h_no_fixed_point : ∀ x : ℝ, P x ≠ x) : ∀ x : ℝ, P (P x) ≠ x :=
by
  sorry

end no_real_solution_l216_216947


namespace solve_inequality_l216_216942

theorem solve_inequality (x : ℝ) :
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3 / 2 :=
sorry

end solve_inequality_l216_216942


namespace xyz_divides_xyz_squared_l216_216273

theorem xyz_divides_xyz_squared (x y z p : ℕ) (hxyz : x < y ∧ y < z ∧ z < p) (hp : Nat.Prime p) (hx3 : x^3 ≡ y^3 [MOD p])
    (hy3 : y^3 ≡ z^3 [MOD p]) (hz3 : z^3 ≡ x^3 [MOD p]) : (x + y + z) ∣ (x^2 + y^2 + z^2) :=
by
  sorry

end xyz_divides_xyz_squared_l216_216273


namespace circle_equation_l216_216853

theorem circle_equation 
  (P : ℝ × ℝ)
  (h1 : ∀ a : ℝ, (1 - a) * 2 + (P.snd) + 2 * a - 1 = 0)
  (h2 : P = (2, -1)) :
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end circle_equation_l216_216853


namespace problem_solution_l216_216611

def prop_p (a b c : ℝ) : Prop := a < b → a * c^2 < b * c^2

def prop_q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

theorem problem_solution : (p ∨ ¬q) := sorry

end problem_solution_l216_216611


namespace find_x_l216_216571

theorem find_x (x : ℝ) (h : (1 / 2) * x + (1 / 3) * x = (1 / 4) * x + 7) : x = 12 :=
by
  sorry

end find_x_l216_216571


namespace bert_average_words_in_crossword_l216_216280

theorem bert_average_words_in_crossword :
  (10 * 35 + 4 * 65) / (10 + 4) = 43.57 :=
by
  sorry

end bert_average_words_in_crossword_l216_216280


namespace num_triangles_with_perimeter_20_l216_216199

theorem num_triangles_with_perimeter_20 : 
  ∃ (triangles : List (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → a + b + c = 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
    triangles.length = 8 :=
sorry

end num_triangles_with_perimeter_20_l216_216199


namespace part1_part2_l216_216254

theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) :
  a*b ≤ 1 := sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) (hba : b > a) :
  1/a^3 - 1/b^3 ≥ 3 * (1/a - 1/b) := sorry

end part1_part2_l216_216254


namespace thirds_side_length_valid_l216_216670

theorem thirds_side_length_valid (x : ℝ) (h1 : x > 5) (h2 : x < 13) : x = 12 :=
sorry

end thirds_side_length_valid_l216_216670


namespace sol_sells_more_candy_each_day_l216_216226

variable {x : ℕ}

-- Definition of the conditions
def sells_candy (first_day : ℕ) (rate : ℕ) (days : ℕ) : ℕ :=
  first_day + rate * (days - 1) * days / 2

def earns (bars_sold : ℕ) (price_cents : ℕ) : ℕ :=
  bars_sold * price_cents

-- Problem statement in Lean:
theorem sol_sells_more_candy_each_day
  (first_day_sales : ℕ := 10)
  (days : ℕ := 6)
  (price_cents : ℕ := 10)
  (total_earnings : ℕ := 1200) :
  earns (sells_candy first_day_sales x days) price_cents = total_earnings → x = 76 :=
sorry

end sol_sells_more_candy_each_day_l216_216226


namespace range_of_a_l216_216650

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 3 → deriv (f a) x ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end range_of_a_l216_216650


namespace find_b9_l216_216523

theorem find_b9 {b : ℕ → ℕ} 
  (h1 : ∀ n, b (n + 2) = b (n + 1) + b n)
  (h2 : b 8 = 100) :
  b 9 = 194 :=
sorry

end find_b9_l216_216523


namespace Mabel_marble_count_l216_216195

variable (K A M : ℕ)

axiom Amanda_condition : A + 12 = 2 * K
axiom Mabel_K_condition : M = 5 * K
axiom Mabel_A_condition : M = A + 63

theorem Mabel_marble_count : M = 85 := by
  sorry

end Mabel_marble_count_l216_216195


namespace lines_per_stanza_l216_216256

-- Define the number of stanzas
def num_stanzas : ℕ := 20

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Theorem statement to prove the number of lines per stanza
theorem lines_per_stanza : 
  (total_words / words_per_line) / num_stanzas = 10 := 
by sorry

end lines_per_stanza_l216_216256


namespace kate_collected_money_l216_216098

-- Define the conditions
def wand_cost : ℕ := 60
def num_wands_bought : ℕ := 3
def extra_charge : ℕ := 5
def num_wands_sold : ℕ := 2

-- Define the selling price per wand
def selling_price_per_wand : ℕ := wand_cost + extra_charge

-- Define the total amount collected from the sale
def total_collected : ℕ := num_wands_sold * selling_price_per_wand

-- Prove that the total collected is $130
theorem kate_collected_money :
  total_collected = 130 :=
sorry

end kate_collected_money_l216_216098


namespace evaluate_expression_l216_216378

theorem evaluate_expression :
  (3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7) = (6^5 + 3^7) :=
sorry

end evaluate_expression_l216_216378


namespace john_more_needed_l216_216865

def john_needs : ℝ := 2.5
def john_has : ℝ := 0.75
def john_needs_more : ℝ := 1.75

theorem john_more_needed : (john_needs - john_has) = john_needs_more :=
by
  sorry

end john_more_needed_l216_216865


namespace easter_egg_problem_l216_216067

-- Define the conditions as assumptions
def total_eggs : Nat := 63
def helen_eggs (H : Nat) := H
def hannah_eggs (H : Nat) := 2 * H
def harry_eggs (H : Nat) := 2 * H + 3

-- The theorem stating the proof problem
theorem easter_egg_problem (H : Nat) (hh : hannah_eggs H + helen_eggs H + harry_eggs H = total_eggs) : 
    helen_eggs H = 12 ∧ hannah_eggs H = 24 ∧ harry_eggs H = 27 :=
sorry -- Proof is omitted

end easter_egg_problem_l216_216067


namespace total_heads_l216_216119

theorem total_heads (h : ℕ) (c : ℕ) (total_feet : ℕ) 
  (h_count : h = 30)
  (hen_feet : h * 2 + c * 4 = total_feet)
  (total_feet_val : total_feet = 140) 
  : h + c = 50 :=
by
  sorry

end total_heads_l216_216119


namespace find_ordered_pair_l216_216384

theorem find_ordered_pair (x y : ℝ) (h : (x - 2 * y)^2 + (y - 1)^2 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end find_ordered_pair_l216_216384


namespace balanced_scale_l216_216463

def children's_book_weight : ℝ := 1.1

def weight1 : ℝ := 0.5
def weight2 : ℝ := 0.3
def weight3 : ℝ := 0.3

theorem balanced_scale :
  (weight1 + weight2 + weight3) = children's_book_weight :=
by
  sorry

end balanced_scale_l216_216463


namespace coconut_grove_produce_trees_l216_216728

theorem coconut_grove_produce_trees (x : ℕ)
  (h1 : 60 * (x + 3) + 120 * x + 180 * (x - 3) = 100 * 3 * x)
  : x = 6 := sorry

end coconut_grove_produce_trees_l216_216728


namespace cost_of_one_of_the_shirts_l216_216439

theorem cost_of_one_of_the_shirts
    (total_cost : ℕ) 
    (cost_two_shirts : ℕ) 
    (num_equal_shirts : ℕ) 
    (cost_of_shirt : ℕ) :
    total_cost = 85 → 
    cost_two_shirts = 20 → 
    num_equal_shirts = 3 → 
    cost_of_shirt = (total_cost - 2 * cost_two_shirts) / num_equal_shirts → 
    cost_of_shirt = 15 :=
by
  intros
  sorry

end cost_of_one_of_the_shirts_l216_216439


namespace sophia_age_in_three_years_l216_216213

def current_age_jeremy : Nat := 40
def current_age_sebastian : Nat := current_age_jeremy + 4

def sum_ages_in_three_years (age_jeremy age_sebastian age_sophia : Nat) : Nat :=
  (age_jeremy + 3) + (age_sebastian + 3) + (age_sophia + 3)

theorem sophia_age_in_three_years (age_sophia : Nat) 
  (h1 : sum_ages_in_three_years current_age_jeremy current_age_sebastian age_sophia = 150) :
  age_sophia + 3 = 60 := by
  sorry

end sophia_age_in_three_years_l216_216213


namespace compute_expression_l216_216180

theorem compute_expression : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : Int) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := 
by 
  sorry

end compute_expression_l216_216180


namespace num_valid_pairs_equals_four_l216_216732

theorem num_valid_pairs_equals_four 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) (hba : b > a)
  (hcond : a * b = 3 * (a - 4) * (b - 4)) :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s → p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧
      p.1 * p.2 = 3 * (p.1 - 4) * (p.2 - 4) := sorry

end num_valid_pairs_equals_four_l216_216732


namespace green_balloons_correct_l216_216238

-- Defining the quantities
def total_balloons : ℕ := 67
def red_balloons : ℕ := 29
def blue_balloons : ℕ := 21

-- Calculating the green balloons
def green_balloons : ℕ := total_balloons - red_balloons - blue_balloons

-- The theorem we want to prove
theorem green_balloons_correct : green_balloons = 17 :=
by
  -- proof goes here
  sorry

end green_balloons_correct_l216_216238


namespace solve_quadratic_eq_l216_216218

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l216_216218


namespace find_y_l216_216690

theorem find_y 
  (x y z : ℕ) 
  (h₁ : x + y + z = 25)
  (h₂ : x + y = 19) 
  (h₃ : y + z = 18) :
  y = 12 :=
by
  sorry

end find_y_l216_216690


namespace count_non_congruent_rectangles_l216_216309

-- Definitions based on conditions given in the problem
def is_rectangle (w h : ℕ) : Prop := 2 * (w + h) = 40 ∧ w % 2 = 0

-- Theorem that we need to prove based on the problem statement
theorem count_non_congruent_rectangles : 
  ∃ n : ℕ, n = 9 ∧ 
  (∀ p : ℕ × ℕ, p ∈ { p | is_rectangle p.1 p.2 } → ∀ q : ℕ × ℕ, q ∈ { q | is_rectangle q.1 q.2 } → p = q ∨ p ≠ q) := 
sorry

end count_non_congruent_rectangles_l216_216309


namespace find_a_l216_216174

theorem find_a (a : ℝ) : (∃ k : ℝ, (x - 2) * (x + k) = x^2 + a * x - 5) ↔ a = 1 / 2 :=
by
  sorry

end find_a_l216_216174


namespace alpine_school_math_students_l216_216354

theorem alpine_school_math_students (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) :
  total_players = 15 → physics_players = 9 → both_players = 4 → 
  ∃ math_players : ℕ, math_players = total_players - (physics_players - both_players) + both_players := by
  sorry

end alpine_school_math_students_l216_216354


namespace simplify_frac_48_72_l216_216125

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l216_216125


namespace no_valid_prime_l216_216394

open Nat

def base_p_polynomial (p : ℕ) (coeffs : List ℕ) : ℕ → ℕ :=
  fun (n : ℕ) => coeffs.foldl (λ sum coef => sum * p + coef) 0

def num_1013 (p : ℕ) := base_p_polynomial p [1, 0, 1, 3]
def num_207 (p : ℕ) := base_p_polynomial p [2, 0, 7]
def num_214 (p : ℕ) := base_p_polynomial p [2, 1, 4]
def num_100 (p : ℕ) := base_p_polynomial p [1, 0, 0]
def num_10 (p : ℕ) := base_p_polynomial p [1, 0]

def num_321 (p : ℕ) := base_p_polynomial p [3, 2, 1]
def num_403 (p : ℕ) := base_p_polynomial p [4, 0, 3]
def num_210 (p : ℕ) := base_p_polynomial p [2, 1, 0]

theorem no_valid_prime (p : ℕ) [Fact (Nat.Prime p)] :
  num_1013 p + num_207 p + num_214 p + num_100 p + num_10 p ≠
  num_321 p + num_403 p + num_210 p := by
  sorry

end no_valid_prime_l216_216394


namespace figure_side_length_l216_216116

theorem figure_side_length (number_of_sides : ℕ) (perimeter : ℝ) (length_of_one_side : ℝ) 
  (h1 : number_of_sides = 8) (h2 : perimeter = 23.6) : length_of_one_side = 2.95 :=
by
  sorry

end figure_side_length_l216_216116


namespace parabola_tangent_sequence_l216_216259

noncomputable def geom_seq_sum (a2 : ℕ) : ℕ :=
  a2 + a2 / 4 + a2 / 16

theorem parabola_tangent_sequence (a2 : ℕ) (h : a2 = 32) : geom_seq_sum a2 = 42 :=
by
  rw [h]
  norm_num
  sorry

end parabola_tangent_sequence_l216_216259


namespace other_divisor_l216_216912

theorem other_divisor (x : ℕ) (h₁ : 261 % 7 = 2) (h₂ : 261 % x = 2) : x = 259 :=
sorry

end other_divisor_l216_216912


namespace fraction_product_equals_l216_216266

theorem fraction_product_equals :
  (7 / 4) * (14 / 49) * (10 / 15) * (12 / 36) * (21 / 14) * (40 / 80) * (33 / 22) * (16 / 64) = 1 / 12 := 
  sorry

end fraction_product_equals_l216_216266


namespace fair_attendance_l216_216201

-- Define the variables x, y, and z
variables (x y z : ℕ)

-- Define the conditions given in the problem
def condition1 := z = 2 * y
def condition2 := x = z - 200
def condition3 := y = 600

-- State the main theorem proving the values of x, y, and z
theorem fair_attendance : condition1 y z → condition2 x z → condition3 y → (x = 1000 ∧ y = 600 ∧ z = 1200) := by
  intros h1 h2 h3
  sorry

end fair_attendance_l216_216201


namespace azalea_wool_price_l216_216243

noncomputable def sheep_count : ℕ := 200
noncomputable def wool_per_sheep : ℕ := 10
noncomputable def shearing_cost : ℝ := 2000
noncomputable def profit : ℝ := 38000

-- Defining total wool and total revenue based on these definitions
noncomputable def total_wool : ℕ := sheep_count * wool_per_sheep
noncomputable def total_revenue : ℝ := profit + shearing_cost
noncomputable def price_per_pound : ℝ := total_revenue / total_wool

-- Problem statement: Proving that the price per pound of wool is equal to $20
theorem azalea_wool_price :
  price_per_pound = 20 := 
sorry

end azalea_wool_price_l216_216243


namespace joann_lollipops_l216_216898

theorem joann_lollipops : 
  ∃ (a : ℚ), 
  (7 * a  + 3 * (1 + 2 + 3 + 4 + 5 + 6) = 150) ∧ 
  (a_4 = a + 9) ∧ 
  (a_4 = 150 / 7) :=
by
  sorry

end joann_lollipops_l216_216898


namespace find_P_and_Q_l216_216793

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l216_216793


namespace total_pies_baked_in_7_days_l216_216747

-- Define the baking rates (pies per day)
def Eddie_rate : Nat := 3
def Sister_rate : Nat := 6
def Mother_rate : Nat := 8

-- Define the duration in days
def duration : Nat := 7

-- Define the total number of pies baked in 7 days
def total_pies : Nat := Eddie_rate * duration + Sister_rate * duration + Mother_rate * duration

-- Prove the total number of pies is 119
theorem total_pies_baked_in_7_days : total_pies = 119 := by
  -- The proof will be filled here, adding sorry to skip it for now
  sorry

end total_pies_baked_in_7_days_l216_216747


namespace white_tshirts_per_pack_l216_216561

def packs_of_white := 3
def packs_of_blue := 2
def blue_in_each_pack := 4
def total_tshirts := 26

theorem white_tshirts_per_pack :
  ∃ W : ℕ, packs_of_white * W + packs_of_blue * blue_in_each_pack = total_tshirts ∧ W = 6 :=
by
  sorry

end white_tshirts_per_pack_l216_216561


namespace length_of_second_train_correct_l216_216903

noncomputable def length_of_second_train : ℝ :=
  let speed_first_train := 60 / 3.6
  let speed_second_train := 90 / 3.6
  let relative_speed := speed_first_train + speed_second_train
  let time_to_clear := 6.623470122390208
  let total_distance := relative_speed * time_to_clear
  let length_first_train := 111
  total_distance - length_first_train

theorem length_of_second_train_correct :
  length_of_second_train = 164.978 :=
by
  unfold length_of_second_train
  sorry

end length_of_second_train_correct_l216_216903


namespace rectangle_perimeter_l216_216155

variable (L W : ℝ)

-- Conditions
def width := 70
def length := (7 / 5) * width

-- Perimeter calculation and proof goal
def perimeter (L W : ℝ) := 2 * (L + W)

theorem rectangle_perimeter : perimeter (length) (width) = 336 := by
  sorry

end rectangle_perimeter_l216_216155


namespace nature_of_roots_l216_216349

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 + 3 * x^2 - 8 * x + 16

theorem nature_of_roots : (∀ x : ℝ, x < 0 → P x > 0) ∧ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ P x = 0 := 
by
  sorry

end nature_of_roots_l216_216349


namespace rowing_time_75_minutes_l216_216983

-- Definition of time duration Ethan rowed.
def EthanRowingTime : ℕ := 25  -- minutes

-- Definition of the time duration Frank rowed.
def FrankRowingTime : ℕ := 2 * EthanRowingTime  -- twice as long as Ethan.

-- Definition of the total rowing time.
def TotalRowingTime : ℕ := EthanRowingTime + FrankRowingTime

-- Theorem statement proving the total rowing time is 75 minutes.
theorem rowing_time_75_minutes : TotalRowingTime = 75 := by
  -- The proof is omitted.
  sorry

end rowing_time_75_minutes_l216_216983


namespace find_sticker_price_l216_216045

-- Define the conditions
def storeX_discount (x : ℝ) : ℝ := 0.80 * x - 70
def storeY_discount (x : ℝ) : ℝ := 0.70 * x

-- Define the main statement
theorem find_sticker_price (x : ℝ) (h : storeX_discount x = storeY_discount x - 20) : x = 500 :=
sorry

end find_sticker_price_l216_216045


namespace vegetable_difference_is_30_l216_216120

def initial_tomatoes : Int := 17
def initial_carrots : Int := 13
def initial_cucumbers : Int := 8
def initial_bell_peppers : Int := 15
def initial_radishes : Int := 0

def picked_tomatoes : Int := 5
def picked_carrots : Int := 6
def picked_cucumbers : Int := 3
def picked_bell_peppers : Int := 8

def given_neighbor1_tomatoes : Int := 3
def given_neighbor1_carrots : Int := 2

def exchanged_neighbor2_tomatoes : Int := 2
def exchanged_neighbor2_cucumbers : Int := 3
def exchanged_neighbor2_radishes : Int := 5

def given_neighbor3_bell_peppers : Int := 3

noncomputable def initial_total := 
  initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers + initial_radishes

noncomputable def remaining_after_picking :=
  (initial_tomatoes - picked_tomatoes) +
  (initial_carrots - picked_carrots) +
  (initial_cucumbers - picked_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers)

noncomputable def remaining_after_exchanges :=
  ((initial_tomatoes - picked_tomatoes - given_neighbor1_tomatoes - exchanged_neighbor2_tomatoes) +
  (initial_carrots - picked_carrots - given_neighbor1_carrots) +
  (initial_cucumbers - picked_cucumbers - exchanged_neighbor2_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers - given_neighbor3_bell_peppers) +
  exchanged_neighbor2_radishes)

noncomputable def remaining_total := remaining_after_exchanges

noncomputable def total_difference := initial_total - remaining_total

theorem vegetable_difference_is_30 : total_difference = 30 := by
  sorry

end vegetable_difference_is_30_l216_216120


namespace jeff_cats_count_l216_216768

theorem jeff_cats_count :
  let initial_cats := 20
  let found_monday := 2 + 3
  let found_tuesday := 1 + 2
  let adopted_wednesday := 4 * 2
  let adopted_thursday := 3
  let found_friday := 3
  initial_cats + found_monday + found_tuesday - adopted_wednesday - adopted_thursday + found_friday = 20 := by
  sorry

end jeff_cats_count_l216_216768


namespace speed_in_still_water_l216_216433

-- Define variables for speed of the boy in still water and speed of the stream.
variables (v s : ℝ)

-- Define the conditions as Lean statements
def downstream_condition (v s : ℝ) : Prop := (v + s) * 7 = 91
def upstream_condition (v s : ℝ) : Prop := (v - s) * 7 = 21

-- The theorem to prove that the speed of the boy in still water is 8 km/h given the conditions
theorem speed_in_still_water
  (h1 : downstream_condition v s)
  (h2 : upstream_condition v s) :
  v = 8 := 
sorry

end speed_in_still_water_l216_216433


namespace total_scissors_l216_216118

def initial_scissors : ℕ := 54
def added_scissors : ℕ := 22

theorem total_scissors : initial_scissors + added_scissors = 76 :=
by
  sorry

end total_scissors_l216_216118


namespace Jonas_initial_socks_l216_216721

noncomputable def pairsOfSocks(Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                              (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) : ℕ :=
    let individualShoes := Jonas_pairsOfShoes * 2
    let individualPants := Jonas_pairsOfPants * 2
    let individualTShirts := Jonas_tShirts
    let totalWithoutSocks := individualShoes + individualPants + individualTShirts
    let totalToDouble := (totalWithoutSocks + Jonas_pairsOfNewSocks * 2) / 2
    (totalToDouble * 2 - totalWithoutSocks) / 2

theorem Jonas_initial_socks (Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                             (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) 
                             (h1 : Jonas_pairsOfShoes = 5)
                             (h2 : Jonas_pairsOfPants = 10)
                             (h3 : Jonas_tShirts = 10)
                             (h4 : Jonas_pairsOfNewSocks = 35) :
    pairsOfSocks Jonas_pairsOfShoes Jonas_pairsOfPants Jonas_tShirts Jonas_pairsOfNewSocks = 15 :=
by
    subst h1
    subst h2
    subst h3
    subst h4
    sorry

end Jonas_initial_socks_l216_216721


namespace distance_between_cars_l216_216028

theorem distance_between_cars (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) :
  t = 1 ∧ v_kmh = 180 ∧ v_ms = v_kmh * 1000 / 3600 → 
  v_ms * t = 50 := 
by 
  sorry

end distance_between_cars_l216_216028


namespace solution_set_of_inequality_l216_216681

noncomputable def f (x : ℝ) : ℝ := (1 / x) * (1 / 2 * (Real.log x) ^ 2 + 1 / 2)

theorem solution_set_of_inequality :
  (∀ x : ℝ, x > 0 → x < e → f x - x > f e - e) ↔ (∀ x : ℝ, 0 < x ∧ x < e) :=
by
  sorry

end solution_set_of_inequality_l216_216681


namespace trey_total_hours_l216_216535

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end trey_total_hours_l216_216535


namespace domain_of_f_l216_216414

theorem domain_of_f :
  {x : ℝ | x > -1 ∧ x ≠ 0 ∧ x ≤ 2} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l216_216414


namespace cost_price_of_ball_l216_216819

theorem cost_price_of_ball (x : ℕ) (h : 13 * x = 720 + 5 * x) : x = 90 :=
by sorry

end cost_price_of_ball_l216_216819


namespace same_terminal_side_l216_216469

theorem same_terminal_side (k : ℤ): ∃ k : ℤ, 1303 = k * 360 - 137 := by
  -- Proof left as an exercise.
  sorry

end same_terminal_side_l216_216469


namespace problem_part_1_problem_part_2_l216_216847

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x - 1

theorem problem_part_1 (m n : ℝ) :
  (∀ x, f x m < 0 ↔ -2 < x ∧ x < n) → m = 5 / 2 ∧ n = 1 / 2 :=
sorry

theorem problem_part_2 (m : ℝ) :
  (∀ x, m ≤ x ∧ x ≤ m + 1 → f x m < 0) → m ∈ Set.Ioo (-Real.sqrt (2) / 2) 0 :=
sorry

end problem_part_1_problem_part_2_l216_216847


namespace proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l216_216274

noncomputable def sin_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A + Real.sin B + Real.sin C) ≤ (3 / 2) * Real.sqrt 3

noncomputable def sin_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A * Real.sin B * Real.sin C) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_sum_double_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C)) ≥ (-3 / 2)

noncomputable def cos_square_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2) ≥ (3 / 4)

noncomputable def cos_half_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A * Real.cos B * Real.cos C) ≤ (1 / 8)

theorem proof_sin_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_sum_ineq A B C hABC := sorry

theorem proof_sin_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_product_ineq A B C hABC := sorry

theorem proof_cos_sum_double_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_sum_double_ineq A B C hABC := sorry

theorem proof_cos_square_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_square_sum_ineq A B C hABC := sorry

theorem proof_cos_half_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_half_product_ineq A B C hABC := sorry

theorem proof_cos_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_product_ineq A B C hABC := sorry

end proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l216_216274


namespace solve_inequality_l216_216761

theorem solve_inequality (x : ℝ) : 
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 ↔ 
  3 < x ∧ x < 17 / 3 :=
by
  sorry

end solve_inequality_l216_216761


namespace weekly_milk_production_l216_216712

theorem weekly_milk_production 
  (bess_milk_per_day : ℕ) 
  (brownie_milk_per_day : ℕ) 
  (daisy_milk_per_day : ℕ) 
  (total_milk_per_day : ℕ) 
  (total_milk_per_week : ℕ) 
  (h1 : bess_milk_per_day = 2) 
  (h2 : brownie_milk_per_day = 3 * bess_milk_per_day) 
  (h3 : daisy_milk_per_day = bess_milk_per_day + 1) 
  (h4 : total_milk_per_day = bess_milk_per_day + brownie_milk_per_day + daisy_milk_per_day)
  (h5 : total_milk_per_week = total_milk_per_day * 7) : 
  total_milk_per_week = 77 := 
by sorry

end weekly_milk_production_l216_216712


namespace product_of_two_numbers_l216_216607

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 :=
sorry

end product_of_two_numbers_l216_216607


namespace ozverin_concentration_after_5_times_l216_216145

noncomputable def ozverin_concentration (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ) : ℝ :=
  C₀ * (1 - v / V) ^ n

theorem ozverin_concentration_after_5_times :
  ∀ (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ), V = 0.5 → C₀ = 0.4 → v = 50 → n = 5 →
  ozverin_concentration V C₀ v n = 0.236196 :=
by
  intros V C₀ v n hV hC₀ hv hn
  rw [hV, hC₀, hv, hn]
  simp only [ozverin_concentration]
  norm_num
  sorry

end ozverin_concentration_after_5_times_l216_216145


namespace chi_square_relationship_l216_216114

noncomputable def chi_square_statistic {X Y : Type*} (data : X → Y → ℝ) : ℝ := 
  sorry -- Actual definition is omitted for simplicity.

theorem chi_square_relationship (X Y : Type*) (data : X → Y → ℝ) :
  ( ∀ Χ2 : ℝ, Χ2 = chi_square_statistic data →
  (Χ2 = 0 → ∃ (credible : Prop), ¬credible)) → 
  (Χ2 > 0 → ∃ (credible : Prop), credible) :=
sorry

end chi_square_relationship_l216_216114


namespace common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l216_216517

-- a) Prove the statements about the number of weeks and extra days
theorem common_year_has_52_weeks_1_day: 
  ∀ (days_in_common_year : ℕ), 
  days_in_common_year = 365 → 
  (days_in_common_year / 7 = 52 ∧ days_in_common_year % 7 = 1)
:= by
  sorry

theorem leap_year_has_52_weeks_2_days: 
  ∀ (days_in_leap_year : ℕ), 
  days_in_leap_year = 366 → 
  (days_in_leap_year / 7 = 52 ∧ days_in_leap_year % 7 = 2)
:= by
  sorry

-- b) If a common year starts on a Tuesday, prove the following year starts on a Wednesday
theorem next_year_starts_on_wednesday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (365 % 7 = 1) → 
  ((start_day + 365 % 7) % 7 = 3)
:= by
  sorry

-- c) If a leap year starts on a Tuesday, prove the following year starts on a Thursday
theorem next_year_starts_on_thursday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (366 % 7 = 2) →
  ((start_day + 366 % 7) % 7 = 4)
:= by
  sorry

end common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l216_216517


namespace min_valid_subset_card_eq_l216_216632

open Finset

def pairs (n : ℕ) : Finset (ℕ × ℕ) := 
  (range n).product (range n)

def valid_subset (X : Finset (ℕ × ℕ)) (n : ℕ) : Prop :=
  ∀ (seq : ℕ → ℕ), ∃ k, (seq k, seq (k+1)) ∈ X

theorem min_valid_subset_card_eq (n : ℕ) (h : n = 10) : 
  ∃ X : Finset (ℕ × ℕ), valid_subset X n ∧ X.card = 55 := 
by 
  sorry

end min_valid_subset_card_eq_l216_216632


namespace max_gross_profit_price_l216_216337

def purchase_price : ℝ := 20
def Q (P : ℝ) : ℝ := 8300 - 170 * P - P^2
def L (P : ℝ) : ℝ := (8300 - 170 * P - P^2) * (P - 20)

theorem max_gross_profit_price : ∃ P : ℝ, (∀ x : ℝ, L x ≤ L P) ∧ P = 30 :=
by
  sorry

end max_gross_profit_price_l216_216337


namespace initial_average_runs_l216_216625

theorem initial_average_runs (A : ℕ) (h : 10 * A + 87 = 11 * (A + 5)) : A = 32 :=
by
  sorry

end initial_average_runs_l216_216625


namespace f_at_6_5_l216_216699

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem f_at_6_5:
  (∀ x : ℝ, f (x + 2) = -1 / f x) →
  even_function f →
  specific_values f →
  f 6.5 = -0.5 :=
by
  sorry

end f_at_6_5_l216_216699


namespace table_height_l216_216985

theorem table_height
  (l d h : ℤ)
  (h_eq1 : l + h - d = 36)
  (h_eq2 : 2 * l + h = 46)
  (l_eq_d : l = d) :
  h = 36 :=
by
  sorry

end table_height_l216_216985


namespace hyperbola_equation_l216_216958

-- Conditions
def center_origin (P : ℝ × ℝ) : Prop := P = (0, 0)
def focus_at (F : ℝ × ℝ) : Prop := F = (0, Real.sqrt 3)
def vertex_distance (d : ℝ) : Prop := d = Real.sqrt 3 - 1

-- Statement
theorem hyperbola_equation
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (d : ℝ)
  (h_center : center_origin center)
  (h_focus : focus_at focus)
  (h_vert_dist : vertex_distance d) :
  y^2 - (x^2 / 2) = 1 := 
sorry

end hyperbola_equation_l216_216958


namespace percentage_of_part_l216_216277

theorem percentage_of_part (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 50) : (Part / Whole) * 100 = 240 := 
by
  sorry

end percentage_of_part_l216_216277


namespace dvd_blu_ratio_l216_216892

theorem dvd_blu_ratio (D B : ℕ) (h1 : D + B = 378) (h2 : (D : ℚ) / (B - 4 : ℚ) = 9 / 2) :
  D / Nat.gcd D B = 51 ∧ B / Nat.gcd D B = 12 :=
by
  sorry

end dvd_blu_ratio_l216_216892


namespace range_of_k_l216_216362

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end range_of_k_l216_216362


namespace probability_opposite_vertex_l216_216293

theorem probability_opposite_vertex (k : ℕ) (h : k > 0) : 
    P_k = (1 / 6 : ℝ) + (1 / (3 * (-2) ^ k) : ℝ) := 
sorry

end probability_opposite_vertex_l216_216293


namespace subset_interval_l216_216074

theorem subset_interval (a : ℝ) : 
  (∀ x : ℝ, (-a-1 < x ∧ x < -a+1 → -3 < x ∧ x < 1)) ↔ (0 ≤ a ∧ a ≤ 2) := 
by
  sorry

end subset_interval_l216_216074


namespace john_moves_3594_pounds_l216_216742

def bench_press_weight := 15
def bench_press_reps := 10
def bench_press_sets := 3

def bicep_curls_weight := 12
def bicep_curls_reps := 8
def bicep_curls_sets := 4

def squats_weight := 50
def squats_reps := 12
def squats_sets := 3

def deadlift_weight := 80
def deadlift_reps := 6
def deadlift_sets := 2

def total_weight_moved : Nat :=
  (bench_press_weight * bench_press_reps * bench_press_sets) +
  (bicep_curls_weight * bicep_curls_reps * bicep_curls_sets) +
  (squats_weight * squats_reps * squats_sets) +
  (deadlift_weight * deadlift_reps * deadlift_sets)

theorem john_moves_3594_pounds :
  total_weight_moved = 3594 := by {
    sorry
}

end john_moves_3594_pounds_l216_216742


namespace percent_difference_l216_216389

variable (w e y z : ℝ)

-- Definitions based on the given conditions
def condition1 : Prop := w = 0.60 * e
def condition2 : Prop := e = 0.60 * y
def condition3 : Prop := z = 0.54 * y

-- Statement of the theorem to prove
theorem percent_difference (h1 : condition1 w e) (h2 : condition2 e y) (h3 : condition3 z y) : 
  (z - w) / w * 100 = 50 := 
by
  sorry

end percent_difference_l216_216389


namespace rational_product_nonpositive_l216_216908

open Classical

theorem rational_product_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 :=
by
  sorry

end rational_product_nonpositive_l216_216908


namespace initial_action_figures_l216_216450

theorem initial_action_figures (x : ℕ) (h1 : x + 2 = 10) : x = 8 := 
by sorry

end initial_action_figures_l216_216450


namespace mass_of_substance_l216_216570

-- The conditions
def substance_density (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) : Prop :=
  mass_cubic_meter_kg = 100 ∧ volume_cubic_meter_cm3 = 1*1000000

def specific_amount_volume_cm3 (volume_cm3 : ℝ) : Prop :=
  volume_cm3 = 10

-- The Proof Statement
theorem mass_of_substance (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) (volume_cm3 : ℝ) (mass_grams : ℝ) :
  substance_density mass_cubic_meter_kg volume_cubic_meter_cm3 →
  specific_amount_volume_cm3 volume_cm3 →
  mass_grams = 10 :=
by
  intros hDensity hVolume
  sorry

end mass_of_substance_l216_216570


namespace area_of_triangle_BXC_l216_216030

/-
  Given:
  - AB = 15 units
  - CD = 40 units
  - The area of trapezoid ABCD = 550 square units

  To prove:
  - The area of triangle BXC = 1200 / 11 square units
-/
theorem area_of_triangle_BXC 
  (AB CD : ℝ) 
  (hAB : AB = 15) 
  (hCD : CD = 40) 
  (area_ABCD : ℝ)
  (hArea_ABCD : area_ABCD = 550) 
  : ∃ (area_BXC : ℝ), area_BXC = 1200 / 11 :=
by
  sorry

end area_of_triangle_BXC_l216_216030


namespace union_sets_l216_216893

-- Define the sets A and B based on their conditions
def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

-- Statement of the proof problem
theorem union_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x | -1 ≤ x ∧ x < 9 }) := sorry

end union_sets_l216_216893


namespace discount_calculation_l216_216671

noncomputable def cost_price : ℝ := 180
noncomputable def markup_percentage : ℝ := 0.4778
noncomputable def profit_percentage : ℝ := 0.20

noncomputable def marked_price (CP : ℝ) (MP_percent : ℝ) : ℝ := CP + (MP_percent * CP)
noncomputable def selling_price (CP : ℝ) (PP_percent : ℝ) : ℝ := CP + (PP_percent * CP)
noncomputable def discount (MP : ℝ) (SP : ℝ) : ℝ := MP - SP

theorem discount_calculation :
  discount (marked_price cost_price markup_percentage) (selling_price cost_price profit_percentage) = 50.004 :=
by
  sorry

end discount_calculation_l216_216671


namespace max_value_l216_216760

variable (x y : ℝ)

def condition : Prop := 2 * x ^ 2 + x * y - y ^ 2 = 1

noncomputable def expression : ℝ := (x - 2 * y) / (5 * x ^ 2 - 2 * x * y + 2 * y ^ 2)

theorem max_value : ∀ x y : ℝ, condition x y → expression x y ≤ (Real.sqrt 2) / 4 :=
by
  sorry

end max_value_l216_216760


namespace distance_from_point_to_line_l216_216530

open Real

noncomputable def point_to_line_distance (a b c x0 y0 : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

theorem distance_from_point_to_line (a b c x0 y0 : ℝ) :
  point_to_line_distance a b c x0 y0 = abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2) :=
by
  sorry

end distance_from_point_to_line_l216_216530


namespace water_speed_l216_216162

theorem water_speed (v : ℝ) (h1 : 4 - v > 0) (h2 : 6 * (4 - v) = 12) : v = 2 :=
by
  -- proof steps
  sorry

end water_speed_l216_216162


namespace find_m_l216_216842

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def symmetric_about_line (x1 y1 x2 y2 m : ℝ) : Prop := (y1 - y2) / (x1 - x2) = -1
def product_y (y1 y2 : ℝ) : Prop := y1 * y2 = -1 / 2

-- Theorem to be proven
theorem find_m 
  (x1 y1 x2 y2 m : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : symmetric_about_line x1 y1 x2 y2 m)
  (h4 : product_y y1 y2) :
  m = 9 / 4 :=
sorry

end find_m_l216_216842


namespace multiplication_is_valid_l216_216606

-- Define that the three-digit number n = 306
def three_digit_number := 306

-- The multiplication by 1995 should result in the defined product
def valid_multiplication (n : ℕ) := 1995 * n

theorem multiplication_is_valid : valid_multiplication three_digit_number = 1995 * 306 := by
  -- Since we only need the statement, we use sorry here
  sorry

end multiplication_is_valid_l216_216606


namespace remainder_of_3_pow_21_mod_11_l216_216339

theorem remainder_of_3_pow_21_mod_11 : (3^21 % 11) = 3 := 
by {
  sorry
}

end remainder_of_3_pow_21_mod_11_l216_216339


namespace max_matrix_det_l216_216990

noncomputable def matrix_det (θ : ℝ) : ℝ :=
  by
    let M := ![
      ![1, 1, 1],
      ![1, 1 + Real.sin θ ^ 2, 1],
      ![1 + Real.cos θ ^ 2, 1, 1]
    ]
    exact Matrix.det M

theorem max_matrix_det : ∃ θ : ℝ, matrix_det θ = 3/4 :=
  sorry

end max_matrix_det_l216_216990


namespace exists_x0_l216_216666

theorem exists_x0 : ∃ x0 : ℝ, x0^2 + 2*x0 + 1 ≤ 0 :=
sorry

end exists_x0_l216_216666


namespace rotated_number_divisibility_l216_216110

theorem rotated_number_divisibility 
  (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h : 7 ∣ (10^5 * a1 + 10^4 * a2 + 10^3 * a3 + 10^2 * a4 + 10 * a5 + a6)) :
  7 ∣ (10^5 * a6 + 10^4 * a1 + 10^3 * a2 + 10^2 * a3 + 10 * a4 + a5) := 
sorry

end rotated_number_divisibility_l216_216110


namespace angle_B_equal_pi_div_3_l216_216827

-- Define the conditions and the statement to be proved
theorem angle_B_equal_pi_div_3 (A B C : ℝ) 
  (h₁ : Real.sin A / Real.sin B = 5 / 7)
  (h₂ : Real.sin B / Real.sin C = 7 / 8) : 
  B = Real.pi / 3 :=
sorry

end angle_B_equal_pi_div_3_l216_216827


namespace substitution_not_sufficient_for_identity_proof_l216_216003

theorem substitution_not_sufficient_for_identity_proof {α : Type} (f g : α → α) :
  (∀ x : α, f x = g x) ↔ ¬ (∀ x, f x = g x ↔ (∃ (c : α), f c ≠ g c)) := by
  sorry

end substitution_not_sufficient_for_identity_proof_l216_216003


namespace shuttle_speeds_l216_216141

def speed_at_altitude (speed_per_sec : ℕ) : ℕ :=
  speed_per_sec * 3600

theorem shuttle_speeds (speed_300 speed_800 avg_speed : ℕ) :
  speed_at_altitude 7 = 25200 ∧ 
  speed_at_altitude 6 = 21600 ∧ 
  avg_speed = (25200 + 21600) / 2 ∧ 
  avg_speed = 23400 := 
by
  sorry

end shuttle_speeds_l216_216141


namespace least_positive_multiple_of_13_gt_418_l216_216709

theorem least_positive_multiple_of_13_gt_418 : ∃ (n : ℕ), n > 418 ∧ (13 ∣ n) ∧ n = 429 :=
by
  sorry

end least_positive_multiple_of_13_gt_418_l216_216709


namespace middle_number_consecutive_sum_l216_216688

theorem middle_number_consecutive_sum (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : a + b + c = 30) : b = 10 :=
by
  sorry

end middle_number_consecutive_sum_l216_216688


namespace doughnuts_left_l216_216934

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end doughnuts_left_l216_216934


namespace seeds_germinated_percentage_l216_216770

theorem seeds_germinated_percentage 
  (n1 n2 : ℕ) 
  (p1 p2 : ℝ) 
  (h1 : n1 = 300)
  (h2 : n2 = 200)
  (h3 : p1 = 0.15)
  (h4 : p2 = 0.35) : 
  ( ( p1 * n1 + p2 * n2 ) / ( n1 + n2 ) ) * 100 = 23 :=
by
  -- Mathematical proof goes here.
  sorry

end seeds_germinated_percentage_l216_216770


namespace karthik_weight_average_l216_216484

theorem karthik_weight_average
  (weight : ℝ)
  (hKarthik: 55 < weight )
  (hBrother: weight < 58 )
  (hFather : 56 < weight )
  (hSister: 54 < weight ∧ weight < 57) :
  (56 < weight ∧ weight < 57) → (weight = 56.5) :=
by 
  sorry

end karthik_weight_average_l216_216484


namespace parallel_vectors_x_val_l216_216955

open Real

theorem parallel_vectors_x_val (x : ℝ) :
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (x, 1/2)
  a.1 * b.2 = a.2 * b.1 →
  x = 3/8 := 
by
  intro h
  -- Use this line if you need to skip the proof
  sorry

end parallel_vectors_x_val_l216_216955


namespace sum_of_consecutive_perfect_squares_l216_216321

theorem sum_of_consecutive_perfect_squares (k : ℕ) (h_pos : 0 < k)
  (h_eq : 2 * k^2 + 2 * k + 1 = 181) : k = 9 ∧ (k + 1) = 10 := by
  sorry

end sum_of_consecutive_perfect_squares_l216_216321


namespace sum_exterior_angles_const_l216_216054

theorem sum_exterior_angles_const (n : ℕ) (h : n ≥ 3) : 
  ∃ s : ℝ, s = 360 :=
by
  sorry

end sum_exterior_angles_const_l216_216054


namespace eval_at_neg_five_l216_216298

def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem eval_at_neg_five : f (-5) = 12 :=
by
  sorry

end eval_at_neg_five_l216_216298


namespace direct_proportion_graph_is_straight_line_l216_216419

-- Defining the direct proportion function
def direct_proportion_function (k x : ℝ) : ℝ := k * x

-- Theorem statement
theorem direct_proportion_graph_is_straight_line (k : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, y = direct_proportion_function k x ∧ 
    ∀ (x1 x2 : ℝ), 
    ∃ a b : ℝ, b ≠ 0 ∧ 
    (a * x1 + b * (direct_proportion_function k x1)) = (a * x2 + b * (direct_proportion_function k x2)) :=
by
  sorry

end direct_proportion_graph_is_straight_line_l216_216419


namespace count_multiples_of_30_between_two_multiples_l216_216501

theorem count_multiples_of_30_between_two_multiples : 
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  count = 871 :=
by
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  sorry

end count_multiples_of_30_between_two_multiples_l216_216501


namespace manuscript_page_count_l216_216928

-- Define the main statement
theorem manuscript_page_count
  (P : ℕ)
  (cost_per_page : ℕ := 10)
  (rev1_pages : ℕ := 30)
  (rev2_pages : ℕ := 20)
  (total_cost : ℕ := 1350)
  (cost_rev1 : ℕ := 15)
  (cost_rev2 : ℕ := 20) 
  (remaining_pages_cost : ℕ := 10 * (P - (rev1_pages + rev2_pages))) :
  (remaining_pages_cost + rev1_pages * cost_rev1 + rev2_pages * cost_rev2 = total_cost)
  → P = 100 :=
by
  sorry

end manuscript_page_count_l216_216928


namespace problem_statement_l216_216929

def contrapositive {P Q : Prop} (h : P → Q) : ¬Q → ¬P :=
by sorry

def sufficient_but_not_necessary (P Q : Prop) : (P → Q) ∧ ¬(Q → P) :=
by sorry

def proposition_C (p q : Prop) : ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

def negate_exists (P : ℝ → Prop) : (∃ x : ℝ, P x) → ¬(∀ x : ℝ, ¬P x) :=
by sorry

theorem problem_statement : 
¬ (∀ (P Q : Prop), ¬(P ∧ Q) → (¬P ∨ ¬Q)) :=
by sorry

end problem_statement_l216_216929


namespace new_percentage_water_is_correct_l216_216509

def initial_volume : ℕ := 120
def initial_percentage_water : ℚ := 20 / 100
def added_water : ℕ := 8

def initial_volume_water : ℚ := initial_percentage_water * initial_volume
def initial_volume_wine : ℚ := initial_volume - initial_volume_water
def new_volume_water : ℚ := initial_volume_water + added_water
def new_total_volume : ℚ := initial_volume + added_water

def calculate_new_percentage_water : ℚ :=
  (new_volume_water / new_total_volume) * 100

theorem new_percentage_water_is_correct :
  calculate_new_percentage_water = 25 := 
by
  sorry

end new_percentage_water_is_correct_l216_216509


namespace distance_traveled_on_foot_l216_216007

theorem distance_traveled_on_foot (x y : ℝ) : x + y = 61 ∧ (x / 4 + y / 9 = 9) → x = 16 :=
by {
  sorry
}

end distance_traveled_on_foot_l216_216007


namespace find_x_l216_216316

noncomputable def x_half_y (x y : ℚ) : Prop := x = (1 / 2) * y
noncomputable def y_third_z (y z : ℚ) : Prop := y = (1 / 3) * z

theorem find_x (x y z : ℚ) (h₁ : x_half_y x y) (h₂ : y_third_z y z) (h₃ : z = 100) :
  x = 16 + (2 / 3 : ℚ) :=
by
  sorry

end find_x_l216_216316


namespace cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l216_216662

-- Definitions for angles A, B, and C forming an arithmetic sequence and their sum being 180 degrees
variables {A B C : ℝ}

-- Definitions for side lengths a, b, and c forming a geometric sequence
variables {a b c : ℝ}

-- Question 1: Prove that cos B = 1/2 under the given conditions
theorem cos_B_equals_half 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) : 
  Real.cos B = 1 / 2 :=
sorry

-- Question 2: Prove that sin A * sin C = 3/4 under the given conditions
theorem sin_A_mul_sin_C_equals_three_fourths 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) 
  (h3 : b^2 = a * c) : 
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

end cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l216_216662


namespace mina_numbers_l216_216446

theorem mina_numbers (a b : ℤ) (h1 : 3 * a + 4 * b = 140) (h2 : a = 20 ∨ b = 20) : a = 20 ∧ b = 20 :=
by
  sorry

end mina_numbers_l216_216446


namespace kenya_peanuts_l216_216172

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l216_216172


namespace alcohol_percentage_proof_l216_216982

noncomputable def percentage_alcohol_new_mixture 
  (original_solution_volume : ℕ)
  (percent_A : ℚ)
  (concentration_A : ℚ)
  (percent_B : ℚ)
  (concentration_B : ℚ)
  (percent_C : ℚ)
  (concentration_C : ℚ)
  (water_added_volume : ℕ) : ℚ :=
((original_solution_volume * percent_A * concentration_A) +
 (original_solution_volume * percent_B * concentration_B) +
 (original_solution_volume * percent_C * concentration_C)) /
 (original_solution_volume + water_added_volume) * 100

theorem alcohol_percentage_proof : 
  percentage_alcohol_new_mixture 24 0.30 0.80 0.40 0.90 0.30 0.95 16 = 53.1 := 
by 
  sorry

end alcohol_percentage_proof_l216_216982


namespace power_function_value_at_quarter_l216_216951

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem power_function_value_at_quarter (α : ℝ) (h : f 4 α = 1 / 2) : f (1 / 4) α = 2 := 
  sorry

end power_function_value_at_quarter_l216_216951


namespace total_bricks_calculation_l216_216998

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end total_bricks_calculation_l216_216998


namespace curve_cartesian_equation_chord_length_l216_216156
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * θ.cos, ρ * θ.sin)

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + 1/2 * t, (Real.sqrt 3) / 2 * t)

theorem curve_cartesian_equation :
  ∀ (ρ θ : ℝ), 
    ρ * θ.sin * θ.sin = 8 * θ.cos →
    (ρ * θ.cos) ^ 2 + (ρ * θ.sin) ^ 2 = 
    8 * (ρ * θ.cos) :=
by sorry

theorem chord_length :
  ∀ (t₁ t₂ : ℝ),
    (3 * t₁^2 - 16 * t₁ - 64 = 0) →
    (3 * t₂^2 - 16 * t₂ - 64 = 0) →
    |t₁ - t₂| = (32 / 3) :=
by sorry

end curve_cartesian_equation_chord_length_l216_216156


namespace angle_equiv_470_110_l216_216512

theorem angle_equiv_470_110 : ∃ (k : ℤ), 470 = k * 360 + 110 :=
by
  use 1
  exact rfl

end angle_equiv_470_110_l216_216512


namespace simplify_polynomial_expression_l216_216166

variable {R : Type*} [CommRing R]

theorem simplify_polynomial_expression (x : R) :
  (2 * x^6 + 3 * x^5 + 4 * x^4 + x^3 + x^2 + x + 20) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 2 * x^2 + 5) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - x^2 + 15 := 
by
  sorry

end simplify_polynomial_expression_l216_216166


namespace total_pieces_on_chessboard_l216_216330

-- Given conditions about initial chess pieces and lost pieces.
def initial_pieces_each : Nat := 16
def pieces_lost_arianna : Nat := 3
def pieces_lost_samantha : Nat := 9

-- The remaining pieces for each player.
def remaining_pieces_arianna : Nat := initial_pieces_each - pieces_lost_arianna
def remaining_pieces_samantha : Nat := initial_pieces_each - pieces_lost_samantha

-- The total remaining pieces on the chessboard.
def total_remaining_pieces : Nat := remaining_pieces_arianna + remaining_pieces_samantha

-- The theorem to prove
theorem total_pieces_on_chessboard : total_remaining_pieces = 20 :=
by
  sorry

end total_pieces_on_chessboard_l216_216330


namespace max_f_max_g_pow_f_l216_216849

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^2 + 7 * x + 14)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 5 * x + 10) / (x^2 + 5 * x + 20)

theorem max_f : ∀ x : ℝ, f x ≤ 2 := by
  intro x
  sorry

theorem max_g_pow_f : ∀ x : ℝ, g x ^ f x ≤ 9 := by
  intro x
  sorry

end max_f_max_g_pow_f_l216_216849


namespace find_fraction_l216_216130

theorem find_fraction (x y : ℝ) (hx : 0 < x) (hy : x < y) (h : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt 15 / 3 :=
sorry

end find_fraction_l216_216130


namespace combined_teaching_years_l216_216344

def Adrienne_Yrs : ℕ := 22
def Virginia_Yrs : ℕ := Adrienne_Yrs + 9
def Dennis_Yrs : ℕ := 40

theorem combined_teaching_years :
  Adrienne_Yrs + Virginia_Yrs + Dennis_Yrs = 93 := by
  -- Proof omitted
  sorry

end combined_teaching_years_l216_216344


namespace only_odd_digit_square_l216_216417

def odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d % 2 = 1

theorem only_odd_digit_square (n : ℕ) : n^2 = n → odd_digits n → n = 1 ∨ n = 9 :=
by
  intros
  sorry

end only_odd_digit_square_l216_216417


namespace distance_between_poles_l216_216187

theorem distance_between_poles (length width : ℝ) (num_poles : ℕ) (h_length : length = 90)
  (h_width : width = 40) (h_num_poles : num_poles = 52) : 
  (2 * (length + width)) / (num_poles - 1) = 5.098 := 
by 
  -- Sorry to skip the proof
  sorry

end distance_between_poles_l216_216187


namespace minimum_pyramid_volume_proof_l216_216031

noncomputable def minimum_pyramid_volume (side_length : ℝ) (apex_angle : ℝ) : ℝ :=
  if side_length = 6 ∧ apex_angle = 2 * Real.arcsin (1 / 3 : ℝ) then 5 * Real.sqrt 23 else 0

theorem minimum_pyramid_volume_proof : 
  minimum_pyramid_volume 6 (2 * Real.arcsin (1 / 3)) = 5 * Real.sqrt 23 :=
by
  sorry

end minimum_pyramid_volume_proof_l216_216031


namespace correct_multiplicand_l216_216683

theorem correct_multiplicand (x : ℕ) (h1 : x * 467 = 1925817) : 
  ∃ n : ℕ, n * 467 = 1325813 :=
by
  sorry

end correct_multiplicand_l216_216683


namespace amrita_bakes_cake_next_thursday_l216_216237

theorem amrita_bakes_cake_next_thursday (n m : ℕ) (h1 : n = 5) (h2 : m = 7) : Nat.lcm n m = 35 :=
by
  -- Proof goes here
  sorry

end amrita_bakes_cake_next_thursday_l216_216237


namespace problem1_problem2_l216_216494

-- Definitions from the conditions
def A (x : ℝ) : Prop := -1 < x ∧ x < 3

def B (x m : ℝ) : Prop := x^2 - 2 * m * x + m^2 - 1 < 0

-- Intersection problem
theorem problem1 (h₁ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₂ : ∀ x, B x 3 ↔ (2 < x ∧ x < 4)) :
  ∀ x, (A x ∧ B x 3) ↔ (2 < x ∧ x < 3) := by
  sorry

-- Union problem
theorem problem2 (h₃ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₄ : ∀ x m, B x m ↔ ((x - m)^2 < 1)) :
  ∀ m, (0 ≤ m ∧ m ≤ 2) ↔ (∀ x, A x ∨ B x m → A x) := by
  sorry

end problem1_problem2_l216_216494


namespace num_photos_to_include_l216_216896

-- Define the conditions
def num_preselected_photos : ℕ := 7
def total_choices : ℕ := 56

-- Define the statement to prove
theorem num_photos_to_include : total_choices / num_preselected_photos = 8 :=
by sorry

end num_photos_to_include_l216_216896


namespace proof_statements_l216_216825

theorem proof_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧       -- corresponding to A
  ¬((∃ m : ℕ, 190 = 19 * m) ∧  ¬(∃ k : ℕ, 57 = 19 * k)) ∧  -- corresponding to B
  ¬((∃ p : ℕ, 90 = 30 * p) ∨ (∃ q : ℕ, 65 = 30 * q)) ∧     -- corresponding to C
  ¬((∃ r : ℕ, 33 = 11 * r) ∧ ¬(∃ s : ℕ, 55 = 11 * s)) ∧    -- corresponding to D
  (∃ t : ℕ, 162 = 9 * t) :=                                 -- corresponding to E
by {
  -- Proof steps would go here
  sorry
}

end proof_statements_l216_216825


namespace min_A_max_B_l216_216152

-- Part (a): prove A = 15 is the smallest value satisfying the condition
theorem min_A (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : A = 15 := 
sorry

-- Part (b): prove B = 76 is the largest value satisfying the condition
theorem max_B (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : B = 76 := 
sorry

end min_A_max_B_l216_216152


namespace max_area_circle_center_l216_216869

theorem max_area_circle_center (k : ℝ) :
  (∃ (x y : ℝ), (x + k / 2)^2 + (y + 1)^2 = 1 - 3 / 4 * k^2 ∧ k = 0) →
  x = 0 ∧ y = -1 :=
sorry

end max_area_circle_center_l216_216869


namespace jellybean_probability_l216_216133

theorem jellybean_probability :
  let total_ways := Nat.choose 15 4
  let red_ways := Nat.choose 5 2
  let blue_ways := Nat.choose 3 2
  let favorable_ways := red_ways * blue_ways
  let probability := favorable_ways / total_ways
  probability = (2 : ℚ) / 91 := by
  sorry

end jellybean_probability_l216_216133


namespace movie_ticket_cost_l216_216812

/--
Movie tickets cost a certain amount on a Monday, twice as much on a Wednesday, and five times as much as on Monday on a Saturday. If Glenn goes to the movie theater on Wednesday and Saturday, he spends $35. Prove that the cost of a movie ticket on a Monday is $5.
-/
theorem movie_ticket_cost (M : ℕ) 
  (wednesday_cost : 2 * M = 2 * M)
  (saturday_cost : 5 * M = 5 * M) 
  (total_cost : 2 * M + 5 * M = 35) : 
  M = 5 := 
sorry

end movie_ticket_cost_l216_216812


namespace gertrude_fleas_l216_216038

variables (G M O : ℕ)

def fleas_maud := M = 5 * O
def fleas_olive := O = G / 2
def total_fleas := G + M + O = 40

theorem gertrude_fleas
  (h_maud : fleas_maud M O)
  (h_olive : fleas_olive G O)
  (h_total : total_fleas G M O) :
  G = 10 :=
sorry

end gertrude_fleas_l216_216038


namespace frustum_lateral_surface_area_l216_216169

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) (hh : h = 5) :
  let d := r1 - r2
  let s := Real.sqrt (h^2 + d^2)
  let A := Real.pi * s * (r1 + r2)
  A = 12 * Real.pi * Real.sqrt 41 :=
by
  -- hr1 and hr2 imply that r1 and r2 are constants, therefore d = 8 - 4 = 4
  -- h = 5 and d = 4 imply s = sqrt (5^2 + 4^2) = sqrt 41
  -- The area A is then pi * sqrt 41 * (8 + 4) = 12 * pi * sqrt 41
  sorry

end frustum_lateral_surface_area_l216_216169


namespace minimum_value_MP_MF_l216_216094

noncomputable def min_value (M P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := |dist M P + dist M F|

theorem minimum_value_MP_MF :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) →
  ∀ (F : ℝ × ℝ), (F = (1, 0)) →
  ∀ (P : ℝ × ℝ), (P = (3, 1)) →
  min_value M P F = 4 :=
by
  intros M h_para F h_focus P h_fixed
  rw [min_value]
  sorry

end minimum_value_MP_MF_l216_216094


namespace solve_printer_problem_l216_216159

noncomputable def printer_problem : Prop :=
  let rate_A := 10
  let rate_B := rate_A + 8
  let rate_C := rate_B - 4
  let combined_rate := rate_A + rate_B + rate_C
  let total_minutes := 20
  let total_pages := combined_rate * total_minutes
  total_pages = 840

theorem solve_printer_problem : printer_problem :=
by
  sorry

end solve_printer_problem_l216_216159


namespace displacement_correct_l216_216364

-- Define the initial conditions of the problem
def init_north := 50
def init_east := 70
def init_south := 20
def init_west := 30

-- Define the net movements
def net_north := init_north - init_south
def net_east := init_east - init_west

-- Define the straight-line distance using the Pythagorean theorem
def displacement_AC := (net_north ^ 2 + net_east ^ 2).sqrt

theorem displacement_correct : displacement_AC = 50 := 
by sorry

end displacement_correct_l216_216364


namespace cafeteria_extra_fruits_l216_216242

def red_apples_ordered : ℕ := 43
def green_apples_ordered : ℕ := 32
def oranges_ordered : ℕ := 25
def red_apples_chosen : ℕ := 7
def green_apples_chosen : ℕ := 5
def oranges_chosen : ℕ := 4

def extra_red_apples : ℕ := red_apples_ordered - red_apples_chosen
def extra_green_apples : ℕ := green_apples_ordered - green_apples_chosen
def extra_oranges : ℕ := oranges_ordered - oranges_chosen

def total_extra_fruits : ℕ := extra_red_apples + extra_green_apples + extra_oranges

theorem cafeteria_extra_fruits : total_extra_fruits = 84 := by
  sorry

end cafeteria_extra_fruits_l216_216242


namespace diameter_of_circle_l216_216113

theorem diameter_of_circle {a b c d e f D : ℕ} 
  (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) (h4 : d = 33) (h5 : e = 56) (h6 : f = 65)
  (h_right_triangle1 : a^2 + b^2 = c^2)
  (h_right_triangle2 : d^2 + e^2 = f^2)
  (h_inscribed_triangles : true) -- This represents that both triangles are inscribed in the circle.
: D = 65 :=
sorry

end diameter_of_circle_l216_216113


namespace students_in_class_l216_216652

theorem students_in_class (n : ℕ) (S : ℕ) (h_avg_students : S / n = 14) (h_avg_including_teacher : (S + 45) / (n + 1) = 15) : n = 30 :=
by
  sorry

end students_in_class_l216_216652


namespace inequality_a3_b3_c3_l216_216016

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3 * a * b * c > a * b * (a + b) + b * c * (b + c) + a * c * (a + c) :=
by
  sorry

end inequality_a3_b3_c3_l216_216016


namespace parabola_translation_l216_216311

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := (x - 1) ^ 2 + 5
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 3

-- Statement of the translation problem in Lean 4
theorem parabola_translation :
  ∀ x : ℝ, g x = f (x + 2) - 3 := 
sorry

end parabola_translation_l216_216311


namespace logarithm_equation_l216_216773

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem logarithm_equation (a : ℝ) : 
  (1 / log_base 2 a + 1 / log_base 3 a + 1 / log_base 4 a = 1) → a = 24 :=
by
  sorry

end logarithm_equation_l216_216773


namespace sixth_day_is_wednesday_l216_216895

noncomputable def day_of_week : Type := 
  { d // d ∈ ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] }

def five_fridays_sum_correct (x : ℤ) : Prop :=
  x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75

def first_is_friday (x : ℤ) : Prop :=
  x = 1

def day_of_6th_is_wednesday (d : day_of_week) : Prop :=
  d.1 = "Wednesday"

theorem sixth_day_is_wednesday (x : ℤ) (d : day_of_week) :
  five_fridays_sum_correct x → first_is_friday x → day_of_6th_is_wednesday d :=
by
  sorry

end sixth_day_is_wednesday_l216_216895


namespace sum_of_squares_of_roots_l216_216544

theorem sum_of_squares_of_roots
  (x1 x2 : ℝ) (h : 5 * x1^2 + 6 * x1 - 15 = 0) (h' : 5 * x2^2 + 6 * x2 - 15 = 0) :
  x1^2 + x2^2 = 186 / 25 :=
sorry

end sum_of_squares_of_roots_l216_216544


namespace find_divisor_l216_216887

-- Define the given conditions
def dividend : ℕ := 122
def quotient : ℕ := 6
def remainder : ℕ := 2

-- Define the proof problem to find the divisor
theorem find_divisor : 
  ∃ D : ℕ, dividend = (D * quotient) + remainder ∧ D = 20 :=
by sorry

end find_divisor_l216_216887


namespace greatest_candies_to_office_l216_216431

-- Problem statement: Prove that the greatest possible number of candies given to the office is 7 when distributing candies among 8 students.

theorem greatest_candies_to_office (n : ℕ) : 
  ∃ k : ℕ, k = n % 8 ∧ k ≤ 7 ∧ k = 7 :=
by
  sorry

end greatest_candies_to_office_l216_216431


namespace maximum_of_fraction_l216_216127

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end maximum_of_fraction_l216_216127


namespace trapezoid_QR_length_l216_216828

variable (PQ RS Area Alt QR : ℝ)
variable (h1 : Area = 216)
variable (h2 : Alt = 9)
variable (h3 : PQ = 12)
variable (h4 : RS = 20)
variable (h5 : QR = 11)

theorem trapezoid_QR_length : 
  (∃ (PQ RS Area Alt QR : ℝ), 
    Area = 216 ∧
    Alt = 9 ∧
    PQ = 12 ∧
    RS = 20) → QR = 11 :=
by
  sorry

end trapezoid_QR_length_l216_216828


namespace Keith_picked_zero_apples_l216_216659

variable (M J T K_A : ℕ)

theorem Keith_picked_zero_apples (hM : M = 14) (hJ : J = 41) (hT : T = 55) (hTotalOranges : M + J = T) : K_A = 0 :=
by
  sorry

end Keith_picked_zero_apples_l216_216659


namespace village_population_equal_in_years_l216_216904

theorem village_population_equal_in_years :
  ∀ (n : ℕ), (70000 - 1200 * n = 42000 + 800 * n) ↔ n = 14 :=
by {
  sorry
}

end village_population_equal_in_years_l216_216904


namespace find_n_from_binomial_condition_l216_216103

theorem find_n_from_binomial_condition (n : ℕ) (h : Nat.choose n 3 = 7 * Nat.choose n 1) : n = 43 :=
by
  -- The proof steps would be filled in here
  sorry

end find_n_from_binomial_condition_l216_216103


namespace x_intersection_difference_l216_216097

-- Define the conditions
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem x_intersection_difference :
  let x₁ := (1 + Real.sqrt 6) / 5
  let x₂ := (1 - Real.sqrt 6) / 5
  (parabola1 x₁ = parabola2 x₁) → (parabola1 x₂ = parabola2 x₂) →
  (x₁ - x₂) = (2 * Real.sqrt 6) / 5 := 
by
  sorry

end x_intersection_difference_l216_216097


namespace total_students_in_school_l216_216871

variable (TotalStudents : ℕ)
variable (num_students_8_years_old : ℕ := 48)
variable (percent_students_below_8 : ℝ := 0.20)
variable (num_students_above_8 : ℕ := (2 / 3) * num_students_8_years_old)

theorem total_students_in_school :
  percent_students_below_8 * TotalStudents + (num_students_8_years_old + num_students_above_8) = TotalStudents :=
by
  sorry

end total_students_in_school_l216_216871


namespace num_solutions_congruence_l216_216072

-- Define the problem context and conditions
def is_valid_solution (y : ℕ) : Prop :=
  y < 150 ∧ (y + 21) % 46 = 79 % 46

-- Define the proof problem
theorem num_solutions_congruence : ∃ (s : Finset ℕ), s.card = 3 ∧ ∀ y ∈ s, is_valid_solution y := by
  sorry

end num_solutions_congruence_l216_216072


namespace find_common_difference_l216_216096

def common_difference (S_odd S_even n : ℕ) (d : ℤ) : Prop :=
  S_even - S_odd = n / 2 * d

theorem find_common_difference :
  ∃ d : ℤ, common_difference 132 112 20 d ∧ d = -2 :=
  sorry

end find_common_difference_l216_216096


namespace smallest_x_l216_216863

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end smallest_x_l216_216863


namespace find_initial_number_l216_216078

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end find_initial_number_l216_216078


namespace negation_of_existential_l216_216682

theorem negation_of_existential :
  (¬ ∃ (x : ℝ), x^2 + x + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_l216_216682


namespace integer_solution_a_l216_216524

theorem integer_solution_a (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by
  sorry

end integer_solution_a_l216_216524


namespace sum_a5_a6_a7_l216_216358

variable (a : ℕ → ℝ) (q : ℝ)

-- Assumptions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

axiom sum_a1_a2_a3 : a 1 + a 2 + a 3 = 1
axiom sum_a2_a3_a4 : a 2 + a 3 + a 4 = 2

-- The theorem we want to prove
theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 16 := sorry

end sum_a5_a6_a7_l216_216358


namespace square_side_length_l216_216053

theorem square_side_length (x : ℝ) (h : x^2 = 12) : x = 2 * Real.sqrt 3 :=
sorry

end square_side_length_l216_216053


namespace circle_area_l216_216753

theorem circle_area (r : ℝ) (h : 2 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 2 := 
by 
  sorry

end circle_area_l216_216753


namespace geometric_sequence_a6_l216_216289

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_a6 
  (a_1 q : ℝ) 
  (a2_eq : a_1 + a_1 * q = -1)
  (a3_eq : a_1 - a_1 * q ^ 2 = -3) : 
  a_n a_1 q 6 = -32 :=
sorry

end geometric_sequence_a6_l216_216289


namespace least_number_to_subtract_l216_216740

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (h1: n = 509) (h2 : d = 9): ∃ k : ℕ, k = 5 ∧ ∃ m : ℕ, n - k = d * m :=
by
  sorry

end least_number_to_subtract_l216_216740


namespace limit_of_an_l216_216878

theorem limit_of_an (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n, a_n n = (4 * n - 3) / (2 * n + 1)) → 
  a = 2 → 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros ha hA ε hε
  sorry

end limit_of_an_l216_216878


namespace teenas_speed_l216_216731

theorem teenas_speed (T : ℝ) :
  (7.5 + 15 + 40 * 1.5 = T * 1.5) → T = 55 := 
by
  intro h
  sorry

end teenas_speed_l216_216731


namespace smallest_possible_area_l216_216960

noncomputable def smallest_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) then l * w else 0

theorem smallest_possible_area : ∃ l w : ℕ, 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) ∧ smallest_area l w = 2100 := by
  sorry

end smallest_possible_area_l216_216960


namespace percentage_difference_l216_216482

theorem percentage_difference :
  ((75 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25) = 10 := 
by
  sorry

end percentage_difference_l216_216482


namespace fraction_identity_l216_216047

theorem fraction_identity (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end fraction_identity_l216_216047


namespace number_of_boys_in_other_communities_l216_216306

-- Definitions from conditions
def total_boys : ℕ := 700
def percentage_muslims : ℕ := 44
def percentage_hindus : ℕ := 28
def percentage_sikhs : ℕ := 10

-- Proof statement
theorem number_of_boys_in_other_communities : 
  (700 * (100 - (44 + 28 + 10)) / 100) = 126 := 
by
  sorry

end number_of_boys_in_other_communities_l216_216306


namespace largest_distance_l216_216612

noncomputable def max_distance_between_spheres 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) : ℝ :=
dist c1 c2 + r1 + r2

theorem largest_distance 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) 
  (h₁ : c1 = (-3, -15, 10))
  (h₂ : r1 = 24)
  (h₃ : c2 = (20, 18, -30))
  (h₄ : r2 = 95) : 
  max_distance_between_spheres c1 r1 c2 r2 = Real.sqrt 3218 + 119 := 
by
  sorry

end largest_distance_l216_216612


namespace monotonic_intervals_of_f_g_minus_f_less_than_3_l216_216899

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x < -1 / Real.exp 1 → f x < f (-1 / Real.exp 1) ∧ x > -1 / Real.exp 1 → f x > f (-1 / Real.exp 1) := sorry

theorem g_minus_f_less_than_3 :
  ∀ x : ℝ, x < 0 → g x - f x < 3 := sorry

end monotonic_intervals_of_f_g_minus_f_less_than_3_l216_216899


namespace divisible_by_6_of_cubed_sum_div_by_18_l216_216991

theorem divisible_by_6_of_cubed_sum_div_by_18 (a b c : ℤ) 
  (h : a^3 + b^3 + c^3 ≡ 0 [ZMOD 18]) : (a * b * c) ≡ 0 [ZMOD 6] :=
sorry

end divisible_by_6_of_cubed_sum_div_by_18_l216_216991


namespace hayley_stickers_l216_216784

theorem hayley_stickers (S F x : ℕ) (hS : S = 72) (hF : F = 9) (hx : x = S / F) : x = 8 :=
by
  sorry

end hayley_stickers_l216_216784


namespace product_of_integers_l216_216609

theorem product_of_integers (x y : ℕ) (h1 : x + y = 72) (h2 : x - y = 18) : x * y = 1215 := 
sorry

end product_of_integers_l216_216609


namespace option_one_correct_l216_216954

theorem option_one_correct (x : ℝ) : 
  (x ≠ 0 → x + |x| > 0) ∧ ¬((x + |x| > 0) → x ≠ 0) := 
by
  sorry

end option_one_correct_l216_216954


namespace intersecting_lines_l216_216404

def diamondsuit (a b : ℝ) : ℝ := a^2 + a * b - b^2

theorem intersecting_lines (x y : ℝ) : 
  (diamondsuit x y = diamondsuit y x) ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l216_216404


namespace third_term_arithmetic_sequence_l216_216884

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end third_term_arithmetic_sequence_l216_216884


namespace inequality_and_equality_l216_216153

theorem inequality_and_equality (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c ∧ (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end inequality_and_equality_l216_216153


namespace quadratic_sum_l216_216567

theorem quadratic_sum (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) → b + c = -106 :=
by
  intro h
  sorry

end quadratic_sum_l216_216567


namespace hash_difference_l216_216801

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference :
  (hash 8 5) - (hash 5 8) = -12 :=
by
  sorry

end hash_difference_l216_216801


namespace complement_of_M_in_U_l216_216791

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x > 0}
def complement_U_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_of_M_in_U : (U \ M) = complement_U_M :=
by sorry

end complement_of_M_in_U_l216_216791


namespace max_k_l216_216854

def A : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}

def valid_collection (B : ℕ → Finset ℕ) (k : ℕ) : Prop :=
  ∀ i j : ℕ, i < k → j < k → i ≠ j → (B i ∩ B j).card ≤ 2

theorem max_k (B : ℕ → Finset ℕ) : ∃ k, valid_collection B k → k ≤ 175 := sorry

end max_k_l216_216854


namespace sufficient_but_not_necessary_l216_216220

theorem sufficient_but_not_necessary (a : ℝ) :
  ((a + 2) * (3 * a - 4) - (a - 2) ^ 2 = 0 → a = 2 ∨ a = 1 / 2) →
  (a = 1 / 2 → ∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) →
  ( (∀ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2 → a = 1/2) ∧ 
  (∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) → a ≠ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_l216_216220


namespace ratio_of_pentagon_to_rectangle_l216_216151

theorem ratio_of_pentagon_to_rectangle (p l : ℕ) 
  (h1 : 5 * p = 30) (h2 : 2 * l + 2 * 5 = 30) : 
  p / l = 3 / 5 :=
by {
  sorry 
}

end ratio_of_pentagon_to_rectangle_l216_216151


namespace luke_fish_fillets_l216_216231

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end luke_fish_fillets_l216_216231


namespace fraction_to_decimal_l216_216105

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l216_216105


namespace range_of_set_l216_216962

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end range_of_set_l216_216962


namespace smallestBeta_satisfies_l216_216380

noncomputable def validAlphaBeta (alpha beta : ℕ) : Prop :=
  16 / 37 < (alpha : ℚ) / beta ∧ (alpha : ℚ) / beta < 7 / 16

def smallestBeta : ℕ := 23

theorem smallestBeta_satisfies :
  (∀ (alpha beta : ℕ), validAlphaBeta alpha beta → beta ≥ 23) ∧
  (∃ (alpha : ℕ), validAlphaBeta alpha 23) :=
by sorry

end smallestBeta_satisfies_l216_216380


namespace part1_part2_l216_216995

-- Part 1: Prove that x < -12 given the inequality 2(-3 + x) > 3(x + 2)
theorem part1 (x : ℝ) : 2 * (-3 + x) > 3 * (x + 2) → x < -12 := 
  by
  intro h
  sorry

-- Part 2: Prove that 0 ≤ x < 3 given the system of inequalities
theorem part2 (x : ℝ) : 
    (1 / 2) * (x + 1) < 2 ∧ (x + 2) / 2 ≥ (x + 3) / 3 → 0 ≤ x ∧ x < 3 :=
  by
  intro h
  sorry

end part1_part2_l216_216995


namespace no_valid_rook_placement_l216_216639

theorem no_valid_rook_placement :
  ∀ (r b g : ℕ), r + b + g = 50 →
  (2 * r ≤ b) →
  (2 * b ≤ g) →
  (2 * g ≤ r) →
  False :=
by
  -- Proof goes here
  sorry

end no_valid_rook_placement_l216_216639


namespace range_of_a_l216_216807

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ :=
  (m * x + n) / (x ^ 2 + 1)

example (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1) : 
  m = 2 ∧ n = 0 :=
sorry

theorem range_of_a (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1)
  (h_m : m = 2) (h_n : n = 0) {a : ℝ} : f (a-1) m n + f (a^2-1) m n < 0 ↔ 0 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l216_216807


namespace second_number_is_90_l216_216811

theorem second_number_is_90 (a b c : ℕ) 
  (h1 : a + b + c = 330) 
  (h2 : a = 2 * b) 
  (h3 : c = (1 / 3) * a) : 
  b = 90 := 
by
  sorry

end second_number_is_90_l216_216811


namespace jovana_added_shells_l216_216633

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h_initial : initial_amount = 5) 
  (h_final : final_amount = 17) 
  (h_equation : final_amount = initial_amount + added_amount) : 
  added_amount = 12 := 
by 
  sorry

end jovana_added_shells_l216_216633


namespace monotonic_subsequence_exists_l216_216021

theorem monotonic_subsequence_exists (n : ℕ) (a : Fin ((2^n : ℕ) + 1) → ℕ)
  (h : ∀ k : Fin (2^n + 1), a k ≤ k.val) : 
  ∃ (b : Fin (n + 2) → Fin (2^n + 1)),
    (∀ i j : Fin (n + 2), i ≤ j → b i ≤ b j) ∧
    (∀ i j : Fin (n + 2), i < j → a (b i) ≤ a ( b j)) :=
by
  sorry

end monotonic_subsequence_exists_l216_216021


namespace grocery_store_more_expensive_l216_216009

def bulk_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def grocery_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def price_difference_in_cents (price1 : ℚ) (price2 : ℚ) : ℚ := (price2 - price1) * 100

theorem grocery_store_more_expensive
  (bulk_total_price : ℚ)
  (bulk_cans : ℕ)
  (grocery_total_price : ℚ)
  (grocery_cans : ℕ)
  (difference_in_cents : ℚ) :
  bulk_total_price = 12.00 →
  bulk_cans = 48 →
  grocery_total_price = 6.00 →
  grocery_cans = 12 →
  difference_in_cents = 25 →
  price_difference_in_cents (bulk_price_per_can bulk_total_price bulk_cans) 
                            (grocery_price_per_can grocery_total_price grocery_cans) = difference_in_cents := by
  sorry

end grocery_store_more_expensive_l216_216009


namespace math_problem_l216_216044

def letters := "MATHEMATICS".toList

def vowels := "AAEII".toList
def consonants := "MTHMTCS".toList
def fixed_t := 'T'

def factorial (n : Nat) : Nat := 
  if n = 0 then 1 
  else n * factorial (n - 1)

def arrangements (n : Nat) (reps : List Nat) : Nat := 
  factorial n / reps.foldr (fun r acc => factorial r * acc) 1

noncomputable def vowel_arrangements := arrangements 5 [2, 2]
noncomputable def consonant_arrangements := arrangements 6 [2]

noncomputable def total_arrangements := vowel_arrangements * consonant_arrangements

theorem math_problem : total_arrangements = 10800 := by
  sorry

end math_problem_l216_216044


namespace solve_for_x_l216_216647

theorem solve_for_x (x : ℝ) :
  5 * (x - 9) = 7 * (3 - 3 * x) + 10 → x = 38 / 13 :=
by
  intro h
  sorry

end solve_for_x_l216_216647


namespace minimum_value_inequality_l216_216322

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end minimum_value_inequality_l216_216322


namespace boat_speed_in_still_water_l216_216481

theorem boat_speed_in_still_water (x y : ℝ) :
  (80 / (x + y) + 48 / (x - y) = 9) ∧ 
  (64 / (x + y) + 96 / (x - y) = 12) → 
  x = 12 :=
by
  sorry

end boat_speed_in_still_water_l216_216481


namespace max_value_a_l216_216069

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 0 < x ∧ x ≤ 150 → ¬∃ (y : ℤ), y = m * x + 3

def valid_m (m : ℚ) (a : ℚ) : Prop :=
  (2 : ℚ) / 3 < m ∧ m < a

theorem max_value_a (a : ℚ) : (a = 101 / 151) ↔ 
  ∀ (m : ℚ), valid_m m a → no_lattice_points m :=
sorry

end max_value_a_l216_216069


namespace four_people_possible_l216_216641

structure Person :=
(first_name : String)
(patronymic : String)
(surname : String)

def noThreePeopleShareSameAttribute (people : List Person) : Prop :=
  ∀ (attr : Person → String), ¬ ∃ (a b c : Person),
    a ∈ people ∧ b ∈ people ∧ c ∈ people ∧ (attr a = attr b) ∧ (attr b = attr c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def anyTwoPeopleShareAnAttribute (people : List Person) : Prop :=
  ∀ (a b : Person), a ∈ people ∧ b ∈ people ∧ a ≠ b →
    (a.first_name = b.first_name ∨ a.patronymic = b.patronymic ∨ a.surname = b.surname)

def validGroup (people : List Person) : Prop :=
  noThreePeopleShareSameAttribute people ∧ anyTwoPeopleShareAnAttribute people

theorem four_people_possible : ∃ (people : List Person), people.length = 4 ∧ validGroup people :=
sorry

end four_people_possible_l216_216641


namespace david_boxes_l216_216324

-- Conditions
def number_of_dogs_per_box : ℕ := 4
def total_number_of_dogs : ℕ := 28

-- Problem
theorem david_boxes : total_number_of_dogs / number_of_dogs_per_box = 7 :=
by
  sorry

end david_boxes_l216_216324


namespace find_a33_in_arithmetic_sequence_grid_l216_216189

theorem find_a33_in_arithmetic_sequence_grid 
  (matrix : ℕ → ℕ → ℕ)
  (rows_are_arithmetic : ∀ i, ∃ a b, ∀ j, matrix i j = a + b * (j - 1))
  (columns_are_arithmetic : ∀ j, ∃ c d, ∀ i, matrix i j = c + d * (i - 1))
  : matrix 3 3 = 31 :=
sorry

end find_a33_in_arithmetic_sequence_grid_l216_216189


namespace paula_remaining_money_l216_216560

theorem paula_remaining_money (initial_amount cost_per_shirt cost_of_pants : ℕ) 
                             (num_shirts : ℕ) (H1 : initial_amount = 109)
                             (H2 : cost_per_shirt = 11) (H3 : num_shirts = 2)
                             (H4 : cost_of_pants = 13) :
  initial_amount - (num_shirts * cost_per_shirt + cost_of_pants) = 74 := 
by
  -- Calculation of total spent and remaining would go here.
  sorry

end paula_remaining_money_l216_216560


namespace minimum_inequality_l216_216522

theorem minimum_inequality 
  (x_1 x_2 x_3 x_4 : ℝ) 
  (h1 : x_1 > 0) 
  (h2 : x_2 > 0) 
  (h3 : x_3 > 0) 
  (h4 : x_4 > 0) 
  (h_sum : x_1^2 + x_2^2 + x_3^2 + x_4^2 = 4) :
  (x_1 / (1 - x_1^2) + x_2 / (1 - x_2^2) + x_3 / (1 - x_3^2) + x_4 / (1 - x_4^2)) ≥ 6 * Real.sqrt 3 :=
by
  sorry

end minimum_inequality_l216_216522


namespace cindys_correct_result_l216_216787

-- Explicitly stating the conditions as definitions
def incorrect_operation_result := 260
def x := (incorrect_operation_result / 5) - 7

theorem cindys_correct_result : 5 * x + 7 = 232 :=
by
  -- Placeholder for the proof
  sorry

end cindys_correct_result_l216_216787


namespace card_dealing_probability_l216_216304

noncomputable def probability_ace_then_ten_then_jack : ℚ :=
  let prob_ace := 4 / 52
  let prob_ten := 4 / 51
  let prob_jack := 4 / 50
  prob_ace * prob_ten * prob_jack

theorem card_dealing_probability :
  probability_ace_then_ten_then_jack = 16 / 33150 := by
  sorry

end card_dealing_probability_l216_216304


namespace candy_distribution_l216_216286

-- Define the problem conditions and theorem.

theorem candy_distribution (X : ℕ) (total_pieces : ℕ) (portions : ℕ) 
  (subsequent_more : ℕ) (h_total : total_pieces = 40) 
  (h_portions : portions = 4) 
  (h_subsequent : subsequent_more = 2) 
  (h_eq : X + (X + subsequent_more) + (X + subsequent_more * 2) + (X + subsequent_more * 3) = total_pieces) : 
  X = 7 := 
sorry

end candy_distribution_l216_216286


namespace cost_per_meter_l216_216165

def length_of_plot : ℝ := 75
def cost_of_fencing : ℝ := 5300

-- Define breadth as a variable b
def breadth_of_plot (b : ℝ) : Prop := length_of_plot = b + 50

-- Calculate the perimeter given the known breadth
def perimeter (b : ℝ) : ℝ := 2 * length_of_plot + 2 * b

-- Define the proof problem
theorem cost_per_meter (b : ℝ) (hb : breadth_of_plot b) : 5300 / (perimeter b) = 26.5 := by
  -- Given hb: length_of_plot = b + 50, perimeter calculation follows
  sorry

end cost_per_meter_l216_216165


namespace side_length_of_square_l216_216430

-- Mathematical definitions and conditions
def square_area (side : ℕ) : ℕ := side * side

theorem side_length_of_square {s : ℕ} (h : square_area s = 289) : s = 17 :=
sorry

end side_length_of_square_l216_216430


namespace trig_expression_value_l216_216767

theorem trig_expression_value (θ : Real) (h1 : θ > Real.pi) (h2 : θ < 3 * Real.pi / 2) (h3 : Real.tan (2 * θ) = 3 / 4) :
  (2 * Real.cos (θ / 2) ^ 2 + Real.sin θ - 1) / (Real.sqrt 2 * Real.cos (θ + Real.pi / 4)) = 2 := by
  sorry

end trig_expression_value_l216_216767


namespace range_of_a_l216_216386

def A : Set ℝ := { x | x^2 - x - 2 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - a) < 3 }

theorem range_of_a (a : ℝ) :
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by
  sorry

end range_of_a_l216_216386


namespace lowest_point_graph_of_y_l216_216357

theorem lowest_point_graph_of_y (x : ℝ) (h : x > -1) :
  (x, (x^2 + 2 * x + 2) / (x + 1)) = (0, 2) ∧ ∀ y > -1, ( (y^2 + 2 * y + 2) / (y + 1) >= 2) := 
sorry

end lowest_point_graph_of_y_l216_216357


namespace tan_alpha_sin_cos_half_alpha_l216_216451

variable (α : ℝ)

-- Conditions given in the problem
def cond1 : Real.sin α = 1 / 3 := sorry
def cond2 : 0 < α ∧ α < Real.pi := sorry

-- Lean proof that given the conditions, the solutions are as follows:
theorem tan_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = Real.sqrt 2 / 4 ∨ Real.tan α = - Real.sqrt 2 / 4 := sorry

theorem sin_cos_half_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin (α / 2) + Real.cos (α / 2) = 2 * Real.sqrt 3 / 3 := sorry

end tan_alpha_sin_cos_half_alpha_l216_216451


namespace train_length_is_199_95_l216_216577

noncomputable def convert_speed_to_m_s (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

noncomputable def length_of_train (bridge_length : ℝ) (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := convert_speed_to_m_s speed_kmh
  speed_ms * time_seconds - bridge_length

theorem train_length_is_199_95 :
  length_of_train 300 45 40 = 199.95 := by
  sorry

end train_length_is_199_95_l216_216577


namespace find_p_l216_216603

variable (m n p : ℚ)

theorem find_p (h1 : m = 8 * n + 5) (h2 : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by
  sorry

end find_p_l216_216603


namespace Walter_receives_49_bananas_l216_216176

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l216_216176


namespace sum_of_squares_l216_216181

theorem sum_of_squares :
  1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 = 5020030 :=
by
  sorry

end sum_of_squares_l216_216181


namespace jordon_machine_number_l216_216333

theorem jordon_machine_number : 
  ∃ x : ℝ, (2 * x + 3 = 27) ∧ x = 12 :=
by
  sorry

end jordon_machine_number_l216_216333


namespace intersection_A_B_l216_216694

open Set

def A : Set ℕ := {x | -2 < (x : ℤ) ∧ (x : ℤ) < 2}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ {x : ℕ | (x : ℤ) ∈ B} = {0, 1} := by
  sorry

end intersection_A_B_l216_216694


namespace probability_not_win_l216_216660

theorem probability_not_win (A B : Fin 16) : 
  (256 - 16) / 256 = 15 / 16 := 
by
  sorry

end probability_not_win_l216_216660


namespace jacob_fifth_test_score_l216_216533

theorem jacob_fifth_test_score (s1 s2 s3 s4 s5 : ℕ) :
  s1 = 85 ∧ s2 = 79 ∧ s3 = 92 ∧ s4 = 84 ∧ ((s1 + s2 + s3 + s4 + s5) / 5 = 85) →
  s5 = 85 :=
sorry

end jacob_fifth_test_score_l216_216533


namespace ceil_sqrt_250_eq_16_l216_216919

theorem ceil_sqrt_250_eq_16 : ⌈Real.sqrt 250⌉ = 16 :=
by
  have h1 : (15 : ℝ) < Real.sqrt 250 := sorry
  have h2 : Real.sqrt 250 < 16 := sorry
  exact sorry

end ceil_sqrt_250_eq_16_l216_216919


namespace selection_probability_correct_l216_216867

def percentage_women : ℝ := 0.55
def percentage_men : ℝ := 0.45

def women_below_35 : ℝ := 0.20
def women_35_to_50 : ℝ := 0.35
def women_above_50 : ℝ := 0.45

def men_below_35 : ℝ := 0.30
def men_35_to_50 : ℝ := 0.40
def men_above_50 : ℝ := 0.30

def women_below_35_lawyers : ℝ := 0.35
def women_below_35_doctors : ℝ := 0.45
def women_below_35_engineers : ℝ := 0.20

def women_35_to_50_lawyers : ℝ := 0.25
def women_35_to_50_doctors : ℝ := 0.50
def women_35_to_50_engineers : ℝ := 0.25

def women_above_50_lawyers : ℝ := 0.20
def women_above_50_doctors : ℝ := 0.30
def women_above_50_engineers : ℝ := 0.50

def men_below_35_lawyers : ℝ := 0.40
def men_below_35_doctors : ℝ := 0.30
def men_below_35_engineers : ℝ := 0.30

def men_35_to_50_lawyers : ℝ := 0.45
def men_35_to_50_doctors : ℝ := 0.25
def men_35_to_50_engineers : ℝ := 0.30

def men_above_50_lawyers : ℝ := 0.30
def men_above_50_doctors : ℝ := 0.40
def men_above_50_engineers : ℝ := 0.30

theorem selection_probability_correct :
  (percentage_women * women_below_35 * women_below_35_lawyers +
   percentage_men * men_above_50 * men_above_50_engineers +
   percentage_women * women_35_to_50 * women_35_to_50_doctors +
   percentage_men * men_35_to_50 * men_35_to_50_doctors) = 0.22025 :=
by
  sorry

end selection_probability_correct_l216_216867


namespace faster_train_cross_time_l216_216977

/-- Statement of the problem in Lean 4 -/
theorem faster_train_cross_time :
  let speed_faster_train_kmph := 72
  let speed_slower_train_kmph := 36
  let length_faster_train_m := 180
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18 : ℝ)
  let time_taken := length_faster_train_m / relative_speed_mps
  time_taken = 18 :=
by
  sorry

end faster_train_cross_time_l216_216977


namespace largest_angle_of_convex_hexagon_l216_216390

theorem largest_angle_of_convex_hexagon 
  (x : ℝ) 
  (hx : (x + 2) + (2 * x - 1) + (3 * x + 1) + (4 * x - 2) + (5 * x + 3) + (6 * x - 4) = 720) :
  6 * x - 4 = 202 :=
sorry

end largest_angle_of_convex_hexagon_l216_216390


namespace kittens_and_mice_count_l216_216014

theorem kittens_and_mice_count :
  let children := 12
  let baskets_per_child := 3
  let cats_per_basket := 1
  let kittens_per_cat := 12
  let mice_per_kitten := 4
  let total_kittens := children * baskets_per_child * cats_per_basket * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice = 2160 :=
by
  sorry

end kittens_and_mice_count_l216_216014


namespace first_discount_percentage_l216_216829

theorem first_discount_percentage (x : ℝ) 
  (h₁ : ∀ (p : ℝ), p = 70) 
  (h₂ : ∀ (d₁ d₂ : ℝ), d₁ = x / 100 ∧ d₂ = 0.01999999999999997 )
  (h₃ : ∀ (final_price : ℝ), final_price = 61.74):
  x = 10 := 
by
  sorry

end first_discount_percentage_l216_216829


namespace stamp_collection_l216_216387

theorem stamp_collection (x : ℕ) :
  (5 * x + 3 * (x + 20) = 300) → (x = 30) ∧ (x + 20 = 50) :=
by
  sorry

end stamp_collection_l216_216387


namespace inequalities_validity_l216_216576

theorem inequalities_validity (x y a b : ℝ) (hx : x ≤ a) (hy : y ≤ b) (hstrict : x < a ∨ y < b) :
  (x + y ≤ a + b) ∧
  ¬((x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b)) :=
by
  -- Here is where the proof would go.
  sorry

end inequalities_validity_l216_216576


namespace candies_markus_l216_216164

theorem candies_markus (m k s : ℕ) (h_initial_m : m = 9) (h_initial_k : k = 5) (h_total_s : s = 10) :
  (m + s) / 2 = 12 := by
  sorry

end candies_markus_l216_216164


namespace gcd_490_910_l216_216656

theorem gcd_490_910 : Nat.gcd 490 910 = 70 :=
by
  sorry

end gcd_490_910_l216_216656


namespace units_digit_7_pow_2023_l216_216444

theorem units_digit_7_pow_2023 :
  (7 ^ 2023) % 10 = 3 := 
sorry

end units_digit_7_pow_2023_l216_216444


namespace total_points_scored_l216_216741

-- Define the points scored by Sam and his friend
def points_scored_by_sam : ℕ := 75
def points_scored_by_friend : ℕ := 12

-- The main theorem stating the total points
theorem total_points_scored : points_scored_by_sam + points_scored_by_friend = 87 := by
  -- Proof goes here
  sorry

end total_points_scored_l216_216741


namespace sum_of_common_ratios_l216_216229

-- Definitions for the geometric sequence conditions
def geom_seq_a (m : ℝ) (s : ℝ) (n : ℕ) : ℝ := m * s^n
def geom_seq_b (m : ℝ) (t : ℝ) (n : ℕ) : ℝ := m * t^n

-- Theorem statement
theorem sum_of_common_ratios (m s t : ℝ) (h₀ : m ≠ 0) (h₁ : s ≠ t) 
    (h₂ : geom_seq_a m s 2 - geom_seq_b m t 2 = 3 * (geom_seq_a m s 1 - geom_seq_b m t 1)) :
    s + t = 3 :=
by
  sorry

end sum_of_common_ratios_l216_216229


namespace neg_neg_one_eq_one_l216_216502

theorem neg_neg_one_eq_one : -(-1) = 1 :=
by
  sorry

end neg_neg_one_eq_one_l216_216502


namespace necessary_but_not_sufficient_l216_216833

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 > 4) → (x > 2 ∨ x < -2) ∧ ¬((x^2 > 4) ↔ (x > 2)) :=
by
  intros h
  have h1 : x > 2 ∨ x < -2 := by sorry
  have h2 : ¬((x^2 > 4) ↔ (x > 2)) := by sorry
  exact And.intro h1 h2

end necessary_but_not_sufficient_l216_216833


namespace pencils_per_row_l216_216042

-- Define the conditions as parameters
variables (total_pencils : Int) (rows : Int) 

-- State the proof problem using the conditions and the correct answer
theorem pencils_per_row (h₁ : total_pencils = 12) (h₂ : rows = 3) : total_pencils / rows = 4 := 
by 
  sorry

end pencils_per_row_l216_216042


namespace chicken_farm_l216_216402

def total_chickens (roosters hens : ℕ) : ℕ := roosters + hens

theorem chicken_farm (roosters hens : ℕ) (h1 : 2 * hens = roosters) (h2 : roosters = 6000) : 
  total_chickens roosters hens = 9000 :=
by
  sorry

end chicken_farm_l216_216402


namespace triangle_is_isosceles_right_l216_216124

theorem triangle_is_isosceles_right (A B C a b c : ℝ) 
  (h : a / (Real.cos A) = b / (Real.cos B) ∧ b / (Real.cos B) = c / (Real.sin C)) :
  A = π/4 ∧ B = π/4 ∧ C = π/2 := 
sorry

end triangle_is_isosceles_right_l216_216124


namespace eight_x_plus_y_l216_216882

theorem eight_x_plus_y (x y z : ℝ) (h1 : x + 2 * y - 3 * z = 7) (h2 : 2 * x - y + 2 * z = 6) : 
  8 * x + y = 32 :=
sorry

end eight_x_plus_y_l216_216882


namespace problem1_problem2_l216_216575

noncomputable def f (x a : ℝ) := x - (x^2 + a * x) / Real.exp x

theorem problem1 (x : ℝ) : (f x 1) ≥ 0 := by
  sorry

theorem problem2 (x : ℝ) : (1 - (Real.log x) / x) * (f x (-1)) > 1 - 1/(Real.exp 2) := by
  sorry

end problem1_problem2_l216_216575


namespace time_via_route_B_l216_216894

-- Given conditions
def time_via_route_A : ℕ := 5
def time_saved_round_trip : ℕ := 6

-- Defining the proof problem
theorem time_via_route_B : time_via_route_A - (time_saved_round_trip / 2) = 2 :=
by
  -- Expected proof here
  sorry

end time_via_route_B_l216_216894


namespace jane_played_rounds_l216_216734

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end jane_played_rounds_l216_216734


namespace prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l216_216032

-- Define the conditions
def quadratic_has_two_different_negative_roots (a : ℝ) : Prop :=
  a^2 - 1/4 > 0 ∧ -a < 0 ∧ 1/16 > 0

def inequality_q (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Prove the results based on the conditions
theorem prove_a_range_if_p (a : ℝ) (hp : quadratic_has_two_different_negative_roots a) : a > 1/2 :=
  sorry

theorem prove_a_range_if_p_or_q_and_not_and (a : ℝ) (hp_or_q : quadratic_has_two_different_negative_roots a ∨ inequality_q a) 
  (hnot_p_and_q : ¬ (quadratic_has_two_different_negative_roots a ∧ inequality_q a)) :
  a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) :=
  sorry

end prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l216_216032


namespace f_at_5_l216_216017

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

axiom odd_function (f: ℝ → ℝ) : ∀ x : ℝ, f (-x) = -f x
axiom functional_equation (f: ℝ → ℝ) : ∀ x : ℝ, f (x + 1) + f x = 0

theorem f_at_5 : f 5 = 0 :=
by {
  -- Proof to be provided here
  sorry
}

end f_at_5_l216_216017


namespace find_integer_to_satisfy_eq_l216_216590

theorem find_integer_to_satisfy_eq (n : ℤ) (h : n - 5 = 2) : n = 7 :=
sorry

end find_integer_to_satisfy_eq_l216_216590


namespace parabola_y_intercepts_l216_216400

theorem parabola_y_intercepts : 
  (∃ y1 y2 : ℝ, 3 * y1^2 - 4 * y1 + 1 = 0 ∧ 3 * y2^2 - 4 * y2 + 1 = 0 ∧ y1 ≠ y2) :=
by
  sorry

end parabola_y_intercepts_l216_216400


namespace problem1_problem2_l216_216210

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 1 }

-- Prove that for a = 1/2, A ∩ B = { x | 0 < x ∧ x < 1 }
theorem problem1 : setA (1/2) ∩ setB = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

-- Prove that if A ∩ B = ∅, then a ≤ -1/2 or a ≥ 2
theorem problem2 (a : ℝ) (h : setA a ∩ setB = ∅) : a ≤ -1/2 ∨ a ≥ 2 :=
by
  sorry

end problem1_problem2_l216_216210


namespace distance_between_homes_l216_216432

-- Define the parameters
def maxwell_speed : ℝ := 4  -- km/h
def brad_speed : ℝ := 6     -- km/h
def maxwell_time_to_meet : ℝ := 2  -- hours
def brad_start_delay : ℝ := 1  -- hours

-- Definitions related to the timings
def brad_time_to_meet : ℝ := maxwell_time_to_meet - brad_start_delay  -- hours

-- Define the distances covered by each
def maxwell_distance : ℝ := maxwell_speed * maxwell_time_to_meet  -- km
def brad_distance : ℝ := brad_speed * brad_time_to_meet  -- km

-- Define the total distance between their homes
def total_distance : ℝ := maxwell_distance + brad_distance  -- km

-- Statement to prove
theorem distance_between_homes : total_distance = 14 :=
by
  -- The proof is omitted; add 'sorry' to indicate this.
  sorry

end distance_between_homes_l216_216432


namespace lines_intersect_at_same_point_l216_216104

theorem lines_intersect_at_same_point : 
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = -3 * x + 4 ∧ y = 4 * x + m) → m = -3 :=
by
  sorry

end lines_intersect_at_same_point_l216_216104


namespace range_of_a_l216_216538

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h_roots : x1 < 1 ∧ 1 < x2) (h_eq : ∀ x, x^2 + a * x - 2 = (x - x1) * (x - x2)) : a < 1 :=
sorry

end range_of_a_l216_216538


namespace emily_expenditure_l216_216941

-- Define the conditions
def price_per_flower : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

-- Total flowers bought
def total_flowers (roses daisies : ℕ) : ℕ :=
  roses + daisies

-- Define the cost function
def cost (flowers price_per_flower : ℕ) : ℕ :=
  flowers * price_per_flower

-- Theorem to prove the total expenditure
theorem emily_expenditure : 
  cost (total_flowers roses_bought daisies_bought) price_per_flower = 12 :=
by
  sorry

end emily_expenditure_l216_216941


namespace cell_population_l216_216622

variable (n : ℕ)

def a (n : ℕ) : ℕ :=
  if n = 1 then 5
  else 1 -- Placeholder for general definition

theorem cell_population (n : ℕ) : a n = 2^(n-1) + 4 := by
  sorry

end cell_population_l216_216622


namespace color_opposite_orange_is_indigo_l216_216092

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end color_opposite_orange_is_indigo_l216_216092


namespace find_other_number_l216_216204

open Nat

theorem find_other_number (A B lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 30) (h_A : A = 231) (h_eq : lcm * hcf = A * B) : 
  B = 300 :=
  sorry

end find_other_number_l216_216204


namespace minimum_value_of_f_l216_216244

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, y = f x ∧ y >= 5 / 2 :=
by
  sorry

end minimum_value_of_f_l216_216244
