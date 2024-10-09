import Mathlib

namespace k_range_condition_l1201_120167

theorem k_range_condition (k : ℝ) :
    (∀ x : ℝ, x^2 - (2 * k - 6) * x + k - 3 > 0) ↔ (3 < k ∧ k < 4) :=
by
  sorry

end k_range_condition_l1201_120167


namespace solution_point_satisfies_inequalities_l1201_120149

theorem solution_point_satisfies_inequalities:
  let x := -1/3
  let y := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧ x - 4 * y ≤ -3 :=
by
  let x := -1/3
  let y := 2/3
  sorry

end solution_point_satisfies_inequalities_l1201_120149


namespace card_trick_l1201_120128

/-- A magician is able to determine the fifth card from a 52-card deck using a prearranged 
    communication system between the magician and the assistant, thus no supernatural 
    abilities are required. -/
theorem card_trick (deck : Finset ℕ) (h_deck : deck.card = 52) (chosen_cards : Finset ℕ)
  (h_chosen : chosen_cards.card = 5) (shown_cards : Finset ℕ) (h_shown : shown_cards.card = 4)
  (fifth_card : ℕ) (h_fifth_card : fifth_card ∈ chosen_cards \ shown_cards) :
  ∃ (prearranged_system : (Finset ℕ) → (Finset ℕ) → ℕ),
    ∀ (remaining : Finset ℕ), remaining.card = 1 → 
    prearranged_system shown_cards remaining = fifth_card := 
sorry

end card_trick_l1201_120128


namespace pressure_force_correct_l1201_120170

-- Define the conditions
noncomputable def base_length : ℝ := 4
noncomputable def vertex_depth : ℝ := 4
noncomputable def gamma : ℝ := 1000 -- density of water in kg/m^3
noncomputable def g : ℝ := 9.81 -- acceleration due to gravity in m/s^2

-- Define the calculation of the pressure force on the parabolic segment
noncomputable def pressure_force (base_length vertex_depth gamma g : ℝ) : ℝ :=
  19620 * (4 * ((2/3) * (4 : ℝ)^(3/2)) - ((2/5) * (4 : ℝ)^(5/2)))

-- State the theorem
theorem pressure_force_correct : pressure_force base_length vertex_depth gamma g = 167424 := 
by
  sorry

end pressure_force_correct_l1201_120170


namespace sum_single_digits_l1201_120113

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end sum_single_digits_l1201_120113


namespace find_a_b_and_compare_y_values_l1201_120158

-- Conditions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- Problem statement
theorem find_a_b_and_compare_y_values (a b y1 y2 y3 : ℝ) (h₀ : quadratic a b (-2) = 1) (h₁ : linear a (-2) = 1)
    (h2 : y1 = quadratic a b 2) (h3 : y2 = quadratic a b b) (h4 : y3 = quadratic a b (a - b)) :
  (a = -1/2) ∧ (b = -2) ∧ y1 < y3 ∧ y3 < y2 :=
by
  -- Placeholder for the proof
  sorry

end find_a_b_and_compare_y_values_l1201_120158


namespace solve_nat_pairs_l1201_120129

theorem solve_nat_pairs (n m : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end solve_nat_pairs_l1201_120129


namespace star_equiv_l1201_120100

variable {m n x y : ℝ}

def star (m n : ℝ) : ℝ := (3 * m - 2 * n) ^ 2

theorem star_equiv (x y : ℝ) : star ((3 * x - 2 * y) ^ 2) ((2 * y - 3 * x) ^ 2) = (3 * x - 2 * y) ^ 4 := 
by
  sorry

end star_equiv_l1201_120100


namespace solve_for_A_l1201_120162

theorem solve_for_A : 
  ∃ (A B : ℕ), (100 * A + 78) - (200 + 10 * B + 4) = 364 → A = 5 :=
by
  sorry

end solve_for_A_l1201_120162


namespace yvon_combination_l1201_120165

theorem yvon_combination :
  let num_notebooks := 4
  let num_pens := 5
  num_notebooks * num_pens = 20 :=
by
  sorry

end yvon_combination_l1201_120165


namespace balloons_popped_on_ground_l1201_120187

def max_rate : Nat := 2
def max_time : Nat := 30
def zach_rate : Nat := 3
def zach_time : Nat := 40
def total_filled_balloons : Nat := 170

theorem balloons_popped_on_ground :
  (max_rate * max_time + zach_rate * zach_time) - total_filled_balloons = 10 :=
by
  sorry

end balloons_popped_on_ground_l1201_120187


namespace boats_seating_problem_l1201_120108

theorem boats_seating_problem 
  (total_boats : ℕ) (total_people : ℕ) 
  (big_boat_seats : ℕ) (small_boat_seats : ℕ) 
  (b s : ℕ) 
  (h1 : total_boats = 12) 
  (h2 : total_people = 58) 
  (h3 : big_boat_seats = 6) 
  (h4 : small_boat_seats = 4) 
  (h5 : b + s = 12) 
  (h6 : b * 6 + s * 4 = 58) 
  : b = 5 ∧ s = 7 :=
sorry

end boats_seating_problem_l1201_120108


namespace sum_of_ages_l1201_120124

variable (a b c : ℕ)

theorem sum_of_ages (h1 : a = 20 + b + c) (h2 : a^2 = 2000 + (b + c)^2) : a + b + c = 80 := 
by
  sorry

end sum_of_ages_l1201_120124


namespace red_apples_ordered_l1201_120116

variable (R : ℕ)

theorem red_apples_ordered (h : R + 32 = 2 + 73) : R = 43 := by
  sorry

end red_apples_ordered_l1201_120116


namespace rational_sum_eq_one_l1201_120111

theorem rational_sum_eq_one (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := 
by
  sorry

end rational_sum_eq_one_l1201_120111


namespace number_of_children_l1201_120160

theorem number_of_children (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 390)) : C = 780 :=
by
  sorry

end number_of_children_l1201_120160


namespace coach_mike_change_l1201_120161

theorem coach_mike_change (cost amount_given change : ℕ) 
    (h_cost : cost = 58) (h_amount_given : amount_given = 75) : 
    change = amount_given - cost → change = 17 := by
    sorry

end coach_mike_change_l1201_120161


namespace product_gcd_lcm_eq_1296_l1201_120146

theorem product_gcd_lcm_eq_1296 : (Int.gcd 24 54) * (Int.lcm 24 54) = 1296 := by
  sorry

end product_gcd_lcm_eq_1296_l1201_120146


namespace volume_units_correct_l1201_120182

/-- Definition for the volume of a bottle of coconut juice in milliliters (200 milliliters). -/
def volume_of_coconut_juice := 200 

/-- Definition for the volume of an electric water heater in liters (50 liters). -/
def volume_of_electric_water_heater := 50 

/-- Prove that the volume of a bottle of coconut juice is measured in milliliters (200 milliliters)
    and the volume of an electric water heater is measured in liters (50 liters).
-/
theorem volume_units_correct :
  volume_of_coconut_juice = 200 ∧ volume_of_electric_water_heater = 50 :=
sorry

end volume_units_correct_l1201_120182


namespace negation_of_forall_x_gt_1_l1201_120139

theorem negation_of_forall_x_gt_1 : ¬(∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by
  sorry

end negation_of_forall_x_gt_1_l1201_120139


namespace reliefSuppliesCalculation_l1201_120173

noncomputable def totalReliefSupplies : ℝ := 644

theorem reliefSuppliesCalculation
    (A_capacity : ℝ)
    (B_capacity : ℝ)
    (A_capacity_per_day : A_capacity = 64.4)
    (capacity_ratio : A_capacity = 1.75 * B_capacity)
    (additional_transport : ∃ t : ℝ, A_capacity * t - B_capacity * t = 138 ∧ A_capacity * t = 322) :
  totalReliefSupplies = 644 := by
  sorry

end reliefSuppliesCalculation_l1201_120173


namespace trinomial_ne_binomial_l1201_120164

theorem trinomial_ne_binomial (a b c A B : ℝ) (h : a ≠ 0) : 
  ¬ ∀ x : ℝ, ax^2 + bx + c = Ax + B :=
by
  sorry

end trinomial_ne_binomial_l1201_120164


namespace second_account_interest_rate_l1201_120181

theorem second_account_interest_rate
  (investment1 : ℝ)
  (rate1 : ℝ)
  (interest1 : ℝ)
  (investment2 : ℝ)
  (interest2 : ℝ)
  (h1 : 4000 = investment1)
  (h2 : 0.08 = rate1)
  (h3 : 320 = interest1)
  (h4 : 7200 - 4000 = investment2)
  (h5 : interest1 = interest2) :
  interest2 / investment2 = 0.1 :=
by
  sorry

end second_account_interest_rate_l1201_120181


namespace largest_x_l1201_120194

-- Define the condition of the problem.
def equation_holds (x : ℝ) : Prop :=
  (5 * x - 20) / (4 * x - 5) ^ 2 + (5 * x - 20) / (4 * x - 5) = 20

-- State the theorem to prove the largest value of x is 9/5.
theorem largest_x : ∃ x : ℝ, equation_holds x ∧ ∀ y : ℝ, equation_holds y → y ≤ 9 / 5 :=
by
  sorry

end largest_x_l1201_120194


namespace find_y_value_l1201_120176

theorem find_y_value (y : ℝ) (h : 12^2 * y^3 / 432 = 72) : y = 6 :=
by
  sorry

end find_y_value_l1201_120176


namespace initial_volume_of_mixture_l1201_120117

theorem initial_volume_of_mixture 
  (V : ℝ)
  (h1 : 0 < V) 
  (h2 : 0.20 * V = 0.15 * (V + 5)) :
  V = 15 :=
by 
  -- proof steps 
  sorry

end initial_volume_of_mixture_l1201_120117


namespace daisy_count_per_bouquet_l1201_120112

-- Define the conditions
def roses_per_bouquet := 12
def total_bouquets := 20
def rose_bouquets := 10
def daisy_bouquets := total_bouquets - rose_bouquets
def total_flowers_sold := 190
def total_roses_sold := rose_bouquets * roses_per_bouquet
def total_daisies_sold := total_flowers_sold - total_roses_sold

-- Define the problem: prove that the number of daisies per bouquet is 7
theorem daisy_count_per_bouquet : total_daisies_sold / daisy_bouquets = 7 := by
  sorry

end daisy_count_per_bouquet_l1201_120112


namespace danivan_drugstore_end_of_week_inventory_l1201_120154

-- Define the initial conditions in Lean
def initial_inventory := 4500
def sold_monday := 2445
def sold_tuesday := 900
def sold_wednesday_to_sunday := 50 * 5
def supplier_delivery := 650

-- Define the statement of the proof problem
theorem danivan_drugstore_end_of_week_inventory :
  initial_inventory - (sold_monday + sold_tuesday + sold_wednesday_to_sunday) + supplier_delivery = 1555 :=
by
  sorry

end danivan_drugstore_end_of_week_inventory_l1201_120154


namespace estimate_m_value_l1201_120177

-- Definition of polynomial P(x) and its roots related to the problem
noncomputable def P (x : ℂ) (a b c : ℂ) : ℂ := x^3 + a * x^2 + b * x + c

-- Statement of the problem in Lean 4
theorem estimate_m_value :
  ∀ (a b c : ℕ),
  a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧
  (∃ z1 z2 z3 : ℂ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧ 
  P z1 a b c = 0 ∧ P z2 a b c = 0 ∧ P z3 a b c = 0) →
  ∃ m : ℕ, m = 8097 :=
sorry

end estimate_m_value_l1201_120177


namespace division_and_multiplication_l1201_120185

theorem division_and_multiplication (dividend divisor quotient factor product : ℕ) 
  (h₁ : dividend = 24) 
  (h₂ : divisor = 3) 
  (h₃ : quotient = dividend / divisor) 
  (h₄ : factor = 5) 
  (h₅ : product = quotient * factor) : 
  quotient = 8 ∧ product = 40 := 
by 
  sorry

end division_and_multiplication_l1201_120185


namespace number_of_pens_sold_l1201_120126

variables (C N : ℝ) (gain_percentage : ℝ) (gain : ℝ)

-- Defining conditions given in the problem
def trader_gain_cost_pens (C N : ℝ) : ℝ := 30 * C
def gain_percentage_condition (gain_percentage : ℝ) : Prop := gain_percentage = 0.30
def gain_condition (C N : ℝ) : Prop := (0.30 * N * C) = 30 * C

-- Defining the theorem to prove
theorem number_of_pens_sold
  (h_gain_percentage : gain_percentage_condition gain_percentage)
  (h_gain : gain_condition C N) :
  N = 100 :=
sorry

end number_of_pens_sold_l1201_120126


namespace percentage_material_B_in_final_mixture_l1201_120131

-- Conditions
def percentage_material_A_in_Solution_X : ℝ := 20
def percentage_material_B_in_Solution_X : ℝ := 80
def percentage_material_A_in_Solution_Y : ℝ := 30
def percentage_material_B_in_Solution_Y : ℝ := 70
def percentage_material_A_in_final_mixture : ℝ := 22

-- Goal
theorem percentage_material_B_in_final_mixture :
  100 - percentage_material_A_in_final_mixture = 78 := by
  sorry

end percentage_material_B_in_final_mixture_l1201_120131


namespace jacque_suitcase_weight_l1201_120196

noncomputable def suitcase_weight_return (original_weight : ℝ)
                                         (perfume_weight_oz : ℕ → ℝ)
                                         (chocolate_weight_lb : ℝ)
                                         (soap_weight_oz : ℕ → ℝ)
                                         (jam_weight_oz : ℕ → ℝ)
                                         (sculpture_weight_kg : ℝ)
                                         (shirt_weight_g : ℕ → ℝ)
                                         (oz_to_lb : ℝ)
                                         (kg_to_lb : ℝ)
                                         (g_to_kg : ℝ) : ℝ :=
  original_weight +
  (perfume_weight_oz 5 / oz_to_lb) +
  chocolate_weight_lb +
  (soap_weight_oz 2 / oz_to_lb) +
  (jam_weight_oz 2 / oz_to_lb) +
  (sculpture_weight_kg * kg_to_lb) +
  ((shirt_weight_g 3 / g_to_kg) * kg_to_lb)

theorem jacque_suitcase_weight :
  suitcase_weight_return 12 
                        (fun n => n * 1.2) 
                        4 
                        (fun n => n * 5) 
                        (fun n => n * 8)
                        3.5 
                        (fun n => n * 300) 
                        16 
                        2.20462 
                        1000 
  = 27.70 :=
sorry

end jacque_suitcase_weight_l1201_120196


namespace find_f2_l1201_120114

variable (f g : ℝ → ℝ) (a : ℝ)

-- Definitions based on conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def equation (f g : ℝ → ℝ) (a : ℝ) := ∀ x, f x + g x = a^x - a^(-x) + 2

-- Lean statement for the proof problem
theorem find_f2
  (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : equation f g a)
  (h4 : g 2 = a) : f 2 = 15 / 4 :=
by
  sorry

end find_f2_l1201_120114


namespace number_of_solutions_l1201_120118

theorem number_of_solutions (x y: ℕ) (hx : 0 < x) (hy : 0 < y) :
    (1 / (x + 1) + 1 / y + 1 / ((x + 1) * y) = 1 / 1991) →
    ∃! (n : ℕ), n = 64 :=
by
  sorry

end number_of_solutions_l1201_120118


namespace data_variance_l1201_120157

def data : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

theorem data_variance : variance data = 0.02 := by
  sorry

end data_variance_l1201_120157


namespace typing_lines_in_10_minutes_l1201_120147

def programmers := 10
def total_lines_in_60_minutes := 60
def total_minutes := 60
def target_minutes := 10

theorem typing_lines_in_10_minutes :
  (total_lines_in_60_minutes / total_minutes) * programmers * target_minutes = 100 :=
by sorry

end typing_lines_in_10_minutes_l1201_120147


namespace pq_eqv_l1201_120178

theorem pq_eqv (p q : Prop) : 
  ((¬ p ∧ ¬ q) ∧ (p ∨ q)) ↔ ((p ∧ ¬ q) ∨ (¬ p ∧ q)) :=
by
  sorry

end pq_eqv_l1201_120178


namespace find_initial_apples_l1201_120168

def initial_apples (a b c : ℕ) : Prop :=
  b + c = a

theorem find_initial_apples (a b initial_apples : ℕ) (h : b + initial_apples = a) : initial_apples = 8 :=
by
  sorry

end find_initial_apples_l1201_120168


namespace sum_and_product_of_roots_l1201_120104

theorem sum_and_product_of_roots (m p : ℝ) 
    (h₁ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α + β = 9)
    (h₂ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α * β = 14) :
    m + p = 69 := 
sorry

end sum_and_product_of_roots_l1201_120104


namespace find_r_for_f_of_3_eq_0_l1201_120141

noncomputable def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

theorem find_r_for_f_of_3_eq_0 : ∃ r : ℝ, f 3 r = 0 ∧ r = -186 := by
  sorry

end find_r_for_f_of_3_eq_0_l1201_120141


namespace area_of_smaller_circle_l1201_120189

noncomputable def radius_of_smaller_circle (r : ℝ) : ℝ := r

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 3 * r

noncomputable def length_PA := 5
noncomputable def length_AB := 5

theorem area_of_smaller_circle (r : ℝ) (h1 : radius_of_smaller_circle r = r)
  (h2 : radius_of_larger_circle r = 3 * r)
  (h3 : length_PA = 5) (h4 : length_AB = 5) :
  π * r^2 = (25 / 3) * π :=
  sorry

end area_of_smaller_circle_l1201_120189


namespace evaluate_expression_l1201_120198

theorem evaluate_expression (x y : ℝ) (h1 : x = 3) (h2 : y = 0) : y * (y - 3 * x) = 0 :=
by sorry

end evaluate_expression_l1201_120198


namespace tagged_fish_in_second_catch_l1201_120102

theorem tagged_fish_in_second_catch (N : ℕ) (initially_tagged second_catch : ℕ)
  (h1 : N = 1250)
  (h2 : initially_tagged = 50)
  (h3 : second_catch = 50) :
  initially_tagged / N * second_catch = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l1201_120102


namespace vector_b_norm_range_l1201_120103

variable (a b : ℝ × ℝ)
variable (norm_a : ‖a‖ = 1)
variable (norm_sum : ‖a + b‖ = 2)

theorem vector_b_norm_range : 1 ≤ ‖b‖ ∧ ‖b‖ ≤ 3 :=
sorry

end vector_b_norm_range_l1201_120103


namespace average_income_N_O_l1201_120137

variable (M N O : ℝ)

-- Condition declaration
def condition1 : Prop := M + N = 10100
def condition2 : Prop := M + O = 10400
def condition3 : Prop := M = 4000

-- Theorem statement
theorem average_income_N_O (h1 : condition1 M N) (h2 : condition2 M O) (h3 : condition3 M) :
  (N + O) / 2 = 6250 :=
sorry

end average_income_N_O_l1201_120137


namespace line_intersects_circle_l1201_120153

theorem line_intersects_circle (k : ℝ) (h1 : k = 2) (radius : ℝ) (center_distance : ℝ) (eq_roots : ∀ x, x^2 - k * x + 1 = 0) :
  radius = 5 → center_distance = k → k < radius :=
by
  intros hradius hdistance
  have h_root_eq : k = 2 := h1
  have h_rad : radius = 5 := hradius
  have h_dist : center_distance = k := hdistance
  have kval : k = 2 := h1
  simp [kval, hradius, hdistance, h_rad, h_dist]
  sorry

end line_intersects_circle_l1201_120153


namespace range_of_x_max_y_over_x_l1201_120155

-- Define the circle and point P(x,y) on the circle
def CircleEquation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9

theorem range_of_x (x y : ℝ) (h : CircleEquation x y) : 1 ≤ x ∧ x ≤ 7 :=
sorry

theorem max_y_over_x (x y : ℝ) (h : CircleEquation x y) : ∀ k : ℝ, (k = y / x) → 0 ≤ k ∧ k ≤ (24 / 7) :=
sorry

end range_of_x_max_y_over_x_l1201_120155


namespace oreo_solution_l1201_120130

noncomputable def oreo_problem : Prop :=
∃ (m : ℤ), (11 + m * 11 + 3 = 36) → m = 2

theorem oreo_solution : oreo_problem :=
sorry

end oreo_solution_l1201_120130


namespace line_equation_l1201_120190

theorem line_equation (x y : ℝ) (h : (2, 3) ∈ {p : ℝ × ℝ | (∃ a, p.1 + p.2 = a) ∨ (∃ k, p.2 = k * p.1)}) :
  (3 * x - 2 * y = 0) ∨ (x + y - 5 = 0) :=
sorry

end line_equation_l1201_120190


namespace total_blue_points_l1201_120179

variables (a b c d : ℕ)

theorem total_blue_points (h1 : a * b = 56) (h2 : c * d = 50) (h3 : a + b = c + d) :
  a + b = 15 :=
sorry

end total_blue_points_l1201_120179


namespace m_ducks_l1201_120169

variable (M C K : ℕ)

theorem m_ducks :
  (M = C + 4) ∧
  (M = 2 * C + K + 3) ∧
  (M + C + K = 90) →
  M = 89 := by
  sorry

end m_ducks_l1201_120169


namespace quadratic_two_distinct_real_roots_l1201_120191

theorem quadratic_two_distinct_real_roots (k : ℝ) :
    (∃ x : ℝ, 2 * k * x^2 + (8 * k + 1) * x + 8 * k = 0 ∧ 2 * k ≠ 0) →
    k > -1/16 ∧ k ≠ 0 :=
by
  intro h
  sorry

end quadratic_two_distinct_real_roots_l1201_120191


namespace Borgnine_total_legs_l1201_120163

def numChimps := 12
def numLions := 8
def numLizards := 5
def numTarantulas := 125

def chimpLegsEach := 2
def lionLegsEach := 4
def lizardLegsEach := 4
def tarantulaLegsEach := 8

def legsSeen := numChimps * chimpLegsEach +
                numLions * lionLegsEach +
                numLizards * lizardLegsEach

def legsToSee := numTarantulas * tarantulaLegsEach

def totalLegs := legsSeen + legsToSee

theorem Borgnine_total_legs : totalLegs = 1076 := by
  sorry

end Borgnine_total_legs_l1201_120163


namespace permutation_value_l1201_120156

theorem permutation_value (n : ℕ) (h : n * (n - 1) = 12) : n = 4 :=
by
  sorry

end permutation_value_l1201_120156


namespace compute_product_fraction_l1201_120121

theorem compute_product_fraction :
  ( ((3 : ℚ)^4 - 1) / ((3 : ℚ)^4 + 1) *
    ((4 : ℚ)^4 - 1) / ((4 : ℚ)^4 + 1) * 
    ((5 : ℚ)^4 - 1) / ((5 : ℚ)^4 + 1) *
    ((6 : ℚ)^4 - 1) / ((6 : ℚ)^4 + 1) *
    ((7 : ℚ)^4 - 1) / ((7 : ℚ)^4 + 1)
  ) = (25 / 210) := 
  sorry

end compute_product_fraction_l1201_120121


namespace cost_of_chlorine_l1201_120120

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end cost_of_chlorine_l1201_120120


namespace danny_bottle_caps_after_collection_l1201_120144

-- Definitions for the conditions
def initial_bottle_caps : ℕ := 69
def bottle_caps_thrown : ℕ := 60
def bottle_caps_found : ℕ := 58

-- Theorem stating the proof problem
theorem danny_bottle_caps_after_collection : 
  initial_bottle_caps - bottle_caps_thrown + bottle_caps_found = 67 :=
by {
  -- Placeholder for proof
  sorry
}

end danny_bottle_caps_after_collection_l1201_120144


namespace expression_value_l1201_120184

theorem expression_value : (4 - 2) ^ 3 = 8 :=
by sorry

end expression_value_l1201_120184


namespace tony_pool_filling_time_l1201_120145

theorem tony_pool_filling_time
  (J S T : ℝ)
  (hJ : J = 1 / 30)
  (hS : S = 1 / 45)
  (hCombined : J + S + T = 1 / 15) :
  T = 1 / 90 :=
by
  -- the setup for proof would be here
  sorry

end tony_pool_filling_time_l1201_120145


namespace inequality_solution_l1201_120199

theorem inequality_solution (x : ℝ) : x ∈ Set.Ioo (-7 : ℝ) (7 : ℝ) ↔ (x^2 - 49) / (x + 7) < 0 :=
by 
  sorry

end inequality_solution_l1201_120199


namespace ratio_of_edges_l1201_120152

theorem ratio_of_edges
  (V₁ V₂ : ℝ)
  (a b : ℝ)
  (hV : V₁ / V₂ = 8 / 1)
  (hV₁ : V₁ = a^3)
  (hV₂ : V₂ = b^3) :
  a / b = 2 / 1 := 
by 
  sorry

end ratio_of_edges_l1201_120152


namespace nine_x_plus_twenty_seven_y_l1201_120188

theorem nine_x_plus_twenty_seven_y (x y : ℤ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := 
by sorry

end nine_x_plus_twenty_seven_y_l1201_120188


namespace proof_theorem_l1201_120174

noncomputable def proof_problem (a b c : ℝ) := 
  (2 * b = a + c) ∧ 
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) ∧ 
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)

theorem proof_theorem (a b c : ℝ) (h : proof_problem a b c) :
  (a = b ∧ b = c) ∨ 
  (∃ (x : ℝ), x ≠ 0 ∧ a = -4 * x ∧ b = -x ∧ c = 2 * x) :=
by
  sorry

end proof_theorem_l1201_120174


namespace basic_astrophysics_budget_percent_l1201_120122

theorem basic_astrophysics_budget_percent
  (total_degrees : ℝ := 360)
  (astrophysics_degrees : ℝ := 108) :
  (astrophysics_degrees / total_degrees) * 100 = 30 := by
  sorry

end basic_astrophysics_budget_percent_l1201_120122


namespace model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l1201_120175

theorem model_to_statue_ratio_inch_per_feet (statue_height_ft : ℝ) (model_height_in : ℝ) :
  statue_height_ft = 120 → model_height_in = 6 → (120 / 6 = 20)
:= by
  intros h1 h2
  sorry

theorem model_inches_for_statue_feet (model_per_inch_feet : ℝ) :
  model_per_inch_feet = 20 → (30 / 20 = 1.5)
:= by
  intros h
  sorry

end model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l1201_120175


namespace min_product_value_max_product_value_l1201_120127

open Real

noncomputable def min_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

noncomputable def max_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

theorem min_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ min_cos_sin_product x y z = 1 / 8 :=
sorry

theorem max_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ max_cos_sin_product x y z = (2 + sqrt 3) / 8 :=
sorry

end min_product_value_max_product_value_l1201_120127


namespace find_k_value_l1201_120133

theorem find_k_value (x k : ℝ) (h : x = 2) (h_sol : (k / (x - 3)) - (1 / (3 - x)) = 1) : k = -2 :=
by
  -- sorry to suppress the actual proof
  sorry

end find_k_value_l1201_120133


namespace sum_n_k_l1201_120192

theorem sum_n_k (n k : ℕ) (h₁ : (x+1)^n = 2 * x^k + 3 * x^(k+1) + 4 * x^(k+2)) (h₂ : 3 * k + 3 = 2 * n - 2 * k)
  (h₃ : 4 * k + 8 = 3 * n - 3 * k - 3) : n + k = 47 := 
sorry

end sum_n_k_l1201_120192


namespace graph_symmetry_about_line_l1201_120180

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - (Real.pi / 3))

theorem graph_symmetry_about_line (x : ℝ) : 
  ∀ x, f (2 * (Real.pi / 3) - x) = f x :=
by
  sorry

end graph_symmetry_about_line_l1201_120180


namespace grooming_time_correct_l1201_120109

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end grooming_time_correct_l1201_120109


namespace combined_cost_is_correct_l1201_120107

-- Definitions based on the conditions
def dryer_cost : ℕ := 150
def washer_cost : ℕ := 3 * dryer_cost
def combined_cost : ℕ := dryer_cost + washer_cost

-- Statement to be proved
theorem combined_cost_is_correct : combined_cost = 600 :=
by
  sorry

end combined_cost_is_correct_l1201_120107


namespace f_is_odd_l1201_120140

open Real

def f (x : ℝ) : ℝ := x^3 + x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end f_is_odd_l1201_120140


namespace parabola_directrix_value_l1201_120183

noncomputable def parabola_p_value (p : ℝ) : Prop :=
(∀ y : ℝ, y^2 = 2 * p * (-2 - (-2)))

theorem parabola_directrix_value : parabola_p_value 4 :=
by
  -- proof steps here
  sorry

end parabola_directrix_value_l1201_120183


namespace roots_in_arithmetic_progression_l1201_120142

theorem roots_in_arithmetic_progression (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x2 = (x1 + x3) / 2) ∧ (x1 + x2 + x3 = -a) ∧ (x1 * x3 + x2 * (x1 + x3) = b) ∧ (x1 * x2 * x3 = -c)) ↔ 
  (27 * c = 3 * a * b - 2 * a^3 ∧ 3 * b ≤ a^2) :=
sorry

end roots_in_arithmetic_progression_l1201_120142


namespace remainder_of_504_divided_by_100_is_4_l1201_120195

theorem remainder_of_504_divided_by_100_is_4 :
  (504 % 100) = 4 :=
by
  sorry

end remainder_of_504_divided_by_100_is_4_l1201_120195


namespace shopper_saved_percentage_l1201_120125

-- Definition of the problem conditions
def amount_saved : ℝ := 4
def amount_spent : ℝ := 36

-- Lean 4 statement to prove the percentage saved
theorem shopper_saved_percentage : (amount_saved / (amount_spent + amount_saved)) * 100 = 10 := by
  sorry

end shopper_saved_percentage_l1201_120125


namespace total_students_in_class_l1201_120148

/-- 
There are 208 boys in the class.
There are 69 more girls than boys.
The total number of students in the class is the sum of boys and girls.
Prove that the total number of students in the graduating class is 485.
-/
theorem total_students_in_class (boys girls : ℕ) (h1 : boys = 208) (h2 : girls = boys + 69) : 
  boys + girls = 485 :=
by
  sorry

end total_students_in_class_l1201_120148


namespace triangle_area_l1201_120150

theorem triangle_area (base height : ℝ) (h_base : base = 8.4) (h_height : height = 5.8) :
  0.5 * base * height = 24.36 := by
  sorry

end triangle_area_l1201_120150


namespace radius_of_sphere_eq_l1201_120110

theorem radius_of_sphere_eq (r : ℝ) : 
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 :=
by
  sorry

end radius_of_sphere_eq_l1201_120110


namespace sum_of_roots_3x2_minus_12x_plus_12_eq_4_l1201_120135

def sum_of_roots_quadratic (a b : ℚ) (h : a ≠ 0) : ℚ := -b / a

theorem sum_of_roots_3x2_minus_12x_plus_12_eq_4 :
  sum_of_roots_quadratic 3 (-12) (by norm_num) = 4 :=
sorry

end sum_of_roots_3x2_minus_12x_plus_12_eq_4_l1201_120135


namespace solve_for_f_sqrt_2_l1201_120132

theorem solve_for_f_sqrt_2 (f : ℝ → ℝ) (h : ∀ x, f x = 2 / (2 - x)) : f (Real.sqrt 2) = 2 + Real.sqrt 2 :=
by
  sorry

end solve_for_f_sqrt_2_l1201_120132


namespace no_common_root_of_quadratics_l1201_120166

theorem no_common_root_of_quadratics (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, (x₀^2 + b * x₀ + c = 0 ∧ x₀^2 + a * x₀ + d = 0) := 
by
  sorry

end no_common_root_of_quadratics_l1201_120166


namespace find_all_real_solutions_l1201_120134

theorem find_all_real_solutions (x : ℝ) :
    (1 / ((x - 1) * (x - 2))) + (1 / ((x - 2) * (x - 3))) + (1 / ((x - 3) * (x - 4))) + (1 / ((x - 4) * (x - 5))) = 1 / 4 →
    x = 1 ∨ x = 5 :=
by
  sorry

end find_all_real_solutions_l1201_120134


namespace inverse_proposition_l1201_120106

-- Definition of the proposition
def complementary_angles_on_same_side (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- The original proposition
def original_proposition (l m : Line) : Prop := complementary_angles_on_same_side l m → parallel_lines l m

-- The statement of the proof problem
theorem inverse_proposition (l m : Line) :
  (complementary_angles_on_same_side l m → parallel_lines l m) →
  (parallel_lines l m → complementary_angles_on_same_side l m) := sorry

end inverse_proposition_l1201_120106


namespace inequality_problem_l1201_120143

open Real

theorem inequality_problem {a b c d : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_ac : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3 := 
by 
  sorry

end inequality_problem_l1201_120143


namespace infinite_impossible_values_of_d_l1201_120115

theorem infinite_impossible_values_of_d 
  (pentagon_perimeter square_perimeter : ℕ) 
  (d : ℕ) 
  (h1 : pentagon_perimeter = 5 * (d + ((square_perimeter) / 4)) )
  (h2 : square_perimeter > 0)
  (h3 : pentagon_perimeter - square_perimeter = 2023) :
  ∀ n : ℕ, n > 404 → ¬∃ d : ℕ, d = n :=
by {
  sorry
}

end infinite_impossible_values_of_d_l1201_120115


namespace range_of_a_l1201_120186

-- Defining the propositions P and Q 
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x1 x2 : ℝ, x1^2 - x1 + a = 0 ∧ x2^2 - x2 + a = 0

-- Stating the theorem
theorem range_of_a (a : ℝ) :
  (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a) ↔ a ∈ Set.Ioo (1/4 : ℝ) 4 ∪ Set.Iio 0 :=
sorry

end range_of_a_l1201_120186


namespace customer_payment_l1201_120101

noncomputable def cost_price : ℝ := 4090.9090909090905
noncomputable def markup : ℝ := 0.32
noncomputable def selling_price : ℝ := cost_price * (1 + markup)

theorem customer_payment :
  selling_price = 5400 :=
by
  unfold selling_price
  unfold cost_price
  unfold markup
  sorry

end customer_payment_l1201_120101


namespace solution_set_of_inequality_l1201_120197

theorem solution_set_of_inequality : 
  {x : ℝ | |x|^3 - 2 * x^2 - 4 * |x| + 3 < 0} = 
  { x : ℝ | -3 < x ∧ x < -1 } ∪ { x : ℝ | 1 < x ∧ x < 3 } := 
by
  sorry

end solution_set_of_inequality_l1201_120197


namespace parabola_vertex_coordinates_l1201_120193

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, -x^2 + 15 ≥ -x^2 + 15 :=
by
  sorry

end parabola_vertex_coordinates_l1201_120193


namespace sum_of_roots_of_qubic_polynomial_l1201_120136

noncomputable def Q (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_qubic_polynomial (a b c d : ℝ) 
  (h₁ : ∀ x : ℝ, Q a b c d (x^4 + x) ≥ Q a b c d (x^3 + 1))
  (h₂ : Q a b c d 1 = 0) : 
  -b / a = 3 / 2 :=
sorry

end sum_of_roots_of_qubic_polynomial_l1201_120136


namespace crushing_load_value_l1201_120172

-- Define the given formula and values
def T : ℕ := 3
def H : ℕ := 9
def K : ℕ := 2

-- Given formula for L
def L (T H K : ℕ) : ℚ := 50 * T^5 / (K * H^3)

-- Prove that L = 8 + 1/3
theorem crushing_load_value :
  L T H K = 8 + 1 / 3 :=
by
  sorry

end crushing_load_value_l1201_120172


namespace member_sum_or_double_exists_l1201_120138

theorem member_sum_or_double_exists (n : ℕ) (k : ℕ) (P: ℕ → ℕ) (m: ℕ) 
  (h_mem : n = 1978)
  (h_countries : m = 6) : 
  ∃ k, (∃ i j, P i + P j = k ∧ P i = P j)
    ∨ (∃ i, 2 * P i = k) :=
sorry

end member_sum_or_double_exists_l1201_120138


namespace proof_problem_l1201_120105

-- Define the conditions: n is a positive integer and (n(n + 1) / 3) is a square
def problem_condition (n : ℕ) : Prop :=
  ∃ m : ℕ, n > 0 ∧ (n * (n + 1)) = 3 * m^2

-- Define the proof problem: given the condition, n is a multiple of 3, n+1 and n/3 are squares
theorem proof_problem (n : ℕ) (h : problem_condition n) : 
  (∃ a : ℕ, n = 3 * a^2) ∧ 
  (∃ b : ℕ, n + 1 = b^2) ∧ 
  (∃ c : ℕ, n = 3 * c^2) :=
sorry

end proof_problem_l1201_120105


namespace equation_of_line_passing_through_center_and_perpendicular_to_l_l1201_120123

theorem equation_of_line_passing_through_center_and_perpendicular_to_l (a : ℝ) : 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  ∃ (b : ℝ), ∀ x y : ℝ, (x + y + 1 = 0) := 
by 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  use 1
  sorry

end equation_of_line_passing_through_center_and_perpendicular_to_l_l1201_120123


namespace student_chose_number_l1201_120119

theorem student_chose_number (x : ℕ) (h : 2 * x - 138 = 112) : x = 125 :=
by
  sorry

end student_chose_number_l1201_120119


namespace sine_wave_solution_l1201_120151

theorem sine_wave_solution (a b c : ℝ) (h_pos_a : a > 0) 
  (h_amp : a = 3) 
  (h_period : (2 * Real.pi) / b = Real.pi) 
  (h_peak : (Real.pi / (2 * b)) - (c / b) = Real.pi / 6) : 
  a = 3 ∧ b = 2 ∧ c = Real.pi / 6 :=
by
  -- Lean code to construct the proof will appear here
  sorry

end sine_wave_solution_l1201_120151


namespace min_unsuccessful_placements_8x8_l1201_120159

-- Define the board, the placement, and the unsuccessful condition
def is_unsuccessful_placement (board : ℕ → ℕ → ℤ) (i j : ℕ) : Prop :=
  (i < 7 ∧ j < 7 ∧ (board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1)) ≠ 0)

-- Main theorem: The minimum number of unsuccessful placements is 36 on an 8x8 board
theorem min_unsuccessful_placements_8x8 (board : ℕ → ℕ → ℤ) (H : ∀ i j, board i j = 1 ∨ board i j = -1) :
  ∃ (n : ℕ), n = 36 ∧ (∀ m : ℕ, (∀ i j, is_unsuccessful_placement board i j → m < 36 ) → m = n) :=
sorry

end min_unsuccessful_placements_8x8_l1201_120159


namespace difference_of_two_numbers_l1201_120171

theorem difference_of_two_numbers :
  ∃ S : ℕ, S * 16 + 15 = 1600 ∧ 1600 - S = 1501 :=
by
  sorry

end difference_of_two_numbers_l1201_120171
