import Mathlib

namespace alpha_range_theorem_l1683_168377

noncomputable def alpha_range (k : ℤ) (α : ℝ) : Prop :=
  2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi

theorem alpha_range_theorem (α : ℝ) (k : ℤ) (h : |Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) :
  alpha_range k α :=
by
  sorry

end alpha_range_theorem_l1683_168377


namespace vendor_has_maaza_l1683_168324

theorem vendor_has_maaza (liters_pepsi : ℕ) (liters_sprite : ℕ) (total_cans : ℕ) (gcd_pepsi_sprite : ℕ) (cans_pepsi : ℕ) (cans_sprite : ℕ) (cans_maaza : ℕ) (liters_per_can : ℕ) (total_liters_maaza : ℕ) :
  liters_pepsi = 144 →
  liters_sprite = 368 →
  total_cans = 133 →
  gcd_pepsi_sprite = Nat.gcd liters_pepsi liters_sprite →
  gcd_pepsi_sprite = 16 →
  cans_pepsi = liters_pepsi / gcd_pepsi_sprite →
  cans_sprite = liters_sprite / gcd_pepsi_sprite →
  cans_maaza = total_cans - (cans_pepsi + cans_sprite) →
  liters_per_can = gcd_pepsi_sprite →
  total_liters_maaza = cans_maaza * liters_per_can →
  total_liters_maaza = 1616 :=
by
  sorry

end vendor_has_maaza_l1683_168324


namespace dealer_sold_70_hondas_l1683_168300

theorem dealer_sold_70_hondas
  (total_cars: ℕ)
  (percent_audi percent_toyota percent_acura percent_honda : ℝ)
  (total_audi := total_cars * percent_audi)
  (total_toyota := total_cars * percent_toyota)
  (total_acura := total_cars * percent_acura)
  (total_honda := total_cars * percent_honda )
  (h1 : total_cars = 200)
  (h2 : percent_audi = 0.15)
  (h3 : percent_toyota = 0.22)
  (h4 : percent_acura = 0.28)
  (h5 : percent_honda = 1 - (percent_audi + percent_toyota + percent_acura))
  : total_honda = 70 := 
  by
  sorry

end dealer_sold_70_hondas_l1683_168300


namespace seq_bound_gt_pow_two_l1683_168393

theorem seq_bound_gt_pow_two (a : Fin 101 → ℕ) 
  (h1 : a 1 > a 0) 
  (h2 : ∀ n : Fin 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2 ^ 99 :=
sorry

end seq_bound_gt_pow_two_l1683_168393


namespace system_of_equations_solution_l1683_168328

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 x4 x5 : ℝ),
  (x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1) ∧
  (x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2) ∧
  (x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) ∧
  (x1 = 1) ∧ (x2 = -1) ∧ (x3 = 1) ∧ (x4 = -1) ∧ (x5 = 1) := by
sorry

end system_of_equations_solution_l1683_168328


namespace raised_bed_section_area_l1683_168327

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end raised_bed_section_area_l1683_168327


namespace ratio_netbooks_is_one_third_l1683_168303

open Nat

def total_computers (total : ℕ) : Prop := total = 72
def laptops_sold (laptops : ℕ) (total : ℕ) : Prop := laptops = total / 2
def desktops_sold (desktops : ℕ) : Prop := desktops = 12
def netbooks_sold (netbooks : ℕ) (total laptops desktops : ℕ) : Prop :=
  netbooks = total - (laptops + desktops)
def ratio_netbooks_total (netbooks total : ℕ) : Prop :=
  netbooks * 3 = total

theorem ratio_netbooks_is_one_third
  (total laptops desktops netbooks : ℕ)
  (h_total : total_computers total)
  (h_laptops : laptops_sold laptops total)
  (h_desktops : desktops_sold desktops)
  (h_netbooks : netbooks_sold netbooks total laptops desktops) :
  ratio_netbooks_total netbooks total :=
by
  sorry

end ratio_netbooks_is_one_third_l1683_168303


namespace paint_house_l1683_168326

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end paint_house_l1683_168326


namespace perpendicular_lines_slope_l1683_168362

theorem perpendicular_lines_slope (m : ℝ) : 
  ((m ≠ -3) ∧ (m ≠ -5) ∧ 
  (- (m + 3) / 4 * - (2 / (m + 5)) = -1)) ↔ m = -13 / 3 := 
sorry

end perpendicular_lines_slope_l1683_168362


namespace shifted_function_is_correct_l1683_168310

def original_function (x : ℝ) : ℝ :=
  (x - 1)^2 + 2

def shifted_up_function (x : ℝ) : ℝ :=
  original_function x + 3

def shifted_right_function (x : ℝ) : ℝ :=
  shifted_up_function (x - 4)

theorem shifted_function_is_correct : ∀ x : ℝ, shifted_right_function x = (x - 5)^2 + 5 := 
by
  sorry

end shifted_function_is_correct_l1683_168310


namespace num_four_digit_with_5_or_7_l1683_168344

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l1683_168344


namespace seeds_total_l1683_168398

def seedsPerWatermelon : Nat := 345
def numberOfWatermelons : Nat := 27
def totalSeeds : Nat := seedsPerWatermelon * numberOfWatermelons

theorem seeds_total :
  totalSeeds = 9315 :=
by
  sorry

end seeds_total_l1683_168398


namespace red_blood_cells_surface_area_l1683_168360

-- Define the body surface area of an adult
def body_surface_area : ℝ := 1800

-- Define the multiplying factor for the surface areas of red blood cells
def multiplier : ℝ := 2000

-- Define the sum of the surface areas of all red blood cells
def sum_surface_area : ℝ := multiplier * body_surface_area

-- Define the expected sum in scientific notation
def expected_sum : ℝ := 3.6 * 10^6

-- The theorem that needs to be proved
theorem red_blood_cells_surface_area :
  sum_surface_area = expected_sum :=
by
  sorry

end red_blood_cells_surface_area_l1683_168360


namespace rate_of_interest_per_annum_l1683_168383

theorem rate_of_interest_per_annum (P R : ℝ) (T : ℝ) 
  (h1 : T = 8)
  (h2 : (P / 5) = (P * R * T) / 100) : 
  R = 2.5 := 
by
  sorry

end rate_of_interest_per_annum_l1683_168383


namespace andrew_bought_6_kg_of_grapes_l1683_168332

def rate_grapes := 74
def rate_mangoes := 59
def kg_mangoes := 9
def total_paid := 975

noncomputable def number_of_kg_grapes := 6

theorem andrew_bought_6_kg_of_grapes :
  ∃ G : ℕ, (rate_grapes * G + rate_mangoes * kg_mangoes = total_paid) ∧ G = number_of_kg_grapes := 
by
  sorry

end andrew_bought_6_kg_of_grapes_l1683_168332


namespace initial_spinach_volume_l1683_168318

theorem initial_spinach_volume (S : ℝ) (h1 : 0.20 * S + 6 + 4 = 18) : S = 40 :=
by
  sorry

end initial_spinach_volume_l1683_168318


namespace shaded_region_area_proof_l1683_168358

/-- Define the geometric properties of the problem -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

noncomputable def shaded_region_area (rect : Rectangle) (circle1 circle2 : Circle) : ℝ :=
  let rect_area := rect.width * rect.height
  let circle_area := (Real.pi * circle1.radius ^ 2) + (Real.pi * circle2.radius ^ 2)
  rect_area - circle_area

theorem shaded_region_area_proof : shaded_region_area 
  {width := 10, height := 12} 
  {radius := 3, center := (0, 0)} 
  {radius := 3, center := (12, 10)} = 120 - 18 * Real.pi :=
by
  sorry

end shaded_region_area_proof_l1683_168358


namespace notebook_costs_2_20_l1683_168329

theorem notebook_costs_2_20 (n c : ℝ) (h1 : n + c = 2.40) (h2 : n = 2 + c) : n = 2.20 :=
by
  sorry

end notebook_costs_2_20_l1683_168329


namespace length_PD_l1683_168306

theorem length_PD (PA PB PC PD : ℝ) (hPA : PA = 5) (hPB : PB = 3) (hPC : PC = 4) :
  PD = 4 * Real.sqrt 2 :=
by
  sorry

end length_PD_l1683_168306


namespace curve_is_line_l1683_168314

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end curve_is_line_l1683_168314


namespace base7_first_digit_l1683_168399

noncomputable def first_base7_digit : ℕ := 625

theorem base7_first_digit (n : ℕ) (h : n = 625) : ∃ (d : ℕ), d = 12 ∧ (d * 49 ≤ n) ∧ (n < (d + 1) * 49) :=
by
  sorry

end base7_first_digit_l1683_168399


namespace solution_set_l1683_168337

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l1683_168337


namespace count_distinct_ways_l1683_168387

theorem count_distinct_ways (p : ℕ × ℕ → ℕ) (h_condition : ∃ j : ℕ × ℕ, j ∈ [(0, 0), (0, 1)] ∧ p j = 4)
  (h_grid_size : ∀ i : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → 1 ≤ p i ∧ p i ≤ 4)
  (h_distinct : ∀ i j : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → j ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → i ≠ j → p i ≠ p j) :
  ∃! l : Finset (ℕ × ℕ → ℕ), l.card = 12 :=
by
  sorry

end count_distinct_ways_l1683_168387


namespace real_seq_proof_l1683_168349

noncomputable def real_seq_ineq (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1

theorem real_seq_proof (a : ℕ → ℝ) (h : real_seq_ineq a) :
  ∀ k m : ℕ, k > 0 → m > 0 → |a k / k - a m / m| < 1 / k + 1 / m :=
by
  sorry

end real_seq_proof_l1683_168349


namespace cards_arrangement_count_is_10_l1683_168323

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l1683_168323


namespace ironman_age_greater_than_16_l1683_168381

variable (Ironman_age : ℕ)
variable (Thor_age : ℕ := 1456)
variable (CaptainAmerica_age : ℕ := Thor_age / 13)
variable (PeterParker_age : ℕ := CaptainAmerica_age / 7)

theorem ironman_age_greater_than_16
  (Thor_13_times_CaptainAmerica : Thor_age = 13 * CaptainAmerica_age)
  (CaptainAmerica_7_times_PeterParker : CaptainAmerica_age = 7 * PeterParker_age)
  (Thor_age_given : Thor_age = 1456) :
  Ironman_age > 16 :=
by
  sorry

end ironman_age_greater_than_16_l1683_168381


namespace number_of_possible_orders_l1683_168397

-- Define the total number of bowlers participating in the playoff
def num_bowlers : ℕ := 6

-- Define the number of games
def num_games : ℕ := 5

-- Define the number of possible outcomes per game
def outcomes_per_game : ℕ := 2

-- Prove the total number of possible orders for bowlers to receive prizes
theorem number_of_possible_orders : (outcomes_per_game ^ num_games) = 32 :=
by sorry

end number_of_possible_orders_l1683_168397


namespace circle_eq_of_points_value_of_m_l1683_168322

-- Define the points on the circle
def P : ℝ × ℝ := (0, -4)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (3, -1)

-- Statement 1: The equation of the circle passing through P, Q, and R
theorem circle_eq_of_points (C : ℝ × ℝ → Prop) :
  (C P ∧ C Q ∧ C R) ↔ ∀ x y : ℝ, C (x, y) ↔ (x - 1)^2 + (y + 2)^2 = 5 := sorry

-- Define the line intersecting the circle and the chord length condition |AB| = 4
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

-- Statement 2: The value of m such that the chord length |AB| is 4
theorem value_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) → m = 4 / 3 := sorry

end circle_eq_of_points_value_of_m_l1683_168322


namespace find_g3_l1683_168348

variable {g : ℝ → ℝ}

-- Defining the condition from the problem
def g_condition (x : ℝ) (h : x ≠ 0) : g x - 3 * g (1 / x) = 3^x + x^2 := sorry

-- The main statement to prove
theorem find_g3 : g 3 = - (3 * 3^(1/3) + 1/3 + 36) / 8 := sorry

end find_g3_l1683_168348


namespace sum_n_terms_max_sum_n_l1683_168390

variable {a : ℕ → ℚ} (S : ℕ → ℚ)
variable (d a_1 : ℚ)

-- Conditions given in the problem
axiom sum_first_10 : S 10 = 125 / 7
axiom sum_first_20 : S 20 = -250 / 7
axiom sum_arithmetic_seq : ∀ n, S n = n * (a 1 + a n) / 2

-- Define the first term and common difference for the arithmetic sequence
axiom common_difference : ∀ n, a n = a_1 + (n - 1) * d

-- Theorem 1: Sum of the first n terms
theorem sum_n_terms (n : ℕ) : S n = (75 * n - 5 * n^2) / 14 := 
  sorry

-- Theorem 2: Value of n that maximizes S_n
theorem max_sum_n : n = 7 ∨ n = 8 ↔ (∀ m, S m ≤ S 7 ∨ S m ≤ S 8) := 
  sorry

end sum_n_terms_max_sum_n_l1683_168390


namespace ellipse_transform_circle_l1683_168396

theorem ellipse_transform_circle (a b x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b)
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (y' : ℝ)
  (h_transform : y' = (a / b) * y) :
  x^2 + y'^2 = a^2 :=
by
  sorry

end ellipse_transform_circle_l1683_168396


namespace original_price_of_shoes_l1683_168367

theorem original_price_of_shoes (
  initial_amount : ℝ := 74
) (sweater_cost : ℝ := 9) (tshirt_cost : ℝ := 11) 
  (final_amount_after_refund : ℝ := 51)
  (refund_percentage : ℝ := 0.90)
  (S : ℝ) :
  (initial_amount - sweater_cost - tshirt_cost - S + refund_percentage * S = final_amount_after_refund) -> 
  S = 30 := 
by
  intros h
  sorry

end original_price_of_shoes_l1683_168367


namespace find_m_l1683_168353

open Real

-- Definitions based on problem conditions
def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

-- The dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Prove the final statement using given conditions
theorem find_m (m : ℝ) (h1 : dot_product (vector_a m) vector_b + dot_product vector_b vector_b = 0) :
  m = 8 :=
sorry

end find_m_l1683_168353


namespace thirteen_y_minus_x_l1683_168312

theorem thirteen_y_minus_x (x y : ℤ) (hx1 : x = 11 * y + 4) (hx2 : 2 * x = 8 * (3 * y) + 3) : 13 * y - x = 1 :=
by
  sorry

end thirteen_y_minus_x_l1683_168312


namespace existential_inequality_false_iff_l1683_168365

theorem existential_inequality_false_iff {a : ℝ} :
  (∀ x : ℝ, x^2 + a * x - 2 * a ≥ 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
by
  sorry

end existential_inequality_false_iff_l1683_168365


namespace tolya_is_older_by_either_4_or_22_years_l1683_168319

-- Definitions of the problem conditions
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def kolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2013

def tolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2014

-- The problem statement
theorem tolya_is_older_by_either_4_or_22_years (k_birth t_birth : ℕ) 
  (hk : kolya_conditions k_birth) (ht : tolya_conditions t_birth) :
  t_birth - k_birth = 4 ∨ t_birth - k_birth = 22 :=
sorry

end tolya_is_older_by_either_4_or_22_years_l1683_168319


namespace pq_square_eq_169_div_4_l1683_168385

-- Defining the quadratic equation and the condition on solutions p and q.
def quadratic_eq (x : ℚ) : Prop := 2 * x^2 + 7 * x - 15 = 0

-- Defining the specific solutions p and q.
def p : ℚ := 3 / 2
def q : ℚ := -5

-- The main theorem stating that (p - q)^2 = 169 / 4 given the conditions.
theorem pq_square_eq_169_div_4 (hp : quadratic_eq p) (hq : quadratic_eq q) : (p - q) ^ 2 = 169 / 4 :=
by
  -- Proof is omitted using sorry
  sorry

end pq_square_eq_169_div_4_l1683_168385


namespace ratio_of_radii_l1683_168364

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end ratio_of_radii_l1683_168364


namespace area_difference_of_circles_l1683_168341

theorem area_difference_of_circles (circumference_large: ℝ) (half_radius_relation: ℝ → ℝ) (hl: circumference_large = 36) (hr: ∀ R, half_radius_relation R = R / 2) :
  ∃ R r, R = 18 / π ∧ r = 9 / π ∧ (π * R ^ 2 - π * r ^ 2) = 243 / π :=
by 
  sorry

end area_difference_of_circles_l1683_168341


namespace fraction_of_plot_occupied_by_beds_l1683_168339

-- Define the conditions based on plot area and number of beds
def plot_area : ℕ := 64
def total_beds : ℕ := 13
def outer_beds : ℕ := 12
def central_bed_area : ℕ := 4 * 4

-- The proof statement showing that fraction of the plot occupied by the beds is 15/32
theorem fraction_of_plot_occupied_by_beds : 
  (central_bed_area + (plot_area - central_bed_area)) / plot_area = 15 / 32 := 
sorry

end fraction_of_plot_occupied_by_beds_l1683_168339


namespace cos_expression_l1683_168325

-- Define the condition for the line l and its relationship
def slope_angle_of_line_l (α : ℝ) : Prop :=
  ∃ l : ℝ, l = 2

-- Given the tangent condition for α
def tan_alpha (α : ℝ) : Prop :=
  Real.tan α = 2

theorem cos_expression (α : ℝ) (h1 : slope_angle_of_line_l α) (h2 : tan_alpha α) :
  Real.cos (2015 * Real.pi / 2 - 2 * α) = -4/5 :=
by sorry

end cos_expression_l1683_168325


namespace max_distance_from_curve_to_line_l1683_168355

theorem max_distance_from_curve_to_line
  (θ : ℝ) (t : ℝ)
  (C_polar_eqn : ∀ θ, ∃ (ρ : ℝ), ρ = 2 * Real.cos θ)
  (line_eqn : ∀ t, ∃ (x y : ℝ), x = -1 + t ∧ y = 2 * t) :
  ∃ (max_dist : ℝ), max_dist = (4 * Real.sqrt 5 + 5) / 5 := sorry

end max_distance_from_curve_to_line_l1683_168355


namespace partial_fraction_product_l1683_168351

theorem partial_fraction_product (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x^2 - 10 * x + 24 ≠ 0 →
            (x^2 - 25) / (x^3 - 3 * x^2 - 10 * x + 24) = A / (x - 2) + B / (x + 3) + C / (x - 4)) →
  A = 1 ∧ B = 1 ∧ C = 1 →
  A * B * C = 1 := by
  sorry

end partial_fraction_product_l1683_168351


namespace sophomores_sampled_correct_l1683_168330

def stratified_sampling_sophomores (total_students num_sophomores sample_size : ℕ) : ℕ :=
  (num_sophomores * sample_size) / total_students

theorem sophomores_sampled_correct :
  stratified_sampling_sophomores 4500 1500 600 = 200 :=
by
  sorry

end sophomores_sampled_correct_l1683_168330


namespace selection_structure_count_is_three_l1683_168301

def requiresSelectionStructure (problem : ℕ) : Bool :=
  match problem with
  | 1 => true
  | 2 => false
  | 3 => true
  | 4 => true
  | _ => false

def countSelectionStructure : ℕ :=
  (if requiresSelectionStructure 1 then 1 else 0) +
  (if requiresSelectionStructure 2 then 1 else 0) +
  (if requiresSelectionStructure 3 then 1 else 0) +
  (if requiresSelectionStructure 4 then 1 else 0)

theorem selection_structure_count_is_three : countSelectionStructure = 3 :=
  by
    sorry

end selection_structure_count_is_three_l1683_168301


namespace book_loss_percentage_l1683_168382

theorem book_loss_percentage (CP SP_profit SP_loss : ℝ) (L : ℝ) 
  (h1 : CP = 50) 
  (h2 : SP_profit = CP + 0.09 * CP) 
  (h3 : SP_loss = CP - L / 100 * CP) 
  (h4 : SP_profit - SP_loss = 9) : 
  L = 9 :=
by
  sorry

end book_loss_percentage_l1683_168382


namespace negation_of_existence_statement_l1683_168343

theorem negation_of_existence_statement :
  (¬ (∃ x : ℝ, x^2 + x + 1 < 0)) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l1683_168343


namespace points_per_member_l1683_168317

theorem points_per_member
    (total_members : ℕ)
    (absent_members : ℕ)
    (total_points : ℕ)
    (present_members : ℕ)
    (points_per_member : ℕ)
    (h1 : total_members = 5)
    (h2 : absent_members = 2)
    (h3 : total_points = 18)
    (h4 : present_members = total_members - absent_members)
    (h5 : points_per_member = total_points / present_members) :
  points_per_member = 6 :=
by
  sorry

end points_per_member_l1683_168317


namespace translated_B_is_B_l1683_168380

def point : Type := ℤ × ℤ

def A : point := (-4, -1)
def A' : point := (-2, 2)
def B : point := (1, 1)
def B' : point := (3, 4)

def translation_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2)

def translate_point (p : point) (v : point) : point :=
  (p.1 + v.1, p.2 + v.2)

theorem translated_B_is_B' : translate_point B (translation_vector A A') = B' :=
by
  sorry

end translated_B_is_B_l1683_168380


namespace mul_exponent_result_l1683_168331

theorem mul_exponent_result : 112 * (5^4) = 70000 := 
by 
  sorry

end mul_exponent_result_l1683_168331


namespace remaining_cubes_count_l1683_168321

-- Define the initial number of cubes
def initial_cubes : ℕ := 64

-- Define the holes in the bottom layer
def holes_in_bottom_layer : ℕ := 6

-- Define the number of cubes removed per hole
def cubes_removed_per_hole : ℕ := 3

-- Define the calculation for missing cubes
def missing_cubes : ℕ := holes_in_bottom_layer * cubes_removed_per_hole

-- Define the calculation for remaining cubes
def remaining_cubes : ℕ := initial_cubes - missing_cubes

-- The theorem to prove
theorem remaining_cubes_count : remaining_cubes = 46 := by
  sorry

end remaining_cubes_count_l1683_168321


namespace problem1_problem2_l1683_168391

-- Problem 1
theorem problem1 : 2 * Real.cos (30 * Real.pi / 180) - Real.tan (60 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) = 1 / 2 :=
by sorry

-- Problem 2
theorem problem2 : (-1) ^ 2023 + 2 * Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) + Real.sin (60 * Real.pi / 180) + (Real.tan (60 * Real.pi / 180)) ^ 2 = 2 + Real.sqrt 2 :=
by sorry

end problem1_problem2_l1683_168391


namespace amount_after_two_years_l1683_168338

def present_value : ℝ := 62000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

theorem amount_after_two_years:
  let amount_after_n_years (pv : ℝ) (r : ℝ) (n : ℕ) := pv * (1 + r)^n
  amount_after_n_years present_value rate_of_increase time_period = 78468.75 := 
  by 
    -- This is where your proof would go
    sorry

end amount_after_two_years_l1683_168338


namespace value_of_C_l1683_168336

theorem value_of_C (C : ℝ) (h : 4 * C + 3 = 25) : C = 5.5 :=
by
  sorry

end value_of_C_l1683_168336


namespace fractions_are_integers_l1683_168313

theorem fractions_are_integers (x y : ℕ) 
    (h : ∃ k : ℤ, (x^2 - 1) / (y + 1) + (y^2 - 1) / (x + 1) = k) :
    ∃ u v : ℤ, (x^2 - 1) = u * (y + 1) ∧ (y^2 - 1) = v * (x + 1) := 
by
  sorry

end fractions_are_integers_l1683_168313


namespace no_nonzero_integer_solution_l1683_168350

theorem no_nonzero_integer_solution 
(a b c n : ℤ) (h : 6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * n ^ 2) : 
a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
sorry

end no_nonzero_integer_solution_l1683_168350


namespace total_floors_combined_l1683_168342

-- Let C be the number of floors in the Chrysler Building
-- Let L be the number of floors in the Leeward Center
-- Given that C = 23 and C = L + 11
-- Prove that the total floors in both buildings combined equals 35

theorem total_floors_combined (C L : ℕ) (h1 : C = 23) (h2 : C = L + 11) : C + L = 35 :=
by
  sorry

end total_floors_combined_l1683_168342


namespace largest_base_5_five_digit_number_in_decimal_l1683_168375

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l1683_168375


namespace manuscript_total_cost_l1683_168340

theorem manuscript_total_cost
  (P R1 R2 R3 : ℕ)
  (RateFirst RateRevision : ℕ)
  (hP : P = 300)
  (hR1 : R1 = 55)
  (hR2 : R2 = 35)
  (hR3 : R3 = 25)
  (hRateFirst : RateFirst = 8)
  (hRateRevision : RateRevision = 6) :
  let RemainingPages := P - (R1 + R2 + R3)
  let CostNoRevisions := RemainingPages * RateFirst
  let CostOneRevision := R1 * (RateFirst + RateRevision)
  let CostTwoRevisions := R2 * (RateFirst + 2 * RateRevision)
  let CostThreeRevisions := R3 * (RateFirst + 3 * RateRevision)
  let TotalCost := CostNoRevisions + CostOneRevision + CostTwoRevisions + CostThreeRevisions
  TotalCost = 3600 :=
by
  sorry

end manuscript_total_cost_l1683_168340


namespace smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l1683_168361

def is_composite (n : ℕ) : Prop := (∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_square_perimeter_of_isosceles_triangle_with_composite_sides :
  ∃ a b : ℕ,
    is_composite a ∧
    is_composite b ∧
    (2 * a + b) ^ 2 = 256 :=
sorry

end smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l1683_168361


namespace optimal_green_tiles_l1683_168315

variable (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ)

def conditions (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ) :=
  n_indigo ≥ n_red + n_orange + n_yellow + n_green + n_blue ∧
  n_blue ≥ n_red + n_orange + n_yellow + n_green ∧
  n_green ≥ n_red + n_orange + n_yellow ∧
  n_yellow ≥ n_red + n_orange ∧
  n_orange ≥ n_red ∧
  n_red + n_orange + n_yellow + n_green + n_blue + n_indigo = 100

theorem optimal_green_tiles : 
  conditions n_red n_orange n_yellow n_green n_blue n_indigo → 
  n_green = 13 := by
    sorry

end optimal_green_tiles_l1683_168315


namespace stairs_left_to_climb_l1683_168371

def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

theorem stairs_left_to_climb : total_stairs - climbed_stairs = 22 := by
  sorry

end stairs_left_to_climb_l1683_168371


namespace distance_between_midpoints_l1683_168394

-- Conditions
def AA' := 68 -- in centimeters
def BB' := 75 -- in centimeters
def CC' := 112 -- in centimeters
def DD' := 133 -- in centimeters

-- Question: Prove the distance between the midpoints of A'C' and B'D' is 14 centimeters
theorem distance_between_midpoints :
  let midpoint_A'C' := (AA' + CC') / 2
  let midpoint_B'D' := (BB' + DD') / 2
  (midpoint_B'D' - midpoint_A'C' = 14) :=
by
  sorry

end distance_between_midpoints_l1683_168394


namespace quadratic_equation_same_solutions_l1683_168346

theorem quadratic_equation_same_solutions :
  ∃ b c : ℝ, (b, c) = (1, -7) ∧ (∀ x : ℝ, (x - 3 = 4 ∨ 3 - x = 4) ↔ (x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_equation_same_solutions_l1683_168346


namespace _l1683_168388

lemma power_of_a_point_theorem (AP BP CP DP : ℝ) (hAP : AP = 5) (hCP : CP = 2) (h_theorem : AP * BP = CP * DP) :
  BP / DP = 2 / 5 :=
by
  sorry

end _l1683_168388


namespace sum_abs_diff_is_18_l1683_168363

noncomputable def sum_of_possible_abs_diff (a b c d : ℝ) : ℝ :=
  let possible_values := [
      abs ((a + 2) - (d - 7)),
      abs ((a + 2) - (d + 1)),
      abs ((a + 2) - (d - 1)),
      abs ((a + 2) - (d + 7)),
      abs ((a - 2) - (d - 7)),
      abs ((a - 2) - (d + 1)),
      abs ((a - 2) - (d - 1)),
      abs ((a - 2) - (d + 7))
  ]
  possible_values.foldl (· + ·) 0

theorem sum_abs_diff_is_18 (a b c d : ℝ) (h1 : abs (a - b) = 2) (h2 : abs (b - c) = 3) (h3 : abs (c - d) = 4) :
  sum_of_possible_abs_diff a b c d = 18 := by
  sorry

end sum_abs_diff_is_18_l1683_168363


namespace avg_speed_correct_l1683_168333

noncomputable def avg_speed_round_trip
  (flight_up_speed : ℝ)
  (tailwind_speed : ℝ)
  (tailwind_angle : ℝ)
  (flight_home_speed : ℝ)
  (headwind_speed : ℝ)
  (headwind_angle : ℝ) : ℝ :=
  let effective_tailwind_speed := tailwind_speed * Real.cos (tailwind_angle * Real.pi / 180)
  let ground_speed_to_mother := flight_up_speed + effective_tailwind_speed
  let effective_headwind_speed := headwind_speed * Real.cos (headwind_angle * Real.pi / 180)
  let ground_speed_back_home := flight_home_speed - effective_headwind_speed
  (ground_speed_to_mother + ground_speed_back_home) / 2

theorem avg_speed_correct :
  avg_speed_round_trip 96 12 30 88 15 60 = 93.446 :=
by
  sorry

end avg_speed_correct_l1683_168333


namespace sum_of_abcd_l1683_168392

theorem sum_of_abcd (a b c d: ℝ) (h₁: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂: c + d = 10 * a) (h₃: c * d = -11 * b) (h₄: a + b = 10 * c) (h₅: a * b = -11 * d)
  : a + b + c + d = 1210 := by
  sorry

end sum_of_abcd_l1683_168392


namespace number_of_spiders_l1683_168370

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) 
  (h1 : total_legs = 40) (h2 : legs_per_spider = 8) : 
  (total_legs / legs_per_spider = 5) :=
by
  -- Placeholder for the actual proof
  sorry

end number_of_spiders_l1683_168370


namespace length_of_bridge_l1683_168368

theorem length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) (length_m : ℝ) :
  speed_kmh = 5 → time_min = 15 → length_m = 1250 :=
by
  sorry

end length_of_bridge_l1683_168368


namespace filtration_minimum_l1683_168357

noncomputable def lg : ℝ → ℝ := sorry

theorem filtration_minimum (x : ℕ) (lg2 : ℝ) (lg3 : ℝ) (h1 : lg2 = 0.3010) (h2 : lg3 = 0.4771) :
  (2 / 3 : ℝ) ^ x ≤ 1 / 20 → x ≥ 8 :=
sorry

end filtration_minimum_l1683_168357


namespace no_real_roots_l1683_168307

-- Define the coefficients of the quadratic equation
def a : ℝ := 1
def b : ℝ := 2
def c : ℝ := 4

-- Define the quadratic equation
def quadratic_eqn (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant : ℝ := b^2 - 4 * a * c

-- State the theorem: The quadratic equation has no real roots because the discriminant is negative
theorem no_real_roots : discriminant < 0 := by
  unfold discriminant
  unfold a b c
  sorry

end no_real_roots_l1683_168307


namespace prove_q_l1683_168311

-- Assume the conditions
variable (p q : Prop)
variable (hpq : p ∨ q) -- "p or q" is true
variable (hnp : ¬p)    -- "not p" is true

-- The theorem to prove q is true
theorem prove_q : q :=
by {
  sorry
}

end prove_q_l1683_168311


namespace mod_11_residue_l1683_168376

theorem mod_11_residue :
  (312 ≡ 4 [MOD 11]) ∧
  (47 ≡ 3 [MOD 11]) ∧
  (154 ≡ 0 [MOD 11]) ∧
  (22 ≡ 0 [MOD 11]) →
  (312 + 6 * 47 + 8 * 154 + 5 * 22 ≡ 0 [MOD 11]) :=
by
  intros h
  sorry

end mod_11_residue_l1683_168376


namespace equivalence_condition_l1683_168320

universe u

variables {U : Type u} (A B : Set U)

theorem equivalence_condition :
  (∃ (C : Set U), A ⊆ C ∧ B ⊆ Cᶜ) ↔ (A ∩ B = ∅) :=
sorry

end equivalence_condition_l1683_168320


namespace part1_part2_l1683_168378

theorem part1 (a x y : ℝ) (h1 : 3 * x - y = 2 * a - 5) (h2 : x + 2 * y = 3 * a + 3)
  (hx : x > 0) (hy : y > 0) : a > 1 :=
sorry

theorem part2 (a b : ℝ) (ha : a > 1) (h3 : a - b = 4) (hb : b < 2) : 
  -2 < a + b ∧ a + b < 8 :=
sorry

end part1_part2_l1683_168378


namespace problem_solution_l1683_168302

theorem problem_solution :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 :=
by sorry

end problem_solution_l1683_168302


namespace unique_intersection_point_l1683_168352

theorem unique_intersection_point (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - 2 * (m + 1) * x - 1 = 0) → x = -1) ↔ m = -2 :=
by
  sorry

end unique_intersection_point_l1683_168352


namespace hyperbola_other_asymptote_l1683_168379

-- Define the problem conditions
def one_asymptote (x y : ℝ) : Prop := y = 2 * x
def foci_x_coordinate : ℝ := -4

-- Define the equation of the other asymptote
def other_asymptote (x y : ℝ) : Prop := y = -2 * x - 16

-- The statement to be proved
theorem hyperbola_other_asymptote : 
  (∀ x y, one_asymptote x y) → (∀ x, x = -4 → ∃ y, ∃ C, other_asymptote x y ∧ y = C + -2 * x - 8) :=
by
  sorry

end hyperbola_other_asymptote_l1683_168379


namespace unique_solution_l1683_168305

noncomputable def func_prop (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (f x)^2 / x - 1 / x)

theorem unique_solution (f : ℝ → ℝ) :
  func_prop f → ∀ x ≥ 1, f x = x + 1 :=
by
  sorry

end unique_solution_l1683_168305


namespace max_value_output_l1683_168389

theorem max_value_output (a b c : ℝ) (h_a : a = 3) (h_b : b = 7) (h_c : c = 2) : max (max a b) c = 7 := 
by
  sorry

end max_value_output_l1683_168389


namespace sales_of_stationery_accessories_l1683_168373

def percentage_of_sales_notebooks : ℝ := 25
def percentage_of_sales_markers : ℝ := 40
def total_sales_percentage : ℝ := 100

theorem sales_of_stationery_accessories : 
  percentage_of_sales_notebooks + percentage_of_sales_markers = 65 → 
  total_sales_percentage - (percentage_of_sales_notebooks + percentage_of_sales_markers) = 35 :=
by
  sorry

end sales_of_stationery_accessories_l1683_168373


namespace simplify_exponent_multiplication_l1683_168354

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end simplify_exponent_multiplication_l1683_168354


namespace arithmetic_sequence_max_sum_l1683_168356

noncomputable def max_S_n (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  n * a 1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum :
  ∃ d, ∃ a : ℕ → ℝ, 
  (a 1 = 1) ∧ (3 * (a 1 + 7 * d) = 5 * (a 1 + 12 * d)) ∧ 
  (∀ n, max_S_n n a d ≤ max_S_n 20 a d) := 
sorry

end arithmetic_sequence_max_sum_l1683_168356


namespace bc_sum_eq_twelve_l1683_168304

theorem bc_sum_eq_twelve (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hb_lt : b < 12) (hc_lt : c < 12) 
  (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : b + c = 12 :=
by
  sorry

end bc_sum_eq_twelve_l1683_168304


namespace geometric_sum_eight_terms_l1683_168366

theorem geometric_sum_eight_terms (a_1 : ℕ) (S_4 : ℕ) (r : ℕ) (S_8 : ℕ) 
    (h1 : r = 2) (h2 : S_4 = a_1 * (1 + r + r^2 + r^3)) (h3 : S_4 = 30) :
    S_8 = a_1 * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) → S_8 = 510 := 
by sorry

end geometric_sum_eight_terms_l1683_168366


namespace find_y_from_triangle_properties_l1683_168347

-- Define angle measures according to the given conditions
def angle_BAC := 45
def angle_CDE := 72

-- Define the proof problem
theorem find_y_from_triangle_properties
: ∀ (y : ℝ), (∃ (BAC ACB ABC ADC ADE AED DEB : ℝ),
    angle_BAC = 45 ∧
    angle_CDE = 72 ∧
    BAC + ACB + ABC = 180 ∧
    ADC = 180 ∧
    ADE = 180 - angle_CDE ∧
    EAD = angle_BAC ∧
    AED + ADE + EAD = 180 ∧
    DEB = 180 - AED ∧
    y = DEB) →
    y = 153 :=
by sorry

end find_y_from_triangle_properties_l1683_168347


namespace radius_increase_by_100_percent_l1683_168309

theorem radius_increase_by_100_percent (A A' r r' : ℝ) (π : ℝ)
  (h1 : A = π * r^2) -- initial area of the circle
  (h2 : A' = 4 * A) -- new area is 4 times the original area
  (h3 : A' = π * r'^2) -- new area formula with new radius
  : r' = 2 * r :=
by
  sorry

end radius_increase_by_100_percent_l1683_168309


namespace unattainable_y_l1683_168386

theorem unattainable_y (x : ℝ) (h1 : x ≠ -3/2) : y = (1 - x) / (2 * x + 3) -> ¬(y = -1 / 2) :=
by sorry

end unattainable_y_l1683_168386


namespace suresh_completion_time_l1683_168374

theorem suresh_completion_time (S : ℕ) 
  (ashu_time : ℕ := 30) 
  (suresh_work_time : ℕ := 9) 
  (ashu_remaining_time : ℕ := 12) 
  (ashu_fraction : ℚ := ashu_remaining_time / ashu_time) :
  (suresh_work_time / S + ashu_fraction = 1) → S = 15 :=
by
  intro h
  -- Proof here
  sorry

end suresh_completion_time_l1683_168374


namespace never_2003_pieces_l1683_168384

theorem never_2003_pieces :
  ¬∃ n : ℕ, (n = 5 + 4 * k) ∧ (n = 2003) :=
by
  sorry

end never_2003_pieces_l1683_168384


namespace condition_necessity_not_sufficiency_l1683_168372

theorem condition_necessity_not_sufficiency (a : ℝ) : 
  (2 / a < 1 → a^2 > 4) ∧ ¬(2 / a < 1 ↔ a^2 > 4) :=
by {
  sorry
}

end condition_necessity_not_sufficiency_l1683_168372


namespace certain_number_divisible_by_9_l1683_168316

theorem certain_number_divisible_by_9 : ∃ N : ℕ, (∀ k : ℕ, (0 ≤ k ∧ k < 1110 → N + 9 * k ≤ 10000 ∧ (N + 9 * k) % 9 = 0)) ∧ N = 27 :=
by
  -- Given conditions:
  -- Numbers are in an arithmetic sequence with common difference 9.
  -- Total count of such numbers is 1110.
  -- The last number ≤ 10000 that is divisible by 9 is 9999.
  let L := 9999
  let n := 1110
  let d := 9
  -- First term in the sequence:
  let a := L - (n - 1) * d
  exists 27
  -- Proof of the conditions would follow here ...
  sorry

end certain_number_divisible_by_9_l1683_168316


namespace weight_difference_l1683_168395

variable (W_A W_D : Nat)

theorem weight_difference : W_A - W_D = 15 :=
by
  -- Given conditions
  have h1 : W_A = 67 := sorry
  have h2 : W_D = 52 := sorry
  -- Proof
  sorry

end weight_difference_l1683_168395


namespace sufficient_but_not_necessary_condition_l1683_168335

def P (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1/x + 4 * x + 6 * m) ≥ 0

def Q (m : ℝ) : Prop :=
  m ≥ -5

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (P m → Q m) ∧ ¬(Q m → P m) := sorry

end sufficient_but_not_necessary_condition_l1683_168335


namespace find_ratio_l1683_168334

-- Definition of the system of equations with k = 5
def system_of_equations (x y z : ℝ) :=
  x + 10 * y + 5 * z = 0 ∧
  2 * x + 5 * y + 4 * z = 0 ∧
  3 * x + 6 * y + 5 * z = 0

-- Proof that if (x, y, z) solves the system, then yz / x^2 = -3 / 49
theorem find_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : system_of_equations x y z) :
  (y * z) / (x ^ 2) = -3 / 49 :=
by
  -- Substitute the system of equations and solve for the ratio.
  sorry

end find_ratio_l1683_168334


namespace find_a_for_inequality_l1683_168359

theorem find_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 3) → -2 * x^2 + a * x + 6 > 0) → a = 2 :=
by
  sorry

end find_a_for_inequality_l1683_168359


namespace solve_for_b_l1683_168308

noncomputable def system_has_solution (b : ℝ) : Prop :=
  ∃ (a : ℝ) (x y : ℝ),
    y = -b - x^2 ∧
    x^2 + y^2 + 8 * a^2 = 4 + 4 * a * (x + y)

theorem solve_for_b (b : ℝ) : system_has_solution b ↔ b ≤ 2 * Real.sqrt 2 + 1 / 4 := 
by 
  sorry

end solve_for_b_l1683_168308


namespace prime_solution_l1683_168369

theorem prime_solution (p : ℕ) (x y : ℕ) (hp : Prime p) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 → p = 2 ∨ p = 3 :=
by
  sorry

end prime_solution_l1683_168369


namespace sin_value_given_cos_condition_l1683_168345

theorem sin_value_given_cos_condition (theta : ℝ) (h : Real.cos (5 * Real.pi / 12 - theta) = 1 / 3) :
  Real.sin (Real.pi / 12 + theta) = 1 / 3 :=
sorry

end sin_value_given_cos_condition_l1683_168345
