import Mathlib

namespace james_age_when_john_turned_35_l123_123944

theorem james_age_when_john_turned_35 :
  ∀ (J : ℕ) (Tim : ℕ) (John : ℕ),
  (Tim = 5) →
  (Tim + 5 = 2 * John) →
  (Tim = 79) →
  (John = 35) →
  (J = John) →
  J = 35 :=
by
  intros J Tim John h1 h2 h3 h4 h5
  rw [h4] at h5
  exact h5

end james_age_when_john_turned_35_l123_123944


namespace no_two_exact_cubes_between_squares_l123_123486

theorem no_two_exact_cubes_between_squares :
  ∀ (n a b : ℤ), ¬ (n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2) :=
by
  intros n a b
  sorry

end no_two_exact_cubes_between_squares_l123_123486


namespace no_cubic_term_l123_123298

noncomputable def p1 (a b k : ℝ) : ℝ := -2 * a * b + (1 / 3) * k * a^2 * b + 5 * b^2
noncomputable def p2 (a b : ℝ) : ℝ := b^2 + 3 * a^2 * b - 5 * a * b + 1
noncomputable def diff (a b k : ℝ) : ℝ := p1 a b k - p2 a b
noncomputable def cubic_term_coeff (a b k : ℝ) : ℝ := (1 / 3) * k - 3

theorem no_cubic_term (a b : ℝ) : ∀ k, (cubic_term_coeff a b k = 0) → k = 9 :=
by
  intro k h
  sorry

end no_cubic_term_l123_123298


namespace sum_squares_nonpositive_l123_123622

theorem sum_squares_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ac ≤ 0 :=
by {
  sorry
}

end sum_squares_nonpositive_l123_123622


namespace percent_of_motorists_receive_speeding_tickets_l123_123076

theorem percent_of_motorists_receive_speeding_tickets
    (p_exceed : ℝ)
    (p_no_ticket : ℝ)
    (h1 : p_exceed = 0.125)
    (h2 : p_no_ticket = 0.20) : 
    (0.8 * p_exceed) * 100 = 10 :=
by
  sorry

end percent_of_motorists_receive_speeding_tickets_l123_123076


namespace probability_red_or_black_probability_red_black_or_white_l123_123842

theorem probability_red_or_black (total_balls red_balls black_balls : ℕ) : 
  total_balls = 12 → red_balls = 5 → black_balls = 4 → 
  (red_balls + black_balls) / total_balls = 3 / 4 :=
by
  intros
  sorry

theorem probability_red_black_or_white (total_balls red_balls black_balls white_balls : ℕ) :
  total_balls = 12 → red_balls = 5 → black_balls = 4 → white_balls = 2 → 
  (red_balls + black_balls + white_balls) / total_balls = 11 / 12 :=
by
  intros
  sorry

end probability_red_or_black_probability_red_black_or_white_l123_123842


namespace no_solution_of_fractional_equation_l123_123470

theorem no_solution_of_fractional_equation (x : ℝ) : ¬ (x - 8) / (x - 7) - 8 = 1 / (7 - x) := 
sorry

end no_solution_of_fractional_equation_l123_123470


namespace min_radius_for_area_l123_123308

theorem min_radius_for_area (A : ℝ) (hA : A = 500) : ∃ r : ℝ, r = 13 ∧ π * r^2 ≥ A :=
by
  sorry

end min_radius_for_area_l123_123308


namespace isosceles_triangle_angle_sum_l123_123504

theorem isosceles_triangle_angle_sum 
  (A B C : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C]
  (AC AB : ℝ) 
  (angle_ABC : ℝ)
  (isosceles : AC = AB)
  (angle_A : angle_ABC = 70) :
  (∃ angle_B : ℝ, angle_B = 55) :=
by
  sorry

end isosceles_triangle_angle_sum_l123_123504


namespace average_monthly_balance_is_150_l123_123701

-- Define the balances for each month
def balance_jan : ℕ := 100
def balance_feb : ℕ := 200
def balance_mar : ℕ := 150
def balance_apr : ℕ := 150

-- Define the number of months
def num_months : ℕ := 4

-- Define the total sum of balances
def total_balance : ℕ := balance_jan + balance_feb + balance_mar + balance_apr

-- Define the average balance
def average_balance : ℕ := total_balance / num_months

-- Goal is to prove that the average monthly balance is 150 dollars
theorem average_monthly_balance_is_150 : average_balance = 150 :=
by
  sorry

end average_monthly_balance_is_150_l123_123701


namespace negation_of_forall_implies_exists_l123_123674

theorem negation_of_forall_implies_exists :
  (¬ ∀ x : ℝ, x^2 > 1) = (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_forall_implies_exists_l123_123674


namespace original_radius_l123_123046

theorem original_radius (r : Real) (h : Real) (z : Real) 
  (V : Real) (Vh : Real) (Vr : Real) :
  h = 3 → 
  V = π * r^2 * h → 
  Vh = π * r^2 * (h + 3) → 
  Vr = π * (r + 3)^2 * h → 
  Vh - V = z → 
  Vr - V = z →
  r = 3 + 3 * Real.sqrt 2 :=
by
  sorry

end original_radius_l123_123046


namespace coeff_x5_term_l123_123246

-- We define the binomial coefficient function C(n, k)
def C (n k : ℕ) : ℕ := Nat.choose n k

-- We define the expression in question
noncomputable def expr (x : ℝ) : ℝ := (1/x + 2*x)^7

-- The coefficient of x^5 term in the expansion
theorem coeff_x5_term : 
  let general_term (r : ℕ) (x : ℝ) := (2:ℝ)^r * C 7 r * x^(2 * r - 7)
  -- r is chosen such that the power of x is 5
  let r := 6
  -- The coefficient for r=6
  general_term r 1 = 448 := 
by sorry

end coeff_x5_term_l123_123246


namespace number_of_people_l123_123112

-- Definitions based on conditions
def per_person_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 100 else max 72 (100 - 2 * (x - 30))

def total_cost (x : ℕ) : ℕ :=
  x * per_person_cost x

-- Main theorem statement
theorem number_of_people (x : ℕ) (h1 : total_cost x = 3150) (h2 : x > 30) : x = 35 :=
by {
  sorry
}

end number_of_people_l123_123112


namespace part1_part2_l123_123296

open Complex

noncomputable def z0 : ℂ := 3 + 4 * Complex.I

theorem part1 (z1 : ℂ) (h : z1 * z0 = 3 * z1 + z0) : z1.im = -3/4 := by
  sorry

theorem part2 (x : ℝ) 
    (z : ℂ := (x^2 - 4 * x) + (x + 2) * Complex.I) 
    (z0_conj : ℂ := 3 - 4 * Complex.I) 
    (h : (z + z0_conj).re < 0 ∧ (z + z0_conj).im > 0) : 
    2 < x ∧ x < 3 :=
  by 
  sorry

end part1_part2_l123_123296


namespace rectangle_width_l123_123651

/-- Given the conditions:
    - length of a rectangle is 5.4 cm
    - area of the rectangle is 48.6 cm²
    Prove that the width of the rectangle is 9 cm.
-/
theorem rectangle_width (length width area : ℝ) 
  (h_length : length = 5.4) 
  (h_area : area = 48.6) 
  (h_area_eq : area = length * width) : 
  width = 9 := 
by
  sorry

end rectangle_width_l123_123651


namespace fifth_boat_more_than_average_l123_123259

theorem fifth_boat_more_than_average :
  let total_people := 2 + 4 + 3 + 5 + 6
  let num_boats := 5
  let average_people := total_people / num_boats
  let fifth_boat := 6
  (fifth_boat - average_people) = 2 :=
by
  sorry

end fifth_boat_more_than_average_l123_123259


namespace vlecks_in_straight_angle_l123_123687

theorem vlecks_in_straight_angle (V : Type) [LinearOrderedField V] (full_circle_vlecks : V) (h1 : full_circle_vlecks = 600) :
  (full_circle_vlecks / 2) = 300 :=
by
  sorry

end vlecks_in_straight_angle_l123_123687


namespace meaningful_fraction_range_l123_123659

theorem meaningful_fraction_range (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) := sorry

end meaningful_fraction_range_l123_123659


namespace largest_integer_mod_l123_123855

theorem largest_integer_mod (a : ℕ) (h₁ : a < 100) (h₂ : a % 5 = 2) : a = 97 :=
by sorry

end largest_integer_mod_l123_123855


namespace find_xyz_area_proof_l123_123396

-- Conditions given in the problem
variable (x y z : ℝ)
-- Side lengths derived from condition of inscribed circle
def conditions :=
  (x + y = 5) ∧
  (x + z = 6) ∧
  (y + z = 8)

-- The proof problem: Show the relationships between x, y, and z given the side lengths
theorem find_xyz_area_proof (h : conditions x y z) :
  (z - y = 1) ∧ (z - x = 3) ∧ (z = 4.5) ∧ (x = 1.5) ∧ (y = 3.5) :=
by
  sorry

end find_xyz_area_proof_l123_123396


namespace number_of_paths_grid_l123_123775

def paths_from_A_to_C (h v : Nat) : Nat :=
  Nat.choose (h + v) v

#eval paths_from_A_to_C 7 6 -- expected result: 1716

theorem number_of_paths_grid :
  paths_from_A_to_C 7 6 = 1716 := by
  sorry

end number_of_paths_grid_l123_123775


namespace baseball_fans_count_l123_123414

theorem baseball_fans_count
  (Y M R : ℕ) 
  (h1 : Y = (3 * M) / 2)
  (h2 : R = (5 * M) / 4)
  (hM : M = 104) :
  Y + M + R = 390 :=
by
  sorry 

end baseball_fans_count_l123_123414


namespace find_A_l123_123009

theorem find_A (A B : ℕ) (h1 : 10 * A + 7 + (30 + B) = 73) : A = 3 := by
  sorry

end find_A_l123_123009


namespace gary_profit_l123_123236

theorem gary_profit :
  let total_flour := 8 -- pounds
  let cost_flour := 4 -- dollars
  let large_cakes_flour := 5 -- pounds
  let small_cakes_flour := 3 -- pounds
  let flour_per_large_cake := 0.75 -- pounds per large cake
  let flour_per_small_cake := 0.25 -- pounds per small cake
  let cost_additional_large := 1.5 -- dollars per large cake
  let cost_additional_small := 0.75 -- dollars per small cake
  let cost_baking_equipment := 10 -- dollars
  let revenue_per_large := 6.5 -- dollars per large cake
  let revenue_per_small := 2.5 -- dollars per small cake
  let num_large_cakes := 6 -- (from calculation: ⌊5 / 0.75⌋)
  let num_small_cakes := 12 -- (from calculation: 3 / 0.25)
  let cost_additional_ingredients := num_large_cakes * cost_additional_large + num_small_cakes * cost_additional_small
  let total_revenue := num_large_cakes * revenue_per_large + num_small_cakes * revenue_per_small
  let total_cost := cost_flour + cost_baking_equipment + cost_additional_ingredients
  let profit := total_revenue - total_cost
  profit = 37 := by
  sorry

end gary_profit_l123_123236


namespace cola_cost_l123_123062

theorem cola_cost (h c : ℝ) (h1 : 3 * h + 2 * c = 360) (h2 : 2 * h + 3 * c = 390) : c = 90 :=
by
  sorry

end cola_cost_l123_123062


namespace vacation_cost_proof_l123_123207

noncomputable def vacation_cost (C : ℝ) :=
  C / 5 - C / 8 = 120

theorem vacation_cost_proof {C : ℝ} (h : vacation_cost C) : C = 1600 :=
by
  sorry

end vacation_cost_proof_l123_123207


namespace wire_weight_l123_123342

theorem wire_weight (w : ℕ → ℕ) (h_proportional : ∀ (x y : ℕ), w (x + y) = w x + w y) : 
  (w 25 = 5) → w 75 = 15 :=
by
  intro h1
  sorry

end wire_weight_l123_123342


namespace correct_sum_of_integers_l123_123619

theorem correct_sum_of_integers :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a - b = 3 ∧ a * b = 63 ∧ a + b = 17 :=
by 
  sorry

end correct_sum_of_integers_l123_123619


namespace find_initial_cookies_l123_123108

-- Definitions based on problem conditions
def initial_cookies (x : ℕ) : Prop :=
  let after_eating := x - 2
  let after_buying := after_eating + 37
  after_buying = 75

-- Main statement to be proved
theorem find_initial_cookies : ∃ x, initial_cookies x ∧ x = 40 :=
by
  sorry

end find_initial_cookies_l123_123108


namespace expand_product_l123_123730

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  sorry

end expand_product_l123_123730


namespace gcd_values_count_l123_123312

noncomputable def count_gcd_values (a b : ℕ) : ℕ :=
  if (a * b = 720 ∧ a + b = 50) then 1 else 0

theorem gcd_values_count : 
  (∃ a b : ℕ, a * b = 720 ∧ a + b = 50) → count_gcd_values a b = 1 :=
by
  sorry

end gcd_values_count_l123_123312


namespace popsicle_sticks_left_l123_123047

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l123_123047


namespace race_times_l123_123456

theorem race_times (x y : ℕ) (h1 : 5 * x + 1 = 4 * y) (h2 : 5 * y - 8 = 4 * x) :
  5 * x = 15 ∧ 5 * y = 20 :=
by
  sorry

end race_times_l123_123456


namespace fraction_under_11_is_one_third_l123_123230

def fraction_under_11 (T : ℕ) (fraction_above_11_under_13 : ℚ) (students_above_13 : ℕ) : ℚ :=
  let fraction_under_11 := 1 - (fraction_above_11_under_13 + students_above_13 / T)
  fraction_under_11

theorem fraction_under_11_is_one_third :
  fraction_under_11 45 (2/5) 12 = 1/3 :=
by
  sorry

end fraction_under_11_is_one_third_l123_123230


namespace generated_surface_l123_123777

theorem generated_surface (L : ℝ → ℝ → ℝ → Prop)
  (H1 : ∀ x y z, L x y z → y = z) 
  (H2 : ∀ t, L (t^2 / 2) t 0) 
  (H3 : ∀ s, L (s^2 / 3) 0 s) : 
  ∀ y z, ∃ x, L x y z → x = (y - z) * (y / 2 - z / 3) :=
by
  sorry

end generated_surface_l123_123777


namespace lord_moneybag_l123_123460

theorem lord_moneybag (n : ℕ) (hlow : 300 ≤ n) (hhigh : n ≤ 500)
           (h6 : 6 ∣ n) (h5 : 5 ∣ (n - 1)) (h4 : 4 ∣ (n - 2)) 
           (h3 : 3 ∣ (n - 3)) (h2 : 2 ∣ (n - 4)) (hprime : Nat.Prime (n - 5)) :
  n = 426 := by
  sorry

end lord_moneybag_l123_123460


namespace more_campers_afternoon_than_morning_l123_123518

def campers_morning : ℕ := 52
def campers_afternoon : ℕ := 61

theorem more_campers_afternoon_than_morning : campers_afternoon - campers_morning = 9 :=
by
  -- proof goes here
  sorry

end more_campers_afternoon_than_morning_l123_123518


namespace no_common_root_l123_123578

theorem no_common_root (a b c d : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 ∧ x^2 + a * x + d = 0 :=
by
  sorry

end no_common_root_l123_123578


namespace number_of_roof_tiles_l123_123011

def land_cost : ℝ := 50
def bricks_cost_per_1000 : ℝ := 100
def roof_tile_cost : ℝ := 10
def land_required : ℝ := 2000
def bricks_required : ℝ := 10000
def total_construction_cost : ℝ := 106000

theorem number_of_roof_tiles :
  let land_total := land_cost * land_required
  let bricks_total := (bricks_required / 1000) * bricks_cost_per_1000
  let remaining_cost := total_construction_cost - (land_total + bricks_total)
  let roof_tiles := remaining_cost / roof_tile_cost
  roof_tiles = 500 := by
  sorry

end number_of_roof_tiles_l123_123011


namespace diagonal_of_rectangular_solid_l123_123195

-- Define the lengths of the edges
def a : ℝ := 2
def b : ℝ := 3
def c : ℝ := 4

-- Prove that the diagonal of the rectangular solid with edges a, b, and c is sqrt(29)
theorem diagonal_of_rectangular_solid (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : (a^2 + b^2 + c^2) = 29 := 
by 
  rw [h1, h2, h3]
  norm_num

end diagonal_of_rectangular_solid_l123_123195


namespace smallest_x_l123_123127

theorem smallest_x (x y : ℕ) (h_pos: x > 0 ∧ y > 0) (h_eq: 8 / 10 = y / (186 + x)) : x = 4 :=
sorry

end smallest_x_l123_123127


namespace constant_term_binomial_expansion_l123_123100

theorem constant_term_binomial_expansion : ∃ T, (∀ x : ℝ, T = (2 * x - 1 / (2 * x)) ^ 6) ∧ T = -20 := 
by
  sorry

end constant_term_binomial_expansion_l123_123100


namespace initial_investment_l123_123193

theorem initial_investment (b : ℝ) (t_b : ℝ) (t_a : ℝ) (ratio_profit : ℝ) (x : ℝ) :
  b = 36000 → t_b = 4.5 → t_a = 12 → ratio_profit = 2 →
  (x * t_a) / (b * t_b) = ratio_profit → x = 27000 := 
by
  intros hb ht_b ht_a hr hp
  rw [hb, ht_b, ht_a, hr] at hp
  sorry

end initial_investment_l123_123193


namespace parabola_focus_l123_123423

theorem parabola_focus (p : ℝ) (hp : 0 < p) (h : ∀ y x : ℝ, y^2 = 2 * p * x → (x = 2 ∧ y = 0)) : p = 4 :=
sorry

end parabola_focus_l123_123423


namespace oranges_per_box_l123_123724

theorem oranges_per_box (total_oranges : ℝ) (total_boxes : ℝ) (h1 : total_oranges = 26500) (h2 : total_boxes = 2650) : 
  total_oranges / total_boxes = 10 :=
by 
  sorry

end oranges_per_box_l123_123724


namespace angle_CAB_EQ_angle_EAD_l123_123905

variable {A B C D E : Type}

-- Define the angles as variables for the pentagon ABCDE
variable (ABC ADE CEA BDA CAB EAD : ℝ)

-- Given conditions
axiom angle_ABC_EQ_angle_ADE : ABC = ADE
axiom angle_CEA_EQ_angle_BDA : CEA = BDA

-- Prove that angle CAB equals angle EAD
theorem angle_CAB_EQ_angle_EAD : CAB = EAD :=
by
  sorry

end angle_CAB_EQ_angle_EAD_l123_123905


namespace neg_p_l123_123441

theorem neg_p (p : ∀ x : ℝ, x^2 ≥ 0) : ∃ x : ℝ, x^2 < 0 := 
sorry

end neg_p_l123_123441


namespace weight_of_replaced_oarsman_l123_123526

noncomputable def average_weight (W : ℝ) : ℝ := W / 20

theorem weight_of_replaced_oarsman (W : ℝ) (W_avg : ℝ) (H1 : average_weight W = W_avg) (H2 : average_weight (W + 40) = W_avg + 2) : W = 40 :=
by sorry

end weight_of_replaced_oarsman_l123_123526


namespace avg_length_remaining_wires_l123_123130

theorem avg_length_remaining_wires (N : ℕ) (avg_length : ℕ) 
    (third_wires_count : ℕ) (third_wires_avg_length : ℕ) 
    (total_length : ℕ := N * avg_length) 
    (third_wires_total_length : ℕ := third_wires_count * third_wires_avg_length) 
    (remaining_wires_count : ℕ := N - third_wires_count) 
    (remaining_wires_total_length : ℕ := total_length - third_wires_total_length) :
    N = 6 → 
    avg_length = 80 → 
    third_wires_count = 2 → 
    third_wires_avg_length = 70 → 
    remaining_wires_count = 4 → 
    remaining_wires_total_length / remaining_wires_count = 85 :=
by 
  intros hN hAvg hThirdCount hThirdAvg hRemainingCount
  sorry

end avg_length_remaining_wires_l123_123130


namespace area_of_regular_inscribed_polygon_f3_properties_of_f_l123_123369

noncomputable def f (n : ℕ) : ℝ :=
  if h : n ≥ 3 then (n / 2) * Real.sin (2 * Real.pi / n) else 0

theorem area_of_regular_inscribed_polygon_f3 :
  f 3 = (3 * Real.sqrt 3) / 4 :=
by
  sorry

theorem properties_of_f (n : ℕ) (hn : n ≥ 3) :
  (f n = (n / 2) * Real.sin (2 * Real.pi / n)) ∧
  (f n < f (n + 1)) ∧ 
  (f n < f (2 * n) ∧ f (2 * n) ≤ 2 * f n) :=
by
  sorry

end area_of_regular_inscribed_polygon_f3_properties_of_f_l123_123369


namespace find_d_l123_123539

theorem find_d (d : ℝ) (h1 : 0 < d) (h2 : d < 90) (h3 : Real.cos 16 = Real.sin 14 + Real.sin d) : d = 46 :=
by
  sorry

end find_d_l123_123539


namespace total_tea_consumption_l123_123096

variables (S O P : ℝ)

theorem total_tea_consumption : 
  S + O = 11 →
  P + O = 15 →
  P + S = 13 →
  S + O + P = 19.5 :=
by
  intros h1 h2 h3
  sorry

end total_tea_consumption_l123_123096


namespace complex_power_identity_l123_123633

theorem complex_power_identity (w : ℂ) (h : w + w⁻¹ = 2) : w^(2022 : ℕ) + (w⁻¹)^(2022 : ℕ) = 2 := by
  sorry

end complex_power_identity_l123_123633


namespace profit_difference_l123_123092

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end profit_difference_l123_123092


namespace binomial_product_l123_123911

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l123_123911


namespace natalia_crates_l123_123352

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l123_123352


namespace scientific_notation_of_138000_l123_123980

noncomputable def scientific_notation_equivalent (n : ℕ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * (10:ℝ)^exp

theorem scientific_notation_of_138000 : scientific_notation_equivalent 138000 1.38 5 :=
by
  sorry

end scientific_notation_of_138000_l123_123980


namespace Liu_Wei_parts_per_day_l123_123664

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end Liu_Wei_parts_per_day_l123_123664


namespace determine_digits_from_expression_l123_123942

theorem determine_digits_from_expression (a b c x y z S : ℕ) 
  (hx : x = 100) (hy : y = 10) (hz : z = 1)
  (hS : S = a * x + b * y + c * z) :
  S = 100 * a + 10 * b + c :=
by
  -- Variables
  -- a, b, c : ℕ -- digits to find
  -- x, y, z : ℕ -- chosen numbers
  -- S : ℕ -- the given sum

  -- Assumptions
  -- hx : x = 100
  -- hy : y = 10
  -- hz : z = 1
  -- hS : S = a * x + b * y + c * z
  sorry

end determine_digits_from_expression_l123_123942


namespace age_of_son_l123_123725

theorem age_of_son (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 20 := 
sorry

end age_of_son_l123_123725


namespace max_truck_speed_l123_123264

theorem max_truck_speed (D : ℝ) (C : ℝ) (F : ℝ) (L : ℝ → ℝ) (T : ℝ) (x : ℝ) : 
  D = 125 ∧ C = 30 ∧ F = 1000 ∧ (∀ s, L s = 2 * s) ∧ (∃ s, D / s * C + F + L s ≤ T) → x ≤ 75 :=
by
  sorry

end max_truck_speed_l123_123264


namespace books_in_june_l123_123171

-- Definitions
def Book_may : ℕ := 2
def Book_july : ℕ := 10
def Total_books : ℕ := 18

-- Theorem statement
theorem books_in_june : ∃ (Book_june : ℕ), Book_may + Book_june + Book_july = Total_books ∧ Book_june = 6 :=
by
  -- Proof will be here
  sorry

end books_in_june_l123_123171


namespace vertical_angles_equal_l123_123849

-- Define what it means for two angles to be vertical angles.
def are_vertical_angles (α β : ℝ) : Prop :=
  ∃ (γ δ : ℝ), α + γ = 180 ∧ β + δ = 180 ∧ γ = β ∧ δ = α

-- The theorem statement:
theorem vertical_angles_equal (α β : ℝ) : are_vertical_angles α β → α = β := 
  sorry

end vertical_angles_equal_l123_123849


namespace number_of_pens_each_student_gets_l123_123621

theorem number_of_pens_each_student_gets 
    (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ)
    (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) :
  (total_pens / Nat.gcd total_pens total_pencils) = 11 :=
by
  sorry

end number_of_pens_each_student_gets_l123_123621


namespace tan_domain_l123_123188

theorem tan_domain (x : ℝ) : 
  (∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) ↔ 
  ¬(∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) :=
sorry

end tan_domain_l123_123188


namespace jugglers_balls_needed_l123_123851

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end jugglers_balls_needed_l123_123851


namespace fraction_irreducible_l123_123036
-- Import necessary libraries

-- Define the problem to prove
theorem fraction_irreducible (n: ℕ) (h: n > 0) : gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end fraction_irreducible_l123_123036


namespace polygon_sides_l123_123081

theorem polygon_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 180 - 360 / n = 150) : n = 12 := by
  sorry

end polygon_sides_l123_123081


namespace find_f_one_third_l123_123821

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def satisfies_condition (f : ℝ → ℝ) : Prop :=
∀ x, f (2 - x) = f x

noncomputable def f (x : ℝ) : ℝ := if (2 ≤ x ∧ x ≤ 3) then Real.log (x - 1) / Real.log 2 else 0

theorem find_f_one_third (h_odd : is_odd_function f) (h_condition : satisfies_condition f) :
  f (1 / 3) = Real.log 3 / Real.log 2 - 2 :=
by
  sorry

end find_f_one_third_l123_123821


namespace chess_grandmaster_time_l123_123794

theorem chess_grandmaster_time :
  let time_to_learn_rules : ℕ := 2
  let factor_to_get_proficient : ℕ := 49
  let factor_to_become_master : ℕ := 100
  let time_to_get_proficient := factor_to_get_proficient * time_to_learn_rules
  let combined_time := time_to_learn_rules + time_to_get_proficient
  let time_to_become_master := factor_to_become_master * combined_time
  let total_time := time_to_learn_rules + time_to_get_proficient + time_to_become_master
  total_time = 10100 :=
by
  sorry

end chess_grandmaster_time_l123_123794


namespace max_value_expression_l123_123391

theorem max_value_expression  
    (x y : ℝ) 
    (h : 2 * x^2 + y^2 = 6 * x) : 
    x^2 + y^2 + 2 * x ≤ 15 :=
sorry

end max_value_expression_l123_123391


namespace window_treatments_cost_l123_123922

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end window_treatments_cost_l123_123922


namespace each_regular_tire_distance_used_l123_123399

-- Define the conditions of the problem
def total_distance_traveled : ℕ := 50000
def spare_tire_distance : ℕ := 2000
def regular_tires_count : ℕ := 4

-- Using these conditions, we will state the problem as a theorem
theorem each_regular_tire_distance_used : 
  (total_distance_traveled - spare_tire_distance) / regular_tires_count = 12000 :=
by
  sorry

end each_regular_tire_distance_used_l123_123399


namespace question1_question2_l123_123800

-- Definitions based on the conditions
def f (x m : ℝ) : ℝ := x^2 + 4*x + m

theorem question1 (m : ℝ) (h1 : m ≠ 0) (h2 : 16 - 4 * m > 0) : m < 4 :=
  sorry

theorem question2 (m : ℝ) (hx : ∀ x : ℝ, f x m = 0 → f (-x - 4) m = 0) 
  (h_circ : ∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1) ∨ (x = -4 ∧ y = 1)) :
  (∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1)) ∨ (∀ (x y : ℝ), (x = -4 ∧ y = 1)) :=
  sorry

end question1_question2_l123_123800


namespace sum_first_2500_terms_eq_zero_l123_123693

theorem sum_first_2500_terms_eq_zero
  (b : ℕ → ℤ)
  (h1 : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2))
  (h2 : (Finset.range 1800).sum b = 2023)
  (h3 : (Finset.range 2023).sum b = 1800) :
  (Finset.range 2500).sum b = 0 :=
sorry

end sum_first_2500_terms_eq_zero_l123_123693


namespace red_candies_count_l123_123871

def total_candies : ℕ := 3409
def blue_candies : ℕ := 3264

theorem red_candies_count : total_candies - blue_candies = 145 := by
  sorry

end red_candies_count_l123_123871


namespace find_f0_f1_l123_123442

noncomputable def f : ℤ → ℤ := sorry

theorem find_f0_f1 :
  (∀ x : ℤ, f (x+5) - f x = 10 * x + 25) →
  (∀ x : ℤ, f (x^3 - 1) = (f x - x)^3 + x^3 - 3) →
  f 0 = -1 ∧ f 1 = 0 := by
  intros h1 h2
  sorry

end find_f0_f1_l123_123442


namespace prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l123_123191

-- Definitions for the conditions
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Part (I): Without Replacement
theorem prob_at_least_one_black_without_replacement : 
  (20 - 4) / 20 = 4 / 5 :=
by sorry

-- Part (II): With Replacement
theorem prob_exactly_one_black_with_replacement : 
  (3 * 2 * 4 * 4) / (6 * 6 * 6) = 4 / 9 :=
by sorry

end prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l123_123191


namespace probability_calculation_correct_l123_123137

def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 20
def yellow_balls : ℕ := 10
def red_balls : ℕ := 17
def purple_balls : ℕ := 3

def number_of_non_red_or_purple_balls : ℕ := total_balls - (red_balls + purple_balls)

def probability_of_non_red_or_purple : ℚ := number_of_non_red_or_purple_balls / total_balls

theorem probability_calculation_correct :
  probability_of_non_red_or_purple = 0.8 := 
  by 
    -- proof goes here
    sorry

end probability_calculation_correct_l123_123137


namespace percentage_of_the_stock_l123_123128

noncomputable def faceValue : ℝ := 100
noncomputable def yield : ℝ := 0.10
noncomputable def quotedPrice : ℝ := 160

theorem percentage_of_the_stock : 
  (yield * faceValue / quotedPrice * 100 = 6.25) :=
by
  sorry

end percentage_of_the_stock_l123_123128


namespace each_person_gets_4_roses_l123_123307

def ricky_roses_total : Nat := 40
def roses_stolen : Nat := 4
def people : Nat := 9
def remaining_roses : Nat := ricky_roses_total - roses_stolen
def roses_per_person : Nat := remaining_roses / people

theorem each_person_gets_4_roses : roses_per_person = 4 := by
  sorry

end each_person_gets_4_roses_l123_123307


namespace incorrect_statement_B_l123_123625

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x - 1)^3 - a * x - b + 2

-- Condition for statement B
axiom eqn_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1

-- The theorem to prove:
theorem incorrect_statement_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1 := by
  exact eqn_B a b

end incorrect_statement_B_l123_123625


namespace ellipse_eccentricity_l123_123217

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

def ellipse_conditions (F1 B : ℝ × ℝ) (c b : ℝ) : Prop :=
  F1 = (-2, 0) ∧ B = (0, 1) ∧ c = 2 ∧ b = 1

theorem ellipse_eccentricity (F1 B : ℝ × ℝ) (c b a : ℝ)
  (h : ellipse_conditions F1 B c b) :
  eccentricity c a = 2 * Real.sqrt 5 / 5 := by
sorry

end ellipse_eccentricity_l123_123217


namespace percent_non_filler_l123_123078

def burger_weight : ℕ := 120
def filler_weight : ℕ := 30

theorem percent_non_filler : 
  let total_weight := burger_weight
  let filler := filler_weight
  let non_filler := total_weight - filler
  (non_filler / total_weight : ℚ) * 100 = 75 := by
  sorry

end percent_non_filler_l123_123078


namespace find_m_l123_123194

variable {S : ℕ → ℤ}
variable {m : ℕ}

/-- Given the sequences conditions, the value of m is 5 --/
theorem find_m (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) (h4 : 2 ≤ m) : m = 5 :=
sorry

end find_m_l123_123194


namespace point_D_not_in_region_l123_123063

-- Define the condition that checks if a point is not in the region defined by 3x + 2y < 6
def point_not_in_region (x y : ℝ) : Prop :=
  ¬ (3 * x + 2 * y < 6)

-- Define the points
def A := (0, 0)
def B := (1, 1)
def C := (0, 2)
def D := (2, 0)

-- The proof problem as a Lean statement
theorem point_D_not_in_region : point_not_in_region (2:ℝ) (0:ℝ) :=
by
  show point_not_in_region 2 0
  sorry

end point_D_not_in_region_l123_123063


namespace inconsistent_coordinates_l123_123347

theorem inconsistent_coordinates
  (m n : ℝ) 
  (h1 : m - (5/2)*n + 1 = 0) 
  (h2 : (m + 1/2) - (5/2)*(n + 1) + 1 = 0) :
  false :=
by
  sorry

end inconsistent_coordinates_l123_123347


namespace cauliflower_production_diff_l123_123496

theorem cauliflower_production_diff
  (area_this_year : ℕ)
  (area_last_year : ℕ)
  (side_this_year : ℕ)
  (side_last_year : ℕ)
  (H1 : side_this_year * side_this_year = area_this_year)
  (H2 : side_last_year * side_last_year = area_last_year)
  (H3 : side_this_year = side_last_year + 1)
  (H4 : area_this_year = 12544) :
  area_this_year - area_last_year = 223 :=
by
  sorry

end cauliflower_production_diff_l123_123496


namespace dealership_sedan_sales_l123_123538

-- Definitions based on conditions:
def sports_cars_ratio : ℕ := 3
def sedans_ratio : ℕ := 5
def anticipated_sports_cars : ℕ := 36

-- Proof problem statement
theorem dealership_sedan_sales :
    (anticipated_sports_cars * sedans_ratio) / sports_cars_ratio = 60 :=
by
  -- Proof goes here
  sorry

end dealership_sedan_sales_l123_123538


namespace condition_relation_l123_123448

variable (A B C : Prop)

theorem condition_relation (h1 : C → B) (h2 : A → B) : 
  (¬(A → C) ∧ ¬(C → A)) :=
by 
  sorry

end condition_relation_l123_123448


namespace smallest_number_to_add_l123_123315

theorem smallest_number_to_add:
  ∃ x : ℕ, x = 119 ∧ (2714 + x) % 169 = 0 :=
by
  sorry

end smallest_number_to_add_l123_123315


namespace solution_to_power_tower_l123_123696

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solution_to_power_tower : ∃ x : ℝ, infinite_power_tower x = 4 ∧ x = Real.sqrt 2 := sorry

end solution_to_power_tower_l123_123696


namespace sum_of_numbers_l123_123314

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := sorry

end sum_of_numbers_l123_123314


namespace equation_of_parallel_plane_l123_123960

theorem equation_of_parallel_plane {A B C D : ℤ} (hA : A = 3) (hB : B = -2) (hC : C = 4) (hD : D = -16)
    (point : ℝ × ℝ × ℝ) (pass_through : point = (2, -3, 1)) (parallel_plane : A * 2 + B * (-3) + C * 1 + D = 0)
    (gcd_condition : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) :
    A * 2 + B * (-3) + C + D = 0 ∧ A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by
  sorry

end equation_of_parallel_plane_l123_123960


namespace distance_CD_l123_123932

theorem distance_CD (d_north: ℝ) (d_east: ℝ) (d_south: ℝ) (d_west: ℝ) (distance_CD: ℝ) :
  d_north = 30 ∧ d_east = 80 ∧ d_south = 20 ∧ d_west = 30 → distance_CD = 50 :=
by
  intros h
  sorry

end distance_CD_l123_123932


namespace total_bill_l123_123190

variable (B : ℝ)
variable (h1 : 9 * (B / 10 + 3) = B)

theorem total_bill : B = 270 :=
by
  -- proof would go here
  sorry

end total_bill_l123_123190


namespace inequality_solution_interval_l123_123072

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l123_123072


namespace tree_height_increase_l123_123209

theorem tree_height_increase
  (initial_height : ℝ)
  (height_increase : ℝ)
  (h6 : ℝ) :
  initial_height = 4 →
  (0 ≤ height_increase) →
  height_increase * 6 + initial_height = (height_increase * 4 + initial_height) + 1 / 7 * (height_increase * 4 + initial_height) →
  height_increase = 2 / 5 :=
by
  intro h_initial h_nonneg h_eq
  sorry

end tree_height_increase_l123_123209


namespace matrix_det_is_zero_l123_123039

noncomputable def matrixDetProblem (a b : ℝ) : ℝ :=
  Matrix.det ![
    ![1, Real.cos (a - b), Real.sin a],
    ![Real.cos (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

theorem matrix_det_is_zero (a b : ℝ) : matrixDetProblem a b = 0 :=
  sorry

end matrix_det_is_zero_l123_123039


namespace Carrie_can_add_turnips_l123_123206

-- Define the variables and conditions
def potatoToTurnipRatio (potatoes turnips : ℕ) : ℚ :=
  potatoes / turnips

def pastPotato : ℕ := 5
def pastTurnip : ℕ := 2
def currentPotato : ℕ := 20
def allowedTurnipAddition : ℕ := 8

-- Define the main theorem to prove, given the conditions.
theorem Carrie_can_add_turnips (past_p_ratio : potatoToTurnipRatio pastPotato pastTurnip = 2.5)
                                : potatoToTurnipRatio currentPotato allowedTurnipAddition = 2.5 :=
sorry

end Carrie_can_add_turnips_l123_123206


namespace problem_1_l123_123010

theorem problem_1 (a : ℝ) : (1 + a * x) * (1 + x) ^ 5 = 1 + 5 * x + 5 * i * x^2 → a = -1 := sorry

end problem_1_l123_123010


namespace overall_average_marks_l123_123533

theorem overall_average_marks (n P : ℕ) (P_avg F_avg : ℕ) (H_n : n = 120) (H_P : P = 100) (H_P_avg : P_avg = 39) (H_F_avg : F_avg = 15) :
  (P_avg * P + F_avg * (n - P)) / n = 35 := 
by
  sorry

end overall_average_marks_l123_123533


namespace find_overlapping_area_l123_123457

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end find_overlapping_area_l123_123457


namespace trip_savings_l123_123700

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l123_123700


namespace sum_of_coefficients_is_1_l123_123829

-- Given conditions:
def polynomial_expansion (x y : ℤ) := (x - 2 * y) ^ 18

-- Proof statement:
theorem sum_of_coefficients_is_1 : (polynomial_expansion 1 1) = 1 := by
  -- The proof itself is omitted as per the instruction
  sorry

end sum_of_coefficients_is_1_l123_123829


namespace polar_to_cartesian_l123_123989

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.sin θ) : 
  ∀ (x y : ℝ) (h₁ : x = ρ * Real.cos θ) (h₂ : y = ρ * Real.sin θ), 
    x^2 + (y - 1)^2 = 1 :=
by
  sorry

end polar_to_cartesian_l123_123989


namespace measure_8_liters_with_buckets_l123_123295

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l123_123295


namespace like_terms_sum_l123_123141

theorem like_terms_sum (m n : ℕ) (a b : ℝ) 
  (h₁ : 5 * a^m * b^3 = 5 * a^m * b^3) 
  (h₂ : -4 * a^2 * b^(n-1) = -4 * a^2 * b^(n-1)) 
  (h₃ : m = 2) (h₄ : 3 = n - 1) : m + n = 6 := by
  sorry

end like_terms_sum_l123_123141


namespace polynomial_sum_l123_123573

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end polynomial_sum_l123_123573


namespace algebraic_expression_solution_l123_123636

theorem algebraic_expression_solution
  (a b : ℝ)
  (h : -2 * a + 3 * b = 10) :
  9 * b - 6 * a + 2 = 32 :=
by 
  -- We would normally provide the proof here
  sorry

end algebraic_expression_solution_l123_123636


namespace cost_of_perfume_l123_123816

-- Definitions and Constants
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def neighbors_yards_mowed : ℕ := 4
def charge_per_yard : ℕ := 5
def dogs_walked : ℕ := 6
def charge_per_dog : ℕ := 2
def additional_amount_needed : ℕ := 6

-- Theorem Statement
theorem cost_of_perfume :
  let christian_earnings := neighbors_yards_mowed * charge_per_yard
  let sue_earnings := dogs_walked * charge_per_dog
  let christian_savings := christian_initial_savings + christian_earnings
  let sue_savings := sue_initial_savings + sue_earnings
  let total_savings := christian_savings + sue_savings
  total_savings + additional_amount_needed = 50 := 
by
  sorry

end cost_of_perfume_l123_123816


namespace vector_relation_condition_l123_123354

variables {V : Type*} [AddCommGroup V] (OD OE OM DO EO MO : V)

-- Given condition
theorem vector_relation_condition (h : OD + OE = OM) :

-- Option B
(OM + DO = OE) ∧ 

-- Option C
(OM - OE = OD) ∧ 

-- Option D
(DO + EO = MO) :=
by {
  -- Sorry, to focus on statement only
  sorry
}

end vector_relation_condition_l123_123354


namespace total_weight_of_rings_l123_123530

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

end total_weight_of_rings_l123_123530


namespace triangle_eq_medians_incircle_l123_123364

-- Define a triangle and the properties of medians and incircle
structure Triangle (α : Type) [Nonempty α] :=
(A B C : α)

def is_equilateral {α : Type} [Nonempty α] (T : Triangle α) : Prop :=
  ∃ (d : α → α → ℝ), d T.A T.B = d T.B T.C ∧ d T.B T.C = d T.C T.A

def medians_segments_equal {α : Type} [Nonempty α] (T : Triangle α) (incr_len : (α → α → ℝ)) : Prop :=
  ∀ (MA MB MC : α), incr_len MA MB = incr_len MB MC ∧ incr_len MB MC = incr_len MC MA

-- The main theorem statement
theorem triangle_eq_medians_incircle {α : Type} [Nonempty α] 
  (T : Triangle α) (incr_len : α → α → ℝ) 
  (h : medians_segments_equal T incr_len) : is_equilateral T :=
sorry

end triangle_eq_medians_incircle_l123_123364


namespace part1_part2_l123_123222

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x : ℝ) := x^2 - 1

theorem part1 {x : ℝ} (h : 1 ≤ x) : f x ≤ (1 / 2) * g x := by
  sorry

theorem part2 {m : ℝ} : (∀ x, 1 ≤ x → f x - m * g x ≤ 0) → m ≥ (1 / 2) := by
  sorry

end part1_part2_l123_123222


namespace sum_of_seven_consecutive_integers_l123_123968

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l123_123968


namespace maximum_value_fraction_l123_123485

theorem maximum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3 :=
sorry

end maximum_value_fraction_l123_123485


namespace quadratic_solution_sum_l123_123609

theorem quadratic_solution_sum
  (x : ℚ)
  (m n p : ℕ)
  (h_eq : (5 * x - 11) * x = -6)
  (h_form : ∃ m n p, x = (m + Real.sqrt n) / p ∧ x = (m - Real.sqrt n) / p)
  (h_gcd : Nat.gcd (Nat.gcd m n) p = 1) :
  m + n + p = 22 := 
sorry

end quadratic_solution_sum_l123_123609


namespace exist_distinct_xy_divisibility_divisibility_implies_equality_l123_123060

-- Part (a)
theorem exist_distinct_xy_divisibility (n : ℕ) (h_n : n > 0) :
  ∃ (x y : ℕ), x ≠ y ∧ (∀ j : ℕ, 1 ≤ j ∧ j ≤ n → (x + j) ∣ (y + j)) :=
sorry

-- Part (b)
theorem divisibility_implies_equality (x y : ℕ) (h : ∀ j : ℕ, (x + j) ∣ (y + j)) : 
  x = y :=
sorry

end exist_distinct_xy_divisibility_divisibility_implies_equality_l123_123060


namespace imaginary_part_div_l123_123535

open Complex

theorem imaginary_part_div (z1 z2 : ℂ) (h1 : z1 = 1 + I) (h2 : z2 = I) :
  Complex.im (z1 / z2) = -1 := by
  sorry

end imaginary_part_div_l123_123535


namespace calculate_expression_l123_123827

theorem calculate_expression (y : ℝ) (hy : y ≠ 0) : 
  (18 * y^3) * (4 * y^2) * (1/(2 * y)^3) = 9 * y^2 :=
by
  sorry

end calculate_expression_l123_123827


namespace perfect_square_of_sides_of_triangle_l123_123935

theorem perfect_square_of_sides_of_triangle 
  (a b c : ℤ) 
  (h1: a > 0 ∧ b > 0 ∧ c > 0)
  (h2: a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_abc: Int.gcd (Int.gcd a b) c = 1)
  (h3: (a^2 + b^2 - c^2) % (a + b - c) = 0)
  (h4: (b^2 + c^2 - a^2) % (b + c - a) = 0)
  (h5: (c^2 + a^2 - b^2) % (c + a - b) = 0) : 
  ∃ n : ℤ, n^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
  ∃ m : ℤ, m^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end perfect_square_of_sides_of_triangle_l123_123935


namespace exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l123_123766

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem exists_a_f_has_two_zeros (a : ℝ) :
  (0 < a ∧ a < 2) ∨ (-2 < a ∧ a < 0) → ∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧ f x₁ a = 0) ∧
  (0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧ f x₂ a = 0) ∧ x₁ ≠ x₂ := sorry

theorem range_of_a_for_f_eq_g :
  ∀ a : ℝ, a ∈ Set.Icc (-2 : ℝ) (3 : ℝ) →
  ∃ x₁ : ℝ, x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ f x₁ a = g 2 ∧
  ∃ x₂ : ℝ, x₂ ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧ f x₁ a = g x₂ := sorry

end exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l123_123766


namespace smallest_class_size_l123_123115

theorem smallest_class_size (n : ℕ) (h : 5 * n + 1 > 40) : ∃ k : ℕ, k >= 41 :=
by sorry

end smallest_class_size_l123_123115


namespace unique_ordered_triple_l123_123761

theorem unique_ordered_triple (a b c : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a^3 + b^3 + c^3 + 648 = (a + b + c)^3) :
  (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) :=
sorry

end unique_ordered_triple_l123_123761


namespace jimin_and_seokjin_total_l123_123974

def Jimin_coins := (5 * 100) + (1 * 50)
def Seokjin_coins := (2 * 100) + (7 * 10)
def total_coins := Jimin_coins + Seokjin_coins

theorem jimin_and_seokjin_total : total_coins = 820 :=
by
  sorry

end jimin_and_seokjin_total_l123_123974


namespace age_ordered_youngest_to_oldest_l123_123277

variable (M Q S : Nat)

theorem age_ordered_youngest_to_oldest 
  (h1 : M = Q ∨ S = Q)
  (h2 : M ≥ Q)
  (h3 : S ≤ Q) : S = Q ∧ M > Q :=
by 
  sorry

end age_ordered_youngest_to_oldest_l123_123277


namespace identify_infected_person_in_4_tests_l123_123167

theorem identify_infected_person_in_4_tests :
  (∀ (group : Fin 16 → Bool), ∃ infected : Fin 16, group infected = ff) →
  ∃ (tests_needed : ℕ), tests_needed = 4 :=
by sorry

end identify_infected_person_in_4_tests_l123_123167


namespace yoongi_average_score_l123_123817

/-- 
Yoongi's average score on the English test taken in August and September was 86, and his English test score in October was 98. 
Prove that the average score of the English test for 3 months is 90.
-/
theorem yoongi_average_score 
  (avg_aug_sep : ℕ)
  (score_oct : ℕ)
  (hp1 : avg_aug_sep = 86)
  (hp2 : score_oct = 98) :
  ((avg_aug_sep * 2 + score_oct) / 3) = 90 :=
by
  sorry

end yoongi_average_score_l123_123817


namespace inequality_inequality_l123_123879

open Real

theorem inequality_inequality (n : ℕ) (k : ℝ) (hn : 0 < n) (hk : 0 < k) : 
  1 - 1/k ≤ n * (k^(1 / n) - 1) ∧ n * (k^(1 / n) - 1) ≤ k - 1 := 
  sorry

end inequality_inequality_l123_123879


namespace coterminal_angle_in_radians_l123_123806

theorem coterminal_angle_in_radians (d : ℝ) (h : d = 2010) : 
  ∃ r : ℝ, r = -5 * Real.pi / 6 ∧ (∃ k : ℤ, d = r * 180 / Real.pi + k * 360) :=
by sorry

end coterminal_angle_in_radians_l123_123806


namespace pastries_average_per_day_l123_123613

theorem pastries_average_per_day :
  let monday_sales := 2
  let tuesday_sales := monday_sales + 1
  let wednesday_sales := tuesday_sales + 1
  let thursday_sales := wednesday_sales + 1
  let friday_sales := thursday_sales + 1
  let saturday_sales := friday_sales + 1
  let sunday_sales := saturday_sales + 1
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
  let days := 7
  total_sales / days = 5 := by
  sorry

end pastries_average_per_day_l123_123613


namespace find_a2_plus_b2_l123_123406

theorem find_a2_plus_b2
  (a b : ℝ)
  (h1 : a^3 - 3 * a * b^2 = 39)
  (h2 : b^3 - 3 * a^2 * b = 26) :
  a^2 + b^2 = 13 :=
sorry

end find_a2_plus_b2_l123_123406


namespace number_of_valid_pairs_l123_123978

theorem number_of_valid_pairs :
  ∃ (n : ℕ), n = 4950 ∧ ∀ (x y : ℕ), 
  1 ≤ x ∧ x < y ∧ y ≤ 200 ∧ 
  (Complex.I ^ x + Complex.I ^ y).im = 0 → n = 4950 :=
sorry

end number_of_valid_pairs_l123_123978


namespace min_value_expr_l123_123200

theorem min_value_expr : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 := 
sorry

end min_value_expr_l123_123200


namespace exist_matrices_with_dets_l123_123245

noncomputable section

open Matrix BigOperators

variables {α : Type} [Field α] [DecidableEq α]

theorem exist_matrices_with_dets (m n : ℕ) (h₁ : 1 < m) (h₂ : 1 < n)
  (αs : Fin m → α) (β : α) :
  ∃ (A : Fin m → Matrix (Fin n) (Fin n) α), (∀ i, det (A i) = αs i) ∧ det (∑ i, A i) = β :=
sorry

end exist_matrices_with_dets_l123_123245


namespace even_number_divisible_by_8_l123_123605

theorem even_number_divisible_by_8 {n : ℤ} (h : ∃ k : ℤ, n = 2 * k) : 
  (n * (n^2 + 20)) % 8 = 0 ∧ 
  (n * (n^2 - 20)) % 8 = 0 ∧ 
  (n * (n^2 + 4)) % 8 = 0 ∧ 
  (n * (n^2 - 4)) % 8 = 0 :=
by
  sorry

end even_number_divisible_by_8_l123_123605


namespace largest_mersenne_prime_is_127_l123_123489

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l123_123489


namespace mean_equivalence_l123_123393

theorem mean_equivalence :
  (20 + 30 + 40) / 3 = (23 + 30 + 37) / 3 :=
by sorry

end mean_equivalence_l123_123393


namespace sale_price_after_discounts_l123_123379

def calculate_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

theorem sale_price_after_discounts :
  calculate_sale_price 500 [0.10, 0.15, 0.20, 0.25, 0.30] = 160.65 :=
by
  sorry

end sale_price_after_discounts_l123_123379


namespace find_coordinates_C_find_range_t_l123_123523

-- required definitions to handle the given points and vectors
structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

-- Given points
def A : Point := ⟨0, 4⟩
def B : Point := ⟨2, 0⟩

-- Proof for coordinates of point C
theorem find_coordinates_C :
  ∃ C : Point, vector A B = {x := 2 * (C.x - B.x), y := 2 * C.y} ∧ C = ⟨3, -2⟩ :=
by
  sorry

-- Proof for range of t
theorem find_range_t (t : ℝ) :
  let P := Point.mk 3 t
  let PA := vector P A
  let PB := vector P B
  (dot_product PA PB < 0 ∧ -3 * t ≠ -1 * (4 - t)) → 1 < t ∧ t < 3 :=
by
  sorry

end find_coordinates_C_find_range_t_l123_123523


namespace initial_average_is_correct_l123_123452

def initial_average_daily_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) (initial_average : ℕ) :=
  let total_initial_production := initial_average * n
  let total_new_production := total_initial_production + today_production
  let total_days := n + 1
  total_new_production = new_average * total_days

theorem initial_average_is_correct :
  ∀ (A n today_production new_average : ℕ),
    n = 19 →
    today_production = 90 →
    new_average = 52 →
    initial_average_daily_production n today_production new_average A →
    A = 50 := by
    intros A n today_production new_average hn htoday hnew havg
    sorry

end initial_average_is_correct_l123_123452


namespace system_solution_l123_123779

theorem system_solution (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧ 
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4) ∧ (y = -1) :=
by
  sorry

end system_solution_l123_123779


namespace inequality_abc_d_l123_123199

theorem inequality_abc_d (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (H1 : d ≥ a) (H2 : d ≥ b) (H3 : d ≥ c) : a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 :=
by
  sorry

end inequality_abc_d_l123_123199


namespace polynomial_real_roots_l123_123728

theorem polynomial_real_roots :
  ∀ x : ℝ, (x^4 - 3 * x^3 + 3 * x^2 - x - 6 = 0) ↔ (x = 3 ∨ x = 2 ∨ x = -1) := 
by
  sorry

end polynomial_real_roots_l123_123728


namespace calculate_expression_l123_123715

theorem calculate_expression : 
  (-7 : ℤ)^7 / (7 : ℤ)^4 + 2^6 - 8^2 = -343 :=
by
  sorry

end calculate_expression_l123_123715


namespace remainder_17_pow_2047_mod_23_l123_123061

theorem remainder_17_pow_2047_mod_23 : (17 ^ 2047) % 23 = 11 := 
by
  sorry

end remainder_17_pow_2047_mod_23_l123_123061


namespace island_count_l123_123595

-- Defining the conditions
def lakes := 7
def canals := 10

-- Euler's formula for connected planar graph
def euler_characteristic (V E F : ℕ) := V - E + F = 2

-- Determine the number of faces using Euler's formula
def faces (V E : ℕ) :=
  let F := V - E + 2
  F

-- The number of islands is the number of faces minus one for the outer face
def number_of_islands (F : ℕ) :=
  F - 1

-- The given proof problem to be converted to Lean
theorem island_count :
  number_of_islands (faces lakes canals) = 4 :=
by
  unfold lakes canals faces number_of_islands
  sorry

end island_count_l123_123595


namespace number_of_guests_l123_123382

def cook_per_minute : ℕ := 10
def time_to_cook : ℕ := 80
def guests_ate_per_guest : ℕ := 5
def guests_to_serve : ℕ := 20 -- This is what we'll prove.

theorem number_of_guests 
    (cook_per_8min : cook_per_minute = 10)
    (total_time : time_to_cook = 80)
    (eat_rate : guests_ate_per_guest = 5) :
    (time_to_cook * cook_per_minute) / guests_ate_per_guest = guests_to_serve := 
by 
  sorry

end number_of_guests_l123_123382


namespace find_angle_y_l123_123008

theorem find_angle_y (ABC BAC BCA DCE CED y : ℝ)
  (h1 : ABC = 80) (h2 : BAC = 60)
  (h3 : ABC + BAC + BCA = 180)
  (h4 : CED = 90)
  (h5 : DCE = BCA)
  (h6 : DCE + CED + y = 180) :
  y = 50 :=
by
  sorry

end find_angle_y_l123_123008


namespace solve_fraction_zero_l123_123781

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 25) / (x + 5) = 0) (h2 : x ≠ -5) : x = 5 :=
sorry

end solve_fraction_zero_l123_123781


namespace gcd_888_1147_l123_123630

/-- Use the Euclidean algorithm to find the greatest common divisor (GCD) of 888 and 1147. -/
theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l123_123630


namespace find_a8_l123_123895

variable (a : ℕ → ℤ)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)

theorem find_a8 (h1 : a 7 + a 9 = 16) (h2 : arithmetic_sequence a) : a 8 = 8 := by
  -- proof would go here
  sorry

end find_a8_l123_123895


namespace rectangle_quadrilateral_inequality_l123_123909

theorem rectangle_quadrilateral_inequality 
  (a b c d : ℝ)
  (h_a : 0 < a) (h_a_bound : a < 3)
  (h_b : 0 < b) (h_b_bound : b < 4)
  (h_c : 0 < c) (h_c_bound : c < 3)
  (h_d : 0 < d) (h_d_bound : d < 4) :
  25 ≤ ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) ∧
  ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) < 50 :=
by 
  sorry

end rectangle_quadrilateral_inequality_l123_123909


namespace consecutive_even_sum_l123_123035

theorem consecutive_even_sum : 
  ∃ n : ℕ, 
  (∃ x : ℕ, (∀ i : ℕ, i < n → (2 * i + x = 14 → i = 2) → 
  2 * x + (n - 1) * n = 52) ∧ n = 4) :=
by
  sorry

end consecutive_even_sum_l123_123035


namespace man_l123_123835

-- Define the speeds and values given in the problem conditions
def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

-- Define the man's speed in still water as a variable
def man_speed_in_still_water : ℝ := man_speed_with_current - speed_of_current

-- The theorem we need to prove
theorem man's_speed_against_current_is_correct :
  (man_speed_in_still_water - speed_of_current = man_speed_against_current) :=
by
  -- Placeholder for proof
  sorry

end man_l123_123835


namespace irrational_sqrt_2023_l123_123600

theorem irrational_sqrt_2023 (A B C D : ℝ) :
  A = -2023 → B = Real.sqrt 2023 → C = 0 → D = 1 / 2023 →
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ B = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ A = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ C = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ D = p / q) := 
by
  intro hA hB hC hD
  sorry

end irrational_sqrt_2023_l123_123600


namespace equivalent_statements_l123_123551

theorem equivalent_statements (P Q : Prop) : (¬P → Q) ↔ (¬Q → P) :=
by
  sorry

end equivalent_statements_l123_123551


namespace store_paid_price_l123_123068

theorem store_paid_price (selling_price : ℕ) (less_amount : ℕ) 
(h1 : selling_price = 34) (h2 : less_amount = 8) : ∃ p : ℕ, p = selling_price - less_amount ∧ p = 26 := 
by
  sorry

end store_paid_price_l123_123068


namespace calculate_meals_l123_123930

-- Given conditions
def meal_cost : ℕ := 7
def total_spent : ℕ := 21

-- The expected number of meals Olivia's dad paid for
def expected_meals : ℕ := 3

-- Proof statement
theorem calculate_meals : total_spent / meal_cost = expected_meals :=
by
  sorry
  -- Proof can be completed using arithmetic simplification.

end calculate_meals_l123_123930


namespace inequaliy_pos_real_abc_l123_123940

theorem inequaliy_pos_real_abc (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 1) : 
  (a / (a * b + 1)) + (b / (b * c + 1)) + (c / (c * a + 1)) ≥ (3 / 2) := 
by
  sorry

end inequaliy_pos_real_abc_l123_123940


namespace mila_father_total_pay_l123_123131

def first_job_pay : ℤ := 2125
def pay_difference : ℤ := 375
def second_job_pay : ℤ := first_job_pay - pay_difference
def total_pay : ℤ := first_job_pay + second_job_pay

theorem mila_father_total_pay :
  total_pay = 3875 := by
  sorry

end mila_father_total_pay_l123_123131


namespace nat_numbers_equal_if_divisible_l123_123986

theorem nat_numbers_equal_if_divisible
  (a b : ℕ)
  (h : ∀ n : ℕ, ∃ m : ℕ, n ≠ m → (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end nat_numbers_equal_if_divisible_l123_123986


namespace line_slope_and_intersection_l123_123083

theorem line_slope_and_intersection:
  (∀ x y : ℝ, x^2 + x / 4 + y / 5 = 1 → ∀ m : ℝ, m = -5 / 4) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → ¬ (x^2 + x / 4 + y / 5 = 1)) :=
by
  sorry

end line_slope_and_intersection_l123_123083


namespace age_difference_l123_123422

theorem age_difference (Rona Rachel Collete : ℕ) (h1 : Rachel = 2 * Rona) (h2 : Collete = Rona / 2) (h3 : Rona = 8) : Rachel - Collete = 12 :=
by
  sorry

end age_difference_l123_123422


namespace parabola_directrix_l123_123743

theorem parabola_directrix (x y : ℝ) :
  (∃ a b c : ℝ, y = (a * x^2 + b * x + c) / 12 ∧ a = 1 ∧ b = -6 ∧ c = 5) →
  y = -10 / 3 :=
by
  sorry

end parabola_directrix_l123_123743


namespace combined_height_l123_123270

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end combined_height_l123_123270


namespace ratio_simplified_l123_123203

theorem ratio_simplified (kids_meals : ℕ) (adult_meals : ℕ) (h1 : kids_meals = 70) (h2 : adult_meals = 49) : 
  ∃ (k a : ℕ), k = 10 ∧ a = 7 ∧ kids_meals / Nat.gcd kids_meals adult_meals = k ∧ adult_meals / Nat.gcd kids_meals adult_meals = a :=
by
  sorry

end ratio_simplified_l123_123203


namespace minimum_sugar_amount_l123_123309

theorem minimum_sugar_amount (f s : ℕ) (h1 : f ≥ 9 + s / 2) (h2 : f ≤ 3 * s) : s ≥ 4 :=
by
  -- Provided conditions: f ≥ 9 + s / 2 and f ≤ 3 * s
  -- Goal: s ≥ 4
  sorry

end minimum_sugar_amount_l123_123309


namespace div_equiv_l123_123703

theorem div_equiv : (0.75 / 25) = (7.5 / 250) :=
by
  sorry

end div_equiv_l123_123703


namespace largest_three_digit_multiple_of_8_and_sum_24_is_888_l123_123219

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end largest_three_digit_multiple_of_8_and_sum_24_is_888_l123_123219


namespace boxes_contain_neither_markers_nor_sharpies_l123_123479

theorem boxes_contain_neither_markers_nor_sharpies :
  (∀ (total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes : ℕ),
    total_boxes = 15 → markers_boxes = 8 → sharpies_boxes = 5 → both_boxes = 4 →
    neither_boxes = total_boxes - (markers_boxes + sharpies_boxes - both_boxes) →
    neither_boxes = 6) :=
by
  intros total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes
  intros htotal hmarkers hsharpies hboth hcalc
  rw [htotal, hmarkers, hsharpies, hboth] at hcalc
  exact hcalc

end boxes_contain_neither_markers_nor_sharpies_l123_123479


namespace seating_arrangements_equal_600_l123_123276

-- Definitions based on the problem conditions
def number_of_people : Nat := 4
def number_of_chairs : Nat := 8
def consecutive_empty_seats : Nat := 3

-- Theorem statement
theorem seating_arrangements_equal_600
  (h_people : number_of_people = 4)
  (h_chairs : number_of_chairs = 8)
  (h_consecutive_empty_seats : consecutive_empty_seats = 3) :
  (∃ (arrangements : Nat), arrangements = 600) :=
sorry

end seating_arrangements_equal_600_l123_123276


namespace seeds_total_l123_123005

noncomputable def seeds_planted (x : ℕ) (y : ℕ) (z : ℕ) : ℕ :=
x + y + z

theorem seeds_total (x : ℕ) (H1 :  y = 5 * x) (H2 : x + y = 156) (z : ℕ) 
(H3 : z = 4) : seeds_planted x y z = 160 :=
by
  sorry

end seeds_total_l123_123005


namespace arithmetic_expression_evaluation_l123_123582

theorem arithmetic_expression_evaluation :
  3^2 + 4 * 2 - 6 / 3 + 7 = 22 :=
by 
  -- Use tactics to break down the arithmetic expression evaluation (steps are abstracted)
  sorry

end arithmetic_expression_evaluation_l123_123582


namespace medicine_supply_duration_l123_123085

noncomputable def pillDuration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) : ℚ :=
  let pillPerDay := pillFractionPerThreeDays / 3
  let daysPerPill := 1 / pillPerDay
  numPills * daysPerPill

theorem medicine_supply_duration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) (daysPerMonth : ℚ) :
  numPills = 90 →
  pillFractionPerThreeDays = 1 / 3 →
  daysPerMonth = 30 →
  pillDuration numPills pillFractionPerThreeDays / daysPerMonth = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [pillDuration]
  sorry

end medicine_supply_duration_l123_123085


namespace porter_previous_painting_price_l123_123893

variable (P : ℝ)

-- Conditions
def condition1 : Prop := 3.5 * P - 1000 = 49000

-- Correct Answer
def answer : ℝ := 14285.71

-- Theorem stating that the answer holds given the conditions
theorem porter_previous_painting_price (h : condition1 P) : P = answer :=
sorry

end porter_previous_painting_price_l123_123893


namespace find_aroon_pin_l123_123537

theorem find_aroon_pin (a b : ℕ) (PIN : ℕ) 
  (h0 : 0 ≤ a ∧ a ≤ 9)
  (h1 : 0 ≤ b ∧ b < 1000)
  (h2 : PIN = 1000 * a + b)
  (h3 : 10 * b + a = 3 * PIN - 6) : 
  PIN = 2856 := 
sorry

end find_aroon_pin_l123_123537


namespace sum_of_three_numbers_l123_123655

theorem sum_of_three_numbers (a b c : ℝ) :
  a + b = 35 → b + c = 47 → c + a = 58 → a + b + c = 70 :=
by
  intros h1 h2 h3
  sorry

end sum_of_three_numbers_l123_123655


namespace ratio_alisha_to_todd_is_two_to_one_l123_123837

-- Definitions
def total_gumballs : ℕ := 45
def todd_gumballs : ℕ := 4
def bobby_gumballs (A : ℕ) : ℕ := 4 * A - 5
def remaining_gumballs : ℕ := 6

-- Condition stating Hector's gumball distribution
def hector_gumballs_distribution (A : ℕ) : Prop :=
  todd_gumballs + A + bobby_gumballs A + remaining_gumballs = total_gumballs

-- Definition for the ratio of the gumballs given to Alisha to Todd
def ratio_alisha_todd (A : ℕ) : ℕ × ℕ :=
  (A / 4, todd_gumballs / 4)

-- Theorem stating the problem
theorem ratio_alisha_to_todd_is_two_to_one : ∃ (A : ℕ), hector_gumballs_distribution A → ratio_alisha_todd A = (2, 1) :=
sorry

end ratio_alisha_to_todd_is_two_to_one_l123_123837


namespace triangle_similarity_proof_l123_123606

-- Define a structure for points in a geometric space
structure Point : Type where
  x : ℝ
  y : ℝ
  deriving Inhabited

-- Define the conditions provided in the problem
variables (A B C D E H : Point)
variables (HD HE : ℝ)

-- Condition statements
def HD_dist := HD = 6
def HE_dist := HE = 3

-- Main theorem statement
theorem triangle_similarity_proof (BD DC AE EC BH AH : ℝ) 
  (h1 : HD = 6) (h2 : HE = 3) 
  (h3 : 2 * BH = AH) : 
  (BD * DC - AE * EC = 9 * BH + 27) :=
sorry

end triangle_similarity_proof_l123_123606


namespace train_speed_kmph_l123_123884

def length_of_train : ℝ := 120
def time_to_cross_bridge : ℝ := 17.39860811135109
def length_of_bridge : ℝ := 170

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60 := 
by
  sorry

end train_speed_kmph_l123_123884


namespace mitzi_amount_brought_l123_123733

-- Define the amounts spent on different items
def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23

-- Define the amount of money left
def amount_left : ℕ := 9

-- Define the total amount spent
def total_spent : ℕ :=
  ticket_cost + food_cost + tshirt_cost

-- Define the total amount brought to the amusement park
def amount_brought : ℕ :=
  total_spent + amount_left

-- Prove that the amount of money Mitzi brought to the amusement park is 75
theorem mitzi_amount_brought : amount_brought = 75 := by
  sorry

end mitzi_amount_brought_l123_123733


namespace boys_to_girls_ratio_l123_123401

theorem boys_to_girls_ratio (boys girls : ℕ) (h_boys : boys = 1500) (h_girls : girls = 1200) : 
  (boys / Nat.gcd boys girls) = 5 ∧ (girls / Nat.gcd boys girls) = 4 := 
by 
  sorry

end boys_to_girls_ratio_l123_123401


namespace two_roses_more_than_three_carnations_l123_123691

variable {x y : ℝ}

theorem two_roses_more_than_three_carnations
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y := 
by 
  sorry

end two_roses_more_than_three_carnations_l123_123691


namespace average_weight_increase_l123_123160

theorem average_weight_increase
 (num_persons : ℕ) (weight_increase : ℝ) (replacement_weight : ℝ) (new_weight : ℝ) (weight_difference : ℝ) (avg_weight_increase : ℝ)
 (cond1 : num_persons = 10)
 (cond2 : replacement_weight = 65)
 (cond3 : new_weight = 90)
 (cond4 : weight_difference = new_weight - replacement_weight)
 (cond5 : weight_difference = weight_increase)
 (cond6 : avg_weight_increase = weight_increase / num_persons) :
avg_weight_increase = 2.5 :=
by
  sorry

end average_weight_increase_l123_123160


namespace sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l123_123532

theorem sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5 : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 :=
sorry

end sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l123_123532


namespace smallest_real_solution_l123_123845

theorem smallest_real_solution (x : ℝ) : 
  (x * |x| = 3 * x + 4) → x = 4 :=
by {
  sorry -- Proof omitted as per the instructions
}

end smallest_real_solution_l123_123845


namespace find_S_l123_123789

noncomputable def A := { x : ℝ | x^2 - 7 * x + 10 ≤ 0 }
noncomputable def B (a b : ℝ) := { x : ℝ | x^2 + a * x + b < 0 }
def A_inter_B_is_empty (a b : ℝ) := A ∩ B a b = ∅
def A_union_B_condition := { x : ℝ | x - 3 < 4 ∧ 4 ≤ 2 * x }

theorem find_S :
  A ∪ B (-12) 35 = { x : ℝ | 2 ≤ x ∧ x < 7 } →
  A ∩ B (-12) 35 = ∅ →
  { x : ℝ | x = -12 + 35 } = { 23 } :=
by
  intro h1 h2
  sorry

end find_S_l123_123789


namespace number_of_SUVs_washed_l123_123172

theorem number_of_SUVs_washed (charge_car charge_truck charge_SUV total_raised : ℕ) (num_trucks num_cars S : ℕ) :
  charge_car = 5 →
  charge_truck = 6 →
  charge_SUV = 7 →
  total_raised = 100 →
  num_trucks = 5 →
  num_cars = 7 →
  total_raised = num_cars * charge_car + num_trucks * charge_truck + S * charge_SUV →
  S = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_SUVs_washed_l123_123172


namespace toot_has_vertical_symmetry_l123_123162

def has_vertical_symmetry (letter : Char) : Prop :=
  letter = 'T' ∨ letter = 'O'

def word_has_vertical_symmetry (word : List Char) : Prop :=
  ∀ letter ∈ word, has_vertical_symmetry letter

theorem toot_has_vertical_symmetry : word_has_vertical_symmetry ['T', 'O', 'O', 'T'] :=
  by
    sorry

end toot_has_vertical_symmetry_l123_123162


namespace douglas_votes_in_county_D_l123_123329

noncomputable def percent_votes_in_county_D (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
    (total_votes : ℝ) (percent_A percent_B percent_C percent_D total_percent : ℝ) : Prop :=
  (votes_A / (5 * x) = 0.70) ∧
  (votes_B / (3 * x) = 0.58) ∧
  (votes_C / (2 * x) = 0.50) ∧
  (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62 ∧
  (votes_D / (4 * x) = percent_D)

theorem douglas_votes_in_county_D 
  (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
  (total_votes : ℝ := 14 * x) 
  (percent_A percent_B percent_C total_percent percent_D : ℝ)
  (h1 : votes_A / (5 * x) = 0.70) 
  (h2 : votes_B / (3 * x) = 0.58) 
  (h3 : votes_C / (2 * x) = 0.50) 
  (h4 : (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62) : 
  percent_votes_in_county_D x votes_A votes_B votes_C votes_D total_votes percent_A percent_B percent_C 0.61 total_percent :=
by
  constructor
  exact h1
  constructor
  exact h2
  constructor
  exact h3
  constructor
  exact h4
  sorry

end douglas_votes_in_county_D_l123_123329


namespace trajectory_of_moving_point_l123_123865

theorem trajectory_of_moving_point (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
  (hF1 : F1 = (-2, 0)) (hF2 : F2 = (2, 0))
  (h_arith_mean : dist F1 F2 = (dist P F1 + dist P F2) / 2) :
  ∃ a b : ℝ, a = 4 ∧ b^2 = 12 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1) :=
sorry

end trajectory_of_moving_point_l123_123865


namespace unique_two_digit_solution_l123_123947

theorem unique_two_digit_solution : ∃! (t : ℕ), 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := sorry

end unique_two_digit_solution_l123_123947


namespace interest_calculation_correct_l123_123854

-- Define the principal amounts and their respective interest rates
def principal1 : ℝ := 3000
def rate1 : ℝ := 0.08
def principal2 : ℝ := 8000 - principal1
def rate2 : ℝ := 0.05

-- Calculate interest for one year
def interest1 : ℝ := principal1 * rate1 * 1
def interest2 : ℝ := principal2 * rate2 * 1

-- Define the total interest
def total_interest : ℝ := interest1 + interest2

-- Prove that the total interest calculated is $490
theorem interest_calculation_correct : total_interest = 490 := by
  sorry

end interest_calculation_correct_l123_123854


namespace largest_percent_error_l123_123383
noncomputable def max_percent_error (d : ℝ) (d_err : ℝ) (r_err : ℝ) : ℝ :=
  let d_min := d - d * d_err
  let d_max := d + d * d_err
  let r := d / 2
  let r_min := r - r * r_err
  let r_max := r + r * r_err
  let area_actual := Real.pi * r^2
  let area_d_min := Real.pi * (d_min / 2)^2
  let area_d_max := Real.pi * (d_max / 2)^2
  let area_r_min := Real.pi * r_min^2
  let area_r_max := Real.pi * r_max^2
  let error_d_min := (area_actual - area_d_min) / area_actual * 100
  let error_d_max := (area_d_max - area_actual) / area_actual * 100
  let error_r_min := (area_actual - area_r_min) / area_actual * 100
  let error_r_max := (area_r_max - area_actual) / area_actual * 100
  max (max error_d_min error_d_max) (max error_r_min error_r_max)

theorem largest_percent_error 
  (d : ℝ) (d_err : ℝ) (r_err : ℝ) 
  (h_d : d = 30) (h_d_err : d_err = 0.15) (h_r_err : r_err = 0.10) : 
  max_percent_error d d_err r_err = 31.57 := by
  sorry

end largest_percent_error_l123_123383


namespace cheesecake_needs_more_eggs_l123_123279

def chocolate_eggs_per_cake := 3
def cheesecake_eggs_per_cake := 8
def num_chocolate_cakes := 5
def num_cheesecakes := 9

theorem cheesecake_needs_more_eggs :
  cheesecake_eggs_per_cake * num_cheesecakes - chocolate_eggs_per_cake * num_chocolate_cakes = 57 :=
by
  sorry

end cheesecake_needs_more_eggs_l123_123279


namespace surface_area_of_cube_l123_123301

theorem surface_area_of_cube (V : ℝ) (H : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end surface_area_of_cube_l123_123301


namespace scrabble_champions_l123_123453

noncomputable def num_champions : Nat := 25
noncomputable def male_percentage : Nat := 40
noncomputable def bearded_percentage : Nat := 40
noncomputable def bearded_bald_percentage : Nat := 60
noncomputable def non_bearded_bald_percentage : Nat := 30

theorem scrabble_champions :
  let male_champions := (male_percentage * num_champions) / 100
  let bearded_champions := (bearded_percentage * male_champions) / 100
  let bearded_bald_champions := (bearded_bald_percentage * bearded_champions) / 100
  let bearded_hair_champions := bearded_champions - bearded_bald_champions
  let non_bearded_champions := male_champions - bearded_champions
  let non_bearded_bald_champions := (non_bearded_bald_percentage * non_bearded_champions) / 100
  let non_bearded_hair_champions := non_bearded_champions - non_bearded_bald_champions
  bearded_bald_champions = 2 ∧ 
  bearded_hair_champions = 2 ∧ 
  non_bearded_bald_champions = 1 ∧ 
  non_bearded_hair_champions = 5 :=
by
  sorry

end scrabble_champions_l123_123453


namespace inequality_lemma_l123_123757

theorem inequality_lemma (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (b * c + c * d + d * a - 1)) +
  (1 / (a * b + c * d + d * a - 1)) +
  (1 / (a * b + b * c + d * a - 1)) +
  (1 / (a * b + b * c + c * d - 1)) ≤ 2 :=
sorry

end inequality_lemma_l123_123757


namespace proof_problem_l123_123403

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x| > 1}

def B : Set ℝ := {x | (0 : ℝ) < x ∧ x ≤ 2}

def complement_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def intersection (s1 s2 : Set ℝ) : Set ℝ := s1 ∩ s2

theorem proof_problem : (complement_A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
by {
  sorry
}

end proof_problem_l123_123403


namespace num_trucks_washed_l123_123490

theorem num_trucks_washed (total_revenue cars_revenue suvs_revenue truck_charge : ℕ) 
  (h_total : total_revenue = 100)
  (h_cars : cars_revenue = 7 * 5)
  (h_suvs : suvs_revenue = 5 * 7)
  (h_truck_charge : truck_charge = 6) : 
  ∃ T : ℕ, (total_revenue - suvs_revenue - cars_revenue) / truck_charge = T := 
by {
  use 5,
  sorry
}

end num_trucks_washed_l123_123490


namespace base8_base9_equivalence_l123_123469

def base8_digit (x : ℕ) := 0 ≤ x ∧ x < 8
def base9_digit (y : ℕ) := 0 ≤ y ∧ y < 9

theorem base8_base9_equivalence 
    (X Y : ℕ) 
    (hX : base8_digit X) 
    (hY : base9_digit Y) 
    (h_eq : 8 * X + Y = 9 * Y + X) :
    (8 * 7 + 6 = 62) :=
by
  sorry

end base8_base9_equivalence_l123_123469


namespace evaluate_expression_l123_123943

theorem evaluate_expression : 7899665 - 12 * 3 * 2 = 7899593 :=
by
  -- This proof is skipped.
  sorry

end evaluate_expression_l123_123943


namespace jerrys_breakfast_calories_l123_123202

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end jerrys_breakfast_calories_l123_123202


namespace polynomial_remainder_l123_123231

theorem polynomial_remainder (P : Polynomial ℝ) (a : ℝ) :
  ∃ (Q : Polynomial ℝ) (r : ℝ), P = Q * (Polynomial.X - Polynomial.C a) + Polynomial.C r ∧ r = (P.eval a) :=
by
  sorry

end polynomial_remainder_l123_123231


namespace profit_without_discount_l123_123451

theorem profit_without_discount
  (CP SP_with_discount : ℝ) 
  (H1 : CP = 100) -- Assume cost price is 100
  (H2 : SP_with_discount = CP + 0.216 * CP) -- Selling price with discount
  (H3 : SP_with_discount = 0.95 * SP_without_discount) -- SP with discount is 95% of SP without discount
  : (SP_without_discount - CP) / CP * 100 = 28 := 
by
  -- proof goes here
  sorry

end profit_without_discount_l123_123451


namespace D_is_necessary_but_not_sufficient_condition_for_A_l123_123358

variable (A B C D : Prop)

-- Conditions
axiom A_implies_B : A → B
axiom not_B_implies_A : ¬ (B → A)
axiom B_iff_C : B ↔ C
axiom C_implies_D : C → D
axiom not_D_implies_C : ¬ (D → C)

theorem D_is_necessary_but_not_sufficient_condition_for_A : (A → D) ∧ ¬ (D → A) :=
by sorry

end D_is_necessary_but_not_sufficient_condition_for_A_l123_123358


namespace positive_value_m_l123_123336

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l123_123336


namespace max_wins_l123_123826

theorem max_wins (Chloe_wins Max_wins : ℕ) (h1 : Chloe_wins = 24) (h2 : 8 * Max_wins = 3 * Chloe_wins) : Max_wins = 9 := by
  sorry

end max_wins_l123_123826


namespace minimum_value_problem_l123_123053

theorem minimum_value_problem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1 / 2) : 
  ∃ m : ℝ, m = 10 ∧ ∀ z, z = (2 / (1 - x) + 1 / (1 - y)) → z ≥ m :=
by
  sorry

end minimum_value_problem_l123_123053


namespace choosing_top_cases_l123_123249

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end choosing_top_cases_l123_123249


namespace find_a_l123_123051

noncomputable def f' (x : ℝ) (a : ℝ) := 2 * x^3 + a * x^2 + x

theorem find_a (a : ℝ) (h : f' 1 a = 9) : a = 6 :=
by
  sorry

end find_a_l123_123051


namespace at_most_two_even_l123_123984

-- Assuming the negation of the proposition
def negate_condition (a b c : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0

-- Proposition to prove by contradiction
theorem at_most_two_even 
  (a b c : ℕ) 
  (h : negate_condition a b c) 
  : False :=
sorry

end at_most_two_even_l123_123984


namespace simplify_expression_l123_123576

open Real

theorem simplify_expression :
    (3 * (sqrt 5 + sqrt 7) / (4 * sqrt (3 + sqrt 5))) = sqrt (414 - 98 * sqrt 35) / 8 :=
by
  sorry

end simplify_expression_l123_123576


namespace gumball_difference_l123_123299

theorem gumball_difference :
  let c := 17
  let l := 12
  let a := 24
  let t := 8
  let n := c + l + a + t
  let low := 14
  let high := 32
  ∃ x : ℕ, (low ≤ (n + x) / 7 ∧ (n + x) / 7 ≤ high) →
  (∃ x_min x_max, x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126) :=
by
  sorry

end gumball_difference_l123_123299


namespace james_present_age_l123_123937

-- Definitions and conditions
variables (D J : ℕ) -- Dan's and James's ages are natural numbers

-- Condition 1: The ratio between Dan's and James's ages
def ratio_condition : Prop := (D * 5 = J * 6)

-- Condition 2: In 4 years, Dan will be 28
def future_age_condition : Prop := (D + 4 = 28)

-- The proof goal: James's present age is 20
theorem james_present_age : ratio_condition D J ∧ future_age_condition D → J = 20 :=
by
  sorry

end james_present_age_l123_123937


namespace totalStudents_l123_123491

-- Define the number of seats per ride
def seatsPerRide : ℕ := 15

-- Define the number of empty seats per ride
def emptySeatsPerRide : ℕ := 3

-- Define the number of rides taken
def ridesTaken : ℕ := 18

-- Define the number of students per ride
def studentsPerRide (seats : ℕ) (empty : ℕ) : ℕ := seats - empty

-- Calculate the total number of students
theorem totalStudents : studentsPerRide seatsPerRide emptySeatsPerRide * ridesTaken = 216 :=
by
  sorry

end totalStudents_l123_123491


namespace choir_students_min_l123_123118

/-- 
  Prove that the minimum number of students in the choir, where the number 
  of students must be a multiple of 9, 10, and 11, is 990. 
-/
theorem choir_students_min (n : ℕ) :
  (∃ n, n > 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ (∀ m, m > 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → n ≤ m) → n = 990 :=
by
  sorry

end choir_students_min_l123_123118


namespace probability_of_Q_section_l123_123187

theorem probability_of_Q_section (sections : ℕ) (Q_sections : ℕ) (h1 : sections = 6) (h2 : Q_sections = 2) :
  Q_sections / sections = 2 / 6 :=
by
  -- solution proof is skipped
  sorry

end probability_of_Q_section_l123_123187


namespace find_h_plus_k_l123_123863

theorem find_h_plus_k (h k : ℝ) :
  (∀ (x y : ℝ),
    (x - 3) ^ 2 + (y + 4) ^ 2 = 49) → 
  h = 3 ∧ k = -4 → 
  h + k = -1 :=
by
  sorry

end find_h_plus_k_l123_123863


namespace sandy_books_l123_123151

theorem sandy_books (x : ℕ)
  (h1 : 1080 + 840 = 1920)
  (h2 : 16 = 1920 / (x + 55)) :
  x = 65 :=
by
  -- Theorem proof placeholder
  sorry

end sandy_books_l123_123151


namespace sqrt_square_identity_l123_123038

-- Define the concept of square root
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Problem statement: prove (sqrt 12321)^2 = 12321
theorem sqrt_square_identity (x : ℝ) : (sqrt x) ^ 2 = x := by
  sorry

-- Specific instance for the given number
example : (sqrt 12321) ^ 2 = 12321 := sqrt_square_identity 12321

end sqrt_square_identity_l123_123038


namespace equation_solution_l123_123598

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 14) = (3 - x) / (x - 2) ↔ x = 3 ∨ x = -5 :=
by 
  sorry

end equation_solution_l123_123598


namespace arithmetic_sequence_sum_l123_123831

variable (a : ℕ → ℤ)

def arithmetic_sequence_condition_1 := a 5 = 3
def arithmetic_sequence_condition_2 := a 6 = -2

theorem arithmetic_sequence_sum :
  arithmetic_sequence_condition_1 a →
  arithmetic_sequence_condition_2 a →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_l123_123831


namespace length_width_percentage_change_l123_123886

variables (L W : ℝ) (x : ℝ)
noncomputable def area_change_percent : ℝ :=
  (L * (1 + x / 100) * W * (1 - x / 100) - L * W) / (L * W) * 100

theorem length_width_percentage_change (h : area_change_percent L W x = 4) :
  x = 20 :=
by
  sorry

end length_width_percentage_change_l123_123886


namespace students_count_l123_123711

noncomputable def num_students (N T : ℕ) : Prop :=
  T = 72 * N ∧ (T - 200) / (N - 5) = 92

theorem students_count (N T : ℕ) : num_students N T → N = 13 :=
by
  sorry

end students_count_l123_123711


namespace smallest_value_of_linear_expression_l123_123705

theorem smallest_value_of_linear_expression :
  (∃ a, 8 * a^2 + 6 * a + 5 = 7 ∧ (∃ b, b = 3 * a + 2 ∧ ∀ c, (8 * c^2 + 6 * c + 5 = 7 → 3 * c + 2 ≥ b))) → -1 = b :=
by
  sorry

end smallest_value_of_linear_expression_l123_123705


namespace finding_f_of_neg_half_l123_123822

def f (x : ℝ) : ℝ := sorry

theorem finding_f_of_neg_half : f (-1/2) = Real.pi / 3 :=
by
  -- Given function definition condition: f (cos x) = x / 2 for 0 ≤ x ≤ π
  -- f should be defined on ℝ -> ℝ such that this condition holds;
  -- Applying this condition should verify our theorem.
  sorry

end finding_f_of_neg_half_l123_123822


namespace output_sequence_value_l123_123694

theorem output_sequence_value (x y : Int) (seq : List (Int × Int))
  (h : (x, y) ∈ seq) (h_y : y = -10) : x = 32 :=
by
  sorry

end output_sequence_value_l123_123694


namespace number_of_integer_pairs_l123_123961

theorem number_of_integer_pairs (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_ineq : m^2 + m * n < 30) :
  ∃ k : ℕ, k = 48 :=
sorry

end number_of_integer_pairs_l123_123961


namespace prob_not_answered_after_three_rings_l123_123497

def prob_first_ring_answered := 0.1
def prob_second_ring_answered := 0.25
def prob_third_ring_answered := 0.45

theorem prob_not_answered_after_three_rings : 
  1 - prob_first_ring_answered - prob_second_ring_answered - prob_third_ring_answered = 0.2 :=
by
  sorry

end prob_not_answered_after_three_rings_l123_123497


namespace infinite_bad_integers_l123_123287

theorem infinite_bad_integers (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ᶠ n in at_top, (¬(n^b + 1) ∣ (a^n + 1)) :=
by
  sorry

end infinite_bad_integers_l123_123287


namespace log_fraction_eq_l123_123455

variable (a b : ℝ)
axiom h1 : a = Real.logb 3 5
axiom h2 : b = Real.logb 5 7

theorem log_fraction_eq : Real.logb 15 (49 / 45) = (2 * (a * b) - a - 2) / (1 + a) :=
by sorry

end log_fraction_eq_l123_123455


namespace simplify_expression_l123_123870

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 :=
by
  sorry

end simplify_expression_l123_123870


namespace percentage_less_than_l123_123769

variable (x y : ℝ)
variable (H : y = 1.4 * x)

theorem percentage_less_than :
  ((y - x) / y) * 100 = 28.57 := by
  sorry

end percentage_less_than_l123_123769


namespace cone_volume_l123_123440

theorem cone_volume (l h : ℝ) (l_eq : l = 5) (h_eq : h = 4) : 
  (1 / 3) * Real.pi * ((l^2 - h^2).sqrt)^2 * h = 12 * Real.pi := 
by 
  sorry

end cone_volume_l123_123440


namespace find_f_2004_l123_123995

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom odd_g : ∀ x : ℝ, g (-x) = -g x
axiom g_eq_f_shift : ∀ x : ℝ, g x = f (x - 1)
axiom g_one : g 1 = 2003

theorem find_f_2004 : f 2004 = 2003 :=
  sorry

end find_f_2004_l123_123995


namespace proposition_p_and_not_q_l123_123477

theorem proposition_p_and_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬ (P ∧ Q)) : (P ↔ ¬ Q) :=
sorry

end proposition_p_and_not_q_l123_123477


namespace basketball_free_throws_l123_123318

/-
Given the following conditions:
1. The players scored twice as many points with three-point shots as with two-point shots: \( 3b = 2a \).
2. The number of successful free throws was one more than the number of successful two-point shots: \( x = a + 1 \).
3. The team’s total score was 84 points: \( 2a + 3b + x = 84 \).

Prove that the number of free throws \( x \) equals 16.
-/
theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = a + 1) 
  (h3 : 2 * a + 3 * b + x = 84) : 
  x = 16 := 
  sorry

end basketball_free_throws_l123_123318


namespace coin_game_goal_l123_123181

theorem coin_game_goal (a b : ℕ) (h_diff : a ≤ 3 * b ∧ b ≤ 3 * a) (h_sum : (a + b) % 4 = 0) :
  ∃ x y p q : ℕ, (a + 2 * x - 2 * y = 3 * (b + 2 * p - 2 * q)) ∨ (a + 2 * y - 2 * x = 3 * (b + 2 * q - 2 * p)) :=
sorry

end coin_game_goal_l123_123181


namespace train_length_l123_123770

theorem train_length
  (S : ℝ)  -- speed of the train in meters per second
  (L : ℝ)  -- length of the train in meters
  (h1 : L = S * 20)
  (h2 : L + 500 = S * 40) :
  L = 500 := 
sorry

end train_length_l123_123770


namespace function_neither_even_nor_odd_l123_123956

noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^3))

theorem function_neither_even_nor_odd :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) := by
  sorry

end function_neither_even_nor_odd_l123_123956


namespace Martha_time_spent_l123_123824

theorem Martha_time_spent
  (x : ℕ)
  (h1 : 6 * x = 6 * x) -- Time spent on hold with Comcast is 6 times the time spent turning router off and on again
  (h2 : 3 * x = 3 * x) -- Time spent yelling at the customer service rep is half of time spent on hold, which is still 3x
  (h3 : x + 6 * x + 3 * x = 100) -- Total time spent is 100 minutes
  : x = 10 := 
by
  -- skip the proof steps
  sorry

end Martha_time_spent_l123_123824


namespace polynomial_strictly_monotone_l123_123467

def strictly_monotone (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem polynomial_strictly_monotone
  (P : ℝ → ℝ)
  (H1 : strictly_monotone (P ∘ P))
  (H2 : strictly_monotone (P ∘ P ∘ P)) :
  strictly_monotone P :=
sorry

end polynomial_strictly_monotone_l123_123467


namespace men_in_club_l123_123671

-- Definitions
variables (M W : ℕ) -- Number of men and women

-- Conditions
def club_members := M + W = 30
def event_participation := W / 3 + M = 18

-- Goal
theorem men_in_club : club_members M W → event_participation M W → M = 12 :=
sorry

end men_in_club_l123_123671


namespace possible_values_of_sum_l123_123561

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l123_123561


namespace parabola_int_x_axis_for_all_m_l123_123754

theorem parabola_int_x_axis_for_all_m {n : ℝ} :
  (∀ m : ℝ, (9 * m^2 - 4 * m - 4 * n) ≥ 0) → (n ≤ -1 / 9) :=
by
  intro h
  sorry

end parabola_int_x_axis_for_all_m_l123_123754


namespace expand_polynomials_l123_123089

variable (x : ℝ)

theorem expand_polynomials : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 :=
  by
  sorry

end expand_polynomials_l123_123089


namespace value_of_x_minus_y_l123_123682

theorem value_of_x_minus_y (x y : ℝ) (h1 : x = -(-3)) (h2 : |y| = 5) (h3 : x * y < 0) : x - y = 8 := 
sorry

end value_of_x_minus_y_l123_123682


namespace B_finishes_remaining_work_in_3_days_l123_123285

theorem B_finishes_remaining_work_in_3_days
  (A_works_in : ℕ)
  (B_works_in : ℕ)
  (work_days_together : ℕ)
  (A_leaves : A_works_in = 4)
  (B_leaves : B_works_in = 10)
  (work_days : work_days_together = 2) :
  ∃ days_remaining : ℕ, days_remaining = 3 :=
by
  sorry

end B_finishes_remaining_work_in_3_days_l123_123285


namespace addilynn_eggs_initial_l123_123644

theorem addilynn_eggs_initial (E : ℕ) (H1 : ∃ (E : ℕ), (E / 2) - 15 = 21) : E = 72 :=
by
  sorry

end addilynn_eggs_initial_l123_123644


namespace number_solution_l123_123241

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end number_solution_l123_123241


namespace jared_annual_salary_l123_123325

def monthly_salary_diploma_holder : ℕ := 4000
def factor_degree_to_diploma : ℕ := 3
def months_in_year : ℕ := 12

theorem jared_annual_salary :
  (factor_degree_to_diploma * monthly_salary_diploma_holder) * months_in_year = 144000 :=
by
  sorry

end jared_annual_salary_l123_123325


namespace remainder_8_digit_non_decreasing_integers_mod_1000_l123_123519

noncomputable def M : ℕ :=
  Nat.choose 17 8

theorem remainder_8_digit_non_decreasing_integers_mod_1000 :
  M % 1000 = 310 :=
by
  sorry

end remainder_8_digit_non_decreasing_integers_mod_1000_l123_123519


namespace school_spent_total_l123_123343

noncomputable def seminar_fee (num_teachers : ℕ) : ℝ :=
  let base_fee := 150 * num_teachers
  if num_teachers >= 20 then
    base_fee * 0.925
  else if num_teachers >= 10 then
    base_fee * 0.95
  else
    base_fee

noncomputable def seminar_fee_with_tax (num_teachers : ℕ) : ℝ :=
  let fee := seminar_fee num_teachers
  fee * 1.06

noncomputable def food_allowance (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  let num_regular := num_teachers - num_special
  num_regular * 10 + num_special * 15

noncomputable def total_cost (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  seminar_fee_with_tax num_teachers + food_allowance num_teachers num_special

theorem school_spent_total (num_teachers num_special : ℕ) (h : num_teachers = 22 ∧ num_special = 3) :
  total_cost num_teachers num_special = 3470.65 :=
by
  sorry

end school_spent_total_l123_123343


namespace multiple_people_sharing_carriage_l123_123748

theorem multiple_people_sharing_carriage (x : ℝ) : 
  (x / 3) + 2 = (x - 9) / 2 :=
sorry

end multiple_people_sharing_carriage_l123_123748


namespace minimum_a1_a2_sum_l123_123812

theorem minimum_a1_a2_sum (a : ℕ → ℕ)
  (h : ∀ n ≥ 1, a (n + 2) = (a n + 2017) / (1 + a (n + 1)))
  (positive_terms : ∀ n, a n > 0) :
  a 1 + a 2 = 2018 :=
sorry

end minimum_a1_a2_sum_l123_123812


namespace tan_of_right_triangle_l123_123380

theorem tan_of_right_triangle (A B C : ℝ) (h : A^2 + B^2 = C^2) (hA : A = 30) (hC : C = 37) : 
  (37^2 - 30^2).sqrt / 30 = (469).sqrt / 30 := by
  sorry

end tan_of_right_triangle_l123_123380


namespace total_yearly_interest_l123_123899

/-- Mathematical statement:
Given Nina's total inheritance of $12,000, with $5,000 invested at 6% interest and the remainder invested at 8% interest, the total yearly interest from both investments is $860.
-/
theorem total_yearly_interest (principal : ℕ) (principal_part : ℕ) (rate1 rate2 : ℚ) (interest_part1 interest_part2 : ℚ) (total_interest : ℚ) :
  principal = 12000 ∧ principal_part = 5000 ∧ rate1 = 0.06 ∧ rate2 = 0.08 ∧
  interest_part1 = (principal_part : ℚ) * rate1 ∧ interest_part2 = ((principal - principal_part) : ℚ) * rate2 →
  total_interest = interest_part1 + interest_part2 → 
  total_interest = 860 := by
  sorry

end total_yearly_interest_l123_123899


namespace sum_of_consecutive_even_integers_is_24_l123_123371

theorem sum_of_consecutive_even_integers_is_24 (x : ℕ) (h_pos : x > 0)
    (h_eq : (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2))) :
    (x - 2) + x + (x + 2) = 24 :=
sorry

end sum_of_consecutive_even_integers_is_24_l123_123371


namespace red_box_position_l123_123163

theorem red_box_position (n : ℕ) (pos_smallest_to_largest : ℕ) (pos_largest_to_smallest : ℕ) 
  (h1 : n = 45) 
  (h2 : pos_smallest_to_largest = 29) 
  (h3 : pos_largest_to_smallest = n - (pos_smallest_to_largest - 1)) :
  pos_largest_to_smallest = 17 := 
  by
    -- This proof is missing; implementation goes here
    sorry

end red_box_position_l123_123163


namespace binary_to_decimal_l123_123208

theorem binary_to_decimal : 
  (0 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4) = 18 := 
by
  -- The proof is skipped
  sorry

end binary_to_decimal_l123_123208


namespace find_f_k_l_l123_123177

noncomputable
def f : ℕ → ℕ := sorry

axiom f_condition_1 : f 1 = 1
axiom f_condition_2 : ∀ n : ℕ, 3 * f n * f (2 * n + 1) = f (2 * n) * (1 + 3 * f n)
axiom f_condition_3 : ∀ n : ℕ, f (2 * n) < 6 * f n

theorem find_f_k_l (k l : ℕ) (h : k < l) : 
  (f k + f l = 293) ↔ 
  ((k = 121 ∧ l = 4) ∨ (k = 118 ∧ l = 4) ∨ 
   (k = 109 ∧ l = 16) ∨ (k = 16 ∧ l = 109)) := 
by 
  sorry

end find_f_k_l_l123_123177


namespace calculate_l123_123975

def q (x y : ℤ) : ℤ :=
  if x > 0 ∧ y ≥ 0 then x + 2*y
  else if x < 0 ∧ y ≤ 0 then x - 3*y
  else 4*x + 2*y

theorem calculate : q (q 2 (-2)) (q (-3) 1) = -4 := 
  by
    sorry

end calculate_l123_123975


namespace price_reduction_equation_l123_123554

theorem price_reduction_equation (x : ℝ) :
  63800 * (1 - x)^2 = 3900 :=
sorry

end price_reduction_equation_l123_123554


namespace daily_earnings_c_l123_123643

theorem daily_earnings_c (A B C : ℕ) (h1 : A + B + C = 600) (h2 : A + C = 400) (h3 : B + C = 300) : C = 100 :=
sorry

end daily_earnings_c_l123_123643


namespace inscribed_quadrilateral_exists_l123_123508

theorem inscribed_quadrilateral_exists (a b c d : ℝ) (h1: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ∃ (p q : ℝ),
    p = Real.sqrt ((a * c + b * d) * (a * d + b * c) / (a * b + c * d)) ∧
    q = Real.sqrt ((a * b + c * d) * (a * d + b * c) / (a * c + b * d)) ∧
    a * c + b * d = p * q :=
by
  sorry

end inscribed_quadrilateral_exists_l123_123508


namespace statement_A_statement_B_statement_C_statement_D_l123_123097

variable (a b : ℝ)

-- Given conditions
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom condition : a + 2 * b = 2 * a * b

-- Prove the statements
theorem statement_A : a + 2 * b ≥ 4 := sorry
theorem statement_B : ¬ (a + b ≥ 4) := sorry
theorem statement_C : ¬ (a * b ≤ 2) := sorry
theorem statement_D : a^2 + 4 * b^2 ≥ 8 := sorry

end statement_A_statement_B_statement_C_statement_D_l123_123097


namespace valid_selling_price_l123_123527

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end valid_selling_price_l123_123527


namespace TotalToysIsNinetyNine_l123_123945

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end TotalToysIsNinetyNine_l123_123945


namespace num_solutions_l123_123402

theorem num_solutions (h : ∀ n : ℕ, (1 ≤ n ∧ n ≤ 455) → n^3 % 455 = 1) : 
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ (1 ≤ n ∧ n ≤ 455) ∧ n^3 % 455 = 1) ∧ s.card = 9) :=
sorry

end num_solutions_l123_123402


namespace rectangle_area_l123_123581

theorem rectangle_area (W : ℕ) (hW : W = 5) (L : ℕ) (hL : L = 4 * W) : ∃ (A : ℕ), A = L * W ∧ A = 100 := 
by
  use 100
  sorry

end rectangle_area_l123_123581


namespace range_of_a_l123_123521

def A (x : ℝ) : Prop := x^2 - 4 * x + 3 ≤ 0
def B (x : ℝ) (a : ℝ) : Prop := x^2 - a * x < x - a

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) ∧ ∃ x, ¬ (A x → B x a) ↔ 1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l123_123521


namespace abc_sum_zero_l123_123917

theorem abc_sum_zero
  (a b c : ℝ)
  (h1 : ∀ x: ℝ, (a * (c * x^2 + b * x + a)^2 + b * (c * x^2 + b * x + a) + c = x)) :
  (a + b + c = 0) :=
by
  sorry

end abc_sum_zero_l123_123917


namespace dream_miles_driven_l123_123326

theorem dream_miles_driven (x : ℕ) (h : 4 * x + 4 * (x + 200) = 4000) : x = 400 :=
by
  sorry

end dream_miles_driven_l123_123326


namespace find_prime_triple_l123_123571

def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_triple :
  ∃ (I M C : ℕ), is_prime I ∧ is_prime M ∧ is_prime C ∧ I ≤ M ∧ M ≤ C ∧ 
  I * M * C = I + M + C + 1007 ∧ (I = 2 ∧ M = 2 ∧ C = 337) :=
by
  sorry

end find_prime_triple_l123_123571


namespace neq_is_necessary_but_not_sufficient_l123_123866

theorem neq_is_necessary_but_not_sufficient (a b : ℝ) : (a ≠ b) → ¬ (∀ a b : ℝ, (a ≠ b) → (a / b + b / a > 2)) ∧ (∀ a b : ℝ, (a / b + b / a > 2) → (a ≠ b)) :=
by {
    sorry
}

end neq_is_necessary_but_not_sufficient_l123_123866


namespace first_candidate_more_gain_l123_123596

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l123_123596


namespace circle_ratio_l123_123037

theorem circle_ratio (R r : ℝ) (h₁ : R > 0) (h₂ : r > 0) 
                     (h₃ : π * R^2 - π * r^2 = 3 * π * r^2) : R = 2 * r :=
by
  sorry

end circle_ratio_l123_123037


namespace find_a_plus_b_l123_123841

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

def parallel_condition (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 - a.2 * b.1 = 0)

theorem find_a_plus_b (m : ℝ) (h_parallel: 
  parallel_condition (⟨vector_a.1 + 2 * (vector_b m).1, vector_a.2 + 2 * (vector_b m).2⟩)
                     (⟨2 * vector_a.1 - (vector_b m).1, 2 * vector_a.2 - (vector_b m).2⟩)) :
  vector_a + vector_b (-1/2) = (-3/2, 3) := 
by
  sorry

end find_a_plus_b_l123_123841


namespace minimum_people_to_save_cost_l123_123921

-- Define the costs for the two event planners.
def cost_first_planner (x : ℕ) : ℕ := 120 + 18 * x
def cost_second_planner (x : ℕ) : ℕ := 250 + 15 * x

-- State the theorem to prove the minimum number of people required for the second event planner to be less expensive.
theorem minimum_people_to_save_cost : ∃ x : ℕ, cost_second_planner x < cost_first_planner x ∧ ∀ y : ℕ, y < x → cost_second_planner y ≥ cost_first_planner y :=
sorry

end minimum_people_to_save_cost_l123_123921


namespace bond_interest_percentage_l123_123079

noncomputable def interest_percentage_of_selling_price (face_value interest_rate : ℝ) (selling_price : ℝ) : ℝ :=
  (face_value * interest_rate) / selling_price * 100

theorem bond_interest_percentage :
  let face_value : ℝ := 5000
  let interest_rate : ℝ := 0.07
  let selling_price : ℝ := 5384.615384615386
  interest_percentage_of_selling_price face_value interest_rate selling_price = 6.5 :=
by
  sorry

end bond_interest_percentage_l123_123079


namespace gcd_of_powers_of_two_l123_123850

noncomputable def m := 2^2048 - 1
noncomputable def n := 2^2035 - 1

theorem gcd_of_powers_of_two : Int.gcd m n = 8191 := by
  sorry

end gcd_of_powers_of_two_l123_123850


namespace constructible_triangle_and_area_bound_l123_123641

noncomputable def triangle_inequality_sine (α β γ : ℝ) : Prop :=
  (Real.sin α + Real.sin β > Real.sin γ) ∧
  (Real.sin β + Real.sin γ > Real.sin α) ∧
  (Real.sin γ + Real.sin α > Real.sin β)

theorem constructible_triangle_and_area_bound 
  (α β γ : ℝ) (h_pos : 0 < α) (h_pos_β : 0 < β) (h_pos_γ : 0 < γ)
  (h_sum : α + β + γ < Real.pi)
  (h_ineq1 : α + β > γ)
  (h_ineq2 : β + γ > α)
  (h_ineq3 : γ + α > β) :
  triangle_inequality_sine α β γ ∧
  (Real.sin α * Real.sin β * Real.sin γ) / 4 ≤ (1 / 8) * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) :=
sorry

end constructible_triangle_and_area_bound_l123_123641


namespace initial_amount_l123_123211

theorem initial_amount (x : ℝ) (h1 : x = (2*x - 10) / 2) (h2 : x = (4*x - 30) / 2) (h3 : 8*x - 70 = 0) : x = 8.75 :=
by
  sorry

end initial_amount_l123_123211


namespace expression_eq_neg_one_l123_123313

theorem expression_eq_neg_one (a b y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : y ≠ a) (h4 : y ≠ -a) :
  ( ( (a + b) / (a + y) + y / (a - y) ) / ( (y + b) / (a + y) - a / (a - y) ) = -1 ) ↔ ( y = a - b ) := 
sorry

end expression_eq_neg_one_l123_123313


namespace john_profit_proof_l123_123959

-- Define the conditions
variables 
  (parts_cost : ℝ := 800)
  (selling_price_multiplier : ℝ := 1.4)
  (monthly_build_quantity : ℝ := 60)
  (monthly_rent : ℝ := 5000)
  (monthly_extra_expenses : ℝ := 3000)

-- Define the computed variables based on conditions
def selling_price_per_computer := parts_cost * selling_price_multiplier
def total_revenue := monthly_build_quantity * selling_price_per_computer
def total_cost_of_components := monthly_build_quantity * parts_cost
def total_expenses := monthly_rent + monthly_extra_expenses
def profit_per_month := total_revenue - total_cost_of_components - total_expenses

-- The theorem statement of the proof
theorem john_profit_proof : profit_per_month = 11200 := 
by
  sorry

end john_profit_proof_l123_123959


namespace cone_lateral_surface_area_l123_123198

theorem cone_lateral_surface_area (r l : ℝ) (h1 : r = 2) (h2 : l = 5) : 
    0.5 * (2 * Real.pi * r * l) = 10 * Real.pi := by
    sorry

end cone_lateral_surface_area_l123_123198


namespace solve_for_y_l123_123507

theorem solve_for_y (y : ℤ) (h : (y ≠ 2) → ((y^2 - 10*y + 24)/(y-2) + (4*y^2 + 8*y - 48)/(4*y - 8) = 0)) : y = 0 :=
by
  sorry

end solve_for_y_l123_123507


namespace area_of_isosceles_trapezoid_l123_123861

def isIsoscelesTrapezoid (a b c h : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2

theorem area_of_isosceles_trapezoid :
  ∀ (a b c : ℝ), 
    a = 8 → b = 14 → c = 5 →
    ∃ h: ℝ, isIsoscelesTrapezoid a b c h ∧ ((a + b) / 2 * h = 44) :=
by
  intros a b c ha hb hc
  sorry

end area_of_isosceles_trapezoid_l123_123861


namespace total_clips_correct_l123_123070

def clips_in_april : ℕ := 48
def clips_in_may : ℕ := clips_in_april / 2
def total_clips : ℕ := clips_in_april + clips_in_may

theorem total_clips_correct : total_clips = 72 := by
  sorry

end total_clips_correct_l123_123070


namespace number_exceeds_80_by_120_l123_123962

theorem number_exceeds_80_by_120 : ∃ x : ℝ, x = 0.80 * x + 120 ∧ x = 600 :=
by sorry

end number_exceeds_80_by_120_l123_123962


namespace max_gcd_of_sequence_l123_123020

/-- Define the sequence as a function. -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- Define the greatest common divisor of the sequence terms. -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- State the theorem of the maximum value of d. -/
theorem max_gcd_of_sequence : ∃ n : ℕ, d n = 401 := sorry

end max_gcd_of_sequence_l123_123020


namespace car_count_is_150_l123_123985

variable (B C K : ℕ)  -- Define the variables representing buses, cars, and bikes

/-- Given conditions: The ratio of buses to cars to bikes is 3:7:10,
    there are 90 fewer buses than cars, and 140 fewer buses than bikes. -/
def conditions : Prop :=
  (C = (7 * B / 3)) ∧ (K = (10 * B / 3)) ∧ (C = B + 90) ∧ (K = B + 140)

theorem car_count_is_150 (h : conditions B C K) : C = 150 :=
by
  sorry

end car_count_is_150_l123_123985


namespace sqrt_47_minus_2_range_l123_123844

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end sqrt_47_minus_2_range_l123_123844


namespace power_sum_l123_123360

theorem power_sum : 1 ^ 2009 + (-1) ^ 2009 = 0 := 
by 
  sorry

end power_sum_l123_123360


namespace first_digit_base_8_of_725_is_1_l123_123409

-- Define conditions
def decimal_val := 725

-- Helper function to get the largest power of 8 less than the decimal value
def largest_power_base_eight (n : ℕ) : ℕ :=
  if 8^3 <= n then 8^3 else if 8^2 <= n then 8^2 else if 8^1 <= n then 8^1 else if 8^0 <= n then 8^0 else 0

-- The target theorem
theorem first_digit_base_8_of_725_is_1 : 
  (725 / largest_power_base_eight 725) = 1 :=
by 
  -- Proof goes here
  sorry

end first_digit_base_8_of_725_is_1_l123_123409


namespace largest_possible_value_of_n_l123_123614

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def largest_product : ℕ :=
  705

theorem largest_possible_value_of_n :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧
  is_prime x ∧ is_prime y ∧
  is_prime (10 * y - x) ∧
  largest_product = x * y * (10 * y - x) :=
by
  sorry

end largest_possible_value_of_n_l123_123614


namespace total_lives_l123_123512

noncomputable def C : ℝ := 9.5
noncomputable def D : ℝ := C - 3.25
noncomputable def M : ℝ := D + 7.75
noncomputable def E : ℝ := 2 * C - 5.5
noncomputable def F : ℝ := 2/3 * E

theorem total_lives : C + D + M + E + F = 52.25 :=
by
  sorry

end total_lives_l123_123512


namespace flowers_not_roses_percentage_l123_123658

def percentage_non_roses (roses tulips daisies : Nat) : Nat :=
  let total := roses + tulips + daisies
  let non_roses := total - roses
  (non_roses * 100) / total

theorem flowers_not_roses_percentage :
  percentage_non_roses 25 40 35 = 75 :=
by
  sorry

end flowers_not_roses_percentage_l123_123658


namespace field_division_l123_123685

theorem field_division
  (total_area : ℕ)
  (part_area : ℕ)
  (diff : ℕ → ℕ)
  (X : ℕ)
  (h_total : total_area = 900)
  (h_part : part_area = 405)
  (h_diff : diff (total_area - part_area - part_area) = (1 / 5 : ℚ) * X)
  : X = 450 := 
sorry

end field_division_l123_123685


namespace sam_current_dimes_l123_123859

def original_dimes : ℕ := 8
def sister_borrowed : ℕ := 4
def friend_borrowed : ℕ := 2
def sister_returned : ℕ := 2
def friend_returned : ℕ := 1

theorem sam_current_dimes : 
  (original_dimes - sister_borrowed - friend_borrowed + sister_returned + friend_returned = 5) :=
by
  sorry

end sam_current_dimes_l123_123859


namespace final_amount_after_bets_l123_123840

theorem final_amount_after_bets :
  let initial_amount := 128
  let num_bets := 8
  let num_wins := 4
  let num_losses := 4
  let bonus_per_win_after_loss := 10
  let win_multiplier := 3 / 2
  let loss_multiplier := 1 / 2
  ∃ final_amount : ℝ,
    (final_amount =
      initial_amount * (win_multiplier ^ num_wins) * (loss_multiplier ^ num_losses) + 2 * bonus_per_win_after_loss) ∧
    final_amount = 60.5 :=
sorry

end final_amount_after_bets_l123_123840


namespace linear_function_points_relation_l123_123184

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ), 
  (y1 = -3 * 2 + 1) ∧ (y2 = -3 * 3 + 1) → y1 > y2 :=
by
  intro y1 y2
  intro h
  cases h
  sorry

end linear_function_points_relation_l123_123184


namespace thomas_total_drawings_l123_123280

theorem thomas_total_drawings :
  let colored_pencil_drawings := 14
  let blending_marker_drawings := 7
  let charcoal_drawings := 4
  colored_pencil_drawings + blending_marker_drawings + charcoal_drawings = 25 := 
by
  sorry

end thomas_total_drawings_l123_123280


namespace glucose_in_mixed_solution_l123_123916

def concentration1 := 20 / 100  -- concentration of first solution in grams per cubic centimeter
def concentration2 := 30 / 100  -- concentration of second solution in grams per cubic centimeter
def volume1 := 80               -- volume of first solution in cubic centimeters
def volume2 := 50               -- volume of second solution in cubic centimeters

theorem glucose_in_mixed_solution :
  (concentration1 * volume1) + (concentration2 * volume2) = 31 := by
  sorry

end glucose_in_mixed_solution_l123_123916


namespace extremum_points_of_f_l123_123240

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^3 * Real.exp (x + 1) - Real.exp 1
  else -((if -x < 0 then (-x + 1)^3 * Real.exp (-x + 1) - Real.exp 1 else 0))

theorem extremum_points_of_f : ∃! (a b : ℝ), 
  (∀ x < 0, f x = (x + 1)^3 * Real.exp (x + 1) - Real.exp 1) ∧ (f a = f b) ∧ a ≠ b :=
sorry

end extremum_points_of_f_l123_123240


namespace cos_alpha_value_l123_123585

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) :
  Real.cos α = 4 / 5 :=
by
  sorry

end cos_alpha_value_l123_123585


namespace q_value_l123_123411

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end q_value_l123_123411


namespace max_n_for_polynomial_l123_123706

theorem max_n_for_polynomial (P : Polynomial ℤ) (hdeg : P.degree = 2022) :
  ∃ n ≤ 2022, ∀ {a : Fin n → ℤ}, 
    (∀ i, P.eval (a i) = i) ↔ n = 2022 :=
by sorry

end max_n_for_polynomial_l123_123706


namespace number_of_green_hats_l123_123261

theorem number_of_green_hats (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) : 
  G = 40 := by
  sorry

end number_of_green_hats_l123_123261


namespace prob_both_hit_prob_at_least_one_hits_l123_123213

variable (pA pB : ℝ)

-- Given conditions
def prob_A_hits : Prop := pA = 0.9
def prob_B_hits : Prop := pB = 0.8

-- Proof problems
theorem prob_both_hit (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  pA * pB = 0.72 := 
  sorry

theorem prob_at_least_one_hits (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  1 - (1 - pA) * (1 - pB) = 0.98 := 
  sorry

end prob_both_hit_prob_at_least_one_hits_l123_123213


namespace dolls_completion_time_l123_123726

def time_to_complete_dolls (craft_time_per_doll break_time_per_three_dolls total_dolls start_time : Nat) : Nat :=
  let total_craft_time := craft_time_per_doll * total_dolls
  let total_breaks := (total_dolls / 3) * break_time_per_three_dolls
  let total_time := total_craft_time + total_breaks
  (start_time + total_time) % 1440 -- 1440 is the number of minutes in a day

theorem dolls_completion_time :
  time_to_complete_dolls 105 30 10 600 = 300 := -- 600 is 10:00 AM in minutes, 300 is 5:00 AM in minutes
sorry

end dolls_completion_time_l123_123726


namespace cost_of_superman_game_l123_123927

-- Define the costs as constants
def cost_batman_game : ℝ := 13.60
def total_amount_spent : ℝ := 18.66

-- Define the theorem to prove the cost of the Superman game
theorem cost_of_superman_game : total_amount_spent - cost_batman_game = 5.06 :=
by
  sorry

end cost_of_superman_game_l123_123927


namespace total_number_of_outfits_l123_123265

-- Definitions of the conditions as functions/values
def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_ties_options : Nat := 4 + 1  -- 4 ties + 1 option for no tie
def num_belts_options : Nat := 2 + 1  -- 2 belts + 1 option for no belt

-- Lean statement to formulate the proof problem
theorem total_number_of_outfits : 
  num_shirts * num_pants * num_ties_options * num_belts_options = 600 := by
  sorry

end total_number_of_outfits_l123_123265


namespace prove_m_set_l123_123646

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}

-- Define set B as dependent on m
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}

-- The main proof statement
theorem prove_m_set : {m : ℝ | B m ∩ A = B m} = {0, 1, 2} :=
by
  -- Code here would prove the above theorem
  sorry

end prove_m_set_l123_123646


namespace solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l123_123282

variable (a x : ℝ)

def inequality (a x : ℝ) : Prop := (1 - a * x) ^ 2 < 1

theorem solve_inequality_zero : a = 0 → ¬∃ x, inequality a x := by
  sorry

theorem solve_inequality_neg (h : a < 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (2 / a < x ∧ x < 0)) := by
  sorry

theorem solve_inequality_pos (h : a > 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (0 < x ∧ x < 2 / a)) := by
  sorry

end solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l123_123282


namespace twenty_seven_divides_sum_l123_123149

theorem twenty_seven_divides_sum (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) : 27 ∣ x + y + z := sorry

end twenty_seven_divides_sum_l123_123149


namespace remainder_when_M_divided_by_32_l123_123244

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l123_123244


namespace cubic_roots_nature_l123_123317

-- Define the cubic polynomial function
def cubic_poly (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x - 4

-- Define the statement about the roots of the polynomial
theorem cubic_roots_nature :
  ∃ a b c : ℝ, cubic_poly a = 0 ∧ cubic_poly b = 0 ∧ cubic_poly c = 0 
  ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end cubic_roots_nature_l123_123317


namespace twice_x_minus_3_gt_4_l123_123717

theorem twice_x_minus_3_gt_4 (x : ℝ) : 2 * x - 3 > 4 :=
sorry

end twice_x_minus_3_gt_4_l123_123717


namespace simplify_expr1_l123_123773

theorem simplify_expr1 (m n : ℝ) :
  (2 * m + n) ^ 2 - (4 * m + 3 * n) * (m - n) = 8 * m * n + 4 * n ^ 2 := by
  sorry

end simplify_expr1_l123_123773


namespace binary_arithmetic_correct_l123_123732

theorem binary_arithmetic_correct :
  (2^3 + 2^2 + 2^0) + (2^2 + 2^1 + 2^0) - (2^3 + 2^2 + 2^1) + (2^3 + 2^0) + (2^3 + 2^1) = 2^4 + 2^3 + 2^0 :=
by sorry

end binary_arithmetic_correct_l123_123732


namespace lcm_factor_of_hcf_and_larger_number_l123_123201

theorem lcm_factor_of_hcf_and_larger_number (A B : ℕ) (hcf : ℕ) (hlarger : A = 450) (hhcf : hcf = 30) (hwrel : A % hcf = 0) : ∃ x y, x = 15 ∧ (A * B = hcf * x * y) :=
by
  sorry

end lcm_factor_of_hcf_and_larger_number_l123_123201


namespace distinguishable_large_equilateral_triangles_l123_123404

-- Definitions based on conditions.
def num_colors : ℕ := 8

def same_color_corners : ℕ := num_colors
def two_same_one_diff_colors : ℕ := num_colors * (num_colors - 1)
def all_diff_colors : ℕ := (num_colors * (num_colors - 1) * (num_colors - 2)) / 6

def corner_configurations : ℕ := same_color_corners + two_same_one_diff_colors + all_diff_colors
def triangle_between_center_and_corner : ℕ := num_colors
def center_triangle : ℕ := num_colors

def total_distinguishable_triangles : ℕ := corner_configurations * triangle_between_center_and_corner * center_triangle

theorem distinguishable_large_equilateral_triangles : total_distinguishable_triangles = 7680 :=
by
  sorry

end distinguishable_large_equilateral_triangles_l123_123404


namespace largest_r_satisfying_condition_l123_123860

theorem largest_r_satisfying_condition :
  ∃ M : ℕ, ∀ (a : ℕ → ℕ) (r : ℝ) (h : ∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))),
  (∀ n : ℕ, n ≥ M → a (n + 2) = a n) → r = 2 := 
by
  sorry

end largest_r_satisfying_condition_l123_123860


namespace min_purchase_amount_is_18_l123_123597

def burger_cost := 2 * 3.20
def fries_cost := 2 * 1.90
def milkshake_cost := 2 * 2.40
def current_total := burger_cost + fries_cost + milkshake_cost
def additional_needed := 3.00
def min_purchase_amount_for_free_delivery := current_total + additional_needed

theorem min_purchase_amount_is_18 : min_purchase_amount_for_free_delivery = 18 := by
  sorry

end min_purchase_amount_is_18_l123_123597


namespace marie_erasers_l123_123873

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) :
  initial_erasers = 95 → lost_erasers = 42 → final_erasers = initial_erasers - lost_erasers → final_erasers = 53 :=
by
  intros h_initial h_lost h_final
  rw [h_initial, h_lost] at h_final
  exact h_final

end marie_erasers_l123_123873


namespace gmat_test_takers_correctly_l123_123797

variable (A B : ℝ)
variable (intersection union : ℝ)

theorem gmat_test_takers_correctly :
  B = 0.8 ∧ intersection = 0.7 ∧ union = 0.95 → A = 0.85 :=
by 
  sorry

end gmat_test_takers_correctly_l123_123797


namespace attendance_second_day_l123_123624

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end attendance_second_day_l123_123624


namespace difference_of_fractions_l123_123150

theorem difference_of_fractions (a b c : ℝ) (h1 : a = 8000 * (1/2000)) (h2 : b = 8000 * (1/10)) (h3 : c = b - a) : c = 796 := 
sorry

end difference_of_fractions_l123_123150


namespace find_c_for_same_solution_l123_123692

theorem find_c_for_same_solution (c : ℝ) (x : ℝ) :
  (3 * x + 5 = 1) ∧ (c * x + 15 = -5) → c = 15 :=
by
  sorry

end find_c_for_same_solution_l123_123692


namespace problem_statement_l123_123883

variable (a b c d : ℝ)

noncomputable def circle_condition_1 : Prop := a = (1 : ℝ) / a
noncomputable def circle_condition_2 : Prop := b = (1 : ℝ) / b
noncomputable def circle_condition_3 : Prop := c = (1 : ℝ) / c
noncomputable def circle_condition_4 : Prop := d = (1 : ℝ) / d

theorem problem_statement (h1 : circle_condition_1 a)
                          (h2 : circle_condition_2 b)
                          (h3 : circle_condition_3 c)
                          (h4 : circle_condition_4 d) :
    2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := 
by
  sorry

end problem_statement_l123_123883


namespace calculation_identity_l123_123809

theorem calculation_identity :
  (3.14 - 1)^0 * (-1 / 4)^(-2) = 16 := by
  sorry

end calculation_identity_l123_123809


namespace range_x_plus_y_l123_123719

theorem range_x_plus_y (x y: ℝ) (h: x^2 + y^2 - 4 * x + 3 = 0) : 
  2 - Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + Real.sqrt 2 :=
by 
  sorry

end range_x_plus_y_l123_123719


namespace minimum_cost_is_8600_l123_123965

-- Defining the conditions
def shanghai_units : ℕ := 12
def nanjing_units : ℕ := 6
def suzhou_needs : ℕ := 10
def changsha_needs : ℕ := 8
def cost_shanghai_suzhou : ℕ := 400
def cost_shanghai_changsha : ℕ := 800
def cost_nanjing_suzhou : ℕ := 300
def cost_nanjing_changsha : ℕ := 500

-- Defining the function for total shipping cost
def total_shipping_cost (x : ℕ) : ℕ :=
  cost_shanghai_suzhou * x +
  cost_shanghai_changsha * (shanghai_units - x) +
  cost_nanjing_suzhou * (suzhou_needs - x) +
  cost_nanjing_changsha * (x - (shanghai_units - suzhou_needs))

-- Define the minimum shipping cost function
def minimum_shipping_cost : ℕ :=
  total_shipping_cost 10

-- State the theorem to prove
theorem minimum_cost_is_8600 : minimum_shipping_cost = 8600 :=
sorry

end minimum_cost_is_8600_l123_123965


namespace min_value_m_plus_n_l123_123417

theorem min_value_m_plus_n (m n : ℕ) (h : 108 * m = n^3) (hm : 0 < m) (hn : 0 < n) : m + n = 8 :=
sorry

end min_value_m_plus_n_l123_123417


namespace cos_A_equals_one_third_l123_123969

-- Noncomputable context as trigonometric functions are involved.
noncomputable def cosA_in_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  let law_of_cosines : (a * Real.cos B) = (3 * c - b) * Real.cos A := sorry
  (Real.cos A = 1 / 3)

-- Define the problem statement to be proved
theorem cos_A_equals_one_third (a b c A B C : ℝ) 
  (h1 : a = Real.cos B)
  (h2 : a * Real.cos B = (3 * c - b) * Real.cos A) :
  Real.cos A = 1 / 3 := 
by 
  -- Placeholder for the actual proof
  sorry

end cos_A_equals_one_third_l123_123969


namespace x5_y5_z5_value_is_83_l123_123709

noncomputable def find_x5_y5_z5_value (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧ 
  (x^3 + y^3 + z^3 = 15) ∧
  (x^4 + y^4 + z^4 = 35) ∧
  (x^2 + y^2 + z^2 < 10) →
  x^5 + y^5 + z^5 = 83

theorem x5_y5_z5_value_is_83 (x y z : ℝ) :
  find_x5_y5_z5_value x y z :=
  sorry

end x5_y5_z5_value_is_83_l123_123709


namespace orange_balls_count_l123_123928

theorem orange_balls_count (total_balls red_balls blue_balls yellow_balls green_balls pink_balls orange_balls : ℕ) 
(h_total : total_balls = 100)
(h_red : red_balls = 30)
(h_blue : blue_balls = 20)
(h_yellow : yellow_balls = 10)
(h_green : green_balls = 5)
(h_pink : pink_balls = 2 * green_balls)
(h_orange : orange_balls = 3 * pink_balls)
(h_sum : red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls) :
orange_balls = 30 :=
sorry

end orange_balls_count_l123_123928


namespace compare_binary_digits_l123_123820

def numDigits_base2 (n : ℕ) : ℕ :=
  (Nat.log2 n) + 1

theorem compare_binary_digits :
  numDigits_base2 1600 - numDigits_base2 400 = 2 := by
  sorry

end compare_binary_digits_l123_123820


namespace right_triangle_hypotenuse_length_l123_123904

theorem right_triangle_hypotenuse_length 
    (AB AC x y : ℝ) 
    (P : AB = x) (Q : AC = y) 
    (ratio_AP_PB : AP / PB = 1 / 3) 
    (ratio_AQ_QC : AQ / QC = 2 / 1) 
    (BQ_length : BQ = 18) 
    (CP_length : CP = 24) : 
    BC = 24 := 
by 
  sorry

end right_triangle_hypotenuse_length_l123_123904


namespace binary_to_decimal_conversion_l123_123565

theorem binary_to_decimal_conversion : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) := by
  sorry

end binary_to_decimal_conversion_l123_123565


namespace correct_option_D_l123_123446

def U : Set ℕ := {1, 2, 4, 6, 8}
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

theorem correct_option_D : A ∩ complement_U_B = {1} := by
  sorry

end correct_option_D_l123_123446


namespace highest_weekly_sales_is_60_l123_123087

/-- 
Given that a convenience store sold 300 bags of chips in a month,
and the following weekly sales pattern:
1. In the first week, 20 bags were sold.
2. In the second week, there was a 2-for-1 promotion, tripling the sales to 60 bags.
3. In the third week, a 10% discount doubled the sales to 40 bags.
4. In the fourth week, sales returned to the first week's number, 20 bags.
Prove that the number of bags of chips sold during the week with the highest demand is 60.
-/
theorem highest_weekly_sales_is_60 
  (total_sales : ℕ)
  (week1_sales : ℕ)
  (week2_sales : ℕ)
  (week3_sales : ℕ)
  (week4_sales : ℕ)
  (h_total : total_sales = 300)
  (h_week1 : week1_sales = 20)
  (h_week2 : week2_sales = 3 * week1_sales)
  (h_week3 : week3_sales = 2 * week1_sales)
  (h_week4 : week4_sales = week1_sales) :
  max (max week1_sales week2_sales) (max week3_sales week4_sales) = 60 := 
sorry

end highest_weekly_sales_is_60_l123_123087


namespace tens_digit_of_6_pow_19_l123_123784

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tens_digit_of_6_pow_19 : tens_digit (6 ^ 19) = 9 := 
by 
  sorry

end tens_digit_of_6_pow_19_l123_123784


namespace emails_difference_l123_123939

def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

theorem emails_difference :
  afternoon_emails - morning_emails = 2 := 
by
  sorry

end emails_difference_l123_123939


namespace min_value_expression_l123_123218

open Real

/-- The minimum value of (14 - x) * (8 - x) * (14 + x) * (8 + x) is -4356. -/
theorem min_value_expression (x : ℝ) : ∃ (a : ℝ), a = (14 - x) * (8 - x) * (14 + x) * (8 + x) ∧ a ≥ -4356 :=
by
  use -4356
  sorry

end min_value_expression_l123_123218


namespace mark_money_l123_123269

theorem mark_money (M : ℝ) 
  (h1 : (1 / 2) * M + 14 + (1 / 3) * M + 16 + (1 / 4) * M + 18 = M) : 
  M = 576 := 
sorry

end mark_money_l123_123269


namespace neg_p_l123_123544

open Nat -- Opening natural number namespace

-- Definition of the proposition p
def p := ∃ (m : ℕ), ∃ (k : ℕ), k * k = m * m + 1

-- Theorem statement for the negation of proposition p
theorem neg_p : ¬p ↔ ∀ (m : ℕ), ¬ ∃ (k : ℕ), k * k = m * m + 1 :=
by {
  -- Provide the proof here
  sorry
}

end neg_p_l123_123544


namespace simplify_and_evaluate_expression_l123_123254

theorem simplify_and_evaluate_expression (a : ℤ) (ha : a = -2) : 
  (1 + 1 / (a - 1)) / ((2 * a) / (a ^ 2 - 1)) = -1 / 2 := by
  sorry

end simplify_and_evaluate_expression_l123_123254


namespace sets_equal_l123_123847

theorem sets_equal :
  {u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l} =
  {u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r} := 
sorry

end sets_equal_l123_123847


namespace fraction_spent_on_furniture_l123_123814

theorem fraction_spent_on_furniture (original_savings : ℝ) (cost_of_tv : ℝ) (f : ℝ)
  (h1 : original_savings = 1800) 
  (h2 : cost_of_tv = 450) 
  (h3 : f * original_savings + cost_of_tv = original_savings) :
  f = 3 / 4 := 
by 
  sorry

end fraction_spent_on_furniture_l123_123814


namespace sum_of_divisors_45_l123_123334

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun i => n % i = 0) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_45 : sum_of_divisors 45 = 78 := 
  sorry

end sum_of_divisors_45_l123_123334


namespace original_number_l123_123767

theorem original_number (N : ℕ) :
  (∃ k m n : ℕ, N - 6 = 5 * k + 3 ∧ N - 6 = 11 * m + 3 ∧ N - 6 = 13 * n + 3) → N = 724 :=
by
  sorry

end original_number_l123_123767


namespace find_angle_C_find_triangle_area_l123_123283

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) 
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : B + C + A = Real.pi) :
  C = Real.pi / 12 :=
by
  sorry

theorem find_triangle_area (A B C : ℝ) (a b c : ℝ)
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : b^2 + c^2 = a - b * c + 2) 
  (h4 : B + C + A = Real.pi) 
  (h5 : a^2 = b^2 + c^2 + b * c) :
  (1/2) * a * b * Real.sin C = 1 - Real.sqrt 3 / 3 :=
by
  sorry

end find_angle_C_find_triangle_area_l123_123283


namespace reconstruct_right_triangle_l123_123271

theorem reconstruct_right_triangle (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  ∃ A X Y: ℝ, (A ≠ X ∧ A ≠ Y ∧ X ≠ Y) ∧ 
  -- Right triangle with hypotenuse c
  (A - X) ^ 2 + (Y - X) ^ 2 = c ^ 2 ∧ 
  -- Difference of legs is d
  ∃ AY XY: ℝ, ((AY = abs (A - Y)) ∧ (XY = abs (Y - X)) ∧ (abs (AY - XY) = d)) := 
by
  sorry

end reconstruct_right_triangle_l123_123271


namespace three_digit_number_parity_count_equal_l123_123013

/-- Prove the number of three-digit numbers with all digits having the same parity is equal to the number of three-digit numbers where adjacent digits have different parity. -/
theorem three_digit_number_parity_count_equal :
  ∃ (same_parity_count alternating_parity_count : ℕ),
    same_parity_count = alternating_parity_count ∧
    -- Condition for digits of the same parity
    same_parity_count = (4 * 5 * 5) + (5 * 5 * 5) ∧
    -- Condition for alternating parity digits (patterns EOE and OEO)
    alternating_parity_count = (4 * 5 * 5) + (5 * 5 * 5) := by
  sorry

end three_digit_number_parity_count_equal_l123_123013


namespace complementary_angle_beta_l123_123292

theorem complementary_angle_beta (α β : ℝ) (h_compl : α + β = 90) (h_alpha : α = 40) : β = 50 :=
by
  -- Skipping the proof, which initial assumption should be defined.
  sorry

end complementary_angle_beta_l123_123292


namespace additional_charge_per_segment_l123_123262

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end additional_charge_per_segment_l123_123262


namespace sum_b_div_5_pow_eq_l123_123589

namespace SequenceSumProblem

-- Define the sequence b_n
def b : ℕ → ℝ
| 0       => 2
| 1       => 3
| (n + 2) => b (n + 1) + b n

-- The infinite series sum we need to prove
noncomputable def sum_b_div_5_pow (Y : ℝ) : Prop :=
  Y = ∑' n : ℕ, (b n) / (5 ^ (n + 1))

-- The statement of the problem
theorem sum_b_div_5_pow_eq : sum_b_div_5_pow (2 / 25) :=
sorry

end SequenceSumProblem

end sum_b_div_5_pow_eq_l123_123589


namespace common_tangent_y_intercept_l123_123438

theorem common_tangent_y_intercept
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (m b : ℝ)
  (h_c1 : c1 = (5, -2))
  (h_c2 : c2 = (20, 6))
  (h_r1 : r1 = 5)
  (h_r2 : r2 = 12)
  (h_tangent : ∃m > 0, ∃b, (∀ x y, y = m * x + b → (x - 5)^2 + (y + 2)^2 > 25 ∧ (x - 20)^2 + (y - 6)^2 > 144)) :
  b = -2100 / 161 :=
by
  sorry

end common_tangent_y_intercept_l123_123438


namespace expression_value_l123_123690

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 :=
by
  sorry

end expression_value_l123_123690


namespace remainder_of_2_pow_33_mod_9_l123_123660

theorem remainder_of_2_pow_33_mod_9 : (2 ^ 33) % 9 = 8 :=
by
  sorry

end remainder_of_2_pow_33_mod_9_l123_123660


namespace cannot_achieve_55_cents_with_six_coins_l123_123332

theorem cannot_achieve_55_cents_with_six_coins :
  ¬∃ (a b c d e : ℕ), 
    a + b + c + d + e = 6 ∧ 
    a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 55 := 
sorry

end cannot_achieve_55_cents_with_six_coins_l123_123332


namespace candy_amount_in_peanut_butter_jar_l123_123420

-- Definitions of the candy amounts in each jar
def banana_jar := 43
def grape_jar := banana_jar + 5
def peanut_butter_jar := 4 * grape_jar
def coconut_jar := (3 / 2) * banana_jar

-- The statement we need to prove
theorem candy_amount_in_peanut_butter_jar : peanut_butter_jar = 192 := by
  sorry

end candy_amount_in_peanut_butter_jar_l123_123420


namespace ways_to_get_off_the_bus_l123_123780

-- Define the number of passengers and stops
def numPassengers : ℕ := 10
def numStops : ℕ := 5

-- Define the theorem that states the number of ways for passengers to get off
theorem ways_to_get_off_the_bus : (numStops^numPassengers) = 5^10 :=
by sorry

end ways_to_get_off_the_bus_l123_123780


namespace gray_region_area_l123_123145

noncomputable def area_gray_region : ℝ :=
  let area_rectangle := (12 - 4) * (12 - 4)
  let radius_c := 4
  let radius_d := 4
  let area_quarter_circle_c := 1/4 * Real.pi * radius_c^2
  let area_quarter_circle_d := 1/4 * Real.pi * radius_d^2
  let overlap_area := area_quarter_circle_c + area_quarter_circle_d
  area_rectangle - overlap_area

theorem gray_region_area :
  area_gray_region = 64 - 8 * Real.pi := by
  sorry

end gray_region_area_l123_123145


namespace weight_of_replaced_person_l123_123179

variable (average_weight_increase : ℝ)
variable (num_persons : ℝ)
variable (weight_new_person : ℝ)

theorem weight_of_replaced_person 
    (h1 : average_weight_increase = 2.5) 
    (h2 : num_persons = 10) 
    (h3 : weight_new_person = 90)
    : ∃ weight_replaced : ℝ, weight_replaced = 65 := 
by
  sorry

end weight_of_replaced_person_l123_123179


namespace range_of_a_l123_123258

open Real

theorem range_of_a (x y z a : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1)
  (heq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by
  sorry

end range_of_a_l123_123258


namespace eq1_solution_eq2_solution_eq3_solution_eq4_solution_l123_123110

-- Equation 1: 3x^2 - 2x - 1 = 0
theorem eq1_solution (x : ℝ) : 3 * x ^ 2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) :=
by sorry

-- Equation 2: (y + 1)^2 - 4 = 0
theorem eq2_solution (y : ℝ) : (y + 1) ^ 2 - 4 = 0 ↔ (y = 1 ∨ y = -3) :=
by sorry

-- Equation 3: t^2 - 6t - 7 = 0
theorem eq3_solution (t : ℝ) : t ^ 2 - 6 * t - 7 = 0 ↔ (t = 7 ∨ t = -1) :=
by sorry

-- Equation 4: m(m + 3) - 2m = 0
theorem eq4_solution (m : ℝ) : m * (m + 3) - 2 * m = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

end eq1_solution_eq2_solution_eq3_solution_eq4_solution_l123_123110


namespace condition_necessary_but_not_sufficient_l123_123639

theorem condition_necessary_but_not_sufficient (a : ℝ) :
  ((1 / a > 1) → (a < 1)) ∧ (∃ (a : ℝ), a < 1 ∧ 1 / a < 1) :=
by
  sorry

end condition_necessary_but_not_sufficient_l123_123639


namespace Nancy_seeds_l123_123214

def big_garden_seeds : ℕ := 28
def small_gardens : ℕ := 6
def seeds_per_small_garden : ℕ := 4

def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem Nancy_seeds : total_seeds = 52 :=
by
  -- Proof here...
  sorry

end Nancy_seeds_l123_123214


namespace simplify_fraction_sum_eq_zero_l123_123306

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)
variable (h : a + b + 2 * c = 0)

theorem simplify_fraction_sum_eq_zero :
  (1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2)) = 0 :=
by sorry

end simplify_fraction_sum_eq_zero_l123_123306


namespace cubic_geometric_progression_l123_123418

theorem cubic_geometric_progression (a b c : ℝ) (α β γ : ℝ) 
    (h_eq1 : α + β + γ = -a) 
    (h_eq2 : α * β + α * γ + β * γ = b) 
    (h_eq3 : α * β * γ = -c) 
    (h_gp : ∃ k q : ℝ, α = k / q ∧ β = k ∧ γ = k * q) : 
    a^3 * c - b^3 = 0 :=
by
  sorry

end cubic_geometric_progression_l123_123418


namespace min_class_size_l123_123506

theorem min_class_size (x : ℕ) (h : 50 ≤ 5 * x + 2) : 52 ≤ 5 * x + 2 :=
by
  sorry

end min_class_size_l123_123506


namespace find_x_l123_123229

theorem find_x (x : ℝ) (h : (1 + x) / (5 + x) = 1 / 3) : x = 1 :=
sorry

end find_x_l123_123229


namespace cindy_gave_25_pens_l123_123950

theorem cindy_gave_25_pens (initial_pens mike_gave pens_given_sharon final_pens : ℕ) (h1 : initial_pens = 5) (h2 : mike_gave = 20) (h3 : pens_given_sharon = 19) (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gave - pens_given_sharon + 25 :=
by 
  -- Insert the proof here later
  sorry

end cindy_gave_25_pens_l123_123950


namespace closest_point_on_ellipse_to_line_l123_123176

theorem closest_point_on_ellipse_to_line :
  ∃ (x y : ℝ), 
    7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0 ∧ (x, y) = (3 / 2, -7 / 4) :=
by
  sorry

end closest_point_on_ellipse_to_line_l123_123176


namespace complex_quadrant_l123_123698

-- Declare the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Declare the complex number z as per the condition
noncomputable def z : ℂ := (2 * i) / (i - 1)

-- State and prove that the complex number z lies in the fourth quadrant
theorem complex_quadrant : (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_quadrant_l123_123698


namespace number_of_beavers_l123_123680

-- Definitions of the problem conditions
def total_workers : Nat := 862
def number_of_spiders : Nat := 544

-- The statement we need to prove
theorem number_of_beavers : (total_workers - number_of_spiders) = 318 := 
by 
  sorry

end number_of_beavers_l123_123680


namespace permute_rows_to_columns_l123_123803

open Function

-- Define the problem
theorem permute_rows_to_columns {α : Type*} [Fintype α] [DecidableEq α] (n : ℕ)
  (table : Fin n → Fin n → α)
  (h_distinct_rows : ∀ i : Fin n, ∀ j₁ j₂ : Fin n, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) :
  ∃ (p : Fin n → Fin n → Fin n), ∀ j : Fin n, ∀ i₁ i₂ : Fin n, i₁ ≠ i₂ →
    table i₁ (p i₁ j) ≠ table i₂ (p i₂ j) := 
sorry

end permute_rows_to_columns_l123_123803


namespace population_change_over_3_years_l123_123304

-- Define the initial conditions
def annual_growth_rate := 0.09
def migration_rate_year1 := -0.01
def migration_rate_year2 := -0.015
def migration_rate_year3 := -0.02
def natural_disaster_rate := -0.03

-- Lemma stating the overall percentage increase in population over three years
theorem population_change_over_3_years :
  (1 + annual_growth_rate) * (1 + migration_rate_year1) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year2) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year3) * 
  (1 + natural_disaster_rate) = 1.195795 := 
sorry

end population_change_over_3_years_l123_123304


namespace machine_work_time_today_l123_123588

theorem machine_work_time_today :
  let shirts_today := 40
  let pants_today := 50
  let shirt_rate := 5
  let pant_rate := 3
  let time_for_shirts := shirts_today / shirt_rate
  let time_for_pants := pants_today / pant_rate
  time_for_shirts + time_for_pants = 24.67 :=
by
  sorry

end machine_work_time_today_l123_123588


namespace vector_parallel_l123_123410

theorem vector_parallel (x : ℝ) : ∃ x, (1, x) = k * (-2, 3) → x = -3 / 2 :=
by 
  sorry

end vector_parallel_l123_123410


namespace flour_amount_indeterminable_l123_123559

variable (flour_required : ℕ)
variable (sugar_required : ℕ := 11)
variable (sugar_added : ℕ := 10)
variable (flour_added : ℕ := 12)
variable (sugar_to_add : ℕ := 1)

theorem flour_amount_indeterminable :
  ¬ ∃ (flour_required : ℕ), flour_additional = flour_required - flour_added :=
by
  sorry

end flour_amount_indeterminable_l123_123559


namespace correct_propositions_l123_123570

-- Definitions of relations between lines and planes
variable {Line : Type}
variable {Plane : Type}

-- Definition of relationships
variable (parallel_lines : Line → Line → Prop)
variable (parallel_plane_with_plane : Plane → Plane → Prop)
variable (parallel_line_with_plane : Line → Plane → Prop)
variable (perpendicular_plane_with_plane : Plane → Plane → Prop)
variable (perpendicular_line_with_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)
variable (distinct_planes : Plane → Plane → Prop)

-- The main theorem we are proving with the given conditions
theorem correct_propositions (m n : Line) (α β γ : Plane)
  (hmn : distinct_lines m n) (hαβ : distinct_planes α β) (hαγ : distinct_planes α γ)
  (hβγ : distinct_planes β γ) :
  -- Statement 1
  (parallel_plane_with_plane α β → parallel_plane_with_plane α γ → parallel_plane_with_plane β γ) ∧
  -- Statement 3
  (perpendicular_line_with_plane m α → parallel_line_with_plane m β → perpendicular_plane_with_plane α β) :=
by
  sorry

end correct_propositions_l123_123570


namespace find_c_value_l123_123954

def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

theorem find_c_value (c : ℝ) :
  (∀ x < -1, deriv (f c) x < 0) ∧ 
  (∀ x, -1 < x → x < 0 → deriv (f c) x > 0) → 
  c = 1 :=
by 
  sorry

end find_c_value_l123_123954


namespace sum_of_solutions_l123_123933

theorem sum_of_solutions (x : ℝ) : 
  (∃ y z, x^2 + 2017 * x - 24 = 2017 ∧ y^2 + 2017 * y - 2041 = 0 ∧ z^2 + 2017 * z - 2041 = 0 ∧ y ≠ z) →
  y + z = -2017 := 
by 
  sorry

end sum_of_solutions_l123_123933


namespace triangle_problem_l123_123804

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (hb : 0 < B ∧ B < Real.pi)
  (hc : 0 < C ∧ C < Real.pi)
  (ha : 0 < A ∧ A < Real.pi)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides : a > b)
  (h_perimeter : a + b + c = 20)
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_eq : a * (Real.sqrt 3 * Real.tan B - 1) = (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C)) :
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := sorry

end triangle_problem_l123_123804


namespace candy_bars_per_friend_l123_123025

-- Definitions based on conditions
def total_candy_bars : ℕ := 24
def spare_candy_bars : ℕ := 10
def number_of_friends : ℕ := 7

-- The problem statement as a Lean theorem
theorem candy_bars_per_friend :
  (total_candy_bars - spare_candy_bars) / number_of_friends = 2 := 
by
  sorry

end candy_bars_per_friend_l123_123025


namespace remainder_of_product_mod_10_l123_123505

-- Definitions as conditions given in part a
def n1 := 2468
def n2 := 7531
def n3 := 92045

-- The problem expressed as a proof statement
theorem remainder_of_product_mod_10 :
  ((n1 * n2 * n3) % 10) = 0 :=
  by
    -- Sorry is used to skip the proof
    sorry

end remainder_of_product_mod_10_l123_123505


namespace base7_of_2345_l123_123458

def decimal_to_base7 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 6 * 7^1 + 0 * 7^0

theorem base7_of_2345 : decimal_to_base7 2345 = 6560 := by
  sorry

end base7_of_2345_l123_123458


namespace brian_distance_more_miles_l123_123140

variables (s t d m n : ℝ)
-- Mike's distance
variable (hd : d = s * t)
-- Steve's distance condition
variable (hsteve : d + 90 = (s + 6) * (t + 1.5))
-- Brian's distance
variable (hbrian : m = (s + 12) * (t + 3))

theorem brian_distance_more_miles :
  n = m - d → n = 200 :=
sorry

end brian_distance_more_miles_l123_123140


namespace calc_dz_calc_d2z_calc_d3z_l123_123415

variables (x y dx dy : ℝ)

def z : ℝ := x^5 * y^3

-- Define the first differential dz
def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

-- Define the second differential d2z
def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

-- Define the third differential d3z
def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem calc_dz : (dz x y dx dy) = (5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) := 
by sorry

theorem calc_d2z : (d2z x y dx dy) = (20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) :=
by sorry

theorem calc_d3z : (d3z x y dx dy) = (60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end calc_dz_calc_d2z_calc_d3z_l123_123415


namespace janessa_initial_cards_l123_123395

theorem janessa_initial_cards (X : ℕ)  :
  (X + 45 = 49) →
  X = 4 :=
by
  intro h
  sorry

end janessa_initial_cards_l123_123395


namespace jean_jail_time_l123_123408

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l123_123408


namespace min_pieces_per_orange_l123_123156

theorem min_pieces_per_orange (oranges : ℕ) (calories_per_orange : ℕ) (people : ℕ) (calories_per_person : ℕ) (pieces_per_orange : ℕ) :
  oranges = 5 →
  calories_per_orange = 80 →
  people = 4 →
  calories_per_person = 100 →
  pieces_per_orange ≥ 4 :=
by
  intro h_oranges h_calories_per_orange h_people h_calories_per_person
  sorry

end min_pieces_per_orange_l123_123156


namespace initial_owls_l123_123768

theorem initial_owls (n_0 : ℕ) (h : n_0 + 2 = 5) : n_0 = 3 :=
by 
  sorry

end initial_owls_l123_123768


namespace sum_of_squares_l123_123122

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 110) : x^2 + y^2 = 1380 := 
by sorry

end sum_of_squares_l123_123122


namespace fewest_printers_l123_123180

theorem fewest_printers (cost1 cost2 : ℕ) (h1 : cost1 = 375) (h2 : cost2 = 150) : 
  ∃ (n : ℕ), n = 2 + 5 :=
by
  have lcm_375_150 : Nat.lcm cost1 cost2 = 750 := sorry
  have n1 : 750 / 375 = 2 := sorry
  have n2 : 750 / 150 = 5 := sorry
  exact ⟨7, rfl⟩

end fewest_printers_l123_123180


namespace sum_of_pairwise_products_does_not_end_in_2019_l123_123480

theorem sum_of_pairwise_products_does_not_end_in_2019 (n : ℤ) : ¬ (∃ (k : ℤ), 10000 ∣ (3 * n ^ 2 - 2020 + k * 10000)) := by
  sorry

end sum_of_pairwise_products_does_not_end_in_2019_l123_123480


namespace proof_problem_l123_123204

noncomputable def red_balls : ℕ := 5
noncomputable def black_balls : ℕ := 2
noncomputable def total_balls : ℕ := red_balls + black_balls
noncomputable def draws : ℕ := 3

noncomputable def prob_red_ball := red_balls / total_balls
noncomputable def prob_black_ball := black_balls / total_balls

noncomputable def E_X : ℚ := (1/7) + 2*(4/7) + 3*(2/7)
noncomputable def E_Y : ℚ := 2*(1/7) + 1*(4/7) + 0*(2/7)
noncomputable def E_xi : ℚ := 3 * (5/7)

noncomputable def D_X : ℚ := (1 - 15/7) ^ 2 * (1/7) + (2 - 15/7) ^ 2 * (4/7) + (3 - 15/7) ^ 2 * (2/7)
noncomputable def D_Y : ℚ := (2 - 6/7) ^ 2 * (1/7) + (1 - 6/7) ^ 2 * (4/7) + (0 - 6/7) ^ 2 * (2/7)
noncomputable def D_xi : ℚ := 3 * (5/7) * (1 - 5/7)

theorem proof_problem :
  (E_X / E_Y = 5 / 2) ∧ 
  (D_X ≤ D_Y) ∧ 
  (E_X = E_xi) ∧ 
  (D_X < D_xi) :=
by {
  sorry
}

end proof_problem_l123_123204


namespace problem1_problem2_l123_123243

-- Problem 1: Proving the given equation under specified conditions
theorem problem1 (x y : ℝ) (h : x + y ≠ 0) : ((2 * x + 3 * y) / (x + y)) - ((x + 2 * y) / (x + y)) = 1 :=
sorry

-- Problem 2: Proving the given equation under specified conditions
theorem problem2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ 1) : ((a^2 - 1) / (a^2 - 4 * a + 4)) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) :=
sorry

end problem1_problem2_l123_123243


namespace smallest_sum_of_factors_of_12_factorial_l123_123738

theorem smallest_sum_of_factors_of_12_factorial :
  ∃ (x y z w : Nat), x * y * z * w = Nat.factorial 12 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w = 147 :=
by
  sorry

end smallest_sum_of_factors_of_12_factorial_l123_123738


namespace original_selling_price_l123_123099

variable (P : ℝ)
variable (S : ℝ) 

-- Conditions
axiom profit_10_percent : S = 1.10 * P
axiom profit_diff : 1.17 * P - S = 42

-- Goal
theorem original_selling_price : S = 660 := by
  sorry

end original_selling_price_l123_123099


namespace worker_b_days_l123_123437

variables (W_A W_B W : ℝ)
variables (h1 : W_A = 2 * W_B)
variables (h2 : (W_A + W_B) * 10 = W)
variables (h3 : W = 30 * W_B)

theorem worker_b_days : ∃ days : ℝ, days = 30 :=
by
  sorry

end worker_b_days_l123_123437


namespace power_function_odd_f_m_plus_1_l123_123881

noncomputable def f (x : ℝ) (m : ℝ) := x^(2 + m)

theorem power_function_odd_f_m_plus_1 (m : ℝ) (h_odd : ∀ x : ℝ, f (-x) m = -f x m)
  (h_domain : -1 ≤ m) : f (m + 1) m = 1 := by
  sorry

end power_function_odd_f_m_plus_1_l123_123881


namespace instantaneous_velocity_at_2_l123_123531

def displacement (t : ℝ) : ℝ := 2 * t^3

theorem instantaneous_velocity_at_2 :
  let velocity := deriv displacement
  velocity 2 = 24 :=
by
  sorry

end instantaneous_velocity_at_2_l123_123531


namespace time_after_hours_l123_123801

def current_time := 9
def total_hours := 2023
def clock_cycle := 12

theorem time_after_hours : (current_time + total_hours) % clock_cycle = 8 := by
  sorry

end time_after_hours_l123_123801


namespace solve_system_eqs_l123_123435

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end solve_system_eqs_l123_123435


namespace number_of_houses_l123_123291

theorem number_of_houses (total_mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : total_mail_per_block = 24) (h2 : mail_per_house = 4) : total_mail_per_block / mail_per_house = 6 :=
by
  sorry

end number_of_houses_l123_123291


namespace find_xy_l123_123547

theorem find_xy : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^4 = y^2 + 71 ∧ x = 6 ∧ y = 35 :=
by
  sorry

end find_xy_l123_123547


namespace even_odd_function_value_l123_123668

theorem even_odd_function_value 
  (f g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_odd : ∀ x, g (-x) = - g x)
  (h_eqn : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := 
by {
  sorry
}

end even_odd_function_value_l123_123668


namespace vector_addition_l123_123502

-- Definitions for the vectors
def a : ℝ × ℝ := (5, 2)
def b : ℝ × ℝ := (1, 6)

-- Proof statement (Note: "theorem" is used here instead of "def" because we are stating something to be proven)
theorem vector_addition : a + b = (6, 8) := by
  sorry

end vector_addition_l123_123502


namespace fraction_of_students_on_trip_are_girls_l123_123665

variable (b g : ℕ)
variable (H1 : g = 2 * b) -- twice as many girls as boys
variable (fraction_girls_on_trip : ℚ := 2 / 3)
variable (fraction_boys_on_trip : ℚ := 1 / 2)

def fraction_of_girls_on_trip (b g : ℕ) (H1 : g = 2 * b) (fraction_girls_on_trip : ℚ) (fraction_boys_on_trip : ℚ) :=
  let girls_on_trip := fraction_girls_on_trip * g
  let boys_on_trip := fraction_boys_on_trip * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip

theorem fraction_of_students_on_trip_are_girls (b g : ℕ) (H1 : g = 2 * b) : 
  fraction_of_girls_on_trip b g H1 (2 / 3) (1 / 2) = 8 / 11 := 
by sorry

end fraction_of_students_on_trip_are_girls_l123_123665


namespace find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l123_123994

theorem find_zeros_of_quadratic {a b : ℝ} (h_a : a = 1) (h_b : b = -2) :
  ∀ x, (a * x^2 + b * x + b - 1 = 0) ↔ (x = 3 ∨ x = -1) := sorry

theorem range_of_a_for_two_distinct_zeros :
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + b - 1 = 0 ∧ a * x2^2 + b * x2 + b - 1 = 0) ↔ (0 < a ∧ a < 1) := sorry

end find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l123_123994


namespace gcd_of_B_l123_123310

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l123_123310


namespace jelly_beans_correct_l123_123182

-- Define the constants and conditions
def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def total_amount : ℕ := sandra_savings + mother_gift + father_gift

def candy_cost : ℕ := 5 / 10 -- == 0.5
def jelly_bean_cost : ℕ := 2 / 10 -- == 0.2

def candies_bought : ℕ := 14
def money_spent_on_candies : ℕ := candies_bought * candy_cost

def remaining_money : ℕ := total_amount - money_spent_on_candies
def money_left : ℕ := 11

-- Prove the number of jelly beans bought is 20
def number_of_jelly_beans : ℕ :=
  (remaining_money - money_left) / jelly_bean_cost

theorem jelly_beans_correct : number_of_jelly_beans = 20 :=
sorry

end jelly_beans_correct_l123_123182


namespace quadratic_has_negative_root_condition_l123_123607

theorem quadratic_has_negative_root_condition (a : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, ax^2 + 2*x + 1 = 0 ∧ x < 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry

end quadratic_has_negative_root_condition_l123_123607


namespace eq_cont_fracs_l123_123642

noncomputable def cont_frac : Nat -> Rat
| 0       => 0
| (n + 1) => (n : Rat) + 1 / (cont_frac n)

theorem eq_cont_fracs (n : Nat) : 
  1 - cont_frac n = cont_frac n - 1 :=
sorry

end eq_cont_fracs_l123_123642


namespace overall_average_of_marks_l123_123714

theorem overall_average_of_marks (n total_boys passed_boys failed_boys avg_passed avg_failed : ℕ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 105)
  (h3 : failed_boys = 15)
  (h4 : total_boys = passed_boys + failed_boys)
  (h5 : avg_passed = 39)
  (h6 : avg_failed = 15) :
  ((passed_boys * avg_passed + failed_boys * avg_failed) / total_boys = 36) :=
by
  sorry

end overall_average_of_marks_l123_123714


namespace coat_price_reduction_l123_123045

theorem coat_price_reduction (original_price : ℝ) (reduction_percent : ℝ)
  (price_is_500 : original_price = 500)
  (reduction_is_30 : reduction_percent = 0.30) :
  original_price * reduction_percent = 150 :=
by
  sorry

end coat_price_reduction_l123_123045


namespace inverse_function_composition_l123_123158

def g (x : ℝ) : ℝ := 3 * x + 7

noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

theorem inverse_function_composition : g_inv (g_inv 20) = -8 / 9 := by
  sorry

end inverse_function_composition_l123_123158


namespace remainder_of_12_factorial_mod_13_l123_123631

open Nat

theorem remainder_of_12_factorial_mod_13 : (factorial 12) % 13 = 12 := by
  -- Wilson's Theorem: For a prime number \( p \), \( (p-1)! \equiv -1 \pmod{p} \)
  -- Given \( p = 13 \), we have \( 12! \equiv -1 \pmod{13} \)
  -- Thus, it follows that the remainder is 12
  sorry

end remainder_of_12_factorial_mod_13_l123_123631


namespace sector_radian_measure_l123_123333

theorem sector_radian_measure {r l : ℝ} 
  (h1 : 2 * r + l = 12) 
  (h2 : (1/2) * l * r = 8) : 
  (l / r = 1) ∨ (l / r = 4) :=
sorry

end sector_radian_measure_l123_123333


namespace red_not_equal_blue_l123_123210

theorem red_not_equal_blue (total_cubes : ℕ) (red_cubes : ℕ) (blue_cubes : ℕ) (edge_length : ℕ)
  (total_surface_squares : ℕ) (max_red_squares : ℕ) :
  total_cubes = 27 →
  red_cubes = 9 →
  blue_cubes = 18 →
  edge_length = 3 →
  total_surface_squares = 6 * edge_length^2 →
  max_red_squares = 26 →
  ¬ (total_surface_squares = 2 * max_red_squares) :=
by
  intros
  sorry

end red_not_equal_blue_l123_123210


namespace find_overtime_hours_l123_123545

theorem find_overtime_hours
  (pay_rate_ordinary : ℝ := 0.60)
  (pay_rate_overtime : ℝ := 0.90)
  (total_pay : ℝ := 32.40)
  (total_hours : ℕ := 50) :
  ∃ y : ℕ, pay_rate_ordinary * (total_hours - y) + pay_rate_overtime * y = total_pay ∧ y = 8 := 
by
  sorry

end find_overtime_hours_l123_123545


namespace wire_cut_equal_area_l123_123987

theorem wire_cut_equal_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a / b = 2 / Real.sqrt Real.pi) ↔ (a^2 / 16 = b^2 / (4 * Real.pi)) :=
by
  sorry

end wire_cut_equal_area_l123_123987


namespace solution_set_of_inequality_l123_123324

theorem solution_set_of_inequality :
  { x : ℝ | (x - 4) / (3 - 2*x) < 0 ∧ 3 - 2*x ≠ 0 } = { x : ℝ | x < 3 / 2 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l123_123324


namespace toby_peanut_butter_servings_l123_123760

theorem toby_peanut_butter_servings :
  let bread_calories := 100
  let peanut_butter_calories_per_serving := 200
  let total_calories := 500
  let bread_pieces := 1
  ∃ (servings : ℕ), total_calories = (bread_calories * bread_pieces) + (peanut_butter_calories_per_serving * servings) → servings = 2 := by
  sorry

end toby_peanut_butter_servings_l123_123760


namespace selling_price_correct_l123_123192

/-- Define the initial cost of the gaming PC. -/
def initial_pc_cost : ℝ := 1200

/-- Define the cost of the new video card. -/
def new_video_card_cost : ℝ := 500

/-- Define the total spending after selling the old card. -/
def total_spending : ℝ := 1400

/-- Define the selling price of the old card -/
def selling_price_of_old_card : ℝ := (initial_pc_cost + new_video_card_cost) - total_spending

/-- Prove that John sold the old card for $300. -/
theorem selling_price_correct : selling_price_of_old_card = 300 := by
  sorry

end selling_price_correct_l123_123192


namespace total_tiles_in_square_hall_l123_123067

theorem total_tiles_in_square_hall
  (s : ℕ) -- integer side length of the square hall
  (black_tiles : ℕ)
  (total_tiles : ℕ)
  (all_tiles_white_or_black : ∀ (x : ℕ), x ≤ total_tiles → x = black_tiles ∨ x = total_tiles - black_tiles)
  (black_tiles_count : black_tiles = 153 + 3) : total_tiles = 6084 :=
by
  sorry

end total_tiles_in_square_hall_l123_123067


namespace sum_of_angles_in_segments_outside_pentagon_l123_123006

theorem sum_of_angles_in_segments_outside_pentagon 
  (α β γ δ ε : ℝ) 
  (hα : α = 0.5 * (360 - arc_BCDE))
  (hβ : β = 0.5 * (360 - arc_CDEA))
  (hγ : γ = 0.5 * (360 - arc_DEAB))
  (hδ : δ = 0.5 * (360 - arc_EABC))
  (hε : ε = 0.5 * (360 - arc_ABCD)) 
  (arc_BCDE arc_CDEA arc_DEAB arc_EABC arc_ABCD : ℝ) :
  α + β + γ + δ + ε = 720 := 
by 
  sorry

end sum_of_angles_in_segments_outside_pentagon_l123_123006


namespace minimum_ticket_cost_l123_123889

theorem minimum_ticket_cost :
  let num_people := 12
  let num_adults := 8
  let num_children := 4
  let adult_ticket_cost := 100
  let child_ticket_cost := 50
  let group_ticket_cost := 70
  num_people = num_adults + num_children →
  (num_people >= 10) →
  ∃ (cost : ℕ), cost = min (num_adults * adult_ticket_cost + num_children * child_ticket_cost) (group_ticket_cost * num_people) ∧
  cost = min (group_ticket_cost * 10 + child_ticket_cost * (num_people - 10)) (group_ticket_cost * num_people) →
  cost = 800 :=
by
  intro h1 h2
  sorry

end minimum_ticket_cost_l123_123889


namespace tan_addition_sin_cos_expression_l123_123321

noncomputable def alpha : ℝ := sorry -- this is where alpha would be defined

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem tan_addition (alpha : ℝ) (h : Real.tan alpha = 2) : (Real.tan (alpha + Real.pi / 4) = -3) :=
by sorry

theorem sin_cos_expression (alpha : ℝ) (h : Real.tan alpha = 2) : 
  (Real.sin (2 * alpha) / (Real.sin (alpha) ^ 2 - Real.cos (2 * alpha) + 1) = 1 / 3) :=
by sorry

end tan_addition_sin_cos_expression_l123_123321


namespace ratio_of_x_to_y_l123_123487

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : x / y = -13 / 2 := 
by 
  sorry

end ratio_of_x_to_y_l123_123487


namespace exterior_angle_BAC_l123_123823

theorem exterior_angle_BAC 
    (interior_angle_nonagon : ℕ → ℚ) 
    (angle_CAD_angle_BAD : ℚ → ℚ → ℚ)
    (exterior_angle_formula : ℚ → ℚ) :
  (interior_angle_nonagon 9 = 140) ∧ 
  (angle_CAD_angle_BAD 90 140 = 230) ∧ 
  (exterior_angle_formula 230 = 130) := 
sorry

end exterior_angle_BAC_l123_123823


namespace different_purchasing_methods_l123_123919

noncomputable def number_of_purchasing_methods (n_two_priced : ℕ) (n_one_priced : ℕ) (total_price : ℕ) : ℕ :=
  let combinations_two_price (k : ℕ) := Nat.choose n_two_priced k
  let combinations_one_price (k : ℕ) := Nat.choose n_one_priced k
  combinations_two_price 5 + (combinations_two_price 4 * combinations_one_price 2)

theorem different_purchasing_methods :
  number_of_purchasing_methods 8 3 10 = 266 :=
by
  sorry

end different_purchasing_methods_l123_123919


namespace price_increase_ratio_l123_123242

theorem price_increase_ratio 
  (c : ℝ)
  (h1 : 351 = c * 1.30) :
  (c + 351) / c = 2.3 :=
sorry

end price_increase_ratio_l123_123242


namespace prime_divides_binom_l123_123938

-- We define that n is a prime number.
def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Lean statement for the problem
theorem prime_divides_binom {n k : ℕ} (h₁ : is_prime n) (h₂ : 0 < k) (h₃ : k < n) :
  n ∣ Nat.choose n k :=
sorry

end prime_divides_binom_l123_123938


namespace quadratic_decreasing_l123_123811

-- Define the quadratic function and the condition a < 0
def quadratic_function (a x : ℝ) := a * x^2 - 2 * a * x + 1

-- Define the main theorem to be proven
theorem quadratic_decreasing (a m : ℝ) (ha : a < 0) : 
  (∀ x, x > m → quadratic_function a x < quadratic_function a (x+1)) ↔ m ≥ 1 :=
by
  sorry

end quadratic_decreasing_l123_123811


namespace find_x_from_angles_l123_123548

theorem find_x_from_angles : ∀ (x : ℝ), (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end find_x_from_angles_l123_123548


namespace simplify_expression_l123_123129

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4))
  = (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 :=
by
  sorry

end simplify_expression_l123_123129


namespace point_third_quadrant_l123_123337

theorem point_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : 3 * m - 2 < 0 ∧ -n < 0 :=
by
  sorry

end point_third_quadrant_l123_123337


namespace blue_flowers_percentage_l123_123123

theorem blue_flowers_percentage :
  let total_flowers := 96
  let green_flowers := 9
  let red_flowers := 3 * green_flowers
  let yellow_flowers := 12
  let accounted_flowers := green_flowers + red_flowers + yellow_flowers
  let blue_flowers := total_flowers - accounted_flowers
  (blue_flowers / total_flowers : ℝ) * 100 = 50 :=
by
  sorry

end blue_flowers_percentage_l123_123123


namespace right_triangle_leg_length_l123_123556

theorem right_triangle_leg_length (a c b : ℝ) (h : a = 4) (h₁ : c = 5) (h₂ : a^2 + b^2 = c^2) : b = 3 := 
by
  -- by is used for the proof, which we are skipping using sorry.
  sorry

end right_triangle_leg_length_l123_123556


namespace money_combination_l123_123524

variable (Raquel Tom Nataly Sam : ℝ)

-- Given Conditions 
def condition1 : Prop := Tom = (1 / 4) * Nataly
def condition2 : Prop := Nataly = 3 * Raquel
def condition3 : Prop := Sam = 2 * Nataly
def condition4 : Prop := Nataly = (5 / 3) * Sam
def condition5 : Prop := Raquel = 40

-- Proving this combined total
def combined_total : Prop := Tom + Raquel + Nataly + Sam = 262

theorem money_combination (h1: condition1 Tom Nataly) 
                          (h2: condition2 Nataly Raquel) 
                          (h3: condition3 Sam Nataly) 
                          (h4: condition4 Nataly Sam) 
                          (h5: condition5 Raquel) 
                          : combined_total Tom Raquel Nataly Sam :=
sorry

end money_combination_l123_123524


namespace parking_lot_cars_l123_123327

theorem parking_lot_cars :
  ∀ (initial_cars cars_left cars_entered remaining_cars final_cars : ℕ),
    initial_cars = 80 →
    cars_left = 13 →
    remaining_cars = initial_cars - cars_left →
    cars_entered = cars_left + 5 →
    final_cars = remaining_cars + cars_entered →
    final_cars = 85 := 
by
  intros initial_cars cars_left cars_entered remaining_cars final_cars h1 h2 h3 h4 h5
  sorry

end parking_lot_cars_l123_123327


namespace n_squared_plus_inverse_squared_plus_four_eq_102_l123_123510

theorem n_squared_plus_inverse_squared_plus_four_eq_102 (n : ℝ) (h : n + 1 / n = 10) :
    n^2 + 1 / n^2 + 4 = 102 :=
by sorry

end n_squared_plus_inverse_squared_plus_four_eq_102_l123_123510


namespace isle_of_unluckiness_l123_123476

-- Definitions:
def is_knight (i : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k = i * n / 100 ∧ k > 0

-- Main statement:
theorem isle_of_unluckiness (n : ℕ) (h : n ∈ [1, 2, 4, 5, 10, 20, 25, 50, 100]) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ is_knight i n := by
  sorry

end isle_of_unluckiness_l123_123476


namespace complement_M_in_U_l123_123474

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_M_in_U :
  U \ M = {3, 5, 6} :=
by sorry

end complement_M_in_U_l123_123474


namespace product_of_two_numbers_l123_123377

-- Define HCF (Highest Common Factor) and LCM (Least Common Multiple) conditions
def hcf_of_two_numbers (a b : ℕ) : ℕ := 11
def lcm_of_two_numbers (a b : ℕ) : ℕ := 181

-- The theorem to prove
theorem product_of_two_numbers (a b : ℕ) 
  (h1 : hcf_of_two_numbers a b = 11)
  (h2 : lcm_of_two_numbers a b = 181) : 
  a * b = 1991 :=
by 
  -- This is where we would put the proof, but we can use sorry for now
  sorry

end product_of_two_numbers_l123_123377


namespace circumscribed_quadrilateral_identity_l123_123238

variables 
  (α β γ θ : ℝ)
  (h_angle_sum : α + β + γ + θ = 180)
  (OA OB OC OD AB BC CD DA : ℝ)
  (h_OA : OA = 1 / Real.sin α)
  (h_OB : OB = 1 / Real.sin β)
  (h_OC : OC = 1 / Real.sin γ)
  (h_OD : OD = 1 / Real.sin θ)
  (h_AB : AB = Real.sin (α + β) / (Real.sin α * Real.sin β))
  (h_BC : BC = Real.sin (β + γ) / (Real.sin β * Real.sin γ))
  (h_CD : CD = Real.sin (γ + θ) / (Real.sin γ * Real.sin θ))
  (h_DA : DA = Real.sin (θ + α) / (Real.sin θ * Real.sin α))

theorem circumscribed_quadrilateral_identity :
  OA * OC + OB * OD = Real.sqrt (AB * BC * CD * DA) := 
sorry

end circumscribed_quadrilateral_identity_l123_123238


namespace product_gt_one_l123_123853

theorem product_gt_one 
  (m : ℚ) (b : ℚ)
  (hm : m = 3 / 4)
  (hb : b = 5 / 2) :
  m * b > 1 := 
by
  sorry

end product_gt_one_l123_123853


namespace range_of_a_second_quadrant_l123_123405

theorem range_of_a_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0) → x < 0 ∧ y > 0) →
  a > 2 :=
sorry

end range_of_a_second_quadrant_l123_123405


namespace max_value_correct_l123_123133

noncomputable def max_value_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) : Prop :=
  x ^ 4 * y ^ 3 * z ^ 2 ≤ 1 / 372008

theorem max_value_correct (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) :
  max_value_ineq x y z h1 h2 h3 h4 :=
sorry

end max_value_correct_l123_123133


namespace sin_double_alpha_l123_123340

theorem sin_double_alpha (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end sin_double_alpha_l123_123340


namespace petya_cannot_form_figure_c_l123_123751

-- Define the rhombus and its properties, including rotation
noncomputable def is_rotatable_rhombus (r : ℕ) : Prop := sorry

-- Define the larger shapes and their properties in terms of whether they can be formed using rotations of the rhombus.
noncomputable def can_form_figure_a (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_b (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_c (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_d (rhombus : ℕ) : Prop := sorry

-- Statement: Petya cannot form the figure (c) using the rhombus and allowed transformations.
theorem petya_cannot_form_figure_c (rhombus : ℕ) (h : is_rotatable_rhombus rhombus) :
  ¬ can_form_figure_c rhombus := sorry

end petya_cannot_form_figure_c_l123_123751


namespace rectangle_area_l123_123375

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l123_123375


namespace sqrt_27_eq_3_sqrt_3_l123_123951

theorem sqrt_27_eq_3_sqrt_3 : Real.sqrt 27 = 3 * Real.sqrt 3 :=
by
  sorry

end sqrt_27_eq_3_sqrt_3_l123_123951


namespace flour_to_add_l123_123667

-- Define the conditions
def total_flour_required : ℕ := 9
def flour_already_added : ℕ := 2

-- Define the proof statement
theorem flour_to_add : total_flour_required - flour_already_added = 7 := 
by {
    sorry
}

end flour_to_add_l123_123667


namespace team_matches_per_season_l123_123106

theorem team_matches_per_season (teams_count total_games : ℕ) (h1 : teams_count = 50) (h2 : total_games = 4900) : 
  ∃ n : ℕ, n * (teams_count - 1) * teams_count / 2 = total_games ∧ n = 2 :=
by
  sorry

end team_matches_per_season_l123_123106


namespace range_of_quadratic_function_is_geq_11_over_4_l123_123708

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - x + 3

-- Define the range of the quadratic function
def range_of_quadratic_function := {y : ℝ | ∃ x : ℝ, quadratic_function x = y}

-- Prove the statement
theorem range_of_quadratic_function_is_geq_11_over_4 : range_of_quadratic_function = {y : ℝ | y ≥ 11 / 4} :=
by
  sorry

end range_of_quadratic_function_is_geq_11_over_4_l123_123708


namespace remainder_when_divided_by_10_l123_123601

theorem remainder_when_divided_by_10 :
  (2457 * 6291 * 9503) % 10 = 1 :=
by
  sorry

end remainder_when_divided_by_10_l123_123601


namespace find_k_eq_l123_123654

theorem find_k_eq (n : ℝ) (k m : ℤ) (h : ∀ n : ℝ, n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2) : k = 3 := 
sorry

end find_k_eq_l123_123654


namespace exponent_property_l123_123632

theorem exponent_property : (-2)^2004 + 3 * (-2)^2003 = -2^2003 :=
by 
  sorry

end exponent_property_l123_123632


namespace small_planters_needed_l123_123522

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end small_planters_needed_l123_123522


namespace greatest_b_for_no_minus_nine_in_range_l123_123253

theorem greatest_b_for_no_minus_nine_in_range :
  ∃ b_max : ℤ, (b_max = 16) ∧ (∀ b : ℤ, (b^2 < 288) ↔ (b ≤ 16)) :=
by
  sorry

end greatest_b_for_no_minus_nine_in_range_l123_123253


namespace problem_sum_value_l123_123120

def letter_value_pattern : List Int := [2, 3, 2, 1, 0, -1, -2, -3, -2, -1]

def char_value (c : Char) : Int :=
  let pos := c.toNat - 'a'.toNat + 1
  letter_value_pattern.get! ((pos - 1) % 10)

def word_value (w : String) : Int :=
  w.data.map char_value |>.sum

theorem problem_sum_value : word_value "problem" = 5 :=
  by sorry

end problem_sum_value_l123_123120


namespace calcium_carbonate_required_l123_123220

theorem calcium_carbonate_required (HCl_moles CaCO3_moles CaCl2_moles CO2_moles H2O_moles : ℕ) 
  (reaction_balanced : CaCO3_moles + 2 * HCl_moles = CaCl2_moles + CO2_moles + H2O_moles) 
  (HCl_moles_value : HCl_moles = 2) : CaCO3_moles = 1 :=
by sorry

end calcium_carbonate_required_l123_123220


namespace laura_running_speed_l123_123964

noncomputable def running_speed (x : ℝ) : ℝ := x^2 - 1

noncomputable def biking_speed (x : ℝ) : ℝ := 3 * x + 2

noncomputable def biking_time (x: ℝ) : ℝ := 30 / (biking_speed x)

noncomputable def running_time (x: ℝ) : ℝ := 5 / (running_speed x)

noncomputable def total_motion_time (x : ℝ) : ℝ := biking_time x + running_time x

-- Laura's total workout duration without transition time
noncomputable def required_motion_time : ℝ := 140 / 60

theorem laura_running_speed (x : ℝ) (hx : total_motion_time x = required_motion_time) :
  running_speed x = 83.33 :=
sorry

end laura_running_speed_l123_123964


namespace evaluate_polynomial_at_neg2_l123_123949

theorem evaluate_polynomial_at_neg2 : 2 * (-2)^4 + 3 * (-2)^3 + 5 * (-2)^2 + (-2) + 4 = 30 :=
by 
  sorry

end evaluate_polynomial_at_neg2_l123_123949


namespace unique_solution_of_equation_l123_123727

theorem unique_solution_of_equation (x y : ℝ) (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
by
  sorry

end unique_solution_of_equation_l123_123727


namespace johns_initial_bench_press_weight_l123_123528

noncomputable def initialBenchPressWeight (currentWeight: ℝ) (injuryPercentage: ℝ) (trainingFactor: ℝ) :=
  (currentWeight / (injuryPercentage / 100 * trainingFactor))

theorem johns_initial_bench_press_weight:
  (initialBenchPressWeight 300 80 3) = 500 :=
by
  sorry

end johns_initial_bench_press_weight_l123_123528


namespace find_natural_numbers_l123_123558

noncomputable def valid_n (n : ℕ) : Prop :=
  2 ^ n % 7 = n ^ 2 % 7

theorem find_natural_numbers :
  {n : ℕ | valid_n n} = {n : ℕ | n % 21 = 2 ∨ n % 21 = 4 ∨ n % 21 = 5 ∨ n % 21 = 6 ∨ n % 21 = 10 ∨ n % 21 = 15} :=
sorry

end find_natural_numbers_l123_123558


namespace geometric_series_sum_l123_123629

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l123_123629


namespace ratio_of_mistakes_l123_123367

theorem ratio_of_mistakes (riley_mistakes team_mistakes : ℕ) 
  (h_riley : riley_mistakes = 3) (h_team : team_mistakes = 17) : 
  (team_mistakes - riley_mistakes) / riley_mistakes = 14 / 3 := 
by 
  sorry

end ratio_of_mistakes_l123_123367


namespace switch_pairs_bound_l123_123286

theorem switch_pairs_bound (odd_blocks_n odd_blocks_prev : ℕ) 
  (switch_pairs_n switch_pairs_prev : ℕ)
  (H1 : switch_pairs_n = 2 * odd_blocks_n)
  (H2 : odd_blocks_n ≤ switch_pairs_prev) : 
  switch_pairs_n ≤ 2 * switch_pairs_prev :=
by
  sorry

end switch_pairs_bound_l123_123286


namespace original_book_price_l123_123234

theorem original_book_price (P : ℝ) (h : 0.85 * P * 1.40 = 476) : P = 476 / (0.85 * 1.40) :=
by
  sorry

end original_book_price_l123_123234


namespace area_of_remaining_figure_l123_123054
noncomputable def π := Real.pi

theorem area_of_remaining_figure (R : ℝ) (chord_length : ℝ) (C : ℝ) 
  (h : chord_length = 8) (hC : C = R) : (π * R^2 - 2 * π * (R / 2)^2) = 12.57 := by
  sorry

end area_of_remaining_figure_l123_123054


namespace evaluate_expression_l123_123634

theorem evaluate_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end evaluate_expression_l123_123634


namespace percentage_of_500_l123_123758

theorem percentage_of_500 : (110 / 100) * 500 = 550 := 
  by
  -- Here we would provide the proof (placeholder)
  sorry

end percentage_of_500_l123_123758


namespace multiple_of_sales_total_l123_123481

theorem multiple_of_sales_total
  (A : ℝ)
  (M : ℝ)
  (h : M * A = 0.3125 * (11 * A + M * A)) :
  M = 5 :=
by
  sorry

end multiple_of_sales_total_l123_123481


namespace meaningful_expression_range_l123_123920

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l123_123920


namespace average_last_4_matches_l123_123894

theorem average_last_4_matches (avg_10_matches avg_6_matches : ℝ) (matches_10 matches_6 matches_4 : ℕ) :
  avg_10_matches = 38.9 →
  avg_6_matches = 41 →
  matches_10 = 10 →
  matches_6 = 6 →
  matches_4 = 4 →
  (avg_10_matches * matches_10 - avg_6_matches * matches_6) / matches_4 = 35.75 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_last_4_matches_l123_123894


namespace sqrt_two_between_one_and_two_l123_123852

theorem sqrt_two_between_one_and_two : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := 
by
  -- sorry placeholder
  sorry

end sqrt_two_between_one_and_two_l123_123852


namespace subset_A_has_only_one_element_l123_123971

theorem subset_A_has_only_one_element (m : ℝ) :
  (∀ x y, (mx^2 + 2*x + 1 = 0) → (mx*y^2 + 2*y + 1 = 0) → x = y) →
  (m = 0 ∨ m = 1) :=
by
  sorry

end subset_A_has_only_one_element_l123_123971


namespace fault_line_movement_year_before_l123_123174

-- Define the total movement over two years
def total_movement : ℝ := 6.5

-- Define the movement during the past year
def past_year_movement : ℝ := 1.25

-- Define the movement the year before
def year_before_movement : ℝ := total_movement - past_year_movement

-- Prove that the fault line moved 5.25 inches the year before
theorem fault_line_movement_year_before : year_before_movement = 5.25 :=
  by  sorry

end fault_line_movement_year_before_l123_123174


namespace max_belts_l123_123482

theorem max_belts (h t b : ℕ) (Hh : h >= 1) (Ht : t >= 1) (Hb : b >= 1) (total_cost : 3 * h + 4 * t + 9 * b = 60) : b <= 5 :=
sorry

end max_belts_l123_123482


namespace geometric_sequence_q_cubed_l123_123159

noncomputable def S (a_1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a_1 else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_q_cubed (a_1 q : ℝ) (h1 : q ≠ 1) (h2 : a_1 ≠ 0)
  (h3 : S a_1 q 3 + S a_1 q 6 = 2 * S a_1 q 9) : q^3 = -1 / 2 :=
by
  sorry

end geometric_sequence_q_cubed_l123_123159


namespace remainder_of_f_when_divided_by_x_plus_2_l123_123875

def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 + 8 * x - 20

theorem remainder_of_f_when_divided_by_x_plus_2 : f (-2) = 72 := by
  sorry

end remainder_of_f_when_divided_by_x_plus_2_l123_123875


namespace students_in_class_l123_123880

def total_eggs : Nat := 56
def eggs_per_student : Nat := 8
def num_students : Nat := 7

theorem students_in_class :
  total_eggs / eggs_per_student = num_students :=
by
  sorry

end students_in_class_l123_123880


namespace scientific_notation_of_274000000_l123_123514

theorem scientific_notation_of_274000000 :
  274000000 = 2.74 * 10^8 := by
  sorry

end scientific_notation_of_274000000_l123_123514


namespace pollution_index_minimum_l123_123815

noncomputable def pollution_index (k a b : ℝ) (x : ℝ) : ℝ :=
  k * (a / (x ^ 2) + b / ((18 - x) ^ 2))

theorem pollution_index_minimum (k : ℝ) (h₀ : 0 < k) (h₁ : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 18) :
  ∀ a b x : ℝ, a = 1 → x = 6 → pollution_index k a b x = pollution_index k 1 8 6 :=
by
  intros a b x ha hx
  rw [ha, hx, pollution_index]
  sorry

end pollution_index_minimum_l123_123815


namespace third_consecutive_even_integer_l123_123434

theorem third_consecutive_even_integer (n : ℤ) (h : (n + 2) + (n + 6) = 156) : (n + 4) = 78 :=
sorry

end third_consecutive_even_integer_l123_123434


namespace find_staff_age_l123_123443

theorem find_staff_age (n_students : ℕ) (avg_age_students : ℕ) (avg_age_with_staff : ℕ) (total_students : ℕ) :
  n_students = 32 →
  avg_age_students = 16 →
  avg_age_with_staff = 17 →
  total_students = 33 →
  (33 * 17 - 32 * 16) = 49 :=
by
  intros
  sorry

end find_staff_age_l123_123443


namespace no_solution_for_inequalities_l123_123366

theorem no_solution_for_inequalities (x : ℝ) : ¬(4 * x ^ 2 + 7 * x - 2 < 0 ∧ 3 * x - 1 > 0) :=
by
  sorry

end no_solution_for_inequalities_l123_123366


namespace sqrt_double_sqrt_four_l123_123652

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_double_sqrt_four :
  sqrt (sqrt 4) = sqrt 2 ∨ sqrt (sqrt 4) = -sqrt 2 :=
by
  sorry

end sqrt_double_sqrt_four_l123_123652


namespace find_p_minus_q_l123_123882

theorem find_p_minus_q (x y p q : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 3 / (x * p) = 8) (h2 : 5 / (y * q) = 18)
  (hminX : ∀ x', x' ≠ 0 → 3 / (x' * 3) ≠ 1 / 8)
  (hminY : ∀ y', y' ≠ 0 → 5 / (y' * 5) ≠ 1 / 18) :
  p - q = 0 :=
sorry

end find_p_minus_q_l123_123882


namespace inequality_holds_l123_123316

theorem inequality_holds (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 :=
by {
  sorry
}

end inequality_holds_l123_123316


namespace ratio_of_areas_l123_123475

theorem ratio_of_areas (r : ℝ) (w_smaller : ℝ) (h_smaller : ℝ) (h_semi : ℝ) :
  (5 / 4) * 40 = r + 40 →
  h_semi = 20 →
  w_smaller = 5 →
  h_smaller = 20 →
  2 * w_smaller * h_smaller / ((1 / 2) * π * h_semi^2) = 1 / π :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_l123_123475


namespace min_value_expression_l123_123018

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    4.5 ≤ (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) :=
by
  sorry

end min_value_expression_l123_123018


namespace jaime_saves_enough_l123_123856

-- Definitions of the conditions
def weekly_savings : ℕ := 50
def bi_weekly_expense : ℕ := 46
def target_savings : ℕ := 135

-- The proof goal
theorem jaime_saves_enough : ∃ weeks : ℕ, 2 * ((weeks * weekly_savings - bi_weekly_expense) / 2) = target_savings := 
sorry

end jaime_saves_enough_l123_123856


namespace parabola_focus_correct_l123_123154

-- defining the equation of the parabola as a condition
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- defining the focus of the parabola
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- the main theorem statement
theorem parabola_focus_correct (y x : ℝ) (h : parabola y x) : focus 1 0 :=
by
  -- proof steps would go here
  sorry

end parabola_focus_correct_l123_123154


namespace students_dont_eat_lunch_l123_123328

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end students_dont_eat_lunch_l123_123328


namespace gcd_lcm_identity_l123_123878

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := a * (b / GCD a b)

theorem gcd_lcm_identity (a b c : ℕ) :
    (LCM a (LCM b c))^2 / (LCM a b * LCM b c * LCM c a) = (GCD a (GCD b c))^2 / (GCD a b * GCD b c * GCD c a) :=
by
  sorry

end gcd_lcm_identity_l123_123878


namespace trigonometric_quadrant_l123_123967

theorem trigonometric_quadrant (θ : ℝ) (h1 : Real.sin θ > Real.cos θ) (h2 : Real.sin θ * Real.cos θ < 0) : 
  (θ > π / 2) ∧ (θ < π) :=
by
  sorry

end trigonometric_quadrant_l123_123967


namespace impossible_to_half_boys_sit_with_girls_l123_123302

theorem impossible_to_half_boys_sit_with_girls:
  ∀ (g b : ℕ), 
  (g + b = 30) → 
  (∃ k, g = 2 * k) →
  (∀ (d : ℕ), 2 * d = g) →
  ¬ ∃ m, (b = 2 * m) ∧ (∀ (d : ℕ), 2 * d = b) :=
by
  sorry

end impossible_to_half_boys_sit_with_girls_l123_123302


namespace part1_part2_l123_123065

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | m - 3 ≤ x ∧ x ≤ m + 3}
noncomputable def C : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem part1 (m : ℝ) (h : A ∩ B m = C) : m = 5 :=
  sorry

theorem part2 (m : ℝ) (h : A ⊆ (B m)ᶜ) : m < -4 ∨ 6 < m :=
  sorry

end part1_part2_l123_123065


namespace geom_arith_seq_first_term_is_two_l123_123105

theorem geom_arith_seq_first_term_is_two (b q a d : ℝ) 
  (hq : q ≠ 1) 
  (h_geom_first : b = a + d) 
  (h_geom_second : b * q = a + 3 * d) 
  (h_geom_third : b * q^2 = a + 6 * d) 
  (h_prod : b * b * q * b * q^2 = 64) :
  b = 2 :=
by
  sorry

end geom_arith_seq_first_term_is_two_l123_123105


namespace world_book_day_l123_123914

theorem world_book_day
  (x y : ℕ)
  (h1 : x + y = 22)
  (h2 : x = 2 * y + 1) :
  x = 15 ∧ y = 7 :=
by {
  -- The proof is omitted as per the instructions
  sorry
}

end world_book_day_l123_123914


namespace evaluate_A_minus10_3_l123_123795

def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem evaluate_A_minus10_3 : A (-10) 3 = 1320 := 
  sorry

end evaluate_A_minus10_3_l123_123795


namespace cos_diff_l123_123273

theorem cos_diff (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.tan α = 2) : 
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_diff_l123_123273


namespace train_speed_is_correct_l123_123075

-- Define the conditions.
def length_of_train : ℕ := 1800 -- Length of the train in meters.
def time_to_cross_platform : ℕ := 60 -- Time to cross the platform in seconds (1 minute).

-- Define the statement that needs to be proved.
def speed_of_train : ℕ := (2 * length_of_train) / time_to_cross_platform

-- State the theorem.
theorem train_speed_is_correct :
  speed_of_train = 60 := by
  sorry -- Proof is not required.

end train_speed_is_correct_l123_123075


namespace combination_of_10_choose_3_l123_123515

theorem combination_of_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end combination_of_10_choose_3_l123_123515


namespace sum_of_numbers_in_row_l123_123215

theorem sum_of_numbers_in_row 
  (n : ℕ)
  (sum_eq : (n * (3 * n - 1)) / 2 = 20112) : 
  n = 1006 :=
sorry

end sum_of_numbers_in_row_l123_123215


namespace find_a_interval_l123_123111

theorem find_a_interval :
  ∀ {a : ℝ}, (∃ b x y : ℝ, x = abs (y + a) + 4 / a ∧ x^2 + y^2 + 24 + b * (2 * y + b) = 10 * x) ↔ (a < 0 ∨ a ≥ 2 / 3) :=
by {
  sorry
}

end find_a_interval_l123_123111


namespace probability_prime_sum_30_l123_123936

def prime_numbers_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def prime_pairs_summing_to_30 : List (ℕ × ℕ) := [(7, 23), (11, 19), (13, 17)]

def num_prime_pairs := (prime_numbers_up_to_30.length.choose 2)

theorem probability_prime_sum_30 :
  (prime_pairs_summing_to_30.length / num_prime_pairs : ℚ) = 1 / 15 :=
sorry

end probability_prime_sum_30_l123_123936


namespace parabola_focus_directrix_distance_l123_123205

theorem parabola_focus_directrix_distance (a : ℝ) (h_pos : a > 0) (h_dist : 1 / (2 * 2 * a) = 1) : a = 1 / 4 :=
by
  sorry

end parabola_focus_directrix_distance_l123_123205


namespace eval_expression_l123_123611

def a : ℕ := 4 * 5 * 6
def b : ℚ := 1/4 + 1/5 - 1/10

theorem eval_expression : a * b = 42 := by
  sorry

end eval_expression_l123_123611


namespace ratio_of_x_to_y_l123_123138

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 3 * x = 0.12 * 250 * y) : x / y = 10 :=
sorry

end ratio_of_x_to_y_l123_123138


namespace square_difference_l123_123834

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l123_123834


namespace proof_problem_l123_123753

def from_base (b : ℕ) (digits : List ℕ) : ℕ :=
digits.foldr (λ (d acc) => d + b * acc) 0

def problem : Prop :=
  let a := from_base 8 [2, 3, 4, 5] -- 2345 base 8
  let b := from_base 5 [1, 4, 0]    -- 140 base 5
  let c := from_base 4 [1, 0, 3, 2] -- 1032 base 4
  let d := from_base 8 [2, 9, 1, 0] -- 2910 base 8
  let result := (a / b + c - d : ℤ)
  result = -1502

theorem proof_problem : problem :=
by
  sorry

end proof_problem_l123_123753


namespace washing_time_is_45_l123_123653

-- Definitions based on conditions
variables (x : ℕ) -- time to wash one load
axiom h1 : 2 * x + 75 = 165 -- total laundry time equation

-- The statement to prove: washing one load takes 45 minutes
theorem washing_time_is_45 : x = 45 :=
by
  sorry

end washing_time_is_45_l123_123653


namespace broth_for_third_l123_123003

theorem broth_for_third (b : ℚ) (h : b = 6 + 3/4) : b / 3 = 2 + 1/4 := by
  sorry

end broth_for_third_l123_123003


namespace complex_exchange_of_apartments_in_two_days_l123_123637

theorem complex_exchange_of_apartments_in_two_days :
  ∀ (n : ℕ) (p : Fin n → Fin n), ∃ (day1 day2 : Fin n → Fin n),
    (∀ x : Fin n, p (day1 x) = day2 x ∨ day1 (p x) = day2 x) ∧
    (∀ x : Fin n, day1 x ≠ x) ∧
    (∀ x : Fin n, day2 x ≠ x) :=
by
  sorry

end complex_exchange_of_apartments_in_two_days_l123_123637


namespace original_number_of_movies_l123_123572

/-- Suppose a movie buff owns movies on DVD, Blu-ray, and digital copies in a ratio of 7:2:1.
    After purchasing 5 more Blu-ray movies and 3 more digital copies, the ratio changes to 13:4:2.
    She owns movies on no other medium.
    Prove that the original number of movies in her library before the extra purchase was 390. -/
theorem original_number_of_movies (x : ℕ) (h1 : 7 * x != 0) 
  (h2 : 2 * x != 0) (h3 : x != 0)
  (h4 : 7 * x / (2 * x + 5) = 13 / 4)
  (h5 : 7 * x / (x + 3) = 13 / 2) : 10 * x = 390 :=
by
  sorry

end original_number_of_movies_l123_123572


namespace claire_photos_l123_123623

variable (C : ℕ) -- Claire's photos
variable (L : ℕ) -- Lisa's photos
variable (R : ℕ) -- Robert's photos

-- Conditions
axiom Lisa_photos : L = 3 * C
axiom Robert_photos : R = C + 16
axiom Lisa_Robert_same : L = R

-- Proof Goal
theorem claire_photos : C = 8 :=
by
  -- Sorry skips the proof and allows the theorem to compile
  sorry

end claire_photos_l123_123623


namespace monotonic_increasing_interval_l123_123782

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb 0.5 (x^2 + 2 * x - 3)

theorem monotonic_increasing_interval :
  ∀ x, f x = Real.logb 0.5 (x^2 + 2 * x - 3) → 
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -3 ∧ x₂ < -3 → f x₁ ≤ f x₂) :=
sorry

end monotonic_increasing_interval_l123_123782


namespace basil_has_winning_strategy_l123_123101

-- Definitions based on conditions
def piles : Nat := 11
def stones_per_pile : Nat := 10
def peter_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3
def basil_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3

-- The main theorem to prove Basil has a winning strategy
theorem basil_has_winning_strategy 
  (total_stones : Nat := piles * stones_per_pile) 
  (peter_first : Bool := true) :
  exists winning_strategy_for_basil, 
    ∀ (piles_remaining : Nat) (sum_stones_remaining : Nat),
    sum_stones_remaining = piles_remaining * stones_per_pile ∨
    (1 ≤ piles_remaining ∧ piles_remaining ≤ piles) ∧
    (0 ≤ sum_stones_remaining ∧ sum_stones_remaining ≤ total_stones)
    → winning_strategy_for_basil = true := 
sorry -- The proof is omitted

end basil_has_winning_strategy_l123_123101


namespace tan_of_angle_in_third_quadrant_l123_123223

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : α < -π / 2 ∧ α > -π) 
  (h2 : Real.sin α = -Real.sqrt 5 / 5) :
  Real.tan α = 1 / 2 := 
sorry

end tan_of_angle_in_third_quadrant_l123_123223


namespace triangle_BC_length_l123_123564

theorem triangle_BC_length (A : ℝ) (AC : ℝ) (S : ℝ) (BC : ℝ)
  (h1 : A = 60) (h2 : AC = 16) (h3 : S = 220 * Real.sqrt 3) :
  BC = 49 :=
by
  sorry

end triangle_BC_length_l123_123564


namespace largest_root_ratio_l123_123175

-- Define the polynomials f(x) and g(x)
def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4

-- Define the property that x1 is the largest root of f(x) and x2 is the largest root of g(x)
def is_largest_root (p : ℝ → ℝ) (r : ℝ) : Prop := 
  p r = 0 ∧ ∀ x : ℝ, p x = 0 → x ≤ r

-- The main theorem
theorem largest_root_ratio (x1 x2 : ℝ) 
  (hx1 : is_largest_root f x1) 
  (hx2 : is_largest_root g x2) : x2 = 2 * x1 :=
sorry

end largest_root_ratio_l123_123175


namespace probability_penny_dime_halfdollar_tails_is_1_over_8_l123_123991

def probability_penny_dime_halfdollar_tails : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_penny_dime_halfdollar_tails_is_1_over_8 :
  probability_penny_dime_halfdollar_tails = 1 / 8 :=
by
  sorry

end probability_penny_dime_halfdollar_tails_is_1_over_8_l123_123991


namespace geometric_sequence_property_l123_123495

variable (a : ℕ → ℤ)
-- Assume the sequence is geometric with ratio r
variable (r : ℤ)

-- Define the sequence a_n as a geometric sequence
def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * r

-- Given condition: a_4 + a_8 = -2
axiom condition : a 4 + a 8 = -2

theorem geometric_sequence_property
  (h : geometric_sequence a r) : a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_sequence_property_l123_123495


namespace special_burger_cost_l123_123926

/-
  Prices of individual items and meals:
  - Burger: $5
  - French Fries: $3
  - Soft Drink: $3
  - Kid’s Burger: $3
  - Kid’s French Fries: $2
  - Kid’s Juice Box: $2
  - Kids Meal: $5

  Mr. Parker purchases:
  - 2 special burger meals for adults
  - 2 special burger meals and 2 kids' meals for 4 children
  - Saving $10 by buying 6 meals instead of the individual items

  Goal: 
  - Prove that the cost of one special burger meal is $8.
-/

def price_burger : Nat := 5
def price_fries : Nat := 3
def price_drink : Nat := 3
def price_kid_burger : Nat := 3
def price_kid_fries : Nat := 2
def price_kid_juice : Nat := 2
def price_kids_meal : Nat := 5

def total_adults_cost : Nat :=
  2 * price_burger + 2 * price_fries + 2 * price_drink

def total_kids_cost : Nat :=
  2 * price_kid_burger + 2 * price_kid_fries + 2 * price_kid_juice

def total_individual_cost : Nat :=
  total_adults_cost + total_kids_cost

def total_meals_cost : Nat :=
  total_individual_cost - 10

def cost_kids_meals : Nat :=
  2 * price_kids_meal

def total_cost_4_meals : Nat :=
  total_meals_cost

def cost_special_burger_meal : Nat :=
  (total_cost_4_meals - cost_kids_meals) / 2

theorem special_burger_cost : cost_special_burger_meal = 8 := by
  sorry

end special_burger_cost_l123_123926


namespace symmetry_xOz_A_l123_123552

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y , z := p.z }

theorem symmetry_xOz_A :
  let A := Point3D.mk 2 (-3) 1
  symmetry_xOz A = Point3D.mk 2 3 1 :=
by
  sorry

end symmetry_xOz_A_l123_123552


namespace find_x_l123_123785

theorem find_x (a b c d x : ℕ) 
  (h1 : x = a + 7) 
  (h2 : a = b + 12) 
  (h3 : b = c + 15) 
  (h4 : c = d + 25) 
  (h5 : d = 95) : 
  x = 154 := 
by 
  sorry

end find_x_l123_123785


namespace alpha_necessary_but_not_sufficient_for_beta_l123_123461

theorem alpha_necessary_but_not_sufficient_for_beta 
  (a b : ℝ) (hα : b * (b - a) ≤ 0) (hβ : a / b ≥ 1) : 
  (b * (b - a) ≤ 0) ↔ (a / b ≥ 1) := 
sorry

end alpha_necessary_but_not_sufficient_for_beta_l123_123461


namespace semifinalists_count_l123_123235

theorem semifinalists_count (n : ℕ) (h : (n - 2) * (n - 3) * (n - 4) = 336) : n = 10 := 
by {
  sorry
}

end semifinalists_count_l123_123235


namespace quadratic_real_roots_l123_123832

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l123_123832


namespace meaningful_sqrt_neg_x_squared_l123_123583

theorem meaningful_sqrt_neg_x_squared (x : ℝ) : (x = 0) ↔ (-(x^2) ≥ 0) :=
by
  sorry

end meaningful_sqrt_neg_x_squared_l123_123583


namespace B_catches_up_with_A_l123_123836

theorem B_catches_up_with_A :
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  tA - tB = 7 := 
by
  -- Definitions
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  -- Goal
  show tA - tB = 7
  sorry

end B_catches_up_with_A_l123_123836


namespace king_and_queen_ages_l123_123577

variable (K Q : ℕ)

theorem king_and_queen_ages (h1 : K = 2 * (Q - (K - Q)))
                            (h2 : K + (K + (K - Q)) = 63) :
                            K = 28 ∧ Q = 21 := by
  sorry

end king_and_queen_ages_l123_123577


namespace tobias_charges_for_mowing_l123_123897

/-- Tobias is buying a new pair of shoes that costs $95.
He has been saving up his money each month for the past three months.
He gets a $5 allowance a month.
He mowed 4 lawns and shoveled 5 driveways.
He charges $7 to shovel a driveway.
After buying the shoes, he has $15 in change.
Prove that Tobias charges $15 to mow a lawn.
--/
theorem tobias_charges_for_mowing 
  (shoes_cost : ℕ)
  (monthly_allowance : ℕ)
  (months_saving : ℕ)
  (lawns_mowed : ℕ)
  (driveways_shoveled : ℕ)
  (charge_per_shovel : ℕ)
  (money_left : ℕ)
  (total_money_before_purchase : ℕ)
  (x : ℕ)
  (h1 : shoes_cost = 95)
  (h2 : monthly_allowance = 5)
  (h3 : months_saving = 3)
  (h4 : lawns_mowed = 4)
  (h5 : driveways_shoveled = 5)
  (h6 : charge_per_shovel = 7)
  (h7 : money_left = 15)
  (h8 : total_money_before_purchase = shoes_cost + money_left)
  (h9 : total_money_before_purchase = (months_saving * monthly_allowance) + (lawns_mowed * x) + (driveways_shoveled * charge_per_shovel)) :
  x = 15 := 
sorry

end tobias_charges_for_mowing_l123_123897


namespace assignment_plans_l123_123695

theorem assignment_plans {students towns : ℕ} (h_students : students = 5) (h_towns : towns = 3) :
  ∃ plans : ℕ, plans = 150 :=
by
  -- Given conditions
  have h1 : students = 5 := h_students
  have h2 : towns = 3 := h_towns
  
  -- The required number of assignment plans
  existsi 150
  -- Proof is not supplied
  sorry

end assignment_plans_l123_123695


namespace ratio_y_to_x_l123_123888

-- Define the setup as given in the conditions
variables (c x y : ℝ)

-- Condition 1: Selling price x results in a loss of 20%
def condition1 : Prop := x = 0.80 * c

-- Condition 2: Selling price y results in a profit of 25%
def condition2 : Prop := y = 1.25 * c

-- Theorem: Prove the ratio of y to x is 25/16 given the conditions
theorem ratio_y_to_x (c : ℝ) (h1 : condition1 c x) (h2 : condition2 c y) : y / x = 25 / 16 := 
sorry

end ratio_y_to_x_l123_123888


namespace sqrt_t6_plus_t4_l123_123493

open Real

theorem sqrt_t6_plus_t4 (t : ℝ) : sqrt (t^6 + t^4) = t^2 * sqrt (t^2 + 1) :=
by sorry

end sqrt_t6_plus_t4_l123_123493


namespace find_k_l123_123783

variables {α : Type*} [CommRing α]

theorem find_k (a b c : α) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c :=
by sorry

end find_k_l123_123783


namespace willows_in_the_park_l123_123484

theorem willows_in_the_park (W O : ℕ) 
  (h1 : W + O = 83) 
  (h2 : O = W + 11) : 
  W = 36 := 
by 
  sorry

end willows_in_the_park_l123_123484


namespace highest_number_paper_l123_123608

theorem highest_number_paper (n : ℕ) (h : 1 / (n : ℝ) = 0.01020408163265306) : n = 98 :=
sorry

end highest_number_paper_l123_123608


namespace second_rate_of_return_l123_123260

namespace Investment

def total_investment : ℝ := 33000
def interest_total : ℝ := 970
def investment_4_percent : ℝ := 13000
def interest_rate_4_percent : ℝ := 0.04

def amount_second_investment : ℝ := total_investment - investment_4_percent
def interest_from_first_part : ℝ := interest_rate_4_percent * investment_4_percent
def interest_from_second_part (R : ℝ) : ℝ := R * amount_second_investment

theorem second_rate_of_return : (∃ R : ℝ, interest_from_first_part + interest_from_second_part R = interest_total) → 
  R = 0.0225 :=
by
  intro h
  sorry

end Investment

end second_rate_of_return_l123_123260


namespace largest_divisor_of_composite_sum_and_square_l123_123657

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_divisor_of_composite_sum_and_square (n : ℕ) (h : is_composite n) : ( ∃ (k : ℕ), ∀ n : ℕ, is_composite n → ∃ m : ℕ, n + n^2 = m * k) → k = 2 :=
by
  sorry

end largest_divisor_of_composite_sum_and_square_l123_123657


namespace distinct_paper_count_l123_123033

theorem distinct_paper_count (n : ℕ) :
  let sides := 4  -- 4 rotations and 4 reflections
  let identity_fixed := n^25 
  let rotation_90_fixed := n^7
  let rotation_270_fixed := n^7
  let rotation_180_fixed := n^13
  let reflection_fixed := n^15
  (1 / 8) * (identity_fixed + 4 * reflection_fixed + rotation_180_fixed + 2 * rotation_90_fixed) 
  = (1 / 8) * (n^25 + 4 * n^15 + n^13 + 2 * n^7) :=
  by 
    sorry

end distinct_paper_count_l123_123033


namespace interest_percentage_correct_l123_123999

noncomputable def encyclopedia_cost : ℝ := 1200
noncomputable def down_payment : ℝ := 500
noncomputable def monthly_payment : ℝ := 70
noncomputable def final_payment : ℝ := 45
noncomputable def num_monthly_payments : ℕ := 12
noncomputable def total_installment_payments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_cost_paid : ℝ := total_installment_payments + down_payment
noncomputable def amount_borrowed : ℝ := encyclopedia_cost - down_payment
noncomputable def interest_paid : ℝ := total_cost_paid - encyclopedia_cost
noncomputable def interest_percentage : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_percentage_correct : interest_percentage = 26.43 := by
  sorry

end interest_percentage_correct_l123_123999


namespace dodecahedron_interior_diagonals_l123_123390

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l123_123390


namespace gain_percent_of_50C_eq_25S_l123_123255

variable {C S : ℝ}

theorem gain_percent_of_50C_eq_25S (h : 50 * C = 25 * S) : 
  ((S - C) / C) * 100 = 100 :=
by
  sorry

end gain_percent_of_50C_eq_25S_l123_123255


namespace ln_gt_ln_sufficient_for_x_gt_y_l123_123749

noncomputable def ln : ℝ → ℝ := sorry  -- Assuming ln is imported from Mathlib

-- Conditions
variable (x y : ℝ)
axiom ln_gt_ln_of_x_gt_y (hxy : x > y) (hx_pos : 0 < x) (hy_pos : 0 < y) : ln x > ln y

theorem ln_gt_ln_sufficient_for_x_gt_y (h : ln x > ln y) : x > y := sorry

end ln_gt_ln_sufficient_for_x_gt_y_l123_123749


namespace circle_diameter_in_feet_l123_123617

/-- Given: The area of a circle is 25 * pi square inches.
    Prove: The diameter of the circle in feet is 5/6 feet. -/
theorem circle_diameter_in_feet (A : ℝ) (hA : A = 25 * Real.pi) :
  ∃ d : ℝ, d = (5 / 6) :=
by
  -- The proof goes here
  sorry

end circle_diameter_in_feet_l123_123617


namespace weather_station_accuracy_l123_123368

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

theorem weather_station_accuracy :
  binomial_probability 3 2 0.9 = 0.243 :=
by
  sorry

end weather_station_accuracy_l123_123368


namespace find_rate_squares_sum_l123_123736

theorem find_rate_squares_sum {b j s : ℤ} 
(H1 : 3 * b + 2 * j + 2 * s = 112)
(H2 : 2 * b + 3 * j + 4 * s = 129) : b^2 + j^2 + s^2 = 1218 :=
by sorry

end find_rate_squares_sum_l123_123736


namespace find_value_of_expression_l123_123513

theorem find_value_of_expression
  (a b : ℝ)
  (h₁ : a = 4 + Real.sqrt 15)
  (h₂ : b = 4 - Real.sqrt 15)
  (h₃ : ∀ x : ℝ, (x^3 - 9 * x^2 + 9 * x = 1) → (x = a ∨ x = b ∨ x = 1))
  : (a / b) + (b / a) = 62 := sorry

end find_value_of_expression_l123_123513


namespace cos_product_l123_123091

theorem cos_product : 
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 8 := 
by
  sorry

end cos_product_l123_123091


namespace factorize_m_l123_123069

theorem factorize_m (m : ℝ) : m^2 - 4 * m - 5 = (m + 1) * (m - 5) := 
sorry

end factorize_m_l123_123069


namespace money_problem_l123_123833

theorem money_problem
  (A B C : ℕ)
  (h1 : A + B + C = 450)
  (h2 : B + C = 350)
  (h3 : C = 100) :
  A + C = 200 :=
by
  sorry

end money_problem_l123_123833


namespace researcher_can_cross_desert_l123_123599

structure Condition :=
  (distance_to_oasis : ℕ)  -- total distance to be covered
  (travel_per_day : ℕ)     -- distance covered per day
  (carry_capacity : ℕ)     -- maximum days of supplies they can carry
  (ensure_return : Bool)   -- flag to ensure porters can return
  (cannot_store_food : Bool) -- flag indicating no food storage in desert

def condition_instance : Condition :=
{ distance_to_oasis := 380,
  travel_per_day := 60,
  carry_capacity := 4,
  ensure_return := true,
  cannot_store_food := true }

theorem researcher_can_cross_desert (cond : Condition) : cond.distance_to_oasis = 380 
  ∧ cond.travel_per_day = 60 
  ∧ cond.carry_capacity = 4 
  ∧ cond.ensure_return = true 
  ∧ cond.cannot_store_food = true 
  → true := 
by 
  sorry

end researcher_can_cross_desert_l123_123599


namespace find_range_of_m_l123_123584

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_of_m (m : ℝ) (h1 : ¬(p m ∧ q m)) (h2 : ¬¬p m) : m ≥ 3 ∨ m < -2 :=
by 
  sorry

end find_range_of_m_l123_123584


namespace arithmetic_expression_equality_l123_123569

theorem arithmetic_expression_equality : 18 * 36 - 27 * 18 = 162 := by
  sorry

end arithmetic_expression_equality_l123_123569


namespace part1_part2_l123_123923

section
variable (x a : ℝ)

def p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) ≤ 0

theorem part1 (h1 : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h2 : ∀ x, ¬p a x → ¬q x) : 1 < a ∧ a ≤ 2 := by
  sorry

end

end part1_part2_l123_123923


namespace distance_between_parallel_lines_l123_123355

theorem distance_between_parallel_lines (A B c1 c2 : Real) (hA : A = 2) (hB : B = 3) 
(hc1 : c1 = -3) (hc2 : c2 = 2) : 
    (abs (c1 - c2) / Real.sqrt (A^2 + B^2)) = (5 * Real.sqrt 13 / 13) := by
  sorry

end distance_between_parallel_lines_l123_123355


namespace alice_bob_numbers_count_101_l123_123983

theorem alice_bob_numbers_count_101 : 
  ∃ n : ℕ, (∀ x, 3 ≤ x ∧ x ≤ 2021 → (∃ k l, x = 3 + 5 * k ∧ x = 2021 - 4 * l)) → n = 101 :=
by
  sorry

end alice_bob_numbers_count_101_l123_123983


namespace fairy_tale_island_counties_l123_123737

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l123_123737


namespace range_of_a_l123_123168

def p (a : ℝ) : Prop := a ≤ -4 ∨ a ≥ 4
def q (a : ℝ) : Prop := a ≥ -12
def either_p_or_q_but_not_both (a : ℝ) : Prop := (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a)

theorem range_of_a :
  {a : ℝ | either_p_or_q_but_not_both a} = {a : ℝ | (-4 < a ∧ a < 4) ∨ a < -12} :=
sorry

end range_of_a_l123_123168


namespace sum_lent_out_l123_123319

theorem sum_lent_out (P R : ℝ) (h1 : 720 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 600 := by
  sorry

end sum_lent_out_l123_123319


namespace balloons_difference_l123_123740

theorem balloons_difference (yours friends : ℝ) (hyours : yours = -7) (hfriends : friends = 4.5) :
  friends - yours = 11.5 :=
by
  rw [hyours, hfriends]
  sorry

end balloons_difference_l123_123740


namespace battery_lasts_12_hours_more_l123_123562

-- Define the battery consumption rates
def standby_consumption_rate : ℚ := 1 / 36
def active_consumption_rate : ℚ := 1 / 4

-- Define the usage times
def total_time_hours : ℚ := 12
def active_use_time_hours : ℚ := 1.5
def standby_time_hours : ℚ := total_time_hours - active_use_time_hours

-- Define the total battery used during standby and active use
def standby_battery_used : ℚ := standby_time_hours * standby_consumption_rate
def active_battery_used : ℚ := active_use_time_hours * active_consumption_rate
def total_battery_used : ℚ := standby_battery_used + active_battery_used

-- Define the remaining battery
def remaining_battery : ℚ := 1 - total_battery_used

-- Define how long the remaining battery will last on standby
def remaining_standby_time : ℚ := remaining_battery / standby_consumption_rate

-- Theorem stating the correct answer
theorem battery_lasts_12_hours_more :
  remaining_standby_time = 12 := 
sorry

end battery_lasts_12_hours_more_l123_123562


namespace no_solution_for_floor_eq_l123_123294

theorem no_solution_for_floor_eq :
  ∀ s : ℝ, ¬ (⌊s⌋ + s = 15.6) :=
by sorry

end no_solution_for_floor_eq_l123_123294


namespace decreasing_interval_l123_123996

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end decreasing_interval_l123_123996


namespace graph_of_equation_is_two_lines_l123_123134

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (x * y - 2 * x + 3 * y - 6 = 0) ↔ ((x + 3 = 0) ∨ (y - 2 = 0)) := 
by
  intro x y
  sorry

end graph_of_equation_is_two_lines_l123_123134


namespace total_lives_l123_123688

theorem total_lives :
  ∀ (num_friends num_new_players lives_per_friend lives_per_new_player : ℕ),
  num_friends = 2 →
  lives_per_friend = 6 →
  num_new_players = 2 →
  lives_per_new_player = 6 →
  (num_friends * lives_per_friend + num_new_players * lives_per_new_player) = 24 :=
by
  intros num_friends num_new_players lives_per_friend lives_per_new_player
  intro h1 h2 h3 h4
  sorry

end total_lives_l123_123688


namespace pentagon_stack_l123_123432

/-- Given a stack of identical regular pentagons with vertices numbered from 1 to 5, rotated and flipped
such that the sums of numbers at each vertex are the same, the number of pentagons in the stacks can be
any natural number except 1 and 3. -/
theorem pentagon_stack (n : ℕ) (h0 : identical_pentagons_with_vertices_1_to_5)
  (h1 : pentagons_can_be_rotated_and_flipped)
  (h2 : stacked_vertex_to_vertex)
  (h3 : sums_at_each_vertex_are_equal) :
  ∃ k : ℕ, k = n ∧ n ≠ 1 ∧ n ≠ 3 :=
sorry

end pentagon_stack_l123_123432


namespace problem_1_problem_2_l123_123955

-- Define the given conditions
variables (a c : ℝ) (cosB : ℝ)
variables (b : ℝ) (S : ℝ)

-- Assuming the values for the variables
axiom h₁ : a = 4
axiom h₂ : c = 3
axiom h₃ : cosB = 1 / 8

-- Prove that b = sqrt(22)
theorem problem_1 : b = Real.sqrt 22 := by
  sorry

-- Prove that the area of triangle ABC is 9 * sqrt(7) / 4
theorem problem_2 : S = 9 * Real.sqrt 7 / 4 := by 
  sorry

end problem_1_problem_2_l123_123955


namespace find_second_sum_l123_123353

theorem find_second_sum (S : ℝ) (x : ℝ) (h : S = 2704 ∧ 24 * x / 100 = 15 * (S - x) / 100) : (S - x) = 1664 := 
  sorry

end find_second_sum_l123_123353


namespace machine_shirt_rate_l123_123540

theorem machine_shirt_rate (S : ℕ) 
  (worked_yesterday : ℕ) (worked_today : ℕ) (shirts_today : ℕ) 
  (h1 : worked_yesterday = 5)
  (h2 : worked_today = 12)
  (h3 : shirts_today = 72)
  (h4 : worked_today * S = shirts_today) : 
  S = 6 := 
by 
  sorry

end machine_shirt_rate_l123_123540


namespace basketballs_count_l123_123877

theorem basketballs_count (x : ℕ) : 
  let num_volleyballs := x
  let num_basketballs := 2 * x
  let num_soccer_balls := x - 8
  num_volleyballs + num_basketballs + num_soccer_balls = 100 →
  num_basketballs = 54 :=
by
  intros h
  sorry

end basketballs_count_l123_123877


namespace trig_identity_example_l123_123963

theorem trig_identity_example:
  (Real.sin (63 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) + 
  Real.cos (63 * Real.pi / 180) * Real.cos (108 * Real.pi / 180)) = 
  Real.sqrt 2 / 2 := 
by 
  sorry

end trig_identity_example_l123_123963


namespace probability_businessmen_wait_two_minutes_l123_123925

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l123_123925


namespace maximum_value_of_k_l123_123997

noncomputable def max_k (m : ℝ) : ℝ := 
  if 0 < m ∧ m < 1 / 2 then 
    1 / m + 2 / (1 - 2 * m) 
  else 
    0

theorem maximum_value_of_k : ∀ m : ℝ, (0 < m ∧ m < 1 / 2) → (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m) ≥ k) → k ≤ 8) :=
  sorry

end maximum_value_of_k_l123_123997


namespace kim_knit_sweaters_total_l123_123756

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l123_123756


namespace part_i_l123_123027

theorem part_i (n : ℤ) : (∃ k : ℤ, n = 225 * k + 99) ↔ (n % 9 = 0 ∧ (n + 1) % 25 = 0) :=
by 
  sorry

end part_i_l123_123027


namespace reunion_handshakes_l123_123516

-- Condition: Number of boys in total
def total_boys : ℕ := 12

-- Condition: Number of left-handed boys
def left_handed_boys : ℕ := 4

-- Condition: Number of right-handed (not exclusively left-handed) boys
def right_handed_boys : ℕ := total_boys - left_handed_boys

-- Function to calculate combinations n choose 2 (number of handshakes in a group)
def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

-- Condition: Number of handshakes among left-handed boys
def handshakes_left (n : ℕ) : ℕ := combinations left_handed_boys

-- Condition: Number of handshakes among right-handed boys
def handshakes_right (n : ℕ) : ℕ := combinations right_handed_boys

-- Problem statement: total number of handshakes
def total_handshakes (total_boys left_handed_boys right_handed_boys : ℕ) : ℕ :=
  handshakes_left left_handed_boys + handshakes_right right_handed_boys

theorem reunion_handshakes : total_handshakes total_boys left_handed_boys right_handed_boys = 34 :=
by sorry

end reunion_handshakes_l123_123516


namespace right_triangle_hypotenuse_l123_123086

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h₁ : a + b + c = 40) 
  (h₂ : a * b = 60) 
  (h₃ : a^2 + b^2 = c^2) : c = 18.5 := 
by 
  sorry

end right_triangle_hypotenuse_l123_123086


namespace pupils_like_burgers_total_l123_123024

theorem pupils_like_burgers_total (total_pupils pizza_lovers both_lovers : ℕ) :
  total_pupils = 200 →
  pizza_lovers = 125 →
  both_lovers = 40 →
  (pizza_lovers - both_lovers) + (total_pupils - pizza_lovers - both_lovers) + both_lovers = 115 :=
by
  intros h_total h_pizza h_both
  rw [h_total, h_pizza, h_both]
  sorry

end pupils_like_burgers_total_l123_123024


namespace boy_two_girls_work_completion_days_l123_123017

-- Work rates definitions
def man_work_rate := 1 / 6
def woman_work_rate := 1 / 18
def girl_work_rate := 1 / 12
def team_work_rate := 1 / 3

-- Boy's work rate
def boy_work_rate := 1 / 36

-- Combined work rate of boy and two girls
def boy_two_girls_work_rate := boy_work_rate + 2 * girl_work_rate

-- Prove that the number of days it will take for a boy and two girls to complete the work is 36 / 7
theorem boy_two_girls_work_completion_days : (1 / boy_two_girls_work_rate) = 36 / 7 :=
by
  sorry

end boy_two_girls_work_completion_days_l123_123017


namespace find_x_given_inverse_relationship_l123_123536

variable {x y : ℝ}

theorem find_x_given_inverse_relationship 
  (h₀ : x > 0) 
  (h₁ : y > 0) 
  (initial_condition : 3^2 * 25 = 225)
  (inversion_condition : x^2 * y = 225)
  (query : y = 1200) :
  x = Real.sqrt (3 / 16) :=
by
  sorry

end find_x_given_inverse_relationship_l123_123536


namespace profit_equation_l123_123990

noncomputable def price_and_profit (x : ℝ) : ℝ :=
  (1 + 0.5) * x * 0.8 - x

theorem profit_equation : ∀ x : ℝ, price_and_profit x = 8 → ((1 + 0.5) * x * 0.8 - x = 8) :=
 by intros x h
    exact h

end profit_equation_l123_123990


namespace odd_function_inequality_l123_123594

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_inequality
  (f : ℝ → ℝ) (h1 : is_odd_function f)
  (a b : ℝ) (h2 : f a > f b) :
  f (-a) < f (-b) :=
by
  sorry

end odd_function_inequality_l123_123594


namespace sum_p_q_r_l123_123929

def b (n : ℕ) : ℕ :=
if n < 1 then 0 else
if n < 2 then 2 else
if n < 4 then 4 else
if n < 7 then 6
else 6 -- Continue this pattern for illustration; an infinite structure would need proper handling for all n.

noncomputable def p := 2
noncomputable def q := 0
noncomputable def r := 0

theorem sum_p_q_r : p + q + r = 2 :=
by sorry

end sum_p_q_r_l123_123929


namespace coeff_matrix_correct_l123_123388

-- Define the system of linear equations as given conditions
def eq1 (x y : ℝ) : Prop := 2 * x + 3 * y = 1
def eq2 (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the coefficient matrix
def coeffMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 3],
  ![1, -2]
]

-- The theorem stating that the coefficient matrix of the system is as defined
theorem coeff_matrix_correct (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : 
  coeffMatrix = ![
    ![2, 3],
    ![1, -2]
  ] :=
sorry

end coeff_matrix_correct_l123_123388


namespace roots_of_polynomial_l123_123030

   -- We need to define the polynomial and then state that the roots are exactly {0, 3, -5}
   def polynomial (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x)

   theorem roots_of_polynomial :
     {x : ℝ | polynomial x = 0} = {0, 3, -5} :=
   by
     sorry
   
end roots_of_polynomial_l123_123030


namespace solution1_solution2_l123_123073

noncomputable def problem1 (x : ℝ) : Prop :=
  4 * x^2 - 25 = 0

theorem solution1 (x : ℝ) : problem1 x ↔ x = 5 / 2 ∨ x = -5 / 2 :=
by sorry

noncomputable def problem2 (x : ℝ) : Prop :=
  (x + 1)^3 = -27

theorem solution2 (x : ℝ) : problem2 x ↔ x = -4 :=
by sorry

end solution1_solution2_l123_123073


namespace fraction_meaningful_iff_nonzero_l123_123136

theorem fraction_meaningful_iff_nonzero (x : ℝ) : (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 :=
by sorry

end fraction_meaningful_iff_nonzero_l123_123136


namespace tarantulas_per_egg_sac_l123_123791

-- Condition: Each tarantula has 8 legs
def legs_per_tarantula : ℕ := 8

-- Condition: There are 32000 baby tarantula legs
def total_legs : ℕ := 32000

-- Condition: Number of egg sacs is one less than 5
def number_of_egg_sacs : ℕ := 5 - 1

-- Calculated: Number of tarantulas in total
def total_tarantulas : ℕ := total_legs / legs_per_tarantula

-- Proof Statement: Number of tarantulas per egg sac
theorem tarantulas_per_egg_sac : total_tarantulas / number_of_egg_sacs = 1000 := by
  sorry

end tarantulas_per_egg_sac_l123_123791


namespace squares_total_l123_123303

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end squares_total_l123_123303


namespace part1_part2_l123_123462

-- Part 1
theorem part1 : (9 / 4) ^ (1 / 2) - (-2.5) ^ 0 - (8 / 27) ^ (2 / 3) + (3 / 2) ^ (-2) = 1 / 2 := 
by sorry

-- Part 2
theorem part2 (lg : ℝ → ℝ) -- Assuming a hypothetical lg function for demonstration
  (lg_prop1 : lg 10 = 1)
  (lg_prop2 : ∀ x y, lg (x * y) = lg x + lg y) :
  (lg 5) ^ 2 + lg 2 * lg 50 = 1 := 
by sorry

end part1_part2_l123_123462


namespace relationship_y1_y2_y3_l123_123157

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_l123_123157


namespace problem_part_1_problem_part_2_l123_123662

theorem problem_part_1 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  |c| ≤ 1 :=
by
  sorry

theorem problem_part_2 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → |g x| ≤ 2 :=
by
  sorry

end problem_part_1_problem_part_2_l123_123662


namespace negation_of_p_l123_123463

def p (x : ℝ) : Prop := x^3 - x^2 + 1 < 0

theorem negation_of_p : (¬ ∀ x : ℝ, p x) ↔ ∃ x : ℝ, ¬ p x := by
  sorry

end negation_of_p_l123_123463


namespace age_difference_l123_123425

theorem age_difference (J P : ℕ) 
  (h1 : P = 16 - 10) 
  (h2 : P = (1 / 3) * J) : 
  (J + 10) - 16 = 12 := 
by 
  sorry

end age_difference_l123_123425


namespace min_buildings_20x20_min_buildings_50x90_l123_123819

structure CityGrid where
  width : ℕ
  height : ℕ

noncomputable def renovationLaw (grid : CityGrid) : ℕ :=
  if grid.width = 20 ∧ grid.height = 20 then 25
  else if grid.width = 50 ∧ grid.height = 90 then 282
  else sorry -- handle other cases if needed

-- Theorem statements for the proof
theorem min_buildings_20x20 : renovationLaw { width := 20, height := 20 } = 25 := by
  sorry

theorem min_buildings_50x90 : renovationLaw { width := 50, height := 90 } = 282 := by
  sorry

end min_buildings_20x20_min_buildings_50x90_l123_123819


namespace eccentricity_of_ellipse_l123_123848

theorem eccentricity_of_ellipse :
  ∀ (x y : ℝ), (x^2) / 25 + (y^2) / 16 = 1 → 
  (∃ (e : ℝ), e = 3 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l123_123848


namespace manny_paula_weight_l123_123224

   variable (m n o p : ℕ)

   -- Conditions
   variable (h1 : m + n = 320) 
   variable (h2 : n + o = 295) 
   variable (h3 : o + p = 310) 

   theorem manny_paula_weight : m + p = 335 :=
   by
     sorry
   
end manny_paula_weight_l123_123224


namespace right_triangle_set_l123_123503

def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_set :
  (is_right_triangle 3 4 2 = false) ∧
  (is_right_triangle 5 12 15 = false) ∧
  (is_right_triangle 8 15 17 = true) ∧
  (is_right_triangle (3^2) (4^2) (5^2) = false) :=
by
  sorry

end right_triangle_set_l123_123503


namespace distance_between_tangent_and_parallel_line_l123_123563

noncomputable def distance_between_parallel_lines 
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ) 
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) : ℝ :=
sorry

variable (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop)

axiom tangent_line_at_point (M : ℝ × ℝ) (C : Set (ℝ × ℝ)) : (ℝ × ℝ → Prop)

theorem distance_between_tangent_and_parallel_line
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) :
  C = { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 } →
  M = (-2, 4) →
  l = tangent_line_at_point M C →
  l1 = { p | a * p.1 + 3 * p.2 + 2 * a = 0 } →
  distance_between_parallel_lines C center r M l a l1 = 12/5 :=
by
  intros hC hM hl hl1
  sorry

end distance_between_tangent_and_parallel_line_l123_123563


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l123_123553

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l123_123553


namespace correct_option_D_l123_123014

noncomputable def total_students := 40
noncomputable def male_students := 25
noncomputable def female_students := 15
noncomputable def class_president := 1
noncomputable def prob_class_president := class_president / total_students
noncomputable def prob_class_president_from_females := 0

theorem correct_option_D
  (h1 : total_students = 40)
  (h2 : male_students = 25)
  (h3 : female_students = 15)
  (h4 : class_president = 1) :
  prob_class_president = 1 / 40 ∧ prob_class_president_from_females = 0 := 
by
  sorry

end correct_option_D_l123_123014


namespace constant_term_of_expansion_l123_123104

theorem constant_term_of_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ ∀ r : ℕ, r = 1 → (Nat.choose 5 r * 3^r * x^((5-5*r)/2) = c)) :=
by
  sorry

end constant_term_of_expansion_l123_123104


namespace garden_perimeter_is_24_l123_123790

def perimeter_of_garden(a b c x: ℕ) (h1: a + b + c = 3) : ℕ :=
  3 + 5 + a + x + b + 4 + c + 4 + 5 - x

theorem garden_perimeter_is_24 (a b c x : ℕ) (h1 : a + b + c = 3) :
  perimeter_of_garden a b c x h1 = 24 :=
  by
  sorry

end garden_perimeter_is_24_l123_123790


namespace possible_values_of_angle_F_l123_123445

-- Define angle F conditions in a triangle DEF
def triangle_angle_F_conditions (D E : ℝ) : Prop :=
  5 * Real.sin D + 2 * Real.cos E = 8 ∧ 3 * Real.sin E + 5 * Real.cos D = 2

-- The main statement: proving the possible values of ∠F
theorem possible_values_of_angle_F (D E : ℝ) (h : triangle_angle_F_conditions D E) : 
  ∃ F : ℝ, F = Real.arcsin (43 / 50) ∨ F = 180 - Real.arcsin (43 / 50) :=
by
  sorry

end possible_values_of_angle_F_l123_123445


namespace sample_size_obtained_l123_123468

/-- A theorem which states the sample size obtained when a sample is taken from a population. -/
theorem sample_size_obtained 
  (total_students : ℕ)
  (sample_students : ℕ)
  (h1 : total_students = 300)
  (h2 : sample_students = 50) : 
  sample_students = 50 :=
by
  sorry

end sample_size_obtained_l123_123468


namespace evaluate_fraction_l123_123988

-- Define the custom operations x@y and x#y
def op_at (x y : ℝ) : ℝ := x * y - y^2
def op_hash (x y : ℝ) : ℝ := x + y - x * y^2 + x^2

-- State the proof goal
theorem evaluate_fraction : (op_at 7 3) / (op_hash 7 3) = -3 :=
by
  -- Calculations to prove the theorem
  sorry

end evaluate_fraction_l123_123988


namespace simplify_expression_l123_123976

theorem simplify_expression (x y z : ℤ) (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by
  sorry

end simplify_expression_l123_123976


namespace smallest_four_digit_divisible_by_34_l123_123066

/-- Define a four-digit number. -/
def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

/-- Define a number to be divisible by another number. -/
def divisible_by (n k : ℕ) : Prop :=
  k ∣ n

/-- Prove that the smallest four-digit number divisible by 34 is 1020. -/
theorem smallest_four_digit_divisible_by_34 : ∃ n : ℕ, is_four_digit n ∧ divisible_by n 34 ∧ 
    (∀ m : ℕ, is_four_digit m → divisible_by m 34 → n ≤ m) :=
  sorry

end smallest_four_digit_divisible_by_34_l123_123066


namespace loss_eq_cost_price_of_x_balls_l123_123032

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end loss_eq_cost_price_of_x_balls_l123_123032


namespace algebraic_expression_values_l123_123055

-- Defining the given condition
def condition (x y : ℝ) : Prop :=
  x^4 + 6 * x^2 * y + 9 * y^2 + 2 * x^2 + 6 * y + 4 = 7

-- Defining the target expression
def target_expression (x y : ℝ) : ℝ :=
  x^4 + 6 * x^2 * y + 9 * y^2 - 2 * x^2 - 6 * y - 1

-- Stating the theorem to be proved
theorem algebraic_expression_values (x y : ℝ) (h : condition x y) :
  target_expression x y = -2 ∨ target_expression x y = 14 :=
by
  sorry

end algebraic_expression_values_l123_123055


namespace sum_of_values_l123_123566

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 5 * x - 3 else x^2 - 4 * x + 3

theorem sum_of_values (s : Finset ℝ) : 
  (∀ x ∈ s, f x = 2) → s.sum id = 4 :=
by 
  sorry

end ProofProblem

end sum_of_values_l123_123566


namespace water_usage_difference_l123_123802

theorem water_usage_difference (C X : ℕ)
    (h1 : C = 111000)
    (h2 : C = 3 * X)
    (days : ℕ) (h3 : days = 365) :
    (C * days - X * days) = 26910000 := by
  sorry

end water_usage_difference_l123_123802


namespace prices_proof_sales_revenue_proof_l123_123125

-- Definitions for the prices and quantities
def price_peanut_oil := 50
def price_corn_oil := 40

-- Conditions from the problem
def condition1 (x y : ℕ) : Prop := 20 * x + 30 * y = 2200
def condition2 (x y : ℕ) : Prop := 30 * x + 10 * y = 1900
def purchased_peanut_oil := 50
def selling_price_peanut_oil := 60

-- Proof statement for Part 1
theorem prices_proof : ∃ (x y : ℕ), condition1 x y ∧ condition2 x y ∧ x = price_peanut_oil ∧ y = price_corn_oil :=
sorry

-- Proof statement for Part 2
theorem sales_revenue_proof : ∃ (m : ℕ), (selling_price_peanut_oil * m > price_peanut_oil * purchased_peanut_oil) ∧ m = 42 :=
sorry

end prices_proof_sales_revenue_proof_l123_123125


namespace red_cookies_count_l123_123946

-- Definitions of the conditions
def total_cookies : ℕ := 86
def pink_cookies : ℕ := 50

-- The proof problem statement
theorem red_cookies_count : ∃ y : ℕ, y = total_cookies - pink_cookies := by
  use 36
  show 36 = total_cookies - pink_cookies
  sorry

end red_cookies_count_l123_123946


namespace combined_capacity_is_40_l123_123297

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l123_123297


namespace sequence_problem_l123_123478

theorem sequence_problem :
  7 * 9 * 11 + (7 + 9 + 11) = 720 :=
by
  sorry

end sequence_problem_l123_123478


namespace suff_not_necessary_condition_l123_123778

noncomputable def p : ℝ := 1
noncomputable def q (x : ℝ) : Prop := x^3 - 2 * x + 1 = 0

theorem suff_not_necessary_condition :
  (∀ x, x = p → q x) ∧ (∃ x, q x ∧ x ≠ p) :=
by
  sorry

end suff_not_necessary_condition_l123_123778


namespace cos_identity_proof_l123_123344

noncomputable def cos_eq_half : Prop :=
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7)) = 1 / 2

theorem cos_identity_proof : cos_eq_half :=
  by sorry

end cos_identity_proof_l123_123344


namespace solve_for_b_l123_123335

theorem solve_for_b 
  (b : ℝ)
  (h : (25 * b^2) - 84 = 0) :
  b = (2 * Real.sqrt 21) / 5 ∨ b = -(2 * Real.sqrt 21) / 5 :=
by sorry

end solve_for_b_l123_123335


namespace solve_price_of_meat_l123_123225

def price_of_meat_per_ounce (x : ℕ) : Prop :=
  16 * x - 30 = 8 * x + 18

theorem solve_price_of_meat : ∃ x, price_of_meat_per_ounce x ∧ x = 6 :=
by
  sorry

end solve_price_of_meat_l123_123225


namespace inequality_solution_l123_123365

open Real

theorem inequality_solution (a x : ℝ) :
  (a = 0 ∧ x > 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a = 1 ∧ ∀ x, ¬ (a * x^2 - (2 * a + 2) * x + 4 > 0)) ∨
  (a < 0 ∧ (x < 2/a ∨ x > 2) ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (0 < a ∧ a < 1 ∧ 2 < x ∧ x < 2/a ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a > 1 ∧ 2/a < x ∧ x < 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) := 
sorry

end inequality_solution_l123_123365


namespace age_of_fourth_child_l123_123918

theorem age_of_fourth_child 
  (avg_age : ℕ) 
  (age1 age2 age3 : ℕ) 
  (age4 : ℕ)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg_age) 
  (h1 : age1 = 6) 
  (h2 : age2 = 8) 
  (h3 : age3 = 11) 
  (h_avg_val : avg_age = 9) : 
  age4 = 11 := 
by 
  sorry

end age_of_fourth_child_l123_123918


namespace ratio_smaller_triangle_to_trapezoid_area_l123_123050

theorem ratio_smaller_triangle_to_trapezoid_area (a b : ℕ) (sqrt_three : ℝ) 
  (h_a : a = 10) (h_b : b = 2) (h_sqrt_three : sqrt_three = Real.sqrt 3) :
  ( ( (sqrt_three / 4 * (b ^ 2)) / 
      ( (sqrt_three / 4 * (a ^ 2)) - 
         (sqrt_three / 4 * (b ^ 2)))) = 1 / 24 ) := 
by
  -- conditions from the problem
  have h1: a = 10 := by exact h_a
  have h2: b = 2 := by exact h_b
  have h3: sqrt_three = Real.sqrt 3 := by exact h_sqrt_three
  sorry

end ratio_smaller_triangle_to_trapezoid_area_l123_123050


namespace calculate_value_l123_123293

theorem calculate_value :
  12 * ( (1 / 3 : ℝ) + (1 / 4) + (1 / 6) )⁻¹ = 16 :=
sorry

end calculate_value_l123_123293


namespace find_m_l123_123439

theorem find_m (x n m : ℝ) (h : (x + n)^2 = x^2 + 4*x + m) : m = 4 :=
sorry

end find_m_l123_123439


namespace evaluate_ratio_is_negative_two_l123_123320

noncomputable def evaluate_ratio (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : ℂ :=
  (a^15 + b^15) / (a + b)^15

theorem evaluate_ratio_is_negative_two (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : 
  evaluate_ratio a b h = -2 := 
sorry

end evaluate_ratio_is_negative_two_l123_123320


namespace chair_cost_l123_123734

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l123_123734


namespace rate_percent_simple_interest_l123_123323

theorem rate_percent_simple_interest
  (SI P : ℚ) (T : ℕ) (R : ℚ) : SI = 160 → P = 800 → T = 4 → (P * R * T / 100 = SI) → R = 5 :=
  by
  intros hSI hP hT hFormula
  -- Assertion that R = 5 is correct based on the given conditions and formula
  sorry

end rate_percent_simple_interest_l123_123323


namespace angle_measure_is_zero_l123_123970

-- Definitions corresponding to conditions
variable (x : ℝ)

def complement (x : ℝ) := 90 - x
def supplement (x : ℝ) := 180 - x

-- Final proof statement
theorem angle_measure_is_zero (h : complement x = (1 / 2) * supplement x) : x = 0 :=
  sorry

end angle_measure_is_zero_l123_123970


namespace total_lateness_l123_123148

/-
  Conditions:
  Charlize was 20 minutes late.
  Ana was 5 minutes later than Charlize.
  Ben was 15 minutes less late than Charlize.
  Clara was twice as late as Charlize.
  Daniel was 10 minutes earlier than Clara.

  Total time for which all five students were late is 120 minutes.
-/

def charlize := 20
def ana := charlize + 5
def ben := charlize - 15
def clara := charlize * 2
def daniel := clara - 10

def total_time := charlize + ana + ben + clara + daniel

theorem total_lateness : total_time = 120 :=
by
  sorry

end total_lateness_l123_123148


namespace quadratic_has_real_roots_l123_123948

theorem quadratic_has_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m-2) * x^2 - 2 * x + 1 = 0) ↔ m ≤ 3 :=
by sorry

end quadratic_has_real_roots_l123_123948


namespace george_second_half_questions_l123_123713

noncomputable def george_first_half_questions : ℕ := 6
noncomputable def points_per_question : ℕ := 3
noncomputable def george_final_score : ℕ := 30

theorem george_second_half_questions :
  (george_final_score - (george_first_half_questions * points_per_question)) / points_per_question = 4 :=
by
  sorry

end george_second_half_questions_l123_123713


namespace sector_area_l123_123346

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 6) (h_α : α = π / 3) : (1 / 2) * (α * r) * r = 6 * π :=
by
  rw [h_r, h_α]
  sorry

end sector_area_l123_123346


namespace find_number_of_hens_l123_123635

def hens_and_cows_problem (H C : ℕ) : Prop :=
  (H + C = 50) ∧ (2 * H + 4 * C = 144)

theorem find_number_of_hens (H C : ℕ) (hc : hens_and_cows_problem H C) : H = 28 :=
by {
  -- We assume the problem conditions and skip the proof using sorry
  sorry
}

end find_number_of_hens_l123_123635


namespace dave_deleted_apps_l123_123227

-- Definitions based on problem conditions
def original_apps : Nat := 16
def remaining_apps : Nat := 5

-- Theorem statement for proving how many apps Dave deleted
theorem dave_deleted_apps : original_apps - remaining_apps = 11 :=
by
  sorry

end dave_deleted_apps_l123_123227


namespace hess_law_delta_H298_l123_123786

def standardEnthalpyNa2O : ℝ := -416 -- kJ/mol
def standardEnthalpyH2O : ℝ := -286 -- kJ/mol
def standardEnthalpyNaOH : ℝ := -427.8 -- kJ/mol
def deltaH298 : ℝ := 2 * standardEnthalpyNaOH - (standardEnthalpyNa2O + standardEnthalpyH2O) 

theorem hess_law_delta_H298 : deltaH298 = -153.6 := by
  sorry

end hess_law_delta_H298_l123_123786


namespace wives_identification_l123_123166

theorem wives_identification (Anna Betty Carol Dorothy MrBrown MrGreen MrWhite MrSmith : ℕ):
  Anna = 2 ∧ Betty = 3 ∧ Carol = 4 ∧ Dorothy = 5 ∧
  (MrBrown = Dorothy ∧ MrGreen = 2 * Carol ∧ MrWhite = 3 * Betty ∧ MrSmith = 4 * Anna) ∧
  (Anna + Betty + Carol + Dorothy + MrBrown + MrGreen + MrWhite + MrSmith = 44) →
  (
    Dorothy = 5 ∧
    Carol = 4 ∧
    Betty = 3 ∧
    Anna = 2 ∧
    MrBrown = 5 ∧
    MrGreen = 8 ∧
    MrWhite = 9 ∧
    MrSmith = 8
  ) :=
by
  intros
  sorry

end wives_identification_l123_123166


namespace total_holes_dug_l123_123602

theorem total_holes_dug :
  (Pearl_digging_rate * 21 + Miguel_digging_rate * 21) = 26 :=
by
  -- Definitions based on conditions
  let Pearl_digging_rate := 4 / 7
  let Miguel_digging_rate := 2 / 3
  -- Sorry placeholder for the proof
  sorry

end total_holes_dug_l123_123602


namespace calc_fraction_l123_123152

theorem calc_fraction :
  ((1 / 3 + 1 / 6) * (4 / 7) * (5 / 9) = 10 / 63) :=
by
  sorry

end calc_fraction_l123_123152


namespace cosine_identity_l123_123805

theorem cosine_identity (alpha : ℝ) (h1 : -180 < alpha ∧ alpha < -90)
  (cos_75_alpha : Real.cos (75 * Real.pi / 180 + alpha) = 1 / 3) :
  Real.cos (15 * Real.pi / 180 - alpha) = -2 * Real.sqrt 2 / 3 := by
sorry

end cosine_identity_l123_123805


namespace subsets_of_A_value_of_a_l123_123034

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 - a*x + 2 = 0}

theorem subsets_of_A : 
  (A = {1, 2} ∧ (∀ S, S ⊆ A → S = ∅ ∨ S = {1} ∨ S = {2} ∨ S = {1, 2}))  :=
by
  sorry

theorem value_of_a (a : ℝ) (B_non_empty : B a ≠ ∅) (B_subset_A : ∀ x, x ∈ B a → x ∈ A): 
  a = 3 :=
by
  sorry

end subsets_of_A_value_of_a_l123_123034


namespace find_m_l123_123256

theorem find_m (x y m : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x + m * y = 5) : m = 3 := 
by
  sorry

end find_m_l123_123256


namespace evaluate_expression_l123_123517

theorem evaluate_expression : 
  3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 :=
by
  sorry

end evaluate_expression_l123_123517


namespace identify_value_of_expression_l123_123165

theorem identify_value_of_expression (x y z : ℝ)
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x * (y + z) - y * (x - y)) :
  (y^2 + z^2 - x^2) / (2 * y * z) = 1 / 2 := 
sorry

end identify_value_of_expression_l123_123165


namespace cody_books_second_week_l123_123830

noncomputable def total_books := 54
noncomputable def books_first_week := 6
noncomputable def books_weeks_after_second := 9
noncomputable def total_weeks := 7

theorem cody_books_second_week :
  let b2 := total_books - (books_first_week + books_weeks_after_second * (total_weeks - 2))
  b2 = 3 :=
by
  sorry

end cody_books_second_week_l123_123830


namespace geo_seq_a3_equals_one_l123_123864

theorem geo_seq_a3_equals_one (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_T5 : a 1 * a 2 * a 3 * a 4 * a 5 = 1) : a 3 = 1 :=
sorry

end geo_seq_a3_equals_one_l123_123864


namespace mathematicians_contemporaries_probability_l123_123857

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end mathematicians_contemporaries_probability_l123_123857


namespace find_y_given_x_inverse_square_l123_123683

theorem find_y_given_x_inverse_square (x y : ℚ) : 
  (∀ k, (3 * y = k / x^2) ∧ (3 * 5 = k / 2^2)) → (x = 6) → y = 5 / 9 :=
by
  sorry

end find_y_given_x_inverse_square_l123_123683


namespace greatest_value_divisible_by_3_l123_123090

theorem greatest_value_divisible_by_3 :
  ∃ (a : ℕ), (168026 + 1000 * a) % 3 = 0 ∧ a ≤ 9 ∧ ∀ b : ℕ, (168026 + 1000 * b) % 3 = 0 → b ≤ 9 → a ≥ b :=
sorry

end greatest_value_divisible_by_3_l123_123090


namespace total_cost_of_toys_l123_123378

-- Define the costs of the yoyo and the whistle
def cost_yoyo : Nat := 24
def cost_whistle : Nat := 14

-- Prove the total cost of the yoyo and the whistle is 38 cents
theorem total_cost_of_toys : cost_yoyo + cost_whistle = 38 := by
  sorry

end total_cost_of_toys_l123_123378


namespace sum_pqrst_is_neg_15_over_2_l123_123143

variable (p q r s t x : ℝ)
variable (h1 : p + 2 = x)
variable (h2 : q + 3 = x)
variable (h3 : r + 4 = x)
variable (h4 : s + 5 = x)
variable (h5 : t + 6 = x)
variable (h6 : p + q + r + s + t + 10 = x)

theorem sum_pqrst_is_neg_15_over_2 : p + q + r + s + t = -15 / 2 := by
  sorry

end sum_pqrst_is_neg_15_over_2_l123_123143


namespace insects_total_l123_123221

def total_insects (n_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
                  (n_stones : ℕ) (ants_per_stone : ℕ) 
                  (total_bees : ℕ) (n_flowers : ℕ) : ℕ :=
  let num_ladybugs := n_leaves * ladybugs_per_leaf
  let num_ants := n_stones * ants_per_stone
  let num_bees := total_bees -- already given as total_bees
  num_ladybugs + num_ants + num_bees

theorem insects_total : total_insects 345 267 178 423 498 6 = 167967 :=
  by unfold total_insects; sorry

end insects_total_l123_123221


namespace num_adults_attended_l123_123281

-- Definitions for the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_children : ℕ := 28
def total_revenue : ℕ := 5122

-- The goal is to prove the number of adults who attended the show
theorem num_adults_attended :
  ∃ (A : ℕ), A * ticket_price_adult + num_children * ticket_price_child = total_revenue ∧ A = 183 :=
by
  sorry

end num_adults_attended_l123_123281


namespace evaluate_f_g_l123_123720

def g (x : ℝ) : ℝ := 3 * x
def f (x : ℝ) : ℝ := x - 6

theorem evaluate_f_g :
  f (g 3) = 3 :=
by
  sorry

end evaluate_f_g_l123_123720


namespace emma_chocolates_l123_123359

theorem emma_chocolates 
  (x : ℕ) 
  (h1 : ∃ l : ℕ, x = l + 10) 
  (h2 : ∃ l : ℕ, l = x / 3) : 
  x = 15 := 
  sorry

end emma_chocolates_l123_123359


namespace problem_part1_problem_part2_l123_123119

-- Problem statements

theorem problem_part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 := 
sorry

theorem problem_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := 
sorry

end problem_part1_problem_part2_l123_123119


namespace find_x_l123_123058

def vector := ℝ × ℝ

def a : vector := (1, 1)
def b (x : ℝ) : vector := (2, x)

def vector_add (u v : vector) : vector :=
(u.1 + v.1, u.2 + v.2)

def scalar_mul (k : ℝ) (v : vector) : vector :=
(k * v.1, k * v.2)

def vector_sub (u v : vector) : vector :=
(u.1 - v.1, u.2 - v.2)

def are_parallel (u v : vector) : Prop :=
∃ k : ℝ, u = scalar_mul k v

theorem find_x (x : ℝ) : are_parallel (vector_add a (b x)) (vector_sub (scalar_mul 4 (b x)) (scalar_mul 2 a)) → x = 2 :=
by
  sorry

end find_x_l123_123058


namespace number_to_match_l123_123153

def twenty_five_percent_less (x: ℕ) : ℕ := 3 * x / 4

def one_third_more (n: ℕ) : ℕ := 4 * n / 3

theorem number_to_match (n : ℕ) (x : ℕ) 
  (h1 : x = 80) 
  (h2 : one_third_more n = twenty_five_percent_less x) : n = 45 :=
by
  -- Proof is skipped as per the instruction
  sorry

end number_to_match_l123_123153


namespace minimum_m_minus_n_l123_123116

theorem minimum_m_minus_n (m n : ℕ) (hm : m > n) (h : (9^m) % 100 = (9^n) % 100) : m - n = 10 := 
sorry

end minimum_m_minus_n_l123_123116


namespace compute_fraction_sum_l123_123376

theorem compute_fraction_sum :
  8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 :=
by
  sorry

end compute_fraction_sum_l123_123376


namespace gain_percentage_second_book_l123_123049

theorem gain_percentage_second_book (CP1 CP2 SP1 SP2 : ℝ)
  (h1 : CP1 = 350) 
  (h2 : CP1 + CP2 = 600)
  (h3 : SP1 = CP1 - (0.15 * CP1))
  (h4 : SP1 = SP2) :
  SP2 = CP2 + (19 / 100 * CP2) :=
by
  sorry

end gain_percentage_second_book_l123_123049


namespace function_equality_l123_123838

theorem function_equality (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f n < f (n + 1) )
  (h2 : f 2 = 2)
  (h3 : ∀ m n : ℕ, f (m * n) = f m * f n) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_equality_l123_123838


namespace fish_weight_l123_123026

theorem fish_weight (θ H T : ℝ) (h1 : θ = 4) (h2 : H = θ + 0.5 * T) (h3 : T = H + θ) : H + T + θ = 32 :=
by
  sorry

end fish_weight_l123_123026


namespace closest_point_on_line_l123_123704

theorem closest_point_on_line :
  ∀ (x y : ℝ), (4, -2) = (4, -2) →
    y = 3 * x - 1 →
    (∃ (p : ℝ × ℝ), p = (-0.5, -2.5) ∧ p = (-0.5, -2.5))
  := by
    -- The proof of the theorem goes here
    sorry

end closest_point_on_line_l123_123704


namespace find_total_cards_l123_123620

def numCardsInStack (n : ℕ) : Prop :=
  let cards : List ℕ := List.range' 1 (2 * n + 1)
  let pileA := cards.take n
  let pileB := cards.drop n
  let restack := List.zipWith (fun x y => [y, x]) pileA pileB |> List.join
  (restack.take 13).getLastD 0 = 13 ∧ 2 * n = 26

theorem find_total_cards : ∃ (n : ℕ), numCardsInStack n :=
sorry

end find_total_cards_l123_123620


namespace no_integers_satisfy_equation_l123_123900

theorem no_integers_satisfy_equation :
  ∀ (a b c : ℤ), a^2 + b^2 - 8 * c ≠ 6 := by
  sorry

end no_integers_satisfy_equation_l123_123900


namespace initial_customers_l123_123021

theorem initial_customers (x : ℕ) (h : x - 3 + 39 = 50) : x = 14 :=
by
  sorry

end initial_customers_l123_123021


namespace midpoint_lattice_point_exists_l123_123126

theorem midpoint_lattice_point_exists (S : Finset (ℤ × ℤ)) (hS : S.card = 5) :
  ∃ (p1 p2 : ℤ × ℤ), p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 ∧
  (∃ (x_mid y_mid : ℤ), 
    (p1.1 + p2.1) = 2 * x_mid ∧
    (p1.2 + p2.2) = 2 * y_mid) :=
by
  sorry

end midpoint_lattice_point_exists_l123_123126


namespace prime_factor_count_l123_123268

theorem prime_factor_count (n : ℕ) (H : 22 + n + 2 = 29) : n = 5 := 
  sorry

end prime_factor_count_l123_123268


namespace valid_grid_iff_divisible_by_9_l123_123266

-- Definitions for the letters used in the grid
inductive Letter
| I
| M
| O

-- Function that captures the condition that each row and column must contain exactly one-third of each letter
def valid_row_col (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ row, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ col, grid row col ∈ [Letter.I, Letter.M, Letter.O])) ∧
  ∀ col, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ row, grid row col ∈ [Letter.I, Letter.M, Letter.O]))

-- Function that captures the condition that each diagonal must contain exactly one-third of each letter when the length is a multiple of 3
def valid_diagonals (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ k, (3 ∣ k → (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = k / 3 ∧ count_M = k / 3 ∧ count_O = k / 3 ∧
    ((∀ (i j : ℕ), (i + j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]) ∨
     (∀ (i j : ℕ), (i - j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]))))

-- The main theorem stating that if we can fill the grid according to the rules, then n must be a multiple of 9
theorem valid_grid_iff_divisible_by_9 (n : ℕ) :
  (∃ grid : ℕ → ℕ → Letter, valid_row_col n grid ∧ valid_diagonals n grid) ↔ 9 ∣ n :=
by
  sorry

end valid_grid_iff_divisible_by_9_l123_123266


namespace max_chocolates_eaten_by_Ben_l123_123031

-- Define the situation with Ben and Carol sharing chocolates
variable (b c k : ℕ) -- b for Ben, c for Carol, k is the multiplier

-- Define the conditions
def chocolates_shared (b c : ℕ) : Prop := b + c = 30
def carol_eats_multiple (b c k : ℕ) : Prop := c = k * b ∧ k > 0

-- The theorem statement that we want to prove
theorem max_chocolates_eaten_by_Ben 
  (h1 : chocolates_shared b c) 
  (h2 : carol_eats_multiple b c k) : 
  b ≤ 15 := by
  sorry

end max_chocolates_eaten_by_Ben_l123_123031


namespace brocard_inequality_part_a_brocard_inequality_part_b_l123_123735

variable (α β γ φ : ℝ)

theorem brocard_inequality_part_a (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  φ^3 ≤ (α - φ) * (β - φ) * (γ - φ) := 
sorry

theorem brocard_inequality_part_b (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  8 * φ^3 ≤ α * β * γ := 
sorry

end brocard_inequality_part_a_brocard_inequality_part_b_l123_123735


namespace value_of_x2_minus_y2_l123_123549

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the conditions
def condition1 : Prop := (x + y) / 2 = 5
def condition2 : Prop := (x - y) / 2 = 2

-- State the theorem to prove
theorem value_of_x2_minus_y2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 40 :=
by
  sorry

end value_of_x2_minus_y2_l123_123549


namespace lemonade_stand_total_profit_l123_123759

theorem lemonade_stand_total_profit :
  let day1_revenue := 21 * 4
  let day1_expenses := 10 + 5 + 3
  let day1_profit := day1_revenue - day1_expenses

  let day2_revenue := 18 * 5
  let day2_expenses := 12 + 6 + 4
  let day2_profit := day2_revenue - day2_expenses

  let day3_revenue := 25 * 4
  let day3_expenses := 8 + 4 + 3 + 2
  let day3_profit := day3_revenue - day3_expenses

  let total_profit := day1_profit + day2_profit + day3_profit

  total_profit = 217 := by
    sorry

end lemonade_stand_total_profit_l123_123759


namespace find_d_l123_123587

theorem find_d 
    (a b c d : ℝ) 
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_c_pos : 0 < c)
    (h_d_pos : 0 < d)
    (max_val : d + a = 7)
    (min_val : d - a = 1) :
    d = 4 :=
by
  sorry

end find_d_l123_123587


namespace leadership_board_stabilizes_l123_123843

theorem leadership_board_stabilizes :
  ∃ n : ℕ, 2 ^ n - 1 ≤ 2020 ∧ 2020 < 2 ^ (n + 1) - 1 := by
  sorry

end leadership_board_stabilizes_l123_123843


namespace remainder_of_13_pow_13_plus_13_div_14_l123_123686

theorem remainder_of_13_pow_13_plus_13_div_14 : ((13 ^ 13 + 13) % 14) = 12 :=
by
  sorry

end remainder_of_13_pow_13_plus_13_div_14_l123_123686


namespace min_abs_phi_l123_123059

theorem min_abs_phi {f : ℝ → ℝ} (h : ∀ x, f x = 3 * Real.sin (2 * x + φ) ∧ ∀ x, f (x) = f (2 * π / 3 - x)) :
  |φ| = π / 6 :=
by
  sorry

end min_abs_phi_l123_123059


namespace heather_blocks_l123_123890

theorem heather_blocks (x : ℝ) (h1 : x + 41 = 127) : x = 86 := by
  sorry

end heather_blocks_l123_123890


namespace minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l123_123173

-- Define the conditions once to reuse them for both proof statements.
variables {a b : ℝ} (ha: a > 0) (hb: b > 0) (h: (1/a) + (1/b) = Real.sqrt (a * b))

-- Problem (I)
theorem minimum_a3_b3 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

-- Problem (II)
theorem no_exist_a_b_2a_3b_eq_6 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  ¬ ∃ (a b : ℝ), 2 * a + 3 * b = 6 :=
sorry

end minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l123_123173


namespace div_count_27n5_l123_123747

theorem div_count_27n5 
  (n : ℕ) 
  (h : (120 * n^3).divisors.card = 120) 
  : (27 * n^5).divisors.card = 324 :=
sorry

end div_count_27n5_l123_123747


namespace same_terminal_side_angle_l123_123002

theorem same_terminal_side_angle (k : ℤ) : 
  (∃ k : ℤ, - (π / 6) = 2 * k * π + a) → a = 11 * π / 6 :=
sorry

end same_terminal_side_angle_l123_123002


namespace find_fourth_vertex_of_square_l123_123828

-- Given the vertices of the square as complex numbers
def vertex1 : ℂ := 1 + 2 * Complex.I
def vertex2 : ℂ := -2 + Complex.I
def vertex3 : ℂ := -1 - 2 * Complex.I

-- The fourth vertex (to be proved)
def vertex4 : ℂ := 2 - Complex.I

-- The mathematically equivalent proof problem statement
theorem find_fourth_vertex_of_square :
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  -- Define vectors from the vertices
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4
  vector_ab = vector_dc :=
by {
  -- Definitions already provided above
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4

  -- Placeholder for proof
  sorry
}

end find_fourth_vertex_of_square_l123_123828


namespace guzman_boxes_l123_123042

noncomputable def total_doughnuts : Nat := 48
noncomputable def doughnuts_per_box : Nat := 12

theorem guzman_boxes :
  ∃ (N : Nat), N = total_doughnuts / doughnuts_per_box ∧ N = 4 :=
by
  use 4
  sorry

end guzman_boxes_l123_123042


namespace volume_of_large_ball_l123_123771

theorem volume_of_large_ball (r : ℝ) (V_small : ℝ) (h1 : 1 = r / (2 * r)) (h2 : V_small = (4 / 3) * Real.pi * r^3) : 
  8 * V_small = 288 :=
by
  sorry

end volume_of_large_ball_l123_123771


namespace number_of_BA3_in_sample_l123_123862

-- Definitions for the conditions
def strains_BA1 : Nat := 60
def strains_BA2 : Nat := 20
def strains_BA3 : Nat := 40
def total_sample_size : Nat := 30

def total_strains : Nat := strains_BA1 + strains_BA2 + strains_BA3

-- Theorem statement translating to the equivalent proof problem
theorem number_of_BA3_in_sample :
  total_sample_size * strains_BA3 / total_strains = 10 :=
by
  sorry

end number_of_BA3_in_sample_l123_123862


namespace solution_set_of_inequality_l123_123488

theorem solution_set_of_inequality (x m : ℝ) : 
  (x^2 - (2 * m + 1) * x + m^2 + m < 0) ↔ m < x ∧ x < m + 1 := 
by
  sorry

end solution_set_of_inequality_l123_123488


namespace marbles_problem_l123_123586

def marbles_total : ℕ := 30
def prob_black_black : ℚ := 14 / 25
def prob_white_white : ℚ := 16 / 225

theorem marbles_problem (total_marbles : ℕ) (prob_bb prob_ww : ℚ) 
  (h_total : total_marbles = 30)
  (h_prob_bb : prob_bb = 14 / 25)
  (h_prob_ww : prob_ww = 16 / 225) :
  let m := 16
  let n := 225
  m.gcd n = 1 ∧ m + n = 241 :=
by {
  sorry
}

end marbles_problem_l123_123586


namespace solve_for_y_l123_123392

theorem solve_for_y : ∃ y : ℝ, y = -2 ∧ y^2 + 6 * y + 8 = -(y + 2) * (y + 6) :=
by
  use -2
  sorry

end solve_for_y_l123_123392


namespace sum_of_exponents_l123_123666

theorem sum_of_exponents (n : ℕ) (h : n = 2^11 + 2^10 + 2^5 + 2^4 + 2^2) : 11 + 10 + 5 + 4 + 2 = 32 :=
by {
  -- The proof could be written here
  sorry
}

end sum_of_exponents_l123_123666


namespace largest_root_eq_l123_123185

theorem largest_root_eq : ∃ x, (∀ y, (abs (Real.cos (Real.pi * y) + y^3 - 3 * y^2 + 3 * y) = 3 - y^2 - 2 * y^3) → y ≤ x) ∧ x = 1 := sorry

end largest_root_eq_l123_123185


namespace largest_prime_factor_of_12321_l123_123868

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end largest_prime_factor_of_12321_l123_123868


namespace unbroken_seashells_l123_123744

theorem unbroken_seashells (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) 
  (h_unbroken : unbroken_seashells = total_seashells - broken_seashells) : 
  unbroken_seashells = 3 :=
by 
  rw [h_total, h_broken] at h_unbroken
  exact h_unbroken

end unbroken_seashells_l123_123744


namespace maximize_k_l123_123250

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end maximize_k_l123_123250


namespace total_value_of_coins_l123_123723

theorem total_value_of_coins :
  (∀ (coins : List (String × ℕ)), coins.length = 12 →
    (∃ Q N : ℕ, 
      Q = 4 ∧ N = 8 ∧
      (∀ (coin : String × ℕ), coin ∈ coins → 
        (coin = ("quarter", Q) → Q = 4 ∧ (Q * 25 = 100)) ∧ 
        (coin = ("nickel", N) → N = 8 ∧ (N * 5 = 40)) ∧
      (Q * 25 + N * 5 = 140)))) :=
sorry

end total_value_of_coins_l123_123723


namespace translated_point_B_coords_l123_123529

-- Define the initial point A
def point_A : ℝ × ℝ := (-2, 2)

-- Define the translation operations
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

-- Define the translation of point A to point B
def point_B :=
  translate_right (translate_down point_A 4) 3

-- The proof statement
theorem translated_point_B_coords : point_B = (1, -2) :=
  by sorry

end translated_point_B_coords_l123_123529


namespace eugene_payment_correct_l123_123799

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l123_123799


namespace pink_highlighters_count_l123_123109

-- Define the necessary constants and types
def total_highlighters : ℕ := 12
def yellow_highlighters : ℕ := 2
def blue_highlighters : ℕ := 4

-- We aim to prove that the number of pink highlighters is 6
theorem pink_highlighters_count : ∃ (pink_highlighters : ℕ), 
  pink_highlighters = total_highlighters - (yellow_highlighters + blue_highlighters) ∧
  pink_highlighters = 6 :=
by
  sorry

end pink_highlighters_count_l123_123109


namespace bucket_capacity_l123_123858

theorem bucket_capacity (x : ℝ) (h1 : 24 * x = 36 * 9) : x = 13.5 :=
by 
  sorry

end bucket_capacity_l123_123858


namespace juanita_spends_more_l123_123001

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end juanita_spends_more_l123_123001


namespace find_equation_of_l_l123_123237

open Real

/-- Define the point M(2, 1) -/
def M : ℝ × ℝ := (2, 1)

/-- Define the original line equation x - 2y + 1 = 0 as a function -/
def line1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- Define the line l that passes through M and is perpendicular to line1 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 5 = 0

/-- The theorem to be proven: the line l passing through M and perpendicular to line1 has the equation 2x + y - 5 = 0 -/
theorem find_equation_of_l (x y : ℝ)
  (hM : M = (2, 1))
  (hl_perpendicular : ∀ x y : ℝ, line1 x y → line_l y (-x / 2)) :
  line_l x y ↔ (x, y) = (2, 1) :=
by
  sorry

end find_equation_of_l_l123_123237


namespace drop_in_water_level_l123_123673

theorem drop_in_water_level (rise_level : ℝ) (drop_level : ℝ) 
  (h : rise_level = 1) : drop_level = -2 :=
by
  sorry

end drop_in_water_level_l123_123673


namespace value_of_g_g_2_l123_123398

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l123_123398


namespace wire_around_field_l123_123012

theorem wire_around_field (A L : ℕ) (hA : A = 69696) (hL : L = 15840) : L / (4 * (Nat.sqrt A)) = 15 :=
by
  sorry

end wire_around_field_l123_123012


namespace at_least_one_not_less_than_two_l123_123088

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l123_123088


namespace find_x_ge_0_l123_123818

-- Defining the condition and the proof problem
theorem find_x_ge_0 :
  {x : ℝ | (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0} = {x : ℝ | 0 ≤ x} :=
by
  sorry -- proof steps not included

end find_x_ge_0_l123_123818


namespace sum_of_squares_base_6_l123_123000

def to_base (n b : ℕ) : ℕ := sorry

theorem sum_of_squares_base_6 :
  let squares := (List.range 12).map (λ x => x.succ ^ 2);
  let squares_base6 := squares.map (λ x => to_base x 6);
  (squares_base6.sum) = to_base 10515 6 :=
by sorry

end sum_of_squares_base_6_l123_123000


namespace parabola_focus_l123_123226

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l123_123226


namespace total_number_of_vehicles_l123_123787

theorem total_number_of_vehicles 
  (lanes : ℕ) 
  (trucks_per_lane : ℕ) 
  (buses_per_lane : ℕ) 
  (cars_per_lane : ℕ := 2 * lanes * trucks_per_lane) 
  (motorcycles_per_lane : ℕ := 3 * buses_per_lane)
  (total_trucks : ℕ := lanes * trucks_per_lane)
  (total_cars : ℕ := lanes * cars_per_lane)
  (total_buses : ℕ := lanes * buses_per_lane)
  (total_motorcycles : ℕ := lanes * motorcycles_per_lane)
  (total_vehicles : ℕ := total_trucks + total_cars + total_buses + total_motorcycles)
  (hlanes : lanes = 4) 
  (htrucks : trucks_per_lane = 60) 
  (hbuses : buses_per_lane = 40) :
  total_vehicles = 2800 := sorry

end total_number_of_vehicles_l123_123787


namespace total_volume_cylinder_cone_sphere_l123_123908

theorem total_volume_cylinder_cone_sphere (r h : ℝ) (π : ℝ)
  (hc : π * r^2 * h = 150 * π)
  (hv : ∀ (r h : ℝ) (π : ℝ), V_cone = 1/3 * π * r^2 * h)
  (hs : ∀ (r : ℝ) (π : ℝ), V_sphere = 4/3 * π * r^3) :
  V_total = 50 * π + (4/3 * π * (150^(2/3))) :=
by
  sorry

end total_volume_cylinder_cone_sphere_l123_123908


namespace hours_to_destination_l123_123765

def num_people := 4
def water_per_person_per_hour := 1 / 2
def total_water_bottles_needed := 32

theorem hours_to_destination : 
  ∃ h : ℕ, (num_people * water_per_person_per_hour * 2 * h = total_water_bottles_needed) → h = 8 :=
by
  sorry

end hours_to_destination_l123_123765


namespace bobs_fruit_drink_cost_l123_123957

theorem bobs_fruit_drink_cost
  (cost_soda : ℕ)
  (cost_hamburger : ℕ)
  (cost_sandwiches : ℕ)
  (bob_total_spent same_amount : ℕ)
  (andy_spent_eq : same_amount = cost_soda + 2 * cost_hamburger)
  (andy_bob_spent_eq : same_amount = bob_total_spent)
  (bob_sandwich_cost_eq : cost_sandwiches = 3)
  (andy_spent_eq_total : cost_soda = 1)
  (andy_burger_cost : cost_hamburger = 2)
  : bob_total_spent - cost_sandwiches = 2 :=
by
  sorry

end bobs_fruit_drink_cost_l123_123957


namespace percentage_difference_is_20_l123_123043

/-
Given:
Height of sunflowers from Packet A = 192 inches
Height of sunflowers from Packet B = 160 inches

Show:
Percentage difference in height between Packet A and Packet B is 20%.
-/

-- Definitions of heights
def height_packet_A : ℤ := 192
def height_packet_B : ℤ := 160

-- Definition of percentage difference formula
def percentage_difference (hA hB : ℤ) : ℤ := ((hA - hB) * 100) / hB

-- Theorem statement
theorem percentage_difference_is_20 :
  percentage_difference height_packet_A height_packet_B = 20 :=
sorry

end percentage_difference_is_20_l123_123043


namespace Rohan_earning_after_6_months_l123_123252

def farm_area : ℕ := 20
def trees_per_sqm : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval : ℕ := 3
def sale_price : ℝ := 0.50
def total_months : ℕ := 6

theorem Rohan_earning_after_6_months :
  farm_area * trees_per_sqm * coconuts_per_tree * (total_months / harvest_interval) * sale_price 
    = 240 := by
  sorry

end Rohan_earning_after_6_months_l123_123252


namespace Morse_code_distinct_symbols_l123_123428

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l123_123428


namespace minimum_price_to_cover_costs_l123_123373

variable (P : ℝ)

-- Conditions
def prod_cost_A := 80
def ship_cost_A := 2
def prod_cost_B := 60
def ship_cost_B := 3
def fixed_costs := 16200
def units_A := 200
def units_B := 300

-- Cost calculations
def total_cost_A := units_A * prod_cost_A + units_A * ship_cost_A
def total_cost_B := units_B * prod_cost_B + units_B * ship_cost_B
def total_costs := total_cost_A + total_cost_B + fixed_costs

-- Revenue requirement
def revenue (P_A P_B : ℝ) := units_A * P_A + units_B * P_B

theorem minimum_price_to_cover_costs :
  (units_A + units_B) * P ≥ total_costs ↔ P ≥ 103 :=
sorry

end minimum_price_to_cover_costs_l123_123373


namespace estimate_y_value_at_x_equals_3_l123_123650

noncomputable def estimate_y (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 3) * x + a

theorem estimate_y_value_at_x_equals_3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ) (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ),
    (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 2 * (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8)) →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 8 →
    estimate_y 3 (1 / 6) = 7 / 6 := by
  intro x1 x2 x3 x4 x5 x6 x7 x8 y1 y2 y3 y4 y5 y6 y7 y8 h_sum hx
  sorry

end estimate_y_value_at_x_equals_3_l123_123650


namespace maximize_volume_l123_123689

theorem maximize_volume
  (R H A : ℝ) (K : ℝ) (hA : 2 * π * R * H + 2 * π * R * (Real.sqrt (R ^ 2 + H ^ 2)) = A)
  (hK : K = A / (2 * π)) :
  R = (A / (π * Real.sqrt 5)) ^ (1 / 3) :=
sorry

end maximize_volume_l123_123689


namespace ship_passengers_percentage_l123_123430

variables (P R : ℝ)

-- Conditions
def condition1 : Prop := (0.20 * P) = (0.60 * R)

-- Target
def target : Prop := R / P = 1 / 3

theorem ship_passengers_percentage
  (h1 : condition1 P R) :
  target P R :=
by
  sorry

end ship_passengers_percentage_l123_123430


namespace lastTwoNonZeroDigits_of_80_fact_is_8_l123_123074

-- Define the factorial function
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Define the function to find the last two nonzero digits of a factorial
def lastTwoNonZeroDigits (n : ℕ) : ℕ := sorry -- Placeholder logic for now

-- State the problem as a theorem
theorem lastTwoNonZeroDigits_of_80_fact_is_8 :
  lastTwoNonZeroDigits 80 = 8 :=
sorry

end lastTwoNonZeroDigits_of_80_fact_is_8_l123_123074


namespace reduction_for_same_profit_cannot_reach_460_profit_l123_123413

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end reduction_for_same_profit_cannot_reach_460_profit_l123_123413


namespace factor_polynomial_l123_123746

variable (x : ℝ)

theorem factor_polynomial : (270 * x^3 - 90 * x^2 + 18 * x) = 18 * x * (15 * x^2 - 5 * x + 1) :=
by 
  sorry

end factor_polynomial_l123_123746


namespace apples_handout_l123_123846

theorem apples_handout {total_apples pies_needed pies_count handed_out : ℕ}
  (h1 : total_apples = 51)
  (h2 : pies_needed = 5)
  (h3 : pies_count = 2)
  (han : handed_out = total_apples - (pies_needed * pies_count)) :
  handed_out = 41 :=
by {
  sorry
}

end apples_handout_l123_123846


namespace negation_equiv_l123_123400
variable (x : ℝ)

theorem negation_equiv :
  (¬ ∃ x : ℝ, x^2 + 1 > 3 * x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3 * x) :=
by 
  sorry

end negation_equiv_l123_123400


namespace equation_not_expression_with_unknowns_l123_123534

def is_equation (expr : String) : Prop :=
  expr = "equation"

def contains_unknowns (expr : String) : Prop :=
  expr = "contains unknowns"

theorem equation_not_expression_with_unknowns (expr : String) (h1 : is_equation expr) (h2 : contains_unknowns expr) : 
  (is_equation expr) = False := 
sorry

end equation_not_expression_with_unknowns_l123_123534


namespace rectangular_garden_length_l123_123228

theorem rectangular_garden_length (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 900) : l = 300 :=
by
  sorry

end rectangular_garden_length_l123_123228


namespace scatter_plot_correlation_l123_123543

noncomputable def correlation_coefficient (points : List (ℝ × ℝ)) : ℝ := sorry

theorem scatter_plot_correlation {points : List (ℝ × ℝ)} 
  (h : ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) :
  correlation_coefficient points = 1 := 
sorry

end scatter_plot_correlation_l123_123543


namespace length_OR_coordinates_Q_area_OPQR_8_p_value_l123_123574

noncomputable def point_R : (ℝ × ℝ) := (0, 4)

noncomputable def OR_distance : ℝ := 0 - 4 -- the vertical distance from O to R

theorem length_OR : OR_distance = 4 := sorry

noncomputable def point_Q (p : ℝ) : (ℝ × ℝ) := (p, 2 * p + 4)

theorem coordinates_Q (p : ℝ) : point_Q p = (p, 2 * p + 4) := sorry

noncomputable def area_OPQR (p : ℝ) : ℝ := 
  let OR : ℝ := 4
  let PQ : ℝ := 2 * p + 4
  let OP : ℝ := p
  1 / 2 * (OR + PQ) * OP

theorem area_OPQR_8 : area_OPQR 8 = 96 := sorry

theorem p_value (h : area_OPQR p = 77) : p = 7 := sorry

end length_OR_coordinates_Q_area_OPQR_8_p_value_l123_123574


namespace point_Q_representation_l123_123465

-- Definitions
variables {C D Q : Type} [AddCommGroup C] [AddCommGroup D] [AddCommGroup Q] [Module ℝ C] [Module ℝ D] [Module ℝ Q]
variable (CQ : ℝ)
variable (QD : ℝ)
variable (r s : ℝ)

-- Given condition: ratio CQ:QD = 7:2
axiom CQ_QD_ratio : CQ / QD = 7 / 2

-- Proof goal: the affine combination representation of the point Q
theorem point_Q_representation : CQ / (CQ + QD) = 7 / 9 ∧ QD / (CQ + QD) = 2 / 9 :=
sorry

end point_Q_representation_l123_123465


namespace complement_intersection_in_U_l123_123144

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_in_U : (U \ (A ∩ B)) = {1, 4, 5, 6, 7, 8} :=
by {
  sorry
}

end complement_intersection_in_U_l123_123144


namespace simple_interest_rate_l123_123186

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (SI_eq : SI = 260)
  (P_eq : P = 910) (T_eq : T = 4)
  (H : SI = P * R * T / 100) : 
  R = 26000 / 3640 := 
by
  sorry

end simple_interest_rate_l123_123186


namespace correct_quotient_l123_123670

theorem correct_quotient (D Q : ℕ) (h1 : 21 * Q = 12 * 56) : Q = 32 :=
by {
  -- Proof to be provided
  sorry
}

end correct_quotient_l123_123670


namespace find_tricycles_l123_123750

theorem find_tricycles (b t w : ℕ) 
  (sum_children : b + t + w = 10)
  (sum_wheels : 2 * b + 3 * t = 26) :
  t = 6 :=
by sorry

end find_tricycles_l123_123750


namespace cone_radius_height_ratio_l123_123178

theorem cone_radius_height_ratio 
  (V : ℝ) (π : ℝ) (r h : ℝ)
  (circumference : ℝ) 
  (original_height : ℝ)
  (new_volume : ℝ)
  (volume_formula : V = (1/3) * π * r^2 * h)
  (radius_from_circumference : 2 * π * r = circumference)
  (base_circumference : circumference = 28 * π)
  (original_height_eq : original_height = 45)
  (new_volume_eq : new_volume = 441 * π) :
  (r / h) = 14 / 9 :=
by
  sorry

end cone_radius_height_ratio_l123_123178


namespace shadow_of_cube_l123_123810

theorem shadow_of_cube (x : ℝ) (h_edge : ∀ c : ℝ, c = 2) (h_shadow_area : ∀ a : ℝ, a = 200 + 4) :
  ⌊1000 * x⌋ = 12280 :=
by
  sorry

end shadow_of_cube_l123_123810


namespace exponential_inequality_l123_123697

-- Define the problem conditions and the proof goal
theorem exponential_inequality (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := 
sorry

end exponential_inequality_l123_123697


namespace school_avg_GPA_l123_123979

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l123_123979


namespace greatest_points_for_top_teams_l123_123233

-- Definitions as per the conditions
def teams := 9 -- Number of teams
def games_per_pair := 2 -- Each team plays every other team twice
def points_win := 3 -- Points for a win
def points_draw := 1 -- Points for a draw
def points_loss := 0 -- Points for a loss

-- Total number of games played
def total_games := (teams * (teams - 1) / 2) * games_per_pair

-- Total points available in the tournament
def total_points := total_games * points_win

-- Given the conditions, prove that the greatest possible number of total points each of the top three teams can accumulate is 42.
theorem greatest_points_for_top_teams :
  ∃ k, (∀ A B C : ℕ, A = B ∧ B = C → A ≤ k) ∧ k = 42 :=
sorry

end greatest_points_for_top_teams_l123_123233


namespace carrots_eaten_after_dinner_l123_123370

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end carrots_eaten_after_dinner_l123_123370


namespace total_students_l123_123568

theorem total_students (x : ℕ) (h1 : 3 * x + 8 = 3 * x + 5) (h2 : 5 * (x - 1) + 3 > 3 * x + 8) : x = 6 :=
sorry

end total_students_l123_123568


namespace total_weight_correct_l123_123615

def weight_male_clothes : ℝ := 2.6
def weight_female_clothes : ℝ := 5.98
def total_weight_clothes : ℝ := weight_male_clothes + weight_female_clothes

theorem total_weight_correct : total_weight_clothes = 8.58 := by
  sorry

end total_weight_correct_l123_123615


namespace cheryl_same_color_probability_l123_123712

/-- Defines the probability of Cheryl picking 3 marbles of the same color from the given box setup. -/
def probability_cheryl_picks_same_color : ℚ :=
  let total_ways := (Nat.choose 9 3) * (Nat.choose 6 3) * (Nat.choose 3 3)
  let favorable_ways := 3 * (Nat.choose 6 3)
  (favorable_ways : ℚ) / (total_ways : ℚ)

/-- Theorem stating the probability that Cheryl picks 3 marbles of the same color is 1/28. -/
theorem cheryl_same_color_probability :
  probability_cheryl_picks_same_color = 1 / 28 :=
by
  sorry

end cheryl_same_color_probability_l123_123712


namespace num_ways_to_distribute_balls_l123_123381

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l123_123381


namespace cost_of_adult_ticket_l123_123311

theorem cost_of_adult_ticket (x : ℕ) (total_persons : ℕ) (total_collected : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) (amount_from_children : ℕ) :
  total_persons = 280 →
  total_collected = 14000 →
  adult_tickets = 200 →
  child_ticket_cost = 25 →
  amount_from_children = 2000 →
  200 * x + amount_from_children = total_collected →
  x = 60 :=
by
  intros h_persons h_total h_adults h_child_cost h_children_amount h_eq
  sorry

end cost_of_adult_ticket_l123_123311


namespace f_eq_32x5_l123_123389

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end f_eq_32x5_l123_123389


namespace smallest_percent_coffee_tea_l123_123095

theorem smallest_percent_coffee_tea (C T : ℝ) (hC : C = 50) (hT : T = 60) : 
  ∃ x, x = C + T - 100 ∧ x = 10 :=
by
  sorry

end smallest_percent_coffee_tea_l123_123095


namespace arithmetic_sequence_property_l123_123902

def arith_seq (a : ℕ → ℤ) (a1 a3 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ a 3 = a3 ∧ (a 3 - a 1) = 2 * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℤ), ∃ d : ℤ, arith_seq a 1 (-3) d →
  (1 - (a 2) - a 3 - (a 4) - (a 5) = 17) :=
by
  intros a
  use -2
  simp [arith_seq, *]
  sorry

end arithmetic_sequence_property_l123_123902


namespace parallelogram_area_l123_123755

def base := 12 -- in meters
def height := 6 -- in meters

theorem parallelogram_area : base * height = 72 := by
  sorry

end parallelogram_area_l123_123755


namespace nine_sided_polygon_diagonals_l123_123541

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l123_123541


namespace length_of_first_train_l123_123907

theorem length_of_first_train
    (speed_first_train_kmph : ℝ) 
    (speed_second_train_kmph : ℝ) 
    (time_to_cross_seconds : ℝ) 
    (length_second_train_meters : ℝ)
    (H1 : speed_first_train_kmph = 120)
    (H2 : speed_second_train_kmph = 80)
    (H3 : time_to_cross_seconds = 9)
    (H4 : length_second_train_meters = 300.04) : 
    ∃ (length_first_train : ℝ), length_first_train = 200 :=
by 
    let relative_speed_m_per_s := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
    let combined_length := relative_speed_m_per_s * time_to_cross_seconds
    let length_first_train := combined_length - length_second_train_meters
    use length_first_train
    sorry

end length_of_first_train_l123_123907


namespace calculate_abc_over_def_l123_123869

theorem calculate_abc_over_def
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  (a * b * c) / (d * e * f) = 1 / 2 :=
by
  sorry

end calculate_abc_over_def_l123_123869


namespace solve_equation_l123_123966

-- Define the equation as a Lean proposition
def equation (x : ℝ) : Prop :=
  (6 * x + 3) / (3 * x^2 + 6 * x - 9) = 3 * x / (3 * x - 3)

-- Define the solution set
def solution (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2

-- Define the condition to avoid division by zero
def valid (x : ℝ) : Prop := x ≠ 1

-- State the theorem
theorem solve_equation (x : ℝ) (h : equation x) (hv : valid x) : solution x :=
by
  sorry

end solve_equation_l123_123966


namespace mark_bench_press_correct_l123_123649

def dave_weight : ℝ := 175
def dave_bench_press : ℝ := 3 * dave_weight

def craig_bench_percentage : ℝ := 0.20
def craig_bench_press : ℝ := craig_bench_percentage * dave_bench_press

def emma_bench_percentage : ℝ := 0.75
def emma_initial_bench_press : ℝ := emma_bench_percentage * dave_bench_press
def emma_actual_bench_press : ℝ := emma_initial_bench_press + 15

def combined_craig_emma : ℝ := craig_bench_press + emma_actual_bench_press

def john_bench_factor : ℝ := 2
def john_bench_press : ℝ := john_bench_factor * combined_craig_emma

def mark_reduction : ℝ := 50
def mark_bench_press : ℝ := combined_craig_emma - mark_reduction

theorem mark_bench_press_correct : mark_bench_press = 463.75 := by
  sorry

end mark_bench_press_correct_l123_123649


namespace find_angle_A_l123_123618

theorem find_angle_A (A B : ℝ) (a b : ℝ) (h1 : b = 2 * a * Real.sin B) (h2 : a ≠ 0) :
  A = 30 ∨ A = 150 :=
by
  sorry

end find_angle_A_l123_123618


namespace triangle_PQR_area_l123_123450

/-- Given a triangle PQR where PQ = 4 miles, PR = 2 miles, and PQ is along Pine Street
and PR is along Quail Road, and there is a sub-triangle PQS within PQR
with PS = 2 miles along Summit Avenue and QS = 3 miles along Pine Street,
prove that the area of triangle PQR is 4 square miles --/
theorem triangle_PQR_area :
  ∀ (PQ PR PS QS : ℝ),
    PQ = 4 → PR = 2 → PS = 2 → QS = 3 →
    (1/2) * PQ * PR = 4 :=
by
  intros PQ PR PS QS hpq hpr hps hqs
  rw [hpq, hpr]
  norm_num
  done

end triangle_PQR_area_l123_123450


namespace matrix_A_pow_50_l123_123898

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 1],
  ![0, 1]
]

theorem matrix_A_pow_50 : A^50 = ![
  ![1, 50],
  ![0, 1]
] :=
sorry

end matrix_A_pow_50_l123_123898


namespace smallest_distance_l123_123080

open Real

/-- Let A be a point on the circle (x-3)^2 + (y-4)^2 = 16,
and let B be a point on the parabola x^2 = 8y.
The smallest possible distance AB is √34 - 4. -/
theorem smallest_distance 
  (A B : ℝ × ℝ)
  (hA : (A.1 - 3)^2 + (A.2 - 4)^2 = 16)
  (hB : (B.1)^2 = 8 * B.2) :
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ sqrt 34 - 4 := 
sorry

end smallest_distance_l123_123080


namespace car_average_speed_l123_123464

def average_speed (speed1 speed2 : ℕ) (time1 time2 : ℕ) : ℕ := 
  (speed1 * time1 + speed2 * time2) / (time1 + time2)

theorem car_average_speed :
  average_speed 60 90 (1/3) (2/3) = 80 := 
by 
  sorry

end car_average_speed_l123_123464


namespace total_books_in_class_l123_123885

theorem total_books_in_class (Tables : ℕ) (BooksPerTable : ℕ) (TotalBooks : ℕ) 
  (h1 : Tables = 500)
  (h2 : BooksPerTable = (2 * Tables) / 5)
  (h3 : TotalBooks = Tables * BooksPerTable) :
  TotalBooks = 100000 := 
sorry

end total_books_in_class_l123_123885


namespace solve_for_a_l123_123934

theorem solve_for_a (x a : ℤ) (h : x = 3) (heq : 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end solve_for_a_l123_123934


namespace twelfth_term_is_three_l123_123007

-- Define the first term and the common difference of the arithmetic sequence
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 4

-- Define the nth term of an arithmetic sequence
def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

-- Prove that the twelfth term is equal to 3
theorem twelfth_term_is_three : nth_term first_term common_difference 12 = 3 := 
  by 
    sorry

end twelfth_term_is_three_l123_123007


namespace probability_both_hit_l123_123419

-- Define the probabilities of hitting the target for shooters A and B.
def prob_A_hits : ℝ := 0.7
def prob_B_hits : ℝ := 0.8

-- Define the independence condition (not needed as a direct definition but implicitly acknowledges independence).
axiom A_and_B_independent : true

-- The statement we want to prove: the probability that both shooters hit the target.
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.56 :=
by
  -- Placeholder for proof
  sorry

end probability_both_hit_l123_123419


namespace similarity_transformation_result_l123_123931

-- Define the original coordinates of point A and the similarity ratio
def A : ℝ × ℝ := (2, 2)
def ratio : ℝ := 2

-- Define the similarity transformation that scales coordinates, optionally considering reflection
def similarity_transform (p : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  (r * p.1, r * p.2)

-- Use Lean to state the theorem based on the given conditions and expected answer
theorem similarity_transformation_result :
  similarity_transform A ratio = (4, 4) ∨ similarity_transform A (-ratio) = (-4, -4) :=
by
  sorry

end similarity_transformation_result_l123_123931


namespace arithmetic_sequence_50th_term_l123_123941

-- Definitions as per the conditions
def a_1 : ℤ := 48
def d : ℤ := -2
def n : ℕ := 50

-- Statement to prove the 50th term in the series
theorem arithmetic_sequence_50th_term : a_1 + (n - 1) * d = -50 :=
by
  sorry

end arithmetic_sequence_50th_term_l123_123941


namespace find_cos_7theta_l123_123739

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l123_123739


namespace toy_production_difference_l123_123707

variables (w t : ℕ)
variable  (t_nonneg : 0 < t) -- assuming t is always non-negative for a valid working hour.
variable  (h : w = 3 * t)

theorem toy_production_difference : 
  (w * t) - ((w + 5) * (t - 3)) = 4 * t + 15 :=
by
  sorry

end toy_production_difference_l123_123707


namespace solve_for_m_l123_123656

theorem solve_for_m {m : ℝ} (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 :=
sorry

end solve_for_m_l123_123656


namespace transform_expression_to_product_l123_123901

variables (a b c d s: ℝ)

theorem transform_expression_to_product
  (h1 : d = a + b + c)
  (h2 : s = (a + b + c + d) / 2) :
    2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) -
    (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 16 * (s - a) * (s - b) * (s - c) * (s - d) :=
by
  sorry

end transform_expression_to_product_l123_123901


namespace yellow_red_chair_ratio_l123_123064

variable (Y B : ℕ)
variable (red_chairs : ℕ := 5)
variable (total_chairs : ℕ := 43)

-- Condition: There are 2 fewer blue chairs than yellow chairs
def blue_chairs_condition : Prop := B = Y - 2

-- Condition: Total number of chairs
def total_chairs_condition : Prop := red_chairs + Y + B = total_chairs

-- Prove the ratio of yellow chairs to red chairs is 4:1
theorem yellow_red_chair_ratio (h1 : blue_chairs_condition Y B) (h2 : total_chairs_condition Y B) :
  (Y / red_chairs) = 4 := 
sorry

end yellow_red_chair_ratio_l123_123064


namespace difference_students_pets_in_all_classrooms_l123_123164

-- Definitions of the conditions
def students_per_classroom : ℕ := 24
def rabbits_per_classroom : ℕ := 3
def guinea_pigs_per_classroom : ℕ := 2
def number_of_classrooms : ℕ := 5

-- Proof problem statement
theorem difference_students_pets_in_all_classrooms :
  (students_per_classroom * number_of_classrooms) - 
  ((rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms) = 95 := by
  sorry

end difference_students_pets_in_all_classrooms_l123_123164


namespace select_books_from_corner_l123_123117

def num_ways_to_select_books (n₁ n₂ k : ℕ) : ℕ :=
  if h₁ : k > n₁ ∧ k > n₂ then 0
  else if h₂ : k > n₂ then 1
  else if h₃ : k > n₁ then Nat.choose n₂ k
  else Nat.choose n₁ k + 2 * Nat.choose n₁ (k-1) * Nat.choose n₂ 1 + Nat.choose n₁ k * 0 +
    (Nat.choose n₂ 1 * Nat.choose n₂ (k-1)) + Nat.choose n₂ k * 1

theorem select_books_from_corner :
  num_ways_to_select_books 3 6 3 = 42 :=
by
  sorry

end select_books_from_corner_l123_123117


namespace find_m_l123_123763

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end find_m_l123_123763


namespace angle_supplement_complement_l123_123676

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l123_123676


namespace rowing_speed_upstream_l123_123498

theorem rowing_speed_upstream (V_s V_downstream : ℝ) (V_s_eq : V_s = 28) (V_downstream_eq : V_downstream = 31) : 
  V_s - (V_downstream - V_s) = 25 := 
by
  sorry

end rowing_speed_upstream_l123_123498


namespace cos_expression_value_l123_123472

theorem cos_expression_value (x : ℝ) (h : Real.sin x = 3 * Real.sin (x - Real.pi / 2)) :
  Real.cos x * Real.cos (x + Real.pi / 2) = 3 / 10 := 
sorry

end cos_expression_value_l123_123472


namespace the_inequality_l123_123661

theorem the_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a / (1 + b)) + (b / (1 + c)) + (c / (1 + a)) ≥ 3 / 2 :=
by sorry

end the_inequality_l123_123661


namespace max_projection_sum_l123_123627

-- Define the given conditions
def edge_length : ℝ := 2

def projection_front_view (length : ℝ) : Prop := length = edge_length
def projection_side_view (length : ℝ) : Prop := ∃ a : ℝ, a = length
def projection_top_view (length : ℝ) : Prop := ∃ b : ℝ, b = length

-- State the theorem
theorem max_projection_sum (a b : ℝ) (ha : projection_side_view a) (hb : projection_top_view b) :
  a + b ≤ 4 := sorry

end max_projection_sum_l123_123627


namespace equation_of_plane_passing_through_points_l123_123663

/-
Let M1, M2, and M3 be points in three-dimensional space.
M1 = (1, 2, 0)
M2 = (1, -1, 2)
M3 = (0, 1, -1)
We need to prove that the plane passing through these points has the equation 5x - 2y - 3z - 1 = 0.
-/

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨1, -1, 2⟩
def M3 : Point3D := ⟨0, 1, -1⟩

theorem equation_of_plane_passing_through_points :
  ∃ (a b c d : ℝ), (∀ (P : Point3D), 
  P = M1 ∨ P = M2 ∨ P = M3 → a * P.x + b * P.y + c * P.z + d = 0)
  ∧ a = 5 ∧ b = -2 ∧ c = -3 ∧ d = -1 :=
by
  sorry

end equation_of_plane_passing_through_points_l123_123663


namespace min_value_of_A_l123_123793

noncomputable def A (a b c : ℝ) : ℝ :=
  (a^3 + b^3) / (8 * a * b + 9 - c^2) +
  (b^3 + c^3) / (8 * b * c + 9 - a^2) +
  (c^3 + a^3) / (8 * c * a + 9 - b^2)

theorem min_value_of_A (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  A a b c = 3 / 8 :=
sorry

end min_value_of_A_l123_123793


namespace jack_needs_more_money_l123_123592

/--
Jack is a soccer player. He needs to buy two pairs of socks, a pair of soccer shoes, a soccer ball, and a sports bag.
Each pair of socks costs $12.75, the shoes cost $145, the soccer ball costs $38, and the sports bag costs $47.
Jack has a 5% discount coupon for the shoes and a 10% discount coupon for the sports bag.
He currently has $25. How much more money does Jack need to buy all the items?
-/
theorem jack_needs_more_money :
  let socks_cost : ℝ := 12.75
  let shoes_cost : ℝ := 145
  let ball_cost : ℝ := 38
  let bag_cost : ℝ := 47
  let shoes_discount : ℝ := 0.05
  let bag_discount : ℝ := 0.10
  let money_jack_has : ℝ := 25
  let total_cost := 2 * socks_cost + (shoes_cost - shoes_cost * shoes_discount) + ball_cost + (bag_cost - bag_cost * bag_discount)
  total_cost - money_jack_has = 218.55 :=
by
  sorry

end jack_needs_more_money_l123_123592


namespace like_terms_sum_l123_123874

theorem like_terms_sum (n m : ℕ) 
  (h1 : n + 1 = 3) 
  (h2 : m - 1 = 3) : 
  m + n = 6 := 
  sorry

end like_terms_sum_l123_123874


namespace perpendicular_lines_l123_123640

theorem perpendicular_lines (m : ℝ) :
  (∃ k l : ℝ, k * m + (1 - m) * l = 3 ∧ (m - 1) * k + (2 * m + 3) * l = 2) → m = -3 ∨ m = 1 :=
by sorry

end perpendicular_lines_l123_123640


namespace mariel_dogs_count_l123_123557

theorem mariel_dogs_count (total_legs : ℤ) (num_dog_walkers : ℤ) (legs_per_walker : ℤ) 
  (other_dogs_count : ℤ) (legs_per_dog : ℤ) (mariel_dogs : ℤ) :
  total_legs = 36 →
  num_dog_walkers = 2 →
  legs_per_walker = 2 →
  other_dogs_count = 3 →
  legs_per_dog = 4 →
  mariel_dogs = (total_legs - (num_dog_walkers * legs_per_walker + other_dogs_count * legs_per_dog)) / legs_per_dog →
  mariel_dogs = 5 :=
by
  intros
  sorry

end mariel_dogs_count_l123_123557


namespace find_integer_n_l123_123992

theorem find_integer_n (n : ℤ) (h1 : n ≥ 3) (h2 : ∃ k : ℚ, k * k = (n^2 - 5) / (n + 1)) : n = 3 := by
  sorry

end find_integer_n_l123_123992


namespace option_D_correct_l123_123416

theorem option_D_correct (a b : ℝ) : -a * b + 3 * b * a = 2 * a * b :=
by sorry

end option_D_correct_l123_123416


namespace area_spot_can_reach_l123_123593

noncomputable def area_reachable_by_spot (s : ℝ) (r : ℝ) : ℝ := 
  if s = 1 ∧ r = 3 then 6.5 * Real.pi else 0

theorem area_spot_can_reach : area_reachable_by_spot 1 3 = 6.5 * Real.pi :=
by
  -- The theorem proof should go here.
  sorry

end area_spot_can_reach_l123_123593


namespace smallest_positive_multiple_of_32_l123_123972

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end smallest_positive_multiple_of_32_l123_123972


namespace part1_part2_l123_123057

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem part1 (a x : ℝ):
  a ≥ 1 → x > 0 → f x a ≥ 0 := 
sorry

theorem part2 (a : ℝ):
  0 < a ∧ a ≤ 2 / 3 → ∃! x, x > -a ∧ f x a = 0 :=
sorry

end part1_part2_l123_123057


namespace seashells_total_correct_l123_123247

def total_seashells (red_shells green_shells other_shells : ℕ) : ℕ :=
  red_shells + green_shells + other_shells

theorem seashells_total_correct :
  total_seashells 76 49 166 = 291 :=
by
  sorry

end seashells_total_correct_l123_123247


namespace remaining_pieces_to_fold_l123_123345

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l123_123345


namespace sum_of_square_face_is_13_l123_123718

-- Definitions based on conditions
variables (x₁ x₂ x₃ x₄ x₅ : ℕ)

-- Conditions
axiom h₁ : x₁ + x₂ + x₃ = 7
axiom h₂ : x₁ + x₂ + x₄ = 8
axiom h₃ : x₁ + x₃ + x₄ = 9
axiom h₄ : x₂ + x₃ + x₄ = 10

-- Properties
axiom h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15

-- Goal to prove
theorem sum_of_square_face_is_13 (h₁ : x₁ + x₂ + x₃ = 7) (h₂ : x₁ + x₂ + x₄ = 8) 
  (h₃ : x₁ + x₃ + x₄ = 9) (h₄ : x₂ + x₃ + x₄ = 10) (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15): 
  x₅ + x₁ + x₂ + x₄ = 13 :=
sorry

end sum_of_square_face_is_13_l123_123718


namespace intersection_of_N_and_not_R_M_l123_123807

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def Not_R_M : Set ℝ := {x | x ≤ 2}

theorem intersection_of_N_and_not_R_M : 
  N ∩ Not_R_M = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_N_and_not_R_M_l123_123807


namespace total_games_l123_123772

variable (G R : ℕ)

axiom cond1 : 85 + (1/2 : ℚ) * R = (0.70 : ℚ) * G
axiom cond2 : G = 100 + R

theorem total_games : G = 175 := by
  sorry

end total_games_l123_123772


namespace max_hot_dogs_with_300_dollars_l123_123741

def num_hot_dogs (dollars : ℕ) 
  (cost_8 : ℚ) (count_8 : ℕ) 
  (cost_20 : ℚ) (count_20 : ℕ)
  (cost_250 : ℚ) (count_250 : ℕ) : ℕ :=
  sorry

theorem max_hot_dogs_with_300_dollars : 
  num_hot_dogs 300 1.55 8 3.05 20 22.95 250 = 3258 :=
sorry

end max_hot_dogs_with_300_dollars_l123_123741


namespace student_B_more_stable_l123_123500

-- Definitions as stated in the conditions
def student_A_variance : ℝ := 0.3
def student_B_variance : ℝ := 0.1

-- Theorem stating that student B has more stable performance than student A
theorem student_B_more_stable : student_B_variance < student_A_variance :=
by
  sorry

end student_B_more_stable_l123_123500


namespace ratio_of_p_q_l123_123322

theorem ratio_of_p_q (b : ℝ) (p q : ℝ) (h1 : p = -b / 8) (h2 : q = -b / 12) : p / q = 3 / 2 := 
by
  sorry

end ratio_of_p_q_l123_123322


namespace trapezoid_area_l123_123764

theorem trapezoid_area
  (A B C D : ℝ)
  (BC AD AC : ℝ)
  (radius circle_center : ℝ)
  (h : ℝ)
  (angleBAD angleADC : ℝ)
  (tangency : Bool) :
  BC = 13 → 
  angleBAD = 2 * angleADC →
  radius = 5 →
  tangency = true →
  1/2 * (BC + AD) * h = 157.5 :=
by
  sorry

end trapezoid_area_l123_123764


namespace range_of_k_in_first_quadrant_l123_123681

theorem range_of_k_in_first_quadrant (k : ℝ) (h₁ : k ≠ -1) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x + y - 1 = 0 ∧ x > 0 ∧ y > 0) ↔ 1 < k := by sorry

end range_of_k_in_first_quadrant_l123_123681


namespace range_of_a_iff_l123_123362

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x → (Real.log x / Real.log a) ≤ x ∧ x ≤ a ^ x

theorem range_of_a_iff (a : ℝ) : (a ≥ Real.exp (Real.exp (-1))) ↔ range_of_a a :=
by
  sorry

end range_of_a_iff_l123_123362


namespace marla_parent_teacher_night_time_l123_123331

def errand_time := 110 -- total minutes on the errand
def driving_time_oneway := 20 -- minutes driving one way to school
def driving_time_return := 20 -- minutes driving one way back home

def total_driving_time := driving_time_oneway + driving_time_return

def time_at_parent_teacher_night := errand_time - total_driving_time

theorem marla_parent_teacher_night_time : time_at_parent_teacher_night = 70 :=
by
  -- Lean proof goes here
  sorry

end marla_parent_teacher_night_time_l123_123331


namespace smallest_multiple_of_4_and_14_is_28_l123_123421

theorem smallest_multiple_of_4_and_14_is_28 :
  ∃ (a : ℕ), a > 0 ∧ (4 ∣ a) ∧ (14 ∣ a) ∧ ∀ b : ℕ, b > 0 → (4 ∣ b) → (14 ∣ b) → a ≤ b := 
sorry

end smallest_multiple_of_4_and_14_is_28_l123_123421


namespace range_of_values_l123_123449

variable {f : ℝ → ℝ}

-- Conditions and given data
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

def is_monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f (x) ≤ f (y)

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (Real.log a / Real.log 2) + f (-Real.log a / Real.log 2) ≤ 2 * f (1)

-- The goal
theorem range_of_values (h1 : is_even f) (h2 : is_monotone_on_nonneg f) (a : ℝ) (h3 : condition f a) :
  1/2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_values_l123_123449


namespace exists_nat_n_l123_123511

theorem exists_nat_n (l : ℕ) (hl : l > 0) : ∃ n : ℕ, n^n + 47 ≡ 0 [MOD 2^l] := by
  sorry

end exists_nat_n_l123_123511


namespace proof_problem_l123_123103

theorem proof_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) :
  (a < 1 → b > 2) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y - 1 / (2 * x) - 2 / y = 3 / 2 → x + y ≥ 3) :=
by
  sorry

end proof_problem_l123_123103


namespace sum_two_integers_l123_123387

theorem sum_two_integers (a b : ℤ) (h1 : a = 17) (h2 : b = 19) : a + b = 36 := by
  sorry

end sum_two_integers_l123_123387


namespace not_possible_to_list_numbers_l123_123384

theorem not_possible_to_list_numbers :
  ¬ (∃ (f : ℕ → ℕ), (∀ n, f n ≥ 1 ∧ f n ≤ 1963) ∧
                     (∀ n, Nat.gcd (f n) (f (n+1)) = 1) ∧
                     (∀ n, Nat.gcd (f n) (f (n+2)) = 1)) :=
by
  sorry

end not_possible_to_list_numbers_l123_123384


namespace smallest_n_for_cubic_sum_inequality_l123_123762

theorem smallest_n_for_cubic_sum_inequality :
  ∃ n : ℕ, (∀ (a b c : ℕ), (a + b + c) ^ 3 ≤ n * (a ^ 3 + b ^ 3 + c ^ 3)) ∧ n = 9 :=
sorry

end smallest_n_for_cubic_sum_inequality_l123_123762


namespace monomial_sum_l123_123197

variable {x y : ℝ}

theorem monomial_sum (a : ℝ) (h : -2 * x^2 * y^3 + 5 * x^(a-1) * y^3 = c * x^k * y^3) : a = 3 :=
  by
  sorry

end monomial_sum_l123_123197


namespace part_a_part_b_part_c_l123_123275

theorem part_a (θ : ℝ) (m : ℕ) : |Real.sin (m * θ)| ≤ m * |Real.sin θ| :=
sorry

theorem part_b (θ₁ θ₂ : ℝ) (m : ℕ) (hm_even : Even m) : 
  |Real.sin (m * θ₂) - Real.sin (m * θ₁)| ≤ m * |Real.sin (θ₂ - θ₁)| :=
sorry

theorem part_c (m : ℕ) (hm_odd : Odd m) : 
  ∃ θ₁ θ₂ : ℝ, |Real.sin (m * θ₂) - Real.sin (m * θ₁)| > m * |Real.sin (θ₂ - θ₁)| :=
sorry

end part_a_part_b_part_c_l123_123275


namespace einstein_birth_weekday_l123_123454

-- Defining the reference day of the week for 31 May 2006
def reference_date := 31
def reference_month := 5
def reference_year := 2006
def reference_weekday := 3  -- Wednesday

-- Defining Albert Einstein's birth date
def einstein_birth_day := 14
def einstein_birth_month := 3
def einstein_birth_year := 1879

-- Defining the calculation of weekday
def weekday_from_reference(reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year : Nat) : Nat :=
  let days_from_reference_to_birth := 46464  -- Total days calculated in solution
  (reference_weekday - (days_from_reference_to_birth % 7) + 7) % 7

-- Stating the theorem
theorem einstein_birth_weekday : weekday_from_reference reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year = 5 :=
by
  -- Proof omitted
  sorry

end einstein_birth_weekday_l123_123454


namespace initial_ratio_milk_water_l123_123501

-- Define the initial conditions
variables (M W : ℕ) (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4)

-- State the theorem to prove the initial ratio of milk to water
theorem initial_ratio_milk_water (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4) :
  (M * 2 = W * 3) :=
by
  sorry

end initial_ratio_milk_water_l123_123501


namespace circle_has_greatest_symmetry_l123_123427

-- Definitions based on the conditions
def lines_of_symmetry (figure : String) : ℕ∞ := 
  match figure with
  | "regular pentagon" => 5
  | "isosceles triangle" => 1
  | "circle" => ⊤  -- Using the symbol ⊤ to represent infinity in Lean.
  | "rectangle" => 2
  | "parallelogram" => 0
  | _ => 0          -- default case

theorem circle_has_greatest_symmetry :
  ∃ fig, fig = "circle" ∧ ∀ other_fig, lines_of_symmetry fig ≥ lines_of_symmetry other_fig := 
by
  sorry

end circle_has_greatest_symmetry_l123_123427


namespace only_set_B_is_right_angle_triangle_l123_123071

def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem only_set_B_is_right_angle_triangle :
  is_right_angle_triangle 3 4 5 ∧ ¬is_right_angle_triangle 1 2 2 ∧ ¬is_right_angle_triangle 3 4 9 ∧ ¬is_right_angle_triangle 4 5 7 :=
by
  -- proof steps omitted
  sorry

end only_set_B_is_right_angle_triangle_l123_123071


namespace overlap_difference_l123_123977

namespace GeometryBiology

noncomputable def total_students : ℕ := 350
noncomputable def geometry_students : ℕ := 210
noncomputable def biology_students : ℕ := 175

theorem overlap_difference : 
    let max_overlap := min geometry_students biology_students;
    let min_overlap := geometry_students + biology_students - total_students;
    max_overlap - min_overlap = 140 := 
by
  sorry

end GeometryBiology

end overlap_difference_l123_123977


namespace person_A_work_days_l123_123028

theorem person_A_work_days (A : ℕ) (h1 : ∀ (B : ℕ), B = 45) (h2 : 4 * (1/A + 1/45) = 2/9) : A = 30 := 
by
  sorry

end person_A_work_days_l123_123028


namespace tail_length_l123_123896

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end tail_length_l123_123896


namespace externally_tangent_circles_proof_l123_123339

noncomputable def externally_tangent_circles (r r' : ℝ) (φ : ℝ) : Prop :=
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r')

theorem externally_tangent_circles_proof (r r' φ : ℝ) 
  (h1: r > 0) (h2: r' > 0) (h3: φ ≥ 0 ∧ φ ≤ π) : 
  externally_tangent_circles r r' φ :=
sorry

end externally_tangent_circles_proof_l123_123339


namespace find_number_l123_123555

theorem find_number (n p q : ℝ) (h1 : n / p = 6) (h2 : n / q = 15) (h3 : p - q = 0.3) : n = 3 :=
by
  sorry

end find_number_l123_123555


namespace four_digit_sum_of_digits_divisible_by_101_l123_123903

theorem four_digit_sum_of_digits_divisible_by_101 (a b c d : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 1 ≤ b ∧ b ≤ 9)
  (h3 : 1 ≤ c ∧ c ≤ 9)
  (h4 : 1 ≤ d ∧ d ≤ 9)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_div : (1001 * a + 110 * b + 110 * c + 1001 * d) % 101 = 0) :
  (a + d) % 101 = (b + c) % 101 :=
by
  sorry

end four_digit_sum_of_digits_divisible_by_101_l123_123903


namespace alpha_inverse_proportional_beta_l123_123993

theorem alpha_inverse_proportional_beta (α β : ℝ) (k : ℝ) :
  (∀ β1 α1, α1 * β1 = k) → (4 * 2 = k) → (β = -3) → (α = -8/3) :=
by
  sorry

end alpha_inverse_proportional_beta_l123_123993


namespace fifth_hexagon_dots_l123_123385

-- Definitions as per conditions
def dots_in_nth_layer (n : ℕ) : ℕ := 6 * (n + 2)

-- Function to calculate the total number of dots in the nth hexagon
def total_dots_in_hexagon (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + dots_in_nth_layer k) (dots_in_nth_layer 0)

-- The proof problem statement
theorem fifth_hexagon_dots : total_dots_in_hexagon 5 = 150 := sorry

end fifth_hexagon_dots_l123_123385


namespace replace_question_with_division_l123_123284

theorem replace_question_with_division :
  ∃ op: (ℤ → ℤ → ℤ), (op 8 2) + 5 - (3 - 2) = 8 ∧ 
  (∀ a b, op = Int.div ∧ ((op a b) = a / b)) :=
by
  sorry

end replace_question_with_division_l123_123284


namespace find_C_equation_l123_123509

def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def N : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]

def C2_equation (x y : ℝ) : Prop := y = (1/8) * x^2

theorem find_C_equation (x y : ℝ) :
  (C2_equation (x) y) → (y^2 = 2 * x) := 
sorry

end find_C_equation_l123_123509


namespace student_weekly_allowance_l123_123023

theorem student_weekly_allowance (A : ℝ) :
  (3 / 4) * (1 / 3) * ((2 / 5) * A + 4) - 2 = 0 ↔ A = 100/3 := sorry

end student_weekly_allowance_l123_123023


namespace number_of_convex_quadrilaterals_with_parallel_sides_l123_123248

-- Define a regular 20-sided polygon
def regular_20_sided_polygon : Type := 
  { p : ℕ // 0 < p ∧ p ≤ 20 }

-- The main theorem statement
theorem number_of_convex_quadrilaterals_with_parallel_sides : 
  ∃ (n : ℕ), n = 765 :=
sorry

end number_of_convex_quadrilaterals_with_parallel_sides_l123_123248


namespace round_trip_time_l123_123433

variable (boat_speed standing_water_speed stream_speed distance : ℕ)

theorem round_trip_time (boat_speed := 9) (stream_speed := 6) (distance := 170) : 
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed)) = 68 := by 
  sorry

end round_trip_time_l123_123433


namespace solve_system_l123_123729

variable {a b c : ℝ}
variable {x y z : ℝ}
variable {e1 e2 e3 : ℤ} -- Sign variables should be integers to express ±1 more easily 

axiom ax1 : x * (x + y) + z * (x - y) = a
axiom ax2 : y * (y + z) + x * (y - z) = b
axiom ax3 : z * (z + x) + y * (z - x) = c

theorem solve_system :
  (e1 = 1 ∨ e1 = -1) ∧ (e2 = 1 ∨ e2 = -1) ∧ (e3 = 1 ∨ e3 = -1) →
  x = (1/2) * (e1 * Real.sqrt (a + b) - e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) ∧
  y = (1/2) * (e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) - e3 * Real.sqrt (c + a)) ∧
  z = (1/2) * (-e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) :=
sorry -- proof goes here

end solve_system_l123_123729


namespace sum_of_divisors_85_l123_123473

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end sum_of_divisors_85_l123_123473


namespace added_amount_l123_123444

theorem added_amount (x y : ℕ) (h1 : x = 17) (h2 : 3 * (2 * x + y) = 117) : y = 5 :=
by
  sorry

end added_amount_l123_123444


namespace eq_condition_l123_123447

theorem eq_condition (a : ℝ) :
  (∃ x : ℝ, a * (4 * |x| + 1) = 4 * |x|) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end eq_condition_l123_123447


namespace rohan_monthly_salary_l123_123040

theorem rohan_monthly_salary (s : ℝ) 
  (h_food : s * 0.40 = f)
  (h_rent : s * 0.20 = hr) 
  (h_entertainment : s * 0.10 = e)
  (h_conveyance : s * 0.10 = c)
  (h_savings : s * 0.20 = 1000) : 
  s = 5000 := 
sorry

end rohan_monthly_salary_l123_123040


namespace quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l123_123397

theorem quadratic_real_roots_discriminant (m : ℝ) :
  (2 * (m + 1))^2 - 4 * m * (m - 1) > 0 ↔ (m > -1/2 ∧ m ≠ 0) := 
sorry

theorem quadratic_real_roots_sum_of_squares (m x1 x2 : ℝ) 
  (h1 : m > -1/2 ∧ m ≠ 0)
  (h2 : x1 + x2 = -2 * (m + 1) / m)
  (h3 : x1 * x2 = (m - 1) / m)
  (h4 : x1^2 + x2^2 = 8) : 
  m = (6 + 2 * Real.sqrt 33) / 8 := 
sorry

end quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l123_123397


namespace solve_first_system_solve_second_system_solve_third_system_l123_123239

-- First system of equations
theorem solve_first_system (x y : ℝ) 
  (h1 : 2*x + 3*y = 16)
  (h2 : x + 4*y = 13) : 
  x = 5 ∧ y = 2 := 
sorry

-- Second system of equations
theorem solve_second_system (x y : ℝ) 
  (h1 : 0.3*x - y = 1)
  (h2 : 0.2*x - 0.5*y = 19) : 
  x = 370 ∧ y = 110 := 
sorry

-- Third system of equations
theorem solve_third_system (x y : ℝ) 
  (h1 : 3 * (x - 1) = y + 5)
  (h2 : (x + 2) / 2 = ((y - 1) / 3) + 1) : 
  x = 6 ∧ y = 10 := 
sorry

end solve_first_system_solve_second_system_solve_third_system_l123_123239


namespace firing_sequence_hits_submarine_l123_123121

theorem firing_sequence_hits_submarine (a b : ℕ) (hb : b > 0) : ∃ n : ℕ, (∃ (an bn : ℕ), (an + bn * n) = a + n * b) :=
sorry

end firing_sequence_hits_submarine_l123_123121


namespace total_fencing_l123_123973

open Real

def playground_side_length : ℝ := 27
def garden_length : ℝ := 12
def garden_width : ℝ := 9
def flower_bed_radius : ℝ := 5
def sandpit_side1 : ℝ := 7
def sandpit_side2 : ℝ := 10
def sandpit_side3 : ℝ := 13

theorem total_fencing : 
    4 * playground_side_length + 
    2 * (garden_length + garden_width) + 
    2 * Real.pi * flower_bed_radius + 
    (sandpit_side1 + sandpit_side2 + sandpit_side3) = 211.42 := 
    by sorry

end total_fencing_l123_123973


namespace isosceles_triangle_perimeter_l123_123648

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4)
  (h3 : a = b ∨ 2 * a > b) :
  (a ≠ b ∨ b = 2 * a) → 
  ∃ p : ℝ, p = a + b + b ∧ p = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l123_123648


namespace syllogism_correct_l123_123424

-- Define that natural numbers are integers
axiom nat_is_int : ∀ (n : ℕ), ∃ (m : ℤ), m = n

-- Define that 4 is a natural number
axiom four_is_nat : ∃ (n : ℕ), n = 4

-- The syllogism's conclusion: 4 is an integer
theorem syllogism_correct : ∃ (m : ℤ), m = 4 :=
by
  have h1 := nat_is_int 4
  have h2 := four_is_nat
  exact h1

end syllogism_correct_l123_123424


namespace cookout_2006_kids_l123_123077

def kids_2004 : ℕ := 60
def kids_2005 : ℕ := kids_2004 / 2
def kids_2006 : ℕ := (2 * kids_2005) / 3

theorem cookout_2006_kids : kids_2006 = 20 := by
  sorry

end cookout_2006_kids_l123_123077


namespace least_positive_integer_addition_l123_123891

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end least_positive_integer_addition_l123_123891


namespace f_neg_val_is_minus_10_l123_123604
-- Import the necessary Lean library

-- Define the function f with the given conditions
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 3

-- Define the specific values
def x_val : ℝ := 2023
def x_neg_val : ℝ := -2023
def f_pos_val : ℝ := 16

-- Theorem to prove
theorem f_neg_val_is_minus_10 (a b : ℝ)
  (h : f a b x_val = f_pos_val) : 
  f a b x_neg_val = -10 :=
by
  -- Sorry placeholder for proof
  sorry

end f_neg_val_is_minus_10_l123_123604


namespace opposite_of_3_l123_123471

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l123_123471


namespace probability_of_nonzero_product_probability_of_valid_dice_values_l123_123300

def dice_values := {x : ℕ | 1 ≤ x ∧ x ≤ 6}

def valid_dice_values := {x : ℕ | 2 ≤ x ∧ x ≤ 6}

noncomputable def probability_no_one : ℚ := 625 / 1296

theorem probability_of_nonzero_product (a b c d : ℕ) 
  (ha : a ∈ dice_values) (hb : b ∈ dice_values) 
  (hc : c ∈ dice_values) (hd : d ∈ dice_values) : 
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  (a ∈ valid_dice_values ∧ b ∈ valid_dice_values ∧ 
   c ∈ valid_dice_values ∧ d ∈ valid_dice_values) :=
sorry

theorem probability_of_valid_dice_values : 
  probability_no_one = (5 / 6) ^ 4 :=
sorry

end probability_of_nonzero_product_probability_of_valid_dice_values_l123_123300


namespace problem1_problem2_problem3_problem4_l123_123048

theorem problem1 : -16 - (-12) - 24 + 18 = -10 := 
by
  sorry

theorem problem2 : 0.125 + (1 / 4) + (-9 / 4) + (-0.25) = -2 := 
by
  sorry

theorem problem3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := 
by
  sorry

theorem problem4 : (-2 + 3) * 3 - (-2)^3 / 4 = 5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l123_123048


namespace range_of_a_l123_123669

theorem range_of_a (a : ℝ) :
  (∃ M : ℝ × ℝ, (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧ (M.1)^2 + (M.2 + 3)^2 = 4 * ((M.1)^2 + (M.2)^2))
  → 0 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_l123_123669


namespace three_digit_largest_fill_four_digit_smallest_fill_l123_123560

theorem three_digit_largest_fill (n : ℕ) (h1 : n * 1000 + 28 * 4 < 1000) : n ≤ 2 := sorry

theorem four_digit_smallest_fill (n : ℕ) (h2 : n * 1000 + 28 * 4 ≥ 1000) : 3 ≤ n := sorry

end three_digit_largest_fill_four_digit_smallest_fill_l123_123560


namespace azure_valley_skirts_l123_123183

variables (P S A : ℕ)

theorem azure_valley_skirts (h1 : P = 10) 
                           (h2 : P = S / 4) 
                           (h3 : S = 2 * A / 3) : 
  A = 60 :=
by sorry

end azure_valley_skirts_l123_123183


namespace algebraic_expression_value_l123_123492

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 - 5 * a + 2 = 0) (h3 : b^2 - 5 * b + 2 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13 / 2 := by
  sorry

end algebraic_expression_value_l123_123492


namespace revenue_difference_l123_123436

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_l123_123436


namespace height_difference_after_3_years_l123_123124

/-- Conditions for the tree's and boy's growth rates per season. --/
def tree_spring_growth : ℕ := 4
def tree_summer_growth : ℕ := 6
def tree_fall_growth : ℕ := 2
def tree_winter_growth : ℕ := 1

def boy_spring_growth : ℕ := 2
def boy_summer_growth : ℕ := 2
def boy_fall_growth : ℕ := 0
def boy_winter_growth : ℕ := 0

/-- Initial heights. --/
def initial_tree_height : ℕ := 16
def initial_boy_height : ℕ := 24

/-- Length of each season in months. --/
def season_length : ℕ := 3

/-- Time period in years. --/
def years : ℕ := 3

/-- Prove the height difference between the tree and the boy after 3 years is 73 inches. --/
theorem height_difference_after_3_years :
    let tree_annual_growth := tree_spring_growth * season_length +
                             tree_summer_growth * season_length +
                             tree_fall_growth * season_length +
                             tree_winter_growth * season_length
    let tree_final_height := initial_tree_height + tree_annual_growth * years
    let boy_annual_growth := boy_spring_growth * season_length +
                            boy_summer_growth * season_length +
                            boy_fall_growth * season_length +
                            boy_winter_growth * season_length
    let boy_final_height := initial_boy_height + boy_annual_growth * years
    tree_final_height - boy_final_height = 73 :=
by sorry

end height_difference_after_3_years_l123_123124


namespace compute_expression_l123_123483

theorem compute_expression :
  -9 * 5 - (-(7 * -2)) + (-(11 * -6)) = 7 :=
by
  sorry

end compute_expression_l123_123483


namespace power_func_passes_point_l123_123135

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_func_passes_point (f : ℝ → ℝ) (h : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) 
  (h_point : f 9 = 1 / 3) : f 25 = 1 / 5 :=
sorry

end power_func_passes_point_l123_123135


namespace negation_correct_l123_123052

def original_statement (x : ℝ) : Prop := x > 0 → x^2 + 3 * x - 2 > 0

def negated_statement (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 2 ≤ 0

theorem negation_correct : (¬ ∀ x, original_statement x) ↔ ∃ x, negated_statement x := by
  sorry

end negation_correct_l123_123052


namespace parabola_vertex_l123_123056

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end parabola_vertex_l123_123056


namespace athlete_A_most_stable_l123_123722

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end athlete_A_most_stable_l123_123722


namespace unique_real_function_l123_123798

theorem unique_real_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, (f (x * y) / 2 + f (x * z) / 2 - f x * f (y * z)) ≥ 1 / 4) →
  (∀ x : ℝ, f x = 1 / 2) :=
by
  intro h
  -- proof steps go here
  sorry

end unique_real_function_l123_123798


namespace plums_for_20_oranges_l123_123044

noncomputable def oranges_to_pears (oranges : ℕ) : ℕ :=
  (oranges / 5) * 3

noncomputable def pears_to_plums (pears : ℕ) : ℕ :=
  (pears / 4) * 6

theorem plums_for_20_oranges :
  oranges_to_pears 20 = 12 ∧ pears_to_plums 12 = 18 :=
by
  sorry

end plums_for_20_oranges_l123_123044


namespace graph_not_in_second_quadrant_l123_123132

theorem graph_not_in_second_quadrant (b : ℝ) (h : ∀ x < 0, 2^x + b - 1 < 0) : b ≤ 0 :=
sorry

end graph_not_in_second_quadrant_l123_123132


namespace find_a9_l123_123263

variable (a : ℕ → ℝ)

theorem find_a9 (h1 : a 4 - a 2 = -2) (h2 : a 7 = -3) : a 9 = -5 :=
sorry

end find_a9_l123_123263


namespace xyz_inequality_l123_123647

theorem xyz_inequality (x y z : ℝ) (h : x + y + z > 0) : x^3 + y^3 + z^3 > 3 * x * y * z :=
by
  sorry

end xyz_inequality_l123_123647


namespace maria_bought_9_hardcover_volumes_l123_123022

def total_volumes (h p : ℕ) : Prop := h + p = 15
def total_cost (h p : ℕ) : Prop := 10 * p + 30 * h = 330

theorem maria_bought_9_hardcover_volumes (h p : ℕ) (h_vol : total_volumes h p) (h_cost : total_cost h p) : h = 9 :=
by
  sorry

end maria_bought_9_hardcover_volumes_l123_123022


namespace ming_dynasty_wine_problem_l123_123278

theorem ming_dynasty_wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + y / 3 = 33 ) : 
  (x = 10 ∧ y = 9) :=
by {
  sorry
}

end ming_dynasty_wine_problem_l123_123278


namespace binary_to_decimal_110011_l123_123616

theorem binary_to_decimal_110011 :
  1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 51 :=
by
  sorry

end binary_to_decimal_110011_l123_123616


namespace cost_of_pears_l123_123407

theorem cost_of_pears (P : ℕ)
  (apples_cost : ℕ := 40)
  (dozens : ℕ := 14)
  (total_cost : ℕ := 1260)
  (h_p : dozens * P + dozens * apples_cost = total_cost) :
  P = 50 :=
by
  sorry

end cost_of_pears_l123_123407


namespace population_density_reduction_l123_123910

theorem population_density_reduction (scale : ℕ) (real_world_population : ℕ) : 
  scale = 1000000 → real_world_population = 1000000000 → 
  real_world_population / (scale ^ 2) < 1 := 
by 
  intros scale_value rw_population_value
  have h1 : scale ^ 2 = 1000000000000 := by sorry
  have h2 : real_world_population / 1000000000000 = 1 / 1000 := by sorry
  sorry

end population_density_reduction_l123_123910


namespace find_other_root_l123_123580

theorem find_other_root (x : ℚ) (h: 63 * x^2 - 100 * x + 45 = 0) (hx: x = 5 / 7) : x = 1 ∨ x = 5 / 7 :=
by 
  -- Insert the proof steps here if needed.
  sorry

end find_other_root_l123_123580


namespace coffee_expenses_l123_123170

-- Define amounts consumed and unit costs for French and Columbian roast
def ounces_per_donut_M := 2
def ounces_per_donut_D := 3
def ounces_per_donut_S := ounces_per_donut_D
def ounces_per_pot_F := 12
def ounces_per_pot_C := 15
def cost_per_pot_F := 3
def cost_per_pot_C := 4

-- Define number of donuts consumed
def donuts_M := 8
def donuts_D := 12
def donuts_S := 16

-- Calculate total ounces needed
def total_ounces_F := donuts_M * ounces_per_donut_M
def total_ounces_C := (donuts_D + donuts_S) * ounces_per_donut_D

-- Calculate pots needed, rounding up since partial pots are not allowed
def pots_needed_F := Nat.ceil (total_ounces_F / ounces_per_pot_F)
def pots_needed_C := Nat.ceil (total_ounces_C / ounces_per_pot_C)

-- Calculate total cost
def total_cost := (pots_needed_F * cost_per_pot_F) + (pots_needed_C * cost_per_pot_C)

-- Theorem statement to assert the proof
theorem coffee_expenses : total_cost = 30 := by
  sorry

end coffee_expenses_l123_123170


namespace find_minimal_positive_n_l123_123267

-- Define the arithmetic sequence
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the conditions
variables (a1 d : ℤ)
axiom condition_1 : arithmetic_seq a1 d 11 / arithmetic_seq a1 d 10 < -1
axiom condition_2 : ∃ n : ℕ, ∀ k : ℕ, k ≤ n → sum_arithmetic_seq a1 d k ≤ sum_arithmetic_seq a1 d n

-- Prove the statement
theorem find_minimal_positive_n : ∃ n : ℕ, n = 19 ∧ sum_arithmetic_seq a1 d n = 0 ∧
  (∀ m : ℕ, 0 < sum_arithmetic_seq a1 d m ∧ sum_arithmetic_seq a1 d m < sum_arithmetic_seq a1 d n) :=
sorry

end find_minimal_positive_n_l123_123267


namespace sin_double_angle_l123_123628

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l123_123628


namespace b_completes_work_in_48_days_l123_123257

noncomputable def work_rate (days : ℕ) : ℚ := 1 / days

theorem b_completes_work_in_48_days (a b c : ℕ) 
  (h1 : work_rate (a + b) = work_rate 16)
  (h2 : work_rate a = work_rate 24)
  (h3 : work_rate c = work_rate 48) :
  work_rate b = work_rate 48 :=
by
  sorry

end b_completes_work_in_48_days_l123_123257


namespace ex_ineq_l123_123232

theorem ex_ineq (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end ex_ineq_l123_123232


namespace area_enclosed_l123_123290

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)
noncomputable def area_between (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem area_enclosed (h₀ : 0 ≤ 2 * Real.pi) (h₁ : 2 * Real.pi ≤ 2 * Real.pi) :
  area_between (2 * Real.pi / 3) (5 * Real.pi / 3) f g = 2 :=
by 
  sorry

end area_enclosed_l123_123290


namespace problem1_problem2_l123_123386

-- Definitions and conditions:
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x m : ℝ) : Prop := (x^2 - 2 * x + 1 - m^2 ≤ 0) ∧ (m > 0)

-- Question (1) statement: Prove that if p is a sufficient condition for q, then m ≥ 4
theorem problem1 (p_implies_q : ∀ x : ℝ, p x → q x m) : m ≥ 4 := sorry

-- Question (2) statement: Prove that if m = 5 and p ∨ q is true but p ∧ q is false,
-- then the range of x is [-4, -1) ∪ (5, 6]
theorem problem2 (m_eq : m = 5) (p_or_q : ∃ x : ℝ, p x ∨ q x m) (p_and_not_q : ¬ (∃ x : ℝ, p x ∧ q x m)) :
  ∃ x : ℝ, (x < -1 ∧ -4 ≤ x) ∨ (5 < x ∧ x ≤ 6) := sorry

end problem1_problem2_l123_123386


namespace probability_of_difference_three_l123_123679

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l123_123679


namespace expected_worth_flip_l123_123912

/-- A biased coin lands on heads with probability 2/3 and on tails with probability 1/3.
Each heads flip gains $5, and each tails flip loses $9.
If three consecutive flips all result in tails, then an additional loss of $10 is applied.
Prove that the expected worth of a single coin flip is -1/27. -/
theorem expected_worth_flip :
  let P_heads := 2 / 3
  let P_tails := 1 / 3
  (P_heads * 5 + P_tails * -9) - (P_tails ^ 3 * 10) = -1 / 27 :=
by
  sorry

end expected_worth_flip_l123_123912


namespace train_people_count_l123_123575

theorem train_people_count :
  let initial := 332
  let first_station_on := 119
  let first_station_off := 113
  let second_station_off := 95
  let second_station_on := 86
  initial + first_station_on - first_station_off - second_station_off + second_station_on = 329 := 
by
  sorry

end train_people_count_l123_123575


namespace find_value_of_x_l123_123913
-- Import the broader Mathlib to bring in the entirety of the necessary library

-- Definitions for the conditions
variables {x y z : ℝ}

-- Assume the given conditions
axiom h1 : x = y
axiom h2 : y = 2 * z
axiom h3 : x * y * z = 256

-- Statement to prove
theorem find_value_of_x : x = 8 :=
by {
  -- Proof goes here
  sorry
}

end find_value_of_x_l123_123913


namespace parallelogram_ratio_l123_123603

-- Definitions based on given conditions
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_ratio (A : ℝ) (B : ℝ) (h : ℝ) (H1 : A = 242) (H2 : B = 11) (H3 : A = parallelogram_area B h) :
  h / B = 2 :=
by
  -- Proof goes here
  sorry

end parallelogram_ratio_l123_123603


namespace train_pass_time_approx_l123_123906

noncomputable def time_to_pass_platform
  (L_t L_p : ℝ)
  (V_t : ℝ) : ℝ :=
  (L_t + L_p) / (V_t * (1000 / 3600))

theorem train_pass_time_approx
  (L_t L_p V_t : ℝ)
  (hL_t : L_t = 720)
  (hL_p : L_p = 360)
  (hV_t : V_t = 75) :
  abs (time_to_pass_platform L_t L_p V_t - 51.85) < 0.01 := 
by
  rw [hL_t, hL_p, hV_t]
  sorry

end train_pass_time_approx_l123_123906


namespace three_digit_number_equality_l123_123114

theorem three_digit_number_equality :
  ∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧
  (100 * x + 10 * y + z = x^2 + y + z^3) ∧
  (100 * x + 10 * y + z = 357) :=
by
  sorry

end three_digit_number_equality_l123_123114


namespace find_n_l123_123813

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_lcm1 : Nat.lcm 40 n = 120) (h_lcm2 : Nat.lcm n 45 = 180) : n = 12 :=
sorry

end find_n_l123_123813


namespace inequality_ab_bc_ca_max_l123_123590

theorem inequality_ab_bc_ca_max (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|))
  ≤ 1 + (1 / 3) * (a + b + c)^2 := sorry

end inequality_ab_bc_ca_max_l123_123590


namespace initial_boys_count_l123_123591

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

end initial_boys_count_l123_123591


namespace sin_365_1_eq_m_l123_123102

noncomputable def sin_value (θ : ℝ) : ℝ := Real.sin (Real.pi * θ / 180)
variables (m : ℝ) (h : sin_value 5.1 = m)

theorem sin_365_1_eq_m : sin_value 365.1 = m :=
by sorry

end sin_365_1_eq_m_l123_123102


namespace count_integer_solutions_l123_123702

theorem count_integer_solutions :
  (2 * 9^2 + 5 * 9 * -4 + 3 * (-4)^2 = 30) →
  ∃ S : Finset (ℤ × ℤ), (∀ x y : ℤ, ((2 * x ^ 2 + 5 * x * y + 3 * y ^ 2 = 30) ↔ (x, y) ∈ S)) ∧ 
  S.card = 16 :=
by sorry

end count_integer_solutions_l123_123702


namespace upstream_speed_is_8_l123_123004

-- Definitions of given conditions
def downstream_speed : ℝ := 13
def stream_speed : ℝ := 2.5
def man's_upstream_speed : ℝ := downstream_speed - 2 * stream_speed

-- Theorem to prove
theorem upstream_speed_is_8 : man's_upstream_speed = 8 :=
by
  rw [man's_upstream_speed, downstream_speed, stream_speed]
  sorry

end upstream_speed_is_8_l123_123004


namespace john_took_away_oranges_l123_123274

-- Define the initial number of oranges Melissa had.
def initial_oranges : ℕ := 70

-- Define the number of oranges Melissa has left.
def oranges_left : ℕ := 51

-- Define the expected number of oranges John took away.
def oranges_taken : ℕ := 19

-- The theorem that needs to be proven.
theorem john_took_away_oranges :
  initial_oranges - oranges_left = oranges_taken :=
by
  sorry

end john_took_away_oranges_l123_123274


namespace equal_real_roots_of_quadratic_l123_123098

theorem equal_real_roots_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
               (∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x)) → 
  m = 6 ∨ m = -6 :=
by
  sorry  -- proof to be filled in.

end equal_real_roots_of_quadratic_l123_123098


namespace find_angle_and_sum_of_sides_l123_123612

noncomputable def triangle_conditions 
    (a b c : ℝ) (C : ℝ)
    (area : ℝ) : Prop :=
  a^2 + b^2 - c^2 = a * b ∧
  c = Real.sqrt 7 ∧
  area = (3 * Real.sqrt 3) / 2 

theorem find_angle_and_sum_of_sides
    (a b c C : ℝ)
    (area : ℝ)
    (h : triangle_conditions a b c C area) :
    C = Real.pi / 3 ∧ a + b = 5 := by
  sorry

end find_angle_and_sum_of_sides_l123_123612


namespace max_vector_sum_l123_123350

theorem max_vector_sum
  (A B C : ℝ × ℝ)
  (P : ℝ × ℝ := (2, 0))
  (hA : A.1^2 + A.2^2 = 1)
  (hB : B.1^2 + B.2^2 = 1)
  (hC : C.1^2 + C.2^2 = 1)
  (h_perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  |(2,0) - A + (2,0) - B + (2,0) - C| = 7 := sorry

end max_vector_sum_l123_123350


namespace min_sum_dimensions_l123_123041

theorem min_sum_dimensions (a b c : ℕ) (h : a * b * c = 2310) : a + b + c ≥ 52 :=
sorry

end min_sum_dimensions_l123_123041


namespace sufficient_condition_for_line_perpendicular_to_plane_l123_123742

variables {Plane Line : Type}
variables (α β γ : Plane) (m n l : Line)

-- Definitions of perpendicularity and inclusion
def perp (l : Line) (p : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def parallel (p₁ p₂ : Plane) : Prop := sorry -- definition of parallel planes
def incl (l : Line) (p : Plane) : Prop := sorry -- definition of a line being in a plane

-- The given conditions
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- The proof goal
theorem sufficient_condition_for_line_perpendicular_to_plane :
  perp m β :=
by
    sorry

end sufficient_condition_for_line_perpendicular_to_plane_l123_123742


namespace pizza_party_l123_123716

theorem pizza_party (boys girls : ℕ) :
  (7 * boys + 3 * girls ≤ 59) ∧ (6 * boys + 2 * girls ≥ 49) ∧ (boys + girls ≤ 10) → 
  boys = 8 ∧ girls = 1 := 
by sorry

end pizza_party_l123_123716


namespace bear_hunting_l123_123412

theorem bear_hunting
    (mother_meat_req : ℕ) (cub_meat_req : ℕ) (num_cubs : ℕ) (num_animals_daily : ℕ)
    (weekly_meat_req : mother_meat_req = 210)
    (weekly_meat_per_cub : cub_meat_req = 35)
    (number_of_cubs : num_cubs = 4)
    (animals_hunted_daily : num_animals_daily = 10)
    (total_weekly_meat : mother_meat_req + num_cubs * cub_meat_req = 350) :
    ∃ w : ℕ, (w * num_animals_daily * 7 = 350) ∧ w = 5 :=
by
  sorry

end bear_hunting_l123_123412


namespace apple_price_36_kgs_l123_123289

theorem apple_price_36_kgs (l q : ℕ) 
  (H1 : ∀ n, n ≤ 30 → ∀ n', n' ≤ 30 → l * n' = 100)
  (H2 : 30 * l + 3 * q = 168) : 
  30 * l + 6 * q = 186 :=
by {
  sorry
}

end apple_price_36_kgs_l123_123289


namespace determine_x_l123_123721

variable {m x : ℝ}

theorem determine_x (h₁ : m > 25)
    (h₂ : ((m / 100) * m = (m - 20) / 100 * (m + x))) : 
    x = 20 * m / (m - 20) := 
sorry

end determine_x_l123_123721


namespace find_second_number_l123_123019

theorem find_second_number (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4)
  (h3 : y / z = 4 / 7) :
  y = 240 / 7 :=
by sorry

end find_second_number_l123_123019


namespace age_ratio_in_two_years_l123_123952

-- Definitions based on conditions
def lennon_age_current : ℕ := 8
def ophelia_age_current : ℕ := 38
def lennon_age_in_two_years := lennon_age_current + 2
def ophelia_age_in_two_years := ophelia_age_current + 2

-- Statement to prove
theorem age_ratio_in_two_years : 
  (ophelia_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 4 ∧
  (lennon_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 1 := 
by 
  sorry

end age_ratio_in_two_years_l123_123952


namespace certain_event_is_A_l123_123196

def isCertainEvent (event : Prop) : Prop := event

axiom event_A : Prop
axiom event_B : Prop
axiom event_C : Prop
axiom event_D : Prop

axiom event_A_is_certain : isCertainEvent event_A
axiom event_B_is_not_certain : ¬ isCertainEvent event_B
axiom event_C_is_impossible : ¬ event_C
axiom event_D_is_not_certain : ¬ isCertainEvent event_D

theorem certain_event_is_A : isCertainEvent event_A := by
  exact event_A_is_certain

end certain_event_is_A_l123_123196


namespace summer_has_150_degrees_l123_123546

-- Define the condition that Summer has five more degrees than Jolly,
-- and the combined number of degrees they both have is 295.
theorem summer_has_150_degrees (S J : ℕ) (h1 : S = J + 5) (h2 : S + J = 295) : S = 150 :=
by sorry

end summer_has_150_degrees_l123_123546


namespace max_fourth_term_l123_123288

open Nat

/-- Constants representing the properties of the arithmetic sequence -/
axiom a : ℕ
axiom d : ℕ
axiom pos1 : a > 0
axiom pos2 : a + d > 0
axiom pos3 : a + 2 * d > 0
axiom pos4 : a + 3 * d > 0
axiom pos5 : a + 4 * d > 0
axiom sum_condition : 5 * a + 10 * d = 75

/-- Theorem stating the maximum fourth term of the arithmetic sequence -/
theorem max_fourth_term : a + 3 * d = 22 := sorry

end max_fourth_term_l123_123288


namespace triangle_is_isosceles_l123_123788

theorem triangle_is_isosceles (A B C a b c : ℝ) (h_sin : Real.sin (A + B) = 2 * Real.sin A * Real.cos B)
  (h_sine_rule : 2 * a * Real.cos B = c)
  (h_cosine_rule : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)) : a = b :=
by
  sorry

end triangle_is_isosceles_l123_123788


namespace june_initial_stickers_l123_123645

theorem june_initial_stickers (J b g t : ℕ) (h_b : b = 63) (h_g : g = 25) (h_t : t = 189) : 
  (J + g) + (b + g) = t → J = 76 :=
by
  sorry

end june_initial_stickers_l123_123645


namespace total_daisies_l123_123678

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l123_123678


namespace nine_digit_number_l123_123684

-- Conditions as definitions
def highest_digit (n : ℕ) : Prop :=
  (n / 100000000) = 6

def million_place (n : ℕ) : Prop :=
  (n / 1000000) % 10 = 1

def hundred_place (n : ℕ) : Prop :=
  n % 1000 / 100 = 1

def rest_digits_zero (n : ℕ) : Prop :=
  (n % 1000000 / 1000) % 10 = 0 ∧ 
  (n % 1000000 / 10000) % 10 = 0 ∧ 
  (n % 1000000 / 100000) % 10 = 0 ∧ 
  (n % 100000000 / 10000000) % 10 = 0 ∧ 
  (n % 100000000 / 100000000) % 10 = 0 ∧ 
  (n % 1000000000 / 100000000) % 10 = 6

-- The nine-digit number
def given_number : ℕ := 6001000100

-- Prove number == 60,010,001,00 and approximate to 6 billion
theorem nine_digit_number :
  ∃ n : ℕ, highest_digit n ∧ million_place n ∧ hundred_place n ∧ rest_digits_zero n ∧ n = 6001000100 ∧ (n / 1000000000) = 6 :=
sorry

end nine_digit_number_l123_123684


namespace perpendicular_lines_m_value_l123_123626

theorem perpendicular_lines_m_value (m : ℝ) (l1_perp_l2 : (m ≠ 0) → (m * (-1 / m^2)) = -1) : m = 0 ∨ m = 1 :=
sorry

end perpendicular_lines_m_value_l123_123626


namespace fred_seashells_l123_123567

-- Define the initial number of seashells Fred found.
def initial_seashells : ℕ := 47

-- Define the number of seashells Fred gave to Jessica.
def seashells_given : ℕ := 25

-- Prove that Fred now has 22 seashells.
theorem fred_seashells : initial_seashells - seashells_given = 22 :=
by
  sorry

end fred_seashells_l123_123567


namespace monochromatic_triangle_probability_l123_123029

noncomputable def probability_monochromatic_triangle : ℚ := sorry

theorem monochromatic_triangle_probability :
  -- Condition: Each of the 6 sides and the 9 diagonals of a regular hexagon are randomly and independently colored red, blue, or green with equal probability.
  -- Proof: The probability that at least one triangle whose vertices are among the vertices of the hexagon has all its sides of the same color is equal to 872/1000.
  probability_monochromatic_triangle = 872 / 1000 :=
sorry

end monochromatic_triangle_probability_l123_123029


namespace green_ball_count_l123_123958

theorem green_ball_count 
  (total_balls : ℕ)
  (n_red n_blue n_green : ℕ)
  (h_total : n_red + n_blue + n_green = 50)
  (h_red : ∀ (A : Finset ℕ), A.card = 34 -> ∃ a ∈ A, a < n_red)
  (h_blue : ∀ (A : Finset ℕ), A.card = 35 -> ∃ a ∈ A, a < n_blue)
  (h_green : ∀ (A : Finset ℕ), A.card = 36 -> ∃ a ∈ A, a < n_green)
  : n_green = 15 ∨ n_green = 16 ∨ n_green = 17 :=
by
  sorry

end green_ball_count_l123_123958


namespace seedling_prices_l123_123675

theorem seedling_prices (x y : ℝ) (a b : ℝ) 
  (h1 : 3 * x + 2 * y = 12)
  (h2 : x + 3 * y = 11) 
  (h3 : a + b = 200) 
  (h4 : 2 * 100 * a + 3 * 100 * b ≥ 50000) :
  x = 2 ∧ y = 3 ∧ b ≥ 100 := 
sorry

end seedling_prices_l123_123675


namespace eggs_left_over_l123_123015

theorem eggs_left_over (David_eggs Ella_eggs Fiona_eggs : ℕ)
  (hD : David_eggs = 45)
  (hE : Ella_eggs = 58)
  (hF : Fiona_eggs = 29) :
  (David_eggs + Ella_eggs + Fiona_eggs) % 10 = 2 :=
by
  sorry

end eggs_left_over_l123_123015


namespace find_interval_solution_l123_123839

def interval_solution : Set ℝ := {x | 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) <= 7}

theorem find_interval_solution (x : ℝ) :
  x ∈ interval_solution ↔
  x ∈ Set.Ioc (49 / 20 : ℝ) (14 / 5 : ℝ) := 
sorry

end find_interval_solution_l123_123839


namespace studios_total_l123_123155

section

variable (s1 s2 s3 : ℕ)

theorem studios_total (h1 : s1 = 110) (h2 : s2 = 135) (h3 : s3 = 131) : s1 + s2 + s3 = 376 :=
by
  sorry

end

end studios_total_l123_123155


namespace bumper_cars_number_of_tickets_l123_123731

theorem bumper_cars_number_of_tickets (Ferris_Wheel Roller_Coaster Jeanne_Has Jeanne_Buys : ℕ)
  (h1 : Ferris_Wheel = 5)
  (h2 : Roller_Coaster = 4)
  (h3 : Jeanne_Has = 5)
  (h4 : Jeanne_Buys = 8) :
  Ferris_Wheel + Roller_Coaster + (13 - (Ferris_Wheel + Roller_Coaster)) = 13 - (Ferris_Wheel + Roller_Coaster) :=
by
  sorry

end bumper_cars_number_of_tickets_l123_123731


namespace solution_set_l123_123251

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (a b : ℝ) : f (a + b) = f a + f b - 1
axiom monotonic (x y : ℝ) : x ≤ y → f x ≤ f y
axiom initial_condition : f 4 = 5

theorem solution_set : {m : ℝ | f (3 * m^2 - m - 2) < 3} = {m : ℝ | -4/3 < m ∧ m < 1} :=
by
  sorry

end solution_set_l123_123251


namespace at_least_one_misses_l123_123147

-- Definitions for the given conditions
variables {p q : Prop}

-- Lean 4 statement proving the equivalence
theorem at_least_one_misses (hp : p → false) (hq : q → false) : (¬p ∨ ¬q) :=
by sorry

end at_least_one_misses_l123_123147


namespace ninety_seven_squared_l123_123107

theorem ninety_seven_squared :
  97 * 97 = 9409 :=
by sorry

end ninety_seven_squared_l123_123107


namespace bella_bakes_most_cookies_per_batch_l123_123872

theorem bella_bakes_most_cookies_per_batch (V : ℝ) :
  let alex_cookies := V / 9
  let bella_cookies := V / 7
  let carlo_cookies := V / 8
  let dana_cookies := V / 10
  alex_cookies < bella_cookies ∧ carlo_cookies < bella_cookies ∧ dana_cookies < bella_cookies :=
sorry

end bella_bakes_most_cookies_per_batch_l123_123872


namespace ryan_bus_meet_exactly_once_l123_123981

-- Define respective speeds of Ryan and the bus
def ryan_speed : ℕ := 6 
def bus_speed : ℕ := 15 

-- Define bench placement and stop times
def bench_distance : ℕ := 300 
def regular_stop_time : ℕ := 45 
def extra_stop_time : ℕ := 90 

-- Initial positions
def ryan_initial_position : ℕ := 0
def bus_initial_position : ℕ := 300

-- Distance function D(t)
noncomputable def distance_at_time (t : ℕ) : ℤ :=
  let bus_travel_time : ℕ := 15  -- time for bus to travel 225 feet
  let bus_stop_time : ℕ := 45  -- time for bus to stop during regular stops
  let extended_stop_time : ℕ := 90  -- time for bus to stop during 3rd bench stops
  sorry -- calculation of distance function

-- Problem to prove: Ryan and the bus meet exactly once
theorem ryan_bus_meet_exactly_once : ∃ t₁ t₂ : ℕ, t₁ ≠ t₂ ∧ distance_at_time t₁ = 0 ∧ distance_at_time t₂ ≠ 0 := 
  sorry

end ryan_bus_meet_exactly_once_l123_123981


namespace negation_of_exist_prop_l123_123867

theorem negation_of_exist_prop :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by {
  sorry
}

end negation_of_exist_prop_l123_123867


namespace inequality_solution_l123_123710

theorem inequality_solution (a x : ℝ) : 
  (a = 0 → ¬(x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a > 0 → (-a < x ∧ x < 3*a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a < 0 → (3*a < x ∧ x < -a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) :=
by
  sorry

end inequality_solution_l123_123710


namespace circus_accommodation_l123_123084

theorem circus_accommodation : 246 * 4 = 984 := by
  sorry

end circus_accommodation_l123_123084


namespace mixed_number_expression_l123_123752

theorem mixed_number_expression :
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + (2 + 1/8)) = 9 + 25/96 :=
by
  -- here we would provide the proof steps
  sorry

end mixed_number_expression_l123_123752


namespace marla_adds_blue_paint_l123_123887

variable (M B : ℝ)

theorem marla_adds_blue_paint :
  (20 = 0.10 * M) ∧ (B = 0.70 * M) → B = 140 := 
by 
  sorry

end marla_adds_blue_paint_l123_123887


namespace probability_of_at_least_2_girls_equals_specified_value_l123_123146

def num_combinations (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def probability_at_least_2_girls : ℚ :=
  let total_committees := num_combinations 24 5
  let all_boys := num_combinations 14 5
  let one_girl_four_boys := num_combinations 10 1 * num_combinations 14 4
  let at_least_2_girls := total_committees - (all_boys + one_girl_four_boys)
  at_least_2_girls / total_committees

theorem probability_of_at_least_2_girls_equals_specified_value :
  probability_at_least_2_girls = 2541 / 3542 := 
sorry

end probability_of_at_least_2_girls_equals_specified_value_l123_123146


namespace susan_spent_75_percent_l123_123699

variables (B b s : ℝ)

-- Conditions
def condition1 : Prop := b = 0.25 * (B - 3 * s)
def condition2 : Prop := s = 0.10 * (B - 2 * b)

-- Theorem
theorem susan_spent_75_percent (h1 : condition1 B b s) (h2 : condition2 B b s) : b + s = 0.75 * B := 
sorry

end susan_spent_75_percent_l123_123699


namespace passed_percentage_l123_123161

theorem passed_percentage (A B C AB BC AC ABC: ℝ) 
  (hA : A = 0.25) 
  (hB : B = 0.50) 
  (hC : C = 0.30) 
  (hAB : AB = 0.25) 
  (hBC : BC = 0.15) 
  (hAC : AC = 0.10) 
  (hABC : ABC = 0.05) 
  : 100 - (A + B + C - AB - BC - AC + ABC) = 40 := 
by 
  rw [hA, hB, hC, hAB, hBC, hAC, hABC]
  norm_num
  sorry

end passed_percentage_l123_123161


namespace range_of_k_l123_123357

theorem range_of_k (k : ℝ) (hₖ : 0 < k) :
  (∃ x : ℝ, 1 = x^2 + (k^2 / x^2)) → 0 < k ∧ k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l123_123357


namespace divide_45_to_get_900_l123_123745

theorem divide_45_to_get_900 (x : ℝ) (h : 45 / x = 900) : x = 0.05 :=
by
  sorry

end divide_45_to_get_900_l123_123745


namespace vartan_recreation_percent_l123_123212

noncomputable def percent_recreation_week (last_week_wages current_week_wages current_week_recreation last_week_recreation : ℝ) : ℝ :=
  (current_week_recreation / current_week_wages) * 100

theorem vartan_recreation_percent 
  (W : ℝ) 
  (h1 : last_week_wages = W)  
  (h2 : last_week_recreation = 0.15 * W)
  (h3 : current_week_wages = 0.90 * W)
  (h4 : current_week_recreation = 1.80 * last_week_recreation) :
  percent_recreation_week last_week_wages current_week_wages current_week_recreation last_week_recreation = 30 :=
by
  sorry

end vartan_recreation_percent_l123_123212


namespace number_of_two_digit_primes_with_ones_digit_three_l123_123494

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l123_123494


namespace min_ratio_l123_123363

theorem min_ratio (x y : ℕ) 
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (mean : (x + y) = 110) :
  x / y = 1 / 9 :=
  sorry

end min_ratio_l123_123363


namespace even_heads_probability_is_17_over_25_l123_123330

-- Definition of the probabilities of heads and tails
def prob_tails : ℚ := 1 / 5
def prob_heads : ℚ := 4 * prob_tails

-- Definition of the probability of getting an even number of heads in two flips
def even_heads_prob (p_heads p_tails : ℚ) : ℚ :=
  p_tails * p_tails + p_heads * p_heads

-- Theorem statement
theorem even_heads_probability_is_17_over_25 :
  even_heads_prob prob_heads prob_tails = 17 / 25 := by
  sorry

end even_heads_probability_is_17_over_25_l123_123330


namespace population_after_4_years_l123_123429

theorem population_after_4_years 
  (initial_population : ℕ) 
  (new_people : ℕ) 
  (people_moved_out : ℕ) 
  (years : ℕ) 
  (final_population : ℕ) :
  initial_population = 780 →
  new_people = 100 →
  people_moved_out = 400 →
  years = 4 →
  final_population = initial_population + new_people - people_moved_out →
  final_population / 2 / 2 / 2 / 2 = 30 :=
by
  sorry

end population_after_4_years_l123_123429


namespace solve_fun_problem_l123_123093

variable (f : ℝ → ℝ)

-- Definitions of the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_monotonic_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The main theorem
theorem solve_fun_problem (h_even : is_even f) (h_monotonic : is_monotonic_on_pos f) :
  {x : ℝ | f (x + 1) = f (2 * x)} = {1, -1 / 3} := 
sorry

end solve_fun_problem_l123_123093


namespace triangle_area_is_correct_l123_123924

noncomputable def triangle_area_inscribed_circle (r : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ := 
  (1 / 2) * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem triangle_area_is_correct :
  triangle_area_inscribed_circle (18 / Real.pi) (Real.pi / 3) (2 * Real.pi / 3) Real.pi =
  162 * Real.sqrt 3 / (Real.pi^2) :=
by sorry

end triangle_area_is_correct_l123_123924


namespace superhero_speed_conversion_l123_123426

theorem superhero_speed_conversion
    (speed_km_per_min : ℕ)
    (conversion_factor : ℝ)
    (minutes_in_hour : ℕ)
    (H1 : speed_km_per_min = 1000)
    (H2 : conversion_factor = 0.6)
    (H3 : minutes_in_hour = 60) :
    (speed_km_per_min * conversion_factor * minutes_in_hour = 36000) :=
by
    sorry

end superhero_speed_conversion_l123_123426


namespace geometric_sequence_n_l123_123361

theorem geometric_sequence_n (a1 an q : ℚ) (n : ℕ) (h1 : a1 = 9 / 8) (h2 : an = 1 / 3) (h3 : q = 2 / 3) : n = 4 :=
by
  sorry

end geometric_sequence_n_l123_123361


namespace continuous_at_1_l123_123272

theorem continuous_at_1 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → |(-4 * x^2 - 6) - (-10)| < ε :=
by
  sorry

end continuous_at_1_l123_123272


namespace area_of_trapezoid_l123_123672

-- Define the parameters as given in the problem
def PQ : ℝ := 40
def RS : ℝ := 25
def h : ℝ := 10
def PR : ℝ := 20

-- Assert the quadrilateral is a trapezoid with bases PQ and RS parallel
def isTrapezoid (PQ RS : ℝ) (h : ℝ) (PR : ℝ) : Prop := true -- this is just a placeholder to state that it's a trapezoid

-- The main statement for the area of the trapezoid
theorem area_of_trapezoid (h : ℝ) (PQ RS : ℝ) (h : ℝ) (PR : ℝ) (is_trapezoid : isTrapezoid PQ RS h PR) : (1/2) * (PQ + RS) * h = 325 :=
by
  sorry

end area_of_trapezoid_l123_123672


namespace solve_divisor_problem_l123_123394

def divisor_problem : Prop :=
  ∃ D : ℕ, 12401 = (D * 76) + 13 ∧ D = 163

theorem solve_divisor_problem : divisor_problem :=
sorry

end solve_divisor_problem_l123_123394


namespace fraction_evaluation_l123_123982

theorem fraction_evaluation :
  (7 / 18 * (9 / 2) + 1 / 6) / ((40 / 3) - (15 / 4) / (5 / 16)) * (23 / 8) =
  4 + 17 / 128 :=
by
  -- conditions based on mixed number simplification
  have h1 : 4 + 1 / 2 = (9 : ℚ) / 2 := by sorry
  have h2 : 13 + 1 / 3 = (40 : ℚ) / 3 := by sorry
  have h3 : 3 + 3 / 4 = (15 : ℚ) / 4 := by sorry
  have h4 : 2 + 7 / 8 = (23 : ℚ) / 8 := by sorry
  -- the main proof
  sorry

end fraction_evaluation_l123_123982


namespace prob1_prob2_prob3_l123_123610

-- Problem 1
theorem prob1 (a b c : ℝ) : ((-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2)) = -6 * a^6 * b^2 :=
by
  sorry

-- Problem 2
theorem prob2 (a : ℝ) : (2 * a + 1)^2 - (2 * a + 1) * (2 * a - 1) = 4 * a + 2 :=
by
  sorry

-- Problem 3
theorem prob3 (x y : ℝ) : (x - y - 2) * (x - y + 2) - (x + 2 * y) * (x - 3 * y) = 7 * y^2 - x * y - 4 :=
by
  sorry

end prob1_prob2_prob3_l123_123610


namespace driver_a_driven_more_distance_l123_123338

-- Definitions based on conditions
def initial_distance : ℕ := 787
def speed_a : ℕ := 90
def speed_b : ℕ := 80
def start_difference : ℕ := 1

-- Statement of the problem
theorem driver_a_driven_more_distance :
  let distance_a := speed_a * (start_difference + (initial_distance - speed_a) / (speed_a + speed_b))
  let distance_b := speed_b * ((initial_distance - speed_a) / (speed_a + speed_b))
  distance_a - distance_b = 131 := by
sorry

end driver_a_driven_more_distance_l123_123338


namespace sine_difference_l123_123520

noncomputable def perpendicular_vectors (θ : ℝ) : Prop :=
  let a := (Real.cos θ, -Real.sqrt 3)
  let b := (1, 1 + Real.sin θ)
  a.1 * b.1 + a.2 * b.2 = 0

theorem sine_difference (θ : ℝ) (h : perpendicular_vectors θ) : Real.sin (Real.pi / 6 - θ) = Real.sqrt 3 / 2 :=
by
  sorry

end sine_difference_l123_123520


namespace derivative_at_2_l123_123139

def f (x : ℝ) : ℝ := x^3 + 2

theorem derivative_at_2 : deriv f 2 = 12 := by
  sorry

end derivative_at_2_l123_123139


namespace range_of_a_l123_123431

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → (a ≤ -1) :=
by 
  sorry

end range_of_a_l123_123431


namespace equal_share_each_shopper_l123_123808

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l123_123808


namespace variance_of_numbers_l123_123082

noncomputable def variance (s : List ℕ) : ℚ :=
  let mean := (s.sum : ℚ) / s.length
  let sqDiffs := s.map (λ n => (n - mean) ^ 2)
  sqDiffs.sum / s.length

def avg_is_34 (s : List ℕ) : Prop := (s.sum : ℚ) / s.length = 34

theorem variance_of_numbers (x : ℕ) 
  (h : avg_is_34 [31, 38, 34, 35, x]) : variance [31, 38, 34, 35, x] = 6 := 
by
  sorry

end variance_of_numbers_l123_123082


namespace temperature_difference_l123_123113

theorem temperature_difference 
    (freezer_temp : ℤ) (room_temp : ℤ) (temperature_difference : ℤ) 
    (h1 : freezer_temp = -4) 
    (h2 : room_temp = 18) : 
    temperature_difference = room_temp - freezer_temp := 
by 
  sorry

end temperature_difference_l123_123113


namespace quadratic_complete_square_l123_123372

theorem quadratic_complete_square : ∃ k : ℤ, ∀ x : ℤ, x^2 + 8*x + 22 = (x + 4)^2 + k :=
by
  use 6
  sorry

end quadratic_complete_square_l123_123372


namespace geom_seq_not_necessary_sufficient_l123_123638

theorem geom_seq_not_necessary_sufficient (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ¬(∀ n, a n > a (n + 1) → false) ∨ ¬(∀ n, a (n + 1) > a n) :=
sorry

end geom_seq_not_necessary_sufficient_l123_123638


namespace tip_count_proof_l123_123305

def initial_customers : ℕ := 29
def additional_customers : ℕ := 20
def customers_who_tipped : ℕ := 15
def total_customers : ℕ := initial_customers + additional_customers
def customers_didn't_tip : ℕ := total_customers - customers_who_tipped

theorem tip_count_proof : customers_didn't_tip = 34 :=
by
  -- This is a proof outline, not the actual proof.
  sorry

end tip_count_proof_l123_123305


namespace plumber_charge_shower_l123_123348

theorem plumber_charge_shower (S : ℝ) 
  (sink_cost : ℝ := 30) 
  (toilet_cost : ℝ := 50)
  (max_earning : ℝ := 250)
  (first_job_toilets : ℝ := 3) (first_job_sinks : ℝ := 3)
  (second_job_toilets : ℝ := 2) (second_job_sinks : ℝ := 5)
  (third_job_toilets : ℝ := 1) (third_job_showers : ℝ := 2) (third_job_sinks : ℝ := 3) :
  2 * S + 1 * toilet_cost + 3 * sink_cost ≤ max_earning → S ≤ 55 :=
by
  sorry

end plumber_charge_shower_l123_123348


namespace max_k_constant_l123_123216

theorem max_k_constant : 
  (∃ k, (∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) <= k * Real.sqrt (x + y + z))) 
  ∧ k = Real.sqrt 6 / 2) :=
sorry

end max_k_constant_l123_123216


namespace multiple_of_rohan_age_l123_123459

theorem multiple_of_rohan_age (x : ℝ) (h1 : 25 - 15 = 10) (h2 : 25 + 15 = 40) (h3 : 40 = x * 10) : x = 4 := 
by 
  sorry

end multiple_of_rohan_age_l123_123459


namespace segment_length_abs_eq_cubrt_27_five_l123_123915

theorem segment_length_abs_eq_cubrt_27_five : 
  (∀ x : ℝ, |x - (3 : ℝ)| = 5) → (8 - (-2) = 10) :=
by 
  intros;
  sorry

end segment_length_abs_eq_cubrt_27_five_l123_123915


namespace lcm_20_45_75_eq_900_l123_123825

theorem lcm_20_45_75_eq_900 : Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
by sorry

end lcm_20_45_75_eq_900_l123_123825


namespace proof_problem_l123_123349

variables (a b : ℝ)

noncomputable def expr := (2 * a⁻¹ + (a⁻¹ / b)) / a

theorem proof_problem (h1 : a = 1/3) (h2 : b = 3) : expr a b = 21 :=
by
  sorry

end proof_problem_l123_123349


namespace total_birds_and_storks_l123_123351

theorem total_birds_and_storks
  (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ)
  (hb : initial_birds = 3) (hs : initial_storks = 4) (has : additional_storks = 6) :
  initial_birds + (initial_storks + additional_storks) = 13 :=
by
  sorry

end total_birds_and_storks_l123_123351


namespace decreasing_implies_inequality_l123_123499

variable (f : ℝ → ℝ)

theorem decreasing_implies_inequality (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) : f 3 < f 2 ∧ f 2 < f 1 :=
  sorry

end decreasing_implies_inequality_l123_123499


namespace flagpole_height_l123_123796

theorem flagpole_height :
  ∃ (AB AC AD DE DC : ℝ), 
    AC = 5 ∧
    AD = 3 ∧ 
    DE = 1.8 ∧
    DC = AC - AD ∧
    AB = (DE * AC) / DC ∧
    AB = 4.5 :=
by
  exists 4.5, 5, 3, 1.8, 2
  simp
  sorry

end flagpole_height_l123_123796


namespace right_triangle_area_l123_123550

theorem right_triangle_area (a b c : ℝ) (h₀ : a = 24) (h₁ : c = 30) (h2 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 :=
by
  sorry

end right_triangle_area_l123_123550


namespace g_x_equation_g_3_value_l123_123774

noncomputable def g : ℝ → ℝ := sorry

theorem g_x_equation (x : ℝ) (hx : x ≠ 1/2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x := sorry

theorem g_3_value : g 3 = 31 / 8 :=
by
  -- Use the provided functional equation and specific input values to derive g(3)
  sorry

end g_x_equation_g_3_value_l123_123774


namespace max_rectangle_area_l123_123525

noncomputable def curve_parametric_equation (θ : ℝ) :
    ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem max_rectangle_area :
  ∃ (θ : ℝ), (θ ∈ Set.Icc 0 (2 * Real.pi)) ∧
  ∀ (x y : ℝ), (x, y) = curve_parametric_equation θ →
  |(1 + 2 * Real.cos θ) * (1 + 2 * Real.sin θ)| = 3 + 2 * Real.sqrt 2 :=
sorry

end max_rectangle_area_l123_123525


namespace neg_exists_eq_forall_l123_123142

theorem neg_exists_eq_forall (p : Prop) :
  (∀ x : ℝ, ¬(x^2 + 2*x = 3)) ↔ ¬(∃ x : ℝ, x^2 + 2*x = 3) := 
by
  sorry

end neg_exists_eq_forall_l123_123142


namespace smallest_whole_number_l123_123776

theorem smallest_whole_number (a : ℕ) : 
  (a % 4 = 1) ∧ (a % 3 = 1) ∧ (a % 5 = 2) → a = 37 :=
by
  intros
  sorry

end smallest_whole_number_l123_123776


namespace number_line_4_units_away_l123_123998

theorem number_line_4_units_away (x : ℝ) : |x + 3.2| = 4 ↔ (x = 0.8 ∨ x = -7.2) :=
by
  sorry

end number_line_4_units_away_l123_123998


namespace find_cos_2beta_l123_123953

noncomputable def cos_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (htan : Real.tan α = 1 / 7) (hcos : Real.cos (α + β) = 2 * Real.sqrt 5 / 5) : Real :=
  2 * (Real.cos β)^2 - 1

theorem find_cos_2beta (α β : ℝ) (h1: 0 < α ∧ α < π / 2) (h2: 0 < β ∧ β < π / 2)
  (htan: Real.tan α = 1 / 7) (hcos: Real.cos (α + β) = 2 * Real.sqrt 5 / 5) :
  cos_2beta α β h1 h2 htan hcos = 4 / 5 := 
sorry

end find_cos_2beta_l123_123953


namespace boys_left_hand_to_girl_l123_123876

-- Definitions based on the given conditions
def num_boys : ℕ := 40
def num_girls : ℕ := 28
def boys_right_hand_to_girl : ℕ := 18

-- Statement to prove
theorem boys_left_hand_to_girl : (num_boys - (num_boys - boys_right_hand_to_girl)) = boys_right_hand_to_girl := by
  sorry

end boys_left_hand_to_girl_l123_123876


namespace caterer_preparations_l123_123189

theorem caterer_preparations :
  let b_guests := 84
  let a_guests := (2/3) * b_guests
  let total_guests := b_guests + a_guests
  let extra_plates := 10
  let total_plates := total_guests + extra_plates

  let cherry_tomatoes_per_plate := 5
  let regular_asparagus_per_plate := 8
  let vegetarian_asparagus_per_plate := 6
  let larger_asparagus_per_plate := 12
  let larger_asparagus_portion_guests := 0.1 * total_plates

  let blueberries_per_plate := 15
  let raspberries_per_plate := 8
  let blackberries_per_plate := 10

  let cherry_tomatoes_needed := cherry_tomatoes_per_plate * total_plates

  let regular_portion_guests := 0.9 * total_plates
  let regular_asparagus_needed := regular_asparagus_per_plate * regular_portion_guests
  let larger_asparagus_needed := larger_asparagus_per_plate * larger_asparagus_portion_guests
  let asparagus_needed := regular_asparagus_needed + larger_asparagus_needed

  let blueberries_needed := blueberries_per_plate * total_plates
  let raspberries_needed := raspberries_per_plate * total_plates
  let blackberries_needed := blackberries_per_plate * total_plates

  cherry_tomatoes_needed = 750 ∧
  asparagus_needed = 1260 ∧
  blueberries_needed = 2250 ∧
  raspberries_needed = 1200 ∧
  blackberries_needed = 1500 :=
by
  -- Proof goes here
  sorry

end caterer_preparations_l123_123189


namespace positive_whole_numbers_with_cube_roots_less_than_15_l123_123792

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l123_123792


namespace correct_operation_l123_123542

theorem correct_operation (a b : ℝ) : 
  (2 * a^2 + a^2 = 3 * a^2) ∧ 
  (a^3 * a^3 ≠ 2 * a^3) ∧ 
  (a^9 / a^3 ≠ a^3) ∧ 
  (¬(7 * a * b - 5 * a = 2)) :=
by 
  sorry

end correct_operation_l123_123542


namespace initial_cookie_count_l123_123016

variable (cookies_left_after_week : ℕ)
variable (cookies_taken_each_day : ℕ)
variable (total_cookies_taken_in_four_days : ℕ)
variable (initial_cookies : ℕ)
variable (days_per_week : ℕ)

theorem initial_cookie_count :
  cookies_left_after_week = 28 →
  total_cookies_taken_in_four_days = 24 →
  days_per_week = 7 →
  (∀ d (h : d ∈ Finset.range days_per_week), cookies_taken_each_day = 6) →
  initial_cookies = 52 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_cookie_count_l123_123016


namespace chocolates_initial_l123_123466

variable (x : ℕ)
variable (h1 : 3 * x + 5 + 25 = 5 * x)
variable (h2 : x = 15)

theorem chocolates_initial (x : ℕ) (h1 : 3 * x + 5 + 25 = 5 * x) (h2 : x = 15) : 3 * 15 + 5 = 50 :=
by sorry

end chocolates_initial_l123_123466


namespace gcd_372_684_is_12_l123_123341

theorem gcd_372_684_is_12 : gcd 372 684 = 12 := by
  sorry

end gcd_372_684_is_12_l123_123341


namespace y_work_days_24_l123_123356

-- Definitions of the conditions
def x_work_days := 36
def y_work_days (d : ℕ) := d
def y_worked_days := 12
def x_remaining_work_days := 18

-- Statement of the theorem
theorem y_work_days_24 : ∃ d : ℕ, (y_worked_days / y_work_days d + x_remaining_work_days / x_work_days = 1) ∧ d = 24 :=
  sorry

end y_work_days_24_l123_123356


namespace shirts_production_l123_123892

-- Definitions
def constant_rate (r : ℕ) : Prop := ∀ n : ℕ, 8 * n * r = 160 * n

theorem shirts_production (r : ℕ) (h : constant_rate r) : 16 * r = 32 :=
by sorry

end shirts_production_l123_123892


namespace age_of_15th_student_l123_123579

theorem age_of_15th_student (avg15: ℕ) (avg5: ℕ) (avg9: ℕ) (x: ℕ)
  (h1: avg15 = 15) (h2: avg5 = 14) (h3: avg9 = 16)
  (h4: 15 * avg15 = x + 5 * avg5 + 9 * avg9) : x = 11 :=
by
  -- Proof will be added here
  sorry

end age_of_15th_student_l123_123579


namespace caroline_lassis_l123_123169

theorem caroline_lassis (c : ℕ → ℕ): c 3 = 13 → c 15 = 65 :=
by
  sorry

end caroline_lassis_l123_123169


namespace problem_l123_123094

theorem problem (a : ℝ) : (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end problem_l123_123094


namespace solution_set_of_inequality_l123_123677

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - 1) / (x + 2) > 1 ↔ x < -2 ∨ x > 3 :=
by
  sorry

end solution_set_of_inequality_l123_123677


namespace sqrt_43_between_6_and_7_l123_123374

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 := sorry

end sqrt_43_between_6_and_7_l123_123374
